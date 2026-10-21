"""E27 verification (fresh sample): pairwise angularity decoupling edits on JEDI-linear, focused on the pT-weighted (kappa 1)
neighbours of M2 = sum pT r^2, with three additions to E27:
  (1) a read-out-free continuous metric: Delta (logit_Z - logit_W) on W-called jets, as a ratio to the uniform dilation on the
      same jets (R_marg), plus h4 / h5 log-m read-out ratios (R_h4, R_h5) as in E27;
  (2) the true jet-mass change of every edit (Delta log m from the particle four-vectors), as a ratio to the dilation (R_true);
  (3) the test of the alternative directly: W->Z flow of edit A against the dilation on the same jets (McNemar), next to
      edit vs its matched null; and the excess fraction F = (A - null) / (dilation - null) with a bootstrap CI;
  (4) a new member (kappa 1, beta 2.5).
Edits, solver, kept-jet rules and null edits are E27's own functions (interp.e27_angularity_jedi), with the member list
extended by (1, 2.5). Sample: interp.jedi_core.sample(per_class, seed) (default 2000, 2207; E27 used seed 1201).
Members: 5 % dose: k0_b1, k0_b2, k1_b0.5, k1_b1, k1_b1.5, k1_b2.5, k1_b3, k2_b0.5, k2_b2; 2 % dose: the five kappa-1 members.
Output: $AI_RESULTS/jedi/e27_verify[_smoke].json
  python -m interp.e27_verify [--smoke 100] [--seed 2207]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import interp.e27_angularity_jedi as E  # noqa: E402

E.MEM = list(E.MEM) + [(1, 2.5)]; E.NM = len(E.MEM); E.M1R = [E.NM, E.NM + 1]
assert E.MEM[E.I12] == (1, 2.0)
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
W_, Z_ = 2, 3; T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def mn(i): return f'k{E.MEM[i][0]}_b{E.MEM[i][1]:g}'


def idx(k, b): return E.MEM.index((k, b))


def main(a):
    from interp.jedi_core import sample
    from interp.x10_das_jedi import clean, holm, mass, RidgeR
    from interp.h12_boost_panel import mcnemar
    sm = a.smoke; pc = sm or a.per_class; seed = a.seed; NB = 200 if sm else 1000; NP = 1000 if sm else 10000
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'e27_verify{"_smoke" if sm else ""}.json'
    R = dict(sample=f'interp.jedi_core.sample(per_class={pc}, seed={seed})', members=[mn(i) for i in range(E.NM)])
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    x, y, _ = sample(pc, seed); J = len(y); R['N'] = J
    x64 = np.asarray(x, np.float64); pt, de, dp = x64[..., 0], x64[..., 1], x64[..., 2]; real = pt > 0
    ref0 = E.reference(pt, de, dp, real); logm0 = np.log(np.maximum(mass(x), 1e-3))
    nets = {'trained': E.NetRun(False), 'untrained': E.NetRun(True)}; base = {}
    for nm, net in nets.items():
        r = net.fwd(x)
        if net.untrained: net.off = r['logits'].mean(0); r['logits'] = r['logits'] - net.off
        base[nm] = r
    c0 = {nm: base[nm]['logits'].argmax(1) for nm in nets}
    R['acc'] = float((c0['trained'] == y).mean()); log(f'N {J}; accuracy {R["acc"]:.4f}')
    pr = np.random.default_rng(seed + 3).permutation(J); hA, hB = np.sort(pr[:J // 2]), np.sort(pr[J // 2:]); nf = 2 * len(hA) // 3
    isB = np.zeros(J, bool); isB[hB] = True; RO = {}; R['readout_r2'] = {}
    for st in ('h4', 'h5'):
        ro = RidgeR(base['trained'][st], logm0, hA[:nf], hA[nf:]); RO[st] = ro
        R['readout_r2'][st] = float(1 - ((ro(base['trained'][st][hB]) - logm0[hB]) ** 2).mean() / logm0[hB].var())
    mh0 = {st: RO[st](base['trained'][st]) for st in RO}
    marg0 = {nm: base[nm]['logits'][:, Z_] - base[nm]['logits'][:, W_] for nm in nets}
    log(f'read-out R2 {R["readout_r2"]}'); save()
    Wb = np.random.default_rng(seed + 11).poisson(1.0, (NB, J)).astype(np.float32)

    def fwd(delta):
        xe = E.apply(x, delta); o = dict(dlogm_true=np.log(np.maximum(mass(xe), 1e-3)) - logm0)
        for nm, net in nets.items():
            r = net.fwd(xe); ce = r['logits'].argmax(1)
            o[nm] = dict(wz=(c0[nm] == W_) & (ce == Z_), ch=ce != c0[nm], marg=r['logits'][:, Z_] - r['logits'][:, W_] - marg0[nm])
            if nm == 'trained':
                for st in ('h4', 'h5'): o[nm][st] = RO[st](r[st]) - mh0[st]
        return o

    def bm(w, v): return (w @ v) / np.maximum(w.sum(1), 1e-9)

    def summary(P, N, F, ks, nm):
        """edit P, its null N, the dilation F (same jets ks), network nm"""
        isW = ks & (c0[nm] == W_); kb = ks & isB; p, n_, f = P[nm], N[nm], F[nm]; w = Wb[:, isW]
        o = dict(n=int(ks.sum()), n_W=int(isW.sum()))
        if isW.sum() == 0: return o
        e, nl, rf = p['wz'][isW].astype(float), n_['wz'][isW].astype(float), f['wz'][isW].astype(float)
        be, bn, bf = bm(w, e), bm(w, nl), bm(w, rf)
        o.update(wz=e.mean(), wz_ci=E.qci(be), wz_null=nl.mean(), wz_null_ci=E.qci(bn), wz_dil=rf.mean(), wz_dil_ci=E.qci(bf),
                 F=(e.mean() - nl.mean()) / (rf.mean() - nl.mean()) if rf.mean() != nl.mean() else None, F_ci=E.qci((be - bn) / (bf - bn)),
                 mcnemar_vs_null=mcnemar(p['wz'][isW], n_['wz'][isW]), mcnemar_vs_dil=mcnemar(p['wz'][isW], f['wz'][isW]),
                 calls_changed=float(p['ch'][ks].mean()), calls_changed_null=float(n_['ch'][ks].mean()))
        m, mn_, mf = p['marg'][isW], n_['marg'][isW], f['marg'][isW]
        o.update(R_marg=m.mean() / mf.mean(), R_marg_ci=E.qci(bm(w, m) / bm(w, mf)), R_marg_null=mn_.mean() / mf.mean(),
                 R_marg_null_ci=E.qci(bm(w, mn_) / bm(w, mf)), dmarg_dil=float(mf.mean()), perm_p_marg=E.signflip_p(m - mn_, NP, seed=7))
        if nm == 'trained':
            for st in ('h4', 'h5'):
                v, vn, vr = p[st][kb], n_[st][kb], f[st][kb]; wb = Wb[:, kb]
                o[f'R_{st}'] = v.mean() / vr.mean(); o[f'R_{st}_ci'] = E.qci(bm(wb, v) / bm(wb, vr)); o[f'R_{st}_null'] = vn.mean() / vr.mean()
            t, tn, tf = P['dlogm_true'][ks], N['dlogm_true'][ks], F['dlogm_true'][ks]
            o.update(R_true=t.mean() / tf.mean(), R_true_ci=E.qci(bm(Wb[:, ks], t) / bm(Wb[:, ks], tf)), R_true_null=tn.mean() / tf.mean())
        return o

    plan = (('5', np.log(1.05), [idx(0, 1.0), idx(0, 2.0), idx(1, .5), idx(1, 1.0), idx(1, 1.5), idx(1, 2.5), idx(1, 3.0), idx(2, .5), idx(2, 2.0)]),
            ('2', np.log(1.02), [idx(1, .5), idx(1, 1.0), idx(1, 1.5), idx(1, 2.5), idx(1, 3.0)]))
    R['results'] = {}
    for dose, eps, mems in plan:
        tol = E.TOL * eps / E.EPS; out = R['results'][dose] = dict(eps=float(eps), members={})
        dil = fwd(np.full_like(pt, 0.5 * eps) * real)
        rows = [E.I12] + E.M1R; b = np.zeros((J, 3)); b[:, 0] = eps
        dl, c = E.gn(pt, de, dp, real, rows, b, np.zeros_like(pt), ref0); k1 = E.ok_mask(c, dl, rows, b, tol)
        dn, cn, k2 = E.null_edit(pt, de, dp, real, rows, E.disp(de, dp, dl), ref0, np.random.default_rng(seed * 1000 + 999 - len(dose)), tol)
        ks = k1 & k2; P, N = fwd(dl), fwd(dn)
        out['M2h'] = {nm: summary(P, N, dil, ks, nm) for nm in nets}
        log(f'dose {dose}: M2h n {ks.sum()} trained {json.dumps(clean({k: out["M2h"]["trained"].get(k) for k in ("wz", "wz_null", "wz_dil", "R_marg", "R_h4", "R_true")}))}'); save()
        for i in mems:
            rows = [i, E.I12] + E.M1R; ED = {}; KP = {}
            for kind, tgt in (('A', i), ('B', E.I12)):
                b = np.zeros((J, 4)); b[:, rows.index(tgt)] = eps
                dl, c = E.gn(pt, de, dp, real, rows, b, np.zeros_like(pt), ref0); KP[kind] = E.ok_mask(c, dl, rows, b, tol)
                dn, cn, KP['n' + kind] = E.null_edit(pt, de, dp, real, rows, E.disp(de, dp, dl), ref0,
                                                     np.random.default_rng(seed * 1000 + 2 * i + (kind == 'B') + 100 * len(dose)), tol)
                ED[kind] = fwd(dl); ED['n' + kind] = fwd(dn)
            ks = KP['A'] & KP['B'] & KP['nA'] & KP['nB']
            e = out['members'][mn(i)] = dict(kept=int(ks.sum()), kept_by_class=np.bincount(y[ks], minlength=5).tolist())
            for kind in ('A', 'B'):
                e[kind] = {nm: summary(ED[kind], ED['n' + kind], dil, ks, nm) for nm in nets}
            t = e['A']['trained']; u = e['B']['trained']
            log(f'dose {dose} {mn(i)} n {ks.sum()} nW {t.get("n_W")}: A wz {t.get("wz", 0):.4f} null {t.get("wz_null", 0):.4f} dil {t.get("wz_dil", 0):.4f} '
                f'F {t.get("F")} R_marg {t.get("R_marg", 0):+.2f} R_h4 {t.get("R_h4", 0):+.2f} R_h5 {t.get("R_h5", 0):+.2f} R_true {t.get("R_true", 0):+.3f} | '
                f'B wz {u.get("wz", 0):.4f} null {u.get("wz_null", 0):.4f} R_marg {u.get("R_marg", 0):+.2f} R_h4 {u.get("R_h4", 0):+.2f} R_true {u.get("R_true", 0):+.3f}')
            save()
        for kind in ('A', 'B'):
            for key, g in (('mcnemar_vs_null', lambda d: d['mcnemar_vs_null']['p']), ('mcnemar_vs_dil', lambda d: d['mcnemar_vs_dil']['p']),
                           ('perm_p_marg', lambda d: d['perm_p_marg'])):
                ms = [m_ for m_ in mems if 'wz' in out['members'][mn(m_)][kind]['trained']]
                adj = holm([g(out['members'][mn(m_)][kind]['trained']) for m_ in ms])
                for m_, p_ in zip(ms, adj): out['members'][mn(m_)][kind]['trained'][f'holm_{key}'] = float(p_)
        save()
    R['seconds'] = time.time() - T0; save(); log(f'done; wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--seed', type=int, default=2207)
    ap.add_argument('--per_class', type=int, default=2000)
    main(ap.parse_args())
