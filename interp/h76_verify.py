"""H76 skeptic verification (loop D). Fresh sample (seed 7621), same edits as interp/h76_angularity_qg.py (imported, unchanged).

Checks
  1. Per-sign decomposition. H76 reports the odd slope (s(t+) - s(t-))/2 minus the odd crand slope. Here each sign is reported
     separately: e(t+/-) = Delta s(t+/-) - mean(Delta s(crand+), Delta s(crand-)), so an edit that moves the network no more than the
     random null-space edit gives e ~ 0 whichever way the odd slope points. Same for N+ / N- / Nsp (exact split).
  2. Norm-matched crand (crand_big): the H76 crand is scaled to the MEAN norm of the + and - edits; LHA+ and width- are 2.5-3.4x
     larger than LHA- and width+. crand_big uses the per-particle profile and the full norm of the larger edit (LHA+, width-),
     applied with both signs and restored to all four angularities, sum pT and the axis.
  3. What the edits change beyond the five angularities: lambda(1, beta) for beta = 0.1, 0.25, 1.5, 3; the radial dispersion
     D = lambda(1,1) / lambda(1,1/2)^2 (>= 1 by Cauchy-Schwarz; = 1 when all energy sits at one radius); leading-particle radius
     r_1 / R and momentum fraction z_1.
  4. Extra references (GBDT, trained on 20k + 20k Pythia train jets): single lambda(1,0.1), lambda(1,0.25), D, r_1; all angularities
     (N, pT^D, lambda(1, beta) beta = 0.1 ... 3) "g_ext"; the same without N "g_ext_irc". Cosine of the 5-component response vectors
     with the networks'.
  5. Padding check: ParT_kin logits on unedited jets padded to max+0 vs max+25 particles.
Output: $AI_RESULTS/physics/h76_verify.json
"""
import argparse
import json
import time

import numpy as np

from interp import h76_angularity_qg as H76

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = H76.V; RES = H76.RES; R0 = H76.R0
BETAS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
NETS = ['part_kin', 'pnet_kin', 'lorentznet', 'pfn_k', 'efn_k', 'hw_part_kin', 'part_kin_perm']
MAIN = ['lha+', 'lha-', 'wid+', 'wid-', 'ptd+', 'ptd-', 'mass+', 'mass-', 'lha_r+', 'lha_r-', 'wid_r+', 'wid_r-', 'ptd_r+', 'ptd_r-',
        'mass_r+', 'mass_r-', 'N+', 'N-', 'Nsp', 'lha_big+', 'lha_big-', 'wid_big+', 'wid_big-']


def ext(a):
    pt, eta, phi = a[:, 0], a[:, 1], a[:, 2]
    px, py, pz, E = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
    yJ = 0.5 * np.log((E.sum() + pz.sum()) / (E.sum() - pz.sum())); pJ = np.arctan2(py.sum(), px.sum())
    dR = np.sqrt((eta - yJ) ** 2 + ((phi - pJ + np.pi) % (2 * np.pi) - np.pi) ** 2) / R0; z = pt / pt.sum()
    o = {f'b{b}': float((z * dR ** b).sum()) for b in BETAS}
    o['D'] = o['b1.0'] / o['b0.5'] ** 2; i = int(np.argmax(pt)); o['r1'] = float(dR[i]); o['z1'] = float(z[i])
    o['ptd'] = float((z ** 2).sum()); o['N'] = float(len(a))
    return o


def ext_table(L):
    O = [ext(a) for a in L]
    return {k: np.array([o[k] for o in O]) for k in O[0]}


FEATS = dict(s_b0p1=['b0.1'], s_b0p25=['b0.25'], s_D=['D'], s_r1=['r1'],
             g_ext=['N', 'ptd'] + [f'b{b}' for b in BETAS], g_ext_irc=['ptd'] + [f'b{b}' for b in BETAS])


def train_ext(n_per_class):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import h14_qg_generator as H14
    Lt, yt, _ = H14.efp_train_lists('py', n_per_class=n_per_class, seed=7601 + 1)
    O = ext_table([np.c_[a, np.zeros(len(a))] for a in Lt]); g = (yt == 0).astype(int)
    clf = {}
    for k, f in FEATS.items():
        X = np.c_[tuple(O[c] for c in f)]
        clf[k] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                                                random_state=0).fit(X, g)
    log('ext refs trained on', len(yt))
    return clf


def extra_controls(L0, seed, gen, dev, chunk=1000):
    """crand_big: null-space random edit with the per-particle profile and the FULL norm of lha+ / wid- (the larger edits)"""
    import torch
    Sv = H76.Solver(dev); out = {}
    for s in range(0, len(L0), chunk):
        Lc = L0[s:s + chunk]; S = Sv.pack(Lc); x0 = S['x']; c0, _ = Sv.jac(S, x0); n = S['n']
        rng = np.random.default_rng([seed, 99, s])
        for t, sg_big in (('lha', np.log(1.03)), ('wid', np.log(0.97))):
            ti = H76.IRCS.index(t); tg = c0.clone(); tg[:, ti] += sg_big; xb, rb = Sv.solve(S, x0, tg)
            d = xb - x0; per = torch.sqrt(d[:, :n] ** 2 + d[:, n:] ** 2)
            v = Sv.null_rand(S, x0, per, d.norm(dim=1), rng)
            for sgn, sg in (('+', 1.0), ('-', -1.0)):
                x, r = Sv.solve(S, x0 + sg * v, c0)
                k = f'{t}_big{sgn}'; out.setdefault(k, ([], [], []))
                out[k][0].extend(Sv.unpack(S, x, Lc)); out[k][1].append((r < 1e-6).cpu().numpy())
                out[k][2].append((x - x0).norm(dim=1).cpu().numpy())
    return {k: (v[0], np.concatenate(v[1]), dict(dnorm=np.concatenate(v[2]))) for k, v in out.items()}


def boot_ci(x, idx):
    b = np.array([x[i].mean() for i in idx]); return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def signflip(x, P, seed=0):
    r = np.random.default_rng(seed); obs = abs(x.mean()); cnt = 0
    for c in range(0, P, 500):
        sg = r.choice(np.array([-1.0, 1.0]), (min(500, P - c), len(x))); cnt += int((np.abs(sg @ x / len(x)) >= obs - 1e-15).sum())
    return (cnt + 1) / (P + 1)


def main(a):
    import torch
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    M = H.Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
    M.add_lnet()
    for kind in ('pfn', 'efn'): H14.add_keras_ef(M, kind)
    H14.add_weaver_dir(M, 'hw_part_kin', 'part', False, V / 'models' / 'qg_herwig' / 'part_kin')
    M.add_permuted('part_kin_perm', 'part_kin')
    names = [n for n in NETS if n in M.m]; log('models', names)
    T, _ = H14.lrn_tables(); refs = H76.train_refs('py', a.train_per_class); xref = train_ext(a.train_per_class)
    OUT = dict(seed=a.seed, per_class=a.per_class, gens={})
    for gen in a.gens.split(','):
        L0, y, idx_rows, _ = H76.load_sample(gen, a.per_class, a.seed); J = len(y)
        log(gen, J, 'jets; first rows', idx_rows[:5].tolist())
        E, kN = H76.make_edits(L0, a.seed, gen, dev); E.update(extra_controls(L0, a.seed, gen, dev))
        O0 = H76.obs_table(L0); X0 = ext_table(L0)
        G = OUT['gens'][gen] = dict(n=J, base={}, conds={}, models={}, obs_shift={})
        S = {}; Xo = {}
        def score(L, O, X):
            Jt = H.from_list(L, Lpad=max(len(x) for x in L)); r = {nm: M.logit(nm, Jt).astype(float) for nm in names}
            for k, v in H76.ref_scores(refs, O, T, 'py').items(): r[k + '_py'] = v.astype(float)
            for k, f in FEATS.items(): r[k] = xref[k].decision_function(np.c_[tuple(X[c] for c in f)])
            return r
        S['base'] = score(L0, O0, X0)
        for nm, v in S['base'].items(): G['base'][nm] = H.roc_metrics(v, y)
        # padding check
        Jt = H.from_list(L0, Lpad=max(len(x) for x in L0) + 25); sp = M.logit('part_kin', Jt)
        G['pad_check_part_kin_maxabs'] = float(np.abs(sp - S['base']['part_kin']).max())
        oks = {}
        for cn in MAIN:
            Le, ok, info = E[cn]; O = H76.obs_table(Le); X = ext_table(Le); S[cn] = score(Le, O, X); Xo[cn] = X
            dv, ach = H76.tolerance(O0, O, cn); ok2 = ok & (dv < H76.TOL); tg_, fr_ = H76.parse(cn)
            if tg_ in H76.IRCS: ok2 &= np.abs(np.abs(ach) - fr_) < H76.TOL * fr_ / 0.1
            elif cn in ('N+', 'N-'): ok2 &= np.abs(np.abs(ach) - kN / O0['N']) < 1e-9
            oks[cn] = ok2
            G['conds'][cn] = dict(ok=float(ok2.mean()), dnorm_median=float(np.median(info['dnorm'])) if 'dnorm' in info else None,
                                  dev_p99=float(np.percentile(dv[ok2], 99)) if ok2.any() else None)
            log(gen, cn, G['conds'][cn])
        okall = np.all([oks[c] for c in MAIN], 0); G['ok_all'] = float(okall.mean()); jj = np.flatnonzero(okall)
        rng = np.random.default_rng(a.seed); bidx = rng.integers(0, len(jj), (a.boot, len(jj)))
        # what each edit changes (median relative change over accepted jets)
        for cn in MAIN:
            G['obs_shift'][cn] = {k: float(np.median(Xo[cn][k][jj] / X0[k][jj] - 1)) for k in ('b0.1', 'b0.25', 'b1.5', 'b3.0', 'D', 'z1')}
            G['obs_shift'][cn]['r1_abs_change'] = float(np.median(Xo[cn]['r1'][jj] - X0['r1'][jj]))
        # per-model per-sign effects
        pvals = []
        for nm in S['base']:
            R = G['models'][nm] = {}; ds = {cn: (S[cn][nm] - S['base'][nm])[jj] for cn in MAIN}
            for t in ('lha', 'wid', 'ptd', 'mass'):
                cr = 0.5 * (ds[f'{t}_r+'] + ds[f'{t}_r-'])
                for sgn in '+-':
                    e = ds[t + sgn] - cr
                    R[f'{t}{sgn}'] = dict(dS=float(ds[t + sgn].mean()), dS_ci=boot_ci(ds[t + sgn], bidx), crand=float(cr.mean()),
                                          crand_ci=boot_ci(cr, bidx), excess=float(e.mean()), excess_ci=boot_ci(e, bidx))
                    if nm in NETS:
                        R[f'{t}{sgn}']['p'] = signflip(e, a.perm, seed=len(pvals)); pvals.append((nm, f'{t}{sgn}', R[f'{t}{sgn}']['p']))
                odd = ((ds[t + '+'] - ds[t + '-']) / 2 - (ds[f'{t}_r+'] - ds[f'{t}_r-']) / 2) * 0.1 / H76.FRAC
                even = (ds[t + '+'] + ds[t + '-']) / 2 - cr
                R[t] = dict(odd_per10=float(odd.mean()), odd_ci=boot_ci(odd, bidx), even=float(even.mean()), even_ci=boot_ci(even, bidx))
            for t, big in (('lha', 'lha+'), ('wid', 'wid-')):
                cb = 0.5 * (ds[f'{t}_big+'] + ds[f'{t}_big-']); e = ds[big] - cb
                R[f'{big}_vs_big'] = dict(crand_big=float(cb.mean()), crand_big_ci=boot_ci(cb, bidx), excess=float(e.mean()),
                                          excess_ci=boot_ci(e, bidx), abs_big=float(np.abs(ds[f'{t}_big+']).mean()),
                                          abs_edit=float(np.abs(ds[big]).mean()))
                if nm in NETS:
                    R[f'{big}_vs_big']['p'] = signflip(e, a.perm, seed=len(pvals)); pvals.append((nm, f'{big}_vs_big', R[f'{big}_vs_big']['p']))
                # call changes at s = 0: McNemar on (edit flips, crand_big+ does not) vs (crand_big+ flips, edit does not)
                s0 = S['base'][nm][jj] > 0; fe = (S[big][nm][jj] > 0) != s0; fc = (S[f'{t}_big+'][nm][jj] > 0) != s0
                b_, c_ = int((fe & ~fc).sum()), int((fc & ~fe).sum())
                from scipy.stats import binom
                pm = float(min(1.0, 2 * binom.cdf(min(b_, c_), b_ + c_, 0.5))) if b_ + c_ else 1.0
                R[f'{big}_calls'] = dict(flip_edit=float(fe.mean()), flip_crand_big=float(fc.mean()), b=b_, c=c_, p_mcnemar=pm)
            for cn in ('N+', 'N-', 'Nsp'):
                R[cn] = dict(dS=float(ds[cn].mean()), dS_ci=boot_ci(ds[cn], bidx))
            N0 = O0['N'][jj]; scN = 0.1 / (kN[jj] / N0)
            R['N'] = dict(odd_per10=float(((ds['N+'] - ds['N-']) / 2 * scN).mean()), even=float(((ds['N+'] + ds['N-']) / 2).mean()),
                          even_ci=boot_ci((ds['N+'] + ds['N-']) / 2, bidx))
            # H76 5-vector (odd, minus odd crand), order N, lha, wid, ptd, mass
            R['g'] = [R['N']['odd_per10']] + [R[t]['odd_per10'] for t in ('lha', 'wid', 'ptd', 'mass')]
        cos = lambda u, v: float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
        G['cos'] = {m: {r: cos(G['models'][m]['g'], G['models'][r]['g']) for r in ['efn_k', 'g5_py', 'g1_N_py'] + list(FEATS)}
                    for m in names}
        G['cos_irc4'] = {m: {r: cos(G['models'][m]['g'][1:], G['models'][r]['g'][1:]) for r in ['efn_k', 'g5_py'] + list(FEATS)}
                         for m in names}
        q = H76._holm([p for *_, p in pvals]); G['holm'] = [dict(model=m, test=t, p=p, p_holm=float(h)) for (m, t, p), h in zip(pvals, q)]
        # C2ST of crand_big+ (is the large random null-space edit on-manifold?)
        sub = np.sort(np.random.default_rng(5).choice(J, min(a.c2st_n, J), replace=False)); Ls0 = [L0[i] for i in sub]
        Os0 = {k: v[sub] for k, v in O0.items()}; G['c2st'] = {}
        for cn in ('lha_big+', 'wid_big+', 'lha+', 'wid-'):
            Le = [E[cn][0][i] for i in sub]; G['c2st'][cn] = H76.c2st(Ls0, Le, Os0, H76.obs_table(Le))
        log(gen, 'c2st', G['c2st'])
        log(gen, 'summary', json.dumps({m: {k: G['models'][m][k] for k in ('lha+', 'lha-', 'wid+', 'wid-')} for m in ('part_kin', 'efn_k')})[:3000])
    p = RES / 'h76_verify.json'; p.write_text(json.dumps(OUT, indent=1, default=float)); log('wrote', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', default='py,hw'); ap.add_argument('--per_class', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=7621); ap.add_argument('--train_per_class', type=int, default=20000)
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--perm', type=int, default=10000)
    ap.add_argument('--c2st_n', type=int, default=4000)
    main(ap.parse_args())
