"""H16v (loop D, skeptic for H16): independent re-run of the H16 boost battery on a fresh sample, plus two missing controls.
1. Fresh samples (seed 9917, random rows, 5,000 + 5,000 per sample): hls_all (hlsA nets), hls_win (hlsB nets), top (top nets);
   every net scored on its native sample only.
2. Independent boost: an explicit 4x4 Lorentz matrix along the jet momentum (written here, not interp.h12_boost_panel.boost);
   checked per particle against h12.boost, and used for B.
3. Edits: B0.9, B1.1, P0.9, P1.1, A0.9, A1.1, R0.9 (rotation, displacement matched to B0.9); at k = 1/0.9:
   M   translation by t u (m and jet pT fixed; m_A -> k m_A; pT centroid moved by t u)        [H16's edit, re-implemented]
   RM  rotation with pT-weighted mean displacement t                                          [H16's control]
   TC  NEW control: shrink about the pT centroid by c, then translate by the same t u as M; c set per jet so that m_A is
       unchanged; the centroid moves exactly as in M; invariant m -> ~ c m (pair distances x c)
   AC  NEW control: the same shrink about the centroid by c, no translation (centroid fixed; m and m_A -> ~ c m)
   Predictions at k = 1/0.9 (c ~ 0.875):
     net reads m_A about the axis:    M large, TC ~ RM, AC ~ A1.14-like
     net reads the centroid offset / absolute positions (not m_A): TC ~ M or larger, TC >> AC
     net reads invariant m / pairwise geometry only: M ~ RM ~ 0, TC ~ AC
4. Dataset contrasts of the seed-mean B % changed with a bootstrap over jets (independent samples), and the seed-pair range.
5. H12's five-class hls4ml checkpoints (JEDI-linear n64/32/16, small nets; H12 per-jet file): % of t/q/g jets whose
   "top" call (argmax == t) changes under B0.9, vs % of all five-class calls changed -> the binary-task level of the H12 nets.
Run: AI_PY=conda ./job.sh d-h16-verify 2 6 'python -m interp.h16_verify'  -> $AI_RESULTS/physics/h16_verify.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.h12_boost_panel import (boost as boost_h12, pscale, ascale, rotate, jet_obs, copyT, rel, set_rel, four, _from_p,  # noqa: E402
                                    mcnemar, holm)
from interp import h16_train as HT  # noqa: E402
from interp.h16_boost_dataset import hls_sample, top_sample, net_input, runs_for  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
SEED = 9917
PER = int(os.environ.get('H16V_PER', '5000'))
K = round(1 / 0.9, 6)
NB = 1000
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def bfrac(x, rng):
    x = np.asarray(x, float); n = len(x); i = rng.integers(0, n, (NB, n)); m = x[i].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ---------------------------------------------------------------------------------------------- independent boost
def boost_mat(T, lam):
    """explicit Lorentz matrix along the jet 3-momentum n with rapidity zeta: |P| -> lam |P|; axis boosted as massless"""
    real = T['pt'] > 0; px, py, pz, E = four(T)
    p4 = np.stack([E, px, py, pz], -1)                                   # (J, L, 4), (E, p)
    PJ = p4.sum(1); pa = np.linalg.norm(PJ[:, 1:], axis=1); m = np.sqrt(np.maximum(PJ[:, 0] ** 2 - pa ** 2, 1e-12))
    zeta = np.arcsinh(lam * pa / m) - np.arcsinh(pa / m); n = PJ[:, 1:] / pa[:, None]
    ch, sh = np.cosh(zeta), np.sinh(zeta); J = len(pa); L = np.zeros((J, 4, 4))
    L[:, 0, 0] = ch; L[:, 0, 1:] = sh[:, None] * n; L[:, 1:, 0] = sh[:, None] * n
    L[:, 1:, 1:] = np.eye(3)[None] + (ch - 1)[:, None, None] * n[:, :, None] * n[:, None, :]
    q = np.einsum('jab,jlb->jla', L, p4)
    U = copyT(T); pt, eta, phi = _from_p(q[..., 1], q[..., 2], q[..., 3])
    U['pt'] = np.where(real, pt, 0); U['eta'] = np.where(real, eta, 0); U['phi'] = np.where(real, phi, 0)
    a = np.stack([np.ones(J), np.cos(T['ax_phi']) / np.cosh(T['ax_eta']), np.sin(T['ax_phi']) / np.cosh(T['ax_eta']), np.tanh(T['ax_eta'])], 1)
    a2 = np.einsum('jab,jb->ja', L, a); _, U['ax_eta'], U['ax_phi'] = _from_p(a2[:, 1], a2[:, 2], a2[:, 3])
    # checks: particle masses, jet |P|
    q4 = np.stack([U['pt'] * np.cos(U['phi']), U['pt'] * np.sin(U['phi']), U['pt'] * np.sinh(U['eta'])], -1) * real[..., None]
    pa2 = np.linalg.norm(q4.sum(1), axis=1)
    return U, dict(jetP_ratio_err=float(np.abs(pa2 / pa / lam - 1).max()))


# ---------------------------------------------------------------------------------------------- M / TC / AC
def centroid_edits(T, k, rng):
    de, dp = rel(T); w = T['pt']; real = w > 0; M0 = w.sum(1); M2 = (w * (de ** 2 + dp ** 2)).sum(1)
    S = np.stack([(w * de).sum(1), (w * dp).sum(1)], 1); rb = S / M0[:, None]
    a = rng.uniform(0, 2 * np.pi, len(M0)); u = np.stack([np.cos(a), np.sin(a)], 1); us = (u * S).sum(1)
    t = (-us + np.sqrt(us ** 2 + M0 * (k ** 2 - 1) * M2)) / M0
    Mt = copyT(T); set_rel(Mt, np.where(real, de + (t * u[:, 0])[:, None], 0), np.where(real, dp + (t * u[:, 1])[:, None], 0))
    c2 = 1 - (k ** 2 - 1) * M2 / np.maximum(M2 - M0 * (rb ** 2).sum(1), 1e-12); clipped = float((c2 < 0.05).mean()); c = np.sqrt(np.clip(c2, 0.05, None))
    sde = rb[:, :1] + c[:, None] * (de - rb[:, :1]); sdp = rb[:, 1:] + c[:, None] * (dp - rb[:, 1:])
    AC = copyT(T); set_rel(AC, np.where(real, sde, 0), np.where(real, sdp, 0))
    TC = copyT(T); set_rel(TC, np.where(real, sde + (t * u[:, 0])[:, None], 0), np.where(real, sdp + (t * u[:, 1])[:, None], 0))
    return Mt, TC, AC, t, dict(c_median=float(np.median(c)), c_p5_p95=np.percentile(c, [5, 95]).tolist(), clipped=clipped)


def centroid_off(T):
    de, dp = rel(T); w = T['pt']; M0 = w.sum(1)
    return np.hypot((w * de).sum(1), (w * dp).sum(1)) / M0


# ---------------------------------------------------------------------------------------------- battery
def battery(name, T0, y, sgn, runs):
    rng = np.random.default_rng(SEED + 1); erng = np.random.default_rng(SEED + 2); J = len(y)
    X0 = net_input(T0, sgn); O0 = jet_obs(T0); c0 = centroid_off(T0)
    res = dict(sample=name, n=J, seed=SEED, models={}, edits={}, checks={})
    nets, S0 = {}, {}
    for rn, d in runs.items():
        net = HT.Net(d); L = net(X0); s = L[:, 1] - L[:, 0]; nets[rn] = net; S0[rn] = s
        known = json.loads((d / 'test.json').read_text())
        res['models'][rn] = dict(acc=float(((s > 0).astype(int) == y).mean()), known_test_acc=known['accuracy'])
        log(f'{name} {rn}: acc {res["models"][rn]["acc"]:.4f} (test.json {known["accuracy"]:.4f})')
    # independent boost vs h12 boost
    Bm, inf = boost_mat(T0, 0.9); Bh, _ = boost_h12(T0, 0.9); real = T0['pt'] > 0
    dmax = max(float(np.abs(np.where(real, Bm[k] - Bh[k], 0)).max()) for k in ('pt', 'eta'))
    dphi = float(np.abs(np.where(real, (Bm['phi'] - Bh['phi'] + np.pi) % (2 * np.pi) - np.pi, 0)).max())
    res['checks']['boost_vs_h12'] = dict(max_abs_diff_pt_eta=dmax, max_abs_diff_phi=dphi, jetP_ratio_err=inf['jetP_ratio_err'],
                                         ax_diff=float(max(np.abs(Bm['ax_eta'] - Bh['ax_eta']).max(), np.abs(Bm['ax_phi'] - Bh['ax_phi']).max())))
    log(f'{name}: independent boost vs h12.boost {res["checks"]["boost_vs_h12"]}')
    B9 = Bm; d0, p0 = rel(T0); d1, p1 = rel(B9); w = T0['pt']
    D = (w * np.hypot(d1 - d0, p1 - p0)).sum(1) / np.maximum(w.sum(1), 1e-12)
    Mt, TC, AC, t, cinfo = centroid_edits(T0, K, erng)
    res['checks']['c'] = cinfo
    E = [('B0.9', B9), ('B1.1', boost_mat(T0, 1.1)[0]), ('P0.9', pscale(T0, 0.9)[0]), ('P1.1', pscale(T0, 1.1)[0]),
         ('A0.9', ascale(T0, 0.9)[0]), ('A1.1', ascale(T0, 1.1)[0]), ('R0.9', rotate(T0, D, erng)[0]),
         ('M', Mt), ('RM', rotate(T0, t, erng)[0]), ('TC', TC), ('AC', AC)]
    CH = {}
    for en, U in E:
        O1 = jet_obs(U); X1 = net_input(U, sgn); c1 = centroid_off(U)
        rc = {k: dict(mean=float(np.mean(O1[k] / np.maximum(O0[k], 1e-12) - 1)), p99_abs=float(np.percentile(np.abs(O1[k] / np.maximum(O0[k], 1e-12) - 1), 99)))
              for k in ('m', 'jpt', 'mA')}
        rc['centroid_offset'] = dict(before_median=float(np.median(c0)), after_median=float(np.median(c1)))
        e = dict(rel_change=rc, models={})
        for rn, net in nets.items():
            L1 = net(X1); s1 = L1[:, 1] - L1[:, 0]; ch = (S0[rn] > 0) != (s1 > 0); CH[(rn, en)] = ch
            e['models'][rn] = dict(changed=float(ch.mean()), ci=bfrac(ch, rng), to_top=float(((S0[rn] <= 0) & (s1 > 0)).mean()),
                                   to_bkg=float(((S0[rn] > 0) & (s1 <= 0)).mean()))
        res['edits'][en] = e
        log(f'{name} {en}: dm {rc["m"]["mean"]:+.4f} dmA {rc["mA"]["mean"]:+.4f} djpt {rc["jpt"]["mean"]:+.4f} centroid {rc["centroid_offset"]} | '
            + ' '.join(f'{rn} {e["models"][rn]["changed"]:.4f}' for rn in nets))
    res['tests'] = {}
    for rn in nets:
        for a, b in (('B0.9', 'R0.9'), ('B0.9', 'A0.9'), ('B0.9', 'P0.9'), ('B1.1', 'A1.1'), ('B1.1', 'P1.1'),
                     ('M', 'RM'), ('TC', 'RM'), ('TC', 'AC'), ('M', 'TC'), ('M', 'A0.9')):
            r = mcnemar(CH[(rn, a)], CH[(rn, b)]); r['diff'] = float(CH[(rn, a)].mean() - CH[(rn, b)].mean())
            res['tests'][f'{rn}|{a}_vs_{b}'] = r
    res['perjet_B09'] = {rn: CH[(rn, 'B0.9')] for rn in nets}
    return res


def h12_binary():
    out = {}
    for part in ('hls_small', 'hls_jedi'):
        p = OUT / f'h12_{part}_perjet.npz'
        if not p.exists(): out[part] = 'missing'; continue
        d = np.load(p); y = d['y'].astype(int); tqg = np.isin(y, [0, 1, 4]); rng = np.random.default_rng(SEED + 7)
        for k in d.files:
            if not k.endswith('__base'): continue
            mn = k[:-6]
            if f'{mn}__B0.9' not in d.files: continue
            c0 = d[k].argmax(1); c1 = d[f'{mn}__B0.9'].astype(int)
            all5 = c0 != c1; topflip = (c0 == 4) != (c1 == 4)
            acc_tqg = float(((c0 == 4) == (y == 4))[tqg].mean())
            out[f'{part}:{mn}'] = dict(five_class_changed_all=float(all5.mean()), five_class_changed_tqg=float(all5[tqg].mean()),
                                       top_call_changed_tqg=float(topflip[tqg].mean()), ci=bfrac(topflip[tqg], rng),
                                       binary_acc_tqg_top_indicator=acc_tqg, n_tqg=int(tqg.sum()))
            log(f'H12 {part} {mn}: 5-class changed all {all5.mean():.4f}, on t/q/g {all5[tqg].mean():.4f}; top-call changed on t/q/g {topflip[tqg].mean():.4f}')
    return out


def main():
    W = json.loads((HT.H16D / 'windows.json').read_text())
    R = {}
    T, y, sgn, _, _ = hls_sample(PER, SEED); R['hls_all'] = battery('hls_all', T, y, sgn, runs_for(('hlsA',)))
    T, y, sgn, _, _ = hls_sample(PER, SEED, W['hlsB_window']); R['hls_win'] = battery('hls_win', T, y, sgn, runs_for(('hlsB',)))
    T, y, sgn, _, _ = top_sample(PER, SEED); R['top'] = battery('top', T, y, sgn, runs_for(('top',)))
    # Holm per family
    fams = {}
    for s, r in R.items():
        for k, t in r['tests'].items(): fams.setdefault(k.split('|')[1], []).append((s, k, t))
    for f, lst in fams.items():
        h = holm([x[2]['p'] for x in lst])
        for (s, k, t), hh in zip(lst, h): t['holm'] = hh; t['n_family'] = len(lst)
    # dataset contrasts, seed-mean per jet, bootstrap over jets (independent samples)
    native = dict(hlsA='hls_all', hlsB='hls_win', top='top'); con = {}; rng = np.random.default_rng(SEED + 9); tests = []
    for var in ('frac', 'abs'):
        for arch in ('jedis', 'tf', 'pfn'):
            for a, b in (('hlsA', 'top'), ('hlsB', 'top'), ('hlsA', 'hlsB')):
                ga = [rn for rn in R[native[a]]['perjet_B09'] if rn.startswith(f'{a}-{var}-{arch}-')]
                gb = [rn for rn in R[native[b]]['perjet_B09'] if rn.startswith(f'{b}-{var}-{arch}-')]
                if not ga or not gb: continue
                xa = np.mean([R[native[a]]['perjet_B09'][r] for r in ga], 0); xb = np.mean([R[native[b]]['perjet_B09'][r] for r in gb], 0)
                ba = xa[rng.integers(0, len(xa), (NB, len(xa)))].mean(1); bb = xb[rng.integers(0, len(xb), (NB, len(xb)))].mean(1)
                dd = ba - bb; d = float(xa.mean() - xb.mean()); se = float(dd.std())
                from scipy.stats import norm
                p = float(2 * norm.sf(abs(d) / max(se, 1e-12)))
                pairs = [float(R[native[a]]['perjet_B09'][i].mean() - R[native[b]]['perjet_B09'][j].mean()) for i in ga for j in gb]
                k = f'{var}-{arch}|B0.9|{a}-{b}'; con[k] = dict(diff=d, ci=np.percentile(dd, [2.5, 97.5]).tolist(), p=p, seed_pair_range=[min(pairs), max(pairs)])
                tests.append((k, p))
    for (k, _), h in zip(tests, holm([t[1] for t in tests])): con[k]['holm'] = h
    for s in R: R[s].pop('perjet_B09')
    R['contrasts'] = con
    R['h12_binary'] = h12_binary()
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / 'h16_verify.json'; p.write_text(json.dumps(R, indent=1, default=float)); log(f'wrote {p}')
    for k, v in con.items(): log(f'contrast {k}: {100 * v["diff"]:+.2f} pp CI {[round(100 * x, 2) for x in v["ci"]]} holm {v["holm"]:.2g} seed pairs {[round(100 * x, 2) for x in v["seed_pair_range"]]}')


if __name__ == '__main__':
    main()
