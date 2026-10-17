"""E15 verification, part 2 (skeptic, 2026-10-16): read-out-free test of sum vs product on the network's own units.

Sample as interp/e15_verify.py: sample(per_class=2000, seed=23) minus the jets also drawn by seed 11.
Edits (real particles only; jets leaving the l1 input range in any cell dropped):
  M0 step: pT × c, angles × c^-1/2   (M0 × c, M2 fixed),  c ∈ {0.8, 1.25}
  M2 step: angles × c^1/2           (M2 × c, M0 fixed),  c ∈ {0.8, 1.25}   (same log step in the moment as the M0 step)
  both:    pT × c                   (M0 × c, M2 × c)
Per jet and unit: e0 = ΔU(M0 step) / Δlog c, e2 = ΔU(M2 step) / Δlog c.  If a unit reads M0 and M2 only through V = √(M0·M2)
(any nonlinearity after that), the two steps give the same V, so e0 = e2 for every jet (d = 0).  If it reads V = λM0 + M2,
e0/(e0 + e2) = s (the M0 term's share), so d = (2s − 1) ≈ +0.53 / 0 / −0.46 in the low / mid / high M2 third.
Statistic per unit and M2 third (jets with the unit > 0 in all 5 cells for post-ReLU stages):
  d = Σ_j (e0 − e2)·sgn(e0 + e2) / Σ_j |e0 + e2|  ;  closure = Σ_j (e_both − e0 − e2)·sgn / Σ_j |e0 + e2|  (0 if U = f(M0, M2) locally)
Units: live S, h4, h5, z units; logit differences.  Same statistic on the moment models (V_sum, V_prod) and on calibration unit
populations built from the true moments (windows on V_sum, windows on V_prod, ReLU on separate z(M0), z(M2)).
Calls: fraction of jets whose call differs between the M0 × 1.25 and the M2 × 1.25 cells (same √(M0·M2)), against the fraction
changed by either step.   Writes $AI_RESULTS/jedi/e15_verify_iso.json.   Run: python -m interp.e15_verify_iso
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_JSON = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi' / 'e15_verify_iso.json'
PER, SEED, SEED_E15 = 2000, 23, 11
LSTEP = float(np.log(1.25) - np.log(0.8))
STG = ('S', 'h4', 'h5', 'z'); KEEP = STG + ('logits',)
CL = ['g', 'q', 'W', 'Z', 't']
LOGIT_DIFFS = [(3, 2), (4, 3), (2, 1), (4, 2), (2, 0)]
NAMED = [('z', u) for u in (0, 7, 9, 10, 11, 12, 14)] + [('h4', 18), ('h5', 0), ('h5', 28), ('S', 0), ('S', 50), ('S', 29)]
NBOOT = 200
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)


from interp.jedi_core import Jedi, sample  # noqa: E402
from jetdistill.config import RESULTS  # noqa: E402

x, y, Q = sample(per_class=PER, seed=SEED); del Q
d_ = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); yall = np.asarray(d_['y']); del d_


def idx_of(seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(yall == c), PER, replace=False) for c in range(5)]))


i_new, i_old = idx_of(SEED), idx_of(SEED_E15)
assert np.array_equal(yall[i_new], y)
fresh = ~np.isin(i_new, i_old)
x = np.ascontiguousarray(x[fresh]).astype(np.float32); y = y[fresh]; J = len(x); real = x[..., 0] > 0
net = Jedi(64)
res = dict(N=J, seed=SEED, dropped_shared_with_seed11=int((~fresh).sum()))


def in_range(net_, xx):
    q = net_.L['l1'].iq.quantizer
    k_, i_, f_ = (np.asarray(getattr(q, a)).reshape(64, 3) for a in ('k', 'i', 'f'))
    hi = 2.0 ** i_ - 2.0 ** -f_; lo = np.where(k_ > 0, -2.0 ** i_, 0.0)
    xn = net_.inputs(xx, 64); rl = xx[..., 0] > 0
    return ~((((xn > hi[None]) | (xn < lo[None])) & rl[..., None]).any((1, 2)))


def moments(xx):
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    return dict(M0=pt.sum(1), M2=(pt * (eta ** 2 + phi ** 2)).sum(1))


def edit(xx, c, k):
    e = xx.copy()
    e[..., 0] = np.where(real, e[..., 0] * np.float32(c), e[..., 0])
    e[..., 1:] = np.where(real[..., None], e[..., 1:] * np.float32(k), e[..., 1:])
    return e


M = moments(x); LAM = float(np.median(M['M2']) / np.median(M['M0']))
share = LAM * M['M0'] / (LAM * M['M0'] + M['M2'])
TERT = np.digitize(M['M2'], np.percentile(M['M2'], [100 / 3, 200 / 3])); TN = ('low', 'mid', 'high')
res['sum_2s_minus_1_by_tertile'] = [float(np.median(2 * share[TERT == t] - 1)) for t in range(3)]
CELLS = {'base': (1.0, 1.0), 'M0_hi': (1.25, 1.25 ** -0.5), 'M0_lo': (0.8, 0.8 ** -0.5), 'M2_hi': (1.0, 1.25 ** 0.5), 'M2_lo': (1.0, 0.8 ** 0.5),
         'both_hi': (1.25, 1.0), 'both_lo': (0.8, 1.0)}
R = {}; MOM = {}; inr = np.ones(J, bool)
for nm, (c, k) in CELLS.items():
    xe = x if nm == 'base' else edit(x, c, k)
    inr &= in_range(net, xe); MOM[nm] = moments(xe)
    R[nm] = {kk: v.astype(np.float32) for kk, v in net.run(xe, keep=KEEP, batch=2500).items()}
for nm in CELLS:
    R[nm]['ldiff'] = np.stack([R[nm]['logits'][:, a] - R[nm]['logits'][:, b] for a, b in LOGIT_DIFFS], 1)
res['in_range'] = int(inr.sum())
res['moment_ratio_medians'] = {nm: [float(np.median(MOM[nm]['M0'] / M['M0'])), float(np.median(MOM[nm]['M2'] / M['M2']))] for nm in CELLS}
res['acc'] = float((R['base']['logits'].argmax(1) == y).mean())
log(f'N {J}, in range {inr.sum()}, acc {res["acc"]:.4f}; moment ratios {res["moment_ratio_medians"]}; sum 2s−1 by tertile {res["sum_2s_minus_1_by_tertile"]}')
WB = np.random.default_rng(SEED + 5).poisson(1.0, size=(NBOOT, J)).astype(np.float64)


def dstat(get, jets, need_on):
    """per tertile: d, closure, share-like r, n; get(cell) -> per-jet values"""
    V = {nm: np.asarray(get(nm), np.float64) for nm in CELLS}
    ok = jets & ((np.stack([V[nm] for nm in CELLS]) > 0).all(0) if need_on else True)
    e0 = (V['M0_hi'] - V['M0_lo']) / LSTEP; e2 = (V['M2_hi'] - V['M2_lo']) / LSTEP; eb = (V['both_hi'] - V['both_lo']) / LSTEP
    sg = np.sign(e0 + e2); den = np.abs(e0 + e2); sd = V['base'][jets].std()
    out = {}
    for ti, tn in enumerate(TN):
        s_ = ok & (TERT == ti)
        if s_.sum() < 30 or den[s_].mean() < 0.05 * max(sd, 1e-12): out[tn] = dict(n=int(s_.sum())); continue
        num = (e0 - e2) * sg; cl = (eb - e0 - e2) * sg
        d = float(num[s_].sum() / den[s_].sum()); w = WB[:, s_]
        dB = (w @ num[s_]) / np.maximum(w @ den[s_], 1e-30)
        out[tn] = dict(n=int(s_.sum()), d=d, d_se=float(dB.std()), closure=float(cl[s_].sum() / den[s_].sum()),
                       resp_sd=float(den[s_].mean() / max(sd, 1e-12)))
    return out


def fmt_d(o):
    return ' '.join(f'{o[t]["d"]:+.2f}±{o[t]["d_se"]:.2f}' if 'd' in o[t] else '  n/a    ' for t in TN)


ALLJ = inr.copy()
res['models'] = {}
for nm, f in (('sum', lambda mm: LAM * mm['M0'] + mm['M2']), ('prod', lambda mm: np.sqrt(mm['M0'] * mm['M2']))):
    res['models'][nm] = dstat(lambda c: f(MOM[c]), ALLJ, False); log(f'model {nm}: d low/mid/high {fmt_d(res["models"][nm])}')

# network units
res['units'] = {}; res['summary'] = {}
for st in STG + ('ldiff',):
    base = R['base'][st]; sdv = base.std(0); on = (base > 0).mean(0)
    live = np.flatnonzero((sdv > 1e-6) & ((on > 0.02) if st in ('S', 'h4', 'h5') else True))
    res['units'][st] = {}
    for u in live:
        res['units'][st][str(u)] = dstat(lambda c: R[c][st][:, u], ALLJ, st in ('S', 'h4', 'h5'))
    summ = {}
    for tn in TN:
        ds = [v[tn]['d'] for v in res['units'][st].values() if 'd' in v[tn] and v[tn]['resp_sd'] > 0.1]
        cs = [v[tn]['closure'] for v in res['units'][st].values() if 'd' in v[tn] and v[tn]['resp_sd'] > 0.1]
        summ[tn] = dict(n_units=len(ds), d_median=float(np.median(ds)) if ds else None,
                        d_q25_q75=[float(np.percentile(ds, 25)), float(np.percentile(ds, 75))] if ds else None,
                        frac_abs_d_below_0p15=float(np.mean(np.abs(ds) < 0.15)) if ds else None,
                        closure_median_abs=float(np.median(np.abs(cs))) if cs else None)
    res['summary'][st] = summ
    log(f'{st}: units with response > 0.1 sd: ' + ' | '.join(f'{tn} n {summ[tn]["n_units"]} d median {summ[tn]["d_median"]} IQR {summ[tn]["d_q25_q75"]} '
                                                            f'|d|<0.15 {summ[tn]["frac_abs_d_below_0p15"]} |closure| med {summ[tn]["closure_median_abs"]}' for tn in TN))
for st, u in NAMED:
    o = res['units'].get(st, {}).get(str(u))
    if o: log(f'   {st}_{u}: d {fmt_d(o)}; closure ' + ' '.join(f'{o[t].get("closure", float("nan")):+.2f}' for t in TN)
              + '; resp ' + ' '.join(f'{o[t].get("resp_sd", float("nan")):.2f}' for t in TN))
for i, (a, b) in enumerate(LOGIT_DIFFS):
    o = res['units']['ldiff'][str(i)]
    log(f'   logit {CL[a]}−{CL[b]}: d {fmt_d(o)}; closure ' + ' '.join(f'{o[t].get("closure", float("nan")):+.2f}' for t in TN))

# calibration populations (unit level, no read-out)
z_ = lambda v, v0: (v - v0.mean()) / v0.std()
rng = np.random.default_rng(300)
Vs = {c: LAM * MOM[c]['M0'] + MOM[c]['M2'] for c in CELLS}; Vp = {c: np.sqrt(MOM[c]['M0'] * MOM[c]['M2']) for c in CELLS}
res['calibration'] = {}
for nm in ('sumwin', 'prodwin', 'sepwin'):
    ds = {tn: [] for tn in TN}
    for i in range(32):
        if nm == 'sepwin':
            th = rng.uniform(0, 2 * np.pi); q = rng.uniform(0.05, 0.95)
            P = {c: z_(MOM[c]['M0'], M['M0']) * np.cos(th) + z_(MOM[c]['M2'], M['M2']) * np.sin(th) for c in CELLS}
            qq = np.quantile(P['base'], q); F = {c: np.maximum(P[c] - qq, 0) for c in CELLS}
        else:
            V = Vs if nm == 'sumwin' else Vp; s = rng.choice([-1.0, 1.0]); qq = np.quantile(z_(V['base'], V['base']), rng.uniform(0.05, 0.95))
            F = {c: np.maximum(s * (z_(V[c], V['base']) - qq), 0) for c in CELLS}
        o = dstat(lambda c: F[c], ALLJ, True)
        for tn in TN:
            if 'd' in o[tn] and o[tn]['resp_sd'] > 0.1: ds[tn].append(o[tn]['d'])
    res['calibration'][nm] = {tn: dict(n_units=len(ds[tn]), d_median=float(np.median(ds[tn])) if ds[tn] else None,
                                       d_q25_q75=[float(np.percentile(ds[tn], 25)), float(np.percentile(ds[tn], 75))] if ds[tn] else None) for tn in TN}
    log(f'calibration {nm}: ' + ' | '.join(f'{tn} n {v["n_units"]} d median {v["d_median"]} IQR {v["d_q25_q75"]}' for tn, v in res['calibration'][nm].items()))

# calls
c0 = R['base']['logits'].argmax(1); cA = R['M0_hi']['logits'].argmax(1); cB = R['M2_hi']['logits'].argmax(1)
cA2 = R['M0_lo']['logits'].argmax(1); cB2 = R['M2_lo']['logits'].argmax(1)
res['calls'] = {}
for tn_i, tn in enumerate(TN + ('all',)):
    s_ = inr & ((TERT == tn_i) if tn != 'all' else True)
    res['calls'][tn] = dict(n=int(s_.sum()), M0hi_vs_M2hi_differ=float((cA[s_] != cB[s_]).mean()), M0lo_vs_M2lo_differ=float((cA2[s_] != cB2[s_]).mean()),
                            M0hi_changed=float((cA[s_] != c0[s_]).mean()), M2hi_changed=float((cB[s_] != c0[s_]).mean()),
                            M0lo_changed=float((cA2[s_] != c0[s_]).mean()), M2lo_changed=float((cB2[s_] != c0[s_]).mean()))
    W = s_ & (y == 2) & (c0 == 2)
    res['calls'][tn].update(W_n=int(W.sum()), W_to_Z_M0hi=float((cA[W] == 3).mean()) if W.sum() >= 20 else None,
                            W_to_Z_M2hi=float((cB[W] == 3).mean()) if W.sum() >= 20 else None)
    log(f'calls {tn}: {res["calls"][tn]}')
res['wall_s'] = float(time.time() - t0)
OUT_JSON.write_text(json.dumps(res))
log(f'done -> {OUT_JSON}')
