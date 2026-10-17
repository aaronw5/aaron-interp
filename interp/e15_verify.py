"""E15 verification (skeptic, 2026-10-16): does the sum/product reading of E15 hold on a fresh sample, and do its metrics
discriminate a sum from a product when the answer is known?

Sample: sample(per_class=2000, seed=23); jets also drawn by seed 11 (E15's sample) are dropped.  Fixed-point JEDI-linear
3-feature n64 (interp.jedi_core.Jedi), stages S, h4, h5, z, logits.  Edits as E15 (real particles only; pT order kept; jets
leaving the l1 input range in any cell dropped): A2 cells pT × c, angles × k/√c (M0 × c, M2 × k²), c, k ∈ {0.8, 1, 1.25};
A cells pT × c at angles × 1 (M0 × c, M2 × c), c ∈ {0.8, 1.25}.  Read-outs: standardised ridge of m (GeV) and log m on a
stage, fit on the unedited train half, evaluated on the test half (as E15).

1  Reproduction: A2 interaction of the read-outs (m̂ GeV and exp(log m̂): fstats β = I / I_pred; log m̂: β_sum), named units,
   T ratios per M2 third (M2-only ÷ pT×c, M0-only ÷ pT×c), B (S0 patch at matched S29/S50; S19 control).
2  Calibration control (missing in E15): the same read-out procedure and metrics applied to unit populations whose combination
   rule is known, built from the true moments of the same jets in the same cells:
     lin       [M0, M2] (an additive sum held linearly)
     sumwin    64 ReLU windows on V = λM0 + M2            (sum, then nonlinear units)
     prodwin   64 ReLU windows on V = √(M0·M2)            (product)
     sepwin    64 ReLU units on random directions in (z(M0), z(M2))          (generic ReLU layer on separate moments)
     logsepwin 64 ReLU units on random directions in (z(log M0), z(log M2))
   and random (untrained) dense ReLU layers applied to the trained network's own S (rand1: 64 units; rand2: 64 → 32), 3 seeds.
3  Target control: ridge read-outs of V_sum and V_prod themselves from S, h4, h5, z (what interaction a read-out of a known sum
   or product target shows from the same units).
4  Carriers: S0 and S50 responses per M2 third (ideal M0 carrier: M2-only ÷ pT×c = 0, M0-only ÷ pT×c = 1; ideal M2 carrier: 2, 0).
5  S0 path share: S0-patch slope (on ½Δlog S0) ÷ 2·(M0-only response), per M2 third; S0 × 0.8 and × 1.25 on every jet.
Writes $AI_RESULTS/jedi/e15_verify.json.   Run: python -m interp.e15_verify
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
OUT_JSON = OUT_DIR / 'e15_verify.json'
PER, SEED, SEED_E15 = 2000, 23, 11
CS = (0.8, 1.0, 1.25); KS = (0.8, 1.0, 1.25)
CORNERS = [(ci, ki) for ci in (0, 2) for ki in (0, 2)]
LSTEP = float(np.log(1.25) - np.log(0.8))
STG = ('S', 'h4', 'h5', 'z'); KEEP = STG + ('logits',)
NBOOT = 300
MIN_N = 30
W_, Z_, T_ = 2, 3, 4
MASSLIKE = [('h4', 18), ('h5', 0), ('h5', 28), ('z', 9), ('z', 11), ('z', 12)]
S_NAMED = [0, 50, 56, 1, 29, 13]
TN = ('low', 'mid', 'high')
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)


from interp.jedi_core import Jedi, sample  # noqa: E402
from jetdistill.config import RESULTS  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
res = dict(per_class=PER, seed=SEED, nboot=NBOOT)

# ---------------------------------------------------------------- sample (fresh: seed-11 jets removed)
x, y, Q = sample(per_class=PER, seed=SEED); del Q
d_ = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); yall = np.asarray(d_['y']); del d_


def idx_of(seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(yall == c), PER, replace=False) for c in range(5)]))


i_new, i_old = idx_of(SEED), idx_of(SEED_E15)
assert np.array_equal(yall[i_new], y), 'index reconstruction does not match sample()'
fresh = ~np.isin(i_new, i_old)
res['overlap_with_seed11'] = int((~fresh).sum())
x = np.ascontiguousarray(x[fresh]).astype(np.float32); y = y[fresh]; J = len(x)
res['N'] = J
real = x[..., 0] > 0
log(f'sample seed {SEED}: {len(fresh)} jets, {res["overlap_with_seed11"]} shared with seed {SEED_E15} dropped -> {J}')

net = Jedi(64)
R0 = net.run(x, keep=KEEP, batch=2500); call0 = R0['logits'].argmax(1)
res['acc'] = float((call0 == y).mean())
chk = net.run_jetwise(x, {'S': (np.ones(64), R0['S'])}, keep=KEEP, batch=2500)
res['jetwise_identity_maxdiff'] = {k: float(np.abs(chk[k] - R0[k]).max()) for k in KEEP}; del chk
log(f'accuracy {res["acc"]:.4f} (binomial sd {np.sqrt(res["acc"] * (1 - res["acc"]) / J):.4f}); run_jetwise identity {res["jetwise_identity_maxdiff"]}')


def in_range(net_, xx):
    q = net_.L['l1'].iq.quantizer
    k_, i_, f_ = (np.asarray(getattr(q, a)).reshape(64, 3) for a in ('k', 'i', 'f'))
    hi = 2.0 ** i_ - 2.0 ** -f_; lo = np.where(k_ > 0, -2.0 ** i_, 0.0)
    xn = net_.inputs(xx, 64); rl = xx[..., 0] > 0
    return ~((((xn > hi[None]) | (xn < lo[None])) & rl[..., None]).any((1, 2)))


def moments(xx):
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    r2_ = eta ** 2 + phi ** 2
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=pt.sum(1), M2=(pt * r2_).sum(1), m=m, logm=np.log(np.maximum(m, 1e-3)))


def edit(xx, c, k):
    e = xx.copy()
    e[..., 0] = np.where(real, e[..., 0] * np.float32(c), e[..., 0])
    e[..., 1:] = np.where(real[..., None], e[..., 1:] * np.float32(k), e[..., 1:])
    return e


M = moments(x)
LAM = float(np.median(M['M2']) / np.median(M['M0']))
share_M = LAM * M['M0'] / (LAM * M['M0'] + M['M2'])
T_EDGES = np.percentile(M['M2'], [100 / 3, 200 / 3]); TERT = np.digitize(M['M2'], T_EDGES)
res['sum_model'] = dict(lambda_M0=LAM, M2_tertile_edges=[float(v) for v in T_EDGES],
                        share_median_by_M2_tertile=[float(np.median(share_M[TERT == t])) for t in range(3)])
log(f'λ {LAM:.4g}; M2 tertile edges {T_EDGES}; sum share by tertile {res["sum_model"]["share_median_by_M2_tertile"]}')

rng = np.random.default_rng(SEED); perm = rng.permutation(J)
tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:]); trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
is_te = np.zeros(J, bool); is_te[te] = True
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)


class Ridge:
    """standardised ridge as E15 (alpha on a validation part of the train half, refit on the whole half)"""

    def __init__(self, X, t, idx_fit=trf, idx_val=trv, idx_all=tr):
        X = np.asarray(X, np.float64)
        self.mu = X[idx_all].mean(0); self.sd = X[idx_all].std(0); self.sd[self.sd < 1e-9] = np.inf
        Zs = (X - self.mu) / self.sd; self.tm = t[idx_all].mean(); self.ts = t[idx_all].std(); tt = (t - self.tm) / self.ts; best = None
        for a in ALPHAS:
            w = self._fit(Zs[idx_fit], tt[idx_fit], a); e = ((Zs[idx_val] @ w - tt[idx_val]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.alpha = best[1]; self.w = self._fit(Zs[idx_all], tt[idx_all], self.alpha)

    @staticmethod
    def _fit(Zs, t, a):
        Zc = Zs - Zs.mean(0); return np.linalg.solve(Zc.T @ Zc + a * len(Zs) * np.eye(Zs.shape[1]), Zc.T @ (t - t.mean()))

    def __call__(self, X):
        return ((np.asarray(X, np.float64) - self.mu) / self.sd) @ self.w * self.ts + self.tm


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


WB = np.random.default_rng(SEED + 1).poisson(1.0, size=(NBOOT, J)).astype(np.uint8)


def inter(V, ci, ki):
    return V[ci, ki] - V[ci, 1] - V[1, ki] + V[1, 1]


def fstats(V, sd, wb, min_n=MIN_N):
    """E15's factorial statistics (verbatim logic): V (3, 3, n); sd; wb (NBOOT, n)"""
    ok = (V > 0).all((0, 1)); n = int(ok.sum()); out = dict(n=n)
    if n < min_n or sd <= 1e-9: return out
    V = V[:, :, ok]; w = wb[:, ok].astype(np.float64); wsum = np.maximum(w.sum(1), 1.0)
    B = V[1, 1]
    dc = V[2, 1] - V[0, 1]; dk = V[1, 2] - V[1, 0]
    ec = float(dc.mean() / LSTEP / sd); ek = float(dk.mean() / LSTEP / sd)
    ssign = np.sign(ec * ek) if abs(ec * ek) > 1e-12 else 1.0
    a = np.zeros(n); p = np.zeros(n); absl = []; absg = []; closer = []
    for ci, ki in CORNERS:
        c, k = CS[ci], KS[ki]
        U = V[ci, ki]; lin = V[ci, 1] + V[1, ki] - B; logn = V[ci, 1] * V[1, ki] / B
        Im = U - lin; Ip = logn - lin
        s = np.sign((c - 1) * (k * k - 1)) * ssign
        a += Im * s; p += Ip * s
        absl.append(np.abs(Im)); absg.append(np.abs(U - logn))
        closer.append(np.where(Ip == 0, 0.5, (np.abs(U - logn) < np.abs(Im)).astype(float)))
    closer = np.concatenate(closer)
    bI = (w @ a) / (4 * wsum) / sd
    Ip_s = float(p.mean() / 4 / sd); I_s = float(a.mean() / 4 / sd)
    return dict(n=n, e_c=ec, e_k=ek, frac_log_closer=float(closer.mean()), I=I_s, I_se=float(bI.std()), I_pred=Ip_s,
                beta=float(I_s / Ip_s) if abs(Ip_s) > 1e-6 else None)


def log_interaction(Vl, Vs, wb):
    """E15's log interaction projected on σ = −sign((c−1)(k−1)); β_sum = I / I(log V_sum); per-corner values kept"""
    w = wb.astype(np.float64); wsum = np.maximum(w.sum(1), 1.0); n = Vl.shape[2]
    a = np.zeros(n); b = np.zeros(n); per = {}
    for ci, ki in CORNERS:
        sg = -np.sign((CS[ci] - 1) * (KS[ki] - 1))
        Il = inter(Vl, ci, ki); Is = inter(Vs, ci, ki)
        per[f'c{CS[ci]}k{KS[ki]}'] = [float(Il.mean()), float(Is.mean())]
        a += sg * Il / 4; b += sg * Is / 4
    bI = (w @ a) / wsum
    return dict(I_proj=float(a.mean()), I_proj_se=float(bI.std()), I_sum_proj=float(b.mean()),
                beta_sum=float(a.mean() / b.mean()), mean_abs_corner=float(np.mean([abs(v[0]) for v in per.values()])),
                per_corner_meas_sum=per)


# ---------------------------------------------------------------- cells
CELLS = [('A2', c, k) for c in CS for k in KS] + [('A', 0.8, 1.0), ('A', 1.25, 1.0)]
BASE = ('A2', 1.0, 1.0)
inr = np.ones(J, bool); MOM = {}; RC = {}
for ck in CELLS:
    d, c, k = ck
    if ck == BASE:
        RC[ck] = {kk: v.astype(np.float32) for kk, v in R0.items()}; MOM[ck] = M; continue
    xe = edit(x, c, k / np.sqrt(c) if d == 'A2' else k)
    inr &= in_range(net, xe); MOM[ck] = moments(xe)
    RC[ck] = {kk: v.astype(np.float32) for kk, v in net.run(xe, keep=KEEP, batch=2500).items()}
    del xe
res['in_range_all_cells'] = int(inr.sum())
res['moment_ratio_medians'] = {f'{d}c{c}k{k}': [float(np.median(MOM[(d, c, k)]['M0'] / M['M0'])), float(np.median(MOM[(d, c, k)]['M2'] / M['M2']))]
                               for (d, c, k) in CELLS}
res['calls'] = {f'{d}c{c}k{k}': dict(W_to_Z=float((RC[(d, c, k)]['logits'].argmax(1)[inr & (y == W_)] == Z_).mean()),
                                     Z_to_t=float((RC[(d, c, k)]['logits'].argmax(1)[inr & (y == Z_)] == T_).mean())) for (d, c, k) in CELLS}
log(f'{len(CELLS) - 1} edited passes; in range in all cells {inr.sum()}; calls {res["calls"]}')
sel = inr & is_te
A2KEYS = [('A2', c, k) for c in CS for k in KS]


def cube_of(get, jets):
    """(3, 3, n) over the A2 cells"""
    return np.stack([np.stack([np.asarray(get(('A2', c, k)))[jets] for k in KS], 0) for c in CS], 0).astype(np.float64)


V_SUM = {ck: LAM * MOM[ck]['M0'] + MOM[ck]['M2'] for ck in CELLS}
V_PROD = {ck: np.sqrt(MOM[ck]['M0'] * MOM[ck]['M2']) for ck in CELLS}
LVS = np.log(cube_of(lambda ck: V_SUM[ck], sel))


def T_ratios(L, jets=None):
    """L: dict cell -> per-jet log read-out.  Per M2 tertile on sel: pT×c gain, M2-only ÷ gain, M0-only ÷ gain (+ bootstrap se)"""
    jets = sel if jets is None else jets
    g = (L[('A', 1.25, 1.0)] - L[('A', 0.8, 1.0)]) / LSTEP
    rk = (L[('A2', 1.0, 1.25)] - L[('A2', 1.0, 0.8)]) / LSTEP
    r0 = (L[('A2', 1.25, 1.0)] - L[('A2', 0.8, 1.0)]) / LSTEP
    out = {}; boots = {}
    for ti, tn in enumerate(TN):
        s_ = jets & (TERT == ti); w = WB[:, s_].astype(np.float64); ws = np.maximum(w.sum(1), 1.0)
        bg, bk, b0 = w @ g[s_] / ws, w @ rk[s_] / ws, w @ r0[s_] / ws
        mg, mk, m0 = float(g[s_].mean()), float(rk[s_].mean()), float(r0[s_].mean())
        out[tn] = dict(n=int(s_.sum()), gain_c=mg, k_over_c=mk / mg, k_over_c_se=float((bk / bg).std()), c0_over_c=m0 / mg,
                       c0_over_c_se=float((b0 / bg).std()), consistency=m0 / mg + 0.5 * mk / mg)
        boots[tn] = (bk / bg, b0 / bg)
    dv = lambda a, b: a / b if abs(b) > 1e-9 else None
    with np.errstate(divide='ignore', invalid='ignore'):
        out['high_over_low'] = dict(k_over_c=dv(out['high']['k_over_c'], out['low']['k_over_c']),
                                    k_over_c_se=float(np.nanstd(boots['high'][0] / boots['low'][0])) if abs(out['low']['k_over_c']) > 1e-9 else None,
                                    c0_over_c=dv(out['high']['c0_over_c'], out['low']['c0_over_c']),
                                    c0_over_c_se=float(np.nanstd(boots['high'][1] / boots['low'][1])) if abs(out['low']['c0_over_c']) > 1e-9 else None)
    return out


def f2(v, p='+.2f'):
    return 'n/a' if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, p)


def readout_suite(F, tag, unit_stats=False):
    """F: dict cell -> (J, d) features.  Ridge read-outs of m and log m fit on the unedited train half; A2 metrics + T."""
    rm, rl = Ridge(F[BASE], M['m']), Ridge(F[BASE], M['logm'])
    Lm = {ck: rm(F[ck]) for ck in CELLS}; Ll = {ck: rl(F[ck]) for ck in CELLS}
    out = dict(r2_m=r2(Lm[BASE][te], M['m'][te]), r2_logm=r2(Ll[BASE][te], M['logm'][te]))
    out['m_gev'] = fstats(cube_of(lambda ck: Lm[ck], sel), float(Lm[BASE][te].std()), WB[:, sel])
    out['exp_logm'] = fstats(cube_of(lambda ck: np.exp(Ll[ck]), sel), float(np.exp(Ll[BASE])[te].std()), WB[:, sel])
    out['logm'] = log_interaction(cube_of(lambda ck: Ll[ck], sel), LVS, WB[:, sel])
    out['T'] = T_ratios(Ll)
    if unit_stats:
        X0 = np.asarray(F[BASE], np.float64); sd = X0.std(0); on = (X0 > 0).mean(0)
        fl, be = [], []
        for u in np.flatnonzero((sd > 1e-6) & (on > 0.02)):
            st_ = fstats(cube_of(lambda ck: np.asarray(F[ck])[:, u], inr), float(sd[u]), WB[:, inr])
            if 'I' not in st_: continue
            fl.append(st_['frac_log_closer'])
            if abs(st_['I_pred']) > 0.05 and st_['beta'] is not None: be.append(st_['beta'])
        out['units'] = dict(n=len(fl), frac_log_closer_median=float(np.median(fl)) if fl else None,
                            frac_log_closer_q10_q90=[float(np.percentile(fl, 10)), float(np.percentile(fl, 90))] if fl else None,
                            beta_n=len(be), beta_median=float(np.median(be)) if be else None,
                            beta_min_max=[float(np.min(be)), float(np.max(be))] if be else None)
    T = out['T']
    log(f'{tag:14s} R² m {out["r2_m"]:.3f} logm {out["r2_logm"]:.3f} | m̂ GeV β {f2(out["m_gev"].get("beta"))} '
        f'(I {out["m_gev"].get("I", 0):+.4f}±{out["m_gev"].get("I_se", 0):.4f}, frac log closer {out["m_gev"].get("frac_log_closer", 0):.2f}) | '
        f'exp(log m̂) β {f2(out["exp_logm"].get("beta"))} | log m̂ β_sum {out["logm"]["beta_sum"]:+.2f}±{out["logm"]["I_proj_se"] / abs(out["logm"]["I_sum_proj"]):.2f} '
        f'(mean |I| per corner {out["logm"]["mean_abs_corner"]:.4f}) | T k/c ' + ' '.join(f'{T[t]["k_over_c"]:.3f}' for t in TN)
        + ' c0/c ' + ' '.join(f'{T[t]["c0_over_c"]:.3f}' for t in TN) + ' cons ' + ' '.join(f'{T[t]["consistency"]:.3f}' for t in TN)
        + (f' | units n {out["units"]["n"]} frac log closer med {out["units"]["frac_log_closer_median"]} β med {out["units"]["beta_median"]} range {out["units"]["beta_min_max"]}' if unit_stats else ''))
    return out


# ================================================================ 1. reproduction on the network
res['network'] = {}
for st in STG:
    res['network'][st] = readout_suite({ck: RC[ck][st] for ck in CELLS}, f'net {st}', unit_stats=True)
res['models'] = {}
for nm, V in (('sum', V_SUM), ('prod', V_PROD)):
    Lg = {ck: np.log(V[ck]) for ck in CELLS}
    res['models'][nm] = dict(gev=fstats(cube_of(lambda ck: V[ck], sel), float(V[BASE][te].std()), WB[:, sel]),
                             log=log_interaction(cube_of(lambda ck: Lg[ck], sel), LVS, WB[:, sel]), T=T_ratios(Lg))
    mm = res['models'][nm]
    log(f'model {nm}: GeV β {mm["gev"]["beta"]} | log β_sum {mm["log"]["beta_sum"]:+.2f} | T k/c ' + ' '.join(f'{mm["T"][t]["k_over_c"]:.3f}' for t in TN)
        + ' c0/c ' + ' '.join(f'{mm["T"][t]["c0_over_c"]:.3f}' for t in TN))
res['named_units'] = {}
for st, u in [('S', s) for s in S_NAMED] + MASSLIKE:
    X0 = R0[st][:, u]
    res['named_units'][f'{st}_{u}'] = fs = fstats(cube_of(lambda ck: RC[ck][st][:, u], inr), float(X0.std()), WB[:, inr])
    if 'I' in fs:
        log(f'unit {st}_{u}: n {fs["n"]} e_c {fs["e_c"]:+.2f} e_k {fs["e_k"]:+.2f} I {fs["I"]:+.4f}±{fs["I_se"]:.4f} pred {fs["I_pred"]:+.4f} β {fs["beta"]} frac log closer {fs["frac_log_closer"]:.2f}')
    else:
        log(f'unit {st}_{u}: n {fs["n"]} (< {MIN_N})')
OUT_JSON.write_text(json.dumps(res))

# ================================================================ 3. target control: read-outs of V_sum and V_prod themselves
res['target_control'] = {}
for st in STG:
    F = {ck: RC[ck][st] for ck in CELLS}; row = {}
    for nm, V in (('sum', V_SUM), ('prod', V_PROD)):
        rr = Ridge(F[BASE], V[BASE]); L_ = {ck: rr(F[ck]) for ck in A2KEYS}
        row[nm] = fstats(cube_of(lambda ck: L_[ck], sel), float(L_[BASE][te].std()), WB[:, sel]); row[nm]['r2'] = r2(L_[BASE][te], V[BASE][te])
    res['target_control'][st] = row
    log(f'target control at {st}: read-out of V_sum β {row["sum"].get("beta")} (R² {row["sum"]["r2"]:.3f}); of V_prod β {row["prod"].get("beta")} (R² {row["prod"]["r2"]:.3f}); '
        f'true: sum {res["models"]["sum"]["gev"]["beta"]:.2f}, prod {res["models"]["prod"]["gev"]["beta"]:.2f}')

# ================================================================ 4. carriers S0, S50 per tertile (raw units, ratios to their pT×c response)
res['carriers'] = {}
for u in (0, 50, 29):
    Lc = {ck: RC[ck]['S'][:, u].astype(np.float64) for ck in CELLS}
    res['carriers'][f'S{u}'] = T_ratios(Lc)
    T = res['carriers'][f'S{u}']
    log(f'carrier S{u}: gain ' + ' '.join(f'{T[t]["gain_c"]:.3f}' for t in TN) + ' | M2-only ÷ pT×c ' + ' '.join(f'{T[t]["k_over_c"]:.3f}' for t in TN)
        + ' | M0-only ÷ pT×c ' + ' '.join(f'{T[t]["c0_over_c"]:.3f}' for t in TN))
OUT_JSON.write_text(json.dumps(res))

# ================================================================ 2. calibration control: known combination rules
res['calibration'] = {}
z_ = lambda v, v0: (v - v0.mean()) / v0.std()


def win1(V, rng_, n=64):
    v0 = V[BASE]; s = rng_.choice([-1.0, 1.0], n); q = np.quantile(z_(v0, v0), rng_.uniform(0.05, 0.95, n))
    return {ck: np.maximum(s[None] * (z_(V[ck], v0)[:, None] - q[None]), 0.0) for ck in CELLS}


def win2(A, B, rng_, n=64):
    a0, b0 = A[BASE], B[BASE]; th = rng_.uniform(0, 2 * np.pi, n); ca, sb = np.cos(th), np.sin(th)
    p0 = z_(a0, a0)[:, None] * ca + z_(b0, b0)[:, None] * sb
    q = np.array([np.quantile(p0[:, i], qq) for i, qq in enumerate(rng_.uniform(0.05, 0.95, n))])
    return {ck: np.maximum(z_(A[ck], a0)[:, None] * ca + z_(B[ck], b0)[:, None] * sb - q[None], 0.0) for ck in CELLS}


M0c = {ck: MOM[ck]['M0'] for ck in CELLS}; M2c = {ck: MOM[ck]['M2'] for ck in CELLS}
for seed_ in (0, 1, 2):
    rg = np.random.default_rng(100 + seed_)
    pops = {'sumwin': win1(V_SUM, rg), 'prodwin': win1(V_PROD, rg), 'sepwin': win2(M0c, M2c, rg),
            'logsepwin': win2({k: np.log(v) for k, v in M0c.items()}, {k: np.log(v) for k, v in M2c.items()}, rg)}
    if seed_ == 0: pops = {'lin': {ck: np.c_[M0c[ck], M2c[ck]] for ck in CELLS}, **pops}
    for nm, F in pops.items():
        res['calibration'].setdefault(nm, {})[str(seed_)] = readout_suite(F, f'cal {nm} s{seed_}', unit_stats=(seed_ == 0 and nm != 'lin'))
    del pops


def rand_layer(F, n_out, rng_):
    X0 = np.asarray(F[BASE], np.float64); mu, sd = X0.mean(0), X0.std(0); live = sd > 1e-6
    W = rng_.normal(0, 1, (int(live.sum()), n_out)) / np.sqrt(live.sum())
    pre0 = ((X0[:, live] - mu[live]) / sd[live]) @ W; b = -np.array([np.quantile(pre0[:, i], q) for i, q in enumerate(rng_.uniform(0.2, 0.8, n_out))])
    return {ck: np.maximum(((np.asarray(F[ck], np.float64)[:, live] - mu[live]) / sd[live]) @ W + b, 0.0) for ck in F}


FS = {ck: RC[ck]['S'] for ck in CELLS}
for seed_ in (0, 1, 2):
    rg = np.random.default_rng(200 + seed_)
    L1 = rand_layer(FS, 64, rg)
    res['calibration'].setdefault('rand1_on_S', {})[str(seed_)] = readout_suite(L1, f'rand1 on S s{seed_}', unit_stats=(seed_ == 0))
    L2 = rand_layer(L1, 32, rg)
    res['calibration'].setdefault('rand2_on_S', {})[str(seed_)] = readout_suite(L2, f'rand2 on S s{seed_}')
    L3 = rand_layer({ck: RC[ck]['h4'] for ck in CELLS}, 32, rg)
    res['calibration'].setdefault('rand1_on_h4', {})[str(seed_)] = readout_suite(L3, f'rand1 on h4 s{seed_}')
    del L1, L2, L3
OUT_JSON.write_text(json.dumps(res))

# ================================================================ 5. B: S0 patch at matched S29/S50, S19 control, S0 × f
S = R0['S']
lS = lambda col: np.log(np.maximum(S[:, col], 0.1))
lS0, lS29, lS50 = lS(0), lS(29), lS(50)


def pick_donors(match, tol, far_key, far, okj, rng_, chunk=500):
    D = np.full(J, -1)
    for i0 in range(0, J, chunk):
        sl = slice(i0, min(i0 + chunk, J)); m_ = np.abs(far_key[sl, None] - far_key[None, :]) >= far
        for kv in match: m_ &= np.abs(kv[sl, None] - kv[None, :]) < tol
        m_ &= okj[None, :] & okj[sl, None]
        m_[np.arange(m_.shape[0]), np.arange(i0, i0 + m_.shape[0])] = False
        r_ = rng_.random(m_.shape) * m_
        D[sl] = np.where(m_.any(1), r_.argmax(1), -1)
    return D


D = pick_donors([lS29, lS50], 0.05, lS0, 0.25, S[:, 0] > 0, np.random.default_rng(SEED + 2)); hm = D >= 0
dl = np.where(hm, lS0[np.maximum(D, 0)] - lS0, 0.0)
res['B'] = dict(donor_frac=float(hm.mean()), dlogS0_abs_median=float(np.median(np.abs(dl[hm]))))
RD = {st: Ridge(R0[st], M['logm']) for st in ('h4', 'h5', 'z')}
LA = {st: {ck: RD[st](RC[ck][st]) for ck in CELLS} for st in RD}
TR = {st: T_ratios(LA[st]) for st in RD}


def patch_run(unit, values):
    Sp = S.copy(); Sp[:, unit] = values; m_ = np.zeros(64); m_[unit] = 1
    return net.run_jetwise(x, {'S': (m_, Sp)}, keep=KEEP, batch=2500)


def slopes(Rp, xv, has, dlv=None):
    out = {}
    for st in RD:
        dy = RD[st](Rp[st]) - RD[st](R0[st]); row = {}
        for ti, tn in enumerate(TN):
            s_ = has & is_te & (TERT == ti); xs_, ys_ = xv[s_], dy[s_]
            sl_ = float((xs_ @ ys_) / (xs_ @ xs_)); w = WB[:, s_].astype(np.float64)
            se = float(((w @ (xs_ * ys_)) / np.maximum(w @ (xs_ * xs_), 1e-30)).std())
            g = TR[st][tn]['gain_c']; c0 = TR[st][tn]['c0_over_c'] * g
            row[tn] = dict(n=int(s_.sum()), slope=sl_, slope_se=se, slope_over_gain=sl_ / g, S0_share_of_M0_path=sl_ / (2 * c0))
        row['high_over_low_over_gain'] = row['high']['slope_over_gain'] / row['low']['slope_over_gain']
        out[st] = row
    c1 = Rp['logits'].argmax(1); mid = TERT == 1
    out['calls_changed'] = float((c1[has] != call0[has]).mean()); out['acc_after'] = float((c1[has] == y[has]).mean())
    wj = has & mid & (y == W_) & (call0 == W_); out['mid_W_to_Z_dlogS0_ge_0.5'] = out['mid_W_to_Z_dlogS0_0.25_0.5'] = None
    if dlv is not None:
        a_, b_ = wj & (dlv >= 0.5), wj & (dlv >= 0.25) & (dlv < 0.5)
        out['mid_W_to_Z_dlogS0_ge_0.5'] = float((c1[a_] == Z_).mean()) if a_.sum() >= 20 else None
        out['mid_W_to_Z_dlogS0_0.25_0.5'] = float((c1[b_] == Z_).mean()) if b_.sum() >= 20 else None
    return out


xB = 0.5 * dl
Rp = patch_run(0, np.where(hm, S[np.maximum(D, 0), 0], S[:, 0])); res['B']['S0_matched'] = slopes(Rp, xB, hm, dl); del Rp
Rp = patch_run(19, np.where(hm, S[np.maximum(D, 0), 19], S[:, 19])); res['B']['S19_control'] = slopes(Rp, xB, hm, dl); del Rp
allj = np.ones(J, bool)
for f_ in (0.8, 1.25):
    Rp = patch_run(0, S[:, 0] * f_); res['B'][f'S0_x{f_}'] = slopes(Rp, np.full(J, 0.5 * np.log(f_)), allj); del Rp
for k_, v_ in res['B'].items():
    if isinstance(v_, dict):
        log(f'B {k_}: calls changed {v_["calls_changed"]:.3f}, acc after {v_["acc_after"]:.3f}; W→Z mid ≥0.5 {v_["mid_W_to_Z_dlogS0_ge_0.5"]}, 0.25–0.5 {v_["mid_W_to_Z_dlogS0_0.25_0.5"]}')
        for st in RD:
            r_ = v_[st]
            log(f'   {st}: slope low/mid/high ' + ' '.join(f'{r_[t]["slope"]:+.3f}±{r_[t]["slope_se"]:.3f}' for t in TN)
                + ' | ÷ gain ' + ' '.join(f'{r_[t]["slope_over_gain"]:.3f}' for t in TN) + f' (high/low {r_["high_over_low_over_gain"]:.3f})'
                + ' | S0 share of M0 path ' + ' '.join(f'{r_[t]["S0_share_of_M0_path"]:.3f}' for t in TN))
res['wall_s'] = float(time.time() - t0)
OUT_JSON.write_text(json.dumps(res))
log(f'done -> {OUT_JSON}')
