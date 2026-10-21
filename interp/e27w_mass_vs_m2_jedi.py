"""E27w (claims A4w / A4; round 13): separate M2 = sum pT r^2 (about the stored axis) from the jet mass m, and fit the power of r
in the pT-weighted spread that JEDI-linear's W/Z output follows.

Notation (per jet; sums over real particles, r_i, theta_i = polar coordinates of (detarel_i, dphirel_i) about the stored axis):
  M0 = sum pT, M1 = sum pT (deta, dphi) (vector; |M1| = pT-weighted centroid offset x M0), M2 = sum pT r^2, m^2 ~ M0 M2 - |M1|^2.
  lambda_beta = sum pT r^beta; count-weighted C1 = sum r.
Network: official JEDI-linear 3-feature n64, exact float forward pass (interp.jedi_core.Jedi).
Samples: fit = sample(per_class=1000, seed=2701) (all edits); test = sample(1000, 2702) minus jets also in the fit sample (arm C
  held-out scoring only).
Read-outs: ridge of log m on h4 (and h5) of unedited fit jets (half A fit, alpha on a third of A), evaluated on half B; W-Z logit
  margin d = logit_Z - logit_W (on jets called W, or called Z, before the edit); calls; W->Z / Z->W flow (share of W- / Z-called
  jets called Z / W after the edit).

Arm B (azimuth-only edit, m down at fixed M2): a subset of the jet's particles (variant 'soft' = the lower-pT half of the real
  particles, ranks >= ceil(n/2); variant 'hard14' = pT ranks 0..13) is turned about the axis toward the direction of M1:
  theta_i -> theta_i + t * wrap(phi_M1 - theta_i), t in [0, 1] per jet solved (grid + bisection) so the true mass (four-vector
  sum of the edited particles) drops by f = 2, 5, 10 %. Every particle keeps pT and r exactly, so M0, M2, every lambda_beta,
  C1, pT ranks and slots are unchanged; |M1| grows.
  Control 'ctrl' (M1 held): the same particles are turned by the same azimuth amounts |dtheta_i|, each clockwise or anticlockwise
  at random, then a minimum-norm azimuth correction on the same particles restores the vector M1 (|dM1| <= 1e-5 M0), alternated
  with a common rescale so the displacement D = sqrt(sum_i |dx_i|^2) (count-weighted, rad) equals the B edit's (within 2 %).
  Same particles, similar turn sizes and D, every radial quantity fixed, M1 and so m fixed (true mass within 10 % of B's change).
  Calibration dil_m: all angles x a, a solved so the true mass after it equals the B edit's (same predicted Delta m; M2 changes).
  Freeze control (B_frz, ctrl_frz, dil_frz): the same inputs with the context vector g (all 64 units) and S31, S58 overridden by
  their unedited values (run_jetwise).
  R_m = mean response (B) / mean response (dil_m) on the same jets: response = Delta (h4 / h5 log-m read-out), or Delta margin.
  Thesis (network reads M2): R_m(B) ~ R_m(ctrl) ~ 0 and flows(B) ~ flows(ctrl). Jet-mass reading: R_m(B) ~ 1 and Z->W flow at the
  dil_m rate.
Arm A (reference, M2 changes at fixed m): rigid translation detarel += 0.06 rad of every real particle; dil_M2 = r x sqrt(M2'/M2)
  (same Delta M2 about the stored axis; m changes).
Arm C (power of r): smooth radial-profile edits r_i -> r_i exp(e (f(r_i) - lambda_J)) with f = sum_k w_k b_k(r), basis
  b = {r, r^2, log r, (r-0.2)+, (0.1-r)+} (each divided by its count-weighted sd over real particles), w a random unit vector
  (60 directions on the fit sample, 20 new ones on the test sample), f centred and scaled so its 99.5th-percentile |f| over real
  particles is 1, e = 0.2; lambda_J per jet holds M2 exactly (set 'H2') or the true mass exactly (set 'Hm', M2 then moves).
  Kept jet: max_i |e (f_i - lambda_J)| <= log 2. Target per edit: R_marg = mean Delta margin / mean Delta margin of the
  5 % M2 dilation (r x sqrt(1.05)) on the same kept W-called jets. Regressors per edit: X_beta = mean Delta log lambda_beta /
  log 1.05 (dilation: X_beta = beta / 2), beta in {0.5, 1, 1.5, 2, 3}, X_C1 (count-weighted sum r; girth = lambda_1 / M0 equals
  lambda_1 at fixed M0), X_m = 2 mean Delta log m / log 1.05, and X(beta) on a grid beta 0.25..3.5 for a single-power fit.
  Fits (no intercept; ridge alpha by leave-one-edit-out CV on the fit edits): M2; m; M2 + beta 1; beta family; family + C1; best
  single power. Scored by R^2 on the 40 test-sample edits (new jets, new shapes). Null: the same ridge on 5 regressors
  sum pT h(r) with h random smooth functions (12 Gaussian bumps in r, random weights), 200 draws.
Statistics: 95 % intervals from 1,000 Poisson jet-bootstrap resamples (both samples resampled for arm C, fits recomputed);
  sign-flip permutation (10,000) on per-jet paired differences; McNemar (exact) on paired call / flow indicators; Holm over the
  6 arm-B variants per test, and over the 6 terms of the family + C1 fit (Freedman-Lane permutation at fixed alpha).
Outputs: $AI_RESULTS/jedi/e27w_mass_vs_m2.json, e27w_example.npz.
  python -m interp.e27w_mass_vs_m2_jedi [--smoke 100]
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
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time(); W_, Z_ = 2, 3
HP = [29, 6, 40, 57, 3, 43, 30, 36, 46]      # E14: the nine l1 half-plane statements
RADS = [29, 1, 0, 19, 30]                   # radial (pT, r) hinge S objects
CTX = [31, 58]                               # context-gated S objects (count multipoles)
BETAS = (0.5, 1.0, 1.5, 2.0, 3.0); BGRID = np.round(np.arange(0.25, 3.501, 0.05), 2)
DOSES = (0.02, 0.05, 0.10); SUBSETS = ('soft', 'hard14')
EC = 0.2; LN2 = np.log(2.0); L105 = np.log(1.05)


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def mass_p(pt, r, th):
    eta, phi = r * np.cos(th), r * np.sin(th)
    E = (pt * np.cosh(eta)).sum(-1); px = (pt * np.cos(phi)).sum(-1); py = (pt * np.sin(phi)).sum(-1); pz = (pt * np.sinh(eta)).sum(-1)
    return np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0))


def M1v(pt, r, th): return np.stack([(pt * r * np.cos(th)).sum(-1), (pt * r * np.sin(th)).sum(-1)], -1)


def to_x(x, real, r, th):
    e = np.array(x, np.float64); e[..., 1] = np.where(real, r * np.cos(th), e[..., 1]); e[..., 2] = np.where(real, r * np.sin(th), e[..., 2])
    return e.astype(np.float32)


def disp(r, th, th2): return np.sqrt(((2 * r * np.sin((th2 - th) / 2)) ** 2).sum(-1))


# ------------------------------------------------------------------------------------------------ arm B solvers
def solve_B(pt, r, th, d, f):
    m0 = mass_p(pt, r, th); mT = (1 - f) * m0; ts = np.linspace(0, 1, 41)
    mg = np.stack([mass_p(pt, r, th + t * d) for t in ts])
    below = mg <= mT[None]; ok = below.any(0) & (m0 > 0); k = np.maximum(np.argmax(below, 0), 1)
    lo, hi = ts[k - 1], ts[k]
    for _ in range(30):
        mid = (lo + hi) / 2; go = mass_p(pt, r, th + mid[:, None] * d) > mT
        lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
    return np.where(ok, hi, 0.0), ok


def t_for_D(r, th, d, DB, it=30):
    ok = disp(r, th, th + d) >= DB; lo = np.zeros(len(r)); hi = np.ones(len(r))
    for _ in range(it):
        mid = (lo + hi) / 2; go = disp(r, th, th + mid[:, None] * d) < DB
        lo, hi = np.where(go, mid, lo), np.where(go, hi, mid)
    return (lo + hi) / 2, ok


def solve_ctrl(pt, r, th, sub, dB, DB, rng, iters=40):
    """control: the same particles turned by the same azimuth amounts |dtheta_i| as the B edit, each clockwise or anticlockwise
    at random, then a minimum-norm azimuth correction on the same particles so the vector M1 is unchanged, alternated with a
    common rescale so the displacement D equals the B edit's.  Radial quantities fixed; M1 (so |M1| and m) held."""
    M10 = M1v(pt, r, th); dl = dB * rng.choice((-1.0, 1.0), dB.shape) * sub
    for _ in range(iters):
        th2 = th + dl; res = M1v(pt, r, th2) - M10
        Jx, Jy = -pt * r * np.sin(th2) * sub, pt * r * np.cos(th2) * sub          # dM1 / d theta_i
        A = np.stack([Jx, Jy], 1); G = np.einsum('jan,jbn->jab', A, A) + 1e-18 * np.eye(2)
        dl = dl - np.einsum('jan,ja->jn', A, np.linalg.solve(G, res[..., None])[..., 0])
        D = disp(r, th, th + dl); dl = dl * np.where(D > 0, DB / np.maximum(D, 1e-15), 1.0)[:, None]
    for _ in range(10):   # final M1 polish (D within tolerance checked below)
        th2 = th + dl; res = M1v(pt, r, th2) - M10
        A = np.stack([-pt * r * np.sin(th2) * sub, pt * r * np.cos(th2) * sub], 1); G = np.einsum('jan,jbn->jab', A, A) + 1e-18 * np.eye(2)
        dl = dl - np.einsum('jan,ja->jn', A, np.linalg.solve(G, res[..., None])[..., 0])
    th2 = th + dl; M0 = pt.sum(1)
    ok = (np.hypot(*(M1v(pt, r, th2) - M10).T) <= 1e-5 * M0) & (np.abs(disp(r, th, th2) / np.maximum(DB, 1e-15) - 1) <= 0.02)
    return th2, ok


# ------------------------------------------------------------------------------------------------ arm C
def basis(r):
    rc = np.maximum(r, 1e-3)
    return np.stack([rc, rc ** 2, np.log(rc), np.maximum(rc - 0.2, 0), np.maximum(0.1 - rc, 0)], -1)


def make_f(w, r, real, sdb):
    """f(r) for every particle with the global normalisation computed on this sample"""
    B = basis(r) / sdb; f = B @ w; fr = f[real]; mu = fr.mean(); q = np.quantile(np.abs(fr - mu), 0.995)
    return (f - mu) / q


def edit_C(pt, r, th, real, f, hold):
    num = (pt * r ** 2 * np.exp(2 * EC * f) * real).sum(1); M2 = (pt * r ** 2).sum(1)
    lam = np.log(num / np.maximum(M2, 1e-30)) / (2 * EC)
    if hold == 'Hm':
        m0 = mass_p(pt, r, th)
        for _ in range(10):
            mm = mass_p(pt, r * np.exp(EC * (f - lam[:, None])), th); lam = lam + np.log(np.maximum(mm, 1e-9) / np.maximum(m0, 1e-9)) / EC
    dl = EC * (f - lam[:, None]) * real
    ok = np.isfinite(dl).all(1) & (np.abs(dl).max(1) <= LN2)
    return r * np.exp(dl), ok


def moments(pt, r, real, betas):
    return np.stack([(pt * r ** b * real).sum(1) for b in betas], 1)


BUMPC = np.linspace(0, 0.8, 12)


def bumpfeat(pt, r, real):
    """sum_i pT_i phi_b(r_i) for 12 Gaussian bumps (width 0.08) -> (J, 12); random smooth h = phi @ v"""
    return np.einsum('jn,jnb->jb', pt * real, np.exp(-((r[..., None] - BUMPC) / 0.08) ** 2))


# ------------------------------------------------------------------------------------------------ statistics
def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


def signflip_p(d, nperm=10000, seed=0):
    d = np.asarray(d, np.float64); d = d[np.isfinite(d)]; n = len(d)
    if n == 0: return 1.0
    obs = abs(d.mean()); rng = np.random.default_rng(seed); cnt = 0; done = 0
    while done < nperm:
        k = min(500, nperm - done); S = rng.choice((-1.0, 1.0), (k, n)); cnt += int((np.abs(S @ d / n) >= obs - 1e-15).sum()); done += k
    return (1 + cnt) / (1 + nperm)


def ridge(X, y, a):
    return np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ y)


ALPHAS = np.r_[0.0, 10.0 ** np.arange(-5, 2.01, 0.5)]


def cv_alpha(X, y):
    best = None
    for a in ALPHAS:
        e = 0.0
        for i in range(len(y)):
            m = np.arange(len(y)) != i
            try: c = ridge(X[m], y[m], a + 1e-12)
            except np.linalg.LinAlgError: e = np.inf; break
            e += (X[i] @ c - y[i]) ** 2
        if best is None or e < best[0]: best = (e, a)
    return best[1] + 1e-12, best[0] / len(y)


def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.x10_das_jedi import clean, holm, mass, RidgeR
    from interp.h12_boost_panel import mcnemar
    sm = a.smoke; pc = sm or 1000; NB = 200 if sm else 1000; NP = 1000 if sm else 10000
    NDIR_FIT, NDIR_TEST, NNULL = (8, 4, 20) if sm else (60, 20, 200)
    OUT.mkdir(parents=True, exist_ok=True); tag = 'e27w_mass_vs_m2' + ('_smoke' if sm else ''); JS = OUT / f'{tag}.json'
    R = dict(samples=dict(fit=f'sample(per_class={pc}, seed=2701)', test=f'sample(per_class={pc}, seed=2702) minus overlap'))
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    net = Jedi(64)

    def fwd(x, keep=('logits', 'h4', 'h5', 'P', 'g', 'S'), ov=None):
        return net.run_jetwise(x, ov, keep=keep, batch=1000) if ov else net.run(x, keep=keep, batch=1000)

    # ---------------------------------------------------------------- samples and base pass
    x, y, _ = sample(pc, 2701); J = len(y)
    xt, yt, _ = sample(pc, 2702)
    hf = {hash(v.tobytes()) for v in x}; keep_t = np.array([hash(v.tobytes()) not in hf for v in xt]); xt, yt = xt[keep_t], yt[keep_t]
    R['N_fit'] = J; R['N_test'] = int(len(yt)); R['test_overlap_dropped'] = int((~keep_t).sum())
    x64 = np.asarray(x, np.float64); pt = x64[..., 0]; real = pt > 0; r = np.hypot(x64[..., 1], x64[..., 2]); th = np.arctan2(x64[..., 2], x64[..., 1])
    B0 = fwd(x); c0 = B0['logits'].argmax(1); marg0 = B0['logits'][:, Z_] - B0['logits'][:, W_]
    R['acc_fit'] = float((c0 == y).mean())
    log(f'fit sample N {J}: accuracy {R["acc_fit"]:.4f} (expected 0.806-0.813); test N {len(yt)} ({R["test_overlap_dropped"]} overlapping dropped)')
    chk = fwd(x, ov={'g': (np.ones(64), B0['g']), 'S': (np.isin(np.arange(64), CTX).astype(float), B0['S'])})
    R['freeze_identity_maxdiff'] = float(np.abs(chk['logits'] - B0['logits']).max()); del chk
    logm0 = np.log(np.maximum(mass(x), 1e-3)); pr = np.random.default_rng(2704).permutation(J); hA, hB = np.sort(pr[:J // 2]), np.sort(pr[J // 2:])
    nf = 2 * len(hA) // 3; RO = {st: RidgeR(B0[st], logm0, hA[:nf], hA[nf:]) for st in ('h4', 'h5')}
    R['readout_r2_heldout'] = {st: float(1 - ((RO[st](B0[st][hB]) - logm0[hB]) ** 2).mean() / logm0[hB].var()) for st in RO}
    ro0 = {st: RO[st](B0[st]) for st in RO}; isB = np.zeros(J, bool); isB[hB] = True
    sdP = B0['P'].std(0) + 1e-12; sdS = B0['S'].std(0) + 1e-12; sdg = B0['g'].std(0); liveg = sdg > 1e-9
    log(f'freeze identity max |dlogit| {R["freeze_identity_maxdiff"]:.2e}; read-out held-out R2 {R["readout_r2_heldout"]}'); save()
    Wb = np.random.default_rng(2705).poisson(1.0, (NB, J)).astype(np.float64)

    def resp(P):
        ce = P['logits'].argmax(1)
        o = dict(call=ce, ch=ce != c0, wz=(c0 == W_) & (ce == Z_), zw=(c0 == Z_) & (ce == W_),
                 dmarg=P['logits'][:, Z_] - P['logits'][:, W_] - marg0, **{f'd{st}': RO[st](P[st]) - ro0[st] for st in RO})
        if 'P' in P:
            o['drive'] = dict(HP_dP=float(np.mean(np.abs(P['P'][:, HP] - B0['P'][:, HP]) / sdP[HP])),
                              S31=float(np.mean(np.abs(P['S'][:, 31] - B0['S'][:, 31]) / sdS[31])), S58=float(np.mean(np.abs(P['S'][:, 58] - B0['S'][:, 58]) / sdS[58])),
                              S_radial=float(np.mean(np.abs(P['S'][:, RADS] - B0['S'][:, RADS]) / sdS[RADS])),
                              g_live=float(np.mean(np.abs(P['g'][:, liveg] - B0['g'][:, liveg]) / sdg[liveg])))
        return o

    def bm(w, v): return (w @ v) / np.maximum(w.sum(1), 1e-12)

    def ratio(v, vr, m):
        if m.sum() == 0: return None, [None, None]
        w = Wb[:, m]; return float(v[m].mean() / vr[m].mean()), qci(bm(w, v[m]) / bm(w, vr[m]))

    def share(v, m):
        if m.sum() == 0: return None, [None, None]
        return float(v[m].mean()), qci(bm(Wb[:, m], v[m].astype(float)))

    def block(E, ref, ks, name):
        """response metrics of edit E relative to reference ref on kept jets ks"""
        o = dict(n=int(ks.sum()), n_W=int((ks & (c0 == W_)).sum()), n_Z=int((ks & (c0 == Z_)).sum()))
        for st in RO: o[f'R_{st}'], o[f'R_{st}_ci'] = ratio(E[f'd{st}'], ref[f'd{st}'], ks & isB)
        o['R_marg_W'], o['R_marg_W_ci'] = ratio(E['dmarg'], ref['dmarg'], ks & (c0 == W_))
        o['R_marg_Z'], o['R_marg_Z_ci'] = ratio(E['dmarg'], ref['dmarg'], ks & (c0 == Z_))
        o['calls_changed'], o['calls_changed_ci'] = share(E['ch'], ks)
        o['WZ'], o['WZ_ci'] = share(E['wz'], ks & (c0 == W_)); o['ZW'], o['ZW_ci'] = share(E['zw'], ks & (c0 == Z_))
        if 'drive' in E: o['drive'] = E['drive']
        return o

    # ---------------------------------------------------------------- arm A: rigid translation vs M2-matched dilation
    t = time.time(); xa = np.array(x, np.float32); xa[..., 1] += np.float32(0.06) * real
    M2 = (pt * r ** 2).sum(1); M2a = (pt * ((x64[..., 1] + 0.06) ** 2 + x64[..., 2] ** 2) * real).sum(1); aA = np.sqrt(M2a / M2)
    xd = to_x(x, real, r * aA[:, None], th)
    EA, ED = resp(fwd(xa)), resp(fwd(xd)); allk = np.ones(J, bool)
    R['armA'] = dict(edit='detarel + 0.06 rad, every real particle', median_dlogM2=float(np.median(np.log(M2a / M2))),
                     median_dlogm_true=float(np.median(np.log(mass(xa) / mass(x)))), dil_median_dlogm_true=float(np.median(np.log(mass(xd) / mass(x)))),
                     translation=block(EA, ED, allk, 'A'), dil_M2=block(ED, ED, allk, 'dil'),
                     mcnemar_WZ_vs_dil=mcnemar(EA['wz'][c0 == W_], ED['wz'][c0 == W_]))
    log(f'arm A ({time.time() - t:.0f} s): translation W->Z {R["armA"]["translation"]["WZ"]:.3f}, dil_M2 W->Z {R["armA"]["dil_M2"]["WZ"]:.3f}, '
        f'R_h4 {R["armA"]["translation"]["R_h4"]:+.3f}, R_marg_W {R["armA"]["translation"]["R_marg_W"]:+.3f}; median dlogm true {R["armA"]["median_dlogm_true"]:+.2e}'); save()

    # ---------------------------------------------------------------- arm B: azimuth-only edits
    rank = np.argsort(np.argsort(-np.where(real, pt, -1), 1), 1); nreal = real.sum(1)
    SUB = dict(soft=real & (rank >= np.ceil(nreal / 2)[:, None]), hard14=real & (rank < 14))
    M10 = M1v(pt, r, th); phu = np.arctan2(M10[:, 1], M10[:, 0]); m0p = mass_p(pt, r, th)
    R['M1_over_M0_median'] = float(np.median(np.hypot(*M10.T) / pt.sum(1))); R['armB'] = {}; EX = {}
    pvals = {k: [] for k in ('ro_B_vs_ctrl', 'marg_B_vs_ctrl', 'ZW_B_vs_ctrl', 'calls_B_vs_ctrl', 'ZW_B_vs_dil', 'ro_B_vs_dil')}; vnames = []
    for sname in SUBSETS:
        sub = SUB[sname]
        for f in DOSES:
            t = time.time(); vn = f'{sname}_{int(round(f * 100))}'; vnames.append(vn)
            d = np.where(sub, wrap(phu[:, None] - th), 0.0); tB, okB = solve_B(pt, r, th, d, f); thB = th + tB[:, None] * d
            DB = disp(r, th, thB); thC, okC = solve_ctrl(pt, r, th, sub, tB[:, None] * d, DB, np.random.default_rng(27000 + int(f * 100) + 50 * (sname == 'soft')))
            xB, xC = to_x(x, real, r, thB), to_x(x, real, r, thC)
            mB, mC = mass(xB), mass(xC); mx0 = mass(x)
            aD = mB / np.maximum(mx0, 1e-9)
            for _ in range(4):
                xDm = to_x(x, real, r * aD[:, None], th); aD = aD * mB / np.maximum(mass(xDm), 1e-9)
            xDm = to_x(x, real, r * aD[:, None], th)
            dlB = np.log(mB / mx0); dlC = np.log(mC / mx0); dlD = np.log(mass(xDm) / mx0)
            ks = okB & okC & (np.abs(dlC) <= 0.1 * np.abs(np.log(1 - f))) & np.isfinite(dlB)
            # exactness checks on kept jets (float32 inputs as the network sees them)
            chkr = lambda xe: float(np.max(np.abs(np.hypot(xe[..., 1].astype(np.float64), xe[..., 2]) - r)[ks])) if ks.any() else None
            M1n = lambda xe: np.hypot((pt * xe[..., 1]).sum(1), (pt * xe[..., 2]).sum(1))
            frz = {'g': (np.ones(64), B0['g']), 'S': (np.isin(np.arange(64), CTX).astype(float), B0['S'])}
            E = dict(B=resp(fwd(xB)), ctrl=resp(fwd(xC)), dil_m=resp(fwd(xDm)),
                     B_frz=resp(fwd(xB, ov=frz)), ctrl_frz=resp(fwd(xC, ov=frz)), dil_frz=resp(fwd(xDm, ov=frz)))
            o = dict(subset=sname, dose=f, kept=int(ks.sum()), kept_frac=float(ks.mean()), feasible_B=int(okB.sum()), feasible_ctrl=int((okB & okC).sum()),
                     kept_by_class=np.bincount(y[ks], minlength=5).tolist(),
                     checks=dict(max_abs_dr_B=chkr(xB), max_abs_dr_ctrl=chkr(xC),
                                 median_dlogm_true=dict(B=float(np.median(dlB[ks])) if ks.any() else None, ctrl=float(np.median(dlC[ks])) if ks.any() else None,
                                                        dil_m=float(np.median(dlD[ks])) if ks.any() else None),
                                 max_abs_dlogm_dil_minus_B=float(np.max(np.abs(dlD - dlB)[ks])) if ks.any() else None,
                                 median_M1_over_M0=dict(before=float(np.median((M1n(x) / pt.sum(1))[ks])) if ks.any() else None,
                                                        B=float(np.median((M1n(xB) / pt.sum(1))[ks])) if ks.any() else None,
                                                        ctrl=float(np.median((M1n(xC) / pt.sum(1))[ks])) if ks.any() else None),
                                 median_t_B=float(np.median(tB[ks])) if ks.any() else None, median_D_rad=float(np.median(DB[ks])) if ks.any() else None,
                                 median_n_moved=float(np.median(sub[ks].sum(1))) if ks.any() else None))
            for k_ in E: o[k_] = block(E[k_], E['dil_m'], ks, k_)
            # paired tests
            mB_, mZ = ks & isB, ks & ((c0 == W_) | (c0 == Z_)); dR = E['dil_m']['dh4']
            w = Wb[:, mB_]
            o['diff_R_h4_B_minus_ctrl'] = float((E['B']['dh4'][mB_] - E['ctrl']['dh4'][mB_]).mean() / dR[mB_].mean()) if mB_.any() else None
            o['diff_R_h4_B_minus_ctrl_ci'] = qci((bm(w, E['B']['dh4'][mB_]) - bm(w, E['ctrl']['dh4'][mB_])) / bm(w, dR[mB_])) if mB_.any() else None
            tests = dict(ro_B_vs_ctrl=signflip_p(E['B']['dh4'][mB_] - E['ctrl']['dh4'][mB_], NP, 1),
                         ro_B_vs_dil=signflip_p(E['B']['dh4'][mB_] - dR[mB_], NP, 2),
                         marg_B_vs_ctrl=signflip_p(E['B']['dmarg'][mZ] - E['ctrl']['dmarg'][mZ], NP, 3),
                         ZW_B_vs_ctrl=mcnemar(E['B']['zw'][ks & (c0 == Z_)], E['ctrl']['zw'][ks & (c0 == Z_)]),
                         ZW_B_vs_dil=mcnemar(E['B']['zw'][ks & (c0 == Z_)], E['dil_m']['zw'][ks & (c0 == Z_)]),
                         calls_B_vs_ctrl=mcnemar(E['B']['ch'][ks], E['ctrl']['ch'][ks]),
                         calls_B_vs_dil=mcnemar(E['B']['ch'][ks], E['dil_m']['ch'][ks]))
            o['tests'] = tests
            for k_ in pvals: pvals[k_].append(tests[k_]['p'] if isinstance(tests[k_], dict) else tests[k_])
            # 5x5 call flow for B, ctrl, dil_m
            o['flow'] = {k_: np.array([[int(((c0 == i) & (E[k_]['call'] == j) & ks).sum()) for j in range(5)] for i in range(5)]).tolist() for k_ in ('B', 'ctrl', 'dil_m')}
            R['armB'][vn] = o
            if sname == 'soft' and f == 0.05:
                cand = np.flatnonzero(ks & (y == W_) & (c0 == W_))
                if len(cand) == 0: cand = np.flatnonzero(ks)
                if len(cand):
                    ex = int(cand[np.argsort(nreal[cand])[len(cand) // 2]])
                    EX = dict(index=ex, x=x[ex], xB=xB[ex], xC=xC[ex], moved=sub[ex], label=int(y[ex]), call0=int(c0[ex]), callB=int(E['B']['call'][ex]),
                              callC=int(E['ctrl']['call'][ex]), callD=int(E['dil_m']['call'][ex]), m0=float(mx0[ex]), mB=float(mB[ex]), mC=float(mC[ex]),
                              M1_0=float(M1n(x)[ex] / pt[ex].sum()), M1_B=float(M1n(xB)[ex] / pt[ex].sum()), M2=float(M2[ex]), M2_B=float((pt[ex] * (xB[ex, :, 1].astype(float) ** 2 + xB[ex, :, 2].astype(float) ** 2)).sum()),
                              ptJ=float(pt[ex].sum()), dh4B=float(E['B']['dh4'][ex]), dh4D=float(E['dil_m']['dh4'][ex]))
                    R['example'] = {k_: v for k_, v in EX.items() if np.ndim(v) == 0}
            b, c, dd = o['B'], o['ctrl'], o['dil_m']
            log(f'arm B {vn} ({time.time() - t:.0f} s): kept {ks.sum()} / {J} (feasible B {okB.sum()}), dlogm B {o["checks"]["median_dlogm_true"]["B"]}, ctrl {o["checks"]["median_dlogm_true"]["ctrl"]}, '
                f'max|dr| {o["checks"]["max_abs_dr_B"]}; R_h4 B {b["R_h4"]} ctrl {c["R_h4"]} Bfrz {o["B_frz"]["R_h4"]}; R_marg_Z B {b["R_marg_Z"]} ctrl {c["R_marg_Z"]}; '
                f'Z->W B {b["ZW"]} ctrl {c["ZW"]} dil {dd["ZW"]}; calls B {b["calls_changed"]} ctrl {c["calls_changed"]} dil {dd["calls_changed"]}')
            save()
    R['armB_holm'] = {k_: dict(zip(vnames, holm(v).tolist())) for k_, v in pvals.items()}
    if EX: np.savez_compressed(OUT / 'e27w_example.npz', **{k_: np.asarray(v) for k_, v in EX.items()})
    save()

    # ---------------------------------------------------------------- arm C: smooth radial-profile edits, power-of-r fit
    t = time.time()
    sdb = basis(r)[real].std(0)
    S = dict(fit=dict(x=x, y=y, c0=c0, marg0=marg0), test=dict(x=xt, y=yt))
    xt64 = np.asarray(xt, np.float64); S['test'].update(pt=xt64[..., 0], r=np.hypot(xt64[..., 1], xt64[..., 2]), th=np.arctan2(xt64[..., 2], xt64[..., 1]))
    S['fit'].update(pt=pt, r=r, th=th)
    lt = net.run(xt, keep=('logits',), batch=1000)['logits']; S['test']['c0'] = lt.argmax(1); S['test']['marg0'] = lt[:, Z_] - lt[:, W_]
    R['acc_test'] = float((S['test']['c0'] == yt).mean()); log(f'test sample accuracy {R["acc_test"]:.4f}')
    rngd = {'fit': np.random.default_rng(27013), 'test': np.random.default_rng(27023)}
    VN = np.random.default_rng(2708).standard_normal((NNULL, 12, 5))
    AC = {}
    for sn, nd in (('fit', NDIR_FIT), ('test', NDIR_TEST)):
        D_ = S[sn]; P_, r_, th_ = D_['pt'], D_['r'], D_['th']; re_ = P_ > 0; pop = D_['c0'] == W_
        Pp, rp = P_[pop], r_[pop]; rep = re_[pop]
        lam0 = moments(Pp, rp, rep, BGRID); c10 = (rp * rep).sum(1); m0_ = mass(D_['x'][pop])
        phi0 = bumpfeat(Pp, rp, rep)
        xdil = to_x(D_['x'], re_, r_ * np.sqrt(1.05), th_); ld = net.run(xdil, keep=('logits',), batch=1000)['logits']
        dmd = (ld[:, Z_] - ld[:, W_] - D_['marg0'])[pop]
        rows = dict(dmarg=[], keep=[], dlam=[], dc1=[], dm=[], dnul=[], set=[], w=[], calls=[])
        for k in range(nd):
            w = rngd[sn].standard_normal(5); w /= np.linalg.norm(w); f = make_f(w, r_, re_, sdb)
            for hold in ('H2', 'Hm'):
                r2_, ok = edit_C(P_, r_, th_, re_, f, hold); xe = to_x(D_['x'], re_, r2_, th_)
                lo = net.run(xe, keep=('logits',), batch=1000)['logits']; ce = lo.argmax(1)
                rows['calls'].append(float((ce != D_['c0'])[ok].mean()))
                rep2 = r2_[pop]
                rows['dmarg'].append((lo[:, Z_] - lo[:, W_] - D_['marg0'])[pop]); rows['keep'].append(ok[pop])
                rows['dlam'].append(np.log(moments(Pp, rep2, rep, BGRID) / lam0)); rows['dc1'].append(np.log((rep2 * rep).sum(1) / c10))
                rows['dm'].append(np.log(np.maximum(mass(xe[pop]), 1e-9) / m0_))
                kp = ok[pop]; rows['dnul'].append((bumpfeat(Pp, rep2, rep) - phi0)[kp].mean(0) if kp.any() else np.zeros(12))
                rows['set'].append(hold); rows['w'].append(w.tolist())
            if k % 10 == 0: log(f'arm C {sn} direction {k}/{nd} ({time.time() - t:.0f} s); kept H2 {rows["keep"][-2].mean():.3f}, Hm {rows["keep"][-1].mean():.3f}')
        AC[sn] = dict(dmd=dmd, **{k_: (np.nan_to_num(np.array(v)) if k_ not in ('set', 'w') else v) for k_, v in rows.items()})
    log(f'arm C passes done ({time.time() - t:.0f} s)')
    ib = {b: int(np.argmin(np.abs(BGRID - b))) for b in BETAS}

    def aggregate(sn, wts):
        """per edit: y = R_marg, X columns (betas family, C1, m, grid), null regressors; wts (Np,) bootstrap weights"""
        A = AC[sn]; K = A['keep'] * wts[None]; den = np.maximum(K.sum(1), 1e-12)
        yv = (K * A['dmarg']).sum(1) / (K * A['dmd'][None]).sum(1)
        mX = lambda Z: (K * Z).sum(-1) / den
        G = np.einsum('kj,kjb->kb', K, A['dlam']) / den[:, None] / L105           # (K, nb grid)
        C1 = mX(A['dc1']) / L105; Mm = 2 * mX(A['dm']) / L105
        return yv, G, C1, Mm, None

    MODELS = dict(M2=lambda G, C1, Mm: G[:, [ib[2.0]]], m=lambda G, C1, Mm: Mm[:, None], M2_beta1=lambda G, C1, Mm: G[:, [ib[2.0], ib[1.0]]],
                  family=lambda G, C1, Mm: G[:, [ib[b] for b in BETAS]], family_C1=lambda G, C1, Mm: np.c_[G[:, [ib[b] for b in BETAS]], C1])
    TERMS = [f'beta{b:g}' for b in BETAS] + ['C1_count_sum_r']
    one_f = np.ones(len(AC['fit']['dmd'])); one_t = np.ones(len(AC['test']['dmd']))
    yf, Gf, C1f, Mf, _ = aggregate('fit', one_f); yt_, Gt, C1t, Mt, _ = aggregate('test', one_t)
    Nf = np.einsum('kb,qbv->kqv', AC['fit']['dnul'], VN); Nt = np.einsum('kb,qbv->kqv', AC['test']['dnul'], VN)
    C = dict(n_edits_fit=len(yf), n_edits_test=len(yt_), kept_frac_fit=float(AC['fit']['keep'].mean()), kept_frac_test=float(AC['test']['keep'].mean()),
             n_W_called_fit=int(len(one_f)), n_W_called_test=int(len(one_t)), calls_changed_median=float(np.median(AC['fit']['calls'])),
             R_marg_range_fit=dict(H2=[float(yf[::2].min()), float(yf[::2].max())], Hm=[float(yf[1::2].min()), float(yf[1::2].max())]),
             median_abs_X=dict(**{f'beta{b:g}': float(np.median(np.abs(Gf[:, ib[b]]))) for b in BETAS}, C1=float(np.median(np.abs(C1f))), m=float(np.median(np.abs(Mf)))),
             models={})
    alph = {}
    for nm, fn in MODELS.items():
        Xf, Xt = fn(Gf, C1f, Mf), fn(Gt, C1t, Mt); al, cve = cv_alpha(Xf, yf); co = ridge(Xf, yf, al); alph[nm] = al
        C['models'][nm] = dict(alpha=al, cv_mse=cve, coef=co.tolist(), r2_fit=r2(yf, Xf @ co), r2_test=r2(yt_, Xt @ co))
    # single power scan (choose beta by CV on fit edits)
    cvs = []
    for gi in range(len(BGRID)):
        Xf = Gf[:, [gi]]; co = ridge(Xf, yf, 1e-12); cvs.append(((yf - Xf @ co) ** 2).mean())
    gb = int(np.argmin(cvs)); co = ridge(Gf[:, [gb]], yf, 1e-12)
    C['models']['single_power'] = dict(beta=float(BGRID[gb]), coef=co.tolist(), r2_fit=r2(yf, Gf[:, [gb]] @ co), r2_test=r2(yt_, Gt[:, [gb]] @ co),
                                       mse_by_beta=dict(zip([f'{b:g}' for b in BGRID], [float(v) for v in cvs])))
    # shape-matched null: 5 random smooth radial sums
    nr2 = []
    for q in range(NNULL):
        Xf, Xt = Nf[:, q], Nt[:, q]; sc = np.abs(Xf).std(0) + 1e-30; al, _ = cv_alpha(Xf / sc, yf); co = ridge(Xf / sc, yf, al); nr2.append(r2(yt_, (Xt / sc) @ co))
    nr2 = np.array(nr2); C['null_random_smooth'] = dict(r2_test_median=float(np.median(nr2)), r2_test_q95=float(np.quantile(nr2, .95)), r2_test_max=float(nr2.max()),
                                                        p_family_C1=float((1 + (nr2 >= C['models']['family_C1']['r2_test']).sum()) / (1 + len(nr2))),
                                                        p_M2=float((1 + (nr2 >= C['models']['M2']['r2_test']).sum()) / (1 + len(nr2))))
    log(f'arm C fits: {json.dumps(clean({k_: (v.get("r2_test"), v.get("coef")) for k_, v in C["models"].items()}))}; null R2 median {np.median(nr2):.3f}'); save()
    # bootstrap over jets (both samples)
    rb = np.random.default_rng(2706); BT = {nm: dict(coef=[], r2_test=[]) for nm in list(MODELS) + ['single_power']}
    for _ in range(NB):
        wf = rb.poisson(1.0, len(one_f)).astype(float); wt = rb.poisson(1.0, len(one_t)).astype(float)
        a1 = aggregate('fit', wf); a2 = aggregate('test', wt)
        for nm, fn in MODELS.items():
            Xf, Xt = fn(*a1[1:4]), fn(*a2[1:4]); co = ridge(Xf, a1[0], alph[nm]); BT[nm]['coef'].append(co); BT[nm]['r2_test'].append(r2(a2[0], Xt @ co))
        cv_ = [((a1[0] - a1[1][:, [gi]] @ ridge(a1[1][:, [gi]], a1[0], 1e-12)) ** 2).mean() for gi in range(len(BGRID))]; g_ = int(np.argmin(cv_))
        co = ridge(a1[1][:, [g_]], a1[0], 1e-12); BT['single_power']['coef'].append([BGRID[g_], co[0]]); BT['single_power']['r2_test'].append(r2(a2[0], a2[1][:, [g_]] @ co))
    for nm in BT:
        cb = np.array(BT[nm]['coef']); C['models'][nm]['coef_ci'] = [qci(cb[:, i]) for i in range(cb.shape[1])]
        C['models'][nm]['r2_test_ci'] = qci(BT[nm]['r2_test'])
        if nm != 'single_power':
            C['models'][nm]['coef_boot_p'] = [float(min(1.0, 2 * min((cb[:, i] <= 0).mean(), (cb[:, i] >= 0).mean()) + 1 / len(cb))) for i in range(cb.shape[1])]
    rb2 = np.array(BT['family_C1']['r2_test']); C['dr2_test_family_C1_minus_M2'] = qci(rb2 - np.array(BT['M2']['r2_test']))
    C['dr2_test_M2_beta1_minus_M2'] = qci(np.array(BT['M2_beta1']['r2_test']) - np.array(BT['M2']['r2_test']))
    C['dr2_test_M2_minus_m'] = qci(np.array(BT['M2']['r2_test']) - np.array(BT['m']['r2_test']))
    # Freedman-Lane permutation per term of family + C1 at fixed alpha, Holm over the 6 terms
    Xf = MODELS['family_C1'](Gf, C1f, Mf); al = alph['family_C1']; cfull = ridge(Xf, yf, al); prng = np.random.default_rng(2707); pfl = []
    for i in range(Xf.shape[1]):
        Xr = np.delete(Xf, i, 1); cr = ridge(Xr, yf, al); fit_r = Xr @ cr; res = yf - fit_r; cnt = 0; npm = 2000 if not sm else 200
        for _ in range(npm):
            yp = fit_r + prng.permutation(res); cnt += abs(ridge(Xf, yp, al)[i]) >= abs(cfull[i])
        pfl.append((1 + cnt) / (1 + npm))
    C['models']['family_C1']['freedman_lane_p'] = dict(zip(TERMS, pfl)); C['models']['family_C1']['freedman_lane_holm'] = dict(zip(TERMS, holm(pfl).tolist()))
    C['edits'] = dict(fit=dict(set=AC['fit']['set'], w=AC['fit']['w'], R_marg=yf.tolist(), X_beta2=Gf[:, ib[2.0]].tolist(), X_beta1=Gf[:, ib[1.0]].tolist(), X_m=Mf.tolist()),
                      test=dict(set=AC['test']['set'], R_marg=yt_.tolist(), X_beta2=Gt[:, ib[2.0]].tolist(), X_m=Mt.tolist()))
    R['armC'] = C; R['seconds'] = time.time() - T0; save()
    log(f'arm C bootstrap done; models {json.dumps(clean({k_: (v.get("r2_test"), v.get("r2_test_ci")) for k_, v in C["models"].items()}))}; done, wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0)
    main(ap.parse_args())
