"""E41 (claims A4 / A4o / A9; round 20, user question "do later layers compute the true invariant mass?"): does any
JEDI-linear stage carry the true invariant mass m separately from the axis-moment mass m_A = sqrt(M0 M2), and does the
decision use it?

PLAN (fixed before the first run)
Definitions (per jet; sums over real particles; (deta_i, dphi_i) about the stored jet axis, r_i = hypot(deta_i, dphi_i)):
  m    = invariant mass of the summed massless constituent four-vectors (exact, interp.x10_das_jedi.mass);
  M0   = sum pT_i;  M1 = sum pT_i (deta_i, dphi_i) (vector);  M2 = sum pT_i r_i^2;
  m_A  = sqrt(M0 M2)  (small-angle: m^2 ~ M0 M2 - |M1|^2, so m = m_A up to the centroid term |M1|^2 / (M0 M2); on natural
         jets m / m_A ~ 0.998);
  Delta = log m - log m_A  (the part of the true mass not contained in m_A; ~ 0.5 log(1 - |M1|^2/(M0 M2))).
Network: official JEDI-linear 3-feature n64, exact forward pass (interp.jedi_core.Jedi). Stages read out (10):
  a (per-particle l1 outputs: mean over real particles and pT-weighted mean, 128 dims), P, g, s and u (mean over real
  particles), S, h4, h5, z, logits.
Samples: fit = sample(per_class=1000, seed=4101); test = sample(1000, 4102) minus jets also in fit.  All five classes.
Edited pool (each jet contributes these versions; a version is kept where its edit is solvable):
  nat      the natural jet;
  shift3 / shift6  rigid translation of every real particle by 0.03 / 0.06 rad in a per-jet random choice of {+deta, -deta,
           +dphi, -dphi} (m exactly unchanged; M2 about the stored axis and so m_A rise; Delta falls);
  turn_<hard14|soft><2|5|10>  E27w arm B: the 14 hardest particles (or the softer half of the real particles) turned in
           azimuth about the axis toward the direction of M1, theta_i -> theta_i + t wrap(phi_M1 - theta_i), t solved per jet so
           that m drops 2 / 5 / 10 %; every particle keeps pT and r, so M0, M2 and m_A are fixed and Delta falls by the mass change;
  turnup   the 14 hardest particles turned away from M1 (toward -M1) by the t in [0, 1] (41-point grid) that maximises m
           (raises m at fixed m_A; on natural jets the gain is bounded by the small centroid term);
  dil<2|5|10>  all angles x a, a solved per jet so that m equals that jet's turn_hard14 mass (m and m_A move together, Delta
           fixed: the 'real mass change' reference).
  C2ST (classifier two-sample test): HistGradientBoosting, 5-fold out-of-fold AUC, natural jets (one random half) vs the edited
  versions of the other half, on axis-free features (log m, log pT_J, N, pT^D, 2-point energy correlators e2^(beta), beta 0.5 /
  1 / 2, leading pT fraction, dR of the two hardest) and on axis-relative features (|M1|^2/(M0 M2), |M1|/M0, girth sum pT r / M0,
  M2/M0); null = natural vs natural.  Families with axis-free AUC > 0.6 flagged.
Read-outs (per stage, targets log m, log m_A, Delta): ridge (alpha by 3-fold CV grouped by jet) and an MLP (1 hidden layer of 64
  ReLU units, sklearn MLPRegressor, early stopping; <= 25,000 training rows).  Fits:
  pool     fit on all fit-sample pool rows, scored on all test-sample pool rows (and on its turn / shift / nat+dil subsets);
  turnT    fit on nat + shift + dil rows (fit sample), scored on turn + turnup rows (test sample): can a read-out learned
           where Delta varies through m_A predict Delta when only m moves?
  revT     fit on turn + turnup rows, scored on nat + shift + dil rows.
  Raw-input baselines: the same read-outs on the 64 x 3 input (pT-sorted, flattened) and on (log M0, log |M1|, log M2).
  Read-out response to the turn (calibrated): R_m = mean change of the turnT log-m read-out under turn_hard14_10 / its mean
  change under dil10 on the same jets (0 = the read-out ignores the true-mass change at fixed m_A, 1 = it follows it as a real
  mass change); slope_Delta = mean change of the pool Delta read-out / mean true change of Delta under the turn.
Controls (arm ctrl): 10 untrained redraws Jedi(untrained=True, seed=k), 10 per-layer row-permuted copies (opt-in helper
  permuted_jedi: the rows (input axis) of every kernel permuted by one random permutation per matrix, biases and batch-norm
  kept), and the 7 non-degenerate retrained / official networks of E39 EDIT_NETS (official_n64, official_n32, perminv_n64,
  hgq150_s0, hgq150_s2, float_s0, float_s1; E33.Model).  Same pools, same read-outs (ridge all fits; MLP Delta pool only).
Statistics: 95 % CIs from 1,000 Poisson jet-bootstrap resamples of the test jets (all versions of a jet resampled together).
  H0: R^2(Delta) at stage X <= the 95th percentile of the untrained + permuted band; H1: above it.  p_rank = (1 + #{band >=
  trained}) / 21 (floor 0.048); p_z = one-sided normal test of trained vs band mean with variance sd_band^2 + se_boot^2.
  Holm over the 10 stages (on p_z; p_rank reported).
Interchange (arm swap; stages S, h4, h5, where the exact fixed-point tail with clipping to each next quantizer's range and a
  straight-through gradient exists, interp.x10_das_jedi.JNet):
  pairs = (turned jet, its unturned original) for turn_hard14_10 and turn_soft10: same m_A, different m.  v_Delta = 1-dim DAS
  direction trained (Adam, 400 steps) so that swapping the v coordinate of the original into the turned jet's activation gives
  the original's call (the true-m counterfactual); also the ridge Delta probe direction.  Positive control v_mA: DAS on shift6
  pairs (original, shifted) toward the shifted jet's call (m_A counterfactual), and the log m_A probe direction.
  Scores on test pairs: IIA = share of pairs whose call after the swap equals the counterfactual call (all pairs; and on the
  informative pairs whose counterfactual call differs from the base call); calls changed = share whose call differs from the
  base call; W->Z / Z->W flow.  Null: 200 random unit directions in raw stage coordinates (p = rank; McNemar of v vs each draw,
  median p reported).  Steering calibration: h -> h + log(0.9) g / |g|^2 with g the raw-coordinate gradient of the Delta (or
  log m_A) ridge probe (moves the read-out by a 10 % mass change), response = change of the W-Z logit margin on W/Z-called
  natural test jets; R = response / response to dil10 (a real 10 % mass change) on the same jets.
Predictions: A4 (M2-like axis mass, true m not built): R^2(Delta) inside the band at every stage, turn R_m ~ 0; Alt 1 (true m
  computed, not used): R^2(Delta) above band somewhere, v_Delta swap moves calls no more than random; Alt 2 (computed and used
  weakly): swap above random with flow toward the counterfactual; Alt 3 (readable for architectural reasons): untrained /
  permuted reach the trained R^2.
Arms / outputs ($AI_RESULTS/jedi/):
  python -m interp.e41_true_mass_layers_jedi --arm main                      -> e41_main.json, e41_example.npz
  python -m interp.e41_true_mass_layers_jedi --arm ctrl --nets u0-9,p0-9     -> e41_ctrl_up.json
  python -m interp.e41_true_mass_layers_jedi --arm ctrl --nets retrained     -> e41_ctrl_rt.json
  python -m interp.e41_true_mass_layers_jedi --arm swap                      -> e41_swap.json
  python -m interp.e41_true_mass_layers_jedi --arm combine                   -> e41_summary.json
  options: --seeds 4101,4102 (verify: 4111,4112; output tag gets the seeds), --smoke N (per_class N, short fits)
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time(); W_, Z_ = 2, 3
STG = ('a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z', 'logits')
RUNK = ('a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z', 'logits')
TGT = ('logm', 'logmA', 'Delta')
DOSES = (0.02, 0.05, 0.10)
RETRAINED = ['official_n64', 'official_n32', 'perminv_n64', 'hgq150_s0', 'hgq150_s2', 'float_s0', 'float_s1']
SWAP_ST = ('S', 'h4', 'h5')
warnings.filterwarnings('ignore')


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


# ================================================================================================ physics
def jet_obs(x):
    from interp.x10_das_jedi import mass
    x = np.asarray(x, np.float64); pt = x[..., 0]; de, dp = x[..., 1] * (pt > 0), x[..., 2] * (pt > 0)
    M0 = pt.sum(1); M1 = np.stack([(pt * de).sum(1), (pt * dp).sum(1)], 1); M2 = (pt * (de ** 2 + dp ** 2)).sum(1)
    m = mass(x); mA = np.sqrt(M0 * M2)
    return dict(M0=M0, M1=np.hypot(*M1.T), M2=M2, m=m, mA=mA, logm=np.log(np.maximum(m, 1e-6)),
                logmA=np.log(np.maximum(mA, 1e-6)), Delta=np.log(np.maximum(m, 1e-6)) - np.log(np.maximum(mA, 1e-6)))


def build_pool(x, smoke=False, seed=0):
    """-> dict(x (rows, 64, 3) float32, jid, fam, vname, rowidx {vname: (J,) row or -1}, info)"""
    from interp.e27w_mass_vs_m2_jedi import solve_B, mass_p, M1v, to_x, wrap
    from interp.x10_das_jedi import mass
    J = len(x); rng = np.random.default_rng(seed)
    x64 = np.asarray(x, np.float64); pt = x64[..., 0]; real = pt > 0; r = np.hypot(x64[..., 1], x64[..., 2]); th = np.arctan2(x64[..., 2], x64[..., 1])
    V = [('nat', 'nat', np.asarray(x, np.float32), np.ones(J, bool))]
    for dose in (0.03, 0.06):
        pick = rng.integers(0, 4, J); xe = x64.copy()
        for c in (1, 2):
            amt = np.where((pick // 2) == (c - 1), np.where(pick % 2 == 0, dose, -dose), 0.0)
            xe[..., c] = xe[..., c] + amt[:, None] * real
        V.append((f'shift{int(round(dose * 100))}', 'shift', xe.astype(np.float32), np.ones(J, bool)))
    rank = np.argsort(np.argsort(-np.where(real, pt, -1), 1), 1); nreal = real.sum(1)
    SUB = dict(hard14=real & (rank < 14), soft=real & (rank >= np.ceil(nreal / 2)[:, None]))
    M10 = M1v(pt, r, th); phu = np.arctan2(M10[:, 1], M10[:, 0]); mx0 = mass(x)
    info = {}
    for sname in ('hard14', 'soft'):
        sub = SUB[sname]; d = np.where(sub, wrap(phu[:, None] - th), 0.0)
        for f in DOSES:
            pc = int(round(f * 100)); tB, ok = solve_B(pt, r, th, d, f); xB = to_x(x, real, r, th + tB[:, None] * d)
            mB = mass(xB); dl = np.log(np.maximum(mB, 1e-9) / np.maximum(mx0, 1e-9))
            ok = ok & np.isfinite(dl) & (np.abs(dl - np.log(1 - f)) <= 0.1 * abs(np.log(1 - f)))
            V.append((f'turn_{sname}{pc}', 'turn', xB, ok))
            info[f'turn_{sname}{pc}'] = dict(kept=int(ok.sum()), median_dlogm=float(np.median(dl[ok])) if ok.any() else None)
            if sname == 'hard14':
                aD = mB / np.maximum(mx0, 1e-9)
                for _ in range(5):
                    xD = to_x(x, real, r * aD[:, None], th); aD = aD * mB / np.maximum(mass(xD), 1e-9)
                xD = to_x(x, real, r * aD[:, None], th); dlD = np.log(np.maximum(mass(xD), 1e-9) / np.maximum(mx0, 1e-9))
                okD = ok & (np.abs(dlD - dl) <= 1e-3)
                V.append((f'dil{pc}', 'dil', xD, okD)); info[f'dil{pc}'] = dict(kept=int(okD.sum()), median_dlogm=float(np.median(dlD[okD])) if okD.any() else None)
    sub = SUB['hard14']; d = np.where(sub, wrap(phu[:, None] + np.pi - th), 0.0); ts = np.linspace(0, 1, 41)
    mg = np.stack([mass_p(pt, r, th + t * d) for t in ts]); kbest = mg.argmax(0); tU = ts[kbest]
    xU = to_x(x, real, r, th + tU[:, None] * d); dlU = np.log(np.maximum(mass(xU), 1e-9) / np.maximum(mx0, 1e-9)); okU = (tU > 0) & (dlU > 1e-6)
    V.append(('turnup', 'turn', xU, okU)); info['turnup'] = dict(kept=int(okU.sum()), median_dlogm=float(np.median(dlU[okU])) if okU.any() else None,
                                                                 q90_dlogm=float(np.quantile(dlU[okU], 0.9)) if okU.any() else None)
    xs, jid, fam, vn = [], [], [], []; rowidx = {}; n0 = 0
    for name, fm, xe, ok in V:
        k = np.flatnonzero(ok); xs.append(xe[k]); jid.append(k); fam += [fm] * len(k); vn += [name] * len(k)
        ri = np.full(J, -1); ri[k] = n0 + np.arange(len(k)); rowidx[name] = ri; n0 += len(k)
    return dict(x=np.concatenate(xs).astype(np.float32), jid=np.concatenate(jid), fam=np.array(fam), vname=np.array(vn),
                rowidx=rowidx, info=info, SUB=SUB)


# ================================================================================================ networks
def permuted_jedi(seed):
    """opt-in helper: Jedi with the rows (input axis) of every kernel permuted (one random permutation per matrix); biases and
    batch-norm parameters kept.  Implemented by swapping jetdistill.network.redraw_untrained for the duration of the load."""
    from jetdistill import network as N
    from interp.jedi_core import Jedi
    orig = N.redraw_untrained

    def perm(model, seed=0):
        rng = np.random.default_rng(10_000 + seed)
        for w in model.weights:
            if getattr(w, 'path', w.name).endswith('/kernel'):
                v = np.asarray(w.numpy())
                if v.ndim >= 2: w.assign(v[rng.permutation(v.shape[0])])
    N.redraw_untrained = perm
    try:
        return Jedi(64, untrained=True, seed=seed)
    finally:
        N.redraw_untrained = orig


def runner(name):
    """-> (callable xb -> dict of RUNK arrays, description)"""
    from interp.jedi_core import Jedi
    if name == 'trained':
        net = Jedi(64); return (lambda xb: net.run(xb, keep=RUNK, batch=1000)), 'official 3-feature n64'
    if name.startswith('u'):
        net = Jedi(64, untrained=True, seed=int(name[1:])); return (lambda xb: net.run(xb, keep=RUNK, batch=1000)), f'untrained redraw seed {name[1:]}'
    if name.startswith('p') and name[1:].isdigit():
        net = permuted_jedi(int(name[1:])); return (lambda xb: net.run(xb, keep=RUNK, batch=1000)), f'row-permuted kernels seed {name[1:]}'
    from interp.e33_program_recurrence_jedi import Model
    M = Model(name); return (lambda xb: M.run(xb, keep=RUNK)), f'E33.Model {name} (n {M.n})'


def feats(run, x, chunk=1000):
    out = {s: [] for s in STG}
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk]; R = run(xb); n = R['a'].shape[1]
        pt = np.asarray(xb[:, :n, 0], np.float64); real = (pt > 0)[..., None]; nr = np.maximum(real.sum(1), 1)
        out['a'].append(np.concatenate([(R['a'] * real).sum(1) / nr, (R['a'] * pt[..., None]).sum(1) / np.maximum(pt.sum(1), 1e-9)[:, None]], 1))
        for s in ('s', 'u'): out[s].append((R[s] * real).sum(1) / nr)
        for s in ('P', 'g', 'S', 'h4', 'h5', 'z', 'logits'): out[s].append(R[s].reshape(len(xb), -1))
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


# ================================================================================================ read-outs
class Ridge:
    ALPHAS = 10.0 ** np.arange(-7, 2.1, 1)

    def __init__(self, X, y, jid):
        X = np.asarray(X, np.float64); self.mu = X.mean(0); sd = X.std(0); self.live = sd > 1e-9 * (1 + np.abs(self.mu))
        self.sd = np.where(self.live, sd, np.inf); Z = ((X - self.mu) / self.sd)[:, self.live]
        if not self.live.any():   # debug 1: a stage with no live unit (untrained redraws) -> constant read-out
            self.alpha = None; self.w = np.zeros(0); self.b = float(y.mean()); return
        fold = np.random.default_rng(0).permutation(int(jid.max()) + 1)[jid] % 3; err = np.zeros(len(self.ALPHAS))
        for f in range(3):
            tr, te = fold != f, fold == f
            mz, my = Z[tr].mean(0), y[tr].mean(); Zc = Z[tr] - mz; lam, Vv = np.linalg.eigh(Zc.T @ Zc); Vb = Vv.T @ (Zc.T @ (y[tr] - my))
            Zt = Z[te] - mz
            for i, a in enumerate(self.ALPHAS):
                w = Vv @ (Vb / (lam + a * tr.sum())); err[i] += ((Zt @ w + my - y[te]) ** 2).sum()
        self.alpha = float(self.ALPHAS[int(np.argmin(err))]); mz, my = Z.mean(0), y.mean(); Zc = Z - mz
        self.w = np.linalg.solve(Zc.T @ Zc + self.alpha * len(y) * np.eye(Z.shape[1]), Zc.T @ (y - my)); self.b = my - mz @ self.w

    def __call__(self, X): return ((np.asarray(X, np.float64) - self.mu) / self.sd)[:, self.live] @ self.w + self.b

    def grad_raw(self):
        g = np.zeros(len(self.mu)); g[self.live] = self.w / self.sd[self.live]; return g


class MLP:
    def __init__(self, X, y, nmax, seed, max_iter):
        from sklearn.neural_network import MLPRegressor
        X = np.asarray(X, np.float64); rng = np.random.default_rng(seed); idx = rng.choice(len(y), min(nmax, len(y)), replace=False)
        self.mu = X.mean(0); sd = X.std(0); self.live = sd > 1e-9 * (1 + np.abs(self.mu)); self.sd = np.where(self.live, sd, 1.0)
        self.ym, self.ys = y.mean(), y.std() + 1e-12; self.n_iter = 0
        if not self.live.any():   # debug 1: no live unit -> constant read-out
            self.m = None; return
        self.m = MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=max_iter,
                              early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=seed)
        self.m.fit(((X[idx] - self.mu) / self.sd)[:, self.live], (y[idx] - self.ym) / self.ys); self.n_iter = int(self.m.n_iter_)

    def __call__(self, X):
        if self.m is None: return np.full(len(X), self.ym)
        return self.m.predict(((np.asarray(X, np.float64) - self.mu) / self.sd)[:, self.live]) * self.ys + self.ym


def r2_boot(y, p, jid, Wb):
    """point R^2, 95 % CI and bootstrap se over jets (Wb: (NB, J) Poisson weights over test jets)"""
    Jn = Wb.shape[1]; n = np.bincount(jid, minlength=Jn).astype(float); sy = np.bincount(jid, y, Jn); syy = np.bincount(jid, y * y, Jn)
    sse = np.bincount(jid, (y - p) ** 2, Jn)

    def r(w):
        N = w @ n; SY = w @ sy; return 1 - (w @ sse) / np.maximum(w @ syy - SY ** 2 / np.maximum(N, 1e-12), 1e-30)
    pt = float(r(np.ones(Jn))); b = r(Wb); b = b[np.isfinite(b)]
    cr = float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 and y.std() > 0 and p.std() > 0 else None
    return dict(r2=pt, ci=np.quantile(b, [.025, .975]).tolist(), se=float(b.std()), corr=cr, n_rows=int(len(y)))


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


# ================================================================================================ data
def load_data(sf, st, pc, smoke):
    from interp.jedi_core import sample
    xf, yf, _ = sample(pc, sf); xt, yt, _ = sample(pc, st)
    hf = {hash(v.tobytes()) for v in xf}; kt = np.array([hash(v.tobytes()) not in hf for v in xt]); xt, yt = xt[kt], yt[kt]
    t = time.time(); Pf = build_pool(xf, smoke, seed=sf); Pt = build_pool(xt, smoke, seed=st)
    log(f'pools: fit {len(Pf["jid"])} rows from {len(yf)} jets, test {len(Pt["jid"])} rows from {len(yt)} jets ({int((~kt).sum())} overlapping dropped) ({time.time() - t:.0f} s)')
    for P in (Pf, Pt): P['obs'] = jet_obs(P['x'])
    return xf, yf, xt, yt, Pf, Pt, int((~kt).sum())


def fit_sets(P):
    f = P['fam']; return dict(pool=np.ones(len(f), bool), turn=f == 'turn', shift=f == 'shift', natdil=(f == 'nat') | (f == 'dil'),
                              nsd=(f == 'nat') | (f == 'shift') | (f == 'dil'))


def readouts(Ff, Ft, Pf, Pt, Wb, mlp_n, mlp_it, mlp_targets=TGT, mlp_transfer=True, ridge_targets=TGT, keep_probe=False):
    """per-stage read-out results (dict) and optionally the pool ridge probes"""
    Sf, St = fit_sets(Pf), fit_sets(Pt); out = {}; probes = {}
    for stg in Ff:
        o = {}; t = time.time()
        for tg in ridge_targets:
            yf, yt = Pf['obs'][tg], Pt['obs'][tg]; d = {}
            for kind in ('ridge', 'mlp'):
                if kind == 'mlp' and tg not in mlp_targets: continue
                mk = (lambda X, y, j: Ridge(X, y, j)) if kind == 'ridge' else (lambda X, y, j: MLP(X, y, mlp_n, 41, mlp_it))
                pr = mk(Ff[stg], yf, Pf['jid']); p = pr(Ft[stg])
                e = dict(pool=r2_boot(yt, p, Pt['jid'], Wb))
                for sb in ('turn', 'shift', 'natdil'):
                    m = St[sb]; e[f'pool_on_{sb}'] = r2_boot(yt[m], p[m], Pt['jid'][m], Wb)
                if kind == 'ridge':
                    e['alpha'] = pr.alpha
                    if keep_probe: probes[(stg, tg)] = pr
                if kind == 'ridge' or (mlp_transfer and tg == 'Delta'):
                    m1, m2 = Sf['nsd'], St['turn']; pr1 = mk(Ff[stg][m1], yf[m1], Pf['jid'][m1]); p1 = pr1(Ft[stg][m2])
                    e['turnT'] = r2_boot(yt[m2], p1, Pt['jid'][m2], Wb)
                    m1, m2 = Sf['turn'], St['nsd']; pr2 = mk(Ff[stg][m1], yf[m1], Pf['jid'][m1]); p2 = pr2(Ft[stg][m2])
                    e['revT'] = r2_boot(yt[m2], p2, Pt['jid'][m2], Wb)
                    if kind == 'ridge':
                        # calibrated read-out response to the turn (test jets with turn_hard14_10 and dil10 both kept)
                        ri = Pt['rowidx']; a0, aT, aD = ri['nat'], ri['turn_hard1410'], ri['dil10']; k = (aT >= 0) & (aD >= 0)
                        pall = pr1(Ft[stg]); dT = pall[aT[k]] - pall[a0[k]]; dD = pall[aD[k]] - pall[a0[k]]
                        wk = Wb[:, k]; e['turn_response_nsd_probe'] = dict(n=int(k.sum()), mean_turn=float(dT.mean()), mean_dil=float(dD.mean()),
                                                                           R=float(dT.mean() / dD.mean()) if abs(dD.mean()) > 0 else None,
                                                                           R_ci=qci((wk @ dT) / np.maximum(wk @ dD, 1e-30) if dD.mean() > 0 else (wk @ dT) / np.minimum(wk @ dD, -1e-30)))
                        if tg == 'Delta':
                            dTrue = Pt['obs']['Delta'][aT[k]] - Pt['obs']['Delta'][a0[k]]; dP = p[aT[k]] - p[a0[k]]
                            e['slope_pool_probe_on_turn'] = dict(slope=float(dP.mean() / dTrue.mean()), ci=qci((wk @ dP) / (wk @ dTrue)))
                d[kind] = e
            o[tg] = d
        out[stg] = o
        log(f'  stage {stg} ({time.time() - t:.0f} s): R2(Delta) ridge pool {o["Delta"]["ridge"]["pool"]["r2"]:.3f} turnT {o["Delta"]["ridge"]["turnT"]["r2"]:.3f}'
            + (f' mlp pool {o["Delta"]["mlp"]["pool"]["r2"]:.3f}' if 'mlp' in o['Delta'] else '')
            + f'; R2(logmA) {o["logmA"]["ridge"]["pool"]["r2"]:.3f}; R2(logm) {o["logm"]["ridge"]["pool"]["r2"]:.3f}')
    return (out, probes) if keep_probe else out


# ================================================================================================ C2ST
def c2st_feats(x):
    x = np.asarray(x, np.float64); pt = x[..., 0]; real = pt > 0; de, dp = x[..., 1], x[..., 2]; o = jet_obs(x)
    ptJ = pt.sum(1); z = pt / ptJ[:, None]; e2 = {b: np.zeros(len(x)) for b in (0.5, 1.0, 2.0)}
    for i in range(0, len(x), 500):
        sl = slice(i, i + 500); dR = np.hypot(de[sl, :, None] - de[sl, None, :], wrap_(dp[sl, :, None] - dp[sl, None, :]))
        zz = z[sl, :, None] * z[sl, None, :] * (real[sl, :, None] & real[sl, None, :])
        for b in e2: e2[b][sl] = 0.5 * (zz * dR ** b).sum((1, 2))
    order = np.argsort(-pt, 1); i0, i1 = order[:, 0], order[:, 1]; ar = np.arange(len(x))
    dR01 = np.hypot(de[ar, i0] - de[ar, i1], dp[ar, i0] - dp[ar, i1])
    free = np.stack([o['logm'], np.log(ptJ), real.sum(1), np.sqrt((pt ** 2).sum(1)) / ptJ, e2[0.5], e2[1.0], e2[2.0], pt.max(1) / ptJ, dR01], 1)
    r = np.hypot(de, dp) * real
    axis = np.stack([o['M1'] ** 2 / np.maximum(o['M0'] * o['M2'], 1e-30), o['M1'] / o['M0'], (pt * r).sum(1) / ptJ, o['M2'] / o['M0']], 1)
    return np.nan_to_num(free), np.nan_to_num(axis)


def wrap_(a): return (a + np.pi) % (2 * np.pi) - np.pi


def c2st_auc(F0, F1, seed, NB=200):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.metrics import roc_auc_score
    X = np.concatenate([F0, F1]); y = np.r_[np.zeros(len(F0)), np.ones(len(F1))]; rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % 5; s = np.zeros(len(y))
    for f in range(5):
        m = HGB(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False, random_state=seed).fit(X[fold != f], y[fold != f])
        s[fold == f] = m.predict_proba(X[fold == f])[:, 1]
    auc = float(roc_auc_score(y, s)); b = []
    for _ in range(NB):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): b.append(roc_auc_score(y[i], s[i]))
    return dict(auc=auc, ci=qci(b), n0=len(F0), n1=len(F1))


# ================================================================================================ arms
def arm_main(a, tag):
    from interp.x10_das_jedi import clean
    sf, st = a.seeds; pc = a.smoke or 1000; NB = 200 if a.smoke else 1000; mlp_n, mlp_it = (3000, 30) if a.smoke else (25000, 120)
    JS = OUT / f'e41_main{tag}.json'; R = dict(node='E41', arm='main', plan='module docstring (fixed before the first run)')
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E731
    xf, yf, xt, yt, Pf, Pt, ov = load_data(sf, st, pc, a.smoke)
    R['samples'] = dict(fit=f'sample(per_class={pc}, seed={sf})', test=f'sample(per_class={pc}, seed={st}) minus fit overlap', N_fit=len(yf), N_test=len(yt),
                        overlap_dropped=ov, rows_fit=len(Pf['jid']), rows_test=len(Pt['jid']),
                        rows_by_version_test={k: int((Pt['vname'] == k).sum()) for k in Pt['rowidx']}, pool_info_fit=Pf['info'], pool_info_test=Pt['info'])
    # natural-jet agreement of the two masses and edit checks
    o0 = jet_obs(xf); R['natural'] = dict(median_m_over_mA=float(np.median(o0['m'] / o0['mA'])), q01_m_over_mA=float(np.quantile(o0['m'] / o0['mA'], 0.01)),
                                          sd_Delta=float(o0['Delta'].std()), corr_logm_logmA=float(np.corrcoef(o0['logm'], o0['logmA'])[0, 1]))
    ob = Pt['obs']; R['pool_target_sd_test'] = {fm: {tg: float(ob[tg][Pt['fam'] == fm].std()) for tg in TGT} for fm in ('nat', 'shift', 'turn', 'dil')}
    R['pool_target_sd_test']['all'] = {tg: float(ob[tg].std()) for tg in TGT}
    chk = {}
    for vn in Pt['rowidx']:
        ri = Pt['rowidx'][vn]; k = ri >= 0; i0 = Pt['rowidx']['nat'][k]; i1 = ri[k]
        if not k.any(): chk[vn] = dict(n=0); continue
        chk[vn] = dict(n=int(k.sum()), median_dlogm=float(np.median(ob['logm'][i1] - ob['logm'][i0])), median_dlogmA=float(np.median(ob['logmA'][i1] - ob['logmA'][i0])),
                       median_dDelta=float(np.median(ob['Delta'][i1] - ob['Delta'][i0])), max_abs_dlogmA=float(np.abs(ob['logmA'][i1] - ob['logmA'][i0]).max()))
    R['edit_checks_test'] = chk; log('edit checks (test):', json.dumps(clean({k: (v['n'], v.get('median_dlogm'), v.get('median_dlogmA')) for k, v in chk.items()})))
    save()
    # network, accuracy
    run, desc = runner('trained'); t = time.time()
    Ff = feats(run, Pf['x']); Ft = feats(run, Pt['x'])
    natf = Pf['fam'] == 'nat'; acc = float((Ff['logits'][natf].argmax(1) == yf[Pf['jid'][natf]]).mean())
    natt = Pt['fam'] == 'nat'; acct = float((Ft['logits'][natt].argmax(1) == yt[Pt['jid'][natt]]).mean())
    R['accuracy'] = dict(fit=acc, test=acct, expected='0.806-0.813'); log(f'forward passes ({time.time() - t:.0f} s); accuracy fit {acc:.4f} test {acct:.4f}'); save()
    # calls under each edit (context)
    c0 = Ft['logits'].argmax(1); cn = c0[Pt['rowidx']['nat'][Pt['jid']]]
    R['calls_changed_by_version_test'] = {vn: dict(changed=float((c0 != cn)[Pt['vname'] == vn].mean()),
                                                   WZ=float(((cn == W_) & (c0 == Z_))[Pt['vname'] == vn].sum() / max(((cn == W_) & (Pt['vname'] == vn)).sum(), 1)),
                                                   ZW=float(((cn == Z_) & (c0 == W_))[Pt['vname'] == vn].sum() / max(((cn == Z_) & (Pt['vname'] == vn)).sum(), 1)))
                                          for vn in Pt['rowidx'] if vn != 'nat'}
    Wb = np.random.default_rng(4141).poisson(1.0, (NB, len(yt))).astype(np.float64)
    log('read-outs, trained network'); R['stages'] = readouts(Ff, Ft, Pf, Pt, Wb, mlp_n, mlp_it, mlp_targets=TGT, mlp_transfer=True); save()
    # raw-input baselines
    def raw(P):
        xx = np.asarray(P['x'], np.float64); o = np.argsort(-xx[..., 0], 1); xs = np.take_along_axis(xx, o[..., None], 1)
        return xs.reshape(len(xx), -1).astype(np.float32)
    def mom(P): o = P['obs']; return np.stack([np.log(o['M0']), np.log(o['M1'] + 1e-6), np.log(o['M2'] + 1e-12)], 1).astype(np.float32)
    log('read-outs, raw-input baselines')
    R['baselines'] = readouts({'input64x3': raw(Pf), 'M0_M1_M2': mom(Pf)}, {'input64x3': raw(Pt), 'M0_M1_M2': mom(Pt)}, Pf, Pt, Wb, mlp_n, mlp_it,
                              mlp_targets=TGT, mlp_transfer=True); save()
    # C2ST on the fit sample: natural half vs edited versions of the other half
    t = time.time(); J = len(yf); half = np.random.default_rng(4143).permutation(J); h0, h1 = half[:J // 2], half[J // 2:]
    FAf, FXf = c2st_feats(Pf['x']); ri = Pf['rowidx']; C = {}
    fams = dict(shift3=['shift3'], shift6=['shift6'], turn_hard14_10=['turn_hard1410'], turn_soft10=['turn_soft10'], turn_hard14_5=['turn_hard145'],
                turnup=['turnup'], dil10=['dil10'], dil5=['dil5'])
    n0 = ri['nat'][h0]
    for nm, vns in fams.items():
        rows = np.concatenate([ri[v][h1][ri[v][h1] >= 0] for v in vns])
        if len(rows) < 50: continue
        C[nm] = dict(axis_free=c2st_auc(FAf[n0], FAf[rows], 4144), axis_relative=c2st_auc(FXf[n0], FXf[rows], 4145))
        C[nm]['flag_axis_free_gt_0.6'] = C[nm]['axis_free']['auc'] > 0.6
    nh = ri['nat'][h1]; C['null_nat_vs_nat'] = dict(axis_free=c2st_auc(FAf[n0], FAf[nh], 4146), axis_relative=c2st_auc(FXf[n0], FXf[nh], 4147))
    R['c2st'] = C; log(f'C2ST ({time.time() - t:.0f} s):', json.dumps(clean({k: (v['axis_free']['auc'], v['axis_relative']['auc']) for k, v in C.items()}))); save()
    # example jet for the figure (true W, called W, hard14 10 % turn and shift6 kept)
    ri = Pf['rowidx']; cand = np.flatnonzero((yf == W_) & (ri['turn_hard1410'] >= 0) & (ri['dil10'] >= 0))
    if len(cand):
        nr = (xf[cand, :, 0] > 0).sum(1); ex = int(cand[np.argsort(nr)[len(cand) // 2]]); ob = Pf['obs']
        EX = dict(index=ex, label=int(yf[ex]), x=xf[ex], x_turn=Pf['x'][ri['turn_hard1410'][ex]], x_shift=Pf['x'][ri['shift6'][ex]], x_dil=Pf['x'][ri['dil10'][ex]],
                  moved=Pf['SUB']['hard14'][ex])
        for k_, v in (('nat', 'nat'), ('turn', 'turn_hard1410'), ('shift', 'shift6'), ('dil', 'dil10')):
            j = ri[v][ex]
            for q in ('m', 'mA', 'M1', 'M0', 'M2', 'Delta'): EX[f'{q}_{k_}'] = float(ob[q][j])
            EX[f'call_{k_}'] = int(Ff['logits'][j].argmax())
        np.savez_compressed(OUT / f'e41_example{tag}.npz', **{k_: np.asarray(v) for k_, v in EX.items()})
        R['example'] = {k_: v for k_, v in EX.items() if np.ndim(v) == 0}
    R['seconds'] = time.time() - T0; save(); log(f'done, wrote {JS}')


def arm_ctrl(a, tag):
    from interp.x10_das_jedi import clean
    sf, st = a.seeds; pc = a.smoke or 1000; NB = 200 if a.smoke else 1000; mlp_n, mlp_it = (3000, 30) if a.smoke else (15000, 80)
    nets = []
    for tok in a.nets.split(','):
        if tok == 'retrained': nets += RETRAINED
        elif '-' in tok and tok[0] in 'up':
            lo, hi = tok[1:].split('-'); nets += [f'{tok[0]}{k}' for k in range(int(lo), int(hi) + 1)]
        else: nets.append(tok)
    JS = OUT / f'e41_ctrl_{a.label}{tag}.json'; R = dict(node='E41', arm='ctrl', nets=nets, results={})
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E731
    xf, yf, xt, yt, Pf, Pt, ov = load_data(sf, st, pc, a.smoke)
    Wb = np.random.default_rng(4141).poisson(1.0, (NB, len(yt))).astype(np.float64)
    for nm in nets:
        t = time.time()
        try:
            run, desc = runner(nm); Ff = feats(run, Pf['x']); Ft = feats(run, Pt['x'])
        except Exception as e:  # noqa: BLE001
            log(f'{nm}: load / forward failed: {e!r}'); R['results'][nm] = dict(error=repr(e)); save(); continue
        natt = Pt['fam'] == 'nat'; acc = float((Ft['logits'][natt].argmax(1) == yt[Pt['jid'][natt]]).mean())
        log(f'{nm} ({desc}): forward {time.time() - t:.0f} s, test accuracy {acc:.4f}')
        st_ = readouts(Ff, Ft, Pf, Pt, Wb, mlp_n, mlp_it, mlp_targets=('Delta',), mlp_transfer=False)
        R['results'][nm] = dict(desc=desc, acc_test=acc, stages=st_); save(); log(f'{nm} done ({time.time() - t:.0f} s)')
    R['seconds'] = time.time() - T0; save(); log(f'done, wrote {JS}')


def arm_swap(a, tag):
    from interp.x10_das_jedi import clean, JNet, das_fit, proj as qclip
    from interp.h12_boost_panel import mcnemar
    sf, st = a.seeds; pc = a.smoke or 1000; NB = 200 if a.smoke else 1000; NR = 20 if a.smoke else 200; steps = 60 if a.smoke else 400
    JS = OUT / f'e41_swap{tag}.json'; R = dict(node='E41', arm='swap', stages=SWAP_ST)
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E731
    xf, yf, xt, yt, Pf, Pt, ov = load_data(sf, st, pc, a.smoke)
    jn = JNet(); keep = ('S', 'h4', 'h5', 'logits')
    Hf = jn.net.run(Pf['x'], keep=keep, batch=1000); Ht = jn.net.run(Pt['x'], keep=keep, batch=1000)
    natt = Pt['fam'] == 'nat'; R['accuracy_test'] = float((Ht['logits'][natt].argmax(1) == yt[Pt['jid'][natt]]).mean())
    nf = Pf['fam'] == 'nat'; R['check'] = jn.check(Pf['x'][nf][:2000], {k: v[nf][:2000] for k, v in Hf.items()})
    log(f'swap arm: test accuracy {R["accuracy_test"]:.4f}')
    cf, ct = Hf['logits'].argmax(1), Ht['logits'].argmax(1)
    Wb = np.random.default_rng(4151).poisson(1.0, (NB, len(yt))).astype(np.float64)

    def pairs(P, base_v, src_v):
        """rows (base, source) over jets where both versions exist; base_v / src_v lists of version names (concatenated)"""
        b, s, j = [], [], []
        for bv, sv in zip(base_v, src_v):
            ri_b, ri_s = P['rowidx'][bv], P['rowidx'][sv]; k = (ri_b >= 0) & (ri_s >= 0); b.append(ri_b[k]); s.append(ri_s[k]); j.append(np.flatnonzero(k))
        return np.concatenate(b), np.concatenate(s), np.concatenate(j)
    PAIRS = dict(turn=(['turn_hard1410', 'turn_soft10'], ['nat', 'nat']), shift=(['nat'], ['shift6']))
    # pool ridge probes (fit sample) at the swap stages
    Sf = fit_sets(Pf); probes = {(s_, tg): Ridge(Hf[s_], Pf['obs'][tg], Pf['jid']) for s_ in SWAP_ST for tg in ('Delta', 'logmA', 'logm')}

    def evals(stage, Q, hb, hs, lab, basec, y_t):
        h = qclip(jn.rng_[stage], hb, hb + ((hs - hb) @ Q) @ Q.T); lo, _ = jn.exact(stage, h); c = lo.argmax(1)
        return c

    def summarise(c, lab, basec, jid, cR):
        inf = lab != basec; o = dict(n=len(c), n_informative=int(inf.sum()))
        ok = (c == lab).astype(float); ch = c != basec; wk = Wb[:, jid]
        o['IIA'] = float(ok.mean()); o['IIA_ci'] = qci((wk @ ok) / wk.sum(1))
        if inf.any():
            o['IIA_informative'] = float(ok[inf].mean()); o['IIA_informative_ci'] = qci((wk[:, inf] @ ok[inf]) / np.maximum(wk[:, inf].sum(1), 1e-12))
        o['calls_changed'] = float(ch.mean()); o['calls_changed_ci'] = qci((wk @ ch.astype(float)) / wk.sum(1))
        bw, bz = basec == W_, basec == Z_
        o['WZ'] = float((c[bw] == Z_).mean()) if bw.any() else None; o['ZW'] = float((c[bz] == W_).mean()) if bz.any() else None
        if cR is not None:
            chR = cR != basec[None]; rate = chR.mean(1); o['random_calls_changed_median'] = float(np.median(rate)); o['random_calls_changed_q95'] = float(np.quantile(rate, 0.95))
            o['p_rank_calls_changed'] = float((1 + (rate >= o['calls_changed']).sum()) / (1 + len(rate)))
            okR = (cR == lab[None]); o['random_IIA_informative_median'] = float(np.median(okR[:, inf].mean(1))) if inf.any() else None
            o['p_rank_IIA_informative'] = float((1 + (okR[:, inf].mean(1) >= o.get('IIA_informative', 1)).sum()) / (1 + len(rate))) if inf.any() else None
            ps = [mcnemar(ch, chR[i])['p'] for i in range(len(rate))]; o['mcnemar_vs_random_median_p'] = float(np.median(ps))
            o['mcnemar_vs_random_median_draw'] = mcnemar(ch, chR[int(np.argsort(rate)[len(rate) // 2])])
        return o
    R['res'] = {}
    for stage in SWAP_ST:
        t = time.time(); d = Hf[stage].shape[1]; o = {}
        rngR = np.random.default_rng(4152 + SWAP_ST.index(stage)); RD = rngR.standard_normal((NR, d)); RD /= np.linalg.norm(RD, axis=1, keepdims=True)
        for pn, (bv, sv) in PAIRS.items():
            # base = jet that is intervened on; source = jet whose coordinate is swapped in; counterfactual = source's call
            if pn == 'turn': bF, sF, jF = pairs(Pf, bv, sv); bT, sT, jT = pairs(Pt, bv, sv)   # base turned, source original
            else: bF, sF, jF = pairs(Pf, bv, sv); bT, sT, jT = pairs(Pt, bv, sv)              # base original, source shifted
            perm = np.random.default_rng(4153).permutation(len(bF)); bF, sF = bF[perm], sF[perm]
            Q, info = das_fit(jn.tail(stage), Hf[stage][bF], Hf[stage][sF], cf[sF], cf[bF], 1, 4154, steps=steps, lr=0.01, rg=jn.rng_[stage])
            hb, hs, lab, basec = Ht[stage][bT], Ht[stage][sT], ct[sT], ct[bT]
            tg = 'Delta' if pn == 'turn' else 'logmA'; g = probes[(stage, tg)].grad_raw(); Qp = (g / max(np.linalg.norm(g), 1e-30))[:, None]
            cR = np.stack([evals(stage, RD[i][:, None], hb, hs, lab, basec, None) for i in range(NR)])
            r_ = dict(pairs=f'{bv} -> {sv}', das_info=info, cos_das_probe=float(abs(Q[:, 0] @ Qp[:, 0])),
                      no_swap=dict(IIA=float((basec == lab).mean()), n_informative=int((lab != basec).sum())),
                      das=summarise(evals(stage, Q, hb, hs, lab, basec, None), lab, basec, jT, cR),
                      probe=summarise(evals(stage, Qp, hb, hs, lab, basec, None), lab, basec, jT, cR),
                      full_stage=summarise(jn.exact(stage, hs)[0].argmax(1), lab, basec, jT, None))
            o[pn] = r_
            log(f'  {stage} {pn}: DAS IIA_inf {r_["das"].get("IIA_informative")} (random median {r_["das"].get("random_IIA_informative_median")}), calls changed DAS {r_["das"]["calls_changed"]:.4f} '
                f'probe {r_["probe"]["calls_changed"]:.4f} random median {r_["das"]["random_calls_changed_median"]:.4f} (p_rank {r_["das"]["p_rank_calls_changed"]:.3f}); full-stage swap IIA {r_["full_stage"]["IIA"]:.3f}; n inf {r_["no_swap"]["n_informative"]}')
        # steering calibration on W/Z-called natural test jets with dil10 kept
        ri = Pt['rowidx']; k = (ri['dil10'] >= 0); n0, nd = ri['nat'][k], ri['dil10'][k]; wz = np.isin(ct[n0], (W_, Z_)); n0, nd = n0[wz], nd[wz]; jk = np.flatnonzero(k)[wz]
        marg = lambda lo: lo[:, Z_] - lo[:, W_]  # noqa: E731
        lo0 = Ht['logits'][n0]; dD = marg(Ht['logits'][nd]) - marg(lo0); wk = Wb[:, jk]; cal = dict(n=len(n0), dil10_dmargin=float(dD.mean()))
        for tg in ('Delta', 'logmA', 'logm'):
            g = probes[(stage, tg)].grad_raw(); h0 = Ht[stage][n0]; h = qclip(jn.rng_[stage], h0, h0 + np.log(0.9) * g[None] / max(g @ g, 1e-30))
            lo, _ = jn.exact(stage, h); dS = marg(lo) - marg(lo0); ach = probes[(stage, tg)](h) - probes[(stage, tg)](h0)
            cal[tg] = dict(dmargin=float(dS.mean()), R=float(dS.mean() / dD.mean()), R_ci=qci((wk @ dS) / np.minimum(wk @ dD, -1e-30) if dD.mean() < 0 else (wk @ dS) / np.maximum(wk @ dD, 1e-30)),
                           achieved_probe_change_median=float(np.median(ach)), calls_changed=float((lo.argmax(1) != ct[n0]).mean()),
                           step_norm=float(abs(np.log(0.9)) / max(np.linalg.norm(g), 1e-30)), act_sd_norm=float(np.linalg.norm(Ht[stage][natt].std(0))))
        o['steer'] = cal; R['res'][stage] = o
        log(f'  {stage} steering: dil10 dmargin {dD.mean():+.3f}; R Delta {cal["Delta"]["R"]:+.3f} logmA {cal["logmA"]["R"]:+.3f} logm {cal["logm"]["R"]:+.3f} ({time.time() - t:.0f} s)')
        save()
    R['seconds'] = time.time() - T0; save(); log(f'done, wrote {JS}')


def arm_combine(a, tag):
    from interp.x10_das_jedi import clean, holm
    from scipy.stats import norm
    M = json.loads((OUT / f'e41_main{tag}.json').read_text()); C = {}
    for lab in ('up', 'u', 'p', 'rt'):
        p = OUT / f'e41_ctrl_{lab}{tag}.json'
        if p.exists(): C.update(json.loads(p.read_text())['results'])
    SW = json.loads((OUT / f'e41_swap{tag}.json').read_text()) if (OUT / f'e41_swap{tag}.json').exists() else None
    band_nets = [n for n in C if (n.startswith('u') or (n.startswith('p') and n[1:].isdigit())) and 'stages' in C[n]]
    rt_nets = [n for n in C if n in RETRAINED and 'stages' in C[n]]
    S = dict(band_nets=band_nets, retrained_nets=rt_nets, tests={})
    # trained values for the band tests: the official_n64 entry of the retrained-control run (same read-out budget as the
    # band networks: MLP 15,000 rows / 80 iterations); the main-arm values (MLP 25,000 rows / 120 iterations) reported alongside
    TR = C['official_n64']['stages'] if 'official_n64' in C and 'stages' in C['official_n64'] else M['stages']
    S['trained_source'] = 'e41_ctrl_rt official_n64' if TR is not M['stages'] else 'e41_main'
    for key, (tg, kind, fit) in dict(Delta_ridge_pool=('Delta', 'ridge', 'pool'), Delta_mlp_pool=('Delta', 'mlp', 'pool'),
                                     Delta_ridge_on_turn=('Delta', 'ridge', 'pool_on_turn'), Delta_mlp_on_turn=('Delta', 'mlp', 'pool_on_turn'),
                                     Delta_ridge_on_shift=('Delta', 'ridge', 'pool_on_shift'), Delta_mlp_on_shift=('Delta', 'mlp', 'pool_on_shift'),
                                     Delta_ridge_turnT=('Delta', 'ridge', 'turnT'), Delta_ridge_revT=('Delta', 'ridge', 'revT'),
                                     logm_ridge_turnT=('logm', 'ridge', 'turnT'), logmA_ridge_pool=('logmA', 'ridge', 'pool')).items():
        rows = {}; pz = []
        for stg in STG:
            e = TR[stg][tg][kind][fit]; tr = e['r2']; se = e['se']; em = M['stages'][stg][tg][kind].get(fit, {})
            band = np.array([C[n]['stages'][stg][tg][kind][fit]['r2'] for n in band_nets if kind in C[n]['stages'][stg][tg] and fit in C[n]['stages'][stg][tg][kind]], float)
            rts = {n: C[n]['stages'][stg][tg][kind][fit]['r2'] for n in rt_nets if kind in C[n]['stages'][stg][tg] and fit in C[n]['stages'][stg][tg][kind]}
            if len(band):
                q95 = float(np.quantile(band, 0.95)); prank = float((1 + (band >= tr).sum()) / (1 + len(band)))
                z = (tr - band.mean()) / np.sqrt(band.var(ddof=1) + se ** 2 + 1e-30); pzz = float(norm.sf(z))
            else: q95 = prank = pzz = None
            rows[stg] = dict(trained=tr, trained_ci=e['ci'], trained_corr=e.get('corr'), main_arm=em.get('r2'), main_arm_ci=em.get('ci'), band_n=len(band), band_median=float(np.median(band)) if len(band) else None,
                             band_q05=float(np.quantile(band, 0.05)) if len(band) else None, band_q95=q95, band_max=float(band.max()) if len(band) else None,
                             above_q95=(tr > q95) if q95 is not None else None, p_rank=prank, p_z=pzz, retrained=rts)
            pz.append(pzz if pzz is not None else 1.0)
        hz = holm(pz)
        for stg, h in zip(STG, hz): rows[stg]['p_z_holm'] = float(h)
        S['tests'][key] = rows
    for b in ('input64x3', 'M0_M1_M2'):
        S.setdefault('baselines', {})[b] = {f'{tg}_{kind}_{fit}': M['baselines'][b][tg][kind][fit]['r2'] for tg in TGT for kind in ('ridge', 'mlp')
                                            for fit in ('pool', 'turnT', 'revT') if kind in M['baselines'][b][tg] and fit in M['baselines'][b][tg][kind]}
    S['turn_response_logm_nsd_probe'] = {stg: M['stages'][stg]['logm']['ridge'].get('turn_response_nsd_probe') for stg in STG}
    dec = [stg for stg in STG if S['tests']['Delta_ridge_pool'][stg]['above_q95'] and S['tests']['Delta_ridge_pool'][stg]['p_z_holm'] < 0.05]
    decm = [stg for stg in STG if S['tests']['Delta_mlp_pool'][stg]['above_q95'] and S['tests']['Delta_mlp_pool'][stg]['p_z_holm'] < 0.05]
    dect = [stg for stg in STG if S['tests']['Delta_ridge_turnT'][stg]['above_q95'] and S['tests']['Delta_ridge_turnT'][stg]['p_z_holm'] < 0.05]
    S['stages_Delta_above_band'] = dict(ridge_pool=dec, mlp_pool=decm, ridge_turn_transfer=dect)
    S['swap_eligible_stages'] = [s for s in SWAP_ST if s in set(dec) | set(decm) | set(dect)]
    if SW: S['swap'] = SW['res']
    (OUT / f'e41_summary{tag}.json').write_text(json.dumps(clean(S), indent=1)); log('combine:', json.dumps(clean(S['stages_Delta_above_band'])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('main', 'ctrl', 'swap', 'combine'))
    ap.add_argument('--seeds', default='4101,4102'); ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--nets', default='u0-9,p0-9'); ap.add_argument('--label', default='up')
    a = ap.parse_args(); a.seeds = tuple(int(s) for s in a.seeds.split(','))
    tag = ('' if a.seeds == (4101, 4102) else f'_s{a.seeds[0]}') + ('_smoke' if a.smoke else '')
    OUT.mkdir(parents=True, exist_ok=True)
    dict(main=arm_main, ctrl=arm_ctrl, swap=arm_swap, combine=arm_combine)[a.arm](a, tag)
