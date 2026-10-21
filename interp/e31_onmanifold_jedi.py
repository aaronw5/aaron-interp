"""E31 (claims A4, A4w, A15, A14; review weakness "no C2ST on any loop-A edit"): are the loop-A input edits on the jet manifold, and
do their headline numbers hold on edited jets that a classifier cannot tell from real jets?  Plus natural (unedited) jet pairs that
differ in jet mass m at matched M2, and in M2 at matched m.

Network: official JEDI-linear 3-feature n64 (inputs pT, deta, dphi about the stored jet axis), exact forward pass (interp.jedi_core).
Notation (per jet, sums over real particles, r_i = sqrt(deta_i^2 + dphi_i^2) about the stored axis):
  M0 = sum pT, M1 = sum pT (deta, dphi) (vector), M2 = sum pT r^2, m = invariant mass of the massless constituents (four-vector sum).
Samples: edits on sample(1600, 3101) = 8,000 jets; natural pool sample(3000, 3102) minus jets also in the edit sample.

Edits (all on the 8,000 jets; 'ok' = jets where the edit applies):
  shift    rigid translation deta += 0.06 rad of every real particle (E6 / X8 / E27w arm A).
  turn5/10 E27w arm B 'hard14': the 14 hardest particles turned about the stored axis toward the azimuth of M1,
           theta_i -> theta_i + t wrap(phi_M1 - theta_i), t per jet solved so m falls 5 / 10 %; pT and r of every particle fixed, so M0,
           M2, all sum pT r^beta, ranks and slots fixed (interp.e27w_mass_vs_m2_jedi.solve_B).
  split0/13/21  E28w exact collinear split at pT rank k (z 0.5; two particles at the same (deta, dphi), pT/2 each, re-sorted; jets
           with N >= k + 2); M0, M1, M2 and the four-momentum unchanged (interp.e28w_split_route_jedi.split_ext).
  soft     one particle of pT 0.5 GeV added at the r of a random real particle and a uniform random azimuth (E28 soft_add; N <= 63).
  boost    exact boost along the jet momentum, |P| x 0.9 at fixed jet mass (pT ~ x 0.9, angles ~ x 1/0.9; interp.h12_boost_panel.boost,
           as E26).
Reference real changes (dilations: every real particle's (deta, dphi) x a; jets that differ in mass the way real heavier jets do):
  dil_M2 (for shift):  a = sqrt(M2'/M2) per jet with M2' the shifted jet's M2 (same Delta M2 about the stored axis).
  dil_m5 / dil_m10 (for turn5 / turn10): a per jet so the true mass equals the turned jet's mass (same Delta m).
  dil_B (for boost): a = sqrt(M2_boosted / M2) per jet (same Delta log M2 as the boost).
  dil5 (for split, soft): a = 1.05 for every jet (a 5 % real mass change; splits and soft additions do not change m or M2).
Null edits (should not change the call; matched displacement):
  rot_<edit> for shift / turn / boost: rigid rotation of the whole jet about the stored axis by an angle (random sign) such that the
           pT-weighted mean displacement of the particles equals the edit's; rot_split<k>: the split particle rotated about the axis
           by a uniform random azimuth without splitting (E28); soft_ctrl: the same 0.5 GeV added to the hardest particle.
C2ST (classifier two-sample test): sklearn HistGradientBoostingClassifier (max_iter 200, lr 0.1, 31 leaves, no early stopping) on
  feature set 'obs' = (m, jet pT, N, tau21, tau32, D2, girth = sum pT r / M0, pT^D = sqrt(sum pT^2) / M0, |M1|^2 / (M0 M2)) or
  'plist' = (pT, deta, dphi) of the 30 hardest particles (zeros for missing).  tau_N: beta = 1, axes from hardest-particle seeding
  plus greedy pT dR^2 seeding and 8 pT-weighted Lloyd iterations; D2 = e3 / e2^3 with beta = 1 energy correlators (all particles).
  The ok jets are split at random into halves A, B; classifier 1 separates A unedited (label 0) from B edited (label 1), classifier 2
  B unedited from A edited; each with 5-fold stratified CV out-of-fold (OOF) scores. So every ok jet has an OOF score for its
  unedited version (s0) and for its edited version (sE), each from a classifier that never saw that jet.  AUC = pooled AUC of sE
  (positives) vs s0 (negatives).  95 % CI: 1,000 bootstrap resamples of jets (pairs (sE, s0) kept together).  H0 (edited and
  unedited jets come from the same distribution): AUC = 0.5; p from 1,000 within-jet label swaps of (sE, s0); also a retrain null
  (the same procedure with both halves unedited, 3 seeds per feature set).  Delta AUC = AUC(edit) - AUC(its reference dilation)
  and - AUC(its null edit) with paired bootstrap CIs on the common ok jets.
On-manifold subset (per edit, per feature set): ok jets whose edited version's OOF score sE lies inside the central half of the
  unedited jets' OOF scores, [q25, q75] of s0 (so 50 % of unedited jets fall in the band; an indistinguishable edit keeps ~50 %);
  'both' = inside for obs and plist; 'obs8_plist' = inside for obs8 and plist.  (Run 1 used |sE - 0.5| < median |s0 - 0.5|, which
  keeps every jet when the classifier separates perfectly, since both scores then sit at |s - 0.5| = 0.5; replaced.)
  Feature set 'obs8' = 'obs' without |M1|^2/(M0 M2), to test whether the centroid offset is the only observable that separates.
Headline numbers recomputed on each subset (and on all ok jets):
  R_h4 = mean Delta(h4 log-m read-out) under the edit / mean Delta under its reference dilation on the same jets (read-out: ridge of
       log m on h4, fit on unedited jets of half A of the edit sample, interp.x10_das_jedi.RidgeR); also R_h4 on half-B jets only.
  W->Z = share of W-called jets called Z after the edit; Z->W likewise; calls changed = share of jets whose call changes.
  McNemar (exact) edit vs its null edit on the same jets for calls changed, W->Z and Z->W indicators; 95 % CIs from 1,000 Poisson
  jet-bootstrap weights.  q->g / g->q counts for splits and soft additions.
Natural pairs (pool, unedited jets): greedy unique nearest-neighbour pairs (same N; |Delta log| <= 0.01 in the matched variables):
  arm 'm_differs'  matched M0, M2, jet pT, N; |Delta log m| >= 0.03 (m differs only through |M1|, i.e. off-centre jets);
  arm 'M2_differs' matched M0, m, jet pT, N; |Delta log M2| >= 0.03;
  control 'all_matched' matched M0, M2, m, jet pT, N; control 'm_only' matched m only (no N grouping), oriented by larger M2.
  Each pair oriented so the second jet has the larger differing variable; Delta ro = read-out(second) - read-out(first).
  Calibration k = mean Delta ro of a real 5 % dilation of the pool jets / log 1.05 (read-out change per unit Delta log m when m and
  sqrt(M2) move together).  R_nat_m = mean Delta ro / (k mean Delta log m) (1 = read-out follows m; 0 = follows M2 at fixed M0);
  R_nat_M2 = mean Delta ro / (k/2 mean Delta log M2) (1 = follows sqrt(M0 M2); 0 = follows m).  Bootstrap over pairs; sign-flip p.
  Same for the W - Z logit margin.  Pooled regression of Delta ro on (Delta log M0, Delta log M2, Delta log m) over all pairs.
Run: ./job.sh e31-run 1 2 'python -m interp.e31_onmanifold_jedi'   (--smoke N: N jets per class)
Writes $AI_RESULTS/jedi/e31_onmanifold[_smoke].json and e31_examples[_smoke].npz.
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
T0 = time.time(); G_, Q_, W_, Z_ = 0, 1, 2, 3
CL = ['g', 'q', 'W', 'Z', 't']
OBS = ['m', 'jpt', 'N', 'tau21', 'tau32', 'D2', 'girth', 'ptD', 'off']


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


# ------------------------------------------------------------------------------------------------ observables
def taus(pt, P, iters=8, ch=1000):
    J = len(pt); out = np.zeros((J, 3))
    for i in range(0, J, ch):
        p, X = pt[i:i + ch], P[i:i + ch]; n = len(p); rr = np.arange(n); M0 = np.maximum(p.sum(1), 1e-12)
        for K in (1, 2, 3):
            ax = np.zeros((n, K, 2)); ax[:, 0] = X[:, 0]
            for k in range(1, K):
                d2 = ((X[:, :, None, :] - ax[:, None, :k, :]) ** 2).sum(-1).min(-1); j = (p * d2).argmax(1); ax[:, k] = X[rr, j]
            for _ in range(iters):
                d = ((X[:, :, None] - ax[:, None]) ** 2).sum(-1); a = d.argmin(-1)
                oh = (a[..., None] == np.arange(K)) * p[..., None]; w = oh.sum(1)
                new = np.einsum('jnk,jnc->jkc', oh, X) / np.maximum(w, 1e-12)[..., None]
                ax = np.where(w[..., None] > 0, new, ax)
            d = np.sqrt(((X[:, :, None] - ax[:, None]) ** 2).sum(-1)).min(-1)
            out[i:i + ch, K - 1] = (p * d).sum(1) / M0
    return out


def d2_obs(pt, P, ch=500):
    J = len(pt); out = np.zeros(J)
    for i in range(0, J, ch):
        p, X = pt[i:i + ch], P[i:i + ch]; z = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        R = np.sqrt(((X[:, :, None] - X[:, None]) ** 2).sum(-1))
        e2 = 0.5 * np.einsum('ji,jik,jk->j', z, R, z); A = z[:, :, None] * R
        e3 = np.einsum('jii->j', A @ A @ A) / 6
        out[i:i + ch] = e3 / np.maximum(e2 ** 3, 1e-30)
    return out


def jet_feats(x):
    x = np.asarray(x, np.float64); pt, de, dp = x[..., 0], x[..., 1], x[..., 2]; real = pt > 0
    pt = np.where(real, pt, 0.0); de = np.where(real, de, 0.0); dp = np.where(real, dp, 0.0)
    M0 = pt.sum(1); r2 = de ** 2 + dp ** 2; M2 = (pt * r2).sum(1); M1x, M1y = (pt * de).sum(1), (pt * dp).sum(1)
    E = (pt * np.cosh(de)).sum(1); px = (pt * np.cos(dp)).sum(1); py = (pt * np.sin(dp)).sum(1); pz = (pt * np.sinh(de)).sum(1)
    m = np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0)); P = np.stack([de, dp], -1); t = taus(pt, P) + 1e-12
    return dict(m=m, jpt=np.hypot(px, py), N=real.sum(1).astype(float), tau21=t[:, 1] / t[:, 0], tau32=t[:, 2] / t[:, 1],
                D2=d2_obs(pt, P), girth=(pt * np.sqrt(r2)).sum(1) / np.maximum(M0, 1e-12), ptD=np.sqrt((pt ** 2).sum(1)) / np.maximum(M0, 1e-12),
                off=(M1x ** 2 + M1y ** 2) / np.maximum(M0 * M2, 1e-30), M0=M0, M2=M2)


def featsets(x):
    F = jet_feats(x); Fo = np.nan_to_num(np.stack([F[k] for k in OBS], 1), nan=0.0, posinf=1e6, neginf=-1e6)
    Fp = np.asarray(x[:, :30, :], np.float64).reshape(len(x), -1)
    return dict(obs=Fo, obs8=Fo[:, :8], plist=Fp), F


# ------------------------------------------------------------------------------------------------ C2ST
def fast_auc(pos, neg):
    from scipy.stats import rankdata
    s = np.concatenate([pos, neg]); rk = rankdata(s); n1, n0 = len(pos), len(neg)
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oof_scores(X, lab, seed, folds):
    s = np.zeros(len(lab)); rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(lab == c) for c in (0, 1)]; fold = np.zeros(len(lab), int)
    for I in idx: fold[rng.permutation(I)] = np.arange(len(I)) % folds
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier as HGB
        mk = lambda: HGB(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False, random_state=seed)
        kind = 'HistGBDT'
    except ImportError:   # numpy logistic regression fallback (standardised features, 300 Newton-free gradient steps)
        kind = 'numpy_logreg'
        class LR:
            def fit(self, X, y):
                self.mu, self.sd = X.mean(0), X.std(0) + 1e-9; Z = (X - self.mu) / self.sd; w = np.zeros(Z.shape[1]); b = 0.0
                for _ in range(300):
                    p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - y; w -= 0.5 * (Z.T @ g / len(y) + 1e-3 * w); b -= 0.5 * g.mean()
                self.w, self.b = w, b; return self
            def predict_proba(self, X):
                p = 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b))); return np.stack([1 - p, p], 1)
        mk = LR
    for f in range(folds):
        te = fold == f; s[te] = mk().fit(X[~te], lab[~te]).predict_proba(X[te])[:, 1]
    return s, kind


def c2st(F0, F1, ok, seed, folds, NB, NP):
    """returns dict(auc, ci, p_swap, auc_assign), sE, s0 (J,) with nan outside the scored jets"""
    J = len(F0); rng = np.random.default_rng(seed); idx = rng.permutation(np.flatnonzero(ok)); h = len(idx) // 2
    A, B = np.sort(idx[:h]), np.sort(idx[h:2 * h]); sE = np.full(J, np.nan); s0 = np.full(J, np.nan); aa = []; kind = None
    for U, E in ((A, B), (B, A)):
        X = np.concatenate([F0[U], F1[E]]); lab = np.r_[np.zeros(len(U)), np.ones(len(E))]
        s, kind = oof_scores(X, lab, seed, folds); s0[U] = s[:len(U)]; sE[E] = s[len(U):]; aa.append(fast_auc(s[len(U):], s[:len(U)]))
    sc = np.flatnonzero(np.isfinite(sE) & np.isfinite(s0)); a = fast_auc(sE[sc], s0[sc])
    bs = [fast_auc(sE[sc[b]], s0[sc[b]]) for b in (rng.integers(0, len(sc), len(sc)) for _ in range(NB))]
    nul = []
    for _ in range(NP):
        sw = rng.random(len(sc)) < 0.5; p_, n_ = np.where(sw, s0[sc], sE[sc]), np.where(sw, sE[sc], s0[sc]); nul.append(fast_auc(p_, n_))
    nul = np.array(nul)
    return dict(auc=a, ci=qci(bs), p_swap=float((1 + (np.abs(nul - 0.5) >= abs(a - 0.5)).sum()) / (1 + NP)), null_q95=float(np.quantile(nul, .95)),
                auc_assign=aa, n=int(len(sc)), classifier=kind), sE, s0


def dauc(a, b, NB, seed):
    """paired bootstrap of AUC(a) - AUC(b) over the jets scored in both C2STs; a, b = (sE, s0)"""
    sc = np.flatnonzero(np.isfinite(a[0]) & np.isfinite(a[1]) & np.isfinite(b[0]) & np.isfinite(b[1]))
    if len(sc) < 10: return None
    rng = np.random.default_rng(seed); f = lambda I: fast_auc(a[0][I], a[1][I]) - fast_auc(b[0][I], b[1][I])
    return dict(value=f(sc), ci=qci([f(sc[rng.integers(0, len(sc), len(sc))]) for _ in range(NB)]), n=int(len(sc)))


# ------------------------------------------------------------------------------------------------ edits
def dilate(x, a):
    e = np.array(x, np.float64); rl = e[..., 0] > 0; a = np.broadcast_to(np.asarray(a, np.float64), (len(x),))[:, None]
    for j in (1, 2): e[..., j] = np.where(rl, e[..., j] * a, e[..., j])
    return e.astype(np.float32)


def rotate_rigid(x, D, rng):
    """rigid rotation about the stored axis; angle per jet so the pT-weighted mean displacement equals D; random sign"""
    e = np.array(x, np.float64); pt = e[..., 0]; rl = pt > 0; r = np.hypot(e[..., 1], e[..., 2])
    rbar = (pt * r).sum(1) / np.maximum(pt.sum(1), 1e-12)
    th = 2 * np.arcsin(np.clip(D / np.maximum(2 * rbar, 1e-12), 0, 1)) * rng.choice([-1.0, 1.0], len(D))
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]; de, dp = e[..., 1].copy(), e[..., 2].copy()
    e[..., 1] = np.where(rl, c * de - s * dp, de); e[..., 2] = np.where(rl, s * de + c * dp, dp)
    return e.astype(np.float32)


def ptw_disp(x0, x1):
    """pT-weighted (unedited pT) mean distance moved by each slot's particle (for same-slot edits)"""
    a, b = np.asarray(x0, np.float64), np.asarray(x1, np.float64); w = a[..., 0] * (a[..., 0] > 0)
    return (w * np.hypot(b[..., 1] - a[..., 1], b[..., 2] - a[..., 2])).sum(1) / np.maximum(w.sum(1), 1e-12)


def build_edits(x, rng, log_):
    from interp.e27w_mass_vs_m2_jedi import solve_B, to_x, mass_p
    from interp.e28w_split_route_jedi import split_ext, rotate as rot_split, nreal
    from interp.e28_irc_split_jedi import soft_add
    from interp.h12_boost_panel import boost, sortT, hls_prep, displacement, rotate as rot_T
    from interp.h1_irc_panel import hls_table
    from interp.x10_das_jedi import mass
    J = len(x); x64 = np.asarray(x, np.float64); pt = x64[..., 0]; real = pt > 0
    r = np.hypot(x64[..., 1], x64[..., 2]); th = np.arctan2(x64[..., 2], x64[..., 1]); N = nreal(x); allk = np.ones(J, bool)
    M2 = (pt * r ** 2).sum(1); m0 = mass(x)
    E, INFO = {}, {}
    # shift
    xs = np.array(x, np.float32); xs[..., 1] += np.float32(0.06) * real
    M2s = (pt * (xs[..., 1].astype(np.float64) ** 2 + xs[..., 2].astype(np.float64) ** 2) * real).sum(1)
    E['shift'] = (xs, allk); E['dil_M2'] = (dilate(x, np.sqrt(M2s / M2)), allk)
    E['rot_shift'] = (rotate_rigid(x, ptw_disp(x, xs), rng), allk)
    INFO['shift'] = dict(median_dlogM2=float(np.median(np.log(M2s / M2))), median_dlogm=float(np.median(np.log(mass(xs) / m0))))
    log_('shift built')
    # azimuth turn of the 14 hardest
    M1 = np.stack([(pt * r * np.cos(th)).sum(1), (pt * r * np.sin(th)).sum(1)], -1); phu = np.arctan2(M1[:, 1], M1[:, 0])
    rank = np.argsort(np.argsort(-np.where(real, pt, -1), 1), 1); sub = real & (rank < 14)
    for f in (0.05, 0.10):
        k = f'turn{int(round(f * 100))}'; d = np.where(sub, wrap(phu[:, None] - th), 0.0); tB, okB = solve_B(pt, r, th, d, f)
        xB = to_x(x, real, r, th + tB[:, None] * d); mB = mass(xB); aD = mB / np.maximum(m0, 1e-9)
        for _ in range(4):
            aD = aD * mB / np.maximum(mass(dilate(x, aD)), 1e-9)
        ok = okB & np.isfinite(mB) & (m0 > 0)
        E[k] = (xB, ok); E[f'dil_m{int(round(f * 100))}'] = (dilate(x, aD), ok); E[f'rot_{k}'] = (rotate_rigid(x, ptw_disp(x, xB), rng), ok)
        INFO[k] = dict(feasible=int(ok.sum()), median_dlogm=float(np.median(np.log(mB[ok] / m0[ok]))),
                       max_abs_dr=float(np.max(np.abs(np.hypot(xB[..., 1].astype(np.float64), xB[..., 2]) - r)[ok])),
                       median_dlogM2=float(np.median(np.log((pt * (xB[..., 1].astype(np.float64) ** 2 + xB[..., 2].astype(np.float64) ** 2)).sum(1)[ok] / M2[ok]))))
        log_(f'{k} built: {INFO[k]}')
    # collinear splits (z 0.5)
    for kr in (0, 13, 21):
        slot = np.where(N >= kr + 2, kr, -1); ok = slot >= 0; Ys, _, _ = split_ext(x, slot, 0.5)
        E[f'split{kr}'] = (Ys, ok); E[f'rot_split{kr}'] = (rot_split(x, slot, rng), ok)
        INFO[f'split{kr}'] = dict(applicable=int(ok.sum()))
    # one soft particle
    Ysoft, oks = soft_add(x, 1, rng, pt=0.5); xc = np.array(x, np.float32); xc[:, 0, 0] += np.float32(0.5)
    E['soft'] = (np.asarray(Ysoft, np.float32), oks); E['soft_ctrl'] = (xc, oks); INFO['soft'] = dict(applicable=int(oks.sum()))
    E['dil5'] = (dilate(x, 1.05), allk)
    log_('splits / soft built')
    # boost at fixed mass
    T = hls_table(x); B, _ = boost(T, 0.9); xb = hls_prep(sortT(B)); Dp = displacement(T, B)
    Rt, _ = rot_T(T, Dp, rng); xr = hls_prep(sortT(Rt))
    M2b = (np.asarray(xb[..., 0], np.float64) * (xb[..., 1].astype(np.float64) ** 2 + xb[..., 2].astype(np.float64) ** 2)).sum(1)
    E['boost'] = (xb, allk); E['rot_boost'] = (xr, allk); E['dil_B'] = (dilate(x, np.sqrt(M2b / M2)), allk)
    INFO['boost'] = dict(max_abs_dlogm=float(np.max(np.abs(np.log(np.maximum(mass(xb), 1e-9) / np.maximum(m0, 1e-9)))[m0 > 1])),
                         median_dlogM2=float(np.median(np.log(M2b / M2))), median_dlogM0=float(np.median(np.log(xb[..., 0].sum(1) / pt.sum(1)))),
                         roundtrip_max_abs=float(np.abs(hls_prep(sortT(T)) - x).max()))
    log_(f'boost built: {INFO["boost"]}')
    return E, INFO


# edit -> (reference dilation, null edit, headline metric)
PLAN = dict(shift=('dil_M2', 'rot_shift', 'wz'), turn5=('dil_m5', 'rot_turn5', 'zw'), turn10=('dil_m10', 'rot_turn10', 'zw'),
            split0=('dil5', 'rot_split0', 'ch'), split13=('dil5', 'rot_split13', 'ch'), split21=('dil5', 'rot_split21', 'ch'),
            soft=('dil5', 'soft_ctrl', 'ch'), boost=('dil_B', 'rot_boost', 'ch'))


# ------------------------------------------------------------------------------------------------ natural pairs
def make_pairs(L, N, keys, diff, thr, tol, group_N, seed):
    rng = np.random.default_rng(seed); J = len(N); used = np.zeros(J, bool); P = []
    groups = [np.flatnonzero(N == n) for n in np.unique(N)] if group_N else [np.arange(J)]
    for G in groups:
        if len(G) < 2: continue
        Lg = {k: L[k][G] for k in L}
        for ii in rng.permutation(len(G)):
            i = G[ii]
            if used[i]: continue
            cand = ~used[G]; cand[ii] = False
            dist = np.max(np.stack([np.abs(Lg[k] - Lg[k][ii]) for k in keys]), 0)
            okc = cand & (dist <= tol)
            if diff is not None: okc &= np.abs(Lg[diff] - Lg[diff][ii]) >= thr
            if not okc.any(): continue
            j = G[np.flatnonzero(okc)[np.argmin(dist[okc])]]; used[i] = used[j] = True
            P.append((i, j))
    return np.array(P, int).reshape(-1, 2)


def main(a):
    from interp.jedi_core import Jedi, sample
    from interp.x10_das_jedi import clean, RidgeR, mass
    from interp.e28w_split_route_jedi import mcnemar, holm
    sm = a.smoke; pc = sm or 1600; ppc = sm or 3000; NB = 200 if sm else 1000; NP = 200 if sm else 1000; folds = 5
    OUT.mkdir(parents=True, exist_ok=True); tag = 'e31_onmanifold' + ('_smoke' if sm else ''); JS = OUT / f'{tag}.json'
    R = dict(samples=dict(edit=f'sample(per_class={pc}, seed=3101)', pool=f'sample(per_class={ppc}, seed=3102) minus overlap'), plan=PLAN)
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    net = Jedi(64)

    def fwd(xx): return net.run(xx, keep=('logits', 'h4'), batch=1000)

    x, y, _ = sample(pc, 3101); J = len(y)
    B0 = fwd(x); c0 = B0['logits'].argmax(1); marg0 = B0['logits'][:, Z_] - B0['logits'][:, W_]
    R['N'] = J; R['acc'] = float((c0 == y).mean())
    log(f'edit sample N {J}: accuracy {R["acc"]:.4f} (expected 0.806-0.813)')
    logm0 = np.log(np.maximum(mass(x), 1e-3)); pr = np.random.default_rng(3103).permutation(J); hA, hB = np.sort(pr[:J // 2]), np.sort(pr[J // 2:])
    nf = 2 * len(hA) // 3; RO = RidgeR(B0['h4'], logm0, hA[:nf], hA[nf:]); ro0 = RO(B0['h4'])
    R['readout_r2_halfB'] = float(1 - ((ro0[hB] - logm0[hB]) ** 2).mean() / logm0[hB].var()); isB = np.zeros(J, bool); isB[hB] = True
    log(f'h4 log-m read-out held-out R2 {R["readout_r2_halfB"]:.3f}'); save()

    rng = np.random.default_rng(3104); E, INFO = build_edits(x, rng, log); R['edit_info'] = INFO
    RESP = {}
    for k, (xe, ok) in E.items():
        P = fwd(xe); ce = P['logits'].argmax(1)
        RESP[k] = dict(call=ce, ch=ce != c0, wz=(c0 == W_) & (ce == Z_), zw=(c0 == Z_) & (ce == W_), dro=RO(P['h4']) - ro0,
                       dmarg=P['logits'][:, Z_] - P['logits'][:, W_] - marg0)
    log('forward passes done: ' + ', '.join(f'{k} ch {RESP[k]["ch"][E[k][1]].mean():.4f}' for k in E)); save()

    # ---------------------------------------------------------------- C2ST
    F0, Fu = featsets(x); FE = {}
    t = time.time()
    for k, (xe, ok) in E.items(): FE[k] = featsets(xe)[0]
    log(f'features done ({time.time() - t:.0f} s)')
    R['feature_means_unedited'] = {o: float(np.mean(Fu[o])) for o in OBS}
    R['c2st'] = {}; SC = {}
    for fs in ('obs', 'obs8', 'plist'):
        R['c2st'][fs] = {}
        nl = []
        for s in range(3):
            res, _, _ = c2st(F0[fs], F0[fs], np.ones(J, bool), 3200 + s, folds, 200, 200); nl.append(res['auc'])
        R['c2st'][fs]['_retrain_null_unedited_vs_unedited'] = nl
        log(f'C2ST {fs}: retrain null AUCs {nl}')
        for i, (k, (xe, ok)) in enumerate(E.items()):
            t = time.time(); res, sE, s0 = c2st(F0[fs], FE[k][fs], ok, 3110 + i, folds, NB, NP)
            R['c2st'][fs][k] = res; SC[(fs, k)] = (sE, s0)
            log(f'C2ST {fs} {k} ({time.time() - t:.0f} s): AUC {res["auc"]:.4f} {res["ci"]} p {res["p_swap"]:.4f} ({res["classifier"]})')
            save()
        R['c2st'][fs]['_delta'] = {}
        for k, (ref, nul, _) in PLAN.items():
            R['c2st'][fs]['_delta'][k] = dict(minus_ref=dauc(SC[(fs, k)], SC[(fs, ref)], NB, 1), minus_null=dauc(SC[(fs, k)], SC[(fs, nul)], NB, 2),
                                              ref=ref, null=nul)
        save()

    # ---------------------------------------------------------------- on-manifold subsets and headline numbers
    Wb = np.random.default_rng(3105).poisson(1.0, (NB, J)).astype(np.float32)
    bm = lambda w, v: (w @ v) / np.maximum(w.sum(1), 1e-12)

    def ratio(v, vr, m):
        if m.sum() < 5 or abs(vr[m].mean()) < 1e-12: return None, [None, None]
        w = Wb[:, m]; return float(v[m].mean() / vr[m].mean()), qci(bm(w, v[m]) / bm(w, vr[m]))

    def share(v, m):
        if m.sum() == 0: return None, [None, None]
        return float(v[m].mean()), qci(bm(Wb[:, m], v[m].astype(np.float32)))

    def headline(k, K):
        ref, nul, hk = PLAN[k]; e, d, n = RESP[k], RESP[ref], RESP[nul]
        o = dict(n=int(K.sum()), n_by_class=np.bincount(y[K], minlength=5).tolist(), n_W_called=int((K & (c0 == W_)).sum()), n_Z_called=int((K & (c0 == Z_)).sum()))
        o['R_h4'], o['R_h4_ci'] = ratio(e['dro'], d['dro'], K); o['R_h4_halfB'], o['R_h4_halfB_ci'] = ratio(e['dro'], d['dro'], K & isB)
        o['R_h4_null'], o['R_h4_null_ci'] = ratio(n['dro'], d['dro'], K)
        o['R_marg_W'], o['R_marg_W_ci'] = ratio(e['dmarg'], d['dmarg'], K & (c0 == W_)); o['R_marg_Z'], o['R_marg_Z_ci'] = ratio(e['dmarg'], d['dmarg'], K & (c0 == Z_))
        for nm, r_ in (('edit', e), ('ref', d), ('null', n)):
            o[f'calls_changed_{nm}'], o[f'calls_changed_{nm}_ci'] = share(r_['ch'], K)
            o[f'WZ_{nm}'], o[f'WZ_{nm}_ci'] = share(r_['wz'], K & (c0 == W_)); o[f'ZW_{nm}'], o[f'ZW_{nm}_ci'] = share(r_['zw'], K & (c0 == Z_))
        dd = (e['ch'].astype(np.float32) - n['ch'].astype(np.float32))
        o['calls_changed_edit_minus_null'], o['calls_changed_edit_minus_null_ci'] = (float(dd[K].mean()), qci(bm(Wb[:, K], dd[K]))) if K.any() else (None, [None, None])
        sel = dict(ch=K, wz=K & (c0 == W_), zw=K & (c0 == Z_))
        o['mcnemar_vs_null'] = {m_: mcnemar(e[m_][sel[m_]], n[m_][sel[m_]]) for m_ in sel}
        o['mcnemar_vs_ref'] = {m_: mcnemar(e[m_][sel[m_]], d[m_][sel[m_]]) for m_ in sel}
        o['q_to_g'] = int((K & (c0 == Q_) & (e['call'] == G_)).sum()); o['g_to_q'] = int((K & (c0 == G_) & (e['call'] == Q_)).sum())
        o['median_dro_edit'] = float(np.median(e['dro'][K])) if K.any() else None
        o['headline_metric'] = hk
        return o

    R['onmanifold'] = {}; pv = []
    for k in PLAN:
        ok = E[k][1]; R['onmanifold'][k] = {}; band = {}
        for fs in ('obs', 'obs8', 'plist'):
            sE, s0 = SC[(fs, k)]; sc = np.isfinite(sE) & np.isfinite(s0); lo, hi = np.quantile(s0[sc], [0.25, 0.75])
            band[fs] = sc & (sE >= lo) & (sE <= hi)
            R['onmanifold'][k][f'band_{fs}'] = [float(lo), float(hi)]
            R['onmanifold'][k][f'kept_frac_{fs}'] = float(band[fs].sum() / max(sc.sum(), 1))
            R['onmanifold'][k][f'unedited_in_band_frac_{fs}'] = float(((s0[sc] >= lo) & (s0[sc] <= hi)).mean())
        scored = np.isfinite(SC[('obs', k)][0]) & np.isfinite(SC[('plist', k)][0])
        for nm, K in (('all', ok), ('scored', ok & scored), ('obs', ok & band['obs']), ('obs8', ok & band['obs8']), ('plist', ok & band['plist']),
                      ('both', ok & band['obs'] & band['plist']), ('obs8_plist', ok & band['obs8'] & band['plist'])):
            R['onmanifold'][k][nm] = headline(k, K)
            hk = PLAN[k][2]; pv.append(R['onmanifold'][k][nm]['mcnemar_vs_null'][hk]['p'])
        h = R['onmanifold'][k]
        log(f'{k}: ' + '; '.join(f'{nm} n {h[nm]["n"]} R_h4 {h[nm]["R_h4"]} {PLAN[k][2]} edit {h[nm][{"wz": "WZ_edit", "zw": "ZW_edit", "ch": "calls_changed_edit"}[PLAN[k][2]]]} '
                                 f'null {h[nm][{"wz": "WZ_null", "zw": "ZW_null", "ch": "calls_changed_null"}[PLAN[k][2]]]}' for nm in ('all', 'obs', 'obs8', 'plist', 'both', 'obs8_plist')))
        save()
    hl = holm(pv); i = 0
    for k in PLAN:
        for nm in ('all', 'scored', 'obs', 'obs8', 'plist', 'both', 'obs8_plist'):
            R['onmanifold'][k][nm]['holm_headline_vs_null'] = hl[i]; i += 1
    R['holm_n_tests'] = len(pv)

    # examples for the figure: one true-W jet called W with N near the median of jets where every edit applies
    allok = np.all(np.stack([E[k][1] for k in E]), 0); cand = np.flatnonzero(allok & (y == W_) & (c0 == W_))
    if len(cand) == 0: cand = np.flatnonzero(allok)
    if len(cand):
        Nr = (x[:, :, 0] > 0).sum(1); ex = int(cand[np.argsort(Nr[cand])[len(cand) // 2]])
        np.savez_compressed(OUT / f'e31_examples{"_smoke" if sm else ""}.npz', index=ex, x0=x[ex], label=y[ex], call0=c0[ex],
                            **{f'x_{k}': E[k][0][ex] for k in E}, **{f'call_{k}': RESP[k]['call'][ex] for k in E})
        R['example'] = dict(index=ex, label=int(y[ex]), call0=int(c0[ex]), calls={k: int(RESP[k]['call'][ex]) for k in E},
                            m0=float(Fu['m'][ex]), N=int(Nr[ex]))
    save()

    # ---------------------------------------------------------------- natural pairs
    t = time.time(); xp, yp, _ = sample(ppc, 3102)
    hf = {hash(v.tobytes()) for v in x}; kp = np.array([hash(v.tobytes()) not in hf for v in xp]); xp, yp = xp[kp], yp[kp]
    Bp = fwd(xp); rop = RO(Bp['h4']); mp = Bp['logits'][:, Z_] - Bp['logits'][:, W_]; cp = Bp['logits'].argmax(1)
    Bd = fwd(dilate(xp, 1.05)); drd = RO(Bd['h4']) - rop; dmd = Bd['logits'][:, Z_] - Bd['logits'][:, W_] - mp
    Fp = jet_feats(xp); Np = Fp['N'].astype(int)
    NAT = dict(N_pool=int(len(yp)), overlap_dropped=int((~kp).sum()), acc_pool=float((cp == yp).mean()),
               k_ro_per_dlogm=float(drd.mean() / np.log(1.05)), k_marg_per_dlogm=float(dmd.mean() / np.log(1.05)),
               off_quantiles=np.quantile(Fp['off'], [.5, .9, .99]).tolist(),
               dlogm_offcentre_quantiles=dict(zip(['q50', 'q90', 'q99', 'q999', 'max'], np.quantile(-0.5 * np.log(np.maximum(1 - Fp['off'], 1e-12)), [.5, .9, .99, .999, 1.0]).tolist())),
               log_m_minus_log_sqrtM0M2_quantiles=np.quantile(np.log(np.maximum(Fp['m'], 1e-3)) - 0.5 * np.log(np.maximum(Fp['M0'] * Fp['M2'], 1e-30)), [.01, .5, .99]).tolist())
    rb = np.random.default_rng(3106); Ib = [rb.integers(0, len(yp), len(yp)) for _ in range(NB)]
    NAT['k_ro_ci'] = qci([drd[I].mean() / np.log(1.05) for I in Ib]); kro = NAT['k_ro_per_dlogm']; kmg = NAT['k_marg_per_dlogm']
    Lg = dict(M0=np.log(Fp['M0']), M2=np.log(np.maximum(Fp['M2'], 1e-30)), m=np.log(np.maximum(Fp['m'], 1e-3)), jpt=np.log(np.maximum(Fp['jpt'], 1e-6)))
    log(f'pool N {len(yp)} acc {NAT["acc_pool"]:.4f}; k_ro {kro:.3f} {NAT["k_ro_ci"]}; |M1|^2/(M0M2) quantiles {NAT["off_quantiles"]}')
    ARMS = dict(m_differs=(['M0', 'M2', 'jpt'], 'm', 0.03, True), m_differs_2pc=(['M0', 'M2', 'jpt'], 'm', 0.02, True),
                M2_differs=(['M0', 'm', 'jpt'], 'M2', 0.03, True), all_matched=(['M0', 'M2', 'm', 'jpt'], None, 0, True),
                m_only=(['m'], 'M2', 0.0, False))
    allp = []
    for nm, (keys, diff, thr, gN) in ARMS.items():
        P = make_pairs(Lg, Np, keys, diff, thr, 0.01, gN, 3107)
        o = dict(match=keys, differs=diff, min_diff=thr, n_pairs=int(len(P)))
        if len(P) >= 3:
            i1, i2 = P[:, 0], P[:, 1]
            orient = Lg[diff] if diff else rb.random(len(yp))
            sw = orient[i2] < orient[i1]; a1, a2 = np.where(sw, i2, i1), np.where(sw, i1, i2)
            dro = rop[a2] - rop[a1]; dmg = mp[a2] - mp[a1]; dl = {k_: Lg[k_][a2] - Lg[k_][a1] for k_ in Lg}
            allp.append(np.c_[dro, dl['M0'], dl['M2'], dl['m']])
            o.update(mean_dro=float(dro.mean()), sd_dro=float(dro.std()), mean_dlog={k_: float(v.mean()) for k_, v in dl.items()},
                     max_abs_dlog_matched={k_: float(np.abs(dl[k_]).max()) for k_ in keys},
                     classes_first=np.bincount(yp[a1], minlength=5).tolist(), classes_second=np.bincount(yp[a2], minlength=5).tolist(),
                     same_class_frac=float((yp[a1] == yp[a2]).mean()), call_differs=float((cp[a1] != cp[a2]).mean()),
                     mean_dmarg=float(dmg.mean()))
            pb = np.random.default_rng(3108); Ip = [pb.integers(0, len(P), len(P)) for _ in range(NB)]
            o['mean_dro_ci'] = qci([dro[I].mean() for I in Ip])
            pM2 = kro * 0.5 * (dl['M0'] + dl['M2']); pm = kro * dl['m']
            o['pred_sqrtM0M2'] = float(pM2.mean()); o['pred_sqrtM0M2_ci'] = qci([pM2[I].mean() for I in Ip])
            o['pred_m'] = float(pm.mean()); o['pred_m_ci'] = qci([pm[I].mean() for I in Ip])
            o['obs_minus_pred_sqrtM0M2_ci'] = qci([(dro - pM2)[I].mean() for I in Ip]); o['obs_minus_pred_m_ci'] = qci([(dro - pm)[I].mean() for I in Ip])
            from interp.e27w_mass_vs_m2_jedi import signflip_p
            o['p_signflip_dro'] = signflip_p(dro, 10000 if not sm else 500, 4)
            if diff == 'm':
                den = kro * dl['m']; o['R_nat_m'] = float(dro.mean() / den.mean()); o['R_nat_m_ci'] = qci([dro[I].mean() / den[I].mean() for I in Ip])
                o['R_nat_m_marg'] = float(dmg.mean() / (kmg * dl['m']).mean())
                o['R_nat_m_marg_ci'] = qci([dmg[I].mean() / (kmg * dl['m'][I]).mean() for I in Ip])
            if diff == 'M2':
                den = 0.5 * kro * dl['M2']; o['R_nat_M2'] = float(dro.mean() / den.mean()); o['R_nat_M2_ci'] = qci([dro[I].mean() / den[I].mean() for I in Ip])
                o['R_nat_M2_marg'] = float(dmg.mean() / (0.5 * kmg * dl['M2']).mean())
                o['R_nat_M2_marg_ci'] = qci([dmg[I].mean() / (0.5 * kmg * dl['M2'][I]).mean() for I in Ip])
                if nm == 'm_only':
                    o['note'] = 'oriented by larger M2; M0 free, so dlog M0 ~ -dlog M2 at matched m'
        NAT[nm] = o; log(f'natural pairs {nm} ({time.time() - t:.0f} s): {json.dumps(clean({k_: v for k_, v in o.items() if not k_.startswith("classes")}))}')
        save()
    if allp:
        A_ = np.concatenate(allp); Xr = np.c_[A_[:, 1:], np.ones(len(A_))]; co = np.linalg.lstsq(Xr, A_[:, 0], rcond=None)[0]
        pb = np.random.default_rng(3109); cb = []
        for _ in range(NB):
            I = pb.integers(0, len(A_), len(A_)); cb.append(np.linalg.lstsq(Xr[I], A_[I, 0], rcond=None)[0])
        cb = np.array(cb)
        NAT['pooled_regression'] = dict(n_pairs=int(len(A_)), terms=['dlogM0', 'dlogM2', 'dlogm', 'const'], coef=co.tolist(),
                                        coef_ci=[qci(cb[:, i]) for i in range(4)], coef_over_k=(co[:3] / kro).tolist(),
                                        expected_M2_reading=[0.5, 0.5, 0.0], expected_m_reading=[0.0, 0.0, 1.0])
    R['natural_pairs'] = NAT; R['seconds'] = time.time() - T0; save()
    log(f'done, wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0)
    main(ap.parse_args())
