"""E19 shared helpers (numpy / scipy only; imported by interp/e19_axis_jedi.py and interp/e19_axis_part.py).

kt3:        exclusive kT clustering to 3 subjets (d_ij = min(pT_i², pT_j²)·ΔR², no beam distance; pT-scheme recombination:
            pT summed, (y, φ) pT-weighted), on the particles in the stored relative coordinates.
three_prong: per jet, the subjets, whether they are resolved (every subjet pT fraction ≥ ZMIN and every pairwise ΔR ≥ RMIN),
            the subjet that holds the new axis (the subjet of the particle nearest to the axis), and the predictions
            pred3   = ½ log(M0·M2 about that subjet's pT-weighted centroid) − ½ log(M0·M2 about the stored axis)  (particles)
            pred3pp = ½ log(Σ_j z_j θ_aj² / Σ_{i<j} z_i z_j θ_ij²)  (point prongs: the subjets as three massless points)
stats:      bootstrap CIs (mean, ratio of means), Wilson intervals, exact McNemar, one-sample t-test with log10 p, Holm.
"""
import numpy as np

ZMIN, RMIN = 0.10, 0.10
NBOOT = 2000


# ------------------------------------------------------------------------------------------------ clustering
def kt3(pt, y, p, n=3):
    """exclusive kT to n subjets; returns (labels per particle (len pt,), subjets (n, 3): pT, y, φ), subjets by pT descending"""
    N = len(pt)
    lab = np.arange(N)
    if N <= n:
        sj = np.stack([pt, y, p], 1).astype(np.float64)
        o = np.argsort(-sj[:, 0]); sj = sj[o]; inv = np.empty(N, int); inv[o] = np.arange(N)
        pad = np.zeros((n - N, 3)); return inv, np.concatenate([sj, pad])
    P = pt.astype(np.float64).copy(); Y = y.astype(np.float64).copy(); F = p.astype(np.float64).copy(); alive = np.ones(N, bool)
    owner = np.arange(N)
    def dist_row(i):
        d = np.minimum(P[i] ** 2, P ** 2) * ((Y - Y[i]) ** 2 + (F - F[i]) ** 2); d[~alive] = np.inf; d[i] = np.inf; return d
    D = np.minimum(P[:, None] ** 2, P[None, :] ** 2) * ((Y[:, None] - Y[None, :]) ** 2 + (F[:, None] - F[None, :]) ** 2)
    np.fill_diagonal(D, np.inf)
    for _ in range(N - n):
        i, j = divmod(int(np.argmin(D)), N)
        s = P[i] + P[j]; Y[i] = (P[i] * Y[i] + P[j] * Y[j]) / s; F[i] = (P[i] * F[i] + P[j] * F[j]) / s; P[i] = s
        alive[j] = False; owner[owner == j] = i; D[j, :] = np.inf; D[:, j] = np.inf
        d = dist_row(i); D[i, :] = d; D[:, i] = d
    roots = np.flatnonzero(alive); o = roots[np.argsort(-P[roots])]
    rank = {r: k for k, r in enumerate(o)}
    return np.array([rank[r] for r in owner]), np.stack([P[o], Y[o], F[o]], 1)


def three_prong(pt, de, dp, ae, ap):
    """pt, de, dp: (J, n) particles (0 = padding) in stored coordinates; ae, ap: (J,) new axis. Returns dict of per-jet arrays."""
    J = len(pt); out = dict(resolved=np.zeros(J, bool), zmin=np.zeros(J), drmin=np.zeros(J), axis_subjet=np.full(J, -1),
                            axis_subjet_z=np.full(J, np.nan), pred3=np.full(J, np.nan), pred3pp=np.full(J, np.nan),
                            axis_to_subjet=np.full(J, np.nan), m_pp=np.full(J, np.nan))
    for j in range(J):
        r = pt[j] > 0
        if r.sum() < 3: continue
        a, b, c = (np.asarray(v[j][r], np.float64) for v in (pt, de, dp))
        lab, sj = kt3(a, b, c)
        z = sj[:, 0] / sj[:, 0].sum()
        dr = lambda u, v: np.hypot(sj[u, 1] - sj[v, 1], sj[u, 2] - sj[v, 2])
        drs = [dr(0, 1), dr(0, 2), dr(1, 2)]
        out['zmin'][j] = z.min(); out['drmin'][j] = min(drs); out['resolved'][j] = z.min() >= ZMIN and min(drs) >= RMIN
        k = int(np.argmin((b - ae[j]) ** 2 + (c - ap[j]) ** 2)); s = lab[k]
        out['axis_subjet'][j] = s; out['axis_subjet_z'][j] = z[s]
        out['axis_to_subjet'][j] = np.hypot(ae[j] - sj[s, 1], ap[j] - sj[s, 2])
        # particle level: M0·M2 about the subjet's pT-weighted centroid vs about the stored axis
        m2_sub = (a * ((b - sj[s, 1]) ** 2 + (c - sj[s, 2]) ** 2)).sum(); m2_0 = (a * (b ** 2 + c ** 2)).sum()
        out['pred3'][j] = 0.5 * np.log(m2_sub / m2_0)
        # point prongs
        th2 = lambda u, v: (sj[u, 1] - sj[v, 1]) ** 2 + (sj[u, 2] - sj[v, 2]) ** 2
        mm = z[0] * z[1] * th2(0, 1) + z[0] * z[2] * th2(0, 2) + z[1] * z[2] * th2(1, 2)
        ms = sum(z[t] * th2(s, t) for t in range(3))
        out['pred3pp'][j] = 0.5 * np.log(ms / mm) if mm > 0 else np.nan
        out['m_pp'][j] = np.sqrt(mm) * a.sum()
    return out


# ------------------------------------------------------------------------------------------------ statistics
def boot_mean(x, rng, B=NBOOT):
    x = np.asarray(x, np.float64); n = len(x)
    if n == 0: return [None, None]
    idx = rng.integers(0, n, (B, n)); m = x[idx].mean(1); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def boot_ratio(num, den, rng, B=NBOOT):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64); n = len(num)
    if n == 0: return [None, None]
    idx = rng.integers(0, n, (B, n)); r = num[idx].mean(1) / den[idx].mean(1)
    return [float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def ttest1(x):
    """one-sample t-test of mean 0: t, two-sided p, log10 p (finite when p underflows)"""
    from scipy import stats
    x = np.asarray(x, np.float64); n = len(x)
    if n < 3 or x.std() == 0: return dict(t=None, p=None, log10p=None, n=n)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n)); lp = (np.log(2) + stats.t.logsf(abs(t), n - 1)) / np.log(10)
    return dict(t=float(t), p=float(10 ** lp), log10p=float(lp), n=n)


def mcnemar(a, b):
    """exact McNemar for paired booleans a (before), b (after): n10 = a & ~b, n01 = ~a & b, two-sided binomial p"""
    from scipy import stats
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5)))
    lp = None
    if n > 0:
        lp = float(min(0.0, (np.log(2) + stats.binom.logcdf(min(n10, n01), n, 0.5)) / np.log(10)))
    return dict(n10=n10, n01=n01, p=p, log10p=lp)


def holm(ps):
    """Holm-adjusted p-values (None passes through), on log10 scale input or p: here plain p list"""
    idx = [i for i, p in enumerate(ps) if p is not None]; order = sorted(idx, key=lambda i: ps[i]); m = len(order)
    out = [None] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


def fit_line(x, y):
    """OLS y = a + b x: slope, intercept, R² (and slope through the origin)"""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0: return dict(n=int(len(x)))
    b, a = np.polyfit(x, y, 1); r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return dict(n=int(len(x)), slope=float(b), intercept=float(a), r2=r2, slope0=float((x * y).sum() / (x * x).sum()))


def class_block(dl, pred, dl_cal, c0, c, y, k, te_mask, rng, ncls):
    """per true class k: read-out change vs prediction (test half), calls (all jets)"""
    s = te_mask & (y == k); a = (y == k)
    d, pr, dc = dl[s], pred[s], dl_cal[s]
    e = dict(n_te=int(s.sum()), n=int(a.sum()),
             dlogm_hat=float(d.mean()), dlogm_hat_ci=boot_mean(d, rng), pred=float(pr.mean()), pred_ci=boot_mean(pr, rng),
             ratio=float(d.mean() / pr.mean()) if abs(pr.mean()) > 1e-9 else None,
             ratio_ci=boot_ratio(d, pr, rng) if abs(pr.mean()) > 1e-9 else [None, None],
             dlogm_hat_cal=float(dc.mean()), dlogm_hat_cal_ci=boot_mean(dc, rng),
             ratio_cal=float(d.mean() / dc.mean()) if abs(dc.mean()) > 1e-9 else None,
             ratio_cal_ci=boot_ratio(d, dc, rng) if abs(dc.mean()) > 1e-9 else [None, None],
             H0_readout_zero=ttest1(d), perjet=fit_line(pr, d))
    ch = int((c[a] != c0[a]).sum()); e['changed'] = ch / max(a.sum(), 1); e['changed_ci'] = wilson(ch, int(a.sum()))
    own0, own1 = c0[a] == k, c[a] == k
    e['own0'] = float(own0.mean()); e['own1'] = float(own1.mean()); e['own0_ci'] = wilson(int(own0.sum()), int(a.sum()))
    e['own1_ci'] = wilson(int(own1.sum()), int(a.sum())); e['mcnemar_own'] = mcnemar(own0, own1)
    fl = np.zeros((ncls, ncls), int); np.add.at(fl, (c0[a], c[a]), 1); e['flow'] = fl.tolist()
    e['new_call_frac'] = [float((c[a] == j).mean()) for j in range(ncls)]
    e['new_call_ci'] = [wilson(int((c[a] == j).sum()), int(a.sum())) for j in range(ncls)]
    e['old_call_frac'] = [float((c0[a] == j).mean()) for j in range(ncls)]
    return e


def curve(mhat_gev, vals, edges):
    """mean of vals (J, K) and fraction > 0 per bin of mhat"""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (mhat_gev >= lo) & (mhat_gev < hi)
        out.append(dict(lo=float(lo), hi=float(hi), n=int(s.sum()),
                        mean=vals[s].mean(0).tolist() if s.sum() else None, on=(vals[s] > 0).mean(0).tolist() if s.sum() else None))
    return out
