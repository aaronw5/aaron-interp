"""E23 — natural off-centre jets and on-manifold axis edits for JEDI-linear (3-feature, n64): does the mass read-out
follow sqrt(M0 M2) about the stored axis (H1) or the invariant mass m (H0) on unedited jets whose stored axis is far
from their pT centroid?

Sample: interp.jedi_core.sample(per_class=PER, seed=101) (PER = 20000 -> 100,000 jets). Read-out sample: sample(PER/5,
seed=103) minus the jets also in the main sample; ridge of log m on z (16 pre-ReLU units), fitted on its first half with
alpha chosen on the last quarter of that half (E17 procedure); held-out R^2 on its second half and on the main sample.
Per jet, about the stored axis (real particles, pT > 0): M0 = sum pT, M1 = sum pT (deta, dphi), M2 = sum pT r^2,
q = |M1|^2 / (M0 M2), d = |M1| / M0, m from massless four-vectors of the 64 particles (small angle: m^2 ~ M0 M2 - |M1|^2).
Sets: T1/T2/T5 = top 1/2/5 % of q (Tq*) and of d (Td*); Tq*c = top 1/2/5 % of q within each class (added after the
smoke run: the global tail holds almost no W / Z jets); bulk B = q below the median.
(A) observational: per true class (W, Z, t) each tail jet matched to a bulk jet (greedy, random order, nearest in the two
    match variables, calipers 0.02 on each, without replacement). Match 1 on (log m, log M0): H1 predicts read-out
    difference = c * dlog sqrt(M0 M2), H0 predicts 0. Match 2 on (log sqrt(M0 M2), log M0): H1 predicts 0, H0 predicts
    c * dlog m. Differences per pair; 95 % bootstrap CIs over pairs (B = 1000); sign-flip permutation p (labels swapped
    within pairs, 4000 draws) and normal p.
(B) edits on tail jets (only real particles move; pT, slot order, padding unchanged):
    recentre  (deta, dphi) -= M1/M0: M1 -> 0, m unchanged (longitudinal boost + azimuthal rotation), M2 -> M2 - M0 d^2
    cal       matched real change: offsets about the axis scaled by k = exp(pred) per jet (pred = 1/2 dlog(M0 M2) of the
              recentre edit), so m and sqrt(M0 M2) both change by ~ pred
    c1        recentre on 5,000 random bulk jets (edit of near-zero size)
    c2        reflected edit (deta, dphi) += M1/M0: M1 -> 2 M1, M2 -> M2 + 3 M0 d^2 (pred opposite in sign), with its own cal
    c3        rotation about the stored axis by theta_j with 2 sin(theta/2) <r>_j = d_j (mean per-particle displacement
              equal to the recentre edit's d_j; M2 unchanged, pred 0)
    c4        recentre with the nine angle-only P units (E6v) frozen at their unedited values (run_jetwise override)
    Metrics: read-out change dread vs pred (ratio of means, 95 % CI), calibrated ratio dread / dread_cal, per-jet slope;
    calls changed, flow, Z->W / W->Z (call before -> after) with exact McNemar, accuracy before -> after (McNemar on
    correctness); per class; training 99 % |deta|, |dphi| (0.342, 0.336) occupancy before / after.
(C) 20,000 bulk jets, rigid shift of size s in a random direction per jet; s = 0 (identity), the median / p90 / p99 of d
    over the main sample, and of |stored axis - E-scheme axis of the 64 particles|; calls changed vs mean 1/2 dlog M2.
Holm over (tail set x edit) for the read-out tests and for the McNemar call tests.
  ./job.sh e23-smoke 'python -m interp.e23_tail_jedi 500'     -> $AI_RESULTS/jedi/e23_tail_500.json
  ./job.sh e23-run   'python -m interp.e23_tail_jedi'         -> $AI_RESULTS/jedi/e23_tail.json
  python -m interp.e23_tail_jedi --plot [json]                -> results/jedi/e23_tail.png (matplotlib)
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'jedi'
CL = ['g', 'q', 'W', 'Z', 't']; W, Z, T = 2, 3, 4
SEED, SEED_RO = 101, 103
TR99 = dict(eta=0.341552734375, phi=0.335693359375)       # E6v training 99 % |deta|, |dphi|
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)
NB = 1000; NPERM = 4000; CAL = 0.02
log = lambda *a: print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def sample_idx(per_class, seed):
    """row indices drawn by interp.jedi_core.sample(per_class, seed) (same code path)"""
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')
    y = np.asarray(d['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def moments(xx):
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    real = pt > 0; pt = np.where(real, pt, 0.0); eta = np.where(real, eta, 0.0); phi = np.where(real, phi, 0.0)
    M0 = pt.sum(1); M1e = (pt * eta).sum(1); M1p = (pt * phi).sum(1); M2 = (pt * (eta ** 2 + phi ** 2)).sum(1)
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    M2s = np.maximum(M2, 1e-12)
    ptJ = np.hypot(p4[:, 0], p4[:, 1]); axE = np.hypot(np.arcsinh(p4[:, 2] / np.maximum(ptJ, 1e-12)), np.arctan2(p4[:, 1], p4[:, 0]))
    nreal = np.maximum(real.sum(1), 1); rmean = np.where(real, np.sqrt(eta ** 2 + phi ** 2), 0).sum(1) / nreal
    return dict(M0=M0, M1e=M1e, M1p=M1p, M2=M2, m=m, logm=np.log(np.maximum(m, 1e-3)), logM0=np.log(M0),
                sq=0.5 * np.log(M0 * M2s), q=(M1e ** 2 + M1p ** 2) / (M0 * M2s), d=np.hypot(M1e, M1p) / M0,
                axis_E=axE, rmean=rmean, nreal=real.sum(1))


def edit_shift(x, ce, cp):
    real = x[..., 0] > 0; e = x.copy()
    e[..., 1] = np.where(real, x[..., 1].astype(np.float64) + ce[:, None], x[..., 1]).astype(np.float32)
    e[..., 2] = np.where(real, x[..., 2].astype(np.float64) + cp[:, None], x[..., 2]).astype(np.float32)
    return e


def edit_scale(x, k):
    real = x[..., 0] > 0; e = x.copy()
    for c in (1, 2): e[..., c] = np.where(real, x[..., c].astype(np.float64) * k[:, None], x[..., c]).astype(np.float32)
    return e


def edit_rot(x, th):
    real = x[..., 0] > 0; e = x.copy(); de, dp = x[..., 1].astype(np.float64), x[..., 2].astype(np.float64)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    e[..., 1] = np.where(real, c * de - s * dp, x[..., 1]).astype(np.float32)
    e[..., 2] = np.where(real, s * de + c * dp, x[..., 2]).astype(np.float32)
    return e


def range_stats(xx):
    real = xx[..., 0] > 0
    out = ((np.abs(xx[..., 1]) > TR99['eta']) | (np.abs(xx[..., 2]) > TR99['phi'])) & real
    return dict(frac_particles_beyond_train99=float(out.sum() / max(real.sum(), 1)), frac_jets_any_beyond=float(out.any(1).mean()),
                max_abs_deta=float(np.abs(np.where(real, xx[..., 1], 0)).max()), max_abs_dphi=float(np.abs(np.where(real, xx[..., 2], 0)).max()))


def quant(v, qs=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)):
    v = np.asarray(v, np.float64)
    return {f'p{int(round(100 * q_))}': float(np.quantile(v, q_)) for q_ in qs} if len(v) else {}


def log10_p_normal(z):
    x = abs(z) / math.sqrt(2.0)
    if x < 26: return math.log10(max(math.erfc(x), 1e-300))
    return (-x * x - math.log(x * math.sqrt(math.pi))) / math.log(10.0)


class Boot:
    """jet bootstrap: fixed (NB, n) index draws for one set of jets"""
    def __init__(self, n, seed):
        self.n = n; self.I = np.random.default_rng(seed).integers(0, n, (NB, n)) if n else None

    def mean(self, a, null=0.0):
        a = np.asarray(a, np.float64)
        if self.n == 0: return dict(mean=None, ci=[None, None])
        mb = a[self.I].mean(1); mu = float(a.mean()); se = float(mb.std(ddof=1)) if self.n > 1 else 0.0
        z = (mu - null) / se if se > 0 else 0.0; lp = log10_p_normal(z)
        return dict(mean=mu, ci=[float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))], se=se, z=float(z), log10_p_z=lp, p_z=10 ** lp)

    def ratio(self, a, b):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        if self.n == 0 or abs(b.mean()) < 1e-6: return dict(ratio=None, ci=[None, None])
        rb = a[self.I].mean(1) / b[self.I].mean(1)
        return dict(ratio=float(a.mean() / b.mean()), ci=[float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))], se=float(rb.std(ddof=1)))


def signflip_p(v, seed):
    """paired permutation test (labels swapped within pairs = sign flips) of mean(v) = 0"""
    v = np.asarray(v, np.float64); n = len(v)
    if n == 0: return None
    obs = abs(v.mean()); rng = np.random.default_rng(seed); k = 0
    for i in range(0, NPERM, 500):
        s = rng.integers(0, 2, (min(500, NPERM - i), n)) * 2 - 1; k += int(np.sum(np.abs((s * v).mean(1)) >= obs - 1e-15))
    return (1 + k) / (NPERM + 1)


def mcnemar(n01, n10):
    n = n01 + n10
    if n == 0: return dict(b=0, c=0, p=1.0, log10_p=0.0)
    k = min(n01, n10)
    lt = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)])
    ls = lt.max() + math.log(np.exp(lt - lt.max()).sum()); l10 = min(0.0, (math.log(2) + ls) / math.log(10))
    return dict(b=int(n01), c=int(n10), p=10 ** l10, log10_p=l10)


def holm(lps):
    lps = np.asarray(lps, np.float64); m = len(lps); order = np.argsort(lps); adj = np.empty(m); run = -np.inf
    for i, j in enumerate(order):
        run = max(run, min(0.0, math.log10(m - i) + lps[j])); adj[j] = run
    return adj


class Ridge:                    # as interp/e17_axis_jedi.py
    def __init__(self, X, t, idx_fit, idx_val, idx_all):
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
        return ((X - self.mu) / self.sd) @ self.w * self.ts + self.tm


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def softmax(L):
    e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def match(tail, pool, F, rng):
    """greedy 1:1 matching without replacement: tail jets in random order, nearest available pool jet in the standardised-free
    feature space F (J, 2), both |differences| <= CAL"""
    avail = np.ones(len(pool), bool); Fp = F[pool]; pairs = []
    for i in rng.permutation(tail):
        dd = np.abs(Fp - F[i]); ok = avail & (dd[:, 0] <= CAL) & (dd[:, 1] <= CAL)
        if not ok.any(): continue
        dist = np.where(ok, (dd ** 2).sum(1), np.inf); j = int(np.argmin(dist)); avail[j] = False; pairs.append((i, pool[j]))
    return np.array(pairs, int).reshape(-1, 2)


# ------------------------------------------------------------------------------------------------ main
def main(PER):
    from interp.jedi_core import Jedi, sample
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    res = dict(per_class=PER, seed=SEED, seed_readout=SEED_RO, classes=CL, n_boot=NB, n_perm=NPERM, caliper=CAL, train99=TR99)
    x, y, _ = sample(per_class=PER, seed=SEED); x = np.ascontiguousarray(x, np.float32); J = len(x); res['N'] = J
    idxA = sample_idx(PER, SEED)
    net = Jedi(64)
    log(f'E23: main sample {J} jets ({PER}/class, seed {SEED}); model {net.path} ({time.time() - t0:.0f} s)')

    # ---- network on the unedited sample
    R0 = net.run(x, keep=('z', 'logits', 'P')); z0, L0, P0 = R0['z'], R0['logits'], R0['P']; c0 = L0.argmax(1)
    acc0 = float((c0 == y).mean())
    res['baseline'] = dict(acc=acc0, W_to_Z=float((c0[y == W] == Z).mean()), Z_to_W=float((c0[y == Z] == W).mean()),
                           own_class=[float((c0[y == k] == k).mean()) for k in range(5)], accuracy_in_expected_range=bool(0.806 <= acc0 <= 0.813))
    log(f'baseline accuracy {acc0:.4f} (expected 0.806-0.813); W->Z {res["baseline"]["W_to_Z"]:.4f}, Z->W {res["baseline"]["Z_to_W"]:.4f} ({time.time() - t0:.0f} s)')

    # ---- read-out (disjoint sample)
    PR = max(PER // 5, 100)
    xr, yr, _ = sample(per_class=PR, seed=SEED_RO); idxR = sample_idx(PR, SEED_RO); keep = ~np.isin(idxR, idxA)
    xr, yr = np.ascontiguousarray(xr[keep], np.float32), yr[keep]
    zr = net.run(xr, keep=('z',))['z']; Mr = moments(xr)
    perm = np.random.default_rng(SEED_RO).permutation(len(xr)); fh, th = np.sort(perm[:len(xr) // 2]), np.sort(perm[len(xr) // 2:])
    ridge = Ridge(zr, Mr['logm'], fh[: len(fh) * 3 // 4], fh[len(fh) * 3 // 4:], fh)
    M = moments(x); lm0 = ridge(z0)
    res['readout'] = dict(n_readout=int(len(xr)), n_overlap_removed=int((~keep).sum()), n_fit=int(len(fh)), n_heldout=int(len(th)),
                          alpha=float(ridge.alpha), r2_heldout_logm=r2(ridge(zr[th]), Mr['logm'][th]),
                          r2_main_logm=r2(lm0, M['logm']), r2_main_logsqrtM0M2=r2(lm0, M['sq']),
                          corr_logm_logsqrtM0M2_main=float(np.corrcoef(M['logm'], M['sq'])[0, 1]))
    log(f'read-out: {res["readout"]} ({time.time() - t0:.0f} s)')

    # ---- moments and sets
    q, d = M['q'], M['d']
    res['moments'] = dict(q=quant(q), d=quant(d), axis_E_minus_stored=quant(M['axis_E']),
                          max_abs_m2_small_angle_rel=float(np.median(np.abs(M['M0'] * M['M2'] - (M['M1e'] ** 2 + M['M1p'] ** 2) - M['m'] ** 2) / M['m'] ** 2)))
    sets = {}
    for k_, f_ in ((1, 0.99), (2, 0.98), (5, 0.95)):
        sets[f'Tq{k_}'] = np.flatnonzero(q >= np.quantile(q, f_)); sets[f'Td{k_}'] = np.flatnonzero(d >= np.quantile(d, f_))
        # class-wise tail (added after the smoke run: the global q tail holds almost no W / Z jets): top k % of q within each class
        sets[f'Tq{k_}c'] = np.sort(np.concatenate([np.flatnonzero((y == c) & (q >= np.quantile(q[y == c], f_))) for c in range(5)]))
    bulk = np.flatnonzero(q < np.median(q)); SETS = ['Tq1', 'Tq2', 'Tq5', 'Td1', 'Td2', 'Td5', 'Tq1c', 'Tq2c', 'Tq5c']
    rng = np.random.default_rng(SEED + 1); bulkc = np.sort(rng.choice(bulk, min(5000, len(bulk)), replace=False))
    res['sets'] = {}
    for nm, S in list(sets.items()) + [('bulk', bulk), ('bulk_c1', bulkc)]:
        res['sets'][nm] = dict(n=int(len(S)), n_class=[int((y[S] == k).sum()) for k in range(5)], q=quant(q[S]), d=quant(d[S]),
                               m_median=float(np.median(M['m'][S])), M0_median=float(np.median(M['M0'][S])), nreal_mean=float(M['nreal'][S].mean()),
                               acc=float((c0[S] == y[S]).mean()), range_unedited=range_stats(x[S]))
        log(f'set {nm}: n {len(S)} classes {res["sets"][nm]["n_class"]} q p50 {res["sets"][nm]["q"]["p50"]:.4g} d p50 {res["sets"][nm]["d"]["p50"]:.4g} '
            f'acc {res["sets"][nm]["acc"]:.3f} beyond99 jets {res["sets"][nm]["range_unedited"]["frac_jets_any_beyond"]:.3f}')

    # ---- (A) observational matching
    pZ = softmax(L0)[:, Z]; isZ = (c0 == Z).astype(float); isW = (c0 == W).astype(float); dWZ = L0[:, W] - L0[:, Z]
    FA = {'m': np.stack([M['logm'], M['logM0']], 1), 'sq': np.stack([M['sq'], M['logM0']], 1)}
    res['A'] = {}; lpA = []
    for nm in SETS:
        res['A'][nm] = {}
        for mk in ('m', 'sq'):
            allp = []; per = {}
            for c in (W, Z, T):
                pr = match(sets[nm][y[sets[nm]] == c], bulk[y[bulk] == c], FA[mk], np.random.default_rng(SEED + 7 * c))
                allp.append(pr); per[CL[c]] = pr
            out = {}
            for grp, pr in [('all', np.concatenate(allp))] + list(per.items()):
                n = len(pr); a, b = (pr[:, 0], pr[:, 1]) if n else (np.zeros(0, int), np.zeros(0, int)); bt = Boot(n, SEED + n)
                dsq, dlm, drd = M['sq'][a] - M['sq'][b], M['logm'][a] - M['logm'][b], lm0[a] - lm0[b]
                e = dict(n_pairs=int(n), n_tail=int(len(sets[nm]) if grp == 'all' else (y[sets[nm]] == CL.index(grp)).sum()),
                         d_logsqrtM0M2=bt.mean(dsq), d_logm=bt.mean(dlm), d_logM0=bt.mean(M['logM0'][a] - M['logM0'][b]),
                         d_readout=bt.mean(drd), p_perm_readout=signflip_p(drd, SEED + 11) if n else None,
                         ratio_readout_over_dlogsqrtM0M2=bt.ratio(drd, dsq), ratio_readout_over_dlogm=bt.ratio(drd, dlm),
                         d_Zcall=bt.mean(isZ[a] - isZ[b]), p_perm_Zcall=signflip_p(isZ[a] - isZ[b], SEED + 12) if n else None,
                         d_Wcall=bt.mean(isW[a] - isW[b]), d_pZ=bt.mean(pZ[a] - pZ[b]), d_logitWminusZ=bt.mean(dWZ[a] - dWZ[b]),
                         q_tail_median=float(np.median(q[a])) if n else None, q_bulk_median=float(np.median(q[b])) if n else None)
                out[grp] = e
                if grp == 'all' and n: lpA.append((nm, mk, math.log10(max(e['p_perm_readout'], 1e-300)), e['d_readout']['log10_p_z']))
            res['A'][nm][mk] = out
            o = out['all']
            if o['n_pairs']:
                log(f'A {nm} match on ({mk}, M0): pairs {o["n_pairs"]}; dlog sqrtM0M2 {o["d_logsqrtM0M2"]["mean"]:+.4f} dlogm {o["d_logm"]["mean"]:+.4f} '
                    f'dread {o["d_readout"]["mean"]:+.4f} {o["d_readout"]["ci"]} p_perm {o["p_perm_readout"]:.4g}; dZcall {o["d_Zcall"]["mean"]:+.4f} {o["d_Zcall"]["ci"]}')
    if lpA:
        hp = holm([v[2] for v in lpA]); hz = holm([v[3] for v in lpA])
        res['A_holm'] = [dict(set=v[0], match=v[1], log10_p_perm=v[2], log10_p_perm_holm=float(h1), log10_p_z=v[3], log10_p_z_holm=float(h2))
                         for v, h1, h2 in zip(lpA, hp, hz)]
    log(f'(A) done ({time.time() - t0:.0f} s)')

    # ---- (B) edits
    U = np.unique(np.concatenate([sets['Tq5'], sets['Td5'], sets['Tq5c'], bulkc])); pos = -np.ones(J, int); pos[U] = np.arange(len(U))
    xU = x[U]; MU = {k: v[U] for k, v in M.items()}
    ce, cp = MU['M1e'] / MU['M0'], MU['M1p'] / MU['M0']
    X = {}; X['recentre'] = edit_shift(xU, -ce, -cp); X['reflect'] = edit_shift(xU, ce, cp)
    th_ = 2 * np.arcsin(np.clip(MU['d'] / (2 * np.maximum(MU['rmean'], 1e-9)), 0, 1)); X['rotate'] = edit_rot(xU, th_)
    ME = {k: moments(v) for k, v in X.items()}
    pred = {k: ME[k]['sq'] - MU['sq'] for k in X}
    X['cal_recentre'] = edit_scale(xU, np.exp(pred['recentre'])); X['cal_reflect'] = edit_scale(xU, np.exp(pred['reflect']))
    for k in ('cal_recentre', 'cal_reflect'): ME[k] = moments(X[k]); pred[k] = ME[k]['sq'] - MU['sq']
    RE = {k: net.run(v, keep=('z', 'logits')) for k, v in X.items()}
    K1 = net.KB['l1'][0]; ANG = [int(u) for u in np.flatnonzero((np.abs(K1[0]) < 1e-9) & (np.abs(K1[1:]).sum(0) > 1e-9))]
    maskP = np.zeros(64); maskP[ANG] = 1
    RE['recentre_Pfreeze'] = net.run_jetwise(X['recentre'], {'P': (maskP, P0[U])}, keep=('z', 'logits'))
    ME['recentre_Pfreeze'] = ME['recentre']; pred['recentre_Pfreeze'] = pred['recentre']; X['recentre_Pfreeze'] = X['recentre']
    chk = net.run_jetwise(xU, {'P': (maskP, P0[U])}, keep=('logits',))['logits']
    res['B_checks'] = dict(angle_only_P_units=ANG, Pfreeze_identity_max_dlogit=float(np.abs(chk - L0[U]).max()),
                           recentre_max_abs_M1_over_M0_after=float(np.max(ME['recentre']['d'])),
                           recentre_max_abs_dlogm=float(np.max(np.abs(ME['recentre']['logm'] - MU['logm'])[MU['m'] > 1])),
                           reflect_max_abs_dlogm=float(np.max(np.abs(ME['reflect']['logm'] - MU['logm'])[MU['m'] > 1])),
                           rotate_max_abs_dlogM2=float(np.max(np.abs(np.log(ME['rotate']['M2'] / MU['M2'])))),
                           rotate_median_abs_dlogm=float(np.median(np.abs(ME['rotate']['logm'] - MU['logm']))),
                           pt_unchanged=bool(all(np.array_equal(X[k][..., 0], xU[..., 0]) for k in X)),
                           padding_unchanged=bool(all(np.array_equal(X[k][xU[..., 0] <= 0], xU[xU[..., 0] <= 0]) for k in X)),
                           rotate_angle=quant(th_))
    log(f'B checks: {res["B_checks"]} ({time.time() - t0:.0f} s)')
    lmU, cU, LU, yU = lm0[U], c0[U], L0[U], y[U]

    def edit_metrics(sel_rows, k, calk):
        s = pos[sel_rows]; n = len(s); bt = Boot(n, SEED + 3 * n + len(k))
        ze, Le = RE[k]['z'][s], RE[k]['logits'][s]; c1_ = Le.argmax(1); cb = cU[s]; yy = yU[s]
        dr = ridge(ze) - lmU[s]; pr = pred[k][s]; dlm = ME[k]['logm'][s] - MU['logm'][s]
        disp = np.hypot(X[k][s][..., 1].astype(np.float64) - xU[s][..., 1], X[k][s][..., 2].astype(np.float64) - xU[s][..., 2])
        real = xU[s][..., 0] > 0; disp_mean = (disp * real).sum(1) / np.maximum(real.sum(1), 1)
        ch = (c1_ != cb).astype(float)
        flow = np.zeros((5, 5), int); np.add.at(flow, (cb, c1_), 1)
        e = dict(n=int(n), pred=bt.mean(pr), dlogm_true=bt.mean(dlm), dread=bt.mean(dr), p_perm_dread=signflip_p(dr, SEED + 5),
                 ratio_dread_over_pred=bt.ratio(dr, pr), per_jet_slope=float(np.polyfit(pr, dr, 1)[0]) if pr.std() > 1e-12 else None,
                 per_jet_corr=float(np.corrcoef(pr, dr)[0, 1]) if pr.std() > 1e-12 and dr.std() > 1e-12 else None,
                 mean_particle_displacement=bt.mean(disp_mean), d_before=bt.mean(MU['d'][s]),
                 changed=bt.mean(ch), flow=flow.tolist(),
                 Z_to_W=int(((cb == Z) & (c1_ == W)).sum()), W_to_Z=int(((cb == W) & (c1_ == Z)).sum()),
                 mcnemar_ZW=mcnemar(int(((cb == Z) & (c1_ == W)).sum()), int(((cb == W) & (c1_ == Z)).sum())),
                 acc_before=float((cb == yy).mean()), acc_after=float((c1_ == yy).mean()),
                 mcnemar_acc=mcnemar(int(((cb == yy) & (c1_ != yy)).sum()), int(((cb != yy) & (c1_ == yy)).sum())),
                 d_logitWminusZ=bt.mean((Le[:, W] - Le[:, Z]) - (LU[s][:, W] - LU[s][:, Z])),
                 range_after=range_stats(X[k][s]), range_before=range_stats(xU[s]))
        wz = (yy == W) | (yy == Z)
        e['trueWZ'] = dict(n=int(wz.sum()), Z_to_W=int(((cb == Z) & (c1_ == W) & wz).sum()), W_to_Z=int(((cb == W) & (c1_ == Z) & wz).sum()),
                           mcnemar_ZW=mcnemar(int(((cb == Z) & (c1_ == W) & wz).sum()), int(((cb == W) & (c1_ == Z) & wz).sum())),
                           changed=float(ch[wz].mean()) if wz.any() else None)
        if calk is not None:
            dc = ridge(RE[calk]['z'][s]) - lmU[s]; cc = (RE[calk]['logits'][s].argmax(1) != cb).astype(float)
            e['cal'] = dict(edit=calk, pred=bt.mean(pred[calk][s]), dlogm_true=bt.mean(ME[calk]['logm'][s] - MU['logm'][s]), dread=bt.mean(dc),
                            ratio_dread_over_pred=bt.ratio(dc, pred[calk][s]), changed=bt.mean(cc),
                            Z_to_W=int(((cb == Z) & (RE[calk]['logits'][s].argmax(1) == W)).sum()),
                            W_to_Z=int(((cb == W) & (RE[calk]['logits'][s].argmax(1) == Z)).sum()))
            e['ratio_dread_over_cal'] = bt.ratio(dr, dc)
            e['dread_minus_cal'] = bt.mean(dr - dc)
        e['per_class'] = {}
        for c in (W, Z, T):
            m_ = yy == c
            if m_.any():
                e['per_class'][CL[c]] = dict(n=int(m_.sum()), pred=float(pr[m_].mean()), dread=float(dr[m_].mean()), changed=float(ch[m_].mean()),
                                             Z_to_W=int(((cb == Z) & (c1_ == W) & m_).sum()), W_to_Z=int(((cb == W) & (c1_ == Z) & m_).sum()),
                                             dread_cal=float(dc[m_].mean()) if calk is not None else None)
        return e, ch

    EDITS = [('recentre', 'cal_recentre'), ('reflect', 'cal_reflect'), ('rotate', 'cal_recentre'), ('recentre_Pfreeze', 'cal_recentre')]
    res['B'] = {}; lpB = []; lpC = []
    for nm in SETS + ['bulk_c1']:
        rows = bulkc if nm == 'bulk_c1' else sets[nm]; res['B'][nm] = {}; chs = {}
        for k, calk in (EDITS if nm != 'bulk_c1' else [('recentre', 'cal_recentre')]):
            e, chs[k] = edit_metrics(rows, k, calk); res['B'][nm][k] = e
            lpB.append((nm, k, math.log10(max(e['p_perm_dread'], 1e-300)), e['dread']['log10_p_z'])); lpC.append((nm, k, e['mcnemar_ZW']['log10_p']))
            log(f'B {nm} {k}: n {e["n"]} pred {e["pred"]["mean"]:+.5f} dread {e["dread"]["mean"]:+.5f} {e["dread"]["ci"]} ratio {e["ratio_dread_over_pred"].get("ratio")} '
                f'{e["ratio_dread_over_pred"].get("ci")} cal-ratio {e.get("cal", {}).get("ratio_dread_over_pred", {}).get("ratio")} obs/cal {e.get("ratio_dread_over_cal", {}).get("ratio")} '
                f'{e.get("ratio_dread_over_cal", {}).get("ci")}; changed {e["changed"]["mean"]:.4f} Z->W {e["Z_to_W"]} W->Z {e["W_to_Z"]} p {e["mcnemar_ZW"]["p"]:.3g}; '
                f'acc {e["acc_before"]:.4f}->{e["acc_after"]:.4f}; disp {e["mean_particle_displacement"]["mean"]:.4f}')
        if nm != 'bulk_c1':
            n = len(rows); bt = Boot(n, SEED + 17)
            res['B'][nm]['compare'] = dict(changed_recentre_minus_rotate=bt.mean(chs['recentre'] - chs['rotate']),
                                           changed_recentre_minus_reflect=bt.mean(chs['recentre'] - chs['reflect']))
    hp = holm([v[2] for v in lpB]); hz = holm([v[3] for v in lpB]); hc = holm([v[2] for v in lpC])
    res['B_holm'] = [dict(set=v[0], edit=v[1], log10_p_perm=v[2], log10_p_perm_holm=float(a), log10_p_z=v[3], log10_p_z_holm=float(b_),
                          log10_p_mcnemar_ZW=w[2], log10_p_mcnemar_ZW_holm=float(c_)) for v, w, a, b_, c_ in zip(lpB, lpC, hp, hz, hc)]
    log(f'(B) done ({time.time() - t0:.0f} s)')

    # ---- (C) physical-size random-direction rigid shift on bulk jets
    rngC = np.random.default_rng(SEED + 2); Cj = np.sort(rngC.choice(bulk, min(20000, len(bulk)), replace=False))
    psi = rngC.uniform(0, 2 * np.pi, len(Cj)); xC = x[Cj]; MC = {k: v[Cj] for k, v in M.items()}; btC = Boot(len(Cj), SEED + 23)
    sizes = [('identity', 0.0)] + [(f'd_{p}', float(np.quantile(d, f))) for p, f in (('p50', 0.5), ('p90', 0.9), ('p99', 0.99))] + \
            [(f'axisE_{p}', float(np.quantile(M['axis_E'], f))) for p, f in (('p50', 0.5), ('p90', 0.9), ('p99', 0.99))]
    res['C'] = dict(n=int(len(Cj)), sizes=dict(sizes), runs={})
    for nm, s_ in sizes:
        xe = edit_shift(xC, s_ * np.cos(psi), s_ * np.sin(psi)); Me = moments(xe); Rr = net.run(xe, keep=('z', 'logits'))
        cE = Rr['logits'].argmax(1); cb = c0[Cj]; pr = Me['sq'] - MC['sq']; dr = ridge(Rr['z']) - lm0[Cj]; ch = (cE != cb).astype(float)
        e = dict(size=s_, pred=btC.mean(pr), dread=btC.mean(dr), ratio=btC.ratio(dr, pr), changed=btC.mean(ch),
                 Z_to_W=int(((cb == Z) & (cE == W)).sum()), W_to_Z=int(((cb == W) & (cE == Z)).sum()),
                 mcnemar_ZW=mcnemar(int(((cb == Z) & (cE == W)).sum()), int(((cb == W) & (cE == Z)).sum())),
                 acc_before=float((cb == y[Cj]).mean()), acc_after=float((cE == y[Cj]).mean()),
                 max_abs_dlogit=float(np.abs(Rr['logits'] - L0[Cj]).max()), max_abs_dlogm=float(np.abs(Me['logm'] - MC['logm'])[MC['m'] > 1].max()),
                 range_after=range_stats(xe))
        res['C']['runs'][nm] = e
        log(f'C {nm} s={s_:.5f}: pred {e["pred"]["mean"]:+.5f} dread {e["dread"]["mean"]:+.5f} {e["dread"]["ci"]} changed {e["changed"]["mean"]:.4f} '
            f'{e["changed"]["ci"]} Z->W {e["Z_to_W"]} W->Z {e["W_to_Z"]} max|dlogit| {e["max_abs_dlogit"]:.3g}')
    name = 'e23_tail.json' if PER == 20000 else f'e23_tail_{PER}.json'
    json.dump(res, open(OUT / name, 'w'), indent=1, default=float)
    log(f'wrote {OUT / name}; E23 DONE ({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------------------------------------ figure
def plot(name='e23_tail.json'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = json.loads((OUT / name).read_text()); sfx = name[len('e23_tail'):-len('.json')]
    SURF, INK, INK2, GRID = '#fcfcfb', '#1f1f1e', '#5f5e59', '#e4e3df'
    PAL = dict(recentre='#2a78d6', reflect='#eb6834', rotate='#1baf7a', recentre_Pfreeze='#8a5cd0', cal='#5f5e59')
    LAB = dict(recentre='recentre (M1 → 0, m fixed)', reflect='c2: reflected (M1 → 2 M1)', rotate='c3: rotation, same mean displacement',
               recentre_Pfreeze='c4: recentre, angle P units frozen', cal='matched real change (offsets × k)')
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURF)
    for a_ in axs:
        a_.set_facecolor(SURF); a_.grid(True, color=GRID, lw=0.8); a_.set_axisbelow(True)
        for sp in ('top', 'right'): a_.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'): a_.spines[sp].set_color(GRID)
        a_.tick_params(colors=INK2, labelsize=9.5)
    ax = axs[0]; SETS = ['Tq1', 'Tq2', 'Tq5', 'Td1', 'Td2', 'Td5', 'Tq1c', 'Tq2c', 'Tq5c']
    mk = dict(Tq1='o', Tq2='o', Tq5='o', Td1='s', Td2='s', Td5='s', Tq1c='^', Tq2c='^', Tq5c='^')
    seen = set()
    for st in [s_ for s_ in SETS if s_ in R['B']]:
        B = R['B'][st]
        for k in ('recentre', 'reflect', 'rotate', 'recentre_Pfreeze', 'cal'):
            e = B['recentre']['cal'] if k == 'cal' else B[k]
            if k == 'cal':
                pts = [(B['recentre']['cal'], 'cal'), (B['reflect']['cal'], 'cal')]
            else:
                pts = [(e, k)]
            for e_, kk in pts:
                xv = 1e3 * e_['pred']['mean']; yv = 1e3 * e_['dread']['mean']; lo, hi = (1e3 * v for v in e_['dread']['ci'])
                ax.errorbar(xv, yv, yerr=[[yv - lo], [hi - yv]], fmt=mk[st], color=PAL[kk], ms=6, mfc=SURF if kk == 'cal' else PAL[kk], mew=1.4, lw=1,
                            label=LAB[kk] if kk not in seen else None, alpha=0.9); seen.add(kk)
    e = R['B']['bulk_c1']['recentre']; xv = 1e3 * e['pred']['mean']; yv = 1e3 * e['dread']['mean']; lo, hi = (1e3 * v for v in e['dread']['ci'])
    ax.errorbar(xv, yv, yerr=[[yv - lo], [hi - yv]], fmt='D', color=INK, ms=6, label=f'c1: recentre bulk jets (n {e["n"]:,})')
    lim = ax.get_xlim(); xx = np.linspace(*lim, 2); ax.plot(xx, xx, color=INK2, lw=1, ls=':', label='Δread-out = prediction')
    ax.axhline(0, color=INK2, lw=0.8); ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel('prediction ½ Δlog(M0·M2) about the stored axis [×10⁻³]', color=INK2, fontsize=10.5)
    ax.set_ylabel('Δ read-out of log m (ridge on z) [×10⁻³]', color=INK2, fontsize=10.5)
    ax.set_title('Recentring (m fixed) lowers the mass read-out; rotation does not', color=INK, fontsize=11.5, loc='left')
    ax.legend(frameon=False, fontsize=8.3, loc='upper left', labelcolor=INK)
    ax = axs[1]; C = R['C']['runs']
    for grp, col, lab in (('d_', INK2, 'random-direction shift, size = p50 / p90 / p99 of d'),):
        ks = [k for k in C if k.startswith(grp)]
        ax.errorbar([1e3 * C[k]['pred']['mean'] for k in ks], [100 * C[k]['changed']['mean'] for k in ks],
                    yerr=[[100 * (C[k]['changed']['mean'] - C[k]['changed']['ci'][0]) for k in ks], [100 * (C[k]['changed']['ci'][1] - C[k]['changed']['mean']) for k in ks]],
                    fmt='o-', color=col, ms=6, lw=1.6, label=lab)
        for k in ks: ax.annotate(f'{k.split("_")[1]}: {1e3 * C[k]["size"]:.1f} mrad', (1e3 * C[k]['pred']['mean'], 100 * C[k]['changed']['mean']), fontsize=8, color=col,
                                 xytext=(6, -12), textcoords='offset points')
    ax.plot([0], [100 * C['identity']['changed']['mean']], 'k*', ms=9, label='size 0 (identity)')
    for k, kk in (('recentre', 'recentre'), ('rotate', 'rotate'), ('reflect', 'reflect')):
        e = R['B']['Tq5'][k]; ax.plot(1e3 * e['pred']['mean'], 100 * e['changed']['mean'], marker='s', color=PAL[kk], ms=8, ls='none', label=f'Tq5 (top 5 % q): {LAB[kk]}')
    ax.set_xlabel('mean ½ Δlog(M0·M2) [×10⁻³]', color=INK2, fontsize=10.5); ax.set_ylabel('calls changed [%]', color=INK2, fontsize=10.5)
    ax.set_title(f'Calls changed: bulk jets (n {R["C"]["n"]:,}) vs top-5 % q tail', color=INK, fontsize=11.5, loc='left')
    ax.legend(frameon=False, fontsize=8.3, loc='upper left', labelcolor=INK)
    fig.text(0.01, 0.005, f'JEDI-linear 3f n64, {R["N"]:,} jets (seed {R["seed"]}); read-out fitted on {R["readout"]["n_fit"]:,} disjoint jets. Left: circle top q, square top d, '
             'triangle top q per class (1/2/5 %); open grey: matched real change. Bars: 95 % bootstrap.', color=INK2, fontsize=8.2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / f'e23_tail{sfx}.png', dpi=140, facecolor=SURF); plt.close(fig)
    log('wrote', OUT / f'e23_tail{sfx}.png')


if __name__ == '__main__':
    if '--plot' in sys.argv:
        args = [a for a in sys.argv[1:] if a != '--plot']; plot(args[0] if args else 'e23_tail.json'); sys.exit(0)
    PER = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(PER)
