"""X5 (claims A1, A3, A10; review weakness 7): do the pure pT-only / angle-only cuts, rank windows, count-weighted angular
objects and the un-recentred shift response of JEDI-linear appear only in a Deep Sets trained with HGQ fixed-point
quantization, or also in float and L1-sparse Deep Sets of the same architecture on the same inputs?

Models: dsets-d64, N = 64 (train/models.py DeepSetsTagger; HGQ version train/train_hgq_dsets.py), three regimes x seeds 0, 1, 2:
  float  models/small/dsets-d64-n64/seed<k>                weights.npz (train/export_npz.py from best.pt)
  l1     models/small/dsets-d64-n64-l1_<lam>/seed<k>       weights_zeroed.npz (phi1 |w| < 1e-3 set to 0 after training)
  hgq    models/small/dsets-d64-n64-hgq_<beta>/seed<k>     model.keras (HGQ2 fixed-point forward, quantizers applied)
One job per model:   python -m interp.x5_dsets_sparsity --regime R --seed K --dir <run dir>
Then:                python -m interp.x5_dsets_sparsity --combine     (regime tests over the 9 models, table)

FROZEN ANALYSIS PLAN (written before the first analysis run)
 Sample: interp.jedi_core.sample(per_class=2000, seed=311) = 10,000 test jets. Inputs as in training: (x - shift) / scale with the
   run's shift / scale (= norm_stats.json['64']), padded rows (raw pT = 0) set to 0, mask = raw pT > 0.
 Step 0  accuracy on the sample vs the run's test.json (260,000 jets); explicit forward vs the stored test_probs on the same rows.
 Step 1  live units. phi1 unit c set to its mean over real particles; S unit set to its mean over jets; live = calls changed >= 1 %.
 (a) cut types of live phi1 units, physical form alpha_pT pT + alpha_eta deta + alpha_phi dphi + beta (alpha = W / scale);
     term spread w_k = |alpha_k| sd(x_k) over real particles, share s_k = w_k^2 / sum w^2; pT ramp: s_pT >= 0.9; angular:
     s_eta + s_phi >= 0.9; else mixed. Statistic P = (ramps + angular) / live. Null (X6v): each feature's column of alpha permuted
     independently over the 64 units (200 draws); statistic over the units with any nonzero weight (observed: P_nz on the same
     set); one-sided p = (1 + #{P_null >= P_nz}) / 201.  [debug 1: the first version evaluated the null at the live positions,
     where permuted all-zero units count as impure; changed to the X6v definition]
 (b) rank windows. Jets with >= 18 real particles. Angle swap W_k: (deta, dphi) of ranks k and k+1 exchanged, pT order kept,
     k = 0..16. Per jet m_k = mean over live S units of |dS_c| / sd(S_c). Edge contrast at k = 1..15: d_k = m_k - (m_{k-1} +
     m_{k+1}) / 2 per jet, c_k = mean m_k / mean of the neighbours; one-sided sign-flip permutation p (2,000) for E[d_k] > 0, Holm
     over the 15 k. Window present iff some k has Holm p < 0.05 and c_k >= 1.5. Element statistic: max_k c_k; element p: the smallest
     Holm-adjusted p among k with c_k >= 1.5 (1 if none). Also: row swap R_k (the whole particle rows k, k+1 exchanged: same jet,
     slot positions changed; 0 for a permutation-invariant network) with the same per-jet metric, and for HGQ the zero-bit masks of
     the pooling input quantizer (kept slots per live S channel, longest contiguous run / kept).
 (c) count-weighted angular objects. Single particle at slot 5 of an otherwise empty jet (pT 1, 3, 10, 30, 100 GeV; r 0-0.4 rad,
     41 steps; theta 72 steps); f = phi2_c at that slot (S_c is exactly the sum of these per-particle values). Angular share
     A = 1 - var(pT means) / var(f); within-pT radial share (E14v) R = 1 - sum (f - ring mean)^2 / sum (f - pT mean)^2; weighting
     of the radial profile h(pT, r) (ring mean minus its mean over r): count model h = h0(r), R2_count; pT-weighted model
     h = pT h1(r), R2_pT. Null for R (E14v ii): ReLU(sum_c w_c a_c + b) of the live phi1 maps at the same slot, w ~ N(0, 1),
     b = -median over the grid, 2,000 draws (E14v; debug 1: 200 draws gave a p floor of 0.005, above the Holm level for
     ~50 units); per-unit p = (1 + #{R_null >= R}) / 201, Holm over the tested units (A >= 0.05).
     Count-weighted radial object: A >= 0.05, Holm p < 0.05, R2_count > R2_pT. Element present iff >= 2 such live S units; element
     statistic: their fraction among live S units; element p: the second-smallest Holm-adjusted p among units with R2_count > R2_pT.
     Jet-level check: per live S unit, held-out R^2 (OLS, half / half) of S_c on count features (n_real, counts in 10 r bins of
     0.04 rad) vs pT-weighted features (sum pT r^n, n = 0..3).
 (d) shift response. Every real particle moved by 0.06 rad along +deta (axis fixed; invariant mass unchanged). W -> Z = fraction
     of true-W jets called Z. Controls: rotation of every real particle about the axis by a per-jet angle with equal mean
     displacement (2 arcsin(min(1, delta / (2 rbar_j))), random sign), and a translation of 0.06 rad in a random direction per jet.
     McNemar (paired Z calls of true-W jets) translation vs rotation. Read-out: ridge fit of log m on h3 (train half; alpha chosen on
     a held-out quarter of it), mean shift on the test half; prediction of sqrt(M0 M2): 1/2 dlog(M0 M2); of the invariant mass: 0;
     calibrated ratio = shift / response to deta, dphi x k with log k = the mean prediction. Element present iff McNemar p < 0.05
     with W -> Z larger under the translation and the calibrated-ratio 95 % CI excludes 0. Element statistic: calibrated ratio.
 Per model: Holm over the 4 element p-values; 1,000 jet bootstraps for every statistic.
 Combine: element x regime present iff present (Holm 0.05) in >= 2 of 3 seeds. H0: element statistic equal across regimes; H1: HGQ
   differs: statistic |mean_HGQ - mean_(float, L1)|, exact permutation p over the 1,680 relabellings of the 9 models' regime
   labels (3/3/3); Holm over the 4 elements. Accuracy matching: every model within 1 point of the float seed mean (test.json),
   otherwise flagged.
Writes $AI_RESULTS/jedi/x5_<regime>_s<seed>.json and x5_combined.json.
"""
import argparse, json, os, sys, time, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
CL = ['g', 'q', 'W', 'Z', 't']; W_, Z_ = 2, 3
NB = 1000; NNULL = 200; NNULL_C = 2000; NPERM = 2000; DELTA = 0.06; SLOT = 5
JEDI_L1_ZERO_FRAC = 1 - 20 / 192          # JEDI-linear 3-feature n64 fused l1 kernel: 4 / 8 / 8 nonzero (pT, deta, dphi) of 64 each


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    return o


def holm(p):
    p = np.asarray(p, np.float64); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (n - i) * p[j])); adj[j] = run
    return adj


def ci95(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
    return dict(n10=n10, n01=n01, p=float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0)


# ================================================================================================ networks
class FloatDS:
    """dsets-d64 float forward in numpy (train/models.py DeepSetsTagger) from weights(_zeroed).npz"""
    def __init__(self, path):
        z = np.load(path); self.z = {k: np.asarray(z[k], np.float32) for k in z.files}
        self.shift, self.scale = np.asarray(z['shift'], np.float64), np.asarray(z['scale'], np.float64)
        W1, b1 = self.z['phi_0_weight'], self.z['phi_0_bias']
        self.K1, self.b1 = W1.T.astype(np.float64), b1.astype(np.float64)       # (3, 64) on normalised inputs
        self.kind = 'float'

    def phi1(self, xn): return np.maximum(xn @ self.z['phi_0_weight'].T + self.z['phi_0_bias'], 0)

    def phi2(self, a, m): return np.maximum(a @ self.z['phi_2_weight'].T + self.z['phi_2_bias'], 0) * m[..., None]

    def pool(self, u): return u.sum(1)

    def rho(self, S):
        h1 = np.maximum(S @ self.z['rho_0_weight'].T + self.z['rho_0_bias'], 0)
        h2 = np.maximum(h1 @ self.z['rho_2_weight'].T + self.z['rho_2_bias'], 0)
        h3 = np.maximum(h2 @ self.z['rho_4_weight'].T + self.z['rho_4_bias'], 0)
        return dict(h1=h1, h2=h2, h3=h3, logits=h3 @ self.z['head_weight'].T + self.z['head_bias'])


class HgqDS:
    """HGQ2 dsets-d64 fixed-point forward (input quantizers + fused quantized kernels, as interp.jedi_core) from model.keras"""
    def __init__(self, run_dir):
        import keras, hgq  # noqa: F401
        import jax, jax.numpy as jnp
        self.model = keras.models.load_model(Path(run_dir) / 'model.keras', compile=False)
        arch = json.loads((Path(run_dir) / 'arch.json').read_text())
        self.shift, self.scale = np.asarray(arch['shift'], np.float64), np.asarray(arch['scale'], np.float64)
        L = {k: self.model.get_layer(k) for k in ('phi1', 'phi2', 'pool', 'h1', 'h2', 'h3', 'head')}; self.L = L; KB = {}
        for k in ('phi1', 'phi2', 'h1', 'h2', 'h3', 'head'):
            K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))
        self.KB = KB; self.K1 = np.asarray(KB['phi1'][0], np.float64); self.b1 = np.asarray(KB['phi1'][1], np.float64).reshape(-1)
        self.kind = 'hgq'

        def dense(k, x, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1]
            return jax.nn.relu(z) if act else z
        self._phi1 = jax.jit(lambda x: dense('phi1', x))
        self._phi2 = jax.jit(lambda a, m: dense('phi2', a) * m[..., None])
        self._pool = jax.jit(lambda u: jnp.sum(L['pool'].iq(u, training=False), axis=1) * L['pool'].scale)

        def rho(S):
            h1 = dense('h1', S); h2 = dense('h2', h1); h3 = dense('h3', h2)
            return dict(h1=h1, h2=h2, h3=h3, logits=dense('head', h3, act=False))
        self._rho = jax.jit(rho)

    def phi1(self, xn): return np.asarray(self._phi1(np.asarray(xn, np.float32)))
    def phi2(self, a, m): return np.asarray(self._phi2(np.asarray(a, np.float32), np.asarray(m, np.float32)))
    def pool(self, u): return np.asarray(self._pool(np.asarray(u, np.float32)))
    def rho(self, S): return {k: np.asarray(v) for k, v in self._rho(np.asarray(S, np.float32)).items()}


def norm(net, x):
    m = (x[..., 0] > 0).astype(np.float32)
    xn = ((x.astype(np.float64) - net.shift) / net.scale).astype(np.float32) * m[..., None]
    return xn, m


def run(net, x, ov_a=None, ov_S=None, keep=('S', 'logits'), B=2500, slot=None):
    """forward on raw particles; ov_a = (unit, value): phi1 unit replaced by value; ov_S = (unit, value) on S"""
    out = {k: [] for k in keep}
    for i in range(0, len(x), B):
        xn, m = norm(net, x[i:i + B]); a = net.phi1(xn)
        if ov_a is not None: a = a.copy(); a[..., ov_a[0]] = ov_a[1]
        u = net.phi2(a, m); S = net.pool(u)
        if ov_S is not None: S = S.copy(); S[:, ov_S[0]] = ov_S[1]
        r = dict(a=a, u=u, S=S, **net.rho(S))
        for k in keep: out[k].append(np.asarray(r[k] if (slot is None or k not in ('a', 'u')) else r[k][:, slot], np.float64 if k not in ('a', 'u') else np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


def moments(x):
    x = np.asarray(x, np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]; r2 = eta ** 2 + phi ** 2
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=pt.sum(1), M2=(pt * r2).sum(1), m=m, logm=np.log(np.maximum(m, 1e-3)))


def edit(x, d_eta=0.0, d_phi=0.0, k=1.0):
    e = x.copy(); rl = (e[..., 0] > 0)
    d_eta = np.broadcast_to(np.asarray(d_eta, np.float32).reshape(-1, 1) if np.ndim(d_eta) else np.float32(d_eta), rl.shape)
    d_phi = np.broadcast_to(np.asarray(d_phi, np.float32).reshape(-1, 1) if np.ndim(d_phi) else np.float32(d_phi), rl.shape)
    e[..., 1] = np.where(rl, x[..., 1] * np.float32(k) + d_eta, x[..., 1]); e[..., 2] = np.where(rl, x[..., 2] * np.float32(k) + d_phi, x[..., 2])
    return e


def rotate(x, th):
    e = x.copy(); rl = e[..., 0] > 0; c, s = np.cos(th).astype(np.float32)[:, None], np.sin(th).astype(np.float32)[:, None]
    e[..., 1] = np.where(rl, c * x[..., 1] - s * x[..., 2], x[..., 1]); e[..., 2] = np.where(rl, s * x[..., 1] + c * x[..., 2], x[..., 2])
    return e


class Ridge:
    ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)

    def __init__(self, X, t, i_fit, i_val, i_all):
        self.mu = X[i_all].mean(0); self.sd = X[i_all].std(0); self.sd[self.sd < 1e-6] = np.inf
        Z = (X - self.mu) / self.sd; self.tm, self.ts = t[i_all].mean(), t[i_all].std(); tt = (t - self.tm) / self.ts; best = None
        for a in self.ALPHAS:
            w = self._fit(Z[i_fit], tt[i_fit], a); e = ((Z[i_val] @ w - tt[i_val]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.alpha = float(best[1]); self.w = self._fit(Z[i_all], tt[i_all], self.alpha)

    @staticmethod
    def _fit(Z, t, a):
        Zc = Z - Z.mean(0); return np.linalg.solve(Zc.T @ Zc + a * len(Z) * np.eye(Z.shape[1]), Zc.T @ (t - t.mean()))

    def __call__(self, X): return ((X - self.mu) / self.sd) @ self.w * self.ts + self.tm


# ================================================================================================ per-model analysis
def analyse(a):
    from interp.jedi_core import sample
    R = dict(regime=a.regime, seed=a.seed, dir=a.dir, plan='module docstring (frozen)', sample=dict(per_class=2000, seed=311))
    JS = OUT / f'x5_{a.label or a.regime}_s{a.seed}.json'
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    rd = Path(a.dir)
    if a.regime == 'hgq':
        net = HgqDS(rd); tj = json.loads((rd / 'test.json').read_text()); probs_f = 'test_probs.npy'
    else:
        net = FloatDS(rd / ('weights_zeroed.npz' if a.regime == 'l1' else 'weights.npz'))
        tj = json.loads((rd / ('test_zeroed.json' if a.regime == 'l1' else 'test.json')).read_text())
        probs_f = 'test_probs_zeroed.npy' if a.regime == 'l1' else 'test_probs.npy'
    x, y, _ = sample(per_class=2000, seed=311); x = np.ascontiguousarray(x); J = len(x)
    from jetdistill.config import RESULTS
    yy = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(311)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yy == c), 2000, replace=False) for c in range(5)])); assert np.array_equal(yy[idx], y)
    real = x[..., 0] > 0; nreal = real.sum(1)
    BI = np.random.default_rng(3110).integers(0, J, size=(NB, J))
    # ---------------- step 0
    r0 = run(net, x, keep=('a', 'S', 'h3', 'logits')); c0 = r0['logits'].argmax(1)
    acc = float((c0 == y).mean()); exp_bal = float(np.mean(list(tj['per_class_accuracy'].values())))
    se = np.sqrt(exp_bal * (1 - exp_bal) / J)
    st = dict(N=J, acc=acc, acc_ci=wilson(int((c0 == y).sum()), J), test_json_acc=tj['accuracy'], balanced_expectation=exp_bal,
              z=(acc - exp_bal) / se, call_frac=(np.bincount(c0, minlength=5) / J).tolist())
    pp = rd / probs_f
    if pp.exists() and len(np.load(pp, mmap_mode='r')) == len(yy):
        P = np.load(pp, mmap_mode='r')[idx]; lo = r0['logits']; q = np.exp(lo - lo.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        st.update(max_abs_dprob_vs_test_probs=float(np.abs(q - P).max()), argmax_agreement=float((P.argmax(1) == c0).mean()))
    R['step0'] = st; log('step0', st); save()
    # ---------------- step 1: live units
    amean = r0['a'][real].mean(0)
    chA = np.zeros((J, 64), bool)
    for c in range(64):
        chA[:, c] = run(net, x, ov_a=(c, amean[c]), keep=('logits',))['logits'].argmax(1) != c0
    Smean = r0['S'].mean(0); chS = np.zeros((J, 64), bool)
    for c in range(64):
        S = r0['S'].copy(); S[:, c] = Smean[c]; chS[:, c] = net.rho(S)['logits'].argmax(1) != c0
    liveA = np.flatnonzero(chA.mean(0) >= 0.01); liveS = np.flatnonzero(chS.mean(0) >= 0.01)
    R['step1'] = dict(phi1_cost=chA.mean(0), S_cost=chS.mean(0), live_phi1=liveA, live_S=liveS, n_live_phi1=len(liveA), n_live_S=len(liveS))
    log('step1 live phi1', len(liveA), 'live S', len(liveS)); save()
    del r0['a']
    # ---------------- (a) cut types
    alpha = net.K1 / net.scale[:, None]                                      # (3, 64) physical
    beta = net.b1 - (alpha * net.shift[:, None]).sum(0)
    xr = x[real].astype(np.float64); sdx = xr.std(0)

    def types(al, sd, units):
        w2 = (np.abs(al[:, units]) * sd[:, None]) ** 2; tot = w2.sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            spt = w2[0] / tot; sang = (w2[1] + w2[2]) / tot
        t = np.where(tot == 0, 'constant', np.where(spt >= 0.9, 'ramp', np.where(sang >= 0.9, 'angular', 'mixed')))
        return t, spt
    t_obs, spt = types(alpha, sdx, liveA)
    P_obs = float(np.isin(t_obs, ['ramp', 'angular']).mean()) if len(liveA) else np.nan
    # null (X6v): statistic over the units with any nonzero weight (the observed value on the same set: P_nz), each feature's
    # column of alpha permuted over the 64 units; units left with all-zero weights after the permutation are excluded
    nzu = np.flatnonzero((alpha != 0).any(0)); t_nz, _ = types(alpha, sdx, nzu)
    P_nz = float(np.isin(t_nz, ['ramp', 'angular']).mean()) if len(nzu) else np.nan
    nr = np.random.default_rng(31101); Pn = []
    for _ in range(NNULL):
        ap_ = np.stack([alpha[f, nr.permutation(64)] for f in range(3)]); u_ = np.flatnonzero((ap_ != 0).any(0))
        tn, _ = types(ap_, sdx, u_); Pn.append(np.isin(tn, ['ramp', 'angular']).mean() if len(u_) else np.nan)
    Pn = np.array(Pn); p_a = float((1 + (Pn >= P_nz - 1e-12).sum()) / (1 + NNULL))
    # jet bootstrap: live set and feature sd from resampled jets
    sx = np.stack([(x[..., f] * real).sum(1) for f in range(3)], 1).astype(np.float64)
    sx2 = np.stack([((x[..., f].astype(np.float64)) ** 2 * real).sum(1) for f in range(3)], 1); cnt = nreal.astype(np.float64)
    Pb = []
    for b in range(NB):
        i = BI[b]; n_ = cnt[i].sum(); mu = sx[i].sum(0) / n_; sd_ = np.sqrt(np.maximum(sx2[i].sum(0) / n_ - mu ** 2, 0))
        lv = np.flatnonzero(chA[i].mean(0) >= 0.01); tb, _ = types(alpha, sd_, lv)
        Pb.append(np.isin(tb, ['ramp', 'angular']).mean() if len(lv) else np.nan)
    K1z = (net.K1 == 0)
    R['a'] = dict(P=P_obs, P_ci=ci95(Pb), P_nonzero_units=P_nz, n_nonzero_units=len(nzu), null_q=[float(np.quantile(Pn, q)) for q in (.025, .5, .975)], p=p_a,
                  counts={t: int((t_obs == t).sum()) for t in ('ramp', 'angular', 'mixed', 'constant')},
                  phi1_zero_frac=float(K1z.mean()), phi1_nonzero_per_feature=(~K1z).sum(1), jedi_l1_zero_frac=JEDI_L1_ZERO_FRAC,
                  units=[dict(unit=int(u), type=str(t), pT_share=float(s), cost=float(chA[:, u].mean()), alpha=alpha[:, u], beta=float(beta[u]))
                         for u, t, s in zip(liveA, t_obs, spt)])
    log('(a)', {k: R['a'][k] for k in ('P', 'P_ci', 'null_q', 'p', 'counts', 'phi1_zero_frac')}); save()
    # ---------------- (b) rank windows
    sel = np.flatnonzero(nreal >= 18); xs = x[sel]; S0 = r0['S'][sel]; sdS = r0['S'].std(0); sdS[sdS == 0] = np.inf
    m_ang, m_row = [], []
    for k in range(17):
        e = xs.copy(); e[:, [k, k + 1], 1:] = xs[:, [k + 1, k], 1:]
        m_ang.append((np.abs(run(net, e, keep=('S',))['S'][:, liveS] - S0[:, liveS]) / sdS[liveS]).mean(1) if len(liveS) else np.zeros(len(sel)))
        e = xs.copy(); e[:, [k, k + 1], :] = xs[:, [k + 1, k], :]
        m_row.append((np.abs(run(net, e, keep=('S',))['S'][:, liveS] - S0[:, liveS]) / sdS[liveS]).mean(1) if len(liveS) else np.zeros(len(sel)))
    m_ang, m_row = np.array(m_ang), np.array(m_row)                      # (17, Jsel)
    pr = np.random.default_rng(31102); pk, ck = [], []
    for k in range(1, 16):
        d = m_ang[k] - 0.5 * (m_ang[k - 1] + m_ang[k + 1]); ck.append(m_ang[k].mean() / max(0.5 * (m_ang[k - 1].mean() + m_ang[k + 1].mean()), 1e-300))
        obs = d.mean(); cntp = 0
        for i in range(0, NPERM, 250):
            s = pr.choice([-1.0, 1.0], size=(min(250, NPERM - i), len(d))); cntp += int(((s * d).mean(1) >= obs - 1e-15).sum())
        pk.append((1 + cntp) / (1 + NPERM))
    pk, ck = np.array(pk), np.array(ck); ph = holm(pk); okk = ck >= 1.5
    p_b = float(ph[okk].min()) if okk.any() else 1.0
    BIs = np.random.default_rng(31103).integers(0, len(sel), size=(NB, len(sel)))
    cb = [max(m_ang[k, i].mean() / max(0.5 * (m_ang[k - 1, i].mean() + m_ang[k + 1, i].mean()), 1e-300) for k in range(1, 16)) for i in BIs[:NB]]
    Rb = dict(N=len(sel), m_angle_swap=m_ang.mean(1), m_angle_swap_ci=[ci95(m_ang[k][BIs].mean(1)) for k in range(17)],
              m_row_swap=m_row.mean(1), m_row_swap_max=m_row.max(1), contrast_k1_15=ck, p_k1_15=pk, p_holm=ph,
              max_contrast=float(ck.max()), max_contrast_k=int(np.argmax(ck) + 1), max_contrast_ci=ci95(cb), p=p_b,
              row_over_angle=float(m_row.mean() / max(m_ang.mean(), 1e-300)))
    if a.regime == 'hgq':
        L = net.L['pool']; q = L.iq.quantizer
        try:
            bits = np.asarray(q.bits, np.float64) if hasattr(q, 'bits') else np.maximum(sum(np.asarray(getattr(q, t), np.float64) for t in ('k', 'i', 'f')), 0)
        except Exception:  # noqa: BLE001
            k_, i_, f_ = (np.asarray(getattr(q, t), np.float64) for t in ('k', 'i', 'f')); bits = np.maximum(k_ + i_ + f_, 0)
        bits = np.broadcast_to(bits.reshape(bits.shape[-2:]) if bits.ndim >= 2 else bits, (64, 64)); kept = bits > 0
        ch = []
        for c in liveS:
            col = kept[:, c]; ks = np.flatnonzero(col)
            runs = np.split(ks, np.flatnonzero(np.diff(ks) != 1) + 1) if len(ks) else []
            ch.append(dict(unit=int(c), kept=int(len(ks)), runs=[[int(r[0]), int(r[-1])] for r in runs],
                           contiguity=float(max(len(r) for r in runs) / len(ks)) if len(ks) else None,
                           edges=[int(e) for e in np.flatnonzero(np.diff(col.astype(int)) != 0)]))
        Rb['pool_masks'] = dict(zero_frac=float((~kept).mean()), slot_varying_channels=int(sum((kept[:, c] != kept[0, c]).any() for c in range(64))),
                                live_channels=ch)
    R['b'] = Rb; log('(b)', {k: Rb[k] for k in ('max_contrast', 'max_contrast_k', 'p', 'row_over_angle')}); save()
    # ---------------- (c) count-weighted angular objects (single-particle grid at slot SLOT)
    PT = np.array([1., 3., 10., 30., 100.]); RR = np.linspace(0, 0.4, 41); TH = np.deg2rad(np.arange(0, 360, 5))
    g = np.stack(np.meshgrid(PT, RR, TH, indexing='ij'), -1).reshape(-1, 3)
    xg = np.zeros((len(g), 64, 3), np.float32); xg[:, SLOT, 0] = g[:, 0]; xg[:, SLOT, 1] = g[:, 1] * np.cos(g[:, 2]); xg[:, SLOT, 2] = g[:, 1] * np.sin(g[:, 2])
    xg[:, :SLOT, 0] = 0  # empty slots before SLOT are padding (masked)
    rg = run(net, xg, keep=('a', 'u'), slot=SLOT); F = rg['u'].reshape(len(PT), len(RR), len(TH), 64).astype(np.float64)
    A1 = rg['a'].reshape(len(PT), len(RR), len(TH), 64).astype(np.float64)

    def shares(f):
        tot = ((f - f.mean()) ** 2).sum(); mp = f.mean((1, 2), keepdims=True); vp = ((mp - f.mean()) ** 2).sum() * 1.0
        vp = ((np.broadcast_to(mp, f.shape) - f.mean()) ** 2).sum()
        A = 1 - vp / tot if tot > 0 else 0.0
        ring = f.mean(2, keepdims=True); den = ((f - mp) ** 2).sum()
        Rr = 1 - ((f - ring) ** 2).sum() / den if den > 1e-12 * max(tot, 1e-300) else np.nan
        h = ring[..., 0] - ring[..., 0].mean(1, keepdims=True)                 # (pT, r)
        hs = (h ** 2).sum()
        if hs <= 0: return A, Rr, np.nan, np.nan
        r2c = 1 - ((h - h.mean(0, keepdims=True)) ** 2).sum() / hs
        h1 = (PT[:, None] * h).sum(0) / (PT ** 2).sum(); r2p = 1 - ((h - PT[:, None] * h1[None]) ** 2).sum() / hs
        return A, Rr, r2c, r2p
    angA = [u for u in liveA if (np.abs(alpha[1:, u]) > 0).any()]
    nl = np.random.default_rng(31104); Rnull = []
    if len(angA):
        Am = A1[..., angA]
        for _ in range(NNULL_C):
            w = nl.normal(size=len(angA)); s = Am @ w; f = np.maximum(s - np.median(s), 0); Rnull.append(shares(f)[1])
    Rnull = np.array([v for v in Rnull if np.isfinite(v)])
    rows = []
    for c in liveS:
        A, Rr, r2c, r2p = shares(F[..., c])
        p = float((1 + (Rnull >= Rr - 1e-12).sum()) / (1 + len(Rnull))) if (np.isfinite(Rr) and len(Rnull)) else 1.0
        rows.append(dict(unit=int(c), cost=float(chS[:, c].mean()), A=A, R=Rr, R2_count=r2c, R2_pT=r2p, p=p))
    tested = [i for i, r_ in enumerate(rows) if r_['A'] >= 0.05 and np.isfinite(r_['R'])]
    ph = holm([rows[i]['p'] for i in tested]) if tested else []
    for i, h_ in zip(tested, ph): rows[i]['p_holm'] = float(h_)
    obj = [r_ for r_ in rows if r_.get('p_holm', 1) < 0.05 and np.isfinite(r_['R2_count']) and r_['R2_count'] > r_['R2_pT']]
    cand = sorted(r_.get('p_holm', 1.0) for r_ in rows if np.isfinite(r_['R2_count']) and r_['R2_count'] > r_['R2_pT'] and 'p_holm' in r_)
    p_c = float(cand[1]) if len(cand) >= 2 else 1.0
    frac = len(obj) / len(liveS) if len(liveS) else np.nan
    fb = [np.isin(np.array([r_['unit'] for r_ in obj]), np.flatnonzero(chS[i].mean(0) >= 0.01)).sum() / max((chS[i].mean(0) >= 0.01).sum(), 1) for i in BI]
    # jet-level check
    rr = np.sqrt(x[..., 1].astype(np.float64) ** 2 + x[..., 2] ** 2); pt = x[..., 0].astype(np.float64)
    Fc = np.column_stack([nreal] + [((rr >= lo) & (rr < lo + 0.04) & real).sum(1) for lo in np.arange(0, 0.4, 0.04)] + [((rr >= 0.4) & real).sum(1)]).astype(np.float64)
    Fp = np.column_stack([(pt * rr ** n * real).sum(1) for n in range(4)])
    hp = np.random.default_rng(31105).permutation(J); i1, i2 = hp[:J // 2], hp[J // 2:]

    def r2ho(X, t):
        X1 = np.column_stack([np.ones(len(X)), X]); w = np.linalg.lstsq(X1[i1], t[i1], rcond=None)[0]; e = t[i2] - X1[i2] @ w
        return float(1 - (e ** 2).sum() / ((t[i2] - t[i2].mean()) ** 2).sum()) if t[i2].std() > 0 else np.nan
    jl = [dict(unit=int(c), R2_count=r2ho(Fc, r0['S'][:, c]), R2_pT=r2ho(Fp, r0['S'][:, c])) for c in liveS]
    R['c'] = dict(slot=SLOT, units=rows, n_tested=len(tested), n_objects=len(obj), frac=frac, frac_ci=ci95(fb), p=p_c,
                  null_R_q=[float(np.quantile(Rnull, q)) for q in (.5, .95, .99)] if len(Rnull) else None, n_null=len(Rnull),
                  n_angular_phi1=len(angA), jet_level=jl,
                  jet_level_median=dict(R2_count=float(np.nanmedian([j['R2_count'] for j in jl])) if jl else None,
                                        R2_pT=float(np.nanmedian([j['R2_pT'] for j in jl])) if jl else None))
    log('(c)', {k: R['c'][k] for k in ('n_tested', 'n_objects', 'frac', 'p', 'null_R_q', 'jet_level_median')}); save()
    del rg, F, A1
    # ---------------- (d) shift response
    M = moments(x); perm = np.random.default_rng(31106).permutation(J); tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:])
    trf, trv = tr[: 3 * len(tr) // 4], tr[3 * len(tr) // 4:]
    rd_ = Ridge(r0['h3'], M['logm'], trf, trv, tr); isW = y == W_
    rng_ = np.random.default_rng(31107)
    rbar = (pt * rr * real).sum(1) / np.maximum((pt * real).sum(1), 1e-9)
    th = 2 * np.arcsin(np.minimum(1, DELTA / (2 * np.maximum(rbar, 1e-9)))) * rng_.choice([-1, 1], J)
    ang = rng_.uniform(0, 2 * np.pi, J)
    E = {'translate_eta': edit(x, d_eta=DELTA), 'rotation': rotate(x, th),
         'translate_random_dir': edit(x, d_eta=DELTA * np.cos(ang), d_phi=DELTA * np.sin(ang))}
    Me = moments(E['translate_eta']); pred = 0.5 * (np.log(Me['M0'] * Me['M2']) - np.log(M['M0'] * M['M2']))
    lk = float(pred[te].mean()); E['matched_scale'] = edit(x, k=float(np.exp(lk)))
    base = rd_(r0['h3']); Rd = dict(readout_alpha=rd_.alpha, readout_r2_test=float(1 - ((base[te] - M['logm'][te]) ** 2).sum() / ((M['logm'][te] - M['logm'][te].mean()) ** 2).sum()),
                                    pred_half_dlogM0M2=lk, dlogm_invariant_max=float(np.abs(np.log(Me['m'] / M['m'])[M['m'] > 0]).max()))
    calls, dm = {}, {}
    for k, xe in E.items():
        r = run(net, xe, keep=('h3', 'logits')); calls[k] = r['logits'].argmax(1); dm[k] = rd_(r['h3']) - base
    zW0 = (c0[isW] == Z_)
    for k in E:
        zW = calls[k][isW] == Z_
        Rd[k] = dict(calls_changed=float((calls[k] != c0).mean()), WtoZ=float(zW.mean()), WtoZ_ci=wilson(int(zW.sum()), int(isW.sum())),
                     dlogmhat=float(dm[k][te].mean()), dlogmhat_ci=ci95(dm[k][te][np.random.default_rng(31108).integers(0, len(te), (NB, len(te)))].mean(1)))
    Rd['WtoZ_unedited'] = float(zW0.mean())
    Rd['mcnemar_translate_vs_rotation'] = mcnemar(calls['translate_eta'][isW] == Z_, calls['rotation'][isW] == Z_)
    Rd['mcnemar_translate_vs_random_dir'] = mcnemar(calls['translate_eta'][isW] == Z_, calls['translate_random_dir'][isW] == Z_)
    bt = np.random.default_rng(31109).integers(0, len(te), (NB, len(te)))
    ob, mt = dm['translate_eta'][te], dm['matched_scale'][te]
    rb = ob[bt].mean(1) / np.where(np.abs(mt[bt].mean(1)) > 1e-12, mt[bt].mean(1), np.nan)
    Rd['calibrated_ratio'] = float(ob.mean() / mt.mean()) if abs(mt.mean()) > 1e-12 else None
    Rd['calibrated_ratio_ci'] = ci95(rb); Rd['raw_ratio_vs_pred'] = float(ob.mean() / lk)
    mc = Rd['mcnemar_translate_vs_rotation']; more = Rd['translate_eta']['WtoZ'] > Rd['rotation']['WtoZ']
    lo_ = Rd['calibrated_ratio_ci'][0]; hi_ = Rd['calibrated_ratio_ci'][1]
    excl0 = lo_ is not None and (lo_ > 0 or hi_ < 0)
    Rd['p'] = float(mc['p']) if (more and excl0) else 1.0
    R['d'] = Rd; log('(d)', {k: Rd[k] for k in ('WtoZ_unedited', 'calibrated_ratio', 'calibrated_ratio_ci', 'p')}, Rd['translate_eta'], Rd['rotation']); save()
    # ---------------- presence
    pe = dict(a=R['a']['p'], b=R['b']['p'], c=R['c']['p'], d=R['d']['p']); ph = holm(list(pe.values()))
    crit = dict(a=True, b=True, c=R['c']['n_objects'] >= 2, d=True)
    R['presence'] = {k: dict(p=pe[k], p_holm=float(h_), present=bool(h_ < 0.05 and crit[k])) for k, h_ in zip(pe, ph)}
    R['stats'] = dict(a=R['a']['P'], b=R['b']['max_contrast'], c=R['c']['frac'], d=R['d']['calibrated_ratio'])
    R['wall_s'] = time.time() - T0; log('presence', R['presence']); save()


# ================================================================================================ combine
def combine(hgq_label='hgq'):
    regs = ('float', 'l1', 'hgq'); D = {}
    for r in regs:
        for s in (0, 1, 2):
            p = OUT / f'x5_{hgq_label if r == "hgq" else r}_s{s}.json'
            if p.exists(): D[(r, s)] = json.loads(p.read_text())
    C = dict(hgq_label=hgq_label, models=sorted(f'{r}_s{s}' for r, s in D), table={}, regime_tests={})
    fl = [D[('float', s)]['step0']['test_json_acc'] for s in (0, 1, 2) if ('float', s) in D]; fm = float(np.mean(fl)) if fl else None
    C['accuracy'] = {f'{r}_s{s}': dict(sample=v['step0']['acc'], test=v['step0']['test_json_acc'], within_1pt=abs(v['step0']['test_json_acc'] - fm) <= 0.01,
                                       phi1_zero_frac=v['a']['phi1_zero_frac']) for (r, s), v in D.items()}
    C['float_mean_test_acc'] = fm
    for el in 'abcd':
        C['table'][el] = {}
        for r in regs:
            pr = [D[(r, s)]['presence'][el]['present'] for s in (0, 1, 2) if (r, s) in D]
            st = [D[(r, s)]['stats'][el] for s in (0, 1, 2) if (r, s) in D]
            stf = [v for v in st if v is not None]
            C['table'][el][r] = dict(present_seeds=int(sum(pr)), n=len(pr), present=bool(sum(pr) >= 2), stat_mean=float(np.mean(stf)) if stf else None,
                                     stat_range=[float(min(stf)), float(max(stf))] if stf else None, stats=st)
    pv = []
    for el in 'abcd':
        keys = [(r, s) for r in regs for s in (0, 1, 2) if (r, s) in D and D[(r, s)]['stats'][el] is not None]
        v = np.array([D[k]['stats'][el] for k in keys], np.float64); lab = np.array([k[0] for k in keys])
        if len(keys) != 9:
            C['regime_tests'][el] = dict(n=len(keys), p=None); pv.append(1.0); continue
        stat = lambda L_: abs(v[L_ == 'hgq'].mean() - v[L_ != 'hgq'].mean())
        obs = stat(lab); cnt = tot = 0
        base = ['float'] * 3 + ['l1'] * 3 + ['hgq'] * 3
        for P_ in set(itertools.permutations(base)):
            tot += 1; cnt += stat(np.array(P_)) >= obs - 1e-12
        C['regime_tests'][el] = dict(n=9, stat_obs=float(obs), p=cnt / tot, n_relabellings=tot,
                                     H0='element statistic equal across regimes', H1='HGQ mean differs from the float / L1 mean')
        pv.append(cnt / tot)
    for el, h_ in zip('abcd', holm(pv)): C['regime_tests'][el]['p_holm'] = float(h_)
    (OUT / ('x5_combined.json' if hgq_label == 'hgq' else f'x5_combined_{hgq_label}.json')).write_text(json.dumps(clean(C), indent=1)); log(json.dumps(clean(C), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', choices=['float', 'l1', 'hgq']); ap.add_argument('--seed', type=int); ap.add_argument('--dir')
    ap.add_argument('--combine', action='store_true')
    ap.add_argument('--label', default='', help='output file label (default: the regime), e.g. hgq_acc for a second HGQ arm')
    ap.add_argument('--hgq_label', default='hgq', help='--combine: label of the HGQ arm')
    a = ap.parse_args()
    combine(a.hgq_label) if a.combine else analyse(a)
