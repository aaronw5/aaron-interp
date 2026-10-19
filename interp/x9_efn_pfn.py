"""X9 (claims A3, A4, A5, D4, B2; review weakness 1): are JEDI-linear's program elements (pT-weighted radial objects over
pT-rank windows, a non-recentred mass response) also found in an Energy Flow Network (EFN, latent L_k = Σ_i z_i Φ_k(Δη_i, Δφ_i))
and a Particle Flow Network (PFN, L_k = Σ_i Φ_k(pT_i, Δη_i, Δφ_i)) trained on the same 3-feature, 64-particle input?

Models: trained by interp/x9_train.py (energyflow 1.x / tensorflow-cpu in /j-jepa-vol/aaron-interp/venv-ef); their Dense
weights are exported to weights.npz and run here by an exact numpy forward pass (ReLU everywhere, softmax output), checked
against the TensorFlow softmax on sample(2000, 313) (check.npz).  This module runs in the base jax env next to
interp.jedi_core, so every statistic is computed on the same jets for the EFN/PFN and for JEDI-linear (paired).
  ./job.sh x9-run-efn0 'python -m interp.x9_efn_pfn --model efn_100-100-128_s0'
  ./job.sh x9-sum 'python -m interp.x9_efn_pfn --summary'          (Holm over all model-vs-JEDI tests, figure data)

FROZEN HEADER (written before the analysis jobs; thresholds not tuned per model)
 Samples: T = sample(per_class=2000, seed=313) (N = 10,000 test jets) for every test; Fs = sample(per_class=6000, seed=0)
   with the T rows removed for fits (MARS naming, ridge read-outs): 2/3 fit, 1/3 dev (permutation seed 0).
 Step 0  accuracy of the unedited JEDI-linear (fixed point, jedi_core) and of the model on T; |TF − numpy| softmax check.
 Step 1  (E2 analog) first Φ layer units of the model, w·x + b on the model input. Live = mean ablation over real particles
   changes >= 1 % of calls on T[::2]. Typing as X6 step 2 (term spread w_k·sd(x_k) over real particles): ramp |w_ang|/|w| <
   0.05; angular |w_pT|/|w| < 0.05 (half-plane through the axis if the line's distance |b + w_pT·median(pT)| / |w_ang| <
   0.01 rad in physical units, else angle-offset); mixed otherwise.  Null (X6v): each input row of the kernel permuted
   across units independently, 20,000 draws; statistic = fraction of live units that are pure (ramp or angular); p =
   P(null >= obs).  EFN: Φ has no pT input by construction (all units angular); only the line distances are reported.
 Step 2  (E5 naming) latent units L_k. Cost = % calls changed on T when L_k is replaced by its value on another jet
   (resample ablation, one fixed derangement, seed 0); live = cost >= 0.1 %.  Each live unit is MARS-fitted
   (jetdistill.mars.fit_neuron, at most 15 terms, 6,000 selection jets) on the E11/E5 library: the 380 finite Q observables
   + the rank-window sums of pT, pT·r, pT·r², r, r², counts and hinges (interp.e11_mars_all.handbuilt), held-out R² on T,
   top term by weight share and its family (E11 families).  JEDI reference: results/jedi/e11_fit_a_S.json (same library,
   same fit sample, 30 terms; dev R² path compared at 1 / 3 / 15 terms).  Direct radial test (exact, no fit): Φ_k on a
   grid r 0–0.4 rad (41) × θ 0–355° (72) (× pT 1, 3, 10, 30, 100 GeV for PFN); radial share = 1 − azimuthal variance /
   total variance (X6 step 5).  Cross-readability: ridge R² of each live JEDI S unit from the model latent and of each live
   model latent unit from JEDI's S (fit on Fs, test on T).
 Step 3  edits on T (padding never moved):
   (a) translation of every real particle by δ = 0.06 rad along +η and +φ (m exactly unchanged). Metrics: % calls changed;
       W → Z among true-W jets; read-out Δlog m̂ (ridge of log m on the latent, and on the last F hidden layer; for JEDI on S
       and z; fitted on unedited Fs) vs the prediction ½Δlog M2 (√(M0·M2) about the stored axis) and vs 0 (invariant m).
       Calibration: angle scaling a → k a of every real particle with log k = the mean ½Δlog M2 of the edit (a real change of
       m of the predicted mean size); calibrated ratio = mean Δlog m̂(translation) / mean Δlog m̂(calibration).  Controls: matched rotation about the
       axis (X6 step 6: equal mean displacement, m and M2 unchanged) and a per-jet random-direction translation of 0.06 rad;
       McNemar of the W → Z indicator, translation vs rotation.
   (b) WTA axis (E17: C/A with winner-take-all recombination): the same metrics, prediction ½Δlog(M0·M2) about the new axis.
   (c) rank-window swap (E20 / X6 step 4): swap (Δη, Δφ) of ranks 13|14 (JEDI's hard-window edge) vs the X6 n64 inside pair;
       m = mean over units of |ΔL_c| / sd(L_c) (model: the 10 live latent units of largest cost; JEDI: X6's units for that
       edge); ratio across / inside on common jets with bootstrap CI and paired sign-flip p.  The EFN/PFN are permutation
       invariant: no rank window can exist; the ratio then follows the pT (z) gaps of the swapped pairs, also reported.
 Step 4  (E22 IIA) per live latent unit with a MARS formula: H = the model's own F layers on L̂, where L̂ = L with every live
   unit replaced by its formula value (ReLU); interchange of unit k on 5,000 random ordered (base, source) pairs of T
   (base ≠ source, seed 0): L gets L_k(source), H gets f_k(source).  Effective IIA = agreement of the two calls on the pairs
   where the swap changes the model's call.  Baselines: permuted alignment (200 derangements of the live units; stage mean;
   one-sided permutation p, floor 1/201; per unit p, Holm), random rotations (200 unit directions in the standardised
   live-unit space, V' = V_b + σ r (r·(V_s − V_b)/σ), clipped at 0, in L and L̂).  JEDI on the same pairs: S with the E11 (a)
   formulas, network downstream (run_jetwise).
 Statistics: 1000 jet (or pair) bootstrap 95 % CIs; H0: model statistic = JEDI statistic (paired over the same jets or pairs,
   bootstrap p, two-sided); Holm over the model-vs-JEDI tests (per model here; over all models in --summary).
 Predictions. Generic (JEDI's mechanism is a property of the task + inputs): PFN latents radial and pT-weighted (radial share
   >= 0.9, R² >= 0.8 with ΣpT·r²-type top terms), translation W → Z >= 60 %, calibrated ratio 0.6–1.0; EFN the same with
   radial Φ; IIA above permuted.  JEDI-specific: other top observables, ratio near 0 for translation, no rank windows
   (ratio = z-gap ratio), IIA at the permuted level.
Writes $AI_RESULTS/jedi/x9_<model>.json (partial after every step) and x9_summary.json.
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'; MOD = OUT / 'x9_models'
T0 = time.time(); NB = 1000; W_, Z_ = 2, 3
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
RNG = np.random.default_rng(313)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


# ------------------------------------------------------------------------------------------------ statistics
def boot(fn, n, B=NB, rng=None):
    rng = rng or RNG; v = []
    for _ in range(B):
        i = rng.integers(0, n, n); r = fn(i)
        if r is not None and np.isfinite(r): v.append(r)
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if v else [None, None], np.asarray(v)


def boot_p(vals, null=0.0):
    """two-sided bootstrap p for H0: statistic = null (percentile method)"""
    v = np.asarray(vals)
    if len(v) == 0: return None
    return float(min(1.0, 2 * min((v <= null).mean(), (v >= null).mean()) + 1 / (len(v) + 1)))


def signflip_p(d, P=10000, rng=None):
    rng = rng or RNG; d = np.asarray(d, np.float64); n = len(d)
    if n == 0 or not (d != 0).any(): return 1.0
    obs = abs(d.mean()); c = 0
    for i in range(0, P, 500):
        s = rng.choice([-1.0, 1.0], size=(min(500, P - i), n)); c += int((np.abs((s * d).mean(1)) >= obs - 1e-15).sum())
    return (1 + c) / (1 + P)


def mcnemar(a, b):
    """paired binary a, b: exact two-sided binomial on discordant pairs"""
    from math import comb
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum()); n = n01 + n10
    if n == 0: return dict(n01=0, n10=0, p=1.0)
    k = min(n01, n10); p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return dict(n01=n01, n10=n10, p=float(p))


def holm(ps):
    ps = np.asarray(ps, np.float64); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


# ------------------------------------------------------------------------------------------------ the EFN / PFN forward pass
class Flow:
    """exact numpy forward of an energyflow EFN / PFN exported by x9_train.export"""

    def __init__(self, name):
        self.name = name; self.dir = MOD / name; self.info = json.loads((self.dir / 'info.json').read_text())
        Wz = np.load(self.dir / 'weights.npz'); self.arch = self.info['arch']
        self.phi = [(Wz[f'phi{k}_W'].astype(np.float64), Wz[f'phi{k}_b'].astype(np.float64)) for k in range(len(self.info['phi']))]
        self.F = [(Wz[f'F{k}_W'].astype(np.float64), Wz[f'F{k}_b'].astype(np.float64)) for k in range(len(self.info['F']))]
        self.out = (Wz['out_W'].astype(np.float64), Wz['out_b'].astype(np.float64))
        from jetdistill.config import DATA
        st = json.loads((Path(DATA) / 'norm_stats.json').read_text())['64']
        self.shift, self.scale = np.asarray(st['shift'], np.float64), np.asarray(st['scale'], np.float64)
        self.ell = self.phi[-1][0].shape[1]; self.d1 = self.phi[0][0].shape[1]

    def inputs(self, x):
        x = np.asarray(x, np.float64); real = x[..., 0] > 0
        if self.arch == 'efn':
            pt = x[..., 0] * real; z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12)
            return z, x[..., 1:3] * real[..., None], real
        xn = (x - self.shift) / self.scale
        return None, xn * real[..., None], real

    def tail(self, L, keep_h=False):
        h = L
        for K, b in self.F: h = np.maximum(h @ K + b, 0)
        lo = h @ self.out[0] + self.out[1]
        return (lo, h) if keep_h else lo

    def run(self, x, ov=None, keep=('logits',), batch=2000):
        """ov: {'phi0': (mask (d1,), value (d1,)) per particle, 'L': (mask (ell,), values (J, ell)), 'h': ...}"""
        ov = ov or {}; out = {k: [] for k in keep}
        for i in range(0, len(x), batch):
            z, a, real = self.inputs(x[i:i + batch]); h = a
            for k, (K, b) in enumerate(self.phi):
                h = np.maximum(h @ K + b, 0)
                if k == 0:
                    if 'phi0' in out: out['phi0'].append(h.astype(np.float32))
                    if 'phi0' in ov:
                        m, v = ov['phi0']; h = h * (1 - m) + m * v
            if self.arch == 'efn': L = np.einsum('jn,jnk->jk', z, h)
            else:
                mask = np.any(a != 0, -1).astype(np.float64)          # energyflow PFN masks rows equal to 0 (padding)
                L = np.einsum('jn,jnk->jk', mask, h)
            if 'L' in ov:
                m, v = ov['L']; L = L * (1 - m) + m * np.asarray(v, np.float64)[i:i + batch]
            lo, hh = self.tail(L, keep_h=True)
            if 'L' in out: out['L'].append(L)
            if 'h' in out: out['h'].append(hh)
            if 'logits' in out: out['logits'].append(lo)
        return {k: np.concatenate(v) for k, v in out.items()}

    def phi_map(self, pts, eta, phi):
        """Φ (last Φ layer) on single particles: arrays of equal shape (physical units) -> (..., ell)"""
        sh = np.shape(eta); pt = np.broadcast_to(pts, sh).ravel(); e = np.ravel(eta); p = np.ravel(phi)
        if self.arch == 'efn': h = np.stack([e, p], -1)
        else: h = (np.stack([pt, e, p], -1) - self.shift) / self.scale
        for K, b in self.phi: h = np.maximum(h @ K + b, 0)
        return h.reshape(*sh, -1)


def softmax(lo):
    e = np.exp(lo - lo.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------------------------------------ physics helpers
def moments(xx):
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    r2 = eta ** 2 + phi ** 2
    M0 = pt.sum(1); M2 = (pt * r2).sum(1)
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=M0, M2=M2, m=m, logm=np.log(np.maximum(m, 1e-3)))


def translate(x, d, th):
    e = x.copy(); rl = e[..., 0] > 0; th = np.broadcast_to(np.asarray(th, np.float64), (len(x),))
    e[..., 1] += (d * np.cos(th))[:, None].astype(np.float32) * rl; e[..., 2] += (d * np.sin(th))[:, None].astype(np.float32) * rl
    return e


def rotate_matched(x, d, rng):            # as interp/x6_models_jedi.py step 6
    pt = x[..., 0].astype(np.float64); rl = pt > 0; r = np.hypot(x[..., 1], x[..., 2])
    rbar = (pt * r).sum(1) / np.maximum(pt.sum(1), 1e-12)
    phi = 2 * np.arcsin(np.minimum(1.0, d / np.maximum(2 * rbar, 1e-12))) * rng.choice([-1, 1], len(x))
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]; e = x.copy()
    e[..., 1] = np.where(rl, c * x[..., 1] - s * x[..., 2], x[..., 1]); e[..., 2] = np.where(rl, s * x[..., 1] + c * x[..., 2], x[..., 2])
    return e


def scale_angles(x, k):
    e = x.copy(); rl = e[..., 0] > 0; k = np.asarray(k, np.float64)[:, None]
    e[..., 1] = np.where(rl, x[..., 1] * k, x[..., 1]); e[..., 2] = np.where(rl, x[..., 2] * k, x[..., 2]); return e.astype(np.float32)


def swap(x, k):                            # as interp/x6_models_jedi.py step 4
    nreal = (x[..., 0] > 0).sum(1); J = np.flatnonzero(nreal > k + 1)
    xe = x[J].copy(); xe[:, [k, k + 1], 1:] = xe[:, [k + 1, k], 1:]; return J, xe


ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)


class Ridge:                               # as interp/e6_verify.py
    def __init__(self, X, t, idx_fit, idx_val):
        idx_all = np.concatenate([idx_fit, idx_val])
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
    return float(1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12))


# library: interp/e11_mars_all.py handbuilt (copied verbatim, halfplanes=False) and its families
WINDOWS = {'all': (0, 64), 'top14': (0, 14), 'r1-63': (1, 64), 'r13-63': (13, 64), 'top39': (0, 39), 'top26': (0, 26),
           'top3': (0, 3), 'top11': (0, 11), 'r8-63': (8, 64)}
FAMILIES = ['ΣpT / pT scale', 'multiplicity', 'radial moments (r, r²)', 'mass family', 'N-subjettiness, ECFs', 'subjet / prong', 'pT shares, other']


def handbuilt(x):
    pt, eta, phi = (np.asarray(x[..., k], np.float64) for k in range(3)); real = pt > 0; r = np.hypot(eta, phi)
    H = {}
    for name, (lo, hi) in WINDOWS.items():
        w = real.copy(); w[:, :lo] = False; w[:, hi:] = False; s = f'[{name}]'
        if name != 'all':
            H[f'H:N{s}'] = w.sum(1); H[f'H:ΣpT{s}'] = (pt * w).sum(1)
        H[f'H:ΣpT·r²{s}'] = (pt * r ** 2 * w).sum(1); H[f'H:Σr²{s}'] = (r ** 2 * w).sum(1); H[f'H:Σr{s}'] = (r * w).sum(1)
        H[f'H:ΣpT·r{s}'] = (pt * r * w).sum(1); H[f'H:Σ(r−0.2)₊{s}'] = (np.maximum(r - 0.2, 0) * w).sum(1)
        H[f'H:N(pT>1){s}'] = (w & (pt > 1)).sum(1); H[f'H:N(pT<2){s}'] = (w & (pt < 2)).sum(1)
        H[f'H:Σr[r<0.165]{s}'] = (r * (w & (r < 0.165))).sum(1); H[f'H:N(pT<2,r>0.1){s}'] = (w & (pt < 2) & (r > 0.1)).sum(1)
    M0 = pt.sum(1); M2 = (pt * r ** 2).sum(1)
    H['H:|M1|'] = np.hypot((pt * eta).sum(1), (pt * phi).sum(1)); H['H:√(M0·M2)'] = np.sqrt(M0 * M2)
    return {k: v.astype(np.float64) for k, v in H.items()}


def family(key):                           # as interp/e11_mars_all.py
    k = key[2:] if key.startswith('H:') else key
    if key.startswith('H:'):
        if k.startswith('ΣpT['): return 0
        if k.startswith('N'): return 1
        if k.startswith('√(M0'): return 3
        return 2
    if k in ('sum_pt', 'log_sum_pt') or re.fullmatch(r'sum_pt_top\d+|pt_\d+|soft\d+_pt', k): return 0
    if k in ('n_particles', 'sd_nremoved') or re.fullmatch(r'n_pt_above_\d+|n_dr_.*|n_real_top\d+|n_for_\d+pct', k): return 1
    if 'mass' in k or k in ('m01', 'm012') or k.startswith(('mratio_', 'sj3_pair')): return 3
    if re.fullmatch(r'tau\d+|tau\d+_b2|e[234]|e2_sq|[CDNM][23](_b2)?', k): return 4
    if k in ('sj2_zsoft', 'sj2_dr', 'sd_zg', 'sd_rg', 'dr01', 'dr02', 'dr12', 'dr_min_012', 'dr_max_012', 'pt1_over_pt0', 'pt2_over_pt0',
             'pt_balance01', 'pt1_dr01') or re.fullmatch(r'sj3_(z\d|dr.*)|dr[01]_\d+|ptdr0_\d+|soft\d+_dr0', k): return 5
    if k in ('girth', 'girth2', 'mean_eta', 'mean_eta2', 'mean_phi', 'mean_phi2', 'width', 'lam1', 'lam2', 'max_dr', 'LHA', 'centroid_offset') \
            or re.fullmatch(r'girth2_top\d+|dr_\d+|abseta_\d+|absphi_\d+|eta_\d+|phi_\d+|zdr_\d+|soft\d+_(dr|abseta|absphi)|psi_.*|z_dr_.*', k): return 2
    return 6


def _x6_radial():
    try:
        d = json.loads((OUT / 'x6_3-feature_n64.json').read_text())['E5']['S']; return {k: v['radial_share'] for k, v in d.items()}
    except Exception as e:  # noqa: BLE001
        return repr(e)


def indices(per_class, seed):
    from jetdistill.config import RESULTS
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def evaluate(units, Qd):
    from jetdistill import mars
    J = len(next(iter(Qd.values()))); out = np.zeros((J, len(units)))
    for i, u in enumerate(units):
        v = np.full(J, float(u['intercept']))
        for t in u['terms']: v += t['coef'] * mars.term_value(t, Qd)
        out[:, i] = np.maximum(v, 0)
    return out


# ------------------------------------------------------------------------------------------------ main analysis
def main(a):
    from interp.jedi_core import Jedi, sample
    from jetdistill import mars
    name = a.model; JS = OUT / f'x9_{name}{"_smoke" if a.smoke else ""}.json'; R = dict(model=name, header='see module docstring (frozen)')
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    M = Flow(name); R['info'] = M.info
    log(f'X9 {name}: arch {M.arch}, Φ {M.info["phi"]}, F {M.info["F"]}, latent {M.ell}; training dev acc {M.info["acc_dev"]:.4f}, '
        f'test acc {M.info["acc_test"]:.4f} (N {M.info["n_test"]})')
    # ---------------- samples
    xT, yT, QT = sample(2000, 313); xF, yF, QF = sample(6000, 0)
    iT, iF = indices(2000, 313), indices(6000, 0); keepF = ~np.isin(iF, iT)
    xF, yF = xF[keepF], yF[keepF]; QF = {k: v[keepF] for k, v in QF.items()}
    xT = np.ascontiguousarray(xT, np.float32); xF = np.ascontiguousarray(xF, np.float32); J = len(xT)
    perm = np.random.default_rng(0).permutation(len(xF)); nf = 2 * len(xF) // 3; fit, dev = np.sort(perm[:nf]), np.sort(perm[nf:])
    R['samples'] = dict(T=dict(per_class=2000, seed=313, N=J), F=dict(per_class=6000, seed=0, N=int(len(xF)), removed_overlap=int((~keepF).sum()),
                                                                     fit=int(len(fit)), dev=int(len(dev))))
    log(f'samples: T N={J}; Fs N={len(xF)} (overlap with T removed: {(~keepF).sum()}), fit {len(fit)} dev {len(dev)}')
    # ---------------- step 0
    ck = np.load(M.dir / 'check.npz'); assert np.array_equal(ck['idx'], iT), 'check.npz jets are not sample(2000, 313)'
    base = M.run(xT, keep=('L', 'h', 'logits', 'phi0')); P0 = base.pop('phi0')
    dprob = float(np.abs(softmax(base['logits']) - ck['prob']).max())
    cM = base['logits'].argmax(1); LT, hT = base['L'], base['h']
    net = Jedi(64); RJ = net.run(xT, keep=('S', 'z', 'logits')); cJ = RJ['logits'].argmax(1)
    chk = float(np.abs(net.run_jetwise(xT[:2000], {'S': (np.ones(64), RJ['S'][:2000])})['logits'] - RJ['logits'][:2000]).max())
    accM, accJ = float((cM == yT).mean()), float((cJ == yT).mean())
    R['step0'] = dict(acc_model_T=accM, acc_model_T_ci=wilson(int((cM == yT).sum()), J), acc_jedi_T=accJ, acc_jedi_T_ci=wilson(int((cJ == yT).sum()), J),
                      max_abs_dprob_tf_vs_numpy=dprob, jedi_S_override_identity_max_dlogit=chk,
                      agreement_model_jedi=float((cM == cJ).mean()), acc_model_full_test=M.info['acc_test'],
                      per_class_model=[float((cM[yT == c] == c).mean()) for c in range(5)], per_class_jedi=[float((cJ[yT == c] == c).mean()) for c in range(5)])
    log(f'step 0: acc on T model {accM:.4f}, JEDI {accJ:.4f}; |TF − numpy| softmax max {dprob:.2e}; JEDI S-override identity {chk:.1e}; '
        f'call agreement {R["step0"]["agreement_model_jedi"]:.3f}'); save()
    # ---------------- step 1: first Φ layer
    xs = xT[::2]; c_s = cM[::2]; real_s = xs[..., 0] > 0
    _, a_s, _ = M.inputs(xs); P0s = P0[::2]; mean1 = np.array([P0s[..., k][real_s].mean() for k in range(M.d1)])
    cost1 = np.zeros(M.d1)
    for k in range(M.d1):
        if P0s[..., k][real_s].std() < 1e-12: continue
        m = np.zeros(M.d1); m[k] = 1; v = np.zeros(M.d1); v[k] = mean1[k]
        cost1[k] = float((M.run(xs, ov={'phi0': (m, v)})['logits'].argmax(1) != c_s).mean())
    live1 = np.flatnonzero(cost1 >= 0.01)
    K1, b1 = M.phi[0]; feat_sd = np.array([a_s[..., f][real_s].std() for f in range(a_s.shape[-1])])
    phys_med_pt = float(np.median(xs[..., 0][real_s]))

    def types(Kmat, units):
        out = []
        for u in units:
            w = np.abs(Kmat[:, u]) * feat_sd
            if M.arch == 'efn': wpt, wang = 0.0, np.hypot(w[0], w[1]); al_ang = Kmat[:, u] / M.scale[1:] if False else Kmat[:, u]
            else: wpt, wang = w[0], np.hypot(w[1], w[2])
            tot = np.hypot(wpt, wang)
            if tot == 0: out.append(('constant', None, 0.0)); continue
            fpt, fang = wpt / tot, wang / tot
            # line distance from the axis in physical units (rad): α_ang·a + (α_pT·median pT + β) = 0
            if M.arch == 'efn': al = Kmat[:, u]; beta = b1[u]; dist = abs(beta) / max(np.hypot(*al), 1e-12)
            else:
                al = Kmat[:, u] / M.scale; beta = b1[u] - (Kmat[:, u] * M.shift / M.scale).sum()
                dist = abs(beta + al[0] * phys_med_pt) / max(np.hypot(al[1], al[2]), 1e-12)
            t = 'ramp' if fang < 0.05 else ('half-plane' if dist < 0.01 else 'angle-offset') if fpt < 0.05 else 'mixed'
            out.append((t, float(dist), float(fpt)))
        return out
    ty = types(K1, live1)
    counts = {t: sum(1 for x_ in ty if x_[0] == t) for t in ('ramp', 'half-plane', 'angle-offset', 'mixed', 'constant')}
    pure = lambda tt: np.mean([x_[0] in ('ramp', 'half-plane', 'angle-offset') for x_ in tt]) if len(tt) else np.nan
    obs = pure(ty); null = []
    if M.arch == 'pfn' and len(live1):
        rr = np.random.default_rng(7)
        for _ in range(20000 if a.full else 2000):
            Kp = np.stack([K1[f, rr.permutation(M.d1)] for f in range(K1.shape[0])])
            w = np.abs(Kp[:, live1]) * feat_sd[:, None]; wpt, wang = w[0], np.hypot(w[1], w[2]); tot = np.maximum(np.hypot(wpt, wang), 1e-30)
            null.append(np.mean((wang / tot < 0.05) | (wpt / tot < 0.05)))
        null = np.asarray(null)
    R['step1'] = dict(d1=M.d1, n_live=int(len(live1)), live=live1.tolist(), cost=cost1.round(5).tolist(), types=counts,
                      per_unit=[dict(unit=int(u), cost=float(cost1[u]), type=t, line_dist_rad=d, pt_share=f) for u, (t, d, f) in zip(live1, ty)],
                      pure_fraction=obs, pure_ci=wilson(int(round(obs * len(live1))), len(live1)) if len(live1) else None,
                      pt_share_quantiles=np.quantile([x_[2] for x_ in ty], [0, .25, .5, .75, 1]).tolist() if len(ty) else None,
                      null=dict(draws=int(len(null)), q=np.quantile(null, [.025, .5, .975]).tolist() if len(null) else None,
                                p=float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null))) if len(null) else None),
                      jedi_reference='X6 n64: 13 live l1 units, 4 ramps + 9 half-planes, 0 mixed (exact zero weights)',
                      note='EFN: Φ input is (Δη, Δφ) only, every unit is angular by construction' if M.arch == 'efn' else '')
    log(f'step 1: {len(live1)} live Φ1 units of {M.d1} (cost >= 1 %): {counts}; pure fraction {obs:.3f}; null {R["step1"]["null"]}'); save()
    # ---------------- step 2: latent units, costs, naming
    der = np.random.default_rng(0).permutation(J)
    while (der == np.arange(J)).any(): der = np.random.default_rng(int(der[0])).permutation(J)
    costL = np.zeros(M.ell)
    for k in range(M.ell):
        if LT[:, k].std() < 1e-12: continue
        Lp = LT.copy(); Lp[:, k] = LT[der, k]; costL[k] = float((M.tail(Lp).argmax(1) != cM).mean())
    liveL = np.flatnonzero(costL >= 0.001); order = liveL[np.argsort(-costL[liveL])]
    costS = np.zeros(64)
    for k in range(64):
        if RJ['S'][:, k].std() < 1e-12: continue
        Sp = RJ['S'].copy(); Sp[:, k] = RJ['S'][der, k]; m = np.zeros(64); m[k] = 1
        costS[k] = float((net.run_jetwise(xT, {'S': (m, Sp)})['logits'].argmax(1) != cJ).mean())
    R['step2'] = dict(cost_latent=costL.round(5).tolist(), live=order.tolist(), n_live=int(len(liveL)), n_cost_ge_1pct=int((costL >= 0.01).sum()),
                      n_dead=int((LT.std(0) < 1e-12).sum()), jedi_S_cost=costS.round(5).tolist(), jedi_n_live=int((costS >= 0.001).sum()),
                      jedi_n_cost_ge_1pct=int((costS >= 0.01).sum()))
    log(f'step 2: latent {M.ell}: dead {R["step2"]["n_dead"]}, live (>= 0.1 %) {len(liveL)}, >= 1 % {R["step2"]["n_cost_ge_1pct"]}; '
        f'top costs {[(int(k), round(costL[k] * 100, 2)) for k in order[:10]]}; JEDI S live {R["step2"]["jedi_n_live"]} (>= 1 % {R["step2"]["jedi_n_cost_ge_1pct"]})'); save()
    # radial share of Φ (exact)
    RG = np.linspace(0, 0.4, 41); TH = np.radians(np.arange(0, 360, 5)); PTG = np.array([1., 3., 10., 30., 100.]) if M.arch == 'pfn' else np.array([1.])
    RRg, TTg = np.meshgrid(RG, TH, indexing='ij'); rad = {}
    for k in order:
        f = np.stack([M.phi_map(p, RRg * np.cos(TTg), RRg * np.sin(TTg))[..., k] for p in PTG])  # (pt, r, θ)
        tot = ((f - f.mean()) ** 2).sum(); az = ((f - f.mean(-1, keepdims=True)) ** 2).sum()
        rad[int(k)] = dict(radial_share=float(1 - az / tot) if tot > 1e-12 else None, var=float(tot / f.size),
                           pt_share=float(((f.mean((1, 2), keepdims=True) - f.mean()) ** 2).sum() * f.shape[1] * f.shape[2] / f.size / max(tot / f.size, 1e-30)) if tot > 1e-12 else None)
    l1rad = {}
    for k in live1:   # control: first-layer units (half-planes are not radial)
        K, b = M.phi[0]
        def l1map(p):
            if M.arch == 'efn': hh = np.stack([RRg * np.cos(TTg), RRg * np.sin(TTg)], -1)
            else: hh = (np.stack([np.full_like(RRg, p), RRg * np.cos(TTg), RRg * np.sin(TTg)], -1) - M.shift) / M.scale
            return np.maximum(hh @ K[:, k] + b[k], 0)
        f = np.stack([l1map(p) for p in PTG]); tot = ((f - f.mean()) ** 2).sum(); az = ((f - f.mean(-1, keepdims=True)) ** 2).sum()
        l1rad[int(k)] = float(1 - az / tot) if tot > 1e-12 else None
    rs = np.array([rad[int(k)]['radial_share'] for k in order if rad[int(k)]['radial_share'] is not None])
    wcost = np.array([costL[k] for k in order if rad[int(k)]['radial_share'] is not None])
    R['step2']['radial'] = dict(per_unit=rad, median=float(np.median(rs)) if len(rs) else None, frac_ge_0_9=float((rs >= 0.9).mean()) if len(rs) else None,
                                cost_weighted_mean=float((rs * wcost).sum() / wcost.sum()) if len(rs) else None,
                                top10=[rad[int(k)]['radial_share'] for k in order[:10]], control_first_layer=l1rad,
                                control_first_layer_median=float(np.median([v for v in l1rad.values() if v is not None])) if l1rad else None,
                                jedi_reference='X6 n64 step 5: radial share of the top-5 S units (u maps at one slot)', jedi_x6=_x6_radial())
    log(f'step 2 radial: Φ radial share median {R["step2"]["radial"]["median"]}, >= 0.9 {R["step2"]["radial"]["frac_ge_0_9"]}, '
        f'cost-weighted {R["step2"]["radial"]["cost_weighted_mean"]}; top10 {np.round(R["step2"]["radial"]["top10"], 3).tolist()}; '
        f'first-layer control median {R["step2"]["radial"]["control_first_layer_median"]}'); save()
    # MARS naming
    HT, HF = handbuilt(xT), handbuilt(xF)
    QTd = {**{k: np.asarray(v, np.float64) for k, v in QT.items()}, **HT}; QFd = {**{k: np.asarray(v, np.float64) for k, v in QF.items()}, **HF}
    keys = [k for k in QFd if np.isfinite(QFd[k]).all() and np.isfinite(QTd[k]).all() and np.ptp(QFd[k][fit]) > 0]
    Qs = dict(fit={k: QFd[k][fit] for k in keys}, dev={k: QFd[k][dev] for k in keys}, test={k: QTd[k] for k in keys})
    del QFd, HF, QF; knots = {k: mars.thresholds(Qs['fit'][k], False, False) for k in keys}
    mars.MAX_TERMS = a.max_terms; sel = np.random.default_rng(1).choice(len(fit), min(6000, len(fit)), replace=False)
    LF = M.run(xF, keep=('L', 'h'))
    units = []; nameu = order if a.full else order[:a.max_units]
    for k in nameu:
        t1 = time.time(); zz = {'fit': LF['L'][fit, k], 'dev': LF['L'][dev, k]}
        terms, coef, path = mars.fit_neuron(zz, Qs, keys, knots, sel)
        Bt = np.stack([np.ones(J)] + [mars.term_value(t, Qs['test']) for t in terms], 1); pt_ = Bt @ coef
        r2t = r2(np.maximum(pt_, 0), LT[:, k])
        spread = [float(mars.term_value(t, Qs['fit']).std()) for t in terms]; w = np.array([abs(c) * s for c, s in zip(coef[1:], spread)])
        ws = w / max(w.sum(), 1e-12); T_ = [dict(t, coef=float(c), weight_share=float(s)) for t, c, s in zip(terms, coef[1:], ws)]
        top = T_[int(np.argmax(w))] if T_ else None
        # best single observable (linear) on test
        units.append(dict(unit=int(k), cost=float(costL[k]), intercept=float(coef[0]), terms=T_, n_terms=len(T_), dev_r2_path=path, test_r2=r2t,
                          top=dict(q=top['q'], kind=top['kind'], t=top.get('t'), q2=top.get('q2'), weight_share=top['weight_share'], family=FAMILIES[family(top['q'])]) if top else None,
                          seconds=round(time.time() - t1, 1)))
        log(f'   L{k} (cost {costL[k] * 100:.2f} %): {len(T_)} terms, test R² {r2t:.3f}, path@1/3 {path[:1]} {path[2:3]}; top '
            + (f"{top['q']} {top['kind']} {top.get('t')}" + (f" × {top['q2']}" if top.get('q2') else '') + f" ({top['weight_share']:.2f}, {FAMILIES[family(top['q'])]})" if top else '-') + f'; {time.time() - t1:.0f} s')
    R['step2']['naming'] = dict(max_terms=a.max_terms, n_keys=len(keys), n_handbuilt=sum(k.startswith('H:') for k in keys), units=units)
    # JEDI reference from E11 (a) S fits
    try:
        e11 = json.loads((OUT / 'e11_fit_a_S.json').read_text()); JU = e11['units']
        R['step2']['naming']['jedi_e11'] = [dict(unit=u['unit'], test_r2=u['test_r2'], dev_r2_path=u.get('dev_r2_path', [])[:15],
                                                 top=u['top3'][0] if u.get('top3') else None, family=FAMILIES[family(u['top3'][0]['q'])] if u.get('top3') else None) for u in JU]
    except Exception as e:  # noqa: BLE001
        JU = None; log('E11 S fits unavailable:', repr(e))

    def at(path, n): return path[min(n, len(path)) - 1] if path else None
    for lab, src in (('model', [dict(dev_r2_path=u['dev_r2_path'], test_r2=u['test_r2'], fam=u['top']['family'] if u['top'] else None, cost=u['cost']) for u in units]),
                     ('jedi', [dict(dev_r2_path=u.get('dev_r2_path', []), test_r2=u['test_r2'], fam=FAMILIES[family(u['top3'][0]['q'])] if u.get('top3') else None, cost=costS[u['unit']]) for u in (JU or [])])):
        if not src: continue
        R['step2']['naming'][f'{lab}_summary'] = dict(
            n=len(src), test_r2_median=float(np.median([s['test_r2'] for s in src])), test_r2_frac_ge_0_8=float(np.mean([s['test_r2'] >= 0.8 for s in src])),
            dev_r2_median_at={n: float(np.median([at(s['dev_r2_path'], n) for s in src if s['dev_r2_path']])) for n in (1, 3, 15)},
            top_family_counts={f: sum(1 for s in src if s['fam'] == f) for f in FAMILIES},
            top_family_cost_share={f: float(sum(s['cost'] for s in src if s['fam'] == f) / max(sum(s['cost'] for s in src), 1e-12)) for f in FAMILIES})
    log(f'step 2 naming: model {R["step2"]["naming"].get("model_summary")}; JEDI E11 {R["step2"]["naming"].get("jedi_summary")}'); save()
    # cross-readability (ridge on all units, fit on Fs, test on T)
    SF = net.run(xF, keep=('S', 'z')); liveS = np.flatnonzero(costS >= 0.001)
    cr = dict(jedi_S_from_model_L={}, model_L_from_jedi_S={})
    for k in liveS:
        rg = Ridge(LF['L'], SF['S'][:, k], fit, dev); cr['jedi_S_from_model_L'][int(k)] = r2(rg(LT), RJ['S'][:, k])
    for k in order[:a.max_units]:
        rg = Ridge(SF['S'], LF['L'][:, k], fit, dev); cr['model_L_from_jedi_S'][int(k)] = r2(rg(RJ['S']), LT[:, k])
    cr['median_jedi_from_model'] = float(np.median(list(cr['jedi_S_from_model_L'].values())))
    cr['median_model_from_jedi'] = float(np.median(list(cr['model_L_from_jedi_S'].values())))
    R['step2']['cross_readability'] = cr
    log(f'step 2 cross-readability (ridge, held-out R²): JEDI S from model latent median {cr["median_jedi_from_model"]:.3f} '
        f'({ {k: round(v, 3) for k, v in cr["jedi_S_from_model_L"].items()} }); model latent from JEDI S median {cr["median_model_from_jedi"]:.3f}'); save()
    # ---------------- step 3: edits
    MT = moments(xT); MFm = moments(xF)
    rd = {('model', 'L'): Ridge(LF['L'], MFm['logm'], fit, dev), ('model', 'h'): Ridge(LF['h'], MFm['logm'], fit, dev),
          ('jedi', 'S'): Ridge(SF['S'], MFm['logm'], fit, dev), ('jedi', 'z'): Ridge(SF['z'], MFm['logm'], fit, dev)}
    R['step3'] = dict(readout_r2={f'{a_}:{b_}': r2(rd[(a_, b_)](dict(L=LT, h=hT, S=RJ['S'], z=RJ['z'])[b_]), MT['logm']) for a_, b_ in rd})
    log(f'step 3 read-out log m held-out R² on T: {R["step3"]["readout_r2"]}')
    unedM = dict(L=LT, h=hT); unedJ = dict(S=RJ['S'], z=RJ['z'])
    isW = yT == W_

    def run_both(xe):
        rm = M.run(xe, keep=('L', 'h', 'logits')); rj = net.run(xe, keep=('S', 'z', 'logits'))
        return rm, rj

    def mhat(rm, rj):
        return {('model', 'L'): rd[('model', 'L')](rm['L']) - rd[('model', 'L')](LT), ('model', 'h'): rd[('model', 'h')](rm['h']) - rd[('model', 'h')](hT),
                ('jedi', 'S'): rd[('jedi', 'S')](rj['S']) - rd[('jedi', 'S')](RJ['S']), ('jedi', 'z'): rd[('jedi', 'z')](rj['z']) - rd[('jedi', 'z')](RJ['z'])}

    def edit_block(lab, xe, pred, ctrl=None, cal=True):
        rm, rj = run_both(xe); cm, cj = rm['logits'].argmax(1), rj['logits'].argmax(1); dm = mhat(rm, rj)
        out = dict(pred_mean=float(pred.mean()), pred_median=float(np.median(pred)), true_dlogm_max=float(np.abs(moments(xe)['logm'] - MT['logm'])[MT['m'] > 1].max()))
        if cal:
            kcal = float(np.exp(pred.mean())); xc = scale_angles(xT, np.full(J, kcal)); rmc, rjc = run_both(xc); dmc = mhat(rmc, rjc)
            out['calibration_k'] = kcal; out['calibration_true_dlogm_mean'] = float((moments(xc)['logm'] - MT['logm'])[MT['m'] > 1].mean())
        for who, c, c0 in (('model', cm, cM), ('jedi', cj, cJ)):
            wz = c[isW] == Z_; o = dict(calls_changed=float((c != c0).mean()), calls_changed_ci=wilson(int((c != c0).sum()), J),
                                         W_to_Z=float(wz.mean()), W_to_Z_ci=wilson(int(wz.sum()), int(isW.sum())), W_to_Z_unedited=float((c0[isW] == Z_).mean()),
                                         acc=float((c == yT).mean()))
            if ctrl is not None:
                o['mcnemar_W_to_Z_vs_control'] = mcnemar(ctrl[who][isW] == Z_, wz)
            for st in (('L', 'h') if who == 'model' else ('S', 'z')):
                d = dm[(who, st)]; ci, bv = boot(lambda i: d[i].mean(), J)
                o[f'dlogm_hat_{st}'] = float(d.mean()); o[f'dlogm_hat_{st}_ci'] = ci; o[f'p_vs_0_{st}'] = boot_p(bv, 0.0)
                o[f'ratio_vs_pred_{st}'] = float(d.mean() / pred.mean()) if abs(pred.mean()) > 1e-9 else None
                if cal:
                    dc = dmc[(who, st)]; o[f'calib_dlogm_hat_{st}'] = float(dc.mean())
                    ci2, bv2 = boot(lambda i: d[i].mean() / dc[i].mean() if abs(dc[i].mean()) > 1e-9 else None, J)
                    o[f'calibrated_ratio_{st}'] = float(d.mean() / dc.mean()) if abs(dc.mean()) > 1e-9 else None; o[f'calibrated_ratio_{st}_ci'] = ci2
                    o[f'_boot_cal_{st}'] = bv2
            out[who] = o
        # paired model vs JEDI on the same jets
        wzm, wzj = (cm[isW] == Z_).astype(float), (cj[isW] == Z_).astype(float); nW = len(wzm)
        ci, bv = boot(lambda i: wzm[i].mean() - wzj[i].mean(), nW)
        out['paired'] = dict(W_to_Z_diff=float(wzm.mean() - wzj.mean()), W_to_Z_diff_ci=ci, W_to_Z_p=boot_p(bv, 0.0), W_to_Z_mcnemar=mcnemar(wzj > 0, wzm > 0))
        if cal:
            dL, dS = dm[('model', 'L')], dm[('jedi', 'S')]; dcL, dcS = dmc[('model', 'L')], dmc[('jedi', 'S')]
            dh, dz = dm[('model', 'h')], dm[('jedi', 'z')]; dch, dcz = dmc[('model', 'h')], dmc[('jedi', 'z')]
            f1 = lambda i: dL[i].mean() / dcL[i].mean() - dS[i].mean() / dcS[i].mean()
            f2 = lambda i: dh[i].mean() / dch[i].mean() - dz[i].mean() / dcz[i].mean()
            ci1, b1_ = boot(f1, J); ci2, b2_ = boot(f2, J)
            out['paired'].update(calratio_L_minus_S=float(f1(np.arange(J))), calratio_L_minus_S_ci=ci1, calratio_L_minus_S_p=boot_p(b1_, 0.0),
                                 calratio_h_minus_z=float(f2(np.arange(J))), calratio_h_minus_z_ci=ci2, calratio_h_minus_z_p=boot_p(b2_, 0.0))
        for who in ('model', 'jedi'):
            for k_ in [k_ for k_ in out[who] if k_.startswith('_boot')]: out[who].pop(k_)
        log(f'   {lab}: pred {out["pred_mean"]:.4f}; model calls {out["model"]["calls_changed"]:.3f} W→Z {out["model"]["W_to_Z"]:.3f} '
            f'(uned {out["model"]["W_to_Z_unedited"]:.3f}) Δm̂ L {out["model"]["dlogm_hat_L"]:.4f} h {out["model"]["dlogm_hat_h"]:.4f}'
            + (f' cal-ratio L {out["model"]["calibrated_ratio_L"]} h {out["model"]["calibrated_ratio_h"]}' if cal else '')
            + f' | JEDI calls {out["jedi"]["calls_changed"]:.3f} W→Z {out["jedi"]["W_to_Z"]:.3f} Δm̂ S {out["jedi"]["dlogm_hat_S"]:.4f} z {out["jedi"]["dlogm_hat_z"]:.4f}'
            + (f' cal-ratio S {out["jedi"]["calibrated_ratio_S"]} z {out["jedi"]["calibrated_ratio_z"]}' if cal else '') + f' | paired {out["paired"]}')
        return out, dict(model=cm, jedi=cj)

    E = {}
    xr = rotate_matched(xT, 0.06, np.random.default_rng(66)); predr = 0.5 * np.log(moments(xr)['M2'] / MT['M2'])
    E['rotation_matched'], crot = edit_block('matched rotation (control)', xr, predr, cal=False)
    th = np.random.default_rng(67).uniform(0, 2 * np.pi, J); xq = translate(xT, 0.06, th)
    E['translation_random_dir'], _ = edit_block('translation 0.06 random direction per jet', xq, 0.5 * np.log(moments(xq)['M2'] / MT['M2']), ctrl=crot)
    for lab, th_ in (('eta', 0.0), ('phi', np.pi / 2)):
        xe = translate(xT, 0.06, th_); pred = 0.5 * np.log(moments(xe)['M2'] / MT['M2'])
        E[f'translation_{lab}'], _ = edit_block(f'translation 0.06 +{lab}', xe, pred, ctrl=crot)
    from interp.e17_axis_jedi import axis_set, reaxis, wta_selftest
    E['wta_selftest'] = wta_selftest(); A, ainfo = axis_set(xT); xw = reaxis(xT, *A['wta']); Mw = moments(xw)
    E['wta_axis'], _ = edit_block('WTA axis', xw, 0.5 * np.log(Mw['M0'] * Mw['M2'] / (MT['M0'] * MT['M2'])), ctrl=crot)
    E['wta_axis']['axis_info'] = ainfo; E['wta_axis']['disp_median'] = float(np.median(np.hypot(*A['wta'])))
    R['step3']['edits'] = E; save()
    # rank-window swap
    try:
        x6 = json.loads((OUT / 'x6_3-feature_n64.json').read_text()); plan = x6['E4']['plan']
    except Exception as e:  # noqa: BLE001
        log('x6 plan unavailable, using edge 13 / inside 5 with JEDI units S0, S43, S52, S63:', repr(e)); plan = [dict(edge=13, inside=5, units=[0, 43, 52, 63])]
    sw = []
    sdL = LT.std(0); sdS = RJ['S'].std(0); topu = [int(k) for k in order[:10]]
    zf = xT[..., 0] / xT[..., 0].sum(1, keepdims=True)
    for p in plan[:2]:
        k, j = int(p['edge']), int(p['inside']); Ja, xa = swap(xT, k); Ji, xi = swap(xT, j); com = np.intersect1d(Ja, Ji)
        ia, ii = np.searchsorted(Ja, com), np.searchsorted(Ji, com)
        rec = dict(edge=k, inside=j, N=int(len(com)), jedi_units=p['units'], model_units=topu)
        for who in ('model', 'jedi'):
            if who == 'model':
                dA = M.run(xa, keep=('L',))['L'] - LT[Ja]; dI = M.run(xi, keep=('L',))['L'] - LT[Ji]; uu, sd = topu, sdL
            else:
                dA = net.run(xa, keep=('S',))['S'] - RJ['S'][Ja]; dI = net.run(xi, keep=('S',))['S'] - RJ['S'][Ji]; uu, sd = [u for u in p['units'] if sdS[u] > 0], sdS
            ma = (np.abs(dA[:, uu]) / sd[uu]).mean(1)[ia]; mi = (np.abs(dI[:, uu]) / sd[uu]).mean(1)[ii]
            ci, _ = boot(lambda i: ma[i].mean() / mi[i].mean() if mi[i].mean() > 0 else None, len(com))
            rec[who] = dict(across=float(ma.mean()), inside=float(mi.mean()), ratio=float(ma.mean() / mi.mean()) if mi.mean() > 0 else None, ratio_ci=ci,
                            signflip_p=signflip_p(ma - mi))
        ga = np.abs(zf[com, k] - zf[com, k + 1]); gi = np.abs(zf[com, j] - zf[com, j + 1])
        rec['z_gap_ratio'] = float(ga.mean() / gi.mean())
        sw.append(rec)
        log(f'   rank swap {k}|{k + 1} vs {j}|{j + 1} (N {len(com)}): model ratio {rec["model"]["ratio"]:.3f} {rec["model"]["ratio_ci"]}, '
            f'JEDI ratio {rec["jedi"]["ratio"]:.3f} {rec["jedi"]["ratio_ci"]}; z-gap ratio {rec["z_gap_ratio"]:.3f}')
    R['step3']['rank_swap'] = sw; save()
    # ---------------- step 4: IIA
    named = [u for u in units]; ku = [u['unit'] for u in named]
    if not ku: log('no named units: step 4 skipped'); save(); return
    fT = evaluate(named, Qs['test'])                         # (J, n_named) formula values
    Lhat = LT.copy(); Lhat[:, ku] = fT
    acc_H = float((M.tail(Lhat).argmax(1) == yT).mean()); agree_H = float((M.tail(Lhat).argmax(1) == cM).mean())
    rp = np.random.default_rng(0); ib = rp.integers(0, J, 5000); isr = rp.integers(0, J, 5000)
    bad = ib == isr
    while bad.any(): isr[bad] = rp.integers(0, J, bad.sum()); bad = ib == isr
    NP = len(ib); baseL = cM[ib]

    def iia_model():
        Lo, Ho = [], []
        for i, k in enumerate(ku):
            Lb = LT[ib].copy(); Lb[:, k] = LT[isr, k]; Lo.append(M.tail(Lb).argmax(1))
            Hb = Lhat[ib].copy(); Hb[:, k] = fT[isr, i]; Ho.append(M.tail(Hb).argmax(1))
        return np.array(Lo), np.array(Ho)
    Lo, Ho = iia_model(); Eff = Lo != baseL[None]
    # JEDI with E11 (a) S formulas
    JUl = [u for u in (JU or []) if not u.get('constant')]; kj = [u['unit'] for u in JUl]
    fJ = evaluate(JUl, {k: QTd[k] for k in set(t['q'] for u in JUl for t in u['terms']) | set(t['q2'] for u in JUl for t in u['terms'] if t.get('q2'))}) if JUl else None
    Shat = RJ['S'].copy()
    if JUl: Shat[:, kj] = fJ
    mS = np.ones(64)
    jt = lambda Sv, rows: net.run_jetwise(xT[rows], {'S': (mS, Sv)})['logits'].argmax(1)
    accHJ = float((jt(Shat, np.arange(J)) == yT).mean()) if JUl else None
    LoJ, HoJ = [], []
    for i, k in enumerate(kj):
        Sb = RJ['S'][ib].copy(); Sb[:, k] = RJ['S'][isr, k]; LoJ.append(jt(Sb, ib))
        Hb = Shat[ib].copy(); Hb[:, k] = fJ[isr, i]; HoJ.append(jt(Hb, ib))
    LoJ, HoJ = np.array(LoJ), np.array(HoJ); EffJ = LoJ != cJ[ib][None]

    def eff_iia(Lo_, Ho_, Ef_, pi=None):
        n = len(Lo_); pi = np.arange(n) if pi is None else pi
        v = [float((Lo_[pi[i]] == Ho_[i])[Ef_[pi[i]]].mean()) if Ef_[pi[i]].sum() >= 20 else np.nan for i in range(n)]
        return np.array(v)

    def derangements(n, B, seed):
        r = np.random.default_rng(seed); out = []
        while len(out) < B:
            p = r.permutation(n)
            if n < 2 or not (p == np.arange(n)).any(): out.append(p)
        return out
    res4 = {}
    for who, Lo_, Ho_, Ef_, units_, V, Vh, tailf, base_c in (
            ('model', Lo, Ho, Eff, ku, LT, Lhat, lambda Vb: M.tail(Vb).argmax(1), cM),
            ('jedi', LoJ, HoJ, EffJ, kj, RJ['S'], Shat, lambda Vb: jt(Vb, ib), cJ)):
        if len(units_) == 0: continue
        obs = eff_iia(Lo_, Ho_, Ef_); ok = np.isfinite(obs); mobs = float(np.nanmean(obs))
        perms = derangements(len(units_), 20 if a.smoke else 200, 11); nullm, nullu = [], []
        for pi in perms:
            v = eff_iia(Lo_, Ho_, Ef_, pi); nullm.append(np.nanmean(v)); nullu.append(v)
        nullm, nullu = np.array(nullm), np.array(nullu)
        p_stage = float((1 + (nullm >= mobs).sum()) / (1 + len(nullm)))
        pu = np.array([(1 + (nullu[:, i][np.isfinite(nullu[:, i])] >= obs[i]).sum()) / (1 + np.isfinite(nullu[:, i]).sum()) if ok[i] else 1.0 for i in range(len(units_))])
        # bootstrap over pairs of the stage mean
        def stat(i):
            v = [float((Lo_[u][i] == Ho_[u][i])[Ef_[u][i]].mean()) for u in range(len(units_)) if Ef_[u][i].sum() >= 20]
            return np.mean(v) if v else None
        ci, bvals = boot(stat, NP, B=30 if a.smoke else 300)
        # random rotations in the standardised live space
        rr = np.random.default_rng(12); sdv = V[:, units_].std(0); sdv[sdv < 1e-12] = 1; rot = []
        for _ in range(10 if a.smoke else 200):
            r_ = rr.normal(size=len(units_)); r_ /= np.linalg.norm(r_)
            def mix(VV):
                Vb = VV[ib].copy(); Zb, Zs = VV[ib][:, units_] / sdv, VV[isr][:, units_] / sdv
                Vb[:, units_] = np.maximum((Zb + np.outer((Zs - Zb) @ r_, r_)) * sdv, 0); return Vb
            lo_, ho_ = tailf(mix(V)), tailf(mix(Vh)); e_ = lo_ != base_c[ib]
            rot.append(float((lo_ == ho_)[e_].mean()) if e_.sum() >= 20 else np.nan)
        rot = np.array(rot)
        res4[who] = dict(units=[int(u) for u in units_], eff_iia=obs.tolist(), n_eff=Ef_.sum(1).tolist(), iia_all=(Lo_ == Ho_).mean(1).tolist(),
                         mean_eff_iia=mobs, mean_eff_iia_ci=ci, permuted_mean=float(np.nanmean(nullm)), permuted_q=np.nanquantile(nullm, [.025, .975]).tolist(),
                         p_perm_stage=p_stage, p_perm_unit=pu.tolist(), p_perm_unit_holm=holm(pu).tolist(), n_units_holm_005=int((holm(pu) < 0.05).sum()),
                         n_units_above_perm_q975=int(sum(obs[i] > np.nanquantile(nullu[:, i], .975) for i in range(len(units_)) if ok[i])),
                         rotation_mean=float(np.nanmean(rot)), rotation_q=np.nanquantile(rot, [.025, .975]).tolist(), p_rot=float((1 + (rot >= mobs).sum()) / (1 + np.isfinite(rot).sum())),
                         _boot=bvals)
        log(f'step 4 {who}: {len(units_)} units, mean effective IIA {mobs:.3f} {ci}; permuted {res4[who]["permuted_mean"]:.3f} {res4[who]["permuted_q"]} p {p_stage:.4f}; '
            f'rotation {res4[who]["rotation_mean"]:.3f} {res4[who]["rotation_q"]} p_rot {res4[who]["p_rot"]:.3f}; units Holm<0.05 {res4[who]["n_units_holm_005"]}')
    if 'jedi' in res4:
        bm, bj = res4['model'].pop('_boot'), res4['jedi'].pop('_boot'); n_ = min(len(bm), len(bj))
        # paired over the same pairs: rerun the bootstrap jointly
        rj_ = np.random.default_rng(99); diffs = []
        for _ in range(300):
            i = rj_.integers(0, NP, NP)
            vm = [float((Lo[u][i] == Ho[u][i])[Eff[u][i]].mean()) for u in range(len(ku)) if Eff[u][i].sum() >= 20]
            vj = [float((LoJ[u][i] == HoJ[u][i])[EffJ[u][i]].mean()) for u in range(len(kj)) if EffJ[u][i].sum() >= 20]
            if vm and vj: diffs.append(np.mean(vm) - np.mean(vj))
        diffs = np.array(diffs)
        res4['paired'] = dict(mean_eff_iia_diff=float(res4['model']['mean_eff_iia'] - res4['jedi']['mean_eff_iia']),
                              ci=[float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))], p=boot_p(diffs, 0.0), B=len(diffs))
        log(f'step 4 paired model − JEDI mean effective IIA {res4["paired"]}')
    else:
        res4['model'].pop('_boot', None)
    res4.update(pairs=NP, H_acc_model=acc_H, H_agree_model=agree_H, H_acc_jedi=accHJ,
                named_r2_ge_0_8=[u['unit'] for u in named if u['test_r2'] >= 0.8])
    R['step4'] = res4; save()
    # model-vs-JEDI tests (Holm within this model)
    tests = []
    for k_, e in R['step3']['edits'].items():
        if not isinstance(e, dict) or 'paired' not in e: continue
        for t_ in ('W_to_Z_p', 'calratio_L_minus_S_p', 'calratio_h_minus_z_p'):
            if e['paired'].get(t_) is not None: tests.append((f'{k_}:{t_}', e['paired'][t_]))
    if 'paired' in R['step4']: tests.append(('iia:mean_eff_diff', R['step4']['paired']['p']))
    adj = holm([p for _, p in tests]) if tests else []
    R['tests_vs_jedi'] = [dict(test=t, p=p, holm=float(h)) for (t, p), h in zip(tests, adj)]
    save(); log(f'wrote {JS}; X9 {name} DONE')


def summary():
    rows = []; out = dict(models={})
    for p in sorted(OUT.glob('x9_*.json')):
        if p.name in ('x9_summary.json',) or p.name.endswith('_smoke.json'): continue
        d = json.loads(p.read_text()); out['models'][d['model']] = d.get('step0'); rows += [dict(model=d['model'], **t) for t in d.get('tests_vs_jedi', [])]
    adj = holm([r['p'] for r in rows]) if rows else []
    for r, h in zip(rows, adj): r['holm_all'] = float(h)
    out['tests'] = rows; out['n_tests'] = len(rows)
    (OUT / 'x9_summary.json').write_text(json.dumps(clean(out), indent=1)); log(f'summary: {len(rows)} tests over {len(out["models"])} models')
    for r in rows: log(f'  {r["model"]} {r["test"]}: p {r["p"]:.4g}, Holm(model) {r["holm"]:.4g}, Holm(all) {r["holm_all"]:.4g}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None); ap.add_argument('--summary', action='store_true')
    ap.add_argument('--max_terms', type=int, default=15); ap.add_argument('--max_units', type=int, default=40)
    ap.add_argument('--full', action='store_true', help='name every live unit and use the full null sizes')
    ap.add_argument('--smoke', action='store_true', help='pipeline check: 3 units, 3 terms, small nulls; writes x9_<model>_smoke.json')
    a = ap.parse_args()
    if a.smoke: a.max_terms, a.max_units, a.full = 3, 3, False
    if a.summary: summary()
    else: main(a)
