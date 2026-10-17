"""R1 — rigour pass on the mass / jet-axis results (E6, E6v, E17, P1, P1v): one fixed read-out per model, calibration of that
read-out against real mass changes, hypothesis tests with bootstrap intervals, call changes against a null edit.

Models: JEDI-linear (3-feature, n64, exact fixed point; CPU) and ParT_full, ParT_kin (JetClass; GPU).
Sample: JEDI 20,000 test jets (4,000 per class, seed 51, interp.jedi_core.sample); ParT 10,000 test jets (rows 8,000-9,000 of each
class's first test file, interp.part_core.sample, rebuilt baseline as P1). Earlier samples: E6 seed 6, E6v 17, E17 41; P1 / P1v rows
3,000-4,000 and 6,000-7,000.
Read-out: one ridge per model from the last hidden layer (JEDI: z, the 16 pre-activation neurons whose ReLU feeds the last layer; ParT:
the 128-d class token after the last LayerNorm) to the true log m, fitted on a random half of the unedited jets (split seed = sample
seed; alpha on the last quarter of that half, grid 1e-6 .. 1e4), applied to the other half (the test half). Every Δlog m̂ is a mean over
the test half.
Edits (true invariant mass unchanged):
  shift_{eta,phi}_{δ}  every real particle's (Δη, Δφ) + δ along η or φ, δ ∈ {0.02, 0.03, 0.045, 0.06} rad (= the axis moved by −δ).
                       JEDI: inputs (pT, Δη, Δφ) shifted, padding fixed (as E6). ParT: deta or dphi + δ on real slots, deltaR recomputed
                       by the input code, 4-vectors unchanged (pair features, e_log, logerel unchanged; as P1v check 2 (f) / check 4).
  escheme   (Δη, Δφ) about the E-scheme axis of the stored particles (JEDI: summed massless 4-vectors of the 64 stored particles;
            ParT: summed stored 4-vectors, as P1v)
  wta       about the WTA axis (C/A clustering in (Δη, Δφ), winner-take-all recombination; P1v's code)
  hardest   about the highest-pT particle
  null_rot90 (null edit) (Δη, Δφ) → (−Δφ, Δη) for every real particle: M0, M2, Σ pT r, Σ r², Σ pT² r² unchanged exactly; m unchanged
            to second order in the angles (measured). ParT: 4-vectors rebuilt (part_core.build).
Per-jet predictions about the new axis: inv (H0a, invariant mass): the true Δlog m; M0M2 (H0b): ½Δlog(M0·M2); G1: Δlog Σ pT r;
sqrtQ2: ½Δlog Σ pT² r² (E6v's alternative "√Q2"); sqrtR2: ½Δlog Σ r² (unweighted, real particles).
Calibration: angles × k (JEDI: Δη, Δφ × k; ParT: part_core.build with deta, dphi × k, 4-vectors rebuilt: a real mass change), k found by
secant on log k so that the mean TRUE Δlog m over the test half equals a target: each edit's mean prediction under M0M2, G1, sqrtQ2,
sqrtR2; plus k = 1.05 and 1.1. c = mean read-out change / mean true change.
Statistics: 2,000 bootstrap resamples of the test-half jets, one set per model shared by every edit and calibration edit (paired);
95 % percentile intervals. Per hypothesis h: expected read-out change E_h = c_h · mean pred_h (c_h from the calibration edit matched to
mean pred_h; E = 0 for H0a); test of mean Δlog m̂ = E_h: two-sided bootstrap p = (1 + #{|θ* − θ̂| ≥ |θ̂|}) / (B + 1) on θ = obs − E_h
(floor 1/2001), and z = θ̂ / s.e.(θ*) with its normal p (used for Holm when the bootstrap p is at its floor). Calibrated ratio =
obs / E_M0M2 = (obs / pred) / c. Per-jet least-squares regression of Δlog m̂ on each hypothesis' per-jet prediction (slope, R²).
Calls (all jets): accuracy, calls changed, W → Z with Wilson 95 % intervals; exact McNemar tests (paired by jet) edit vs null edit
(W → Z, and calls changed) and edit vs the matched calibration edit (W → Z).
  ./job.sh r1-jedi 'python -m interp.r1_mass_stats --model jedi'         -> AI_RESULTS/stats/r1_mass_axis_jedi.json
  ./part_job.sh r1-part 'python -m interp.r1_mass_stats --model part'    -> AI_RESULTS/stats/r1_mass_axis_part.json (ParT_full, ParT_kin)
  ./job.sh r1-plot 'python -m interp.r1_mass_stats --plot'               -> AI_RESULTS/stats/r1_mass_axis.json (merged; Holm over every
                                                                            test), AI_RESULTS/site/T13_mass_axis_stats.png,
                                                                            AI_RESULTS/site/F7_mass_axis_calibrated.png
  --per N (jets per class) and --tag T for a smoke run (files get the suffix _T).
"""
import argparse, json, math, os, sys, textwrap, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'stats'
SITE = RES / 'site'
B = 2000
ALPHAS = 10.0 ** np.arange(-6, 4.1, 0.5)
SHIFTS = (0.02, 0.03, 0.045, 0.06)
HYPS = ('inv', 'M0M2', 'G1', 'sqrtQ2', 'sqrtR2')
CALH = ('M0M2', 'G1', 'sqrtQ2', 'sqrtR2')
KFIX = (1.05, 1.1)
log = lambda *a: print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ shared pieces
class Ridge:                    # as interp/e17_axis_jedi.py and interp/p1_translation_part.py
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


def wta_axis_one(pt, y, p):     # verbatim from interp/p1v_verify_part.py
    """C/A clustering (smallest ΔR² in (y, p) merged first, no beam distance) with winner-take-all recombination: the merged
    branch has pT_i + pT_j and the (y, p) of the harder branch; returns the (y, p) of the final branch"""
    n = len(pt)
    if n == 1: return float(y[0]), float(p[0])
    pt = pt.astype(np.float64).copy(); y = y.astype(np.float64).copy(); p = p.astype(np.float64).copy(); alive = np.ones(n, bool)
    D = (y[:, None] - y[None, :]) ** 2 + (p[:, None] - p[None, :]) ** 2; np.fill_diagonal(D, np.inf); i = 0
    for _ in range(n - 1):
        i, j = divmod(int(np.argmin(D)), n)
        if pt[j] > pt[i]: y[i], p[i] = y[j], p[j]
        pt[i] += pt[j]; alive[j] = False; D[j, :] = np.inf; D[:, j] = np.inf
        d = (y - y[i]) ** 2 + (p - p[i]) ** 2; d[~alive] = np.inf; d[i] = np.inf; D[i, :] = d; D[:, i] = d
    return float(y[i]), float(p[i])


def wta_selftest():             # as interp/p1v_verify_part.py
    a = wta_axis_one(np.array([40., 30., 25., 5.]), np.array([0., 0.3, 0.31, 0.01]), np.zeros(4))
    b = wta_axis_one(np.array([50., 30., 25., 10.]), np.array([0., 0.3, 0.31, 0.01]), np.zeros(4))
    c = wta_axis_one(np.array([7.]), np.array([0.2]), np.array([-0.1]))
    return dict(ok=bool(a == (0.3, 0.0) and b == (0.0, 0.0) and c == (0.2, -0.1)), a=a, b=b, c=c)


def hvars(pt, de, dp):
    """per-jet quantities about the axis of the given relative coordinates (padded slots: pT = 0)"""
    pt, de, dp = (np.asarray(a, np.float64) for a in (pt, de, dp)); real = pt > 0
    r2_ = de ** 2 + dp ** 2; r = np.sqrt(r2_)
    return dict(M0=pt.sum(1), M2=(pt * r2_).sum(1), G1=(pt * r).sum(1), Q2=(pt ** 2 * r2_).sum(1), R2=(r2_ * real).sum(1),
                M1e=(pt * de).sum(1), M1p=(pt * dp).sum(1))


def preds(V0, V, dlogm):
    f = lambda a: np.maximum(a, 1e-300)
    return dict(inv=dlogm, M0M2=0.5 * np.log(f(V['M0'] * V['M2']) / f(V0['M0'] * V0['M2'])), G1=np.log(f(V['G1']) / f(V0['G1'])),
                sqrtQ2=0.5 * np.log(f(V['Q2']) / f(V0['Q2'])), sqrtR2=0.5 * np.log(f(V['R2']) / f(V0['R2'])))


def boot_weights(n, seed):
    """(B, n) float32 resampling weights (counts / n) of the test-half jets"""
    rng = np.random.default_rng(seed); W = np.empty((B, n), np.float32)
    for b in range(B): W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W / np.float32(n)


def bmean(Wb, a):
    a = np.asarray(a, np.float64); mu = a.mean(); return mu + Wb @ (a - mu).astype(np.float32)


def ci(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def log10_p_normal(z):
    x = abs(z) / math.sqrt(2.0)
    if x < 26: return math.log10(max(math.erfc(x), 1e-300))
    return (-x * x - math.log(x * math.sqrt(math.pi))) / math.log(10.0)


def ptest(th, thb, th0=0.0):
    """two-sided bootstrap test of θ = θ0 (bootstrap distribution shifted to the null) and its normal approximation"""
    thb = np.asarray(thb, np.float64); dev = abs(th - th0)
    pb = (1 + int(np.sum(np.abs(thb - th) >= dev))) / (len(thb) + 1)
    se = float(np.std(thb, ddof=1)); z = (th - th0) / se if se > 0 else (0.0 if dev == 0 else math.copysign(1e9, th - th0))
    lp = log10_p_normal(z)
    return dict(p_boot=pb, p_boot_at_floor=bool(pb <= 1.0 / (len(thb) + 1) + 1e-15), z=float(z), p_z=10 ** lp, log10_p_z=lp, se=se)


def wilson(k, n, zc=1.959963984540054):
    if n == 0: return dict(k=0, n=0, p=None, ci=[None, None])
    p = k / n; den = 1 + zc * zc / n; cen = (p + zc * zc / (2 * n)) / den; half = zc * math.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / den
    return dict(k=int(k), n=int(n), p=p, ci=[max(0.0, cen - half), min(1.0, cen + half)])


def mcnemar(a, b):
    """exact two-sided McNemar test on paired booleans a (edit), b (comparison)"""
    a, b = np.asarray(a, bool), np.asarray(b, bool); n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b)); n = n01 + n10
    if n == 0: return dict(a_only=0, b_only=0, p=1.0, log10_p=0.0)
    k = min(n01, n10)
    lt = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)])
    ls = lt.max() + math.log(np.exp(lt - lt.max()).sum()); l10 = min(0.0, (math.log(2) + ls) / math.log(10))
    return dict(a_only=n01, b_only=n10, p=10 ** l10, log10_p=l10)


def holm_log10(lps):
    """Holm adjustment on log10 p-values -> adjusted log10 p"""
    lps = np.asarray(lps, np.float64); m = len(lps); order = np.argsort(lps); adj = np.empty(m); run = -np.inf
    for i, j in enumerate(order):
        run = max(run, min(0.0, math.log10(m - i) + lps[j])); adj[j] = run
    return adj


def secant_k(target, f, tol):
    """k with f(k) (mean true Δlog m over the test half) = target; secant in log k (slope ≈ 1)"""
    l = [target, target + 0.02 * (abs(target) + 0.01)]; g = [f(math.exp(l[0])) - target, f(math.exp(l[1])) - target]
    while abs(g[-1]) > tol and len(l) < 40 and g[-1] != g[-2]:
        l.append(l[-1] - g[-1] * (l[-1] - l[-2]) / (g[-1] - g[-2])); g.append(f(math.exp(l[-1])) - target)
    i = int(np.argmin(np.abs(g))); return math.exp(l[i]), g[i] + target, len(l)


# ------------------------------------------------------------------------------------------------ JEDI adapter
class JediA:
    def __init__(self, per, seed):
        from interp.jedi_core import Jedi, sample
        t = time.time(); x, y, _ = sample(per_class=per, seed=seed)
        self.x = np.ascontiguousarray(x, np.float32); self.y = np.asarray(y).astype(int); self.N = len(self.y)
        self.classes = ['g', 'q', 'W', 'Z', 't']; self.iW, self.iZ = 2, 3
        self.net = Jedi(64)
        self.nets = {'jedi': self._run}
        q = self.net.L['l1'].iq.quantizer
        kq, iq, fq = (np.asarray(getattr(q, a), np.float64).reshape(64, 3) for a in ('k', 'i', 'f'))
        self.QLO = np.where(kq > 0, -2.0 ** iq, 0.0); self.QHI = 2.0 ** iq - 2.0 ** (-fq)
        self.info = dict(model='JEDI-linear 3-feature n64, exact fixed point', path=str(self.net.path), per_class=per, sample_seed=seed,
                         layer='z: the 16 pre-activation neurons of d6 (logits = d7(ReLU(z)))', load_s=time.time() - t,
                         true_mass='massless four-vectors of the 64 stored particles (pT, Δη, Δφ)')

    def _run(self, x):
        R = self.net.run(x, keep=('z', 'logits')); return R['z'], R['logits']

    def base(self): return self.x

    @staticmethod
    def feats(x): return x[..., 0], x[..., 1], x[..., 2]

    @staticmethod
    def logm(x):
        xx = np.asarray(x, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
        p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
        return np.log(np.maximum(np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0)), 1e-3))

    def wrap_frac(self, x):
        xn = (np.asarray(x, np.float64) - self.net.shift) / self.net.scale
        return float(((xn < self.QLO) | (xn > self.QHI)).any((1, 2)).mean())

    def shift(self, de, dp):
        x = self.x; e = x.copy(); rl = x[..., 0] > 0
        e[..., 1] = np.where(rl, x[..., 1] + np.float32(de), x[..., 1]); e[..., 2] = np.where(rl, x[..., 2] + np.float32(dp), x[..., 2])
        return e

    def rot90(self):
        x = self.x; e = x.copy(); rl = x[..., 0] > 0
        e[..., 1] = np.where(rl, -x[..., 2], x[..., 1]); e[..., 2] = np.where(rl, x[..., 1], x[..., 2]); return e

    def scale(self, k, rows=None):
        e = (self.x if rows is None else self.x[rows]).copy(); e[..., 1:] *= np.float32(k); return e

    def axes(self):
        x = self.x; pt = x[..., 0].astype(np.float64); de = x[..., 1].astype(np.float64); dp = x[..., 2].astype(np.float64); real = pt > 0
        k = pt.argmax(1); r = np.arange(len(x))
        t = time.time(); wta = np.array([wta_axis_one(pt[j, real[j]], de[j, real[j]], dp[j, real[j]]) for j in range(len(x))]); tw = time.time() - t
        Px, Py, Pz = (pt * np.cos(dp)).sum(1), (pt * np.sin(dp)).sum(1), (pt * np.sinh(de)).sum(1)
        A = dict(escheme=(np.arcsinh(Pz / np.hypot(Px, Py)), np.arctan2(Py, Px)), wta=(wta[:, 0], wta[:, 1]), hardest=(de[r, k], dp[r, k]))
        disp = {a: np.hypot(*A[a]) for a in A}
        info = dict(wta_seconds=tw, wta_selftest=wta_selftest(),
                    wta_is_hardest=float(np.mean((np.abs(wta[:, 0] - de[r, k]) < 1e-9) & (np.abs(wta[:, 1] - dp[r, k]) < 1e-9))),
                    disp_median={a: float(np.median(v)) for a, v in disp.items()}, disp_p90={a: float(np.quantile(v, 0.9)) for a, v in disp.items()})
        return A, info

    def reaxis(self, ae, ap):   # as interp/e17_axis_jedi.py
        x = self.x; real = x[..., 0] > 0; e = x.copy()
        de = x[..., 1].astype(np.float64) - ae[:, None]; dp = (x[..., 2].astype(np.float64) - ap[:, None] + np.pi) % (2 * np.pi) - np.pi
        e[..., 1] = np.where(real, de, x[..., 1]).astype(np.float32); e[..., 2] = np.where(real, dp, x[..., 2]).astype(np.float32)
        return e


# ------------------------------------------------------------------------------------------------ ParT adapter
class PartA:
    def __init__(self, per, start):
        from interp import part_core as pc
        self.pc = pc; t = time.time()
        self.J0 = pc.sample(per, start); self.y = self.J0['label'].astype(int); self.N = len(self.y)
        pt, de, dp, ms = pc.kin(self.J0); self.k0 = (pt, de, dp, ms)
        self.Jb = pc.build(self.J0, pt, de, dp, ms)
        self.classes = list(pc.CL); self.iW, self.iZ = self.classes.index('Wqq'), self.classes.index('Zqq')
        self.net = {n: pc.ParTNetwork(n) for n in ('full', 'kin')}
        self.nets = {f'part_{n}': (lambda J, n=n: self._run(n, J)) for n in ('full', 'kin')}
        self.info = dict(model='ParT_full and ParT_kin (JetClass, weaver-core 0.4.17)', per_class=per, rows=[start, start + per],
                         layer='class token after the last LayerNorm (128; logits = h W + b)', load_s=time.time() - t,
                         device=self.net['full'].device, true_mass='part_core.jmass of the (rebuilt) four-vectors')

    def _run(self, n, J):
        R = self.net[n].run(J); return R['z'], R['logits']

    def base(self): return self.Jb

    def feats(self, J):
        pt, de, dp, _ = self.pc.kin(J); return pt, de, dp

    def logm(self, J): return np.log(np.maximum(self.pc.jmass(J), 1.0))

    def wrap_frac(self, J): return None

    def shift(self, de, dp):
        J = self.Jb; m = J['mask'] > 0; E = dict(J)
        E['deta'] = np.where(m, J['deta'] + np.float32(de), 0).astype(np.float32)
        E['dphi'] = np.where(m, J['dphi'] + np.float32(dp), 0).astype(np.float32)
        return E

    def rot90(self):
        pt, de, dp, ms = self.k0; return self.pc.build(self.J0, pt, -dp, de, ms)

    def scale(self, k, rows=None):
        pt, de, dp, ms = self.k0; J0 = self.J0
        if rows is not None: J0 = self.pc.sub(J0, rows); pt, de, dp, ms = pt[rows], de[rows], dp[rows], ms[rows]
        return self.pc.build(J0, pt, de * np.float32(k), dp * np.float32(k), ms)

    def axes(self):             # as interp/p1v_verify_part.py axis_set
        J = self.Jb; m = J['mask'] > 0; sgn = np.where(J['jet_eta'] < 0, -1.0, 1.0); je, jp = J['jet_eta'].astype(np.float64), J['jet_phi'].astype(np.float64)
        Px, Py, Pz = ((J[k].astype(np.float64) * m).sum(1) for k in ('px', 'py', 'pz'))
        pt = np.hypot(J['px'], J['py']) * m; k = pt.argmax(1); r = np.arange(len(pt))
        t = time.time(); wd = np.zeros(len(pt)); wp = np.zeros(len(pt))
        for j in range(len(pt)):
            q = m[j]; wd[j], wp[j] = wta_axis_one(pt[j, q], J['deta'][j, q], J['dphi'][j, q])
        tw = time.time() - t
        hd, hp = J['deta'][r, k].astype(np.float64), J['dphi'][r, k].astype(np.float64)
        A = dict(escheme=(np.arcsinh(Pz / np.hypot(Px, Py)), np.arctan2(Py, Px)), wta=(je + sgn * wd, jp + wp), hardest=(je + sgn * hd, jp + hp))
        disp = {a: np.hypot(ea - je, (pa - jp + np.pi) % (2 * np.pi) - np.pi) for a, (ea, pa) in A.items()}
        info = dict(wta_seconds=tw, wta_selftest=wta_selftest(),
                    wta_is_hardest=float(np.mean((np.abs(wd - hd) < 1e-7) & (np.abs(wp - hp) < 1e-7))),
                    disp_median={a: float(np.median(v)) for a, v in disp.items()}, disp_p90={a: float(np.quantile(v, 0.9)) for a, v in disp.items()})
        return A, info

    def reaxis(self, eta_a, phi_a):     # as interp/p1v_verify_part.py reaxis: 4-vectors unchanged, deta / dphi about the new axis
        J = self.Jb; m = J['mask'] > 0; sgn = np.where(J['jet_eta'] < 0, -1.0, 1.0)
        eta_i = J['jet_eta'][:, None].astype(np.float64) + sgn[:, None] * J['deta']; phi_i = J['jet_phi'][:, None].astype(np.float64) + J['dphi']
        sa = np.where(eta_a < 0, -1.0, 1.0)
        de = sa[:, None] * (eta_i - eta_a[:, None]); dp = (phi_i - phi_a[:, None] + np.pi) % (2 * np.pi) - np.pi
        E = dict(J); E['deta'] = np.where(m, de, 0).astype(np.float32); E['dphi'] = np.where(m, dp, 0).astype(np.float32)
        E['jet_eta'] = np.asarray(eta_a, np.float32); E['jet_phi'] = np.asarray(phi_a, np.float32)
        return E


# ------------------------------------------------------------------------------------------------ measurement
def edit_list():
    E = [(f'shift_{a}_{d}', dict(kind='shift', dir=a, delta=d)) for a in ('eta', 'phi') for d in SHIFTS]
    E += [(a, dict(kind='axis', axis=a)) for a in ('escheme', 'wta', 'hardest')]
    return E + [('null_rot90', dict(kind='null'))]


def run_model(A, seed, tol):
    t0 = time.time(); N, y = A.N, A.y
    X0 = A.base(); lm0 = A.logm(X0); V0 = hvars(*A.feats(X0))
    rng = np.random.default_rng(seed); perm = rng.permutation(N)
    tr, te = np.sort(perm[:N // 2]), np.sort(perm[N // 2:]); trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    res = dict(info=A.info, classes=A.classes, N=N, N_fit=len(tr), N_test=len(te), split_seed=seed,
               class_counts=np.bincount(y, minlength=len(A.classes)).tolist(),
               bootstrap=dict(B=B, seed=seed + 1000, unit='test-half jets, one set of resamples per model shared by all edits'),
               wrap_frac_unedited=A.wrap_frac(X0))
    log(f'{N} jets; fit half {len(tr)}, test half {len(te)} ({time.time() - t0:.0f} s)')
    axes, ainfo = A.axes(); res['axes_info'] = ainfo; log(f'axes: {ainfo}')
    EDITS = edit_list()

    def make(meta):
        if meta['kind'] == 'shift':
            d = meta['delta']; return A.shift(d if meta['dir'] == 'eta' else 0.0, d if meta['dir'] == 'phi' else 0.0)
        if meta['kind'] == 'axis': return A.reaxis(*axes[meta['axis']])
        return A.rot90()

    PRED, EINFO = {}, {}
    for name, meta in EDITS:
        Xe = make(meta); V = hvars(*A.feats(Xe)); dl = A.logm(Xe) - lm0; P = preds(V0, V, dl); PRED[name] = P
        EINFO[name] = dict(meta, true_dlogm_mean=float(dl.mean()), true_dlogm_absmax=float(np.abs(dl).max()),
                           true_dlogm_mean_test=float(dl[te].mean()),
                           M0_absmax_dlog=float(np.abs(np.log(V['M0'] / V0['M0'])).max()),
                           M1sq_over_M0M2_median=float(np.median((V['M1e'] ** 2 + V['M1p'] ** 2) / np.maximum(V['M0'] * V['M2'], 1e-300))),
                           pred_mean_test={h: float(P[h][te].mean()) for h in HYPS}, pred_median_test={h: float(np.median(P[h][te])) for h in HYPS},
                           nonfinite={h: int((~np.isfinite(P[h])).sum()) for h in HYPS}, wrap_frac=A.wrap_frac(Xe))
        log(f'edit {name}: ' + ', '.join(f'{k} {v}' for k, v in EINFO[name].items()))
        del Xe
    res['edit_info'] = EINFO
    # ---- calibration edits (network independent): k for each target
    CAL = {}
    f = lambda k: float(np.mean(A.logm(A.scale(k, te)) - lm0[te]))
    for name, meta in EDITS:
        if meta['kind'] == 'null': continue
        for h in CALH:
            tg = float(PRED[name][h][te].mean()); key = f't{tg:+.9f}'
            CAL.setdefault(key, dict(target=tg, k=None, used_by=[]))['used_by'].append([name, h])
    for k in KFIX: CAL[f'k{k}'] = dict(target=None, k=k, used_by=[['fixed', k]])
    for key, c in CAL.items():
        if c['k'] is None:
            c['k'], c['achieved'], c['iterations'] = secant_k(c['target'], f, tol)
        else:
            c['achieved'] = f(c['k']); c['iterations'] = 0
    log(f'calibration: {len(CAL)} angle scalings; max |achieved − target| '
        f'{max(abs(c["achieved"] - c["target"]) for c in CAL.values() if c["target"] is not None):.2e} ({time.time() - t0:.0f} s)')
    # ---- networks: read-out, then every edit and calibration edit
    RO = {}
    for n, run in A.nets.items():
        z0, L0 = run(X0); rid = Ridge(z0, lm0, trf, trv, tr); b0 = rid(z0[te]); c0 = L0.argmax(1)
        RO[n] = dict(ridge=rid, base=b0, c0=c0)
        res.setdefault('models', {})[n] = dict(readout=dict(r2_test=r2(b0, lm0[te]), r2_fit=r2(rid(z0[tr]), lm0[tr]), alpha=float(rid.alpha),
                                                            n_features=int(z0.shape[1]), alpha_grid=[float(ALPHAS[0]), float(ALPHAS[-1])]),
                                               baseline=dict(acc=float((c0 == y).mean()), W_to_Z=wilson(int((c0[y == A.iW] == A.iZ).sum()), int((y == A.iW).sum())),
                                                             Z_to_W=wilson(int((c0[y == A.iZ] == A.iW).sum()), int((y == A.iZ).sum()))))
        log(f'{n}: read-out {res["models"][n]["readout"]}; baseline acc {res["models"][n]["baseline"]["acc"]:.4f}')
    OUTS = {n: {} for n in A.nets}; TC = {}
    jobs = [(name, (lambda meta=meta: make(meta))) for name, meta in EDITS] + [(f'cal:{key}', (lambda k=c['k']: A.scale(k))) for key, c in CAL.items()]
    for i, (name, mk) in enumerate(jobs):
        Xe = mk()
        if name.startswith('cal:'): TC[name[4:]] = (A.logm(Xe) - lm0)[te]
        for n, run in A.nets.items():
            z, L = run(Xe); OUTS[n][name] = dict(d=RO[n]['ridge'](z[te]) - RO[n]['base'], c=L.argmax(1))
        del Xe
        if i % 10 == 0: log(f'   ran {i + 1} / {len(jobs)} ({time.time() - t0:.0f} s)')
    Wb = boot_weights(len(te), seed + 1000)
    for n in A.nets:
        res['models'][n].update(net_stats(OUTS[n], RO[n]['c0'], PRED, EINFO, CAL, TC, EDITS, te, y, A.iW, A.iZ, Wb))
        log_model(n, res['models'][n])
    res['calibration_edits'] = {key: dict(target=c['target'], k=c['k'], achieved=c['achieved'], iterations=c['iterations'], used_by=c['used_by'])
                                for key, c in CAL.items()}
    res['elapsed_s'] = time.time() - t0
    return res


def net_stats(O, c0, PRED, EINFO, CAL, TC, EDITS, te, y, iW, iZ, Wb):
    Wm = y == iW; Zm = y == iZ; N = len(y)
    out = dict(calibration={}, edits={}, tests=[])
    CB = {}
    for key, c in CAL.items():
        d, t = O[f'cal:{key}']['d'], TC[key]; db, tb = bmean(Wb, d), bmean(Wb, t); cc = float(d.mean() / t.mean()); cb = db / tb
        CB[key] = (cc, cb); cl = O[f'cal:{key}']['c']
        out['calibration'][key] = dict(k=c['k'], target=c['target'], true_mean=float(t.mean()), true_ci=ci(tb), readout_mean=float(d.mean()),
                                       readout_ci=ci(db), c=cc, c_ci=ci(cb), used_by=c['used_by'],
                                       W_to_Z=wilson(int((cl[Wm] == iZ).sum()), int(Wm.sum())), changed=wilson(int((cl != c0).sum()), N),
                                       acc=float((cl == y).mean()))
    cn = O['null_rot90']['c']; dn = O['null_rot90']['d']; dnb = bmean(Wb, dn)
    for name, meta in EDITS:
        d = O[name]['d']; ce = O[name]['c']; db = bmean(Wb, d); obs = float(d.mean()); P = PRED[name]
        e = dict(kind=meta['kind'], observed=dict(mean=obs, ci=ci(db), median=float(np.median(d)), se=float(np.std(db, ddof=1))))
        e['pred'] = {h: dict(mean=float(P[h][te].mean()), ci=ci(bmean(Wb, P[h][te]))) for h in HYPS}
        e['calls'] = dict(acc=float((ce == y).mean()), changed=wilson(int((ce != c0).sum()), N),
                          W_to_Z=wilson(int((ce[Wm] == iZ).sum()), int(Wm.sum())), Z_to_W=wilson(int((ce[Zm] == iW).sum()), int(Zm.sum())),
                          changed_W=wilson(int((ce[Wm] != c0[Wm]).sum()), int(Wm.sum())))
        if meta['kind'] == 'null':
            e['test_zero'] = ptest(obs, db, 0.0); e['true_dlogm_mean'] = EINFO[name]['true_dlogm_mean']
            out['edits'][name] = e; continue
        e['calls']['mcnemar_vs_null_W_to_Z'] = mcnemar(ce[Wm] == iZ, cn[Wm] == iZ)
        e['calls']['mcnemar_vs_null_changed'] = mcnemar(ce != c0, cn != c0)
        e['vs_null'] = dict(diff=obs - float(dn.mean()), diff_ci=ci(db - dnb), **ptest(obs - float(dn.mean()), db - dnb, 0.0))
        e['hyp'] = {}
        for h in HYPS:
            ph = P[h][te]; Ph = float(ph.mean()); Pb = bmean(Wb, ph)
            if h == 'inv':
                Eh, Eb, cc, cb, ck = 0.0, np.zeros(B), None, None, None
            else:
                ck = f't{Ph:+.9f}'; cc, cb = CB[ck]; Eh = cc * Ph; Eb = cb * Pb
            tst = ptest(obs - Eh, db - Eb, 0.0)
            hh = dict(pred_mean=Ph, pred_ci=ci(Pb), expected_readout=Eh, expected_readout_ci=ci(Eb) if h != 'inv' else [0.0, 0.0],
                      observed=obs, diff=obs - Eh, diff_ci=ci(db - Eb), calib_key=ck, c=cc, c_ci=ci(cb) if cb is not None else None, **tst)
            if h != 'inv':
                hh['raw_ratio'] = obs / Ph if Ph != 0 else None; hh['raw_ratio_ci'] = ci(db / Pb)
                hh['calibrated_ratio'] = obs / Eh if Eh != 0 else None; hh['calibrated_ratio_ci'] = ci(db / Eb)
                hh['observed_calibrated'] = obs / cc; hh['observed_calibrated_ci'] = ci(db / cb)
                hh['perjet'] = perjet(Wb, ph, d, cc)
            e['hyp'][h] = hh
            out['tests'].append(dict(edit=name, hyp=h, log10_p_z=tst['log10_p_z'], p_boot=tst['p_boot']))
        cal = O[f'cal:t{e["hyp"]["M0M2"]["pred_mean"]:+.9f}']['c']
        e['calls']['matched_real'] = dict(W_to_Z=wilson(int((cal[Wm] == iZ).sum()), int(Wm.sum())), changed=wilson(int((cal != c0).sum()), N),
                                          mcnemar_W_to_Z=mcnemar(ce[Wm] == iZ, cal[Wm] == iZ))
        out['edits'][name] = e
    adj = holm_log10([t['log10_p_z'] for t in out['tests']])
    for t, a in zip(out['tests'], adj): t['holm_log10_p_z_within_model'] = float(a)
    out['null_calls'] = out['edits']['null_rot90']['calls']
    return out


def perjet(Wb, x, yv, c):
    x = np.asarray(x, np.float64); yv = np.asarray(yv, np.float64)
    xc, yc = x - x.mean(), yv - yv.mean(); sxx, sxy, syy = (xc * xc).mean(), (xc * yc).mean(), (yc * yc).mean()
    if sxx < 1e-18: return dict(slope=None, note='prediction constant across jets')
    mx, my = bmean(Wb, xc), bmean(Wb, yc)
    sxx_b = bmean(Wb, xc * xc) - mx * mx; sxy_b = bmean(Wb, xc * yc) - mx * my; syy_b = bmean(Wb, yc * yc) - my * my
    s = sxy / sxx; sb = sxy_b / sxx_b
    return dict(slope=float(s), slope_ci=ci(sb), intercept=float(yv.mean() - s * x.mean()), r2=float(sxy * sxy / (sxx * syy)),
                r2_ci=ci(sxy_b ** 2 / (sxx_b * syy_b)), slope_over_c=float(s / c) if c else None, pred_sd=float(np.sqrt(sxx)))


def log_model(n, M):
    log(f'== {n}: read-out R² {M["readout"]["r2_test"]:.4f} (alpha {M["readout"]["alpha"]:g}); baseline W→Z {M["baseline"]["W_to_Z"]["p"]:.4f}')
    for key, c in sorted(M['calibration'].items(), key=lambda kv: kv[1]['true_mean']):
        log(f'   cal k {c["k"]:.5f}: true {c["true_mean"]:+.5f} read-out {c["readout_mean"]:+.5f} c {c["c"]:.3f} [{c["c_ci"][0]:.3f}, {c["c_ci"][1]:.3f}] '
            f'W→Z {c["W_to_Z"]["p"]:.3f}')
    for name, e in M['edits'].items():
        s = f'   {name:14s} obs {e["observed"]["mean"]:+.5f} [{e["observed"]["ci"][0]:+.5f}, {e["observed"]["ci"][1]:+.5f}]'
        if 'hyp' in e:
            h = e['hyp']['M0M2']
            s += (f' | pred {h["pred_mean"]:+.5f} raw {h["raw_ratio"]:.3f} c {h["c"]:.3f} cal {h["calibrated_ratio"]:.3f} '
                  f'[{h["calibrated_ratio_ci"][0]:.3f}, {h["calibrated_ratio_ci"][1]:.3f}] | log10 p: ' +
                  ' '.join(f'{k} {v["log10_p_z"]:.1f}' for k, v in e['hyp'].items()) +
                  f' | W→Z {e["calls"]["W_to_Z"]["p"]:.3f} matched {e["calls"]["matched_real"]["W_to_Z"]["p"]:.3f} null {M["null_calls"]["W_to_Z"]["p"]:.3f}')
        else:
            s += f' | p(0) {e["test_zero"]["p_boot"]:.3g} | W→Z {e["calls"]["W_to_Z"]["p"]:.3f} changed {e["calls"]["changed"]["p"]:.4f}'
        log(s)


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else (o.tolist() if hasattr(o, 'tolist') else str(o))))
    log(f'wrote {path}')


# ------------------------------------------------------------------------------------------------ merge + figures
MODELS = [('jedi', 'JEDI-linear'), ('part_full', 'ParT_full'), ('part_kin', 'ParT_kin')]
HLAB = dict(inv='inv. m', M0M2='√(M0·M2)', G1='Σ pT r', sqrtQ2='√Σ pT²r²', sqrtR2='√Σ r²')


def elabel(name):
    if name.startswith('shift_'):
        _, a, d = name.split('_'); return f'shift {"η" if a == "eta" else "φ"} {float(d):g} rad'
    return dict(escheme='E-scheme axis', wta='WTA axis', hardest='hardest-particle axis', null_rot90='null: rotate 90°')[name]


def merge(tag):
    sfx = f'_{tag}' if tag else ''
    M = {}; src = {}
    for f in ('jedi', 'part'):
        p = OUT / f'r1_mass_axis_{f}{sfx}.json'
        if p.exists():
            R = json.loads(p.read_text()); src[f] = {k: v for k, v in R.items() if k != 'models'}
            for n, v in R['models'].items(): M[n] = dict(v, source=f)
    tests = [(n, i) for n in M for i in range(len(M[n]['tests']))]
    adj = holm_log10([M[n]['tests'][i]['log10_p_z'] for n, i in tests])
    for (n, i), a in zip(tests, adj):
        t = M[n]['tests'][i]; t['holm_log10_p_z_all'] = float(a); M[n]['edits'][t['edit']]['hyp'][t['hyp']]['holm_log10_p_z_all'] = float(a)
    for n in M:
        for name, e in M[n]['edits'].items():
            if 'hyp' in e:
                e['excluded_holm_p_lt_0.01'] = [h for h in HYPS if e['hyp'][h]['holm_log10_p_z_all'] < -2]
                e['excluded_raw_p_lt_0.01'] = [h for h in HYPS if e['hyp'][h]['log10_p_z'] < -2 and e['hyp'][h]['p_boot'] < 0.01]
    R = dict(note='R1 rigour pass; see interp/r1_mass_stats.py docstring. Holm over every hypothesis test of every model (normal p from the '
                  'bootstrap s.e.; the bootstrap p has the floor 1/(B+1)).', n_tests_holm=len(tests), sources=src, models=M)
    dump(R, OUT / f'r1_mass_axis{sfx}.json')
    return R


def fmt_p(h):
    if h['p_boot_at_floor']:
        return f'< 5e-4\n(z = {h["z"]:.0f})' if abs(h['z']) >= 9.5 else f'< 5e-4\n(z = {h["z"]:.1f})'
    return f'{h["p_boot"]:.2g}'


def fmt_lp(lp):
    if lp > -3: return f'{10 ** lp:.2g}'
    e = math.floor(lp); m = 10 ** (lp - e); return f'{m:.1f}e{e}'


def plot(tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    R = merge(tag); M = R['models']; sfx = f'_{tag}' if tag else ''
    SURF, INK, INK2, GRID, HEAD, ALT = '#fcfcfb', '#1f1f1e', '#5f5e59', '#e4e3df', '#eeebe3', '#f8f7f3'
    # ---------------- table T13
    cols = [('model', 1.65), ('edit', 1.6), ('predicted\n½Δlog(M0·M2)', 1.05), ('observed Δlog m̂\n[95 % CI]', 1.45),
            ('raw\nratio', 0.6), ('calibration c\nat matched Δ', 1.2), ('calibrated ratio\n[95 % CI]', 1.35),
            ('p, H0a\n(invariant m)', 0.95), ('p, H0b\n(√(M0·M2))', 0.95), ('excluded at\nHolm p < 0.01', 1.55),
            ('W → Z\n[95 % CI]', 1.3), ('W → Z, real Δm\nof the same size', 1.3), ('McNemar p,\nW → Z vs null', 1.05)]
    rows = []
    for key, lab in MODELS:
        if key not in M: continue
        m = M[key]; nul = m['edits']['null_rot90']; first = True
        for name, e in m['edits'].items():
            cell0 = (f'{lab}\nread-out R² {m["readout"]["r2_test"]:.3f}\nunedited W → Z {m["baseline"]["W_to_Z"]["p"]:.3f}') if first else ''
            first = False; o = e['observed']; ow = f'{o["mean"]:+.4f}\n[{o["ci"][0]:+.4f}, {o["ci"][1]:+.4f}]'
            wz = e['calls']['W_to_Z']; wzs = f'{wz["p"]:.3f}\n[{wz["ci"][0]:.3f}, {wz["ci"][1]:.3f}]'
            if 'hyp' not in e:
                rows.append([cell0, elabel(name), f'{e["pred"]["M0M2"]["mean"]:+.4f}', ow, '–', '–', '–',
                             f'{e["test_zero"]["p_boot"]:.2g}\n(Δ = 0)', '–', '–', wzs, '–', '(null)'])
                continue
            h = e['hyp']['M0M2']; ha = e['hyp']['inv']; mr = e['calls']['matched_real']['W_to_Z']
            exc = ', '.join(HLAB[x] for x in e['excluded_holm_p_lt_0.01']) or 'none'
            rows.append([cell0, elabel(name), f'{h["pred_mean"]:+.4f}', ow, f'{h["raw_ratio"]:.2f}',
                         f'{h["c"]:.3f}\n[{h["c_ci"][0]:.3f}, {h["c_ci"][1]:.3f}]',
                         f'{h["calibrated_ratio"]:.2f}\n[{h["calibrated_ratio_ci"][0]:.2f}, {h["calibrated_ratio_ci"][1]:.2f}]',
                         fmt_p(ha), fmt_p(h), textwrap.fill(exc, 18), wzs, f'{mr["p"]:.3f}\n[{mr["ci"][0]:.3f}, {mr["ci"][1]:.3f}]',
                         fmt_lp(e['calls']['mcnemar_vs_null_W_to_Z']['log10_p'])])
    LH = 0.165; PADY = 0.07; W = sum(w for _, w in cols)
    nl = lambda r: max(str(c).count('\n') + 1 for c in r)
    hh = nl([c for c, _ in cols]) * LH + PADY; rh = [max(2, nl(r)) * LH + PADY for r in rows]
    title_h = 0.95
    H = title_h + hh + sum(rh) + 1.6
    fig = plt.figure(figsize=(W + 0.3, H), facecolor=SURF); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, W + 0.3); ax.set_ylim(H, 0)
    ax.text(0.15, 0.3, 'Mass read-out under jet-axis edits (true invariant mass unchanged): calibrated ratios, hypothesis tests, W → Z calls', fontsize=12.5, color=INK, weight='bold', va='center')
    ax.text(0.15, 0.52, 'Observed: mean change of a ridge read-out of log m (last hidden layer → log m, fitted on half of the unedited jets) '
            'on the other half.\nCalibrated ratio = observed / (c × predicted): H0b predicts 1, H0a (invariant mass) predicts 0.',
            fontsize=9.5, color=INK2, va='top', linespacing=1.3)
    xs = np.cumsum([0.15] + [w for _, w in cols])
    y0 = title_h
    ax.add_patch(Rectangle((0.15, y0), W, hh, facecolor=HEAD, edgecolor='#d8d4ca', lw=0.6))
    for (c, w), x in zip(cols, xs):
        ax.text(x + 0.05, y0 + PADY / 2, c, fontsize=8.6, weight='bold', color=INK, va='top', linespacing=1.15)
    y = y0 + hh; blk = 0
    for i, (r, h_) in enumerate(zip(rows, rh)):
        if r[0]: blk += 1
        if r[0] and i: ax.plot([0.15, 0.15 + W], [y, y], color=INK2, lw=1.0)
        ax.add_patch(Rectangle((0.15 + cols[0][1], y), W - cols[0][1], h_, facecolor=ALT if i % 2 else SURF, edgecolor='#e4e1d8', lw=0.4))
        for j, (c, x) in enumerate(zip(r, xs)):
            ax.text(x + 0.05, y + PADY / 2, c, fontsize=8.4, color=INK if j != 1 else INK2, va='top', linespacing=1.15,
                    weight='bold' if j == 0 else 'normal')
        y += h_
    notes = [f'Samples: JEDI-linear 20,000 hls4ml test jets (seed 51), read-out on z (16 units); ParT 10,000 JetClass test jets (rows 8,000–9,000 per class), '
             f'read-out on the class token (128).  Fit half / test half random (split seed 51).  CI: 2,000 bootstrap resamples of the test-half jets.',
             f'c = read-out change / true Δlog m under angles × k, with k set so that the true mean Δlog m equals the edit\'s predicted ½Δlog(M0·M2).  '
             f'p: two-sided bootstrap (floor 1/2001, shown as "< 5e-4" with z from the bootstrap s.e.).  Holm over {R["n_tests_holm"]} tests '
             f'(5 hypotheses × 11 edits × {len([k for k, _ in MODELS if k in M])} models), on the normal p.',
             'Hypotheses: inv. m (Δ = 0), √(M0·M2), Σ pT r, √Σ pT²r², √Σ r² (unweighted), each with its own matched calibration.  '
             'W → Z: fraction of true W jets called Z (all W jets of the sample; Wilson CI).  Real Δm of the same size: angles × k at the matched k.',
             'Null edit: every particle rotated 90° about the axis (M0, M2 unchanged).  ParT shifts and axes: 4-vectors unchanged, (Δη, Δφ, ΔR) '
             'recomputed about the moved axis.  McNemar: exact, paired by jet.']
    note = '\n'.join(textwrap.fill(t, 190) for t in notes)
    ax.text(0.15, y + 0.15, note, fontsize=8.2, color=INK2, va='top', linespacing=1.35)
    fig.savefig(SITE / f'T13_mass_axis_stats{sfx}.png', dpi=130, facecolor=SURF); plt.close(fig); log('wrote', SITE / f'T13_mass_axis_stats{sfx}.png')
    # ---------------- figure F7
    PAL = dict(eta='#2a78d6', phi='#eb6834', axis='#1baf7a')
    keys = [k for k, _ in MODELS if k in M]
    fig, axs = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 5.6), facecolor=SURF, squeeze=False); axs = axs[0]
    for ax, key in zip(axs, keys):
        m = M[key]; lab = dict(MODELS)[key]
        ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9.5)
        E = {n: e for n, e in m['edits'].items() if 'hyp' in e}
        px = [e['hyp']['M0M2']['pred_mean'] for e in E.values()]; py = [e['observed']['mean'] for e in E.values()]
        xmax = 1.12 * max(max(px), max(py)); xmin = -0.04 * xmax
        cal = sorted(m['calibration'].values(), key=lambda c: c['true_mean'])
        cal = [c for c in cal if c['true_mean'] <= 1.08 * xmax]
        cx = np.array([c['true_mean'] for c in cal]); cy = np.array([c['readout_mean'] for c in cal])
        lo = np.array([c['readout_ci'][0] for c in cal]); hi = np.array([c['readout_ci'][1] for c in cal])
        ax.fill_between(cx, lo, hi, color=INK2, alpha=0.18, lw=0)
        ax.plot(cx, cy, color=INK, lw=2, label='calibration: angles × k\n(real mass change; x = true Δlog m)')
        ax.plot([xmin, xmax], [xmin, xmax], color=INK2, lw=1.2, ls='--', label='y = x')
        ax.axhline(0, color=INK2, lw=1.0, ls=':', label='invariant mass (H0a): 0')
        for grp, mk, lab_ in (('eta', 'o', 'shift along η (δ = .02–.06)'), ('phi', 's', 'shift along φ'), ('axis', '^', 'E-scheme / WTA / hardest axis')):
            sel = [(n, e) for n, e in E.items() if (n.startswith(f'shift_{grp}') if grp != 'axis' else e['kind'] == 'axis')]
            xx = [e['hyp']['M0M2']['pred_mean'] for _, e in sel]; yy = [e['observed']['mean'] for _, e in sel]
            yl = [e['observed']['mean'] - e['observed']['ci'][0] for _, e in sel]; yh = [e['observed']['ci'][1] - e['observed']['mean'] for _, e in sel]
            ax.errorbar(xx, yy, yerr=[yl, yh], fmt=mk, color=PAL[grp], ms=8, mfc=PAL[grp], mec=SURF, mew=1.5, elinewidth=1.4, capsize=0,
                        label=lab_, zorder=5)
            if grp == 'axis':
                for (n, e), x_, y_ in zip(sel, xx, yy):
                    if n == 'escheme': continue
                    ax.annotate({'wta': 'WTA', 'hardest': 'hardest'}[n], (x_, y_), xytext=(7, -10), textcoords='offset points', fontsize=9.5, color=INK)
        nl_ = m['edits']['null_rot90']['observed']
        ax.errorbar([0], [nl_['mean']], yerr=[[nl_['mean'] - nl_['ci'][0]], [nl_['ci'][1] - nl_['mean']]], fmt='x', color=INK2, ms=8, mew=1.8,
                    label='null: rotate 90° (pred. 0)', zorder=6)
        ax.set_xlim(xmin, xmax); ax.set_ylim(min(xmin, min(py) - 0.02 * xmax), xmax)
        ax.set_xlabel('predicted change ½Δlog(M0·M2) about the new axis\n(calibration line: true Δlog m)', color=INK2, fontsize=10.5)
        ax.set_ylabel('read-out change Δlog m̂ (mean, test half)', color=INK2, fontsize=10.5)
        ax.set_title(f'{lab}  (read-out R² {m["readout"]["r2_test"]:.3f})', color=INK, fontsize=12, loc='left')
    axs[0].legend(frameon=False, fontsize=8.8, loc='upper left', labelcolor=INK)
    cap = ('Points: mean read-out change under each axis edit (true invariant mass unchanged), 95 % bootstrap CI over test-half jets.  '
           'Black line and grey band: the same read-out under real mass changes (angles × k).  On the black line the read-out moves as for a real '
           'mass change of ½Δlog(M0·M2); on y = 0 it moves as the invariant mass would.')
    fig.text(0.01, 0.012, textwrap.fill(cap, int(60 * len(keys) + 20)), color=INK2, fontsize=8.8, va='bottom')
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(SITE / f'F7_mass_axis_calibrated{sfx}.png', dpi=140, facecolor=SURF); plt.close(fig)
    log('wrote', SITE / f'F7_mass_axis_calibrated{sfx}.png')


# ------------------------------------------------------------------------------------------------ main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=('jedi', 'part')); ap.add_argument('--plot', action='store_true')
    ap.add_argument('--per', type=int, default=None); ap.add_argument('--tag', default='')
    a = ap.parse_args(); sfx = f'_{a.tag}' if a.tag else ''
    if a.plot:
        SITE.mkdir(parents=True, exist_ok=True); plot(a.tag); sys.exit(0)
    t0 = time.time()
    if a.model == 'jedi':
        A = JediA(a.per or 4000, 51); R = run_model(A, 51, 1e-8)
    else:
        A = PartA(a.per or 1000, 8000); R = run_model(A, 51, 1e-6)
    dump(R, OUT / f'r1_mass_axis_{a.model}{sfx}.json'); log(f'R1 {a.model} DONE ({time.time() - t0:.0f} s)')
