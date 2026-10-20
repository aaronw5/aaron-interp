"""E24 (claims A3 / A1 / A4d; review weakness 2 "objects vs generic Deep Sets"; exploratory: N-subjettiness inside JEDI-linear):
does redistributing particles in azimuth about the stored axis, at fixed pT, r and pT rank, change JEDI-linear's S objects and
calls, and how does that compare with a Deep Sets, a PFN and a transformer on the same input?

Sample: interp.jedi_core.sample(per_class=2000, seed=2401), N = 10,000 test jets (padding never edited).
Edits (every real particle i of a jet, polar coordinates (r_i, θ_i) of (Δη_i, Δφ_i) about the stored axis):
  rot   rigid rotation θ_i → θ_i + φ_J, φ_J ~ U(0, 2π) per jet (control: m, M0, M2, |M1|, τ21, D2 unchanged).
  A     scramble: θ_i → independent U(0, 2π) (r_i, pT_i, rank unchanged; M0, M2 = Σ pT r² unchanged; M1 = Σ pT (Δη, Δφ) changes).
  At    A, then every real particle translated by c − c' (c, c' = M1/M0 before / after) → M1 held, M2 about the axis changes.
  A'    A, then q_i = c + s (p'_i − c') with s² = (M2 − M0|c|²) / (M2 − M0|c'|²) → M0, M1 (vector) and M2 held exactly
        (r_i changes by the translation / rescale; reported).
  B     half-plane-cell scramble (JEDI official only): the nine l1 half-plane cuts of the official model define 18 boundary
        directions (normal ± 90°); θ_i → U(sector of θ_i), so each particle stays in its cell of the nine-line arrangement
        (every half-plane statement keeps its on/off state up to the lines' ≤ 0.008 rad offsets; measured).
  B'    B followed by the A' moment hold.
  Br    control for B (added after the first run): the same 18-sector arrangement rotated by a random angle per jet (same
        sector sizes, so the same angular displacement distribution, but the sectors no longer align with the cuts).
  nulls 20 redraws of the A and A' angles (official; 5 for the other models).
Calibration with real jets (no edit): pool = sample(per_class=20000, seed=2402) minus the main jets. For each main jet, pool
  jets of the same class with |Δlog m|, |Δlog M0|, |Δlog M2|, |Δlog pT_jet| < 0.02; partner "shift" = the one whose τ21 is
  closest to τ21(jet) + Δτ21(A' of the jet) (|miss| < 0.02), partner "same" = closest to τ21(jet) (|miss| < 0.02, ≠ shift).
  Matched real change = P(call(shift) ≠ call(jet)) − P(call(same) ≠ call(jet)).
Read-outs: % calls changed (95 % jet bootstrap, 1,000), 5×5 call flow, W/Z/t → q/g; JEDI official: per live l1 cut mean
  |Δa|/sd over real particles and |ΔP|/sd; per live S unit E|ΔS|/sd and relative change E|ΔS| / E|S|; h4, h5, z; logits.
  τ21 (pT-weighted k-means axes, β = 1, R0 = 0.8, as jetdistill.observables) and D2 (β = 1, 24 hardest) before / after.
Stats: H0: P(call changed | edit) = P(call changed | rot); exact McNemar on the paired change indicators; Holm over models
  (A' vs rot) and over live S units (paired bootstrap p of E|ΔS_c|/sd, A' vs rot).
Models: --group main: JEDI-linear official n64 (jedi_core), dsets-d64 s0/s1 (float numpy), PFN 64-64-32 s0/s1 (X9 numpy);
        --group x12: the 8 X12 retrained JEDI-linear copies (hgq-r150_s0-2, hgq_s0-2, float_s0-1);
        --group tf (AI_PY=conda, torch): tf-2b-d32-n64 s0/s1.
  ./job.sh e24-run 'python -m interp.e24_azimuth_prong --group main'   → $AI_RESULTS/jedi/e24_main.json (+ e24_x12.json, e24_tf.json)
  python -m interp.e24_azimuth_prong --combine                         → e24_summary.json (Holm over models)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
T0 = time.time(); NB = 1000; R0 = 0.8
CL = ['g', 'q', 'W', 'Z', 't']
HP = [29, 6, 40, 57, 3, 43, 30, 36, 46]          # E14: the nine l1 half-plane statements of the official model
RADIAL_S = [29, 1, 0, 19, 30]                    # E4/E5/E11 radial (pT, r) hinge objects


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


# ------------------------------------------------------------------------------------------------ statistics
BIDX = None


def bidx(n):
    global BIDX
    if BIDX is None or BIDX.shape[1] != n: BIDX = np.random.default_rng(24).integers(0, n, (NB, n)).astype(np.int32)
    return BIDX


def ci(v):
    """mean of per-jet values with 95 % percentile bootstrap interval over jets"""
    v = np.asarray(v, np.float64); b = v[bidx(len(v))].mean(1)
    return dict(mean=float(v.mean()), ci=[float(np.quantile(b, .025)), float(np.quantile(b, .975))])


def ci_ratio(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); I = bidx(len(a))
    r = a[I].mean(1) / np.maximum(b[I].mean(1), 1e-300)
    return dict(value=float(a.mean() / max(b.mean(), 1e-300)), ci=[float(np.quantile(r, .025)), float(np.quantile(r, .975))])


def boot_p(d):
    """two-sided paired bootstrap p for H0: E[d] = 0 (percentile, floor 1/(NB+1))"""
    d = np.asarray(d, np.float64); b = d[bidx(len(d))].mean(1); b = b - b.mean() + 0.0
    obs = d.mean(); return float(min(1.0, ((np.abs(b) >= abs(obs)).sum() + 1) / (NB + 1)))


def mcnemar(a, b):
    """paired binary a (edit), b (control): exact two-sided binomial on discordant pairs"""
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n01 + n10
    if n == 0: return dict(n10=0, n01=0, p=1.0)
    try:
        from scipy.stats import binomtest
        p = float(binomtest(min(n01, n10), n, 0.5).pvalue)
    except Exception:
        from math import lgamma, log as ln, exp
        k = min(n01, n10); lp = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * ln(2) for i in range(k + 1)]
        mx = max(lp); p = min(1.0, 2 * exp(mx) * sum(exp(x - mx) for x in lp))
    return dict(n10=n10, n01=n01, p=p)


def holm(ps):
    ps = np.asarray(ps, np.float64); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------------------------------------ physics
def polar(x):
    x = np.asarray(x, np.float64); return x[..., 0], np.hypot(x[..., 1], x[..., 2]), np.arctan2(x[..., 2], x[..., 1]), x[..., 0] > 0


def from_polar(x, r, th, real):
    e = np.array(x, np.float64, copy=True)
    e[..., 1] = np.where(real, r * np.cos(th), x[..., 1]); e[..., 2] = np.where(real, r * np.sin(th), x[..., 2])
    return e


def moments(x):
    x = np.asarray(x, np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]
    M0 = pt.sum(1); M1 = np.stack([(pt * eta).sum(1), (pt * phi).sum(1)], -1); M2 = (pt * (eta ** 2 + phi ** 2)).sum(1)
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0)); ptj = np.hypot(p4[:, 0], p4[:, 1])
    return dict(M0=M0, M1=M1, M2=M2, m=m, ptj=ptj)


def hold_moments(x_new, x_ref, rescale=True):
    """translate the real particles of x_new so that M1 equals x_ref's; with rescale also scale about the centroid so that
    M2 (about the stored axis) equals x_ref's (M0 is unchanged by both)"""
    a, b = moments(x_ref), moments(x_new); real = np.asarray(x_new)[..., 0] > 0
    c = a['M1'] / a['M0'][:, None]; c2 = b['M1'] / b['M0'][:, None]
    e = np.array(x_new, np.float64, copy=True); p = e[..., 1:3]
    if rescale:
        num = a['M2'] - a['M0'] * (c ** 2).sum(1); den = b['M2'] - b['M0'] * (c2 ** 2).sum(1)
        s = np.sqrt(np.maximum(num, 0) / np.maximum(den, 1e-300))
        q = c[:, None, :] + s[:, None, None] * (p - c2[:, None, :])
    else:
        q = p + (c - c2)[:, None, :]
    e[..., 1:3] = np.where(real[..., None], q, p); return e


def kmeans(pt, eta, phi, k, iters=6):           # jetdistill.observables.compute._kmeans (copied verbatim)
    J = len(pt); ax = np.zeros((J, k, 2)); pts = np.stack([eta, phi], -1)
    ax[:, 0] = pts[np.arange(J), pt.argmax(1)]
    for j in range(1, k):
        d = np.min(np.linalg.norm(pts[:, :, None] - ax[:, None, :j], axis=-1), -1) * pt
        ax[:, j] = pts[np.arange(J), d.argmax(1)]
    for _ in range(iters):
        a = np.linalg.norm(pts[:, :, None] - ax[:, None], axis=-1).argmin(-1)
        for j in range(k):
            w = pt * (a == j); s = w.sum(1, keepdims=True)
            ax[:, j] = np.where(s > 0, (w[..., None] * pts).sum(1) / np.maximum(s, 1e-12), ax[:, j])
    return ax, np.linalg.norm(pts[:, :, None] - ax[:, None], axis=-1)


def substructure(x, with_d2=True, chunk=2000):
    """τ21 (β = 1, k-means axes, R0 0.8) and D2 (β = 1, the 24 hardest) as jetdistill.observables.compute"""
    t21, d2 = [], []
    for i in range(0, len(x), chunk):
        X = np.asarray(x[i:i + chunk], np.float64); pt, eta, phi = X[..., 0], X[..., 1], X[..., 2]; z = pt / pt.sum(1, keepdims=True)
        tau = {k: (z * kmeans(pt, eta, phi, k)[1].min(-1)).sum(1) / R0 for k in (1, 2)}
        t21.append(tau[2] / np.maximum(tau[1], 1e-12))
        if with_d2:
            top = np.argsort(-pt, 1, kind='stable')[:, :24]
            zt, et, pht = (np.take_along_axis(a, top, 1) for a in (z, eta, phi))
            R = np.sqrt((et[:, :, None] - et[:, None]) ** 2 + (pht[:, :, None] - pht[:, None]) ** 2)
            W = zt[:, :, None] * zt[:, None, :] * R                     # z_i z_j R_ij (diagonal 0)
            e2 = W.sum((1, 2)) / 2
            RzR = np.einsum('jik,jk,jkl->jil', R, zt, R)                # Σ_k R_ik z_k R_kl
            e3 = (W * RzR).sum((1, 2)) / 6
            d2.append(e3 / np.maximum(e2 ** 3, 1e-12))
    return np.concatenate(t21), (np.concatenate(d2) if with_d2 else None)


# ------------------------------------------------------------------------------------------------ edits
def ed_rot(x, seed):
    pt, r, th, real = polar(x); g = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(x))
    return from_polar(x, r, th + g[:, None], real)


def ed_scramble(x, seed):
    pt, r, th, real = polar(x); t = np.random.default_rng(seed).uniform(0, 2 * np.pi, th.shape)
    return from_polar(x, r, t, real)


def hp_boundaries(normals):
    b = np.mod(np.concatenate([normals + np.pi / 2, normals - np.pi / 2]), 2 * np.pi); return np.sort(b)


def ed_cell(x, seed, bnd, rot_seed=None):
    """rot_seed: the cell arrangement rotated by a per-jet random angle (same sector sizes, not aligned with the cuts)"""
    pt, r, th, real = polar(x)
    off = np.zeros(len(x)) if rot_seed is None else np.random.default_rng(rot_seed).uniform(0, 2 * np.pi, len(x))
    th = np.mod(th - off[:, None], 2 * np.pi)
    lo_ = np.concatenate([[bnd[-1] - 2 * np.pi], bnd]); hi_ = np.concatenate([bnd, [bnd[0] + 2 * np.pi]])
    k = np.searchsorted(bnd, th, side='right')                     # sector k: [lo_[k], hi_[k])
    u = np.random.default_rng(seed).uniform(0, 1, th.shape)
    t = lo_[k] + u * (hi_[k] - lo_[k]) + off[:, None]
    return from_polar(x, r, t, real)


def make_edits(x, bnd=None, nnull=20):
    E = {'rot': ed_rot(x, 2410), 'A': ed_scramble(x, 2411)}
    E['At'] = hold_moments(E['A'], x, rescale=False); E["A'"] = hold_moments(E['A'], x, rescale=True)
    if bnd is not None:
        E['B'] = ed_cell(x, 2412, bnd); E["B'"] = hold_moments(E['B'], x, rescale=True)
        E['Br'] = ed_cell(x, 2412, bnd, rot_seed=2413)               # control: same cells, rotated per jet
    return {k: v.astype(np.float32) for k, v in E.items()}


def null_edit(x, k):
    a = ed_scramble(x, 3000 + k); return a.astype(np.float32), hold_moments(a, x, rescale=True).astype(np.float32)


def edit_checks(x, E):
    base = moments(x); pt0, r0, th0, real = polar(x); out = {}
    for k, e in E.items():
        mo = moments(e); _, r, th, _ = polar(e)
        out[k] = dict(
            max_rel_dM0=float(np.max(np.abs(mo['M0'] / base['M0'] - 1))),
            max_rel_dM2=float(np.max(np.abs(mo['M2'] / base['M2'] - 1))),
            med_rel_dM2=float(np.median(np.abs(mo['M2'] / base['M2'] - 1))),
            med_abs_dM1_over_M0_mrad=float(1e3 * np.median(np.linalg.norm(mo['M1'] - base['M1'], axis=1) / base['M0'])),
            p90_abs_dM1_over_M0_mrad=float(1e3 * np.quantile(np.linalg.norm(mo['M1'] - base['M1'], axis=1) / base['M0'], .9)),
            med_rel_dm=float(np.median(np.abs(mo['m'] / np.maximum(base['m'], 1e-9) - 1))),
            p90_rel_dm=float(np.quantile(np.abs(mo['m'] / np.maximum(base['m'], 1e-9) - 1), .9)),
            max_dpt=float(np.max(np.abs(np.asarray(e)[..., 0] - np.asarray(x)[..., 0]))),
            padding_changed=int((np.asarray(e)[~real] != np.asarray(x)[~real]).sum()),
            med_abs_dr_mrad=float(1e3 * np.median(np.abs(r - r0)[real])),
            p90_abs_dr_mrad=float(1e3 * np.quantile(np.abs(r - r0)[real], .9)))
    return out


# ------------------------------------------------------------------------------------------------ models
class FloatDS:
    """dsets-d64 float forward in numpy (as interp.x11_sae_atlas.FloatDS)"""
    def __init__(self, path):
        z = np.load(path); self.z = {k: np.asarray(z[k], np.float64) for k in z.files}
        self.shift, self.scale = self.z['shift'], self.z['scale']

    def run(self, x, batch=2500):
        out = dict(L=[], logits=[]); z = self.z
        for i in range(0, len(x), batch):
            xb = np.asarray(x[i:i + batch], np.float64); m = (xb[..., 0] > 0).astype(np.float64)
            xn = ((xb - self.shift) / self.scale) * m[..., None]
            a = np.maximum(xn @ z['phi_0_weight'].T + z['phi_0_bias'], 0)
            u = np.maximum(a @ z['phi_2_weight'].T + z['phi_2_bias'], 0) * m[..., None]; S = u.sum(1)
            h = np.maximum(S @ z['rho_0_weight'].T + z['rho_0_bias'], 0); h = np.maximum(h @ z['rho_2_weight'].T + z['rho_2_bias'], 0)
            h = np.maximum(h @ z['rho_4_weight'].T + z['rho_4_bias'], 0)
            out['L'].append(S); out['logits'].append(h @ z['head_weight'].T + z['head_bias'])
        return {k: np.concatenate(v) for k, v in out.items()}


def get_model(name):
    """-> (run(x) -> dict with 'logits' and 'L' (pooled latent; absent for tf), reference accuracy, kind)"""
    if name == 'jedi':
        from interp.jedi_core import Jedi
        j = _JEDI.get('net') or Jedi(64)
        return (lambda x: (lambda r: dict(L=r['S'], logits=r['logits']))(j.run(x, keep=('S', 'logits')))), 0.809, 'jedi', j
    if name.startswith('dsets'):
        d = SMALL / 'dsets-d64-n64' / f'seed{name[-1]}'; net = FloatDS(d / 'weights.npz')
        return net.run, json.loads((d / 'test.json').read_text())['accuracy'], 'dsets', net
    if name.startswith('pfn'):
        from interp.x9_efn_pfn import Flow
        net = Flow(f'pfn_64-64-32_s{name[-1]}')
        return (lambda x: net.run(x, keep=('L', 'logits'))), net.info.get('acc_test'), 'pfn', net
    if name.startswith('x12:'):
        from interp.x12_retrained_jedi import Net, model_path
        m = name[4:]; mp, _, tp = model_path(m, 'final'); net = Net('x12', mp)
        acc = json.loads(Path(tp).read_text()).get('accuracy') if Path(tp).exists() else None
        return (lambda x: (lambda r: dict(L=r['S'], logits=r['logits']))(net.run(x, keep=('S', 'logits')))), acc, 'x12', net
    if name.startswith('tf'):
        import importlib.util, torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
        spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
        M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
        d = SMALL / 'tf-2b-d32-n64' / f'seed{name[-1]}'; arch = json.loads((d / 'arch.json').read_text())
        net = M.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
        net.load_state_dict(torch.load(d / 'best.pt', map_location='cpu')); net.eval()
        sh = np.asarray(arch['inputs']['shift'], np.float32); sc = np.asarray(arch['inputs']['scale'], np.float32)

        def run(x, batch=2000):
            out = []
            with torch.no_grad():
                for i in range(0, len(x), batch):
                    xb = np.asarray(x[i:i + batch], np.float32); m = xb[..., 0] > 0; xn = (xb - sh) / sc; xn[~m] = 0
                    out.append(net(torch.from_numpy(xn), torch.from_numpy(m)).double().numpy())
            return dict(logits=np.concatenate(out))
        return run, json.loads((d / 'test.json').read_text())['accuracy'], 'tf', net
    raise ValueError(name)


_JEDI = {}
GROUPS = {'main': ['jedi', 'dsets_s0', 'dsets_s1', 'pfn_s0', 'pfn_s1'],
          'x12': [f'x12:{m}' for m in ('hgq-r150_s0', 'hgq-r150_s1', 'hgq-r150_s2', 'hgq_s0', 'hgq_s1', 'hgq_s2', 'float_s0', 'float_s1')],
          'tf': ['tf_s0', 'tf_s1']}


# ------------------------------------------------------------------------------------------------ samples and calibration pairs
def sample_idx(y_all, per_class, seed):
    rng = np.random.default_rng(seed)        # the same draws as interp.jedi_core.sample
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y_all == c), per_class, replace=False) for c in range(5)]))


def prepare(nnull):
    from interp.jedi_core import sample
    from jetdistill.config import RESULTS
    x, y, Q = sample(2000, 2401); log(f'main sample: sample(2000, 2401), N {len(x)}')
    xp, yp, _ = sample(20000, 2402)
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y_all = np.asarray(d['y'])
    im, ip = sample_idx(y_all, 2000, 2401), sample_idx(y_all, 20000, 2402)
    assert np.array_equal(y_all[im], y) and np.array_equal(y_all[ip], yp)
    assert np.array_equal(np.asarray(d['x'][im[:50]], np.float32), x[:50])
    keep = ~np.isin(ip, im); xp, yp = xp[keep], yp[keep]
    log(f'pool: sample(20000, 2402) minus {int((~keep).sum())} main jets → {len(xp)}')
    return x, y, Q, xp, yp


def calibration_pairs(x, y, xp, yp, dt_target, t21, t21p, tol=0.02, miss=0.02, radial=False):
    """radial=False: match on log m, log M0, log M2, log pT_jet (|Δ| < tol); radial=True: also log girth (Σ pT r / M0, < tol)
    and multiplicity (|Δn| <= 2, entered as n / 100 with tol 0.02 → |Δn| < 2 + rounding)"""
    def f(xx):
        mo = moments(xx); cols = [np.log(np.maximum(mo['m'], 1e-6)), np.log(mo['M0']), np.log(mo['M2']), np.log(mo['ptj'])]
        if radial:
            xx = np.asarray(xx, np.float64); r = np.hypot(xx[..., 1], xx[..., 2])
            cols += [np.log(np.maximum((xx[..., 0] * r).sum(1) / mo['M0'], 1e-9)), (xx[..., 0] > 0).sum(1) / 100.0 * (0.02 / 0.0201)]
        return np.stack(cols, 1)
    F, Fp = f(x), f(xp)
    shift = -np.ones(len(x), int); same = -np.ones(len(x), int); ncand = np.zeros(len(x), int)
    for c in range(5):
        I = np.flatnonzero(y == c); P = np.flatnonzero(yp == c)
        for s in range(0, len(I), 100):
            Ii = I[s:s + 100]
            ok = (np.abs(F[Ii, None, :] - Fp[None, P, :]) < tol).all(-1)       # (b, |P|)
            ncand[Ii] = ok.sum(1)
            tp = t21p[P][None, :]
            d1 = np.where(ok, np.abs(tp - (t21[Ii] + dt_target[Ii])[:, None]), np.inf)
            k1 = d1.argmin(1); g1 = d1[np.arange(len(Ii)), k1] < miss
            d0 = np.where(ok, np.abs(tp - t21[Ii][:, None]), np.inf); d0[np.arange(len(Ii)), k1] = np.inf
            k0 = d0.argmin(1); g0 = d0[np.arange(len(Ii)), k0] < miss
            shift[Ii] = np.where(g1, P[k1], -1); same[Ii] = np.where(g0, P[k0], -1)
    return shift, same, ncand


# ------------------------------------------------------------------------------------------------ JEDI official internals
def jedi_internals(j, x, live_a, batch=2000):
    out = {k: [] for k in ('a', 'P', 'S', 'h4', 'h5', 'z', 'logits')}
    for i in range(0, len(x), batch):
        r = j.run(x[i:i + batch], keep=('a', 'P', 'S', 'h4', 'h5', 'z', 'logits'))
        out['a'].append(r['a'][:, :, live_a].astype(np.float32))
        for k in ('P', 'S', 'h4', 'h5', 'z', 'logits'): out[k].append(r[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def run_group(a):
    groups = GROUPS[a.group]; JS = OUT / f'e24_{a.group}{"_smoke" if a.smoke else ""}.json'
    R = dict(node='E24', group=a.group, plan='module docstring', sample='interp.jedi_core.sample(per_class=2000, seed=2401)',
             pool='interp.jedi_core.sample(per_class=20000, seed=2402) minus main jets', models={})
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    x, y, Q, xp, yp = prepare(a.nnull)
    if a.smoke: x, y = x[::10], y[::10]; Q = {k: v[::10] for k, v in Q.items()}; xp, yp = xp[::4], yp[::4]
    N = len(x); R['N'] = N
    # half-plane cells from the official l1 weights (numpy only: the fused kernel is read once below for the jedi group;
    # for the other groups the boundaries are taken from e24_main.json if present)
    bnd = None; hp_info = None
    if 'jedi' in groups:
        from interp.jedi_core import Jedi
        j0 = Jedi(64); _JEDI['net'] = j0; K, b = j0.KB['l1']; b = b.reshape(-1); K = K.reshape(3, 64); sc, sh = j0.scale, j0.shift
        alpha = K / sc[:, None]; beta = b - (K * (sh / sc)[:, None]).sum(0)          # physical: α·(pT, Δη, Δφ) + β
        normals = np.array([np.arctan2(alpha[2, c], alpha[1, c]) for c in HP])
        hp_info = {f'a{c}': dict(normal_deg=float(np.degrees(np.mod(n_, 2 * np.pi))), w_pt=float(alpha[0, c]),
                                 w_eta=float(alpha[1, c]), w_phi=float(alpha[2, c]), beta=float(beta[c]),
                                 offset_rad=float(abs(beta[c]) / max(np.hypot(alpha[1, c], alpha[2, c]), 1e-12)))
                   for c, n_ in zip(HP, normals)}
        bnd = hp_boundaries(normals); R['halfplanes'] = hp_info; R['cell_boundaries_deg'] = np.degrees(bnd)
        live_a = [c for c in range(64) if np.abs(K[:, c]).sum() > 0]; R['l1_nonzero_units'] = live_a
        log(f'half-plane normals (deg): {np.round(np.degrees(np.mod(normals, 2 * np.pi)), 1).tolist()}; l1 nonzero units {live_a}')
    E = make_edits(x, bnd, nnull=a.nnull)
    R['edit_checks'] = edit_checks(x, E); log('edit checks:', json.dumps(clean(R['edit_checks']))); save()
    # substructure
    t0, d0 = substructure(x)
    if 'tau21' in Q:
        R['tau21_check_vs_Q'] = dict(max_abs=float(np.max(np.abs(t0 - Q['tau21']))), corr=float(np.corrcoef(t0, Q['tau21'])[0, 1]))
    if 'D2' in Q:
        R['D2_check_vs_Q'] = dict(max_abs=float(np.max(np.abs(d0 - Q['D2']))), corr=float(np.corrcoef(d0, Q['D2'])[0, 1]))
    log('substructure checks:', R.get('tau21_check_vs_Q'), R.get('D2_check_vs_Q'))
    SUB = {}; TT = {}
    for k, e in E.items():
        t, d = substructure(e); TT[k] = t
        SUB[k] = {}
        for c in range(5):
            m_ = y == c
            SUB[k][CL[c]] = dict(tau21_before=float(np.median(t0[m_])), tau21_after=float(np.median(t[m_])),
                                 dtau21=ci(t[m_] - t0[m_]), D2_before=float(np.median(d0[m_])), D2_after=float(np.median(d[m_])),
                                 dD2=ci(d[m_] - d0[m_]))
        log(f'substructure {k}: ' + ', '.join(f'{c} τ21 {v["tau21_before"]:.3f}→{v["tau21_after"]:.3f} D2 {v["D2_before"]:.2f}→{v["D2_after"]:.2f}' for c, v in SUB[k].items()))
    R['substructure'] = SUB; save()
    # calibration pairs (real jets)
    t0p, _ = substructure(xp, with_d2=False)
    dt = TT["A'"] - t0
    shift, same, ncand = calibration_pairs(x, y, xp, yp, dt, t0, t0p)
    ok = (shift >= 0) & (same >= 0)
    R['calibration_pairs'] = dict(n_with_candidates=int((ncand > 0).sum()), median_candidates=float(np.median(ncand)),
                                  n_shift=int((shift >= 0).sum()), n_same=int((same >= 0).sum()), n_both=int(ok.sum()),
                                  per_class={CL[c]: int((ok & (y == c)).sum()) for c in range(5)},
                                  dtau21_target_on_pairs=ci(dt[ok]) if ok.any() else None,
                                  dtau21_shift_partner=ci(t0p[shift[ok]] - t0[ok]) if ok.any() else None,
                                  dtau21_same_partner=ci(t0p[same[ok]] - t0[ok]) if ok.any() else None)
    log('calibration pairs:', json.dumps(clean(R['calibration_pairs'])))
    shiftR, sameR, ncandR = calibration_pairs(x, y, xp, yp, dt, t0, t0p, radial=True)
    okR = (shiftR >= 0) & (sameR >= 0)
    R['calibration_pairs_radial'] = dict(n_with_candidates=int((ncandR > 0).sum()), n_both=int(okR.sum()),
                                         per_class={CL[c]: int((okR & (y == c)).sum()) for c in range(5)},
                                         dtau21_shift_partner=ci(t0p[shiftR[okR]] - t0[okR]) if okR.any() else None)
    log('calibration pairs (radial-matched):', json.dumps(clean(R['calibration_pairs_radial'])))
    upart = np.unique(np.concatenate([shift[ok], same[ok], shiftR[okR], sameR[okR]])); xpart = xp[upart]; pos = {v: i for i, v in enumerate(upart)}
    save()

    for name in groups:
        tm = time.time(); log(f'===== model {name}')
        try:
            run, acc_ref, kind, net = get_model(name)
        except Exception as ex:
            log(f'model {name} failed to load: {ex!r}'); R['models'][name] = dict(error=repr(ex)); save(); continue
        M = dict(kind=kind, acc_ref=acc_ref); R['models'][name] = M
        base = run(x); c0 = base['logits'].argmax(1); M['acc'] = float((c0 == y).mean())
        M['acc_per_class'] = {CL[c]: float((c0[y == c] == c).mean()) for c in range(5)}
        log(f'{name}: accuracy {M["acc"]:.4f} (reference {acc_ref})')
        outs = {k: run(e) for k, e in E.items()}
        ch = {k: o['logits'].argmax(1) != c0 for k, o in outs.items()}
        M['edits'] = {}
        heavy = np.isin(c0, [2, 3, 4]); light = np.isin(c0, [0, 1])
        for k, o in outs.items():
            c1 = o['logits'].argmax(1); flow = np.zeros((5, 5), int); np.add.at(flow, (c0, c1), 1)
            dlog = np.linalg.norm(o['logits'] - base['logits'], axis=1)
            em = dict(calls_changed=ci(ch[k]), acc_after=float((c1 == y).mean()), flow=flow,
                      heavy_to_light=ci((np.isin(c1, [0, 1]) & heavy)[heavy]) if heavy.any() else None,
                      light_to_heavy=ci((np.isin(c1, [2, 3, 4]) & light)[light]) if light.any() else None,
                      W_to_Z=float(((c1 == 3) & (y == 2)).sum() / max((y == 2).sum(), 1) - ((c0 == 3) & (y == 2)).sum() / max((y == 2).sum(), 1)),
                      dlogit_norm=ci(dlog))
            if k != 'rot':
                em['mcnemar_vs_rot'] = mcnemar(ch[k], ch['rot']); em['diff_vs_rot'] = ci(ch[k].astype(float) - ch['rot'])
            if 'L' in base:
                Lc = base['L'] - base['L'].mean(0)
                em['latent_change_over_spread'] = ci_ratio(np.linalg.norm(o['L'] - base['L'], axis=1), np.linalg.norm(Lc, axis=1))
            M['edits'][k] = em
            log(f'{name} {k}: calls changed {em["calls_changed"]["mean"]:.4f} {np.round(em["calls_changed"]["ci"], 4).tolist()}'
                f' heavy→light {em["heavy_to_light"]["mean"] if em["heavy_to_light"] else None}'
                + (f' McNemar vs rot p {em["mcnemar_vs_rot"]["p"]:.2e}' if k != 'rot' else '')
                + (f' latent Δ/spread {em["latent_change_over_spread"]["value"]:.4f}' if 'L' in base else ''))
        # redraw nulls
        nn_ = {}
        for k in range(a.nnull if name == 'jedi' else a.nnull_other):
            ea, eap = null_edit(x, k)
            nn_[f'A#{k}'] = float((run(ea)['logits'].argmax(1) != c0).mean())
            nn_[f"A'#{k}"] = float((run(eap)['logits'].argmax(1) != c0).mean())
        M['redraws'] = {pref: dict(values=[v for k, v in nn_.items() if k.split('#')[0] == pref],
                                   mean=float(np.mean([v for k, v in nn_.items() if k.split('#')[0] == pref])),
                                   min=float(np.min([v for k, v in nn_.items() if k.split('#')[0] == pref])),
                                   max=float(np.max([v for k, v in nn_.items() if k.split('#')[0] == pref])))
                        for pref in ('A', "A'")}
        log(f'{name} redraws: A {M["redraws"]["A"]["min"]:.4f}–{M["redraws"]["A"]["max"]:.4f}, '
            f'A\' {M["redraws"][chr(65) + chr(39)]["min"]:.4f}–{M["redraws"][chr(65) + chr(39)]["max"]:.4f}')
        # calibration with real jets
        cp = run(xpart)['logits'].argmax(1) if len(xpart) else None
        for lab, sh_, sa_, ok_ in (('calibration', shift, same, ok), ('calibration_radial', shiftR, sameR, okR)):
            if not ok_.any(): continue
            cs = cp[[pos[v] for v in sh_[ok_]]]; cz = cp[[pos[v] for v in sa_[ok_]]]; cj = c0[ok_]
            d_shift = cs != cj; d_same = cz != cj
            M[lab] = dict(n=int(ok_.sum()), differ_shift=ci(d_shift), differ_same=ci(d_same),
                                    tau21_attributable=ci(d_shift.astype(float) - d_same), mcnemar_shift_vs_same=mcnemar(d_shift, d_same),
                                    heavy_to_light_shift=ci((np.isin(cs, [0, 1]))[np.isin(cj, [2, 3, 4])]) if np.isin(cj, [2, 3, 4]).any() else None,
                                    heavy_to_light_same=ci((np.isin(cz, [0, 1]))[np.isin(cj, [2, 3, 4])]) if np.isin(cj, [2, 3, 4]).any() else None,
                                    Aprime_calls_changed_same_jets=ci(ch["A'"][ok_]), rot_calls_changed_same_jets=ci(ch['rot'][ok_]),
                                    Aprime_minus_tau21_attributable=ci(ch["A'"][ok_].astype(float) - (d_shift.astype(float) - d_same)),
                                    mcnemar_Aprime_vs_shift=mcnemar(ch["A'"][ok_], d_shift))
            log(f'{name} {lab}: differ shift {M[lab]["differ_shift"]["mean"]:.4f}, same {M[lab]["differ_same"]["mean"]:.4f}, '
                f'τ21-attributable {M[lab]["tau21_attributable"]["mean"]:.4f} {np.round(M[lab]["tau21_attributable"]["ci"], 4).tolist()}; '
                f'A\' on same jets {M[lab]["Aprime_calls_changed_same_jets"]["mean"]:.4f}')
        if kind == 'jedi':
            M['internals'] = jedi_block(net, x, E, base, live_a)
        M['wall_s'] = time.time() - tm; save()
        del outs
    save(); log(f'wrote {JS}')


def jedi_block(j, x, E, base, live_a):
    """per-unit changes in the official network (l1 cuts, P, S, h4, h5, z)"""
    I0 = jedi_internals(j, x, live_a); real = x[..., 0] > 0
    # live l1: mean ablation over real particles (on x[::2])
    xs = x[::2]; c0s = I0['logits'][::2].argmax(1); live_l1 = []
    a_full_mean = {}
    for ci_, c in enumerate(live_a):
        am = float(I0['a'][..., ci_][real].mean()); a_full_mean[c] = am
        m = np.zeros(64); m[c] = 1; v = np.zeros(64); v[c] = am
        chg = float((j.run(xs, override={'a': (m, v)})['logits'].argmax(1) != c0s).mean())
        if chg >= 0.01: live_l1.append((c, chg))
    # live S: resample by mean (unit set to its sample mean) on x[::2]
    live_S = []
    Ssd = I0['S'].std(0)
    for c in np.flatnonzero(Ssd > 1e-9):
        m = np.zeros(64); m[c] = 1; v = np.zeros(64); v[c] = I0['S'][:, c].mean()
        chg = float((j.run(xs, override={'S': (m, v)})['logits'].argmax(1) != c0s).mean())
        if chg >= 0.01: live_S.append((int(c), chg))
    log(f'jedi live l1 {[(c, round(v, 3)) for c, v in live_l1]}; live S {[(c, round(v, 3)) for c, v in live_S]}')
    out = dict(live_l1={f'a{c}': v for c, v in live_l1}, live_S={f'S{c}': v for c, v in live_S}, per_edit={})
    asd = {c: float(I0['a'][..., live_a.index(c)][real].std()) for c, _ in live_l1}
    Psd = I0['P'].std(0); nreal = np.maximum(real.sum(1), 1)
    dS_by = {}
    for k, e in E.items():
        I1 = jedi_internals(j, e, live_a); pe = dict(l1={}, S={}, stages={})
        for c, _ in live_l1:
            ia = live_a.index(c)
            da = np.abs(I1['a'][..., ia] - I0['a'][..., ia]) * real
            on0 = (I0['a'][..., ia] > 0); on1 = (I1['a'][..., ia] > 0)
            pe['l1'][f'a{c}'] = dict(kind='half-plane' if c in HP else 'other',
                                     particle_dA_over_sd=ci(da.sum(1) / nreal / max(asd[c], 1e-12)),
                                     pooled_dP_over_sd=ci(np.abs(I1['P'][:, c] - I0['P'][:, c]) / max(Psd[c], 1e-12)),
                                     onoff_flip_frac=float(((on0 != on1) & real).sum() / real.sum()))
        for c, _ in live_S:
            d = np.abs(I1['S'][:, c] - I0['S'][:, c])
            pe['S'][f'S{c}'] = dict(dS_over_sd=ci(d / max(Ssd[c], 1e-12)), rel_change=ci_ratio(d, np.abs(I0['S'][:, c])),
                                    radial_named=c in RADIAL_S)
            dS_by.setdefault(k, {})[c] = d / max(Ssd[c], 1e-12)
        for st in ('P', 'S', 'h4', 'h5', 'z', 'logits'):
            sd = I0[st].std(0); keep = sd > 1e-9
            pe['stages'][st] = ci((np.abs(I1[st] - I0[st])[:, keep] / sd[keep]).mean(1))
        out['per_edit'][k] = pe
        log(f'jedi internals {k}: stages ' + ', '.join(f'{s} {v["mean"]:.4f}' for s, v in pe['stages'].items()) + '; S '
            + ', '.join(f'{u} {v["dS_over_sd"]["mean"]:.3f}/{v["rel_change"]["value"]:.3f}' for u, v in pe['S'].items()))
        log(f'jedi l1 {k}: ' + ', '.join(f'{u} part {v["particle_dA_over_sd"]["mean"]:.3f} pool {v["pooled_dP_over_sd"]["mean"]:.3f} flip {v["onoff_flip_frac"]:.3f}' for u, v in pe['l1'].items()))
    # tests per S unit: edit vs rot (paired bootstrap p), Holm over live S units, for A, A', B, B'
    out['S_tests'] = {}
    for k in [e for e in E if e != 'rot']:
        units = sorted(dS_by[k]); ps = [boot_p(dS_by[k][c] - dS_by['rot'][c]) for c in units]; hp = holm(ps)
        out['S_tests'][k] = {f'S{c}': dict(diff=ci(dS_by[k][c] - dS_by['rot'][c]), p=p, p_holm=float(h)) for c, p, h in zip(units, ps, hp)}
        out['S_tests'][k]['_n_tested'] = len(units)
    # A vs A' per unit (the M1 part of the scramble)
    units = sorted(dS_by['A']); ps = [boot_p(dS_by['A'][c] - dS_by["A'"][c]) for c in units]; hp = holm(ps)
    out['S_tests']['A_vs_Aprime'] = {f'S{c}': dict(diff=ci(dS_by['A'][c] - dS_by["A'"][c]), p=p, p_holm=float(h)) for c, p, h in zip(units, ps, hp)}
    return out


def combine(a):
    S = dict(node='E24', models={}, holm={})
    rows = []
    for g in ('main', 'x12', 'tf'):
        p = OUT / f'e24_{g}.json'
        if not p.exists(): continue
        D = json.loads(p.read_text())
        for m, v in D['models'].items():
            if 'edits' not in v: continue
            S['models'][m] = dict(acc=v['acc'], **{k: v['edits'][k]['calls_changed'] for k in v['edits']},
                                  calib=v.get('calibration', {}).get('tau21_attributable'))
            for k in ('A', 'At', "A'"):
                rows.append((m, k, v['edits'][k]['mcnemar_vs_rot']['p']))
    if rows:
        hp = holm([r[2] for r in rows])
        S['holm'] = {f'{m}:{k}': dict(p=p, p_holm=float(h)) for (m, k, p), h in zip(rows, hp)}; S['n_tests'] = len(rows)
    (OUT / 'e24_summary.json').write_text(json.dumps(clean(S), indent=1)); log('wrote e24_summary.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=list(GROUPS)); ap.add_argument('--combine', action='store_true')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--nnull', type=int, default=20)
    ap.add_argument('--nnull-other', dest='nnull_other', type=int, default=5)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    combine(a) if a.combine else run_group(a)
