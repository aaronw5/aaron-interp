"""E14 (JEDI-linear 64 p, 3-feature; claims A3 / A4): how the radial second-round objects are built from the nine half-plane
cuts of round one, and whether that construction predicts the direction-dependent (odd) term of the mass response under a
rigid translation.

Sample: sample(per_class=2000, seed=8) = 10,000 test jets.  Reference: the fixed-point network (jedi_core.Jedi).  Program: the
float + zero-bit-mask program (E3m / E5: every quantizer replaced by its zero-bit mask, no rounding, no wrap; same arithmetic as
e4_e5_jedi.make_masked, written here in numpy so the pre-activations can be decomposed) with the context g frozen at the
program's sample mean (E5's per-particle response f_c).
Part 1  structure (exact in the program):
          pre_u_c(i) = Σ_j G_jc(i) a_j(i) + Σ_k W3[k,c] m3[i,k] mas[i,k] 1[z_k(i)>0] b2[k] + ctx_c(i) + b3[c]
          G_jc(i) = ∂pre_u_c/∂a_j = Σ_k W3[k,c] m3[i,k] mas[i,k] 1[z_k(i)>0] W2[j,k] ml2[i,j]
        variance shares (cov(term, pre)/var(pre)) of the sources over real particles in the unit's pooled slots: the nine half-
        planes a3 a6 a29 a30 a36 a40 a43 a46 a57, the pT ramps a15 a21 a32 a56, the constant a4, the s biases, the context g;
        G_jc averaged over active particles (u_c > 0); the s units in between with the sign of each path.
Part 2  polar maps f_c(pT, r, θ) of one particle at slot 5 and 20 (pT 1, 3, 10, 30, 100 GeV; r 0–0.4 in 41 steps; θ 0–360° in 72
        steps): program with g at its mean (primary), fixed-point network with g at its mean (check; grid points outside the slot's
        input quantizer range are dropped), untrained network (control).  Per map: θ-Fourier power fractions (m = 0, 1, 2, ≥ 3);
        the on-contour r_c(θ) (zero crossing of the pre-activation along r), its anisotropy max/min and the θ of its minimum;
        comparison with the polygon ρ_c(θ) = Σ_j w_jc ReLU(n_j·x̂(θ)) from the nine half-plane normals n_j (rad⁻¹) and the Part-1
        weights w_jc (r_pred(θ) ∝ 1/|ρ_c(θ)|); controls: w_jc permuted among the nine half-planes, equal weights.
Part 3  prediction of the direction term: rigid translation δ ∈ {0.03, 0.06} along ±η, ±φ, ±diag (real particles only; padded
        slots untouched): ΔS_c predicted by the program (g at the sample mean = the maps; g at the jet's own unedited g; g free)
        vs measured on the fixed-point network (g free; g frozen at the unedited jet's g); even / odd parts in δ per axis; W→Z
        rates; ordering against E6v's h5 pattern.  45° rotation the same way (prediction added; E5v measures it).
Part 4  causal: a_j → 0 for all particles (override of stage a on the fixed-point network; g free and g frozen) for the nine half-
        planes, a dead l1 unit and the pT ramp a56: per azimuth sector of the particle (12 × 30°) the mean change of u_c per
        particle and the sector's share of ΔS_c / sd(S_c); % calls changed; the program's prediction (g at its mean); own-sector /
        opposite-sector selectivity (own = the sector of the half-plane normal n_j).
Run:  python -m interp.e14_halfplanes_jedi [--smoke] [--plot]
Out:  $AI_RESULTS/jedi/e14_halfplanes.json, e14_maps.npz, e14_halfplanes.png, e14_structure_table.png
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
SMOKE = '--smoke' in sys.argv
TAG = 'e14_smoke' if SMOKE else 'e14_halfplanes'
JSON_PATH = OUT / f'{TAG}.json'; NPZ_PATH = OUT / ('e14_smoke_maps.npz' if SMOKE else 'e14_maps.npz')
PNG = OUT / f'{TAG}.png'; PNG_TAB = OUT / ('e14_smoke_table.png' if SMOKE else 'e14_structure_table.png')

SEED, PER = 8, 2000
HP = [29, 6, 36, 40, 43, 46, 3, 57, 30]              # the nine angular half-planes (zero pT weight)
PTU = [15, 21, 32, 56]                                # the pT ramps
CONST = [4, 50]                                       # constant l1 units (a50 feeds nothing)
MAIN = [29, 1, 50, 43, 30]                            # the units of Part 3
PT_GRID = [1.0, 3.0, 10.0, 30.0, 100.0]
R_GRID = np.linspace(0.0, 0.4, 41)
TH_DEG = np.arange(0, 360, 15 if SMOKE else 5)
SLOTS = (5, 20)
DIRS = {'+eta': (1.0, 0.0), '-eta': (-1.0, 0.0), '+phi': (0.0, 1.0), '-phi': (0.0, -1.0),
        '+diag': (2 ** -0.5, 2 ** -0.5), '-diag': (-2 ** -0.5, -2 ** -0.5)}
DELTAS = (0.03, 0.06)
E6V_H5 = {'+eta': .085, '-eta': .117, '+phi': .087, '-phi': .110, '+diag': .078, '-diag': .118}   # E6v item 4, δ = 0.03, h5
E6V_WZ = {'+eta': .28, '-eta': .62, '+phi': .32, '-phi': .54, '+diag': .24, '-diag': .64}
NSEC = 12
NPERM = 60 if SMOKE else 500
_LOG = []


def log(*a):
    s = ' '.join(str(v) for v in a); print(s, flush=True); _LOG.append(s)


def relu(v):
    return np.maximum(v, 0)


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))); rb = np.argsort(np.argsort(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


# ------------------------------------------------------------------ the program (float + zero-bit masks) in numpy
class Program:
    def __init__(self, net):
        L = net.L

        def qm(k):
            iq = L[k].iq; qq = iq.quantizers if hasattr(iq, 'quantizers') else [iq]
            return [np.asarray(np.asarray(q.quantizer.b) > 0, np.float32) for q in qq]
        m = {k: qm(k) for k in ('l1', 'qs', 'dd', 'l2', 'add', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')}
        self.ml1 = m['l1'][0].reshape(64, 3); self.mqs = m['qs'][0].reshape(64, 64); self.mdd = m['dd'][0].reshape(-1)
        self.ml2 = m['l2'][0].reshape(64, 64); self.mas = m['add'][0].reshape(64, 64); self.mag = m['add'][1].reshape(-1)
        self.m3 = m['l3'][0].reshape(64, 64); self.mq1 = m['qs1'][0].reshape(64, 64)
        self.md = {k: m[k][0].reshape(-1) for k in ('d4', 'd5', 'd6', 'd7')}
        self.W = {k: (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32)) for k, v in net.KB.items()}
        self.sc = net.pool_scale; self.shift = net.shift.astype(np.float32); self.scale = net.scale.astype(np.float32)

    def xn(self, x):
        return ((np.asarray(x, np.float32) - self.shift) / self.scale).astype(np.float32)

    def a_of(self, xn, slots=None):
        ml1 = self.ml1 if slots is None else self.ml1[slots]; W1, b1 = self.W['l1']
        return relu((xn * ml1) @ W1 + b1)

    def z2_of(self, a, slots=None):
        ml2 = self.ml2 if slots is None else self.ml2[slots]; W2, b2 = self.W['l2']
        return (a * ml2) @ W2 + b2

    def g_of(self, a):
        P = (a * self.mqs).sum(1) * self.sc[0]; Wd, bd = self.W['dd']
        return P, relu((P * self.mdd) @ Wd + bd)

    def pre_of(self, s, g, slots=None):
        W3, b3 = self.W['l3']
        if slots is None:
            t = s * self.mas + g[:, None, :] * self.mag; return (t * self.m3) @ W3 + b3
        t = s * self.mas[slots] + g * self.mag; return (t * self.m3[slots]) @ W3 + b3

    def S_of(self, u):
        return (u * self.mq1).sum(1) * self.sc[1]

    def head(self, S):
        h4 = relu((S * self.md['d4']) @ self.W['d4'][0] + self.W['d4'][1])
        h5 = relu((h4 * self.md['d5']) @ self.W['d5'][0] + self.W['d5'][1])
        z = (h5 * self.md['d6']) @ self.W['d6'][0] + self.W['d6'][1]
        lo = (relu(z) * self.md['d7']) @ self.W['d7'][0] + self.W['d7'][1]
        return h4, h5, z, lo

    def run(self, x, gov=None, azero=None, keep=('S', 'logits'), live=None, batch=2500):
        """gov: None (g free), (64,) or (J, 64) (g replaced); azero: l1 unit set to 0 for every slot (before the pool)"""
        out = {k: [] for k in keep}
        for i in range(0, len(x), batch):
            xb = self.xn(x[i:i + batch]); a = self.a_of(xb)
            if azero is not None: a[..., azero] = 0.0
            P, g = self.g_of(a)
            if gov is not None:
                g = np.broadcast_to(np.asarray(gov, np.float32), g.shape) if np.ndim(gov) == 1 else np.asarray(gov[i:i + batch], np.float32)
            z2 = self.z2_of(a); s = relu(z2); pre = self.pre_of(s, g); u = relu(pre); S = self.S_of(u); h4, h5, z, lo = self.head(S)
            loc = dict(a=a, P=P, g=g, z2=z2, s=s, pre=pre, u=u, S=S, h4=h4, h5=h5, z=z, logits=lo)
            for k in keep:
                v = loc[k]; out[k].append(v[..., live] if (live is not None and k in ('u', 'pre')) else v)
        return {k: np.concatenate(v) for k, v in out.items()}

    def particle(self, slot, pt, eta, phi, g):
        """pre-activation of all 64 u units for one particle (pt GeV, Δη, Δφ) at the given slot with context g (64,)"""
        x3 = np.stack([np.asarray(pt, np.float32).ravel(), np.asarray(eta, np.float32).ravel(), np.asarray(phi, np.float32).ravel()], 1)
        slots = np.full(len(x3), slot); a = self.a_of(self.xn(x3), slots); s = relu(self.z2_of(a, slots))
        return self.pre_of(s, np.asarray(g, np.float32), slots)


# ------------------------------------------------------------------ fixed-point network runs
def fixed(net, x, keep=('S', 'logits'), g_jet=None, a_zero=None, live=None, batch=2500):
    """jedi_core forward with per-jet g override (g_jet (J, 64)) and / or one l1 unit set to 0 at every slot (a_zero)"""
    out = {k: [] for k in keep}; one = np.ones(64, np.float32)
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]; O = {}
        if g_jet is not None: O['g'] = (one, np.asarray(g_jet[i:i + batch], np.float32)[:, None, :])
        if a_zero is not None:
            m = np.zeros(64, np.float32); m[a_zero] = 1; O['a'] = (m, np.zeros(64, np.float32))
        r = net._f(net.inputs(xb, net.n), O)
        for k in keep:
            v = np.asarray(r[k], np.float32)
            if k in ('u', 'a', 's'):
                if live is not None: v = v[..., live]
            else: v = v.reshape(len(xb), -1)
            out[k].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def fixed_particle_u(net, slot, pt, eta, phi, gm, batch=2000):
    """u of a single particle at `slot` of an otherwise empty jet, fixed-point network, g replaced by gm (64,)"""
    N = np.size(pt); xs = np.zeros((N, 64, 3), np.float32)
    xs[:, slot, 0] = np.ravel(pt); xs[:, slot, 1] = np.ravel(eta); xs[:, slot, 2] = np.ravel(phi)
    O = {'g': (np.ones(64, np.float32), np.asarray(gm, np.float32))}; out = []
    for i in range(0, N, batch):
        out.append(np.asarray(net._f(net.inputs(xs[i:i + batch], net.n), O)['u'], np.float32)[:, slot, :])
    return np.concatenate(out), xs


def range_tables(net):
    q = net.L['l1'].iq.quantizer
    k_, i_, f_ = (np.asarray(getattr(q, a_), np.float64).reshape(64, 3) for a_ in ('k', 'i', 'f'))
    return np.where(k_ > 0, -2.0 ** i_, 0.0), 2.0 ** i_ - 2.0 ** -f_


def in_range(net, x, LO, HI):
    """jets whose real particles all lie inside their slot's fixed-point input range (no WRAP)"""
    xn = (np.asarray(x, np.float64) - net.shift) / net.scale; real = x[..., 0] > 0
    return ~((((xn < LO) | (xn > HI)) & real[..., None]).any((1, 2)))


def shift(x, de, dp):
    e = x.copy(); rl = e[..., 0] > 0
    e[..., 1] += np.float32(de) * rl; e[..., 2] += np.float32(dp) * rl; return e


def rotate(x, deg):
    e = x.copy(); rl = e[..., 0] > 0; c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    eta, phi = x[..., 1], x[..., 2]
    e[..., 1] = np.where(rl, c * eta - s_ * phi, eta).astype(np.float32); e[..., 2] = np.where(rl, s_ * eta + c * phi, phi).astype(np.float32)
    return e


# ------------------------------------------------------------------ Part 1: decomposition
def part1(prog, x, live, KEEP, real, batch=2000):
    W2, b2 = prog.W['l2']; W3, b3 = prog.W['l3']
    GRP = {'HP': HP, 'PT': PTU, 'CONST': CONST}
    A = {c: dict(n=0, sp=0.0, spp=0.0, sA=np.zeros(64), sAp=np.zeros(64), nact=0, sG=np.zeros(64), sM=np.zeros((64, 64)), err=0.0,
                 grp={k: dict(s=0.0, ss=0.0, sp=0.0) for k in list(GRP) + ['bias', 'ctx']}) for c in live}
    for i0 in range(0, len(x), batch):
        xb = prog.xn(x[i0:i0 + batch]); a = prog.a_of(xb); z2 = prog.z2_of(a); s = relu(z2); act2 = (z2 > 0).astype(np.float32)
        P, g = prog.g_of(a); pre = prog.pre_of(s, g); rb = real[i0:i0 + batch]
        for c in live:
            D = (prog.m3 * prog.mas)[None] * act2 * W3[:, c]                 # (B, 64, 64k)
            G = np.einsum('bik,jk->bij', D, W2, optimize=True) * prog.ml2[None]   # (B, 64, 64j)  ∂pre_c/∂a_j
            Aj = a * G; bs = D @ b2; ctx = (g[:, None, :] * prog.mag * prog.m3) @ W3[:, c]; p = pre[..., c]
            A[c]['err'] = max(A[c]['err'], float(np.abs(Aj.sum(-1) + bs + ctx + b3[c] - p).max()))
            w = rb & KEEP[None, :, c]; pw = p[w].astype(np.float64); Aw = Aj[w].astype(np.float64)
            A[c]['n'] += len(pw); A[c]['sp'] += pw.sum(); A[c]['spp'] += (pw ** 2).sum()
            A[c]['sA'] += Aw.sum(0); A[c]['sAp'] += (Aw * pw[:, None]).sum(0)
            terms = {k: Aw[:, idx].sum(1) for k, idx in GRP.items()}; terms['bias'] = bs[w].astype(np.float64); terms['ctx'] = ctx[w].astype(np.float64)
            for k, v in terms.items():
                A[c]['grp'][k]['s'] += v.sum(); A[c]['grp'][k]['ss'] += (v ** 2).sum(); A[c]['grp'][k]['sp'] += (v * pw).sum()
            act = w & (p > 0); A[c]['nact'] += int(act.sum()); A[c]['sG'] += G[act].astype(np.float64).sum(0)
            A[c]['sM'] += np.einsum('bik,bi,ij->kj', D, act.astype(np.float32), prog.ml2, optimize=True).astype(np.float64) * W2.T
    res = {}
    for c in live:
        d = A[c]; n = max(d['n'], 1); mp = d['sp'] / n; var = d['spp'] / n - mp ** 2
        share = (d['sAp'] / n - (d['sA'] / n) * mp) / var if var > 0 else np.zeros(64)
        grp = {}
        for k, v in d['grp'].items():
            m_ = v['s'] / n; grp[k] = dict(share=float((v['sp'] / n - m_ * mp) / var) if var > 0 else 0.0, sd=float(np.sqrt(max(v['ss'] / n - m_ ** 2, 0))), mean=float(m_))
        Gbar = d['sG'] / max(d['nact'], 1); M = d['sM'] / max(d['nact'], 1)
        hp, ctx = grp['HP']['share'], grp['ctx']['share']
        paths = []
        for k in np.flatnonzero(np.abs(W3[:, c]) > 0):
            feeds = [(int(j), float(M[k, j])) for j in HP + PTU + CONST if abs(M[k, j]) > 1e-6]
            paths.append(dict(s=int(k), W3=float(W3[k, c]), b2=float(b2[k]), slots_l3=int(prog.m3[:, k].sum()), eff_weights={f'a{j}': w_ for j, w_ in feeds},
                              W2={f'a{j}': float(W2[j, k]) for j in HP + PTU + CONST if abs(W2[j, k]) > 0}))
        res[int(c)] = dict(n_particles=int(d['n']), n_active=int(d['nact']), sd_pre=float(np.sqrt(max(var, 0))), mean_pre=float(mp), decomposition_max_err=d['err'],
                           share={f'a{j}': float(share[j]) for j in HP + PTU + CONST}, groups=grp,
                           hp_over_hp_plus_ctx=float(hp / (hp + abs(ctx))) if (hp + abs(ctx)) > 0 else None,
                           hp_sd_over_ctx_sd=float(grp['HP']['sd'] / grp['ctx']['sd']) if grp['ctx']['sd'] > 0 else None,
                           eff_weight_active={f'a{j}': float(Gbar[j]) for j in HP + PTU + CONST}, paths=paths, b3=float(b3[c]))
    return res


# ------------------------------------------------------------------ Part 2: maps, contours, Fourier, polygon
def grid():
    PT, RR, TH = np.meshgrid(np.array(PT_GRID), R_GRID, np.radians(TH_DEG), indexing='ij')
    return PT, RR * np.cos(TH), RR * np.sin(TH)


def fourier_stats(f):
    """θ-power fractions of a map (..., nθ) summed over the cells that respond; m = 0, 1, 2, >= 3"""
    f = np.where(np.isfinite(f), f, 0.0); n = f.shape[-1]
    F = np.fft.rfft(f, axis=-1); Pw = np.abs(F) ** 2; Pw[..., 1:] *= 2
    if n % 2 == 0: Pw[..., -1] /= 2
    cells = (f > 0).any(-1)
    if not cells.any(): return None
    tot = Pw[cells].sum(0); T = tot.sum()
    return dict(m0=float(tot[0] / T), m1=float(tot[1] / T), m2=float(tot[2] / T), m3plus=float(tot[3:].sum() / T), n_cells=int(cells.sum()))


def contour_pre(pre, r=R_GRID):
    """zero crossing of the pre-activation along r per θ (first sign change), linear interpolation; NaN if none"""
    rc = np.full(pre.shape[1], np.nan)
    for t in range(pre.shape[1]):
        v = pre[:, t]
        if not np.all(np.isfinite(v)): continue
        s0 = v[0] > 0; idx = np.flatnonzero((v > 0) != s0)
        if len(idx) == 0: continue
        k = idx[0]; rc[t] = r[k - 1] + (0 - v[k - 1]) * (r[k] - r[k - 1]) / (v[k] - v[k - 1])
    return rc


def contour_relu(f, r=R_GRID):
    """on/off boundary from a post-ReLU map: the first (outer) or last (inner) active point, refined by the local slope"""
    rc = np.full(f.shape[1], np.nan); dr = r[1] - r[0]
    for t in range(f.shape[1]):
        v = f[:, t]; ok = np.isfinite(v)
        if ok.sum() < 3: continue
        on = (v > 0) & ok; idx = np.flatnonzero(on)
        if len(idx) == 0 or len(idx) == ok.sum(): continue
        if not on[np.flatnonzero(ok)[0]]:                       # outer: turns on at k
            k = idx[0]
            if k + 1 < len(v) and ok[k + 1] and v[k + 1] > v[k]: rc[t] = np.clip(r[k] - v[k] * dr / (v[k + 1] - v[k]), r[k] - dr, r[k])
            else: rc[t] = r[k] - dr / 2
        else:                                                    # inner: turns off after k
            k = idx[-1]
            if k >= 1 and ok[k - 1] and v[k - 1] > v[k]: rc[t] = np.clip(r[k] + v[k] * dr / (v[k - 1] - v[k]), r[k], r[k] + dr)
            else: rc[t] = r[k] + dr / 2
    return rc


def contour_stats(rc):
    ok = np.isfinite(rc); d = dict(n_finite=int(ok.sum()), profile=rc.tolist())
    if ok.sum() < 0.9 * len(rc): return d
    v = rc[ok]; d.update(mean=float(v.mean()), min=float(v.min()), max=float(v.max()), ratio=float(v.max() / v.min()) if v.min() > 0 else None,
                         theta_min_deg=float(TH_DEG[ok][np.argmin(v)]), theta_max_deg=float(TH_DEG[ok][np.argmax(v)]))
    if ok.all():
        F = np.abs(np.fft.rfft(rc)) ** 2; F[1:] *= 2
        d['fourier_m1plus'] = float(F[1:].sum() / F.sum()); d['fourier_m1'] = float(F[1] / F.sum()); d['fourier_m2'] = float(F[2] / F.sum())
    return d


def circ_diff(a, b):
    return float(abs((a - b + 180) % 360 - 180))


def analyse_map(pre, f):
    """pre: pre-activation (5, 41, nθ) or None; f: post-ReLU map"""
    res = dict(fourier_f=fourier_stats(f), fourier_f_per_pt={str(p): fourier_stats(f[a]) for a, p in enumerate(PT_GRID)}, per_pt={})
    if pre is not None: res['fourier_pre'] = fourier_stats(pre - np.nanmin(pre))
    for a, p in enumerate(PT_GRID):
        rc = contour_pre(pre[a]) if pre is not None else contour_relu(f[a])
        res['per_pt'][str(p)] = contour_stats(rc)
        if pre is not None: res['per_pt'][str(p)]['relu_method_ratio'] = contour_stats(contour_relu(f[a])).get('ratio')
    return res


def polygon(w, nvec, th_rad):
    xh = np.stack([np.cos(th_rad), np.sin(th_rad)], 1)
    return (w[None, :] * relu(xh @ nvec.T)).sum(1)


def polygon_compare(rc, w, nvec, rng, nperm=NPERM):
    th = np.radians(TH_DEG); ok = np.isfinite(rc)

    def pred(wv):
        rho = polygon(wv, nvec, th); sgn = np.sign(np.median(rho[ok])) if ok.any() else 1.0
        p = np.where(rho * sgn > 1e-9, 1.0 / np.abs(rho), np.nan)
        return p

    def score(p):
        m = ok & np.isfinite(p)
        if m.sum() < 0.5 * len(rc) or np.std(p[m]) == 0 or np.std(rc[m]) == 0: return None, None
        return float(np.corrcoef(p[m], rc[m])[0, 1]), float(np.nanmax(p) / np.nanmin(p))
    def m0_of(wv):
        rho = polygon(wv, nvec, th); sgn = np.sign(np.median(rho[ok])) if ok.any() else 1.0
        F = np.abs(np.fft.rfft(relu(sgn * rho))) ** 2; F[1:] *= 2
        return float(F[0] / F.sum()) if F.sum() > 0 else None
    p0 = pred(w); corr0, an0 = score(p0); rho0 = polygon(w, nvec, th)
    rm = float(np.nanmax(rc) / np.nanmin(rc)) if ok.any() and np.nanmin(rc) > 0 else None
    out = dict(pred_corr=corr0, pred_ratio=an0, pred_theta_min_deg=float(TH_DEG[np.nanargmin(p0)]) if np.isfinite(p0).any() else None,
               pred_profile=(p0 / np.nanmean(p0[ok]) * np.nanmean(rc[ok])).tolist() if ok.any() and np.isfinite(p0).any() else None,
               rho_sign_changes=int((np.sign(rho0)[1:] != np.sign(rho0)[:-1]).sum()), pred_m0_angular=m0_of(w))
    pe = pred(np.ones(len(w))); ce, ae = score(pe)
    out['equal_weights'] = dict(corr=ce, ratio=ae, theta_min_deg=float(TH_DEG[np.nanargmin(pe)]) if np.isfinite(pe).any() else None, m0_angular=m0_of(np.ones(len(w))))
    cs, ans = [], []
    for _ in range(nperm):
        c_, a_ = score(pred(w[rng.permutation(len(w))]))
        if c_ is not None: cs.append(c_); ans.append(a_)
    if cs and corr0 is not None:
        out['perm'] = dict(n=len(cs), corr_median=float(np.median(cs)), corr_p=float(np.mean(np.array(cs) >= corr0)), ratio_median=float(np.median(ans)))
        if rm and an0:
            d0 = abs(np.log(an0) - np.log(rm)); out['perm']['ratio_p'] = float(np.mean(np.abs(np.log(ans) - np.log(rm)) <= d0))
    return out


# ------------------------------------------------------------------ Part 4 helper
def sector_stats(dU, wmask, sector, sd, J):
    du, sh = [], []
    for k in range(NSEC):
        sel = wmask & (sector == k)
        du.append(float(dU[sel].mean()) if sel.any() else 0.0); sh.append(float(dU[sel].sum() / 16 / J / sd))
    return du, sh


# ------------------------------------------------------------------ main
def main():
    from interp.jedi_core import Jedi, sample
    from interp.e4_e5_jedi import masks, make_masked
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); np.seterr(all='ignore')
    per = 40 if SMOKE else PER
    x, y, Q = sample(per, seed=SEED); x = np.asarray(x, np.float32); J = len(x)
    real = x[..., 0] > 0; th_deg = np.degrees(np.arctan2(x[..., 2], x[..., 1])) % 360; sector = (th_deg // (360 / NSEC)).astype(int)
    net = Jedi(64); M3, KEEP = masks(net); LO, HI = range_tables(net)
    K1, b1 = net.KB['l1']; W2, b2 = net.KB['l2']; W3, b3 = net.KB['l3']
    nvec = np.stack([K1[1, HP] / net.scale[1], K1[2, HP] / net.scale[2]], 1)              # physical normals (rad⁻¹)
    beta = np.array([b1[j] - (K1[:, j] * net.shift / net.scale).sum() for j in HP])
    ang = np.degrees(np.arctan2(nvec[:, 1], nvec[:, 0])) % 360
    res = dict(N=J, per_class=per, seed=SEED, smoke=SMOKE, grid=dict(pt=PT_GRID, r=R_GRID.tolist(), theta_deg=TH_DEG.tolist(), slots=list(SLOTS)),
               halfplanes={f'a{j}': dict(n_eta=float(nvec[k, 0]), n_phi=float(nvec[k, 1]), beta=float(beta[k]), angle_deg=float(ang[k]),
                                         own_sector=int(ang[k] // (360 / NSEC))) for k, j in enumerate(HP)})
    log(f'E14: {J} jets ({per}/class, seed {SEED}); half-plane normals (rad⁻¹) and angles: ' +
        ', '.join(f"a{j} ({nvec[k,0]:+.0f},{nvec[k,1]:+.0f}) {ang[k]:.0f}°" for k, j in enumerate(HP)))

    # ---------------- reference run
    R0 = fixed(net, x, keep=('g', 'u', 'S', 'logits'))
    call0 = R0['logits'].argmax(1); acc = float((call0 == y).mean()); sdS = R0['S'].std(0)
    live = [int(c) for c in np.flatnonzero(sdS > 1e-9)]; nl = len(live)
    res['acc_fixed'] = acc; res['live_u'] = live; res['sd_S'] = {c: float(sdS[c]) for c in live}
    log(f'fixed-point network: accuracy {acc:.4f}; live u/S units ({nl}): {live}   ({time.time()-t0:.0f}s)')
    U0 = R0['u'][..., live]; del R0['u']
    prog = Program(net)
    MF = make_masked(net, live); Mf = MF(x)
    Pf = prog.run(x, keep=('g', 'u', 'S', 'logits'), live=live)
    accp = float((Pf['logits'].argmax(1) == y).mean())
    res['program'] = dict(acc=accp, calls_changed_vs_fixed=float((Pf['logits'].argmax(1) != call0).mean()),
                          max_abs_diff_vs_make_masked=dict(u=float(np.abs(Pf['u'] - Mf['u']).max()), S=float(np.abs(Pf['S'] - Mf['S']).max()),
                                                           logits=float(np.abs(Pf['logits'] - Mf['logits']).max())))
    log(f"program (float + masks): accuracy {accp:.4f}, calls changed vs fixed {res['program']['calls_changed_vs_fixed']*100:.2f}%; "
        f"max |diff| vs e4_e5_jedi.make_masked: u {res['program']['max_abs_diff_vs_make_masked']['u']:.2e}, S {res['program']['max_abs_diff_vs_make_masked']['S']:.2e}")
    gmf = Pf['g'].mean(0); Gown = Pf['g']; gm_fixed = R0['g'].mean(0)
    Pm = prog.run(x, gov=gmf, keep=('u', 'S', 'logits'), live=live)
    res['program']['acc_g_mean'] = float((Pm['logits'].argmax(1) == y).mean())
    del Mf

    # ---------------- Part 1
    P1 = part1(prog, x, live, KEEP, real)
    res['part1'] = P1
    SRC = [f'a{j}' for j in HP + PTU + CONST]
    log('\nPart 1: variance share (%) of pre_u_c over real particles in the pooled slots | groups HP / pT / const / s-bias / ctx g | HP/(HP+|ctx|) | SD(HP)/SD(ctx) | decomposition max err')
    log('   u   ' + ' '.join(f'{s:>5s}' for s in SRC) + ' |   HP    pT  const  bias   ctx |  frac  sdratio  err')
    for c in live:
        d = P1[c]; g_ = d['groups']
        log(f'  u{c:<3d} ' + ' '.join(f"{100*d['share'][s]:5.0f}" for s in SRC) +
            f" | {100*g_['HP']['share']:5.0f} {100*g_['PT']['share']:5.0f} {100*g_['CONST']['share']:5.0f} {100*g_['bias']['share']:5.0f} {100*g_['ctx']['share']:5.0f} | "
            f"{(d['hp_over_hp_plus_ctx'] if d['hp_over_hp_plus_ctx'] is not None else float('nan')):5.2f} {(d['hp_sd_over_ctx_sd'] if d['hp_sd_over_ctx_sd'] is not None else float('nan')):7.2f}  {d['decomposition_max_err']:.1e}")
    log('Part 1: effective weights ∂u_c/∂a_j averaged over active particles (half-planes), and the s paths (s unit: W3, then W2 of each a_j feeding it)')
    for c in live:
        d = P1[c]
        log(f'  u{c:<3d} ' + ' '.join(f"a{j}:{d['eff_weight_active'][f'a{j}']:+.2f}" for j in HP) + ' | pT ' + ' '.join(f"a{j}:{d['eff_weight_active'][f'a{j}']:+.2f}" for j in PTU) +
            ' | paths ' + '; '.join(f"s{p['s']} (W3 {p['W3']:+.2f}, {p['slots_l3']} slots) <- " + ' '.join(f"{k}:{v:+.2f}" for k, v in p['W2'].items()) for p in d['paths']))
    log(f'   ({time.time()-t0:.0f}s)')

    # ---------------- Part 2
    PT, ETA, PHI = grid(); shp = PT.shape
    maps = {}; P2 = dict(program={}, fixed={}, untrained={}, polygon={})
    for slot in SLOTS:
        pre = prog.particle(slot, PT, ETA, PHI, gmf).reshape(*shp, 64); maps[f'program_pre_slot{slot}'] = pre.astype(np.float32)
        uf, xs = fixed_particle_u(net, slot, PT, ETA, PHI, gm_fixed)
        xn = (xs[:, slot, :].astype(np.float64) - net.shift) / net.scale
        inr = ~(((xn < LO[slot]) | (xn > HI[slot])).any(1)).reshape(shp)
        uf = np.where(inr[..., None], uf.reshape(*shp, 64), np.nan); maps[f'fixed_u_slot{slot}'] = uf.astype(np.float32); maps[f'inrange_slot{slot}'] = inr
        P2['program'][slot] = {}; P2['fixed'][slot] = {}
        for c in live:
            P2['program'][slot][c] = analyse_map(pre[..., c], relu(pre[..., c])); P2['program'][slot][c]['in_window'] = bool(KEEP[slot, c])
            P2['fixed'][slot][c] = analyse_map(None, uf[..., c]); P2['fixed'][slot][c]['in_window'] = bool(KEEP[slot, c])
            P2['fixed'][slot][c]['frac_in_range'] = float(inr.mean())
            fp, ff = P2['program'][slot][c]['per_pt'], P2['fixed'][slot][c]['per_pt']
            P2['fixed'][slot][c]['contour_vs_program'] = {p: dict(max_abs_diff=float(np.nanmax(np.abs(np.array(fp[p]['profile']) - np.array(ff[p]['profile']))))
                                                                   if np.isfinite(np.array(fp[p]['profile'])).any() and np.isfinite(np.array(ff[p]['profile'])).any() else None) for p in fp}
    # untrained control
    netu = Jedi(64, untrained=True, seed=0); progu = Program(netu)
    Ru = fixed(netu, x, keep=('S', 'logits')); live_u = [int(c) for c in np.flatnonzero(Ru['S'].std(0) > 1e-9)]
    gmu = progu.run(x, keep=('g',))['g'].mean(0)
    res['untrained'] = dict(acc=float((Ru['logits'].argmax(1) == y).mean()), live_u=live_u)
    for slot in SLOTS:
        pre = progu.particle(slot, PT, ETA, PHI, gmu).reshape(*shp, 64); maps[f'untrained_pre_slot{slot}'] = pre.astype(np.float32)
        P2['untrained'][slot] = {}
        for c in sorted(set(live_u) | set(live)):
            P2['untrained'][slot][c] = analyse_map(pre[..., c], relu(pre[..., c])); P2['untrained'][slot][c]['live_in_untrained'] = c in live_u
            P2['untrained'][slot][c]['responds'] = bool((relu(pre[..., c]) > 0).any() and (pre[..., c] <= 0).any())
    log(f'\nPart 2: untrained control: accuracy {res["untrained"]["acc"]:.3f}, live S units {live_u}; units with a map that turns on/off at slot 5: '
        f'{[c for c in P2["untrained"][5] if P2["untrained"][5][c]["responds"]]}')
    # polygon comparison (program, slot 5, each pT; primary 3 GeV)
    rng = np.random.default_rng(SEED)
    for c in live:
        w = np.array([P1[c]['eff_weight_active'][f'a{j}'] for j in HP])
        P2['polygon'][c] = {}
        for p in PT_GRID:
            rc = np.array(P2['program'][5][c]['per_pt'][str(p)]['profile'], float)
            if np.isfinite(rc).sum() < 0.9 * len(rc) or np.abs(w).sum() == 0: P2['polygon'][c][str(p)] = None; continue
            P2['polygon'][c][str(p)] = polygon_compare(rc, w, nvec, rng, nperm=NPERM if p == 3.0 else 0)
            P2['polygon'][c][str(p)]['theta_min_meas'] = P2['program'][5][c]['per_pt'][str(p)].get('theta_min_deg')
            P2['polygon'][c][str(p)]['ratio_meas'] = P2['program'][5][c]['per_pt'][str(p)].get('ratio')
            P2['polygon'][c][str(p)]['ratio_meas_over_pred'] = (P2['polygon'][c][str(p)]['ratio_meas'] / P2['polygon'][c][str(p)]['pred_ratio']) \
                if P2['polygon'][c][str(p)].get('pred_ratio') and P2['polygon'][c][str(p)]['ratio_meas'] else None
            P2['polygon'][c][str(p)]['theta_min_diff_deg'] = circ_diff(P2['polygon'][c][str(p)]['theta_min_meas'], P2['polygon'][c][str(p)]['pred_theta_min_deg']) \
                if P2['polygon'][c][str(p)]['theta_min_meas'] is not None and P2['polygon'][c][str(p)]['pred_theta_min_deg'] is not None else None
    res['part2'] = P2
    F1 = np.abs(np.fft.rfft(relu(np.cos(np.radians(TH_DEG))))) ** 2; F1[1:] *= 2
    res['reference_m0_single_halfplane'] = float(F1[0] / F1.sum())
    log('Part 2: program maps at slot 5, g at its mean. Per unit: m=0 power fraction of f (all cells; at 3 GeV) | r_c(θ) at 1/3/10 GeV: mean, max/min, θ_min | '
        'polygon (Part-1 weights): pred max/min, pred θ_min, corr, perm p | equal-weights corr | fixed-point max/min at 3 GeV | untrained')

    def fmt(v, f='{:.2f}'):
        return '  -  ' if v is None else f.format(v)
    for c in live:
        a_ = P2['program'][5][c]; fz = P2['fixed'][5][c]; pg = P2['polygon'][c].get('3.0'); un = P2['untrained'][5][c]
        s = f"  u{c:<3d} m0 {fmt(a_['fourier_f'] and a_['fourier_f']['m0'])} ({fmt(a_['fourier_f_per_pt']['3.0'] and a_['fourier_f_per_pt']['3.0']['m0'])}) |"
        for p in ('1.0', '3.0', '10.0'):
            cc = a_['per_pt'][p]; s += f" {p[:-2]}GeV: {fmt(cc.get('mean'), '{:.3f}')} {fmt(cc.get('ratio'))} {fmt(cc.get('theta_min_deg'), '{:.0f}°')} |"
        if pg: s += f" poly {fmt(pg['pred_ratio'])} {fmt(pg['pred_theta_min_deg'], '{:.0f}°')} corr {fmt(pg['pred_corr'])} p {fmt(pg.get('perm', {}).get('corr_p'), '{:.3f}')} | eq {fmt(pg['equal_weights']['corr'])} |"
        else: s += ' poly - |'
        s += f" fixed {fmt(fz['per_pt']['3.0'].get('ratio'))} (in range {fz['frac_in_range']:.2f}) | untr m0 {fmt(un['fourier_f'] and un['fourier_f']['m0'])} ratio {fmt(un['per_pt']['3.0'].get('ratio'))}"
        log(s)
    log(f'   ({time.time()-t0:.0f}s)')

    # ---------------- Part 3
    P3 = {}; W_, Z_ = 2, 3
    base = dict(fixed_free=R0['S'], fixed_frozen=R0['S'], prog_mean=Pm['S'], prog_own=Pf['S'], prog_free=Pf['S'])
    E = {}
    for dd in DELTAS:
        for name, (de, dp) in DIRS.items(): E[f'{dd}_{name}'] = shift(x, dd * de, dd * dp)
    E['rot45'] = rotate(x, 45.0)
    for ename, xe in E.items():
        ok = in_range(net, xe, LO, HI); n_ok = int(ok.sum())
        Rf = fixed(net, xe, keep=('S', 'logits')); Rz = fixed(net, xe, keep=('S', 'logits'), g_jet=R0['g'])
        Qm = prog.run(xe, gov=gmf, keep=('S', 'logits')); Qo = prog.run(xe, gov=Gown, keep=('S', 'logits')); Qf = prog.run(xe, keep=('S', 'logits'))
        runs = dict(fixed_free=Rf, fixed_frozen=Rz, prog_mean=Qm, prog_own=Qo, prog_free=Qf)
        row = dict(n=n_ok, frac_wrap=float(1 - ok.mean()), units={}, WZ={}, calls_changed={})
        for k, r_ in runs.items():
            c_ = r_['logits'].argmax(1); row['WZ'][k] = float((c_[(y == W_) & ok] == Z_).mean()); row['calls_changed'][k] = float((c_ != call0)[ok].mean())
        for c in live:
            dS = {k: (r_['S'][:, c] - base[k][:, c])[ok] for k, r_ in runs.items()}
            d = {k: dict(mean=float(v.mean()), sem=float(v.std() / np.sqrt(len(v))), mean_sd=float(v.mean() / sdS[c])) for k, v in dS.items()}
            for k in ('prog_mean', 'prog_own'):
                a_, b_ = dS[k], dS['fixed_frozen']
                if a_.std() > 1e-9 and b_.std() > 1e-9:
                    d[k]['corr_vs_fixed_frozen'] = float(np.corrcoef(a_, b_)[0, 1]); d[k]['slope_vs_fixed_frozen'] = float((a_ @ b_) / (a_ @ a_))
            row['units'][c] = d
        P3[ename] = row
        log(f"Part 3 {ename:11s} n {n_ok} wrap {row['frac_wrap']:.4f} | ΔS/sd meas(frozen/free) vs prog(mean/own/free): " +
            ' '.join(f"S{c} {row['units'][c]['fixed_frozen']['mean_sd']:+.3f}/{row['units'][c]['fixed_free']['mean_sd']:+.3f} vs "
                     f"{row['units'][c]['prog_mean']['mean_sd']:+.3f}/{row['units'][c]['prog_own']['mean_sd']:+.3f}/{row['units'][c]['prog_free']['mean_sd']:+.3f}" for c in MAIN if c in row['units']) +
            f" | W→Z fixed {row['WZ']['fixed_free']:.3f} (frozen {row['WZ']['fixed_frozen']:.3f}) prog {row['WZ']['prog_mean']:.3f}/{row['WZ']['prog_own']:.3f}/{row['WZ']['prog_free']:.3f}")
    # summary: even / odd, ordering, slopes
    summ = {}
    for dd in DELTAS:
        sd_ = {}
        for c in MAIN:
            if c not in live: continue
            e = {}
            for k in ('fixed_frozen', 'fixed_free', 'prog_mean', 'prog_own', 'prog_free'):
                v = {name: P3[f'{dd}_{name}']['units'][c][k]['mean'] for name in DIRS}
                oe = {}
                for ax in ('eta', 'phi', 'diag'):
                    ev = 0.5 * (v[f'+{ax}'] + v[f'-{ax}']); od = 0.5 * (v[f'+{ax}'] - v[f'-{ax}'])
                    oe[ax] = dict(even=ev, odd=od, odd_over_even=float(od / ev) if abs(ev) > 1e-12 else None)
                e[k] = dict(by_dir=v, even_odd=oe, spearman_vs_E6v_h5=spearman([v[n] for n in DIRS], [E6V_H5[n] for n in DIRS]),
                            order=[n for n in sorted(DIRS, key=lambda n: -v[n])])
            pm = np.array([P3[f'{dd}_{n}']['units'][c]['prog_mean']['mean'] for n in DIRS]); mf = np.array([P3[f'{dd}_{n}']['units'][c]['fixed_frozen']['mean'] for n in DIRS])
            mfr = np.array([P3[f'{dd}_{n}']['units'][c]['fixed_free']['mean'] for n in DIRS])
            e['spearman_prog_mean_vs_fixed_frozen'] = spearman(pm, mf); e['spearman_prog_mean_vs_fixed_free'] = spearman(pm, mfr)
            e['slope_fixed_frozen_on_prog_mean'] = float((pm @ mf) / (pm @ pm)) if (pm @ pm) > 0 else None
            e['spread_over_mean'] = dict(prog_mean=float((pm.max() - pm.min()) / abs(pm.mean())) if pm.mean() else None, fixed_frozen=float((mf.max() - mf.min()) / abs(mf.mean())) if mf.mean() else None)
            sd_[c] = e
        wz = {k: {n: P3[f'{dd}_{n}']['WZ'][k] for n in DIRS} for k in ('fixed_free', 'fixed_frozen', 'prog_mean', 'prog_own', 'prog_free')}
        sd_['WZ'] = {k: dict(by_dir=v, spearman_vs_E6v_WZ=spearman([v[n] for n in DIRS], [E6V_WZ[n] for n in DIRS]),
                             odd_over_even={ax: float((v[f'+{ax}'] - v[f'-{ax}']) / (v[f'+{ax}'] + v[f'-{ax}'])) if (v[f'+{ax}'] + v[f'-{ax}']) > 0 else None for ax in ('eta', 'phi', 'diag')}) for k, v in wz.items()}
        summ[str(dd)] = sd_
        for c in MAIN:
            if c not in live: continue
            e = sd_[c]
            log(f"Part 3 δ={dd} S{c}: odd/even (η, φ, diag) measured frozen " + ' '.join(fmt(e['fixed_frozen']['even_odd'][ax]['odd_over_even'], '{:+.3f}') for ax in ('eta', 'phi', 'diag')) +
                ' | measured free ' + ' '.join(fmt(e['fixed_free']['even_odd'][ax]['odd_over_even'], '{:+.3f}') for ax in ('eta', 'phi', 'diag')) +
                ' | program mean g ' + ' '.join(fmt(e['prog_mean']['even_odd'][ax]['odd_over_even'], '{:+.3f}') for ax in ('eta', 'phi', 'diag')) +
                ' | program own g ' + ' '.join(fmt(e['prog_own']['even_odd'][ax]['odd_over_even'], '{:+.3f}') for ax in ('eta', 'phi', 'diag')) +
                f" | ordering: Spearman prog vs meas(frozen) {e['spearman_prog_mean_vs_fixed_frozen']:.2f}, prog vs E6v h5 {e['prog_mean']['spearman_vs_E6v_h5']:.2f}, "
                f"meas vs E6v h5 {e['fixed_frozen']['spearman_vs_E6v_h5']:.2f}; slope meas/prog {fmt(e['slope_fixed_frozen_on_prog_mean'])}; order prog {e['prog_mean']['order']}")
        log(f"Part 3 δ={dd} W→Z by direction: fixed " + ' '.join(f"{n} {wz['fixed_free'][n]:.2f}" for n in DIRS) + ' | program(own g) ' + ' '.join(f"{n} {wz['prog_own'][n]:.2f}" for n in DIRS) +
            f" | Spearman vs E6v: fixed {sd_['WZ']['fixed_free']['spearman_vs_E6v_WZ']:.2f}, prog own {sd_['WZ']['prog_own']['spearman_vs_E6v_WZ']:.2f}, prog mean {sd_['WZ']['prog_mean']['spearman_vs_E6v_WZ']:.2f}")
    res['part3'] = P3; res['part3_summary'] = summ
    log(f'   ({time.time()-t0:.0f}s)')

    # ---------------- Part 4
    dead_c = [j for j in range(64) if np.all(K1[:, j] == 0) and b1[j] <= 0]
    dead = int(dead_c[0]) if dead_c else int(np.argmin(np.abs(K1).sum(0) + np.abs(b1)))
    ABL = [(j, 'halfplane') for j in HP] + [(dead, 'dead'), (56, 'pt_ramp')]
    P4 = {}
    for j, kind in ABL:
        Rf = fixed(net, x, keep=('u', 'S', 'logits'), a_zero=j, live=live); Rz = fixed(net, x, keep=('u', 'S', 'logits'), a_zero=j, live=live, g_jet=R0['g'])
        Qm = prog.run(x, gov=gmf, azero=j, keep=('u', 'S', 'logits'), live=live)
        row = dict(kind=kind, calls_changed_free=float((Rf['logits'].argmax(1) != call0).mean()), calls_changed_frozen=float((Rz['logits'].argmax(1) != call0).mean()),
                   calls_changed_prog=float((Qm['logits'].argmax(1) != Pm['logits'].argmax(1)).mean()), acc_free=float((Rf['logits'].argmax(1) == y).mean()), units={})
        if kind == 'halfplane':
            row['angle_deg'] = float(ang[HP.index(j)]); row['own_sector'] = int(row['angle_deg'] // (360 / NSEC)); row['opp_sector'] = (row['own_sector'] + NSEC // 2) % NSEC
        for q, c in enumerate(live):
            wmask = real & KEEP[None, :, c]
            d = {}
            for k, (dU, dS) in dict(free=(Rf['u'][..., q] - U0[..., q], Rf['S'][:, c] - R0['S'][:, c]), frozen=(Rz['u'][..., q] - U0[..., q], Rz['S'][:, c] - R0['S'][:, c]),
                                    prog=(Qm['u'][..., q] - Pm['u'][..., q], Qm['S'][:, c] - Pm['S'][:, c])).items():
                du, sh = sector_stats(dU, wmask, sector, sdS[c], J)
                d[k] = dict(dS_sd=float(dS.mean() / sdS[c]), dS_sd_abs=float(np.abs(dS).mean() / sdS[c]), sector_du=du, sector_share=sh,
                            n_particles_changed=int((np.abs(dU) > 1e-6)[wmask].sum()))
                if kind == 'halfplane':
                    own, opp = row['own_sector'], row['opp_sector']
                    d[k]['own'] = sh[own]; d[k]['opp'] = sh[opp]
                    d[k]['selectivity'] = float(abs(sh[own]) / max(abs(sh[opp]), 1e-12)) if abs(sh[own]) > 0 else None
                    # own half-plane (sectors within ±90° of the normal) vs the opposite half
                    hs = [k_ for k_ in range(NSEC) if circ_diff((k_ + 0.5) * 360 / NSEC, row['angle_deg']) < 90]
                    d[k]['own_half'] = float(sum(sh[k_] for k_ in hs)); d[k]['opp_half'] = float(sum(sh[k_] for k_ in range(NSEC) if k_ not in hs))
                sv = np.array(sh); d[k]['sector_cv'] = float(sv.std() / abs(sv.mean())) if abs(sv.mean()) > 1e-12 else None
            a_, b_ = np.array(d['prog']['sector_share']), np.array(d['frozen']['sector_share'])
            d['corr_sectors_prog_vs_frozen'] = float(np.corrcoef(a_, b_)[0, 1]) if a_.std() > 0 and b_.std() > 0 else None
            row['units'][c] = d
        P4[f'a{j}'] = row
        s = f"Part 4 a{j:<2d} ({kind:9s}" + (f", {row['angle_deg']:.0f}°, own sector {row['own_sector']}" if kind == 'halfplane' else '') + \
            f"): calls changed free {row['calls_changed_free']*100:.2f}% frozen {row['calls_changed_frozen']*100:.2f}% prog {row['calls_changed_prog']*100:.2f}% |"
        for c in (29, 1):
            if c not in live: continue
            d = row['units'][c]
            s += f" S{c}: ΔS/sd {d['frozen']['dS_sd']:+.3f} (free {d['free']['dS_sd']:+.3f}, prog {d['prog']['dS_sd']:+.3f})"
            if kind == 'halfplane': s += f" own {d['frozen']['own']:+.4f} opp {d['frozen']['opp']:+.4f} sel {fmt(d['frozen']['selectivity'], '{:.1f}')} (prog sel {fmt(d['prog']['selectivity'], '{:.1f}')})"
            else: s += f" sector CV {fmt(d['frozen']['sector_cv'])}"
            s += f" corr sectors prog/meas {fmt(d['corr_sectors_prog_vs_frozen'])} |"
        log(s)
        if kind == 'halfplane' and j == 29:
            log('   a29 sector shares of ΔS29/sd (frozen g), sectors 0..11 (centres 15°, 45°, ...): ' + ' '.join(f'{v:+.4f}' for v in row['units'][29]['frozen']['sector_share']) if 29 in live else '')
    res['part4'] = P4; res['dead_unit'] = dead
    sel = [P4[f'a{j}']['units'][29]['frozen']['selectivity'] for j in HP if 29 in live and P4[f'a{j}']['units'][29]['frozen']['selectivity'] is not None]
    res['part4_summary'] = dict(S29_selectivity_min=float(min(sel)) if sel else None, S29_selectivity_median=float(np.median(sel)) if sel else None,
                                S29_selectivity_all={f'a{j}': P4[f'a{j}']['units'][29]['frozen']['selectivity'] for j in HP} if 29 in live else None,
                                S1_selectivity_all={f'a{j}': P4[f'a{j}']['units'][1]['frozen']['selectivity'] for j in HP} if 1 in live else None,
                                dead_max_abs_dS_sd=float(max(abs(P4[f'a{dead}']['units'][c]['free']['dS_sd']) for c in live)),
                                a56_sector_cv={c: P4['a56']['units'][c]['frozen']['sector_cv'] for c in (29, 1) if c in live})
    log(f"Part 4 summary: S29 own/opposite selectivity over the nine half-planes: min {fmt(res['part4_summary']['S29_selectivity_min'], '{:.1f}')}, median {fmt(res['part4_summary']['S29_selectivity_median'], '{:.1f}')}; "
        f"dead unit a{dead} max |ΔS/sd| {res['part4_summary']['dead_max_abs_dS_sd']:.2e}; a56 sector CV S29/S1 {res['part4_summary']['a56_sector_cv']}")
    res['elapsed_s'] = float(time.time() - t0); res['log'] = _LOG
    JSON_PATH.write_text(json.dumps(clean(res))); np.savez_compressed(NPZ_PATH, **maps)
    log(f'wrote {JSON_PATH} and {NPZ_PATH} ({time.time()-t0:.0f}s)')
    try:
        plot(); log(f'wrote {PNG} and {PNG_TAB}')
    except Exception as ex:  # noqa: BLE001
        log(f'plot failed: {ex!r}')


# ------------------------------------------------------------------ figure
def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    R = json.loads(JSON_PATH.read_text()); th = np.radians(R['grid']['theta_deg']); live = R['live_u']
    surf, ink, ink2, grid_c = '#fcfcfb', '#1f1f1e', '#5f5e59', '#e4e3df'
    BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
    hp = R['halfplanes']
    fig = plt.figure(figsize=(16, 10.5), facecolor=surf)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.42, wspace=0.32)
    # (a) polar contours
    for k, c in enumerate((29, 1, 50)):
        ax = fig.add_subplot(gs[0, k], projection='polar'); ax.set_facecolor(surf)
        if str(c) not in R['part2']['program']['5']: continue
        pm = R['part2']['program']['5'][str(c)]['per_pt']['3.0']; fx = R['part2']['fixed']['5'][str(c)]['per_pt']['3.0']
        pg = (R['part2']['polygon'].get(str(c)) or {}).get('3.0'); un = R['part2']['untrained']['5'].get(str(c), {}).get('per_pt', {}).get('3.0')
        prof = np.array([np.nan if v is None else v for v in pm['profile']], float)
        ax.plot(np.r_[th, th[0]], np.r_[prof, prof[0]], color=BLUE, lw=2.2, label='program (float + masks)')
        fpr = np.array([np.nan if v is None else v for v in fx['profile']], float)
        ax.plot(np.r_[th, th[0]], np.r_[fpr, fpr[0]], color=ink, lw=1.2, ls=':', label='fixed-point network')
        if pg and pg.get('pred_profile'):
            pp = np.array([np.nan if v is None else v for v in pg['pred_profile']], float)
            ax.plot(np.r_[th, th[0]], np.r_[pp, pp[0]], color=ORANGE, lw=1.8, ls='--', label='nine-half-plane polygon (Part-1 weights)')
        if un and un.get('profile') and any(v is not None for v in un['profile']):
            up = np.array([np.nan if v is None else v for v in un['profile']], float)
            ax.plot(np.r_[th, th[0]], np.r_[up, up[0]], color=ink2, lw=1.2, alpha=0.7, label='untrained network')
        rmax = 0.4
        for name, h in hp.items():
            a_ = np.radians(h['angle_deg']); ax.plot([a_, a_], [rmax * 0.93, rmax], color=ink2, lw=1.4)
            if k == 0: ax.annotate(name, (a_, rmax * 0.86), ha='center', va='center', fontsize=7.5, color=ink2)
        ax.set_rlim(0, rmax); ax.set_rticks([0.1, 0.2, 0.3]); ax.set_rlabel_position(112); ax.tick_params(labelsize=8, colors=ink2); ax.grid(color=grid_c, lw=0.8)
        ax.set_thetagrids([0, 90, 180, 270], ['+Δη', '+Δφ', '−Δη', '−Δφ'])
        an = f"max/min {pm.get('ratio'):.2f}" if pm.get('ratio') else ''
        if pg and pg.get('pred_ratio'): an += f" (polygon {pg['pred_ratio']:.2f})"
        ax.set_title(f'S{c}: on-contour r_c(θ), 3 GeV particle, slot 5\n{an}', fontsize=10, color=ink, pad=14)
        if k == 0: ax.legend(loc='upper left', bbox_to_anchor=(-0.35, -0.08), frameon=False, fontsize=8, labelcolor=ink, ncol=1)
    # (b) predicted vs measured directional ΔS
    ax = fig.add_subplot(gs[0, 3]); ax.set_facecolor(surf)
    for c, col in ((29, BLUE), (1, ORANGE)):
        if str(c) not in R['part3']['0.03_+eta']['units']: continue
        for dd, mk in (('0.03', 'o'), ('0.06', 's')):
            xs = [R['part3'][f'{dd}_{n}']['units'][str(c)]['prog_mean']['mean_sd'] for n in DIRS]; ys = [R['part3'][f'{dd}_{n}']['units'][str(c)]['fixed_frozen']['mean_sd'] for n in DIRS]
            ax.scatter(xs, ys, s=46, marker=mk, facecolor=surf, edgecolor=col, lw=1.8, zorder=3, label=f'S{c}, δ = {dd}')
            if dd == '0.06':
                for n, x_, y_ in zip(DIRS, xs, ys): ax.annotate(n.replace('eta', 'η').replace('phi', 'φ'), (x_, y_), xytext=(4, 3), textcoords='offset points', fontsize=7.5, color=ink2)
    lim = ax.get_xlim(); lim2 = ax.get_ylim(); lo, hi = min(lim[0], lim2[0]), max(lim[1], lim2[1])
    ax.plot([lo, hi], [lo, hi], color=ink2, lw=1, ls='--'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('ΔS / sd(S) predicted by the maps (program, g at its mean)', fontsize=9, color=ink2); ax.set_ylabel('ΔS / sd(S) measured, fixed-point network (g frozen)', fontsize=9, color=ink2)
    ax.set_title('Rigid translation: six directions, two step sizes', fontsize=10, color=ink, loc='left')
    ax.grid(color=grid_c, lw=0.8); ax.tick_params(labelsize=8, colors=ink2)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc='upper left')
    # (c) heatmaps
    cmap = LinearSegmentedColormap.from_list('div', [ORANGE, '#f3f2ee', BLUE])
    order = sorted([j for j in HP], key=lambda j: hp[f'a{j}']['angle_deg'])
    rows = [f'a{j}' for j in order] + ['a56', f"a{R['dead_unit']}"]
    secs = [f'{int((k + 0.5) * 360 / NSEC)}°' for k in range(NSEC)]
    for col, (c, span) in enumerate(((29, slice(0, 2)), (1, slice(2, 4)))):
        if str(c) not in R['part4']['a29']['units']: continue
        ax = fig.add_subplot(gs[1, span]); ax.set_facecolor(surf)
        Mx = np.array([R['part4'][r_]['units'][str(c)]['frozen']['sector_share'] for r_ in rows])
        v = np.nanmax(np.abs(Mx)) or 1e-6
        im = ax.imshow(Mx, cmap=cmap, norm=TwoSlopeNorm(0, -v, v), aspect='auto')
        for i_, r_ in enumerate(rows):
            if 'own_sector' in R['part4'][r_]: ax.plot(R['part4'][r_]['own_sector'], i_, marker='o', ms=5, color=ink, mfc='none', mew=1.3)
            lab = f"{r_} ({R['part4'][r_]['angle_deg']:.0f}°)" if 'angle_deg' in R['part4'][r_] else (f'{r_} (pT ramp)' if r_ == 'a56' else f'{r_} (dead)')
            rows_lab = lab
            ax.text(-0.6, i_, rows_lab, ha='right', va='center', fontsize=8, color=ink)
        ax.set_yticks([]); ax.set_xticks(range(NSEC)); ax.set_xticklabels(secs, fontsize=8, color=ink2)
        ax.set_xlabel('azimuth sector of the particle (θ = atan2(Δφ, Δη))', fontsize=9, color=ink2)
        ax.set_title(f'S{c}: sector share of ΔS{c} / sd when one l1 statement is set to 0 (fixed point, g frozen); ○ = sector of the half-plane normal', fontsize=9.5, color=ink, loc='left')
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.ax.tick_params(labelsize=7.5, colors=ink2); cb.outline.set_visible(False)
        for s_ in ax.spines.values(): s_.set_visible(False)
    fig.suptitle(f"JEDI-linear 64 p: the radial second-round objects and the nine half-planes — {R['N']} test jets (seed {R['seed']}), fixed-point acc {R['acc_fixed']:.3f}",
                 color=ink2, fontsize=9.5, x=0.01, ha='left', y=0.995)
    fig.savefig(PNG, dpi=140, facecolor=surf, bbox_inches='tight'); plt.close(fig)
    # structure table
    SRC = [f'a{j}' for j in HP + PTU + CONST]; cols = SRC + ['s bias', 'ctx g']
    live_s = [str(c) for c in live]
    M_ = np.array([[R['part1'][c]['share'][s] for s in SRC] + [R['part1'][c]['groups']['bias']['share'], R['part1'][c]['groups']['ctx']['share']] for c in live_s]) * 100
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(live_s) + 2.2), facecolor=surf); ax.set_facecolor(surf)
    seq = LinearSegmentedColormap.from_list('seq', ['#f3f2ee', BLUE])
    ax.imshow(np.clip(np.abs(M_), 0, 100), cmap=seq, vmin=0, vmax=100, aspect='auto')
    for i_ in range(M_.shape[0]):
        for j_ in range(M_.shape[1]):
            v = M_[i_, j_]
            if abs(v) >= 0.5: ax.text(j_, i_, f'{v:.0f}', ha='center', va='center', fontsize=8, color='white' if abs(v) > 60 else ink)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8.5, color=ink); ax.xaxis.tick_top()
    ax.set_yticks(range(len(live_s)))
    ax.set_yticklabels([f"u{c}  (HP/(HP+|ctx|) {R['part1'][c]['hp_over_hp_plus_ctx']:.2f})" if R['part1'][c]['hp_over_hp_plus_ctx'] is not None else f'u{c}' for c in live_s], fontsize=8.5, color=ink)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('Variance share (%) of each l1 statement in the second-round pre-activation u_c (program, real particles in the unit\'s pooled slots; '
                 'half-planes | pT ramps | constants | s biases | context g)', fontsize=9, color=ink, loc='left', pad=28)
    ax.text(-0.5, len(live_s) - 0.2, f"{R['N']} jets (seed {R['seed']}); shares = cov(term, pre_u)/var(pre_u), sum to 100 up to the constant b3", fontsize=7.5, color=ink2, va='top')
    fig.savefig(PNG_TAB, dpi=140, facecolor=surf, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    if '--plot' in sys.argv: plot(); print('wrote', PNG, PNG_TAB)
    else: main()
