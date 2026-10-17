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
        variance shares (cov(term, pre)/var(pre)) over real particles in the unit's pooled slots: the nine half-planes a3 a6 a29
        a30 a36 a40 a43 a46 a57, the pT ramps a15 a21 a32 a56, the constants a4 a50, the s biases, the context g, and the
        angular part of the context (g minus g recomputed with the nine angular P units at their sample mean); G_jc averaged
        over active particles (u_c > 0), all slots and per slot; the s units in between with the sign of each path.
Part 2  polar maps f_c(pT, r, θ) of one particle at slot 5 and 20 (pT 1, 3, 10, 30, 100 GeV; r 0–0.4 in 41 steps; θ 0–360° in
        72 steps): program with g at its mean (primary), fixed-point network with g at its mean (check; grid points outside the
        slot's input quantizer range dropped), untrained network (control).  θ-Fourier power fractions (m = 0, 1, 2, >= 3).
        On-contour r_c(θ) of the program on a fine r grid (0.0005 rad): first change of the on/off state along r; anisotropy
        max/min, θ of the minimum, m = 1 component (direction of its minimum, relative amplitude).
        Polygon model: level set Σ_j w_j ReLU(r n_j·x̂(θ) + β_j) = L of the nine half-plane statements (n_j, β_j the l1 cut in
        rad⁻¹; L one level fitted to the measured contour); weights w from Part 1 (all slots: 'global'; this slot: 'slot'), the
        local gradient at the contour ('local', exactness check), nine fitted weights ('fitted', NNLS); controls: global weights
        permuted among the nine, equal weights, nine fitted weights on random directions and on the nine directions rotated.
Part 3  rigid translation δ ∈ {0.03, 0.06} along ±η, ±φ, ±diag (real particles only; padded slots untouched), 45° rotation:
        ΔS_c predicted by the maps (program, g at the sample mean, = Σ over the jet's particles of f_c) vs measured on the
        fixed-point network (g free; g frozen at the unedited jet's g); also the program with g at the jet's own g and free.
        Mass read-out: ridge of log m (invariant mass from the particles) on the fixed-point network's h5 (fit on half the jets);
        Δlog m̂ measured vs predicted (S_own + ΔS_maps through the program head).  Counterfactual maps (program, g at mean):
        point-symmetrized ½[f(x)+f(−x)], even in η ½[f(η,φ)+f(−η,φ)], even in φ, θ-averaged over 8 rotations (radial); each
        through the program head: Δlog m̂ and W→Z by direction; even / odd parts in δ per axis.
Part 4  causal: a_j → 0 for all particles (override of stage a on the fixed-point network; g free and g frozen) for the nine half-
        planes, a dead l1 unit and the pT ramp a56: per azimuth sector of the particle (12 × 30°) the mean change of u_c per
        particle and the sector's share of ΔS_c / sd(S_c); % calls changed; program prediction (g at its mean); own-sector /
        opposite-sector and own-half (sectors within 90° of the normal n_j) fractions.  With g frozen the change of a particle
        off the half-plane is zero by construction (a_j = 0 there); with g free it can change through P_j → g.
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
R_FINE = np.linspace(0.0, 0.4, 801)
TH_DEG = np.arange(0, 360, 15 if SMOKE else 5)
SLOTS = (5, 20)
DIRS = {'+eta': (1.0, 0.0), '-eta': (-1.0, 0.0), '+phi': (0.0, 1.0), '-phi': (0.0, -1.0),
        '+diag': (2 ** -0.5, 2 ** -0.5), '-diag': (-2 ** -0.5, -2 ** -0.5)}
AXES = ('eta', 'phi', 'diag')
DELTAS = (0.03, 0.06)
E6V_H5 = {'+eta': .085, '-eta': .117, '+phi': .087, '-phi': .110, '+diag': .078, '-diag': .118}   # E6v item 4, δ = 0.03, h5
E6V_WZ = {'+eta': .28, '-eta': .62, '+phi': .32, '-phi': .54, '+diag': .24, '-diag': .64}
NSEC = 12
NPERM = 30 if SMOKE else 200
NCTRL = 20 if SMOKE else 200
CTRL_PT = (3.0, 10.0)
CF = ('maps', 'maps_point_even', 'maps_even_in_eta', 'maps_even_in_phi', 'maps_radial')
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


def fmt(v, f='{:.2f}'):
    return '  -  ' if v is None else f.format(v)


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
        P = (a * self.mqs).sum(1) * self.sc[0]
        return P, self.g_from_P(P)

    def g_from_P(self, P):
        Wd, bd = self.W['dd']
        return relu((P * self.mdd) @ Wd + bd)

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

    def run(self, x, gov=None, azero=None, keep=('S', 'logits'), live=None, batch=1250):
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

    def grad_a(self, slot, pt, eta, phi, c):
        """G_jc = ∂pre_u_c/∂a_j at single-particle points (N, 64 j)"""
        x3 = np.stack([np.asarray(pt, np.float32).ravel() * np.ones(np.size(eta), np.float32), np.asarray(eta, np.float32).ravel(),
                       np.asarray(phi, np.float32).ravel()], 1)
        slots = np.full(len(x3), slot); a = self.a_of(self.xn(x3), slots); z2 = self.z2_of(a, slots)
        W2, _ = self.W['l2']; W3, _ = self.W['l3']
        D = (self.m3[slot] * self.mas[slot])[None] * (z2 > 0) * W3[:, c][None]
        return (D @ W2.T) * self.ml2[slot][None]


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


def mirror(x, col):
    """col 1: Δη → −Δη; col 2: Δφ → −Δφ (real particles only)"""
    e = x.copy(); rl = e[..., 0] > 0
    e[..., col] = np.where(rl, -x[..., col], x[..., col]).astype(np.float32); return e


def inv_mass(x):
    pt, eta, phi = (np.asarray(x[..., k], np.float64) for k in range(3)); rl = pt > 0
    E = (pt * np.cosh(eta) * rl).sum(1); px = (pt * np.cos(phi) * rl).sum(1); py = (pt * np.sin(phi) * rl).sum(1); pz = (pt * np.sinh(eta) * rl).sum(1)
    return np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0))


def ridge(X, t, lam=1.0):
    X = np.asarray(X, np.float64); mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; tm = t.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (t - tm))
    return lambda X_: ((np.asarray(X_, np.float64) - mu) / sd) @ w + tm


# ------------------------------------------------------------------ Part 1: decomposition
def part1(prog, x, live, KEEP, real, Pmean, batch=1000):
    W2, b2 = prog.W['l2']; W3, b3 = prog.W['l3']
    GRP = {'HP': HP, 'PT': PTU, 'CONST': CONST}
    TERMS = list(GRP) + ['bias', 'ctx', 'ctx_ang']
    A = {c: dict(n=0, sp=0.0, spp=0.0, sA=np.zeros(64), sAp=np.zeros(64), nact=0, sG=np.zeros(64), sGs=np.zeros((64, 64)), nas=np.zeros(64),
                 sM=np.zeros((64, 64)), err=0.0, grp={k: dict(s=0.0, ss=0.0, sp=0.0) for k in TERMS}) for c in live}
    for i0 in range(0, len(x), batch):
        xb = prog.xn(x[i0:i0 + batch]); a = prog.a_of(xb); z2 = prog.z2_of(a); s = relu(z2); act2 = (z2 > 0).astype(np.float32)
        P, g = prog.g_of(a); pre = prog.pre_of(s, g); rb = real[i0:i0 + batch]
        Pn = P.copy(); Pn[:, HP] = Pmean[HP]; g_na = prog.g_from_P(Pn)
        for c in live:
            D = (prog.m3 * prog.mas)[None] * act2 * W3[:, c]                 # (B, 64, 64k)
            G = np.einsum('bik,jk->bij', D, W2, optimize=True) * prog.ml2[None]   # (B, 64, 64j)  ∂pre_c/∂a_j
            Aj = a * G; bs = D @ b2
            ctx = (g[:, None, :] * prog.mag * prog.m3) @ W3[:, c]; ctx_na = (g_na[:, None, :] * prog.mag * prog.m3) @ W3[:, c]
            p = pre[..., c]
            A[c]['err'] = max(A[c]['err'], float(np.abs(Aj.sum(-1) + bs + ctx + b3[c] - p).max()))
            w = rb & KEEP[None, :, c]; pw = p[w].astype(np.float64); Aw = Aj[w].astype(np.float64)
            A[c]['n'] += len(pw); A[c]['sp'] += pw.sum(); A[c]['spp'] += (pw ** 2).sum()
            A[c]['sA'] += Aw.sum(0); A[c]['sAp'] += (Aw * pw[:, None]).sum(0)
            terms = {k: Aw[:, idx].sum(1) for k, idx in GRP.items()}
            terms['bias'] = bs[w].astype(np.float64); terms['ctx'] = ctx[w].astype(np.float64); terms['ctx_ang'] = (ctx - ctx_na)[w].astype(np.float64)
            for k, v in terms.items():
                A[c]['grp'][k]['s'] += v.sum(); A[c]['grp'][k]['ss'] += (v ** 2).sum(); A[c]['grp'][k]['sp'] += (v * pw).sum()
            act = w & (p > 0); A[c]['nact'] += int(act.sum()); A[c]['sG'] += G[act].astype(np.float64).sum(0)
            A[c]['sGs'] += np.einsum('bi,bij->ij', act.astype(np.float32), G, optimize=True).astype(np.float64); A[c]['nas'] += act.sum(0)
            A[c]['sM'] += np.einsum('bik,bi,ij->kj', D, act.astype(np.float32), prog.ml2, optimize=True).astype(np.float64) * W2.T
    res = {}
    for c in live:
        d = A[c]; n = max(d['n'], 1); mp = d['sp'] / n; var = d['spp'] / n - mp ** 2
        share = (d['sAp'] / n - (d['sA'] / n) * mp) / var if var > 0 else np.zeros(64)
        grp = {}
        for k, v in d['grp'].items():
            m_ = v['s'] / n; grp[k] = dict(share=float((v['sp'] / n - m_ * mp) / var) if var > 0 else 0.0, sd=float(np.sqrt(max(v['ss'] / n - m_ ** 2, 0))), mean=float(m_))
        Gbar = d['sG'] / max(d['nact'], 1); M = d['sM'] / max(d['nact'], 1)
        hp, ca = grp['HP']['share'], grp['ctx_ang']['share']
        paths = []
        for k in np.flatnonzero(np.abs(W3[:, c]) > 0):
            feeds = [(int(j), float(M[k, j])) for j in HP + PTU + CONST if abs(M[k, j]) > 1e-6]
            paths.append(dict(s=int(k), W3=float(W3[k, c]), b2=float(b2[k]), slots_l3=int(prog.m3[:, k].sum()), eff_weights={f'a{j}': w_ for j, w_ in feeds},
                              W2={f'a{j}': float(W2[j, k]) for j in HP + PTU + CONST if abs(W2[j, k]) > 0}))
        res[int(c)] = dict(n_particles=int(d['n']), n_active=int(d['nact']), sd_pre=float(np.sqrt(max(var, 0))), mean_pre=float(mp), decomposition_max_err=d['err'],
                           share={f'a{j}': float(share[j]) for j in HP + PTU + CONST}, groups=grp,
                           hp_over_hp_plus_ctx_ang=float(hp / (hp + abs(ca))) if (hp + abs(ca)) > 1e-12 else None,
                           hp_sd_over_ctx_ang_sd=float(grp['HP']['sd'] / grp['ctx_ang']['sd']) if grp['ctx_ang']['sd'] > 0 else None,
                           eff_weight_active={f'a{j}': float(Gbar[j]) for j in HP + PTU + CONST},
                           eff_weight_slot={str(sl): {f'a{j}': float(d['sGs'][sl, j] / max(d['nas'][sl], 1)) for j in HP} for sl in SLOTS},
                           n_active_slot={str(sl): int(d['nas'][sl]) for sl in SLOTS}, paths=paths, b3=float(b3[c]))
    return res


# ------------------------------------------------------------------ Part 2: maps, contours, Fourier, polygon
def grid():
    PT, RR, TH = np.meshgrid(np.array(PT_GRID), R_GRID, np.radians(TH_DEG), indexing='ij')
    return PT, RR * np.cos(TH), RR * np.sin(TH)


def fourier_stats(f):
    """θ-power fractions of a map (..., nθ) summed over the cells that respond; m = 0, 1, 2, >= 3"""
    f = np.where(np.isfinite(f), f, 0.0)
    F = np.fft.rfft(f, axis=-1); Pw = np.abs(F) ** 2; Pw[..., 1:] *= 2
    if f.shape[-1] % 2 == 0: Pw[..., -1] /= 2
    cells = (f > 0).any(-1)
    if not cells.any(): return None
    tot = Pw[cells].sum(0); T = tot.sum()
    return dict(m0=float(tot[0] / T), m1=float(tot[1] / T), m2=float(tot[2] / T), m3plus=float(tot[3:].sum() / T), n_cells=int(cells.sum()))


def first_cross(v, r):
    """v (nr, nθ): first r at which the on/off state (v > 0) differs from its state at r[0], linear interpolation; NaN if none.
    Also the number of state changes along r per θ."""
    on = v > 0; ch = on[1:] != on[:1]; any_ = ch.any(0)
    k = np.argmax(ch, 0) + 1; idx = np.arange(v.shape[1])
    v1, v0 = v[k, idx], v[k - 1, idx]; den = v1 - v0
    t = np.clip(np.where(np.abs(den) > 0, -v0 / np.where(np.abs(den) > 0, den, 1.0), 1.0), 0, 1)
    rc = np.where(any_, r[k - 1] + t * (r[k] - r[k - 1]), np.nan)
    return rc, (on[1:] != on[:-1]).sum(0)


def contour_relu(f, r=R_GRID):
    """on/off boundary from a post-ReLU map (fixed-point network): first (outer) or last (inner) active point, refined by the slope"""
    rc = np.full(f.shape[1], np.nan); dr = r[1] - r[0]
    for t in range(f.shape[1]):
        v = f[:, t]; ok = np.isfinite(v)
        if ok.sum() < 3: continue
        on = (v > 0) & ok; idx = np.flatnonzero(on)
        if len(idx) == 0 or len(idx) == ok.sum(): continue
        if not on[np.flatnonzero(ok)[0]]:
            k = idx[0]
            if k + 1 < len(v) and ok[k + 1] and v[k + 1] > v[k]: rc[t] = np.clip(r[k] - v[k] * dr / (v[k + 1] - v[k]), r[k] - dr, r[k])
            else: rc[t] = r[k] - dr / 2
        else:
            k = idx[-1]
            if k >= 1 and ok[k - 1] and v[k - 1] > v[k]: rc[t] = np.clip(r[k] + v[k] * dr / (v[k - 1] - v[k]), r[k], r[k] + dr)
            else: rc[t] = r[k] + dr / 2
    return rc


def contour_stats(rc):
    ok = np.isfinite(rc); d = dict(n_finite=int(ok.sum()), profile=rc.tolist())
    if ok.sum() < 0.9 * len(rc): return d
    v = rc[ok]; th = np.radians(TH_DEG[ok])
    d.update(mean=float(v.mean()), min=float(v.min()), max=float(v.max()), ratio=float(v.max() / v.min()) if v.min() > 0 else None,
             theta_min_deg=float(TH_DEG[ok][np.argmin(v)]), theta_max_deg=float(TH_DEG[ok][np.argmax(v)]))
    F1 = np.mean(v * np.exp(-1j * th)); d['m1_rel_amp'] = float(2 * abs(F1) / v.mean()); d['m1_min_dir_deg'] = float((np.degrees(np.angle(F1)) + 180) % 360)
    if ok.all():
        F = np.abs(np.fft.rfft(rc - rc.mean())) ** 2; F[1:] *= 2; T = F.sum()
        if T > 0: d.update(ac_m1=float(F[1] / T), ac_m2=float(F[2] / T), ac_m3=float(F[3] / T), ac_m4plus=float(F[4:].sum() / T))
    return d


def circ_diff(a, b):
    return float(abs((a - b + 180) % 360 - 180))


def poly_rc(w, nvec, beta, L, th, r=R_FINE):
    """contour of the polygon model Σ_j w_j ReLU(r n_j·x̂ + β_j) = L along each θ (first crossing from r = 0)"""
    proj = np.stack([np.cos(th), np.sin(th)], 1) @ nvec.T                              # (nθ, 9)
    v = relu(r[:, None, None] * proj[None] + beta[None, None, :]) @ w - L
    return first_cross(v, r)[0]


def a_at(rc, th, nvec, beta):
    return relu(rc[:, None] * (np.stack([np.cos(th), np.sin(th)], 1) @ nvec.T) + beta[None, :])


def nnls_fit(A):
    try:
        from scipy.optimize import nnls
        return nnls(A, np.ones(len(A)))[0]
    except Exception:  # noqa: BLE001
        return np.linalg.lstsq(A, np.ones(len(A)), rcond=None)[0]


def shape_metrics(p, rc):
    m = np.isfinite(p) & np.isfinite(rc)
    if m.sum() < 0.9 * len(rc): return dict(n=int(m.sum()))
    d = dict(n=int(m.sum()), rms_rel=float(np.sqrt(np.mean(((p[m] - rc[m]) / rc[m]) ** 2))),
             corr=float(np.corrcoef(p[m], rc[m])[0, 1]) if p[m].std() > 0 and rc[m].std() > 0 else None,
             ratio=float(p[m].max() / p[m].min()) if p[m].min() > 0 else None, theta_min_deg=float(TH_DEG[m][np.argmin(p[m])]))
    th = np.radians(TH_DEG[m]); F1 = np.mean(p[m] * np.exp(-1j * th))
    d['m1_min_dir_deg'] = float((np.degrees(np.angle(F1)) + 180) % 360); d['m1_rel_amp'] = float(2 * abs(F1) / p[m].mean())
    return d


def polygon_compare(rc, W, nvec, beta, rng, controls):
    """W: {name: weight vector (9,)}; one level L per model fitted to the measured contour; returns metrics per model"""
    th = np.radians(TH_DEG); ok = np.isfinite(rc); Ac = a_at(np.where(ok, rc, 0), th, nvec, beta)[ok]
    out = {}

    def model(w):
        if np.abs(w).sum() == 0: return None
        L = float(np.median(Ac @ w)); return poly_rc(w, nvec, beta, L, th)
    for name, w in W.items():
        p = model(np.asarray(w, float))
        out[name] = None if p is None else dict(shape_metrics(p, rc), profile=p.tolist(), w=list(map(float, w)))
    wf = nnls_fit(Ac); pf = poly_rc(wf, nvec, beta, 1.0, th); out['fitted'] = dict(shape_metrics(pf, rc), profile=pf.tolist(), w=list(map(float, wf)))
    if not controls: return out
    wg = np.asarray(W['global'], float); base = out['global']['rms_rel'] if out.get('global') and 'rms_rel' in out['global'] else None
    perm = []
    for _ in range(NPERM):
        p = model(wg[rng.permutation(len(wg))]); sm = shape_metrics(p, rc) if p is not None else {}
        if 'rms_rel' in sm: perm.append(sm['rms_rel'])
    out['perm_global'] = dict(n=len(perm), rms_rel_median=float(np.median(perm)) if perm else None,
                              p_le=float(np.mean(np.array(perm) <= base)) if (perm and base is not None) else None)
    mag = np.linalg.norm(nvec, axis=1); ang = np.arctan2(nvec[:, 1], nvec[:, 0]); fbase = out['fitted'].get('rms_rel')
    for kind in ('random_dirs', 'rotated_dirs'):
        vals = []
        for _ in range(NCTRL):
            a2 = rng.uniform(0, 2 * np.pi, len(ang)) if kind == 'random_dirs' else ang + rng.uniform(np.radians(10), np.radians(350))
            nv = np.stack([mag * np.cos(a2), mag * np.sin(a2)], 1); A2 = a_at(np.where(ok, rc, 0), th, nv, beta)[ok]
            p = poly_rc(nnls_fit(A2), nv, beta, 1.0, th); sm = shape_metrics(p, rc)
            vals.append(sm.get('rms_rel', np.inf))
        vals = np.array(vals)
        out[kind] = dict(n=len(vals), rms_rel_median=float(np.median(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else None,
                         frac_incomplete=float(np.mean(~np.isfinite(vals))), p_le=float(np.mean(vals <= fbase)) if fbase is not None else None)
    return out


# ------------------------------------------------------------------ Part 4 helpers
def sector_stats(dU, wmask, sector, sd, J):
    du, sh = [], []
    for k in range(NSEC):
        sel = wmask & (sector == k)
        du.append(float(dU[sel].mean()) if sel.any() else 0.0); sh.append(float(dU[sel].sum() / 16 / J / sd))
    return du, sh


def sel_metrics(sh, own, opp, half):
    sh = np.asarray(sh, float); a = np.abs(sh); tot = a.sum(); tol = 1e-7 + 1e-6 * a.max()
    d = dict(own=float(sh[own]), opp=float(sh[opp]), total_abs=float(tot), argmax_sector=int(np.argmax(a)) if tot > 0 else None,
             own_half_frac=float(a[half].sum() / tot) if tot > 0 else None, opp_zero=bool(a[opp] <= tol))
    d['own_over_opp'] = float(a[own] / a[opp]) if (a[opp] > tol and a[own] > 0) else None
    return d


# ------------------------------------------------------------------ main
def main():
    from interp.jedi_core import Jedi, sample
    from interp.e4_e5_jedi import masks, make_masked
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); np.seterr(all='ignore')
    per = 40 if SMOKE else PER
    x, y, Q = sample(per, seed=SEED); x = np.asarray(x, np.float32); J = len(x); del Q
    log(f'E14: sample loaded ({J} jets, {per}/class, seed {SEED}) ({time.time()-t0:.0f}s)')
    real = x[..., 0] > 0; th_deg = np.degrees(np.arctan2(x[..., 2], x[..., 1])) % 360; sector = (th_deg // (360 / NSEC)).astype(int)
    net = Jedi(64); M3, KEEP = masks(net); LO, HI = range_tables(net)
    K1, b1 = net.KB['l1']; W2, b2 = net.KB['l2']; W3, b3 = net.KB['l3']
    nvec = np.stack([K1[1, HP] / net.scale[1], K1[2, HP] / net.scale[2]], 1)              # physical normals (rad⁻¹)
    beta = np.array([b1[j] - (K1[:, j] * net.shift / net.scale).sum() for j in HP])      # a_j = ReLU(n_j·x + β_j)
    ang = np.degrees(np.arctan2(nvec[:, 1], nvec[:, 0])) % 360
    res = dict(N=J, per_class=per, seed=SEED, smoke=SMOKE, grid=dict(pt=PT_GRID, r=R_GRID.tolist(), r_fine_step=float(R_FINE[1]), theta_deg=TH_DEG.tolist(), slots=list(SLOTS)),
               halfplanes={f'a{j}': dict(n_eta=float(nvec[k, 0]), n_phi=float(nvec[k, 1]), beta=float(beta[k]), angle_deg=float(ang[k]),
                                         offset_rad=float(-beta[k] / np.linalg.norm(nvec[k])), own_sector=int(ang[k] // (360 / NSEC))) for k, j in enumerate(HP)})
    log('half-plane statements a_j = ReLU(n_j·(Δη,Δφ) + β_j), n_j in rad⁻¹, normal angle, cut offset from the axis: ' +
        ', '.join(f"a{j} ({nvec[k,0]:+.0f},{nvec[k,1]:+.0f}) {ang[k]:.0f}° {-beta[k]/np.linalg.norm(nvec[k]):+.4f} rad" for k, j in enumerate(HP)))

    # ---------------- reference run
    R0 = fixed(net, x, keep=('g', 'S', 'logits', 'h5'))
    call0 = R0['logits'].argmax(1); acc = float((call0 == y).mean()); sdS = R0['S'].std(0)
    live = [int(c) for c in np.flatnonzero(sdS > 1e-9)]; nl = len(live)
    U0 = fixed(net, x, keep=('u',), live=live)['u']
    res['acc_fixed'] = acc; res['live_u'] = live; res['sd_S'] = {c: float(sdS[c]) for c in live}
    log(f'fixed-point network: accuracy {acc:.4f}; live u/S units ({nl}): {live}   ({time.time()-t0:.0f}s)')
    prog = Program(net)
    MF = make_masked(net, live); Mf = MF(x)
    Pf = prog.run(x, keep=('P', 'g', 'u', 'S', 'logits', 'h5'), live=live)
    accp = float((Pf['logits'].argmax(1) == y).mean())
    res['program'] = dict(acc=accp, calls_changed_vs_fixed=float((Pf['logits'].argmax(1) != call0).mean()),
                          max_abs_diff_vs_make_masked=dict(u=float(np.abs(Pf['u'] - Mf['u']).max()), S=float(np.abs(Pf['S'] - Mf['S']).max()),
                                                           logits=float(np.abs(Pf['logits'] - Mf['logits']).max())))
    del Mf
    log(f"program (float + masks): accuracy {accp:.4f}, calls changed vs fixed {res['program']['calls_changed_vs_fixed']*100:.2f}%; "
        f"max |diff| vs e4_e5_jedi.make_masked: u {res['program']['max_abs_diff_vs_make_masked']['u']:.2e}, S {res['program']['max_abs_diff_vs_make_masked']['S']:.2e}")
    gmf = Pf['g'].mean(0); Gown = Pf['g']; gm_fixed = R0['g'].mean(0); Pmean = Pf['P'].mean(0); del Pf['P'], Pf['u']
    Pm = prog.run(x, gov=gmf, keep=('u', 'S', 'logits'), live=live)
    res['program']['acc_g_mean'] = float((Pm['logits'].argmax(1) == y).mean())
    log(f"program with g at its sample mean: accuracy {res['program']['acc_g_mean']:.4f}   ({time.time()-t0:.0f}s)")

    # ---------------- Part 1
    P1 = part1(prog, x, live, KEEP, real, Pmean)
    res['part1'] = P1
    SRC = [f'a{j}' for j in HP + PTU + CONST]
    log('\nPart 1: variance share (%) of pre_u_c over real particles in the pooled slots | groups HP / pT / const / s-bias / ctx g / angular part of ctx | '
        'HP/(HP+|ctx_ang|) | SD(HP)/SD(ctx_ang) | decomposition max err')
    log('   u   ' + ' '.join(f'{s:>5s}' for s in SRC) + ' |   HP    pT  const  bias   ctx  cang |  frac  sdratio  err')
    for c in live:
        d = P1[c]; g_ = d['groups']
        log(f'  u{c:<3d} ' + ' '.join(f"{100*d['share'][s]:5.0f}" for s in SRC) +
            f" | {100*g_['HP']['share']:5.0f} {100*g_['PT']['share']:5.0f} {100*g_['CONST']['share']:5.0f} {100*g_['bias']['share']:5.0f} {100*g_['ctx']['share']:5.0f} {100*g_['ctx_ang']['share']:5.1f} | "
            f"{fmt(d['hp_over_hp_plus_ctx_ang'])} {fmt(d['hp_sd_over_ctx_ang_sd'], '{:7.1f}')}  {d['decomposition_max_err']:.1e}")
    log('Part 1: effective weights ∂u_c/∂a_j averaged over active particles (all slots), and the s paths (s unit: W3, slots, then W2 of each a_j feeding it)')
    for c in live:
        d = P1[c]
        log(f'  u{c:<3d} ' + ' '.join(f"a{j}:{d['eff_weight_active'][f'a{j}']:+.2f}" for j in HP) + ' | pT ' + ' '.join(f"a{j}:{d['eff_weight_active'][f'a{j}']:+.2f}" for j in PTU) +
            ' | paths ' + '; '.join(f"s{p['s']} (W3 {p['W3']:+.2f}, {p['slots_l3']} slots) <- " + ' '.join(f"{k}:{v:+.2f}" for k, v in p['W2'].items()) for p in d['paths']))
    log(f'   ({time.time()-t0:.0f}s)')

    # ---------------- Part 2
    PT, ETA, PHI = grid(); shp = PT.shape; th = np.radians(TH_DEG)
    RRf, TTf = np.meshgrid(R_FINE, th, indexing='ij'); EF, PF = (RRf * np.cos(TTf)).ravel(), (RRf * np.sin(TTf)).ravel()
    maps = {}; P2 = dict(program={}, fixed={}, untrained={}, polygon={}); RC = {}
    for slot in SLOTS:
        pre = prog.particle(slot, PT, ETA, PHI, gmf).reshape(*shp, 64); maps[f'program_pre_slot{slot}'] = pre[..., live].astype(np.float32)
        uf, xs = fixed_particle_u(net, slot, PT, ETA, PHI, gm_fixed)
        xn = (xs[:, slot, :].astype(np.float64) - net.shift) / net.scale
        inr = ~(((xn < LO[slot]) | (xn > HI[slot])).any(1)).reshape(shp)
        uf = np.where(inr[..., None], uf.reshape(*shp, 64), np.nan); maps[f'fixed_u_slot{slot}'] = uf[..., live].astype(np.float32); maps[f'inrange_slot{slot}'] = inr
        P2['program'][slot] = {c: dict(fourier_f=fourier_stats(relu(pre[..., c])), fourier_f_per_pt={str(p): fourier_stats(relu(pre[a_, ..., c])) for a_, p in enumerate(PT_GRID)},
                                       in_window=bool(KEEP[slot, c]), per_pt={}) for c in live}
        P2['fixed'][slot] = {c: dict(fourier_f=fourier_stats(uf[..., c]), in_window=bool(KEEP[slot, c]), frac_in_range=float(inr.mean()),
                                     per_pt={str(p): contour_stats(contour_relu(uf[a_, ..., c])) for a_, p in enumerate(PT_GRID)}) for c in live}
        RC[slot] = {c: {} for c in live}; rcs = np.full((len(PT_GRID), len(th), nl), np.nan, np.float32)
        for a_, p in enumerate(PT_GRID):
            pf = prog.particle(slot, np.full(EF.size, p), EF, PF, gmf)[:, live].reshape(len(R_FINE), len(th), nl)
            for q, c in enumerate(live):
                rc, nx = first_cross(pf[..., q], R_FINE); RC[slot][c][p] = rc; rcs[a_, :, q] = rc
                st = contour_stats(rc); st['n_cross_max'] = int(nx.max()); st['on_at_r0'] = bool(pf[0, 0, q] > 0)
                P2['program'][slot][c]['per_pt'][str(p)] = st
                fz = P2['fixed'][slot][c]['per_pt'][str(p)]
                if 'mean' in fz and 'mean' in st: fz['vs_program_max_abs_diff'] = float(np.nanmax(np.abs(np.array(fz['profile'], float) - rc)))
        maps[f'program_rc_fine_slot{slot}'] = rcs
        log(f'Part 2: slot {slot} maps and fine contours done ({time.time()-t0:.0f}s)')
    # untrained control
    netu = Jedi(64, untrained=True, seed=0); progu = Program(netu)
    Ru = fixed(netu, x, keep=('S', 'logits')); live_un = [int(c) for c in np.flatnonzero(Ru['S'].std(0) > 1e-9)]
    gmu = progu.run(x, keep=('g',))['g'].mean(0)
    res['untrained'] = dict(acc=float((Ru['logits'].argmax(1) == y).mean()), live_u=live_un)
    for slot in SLOTS:
        pre = progu.particle(slot, PT, ETA, PHI, gmu).reshape(*shp, 64); maps[f'untrained_pre_slot{slot}'] = pre[..., sorted(set(live_un) | set(live))].astype(np.float32)
        P2['untrained'][slot] = {}
        for c in sorted(set(live_un) | set(live)):
            P2['untrained'][slot][c] = dict(fourier_f=fourier_stats(relu(pre[..., c])), live_in_untrained=c in live_un,
                                            responds=bool((pre[..., c] > 0).any() and (pre[..., c] <= 0).any()),
                                            per_pt={str(p): contour_stats(first_cross(pre[a_, ..., c], R_GRID)[0]) for a_, p in enumerate(PT_GRID)})
    del Ru
    resp = {slot: [c for c in P2['untrained'][slot] if P2['untrained'][slot][c]['responds']] for slot in SLOTS}
    log(f'Part 2: untrained control: accuracy {res["untrained"]["acc"]:.3f}, live S units {live_un}; units whose map turns on/off within the grid: '
        + '; '.join(f'slot {s}: {v}' for s, v in resp.items()))
    # polygon comparison
    rng = np.random.default_rng(SEED)
    for slot in SLOTS:
        P2['polygon'][slot] = {}
        for c in live:
            wg = np.array([P1[c]['eff_weight_active'][f'a{j}'] for j in HP]); ws = np.array([P1[c]['eff_weight_slot'][str(slot)][f'a{j}'] for j in HP])
            if np.abs(wg).sum() == 0 and np.abs(ws).sum() == 0: continue
            P2['polygon'][slot][c] = {}
            for p in PT_GRID:
                rc = RC[slot][c][p]; ok = np.isfinite(rc)
                if ok.mean() < 0.9: P2['polygon'][slot][c][str(p)] = None; continue
                Gl = prog.grad_a(slot, p, rc[ok] * np.cos(th[ok]), rc[ok] * np.sin(th[ok]), c)[:, HP].mean(0)
                sg = np.sign(wg.sum()) if wg.sum() != 0 else 1.0
                W = dict(global_=wg, slot=ws, local=Gl, equal_all9=sg * np.ones(len(HP)))
                W = {('global' if k == 'global_' else k): v for k, v in W.items()}
                pc = polygon_compare(rc, W, nvec, beta, rng, controls=(slot == 5 and p in CTRL_PT))
                pc['measured'] = {k: P2['program'][slot][c]['per_pt'][str(p)].get(k) for k in ('ratio', 'theta_min_deg', 'm1_min_dir_deg', 'm1_rel_amp', 'mean')}
                for k in ('global', 'slot', 'local', 'fitted'):
                    if pc.get(k) and pc[k].get('ratio') and pc['measured']['ratio']:
                        pc[k]['excess_ratio_pred_over_meas'] = float((pc[k]['ratio'] - 1) / (pc['measured']['ratio'] - 1)) if pc['measured']['ratio'] > 1 else None
                        pc[k]['m1_dir_diff_deg'] = circ_diff(pc[k]['m1_min_dir_deg'], pc['measured']['m1_min_dir_deg'])
                P2['polygon'][slot][c][str(p)] = pc
        log(f'Part 2: polygon models at slot {slot} done ({time.time()-t0:.0f}s)')
    res['part2'] = P2
    F1 = np.abs(np.fft.rfft(relu(np.cos(np.radians(TH_DEG))))) ** 2; F1[1:] *= 2
    res['reference_m0_single_halfplane'] = float(F1[0] / F1.sum())
    log(f"Part 2 (slot 5, program, g at its mean). Reference: m=0 power fraction of one half-plane ReLU(cos θ) = {res['reference_m0_single_halfplane']:.3f}.")
    log('  per unit: m0 fraction (all cells) | r_c at 3 GeV: mean, max/min, θ_min, m1-min direction | polygon (global / slot / local / fitted): rms rel. error, max/min, m1-min dir '
        '| controls at 3 GeV: perm-weights p, random-dirs p, rotated-dirs p | fixed-point max/min | untrained responds')
    for c in live:
        a_ = P2['program'][5][c]; cc = a_['per_pt']['3.0']; fz = P2['fixed'][5][c]['per_pt']['3.0']
        s = f"  u{c:<3d} m0 {fmt(a_['fourier_f'] and a_['fourier_f']['m0'], '{:.3f}')} | 3GeV r_c {fmt(cc.get('mean'), '{:.3f}')} {fmt(cc.get('ratio'), '{:.3f}')} " \
            f"θmin {fmt(cc.get('theta_min_deg'), '{:.0f}')} m1min {fmt(cc.get('m1_min_dir_deg'), '{:.0f}')} |"
        pg = (P2['polygon'][5].get(c) or {}).get('3.0')
        if pg:
            for k in ('global', 'slot', 'local', 'fitted'):
                v = pg.get(k) or {}
                s += f" {k} {fmt(v.get('rms_rel'), '{:.3f}')} {fmt(v.get('ratio'), '{:.3f}')} {fmt(v.get('m1_min_dir_deg'), '{:.0f}')};"
            s += f" perm p {fmt(pg.get('perm_global', {}).get('p_le'), '{:.3f}')} rand p {fmt(pg.get('random_dirs', {}).get('p_le'), '{:.3f}')} rot p {fmt(pg.get('rotated_dirs', {}).get('p_le'), '{:.3f}')} |"
        else: s += ' polygon - |'
        s += f" fixed {fmt(fz.get('ratio'), '{:.3f}')} | untr {P2['untrained'][5][c]['responds']}"
        log(s)
    log(f'   ({time.time()-t0:.0f}s)')

    # ---------------- Part 3
    lm = np.log(np.maximum(inv_mass(x), 1e-3)); half = rng.permutation(J) < J // 2
    rd = ridge(R0['h5'][half], lm[half]); pr = rd(R0['h5'])
    r2_te = float(1 - np.mean((pr[~half] - lm[~half]) ** 2) / np.var(lm[~half]))
    lm0_fix = pr; lm0_prog = rd(Pf['h5']); lm0_prog_own = lm0_prog
    res['mass_readout'] = dict(r2_test=r2_te, r2_test_program_h5=float(1 - np.mean((lm0_prog[~half] - lm[~half]) ** 2) / np.var(lm[~half])))
    log(f"\nPart 3: log m read-out on h5 (ridge, fit on {int(half.sum())} jets): held-out R² {r2_te:.3f} (fixed point), {res['mass_readout']['r2_test_program_h5']:.3f} (program h5)")
    W_, Z_ = 2, 3

    def cf_S(xe, Smaps):
        rots = [Smaps] + [prog.run(rotate(xe, 45.0 * k), gov=gmf, keep=('S',))['S'] for k in range(1, 8)]
        Se = prog.run(mirror(xe, 1), gov=gmf, keep=('S',))['S']; Sp = prog.run(mirror(xe, 2), gov=gmf, keep=('S',))['S']
        return dict(maps=Smaps, maps_point_even=0.5 * (Smaps + rots[4]), maps_even_in_eta=0.5 * (Smaps + Se), maps_even_in_phi=0.5 * (Smaps + Sp),
                    maps_radial=np.mean(rots, 0))
    CF0 = cf_S(x, Pm['S'])
    call0_prog = Pf['logits'].argmax(1)
    base = dict(fixed_free=R0['S'], fixed_frozen=R0['S'], prog_mean=Pm['S'], prog_own=Pf['S'], prog_free=Pf['S'])
    E = {}
    for dd in DELTAS:
        for name, (de, dp) in DIRS.items(): E[f'{dd}_{name}'] = (de * dd, dp * dd)
    E['rot45'] = None
    P3 = {}
    for ename, sh_ in E.items():
        xe = rotate(x, 45.0) if sh_ is None else shift(x, *sh_)
        ok = in_range(net, xe, LO, HI); n_ok = int(ok.sum())
        Rf = fixed(net, xe, keep=('S', 'logits', 'h5')); Rz = fixed(net, xe, keep=('S', 'logits', 'h5'), g_jet=R0['g'])
        Qm = prog.run(xe, gov=gmf, keep=('S', 'logits')); Qo = prog.run(xe, gov=Gown, keep=('S', 'logits', 'h5')); Qf = prog.run(xe, keep=('S', 'logits', 'h5'))
        CFe = cf_S(xe, Qm['S'])
        runs = dict(fixed_free=Rf, fixed_frozen=Rz, prog_mean=Qm, prog_own=Qo, prog_free=Qf)
        row = dict(n=n_ok, frac_wrap=float(1 - ok.mean()), units={}, WZ={}, calls_changed={}, mass={}, cf_units={})
        for k, r_ in runs.items():
            c_ = r_['logits'].argmax(1); row['WZ'][k] = float((c_[(y == W_) & ok] == Z_).mean())
            row['calls_changed'][k] = float((c_ != (call0 if k.startswith('fixed') else call0_prog))[ok].mean())
        dlm = dict(fixed_free=rd(Rf['h5']) - lm0_fix, fixed_frozen=rd(Rz['h5']) - lm0_fix, prog_own=rd(Qo['h5']) - lm0_prog, prog_free=rd(Qf['h5']) - lm0_prog)
        for v in CF:
            dS = CFe[v] - CF0[v]; _, h5c, _, loc_ = prog.head(Pf['S'] + dS); dlm[v] = rd(h5c) - lm0_prog_own
            cc_ = loc_.argmax(1); row['WZ'][v] = float((cc_[(y == W_) & ok] == Z_).mean()); row['calls_changed'][v] = float((cc_ != call0_prog)[ok].mean())
            row['cf_units'][v] = {c: dict(mean=float(dS[ok, c].mean()), mean_sd=float(dS[ok, c].mean() / sdS[c])) for c in live}
        for k, v in dlm.items():
            vv = v[ok]; row['mass'][k] = dict(mean=float(vv.mean()), sem=float(vv.std() / np.sqrt(len(vv))), median=float(np.median(vv)))
        for c in live:
            dS = {k: (r_['S'][:, c] - base[k][:, c])[ok] for k, r_ in runs.items()}
            d = {k: dict(mean=float(v.mean()), sem=float(v.std() / np.sqrt(len(v))), mean_sd=float(v.mean() / sdS[c])) for k, v in dS.items()}
            for k in ('prog_mean', 'prog_own'):
                a_, b_ = dS[k], dS['fixed_frozen']
                if a_.std() > 1e-9 and b_.std() > 1e-9:
                    d[k]['corr_vs_fixed_frozen'] = float(np.corrcoef(a_, b_)[0, 1]); d[k]['slope_vs_fixed_frozen'] = float((a_ @ b_) / (a_ @ a_))
            row['units'][c] = d
        P3[ename] = row
        log(f"Part 3 {ename:11s} n {n_ok} wrap {row['frac_wrap']:.4f} | ΔS/sd meas(frozen/free) vs maps: " +
            ' '.join(f"S{c} {row['units'][c]['fixed_frozen']['mean_sd']:+.3f}/{row['units'][c]['fixed_free']['mean_sd']:+.3f} vs {row['units'][c]['prog_mean']['mean_sd']:+.3f}"
                     for c in MAIN if c in row['units']) +
            ' | Δlog m̂(h5) meas free/frozen ' + f"{row['mass']['fixed_free']['mean']:+.4f}/{row['mass']['fixed_frozen']['mean']:+.4f}" +
            ' maps ' + ' '.join(f"{v[5:] or 'maps'} {row['mass'][v]['mean']:+.4f}" for v in CF) +
            f" | W→Z fixed {row['WZ']['fixed_free']:.3f} maps {row['WZ']['maps']:.3f} radial {row['WZ']['maps_radial']:.3f}  ({time.time()-t0:.0f}s)")
    # summary: even / odd, ordering, slopes
    def eo(v):
        out = {}
        for ax in AXES:
            ev = 0.5 * (v[f'+{ax}'] + v[f'-{ax}']); od = 0.5 * (v[f'+{ax}'] - v[f'-{ax}'])
            out[ax] = dict(even=float(ev), odd=float(od), odd_over_even=float(od / ev) if abs(ev) > 1e-12 else None)
        return out
    summ = {}
    for dd in DELTAS:
        sd_ = dict(units={}, mass={}, WZ={})
        for c in MAIN:
            if c not in live: continue
            e = {}
            for k in ('fixed_frozen', 'fixed_free', 'prog_mean', 'prog_own', 'prog_free'):
                v = {name: P3[f'{dd}_{name}']['units'][c][k]['mean'] for name in DIRS}
                e[k] = dict(by_dir=v, even_odd=eo(v), spearman_vs_E6v_h5=spearman([v[n] for n in DIRS], [E6V_H5[n] for n in DIRS]),
                            order=sorted(DIRS, key=lambda n: -v[n]))
            for k in CF[1:]:
                v = {name: P3[f'{dd}_{name}']['cf_units'][k][c]['mean'] for name in DIRS}; e[k] = dict(by_dir=v, even_odd=eo(v))
            pm = np.array([P3[f'{dd}_{n}']['units'][c]['prog_mean']['mean'] for n in DIRS]); mf = np.array([P3[f'{dd}_{n}']['units'][c]['fixed_frozen']['mean'] for n in DIRS])
            mfr = np.array([P3[f'{dd}_{n}']['units'][c]['fixed_free']['mean'] for n in DIRS])
            e['spearman_maps_vs_fixed_frozen'] = spearman(pm, mf); e['spearman_maps_vs_fixed_free'] = spearman(pm, mfr)
            e['slope_fixed_frozen_on_maps'] = float((pm @ mf) / (pm @ pm)) if (pm @ pm) > 0 else None
            sd_['units'][c] = e
        for k in ('fixed_free', 'fixed_frozen', 'prog_own', 'prog_free') + CF:
            v = {n: P3[f'{dd}_{n}']['mass'][k]['mean'] for n in DIRS}
            sd_['mass'][k] = dict(by_dir=v, even_odd=eo(v), spearman_vs_E6v_h5=spearman([v[n] for n in DIRS], [E6V_H5[n] for n in DIRS]), order=sorted(DIRS, key=lambda n: -v[n]),
                                  spearman_vs_fixed_free=spearman([v[n] for n in DIRS], [P3[f'{dd}_{n}']['mass']['fixed_free']['mean'] for n in DIRS]))
        for k in ('fixed_free', 'fixed_frozen', 'prog_own', 'prog_free') + CF:
            v = {n: P3[f'{dd}_{n}']['WZ'][k] for n in DIRS}
            sd_['WZ'][k] = dict(by_dir=v, spearman_vs_E6v_WZ=spearman([v[n] for n in DIRS], [E6V_WZ[n] for n in DIRS]),
                                odd_over_even={ax: float((v[f'+{ax}'] - v[f'-{ax}']) / (v[f'+{ax}'] + v[f'-{ax}'])) if (v[f'+{ax}'] + v[f'-{ax}']) > 0 else None for ax in AXES})
        summ[str(dd)] = sd_
        for c in MAIN:
            if c not in live: continue
            e = sd_['units'][c]
            log(f"Part 3 δ={dd} S{c}: odd/even (η, φ, diag) measured frozen " + ' '.join(fmt(e['fixed_frozen']['even_odd'][ax]['odd_over_even'], '{:+.3f}') for ax in AXES) +
                ' | free ' + ' '.join(fmt(e['fixed_free']['even_odd'][ax]['odd_over_even'], '{:+.3f}') for ax in AXES) +
                ' | maps ' + ' '.join(fmt(e['prog_mean']['even_odd'][ax]['odd_over_even'], '{:+.3f}') for ax in AXES) +
                ' | radial maps ' + ' '.join(fmt(e['maps_radial']['even_odd'][ax]['odd_over_even'], '{:+.3f}') for ax in AXES) +
                f" | Spearman maps vs meas(frozen) {e['spearman_maps_vs_fixed_frozen']:.2f}, meas vs E6v h5 {e['fixed_frozen']['spearman_vs_E6v_h5']:.2f}; slope meas/maps {fmt(e['slope_fixed_frozen_on_maps'])}")
        for k in ('fixed_free', 'fixed_frozen', 'prog_own') + CF:
            m_ = sd_['mass'][k]
            log(f"Part 3 δ={dd} Δlog m̂ (h5) {k:17s}: " + ' '.join(f"{n} {m_['by_dir'][n]:+.4f}" for n in DIRS) + ' | odd/even ' +
                ' '.join(f"{ax} {fmt(m_['even_odd'][ax]['odd_over_even'], '{:+.3f}')}" for ax in AXES) +
                f" | Spearman vs E6v {m_['spearman_vs_E6v_h5']:.2f}, vs measured {m_['spearman_vs_fixed_free']:.2f} | W→Z " + ' '.join(f"{sd_['WZ'][k]['by_dir'][n]:.2f}" for n in DIRS))
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
            row['angle_deg'] = float(ang[HP.index(j)]); own = int(row['angle_deg'] // (360 / NSEC)); opp = (own + NSEC // 2) % NSEC
            half_ = [k_ for k_ in range(NSEC) if circ_diff((k_ + 0.5) * 360 / NSEC, row['angle_deg']) < 90]
            row.update(own_sector=own, opp_sector=opp, own_half_sectors=half_)
        tot = {k: np.zeros(NSEC) for k in ('free', 'frozen', 'prog')}; nchg = {k: 0 for k in tot}
        for q, c in enumerate(live):
            wmask = real & KEEP[None, :, c]
            d = {}
            for k, (dU, dS) in dict(free=(Rf['u'][..., q] - U0[..., q], Rf['S'][:, c] - R0['S'][:, c]), frozen=(Rz['u'][..., q] - U0[..., q], Rz['S'][:, c] - R0['S'][:, c]),
                                    prog=(Qm['u'][..., q] - Pm['u'][..., q], Qm['S'][:, c] - Pm['S'][:, c])).items():
                du, sh = sector_stats(dU, wmask, sector, sdS[c], J); tot[k] += np.abs(sh)
                d[k] = dict(dS_sd=float(dS.mean() / sdS[c]), dS_sd_abs=float(np.abs(dS).mean() / sdS[c]), sector_du=du, sector_share=sh,
                            n_particles_changed=int((np.abs(dU) > 1e-6)[wmask].sum()))
                if abs(d[k]['dS_sd_abs']) > 1e-3: nchg[k] += 1
                if kind == 'halfplane': d[k].update(sel_metrics(sh, row['own_sector'], row['opp_sector'], row['own_half_sectors']))
                sv = np.array(sh); d[k]['sector_cv'] = float(sv.std() / abs(sv.mean())) if abs(sv.mean()) > 1e-12 else None
            a_, b_ = np.array(d['prog']['sector_share']), np.array(d['frozen']['sector_share'])
            d['corr_sectors_prog_vs_frozen'] = float(np.corrcoef(a_, b_)[0, 1]) if a_.std() > 0 and b_.std() > 0 else None
            row['units'][c] = d
        row['all_units'] = {}
        for k in tot:
            e = dict(sector_abs_share=tot[k].tolist(), n_units_changed=nchg[k],
                     sector_cv=float(tot[k].std() / tot[k].mean()) if tot[k].mean() > 0 else None)
            if kind == 'halfplane': e.update(sel_metrics(tot[k], row['own_sector'], row['opp_sector'], row['own_half_sectors']))
            row['all_units'][k] = e
        P4[f'a{j}'] = row
        au = row['all_units']
        s = f"Part 4 a{j:<2d} ({kind:9s}" + (f", normal {row['angle_deg']:.0f}°, own sector {row['own_sector']}" if kind == 'halfplane' else '') + \
            f"): calls changed free {row['calls_changed_free']*100:.2f}% frozen {row['calls_changed_frozen']*100:.2f}% prog {row['calls_changed_prog']*100:.2f}% | " \
            f"all units Σ|share| frozen {au['frozen']['total_abs'] if 'total_abs' in au['frozen'] else tot['frozen'].sum():.4f} free {tot['free'].sum():.4f}, units changed {au['frozen']['n_units_changed']}"
        if kind == 'halfplane':
            s += f", own-half frac frozen {fmt(au['frozen']['own_half_frac'], '{:.3f}')} free {fmt(au['free']['own_half_frac'], '{:.3f}')}, opp sector frozen {au['frozen']['opp']:+.5f} free {au['free']['opp']:+.5f}" \
                 f" (own/opp free {fmt(au['free']['own_over_opp'], '{:.1f}')}), argmax sector {au['frozen']['argmax_sector']}"
        else: s += f", sector CV frozen {fmt(au['frozen']['sector_cv'])}"
        for c in (29, 1):
            if c not in live: continue
            d = row['units'][c]
            s += f" | S{c}: ΔS/sd {d['frozen']['dS_sd']:+.3f} (free {d['free']['dS_sd']:+.3f}, prog {d['prog']['dS_sd']:+.3f}) corr sectors prog/meas {fmt(d['corr_sectors_prog_vs_frozen'])}"
        log(s + f'  ({time.time()-t0:.0f}s)')
    res['part4'] = P4; res['dead_unit'] = dead
    hp_rows = [P4[f'a{j}']['all_units'] for j in HP]
    res['part4_summary'] = dict(own_half_frac_frozen={f'a{j}': P4[f'a{j}']['all_units']['frozen']['own_half_frac'] for j in HP},
                                own_half_frac_free={f'a{j}': P4[f'a{j}']['all_units']['free']['own_half_frac'] for j in HP},
                                own_over_opp_free={f'a{j}': P4[f'a{j}']['all_units']['free']['own_over_opp'] for j in HP},
                                opp_zero_frozen={f'a{j}': P4[f'a{j}']['all_units']['frozen']['opp_zero'] for j in HP},
                                argmax_is_own_or_adjacent={f'a{j}': (circ_diff(P4[f'a{j}']['all_units']['frozen']['argmax_sector'] * 30 + 15, P4[f'a{j}']['angle_deg']) <= 45)
                                                           if P4[f'a{j}']['all_units']['frozen']['argmax_sector'] is not None else None for j in HP},
                                min_own_half_frac_frozen=float(min(r_['frozen']['own_half_frac'] for r_ in hp_rows if r_['frozen']['own_half_frac'] is not None)),
                                min_own_half_frac_free=float(min(r_['free']['own_half_frac'] for r_ in hp_rows if r_['free']['own_half_frac'] is not None)),
                                dead_max_abs_dS_sd=float(max(abs(P4[f'a{dead}']['units'][c]['free']['dS_sd_abs']) for c in live)),
                                a56_sector_cv_all_units=P4['a56']['all_units']['frozen']['sector_cv'],
                                a56_sector_cv={c: P4['a56']['units'][c]['frozen']['sector_cv'] for c in (29, 1) if c in live})
    ps = res['part4_summary']
    log(f"Part 4 summary: own-half fraction of Σ|ΔS share| over the 19 units, min over the nine half-planes: frozen {ps['min_own_half_frac_frozen']:.3f}, free {ps['min_own_half_frac_free']:.3f}; "
        f"dead unit a{dead} max mean|ΔS|/sd {ps['dead_max_abs_dS_sd']:.2e}; a56 sector CV (all units) {fmt(ps['a56_sector_cv_all_units'])}")
    res['elapsed_s'] = float(time.time() - t0); res['log'] = _LOG
    JSON_PATH.write_text(json.dumps(clean(res))); np.savez_compressed(NPZ_PATH, **maps)
    log(f'wrote {JSON_PATH} and {NPZ_PATH} ({time.time()-t0:.0f}s)')
    try:
        plot(); log(f'wrote {PNG} and {PNG_TAB}')
    except Exception as ex:  # noqa: BLE001
        log(f'plot failed: {ex!r}')


# ------------------------------------------------------------------ figure
def _arr(v):
    return np.array([np.nan if a is None else a for a in v], float)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    R = json.loads(JSON_PATH.read_text()); thd = np.array(R['grid']['theta_deg'], float); live = R['live_u']
    surf, ink, ink2, grid_c = '#fcfcfb', '#1f1f1e', '#5f5e59', '#e4e3df'
    BLUE, ORANGE, AQUA, YELLOW, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#9a9890'
    hp = R['halfplanes']
    fig = plt.figure(figsize=(17, 11.5), facecolor=surf)
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.42)
    top = outer[0].subgridspec(1, 4, wspace=0.48); bot = outer[1].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.45)

    def style(ax):
        ax.set_facecolor(surf); ax.grid(color=grid_c, lw=0.8); ax.tick_params(labelsize=8, colors=ink2)
        for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'): ax.spines[s_].set_color(grid_c)
    # (a) on-contours r_c(θ), polar
    def cl(v):
        return np.r_[v, v[0]]
    thc = np.radians(cl(thd))
    for k, c in enumerate((29, 1, 50)):
        ax = fig.add_subplot(top[0, k], projection='polar'); ax.set_facecolor(surf)
        if str(c) not in R['part2']['program']['5']: continue
        pp = R['part2']['program']['5'][str(c)]['per_pt']
        ptk = next((p for p in ('3.0', '10.0', '30.0', '1.0', '100.0') if 'mean' in pp[p]), None)
        if ptk is None: continue
        pm = pp[ptk]; prof = _arr(pm['profile'])
        pg = ((R['part2']['polygon'].get('5') or {}).get(str(c)) or {}).get(ptk) or {}
        fx = R['part2']['fixed']['5'][str(c)]['per_pt'][ptk]
        if 'profile' in fx: ax.plot(thc, cl(_arr(fx['profile'])), color=ink, lw=1.0, ls=':', label='trained network, fixed point (r grid 0.01 rad)', zorder=3)
        ax.plot(thc, cl(prof), color=BLUE, lw=2.4, label='trained network, float + masks', zorder=4)
        if pg.get('global') and pg['global'].get('profile'):
            ax.plot(thc, cl(_arr(pg['global']['profile'])), color=ORANGE, lw=1.6, ls='--', label='polygon Σ_j w_j ReLU(n_j·x + β_j) = L, Part-1 weights', zorder=5)
        un = R['part2']['untrained']['5'].get(str(c), {})
        has_un = bool(un.get('per_pt', {}).get(ptk, {}).get('mean'))
        if has_un: ax.plot(thc, cl(_arr(un['per_pt'][ptk]['profile'])), color=GRAY, lw=1.2, label='untrained network')
        rmax = 1.15 * np.nanmax(prof)
        ax.fill_between(np.radians(np.linspace(180, 270, 60)), 0, rmax, color=YELLOW, alpha=0.12, lw=0, zorder=0)
        for name, h in hp.items():
            a_ = np.radians(h['angle_deg']); ax.plot([a_, a_], [rmax * 0.9, rmax], color=ink2, lw=1.4, zorder=2)
            ax.text(a_, rmax * 0.82, name.replace('a', 'a', 1), ha='center', va='center', fontsize=6.8, color=ink2, zorder=6)
        ax.set_rlim(0, rmax); ax.set_rlabel_position(78); ax.tick_params(labelsize=7.5, colors=ink2); ax.grid(color=grid_c, lw=0.8)
        ax.set_thetagrids([0, 90, 180, 270], ['+Δη', '+Δφ', '−Δη', '−Δφ']); ax.xaxis.set_tick_params(pad=1)
        txt = f"max/min {pm['ratio']:.3f}" + (f", polygon {pg['global']['ratio']:.3f}" if pg.get('global') and pg['global'].get('ratio') else '')
        ax.set_title(f'(a) S{c}: r_c(θ) (rad) where u{c} turns on\n{float(ptk):.0f} GeV, slot 5; {txt}', fontsize=9.2, color=ink, loc='left', pad=16)
        if not has_un: ax.text(0.5, -0.17, 'untrained network: no on/off contour at r ≤ 0.4', transform=ax.transAxes, ha='center', fontsize=7.5, color=ink2)
        if k == 0:
            ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc='upper left', bbox_to_anchor=(-0.18, -0.22))
            ax.text(-0.18, -0.49, 'ticks: normals n_j of the nine half-plane statements; shaded: −Δη/−Δφ quadrant', transform=ax.transAxes, fontsize=7.5, color=ink2)
    # (b) predicted vs measured directional ΔS
    ax = fig.add_subplot(top[0, 3]); style(ax)
    for c, col in ((29, BLUE), (1, ORANGE)):
        if str(c) not in R['part3']['0.03_+eta']['units']: continue
        for dd, mk in (('0.03', 'o'), ('0.06', 's')):
            xs = [R['part3'][f'{dd}_{n}']['units'][str(c)]['prog_mean']['mean_sd'] for n in DIRS]; ys = [R['part3'][f'{dd}_{n}']['units'][str(c)]['fixed_frozen']['mean_sd'] for n in DIRS]
            ax.scatter(xs, ys, s=44, marker=mk, facecolor=surf, edgecolor=col, lw=1.8, zorder=3, label=f'S{c}, δ = {dd} rad')
            if dd == '0.06':
                for n, x_, y_ in zip(DIRS, xs, ys): ax.annotate(n.replace('eta', 'η').replace('phi', 'φ'), (x_, y_), xytext=(4, 2), textcoords='offset points', fontsize=7, color=ink2)
    lim = ax.get_xlim(); lim2 = ax.get_ylim(); lo, hi = min(lim[0], lim2[0]), max(lim[1], lim2[1])
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=1, ls='--'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('ΔS / sd(S) predicted: Σ over particles of the maps f_c', fontsize=8.5, color=ink2)
    ax.set_ylabel('ΔS / sd(S) measured, fixed-point network (g frozen)', fontsize=8.5, color=ink2)
    ax.set_title('(b) rigid translation by δ in six directions', fontsize=9.5, color=ink, loc='left')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc='upper left')
    # (c) heatmap: ablation × sector, all live units
    cmap = LinearSegmentedColormap.from_list('seq', ['#f3f2ee', BLUE])
    order = sorted(HP, key=lambda j: hp[f'a{j}']['angle_deg'])
    rows = [f'a{j}' for j in order] + ['a56', f"a{R['dead_unit']}"]
    ax = fig.add_subplot(bot[0, 0]); ax.set_facecolor(surf)
    Mx = np.array([R['part4'][r_]['all_units']['frozen']['sector_abs_share'] for r_ in rows], float)
    tot = Mx.sum(1); Mn = np.where(tot[:, None] > 0, Mx / np.maximum(Mx.max(1, keepdims=True), 1e-12), 0)
    ax.imshow(Mn, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i_, r_ in enumerate(rows):
        P4r = R['part4'][r_]
        if 'own_sector' in P4r:
            ax.plot(P4r['angle_deg'] / 30 - 0.5, i_, marker='o', ms=6, color=ORANGE, mfc='none', mew=1.6)
            lab = f"{r_} (normal {P4r['angle_deg']:.0f}°)"
        else: lab = f'{r_} (pT ramp)' if r_ == 'a56' else f'{r_} (dead unit)'
        ax.text(-0.7, i_, lab, ha='right', va='center', fontsize=8, color=ink)
        ax.text(NSEC - 0.3, i_, f"{tot[i_]:.2f}   {P4r['calls_changed_frozen']*100:4.1f}", ha='left', va='center', fontsize=7.5, color=ink2)
    ax.text(NSEC - 0.3, -0.75, 'Σ      calls %', ha='left', va='bottom', fontsize=7.5, color=ink2)
    ax.set_yticks([]); ax.set_xticks(np.arange(NSEC + 1) - 0.5); ax.set_xticklabels([f'{30*k}°' for k in range(NSEC + 1)], fontsize=8, color=ink2)
    ax.set_xlabel('azimuth sector of the particle, θ = atan2(Δφ, Δη)', fontsize=8.5, color=ink2)
    ax.set_title('(c) one l1 statement set to 0 for every particle (fixed point, g frozen): per-sector Σ over the 19 units of |ΔS_c|/sd, each row scaled to its max;\n'
                 '○ = direction of the half-plane normal; right: row total and % calls changed', fontsize=9, color=ink, loc='left')
    for s_ in ax.spines.values(): s_.set_visible(False)
    # (d) Δlog m̂ (h5) by direction: measured vs maps vs symmetrized maps
    ax = fig.add_subplot(bot[0, 1]); style(ax)
    dd = '0.03'; M = R['part3_summary'][dd]['mass']; dn = list(DIRS)
    series = [('fixed_free', 'measured, fixed-point network', ink), ('maps', 'predicted from the maps f_c', BLUE),
              ('maps_radial', 'maps averaged over θ (radial)', AQUA), ('maps_point_even', 'maps symmetrized f(x)+f(−x)', YELLOW)]
    wbar = 0.2
    for i_, (k, lab, col) in enumerate(series):
        v = [M[k]['by_dir'][n] for n in dn]; oe = M[k]['even_odd']
        lab2 = lab + '; odd/even along η, φ, diag: ' + ', '.join(f"{oe[a]['odd_over_even']:+.2f}" if oe[a]['odd_over_even'] is not None else '-' for a in AXES)
        ax.bar(np.arange(6) + (i_ - 1.5) * wbar, v, wbar * 0.92, color=col, label=lab2)
    ax.scatter(np.arange(6), [E6V_H5[n] for n in dn], marker='_', s=260, color=ORANGE, lw=2.2, zorder=5, label='E6v (seed 17, other 10,000 jets), measured')
    ax.set_xticks(range(6)); ax.set_xticklabels([n.replace('eta', 'η').replace('phi', 'φ') for n in dn], fontsize=9, color=ink)
    ax.set_ylabel('Δ log m̂ read from h5 (ridge on log m)', fontsize=8.5, color=ink2); ax.set_xlabel(f'translation direction, δ = {dd} rad (invariant mass unchanged)', fontsize=8.5, color=ink2)
    ax.set_title('(d) mass read-out under translation: the direction term and the symmetrized-map counterfactuals', fontsize=9.5, color=ink, loc='left')
    ax.legend(frameon=False, fontsize=7.4, labelcolor=ink, loc='upper left', bbox_to_anchor=(0, -0.14), ncol=1)
    allv = [v for k, _, _ in series for v in M[k]['by_dir'].values()]
    ax.set_ylim(min(0, min(allv) * 1.2), max(allv) * 1.12)
    fig.suptitle(f"JEDI-linear 64 p, 3 features: the second-round maps and the nine half-planes. {R['N']} test jets (seed {R['seed']}), fixed-point accuracy {R['acc_fixed']:.3f}",
                 color=ink2, fontsize=9.5, x=0.01, ha='left', y=0.995)
    fig.savefig(PNG, dpi=140, facecolor=surf, bbox_inches='tight'); plt.close(fig)
    # structure table
    SRC = [f'a{j}' for j in HP + PTU + CONST]; cols = SRC + ['s bias', 'ctx g', 'ctx g: part from the 9 half-plane P units']
    live_s = [str(c) for c in live]
    M_ = np.array([[R['part1'][c]['share'][s] for s in SRC] + [R['part1'][c]['groups']['bias']['share'], R['part1'][c]['groups']['ctx']['share'],
                    R['part1'][c]['groups']['ctx_ang']['share']] for c in live_s]) * 100
    fig, ax = plt.subplots(figsize=(14, 0.42 * len(live_s) + 2.4), facecolor=surf); ax.set_facecolor(surf)
    seq = LinearSegmentedColormap.from_list('seq', ['#f3f2ee', BLUE])
    ax.imshow(np.clip(np.abs(M_), 0, 100), cmap=seq, vmin=0, vmax=100, aspect='auto')
    for i_ in range(M_.shape[0]):
        for j_ in range(M_.shape[1]):
            v = M_[i_, j_]
            if abs(v) >= 0.5: ax.text(j_, i_, f'{v:.0f}', ha='center', va='center', fontsize=8, color='white' if abs(v) > 60 else ink)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8.5, color=ink, rotation=30, ha='left'); ax.xaxis.tick_top()
    for xb in (8.5, 12.5, 14.5): ax.axvline(xb, color=ink2, lw=0.8)
    ax.set_yticks(range(len(live_s)))
    ax.set_yticklabels([f"u{c}  HP via s / (HP via s + |HP via g|) = {R['part1'][c]['hp_over_hp_plus_ctx_ang']:.2f}" if R['part1'][c]['hp_over_hp_plus_ctx_ang'] is not None else f'u{c}' for c in live_s],
                       fontsize=8.5, color=ink)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('Variance share (%) of each source in the second-round pre-activation u_c: nine half-planes | four pT ramps | constants | s biases | context g\n'
                 f"(float + masks program, real particles in the unit's pooled slots, {R['N']} jets, seed {R['seed']}; share = cov(term, pre_u)/var(pre_u); "
                 'HP via g = ctx term of g minus ctx term of g recomputed with the nine half-plane P units at their sample mean)', fontsize=9, color=ink, loc='left', pad=48)
    fig.savefig(PNG_TAB, dpi=140, facecolor=surf, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    if '--plot' in sys.argv: plot(); print('wrote', PNG, PNG_TAB)
    else: main()
