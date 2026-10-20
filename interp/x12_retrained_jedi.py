"""X12 (claims A1, A1q, A3, A10; review weaknesses 2, 6, 7; absorbs X5w and X8w): do the program elements of JEDI-linear
3-feature n64 recur in independently retrained JEDI-linear models, and do the cuts / masks need quantization-aware training?

Models (train/train_hgq_jedi.py, official get_gnn architecture, official HGQ2 settings; never the official checkpoints):
  hgq_s0, hgq_s1, hgq_s2      models/small/jedi-n64-x12-hgq/seed<k>      HGQ2 fixed point, beta controlled to the official l1 zero
                                                                         fraction (0.896)
  float_s0, float_s1          models/small/jedi-n64-x12-float/seed<k>    the same script with every quantizer = identity (EBOPs off):
                                                                         same optimizer, BN folding, lr, init, batch, epochs
  official                    the official checkpoint (interp.jedi_core.Jedi(64)), reference row run through the same battery
  --ckpt final (model.keras, test_probs.npy) or e60 (ckpt/epoch_0060.keras, test_probs_e60.npy)
One job per model:  python -m interp.x12_retrained_jedi --model hgq_s0 --ckpt final        -> $AI_RESULTS/jedi/x12_<model>_<ckpt>.json
Combine:            python -m interp.x12_retrained_jedi --combine --ckpt final              -> x12_summary_<ckpt>.json, x12_<ckpt>.png

FROZEN ANALYSIS PLAN (written before the first battery run)
Sample: interp.jedi_core.sample(per_class=5000, seed=0), N = 25,000 test jets; xs = x[::5] (5,000) for unit costs.
Generic forward: the official stage order (a, P, g, s, u, S, h4, h5, z, logits) with each layer's own input quantizer and fused
  quantized kernel (as jedi_core); checked against the stored test_probs.npy on the sample rows (max |dp|, argmax agreement) and,
  for the official model, against Jedi.run.
 (1) accuracy on the sample (Wilson 95 %) and on the 260,000-jet test archive (test.json).
 (2) live l1 cuts: unit a_c set to its mean over real particles, live = >= 1 % calls changed on xs. Physical form
     alpha_pT pT + alpha_eta deta + alpha_phi dphi + beta; term spread w_k = |alpha_k| sd(x_k) over real particles; share
     s_k = w_k^2 / sum w^2; pure = s_pT >= 0.9 (pT ramp) or s_eta + s_phi >= 0.9 (angular). P = pure / live; P_nz over units with
     any nonzero weight; null (X5 / X6v): each feature column of alpha permuted over the 64 units, 2,000 draws, p = (1 + #{P_null
     >= P_nz}) / 2001. CI of P: 1,000 jet bootstraps of xs (live set and sd recomputed). Also the X6 categories (ramp / half-plane
     / angle-offset / mixed) and the l1 zero fraction.
 (3) zero-bit masks of qs (P pooling) and qs1 (S pooling) input quantizers (bits <= 0) on live channels (S live = >= 1 % calls
     on xs): zero-bit slot fraction, partially kept channels, contiguity (longest run / kept) vs the same number of kept slots at
     random ranks (1,000 draws; Holm over channels), window channels = partial and contiguous (Holm p < 0.01). Float arm: no masks
     (identity quantizers), reported as 0.
 (4) rank-window swap test (E20 / X6 step 4): the two edges k|k+1 of largest summed S cost over live S units whose qs1 mask
     changes there (ranks held by >= 50 % of jets); inside pair = the adjacent pair nearest k with no qs1 edge for any live S unit;
     across = swap (deta, dphi) of ranks k, k+1, inside = of j, j+1 (pT order kept); per common jet m = mean over the edge's units
     of |dS_c| / sd(S_c); ratio across / inside with 2,000-draw jet bootstrap CI; paired sign-flip p (10,000) for E[m_a - m_i];
     McNemar on call changes across vs inside. No mask edge (float): ranks 13|14 vs 5|6 and the 10 live S units of largest cost.
     NULL (X8w): mask-keeping reassignment copies of the same model: a permutation pi of the live S channels applied to the l3
     output columns (fused kernel columns, bias) and consistently to the d4 input rows and d4 input-quantizer tables, so every
     function u_pi(c) is pooled through channel c's qs1 quantizer (mask) and read downstream by the d4 weights trained for it; l1, l2,
     dd, all masks unchanged. A draw is kept if it predicts >= 3 classes with >= 1 % of calls each and accuracy > 30 % on xs;
     draws until 10 are kept (at most 200). Same edges, inside pairs and units as the trained model, on x[::2]. Null p90 of the
     ratio; rank p = (1 + #{null >= trained}) / 11.
 (5) translation of every real particle by 0.06 rad along +deta (T); controls: 0.06 rad in a random direction per jet (Tr, rng
     1201) and the X9v M2-held translation (Tr followed by a per-jet angle scaling s = sqrt(M2 / M2'), M0 and M2 about the stored
     axis unchanged); matched rotation (X6, rng 66). W -> Z = fraction of true-W jets called Z; flow = edited - unedited, jet
     bootstrap CI (2,000); exact McNemar on the paired Z calls: T vs Tr, T vs M2-held.
 (6) naming of S objects: for each live S unit (up to 20 by cost), single particle at the kept qs1 slot nearest rank 5 (l1 input
     bits > 0 on all three features there) of an otherwise empty jet, g frozen at its sample mean; grid pT 1, 3, 10, 30, 100 GeV x
     r 0-0.4 (41) x theta 0-355 deg (72); f = u_c at that slot. Azimuthal share A = sum (f - ring mean)^2 / sum (f - mean)^2
     (rings outside the l1 fixed-point range dropped); radial iff A <= 0.017. (pT, r) hinge fit: forward selection (MARS-style, at
     most 6 terms) over hinges max(0, +-(r - t)), max(0, +-(log10 pT - t)) at the grid knots and products of a chosen term with
     one candidate, least squares; R^2 against the full map f (theta included). Hinge object iff R^2 >= 0.9 and radial.
Element calls (per model), thesis predictions: pure cuts: p < 0.05; masks / windows: >= 1 window channel and trained swap ratio
  > null p90 and sign-flip p < 0.05; mass: W -> Z(T) > 0.5 and W -> Z(M2-held) < 0.15 with McNemar T vs M2-held p < 0.05.
  Holm over the element p-values (pure-cut p, window sign-flip p, mass McNemar p) x models in --combine.
Predictions: thesis: HGQ seeds reproduce >= 2 of 3 element types; float arm keeps the mass response, cuts at the null, no masks.
  Alternative: retrained HGQ seeds lack windows / masks, or the float arm shows the same purity.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
KEYS = ('l1', 'qs', 'l2', 'dd', 'add', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')
W_, Z_ = 2, 3
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    return o


def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def mcnemar(a, b):
    from scipy import stats
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5)))
    lp = float(min(0.0, (np.log(2) + stats.binom.logcdf(min(n10, n01), n, 0.5)) / np.log(10))) if n else None
    return dict(first_only=n10, second_only=n01, p=p, log10p=lp)


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]; order = sorted(idx, key=lambda i: ps[i]); m = len(order)
    out = [None] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


RNG = np.random.default_rng(1200)


def boot_mean(v, B_=2000):
    v = np.asarray(v, np.float64); n = len(v)
    if n == 0: return [None, None]
    m = np.array([v[RNG.integers(0, n, n)].mean() for _ in range(B_)]); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def boot_ratio(a, b, B_=2000):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64); n = len(a)
    if n == 0 or b.mean() == 0: return [None, None]
    r = []
    for _ in range(B_):
        i = RNG.integers(0, n, n); d = b[i].mean(); r.append(a[i].mean() / d if d > 0 else np.nan)
    return [float(np.nanquantile(r, .025)), float(np.nanquantile(r, .975))]


def signflip_p(d, P_=10000):
    d = np.asarray(d, np.float64); n = len(d)
    if n == 0 or not (d != 0).any(): return 1.0
    obs = abs(d.mean()); cnt = 0
    for i in range(0, P_, 500):
        s = RNG.choice([-1.0, 1.0], size=(min(500, P_ - i), n)); cnt += int((np.abs((s * d).mean(1)) >= obs - 1e-15).sum())
    return (1 + cnt) / (1 + P_)


# ================================================================================================ generic JEDI-linear forward
def quantizers(L, k):
    iq = L[k].iq; return list(iq.quantizers) if hasattr(iq, 'quantizers') else [iq]


def qbits(q):
    for a in ('b', 'bits'):
        if hasattr(q, a): return np.asarray(getattr(q, a), np.float64)
    k, i, f = (np.asarray(getattr(q, a), np.float64) for a in ('k', 'i', 'f'))
    return np.maximum(k + i + f, 0)


class Net:
    """JEDI-linear-architecture network (official checkpoint or an X12 retrained model) with jedi_core's interface:
    run(x, override, keep, batch), run_jetwise-style per-jet override values, KB (fused kernels), L (layers), shift / scale."""

    def __init__(self, kind, path=None):
        import keras, hgq  # noqa: F401
        from jetdistill.config import DATA
        st = json.loads((DATA / 'norm_stats.json').read_text())['64']
        self.shift, self.scale = np.asarray(st['shift'], np.float64), np.asarray(st['scale'], np.float64)
        self._sh, self._sc = np.float32(st['shift']), np.float32(st['scale'])
        if kind == 'official':
            from interp.jedi_core import Jedi
            j = Jedi(64); self.model, self.L, self.path, self.jedi = j.model, j.L, j.path, j
        else:
            self.model = keras.models.load_model(path, compile=False); self.path = Path(path); self.jedi = None
            self.L = {k: self.model.get_layer(k) for k in KEYS}
        self.is_float = bool(getattr(quantizers(self.L, 'qs1')[0].quantizer, '__dummy__', False))
        self.KB = {}
        for k in ('l1', 'l2', 'dd', 'l3', 'd4', 'd5', 'd6', 'd7'):
            K, b = self.L[k].get_fused_qkernel_and_qbias(False, self.L[k].moving_mean, self.L[k].moving_variance)
            self.KB[k] = (np.asarray(K, np.float64), np.asarray(b, np.float64))
        self.build()

    def inputs(self, x, n=64):
        return (np.asarray(x[:, :n]).astype(np.float32) - self._sh) / self._sc

    def build(self, KB=None):
        """(re)compile the forward; the quantizer tables are read at trace time (call again after changing them)"""
        import jax, jax.numpy as jnp
        L = self.L; KB = KB if KB is not None else self.KB
        self.KBj = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in KB.items()}

        def dense(k, x, KB, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def pool(k, a): return jnp.sum(L[k].iq(a, training=False), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def ov(name, t, O):
            if name not in O: return t
            m, v = O[name]
            if v.ndim == 2 and t.ndim == 3 and t.shape[1] == 1: v = v[:, None, :]
            return t * (1 - m) + m * v

        def forward(x, KB, O):
            a = ov('a', dense('l1', x, KB), O); P = ov('P', pool('qs', a), O); g = ov('g', dense('dd', P, KB), O)
            s = dense('l2', a, KB); qs_, qd_ = L['add'].iq([s, g], training=False)
            u = ov('u', dense('l3', qs_ + qd_, KB), O); S = ov('S', pool('qs1', u), O)
            h4 = dense('d4', S, KB); h5 = dense('d5', h4, KB); z = dense('d6', h5, KB, act=False)
            lo = dense('d7', jax.nn.relu(z), KB); J = x.shape[0]
            return dict(a=a, P=P.reshape(J, -1), g=g.reshape(J, -1), u=u, S=S.reshape(J, -1), z=z.reshape(J, -1), logits=lo.reshape(J, -1))
        jax.clear_caches(); self._f = jax.jit(forward)

    def run(self, x, override=None, keep=('logits',), batch=2500):
        out = {k: [] for k in keep}; O = override or {}
        for i in range(0, len(x), batch):
            xb = x[i:i + batch]; Ob = {}
            for k, (m, v) in O.items():
                v = np.asarray(v, np.float32); v = v[i:i + batch] if (v.ndim >= 2 and v.shape[0] == len(x) and len(x) > 1) else v
                Ob[k] = (np.asarray(m, np.float32), v)
            r = self._f(self.inputs(xb), self.KBj, Ob)
            for k in keep: out[k].append(np.asarray(r[k], np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}


def slot_mask(net, k, idx=0, n=64):
    """(n, C) kept (> 0 bits) of a per-particle input quantizer; ones for identity (float) quantizers"""
    q = quantizers(net.L, k)[idx].quantizer
    if getattr(q, '__dummy__', False): return None, dict(dummy=True)
    b = qbits(q); b2 = b.reshape(b.shape[-2:]) if b.ndim >= 2 else b[None]
    m = np.broadcast_to(b2 > 0, (n, b2.shape[-1])).astype(np.float32)
    return m, dict(bits_shape=list(b.shape), varies_along_slots=bool(b2.shape[0] > 1 and (b2 != b2[:1]).any()),
                   zero_bit_frac=float((b2 <= 0).mean()))


def runs_of(col):
    kept = np.flatnonzero(col > 0)
    if len(kept) == 0: return []
    out, st = [], kept[0]
    for i in range(1, len(kept) + 1):
        if i == len(kept) or kept[i] != kept[i - 1] + 1:
            out.append((int(st), int(kept[i - 1]))); st = kept[i] if i < len(kept) else None
    return out


def contiguity(col):
    r = runs_of(col); nk = int((col > 0).sum())
    return (max(b - a + 1 for a, b in r) / nk) if nk else None


# ================================================================================================ edits
def moments(xx):
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    M0 = pt.sum(1); M2 = (pt * (eta ** 2 + phi ** 2)).sum(1)
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=M0, M2=M2, m=m)


def translate(x, d, th):
    e = x.copy(); rl = e[..., 0] > 0; th = np.broadcast_to(np.asarray(th, np.float64), (len(x),))
    e[..., 1] += (d * np.cos(th))[:, None].astype(np.float32) * rl; e[..., 2] += (d * np.sin(th))[:, None].astype(np.float32) * rl
    return e


def rotate_matched(x, d, rng):
    pt = x[..., 0].astype(np.float64); rl = pt > 0; r = np.hypot(x[..., 1], x[..., 2])
    rbar = (pt * r).sum(1) / np.maximum(pt.sum(1), 1e-12)
    phi = 2 * np.arcsin(np.minimum(1.0, d / np.maximum(2 * rbar, 1e-12))) * rng.choice([-1, 1], len(x))
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]; e = x.copy()
    e[..., 1] = np.where(rl, c * x[..., 1] - s * x[..., 2], x[..., 1]); e[..., 2] = np.where(rl, s * x[..., 1] + c * x[..., 2], x[..., 2])
    return e.astype(np.float32)


def scale_angles(x, k):
    e = x.copy(); rl = e[..., 0] > 0; k = np.asarray(k, np.float64)[:, None]
    e[..., 1] = np.where(rl, x[..., 1] * k, x[..., 1]); e[..., 2] = np.where(rl, x[..., 2] * k, x[..., 2]); return e.astype(np.float32)


def swap(x, k):
    nreal = (x[..., 0] > 0).sum(1); J = np.flatnonzero(nreal > k + 1)
    xe = x[J].copy(); xe[:, [k, k + 1], 1:] = xe[:, [k + 1, k], 1:]; return J, xe


# ================================================================================================ steps
def live_units(net, xs, c0s):
    """mean-ablation cost (fraction of calls changed on xs) of every a (per real particle mean) and S unit"""
    real = xs[..., 0] > 0; R = net.run(xs, keep=('a', 'S', 'g'))
    amean = R['a'][real].mean(0); Smean = R['S'].mean(0); out = {}
    for st, mean, V in (('a', amean, R['a']), ('S', Smean, R['S'])):
        cost = np.zeros(64); ch = np.zeros((len(xs), 64), bool)
        for c in range(64):
            v = V[..., c]
            if np.all(v == v.reshape(-1)[0]) and np.float32(mean[c]) == np.float32(v.reshape(-1)[0]): continue
            m = np.zeros(64); m[c] = 1
            ch[:, c] = net.run(xs, override={st: (m, mean)})['logits'].argmax(1) != c0s; cost[c] = ch[:, c].mean()
        out[st] = dict(cost=cost, ch=ch, live=[int(c) for c in np.flatnonzero(cost >= 0.01)], mean=mean)
    out['gmean'] = R['g'].mean(0)
    return out


def cut_types(alpha, sd, units):
    w2 = (np.abs(alpha[:, units]) * sd[:, None]) ** 2; tot = w2.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        spt = w2[0] / tot; sang = (w2[1] + w2[2]) / tot
    t = np.where(tot == 0, 'constant', np.where(spt >= 0.9, 'ramp', np.where(sang >= 0.9, 'angular', 'mixed')))
    return t, spt


def step_purity(net, xs, U):
    K1 = net.KB['l1'][0].reshape(3, 64); b1 = net.KB['l1'][1].reshape(-1)
    alpha = K1 / net.scale[:, None]; beta = b1 - (K1 * (net.shift / net.scale)[:, None]).sum(0)
    real = xs[..., 0] > 0; xr = xs[real].astype(np.float64); sdx = xr.std(0); medpt = float(np.median(xr[:, 0]))
    live = U['a']['live']; t_obs, spt = cut_types(alpha, sdx, live)
    P = float(np.isin(t_obs, ['ramp', 'angular']).mean()) if live else None
    nzu = np.flatnonzero((alpha != 0).any(0)); t_nz, _ = cut_types(alpha, sdx, nzu)
    P_nz = float(np.isin(t_nz, ['ramp', 'angular']).mean()) if len(nzu) else None
    nr = np.random.default_rng(12011); Pn = []
    for _ in range(2000):
        ap_ = np.stack([alpha[f, nr.permutation(64)] for f in range(3)]); u_ = np.flatnonzero((ap_ != 0).any(0))
        tn, _ = cut_types(ap_, sdx, u_); Pn.append(np.isin(tn, ['ramp', 'angular']).mean() if len(u_) else np.nan)
    Pn = np.array(Pn); p = float((1 + (Pn >= (P_nz if P_nz is not None else 2) - 1e-12).sum()) / 2001)
    # jet bootstrap of P (live set and sd from resampled jets)
    ch = U['a']['ch']; J = len(xs); nreal = real.sum(1).astype(np.float64)
    sx = np.stack([(xs[..., f] * real).sum(1) for f in range(3)], 1).astype(np.float64)
    sx2 = np.stack([((xs[..., f].astype(np.float64)) ** 2 * real).sum(1) for f in range(3)], 1); Pb = []
    bi = np.random.default_rng(12012).integers(0, J, size=(1000, J))
    for i in bi:
        n_ = nreal[i].sum(); mu = sx[i].sum(0) / n_; sd_ = np.sqrt(np.maximum(sx2[i].sum(0) / n_ - mu ** 2, 0))
        lv = np.flatnonzero(ch[i].mean(0) >= 0.01); tb, _ = cut_types(alpha, sd_, lv)
        Pb.append(np.isin(tb, ['ramp', 'angular']).mean() if len(lv) else np.nan)
    # X6 categories
    cats = {}; units = []
    for c, t, s in zip(live, t_obs, spt):
        w = np.abs(alpha[:, c]) * sdx; nw = np.linalg.norm(w); ang = np.hypot(alpha[1, c], alpha[2, c])
        if nw == 0: cat = 'constant'; dist = None
        else:
            fa = np.hypot(w[1], w[2]) / nw; fp = w[0] / nw; dist = float(abs(beta[c] + alpha[0, c] * medpt) / ang) if ang > 0 else None
            cat = 'ramp' if fa < 0.05 else (('halfplane' if (dist is not None and dist < 0.01) else 'angle_offset') if fp < 0.05 else 'mixed')
        cats[cat] = cats.get(cat, 0) + 1
        units.append(dict(unit=int(c), type=str(t), x6_cat=cat, pT_share=float(s), cost=float(U['a']['cost'][c]), alpha=alpha[:, c],
                          beta=float(beta[c]), line_dist_rad=dist,
                          pt_threshold_GeV=float(-beta[c] / alpha[0, c]) if (t == 'ramp' and alpha[0, c] != 0) else None))
    out = dict(n_live=len(live), P=P, P_ci=[float(np.nanquantile(Pb, .025)), float(np.nanquantile(Pb, .975))], P_nonzero_units=P_nz,
               n_nonzero_units=int(len(nzu)), null_q=[float(np.nanquantile(Pn, q)) for q in (.025, .5, .9, .975)], p=p,
               counts={t: int((t_obs == t).sum()) for t in ('ramp', 'angular', 'mixed', 'constant')}, x6_categories=cats,
               sd_features=sdx, alpha_all=alpha,
               l1_zero_frac=float((K1 == 0).mean()), l1_nonzero=int((K1 != 0).sum()), units=units)
    log(f'(2) purity: live l1 {len(live)}, P {P} {out["P_ci"]}, P_nz {P_nz} over {len(nzu)} nonzero units, null {out["null_q"]}, '
        f'p {p:.4f}; counts {out["counts"]}; X6 cats {cats}; l1 zero frac {out["l1_zero_frac"]:.3f}')
    return out


def step_masks(net, U, U_P_live):
    out = {}
    for layer, live in (('qs', U_P_live), ('qs1', U['S']['live'])):
        M, info = slot_mask(net, layer)
        if M is None:
            out[layer] = dict(info, n_partial=0, n_windows=0, zero_bit_frac=0.0, channels={}); continue
        ch, ps = {}, []
        for c in live:
            col = M[:, c]; nk = int(col.sum()); obs = contiguity(col)
            if nk in (0, 64) or obs is None:
                p = 1.0; null = None
            else:
                null = np.array([contiguity(np.isin(np.arange(64), RNG.choice(64, nk, replace=False)).astype(float)) for _ in range(1000)])
                p = (1 + int((null >= obs - 1e-12).sum())) / 1001
            ps.append(p)
            ch[int(c)] = dict(kept=nk, runs=runs_of(col), contiguity=obs, p=p,
                              edges=[k for k in range(63) if col[k] != col[k + 1]])
        for (c, e), h in zip(ch.items(), holm(ps)): e['holm_p'] = h
        part = [c for c, e in ch.items() if 0 < e['kept'] < 64]
        win = [c for c in part if ch[c]['holm_p'] < 0.01 or ch[c]['contiguity'] == 1.0]
        out[layer] = dict(info, channels=ch, n_live=len(live), n_partial=len(part), n_windows=len(win), window_channels=win,
                          kept_slots_live=[ch[c]['kept'] for c in ch])
        log(f'(3) {layer}: zero-bit slot frac {info["zero_bit_frac"]:.3f}, live channels {len(live)}, partial {len(part)}, windows {len(win)}: '
            + ' '.join(f'{c}:{"/".join(f"{a}-{b}" for a, b in ch[c]["runs"])}' for c in part[:20]))
    return out


def choose_edges(M1, liveS, costS, x, minfrac=0.5):
    nreal = (x[..., 0] > 0).sum(1); okk = [k for k in range(63) if (nreal > k + 1).mean() >= minfrac]
    if M1 is None:
        top = sorted(liveS, key=lambda c: -costS[c])[:10]
        return [dict(edge=13, inside=5, units=[int(c) for c in top], mass=0.0, note='no qs1 masks: n64 ranks 13|14 vs 5|6')], {}
    mass = {k: sum(costS[c] for c in liveS if M1[k, c] != M1[k + 1, c]) for k in okk}
    edges = [k for k in sorted(mass, key=lambda k: -mass[k]) if mass[k] > 0][:2]
    plan = []
    for k in edges:
        units = [c for c in liveS if M1[k, c] != M1[k + 1, c]]
        free = [j for j in okk if all(M1[j, c] == M1[j + 1, c] for c in liveS) and j not in (k - 1, k, k + 1)]
        if not free:
            free = sorted(okk, key=lambda j: (sum(M1[j, c] != M1[j + 1, c] for c in units), abs(j - k)))
            free = [j for j in free if j not in (k - 1, k, k + 1)]
        j = min(free, key=lambda j: (-sum(M1[j, c] * M1[j + 1, c] for c in units), abs(j - k)))
        plan.append(dict(edge=int(k), inside=int(j), units=[int(c) for c in units], mass=float(mass[k]),
                         units_kept_at_inside=[int(c) for c in units if M1[j, c] > 0]))
    if not plan:
        top = sorted(liveS, key=lambda c: -costS[c])[:10]
        plan = [dict(edge=13, inside=5, units=[int(c) for c in top], mass=0.0, note='no qs1 edge on live S units: n64 ranks used')]
    return plan, {int(k): float(v) for k, v in mass.items() if v > 0}


def window_test(net, x, plan, label, full=True):
    R0 = net.run(x, keep=('S', 'logits')); S0 = R0['S']; c0 = R0['logits'].argmax(1); sd = S0.std(0); res = []
    for p in plan:
        k, j = p['edge'], p['inside']; Ja, xa = swap(x, k); Ji, xi = swap(x, j)
        Ra = net.run(xa, keep=('S', 'logits')); Ri = net.run(xi, keep=('S', 'logits'))
        u = [c for c in p['units'] if sd[c] > 0]
        ma = (np.abs(Ra['S'][:, u] - S0[Ja][:, u]) / sd[u]).mean(1) if u else np.zeros(len(Ja))
        mi = (np.abs(Ri['S'][:, u] - S0[Ji][:, u]) / sd[u]).mean(1) if u else np.zeros(len(Ji))
        com = np.intersect1d(Ja, Ji); ia, ii = np.searchsorted(Ja, com), np.searchsorted(Ji, com); a, b = ma[ia], mi[ii]
        ca = Ra['logits'].argmax(1)[ia] != c0[com]; cb = Ri['logits'].argmax(1)[ii] != c0[com]
        r = dict(edge=k, inside=j, units_used=u, N=int(len(com)), across=float(a.mean()), inside_mean=float(b.mean()),
                 ratio=float(a.mean() / b.mean()) if b.mean() > 0 else (np.inf if a.mean() > 0 else None),
                 calls_across=float(ca.mean()), calls_inside=float(cb.mean()))
        if full:
            r.update(across_ci=boot_mean(a), inside_ci=boot_mean(b), ratio_ci=boot_ratio(a, b), signflip_p=signflip_p(a - b),
                     signflip_p_one_sided=None, mcnemar_calls=mcnemar(ca, cb),
                     per_unit={int(c): dict(across=float(np.abs(Ra['S'][ia, c] - S0[com, c]).mean() / sd[c]),
                                            inside=float(np.abs(Ri['S'][ii, c] - S0[com, c]).mean() / sd[c])) for c in u})
            r['signflip_p_one_sided'] = r['signflip_p'] / 2 if a.mean() > b.mean() else 1 - r['signflip_p'] / 2
        res.append(r)
        log(f'(4) {label} edge {k}|{k + 1} vs inside {j}|{j + 1}: N={len(com)} units {u}: across {a.mean():.4f} inside {b.mean():.4f} '
            f'ratio {r["ratio"]}' + (f' {r["ratio_ci"]} p {r["signflip_p"]:.2e}' if full else ''))
    return res


def mask_null(net, x, y, xs, ys, plan, liveS, ndraw=10, maxtry=200):
    """X8w null: live S channels reassigned to other trained functions (l3 output columns, d4 input rows and d4 input-quantizer
    tables permuted together), masks kept"""
    K3, b3 = net.KB['l3']; K4, b4 = net.KB['d4']; live = np.array(sorted(liveS)); rng = np.random.default_rng(12013)
    q4 = quantizers(net.L, 'd4')[0].quantizer; vs = list(q4.variables); orig = [np.asarray(v.numpy()).copy() for v in vs]
    kept, tried = [], 0
    while len(kept) < ndraw and tried < maxtry and len(live) >= 2:
        tried += 1
        pi = np.arange(64); pi[live] = live[rng.permutation(len(live))]
        if np.all(pi == np.arange(64)): continue
        KB = dict(net.KB); KB['l3'] = (K3[..., pi], b3.reshape(-1)[pi].reshape(b3.shape)); KB['d4'] = (K4[pi, :], b4)
        for v, a in zip(vs, orig):
            if a.ndim >= 1 and a.shape[-1] == 64: v.assign(a[..., pi])
        net.build(KB)
        c = net.run(xs)['logits'].argmax(1); acc = float((c == ys).mean()); frac = np.bincount(c, minlength=5) / len(c)
        ok = acc > 0.30 and int((frac >= 0.01).sum()) >= 3
        if ok:
            wt = window_test(net, x[::2], plan, f'null draw {tried}', full=False)
            kept.append(dict(draw=tried, acc_xs=acc, call_frac=frac.tolist(), windows=wt))
            log(f'(4) null draw {tried}: acc {acc:.3f}, call frac {np.round(frac, 3).tolist()} -> kept ({len(kept)}/{ndraw})')
        else:
            log(f'(4) null draw {tried}: acc {acc:.3f}, call frac {np.round(frac, 3).tolist()} -> rejected')
    for v, a in zip(vs, orig): v.assign(a)
    net.build()
    return dict(n_kept=len(kept), n_tried=tried, draws=kept,
                rule='pi over live S channels on l3 columns + d4 rows + d4 iq tables; keep if acc > 0.30 and >= 3 classes >= 1 %')


def step_translation(net, x, y):
    J = len(x); W = y == W_; nW = int(W.sum())
    c0 = net.run(x)['logits'].argmax(1); z0 = c0[W] == Z_
    th = np.random.default_rng(1201).uniform(0, 2 * np.pi, J)
    E = dict(T=translate(x, 0.06, 0.0), Tr=translate(x, 0.06, th), rotation=rotate_matched(x, 0.06, np.random.default_rng(66)))
    M = moments(x); Mq = moments(E['Tr']); s = np.sqrt(M['M2'] / np.maximum(Mq['M2'], 1e-30)); E['M2held'] = scale_angles(E['Tr'], s)
    Mh = moments(E['M2held']); ok = M['m'] > 1
    hyg = dict(M2held_dlogM2_maxabs=float(np.abs(np.log(np.maximum(Mh['M2'], 1e-30) / np.maximum(M['M2'], 1e-30)))[M['M2'] > 0].max()),
               M2held_dlogm_mean=float(np.log(Mh['m'][ok] / M['m'][ok]).mean()),
               T_dlogm_maxabs=float(np.abs(np.log(moments(E['T'])['m'][ok] / M['m'][ok])).max()))
    out = dict(N=J, N_W=nW, W_to_Z_unedited=float(z0.mean()), W_to_Z_unedited_ci=wilson(int(z0.sum()), nW), hygiene=hyg, edits={})
    Z = {}
    for k, xe in E.items():
        ce = net.run(xe)['logits'].argmax(1); ze = ce[W] == Z_; Z[k] = ze; flow = ze.astype(float) - z0
        out['edits'][k] = dict(W_to_Z=float(ze.mean()), ci=wilson(int(ze.sum()), nW), flow=float(flow.mean()), flow_ci=boot_mean(flow),
                               calls_changed=float((ce != c0).mean()), calls_changed_ci=wilson(int((ce != c0).sum()), J),
                               acc=float((ce == y).mean()))
    out['mcnemar_Z_T_vs_Tr'] = mcnemar(Z['T'], Z['Tr']); out['mcnemar_Z_T_vs_M2held'] = mcnemar(Z['T'], Z['M2held'])
    out['mcnemar_Z_M2held_vs_rotation'] = mcnemar(Z['M2held'], Z['rotation'])
    e = out['edits']
    log(f'(5) W->Z unedited {out["W_to_Z_unedited"]:.3f}; T {e["T"]["W_to_Z"]:.3f}; Tr {e["Tr"]["W_to_Z"]:.3f}; M2-held {e["M2held"]["W_to_Z"]:.3f}; '
        f'rotation {e["rotation"]["W_to_Z"]:.3f}; McNemar T vs M2held {out["mcnemar_Z_T_vs_M2held"]}; hygiene {hyg}')
    return out


PTG = np.array([1., 3., 10., 30., 100.]); RG = np.linspace(0, 0.4, 41); TH = np.radians(np.arange(0, 360, 5))


def l1_range(net):
    q = quantizers(net.L, 'l1')[0].quantizer
    if getattr(q, '__dummy__', False): return None, None
    try:
        def as_slots(arr):
            a_ = np.asarray(arr, np.float64); a2 = a_.reshape(a_.shape[-2:]) if a_.ndim >= 2 else a_.reshape(1, -1)
            return np.broadcast_to(a2, (64, 3))
        k_, i_, f_ = (as_slots(getattr(q, a_)) for a_ in ('k', 'i', 'f'))
        return np.where(k_ > 0, -2.0 ** i_, 0.0), 2.0 ** i_ - 2.0 ** -f_
    except Exception as e:  # noqa: BLE001
        log('l1 range table unavailable:', repr(e)); return None, None


def hinge_fit(f, keep, maxterms=6):
    """forward selection over hinges of (log10 pT, r) and pairwise products of a chosen hinge with a candidate; least squares;
    R^2 against f on the kept rings (pT x r x theta grid points)"""
    PT, RR, _ = np.meshgrid(PTG, RG, TH, indexing='ij'); sel = np.broadcast_to(keep[..., None], PT.shape).ravel()
    lp = np.log10(PT).ravel()[sel]; r = RR.ravel()[sel]; t = f.ravel()[sel]
    cand = {'r': r, 'lpT': lp}
    for kn in RG[2:-1:2]:
        cand[f'(r-{kn:.2f})+'] = np.maximum(r - kn, 0); cand[f'({kn:.2f}-r)+'] = np.maximum(kn - r, 0)
    for kn in np.log10(PTG)[:-1]:
        cand[f'(lpT-{kn:.2f})+'] = np.maximum(lp - kn, 0); cand[f'({kn:.2f}-lpT)+'] = np.maximum(kn - lp, 0)
    if len(t) == 0: return dict(r2=None, terms=[])
    sst = ((t - t.mean()) ** 2).sum()
    if sst == 0: return dict(r2=None, terms=[])
    chosen = []; X = np.ones((len(t), 1)); r2 = 0.0
    for _ in range(maxterms):
        pool = dict(cand)
        for nm in chosen:
            if '*' in nm: continue
            for k2, v2 in cand.items():
                if k2 != nm: pool[f'{nm}*{k2}'] = cand[nm] * v2
        best = None
        for nm, v in pool.items():
            if nm in chosen: continue
            Xn = np.column_stack([X, v]); w, *_ = np.linalg.lstsq(Xn, t, rcond=None); rss = ((t - Xn @ w) ** 2).sum()
            if best is None or rss < best[1]: best = (nm, rss, v)
        if best is None: break
        r2_new = 1 - best[1] / sst
        if r2_new - r2 < 1e-3: break
        chosen.append(best[0]); X = np.column_stack([X, best[2]]); r2 = r2_new
    return dict(r2=float(r2), terms=chosen)


def step_naming(net, U, M1, maxunits=20):
    LO, HI = l1_range(net); m_l1, _ = slot_mask(net, 'l1'); l1ok = (m_l1.min(1) > 0) if m_l1 is not None else np.ones(64, bool)
    liveS = sorted(U['S']['live'], key=lambda c: -U['S']['cost'][c])[:maxunits]; gmean = U['gmean']
    PT, RR, TT = np.meshgrid(PTG, RG, TH, indexing='ij'); out = {}
    for c in liveS:
        col = M1[:, c] if M1 is not None else np.ones(64)
        cand = [s for s in range(64) if col[s] > 0 and l1ok[s]]
        if not cand: out[int(c)] = dict(slot=None); continue
        slot = min(cand, key=lambda s: abs(s - 5))
        xs = np.zeros((PT.size, 64, 3), np.float32)
        xs[:, slot, 0] = PT.ravel(); xs[:, slot, 1] = (RR * np.cos(TT)).ravel(); xs[:, slot, 2] = (RR * np.sin(TT)).ravel()
        # batched, keeping only u[:, slot, c] (the full u of the 14,760-jet grid in float64 is ~0.5 GB per copy; the first
        # official smoke run was OOM-killed at 2 GiB here)
        Og = {'g': (np.ones(64, np.float32), np.asarray(gmean, np.float32))}
        f = np.concatenate([np.asarray(net._f(net.inputs(xs[i:i + 1000]), net.KBj, Og)['u'][:, slot, c], np.float64)
                            for i in range(0, len(xs), 1000)]).reshape(PT.shape)
        if LO is not None:
            xn = (xs[:, slot].astype(np.float64) - net.shift) / net.scale
            ring_bad = ((xn < LO[slot]) | (xn > HI[slot])).any(1).reshape(PT.shape).any(-1)
        else:
            ring_bad = np.zeros(PT.shape[:2], bool)
        fr = f[~ring_bad]; tot = ((fr - fr.mean()) ** 2).sum() if fr.size else 0.0
        az = ((fr - fr.mean(1, keepdims=True)) ** 2).sum() if fr.size else 0.0
        A = float(az / tot) if tot > 0 else None
        hf = hinge_fit(f, ~ring_bad)
        out[int(c)] = dict(slot=int(slot), cost=float(U['S']['cost'][c]), azimuthal_share=A, rings_used=int((~ring_bad).sum()),
                           total_var=float(tot / max(fr.size, 1)), hinge_r2=hf['r2'], hinge_terms=hf['terms'],
                           radial=bool(A is not None and A <= 0.017), hinge=bool(hf['r2'] is not None and hf['r2'] >= 0.9),
                           constant=bool(tot == 0))
    ok = [e for e in out.values() if e.get('slot') is not None and not e.get('constant')]
    summ = dict(n_units=len(out), n_mapped=len(ok), n_radial=sum(e['radial'] for e in ok), n_hinge=sum(e['hinge'] for e in ok),
                n_radial_hinge=sum(e['radial'] and e['hinge'] for e in ok))
    summ['frac_radial_hinge'] = summ['n_radial_hinge'] / summ['n_mapped'] if summ['n_mapped'] else None
    log(f'(6) naming: {summ}; ' + ', '.join(f'S{c}@{e.get("slot")}:A={e.get("azimuthal_share") if e.get("azimuthal_share") is None else round(e["azimuthal_share"], 4)},'
                                             f'R2={e.get("hinge_r2") if e.get("hinge_r2") is None else round(e["hinge_r2"], 3)}' for c, e in out.items()))
    return dict(units=out, summary=summ)


# ================================================================================================ one model
def model_path(name, ckpt):
    if name == 'official': return None, None, None
    arm, s = name.split('_s'); d = SMALL / f'jedi-n64-x12-{arm}' / f'seed{s}'
    if ckpt == 'final': return d / 'model.keras', d / 'test_probs.npy', d / 'test.json'
    e = int(ckpt[1:]); return d / 'ckpt' / f'epoch_{e:04d}.keras', d / f'test_probs_e{e}.npy', d / f'test_e{e}.json'


def run_model(a):
    from interp.jedi_core import sample
    from jetdistill.config import RESULTS
    JS = OUT / f'x12_{a.model}_{a.ckpt}{"_smoke" if a.smoke else ""}.json'
    R = dict(model=a.model, ckpt=a.ckpt, plan='module docstring (frozen)')
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    per = 300 if a.smoke else 5000
    x, y, _ = sample(per_class=per, seed=0); x = np.ascontiguousarray(x, np.float32); J = len(x)
    yy = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(0)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yy == c), per, replace=False) for c in range(5)])); assert np.array_equal(yy[idx], y)
    xs, ys = x[::5], y[::5]
    R['sample'] = dict(per_class=per, seed=0, N=J, N_units=len(xs))
    mp, pp, tp = model_path(a.model, a.ckpt)
    net = Net('official' if a.model == 'official' else 'x12', mp); R['path'] = str(net.path); R['is_float'] = net.is_float
    log(f'model {a.model} {a.ckpt}: {net.path} float {net.is_float}; sample N {J}')
    # (1)
    lo = net.run(x)['logits']; c0 = lo.argmax(1)
    st = dict(acc=float((c0 == y).mean()), acc_ci=wilson(int((c0 == y).sum()), J), call_frac=(np.bincount(c0, minlength=5) / J).tolist())
    if a.model == 'official':
        ref = net.jedi.run(x[:2000])['logits']; st['forward_vs_jedi_core_maxabs'] = float(np.abs(ref - lo[:2000]).max())
        from jetdistill.config import DATA
        oa = DATA / 'official_acc.json'
        st['test_acc_official'] = json.loads(oa.read_text()).get('3-feature', {}).get('64') if oa.exists() else None
    else:
        tj = json.loads(Path(tp).read_text()); st['test_acc'] = tj['accuracy']; st['test_json'] = {k: tj.get(k) for k in ('accuracy', 'l1_zero_frac', 'qs1_zero_bit_frac', 'epochs_run', 'epoch')}
        P = np.load(pp, mmap_mode='r')
        if len(P) == len(yy):
            P = np.asarray(P[idx]); q = np.exp(lo - lo.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
            st['max_abs_dprob_vs_test_probs'] = float(np.abs(q - P).max()); st['argmax_agreement'] = float((P.argmax(1) == c0).mean())
        # exactness: the Keras model itself on CPU, same float32 inputs (2,000 jets); and on float16-rounded inputs (the
        # training script's input dtype, also used for the stored test_probs) against test_probs
        # 500 jets (2,000 jets OOM-killed the hgq_s0 rerun at the 2 GiB request)
        NK = 500; xin = net.inputs(x[:NK]); lk = np.asarray(net.model(xin, training=False), np.float64)
        st['keras_check_N'] = NK
        st['forward_vs_keras_cpu_maxabs_logit'] = float(np.abs(lk - lo[:NK]).max())
        st['forward_vs_keras_cpu_argmax_agree'] = float((lk.argmax(1) == c0[:NK]).mean())
        lk16 = np.asarray(net.model(xin.astype(np.float16).astype(np.float32), training=False), np.float64)
        st['f16_inputs_calls_changed_vs_f32'] = float((lk16.argmax(1) != c0[:NK]).mean())
        if len(np.load(pp, mmap_mode='r')) == len(yy):
            P2 = np.asarray(np.load(pp, mmap_mode='r')[idx[:NK]]); q16 = np.exp(lk16 - lk16.max(1, keepdims=True)); q16 /= q16.sum(1, keepdims=True)
            st['keras_cpu_f16_vs_test_probs_maxabs'] = float(np.abs(q16 - P2).max())
            st['keras_cpu_f16_vs_test_probs_argmax_agree'] = float((P2.argmax(1) == lk16.argmax(1)).mean())
        logf = Path(mp).parents[1 if a.ckpt != 'final' else 0] / 'log.json'
        if logf.exists():
            rows = json.loads(logf.read_text()); R['train_log_last'] = rows[-1]
            R['train_log_e60'] = next((r for r in rows if r['epoch'] == 60), None)
    R['step1_accuracy'] = st; log('(1)', st)
    # architecture record: fused kernel shapes, input-quantizer table shapes, parameter count (retrained vs official)
    arch = dict(n_params=int(net.model.count_params()), layers={})
    for k in KEYS:
        e = dict(iq=[])
        for q in quantizers(net.L, k):
            qq = q.quantizer; e['iq'].append(dict(type=type(qq).__name__, dummy=bool(getattr(qq, '__dummy__', False)),
                                                  vars={v.path.split('/')[-1]: list(v.shape) for v in qq.variables}))
        if k in net.KB: e['kernel'] = list(net.KB[k][0].shape); e['bias'] = list(net.KB[k][1].shape)
        arch['layers'][k] = e
    R['arch'] = arch; log('arch', json.dumps(clean(arch))); save()
    # live units
    U = live_units(net, xs, c0[::5])
    R['live'] = dict(a=U['a']['live'], S=U['S']['live'], a_cost=U['a']['cost'], S_cost=U['S']['cost'])
    log(f'live a {len(U["a"]["live"])}: {U["a"]["live"]}; live S {len(U["S"]["live"])}'); save()
    # live P units (for the qs masks): P unit set to its mean
    RP = net.run(xs, keep=('P',))['P']; Pm = RP.mean(0); costP = np.zeros(64)
    for c in range(64):
        if np.all(RP[:, c] == RP[0, c]): continue
        m = np.zeros(64); m[c] = 1; costP[c] = (net.run(xs, override={'P': (m, Pm)})['logits'].argmax(1) != c0[::5]).mean()
    liveP = [int(c) for c in np.flatnonzero(costP >= 0.01)]; R['live']['P'] = liveP
    # (2)
    R['purity'] = step_purity(net, xs, U); save()
    # (3)
    R['masks'] = step_masks(net, U, liveP); save()
    # (4)
    M1, _ = slot_mask(net, 'qs1'); costS = {c: U['S']['cost'][c] for c in range(64)}
    plan, mass = choose_edges(M1, U['S']['live'], costS, x); R['windows'] = dict(plan=plan, edge_mass=mass)
    R['windows']['trained'] = window_test(net, x, plan, 'trained'); save()
    if M1 is not None and not a.no_null:
        R['windows']['trained_half'] = window_test(net, x[::2], plan, 'trained x[::2]', full=False)
        R['windows']['null'] = mask_null(net, x, y, xs, ys, plan, U['S']['live'], ndraw=3 if a.smoke else 10)
        for pi, p in enumerate(plan):
            nr = [d['windows'][pi]['ratio'] for d in R['windows']['null']['draws'] if d['windows'][pi]['ratio'] is not None]
            tr = R['windows']['trained_half'][pi]['ratio']
            p['null_ratios'] = nr; p['null_p90'] = float(np.quantile(nr, 0.9)) if nr else None
            p['rank_p'] = (1 + sum(1 for v in nr if v >= tr)) / (1 + len(nr)) if (nr and tr is not None) else None
            log(f'(4) edge {p["edge"]}: trained ratio (x[::2]) {tr}; null ratios {np.round(nr, 3).tolist()}; p90 {p["null_p90"]}; rank p {p["rank_p"]}')
        chk = net.run(x[:500])['logits']; R['windows']['restore_check_maxabs'] = float(np.abs(chk - lo[:500]).max())
    save()
    # (5)
    R['translation'] = step_translation(net, x, y); save()
    # (6)
    R['naming'] = step_naming(net, U, M1, maxunits=5 if a.smoke else 20); save()
    # element calls
    pur = R['purity']; w = R['windows']['trained'][0]; pl = R['windows']['plan'][0]; tr = R['translation']
    R['elements'] = dict(
        pure_cuts=dict(stat=pur['P_nonzero_units'], p=pur['p'], present=bool(pur['p'] < 0.05)),
        windows=dict(stat=w['ratio'], ci=w.get('ratio_ci'), p=w.get('signflip_p_one_sided'), null_p90=pl.get('null_p90'),
                     n_window_channels=R['masks']['qs1'].get('n_windows', 0),
                     present=bool(R['masks']['qs1'].get('n_windows', 0) >= 1 and w['ratio'] is not None and pl.get('null_p90') is not None
                                  and w['ratio'] > pl['null_p90'] and (w.get('signflip_p_one_sided') or 1) < 0.05)),
        mass=dict(stat=tr['edits']['T']['W_to_Z'], m2held=tr['edits']['M2held']['W_to_Z'],
                  p=tr['mcnemar_Z_T_vs_M2held']['p'] if tr['mcnemar_Z_T_vs_M2held']['first_only'] > tr['mcnemar_Z_T_vs_M2held']['second_only'] else 1.0,
                  present=bool(tr['edits']['T']['W_to_Z'] > 0.5 and tr['edits']['M2held']['W_to_Z'] < 0.15)),
        naming=R['naming']['summary'])
    import resource
    R['wall_s'] = time.time() - T0; R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    save(); log('elements', json.dumps(clean(R['elements']))); log(f'done; wrote {JS}')


# ================================================================================================ combine
MODELS = ['official', 'hgq-r150_s0', 'hgq-r150_s1', 'hgq-r150_s2', 'hgq_s0', 'hgq_s1', 'hgq_s2', 'float_s0', 'float_s1']
# HGQ arms (decided 2026-10-19 before any battery result of either arm): primary = hgq-r150 (300 epochs, target zero fraction
# reached at epoch 150, beta <= 3e-6 = the official schedule's maximum); secondary = hgq (200 epochs, ramp to epoch 50, beta
# <= 1e-4: train accuracy fell to 0.59-0.61 at epoch 58, beta overshot to 1-4e-5). Holm family = primary HGQ + float rows.


def combine(a):
    D = {}
    for m in MODELS:
        p = OUT / f'x12_{m}_{"final" if m == "official" else a.ckpt}.json'
        if p.exists(): D[m] = json.loads(p.read_text())
    keys, ps = [], []
    for m, d in D.items():
        if m == 'official' or m.startswith('hgq_') or 'elements' not in d: continue   # secondary HGQ arm outside the Holm family
        for el in ('pure_cuts', 'windows', 'mass'):
            keys.append((m, el)); ps.append(d['elements'][el]['p'])
    hp = holm(ps); S = dict(ckpt=a.ckpt, models=list(D), holm={f'{m}:{el}': dict(p=p, p_holm=h) for (m, el), p, h in zip(keys, ps, hp)}, rows={})
    for m, d in D.items():
        if 'elements' not in d: continue
        e = d['elements']; tr = d['translation']['edits']
        S['rows'][m] = dict(acc_sample=d['step1_accuracy']['acc'], acc_test=d['step1_accuracy'].get('test_acc', d['step1_accuracy'].get('test_acc_official')),
                            l1_zero_frac=d['purity']['l1_zero_frac'], n_live_l1=d['purity']['n_live'], P=d['purity']['P'],
                            P_nz=d['purity']['P_nonzero_units'], P_null_q=d['purity']['null_q'], P_p=d['purity']['p'],
                            qs1_zero_bit_frac=d['masks']['qs1'].get('zero_bit_frac', 0.0), qs1_windows=d['masks']['qs1'].get('n_windows', 0),
                            qs1_partial=d['masks']['qs1'].get('n_partial', 0), n_live_S=len(d['live']['S']),
                            window=e['windows'], WZ_unedited=d['translation']['W_to_Z_unedited'], WZ_T=tr['T']['W_to_Z'],
                            WZ_Tr=tr['Tr']['W_to_Z'], WZ_M2held=tr['M2held']['W_to_Z'], WZ_rotation=tr['rotation']['W_to_Z'],
                            naming=e['naming'], present={k: e[k]['present'] for k in ('pure_cuts', 'windows', 'mass')},
                            flow={k: dict(flow=tr[k]['flow'], ci=tr[k]['flow_ci'], calls_changed=tr[k]['calls_changed']) for k in tr},
                            mcnemar_T_vs_Tr=d['translation']['mcnemar_Z_T_vs_Tr'], mcnemar_T_vs_M2held=d['translation']['mcnemar_Z_T_vs_M2held'],
                            window_plan=[{k: p_.get(k) for k in ('edge', 'inside', 'units', 'null_p90', 'rank_p', 'null_ratios')} for p_ in d['windows']['plan']],
                            window_trained=[{k: w_.get(k) for k in ('edge', 'inside', 'N', 'across', 'inside_mean', 'ratio', 'ratio_ci', 'signflip_p',
                                                                    'mcnemar_calls', 'calls_across', 'calls_inside')} for w_ in d['windows']['trained']],
                            null_kept=d['windows'].get('null', {}).get('n_kept'), null_tried=d['windows'].get('null', {}).get('n_tried'),
                            null_acc=[x_['acc_xs'] for x_ in d['windows'].get('null', {}).get('draws', [])],
                            qs_windows=d['masks']['qs'].get('n_windows', 0), qs_zero_bit_frac=d['masks']['qs'].get('zero_bit_frac', 0.0),
                            exactness={k: d['step1_accuracy'].get(k) for k in ('forward_vs_jedi_core_maxabs', 'max_abs_dprob_vs_test_probs',
                                       'argmax_agreement', 'keras_check_N', 'forward_vs_keras_cpu_maxabs_logit', 'f16_inputs_calls_changed_vs_f32',
                                       'keras_cpu_f16_vs_test_probs_maxabs', 'keras_cpu_f16_vs_test_probs_argmax_agree')},
                            x6_categories=d['purity']['x6_categories'], l1_nonzero=d['purity']['l1_nonzero'])
    # sparsity-matched purity (structural, weights only): each float model's first-layer term spreads |alpha_k| sd(x_k), only the
    # k largest of the 3 x 64 entries kept, k = the nonzero l1 entry count of each HGQ model; purity among units with a kept entry
    def topk_purity(alpha, sd, k):
        w = np.abs(np.asarray(alpha)) * np.asarray(sd)[:, None]; thr = np.sort(w.ravel())[::-1][k - 1]; wk = np.where(w >= thr, w, 0.0)
        u = np.flatnonzero((wk > 0).any(0)); t, _ = cut_types(wk / np.asarray(sd)[:, None], np.asarray(sd), u)
        return float(np.isin(t, ['ramp', 'angular']).mean()), int(len(u))
    S['sparsity_matched_purity'] = {}
    for m, d in D.items():
        if not m.startswith('float') or 'alpha_all' not in d.get('purity', {}): continue
        for h, dh in D.items():
            if h.startswith('float') or 'purity' not in dh: continue
            k = dh['purity']['l1_nonzero']; P_, nu = topk_purity(d['purity']['alpha_all'], d['purity']['sd_features'], k)
            S['sparsity_matched_purity'][f'{m}@k={k}({h})'] = dict(P=P_, n_units=nu, hgq_P_nz=dh['purity']['P_nonzero_units'])
            log(f'sparsity-matched purity {m} top-{k} entries (as {h}): P {P_:.3f} over {nu} units; {h} P_nz {dh["purity"]["P_nonzero_units"]}')
    # radial (pT, r)-hinge objects, pooled per arm; float units cut to the top-n by cost with n = the mean number of mapped HGQ
    # units (naming lists are ordered by cost); Fisher exact test (two-sided), H0: same radial-hinge share in both arms
    from scipy import stats
    def counts(ms, top=None):
        k = n = 0
        for m in ms:
            if m not in D: continue
            us = [u for u in D[m]['naming']['units'].values() if u.get('slot') is not None and not u.get('constant')]
            us = us[:top] if top else us; k += sum(bool(u['radial'] and u['hinge']) for u in us); n += len(us)
        return k, n
    prim = ['hgq-r150_s0', 'hgq-r150_s1', 'hgq-r150_s2']; sec = ['hgq_s0', 'hgq_s1', 'hgq_s2']; flo = ['float_s0', 'float_s1']
    nh = int(round(np.mean([counts([m])[1] for m in prim if m in D]))) if any(m in D for m in prim) else None
    S['radial_hinge'] = {}
    for lab, hg in (('primary', prim), ('secondary', sec)):
        kh, nh_ = counts(hg); kf, nf = counts(flo); kf2, nf2 = counts(flo, top=nh)
        S['radial_hinge'][lab] = dict(hgq=[kh, nh_], float_all20=[kf, nf], float_topn=[kf2, nf2], topn=nh,
                                      fisher_p_all20=float(stats.fisher_exact([[kh, nh_ - kh], [kf, nf - kf]])[1]),
                                      fisher_p_topn=float(stats.fisher_exact([[kh, nh_ - kh], [kf2, nf2 - kf2]])[1]),
                                      official=list(counts(['official'])))
        log(f'radial hinge objects {lab}: HGQ {kh}/{nh_}; float {kf}/{nf} (top-{nh} by cost: {kf2}/{nf2}); official {S["radial_hinge"][lab]["official"]}; '
            f'Fisher p {S["radial_hinge"][lab]["fisher_p_all20"]:.2e} / {S["radial_hinge"][lab]["fisher_p_topn"]:.2e}')
    for k, v in S['holm'].items(): log(f'Holm {k}: p {v["p"]} -> {v["p_holm"]}')
    for m, r in S['rows'].items(): log(m, json.dumps(clean(r)))
    (OUT / f'x12_summary_{a.ckpt}.json').write_text(json.dumps(clean(S), indent=1)); log('wrote', OUT / f'x12_summary_{a.ckpt}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='official'); ap.add_argument('--ckpt', default='final')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--no-null', dest='no_null', action='store_true')
    ap.add_argument('--combine', action='store_true')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    combine(a) if a.combine else run_model(a)
