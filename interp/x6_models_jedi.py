"""X6 (claims A1, A3, A10, B1; round-2 weaknesses 1 "single model" and 2 "objects = sum pooling + HGQ pruning"): do the
program elements found in JEDI-linear 3-feature n64 recur in separately trained official JEDI-linear checkpoints?

Models (official_models/<family>/jet_classifier_large_<n>, no training): 3-feature n32, 3-feature n128, 3-feature-perminv n64,
and 3-feature n64 as the reference. One job per model:  python -m interp.x6_models_jedi --family F --n N
Each job also runs the untrained redraw Jedi(untrained=True, seed=0) (trained number formats / zero-bit masks, redrawn weights).

Samples: measurement M = sample(per_class=4000, seed=0) (20,000 test jets); confirmation C = sample(per_class=2000, seed=77)
with every jet of M removed (disjoint).  For n = 128 the same rows are read from val_ptetaphi_f16.npy with 128 slots (the
first 64 slots are checked equal to full_test.npz).  For n = 32 the first 32 slots.

FROZEN ANALYSIS PLAN (written before launch; the n64 thresholds are fixed and not tuned per model)
 Step 0  accuracy of the unedited fixed-point network on M and C (reference: official accuracy of that checkpoint).
 Step 1  E1 per stage (a P g s u S h4 h5 z): each unit set to its mean over real data (a, s, u: per real particle; others per
         jet) -> fraction of calls changed. live = changed >= 1 % (point). 95 % interval: percentile bootstrap over jets
         (binomial resampling of the changed indicator, 2000 draws). H0: changed rate <= 1 %, H1: > 1 % (one-sided exact binomial),
         Holm over the 64 (or 32 / 16) units of the stage; "live_holm" = Holm p < 0.05.  Untrained redraw: same, on M[::2].
 Step 2  E2: every live l1 unit (stage a, point criterion) as  α_pT pT + α_η Δη + α_φ Δφ + β > 0  in physical units; term spread
         w_k = |α_k| sd(x_k) over real particles. ramp: |w_ang|/|w| < 0.05 (w_ang = (w_η, w_φ)); half-plane: |w_pT|/|w| < 0.05 and
         the line's distance from the axis |β + α_pT median(pT)| / |(α_η, α_φ)| < 0.01 rad; angle-offset: |w_pT|/|w| < 0.05,
         distance >= 0.01 rad; constant: all α = 0; mixed: otherwise. Fractions with Wilson 95 % intervals; Fisher exact test of
         (pT-only + angle-only) vs mixed against the n64 counts (4 ramps / 9 half-planes / 0 mixed).
 Step 3  zero-bit masks (HGQ input-quantizer bits > 0) of qs (channels = live P units) and qs1 (channels = live S units), slot
         axis = pT rank.  Per channel: kept slots, contiguity = longest run of consecutive kept slots / kept slots, edges k|k+1
         where the mask changes. Null: the same number of kept slots at random ranks (1000 draws), p = (1 + #{null >= obs})/1001;
         Holm over the channels of the layer. Whether the mask varies along the slot axis at all is reported (perminv:
         uniform quantizers along the particle axis by construction, gnn_uq1).
 Step 4  E20 causal window test at the model's own largest-mass edges: edge mass(k) = sum over live S units c with a qs1 mask
         change at k|k+1 of cost_S(c) (step 1); k restricted to ranks with >= 50 % of M jets holding a particle at rank k+1;
         the two largest. Edits: across = swap (Δη, Δφ) of ranks k and k+1 (pT order kept); inside = swap of the adjacent pair
         j|j+1 nearest to k with no qs1 edge for any live S unit (pT order kept). Units = the live S units with an edge at k.
         Per jet m = mean over those units of |ΔS_c| / sd(S_c). Ratio across / inside (common jets) with bootstrap 95 % CI;
         H0: E[m_across] = E[m_inside], H1: not equal, paired sign-flip permutation p (10,000). Masks removed: the same two
         edits on the float + zero-bit-mask program (E3m) and on the same program with the slot masks of qs, l2, add, l3, qs1
         set to one; ratio across / inside in each; untrained fixed-point network: same ratio.  For a model without qs1 edges
         (perminv) the n64 ranks are used: across 13|14, inside 5|6, units = the 10 live S units of largest cost.
         Repeated on C.
 Step 5  radial test of the 5 live S units of largest cost: single particle (pT 1, 3, 10, 30, 100 GeV; r 0-0.4 rad, 41 steps;
         θ 0-355°, 72 steps) at the kept qs1 slot nearest rank 5 (l1 bits > 0 on all three features at that slot) of an
         otherwise empty jet, g frozen at its M mean, fixed-point network; f = u_c at that slot. radial share = 1 - Σ_rings
         Σ_θ (f - ring mean)² / Σ (f - grand mean)² (rings with any grid point outside the slot's l1 fixed-point range dropped).
         Controls: the live l1 units' own maps (a_c at the slot; the half-planes are not radial), the untrained network's 5 S units
         of largest sd.
 Step 6  translation edit δ = 0.06 rad of every real particle along 8 directions (0, 45, ..., 315°; 0° = +Δη): fraction of
         true-W jets called Z (unedited, edited); % calls changed. Control: rotation of every real particle about the axis by a
         per-jet angle φ_j = 2 arcsin(min(1, δ / (2 r̄_j))) (r̄_j = pT-weighted mean distance from the axis; equal mean
         displacement to the translation), random sign. Wilson 95 % CI; paired McNemar (Z call under translation vs rotation, per W
         jet). Repeated on C; untrained network: same.
 Predictions. Thesis (elements are learned regularities of JEDI-linear training): pT-only + angle-only >= 70 % of live l1;
   contiguous windows (Holm p < 0.01); across / inside ratio > 5; radial share >= 0.9; untrained 0 live units.
   Alternative (artefact of one HGQ checkpoint): mixed cuts dominate; kept slots not contiguous (p > 0.05); across ≈ inside.
   perminv: no slot-dependent masks can exist; predicts across ≈ inside; cut types and radial objects can still recur.
Writes $AI_RESULTS/jedi/x6_<family>_n<n>.json (partial after every step).    --resume reuses step 1 from that file.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument('--family', default='3-feature', choices=['3-feature', '3-feature-perminv'])
ap.add_argument('--n', type=int, default=64, choices=[32, 64, 128])
ap.add_argument('--per-class', type=int, default=4000)
ap.add_argument('--conf-per-class', type=int, default=2000)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--extra-edges', default='', help='post hoc (not in the frozen plan): comma-separated ranks k of extra k|k+1 edges, '
                'inside pair by the same rule, rank limit relaxed to >= 20 %% of jets')
args = ap.parse_args()

import jax, jax.numpy as jnp
from scipy import stats
from interp.jedi_core import Jedi, sample, STAGES, WIDTH
from interp import e19_common as C

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)

N = args.n; FAM = args.family; TAG = f'x6_{FAM}_n{N}'; JSON = OUT / f'{TAG}.json'
B = 2500 if N <= 64 else 1000
NB = 2000; NPERM = 10000; NNULL = 1000
RNG = np.random.default_rng(606)
N64 = dict(ramp=4, halfplane=9, mixed=0)
R = json.loads(JSON.read_text()) if (args.resume and JSON.exists()) else {}
R.update(family=FAM, n=N, plan='see module docstring (frozen before launch)')


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, float): return None if not np.isfinite(o) else o
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.bool_,)): return bool(o)
    return o


def save(): JSON.write_text(json.dumps(clean(R), indent=1))


def boot_mean(v, B_=NB):
    v = np.asarray(v, np.float64); n = len(v)
    if n == 0: return [None, None]
    m = np.array([v[RNG.integers(0, n, n)].mean() for _ in range(B_)]); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def boot_ratio(a, b, B_=NB):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64); n = len(a)
    if n == 0 or b.mean() == 0: return [None, None]
    r = []
    for _ in range(B_):
        i = RNG.integers(0, n, n); d = b[i].mean(); r.append(a[i].mean() / d if d > 0 else np.nan)
    return [float(np.nanquantile(r, .025)), float(np.nanquantile(r, .975))]


def signflip_p(d, P_=NPERM):
    d = np.asarray(d, np.float64); n = len(d)
    if n == 0 or not (d != 0).any(): return 1.0
    obs = abs(d.mean()); cnt = 0
    for i in range(0, P_, 500):
        s = RNG.choice([-1.0, 1.0], size=(min(500, P_ - i), n)); cnt += int((np.abs((s * d).mean(1)) >= obs - 1e-15).sum())
    return (1 + cnt) / (1 + P_)


# ------------------------------------------------------------------------------------------------ samples
def rows(per, seed, y):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))


def load_samples():
    from jetdistill.config import RESULTS, DATA
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz')
    y = np.asarray(d['y']); X64 = np.asarray(d['x'])
    iM = rows(args.per_class, 0, y); iC = np.setdiff1d(rows(args.conf_per_class, 77, y), iM)
    xs, ys, _ = sample(per_class=args.per_class, seed=0)            # check: identical to the project's sample()
    assert np.array_equal(ys, y[iM]) and np.array_equal(xs, X64[iM].astype(np.float32)); del xs, ys
    out = {}
    for k, idx in (('M', iM), ('C', iC)):
        if N <= 64:
            x = X64[idx, :N].astype(np.float32)
        else:
            V = np.load(DATA / 'val_ptetaphi_f16.npy', mmap_mode='r'); x = np.asarray(V[idx, :N]).astype(np.float32)
            assert np.array_equal(x[:, :64], X64[idx].astype(np.float32)), 'val_ptetaphi rows != full_test rows'
        out[k] = (x, y[idx], idx)
    del X64
    return out


# ------------------------------------------------------------------------------------------------ masks
def qbits(q):
    for a in ('b', 'bits'):
        if hasattr(q, a): return np.asarray(getattr(q, a), np.float64)
    k, i, f = (np.asarray(getattr(q, a), np.float64) for a in ('k', 'i', 'f'))
    return np.maximum(k + i + f, 0)


def quantizers(L, k):
    iq = L[k].iq; return list(iq.quantizers) if hasattr(iq, 'quantizers') else [iq]


def slot_mask(L, k, idx=0):
    """(n, C) kept (> 0 bits) of a per-particle quantizer; also whether its bits vary along the slot axis"""
    b = qbits(quantizers(L, k)[idx].quantizer)
    b2 = b.reshape(b.shape[-2:]) if b.ndim >= 2 else b[None]
    m = np.broadcast_to(b2 > 0, (N, b2.shape[-1])).astype(np.float32)
    return m, dict(bits_shape=list(b.shape), varies_along_slots=bool(b2.shape[0] > 1 and (b2 != b2[:1]).any()))


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


# ------------------------------------------------------------------------------------------------ step 1
def e1(net, x, y, label):
    real = x[..., 0] > 0; Acc = {s: dict(pos=0, s1=0., n=0, lo=np.inf, hi=-np.inf) for s in STAGES}; call = []
    for i in range(0, len(x), B):
        r = net.run(x[i:i + B], keep=STAGES + ('logits',), batch=B); rm = real[i:i + B]
        for s in STAGES:
            v = r[s]; flat = v[rm] if v.ndim == 3 else v; allv = v.reshape(-1, v.shape[-1]); d = Acc[s]
            d['pos'] = d['pos'] + (flat > 0).sum(0); d['s1'] = d['s1'] + flat.sum(0); d['n'] += len(flat)
            d['lo'] = np.minimum(d['lo'], allv.min(0)); d['hi'] = np.maximum(d['hi'], allv.max(0))
        call.append(r['logits'].argmax(1)); del r
    call = np.concatenate(call); out = dict(N=len(x), acc=float((call == y).mean()),
                                             call_frac=(np.bincount(call, minlength=5) / len(call)).tolist(), stages={})
    means = {}
    for s in STAGES:
        d = Acc[s]; W = WIDTH[s]; mean = d['s1'] / d['n']; means[s] = mean; t1 = time.time()
        m32 = mean.astype(np.float32).astype(np.float64); noop = (d['lo'] == m32) & (d['hi'] == m32)
        k_ch = np.zeros(W, int)
        for c in np.flatnonzero(~noop):
            m = np.zeros(W); m[c] = 1
            k_ch[c] = int((net.run(x, override={s: (m, mean)}, batch=B)['logits'].argmax(1) != call).sum())
        n = len(x); ch = k_ch / n
        p = [float(stats.binomtest(int(k), n, 0.01, alternative='greater').pvalue) if not noop[c] else 1.0 for c, k in enumerate(k_ch)]
        ph = C.holm(p)
        ci = [[float(q) for q in np.quantile(RNG.binomial(n, ch[c], NB) / n, [.025, .975])] if ch[c] > 0 else [0., 0.] for c in range(W)]
        live = np.flatnonzero(ch >= 0.01)
        out['stages'][s] = dict(width=W, n_run=int((~noop).sum()), changed=ch.round(6).tolist(), changed_ci=ci, holm_p=ph,
                                live=live.tolist(), n_live=int(len(live)), n_live_holm=int(sum(h < 0.05 for h in ph)),
                                on_frac=(d['pos'] / max(d['n'], 1)).round(4).tolist())
        log(f'{label} E1 {s:3s}: live (>= 1 % calls) {len(live):2d}/{W}, Holm p < 0.05 {out["stages"][s]["n_live_holm"]:2d}; top '
            + ', '.join(f'{c}:{ch[c] * 100:.1f}%' for c in np.argsort(-ch)[:6]) + f' ({int((~noop).sum())} runs, {time.time() - t1:.0f}s)')
    return out, call, means


# ------------------------------------------------------------------------------------------------ step 2
def e2(net, x, live_a):
    K1, b1 = net.KB['l1']; b1 = b1.reshape(-1); K1 = K1.reshape(3, 64)
    alpha = K1 / net.scale[:, None]; beta = b1 - (K1 * (net.shift / net.scale)[:, None]).sum(0)
    xr = x[x[..., 0] > 0].astype(np.float64); sdx = xr.std(0); medpt = float(np.median(xr[:, 0]))
    units, cnt = [], dict(ramp=0, halfplane=0, angle_offset=0, constant=0, mixed=0)
    for c in live_a:
        w = np.abs(alpha[:, c]) * sdx; nw = np.linalg.norm(w); ang = np.hypot(alpha[1, c], alpha[2, c])
        if nw == 0: cat, fa, fp, dist = 'constant', None, None, None
        else:
            fa = float(np.hypot(w[1], w[2]) / nw); fp = float(w[0] / nw)
            dist = float(abs(beta[c] + alpha[0, c] * medpt) / ang) if ang > 0 else None
            if fa < 0.05: cat = 'ramp'
            elif fp < 0.05: cat = 'halfplane' if (dist is not None and dist < 0.01) else 'angle_offset'
            else: cat = 'mixed'
        cnt[cat] += 1
        e = dict(c=int(c), cat=cat, alpha=alpha[:, c].tolist(), beta=float(beta[c]), spread=w.tolist(), frac_angle=fa, frac_pt=fp,
                 line_dist_rad=dist, normal_deg=float(np.degrees(np.arctan2(alpha[2, c], alpha[1, c]))) if ang > 0 else None,
                 pt_threshold_GeV=float(-beta[c] / alpha[0, c]) if (cat == 'ramp' and alpha[0, c] != 0) else None)
        units.append(e)
        log(f'E2 a{c}: {cat:12s} α=({alpha[0, c]:+.4g}/GeV, {alpha[1, c]:+.4g}/rad, {alpha[2, c]:+.4g}/rad) β={beta[c]:+.4g} '
            f'|w_ang|/|w|={fa if fa is None else round(fa, 3)} |w_pT|/|w|={fp if fp is None else round(fp, 3)} dist={dist if dist is None else round(dist, 4)}')
    nl = len(live_a); clean_n = cnt['ramp'] + cnt['halfplane'] + cnt['angle_offset']
    fr = {k: dict(k=v, frac=v / nl if nl else None, wilson=C.wilson(v, nl)) for k, v in cnt.items()}
    fr['pt_or_angle_only'] = dict(k=clean_n, frac=clean_n / nl if nl else None, wilson=C.wilson(clean_n, nl))
    fisher = stats.fisher_exact([[clean_n, nl - clean_n], [13, 0]]) if nl else (None, None)
    return dict(n_live=nl, median_pt_GeV=medpt, counts=cnt, fractions=fr, units=units, n64_counts=N64,
                fisher_vs_n64=dict(table=[[clean_n, nl - clean_n], [13, 0]], p=float(fisher[1]) if nl else None))


# ------------------------------------------------------------------------------------------------ step 3
def e3(L, liveP, liveS):
    out = {}
    for layer, live in (('qs', liveP), ('qs1', liveS)):
        M, info = slot_mask(L, layer); ch, ps = {}, []
        for c in live:
            col = M[:, c]; nk = int(col.sum()); obs = contiguity(col)
            edges = [f'{k}|{k + 1}' for k in range(N - 1) if col[k] != col[k + 1]]
            if nk in (0, N) or obs is None:
                p = 1.0; null = None
            else:
                null = np.array([contiguity(np.isin(np.arange(N), RNG.choice(N, nk, replace=False)).astype(float)) for _ in range(NNULL)])
                p = (1 + int((null >= obs - 1e-12).sum())) / (1 + NNULL)
            ps.append(p)
            ch[int(c)] = dict(kept=nk, runs=runs_of(col), contiguity=obs, null_mean=float(null.mean()) if null is not None else None,
                              null_q95=float(np.quantile(null, .95)) if null is not None else None, p=p, edges=edges)
        hp = C.holm(ps)
        for (c, e), h in zip(ch.items(), hp): e['holm_p'] = h
        nt = sum(1 for e in ch.values() if 0 < e['kept'] < N)
        out[layer] = dict(info, channels=ch, n_channels=len(ch), n_partial=nt,
                          n_contig_holm01=int(sum(1 for e in ch.values() if e['holm_p'] < 0.01)),
                          n_fully_contiguous=int(sum(1 for e in ch.values() if e['contiguity'] == 1.0 and 0 < e['kept'] < N)))
        log(f'E3 {layer}: bits shape {info["bits_shape"]}, varies along slots {info["varies_along_slots"]}; {len(ch)} live channels, '
            f'{nt} partially kept, contiguous (Holm p < 0.01) {out[layer]["n_contig_holm01"]}; '
            + ' '.join(f'{c}:{"/".join(f"{a}-{b}" for a, b in e["runs"])}(p{e["holm_p"]:.3f})' for c, e in list(ch.items())[:24]))
    return out


# ------------------------------------------------------------------------------------------------ float + mask program
LAYER_KEYS = ('l1', 'qs', 'l2', 'dd', 'add', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')
SLOT_LAYERS = ('qs', 'l2', 'add', 'l3', 'qs1')


def prog_masks(L):
    M = {}
    for k in LAYER_KEYS:
        M[k] = [(qbits(q.quantizer) > 0).astype(np.float32) for q in quantizers(L, k)]
    return M


def ones_masks(M):
    O = dict(M)
    for k in SLOT_LAYERS:
        O[k] = [np.ones_like(M[k][0])] + M[k][1:] if k == 'add' else [np.ones_like(m) for m in M[k]]
    return O


def make_prog(net):
    L = net.L; KB = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in net.KB.items()}

    def f(xn, M):
        def dense(k, t, act=True):
            z = jnp.einsum(L[k].equation, t * M[k][0], KB[k][0]) + KB[k][1]
            return jax.nn.relu(z) if act and L[k].activation is not None else z
        def pool(k, a): return jnp.sum(a * M[k][0], axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale
        a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a)
        u = dense('l3', s * M['add'][0] + g * M['add'][1]); S = pool('qs1', u)
        lo = dense('d7', jax.nn.relu(dense('d6', dense('d5', dense('d4', S)), act=False)))
        J = xn.shape[0]; return lo.reshape(J, -1), S.reshape(J, -1)
    F = jax.jit(f)

    def run(x, M):
        Mj = {k: [jnp.asarray(m) for m in v] for k, v in M.items()}; lo, S = [], []
        for i in range(0, len(x), B):
            l_, s_ = F(jnp.asarray(net.inputs(x[i:i + B], N)), Mj); lo.append(np.asarray(l_, np.float64)); S.append(np.asarray(s_, np.float64))
        return np.concatenate(lo), np.concatenate(S)
    return run


# ------------------------------------------------------------------------------------------------ step 4
def swap(x, k):
    nreal = (x[..., 0] > 0).sum(1); J = np.flatnonzero(nreal > k + 1)
    xe = x[J].copy(); xe[:, [k, k + 1], 1:] = xe[:, [k + 1, k], 1:]; return J, xe


def choose_edges(M1, liveS, costS, x, forced=None, minfrac=0.5):
    nreal = (x[..., 0] > 0).sum(1); okk = [k for k in range(N - 1) if (nreal > k + 1).mean() >= minfrac]
    mass = {k: sum(costS[c] for c in liveS if M1[k, c] != M1[k + 1, c]) for k in okk}
    edges = [k for k in sorted(mass, key=lambda k: -mass[k]) if mass[k] > 0][:2] if forced is None else list(forced)
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
        plan = [dict(edge=13, inside=5, units=[int(c) for c in top], mass=0.0, note='no qs1 edge: n64 ranks used')]
    return plan, {int(k): float(v) for k, v in mass.items() if v > 0}


def m_of(dS, units, sd):
    u = [c for c in units if sd[c] > 0]
    return (np.abs(dS[:, u]) / sd[u]).mean(1) if u else np.zeros(len(dS)), u


def window_test(runS, x, plan, label):
    """runS(x) -> S (J, 64). Per plan entry: across vs inside on common jets."""
    S0 = runS(x); sd = S0.std(0); res = []
    for p in plan:
        k, j = p['edge'], p['inside']
        Ja, xa = swap(x, k); Ji, xi = swap(x, j)
        ma, u = m_of(runS(xa) - S0[Ja], p['units'], sd); mi, _ = m_of(runS(xi) - S0[Ji], p['units'], sd)
        com = np.intersect1d(Ja, Ji); a = ma[np.searchsorted(Ja, com)]; b = mi[np.searchsorted(Ji, com)]
        r = dict(edge=k, inside=j, units_used=u, N=len(com), across=float(a.mean()), across_ci=boot_mean(a),
                 inside_mean=float(b.mean()), inside_ci=boot_mean(b), ratio=float(a.mean() / b.mean()) if b.mean() > 0 else None,
                 ratio_ci=boot_ratio(a, b), perm_p=signflip_p(a - b))
        res.append(r)
        log(f'{label} edge {k}|{k + 1} vs inside {j}|{j + 1}: N={len(com)} units {u}: m_across {a.mean():.4f} {r["across_ci"]} '
            f'm_inside {b.mean():.4f} ratio {r["ratio"]} {r["ratio_ci"]} perm p {r["perm_p"]:.2e}')
    return res


def per_unit_window(runS, x, plan):
    S0 = runS(x); sd = S0.std(0); out = []
    for p in plan:
        Ja, xa = swap(x, p['edge']); Ji, xi = swap(x, p['inside']); dA = runS(xa) - S0[Ja]; dI = runS(xi) - S0[Ji]
        out.append({int(c): dict(across=float(np.abs(dA[:, c]).mean() / sd[c]), inside=float(np.abs(dI[:, c]).mean() / sd[c]))
                    for c in p['units'] if sd[c] > 0})
    return out


# ------------------------------------------------------------------------------------------------ step 5
PTG = np.array([1., 3., 10., 30., 100.]); RG = np.linspace(0, 0.4, 41); TH = np.radians(np.arange(0, 360, 5))


def as_slots(arr, C_):
    a = np.asarray(arr, np.float64); a2 = a.reshape(a.shape[-2:]) if a.ndim >= 2 else a.reshape(1, -1)
    return np.broadcast_to(a2, (N, C_))


def l1_range(L):
    q = quantizers(L, 'l1')[0].quantizer
    try:
        k_, i_, f_ = (as_slots(getattr(q, a_), 3) for a_ in ('k', 'i', 'f'))
        return np.where(k_ > 0, -2.0 ** i_, 0.0), 2.0 ** i_ - 2.0 ** -f_
    except Exception as e:  # noqa: BLE001
        log('l1 range table unavailable:', repr(e)); return None, None


def radial_maps(net, units, stage, slots, gmean, LO, HI):
    PT, RR, TT = np.meshgrid(PTG, RG, TH, indexing='ij'); out = {}
    for c, slot in zip(units, slots):
        xs = np.zeros((PT.size, N, 3), np.float32)
        xs[:, slot, 0] = PT.ravel(); xs[:, slot, 1] = (RR * np.cos(TT)).ravel(); xs[:, slot, 2] = (RR * np.sin(TT)).ravel()
        f = np.concatenate([net.run(xs[i:i + B], override={'g': (np.ones(64), gmean)}, keep=(stage,), batch=B)[stage][:, slot, c]
                            for i in range(0, len(xs), B)]).reshape(PT.shape)
        if LO is not None:
            xn = (xs[:, slot].astype(np.float64) - net.shift) / net.scale
            bad = ((xn < LO[slot]) | (xn > HI[slot])).any(1).reshape(PT.shape)
            ring_bad = bad.any(-1)
        else:
            ring_bad = np.zeros(PT.shape[:2], bool)
        fr = f[~ring_bad]                                                   # (rings, θ)
        tot = ((fr - fr.mean()) ** 2).sum() if fr.size else 0.0
        az = ((fr - fr.mean(1, keepdims=True)) ** 2).sum() if fr.size else 0.0
        out[int(c)] = dict(slot=int(slot), radial_share=float(1 - az / tot) if tot > 0 else None, total_var=float(tot / max(fr.size, 1)),
                           rings_used=int((~ring_bad).sum()), rings=int(ring_bad.size),
                           map_pt10=f[2].round(5).tolist())
    return out


def pick_slot(col_keep, l1ok, target=5):
    cand = [s for s in range(N) if col_keep[s] > 0 and l1ok[s]]
    return min(cand, key=lambda s: abs(s - target)) if cand else None


# ------------------------------------------------------------------------------------------------ step 6
DIRS = np.radians(np.arange(0, 360, 45))


def translate(x, d, th):
    e = x.copy(); rl = e[..., 0] > 0
    e[..., 1] += np.float32(d * np.cos(th)) * rl; e[..., 2] += np.float32(d * np.sin(th)) * rl; return e


def rotate_matched(x, d, rng):
    pt = x[..., 0].astype(np.float64); rl = pt > 0; r = np.hypot(x[..., 1], x[..., 2])
    rbar = (pt * r).sum(1) / np.maximum(pt.sum(1), 1e-12)
    phi = 2 * np.arcsin(np.minimum(1.0, d / np.maximum(2 * rbar, 1e-12))) * rng.choice([-1, 1], len(x))
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]; e = x.copy()
    e[..., 1] = np.where(rl, c * x[..., 1] - s * x[..., 2], x[..., 1]); e[..., 2] = np.where(rl, s * x[..., 1] + c * x[..., 2], x[..., 2])
    return e, phi


def translation_test(net, x, y, label, d=0.06):
    c0 = net.run(x, batch=B)['logits'].argmax(1); W = y == 2; nW = int(W.sum())
    base = float((c0[W] == 3).mean())
    xr, phi = rotate_matched(x, d, np.random.default_rng(66)); cr = net.run(xr, batch=B)['logits'].argmax(1); zr = cr[W] == 3
    out = dict(N=len(x), N_W=nW, W_to_Z_unedited=base, W_to_Z_unedited_ci=C.wilson(int((c0[W] == 3).sum()), nW),
               rotation=dict(W_to_Z=float(zr.mean()), ci=C.wilson(int(zr.sum()), nW), calls_changed=float((cr != c0).mean()),
                             median_angle_deg=float(np.degrees(np.median(np.abs(phi))))), directions={})
    for th in DIRS:
        ct = net.run(translate(x, d, th), batch=B)['logits'].argmax(1); zt = ct[W] == 3
        out['directions'][f'{np.degrees(th):.0f}'] = dict(W_to_Z=float(zt.mean()), ci=C.wilson(int(zt.sum()), nW),
                                                          calls_changed=float((ct != c0).mean()), mcnemar_vs_rotation=C.mcnemar(zr, zt))
    v = [e['W_to_Z'] for e in out['directions'].values()]; out['W_to_Z_mean_8dirs'] = float(np.mean(v))
    log(f'{label} translation 0.06: W→Z unedited {base:.3f}; +η {out["directions"]["0"]["W_to_Z"]:.3f}; 8-dir mean {np.mean(v):.3f} '
        f'[{min(v):.3f}, {max(v):.3f}]; matched rotation {zr.mean():.3f} (median {out["rotation"]["median_angle_deg"]:.1f}°)')
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    S = load_samples(); xM, yM, _ = S['M']; xC, yC, iC = S['C']
    log(f'{FAM} n{N}: M N={len(yM)}, C N={len(yC)} (seed 77, M jets removed); real particles per jet M {(xM[..., 0] > 0).sum(1).mean():.1f}')
    net = Jedi(N, FAM); L = net.L
    from jetdistill.config import DATA
    off = json.loads((DATA / 'official_acc.json').read_text()).get(FAM, {}).get(str(N)) if (DATA / 'official_acc.json').exists() else None
    R['model'] = net.path.name; R['official_acc'] = off; R['pool_scale'] = net.pool_scale
    cC = net.run(xC, batch=B)['logits'].argmax(1)
    R['acc_C'] = float((cC == yC).mean()); R['N_C'] = len(yC)
    log(f'model {net.path.name}; official accuracy {off}; accuracy on C {R["acc_C"]:.4f}')

    # step 1
    if 'E1' in R and args.resume:
        log('step 1: reused from', JSON); callM = net.run(xM, batch=B)['logits'].argmax(1)
        gmean = np.array(R['gmean'])
    else:
        R['E1'], callM, means = e1(net, xM, yM, 'trained'); gmean = means['g']; R['gmean'] = gmean.tolist()
        R['acc_M'] = R['E1']['acc']; log(f'step 1 done; accuracy on M {R["acc_M"]:.4f}'); save()
        un = Jedi(N, FAM, untrained=True, seed=0)
        R['E1_untrained'], _, mu = e1(un, xM[::2], yM[::2], 'untrained'); R['gmean_untrained'] = mu['g'].tolist(); del un; save()
    E1 = R['E1']['stages']; R['acc_M'] = R['E1']['acc']
    liveA, liveP, liveS = (E1[s]['live'] for s in ('a', 'P', 'S'))
    costS = {c: E1['S']['changed'][c] for c in range(64)}

    # step 2
    R['E2'] = e2(net, xM, liveA); log('step 2 counts', R['E2']['counts']); save()
    # step 3
    R['E3'] = e3(L, liveP, liveS); save()

    # step 4
    M1, _ = slot_mask(L, 'qs1'); plan, mass = choose_edges(M1, liveS, costS, xM)
    R['E4'] = dict(plan=plan, edge_mass=mass); log('step 4 plan', plan)
    runN = lambda N_: (lambda x: N_.run(x, keep=('S',), batch=B)['S'])
    R['E4']['net_M'] = window_test(runN(net), xM, plan, 'net M')
    R['E4']['per_unit_M'] = per_unit_window(runN(net), xM, plan)
    prog = make_prog(net); PM = prog_masks(L); PO = ones_masks(PM)
    loP, _ = prog(xM, PM); loO, _ = prog(xM, PO)
    R['E4']['program'] = dict(acc_mask=float((loP.argmax(1) == yM).mean()), calls_vs_net_mask=float((loP.argmax(1) != callM).mean()),
                              acc_ones=float((loO.argmax(1) == yM).mean()), calls_vs_net_ones=float((loO.argmax(1) != callM).mean()))
    log('float programs', R['E4']['program'])
    R['E4']['prog_mask_M'] = window_test(lambda x: prog(x, PM)[1], xM, plan, 'prog masks M')
    R['E4']['prog_ones_M'] = window_test(lambda x: prog(x, PO)[1], xM, plan, 'prog ones M')
    R['E4']['net_C'] = window_test(runN(net), xC, plan, 'net C'); save()
    un = Jedi(N, FAM, untrained=True, seed=0)
    R['E4']['untrained_M'] = window_test(runN(un), xM[::2], plan, 'untrained M[::2]'); save()
    if args.extra_edges:
        ex, _ = choose_edges(M1, liveS, costS, xM, forced=[int(k) for k in args.extra_edges.split(',')], minfrac=0.2)
        X = dict(note='post hoc, not in the frozen plan: edges shared by the most live S units', plan=ex)
        log('post hoc edges', ex)
        X['net_M'] = window_test(runN(net), xM, ex, 'post hoc net M'); X['per_unit_M'] = per_unit_window(runN(net), xM, ex)
        X['prog_mask_M'] = window_test(lambda x: prog(x, PM)[1], xM, ex, 'post hoc prog masks M')
        X['prog_ones_M'] = window_test(lambda x: prog(x, PO)[1], xM, ex, 'post hoc prog ones M')
        X['net_C'] = window_test(runN(net), xC, ex, 'post hoc net C'); R['E4']['post_hoc'] = X; save()
    del prog; jax.clear_caches(); import gc; gc.collect()

    # step 5
    LO, HI = l1_range(L); m_l1, _ = slot_mask(L, 'l1'); l1ok = m_l1.min(1) > 0
    topS = sorted(liveS, key=lambda c: -costS[c])[:5]
    slots = [pick_slot(M1[:, c], l1ok) for c in topS]; keepS = [(c, s) for c, s in zip(topS, slots) if s is not None]
    R['E5'] = dict(S=radial_maps(net, [c for c, _ in keepS], 'u', [s for _, s in keepS], gmean, LO, HI))
    sA = pick_slot(np.ones(N), l1ok); R['E5']['a_controls'] = radial_maps(net, liveA, 'a', [sA] * len(liveA), gmean, LO, HI)
    sdS_un = None
    try:
        sdS_un = un.run(xM[::4], keep=('S',), batch=B)['S'].std(0); topU = list(np.argsort(-sdS_un)[:5])
        slotsU = [pick_slot(M1[:, c], l1ok) for c in topU]; kU = [(c, s) for c, s in zip(topU, slotsU) if s is not None]
        R['E5']['untrained_S'] = radial_maps(un, [c for c, _ in kU], 'u', [s for _, s in kU], np.array(R['gmean_untrained']), LO, HI)
    except Exception as e:  # noqa: BLE001
        log('untrained radial maps failed', repr(e))
    log('step 5 radial share S: ' + ', '.join(f'S{c}@{e["slot"]}:{e["radial_share"]}' for c, e in R['E5']['S'].items()))
    log('step 5 l1 controls: ' + ', '.join(f'a{c}:{e["radial_share"] if e["radial_share"] is None else round(e["radial_share"], 3)}' for c, e in R['E5']['a_controls'].items()))
    if 'untrained_S' in R['E5']:
        log('step 5 untrained S: ' + ', '.join(f'S{c}:{e["radial_share"]}' for c, e in R['E5']['untrained_S'].items()))
    save()

    # step 6
    R['E6'] = dict(M=translation_test(net, xM, yM, 'net M'), C=translation_test(net, xC, yC, 'net C'),
                   untrained_M=translation_test(un, xM[::2], yM[::2], 'untrained M[::2]'))
    R['wall_s'] = time.time() - T0
    import resource
    R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    save(); log(f'done; wrote {JSON}; peak RSS {R["peak_rss_mib"]:.0f} MiB')


if __name__ == '__main__':
    main()
