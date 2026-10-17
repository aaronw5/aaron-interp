"""X6 verification (fresh sample, extra controls). One model per call:  python -m interp.x6_verify --family F --n N
then  python -m interp.x6_verify --combine  (paired n64 vs perminv accuracy from the stored calls).

Sample V = sample(per_class=1500, seed=4242) minus every jet of X6's M (sample(4000, 0)); calls also on M and on X6's C
(sample(2000, 77) minus M) for the paired accuracy test.  Reads X6's per-model JSON ($AI_RESULTS/jedi/x6_<fam>_n<n>.json) for the
live sets, the E4 plans (frozen + post hoc), the E5 units/slots and g mean; never refits them.
Checks:
 V1  accuracy on V; calls on M, C, V stored.
 V2  E2 null: exact-zero pattern of the fused l1 kernel (3 x 64). Statistic: fraction of units (live set; all units with a nonzero
     weight) that are pT-only or angle-only. Null: each feature's zero/nonzero column permuted independently over the 64 units
     (per-feature sparsity kept), 20,000 draws; one-sided p = P(null >= obs).  Also the untrained redraw's zero pattern.
 V3  E20 window test on V at every X6 plan edge (frozen and post hoc): network; float + masks; masks removed as in X6 (slot layers
     set to one, channel pruning removed too); slot-uniform masks (each channel kept at every slot iff kept at any slot: removes
     the slot dependence only, keeps channel pruning, l1 included).
 V4  float-program accuracy: masks / X6 ones / slot-uniform / ones on real slots only (padded slots keep their masks) /
     slot-uniform on real slots only.  Separates the cost of removing rank windows for real particles from the cost of letting
     padded slots (input (0 - shift)/scale != 0) into the sums.
 V5  radial share of X6's top-5 S units at X6's slots: X6's definition (total variance includes the pT dependence) and the
     within-pT definition (pT means removed: 1 - sum_(pT,r) sum_theta (f - ring mean)^2 / sum_pT sum_(r,theta) (f - pT mean)^2).
     Null: random combinations of the model's live angle-only l1 maps at the same slot: (i) Dirichlet(1) non-negative weights,
     (ii) ReLU(sum w_c a_c + b), w ~ N(0, 1), b = -median of the sum over the grid; 2000 draws each.
 V6  translation 0.06 rad along +eta vs matched rotation on V (W -> Z, Wilson, McNemar).
Writes $AI_RESULTS/jedi/x6_verify_<fam>_n<n>.json and x6_verify_combined.json.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument('--family', default='3-feature'); ap.add_argument('--n', type=int, default=64)
ap.add_argument('--combine', action='store_true')
ap.add_argument('--only56', action='store_true', help='resume: V5 + V6 only (V sample only), into the existing JSON; grid batch 250')
args = ap.parse_args()
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
RNG = np.random.default_rng(4243)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    return o


def mcnemar(a, b):
    from scipy import stats
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5)))
    lp = float(min(0.0, (np.log(2) + stats.binom.logcdf(min(n10, n01), n, 0.5)) / np.log(10))) if n else None
    return dict(n10=n10, n01=n01, p=p, log10p=lp)


def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def combine():
    R = {t: json.loads((OUT / f'x6_verify_{t}.json').read_text()) for t in ('3-feature_n64', '3-feature-perminv_n64')}
    y = {k: np.array(v) for k, v in R['3-feature_n64']['labels'].items()}; out = {}
    for k in ('M', 'C', 'V'):
        a = np.array(R['3-feature_n64']['calls'][k]) == y[k]; b = np.array(R['3-feature-perminv_n64']['calls'][k]) == y[k]
        d = b.astype(float) - a.astype(float); bs = [d[RNG.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        out[k] = dict(N=len(d), acc_n64=float(a.mean()), acc_perminv=float(b.mean()), diff=float(d.mean()),
                      diff_ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], mcnemar=mcnemar(a, b),
                      H0='P(perminv right, n64 wrong) = P(n64 right, perminv wrong)')
        log(k, out[k])
    (OUT / 'x6_verify_combined.json').write_text(json.dumps(clean(out), indent=1))


if args.combine:
    combine(); sys.exit(0)

import jax, jax.numpy as jnp
from interp.jedi_core import Jedi, sample
N = args.n; FAM = args.family; TAG = f'{FAM}_n{N}'; B = 2500 if N <= 64 else 1000
X6 = json.loads((OUT / f'x6_{TAG}.json').read_text())
R = dict(family=FAM, n=N, x6_json=f'x6_{TAG}.json')
if args.only56 and (OUT / f'x6_verify_{TAG}.json').exists(): R = json.loads((OUT / f'x6_verify_{TAG}.json').read_text())
GB = 250 if args.only56 else B
def save(): (OUT / f'x6_verify_{TAG}.json').write_text(json.dumps(clean(R), indent=1))


# ------------------------------------------------------------------------------------------------ samples
def rows(y, per, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))


def load():
    from jetdistill.config import RESULTS, DATA
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz')['y'])
    iM = rows(y, 4000, 0); out = {}
    for k, per, seed in ((('V', 1500, 4242),) if args.only56 else (('M', 4000, 0), ('C', 2000, 77), ('V', 1500, 4242))):
        x, ys, _ = sample(per_class=per, seed=seed); idx = rows(y, per, seed); assert np.array_equal(ys, y[idx])
        keep = np.ones(len(idx), bool) if k == 'M' else ~np.isin(idx, iM)
        x, idx = x[keep], idx[keep]
        if N < 64: x = x[:, :N]
        elif N > 64:
            Vv = np.load(DATA / 'val_ptetaphi_f16.npy', mmap_mode='r'); x128 = np.asarray(Vv[idx, :N]).astype(np.float32)
            assert np.array_equal(x128[:, :64], x), 'val rows != full_test rows'; x = x128
        out[k] = (np.ascontiguousarray(x), y[idx]); log(f'sample {k}: per_class {per}, seed {seed}, N {len(idx)} (M removed: {k != "M"})')
    return out


# ------------------------------------------------------------------------------------------------ masks / programs
def qbits(q):
    for a in ('b', 'bits'):
        if hasattr(q, a): return np.asarray(getattr(q, a), np.float64)
    k, i, f = (np.asarray(getattr(q, a), np.float64) for a in ('k', 'i', 'f')); return np.maximum(k + i + f, 0)


def quantizers(L, k):
    iq = L[k].iq; return list(iq.quantizers) if hasattr(iq, 'quantizers') else [iq]


LAYER_KEYS = ('l1', 'qs', 'l2', 'dd', 'add', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')
SLOT_LAYERS = ('qs', 'l2', 'add', 'l3', 'qs1')


def is_slot(m): return m.ndim >= 2 and m.shape[-2] == N and N > 1


def masks(L):
    M = {k: [(qbits(q.quantizer) > 0).astype(np.float32) for q in quantizers(L, k)] for k in LAYER_KEYS}
    O = dict(M)
    for k in SLOT_LAYERS: O[k] = [np.ones_like(M[k][0])] + M[k][1:] if k == 'add' else [np.ones_like(m) for m in M[k]]
    U = dict(M)
    for k in SLOT_LAYERS + ('l1',):
        U[k] = [np.broadcast_to(m.max(axis=-2, keepdims=True), m.shape).astype(np.float32).copy() if is_slot(m) else m for m in M[k]]
    info = {k: [dict(shape=list(m.shape), slot=bool(is_slot(m)), zero_frac=float((m == 0).mean()),
                     slot_varying=bool(is_slot(m) and (m != m.max(axis=-2, keepdims=True)).any())) for m in M[k]] for k in LAYER_KEYS}
    return M, O, U, info


def make_prog(net):
    L = net.L; KB = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in net.KB.items()}

    def f(xn, real, Ma, Mb):
        r3 = real[:, :, None]
        def mk(k, i):
            a, b = Ma[k][i], Mb[k][i]
            return r3 * b + (1 - r3) * a if (a.ndim >= 2 and a.shape[-2] == N) else b
        def dense(k, t, act=True):
            z = jnp.einsum(L[k].equation, t * mk(k, 0), KB[k][0]) + KB[k][1]
            return jax.nn.relu(z) if act and L[k].activation is not None else z
        def pool(k, a): return jnp.sum(a * mk(k, 0), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale
        a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a)
        u = dense('l3', s * mk('add', 0) + g * Ma['add'][1]); S = pool('qs1', u)
        lo = dense('d7', jax.nn.relu(dense('d6', dense('d5', dense('d4', S)), act=False)))
        J = xn.shape[0]; return lo.reshape(J, -1), S.reshape(J, -1)
    F = jax.jit(f)

    def run(x, Ma, Mb=None):
        Mb = Ma if Mb is None else Mb
        A = {k: [jnp.asarray(m) for m in v] for k, v in Ma.items()}; Bm = {k: [jnp.asarray(m) for m in v] for k, v in Mb.items()}
        lo, S = [], []
        for i in range(0, len(x), B):
            xb = x[i:i + B]; l_, s_ = F(jnp.asarray(net.inputs(xb, N)), jnp.asarray((xb[..., 0] > 0).astype(np.float32)), A, Bm)
            lo.append(np.asarray(l_, np.float64)); S.append(np.asarray(s_, np.float64))
        return np.concatenate(lo), np.concatenate(S)
    return run


# ------------------------------------------------------------------------------------------------ window test
def swap(x, k):
    nreal = (x[..., 0] > 0).sum(1); J = np.flatnonzero(nreal > k + 1)
    xe = x[J].copy(); xe[:, [k, k + 1], 1:] = xe[:, [k + 1, k], 1:]; return J, xe


def boot_ratio(a, b, nb=2000):
    r = []
    for _ in range(nb):
        i = RNG.integers(0, len(a), len(a)); d = b[i].mean(); r.append(a[i].mean() / d if d > 0 else np.nan)
    return [float(np.nanquantile(r, .025)), float(np.nanquantile(r, .975))] if np.isfinite(r).any() else [None, None]


def signflip_p(d, P=10000):
    if not (d != 0).any(): return 1.0
    obs = abs(d.mean()); cnt = 0
    for i in range(0, P, 500):
        s = RNG.choice([-1.0, 1.0], size=(min(500, P - i), len(d))); cnt += int((np.abs((s * d).mean(1)) >= obs - 1e-15).sum())
    return (1 + cnt) / (1 + P)


def window(runS, x, plan):
    S0 = runS(x); sd = S0.std(0); res = []
    for p in plan:
        k, j = p['edge'], p['inside']; u = [c for c in p['units'] if sd[c] > 0]
        Ja, xa = swap(x, k); Ji, xi = swap(x, j)
        if not u: res.append(dict(edge=k, inside=j, units_used=[], ratio=None)); continue
        ma = (np.abs(runS(xa) - S0[Ja])[:, u] / sd[u]).mean(1); mi = (np.abs(runS(xi) - S0[Ji])[:, u] / sd[u]).mean(1)
        com = np.intersect1d(Ja, Ji); a = ma[np.searchsorted(Ja, com)]; b = mi[np.searchsorted(Ji, com)]
        res.append(dict(edge=k, inside=j, units_used=u, N=len(com), across=float(a.mean()), inside_mean=float(b.mean()),
                        ratio=float(a.mean() / b.mean()) if b.mean() > 0 else None, ratio_ci=boot_ratio(a, b) if b.mean() > 0 else None,
                        perm_p=signflip_p(a - b)))
    return res


# ------------------------------------------------------------------------------------------------ radial
PTG = np.array([1., 3., 10., 30., 100.]); RG = np.linspace(0, 0.4, 41); TH = np.radians(np.arange(0, 360, 5))


def l1_range(L):
    q = quantizers(L, 'l1')[0].quantizer
    try:
        k_, i_, f_ = (np.broadcast_to(np.asarray(getattr(q, a_), np.float64).reshape(-1, 3) if np.asarray(getattr(q, a_)).size > 3
                                      else np.asarray(getattr(q, a_), np.float64).reshape(1, 3), (N, 3)) for a_ in ('k', 'i', 'f'))
        return np.where(k_ > 0, -2.0 ** i_, 0.0), 2.0 ** i_ - 2.0 ** -f_
    except Exception as e:  # noqa: BLE001
        log('l1 range unavailable', repr(e)); return None, None


def grid_maps(net, slot, stage, gmean):
    PT, RR, TT = np.meshgrid(PTG, RG, TH, indexing='ij')
    xs = np.zeros((PT.size, N, 3), np.float32)
    xs[:, slot, 0] = PT.ravel(); xs[:, slot, 1] = (RR * np.cos(TT)).ravel(); xs[:, slot, 2] = (RR * np.sin(TT)).ravel()
    f = np.concatenate([net.run(xs[i:i + GB], override={'g': (np.ones(64), gmean)}, keep=(stage,), batch=GB)[stage][:, slot, :]
                        for i in range(0, len(xs), GB)])
    return f.reshape(PT.shape + (f.shape[-1],)), xs


def shares(f, ring_bad):
    """f (pT, r, θ); returns X6 share (grand mean) and within-pT share"""
    out = {}
    m = ~ring_bad; fr = f[m]
    tot = ((fr - fr.mean()) ** 2).sum(); az = ((fr - fr.mean(1, keepdims=True)) ** 2).sum()
    out['share_x6'] = float(1 - az / tot) if tot > 1e-18 else None
    wp = 0.0
    for ip in range(f.shape[0]):
        fp = f[ip][m[ip]]
        if fp.size: wp += ((fp - fp.mean()) ** 2).sum()
    out['share_within_pt'] = float(1 - az / wp) if wp > 1e-18 else None
    out['pt_var_frac'] = float(1 - wp / tot) if tot > 1e-18 else None
    return out


def share_ang(f2):
    """f2 (r, θ) pT-independent map"""
    tot = ((f2 - f2.mean()) ** 2).sum(); az = ((f2 - f2.mean(1, keepdims=True)) ** 2).sum()
    return float(1 - az / tot) if tot > 1e-18 else np.nan


# ------------------------------------------------------------------------------------------------ main
def main():
    S = load(); xV, yV = S['V']
    net = Jedi(N, FAM); L = net.L
    if args.only56:
        callV = net.run(xV, batch=B)['logits'].argmax(1)
    else:
        R['labels'] = {k: v[1] for k, v in S.items()}; R['calls'] = {}
        for k, (x, y) in S.items():
            c = net.run(x, batch=B)['logits'].argmax(1); R['calls'][k] = c; R[f'acc_{k}'] = float((c == y).mean()); R[f'N_{k}'] = len(y)
        callV = np.asarray(R['calls']['V']); log('accuracy', {k: R[f'acc_{k}'] for k in S}); save()
        del S['M'], S['C']

        # V2 E2 null
        K1 = net.KB['l1'][0].reshape(3, 64); nz = K1 != 0; liveA = X6['E1']['stages']['a']['live']
        def pure(nzm, units):
            u = [c for c in units if nzm[:, c].any()]
            if not u: return np.nan
            mixed = sum(1 for c in u if nzm[0, c] and nzm[1:, c].any()); return 1 - mixed / len(u)
        allU = list(range(64)); obs_live, obs_all = pure(nz, liveA), pure(nz, allU)
        nl, na = [], []
        for _ in range(20000):
            P = np.stack([nz[f][RNG.permutation(64)] for f in range(3)]); nl.append(pure(P, liveA)); na.append(pure(P, allU))
        nl, na = np.array(nl), np.array(na)
        un = Jedi(N, FAM, untrained=True, seed=0); K1u = un.KB['l1'][0].reshape(3, 64)
        R['V2'] = dict(nonzero_per_feature=nz.sum(1).tolist(), units_any_nonzero=int(nz.any(0).sum()),
                       pattern_counts={p: int(sum(1 for c in range(64) if ''.join('1' if v else '0' for v in nz[:, c]) == p)) for p in
                                       ('000', '100', '010', '001', '011', '110', '101', '111')},
                       live=dict(n=len(liveA), obs=obs_live, null_mean=float(np.nanmean(nl)), null_q=[float(np.nanquantile(nl, q)) for q in (.025, .5, .975)],
                                 p=float((1 + (nl >= obs_live - 1e-12).sum()) / (1 + len(nl)))),
                       all_units=dict(obs=obs_all, null_mean=float(np.nanmean(na)), null_q=[float(np.nanquantile(na, q)) for q in (.025, .5, .975)],
                                      p=float((1 + (na >= obs_all - 1e-12).sum()) / (1 + len(na)))),
                       untrained_same_zero_pattern=bool(np.array_equal(K1u != 0, nz)), untrained_nonzero_per_feature=(K1u != 0).sum(1).tolist())
        del un; log('V2', R['V2']); save()

        # V3 / V4 programs
        prog = make_prog(net); M, O, U, info = masks(L); R['mask_info'] = info
        acc = {}
        for name, (Ma, Mb) in dict(mask=(M, M), ones_x6=(O, O), slot_uniform=(U, U), ones_real_only=(M, O), uniform_real_only=(M, U)).items():
            lo, _ = prog(xV, Ma, Mb); c = lo.argmax(1); acc[name] = dict(acc=float((c == yV).mean()), calls_vs_net=float((c != callV).mean()))
        R['V4'] = acc; log('V4 program accuracy', acc); save()
        plan = list(X6['E4']['plan']) + (list(X6['E4']['post_hoc']['plan']) if 'post_hoc' in X6['E4'] else [])
        runN = lambda x: net.run(x, keep=('S',), batch=B)['S']
        R['V3'] = dict(plan=plan, net=window(runN, xV, plan), mask=window(lambda x: prog(x, M)[1], xV, plan),
                       ones_x6=window(lambda x: prog(x, O)[1], xV, plan), slot_uniform=window(lambda x: prog(x, U)[1], xV, plan))
        for i, p in enumerate(plan):
            log(f"V3 edge {p['edge']}|{p['edge'] + 1} vs {p['inside']}|{p['inside'] + 1} units {p['units']}: " +
                '; '.join(f"{k} {R['V3'][k][i].get('ratio')} {R['V3'][k][i].get('ratio_ci')} p {R['V3'][k][i].get('perm_p')}" for k in ('net', 'mask', 'ones_x6', 'slot_uniform')))
        save(); del prog; jax.clear_caches()

    # V5 radial
    gmean = np.array(X6['gmean']); LO, HI = l1_range(L); E5 = X6['E5']['S']
    slots = sorted({e['slot'] for e in E5.values()}); R['V5'] = dict(S={}, null={})
    angle_only = [u['c'] for u in X6['E2']['units'] if u['cat'] in ('halfplane', 'angle_offset')]
    for slot in slots:
        fu, xs = grid_maps(net, slot, 'u', gmean)
        if LO is not None:
            xn = (xs[:, slot].astype(np.float64) - net.shift) / net.scale
            rb = ((xn < LO[slot]) | (xn > HI[slot])).any(1).reshape(fu.shape[:3]).any(-1)
        else: rb = np.zeros(fu.shape[:2], bool)
        for c, e in E5.items():
            if e['slot'] != slot: continue
            d = shares(fu[..., int(c)], rb); d.update(slot=slot, x6_share=e['radial_share'], var_pt10=float(fu[2, :, :, int(c)].var()))
            R['V5']['S'][c] = d; log(f'V5 S{c}@{slot}', d)
        fa, _ = grid_maps(net, slot, 'a', gmean); A = fa[2][..., angle_only]          # (r, θ, units), pT 10 (angle-only: pT-free)
        A = A[~rb[2]] if rb[2].any() else A
        own = [share_ang(A[..., j]) for j in range(A.shape[-1])]
        dsh, rsh = [], []
        for _ in range(2000):
            w = RNG.dirichlet(np.ones(len(angle_only))); dsh.append(share_ang(A @ w))
            w = RNG.normal(size=len(angle_only)); s_ = A @ w; rsh.append(share_ang(np.maximum(s_ - np.median(s_), 0)))
        dsh, rsh = np.array(dsh), np.array(rsh)
        eq = share_ang(A.sum(-1))
        R['V5']['null'][str(slot)] = dict(n_angle_units=len(angle_only), own=own, equal_weights=eq,
                                          dirichlet=dict(q=[float(np.nanquantile(dsh, q)) for q in (.025, .5, .975)], frac_ge_097=float(np.nanmean(dsh >= 0.97)),
                                                         max=float(np.nanmax(dsh))),
                                          relu_random=dict(q=[float(np.nanquantile(rsh, q)) for q in (.025, .5, .975)], frac_ge_097=float(np.nanmean(rsh >= 0.97)),
                                                           max=float(np.nanmax(rsh))))
        log(f'V5 null slot {slot}', R['V5']['null'][str(slot)])
    save()

    # V6 translation
    W = yV == 2; nW = int(W.sum()); e = xV.copy(); rl = e[..., 0] > 0; e[..., 1] += np.float32(0.06) * rl
    ct = net.run(e, batch=B)['logits'].argmax(1); zt = ct[W] == 3
    pt = xV[..., 0].astype(np.float64); r = np.hypot(xV[..., 1], xV[..., 2]); rbar = (pt * r).sum(1) / np.maximum(pt.sum(1), 1e-12)
    phi = 2 * np.arcsin(np.minimum(1.0, 0.06 / np.maximum(2 * rbar, 1e-12))) * np.random.default_rng(67).choice([-1, 1], len(xV))
    cph, sph = np.cos(phi)[:, None], np.sin(phi)[:, None]; xr = xV.copy()
    xr[..., 1] = np.where(rl, cph * xV[..., 1] - sph * xV[..., 2], xV[..., 1]); xr[..., 2] = np.where(rl, sph * xV[..., 1] + cph * xV[..., 2], xV[..., 2])
    cr = net.run(xr, batch=B)['logits'].argmax(1); zr = cr[W] == 3
    R['V6'] = dict(N_W=nW, unedited=float((callV[W] == 3).mean()), plus_eta=float(zt.mean()), plus_eta_ci=wilson(int(zt.sum()), nW),
                   rotation=float(zr.mean()), rotation_ci=wilson(int(zr.sum()), nW), mcnemar=mcnemar(zr, zt))
    log('V6', R['V6']); R['wall_s'] = time.time() - T0
    import resource; R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; save(); log('done')


main()
