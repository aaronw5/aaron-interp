"""P33b: input-level geometry of the P33 single-particle and neighbour-holding moves (loop C, round C7; CPU only).

Question: is the non-smooth, even part of ParT's pair-route response (P27v, P33) produced by the pair-embedding hinges, or is
it already present in the pair inputs, whose ln dR has a log singularity at dR -> 0 (a 0.01 rad step is comparable to the
nearest-neighbour distance of half of the moved particles)?
Input: the P33 npz files (same jets, same moved particles, same unedited logits), nothing re-run through a network.
For each moved particle i (10 leading stored particles per jet) and step delta in (0.01, 0.003) along its radial direction,
and for the P33 rigid cluster move (every particle within dR < 0.05 moved with it), we compute for every other real particle j
  D_j(+-delta) = ln dR_ij(edited) - ln dR_ij(unedited)      (ln kT and ln m2 change by the same and by twice this amount)
  input even  E_in = sum_j [(D_j(+) + D_j(-)) / 2]^2,  input odd  O_in = sum_j [(D_j(+) - D_j(-)) / 2]^2.
Compared with the network's even/odd pair-route response from P33 ('pair' and float64 'pair64' conditions):
  1) input smoothness ratio (E_in/O_in at 0.003) / (E_in/O_in at 0.01) per nearest-neighbour-dR bin vs the network's;
  2) Spearman of |even pair response| with sqrt(E_in) and with the layer-1 near-collinear hinge-crossing count, and the
     partial rank correlation with the crossing count at fixed sqrt(E_in) (bootstrap CI over jets);
  3) input cluster / single ratio of E_in vs the network's even cluster / single ratio.
  AI_PY=conda ./job.sh c-p33b-geo 'python -m interp.p33b_input_geometry'
  -> $AI_RESULTS/part/p33b_input_geometry.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
ETYPES = ['base', 'phi', 'dil', 's+', 's-', 's3+', 's3-', 'c+', 'c-', 'n+', 'n-']
DR, DR3, RCL, SEED, NB = 0.01, 0.003, 0.05, 3316, 1000
EDGES = [0, 0.003, 0.01, 0.03, 0.05, np.inf]
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def input_terms(pt, de, dp, meta_s):
    """per moved particle: E_in, O_in at 0.01 and 0.003, E_in for the cluster move; same row order as meta_s"""
    out = np.zeros((len(meta_s), 6))
    for r, (j, i) in enumerate(meta_s[:, [0, 2]].astype(int)):
        m = pt[j] > 0; e0, p0 = de[j][m].astype(np.float64), dp[j][m].astype(np.float64)
        rr = np.hypot(e0[i], p0[i]); ue, up = (e0[i] / rr, p0[i] / rr) if rr > 1e-6 else (1.0, 0.0)
        dist = np.hypot(e0 - e0[i], p0 - p0[i]); oth = np.arange(len(e0)) != i; ln0 = np.log(np.maximum(dist[oth], 1e-8))
        cl = dist < RCL; outside = ~cl
        def D(step):
            d = np.hypot(e0[oth] - (e0[i] + step * ue), p0[oth] - (p0[i] + step * up)); return np.log(np.maximum(d, 1e-8)) - ln0
        res = []
        for s in (DR, DR3):
            a, b = D(s), D(-s); res += [(((a + b) / 2) ** 2).sum(), (((a - b) / 2) ** 2).sum()]
        # rigid cluster move: only pairs (k in cluster, l outside) change
        k_in, l_out = np.where(cl)[0], np.where(outside)[0]
        if len(l_out):
            dd0 = np.hypot(e0[k_in, None] - e0[None, l_out], p0[k_in, None] - p0[None, l_out]); l0 = np.log(np.maximum(dd0, 1e-8))
            Dc = lambda s: np.log(np.maximum(np.hypot(e0[k_in, None] + s * ue - e0[None, l_out], p0[k_in, None] + s * up - p0[None, l_out]), 1e-8)) - l0
            a, b = Dc(DR), Dc(-DR); ec, oc = (((a + b) / 2) ** 2).sum(), (((a - b) / 2) ** 2).sum()
        else: ec = oc = 0.0
        out[r] = res + [ec, oc]
    return out


def rank(a): return np.argsort(np.argsort(a)).astype(float)


def partial_spearman(y, x, z):
    """rank correlation of y with x at fixed z (linear residuals of the ranks)"""
    ry, rx, rz = rank(y), rank(x), rank(z); A = np.c_[np.ones_like(rz), rz]
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    a, b = res(ry), res(rx); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def model(mn, geo_cache):
    Z = np.load(OUTD / f'p33_pair_channels_{mn}.npz'); meta, lg, cr = Z['meta'], Z['lg'].astype(np.float64), Z['cr']
    cond = [str(c) for c in Z['cond']]; C = {c: k for k, c in enumerate(cond)}
    jet, et = meta[:, 0].astype(int), meta[:, 1].astype(int); N = len(Z['y'])
    sc = lg[:, :, 1] - lg[:, :, 0]; ib = np.where(et == 0)[0]; brow = np.full(N, -1); brow[jet[ib]] = ib
    d = sc - sc[brow[jet]]
    idx = {t: np.where(et == ETYPES.index(t))[0] for t in ETYPES[3:9]}
    I = idx['s+']; ms = meta[I]; jr = jet[I]
    key = (N, int(len(I)))
    if key not in geo_cache:
        geo_cache[key] = input_terms(Z['pt'], Z['de'], Z['dp'], ms); log(f'{mn}: input terms for {len(I)} particles')
    G = geo_cache[key]; Ein1, Oin1, Ein3, Oin3, Ecl, Ocl = G.T
    nn, cs = ms[:, 4], ms[:, 3]; ncx = cr[idx['s+'], 1] + cr[idx['s-'], 1]
    rng = np.random.default_rng(SEED + 11); Wb = np.zeros((NB, N))
    for b in range(NB): Wb[b] = np.bincount(rng.integers(0, N, N), minlength=N)
    S = lambda v, msk=None: np.bincount(jr if msk is None else jr[msk], v, minlength=N)
    st = lambda num, den: dict(v=float(num.sum() / den.sum()), ci=[float(q) for q in np.percentile((Wb @ num) / (Wb @ den), [2.5, 97.5])])
    R = dict(n_particles=int(len(I)), n_jets=int(N))
    # input-level smoothness ratio, overall and per nn bin
    def sm_in(msk):
        e1, o1, e3, o3 = S(Ein1[msk], msk), S(Oin1[msk], msk), S(Ein3[msk], msk), S(Oin3[msk], msk)
        v = (e3.sum() / o3.sum()) / (e1.sum() / o1.sum()); bb = ((Wb @ e3) / (Wb @ o3)) / ((Wb @ e1) / (Wb @ o1))
        return dict(v=float(v), ci=[float(q) for q in np.nanpercentile(bb, [2.5, 97.5])])
    allm = np.ones(len(I), bool)
    R['input_smooth_ratio'] = dict(all=sm_in(allm))
    out = {}
    for c in ('pair', 'pair64'):
        k = C[c]; ev = lambda a, b: (d[idx[a], k] + d[idx[b], k]) / 2; od = lambda a, b: (d[idx[a], k] - d[idx[b], k]) / 2
        e1, o1, e3, o3 = ev('s+', 's-'), od('s+', 's-'), ev('s3+', 's3-'), od('s3+', 's3-'); ecl = ev('c+', 'c-')
        def sm_net(msk):
            a1, b1, a3, b3 = (S(v[msk] ** 2, msk) for v in (e1, o1, e3, o3))
            v = (a3.sum() / b3.sum()) / (a1.sum() / b1.sum()); bb = ((Wb @ a3) / (Wb @ b3)) / ((Wb @ a1) / (Wb @ b1))
            return dict(v=float(v), ci=[float(q) for q in np.nanpercentile(bb, [2.5, 97.5])])
        o = dict(net_smooth_ratio=dict(all=sm_net(allm)), bins={})
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            msk = (nn >= lo) & (nn < hi); lab = f'nn_dR_{lo}_{hi}'
            o['bins'][lab] = dict(n=int(msk.sum()), net=sm_net(msk), inp=sm_in(msk), net_even_ms_share=float((e1[msk] ** 2).sum() / (e1 ** 2).sum()),
                                  inp_even_share=float(Ein1[msk].sum() / Ein1.sum()))
        ae = np.abs(e1); x = np.sqrt(Ein1)
        from scipy.stats import spearmanr
        o['spearman_absEven_vs_sqrtEin'] = float(spearmanr(ae, x).correlation)
        o['spearman_absEven_vs_ncCross'] = float(spearmanr(ae, ncx).correlation)
        o['spearman_ncCross_vs_sqrtEin'] = float(spearmanr(ncx, x).correlation)
        pv = partial_spearman(ae, ncx, x); bs = []
        for b in range(200):
            jj = rng.integers(0, N, N); w = np.bincount(jj, minlength=N)[jr]; sel = np.repeat(np.arange(len(I)), w)
            bs.append(partial_spearman(ae[sel], ncx[sel], x[sel]))
        o['partial_spearman_absEven_ncCross_given_sqrtEin'] = dict(v=pv, ci=[float(q) for q in np.percentile(bs, [2.5, 97.5])],
                                                                   p_boot_le_0=float(max(1 / 200, (np.array(bs) <= 0).mean())))
        msk = cs >= 2
        o['cluster_over_single_even'] = dict(net=st(S(ecl[msk] ** 2, msk), S(e1[msk] ** 2, msk)), inp=st(S(Ecl[msk], msk), S(Ein1[msk], msk)))
        out[c] = o
    R['cond'] = out
    log(f"{mn}: input smooth {R['input_smooth_ratio']['all']['v']:.3f}; net pair {out['pair']['net_smooth_ratio']['all']['v']:.3f}, pair64 "
        f"{out['pair64']['net_smooth_ratio']['all']['v']:.3f}; cluster/single net {out['pair64']['cluster_over_single_even']['net']['v']:.3f} "
        f"input {out['pair64']['cluster_over_single_even']['inp']['v']:.3f}; partial rho {out['pair64']['partial_spearman_absEven_ncCross_given_sqrtEin']['v']:.3f}")
    return R


if __name__ == '__main__':
    res, cache = {}, {}
    for mn in ('ParT_full', 'ParT_kin', 'ParT_untr', 'ParT_kin_perm'):
        if (OUTD / f'p33_pair_channels_{mn}.npz').exists(): res[mn] = model(mn, cache)
    fn = OUTD / 'p33b_input_geometry.json'; fn.write_text(json.dumps(res, indent=1)); log(f'wrote {fn}')
