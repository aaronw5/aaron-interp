"""X12 verify: fresh-sample rerun of the X12 translation and rank-window tests on every X12 model, plus missing controls.

    python -m interp.x12_verify            -> $AI_RESULTS/jedi/x12_verify_<model>.json per model and x12_verify.json (summary)

Sample: interp.jedi_core.sample(per_class=4000, seed=1) minus the jets also in the X12 sample (seed 0); N ~ 18,000.
Per model (official + the 8 X12 retrained models; imports Net, edits and statistics from interp.x12_retrained_jedi unchanged):
 (1) accuracy on the fresh sample.
 (2) translation, W -> Z on true W: unedited; T = +deta 0.06 rad; Tr = random direction 0.06 rad (rng 2201); M2-held = Tr then
     angle scaling restoring sum pT r^2 about the stored axis (as X12); competing-hypothesis controls:
       M1-held  = Tr then angle scaling restoring sum pT r (H_M1: the response follows the pT-weighted mean radius instead);
       shrink   = no translation, angles scaled by the M2-held factor s (what the scaling alone does to W -> Z);
     flows with jet-bootstrap 95 % CI (1,000), exact McNemar on paired Z calls vs T and vs unedited; invariant-mass change per edit.
 (3) rank-window swap (X12 step 4 code): (a) the model's own top mask edge from X12 (stored plan); (b) the official edge 13|14 vs
     11|12 with the live S units whose qs1 mask changes at 13|14 (if none: top-10 live S units by X12 cost); (c) the same edge with
     all live S units; (d) scan of adjacent-rank swaps k|k+1, k = 0..K (ranks held by >= 50 % of jets), m_k = mean over all live S
     units of |dS|/sd(S), on every 3rd jet; peak index = m_k / median_j m_j.
Verify 2 (--seed 2 -> x12_verify_s2_<model>.json, x12_verify_s2_summary.json): same steps on the seed-2 sample (minus seed-0
     jets) plus two calibration edits: dilM2 = angles scaled per jet by sT = sqrt(M2(T)/M2) (M0, M2 as under T; invariant mass
     x sT) and dilZ = angles x 91.19/80.38; ratio flow(T)/flow(dilM2) with paired jet bootstrap; scan on x[::4].
Combine: seed-level tests (networks as the unit): radial-hinge share and translation flow, HGQ vs float, exact permutation over
     network labels; l1 entry structure of nonzero units (how many units carry > 1 nonzero feature).
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import x12_retrained_jedi as X  # noqa: E402

OUT = X.OUT
MODELS = X.MODELS
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def fresh_sample(per=4000, seed=1):
    """seed-<seed> sample minus the X12 (seed 0) jets"""
    from interp.jedi_core import sample
    from jetdistill.config import RESULTS
    x, y, _ = sample(per_class=per, seed=seed)
    yy = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    def idx_of(p, s):
        r = np.random.default_rng(s)
        return np.sort(np.concatenate([r.choice(np.flatnonzero(yy == c), p, replace=False) for c in range(5)]))
    i1 = idx_of(per, seed); assert np.array_equal(yy[i1], y)
    i0 = idx_of(5000, 0); keep = ~np.isin(i1, i0)
    return np.ascontiguousarray(x[keep], np.float32), y[keep], dict(per_class=per, seed=seed, N_before=len(y), N=int(keep.sum()),
                                                                   overlap_removed=int((~keep).sum()))


def bci(v, B=1000, rng=np.random.default_rng(2202)):
    v = np.asarray(v, np.float64); n = len(v)
    m = np.array([v[rng.integers(0, n, n)].mean() for _ in range(B)]); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def M1(xx):
    xx = np.asarray(xx, np.float64); return (xx[..., 0] * np.hypot(xx[..., 1], xx[..., 2])).sum(1)


def translation(net, x, y):
    W = y == X.W_; c0 = net.run(x)['logits'].argmax(1); z0 = c0[W] == X.Z_
    th = np.random.default_rng(2201).uniform(0, 2 * np.pi, len(x))
    Tr = X.translate(x, 0.06, th); M = X.moments(x); Mq = X.moments(Tr)
    s2 = np.sqrt(M['M2'] / np.maximum(Mq['M2'], 1e-30)); s1 = M1(x) / np.maximum(M1(Tr), 1e-30)
    T = X.translate(x, 0.06, 0.0); sT = np.sqrt(X.moments(T)['M2'] / np.maximum(M['M2'], 1e-30))
    E = dict(T=T, Tr=Tr, M2held=X.scale_angles(Tr, s2), M1held=X.scale_angles(Tr, s1), shrink=X.scale_angles(x, s2))
    # verify-2 calibration edits (matched real change): dilM2 = angles about the stored axis scaled by sT per jet, so that M0 and
    # M2 equal those of T exactly while the invariant mass is multiplied by ~sT (H_M2: W->Z(dilM2) = W->Z(T); H_m (recentred /
    # invariant mass): W->Z(T) ~ unedited << W->Z(dilM2)); dilZ = angles scaled by m_Z / m_W = 91.19 / 80.38 (a real W -> Z mass shift)
    if getattr(net, '_dil', True):
        E['dilM2'] = X.scale_angles(x, sT); E['dilZ'] = X.scale_angles(x, np.full(len(x), 91.19 / 80.38))
    ok = M['m'] > 1; out = dict(N_W=int(W.sum()), W_to_Z_unedited=float(z0.mean()), edits={}, scale_M2held_mean=float(s2[W].mean()),
                                scale_M1held_mean=float(s1[W].mean()))
    Z = {}
    for k, xe in E.items():
        ze = net.run(xe)['logits'].argmax(1)[W] == X.Z_; Z[k] = ze; fl = ze.astype(float) - z0; Me = X.moments(xe)
        out['edits'][k] = dict(W_to_Z=float(ze.mean()), flow=float(fl.mean()), flow_ci=bci(fl),
                               dlogm_mean_W=float(np.log(Me['m'][ok & W] / M['m'][ok & W]).mean()),
                               dlogM2_mean_W=float(np.log(np.maximum(Me['M2'][W], 1e-30) / np.maximum(M['M2'][W], 1e-30)).mean()),
                               mcnemar_vs_unedited=X.mcnemar(ze, z0))
    for k in [k for k in E if k != 'T']:
        out['edits'][k]['mcnemar_vs_T'] = X.mcnemar(Z['T'], Z[k])
    if 'dilM2' in Z:   # share of the M2-matched dilation response that the translation reproduces, paired jet bootstrap over W
        fT = Z['T'].astype(float) - z0; fD = Z['dilM2'].astype(float) - z0; rr = np.random.default_rng(2203); n = len(fT); B = []
        for _ in range(1000):
            i = rr.integers(0, n, n); d = fD[i].mean(); B.append(fT[i].mean() / d if d > 0 else np.nan)
        out['ratio_flowT_over_flow_dilM2'] = [float(fT.mean() / fD.mean()) if fD.mean() > 0 else None,
                                              float(np.nanquantile(B, .025)), float(np.nanquantile(B, .975))]
        out['scale_T_mean_W'] = float(sT[W].mean())
    out['mcnemar_M1held_vs_M2held'] = X.mcnemar(Z['M1held'], Z['M2held'])
    e = out['edits']
    log('(2) W->Z unedited %.3f | ' % out['W_to_Z_unedited'] + ' '.join(f'{k} {v["W_to_Z"]:.3f} (dlogm {v["dlogm_mean_W"]:+.3f})' for k, v in e.items()))
    return out


def windows(net, x, d0):
    """d0: the X12 per-model json (stored plan, live S units, S costs)"""
    M1m, _ = X.slot_mask(net, 'qs1'); liveS = d0['live']['S']; cost = np.asarray(d0['live']['S_cost'])
    res = dict(own=None, official_edge=None, official_edge_all=None, scan=None)
    plan0 = d0['windows']['plan'][:1]
    res['own'] = X.window_test(net, x, [dict(edge=p['edge'], inside=p['inside'], units=p['units']) for p in plan0], 'own edge')
    u13 = [c for c in liveS if M1m is not None and M1m[13, c] != M1m[14, c]]
    top = sorted(liveS, key=lambda c: -cost[c])[:10]
    units = u13 if u13 else top
    res['official_edge_units'] = dict(units=[int(c) for c in units], rule='qs1 mask edge at 13|14' if u13 else 'top-10 live S by X12 cost')
    res['official_edge'] = X.window_test(net, x, [dict(edge=13, inside=11, units=units), dict(edge=13, inside=5, units=units)], '13|14')
    res['official_edge_all'] = X.window_test(net, x, [dict(edge=13, inside=11, units=liveS)], '13|14 all live S', full=False)
    # scan
    xs = x[::4]; R0 = net.run(xs, keep=('S',))['S']; sd = R0.std(0); u = [c for c in liveS if sd[c] > 0]
    nreal = (xs[..., 0] > 0).sum(1); K = [k for k in range(63) if (nreal > k + 1).mean() >= 0.5]
    prof = {}
    for k in K:
        J, xe = X.swap(xs, k); S = net.run(xe, keep=('S',))['S']
        prof[k] = float((np.abs(S[:, u] - R0[J][:, u]) / sd[u]).mean())
    v = np.array(list(prof.values())); med = float(np.median(v)); kmax = max(prof, key=prof.get)
    mask_edges = sorted({k for k in K for c in liveS if M1m is not None and M1m[k, c] != M1m[k + 1, c]})
    res['scan'] = dict(N=len(xs), units=u, profile=prof, median=med, peak_k=int(kmax), peak_index=float(prof[kmax] / med) if med > 0 else None,
                       index_13=float(prof.get(13, np.nan) / med) if med > 0 else None, live_mask_edges=mask_edges)
    log(f'(3d) scan peak k={kmax} index {res["scan"]["peak_index"]:.2f}; index at 13: {res["scan"]["index_13"]:.2f}; mask edges {mask_edges}')
    return res


def tag(a): return '' if a.seed == 1 else f's{a.seed}_'


def run_all(a):
    x, y, info = fresh_sample(seed=a.seed); log('sample', info)
    for m in (a.models.split(',') if a.models else MODELS):
        p0 = OUT / f'x12_{m}_final.json'
        if not p0.exists(): log('missing', p0); continue
        d0 = json.loads(p0.read_text())
        mp, _, _ = X.model_path(m, 'final')
        net = X.Net('official' if m == 'official' else 'x12', mp)
        c0 = net.run(x)['logits'].argmax(1)
        R = dict(model=m, sample=info, acc=float((c0 == y).mean()), acc_ci=X.wilson(int((c0 == y).sum()), len(y)))
        log(m, 'acc', R['acc'])
        R['translation'] = translation(net, x, y)
        R['windows'] = windows(net, x, d0)
        R['peak_rss_mib'] = rss()
        (OUT / f'x12_verify_{tag(a)}{m}.json').write_text(json.dumps(X.clean(R), indent=1)); log('wrote', m, 'rss', R['peak_rss_mib'])
        del net; gc.collect()
        import jax, keras; jax.clear_caches(); keras.backend.clear_session()


def perm_p(a, b):
    """exact one-sided permutation p over network labels: mean(a) - mean(b) >= observed"""
    from itertools import combinations
    v = np.array(list(a) + list(b), float); n = len(a); obs = np.mean(a) - np.mean(b); c = t = 0
    for S in combinations(range(len(v)), n):
        m = np.zeros(len(v), bool); m[list(S)] = True; t += 1; c += (v[m].mean() - v[~m].mean()) >= obs - 1e-12
    return c / t, t


def combine(a):
    from scipy import stats
    D = {m: json.loads((OUT / f'x12_verify_{tag(a)}{m}.json').read_text()) for m in MODELS if (OUT / f'x12_verify_{tag(a)}{m}.json').exists()}
    O = {m: json.loads((OUT / f'x12_{m}_final.json').read_text()) for m in MODELS}
    S = dict(models=list(D), rows={})
    for m, d in D.items():
        t = d['translation']['edits']; w = d['windows']
        S['rows'][m] = dict(acc=d['acc'], WZ0=d['translation']['W_to_Z_unedited'], **{f'WZ_{k}': v['W_to_Z'] for k, v in t.items()},
                            flow={k: [v['flow']] + v['flow_ci'] for k, v in t.items()}, dlogm={k: v['dlogm_mean_W'] for k, v in t.items()},
                            mcn_T_vs_M2held=t['M2held']['mcnemar_vs_T'], mcn_T_vs_M1held=t['M1held']['mcnemar_vs_T'],
                            mcn_M1_vs_M2=d['translation']['mcnemar_M1held_vs_M2held'], mcn_M2held_vs_unedited=t['M2held']['mcnemar_vs_unedited'],
                            own_edge={k: w['own'][0].get(k) for k in ('edge', 'inside', 'N', 'ratio', 'ratio_ci', 'signflip_p_one_sided')},
                            own_edge_x12=O[m]['windows']['trained'][0]['ratio'], own_null_p90=O[m]['windows']['plan'][0].get('null_p90'),
                            edge13={f'vs{r["inside"]}': {k: r.get(k) for k in ('N', 'ratio', 'ratio_ci', 'signflip_p_one_sided', 'units_used')} for r in w['official_edge']},
                            edge13_units_rule=w['official_edge_units']['rule'], edge13_all_live=w['official_edge_all'][0]['ratio'],
                            scan_peak=[w['scan']['peak_k'], w['scan']['peak_index']], scan_index13=w['scan']['index_13'],
                            ratio_T_over_dilM2=d['translation'].get('ratio_flowT_over_flow_dilM2'),
                            mcn_T_vs_dilM2=t.get('dilM2', {}).get('mcnemar_vs_T'))
    # Holm over the new window tests at 13|14 (vs 11|12) and the mass tests, retrained models
    keys, ps = [], []
    for m, r in S['rows'].items():
        if m == 'official': continue
        keys.append(f'{m}:edge13'); ps.append(r['edge13']['vs11']['signflip_p_one_sided'])
        keys.append(f'{m}:T_vs_M2held'); ps.append(r['mcn_T_vs_M2held']['p'])
    S['holm'] = {k: dict(p=p, p_holm=h) for k, p, h in zip(keys, ps, X.holm(ps))}
    # seed-level (network = unit) tests
    usable_h = ['hgq-r150_s0', 'hgq-r150_s2', 'hgq_s0', 'hgq_s1', 'hgq_s2']; prim = ['hgq-r150_s0', 'hgq-r150_s2']; flo = ['float_s0', 'float_s1']
    def rh(m):
        us = [u for u in O[m]['naming']['units'].values() if u.get('slot') is not None and not u.get('constant')]
        return sum(bool(u['radial'] and u['hinge']) for u in us) / len(us), len(us)
    def rh_top(m, n):
        us = [u for u in O[m]['naming']['units'].values() if u.get('slot') is not None and not u.get('constant')][:n]
        return sum(bool(u['radial'] and u['hinge']) for u in us) / len(us)
    S['radial_hinge_per_network'] = {m: rh(m) for m in MODELS}
    S['radial_hinge_float_top7'] = {m: rh_top(m, 7) for m in flo}
    S['radial_hinge_pooled_fisher_primary_excl_s1'] = float(stats.fisher_exact([[sum(round(rh(m)[0] * rh(m)[1]) for m in prim), sum(rh(m)[1] for m in prim) - sum(round(rh(m)[0] * rh(m)[1]) for m in prim)],
                                                                              [sum(round(rh(m)[0] * rh(m)[1]) for m in flo), sum(rh(m)[1] for m in flo) - sum(round(rh(m)[0] * rh(m)[1]) for m in flo)]])[1])
    S['seed_level'] = dict(
        radial_hinge_primary_vs_float=perm_p([rh(m)[0] for m in prim], [rh(m)[0] for m in flo]),
        radial_hinge_allhgq_vs_float=perm_p([rh(m)[0] for m in usable_h], [rh(m)[0] for m in flo]),
        dilZ_allhgq_vs_float=perm_p([S['rows'][m]['flow']['dilZ'][0] for m in usable_h if m in S['rows'] and 'dilZ' in S['rows'][m]['flow']],
                                    [S['rows'][m]['flow']['dilZ'][0] for m in flo if m in S['rows'] and 'dilZ' in S['rows'][m]['flow']]),
        ratioT_dil_allhgq_vs_float=perm_p([S['rows'][m]['ratio_T_over_dilM2'][0] for m in usable_h if m in S['rows'] and S['rows'][m]['ratio_T_over_dilM2']],
                                          [S['rows'][m]['ratio_T_over_dilM2'][0] for m in flo if m in S['rows'] and S['rows'][m]['ratio_T_over_dilM2']]),
        flowT_allhgq_vs_float=perm_p([S['rows'][m]['flow']['T'][0] for m in usable_h if m in S['rows']], [S['rows'][m]['flow']['T'][0] for m in flo if m in S['rows']]),
        note='exact one-sided permutation over network labels; minimum attainable p = 1/C(n, k)')
    # l1 structure: units with > 1 nonzero feature (|alpha| sd share)
    S['l1_multi_entry_units'] = {}
    for m in MODELS:
        if 'alpha_all' not in O[m]['purity']: continue
        al = np.asarray(O[m]['purity']['alpha_all']); nz = (al != 0); u = np.flatnonzero(nz.any(0))
        S['l1_multi_entry_units'][m] = dict(nonzero_units=int(len(u)), units_gt1_feature=int((nz[:, u].sum(0) > 1).sum()),
                                            units_pT_and_angle=int((nz[0, u] & (nz[1, u] | nz[2, u])).sum()))
    for k, v in S.items():
        if k != 'rows': log(k, json.dumps(X.clean(v)))
    for m, r in S['rows'].items(): log(m, json.dumps(X.clean(r)))
    S['seed'] = a.seed; fo = OUT / f'x12_verify_{tag(a)}summary.json'
    fo.write_text(json.dumps(X.clean(S), indent=1)); log('wrote', fo)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--combine', action='store_true'); ap.add_argument('--models', default='')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    combine(a) if a.combine else run_all(a)
