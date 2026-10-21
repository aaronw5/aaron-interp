"""X15 verify: re-run of X15 step 1-3 on a fresh sample, plus checks of smooth (non-edge) rank structure in the Linformer E.

Part 0 (re-run): interp.x15_linformer_ranks.small_mode with sample seed 2602 instead of 1501 (same per_class 1000, N = 5,000,
  same bootstrap weights seed, same derangement seeds).  Output $AI_RESULTS/jedi/x15_verify_small.json.
Part A (weights, slot-invariant smooth statistics; the X15 edge score only sees jumps between adjacent ranks):
  row norm r_b(n) = ||E_b[n]||_2 of rank n in block b; low-rank emphasis LRE_b = mean r_b(0..4) / median r_b(5..39);
  same for each trained net's initial E and for the learned change dE = E - E_init.  Null: 4,000 random E ~ N(0, 1/64) (the PyTorch
  init of E, so the H0 "E has no rank structure beyond its random init") and the 5 untrained Linformers.
  Cross-seed agreement (Spearman over ranks 0..39) of r_b, of a smoothed edge score (w = 4: ||mean E[i+1..i+4] - mean E[i-3..i]||),
  and of the rank Gram matrix G = E E^T (ranks 0..39, off-diagonal entries; invariant to permuting the 8 slots), for E and dE,
  against the same statistics for 2,000 pairs of independent random E.
  Max edge score over C (the X15 statistic) for 4,000 random E pairs (2 blocks): q95, q99 (the X15 'untrained max' used 5 nets).
Part B (causal: is the low-rank emphasis used?): for each trained Linformer and block b, rows 0..4 of E_b are rescaled to the median
  row norm of ranks 5..39 (direction kept).  Metric: accuracy change, mean |delta logit| (5-class mean), % calls changed, McNemar.
  Controls (20 draws each): (i) the same multiplicative factors applied to 5 random rows from ranks 5..P (P = last rank occupied in
  >= 90 % of jets); (ii) random-direction perturbations of the same per-row norm as the real edit, added to 5 random rows of that pool.
  p = (1 + #controls >= real) / 21 on |delta logit|; 95 % jet-bootstrap CI (1,000) on the real effects.
Part C (causal, cross-seed): Spearman between two networks' 62-boundary swap-response profiles (from part 0), raw and partial given a
  quadratic in rank and the Deep Sets D-s0 swap response (which carries the data-only part: pT gap and angular distance of the pair).
Run: AI_PY=conda ./job.sh x15-verify 'python -m interp.x15_verify'  -> $AI_RESULTS/jedi/x15_verify.json
"""
import itertools, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import interp.x15_linformer_ranks as X

VSEED = 2602; K = 5; NCTRL = 20; NSIM = 4000; NPAIR = 2000; RMAX = 40
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def spear(a, b):
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def lre(E):
    r = np.linalg.norm(E, axis=1); return float(r[:K].mean() / np.median(r[K:RMAX]))


def smooth_edge(E, w=4):
    return np.array([np.linalg.norm(E[i + 1:i + 1 + w].mean(0) - E[max(0, i - w + 1):i + 1].mean(0)) for i in range(RMAX - 1)])


def gram_off(E):
    G = E[:RMAX] @ E[:RMAX].T; iu = np.triu_indices(RMAX, 1); return G[iu]


def stats_pair(Ea, Eb):
    """Ea, Eb: lists of 2 blocks -> cross-net agreement statistics averaged over the 2 blocks"""
    rn = np.mean([spear(np.linalg.norm(a, axis=1)[:RMAX], np.linalg.norm(b, axis=1)[:RMAX]) for a, b in zip(Ea, Eb)])
    rn5 = np.mean([spear(np.linalg.norm(a, axis=1)[K:RMAX], np.linalg.norm(b, axis=1)[K:RMAX]) for a, b in zip(Ea, Eb)])
    se = np.mean([spear(smooth_edge(a), smooth_edge(b)) for a, b in zip(Ea, Eb)])
    gr = np.mean([float(np.corrcoef(gram_off(a), gram_off(b))[0, 1]) for a, b in zip(Ea, Eb)])
    return dict(rownorm_rho=float(rn), rownorm_rho_ranks5plus=float(rn5), smooth_edge_rho=float(se), gram_r=float(gr))


def build(M, run, seed, untrained=False):
    import torch
    d = X.SMALL / run / f'seed{0 if untrained else seed}'; arch = json.loads((d / 'arch.json').read_text())
    if untrained: torch.manual_seed(5000 + seed)
    net = M.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
    if not untrained: net.load_state_dict(torch.load(d / 'best.pt', map_location='cpu'))
    net.eval(); return dict(net=net, arch=arch, axis=arch['inputs'].get('axis', 'stored'), dir=d)


def fwd(N, xr):
    import torch
    a = N['arch']; xi = X.recentre_np(xr) if N['axis'] == 'hardest' else xr
    sh = np.asarray(a['inputs']['shift'], np.float32); sc = np.asarray(a['inputs']['scale'], np.float32)
    m = xi[..., 0] > 0; xn = (((xi - sh) / sc) * m[..., None]).astype(np.float32); out = []
    with torch.no_grad():
        for i in range(0, len(xn), 1000):
            out.append(N['net'](torch.from_numpy(xn[i:i + 1000]), torch.from_numpy(m[i:i + 1000])).double().numpy())
    return np.concatenate(out)


def main():
    import importlib.util, torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    final = X.OUT; final.mkdir(parents=True, exist_ok=True); res = dict(seed=VSEED)
    # ------------------------------------------------ part 0: re-run on a fresh sample
    tmp = Path(tempfile.mkdtemp()); X.SEED = VSEED; X.OUT = tmp
    X.small_mode()
    S = json.loads((tmp / 'x15_small.json').read_text()); (final / 'x15_verify_small.json').write_text(json.dumps(S)); X.OUT = final
    log('part 0 written', final / 'x15_verify_small.json')
    C = S['candidates']; nets = S['nets']
    # ------------------------------------------------ part A: weights
    spec = importlib.util.spec_from_file_location('small_models', X.ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    names = {'L-st-s0': ('linf-2b-d32-k8-n64', 0), 'L-st-s1': ('linf-2b-d32-k8-n64', 1),
             'L-ax-s0': ('linf-2b-d32-k8-n64-axhard', 0), 'L-ax-s1': ('linf-2b-d32-k8-n64-axhard', 1)}
    E = {nm: [np.array(e) for e in nets[nm]['E']] for nm in list(names) + [f'U{k}' for k in range(5)]}
    dE = {nm: [np.array(e) for e in nets[nm]['init']['dE']] for nm in names}
    Ei = {nm: [a - b for a, b in zip(E[nm], dE[nm])] for nm in names}
    rng = np.random.default_rng(15015)
    def rnd(): return [rng.standard_normal((64, 8)) / 8.0 for _ in range(2)]
    sims = [rnd() for _ in range(NSIM)]
    lre_null = np.array([lre(s[b]) for s in sims for b in range(2)])
    maxedge_null = np.array([max(X.edge_profile(s, C)[0]) for s in sims])
    A = dict(lre={}, lre_null=dict(q50=float(np.median(lre_null)), q95=float(np.quantile(lre_null, 0.95)), q99=float(np.quantile(lre_null, 0.99)),
                                   max=float(lre_null.max()), n=len(lre_null)),
             maxedge_null=dict(q95=float(np.quantile(maxedge_null, 0.95)), q99=float(np.quantile(maxedge_null, 0.99)), n=NSIM))
    for nm in E:
        A['lre'][nm] = dict(E=[lre(e) for e in E[nm]], rownorm_0_6=[np.linalg.norm(e, axis=1)[:7].round(3).tolist() for e in E[nm]])
        if nm in names:
            A['lre'][nm].update(E_init=[lre(e) for e in Ei[nm]], dE=[lre(e) for e in dE[nm]],
                                p_vs_null=[float((1 + np.sum(lre_null >= lre(e))) / (1 + len(lre_null))) for e in E[nm]])
        A['lre'][nm]['max_edge'] = float(max(nets[nm]['edge_score'])); A['lre'][nm]['edge_1_2_3'] = [nets[nm]['edge_score'][C.index(i)] for i in (1, 2, 3)]
        log(f'A {nm}: LRE E {np.round(A["lre"][nm]["E"], 2).tolist()} init {np.round(A["lre"][nm].get("E_init", []), 2).tolist()} '
            f'dE {np.round(A["lre"][nm].get("dE", []), 2).tolist()}; max edge {A["lre"][nm]["max_edge"]:.2f}')
    log('A LRE null', A['lre_null'], 'max-edge null', A['maxedge_null'])
    pnull = [stats_pair(sims[2 * k], sims[2 * k + 1]) for k in range(NPAIR)]
    A['pair_null'] = {k: dict(q95=float(np.quantile([p[k] for p in pnull], 0.95)), q99=float(np.quantile([p[k] for p in pnull], 0.99)),
                              mean=float(np.mean([p[k] for p in pnull]))) for k in pnull[0]}
    pairs = [('L-st-s0', 'L-st-s1'), ('L-ax-s0', 'L-ax-s1'), ('L-st-s0', 'L-ax-s1'), ('L-st-s1', 'L-ax-s0'), ('L-st-s0', 'L-ax-s0'),
             ('L-st-s1', 'L-ax-s1')]
    A['pairs'] = {}
    for a, b in pairs:
        st = dict(E=stats_pair(E[a], E[b]), dE=stats_pair(dE[a], dE[b]))
        for kind in ('E', 'dE'):
            st[kind]['p_vs_null'] = {k: float((1 + sum(p[k] >= st[kind][k] for p in pnull)) / (1 + NPAIR)) for k in pnull[0]}
        A['pairs'][f'{a}|{b}'] = st; log(f'A pair {a}|{b}: {json.dumps(st)}')
    A['untrained_pairs'] = {f'U{a}|U{b}': stats_pair(E[f'U{a}'], E[f'U{b}']) for a, b in itertools.combinations(range(5), 2)}
    log('A pair null', json.dumps(A['pair_null'])); res['A'] = A
    # ------------------------------------------------ part B: causal flatten of the low-rank emphasis
    x, y, _ = X.load(); J = len(y); nr = X.nreal(x); occ = np.array([(nr > n).mean() for n in range(64)])
    P = max(n for n in range(64) if occ[n] >= 0.90); pool = list(range(K, P + 1)); W = X.boot_w(J)
    res['B'] = dict(occupancy=occ.round(4).tolist(), pool=[K, P], nets={})
    log(f'B: pool ranks {K}..{P} (occupancy >= 0.90)')
    crng = np.random.default_rng(15016)
    for nm, (run, sd) in names.items():
        N = build(M, run, sd); L0 = fwd(N, x); c0 = L0.argmax(1); ok0 = c0 == y; out = dict(acc0=float(ok0.mean()), blocks=[])
        for b in range(2):
            Eb = N['net'].blocks[b].attn.E; E0 = Eb.detach().clone(); En = E0.numpy().astype(np.float64)
            rn = np.linalg.norm(En, axis=1); target = np.median(rn[K:RMAX]); fac = target / rn[:K]
            def run_edit(Enew):
                with torch.no_grad(): Eb.copy_(torch.from_numpy(Enew.astype(np.float32)))
                L1 = fwd(N, x)
                with torch.no_grad(): Eb.copy_(E0)
                return L1
            Er = En.copy(); Er[:K] *= fac[:, None]; L1 = run_edit(Er)
            d = np.abs(L1 - L0).mean(1); c1 = L1.argmax(1); ok1 = c1 == y; ch = c1 != c0
            dacc_b = (W @ (ok1.astype(float) - ok0)) / J; d_b = (W @ d) / J
            real = dict(rows=list(range(K)), factors=fac.round(3).tolist(), dlogit=float(d.mean()), dlogit_ci=X.ci(d_b), calls=float(ch.mean()),
                        acc=float(ok1.mean()), dacc=float(ok1.mean() - ok0.mean()), dacc_ci=X.ci(dacc_b), mcnemar_correct=X.mcnemar(ok0, ok1))
            ctl_mult, ctl_dir = [], []
            for _ in range(NCTRL):
                rows = crng.choice(pool, K, replace=False); Ec = En.copy(); Ec[rows] *= fac[:, None]; Lc = fwd_c = run_edit(Ec)
                ctl_mult.append(dict(rows=rows.tolist(), dlogit=float(np.abs(Lc - L0).mean(1).mean()), calls=float((Lc.argmax(1) != c0).mean()),
                                     dacc=float((Lc.argmax(1) == y).mean() - ok0.mean())))
                rows = crng.choice(pool, K, replace=False); Ec = En.copy(); dn = np.linalg.norm(Er[:K] - En[:K], axis=1)
                v = crng.standard_normal((K, En.shape[1])); v *= (dn / np.linalg.norm(v, axis=1))[:, None]; Ec[rows] += v; Lc = run_edit(Ec)
                ctl_dir.append(dict(rows=rows.tolist(), dlogit=float(np.abs(Lc - L0).mean(1).mean()), calls=float((Lc.argmax(1) != c0).mean()),
                                    dacc=float((Lc.argmax(1) == y).mean() - ok0.mean())))
            blk = dict(block=b, real=real)
            for nmc, ctl in (('ctrl_same_factors', ctl_mult), ('ctrl_same_norm_random_dir', ctl_dir)):
                dl = np.array([q['dlogit'] for q in ctl]); da = np.array([q['dacc'] for q in ctl])
                blk[nmc] = dict(dlogit_mean=float(dl.mean()), dlogit_max=float(dl.max()), dacc_mean=float(da.mean()), dacc_min=float(da.min()),
                                calls_mean=float(np.mean([q['calls'] for q in ctl])), ratio_real_over_ctrl=float(real['dlogit'] / max(dl.mean(), 1e-12)),
                                p_dlogit=float((1 + np.sum(dl >= real['dlogit'])) / (1 + len(dl))),
                                p_dacc=float((1 + np.sum(da <= real['dacc'])) / (1 + len(da))), draws=ctl)
            out['blocks'].append(blk)
            log(f'B {nm} block {b}: rows 0-4 x {np.round(fac, 2).tolist()}: |dl| {real["dlogit"]:.4f} {np.round(real["dlogit_ci"], 4).tolist()} '
                f'calls {100 * real["calls"]:.2f} % dacc {100 * real["dacc"]:+.2f} {np.round(100 * np.array(real["dacc_ci"]), 2).tolist()} McN p {real["mcnemar_correct"]["p"]:.3g} | '
                f'ctrl mult |dl| {blk["ctrl_same_factors"]["dlogit_mean"]:.4f} (max {blk["ctrl_same_factors"]["dlogit_max"]:.4f}) dacc {100 * blk["ctrl_same_factors"]["dacc_mean"]:+.2f} '
                f'p {blk["ctrl_same_factors"]["p_dlogit"]:.3f} | ctrl dir |dl| {blk["ctrl_same_norm_random_dir"]["dlogit_mean"]:.4f} (max {blk["ctrl_same_norm_random_dir"]["dlogit_max"]:.4f}) '
                f'dacc {100 * blk["ctrl_same_norm_random_dir"]["dacc_mean"]:+.2f} p {blk["ctrl_same_norm_random_dir"]["p_dlogit"]:.3f}')
        res['B']['nets'][nm] = out
    # ------------------------------------------------ part C: cross-seed agreement of swap-response profiles
    from scipy.stats import rankdata
    r = np.asarray(C, float); ds = np.log(np.asarray(nets['D-s0']['scan_dlogit']))
    def resid(v, extra=True):
        cols = [np.ones_like(r), r, r ** 2] + ([rankdata(ds)] if extra else []); Xm = np.stack(cols, 1); yy = rankdata(v)
        return yy - Xm @ np.linalg.lstsq(Xm, yy, rcond=None)[0]
    prng = np.random.default_rng(15017); Cres = {}
    for a, b in pairs + [('D-s0', 'D-s1')] + [(f'U{i}', f'U{j}') for i, j in itertools.combinations(range(5), 2)]:
        va, vb = nets[a]['scan_dlogit'], nets[b]['scan_dlogit']; raw = spear(va, vb)
        ea = a.startswith('D'); ra, rb_ = resid(va, not ea), resid(vb, not ea); pr = float(np.corrcoef(ra, rb_)[0, 1])
        null = np.array([np.corrcoef(ra, prng.permutation(rb_))[0, 1] for _ in range(2000)])
        Cres[f'{a}|{b}'] = dict(raw=raw, partial=pr, p_partial=float(np.mean(np.abs(null) >= abs(pr))), partial_on='rank quad' + ('' if ea else ' + D-s0 swap response'))
        log(f'C {a}|{b}: swap-profile Spearman raw {raw:.3f}, partial {pr:.3f} (p {Cres[f"{a}|{b}"]["p_partial"]:.4f})')
    res['C'] = Cres
    # ------------------------------------------------ headline re-run numbers
    head = {}
    for nm, R in nets.items():
        t = R['step2']; h = dict(acc=R['acc'])
        for k in ('own', 'at_ref'):
            if k in t: h[k] = dict(edges=t[k]['edges'], flats=t[k]['flats'], ratio=t[k]['dlogit']['ratio'], ci=t[k]['dlogit']['ratio_ci'],
                                   p_perm=t[k]['dlogit']['p_perm'], calls_edge=t[k]['calls']['edge_mean'], calls_flat=t[k]['calls']['flat_mean'])
        for k in ('effect_vs_score_own', 'effect_vs_score_ref'):
            if k in t: h[k] = t[k]
        h['step3'] = [(q['edge'], round(q['ratio'], 3)) for q in R['step3']['rows']]
        head[nm] = h
    res['rerun'] = dict(untrained_ratio=S['untrained_ratio'], nets=head, recurrence={k: dict(exact=v['exact'], p_exact=v['p_exact'], rho=v['profile']['spearman'])
                                                                                       for k, v in S['recurrence'].items()})
    fo = final / 'x15_verify.json'; fo.write_text(json.dumps(X.clean(res))); log('wrote', fo)


if __name__ == '__main__':
    main()
