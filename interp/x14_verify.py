"""X14 verification (fresh jets): recompute the axis-tied share R of the X14 grid runs and schedule-length pairs on a new sample,
and add two controls.

Sample: interp.jedi_core.sample(per_class=1000, seed=2029) (X13 / X14 used seed 1313); fit / test halves by the same rule
(np.random.default_rng(seed).permutation). Same R as X14 (interp.x13_axis_size edits; ridge read-out of log m on the last hidden layer,
refitted per checkpoint; R = mean_test d(T, 8-azimuth mean) / mean_test d(dil); null: rotation about the stored axis).
Runs: tf-2b-d32 and linf-2b-d32-k8 grid (<model>-n64-x14d<d>/seed<i>) at epochs 3, 10, 30; schedule pairs tf-2b-d32-n64 s0 /
-e100 s0 / -p30 s1 / -e100 s1 at epoch 30 and last.
Controls added:
  R_logit  read-out-free version: d(logit_Z - logit_W) under T / under dil (network's own output, all test jets).
  R_shuf   ridge alpha chosen on a random quarter of the fit half (X13 Ridge picks alpha on the positional last quarter; the
           sampled rows are in file order) -> checks that R does not depend on that choice.
  interaction null: probability, under full permutation of the 9 cells, of a residual (interaction) share >= the observed one.
    python -m interp.x14_verify          -> $AI_RESULTS/jedi/x14_verify.json
"""
import json, os, sys, time, csv, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import interp.x13_axis_size as X

SEED = 2029; X.SEED = SEED   # fresh sample (load_sample reads the module globals at call time)
OUT = X.OUT; SMALL = X.SMALL; B = 1000; W_, Z_ = 2, 3; T0 = time.time()
ED = [f'T{k}' for k in range(8)] + ['dil', 'rot']
GRID = {m: {(i, d): (f'{m}-n64-x14d{d}', f'seed{i}') for i in range(3) for d in range(3)} for m in ('tf-2b-d32', 'linf-2b-d32-k8')}
PAIRS = [('tf-2b-d32-n64', 'seed0'), ('tf-2b-d32-n64-e100', 'seed0'), ('tf-2b-d32-n64-p30', 'seed1'), ('tf-2b-d32-n64-e100', 'seed1')]


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


def bp(v):
    v = np.asarray(v, float); return float(min(1.0, 2 * (1 + min((v <= 0).sum(), (v >= 0).sum())) / (len(v) + 1)))


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0])


def shares(Mx):
    g = Mx.mean(); sst = ((Mx - g) ** 2).sum(); si = 3 * ((Mx.mean(1) - g) ** 2).sum(); sd = 3 * ((Mx.mean(0) - g) ** 2).sum()
    return np.array([si / sst, sd / sst, (sst - si - sd) / sst])


def main():
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    x128, y, fit, te = X.load_sample()
    log(f'sample(per_class={X.PER}, seed={X.SEED}): N {len(y)}, fit {len(fit)}, test {len(te)}')
    # class composition of the positional last quarter of the fit half (where X13's Ridge picks alpha)
    q = fit[len(fit) * 3 // 4:]; comp = np.bincount(y[q], minlength=5).tolist(); log('classes in the alpha-selection quarter:', comp)
    rs = np.random.default_rng(SEED + 7); fit_sh = rs.permutation(fit)
    E = X.make_edits(x128[:, :64]); E = {k: E[k] for k in ['0'] + ED}
    logm = np.log(np.maximum(X.moments(x128[:, :64])['m'], 1e-3))
    rng = np.random.default_rng(141430); I = rng.integers(0, len(te), (B, len(te)))
    CNT = np.stack([np.bincount(i, minlength=len(te)) for i in I]).astype(np.float64) / len(te)
    todo = []
    for m, G in GRID.items():
        for (i, d), (run, sd) in G.items(): todo.append((run, sd, [3, 10, 30]))
    for run, sd in PAIRS:
        a = json.loads((SMALL / run / sd / 'arch.json').read_text()); last = a['best']['epochs_run']
        todo.append((run, sd, sorted({30, last})))
    res = {}
    for run, sd, eps in todo:
        key = f'{run}__{sd}'
        if key in res: continue
        d = SMALL / run / sd; arch = json.loads((d / 'arch.json').read_text())
        net = M.build_model(arch['spec'], 64, cls_token=False, final_ln=arch['final_ln']); f = X.torch_runner(net, arch)
        L = {int(r['epoch']): r for r in csv.DictReader(open(d / 'log.csv'))}
        rr = dict(epochs=[], R=[], R_ci=[], R_shuf=[], R_logit=[], R_logit_ci=[], rot_over_dil=[], T_minus_rot_p=[], acc=[],
                  val_acc=[], val_loss=[], Rb=[], RLb=[])
        for e in eps:
            net.load_state_dict(torch.load(d / 'ckpt' / f'epoch_{e}.pt', map_location='cpu')); net.eval()
            O = {k: f(E[k]) for k in ['0'] + ED}
            rd = X.Ridge(O['0']['last'][fit], logm[fit]); rd2 = X.Ridge(O['0']['last'][fit_sh], logm[fit_sh])
            dd = {k: rd(O[k]['last'][te]) - rd(O['0']['last'][te]) for k in ED}
            d2 = {k: rd2(O[k]['last'][te]) - rd2(O['0']['last'][te]) for k in ED}
            zl = lambda o: o['logits'][te, Z_] - o['logits'][te, W_]
            dl = {k: zl(O[k]) - zl(O['0']) for k in ED}
            Tm = np.mean([dd[f'T{k}'] for k in range(8)], 0); Tm2 = np.mean([d2[f'T{k}'] for k in range(8)], 0)
            Tl = np.mean([dl[f'T{k}'] for k in range(8)], 0)
            bT, bD, bR = CNT @ Tm, CNT @ dd['dil'], CNT @ dd['rot']; bTl, bDl = CNT @ Tl, CNT @ dl['dil']
            rr['epochs'].append(e); rr['R'].append(float(Tm.mean() / dd['dil'].mean())); rr['R_ci'].append(ci(bT / bD))
            rr['R_shuf'].append(float(Tm2.mean() / d2['dil'].mean()))
            rr['R_logit'].append(float(Tl.mean() / dl['dil'].mean())); rr['R_logit_ci'].append(ci(bTl / bDl))
            rr['rot_over_dil'].append(float(dd['rot'].mean() / dd['dil'].mean())); rr['T_minus_rot_p'].append(bp(bT - bR))
            rr['acc'].append(float((O['0']['logits'].argmax(1) == y).mean()))
            rr['val_acc'].append(float(L[e]['val_acc'])); rr['val_loss'].append(float(L[e]['val_loss']))
            rr['Rb'].append(bT / bD); rr['RLb'].append(bTl / bDl)
            log(f'{key} ep {e}: R {rr["R"][-1]:+.3f} {rr["R_ci"][-1]} R_shuf {rr["R_shuf"][-1]:+.3f} R_logit {rr["R_logit"][-1]:+.3f} '
                f'rot/dil {rr["rot_over_dil"][-1]:+.3f} acc {rr["acc"][-1]:.4f}')
        res[key] = rr
    # ---------------------------------------------------------------- comparison with X14 (seed 1313)
    S14 = json.loads((OUT / 'x14_summary.json').read_text())['runs']
    out = dict(sample=dict(per_class=X.PER, seed=SEED, N=len(y), N_test=len(te), alpha_quarter_classes=comp), B=B, runs={}, grid={}, pairs={})
    cmp = []
    for k, rr in res.items():
        o = S14.get(k); r14 = {e: o['R'][o['epochs'].index(e)] for e in rr['epochs'] if o and e in o['epochs']}
        out['runs'][k] = {kk: v for kk, v in rr.items() if kk not in ('Rb', 'RLb')}; out['runs'][k]['R_x14'] = [r14.get(e) for e in rr['epochs']]
        cmp += [(r14[e], rr['R'][j]) for j, e in enumerate(rr['epochs']) if e in r14]
    cmp = np.array(cmp); out['replication'] = dict(n=len(cmp), max_abs_dR=float(np.abs(cmp[:, 0] - cmp[:, 1]).max()),
                                                    median_abs_dR=float(np.median(np.abs(cmp[:, 0] - cmp[:, 1]))), spearman=spearman(cmp[:, 0], cmp[:, 1]),
                                                    max_abs_R_minus_Rshuf=float(max(abs(a - b) for rr in res.values() for a, b in zip(rr['R'], rr['R_shuf']))))
    log('replication vs X14:', out['replication'])
    r3 = np.random.default_rng(141450)
    for m, G in GRID.items():
        ix = lambda i, d: res[f'{G[(i, d)][0]}__{G[(i, d)][1]}']
        g = {}
        for lab, fld, bfld in (('R', 'R', 'Rb'), ('R_logit', 'R_logit', 'RLb')):
            mat = np.array([[ix(i, d)[fld][-1] for d in range(3)] for i in range(3)])
            matb = np.stack([np.stack([ix(i, d)[bfld][-1] for d in range(3)], 1) for i in range(3)], 1)
            sh = shares(mat); bsh = np.array([shares(matb[b]) for b in range(B)])
            pf = np.array([shares(r3.permutation(mat.ravel()).reshape(3, 3)) for _ in range(10000)])
            pi_ = np.array([shares(np.stack([r3.permutation(mat[:, d]) for d in range(3)], 1))[0] for _ in range(10000)])
            pd_ = np.array([shares(np.stack([r3.permutation(mat[i]) for i in range(3)], 0))[1] for _ in range(10000)])
            rowr = mat.max(1) - mat.min(1); colr = mat.max(0) - mat.min(0)
            rowr_b = matb.max(2) - matb.min(2); colr_b = matb.max(1) - matb.min(1)
            g[lab] = dict(R=mat.tolist(), shares=dict(init=sh[0], data=sh[1], resid=sh[2]),
                          shares_ci=dict(init=ci(bsh[:, 0]), data=ci(bsh[:, 1]), resid=ci(bsh[:, 2])),
                          p_init_within_data=float((1 + (pi_ >= sh[0] - 1e-12).sum()) / 10001), p_data_within_init=float((1 + (pd_ >= sh[1] - 1e-12).sum()) / 10001),
                          p_init_full=float((1 + (pf[:, 0] >= sh[0] - 1e-12).sum()) / 10001), p_data_full=float((1 + (pf[:, 1] >= sh[1] - 1e-12).sum()) / 10001),
                          p_resid_ge_obs_full=float((1 + (pf[:, 2] >= sh[2] - 1e-12).sum()) / 10001), null_mean_shares=pf.mean(0).tolist(),
                          max_change_order_alone=float(rowr.max()), max_change_order_alone_ci=ci(rowr_b.max(1)),
                          max_change_init_alone=float(colr.max()), max_change_init_alone_ci=ci(colr_b.max(1)), sd=float(mat.std(ddof=1)))
            log(f'grid {m} [{lab}]: {np.round(mat, 3).tolist()} shares {np.round(sh, 3)} p init {g[lab]["p_init_within_data"]:.3f} '
                f'data {g[lab]["p_data_within_init"]:.3f} resid>=obs {g[lab]["p_resid_ge_obs_full"]:.3f}; max change order-alone {rowr.max():.3f} init-alone {colr.max():.3f}')
        R = np.array(g['R']['R']).ravel(); RL = np.array(g['R_logit']['R']).ravel()
        va = np.array([[ix(i, d)['val_acc'][-1] for d in range(3)] for i in range(3)]).ravel()
        vl = np.array([[ix(i, d)['val_loss'][-1] for d in range(3)] for i in range(3)]).ravel()
        for nm, v in (('R_valacc', va), ('R_valloss', vl), ('R_Rlogit', RL)):
            rho = spearman(R, v); pp = np.array([spearman(R, r3.permutation(v)) for _ in range(5000)])
            g[f'spearman_{nm}'] = dict(rho=rho, p=float((1 + (np.abs(pp) >= abs(rho) - 1e-12).sum()) / 5001))
        for e in (3, 10):
            j = [3, 10, 30].index(e); A = np.array([[ix(i, d)['R'][j] for d in range(3)] for i in range(3)]).ravel()
            rho = spearman(A, R); pp = np.array([spearman(A, r3.permutation(R)) for _ in range(5000)])
            g[f'spearman_ep{e}_final'] = dict(rho=rho, p=float((1 + (np.abs(pp) >= abs(rho) - 1e-12).sum()) / 5001))
        log(f'grid {m}: ' + str({k: v for k, v in g.items() if k.startswith('spearman')}))
        out['grid'][m] = g
    for a_, b_ in (('tf-2b-d32-n64__seed0', 'tf-2b-d32-n64-e100__seed0'), ('tf-2b-d32-n64-p30__seed1', 'tf-2b-d32-n64-e100__seed1')):
        A, Bq = res[a_], res[b_]; db = Bq['Rb'][-1] - A['Rb'][-1]; dbl = Bq['RLb'][-1] - A['RLb'][-1]
        out['pairs'][f'{a_} vs {b_}'] = dict(R=[A['R'][-1], Bq['R'][-1]], diff=Bq['R'][-1] - A['R'][-1], diff_ci=ci(db), p=bp(db),
                                             R_long_at_30=Bq['R'][Bq['epochs'].index(30)], R_logit=[A['R_logit'][-1], Bq['R_logit'][-1]],
                                             logit_diff_ci=ci(dbl), logit_p=bp(dbl))
        log('pair', a_, b_, out['pairs'][f'{a_} vs {b_}'])
    fo = OUT / 'x14_verify.json'; fo.write_text(json.dumps(X.clean(out), indent=1)); log('wrote', fo)


if __name__ == '__main__':
    main()
