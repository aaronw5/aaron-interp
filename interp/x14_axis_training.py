"""X14 (claims A10s, A10, A10m, A4; exploratory; "formation over training", item 5 of the what-to-look-for checklist):
what sets the axis-tied share R in transformer-type networks: per-epoch R from saved checkpoints, and an init-seed x
data-order-seed factorial.

R (X13 definition, code imported from interp.x13_axis_size): ridge read-out of log m (invariant mass of the n particles) on
the last hidden layer (input of the output layer), fitted on the unedited fit half of sample(per_class=1000, seed=1313)
(X13's jets and split), applied to the test half (2,500 jets) before / after an edit;
  R = mean_test[ mean_k d(T_k) ] / mean_test[ d(dil) ]
with T_k a rigid 0.06 rad translation of the real particles at azimuth 2 pi k / 8 (k = 0 eta+, 2 phi+, 4 eta-, 6 phi-) and dil
the matched dilation (angles x s per jet, 1/2 dlog(M0 M2) matched to its azimuth mean under T).  The read-out is refitted at
every checkpoint (it measures what the network's last layer encodes at that epoch).  R_dir = mean d(T_k) / mean d(dil) per
direction.  Null: rigid rotation about the stored axis (rot / dil).  Calls: W -> Z fraction among true W jets (1,000 W in the
5,000 jets) under T (mean of 8 azimuths), dil, rot.  Radial-only edit (R_rad) at epoch 0 and the best epoch only.
Input-only oracle references (no network; X13v): true log m (R 0), stored-axis moment 1/2 log(M0 M2) (R 1).

Arms
  ckpt  existing small-tagger runs (models/small/<run>/seed<k>/ckpt/epoch_<k>.pt; train_small.py): transformers and Linformers
        of X13's S64 / S32 lists (tf-1b/2b-d16/d32, linf-1b-d32-k4, linf-2b-d32-k8, N 64 and 32, all seeds; tf-2b-d32/d64 e100
        and the p30 reruns) and dsets-d64 (set-network reference).  Every epoch; e100 runs: epochs 0-30, then every 5th.
  grid  tf-2b-d32 and linf-2b-d32-k8, N 64, init seed {0,1,2} x data-order seed {0,1,2} (train_small.py --data_seed,
        30 epochs, patience 30; runs <model>-n64-x14d<data seed>/seed<init seed>).
  ana   analysis of everything measured -> $AI_RESULTS/jedi/x14_summary.json:
        per run and epoch R, R_dir, rot/dil, flows with jet-bootstrap CIs (B = 1000, the same jet indices for every run);
        settle epoch (first epoch after which |R_t - R_last| < 0.1 at every later evaluated epoch; bootstrap CI);
        Spearman of R at epochs 1-3 with final R across runs; drift after the val-loss plateau; grid: two-way variance
        decomposition of final R into init / data order / residual with permutation tests (10,000; full and restricted
        within the other factor); Spearman of final R with val accuracy / val loss across the 9 grid runs per model;
        reproduction pairs (grid cell (s, s) vs the earlier run with the same init and data seed).
Measurement is split into parts (balanced by estimated cost) so each part fits one 1 CPU / 2 GiB job:
  AI_PY=conda ./job.sh x14-ckpt-<i> 'python -m interp.x14_axis_training --arm ckpt --part <i> --nparts 8'
  AI_PY=conda ./job.sh x14-grid-<i> 'python -m interp.x14_axis_training --arm grid --part <i> --nparts 6'
  AI_PY=conda ./job.sh x14-ana 'python -m interp.x14_axis_training --arm ana'
Outputs: $AI_RESULTS/jedi/x14_runs/<run>__seed<k>.npz (per-epoch per-jet read-out changes, W-jet call flags),
         $AI_RESULTS/jedi/x14_runs/<run>__seed<k>.json (point values), $AI_RESULTS/jedi/x14_summary.json.
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import interp.x13_axis_size as X

OUT = X.OUT; RUNS = OUT / 'x14_runs'; SMALL = X.SMALL
B = 1000; T0 = time.time(); W_, Z_ = 2, 3
ED = [f'T{k}' for k in range(8)] + ['dil', 'rot']
DIRS = dict(eta_plus=0, phi_plus=2, eta_minus=4, phi_minus=6)
ARM1 = ['tf-1b-d16-n64', 'tf-1b-d32-n64', 'tf-2b-d16-n64', 'tf-2b-d32-n64', 'tf-2b-d16-n64-p30', 'tf-2b-d32-n64-p30',
        'tf-2b-d32-n64-e100', 'tf-2b-d64-n64-e100', 'linf-1b-d32-k4-n64', 'linf-2b-d32-k8-n64', 'dsets-d64-n64',
        'tf-1b-d16-n32', 'tf-1b-d32-n32', 'tf-2b-d16-n32', 'tf-2b-d32-n32', 'linf-1b-d32-k4-n32', 'linf-2b-d32-k8-n32',
        'dsets-d64-n32']
GRID_MODELS = ['tf-2b-d32', 'linf-2b-d32-k8']
GRID = [f'{m}-n64-x14d{d}' for m in GRID_MODELS for d in (0, 1, 2)]
# seconds per 5,000-jet forward on 1 CPU (X13 s64 / s32 job: network seconds / 21 edits)
FWD_S = {'tf-1b-d16-n64': 1.1, 'tf-1b-d32-n64': 1.05, 'tf-2b-d16-n64': 1.5, 'tf-2b-d32-n64': 2.4, 'tf-2b-d64-n64': 6.2,
         'linf-1b-d32-k4-n64': 1.3, 'linf-2b-d32-k8-n64': 2.4, 'dsets-d64-n64': 0.45, 'tf-1b-d16-n32': 0.5, 'tf-1b-d32-n32': 0.8,
         'tf-2b-d16-n32': 0.85, 'tf-2b-d32-n32': 1.45, 'linf-1b-d32-k4-n32': 0.72, 'linf-2b-d32-k8-n32': 1.3, 'dsets-d64-n32': 0.26}


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def base_key(run):
    for k in sorted(FWD_S, key=len, reverse=True):
        if run.startswith(k): return k
    return None


def ckpt_epochs(d):
    eps = sorted(int(p.stem.split('_')[1]) for p in (d / 'ckpt').glob('epoch_*.pt'))
    arch = json.loads((d / 'arch.json').read_text()); best = arch.get('best', {}).get('epoch')
    if len(eps) > 40:   # e100: epochs 0-30, then every 5th, plus the last and the best
        eps = [e for e in eps if e <= 30 or e % 5 == 0 or e == eps[-1] or e == best]
    return eps, best


def run_list(names):
    out = []
    for run in names:
        for p in sorted((SMALL / run).glob('seed*')):
            if (p / 'test.json').exists() and (p / 'arch.json').exists() and (p / 'ckpt').is_dir():   # finished runs only
                eps, _ = ckpt_epochs(p); out.append((run, p.name, len(eps) * (len(ED) + 1) * FWD_S.get(base_key(run), 2.0)))
    return out


def parts(items, nparts):
    """longest-processing-time partition of (run, seed, cost) into nparts lists"""
    bins = [[0.0, []] for _ in range(nparts)]
    for it in sorted(items, key=lambda t: -t[2]):
        b = min(bins, key=lambda q: q[0]); b[0] += it[2]; b[1].append(it)
    return bins


def read_log(d):
    rows = list(csv.DictReader(open(d / 'log.csv')))
    return {int(r['epoch']): dict(val_loss=float(r['val_loss']), val_acc=float(r['val_acc'])) for r in rows}


# ================================================================================================ measurement
def measure_run(run, sd, M, S, cache):
    import torch
    d = SMALL / run / sd; arch = json.loads((d / 'arch.json').read_text()); n = arch['n_particles']
    fo = RUNS / f'{run}__{sd}.npz'
    if fo.exists(): log('skip (exists)', fo.name); return
    assert not arch['cls_token'] and not arch['inputs'].get('recentre') and not arch['inputs'].get('aug') and arch['inputs'].get('axis', 'stored') == 'stored'
    if cache.get('n') != n:
        E = X.make_edits(S['x128'][:, :n]); cache.clear(); cache['n'] = n
        cache['E'] = {k: E[k] for k in ['0'] + ED + [f'R{k}' for k in range(8)]}
        cache['logm'] = np.log(np.maximum(X.moments(S['x128'][:, :n])['m'], 1e-3))
        log(f'edits for n={n} ready')
    E, logm, fit, te, y = cache['E'], cache['logm'], S['fit'], S['te'], S['y']
    net = M.build_model(arch['spec'], n, cls_token=False, final_ln=arch['final_ln']); f = X.torch_runner(net, arch)
    eps, best = ckpt_epochs(d); L = read_log(d); W = y == W_
    res = dict(epochs=[], acc=[], r2=[], alpha=[], dT=[], dD=[], dRot=[], zW=[], Rrad={})
    t1 = time.time()
    for e in eps:
        net.load_state_dict(torch.load(d / 'ckpt' / f'epoch_{e}.pt', map_location='cpu')); net.eval()
        R0 = f(E['0']); c0 = R0['logits'].argmax(1)
        rd = X.Ridge(R0['last'][fit], logm[fit]); p0 = rd(R0['last'][te])
        r2 = float(1 - ((p0 - logm[te]) ** 2).sum() / ((logm[te] - logm[te].mean()) ** 2).sum())
        dd, zz = {}, {'0': (c0[W] == Z_)}
        for k in ED:
            Re = f(E[k]); dd[k] = rd(Re['last'][te]) - p0; zz[k] = Re['logits'].argmax(1)[W] == Z_
        if e in (0, best):
            dR = np.mean([rd(f(E[f'R{k}'])['last'][te]) - p0 for k in range(8)], 0)
            res['Rrad'][e] = dict(R_rad=float(dR.mean() / dd['dil'].mean()), dR=dR.astype(np.float32))
        res['epochs'].append(e); res['acc'].append(float((c0 == y).mean())); res['r2'].append(r2); res['alpha'].append(rd.alpha)
        res['dT'].append(np.stack([dd[f'T{k}'] for k in range(8)]).astype(np.float32)); res['dD'].append(dd['dil'].astype(np.float32))
        res['dRot'].append(dd['rot'].astype(np.float32))
        res['zW'].append(np.stack([zz['0'], np.mean([zz[f'T{k}'] for k in range(8)], 0), zz['dil'], zz['rot']]).astype(np.float32))
        Rq = float(np.mean(res['dT'][-1]) / np.mean(dd['dil']))
        log(f'{run}/{sd} epoch {e:3d}: acc {res["acc"][-1]:.4f} val_acc {L.get(e, {}).get("val_acc")} R² {r2:.3f} R {Rq:+.3f} '
            f'rot/dil {dd["rot"].mean() / dd["dil"].mean():+.3f} W->Z T {res["zW"][-1][1].mean() - res["zW"][-1][0].mean():+.3f} '
            f'[{time.time() - t1:.0f}s]')
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fo, epochs=np.array(res['epochs']), acc=np.array(res['acc']), r2=np.array(res['r2']), alpha=np.array(res['alpha']),
                        dT=np.stack(res['dT']), dD=np.stack(res['dD']), dRot=np.stack(res['dRot']), zW=np.stack(res['zW']),
                        **{f'dRrad_{e}': v['dR'] for e, v in res['Rrad'].items()})
    meta = dict(run=run, seed_dir=sd, n=n, arch_spec=arch['spec'], best_epoch=best, epochs_run=arch.get('best', {}).get('epochs_run'),
                init_seed=arch['training'].get('init_seed', arch['seed']), data_seed=arch['training'].get('data_seed', arch['seed']),
                patience=arch['training']['patience'], log={str(k): v for k, v in L.items()},
                R_rad={str(e): v['R_rad'] for e, v in res['Rrad'].items()},
                test_acc=json.loads((d / 'test.json').read_text()).get('accuracy') if (d / 'test.json').exists() else None,
                seconds=time.time() - t1)
    fo.with_suffix('.json').write_text(json.dumps(X.clean(meta), indent=1)); log('wrote', fo.name, f'{len(eps)} epochs, {meta["seconds"]:.0f}s')


def run_measure(a):
    import importlib.util, torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    items = run_list(ARM1 if a.arm == 'ckpt' else GRID)
    if a.only:   # explicit list of <run>__seed<k>
        want = set(a.only.split(',')); mine = [t for t in items if f'{t[0]}__{t[1]}' in want]; P = {a.part: [sum(t[2] for t in mine), mine]}
    else:
        P = parts(items, a.nparts); mine = sorted(P[a.part][1], key=lambda t: t[0])
    log(f'arm {a.arm}: {len(items)} runs, part {a.part}/{a.nparts}: {[(r, s, round(c)) for r, s, c in mine]} est {P[a.part][0] / 60:.0f} min')
    x128, y, fit, te = X.load_sample(); S = dict(x128=x128, y=y, fit=fit, te=te); cache = {}
    log(f'sample(per_class={X.PER}, seed={X.SEED}): N {len(y)}, fit {len(fit)}, test {len(te)}')
    for run, sd, _ in sorted(mine, key=lambda t: (int(t[0].split('-n')[-1][:2]), t[0])):
        measure_run(run, sd, M, S, cache)


# ================================================================================================ analysis
def rank(v):
    v = np.asarray(v, float); o = np.argsort(v, kind='mergesort'); r = np.empty(len(v)); r[o] = np.arange(1, len(v) + 1)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1: r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4: return np.nan
    return float(np.corrcoef(rank(a[ok]), rank(b[ok]))[0, 1])


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) else [None, None]


def bp(v, ref=0.0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(min(1.0, 2 * (1 + min((v <= ref).sum(), (v >= ref).sum())) / (len(v) + 1))) if len(v) else None


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def settle_epoch(eps, R, tol=0.1):
    """first evaluated epoch e such that |R_t - R_last| < tol for every evaluated t >= e"""
    bad = np.abs(R - R[-1]) >= tol
    idx = np.flatnonzero(bad)
    return int(eps[0]) if not len(idx) else int(eps[min(idx[-1] + 1, len(eps) - 1)])


def oracle(S):
    E = X.make_edits(S['x128'][:, :64]); te = S['te']
    def feats(x):
        Mo = X.moments(x[te]); return dict(logm=np.log(np.maximum(Mo['m'], 1e-3)), axisM2=0.5 * np.log(Mo['M0'] * Mo['M2']))
    F = {k: feats(E[k]) for k in ['0'] + ED + [f'R{k}' for k in range(8)]}; out = {}
    for q in ('logm', 'axisM2'):
        d = {k: F[k][q] - F['0'][q] for k in F if k != '0'}; dD = d['dil'].mean()
        out[q] = dict(R=float(np.mean([d[f'T{k}'] for k in range(8)]) / dD),
                      R_dir={n: float(d[f'T{k}'].mean() / dD) for n, k in DIRS.items()},
                      R_rad=float(np.mean([d[f'R{k}'] for k in range(8)]) / dD), rot_over_dil=float(d['rot'].mean() / dD))
        log(f'oracle {q}: {out[q]}')
    return out


def analyze(a):
    x128, y, fit, te = X.load_sample(); S = dict(x128=x128, y=y, fit=fit, te=te)
    rng = np.random.default_rng(141430)
    I = rng.integers(0, len(te), (B, len(te))); CNT = np.stack([np.bincount(i, minlength=len(te)) for i in I]).astype(np.float64) / len(te)
    nW = int((y == W_).sum()); IW = rng.integers(0, nW, (B, nW)); CW = np.stack([np.bincount(i, minlength=nW) for i in IW]).astype(np.float64) / nW
    Summ = dict(sample=dict(per_class=X.PER, seed=X.SEED, N=len(y), N_test=len(te), N_W=nW), B=B, c=X.C, settle_tol=0.1, oracle=oracle(S), runs={})
    x13 = {}
    for g in ('s64', 's32'):
        f = OUT / f'x13_{g}.json'
        if f.exists(): x13.update({r['name']: r for r in json.loads(f.read_text())['rows']})
    REP = {}
    for fo in sorted(RUNS.glob('*.npz')):
        key = fo.stem; meta = json.loads(fo.with_suffix('.json').read_text()); z = np.load(fo)
        eps = z['epochs']; dT = z['dT'].astype(np.float64); dD = z['dD'].astype(np.float64); dRot = z['dRot'].astype(np.float64)
        Tm = dT.mean(1)                                        # (n_ep, J) azimuth mean
        mT, mD, mRot = Tm.mean(1), dD.mean(1), dRot.mean(1)
        R = mT / mD; bT, bD, bRot = CNT @ Tm.T, CNT @ dD.T, CNT @ dRot.T   # (B, n_ep)
        Rb = bT / bD
        Rdir = {n: dT[:, k].mean(1) / mD for n, k in DIRS.items()}; Rdir_b = {n: (CNT @ dT[:, k].T) / bD for n, k in DIRS.items()}
        zW = z['zW'].astype(np.float64)                         # (n_ep, 4: 0, T, dil, rot, nW)
        fl = {nm: (zW[:, i] - zW[:, 0]).mean(1) for i, nm in ((1, 'T'), (2, 'dil'), (3, 'rot'))}
        flb = {nm: CW @ (zW[:, i] - zW[:, 0]).T for i, nm in ((1, 'T'), (2, 'dil'), (3, 'rot'))}
        se = settle_epoch(eps, R); seb = np.array([settle_epoch(eps, Rb[b]) for b in range(B)])
        L = meta['log']; vl = np.array([L.get(str(int(e)), {}).get('val_loss', np.nan) for e in eps])
        va = np.array([L.get(str(int(e)), {}).get('val_acc', np.nan) for e in eps])
        # val-loss plateau: first evaluated epoch with val loss within 1 % of the run's minimum (over evaluated epochs >= 1)
        ok1 = eps >= 1; vmin = np.nanmin(vl[ok1]); ip = int(np.flatnonzero(ok1 & (vl <= 1.01 * vmin))[0])
        drift = Rb[:, -1] - Rb[:, ip]
        best = meta['best_epoch']; ib = int(np.flatnonzero(eps == best)[0]) if best in eps else None
        nm13 = f'{meta["run"]}_s{int(meta["seed_dir"][4:])}'
        r = dict(meta={k: v for k, v in meta.items() if k != 'log'}, epochs=eps.tolist(), R=R.tolist(), R_ci=[ci(Rb[:, i]) for i in range(len(eps))],
                 R_dir={n: v.tolist() for n, v in Rdir.items()}, R_dir_ci={n: [ci(v[:, i]) for i in range(len(eps))] for n, v in Rdir_b.items()},
                 rot_over_dil=(mRot / mD).tolist(), rot_over_dil_ci=[ci((bRot / bD)[:, i]) for i in range(len(eps))],
                 T_minus_rot_p=[bp(bT[:, i] - bRot[:, i]) for i in range(len(eps))],
                 flow=({k: v.tolist() for k, v in fl.items()}), flow_ci={k: [ci(v[:, i]) for i in range(len(eps))] for k, v in flb.items()},
                 flowT_minus_rot_p=[bp(flb['T'][:, i] - flb['rot'][:, i]) for i in range(len(eps))],
                 acc_sample=z['acc'].tolist(), readout_r2=z['r2'].tolist(), val_loss=vl.tolist(), val_acc=va.tolist(),
                 settle_epoch=se, settle_ci=ci(seb), settle_frac=se / float(eps[-1]), R_last=float(R[-1]), R_last_ci=ci(Rb[:, -1]),
                 R_epoch0=float(R[0]), R_epoch0_ci=ci(Rb[:, 0]), plateau_epoch=int(eps[ip]), R_plateau=float(R[ip]),
                 drift_after_plateau=float(R[-1] - R[ip]), drift_ci=ci(drift), drift_p=bp(drift),
                 R_best=float(R[ib]) if ib is not None else None, R_best_ci=ci(Rb[:, ib]) if ib is not None else None,
                 acc_best=float(z['acc'][ib]) if ib is not None else None,
                 x13=dict(R=x13.get(nm13, {}).get('quick', {}).get('R'), acc=x13.get(nm13, {}).get('acc')) if nm13 in x13 else None)
        Summ['runs'][key] = r; REP[key] = dict(Rb=Rb, eps=eps, R=R)
        log(f'{key:34s} ep {int(eps[-1]):3d} | R ep0 {R[0]:+.2f} ep1 {R[min(1, len(R) - 1)]:+.2f} ep3 {R[min(3, len(R) - 1)]:+.2f} last {R[-1]:+.2f} {ci(Rb[:, -1])} '
            f'| settle {se} {ci(seb)} | plateau ep {int(eps[ip])} drift {R[-1] - R[ip]:+.2f} {ci(drift)} | best ep {best} R {r["R_best"]} acc {r["acc_best"]} '
            f'| X13 {r["x13"]} | rot/dil last {(mRot / mD)[-1]:+.3f} | flow T {fl["T"][-1]:+.3f} rot {fl["rot"][-1]:+.3f}')
    runs = Summ['runs']
    hp = holm([v['drift_p'] for v in runs.values()])
    for (k, v), q in zip(runs.items(), hp): v['drift_p_holm'] = float(q)
    cls = lambda k: k.split('-')[0]
    # reproduction of X13 at the best epoch
    rep = [(k, v['R_best'], v['x13']['R'], v['acc_best'], v['x13']['acc']) for k, v in runs.items() if v.get('x13') and v['x13']['R'] is not None]
    Summ['x13_reproduction'] = dict(n=len(rep), max_abs_dR=max(abs(a_ - b_) for _, a_, b_, _, _ in rep) if rep else None,
                                    max_abs_dacc=max(abs(a_ - b_) for _, _, _, a_, b_ in rep) if rep else None)
    log('X13 reproduction at best epoch:', Summ['x13_reproduction'])
    # ------------------------------------------------------------------ early R vs final R across runs
    def across(keys, k_early, label):
        keys = [k for k in keys if k_early in REP[k]['eps']]
        if len(keys) < 4: return None
        ie = {k: int(np.flatnonzero(REP[k]['eps'] == k_early)[0]) for k in keys}
        A_ = np.array([REP[k]['R'][ie[k]] for k in keys]); B_ = np.array([REP[k]['R'][-1] for k in keys]); rho = spearman(A_, B_)
        RA = np.stack([REP[k]['Rb'][:, ie[k]] for k in keys], 1); RB = np.stack([REP[k]['Rb'][:, -1] for k in keys], 1)
        r2 = np.random.default_rng(141440); bs = []
        for b in range(B):
            j = r2.integers(0, len(keys), len(keys)); bs.append(spearman(RA[b, j], RB[b, j]))
        perm = np.array([spearman(A_, r2.permutation(B_)) for _ in range(10000)])
        out = dict(n=len(keys), rho=rho, ci=ci(bs), p_perm=float((1 + np.sum(np.abs(perm) >= abs(rho) - 1e-12)) / 10001),
                   p_gt07=float(np.mean(np.array(bs) <= 0.7)), runs=keys)
        log(f'Spearman R(epoch {k_early}) vs R(last) [{label}]: {rho:+.3f} {out["ci"]} n {len(keys)} perm p {out["p_perm"]:.4f}')
        return out
    arm1 = [k for k in runs if '-x14d' not in k]; grid = [k for k in runs if '-x14d' in k]
    att1 = [k for k in arm1 if cls(k) in ('tf', 'linf')]; ds1 = [k for k in arm1 if cls(k) == 'dsets']
    EA = {}
    for lab, ks in (('arm1_attention', att1), ('grid', grid), ('attention_all', att1 + grid), ('grid_tf', [k for k in grid if k.startswith('tf')]),
                    ('grid_linf', [k for k in grid if k.startswith('linf')])):
        EA[lab] = {str(e): across(ks, e, lab) for e in (0, 1, 2, 3, 5, 10)}
    Summ['early_vs_final'] = EA
    # ------------------------------------------------------------------ settle epochs and seed spread by class
    SE = {}
    for lab, ks in (('dsets', ds1), ('linf', [k for k in arm1 if cls(k) == 'linf']), ('tf', [k for k in arm1 if cls(k) == 'tf']),
                    ('grid_tf', [k for k in grid if k.startswith('tf')]), ('grid_linf', [k for k in grid if k.startswith('linf')])):
        if not ks: continue
        st = [runs[k]['settle_epoch'] for k in ks]; fr = [runs[k]['settle_frac'] for k in ks]; dr = [abs(runs[k]['drift_after_plateau']) for k in ks]
        SE[lab] = dict(n=len(ks), settle_median=float(np.median(st)), settle_range=[int(min(st)), int(max(st))], settle_frac_median=float(np.median(fr)),
                       abs_drift_after_plateau_median=float(np.median(dr)), n_drift_holm_lt05=int(sum(runs[k]['drift_p_holm'] < 0.05 for k in ks)))
        log(f'settle [{lab}]: {SE[lab]}')
    # seed spread of final R within family (arm 1): SD over seeds of the same run name
    fam = {}
    for k in arm1: fam.setdefault(k.split('__')[0], []).append(runs[k]['R_last'])
    SE['seed_spread_by_family'] = {f: dict(R_last=v, sd=float(np.std(v, ddof=1)) if len(v) > 1 else None) for f, v in fam.items()}
    Summ['settle'] = SE
    # ------------------------------------------------------------------ grid: two-way decomposition
    G = {}
    for m in GRID_MODELS:
        ks = {(runs[k]['meta']['init_seed'], runs[k]['meta']['data_seed']): k for k in grid if k.startswith(m + '-n64')}
        if len(ks) < 9: log(f'grid {m}: {len(ks)} of 9 runs measured; skipped'); continue
        mat = np.array([[runs[ks[(i, d)]]['R_last'] for d in range(3)] for i in range(3)])
        matb = np.stack([np.stack([REP[ks[(i, d)]]['Rb'][:, -1] for d in range(3)], 1) for i in range(3)], 1)   # (B, 3, 3)
        def shares(Mx):
            g = Mx.mean(); sst = ((Mx - g) ** 2).sum(); si = 3 * ((Mx.mean(1) - g) ** 2).sum(); sdd = 3 * ((Mx.mean(0) - g) ** 2).sum()
            return si / sst, sdd / sst, (sst - si - sdd) / sst
        si, sdd, sr = shares(mat); r3 = np.random.default_rng(141450)
        pf = np.array([shares(r3.permutation(mat.ravel()).reshape(3, 3)) for _ in range(10000)])
        p_init_full = float((1 + np.sum(pf[:, 0] >= si - 1e-12)) / 10001); p_data_full = float((1 + np.sum(pf[:, 1] >= sdd - 1e-12)) / 10001)
        # restricted: permute init labels within each data-order column (keeps the data effect), and vice versa
        pi_ = np.array([shares(np.stack([r3.permutation(mat[:, d]) for d in range(3)], 1))[0] for _ in range(10000)])
        pd_ = np.array([shares(np.stack([r3.permutation(mat[i]) for i in range(3)], 0))[1] for _ in range(10000)])
        bsh = np.array([shares(matb[b]) for b in range(B)])
        # F statistics (two-way without replication; residual df 4)
        Fi = (si / 2) / (sr / 4) if sr > 0 else np.inf; Fd = (sdd / 2) / (sr / 4) if sr > 0 else np.inf
        vl = np.array([[runs[ks[(i, d)]]['val_loss'][-1] for d in range(3)] for i in range(3)])
        va = np.array([[runs[ks[(i, d)]]['val_acc'][-1] for d in range(3)] for i in range(3)])
        G[m] = dict(R=mat.tolist(), R_ci=[[runs[ks[(i, d)]]['R_last_ci'] for d in range(3)] for i in range(3)], rows='init seed 0-2', cols='data seed 0-2',
                    share_init=float(si), share_data=float(sdd), share_resid=float(sr), share_ci=dict(init=ci(bsh[:, 0]), data=ci(bsh[:, 1]), resid=ci(bsh[:, 2])),
                    p_init_full=p_init_full, p_data_full=p_data_full,
                    p_init_within_data=float((1 + np.sum(pi_ >= si - 1e-12)) / 10001), p_data_within_init=float((1 + np.sum(pd_ >= sdd - 1e-12)) / 10001),
                    F_init=float(Fi), F_data=float(Fd), sd_R=float(mat.std(ddof=1)), R_range=[float(mat.min()), float(mat.max())],
                    val_loss=vl.tolist(), val_acc=va.tolist(),
                    spearman_R_valacc=spearman(mat.ravel(), va.ravel()), spearman_R_valloss=spearman(mat.ravel(), vl.ravel()),
                    settle=[[runs[ks[(i, d)]]['settle_epoch'] for d in range(3)] for i in range(3)],
                    R_epoch0=[[runs[ks[(i, d)]]['R_epoch0'] for d in range(3)] for i in range(3)])
        for nm, v in (('valacc', va), ('valloss', vl)):
            rho = G[m][f'spearman_R_{nm}']; pp = np.array([spearman(mat.ravel(), r3.permutation(v.ravel())) for _ in range(10000)])
            G[m][f'spearman_R_{nm}_p'] = float((1 + np.sum(np.abs(pp) >= abs(rho) - 1e-12)) / 10001)
            bb = [spearman(matb[b].ravel(), v.ravel()) for b in range(B)]; G[m][f'spearman_R_{nm}_jetci'] = ci(bb)
        # init-seed effect at epoch 0 (untrained): the same init gives the same epoch-0 network for every data seed
        log(f'grid {m}: R (rows init, cols data) {np.round(mat, 3).tolist()} shares init {si:.3f} data {sdd:.3f} resid {sr:.3f} '
            f'p init {G[m]["p_init_within_data"]:.4f} (full {p_init_full:.4f}) data {G[m]["p_data_within_init"]:.4f} (full {p_data_full:.4f}); '
            f'rho(R, val acc) {G[m]["spearman_R_valacc"]:+.3f} p {G[m]["spearman_R_valacc_p"]:.3f}; rho(R, val loss) {G[m]["spearman_R_valloss"]:+.3f} p {G[m]["spearman_R_valloss_p"]:.3f}')
    Summ['grid'] = G
    # ------------------------------------------------------------------ reproduction pairs (GPU run-to-run): same init and data seed
    RP = {}
    for a_, b_ in (('tf-2b-d32-n64__seed0', 'tf-2b-d32-n64-x14d0__seed0'), ('tf-2b-d32-n64-p30__seed1', 'tf-2b-d32-n64-x14d1__seed1'),
                   ('linf-2b-d32-k8-n64__seed0', 'linf-2b-d32-k8-n64-x14d0__seed0'), ('linf-2b-d32-k8-n64__seed1', 'linf-2b-d32-k8-n64-x14d1__seed1')):
        if a_ in runs and b_ in runs:
            n_ = min(len(runs[a_]['epochs']), len(runs[b_]['epochs']))
            dvl = np.nanmax(np.abs(np.array(runs[a_]['val_loss'][:n_]) - np.array(runs[b_]['val_loss'][:n_])))
            db = REP[a_]['Rb'][:, n_ - 1] - REP[b_]['Rb'][:, n_ - 1]
            dR0 = runs[a_]['R'][0] - runs[b_]['R'][0]
            RP[f'{a_} vs {b_}'] = dict(R=[runs[a_]['R'][n_ - 1], runs[b_]['R'][n_ - 1]], diff_ci=ci(db), p=bp(db), max_abs_dvalloss=float(dvl),
                                       R_epoch0_diff=float(dR0), R_epoch1=[runs[a_]['R'][1], runs[b_]['R'][1]])
            log(f'reproduction {a_} vs {b_}: {RP[f"{a_} vs {b_}"]}')
    Summ['reproduction_pairs'] = RP
    fo = OUT / 'x14_summary.json'; fo.write_text(json.dumps(X.clean(Summ), indent=1)); log('wrote', fo)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True, choices=['ckpt', 'grid', 'ana'])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--only', default='', help='comma-separated <run>__seed<k> to measure (overrides --part)')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    analyze(a) if a.arm == 'ana' else run_measure(a)
