"""X9w (claims A10m / A4 / D4; review weakness 1, novelty vs EFN/PFN): is the un-recentred M2-like mass response (W -> Z under a rigid
0.06 rad translation of all particles) a consequence of the stored-axis input representation, the same in any set network, or
specific to JEDI-linear's architecture / HGQ training?  2 x 2 design: input representation {stored axis, recentred} x training
{no augmentation, translation augmentation}, in four architectures.

Models (training flags are opt-in; official checkpoints untouched):
  pfn    energyflow PFN, Phi 64-64-32, F 100-100-100 (interp/x9_train.py)       $AI_RESULTS/jedi/x9_models/pfn_64-64-32_s<k><tag>
  efn    energyflow EFN, Phi 100-100-128 (interp/x9_train.py)                    $AI_RESULTS/jedi/x9_models/efn_100-100-128_s<k><tag>
  dsets  Deep Sets d64, N 64 (train/train_small.py)                              models/small/dsets-d64-n64<tag>/seed<k>
  jedi   JEDI-linear n64 HGQ-50 (train/train_hgq_jedi.py, X12 arm hgq)           models/small/jedi-n64-x12-hgq<tag>/seed0
  tags: '' = stored / no aug (the X9, small-model and X12 checkpoints), -aug, -rc, -rcaug; seeds 0, 1 (jedi: 0; jedi stored and
  aug also seeds 1, 2, added after the seed-0 aug run reached 64.4 % test accuracy; JEDI aug vs stored: exact 3 vs 3 permutation).
  recentre: angles of the real particles about the pT-weighted centroid c = sum pT_i (deta_i, dphi_i) / sum pT_i (first 64 slots),
  applied inside the model's input pipeline (training and here); aug: per-jet rigid translation |c| ~ U(0, 0.08) rad, uniform
  direction, redrawn every epoch, training jets only (after recentring in -rcaug).
  Reference row: the official JEDI-linear 3-feature n64 checkpoint (interp.jedi_core) on the same jets.
Sample: interp.jedi_core.sample(per_class=2000, seed=909), N = 10,000 (2,000 true W).
Edits on the raw jets (padding never moved). For recentred models every edit is applied BEFORE recentring (the model's own
  input step), so a rigid translation is removed exactly up to float32 rounding; the reported quantity is that residual.
  T       every real particle +0.06 rad in deta (invariant mass unchanged)
  Tr      +0.06 rad in a random direction per jet (rng 9091)                     null: random-direction translation of equal length
  rot     rotation about the stored axis with mean displacement 0.06 rad (X6 rotate_matched, rng 9092; m and M2 unchanged)  null
  M2held  Tr, then angles scaled per jet by s = sqrt(M2 / M2(Tr)) about the stored axis (X9v construction; M0, M2 as before)  control
  dilM2   angles scaled per jet by sT = sqrt(M2(T) / M2) (X12v; M0, M2 as under T, invariant mass x sT): matched real change
Metrics (per model): accuracy on the sample (Wilson) and on the 260k test archive (training json); W -> Z = fraction of true-W
  jets called Z; flow = W -> Z(edit) - W -> Z(unedited) (2,000-draw bootstrap over W jets); calls changed (all jets); exact McNemar
  on the paired Z indicators (W jets): edit vs unedited and T vs rot; calibration ratio flow(T) / flow(dilM2) (paired bootstrap);
  max |delta logit| under T (the residual of recentred models). Holm over models for T vs rot and for M2held vs unedited.
Seed-level 2 x 2 tests (pfn, efn, dsets: 8 networks each): flow(T) and test accuracy; main effect of representation
  (mean stored cells - mean recentred cells), of augmentation (mean no-aug - mean aug), interaction ((stored - aug) - (rc - rcaug));
  exact permutation of the 8 networks over the cell labels (two-sided); pooled over the three architectures by a permutation
  stratified by architecture (20,000 draws). JEDI-linear: one seed per cell, no seed-level test.
Thesis predictions (input representation alone): stored/no-aug W -> Z 0.45-0.8 in every architecture; aug alone -> rotation level
  (< 0.05) at <= 1 point accuracy; recentred ~0; M2held <= 0.1 everywhere.  Alternative (architecture / training specific):
  response persists in stored/aug, or JEDI-linear differs from PFN/EFN/dsets in the same cell; an accuracy loss > 2 points under
  aug would mean the stored axis carries class information the networks use.
Modes:
  AI_PY=conda ./job.sh x9w-dsx 'python -m interp.x9w_repr_2x2 --export_dsets'   (torch env: Deep Sets state_dicts -> npz)
  ./job.sh x9w-run 'python -m interp.x9w_repr_2x2'                              -> $AI_RESULTS/jedi/x9w_repr_2x2.json
"""
import argparse, gc, itertools, json, os, sys, time
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'; DSX = OUT / 'x9w_models'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
W_, Z_ = 2, 3; T0 = time.time()
CELLS = {'stored': '', 'aug': '-aug', 'rc': '-rc', 'rcaug': '-rcaug'}
ARCHS = {'pfn': (0, 1), 'efn': (0, 1), 'dsets': (0, 1), 'jedi': (0,)}
JEDI_EXTRA = {'stored': (0, 1, 2), 'aug': (0, 1, 2)}   # HGQ-50 seeds 1, 2: X12 stored checkpoints + X9w aug runs (added after seed 0
                                                     # aug ended at 64.4 % test accuracy vs 78.4 % stored)


def seeds(a, c):
    return JEDI_EXTRA.get(c, ARCHS[a]) if a == 'jedi' else ARCHS[a]
SEED_SAMPLE = 909


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    return o


# ------------------------------------------------------------------------------------------------ statistics
RNG = np.random.default_rng(9090)


def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def bci(v, B=2000):
    v = np.asarray(v, np.float64); n = len(v); m = np.array([v[RNG.integers(0, n, n)].mean() for _ in range(B)])
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def bratio(a, b, B=2000):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64); n = len(a); r = []
    for _ in range(B):
        i = RNG.integers(0, n, n); d = b[i].mean(); r.append(a[i].mean() / d if d > 0 else np.nan)
    r = np.asarray(r); ok = np.isfinite(r)
    return [float(np.quantile(r[ok], .025)), float(np.quantile(r[ok], .975))] if ok.mean() > 0.9 else [None, None]


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans: n10 = a only, n01 = b only"""
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(a_only=0, b_only=0, p=1.0)
    k = min(n10, n01); p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return dict(a_only=n10, b_only=n01, p=float(p))


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]; order = sorted(idx, key=lambda i: ps[i]); m = len(order)
    out = [None] * len(ps); run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


def effects(v):
    """v: {cell: [values over seeds]} -> (representation, augmentation, interaction)"""
    m = {c: float(np.mean(v[c])) for c in CELLS}
    return dict(representation=(m['stored'] + m['aug']) / 2 - (m['rc'] + m['rcaug']) / 2,
                augmentation=(m['stored'] + m['rc']) / 2 - (m['aug'] + m['rcaug']) / 2,
                interaction=(m['stored'] - m['aug']) - (m['rc'] - m['rcaug']))


def perm_test(v):
    """exact permutation of the networks over the cell labels (equal cell sizes); two-sided p per effect"""
    cells = list(CELLS); k = len(v[cells[0]]); vals = np.concatenate([np.asarray(v[c], float) for c in cells]); obs = effects(v)
    cnt = {e: 0 for e in obs}; tot = 0
    for p in itertools.permutations(range(len(vals))):
        vv = {c: vals[list(p[i * k:(i + 1) * k])] for i, c in enumerate(cells)}; e = effects(vv); tot += 1
        for kk in obs: cnt[kk] += abs(e[kk]) >= abs(obs[kk]) - 1e-12
    return {kk: dict(effect=obs[kk], p=cnt[kk] / tot) for kk in obs}, tot


def perm2(a, b):
    """exact two-sided permutation of mean(a) - mean(b) over all relabellings"""
    v = np.asarray(list(a) + list(b), float); n = len(a); obs = np.mean(a) - np.mean(b); c = t = 0
    for S in itertools.combinations(range(len(v)), n):
        m = np.zeros(len(v), bool); m[list(S)] = True; t += 1; c += abs(v[m].mean() - v[~m].mean()) >= abs(obs) - 1e-12
    return dict(effect=float(obs), p=c / t, n_relabellings=t)


def perm_test_pooled(V, draws=20000, rng=np.random.default_rng(9093)):
    """V: {arch: {cell: [values]}}; statistic = mean over archs of the per-arch effect; labels permuted within arch"""
    archs = list(V); cells = list(CELLS)
    def stat(VV): return {e: float(np.mean([effects(VV[a])[e] for a in archs])) for e in ('representation', 'augmentation', 'interaction')}
    obs = stat(V); cnt = {e: 0 for e in obs}
    flat = {a: np.concatenate([np.asarray(V[a][c], float) for c in cells]) for a in archs}; k = {a: len(V[a][cells[0]]) for a in archs}
    for _ in range(draws):
        VV = {}
        for a in archs:
            p = rng.permutation(len(flat[a])); VV[a] = {c: flat[a][p[i * k[a]:(i + 1) * k[a]]] for i, c in enumerate(cells)}
        e = stat(VV)
        for kk in obs: cnt[kk] += abs(e[kk]) >= abs(obs[kk]) - 1e-12
    return {kk: dict(effect=obs[kk], p=(1 + cnt[kk]) / (1 + draws)) for kk in obs}


# ------------------------------------------------------------------------------------------------ inputs and edits
def recentre(x):
    """(J, n, 3) raw -> angles of the real particles about the pT-weighted centroid (same formula as the training flags)"""
    x = np.array(x, np.float32, copy=True); real = x[..., 0] > 0; pt = np.where(real, x[..., 0], 0).astype(np.float64)
    c = (pt[..., None] * x[..., 1:3]).sum(1) / np.maximum(pt.sum(1), 1e-12)[:, None]
    x[..., 1:3] = np.where(real[..., None], x[..., 1:3] - c[:, None, :].astype(np.float32), x[..., 1:3]); return x


def build_edits(x):
    from interp.x9_efn_pfn import moments, translate, rotate_matched, scale_angles
    J = len(x); M = moments(x); E = {}
    E['T'] = translate(x, 0.06, 0.0)
    E['Tr'] = translate(x, 0.06, np.random.default_rng(9091).uniform(0, 2 * np.pi, J))
    E['rot'] = rotate_matched(x, 0.06, np.random.default_rng(9092))
    E['M2held'] = scale_angles(E['Tr'], np.sqrt(M['M2'] / np.maximum(moments(E['Tr'])['M2'], 1e-30)))
    sT = np.sqrt(moments(E['T'])['M2'] / np.maximum(M['M2'], 1e-30)); E['dilM2'] = scale_angles(x, sT)
    hyg = {}; ok = M['m'] > 1; real = x[..., 0] > 0
    for k, v in E.items():
        Mv = moments(v); cen = np.hypot((v[..., 0] * v[..., 1]).sum(1), (v[..., 0] * v[..., 2]).sum(1)) / Mv['M0']
        hyg[k] = dict(dlogm_mean=float(np.log(Mv['m'][ok] / M['m'][ok]).mean()), dlogM2_mean=float(np.log(Mv['M2'] / M['M2']).mean()),
                      dlogM0_maxabs=float(np.abs(np.log(Mv['M0'] / M['M0'])).max()), centroid_offset_median=float(np.median(cen)),
                      padding_changed=bool((v[~real] != x[~real]).any()))
    hyg['dilM2']['scale_mean'] = float(sT.mean())
    return E, hyg


# ------------------------------------------------------------------------------------------------ models
def flow_dir(arch, seed, cell):
    return f'{"pfn_64-64-32" if arch == "pfn" else "efn_100-100-128"}_s{seed}{CELLS[cell]}'


class DSNumpy:
    """exact numpy forward of train/models.py DeepSetsTagger from the exported state_dict (torch Linear: y = x W^T + b)"""
    def __init__(self, npz):
        d = np.load(npz); self.P = {k: d[k].astype(np.float64) for k in d.files}
        self.sh, self.sc = self.P['_shift'].astype(np.float32), self.P['_scale'].astype(np.float32)

    def logits(self, x, batch=2500):
        out = []
        for i in range(0, len(x), batch):
            xb = np.asarray(x[i:i + batch], np.float32); m = xb[..., 0] > 0
            h = (((xb - self.sh) / self.sc) * m[..., None]).astype(np.float64)
            lin = lambda h, k: h @ self.P[f'{k}.weight'].T + self.P[f'{k}.bias']
            h = np.maximum(lin(h, 'phi.0'), 0); h = np.maximum(lin(h, 'phi.2'), 0); S = (h * m[..., None]).sum(1)
            for k in ('rho.0', 'rho.2', 'rho.4'): S = np.maximum(lin(S, k), 0)
            out.append(lin(S, 'head'))
        return np.concatenate(out)


def load_model(arch, seed, cell):
    """-> (logits_fn on model-ready raw jets, info dict)"""
    if arch in ('pfn', 'efn'):
        from interp.x9_efn_pfn import Flow
        M = Flow(flow_dir(arch, seed, cell)); inf = M.info
        assert bool(inf.get('recentre', False)) == cell.startswith('rc') and (inf.get('aug', 0) > 0) == cell.endswith('aug'), inf
        return (lambda x: M.run(x)['logits']), dict(dir=str(M.dir), acc_test=inf['acc_test'], acc_dev=inf['acc_dev'],
                                                    best_epoch=inf['best_epoch'], epochs_run=inf['epochs_run'])
    if arch == 'dsets':
        d = SMALL / f'dsets-d64-n64{CELLS[cell]}' / f'seed{seed}'; t = json.loads((d / 'test.json').read_text())
        ar = json.loads((d / 'arch.json').read_text())['inputs']
        assert bool(ar.get('recentre', False)) == cell.startswith('rc') and (ar.get('aug', 0) > 0) == cell.endswith('aug'), ar
        M = DSNumpy(DSX / f'dsets-d64-n64{CELLS[cell]}_s{seed}.npz')
        return M.logits, dict(dir=str(d), acc_test=t['accuracy'], best_epoch=t.get('best_epoch'), test_probs=str(d / 'test_probs.npy'))
    if arch == 'jedi':
        from interp.x12_retrained_jedi import Net
        d = SMALL / f'jedi-n64-x12-hgq{CELLS[cell]}' / f'seed{seed}'; t = json.loads((d / 'test.json').read_text())
        assert bool(t.get('recentre') or False) == cell.startswith('rc') and ((t.get('aug') or 0) > 0) == cell.endswith('aug'), t.get('recentre')
        net = Net('x12', d / 'model.keras')
        return (lambda x: net.run(x)['logits']), dict(dir=str(d), acc_test=t['accuracy'], l1_zero_frac=t.get('l1_zero_frac'),
                                                      recentre_flag=t.get('recentre'), aug_flag=t.get('aug'), test_probs=str(d / 'test_probs.npy'))
    if arch == 'official':
        from interp.x12_retrained_jedi import Net
        net = Net('official'); return (lambda x: net.run(x)['logits']), dict(dir=str(net.path), acc_test=0.8092)
    raise ValueError(arch)


def sample_rows(per_class, seed):
    from jetdistill.config import RESULTS
    yy = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); r = np.random.default_rng(seed)
    return np.sort(np.concatenate([r.choice(np.flatnonzero(yy == c), per_class, replace=False) for c in range(5)]))


def evaluate(fn, rc, x, y, E, idx, info):
    prep = recentre if rc else (lambda v: v)
    W = y == W_; nW = int(W.sum()); J = len(y)
    lo0 = fn(prep(x)); c0 = lo0.argmax(1); z0 = c0[W] == Z_
    o = dict(acc=float((c0 == y).mean()), acc_ci=wilson(int((c0 == y).sum()), J), W_to_Z_unedited=float(z0.mean()),
             W_to_Z_unedited_ci=wilson(int(z0.sum()), nW), edits={})
    if 'test_probs' in info:            # stored softmax of the training job on the same test rows
        tp = np.load(info['test_probs'], mmap_mode='r')[idx]; p = np.exp(lo0 - lo0.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        o['check_test_probs'] = dict(max_abs_dp=float(np.abs(p - tp).max()), argmax_agree=float((p.argmax(1) == tp.argmax(1)).mean()))
    Zc, CH = {}, {}
    for k, xe in E.items():
        lo = fn(prep(xe)); ce = lo.argmax(1); ze = ce[W] == Z_; fl = ze.astype(float) - z0; ch = ce != c0; Zc[k] = ze; CH[k] = ch
        o['edits'][k] = dict(W_to_Z=float(ze.mean()), W_to_Z_ci=wilson(int(ze.sum()), nW), flow=float(fl.mean()), flow_ci=bci(fl),
                             calls_changed=float(ch.mean()), calls_changed_ci=wilson(int(ch.sum()), J),
                             W_to_gq=float(np.isin(ce[W], [0, 1]).mean()), max_abs_dlogit=float(np.abs(lo - lo0).max()),
                             mcnemar_vs_unedited=mcnemar(ze, z0))
    for k in E:
        if k != 'rot':
            o['edits'][k]['mcnemar_Z_vs_rot'] = mcnemar(Zc[k], Zc['rot'])
            o['edits'][k]['mcnemar_calls_vs_rot'] = mcnemar(CH[k], CH['rot'])
    fT = Zc['T'].astype(float) - z0; fD = Zc['dilM2'].astype(float) - z0; fR = Zc['rot'].astype(float) - z0
    o['ratio_T_over_dilM2'] = float(fT.mean() / fD.mean()) if fD.mean() > 0 else None
    o['ratio_T_over_dilM2_ci'] = bratio(fT, fD) if fD.mean() > 0 else [None, None]
    o['flowT_minus_rot'] = float((fT - fR).mean()); o['flowT_minus_rot_ci'] = bci(fT - fR)
    return o


def export_dsets():
    import torch, importlib.util
    from interp.jedi_core import sample
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py'); Mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Mm); DSX.mkdir(parents=True, exist_ok=True)
    x, y, _ = sample(2000, SEED_SAMPLE)
    for cell, tag in CELLS.items():
        for s in ARCHS['dsets']:
            d = SMALL / f'dsets-d64-n64{tag}' / f'seed{s}'
            if not (d / 'best.pt').exists(): log('missing', d); continue
            arch = json.loads((d / 'arch.json').read_text()); net = Mm.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
            sd = torch.load(d / 'best.pt', map_location='cpu'); net.load_state_dict(sd); net.eval()
            P = {k: v.numpy() for k, v in sd.items()}; P['_shift'] = np.asarray(arch['inputs']['shift'], np.float32); P['_scale'] = np.asarray(arch['inputs']['scale'], np.float32)
            xx = recentre(x) if cell.startswith('rc') else x; m = xx[..., 0] > 0
            xn = ((xx - P['_shift']) / P['_scale']) * m[..., None]
            with torch.no_grad(): lo = net(torch.from_numpy(xn.astype(np.float32)), torch.from_numpy(m)).double().numpy()
            P['_check_logits'] = lo; np.savez(DSX / f'dsets-d64-n64{tag}_s{s}.npz', **P)
            log('exported', d, 'recentre', arch['inputs'].get('recentre'), 'aug', arch['inputs'].get('aug'), 'acc on sample', float((lo.argmax(1) == y).mean()))


def main():
    from interp.jedi_core import sample
    x, y, _ = sample(2000, SEED_SAMPLE); x = np.ascontiguousarray(x, np.float32); idx = sample_rows(2000, SEED_SAMPLE)
    E, hyg = build_edits(x)
    R = dict(header='see interp/x9w_repr_2x2.py docstring', sample=dict(per_class=2000, seed=SEED_SAMPLE, N=len(y), N_W=int((y == W_).sum())),
             edit_hygiene=hyg, models={})
    log('edit hygiene', json.dumps(clean(hyg)))
    fo = OUT / 'x9w_repr_2x2.json'; save = lambda: fo.write_text(json.dumps(clean(R), indent=1))
    todo = [('official', 0, 'stored')] + [(a, s, c) for a in ARCHS for c in CELLS for s in seeds(a, c)]
    for a, s, c in todo:
        key = f'{a}_{c}_s{s}'
        try:
            fn, info = load_model(a, s, c)
        except Exception as e:  # noqa: BLE001
            log('SKIP', key, repr(e)[:300]); R['models'][key] = dict(error=repr(e)[:500]); continue
        rc = c.startswith('rc')
        if a == 'dsets':        # numpy forward vs the torch logits written by --export_dsets
            chk = np.load(DSX / f'dsets-d64-n64{CELLS[c]}_s{s}.npz')['_check_logits']; info['check_torch_max_abs_dlogit'] = float(np.abs(fn(recentre(x) if rc else x) - chk).max())
        if a in ('pfn', 'efn'):  # numpy forward vs the TF softmax on sample(2000, 313) written at training time
            from interp.x9_efn_pfn import softmax
            ck = np.load(Path(info['dir']) / 'check.npz'); xc, _, _ = sample(2000, 313)
            info['check_tf_max_abs_dp'] = float(np.abs(softmax(fn(recentre(xc) if rc else xc)) - ck['prob']).max())
        o = evaluate(fn, rc, x, y, E, idx, info); o.update(arch=a, cell=c, seed=s, recentred=rc, info=info); R['models'][key] = o
        e = o['edits']
        log(f'{key}: acc {o["acc"]:.4f} (test {info["acc_test"]:.4f}); W->Z unedited {o["W_to_Z_unedited"]:.3f}; ' +
            ' '.join(f'{k} {v["W_to_Z"]:.3f}/{v["calls_changed"]:.3f}' for k, v in e.items()) +
            f'; ratio T/dilM2 {o["ratio_T_over_dilM2"]}; checks ' + json.dumps({k: v for k, v in info.items() if k.startswith('check')} | ({'tp': o['check_test_probs']} if 'check_test_probs' in o else {})))
        save(); del fn; gc.collect()
        if a in ('jedi', 'official'):
            import jax, keras; jax.clear_caches(); keras.backend.clear_session()
    # ---- Holm over models (per-cell tests)
    keys = [k for k, v in R['models'].items() if 'edits' in v]
    for test, get in (('T_vs_rot', lambda v: v['edits']['T']['mcnemar_Z_vs_rot']['p']), ('M2held_vs_unedited', lambda v: v['edits']['M2held']['mcnemar_vs_unedited']['p']),
                      ('T_vs_unedited', lambda v: v['edits']['T']['mcnemar_vs_unedited']['p'])):
        ps = [get(R['models'][k]) for k in keys]; h = holm(ps)
        for k, p, hh in zip(keys, ps, h): R['models'][k].setdefault('holm', {})[test] = dict(p=p, p_holm=hh, m=len(keys))
    # ---- seed-level 2 x 2 tests
    S = dict(per_arch={}, pooled={})
    metr = {'flow_T': lambda v: v['edits']['T']['flow'], 'W_to_Z_T': lambda v: v['edits']['T']['W_to_Z'],
            'acc_test': lambda v: v['info']['acc_test'], 'flow_M2held': lambda v: v['edits']['M2held']['flow'],
            'flow_dilM2': lambda v: v['edits']['dilM2']['flow'], 'calls_T': lambda v: v['edits']['T']['calls_changed']}
    V = {}
    for a in ('pfn', 'efn', 'dsets', 'jedi'):
        S['per_arch'][a] = {}
        for mn, g in metr.items():
            v = {c: [g(R['models'][f'{a}_{c}_s{s}']) for s in seeds(a, c) if 'edits' in R['models'].get(f'{a}_{c}_s{s}', {})] for c in CELLS}
            ent = dict(values=v, cell_means={c: (float(np.mean(vv)) if vv else None) for c, vv in v.items()})
            if all(len(vv) == 2 for vv in v.values()):
                ent['perm'], ent['n_perm'] = perm_test(v); V.setdefault(mn, {})[a] = v
            if a == 'jedi' and len(v['stored']) >= 2 and len(v['aug']) >= 2:   # exact two-sample permutation, stored vs aug
                ent['stored_vs_aug'] = perm2(v['stored'], v['aug'])
            S['per_arch'][a][mn] = ent
    for mn, VV in V.items():
        if len(VV) == 3: S['pooled'][mn] = perm_test_pooled(VV)
    R['seed_level'] = S; save()
    for a, d in S['per_arch'].items():
        log(a, json.dumps(clean({mn: (e['cell_means'], {k: (round(v['effect'], 4), round(v['p'], 4)) for k, v in e.get('perm', {}).items()}) for mn, e in d.items() if mn in ('flow_T', 'acc_test')})))
    log('pooled', json.dumps(clean(S['pooled'])))
    log('wrote', fo, 'X9w DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--export_dsets', action='store_true'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    export_dsets() if a.export_dsets else main()
