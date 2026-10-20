"""X12: JEDI-linear 3-feature n64 retrained from scratch (never touches the official checkpoints), HGQ2 fixed-point
quantization-aware training as in the official JEDI-linear code (github.com/calad0i/JEDI-linear src/model.py get_model / get_gnn,
src/train.py train_hgq), and a pipeline-matched float arm (same script, same optimizer, BN folding, lr schedule, init, batch and
epochs; every quantizer replaced by hgq's identity DummyQuantizer, EBOPs off).

    python train/train_hgq_jedi.py --arm hgq   --seed 0 --epochs 200 --out /j-jepa-vol/aaron-interp/models/small/jedi-n64-x12-hgq/seed0
    python train/train_hgq_jedi.py --arm float --seed 0 --epochs 200 --out /j-jepa-vol/aaron-interp/models/small/jedi-n64-x12-float/seed0

Architecture (official get_gnn, conf.model_class = gnn, N = 64, 3 features; layer names here are the stage letters of
interp/jedi_core.py so a generic forward can read them):
    l1  = QEinsumDenseBatchnorm('bnc,cC->bnC', (64, 64), relu)               a
    l2  = QEinsumDenseBatchnorm('bnc,cC->bnC', (64, 64), relu)(a)             s
    qs  = QSum(axes=1, scale=2^-round(log2 64), keepdims=True)(a)             P
    dd  = QEinsumDenseBatchnorm('bnc,cC->bnC', (1, 64), relu)(P)              g
    add = QAdd()([s, g]);  l3 = QEinsumDenseBatchnorm (64, 64) relu           u
    qs1 = QSum(axes=1, scale=1/16)                                            S
    d4 64, d5 32, d6 16 (relu), d7 5 (logits)
  quantizer scopes exactly as get_model: weights kbi b0 = 7, i0 = 0, wrap, MonoL1(1e-8) on f and i bits; weights / bias SAT_SYM;
  datalane kif f0 = 7, wrap, MonoL1(1e-8) on f, integer bits MinMax(0, 12), heterogeneous per element (per slot x channel);
  LayerConfigScope(beta0=0) with the beta set per epoch by a callback.
Inputs exactly as the official dataloader / interp.jedi_core: (x - shift) / scale with data/norm_stats.json['64'] for every slot,
  padding included (no mask), stored as float16 (the official train_hgq casts X to float16).
Training (official sweep-n64-f3.yaml except the epoch count and the beta schedule): Adam, batch 2790 (drop_last), lr per epoch by
  the official cosine decay with restarts (3e-3, period 100 epochs, alpha 1e-6, 10 epochs at the minimum), jax matmul precision
  tensorfloat32, no early stopping. Split: train archive minus splits.npz['dev'] (595,000) / dev (25,000) for validation; test =
  val archive (260,000).
Beta (HGQ arm; the official schedule ramps 2e-8 -> 3e-7 over 2,000 epochs and its checkpoint is at epoch 2,868 with l1 zero
  fraction 1 - 20/192 = 0.896): --beta_mode control (default) sets beta at each epoch start by a multiplicative controller on the
  fused-l1-kernel zero fraction z: target z*(e) = 0.896 * clip((e - 5) / (ramp_end - 5), 0, 1); beta *= 1.3 if z < z* - 0.02,
  beta /= 1.3 if z > z* + 0.02; start 2e-8, clipped to [1e-9, --beta_max]. --beta_mode official: the official piecewise schedule
  compressed in time by --epochs / 3000. Float arm: beta 0, EBOPs off.
X9w opt-in flags (defaults reproduce X12 exactly):
  --recentre  raw (deta, dphi) of the real particles (pT > 0) re-measured about the pT-weighted centroid of the first 64 slots
              before normalisation, for train, dev and test jets (padding stays 0 raw, i.e. -shift/scale normalised as before).
  --aug A     (A > 0) training jets only: every real particle of a jet translated by the same c, |c| ~ U(0, A) rad, direction
              ~ U(0, 2 pi), drawn per jet and redrawn every epoch (rng seed + 1000), after recentring when both are set; the
              normalised training array is rebuilt each epoch and model.fit runs one epoch per call (initial_epoch = e, so the
              lr / beta callbacks see the same epoch index).
Writes to --out: ckpt/epoch_<e>.keras (every --save_every epochs, plus --keep_epochs and the last), model.keras (last epoch),
  log.json (per epoch: losses, accuracies, lr, beta, EBOPs, l1 zero fraction, qs1 zero-bit slot fraction, seconds),
  test.json / test_probs.npy (last epoch, 260,000 test jets), test_e<k>.json / test_probs_e<k>.npy for each --keep_epochs k,
  arch.json.
"""
import argparse, json, os, sys, time
from math import cos, log2, pi
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
CLASSES = ['g', 'q', 'W', 'Z', 't']
L1_TARGET = 1 - 20 / 192
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:8.1f}s]', *a, flush=True)


def load(data_dir, prefix, rows=None, n=64, chunk=50000):
    """sequential reads in row blocks of the memory map (random-row reads from CephFS took ~300 s for 60,000 rows; reading
    the whole 150-slot file at once in 5 concurrent processes exceeded the container memory)"""
    import h5py
    xm = np.load(os.path.join(data_dir, f'{prefix}_ptetaphi_f16.npy'), mmap_mode='r')
    with h5py.File(os.path.join(data_dir, f'{prefix}_meta.h5'), 'r') as f:
        y = f['label'][:].astype(np.int32)
    rows = np.arange(len(xm)) if rows is None else np.asarray(rows)
    out = np.empty((len(rows), n, 3), np.float16)
    for s in range(0, len(xm), chunk):
        sel = (rows >= s) & (rows < s + chunk)
        if sel.any(): out[sel] = np.asarray(xm[s:s + chunk, :n, :])[rows[sel] - s]
    return out, y[rows]


def recentre16(raw, chunk=50000):
    """float16 raw (M, n, 3) -> float16 with the real particles' angles about the pT-weighted centroid (X9w)"""
    out = np.empty(raw.shape, np.float16)
    for s in range(0, len(raw), chunk):
        x = np.asarray(raw[s:s + chunk], np.float32); real = x[..., 0] > 0; pt = np.where(real, x[..., 0], 0).astype(np.float64)
        c = (pt[..., None] * x[..., 1:3]).sum(1) / np.maximum(pt.sum(1), 1e-12)[:, None]
        x[..., 1:3] = np.where(real[..., None], x[..., 1:3] - c[:, None, :].astype(np.float32), x[..., 1:3]); out[s:s + chunk] = x.astype(np.float16)
    return out


def augment_normalise(raw, shift, scale, rng, amax, chunk=50000):
    """per-jet rigid translation of the real particles (|c| ~ U(0, amax), uniform direction), then (x - shift) / scale (X9w)"""
    out = np.empty(raw.shape, np.float16); sh, sc = np.float32(shift), np.float32(scale)
    r = rng.uniform(0, amax, len(raw)); th = rng.uniform(0, 2 * np.pi, len(raw))
    for s in range(0, len(raw), chunk):
        x = np.asarray(raw[s:s + chunk], np.float32); real = x[..., 0] > 0; e = slice(s, s + chunk)
        x[..., 1] += np.where(real, (r[e] * np.cos(th[e]))[:, None], 0).astype(np.float32)
        x[..., 2] += np.where(real, (r[e] * np.sin(th[e]))[:, None], 0).astype(np.float32)
        out[s:s + chunk] = ((x - sh) / sc).astype(np.float16)
    return out


def normalise(raw, shift, scale, chunk=50000):
    out = np.empty(raw.shape, np.float16); sh, sc = np.float32(shift), np.float32(scale)
    for s in range(0, len(raw), chunk):
        out[s:s + chunk] = ((np.asarray(raw[s:s + chunk], np.float32) - sh) / sc).astype(np.float16)
    return out


def build(arm, n=64):
    import keras
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    from hgq.constraints import MinMax
    from hgq.layers import QAdd, QEinsumDenseBatchnorm, QSum
    from hgq.regularizers import MonoL1
    scopes = [QuantizerConfigScope(default_q_type='kbi', b0=7, overflow_mode='wrap', i0=0, fr=MonoL1(1e-8), ir=MonoL1(1e-8)),
              QuantizerConfigScope(default_q_type='kif', place='datalane', overflow_mode='wrap', f0=7, fr=MonoL1(1e-8), ic=MinMax(0, 12)),
              LayerConfigScope(beta0=0) if arm == 'hgq' else LayerConfigScope(beta0=0, enable_ebops=False),
              QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM'),
              QuantizerConfigScope(place='datalane', heterogeneous_axis=None)]
    if arm == 'float':
        scopes.append(QuantizerConfigScope(default_q_type='dummy'))
    for s in scopes: s.__enter__()
    try:
        inp = keras.layers.Input((n, 3), name='x')
        a = QEinsumDenseBatchnorm('bnc,cC->bnC', (n, 64), bias_axes='C', activation='relu', name='l1')(inp)
        s = QEinsumDenseBatchnorm('bnc,cC->bnC', (n, 64), bias_axes='C', activation='relu', name='l2')(a)
        d = QEinsumDenseBatchnorm('bnc,cC->bnC', (1, 64), bias_axes='C', activation='relu', name='dd')(
            QSum(axes=1, scale=2.0 ** -round(log2(n)), keepdims=True, name='qs')(a))
        x = QAdd(name='add')([s, d])
        x = QEinsumDenseBatchnorm('bnc,cC->bnC', (n, 64), bias_axes='C', activation='relu', name='l3')(x)
        x = QSum(axes=1, scale=1 / 16, keepdims=False, name='qs1')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 64, bias_axes='C', activation='relu', name='d4')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 32, bias_axes='C', activation='relu', name='d5')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu', name='d6')(x)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C', name='d7')(x)
    finally:
        for s in reversed(scopes): s.__exit__(None, None, None)
    return keras.Model(inp, out)


def qbits(q):
    for a in ('b', 'bits'):
        if hasattr(q, a): return np.asarray(getattr(q, a), np.float64)
    k, i, f = (np.asarray(getattr(q, a), np.float64) for a in ('k', 'i', 'f'))
    return np.maximum(k + i + f, 0)


def structure_stats(model):
    L = model.get_layer('l1')
    K, _ = L.get_fused_qkernel_and_qbias(False, L.moving_mean, L.moving_variance)
    K = np.asarray(K, np.float64); out = dict(l1_zero_frac=float((K == 0).mean()), l1_nonzero=int((K != 0).sum()))
    try:
        iq = model.get_layer('qs1').iq; q = (iq.quantizers[0] if hasattr(iq, 'quantizers') else iq).quantizer
        if not getattr(q, '__dummy__', False):
            out['qs1_zero_bit_frac'] = float((qbits(q) <= 0).mean())
    except Exception as e:  # noqa: BLE001
        out['qs1_error'] = repr(e)
    return out


def lr_official(epoch, first=100, lr0=3e-3, alpha=1e-6, alpha_steps=10):
    cyc = epoch % first; t = min(cyc / (first - alpha_steps), 1)
    return alpha + 0.5 * (lr0 - alpha) * (1 + cos(pi * t))


def evaluate(model, Xte, yte, batch=8192):
    lo = model.predict(Xte, batch_size=batch, verbose=0)
    p = np.exp(lo - lo.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); pred = p.argmax(1)
    conf = np.zeros((5, 5), np.int64); np.add.at(conf, (yte, pred), 1)
    return p, dict(n=int(len(yte)), accuracy=float((pred == yte).mean()), confusion=conf.tolist(), classes=CLASSES,
                   per_class_accuracy={CLASSES[c]: float(conf[c, c] / max(conf[c].sum(), 1)) for c in range(5)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['hgq', 'float'], required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data_dir', default=os.environ.get('JETDISTILL_DATA', '/j-jepa-vol/aaron-interp/data'))
    ap.add_argument('--epochs', type=int, default=200); ap.add_argument('--batch', type=int, default=2790)
    ap.add_argument('--beta_mode', choices=['control', 'official'], default='control')
    ap.add_argument('--ramp_end', type=int, default=50); ap.add_argument('--beta_max', type=float, default=1e-4)
    ap.add_argument('--save_every', type=int, default=20); ap.add_argument('--keep_epochs', default='60')
    ap.add_argument('--max_train', type=int, default=0); ap.add_argument('--max_test', type=int, default=0)
    ap.add_argument('--device', default='gpu:0')
    ap.add_argument('--recentre', action='store_true', help='X9w: angles about the pT-weighted centroid (all jets)')
    ap.add_argument('--aug', type=float, default=0.0, help='X9w: max rigid translation (rad) of the training jets; 0 = off')
    ap.add_argument('--aug_device', default='gpu:0', help='X9w: device of the per-epoch augmented training array')
    a = ap.parse_args()
    import jax, keras, hgq
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
    from hgq.utils.sugar import FreeEBOPs, Dataset, PieceWiseSchedule
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=True)
    keras.utils.set_random_seed(a.seed)
    log('args', vars(a), 'keras', keras.__version__, 'hgq', hgq.__version__, 'jax', jax.__version__, 'devices', jax.devices())
    ns = json.load(open(os.path.join(a.data_dir, 'norm_stats.json')))['64']
    shift, scale = [float(v) for v in ns['shift']], [float(v) for v in ns['scale']]
    ntot = len(np.load(os.path.join(a.data_dir, 'train_ptetaphi_f16.npy'), mmap_mode='r'))
    dev = np.sort(np.asarray(np.load(os.path.join(a.data_dir, 'splits.npz'))['dev'], np.int64))
    isdev = np.zeros(ntot, bool); isdev[dev] = True; tr = np.flatnonzero(~isdev)
    if a.max_train: tr = np.sort(np.random.RandomState(a.seed).choice(tr, a.max_train, replace=False))
    prep = recentre16 if a.recentre else (lambda v: v)
    xr, ytr = load(a.data_dir, 'train', tr); xr = prep(xr)
    if a.aug > 0: xtr_raw = xr; Xtr = None
    else: Xtr = normalise(xr, shift, scale); del xr
    xr, yva = load(a.data_dir, 'train', dev); Xva = normalise(prep(xr), shift, scale); del xr
    perm = np.random.RandomState(a.seed).permutation(len(ytr)); ytr = ytr[perm]
    if a.aug > 0: xtr_raw = xtr_raw[perm]
    else: Xtr = Xtr[perm]
    log(f'data: train {len(ytr)} val {len(yva)}; shift {shift} scale {scale}')
    model = build(a.arm)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary(print_fn=lambda s, **k: print(s, flush=True))
    keep = sorted({int(v) for v in a.keep_epochs.split(',') if v})
    official = PieceWiseSchedule([(0, 2e-8, 'linear'), (int(2000 * a.epochs / 3000), 3e-7, 'log'), (int(7000 * a.epochs / 3000), 3e-6, 'constant')])
    rows = []; state = dict(beta=2e-8 if a.arm == 'hgq' else 0.0)
    if a.max_test:
        nte = len(np.load(os.path.join(a.data_dir, 'val_ptetaphi_f16.npy'), mmap_mode='r'))
        te = np.sort(np.random.RandomState(a.seed + 1).choice(nte, a.max_test, replace=False))
    else:
        te = None
    xr, yte = load(a.data_dir, 'val', te); Xte = normalise(prep(xr), shift, scale).astype(np.float32); del xr

    def set_beta(m, beta):
        for layer in m._flatten_layers():
            if getattr(layer, '_beta', None) is not None:
                layer._beta.assign(keras.ops.convert_to_tensor(beta, dtype=layer._beta.dtype))

    class Ctrl(keras.callbacks.Callback):
        def on_epoch_begin(self, ep, logs=None):
            self.t = time.time()
            self.model.optimizer.learning_rate = lr_official(ep)
            if a.arm == 'hgq':
                if a.beta_mode == 'official':
                    state['beta'] = float(official(ep))
                elif rows:
                    z = rows[-1]['l1_zero_frac']; tgt = L1_TARGET * float(np.clip((ep - 5) / max(a.ramp_end - 5, 1), 0, 1))
                    if z < tgt - 0.02: state['beta'] = min(state['beta'] * 1.3, a.beta_max)
                    elif z > tgt + 0.02: state['beta'] = max(state['beta'] / 1.3, 1e-9)
                set_beta(self.model, state['beta'])

        def on_epoch_end(self, ep, logs=None):
            r = dict(epoch=ep + 1, time_s=time.time() - self.t, lr=lr_official(ep), beta=state['beta'],
                     **{k: float(v) for k, v in (logs or {}).items() if np.ndim(v) == 0})
            r.update(structure_stats(self.model)); rows.append(r)
            log('epoch', json.dumps(r))
            with open(os.path.join(a.out, 'log.json'), 'w') as f: json.dump(rows, f, indent=1)
            e = ep + 1
            if e % a.save_every == 0 or e in keep or e == a.epochs:
                self.model.save(os.path.join(a.out, 'ckpt', f'epoch_{e:04d}.keras'))
            if e in keep:
                p, rep = evaluate(self.model, Xte, yte); rep.update(epoch=e, **structure_stats(self.model))
                json.dump(rep, open(os.path.join(a.out, f'test_e{e}.json'), 'w'), indent=1)
                np.save(os.path.join(a.out, f'test_probs_e{e}.npy'), p.astype(np.float32))
                log(f'test at epoch {e}: accuracy {rep["accuracy"]:.4f}')

    dva = Dataset(Xva, yva, batch_size=a.batch, drop_last=False, device=a.device)
    cb = [Ctrl()] + ([FreeEBOPs()] if a.arm == 'hgq' else []) + [keras.callbacks.TerminateOnNaN()]
    if a.aug > 0:   # X9w: fresh augmentation each epoch, one epoch per fit call
        # the augmented array is rebuilt every epoch on --aug_device (default gpu:0); with three runs sharing one GPU the fresh
        # 464 MB device copy per epoch ran out of GPU memory at epoch 52 (job train-x9w, 2026-10-20), so the old copy is freed
        # explicitly (del + gc) and aug runs share a GPU with at most one other run; cpu:0 made jax run the step on the CPU
        # (46 s/epoch, GPU idle; job train-x9w-b)
        import gc
        rng = np.random.default_rng(a.seed + 1000)
        for e in range(a.epochs):
            dtr = Dataset(augment_normalise(xtr_raw, shift, scale, rng, a.aug), ytr, batch_size=a.batch, drop_last=True, device=a.aug_device)
            model.fit(dtr, epochs=e + 1, initial_epoch=e, validation_data=dva, callbacks=cb, verbose=0); del dtr; gc.collect()
            if model.stop_training: log('stop_training set (NaN?)'); break
    else:
        dtr = Dataset(Xtr, ytr, batch_size=a.batch, drop_last=True, device=a.device)
        model.fit(dtr, epochs=a.epochs, validation_data=dva, callbacks=cb, verbose=0)
    model.save(os.path.join(a.out, 'model.keras'))
    p, rep = evaluate(model, Xte, yte)
    rep.update(recentre=bool(a.recentre), aug=a.aug, seed=a.seed, arm=a.arm, epochs_run=len(rows), max_test=a.max_test, wall_s=time.time() - T0, last=rows[-1] if rows else None,
               run_name=f'jedi-n64-x12-{a.arm}', **structure_stats(model))
    json.dump(rep, open(os.path.join(a.out, 'test.json'), 'w'), indent=1)
    np.save(os.path.join(a.out, 'test_probs.npy'), p.astype(np.float32))
    json.dump(dict(model='JEDI-linear gnn n64 3-feature (official get_gnn)', arm=a.arm, n=64, shift=shift, scale=scale, args=vars(a),
                   doc=__doc__), open(os.path.join(a.out, 'arch.json'), 'w'), indent=1)
    log('test', json.dumps({k: rep[k] for k in ('accuracy', 'l1_zero_frac', 'epochs_run')}))


if __name__ == '__main__':
    main()
