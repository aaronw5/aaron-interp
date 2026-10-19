"""X5: Deep Sets dsets-d64 (same widths as train/models.py DeepSetsTagger) trained with HGQ2 fixed-point quantization-aware
training (keras 3 / jax CPU, hgq2 0.1.9), on the same inputs, split and test set as train/train_small.py.

    python train/train_hgq_dsets.py --seed 0 --beta 1e-7 --out /j-jepa-vol/aaron-interp/models/small/dsets-d64-n64-hgq_1e-7/seed0

Architecture (per particle, then sum, then MLP; JEDI-linear's HGQ settings from JEDI-linear src/model.py get_model / get_gnn):
    x (B, 64, 3) normalised (pT, deta, dphi) with norm_stats.json['64'], padded rows (raw pT == 0) set to 0; mask (B, 64, 1)
    phi1 = QEinsumDenseBatchnorm('bnc,cC->bnC', (64, 64), relu)    input quantizer per (slot, feature)
    phi2 = QEinsumDenseBatchnorm('bnc,cC->bnC', (64, 64), relu)    input quantizer per (slot, channel)
    S    = QSum(axes=1, scale=1)(phi2 * mask)                       input quantizer per (slot, channel)  -> zero-bit slot masks possible
    h1, h2, h3 = QEinsumDenseBatchnorm 64, 32, 16 (relu); logits = QEinsumDenseBatchnorm 5
  quantizers: weights 'kbi' b0 = 7, i0 = 0, wrap (weights/bias SAT_SYM), MonoL1(1e-8) on bits; datalane 'kif' f0 = 7, wrap,
  integer bits constrained to [0, 12], MonoL1(1e-8); EBOPs penalty beta (constant over training, --beta).
Differences from the float run (train_small.py): Adam (no weight decay: AdamW would also decay the bit-width variables), batch
norm folded into the quantized dense layers (as in JEDI-linear); lr 1e-3, 200 warm-up steps then cosine to 0, batch 1024,
--epochs (30), early stopping on val loss with patience 5 and the best weights restored.
Writes to --out: model.keras, log.csv (per epoch: losses, accuracy, EBOPs, time), arch.json, test.json (260,000-jet test archive:
accuracy, per-class accuracy, confusion, phi1 zero fraction and mean kernel bit-width), test_probs.npy.
"""
import argparse, json, os, sys, time
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
CLASSES = ['g', 'q', 'W', 'Z', 't']
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:8.1f}s]', *a, flush=True)


def load(data_dir, prefix, n, rows=None):
    import h5py
    x = np.load(os.path.join(data_dir, f'{prefix}_ptetaphi_f16.npy'), mmap_mode='r')
    with h5py.File(os.path.join(data_dir, f'{prefix}_meta.h5'), 'r') as f:
        y = f['label'][:].astype(np.int64)
    if rows is None:
        return np.ascontiguousarray(x[:, :n, :]), y
    return np.ascontiguousarray(x[rows, :n, :]), y[rows]


def normalise(raw, shift, scale):
    raw = np.asarray(raw, np.float32); m = raw[..., 0] > 0
    x = (raw - np.float32(shift)) / np.float32(scale); x = x * m[..., None]
    return x.astype(np.float32), m[..., None].astype(np.float32)


def build(n, beta, d=64, rho=(32, 16)):
    import keras
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    from hgq.constraints import MinMax
    from hgq.layers import QEinsumDenseBatchnorm, QSum
    from hgq.regularizers import MonoL1
    s0 = QuantizerConfigScope(default_q_type='kbi', b0=7, overflow_mode='wrap', i0=0, fr=MonoL1(1e-8), ir=MonoL1(1e-8))
    s1 = QuantizerConfigScope(default_q_type='kif', place='datalane', overflow_mode='wrap', f0=7, fr=MonoL1(1e-8), ic=MinMax(0, 12))
    s2 = LayerConfigScope(beta0=beta)
    s3 = QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM')
    s4 = QuantizerConfigScope(place='datalane', heterogeneous_axis=None)
    with s0, s1, s2, s3, s4:
        inp = keras.layers.Input((n, 3), name='x'); msk = keras.layers.Input((n, 1), name='mask')
        a = QEinsumDenseBatchnorm('bnc,cC->bnC', (n, d), bias_axes='C', activation='relu', name='phi1')(inp)
        u = QEinsumDenseBatchnorm('bnc,cC->bnC', (n, d), bias_axes='C', activation='relu', name='phi2')(a)
        u = keras.layers.Multiply(name='masked')([u, msk])
        h = QSum(axes=1, scale=1.0, keepdims=False, name='pool')(u)
        for i, w in enumerate((d,) + tuple(rho)):
            h = QEinsumDenseBatchnorm('bc,cC->bC', w, bias_axes='C', activation='relu', name=f'h{i + 1}')(h)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C', name='head')(h)
    return keras.Model([inp, msk], out)


def qbits(q):
    for a in ('b', 'bits'):
        if hasattr(q, a): return np.asarray(getattr(q, a), np.float64)
    k, i, f = (np.asarray(getattr(q, a), np.float64) for a in ('k', 'i', 'f'))
    return np.maximum(k + i + f, 0)


def phi1_stats(model):
    L = model.get_layer('phi1')
    K, b = L.get_fused_qkernel_and_qbias(False, L.moving_mean, L.moving_variance)
    K = np.asarray(K, np.float64)
    out = dict(phi1_zero_frac=float((K == 0).mean()), phi1_nonzero_per_feature=(K != 0).sum(1).tolist())
    try:
        kb = qbits(L.kq.quantizer); out['phi1_kernel_mean_bits'] = float(np.broadcast_to(kb, K.shape).mean())
    except Exception as e:  # noqa: BLE001
        out['phi1_kernel_mean_bits'] = None; out['bits_error'] = repr(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--beta', type=float, required=True)
    ap.add_argument('--n', type=int, default=64); ap.add_argument('--out', required=True)
    ap.add_argument('--data_dir', default=os.environ.get('JETDISTILL_DATA', '/j-jepa-vol/aaron-interp/data'))
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch', type=int, default=1024); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup_steps', type=int, default=200)
    ap.add_argument('--max_train', type=int, default=0); ap.add_argument('--max_test', type=int, default=0)
    a = ap.parse_args()
    import keras, hgq
    from hgq.utils.sugar import FreeEBOPs
    os.makedirs(a.out, exist_ok=True)
    keras.utils.set_random_seed(a.seed)
    log('args', vars(a), 'keras', keras.__version__, 'hgq', hgq.__version__)
    ns = json.load(open(os.path.join(a.data_dir, 'norm_stats.json')))[str(a.n)]
    shift, scale = [float(v) for v in ns['shift']], [float(v) for v in ns['scale']]
    ntot = len(np.load(os.path.join(a.data_dir, 'train_ptetaphi_f16.npy'), mmap_mode='r'))
    dev = np.sort(np.asarray(np.load(os.path.join(a.data_dir, 'splits.npz'))['dev'], np.int64))
    isdev = np.zeros(ntot, bool); isdev[dev] = True; tr = np.flatnonzero(~isdev)
    if a.max_train: tr = np.sort(np.random.RandomState(a.seed).choice(tr, a.max_train, replace=False))
    xr, ytr = load(a.data_dir, 'train', a.n, tr); Xtr, Mtr = normalise(xr, shift, scale); del xr
    xr, yva = load(a.data_dir, 'train', a.n, dev); Xva, Mva = normalise(xr, shift, scale); del xr
    log(f'data: train {len(ytr)} val {len(yva)} (splits.npz dev); shift {shift} scale {scale}')
    model = build(a.n, a.beta)
    steps = int(np.ceil(len(ytr) / a.batch)) * a.epochs
    sched = keras.optimizers.schedules.CosineDecay(0.0, decay_steps=max(1, steps - a.warmup_steps), alpha=0.0,
                                                   warmup_target=a.lr, warmup_steps=a.warmup_steps)
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary(print_fn=lambda s, **k: print(s, flush=True))
    rows = []

    class Rec(keras.callbacks.Callback):
        def on_epoch_begin(self, ep, logs=None): self.t = time.time()

        def on_epoch_end(self, ep, logs=None):
            r = dict(epoch=ep + 1, time_s=time.time() - self.t, **{k: float(v) for k, v in (logs or {}).items()})
            r.update(phi1_stats(self.model)); rows.append(r)
            log('epoch', json.dumps(r))
            with open(os.path.join(a.out, 'log.json'), 'w') as f: json.dump(rows, f, indent=1)

    cb = [FreeEBOPs(), Rec(), keras.callbacks.EarlyStopping(monitor='val_loss', patience=a.patience, restore_best_weights=True),
          keras.callbacks.TerminateOnNaN()]
    model.fit([Xtr, Mtr], ytr, batch_size=a.batch, epochs=a.epochs, validation_data=([Xva, Mva], yva), callbacks=cb,
              shuffle=True, verbose=2)
    del Xtr, Mtr
    model.save(os.path.join(a.out, 'model.keras'))
    te = None
    if a.max_test:
        nte = len(np.load(os.path.join(a.data_dir, 'val_ptetaphi_f16.npy'), mmap_mode='r'))
        te = np.sort(np.random.RandomState(a.seed + 1).choice(nte, a.max_test, replace=False))
    xr, yte = load(a.data_dir, 'val', a.n, te); Xte, Mte = normalise(xr, shift, scale); del xr
    lo = model.predict([Xte, Mte], batch_size=4096, verbose=0)
    p = np.exp(lo - lo.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); pred = p.argmax(1)
    conf = np.zeros((5, 5), np.int64); np.add.at(conf, (yte, pred), 1)
    best = min(rows, key=lambda r: r.get('val_loss', np.inf)) if rows else {}
    rep = dict(n=int(len(yte)), accuracy=float((pred == yte).mean()), per_class_accuracy={CLASSES[c]: float(conf[c, c] / max(conf[c].sum(), 1)) for c in range(5)},
               confusion=conf.tolist(), classes=CLASSES, seed=a.seed, beta=a.beta, epochs_run=len(rows), best_epoch=best.get('epoch'),
               best_val=best, run_name=f'dsets-d64-n{a.n}-hgq', max_test=a.max_test, wall_s=time.time() - T0, **phi1_stats(model))
    json.dump(rep, open(os.path.join(a.out, 'test.json'), 'w'), indent=1)
    np.save(os.path.join(a.out, 'test_probs.npy'), p.astype(np.float32))
    json.dump(dict(model='dsets-d64 HGQ', n=a.n, shift=shift, scale=scale, args=vars(a), doc=__doc__), open(os.path.join(a.out, 'arch.json'), 'w'), indent=1)
    log('test', json.dumps({k: rep[k] for k in ('accuracy', 'phi1_zero_frac', 'phi1_kernel_mean_bits', 'epochs_run', 'best_epoch')}))


if __name__ == '__main__':
    main()
