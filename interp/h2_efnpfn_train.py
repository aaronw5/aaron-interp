"""H2 training: Energy Flow Network (EFN) and Particle Flow Network (PFN) (energyflow.archs; Komiske, Metodiev, Thaler, JHEP 01
(2019) 121) on Top Landscape and Quark-Gluon, from the converted arrays of interp.data_top / interp.data_qg (kin3: pT, deta,
dphi of the 128 hardest constituents).
  efn  inputs z_i = pT_i / sum_j pT_j and (deta_i, dphi_i); latent sum_i z_i Phi(deta_i, dphi_i): IRC-safe by construction
       (a collinear split leaves sum z Phi unchanged; a soft particle has z -> 0), the zero control for collinear / soft edits
  pfn  inputs (z_i, deta_i, dphi_i); latent sum_i Phi(z_i, deta_i, dphi_i): not IRC-safe (a soft particle adds Phi(0, ...))
Phi = (100, 100, 256), F = (100, 100, 100) (EFN paper, quark-gluon setting), ReLU, softmax(2), Adam 1e-3, batch 500, early
stopping on the validation loss (patience 5, best weights restored), at most --epochs epochs / --max_minutes per dataset.
Splits: top = train_file (1.21M) / val_file (403k, early stopping) / test_file (404k); qg = train files minus is_val (1.6M) /
is_val (200k, early stopping) / test files (200k). Writes /j-jepa-vol/aaron-interp/models/<ds>/<arch>/: weights.npz,
model.h5, info.json (test accuracy, AUC, rejection at 30 / 50 % signal efficiency), history.json, test_prob.npy.
Run (CPU, isolated venv):
  ./job.sh d-h2-efn 4 10 '/j-jepa-vol/aaron-interp/venv-ef/bin/python -m interp.h2_efnpfn_train --arch efn --ds top,qg'
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')


def feats(k3, arch):
    k3 = np.asarray(k3, np.float32); pt = k3[..., 0]; real = pt > 0
    z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12); ang = k3[..., 1:3] * real[..., None]
    if arch == 'efn': return [z.astype(np.float32), ang.astype(np.float32)]
    return np.concatenate([z[..., None], ang], -1).astype(np.float32)


def splits(ds):
    D = V / 'data' / ds
    if ds == 'top':
        ld = lambda s: (np.load(D / s / 'kin3.npy', mmap_mode='r'), np.load(D / s / 'label.npy').astype(int))
        (ktr, ytr), (kva, yva), (kte, yte) = ld('train'), ld('val'), ld('test')
        return (ktr, ytr), (kva, yva), (kte, yte)
    k = np.load(D / 'train' / 'kin3.npy', mmap_mode='r'); y = np.load(D / 'train' / 'label.npy').astype(int); v = np.load(D / 'train' / 'is_val.npy')
    tr, va = np.flatnonzero(~v), np.flatnonzero(v)
    return (k[tr], y[tr]), (k[va], y[va]), (np.load(D / 'test' / 'kin3.npy', mmap_mode='r'), np.load(D / 'test' / 'label.npy').astype(int))


def metrics(p, y):
    from sklearn.metrics import roc_auc_score
    s = p[:, 1]; sig = np.sort(s[y == 1]); r = {}
    for e in (0.3, 0.5):
        thr = sig[int(np.floor((1 - e) * len(sig)))]; eb = np.mean(s[y == 0] >= thr); r[f'R{int(e * 100)}'] = float(1 / eb) if eb > 0 else None
    return dict(acc=float(((s > 0.5).astype(int) == y).mean()), AUC=float(roc_auc_score(y, s)), **r)


def run(arch, ds, a):
    import tensorflow as tf
    import energyflow as ef
    from energyflow.archs import EFN, PFN
    try:
        import tf_keras as K2
    except ImportError:
        K2 = tf.keras
    tf.keras.utils.set_random_seed(a.seed); K2.utils.set_random_seed(a.seed)
    out = V / 'models' / ds / arch; out.mkdir(parents=True, exist_ok=True)
    if (out / 'info.json').exists(): log(f'{ds}/{arch}: info.json exists, skip'); return
    t_ds = time.time()
    (ktr, ytr), (kva, yva), (kte, yte) = splits(ds)
    log(f'{ds}/{arch}: train {len(ytr)} (signal frac {ytr.mean():.3f}), val {len(yva)}, test {len(yte)}; energyflow {ef.__version__}, tf {tf.__version__}')
    Xtr, Xva = feats(ktr, arch), feats(kva, arch); Y = np.eye(2, dtype=np.float32)
    phi = (100, 100, 256); Fs = (100, 100, 100)
    m = (EFN(input_dim=2, Phi_sizes=phi, F_sizes=Fs, output_dim=2, summary=False) if arch == 'efn'
         else PFN(input_dim=3, Phi_sizes=phi, F_sizes=Fs, output_dim=2, summary=False))

    class Clock(K2.callbacks.Callback):
        def on_epoch_end(self, ep, logs=None):
            log(f'{ds}/{arch} epoch {ep + 1}: ' + ', '.join(f'{k} {v:.4f}' for k, v in (logs or {}).items()))
            if time.time() - t_ds > 60 * a.max_minutes: log('wall-time limit reached: stopping'); self.model.stop_training = True
    es = K2.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    h = m.fit(Xtr, Y[ytr], epochs=a.epochs, batch_size=500, validation_data=(Xva, Y[yva]), verbose=0, callbacks=[es, Clock()])
    del Xtr
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    pte = m.predict(feats(kte, arch), batch_size=5000, verbose=0)
    np.save(out / 'test_prob.npy', pte.astype(np.float32))
    W = {}
    for i, L in enumerate(m.model.layers):
        for j, w in enumerate(L.get_weights()): W[f'{i:02d}_{L.name}_{j}'] = w
    np.savez(out / 'weights.npz', **W)
    try: m.model.save(out / 'model.h5')
    except Exception as e: log(f'model.h5 not saved: {e}')
    info = dict(arch=arch, ds=ds, phi=phi, F=Fs, seed=a.seed, epochs_run=len(hist['loss']), best_epoch=int(np.argmin(hist['val_loss'])) + 1,
                seconds=time.time() - t_ds, n_train=int(len(ytr)), n_val=int(len(yva)), n_test=int(len(yte)), test=metrics(pte, yte),
                val_acc_best=float(max(hist.get('val_acc', [np.nan]))))
    (out / 'info.json').write_text(json.dumps(info, indent=1)); (out / 'history.json').write_text(json.dumps(hist, indent=1))
    log(f'{ds}/{arch}: test {info["test"]}; wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['efn', 'pfn'], required=True); ap.add_argument('--ds', default='top,qg')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--max_minutes', type=float, default=330.0); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    import tensorflow as tf
    n = int(os.environ.get('OMP_NUM_THREADS', '1')); tf.config.threading.set_intra_op_parallelism_threads(n); tf.config.threading.set_inter_op_parallelism_threads(2)
    for ds in a.ds.split(','): run(a.arch, ds, a)
    log('done')
