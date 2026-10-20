"""H39 training arm (loop D, claim D4m): EFN / PFN variants for architecture-matched nulls of the axis-direction test.
Same architecture, inputs, optimiser and data splits as interp/h2_efnpfn_train.py (energyflow EFN / PFN, Phi = (100, 100, 256),
F = (100, 100, 100), Adam 1e-3, batch 500, kin3 of the 128 hardest constituents).
  --variant s1     second training seed (--seed 1): early stopping on the validation loss (patience 5, best weights restored),
                   as the H2 seed-0 model. -> models/<ds>/<arch>_s1/
  --variant shuf   shuffled labels: train and validation labels permuted (rng seed 3900 + seed; class fractions kept); trained
                   for exactly the number of epochs of the seed-0 model's best epoch (H2 info.json best_epoch), no early
                   stopping, final weights kept (the same number of steps as the seed-0 model). -> models/<ds>/<arch>_shuf/
  --variant init   epoch-0 networks: the energyflow model built with the seed and saved untrained (Keras default kernel
                   initialiser of energyflow, he_uniform; zero biases). -> models/h39_init/<arch>_init<seed>/ (no data read)
Outputs per run: weights.npz (same keys as H2), info.json (test metrics on the true test labels; for shuf also the std of the
test logit (non-degeneracy) and the training accuracy against the shuffled labels), history.json, test_prob.npy.
Run (CPU, venv-ef): see interp/h39_ef_jobs.sh
"""
import argparse
import json
import os
import time
import numpy as np
from interp import h2_efnpfn_train as H2

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = H2.V


def build(arch, seed):
    import tensorflow as tf
    from energyflow.archs import EFN, PFN
    try:
        import tf_keras as K2
    except ImportError:
        K2 = tf.keras
    tf.keras.utils.set_random_seed(seed); K2.utils.set_random_seed(seed)
    phi = (100, 100, 256); Fs = (100, 100, 100)
    m = (EFN(input_dim=2, Phi_sizes=phi, F_sizes=Fs, output_dim=2, summary=False) if arch == 'efn'
         else PFN(input_dim=3, Phi_sizes=phi, F_sizes=Fs, output_dim=2, summary=False))
    return m, K2


def save_weights(m, out):
    W = {}
    for i, L in enumerate(m.model.layers):
        for j, w in enumerate(L.get_weights()): W[f'{i:02d}_{L.name}_{j}'] = w
    np.savez(out / 'weights.npz', **W)
    return W


def run(a):
    arch, ds, var, seed = a.arch, a.ds, a.variant, a.seed
    out = V / 'models' / 'h39_init' / f'{arch}_init{seed}' if var == 'init' else V / 'models' / ds / f'{arch}_{var}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'info.json').exists(): log(f'{out}: info.json exists, skip'); return
    m, K2 = build(arch, seed)
    if var == 'init':
        W = save_weights(m, out)
        (out / 'info.json').write_text(json.dumps(dict(arch=arch, variant='init', seed=seed, n_tensors=len(W),
                                                       note='energyflow model built with this seed, not trained'), indent=1))
        log(f'init {arch} seed {seed}: {len(W)} tensors -> {out}'); return
    t_ds = time.time()
    (ktr, ytr), (kva, yva), (kte, yte) = H2.splits(ds)
    s0 = json.loads((V / 'models' / ds / arch / 'info.json').read_text())
    if var == 'shuf':
        rng = np.random.default_rng(3900 + seed); ytr_t, yva_t = rng.permutation(ytr), rng.permutation(yva)
        epochs, cbs_es = int(s0['best_epoch']), []
        log(f'{ds}/{arch} shuf: labels permuted (train agreement with true labels {np.mean(ytr_t == ytr):.4f}); {epochs} epochs '
            f'(= seed-0 best epoch), no early stopping')
    else:
        ytr_t, yva_t = ytr, yva; epochs = a.epochs
        cbs_es = [K2.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    log(f'{ds}/{arch} {var} seed {seed}: train {len(ytr)}, val {len(yva)}, test {len(yte)}')
    Xtr, Xva = H2.feats(ktr, arch), H2.feats(kva, arch); Y = np.eye(2, dtype=np.float32)

    class Clock(K2.callbacks.Callback):
        def on_epoch_end(self, ep, logs=None):
            log(f'{ds}/{arch}/{var} epoch {ep + 1}: ' + ', '.join(f'{k} {v:.4f}' for k, v in (logs or {}).items()))
            if time.time() - t_ds > 60 * a.max_minutes: log('wall-time limit reached: stopping'); self.model.stop_training = True
    h = m.fit(Xtr, Y[ytr_t], epochs=epochs, batch_size=500, validation_data=(Xva, Y[yva_t]), verbose=0, callbacks=cbs_es + [Clock()])
    extra = {}
    if var == 'shuf':
        ptr = m.predict(Xtr[:200000] if not isinstance(Xtr, list) else [x[:200000] for x in Xtr], batch_size=5000, verbose=0)
        extra['train_acc_vs_shuffled_labels_first200k'] = float(np.mean(ptr.argmax(1) == ytr_t[:200000]))
    del Xtr
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    pte = m.predict(H2.feats(kte, arch), batch_size=5000, verbose=0)
    lg = np.log(np.clip(pte[:, 1], 1e-12, 1)) - np.log(np.clip(pte[:, 0], 1e-12, 1))
    np.save(out / 'test_prob.npy', pte.astype(np.float32)); save_weights(m, out)
    info = dict(arch=arch, ds=ds, variant=var, seed=seed, epochs_run=len(hist['loss']),
                best_epoch=(int(np.argmin(hist['val_loss'])) + 1) if var != 'shuf' else len(hist['loss']),
                seed0_best_epoch=int(s0['best_epoch']), seconds=time.time() - t_ds, n_train=int(len(ytr)), test=H2.metrics(pte, yte),
                test_logit_std=float(lg.std()), test_logit_mean=float(lg.mean()), **extra)
    (out / 'info.json').write_text(json.dumps(info, indent=1)); (out / 'history.json').write_text(json.dumps(hist, indent=1))
    log(f'{ds}/{arch}/{var}: test {info["test"]}; logit std {info["test_logit_std"]:.4f}; wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['efn', 'pfn'], required=True); ap.add_argument('--ds', default='top')
    ap.add_argument('--variant', choices=['s1', 'shuf', 'init'], required=True); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--max_minutes', type=float, default=1260.0)
    a = ap.parse_args()
    import tensorflow as tf
    n = int(os.environ.get('OMP_NUM_THREADS', '1')); tf.config.threading.set_intra_op_parallelism_threads(n); tf.config.threading.set_inter_op_parallelism_threads(2)
    run(a); log('done')
