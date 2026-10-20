"""X9 training: an Energy Flow Network (EFN) or Particle Flow Network (PFN) (energyflow.archs, Komiske, Metodiev, Thaler 2019)
on JEDI-linear's 3-feature input (pT, Δη, Δφ), first 64 particles, for the comparison of node X9.

Run with the isolated venv (never the base env):
  ./job.sh x9-efn-a 4 8 '/j-jepa-vol/aaron-interp/venv-ef/bin/python -m interp.x9_train --arch efn --phi 100,100,128 --seed 0'
Inputs (per jet, padding = rows with pT = 0):
  efn  z_i = pT_i / Σ_j pT_j (over the first 64 slots), angles (Δη_i, Δφ_i) as stored (rad, about the stored jet axis);
       latent L_k = Σ_i z_i Φ_k(Δη_i, Δφ_i)  (IRC-safe; padding contributes 0 through z = 0)
  pfn  (pT, Δη, Δφ) normalised exactly as JEDI-linear / train_small: (x − shift[64]) / scale[64] from norm_stats.json, padding
       rows then set to 0 (energyflow masks rows equal to 0); latent L_k = Σ_i Φ_k(pT̃_i, Δη̃_i, Δφ̃_i)
Split (as train/train_small.py): training = train archive minus splits.npz['dev'] (595,000 jets); early stopping on dev (25,000);
test = the whole val archive (260,000 jets; never used for selection).  F = (100, 100, 100), ReLU, softmax output, Adam 1e-3,
batch 500, early stopping on dev loss (patience 5, best weights restored), at most --epochs epochs and --max_minutes wall.
Writes $AI_RESULTS/jedi/x9_models/<arch>_<phi>_s<seed>/: weights.npz (dense kernels / biases in forward order, keys
phi{k}_W, phi{k}_b, F{k}_W, F{k}_b, out_W, out_b), info.json (accuracies, epochs, timing), check.npz (softmax on
sample(2000, 313) test jets, to verify the numpy forward of interp.x9_efn_pfn), history.json.
X9w opt-in flags (defaults reproduce X9 exactly):
  --recentre   every jet's real particles get (Δη, Δφ) − c, c = Σ_i pT_i (Δη_i, Δφ_i) / Σ_i pT_i over the real particles of the first
               64 slots (pT-weighted centroid), for train, dev, test and check jets alike, before the EFN / PFN features
               (PFN: then normalised with the unchanged norm_stats.json); recentre_np below is the definition the analysis uses.
  --aug A      (A > 0) translation augmentation of the training jets: every real particle of a jet moved by the same
               c = |c| (cos θ, sin θ), |c| ~ U(0, A) rad, θ ~ U(0, 2π), drawn per jet and redrawn every epoch (rng seed + 1000);
               applied after recentring when both flags are set (otherwise the two would cancel exactly). Dev / test unaugmented.
               With --aug the fit runs one epoch per call (fresh augmented features each epoch) with the same early stopping
               (patience 5 on dev loss, best weights restored), written out by hand.
  --max_train  random subset of the training jets (smoke tests only).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
DATA = Path(os.environ['JETDISTILL_DATA']); RES = Path(os.environ['AI_RESULTS']) / 'jedi' / 'x9_models'
N = 64


def load(prefix, rows=None):
    import h5py
    X = np.load(DATA / f'{prefix}_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(DATA / f'{prefix}_meta.h5', 'r') as f: y = f['label'][:].astype(np.int64)
    if rows is None: return np.asarray(X[:, :N]).astype(np.float32), y
    return np.asarray(X[rows, :N]).astype(np.float32), y[rows]


def recentre_np(x):
    """(J, n, 3) raw (pT, Δη, Δφ) -> copy with the real particles' angles about the pT-weighted centroid (padding stays 0)"""
    x = np.array(x, np.float32, copy=True); real = x[..., 0] > 0; pt = np.where(real, x[..., 0], 0).astype(np.float64)
    c = (pt[..., None] * x[..., 1:3]).sum(1) / np.maximum(pt.sum(1), 1e-12)[:, None]
    x[..., 1:3] = np.where(real[..., None], x[..., 1:3] - c[:, None, :].astype(np.float32), x[..., 1:3]); return x


def augment_np(x, rng, amax):
    """rigid translation of every real particle of a jet by |c| ~ U(0, amax), direction ~ U(0, 2π), per jet"""
    x = np.array(x, np.float32, copy=True); real = x[..., 0] > 0; J = len(x)
    r = rng.uniform(0, amax, J); th = rng.uniform(0, 2 * np.pi, J)
    x[..., 1] += np.where(real, (r * np.cos(th))[:, None], 0).astype(np.float32)
    x[..., 2] += np.where(real, (r * np.sin(th))[:, None], 0).astype(np.float32); return x


def features(x, arch, st, ang_scale=1.0):
    """raw particles (J, 64, 3) -> the model input (EFN angles divided by ang_scale; 1 = as stored)"""
    real = x[..., 0] > 0
    if arch == 'efn':
        pt = x[..., 0] * real; z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12)
        return [z.astype(np.float32), (x[..., 1:3] / np.float32(ang_scale) * real[..., None]).astype(np.float32)]
    xn = (x - np.float32(st['shift'])) / np.float32(st['scale'])
    return (xn * real[..., None]).astype(np.float32)


def check_indices(per_class=2000, seed=313):
    y = np.asarray(np.load(DATA / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def export(model):
    """Dense kernels / biases in forward order; Φ layers are the TimeDistributed ones"""
    Ws = []
    for L in model.layers:
        w = L.get_weights()
        if len(w) == 2: Ws.append((L.__class__.__name__, L.name, w[0], w[1]))
    out, k_phi, k_F = {}, 0, 0
    names = []
    for i, (cls, name, K, b) in enumerate(Ws):
        if cls == 'TimeDistributed' or name.startswith('tdist'):
            key = f'phi{k_phi}'; k_phi += 1
        elif i == len(Ws) - 1:
            key = 'out'
        else:
            key = f'F{k_F}'; k_F += 1
        out[key + '_W'] = K.astype(np.float32); out[key + '_b'] = b.astype(np.float32); names.append((key, cls, name, list(K.shape)))
    return out, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['efn', 'pfn'], required=True)
    ap.add_argument('--phi', default='100,100,128'); ap.add_argument('--F', default='100,100,100')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--max_minutes', type=float, default=55.0); ap.add_argument('--batch', type=int, default=500)
    ap.add_argument('--tag', default='')
    ap.add_argument('--recentre', action='store_true', help='X9w: angles about the pT-weighted centroid (all jets)')
    ap.add_argument('--aug', type=float, default=0.0, help='X9w: max translation (rad) of the training augmentation; 0 = off')
    ap.add_argument('--max_train', type=int, default=0, help='random subset of the training jets (smoke tests)')
    ap.add_argument('--ang_scale', type=float, default=1.0, help='EFN only, exploratory: angles divided by this (rad); 1 = as stored')
    a = ap.parse_args()
    import tensorflow as tf
    nthr = int(os.environ.get('OMP_NUM_THREADS', '1'))
    tf.config.threading.set_intra_op_parallelism_threads(nthr); tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(a.seed)
    import energyflow as ef
    from energyflow.archs import EFN, PFN
    try:
        import tf_keras as K2                        # energyflow >= 1.4 builds its models with tf_keras (Keras 2)
    except ImportError:
        K2 = tf.keras
    K2.utils.set_random_seed(a.seed)
    phi = tuple(int(v) for v in a.phi.split(',')); Fs = tuple(int(v) for v in a.F.split(','))
    name = f'{a.arch}_{"-".join(map(str, phi))}_s{a.seed}{a.tag}'; out = RES / name; out.mkdir(parents=True, exist_ok=True)
    log(f'{name}: tensorflow {tf.__version__}, energyflow {ef.__version__}, threads {nthr}')
    st = json.loads((DATA / 'norm_stats.json').read_text())[str(N)]
    dev = np.sort(np.load(DATA / 'splits.npz')['dev']); x, y = load('train')
    tr = np.setdiff1d(np.arange(len(y)), dev)
    if a.max_train: tr = np.sort(np.random.default_rng(a.seed).choice(tr, a.max_train, replace=False))
    prep = recentre_np if a.recentre else (lambda v: v)
    if a.recentre:
        x = prep(x); log('recentred: max |pT-weighted centroid| after', float(np.abs((x[..., 0:1] * x[..., 1:3]).sum(1) / x[..., 0:1].sum(1)).max()))
    log(f'train archive {len(y)} jets; training {len(tr)}, dev {len(dev)}')
    Y = np.eye(5, dtype=np.float32)
    Xdv = features(x[dev], a.arch, st, a.ang_scale); ytr, ydv = y[tr], y[dev]
    if a.aug > 0:
        xtr_raw = np.ascontiguousarray(x[tr]); Xtr = None
    else:
        Xtr = features(x[tr], a.arch, st, a.ang_scale)
    del x
    if a.arch == 'efn':
        m = EFN(input_dim=2, Phi_sizes=phi, F_sizes=Fs, output_dim=5, summary=False)
    else:
        m = PFN(input_dim=3, Phi_sizes=phi, F_sizes=Fs, output_dim=5, summary=False)

    class Clock(K2.callbacks.Callback):
        def on_epoch_end(self, ep, logs=None):
            log(f'epoch {ep + 1}: ' + ', '.join(f'{k} {v:.4f}' for k, v in (logs or {}).items()))
            if time.time() - T0 > 60 * a.max_minutes:
                log('wall-time limit reached: stopping'); self.model.stop_training = True
    es = K2.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    t1 = time.time()
    if a.aug > 0:          # one epoch per fit call on freshly augmented features; early stopping by hand (as EarlyStopping)
        rng = np.random.default_rng(a.seed + 1000); hist = {}; best_v, best_w, bad = np.inf, None, 0
        for ep in range(a.epochs):
            te0 = time.time(); Xa = features(augment_np(xtr_raw, rng, a.aug), a.arch, st, a.ang_scale)
            h = m.fit(Xa, Y[ytr], epochs=1, batch_size=a.batch, validation_data=(Xdv, Y[ydv]), verbose=0); del Xa
            for k, v in h.history.items(): hist.setdefault(k, []).append(float(v[-1]))
            log(f'epoch {ep + 1} (aug): ' + ', '.join(f'{k} {v[-1]:.4f}' for k, v in hist.items()) + f' ({time.time() - te0:.0f} s)')
            if hist['val_loss'][-1] < best_v: best_v, best_w, bad = hist['val_loss'][-1], m.model.get_weights(), 0
            else: bad += 1
            if bad >= 5: log('early stop (patience 5)'); break
            if time.time() - T0 > 60 * a.max_minutes: log('wall-time limit reached: stopping'); break
        m.model.set_weights(best_w); del xtr_raw
    else:
        h = m.fit(Xtr, Y[ytr], epochs=a.epochs, batch_size=a.batch, validation_data=(Xdv, Y[ydv]), verbose=0, callbacks=[es, Clock()])
        hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    train_s = time.time() - t1; del Xtr
    best = int(np.argmin(hist['val_loss'])) + 1
    pdv = m.predict(Xdv, batch_size=5000, verbose=0); acc_dev = float((pdv.argmax(1) == ydv).mean())
    xt, yt = load('val'); xt = prep(xt); pt_ = m.predict(features(xt, a.arch, st, a.ang_scale), batch_size=5000, verbose=0); acc_test = float((pt_.argmax(1) == yt).mean())
    ic = check_indices(); xc = prep(np.asarray(np.load(DATA / 'n64' / 'full_test.npz', mmap_mode='r')['x'][ic]).astype(np.float32))
    assert np.array_equal(xc, xt[ic]), 'full_test.npz rows differ from the val archive'
    pc = m.predict(features(xc, a.arch, st, a.ang_scale), batch_size=5000, verbose=0)
    W, names = export(m.model)
    np.savez(out / 'weights.npz', **W); np.savez(out / 'check.npz', idx=ic, prob=pc.astype(np.float64))
    info = dict(arch=a.arch, ang_scale=a.ang_scale, recentre=bool(a.recentre), aug=a.aug, max_train=a.max_train, phi=phi, F=Fs, seed=a.seed, epochs_run=len(hist['loss']), best_epoch=best, train_seconds=train_s,
                seconds_per_epoch=train_s / max(len(hist['loss']), 1), acc_dev=acc_dev, acc_test=acc_test, n_train=int(len(tr)), n_dev=int(len(dev)),
                n_test=int(len(yt)), layers=names, tensorflow=tf.__version__, energyflow=ef.__version__, batch=a.batch, threads=nthr,
                per_class_test=[float((pt_.argmax(1)[yt == c] == c).mean()) for c in range(5)])
    (out / 'info.json').write_text(json.dumps(info, indent=1)); (out / 'history.json').write_text(json.dumps(hist, indent=1))
    log(f'{name}: dev acc {acc_dev:.4f}, test acc {acc_test:.4f} (N {len(yt)}), best epoch {best}/{len(hist["loss"])}, '
        f'{info["seconds_per_epoch"]:.0f} s/epoch; layers {names}; wrote {out}')


if __name__ == '__main__':
    main()
