"""The hls4ml LHC jet dataset (150 particles per jet; the networks see the first N) and the network's input scaling.

Files in config.DATA (made once by `python -m jetdistill.data <train.tar> <val.tar>`):
  train_ptetaphi_f16.npy, val_ptetaphi_f16.npy   (J, 150, 3) float16: pT [GeV], Δη, Δφ, in the archive's order
  train_meta.h5, val_meta.h5                      'label' (0..4 = g, q, W, Z, t)
  norm_stats.json                                 per-column mean / std over the whole training archive (padding
                                                  included), as JEDI-linear's dataloader computes them
  splits.npz                                      'fit', 'dev' (training archive), 'test' (validation archive = test file)
Nothing is fitted or chosen on the validation archive: it is the test file."""
import io, json, sys, tarfile
import h5py
import numpy as np
from .config import DATA

COLS = [5, 8, 11]                     # pT, Δη, Δφ columns of jetConstituentList
NS = [8, 16, 32, 64, 128]
SPLIT_SIZES = dict(fit=30000, dev=5000)   # per class, from the training archive
TEST_PER_CLASS = 10000                     # per class, from the validation archive (the 50,000-jet test split)


def archive(split):
    return 'val' if split in ('test', 'val', 'full_test') else 'train'


def rows(split):
    """row numbers of a split in its archive; 'full_test' = the whole test file"""
    if split == 'full_test':
        return np.arange(len(np.load(DATA / 'val_ptetaphi_f16.npy', mmap_mode='r')))
    return np.load(DATA / 'splits.npz')[split]


def particles(split, n, start=0, stop=None):
    """(J, n, 3) float32 particles of jets start..stop of a split"""
    X = np.load(DATA / f'{archive(split)}_ptetaphi_f16.npy', mmap_mode='r')
    return np.asarray(X[rows(split)[start:stop], :n]).astype(np.float32)


def labels(split, start=0, stop=None):
    with h5py.File(DATA / f'{archive(split)}_meta.h5') as f: y = f['label'][:]
    return y[rows(split)[start:stop]].astype(int)


def inputs(x, n):
    """particles (J, n, 3) -> the network's input, scaled exactly as JEDI-linear's dataloader"""
    st = json.loads((DATA / 'norm_stats.json').read_text())[str(n)]
    return (np.asarray(x[:, :n]).astype(np.float32) - np.float32(st['shift'])) / np.float32(st['scale'])


# ---------------------------------------------------------------- building the files (once)
def build(tar_path, split):
    feats, labs = [], []
    with tarfile.open(tar_path, 'r|*') as t:
        for m in t:
            if not (m.isfile() and m.name.endswith('.h5')): continue
            with h5py.File(io.BytesIO(t.extractfile(m).read())) as f:
                lab = f['jets'][:, -6:-1]; assert np.all(lab.sum(1) == 1)
                feats.append(np.array(f['jetConstituentList']).astype(np.float16)[..., COLS]); labs.append(lab.argmax(1))
    np.save(DATA / f'{split}_ptetaphi_f16.npy', np.concatenate(feats))
    with h5py.File(DATA / f'{split}_meta.h5', 'w') as f: f['label'] = np.concatenate(labs)


def norm_stats():
    X = np.load(DATA / 'train_ptetaphi_f16.npy', mmap_mode='r'); out = {}
    for n in NS:
        a = np.array(X[:, :n]).astype(np.float32); out[n] = dict(shift=np.mean(a, axis=(0, 1)).tolist(), scale=np.std(a, axis=(0, 1)).tolist())
    (DATA / 'norm_stats.json').write_text(json.dumps(out, indent=1))


def make_splits():
    """class-balanced splits: fit / dev from the training archive, test from the validation archive (fixed seeds)"""
    rng = np.random.default_rng(20260923)
    with h5py.File(DATA / 'train_meta.h5') as f: y = f['label'][:]
    perm = {c: rng.permutation(np.flatnonzero(y == c)) for c in range(5)}; out = {}; lo = 0
    for name, k in SPLIT_SIZES.items():
        out[name] = np.sort(np.concatenate([perm[c][lo:lo + k] for c in range(5)])); lo += k
    with h5py.File(DATA / 'val_meta.h5') as f: yv = f['label'][:]
    rt = np.random.default_rng(20260924)
    out['test'] = np.sort(np.concatenate([rt.choice(np.flatnonzero(yv == c), TEST_PER_CLASS, replace=False) for c in range(5)]))
    np.savez(DATA / 'splits.npz', **out)


if __name__ == '__main__':
    DATA.mkdir(parents=True, exist_ok=True)
    build(sys.argv[1], 'train'); build(sys.argv[2], 'val'); norm_stats(); make_splits()
