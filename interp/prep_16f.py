"""The 16-feature inputs of the official JEDI-linear 16-feature models, from the raw hls4ml 150p archive (opened read-only).

  python -m interp.prep_16f <raw_dir>        (raw_dir has train/*.h5 and val/*.h5, the zenodo 3602260 150p files)

Reads   jetConstituentList (J, 150, 16) and jets[:, -6:-1] (one-hot g, q, W, Z, t) of every file.
Writes  DATA/norm_stats_16f.json  {n: {shift, scale}} for n in 8..128: mean / std of the float16-rounded features over the
                                  whole training archive's first n slots (padding included), computed exactly as JEDI-linear's
                                  prepare_dataset.py (float16) + src/dataloader.py: np.mean / np.std over axes (0, 1) of the
                                  float32 array. numpy accumulates these in float32 (they differ from float64 by up to 4%), so
                                  the float32 computation is kept; file order changes them by ~1e-4 relative, the number of
                                  columns in the array not at all (checked on the 3-feature data).
        DATA/val_c16_f16.npy      (260000, 128, 16) float16: the test file's first 128 slots, all 16 columns, in the jet order
                                  of val_ptetaphi_f16.npy (each raw file located there by its jets; checked below)
Checks  columns 5, 8, 11 == val_ptetaphi_f16[:, :128]; labels == val_meta.h5; the 3-feature columns' stats == norm_stats.json.
"""
import json, sys, time
from pathlib import Path
import h5py
import numpy as np
from jetdistill.config import DATA

NS = [8, 16, 32, 64, 128]
NMAX = 128
t0 = time.time()
raw = Path(sys.argv[1])
tr, va = sorted((raw / 'train').glob('*.h5')), sorted((raw / 'val').glob('*.h5'))
print(f'{len(tr)} train files, {len(va)} val files', flush=True)


def read(p):
    with h5py.File(p, 'r') as f:
        lab = np.asarray(f['jets'][:, -6:-1]); assert np.all(lab.sum(1) == 1)
        x = np.asarray(f['jetConstituentList']).astype(np.float16)
    assert x.shape[1:] == (150, 16), x.shape
    return x, lab.argmax(1)


# ---------------- training-archive mean / std, float32 as the official dataloader
X16 = []
for i, p in enumerate(tr):
    X16.append(read(p)[0][:, :NMAX].copy())
    if i % 10 == 0: print(f'  train {i + 1}/{len(tr)} ({time.time() - t0:.0f}s)', flush=True)
X16 = np.concatenate(X16); J = len(X16)
out = {}
for n in NS:
    sh, sc = np.zeros(16), np.zeros(16)
    for c0 in (0, 4, 8, 12):   # C-contiguous (J, n, 4) float32, the official layout (a fancy-indexed copy is not C-ordered
        #                         and numpy then sums pairwise, which gives the float64 values instead)
        a = np.ascontiguousarray(X16[:, :n, c0:c0 + 4], dtype=np.float32); assert a.flags.c_contiguous
        sh[c0:c0 + 4] = np.mean(a, axis=(0, 1)); sc[c0:c0 + 4] = np.std(a, axis=(0, 1)); del a
    out[str(n)] = dict(shift=sh.tolist(), scale=sc.tolist())
del X16
print(f'train jets {J}', flush=True)
ref = json.loads((DATA / 'norm_stats.json').read_text())
for n in NS:
    d_sh = np.abs(np.array(out[str(n)]['shift'])[[5, 8, 11]] - ref[str(n)]['shift']).max()
    d_sc = np.abs(np.array(out[str(n)]['scale'])[[5, 8, 11]] / np.array(ref[str(n)]['scale']) - 1).max()
    d_shr = np.abs((np.array(out[str(n)]['shift'])[[5, 8, 11]] - ref[str(n)]['shift']) / np.array(ref[str(n)]['scale'])).max()
    print(f'  n={n:3d}: 3-feature columns vs norm_stats.json: max |Δshift| {d_sh:.2e} ({d_shr:.1e} of scale), max |Δscale/scale| {d_sc:.2e}', flush=True)
    assert d_shr < 1e-3 and d_sc < 1e-3, 'training stats do not match norm_stats.json'
(DATA / 'norm_stats_16f.json').write_text(json.dumps(out, indent=1))

# ---------------- test file, aligned to val_ptetaphi_f16.npy
V = np.load(DATA / 'val_ptetaphi_f16.npy')
with h5py.File(DATA / 'val_meta.h5', 'r') as f: yv = f['label'][:]
tmp = DATA / 'val_c16_f16.tmp.npy'
Y = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=(len(V), NMAX, 16))
filled = np.zeros(len(V), bool)
for i, p in enumerate(va):
    x, y = read(p)
    hit = np.flatnonzero(np.all(V.reshape(len(V), -1) == x[0][:, [5, 8, 11]].reshape(-1), axis=1))
    ok = [h for h in hit if h + len(x) <= len(V) and np.array_equal(V[h:h + len(x)], x[..., [5, 8, 11]])]
    assert len(ok) == 1, f'{p.name}: {len(hit)} first-jet matches, {len(ok)} block matches'
    h = ok[0]; assert not filled[h:h + len(x)].any() and np.array_equal(yv[h:h + len(x)], y), p.name
    Y[h:h + len(x)] = x[:, :NMAX]; filled[h:h + len(x)] = True
    print(f'  val {i + 1}/{len(va)} {p.name} -> rows {h}..{h + len(x)} ({time.time() - t0:.0f}s)', flush=True)
assert filled.all(), f'{(~filled).sum()} test jets not found in the raw val files'
Y.flush(); del Y
Z = np.load(tmp, mmap_mode='r')
assert np.array_equal(np.asarray(Z[:, :, [5, 8, 11]]), V[:, :NMAX])
tmp.rename(DATA / 'val_c16_f16.npy')
print(f'wrote {DATA / "val_c16_f16.npy"} {Z.shape} and norm_stats_16f.json; checks passed; {time.time() - t0:.0f}s', flush=True)
