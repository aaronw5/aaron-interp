"""Top Tagging Landscape loader (Kasieczka et al., SciPost Phys. 7, 014 (2019); parquet conversion of the particle_transformer
repo, utils/convert_top_datasets.py). Read-only source: /j-jepa-vol/ParT_top/TopLandscape/{train,val,test}_file.parquet
(1.2M / 400k / 404k jets; anti-kT R = 0.8, 550 < pT < 650 GeV, |eta| < 2, Delphes, up to 200 constituents; label 1 = top).

Converted arrays (written only by convert(); read by sample()) in $AI_DATA_TOP (default /j-jepa-vol/aaron-interp/data/top)/<split>/:
  label.npy   (J,) int8                      1 = signal (top for this set; quark for QG)
  jet.npy     (J, 6) float32                 jet_pt, jet_eta, jet_phi, jet_energy, jet_mass, jet_nparticles
  kin3.npy    (J, 128, 3) float16            (pT [GeV], deta, dphi) of the 128 hardest constituents, pT-sorted, 0-padded
  full.npy    (J, L, 6) float32 (val/test)   (px, py, pz, E, deta, dphi) of all constituents (L = max multiplicity), pT-sorted
  pid.npy     (J, L) int16 (QG only)         PDG id (0 in padding)
  is_val.npy  (J,) bool (QG train only)      last 11.11 % rows of each source file = the val split weaver uses (--train-val-split 0.8889)
part_deta is (eta_i - jet_eta) * sign(jet_eta) (as JetClass); part_dphi = phi_i - jet_phi wrapped to (-pi, pi].

sample(per_class, seed, split='test') -> dict of numpy arrays for random rows (rng.choice per class, sorted): 'idx', 'label',
'jet', 'full', 'kin3' (+ 'pid'). Same API shape as interp.jedi_core.sample / interp.part_core.sample: random rows, never first-k.
"""
import os
import time
from pathlib import Path
import numpy as np

SRC = Path('/j-jepa-vol/ParT_top/TopLandscape')
DST = Path(os.environ.get('AI_DATA_TOP', '/j-jepa-vol/aaron-interp/data/top'))
L_KIN = 128
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def _pad(col, L, dtype=np.float32):
    """pyarrow (large_)list array -> (J, L) padded numpy, rows truncated at L"""
    off = np.asarray(col.offsets); val = np.asarray(col.values)
    n = np.diff(off); J = len(n); out = np.zeros((J, L), dtype)
    rows = np.repeat(np.arange(J), n); pos = np.arange(len(val)) - np.repeat(off[:-1] - off[0], n)
    keep = pos < L; out[rows[keep], pos[keep]] = val[keep]
    return out


def convert_files(files, dst, full=True, pid=False, val_frac=None, batch=50_000):
    """parquet files -> dst/*.npy (see module docstring). val_frac: mark the last val_frac of each file's rows as is_val."""
    import pyarrow.parquet as pq
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    pfs = [pq.ParquetFile(f) for f in files]; Ns = [p.metadata.num_rows for p in pfs]; J = sum(Ns)
    L = int(max(np.max(p.read(columns=['jet_nparticles'])['jet_nparticles'].to_numpy()) for p in pfs))
    log(f'convert {len(files)} file(s) -> {dst}: {J} jets, max multiplicity {L}')
    o = lambda name, shape, dt: np.lib.format.open_memmap(dst / name, mode='w+', dtype=dt, shape=shape)
    LAB, JET, K3 = o('label.npy', (J,), np.int8), o('jet.npy', (J, 6), np.float32), o('kin3.npy', (J, L_KIN, 3), np.float16)
    FULL = o('full.npy', (J, L, 6), np.float32) if full else None
    PID = o('pid.npy', (J, L), np.int16) if pid else None
    ISV = np.zeros(J, bool)
    cols = ['label', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_mass', 'jet_nparticles', 'part_px', 'part_py', 'part_pz',
            'part_energy', 'part_deta', 'part_dphi'] + (['part_pid'] if pid else [])
    s = 0
    for pf, Nf in zip(pfs, Ns):
        if val_frac is not None: ISV[s + int(round(Nf * (1 - val_frac))): s + Nf] = True
        for b in pf.iter_batches(batch_size=batch, columns=cols):
            n = b.num_rows; g = lambda c: b.column(cols.index(c))
            P = {c: _pad(g('part_' + c), L) for c in ('px', 'py', 'pz', 'energy', 'deta', 'dphi')}
            pt = np.hypot(P['px'], P['py'])
            order = np.argsort(-pt, axis=1, kind='stable'); tk = lambda a: np.take_along_axis(a, order, 1)
            pt = tk(pt); P = {k: tk(v) for k, v in P.items()}
            dphi = (P['dphi'] + np.pi) % (2 * np.pi) - np.pi; dphi[pt == 0] = 0; P['dphi'] = dphi
            LAB[s:s + n] = g('label').to_numpy().astype(np.int8)
            JET[s:s + n] = np.stack([g(c).to_numpy().astype(np.float32) for c in
                                     ('jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_mass', 'jet_nparticles')], 1)
            K3[s:s + n] = np.stack([pt[:, :L_KIN], P['deta'][:, :L_KIN], P['dphi'][:, :L_KIN]], -1).astype(np.float16)
            if full: FULL[s:s + n] = np.stack([P[k] for k in ('px', 'py', 'pz', 'energy', 'deta', 'dphi')], -1)
            if pid: PID[s:s + n] = tk(_pad(g('part_pid'), L)).astype(np.int16)
            s += n
        log(f'  {s}/{J} rows')
    assert s == J
    if val_frac is not None: np.save(dst / 'is_val.npy', ISV)
    for a in (LAB, JET, K3, FULL, PID):
        if a is not None: a.flush()
    return dict(J=J, L=L, n_val=int(ISV.sum()) if val_frac is not None else None)


def convert():
    info = {}
    for split in ('test', 'val', 'train'):
        if (DST / split / 'kin3.npy').exists() and (DST / split / 'done').exists():
            log(f'{split}: exists'); continue
        info[split] = convert_files([SRC / f'{split}_file.parquet'], DST / split, full=split != 'train')
        (DST / split / 'done').write_text('ok')
    return info


def sample(per_class=5000, seed=0, split='test', root=None, classes=(0, 1)):
    d = Path(root or DST) / split
    y = np.load(d / 'label.npy'); rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in classes]))
    out = dict(idx=idx, label=y[idx], jet=np.asarray(np.load(d / 'jet.npy', mmap_mode='r')[idx]),
               kin3=np.asarray(np.load(d / 'kin3.npy', mmap_mode='r')[idx], np.float32))
    for k in ('full', 'pid'):
        if (d / f'{k}.npy').exists(): out[k] = np.asarray(np.load(d / f'{k}.npy', mmap_mode='r')[idx])
    return out


def load_split(split, root=None, keys=('label', 'jet', 'full')):
    """all rows of a split (memory-mapped)"""
    d = Path(root or DST) / split
    return {k: np.load(d / f'{k}.npy', mmap_mode='r') for k in keys if (d / f'{k}.npy').exists()}


if __name__ == '__main__':
    print(convert())
