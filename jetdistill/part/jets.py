"""The cached jets of a ParT network (the counterpart of pipeline.jets for JEDI): particles, labels, the network's outputs
(stored with the dataset, see data.py) and the observables, in results/_jets/<network>/<which>/.

which: 'fit' (training jets, the first N_STEP4_FIT of the shuffled 'fit' split), 'dev' (validation), 'explain', 'test'
(the step-1 test split), 'full_test' (the whole test folder: millions of jets).
The jet-level observables are computed once and saved, one file per quantity (float32). ParT's per-particle and pair inputs
(blocks A and B of observables.library, ~35,000 quantities) are computed from the particles when a step uses them
(BlockQ); for 'full_test' the particles are read from the dataset folder (memory-mapped)."""
import json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .. import config
from ..config import DATA, RESULTS
from ..observables import compute, library
from ..observables.library import is_block
from ..observables.compute import block_values

LAZY = ('full_test',)


def rows(which):
    if which == 'full_test':
        r = np.arange(len(np.load(DATA / 'test' / 'label.npy', mmap_mode='r'))); return 'test', r[:config.N_FULL_TEST] if config.N_FULL_TEST else r
    S = np.load(DATA / 'splits.npz'); stop = dict(fit=config.N_STEP4_FIT, dev=config.N_DEV, explain=config.N_EXPLAIN, test=config.N_TEST_SPLIT)[which]
    return ('test' if which == 'test' else 'train'), S[which][:stop]


def core_ids(n):
    return [k for k in library(n) if not is_block(k)]


class BlockQ(dict):
    """observables of a set of jets (rows start..stop): the saved jet-level ones (read from their files, as float64) and
    blocks A and B (computed from the particles when asked for; the last `keep` are kept in memory). Iterating gives the
    jet-level ones only (every quantity that is summarised over all observables, e.g. ranges and rank correlations)."""
    def __init__(self, d, ids, part, n, lazy, start=0, stop=None, keep=4000):
        super().__init__({k: None for k in ids}); self.d, self.part, self.n, self.lazy, self.start, self.stop, self.keep = d, part, n, lazy, start, stop, keep
        self.cache, self.lib = {}, library(n)

    def __contains__(self, k): return k in self.lib

    def rows(self, a):
        return a[self.start:self.stop]

    def __getitem__(self, k):
        if k in self.cache: return self.cache[k]
        if is_block(k):
            if k not in self.lib: raise KeyError(k)
            x, jet, ext = (self.rows(a) for a in self.part()); v = block_values([k], x, jet, ext)[k]
        else:
            if not dict.__contains__(self, k): raise KeyError(k)
            v = np.asarray(self.rows(np.load(self.d / 'Q' / f'{k}.npy', mmap_mode='r')), np.float64)
        if len(self.cache) >= self.keep: self.cache.pop(next(iter(self.cache)))
        self.cache[k] = v; return v

    def many(self, ids):
        """several quantities at once (block quantities are computed together)"""
        need = [k for k in ids if k not in self.cache and is_block(k)]
        if need:
            x, jet, ext = (self.rows(a) for a in self.part()); self.cache.update(block_values(need, x, jet, ext))
        return {k: self[k] for k in ids}

    def items(self): return ((k, self[k]) for k in dict.keys(self))
    def values(self): return (self[k] for k in dict.keys(self))
    def keys(self): return dict.keys(self)
    def __iter__(self): return iter(dict.keys(self))

    def view(self, start, stop, keep=None):
        s0 = self.start + start; s1 = self.start + stop if self.stop is None else min(self.stop, self.start + stop)
        return BlockQ(self.d, list(dict.keys(self)), self.part, self.n, self.lazy, s0, s1, keep or self.keep)

    def sub(self, stop):
        return self.view(0, stop)

    def n_jets(self):
        return len(self.rows(np.load(self.d / 'y.npy', mmap_mode='r')))

    def chunks(self, size=200000):
        """views of consecutive rows (a formula on millions of jets is evaluated chunk by chunk)"""
        N = self.n_jets()
        for s in range(0, N, size): yield self.view(s, min(N, s + size), keep=100000)


def evaluate(fn, Q, size=200000):
    """fn(Q) -> array over the jets, chunk by chunk for the lazy test jets"""
    if not getattr(Q, 'lazy', False): return fn(Q)
    return np.concatenate([fn(q) for q in Q.chunks(size)])


def _core_chunk(args):
    n, folder, r, ids = args
    X = np.load(DATA / folder / 'x_f16.npy', mmap_mode='r'); E = np.load(DATA / folder / 'ext_f16.npy', mmap_mode='r'); jet = np.load(DATA / folder / 'jet.npy', mmap_mode='r')
    o = np.argsort(r); inv = np.argsort(o); rs = r[o]
    x, e, j = (np.asarray(A[rs], np.float32)[inv] for A in (X, E, jet))
    Q = compute(x, n, ids, jet=j, ext=e); return {k: Q[k].astype(np.float32) for k in ids}


def build(n, which, log=print, chunk=5000, workers=None):
    t0 = time.time(); folder, r = rows(which); d = RESULTS / '_jets' / str(n) / which; (d / 'Q').mkdir(parents=True, exist_ok=True); src = DATA / folder
    net = np.load(src / f'net_{n}.npz'); np.save(d / 'Z.npy', net['Z'][r]); np.save(d / 'L.npy', net['L'][r])
    np.save(d / 'y.npy', np.load(src / 'label.npy')[r]); np.save(d / 'jet.npy', np.load(src / 'jet.npy')[r])
    if which not in LAZY:                                   # particles of the smaller splits are kept with them
        for name in ('x_f16', 'ext_f16'):
            A = np.load(src / f'{name}.npy', mmap_mode='r'); o = np.argsort(r); a = np.empty((len(r),) + A.shape[1:], np.float16); a[o] = A[r[o]]; np.save(d / f'{name}.npy', a)
    ids = core_ids(n); out = {k: np.lib.format.open_memmap(d / 'Q' / f'{k}.npy', 'w+', np.float32, (len(r),)) for k in ids}
    parts = [(n, folder, r[s:s + chunk], ids) for s in range(0, len(r), chunk)]; workers = workers or min(int(os.environ.get('JETDISTILL_WORKERS', 4)), len(parts))
    with ProcessPoolExecutor(workers) as ex:
        for i, Q in enumerate(ex.map(_core_chunk, parts)):
            s = i * chunk
            for k in ids: out[k][s:s + len(Q[k])] = Q[k]
            if len(parts) > 1: log(f'  observables {which}: {min(s + chunk, len(r))} / {len(r)} jets, {time.time() - t0:.0f} s')
    for a in out.values(): a.flush()
    (d / 'meta.json').write_text(json.dumps(dict(folder=folder, n_jets=int(len(r)), ids=ids)))
    log(f'jets {which}: {len(r)} in {time.time() - t0:.0f} s')


def jets(n, which, log=print):
    """dict(x, jet, ext, y, Q, net, P, L, Z, H) as pipeline.jets gives it (H = Z: no activation). x and ext are float16
    arrays (memory-mapped for 'full_test'); Q is a BlockQ."""
    from ..metrics import softmax
    d = RESULTS / '_jets' / str(n) / which
    if not (d / 'meta.json').exists(): build(n, which, log)
    M = json.loads((d / 'meta.json').read_text()); lazy = which in LAZY
    if lazy:
        src = DATA / M['folder']; x = np.load(src / 'x_f16.npy', mmap_mode='r'); ext = np.load(src / 'ext_f16.npy', mmap_mode='r')
        if M['n_jets'] < len(x): x, ext = x[:M['n_jets']], ext[:M['n_jets']]
    else:
        x = np.load(d / 'x_f16.npy'); ext = np.load(d / 'ext_f16.npy')
    jet = np.load(d / 'jet.npy'); Q = BlockQ(d, M['ids'], lambda: (x, jet, ext), n, lazy, keep=64 if lazy else 4000)
    L = np.load(d / 'L.npy').astype(np.float64); Z = np.load(d / 'Z.npy', mmap_mode='r' if lazy else None)
    Z = Z if lazy else Z.astype(np.float64)
    return dict(x=x, jet=jet, ext=ext, y=np.load(d / 'y.npy').astype(int), Q=Q, net=L.argmax(1), P=softmax(L), L=L, Z=Z, H=Z)
