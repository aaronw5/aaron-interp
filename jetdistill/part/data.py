"""JetClass (Qu, Li & Qian 2022; zenodo.org/record/6619768) for the ParT tagger: ROOT files -> arrays, network outputs.

Each split is a folder config.DATA/<split>/ (made once by the commands below):
  x_f16.npy      (J, 128, 3) float16  pT [GeV], Δη, Δφ of each particle (relative to the jet axis), hardest first, pT = 0 after
  ext_f16.npy    (J, 128, 7) float16  energy [GeV], charge, particle type (0 none, 1 charged hadron, 2 neutral hadron,
                                      3 photon, 4 electron, 5 muon), d0 [mm], σ(d0), dz [mm], σ(dz) (as the files store them)
  jet.npy        (J, 4) float32       jet pT [GeV], jet η, jet φ, jet energy [GeV]
  label.npy      (J,) int8            index into config.CLASSES
  net_<n>.npz    Z (J, 128), L (J, 10) float32: the network's neurons and class scores, computed from the file's own
                 float32 four-vectors (not from the stored float16 copies)
  files.json     the ROOT files and how many jets each gave
config.DATA/splits.npz: 'fit', 'dev', 'explain' (rows of 'train'), 'test' (rows of 'test', the step-1 test split).
Nothing is fitted or chosen on the test folder.

  python -m jetdistill.part.data example JetClass_example_100k.root      # the notebook's 100k jets: 50k train, 50k test
  python -m jetdistill.part.data build train <ROOT files ...> [--per-file K] [--nets full]
  python -m jetdistill.part.data build test  <ROOT files ...> [--per-file K] [--nets full]
  python -m jetdistill.part.data splits"""
import json, sys, time
import numpy as np
from ..config import DATA, CLASSES, NETWORKS

P = 128
LABELS = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
PBR = ['part_px', 'part_py', 'part_pz', 'part_energy', 'part_deta', 'part_dphi', 'part_d0val', 'part_d0err', 'part_dzval', 'part_dzerr', 'part_charge',
       'part_isChargedHadron', 'part_isNeutralHadron', 'part_isPhoton', 'part_isElectron', 'part_isMuon']
JBR = ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy']
PID = ['isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon']


def log(*a):
    print(*a, flush=True)


def read_root(path, start=None, stop=None):
    """a jet dict: every part_* branch as (J, 128) float32 (hardest first by pT, zero-padded), 'mask', jet_*, 'label'"""
    import uproot, awkward as ak
    t = uproot.open(path)['tree']; a = t.arrays(PBR + JBR + LABELS, entry_start=start, entry_stop=stop)
    lab = np.stack([ak.to_numpy(a[k]) for k in LABELS], 1); assert np.all(lab.sum(1) == 1)
    pt = np.hypot(a['part_px'], a['part_py']); order = ak.argsort(pt, ascending=False, stable=True)
    pad = lambda x: ak.to_numpy(ak.fill_none(ak.pad_none(x[order], P, clip=True), 0)).astype(np.float32)
    J = {k[5:]: pad(a[k]) for k in PBR}; J['mask'] = pad(ak.ones_like(a['part_px']))
    J.update({k: ak.to_numpy(a[k]).astype(np.float32) for k in JBR}); J['label'] = lab.argmax(1).astype(np.int8)
    return J


def stored(J):
    """the arrays kept on disk (float16 particles, the jet, the label)"""
    pt = np.hypot(J['px'], J['py']) * J['mask']
    x = np.stack([pt, J['deta'] * J['mask'], J['dphi'] * J['mask']], -1).astype(np.float16)
    typ = sum((i + 1) * J[k] for i, k in enumerate(PID))
    ext = np.stack([J['energy'], J['charge'], typ, J['d0val'], J['d0err'], J['dzval'], J['dzerr']], -1) * J['mask'][..., None]
    return dict(x_f16=x, ext_f16=ext.astype(np.float16), jet=np.stack([J[k] for k in JBR], 1).astype(np.float32), label=J['label'])


class Writer:
    """the folder of a split, filled chunk by chunk (memory-mapped: the test folder has millions of jets)"""
    SHAPES = dict(x_f16=((P, 3), np.float16), ext_f16=((P, 7), np.float16), jet=((4,), np.float32), label=((), np.int8))

    def __init__(self, split, N, nets=NETWORKS):
        self.d = DATA / split; self.d.mkdir(parents=True, exist_ok=True); self.N, self.i, self.nets = N, 0, nets
        mm = lambda k, shp, dt: np.lib.format.open_memmap(self.d / f'{k}.npy', 'w+', dt, (N,) + shp)
        self.a = {k: mm(k, shp, dt) for k, (shp, dt) in self.SHAPES.items()}
        for n in nets: self.a[f'Z_{n}'] = mm(f'_Z_{n}', (128,), np.float32); self.a[f'L_{n}'] = mm(f'_L_{n}', (10,), np.float32)

    def add(self, part):
        k = len(part['label'])
        for key, arr in self.a.items(): arr[self.i:self.i + k] = part[key]
        self.i += k

    def close(self, files):
        assert self.i == self.N, (self.i, self.N)
        for arr in self.a.values(): arr.flush()
        for n in self.nets:        # the network outputs as one file per network
            np.savez(self.d / f'net_{n}.npz', Z=np.load(self.d / f'_Z_{n}.npy'), L=np.load(self.d / f'_L_{n}.npy'))
            (self.d / f'_Z_{n}.npy').unlink(); (self.d / f'_L_{n}.npy').unlink()
        (self.d / 'files.json').write_text(json.dumps(files, indent=1))


def process(J, nets, log=log):
    out = stored(J)
    for n, net in nets.items():
        r = net.run(J, log=log); out[f'Z_{n}'] = r['z'].astype(np.float32); out[f'L_{n}'] = r['logits'].astype(np.float32)
    return out


def networks(nets=NETWORKS):
    from .network import ParTNetwork
    return {n: ParTNetwork(n) for n in nets}


def build(split, paths, per_file=None, chunk=20000, log=log, nets=NETWORKS):
    """every jet (or the first per_file jets) of each ROOT file, with the network outputs"""
    import uproot
    names = nets; nets = networks(names); sizes = [min(uproot.open(p)['tree'].num_entries, per_file or 10 ** 12) for p in paths]
    W = Writer(split, sum(sizes), names); files = []
    for p, N in zip(paths, sizes):
        t0 = time.time()
        for s in range(0, N, chunk): W.add(process(read_root(p, s, min(N, s + chunk)), nets, log))
        files.append(dict(file=str(p), jets=N)); log(f'{p}: {N} jets in {time.time() - t0:.0f} s ({W.i} / {W.N})')
    W.close(files)


def example(path, log=log):
    """the notebook's 100k-jet file, shuffled (fixed seed): the first 50k jets = 'train', the other 50k = 'test'"""
    nets = networks(); J = read_root(path); perm = np.random.default_rng(20260930).permutation(len(J['label']))
    for split, rows in (('train', perm[:len(perm) // 2]), ('test', perm[len(perm) // 2:])):
        t0 = time.time(); rows = np.sort(rows); W = Writer(split, len(rows))
        for s in range(0, len(rows), 10000): W.add(process({k: v[rows[s:s + 10000]] for k, v in J.items()}, nets, log))
        W.close([dict(file=str(path), jets=len(rows), rows='shuffled half')]); log(f'{split}: {len(rows)} jets in {time.time() - t0:.0f} s')


def labels(split):
    return np.load(DATA / split / 'label.npy').astype(int)


def make_splits(n_dev=None, n_explain=None):
    """class-mixed random splits of the training folder: 'dev' (validation), 'explain' (texts and pictures; may overlap
    'fit' as in the JEDI runs, never 'dev'), 'fit' (everything else); 'test' = a random subset of the test folder"""
    from .. import config
    y = labels('train'); rng = np.random.default_rng(20260923); perm = rng.permutation(len(y))
    n_dev = n_dev or min(config.N_DEV, len(y) // 5); dev = np.sort(perm[:n_dev]); fit = np.sort(perm[n_dev:])
    explain = np.sort(rng.choice(fit, min(config.N_EXPLAIN, len(fit)), replace=False))
    yt = labels('test'); test = np.sort(np.random.default_rng(20260924).choice(len(yt), min(config.N_TEST_SPLIT, len(yt)), replace=False))
    np.savez(DATA / 'splits.npz', fit=np.random.default_rng(1).permutation(fit), dev=dev, explain=explain, test=test)
    log(f'splits: fit {len(fit)}, dev {len(dev)}, explain {len(explain)}, test {len(test)} (of {len(yt)} test jets)')


if __name__ == '__main__':
    cmd, *args = sys.argv[1:]
    if cmd == 'example': example(args[0]); make_splits()
    elif cmd == 'build':
        per, nets = None, NETWORKS
        if '--per-file' in args: i = args.index('--per-file'); per = int(args[i + 1]); args = args[:i] + args[i + 2:]
        if '--nets' in args: i = args.index('--nets'); nets = tuple(args[i + 1].split(',')); args = args[:i] + args[i + 2:]
        build(args[0], args[1:], per, nets=nets)
    elif cmd == 'splits': make_splits()
