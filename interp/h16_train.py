"""H16 (loop D, claim D9, open question a): is the boost sensitivity of hls4ml-trained small nets caused by the training
dataset or by the architecture / inputs?  Trains identical small binary taggers (signal = top) on three datasets with a
matched number of training jets:
  hlsA  hls4ml jet archive, t vs (q or g), full hls4ml jet-pT spectrum;
  hlsB  the same, restricted to jets whose pT_J lies within +-8 % of the median pT_J of the hls4ml t + q/g training pool
        (the relative width of Top Landscape's 550-650 GeV window);
  top   Top Tagging Landscape, t vs QCD (550 < pT < 650 GeV).
pT_J = |sum over all stored particles of the pT vector| (hls4ml: the 150 stored slots, particles at phi = dphi about the stored
axis; Top: the parquet jet_pt). Number of training jets per class n_tr = min(#t, #q/g) inside the hlsB window, used for all
three datasets (random rows, seed 16; q/g = random rows of q union g). Validation: hls4ml splits.npz['dev'] rows (t, q/g,
balanced; window applied for hlsB), Top val split (10,000 per class). Test: hls4ml val archive (balanced, up to 25,000 per
class; window applied for hlsB), Top test split (25,000 per class); random rows, never first-k.
Inputs: the 64 hardest particles, (f, deta, dphi) with
  var 'frac'  f = pT_i / sum_64 pT  (hlsA, hlsB, top)
  var 'abs'   f = pT_i [GeV]        (hlsA, top)
normalised per feature with mean / std over the real particles of the training jets; padded slots set to 0 and masked.
Architectures (2 output logits, class 1 = top):
  jedis  train/models.py JediStyleTagger(d=64)             (the jedis-d64 of models/small, n_classes = 2)
  tf     train/models.py TransformerTagger(64, d=32, 2 blocks, 2 heads)  (tf-2b-d32)
  pfn    Particle Flow Network (Komiske, Metodiev, Thaler 2019; energyflow default sizes): Phi 3-100-100-128 (ReLU), masked sum,
         F 128-100-100-100 (ReLU), Linear(100 -> 2)
Training (train/train_small.py defaults): AdamW (wd 0.01), lr 1e-3, 200 warm-up steps then cosine decay over 30 epochs, batch
1024, early stop on validation loss (patience 5), best = lowest validation loss.
Output: /j-jepa-vol/aaron-interp/models/h16/<ds>-<var>-<arch>/seed<k>/{arch.json, log.csv, best.pt, test.json};
data: /j-jepa-vol/aaron-interp/data/h16/<ds>.npz + windows.json (written by --prep).
Run (GPU job):
  GPU: ./part_job.sh d-h16-train 'python -m interp.h16_train --prep && python -m interp.h16_train --all --conc 6'
  CPU (used, GPUs unavailable 2026-10-19; one thread per run):
  AI_PY=conda ./job.sh d-h16-train 8 10 'python -m interp.h16_train --prep && python -m interp.h16_train --all --conc 8'
  smoke: python -m interp.h16_train --prep --smoke && python -m interp.h16_train --all --smoke --conc 6
load(run_dir) -> Net (callable on raw (J, 64, 3) arrays (pT, deta, dphi) -> (J, 2) logits) is used by interp.h16_boost_dataset.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = Path('/j-jepa-vol/aaron-interp')
DATA = V / 'data'
H16D = DATA / 'h16'
MOD = V / 'models' / 'h16'
NP = 64
SEED_DATA = 16
CONFIGS = [(ds, var, arch) for ds, var in (('hlsA', 'frac'), ('hlsB', 'frac'), ('top', 'frac'), ('hlsA', 'abs'), ('top', 'abs'))
           for arch in ('jedis', 'tf', 'pfn')]
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def _models():
    import importlib.util
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M); return M


def jpt_of(x):
    """(J, L, 3) (pT, deta, dphi) -> |sum pT vector| (axis at phi 0)"""
    pt = x[..., 0].astype(np.float64); ph = x[..., 2].astype(np.float64)
    return np.hypot((pt * np.cos(ph)).sum(1), (pt * np.sin(ph)).sum(1))


# ======================================================================================================== data preparation
def prep(smoke=False):
    import h5py
    global H16D
    if smoke: H16D = DATA / 'h16' / '_smoke'
    H16D.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED_DATA)
    # ---- hls4ml
    t = time.time(); X = np.load(DATA / 'train_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(DATA / 'train_meta.h5', 'r') as f: y = f['label'][:].astype(np.int64)
    jpt = np.concatenate([jpt_of(np.asarray(X[i:i + 50000], np.float32)) for i in range(0, len(X), 50000)])
    log(f'hls4ml train archive {X.shape}, pT_J computed ({time.time() - t:.0f} s)')
    dev = np.sort(np.asarray(np.load(DATA / 'splits.npz')['dev'], np.int64)); isdev = np.zeros(len(y), bool); isdev[dev] = True
    pool = ~isdev & np.isin(y, [0, 1, 4])
    med = float(np.median(jpt[pool])); win = [0.92 * med, 1.08 * med]
    inw = (jpt >= win[0]) & (jpt <= win[1])
    tB = np.flatnonzero(pool & inw & (y == 4)); qB = np.flatnonzero(pool & inw & (y != 4))
    n_tr = int(min(len(tB), len(qB)))
    if smoke: n_tr = min(n_tr, 4000)
    info = dict(hls_median_jpt=med, hlsB_window=win, hls_pool=dict(t=int((pool & (y == 4)).sum()), qg=int((pool & (y != 4)).sum())),
                hlsB_pool=dict(t=int(len(tB)), qg=int(len(qB)), frac_in_window=float(inw[pool].mean())), n_train_per_class=n_tr,
                hls_jpt_quantiles=np.percentile(jpt[pool], [0.5, 2.5, 16, 50, 84, 97.5, 99.5]).tolist())
    log(f'hls4ml median pT_J {med:.1f} GeV, window {win[0]:.1f}-{win[1]:.1f}; in-window pool t {len(tB)} q/g {len(qB)} -> n_tr {n_tr}')
    pick = lambda a, n: np.sort(rng.choice(a, min(n, len(a)), replace=False))
    tA = np.flatnonzero(pool & (y == 4)); qA = np.flatnonzero(pool & (y != 4))
    dvt = np.flatnonzero(isdev & (y == 4)); dvq = np.flatnonzero(isdev & np.isin(y, [0, 1]))   # q/g only (fixed 2026-10-20: W/Z were included)
    nv = 300 if smoke else 10 ** 9
    Xv = np.load(DATA / 'val_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(DATA / 'val_meta.h5', 'r') as f: yv = f['label'][:].astype(np.int64)
    jv = np.concatenate([jpt_of(np.asarray(Xv[i:i + 50000], np.float32)) for i in range(0, len(Xv), 50000)])
    nte = 1000 if smoke else 25000
    for ds in ('hlsA', 'hlsB'):
        w = (lambda r: r) if ds == 'hlsA' else (lambda r: r[(jpt[r] >= win[0]) & (jpt[r] <= win[1])])
        wv = (lambda r: r) if ds == 'hlsA' else (lambda r: r[(jv[r] >= win[0]) & (jv[r] <= win[1])])
        tr = np.sort(np.r_[pick(w(tA), n_tr), pick(w(qA), n_tr)])
        dt_, dq_ = w(dvt), w(dvq); k = min(len(dt_), len(dq_), nv); va = np.sort(np.r_[pick(dt_, k), pick(dq_, k)])
        et, eq = wv(np.flatnonzero(yv == 4)), wv(np.flatnonzero(np.isin(yv, [0, 1]))); k2 = min(len(et), len(eq), nte)
        te = np.sort(np.r_[pick(et, k2), pick(eq, k2)])
        np.savez(H16D / f'{ds}.npz', x_tr=np.asarray(X[tr, :NP]), y_tr=(y[tr] == 4).astype(np.int8), jpt_tr=jpt[tr].astype(np.float32), idx_tr=tr,
                 x_va=np.asarray(X[va, :NP]), y_va=(y[va] == 4).astype(np.int8), idx_va=va,
                 x_te=np.asarray(Xv[te, :NP]), y_te=(yv[te] == 4).astype(np.int8), jpt_te=jv[te].astype(np.float32), idx_te=te)
        info[ds] = dict(n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
                        jpt_train_q=np.percentile(jpt[tr], [0.5, 50, 99.5]).tolist())
        log(f'{ds}: train {len(tr)} val {len(va)} test {len(te)}; pT_J train 0.5/50/99.5 % {info[ds]["jpt_train_q"]}')
    del X
    # ---- Top Landscape (kept when already written: the 2026-10-20 hls dev fix leaves it unchanged)
    info['hlsA_window'] = [float(np.percentile(jpt[pool], 0.5)), float(np.percentile(jpt[pool], 99.5))]
    old = H16D / 'windows.json'
    if (H16D / 'top.npz').exists() and old.exists() and not smoke:
        info['top'] = json.loads(old.read_text())['top']; info['smoke'] = smoke; old.write_text(json.dumps(info, indent=1))
        log(f'top.npz kept; wrote {old}'); return
    from interp import data_top as dt
    out = {}
    for split, n in (('train', n_tr), ('val', min(10000, nv)), ('test', nte)):
        d = dt.DST / split; yl = np.load(d / 'label.npy').astype(np.int64)
        r = np.sort(np.r_[pick(np.flatnonzero(yl == 1), n), pick(np.flatnonzero(yl == 0), n)])
        K = np.load(d / 'kin3.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy', mmap_mode='r')
        out[split] = (np.asarray(K[r, :NP]), yl[r].astype(np.int8), np.asarray(jet[r, 0], np.float32), r)
    np.savez(H16D / 'top.npz', x_tr=out['train'][0], y_tr=out['train'][1], jpt_tr=out['train'][2], idx_tr=out['train'][3],
             x_va=out['val'][0], y_va=out['val'][1], idx_va=out['val'][3],
             x_te=out['test'][0], y_te=out['test'][1], jpt_te=out['test'][2], idx_te=out['test'][3])
    info['top'] = dict(n_train=int(len(out['train'][1])), n_val=int(len(out['val'][1])), n_test=int(len(out['test'][1])),
                       jpt_train_q=np.percentile(out['train'][2], [0.5, 50, 99.5]).tolist(), window=[550.0, 650.0])
    info['hlsA_window'] = [float(np.percentile(jpt[pool], 0.5)), float(np.percentile(jpt[pool], 99.5))]
    info['smoke'] = smoke
    (H16D / 'windows.json').write_text(json.dumps(info, indent=1))
    log(f'top: train {info["top"]["n_train"]} val {info["top"]["n_val"]} test {info["top"]["n_test"]}; wrote {H16D}/windows.json')


# ======================================================================================================== models
def build(arch):
    import torch.nn as nn
    M = _models()
    if arch == 'jedis': return M.JediStyleTagger(d=64, rho=(32, 16), n_in=3, n_classes=2)
    if arch == 'tf': return M.TransformerTagger(NP, d=32, blocks=2, heads=2, mlp_mult=2, n_in=3, n_classes=2)
    if arch == 'pfn':
        class PFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = nn.Sequential(nn.Linear(3, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 128), nn.ReLU())
                self.F = nn.Sequential(nn.Linear(128, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 100), nn.ReLU(),
                                       nn.Linear(100, 2))

            def forward(self, x, mask):
                return self.F((self.phi(x) * mask.unsqueeze(-1).to(x.dtype)).sum(1))
        return PFN()
    raise ValueError(arch)


def features(x, var):
    """raw (J, 64, 3) (pT, deta, dphi) -> (features before normalisation, mask)"""
    x = np.asarray(x, np.float32).copy(); m = x[..., 0] > 0
    if var == 'frac': x[..., 0] = x[..., 0] / np.maximum(x[..., 0].sum(1, keepdims=True), 1e-9)
    x[~m] = 0; return x, m


class Net:
    """trained run: Net(run_dir)(raw (J, 64, 3)) -> (J, 2) float64 logits (CPU)"""

    def __init__(self, run_dir, device='cpu'):
        import torch
        self.A = json.loads((Path(run_dir) / 'arch.json').read_text()); self.dev = device
        self.net = build(self.A['arch']); self.net.load_state_dict(torch.load(Path(run_dir) / 'best.pt', map_location=device))
        self.net.to(device).eval(); self.sh = np.asarray(self.A['shift'], np.float32); self.sc = np.asarray(self.A['scale'], np.float32)

    def __call__(self, x, batch=5000):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                z, m = features(x[i:i + batch], self.A['var']); z = (z - self.sh) / self.sc; z[~m] = 0
                out.append(self.net(torch.from_numpy(z).to(self.dev), torch.from_numpy(m).to(self.dev)).double().cpu().numpy())
        return np.concatenate(out)


def load(run_dir, device='cpu'):
    return Net(run_dir, device)


# ======================================================================================================== training
def train_one(ds, var, arch, seed, epochs=30, patience=5, smoke=False):
    import torch, torch.nn.functional as F
    out = MOD / f'{ds}-{var}-{arch}' / f'seed{seed}'
    if smoke: out = MOD / '_smoke' / f'{ds}-{var}-{arch}' / f'seed{seed}'
    if (out / 'test.json').exists() and not smoke: log(f'{out}: done, skipped'); return
    out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(int(os.environ.get('H16_THREADS', '1')))
    D = np.load((H16D / '_smoke' if smoke else H16D) / f'{ds}.npz')
    ztr, mtr = features(D['x_tr'], var); zva, mva = features(D['x_va'], var); zte, mte = features(D['x_te'], var)
    real = ztr[mtr]; shift = real.mean(0).astype(float).tolist(); scale = real.std(0).astype(float).tolist()
    sh, sc = np.asarray(shift, np.float32), np.asarray(scale, np.float32)
    norm = lambda z, m: np.where(m[..., None], (z - sh) / sc, 0).astype(np.float32)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    Xtr, Mtr, Ytr = T(norm(ztr, mtr)), T(mtr), T(D['y_tr'].astype(np.int64))
    Xva, Mva, Yva = T(norm(zva, mva)), T(mva), T(D['y_va'].astype(np.int64))
    net = build(arch).to(dev); npar = sum(p.numel() for p in net.parameters())
    A = dict(ds=ds, var=var, arch=arch, seed=seed, n=NP, shift=shift, scale=scale, n_params=int(npar), n_train=int(len(Ytr)),
             n_val=int(len(Yva)), train=dict(epochs=epochs, patience=patience, lr=1e-3, wd=0.01, batch=1024, warmup=200),
             note='interp/h16_train.py (H16)')
    (out / 'arch.json').write_text(json.dumps(A, indent=1))
    log(f'{ds}-{var}-{arch} s{seed}: train {len(Ytr)} val {len(Yva)} test {len(D["y_te"])} params {npar} on {dev}')
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01); bs = 1024; spe = max(1, len(Ytr) // bs); total = epochs * spe
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / 200 if s < 200 else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - 200) / max(1, total - 200)))))
    best, bad = 1e9, 0
    with open(out / 'log.csv', 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'seconds'])
        for ep in range(1, epochs + 1):
            net.train(); perm = torch.randperm(len(Ytr), device=dev); tl = 0.0; t = time.time()
            for i in range(spe):
                b = perm[i * bs:(i + 1) * bs]; loss = F.cross_entropy(net(Xtr[b], Mtr[b]), Ytr[b])
                opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item()
            net.eval()
            with torch.no_grad():
                lo = torch.cat([net(Xva[i:i + 8192], Mva[i:i + 8192]) for i in range(0, len(Yva), 8192)])
                vl = F.cross_entropy(lo, Yva).item(); va = (lo.argmax(1) == Yva).float().mean().item()
            wr.writerow([ep, tl / spe, vl, va, time.time() - t]); fh.flush()
            log(f'{ds}-{var}-{arch} s{seed} epoch {ep}: train {tl / spe:.4f} val {vl:.4f} acc {va:.4f} ({time.time() - t:.1f} s)')
            if vl < best: best, bad = vl, 0; torch.save(net.state_dict(), out / 'best.pt'); A['best_epoch'] = ep
            else:
                bad += 1
                if bad >= patience: break
    (out / 'arch.json').write_text(json.dumps(A, indent=1))
    N = Net(out, dev); L = N(D['x_te']); yte = D['y_te'].astype(int); s = L[:, 1] - L[:, 0]
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, _ = roc_curve(yte, s); f5 = float(np.interp(0.5, tpr, fpr)); f3 = float(np.interp(0.3, tpr, fpr))
    rep = dict(accuracy=float((L.argmax(1) == yte).mean()), auc=float(roc_auc_score(yte, s)), rej_eps0p5=1 / f5 if f5 > 0 else None,
               rej_eps0p3=1 / f3 if f3 > 0 else None, n_test=int(len(yte)), best_epoch=A.get('best_epoch'))
    (out / 'test.json').write_text(json.dumps(rep, indent=1))
    log(f'{ds}-{var}-{arch} s{seed}: test accuracy {rep["accuracy"]:.4f} AUC {rep["auc"]:.4f} 1/eps_B(0.5) {rep["rej_eps0p5"]} -> {out}')


def run_all(conc, seeds, smoke):
    """runs every config x seed as subprocesses, at most conc at once; GPU monitor line every 3 min"""
    jobs = [(ds, var, arch, s) for s in seeds for ds, var, arch in CONFIGS]
    running, done, fails = [], 0, []
    lastmon = 0.0
    while jobs or running:
        while jobs and len(running) < conc:
            ds, var, arch, s = jobs.pop(0)
            cmd = [sys.executable, '-m', 'interp.h16_train', '--one', ds, var, arch, str(s)] + (['--smoke'] if smoke else [])
            lf = open(V / 'logs' / f'd-h16-{ds}-{var}-{arch}-s{s}{"-smoke" if smoke else ""}.log', 'w')
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', H16_THREADS='1')
            running.append(((ds, var, arch, s), subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env), lf))
        time.sleep(5)
        for r in list(running):
            if r[1].poll() is not None:
                r[2].close(); running.remove(r); done += 1
                if r[1].returncode != 0: fails.append(r[0])
                log(f'finished {r[0]} rc {r[1].returncode} ({done} done, {len(jobs)} queued, {len(running)} running)')
        if time.time() - lastmon > 180:
            lastmon = time.time()
            try:
                g = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
            except Exception as e: g = repr(e)
            log(f'monitor: GPU {g}; running {[r[0] for r in running]}')
    log(f'all done; failures {fails}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', action='store_true'); ap.add_argument('--all', action='store_true')
    ap.add_argument('--one', nargs=4); ap.add_argument('--conc', type=int, default=6); ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.prep: prep(a.smoke)
    if a.one:
        ds, var, arch, s = a.one; train_one(ds, var, arch, int(s), epochs=2 if a.smoke else 30, smoke=a.smoke)
    if a.all: sys.exit(run_all(a.conc, [int(s) for s in a.seeds.split(',')], a.smoke))


if __name__ == '__main__':
    main()
