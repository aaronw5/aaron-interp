"""H12 arm 2 (weakness 7: is D1's sum-pT shortcut an artefact of the hls4ml pT spectrum?): retrain the float JEDI-style tagger
jedis-d64 at N = 64 (train/models.py JediStyleTagger, the architecture of models/small/jedis-d64-n64) with
  ptnorm  constituent pT replaced by pT_i / sum_j pT_j (sum over the 64 visible particles; the jet pT is not in the 3-feature
          archive), (deta, dphi) unchanged; normalisation shift / scale = per-feature mean / std of the training jets (padding
          included, JEDI's procedure) measured on 100,000 random training jets;
  ptflat  raw (pT, deta, dphi) with the norm_stats.json[64] normalisation of every small model, and per-jet training weights
          that make the sum-pT distribution of each class flat: weight = 1 / (class count in its sum-pT bin), 40 bins between
          the 0.5 and 99.5 % quantiles of the training sum pT (outside: edge bins), normalised to mean 1 per class, cap 20;
  raw     pipeline-matched control: raw inputs, norm_stats.json[64], no weights (same script, same schedule).
Training as train/train_small.py defaults (AdamW wd 0.01, lr 1e-3, 200 warm-up steps, cosine over 30 epochs, batch 1024,
early stop on the dev split's unweighted loss, patience 5); training = train archive minus splits.npz['dev']; test = the val
archive (260,000 jets). Output: /j-jepa-vol/aaron-interp/models/small/jedis-d64-n64-h12<variant>/seed<k>/ (arch.json, log.csv,
best.pt, test.json). Run (GPU; four runs on one GPU):
  ./part_job.sh d-h12-jnorm 'for v in ptnorm ptflat; do for s in 0 1; do python -m interp.h12_jnorm_train --variant $v --seed $s & done; done; wait'
load(run_dir) -> (net, prep), logits(net, prep, x_raw (J, 64, 3)) are used by interp.h12_boost_panel.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = Path('/j-jepa-vol/aaron-interp')
CLASSES = ['g', 'q', 'W', 'Z', 't']


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


def _models():
    import importlib.util
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M); return M


def transform(x, variant):
    """raw (J, n, 3) float32 -> variant features (pT column replaced for ptnorm)"""
    x = np.asarray(x, np.float32).copy()
    if variant == 'ptnorm':
        s = x[..., 0].sum(1, keepdims=True); x[..., 0] = x[..., 0] / np.maximum(s, 1e-6)
    return x


def make_prep(variant, shift, scale):
    shift = np.asarray(shift, np.float32); scale = np.asarray(scale, np.float32)

    def prep(x):
        m = np.asarray(x)[..., 0] > 0; z = (transform(x, variant) - shift) / scale; z[~m] = 0; return z, m
    return prep


def load(run_dir):
    import torch
    run_dir = Path(run_dir); A = json.loads((run_dir / 'arch.json').read_text()); M = _models()
    net = M.build_model(M.parse_model(A['model']), A['n']); net.load_state_dict(torch.load(run_dir / 'best.pt', map_location='cpu')); net.eval()
    return net, make_prep(A['variant'], A['shift'], A['scale'])


def logits(net, prep, x, batch=4000):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            z, m = prep(x[i:i + batch]); out.append(net(torch.from_numpy(z), torch.from_numpy(m)).double().numpy())
    return np.concatenate(out)


def flat_weights(sumpt, y, nb=40, cap=20.0):
    edges = np.quantile(sumpt, np.linspace(0.005, 0.995, nb + 1)); b = np.clip(np.searchsorted(edges, sumpt) - 1, 0, nb - 1)
    w = np.zeros(len(y))
    for c in range(5):
        s = y == c; cnt = np.bincount(b[s], minlength=nb).astype(float); w[s] = 1.0 / np.maximum(cnt[b[s]], 1)
        w[s] = np.minimum(w[s] / w[s].mean(), cap); w[s] /= w[s].mean()
    return w.astype(np.float32), edges


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', required=True, choices=['ptnorm', 'ptflat', 'raw'])
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--data_dir', default=str(V / 'data'))
    ap.add_argument('--out', default=''); ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--max_train', type=int, default=0); ap.add_argument('--max_test', type=int, default=0)
    a = ap.parse_args()
    import torch, torch.nn.functional as F
    sys.path.insert(0, str(ROOT / 'train'))
    from train_small import load_archive, dev_indices, roc_metrics
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out or V / 'models' / 'small' / f'jedis-d64-n64-h12{a.variant}' / f'seed{a.seed}'); out.mkdir(parents=True, exist_ok=True)
    n = 64; n_tot = len(np.load(os.path.join(a.data_dir, 'train_ptetaphi_f16.npy'), mmap_mode='r'))
    dv, dsrc = dev_indices(a.data_dir, n_tot); tr = np.setdiff1d(np.arange(n_tot), dv)
    if a.max_train: tr = np.sort(np.random.default_rng(a.seed).choice(tr, a.max_train, replace=False))
    t = time.time(); xtr, ytr = load_archive(a.data_dir, 'train', n, tr); xdv, ydv = load_archive(a.data_dir, 'train', n, dv)
    xte, yte = load_archive(a.data_dir, 'val', n)
    if a.max_test: k = np.sort(np.random.default_rng(0).choice(len(yte), a.max_test, replace=False)); xte, yte = xte[k], yte[k]
    log(f'{a.variant} seed {a.seed}: train {len(ytr)} dev {len(ydv)} test {len(yte)} on {dev} ({time.time() - t:.0f} s load)')
    sp_tr = xtr[..., 0].astype(np.float32).sum(1)
    if a.variant == 'ptnorm':
        r = np.random.default_rng(0).choice(len(ytr), min(100000, len(ytr)), replace=False); z = transform(xtr[np.sort(r)], 'ptnorm')
        shift = z.reshape(-1, 3).mean(0).astype(float).tolist(); scale = z.reshape(-1, 3).std(0).astype(float).tolist()
        w = np.ones(len(ytr), np.float32); edges = None
    else:
        st = json.loads((Path(a.data_dir) / 'norm_stats.json').read_text())[str(n)]; shift, scale = st['shift'], st['scale']
        if a.variant == 'ptflat': w, edges = flat_weights(sp_tr, ytr)
        else: w, edges = np.ones(len(ytr), np.float32), None
    prep = make_prep(a.variant, shift, scale)
    M = _models(); net = M.build_model(M.parse_model('jedis-d64'), n).to(dev)
    A = dict(model='jedis-d64', n=n, variant=a.variant, seed=a.seed, shift=shift, scale=scale, split=dsrc, n_train=int(len(ytr)),
             flat_edges=None if edges is None else edges.tolist(), weight_stats=dict(min=float(w.min()), max=float(w.max()), mean=float(w.mean())),
             train=dict(epochs=a.epochs, patience=a.patience, lr=1e-3, wd=0.01, batch=1024, warmup=200),
             note='interp/h12_jnorm_train.py (H12 arm 2, weakness 7)')
    (out / 'arch.json').write_text(json.dumps(A, indent=1))

    def to_dev(x):
        z, m = prep(x); return torch.from_numpy(z).to(dev), torch.from_numpy(m).to(dev)
    Xtr, Mtr = to_dev(xtr); Ytr = torch.from_numpy(ytr).to(dev); Wtr = torch.from_numpy(w).to(dev)
    Xdv, Mdv = to_dev(xdv); Ydv = torch.from_numpy(ydv).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01); bs = 1024; spe = len(ytr) // bs; total = a.epochs * spe
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / 200 if s < 200 else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - 200) / max(1, total - 200)))))
    best, bad, step = 1e9, 0, 0
    with open(out / 'log.csv', 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'seconds'])
        for ep in range(1, a.epochs + 1):
            net.train(); perm = torch.randperm(len(ytr), device=dev); tl = 0.0; t = time.time()
            for i in range(spe):
                b = perm[i * bs:(i + 1) * bs]
                loss = (F.cross_entropy(net(Xtr[b], Mtr[b]), Ytr[b], reduction='none') * Wtr[b]).sum() / Wtr[b].sum()
                opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item(); step += 1
            net.eval()
            with torch.no_grad():
                lo = torch.cat([net(Xdv[i:i + 8192], Mdv[i:i + 8192]) for i in range(0, len(ydv), 8192)])
                vl = F.cross_entropy(lo, Ydv).item(); va = (lo.argmax(1) == Ydv).float().mean().item()
            wr.writerow([ep, tl / spe, vl, va, time.time() - t]); fh.flush()
            log(f'epoch {ep}: train {tl / spe:.4f} val {vl:.4f} acc {va:.4f} ({time.time() - t:.0f} s)')
            if vl < best: best, bad = vl, 0; torch.save(net.state_dict(), out / 'best.pt'); A['best_epoch'] = ep
            else:
                bad += 1
                if bad >= a.patience: break
    net.load_state_dict(torch.load(out / 'best.pt', map_location=dev)); net.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(yte), 8192):
            z, m = to_dev(xte[i:i + 8192]); P.append(torch.softmax(net(z, m), -1).float().cpu().numpy())
    P = np.concatenate(P); acc = float((P.argmax(1) == yte).mean())
    rep = dict(accuracy=acc, per_class={c: float((P.argmax(1)[yte == k] == k).mean()) for k, c in enumerate(CLASSES)}, n_test=int(len(yte)))
    try: rep['auc'], rep['rej50'] = roc_metrics(P, yte)
    except Exception as e: rep['roc_error'] = repr(e)
    (out / 'test.json').write_text(json.dumps(rep, indent=1, default=float)); (out / 'arch.json').write_text(json.dumps(A, indent=1))
    log(f'test accuracy {acc:.4f} -> {out}')


if __name__ == '__main__':
    main()
