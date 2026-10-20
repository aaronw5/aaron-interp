"""P22 — what in JetClass training produces ParT's axis response (claims C3s / C10). Training half; analysis in
interp/p22_planted_true_part.py.

P14 / P16 trained ParT_kin (2 outputs) on a planted invariant-mass window label (arm a) and found R ~ 0.19 (R = read-out change
under a rigid phi + 0.06 rad shift about the stored axis / change under a matched real mass change); the released ParT gives
R 0.72-0.79. The two differ in labels (planted window vs 10 true classes) and in data scale. P22 trains, with the P14 code path
(interp/p14_planted_train.py: same 1.5M jets = 5 random train_100M files per class of WToQQ / ZToQQ / ZJetsToNuNu
(default_rng(14)), same 30k validation rows (default_rng(140)), same 20k held-out test rows of test_20M *_100 (default_rng(14)),
same architecture (weaver ParticleTransformer, kin inputs), optimiser (Ranger lr 1e-3, batch 512, AMP, 10 epochs, flat + decay
over the last 3 epochs)):
  t   true 3-class labels (0 QCD, 1 Z, 2 W), num_classes 3, seed 14 (initialisation torch.manual_seed(14), batch order
      default_rng(1000 + epoch): the seed of P14 arm a)
  t2  arm t with seed 15 (torch.manual_seed(15), batch order default_rng(15000 + epoch))
  u   arm t (seed 14) with random rigid-offset augmentation: every training jet in every batch gets a fresh offset
      c = |c| (cos psi, sin psi), |c| ~ U(0, 0.08) rad, psi ~ U(0, 2 pi) (numpy default_rng((14, epoch, batch))), added to every
      real particle's (deta, dphi) about the stored axis; px, py, pz, E rebuilt from (pT, eta = jet_eta + sign(jet_eta) deta,
      phi = jet_phi + dphi, m) as interp/part_core.build does (the released-input convention); jet_pt, jet_energy recomputed;
      pT order unchanged. Validation and test jets are not offset.
  a2  P14 arm a (label 1[|m_inv - 85| < 10 GeV] from the P14 side file data/p14/labels.npz) with seed 15
Outputs: /j-jepa-vol/aaron-interp/models/p22/<arm>/epoch-<e>.pt, final.pt, log.json (per-epoch train loss / acc, val acc,
held-out acc, 3x3 held-out confusion for 3-class arms, GPU utilisation samples).

  interp/p10_part_seeds_job.sh c-p22-t 'python -m interp.p22_planted_true_train --arm t'
  python -m interp.p22_planted_true_train --arm u --smoke      (1 file per class, 7k jets each, 60 steps)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.p14_planted_train import (V, SIDE, load_pool, load_test, Batcher, gpu_util, feed, log)

MOD = V / 'models' / 'p22'
ARMS = dict(t=dict(label='true', seed=14, aug=False, ncls=3), t2=dict(label='true', seed=15, aug=False, ncls=3),
            u=dict(label='true', seed=14, aug=True, ncls=3), a2=dict(label='a', seed=15, aug=False, ncls=2))
CMAX = 0.08


def offset(B, rng, cmax=CMAX):
    """random rigid offset of every real particle about the stored axis; B: batch jet dict (px, py, pz, energy, deta, dphi,
    mask, jet_eta, jet_phi, ...). Returns a new dict with the four-vectors rebuilt (part_core.build convention)."""
    m = B['mask'] > 0; n = len(m)
    r = cmax * rng.random(n); psi = 2 * np.pi * rng.random(n); ce, cp = (r * np.cos(psi))[:, None], (r * np.sin(psi))[:, None]
    px, py, pz, E = (B[k].astype(np.float64) for k in ('px', 'py', 'pz', 'energy'))
    pt = np.hypot(px, py); mass = np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))
    deta = np.where(m, B['deta'] + ce, 0); dphi = np.where(m, B['dphi'] + cp, 0)
    sgn = np.where(B['jet_eta'] < 0, -1.0, 1.0)[:, None]
    eta = B['jet_eta'][:, None] + sgn * deta; phi = B['jet_phi'][:, None] + dphi
    O = dict(B)
    O.update(px=np.where(m, pt * np.cos(phi), 0), py=np.where(m, pt * np.sin(phi), 0), pz=np.where(m, pt * np.sinh(eta), 0),
             energy=np.where(m, np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2), 0), deta=deta, dphi=dphi)
    for k in ('px', 'py', 'pz', 'energy', 'deta', 'dphi'): O[k] = O[k].astype(np.float32)
    O['jet_pt'] = np.hypot(O['px'].sum(1), O['py'].sum(1)).astype(np.float32); O['jet_energy'] = O['energy'].sum(1).astype(np.float32)
    return O


class AugBatcher(Batcher):
    """P14 Batcher; arm u: each batch's jets offset (rng keyed on (seed, epoch, batch index)) before the input features"""

    def __init__(self, *a, aug=False, key=(0,), **k):
        self.aug, self.key = aug, key
        super().__init__(*a, **k)

    def _loop(self, k, nt):
        from jetdistill.part.network import inputs
        for i in range(k, self.n, nt):
            b = np.sort(self.chunks[i]); B = {kk: a[b] for kk, a in self.J.items() if a.ndim == 2 or kk.startswith('jet_')}
            if self.aug: B = offset(B, np.random.default_rng(self.key + (i,)))
            x, v, m = inputs(B, 'kin'); x = feed(x, self.arm)
            with self.lock:
                while i - self._next > 6: self.lock.wait()
                self.res[i] = (x, v, m, self.y[b].astype(np.int64)); self.lock.notify_all()


def build_model(device, ncls):
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import CFG, FEATURES
    cfg = dict(CFG); cfg['num_classes'] = ncls
    return ParticleTransformer(input_dim=len(FEATURES['kin']), **cfg).to(device)


def evaluate(model, J, idx, y, torch, dev, ncls, bs=512):
    model.eval(); C = np.zeros((ncls, ncls), int)
    with torch.no_grad():
        for x, v, m, t in AugBatcher(J, idx, y, bs, 'kin', shuffle=False):
            with torch.cuda.amp.autocast(enabled=dev == 'cuda'):
                L = model(*(torch.from_numpy(a).to(dev) for a in (x, v, m)))
            np.add.at(C, (t, L.argmax(1).cpu().numpy()), 1)
    model.train(); return float(np.trace(C) / max(C.sum(), 1)), C.tolist()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True, choices=list(ARMS)); ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--bs', type=int, default=512); a = ap.parse_args()
    import torch
    from weaver.utils.nn.optimizer.ranger import Ranger
    S = ARMS[a.arm]; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log(f'P22 arm {a.arm} {S}; device {dev}; {gpu_util()}; torch {torch.__version__}')
    P, src = load_pool(1 if a.smoke else 5, a.smoke); T, tsrc = load_test(3000 if a.smoke else 20000)
    log(f'pool {len(P["mask"])} jets, test {len(T["mask"])} jets; memory {sum(v.nbytes for v in P.values()) / 1e9:.2f} GB')
    if S['label'] == 'true':
        y = P['cls'].astype(np.int64); yt = T['cls'].astype(np.int64)
    else:
        L = np.load(SIDE / ('labels_smoke.npz' if a.smoke else 'labels.npz'))
        assert np.array_equal(L['pool_row'], P['row']) and np.array_equal(L['pool_cls'], P['cls']) and np.array_equal(L['test_row'], T['row']), 'side file does not match the pool'
        y = L['pool_a'].astype(np.int64); yt = L['test_a'].astype(np.int64)
    epochs = 1 if a.smoke else a.epochs; seed = S['seed']; bs = a.bs
    out = MOD / (a.arm + ('_smoke' if a.smoke else '')); out.mkdir(parents=True, exist_ok=True)
    N = len(y); val = np.sort(np.random.default_rng(140).choice(N, min(30000, N // 10), replace=False)); tr = np.setdiff1d(np.arange(N), val)
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(dev, S['ncls']); opt = Ranger(model.parameters(), lr=1e-3)
    nd = max(1, int(epochs * 0.3)); ms = list(range(epochs - nd, epochs))
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=0.01 ** (1. / nd))
    scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda'); lossf = torch.nn.CrossEntropyLoss()
    rec = dict(arm=a.arm, spec=S, cmax=CMAX if S['aug'] else None, epochs=epochs, batch=bs, n_train=len(tr), n_val=len(val), n_test=len(yt),
               class_rate_train=np.bincount(y[tr], minlength=S['ncls']).tolist(), pool_files=src, test_files=tsrc, epoch=[])
    log(f'arm {a.arm}: {len(tr)} train, {len(val)} val, {len(yt)} test; label counts {rec["class_rate_train"]}; milestones {ms}')
    if S['aug']:
        B = offset({k: v[tr[:2000]] for k, v in P.items() if v.ndim == 2 or k.startswith('jet_')}, np.random.default_rng(0))
        from interp.p14_planted_train import masses
        P0 = {k: v[tr[:2000]] for k, v in P.items()}; mi0, mA0 = masses(P0); mi1, mA1 = masses(B)
        rec['aug_check'] = dict(max_rel_dminv=float(np.max(np.abs(mi1 - mi0) / np.maximum(mi0, 1))), mean_dlogmA=float(np.mean(np.log(np.maximum(mA1, 1e-6) / np.maximum(mA0, 1e-6)))))
        log(f'augmentation check on 2,000 pool jets: max |dm_inv|/m_inv {rec["aug_check"]["max_rel_dminv"]:.2e}; mean dlog m_A {rec["aug_check"]["mean_dlogmA"]:+.4f}')
    model.train()
    for ep in range(epochs):
        t1 = time.time(); tl, ta, tn = 0.0, 0, 0; gu = []; bseed = 1000 + ep if seed == 14 else seed * 1000 + ep
        for i, (x, v, m, t) in enumerate(AugBatcher(P, tr, y, bs, 'kin', seed=bseed, aug=S['aug'], key=(seed, ep))):
            if a.smoke and i >= 60: break
            xt, vt, mt, tt = (torch.from_numpy(z).to(dev, non_blocking=True) for z in (x, v, m, t))
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=dev == 'cuda'):
                L_ = model(xt, vt, mt); loss = lossf(L_, tt)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss) * len(t); ta += int((L_.argmax(1) == tt).sum()); tn += len(t)
            if i % 200 == 0:
                g = gpu_util(); gu.append(g)
                log(f'  {a.arm} ep {ep} step {i}: loss {tl / tn:.4f} acc {ta / tn:.4f} lr {opt.param_groups[0]["lr"]:.2e} '
                    f'{(i + 1) / (time.time() - t1):.2f} steps/s; gpu {g}; torch max alloc {torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0:.2f} GiB')
        sch.step()
        va, _ = evaluate(model, P, val, y, torch, dev, S['ncls'], bs); te, C = evaluate(model, T, np.arange(len(yt)), yt, torch, dev, S['ncls'], bs)
        torch.save(model.state_dict(), out / f'epoch-{ep}.pt')
        e = dict(epoch=ep, train_loss=tl / tn, train_acc=ta / tn, val_acc=va, test_acc=te, test_confusion=C, sec=time.time() - t1, steps=i + 1, gpu=gu)
        rec['epoch'].append(e); json.dump(rec, open(out / 'log.json', 'w'), indent=1)
        log(f'arm {a.arm} EPOCH {ep}: train loss {e["train_loss"]:.4f} acc {e["train_acc"]:.4f}; val acc {va:.4f}; held-out test acc {te:.4f}; confusion {C}; {e["sec"]:.0f} s')
    torch.save(model.state_dict(), out / 'final.pt'); rec['final_test_acc'] = rec['epoch'][-1]['test_acc']
    json.dump(rec, open(out / 'log.json', 'w'), indent=1); log(f'arm {a.arm} DONE -> {out}/final.pt')


if __name__ == '__main__':
    main()
