"""P14 — planted ground truth for the ParT methods: train ParT (standard architecture, kin inputs, 2 outputs) on W/Z/QCD JetClass
jets with labels that are a known function of one mass variable, so that the P1 / P2 / P8 / P9 tools can be scored against a
known answer (review weaknesses 2 and 8).

Arms (labels precomputed per jet, side file /j-jepa-vol/aaron-interp/data/p14/labels.npz):
  a  1[|m_inv - 85| < 10 GeV], m_inv = invariant mass of the four-vector sum of the stored constituents
  b  1[|m_A - c| < w], m_A = sqrt(sum pT * sum pT dR^2) about the stored jet axis (dR from the stored deta, dphi);
     c = median m_A of the arm-a-positive training jets, w solved (bisection) so that the training positive rate equals arm a's
  c  arm-a labels; per-particle inputs that carry position (deltaR, deta, dphi, and e_log, logerel, which depend on the particle's
     eta through E = pT cosh eta) are set to 0 after scaling, so geometry reaches the network only through the pair features
     built from the four-vectors (ln kT, ln z, ln dR, ln m^2): planted pair route
  s  arm-a labels permuted over the training jets (numpy default_rng(14)): shuffled-label control, 2 epochs
Inputs: the ParT_kin feature set of the authors' JetClass_kin.yaml (jetdistill/part/network.py: pt_log, e_log, logptrel, logerel,
deltaR, deta, dphi; four-vectors for the pair features), weaver 'wrap' padding. Architecture: weaver-core 0.4.17
ParticleTransformer with the authors' example_ParticleTransformer.py configuration (8 particle blocks, 2 class blocks, 128 / 512
dims, 8 heads, pair embed 64x3), num_classes = 2. Deviation from the brief (which said "ParT standard"): the kin input set is used
(the labels are kinematic; the PID/track inputs of ParT_full would not change the question and would double the memory).
Optimiser as train_JetClass.sh for ParT: Ranger (weaver), lr 1e-3, batch 512, AMP, CrossEntropy, flat+decay (lr x 0.01^(1/3)
at each of the last 3 of 10 epochs, per-epoch MultiStepLR as weaver).
Data: JetClass train_100M, 5 random files (default_rng(14).choice(100, 5)) of each of WToQQ, ZToQQ, ZJetsToNuNu = 1.5M jets;
30,000 random pool rows held out for per-epoch validation. Held-out test: 20,000 random rows (default_rng(14)) of the union of
test_20M/{WToQQ,ZToQQ,ZJetsToNuNu}_100.root (the P14 analysis sample); label accuracy checked there (target > 95 %).
Outputs: /j-jepa-vol/aaron-interp/models/p14/<arm>/epoch-<e>.pt, final.pt (state_dict of ParticleTransformer), log.json
(per-epoch train loss / acc, val acc, held-out acc, steps/s, GPU utilisation samples).

  interp/p10_part_seeds_job.sh c-p14-a 'python -m interp.p14_planted_train --arms a,s'
  python -m interp.p14_planted_train --smoke     (1 file per class, 20k jets, 60 steps)
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

V = Path('/j-jepa-vol/aaron-interp'); JC = Path(os.environ.get('P14_JC', '/j-jepa-vol/JetClass/Pythia'))
SIDE = V / 'data' / 'p14'; MOD = V / 'models' / 'p14'
CLS3 = ['ZJetsToNuNu', 'ZToQQ', 'WToQQ']          # class index 0 = QCD, 1 = Z, 2 = W
KEYS = ['px', 'py', 'pz', 'energy', 'deta', 'dphi']
POS_ZERO = ['e_log', 'logerel', 'deltaR', 'deta', 'dphi']   # arm c
M0_, W0_ = 85.0, 10.0
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def read(path, rows=None):
    """jet dict (J, 128) float32 particles sorted by pT (as jetdistill.part.data.read_root), mask, jet_pt, jet_energy, jet_eta, jet_phi"""
    import uproot, awkward as ak
    t = uproot.open(path)['tree']
    a = t.arrays([f'part_{k}' for k in KEYS] + ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'])
    if rows is not None: a = a[rows]
    pt = np.hypot(a['part_px'], a['part_py']); o = ak.argsort(pt, ascending=False, stable=True)
    pad = lambda x: ak.to_numpy(ak.fill_none(ak.pad_none(x[o], 128, clip=True), 0)).astype(np.float32)
    J = {k: pad(a[f'part_{k}']) for k in KEYS}; J['mask'] = pad(ak.ones_like(a['part_px']))
    for k in ('jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'): J[k] = ak.to_numpy(a[k]).astype(np.float32)
    return J


def masses(J):
    """m_inv (four-vector sum of the stored particles) and m_A = sqrt(sum pT * sum pT dR^2) about the stored axis [GeV]"""
    m = J['mask'] > 0; f = lambda k: (J[k].astype(np.float64) * m).sum(1)
    minv = np.sqrt(np.maximum(f('energy') ** 2 - f('px') ** 2 - f('py') ** 2 - f('pz') ** 2, 0))
    pt = np.hypot(J['px'], J['py']).astype(np.float64) * m; r2 = (J['deta'].astype(np.float64) ** 2 + J['dphi'].astype(np.float64) ** 2) * m
    mA = np.sqrt(pt.sum(1) * (pt * r2).sum(1)); return minv, mA


def cat(Js):
    """concatenate jet dicts key by key, freeing each key of the parts as it goes (peak = total + one key)"""
    out = {}
    for k in list(Js[0]): out[k] = np.concatenate([J.pop(k) for J in Js])
    return out


def load_pool(nfile, smoke):
    rng = np.random.default_rng(14); Js, src = [], []
    for ci, c in enumerate(CLS3):
        idx = sorted(rng.choice(100, nfile, replace=False).tolist())
        for i in idx:
            f = JC / 'train_100M' / f'{c}_{i:03d}.root'; J = read(f)
            if smoke: J = {k: v[::14][:7000] for k, v in J.items()}      # strided rows, not first-k
            J['cls'] = np.full(len(J['mask']), ci, np.int8); J['file'] = np.full(len(J['mask']), len(src), np.int16); J['row'] = np.arange(len(J['mask']), dtype=np.int32)
            src.append(str(f)); Js.append(J); log(f'read {f.name}: {len(J["mask"])} jets')
    return cat(Js), src


def load_test(n, seed=14):
    """n random rows (default_rng(seed)) of the union of the three test files _100 -> the P14 analysis sample"""
    import uproot
    fs = [JC / 'test_20M' / f'{c}_100.root' for c in CLS3]; ne = [uproot.open(f)['tree'].num_entries for f in fs]
    rng = np.random.default_rng(seed); g = np.sort(rng.choice(sum(ne), n, replace=False)); off = np.cumsum([0] + ne)
    Js = []
    for ci, f in enumerate(fs):
        r = g[(g >= off[ci]) & (g < off[ci + 1])] - off[ci]; J = read(f, r)
        J['cls'] = np.full(len(r), ci, np.int8); J['file'] = np.full(len(r), ci, np.int16); J['row'] = r.astype(np.int32); Js.append(J)
    return cat(Js), [str(f) for f in fs]


def make_labels(P, T):
    """side file: per-jet m_inv, m_A, labels of arms a / b / s for the training pool and the test sample"""
    mi, mA = masses(P); ti, tA = masses(T)
    la = (np.abs(mi - M0_) < W0_); rate = la.mean(); c = float(np.median(mA[la]))
    lo, hi = 0.0, 200.0
    for _ in range(60):
        w = 0.5 * (lo + hi); r = (np.abs(mA - c) < w).mean(); lo, hi = (w, hi) if r < rate else (lo, w)
    w = 0.5 * (lo + hi); lb = np.abs(mA - c) < w
    ls = np.random.default_rng(14).permutation(la)
    L = dict(pool_minv=mi.astype(np.float32), pool_mA=mA.astype(np.float32), pool_a=la, pool_b=lb, pool_s=ls,
             test_minv=ti.astype(np.float32), test_mA=tA.astype(np.float32), test_a=np.abs(ti - M0_) < W0_, test_b=np.abs(tA - c) < w,
             c=c, w=w, m0=M0_, w0=W0_, rate_a=float(rate), rate_b=float(lb.mean()))
    L['test_s'] = L['test_a']; L['pool_c'] = la; L['test_c'] = L['test_a']
    return L


class Batcher:
    """random batches -> (x, v, mask) numpy via jetdistill.part.network.inputs('kin'), prefetched in 2 threads"""

    def __init__(self, J, idx, y, bs, arm, shuffle=True, seed=0, nthreads=2):
        self.J, self.idx, self.y, self.bs, self.arm = J, np.asarray(idx), y, bs, arm
        o = np.random.default_rng(seed).permutation(self.idx) if shuffle else self.idx
        self.chunks = [o[i:i + bs] for i in range(0, len(o), bs)]; self.n = len(self.chunks); self.err = None
        self.ths = [threading.Thread(target=self._work, args=(k, nthreads), daemon=True) for k in range(nthreads)]
        self.res = {}; self.lock = threading.Condition()
        for t in self.ths: t.start()

    def _work(self, k, nt):
        try:
            self._loop(k, nt)
        except BaseException as ex:          # surface worker errors in the consumer instead of hanging
            with self.lock: self.err = ex; self.lock.notify_all()

    def _loop(self, k, nt):
        from jetdistill.part.network import inputs
        for i in range(k, self.n, nt):
            b = np.sort(self.chunks[i]); x, v, m = inputs({kk: a[b] for kk, a in self.J.items() if a.ndim == 2 or kk.startswith('jet_')}, 'kin')
            x = feed(x, self.arm)
            with self.lock:
                while i - self._next > 6: self.lock.wait()
                self.res[i] = (x, v, m, self.y[b].astype(np.int64)); self.lock.notify_all()

    _next = 0

    def __iter__(self):
        for i in range(self.n):
            with self.lock:
                while i not in self.res:
                    if self.err is not None: raise RuntimeError(f'batch worker failed: {self.err!r}')
                    self.lock.wait(timeout=60)
                r = self.res.pop(i); self._next = i + 1; self.lock.notify_all()
            yield r


def feed(x, arm):
    """arm c: position-carrying per-particle channels set to 0 (after scaling)"""
    if arm == 'c':
        from jetdistill.part.network import FEATURES
        names = [f[0] for f in FEATURES['kin']]; x = x.copy()
        for k in POS_ZERO: x[:, names.index(k)] = 0
    return x


def build_model(device):
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import CFG, FEATURES
    cfg = dict(CFG); cfg['num_classes'] = 2
    return ParticleTransformer(input_dim=len(FEATURES['kin']), **cfg).to(device)


def evaluate(model, J, idx, y, arm, torch, dev, bs=512):
    model.eval(); ok = 0; n = 0
    with torch.no_grad():
        for x, v, m, t in Batcher(J, idx, y, bs, arm, shuffle=False):
            with torch.cuda.amp.autocast(enabled=dev == 'cuda'):
                L = model(*(torch.from_numpy(a).to(dev) for a in (x, v, m)))
            ok += int((L.argmax(1).cpu().numpy() == t).sum()); n += len(t)
    model.train(); return ok / max(n, 1)


def gpu_util():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as ex:
        return repr(ex)


def train_arm(arm, P, T, L, epochs, smoke, torch, dev, bs=512):
    from weaver.utils.nn.optimizer.ranger import Ranger
    out = MOD / (arm + ('_smoke' if smoke else '')); out.mkdir(parents=True, exist_ok=True)
    y = L[f'pool_{arm}'].astype(np.int64); yt = L[f'test_{arm}'].astype(np.int64)
    N = len(y); rng = np.random.default_rng(140); val = np.sort(rng.choice(N, min(30000, N // 10), replace=False))
    tr = np.setdiff1d(np.arange(N), val)
    torch.manual_seed(14); np.random.seed(14)
    model = build_model(dev); opt = Ranger(model.parameters(), lr=1e-3)
    nd = max(1, int(epochs * 0.3)); ms = list(range(epochs - nd, epochs))
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=0.01 ** (1. / nd))
    scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda'); lossf = torch.nn.CrossEntropyLoss()
    rec = dict(arm=arm, epochs=epochs, batch=bs, n_train=len(tr), n_val=len(val), n_test=len(yt), pos_rate_train=float(y[tr].mean()),
               pos_rate_test=float(yt.mean()), c=L['c'], w=L['w'], epoch=[])
    log(f'arm {arm}: {len(tr)} train, {len(val)} val, {len(yt)} test; positive rate {y[tr].mean():.3f} (test {yt.mean():.3f}); milestones {ms}')
    model.train(); step = 0
    for ep in range(epochs):
        t1 = time.time(); tl, ta, tn = 0.0, 0, 0; gu = []
        maxstep = 60 if smoke else None
        for i, (x, v, m, t) in enumerate(Batcher(P, tr, y, bs, arm, seed=1000 + ep)):
            if maxstep and i >= maxstep: break
            xt, vt, mt, tt = (torch.from_numpy(a).to(dev, non_blocking=True) for a in (x, v, m, t))
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=dev == 'cuda'):
                L_ = model(xt, vt, mt); loss = lossf(L_, tt)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step += 1
            tl += float(loss) * len(t); ta += int((L_.argmax(1) == tt).sum()); tn += len(t)
            if i % 200 == 0:
                g = gpu_util(); gu.append(g)
                log(f'  {arm} ep {ep} step {i}: loss {tl / tn:.4f} acc {ta / tn:.4f} lr {opt.param_groups[0]["lr"]:.2e} '
                    f'{(i + 1) / (time.time() - t1):.2f} steps/s; gpu {g}; torch max alloc {torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0:.2f} GiB')
        sch.step()
        va = evaluate(model, P, val, y, arm, torch, dev, bs=bs); te = evaluate(model, T, np.arange(len(yt)), yt, arm, torch, dev, bs=bs)
        torch.save(model.state_dict(), out / f'epoch-{ep}.pt')
        e = dict(epoch=ep, train_loss=tl / tn, train_acc=ta / tn, val_acc=va, test_acc=te, sec=time.time() - t1, steps=i + 1, gpu=gu)
        rec['epoch'].append(e); json.dump(rec, open(out / 'log.json', 'w'), indent=1)
        log(f'arm {arm} EPOCH {ep}: train loss {e["train_loss"]:.4f} acc {e["train_acc"]:.4f}; val acc {va:.4f}; held-out test acc {te:.4f}; {e["sec"]:.0f} s')
    torch.save(model.state_dict(), out / 'final.pt'); rec['final_test_acc'] = rec['epoch'][-1]['test_acc']
    json.dump(rec, open(out / 'log.json', 'w'), indent=1); log(f'arm {arm} DONE -> {out}/final.pt')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arms', default='a'); ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--s-epochs', type=int, default=2); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--bs', type=int, default=512); a = ap.parse_args()
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log(f'device {dev}; {gpu_util()}; torch {torch.__version__}')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    P, src = load_pool(1 if a.smoke else 5, a.smoke); T, tsrc = load_test(3000 if a.smoke else 20000)
    log(f'pool {len(P["mask"])} jets, test {len(T["mask"])} jets; memory {sum(v.nbytes for v in P.values()) / 1e9:.2f} GB')
    L = make_labels(P, T); L['pool_cls'] = P['cls']; L['pool_file'] = P['file']; L['pool_row'] = P['row']
    L['test_cls'] = T['cls']; L['test_file'] = T['file']; L['test_row'] = T['row']
    SIDE.mkdir(parents=True, exist_ok=True); fn = SIDE / ('labels_smoke.npz' if a.smoke else 'labels.npz')
    tmp = SIDE / f'.{fn.name}.{os.getpid()}.npz'; np.savez(tmp, pool_files=np.array(src), test_files=np.array(tsrc), **L); os.replace(tmp, fn)
    for k in ('a', 'b'):
        lab = L[f'pool_{k}']
        log(f'labels {k}: positive rate {lab.mean():.4f}; by class QCD/Z/W ' + ' / '.join(f'{lab[P["cls"] == c].mean():.3f}' for c in range(3)))
    log(f'arm b window c {L["c"]:.2f} GeV, w {L["w"]:.2f} GeV; a/b label agreement {np.mean(L["pool_a"] == L["pool_b"]):.4f}; '
        f'corr(m_inv, m_A) {np.corrcoef(L["pool_minv"], L["pool_mA"])[0, 1]:.3f}; side file {fn}')
    for arm in a.arms.split(','):
        train_arm(arm, P, T, L, 1 if a.smoke else (a.s_epochs if arm == 's' else a.epochs), a.smoke, torch, dev, a.bs)
    log('P14 train DONE')


if __name__ == '__main__':
    main()
