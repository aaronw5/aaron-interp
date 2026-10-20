"""P31 — what enlarges ParT's axis response in the released training: 10 classes vs 3, and more optimisation steps (claims C10,
C3s; takes up proposed P28). Training half; analysis in interp/p31_classes_steps_part.py.

P22 / P22v: ParT_kin trained on true 3-class W/Z/QCD labels (1.5M jets, 10 epochs, the P14 code path) gives R 0.18-0.27 (phi + 0.06)
and 0.41-0.49 (phi - 0.06); the released ParT_kin gives 0.72-0.86 in every direction. R = read-out change under a rigid shift of
every particle about the stored axis (invariant mass fixed) / change under a matched real dilation. P31 changes one factor at a
time on the P22 code path (interp/p14_planted_train.py helpers + the P22 AugBatcher, ParT kin inputs, weaver ParticleTransformer
with the authors' configuration, Ranger lr 1e-3, batch 512, AMP, flat + decay over the last 30 % of epochs, torch seed 14,
batch order default_rng(1000 + epoch), 30k validation rows default_rng(140) of the arm's own pool, the same 20k held-out
test_20M *_100 W/Z/QCD rows as P14 / P22 (default_rng(14))):
  m10   10 true classes (labels = released output order config.CLASSES: QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl);
        the same 1.5M W/Z/QCD jets as P22 arm t (load_pool: 5 files per class, default_rng(14)) plus 5 random train_100M files of
        each other class (default_rng(31).choice(100, 5) per class in CLASSES order) = 500k jets per class, 5.0M jets, 10 epochs
        (3.3x the steps of arm t)
  m10s  10 true classes at 150k jets per class = 1.5M jets (same step count as arm t): W/Z/QCD rows = 150k random rows
        (default_rng(31)) of each class's 500k arm-t pool rows; other classes = 150k random rows of each class's m10 files.
        Separates class count from data volume and step count.
  L     arm t (3 classes QCD=0, Z=1, W=2, the same 1.5M jets) for 50 epochs (5x the steps), flat then decay over the last 15
        epochs (lr x 0.01^(1/15) per epoch); checkpoint every epoch.
Accuracy logged per epoch: validation (own labels), held-out W/Z/QCD 20k (3-class arms: 3-class; 10-class arms: 10-class argmax
and argmax restricted to QCD/Z/W, with confusion), and for 10-class arms a 10-class held-out sample of 2,000 random rows per
class of test_20M *_100 (default_rng(31)).
Pool storage: preallocated (N, 128) float32 arrays filled file by file (m10: 5.0M jets, ~18 GB), so peak memory = pool + one file.
Outputs: /j-jepa-vol/aaron-interp/models/p31/<arm>/epoch-<e>.pt, final.pt, log.json; one log line every 200 steps.

  P10_MEM=32 interp/p10_part_seeds_job.sh c-p31-m10 'python -m interp.p31_classes_steps_train --arm m10'
  python -m interp.p31_classes_steps_train --arm m10s --smoke      (1 file per class, strided 7k jets each, 60 steps)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.p14_planted_train import V, JC, CLS3, KEYS, read, load_test, gpu_util, log
from interp.p22_planted_true_train import AugBatcher, build_model, evaluate

MOD = V / 'models' / 'p31'
CLASSES = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']      # = jetdistill.config.CLASSES (part)
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
ARMS = dict(m10=dict(ncls=10, per=None, epochs=10), m10s=dict(ncls=10, per=150000, epochs=10), L=dict(ncls=3, per=None, epochs=50))


def plan(arm, smoke):
    """list of (file, class name) for the arm's pool. W/Z/QCD files: P14/P22 choice (default_rng(14), CLS3 order, 5 per class);
    other classes: default_rng(31).choice(100, 5) per class in CLASSES order."""
    nf = 1 if smoke else 5; out = []
    r14 = np.random.default_rng(14)
    for c in CLS3:
        for i in sorted(r14.choice(100, nf, replace=False).tolist()): out.append((JC / 'train_100M' / f'{c}_{i:03d}.root', c))
    if ARMS[arm]['ncls'] == 10:
        r31 = np.random.default_rng(31)
        for c in CLASSES:
            s = PRE[c]
            if s in CLS3: continue
            for i in sorted(r31.choice(100, nf, replace=False).tolist()): out.append((JC / 'train_100M' / f'{s}_{i:03d}.root', s))
    return out


def load_pool(arm, smoke):
    """preallocated pool; label y in the arm's convention (3-class: QCD 0, Z 1, W 2; 10-class: CLASSES index).
    m10s keeps `per` random rows per class (default_rng(31) over the class's concatenated rows, files in plan order)."""
    import uproot
    F = plan(arm, smoke); per = ARMS[arm]['per']; ncls = ARMS[arm]['ncls']
    ne = [min(uproot.open(f)['tree'].num_entries, 7000) if smoke else uproot.open(f)['tree'].num_entries for f, _ in F]
    keep = [None] * len(F)
    if per is not None:
        rng = np.random.default_rng(31)
        for s in dict.fromkeys(s for _, s in F):
            ii = [k for k, (_, s2) in enumerate(F) if s2 == s]; off = np.cumsum([0] + [ne[k] for k in ii])
            pc = min(per, int(off[-1]) // 2) if smoke else per
            g = np.sort(rng.choice(int(off[-1]), pc, replace=False))
            for j, k in enumerate(ii): keep[k] = g[(g >= off[j]) & (g < off[j + 1])] - off[j]
    nk = [ne[k] if keep[k] is None else len(keep[k]) for k in range(len(F))]; N = int(sum(nk))
    P = {k: np.zeros((N, 128), np.float32) for k in KEYS + ['mask']}
    for k in ('jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'): P[k] = np.zeros(N, np.float32)
    P['y'] = np.zeros(N, np.int8); P['file'] = np.zeros(N, np.int16); P['row'] = np.zeros(N, np.int32)
    o = 0; src = []
    for k, (f, s) in enumerate(F):
        J = read(f)
        rows = np.arange(len(J['mask']))
        if smoke: rows = rows[::14][:7000]
        if keep[k] is not None: rows = rows[keep[k]]
        n = len(rows)
        for kk in J: P[kk][o:o + n] = J[kk][rows]
        cname = [c for c, v in PRE.items() if v == s][0]
        P['y'][o:o + n] = CLS3.index(s) if ncls == 3 else CLASSES.index(cname)
        P['file'][o:o + n] = k; P['row'][o:o + n] = rows; o += n; src.append(str(f)); del J
        log(f'read {f.name}: kept {n} jets (class {cname}); pool filled {o}/{N}')
    assert o == N
    return P, src


def load_test10(per, seed=31):
    """10-class held-out sample: `per` random rows (default_rng(seed)) of each class's test_20M *_100 file; y = CLASSES index"""
    import uproot
    rng = np.random.default_rng(seed); Js = []
    for c in CLASSES:
        f = JC / 'test_20M' / f'{PRE[c]}_100.root'; n = uproot.open(f)['tree'].num_entries
        r = np.sort(rng.choice(n, per, replace=False)); J = read(f, r); J['y'] = np.full(per, CLASSES.index(c), np.int8); Js.append(J)
    return {k: np.concatenate([J[k] for J in Js]) for k in Js[0]}


def evaluate3of10(model, T, yt3, torch, dev, bs=512):
    """10-output model on the W/Z/QCD held-out rows: 10-class argmax accuracy and accuracy of argmax over (QCD, Z, W)"""
    model.eval(); cols = [CLASSES.index('QCD'), CLASSES.index('Zqq'), CLASSES.index('Wqq')]; y10 = np.array(cols)[yt3]
    a10 = a3 = 0; C = np.zeros((3, 3), int)
    with torch.no_grad():
        for x, v, m, t in AugBatcher(T, np.arange(len(yt3)), yt3, bs, 'kin', shuffle=False):
            with torch.cuda.amp.autocast(enabled=dev == 'cuda'):
                L = model(*(torch.from_numpy(z).to(dev) for z in (x, v, m))).float().cpu().numpy()
            a10 += int((L.argmax(1) == np.array(cols)[t]).sum()); c3 = L[:, cols].argmax(1); a3 += int((c3 == t).sum())
            np.add.at(C, (t, c3), 1)
    model.train(); return a10 / len(yt3), a3 / len(yt3), C.tolist()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True, choices=list(ARMS)); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--bs', type=int, default=512); a = ap.parse_args()
    import torch
    from weaver.utils.nn.optimizer.ranger import Ranger
    S = ARMS[a.arm]; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; seed = 14; bs = a.bs; ncls = S['ncls']
    log(f'P31 arm {a.arm} {S}; device {dev}; {gpu_util()}; torch {torch.__version__}')
    P, src = load_pool(a.arm, a.smoke); T, tsrc = load_test(3000 if a.smoke else 20000); yt = T['cls'].astype(np.int64)
    T10 = load_test10(300 if a.smoke else 2000) if ncls == 10 else None
    y = P['y'].astype(np.int64)
    log(f'pool {len(y)} jets, memory {sum(v.nbytes for v in P.values()) / 1e9:.2f} GB; class counts {np.bincount(y, minlength=ncls).tolist()}; test {len(yt)}'
        + (f'; 10-class test {len(T10["y"])}' if T10 is not None else ''))
    epochs = 1 if a.smoke else S['epochs']
    out = MOD / (a.arm + ('_smoke' if a.smoke else '')); out.mkdir(parents=True, exist_ok=True)
    N = len(y); val = np.sort(np.random.default_rng(140).choice(N, min(30000, N // 10), replace=False)); tr = np.setdiff1d(np.arange(N), val)
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(dev, ncls); opt = Ranger(model.parameters(), lr=1e-3)
    nd = max(1, int(epochs * 0.3)); ms = list(range(epochs - nd, epochs))
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=0.01 ** (1. / nd))
    scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda'); lossf = torch.nn.CrossEntropyLoss()
    rec = dict(arm=a.arm, spec=S, classes=CLASSES if ncls == 10 else ['QCD', 'Zqq', 'Wqq'], epochs=epochs, batch=bs, seed=seed, n_train=len(tr),
               n_val=len(val), n_test=len(yt), class_count_train=np.bincount(y[tr], minlength=ncls).tolist(), milestones=ms,
               pool_files=src, test_files=tsrc, epoch=[])
    log(f'arm {a.arm}: {len(tr)} train, {len(val)} val; label counts {rec["class_count_train"]}; {len(tr) // bs + 1} steps/epoch; milestones {ms}')
    model.train(); step = 0
    for ep in range(epochs):
        t1 = time.time(); tl, ta, tn = 0.0, 0, 0; gu = []
        for i, (x, v, m, t) in enumerate(AugBatcher(P, tr, y, bs, 'kin', seed=1000 + ep)):
            if a.smoke and i >= 60: break
            xt, vt, mt, tt = (torch.from_numpy(z).to(dev, non_blocking=True) for z in (x, v, m, t))
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=dev == 'cuda'):
                L_ = model(xt, vt, mt); loss = lossf(L_, tt)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); step += 1
            tl += float(loss) * len(t); ta += int((L_.argmax(1) == tt).sum()); tn += len(t)
            if i % 200 == 0:
                g = gpu_util(); gu.append(g)
                log(f'  {a.arm} ep {ep} step {i} (total {step}): loss {tl / tn:.4f} acc {ta / tn:.4f} lr {opt.param_groups[0]["lr"]:.2e} '
                    f'{(i + 1) / (time.time() - t1):.2f} steps/s; gpu {g}; torch max alloc {torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0:.2f} GiB')
        sch.step()
        va, _ = evaluate(model, P, val, y, torch, dev, ncls, bs)
        e = dict(epoch=ep, train_loss=tl / tn, train_acc=ta / tn, val_acc=va, sec=None, steps=i + 1, total_steps=step, gpu=gu)
        if ncls == 3:
            e['test_acc3'], e['test_confusion3'] = evaluate(model, T, np.arange(len(yt)), yt, torch, dev, 3, bs)
        else:
            e['test_acc10_on_WZQCD'], e['test_acc3'], e['test_confusion3'] = evaluate3of10(model, T, yt, torch, dev, bs)
            e['test10_acc'], e['test10_confusion'] = evaluate(model, T10, np.arange(len(T10['y'])), T10['y'].astype(np.int64), torch, dev, 10, bs)
        e['sec'] = time.time() - t1
        torch.save(model.state_dict(), out / f'epoch-{ep}.pt')
        rec['epoch'].append(e); json.dump(rec, open(out / 'log.json', 'w'), indent=1)
        log(f'arm {a.arm} EPOCH {ep}: train loss {e["train_loss"]:.4f} acc {e["train_acc"]:.4f}; val acc {va:.4f}; held-out W/Z/QCD 3-class acc {e["test_acc3"]:.4f} '
            f'confusion {e["test_confusion3"]}' + (f'; 10-class held-out acc {e["test10_acc"]:.4f}' if ncls == 10 else '') + f'; {e["sec"]:.0f} s')
    torch.save(model.state_dict(), out / 'final.pt'); rec['final'] = {k: v for k, v in rec['epoch'][-1].items() if k.startswith('test')}
    json.dump(rec, open(out / 'log.json', 'w'), indent=1); log(f'arm {a.arm} DONE -> {out}/final.pt')


if __name__ == '__main__':
    main()
