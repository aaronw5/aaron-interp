"""Train one small tagger on the hls4ml jet dataset and evaluate it on the test archive.

    python train/train_small.py --model tf-1b-d16 --n 32 --seed 0 --data_dir <dir> --out <dir>

Inputs: <data_dir>/train_ptetaphi_f16.npy (N_jets, 150, 3) float16, per jet the particles sorted by pT descending,
columns (pT [GeV], deta, dphi), zero rows = padding; train_meta.h5['label'] in 0..4 = g, q, W, Z, t; the val archive
(val_ptetaphi_f16.npy, val_meta.h5) is the TEST set and is never used for training or model selection.
Normalisation and truncation follow JEDI-linear: x[:, :N] -> (x - shift[N]) / scale[N] from norm_stats.json, per feature;
padded rows are then set to 0 and a mask is passed to the model.
--axis hardest (E18): before normalisation every real particle's (deta, dphi) is re-measured about the hardest particle
(slot 0 -> (0, 0); dphi wrapped to [-pi, pi); padded rows stay 0; pT unchanged), for train, dev and test jets alike; the
default normalisation file is then norm_stats_axhard.json (made by train/axis_stats.py with JEDI's procedure).
Held-out split for early stopping: splits.npz['dev'] (indices into the train archive) when present, else the last 5 %
of the train archive. Training uses the rest of the train archive.

Writes to --out:
    arch.json          architecture args, normalisation, split, training args, parameter count, best epoch
    log.csv            per epoch: train/val loss and accuracy, and the fraction of dead units per activation layer
                       (unit never above 0 pre-activation on 10k held-out jets, valid tokens only)
    ckpt/epoch_<k>.pt  state_dict after epoch k (k = 1..), epoch_0.pt = initialisation
    best.pt            state_dict at the lowest validation loss
    test.json          accuracy, per-class accuracy, AUC (one vs rest), background rejection at 50 % signal efficiency,
                       confusion matrix, on the test archive
    test_probs.npy     softmax outputs on the test archive (N_test, 5) float32
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, n_params, parse_model  # noqa: E402

CLASSES = ["g", "q", "W", "Z", "t"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------- data
def load_archive(data_dir, prefix, n, rows=None):
    """(x float16 (M, n, 3) raw, labels int64 (M,)); rows = optional sorted index array into the archive."""
    x = np.load(os.path.join(data_dir, f"{prefix}_ptetaphi_f16.npy"), mmap_mode="r")
    with h5py.File(os.path.join(data_dir, f"{prefix}_meta.h5"), "r") as f:
        y = f["label"][:].astype(np.int64)
    assert len(x) == len(y), (x.shape, y.shape)
    if rows is None:
        return np.ascontiguousarray(x[:, :n, :]), y
    return np.ascontiguousarray(x[rows, :n, :]), y[rows]


def dev_indices(data_dir, n_total):
    p = os.path.join(data_dir, "splits.npz")
    if os.path.exists(p):
        z = np.load(p)
        if "dev" in z.files:
            idx = np.asarray(z["dev"], dtype=np.int64)
            return np.sort(idx), "splits.npz:dev"
    k = int(round(0.05 * n_total))
    return np.arange(n_total - k, n_total, dtype=np.int64), "last 5 % of the train archive"


def recentre_hardest(raw, mask):
    """(B, N, 3) raw (pt, deta, dphi), float32 -> same with (deta, dphi) measured about slot 0 (the hardest particle).
    dphi differences are wrapped to [-pi, pi); padded rows (mask False) stay 0; pt unchanged. Same formula as axis_stats.py."""
    d = raw[..., 1:3] - raw[:, :1, 1:3]
    d = torch.stack([d[..., 0], torch.remainder(d[..., 1] + math.pi, 2 * math.pi) - math.pi], dim=-1)
    d = d * mask.unsqueeze(-1).to(d.dtype)
    return torch.cat([raw[..., :1], d], dim=-1)


class GPUData:
    """Whole (truncated) archive on the device as float16; batches are (re-centred and) normalised on the fly."""

    def __init__(self, x_raw, y, shift, scale, device, axis="stored"):
        self.axis = axis
        self.x = torch.from_numpy(x_raw).to(device)  # (M, n, 3) float16
        self.y = torch.from_numpy(y).to(device)
        self.shift = torch.tensor(shift, dtype=torch.float32, device=device)
        self.scale = torch.tensor(scale, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.y)

    def batch(self, idx):
        raw = self.raw(idx)
        mask = raw[..., 0] > 0
        x = (raw - self.shift) / self.scale
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x, mask, self.y[idx]

    def raw(self, idx):
        """Un-normalised (pt, deta, dphi) float32 in the chosen axis convention."""
        raw = self.x[idx].float()
        if self.axis == "hardest":
            raw = recentre_hardest(raw, raw[..., 0] > 0)
        return raw


# ----------------------------------------------------------------------------------------------------------- eval
@torch.no_grad()
def predict(model, data, batch=4096):
    model.eval()
    probs, loss_sum = [], 0.0
    for i in range(0, len(data), batch):
        idx = torch.arange(i, min(i + batch, len(data)), device=data.y.device)
        x, m, y = data.batch(idx)
        logits = model(x, m)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        probs.append(F.softmax(logits, dim=-1).float().cpu())
    probs = torch.cat(probs).numpy()
    y = data.y.cpu().numpy()
    acc = float((probs.argmax(1) == y).mean())
    return probs, y, loss_sum / len(data), acc


def activation_layers(model):
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, (nn.ReLU, nn.GELU))]


@torch.no_grad()
def dead_units(model, data, rows, batch=2048):
    """Per activation layer (ReLU/GELU module): fraction of units whose pre-activation never exceeds 0 over the jets in
    `rows` (per-token layers: valid tokens only). Returns {layer_name: fraction}."""
    model.eval()
    mx, cur = {}, {}

    def make_hook(name):
        def hook(mod, inp):
            z = inp[0]
            if z.dim() == 3:  # (B, T, d) token-level: keep valid tokens
                m = cur["mask"]
                if m.shape[1] != z.shape[1]:  # CLS token prepended
                    m = torch.cat([torch.ones_like(m[:, :1]), m], dim=1)
                z = z[m]
            v = z.max(0).values
            mx[name] = torch.maximum(mx[name], v) if name in mx else v
        return hook

    hooks = [mod.register_forward_pre_hook(make_hook(name)) for name, mod in activation_layers(model)]
    for i in range(0, len(rows), batch):
        x, m, _ = data.batch(rows[i:i + batch])
        cur["mask"] = m
        model(x, m)
    for h in hooks:
        h.remove()
    return {name: float((v <= 0).float().mean()) for name, v in mx.items()}


def roc_metrics(probs, y):
    """Per class (one vs rest, score = softmax prob of the class): AUC and background rejection 1/FPR at TPR = 0.5."""
    from sklearn.metrics import roc_auc_score, roc_curve
    auc, rej50 = [], []
    for c in range(probs.shape[1]):
        sig = (y == c).astype(np.int64)
        s = probs[:, c]
        auc.append(float(roc_auc_score(sig, s)))
        fpr, tpr, _ = roc_curve(sig, s)
        f = float(np.interp(0.5, tpr, fpr))
        rej50.append(float(1.0 / f) if f > 0 else float("inf"))
    return auc, rej50


def test_report(probs, y):
    pred = probs.argmax(1)
    conf = np.zeros((5, 5), dtype=np.int64)
    np.add.at(conf, (y, pred), 1)
    auc, rej50 = roc_metrics(probs, y)
    return dict(
        n=int(len(y)),
        accuracy=float((pred == y).mean()),
        per_class_accuracy={CLASSES[c]: float(conf[c, c] / max(conf[c].sum(), 1)) for c in range(5)},
        auc={CLASSES[c]: auc[c] for c in range(5)},
        rej50={CLASSES[c]: rej50[c] for c in range(5)},
        confusion=conf.tolist(),
        classes=CLASSES,
    )


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="tf-{B}b-d{D} | linf-{B}b-d{D}-k{K} | dsets-d{D}")
    ap.add_argument("--n", type=int, required=True, help="particles per jet (truncation and norm_stats key)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cls_token", action="store_true", help="read out from a learned CLS token instead of mean pooling")
    ap.add_argument("--no_final_ln", action="store_true", help="drop the LayerNorm after the last block")
    ap.add_argument("--axis", choices=["stored", "hardest"], default="stored",
                    help="stored: (deta, dphi) as in the archive (jet axis); hardest: re-measured about the hardest particle")
    ap.add_argument("--norm_file", default="", help="default: norm_stats.json (stored) / norm_stats_axhard.json (hardest)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5, help="epochs without a lower val loss before stopping")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--dead_jets", type=int, default=10000, help="held-out jets used for the dead-unit count")
    ap.add_argument("--max_train", type=int, default=0, help="random subset of the training jets (smoke tests)")
    ap.add_argument("--max_test", type=int, default=0, help="random subset of the test jets (smoke tests)")
    ap.add_argument("--l1_phi1", type=float, default=0.0,
                    help="X5 (opt-in, dsets only): add lambda * sum |W| of the first per-particle Linear (phi1 weights, normalised "
                         "inputs) to the loss; after training, phi1 weights with |w| < --zero_thr are set to 0 and the test is rerun "
                         "(best_zeroed.pt, test_zeroed.json)")
    ap.add_argument("--zero_thr", type=float, default=1e-3)
    ap.add_argument("--shuffle_labels", action="store_true",
                    help="X8 (opt-in null): the labels of the training jets are permuted once with np.random.default_rng(seed) "
                         "before training; held-out (dev) and test labels are not permuted")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    log("args", vars(args))
    log("torch", torch.__version__, "device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    # data ------------------------------------------------------------------------------------------------------
    norm_file = args.norm_file or os.path.join(args.data_dir, "norm_stats.json" if args.axis == "stored" else "norm_stats_axhard.json")
    with open(norm_file) as f:
        norm_all = json.load(f)
    ns = norm_all[str(args.n)]
    if args.axis == "hardest":
        assert norm_all.get("_axis") == "hardest", f"{norm_file} was not made for --axis hardest"
    shift, scale = [float(v) for v in ns["shift"]], [float(v) for v in ns["scale"]]
    t0 = time.time()
    n_total = len(np.load(os.path.join(args.data_dir, "train_ptetaphi_f16.npy"), mmap_mode="r"))
    dev_idx, split_src = dev_indices(args.data_dir, n_total)
    is_dev = np.zeros(n_total, dtype=bool)
    is_dev[dev_idx] = True
    tr_idx = np.flatnonzero(~is_dev)
    if args.max_train:
        rng = np.random.RandomState(args.seed)
        tr_idx = np.sort(rng.choice(tr_idx, size=min(args.max_train, len(tr_idx)), replace=False))
    x_tr, y_tr = load_archive(args.data_dir, "train", args.n, rows=tr_idx)
    if args.shuffle_labels:  # X8 null: training labels permuted once (dev / test labels untouched)
        y_tr = y_tr[np.random.default_rng(args.seed).permutation(len(y_tr))]
        log(f"shuffle_labels: {len(y_tr)} training labels permuted with np.random.default_rng({args.seed})")
    train = GPUData(x_tr, y_tr, shift, scale, device, axis=args.axis)
    del x_tr
    x_va, y_va = load_archive(args.data_dir, "train", args.n, rows=dev_idx)
    val = GPUData(x_va, y_va, shift, scale, device, axis=args.axis)
    del x_va
    # fixed random subset of the held-out jets for the dead-unit count (the dev rows are not in random class order)
    dead_rows = np.sort(np.random.RandomState(12345).choice(len(val), size=min(args.dead_jets, len(val)), replace=False))
    dead_rows_t = torch.from_numpy(dead_rows).to(device)
    # input checks on the first 4096 held-out jets: slot 0 at (0, 0) under --axis hardest; padded rows all zero
    chk = val.raw(torch.arange(min(4096, len(val)), device=val.y.device))
    cm = chk[..., 0] > 0
    pad_max = float(chk[~cm].abs().max()) if bool((~cm).any()) else 0.0  # raw padded rows (masked in the model anyway)
    assert bool(cm[:, 0].all()), "a jet without a particle in slot 0"
    if args.axis == "hardest":
        assert float(chk[:, 0, 1:].abs().max()) == 0.0, "slot 0 not at (0, 0) after re-centring"
        assert pad_max == 0.0, "padded rows not zero after re-centring"
    input_check = dict(jets=int(len(chk)), max_abs_deta=float(chk[..., 1][cm].abs().max()),
                       max_abs_dphi=float(chk[..., 2][cm].abs().max()), slot0_max_abs_offset=float(chk[:, 0, 1:].abs().max()),
                       padded_rows_max_abs=pad_max)
    log(f"axis={args.axis} norm_file={norm_file} shift={shift} scale={scale} input check {input_check}")
    del chk, cm
    log(f"data: train {len(train)} val {len(val)} ({split_src}) n={args.n} loaded in {time.time() - t0:.1f}s; "
        f"train class counts {np.bincount(y_tr, minlength=5).tolist()}")

    # model -----------------------------------------------------------------------------------------------------
    spec = parse_model(args.model)
    model = build_model(spec, args.n, cls_token=args.cls_token, final_ln=not args.no_final_ln).to(device)
    run_name = f"{args.model}-n{args.n}" + ("-cls" if args.cls_token else "")
    log(f"model {run_name}: {n_params(model)} parameters\n{model}")
    torch.save(model.state_dict(), os.path.join(args.out, "ckpt", "epoch_0.pt"))
    act_layers = [(name, mod) for name, mod in activation_layers(model)]
    act_widths = {}  # unit count per activation layer = out_features of the Linear feeding it
    for name, _ in act_layers:
        parent, _, idx = name.rpartition(".")
        seq = model.get_submodule(parent) if parent else model
        act_widths[name] = int(seq[int(idx) - 1].out_features) if idx.isdigit() else None

    steps_per_epoch = math.ceil(len(train) / args.batch)
    total_steps = args.epochs * steps_per_epoch
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=True)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_mult(step):
        warm = min(1.0, (step + 1) / max(1, args.warmup_steps))
        return warm * 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)

    arch = dict(
        run_name=run_name, model=args.model, spec=spec, n_particles=args.n, cls_token=args.cls_token,
        final_ln=not args.no_final_ln, n_params=n_params(model), seed=args.seed,
        inputs=dict(features=["pt", "deta", "dphi"], source="train_ptetaphi_f16.npy[:, :N]", axis=args.axis,
                    axis_definition=("stored archive values (relative to the jet axis)" if args.axis == "stored" else
                                     "deta, dphi re-measured about slot 0 (hardest particle): d = x - x[slot 0], dphi "
                                     "wrapped to [-pi, pi), padded rows stay 0, pt unchanged (train, dev and test)"),
                    normalisation="(x - shift) / scale per feature, then padded rows (raw pt == 0) set to 0",
                    norm_file=norm_file, norm_procedure=norm_all.get("_procedure", "JEDI-linear norm_stats.json"),
                    shift=shift, scale=scale, mask="raw pt > 0", input_check=input_check),
        split=dict(val=split_src, n_train=len(train), n_val=len(val), test="val archive",
                   max_train=args.max_train, shuffle_labels=bool(args.shuffle_labels), dead_unit_jets=f"{len(dead_rows)} fixed random rows of val (seed 12345)"),
        training=dict(l1_phi1=args.l1_phi1, zero_thr=args.zero_thr, optimizer="AdamW", lr=args.lr, weight_decay=args.wd, schedule="linear warmup then cosine to 0",
                      warmup_steps=args.warmup_steps, batch=args.batch, epochs=args.epochs, patience=args.patience,
                      label_smoothing=args.label_smoothing, selection="lowest val loss", steps_per_epoch=steps_per_epoch,
                      checkpoints="ckpt/epoch_<k>.pt after epoch k; epoch_0.pt = initialisation"),
        activation_layers=act_widths,
        dead_unit_definition="pre-activation max <= 0 over the dead-unit jets (valid tokens only)",
        torch=torch.__version__, device=str(device),
    )
    with open(os.path.join(args.out, "arch.json"), "w") as f:
        json.dump(arch, f, indent=1)

    # train -----------------------------------------------------------------------------------------------------
    dead_cols = [f"dead_{name}" for name, _ in act_layers]
    csv_path = os.path.join(args.out, "log.csv")

    def write_row(row):
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "time_s", "best"]
                               + dead_cols)
    dead0 = dead_units(model, val, dead_rows_t)
    _, _, va_loss0, va_acc0 = predict(model, val)
    write_row([0, 0, "", "", "", f"{va_loss0:.5f}", f"{va_acc0:.5f}", "", 0] + [f"{dead0[n]:.4f}" for n, _ in act_layers])
    log(f"epoch  0 (init) val {va_loss0:.4f}/{va_acc0:.4f} dead {[round(dead0[n], 3) for n, _ in act_layers]}")

    best = dict(val_loss=float("inf"), epoch=-1, val_acc=0.0)
    bad_epochs, step, epoch = 0, 0, 0
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(train), generator=g).to(device)
        loss_sum, correct = 0.0, 0
        for i in range(0, len(train), args.batch):
            idx = perm[i:i + args.batch]
            x, m, y = train.batch(idx)
            logits = model(x, m)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            if args.l1_phi1 > 0:
                loss = loss + args.l1_phi1 * model.phi[0].weight.abs().sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        tr_loss, tr_acc = loss_sum / len(train), correct / len(train)
        _, _, va_loss, va_acc = predict(model, val)
        dead = dead_units(model, val, dead_rows_t)
        torch.save(model.state_dict(), os.path.join(args.out, "ckpt", f"epoch_{epoch}.pt"))
        improved = va_loss < best["val_loss"]
        if improved:
            best = dict(val_loss=va_loss, epoch=epoch, val_acc=va_acc)
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            bad_epochs = 0
        else:
            bad_epochs += 1
        lr_now = opt.param_groups[0]["lr"]
        write_row([epoch, step, f"{lr_now:.3e}", f"{tr_loss:.5f}", f"{tr_acc:.5f}", f"{va_loss:.5f}", f"{va_acc:.5f}",
                   f"{time.time() - t0:.1f}", int(improved)] + [f"{dead[n]:.4f}" for n, _ in act_layers])
        log(f"epoch {epoch:2d} lr {lr_now:.2e} train {tr_loss:.4f}/{tr_acc:.4f} val {va_loss:.4f}/{va_acc:.4f} "
            f"dead {[round(dead[n], 3) for n, _ in act_layers]} {time.time() - t0:.1f}s{' *' if improved else ''}")
        if bad_epochs >= args.patience:
            log(f"early stop: no lower val loss for {args.patience} epochs")
            break
    arch["best"] = dict(epoch=best["epoch"], val_loss=best["val_loss"], val_acc=best["val_acc"], epochs_run=epoch)
    with open(os.path.join(args.out, "arch.json"), "w") as f:
        json.dump(arch, f, indent=1)

    # test ------------------------------------------------------------------------------------------------------
    del train
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device))
    rows = None
    if args.max_test:
        n_te = len(np.load(os.path.join(args.data_dir, "val_ptetaphi_f16.npy"), mmap_mode="r"))
        rows = np.sort(np.random.RandomState(args.seed + 1).choice(n_te, size=min(args.max_test, n_te), replace=False))
    x_te, y_te = load_archive(args.data_dir, "val", args.n, rows=rows)
    test = GPUData(x_te, y_te, shift, scale, device, axis=args.axis)
    del x_te
    probs, y, te_loss, te_acc = predict(model, test)
    rep = test_report(probs, y)
    rep.update(loss=te_loss, run_name=run_name, seed=args.seed, best_epoch=best["epoch"], val_loss=best["val_loss"],
               val_acc=best["val_acc"], n_params=arch["n_params"], max_test=args.max_test)
    with open(os.path.join(args.out, "test.json"), "w") as f:
        json.dump(rep, f, indent=1)
    np.save(os.path.join(args.out, "test_probs.npy"), probs.astype(np.float32))
    log(f"test ({rep['n']} jets): acc {rep['accuracy']:.4f} loss {te_loss:.4f} "
        f"per-class acc {[round(v, 3) for v in rep['per_class_accuracy'].values()]} "
        f"auc {[round(v, 3) for v in rep['auc'].values()]} rej50 {[round(v, 1) for v in rep['rej50'].values()]}")
    if args.l1_phi1 > 0:  # X5: hard-zero the small phi1 weights and re-evaluate
        with torch.no_grad():
            w = model.phi[0].weight
            small = w.abs() < args.zero_thr
            w[small] = 0.0
        zf = float(small.float().mean())
        torch.save(model.state_dict(), os.path.join(args.out, "best_zeroed.pt"))
        probs_z, y, te_loss_z, te_acc_z = predict(model, test)
        rep_z = test_report(probs_z, y)
        rep_z.update(loss=te_loss_z, run_name=run_name, seed=args.seed, l1_phi1=args.l1_phi1, zero_thr=args.zero_thr,
                     phi1_zero_frac=zf, phi1_zero_per_feature=small.float().mean(0).cpu().tolist(),
                     acc_before_zeroing=rep["accuracy"], max_test=args.max_test)
        with open(os.path.join(args.out, "test_zeroed.json"), "w") as f:
            json.dump(rep_z, f, indent=1)
        np.save(os.path.join(args.out, "test_probs_zeroed.npy"), probs_z.astype(np.float32))
        log(f"zeroed phi1 |w| < {args.zero_thr}: zero fraction {zf:.4f}; test acc {rep['accuracy']:.4f} -> {rep_z['accuracy']:.4f}")
    log("done")


if __name__ == "__main__":
    main()
