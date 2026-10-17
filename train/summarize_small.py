"""Summarise the small-tagger runs.

    collect (on the cluster, stdlib only):
        python3 train/summarize_small.py collect --models_dir /j-jepa-vol/aaron-interp/models/small --out summary_small.json
    render (matplotlib):
        python train/summarize_small.py render summary_small.json --png results/site/T7_small_models.png --md table.md

collect: one record per run directory <model>-n<N><tag>/seed<k>/ that has test.json (arch.json + log.csv for the training
facts). render: per-run Markdown table, per-(model, N, tag) mean ± spread over seeds (spread = half the difference between the
two seeds, i.e. half the range), and a PNG table in the style of site/tables.py.
"""
import argparse
import csv
import json
import os
import re
import sys

ORDER = ["tf-1b-d16", "tf-2b-d16", "tf-1b-d32", "tf-2b-d32", "tf-2b-d64", "linf-1b-d32-k4", "linf-2b-d32-k8", "dsets-d64"]
CLASSES = ["g", "q", "W", "Z", "t"]


def collect(models_dir):
    runs = []
    for d in sorted(os.listdir(models_dir)):
        m = re.match(r"^(.+)-n(\d+)(-[a-z0-9]+)?$", d)
        if not m or d.startswith("_"):
            continue
        model, n, tag = m.group(1), int(m.group(2)), m.group(3) or ""
        for sd in sorted(os.listdir(os.path.join(models_dir, d))):
            p = os.path.join(models_dir, d, sd)
            if not (sd.startswith("seed") and os.path.exists(os.path.join(p, "test.json"))):
                continue
            t = json.load(open(os.path.join(p, "test.json")))
            a = json.load(open(os.path.join(p, "arch.json")))
            rows = list(csv.DictReader(open(os.path.join(p, "log.csv"))))
            tr = a["training"]
            b = a.get("best", {})
            runs.append(dict(
                dir=f"{d}/{sd}", model=model, n=n, tag=tag, seed=int(sd[4:]), n_params=a["n_params"],
                epochs=tr["epochs"], patience=tr["patience"], best_epoch=b.get("epoch"), epochs_run=b.get("epochs_run"),
                early_stopped=bool(b.get("epochs_run", tr["epochs"]) < tr["epochs"]),
                val_loss=b.get("val_loss"), val_acc=b.get("val_acc"), test_acc=t["accuracy"], test_loss=t.get("loss"),
                per_class_acc=t["per_class_accuracy"], auc=t["auc"], rej50=t["rej50"],
                n_ckpt=len([f for f in os.listdir(os.path.join(p, "ckpt")) if f.startswith("epoch_")]),
                last_val_acc=float(rows[-1]["val_acc"]) if rows else None,
            ))
    return runs


def groups(runs):
    """{(model, n, tag): [runs sorted by seed]} in display order."""
    g = {}
    for r in runs:
        g.setdefault((r["model"], r["n"], r["tag"]), []).append(r)
    key = lambda k: (k[1], k[2] != "", ORDER.index(k[0]) if k[0] in ORDER else 99, k[2])  # by N, base runs first
    return {k: sorted(g[k], key=lambda r: r["seed"]) for k in sorted(g, key=key)}


def mean_spread(vals):
    m = sum(vals) / len(vals)
    return m, (max(vals) - min(vals)) / 2


def pct(x):
    return f"{100 * x:.2f}"


def render(summary, png, md, ref):
    runs = summary["runs"]
    G = groups(runs)
    lines = []
    lines.append("| run | params | epochs run / best epoch | val acc | test acc | test AUC g/q/W/Z/t | rej50 g/q/W/Z/t |")
    lines.append("|---|---|---|---|---|---|---|")
    for (model, n, tag), rs in G.items():
        for r in rs:
            es = " (early stop)" if r["early_stopped"] else ""
            lines.append(f"| {model}-n{n}{tag} s{r['seed']} | {r['n_params']:,} | {r['epochs_run']}/{r['best_epoch']}{es} | "
                         f"{pct(r['val_acc'])} | {pct(r['test_acc'])} | "
                         + "/".join(f"{r['auc'][c]:.3f}" for c in CLASSES) + " | "
                         + "/".join(f"{r['rej50'][c]:.0f}" for c in CLASSES) + " |")
    per_run = "\n".join(lines)

    lines = ["| model | N | variant | params | test acc seed 0 / seed 1 [%] | mean ± spread [%] | AUC W / Z / t (mean) |",
             "|---|---|---|---|---|---|---|"]
    rows_png = []
    keys = list(G)
    for i, (model, n, tag) in enumerate(keys):
        rs = G[(model, n, tag)]
        accs = [r["test_acc"] for r in rs]
        m, s = mean_spread(accs)
        var = {"": "30 ep, patience 5", "-p30": "30 ep, patience 30", "-e100": "100 ep, no early stop"}.get(tag, tag)
        by_seed = {r["seed"]: f"{pct(r['test_acc'])}{'*' if r['early_stopped'] else ''}" for r in rs}
        per_seed = " / ".join(by_seed.get(k, "–") for k in (0, 1))  # value under its own seed; – = not run
        aucs = " / ".join(f"{sum(r['auc'][c] for r in rs) / len(rs):.3f}" for c in ("W", "Z", "t"))
        ms = f"{100 * m:.2f} ± {100 * s:.2f}" if len(rs) > 1 else f"{100 * m:.2f}"
        lines.append(f"| {model} | {n} | {var} | {rs[0]['n_params']:,} | {per_seed} | {ms} | {aucs} |")
        rows_png.append([model, str(n), var, f"{rs[0]['n_params']:,}", per_seed, ms, aucs])
        last_of_n = i + 1 == len(keys) or keys[i + 1][1] != n
        if last_of_n and str(n) in ref:  # reference row directly under its N block
            lines.append(f"| JEDI-linear 3-feature (official) | {n} | reference | – | – | {100 * ref[str(n)]:.2f} | – |")
            rows_png.append(["JEDI-linear 3-feature (official)", str(n), "reference", "–", "–", f"{100 * ref[str(n)]:.2f}", "–"])
    agg = "\n".join(lines)
    if md:
        with open(md, "w") as f:
            f.write("## Per run\n\n" + per_run + "\n\n## Mean ± spread over seeds\n\n" + agg + "\n")
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = ["model", "N", "training", "params", "test acc, seed 0 / seed 1 [%]", "mean ± spread [%]",
                "AUC W / Z / t\n(mean over seeds)"]
        widths = [2.6, 0.5, 1.9, 0.9, 2.4, 1.5, 1.9]
        n_seeds = len({r["seed"] for r in runs})
        title = (f"Small taggers trained on JEDI-linear's inputs (hls4ml, (pT, Δη, Δφ), first N particles); "
                 f"test = 260,000 jets; {len(runs)} runs")
        note = ("spread = half the difference between the two seeds. * = early-stopped (patience 5 on val loss). "
                "Best checkpoint = lowest val loss on 25,000 held-out training jets.\n"
                "Reference: JEDI-linear official models, data/official_acc.json. AUC: one class vs rest, score = class probability.\n"
                "Patience-30 rows: reruns of the early-stopped seed-1 runs (same seed); the seed-0 runs never triggered the patience-5 stop.")
        h = 0.31
        fig, ax = plt.subplots(figsize=(sum(widths), 0.9 + h * len(rows_png) + 0.5), dpi=160)
        ax.axis("off")
        t = ax.table(cellText=rows_png, colLabels=cols, colWidths=[w / sum(widths) for w in widths], loc="upper center",
                     cellLoc="left", colLoc="left")
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1, 1.5)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor("#d8d4ca")
            cell.PAD = 0.03
            if r == 0:
                cell.set_facecolor("#eeebe3")
                cell.set_text_props(weight="bold")
            elif rows_png[r - 1][2] == "reference":
                cell.set_facecolor("#e8eef5")
            elif r % 2 == 0:
                cell.set_facecolor("#f8f7f3")
        ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=8)
        fig.text(0.01, 0.005, note, fontsize=8, color="#555")
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
        fig.savefig(png, bbox_inches="tight")
        print("wrote", png)
    return per_run, agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--models_dir", required=True)
    c.add_argument("--official", default="", help="official_acc.json (JEDI-linear reference)")
    c.add_argument("--out", default="-")
    r = sub.add_parser("render")
    r.add_argument("summary")
    r.add_argument("--png", default="")
    r.add_argument("--md", default="")
    args = ap.parse_args()
    if args.cmd == "collect":
        runs = collect(args.models_dir)
        ref = json.load(open(args.official))["3-feature"] if args.official else {}
        out = dict(models_dir=args.models_dir, runs=runs, jedi_linear_3feature=ref)
        s = json.dumps(out, indent=1)
        if args.out == "-":
            sys.stdout.write(s + "\n")
        else:
            with open(args.out, "w") as f:
                f.write(s + "\n")
            print(f"{len(runs)} runs -> {args.out}")
    else:
        summary = json.load(open(args.summary))
        per_run, agg = render(summary, args.png, args.md, summary.get("jedi_linear_3feature", {}))
        print(agg)


if __name__ == "__main__":
    main()
