"""Normalisation stats for inputs re-measured about the hardest particle (E18), with JEDI-linear's procedure.

    python3 train/axis_stats.py --data_dir /j-jepa-vol/aaron-interp/data --ns 32,64 --out <data_dir>/norm_stats_axhard.json

JEDI-linear's norm_stats.json (jedi-distill/jedi/data.py): a = np.array(X[:, :N]).astype(np.float32) over the FULL training
archive train_ptetaphi_f16.npy (620,000 jets, padding rows included); shift = np.mean(a, axis=(0, 1)), scale =
np.std(a, axis=(0, 1)) — float32 reductions. The same code is applied here after re-centring every real particle's
(deta, dphi) about slot 0 (the hardest particle; the archive is pT-ordered): d = x - x[slot 0], dphi wrapped to
[-pi, pi), padded rows (pt == 0) left at 0, pt unchanged — the formula of train_small.recentre_hardest. Because pt is not
changed, the pt shift / scale come out identical to norm_stats.json.

Key "_check":
  - jedi_procedure_stored_n<N>: the same code on the stored inputs vs norm_stats.json (reproduction of the procedure);
  - exact_float64_n<N>: chunked float64 moments of both input conventions (the float32 reductions differ from these;
    recorded for reference, not used);
  - scan_<archive>_n<N>: jets with an empty slot 0, padded rows with nonzero angles, dphi differences that needed the
    wrap, and the distance of the hardest particle from the stored axis (quantiles of sqrt(deta0^2 + dphi0^2)).
numpy + json only. Peak memory about 2 GB at N = 64.
"""
import argparse
import json
import os

import numpy as np

PI = np.float32(np.pi)
TWO_PI = np.float32(2 * np.pi)


def recentre_inplace(a):
    """a: (J, N, 3) float32 raw, modified in place; returns the number of dphi differences that were wrapped."""
    mask = a[..., 0] > 0
    c = a[:, :1, 1:3].copy()
    a[..., 1:3] -= c
    dphi = a[..., 2]
    n_wrap = int(((dphi >= PI) | (dphi < -PI))[mask].sum())
    a[..., 2] = np.remainder(dphi + PI, TWO_PI) - PI
    a[..., 1:3] *= mask[..., None]
    return n_wrap


def jedi_stats(x16, n, axis):
    a = np.array(x16[:, :n]).astype(np.float32)  # same dtype path as jedi/data.py norm_stats()
    if axis == "hardest":
        recentre_inplace(a)
    out = dict(shift=np.mean(a, axis=(0, 1)).tolist(), scale=np.std(a, axis=(0, 1)).tolist())
    del a
    return out


def exact_stats(x16, n, axis, chunk=20000):
    s1, s2, cnt = np.zeros(3), np.zeros(3), 0
    for i in range(0, len(x16), chunk):
        a = np.asarray(x16[i:i + chunk, :n]).astype(np.float32)
        if axis == "hardest":
            recentre_inplace(a)
        a = a.reshape(-1, 3).astype(np.float64)
        s1 += a.sum(0)
        s2 += (a * a).sum(0)
        cnt += len(a)
    mean = s1 / cnt
    return dict(shift=mean.tolist(), scale=np.sqrt(np.maximum(s2 / cnt - mean * mean, 0.0)).tolist())


def scan(x16, n, chunk=20000):
    empty0 = pad_nonzero = n_wrap = 0
    r0 = []
    for i in range(0, len(x16), chunk):
        a = np.asarray(x16[i:i + chunk, :n]).astype(np.float32)
        mask = a[..., 0] > 0
        empty0 += int((~mask[:, 0]).sum())
        pad_nonzero += int((np.abs(a[..., 1:3]).max(-1) > 0)[~mask].sum())
        r0.append(np.hypot(a[:, 0, 1], a[:, 0, 2]))
        n_wrap += recentre_inplace(a)
    r0 = np.concatenate(r0)
    q = np.percentile(r0, [10, 50, 90, 99])
    return dict(jets=int(len(x16)), n=n, empty_slot0=empty0, padded_rows_nonzero_angles=pad_nonzero, dphi_wrapped=n_wrap,
                hardest_particle_distance_from_stored_axis=dict(p10=float(q[0]), median=float(q[1]), p90=float(q[2]),
                                                                p99=float(q[3]), mean=float(r0.mean())))


def diff(a, b):
    d = [abs(x - y) for k in ("shift", "scale") for x, y in zip(a[k], b[k])]
    r = [abs(x - y) / abs(y) for k in ("shift", "scale") for x, y in zip(a[k], b[k]) if y != 0]
    return dict(max_abs_diff=max(d), max_rel_diff=max(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ns", default="32,64")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ns = [int(v) for v in args.ns.split(",")]
    xtr = np.load(os.path.join(args.data_dir, "train_ptetaphi_f16.npy"), mmap_mode="r")
    xva = np.load(os.path.join(args.data_dir, "val_ptetaphi_f16.npy"), mmap_mode="r")
    ref = json.load(open(os.path.join(args.data_dir, "norm_stats.json")))
    out = {"_axis": "hardest",
           "_procedure": ("JEDI-linear norm_stats procedure (jedi-distill/jedi/data.py: float32 np.mean / np.std over axes "
                          "(0, 1) of the full training archive train_ptetaphi_f16.npy[:, :N], 620,000 jets, padding rows "
                          "included) applied after re-centring (deta, dphi) about slot 0 (hardest particle): d = x - "
                          "x[slot 0], dphi wrapped to [-pi, pi), padded rows stay 0, pt unchanged"),
           "_made_by": "train/axis_stats.py", "_check": {}}
    for n in ns:
        out[str(n)] = jedi_stats(xtr, n, "hardest")
        st = jedi_stats(xtr, n, "stored")
        out["_check"][f"jedi_procedure_stored_n{n}"] = dict(recomputed=st, norm_stats_json=ref[str(n)], **diff(st, ref[str(n)]))
        ex = dict(stored=exact_stats(xtr, n, "stored"), hardest=exact_stats(xtr, n, "hardest"))
        out["_check"][f"exact_float64_n{n}"] = ex
        print(f"N={n} hardest (used) {out[str(n)]}\n  stored, same code   {st}\n  norm_stats.json     {ref[str(n)]}\n"
              f"  reproduction {diff(st, ref[str(n)])}\n  exact float64: stored {ex['stored']} hardest {ex['hardest']}",
              flush=True)
    nmax = max(ns)
    for name, x in (("train", xtr), ("val", xva)):
        out["_check"][f"scan_{name}_n{nmax}"] = sc = scan(x, nmax)
        print(name, sc, flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
