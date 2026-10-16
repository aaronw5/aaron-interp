# E9 (scope v3) + cluster setup (agent report, 2026-10-16)

Cluster: sync.sh (repo → /j-jepa-vol/aaron-interp/code, nothing removed); job.sh <name> '<cmd>' defaults to 1 CPU / 2 GiB (NRP-exempt),
`job.sh <name> <cpu> <mem_gi> '<cmd>'` for measured sizes; log trailer: exit code, wall, cgroup peak memory, average cores; AI_PY=conda
switches to the image python (torch 1.13, uproot 5, awkward 2, weaver-core). Env /j-jepa-vol/phat-jet-aaron/env (read-only): keras 3.12.4,
jax 0.6.2 CPU, hgq2 0.1.9, scikit-learn 1.7.2, matplotlib 3.10.9, h5py 3.16, pandas, scipy. Env import from CephFS ≈ 5 min per job.
Official n64 model on the cluster: 0.80938 on 260k jets (official 0.80921).

Volume data: norm_stats.json, splits.*, val_*.{h5,npy}, train_*, n64/full_test.npz, official_acc.json (all 20 reported accuracies),
norm_stats_16f.json + val_c16_f16.npy (16-feature test inputs prepared with the official layout; the 16-feature models were NOT run).

Results (results/sweep/summary.md; 260,000 test jets):
| model | acc | official | live l1 | units > 1 % calls a/P/g/s/u/S/h4/h5/z | identity all slots max (rel) | identity on kept slots | zero-bit qs/l3/qs1 |
| 3-feature n64 | 0.8094 | 0.8092 | 15 | 13/10/7/9/19/19/38/29/13 | 4.62 (0.38) | 0.96 (0.078) | 0.82/0.80/0.84 |
| n64 untrained (trained bit widths, redrawn weights) | 0.2023 | – | 11 | 0/0/0/0/0/0/0/0/0 | 1.04 (0.077) | 1.04 (0.077) | same |
Zero-bit per layer (trained): l1 0, qs 0.816, l2 0.794, dd 0.781, l3 0.801, qs1 0.842, d4 0.703, d5 0.125, d6 0, d7 0.
qs1 kept slots per live u: u0 14, u1 63, u13 11, u17 10, u19 51, u20 3, u29 64, u30 64, u31 47, u40 23, u43 14, u50 42, u51 14, u52 14,
u56 56, u58 60, u60 26, u62 59, u63 14. The all-slot identity residual is the qs quantizer's 0-bit slots; on kept slots it is 7.8 % relative,
equal to the untrained network's value (fixed-point rounding remainder). The untrained network calls everything t (no unit changes any call):
as a control for the object structure it is degenerate.
Reported accuracies of the 20 official variants: best overall 16-feature-perminv n64 82.36 %; best 3-feature family: 3-feature-perminv n64
81.81 %; 3-feature n64 80.92 %.
Job sizes measured: e1e3 n64 at 1 CPU / 2 GiB: 0.50 / 0.46 cores average, working set 1.1–1.3 GiB (cgroup peak at the limit via page cache,
no OOM). Cancelled (own, after the scope cut): sw-3f-n8/16/32/128, sw-3fp-*.
