# Small-tagger training: status (updated 2026-10-17 ~02:20 UTC)

Main set: 7 models (`tf-1b-d16`, `tf-2b-d16`, `tf-1b-d32`, `tf-2b-d32`, `linf-1b-d32-k4`, `linf-2b-d32-k8`, `dsets-d64`)
× N ∈ {32, 64} × seeds {0, 1} = 28 runs, **all 28 complete**. Defaults of `train_small.py`: AdamW lr 1e-3 (200 warm-up
steps, cosine to 0 over 30 epochs), batch 1024, patience 5 on val loss, train = train archive minus `splits.npz` dev
(595,000), val = dev (25,000), test = val archive (260,000). Code snapshot per job under
`/j-jepa-vol/aaron-interp/models/small/_snapshots/`.
Extra set `extra1`: the two early-stopped runs again with patience 30 (directory tag `-p30`, **complete**), and
tf-2b-d32 / tf-2b-d64 at N = 64 for 100 epochs without early stopping, seeds 0 and 1 (tag `-e100`, **running**).

## Jobs (namespace cms-ml, labels owner=anrunw, study=aaron-interp, set=small-taggers; 1 GPU each, limits = requests)
| job | runs | request | state / measured use |
|---|---|---|---|
| `anrunw-ai-train-smoke` | CPU smoke test: tf-1b-d16, linf-1b-d32-k4, dsets-d64 (N=32), tf-1b-d16 `--cls_token` (N=64); 1 epoch, 20k jets → `models/small-smoke/` | 1 CPU, 2 Gi (exempt size) | complete, exit 0, 127 s |
| `anrunw-ai-train-n32-s0` | 7 models, N=32, seed 0 (7 concurrent) | 6 CPU, 16 Gi | complete 7/7; 5 min on an L40; GPU 100 %, memory 15.0 GiB |
| `anrunw-ai-train-n32-s1` | 7 models, N=32, seed 1 | 6 CPU, 20 Gi | **failed** before its container ran (pod failed 20:36:27 UTC, 32 min after start, `BackoffLimitExceeded` with backoffLimit 0; no log, no snapshot, no run dirs; pod and events gone when checked, cause not recorded) |
| `anrunw-ai-train-n32-s1-b` | resubmit of n32-s1 | 6 CPU, 20 Gi | complete 7/7; 10 min on an RTX 2080 Ti; GPU 100 %, memory 12.75 GiB, CPU 5.05 cores average |
| `anrunw-ai-train-n64-s0` | 7 models, N=64, seed 0 | 6 CPU, 20 Gi | complete 7/7; 21 min on an RTX 2080 Ti; GPU 100 %, memory 12.8 GiB, CPU 5.13 cores |
| `anrunw-ai-train-n64-s1` | 7 models, N=64, seed 1 | 6 CPU, 20 Gi | complete 7/7; 16 min on an RTX 2080 Ti; GPU 98–100 %, memory 12.7 GiB, CPU 5.01 cores |
| `anrunw-ai-train-n64-p30` | tf-2b-d16, tf-2b-d32, N=64, seed 1, `--patience=30` → `<model>-n64-p30/seed1` (2 concurrent) | 2 CPU, 6 Gi | complete 2/2; first image pull on k8s-haosu-10 failed (`failed to extract layer … Canceled`, ErrImagePull), second took 8 min 42 s; ran 01:35:17–01:45:20 UTC on an RTX 2080 Ti; GPU 100 %, memory 4.0 GiB, CPU 1.78 cores average |
| `anrunw-ai-train-n64-e100` | tf-2b-d32, tf-2b-d64, N=64, seeds 0/1, `--epochs=100 --patience=100` → `<model>-n64-e100/seed<k>` (4 concurrent) | 4 CPU, 10 Gi | running since 01:25:37 UTC on an RTX 2080 Ti (k8s-haosu-05); GPU 100 %, memory 7.5 GiB, CPU 3.95 cores; tf-2b-d64 reached epoch 22 after 27 min (≈ 74 s per epoch, ≈ 2 h for 100 epochs); at 01:53 UTC val acc tf-2b-d32 epoch 30: 0.738 / 0.757, tf-2b-d64 epoch 22: 0.754 / 0.759 (seed 0 / 1) |
| `anrunw-ai-train-axstats` (CPU, job.sh) | E18: first stats version (float64 moments; superseded) + 1-epoch smoke tests `--axis hardest` dsets-d64 / linf-2b-d32-k8 N=64 and stored-axis dsets-d64 → `models/small-smoke/` | 1 CPU, 2 Gi | complete, exit 0, 184 s; input check slot 0 at (0, 0), padded rows 0 |
| `anrunw-ai-train-axstats2` (CPU, job.sh) | E18: `axis_stats.py` → `data/norm_stats_axhard.json` (JEDI procedure) | 1 CPU, 3 Gi | complete, 45 s; peak memory 973 MiB, 0.95 cores; stored-axis reproduction of norm_stats.json exact |
| `anrunw-ai-train-n64-axhard` | E18: dsets-d64, linf-2b-d32-k8, N=64, seeds 0/1, `--axis=hardest` → `<model>-n64-axhard/seed<k>` (4 concurrent) | 4 CPU, 10 Gi | complete 4/4; 02:01:34–02:12:16 UTC on an RTX 2080 Ti (k8s-haosu-10); GPU 100 %, memory 7.5 GiB peak, CPU 2.52 cores average |

Since n32-s1-b, jobs use `backoffLimit: 1`; a retry skips runs that already have `test.json`. Sizing: about 0.75–1 CPU
core and 1.8–2.1 GiB of container memory per concurrent run (measured on the 7-run jobs).

## Results: mean ± spread over seeds (test archive, 260,000 jets; spread = half the seed-0/seed-1 difference)
`*` = early-stopped (patience 5). Patience-30 rows: reruns of the two early-stopped seed-1 runs (same seed).
Reference rows: JEDI-linear official 3-feature models, `data/official_acc.json`. PNG: `results/site/T7_small_models.png` (also holds the E18 rows).

| model | N | variant | params | test acc seed 0 / seed 1 [%] | mean ± spread [%] | AUC W / Z / t (mean) |
|---|---|---|---|---|---|---|
| tf-1b-d16 | 32 | 30 ep, patience 5 | 2,405 | 68.08 / 70.11 | 69.10 ± 1.02 | 0.920 / 0.909 / 0.942 |
| tf-2b-d16 | 32 | 30 ep, patience 5 | 4,629 | 75.24 / 75.71 | 75.47 ± 0.23 | 0.955 / 0.946 / 0.955 |
| tf-1b-d32 | 32 | 30 ep, patience 5 | 8,901 | 71.83 / 72.08 | 71.96 ± 0.12 | 0.933 / 0.922 / 0.946 |
| tf-2b-d32 | 32 | 30 ep, patience 5 | 17,445 | 77.24 / 77.53 | 77.39 ± 0.14 | 0.963 / 0.955 / 0.960 |
| linf-1b-d32-k4 | 32 | 30 ep, patience 5 | 9,029 | 74.16 / 73.58 | 73.87 ± 0.29 | 0.947 / 0.938 / 0.949 |
| linf-2b-d32-k8 | 32 | 30 ep, patience 5 | 17,957 | 78.41 / 78.34 | 78.37 ± 0.03 | 0.968 / 0.963 / 0.962 |
| dsets-d64 | 32 | 30 ep, patience 5 | 11,269 | 77.78 / 77.80 | 77.79 ± 0.01 | 0.965 / 0.958 / 0.958 |
| JEDI-linear 3-feature (official) | 32 | reference | – | – | 78.00 | – |
| tf-1b-d16 | 64 | 30 ep, patience 5 | 2,405 | 69.31 / 69.21 | 69.26 ± 0.05 | 0.931 / 0.915 / 0.940 |
| tf-2b-d16 | 64 | 30 ep, patience 5 | 4,629 | 73.84 / 72.39* | 73.12 ± 0.73 | 0.952 / 0.934 / 0.947 |
| tf-1b-d32 | 64 | 30 ep, patience 5 | 8,901 | 69.38 / 69.15 | 69.27 ± 0.11 | 0.927 / 0.910 / 0.943 |
| tf-2b-d32 | 64 | 30 ep, patience 5 | 17,445 | 76.19 / 73.42* | 74.80 ± 1.39 | 0.956 / 0.939 / 0.953 |
| linf-1b-d32-k4 | 64 | 30 ep, patience 5 | 9,157 | 76.69 / 77.78 | 77.23 ± 0.54 | 0.956 / 0.951 / 0.958 |
| linf-2b-d32-k8 | 64 | 30 ep, patience 5 | 18,469 | 81.06 / 80.76 | 80.91 ± 0.15 | 0.975 / 0.971 / 0.967 |
| dsets-d64 | 64 | 30 ep, patience 5 | 11,269 | 80.64 / 80.81 | 80.73 ± 0.09 | 0.974 / 0.970 / 0.964 |
| tf-2b-d16 | 64 | 30 ep, patience 30 | 4,629 | – / 74.09 | 74.09 | 0.955 / 0.939 / 0.948 |
| tf-2b-d32 | 64 | 30 ep, patience 30 | 17,445 | – / 76.24 | 76.24 | 0.960 / 0.944 / 0.956 |
| linf-2b-d32-k8 | 64 | 30 ep, axis = hardest particle | 18,469 | 80.90 / 80.89 | 80.90 ± 0.01 | 0.975 / 0.971 / 0.967 |
| dsets-d64 | 64 | 30 ep, axis = hardest particle | 11,269 | 80.00 / 80.30 | 80.15 ± 0.15 | 0.971 / 0.966 / 0.963 |
| JEDI-linear 3-feature (official) | 64 | reference | – | – | 80.92 | – |

Per-model differences between N = 32 and N = 64 (mean test acc; for tf-2b-d16/d32 at N = 64: seed 0 main + seed 1 p30):
tf-1b-d16 69.10 → 69.26; tf-2b-d16 75.47 → 73.97; tf-1b-d32 71.96 → 69.27; tf-2b-d32 77.39 → 76.21; linf-1b-d32-k4
73.87 → 77.23; linf-2b-d32-k8 78.37 → 80.91; dsets-d64 77.79 → 80.73.

## Results: per run
Epochs run / best epoch; best = lowest val loss. Every run that did not stop early has its best epoch at 28–30.

| run | params | epochs run / best epoch | val acc | test acc | test AUC g/q/W/Z/t | rej50 g/q/W/Z/t |
|---|---|---|---|---|---|---|
| tf-1b-d16-n32 s0 | 2,405 | 30/30 | 68.36 | 68.08 | 0.866/0.883/0.914/0.904/0.940 | 17/22/29/37/52 |
| tf-1b-d16-n32 s1 | 2,405 | 30/30 | 69.95 | 70.11 | 0.881/0.890/0.925/0.914/0.944 | 20/24/35/54/70 |
| tf-2b-d16-n32 s0 | 4,629 | 30/30 | 75.16 | 75.24 | 0.910/0.909/0.954/0.946/0.956 | 30/29/80/202/119 |
| tf-2b-d16-n32 s1 | 4,629 | 30/30 | 75.76 | 75.71 | 0.912/0.910/0.956/0.947/0.954 | 32/31/86/220/88 |
| tf-1b-d32-n32 s0 | 8,901 | 30/30 | 71.89 | 71.83 | 0.894/0.898/0.933/0.921/0.947 | 24/27/39/75/96 |
| tf-1b-d32-n32 s1 | 8,901 | 30/30 | 72.00 | 72.08 | 0.896/0.900/0.934/0.923/0.944 | 25/28/44/76/85 |
| tf-2b-d32-n32 s0 | 17,445 | 30/30 | 77.37 | 77.24 | 0.922/0.917/0.963/0.955/0.960 | 37/33/114/319/149 |
| tf-2b-d32-n32 s1 | 17,445 | 30/30 | 77.53 | 77.53 | 0.924/0.918/0.963/0.955/0.961 | 38/34/117/339/157 |
| linf-1b-d32-k4-n32 s0 | 9,029 | 30/29 | 74.39 | 74.16 | 0.906/0.903/0.948/0.940/0.952 | 28/31/83/153/81 |
| linf-1b-d32-k4-n32 s1 | 9,029 | 30/30 | 73.57 | 73.58 | 0.903/0.900/0.945/0.936/0.947 | 27/30/78/117/64 |
| linf-2b-d32-k8-n32 s0 | 17,957 | 30/29 | 78.53 | 78.41 | 0.930/0.919/0.968/0.963/0.962 | 41/37/202/493/133 |
| linf-2b-d32-k8-n32 s1 | 17,957 | 30/30 | 78.62 | 78.34 | 0.929/0.919/0.968/0.963/0.962 | 40/37/202/509/136 |
| dsets-d64-n32 s0 | 11,269 | 30/30 | 77.94 | 77.78 | 0.923/0.916/0.966/0.959/0.958 | 36/35/163/364/115 |
| dsets-d64-n32 s1 | 11,269 | 30/30 | 77.78 | 77.80 | 0.923/0.917/0.965/0.957/0.959 | 36/35/151/352/131 |
| tf-1b-d16-n64 s0 | 2,405 | 30/30 | 68.92 | 69.31 | 0.858/0.889/0.933/0.917/0.940 | 13/24/46/72/49 |
| tf-1b-d16-n64 s1 | 2,405 | 30/30 | 68.83 | 69.21 | 0.859/0.888/0.930/0.914/0.940 | 14/24/43/68/48 |
| tf-2b-d16-n64 s0 | 4,629 | 30/30 | 73.64 | 73.84 | 0.895/0.902/0.954/0.936/0.950 | 21/26/70/208/89 |
| tf-2b-d16-n64 s1 | 4,629 | 14/9 (early stop) | 72.23 | 72.39 | 0.883/0.897/0.950/0.932/0.944 | 18/25/64/165/55 |
| tf-1b-d32-n64 s0 | 8,901 | 30/30 | 68.92 | 69.38 | 0.868/0.890/0.930/0.911/0.942 | 15/24/42/64/61 |
| tf-1b-d32-n64 s1 | 8,901 | 30/30 | 68.92 | 69.15 | 0.874/0.891/0.925/0.910/0.943 | 16/24/40/55/68 |
| tf-2b-d32-n64 s0 | 17,445 | 30/30 | 75.92 | 76.19 | 0.917/0.913/0.960/0.944/0.956 | 30/29/85/313/140 |
| tf-2b-d32-n64 s1 | 17,445 | 12/7 (early stop) | 73.54 | 73.42 | 0.895/0.903/0.952/0.934/0.950 | 21/26/66/189/89 |
| linf-1b-d32-k4-n64 s0 | 9,157 | 30/30 | 76.85 | 76.69 | 0.936/0.910/0.953/0.948/0.956 | 54/34/120/188/77 |
| linf-1b-d32-k4-n64 s1 | 9,157 | 30/30 | 77.77 | 77.78 | 0.938/0.916/0.960/0.954/0.960 | 55/35/167/392/97 |
| linf-2b-d32-k8-n64 s0 | 18,469 | 30/29 | 81.01 | 81.06 | 0.950/0.929/0.976/0.972/0.968 | 71/38/367/1236/170 |
| linf-2b-d32-k8-n64 s1 | 18,469 | 30/29 | 80.78 | 80.76 | 0.948/0.928/0.975/0.971/0.966 | 65/38/346/1071/153 |
| dsets-d64-n64 s0 | 11,269 | 30/29 | 80.48 | 80.64 | 0.945/0.926/0.973/0.968/0.964 | 64/37/343/980/153 |
| dsets-d64-n64 s1 | 11,269 | 30/28 | 80.75 | 80.81 | 0.945/0.926/0.974/0.971/0.964 | 61/37/356/1054/149 |
| tf-2b-d16-n64-p30 s1 | 4,629 | 30/30 | 73.81 | 74.09 | 0.897/0.902/0.955/0.939/0.948 | 22/26/74/224/70 |
| tf-2b-d32-n64-p30 s1 | 17,445 | 30/30 | 76.04 | 76.24 | 0.918/0.913/0.960/0.944/0.956 | 31/30/87/312/139 |
| linf-2b-d32-k8-n64-axhard s0 | 18,469 | 30/30 | 80.94 | 80.90 | 0.950/0.928/0.975/0.971/0.967 | 71/39/336/1018/162 |
| linf-2b-d32-k8-n64-axhard s1 | 18,469 | 30/30 | 80.70 | 80.89 | 0.950/0.929/0.975/0.971/0.967 | 70/38/332/984/172 |
| dsets-d64-n64-axhard s0 | 11,269 | 30/30 | 79.62 | 80.00 | 0.941/0.922/0.971/0.965/0.962 | 60/36/252/560/131 |
| dsets-d64-n64-axhard s1 | 11,269 | 30/30 | 80.26 | 80.30 | 0.945/0.925/0.971/0.966/0.963 | 61/38/263/653/148 |

Early stops (patience 5): tf-2b-d16-n64 seed 1 stopped after epoch 14 (best epoch 9, val loss 0.763, lr 7.9e-4 at the
best epoch); tf-2b-d32-n64 seed 1 stopped after epoch 12 (best 7, val loss 0.734, lr 8.7e-4). Checkpoints of those runs
cover epochs 0–14 and 0–12.
Reruns with patience 30 (same seed 1; the rerun of tf-2b-d32 reproduces the original epoch-7 val loss / acc, 0.7338 /
0.7354 vs 0.73384 / 0.73536): both ran all 30 epochs, best epoch 30, test acc tf-2b-d16 74.09 % (was 72.39), tf-2b-d32
76.24 % (was 73.42). The seed-0 runs never triggered the patience-5 stop, so their training is the same as under
patience 30; seed 0 (main) with seed 1 (p30): tf-2b-d16-n64 73.97 ± 0.12, tf-2b-d32-n64 76.21 ± 0.03.

## E18: inputs measured about the hardest particle (`--axis hardest`)
Question: with the jet axis not at the pT centroid during training, does a network compute the recentred (invariant)
mass instead of Σ pT·r² about the given axis? This node only trains the models; the shift test runs separately.
- Inputs: every real particle's (Δη, Δφ) re-measured about slot 0 (hardest particle): d = x − x[slot 0], Δφ wrapped to
  [−π, π) (no difference needed the wrap: 0 of 620,000 + 260,000 jets), padded rows stay 0, pT unchanged; applied
  identically to train, dev and test jets (`train_small.recentre_hardest`, `axis_stats.recentre_inplace`).
- Distance of the hardest particle from the stored axis (train, N = 64): median 0.042, p10 0.0056, p90 0.111,
  p99 0.212 (val archive the same to 3 digits).
- Normalisation: JEDI-linear's procedure (float32 `np.mean` / `np.std` over axes (0, 1) of the full 620,000-jet training
  archive, padding included) on the re-centred inputs; the same code reproduces `norm_stats.json` exactly (max diff 0.0)
  on the stored inputs. N = 64: shift (14.874188, −4.159e-05, −5.031e-05), scale (39.764400, 0.106536, 0.105346); pT
  entries identical to the stored-axis stats. File `data/norm_stats_axhard.json` (also holds the checks). Note: the float32
  reductions differ from float64 moments (pT shift 14.87 vs 16.31, scales 2–4 % lower); both stored and re-centred
  models use the float32-procedure values.
- Every run's `arch.json` records `inputs.axis`, the re-centring definition, `norm_file`, shift/scale and an input
  check on 4,096 held-out jets.

Results (test archive, 260,000 jets, each model evaluated in its own input convention; 30 epochs, best epoch 30 in all
four runs, no early stop):

| model (N = 64) | stored axis: test acc s0 / s1 → mean ± spread | hardest-particle axis: s0 / s1 → mean ± spread | difference of means |
|---|---|---|---|
| dsets-d64 | 80.64 / 80.81 → 80.73 ± 0.09 | 80.00 / 80.30 → 80.15 ± 0.15 | −0.58 |
| linf-2b-d32-k8 | 81.06 / 80.76 → 80.91 ± 0.15 | 80.90 / 80.89 → 80.90 ± 0.01 | −0.01 |

Background rejection at 50 % signal efficiency (s0 / s1; the number of background jets passing is 170–1,600, Poisson
2.5–7.7 %):

| model | W stored → hardest | Z stored → hardest | t stored → hardest |
|---|---|---|---|
| dsets-d64 | 343 / 356 → 252 / 263 | 980 / 1054 → 560 / 653 | 153 / 149 → 131 / 148 |
| linf-2b-d32-k8 | 367 / 346 → 336 / 332 | 1236 / 1071 → 1018 / 984 | 170 / 153 → 162 / 172 |

Mean AUC W / Z / t: dsets-d64 0.974 / 0.970 / 0.964 (stored) vs 0.971 / 0.966 / 0.963 (hardest); linf-2b-d32-k8
0.975 / 0.971 / 0.967 in both conventions. Dead units dsets-d64-axhard seed 0 at epoch 30: 1.6 / 9.4 / 17.2 / 12.5 / 25.0 %.
Runs: `/j-jepa-vol/aaron-interp/models/small/{dsets-d64,linf-2b-d32-k8}-n64-axhard/seed{0,1}/` (31 checkpoints each).

## Where results land
- Runs: `/j-jepa-vol/aaron-interp/models/small/<model>-n<N>[<tag>]/seed<k>/`: `arch.json`, `log.csv` (per epoch incl.
  `dead_<layer>`), `ckpt/epoch_0.pt` (init) … `epoch_<last>.pt`, `best.pt`, `test.json`, `test_probs.npy`, `train.log`.
- Job logs: `/j-jepa-vol/aaron-interp/logs/train-<job>.log` (start/end per run, monitor line every 3 min).
- Summary: `/j-jepa-vol/aaron-interp/results/small/summary_small.json` (copy in the local repo `results/small/`), tables
  `results/small/table_small.md`, PNG `results/site/T7_small_models.png`. Regenerate:
  ```
  # on the cluster (helper pod, stdlib only)
  python3 code/train/summarize_small.py collect --models_dir models/small --official data/official_acc.json --out results/small/summary_small.json
  # locally (matplotlib)
  python train/summarize_small.py render results/small/summary_small.json --png results/site/T7_small_models.png --md results/small/table_small.md
  ```

## Check progress
```
kubectl -n cms-ml get jobs -l set=small-taggers
POD=$(kubectl -n cms-ml get pods -l owner=anrunw -o name | sed 's|pod/||' | grep '^anrunw-helper-' | head -1)
kubectl -n cms-ml exec $POD -- tail -5 /j-jepa-vol/aaron-interp/logs/train-n64-e100.log
kubectl -n cms-ml exec $POD -- tail -3 /j-jepa-vol/aaron-interp/models/small/tf-2b-d64-n64-e100/seed0/train.log
```
A failed or missing run is rerun by resubmitting its job with a suffix: `python train/make_jobs.py --set <set> --only <job>
--suffix=-b` (`=` needed: `-b` alone parses as a flag), then `kubectl apply -f train/jobs/<job>-b.yaml`; runs with
`test.json` are skipped.
