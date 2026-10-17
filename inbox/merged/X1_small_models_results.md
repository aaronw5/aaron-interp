# X1 — small taggers on hls4ml: all 28 main runs (agent report, 2026-10-17)

**Procedure.** 7 architectures trained from scratch on JEDI-linear's inputs: the first N pT-ranked particles, (pT, Δη, Δφ)
normalised with `norm_stats.json[N]`, padded rows set to zero and a mask passed. Models:
- native pre-LN transformer `tf-{1,2}b-d{16,32}` (2 heads, MLP ×2, GELU, masked mean pooling, no positional encoding);
- Linformer `linf-1b-d32-k4` and `linf-2b-d32-k8`;
- Deep Sets `dsets-d64`.

Each at N ∈ {32, 64} with seeds 0 and 1. AdamW at lr 1e-3, cosine schedule, batch 1024, 30 epochs, early stopping at
patience 5 on val loss; the best checkpoint is the one with the lowest val loss.

**Samples.**
- train: the train archive minus the `splits.npz` dev jets (595,000);
- val: the dev jets (25,000);
- test: the val archive (260,000 jets), never used for training or selection.

Code: `train/` (models.py, train_small.py, gpu_job.sh, make_jobs.py, summarize_small.py). Table: `train/STATUS.md`
(30 runs: the 28 main runs plus the 2 patience-30 reruns).
PNG: `results/site/T7_small_models.png`.

**Results.** Test accuracy in %, mean ± spread over the 2 seeds, where spread = half the difference between the seeds.

| model | params | N = 32 | N = 64 |
|---|---|---|---|
| tf-1b-d16 | 2,405 | 69.10 ± 1.02 | 69.26 ± 0.05 |
| tf-2b-d16 | 4,629 | 75.47 ± 0.23 | 73.12 ± 0.73 (seed 1 early-stopped, epoch 14); patience-30 rerun of seed 1: 74.09 → seeds 0+1 73.97 ± 0.12 |
| tf-1b-d32 | 8,901 | 71.96 ± 0.12 | 69.27 ± 0.11 |
| tf-2b-d32 | 17,445 | 77.39 ± 0.14 | 74.80 ± 1.39 (seed 1 early-stopped, epoch 12); patience-30 rerun of seed 1: 76.24 → seeds 0+1 76.21 ± 0.03 |
| linf-1b-d32-k4 | 9,029 / 9,157 | 73.87 ± 0.29 | 77.23 ± 0.54 |
| linf-2b-d32-k8 | 17,957 / 18,469 | 78.37 ± 0.03 | 80.91 ± 0.15 |
| dsets-d64 | 11,269 | 77.79 ± 0.01 | 80.73 ± 0.09 |
| JEDI-linear 3-feature (official, `data/official_acc.json`) | – | 78.00 | 80.92 |

Best epochs:
- Every run that did not stop early has its best epoch at 28–30 of 30, so val loss was still falling when the cosine
  schedule ended.
- The two early stops (tf-2b-d16-n64 seed 1, tf-2b-d32-n64 seed 1) stopped while the learning rate was still 7.9e-4 and
  8.7e-4. Their seed-0 counterparts reached 73.84 % and 76.19 %.

Per class, AUC one-vs-rest averaged over seeds, for W / Z / t:
- linf-2b-d32-k8 at N = 64: 0.975 / 0.971 / 0.967;
- dsets-d64 at N = 64: 0.974 / 0.970 / 0.964;
- tf-2b-d32 at N = 32: 0.963 / 0.955 / 0.960.

Dead units, measured on 10k held-out jets:
- 0 in every transformer GELU layer at every epoch;
- dsets-d64 ReLU layers at N = 32, seed 0: 1.6 / 9.4 / 21.9 / 15.6 / 18.8 % from epoch 1 on.

**Checkpoints and outputs.** Every run directory holds:
- `ckpt/epoch_0.pt` (initialisation) through the last epoch;
- `best.pt`, `log.csv` (per epoch, including the dead-unit fraction per layer);
- `test.json`, `test_probs.npy`.

Location: `/j-jepa-vol/aaron-interp/models/small/<model>-n<N>/seed<k>/`.

**Jobs.**
- 4 GPU jobs, each running the 7 models of one (N, seed) at once on one GPU.
- Measured per job: GPU 98–100 %, 12.7–15.0 GiB container memory, 5.0–5.1 CPU cores out of 6.
- `anrunw-ai-train-n32-s1` failed before its container started (BackoffLimitExceeded, cause not recorded) and was rerun
  as `-n32-s1-b`.

**Patience-30 reruns (done).**
- Job `anrunw-ai-train-n64-p30` reran the two early-stopped seed-1 runs with the same seed and `--patience=30`; output
  directories `<model>-n64-p30/seed1`.
- The tf-2b-d32 rerun reproduces the original run's epoch-7 val loss / acc: 0.7338 / 0.7354 vs 0.73384 / 0.73536.
- Both reruns ran all 30 epochs with the best epoch at 30. Test acc: tf-2b-d16 74.09 % (was 72.39), tf-2b-d32 76.24 %
  (was 73.42).
- The seed-0 runs never triggered the patience-5 stop, so their training is the same as under patience 30. Combining
  them with the seed-1 reruns gives the seed means in the table above.

**Running (set extra1).**
- `anrunw-ai-train-n64-e100`: tf-2b-d32 and tf-2b-d64 (2 heads, MLP ×2, 67,653 params) at N = 64, 100 epochs of
  cosine schedule with no early stop, seeds 0 and 1, output directories `<model>-n64-e100/seed<k>`.
  - Started 01:25:37 UTC.
  - tf-2b-d64 takes about 55 s per epoch with 4 runs on an RTX 2080 Ti.
  - Epoch 1 val acc: 0.654.
- At 01:53 UTC tf-2b-d64 was at epoch 22 (≈ 74 s per epoch, ≈ 2 h for 100 epochs), val acc 0.754 / 0.759 (seed 0 / 1); tf-2b-d32 at epoch 30, val acc 0.738 / 0.757. These go into the table when they finish.
