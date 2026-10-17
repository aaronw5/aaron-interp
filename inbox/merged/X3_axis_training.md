# X3 — E18: small taggers trained with (Δη, Δφ) measured about the hardest particle (agent report, 2026-10-17)

**Question (training part only).** If the origin of the angular inputs is not the pT centroid during training, does a
network compute the recentred mass instead of Σ pT·r² about the given origin? This node trains the models. The shift
test about each model's own axis runs separately.

**Edit (inputs).** `train_small.py --axis hardest` re-measures every real particle's (Δη, Δφ) about slot 0, the hardest
particle (the archive is pT-ordered):
- d = x − x[slot 0], so slot 0 sits at (0, 0);
- Δφ differences are wrapped to [−π, π);
- padded rows stay 0 and pT is unchanged;
- train, dev and test jets are treated identically.

**Sample.** Train 595,000 jets (train archive minus the `splits.npz` dev jets), dev 25,000, test 260,000 (the val
archive). Scans of all 620,000 + 260,000 jets at N = 64:
- no jet has an empty slot 0;
- no padded row has nonzero angles;
- no Δφ difference needed the wrap.

Distance of the hardest particle from the stored axis (train archive): median 0.042, p10 0.0056, p90 0.111, p99 0.212.
The val archive agrees to 3 digits.

**Normalisation.** JEDI-linear's procedure (`jedi-distill/jedi/data.py`): float32 `np.mean` / `np.std` over axes (0, 1)
of the full 620,000-jet training archive `[:, :N]`, padding included, applied to the recentred inputs
(`train/axis_stats.py`).
- On the stored inputs the same code reproduces `data/norm_stats.json` exactly (max difference 0.0, N = 32 and 64).
- N = 64, recentred: shift (14.874188, −4.159e-05, −5.031e-05), scale (39.764400, 0.106536, 0.105346).
- Stored, for comparison: Δη / Δφ scale 0.090830 / 0.089523. The pT entries are identical to the stored ones.
- File: `/j-jepa-vol/aaron-interp/data/norm_stats_axhard.json`. It also records the checks and the float64 moments. Those
  differ from the float32 reductions (pT shift 16.31 vs 14.87); the stored-axis models also use the float32 values.

Each run's `arch.json` records:
- `inputs.axis` and the recentring definition;
- `norm_file` and the shift/scale used;
- an input check on 4,096 held-out jets: slot-0 offset 0, padded rows 0, max |Δη| 1.405, max |Δφ| 0.617.

**Training.** dsets-d64 and linf-2b-d32-k8 at N = 64, seeds 0 and 1. Settings identical to the stored-axis runs: AdamW,
lr 1e-3 cosine, batch 1024, 30 epochs, patience 5 (not triggered; best epoch 30 in all four runs).
- Job `anrunw-ai-train-n64-axhard`: 4 runs at once on one RTX 2080 Ti, GPU 100 %, 7.5 GiB, 2.5 CPU cores, 11 min.
- Checkpoints `ckpt/epoch_0..30.pt` per run.

**Results** (test archive; each model evaluated in its own input convention):

| N = 64 | stored axis: s0 / s1 → mean ± spread | hardest-particle axis: s0 / s1 → mean ± spread | difference |
|---|---|---|---|
| dsets-d64 | 80.64 / 80.81 → 80.73 ± 0.09 | 80.00 / 80.30 → 80.15 ± 0.15 | −0.58 |
| linf-2b-d32-k8 | 81.06 / 80.76 → 80.91 ± 0.15 | 80.90 / 80.89 → 80.90 ± 0.01 | −0.01 |

Background rejection at 50 % signal efficiency, seed 0 / seed 1:

| | W | Z | t |
|---|---|---|---|
| dsets-d64, stored axis | 343 / 356 | 980 / 1054 | 153 / 149 |
| dsets-d64, hardest-particle axis | 252 / 263 | 560 / 653 | 131 / 148 |
| linf-2b-d32-k8, stored axis | 367 / 346 | 1236 / 1071 | 170 / 153 |
| linf-2b-d32-k8, hardest-particle axis | 336 / 332 | 1018 / 984 | 162 / 172 |

Between 168 and 1,586 background jets pass at these working points (Poisson 2.5–7.7 %).

Mean AUC W / Z / t:
- dsets-d64: 0.974 / 0.970 / 0.964 (stored), 0.971 / 0.966 / 0.963 (hardest);
- linf-2b-d32-k8: 0.975 / 0.971 / 0.967 in both conventions.

**Outputs.**
- Runs: `/j-jepa-vol/aaron-interp/models/small/{dsets-d64,linf-2b-d32-k8}-n64-axhard/seed{0,1}/` (`arch.json`, `log.csv`,
  `ckpt/`, `best.pt`, `test.json`, `test_probs.npy`).
- Table: `train/STATUS.md` (E18 section).
- PNG: `results/site/T7_small_models.png` (rows "axis = hardest particle").
- Recentring code for the shift test: `train/train_small.py` `recentre_hardest()` and `GPUData.raw()`.
