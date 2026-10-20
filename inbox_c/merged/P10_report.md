# P10 — ParT_full / ParT_kin retrained from scratch, 3 seeds each, on L-GATr's 2M JetClass subset: launch, smoke check, interim axis battery (claims C3, D4; weakness 3, confound C6), 2026-10-19

Status: **running**. There are six training jobs, one seed per GPU: `anrunw-ai-c-p10-full-s0`, `-full-s1`, `-full-s2`, `-kin-s0`, `-kin-s1`, `-kin-s2`. They should finish at about 2026-10-20 06:00 UTC. The interim battery is done on epochs 0 and 3.
Scripts:
- `interp/p10_part_seeds_train.sh` (training payload)
- `interp/p10_part_seeds_job.sh` (GPU job launcher; a copy of `interp/h2_gpu_job.sh` with a 72 h deadline and CPU, memory and GPU set by `P10_*` variables)
- `interp/p10_seed_battery.py` (battery, with the pre-registration in its docstring)

Results:
- `results/part/p10_battery_e0.json`, `p10_battery_e3.json`, `p10_battery_smoke.json`
- figures `results/part/p10_battery_e0.png` and `p10_battery_e3.png`

Logs are on the volume: `/j-jepa-vol/aaron-interp/logs/c-p10-{full,kin}-s{0,1,2}.log`, `c-p10-run.log`, `c-p10-run-e3.log`, `c-p10-smoke.log`. Each seed's training log is in its model directory: `/j-jepa-vol/aaron-interp/models/part_seeds/<feat>_s<k>/train.log`.

## Conclusion
We are retraining the Particle Transformer six times from scratch (two input sets, three random seeds each) on the same 2 million jets used for our L-GATr run. All six runs are training normally and should finish overnight. On the same 10,000 test jets, the released network turns 22 % more true-W jets into Z under a rotation of the particles by 0.06 rad around the stored jet axis. That edit leaves the jet's invariant mass unchanged. After 3 of 50 epochs, the one finished full-input seed shows the same change (+23 points). The three kinematics-only seeds show a smaller change (+4 to +9 points), although they respond to a real 10 % mass change about as strongly as the released network does. These are early checkpoints, so this does not yet tell us whether the axis dependence is a property of the released checkpoint or of the training. The final checkpoints answer that, with the four pre-registered tests below.

## Procedure
1. **Released weights check (2026-10-19).**
   - The `jet-universe/particle_transformer` repository (GitHub API listing of `models/`) has one checkpoint per feature set: ParT_full/kin/kinpid and ParticleNet_full/kin/kinpid.
   - HuggingFace model search for "jetclass" and "particle transformer jetclass" found nothing.
   - The ParT paper (arXiv 2202.03772) reports ParT trained on 2 % and 10 % of JetClass (accuracy −2.5 % and −1.1 %) but releases no weights for them.
   - The scaling-law paper arXiv 2602.15781 lists no checkpoints.
   - No multi-seed or reduced-data ParT weights were found, so we train our own.
2. **Training.** Authors' `train_JetClass.sh` (particle_transformer @ 2925bdb in `/j-jepa-vol/aaron-interp/ext/particle_transformer`), weaver-core 0.4.17 (env-part).
   - **Unchanged:** ParT network config, batch 512, start lr 1e-3, ranger, AMP, 50 epochs, flat+decay schedule (decay over the last 15 epochs), 2 loader workers, fetch step 0.01, streaming from ROOT.
   - **Data:** the P5 subset (`/j-jepa-vol/aaron-interp/data/jetclass2m/selection.json`). Per class, 2 random training files (2M jets in total), 1 random validation file (1M jets in total), and test file 100. The script reads `./datasets/JetClass/Pythia/<split>`, which is symlinked to that subset.
   - **Changed:**
     - Samples per epoch are 1,024,000 for training and 128,000 for validation, 1/10 of the authors' values. The authors' total of 512M samples would be 256 passes over 2M jets. Here it is 51.2M samples, 25.6 passes; for comparison, L-GATr in P5 runs 12.8 passes.
     - Output paths are redirected.
     - **Seeds.** weaver 0.4.17 has no seed option, and torch's default seed is a fixed constant, so unseeded runs would start from identical weights. A per-seed `weaver` wrapper first on PATH sets the python, numpy, torch and cuda seeds to k, then calls `weaver.train.main`. The logs confirm "torch initial_seed 0/1/2".
3. **Launch history.**
   - **Two jobs with 3 seeds per GPU, as the plan specified.** These were `c-p10-full` and `c-p10-kin`, restricted to 48 GB GPUs. They stayed Pending for 20 min because no such GPU was free.
   - **Relaunch allowing 22–24 GB GPUs.** Two of the three kin seeds ran out of GPU memory on an A10 (22 GB). The third, running alone, kept the A10 85–100 % busy at 2.1 steps/s.
   - **Final layout.** Both jobs were deleted and relaunched as six single-seed jobs (2 CPU, 12 Gi, 1 GPU of ≥ 22 GB). The partial outputs of the deleted jobs were removed first.
   - **Why six jobs is fine.** One seed per GPU saturates the GPU, so three seeds on one GPU would not have trained faster in total.
4. **Battery.** `python -m interp.p10_seed_battery --epoch <e>` as GPU jobs `c-p10-run` (epoch 0) and `c-p10-run-e3` (epoch 3). The first attempt of `c-p10-run` crashed with a JSONDecodeError while reading `files.json`, which a concurrent `./sync.sh` was rewriting. The fix reads all file paths once with a retry.
   - **Sample:** 1,000 random rows per class (numpy `default_rng(2026)`, without replacement) of each class's test file 100, N = 10,000 (1,000 true W), rebuilt with `part_core.build`.
   - **Models:** released ParT_full and ParT_kin, plus each seed that had saved the epoch. At epoch 0, all seeds. At epoch 3: full s0, kin s0/s1/s2. Full s1 and s2 had been queued for 15 and 30 min and had not reached epoch 3.
   - **Edits:**
     - phi0.06: every particle's Δφ + 0.06 with the stored axis kept (Δlog m mean 1e-7, max |Δlog m| 6e-4).
     - k1.1: angles × 1.1, which is a real mass change (Δlog m +0.094).
     - wta: Δη, Δφ and ΔR about the winner-take-all axis (4-vectors unchanged; WTA axis = hardest particle in 57.7 % of jets).
     - escheme: the same recomputation about the E-scheme axis, used as the control for wta.
     - rot0.06: the whole jet including the stored axis rotated by 0.06, which is the float-noise null.
   - **Metrics:**
     - accuracy
     - % calls changed (Wilson CI)
     - W→Z before → after, with a paired bootstrap CI (2,000 resamples over W jets) and an exact McNemar test, Holm-corrected over the 30 tests per run (6 models × 5 edits)
     - mean Δ(logit_W − logit_Z) on W jets
     - R = mean Δ(lW − lZ) under phi0.06 / mean Δ(lW − lZ) under k1.1, with a paired bootstrap CI. The axis-moment model √(M0·M2) predicts R = ½Δlog(M0M2)|phi / ½Δlog(M0M2)|k1.1 = 0.972 on these W jets. The invariant-mass hypothesis predicts R ≈ 0.
5. **Pre-registration** (in the docstring of `interp/p10_seed_battery.py`, written before any final checkpoint exists). These are tested later on each seed's final checkpoint, N = 10,000 random test jets, with a fresh seed.
   - **S1:** the calibrated ridge read-out ratio under phi0.06 lies in 0.7–1.4 of the √(M0M2) prediction in every seed (vs ~0 for an invariant mass).
   - **S2:** the per-particle ΔR route carries > 0.8 of the phi0.06 read-out change in every full seed (kin: Δφ and ΔR together).
   - **S3:** the n98-analogue is the class-token unit with the largest |W_W − W_Z| × sd. Particle-block-1 heads carry > 0.9 of its response to k1.1 under resample ablation.
   - **S4:** a head exists whose attention switches with jet mass (logistic m50 in 60–110 GeV, width < 15 GeV).
   - **Expected under the thesis:** all seeds show the axis dependence and the block-level route, with different indices.
   - **Alternative:** the axis dependence varies strongly by seed, or vanishes at 2M jets.

## Numbers
**Smoke check of the trainings** (train AvgLoss per epoch | validation accuracy per epoch; 2026-10-19 20:32 UTC):

| run | GPU | train loss e0 → e3 | val acc e0 → e3 | GPU util (snapshot) | pod CPU / memory (request 2 / 12 Gi) |
|---|---|---|---|---|---|
| full_s0 | A10 | 0.835 → 0.640 → 0.593 → 0.568 | 0.760 → 0.790 → 0.799 → 0.806 | 69–93 % | 1.34 / 7.6 GiB |
| full_s1 | Quadro RTX 6000 | 0.821 → 0.638 (e0, e1) | 0.764 → 0.787 | 65 % | 1.33 / 10.2 GiB |
| full_s2 | A10 | 0.827 (e0) | 0.760 | 74 % | 1.45 / 6.9 GiB |
| kin_s0 | A10 | 1.165 → 0.975 → 0.901 → 0.874 | 0.646 → 0.682 → 0.676 → 0.701 | 75–91 % | 1.32 / 6.0 GiB |
| kin_s1 | A10 | 1.164 → 0.971 → 0.911 → 0.882 | 0.651 → 0.668 → 0.692 → 0.699 | 42–75 % | 1.27 / 5.6 GiB |
| kin_s2 | A10 | 1.172 → 0.988 → 0.934 → 0.890 | 0.635 → 0.660 → 0.688 | 70–98 % | 1.44 / 10.0 GiB |

- **Speed and ETA.** One epoch (2,000 steps + 250 validation steps) takes about 10.5 min, so 50 epochs take about 9 h. The ETA is about 2026-10-20 05:00 UTC for full_s0 and the kin seeds and about 06:00 UTC for full_s2, plus 1M-jet test inference.
- **Memory.** Pod memory includes page cache. Two pods were at 83–87 % of the 12 Gi limit in the snapshot.

**Unedited accuracy on the battery sample (N = 10,000):** released ParT_full 86.04 % (85.9 % on the P1 sample) and ParT_kin 74.22 % (74.5 %). The released models also reproduce P1/P1v on these random rows:

| quantity | here | P1 / P1v reference |
|---|---|---|
| phi0.06 W→Z, full | 9.4 → 31.4 % | 12 → 31 % |
| phi0.06 W→Z, kin | 16.7 → 35.7 % | 16 → 37 % |
| WTA calls changed, full | 11.1 % [10.5, 11.7] | 11.7 % |
| WTA calls changed, kin | 34.3 % [33.4, 35.2] | 34.1 % |

**Edits.** Δ(W→Z) is in percentage points with a 95 % bootstrap CI, p is Holm-corrected McNemar, and R is the logit ratio phi/k with its CI. The √(M0M2) prediction for R is 0.97.

| model @ epoch | acc % | W→Z unedited % | Δ(W→Z) phi0.06 | p | Δ(W→Z) k1.1 | Δ(W→Z) rot0.06 null | R | wta calls changed % (escheme ctrl) |
|---|---|---|---|---|---|---|---|---|
| released_full | 86.0 | 9.4 | +22.0 [19.3, 24.8] | 7e-52 | +29.3 [26.3, 32.3] | 0.0 [0.0, 0.0] | 0.76 [0.73, 0.79] | 11.1 (0.0) |
| full_s0 @ e0 | 76.5 | 27.1 | +17.6 [14.4, 20.5] | 4e-26 | +18.9 [15.9, 22.2] | −0.1 [−0.5, 0.2] | 0.99 [0.95, 1.03] | 26.1 (0.1) |
| full_s0 @ e3 | 80.2 | 11.5 | +22.6 [19.7, 25.4] | 1e-46 | +23.7 [20.8, 26.8] | +0.1 [0.0, 0.3] | 0.94 [0.89, 0.99] | 16.1 (0.1) |
| released_kin | 74.2 | 16.7 | +19.0 [16.1, 21.8] | 9e-34 | +29.0 [25.9, 32.3] | −0.4 [−0.9, 0.0] | 0.69 [0.65, 0.73] | 34.3 (0.1) |
| kin_s0 @ e0 | 64.5 | 24.5 | −7.8 [−9.9, −5.6] | 3e-11 | +20.3 [17.0, 23.5] | +0.1 [0.0, 0.3] | −0.28 [−0.32, −0.25] | 28.9 (0.3) |
| kin_s1 @ e0 | 65.4 | 17.0 | +2.6 [0.8, 4.4] | 0.08 | +18.4 [15.7, 21.1] | 0.0 [0.0, 0.0] | 0.19 [0.14, 0.23] | 26.9 (0.2) |
| kin_s2 @ e0 | 63.9 | 36.0 | −9.9 [−12.3, −7.5] | 6e-14 | +19.1 [16.1, 22.4] | 0.0 [−0.4, 0.4] | −0.26 [−0.30, −0.21] | 31.2 (0.2) |
| kin_s0 @ e3 | 69.6 | 13.7 | +8.5 [6.3, 10.8] | 7e-12 | +20.7 [17.7, 23.8] | +0.3 [−0.3, 0.9] | 0.35 [0.31, 0.39] | 19.2 (0.1) |
| kin_s1 @ e3 | 69.7 | 13.9 | +4.2 [2.1, 6.3] | 2e-3 | +21.8 [18.9, 24.8] | +0.2 [−0.4, 0.8] | 0.17 [0.13, 0.21] | 18.6 (0.1) |
| kin_s2 @ e3 | 68.9 | 19.8 | +4.4 [2.3, 6.5] | 1e-3 | +23.4 [20.1, 26.8] | +0.5 [0.1, 0.9] | 0.20 [0.17, 0.24] | 20.3 (0.3) |

WTA edit, Δ(W→Z) (points; Holm McNemar p):

| model | Δ(W→Z) | p |
|---|---|---|
| released_full | +23.0 | 1e-52 |
| released_kin | +21.5 | 1e-37 |
| full_s0 @ e3 | +24.0 | 2e-50 |
| kin_s0 @ e3 | +16.0 | 2e-28 |
| kin_s1 @ e3 | +12.2 | 6e-18 |
| kin_s2 @ e3 | +9.8 | 7e-10 |

## Reading
- **Hypotheses and the null.** H0 (invariant mass) predicts Δ(W→Z) under phi0.06 and wta at the rot0.06 floor, which is 0.0–0.5 points here, and R ≈ 0. H_axis (√(M0M2) about the stored axis) predicts R ≈ 0.97.
- **full_s0 at epoch 3, the only full seed with epoch 3:**
  - Δ(W→Z) under phi0.06 is +22.6 points, within the released model's CI.
  - R = 0.94 [0.89, 0.99], which excludes H0 and is consistent with the √(M0M2) prediction. For the released model R = 0.76.
  - WTA changes 16.1 % of calls (released 11.1 %).
- **kin seeds at epoch 3:**
  - Δ(W→Z) under phi0.06 is +4.2 to +8.5 points and R = 0.17–0.35. Both exclude H0 (Holm p ≤ 2e-3) and lie below the released ParT_kin (+19.0, R 0.69) and below H_axis.
  - Their response to the real mass change k1.1 (+20.7 to +23.4 points) is 70–80 % of the released model's.
  - At epoch 0, two kin seeds had moved W→Z in the opposite direction (−7.8, −9.9). By epoch 3 all three moved to positive values.
- **What the epoch-3 numbers can and cannot say.** The axis response of the kin seeds is changing with training, so these numbers do not yet separate "seed / data-scale dependence" from "not yet trained". The kin checkpoints are at 69–70 % accuracy vs 74.2 % for the released model.
- **Seed-to-seed spread at the same epoch.** It is small relative to the gap to the released model: kin R 0.17–0.35 at epoch 3, phi Δ(W→Z) CIs overlapping for s1 and s2.
- **Not measured here:** S1–S4 (read-out calibration, input routes, block-level route, threshold head). The P6 L-GATr comparison can use these checkpoints once final; data size is matched (same 2M files), but samples seen differ (25.6 vs 12.8 passes).

## Issues
- **Launch deviation.** The plan said 3 seeds per GPU in two jobs. That ran out of GPU memory on a 22 GB A10, and no 48 GB GPU was free for 20 min, so the runs are six single-seed jobs instead. One seed saturates a GPU (2.1 steps/s, 85–100 % busy), so the total time is shorter.
- **Schedule deviation.** Samples per epoch are 1/10 of the authors' values (51.2M samples in total); everything else in `train_JetClass.sh` is unchanged.
- **Memory headroom.** Two pods were at 83–87 % of the 12 Gi memory limit (this includes page cache). If a pod is OOM-killed, it can be resumed with `--load-epoch`.
- **Symlinks.** `ext/particle_transformer/datasets/JetClass/Pythia/{train_100M,val_5M,test_20M}` are symlinks to the 2M subset. Any other use of `train_JetClass.sh` from that clone would read the subset.
- **Division warnings.** The battery's ½Δlog(M0M2) prediction gives division warnings for jets with M2 = 0 (single-particle jets). Reported means are over W jets and finite.

## Explainer entry (for site/explain.py)
- **Question:** Is ParT's axis-relative mass response a property of the one released checkpoint (trained on 100M jets), or does every ParT trained on JetClass learn it, including at L-GATr's 2M-jet scale?
- **Background:** C3 found that rotating all particles by 0.06 rad around the stored jet axis leaves the invariant mass $m$ unchanged but raises the axis moment $M_2=\sum_i p_{T,i}\,r_i^2$. Under that edit, the released ParT turns true-W jets into Z (12 → 31 %).
- **Method:**
  - Train ParT from scratch with the authors' script on 2M jets, seeds 0/1/2, full and kinematic inputs.
  - On 10,000 random test jets, measure the W→Z change under the rotation (phi0.06), under a real 10 % mass change (angles ×1.1), and under a whole-jet rotation (null).
  - Compute the ratio $R=\overline{\Delta(\ell_W-\ell_Z)}_{\phi}/\overline{\Delta(\ell_W-\ell_Z)}_{\times1.1}$. Compare it with the axis-moment prediction $\tfrac12\Delta\log(M_0M_2)_{\phi}/\tfrac12\Delta\log(M_0M_2)_{\times1.1}=0.97$ and with 0, the prediction of an invariant-mass tagger.
- **Result (figure `results/part/p10_battery_e3.png`):**
  - At epoch 3 of 50, full-input seed 0 gives +22.6 points (released +22.0), $R=0.94$.
  - The kin seeds give +4 to +9 points, $R=0.17$–$0.35$ (released kin 0.69).
- **Checks:**
  - The released models reproduce their known accuracies (86.0 %, 74.2 %) and the P1 numbers on these random rows.
  - The whole-jet-rotation null stays within 0.5 points.
  - The E-scheme re-axis control changes ≤ 0.3 % of calls.
- **Why it matters:** Mechanistic claims about head indices and the mass route rest on one checkpoint. The final seeds test the pre-registered statements S1–S4 per seed, and they give a ParT matched in data size to L-GATr.
