# P54 — L-GATr equivariant-MLP channel atlas and literal channel circuits (loop C, round C12, exploratory; claim C25)

Script `interp/p54_lgatr_channel_atlas.py`; job `c-p54-run` (exit 0, 96 min, TITAN Xp; smoke pass in the same process first) and `c-p54-fig` (figure only, CPU).
Results `results/lgatr/p54_lgatr_channel_atlas.json`, `p54_arrays.npz`, figure `results/lgatr/p54_lgatr_channel_atlas.png`.

## Conclusion

Taken one at a time, L-GATr's hidden MLP channels read as a single class or observable somewhat more often than directions of a randomly rotated channel basis. At held-out R² ≥ 0.5, 18.8 % of channels qualify against 11.7–13.8 % across 10 rotations. An untrained network already reaches 13.5 %, almost all of it from the particle count N. Acting on single channels shows no consistent advantage of the channel basis: pushing the best channel for a label moves the matching class score the predicted way more often than the best rotated direction for 8 of 17 labels, less often for 7, and about as often for the remaining 2. Small "keep only these channels" circuits stay near chance (64 of 384 channels: 14.3 % accuracy against a 10.1 % floor and 84.1 % unedited). So the channel basis is somewhat privileged for reading out information, but not for intervening on it. Per-channel gating does not give L-GATr the sparse, readable units seen in JEDI. Two things remain open. Steering touched only the scalar part of each channel. In the circuit test every other hidden channel and every hidden scalar was replaced at once, which may sit below what any small set can rescue.

Terms used here. "Channel" means one of the 32 hidden multivector channels in a block's MLP, taken after the gated GELU, so 12 × 32 = 384 in total. A multivector is the 16-component object of the space-time algebra: grade 0 is a scalar and grade 1 is a four-vector. "Gated GELU" means channel c becomes GELU(x_c,0)·x_c, where the gate is the channel's own scalar part. A "rotated direction" is one column of a random orthogonal matrix that mixes the 32 channels within each grade. Such a matrix is an equally valid basis for the same computation, because the next linear layer could undo it.

## Thesis and alternatives (as pre-registered in the plan)

- **Privileged basis from per-channel gating.** Predicted three things: (a) trained clean and selective fractions above every rotation, (b) direction-correct steering above rotated directions (p < 0.01 after Holm), and (c) a circuit of ≤ 32 channels keeping most of the class accuracy. Result: (a) holds for the multivector channels at all three thresholds except the lenient one with raw rotations. (b) holds for 8 of 17 labels, is reversed for 7, and shows no difference for 2. (c) fails.
- **Distributed code, as in ParT's P40.** Predicted: fractions inside the rotation range, steering no better than rotated directions, small circuits at the floor. Result: (a) is excluded for the multivector channels. For the 768 auxiliary scalar neurons it holds against raw rotations: 16.7 % vs 14.7–19.0 %, as in ParT. (b) matches overall. (c) matches.
- **Net effect on the thesis:** refines it. The privilege shows in read-out statistics but not under interventions.

## Procedure

**Model.** L-GATr final checkpoint `model_run0.pt` (run anrunw-ai-c-p5-lgatr2, authors' jctagging config, 12 blocks). Each block's MLP is GeometricBilinear → gated GELU (32 multivector channels plus 64 auxiliary scalars with a plain GELU) → EquiLinear. Hooks sit on the output of the gated nonlinearity.

**Unedited accuracy.**

| Sample | N | Accuracy |
|---|---|---|
| P1 sample (rows 3,000–3,300 of each _100 test file) | 3,000 | 83.77 % (expected 83.73 %) |
| P54 sample 1 | 10,000 | 85.06 % |
| P54 sample 2 | 10,000 | 84.14 % [83.40, 84.78] |

**Controls.**
- Untrained L-GATr: the trained run's config, torch seed 5454, no weights loaded. Accuracy on sample 2 is 10.56 %.
- Channel-permuted L-GATr: trained weights, but in every block the 32 hidden multivector channels and the 64 hidden scalars are permuted (seed 5454 + block) between the gated GELU and the read-out EquiLinear. Each channel is therefore read out by another channel's weights. Accuracy is 9.84 %.
- Rotations: 10 random grade-preserving rotations per block. Each grade g gets an independent Haar orthogonal 32 × 32 matrix R_g that mixes the channels' grade-g components; the same R_1 acts on t, x, y and z. A 64 × 64 Haar matrix mixes the auxiliary scalars. "Raw" rotations act on the activations directly. "Std" rotations act after each channel is scaled to unit size on sample 1.

**Sample.** JetClass test_20M `<class>_106.root`. val_5M has only _120–_124, all already used in loop C; test_20M _106 had been used only for ParT's W and Z jets in P13. 2,000 random rows per class were drawn with default_rng(5454), never first-k. Within each class they were split at random into sample 1 (labelling and ranking, 1,000 per class, N 10,000) and sample 2 (held-out evaluation, N 10,000).

**Step 1: atlas.**
- Features per jet, block and channel c:
  - f0 = mean over particle tokens of the scalar part x_c,0.
  - f1 = mean over particle tokens of the Minkowski square of the grade-1 part, x_t² − x_x² − x_y² − x_z².
  - For each auxiliary scalar unit, the mean over particle tokens.
- Every feature and observable is mapped to normal scores through sample 1's distribution. This is a monotone map, so R² then behaves like a squared rank correlation.
- Labels (17):
  - log m: invariant mass of the particles.
  - log pT.
  - N: number of particles.
  - tau21, tau32: N-subjettiness ratios τ2/τ1 and τ3/τ2. Small values mean the jet looks two-prong (τ21) or three-prong (τ32).
  - D2: energy-correlation ratio e3/e2³ with β = 1; small values mean two-prong.
  - log M2/M0: M2 = Σ pT r² about the stored jet axis and M0 = Σ pT, so this is the pT-weighted mean squared angular distance from the axis.
  - The 10 classes, each one vs rest.
- Fit: for each unit and label, a least-squares fit on sample 1 of the label on the unit's features plus an intercept, evaluated on sample 2.
  - Observable score = √max(R², 0).
  - Class score = 2·AUC − 1.
- Definitions:
  - Clean: best score ≥ √0.5, i.e. held-out R² ≥ 0.5, or 2AUC − 1 ≥ 0.707 for a class. Also reported at score ≥ 0.3, and at the untrained network's 95th percentile (0.778 for multivector channels, 0.625 for auxiliary scalars).
  - Selective: clean, and the best label's family beats the second-best family by ≥ 0.1. Families are mass, pT, count, shape (tau21, tau32, D2), width (M2/M0), and each class on its own.
- Statistics:
  - Effect = trained fraction minus the mean rotated fraction.
  - 95 % bootstrap over sample-2 jets: 200 draws, fitted maps held fixed, the same draws for every basis.
  - H0: channel units are no cleaner than rotated directions. Tested by (i) the rank among 10 rotations (smallest possible p = 1/11 = 0.091) and (ii) a permutation of unit membership between the trained basis and the pooled rotated bases (10,000 permutations, one-sided).

**Step 2: steering and ablation.** Sample: 200 random sample-2 jets per class (N 2,000).

- Steering.
  - For each label, take the channel whose f0 scores highest on sample 1 (all 384 channels). As control, take the highest-scoring direction for the same label in each of 3 raw grade-0 rotations (all blocks).
  - Edit: at every particle token, add k·sd to that channel's scalar part (or k·sd_u·u to the 32 scalar parts, for a direction u). k = ±1, 2, 4 and sd is the s.d. of the pooled feature on sample 1, so the jet-level feature moves by exactly k s.d.
  - Target class c*:
    - For a class label, the class itself; the predicted sign is the sign of the unit's score.
    - For an observable o, the class in which o is most shifted in the direction the unit follows: argmax_c σ·(AUC_c(o) − 0.5), where σ is the sign of the unit's correlation with o. The predicted sign is +.
  - Read-out: l_c* = log p_c* − log(1 − p_c*).
  - A step is direction-correct when (predicted sign) × sign(k) × Δl_c* > 0.
  - Metric: share of correct (jet, step) pairs, channel minus the mean of the 3 directions.
  - Statistics: bootstrap over jets, sign-flip permutation p over jets (5,000), McNemar on (jet, step) pairs against direction 1, Holm over 17 labels.
- Mean ablation.
  - Each of the 384 channels in turn: its whole multivector is set to its sample-1 mean for its token type (global, reference 1, reference 2, particle).
  - Metrics: change in accuracy, McNemar, Holm over 384.
  - Null: 30 random raw-rotation directions, each with its component set to its mean.

**Step 3: literal circuits.**
- Ranking (sample 1, 500 jets per class, then frozen): channels ordered by mean |a|, where a = Σ_tokens,components (h_donor − h)·∂l_k/∂h. This is the first-order effect of resampling the channel on class k's log-odds.
- Evaluation (sample 2, N 10,000): keep the top K channels (K = 8, 16, 32, 64 of 384). Every other hidden multivector channel, and every hidden auxiliary scalar of every block, is resample-ablated: replaced by its value in the clean run of a pT-matched donor jet (the batch neighbour in jet pT, with token k taken from donor token k mod n_donor). Attention, layer norms and the residual stream run live.
- Comparisons: 3 random channel sets of the same size; the floor (no channel kept); a variant with the auxiliary scalars left live (K = 32); one all-class ranking (sum over k).

**Calibration (E4).**
- Edit: every particle's (Δη, Δφ) × 1.05 about the stored axis, at fixed pT and particle mass. This raises the mean log m by 0.048.
- Jets: the true W and Z jets of sample 2 (N 2,000).
- Read-out: d = l_W − l_Z, in logits.
- Patch: the chosen channels take, at all tokens, their activations from the dilated jet's run.
- R_patch = Σ Δd(patch) / Σ Δd(dilation), the share of the network's response to a real mass change that the patched channels carry.

## Numbers

**Atlas, 384 multivector channels.** All rows use 95 % bootstrap intervals over jets.

| threshold | trained clean | raw rotations mean [min, max] | std rotations mean [min, max] | untrained | channel-permuted | trained − raw (95 % CI) | p_rank / p_perm |
|---|---|---|---|---|---|---|---|
| R² ≥ 0.5 | 0.188 [0.182, 0.219] | 0.129 [0.117, 0.138] | 0.091 [0.078, 0.107] | 0.135 | 0.073 | +0.059 [+0.057, +0.082] | 0.091 / 0.002 |
| score ≥ 0.3 | 0.997 | 0.992 [0.982, 0.997] | 0.985 | 0.948 | 0.940 | +0.006 [−0.002, +0.008] | 0.27 / 0.20 |
| untrained 95th pct (0.778) | 0.063 | 0.029 [0.018, 0.037] | 0.018 | 0.052 | 0.013 | +0.033 [+0.020, +0.039] | 0.091 / 0.002 |

| threshold | trained selective | raw rotations mean [min, max] | std rotations | untrained | trained − raw (95 % CI) | p_perm (raw / std) |
|---|---|---|---|---|---|---|
| R² ≥ 0.5 | 0.122 [0.112, 0.141] | 0.092 [0.076, 0.107] | 0.069 | 0.125 | +0.031 [+0.022, +0.051] | 0.036 / 0.0006 |
| score ≥ 0.3 | 0.388 | 0.363 [0.320, 0.398] | 0.353 | 0.263 | +0.025 [+0.002, +0.047] | 0.18 / 0.095 |
| untrained 95th pct | 0.049 | 0.025 [0.013, 0.034] | 0.017 | 0.052 | +0.024 [+0.012, +0.031] | 0.008 / 0.0003 |

- Labels of the clean trained channels at R² ≥ 0.5: Tbl 30, Hbb 25, Tbqq 10, Hgg 4, N 2, Hqql 1. For the untrained network: N 49, log M2/M0 3.
- No channel reaches held-out R² ≥ 0.5 for log m. The best log-m channels are b0c0, b0c19 and b5c9 (score 0.70, 0.69, 0.69).
- Clean channels by block (R² ≥ 0.5): blocks 7–8 have 0.47–0.59; blocks 2, 3 and 11 have 0.
- For comparison, ParT's class-token MLP neurons in P40 (Spearman score, untrained-95 threshold) had selective fraction 0.488 against 0.49–0.50 for rotations.

**Atlas, 768 auxiliary scalar neurons** (plain GELU, no gate).

| threshold | trained clean | raw rotations [min, max] | p_perm | std rotations | p_perm |
|---|---|---|---|---|---|
| R² ≥ 0.5 | 0.167 | 0.170 [0.147, 0.190] | 0.62 | 0.109 | 0.0001 |

- Selective at R² ≥ 0.5: 0.129 against raw 0.139 [0.120, 0.159] and std 0.093.
- Untrained 0.010; channel-permuted 0.049.

**Steering (N 2,000; share of direction-correct steps, top channel vs mean of the 3 top rotated directions).**
- The channel is higher with Holm p < 0.01 for 8 labels:

| label | channel | channel share | rotated share |
|---|---|---|---|
| log m | b5c9 | 0.462 | 0.401 |
| N | b1c26 | 0.485 | 0.336 |
| tau32 | b2c25 | 0.820 | 0.655 |
| D2 | b0c31 | 0.595 | 0.387 |
| Hcc | b7c13 | 0.999 | 0.794 |
| Hgg | b10c13 | 0.887 | 0.583 |
| Hqql | b1c25 | 0.963 | 0.930 |
| Tbl | b1c25 | 0.860 | 0.267 |

- The channel is lower for 7 labels: log pT, tau21, QCD, Hbb, H4q, Zqq and Wqq. The difference is not significant for 2: log M2/M0 (+0.015 [−0.009, +0.038]) and Tbqq (+0.004). Examples:
  - tau21 channel b2c20: 0.043 against 0.372. It moves the Tbl log-odds by +0.42 at −4 s.d., which is opposite to the sign predicted from its tau21 correlation.
  - Wqq channel b6c21: 0.448 against 0.846.
- Size of the moves: most labelled channels shift the target log-odds by |Δl| ≤ 0.03 at ±4 s.d. Exceptions are tau21 (0.42), tau32 (0.19), Hqql (0.30), Tbl (0.13) and Hcc (0.10). The log-m channel b5c9 moves it by < 0.001, so its share is not informative.

**Mean ablation (N 2,000, unedited 84.85 %).**
- 119 of 384 channels change accuracy at Holm p < 0.05.
- Median Δacc −0.006; median calls changed 0.029, 90th percentile 0.36.
- Rotated-direction null: median calls changed 0.021, maximum 0.89.
- No channel changes more calls than the most damaging rotated direction.
- Mean calls changed by block 0 → 11: 0.61, 0.27, 0.11, 0.06, 0.08, 0.05, 0.06, 0.03, 0.01, 0.01, 0.02, 0.001. Single block-0 channels (b0c6, b0c5, b0c4, b0c0) lower accuracy by 0.67–0.74.

**Literal circuits (sample 2, N 10,000; unedited 84.14 %, mean recall 0.841).**
- Floor (all hidden channels and scalars resampled): accuracy 10.08 % [9.49, 10.69].

| K | own-class recall, mean | random sets | own-class AUC | random AUC | all-class circuit accuracy | random-set accuracy |
|---|---|---|---|---|---|---|
| 8 | 0.101 | 0.101 | 0.519 | 0.517 | 10.10 % | 10.0–10.2 % |
| 16 | 0.110 | 0.102 | 0.536 | 0.518 | 10.97 % | 10.1–10.3 % |
| 32 | 0.128 | 0.104 | 0.557 | 0.519 | 12.78 % | 10.2–10.7 % |
| 64 | 0.144 | 0.100 | 0.594 | 0.515 | 14.25 % [13.58, 14.96] | 9.9–10.2 % |

- With the auxiliary scalars left live: floor 13.56 %, class circuits at K = 32 14.11 %, random sets 13.0–14.6 %.
- The attribution ranking is almost the same for every class. It is dominated by block-0 channels (b0c31, b0c6, b0c3, b0c0, b0c4 at the top for all 10 classes). At K = 8 and K = 32 all ten class circuits are the same channel set, so the "class-k circuits" are not class-specific.

**Calibration (E4, N 2,000 W and Z jets).** Δd under the dilation is −0.484 [−0.519, −0.449] logits.

| patched set | R_patch | random K-sets mean [min, max] | top-K log-m rotated directions (raw rotation 0) |
|---|---|---|---|
| top-1 log-m channel (b0c0) | 0.023 [0.021, 0.025] | 0.003 [−0.000, 0.017] | 0.002 |
| top-8 | 0.173 [0.166, 0.179] | 0.026 [0.001, 0.104] | 0.106 |
| top-32 | 0.324 [0.315, 0.332] | 0.063 [−0.002, 0.139] | 0.213 |
| all 384 channels | 0.648 | – | – |

- The top-8 log-m channels are b0c0, b0c19, b5c9, b5c16, b5c3, b1c26, b0c30 and b0c6.

## Reading

- **Read-out.** In the multivector channels, read-out statistics favour the channel basis. Rotating the 32 channels within each grade lowers the clean fraction at R² ≥ 0.5 from 0.188 to 0.12–0.14 (raw rotations) and to 0.08–0.11 (std rotations). The trained clean set is mostly class units for Tbl, Hbb and Tbqq in blocks 4–8. The untrained network's clean channels are count (N) units, so the trained-over-rotation excess is in class labels. Rotation also lowers the class-labelled counts (Hbb 25 → 14, Tbl 30 → 19 in rotation 0).
- **Gated vs ungated units.** The ungated auxiliary scalars show no excess over raw rotations (0.167 against 0.147–0.190), which is the P40 pattern for ParT's GELU neurons. Within this one network, then, the gated multivector channels differ from the ungated scalar neurons in this statistic. The std-rotation comparison is lower for both unit types; that partly reflects the scale choice.
- **Interventions.** These do not follow the read-out ordering. Steering the scalar part of the best channel changes the target log-odds by small amounts for most labels. The direction-correct share is higher than for rotated directions for 8 of 17 labels and lower for 7. Mean-ablation damage depends mainly on block depth (blocks 0–1 dominate), and rotated directions in the same blocks are as damaging as single channels.
- **Circuits.** Keeping ≤ 64 of 384 channels while resampling everything else in the MLPs leaves accuracy within 4.2 points of the 10.1 % floor. The attribution ranking does not separate classes.
- **Calibration.** Patching the top log-m channels carries 0.17 (8 channels) and 0.32 (32 channels) of the real-mass-change response. This is above random channel sets and above the same number of top log-m rotated directions (0.11, 0.21).

## Issues and limits

- Steering edits only the grade-0 (scalar) part of a channel. The grade-1 Minkowski feature f1 is used in the atlas but not steered.
- The circuit test resamples all 768 auxiliary scalars together with the dropped channels. The floor is at chance, so this test cannot separate "no small circuit exists" from "MLP hidden units are needed jointly".
- The attribution ranking (mean |first-order effect|) picks the same fragile block-0 channels for every class. A class-specific ranking, such as a signed effect on l_k minus the mean over other classes, was not tried.
- p_rank is bounded below by 1/11 with 10 rotations. The permutation p treats units as independent.
- Selective-fraction values are not on the same scale as P40's (held-out R² / AUC here, Spearman with a BH threshold there). Only the trained-vs-rotation comparison within each study carries over.
- The figure was produced by a separate CPU job (`c-p54-fig`): the main job's figure call failed on matplotlib usetex (missing mathdesign.sty). That is fixed in the script (`text.usetex = False`).
- The smoke pass (150 jets) ran in the same process before the full run and passed.

## Suggested verifier (P54v)

1. Fresh file (test_20M *_107).
2. Steer f1 by scaling the grade-1 part.
3. Use a class-specific signed ranking for the circuits, with the auxiliary scalars resampled only in the blocks where channels are dropped.
4. Compare with circuits built from the top-K rotated directions.
5. Repeat the atlas at the GeometricBilinear output (before the gate) to see whether the excess is created by the gate itself.
