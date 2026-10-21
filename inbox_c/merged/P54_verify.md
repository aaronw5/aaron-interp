# P54v — verification of P54 (L-GATr equivariant-MLP channel atlas and literal channel circuits)

Script `interp/p54v_verify.py` (imports P54's own functions from `interp/p54_lgatr_channel_atlas.py`). Jobs:
- `c-p54-verify` (exit 0, 54 min, RTX 2080 Ti): atlas, circuit ceiling, steering. Its "extra", "calib" and "ablate" sections stopped on a bug in the verification script (a wrong `compare()` signature, then a missing key that depended on it). That bug is in P54v, not in P54.
- `c-p54-verify2` (exit 0, 47 min, GTX 1080 Ti): the fixed script re-ran atlas (identical numbers), then extra, calib and ablate.

Results: `results/part/p54_verify.json` (atlas, circuit ceiling, steering) and `results/part/p54_verify_b.json` (atlas, extra rotation and pre-gate controls, calibration, block-matched ablation).

## Conclusion

On a fresh set of jets, L-GATr's gated hidden channels again read out a class or physics quantity more often than directions of a randomly rotated basis: 19 % of channels, against at most 14 % for 40 rotations. The excess is absent in the same channels before the gate is applied, which ties it to the per-channel gating. Two of P54's three causal "no" results do not hold as stated:
- The literal-circuit test cannot reach the full tagger even when every hidden channel is kept (36 % accuracy, because the hidden scalars are also replaced). Its near-chance results for small circuits therefore do not show that small channel sets are insufficient.
- The calibration control was mismatched. With matched controls, the 8 best mass-reading channels carry about four times more of the network's response to a real mass change than random channels from the same blocks or rotated directions.

Still uncertain: whether any small channel set reproduces the class calls under a less destructive ablation, and whether steering the four-vector part behaves differently from steering the scalar part.

## What was checked in the code (P54 script)

- **Hook location.** Forward hooks are on the output of each block's `ScalarGatedNonlinearity` in `blk.mlp.layers`, 12 of them, asserted. The `GeoMLP` source (`ext/lorentz-gatr/gatr/layers/mlp/mlp.py`) calls `layer(mv, scalars=s)` for gate → EquiLinear, so the hook output is the gated hidden state that the read-out EquiLinear receives. This is the stated unit.
- **Padding and tokens.**
  - `dense_to_sparse_jet` drops padded slots. The script asserts that each jet has (particle count + 3) tokens and that the global token comes first.
  - Pooling averages only token positions ≥ 3, i.e. the particles. The global token and the two beam/time reference tokens are excluded.
  - No padded particle is counted.
- **Row selection.** Rows are drawn at random from each class file (`rng.choice`, sorted afterwards for reading) and split at random within each class. There is no first-k selection.
- **η build.** The calibration edit uses `part_core.build`, which sets η = jet_eta + sign(jet_eta)·Δη, so the mirroring bug is absent. Unedited and dilated jets are both rebuilt by `build`, so float rounding is the same in both runs.
- **Edits.**
  - Steering adds to the scalar component (index 0) of the chosen channel only, at particle tokens only, in one block.
  - Mean ablation replaces all 16 components of the channel. The direction-ablation null projects the direction out of all 16 components, which matches.
  - Resample ablation takes values from the donor jet's clean run.
  - None of these edits changes more than stated.
- **Problems found in the controls (no bug in the measured quantities):**
  1. **Untied rotations in the atlas.** The rotation applied to the scalar parts (R_0) and the one applied to the four-vector parts (R_1) are independent Haar matrices. A rotated "unit" therefore pairs the scalar of one direction with the four-vector of an unrelated direction, while a channel's f0 and f1 come from the same channel. The test was repeated with tied rotations (one matrix for all grades); see below.
  2. **Calibration rotated-direction control.** Rotated directions were ranked by their atlas log-m score, which used R_1 for the four-vector part, but they were patched along R_0 in all 16 components. The scored and patched directions differ. In addition, P54's random channel sets were not matched to the blocks of the top set, even though damage depends strongly on block depth.
  3. **Ablation null.** P54 compared every channel with the maximum over 30 rotated directions drawn from random blocks. Because damage falls by a factor of about 400 from block 0 to block 11, a pooled maximum mostly reflects which block a direction was drawn from.
  4. **Circuit test.** There is no ceiling: no variant keeps all 384 channels. Without one, a near-chance circuit cannot be told apart from a protocol that is at chance whatever is kept.
  5. **Bootstrap intervals.** The bootstrap interval of the clean fraction sits above its point estimate (0.188 [0.182, 0.219]). The fraction is a thresholded maximum over 17 labels, which a resample of jets biases upward. The trained − rotated difference uses the same draws for both bases, so this bias largely cancels there.
- **Figure** (`results/lgatr/p54_lgatr_channel_atlas.png`, read).
  - Panel (a) shows no interval for the trained, untrained or permuted bars. The light score ≥ 0.3 bars (all near 1) dominate the visual, so the R² ≥ 0.5 difference is hard to see.
  - Panel (c) shows the circuit curves without a ceiling. Given the ceiling below, the panel suggests a stronger negative result than the test supports.

## Sample

JetClass test_20M `<class>_107.root`. P54 used `_106`; `_107` has not been used in loop C. 2,000 random rows per class with numpy default_rng(5455), split at random within class into sample 1 (labelling and ranking, N 10,000) and sample 2 (held out, N 10,000).

Unedited L-GATr (final checkpoint `model_run0.pt`):

| sample | accuracy |
|---|---|
| P1 sample (N 3,000) | 83.73 % (job 1), 83.80 % (job 2; GPU nondeterminism); expected 83.73 % |
| sample 2 | 84.84 % [84.10, 85.55] |
| untrained network | 10.23 % |
| channel-permuted network | 10.07 % |

## 1. Atlas replication (P54's `sec_atlas` unchanged, fresh jets)

Definitions:
- Unit: one of the 12 × 32 = 384 gated hidden multivector channels.
- Features per jet: f0 = particle-mean scalar part; f1 = particle-mean Minkowski square of the four-vector (grade-1) part.
- Clean: held-out R² ≥ 0.5 for one of 7 observables, or 2·AUC − 1 ≥ 0.707 for one class against the rest.
- Selective: clean, and the best label family beats the second-best by ≥ 0.1.
- H0: channel units are no cleaner than rotated directions. H1: they are cleaner.
- p_perm: one-sided permutation of unit membership between the channel basis and the pooled rotated bases (10,000 permutations).

| quantity (R² ≥ 0.5) | P54 (_106) | P54v (_107) |
|---|---|---|
| clean, trained channels | 0.188 | 0.193 [0.180, 0.214] |
| clean, 10 raw rotations, mean [min, max] | 0.129 [0.117, 0.138] | 0.132 [0.115, 0.143] |
| trained − raw (95 % bootstrap CI) | +0.059 [+0.057, +0.082] | +0.061 [+0.051, +0.080] |
| p_perm (clean, raw) | 0.002 | 0.0009 |
| clean, 10 std rotations | 0.091 | 0.088 [0.078, 0.099] |
| clean, untrained / permuted | 0.135 / 0.073 | 0.141 / 0.070 |
| selective, trained vs raw | 0.122 vs 0.092, p 0.036 | 0.128 vs 0.091, p 0.015 |
| auxiliary scalars, clean vs raw rotations | 0.167 vs 0.147–0.190, p 0.62 | 0.171 vs 0.161–0.185, p 0.66 |

- Clean trained channels by label (_107): Tbl 28, Hbb 28, Tbqq 9, Hgg 5, N 2, Hqql 1, log m 1. Untrained network: N 49, log M2/M0 4, Tbqq 1.
- By block: 0.50 and 0.56 in blocks 7–8, 0 in blocks 2, 3 and 11, the same pattern as P54.
- Not replicated at one threshold: "selective vs raw at the untrained-95th-percentile threshold" gives p_perm 0.12 on _107, against 0.008 on _106.

## 2. Added rotation controls (sample 2 of _107, point estimates over rotations)

z = (trained fraction − mean over rotations) / s.d. over rotations.

| comparison (R² ≥ 0.5) | trained | rotations mean [min, max] | z | p_perm |
|---|---|---|---|---|
| clean vs 20 fresh untied raw rotations | 0.193 | 0.124 [0.099, 0.141] | 5.4 | 0.0001 |
| clean vs 10 tied raw rotations (one matrix for all grades) | 0.193 | 0.121 [0.102, 0.130] | 7.5 | 0.0002; bootstrap diff +0.071 [+0.059, +0.084] |
| clean vs 10 tied std rotations | 0.193 | 0.082 [0.062, 0.109] | 8.6 | 0.0001 |
| selective vs tied raw | 0.128 | 0.086 [0.068, 0.096] | 4.0 | 0.0065 |
| auxiliary scalars vs 20 untied raw | 0.171 | 0.177 [0.152, 0.191] | −0.5 | 0.68 |
| **pre-gate** channels (input of the gated GELU) vs 10 untied raw | 0.503 | 0.491 [0.466, 0.516] | 0.7 | 0.36 |
| **pre-gate** channels vs 10 tied raw | 0.503 | 0.492 [0.464, 0.516] | 0.6 | 0.38 |
| **pre-gate** selective vs tied raw | 0.263 | 0.275 [0.237, 0.302] | −0.6 | 0.72 |

Competing hypotheses for the read-out excess, and what the data say:
- (a) **The excess comes from pairing f0 and f1 of one channel.** Predicted no excess against tied rotations. Excluded: z 7.5 against tied rotations.
- (b) **The channel basis is already privileged before the nonlinearity**, for example through the GeometricBilinear weights. Predicted an excess before the gate. Excluded at this sample size: z 0.6–0.7.
- (c) **The excess is produced by the per-channel gate GELU(x_c,0)·x_c.** Consistent with the data: no excess before the gate, an excess after it, and none in the ungated auxiliary scalars.

The gate also lowers overall readability. Before the gate, 50 % of channels are clean (mostly class units: Hbb 59, Tbl 41, Tbqq 34, Hgg 32, N 17). After the gate, 19 % are.

## 3. Steering replication (P54's `sec_steer`, N 1,000 = 100 random sample-2 jets per class)

- Edit: add k × s.d. (k = ±1, 2, 4) to the scalar part of the best channel for each label, or of the best direction of each of 3 raw scalar rotations, at every particle token.
- Metric: share of (jet, step) pairs that move the target class's log-odds l = log p − log(1 − p) in the predicted direction.
- Test: sign-flip permutation, Holm over 17 labels.

| | P54 (_106, N 2,000) | P54v (_107, N 1,000) |
|---|---|---|
| channel > rotated, Holm p < 0.01 | 8 labels | 9 labels (N, tau32, D2, log M2/M0, Hcc, Hgg, Hqql, Tbqq, Tbl) |
| channel < rotated | 7 labels | 6 labels (log pT, tau21, QCD, Hbb, Zqq, Wqq) |
| no difference | 2 labels | 2 labels (log m, H4q) |

- The selected channels are the same on both files for 16 of 17 labels; for Tbqq, b4c7 replaced b4c13.
- Four labels changed category between the files: log m, log M2/M0, Tbqq and H4q. Their channels move the target log-odds by |Δl| < 0.003 at +2 s.d., so their direction share is close to noise.
- The labels with large effects reproduce:
  - Tbl, channel b1c25: 0.861 vs 0.269.
  - tau21, channel b2c20: 0.048 vs 0.375, with Δl = −0.21 at +2 s.d., opposite to the predicted sign.

## 4. Block-matched single-unit ablation (N 1,000, unedited 85.3 %)

- Edit: in each block, every one of the 32 channels, and 8 directions of a tied rotation, is set to its sample-1 mean for its token type (all 16 components).
- Metric: share of jets whose call changes.

| block | channel mean | channel median | direction median | share of channels above the block's direction maximum |
|---|---|---|---|---|
| 0 | 0.613 | 0.740 | 0.838 | 0.00 |
| 1 | 0.273 | 0.273 | 0.345 | 0.03 |
| 2 | 0.121 | 0.083 | 0.152 | 0.09 |
| 4 | 0.079 | 0.036 | 0.060 | 0.25 |
| 7 | 0.022 | 0.018 | 0.016 | 0.28 |
| 11 | 0.001 | 0.001 | 0.002 | 0.06 |

- Mean over the 12 blocks of (channel mean − direction mean) = −0.031. Two-sided permutation within blocks (10,000 permutations): p 0.006. H0: channels and directions within a block are exchangeable.
- If channels and directions were exchangeable, a channel would exceed the maximum of 8 directions with probability 1/9 = 0.11. The observed share, averaged over blocks, is 0.105.
- Reading: within a block, a single channel does no more damage than a random direction. In blocks 0–2 it does less, because a random direction touches all 32 channels at once.
- P54's statement "matched by rotated directions" holds. Its supporting comparison ("no channel exceeds the pooled null maximum 0.89") was not block-matched.

## 5. Circuit ceiling (N 5,000 = 500 random sample-2 jets per class; unedited 85.34 %)

- Protocol (P54's): resample ablation with the donor = the jet's neighbour in jet pT within the batch (a jet of random class). Kept channels run live; all others take the donor's clean-run values.
- Ranking: P54's frozen all-class ranking, computed on _106 sample 1.

| variant | accuracy |
|---|---|
| floor: nothing kept | 10.54 % |
| **all 384 channels kept, 768 hidden scalars resampled** | **35.68 % [34.32, 37.04]** |
| all 384 channels kept, scalars live (plumbing check) | 85.34 % (same call 1.000) |
| only block 0 resampled (channels + scalars), blocks 1–11 live | 9.90 % |
| blocks 1–11 resampled, block 0 live | 18.90 % |
| ranked K = 32 / 64 / 128 / 256 / 320, scalars resampled | 13.4 / 14.7 / 21.2 / 28.8 / 33.1 % |
| random K = 32 / 64 / 128 / 256 / 320, scalars resampled | 10.6 / 10.5 / 11.8 / 12.0 / 14.5 % |
| ranked K = 32 / 64 / 128 / 256 / 320, scalars live | 14.0 / 17.7 / 33.6 / 70.6 / 83.0 % |
| random K = 32 / 64 / 128 / 256 / 320, scalars live | 14.4 / 14.2 / 14.9 / 14.6 / 14.2 % |

- The ranking is ordered by block: the top 64 channels are 30 from block 0, 28 from block 1, 5 from block 2 and 1 from block 3. All block-0 channels are inside the top 91.
- P54's numbers at K ≤ 64 replicate: ranked 14.7 % at K = 64 against 14.25 %.
- In this protocol, resample-ablating block 0 alone sends the tagger to chance. Keeping every channel cannot exceed 36 % unless the scalars are left live.
- The test therefore separates channel sets only on the scale "keep almost all of the early blocks" (70.6 % at K = 256 with live scalars). It does not test whether a small set of later-block channels carries the class information.

## 6. Calibration with matched controls (true W/Z jets of sample 2, N 2,000)

- Edit E4: every particle's (Δη, Δφ) is multiplied by 1.05 about the stored axis, at fixed pT and particle mass. This raises the mean log m by 0.048.
- Read-out: d = logit_W − logit_Z.
- R_patch = Σ Δd(patched) / Σ Δd(dilation). It is the share of the network's response to the real mass change that is carried by the patched units, when only those units take their values from the dilated jet's run.

| set | K = 8 | K = 32 |
|---|---|---|
| top-K log-m channels (held-out score on _107) | 0.162 [0.156, 0.168] | 0.326 [0.318, 0.335] |
| 10 uniformly random channel sets (P54's control) | 0.036 [0.003, 0.087] | 0.079 [0.022, 0.196] |
| 10 block-matched random sets (same channels per block as the top set) | 0.030 [0.005, 0.089]; diff +0.132 [+0.126, +0.137] | 0.167 [0.058, 0.354]; 1 of 10 ≥ top; diff +0.159 [+0.154, +0.164] |
| top-K log-m directions of 3 tied rotations (scored = patched) | 0.043 [0.031, 0.053] | 0.187 [0.162, 0.222] |
| P54's untied rotation-0 directions (mismatched) | 0.104 (P54: 0.106) | 0.213 (P54: 0.213) |
| all 384 channels | 0.662 [0.653, 0.671] | |

- At K = 8, the channel advantage over directions is larger with the matched control (0.162 vs 0.043) than P54 reported (0.173 vs 0.106).
- At K = 32, the advantage over block-matched random sets is about half of that over uniform random sets, and one block-matched set reaches the top set.

## Corrected reading

1. **Read-out atlas.** The gated multivector channels are cleaner than directions of a rotated basis. Replicated on fresh jets: 0.193 vs 0.132; 40 rotations in total, tied and untied, z 5.4–8.6. Before the gate there is no excess (0.503 vs 0.49), and the ungated auxiliary scalars show none. The read-out privilege is therefore produced at the per-channel gate.
2. **Steering.** Replicated in aggregate (9 higher / 6 lower / 2 tied vs P54's 8 / 7 / 2). Most selected channels move the class log-odds by less than 0.03, and the label-level outcome changes between files for the labels with near-zero effect.
3. **Mean ablation.** Within a block, single channels do no more damage than random tied directions; in blocks 0–2 they do less (p 0.006). Damage falls with depth. P54's pooled-null comparison is replaced by this block-matched one.
4. **Literal circuits.** The numbers replicate, but the protocol caps accuracy at 36 % even with every channel kept, and resample-ablating block 0 alone reaches chance. The statement "circuits of ≤ 64 channels stay near the floor" is correct. The inference that the channels cannot be isolated into a circuit is not supported by this test.
5. **Calibration.** The top-8 log-m channels carry 0.16 of the real-mass response, against 0.03 for block-matched random channels and 0.04 for matched rotated directions. At K = 32 the margin over block-matched sets narrows to 0.33 vs 0.17.

## Notes for the ledger

- Thesis note for C25 (privileged basis from per-channel gating): the read-out part is supported with a new control (no excess before the gate; tied rotations). The causal part is mixed:
  - Steering: about half of the labels favour channels.
  - Ablation: no advantage for channels.
  - Calibration: an advantage for the mass-carrying channels at K = 8 against matched controls.
  - Circuits: not informative under this protocol.
- Suggested next test: a literal-circuit protocol that keeps the hidden scalars and block 0 live (or mean-ablates instead of resampling from random-class donors), ranking channels within blocks 4–10.
