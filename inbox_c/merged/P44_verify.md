# P44v: skeptical check of P44 (L-GATr channel grades, symmetry-breaking inputs, mass read-out)

## Conclusion
In our Lorentz-equivariant tagger (L-GATr), the class decision runs through its Lorentz-invariant channels. These are the 16 pure-scalar parts of the multivector channels ("grade 0") and 32 extra scalar channels. The two sets hold overlapping copies of the same information. Blanking grade 0 alone costs at most about 4 accuracy points, but blanking it after the extra scalars are already blanked costs a further 6–43 points. So P44's statement that grade 0 carries little of the decision does not hold. The boost results reproduce on fresh jets: a boost along the beam changes about 3 % of calls, and this happens only through the time reference vector and the log-energy features, which push the output in roughly opposite directions. The mass is linearly readable from the invariant channels, better in the trained network than in an untrained copy, and better from the extra scalars than from grade 0. P44 did not test the pre-registered axis-moment question (claim C6), so it gives no support for C6.

## What was checked
- **Model and sample.** Same interim checkpoint, `model_run0_it154999.pt`. Fresh sample: JetClass val_5M `<class>_120.root`; HToGG uses `_124` because `_120` is L-GATr's validation file for that class (`selection.json`). P44 used `_122`. 500 random rows per class, `default_rng(7777)`, N 5,000, shuffled. Script `interp/p44v_lgatr_verify.py`, job **c-p44-verify** (GTX 1080 Ti, 81 min, of which ~63 min was imports and model loading from CephFS). A second CPU job, **c-p44-verify-an** (`interp/p44v_analyze.py`), ran on P44's own per-jet logits (`results/lgatr/p44_main.npz`, N 10,000). Results are in `results/part/p44_verify.json` and `results/part/p44_verify_an.json`. The log is `logs/c-p44-verify.log`.
- **Unedited model.** Accuracy 85.20 % [84.22, 86.20] (stored inputs 85.24 %). An identical rerun changes 0.00 % of calls (mean |Δlogit| 0.0035). P44 reported 84.80 % on `_122`.

## Code review of interp/p44_lgatr_grades.py
- **Blade indices.** The script groups the 16 multivector components as g0 = [0], g1 = [1–4], g2 = [5–10], g3 = [11–14], g4 = [15]. This matches the lorentz-gatr blade order (1, e0, e1, e2, e3, e01, …, e0123). The script asserts that the beam spurion is index 8 (e12, the x–y plane) and the time spurion is index 1 (e0); both asserts pass on the cluster.
- **Hooks.** Activations are taken from the outputs of `net.blocks[j]`, using the same attention mask as the model's own forward. Split forward vs the wrapper's forward: max |Δlogit| 0.20, with calls equal to the float floor.
- **Padding and row selection.** `dense_to_sparse_jet` drops padded slots, so no padded particles enter. Rows are random per class and then shuffled; there is no first-k selection. The jets are rebuilt with `part_core.build`, which uses η = jet_eta + sign(jet_eta)·Δη (the fixed version). Rebuilt and stored inputs give the same accuracy (84.80 vs 84.82 %).
- **Boost edit.** The boost is an exact transform of E and pz. However, a longitudinal boost changes more input scalars than P44 states. Max |change| in the preprocessed inputs at y = 0.5: log E 0.35, log E/E_jet 0.53, **Δη 0.35, ΔR 0.93**. Pseudorapidity is not additive for massive particles or for the massive jet axis, so the angular features about the particle-sum axis also move. Test below: the angular columns do not carry the response; log E does.
- **Control design (limits, not bugs).**
  - P44's random-component control draws from all 288 components. At block 0 it hits the input four-momenta in the residual stream, so any 16 random components leave 18 %. That control cannot tell "grade 0 is unimportant" apart from "grade 0 is not one of the momentum components".
  - Mean ablation of covariant components (one sample mean per component) is not a Lorentz-covariant operation. A g1/g3 drop at blocks 0–2 therefore measures both lost information and an off-manifold input.
- **Claim IDs.** Claim "C21" does not appear in PAPERS.md, TREE.md or EXPERIMENTS.md. Claim C6 in PAPERS.md reads: "L-GATr responds to iso-invariant-mass edits that change M2 with read-out ratio ≈ 0". M2 is the pT-weighted angular spread about the jet axis. The pre-registered test for C6 is P6, which P44 did not run.

## Reproduction on the fresh sample (mean / resample ablation, accuracy %, N 5,000; P44 value in brackets)
Unedited accuracy: 85.2 % here, 84.8 % for P44.

| group | b0 | b2 | b3 | b5 | b8 | b11 |
|---|---|---|---|---|---|---|
| s (32 aux. scalars) | 53.5 / 47.3 (53) | 50.5 / 31.0 (50) | 41.0 / 30.6 (40) | 55.2 / 33.5 (55) | 62.2 / 40.6 (62) | 69.9 / 44.0 (70) |
| g0 (16) | 83.8 / 83.5 (84) | 84.3 / 83.8 (84) | 84.1 / 83.1 (84) | 83.8 / 81.8 (84) | 83.0 / 77.4 (83) | 81.5 / 67.8 (82) |
| g1 (64) | 13.9 (14) | 23.6 (24) | 56.0 (57) | 73.1 (73) | 80.6 (81) | 85.2 (85) |
| g3 (64) | 19.5 (19) | 20.7 (21) | 57.9 (58) | 69.8 (70) | 82.4 (83) | 85.2 (85) |
| g2 / g4 | 82.7 / 84.6 | 82.9 / 83.9 | 84.1 / 83.4 | 84.8 / 83.5 | 84.8 / 83.9 | 85.2 / 83.3 |
| inv = g0 + s (48) | 46.8 (47) | 41.9 (42) | 25.0 (25) | 30.4 (31) | 28.6 (29) | 26.1 (26) |

P44's grade table reproduces to within 1 point. Boost, calls changed at y = +0.5 (P44 value in brackets):
- A (physical edit): 2.84 % [2.38, 3.32] (2.58).
- C (exact symmetry): 0.14 % [0.04, 0.26] (0.11).
- D (time-reference route): 7.62 % (7.89).
- F (feature route): 8.80 % (8.39).
- Rigid φ + 0.06: 0.10 %.

## New controls

### 1. Grade 0 vs the auxiliary scalars: redundancy (on P44's own logits, N 10,000)
- **H0:** g0 and s act independently. Removing both costs the sum of the two separate costs (interaction I = 0).
- **H1:** each stands in for the other (I > 0). Under H1, removing g0 once s is gone costs much more than removing g0 alone.
- **Interaction.** I = (drop with g0 and s both removed) − (drop with s removed) − (drop with g0 removed), in accuracy points, with a paired bootstrap over jets.
- **Cost of g0 given s removed.** Accuracy with s removed minus accuracy with g0 and s removed.

| block | drop g0 alone (mean abl.) | g0 cost given s removed (mean abl.) | interaction I | g0 cost given s removed (resample) | McNemar (s vs inv) |
|---|---|---|---|---|---|
| 0 | 1.3 | 6.1 [5.5, 6.8] | 4.9 [4.1, 5.7] | 8.9 [8.2, 9.5] | 878 lost / 265 gained, p 4e-77 |
| 3 | 1.1 | 14.9 [14.0, 15.9] | 13.9 [12.8, 14.9] | 13.9 | p 2e-214 |
| 5 | 0.8 | 24.9 [23.9, 25.8] | 24.0 [23.0, 25.1] | 19.5 | p < 1e-300 |
| 8 | 1.5 | 32.7 [31.7, 33.7] | 31.3 [30.2, 32.3] | 27.1 | p < 1e-300 |
| 11 | 3.2 | 43.3 [42.3, 44.3] | 40.1 [39.1, 41.2] | 30.4 | p < 1e-300 |

The interaction is positive at all 12 blocks, with every CI excluding 0, so H0 is excluded. Grade 0 holds class information that overlaps with s. Mean-ablating g0 alone is cheap because s still carries that information. With s removed, g0 is worth 6–43 points.

### 2. Type-matched control for grade 0 (fresh sample)
16 random channels out of the 32 auxiliary scalars, the same count as g0; 8 draws per block. Accuracy is g0 vs the mean over draws, with a paired sign-flip test and Holm over 12 tests.
- Mean ablation: g0 is 2.9–4.8 points above the s-half control at blocks 0–5, for example b3 84.1 vs 79.2 %, difference +4.8 [+4.1, +5.5], Holm p 0.002. At b8 the difference is +0.9 [+0.2, +1.5] (Holm p 0.02) and at b11 +1.2 [+0.4, +2.1] (Holm p 0.015).
- Resample: g0 is above the s-half control at blocks 0–5 (+7 to +11 points). The two are equal at b8 (p 0.84). At b11, g0 is **below** it: 67.8 vs 71.4 %, difference −3.7 [−4.8, −2.5], Holm p 0.002.
- Per channel, grade 0 is somewhat less needed than the auxiliary scalars in early blocks and about equally needed in the last blocks.

Variance share (across particle tokens, of all 288 components): g0 0.000–0.001, s 0.001–0.003, g1 0.50–0.59, g3 0.27–0.48, g2 0.02–0.19. Most activation variance sits in the frame-dependent vector channels. Ablation size therefore does not follow variance.

### 3. Mass read-out split by channel type (ridge of log m, 2,500 train / 2,500 test jets)
The probe is calibrated by an angular dilation: all Δη, Δφ × 1.1 about the stored axis, true Δlog m = +0.094. "Gain" = probe change / true change under the dilation.

| features | trained R² b0 → b5 → b11 | trained gain b0 / b5 / b11 | untrained R² b0 → b5 → b11 | untrained gain |
|---|---|---|---|---|
| g0 only (32) | 0.950 → 0.838 → 0.735 | 0.94 / 0.80 / 0.66 | 0.735 → 0.572 → 0.585 | 0.87 / 0.64 / 0.69 |
| s only (64) | 0.966 → 0.936 → 0.879 | 0.97 / 0.95 / 0.93 | 0.833 → 0.771 → 0.708 | 0.98 / 0.84 / 0.78 |
| inv (96) | 0.975 → 0.958 → 0.906 | 0.97 / 0.99 / 0.95 | 0.864 → 0.802 → 0.763 | 1.00 / 0.89 / 0.83 |

- The mass is read better from s than from g0 at every block.
- The trained network beats the untrained copy at every block: unexplained variance 1 − R² is 0.025–0.094 vs 0.136–0.237 for inv. Gains differ less.
- P44's "the mass is just as readable from an untrained copy" is not supported. Readable, yes, but with 2.5–5.4 times the unexplained variance.

### 4. Boost routes: which feature, and do the routes cancel?
Edits at y = ±0.5:
- F_logE: only the two log-E input columns are taken from the boosted jet.
- F_rest: only log pT, log pT/pT_jet, Δφ, Δη and ΔR are taken from the boosted jet.

Results:
- **Which feature.** F_logE changes 8.92 % / 8.42 % of calls (= F, 8.80 / 8.36 %). F_rest changes 0.16 % / 0.30 %, near the floor. The feature route is the log-E features, as P44 stated; the moving Δη/ΔR columns contribute nothing measurable.
- **Do the routes cancel?** Per-jet cosine between the logit changes of D and F, with logits centred per jet: −0.60 [−0.61, −0.59] at both signs of y; 92 % of jets are negative. So the two routes push roughly opposite ways, which supports P44's "partly cancel".
- **They are not additive.** ‖ΔD + ΔF‖ is 2.2 × ‖ΔA‖, and the residual ‖ΔD + ΔF − ΔA‖ is 0.91–0.96 of ‖ΔD‖. Both routes raise W → Z: base 13.6 %, D 17.0 / 22.4 %, F 27.8 / 22.4 %. A part of the response is even in the mismatch between the two energy inputs. "Partly cancel" holds for the logit direction and not as a linear sum.
- **Sign of y.** Cosine of D(+y) with D(−y) is −0.54, and of F(+y) with F(−y) −0.39. Mostly odd in y, with an even part.

## Assessment of the reading
1. *"Grade-0 components carry little of the decision."* **Excluded as a causal statement** (control 1: g0 is worth 6–43 points once s is removed; resampled at b11 it costs 17 points). Corrected: the invariant decision information is held jointly by g0 and s with large overlap. s is the larger share per channel in early blocks; in the last blocks the per-channel share is about equal.
2. *"The aux scalars, and g1/g3 in blocks 0–2, carry most."* The s part holds. The g1/g3 part holds as an absolute drop (14–24 % accuracy), but it equals P44's random equal-size control (+1.6 points at b0). Mean-ablating covariant components also removes the input four-momenta and builds non-covariant inputs. The g1/g3 drop does not single out these channels beyond "the momentum embedding is destroyed".
3. *Boost via the time spurion and log E, partly cancelling; C at float level.* Reproduced (A 2.8 %, C 0.12–0.14 %, F = F_logE). C at float level checks that the implementation is equivariant; it holds by construction, not by training. The cancellation is in direction (cos −0.6), not additive.
4. *Mass read-out "with gain ≈ 1, but an untrained copy has it too."* Partly holds: an untrained copy has a linear read-out, but a worse one (R² 0.76–0.86 vs 0.91–0.98).
5. *Thesis note "C6 is supported" and "no dependence on a jet axis as in ParT."* **Not supported by P44.**
   - C6 concerns iso-invariant-mass edits that change M2 about the axis, which P44 did not run (P6 is the test).
   - L-GATr's inputs do include Δη, Δφ, ΔR about the particle-sum axis. The φ shift is an exact symmetry for L-GATr because that axis is recomputed from the particles, so its null result says nothing about learned axis dependence.
   - What P44 shows: the stored-axis route that drives ParT's response does not exist in L-GATr's inputs, and the tested symmetry-preserving edits move ≤ 3 % of calls.
   - PAPERS.md also notes an unverified interim L-GATr R of 0.2–0.3 for C6.

## Corrected reading
In L-GATr (interim checkpoint it154999), the class decision uses its Lorentz-invariant channels: the 16 grade-0 multivector components and the 32 auxiliary scalars. These hold overlapping information. Mean-ablating grade 0 alone costs ≤ 4 points, but after the auxiliary scalars are removed it costs a further 6–43 points (rising with depth). Mean-ablating both together drops accuracy from 85 % to 25–47 %. Vector and axial-vector ablations at blocks 0–2 drop accuracy as much as random component sets of the same size. A beam-axis boost changes 2.6–2.8 % of calls (y = 0.5). The response enters only through the time reference vector and the log-E features; their logit changes are anti-aligned (cos −0.6) but not additive. Boosting the time reference along with the particles, with features held, leaves calls at the 0.1 % float floor, which is an implementation check. log m is linearly readable from the invariant channels with gain ≈ 1. It reads better from the auxiliary scalars than from grade 0, and better than in an untrained copy (R² 0.91–0.98 vs 0.76–0.86). The M2-changing iso-mass test of C6 was not done.

## Glossary additions (for site/explain.py GLOSSARY)
- **Redundancy interaction I.** The accuracy drop when two channel groups are removed together, minus the sum of their separate drops. I > 0 means each group can stand in for the other.
- **Type-matched control.** A random set of the same number of channels drawn from the same channel type (here: 16 of the 32 auxiliary scalars).
- **F_logE / F_rest.** Boost edits in which only the log-energy inputs (F_logE), or only the other per-particle scalar inputs (F_rest), are taken from the boosted jet.

## Explainer addition (check step of the P44 entry)
**Check.** Removing one group at a time can hide a group that has a backup copy. We therefore also removed grade 0 after the auxiliary scalars were gone: $\Delta_{g0\mid s} = \text{acc}(\text{no } s) - \text{acc}(\text{no } s, \text{no } g_0)$. This came out at 6 points at block 0 and 43 points at block 11, so grade 0 is a backup carrier, not an unused one. Splitting the boost's scalar-feature route showed that only $\log E$ and $\log E/E_\text{jet}$ matter (8.9 % vs 0.2 % of calls). The logit changes from the time-reference route and the feature route have cosine $-0.6$: they push in roughly opposite directions.

## Issues
- Same interim checkpoint as P44; a newer one (it174999) now exists and the training job is still running. Neither was touched.
- The fresh-sample grade scan used 6 of the 12 blocks; the redundancy test used P44's 10,000-jet logits at all 12 blocks.
- The analysis job ran `python -m interp.p44v_analyze` with the conda image python. Neither verify job used DAS.
