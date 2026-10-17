# X7: the axis-convention response in both directions, with invariant and matched-rotation controls (D4 / A12, review weakness 2)

Script: `interp/x7_axis_reverse.py` (modes `small`, `jedi`, `combine`, `plot`).
Jobs: `x7-small2` (conda env, 1 CPU / 3 GiB, peak 1.64 GiB, 380 s), `x7-jedi2` (jax env, 1 CPU / 2 GiB, peak 1.71 GiB), `x7-comb2`. Smoke runs: `x7-*-smoke`.
The first `x7-small` run peaked at 1.96 GiB of 2 GiB, so the rerun requested 3 GiB. The first runs give the same numbers; the reruns add the call metrics of the calibration edit.
Results: `results/jedi/x7_small.json/.npz`, `x7_jedi.json/.npz`, `x7_summary.json` (table and Holm list), figure `results/jedi/x7_axis_reverse.png`. Logs: `logs/x7-*.log`.
Explainer: `site/explain.py` X7_EXPLAINER, in the Paper D tab.

## Procedure
- **Samples.**
  - JEDI-linear 3-feature n64 and controls A/B: `jedi_core.sample(4000, seed=107)`, 20,000 jets.
  - Small models: the val archive, X4's rows (rng 71, 4,000 per class), 20,000 jets.
  - Small-model inputs are built as in X4/training: `pt_sort`, `train_small.recentre_hardest` (the numpy re-implementation matches torch to 0.0), and `arch.json` scaling.
- **Models.**
  - JEDI-linear.
  - Deep Sets d64 and Linformer 2b-d32-k8, n64. Each is trained on stored-axis inputs ("stored") or about the hardest particle ("axhard"), with seeds 0 and 1 (8 models).
  - A: JEDI-linear applied after recentring each jet on its 64-particle pT centroid.
  - B: L1 logistic regression (saga, C = 1 chosen on a held-out third) on the 32 B4 EFP features. It is fitted to truth labels on `sample(6000, 0)` minus the 2,299 evaluation rows (18,467 fit / 9,234 val). It reaches 68.8 % on the evaluation sample.
- **Edits.** Real particles are re-measured about a new axis (dphi wrapped); pT and padding are unchanged.
  - Forward: stored-trained models evaluated about the hardest particle and about the WTA axis.
  - Reverse: axhard models evaluated on the stored axis and on the WTA axis.
  - On the JEDI rows the WTA axis equals the hardest particle in 66.6 % of jets. Median axis displacement is 0.041 rad (hardest) and 0.040 rad (WTA).
- **Prediction.** Per jet, $\Delta = \tfrac12\Delta\log(M_0M_2)$ about the model's training axis, i.e. $\tfrac12\log(M_2^{\rm new}/M_2^{\rm train})$.
  - Forward: mean +0.139 (JEDI) and +0.142 (small models); positive for every jet.
  - Reverse: mean −0.142; negative for every jet.
- **Formula reference (calibration edit).** The offsets about the training axis are scaled by $e^{\Delta}$ per jet. This is a real change of $\tfrac12\log(M_0M_2)$ of exactly $\Delta$ (error ≤ 4e-6).
- **Control (matched rotation).** Rotation about the training axis by $\theta = 2\arcsin(\bar d/2\bar r)$ per jet, with random sign and clipped at π (0.1 % of jets clipped).
  - The mean per-particle input displacement equals the edit's (median ratio 1.000).
  - $\tfrac12\log(M_0M_2)$ is unchanged to 4e-8.
  - Median |θ| is 0.35–0.36 rad.
- **Metrics.**
  - Accuracy, with McNemar on correctness.
  - % calls changed (bootstrap CI, 1,000 resamples over jets).
  - W→Z, Z→W, W→QCD and Z→QCD, each with McNemar.
  - W↔Z direction of switched calls (binomial test).
  - 5×5 flow table.
  - Spearman correlation of the per-jet change of the Z−W logit margin with Δ, on W jets, on Z jets and on both pooled. Bootstrap CI and asymptotic p, with a permutation p as a check.
  - Read-out: ridge (X4/E6v Ridge) from the last hidden layer (JEDI z, Deep Sets h3, Linformer pool) to ½log(M0M2) about the training axis. It is fitted on the natural jets of one half and evaluated on the other half (10,000 jets). The response ratio is the read-out change divided by the read-out change under the calibration edit (2,000 bootstrap resamples).
  - Paired bootstrap of the forward-minus-reverse response ratio.
  - Holm correction over 266 tests (every model × edit × test).

## Checks
- JEDI-linear on its sample: 80.75 %.
- Small models: 79.8–80.9 %, test_probs reproduced to ≤ 9e-6, argmax agreement 1.0.
- Read-out held-out R² is 0.94 (JEDI) and 0.89–0.95 (small models).
- Control A, natural: 80.85 %. Recentring changes 1.3 % of JEDI-linear's calls, as in E17's centroid axis.

## Numbers (20,000 jets each; CI 95 % bootstrap)
Columns:
- **Changed (edit):** % of calls changed by the axis edit.
- **Real Δ change:** % of calls changed by the matched calibration edit.
- **Rotation:** % of calls changed by the matched rotation.
- **W→Z / Z→W:** the number of switched calls in each direction.
- **Ratio:** the read-out response ratio.

| model (edit) | acc before → after | changed (edit) | real Δ change | rotation | W→Z / Z→W | ratio |
|---|---|---|---|---|---|---|
| JEDI-linear stored → hardest | 80.75 → 58.68 | 30.98 [30.39, 31.63] | 32.29 [31.68, 32.92] | 3.24 [2.99, 3.48] | 2118 / 2 | 0.80 [0.78, 0.82] |
| JEDI-linear stored → WTA | 80.75 → 59.33 | 29.98 [29.38, 30.61] | 31.85 | 3.23 [3.00, 3.47] | 2397 / 4 | 0.73 [0.72, 0.75] |
| A (centroid-recentred JEDI) → hardest / WTA | 80.85 → 80.85 | 0.00 (0 jets) | 0 | 3.33 / 3.31 | 0 / 0 | – |
| B (EFP logistic) → hardest / WTA | 68.80 → 68.80 | 0.00 (0 jets) | 0 | 0.00 | 0 / 0 | – |
| Deep Sets s0 stored → hardest | 80.48 → 48.43 | 42.80 [42.12, 43.49] | 35.24 | 2.54 [2.33, 2.76] | 2081 / 6 | 0.68 [0.67, 0.69] |
| Deep Sets s1 stored → hardest | 80.49 → 49.29 | 41.77 [41.11, 42.39] | 34.98 | 2.51 [2.29, 2.74] | 2230 / 1 | 0.47 [0.46, 0.49] |
| Linformer s0 stored → hardest | 80.87 → 56.28 | 32.40 [31.81, 33.02] | 33.91 | 0.90 [0.77, 1.02] | 1740 / 45 | 1.09 [1.05, 1.15] |
| Linformer s1 stored → hardest | 80.61 → 55.07 | 34.62 [33.97, 35.30] | 34.46 | 1.04 [0.92, 1.19] | 1858 / 17 | 0.71 [0.69, 0.74] |
| Deep Sets s0 axhard → stored | 79.81 → 75.10 | 11.09 [10.65, 11.54] | 28.82 | 2.77 [2.54, 3.00] | 233 / 724 | 0.16 [0.15, 0.17] |
| Deep Sets s1 axhard → stored | 80.17 → 76.21 | 10.12 [9.71, 10.53] | 30.50 | 2.85 [2.62, 3.08] | 78 / 694 | 0.39 [0.38, 0.40] |
| Linformer s0 axhard → stored | 80.58 → 66.12 | 20.54 [20.00, 21.11] | 32.37 | 0.79 [0.67, 0.92] | 1 / 1997 | 0.57 [0.55, 0.58] |
| Linformer s1 axhard → stored | 80.65 → 68.22 | 17.59 [17.09, 18.09] | 32.57 | 0.96 [0.83, 1.09] | 1 / 1765 | 0.62 [0.61, 0.63] |
| Deep Sets axhard → WTA (s0 / s1) | 79.81 → 79.80 / 80.17 → 80.22 | 1.57 / 1.80 | 5.08 / 5.30 | 0.96 / 0.88 | 31/45, 34/48 | 0.33 / 0.31 |
| Linformer axhard → WTA (s0 / s1) | 80.58 → 75.78 / 80.65 → 76.45 | 6.82 / 6.08 | 5.75 / 5.42 | 0.25 / 0.39 | 17/312, 10/291 | 1.80 / 1.72 |

Stored-trained small models on the WTA axis: 39.5–40.3 % (Deep Sets) and 30.8–32.4 % (Linformer) of calls changed, against 0.85–2.56 % for the rotation.

- **Ratio of calls changed (edit / matched real change).**
  - Forward: 0.96 (JEDI), 1.19–1.21 (Deep Sets), 0.96–1.00 (Linformer).
  - Reverse: 0.33–0.38 (Deep Sets), 0.54–0.63 (Linformer).
- **Class transitions, forward (stored-trained → hardest).**
  - True W called Z: 2.2–2.7 → 41.3–51.9 %.
  - True Z called QCD: about 10 → 39.5–46.4 %.
  - JEDI: W→Z 2.4 → 49.5 %, Z→QCD 10.2 → 44.5 %.
- **Class transitions, reverse (axhard → stored).**
  - True Z called W: 5.1 → 19.4 % and 5.5 → 19.1 % (Deep Sets); 4.8 → 50.3 % and 4.5 → 45.3 % (Linformer). McNemar p ≤ 1e-120.
  - True W called QCD: 9.9 → 11.2 % and 11.6 % (Deep Sets); 9.8 → 26.1 % and 10.2 → 23.8 % (Linformer).
  - True W called Z falls: 2.8 → 0.7 and 1.1 % (Linformer), 2.9 → 1.9 % (Deep Sets s1). For Deep Sets s0 it rises, 3.6 → 5.8 %.
- **Direction under the calibration edit.**
  - Forward: 2003–2092 W→Z against 0–3 Z→W.
  - Reverse: 0–1 W→Z against 2322–2885 Z→W.
- **Rotation control, JEDI.**
  - W→Z 2.4 → 2.2 % (p 0.40); Z→W 5.6 → 5.9 % (p 0.23).
  - Accuracy 80.75 → 80.79 / 80.69 %.
  - Read-out change +0.0000 [−0.0006, +0.0007].
  - Calls changed, edit against rotation (paired McNemar): p ≈ 0 in every row. The smallest gap is Deep Sets axhard → WTA, 1.57 against 0.96 % (p 1e-11).
- **Per-jet correlations.**
  - Read-out change against Δ (Spearman): forward 0.26–0.72, reverse 0.50–0.74. Every CI excludes 0.
  - Z−W margin change against Δ, forward: W jets +0.46 to +0.74; Z jets −0.59 to −0.76.
  - Same, reverse: W jets −0.27 to −0.65; Z jets +0.24 to +0.74.
  - Pooled over W and Z jets: −0.08 to +0.07. The W and Z classes move in opposite directions, so pooling cancels.
- **Forward minus reverse read-out ratio** (paired bootstrap over 10,000 jets):

  | model | hardest / stored | WTA / WTA |
  |---|---|---|
  | Deep Sets s0 | +0.52 [0.51, 0.53] | +0.25 |
  | Deep Sets s1 | +0.085 [0.067, 0.104] | +0.039 [0.002, 0.076] |
  | Linformer s0 | +0.53 [0.47, 0.58] | −0.84 |
  | Linformer s1 | +0.090 [0.064, 0.115] | −0.99 |

  For the axhard Linformer on the WTA axis, Δ is small (−0.021) and the read-out falls 1.7–1.8 times as much as under the matched real change.
- **Holm (266 tests).**
  - Every forward and axhard → stored test on correctness, W→Z, Z→W, direction and per-class correlation has p_Holm < 0.05.
  - Not significant: all tests on A and B (p = 1), and most axhard → WTA Deep Sets tests.
  - No rotation-control test reaches p_Holm < 0.05.

## Hypotheses
- **H0 / H_cov (generic covariate shift).** Every non-invariant model changes calls under both directions by an amount set by the input displacement. A rotation of the same mean displacement changes a similar fraction of calls. The W↔Z direction is not tied to the sign of Δ.
- **H1 / H_formula.** Switched calls follow the sign of Δ about the training axis: W→Z when Δ > 0; Z→W and W→QCD when Δ < 0. The size follows the model's read-out response. The rotation changes ≤ 3 % of calls. The invariant models change 0.
- **Measured against H_cov.**
  - The rotation changes 0.25–3.33 % against 6–43 % for the axis edits (p ≈ 0 per row).
  - The W↔Z direction reverses with the sign of Δ (binomial p ≈ 0 in all 8 forward and reverse hardest/stored rows).
  - These two predictions of H_cov are excluded. Its prediction that every non-invariant model loses accuracy in both directions holds: the reverse drop is 4–15 points.
- **Measured against H_formula.**
  - The sign, the invariant zeros and the forward size (0.96–1.21 of the calls changed by the matched real change) agree with it.
  - The JEDI-linear rotation changes 3.24 % [2.99, 3.48], above the stated 3 % bound. Rotations change 3.3 % for control A, which is translation-invariant but not rotation-invariant. They change 0.8–2.9 % for the small models.
  - The reverse size differs from the stated expectation. The read-out response is 0.16–0.62, and calls change 0.33–0.63 of the matched real change (10–21 % of jets). The X4 axis-jump responses were −0.41 to +0.14, from a different edit.

## Reading (scoped to these measurements)
- **Direction.** Under the change of axis convention, the switched W/Z calls follow the sign of the predicted ½Δlog(M0M2) about each model's own training axis, in both directions:
  - Stored-trained models send W → Z.
  - Hardest-trained models evaluated on the stored axis send Z → W and W → QCD.
- **Size of the rotation control.** A rotation with the same mean particle displacement and fixed M2 changes 3.6–38 times fewer calls than the axis change. Its changes have no W/Z direction.
- **Invariant controls.** Translation-invariant models give exactly 0 changed calls. Their inputs are identical bit for bit, because the stored coordinates' differences are exact in float32.
- **Size for hardest-trained models.** These models respond to the reverse change with 0.16–0.62 of a matched real mass change. That is less than the stored-trained models (forward minus reverse +0.09 to +0.53, all CIs above 0) but not near zero. The A12 statement "mostly recentred (0.13–0.41)" rests on the X4 axis-jump edit. On this edit the Linformer response is 0.57–0.62.

## Limits
- JEDI-linear has no hardest-axis-trained version, so the reverse direction is tested on the small models only.
- Control B's accuracy (68.8 %) is lower than the networks'. It serves only as an invariance check.
- The rotation control is the only matched non-mass shift tested here.
- The per-jet margin correlation is reported per class, because the pooled W+Z value cancels between classes.
