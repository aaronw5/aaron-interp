# E25w verify — fresh-sample rerun of the 16-feature DAS and dose, matched 3-feature DAS, call-DAS control, alternative matched dilations

Script `interp/e25w_verify.py` (imports the E25w functions `edit16`, `pair_edits`, `Net`, `das_block`, `das_fit`, `eval_Q` unchanged).
Cluster job `e25w-verify` (1 CPU / 2 GiB; wall 852 s; process peak RSS 1.70 GiB, pod cgroup peak 2,047 MiB including page cache from
the memory-mapped `val_c16_f16.npy`; 0.17 cores on average). Smoke job `e25w-vsmoke`. Result `results/jedi/e25w_verify.json`
(smoke `e25w_verify_smoke.json`), logs `logs/e25w-verify.log`, `logs/e25w-vsmoke.log`. Nothing committed.

## Conclusion

On a new set of jets, the 16-feature network again lets total pT (M0) and the pT-weighted squared spread about the axis (M2) be swapped
between jets through two learned directions at the second pooled layer, and moving every particle off the axis again sends W jets to
Z (87 % at 0.045 rad) and then to top. Running the identical test on the 3-feature network shows the same thing for M0 and M2 at that
layer, so "M0 and M2 held separately from the second pooled layer on" is common to both networks; the two differ only in the
combination ½ log(M0·M2), which gains swap accuracy from the second pooled layer to the next dense layer in the 3-feature network and
loses it in the 16-feature network. Stretching only the ΔR input to match M2 reproduces the translation's W→Z count, while stretches
matched to the pT-weighted mean radius or the plain mean radius do not, so of these three radial quantities the response follows M2;
per jet the two edits still give different calls on up to a quarter of jets. Where M0 and M2 first become available (between the
first and second pooled layers) remains untested.

## Code review of `interp/e25w_jedi16_mass.py` and `interp/e25w_plot.py`

- Stages, masks, quantizers: the own forward pass reproduces `Jedi.run` (max |Δlogit| 0, rechecked here for 16f and 3f); DAS
  evaluation uses the exact fixed-point tail with the X10 range clip; the STE tail is used only for training. No error found.
- Padded slots: real slots are `pt > 0`; M0, M2, r̄ and every edit act on real slots only. No error found.
- Split: DAS training pairs from the fit half, test pairs from the test half, model selection on the last 20 % of training pairs.
  No leakage found.
- `costhetarel` multiplied by the angle scale: E25's column check gives costhetarel = etarot (within 0.014 for 99 % of particles),
  so scaling it with etarot is consistent. Not an error.
- T_d and R_d leave the rotated-frame columns (etarot, phirot, costhetarel) unchanged. 16f n64 has one live phirot cut (a62), so
  T and R are ΔR-route edits, not full translations / rotations. This matches E25's "all of the response goes through ΔR" and does
  not affect the T vs dilR comparison; it is a limit on calling T a complete translation.
- "P full swap" (report section 1) replaces P, which feeds only g (dd); the per-particle path a → s → u → S does not pass through P.
  The 0.27 / 0.18 / 0.13 numbers therefore measure a swap of the context route g, not a stage-level swap comparable to the S swap.
  The report labels it "P (first pooled sum) full swap"; the label should say "P → g context swap".
- do(D) is a pT rescaling of the whole jet (angles fixed), so it also moves the pT-rank-window objects (as X10 noted). D's IIA is not
  a clean test of a product variable.
- Figure `results/jedi/e25w.png`: panels match the JSON values checked (DAS, full-swap lines, dose fractions, baseline scatter).
  Panel titles A ("sit ... already at S") and C ("Same ratio, smaller size") are declarative; panel A's grey band is the maximum
  over the three variables of the random 97.5 % quantile.

## Sample

`sample(per_class=1500, seed=2541)` minus the rows of sample(3000, 2501), sample(1500, 2502), sample(1500, 2507) and
sample(1500, 2521) (E25 main, E25 fresh, E25v, E25w): N = 6,448 (g 1279, q 1307, W 1278, Z 1287, t 1297). 16 columns from
`val_c16_f16.npy`, max difference to `sample()` x 0; max |ΔR − hypot(Δη, Δφ)| 6.5e-4. Halves `rng(2542).permutation` (3,224 / 3,224).
Pairs: 3,000 train (rng 2543), 2,000 test (rng 2544). Accuracy: 16f n64 81.64 % [80.67, 82.56]; 3f n64 80.33 % [79.35, 81.29].

## (1) DAS rerun, 16f n64, and the matched 3f n64 run (k = 2)

Procedure as E25w (600 Adam steps, balanced selection, exact tail, effective IIA on pairs whose counterfactual call differs from
the base call, 1,000-resample bootstrap over pairs). Null: 500 random 2-dim subspaces per cell (Holm over 12 cells; floor 0.024).
3f input = the same edited jets' (pt, etarel, phirel) columns, so both networks see the same pairs and the same edits.
Added control "call-DAS": a 2-dim DAS trained with the source jet's call as label (the best 2-dim subspace for moving the call), scored
on each variable's counterfactuals; paired bootstrap DAS_V − call-DAS (H0: equal; H1: DAS_V higher).
Counterfactual call changes (test pairs): 16f M0 0.187, M2 0.556, D 0.578; 3f 0.206, 0.542, 0.573.

Effective IIA [95 % CI] (E25w original 16f values in parentheses):

| stage | 16f M0 | 16f M2 | 16f D | 3f M0 | 3f M2 | 3f D |
|---|---|---|---|---|---|---|
| S | 0.778 [0.737, 0.819] (0.715) | 0.668 [0.639, 0.694] (0.707) | 0.584 [0.556, 0.611] (0.581) | 0.833 [0.799, 0.868] | 0.715 [0.687, 0.740] | 0.465 [0.437, 0.492] |
| h4 | 0.810 [0.767, 0.849] (0.728) | 0.688 [0.661, 0.717] (0.659) | 0.529 [0.501, 0.556] (0.491) | 0.837 [0.802, 0.871] | 0.705 [0.677, 0.730] | 0.555 [0.526, 0.583] |
| h5 | 0.535 [0.488, 0.587] (0.572) | 0.619 [0.590, 0.648] (0.622) | 0.386 [0.356, 0.414] (0.405) | 0.706 [0.660, 0.750] | 0.699 [0.671, 0.727] | 0.447 [0.418, 0.475] |
| z | 0.449 [0.401, 0.500] (0.452) | 0.593 [0.563, 0.622] (0.585) | 0.305 [0.280, 0.333] (0.305) | 0.558 [0.513, 0.609] | 0.645 [0.615, 0.673] | 0.242 [0.219, 0.268] |
| full swap | 0.332 | 0.559 | 0.331 | 0.359 | 0.608 | 0.347 |
| call-DAS at S | 0.187 | 0.564 | 0.295 | 0.189 | 0.620 | 0.260 |

- Random 97.5 % quantile at S: 16f 0.18 / 0.11 / 0.06; 3f 0.24 / 0.19 / 0.10. Holm p = 0.024 (the floor) in all 24 cells.
- DAS − call-DAS, M2: 16f S +0.104 [0.077, 0.133], h4 +0.141, h5 +0.165, z +0.226 (all p 0.001); 3f S +0.095 [0.061, 0.128].
  DAS − full swap, M2: 16f S +0.109 [0.075, 0.144], h4 +0.129, h5 +0.059 [0.031, 0.093], z +0.033 [0.004, 0.067] (p 0.021).
- Transfer at S (16f): M0 subspace on M2 counterfactuals 0.058; M2 subspace on M0 0.174.
- On-manifold subset (M2 pairs with angle scale in [1/2, 2]; 67 % of pairs, 676 call-changing): 16f M2 DAS S 0.698 [0.666, 0.732],
  h4 0.725, full swap 0.570; 3f S 0.713, full swap 0.564.
- Stage trend of D (CIs from the table; no paired between-stage test was run): 3f S → h4 +0.090 (CIs do not overlap); 16f S → h4
  −0.055 (CIs touch at 0.556).

Competing predictions:
- "16f differs from 3f in holding M0 / M2 at S" (thesis note of E25w): excluded on these pairs; 3f M0 0.83, M2 0.72 at S with the
  same pipeline, and neither rises at h4/h5 in 3f (M2 0.715 → 0.705 → 0.699).
- "D gains IIA from S to h4 in both nets": holds for 3f only; 16f D decreases.
- "M2 DAS is only moving the call" (call-DAS / full swap explain it): excluded at S–h5 by +0.10 to +0.17 over call-DAS, but
  call-DAS alone reaches 0.56 of M2 counterfactuals at S, so most of the M2 IIA is shared with moving the call.

## (2) Dose rerun, 16f n64, with alternative matched dilations

Procedure: T_d (Δη += d on real slots, all 16 columns recomputed except rotated-frame), dilR_M2 (ΔR column × s, s = √(M2_T/M2), as
E25w), dilR_M1 (ΔR × Σ pT r_T / Σ pT r), dilR_Mr (ΔR × Σ r_T / Σ r, unweighted), R_d (rotation, as E25w). All 6,448 jets; true W
N 1,278 (unedited Z 0.015). Metrics: Z / top / q+g fraction among true W, W→Z flow with bootstrap CI, ratio flow(T)/flow(arm) with CI,
per-jet agreement of the arm's call with T's call (all jets), McNemar on the Z indicator (true W) T vs arm, Holm over 15 tests
(H0: same Z rate under T and the arm).

| d (rad) | T: Z / t / q+g | dilR_M2: Z, ratio [CI], agree, Holm p | dilR_M1: Z, ratio, Holm p | dilR_Mr: Z, ratio, Holm p | R: Z |
|---|---|---|---|---|---|
| 0.015 | 0.027 / 0.018 / 0.099 | 0.029, 0.89 [0.55, 1.33], 0.967, 0.77 | 0.210, 0.06, 8e-53 | 0.055, 0.31, 3e-5 | 0.015 |
| 0.03 | 0.500 / 0.031 / 0.089 | 0.514, 0.97 [0.93, 1.01], 0.914, 0.50 | 0.389, 1.30, 2e-9 | 0.299, 1.71, 1e-37 | 0.015 |
| 0.045 | 0.874 / 0.020 / 0.076 | 0.861, 1.02 [1.00, 1.03], 0.869, 0.13 | 0.408, 2.19, 5e-155 | 0.570, 1.55, 5e-100 | 0.015 |
| 0.06 | 0.494 / 0.231 / 0.252 | 0.476, 1.04 [0.99, 1.10], 0.739, 0.50 | 0.318, 1.58, 4e-21 | 0.563, 0.87, 0.001 | 0.015 |
| 0.09 | 0.036 / 0.626 / 0.335 | 0.020 (t 0.525, q+g 0.444), 0.646, 0.04 | 0.072, 0.37, 4e-4 | 0.203, 0.11, 8e-38 | 0.014 |

Median scale factors at 0.045: s_M2 1.143, s_M1 1.213, s_Mr 1.092.
Competing predictions: "response follows ΣpT r" or "follows Σ r" predict ratio ≈ 1 for dilR_M1 / dilR_Mr; both excluded at every dose
(Holm p ≤ 0.001). "Follows M2" predicts ratio ≈ 1 for dilR_M2: not rejected at 0.015–0.06 (Holm p 0.13–0.77). Per-jet call agreement
T vs dilR_M2 falls from 0.97 to 0.74 (0.015 → 0.06), so the match is in the W→Z count, not jet by jet. At 0.09 T sends more true W to
top than dilR_M2 (0.626 vs 0.525), as in E25w (0.645 vs 0.518).

## (3) Baselines

Not rerun (12–15 min job; reading already states p = 0.091 is the n = 10 floor). Code check: p_flow compares the trained network's
original read-out with refit baselines; the trained body with a refit read-out (flow 0.384) also exceeds every qualifying baseline
(max 0.231), so the ordering does not depend on that choice.

## Comparison with the E25w reading

- Reproduced on a fresh sample: M0 / M2 / D DAS values at S–z (within ±0.08), D below M2 at every stage, W→Z peak at 0.045 rad
  (0.874 vs 0.883), T/dilR_M2 0.97–1.04 at 0.03–0.06, top excess under T at 0.09.
- To correct: (a) the thesis note's contrast "no late h4/h5 formation step like the 3-feature net's" applies only to D; the matched
  3f run holds M0 and M2 at S as well and does not raise them at h4/h5. (b) "P full swap" is a swap of the g-context route only.
  (c) "reproduces the W→Z flow" is an aggregate match; per-jet agreement is 0.74–0.97.
- Added: the M2-matched dilation is singled out against ΣpT r- and Σ r-matched dilations; DAS M2 exceeds a call-moving subspace by
  0.10 at S.

## Proposed explainer entry (E25W_VERIFY)

- **Question.** Do the E25w interchange and dose results hold on new jets, do they differ from the 3-feature network under the same
  test, and does the translation response follow $M_2$ rather than another radial moment?
- **Background.** E25w: in the 16-feature JEDI-linear, 2-dim DAS subspaces for $M_0=\sum p_T$ and $M_2=\sum p_T\Delta R^2$ at $S$;
  a 0.045 rad translation turns 88 % of W calls into Z; a ΔR-only stretch with the same $M_2$ gives the same count.
- **Method.** Same DAS on 6,448 new jets for 16f and 3f; control "call-DAS" $Q^\ast=\arg\max$ agreement with the source call. Dose:
  ΔR stretched by $s_{M_2}=\sqrt{M_2^T/M_2}$, $s_{M_1}=\sum p_T r^T/\sum p_T r$, $s_{r}=\sum r^T/\sum r$; ratio
  $\mathrm{flow}(T)/\mathrm{flow}(\mathrm{dil})$, McNemar.
- **Result.** $M_2$ IIA at $S$ 0.67 (16f), 0.72 (3f); $D=\tfrac12\log M_0M_2$ rises $S\to h_4$ only in 3f (0.47 → 0.56). Ratios at
  0.045 rad: $s_{M_2}$ 1.02, $s_{M_1}$ 2.19, $s_r$ 1.55.
- **Checks.** 500 random subspaces (Holm), full swap, call-DAS, on-manifold subset, rotation null.
- **Why it matters.** Places the 16f/3f difference in the combination step, not in $M_0$/$M_2$, and ties the off-axis response to
  $M_2$ among the radial moments tested.
