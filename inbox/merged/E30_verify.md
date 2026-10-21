# E30 verification: weak activations and negative poles of S29, S1, S0, z0, z7, z10

Script `interp/e30_verify.py` (cluster job e30-verify, 1 CPU / 2 GiB, wall 502 s, pod peak 2,047 MiB, 0.12 cores on
average). Results `results/jedi/e30_verify.json`, log `logs/e30-verify.log`. Fresh sample: `sample(1500, 4030)`, N = 7,500
jets, fixed-point accuracy 80.88 %. No figure was made (outputs limited to the JSON and this report); the E30 figure was read.

## Conclusion
The fresh sample reproduces E30's main numbers: S1 responds to a one-particle edit by the predicted amount in every tenth of its
range, S29 does so except in its weakest tenth (about 0.77 of the prediction), and S0 responds at about 2.6 times its direct
term. Two points in E30's reading do not hold. S0's strongest tenth responds at 2.0 rather than 2.6, a difference E30's test
could not see because that tenth was part of its reference group. Most of the accuracy loss from passing the mass neurons'
negative side to the output comes from the size of the added values, not from what they say about each jet: shuffling the same
values across jets costs almost as much. A new mass edit (all particle angles scaled by ±5 %, which scales the jet mass by the
same factor) moves the negative side of z0 and z7 in the direction and by about the amount that the curve of mean z against
mass predicts. So the claim that the negative side keeps reading mass now rests on an input edit, not only on correlations.
Still open: why S29's weakest tenth responds less. It is not because the moved particles sit near the switch-on radius.

## Code review of `interp/e30_weak_poles_jedi.py` and `interp/e30_plot.py`
- Stage, masks and quantizer: S is read from `Jedi.run` (after the qs1 pool, fixed point). Rank windows come from `masks(net)`
  (zero-bit masks of the pool). Padded slots are excluded from edits (`real = pT > 0`); `hinge_sum` adds `f_pad` per empty slot
  in the window, the same before and after the edit, so it cancels in ΔS. Edited jets outside the l1 input quantizer range are
  dropped (`in_range`). No error found.
- One particle changed per edit: checked in the rerun. Radial edits change exactly 1 particle in 7,492 / 7,492 (S29) and
  5,584 / 5,584 (S1) jets, and the controls also change exactly 1. In the S0 edit, at least 60 of the 64 pT values are
  unchanged after re-sorting (one value scaled; the rest are padding ties).
- Split leakage: fit and test are disjoint after removing 308 shared jets. Part 2 uses only the E5 hinge (fitted elsewhere) and,
  for S0, a top-fit coefficient 0.01999 per GeV, which equals the E5 direct coefficient 0.320 / 16 = 0.01998. No leakage effect.
- S1 control: the rank-0 particle sits outside S1's window, and the zero-bit pool mask removes it, so "exactly 0" holds by
  construction and does not test the network. A non-structural control was added (below).
- Untrained control: S29, S1 and S0 of the redrawn network do not move at all (exactly 0), so it gives no null slope. Not rerun.
- Statistical design: each lower decile was compared with the pooled top 3 deciles, and the top deciles themselves were never
  tested. This hides S0's decile 10 (2.07 against 2.70–2.90 in deciles 8–9). The rerun tests every decile against all the others
  and adds a global heterogeneity test.
- The ratio slope Σ measured / Σ predicted is not the per-jet proportional response. The per-jet least-squares slope of
  measured on predicted ΔS (the "OLS slope"), stored in E30's JSON as `ols_slope`, is 0.59–0.90 for S29 and 0.47–0.97 for S1.
  The report does not mention it.
- Part 1 counting: for S0 the free search picks ΣpT of all particles in 8 of 10 deciles, not 9 (decile 1: ΣpT(pT > 5); decile 10:
  ΣpT(pT > 15)). In S0's decile 1, refitting the main term gives RMSE 0.49 sd against 0.19 sd for the free search. So "the same
  terms refit match the free search in each weak decile" does not hold for S0's weakest decile.
- Figure (`results/jedi/e30_weak_poles.png`): readable. Left panel shows slopes with intervals and the dotted controls. The
  example jet on the right was found by `fig_jet()`, which searches for the first jet whose measured ΔS29 is > 0, so it was
  chosen on the outcome. The 0.176 → 0.188 example is a selected case, not a typical one.
- Cluster: e30-run requested 3 GiB, a size not exempt from the utilisation rules, and averaged 0.10 cores against a 1 CPU request,
  below the 20 % floor. The 2 GiB rerun of a smaller sample peaked at 2,047 MiB.

## Procedure of the rerun (definitions)
- Edit slope: Σ over jets of measured ΔS divided by Σ predicted ΔS, with the sign of each edit aligned so that ± edits add.
  1 means the network moves by the predicted amount. Per-jet OLS slope: least-squares slope of measured on predicted ΔS
  across jets. Decile: one tenth of the 7,500 jets, ordered by the unedited unit value (ties at zero broken at random).
- A (as E30 Part 2, same edits and predictions):
  - S29 = Σ over all rank slots of ReLU(4.44 ln pT + 93.8 r − 21.2)/16 (counts wide-angle particles, with a switch-on radius
    that falls with pT). S1 = Σ over ranks ≥ 1 of ReLU(0.55 ln pT + 89.3 r − 22.9)/16 (≈ Σ(r − 0.24)₊). For each, one particle in
    the window whose hinge is on before or after is moved radially by Δr ∈ ±{0.01, 0.02, 0.03} rad. Its pT and azimuth are kept,
    so the rank order is unchanged.
  - S0 = (0.32 pT + 1.40)/16 summed over the 14 hardest (direct term) plus a context route through g. One of the 14 hardest has
    its pT multiplied by 1 ± 0.02, then re-sorted. Prediction = 0.01998 per GeV × Δ(ΣpT of slots 0–13).
  - New control for S29 and S1: the same Δr on an in-window particle whose fitted hinge is off before and after (prediction 0).
    For S1 this replaces the structural rank-0 control (also kept). S0 control: the same nudge on a particle of rank ≥ 14.
  - New split: edits whose particle has the hinge on before and after ("on-both") against edits that cross the switch-on
    radius ("crossing"). This tests E30's proposed explanation for S29's decile 1.
  - Tests: H0 slope(decile d) = slope(all other deciles), 2,000-shuffle permutation, BH over 30 tests. Global heterogeneity
    H0 "the same slope in all deciles": T = Σ_d n_d (s_d − s)², 2,000 permutations of the decile labels. Intervals: 1,000
    bootstrap resamples over jets (300 for the OLS slopes and the controls).
- B (as E30 Part 3): ReLU bypass logits' = logits + min(z_k, 0)·W7[k, :] (α = 1; W7[k, :] = the output-layer row of neuron k),
  McNemar test on paired correctness. Controls (20 draws each):
  - random h5 directions scaled to z_k's negative fraction and sd (E30's control);
  - new: global permutation of min(z_k, 0) across jets. Same values and the same distribution, but no link to the jet;
  - new: permutation within each true class. Keeps only class-level information.
- C (new, causal mass test of the negative side): angle edit Δη, Δφ × k with k = 1 ± 0.05. At fixed pT this scales the jet
  mass by k (Δln m = 0.100), and it leaves pT and the rank order unchanged. Measured: central difference dz_k / dln m per jet-mass
  bin and sign of z_k. Controls:
  - iso-mass edit (angles × k, pT × 1/k): mass unchanged, the same angular displacement;
  - rotation of the jet about its axis by ±0.05 rad: mass and every r unchanged, displacement 0.05·r, as the angle edit.
  Calibration: the slope of the binned mean z_k against the binned mean ln m on unedited jets (the descriptive curve E30 used).
  The four-vector mass computed from the particles correlates 1.0000 with the dataset mass.
  Mass bins: 0, 40, 60, 80, 90, 100, 120, 150, 200, 1000 GeV.

## Results

### A. One-particle edits (N = 7,492 / 5,584 / 7,432 jets for S29 / S1 / S0)
| unit | overall slope [95 % CI] (E30) | per-jet OLS slope | deciles 1 … 10 (rerun) | heterogeneity p | control / predicted |
|---|---|---|---|---|---|
| S29 | 0.970 [0.950, 0.990] (0.999) | 0.78 | 0.77, 0.93, 0.99, 1.04, 1.00, 0.97, 0.96, 1.03, 0.97, 1.03 | 0.0005 | in-window hinge-off 0.123 [0.113, 0.134] (E30 0.120) |
| S1 | 1.009 [0.975, 1.046] (0.960) | 0.83 | 1.18, 1.05, 1.00, 1.03, 1.02, 1.01, 1.03, 0.95, 1.01, 0.98 | 0.91 | in-window hinge-off 0.024 [0.018, 0.030]; rank 0: 0 (structural) |
| S0 | 2.61 [2.54, 2.68] (2.65) | 2.50 | 2.42, 2.57, 2.66, 2.75, 2.72, 2.64, 2.86, 2.84, 2.80, 2.00 | 0.0005 | rank ≥ 14: 0.198 [0.172, 0.229] (E30 0.19) |
- After BH, two deciles differ from the rest:
  - S29 decile 1: 0.77 [0.71, 0.83] against 0.99 for the other deciles, q 0.0075 (E30 0.79 [0.73, 0.85]).
  - S0 decile 10: 2.00 [1.82, 2.20] against 2.70, q 0.0075 (E30 2.07 [1.88, 2.25], not tested there).
  - No S1 decile differs (smallest q 0.59).
- S29 decile 1, on-both vs crossing edits: on-both 0.78 (n 554), crossing 0.72 (n 191). In deciles 2–10, on-both edits give
  0.89–1.00 and crossing edits give 1.16–1.66. In decile 1 the edited particle sits 0.033 rad (median) beyond its switch-on radius,
  against 0.039–0.091 rad in deciles 2–10. The deficit is the same for particles well past the onset. So E30's suggested
  explanation, that decile-1 edits are closer to the onset, does not account for it.
- Ratio vs per-jet slope: in every S29 and S1 decile the network moves an on-hinge particle by less than the hinge formula
  predicts. On-both: S29 0.93, S1 0.96 overall. Particles that cross the fitted onset move it by more: crossing S29 1.31,
  S1 1.37. This matches a softer switch-on than the fitted hinge. The ratio slope of ≈ 1 is the sum of the two. The per-jet OLS
  slope is 0.67–0.89 in all S29 deciles and 0.68–0.94 in S1 deciles, so the shortfall is not specific to the weak range.

### B. ReLU bypass of the negative side
| neuron | z < 0 | bypass Δacc [CI], calls changed | McNemar lost / gained | random h5 Δacc (range) | permuted min(z,0) Δacc (range) | within-class permuted |
|---|---|---|---|---|---|---|
| z0 (≈ 75 GeV window) | 43.4 % | −14.6 [−15.5, −13.7] pts, 19.2 % | 1,124 / 29 | −8.8 (−17.6 to −2.8), p 0.14 | −13.5 (−13.9 to −13.0), p 0.048 | −13.3 (−13.7 to −13.0), p 0.048 |
| z7 (94–98 GeV) | 70.3 % | −9.5 [−10.3, −8.7] pts, 16.4 % | 862 / 150 | −6.9 (−13.9 to −2.4), p 0.38 | −6.8 (−7.2 to −6.3), p 0.048 | −7.8 (−8.1 to −7.3), p 0.048 |
| z10 (135–148 GeV) | 26.2 % | −0.05 pts, 0.33 % | 12 / 8 (p 0.50) | −0.7 | −0.4 | +0.25 |
- p = rank of the true bypass among the 20 control draws (0.048 = more harmful than all 20).
- The true negative side is more harmful than every permutation. The excess over the global permutation is 1.1 points for z0
  and 2.7 points for z7; over the within-class permutation it is 1.3 and 1.7 points. The rest of the 9.5–14.6-point loss comes
  from adding values of that size through W7[k], whatever jet they belong to.
- E30's "z0 lowers accuracy more than all 20 random h5 directions" (p 0.048) does not replicate: in the rerun, p is 0.14 and
  one random direction costs 17.6 points.

### C. Mass edit on the negative side (dz/dln m; per 0.1 change in ln m multiply by 0.1)
| neuron, side, mass bin | n | angle edit dz/dln m [CI] | descriptive slope near this bin | iso-mass | rotation |
|---|---|---|---|---|---|
| z0 neg, 80–90 GeV | 365 | −21.6 [−22.4, −20.8] | −20.6 (80–90 → 90–100) | −7.1 | −0.1 |
| z0 neg, 90–100 | 916 | −24.3 [−24.8, −23.9] | −17.3 | −5.9 | 0.0 |
| z0 neg, 100–120 | 385 | −6.7 [−7.3, −6.2] | −6.8 | −3.8 | 0.0 |
| z0 neg, 120–150 | 436 | −1.5 [−1.8, −1.2] | −1.9 | −3.4 | 0.0 |
| z0 neg, 150–200 | 935 | −0.9 [−1.1, −0.7] | −1.2 | −4.9 | 0.0 |
| z7 neg, 0–40 | 897 | +0.5 [0.4, 0.6] | +1.4 | −3.9 | 0.0 |
| z7 neg, 40–60 | 821 | +2.3 [2.2, 2.5] | +5.7 | −4.0 | −0.1 |
| z7 neg, 60–80 | 1,243 | +11.2 [10.8, 11.6] | +14.3 | −2.4 | 0.0 |
| z7 neg, 80–90 | 646 | +16.6 [16.0, 17.2] | +19.1 | −1.3 | +0.1 |
| z7 neg, 100–120 | 211 | −5.2 [−6.1, −4.3] | −6.4 | −2.0 | +0.1 |
| z7 neg, 120–150 | 392 | −5.4 [−6.1, −4.7] | −6.8 | −1.6 | −0.3 |
| z7 neg, 150–200 | 926 | −3.0 [−3.6, −2.4] | −6.8 | +0.5 | +0.1 |
- H0: dz/dln m = 0 on the negative side (bootstrap p < 0.001 in every z0/z7 bin listed, except z7 at 200–1000 GeV, n 56, p 0.80).
  H1, "same mass variable continued": the sign follows the descriptive curve. Below the window z rises with mass and above it
  z falls (z7 on both sides, z0 above). H1′, "a different, mass-independent second read": dz/dln m ≈ 0 or of the opposite sign.
  The observed signs match H1 in all 12 rows. The magnitudes are 0.4–1.4 times the descriptive slope (0.4–0.8 for z7 below
  40 GeV and above 150 GeV).
- Rotation (no change in mass or r): |dz| ≤ 0.3 per unit ln m in every row, so a generic particle displacement does not move z.
- Iso-mass edit (angles changed, mass fixed): responses of −1 to −7 per unit ln m. Part of the angle-edit response therefore
  comes from changing angles at fixed mass. For z0 at 120–200 GeV the iso-mass response (−3.4, −4.9) is larger than the angle
  edit's (−1.5, −0.9), so in that far range the mass-specific share (angle − iso-mass = +2.0, +4.1) is not negative. z0 is close
  to flat there (mean z −5.3 to −5.7).
- The positive side behaves the same way. For example z0 pos 80–90 GeV: −14.3, iso −3.3. z7 pos 80–90 GeV: +16.2, iso −0.1.

## Assessment of the E30 reading
- Reproduced: S1 slope ≈ 1 in every decile with no significant decile; S29 0.94–1.07 in deciles 2–10 and 0.77–0.79 in decile 1;
  S0 ≈ 2.6 × direct term in deciles 1–9; S29 control 0.12; S0 control 0.19; bypass sizes and flows; z10 bypass ≈ 0.
- Not supported as stated:
  1. "In every decile, S1 and S0 respond at the same multiple … No lower decile differs." S0's decile 10 responds at 2.00–2.07
     against 2.6–2.9 (q 0.0075; global heterogeneity p 0.0005), reproduced on two samples. It was not visible in E30 because the
     top deciles formed the reference. The statement holds for S1 in all deciles and for S0 in deciles 1–9.
  2. "Passing it on would lower accuracy by 10–15 points" (Conclusion), read as the negative side carrying information the
     output would misuse. Permuting the same values across jets costs 13.5 (z0) and 6.8 (z7) points. The jet-specific part of
     the loss is 1–3 points.
  3. "z0 lowers accuracy more than all 20 random directions": p 0.048 → 0.14 on the fresh sample.
  4. Part 1: "In each weak decile, refitting the main formula's own terms fits as well as a free search, and the free search
     picks the same quantity". This fails for S0 decile 1 (same-form 0.49 sd vs free 0.19 sd; free term ΣpT(pT > 5)). The
     count for S0 is 8/10, not 9/10.
  5. The suggested cause of S29's decile-1 deficit (edits close to the onset) is excluded: on-both edits give 0.78.
- Strengthened: the negative pole of z0 (80–200 GeV) and z7 (below 90 and above 100 GeV) responds to a mass-scaling input edit,
  in the sign and roughly the size of the descriptive mass curve. A matched rotation gives ≈ 0. For z0 above 120 GeV, the
  iso-mass control shows that the residual response is not mass-specific.

## Corrected reading
- S1: under one-particle radial edits, the measured ΔS1 / predicted ΔS1 is 0.95–1.18 in every decile (heterogeneity p 0.91), on
  two samples. An in-window particle below the fitted onset moves S1 by 0.02 of an on-hinge move.
- S29: 0.93–1.04 in deciles 2–10, 0.77 [0.71, 0.83] in decile 1 (q 0.0075). Particles past the onset move S29 by 0.89–1.00 of
  the hinge prediction, and particles crossing the fitted onset by 1.2–1.7 of it. The decile-1 deficit (0.78) is present for
  particles well past the onset, and its cause is not identified.
- S0: 2.4–2.9 × the direct term in deciles 1–9 and 2.0 in decile 10 (the jets with the largest ΣpT of the 14 hardest), on two
  samples. This is consistent with a context route that grows more slowly than linearly at the highest ΣpT. It is the same input
  quantity with a smaller gain at the top, not a constant multiple.
- z0 and z7, negative side: scaling all particle angles by ±5 % (jet mass ±5 %) moves the pre-ReLU value in the direction and
  at 0.4–1.4 times the size of the mean-z-vs-mass curve. A rotation of the same displacement does not move it. Adding the
  negative side to the logits through the positive side's output row lowers accuracy by 9.5–14.6 points. Shuffling the same
  values across jets lowers it by 6.8–13.5 points, so 1–3 points are attributable to the per-jet content.
- z neurons, positive weak range: not determined (unchanged from E30).

## Issues
- The JSON stores fewer bootstrap draws for OLS slopes and controls (300) than for ratio slopes (1,000).
- Section C's mass-specific decomposition depends on the iso-mass control, which also changes pT by ±5 %. The rotation control is
  the cleaner null for "any displacement" but does not separate mass from other angular quantities (r, M2).
- The rerun memory peak equals the 2 GiB request. A larger sample needs a measured larger request.

## Glossary additions (for explain.GLOSSARY)
- per-jet OLS slope: least-squares slope of measured on predicted unit change across edited jets. It differs from the ratio
  slope when the network under-responds to large predicted changes and over-responds to small ones.
- on-both / crossing edit: a radial move whose particle has the fitted hinge on both before and after, or on only on one side.
- iso-mass edit: angles × k and pT × 1/k, which keeps the jet mass fixed while moving particles like the angle edit.
- permuted-bypass control: the same negative values added to the logits after shuffling them across jets (globally or within
  each true class).
