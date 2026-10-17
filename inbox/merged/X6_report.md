# X6: program elements on other official JEDI-linear checkpoints (claims A1, A3, A10, B1)

Script `interp/x6_models_jedi.py` (frozen plan in its docstring), summary + figure `interp/x6_plot.py`.
Results `results/jedi/x6_3-feature_n{32,64,128}.json`, `results/jedi/x6_3-feature-perminv_n64.json`, `results/jedi/x6_summary.json`,
figure `results/jedi/x6_models.png`. Logs `logs/x6-n32.log`, `x6-n64.log`, `x6-pinv.log`, `x6-n128b.log`. The first n128 job (`x6-n128`, 2 GiB)
stopped at the memory limit after step 4. It was rerun as `x6-n128b` with 1 CPU / 3 GiB and `--extra-edges 19,61` (post hoc edges, labelled as such).
Measured use: 0.46 cores and 3016 MiB peak. The other three jobs used 1 CPU / 2 GiB.

## Procedure
- Models (no training): official `3-feature` n32, n64 (reference) and n128, plus `3-feature-perminv` n64. Each job also loads the untrained redraw
  `Jedi(untrained=True, seed=0)`.
- Samples: M = `sample(per_class=4000, seed=0)`, N = 20,000 test jets. C = `sample(per_class=2000, seed=77)` with the M jets removed, N = 9,223.
  n128 reads the same rows with 128 slots from `val_ptetaphi_f16.npy`; the first 64 slots were asserted equal to `full_test.npz`.
- Steps 1–6 follow the docstring: E1 mean ablation; E2 cut classification; zero-bit masks of qs/qs1 with a random-rank null; E20 across/inside
  swap at each model's own largest-mass qs1 edges (frozen rule); radial share of the top-5 S units; 0.06 rad translation against a matched rotation.
- Post hoc additions, not part of the frozen plan:
  - n128 edges 19|20 and 61|62.
  - An exact single-run probability $(n-k+1)/\binom{n}{k}$ for fully contiguous masks (x6_plot). The Monte Carlo null has a floor of $1/1001$, so Holm over about 19 channels cannot fall below 0.019.

## Step 0: accuracy of the unedited fixed-point network
| model | official | M (N = 20,000) | C (N = 9,223) |
|---|---|---|---|
| 3-feature n64 | 80.92 % | 80.59 % | 80.68 % |
| 3-feature n32 | 78.00 % | 77.86 % | 77.63 % |
| 3-feature n128 | 80.91 % | 80.30 % | 81.01 % |
| 3-feature-perminv n64 | 81.81 % | 81.48 % | 82.02 % |

## Step 1: live units (≥ 1 % of calls changed by mean ablation)
- Trained networks, live a / P / g / S units:
  - n64: 13 / 10 / 7 / 19
  - n32: 12 / 10 / 8 / 19
  - n128: 15 / 15 / 11 / 19
  - perminv: 17 / 12 / 10 / 21
- Units that also pass Holm (64 units per stage, H0: rate ≤ 1 %, one-sided binomial, p < 0.05): the counts are the same for a, P and S, and 1–2 lower for g.
- Untrained redraw, M[::2]: 0 live units at every stage in all four models. This control is degenerate: the untrained networks give one class
  to every jet (t for the 3-feature redraws, Z for perminv), so no ablation can change a call.

## Step 2: first-layer cut types (live l1 units)
| model | pT-only ramp | angle-only, line through axis (< 0.01 rad) | angle-only, line 0.011–0.026 rad off axis | mixed | pT-only + angle-only (Wilson 95 %) |
|---|---|---|---|---|---|
| n64 | 4 | 9 | 0 | 0 | 13/13 [0.77, 1] |
| n32 | 3 | 8 | 1 | 0 | 12/12 [0.76, 1] |
| n128 | 3 | 9 | 3 | 0 | 15/15 [0.80, 1] |
| perminv | 7 | 6 | 4 | 0 | 17/17 [0.82, 1] |

- In every unit the weights of the other input group are exactly 0 ($|w_{pT}|/|w| = 0$ or $|w_{ang}|/|w| = 0$).
- Fisher test of (pure : mixed) against n64's 13 : 0: p = 1 for each model.
- H0 (alternative hypothesis: mixed cuts dominate) predicts a mixed fraction > 0.5. The Wilson upper bound of the mixed fraction is ≤ 0.24 in every model.
- Not measured: the separation rate that HGQ weight pruning would give by chance. There is no null for "exact zeros fall on whole input groups".

## Step 3: zero-bit masks (slot axis = pT rank)
- qs1 (live S channels) in the three 3-feature models:
  - n64: 17 of 19 channels partially kept; 12 fully contiguous; median contiguity 1.00. Holm p < 0.01 for 15 (exact p for fully contiguous, Monte Carlo otherwise); Holm p > 0.05 for 2.
  - n32: 14 of 19 partially kept; 10 fully contiguous; median contiguity 1.00. Holm p < 0.01 for 8; Holm p > 0.05 for 2.
  - n128: 18 of 19 partially kept; 5 fully contiguous; median contiguity 0.98. Holm p < 0.01 for 5; Holm p > 0.05 for 0. The other 13 have Holm p = 0.019, the Monte Carlo floor.
- A hard-particle window recurs in all three:
  - qs1 ranks 0–9 in n32: S4, S12, S41, S54
  - qs1 ranks 0–13 in n64: S0, S43, S52, S63
  - qs1 ranks 0–19 in n128: S43, S57, S63
- Most n128 qs channels keep ranks 0 to about 71 and drop the rest. Ranks ≥ 72 hold a particle in a minority of jets (mean 49.1 real particles per jet).
- perminv: the qs and qs1 bit tables have shape [1, 1, 64], one bit width per channel shared by all slots. No slot-dependent mask exists.

## Step 4: E20 window test (|ΔS|/sd(S) per jet, across-edge swap vs inside swap of (Δη, Δφ), pT order kept)
H0: E[m_across] = E[m_inside]; H1: the two differ. Each result gives the ratio with a bootstrap 95 % interval and a paired sign-flip p (10,000 permutations).

| model, edge vs inside (units) | network, M | network, C | float + masks | masks removed |
|---|---|---|---|---|
| n64 13\|14 vs 11\|12 (S0, S43, S51, S52, S63) | 5.30 [5.10, 5.53], p < 1e-4 | 5.05 [4.78, 5.35] | 24.9 [24.1, 25.9] | 0.78 [0.75, 0.82] |
| n32 9\|10 vs 7\|8 (S4, S12, S34, S41, S54) | 2.31 [2.23, 2.40], p < 1e-4 | 2.27 [2.15, 2.40] | 3.56 [3.41, 3.70] | 0.70 [0.67, 0.74] |
| n128 23\|24 vs 20\|21 (S10), frozen | 0.75 [0.72, 0.79] | 0.76 [0.72, 0.82] | 1.26 [0.79, 2.06], p = 0.33 | 0.95 [0.52, 1.66] |
| n128 19\|20 vs 16\|17 (S43, S57, S63), post hoc | 4.61 [4.45, 4.79], p < 1e-4 | 4.69 [4.44, 4.95] | 8.70 [8.37, 9.03] | 0.85 [0.82, 0.89] |
| perminv 13\|14 vs 5\|6 (top-10 S), n64 ranks | 0.36 [0.34, 0.37], p < 1e-4 | 0.34 [0.32, 0.37] | 0.35 | 0.35 (no masks to remove) |

- Untrained network, every edge: m = 0 for both swaps.
- Second frozen edges:
  - n64 0|1 (S1): 0.14
  - n32 5|6 (S34, S45, S63): ≈ 0
  - n128 25|26 (S10): 1.00, p = 0.82
- Frozen-rule limitation: the edge is chosen by the ablation cost of the S units with a mask change there, whether or not those units depend on angle.
  n128 S10 and n32 S34/S45 are constant in the single-particle angle map at pT 10 GeV (step 5, variance 0). In the float program n128 S10 gives
  |ΔS| ≈ 2e-9. A (Δη, Δφ) swap cannot test these units.
- Float programs (no fixed point), accuracy on M:

  | model | float + masks | masks removed |
  |---|---|---|
  | n64 | 80.14 % | 43.58 % |
  | n32 | 77.48 % | 74.43 % |
  | n128 | 76.47 % | 23.57 % |
  | perminv | 80.06 % | 80.06 % |

## Step 5: radial share of the top-5 S units by call cost (single particle at the kept slot nearest rank 5, g frozen at its mean)
- Units whose map depends on position (variance at pT 10 GeV > 0):
  - n64: S29 0.995, S1 0.978, S50 0.997
  - n32: S5 0.996
  - n128: S48 0.999, S21 0.999, S54 0.971, S12 0.999
  - perminv: S54 0.997, S13 0.998, S48 0.992
- Position-independent units in the top 5 (share 1 by construction, or undefined):
  - n64: S0, S19
  - n32: S41, S34, S45, S10
  - n128: S10
  - perminv: S34, S55
- Controls:
  - Live l1 angle units at the same slot: 0.137–0.168. The ramps give 1.0 by construction.
  - Untrained S units of largest sd: constant (undefined) or 0.045–0.154.

## Step 6: translation by 0.06 rad of every particle (fraction of true-W jets called Z, N_W = 4,000 on M; Wilson 95 %)
| model | unedited | +Δη | range over 8 directions | matched rotation | McNemar +Δη vs rotation (n10 / n01) |
|---|---|---|---|---|---|
| n64 | 0.024 | 0.779 [0.766, 0.791] | 0.419–0.820 | 0.025 [0.020, 0.030] | 59 / 3076 |
| n32 | 0.026 | 0.500 [0.484, 0.515] | 0.447–0.603 | 0.024 [0.020, 0.029] | 79 / 1981 |
| n128 | 0.033 | 0.753 [0.740, 0.766] | 0.592–0.789 | 0.032 [0.027, 0.038] | 78 / 2964 |
| perminv | 0.018 | 0.587 [0.571, 0.602] | 0.535–0.637 | 0.018 [0.014, 0.023] | 56 / 2330 |

- The McNemar p underflows to 0 on M. On C: log10 p = −209 (n32) and −253 (perminv). The C numbers are within 0.03 of M.
- Untrained: no change; 0 for the 3-feature redraws, 1.0 for the perminv redraw, which calls everything Z.

## perminv vs n64 accuracy (the "rank windows carry the objects" alternative)
- perminv − n64 on the same jets:
  - M: +0.90 ± 0.77 points (unpaired z = 2.28)
  - C: +1.34 ± 1.12 points (z = 2.34)
  - official: 81.81 vs 80.92 %
- Per-jet calls were not stored, so the test is unpaired. Pairing would narrow the interval.
- The alternative predicts perminv below n64. The measured difference has the opposite sign, p ≈ 0.02.

## Predictions vs measurements
- Thesis predictions:
  - pT-only + angle-only ≥ 70 %: measured 100 % in all four models.
  - Contiguous windows at Holm p < 0.01: 15/17 (n64), 8/14 (n32), 5/18 (n128) qs1 channels. For most of the rest the limit is the Monte Carlo floor, 0.019.
  - Across/inside > 5: n64 5.30. n32 2.31. n128 4.61 at the post hoc edge (0.75 at the frozen edge, an angle-independent unit).
  - Radial share ≥ 0.9: all 12 position-dependent top-5 S units, 0.971–0.999.
  - Untrained 0 live units: measured, but the control is degenerate (constant output).
- Alternative ("artefact of one HGQ checkpoint"):
  - Mixed cuts dominate: 0 mixed in 57 live cuts.
  - Kept slots non-contiguous (Holm p > 0.05): 2, 2 and 0 qs1 channels.
  - Across ≈ inside: the ratio interval excludes 1 at n64 13|14, n32 9|10 and n128 19|20 (post hoc). With masks removed the ratio is 0.70–0.85 in all three, < 1.
- perminv predictions:
  - Non-contiguous masks / across ≈ inside: no slot-dependent mask exists; across/inside = 0.36.
  - Cut types recur: 17/17 pure.
  - Radial objects recur: 3/3 position-dependent units ≥ 0.99.
  - Shift response: 0.587 against 0.018.
  - Accuracy: above n64 by +0.90 points.

## Reading (scoped to these four checkpoints and the measurements above)
- Recur in all four checkpoints:
  - first-layer cuts with exactly zero weight on either pT or angle
  - near-radial (≥ 0.97) position-dependent S units
  - the W → Z response to a rigid 0.06 rad shift (0.50–0.78 against 0.02–0.03 for the matched rotation)
- These occur in the permutation-invariant model too, which has no slot-dependent masks and is 0.9 points more accurate. So in this model
  family the cut types, the radial objects and the shift response do not require pT-rank windows.
- In the three sorted-input models, contiguous rank windows recur in the qs1 masks: a hard-particle window of 10 / 14 / 20 ranks at n = 32 / 64 / 128.
- Swapping angles across such a window edge changes the windowed S units 2.3–5.3 times more than swapping inside the window. The ratio falls below 1 when the masks are removed.
- In the three sorted-input models, removing the masks from the float program changes accuracy:
  - n64: 80.1 → 43.6 %
  - n128: 76.5 → 23.6 %
  - n32: 77.5 → 74.4 %
- The size of the window effect is model dependent. The > 5 threshold set from n64 is reached only by n64.

## Issues
- The frozen edge rule selects by ablation cost, not by angular dependence. Two frozen edges (n128 23|24 and 25|26, n32 5|6) carry only
  angle-independent units, so the swap test there is uninformative. The informative n128 edge 19|20 is post hoc.
- Monte Carlo null floor (1/1001): exact p is used post hoc for fully contiguous masks only.
- The untrained control outputs one class for all jets, so it is degenerate for E1 and step 6.
- There is no null for the pT/angle weight separation under HGQ pruning.
- The accuracy comparison is unpaired.
- The n128 job needed 3 GiB.

## Explainer entry (for site/explain.py, Paper A; not inserted: site/ is outside this worker's write scope)
```python
X6_EXPLAINER = [
 ('Question', 'Are the program elements found in JEDI-linear (64 particles) properties of one trained checkpoint, or do they appear in other, separately trained official JEDI-linear models?', None),
 ('Background',
  'Three other official checkpoints read the same (pT, Δη, Δφ) inputs: 32 particles (78.0 %), 128 particles (80.9 %), and a permutation-invariant variant (81.8 %) whose fixed-point number formats are the same for every particle slot, so it cannot keep or drop particles by pT rank. '
  'In the 64-particle model, zero-bit number formats in the two sum layers keep only certain pT ranks for each object ("rank windows"). The permutation-invariant model tests whether the objects need such windows.', None),
 ('Method',
  'The same tests as for the reference model, with thresholds fixed in advance, on 20,000 test jets (seed 0) and 9,223 disjoint jets (seed 77): mean ablation of every unit; classification of each live first-layer unit as '
  '$\\alpha_{p_T} p_T + \\beta > 0$ (pT-only), $\\alpha_\\eta \\Delta\\eta + \\alpha_\\phi \\Delta\\phi + \\beta > 0$ (angle-only) or mixed; the kept pT ranks of each second-round object against random ranks; '
  'swapping $(\\Delta\\eta, \\Delta\\phi)$ of two adjacent ranks across a window edge vs inside a window; the radial share of each object\'s response to one particle; and a rigid shift of every particle by 0.06 rad, '
  'which leaves the invariant mass unchanged, against a rotation about the axis with the same mean displacement.', None),
 ('Result',
  'All 57 live first-layer units in the four models have exactly zero weight on either pT or angle (none mixed). Every position-dependent object among the five most used has radial share 0.97–0.999 (first-layer angle cuts: 0.14–0.17). '
  'The 0.06 rad shift turns 50–78 % of W jets into Z calls, against 2–3 % for the matched rotation. In the three sorted-input models the masks contain a hard-particle window of 10, 14 and 20 ranks at 32, 64 and 128 particles; '
  'an angle swap across its edge changes the windowed objects 2.3, 5.3 and 4.6 times more than a swap inside (ratio 0.70–0.85 with the masks removed). The permutation-invariant model has no rank windows and is 0.9 points more accurate than the 64-particle model.',
  'results/jedi/x6_models.png'),
 ('Checks', 'Unedited accuracy matches the official values within 0.6 points. The confirmation jets reproduce every ratio within 0.25. The untrained redraws give one class to every jet, so they cannot change calls under any edit. Two of the pre-selected window edges carry only angle-independent objects; the 128-particle edge 19|20 was chosen after the run.', None),
 ('Why it matters', 'The cut types, the near-radial objects and the shift-based mass response are present in four separately trained checkpoints, including one without rank windows; the rank windows themselves recur in the sorted-input models with a size that grows with the number of particles, and their causal strength differs between checkpoints.', None),
]
```
