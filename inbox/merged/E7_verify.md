# E7v — verification of E7 (16 z neurons as formulas on S vs on hand observables)

Scripts: `interp/e7_verify.py` (executes E7's source `interp/e7_z_objects_jedi.py` with asserted constant replacements: seeds,
output tag, K ∈ {8, 32}, 3 randS draws, plus a new basis `hand_nomass`), `interp/e7_verify_flips.py` (per-jet overlap of call
changes). Cluster jobs (1 CPU / 2 GiB each): e7-verify (1,004 s), e7-verify-fl43 (E7's own samples), e7-verify-fl71 (E7v samples).
Results: `results/jedi/e7_verify.json`, `e7_verify_z_objects.json`, `e7_verify_{S,hand,hand_nomass,randS}.{json,npz}`,
`e7_verify_flips_e7.json`, `e7_verify_flips_e7_verify.json`; figure `results/jedi/e7_verify_z_objects.png`; logs `logs/e7-verify*.log`.

## 1. Code review of `e7_z_objects_jedi.py`
- Stage: the override replaces pre-ReLU z inside the jitted forward; d7 applies ReLU then its own input quantizer, so max(0, formula)
  goes through the same path as the network's z. Identity check (new): substituting the network's own z changes 0.0000 % of calls.
- Edits: translation shifts only real particles (pT > 0); invariant mass of the recomputed observables changes by at most 5.7e-8
  (relative) on 2,000 jets; median girth2 (= Σ z r² about the stored axis) after/before 1.548. Rotation acts on all slots (padded
  slots stay at 0). The a32 ramp keeps the pT rank (neighbour checks) and touches one real particle.
- Split: fit / e12ev split by a fixed permutation of the seed-2 sample; test jets with any overlap removed by row index; FLOOR, SDH,
  knots and selection rows come from fit jets only. No leakage found.
- Hand observables on edited jets are recomputed from the edited particles (recomputed vs stored 5.2e-8 on unedited jets).
- Statistics code (paired bootstrap via shared count matrices, slope/r, Holm, McNemar) checked; no error found.
- Minor: the log line "also in the seed-2 sample" is hard-coded; E7's figure omits randS from the edit panels and the before-edit
  W → Z rates; the initS control has no zero-bit masks or quantizers (float He-normal network).

## 2. Rerun on a fresh sample (fit `sample(6000, seed 7)`, 20,000 fit / 10,000 e12ev; test `sample(3000, seed 71)` minus 1,668 overlapping → N = 13,332; network 81.05 %)
| quantity (K = 32) | E7 (seeds 2 / 43) | E7v (seeds 7 / 71) |
|---|---|---|
| calls changed, S | 10.37 [9.82, 10.90] | 9.23 [8.72, 9.70] |
| calls changed, hand | 7.45 [7.01, 7.88] | 7.42 [6.99, 7.81] |
| hand − S | −2.92 [−3.47, −2.36] | −1.81 [−2.32, −1.28] |
| randS (random terms on S) | 23.0 | 21.6 [21.0, 22.2] |
| accuracy − network, hand | +0.07 [−0.38, +0.47], McNemar p 0.78 | −0.72 [−1.10, −0.30], McNemar p 7.5e-4 |
| accuracy − network, S | −3.46 [−3.96, −2.98] | −3.21 [−3.66, −2.74] |
| translation Δz slope S / hand | 0.578 / 0.158 | 0.572 [0.567, 0.576] / 0.123 [0.121, 0.125] |
| translation calibrated slope S / hand | 0.659 / 0.178 | 0.647 / 0.138 |
| translation W → Z after: network / S / hand | 76.2 / 88.5 / 4.3 | 77.0 / 87.6 / 1.7 |
| rotation Δz slope S / hand | 0.416 / 0.008 | 0.417 / 0.011 |
| rotation calls changed: network / S / hand | 3.89 / 4.78 / 0.35 | 3.62 / 4.86 / 0.29 |
| a32 crossing/control flip ratio: network | 1.72 [1.11, 2.65] | 1.52 [0.94, 2.57] |
| a32 ratio S / hand | 2.81 / 1.04 | 1.76 [1.17, 2.82] / 0.81 [0.51, 1.23] |
| H0 slope S − hand, translation | +0.420, 15/16 neurons (Holm) | +0.449 [+0.443, +0.454], p < 0.001, 16/16 |
| H0 slope S − hand, rotation / a32x / a32c | +0.408 / +0.234 / +0.083 | +0.406 / +0.232 / +0.083 (16, 12, 8 of 16 neurons) |

Negative poles reproduce (most negative 1 %: z1 W 67 / Z 29 %; z6 W 78 %; z8 W 98 %; z0, z2, z3, z11 t 84–92 %; z4, z14 q 76–80 %;
off-side R² on S 0.45–0.95, on hand 0.24–0.96; z1 R² 0.45 (S) / 0.24 (hand)).

## 3. Checks E7 did not run
### 3a. Hand basis without the invariant-mass family (`hand_nomass`, 77 observables: the 87 minus mass, m/ΣpT, (m/ΣpT)², sd_mass, min/max_pair_mass, m01, m012, mratio_*)
The 87-observable hand formulas put 37 % of their term weight (|c|·sd) on the invariant mass (E7 sample: mass is also the largest).
The basis also contains axis-relative quantities that respond to the translation (girth, girth2 = Σ z r², LHA, width, λ1, n_dr / z_dr
bins, psi). With the mass family removed (K = 32, fresh sample):
- calls changed 9.05 % [8.55, 9.54]; difference to S −0.17 [−0.71, +0.37] (p 0.56); accuracy −1.68 [−2.12, −1.22] vs network.
- formula weight: e2_sq 0.19, log ΣpT 0.17, ΣpT 0.17, girth2 0.11, λ1 0.06.
- translation: Δz slope 0.374 [0.369, 0.379] (calibrated 0.432), W → Z 2.5 → 75.7 % [74.1, 77.3] (network 2.5 → 77.0 %), calls changed
  34.5 % (network 55.7 %); S − hand_nomass slope +0.197 [+0.191, +0.204], p < 0.001, 16/16 neurons after Holm.
- rotation: slope 0.006, calls changed 0.17 %; a32 ratio 1.36 [0.82, 2.30].
So the hand basis's translation failure in E7 follows from the fit choosing the invariant mass among correlated observables; a hand
basis that lacks it follows the translation's call change (W → Z within 1.3 points of the network, closer than the S formula's +10.6)
at a Δz slope of 0.37 (S 0.57). No hand basis responds to the rotation (slope ≤ 0.011).

### 3b. Extrapolation (E7's open issue)
15.2 % of translated jets have at least one S column outside the fit jets' 0.5–99.5 % range (0.28 % outside min–max). S formula
translation slope, K = 32: inside 0.572 (N 11,305), outside 0.567 (N 2,027); K = 8: 0.354 / 0.310. Restricted to the (jet, neuron)
entries where the network's max(0, z) changes: 0.572 (K = 32). The 0.57 shortfall is not concentrated on out-of-range jets.

### 3c. Do the call changes fall on the same jets? (precision = P(network changes | formula changes), recall = P(formula changes | network changes), K = 32)
| edit | basis | E7 sample (seed 43) precision / recall | E7v sample (seed 71) | randS draws (both samples) |
|---|---|---|---|---|
| translation | S | 74.0 [73.0, 75.0] / 64.5 [63.5, 65.6] (chance 56.0) | 73.7 / 64.2 | 64–70 / 50–58 |
| translation | hand | 75.5 / 7.0 | 74.5 / 9.6 | |
| translation | hand_nomass | – | 80.4 [79.2, 81.5] / 49.8 [48.6, 50.9] | |
| rotation | S | 15.9 [13.0, 18.9] / 19.6 [16.1, 23.3] (chance 3.9) | 18.2 / 24.4 | 5–8 / 12–19 |
| rotation | hand | 14.9 / 1.4 | 17.9 / 1.4 | |
| a32x | S | 15.9 / 20.0 (10 of 50 network flips) | 12.0 / 16.2 (6 of 37) | 3–11 / 4–16 |
Under the rotation the S formula changes about as many calls as the network (4.78 vs 3.89 %; 4.86 vs 3.62 %), but 80 % of the
network's rotation flips are not among them (recall 20–24 %; random hinge terms on S 12–19 %).

### 3d. W → Z under translation is largely carried by the S basis itself
Random hinge terms on S (no selection, least-squares refit), K = 32: W → Z 30–36 % before → 72–86 % after (E7, 10 draws) and 84–85 %
(E7v, 3 draws), with Δz slope 0.25. The post-edit W → Z rate therefore does not separate the fitted S formula (87.6–88.5 %) from
random formulas on the same inputs; the Δz slope (0.57 vs 0.25) and flip precision/recall (74/64 vs 64–70/50–58) do.

## 4. Statements in E7's reading, checked
- "hand formula changes fewer calls than S (7.45 vs 10.37 %)": reproduced in direction (7.42 vs 9.23; gap −1.81 instead of −2.92).
- "hand formula accuracy equal to the network's" (thesis note): not reproduced (−0.72 [−1.10, −0.30] points, McNemar p 7.5e-4).
- "hand formula: translation 0.16 of the network, W → Z 4.3 %": reproduced (0.12, 1.7 %). Scope: holds for bases containing the
  invariant mass; without it, W → Z 75.7 % and slope 0.37 (3a).
- "rotation slope 0.008": reproduced (0.011). "a32 ratio 1.04": 0.81 [0.51, 1.23]. The network's own a32 ratio is 1.52 [0.94, 2.57]
  on the fresh sample (CI includes 1; 30–37 flips per arm), so the a32 test has low power for either formula.
- "S formula: same sign and ordering ... rotation flips 4.78 vs 3.89 %": the Δz slope (0.42, r 0.50) holds; the flip-rate match does
  not mean the same jets flip (recall 20–24 %, 3c). W → Z 88.5 % is largely inherited from the S inputs (3d).
- "0.58 of their size (0.66 calibrated); slope rises with K": reproduced (0.572, 0.647; K = 8 0.350). "Not tested: extrapolation":
  tested here; inside/outside fit range 0.572 / 0.567.
- "H0 rejected (p < 0.001, 11–16/16 neurons)": reproduced (8–16/16 depending on edit).
- Paper B note "a hand-observable fit with a better same-call rate fails the translation ... so the hand fit is not one section of A":
  holds for the 87-observable basis E7 used; a hand basis without the invariant-mass family has the S formula's same-call rate
  (9.05 vs 9.23 %) and matches the network's W → Z under translation (75.7 vs 77.0 %), with a Δz slope 0.20 below the S formula's.

## Corrected reading
At K = 32, formulas on the 19 S objects change 9.2–10.4 % of calls (two samples) and formulas on 87 hand observables 7.4 %; the hand
formulas' accuracy is 0.07 (E7) and −0.72 [−1.10, −0.30] (E7v) points from the network's. Under the 0.06 rad translation the S formula's
Δz is 0.57–0.58 of the network's (calibrated 0.65–0.66; same on jets inside and outside the fit range), the 87-observable hand formula's
0.12–0.16 (W → Z 1.7–4.3 % vs network 76–77 %). The hand formulas put 37 % of their weight on the invariant mass; with the mass family
removed, the hand formula changes as many calls as the S formula (9.05 %) and sends 75.7 % of W to Z (network 77.0 %) at a Δz slope of
0.37. Under the 45° rotation the S formula's Δz slope is 0.42 and no hand basis responds (≤ 0.011); the S formula's rotation flips
overlap the network's on 20–24 % of the network's flips (random S terms 12–19 %). The post-translation W → Z rate is reproduced by random
hinge terms on S (72–86 %), so it reflects the S inputs. The a32 crossing/control ratio of the network is 1.52 [0.94, 2.57] on a fresh
sample and does not separate the formulas.

## Explainer entry (site/explain.py, Paper B tab; cross-reference from A5)
**Question.** Does the hand-observable formula fail the translation test because hand observables cannot express the network's
variable, or because the fit picked the invariant mass among correlated observables?

**Background.** E7 fitted each $z_k$ as $\hat z_k = c_0+\sum_{m\le K} c_m h_m(X)$ with hinges $h_m$ on either the 19 $S$ units or 87 jet
observables. A rigid shift $\eta_i\to\eta_i+\delta$ leaves the invariant mass $m$ unchanged and raises $\sum_i z_i r_i^2$ (girth2) by a factor
1.55 (median, $\delta = 0.06$). The observable list contains both $m$ and girth2.

**Method.** Same fit, fresh jets (seeds 7 / 71, $N = 13{,}332$). Third basis: the 87 observables minus every invariant-mass quantity
(77 columns). Edits without refit; slope of $\Delta\max(0,\hat z_k)$ on $\Delta\max(0,z_k)$; per-jet overlap of call changes; random
hinge terms on $S$ as control.

**Result (figure `e7_verify_z_objects.png`).** Same-call: $S$ 9.23 %, hand 7.42 %, hand without mass 9.05 %. Translation W → Z:
network 77.0 %, $S$ 87.6 %, hand 1.7 %, hand without mass 75.7 %; $\Delta z$ slope 0.57 / 0.12 / 0.37. Rotation slope 0.42 / 0.011 / 0.006.

**Checks.** Own $z$ substituted changes 0 calls; invariant mass changes by $\le 6\times10^{-8}$ under the shift; $S$ slope equal inside and
outside the fit range (0.572 / 0.567); random terms on $S$ give W → Z 84–85 % and slope 0.25.

**Why it matters.** Whether a direct surrogate follows a mechanism-separating edit depends on which correlated observable the fit selects,
not only on whether the observables are physics quantities; the edit test detects this, the same-call rate does not.
