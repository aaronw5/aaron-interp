# E7 — z neurons as formulas on the network's S objects vs on hand observables (claim A5; Paper B open question)

Script `interp/e7_z_objects_jedi.py` (cluster jobs e7-s [S, SP, initS, randS], e7-hand, e7-handall [1 CPU / 3 GiB, measured peak
2,922 MiB], e7-comb [statistics + figure]; smoke run e7-smoke). Results: `results/jedi/e7_z_objects.json` (all statistics),
`results/jedi/e7_<basis>.json` (formulas, negative-pole fits), `results/jedi/e7_<basis>.npz` (per-jet formula values and calls),
figure `results/jedi/e7_z_objects.png`. Logs: `logs/e7-*.log`.

## Procedure
- Network: JEDI-linear 3-feature n64 (fixed point, `jedi_core`). Checks: seed-0 sample 81.29 % (E1: 81.29 %); fresh test sample 80.69 %.
- Fit sample: `sample(per_class=6000, seed=2)` = 30,000 jets, split as E12 (rng 0): 20,000 fit / 10,000 "e12ev".
- Test sample: `sample(per_class=3000, seed=43)` minus the 1,725 jets also in the seed-2 sample → N = 13,275 fresh jets.
- Targets: the 16 pre-ReLU z, clipped below at −0.2 sd of their positive values (as E12).
- Bases, equal term budget K ∈ {4, 8, 16, 32} per neuron, hinge (MARS) family of E12 (X, max(0, X−t), max(0, t−X), 19 quantile knots,
  products with the 10 columns most correlated with the residual; forward selection on 12,000 fit rows [6,000 for hand_all];
  least-squares refit on all 20,000):
  S = 19 live S units; SP = S + 11 live P units; hand = 87 jet-level Q observables (E12 list); hand_all = 380 Q observables;
  controls initS = S of a He-normal random-init network (19 highest-variance units), randS = K random hinge terms on S (10 draws).
- Substitution: z → max(0, formula) inside the forward pass; the network's own d7 makes the call. All 16 at once; one at a time.
- Edits on the test jets, no refit (hand observables recomputed from the edited particles with `jetdistill.observables.compute`;
  recomputed vs stored on unedited jets: max rel. diff 5.4e-8; S, P read off the network on the edited jets):
  (a) trans: rigid +0.06 rad shift in η of the real particles (E6/E6v), N = 13,275;
  (b) rot: rotation by 45° about the axis (E5v), N = 13,275;
  (c) a32x: one particle with 9–13 GeV set to 10 GeV (before) and 12 GeV (after), pT rank kept (E13), N = 5,179;
      matched control a32c: one particle 13 → 15 GeV (no a32 crossing), N = 6,237.
- Edit metrics: Δh = max(0, z_after) − max(0, z_before) per neuron, divided by the sd of max(0, z) on fit jets; OLS slope (with
  intercept) of formula Δh on network Δh, pooled over the 16 neurons and per neuron; Pearson r. Read-out calibration: the same
  regression across ordinary jets (formula level on network level, unedited "before" jets) gives the slope for a real difference
  between jets (0.88 for S, 0.89 for hand at K = 32); calibrated slope = edit slope / that slope. Call change: flip rate,
  agreement on whether the call changes and to which class; W → Z = share of true-W jets called Z.
- Statistics: 1,000 bootstrap resamples of jets, the same resamples for every quantity on a jet set (paired differences);
  two-sided bootstrap p (floor 0.001); McNemar (exact binomial) for paired correctness and paired call changes; Holm over 16 neurons.
- Negative pole (S, hand): per neuron, hinge K = 8 fitted on the fit jets with z < 0 (raw pre-ReLU target), R² on fresh z < 0 jets;
  class composition of the most negative 1 % of the fresh jets (as E16) and AUC of −z per class within z < 0.

## Hypotheses
- H0: the S formula and the hand formula have the same fidelity under the edits (difference in Δz slope, CI includes 0).
- H1 (thesis): the S formula reproduces the edit responses (slope 0.9–1.1; W → Z within 5 points of the network), while a hand formula
  with equal or better same-call rate on ordinary jets does not.
- Alternative A: both bases track the edits equally (B is one section of A). Alternative B: the S formula fails substitution
  (> 15 % calls changed at K = 32).

## Results (fresh jets, N = 13,275; 95 % bootstrap intervals)

### 1. Substitution (all 16 neurons replaced)
| basis | K=4 | K=8 | K=16 | K=32 changed | K=32 accuracy (network 80.69 %) | McNemar vs network | mean R² |
|---|---|---|---|---|---|---|---|
| S (19) | 22.20 | 16.10 | 12.54 | 10.37 [9.82, 10.90] | 77.23 | p = 1.4e-42 | 0.876 |
| S+P (30) | 22.50 | 16.16 | 12.41 | 10.02 [9.50, 10.51] | 77.59 | p = 3.8e-36 | 0.873 |
| hand (87) | 13.28 | 10.22 | 8.85 | 7.45 [7.01, 7.88] | 80.75 | p = 0.78 | 0.886 |
| hand_all (380) | 13.15 | 9.99 | 8.37 | 7.56 [7.12, 7.97] | 80.77 | p = 0.73 | 0.886 |
| initS (control) | 54.20 | 49.91 | 43.92 | 38.71 [37.88, 39.50] | 56.00 | p ≈ 0 | 0.497 |
| randS (control, 10 draws) | 47.3 | 38.6 | 28.9 | 23.0 | 68.3 | – | – |

- E12 reproduction on the e12ev jets: S K = 32 10.28 % (E12: 10.23 %).
- Paired differences at K = 32 (calls changed, points): hand − S −2.92 [−3.47, −2.36]; hand_all − S −2.82 [−3.34, −2.27];
  S+P − S −0.35 [−0.68, −0.02] (p = 0.03); initS − S +28.3 [+27.6, +29.2]; randS − S +12.6 [+12.1, +13.1] (all p ≤ 0.001 except S+P).
- Accuracy difference vs network at K = 32: hand +0.07 [−0.38, +0.47]; S −3.46 [−3.96, −2.98].
- One neuron at a time, K = 32, % calls changed (S / hand): z0 2.13 / 1.41, z1 2.27 / 2.84, z3 0.29 / 0.45, z6 0.90 / 0.59 differ
  after Holm (McNemar on the call-change indicator, p_Holm < 0.05); the other 12 neurons do not.

### 2. Edits (formulas not refit), K = 32
| edit | quantity | network | S | S+P | hand | hand_all | initS |
|---|---|---|---|---|---|---|---|
| trans 0.06 rad | Δz slope (calibrated) | 1 | 0.578 [0.574, 0.583] (0.659) | 0.555 (0.635) | 0.158 [0.156, 0.160] (0.178) | 0.156 (0.176) | 0.168 (0.302) |
| | Pearson r | | 0.719 | 0.708 | 0.436 | 0.336 | 0.293 |
| | true W → Z before / after [%] | 2.5 / 76.2 | 10.7 / 88.5 [87.4, 89.7] | 9.5 / 88.6 | 1.9 / 4.3 [3.5, 5.0] | 1.9 / 9.6 | 36.5 / 59.7 |
| | calls changed [%] | 56.04 | 48.86 | 49.60 | 5.20 | 7.44 | 52.99 |
| | call-change agreement [%] | | 59.08 | 59.70 | 44.16 | 45.24 | 38.91 |
| rot 45° | Δz slope | 1 | 0.416 [0.408, 0.424] | 0.412 | 0.008 [0.008, 0.009] | 0.007 | 0.075 |
| | calls changed [%] | 3.89 | 4.78 | 4.71 | 0.35 | 0.37 | 28.16 |
| | call-change agreement [%] | | 92.66 | 93.02 | 95.84 | 95.84 | 69.67 |
| a32x 10 → 12 GeV | Δz slope | 1 | 0.408 [0.375, 0.445] | 0.365 | 0.175 [0.163, 0.186] | 0.172 | 0.012 |
| a32c 13 → 15 GeV (control) | Δz slope | 1 | 0.258 [0.249, 0.268] | 0.253 | 0.175 [0.166, 0.183] | 0.167 | 0.017 |
| a32 crossing / control | flip-rate ratio | 1.72 [1.11, 2.65] (0.97 / 0.56 %) | 2.81 [1.89, 4.63] | 2.03 [1.40, 3.06] | 1.04 [0.59, 1.76] | 1.12 [0.65, 1.95] | 0.99 [0.69, 1.36] |

- Translation slope vs K (S / hand): 0.286 / 0.134 (K=4), 0.394 / 0.177 (8), 0.490 / 0.136 (16), 0.578 / 0.158 (32).
  Translation W → Z vs K: S 68.7, 85.3, 86.8, 88.5 %; hand 36.6, 16.9, 3.3, 4.3 %; hand_all 95.7, 43.1, 30.8, 9.6 %.
- randS (5 draws with saved values, K = 32): translation slope 0.25–0.26, W → Z 72–86 %; rotation slope 0.16–0.20.
- Per-neuron translation slope, S formula: 0.33 (z7) to 0.92 (z0); hand formula: −0.00 (z13) to 0.45 (z15).
- H0 test (paired bootstrap, K = 32), S − hand: translation slope +0.420 [+0.416, +0.425], calibrated +0.481 [+0.476, +0.486], p < 0.001;
  W → Z after +84.3 [+83.0, +85.8] points; rotation slope +0.408 [+0.400, +0.415]; a32x +0.234 [+0.205, +0.266]; a32c control
  +0.083 [+0.072, +0.094] (all p < 0.001). Per neuron, Holm over 16: translation 15/16 neurons (all but z3), rotation 16/16,
  a32x 13/16, a32c 11/16. S − hand_all: same pattern (translation +0.422 [+0.418, +0.427]). S − S+P: translation +0.023 [+0.021, +0.025].
- S formula vs network under translation: W → Z after +12.4 [+10.9, +14.0] points (the S formula already sends 10.7 % of true W to Z
  before the edit, the network 2.5 %; increase +77.8 vs +73.7 points); calls changed −7.2 [−8.1, −6.2] points.

### 3. Negative poles (pre-ReLU z < 0)
| neuron | off on | most negative 1 % (g q W Z t, %) | S formula of the off side (R²; largest terms) | hand formula (R²; largest terms) |
|---|---|---|---|---|
| z0 | 43.6 % | 6 8 0 1 86 | 0.907; (16.15 − S29)₊, (1.85 − S13)₊ | 0.935; (0.0162 − m²/pT²)₊, (103 − m)₊ |
| z1 | 9.3 % | 1 4 69 26 1 | 0.435; (10.94 − S31)₊, (1.13 − S20)₊, (S0 − 16.3)₊ | 0.251; (0.536 − pT dispersion)₊, (0.0055 − λ1)₊ |
| z2 | 64.0 % | 6 5 0 1 89 | 0.814; (S13 − 0.56)₊(1.31 − S52)₊ | 0.734; LHA, (62 − m)₊ |
| z3 | 54.4 % | 7 2 0 1 91 | 0.886; (18.93 − S29)₊, (S30 − 4.17)₊ | 0.847; (123 − m)₊, girth2 |
| z4 | 37.3 % | 10 86 2 3 0 | 0.942; (9.32 − S29)₊(32.2 − S0)₊ | 0.956; (76 − m)₊ |
| z5 | 22.8 % | 46 12 2 2 38 | 0.889; (S0 − 17.8)₊, (11.6 − S0)₊ | 0.903; (973 − ΣpT)₊, (m − 132)₊ |
| z6 | 47.4 % | 2 0 85 1 13 | 0.410; (2.92 − S1)₊ | 0.623; (93 − m)₊, (80 − m)₊ |
| z7 | 71.4 % | 19 5 0 1 76 | 0.792; (0.97 − S50)₊ | 0.821; (80 − m)₊ |
| z8 | 53.8 % | 1 1 98 0 0 | 0.570; (3.88 − S62)₊, (S13 − 0.30)₊ | 0.708; (m − 80.6)₊, (0.029 − e2)₊ |
| z9 | 42.5 % | 53 9 2 2 34 | 0.809; (S29 − 28.7)₊ | 0.887; (m − 161)₊ |
| z10 | 26.2 % | 27 49 5 7 12 | 0.837; (S19 − 6.16)₊ | 0.811; (m − 186)₊ |
| z11 | 33.8 % | 5 5 0 0 90 | 0.849; (17.07 − S29)₊ | 0.839; (126 − m)₊, girth |
| z12 | 54.5 % | 31 4 1 0 64 | 0.567; (12.01 − S29)₊ | 0.747; (m − 120)₊, (91 − m)₊ |
| z13 | 13.9 % | 41 11 2 2 45 | 0.897; (S29 − 28.4)₊, (S50 − 10.7)₊ | 0.932; (m − 171)₊ |
| z14 | 72.5 % | 3 84 5 5 3 | 0.820; (1.52 − S50)₊ | 0.866; (76 − m)₊ |
| z15 | 42.1 % | 6 5 1 4 85 | 0.614; (6.09 − S60)₊, (2.13 − S52)₊ | 0.559; z(ΔR 0.1–0.2), z(ΔR < 0.05) |
- z1's off side (atlas: W 60 %): most negative 1 % W 69 %, Z 26 %; within z1 < 0, AUC of −z for W 0.60. Its off side is fitted
  poorly by both bases (R² 0.44 on S, 0.25 on hand). z6 and z8 off sides select W (85 %, 98 % of the most negative 1 %; AUC 0.88, 0.85).
- Off-side formulas read mostly S29 (ΣpT·r² family), S50, S0, S1, S13, S62 on S, and the jet mass m on the hand side; the positive-side
  formulas (K = 8) read the same S units with other knots (e.g. z0 on: (9.997 − S29)₊ · (1.66 − S13)₊; off: (16.15 − S29)₊).

## Reading (scoped to these measurements)
- H0 (equal edit fidelity of S and hand formulas) is rejected for all three edits and for the matched control (slope differences
  +0.08 to +0.42, p < 0.001; 11–16 of 16 neurons after Holm). Alternative A (both track equally) is excluded.
- On ordinary fresh jets the hand formula reproduces the calls better than the S formula at every K (7.45 vs 10.37 % changed at K = 32;
  its accuracy equals the network's, McNemar p = 0.78). Under the 0.06 rad translation it moves 4.3 % of true W to Z (network 76.2 %),
  with a Δz slope of 0.16; under the 45° rotation its Δz slope is 0.008 and it changes 0.35 % of calls (network 3.89 %); its a32
  crossing/control flip ratio is 1.04 [0.59, 1.76] (network 1.72 [1.11, 2.65]). This reproduces E11v's direct-formula result
  (0.1 % W → Z) at equal term budget against the S formula.
- The S formula follows the direction of the network's edit responses (translation r = 0.72, W → Z 88.5 %; rotation 4.78 % calls
  changed; a32 ratio 2.81 [1.89, 4.63]), but at 0.58 of the network's Δz (0.66 after read-out calibration), not the 0.9–1.1 of H1;
  its W → Z exceeds the network's by 12.4 points, of which 8.2 points are present before the edit. The slope rises with K
  (0.29 → 0.58 from K = 4 to 32), so part of the shortfall follows the term budget. Not tested here: how much comes from hinge
  extrapolation beyond the fit range of the translated S values.
- Alternative B (S formula fails substitution, > 15 % at K = 32) is not observed: 10.37 % [9.82, 10.90]. A5 holds as "16 neurons
  written on 19 S objects change 10.4 % of calls at 32 terms per neuron and keep the edit responses' sign and ordering at 0.4–0.6
  of their size".
- Controls: random-init S formula 38.7 % calls changed (gap to S +28.3 [+27.6, +29.2]); random terms on S 23.0 % (+12.6); both
  fall below S and hand on substitution; initS Δz slopes 0.01–0.17.
- S + P adds 0.35 points on substitution and −0.02 on the translation slope.

## Thesis effect
A5: refines (substitution at 10.4 % calls changed; edit responses kept in sign and ranking at 0.4–0.6 of magnitude).
Paper B: supports the claim that a direct surrogate on hand observables with a better same-call rate does not follow mechanism-separating
edits while the object-level formula does (to 0.4–0.6 in slope). For the A/B merge rule: the hand fit is not one section of A
by this test (H0 and alternative A excluded).

## Suggested TREE.md row
| E7 | E5,E6 | 3/4 | A5, B | 16 z as hinge formulas on S vs hand observables, K 4–32; substitution on 13,275 fresh jets; translation / rotation / a32 edits without refit; negative poles | interp/e7_z_objects_jedi.py | done | % calls changed, Δz slope, W→Z | results/jedi/e7_z_objects.png | S 10.4 % vs hand 7.5 % changed; translation slope 0.58 vs 0.16, W→Z 88.5 / 4.3 % (network 76.2 %) |

## Explainer entry (for site/explain.py, Paper A tab, claim A5; cross-reference from Paper B)
**Question.** Are the 16 last-layer neurons $z_k$ functions of the network's own second-round objects $S$, and does a formula on
$S$ respond to input edits the way the network does, where a formula on hand-chosen jet observables does not?

**Background.** The 19 live $S$ units are named sums over pT-rank windows (E5: $S_{29}\approx\sum p_T r^2+\dots$, $S_0=\sum_{\le 14}p_T$, …).
A formula can match a neuron on ordinary jets for different reasons: because it computes what the neuron computes, or because
the observables it uses are correlated with it on the data. A rigid translation of the jet by $\delta$ leaves the invariant mass
unchanged but raises $M_2=\sum_i p_{T,i} r_i^2$; the network responds (E6: 77 % of W jets become Z). An edit of this kind separates the two reasons.

**Method.** Each neuron is fitted as a sum of $K$ hinge terms $\max(0, X_j-t)$, $\max(0, t-X_j)$ and their pairwise products,
$\hat z_k = c_0+\sum_{m=1}^{K} c_m\,h_m(X)$, with $X$ the 19 $S$ units or 87 jet observables, on 20,000 jets. The network's $z$ is
replaced by $\max(0,\hat z)$ and its own last layer makes the call on 13,275 other jets. Then the input jets are edited (translation
0.06 rad, rotation 45°, one particle 10 → 12 GeV across the $a_{32}$ cut) without refitting; the formula's change
$\Delta\max(0,\hat z_k)$ is regressed on the network's change $\Delta\max(0, z_k)$.

**Result (figure `e7_z_objects.png`).** At $K=32$ the $S$ formula changes 10.4 % of calls and the observable formula 7.5 %. Under the
translation the $S$ formula's slope is 0.58 and it sends 88.5 % of W jets to Z (network 76.2 %); the observable formula's slope is
0.16 and it sends 4.3 %. Under the rotation the slopes are 0.42 and 0.008.

**Checks.** Network accuracy reproduced (81.29 %, 80.69 %); E12's 10.23 % reproduced (10.28 %); random-init $S$ and random terms
on $S$ change 38.7 % and 23.0 % of calls; paired bootstrap over jets (1,000 resamples), Holm over 16 neurons; read-out calibration
against the slope across ordinary jets (0.88–0.89).

**Why it matters.** Agreement on ordinary jets does not select the formula that follows the network under edits: the formula with
fewer call changes on data responds to a mass-preserving translation at 0.16 of the network's size, the formula on the network's
own objects at 0.58. The test that separates them is an input edit, not a same-call rate.
