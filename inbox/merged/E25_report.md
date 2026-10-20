# E25: the official 16-feature JEDI-linear (n64): l1 cut inputs, origin of its radial objects, axis dependence of its mass response

Claims A1, A3, A4, A10 (exploratory). Script `interp/e25_jedi16.py` (figure `interp/e25_plot.py`; column relations from `interp/e25_diag.py`).
Results `results/jedi/e25_main.json`, `e25_fresh.json`, `e25_main_range.json`, `e25_diag.json`; figure `results/jedi/e25_edits.png`.
Logs `logs/e25-run3.log` (main), `logs/e25-verify3.log` (fresh), `logs/e25-range.log`, `logs/e25-diag.log`; failed attempts `logs/e25-run.log`, `logs/e25-run2.log`.

## Conclusion
The more accurate 16-feature JEDI-linear does not build its angular information the way the 3-feature model does. Its first layer puts
zero weight on the relative angles Δη and Δφ and on absolute η and φ. It reads angles only through ΔR (each particle's distance from the
stored jet axis) plus one rotated-frame coordinate, often mixed with pT in the same cut. Its second-round objects are therefore radial
because their input is radial, not because the network assembles them from half-plane cuts.

Its mass-like response is still not recentred. Shifting all particles 0.06 rad off the stored axis turns 49 % of true W jets into Z calls,
against 1.5 % for a matched rotation about the axis. All of this response goes through the ΔR column, and it equals the response to a
dilation with the same pT-weighted Σ pT r² (ratio 1.05). The 3-feature model gives 77 % under the same shift (16-feature / 3-feature
flow ratio 0.63).

Two things are open. The trained-vs-untrained comparison is weak, because two of the three untrained redraws give a constant call.
Whether the mass variable is formed in h4/h5 as in the 3-feature model was not tested here.

## Setup
- **Models:**
  - official 16-feature n64 (`Jedi(64, '16-feature')`; official 81.77 %);
  - 16-feature-perminv n64;
  - 3-feature n64 (comparison);
  - untrained redraws of 16-feature n64, seeds 0, 1, 2.
- **Samples:**
  - main = the rows of `sample(3000, 2501)`, N = 15,000 (3,000 per class; 3,000 true W);
  - fresh = `sample(1500, 2502)` minus every main row, N = 7,059 (1,415 true W).
- **Opt-in loader `load16`:** reads all 16 columns from `val_c16_f16.npy` at the same rows. Its (pt, etarel, phirel) columns equal
  `sample()`'s x exactly (max difference 0 on both samples), and its labels equal `val_meta.h5`.
- **Unedited accuracy:**

  | model | main | fresh |
  |---|---|---|
  | 16-feature | 81.62 % [80.99, 82.23] | 81.88 % |
  | perminv | 82.08 % | 82.72 % |
  | 3-feature | 81.29 % | 81.56 % |

- **Column relations** (`e25_diag.json`, 2,000 jets), used to recompute features after an edit:

  | relation | measured agreement |
  |---|---|
  | $p_T=\sqrt{p_x^2+p_y^2}$, $\phi=\mathrm{atan2}(p_y,p_x)$, $\eta=\sinh^{-1}(p_z/p_T)$, $\cos\theta=\tanh\eta$ | max error $\le 1.4\times10^{-3}$ |
  | $\eta-\Delta\eta$ and $\phi-\Delta\phi$ constant per jet (the stored axis) | within $1.2\times10^{-3}$ |
  | $\Delta R=\sqrt{\Delta\eta^2+\Delta\phi^2}$ | within $5\times10^{-4}$ |
  | ptrel = $p_T/p_{T,J}$, erel = $E/E_J$ | within 0.1 % |
  | etarot, phirot = offsets from the leading particle, in a frame aligned with the second particle | linear-map residual median 0.003 |
  | costhetarel equals etarot | within 0.014 for 99 % of particles |

  The leading-particle frame of etarot and phirot is unchanged by a common translation or rotation of the particles.
- **Edits:** each particle's 16 columns are changed by $F(\text{edited})-F(\text{unedited})$. Stored values are kept exactly when the
  edit is zero. Padded slots are not touched. Particle masses and the stored $p_{T,J}$, $E_J$ and axis are kept.
- **Out-of-range inputs** (beyond the l1 input quantizer's range, counting only inputs it keeps):
  - 70.8 % of jets have at least one such input unedited, mostly e (5.2 % of particle entries) and erel (1.9 %).
  - This is unchanged by T (70.7 %) and rises to 72.7 % under dilM2.

## Step 1: E1 resample ablation (16-feature, main N = 15,000; fresh in brackets)
- **Procedure:**
  - A unit's value is replaced by its value in a random other jet. For l1, this is done at the recipient's real-particle slots only.
  - Metric: % of calls changed, with a Wilson CI.
  - Test: binomial test vs 1 %, Holm per stage. "Live" means at least 1 % of calls changed and Holm p < 0.05.
- **a (l1):**
  - 17/64 live. The other 47 units are constant on the data.
  - Largest: a63 40.2 % (41.2), a42 37.2 %, a3 27.3 %, a33 26.6 %, a48 20.7 %, a17 19.8 %, a58 18.5 %, a45 15.1 %.
  - The live set is exactly the 17 units with any nonzero effective weight.
- **P:** 16 live. Largest P63 26.5 %, then P41 13.5 %.
- **g:** 8 live. g9 29.6 %; all others ≤ 5.0 %.
- **S:** 23 live. Largest S44 32.2 %, S25 29.6 %, S31 19.5 %, S43 13.3 %, S48 12.1 %.
- The fresh sample reproduces every live count (17/16/8/23) and the top units to within 1 point.

## Step 2: E2 l1 kernel in physical units
- **Effective weights:**
  - The l1 input quantizer keeps px, py, pz at 1/64 slots, phi at 2/64, costheta at 6/64, and etarel and phirel at 24 and 20 of 64.
  - Even so, the fused kernel has **zero weight on etarel, phirel, eta, phi, etarot, costheta and costhetarel** for every unit.
  - Number of units with a nonzero weight, per feature: pt 9, deltaR 10, ptrel 5, pz 2, erel 2, px 1, phirot 1.
- **The 17 live cuts** ($a=\mathrm{ReLU}(\sum_f\alpha_f x_f+\beta)$; α per GeV, per rad or per unit):

  | type | units |
  |---|---|
  | pT / momentum only (6) | a1 ($p_T<7.4$ GeV), a10 ($p_T>6.6$ GeV), a34, a45 (ptrel), a44, a55 (pz; pz is kept at one slot only) |
  | ΔR only (2) | a29 ($\Delta R<0.154$), a41 ($\Delta R>0.024$) |
  | phirot only (1) | a62 |
  | mixed pT and ΔR (8) | a3, a13, a17, a33, a42, a48, a58, a63 |

  Example mixed cut, a58: $\mathrm{ReLU}(8.19-89.7\,\Delta R+0.025\,p_T)$, i.e. $\Delta R<0.091+2.8\times10^{-4}\,p_T/\mathrm{GeV}$.
- **Purity,** with groups pT-like / px,py / Δη,Δφ / rotated frame / ΔR / absolute:
  - Exact: 8/17 = 0.47 of units have all nonzero weights in one group.
  - Soft: 13/17 = 0.77 have ≥ 95 % of the spread-weighted squared weight in one group.
- **Null:** each feature's column permuted over the 64 units, 20,000 draws.
  - Exact: median 0.875, 2.5 % quantile 0.56; P(null ≥ obs) = 0.993.
  - Soft: median 1.0; P(null ≥ obs) = 0.90.
  - The observed exact purity lies below the null's 2.5 % quantile, so mixed pT–ΔR cuts are more frequent than random placement of the
    nonzero weights.
  - The H1 of the thesis ("pure pT vs pure angle cuts, as in the 3-feature model, 13/13 pure") is not supported for this model.
- **16-feature-perminv:** 32 units with nonzero weight; exact purity 0.28 (null 2.5 % quantile 0.33, P(null ≥ obs) 0.994).
  - deltaR is used in 16 units, px/py/pz in 10–15.
  - etarot and costhetarel are used in a12 and a14, with equal and opposite weights.
  - etarel and phirel have zero weight in this model as well.

## Step 3: radial objects
- **Procedure:**
  - 200 main jets with ≥ 12 real particles.
  - The particle at slot k (3 and 10) is moved to $(r\cos t, r\sin t)$ about the stored axis: r = 0.02–0.3, t every 15°, pT kept.
  - All columns are recomputed; etarot and phirot use the jet's fitted lead-frame map (residual median 0.003).
  - g is frozen at the jet's own value.
  - Metric: within-jet radial share of $u_c$ at slot k, $1-\sum(f-\bar f_{\rm ring})^2/\sum(f-\bar f_{\rm jet})^2$.
  - Units: the top-10 position-dependent S units by E1.
  - Null: Dirichlet(1) combinations of the live position-dependent l1 maps, 2,000 draws.
- **16-feature:**
  - At slot 3 the top-10 are S44, S31, S62, S15, S49, S8, S53, S27, S38, S20. All have share 1.000 except **S38 0.45**, the one fed by the
    phirot cut a62. Slot 10 gives the same picture (S38 0.48).
  - The position-dependent l1 maps are 8 ΔR cuts (share 1.000 each) and a62 (0.06).
  - Dirichlet null: median 0.91, 97.5 % quantile 0.9999; 31 % of draws ≥ 0.97.
  - The radial share is therefore inherited from the ΔR input column. It does not need to be assembled from half-planes.
- **3-feature (same jets):** shares 0.94–0.995. Its own half-plane maps have shares 0.18–0.19; Dirichlet null median 0.68, 97.5 % quantile
  0.94; 0.8 % of draws ≥ 0.97. This reproduces X6v.
- **Input-group zeroing** (16-feature, main): the normalised input is set to 0 (the training mean) at real slots.

  | group zeroed | calls changed [95 % CI] | effect on the top-10 S units |
  |---|---|---|
  | ΔR | 75.1 % [74.4, 75.8] | $R^2$ of $S_c$ vs unedited from −0.83 to −0.04 for 9 of 10 units (S20: 0.91) |
  | pT-like (pt, ptrel, e, erel, pz) | 74.0 % | |
  | rotated frame | 7.6 % | S38 $R^2$ 0.48; all others ≥ 0.97 |
  | Δη, Δφ (together or alone) | 0.00 % | |
  | absolute η, φ, cos θ | 0.00 % | |
  | px, py | 0.03 % | |

  Fresh sample: 75.4 % (ΔR), 74.1 % (pT-like), 7.4 % (rotated frame), 0.00 % (Δη, Δφ).

## Step 4: axis edits (W → Z on true W jets; main N_W = 3,000; fresh N_W = 1,415)
- **Edits:** see Setup for how columns are recomputed.

  | arm | edit |
  |---|---|
  | T | rigid +0.06 rad in η about the stored axis, all columns recomputed (pz, e, erel, eta, costheta, etarel, deltaR change; the lead frame does not) |
  | T_dRonly | only the ΔR column edited to its value under T |
  | T_dR | T with ΔR and costhetarel held |
  | T_rel | only etarel, phirel edited |
  | Tabs | η + 0.06 with every relative column held |
  | Tphi | φ + 1 rad, absolute (px, py, phi) |
  | R | rotation about the axis matched to a 0.06 rad mean displacement (median 53°) |
  | dilM2 | Δη, Δφ (and the lead frame) × $s_T=\sqrt{M_2(T)/M_2}$ per jet, with $M_2=\sum p_T r^2$; median $s_T$ 1.245; $\tfrac12\Delta\log(M_0M_2)$ of T = 0.305 |

- **Hypotheses:**
  - H0: the calls are invariant under T beyond the rotation rate (recentring).
  - H1: W→Z(T) ≫ W→Z(R).
- **Statistics:** McNemar of each arm vs R on the Z indicator, Holm over 7 arms × 3 trained models (21 tests).
  Flow = W→Z(edit) − W→Z(unedited); the flow ratio to dilM2 uses a paired bootstrap.

| model | arm | W→Z main [95 % boot] | flow / flow(dilM2) [95 %] | McNemar vs R (n10/n01), Holm p | W→Z fresh |
|---|---|---|---|---|---|
| 16-feature | unedited | 0.015 | | | 0.020 |
| | R | 0.015 [0.011, 0.019] | 0.001 | | 0.019 |
| | **T** | **0.489 [0.470, 0.506]** | 1.05 [1.01, 1.09] | 35/1457, p < 1e-300 | 0.489 [0.463, 0.515] (ratio 1.03 [0.99, 1.09]) |
| | T_dRonly | 0.494 [0.476, 0.512] | 1.06 [1.02, 1.10] | 34/1471, p < 1e-300 | 0.493 |
| | T_dR | 0.014 [0.011, 0.019] | −0.001 | 3/1, 1.0 | 0.020 |
| | T_rel | 0.015 [0.010, 0.019] | 0.000 | 1/0, 1.0 | 0.020 |
| | Tabs | 0.014 | −0.001 | 2/0, 1.0 | 0.019 |
| | Tphi | 0.015 | 0.000 | 1/0, 1.0 | 0.020 |
| | dilM2 | 0.468 [0.450, 0.486] | 1 | 36/1394, p < 1e-300 | 0.475 |
| 16-feature-perminv | R | 0.025 | | | 0.026 |
| | T | 0.505 [0.487, 0.523] | 1.16 [1.12, 1.21] | 64/1503, p < 1e-300 | 0.505 |
| | T_dRonly | 0.509 | 1.17 | 64/1515 | 0.502 |
| | T_dR / T_rel / Tabs / Tphi | 0.027 / 0.026 / 0.025 / 0.025 | ≤ 0.001 | Holm p = 1.0 each | 0.028 / 0.026 / 0.028 / 0.028 |
| | dilM2 | 0.438 | 1 | 62/1301 | 0.456 |
| 3-feature | R | 0.022 | | | 0.020 |
| | T (= T_dR = T_rel on its 3 inputs) | 0.768 [0.754, 0.783] | 1.26 [1.23, 1.29] | 48/2287, p < 1e-300 | 0.754 (ratio 1.22) |
| | T_dRonly / Tabs / Tphi (its inputs unchanged) | 0.020 | 0 | 24/19, 1.0 | 0.029 |
| | dilM2 | 0.616 | 1 | 49/1830 | 0.621 |

- **Paired comparisons, 16-feature main:**
  - T vs T_dRonly: n10/n01 = 35/20, p = 0.058 (fresh p = 0.36).
  - T vs T_rel and T vs T_dR: 35/1458 and 35/1459, p < 1e-300.
- **Comparison with 3-feature:**
  - 16-feature flow(T) / 3-feature flow(T) = 0.474 / 0.748 = 0.63 (fresh 0.469 / 0.725 = 0.65).
  - perminv / 3-feature = 0.64.
- **Untrained 16-feature redraws:**
  - Seed 0 (accuracy 18 %): W→Z 0 under every arm; calls changed 14.2 % under T and T_dRonly, 11.4 % under dilM2, 0 % under R, T_rel, Tabs and Tphi.
  - Seeds 1 and 2 give one constant call (accuracy 20 %, 0 calls changed under every arm).
  - These redraws do not give a usable W→Z baseline.

### Competing hypotheses
| hypothesis | predicted pattern | result |
|---|---|---|
| H0, recentred (rotation-like) response | T ≈ R | Excluded: T 0.489 vs R 0.015, Holm p < 1e-300, both models and both samples. |
| (a) mass read through ΔR only | T_dR ≈ 0, T_rel ≈ 0, T_dRonly ≈ T | Observed in both 16-feature models on both samples. |
| (b) half-plane route as in the 3-feature model | T_rel ≈ T | Excluded: T_rel flow 0.000 [0.000, 0.000]; zero l1 weight on Δη, Δφ. |
| (c) absolute η/φ used | Tabs > 0 | Excluded: Tabs flow −0.001 [−0.001, 0.000]; Tphi 0.000. |

The thesis expected the response to be carried by the relative-angle columns at 0.5–0.8 of the 3-feature value. The magnitude is in that
range (0.63), but the response is carried by the ΔR column.

## Reading (scoped to these measurements)
- **First layer:**
  - In the official 16-feature n64, 17 l1 units are live.
  - Their angle information comes from ΔR (10 units) and phirot (1 unit). Δη, Δφ, η, φ and cos θ have zero fused weight.
  - 8 of the 17 cuts are mixed pT–ΔR lines. This is more mixing than random placement of the nonzero weights gives.
- **Second-round objects:** the top S units are exactly radial at the probed slots (share 1.000; S38 0.45 through phirot). The radial
  share follows from the ΔR input. Zeroing ΔR changes 75 % of calls.
- **Axis edits:**
  - A rigid 0.06 rad translation off the stored axis changes 49 % of W calls to Z; the matched rotation changes 1.5 %.
  - The whole effect is reproduced by editing only the ΔR column, and removed by holding ΔR.
  - It equals the effect of a dilation with the same $\sum p_T r^2$ (ratio 1.05 [1.01, 1.09]; fresh 1.03 [0.99, 1.09]).
  - The 3-feature net's ratio is 1.26, consistent with its known odd (direction-dependent) half-plane term. The 16-feature net uses no
    half-planes, and its ratio is close to 1.
- **The un-recentred, M2-like mass response recurs in this family through a different input route.** The "pure pT / angular half-plane
  cut → assembled radial object" part of the program does not.

## Issues and deviations
- **Debug attempts (2):**
  - `e25-run` / `e25-verify` (1 CPU, 2 GiB) and `e25-run2` / `e25-verify2` (3 and 4 GiB) were OOM-killed in step 3.
  - Cause: a reshaped slice of each per-jet (144, 64, 64) float64 output was stored as a view, keeping the whole array alive.
  - Fixed by copying the slice. `--resume` reuses steps 1–2.
  - Final jobs `e25-run3` / `e25-verify3` (1 CPU, 3 GiB): peak 2,465 / 2,413 MiB, 0.49 / 0.37 cores, wall 742 / 597 s.
  - The 3 GiB request was used at 79–80 %.
- **Out-of-range check:** the check in the first run counted inputs the quantizer drops (bits = 0). It was corrected (`e25-range`, main)
  to count only kept inputs. The S0 numbers in `e25_main.json` / `e25_fresh.json` come from the corrected code.
- **Purity null:** the script reports P(null ≥ obs). The lower-tail p is not computed exactly; the observed value lies below the 2.5 %
  quantile.
- **Feature grouping:** a44 (px and pz, each kept at one slot) counts as "mixed" because px and pz are in different groups. Without
  it, 9/17 cuts are pure.
- **Radial null:** for this model the Dirichlet null is near 1 (its input maps are already radial), so "above null" is not informative.
  The informative comparisons are the ΔR-zeroing and T_dR / T_dRonly arms.
- **3-feature S3 ranking:** the 3-feature top-10 in step 3 is ranked by S variance, not by E1. E1 was not rerun for the 3-feature model.
- **Not tested:** where in h4/h5 the mass variable forms in this model.

## Explainer entry (for site/explain.py, Paper A)
**Question.** Does the program found in the 3-feature JEDI-linear also appear in the official 16-feature JEDI-linear? That program is:
pure pT cuts and angular half-plane cuts, assembled into radial objects, then read as an un-recentred mass. The 16-feature model also
receives ΔR, absolute η and φ, energies and a rotated frame.

**Background.** The 16-feature model is the more accurate official JEDI-linear: 81.6 % on these jets vs 81.3 % for the 3-feature model.
Its inputs include $\Delta R_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$, each particle's distance from the stored jet axis. In the 3-feature
model:
- the first layer has only $p_T$ ramps and half-plane cuts $\mathrm{ReLU}(w_\eta\Delta\eta+w_\phi\Delta\phi+b)$;
- moving all particles 0.06 rad off the axis turns 77 % of W jets into Z calls, because the mass variable behaves like
  $\sqrt{M_0M_2}$ with $M_2=\sum_i p_{T,i}r_i^2$ measured about the stored axis.

**Method.**
1. Resample-ablate each unit to find the live ones.
2. Write each live first-layer cut as $\mathrm{ReLU}(\sum_f\alpha_f x_f+\beta)$ in physical units and group its inputs.
3. Move one particle around rings about the axis and measure how radial the second-round units are; set input columns to their
   training mean.
4. Edit jets with every column recomputed from the edited $(p_T,\eta,\phi)$:
   - T: translate by 0.06 rad;
   - T with only ΔR changed, or with ΔR held;
   - absolute shifts of η and φ;
   - R: a matched rotation (control);
   - dilM2: a dilation with the same $M_2$ as T.
   Count W→Z calls.

**Result** (figure `e25_edits.png`).
- The first layer has zero weight on $\Delta\eta,\Delta\phi,\eta,\phi$. Angles enter only through ΔR (10 of 17 live cuts, often mixed
  with $p_T$) and one rotated-frame cut.
- The second-round units are exactly radial because their angular input is ΔR.
- Translation T: 48.9 % of W → Z vs 1.5 % for the rotation.
  - Editing only ΔR gives 49.4 %; holding ΔR gives 1.4 %.
  - Absolute shifts give 1.4–1.5 %.
  - The M2-matched dilation gives 46.8 % (ratio 1.05).

**Checks.**
- A fresh sample (N = 7,059) reproduces every number within 0.01: T 0.489, T_dR 0.020, ratio 1.03.
- The 16-feature-perminv model gives the same pattern: T 0.505, T_dR 0.027.
- The loader matches `sample()` exactly.
- Untrained redraws are not a usable baseline: two give a constant call.

**Why it matters.**
- The not-recentred, $M_2$-like mass response is common to two official JEDI-linear families.
- The half-plane construction is specific to the 3-feature inputs. Given ΔR directly, the network uses it and builds radial objects
  without half-planes.
- Claims about "how JEDI-linear computes mass" therefore hold at the level of the observable, an un-recentred $\sum p_T r^2$ about the
  stored axis, but not at the level of the circuit.

**Conclusion.** See the Conclusion at the top of this report.
