# E24 verification: azimuth scramble at fixed radial moments (fresh sample, exact-cell control, g-freeze decomposition) (2026-10-19)

## Conclusion
On a new set of 10,000 test jets, randomising every particle's direction around the jet axis (with momenta, distances from the axis and the jet's pT-weighted centroid and spread held) again changes about 7 % of JEDI-linear's calls, against 4 % for a rigid rotation. Deep Sets and PFN networks again change about as many calls. A new causal test shows that most of JEDI-linear's extra change over rotation (about 2.4 of 3.0 points) passes through two second-round units, S58 and S31, that are computed only from the jet-level context vector g. Freezing them at their unedited values removes that part. Two parts of the E24 reading do not hold as stated. First, the real-jet comparison set the raw scramble change next to a rotation-subtracted real-jet change; on the same jets, the scramble's excess over rotation (about 1.7 points) is about a third of the real τ21 effect. Second, with exactly constructed half-plane cells, the cut-aligned within-cell scramble changes slightly more JEDI calls than the rotated-cell control (+0.5 points, p = 0.003), not the same number.

## Procedure
- Script: `interp/e24_verify.py`. Job `e24-verify`: 1 CPU / 2 GiB, wall 184 s, peak 1,529 MiB, 0.97 cores. Output: `$AI_RESULTS/jedi/e24_verify.json`, copied to `results/jedi/e24_verify.json`.
- Sample: `sample(2000, 2501)`, N = 10,000 test jets. 411 of these jets are also in the E24 main sample. Calibration pool: `sample(20000, 2502)` minus the main jets, 96,268 jets.
- Edits were re-implemented independently of `e24_azimuth_prong.py`. Only the τ21/D2 code, `get_model`, `calibration_pairs` and `sample_idx` were reused from it.
  - **rot, rot2**: two independent rigid rotations. rot2 is the null for the rot-vs-rot comparison.
  - **A**: each θ_i drawn from U(0, 2π).
  - **A'**: A, then an affine map about the pT centroid that holds M0, the vector M1 and M2.
  - **Bx (new; exact cells)**: at each particle's own radius r_i, the nine official half-plane lines α·(pT, Δη, Δφ) + β = 0 cross the circle at up to 18 angles. θ_i is redrawn uniformly in the arc that contains it, so the line offsets are included exactly. (E24's B used fixed direction sectors, which ignore the offsets.)
  - **Brx (control)**: the same construction, with the crossing angles rotated by φ_J ~ U(0, 2π) per jet. Bx and Brx use the same uniform draws.
- Edit checks:
  - Padding: 0 entries changed. pT: exact.
  - M0, M2: changed by ≤ 9e-8 (relative) under every edit.
  - M1 under A': held to 3e-7 mrad.
  - r under A': changes by median 7.2 mrad.
  - Median |Δθ|: Bx 0.110 rad, Brx 0.111 rad (p90 0.302 / 0.303).
  - τ21 recomputed against the stored Q values: max difference 5.9e-8.
- New JEDI-only interventions, using `run_jetwise` on the edited jets:
  - **gfreeze**: g held at the jet's unedited value.
  - **S58S31frz**: S58 and S31 held at their unedited values.
  - **only58_31**: the unedited jet, with only S58 and S31 set to their edited values.
- Exactness checks:
  - g substitution with its own values: max |Δlogit| = 0.
  - Substituting the full edited S into the unedited jet reproduces the edited logits: max difference 0.
  - With g frozen, S31 and S58 change by exactly 0. This confirms E5: they are functions of g only.
- Statistics:
  - 95 % intervals: 1,000 jet bootstraps.
  - H0: P(call changed | edit) = P(call changed | rot). H1: they differ. Exact McNemar test.
  - For Bx vs Brx: H0 P(change | Bx) = P(change | Brx), with Holm over the 5 models.
  - Per S unit: paired bootstrap p, with Holm over the 19 live units.

## Numbers
**Reproduction** (fresh sample; E24 values in parentheses)

| model | acc | rot | A | A' | A' − rot [CI], McNemar p | latent Δ/spread rot → A' |
|---|---|---|---|---|---|---|
| JEDI official | 81.10 % | 4.18 (4.0) | 6.78 (6.9) | 7.17 (7.2) | +2.99 [2.47, 3.51], 495 vs 196, p 1.0e-30 | 0.214 → 0.473 (0.21 → 0.47) |
| Deep Sets s0 / s1 | 80.6 / 81.1 % | 4.34 / 3.74 | 8.19 / 7.11 | 8.54 / 7.28 | p 9e-51 / 1e-41 | 0.60 → 0.53 / 0.62 → 0.54 |
| PFN s0 / s1 | 81.1 / 80.9 % | 5.05 / 5.61 | 7.86 / 8.97 | 7.72 / 8.84 | p 2e-23 / 8e-30 | 0.58 → 0.50 / 0.59 → 0.51 |

- rot2 vs rot (null): JEDI +0.02 [−0.39, 0.43] points, McNemar p 0.96. For the other models, p = 0.35–1.0.
- W τ21: median 0.254 → 0.718 under A'; mean Δ +0.411 [0.403, 0.417].
- JEDI W/Z/t → q/g: A' 3.5 %, rot 1.8 %.
- JEDI S units, A' − rot in E|ΔS|/sd: S58 +0.688, S31 +0.543, all others +0.006 to +0.058 (Holm 0.019, the bootstrap floor). S0 and S19 change by 0.
- Under A', S56 = 0.273 sd and S62 = 0.182 sd; S1 = 0.188 sd (radial).

**Decomposition of the JEDI call excess** (A' − rot, paired, same jets)

| intervention | rot | A' | A' − rot [95 % CI] | McNemar p |
|---|---|---|---|---|
| none | 4.18 | 7.17 | +2.99 [2.47, 3.51] | 1.0e-30 |
| g frozen | 3.13 | 3.70 | +0.57 [0.18, 0.92] | 0.0025 |
| S58, S31 frozen | 3.32 | 3.86 | +0.54 [0.13, 0.92] | 0.0054 |
| only S58, S31 edited | 1.99 | 5.23 | +3.24 [2.80, 3.67] | 1.1e-48 |
| null: rot2 − rot, with g frozen / S58,S31 frozen / only S58,S31 | | | +0.10 / +0.05 / +0.13 | 0.62 / 0.83 / 0.41 |

- Under A (r exact) with g frozen: 3.13 %, the same as rot with g frozen (3.13 %). These are means only; per-jet arrays were not stored. The 0.57-point residual under A' therefore appears with the A' rescale, which moves r by a median of 7 mrad.

**Real-jet calibration** (4-variable matching, N = 5,912 jets with both partners; Δτ21 of the shift partner +0.315)

| model | τ21-attributable (real) | A' on same jets | A' − rot on same jets |
|---|---|---|---|
| JEDI | 4.79 [3.92, 5.78] | 4.36 [3.82, 4.92] | 1.59 [1.05, 2.13] |
| Deep Sets s0 | 5.94 [5.06, 6.95] | 5.31 | 2.62 [2.06, 3.18] |

- E24's own file gives the same pattern. JEDI: τ21-attributable 5.2 %, A' 4.43 %, rot 2.77 %, so A' − rot = 1.7 points. Deep Sets s0: 6.5 vs 4.33 − 2.41 = 1.9 points.

**Cells: exact Bx vs rotated Brx**
- Half-plane on/off flips (through the network, including l1's input quantizer):
  - Bx: 2.0–2.8 %.
  - Brx: 4.2–4.6 %.
  - rot: 46–49 %.
  - A': 47–49 %.
- Calls changed and differences:

  | model | Bx | Brx | Bx − Brx [CI] | McNemar p (Holm over 5) |
  |---|---|---|---|---|
  | JEDI | 2.91 | 2.45 | +0.46 [0.16, 0.76] | 0.0034 (0.017) |
  | Deep Sets s0 | | | −0.08 [−0.33, 0.16] | 0.56 |
  | Deep Sets s1 | | | +0.08 [−0.14, 0.29] | 0.55 |
  | PFN s0 | | | +0.10 [−0.16, 0.36] | 0.48 |
  | PFN s1 | | | +0.18 [−0.07, 0.43] | 0.20 |

- JEDI decomposition:
  - g frozen: 2.22 vs 1.98.
  - only S58, S31 edited: 1.05 vs 0.96.
- E24's B (fixed direction sectors): 2.0–3.8 % flips vs Br 3.6–3.9 %. Its B 2.5 vs Br 2.6 % compared two edits whose cut-state preservation differed by at most about 2 points per cut.

## Code review of `interp/e24_azimuth_prong.py` / `e24_plot.py`
- **Checked with no problem found**:
  - Padding is never edited (real = pT > 0).
  - Edits act on real particles only; the moment hold is exact.
  - Mean-ablation lists for live l1 and live S units.
  - McNemar and Holm.
  - The pool excludes the main jets, and calibration matching is within class.
- **Calibration read-out**:
  - `Aprime_minus_tau21_attributable` subtracts a baseline-free real effect from the raw A' change. The raw A' change includes the orientation response that rot measures.
  - The like-for-like quantity is A' − rot on the same jets (stored as `rot_calls_changed_same_jets`, but not used).
  - The report's "this is the same size as A' on those jets" and the "difference −0.8 [−1.8, 0.3]" therefore compare unlike quantities.
- **B construction**: the sectors use the line directions only, so the 0–8 mrad offsets are ignored. This is now replaced by the exact-arc Bx above.
- **Figure, top panel**:
  - The orange bar (τ21-attributable, a difference over the 5,847 calibration jets) is drawn next to the raw A' rate over all 10,000 jets.
  - The matched bar would be A' − rot on the calibration jets: 1.7 vs 5.2 for JEDI.
  - The bottom panel matches the JSON values.

## Corrections to the E24 reading
1. "In the units with per-particle half-plane input this cancels to ≤ 0.18 sd": the measured maximum is S56 at 0.27 sd under A' (rot 0.21). S1, S62 and S63 are at 0.14–0.19.
2. "Real τ21 differences ... change about 5 % more calls" (and the report's "same size as A' on those jets"): on the same jets, A' − rot is 1.6–1.7 points for JEDI against 4.8–5.2 points for the real τ21-matched difference. Deep Sets s0: 1.9–2.6 against 5.9–6.5. The synthetic τ21 change moves about a third as many calls as a real τ21 difference of the same size at fixed m, M0, M2 and pT. The radial-matched result (≈ 0, N = 351) is unchanged.
3. "B vs Br: 2.5 vs 2.6 %, B's effect follows the size of the angular displacement, not alignment with the cuts":
   - With exact cells on a fresh sample, JEDI shows +0.46 [0.16, 0.76] points (Holm p 0.017). The aligned cells change more calls, not fewer. Deep Sets and PFN show no difference.
   - The aligned and rotated cells differ by only about 2 points in preserved cut states (97–98 % vs 95–96 %). The edit pair therefore has little leverage on cut alignment.
4. Added (supports the reading): the claim that the excess over rotation sits in S58/S31 via g now has a causal test. Freezing g, or only S58 and S31, reduces the A' − rot call excess from 2.99 to 0.57 / 0.54 points. Editing only S58 and S31 reproduces +3.24 points.

## Status of the E24 claims
- Call change under A' (JEDI 7.2 %, rot 4.0 %; Deep Sets/PFN similar): reproduced (7.17 / 4.18; 7.3–8.8 %).
- 20–40 % alternative excluded: reproduced.
- Radial objects S0, S19, S29, S30, S1 change ≤ 0.19 sd: reproduced.
- Excess in S58/S31 (latent): reproduced, and now causal at the call level (about 80 % of the excess).
- A ≈ A' (M1 hold does not remove it): reproduced (6.78 vs 7.17).
- Latent pattern (JEDI up, Deep Sets/PFN down): reproduced.
- Calibration "same size": corrected (scramble excess ≈ ⅓ of the real τ21 effect).
- B ≈ Br: corrected (small JEDI-specific difference with exact cells; low-contrast test).
- "≤ 0.18 sd" for per-particle units: corrected (≤ 0.27 sd, S56).

---

# E24 second verification: positive control with a τ21-using classifier (fresh sample 2601) (2026-10-20)

## Conclusion
A boosted-tree classifier built on jet mass, radial shape and the two-prong variable τ21 changes 6.9 % of its calls when every particle's direction around the jet axis is randomised (with the jet's pT-weighted centroid and spread held), the same as JEDI-linear's 7.1 % on the same jets. So "only about 7 % of calls" is the size of change expected from a classifier that does use two-prong shape on this dataset, because τ21 and D2 add only 0.2–0.4 points of accuracy once mass and radial shape are known. The E24 call-level result therefore does not show that JEDI-linear ignores prong structure, and the planned 20–40 % prediction it was compared with was not calibrated. Whether JEDI-linear reads prong structure at all is still open: its excess over rigid rotation (2.8 points) lies between the classifier without τ21 (1.1) and the one with it (6.4), and its accuracy falls about as much as the τ21 classifier's under the scramble.

## Procedure
- Script `interp/e24_verify2.py`; job `e24-verify2`, 1 CPU / 2 GiB, wall 168 s, peak 1,718 MiB, 0.23 cores on average. Output `$AI_RESULTS/jedi/e24_verify2.json`, copied to `results/jedi/e24_verify2.json`.
- Sample `jedi_core.sample(2000, 2601)`, N = 10,000 test jets (377 jets shared with each of the 2401 and 2501 samples).
- Edits (functions of `interp/e24_verify.py`): rot, rot2 (two independent rigid rotations about the stored axis; rot2 is the null), A′ (θ_i ~ U(0, 2π), then the affine map about the pT centroid that holds M0, vector M1 and M2). Checks: padding changed 0 entries, pT exact, M0 and M2 held to 7e-8 (relative), M1 held to 3e-7 mrad, r moved by a median of 7.2 mrad, jet mass changed by a median of 0.10 % (rot: 0.10 %, float32 rounding level). τ21 recomputation vs stored values: max difference 5.9e-8.
- Substructure under A′ (median τ21, before → after): W 0.263 → 0.720, Z 0.242 → 0.722, t 0.458 → 0.729, q 0.645 → 0.830, g 0.604 → 0.824.
- Control classifiers (new): `sklearn` HistGradientBoostingClassifier (300 iterations, learning rate 0.1, seed 0), fit on `sample(4000, 2602)` minus the 2601, 2501 and 2401 main jets (17,693 jets), features recomputed on the edited particles:
  - F0: log m, log pT_jet, log M0, log(M2/M0), multiplicity, girth Σ pT r / M0 (no azimuthal information beyond what A′ holds).
  - F1: F0 + τ21 (β = 1, k-means axes).
  - F2: F0 + τ21 + log D2.
- Networks: JEDI-linear official n64, Deep Sets d64 s0, PFN 64-64-32 s0.
- Statistics: 1,000-jet bootstrap 95 % intervals; exact McNemar on paired call-change indicators; Holm over the 6 models for A′ vs rot. H0 (per model): P(call changed | A′) = P(call changed | rot). H0 (JEDI vs F): P(JEDI call changed | A′) = P(F call changed | A′), paired on the same jets.

## Numbers

| model | accuracy | rot | A′ | A′ − rot [95 % CI] | McNemar p (Holm) | rot2 − rot, p | accuracy drop A′ vs rot | W/Z/t → q/g rot → A′ |
|---|---|---|---|---|---|---|---|---|
| F0 (mass, radial) | 78.71 % | 0.62 | 1.74 | +1.12 [0.85, 1.38] | 1e-18 | +0.06, 0.52 | 0.00 [−0.23, 0.22] | 0.28 → 1.17 |
| F1 (F0 + τ21) | 78.93 % | 0.51 | 6.89 | +6.38 [5.89, 6.87] | 9e-165 | −0.11, 0.14 | 1.14 [0.65, 1.61] | 0.22 → 2.18 |
| F2 (F1 + D2) | 79.13 % | 0.53 | 7.44 | +6.91 [6.40, 7.42] | 6e-180 | −0.09, 0.27 | 0.94 [0.51, 1.44] | 0.17 → 2.05 |
| JEDI official | 80.68 % | 4.28 | 7.07 | +2.79 [2.28, 3.31] | 3e-28 | −0.59 [−0.99, −0.21], 0.003 | 1.45 [0.97, 1.90] | 1.54 → 3.14 |
| Deep Sets s0 | 80.10 % | 4.52 | 8.46 | +3.94 [3.45, 4.47] | 2e-44 | +0.17, 0.47 | 1.82 [1.30, 2.32] | 1.84 → 2.88 |
| PFN s0 | 80.77 % | 5.20 | 7.67 | +2.47 [1.95, 3.00] | 4e-19 | −0.32, 0.16 | 1.54 [1.07, 2.02] | 2.24 → 3.01 |

(calls changed in %, N = 10,000)

- JEDI vs feature classifiers under A′, paired on the same jets:
  - F1 − JEDI calls changed: −0.18 [−0.83, 0.43] points; McNemar 534 vs 516 discordant jets, p = 0.60.
  - F2 − JEDI: +0.37 [−0.22, 0.99] points; p = 0.27.
  - F0 − JEDI: −5.33 [−5.89, −4.79] points; p = 2e-85.
  - Excess over rot, F − JEDI: F0 −1.67 [−2.24, −1.11], F1 +3.59 [2.92, 4.26], F2 +4.12 [3.48, 4.84] points.
- Accuracy per class, unedited → A′ (rot equal to unedited within 0.4 points for every model):
  - JEDI: W 87.0 → 84.7, Z 84.1 → 82.2, t 83.3 → 85.6, q 70.2 → 65.2, g 78.9 → 78.4 %.
  - F1: W 85.3 → 82.7, Z 84.9 → 85.5, t 82.3 → 83.6, q 69.3 → 65.0, g 72.9 → 72.1 %.
- τ21 and D2 add 0.22 and 0.42 points of accuracy over F0 (78.71 → 78.93 → 79.13 %).
- Reproduction of E24 on this third sample: JEDI rot 4.28 % (E24 4.0, verify-1 4.18), A′ 7.07 % (7.2, 7.17), excess +2.79 (3.2, 2.99); Deep Sets s0 A′ 8.46 % (8.0, 8.54); PFN s0 7.67 % (8.0, 7.72).

Hypotheses for the call-level test:
- H_ignore (JEDI uses no azimuthal information beyond M0, M1, M2 and the radial profile): predicts an A′ change near F0 (1.7 %) and an A′ − rot excess near F0's 1.1 points, with no accuracy drop. Not supported: JEDI's excess is 2.8 points and its accuracy falls 1.45 points.
- H_prong (JEDI uses τ21-like information as a τ21-based tagger does): predicts an A′ change near F1 (6.9 %). JEDI's total change matches (p = 0.60); its excess over rotation is 3.6 points smaller than F1's. Not excluded by calls.
- E24's plan alternative (20–40 % of calls): not a prediction of any measured τ21-using classifier here (≤ 7.4 %); its exclusion carries no information about prong use.

## Code and figure review (additions to verify-1)
- `e24_azimuth_prong.py`: no new defect found. Padding untouched, moment hold exact, calibration pool excludes the main jets. The l1 half-plane `normals` are the line directions only (noted in verify-1).
- The JEDI rot2 vs rot null on this sample is −0.59 [−0.99, −0.21] points (p = 0.003; Holm over the 6 models 0.018); on verify-1's sample it was +0.02 (p = 0.96). The jet bootstrap and McNemar treat the rotation draw as fixed, so the call-change rate under one rotation draw carries an additional draw-to-draw spread of about ±0.5 points for JEDI. The A′ − rot excess (2.8–3.2 points on three samples) exceeds this spread.
- Figure `results/jedi/e24_azimuth.png`: readable; the top-panel orange bar remains the non-like-for-like τ21-attributable value (verify-1). The figure has no τ21-using reference, so the blue bars can be read as small without a calibrated comparison; adding F1/F2 bars (6.9 / 7.4 %) would supply it.

## Corrections to the E24 reading (in addition to verify-1)
1. "yet JEDI-linear changes only about 7 % of its calls" and "near-invariance of the calls to this edit": a τ21 + D2 boosted-tree classifier changes 7.4 % and a τ21 classifier 6.9 % under the same edit (not different from JEDI, p = 0.27 / 0.60). The call-change size does not indicate insensitivity to prong structure on this dataset.
2. "The prediction of the pooled-latent / prong-use alternative (20–40 % of calls) is excluded for every model": that prediction was not calibrated; a classifier that does use τ21 gives 6.9–7.4 %. The exclusion should be removed or restated as "the uncalibrated 20–40 % figure is not reached by any model, including a τ21-based classifier".
3. Added: under A′, JEDI's accuracy falls 1.45 [0.97, 1.90] points relative to rot, as much as F1's 1.14 [0.65, 1.61]; W accuracy falls 2.3 points in JEDI and 2.6 in F1. JEDI's A′ − rot excess (2.8 points) lies between F0 (1.1) and F1 (6.4).
4. Unchanged: radial S objects change ≤ 0.19 sd; the JEDI excess passes mostly through S58/S31 via g (verify-1 causal freeze); Deep Sets and PFN behave like JEDI at call level.

## Explainer entry (for site/explain.py)
- Question: is a 7 % call change under the azimuth scramble small for a classifier that uses two-prong shape?
- Background: τ21 = τ2/τ1 with $\tau_N = \frac{1}{R_0}\sum_i z_i \min_k \Delta R_{ik}$ measures how two-prong a jet is. The scramble A′ sets $\theta_i \sim U(0, 2\pi)$ for each particle at fixed $r_i$, $p_{T,i}$, then applies $q_i = c + s\,(p'_i - c')$ so that $M_0 = \sum p_T$, $\vec M_1 = \sum p_T \vec r$ and $M_2 = \sum p_T r^2$ equal their original values.
- Method: train gradient-boosted trees on (i) mass and radial features $F_0$ and (ii) $F_0 + \tau_{21}$ ($F_1$) on held-out test jets; apply A′ and a rigid rotation to 10,000 other jets; count changed calls $\Delta = \frac1N\sum_j \mathbb{1}[\hat y_j^{\text{edit}} \ne \hat y_j]$ for the trees and the networks.
- Result: $\Delta_{A'}$ = 1.7 % ($F_0$), 6.9 % ($F_1$), 7.1 % (JEDI-linear); McNemar JEDI vs $F_1$ p = 0.60.
- Checks: rot vs second rot null; edit exactness ($M_0, M_2$ to 7e-8; $\vec M_1$ to 3e-7 mrad); τ21 recomputation to 6e-8; pool disjoint from the evaluation jets.
- Why it matters: the call-level scramble test cannot separate "uses prong shape" from "ignores it", because prong shape adds little accuracy beyond mass and radial shape on this dataset.
