# E24: azimuth scrambling at fixed pT, r and rank (N-subjettiness inside JEDI-linear), cross-model (2026-10-19)

Node E24, round 7, stage 4. Claims A3 / A1 / A4d; review weakness 2 (objects vs generic Deep Sets); exploratory (prong structure).

**Files**
- Script: `interp/e24_azimuth_prong.py`. Figure script: `interp/e24_plot.py`.
- Jobs, all 1 CPU / 2 GiB:
  - `e24-smoke`: 1,000 jets.
  - `e24-run`: first full run, kept as `e24_main_v1.json`.
  - `e24-run2`: final `main` group, which adds the `Br` control.
  - `e24-x12`, `e24-tf` (conda env, torch).
  - `e24-sum`, `e24-sum2`: combine step.
  - Measured use: peak 954–1,447 MiB, about 0.98 cores, wall time 172–559 s.
- Results:
  - `results/jedi/e24_main.json`: JEDI official, Deep Sets s0/s1, PFN s0/s1.
  - `results/jedi/e24_x12.json`: the 8 X12 copies.
  - `results/jedi/e24_tf.json`: tf-2b-d32 s0/s1.
  - `results/jedi/e24_summary.json`: Holm over models.
- Figure: `results/jedi/e24_azimuth.png`.
- Logs: `logs/e24-*.log`.

## Conclusion
Each particle's direction around the jet axis was replaced by a random one. Its momentum, its distance from the axis and its pT rank stayed the same, and so did the jet's pT sum, centroid and pT-weighted spread $\sum p_T r^2$. This destroys the two-prong structure: for W jets, the N-subjettiness ratio τ21 (low for two-prong jets) goes from 0.26 to 0.72. JEDI-linear changed 7.2 % of its calls, against 4.0 % under a rigid rotation of the jet. A Deep Sets and a PFN trained on the same input change about as many calls (7–9 %), and a small transformer changes 14 %. So under this edit JEDI-linear makes little use of prong information beyond its radial quantities, but the same holds for the generic Deep Sets/PFN models. This test therefore does not separate the "radial objects" description from a generic pooled latent. Uncertain: the real-jet calibration attributes a 4–9 % call difference to τ21 at fixed mass and $\sum p_T r^2$, but this difference goes to zero once the radial profile (Σ pT r, multiplicity) is also matched, and only 351 such pairs exist (mostly top).

## Procedure
**Sample and models**
- Sample: `interp.jedi_core.sample(per_class=2000, seed=2401)`, N = 10,000 test jets.
- Unedited accuracies:

  | model | accuracy on the sample | reference |
  |---|---|---|
  | JEDI-linear official n64 | 80.89 % | 80.9 % (in the 80.6–81.3 % check range) |
  | Deep Sets dsets-d64 s0 / s1 | 80.82 / 80.55 % | |
  | PFN 64-64-32 s0 / s1 | 81.49 / 81.29 % | |
  | X12 hgq-r150 s0 / s1 / s2 | 77.5 / 56.8 / 78.1 % | s1 is the collapsed seed |
  | X12 hgq s0 / s1 / s2 | 78.3 / 78.8 / 80.1 % | |
  | X12 float s0 / s1 | 82.3 / 82.5 % | |
  | Transformer tf-2b-d32 s0 / s1 | 76.0 / 73.2 % | |

**Edits.** Each edit acts on every real particle; padding is unchanged (0 entries changed). Coordinates are polar $(r,\theta)$ about the stored axis.
- **rot** (control): rigid rotation by $\varphi_J\sim U(0,2\pi)$ per jet. M0, M2, $|M_1|$, τ21 and D2 are unchanged.
- **A** (scramble): $\theta_i\sim U(0,2\pi)$, independent per particle.
  - pT, r and rank are exact.
  - M0, M2 are unchanged to ≤ 8e-8 (relative).
  - $|\Delta\vec M_1|/M_0$: median 13.1 mrad, p90 35.8 mrad.
  - Invariant mass m changes by median 1.2 %, p90 5.8 %.
- **At**: A, then a translation by $\vec c-\vec c'$, where $\vec c=\vec M_1/M_0$. This holds $\vec M_1$ (≤ 3e-7 mrad). M2 about the axis changes by median 2.3 %.
- **A'** (main edit): A, then $q_i=\vec c+s(p'_i-\vec c')$ with $s^2=(M_2-M_0|\vec c|^2)/(M_2-M_0|\vec c'|^2)$.
  - M0, $\vec M_1$ and M2 are held exactly (≤ 7e-8 relative).
  - m is held to median 0.10 %, p90 0.31 %.
  - The rescale moves r: median 7.3 mrad, p90 27 mrad.
- **B** (half-plane cells): the nine official l1 half-planes have normals 0, 51, 131, 144, 161, 204, 249, 270 and 290°. They give 18 boundary directions. Each $\theta_i$ is redrawn uniformly within its own sector.
- **B'**: B with the A' moment hold.
- **Br** (control for B, added after the first full run): the same sector arrangement, rotated by a random angle per jet. The sector sizes are the same, but the sectors are not aligned with the cuts.
- **Redraws**: 20 redraws of the A and A' angles for JEDI official, 5 for the other models.

**Substructure** (τ21: β = 1, k-means axes, R0 0.8; D2: β = 1, 24 hardest particles; both as `jetdistill.observables`)
- Check against the stored Q values: max |Δτ21| = 5.9e-8, max |ΔD2| = 1.4e-6.
- Medians, before → after A or A' (both give the same values):

  | class | τ21 | D2 |
  |---|---|---|
  | W | 0.261 → 0.721 (mean Δ +0.405 [0.398, 0.413]) | 1.33 → 2.09 |
  | Z | 0.242 → 0.723 | 1.28 → 2.08 |
  | t | 0.446 → 0.728 | 1.90 → 2.44 |
  | q | 0.653 → 0.828 | 4.55 → 4.63 |
  | g | 0.613 → 0.825 | 3.33 → 3.48 |

- B / Br: W τ21 0.261 → 0.294 / 0.292.

**Calibration with real jets (no edit)**
- Pool: `sample(20000, 2402)` minus the 3,851 jets shared with the main sample, leaving 96,149 jets.
- Matching: same class, with $|\Delta\log|$ < 0.02 in m, M0, M2 and jet pT.
  - "shift" partner: τ21 within 0.02 of τ21(jet) + Δτ21(A').
  - "same" partner: τ21 within 0.02 of τ21(jet).
  - 5,847 jets have both partners. Mean Δτ21 is +0.314 for the shift partner and −0.000003 for the same partner.
- Radial-matched variant: additionally |Δlog Σ pT r / M0| < 0.02 and |Δn| ≤ 2. This leaves 351 jets (g 4, q 6, W 64, Z 44, t 233), with Δτ21 +0.240.
- τ21-attributable change = P(call(shift) ≠ call(jet)) − P(call(same) ≠ call(jet)), paired over jets.

**Statistics**
- 95 % intervals: 1,000 jet bootstraps.
- H0: P(call changed | edit) = P(call changed | rot). H1: they differ. Exact McNemar test on the paired change indicators.
- Holm correction over 45 tests (15 models × A, At, A').
- Per S unit: paired bootstrap p of $E|\Delta S_c|/{\rm sd}$, edit vs rot, with Holm over the 19 live S units.

## Numbers
**Calls changed [%] (95 % CI)**

"cal" is the τ21-attributable real change (4-variable matching).

| model | rot | A | At | A' | B | Br | cal |
|---|---|---|---|---|---|---|---|
| JEDI official | 4.0 [3.6, 4.4] | 6.9 [6.4, 7.4] | 11.8 [11.2, 12.5] | 7.2 [6.7, 7.7] | 2.5 [2.2, 2.9] | 2.6 [2.3, 2.9] | 5.2 [4.3, 6.1] |
| Deep Sets s0 | 4.5 | 8.0 | 12.0 | 8.0 [7.5, 8.6] | 1.3 | 1.1 | 6.5 |
| Deep Sets s1 | 4.0 | 7.3 | 11.1 | 7.3 [6.7, 7.8] | 1.3 | 1.4 | 6.3 |
| PFN s0 | 5.2 | 8.1 | 11.3 | 8.0 [7.5, 8.5] | 1.7 | 1.7 | 4.6 |
| PFN s1 | 5.4 | 8.4 | 12.8 | 8.7 [8.2, 9.2] | 1.9 | 1.8 | 5.9 |
| X12 hgq-r150 s0 / s1* / s2 | 7.8 / 11.1 / 5.8 | 8.5 / 12.4 / 7.1 | 11.9 / 16.4 / 11.5 | 9.1 / 13.1 / 8.1 | – | – | 4.9 / 6.2 / 6.7 |
| X12 hgq s0 / s1 / s2 | 5.8 / 6.3 / 4.6 | 7.8 / 7.3 / 6.0 | 11.6 / 11.4 / 9.7 | 8.4 / 7.4 / 6.2 | – | – | 8.7 / 7.6 / 5.6 |
| X12 float s0 / s1 | 4.0 / 3.9 | 8.3 / 8.3 | 12.2 / 12.5 | 8.7 / 9.0 | – | – | 4.3 / 4.0 |
| Transformer s0 / s1 | 5.1 / 6.3 | 13.5 / 13.5 | 14.9 / 14.2 | 14.3 / 13.7 | – | – | 8.0 / 9.0 |

(* collapsed seed.)
- Holm over the 45 tests: max adjusted p = 0.014, so every model rejects H0 for A, At and A'. For JEDI A' vs rot: 517 jets change only under A', 191 only under rot, p = 1.5e-35.
- Redraw ranges for A' are narrow: JEDI 6.8–7.3 %, Deep Sets 7.1–8.2 %, PFN 7.7–8.6 %, transformer 13.3–14.3 %.
- Heavy → light flow (unedited call W/Z/t → edited call q/g):

  | model | A' | rot | real shift partners | real same partners |
  |---|---|---|---|---|
  | JEDI | 3.4 % | 1.5 % | 10.3 % | 2.9 % |
  | Deep Sets / PFN | 2.6–3.5 % | 1.7–2.3 % | | |

- JEDI A' call flow (rows = unedited call, columns = edited call, order g q W Z t):
  - g → t: 107 (rot 33)
  - q → g: 139 (rot 39)
  - q → t: 73 (rot 10)
  - W → g/q: 113 (rot 21)
  - t → g: 22 (rot 35)

**Calibration on the same jets (JEDI)**
- A' changes 4.4 % [3.9, 5.0] of calls, against a τ21-attributable 5.2 % [4.3, 6.1]. The difference is −0.8 % [−1.8, 0.3].
- Radial-matched pairs: τ21-attributable is −0.3 % [−1.4, 0.6], and A' changes 1.1 % on those jets.
- Other models, radial-matched: −0.6 to +2.6 %. All intervals include 0 except X12 hgq-r150 s2 (+2.3 % [0.3, 4.6]).

**Latent change / latent spread** ($E\|\Delta L\|/E\|L-\bar L\|$)

| model | rot | A' | B |
|---|---|---|---|
| JEDI S | 0.21 | 0.47 | 0.12 |
| Deep Sets | 0.61–0.62 | 0.53–0.54 | 0.06 |
| PFN | 0.58–0.60 | 0.50–0.51 | 0.07 |

- In JEDI, the S latent moves more under the scramble than under rotation. In Deep Sets and PFN it moves less.

**JEDI official internals: stage change** (mean |Δ|/sd over units)

| stage | rot | A' | B | Br |
|---|---|---|---|---|
| P | 0.49 | 0.43 | 0.05 | 0.05 |
| S | 0.11 | 0.20 | 0.05 | 0.05 |
| h4 | 0.10 | 0.16 | 0.05 | 0.05 |
| h5 | 0.09 | 0.15 | 0.05 | 0.05 |
| z | 0.10 | 0.16 | 0.06 | 0.06 |
| logits | 0.10 | 0.16 | 0.06 | 0.06 |

**JEDI official internals: live l1 cuts**
- 13 live units: 9 half-planes, plus a32, a56, a15, a21.
- The four non-half-plane units change by exactly 0 under every edit.
- The nine half-planes:

  | quantity | rot | A' | B | Br |
  |---|---|---|---|---|
  | per-particle mean $|\Delta a|$/sd | 0.66–0.71 | 0.68–0.72 | 0.065–0.079 | 0.063–0.073 |
  | pooled $|\Delta P|$/sd | 0.64–0.70 | 0.56–0.63 | | |
  | on/off flips | 47–50 % | 47–49 % | 2.0–3.8 % | 3.6–3.9 % |

**JEDI official internals: S units, $E|\Delta S|$/sd** (relative change $E|\Delta S|/E|S|$ in parentheses)

| unit | rot | A' |
|---|---|---|
| S0 (radial) | 0 | 0 |
| S19 (radial) | 0 | 0 |
| S30 (radial) | 0.074 | 0.084 (0.044) |
| S29 (radial) | 0.061 | 0.089 (0.081) |
| S1 (radial) | 0.139 | 0.182 (0.192) |
| S58 (context-only) | 0.47 | 1.16 (0.55) |
| S31 (context-only) | 0.27 | 0.82 (0.99) |
| S56 | 0.21 | 0.27 |
| other units | 0.03–0.17 | 0.07–0.18 |

- A' − rot: S58 +0.69, S31 +0.55; every other live unit +0.007 to +0.056.
- Holm: 17 of 19 units have p_holm = 0.019, which is the bootstrap floor. S0 and S19 have p_holm = 1 (zero change).
- A − A' (the M1 part of the scramble): −0.06 to 0 per unit. Under pure A, most units change less than under rot (−0.013 to −0.002), except S31 (+0.53) and S58 (+0.69).
- B < rot for every unit. B ≈ Br.

## Reading (scoped to these measurements)
**Size of the call change**
- In JEDI-linear official, scrambling azimuth at fixed M0, $\vec M_1$ and M2 changes 7.2 % of calls, against 4.0 % for rotation. The difference of 3.2 points has p = 1.5e-35.
- W/Z/t → q/g flow is 3.4 % (rot 1.5 %).
- The alternative prediction of the plan (20–40 % of calls changed, flow to QCD) is excluded for every model tested. The largest change is 14.3 % (transformer).

**Where the change sits inside JEDI-linear**
- The named radial objects change by 0 (S0, S19) to 0.18 sd (S1). Their relative change is ≤ 0.08 for S29 and S30, and 0.19 for S1.
- The excess over rotation is carried by the two context-only units S58 and S31. These units read g, which is computed from the pooled half-plane P units.
- The half-plane statements flip on/off for about 48 % of particles under both rotation and scramble. In the S units with per-particle half-plane input, this cancels to ≤ 0.18 sd.
- The A' − A difference per unit is ≤ 0.06 sd. So the M1 (direction) term contributes little here, and the residual is not removed by holding M1.

**Prong information at fixed radial profile**
- Real jets with τ21 higher by 0.31, at matched m, M0, M2 and pT, change JEDI's call 5.2 % more often than τ21-matched jets. This is the same size as A' on those jets.
- When Σ pT r and multiplicity are also matched, the τ21-attributable change is −0.3 % [−1.4, 0.6]. This is limited to 351 pairs, mostly top jets.

**Cross-model comparison**
- Deep Sets and PFN show the same call change (7.3–8.7 %) and similar flows as JEDI. The retrained JEDI copies show 6.2–9.1 % (excluding the collapsed seed).
- The near-invariance of calls to azimuthal redistribution at fixed radial moments is therefore not specific to JEDI-linear.
- One difference is at the latent level: JEDI's S changes more under scramble than under rotation (0.47 vs 0.21 of spread), while the Deep Sets / PFN latents change less (0.50–0.54 vs 0.58–0.62).
- The transformer changes 13.7–14.3 % of calls. It is also 5–8 points less accurate.

**B (cut-aligned cells)**
- B changes fewer calls than rotation in every model (1.3–2.5 %). The rotated-cell control Br gives the same values (JEDI 2.6 vs 2.5 %).
- So keeping each particle inside JEDI's own half-plane cells has no measurable effect beyond limiting the angular displacement to the sector width.
- B preserves the half-plane on/off state for 96–98 % of particle-cut pairs, not 100 %. The reason is the 0–8 mrad line offsets and particles near the axis.

## Predictions vs outcome
| hypothesis | prediction | outcome |
|---|---|---|
| object thesis (JEDI) | calls ≤ 5–8 %, near rot; B ≈ A'; radial S rel change < 0.05; residual in odd/direction units that vanishes under A' | 7.2 % vs rot 4.0 % (different, p 1.5e-35). B (2.5 %) < A' (7.2 %). Radial S: 0–0.08, S1 0.19. Residual sits in context units S58/S31 and persists under A'. |
| pooled latent / prong use | 20–40 %, flow to QCD, spread across many S units | excluded (≤ 14.3 % in all models; JEDI excess concentrated in 2 of 19 units) |
| radial invariance specific to JEDI-linear | Deep Sets / PFN change ≫ JEDI | not found: Deep Sets / PFN 7.3–8.7 % vs JEDI 7.2 % |

## Changes to the plan
- Added At (M1 hold only), B' and Br.
- Added the radial-matched calibration (Σ pT r, n).
- B for the X12 and transformer models was not run: the cells are defined from the official l1 weights. The main group applies the official cells to Deep Sets and PFN.
- The l1 "13 cuts" read-out uses mean ablation: 13 units with ≥ 1 % calls changed, the same as the nonzero l1 units.

## Issues
- The A' rescale moves r by a median of 7 mrad. A (exact r) gives a similar result for JEDI (6.9 %).
- The radial-matched calibration has small N and is class-imbalanced.
- For S58 and S31 (context-only), the jet-level quantity they read is not named here. A candidate is the anisotropy of the nine pooled half-plane P values, which the scramble makes near-isotropic.
- No figure for the flows.

## Explainer
`E24_EXPLAINER` added to `site/explain.py` (defined before E23_EXPLAINER; not yet registered in PAPER_EXPLAINERS — left to the ledger step).
