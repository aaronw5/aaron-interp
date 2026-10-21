# E35v — skeptic check of E35 (route of JEDI-linear's response to two-prong momentum sharing at fixed jet four-momentum)

Script `interp/e35v_verify.py` (imports E35's edit, tail and projection code; new code for the per-cell nulls, the dilation positive control, the learned holds and the W+Z-only probe). Cluster job `e35-verify` (1 CPU / 2 GiB; wall 412 s; peak 1,894 MiB). Results `results/jedi/e35_verify.json`, log `logs/e35-verify.log`.

## Conclusion
On a fresh set of 4,000 W and Z jets, JEDI-linear again moves W jets toward Z and Z jets toward W by about 1.5 log-odds when the momentum split between the two prongs is made more lopsided at fixed jet four-momentum, while rotations that keep the split fixed move it by at most 0.16. Holding the network's mass direction fixed removes about 70–90 % of the response to a real 10 % mass reduction but almost none of the response to the lopsided split, so this response does not travel through the network's mass direction, although a linear mass probe reads the edited jets as about 10 % lighter. E35's statement that no single internal object removes more than 0.30 of the lopsided-split response holds only for the largest edit; at the smaller edit one object (S60) removes about half of the Z-jet response. Which combination of internal objects carries the response remains open.

## What was checked and why
E35 claims (i) a specific W/Z response to D at fixed jet four-momentum, (ii) that no named object or single S unit carries the dc > 0 response, (iii) that holding the mass-like object S29 or the h4 mass direction removes nothing, and (iv) in its thesis note, that "the response passes through the network's mass representation", based on a probe read-out. A hold that removes nothing is only informative if the same hold removes a response that is known to go through the held element. E35 had no such positive control, and it compared its holds against random-hold nulls only for the pooled share (which mixes cells of opposite sign), not per cell.

## Terms
- **margin** $=\ell_W-\ell_Z$: difference of the network's W and Z logits (log of $p_W/p_Z$). **Δmargin** = edited minus unedited, per jet.
- **D(dc)**: H45's edit. The jet's constituents (massless, $(p_T,\eta=\Delta\eta,\phi=\Delta\phi)$) are split into two exclusive-$k_T$ prongs, boosted to the jet rest frame, rotated so that $|\cos\theta^*|$ (cosine of the angle between the prong axis and the jet flight direction; 0 = equal momentum sharing) changes by dc, and boosted back. One Lorentz transformation for every particle: jet four-momentum, both prong masses and N fixed. dc > 0 = more lopsided split.
- **cn**: same rest-frame angle, rotation about the prong axis (both prong four-momenta fixed). **cphi**: same angle about the jet flight direction. **R**: rigid rotation of the jet about its axis with D's mean displacement.
- **S units**: the 64 second-round pooled objects (S = Σ_i u_i / 16), 19 of them live. S29 = pT-weighted squared radius + Σr² + soft count (mass-like); S1 = Σ over ranks ≥ 1 of (r − 0.2)₊; S30 = Σ r over the core; S19 = soft multiplicity; S0 = pT sum of the 14 hardest; S31 / S58 = g-thresholded dipole / quadrupole counts; S60, S56, S51 not named (S56 has the largest shift under D).
- **hold**: the edited jet is run through the exact fixed-point tail S → h4 → h5 → z → logits with one element replaced by its unedited value (for a direction: the component along it). **share removed** (per cell) = $1-\overline{\Delta\text{margin}}_\text{held}/\overline{\Delta\text{margin}}_\text{free}$; 1 = the hold removes the whole response, negative = the hold makes it larger. **cell** = (dc, true class).
- **h4dir**: E35's mass direction at h4, the mean standardised h4 displacement under a real mass change (all angles × 1.05, "dilation"; it multiplies the invariant mass by ≈ 1.05 at fixed pT and N).
- **R (calibration ratio)** = mean Δmargin under D divided by mean Δmargin under a real dilation of the size the h4 mass probe reads for D.
- **McNemar**: paired test on jets whose call changes under one edit but not the other. **Holm**: step-down correction for many p-values.

## Procedure
- Sample: `jedi_core.sample(2000, 3601)` (seed not used by E35, which used 3501 / 3502); W and Z jets only, N = 2,000 + 2,000. All 4,000 have ≥ 3 particles and pass H45's acceptance at every dc.
- Inputs rebuilt about each jet's own E-scheme axis for edited and unedited jets (E35's `make_edits`; D identical to H45's `edit_one`, max |diff| 0).
- Code checks: real particles occupy the first N slots of every stored jet (padding contiguous: yes), so `P4[j, :N]` is the whole jet. Independently re-measured on 300 jets at dc +0.3: jet four-momentum max relative change 1.6e−13; both prong masses under D max change 6.7e−7 GeV; both prong four-momenta under cn max change 1.9e−10 GeV; invariant mass under D max change 6.6e−6 GeV. Tail free check (tail on the edited S vs the full network): 0. Holding all 19 live S removes 1.000 in every cell.
- Bootstrap: 1000 jet resamples. Sign-flip tests: 2,000 permutations (smallest attainable p 5e−4).

## Results

### A. Response to D and the controls (fresh sample; Δmargin, 95 % CI)
| dc | D, W jets | D, Z jets | W→Z (D) | Z→W (D) | cn W / Z | cphi W / Z | R W / Z |
|---|---|---|---|---|---|---|---|
| −0.30 | +0.62 [0.56, 0.68] | −0.18 [−0.26, −0.09] | 0.4 % | 3.9 % | +0.01 / +0.15 | −0.00 / −0.05 | −0.02 / −0.01 |
| −0.15 | +0.50 [0.45, 0.55] | −0.15 [−0.22, −0.08] | 0.3 % | 2.6 % | −0.03 / +0.05 | −0.01 / −0.02 | −0.03 / −0.03 |
| +0.15 | −0.76 [−0.81, −0.70] | +0.67 [0.58, 0.75] | 4.3 % | 2.4 % | −0.01 / +0.05 | −0.02 / −0.04 | −0.02 / −0.02 |
| +0.30 | −1.54 [−1.62, −1.45] | +1.55 [1.44, 1.65] | 7.7 % | 3.5 % | +0.02 / +0.16 | −0.00 / +0.01 | −0.04 / −0.02 |

- E35 main: +0.3 W −1.48, Z +1.67; −0.3 W +0.58, Z −0.06 (rep −0.18).
- H0: D's mean Δmargin equals the control's (paired, per jet). D − cn at +0.3: W −1.55 [−1.64, −1.46], Z +1.38 [1.28, 1.49], sign-flip p ≤ 5e−4; McNemar (calls changed by D only vs by cn only) W 534 vs 42, Z 491 vs 57, p < 1e−15. At dc +0.15, D − cn W −0.75, Z +0.62, p ≤ 5e−4. At −0.15 on Z jets, the McNemar p is 0.10 (margin difference −0.19, p 5e−4).

### B. Holds per cell (fresh sample; share removed, 95 % CI)
| hold | +0.15 W | +0.15 Z | +0.3 W | +0.3 Z | pooled (E35 formula) | pooled over dc > 0 cells only |
|---|---|---|---|---|---|---|
| S29 | −0.36 [−0.43, −0.29] | −2.02 | −0.34 [−0.38, −0.29] | −1.14 | −0.38 | −0.87 |
| S1 | +0.21 [0.07, 0.34] | −0.55 | +0.10 [0.01, 0.18] | −0.32 | +0.51 | −0.12 |
| radial {S1, S30, S19} | +0.49 [0.36, 0.61] | −0.78 | +0.35 [0.27, 0.43] | −0.48 | +0.52 | −0.08 |
| named7 | +0.98 [0.89, 1.09] | −2.44 | +0.61 [0.56, 0.66] | −1.68 | +0.00 | −0.57 |
| S29 + S1 | +0.99 [0.89, 1.09] | −2.11 | +0.66 [0.60, 0.71] | −1.47 | +0.05 | −0.43 |
| S60 | −0.15 | **+0.52 [0.46, 0.61]** | −0.15 | +0.30 [0.27, 0.34] | +0.31 | +0.11 |
| S56 (largest ΔS) | −1.07 | −0.42 | −0.67 | −0.28 | −0.10 | −0.56 |
| h4dir | +0.08 [0.04, 0.12] | −0.58 | +0.03 [0.01, 0.06] | −0.38 [−0.44, −0.32] | +0.07 | −0.19 |
| h5dir | −0.03 | +0.05 | +0.00 | +0.06 | +0.13 | +0.02 |
| mass = S29 + h4dir | −0.24 | −1.43 | −0.21 | −0.80 | −0.11 | −0.60 |
| S0, S19, S31, S58 | \|share\| ≤ 0.11 in every dc > 0 cell | | | | | |

- E35's pooled shares are reproduced (radial 0.516 vs 0.51; S1 0.505 vs 0.51; S29 −0.38 vs −0.34; h5dir 0.13 vs 0.17). Restricted to the dc > 0 cells, the pooled share of S1 is −0.12 and of the radial set −0.08: the ≈ 0.5 comes from the dc < 0 cells, as E35 states.
- **Per-cell nulls (new).** For each dc > 0 cell, all 19 single live-S holds, all 171 pairs, all 969 triples, 500 random 1-dim and 3-dim live-S directions, 500 random h4 directions. Largest single: S1 0.21 / 0.10 (W, +0.15 / +0.3); **S60 0.52 (Z, +0.15)** / 0.30 (Z, +0.3). Pairs with share ≥ 0.5: W 1 of 171 (S29+S1) at both dc; Z 10 of 171 at +0.15, 0 at +0.3. The best triples all contain S29+S1 (W, up to 1.18) or S60 (Z).
- H0: the radial set removes no more than a random triple. W cells: p 0.022 (+0.15) and 0.022 (+0.3) uncorrected, Holm 0.41 over 20 tests; Z cells p 0.83–0.84. S29 vs random 1-dim directions p 0.88–1.0; h4dir vs random h4 directions p 0.06–1.0 (Holm 1).
- E35 main sample (from `e35_singles.json`) at dc +0.15: S60 0.47 (Z), S29+S1 0.91 (W), 6 Z pairs ≥ 0.5. The fresh sample agrees.

### C. Positive control of the hold method (new): the same holds on a real mass change
Edit: all Δη, Δφ × 0.90 (mass −10 % at fixed pT and N; Δmargin W −1.11, Z +7.34; 81 % of Z calls change) and × 1.10 (W −5.49, Z +0.26).

| hold | share of the × 0.90 response removed, W / Z | share of the × 1.10 response removed, W |
|---|---|---|
| h4dir | **+0.69 [0.67, 0.72] / +0.87 [0.86, 0.87]** | +0.95 |
| S29 | +0.92 / +0.45 | +0.56 |
| mass = S29 + h4dir | +0.91 / +1.00 | +0.97 |
| radial | −0.45 / +0.13 | −0.05 |
| S1 | −0.57 / +0.08 | −0.17 |

(The × 1.10 Z cell has a free response of only +0.26 and its shares are unstable.)
- Reading: the h4dir and mass holds remove 70–100 % of the network's response to a real mass change, so the hold method does detect a mass route when one exists. Under D at dc > 0 the same holds remove +0.03 to +0.08 (W) and −0.38 to −0.58 (Z), and the mass set −0.21 to −1.43.
- The direction of D's mean standardised h4 displacement has cosine 0.10–0.21 with h4dir (all four dc > 0 cells).
- Competing hypotheses for the dc > 0 response: (M) it goes through the h4 mass direction, predicted h4dir share near the dilation value (≈ 0.7–0.9): excluded in all four cells (CIs ≤ 0.12). (B) distributed over several S objects: consistent.

### D. Learned low-dimensional holds (new; fitted on a random half of the cell's jets, held on the other half)
| cell | mean ΔS direction (k = 1) | top-3 SVD of ΔS (k = 3) | random k=1 / k=3 q95 | p (k=1 / k=3) | mean Δh4 direction under D (k = 1) |
|---|---|---|---|---|---|
| +0.15 W | 0.30 [0.22, 0.39] | 0.36 [0.26, 0.46] | 0.28 / 0.34 | 0.035 / 0.055 | 0.25 |
| +0.15 Z | 0.61 [0.47, 0.77] | 0.95 [0.76, 1.14] | 0.37 / 0.48 | 0.020 / 0.010 | 0.55 |
| +0.3 W | 0.15 [0.05, 0.23] | 0.29 [0.20, 0.37] | 0.17 / 0.16 | 0.075 / 0.025 | 0.46 |
| +0.3 Z | 0.36 [0.27, 0.46] | 0.46 [0.35, 0.56] | 0.18 / 0.31 | 0.005 / 0.015 | 0.13 |

- The mean ΔS direction is dominated by S56 (loading 0.71), S1 (0.54) and S29 (0.30) in every cell.
- At the largest edit, a 3-dim S subspace fitted to D's displacement removes 0.29 (W) and 0.46 (Z), above random 3-dim subspaces. The subspace is the displacement direction, not a subspace optimised for the margin (DAS), so these are lower bounds for what a 3-dim S subspace can carry.

### E. Calibration (probe fitted on W+Z jets only, fit half / test half; E35 fitted on all five classes)
- Probe of log m on standardised h4: held-out R² 0.93; gain 1.03 per unit log dilation.
- dc +0.3: probe Δlog m̂ −0.083 (W) and −0.096 (Z) with true Δm = 0 (E35: −0.124 / −0.135 with the all-class probe). R(Z) = 0.22 (E35 0.22, rep 0.20); R(W) = 2.3 (E35 1.50, rep 2.88; the W denominator is small: −0.67).
- The size of the probe read-out depends on the jets the probe is fitted on (8–10 % here, 12–13 % in E35). By section C, the probe's read-out under D is not a causal mass signal: holding the mass direction leaves the D response in place.

### F. Mask program (float + HGQ zero-bit masks) at dc +0.3
- Reproduces 0.86 (W) and 0.83 (Z) of the network's call flips (E35: 0.87 / 0.82).
- McNemar: W 80 vs 80 (p 1.0), Z 97 vs 94 (p 0.88). E35's Z p 0.004 does not reproduce on this sample.

## Code review of `e35_prong_sharing_route_jedi.py` and `e35_plot.py`
- No bug found in the edits, stage handling, masks or tail. D matches H45's `edit_one`. Padded slots are excluded: P4 is multiplied by the real mask, real slots are contiguous, and `model_x` zeroes padding. The tail free check is 0. The projection holds are applied in SD-standardised coordinates, and the clipping rule (only at projected, non-diagonal layers) is as described. The g hold overrides g before the `add` quantizer.
- Calibration: the probe is fitted on all five classes, but R is evaluated on W+Z test jets. This is not a bug, but the size of the read-out depends on the choice (see E).
- `e35_plot.py`: the docstring says panel (b) shows each hold's 95th-percentile random-hold null, but the figure does not draw it. The panel (b) title says "holding one element" for set holds too. The untrained band is loaded but not plotted. In panel (b), only the dc −0.3 W and dc +0.3 cells are shown; the dc +0.15 cells, where S60 removes ≈ 0.5 of the Z response, are not shown.
- Figure `e35_edit.png`: a correct depiction of one W jet under D and cn (prong pT 791/204 → 931/64 GeV, same mass 82.0 GeV, the softer prong spread to wider angles). Figure `e35_route.png` (a) matches the tables.

## Assessment of E35's reading
Reproduced on a fresh sample:
- D response and its sign pattern: W −1.54 / Z +1.55 at +0.3.
- Smallness of the cn, cphi and R controls (≤ 0.16).
- W toward W at dc < 0.
- Pooled shares ≈ 0.5 for S1 and the radial set, driven by dc < 0.
- No significant set after Holm.
- Only S29+S1 among the W pairs at +0.3.
- R(Z) ≈ 0.22.
- Mask-program fidelity.

Needs correction:
1. "In the dc > 0 cells …, no named object and no single live S unit removes more than 0.30" holds only for single units at dc +0.3. At dc +0.15, S60 removes 0.47 (main) / 0.52 (fresh) of the Z response, and 6 / 10 pairs remove ≥ 0.5 for Z. The named sets named7 and S29+S1 remove 0.6–1.0 of the W response at both dc > 0, but the same holds increase the Z response by a factor of 2.5–3.4 (shares −1.5 to −2.4). These are class-asymmetric, off-manifold effects, not a route.
2. "Holding S29 or the h4 dilation direction removes nothing or increases the response" holds in the dc > 0 cells and pooled. In the dc < 0 W cells, h4dir removes 0.5–0.8 and S29 overshoots (2.5).
3. The thesis-note sentence "The response passes through the network's mass representation: the h4 probe reads −12 % with the true mass fixed" is not supported. With the positive control, the h4 mass-direction hold removes 0.69–0.95 of a real mass change's response but 0.03–0.08 (W) and −0.38 to −0.58 (Z) of D's dc > 0 response. D's h4 displacement has cosine 0.10–0.21 with that direction. The probe read-out is a correlational read: its size depends on the probe's fit set (8–13 %), and it does not mark the causal path. The report's Conclusion sentence "the response passes through the network's mass representation in a way that the hold tests used here do not isolate" should be replaced: the hold test isolates a real mass change's route, and D's dc > 0 response does not use it.

Corrected reading (for the ledger):
- At fixed jet four-momentum, a more lopsided two-prong split moves W jets toward Z and Z jets toward W: −1.5 / +1.5 log-odds at dc +0.3 on two independent samples. Same-angle rotations that keep the split fixed (cn, cphi, R) move the margin by ≤ 0.17.
- In the dc > 0 cells, no single live S unit removes more than 0.30 of the response at dc +0.3. At dc +0.15, S60 removes about 0.5 of the Z-jet response.
- Holding the network's mass direction at h4 removes 70–95 % of the response to a real mass change but ≤ 0.08 of the W response to D and none of the Z response. The dc > 0 response therefore does not run through the mass direction, although a linear h4 mass probe reads the edited jets as 8–13 % lighter.
- A 3-dim S subspace fitted to D's displacement (S56, S1, S29 dominant) removes 0.29 (W) and 0.46 (Z) at dc +0.3. The carrier is spread over several S objects and is not yet identified.

## Explainer entry (for site/explain.py, Paper A / D18 tab)
- **Question.** When JEDI-linear changes its W/Z call because the momentum split between the two prongs became more lopsided (jet mass unchanged), does that change travel through the network's internal mass variable?
- **Background.** At fixed jet mass $m$ and $p_T$, the angular separation of two prongs grows as the split becomes lopsided, $\Delta R \approx m/(p_T\sqrt{z(1-z)})$, with $z$ the momentum fraction of the softer prong. The pT-weighted second moment $M_2=\sum_i p_{T,i} r_i^2 \approx m^2/p_T$ is nearly unchanged (+0.12 %), but the unweighted angular spread of the soft prong grows.
- **Method.** Edit D: one Lorentz transformation per jet that changes $|\cos\theta^*|$ by dc in the jet rest frame, so the jet four-momentum, prong masses and multiplicity are exactly fixed. Hold: replace one internal element by its unedited value and measure the share of the margin change that disappears, $1-\overline{\Delta\text{margin}}_\text{held}/\overline{\Delta\text{margin}}_\text{free}$. Positive control: apply the same hold to a real 10 % mass change (all angles × 0.9).
- **Result.** Holding the h4 mass direction removes 0.69 (W) and 0.87 (Z) of the response to a real mass change, but 0.03 (W) and −0.38 (Z) of the response to D at dc +0.3 (fresh sample, N = 2,000 + 2,000).
- **Checks.** Jet four-momentum fixed to 1e−13 and prong masses to 1e−6 GeV. Holding all live S removes 1.000. Random-hold nulls are computed per cell. Seeds 3501, 3502 (E35) and 3601 (this check).
- **Why it matters.** The network's W/Z decision depends on how momentum is shared between the prongs through a path other than its mass variable. A tagger decorrelated from mass can therefore still respond to the energy sharing, and an interpretation of JEDI-linear as "mass windows plus ΣpT" misses this path.

Glossary additions: share removed (hold), positive control (hold on a real dilation), learned hold (subspace fitted to an edit's mean displacement), cell (dc × true class).
