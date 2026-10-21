# E35 — Route of JEDI-linear's response to two-prong momentum sharing at fixed jet four-momentum

Node E35 (stage 3; claims A4, A3, D18 handed from loop D). Status: done (one rerun after an evicted pod and a hold-clipping fix; see Issues).
Script `interp/e35_prong_sharing_route_jedi.py` (cluster jobs e35-run3 = final main run, e35-singles = per-cell single/pair holds; e35-run = first run, superseded), figures `interp/e35_plot.py`.
Results `results/jedi/e35_prong_sharing_route.json`, `e35_prong_sharing_route_perjet[_rep].npz`, `e35_singles.json`, figures `results/jedi/e35_edit.png`, `results/jedi/e35_route.png`. Logs `logs/e35-run3.log`, `logs/e35-singles.log`.

## Conclusion
JEDI-linear moves W jets toward Z and Z jets toward W when the momentum split between a jet's two prongs becomes more lopsided, even though the jet's four-momentum and mass do not change at all. Rotations of the same size that keep the split fixed have almost no effect. No single internal object carries this response for the lopsided direction: holding any one named object (the radial sums S1, S30 and S19, or the mass-like object S29) at its unedited value removes at most about a third of the response, and holding the mass-like objects makes it larger. The network's own mass read-out does move (it reads the edited jet as about 12 % lighter), so the response passes through the network's mass representation in a way that the hold tests used here do not isolate. The extracted float-with-masks program reproduces 82–87 % of the network's call flips at the largest edit.

## Question and claims tested
Loop D (H45/H45v) found that JEDI-linear changes W/Z calls under an edit that keeps the jet four-momentum exactly fixed. Thesis prediction handed from loop D: the response is carried by the unweighted radial / count objects (S1, S30, S19), not by the pT-weighted squared-radius mass object S29. The predicted outcomes were: the radial set removes ≥ 0.5 of the response, the mass set removes < 0.2, and the extracted program reproduces ≥ 70 % of the call flips.

## Procedure

**Network.** JEDI-linear official n64, 3-feature, fixed point (`interp.jedi_core`). Accuracy on the stored inputs: 80.68 % on the main sample and 80.84 % on the replicate. Accuracy on inputs rebuilt about each jet's own E-scheme axis (the axis given by the sum of the constituent four-momenta): 80.63 % and 80.72 %; 98.5 % / 98.8 % of calls are the same as on the stored inputs. All comparisons below use the rebuilt inputs for both edited and unedited jets.

**Samples.**
- Main: `sample(2000, 3501)`, 10,000 jets (2,000 per class). W and Z jets are primary; q, g and t are reported as secondary.
- Replicate: `sample(2000, 3502)`. The replicate repeats every measurement except the untrained redraws and the program fidelity.

**Edit D(dc)** (H45's `edit_one`; identical to it, max |difference| 0 on the checked jets).
1. Treat each constituent as massless, with (pT, η = Δη, φ = Δφ).
2. Split the jet into two exclusive-kT prongs (kT clustering with p = 1, E-scheme recombination, rapidity–azimuth distance; every constituent goes to one prong). The numpy port of the clustering matches `h41_colour_flow._ktn_labels` on 30 of 30 test jets.
3. Boost all constituents into the jet rest frame. There, $|\cos\theta^*|$ is the cosine of the angle between the prong axis and the jet's flight direction: 0 means the two prongs share the momentum equally, values near 1 mean one prong takes nearly all of it.
4. Rotate every constituent by $\alpha = \arccos c_0 - \arccos c_1$ about $\hat n_+ \times \hat\beta$, with $c_1 = \mathrm{clip}(c_0 + dc, 0, 0.95)$ for dc ∈ {−0.3, −0.15, +0.15, +0.3}. Boost back.

All constituents undergo the same Lorentz transformation, so the jet four-momentum (pT, η, φ, m), every pairwise invariant mass (hence each prong mass) and the multiplicity N stay exactly fixed. What changes is how the energy is shared between the prongs, and the lab-frame angles.
- On W/Z jets the mean $c_0$ is 0.46 and the realised mean change is +0.261 at dc = +0.3. The median rest-frame angle at dc = +0.3 is 19.8°.
- Measured invariants: the largest relative change of the jet four-momentum is 1.1e−12, and the largest change of the invariant mass is below 0.001 GeV.
- About the jet axis, $M_0 = \sum p_T$ changes by +0.002 % on average (largest 0.33 %), and $M_2 = \sum p_T r^2$ by +0.12 % on average (largest 36 % in one jet).
- All jets passed the H45 acceptance (both prong centres within ΔR 0.75 of the axis).

**Example jet (Figure `e35_edit.png`, main index 59).** A W jet with N = 26, pT 993 GeV and m 82.0 GeV.
- At dc = +0.3, $|\cos\theta^*|$ goes from 0.60 to 0.90, and the prong pT values go from (791, 204) to (931, 64) GeV.
- The prong masses (17.5 and 9.9 GeV) and the jet mass are unchanged. The softer prong spreads to wider angles (its angular size scales roughly as m/pT).
- The call goes W → Z, with $\ell_W-\ell_Z$ going from +3.90 to −0.35.
- The cn control on the same jet gives +4.20, call W.

**Controls** (same jets, same rest-frame angle |α|; random sign):
- **cn**: rotation about the prong axis $\hat n_+$ in the rest frame. Both prong four-momenta stay exactly fixed (checked: max |Δ| 4e−14 GeV), so z, $|\cos\theta^*|$ and ΔR are unchanged; the constituents turn about their own prong axis.
- **cphi**: H45's control, a rotation about $\hat\beta$. The momentum split is fixed and the prongs move around the jet axis.
- **R**: a rigid rotation of the whole jet about its axis in (Δη, Δφ). The angle θ is set so that the pT-weighted mean constituent displacement equals D's: $2\sin(\theta/2)\sum p_T r/\sum p_T$ = D's displacement, which is 0.035 rad on average at dc +0.3. This keeps every r, pT, M0, M2 and N fixed.

**Readouts.**
- Margin = $\ell_W-\ell_Z$ (the difference of the d7 logits, i.e. log p_W/p_Z). Δmargin = edited minus unedited, per jet.
- W→Z = % of W jets whose call goes from W to Z; Z→W likewise.
- ΔS of every live S unit, in units of its standard deviation (SD) over unedited W+Z jets.

**Route holds.** The edited jet is run with one element set to its unedited value; the tail S → h4 → h5 → z → logits is the exact fixed-point tail (check: max |logit difference| 0).
- Named elements:
  - S29 = pT-weighted squared radius + Σr² + soft count (the mass-like object).
  - S1 = $\sum_{\text{ranks}\ge1}(r-0.2)_+$.
  - S0 = pT sum of the 14 hardest particles.
  - S19 = soft multiplicity.
  - S30 = Σ r over the core.
  - S31 / S58 = dipole / quadrupole counts thresholded by g.
- Sets: named7 (all seven), radial = {S1, S30, S19}, g (all 64 context units, held through a full forward pass).
- Mass directions: h4dir / h5dir = the mass direction at h4 / h5, held by projection in SD-standardised coordinates, $h' = h - ((h-h_0)\cdot d)\,d$. The X10 DAS rotations were not saved, so d is the mean standardised displacement of h4 (h5) under a real mass change: dil_M2 = all angles × 1.05, on the same W+Z jets. The top h4 units are h4:47, 30, 22, 49; the top h5 units are h5:26, 13, 15.
- Combined mass sets: mass = S29 + h4dir; mass5 = S29 + h5dir.
- Range clipping (to the observed range, to avoid fixed-point wrap; X10 debug 2) is applied only at a projected layer. Clipping occurred for 0 % of S values.

**Share removed.** Share = 1 − P_held / P_free, where P = Σ over the 8 cells (dc × {W, Z}) of s_cell × mean Δmargin and s_cell = sign of the free mean. A per-class sign is needed because W and Z jets move in opposite directions. Per-cell shares (1 − held mean / free mean) are also reported. 95 % CI: 1000 jet-bootstrap resamples.

**Random-hold nulls** (point estimates of the same pooled share):
- 'sets': size-matched live-S unit sets — the 19 singles, all 969 triples, and 1000 random 7-sets.
- 'dirs': 1000 random k-dimensional subspaces of standardised live-S space (k = 1, 3, 7), or of live h4 / h5 (k = 1).
- mass null: a random live-S single plus a random h4 direction.
- p = (1 + #null ≥ named)/(1 + #null). The primary p uses the 'sets' null when it has ≥ 900 members, otherwise 'dirs'. Holm correction over the 12 holds that have a null.

**Supplement (e35-singles, main sample).** Per-cell shares for each of the 19 live S units (with CI) and each of the 171 pairs.

**Program fidelity** (main; same D edits):
- (i) mask: the float program with HGQ's zero-bit entries as masks (E3m 'mask').
- (ii) chain: the E11 hinge-formula chain (b), P ← physics, g ← P̂, S ← ĝ + physics, h4 ← Ŝ, h5 ← ĥ4, z ← ĥ5. The network's last layer d7 is kept, and the physics inputs are recomputed from the edited inputs (`jetdistill.observables.compute` + handbuilt).
- (iii) chainS: the network's own P and g, with formulas from S onward.

Per program: the share of network call flips that the program also flips ("reproduced"), with the same new call, and the share of program flips that are also network flips. McNemar test (changed by network vs by program, paired over jets). Correlation of Δmargin.

**Untrained control.** `Jedi(untrained=True)`, seeds 0–9: free D and cn responses in units of each network's own margin SD. Seeds 7 and 8 give a constant margin on every jet and are excluded from the SD-unit band (8 redraws remain).

**Calibration.**
- Probe: ridge regression of log m (invariant mass) on standardised h4, fitted on a random half of the unedited jets and scored on the other half. Held-out R² = 0.972. Probe gain = 1.21 per unit log dilation (from ±5 % dilations).
- Matched dilation: per (dc, class), angle factor $k = \exp(\Delta\log\hat m_D/\text{gain})$.
- R = mean Δmargin(D) / mean Δmargin(dil matched): the network's response to D divided by its response to a real mass change of the same size as the mass change that D appears to make in the probe read-out.

**Hypotheses.**
- H0: the D response lies within the cn / cphi / R control and the random-hold band (no specific carrier).
- H1 (thesis): the radial set removes ≥ 0.5, the mass set < 0.2, and the program reproduces ≥ 70 % of flips.
- Alternative A (mass variable): the mass-set hold removes most of the response.
- Alternative B (distributed): no single set beats size-matched random holds.

## Results

### 1. Response to D vs controls (main; Δmargin, 95 % CI; W→Z among W jets, Z→W among Z jets)
| dc | W jets: D | Z jets: D | W→Z (D) | Z→W (D) | cn W / Z | cphi W / Z | R W / Z |
|---|---|---|---|---|---|---|---|
| −0.30 | +0.584 [0.525, 0.643] | −0.059 [−0.138, 0.020] | 0.5 % | 3.5 % | −0.006 / +0.117 | −0.001 / +0.057 | −0.016 / +0.053 |
| −0.15 | +0.461 [0.410, 0.513] | −0.087 [−0.151, −0.025] | 0.6 % | 2.5 % | −0.007 / +0.060 | −0.009 / +0.046 | −0.008 / +0.037 |
| +0.15 | −0.747 [−0.804, −0.688] | +0.779 [0.697, 0.859] | 4.5 % | 2.1 % | +0.014 / +0.040 | −0.004 / +0.029 | +0.012 / +0.049 |
| +0.30 | **−1.483 [−1.562, −1.400]** | **+1.666 [1.558, 1.770]** | **7.8 %** | **4.0 %** | +0.031 / +0.165 | +0.014 / +0.043 | +0.007 / +0.058 |

- **D minus cn** at +0.3: W −1.514 [−1.594, −1.425] and Z +1.501 [1.394, 1.610]. Sign-flip p = 2e−4, the smallest attainable with 5,000 permutations. McNemar on calls changed (D only vs cn only): W 536 vs 41, Z 478 vs 45, p < 1e−15.
- On the main sample, every D − control difference at every dc is nonzero (sign-flip p ≤ 0.011; McNemar p ≤ 0.04). On the replicate, McNemar for D vs cn on W jets at dc −0.15 gives p 0.079. Against cphi and R the margin differences are of the same size as against cn.
- **Replicate:** +0.3 gives W −1.568, Z +1.593; −0.3 gives W +0.603, Z −0.180.
- **Secondary classes** at +0.3: t −0.22, q −0.04, g −0.05.
- **Reading:** the response is asymmetric in dc. A more lopsided split (dc > 0) moves W jets toward Z and Z jets toward W. A more balanced split (dc < 0) moves W jets further toward W and changes Z jets little.

### 2. What D changes inside the network (main, dc = +0.3, ΔS in SD units, W / Z jets)
- Largest S shifts: S56 +4.15 / +5.33, S1 +3.29 / +4.17, S29 +1.78 / +2.17, S31 +0.90 / +1.11, S62 +0.84 / +0.90, S17 +0.81 / +0.81, S60 +0.73 / +0.84.
- Smaller shifts: S30 −0.53 / −0.53, S19 +0.11 / +0.12, S58 +0.10 / +0.03, S0 +0.01 / +0.00.
- At dc = −0.3 all |ΔS| ≤ 0.63 SD (largest: S60 −0.54 / −0.63, S17 −0.56 / −0.51).
- The input moments barely move: M2 +0.12 % on average, |M1| +0.11 GeV·rad. S29 still rises by 1.8 SD, because S29 also contains Σr² and a soft count, which follow the lab-angle spread of the softer prong.

### 3. Route holds (share of D's margin response removed; main, replicate in brackets)
| hold (what it computes) | pooled share [95 % CI] | dc +0.3, W | dc +0.3, Z | dc −0.3, W | primary p (null) | Holm |
|---|---|---|---|---|---|---|
| S29 (pT-weighted r² + soft count) | −0.34 [−0.40, −0.28] (−0.38) | −0.33 | −1.05 | +2.51 | 0.98 (dirs) | 1 |
| h4 mass direction | +0.02 [−0.02, 0.06] (+0.07) | +0.04 | −0.40 | +0.79 | 0.34 (dirs) | 1 |
| h5 mass direction | +0.17 [0.15, 0.19] (+0.17) | −0.01 | +0.09 | +0.51 | 0.011 (dirs) | 0.13 |
| mass = S29 + h4 direction | −0.13 [−0.20, −0.08] (−0.10) | −0.24 | −0.80 | +2.18 | 0.92 (mass null) | 1 |
| S1 (Σ (r−0.2)₊, ranks ≥ 1) | **+0.51 [0.45, 0.56]** (+0.47) | +0.09 | −0.24 | +2.44 | 0.048 (dirs); 0.10 (top of 19 singles) | 0.53 |
| S30 (Σ r, core) | −0.08 [−0.09, −0.06] (−0.08) | +0.09 | −0.20 | +0.27 | 0.78 | 1 |
| S19 (soft count) | +0.004 [0.002, 0.006] | +0.00 | +0.00 | −0.01 | 0.56 | 1 |
| radial = S1 + S30 + S19 | **+0.51 [0.45, 0.57]** (+0.48) | +0.36 | −0.40 | +2.58 | 0.057 (969 triples; q95 0.52) | 0.57 |
| S0 (pT of 14 hardest) | −0.001 [−0.004, 0.002] | −0.00 | +0.00 | −0.01 | 0.58 | 1 |
| S31 / S58 (counts) | −0.02 / −0.004 | −0.03 / +0.01 | −0.06 / +0.00 | +0.07 / −0.00 | 0.65 / 0.59 | 1 |
| all 7 named S | −0.00 [−0.08, 0.07] (+0.01) | +0.60 | −1.51 | +2.69 | 0.72 (1000 7-sets) | 1 |
| g (all context units) | −0.05 [−0.06, −0.04] (−0.06) | +0.02 | −0.13 | +0.16 | – | – |

- **Null quantiles (95th percentile):** singles 0.33 (19 units), 1-dim directions 0.50, triples 0.52, 7-sets 0.59, h4 directions 0.11, h5 directions 0.11, mass null 0.43.
- **Pooled vs per-cell.** The pooled shares of S1 and the radial set (≈ 0.5) come from the dc < 0 cells. There, holding S1 overshoots: the held response points the other way, and shares exceed 1. In the dc > 0 cells, which carry the W→Z flips from H45, S1 removes 0.09 (W) and −0.24 (Z), and the radial set removes 0.36 (W) and −0.40 (Z).
- **Supplement (all 19 singles and 171 pairs, per cell, dc = +0.3).**
  - Largest single for W: S1 0.09 [−0.003, 0.18], then S30 0.09 and S40 0.08. Largest single for Z: S60 0.30 [0.27, 0.34], then S51 0.10.
  - Largest pair for W: S29+S1 0.62. The pair is not additive: S29 alone gives −0.33 and S1 alone +0.09. Largest pair for Z: S60+S51 0.37.
  - At dc = +0.15 the largest pairs are S29+S1 0.91 (W) and S60+S17 0.62 (Z).
  - Only 1 of 171 pairs removes ≥ 0.5 at dc +0.3 for W, and 0 of 171 for Z.
- **Holds and calls.** Holding S29 roughly doubles the number of changed calls among W+Z jets (2,274 → 4,268). Holding all seven named S raises it to 3,444. A held object at its unedited value combined with moved others puts the S vector off the set of states that ordinary jets reach.

### 4. Calibration against a real mass change (main; per class)
| dc, class | probe Δlog m̂ under D | true Δlog m | matched angle factor | Δmargin D | Δmargin dil (matched) | R [95 % CI] | rep R |
|---|---|---|---|---|---|---|---|
| +0.3, W | −0.124 | 0 | 0.903 | −1.48 | −0.99 | 1.50 [1.34, 1.67] | 2.88 [2.46, 3.43] |
| +0.3, Z | −0.135 | 0 | 0.895 | +1.69 | +7.73 | 0.22 [0.20, 0.24] | 0.20 [0.18, 0.22] |
| +0.15, Z | −0.056 | 0 | 0.955 | +0.77 | +3.37 | 0.23 [0.20, 0.26] | 0.24 [0.20, 0.29] |
| −0.3, W | +0.001 | 0 | 1.001 | +0.58 | −0.02 | −36 (dil ≈ 0) | −86 |

- The h4 mass probe reads the D-edited jets (dc = +0.3) as 12–13 % lighter, although the true mass is unchanged.
- For Z jets, the network's D response is 0.22 of its response to a real 10 % mass reduction (the reduction the probe reads).
- For W jets, the D response is 1.5–2.9 times the matched-dilation response. Here the denominator is small: a 10 % lighter W jet moves the margin by only −0.99.
- At dc < 0 the probe reads no mass change (|Δlog m̂| ≤ 0.008) while W jets still respond by +0.58, so R is unbounded.

### 5. Program fidelity (main; W / Z jets)
Unedited calls: the mask program has accuracy 80.48 %. The chain has accuracy 80.65 % and makes the same call as the network on 94.2 % of jets; chainS has 80.59 % and 94.1 %.

| dc | program | network flips reproduced (W / Z) | same new call | program flips that are network flips | McNemar (net only vs prog only), p | corr Δmargin | Δmargin prog vs net (W) |
|---|---|---|---|---|---|---|---|
| +0.3 | mask | **0.87 / 0.82** | 0.78 / 0.72 | 0.87 / 0.88 | W 77 vs 77, p 1.0; Z 102 vs 64, p 0.004 | 0.86 / 0.87 | −1.35 vs −1.48 |
| +0.3 | chain | 0.87 / 0.84 | 0.59 / 0.60 | 0.74 / 0.72 | W 80 vs 186, p < 1e−9; Z 90 vs 186, p < 1e−8 | 0.71 / 0.67 | −1.47 vs −1.48 |
| +0.3 | chainS | 0.82 / 0.81 | 0.61 / 0.63 | 0.76 / 0.71 | W p 0.006; Z p 4e−6 | 0.69 / 0.67 | −1.41 vs −1.48 |
| +0.15 | mask | 0.78 / 0.72 | 0.77 / 0.74 | 0.81 / 0.83 | W p 0.36; Z p 5e−4 | 0.87 / 0.82 | −0.66 vs −0.75 |
| −0.3 | mask | 0.68 / 0.73 | 0.80 / 0.79 | 0.69 / 0.70 | W p 0.91; Z p 0.53 | 0.80 / 0.79 | +0.40 vs +0.58 |
| −0.3 | chain | 0.57 / 0.54 | 0.68 / 0.62 | 0.55 / 0.47 | W p 0.71; Z p 0.10 | 0.60 / 0.48 | +0.59 vs +0.58 |

Base rate: 30 % of W jets change call under D at +0.3, so reproducing 87 % of the flips is well above what a program with uncorrelated flips at the same rate would reach. The chain reproduces the mean response (−1.47 vs −1.48), but it adds about 2.3 times as many flips that the network does not make.

### 6. Untrained network (free response in units of each network's margin SD)
- At dc = +0.3 the trained network gives W −0.32 and Z +0.36 SD. The 8 non-constant untrained redraws span W [−0.79, +0.13] and Z [−2.02, +0.25].
- The widest values come from seed 1, whose margin SD is only 0.0013, so its SD-unit values are large.
- Contrast between the classes, (Z response − W response): trained +0.67 SD; untrained −1.23 (seed 1) and −0.06 to +0.12 for the other seven.
- The opposite-sign response of W and Z jets therefore appears only in the trained network. Untrained cn responses: |·| ≤ 0.06 SD.

## Reading (scoped to these samples, edits and holds)
- **H0 (no specific response):** excluded for the network's response. At every dc, D moves the W−Z margin and the calls far beyond the cn, cphi and R controls (sign-flip p ≤ 2e−4 at |dc| = 0.3). This reproduces H45 (W −1.48 at +0.3; H45: −1.8 to −2.0 on another sample).
- **Thesis (radial set carries the response, ≥ 0.5; mass set < 0.2):** the pooled numbers meet both thresholds. The radial set removes 0.51 (rep 0.48) and the mass set −0.13 (rep −0.10). But the radial value is not above its size-matched random-triple null (p 0.057, Holm 0.57), and it comes from the dc < 0 cells. For the lopsided direction that carries the W→Z flips, the radial set removes 0.36 (W) and −0.40 (Z). The measurement supports the "mass set < 0.2" part. It does not support "radial set carries the dc > 0 response".
- **Alternative A (mass variable):** holding the mass-like object S29, the h4 mass direction, or both, removes nothing or raises the response (−0.34, +0.02, −0.13). The h5 mass direction removes 0.17 (p 0.011 against random h5 directions; Holm 0.13), almost entirely at dc < 0. The probe calibration shows that the network's h4 mass read-out does move under D (−12 %), with no true mass change. So D reaches the mass representation, but not along the dilation direction that the h4dir hold fixes, and fixing that one direction does not undo the response.
- **Alternative B (distributed):** for the dc > 0 response, no named object, no single live S unit (largest 0.09 for W, 0.30 for Z), and only one of 171 pairs (S29+S1, 0.62, W only) removes half or more. This pattern matches a response spread over several S objects (S1, S29, S56, S60 and others move by 1–5 SD) whose holds interact non-additively. The pooled named holds do not exceed size-matched random holds after Holm correction.
- **Program fidelity:** the mask program reproduces 82–87 % of the network's flips at dc = +0.3, with balanced disagreements for W (McNemar p 1.0). This meets the ≥ 70 % threshold at dc > 0 but not at dc = −0.3 (68–73 %). The E11 hinge chain reproduces the mean margin response at dc = +0.3 (−1.47 vs −1.48) and 84–87 % of flips, but adds more flips than the network makes (p < 1e−8). Both programs extracted on ordinary jets therefore carry the four-momentum-blind sharing response into these out-of-distribution edits.
- **Physics meaning:** at fixed jet mass, making the two-body split more asymmetric spreads the softer prong to wider angles (angular size ∝ m/pT of that prong). JEDI-linear reads this as "less W-like" for W jets and "less Z-like" for Z jets. Its internal mass representation reads the edited jets as lighter, although an exact mass calculation would not change.

## Issues
1. **Debug attempt 1 (hold clipping).**
   - What happened: the first full run (e35-run) clipped h4 and h5 to their observed range for every hold, including S-unit holds that do not project those layers. This changed the per-cell shares of S holds slightly (S29 at dc +0.3 W: −0.24 with the clip vs −0.33 exact).
   - Fix: clipping is now applied only at a projected layer; the results here are from e35-run3.
   - Effect: pooled shares changed by ≤ 0.01. The e35-singles supplement has no clip and agrees with e35-run3 per cell.
2. **Evicted pod.** e35-run2 was evicted by a node taint (TaintManagerEviction) before any output and was resubmitted as e35-run3.
3. **X10 DAS rotations not stored.** The "h4/h5 mass direction" is the mean standardised dilation displacement (a 1-dimensional direction), not the X10 DAS subspace. A hold of the DAS subspace itself would need X10 to be rerun with its rotations saved.
4. **Pooled share mixes directions.** The pooled share sums cells whose responses differ in sign and size (dc < 0 vs dc > 0, W vs Z). Per-cell shares are the more direct read-out of the W→Z mechanism and are given in Table 3 and the supplement. Shares in cells with small free responses (Z at dc −0.3: −0.06) are unstable, e.g. S1 10.1.
5. **Off-manifold holds.** Holds of single S objects raise the number of changed calls (S29: 2,274 → 4,268). A share above 1 or below 0 means the held state is not a state that ordinary jets reach. These are not removal fractions in the usual sense.
6. **R is unbounded for W jets at dc < 0 and +0.15.** The probe reads no mass change there, or the matched dilation moves the W margin by ≈ 0. R is informative only for Z jets and for W at dc +0.3.
7. **Untrained band.** Two of ten redraws (seeds 7, 8) have a constant margin. Seed 1's margin SD is 0.0013, which inflates its SD-unit values. The class contrast (Z − W) is the more stable comparison.
8. **Resources.** Pod peak memory reached the 2 GiB limit (2047 MiB reported, which includes page cache from the CephFS reads) in all three runs; no job was OOM-killed. CPU use averaged 0.4–0.7 cores. Model import on the cluster took 400–900 s per job.
9. **Kept as specified.** All jets passed the 0.75 prong acceptance (the hls4ml jets are narrow), so acceptance does not select jets here.

## Explainer (site/explain.py, Paper A; claims A4 / A3, D18)
**Question.** JEDI-linear changes W/Z calls when only the way momentum is shared between a jet's two prongs changes, with the jet's mass and momentum fixed. Which of its internal objects carries that response?

**Background.** In a hadronic W or Z decay, the two quarks share the jet momentum according to the decay angle $\theta^*$ in the boson rest frame: prong energies are $\tfrac{E}{2}(1\pm\beta\cos\theta^*)$. At fixed jet mass, a more lopsided split means a softer second prong that spreads to wider angles. Earlier work (E6, E27w) found that JEDI-linear forms an un-recentred $M_2$-like mass variable (S29 → h4/h5) and radial sums such as S1 = $\sum_{\text{rank}\ge1}(r-0.2)_+$.

**Method.**
1. On 2,000 W and 2,000 Z test jets, boost each jet into its rest frame and turn every particle by the same angle, so that $|\cos\theta^*|$ changes by dc ∈ {±0.15, ±0.3}; then boost back. One Lorentz transformation for all particles keeps $(p_T,\eta,\phi,m)$ of the jet and both prong masses exactly fixed.
2. Controls: the same turn about the prong axis (cn: the split is unchanged), a turn about the flight axis (cphi), and a rigid rotation of the jet (R).
3. Route test: rerun the edited jet with one internal object held at its unedited value, and measure the share of the margin change $\Delta(\ell_W-\ell_Z)$ that disappears. Compare with 1,000 random holds of the same size.
4. Also run the edit through the extracted programs (float + masks; E11 hinge chain) and through 10 untrained redraws.

**Result (figures `e35_edit.png`, `e35_route.png`).**
- At dc = +0.3, W jets move by −1.48 log-odds toward Z (7.8 % W→Z) and Z jets by +1.67 toward W (4.0 % Z→W); the three controls move both by less than 0.17.
- No single named object, and no single live S unit, removes more than 0.30 of this response.
- Holding the mass-like object S29 or the h4 mass direction removes nothing or increases it.
- The radial set {S1, S30, S19} removes 0.51 of the response pooled over all edits, but mainly for the balanced direction (dc < 0).
- The float-with-masks program reproduces 87 % (W) and 82 % (Z) of the network's call flips at dc = +0.3.

**Checks.**
- Jet four-momentum unchanged to 1e−12; jet mass unchanged to < 0.001 GeV.
- Replicate sample 3502: same responses (W −1.57, Z +1.59) and same hold shares within 0.07.
- The untrained networks show no opposite-sign W/Z response.
- An $h_4$ mass probe ($R^2$ 0.97) reads the edited jets as 12 % lighter. For Z jets, a real mass change of that size moves the margin about 4.5 times as much as the edit does ($R = 0.22$).

**Why it matters.**
- The network uses information beyond the jet mass and total momentum to separate W from Z.
- That information reaches the network's mass representation, but no single named object carries it, so a description of JEDI-linear as "mass windows plus radial sums" misses this response.
- The extracted programs reproduce most of it, which is an out-of-distribution check of those programs.

**Glossary additions:**
- $|\cos\theta^*|$: decay-axis angle in the jet rest frame; 0 = balanced split.
- D(dc): the rest-frame rotation edit.
- cn / cphi / R: the controls.
- Share removed: 1 − held response / free response.
- Per-cell share.
- R (calibration): network response to D ÷ response to a real dilation of the same probe-read size.
- Network flips reproduced: share of the network's call changes that the program also makes.
- Exclusive-kT prongs.
- E-scheme axis.

## Conclusion (for the round table row)
When the two prongs of a W or Z jet share their momentum more unevenly, with the jet's mass and momentum unchanged, JEDI-linear calls W jets more Z-like and Z jets more W-like (about 1.5 log-odds at the largest edit), while matched rotations that keep the split have almost no effect. No single internal object carries this response for the uneven direction, and fixing the mass-like objects does not remove it. The network's internal mass read-out does treat these jets as about 12 % lighter. The extracted float program reproduces most of the call changes. Which combination of objects carries the response remains open.
