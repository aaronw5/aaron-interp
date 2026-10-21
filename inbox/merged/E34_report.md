# E34: the gluon / light-quark boundary inside JEDI-linear: count, girth, $p_T^D$ or mass (new claim A17, exploratory; A3, A15; stage 3)

Script `interp/e34_quark_gluon_jedi.py`; figure `interp/e34_plot.py` -> `results/jedi/e34_quark_gluon.png`. Results `results/jedi/e34_quark_gluon.json`, per-jet arrays `results/jedi/e34_quark_gluon_perjet_{main,rep}.npz`. Logs `logs/e34-smoke.log` (smoke, 300 jets per class), `logs/e34-run.log` (first full run), `logs/e34-run2.log` (final run, after one bug fix, see Issues 1). Final job e34-run2: 1 CPU / 2 GiB, wall 267 s, peak memory 1,186 MiB, 0.98 cores.

## Conclusion
JEDI-linear's gluon-versus-quark score responds to all four jet properties tested. Adding particles (by splitting soft ones), raising the jet mass, raising girth at fixed second radial moment, and concentrating more pT in the hardest particle each move the score, and each response exceeds a matched control edit that moves the same particles without changing any of the four properties. Per 5 % change, adding particles moves the score most, and its effect is about half of what a small classifier trained on these four properties gives (ratio 0.54). Raising girth while the second radial moment is fixed moves the score toward quark. The classifier on all four properties does the same, although girth on its own favours gluon. The responses pass mainly through one second-round object, S1 (a pT-independent sum of how far each non-leading particle lies beyond r = 0.2), and not through the soft-multiplicity object S19 or the hardest-14 pT sum S0 named in the plan. The holds were not significant after correction, so the route remains uncertain.

## Definitions
- **Candidates** (computed on the 64-slot input the network sees; $r_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$, distance of particle $i$ to the jet axis):
  - $N$ = number of real particles.
  - **girth** $g = \sum_i p_{T,i} r_i / \sum_i p_{T,i} = M_1/M_0$: the pT-weighted mean distance to the axis.
  - **$p_T^D$** $= \sqrt{\sum_i p_{T,i}^2}/\sum_i p_{T,i}$: near 1 when one particle carries most of the pT, small when pT is shared among many. Quark jets have larger $p_T^D$.
  - **mass** $m$: the invariant mass of the massless constituents (exact four-vectors with $\eta=\Delta\eta$, $\phi=\Delta\phi$).
  - $M_\kappa = \sum_i p_{T,i} r_i^\kappa$ ($M_0$ = summed pT, $M_1$, $M_2$ = second radial moment, $m^2 \approx M_0 M_2$ for a centred jet).
- **Margin** = $\text{logit}_g - \text{logit}_q$ of the 5-class output, i.e. $\log(p_g/p_q)$. Positive changes = toward gluon. Trained-network margin SD over the sample: 2.60.
- **Edit − null**: mean over jets of (margin change under the edit) − (margin change under its matched null). The paired sign-flip permutation p tests H0: mean = 0.
- **Slope per %** = Σ (edit − null margin change) / Σ (100 × relative change of the target candidate). **Per sd** = the same per one standard deviation of the candidate over the g/q sample (a linear extrapolation from a 5 % edit).
- **MLP_x**: a 2 × 32 ReLU MLP (sklearn) trained to separate g from q using only candidate x (log N, girth, $p_T^D$, log m). **MLP_all** uses all four. They were trained on 20,000 g/q jets from the same test file that neither sample draws; no jet overlaps. Each MLP's output is converted to the same log-odds scale as the margin. **Ratio** = (JEDI edit − null) / (MLP edit − null) on the same jets. 1 means the network moves as much as that reference classifier.
- **Share removed** by a hold = $1 - \sum_j \Delta\text{margin}_j(\text{edit, element held at unedited value}) / \sum_j \Delta\text{margin}_j(\text{edit})$. 0 means the element carries none of the response. 1 means it carries all of it. Above 1, holding the element reverses the response, so the remaining elements together push the other way. Below 0, the element works against the response.
- **Objects** (E5, E5v, E28w):
  - **S1** = $\sum_{\text{ranks}\ge1}(r-0.2)_+$, the summed distance beyond r = 0.2 of every particle except the hardest; pT-independent.
  - **S30** = $\sum \text{ReLU}(3.47-21\,r)$ over all slots, the core-r sum, which gives 3.38 per empty slot.
  - **S19** = soft multiplicity: $\text{ReLU}(2.86-1.39\,p_T)$ over ranks 13–63, 2.86 per empty slot.
  - **S0** = summed pT of the 14 hardest particles.
  - **S29** ≈ $\sum p_T r^2 + \sum r^2$ + soft count.
  - **S43** = $\text{ReLU}(0.105\,p_T + 74.3\,r - 11.3)$ over the 14 hardest particles.
  - **S56** = a (pT, r) hinge with a slot window.
  - **g** = the 64 jet-context units.
  - **a32** = l1 soft-pT cut $\text{ReLU}(3.86-0.352\,p_T)$, on below 11 GeV.
- **Collinear split**: one particle replaced by two of half its pT at the same $(\Delta\eta,\Delta\phi)$; $M_0$, $M_1$, $M_2$, girth and mass stay exactly the same. **IRC-safe** = infrared- and collinear-safe: unchanged by such splits and by adding zero-pT particles. $N$ and $p_T^D$ are not IRC-safe; girth and mass are.
- **McNemar**: exact binomial test on jets whose call changes under one edit but not the other. **Holm**: step-down family-wise correction.

## Procedure
- Network: official JEDI-linear 3-feature n64, exact fixed-point forward (`jedi_core`, via `e28w_split_route_jedi.Net`). Check: `sample(5000, 3401)` (25,000 jets) 5-class accuracy **80.89 %** (band 80.6–81.3 %). The S → logits tail reproduces every call (1.00000, max |Δlogit| 0).
- Samples: **main** = the 10,000 g and q jets (5,000 each) of `sample(5000, 3401)`. **rep** = the 10,000 g/q jets of `sample(5000, 3402)`. JEDI g-vs-q AUC of the margin 0.921 (main) / 0.922 (rep). MLP AUCs on main: MLP_all 0.888, MLP_N 0.850, MLP_ptD 0.802, MLP_m 0.760, MLP_girth 0.726. Means g / q: N 56.7 / 37.4, girth 0.057 / 0.034, $p_T^D$ 0.274 / 0.396, m 87.9 / 53.9 GeV.
- **Edits.** Each moves one candidate by +5 % and keeps the other three within 1 %. Jets where the construction fails, or where a held candidate moves by more than 1 %, are left out; the fractions are listed below.
  - **Ecount.** The $k=\max(1,\text{round}(0.05N))$ softest particles (median k = 2) are each split collinearly into two halves; the jet is then re-sorted. N rises by k (mean +5.04 %); girth, mass, $M_0$ and $M_2$ are exactly unchanged; $p_T^D$ changes by at most 0.03 %. Applicable only when $N+k\le 64$: 67.5 % of jets (g 2,262, q 4,487), because many gluon jets already fill the 64 slots.
  - **Egirth.** A radial power map $r\to c\,r^{\gamma}$ is applied to every particle, keeping azimuth and pT. $\gamma<1$ is solved per jet for girth × 1.05 (median γ 0.950), and $c$ is set so that $M_2$ is unchanged. This pulls far particles in and pushes core particles out. N, pT and $p_T^D$ are exact; mass changes −0.06 % on average (max 1.0 %). Applicable 97.5 % × 99.2 % = 9,676 jets.
  - **EptD.** pT δ moves from the second-hardest to the hardest particle at fixed positions, with δ solved for $p_T^D$ × 1.05 (median δ 42 GeV); N and $M_0$ are unchanged. The resulting change of $M_1$, δ(r₁ − r₂), is cancelled by a radial move of the one particle with the largest pT·r among ranks ≥ 2 (median rank 5), so girth is exact. 73.6 % of applicable jets keep mass within 1 %; n = 7,343. **EptDm** is the reverse transfer ($p_T^D$ × 0.95), feasible on 24.6 % × 66 % of jets (n 1,628), and serves as a sign check.
  - **Emass.** The same power map with $\gamma>1$ (median 1.052) and $c$ fixing $M_1$ (girth), solved for exact mass × 1.05. N, pT, girth and $p_T^D$ are exact; n = 9,955.
  - **Edil** (reference, outside the Holm family): all angles × s, solved for mass × 1.05; girth also rises 5 %. This is the dilation used in the earlier W/Z/top nodes.
- **Nulls** (same particles moved; none of the four candidates changed):
  - Egirth, Emass and Edil: each moved particle is rotated about the axis by ±|Δr_i|/r_i (random sign). This is the same displacement length along the circle instead of along the radius, and it keeps N, pT, girth and $M_2$ exact.
  - Ecount: the k softest particles are rotated about the axis by uniform random azimuths. N is unchanged; mass moves by ≤ 13 % on single jets (mean −0.01 %).
  - EptD/EptDm: the compensating particle is rotated by its arc-matched angle. The pT transfer itself has no candidate-neutral analogue at fixed positions, because only a swap of two pTs keeps $\sum p_T^2$.
  - Untrained network (`Jedi(untrained=True, seed 0)`) on the same edits and nulls. Its margin SD is 0.025, so its changes are given in units of its own margin SD.
- **Route holds**: the edited jet is run with one element held at its unedited value. The elements are S19, S30, S1, S0 (each alone), S19+S30, S19+S30+S1+S0, all of g, and a32 per slot. The null for each hold is the other live-S sets of the same size: all 19 singles, all 171 pairs, and 200 random 4-sets. Rank p = fraction of the null with share ≥ the named set's share; for singles the smallest attainable value is 1/19 = 0.053. The random-set scans ran on main; the named holds ran on main and rep.
- Statistics: 1,000 multinomial bootstrap resamples over jets for every mean, share and ratio; paired sign-flip permutation (10,000) for edit − null; McNemar for paired call changes (edit vs null); Holm over the four primary edits.

## Results (main; rep in brackets where given)

### 1. Margin response, edit vs null (H0: edit − null = 0)
| edit | n | Δmargin edit [95 % CI] | Δmargin null [CI] | edit − null [CI] | perm p (Holm 4) | slope per % [CI] | per sd [CI] | rep edit − null [CI] |
|---|---|---|---|---|---|---|---|---|
| Ecount (N +5 %) | 6,749 | +0.137 [0.131, 0.142] | +0.025 [0.019, 0.030] | **+0.112 [0.105, 0.120]** | 1e-4 (4e-4) | +0.0223 [0.0208, 0.0238] | +0.90 [0.84, 0.96] | +0.108 [0.100, 0.115] |
| Egirth (girth +5 %, M2 fixed) | 9,676 | −0.089 [−0.094, −0.085] | 0.000 [−0.002, 0.002] | **−0.089 [−0.094, −0.085]** | 1e-4 (4e-4) | −0.0179 [−0.0187, −0.0170] | −1.65 [−1.73, −1.58] | −0.086 [−0.090, −0.082] |
| EptD ($p_T^D$ +5 %) | 7,343 | −0.052 [−0.054, −0.049] | −0.000 [−0.001, 0.001] | **−0.052 [−0.054, −0.049]** | 1e-4 (4e-4) | −0.0103 [−0.0108, −0.0098] | −0.40 [−0.42, −0.38] | −0.052 [−0.055, −0.050] |
| Emass (m +5 %, girth fixed) | 9,955 | +0.084 [0.080, 0.087] | +0.001 [−0.002, 0.003] | **+0.083 [0.079, 0.087]** | 1e-4 (4e-4) | +0.0165 [0.0157, 0.0174] | +1.03 [0.98, 1.08] | +0.083 [0.079, 0.087] |
| EptDm ($p_T^D$ −5 %) | 1,628 | +0.030 | −0.001 | +0.031 [0.027, 0.035] | 1e-4 | −0.0062 [−0.0071, −0.0053] | −0.17 [−0.20, −0.15] | +0.030 |
| Edil (m and girth +5 %) | 10,000 | +0.028 | +0.001 | +0.027 [0.024, 0.030] | 1e-4 | +0.0054 (per % mass) | +0.33 | +0.030 [0.026, 0.033] |

By class (edit, main): Ecount g +0.145 / q +0.133; Egirth g −0.119 / q −0.061; EptD g −0.060 / q −0.042; Emass g +0.097 / q +0.070. H0 is rejected for all four primary edits. The sign of each response is the same in both classes and in rep.

### 2. Calls (g/q jets; McNemar edit vs null, Holm over 4)
| edit | calls changed edit [CI] | null [CI] | McNemar p (Holm) | edit flows g→q / q→g (sign p) | null g→q / q→g |
|---|---|---|---|---|---|
| Ecount | 3.87 % [3.43, 4.32] | 2.64 % [2.26, 3.06] | 7e-6 (7e-6) | 12 / 139 (1e-28) | 55 / 52 |
| Egirth | 3.87 % [3.50, 4.23] | 1.99 % [1.72, 2.29] | 2e-21 (4e-21) | 92 / 21 (9e-12) | 41 / 36 |
| EptD | 1.99 % [1.69, 2.30] | 0.48 % [0.33, 0.65] | 3e-22 (9e-22) | 50 / 11 (5e-7) | 6 / 2 |
| Emass | 5.57 % [5.13, 6.02] | 2.50 % [2.21, 2.81] | 1e-39 (4e-39) | 27 / 90 (4e-9) | 39 / 43 |

The remaining changed calls go to or from W, Z and top. Under Emass, 216 calls move from g/q to W/Z/t and 144 move back; under Egirth, 155 and 81.

### 3. Calibration against classifiers that use only these variables (ratio = JEDI / MLP, edit − null, 95 % CI)
| edit | MLP of that variable: Δ log-odds | JEDI / MLP_x | MLP_all: Δ log-odds | JEDI / MLP_all | rep JEDI / MLP_all |
|---|---|---|---|---|---|
| Ecount | MLP_N +0.207 | **0.54 [0.51, 0.58]** | +0.208 | 0.54 [0.51, 0.58] | 0.52 [0.48, 0.55] |
| Egirth | MLP_girth **+0.044** (toward gluon) | −2.01 [−2.12, −1.91] | **−0.114** (toward quark) | 0.78 [0.75, 0.82] | 0.74 [0.71, 0.78] |
| EptD | MLP_ptD −0.178 | 0.29 [0.28, 0.30] | −0.091 | 0.57 [0.54, 0.60] | 0.57 [0.54, 0.60] |
| Emass | MLP_m +0.081 | **1.02 [0.97, 1.07]** | +0.144 | 0.58 [0.55, 0.60] | 0.57 [0.55, 0.60] |
| EptDm | MLP_ptD +0.178 | 0.17 [0.15, 0.20] | +0.071 | 0.44 [0.38, 0.50] | 0.40 [0.35, 0.46] |
| Edil | MLP_m +0.081, MLP_girth +0.043 | 0.33 (m), 0.62 (girth) | +0.027 | 1.00 [0.89, 1.13] | 1.09 [0.98, 1.22] |

A single-variable MLP gives the marginal effect of its variable, with the other three free to co-vary as they do in data. MLP_all gives the effect with the other three held fixed, which is what the edits do. For three of the four edits, JEDI's response is 0.54–0.58 of MLP_all's; for Egirth it is 0.78. The edits with the other three properties held are therefore the comparison that matches MLP_all.

### 4. Untrained network (edit − null, in units of each network's margin SD; main)
| edit | untrained [CI] | trained [CI] |
|---|---|---|
| Ecount | −0.008 [−0.023, 0.007] | +0.043 [0.040, 0.046] |
| Egirth | +0.007 [−0.004, 0.018] | −0.034 [−0.036, −0.033] |
| EptD | −0.003 [−0.009, 0.003] | −0.020 [−0.021, −0.019] |
| Emass | +0.021 [0.007, 0.035] | +0.032 [0.030, 0.033] |
| Edil | +0.023 [0.011, 0.035] | +0.010 [0.009, 0.012] |

The untrained network shows no count, girth or $p_T^D$ response. It does respond to the radial mass edits: its Emass response is 0.66 of the trained network's in SD units, and its Edil response is larger than the trained one.

### 5. Route holds (share of the edit's margin change removed; main, 95 % CI; rank p among size-matched live-S sets)
| hold | Ecount | Egirth | EptD | Emass | Edil |
|---|---|---|---|---|---|
| S19 (soft multiplicity) | 0.11 [0.10, 0.12] (p 0.21) | 0.00 | 0.00 | 0.00 | 0.00 |
| S30 (core r sum) | 0.89 [0.86, 0.92] (p 0.11) | −0.01 [−0.02, 0.01] | 0.00 | 0.48 [0.45, 0.50] (p 0.11) | 1.59 |
| S1 (Σ over ranks ≥ 1 of (r−0.2)₊) | **1.40 [1.33, 1.48]** (p 0.053, top of 19) | **1.19 [1.14, 1.23]** (p 0.053, top) | −0.01 | **4.35 [4.17, 4.55]** (p 0.053, top) | 5.01 (top) |
| S0 (pT of 14 hardest) | 0.015 [0.010, 0.020] | 0.00 | −0.00 [−0.02, 0.02] | 0.00 | 0.00 |
| S19+S30 | 0.99 (p 0.12; pair q95 1.40) | −0.01 (p 0.65) | 0.00 (p 0.56) | 0.48 (p 0.17) | 1.59 (p 0.16) |
| S19+S30+S1+S0 | 2.30 (p 0.015; 4-set q95 1.75) | 1.18 (p 0.11) | −0.00 (p 0.80) | 4.75 (p 0.030) | 6.51 (p 0.010) |
| g (all context units) | 0.31 [0.28, 0.33] | −0.27 [−0.31, −0.24] | −0.02 [−0.04, 0.01] | −0.09 [−0.12, −0.06] | 0.71 |
| a32 (soft cut, per slot) | 0.17 [0.16, 0.18] | 0 (pT unchanged) | 0.00 | 0 | 0 |
| largest other singles | S56 −0.40, S62 +0.32, S31 −0.19, S29 −0.12 | S56 −1.27, S60 +0.36, S31 −0.34, S29 −0.23 | **S43 +0.91** (top of 19), S51 +0.09 | S29 −0.69, S56 −0.56, S31 −0.46, S62 +0.26 | S29 −1.51, S56 −0.92 |

Holm over the four primary edits for the pre-named 4-set: 0.06 (Ecount), 0.09 (Emass), 0.22 (Egirth), 0.80 (EptD). No hold passes after correction. For a single unit the smallest attainable rank p is 0.053, so none can pass at 0.05. Rep reproduces the named shares: Ecount S19 0.12, S30 0.91, S1 1.45, g 0.31, a32 0.17; Egirth S1 1.20, g −0.30; Emass S1 4.29, S30 0.45; EptD all named sets within ±0.015.

### 6. Example (figure panel a)
Gluon jet (main g/q index 13, N = 39, girth 0.010, $p_T^D$ 0.291, m 37.0 GeV, margin +2.31, called g). Ecount adds 2 particles (margin +2.77) and Emass raises m to 38.9 GeV (margin +2.71). Egirth (girth 0.0106) and EptD ($p_T^D$ 0.306) leave this jet's margin at +2.31.

## Reading (scoped to these samples and edits)
- All four properties move JEDI-linear's g−q margin beyond their matched nulls on both samples (Holm p 4e-4 for each). Directions: more particles → gluon; more mass at fixed girth → gluon; higher $p_T^D$ → quark; higher girth at fixed $M_2$ → quark. Per +5 %, the responses are 0.112 (count), 0.089 (girth), 0.083 (mass) and 0.052 ($p_T^D$) in log-odds. Per sd of the property (linear extrapolation), they are girth 1.65, mass 1.03, count 0.90 and $p_T^D$ 0.40.
- Against a classifier that uses only these four properties (MLP_all), the network moves 0.54–0.58 as much for count, $p_T^D$ and mass, and 0.78 as much for girth. The count response is 0.54 of MLP_N's. This matches the plan's expectation that the count response sits well below a count-only classifier.
- Girth at fixed $M_2$: on its own (MLP_girth), higher girth means gluon. With the other three properties held (MLP_all), it means quark, and the network agrees with MLP_all in sign. Raising $M_1$ at fixed $M_2$ narrows the spread of the radial profile (more uniform r). The measurement shows the network reads this as quark-like; it does not show that the network computes girth itself.
- Competing hypotheses from the plan:
  - "Count response removed by the S19 hold": excluded. S19 removes 0.11 [0.10, 0.12], below S30 (0.89) and S1 (1.40).
  - "Girth removed by S30/S29 holds": excluded. S30 −0.01 and S29 −0.23 work against the response; S1 carries 1.19.
  - "$p_T^D$ removed by the S0 hold": excluded. S0 −0.00 [−0.02, 0.02]. The largest single is S43 (0.91, a hinge over the 14 hardest particles that rises with pT and r), top of 19 but at the attainable floor p 0.053.
  - "Mass dominates": not observed. Mass is not the largest response per 5 % or per sd.
- S1, the pT-independent sum of distance beyond r = 0.2 over the non-leading particles, is the top single hold for count, girth and mass. Its share exceeds 1 in each case, so with S1 held the other objects (S29, S56, S31) push the margin the other way. For count, S1 rises because a split wide-angle particle adds a second term. For count, holding g removes 0.31 and holding a32 removes 0.17.
- The untrained network responds to the radial mass edits (0.66 of the trained response in SD units) but not to count, girth or $p_T^D$. Part of the mass response may therefore come from the architecture's radial features rather than training.

## Issues
1. **Bug fixed (debug attempt 1).** The first full run (e34-run) took `Boot.ratio` from E28w for ratio CIs. That function clamps the denominator at 1 because it was written for counts. Every ratio CI with a negative or small denominator was invalid: EptDm slope, and the JEDI/MLP ratios for Egirth, EptD and EptDm. A local `bratio` without the clamp was added, together with per-jet saves, and the job was rerun as e34-run2. All point estimates were identical between the two runs.
2. Ecount applies only to jets with N + k ≤ 64. It covers 45 % of gluon jets vs 90 % of quark jets, because the 64-slot input cannot show a count above 64.
3. The EptD null rotates only the compensating particle. The pT transfer has no candidate-neutral analogue at fixed positions, so the EptD comparison rests on the near-zero null (0.48 % calls changed), the sign-reversed EptDm (+0.031), the MLP calibration and the untrained network.
4. The Egirth construction differs from the plan. The plan described a rescale of particles beyond r > 0.1 with a core compensation; the run used one per-jet power map $r\to c r^\gamma$, which holds $M_2$ exactly and moves every particle. Egirth and Emass are then mirror images (γ < 1 at fixed $M_2$ vs γ > 1 at fixed $M_1$).
5. The Ecount null (random azimuth of the k softest particles) changes the mass of single jets by up to 13 %, mean −0.01 %. Its call-change rate (2.64 %) is high because those particles are often at wide angle.
6. The plan's "200 size-matched random holds" was used for the 4-set. Singles and pairs were enumerated (19, 171). The S19 and S30 holds were not compared with a separate S29 hold set; S29 appears among the singles.
7. Per-sd slopes extrapolate a 5 % edit linearly to one sd. Girth's sd (0.039) is large relative to its mean, so its per-sd value is the most extrapolated.
8. The import stage of the smoke job took 522 s on its node; the full runs took 13–35 s.

## Explainer (site/explain.py, Paper A, new claim A17)
**Question.** When JEDI-linear separates gluon jets from light-quark jets, which jet property moves its decision: the number of particles, girth, $p_T^D$ or the mass, and through which named objects?
**Background.** Gluon jets carry more particles, spread wider and share their pT more evenly than quark jets ($N$ 56.7 vs 37.4, girth 0.057 vs 0.034, $p_T^D$ 0.27 vs 0.40, $m$ 88 vs 54 GeV in this sample). These four properties are correlated in data, so a correlation with the network's score cannot show which one the network uses.
**Method.** On 10,000 g/q test jets, change one property by 5 % and keep the other three within 1 %:
- count: split the $k\approx0.05N$ softest particles into collinear halves;
- girth: $r\to c\,r^{\gamma}$ with $\sum p_T r^2$ fixed;
- $p_T^D$: move pT from the second to the hardest particle, with a one-particle radial move to keep girth;
- mass: $r\to c\,r^{\gamma}$ with $\sum p_T r$ fixed.

Each edit is compared with a null that moves the same particles by the same amount around the axis. The response $\Delta(\text{logit}_g-\text{logit}_q)$ is compared with MLPs trained on each property alone and on all four. Second-round objects are then held at their unedited values to see which carry the response.
**Result.** (figure `e34_quark_gluon.png`) All four edits move the score beyond their nulls: count +0.112, girth −0.089, mass +0.083, $p_T^D$ −0.052 log-odds per 5 % (Holm p 4e-4 each). The network moves 0.54–0.78 as much as the four-variable MLP. Raising girth at fixed $\sum p_T r^2$ moves both the network and the four-variable MLP toward quark. Holding $S_1=\sum_{\text{rank}\ge1}(r-0.2)_+$ removes more than the whole count, girth and mass responses. S19 and S0 carry little.
**Checks.** Accuracy 80.89 %. A second sample (seed 3402) reproduces every response within its CI. Nulls change the score by ≤ 0.025. An untrained network shows no count, girth or $p_T^D$ response, but does respond to the mass edit.
**Why it matters.** The network's gluon/quark decision is not a count tagger or a mass tagger alone: it uses all four properties at about half the strength of a classifier built on them. The common carrier is a count of wide-angle particles (S1), not the soft-multiplicity object.

Glossary additions:
- **girth**: the pT-weighted mean distance of the particles from the jet axis.
- **$p_T^D$**: $\sqrt{\sum p_T^2}/\sum p_T$, which measures how much of the pT is concentrated in few particles.
- **collinear split**: one particle replaced by two at the same angle that share its pT.
- **IRC-safe**: unchanged by collinear splits and by adding zero-pT particles.
- **g−q margin**: $\text{logit}_g-\text{logit}_q=\log(p_g/p_q)$.
- **MLP_x / MLP_all**: reference classifiers on one candidate or on all four.
- **radial power map**: $r\to c\,r^\gamma$, with $c$ fixing one radial moment.
- **arc-matched rotation null**: each particle moved by the same distance along the circle around the axis instead of along the radius.
- **share removed > 1**: holding the element reverses the response.

Conclusion (table row): JEDI-linear's gluon-versus-quark score responds to particle count, girth, $p_T^D$ and mass, each beyond a matched control, at about half to three quarters of the response of a small classifier trained on these four properties. The responses pass mainly through one object that sums how far the non-leading particles lie beyond r = 0.2, not through the soft-multiplicity or hardest-pT objects. The hold results are not significant after correction, and the untrained network also responds to the mass edit.
