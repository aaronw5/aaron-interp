# E40: quark vs gluon inside JEDI-linear — does the call count particles, and through which objects? (claims A1, A3; new exploratory claim A19qg; stage 3)

Script `interp/e40_qg_count_jedi.py`; figure `interp/e40_plot.py` -> `results/jedi/e40_qg_count.png`. Results `results/jedi/e40_qg_count.json`, per-jet margin changes and calls `results/jedi/e40_qg_count_perjet.npz`; smoke `results/jedi/e40_qg_count_smoke.json`. Logs `logs/e40-smoke.log` (150 jets per class), `logs/e40-run.log`. Diagnostic `interp/e40_check.py` (job e40-chk; see Issues 2). Job e40-run: 1 CPU / 3 GiB, wall 560 s (about 370 s of it at import), peak memory 2,993 MiB (97 % of request), 0.31 cores on average.

## Conclusion
JEDI-linear moves its gluon-versus-quark score toward "gluon" when the jet gets more particles, even when the total jet momentum and mass are held exactly fixed. This holds when a particle is split into two at the same point, and when soft particles are added. Adding one 0.5 GeV particle moves the score about 1.3 times as much as one extra particle does on average across real jets. The response barely shrinks when the added particles carry 10 times less momentum. The network is therefore not infrared- and collinear-safe (IRC-safe) in its quark/gluon decision: it counts particles. The count is carried by several objects, and which one depends on where the particle is added. For splits that push particles across the rank-13/14 window edge, the soft-multiplicity object S19 carries nearly all of it. For splits of soft, wide-angle particles and for added soft particles, it is spread over S1 (a pT-independent sum over particles beyond r = 0.2), S30 (an empty-slot count) and the context units g. None of these single-object results passes multiple-comparison correction against all other objects.

## Definitions
- **Margin** $= \text{logit}_g - \text{logit}_q = \log p(g) - \log p(q)$ of the 5-class output; positive changes point toward gluon. Over the unedited q/g test jets the SD is 2.63.
- **IRC safety** (infrared and collinear safety): an observable is IRC-safe if it does not change when (i) a particle is split into two particles at the same direction (collinear split), or (ii) a particle of vanishing momentum is added (soft emission). The jet mass and $M_\kappa = \sum_i p_{T,i} r_i^\kappa$ are IRC-safe. The particle count $N$ is not.
- **Collinear split (C1)**: the particle at pT rank $k$ (0 = hardest) with $(p_T, \Delta\eta, \Delta\phi)$ is replaced by $(f\,p_T, \Delta\eta, \Delta\phi)$ and $((1-f)\,p_T, \Delta\eta, \Delta\phi)$, $f \in \{0.5, 0.2\}$, and the jet is re-sorted by pT. The split particle is the hardest (r0), the particle at rank 13 (r13), or one particle at a random rank $\ge 30$ (soft). $N$ rises by 1. The summed massless four-momentum is exact: maximum relative change $3.4\times10^{-8}$ (float32 rounding at f = 0.2), jet-mass change $\le 2.6\times10^{-6}$ GeV. Only jets with $N \le 63$ are edited, so no particle leaves the 64 input slots. r0 needs $N \ge 2$, r13 needs $N \ge 15$, soft needs $N \ge 31$.
- **Soft emission (C2)**: $k$ = 1, 3 or 5 particles of 0.5 GeV are added at $r = 0.3$ from the axis, each at a uniform random azimuth. All other particles are then scaled by one common factor and given one common small Lorentz boost, so that the summed jet four-momentum equals the unedited one exactly. The particles stay massless; maximum relative four-momentum change $7.8\times10^{-12}$; the median per-jet largest pT change of an existing particle is 0.09-0.34 %. **emit_k3_pt0.05** is the same edit with 0.05 GeV particles, the infrared limit. Only jets with $N+k \le 64$ are edited.
- **Merge (C3)**: the two softest particles are replaced by one particle that carries their summed pT and sits at their pT-weighted $(\Delta\eta, \Delta\phi)$. $N$ falls by 1. The jet mass falls by 0.24 GeV on average (max 7.0 GeV), because the two softest particles are usually not collinear.
- **Controls**:
  - (a) **nc** (non-collinear split): the same split with the same $f$, but the two halves are 0.05 apart in $(\Delta\eta,\Delta\phi)$ along a random direction, with their pT-weighted position at the parent's position. The pair's three-momentum is fixed to first order; the pair gains a small mass (jet mass +1.5 GeV on average at r0 f 0.5, +0.007 GeV at soft). This control separates "one more particle" from "the angular pattern changes".
  - (b) **rankpres** (rank-preserving split): computed exactly inside the network from the per-slot terms of the pooled sums $P = s_0\sum_i q_i(a_i)$ and $S = s_1\sum_i q_i(u_i)$ (e28w `Net.rank_patch`, mode PS). Every particle other than the two halves contributes the term it had in its unsplit slot, so no particle crosses a rank-window edge. Only the halves and the one filled empty slot change. On the split jets, the reconstruction from per-slot terms reproduces S exactly (max |ΔS| = 0). This control separates "window crossing" from "count".
  - **rot**: the same particle rotated about the jet axis by a random azimuth; $N$ and all ranks stay fixed.
  - **comp_k5**: the scale-and-boost compensation of emit_k5 without adding the particles.
  - **move2**: the two softest particles moved to their pT-weighted position without merging. $N$ stays fixed, and the angular change is the same as merge2's (same jet mass change).
  - (c) **untrained**: JEDI-linear with redrawn weights (`Jedi(untrained=True)`, seeds 0-4), margin change in units of that network's margin SD.
  - (d) **null_resort**: particles randomly permuted, then re-sorted by pT.
- **Natural slope**: OLS slope of the margin against $N$ over the unedited q/g jets of the fit sample. **Pooled**: g and q jets together, the slope seen across real jets. **Within class**: with a class offset.
- **$R_N$** = (mean margin change of the edit) / (natural slope × ΔN). In words: the network's response to the edit divided by the response expected from the way its margin rises with N across real jets, for the same change in N. $R_N = 1$: one particle added by the edit moves the margin as much as one more particle does on average in data. $R_N$ is quoted with the pooled slope unless stated otherwise.
- **Holds**: the edited jet is run with one second-round object (S unit) set to its value on the unedited jet. **Share removed** $= 1 - \sum_j \Delta m_j(\text{edit + hold}) / \sum_j \Delta m_j(\text{edit})$ over q/g jets. 1 = the object carries all of the response; > 1 = with it held, the rest pushes the other way; < 0 = it works against the response.
  - Named holds: **S19** (soft multiplicity, $\sum_{\text{ranks }13\text{-}63}\mathrm{ReLU}(2.86-1.39\,p_T)$, which also gives 2.86 per empty slot), **S0** (pT of the 14 hardest particles, via the context g), **S29** ($\approx \sum p_T r^2 + \sum r^2$ + soft count), and **S19+S0**.
  - Also reported: **S1** ($\sum_{\text{ranks}\ge1}(r-0.2)_+$, pT-independent), **S30** ($\sum \mathrm{ReLU}(3.47-21r)$ over all slots, 3.38 per empty slot), **S43** ($\mathrm{ReLU}(0.105\,p_T + 74.3\,r - 11.3)$ over the 14 hardest), and **g** (all 64 context units held).
  - Null: every other live S set of the same size (18 singles, 170 pairs; enumerated). **Rank p** = (1 + #null sets with share ≥ named) / (1 + #null). For a single it cannot go below 1/19 = 0.053.
- **McNemar**: exact binomial test on jets whose call changes under one edit but not the other. **Holm**: step-down family-wise correction, here over 20 tests (4 named holds × 5 primary edits).

## Procedure
- Network: official JEDI-linear 3-feature n64, exact fixed-point forward (`jedi_core` through e28w `Net`). The S → logits tail reproduces every call (1.00000, max |Δlogit| 0).
- **Fit sample**: `sample(2000, 4001)`, 10,000 jets, 5-class accuracy **81.06 %**. Its 4,000 q/g jets give the natural slope.
- **Test sample**: `sample(2000, 4002)`, 10,000 jets, accuracy **80.38 %** (see Issues 1). All edits ran on all 10,000 test jets. The primary rows are the 4,000 q/g jets; W/Z/t are side rows. Mean N: g 56.7, q 37.4, W 38.7, Z 39.5, t 56.6. Fraction with N = 64: g 51 %, q 9 %. The N ≤ 63 requirement therefore removes about half of the gluon jets from the split rows.
- Statistics: 1,000 multinomial bootstrap resamples over jets for every mean, share and $R_N$. The $R_N$ CI combines independent bootstraps of the test edit and the fit slope. Paired sign-flip permutation tests (5,000) for edit vs control; exact McNemar on call changes and on q→g flips; Holm over the 20 named-hold tests. Bootstrap p-values are bounded below by 0.001 (1,000 resamples).
- Hypotheses:
  - H0: collinear response = non-collinear control = untrained.
  - H1 (counting): the split moves the margin toward g; collinear = non-collinear; the S19 hold removes most of the response for soft splits, and the S0 hold for rank-13 splits.
  - H1′ (angular / IRC-safe): the collinear response ≈ 0, and only the non-collinear control moves the call.
  - H1″ (rank only): a response appears only at the 13|14 crossing.

## Results

### 1. Natural slope (fit sample, 4,000 q/g jets)
The margin rises with N: **0.137 per particle [0.134, 0.139] pooled**, and 0.098 [0.095, 0.101] within class. Bin means run from −3.76 (N < 20) to +2.52 (N = 64), figure panel c.

### 2. Responses (q/g test jets; mean Δmargin, 95 % CI; perm p vs 0 ≤ 2e-4 for every row except null_resort and rot_r0)
| edit | n | Δmargin [CI] | $R_N$ pooled [CI] | $R_N$ within | calls changed | q→g / g→q |
|---|---|---|---|---|---|---|
| null_resort (d) | 4,000 | −0.0001 [−0.0006, 0.0004] | – | – | 0.10 % | 1 / 0 |
| split r0 f 0.5 | 2,794 | **+0.282 [0.273, 0.293]** | 2.07 [1.99, 2.15] | 2.87 | 5.65 % | 98 / 4 |
| nc r0 f 0.5 | 2,794 | +0.188 [0.178, 0.198] | 1.38 | 1.91 | 5.08 % | 71 / 15 |
| rankpres r0 f 0.5 | 2,794 | +0.458 [0.447, 0.468] | 3.35 | 4.65 | 7.73 % | 164 / 0 |
| rot r0 | 2,794 | −0.003 [−0.005, −0.000] | – | – | 0.79 % | 1 / 1 |
| split r0 f 0.2 | 2,794 | +0.210 [0.201, 0.219] | 1.54 | 2.13 | 4.33 % | 67 / 5 |
| split r13 f 0.5 | 2,766 | **+0.079 [0.072, 0.087]** | 0.58 [0.52, 0.64] | 0.81 | 3.25 % | 38 / 8 |
| nc r13 f 0.5 | 2,766 | +0.085 [0.078, 0.093] | 0.62 | 0.86 | 3.25 % | 40 / 8 |
| rankpres r13 f 0.5 | 2,766 | **−0.008 [−0.014, −0.002]** | −0.05 | −0.07 | 2.24 % | 11 / 19 |
| rot r13 | 2,766 | +0.009 [0.006, 0.013] | – | – | 1.12 % | 6 / 7 |
| split r13 f 0.2 | 2,766 | +0.074 [0.067, 0.081] | 0.54 | 0.75 | 3.11 % | 31 / 8 |
| rankpres r13 f 0.2 | 2,766 | +0.018 [0.012, 0.024] | 0.13 | 0.18 | 1.95 % | 12 / 14 |
| split soft f 0.5 | 1,977 | **+0.062 [0.054, 0.069]** | 0.45 [0.39, 0.51] | 0.63 | 3.54 % | 30 / 9 |
| nc soft f 0.5 | 1,977 | +0.064 [0.056, 0.071] | 0.47 | 0.65 | 3.39 % | 31 / 7 |
| rankpres soft f 0.5 | 1,977 | +0.067 [0.060, 0.074] | 0.49 | 0.68 | 2.83 % | 27 / 3 |
| rot soft | 1,977 | +0.017 [0.009, 0.024] | – | – | 2.28 % | 19 / 4 |
| split soft f 0.2 | 1,977 | +0.056 [0.049, 0.063] | 0.41 | 0.57 | 2.93 % | 23 / 8 |
| emit k1 (0.5 GeV) | 2,794 | **+0.180 [0.172, 0.188]** | 1.32 [1.25, 1.38] | 1.83 | 3.65 % | 58 / 2 |
| emit k3 | 2,683 | +0.530 [0.517, 0.544] | 1.29 [1.25, 1.34] | 1.80 | 9.21 % | 168 / 0 |
| emit k5 | 2,563 | +0.860 [0.841, 0.878] | 1.26 [1.23, 1.30] | 1.75 | 14.98 % | 282 / 0 |
| emit k3, 0.05 GeV | 2,693 | **+0.385 [0.374, 0.396]** | 0.94 [0.91, 0.97] | 1.30 | 6.46 % | 103 / 0 |
| comp k5 (no particles added) | 2,563 | −0.059 [−0.065, −0.051] | – | – | 2.58 % | 10 / 20 |
| merge2 | 4,000 | **−0.203 [−0.213, −0.193]** | 1.49 [1.41, 1.56] | 2.07 | 5.12 % | 6 / 47 |
| move2 | 4,000 | −0.064 [−0.069, −0.060] | – | – | 2.05 % | 6 / 24 |

By class, the response has the same sign in g and q jets: split soft f 0.5 is g +0.063 / q +0.061; emit k3 g +0.47 / q +0.56. Side rows (W / Z / t Δmargin): emit k3 +0.86 / +0.79 / +0.28, with 4.6 / 5.1 / 4.3 % of calls changed; split r0 f 0.5 +0.17 / +0.18 / +0.13; merge2 −0.18 / −0.18 / −0.16. The emission response is linear in k: 0.180, 0.177 and 0.172 per added particle for k = 1, 3, 5.

### 3. Paired comparisons (q/g jets; diff = edit − control, 95 % CI, sign-flip p; McNemar on q→g flips)
| edit vs control | diff [CI] | perm p | q→g flips edit-only / control-only (McNemar p) |
|---|---|---|---|
| split r0 f 0.5 vs nc | +0.094 [0.089, 0.100] | 2e-4 | 29 / 2 (5e-7) |
| split r0 f 0.2 vs nc | +0.049 [0.044, 0.054] | 2e-4 | 23 / 9 (0.02) |
| split r13 f 0.5 vs nc | −0.006 [−0.008, −0.003] | 2e-4 | 2 / 4 (0.69) |
| split r13 f 0.2 vs nc | −0.011 [−0.013, −0.008] | 2e-4 | 0 / 6 (0.03) |
| **split soft f 0.5 vs nc** | **−0.002 [−0.006, 0.0015]** | 0.24 | 3 / 4 (1.0) |
| split soft f 0.2 vs nc | −0.003 [−0.007, 0.0001] | 0.065 | 2 / 2 (1.0) |
| split r0 / r13 / soft f 0.5 vs rot | +0.285 / +0.070 / +0.045 | 2e-4 each | 97/0, 35/3, 22/11 (1e-29, 7e-8, 0.08) |
| split r0 f 0.5 vs rankpres | −0.175 [−0.185, −0.166] | 2e-4 | 9 / 75 (4e-14) |
| **split r13 f 0.5 vs rankpres** | **+0.087 [0.079, 0.095]** | 2e-4 | 29 / 2 (5e-7) |
| split r13 f 0.2 vs rankpres | +0.056 [0.049, 0.063] | 2e-4 | 23 / 4 (3e-4) |
| split soft f 0.5 vs rankpres | −0.005 [−0.010, 0.0003] | 0.063 | 7 / 4 (0.55) |
| emit k5 vs comp k5 | +0.918 [0.899, 0.938] | 2e-4 | 272 / 0 (3e-82) |
| emit k3 0.5 GeV vs 0.05 GeV | +0.143 [0.137, 0.150] | 2e-4 | 70 / 5 (1e-15) |
| merge2 vs move2 | −0.139 [−0.148, −0.129] | 2e-4 | 5 / 5 (1.0); calls changed 150 / 27 (7e-22) |

### 4. Holds (share of the q/g margin change removed; 95 % CI; rank p among size-matched live S sets; Holm over 20)
| edit | S19 | S0 | S29 | S19+S0 (rank p; Holm) | g (all) | largest singles | single q95 |
|---|---|---|---|---|---|---|---|
| split r0 f 0.5 | 0.28 [0.26, 0.29] (p 0.11) | 0.03 | −0.01 | 0.31 (0.13; 1.0) | 0.15 | **S43 0.55**, S19 0.28, S60 0.14 | 0.30 |
| split r13 f 0.5 | **0.86 [0.80, 0.94]** (p 0.053, top of 19) | 0.10 [0.08, 0.12] | −0.00 | **0.97 [0.90, 1.05]** (0.018; 0.35) | 0.15 | S19 0.86, S1 0.33, S30 0.20 | 0.38 |
| split r13 f 0.2 | 0.88 [0.81, 0.95] (p 0.053) | 0.03 | −0.02 | 0.91 (0.029) | 0.11 | S19, S1 0.36, S30 0.21 | 0.41 |
| split soft f 0.5 | 0.27 [0.23, 0.32] (p 0.16) | 0.02 | **−0.14** | 0.29 (0.21; 1.0) | 0.20 | **S1 1.22**, S30 0.84, S19 0.27 | 0.87 |
| nc soft f 0.5 | 0.26 | 0.02 | −0.14 | 0.28 | 0.20 | S1 1.18, S30 0.82 | 0.86 |
| emit k3 | 0.16 [0.15, 0.17] (p 0.21) | 0.01 | −0.05 | 0.18 (0.30; 1.0) | **0.38** | **S1 0.53**, S30 0.43, S62 0.18 | 0.44 |
| merge2 | **0.46 [0.43, 0.49]** (p 0.053, top of 19) | 0.00 | −0.11 | 0.46 (0.041; 0.78) | 0.23 | S19 0.46, S30 0.31, S1 0.26 | 0.33 |

Against zero (bootstrap, Holm over the 20 tests), every named share differs from 0 at Holm p < 0.001 except S29 at r0 (−0.009, Holm 0.064), S29 at r13 (Holm 0.88), S0 at soft (Holm 0.20) and S0 at merge2 (Holm 0.75). Against the other objects, no named hold passes Holm: the smallest is S19+S0 at r13, rank p 0.018, Holm 0.35. "S19 removes > 0.5": met at r13 (bootstrap p(share ≤ 0.5) < 0.001) and not met at soft splits (0.27), emissions (0.16) or merges (0.46). The rank-13 holds and the rankpres control agree: rankpres removes 1.10 of the r13 f 0.5 response, and S19 alone removes 0.86.

### 5. Untrained controls (Δmargin in units of each network's own margin SD; q/g jets; 5 seeds)
The untrained margin SDs are 0.024, 0.0003, 0.011, 0.0001 and 0.017; seeds 1 and 3 give a nearly constant output. No untrained seed changed any call for any edit. Per-seed responses lie within ±0.16 SD. The trained responses, in its own SD (2.63), are: split r0 f 0.5 +0.107 (untrained −0.04 to +0.03), split r13 f 0.5 +0.030 (−0.003 to +0.008), split soft f 0.5 +0.023 (−0.004 to +0.008), emit k3 +0.20 (−0.07 to +0.12), emit k5 +0.33 (−0.09 to +0.16), merge2 −0.077 (−0.05 to +0.03). Each trained value lies outside the range of all 5 untrained seeds (with 5 seeds, the smallest attainable rank p is 1/6 = 0.17).

### 6. Example (figure panel a)
A real gluon jet (test index 33, N = 39, m = 35.5 GeV) that JEDI-linear calls quark (margin −0.46). Its rank-36 particle (0.48 GeV, at $\Delta\eta = 0.19$, $\Delta\phi = -0.20$) is split into two 0.24 GeV particles at the same point. The jet four-momentum is unchanged. The margin goes from −0.46 to −0.09 (non-collinear control: −0.28). The call stays quark.

## Reading (scoped to these samples and edits)
- **H1′ (IRC-safe in practice): excluded.** At fixed jet four-momentum, each split moves the margin toward gluon (+0.06 to +0.28, p ≤ 2e-4 against 0), well above the rotation control. Adding soft particles moves it +0.18 per particle. With 0.05 GeV particles the response is still 0.73 of the 0.5 GeV response (+0.385 vs +0.530 for k = 3), so it does not vanish in the infrared limit.
- **Count vs angle.** For soft splits, the collinear and non-collinear responses are equal (diff −0.002 [−0.006, 0.0015]). At rank 13 they differ by −0.006 to −0.011. At the hardest particle the collinear split moves the margin 0.094 more than the non-collinear one. There, the top single hold is S43, a hinge over the 14 hardest particles in pT and r. Merging the two softest particles moves the margin −0.203; moving them to the same point without merging gives −0.064, so −0.139 comes from the change of N at fixed angles.
- **Size relative to data ($R_N$).** One added 0.5 GeV particle gives $R_N$ = 1.32 [1.25, 1.38], larger than the average margin rise per particle across real q/g jets. A merge gives 1.49. Collinear splits give 0.45 (soft) to 0.58 (rank 13) and 2.07 (hardest). With the within-class slope every $R_N$ is about 1.39 times larger.
- **H1″ (rank only): excluded as the general explanation, but it describes the rank-13 split.** Soft splits (no particle crosses rank 13; rankpres equal to split, diff −0.005 [−0.010, 0.0003]), emissions and merges all respond. At rank 13, the rank-preserving reconstruction removes the response (−0.008 [−0.014, −0.002] vs +0.079). That response therefore comes from the softer particles moving down one slot across the 13|14 window edge. S19, whose window starts at rank 13, carries 0.86 of it. At rank 0 the rank-preserving reconstruction increases the response (+0.458): there, the shift of the softer particles works against the halves' own push toward gluon, as found in E28w.
- **Carriers.**
  - S19 carries most of the rank-13 split response (0.86) and about half of the merge response (0.46). It carries 0.16-0.28 of the soft-split, emission and hardest-split responses.
  - S0 removes ≤ 0.10 everywhere, so the prediction "S0 carries the rank-13 split" is excluded.
  - S29 removes −0.14 to 0, so the radial object does not carry the count.
  - For soft splits, the top singles are S1 (1.22) and S30 (0.84). For emissions they are S1 (0.53), S30 (0.43) and the context g (0.38). S1 counts every non-leading particle beyond r = 0.2, weighted by its distance and independent of pT; the emitted particles sit at r = 0.3, inside S1's range. S30 changes when an empty slot is filled.
  - No named hold passes Holm against the other live objects (smallest Holm 0.35).
- Competing predictions: H1 (counting) is supported for the response itself (toward gluon; collinear = non-collinear for soft splits). Its mechanism prediction, that S19 carries the soft splits and S0 the rank-13 splits, is not observed. H1′ is excluded. H1″ is excluded except at the rank-13 edge. H0 is rejected against untrained, and against the non-collinear control only at ranks 0 and 13.

## Issues
1. The test-sample accuracy, 80.38 %, is 0.2 pp below the 80.6-81.3 % check band. The fit sample (81.06 %) is inside it. The binomial SE on 10,000 jets is 0.40 pp, E34 obtained 80.89 % on 25,000 jets with the same code path, and the S → logits tail reproduces every call exactly. The shortfall is attributed to sample variation; the run was not repeated.
2. The four-momentum diagnostic in the log printouts was misread at first: a format string cut the exponent, so 4.3e-16 printed as "4.33". Job e40-chk (`interp/e40_check.py`) confirmed that the null edit's maximum relative change is 4.3e-16. The JSON values are correct.
3. null_resort changes 0.10 % of calls (4 jets; mean Δ −0.0001). Particles with equal pT are re-ordered by the permutation, and the network depends on slot order.
4. Every split requires N ≤ 63, which removes 51 % of gluon jets (N = 64). The split rows are therefore weighted toward quark-like gluon jets. E28w showed that the split rate at N = 64 equals the rate at N ≤ 60.
5. The emitted particles sit at r = 0.3, inside S1's range (r > 0.2). The S1 share for emissions therefore mixes "one more particle" with "one more particle at wide angle". Emissions at r < 0.2 were not run.
6. The plan asked for 500 random size-matched S sets. The live-S singles and pairs were enumerated instead (19 and 171 sets), which is exhaustive for these sizes. Single holds cannot reach rank p < 0.053.
7. The non-collinear control gives the pair a small mass, and the merge changes the jet mass (mean −0.24 GeV). move2 has the same mass change as merge2 and serves as the control for it.
8. Untrained seeds 1 and 3 produce a nearly constant margin (SD ≤ 3e-4), so their SD-normalised values are noisy.
9. Peak memory was 2,993 MiB of the 3 GiB request (97 %, inside the 20-150 % band). The smoke run on 750 jets peaked at 1,680 MiB, so the exempt 2 GiB request would not have fit.

## Explainer (site/explain.py, Paper A, new claim A19qg)
**Question.** When JEDI-linear separates gluon from quark jets, does it count particles, a quantity that changes when a particle is split in two? If so, which internal objects carry the count?
**Background.** Gluon jets carry more particles than quark jets: on average 56.7 vs 37.4 here. The particle count $N$ is not infrared- and collinear-safe: splitting a particle $(p_T,\eta,\phi)\to(f p_T,\eta,\phi)+((1-f)p_T,\eta,\phi)$ or adding a particle with $p_T\to0$ changes $N$ but no IRC-safe observable, e.g. $m$ or $M_\kappa=\sum_i p_{T,i}r_i^\kappa$. The network sees pT-ordered slots. Its second-round objects include a soft-multiplicity sum $S_{19}=\sum_{\text{ranks}\ge13}\mathrm{ReLU}(2.86-1.39\,p_T)$, a sum over the 14 hardest particles $S_0$, and a radial sum $S_{29}$.
**Method.** On 4,000 test q/g jets, edit the input at fixed jet four-momentum: split the hardest, the 13th or a soft particle collinearly ($f$ = 0.5, 0.2); add 1, 3 or 5 particles of 0.5 GeV (or 0.05 GeV) at $r=0.3$, rescaling and boosting the rest so that $\sum p^\mu$ is unchanged; merge the two softest particles. Compare with the same split with halves 0.05 apart, an exact rank-preserving version of the split computed from per-slot pooled terms, a rotation, untrained networks and a re-sort no-op. Measure $\Delta(\log p_g - \log p_q)$ and calibrate it by $R_N=\Delta/(\frac{d\,\text{margin}}{dN}\Delta N)$, where $\frac{d\,\text{margin}}{dN}=0.137$ is the slope across real jets. Hold S objects at unedited values and measure the share of the change removed.
**Result.** (figure `e40_qg_count.png`) Every edit that adds a particle moves the margin toward gluon: soft split +0.062 ($R_N$ 0.45), rank-13 split +0.079, hardest split +0.282, one 0.5 GeV emission +0.180 ($R_N$ 1.32). A merge moves it −0.203. Soft splits respond as much with the halves 0.05 apart as at the same point. With 0.05 GeV emissions the response is 0.73 of that with 0.5 GeV. The rank-13 response disappears when the softer particles keep their slots, and $S_{19}$ carries 0.86 of it. For soft splits and emissions, the change is spread over $S_1$ (distance beyond $r=0.2$, pT-independent), $S_{30}$ (empty-slot count) and the context units.
**Checks.** The four-momentum is exact to $10^{-8}$. The tail and per-slot reconstructions reproduce the network exactly. The re-sort no-op gives −0.0001. Untrained networks change no calls. No named hold beats all other objects after Holm correction.
**Why it matters.** JEDI-linear's quark/gluon decision is not IRC-safe: it reads the number of particles, through several count-like objects (soft-pT hinges, empty-slot terms, wide-angle particle sums), not through the radial mass-like object $S_{29}$. This bears on how the object layer is described: some objects compute particle counts, which are not functions of the jet's energy flow.

Glossary additions: **IRC safety** (unchanged by collinear splits and by adding zero-momentum particles); **collinear split** (one particle → two at the same direction with pT fractions f, 1−f); **soft emission at fixed four-momentum** (added soft particles, others rescaled and boosted so $\sum p^\mu$ is unchanged); **$R_N$** (edit response / (natural margin-vs-N slope × ΔN)); **rank-preserving split** (per-slot reconstruction in which no other particle changes slot).

Conclusion (table row): At fixed jet four-momentum, adding a particle by a collinear split or a soft emission moves JEDI-linear's quark/gluon score toward gluon. One added soft particle moves it about 1.3 times the average per-particle rise seen across real jets, and much of the response remains when the particle's momentum goes toward zero. The network's q/g call therefore counts particles and is not IRC-safe. The count passes through several objects: the soft-multiplicity object S19 carries the response where particles cross the rank-13 window edge, while wide-angle and empty-slot sums carry most of it for soft particles. No single object passes multiple-comparison correction.
