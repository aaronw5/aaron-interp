# E34 verification: fresh-sample re-run and three added controls (count location, mass at fixed radii, sign reversal)

Script `interp/e34_verify.py` (imports `interp/e34_quark_gluon_jedi.py` and runs its `analyse()` unchanged, with added edits). Results `results/jedi/e34_verify.json` (smoke: `e34_verify_smoke.json`). Logs `logs/e34-verify.log` (final), `logs/e34-vsmoke.log` (first smoke; failed on an infeasible edit, see Issues 1), `logs/e34-vsmoke2.log`. Job e34-verify: 1 CPU / 2 GiB, wall 269 s, peak 1,226 MiB, 0.98 cores. No figure was made; the E34 figure was inspected (see Figure check).

## Conclusion
The four E34 responses reproduce on a fresh sample: adding particles and raising the mass at fixed girth move JEDI-linear toward gluon, while raising girth at fixed $\sum p_T r^2$ and raising $p_T^D$ move it toward quark, each at nearly the same size as in E34. Three parts of the reading do not hold as stated. First, the network does not respond when the invariant mass is lowered 5 % by rotating one particle around the axis at fixed distances (a classifier on mass alone does), so the "mass" response is a response to the radial distribution of particles, not to the invariant mass. Second, the count response depends on where the added particles sit (core splits give 0.68 of the response of wide-angle splits) and core splits do not pass through S1. Third, the ratios to the reference classifiers change when those classifiers are retrained (count / count-only classifier 0.54 → 0.70). The untrained network's response to the mass edit matches the trained network's in units of each network's score spread, so it is unclear how much of the mass response is learned.

## Definitions (as in E34)
- Margin = $\text{logit}_g-\text{logit}_q=\log(p_g/p_q)$; positive change = toward gluon. Edit − null = mean over jets of (margin change under the edit) − (margin change under the matched null edit).
- $r$ = distance of a particle to the jet axis in $(\Delta\eta,\Delta\phi)$. $M_\kappa=\sum p_T r^\kappa$. Girth $=M_1/M_0$. $p_T^D=\sqrt{\sum p_T^2}/\sum p_T$. Mass = invariant mass of the massless constituents. For a jet whose axis is the pT centroid, $m^2\approx M_0M_2-|\sum p_T\vec r|^2$ (small angles); the second term is the recentring term.
- MLP_x: 2 × 32 ReLU classifier trained on g vs q using candidate x only (log N, girth, $p_T^D$, log m); MLP_all on all four. Ratio = JEDI (edit − null) / MLP (edit − null) on the same jets.
- Share removed by a hold = $1-\sum\Delta\text{margin(edit, element held at unedited value)}/\sum\Delta\text{margin(edit)}$. Rank p = fraction of size-matched live-S sets with share ≥ the named set's (singles: 19 sets, floor 1/19 = 0.053; pairs: 171; 4-sets: 200 random).
- S1 = $\sum_{\text{ranks}\ge1}(r-0.2)_+$ (pT-independent sum of distance beyond r = 0.2 of all particles except the hardest). S30 = $\sum\text{ReLU}(3.47-21r)$ over all 64 slots (3.38 per empty slot, so it falls by about 3.4 for every new particle placed at r > 0.165 and by about $21r$ for a new particle near the axis). S19 = soft multiplicity over ranks 13–63. S43 = a (pT, r) hinge over the 14 hardest particles.

## Procedure
- Network: official JEDI-linear 3-feature n64, exact fixed-point forward (`jedi_core` via `e28w_split_route_jedi.Net`). Check: `sample(5000, 3421)` (25,000 jets) 5-class accuracy **80.68 %** (E34: 80.89 % on seed 3401). Tail check (S → logits) reproduces every call.
- Sample: the 10,000 g and q jets of `sample(5000, 3421)` (fresh seed, not used by E34). JEDI g-vs-q AUC 0.923 (E34 0.921 / 0.922).
- MLP references retrained (same architecture and code) on 20,000 g/q jets of `full_test` disjoint from this sample (seed 3423; overlap 0). AUCs: MLP_all 0.886, MLP_N 0.850, MLP_ptD 0.801, MLP_m 0.757, MLP_girth 0.723.
- E34's six edits, nulls, holds, scans and statistics were run unchanged (1,000 jet bootstraps; 10,000 sign-flip permutations; McNemar; Holm over the four primary edits).
- Added edits (same machinery, same 1 % isolation filter on the other three candidates):
  - **Egirthm**: girth × 0.95 at fixed $M_2$ (power map $r\to c\,r^\gamma$, γ > 1, median 1.053). Sign check of Egirth.
  - **Emassm**: mass × 0.95 at fixed $M_1$ (power map, γ < 1, median 0.948). Sign check of Emass.
  - **Emrotm**: mass × 0.95 by rotating one particle (the largest $p_T r$ among ranks ≥ 1; median rank 1, median r 0.13) about the jet axis at fixed r, by the smallest angle that reaches the target (median 1.94 rad). Every pT and every r is unchanged, so N, $M_0$, $M_1$, $M_2$, girth, $p_T^D$ and every radial sum are exact; only the azimuthal arrangement, and hence the invariant mass, changes. Null = no edit. A mass-raising version is not possible: with the axis at the pT centroid the recentring term is already near 0, so at fixed radii no rearrangement raises the mass (smoke run: 0 feasible jets). Emrotm is feasible on 1.7 % of jets (n = 169: 70 g, 99 q).
  - **Ecin / Ecout**: Ecount's collinear split (k = max(1, round(0.05 N)) particles each replaced by two halves at the same position, N + k ≤ 64), but the split particles are the k softest with r < 0.2 (Ecin) or with r ≥ 0.2 (Ecout), instead of the k softest overall. N rises 5.0 % in both; girth, mass, $M_0$, $M_2$ exact; $p_T^D$ within 0.02 %. Null: the same particles rotated by random azimuths (as Ecount).
  - Second untrained network (`Net(untrained=True, seed=1)`) on the primary edits, Edil, Emrotm, Ecin, Ecout.

## Results

### 1. Reproduction of the E34 numbers (fresh seed 3421 vs E34 main seed 3401)
| edit | n | edit − null [95 % CI] | E34 main | perm p (Holm 4) | per sd | E34 per sd |
|---|---|---|---|---|---|---|
| Ecount (N +5 %) | 6,806 | +0.109 [0.102, 0.117] | +0.112 [0.105, 0.120] | 1e-4 (4e-4) | +0.89 | +0.90 |
| Egirth (girth +5 %, $M_2$ fixed) | 9,626 | −0.083 [−0.087, −0.079] | −0.089 [−0.094, −0.085] | 1e-4 (4e-4) | −1.59 | −1.65 |
| EptD ($p_T^D$ +5 %) | 7,265 | −0.055 [−0.057, −0.052] | −0.052 [−0.054, −0.049] | 1e-4 (4e-4) | −0.42 | −0.40 |
| Emass (m +5 %, girth fixed) | 9,946 | +0.081 [0.077, 0.086] | +0.083 [0.079, 0.087] | 1e-4 (4e-4) | +1.02 | +1.03 |
| EptDm ($p_T^D$ −5 %) | 1,552 | +0.031 [0.027, 0.036] | +0.031 | 1e-4 | | |
| Edil (m and girth +5 %) | 10,000 | +0.025 [0.022, 0.029] | +0.027 | 1e-4 | | |

Signs agree in both classes. Calls changed, edit vs null (McNemar, Holm 4): Ecount 3.42 % vs 2.88 % (p 0.041; E34 7e-6), Egirth 3.65 % vs 2.29 % (1e-11), EptD 1.97 % vs 0.51 % (1e-18), Emass 5.39 % vs 2.41 % (4e-41). The Ecount call-change excess is smaller than in E34 and only marginal here.

Holds reproduce (share removed, fresh seed): Ecount S1 1.43 [1.36, 1.50], S30 0.89, S19 0.11 [0.10, 0.12], S0 0.01, g 0.30, a32 0.16, 4-set 2.33 (rank p 0.015); Egirth S1 1.19, S56 −1.31, S30 −0.005, g −0.30; EptD S43 0.90 (top single), S0 0.00 [−0.02, 0.02]; Emass S1 4.28, S30 0.46, S29 −0.67, 4-set 4.68 (p 0.010). Untrained network (seed 0) in units of its margin SD (0.024): Ecount −0.005 [−0.021, 0.012], Egirth +0.008 [−0.003, 0.019], EptD +0.002 [−0.003, 0.008].

### 2. Ratios to the reference classifiers (retrained MLPs; JEDI / MLP, edit − null, 95 % CI over jets)
| edit | JEDI / MLP_all, fresh | E34 main / rep | JEDI / MLP_x, fresh | E34 main |
|---|---|---|---|---|
| Ecount | 0.51 [0.48, 0.54] | 0.54 / 0.52 | MLP_N **0.70 [0.65, 0.75]** | 0.54 [0.51, 0.58] |
| Egirth | **0.68 [0.65, 0.72]** | 0.78 / 0.74 | MLP_girth −1.83 | −2.01 |
| EptD | **0.72 [0.68, 0.75]** | 0.57 / 0.57 | MLP_ptD 0.31 [0.29, 0.32] | 0.29 |
| Emass | 0.59 [0.56, 0.62] | 0.58 / 0.57 | MLP_m 1.04 [0.98, 1.09] | 1.02 |

The retrained MLP_N responds +0.156 to Ecount (E34's +0.207) and MLP_all responds −0.076 to EptD (E34's −0.091); JEDI's responses are unchanged. The jet-bootstrap CIs do not include the variation between MLP fits, which here moves the Ecount / MLP_N ratio from 0.54 to 0.70 and reorders the MLP_all ratios (E34: girth highest at 0.78; fresh: $p_T^D$ highest at 0.72). MLP_girth's response to Egirth is +0.046 (toward gluon) and MLP_all's −0.122 (toward quark), as in E34.

### 3. Sign reversal (H0: response to −5 % = −(response to +5 %))
| candidate | +5 % edit − null | −5 % edit − null | −5 % / +5 % magnitude |
|---|---|---|---|
| girth (fixed $M_2$) | −0.083 [−0.087, −0.079] | +0.065 [0.061, 0.068] | 0.78 |
| mass (fixed $M_1$) | +0.081 [0.077, 0.086] | −0.136 [−0.141, −0.131] | **1.67** |
| $p_T^D$ | −0.055 | +0.031 (n 1,552, different jets) | 0.58 |

Signs reverse in all three. The magnitudes are not symmetric (CIs disjoint for girth and mass; H0 rejected). For a 5 % decrease the order of responses is mass (0.136) > count (0.109, increase only) > girth (0.065), so the E34 ranking "count > girth > mass > $p_T^D$ per 5 %" holds for increases only. MLP_all responds −0.135 to Emassm (ratio JEDI / MLP_all 1.01 [0.97, 1.04]).

### 4. Invariant mass at fixed radii (Emrotm; H1 "network reads invariant mass": JEDI response ≈ MLP_m response × 1.04, the Emass calibration; H0' "network reads only radial, fixed-axis sums": response 0)
| read-out (n = 169) | change for mass −5 % [95 % CI] |
|---|---|
| JEDI-linear margin | **+0.003 [−0.025, 0.031]** (perm p 0.81 vs 0) |
| MLP_m | −0.062 |
| MLP_all | −0.112 |
| JEDI / MLP_m | −0.06 [−0.52, 0.41] (Emass: 1.04; Emassm: 1.59) |
| untrained (seed 0), per its SD | −0.012 [−0.062, 0.034] |

H1 predicts about −0.064; the CI excludes it and the ratio CI excludes 1, so H1 is excluded at the 95 % level on these 169 jets. The response is consistent with 0 (H0' not rejected). The 169 jets are those where one rotation can lower the mass by 5 % (1.7 % of jets), so this is a selected subset.

### 5. Count edit by location of the split particles (H1 "count alone": same response per added particle; H2 "wide-angle sums S1 / S30 carry it": Ecout ≫ Ecin, Ecin ≈ 0)
| edit (N +5.0 %) | n | edit − null [CI] | null | JEDI / MLP_N | S1 share | S30 share | S19 share | S19+S30 share (rank p, 171 pairs) |
|---|---|---|---|---|---|---|---|---|
| Ecin (split particles r < 0.2) | 6,806 | **+0.082 [0.077, 0.087]** | +0.012 | 0.52 | 0.12 [0.09, 0.14] | **0.68 [0.66, 0.71]** (top single, p 0.053) | 0.16 | 0.84 (p 0.012) |
| Ecout (split particles r ≥ 0.2) | 6,543 | **+0.120 [0.111, 0.128]** | +0.044 | 0.77 | **2.02 [1.94, 2.11]** | 0.99 | 0.10 | 1.08 (p 0.12) |
| Ecount (softest overall) | 6,806 | +0.109 [0.102, 0.117] | +0.028 | 0.70 | 1.43 | 0.89 | 0.11 | 1.00 (p 0.12) |

Ecin / Ecout = 0.68; the difference 0.038 has an approximate z of 7.6 from the two bootstrap intervals (p < 1e-12). H1 (count alone) is excluded, and H2 in its strong form (Ecin ≈ 0) is also excluded: core splits move the margin by two thirds as much, and that response passes through S30 (and S19 + S30, rank p 0.012 among pairs, the only named hold below 0.05 here), not through S1. S1 is the top single only when wide-angle particles are split. The untrained network (seed 0) does not respond to either (−0.008, −0.014 SD units, CIs include 0).

### 6. Untrained-network control for the mass edits
| edit | untrained seed 0, per its SD [CI] | trained, per its SD | E34 untrained / trained |
|---|---|---|---|
| Emass | +0.032 [0.018, 0.046] | +0.031 | +0.021 / +0.032 |
| Emassm | −0.023 [−0.038, −0.009] | −0.052 | not run |
| Edil | +0.028 [0.016, 0.041] | +0.010 | +0.023 / +0.010 |

On this sample the untrained network's Emass response equals the trained one in SD units (ratio 1.03; E34 0.66). The second untrained network (seed 1) has a nearly constant output in fixed point (margin SD 2.4e-4, 100 × smaller than seed 0) and shows no response to any edit (all |effect| < 0.03 SD, all p ≥ 0.22). It is too degenerate to serve as a control; the seed-0 result is therefore the only informative untrained comparison and is seed-specific in an unknown way.

## Code review of `interp/e34_quark_gluon_jedi.py` and `e34_plot.py`
- Stages, masks and quantizers: edits act on raw (pT, Δη, Δφ) inputs and go through the full fixed-point forward (`Net.fwd`); holds use `Net.tail` from the S stage (the S → logits tail reproduces every call). Padded slots: candidates use pT > 0; Ecount places the new halves in former padding slots (intended); the network still sees padding as in normal operation. No error found.
- Split leakage: MLP training jets disjoint from the evaluation samples (overlap 0, here and in E34).
- Edits change only what is stated: the per-candidate relative changes in the JSON confirm targets at 5.0 % and the other three within 1 % (most exactly 0).
- Minor: (a) the a32 per-slot hold uses the unedited jet's slot order, but Ecount and EptD re-sort the edited jet, so the held a32 values are not aligned with the same particles for re-sorted slots (affects only the a32 share, 0.16–0.18). (b) `jedi_over_mlp` is printed for MLPs whose input variable the edit does not change (denominator ~1e-9), giving values such as 1e10 in the JSON; these entries are meaningless and are not used in the reading. (c) The Ecount null (random azimuths of the k softest particles) has a non-zero mean response (+0.025 to +0.028; Ecout null +0.044), so it is not a zero-response null; it is subtracted, which reduces the reported effect. (d) Ratio CIs are over jets only; they do not include MLP refit variance (Section 2).
- Figure `results/jedi/e34_quark_gluon.png`: panel (b) is readable and matches the JSON. The untrained-network bars are drawn in raw log-odds (its margin SD is 0.025 vs 2.60), so they appear as zero; the caption does not say this, and a reader would take them as "no response", while in SD units the untrained Emass response is 0.66–1.03 of the trained one. Panel (a): the edits are too small to see at this marker scale except Ecount's added markers. The title's "10,000 g/q test jets" applies to the sample; n per edit is in the tick labels.

## Issues
1. The first smoke run (e34-vsmoke) included Emrot (mass +5 % by one-particle rotation); it had 0 feasible jets and the run stopped on an empty selection. Emrot was removed (reason in the script docstring) and the runs were relaunched (e34-vsmoke2, e34-verify).
2. Emrotm is feasible on 1.7 % of jets (n = 169), selected for having one particle whose rotation can lower the mass 5 %; the result is limited to that subset and has a wide CI (±0.028).
3. Per-jet arrays were not saved (outputs restricted to `e34_verify*.json`), so the Ecin vs Ecout difference test uses the two bootstrap intervals (normal approximation), not a paired test.
4. Holm over the primary edits was applied as in E34; the added edits are reported with raw permutation p (all 1e-4 except Emrotm 0.81); with Holm over all 11 edits every p ≤ 1.1e-3 except Emrotm.

## Corrected reading
On 10,000 g/q jets from a fresh sample, the four E34 edits reproduce: splitting soft particles (N +5 %) +0.109, raising girth at fixed $M_2$ −0.083, raising $p_T^D$ −0.055, raising mass at fixed girth by a radial power map +0.081 log-odds, each beyond its null (Holm p 4e-4), with the same object shares (S1 top single for count, girth and mass edits; S43 for $p_T^D$; S19 0.11; S0 0.00). Corrections: (i) Lowering the invariant mass 5 % by an azimuthal rotation at fixed radii gives +0.003 [−0.025, 0.031] while MLP_m gives −0.062 (n 169), so the "mass" response is a response to the radial profile ($M_2$-like, fixed-axis), not to invariant mass. (ii) Count edits made by splitting core particles (r < 0.2) give 0.68 of the response of wide-angle splits; the core-split response is carried by S30 (0.68) and S19 + S30 (0.84, rank p 0.012), not S1 (0.12), so S1 is the leading object only for count edits that add wide-angle particles. (iii) Responses are not sign-symmetric: −5 % mass gives 1.67× the +5 % response and −5 % girth 0.78×, so the ranking of the four per 5 % depends on direction. (iv) JEDI/MLP ratios depend on the MLP fit: with retrained MLPs, MLP_all ratios are 0.51 (count), 0.68 (girth), 0.72 ($p_T^D$), 0.59 (mass), and count / MLP_N is 0.70, not 0.54. (v) The untrained network (seed 0) responds to the mass edit as strongly as the trained one in SD units (1.03), so the learned share of the radial-mass response is not established; a second untrained seed is degenerate and uninformative.

## Explainer (site/explain.py, Paper A, A17 verification)
**Question.** Do E34's responses hold on new jets, and does each edit test the property it is named after?
**Background.** E34 moved one of four jet properties (N, girth, $p_T^D$, mass) by 5 % with the other three held, and read the change of $\log(p_g/p_q)$. Two jets can have the same mass but different radial profiles: for a jet centred on its pT centroid, $m^2\approx M_0M_2-|\sum p_T\vec r|^2$, so the mass also depends on the azimuthal arrangement through the second term.
**Method.** Re-run on `sample(5000, 3421)` with retrained reference MLPs; add (a) a mass edit that rotates one particle around the axis at fixed $r$ (all radial quantities exact, mass −5 %), (b) count edits that split only core ($r<0.2$) or only wide-angle ($r\ge0.2$) particles, (c) the −5 % versions of the girth and mass edits, (d) a second untrained network.
**Result.** All four E34 responses reproduce within 0.006 log-odds. The rotation mass edit moves the network by $+0.003\,[-0.025,0.031]$ while a mass-only classifier moves by −0.062. Core splits move the score 0.68 as much as wide splits, through S30 rather than S1. The −5 % mass edit moves the score 1.67 times as much as the +5 % edit. Ratios to the reference classifiers change by up to 0.16 when they are retrained.
**Checks.** Accuracy 80.68 %; tail identity exact; MLP training jets disjoint; null edits subtracted; untrained seed 0 (seed 1 degenerate).
**Why it matters.** JEDI-linear's "mass" sensitivity in q/g is a sensitivity to how far particles sit from the fixed axis, and its "count" sensitivity has a wide-angle part (S1) and a part that counts filled slots (S30).

Glossary additions: **recentring term** $|\sum p_T\vec r|^2$, the part of the jet mass that depends on the azimuthal arrangement of particles around the axis; **azimuthal rotation edit**, a particle moved along the circle around the jet axis at fixed distance.

Nothing committed; the ledger agent commits.
