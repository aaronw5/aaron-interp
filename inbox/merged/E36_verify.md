# E36 verification: smallest call-flipping edits at fixed jet pT, mass and axis (JEDI-linear 3f n64)

Verifier script `interp/e36_verify.py` (imports the E36 functions unchanged). Jobs: `e36-verify` (1 CPU / 2 GiB; fresh-sample rerun,
metric variant, direction controls and program baseline finished and saved; the last block, the hand-made hardness edit, crashed in
E36's `project()` on its random control: overflow, then a singular matrix), `e36-verify2` (1 CPU / 2 GiB, `--only-hardness`: hardness edit with
a ramped pseudo-inverse projection, plus a check of E36's "+83 %" number; exit 0, wall 1,072 s, peak 2,047 MiB, which is the
memory.peak value including page cache). Two smoke jobs (`e36-verify-smoke`, `-smoke2`) failed in the same hardness block.
Results: `results/jedi/e36_verify.json`, `results/jedi/e36_verify_hardness.json`; logs `logs/e36-verify*.log`.

## Conclusion
On new jets, the main E36 numbers come out the same. Edits found on the float copy of the network flip the real network's call on 40 % of jets, against about 2 % for random edits of the same size. Pooled over all jet types, the found edits put most of their size into pT changes, and this still holds when angular moves are made 6 times cheaper. Three parts of the reading do not hold as written:
- For W → Z flips, the found edits are mostly radial moves (particles moved toward or away from the jet axis), not pT changes.
- The "hardest particle gains 83 %" figure compares the leading pT after re-sorting with the leading pT before. The originally hardest particle itself gains a median 39 %, and it loses pT in 39 % of jets.
- A hand-made edit that only moves pT into the leading particle changes few calls. It raises pT^D (how unevenly pT is shared among particles) about as much as the found edits do, but flips 4 % of calls.

Separately, the extracted programs' 61 % agreement with the network is barely above the 59 % that a rule predicting "never flips" gets. What drives the remaining flips is still open.

## 1. Code review (`interp/e36_counterfactual_flip_jedi.py`, `interp/e36_plot.py`)
Checked, no defect found:
- **Float + masks program.** `float_program` keeps every zero-bit input-quantizer entry as a 0/1 mask (l1, qs, l2, dd, l3, qs1, d4–d7), as in `e3m_masks.py`. It uses the fused kernels from `jedi_core`, the same input scaling `(x − shift)/scale` as `jetdistill.data.inputs`, and the pooling scales of `qs`/`qs1`. On the fresh sample, float and exact give the same call on 95.4 % of jets (E36: 95.4 %).
- **Edit and padding.** Only real particles (pT > 0) are edited. Padding keeps pT = 0 and does not count in ε. pT stays positive through exp(u). Particles are re-sorted by pT before every forward pass, with the sort order held out of the gradient.
- **Constraints.** The Gauss–Newton Jacobian of Σp^μ with respect to (u, w, w′) is correct. On the found edits, |Δm|/m reached 8.9e-13 and the axis shift 2.6e-14 mrad (99th percentiles). The measured change in M2 = ΣpT·r² is −1e-5 sd.
- **No split leakage.** The verify sample is `sample(300, 3701)` minus the 129 jets that overlap the E36 main, replicate and pool samples (N 1,371).

Problems in the reading or report (not code bugs):
1. **The "+83 %" number is mislabelled.** Recomputed from `e36_edits.npz` (main, N 2,783 primary jets):
   - Median (leading pT after re-sorting) / (leading pT before) − 1 = +82.6 %. This is the reported number, but the leading particle after the edit can be a different particle.
   - The originally hardest particle's own pT change has a median of +38.8 % (mean +85.9 %).
   - That particle gains pT on 61.4 % of jets and loses pT on the other 38.6 %.
   - Fresh sample: median +40.5 % (SC 0.02).
2. **The program agreement has no trivial baseline.** Agreement means "the program flips to the target iff the network does". The network flips on 41 % of jets, so a rule that always predicts "no flip" agrees on 59 %. See §4.
3. **Pooled coordinate shares hide a group difference.** The pooled result (67 % of ε² in pT changes) comes from Z→W, q↔g and top jets. In E36's own `norm_shares_by_group` (main), the W→Z edits put 0.41 of ε² into pT changes vs 0.49 for random edits, and 0.47 into radial moves vs 0.26 for random edits. The fresh sample gives pT 0.43 and radial 0.45 for W→Z. W→Z is the group with the highest transfer (50 %).
4. **thesis_note: "a direction the radial-object description does not name".** S20 (hinge on leading-particle pT), S51 (pT of the 2–3 hardest) and S43 are named objects in the E11 program. pT^D is already a causal variable of the g/q score in A17 (E34). The supported statement is narrower: the headline radial objects (S1, S29, S30) move little under the found edits.
5. **Figure `results/jedi/e36_edit.png`.**
   - The left panel's colour scale is clipped at ±30 %. The pT changes it is meant to show are +40–80 % gains and > 90 % losses.
   - The example is a W→Z jet whose edit is mostly angular moves. That matches item 3, but not the right panel's pooled message ("pT changes").
   - Axes are in physical units and the right panel is readable.
   - Suggested changes: colour range ±100 %, and a per-group panel or a note that the right panel pools all groups.
6. **Search method (caveat).** Adam rescales each coordinate's step separately, so the search does not minimise ε exactly. The found edits are an upper bound on the smallest edit, and their direction partly reflects the optimiser. E36 lists this as issue 4.

## 2. Rerun on a fresh sample (N 1,371; primary W/Z/q/g jets with an edit inside the tolerances: 778)
Procedure as in E36: 200 Adam steps, lr 0.05, margin 0.05, angle unit SC = 0.02 rad. The random control is 20 random edits per jet at the found edit's v-space norm, tangent to the constraints and projected back with 30 Gauss–Newton steps. All draws were inside the tolerances.

| quantity | E36 main | fresh, SC 0.02 | fresh, SC 0.05 |
|---|---|---|---|
| exact-network accuracy (unedited) | 81.8 % | 82.8 % | 82.8 % |
| float program flipped (primary jets) | 74.4 % | 74.7 % | 77.9 % |
| median ε | 6.0 | 6.2 | 4.4 |
| transfer to exact network | 41.1 % [39.3, 42.8] | 40.5 % [37.0, 44.2] | 40.5 % [37.2, 43.3] |
| random same-size edits, flip to target | 1.65 % | 1.88 % [1.36, 2.46] | 1.54 % [1.13, 2.02] |
| found − random | +0.394 | +0.386 [0.352, 0.423], p 0.001; McNemar 304:10, p 1e-76 | +0.390 [0.357, 0.418]; McNemar 322:7 |
| robust edits (float margin < −0.5), transfer | 67.5 % | 66.9 % (N 311) | 62.6 % (N 310) |
| share of ε² in pT changes, found / random | 0.670 / 0.414 | 0.682 / 0.417 | 0.646 / 0.400 |
| share in azimuthal moves | 0.073 / 0.291 | 0.070 / 0.294 | 0.078 / 0.297 |
| share in pT ranks 0–13 | 0.601 / 0.263 | 0.586 / 0.264 | 0.617 / 0.271 |
| share on the hardest particle alone | not reported | 0.107 / 0.009 | 0.128 / 0.008 |
| Δ pT^D / sd, found / random | +1.57 / +0.09 | +1.65 [1.49, 1.79] / +0.11 | +1.36 / +0.18 |
| \|ΔS20\| / sd, found / random | 1.94 / 0.27 | 2.14 / 0.35 | 1.80 / 0.36 |
| \|ΔS51\|, \|ΔS43\| / sd, found / random | 1.26, 1.18 / 0.24, 0.22 | 1.31, 1.27 / 0.25, 0.25 | 1.11, 1.11 / 0.31, 0.22 |
| \|ΔS1\|, \|ΔS29\|, \|ΔS30\| / sd, found / random | 0.22, 0.24, 0.20 / 0.13, 0.09, 0.13 | 0.23, 0.25, 0.22 / 0.13, 0.10, 0.15 | 0.28, 0.23, 0.23 / 0.22, 0.13, 0.22 |

All share differences have bootstrap p ≤ 0.003. The fresh sample did not rerun C2ST, kNN or the programs; those are taken from E36.

**Metric variant.** SC = 0.05 rad makes an angular move 0.05 rad, not 0.02 rad, cost ε = 1, so angular moves cost 6.25 times less in ε². The pooled pT share drops by 0.04 and the azimuthal share stays at 0.08. The pooled preference for pT changes over angular moves therefore does not come from the exchange rate between pT and angle in the metric. The relative-pT form of the metric (removing a particle costs at most 1 in ε²) was not varied.

**Per group (fresh, SC 0.02; share of ε² in pT / radial / ranks 0–13, found edits):**
- W→Z: 0.43 / 0.45 / 0.39 (transfer 47.7 %, N 197)
- Z→W: 0.76 / 0.19 / 0.76 (26.2 %, N 202)
- q→g: 0.71 / 0.21 / 0.54 (10.5 %, N 114)
- g→q: 0.80 / 0.16 / 0.62 (58.9 %, N 265)

Observable changes on W→Z edits, in sd: τ21 +1.00, D2 +0.40, girth −0.28, pT^D +1.99.

## 3. Direction controls on the found edits (fresh, SC 0.02, N 778; exact network)
All controls are projected back onto the constraints with 30 Gauss–Newton steps, and 100 % stay inside the tolerances. Δmargin is the change in (logit of the target class − logit of the current call). The found edits give Δmargin +3.08 [2.94, 3.23], from an unedited mean of −3.32. Each McNemar test compares the found edit with the control on the same jets.

| control | what it keeps | flip to target | Δmargin | ε / found ε | McNemar (found only : control only) |
|---|---|---|---|---|---|
| found edit | | 40.5 % | +3.08 | 1 | |
| sign-randomised | same \|v\| per particle and coordinate (same shares), random signs | 3.1 % [1.9, 4.2] | +0.64 | 1.00 | 297 : 6 |
| negated (−v) | same shares, opposite direction | 1.4 % [0.8, 2.3] | +0.04 [−0.05, 0.13] | 0.91 | 310 : 6 |
| pT part only | u kept, angular moves set to 0 | 19.2 % [16.5, 21.9] | +2.02 | 0.86 | 191 : 25 |
| angle part only | angular moves kept, u set to 0 | 3.3 % [2.2, 4.6] | +0.73 | 0.45 | 295 : 6 |

SC 0.05 gives the same pattern: 2.2 %, 1.4 %, 11.6 % and 4.9 %.

H0: the found edits' flips come only from where the edit size sits (coordinate and particle profile), whatever the direction. Prediction: sign-randomised edits flip about as often as the found edits. Measured: 3.1 % vs 40.5 %, so H0 is excluded (McNemar p 1e-79).

What the controls show:
- The pT component alone produces two thirds of the margin change and half of the flips. The angular component alone does little. Neither alone reproduces 40 %.
- The negated edit leaves the margin unchanged on average, where a linear response would make it about −3. Over these edit sizes the response is far from linear.

## 4. Program agreement against a constant predictor (E36's stored counts)
Recall, specificity and N reproduce the stored 2×2 table exactly. On the main sample, E11 has 276 network flips predicted, 866 network flips missed, 226 false flips and 1,415 correct non-flips. The constant predictor always predicts "no flip". Cohen's κ is agreement corrected for chance (0 = chance, 1 = perfect). Balanced accuracy is the mean of recall and specificity. Intervals are 1,000 bootstrap draws over jets.

| predictor (main; replicate) | agreement | constant "never flips" | difference; McNemar program vs constant | Cohen κ [95 %] | balanced accuracy |
|---|---|---|---|---|---|
| E11 nested program | 0.608; 0.609 | 0.590; 0.585 | +0.018 (276:226, p 0.029); +0.023 (p 0.006) | 0.114 [0.081, 0.147]; 0.126 | 0.552; 0.558 |
| B6 prog30 | 0.605; 0.609 | 0.590; 0.585 | +0.015 (200:158, p 0.030); +0.024 (p 2e-4) | 0.088 [0.058, 0.118]; 0.100 | 0.539; 0.545 |
| MLP(m, pT, τ21, D2, N) | 0.593; 0.587 | 0.590; 0.585 | +0.004 (34:24, p 0.24); +0.002 (p 0.51) | 0.018 [0.005, 0.031]; 0.010 | 0.508; 0.504 |

The McNemar p-values in the table are uncorrected. With Holm correction over the three predictors within each sample:
- main sample: E11 and B6 p 0.086, MLP p 0.24;
- replicate: B6 p 7e-4, E11 p 0.013, MLP p 0.51.

Reading:
- The 61 % agreement is mostly the 59 % that "never flips" scores.
- The E11 and B6 programs carry a small amount of information about which jets flip (κ about 0.1).
- The MLP carries almost none (κ 0.02).
- Compared with each other, the programs do differ slightly from the MLP. That margin is in κ, not in the agreement percentage.

## 5. Hand-made hardness edit (fresh; 1,042 W/Z/q/g-called jets with ≥ 8 particles on which float and exact agree)
This tests the E36 physics reading: "at fixed mass, pT and axis, the cheapest way to change the call is to change how pT is shared among the leading particles (pT^D)".

**Edit.** The leading particle's pT is multiplied by 1.5, by 2, or by 2/3, with its angles fixed. All other particles' (pT, η, φ) then take the smallest correction, in the same v-space, that restores the jet four-momentum (ramped in 10 steps with pseudo-inverse Gauss–Newton). Jet pT, mass and axis are unchanged.

**Matched control.** The same absolute pT is given to (or taken from) the rank-5 particle instead, so pT is moved without being concentrated in the leading particle. For ×2/3 this is defined on only 28 % of jets, because rank-5 particles are too soft.

**Null.** Ten random edits per jet of the same v-space size, tangent to the constraints and ramp-projected.

| leading pT | Δ pT^D / sd | ε (median) | flip to partner (W↔Z, q↔g) | rank-5 control | random same size | leading − random |
|---|---|---|---|---|---|---|
| × 1.5 | +0.68 | 0.77 | 1.6 % [0.9, 2.4] | 0.8 % | 0.8 % | +0.9 % [0.2, 1.7], p 0.011 |
| × 2 | +1.37 | 1.59 | 3.7 % [2.6, 4.9] | 1.2 % | 1.1 % | +3.1 % [2.0, 4.3], p 0.001; vs rank-5 McNemar 30:3, p 1e-6 |
| × 2/3 | −0.24 | 0.50 | 0.5 % | 0.2 % (N 291) | 0.6 % | −0.1 % (p 0.60) |

Δmargin (toward the partner class) for leading pT × 2, by called class:

| called class | Δmargin | flips |
|---|---|---|
| Z | +0.69 | 1.6 % |
| g | +0.47 | 8.0 % |
| W | +0.07 | 1.4 % |
| q | −0.23 | 3.8 % |

The rank-5 control gives +0.28, +0.20, +0.02 and −0.10 for the same classes.

H1 (E36 physics reading): a pT^D increase at fixed m, pT and axis is enough to flip the call. Prediction: flip rate near the 40 % of the found edits at a matched pT^D change. Measured: the ×2 edit raises pT^D by 1.37 sd (found edits: +1.65 sd) and flips 3.7 % of calls.

So the hardness-sharing direction is causal:
- It moves Z jets toward W and g jets toward q, above the matched control and above random edits.
- It does not by itself reproduce the found edits' flip rate.
- Its ε (1.6) is smaller than the found edits' (6.2), so the comparison is matched in pT^D, not in ε.

## 6. Corrected reading
At fixed jet pT, mass and E-scheme axis, edits found on the float + masks program flip the exact network on 40–41 % of jets, against 1.5–1.9 % for random edits of the same size. This holds on the E36 samples and on a fresh sample (N 778), and with the angle unit raised from 0.02 to 0.05 rad.

Pooled over groups, about two thirds of the edit size is in pT changes, against about 0.41 for random edits, and the hardest 14 particles carry about 0.6 of it. This pooled pattern comes from Z→W, q↔g and top jets. W→Z edits, which have the highest transfer, put more of their size into radial moves (0.45–0.47 vs 0.26 random) than into pT changes (0.41–0.43 vs 0.49 random).

The originally hardest particle gains a median 39–41 % pT; the reported +83 % is the change of the leading pT after re-sorting. It loses pT on 39 % of jets.

The specific direction of the found edit is needed. With the same per-coordinate profile but random signs, 3 % of calls flip. The pT component alone flips 19 % and carries two thirds of the margin change.

A pure hardness-sharing edit (leading pT × 2, pT^D +1.4 sd) moves the Z→W and g→q margins (+0.69 and +0.47 logit) but flips 3.7 % of calls. pT^D sharing is therefore one causal component of the found edits, not a sufficient description of them.

The E11 and B6 programs agree with the network's flips on 61 % of jets, against 59 % for a constant "never flips" rule (Cohen κ 0.09–0.13; the MLP baseline 0.01–0.02).

## 7. Explainer / glossary notes for the ledger step
- **Sign-randomised control**: the found edit with the sign of every per-particle, per-coordinate component drawn at random. It keeps where the edit size sits and removes its direction.
- **Cohen's κ**: (observed agreement − agreement expected from the two predictors' flip rates alone) / (1 − expected agreement). 0 = no better than chance, 1 = perfect.
- **Constant ("never flips") baseline**: the agreement a predictor gets by always predicting that the network does not flip. It equals 1 − (network flip rate).
- **Hardness-sharing edit**: the leading particle's pT is scaled at fixed angles, and the other particles take the smallest correction that restores the jet four-momentum.
