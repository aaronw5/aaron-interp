# E40 verify: JEDI-linear q/g particle counting at fixed jet four-momentum (fresh seeds + emission-radius control)

Script `interp/e40_verify.py` (re-uses the E40 edit functions unchanged); figure `interp/e40_verify_plot.py` -> `results/jedi/e40_verify.png`; results `results/jedi/e40_verify.json` (also on the volume); log `logs/e40-verify.log`. Job e40-verify: 1 CPU / 2 GiB, wall 50 s, peak 999 MiB.

## Conclusion
On a fresh set of jets, JEDI-linear again moves its quark/gluon score toward "gluon" when a particle is split in two or a soft particle is added while the total jet momentum and mass stay fixed, with the same sizes as in E40. A new control shows that this depends strongly on where the soft particle is added. A soft particle added 0.3 from the jet axis moves the score almost as much at 0.05 GeV as at 0.5 GeV. Added 0.02 from the axis, the response almost disappears in the same low-momentum limit (about 2 % of one average extra particle). So the network does not count every particle: in the soft limit it counts particles away from the jet core (beyond about r = 0.15), through the wide-angle sum S1 and the empty-slot term S30. Collinear splits still move the score at all radii. What sets the drop between r = 0.3 and r = 0.5 was not tested.

## Code review (interp/e40_qg_count_jedi.py, e40_plot.py, e40_check.py)
- Stages: margin from the 5-class logits (logit_g − logit_q, class order g, q, W, Z, t); S-level holds go through `Net.tail` with the unedited S as the hold value (tail reproduces the logits exactly, max |Δlogit| 0, rerun here). The g hold freezes all 64 context units in a full forward pass. Correct stages.
- Padding: `nreal` counts slots with pT > 0. Splits write the second half into slot N (first empty slot) and restrict to N ≤ 63, so no particle is dropped. The merge sets the vacated slot to (0, 0, 0), the padding value. No padded slot is counted as a particle.
- Split leakage: halves of the non-collinear control sit at −(1−f)·0.05 and +f·0.05 from the parent along one random direction (separation 0.05), with the pT-weighted centroid at the parent (checked algebraically). The rank-preserving control (`rank_patch`, mode PS) reproduces the split jet's S from per-slot terms exactly (max |ΔS| 0 on r13 and soft, rerun here).
- Emission: the other particles are scaled and boosted so that Σp^μ equals the unedited value (max relative change 4e-8 here, float32 storage). Since (Δη, Δφ) differ from (η, φ) by a longitudinal boost and an azimuthal rotation, the jet mass and pT in this frame equal the lab values.
- The k3 emissions at 0.5 and 0.05 GeV use the same random azimuths in E40 (rng seed 103) and here (same seed per radius), so that comparison is paired.
- No bug found that changes a number. Figure `e40_qg_count.png`: panels match the JSON. Two cosmetic defects: in panel (b) the x-tick labels of the five right-hand bars overlap ("+3 part.0.5 GeV0.05 GeV"), and in panel (a) the legend marker of the unedited jet is the size of the leading particle. Panel (d) shows S1 (hatched) without CI and omits S30, which the reading names as a leading carrier.

## Procedure of the rerun
- Network: official JEDI-linear 3-feature n64, exact forward (jedi_core via e28w `Net`).
- **Fit sample**: `sample(1500, 4501)`, 7,500 jets, accuracy **81.21 %**. Its 3,000 q/g jets give the natural slope (OLS of margin on N).
- **Test sample**: `sample(1500, 4502)`, 7,500 jets, accuracy **81.07 %** (inside the 80.6–81.3 % band; this resolves E40 Issue 1). Edits on its 3,000 q/g jets. Edit randomness: rng seed 4540.
- Edits as in E40 (f = 0.5 splits only). **New control (E40 Issue 3)**: emission radius scan. 3 massless particles of 0.5 GeV at r = 0.02, 0.1, 0.15, 0.3, 0.5 from the jet axis (uniform random azimuth), and of 0.05 GeV at r = 0.02 and 0.3, at fixed jet four-momentum; compensation-only controls at r = 0.02 and 0.3. Also the soft-split response split by the parent particle's r.
- Why this control: S1 $=\sum_{\text{ranks}\ge1}(r-0.2)_+$ is independent of pT, and S30 $=\sum_{\text{slots}}\mathrm{ReLU}(3.47-21r)$ gives 3.38 per empty slot. Filling an empty slot with a particle at r changes S30 by $\mathrm{ReLU}(3.47-21r)-3.38$: −3.38 for r ≥ 0.165, −0.33 at r = 0.02. S19 $=\sum_{\text{ranks}\ge13}\mathrm{ReLU}(2.86-1.39\,p_T)$ changes by −1.39 pT per added soft particle, which goes to 0 as pT → 0. Predictions: H_count (the call counts every particle) → the emission response is roughly independent of r and survives pT → 0 at every r. H_wide (the count enters through S1 / S30) → in the soft limit the response is small for r ≲ 0.15 and large beyond.
- Statistics: 1,000 bootstrap resamples over jets; sign-flip permutation (5,000) for paired differences; exact McNemar on call changes; holds: share removed $=1-\sum\Delta m(\text{edit+hold})/\sum\Delta m(\text{edit})$, all 19 live S singles enumerated.

## Results
Natural slope: **0.138 / particle [0.135, 0.141]** pooled (E40 0.137), 0.101 within class (E40 0.098). Margin SD on q/g jets 2.63.

### Replication of E40 rows (q/g test jets, mean Δmargin [95 % CI], R_N = Δmargin / (slope × ΔN))
| edit | n | E40 | rerun (seed 4502) | R_N rerun |
|---|---|---|---|---|
| null re-sort | 3,000 | −0.0001 | −0.0001 [−0.0007, 0.0005] | – |
| split hardest f 0.5 | 2,125 | +0.282 | +0.287 [0.275, 0.300] | 2.08 |
| non-collinear, hardest | 2,125 | +0.188 | +0.197 [0.186, 0.209] | 1.43 |
| split rank 13 | 2,101 | +0.079 | +0.087 [0.079, 0.096] | 0.63 |
| rank-preserving, rank 13 | 2,101 | −0.008 | −0.002 [−0.009, 0.005] | −0.01 |
| split soft (rank ≥ 30) | 1,484 | +0.062 | +0.065 [0.056, 0.073] | 0.47 [0.41, 0.53] |
| non-collinear, soft | 1,484 | +0.064 | +0.069 [0.061, 0.078] | 0.50 |
| rotation, soft | 1,484 | +0.017 | +0.016 [0.009, 0.024] | – |
| emit 1 × 0.5 GeV, r 0.3 | 2,124 | +0.180 | +0.173 [0.164, 0.182] | 1.25 [1.19, 1.33] |
| emit 3 × 0.5 GeV, r 0.3 | 2,039 | +0.530 | +0.521 [0.505, 0.536] | 1.26 |
| emit 3 × 0.05 GeV, r 0.3 | 2,043 | +0.385 | +0.379 [0.366, 0.393] | 0.92 |
| merge 2 softest | 3,000 | −0.203 | −0.196 [−0.207, −0.186] | 1.42 |
| move 2 softest | 3,000 | −0.064 | −0.063 [−0.068, −0.058] | – |

Paired: soft split − non-collinear = −0.0046 [−0.0096, −0.0000], p 0.043 (E40: −0.002, p 0.24); combined, the collinear soft split is 3–7 % below the non-collinear one, not exactly equal. Soft split − rank-preserving −0.004 [−0.010, 0.002], p 0.23. Rank-13 split − rank-preserving +0.089 [0.080, 0.098], p 2e-4. Merge − move −0.133 [−0.145, −0.123], p 2e-4. 0.05 / 0.5 GeV ratio at r 0.3: 0.73 (E40 0.73).

Holds replicate (share removed): rank-13 split S19 **0.81 [0.75, 0.89]** (E40 0.86), S0 0.10; soft split S1 1.15, S30 0.80, S19 0.26, S29 −0.12; emit 3 × 0.5 GeV r 0.3 S1 0.50, S30 0.42, S19 0.17; merge S19 0.45, S30 0.33, S1 0.25.

### New: emission radius scan (3 particles, fixed jet four-momentum; figure panel a)
| r | 0.5 GeV Δmargin [CI] | R_N | 0.05 GeV Δmargin [CI] | R_N | largest holds (0.5 GeV) |
|---|---|---|---|---|---|
| 0.02 | +0.087 [0.081, 0.094] | 0.21 | **+0.0074 [0.0050, 0.0099]** | **0.02** | S19 0.85, S1 0.16, S30 0.03 |
| 0.1 | +0.205 [0.197, 0.214] | 0.50 | – | – | S30 0.42, S19 0.37, S1 0.03 |
| 0.15 | +0.298 [0.288, 0.310] | 0.72 | – | – | S30 0.46, S19 0.27 |
| 0.3 | +0.521 [0.505, 0.536] | 1.26 | +0.379 [0.366, 0.393] | 0.92 | S1 0.50, S30 0.42, S19 0.17 |
| 0.5 | +0.401 [0.383, 0.419] | 0.97 | – | – | S1 2.96, S30 0.68 |
Compensation only: r 0.02 −0.010 [−0.013, −0.007]; r 0.3 −0.033 [−0.039, −0.027].
- r 0.3 − r 0.02: +0.433 [0.419, 0.448] at 0.5 GeV and +0.372 [0.359, 0.386] at 0.05 GeV (p 2e-4 each; McNemar on call changes 154/10 and 145/3, p < 1e-33).
- Infrared ratio (0.05 GeV / 0.5 GeV response): 0.73 at r 0.3, **0.085 at r 0.02**. At r 0.02 the 0.5 GeV response runs through S19 (0.85), the hinge whose change is proportional to pT; at 0.05 GeV r 0.02 the residual is shared by S30 0.43 and S19 0.37.
- Calls changed by 3 × 0.05 GeV: 7.3 % at r 0.3 (89 q→g / 1 g→q), 0.39 % at r 0.02 (3 / 1).

### New: soft collinear split by the parent's radius
| parent r | n | split Δmargin [CI] | non-collinear | rank-preserving |
|---|---|---|---|---|
| < 0.165 | 717 | +0.049 [0.037, 0.062] | +0.054 | +0.054 |
| 0.165–0.2 | 132 | +0.058 [0.028, 0.086] | +0.066 | +0.074 |
| > 0.2 | 635 | +0.083 [0.070, 0.097] | +0.087 | +0.084 |
Collinear splits respond at every radius (R_N ≈ 0.36 near the core, 0.60 beyond 0.2).

## Assessment of the E40 reading
- Reproduced on a fresh sample within the CIs: every E40 response, the rank-13 rank-preserving null, the merge/move difference, the 0.73 infrared ratio at r 0.3, R_N ≈ 1.25–1.32 for one 0.5 GeV particle at r 0.3, and the hold pattern (S19 for rank-13, S1/S30 for soft splits and wide emissions, S0 and S29 ≤ 0.1).
- Not general as stated: "Adding one 0.5 GeV particle gives 1.32 times the average per-particle margin rise; with 0.05 GeV, 0.73 remains" holds at r = 0.3 only (the one radius E40 ran). At r = 0.02 the 0.5 GeV response is R_N 0.21 and the 0.05 GeV response is R_N 0.02, i.e. 0.085 of the 0.5 GeV response. The infrared-limit survival, and hence the soft (infrared) unsafety, applies to particles at r ≳ 0.15; near the axis the soft response goes to zero with pT, as an IR-safe quantity would. Collinear unsafety is present at all radii.
- "For splits of soft particles the response is equal for collinear and non-collinear": approximately; the rerun gives a 0.005 (7 %) shortfall at p 0.043.
- S19's per-particle change is −1.39 pT (it vanishes as pT → 0), so S19 is a soft-pT hinge with a rank window, not an IR-unsafe count; its rank-13 role is window crossing, as E40 states. The IR-unsafe count in the soft limit is carried by S1 (pT-independent, r > 0.2) and S30 (filled-slot term, r ≳ 0.1).
- Missing controls still open: emissions between r 0.3 and 0.5 (the response falls from 0.52 to 0.40; not explained), and holds of S1+S30 against size-matched pairs (S1+S30 removes 0.87 of emit r 0.3 and 1.90 of soft splits; pair null not enumerated here). Untrained controls were not rerun (E40: 5 seeds, 0 calls changed).

## Corrected reading
At fixed jet four-momentum, every edit that adds a particle moves JEDI-linear's g−q margin toward gluon (fresh sample: soft split +0.065, R_N 0.47; rank-13 split +0.087; hardest split +0.287; merge −0.196). Collinear splits respond at every radius. Soft emissions respond in proportion to their distance from the axis: 3 particles of 0.5 GeV give R_N 0.21 at r = 0.02, 0.50 at 0.1, 1.26 at 0.3. The infrared limit survives only away from the core: with 0.05 GeV, 0.73 of the 0.5 GeV response remains at r = 0.3 but 0.085 at r = 0.02 (R_N 0.02). H1′ (IRC-safe) is excluded for collinear splits and for soft emissions at r ≳ 0.15; near the axis the soft response behaves IR-safely. H1″ (rank only) describes the rank-13 split (rank-preserving −0.002, S19 0.81). The soft-limit count runs through S1 (pT-independent, r > 0.2) and S30 (filled-slot term, r ≳ 0.1); S19 changes by −1.39 pT per soft particle and carries the near-axis 0.5 GeV response (0.85) and the rank-13 crossing, not the soft-limit count. S0 and S29 carry ≤ 0.10.
