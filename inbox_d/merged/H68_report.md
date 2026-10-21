# H68 report: which particle-type channel the full-input JetClass taggers read (loop D, round D14)

Node H68 (exploratory). Claim: D10 v2 and a new D claim on particle-type composition. Script `interp/h68_particle_type_channel.py`.
Results: `results/physics/h68_jc.json` (JetClass relabel and ghost arms), `results/physics/h68_top.json` (Top Landscape clustered-ghost arm).
Figure: `results/physics/h68_particle_type.png`.
Jobs: `anrunw-ai-d-h68-run2` (JetClass, GPU, 948 s), `anrunw-ai-d-h68-run` (Top arm, 175 s; its JetClass part had a condition-naming bug and was replaced by run2), `anrunw-ai-d-h68-fig` (CPU).
Logs: `logs/d-h68-run.log`, `logs/d-h68-run2.log`.
Status: done.

## Conclusion

We changed the particle-type labels of JetClass jets (charged hadron, photon, neutral hadron) and left every particle's momentum and energy unchanged. ParticleNet and ParT with type inputs lowered the scores of all nine signal classes. The scores fell in proportion to the share of jet energy moved from charged hadrons to neutral hadrons, by about 0.5 log-odds per 10 % of jet energy for ParT. Calling the same particles photons instead of neutral hadrons gave a smaller drop. Swapping charge signs, with the type and the jet's total charge unchanged, also gave drops of up to 0.6 log-odds.

Physically, the networks treat the neutral-hadron label as evidence against a heavy-particle decay. They read more than the jet's charged-energy fraction. A classifier trained only on the energy fractions and shape variables predicts a much smaller response, and for four of nine classes a response of the opposite sign.

Still uncertain:
- whether the extra response reflects real correlations in Delphes jets that our reference cannot see (such as which particles are neutral, or where they sit), or sensitivity to label patterns that never occur in the data.

## Question

At fixed four-momenta, how do the full-input JetClass taggers respond when the type of particles changes? The types are charged hadron, photon and neutral hadron.

Two hypotheses:
- **H1 (composition reader):** the response follows the data's class-vs-QCD dependence on the charged-energy fraction f_ch and the neutral-hadron fraction f_nh. It then scales with Δf_ch, matches a fraction-aware reference classifier in sign per class, and the within-category control (c_same) and the kinematic jitter (c_kin) stay near 0.
- **H0 (label artefact):** the response concentrates on one label, the neutral-hadron flag, whatever the energy fraction. It does not follow the reference and also appears in weight-permuted ParT_full models.

H65v found that most of ParT_full's response to added soft particles comes from their neutral-hadron label. That is one instance of this question.

## Procedure

**Sample.** All 10 JetClass classes, from the first test file of each class (`test_20M/<sample>_100.root`). 1,000 random rows per class (numpy `default_rng(6801)`, never first-k), N = 10,000 jets.

**Reference rows.** `train_100M/<sample>_000.root`, 2,000 random rows per class (`default_rng(6802)`), disjoint from the test sample.

**Models.**
- ParT_full and PN_full (ParticleNet): released JetClass models; their inputs include charge, the five type flags and the impact-parameter fields.
- ParT_kin and PN_kin: kinematic inputs only, so no type input. Their response to every relabel must be exactly 0.
- Six ParT_full copies with every weight tensor's entries permuted (torch seeds 1–6): the label-artefact null.

**Definitions.**
- **f_ch** = energy of charged particles / summed particle energy. f_nh and f_γ are defined the same way for neutral hadrons and photons.
- **Score s** = logit_c − logit_QCD, evaluated on jets of true class c. For QCD jets, s = logit_QCD − logsumexp(the other nine logits).
- **Δs** = mean over the class's 1,000 jets of (edited score − unedited score), in natural-log odds.

**Delphes convention, checked on the sample.** Every real neutral particle (photon or neutral hadron) has charge = d0val = d0err = dzval = dzerr = 0 (maximum |value| 0.0). Every relabel to a neutral type sets these five fields to 0. Every edit keeps (px, py, pz, E) of every particle unchanged.

**Relabel edits.** The figure's left panel shows the fr10 set on a real H→bb jet.

| edit | what changes | particles changed per jet (mean) | Δf_ch (mean) |
|---|---|---|---|
| nh2ph | every neutral hadron → photon | 8.5 | 0 |
| ph2nh | every photon → neutral hadron | 32.1 | 0 |
| ch2nh2 | every charged hadron with pT < 2 GeV → neutral hadron (track fields removed) | 1.5 | −0.004 |
| ch2g2 | the same particles → photon | 1.5 | −0.004 |
| fr05 / fr10 / fr20 | charged hadrons taken in random order (rng [6801, 2, f]) until their energy is closest to 5 / 10 / 20 % of the jet energy, → neutral hadron | 2.6 / 4.5 / 7.9 | −0.045 / −0.097 / −0.199 |
| frg10 | the fr10 particles → photon | 4.5 | −0.097 |

**Controls**, on the fr10 set and the ch2nh2 set:
- **c_same (csame_*)**: a within-category relabel. JetClass has no π/K flag, so the only within-category label is the charge sign. Charges are flipped on equal numbers of + and − charged hadrons in the set. The type stays charged hadron, the track fields are unchanged and the net jet charge is unchanged. Flipped per jet: 3.0 of the 4.5 fr10 particles, 0.76 of the 1.5 ch2 particles.
- **c_kin (ckin_*)**: the same particles keep their labels. Their four-momenta are scaled by (1 + ε), ε ~ N(0, 0.005) per particle, and jet_pt / jet_energy are updated.
- **Kinematics-only models**: response must be 0.
- **Permuted ParT_full ×6**: the label-artefact null.

**References** (trained on the reference rows):
- **LR(f_ch), LR(f_nh)**: class-vs-QCD histogram likelihood ratios (20 bins on [0, 1], +1 smoothing).
- **GBDT_frac**: sklearn HistGradientBoosting on (m, τ21, τ32, N, f_ch, f_nh, f_γ). Here m is the invariant mass of the particles, τ_N are the stored N-subjettiness values, τ21 = τ2/τ1, τ32 = τ3/τ2, and N is the particle count.
- **GBDT_kin**: the same classifier without the three fractions.
- Relabels leave m, τ and N unchanged, so GBDT_kin's Δ is exactly 0, and GBDT_frac's Δ log-odds is the H1 reference response.

**Ghost arm (same sample).** Soft particles are added uniformly in the disc of radius 0.8 about the stored axis, then passed through `part_core.build`.
- **Composition:** the type is drawn with the proportions measured on the sample's real particles with pT < 2 GeV, leptons excluded: charged 23.2 %, photon 73.7 %, neutral hadron 3.1 %.
- **pT** is drawn from that type's measured spectrum below 2 GeV; the mean is 1.37 GeV.
- **Mass** is the type's median: π± 0.1396 GeV, photon 0, neutral hadron 0.
- **Charged ghosts** get charge ±1 and the (d0, dz) fields of a random real soft charged hadron.
- **Count:** Poisson with mean ρ·π·0.8²/1.37 GeV, where ρ = added pT per unit η–φ area. The sets are nested (one ρ = 20 master set, thinned).

Conditions:
- **gmix5/10/20:** composition ghosts at ρ = 5, 10, 20 GeV.
- **gnh10 / gph10 / gch10:** the same ρ = 10 ghosts all labelled neutral hadron (H65's label), photon, or charged hadron.
- **gstack10:** the ρ = 10 ghosts all 0.002 rad from the axis (H65's c2 control).
- **gclus10:** the physical replacement for the stack. The same ghosts are placed in groups of 5; each group sits at a random azimuth about the axis, and its members are at radius U[0.03, 0.1] rad, within ±0.3 rad in azimuth of the group. Count and summed pT are matched to gmix10.

At ρ = 10 the ghosts add 14.7 particles and 20.2 GeV of pT. The jet mass rises by 19.7 GeV (gmix10), 2.07 GeV (gstack10) and 2.36 GeV (gclus10).

**Top arm.** Top Landscape test, 1,000 top + 1,000 QCD jets (rng 6803), with H65's massless 0.5–1 GeV ghosts at ρ = 10:
- pu10: uniform in the disc;
- c2_10: on the axis;
- cl10: clustered as gclus10.

Models: ParT, PN, LorentzNet (LNet), PFN, EFN, and a permuted PFN. The score is the top-vs-QCD log-odds. LorentzNet has no JetClass model, so its clustered replacement is tested here.

**Statistics.**
- 95 % bootstrap intervals over jets (B = 1,000).
- p-values from the bootstrap standard error (normal approximation).
- Slope dΔs/dΔf_ch through the origin, pooled over fr05/10/20 per jet, with a bootstrap interval.
- McNemar exact test on 10-class correct→wrong vs wrong→correct changes.
- Holm correction over ParT_full and PN_full × 20 conditions × 10 classes (400 tests), separately for the Δs tests and the McNemar tests.

## Numbers

**Unedited 10-class accuracy on this sample** (known values from P1 in brackets):

| model | raw jets | after build (ghost-arm base) | P1 |
|---|---|---|---|
| ParT_full | 0.8606 | 0.8607 | 0.8593 |
| PN_full | 0.8415 | 0.8415 | 0.8457 |
| ParT_kin | 0.7395 | 0.7349 | 0.7447 |
| PN_kin | 0.6976 | 0.6975 | 0.7034 |

- Permuted ParT_full: 0.091–0.116 (near chance; each permuted model calls mostly one or two classes).
- ParT_full per-class accuracy: QCD 0.770, Hbb 0.921, Hcc 0.853, Hgg 0.806, H4q 0.853, Hqql 0.981, Zqq 0.682, Wqq 0.808, Tbqq 0.948, Tbl 0.984.
- The full 10×10 confusion matrices are in `h68_jc.json` under `confusion`.
- Top AUC: ParT 0.9878, PN 0.9872, LNet 0.9884, PFN 0.9823, EFN 0.9779 (H65 values 0.9858–0.9870, EFN 0.975–0.979).

**Kinematics-only models.** ParT_kin and PN_kin give Δs = 0.00 exactly for every relabel and every class. Under c_kin (0.5 % jitter) every model gives |Δs| ≤ 0.003.

**ParT_full Δs** [95 % CI], per class. Classes QCD / Hbb / Hcc / Hgg / H4q / Hqql / Zqq / Wqq / Tbqq / Tbl:

| edit | QCD | Hbb | Hcc | Hgg | H4q | Hqql | Zqq | Wqq | Tbqq | Tbl |
|---|---|---|---|---|---|---|---|---|---|---|
| ph2nh | +0.23 | −2.02 [−2.10, −1.95] | −2.30 | −0.97 | −2.55 | −2.50 | −2.11 | −2.42 | −2.13 | −0.85 |
| nh2ph | −0.46 | −0.48 [−0.56, −0.40] | −1.00 | −0.96 | −0.76 | −0.52 | −0.59 | −0.66 | −0.57 | −0.10 |
| ch2nh2 | +0.02 | −0.16 | −0.14 | −0.21 | −0.18 | −0.13 | −0.15 | −0.15 | −0.11 | −0.07 |
| ch2g2 | −0.06 | −0.01 | −0.01 | −0.04 | −0.01 | −0.03 | +0.01 | −0.00 | −0.04 | −0.04 |
| fr05 | −0.03 | −0.23 | −0.43 | −0.11 | −0.34 | −0.56 | −0.27 | −0.37 | −0.37 | −0.28 |
| fr10 | −0.03 | −0.46 [−0.52, −0.40] | −0.70 | −0.16 | −0.67 | −1.00 | −0.50 | −0.61 | −0.78 | −0.55 |
| fr20 | −0.15 | −0.99 [−1.08, −0.89] | −1.38 | −0.27 | −1.21 | −1.78 | −0.87 | −0.98 | −1.53 | −1.24 |
| frg10 | −0.14 | −0.18 [−0.24, −0.13] | −0.43 | −0.10 | −0.24 | −0.67 | −0.17 | −0.25 | −0.52 | −0.56 |
| csame_fr10 | −0.04 | −0.07 [−0.10, −0.04] | −0.47 | −0.05 | −0.46 | −0.61 | −0.17 | −0.37 | −0.42 | −0.12 |
| ckin_fr10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**PN_full fr10 Δs:** QCD +0.00, Hbb −0.43, Hcc −0.53, Hgg −0.11, H4q −0.39, Hqql −0.72, Zqq −0.37, Wqq −0.43, Tbqq −0.50, Tbl −0.13.
- frg10: −0.16, −0.13, −0.23, −0.10, +0.04, −0.41, +0.00, −0.05, −0.18, −0.54.
- ph2nh: +0.53, −1.69, −1.94, −0.72, −2.14, −1.25, −1.83, −2.14, −1.68, +0.04.
- csame_fr10: −0.02 to −0.35, and +0.10 for Tbl.

**Slope dΔs/dΔf_ch** (log-odds per unit f_ch, pooled over fr05/10/20; positive means the score falls as charged energy is removed):

| model | QCD | Hbb | Hcc | Hgg | H4q | Hqql | Zqq | Wqq | Tbqq | Tbl |
|---|---|---|---|---|---|---|---|---|---|---|
| ParT_full | 0.75 [0.51, 0.99] | 4.70 [4.24, 5.15] | 6.82 | 1.38 | 6.13 | 9.25 | 4.29 | 5.10 | 7.60 | 5.89 |
| PN_full | 0.26 | 4.61 | 5.49 | 0.94 | 3.98 | 6.97 | 3.28 | 3.91 | 5.34 | 1.58 |

ParT_kin and PN_kin slopes are 0.0. The fr05 : fr10 : fr20 responses are close to the 1 : 2 : 4 ratio of Δf_ch (for example, ParT_full Hbb −0.23 : −0.46 : −0.99).

**Neutral-hadron label vs photon label at the same particles** (ParT_full, paired difference, all p_Holm < 1e-20):
- fr10 − frg10: Hbb −0.27 [−0.31, −0.24], Hcc −0.27, Hqql −0.33, Wqq −0.36, Tbqq −0.27. QCD +0.11.
- ch2nh2 − ch2g2: −0.07 to −0.15 on signal classes.
- frg10 / fr10 ratio per class: 0.38 (Hbb), 0.61 (Hcc), 0.36 (H4q), 0.67 (Hqql), 0.34 (Zqq), 0.41 (Wqq), 0.67 (Tbqq), 1.02 (Tbl).

**References** (Δ log-odds of the reference under each edit, per class Hbb … Tbl):
- GBDT_frac AUC, with fractions vs without: Hbb 0.914 vs 0.905, Hqql 0.950 vs 0.931, Tbl 0.949 vs 0.931, other classes +0.002 to +0.008.
- Mean f_ch, QCD vs class: 0.563 vs 0.554–0.587 for the hadronic classes; 0.705 (Hqql) and 0.729 (Tbl).
- GBDT_frac under fr10: −0.07, +0.07, +0.25, −0.00, −0.96, +0.04, +0.11, −0.16, −0.79.
- LR(f_ch) under fr10: −0.17, −0.06, −0.15, −0.11, −0.45, −0.04, −0.05, −0.14, −0.56.
- LR(f_ch) under fr20: −0.60, −0.27, −0.76, −0.47, −0.91, −0.17, −0.11, −0.61, −1.07.
- GBDT_frac under ph2nh: −0.89, −0.18, +1.06, −0.91, −0.97, +0.03, −0.02, −4.13, −2.05. This edit takes f_nh outside the training range (8.5 → 32 neutral hadrons per jet), so the GBDT is extrapolating.

**Sign agreement with GBDT_frac** (out of 9 signal classes):

| model | fr10 | fr20 | frg10 | nh2ph | ph2nh | csame |
|---|---|---|---|---|---|---|
| ParT_full | 5 | 6 | 9 | 6 | 7 | 0 (reference Δ = 0) |
| PN_full | 5 | 6 | 7 | 5 | 6 | 0 (reference Δ = 0) |

Sign agreement with LR(f_ch) under fr10: 9 / 9. The ParT_full / LR(f_ch) magnitude ratio is 1.1 (Hgg) and 2.7–12.5 for the other hadronic classes, 2.2 for Hqql and 1.0 for Tbl.

**Permuted-ParT_full null** (6 seeds × 10 classes):

| edit | permuted mean \|Δs\| | permuted median \|Δs\| | ParT_full mean \|Δs\| |
|---|---|---|---|
| fr10 | 0.10 | 0.04 | 0.55 |
| fr20 | 0.16 | — | 1.04 |
| ph2nh | 0.41 | — | 1.81 |
| nh2ph | 0.12 | — | 0.61 |
| csame_fr10 | 0.015 | — | 0.28 |

The largest single permuted cell is 1.41 (ph2nh).

**Calls** (10-class argmax, ParT_full; right→wrong / wrong→right of 10,000 jets; accuracy after the edit):

| edit | right→wrong | wrong→right | accuracy after | classes with McNemar p_Holm < 0.05 |
|---|---|---|---|---|
| ph2nh | 974 | 271 | 0.790 | 9 |
| nh2ph | 817 | 241 | 0.803 | 4 |
| fr10 | 426 | 185 | 0.837 | 3 |
| fr20 | 730 | 238 | 0.811 | 6 |
| frg10 | 347 | 159 | 0.842 | — |
| csame_fr10 | 342 | 115 | 0.838 | 4 |
| ch2nh2 | 75 | 67 | — | 0 |

PN_full: fr10 386 / 178 (0.821); ph2nh 979 / 291 (0.773).

Holm: Δs family 312 / 400 with p_Holm < 0.05; McNemar family 159 / 400. Controls with p_Holm < 0.05:
- ckin: 0 / 20 cells per model.
- csame_fr10: ParT_full 9 / 10 classes, PN_full 7 / 10.
- csame_ch2: ParT_full 8 / 10 (|Δs| ≤ 0.05), PN_full 3 / 10.

**Ghost arm, ρ = 10, Wqq Δs** (gmix10 / gnh10 / gph10 / gch10 / gstack10 / gclus10):

| model | gmix10 | gnh10 | gph10 | gch10 | gstack10 | gclus10 |
|---|---|---|---|---|---|---|
| ParT_full | −2.78 [−2.85, −2.71] | −2.91 | −2.56 | −3.26 | −1.08 | −1.31 |
| PN_full | −2.98 | −3.09 | −2.78 | −3.23 | −1.11 | −1.36 |
| ParT_kin | −2.85 | −2.70 | −2.70 | −3.23 | −0.76 | −1.00 |
| PN_kin | −2.95 | −2.95 | −2.95 | −2.95 | −10.94 | −1.64 |

- **Labelled as neutral hadrons instead of photons** (gnh10 − gph10, ParT_full): Hbb −0.51 [−0.54, −0.49], Hcc −0.51, Wqq −0.35, Tbqq −0.30, Hqql −0.28; QCD +0.44.
- **Stacked share** gstack10 / gmix10 for ParT_full: Hbb 0.10, Hcc 0.20, H4q 0.30, Zqq 0.30, Wqq 0.39, Tbqq 0.80, Hqql 0.65, Tbl 1.29. H65v reported a Wqq share of 1.19 with neutral-hadron ghosts.
- **Density scan** (gmix5 / gmix10 / gmix20), ParT_full Wqq: −1.72 / −2.78 / −4.13.
- **PN_kin stack vs clusters:** gstack10 gives −10 to −23 (Tbqq −23.4), while gclus10 gives −0.3 to −2.3, at the gmix scale.
- **ParT_full stack vs clusters:** gclus10 − gstack10 = −0.01 (Hbb) to −0.23 (Wqq).

**Top arm** (top-jet Δs; QCD-jet Δs in brackets; Δm = +18.5 GeV for pu10, +2.25 for c2_10, +2.53 for cl10):

| model | pu10 | c2_10 | cl10 |
|---|---|---|---|
| LNet | −2.07 (+2.15) | −2.75 (−3.18) | −2.65 (+0.96) |
| PN | −2.07 (+1.65) | −0.21 (+3.88) | −1.26 (+1.90) |
| ParT | −1.61 (+0.45) | −1.40 (−0.08) | −1.45 (−0.03) |
| PFN | −1.80 | −1.41 | −1.46 |
| EFN | −0.91 | −0.11 | −0.13 |
| PFN permuted | +0.48 | +0.11 | +0.20 |

## Reading

**Predictions of H1 against the data.**
1. "Δs scales with Δf_ch": met for ParT_full and PN_full on nine signal classes. The response is close to linear (fr05 : fr10 : fr20 ≈ 1 : 2 : 4) with positive slopes of 4–9 log-odds per unit f_ch.
2. "Sign matches GBDT_frac per class": met for 5 of 9 classes under fr10. GBDT_frac predicts a rise for H→cc, H→gg, Z and W where the networks fall.
3. "c_same near 0": not met. Charge-sign swaps at fixed type, at fixed net charge and on fewer particles move ParT_full by −0.07 to −0.61, and 9 of 10 classes pass Holm.
4. "c_kin near 0": met, but the 0.5 % jitter gives no response in any model, kinematic-only ones included. So this control does not bound a kinematic response of comparable size.

**Predictions of H0 against the data.**
1. "Response concentrates on the neutral-hadron flag": partly met.
   - Soft charged hadrons below 2 GeV called photons move ParT_full by at most 0.06, and called neutral hadrons by 0.07–0.21.
   - At 10 % of jet energy, the photon relabel gives 0.34–1.02 of the neutral-hadron relabel, so a photon route of comparable size remains for the harder particles.
   - Relabelling all photons as neutral hadrons at fixed f_ch gives the largest response (−0.85 to −2.55).
2. "Response appears in the permuted ParT_full": not met at the same size. The permuted mean |Δs| is 0.10 against 0.55 for fr10, and 0.015 against 0.28 for csame.

Neither hypothesis's full set of predictions holds:
- The response scales with the moved energy fraction and is about 5× the permuted null, as H1 predicts.
- Its size per class is 2.7–12.5× that of LR(f_ch) on the hadronic classes other than H→gg (1.1×), and its sign disagrees with the fraction-aware GBDT for 4 of 9 classes.
- It depends on which neutral label is used and on the charge signs, quantities that f_ch and f_nh do not contain.

The relabel edits keep the four-momenta but produce type patterns that may be rare in the data, for example a 10 GeV charged pion relabelled as a neutral hadron with no track. The networks' extra response can therefore be either a use of real type–kinematics correlations that the 7-variable GBDT does not see, or an off-distribution response. This test does not separate the two.

**Ghost arm.**
- With the JetClass soft composition (74 % photons), ParT_kin, which sees no labels, gives a response to added soft particles of the same size as ParT_full (within 0.2 log-odds on signal classes; Wqq gmix10 −2.85 vs −2.78).
- The label modulates the response by up to 0.5 log-odds: neutral hadron vs photon −0.28 to −0.51 on signal classes, and charged lowest.
- The H65/H65v stacked share of 1.19 for Wqq is specific to the all-neutral-hadron label. With the measured composition the Wqq share is 0.39.
- For PN_kin, the clustered replacement gives Δs on the gmix scale where the on-axis stack gave −10 to −23. The stack is therefore an artefact for ParticleNet's k-nearest-neighbour graph, and gclus10 is a usable count-and-pT-matched low-mass control there.
- For LorentzNet on Top, the stack and the cluster give similar top-jet Δs (−2.75 vs −2.65), but on QCD jets they give opposite signs (−3.18 vs +0.96). So the stack is not a reliable control for LorentzNet either.

## Checks and caveats

- **Unedited accuracies** reproduce P1 within 0.6 points for all four models, on raw and built jets. Top AUCs are within 0.003 of H65.
- **Kinematics-only models** respond exactly 0 to every relabel, which confirms that the edits do not touch the four-momentum inputs.
- **Debug attempt 1.** In the first run the edit "g2nh" was classified as a ghost condition because of its leading "g", so it was compared against the built base. It was renamed ph2nh / nh2ph and rerun in `d-h68-run2`; all JetClass numbers above are from run2.
- **c_same.** The only within-category relabel available is the charge sign. Its large response means it is not a null for these networks. It also measures the response to rearranging particle charges at fixed net charge, which links to D16 (charge in quark–gluon jets).
- **c_kin** at 0.5 % is too small to give any response, so it does not bound kinematic sensitivity at the size of the type edits.
- **GBDT_frac under ph2nh and nh2ph** extrapolates beyond the training range of f_nh and f_γ. Its sign under fr10 is the cleaner comparison.
- **Ghosts** take their track fields from real prompt soft tracks, so they are not displaced like real pileup tracks in dz. JetClass has no pileup. This is a composition-matched toy, not a pileup simulation.
- **Cluster placement.** gclus10 keeps the clusters within 0.1 rad of the axis, so it is a low-mass, count-matched control. It is not a test of pileup clusters at wide angle.
- **Sample size.** One sample of 1,000 jets per class, with no verifier rerun yet.

## Explainer entry (for site/explain.py, Paper D tab)

**Question.** Do the full-input JetClass taggers read the particle-type composition of a jet (charged hadrons, photons, neutral hadrons) as a physical quantity, i.e. its charged-energy fraction? Or do they respond to the type labels in a way the composition does not explain?

**Physics background.** In a jet, about 60 % of the energy is carried by charged hadrons (seen as tracks), about 25 % by photons (mostly from π⁰ → γγ) and about 15 % by neutral hadrons (K_L, neutrons). Leptonic decays (H→ℓνqq′, t→bℓν) have a higher charged fraction because of the lepton. Hadronic decays differ from QCD jets only slightly in f_ch (0.55–0.59 vs 0.56). The class-vs-QCD log-likelihood ratio of f_ch is
$$\mathrm{LR}(f_{ch}) = \log \frac{p(f_{ch}\mid c)}{p(f_{ch}\mid \mathrm{QCD})}, \qquad f_{ch} = \frac{\sum_{i \in \text{charged}} E_i}{\sum_i E_i}.$$

**Method.** Each particle keeps its four-momentum $(p_x, p_y, p_z, E)$, and only its type flags change. Neutral particles get charge 0 and zero impact parameters, as Delphes stores them. In the energy-fraction-matched edit, charged hadrons are taken in random order until their energy reaches $f \cdot E_{\rm jet}$ ($f$ = 5, 10, 20 %), and they are relabelled as neutral hadrons (or photons). The network response is
$$\Delta s = \big\langle (\ell_c - \ell_{\rm QCD})_{\rm edited} - (\ell_c - \ell_{\rm QCD})_{\rm original} \big\rangle_{\text{jets of class } c},$$
and the slope is $d\Delta s / d\Delta f_{ch}$. The H1 reference is a gradient-boosted classifier on $(m, \tau_{21}, \tau_{32}, N, f_{ch}, f_{nh}, f_\gamma)$, evaluated before and after the same edit.

**Result (figure `h68_particle_type.png`).**
- ParT_full's own-class score falls on all nine signal classes by about 0.5 log-odds per 10 % of jet energy moved from charged to neutral hadrons, linearly in the moved fraction.
- Calling the same particles photons gives 0.34–1.0 of that drop.
- Calling every photon a neutral hadron, at unchanged charged fraction, gives −0.85 to −2.55.

**Checks.**
- Kinematics-only models respond exactly 0.
- Weight-permuted models respond about 5× less.
- A charge-sign swap at fixed type responds up to −0.6.
- The fraction-aware classifier agrees in sign for 5 of 9 classes, and the f_ch likelihood ratio is 1.1–12.5× smaller on hadronic classes.

**Physical significance.** The type-aware taggers lower signal scores when neutral hadrons replace charged hadrons or photons. They do so more strongly than the jets' measured charged-energy fraction warrants, and they also respond to charge signs. So they use type information beyond the composition fractions. Whether this reflects real correlations in Delphes jets or a response to label patterns absent from the data is not settled by this test.

**New glossary terms (explain.GLOSSARY):**
- **f_ch (charged-energy fraction)**: the share of the jet's summed particle energy carried by charged particles.
- **f_nh, f_γ**: the same shares for neutral hadrons and photons.
- **Relabel edit**: a change of a particle's type flags (and of the track fields that follow from its type) at fixed four-momentum.
- **c_same**: a within-category relabel control; here, swapping charge signs of equal numbers of + and − charged hadrons.
- **Composition ghosts**: added soft particles whose types, pT and masses are drawn from the measured soft-particle population of the dataset.
- **Clustered ghosts (gclus)**: added soft particles placed in groups of 5 within 0.03–0.1 rad of the jet axis, the count- and pT-matched low-mass control that replaces the on-axis stack.
