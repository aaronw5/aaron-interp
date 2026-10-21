# H55: classical-tagger fingerprint panel across hls4ml, JetClass, Top Landscape and quark/gluon (exploratory)

## Conclusion
The same nine edits were applied to jets from all four datasets. On every two-prong and three-prong task, every trained network's pattern of call changes is closer to the classical N-subjettiness / energy-correlation taggers (m_SD + τ21, m_SD + τ32, m_SD + D2) than to a plain jet-mass window. Here a call is the network's yes/no decision at its 50 % signal-efficiency threshold. All trained networks also change 2–5 % of their calls when one particle is split into two collinear halves, an edit under which every IRC-safe classical tagger changes 0 calls. They change 8–20 % under a shift of the axis the inputs are measured about, which no axis-free classical tagger responds to. The pre-registered nearest-tagger pattern holds only in part. JEDI-linear's nearest tagger is m_SD + τ21-DDT for W and the plain mass window for Z, not the mass window throughout. The quark/gluon networks other than the EFN are nearest the count-based LR(N, pT^D), not the IRC-safe girth tagger. The similarities are moderate (cosine 0.3–0.6 on two-prong and three-prong tasks). After Holm correction they exceed the weight-permuted null only for the JetClass and hls4ml W/Z cells and a few Top/QG cells, so the ranking of taggers is a description, not a strong test.

## Procedure
Script `interp/h55_classical_fingerprint.py`. Cluster jobs: d-h55-edit (conda, edits and observables), d-h55-jedi (jax), d-h55-small (conda), d-h55-gpu (JetClass, Top, QG networks), d-h55-stats2 (statistics and figures). Results are in `results/physics/h55_fingerprint.json`, `h55_heatmap.png` and `h55_edits.png`. Per-dataset tables and logits are on the volume only (`/j-jepa-vol/aaron-interp/results/physics/h55_*`).

**Samples** (random rows only):
- **hls4ml:** `jedi_core.sample(2000, 5501)`, 10,000 jets (g, q, W, Z, t). 64 slots, massless particles, stored axis (0, 0).
- **JetClass:** the first test file of Hbb, Wqq, Zqq, Tbqq and QCD, 2,000 random rows each (rng 5502). 128 slots, stored axis jet_eta / jet_phi, with η = jet_eta + sign(jet_eta)·Δη.
- **Top Landscape:** test split, 2,000 top + 2,000 QCD (rng 5503).
- **QG (Pythia):** test split, 2,000 quark + 2,000 gluon (rng 5504).
- For Top and QG the axis is the E-scheme sum of the particles, recomputed after each particle edit, as in the conversion the networks were trained on.

**Edits.** Each edit produces a new particle list. Particles are re-sorted by pT and cut to the slot limit. Metrics use only the jets the edit applies to ("ok" jets).
| edit | what changes | what is held fixed |
|---|---|---|
| split (H1) | hardest particle replaced by two halves (pT/2, mass/2 each) in the same direction; daughters keep PID / charge / impact parameters | four-momentum exactly; every IRC-safe observable; N → N+1 |
| ksplit3 (H18v, δ = 0) | 3 random particles of the softer half each split into two collinear halves | as split; N → N+3 (jets with < 6 particles not ok) |
| soft (H1/H5) | one massless neutral particle, pT = 0.1 % of jet pT, uniformly in the annulus R/2 < ΔR < R about the axis (R = 0.8; QG 0.4); its 3-momentum is subtracted from the hardest particle | jet 3-momentum; mean Δm = +0.9 GeV (JetClass), +3.3 GeV (hls4ml) |
| wta (H36) | inputs (and girth) measured about the winner-take-all axis (C/A with WTA recombination = one particle's direction) instead of the stored / E-scheme axis | every particle; median axis offset 0.040 (hls4ml), 0.085 (JetClass), 0.093 (Top), 0.017 (QG) |
| ctrl | axis moved by the same per-jet offset length in a uniformly random direction (equivalent to a rigid translation of the inputs) | every particle; control for wta |
| gin (H49) | 3 massless neutral particles, 2 % of jet pT in total, placed beyond the farther soft-drop prong so that C/A attaches them to that prong; soft drop keeps them | matched to gout in ungroomed m (|Δ| < 1 GeV); Δm_SD(gin) ≥ 0.5 Δm |
| gout (H49) | the same three particles away from both prongs; soft drop removes them (|Δm_SD| < 0.5 GeV) | same added count, pT and ungroomed Δm as gin (mean +8 GeV in JetClass) |
| decay (H45 D(+0.3)) | two exclusive-kT prongs; every particle boosted to the jet rest frame, rotated so that \|cos θ*\| of the prong axis rises by 0.3 (more asymmetric momentum sharing), boosted back | jet four-momentum, jet mass (max \|Δm\| < 0.01 GeV), N, all pairwise masses exactly; ok if both prong centres stay within 0.75 (QG 0.35) of the axis |
| zboost (new) | Lorentz boost along the beam axis, rapidity shift 0.5·sign(η_jet); stored axis moved with it | m, pT, Δy, Δφ exactly; particle energies and pseudorapidities change. hls4ml inputs (pT, Δη, Δφ of massless particles) are invariant by construction |

Unlike H12, which boosted along the jet momentum, zboost is a beam-axis ("longitudinal" in collider coordinates) boost. In hls4ml, split, ksplit3 and soft are not ok for the jets whose softest real particle would leave the 64-slot window. These are 22 %, 25 % and 22 % of hls4ml jets, so for those edits the hls4ml sample is biased toward lower-multiplicity jets. In the other datasets no jets overflowed. gin / gout are matched sets: 61 % (hls4ml), 73 % (JetClass) and 63 % (Top) of jets. They are not applied to QG, because the H49 geometry places particles at up to ΔR 0.78 and QG jets have R = 0.4. decay ok fractions: 99.6 % (hls4ml), 87.6 % (JetClass), 86.6 % (Top), 90.0 % (QG).

**Classical taggers.** These are computed on every edited jet. Observable definitions:
- **m:** invariant mass of the particles.
- **m_SD:** soft-drop mass (C/A clustering, β = 0, z_cut = 0.1).
- **τ21, τ32:** N-subjettiness ratios τ2/τ1 and τ3/τ2, with exclusive-kT axes and β = 1.
- **D2:** energy-correlation ratio e3/e2³ with β = 1.
- **N:** particle multiplicity.
- **pT^D:** √(Σ pT²) / Σ pT.
- **girth:** Σ pT ΔR / Σ pT about the axis in use.

The taggers and their working points:
- **V_τ21:** 65 < m_SD < 105 GeV and τ21 < 0.45. CMS 2016 W/Z high-purity working point (JME-16-003).
- **V_DDT:** the same mass window and τ21 + 0.063 ρ′ < 0.55, with ρ′ = ln(m_SD² / (pT·1 GeV)). CMS designed-decorrelated-tagger (DDT) form, EXO-17-001. The slope and cut are taken from CMS and not re-optimised.
- **V_D2:** the same window and D2 below the median D2 of the task's in-window unedited signal jets (cut 1.05 for hls4ml W, 1.34 for JetClass W). This gives 50 % in-window signal efficiency; no published value exists for these samples.
- **mass_V:** 65 < m < 105 GeV, ungroomed.
- **top_τ32:** 105 < m_SD < 210 GeV and τ32 < 0.54 (CMS 2016, as in H49).
- **mass_top:** 105 < m < 210 GeV.
- **LR(N), LR(N, pT^D), LR(girth):** histogram likelihood ratios for quark vs gluon, 2-fold cross-fit on the unedited sample. A jet is called quark above the threshold giving 50 % quark efficiency.

**Networks.** Each task score is the log-odds log p(sig) − log Σ p(bkg). A network calls a jet signal when this score exceeds the median score of unedited signal jets (50 % signal efficiency).
- **hls4ml:** JEDI-linear official n64 3-feature, Deep Sets d64, Linformer 2b-d32-k8, Transformer 2b-d64-e100, JEDI-style float d64.
- **JetClass:** ParT_full, ParT_kin, PN_full, PN_kin.
- **Top:** ParT_kin, PN_kin (H2 weaver), PFN, EFN (H2 energyflow), LorentzNet (released).
- **QG:** part_kin, pnet_kin (H5), pfn_k, efn_k (H2 energyflow, ported), lorentznet (released).
- **Nulls:** 6 weight-permuted copies per architecture: every weight tensor's entries shuffled, seeds 1–6, or 11–16 for JEDI. Permuted copies were made for all hls4ml architectures, ParT_full and PN_full (JetClass), ParT, PN, PFN, EFN (Top), and part, pnet, pfn, efn (QG). Totals: 30 (hls4ml), 12 (JetClass), 24 (Top), 24 (QG).

**Metrics.** These are computed per row (network, permuted network or classical tagger), per binary task, and per edit, on the task's signal and background jets where the edit is ok:
- **chg:** fraction of calls changed.
- **flow:** (calls gained − calls lost) / jets.
- **Networks only:**
  - Δlog-odds (edit − unedited), with a 95 % bootstrap CI.
  - Exact McNemar test on gained vs lost calls (flow family).
  - For wta, McNemar on the changed indicator wta vs ctrl.
- **Fingerprint:** the vector (chg_e, flow_e) over the dataset's edits. Each component is divided by its RMS over the trained and classical rows of the task.
- **Similarity:** cosine similarity between a network's fingerprint and a classical tagger's, with a 95 % bootstrap CI over jets (B = 1000; the standardisation is recomputed in each draw).
- **Hypotheses:** H0: cos(net, tagger) ≤ the mean cosine of the permuted networks to that tagger. H1: it is larger.
- **Test:** z = Δ/√(sd_boot² + sd_perm²), where Δ = cos(net, tagger) − mean_perm cos(perm, tagger), sd_boot is the bootstrap spread of the network's cosine and sd_perm is the spread across permuted networks. One-sided p = 1 − Φ(z), Holm-corrected over all 130 network × tagger cells. The bootstrap p = P(Δ_b ≤ 0) is also stored, but its floor of 1/1001 gives Holm ≥ 0.13 for 130 tests, so the z test is used for Holm.

## Unedited models on these samples (reproduction check)
- **JEDI-linear:** 5-class accuracy 0.8092 (known 0.8092).
- **Other hls4ml networks:** Deep Sets 0.8082 (known test 0.806), Linformer 0.8133 (0.811), Transformer 0.8162, JEDI-style float 0.8006.
- **JetClass per-class accuracy (10-class argmax):**
  - ParT_full: QCD 0.776, Zqq 0.679, Wqq 0.790, Tbqq 0.952, Hbb 0.927 (known 0.783 / 0.689 / 0.805 / 0.956).
  - ParT_kin: 0.731 / 0.558 / 0.684 / 0.897.
  - PN_full: 0.765 / 0.649 / 0.773 / 0.951.
  - PN_kin: 0.705 / 0.526 / 0.669 / 0.898.
  - All are within 0.02 of the H18 / H49 values.
- **Top accuracy (AUC):** ParT 0.941 (0.985), PN 0.935 (0.986), PFN 0.923 (0.979), EFN 0.919 (0.976), LorentzNet 0.942 (0.987). Known: 0.940, 0.940, 0.928, 0.922.
- **QG accuracy (AUC):** part_kin 0.827 (0.909), pnet_kin 0.833 (0.909), pfn_k 0.818 (0.898), efn_k 0.808 (0.888), lorentznet 0.851 (0.923). Known part_kin: 0.827.
- **Permuted nulls:** accuracy 0.18–0.25 (hls4ml), 0.00–0.24 (JetClass), 0.45–0.66 (Top / QG).

**Classical taggers on unedited jets (signal efficiency / background efficiency):**
| tagger | hls4ml W vs q+g | JetClass W vs QCD | JetClass Z vs QCD |
|---|---|---|---|
| V_τ21 | 0.674 / 0.054 | 0.578 / 0.040 | 0.560 / 0.040 |
| V_DDT | 0.662 / 0.049 | 0.517 / 0.025 | 0.484 / 0.025 |
| V_D2 | 0.384 / 0.006 | 0.354 / 0.011 | 0.350 / 0.013 |
| mass_V | 0.923 / 0.263 | 0.843 / 0.251 | 0.797 / 0.251 |

| tagger | hls4ml t | JetClass top | Top Landscape | tagger | hls4ml q vs g | QG |
|---|---|---|---|---|---|---|
| top_τ32 | 0.406 / 0.018 | 0.497 / 0.013 | 0.448 / 0.011 | LR(N) | 0.498 / 0.042 | 0.496 / 0.064 |
| mass_top | 0.774 / 0.175 | 0.916 / 0.246 | 0.931 / 0.312 | LR(N, pT^D) | 0.502 / 0.044 | 0.500 / 0.062 |
| | | | | LR(girth) | 0.495 / 0.152 | 0.500 / 0.148 |

## Checks of the predictions written before running
- **IRC-safe classical taggers do not respond to split or ksplit3.** Confirmed: 0.0 % calls changed for V_τ21, V_DDT, V_D2, mass windows, top_τ32 and LR(girth) on every dataset. The observable changes are exactly 0 (max |Δ| = 0 for m, m_SD, τ21, τ32, D2, girth).
- **LR(N) responds to split but not to wta.** Confirmed: split changes 4.4 % of calls on both hls4ml and QG, ksplit3 changes 6.9 % and 10.2 %, and wta changes 0.0 %.
- **The mass window does not respond to decay.** Confirmed: 0.0 % on every dataset (max |Δm| < 0.01 GeV).
- **Exact by-construction checks:**
  - hls4ml networks under zboost: 0.0 %.
  - EFN under split and ksplit3: 0.0 % (Top and QG; IRC-safe by construction).
  - LorentzNet under wta and ctrl: 0.0 % (four-vector input, no axis).
  - Classical taggers under zboost: 0.0 %, except LR(girth) on QG at 0.9 %. Girth there is measured about the recomputed E-scheme axis, whose pseudorapidity does not shift by exactly 0.5 for a massive jet.

## Numbers (% of calls changed at 50 % signal efficiency; full table in the heat map and JSON)
| task | row | split | ksplit3 | soft | wta | ctrl | gin | gout | decay | zboost |
|---|---|---|---|---|---|---|---|---|---|---|
| hls4ml W | JEDI-linear | 3.4 | 2.7 | 6.5 | 15.8 | 16.2 | 25.9 | 25.9 | 8.7 | 0.0 |
| hls4ml W | V_τ21 / mass_V | 0 / 0 | 0 / 0 | 0.3 / 3.0 | 0 / 0 | 0 / 0 | 3.2 / 5.1 | 3.4 / 5.1 | 11.9 / 0 | 0 / 0 |
| JetClass W | ParT_full | 2.8 | 3.5 | 2.3 | 9.6 | 12.5 | 19.2 | 15.7 | 14.6 | 3.9 |
| JetClass W | PN_kin | 1.5 | 3.8 | 2.8 | 7.9 | 8.1 | 24.2 | 22.4 | 12.8 | 4.4 |
| JetClass W | V_τ21 / V_D2 / mass_V | 0 | 0 | 0.4 / 1.6 / 1.2 | 0 | 0 | 4.5 / 4.3 / 6.2 | 3.9 / 7.3 / 6.2 | 14.4 / 12.4 / 0 | 0 |
| Top | ParT_kin | 1.5 | 1.2 | 0.8 | 10.7 | 12.0 | 6.7 | 3.6 | 8.8 | 2.6 |
| Top | EFN | 0.0 | 0.0 | 0.7 | 20.2 | 20.2 | 6.9 | 10.3 | 9.7 | 2.1 |
| Top | top_τ32 / mass_top | 0 | 0 | 0.4 / 0.5 | 0 | 0 | 3.7 / 4.0 | 5.2 / 4.0 | 8.4 / 0 | 0 |
| QG | part_kin | 2.5 | 1.2 | 1.9 | 1.1 | 1.3 | – | – | 7.1 | 1.4 |
| QG | LR(N) / LR(N,pT^D) / LR(girth) | 4.4 / 7.5 / 0 | 10.2 / 6.2 / 0 | 4.4 / 2.3 / 3.0 | 0 / 0 / 12.1 | 0 / 0 / 10.7 | – | – | 0 / 1.8 / 15.9 | 0 / 0 / 0.9 |

**Direction of the call change** (mean Δlog-odds of the task score, edit − unedited; on all task jets):
- **Count edits** (split, ksplit3, soft):
  - JetClass W/Z networks: toward signal (e.g. ParT_full W vs QCD, split: Δlog-odds − Δlog-odds(ctrl) = +0.52).
  - QG networks: toward gluon (part_kin, split: −0.23).
  - Top PFN / EFN: toward QCD.
  - Top ParT / PN: toward top.
- **gin vs gout:**
  - JetClass ParT_full W: gin changes 19.2 % of calls, gout 15.7 %.
  - Top ParT_kin: gin 6.7 %, gout 3.6 %.
  - hls4ml networks respond equally (25.9 % vs 25.9 %). This is consistent with H49: the hls4ml networks follow ungroomed mass, while JetClass and Top networks respond more to the particles soft drop keeps.
- **decay:** moves every JetClass W/Z network toward QCD (Δlog-odds −0.7 to −1.0 relative to ctrl).
- **zboost:** changes 2.5–4.9 % of JetClass network calls and 0.9–2.7 % of Top / QG network calls. The networks read particle energies (log E, log E/E_jet) in addition to pT, so a boost that leaves every Lorentz-invariant and every Δy, Δφ unchanged still moves their output.
- **wta vs ctrl** (McNemar on the changed indicator): 14 of 46 network × task cells differ after Holm. In the other cells the WTA axis change and a random axis shift of the same size change a similar fraction of calls (e.g. JEDI W 15.8 % vs 16.2 %). This is the same pattern as H36 / D4m.
- **Flow McNemar** (gained vs lost calls, H0: no shift of the call rate): 312 of 404 network × task × edit tests are below 0.05 after Holm.

**Cosine similarity to classical taggers** (95 % bootstrap CI; permuted-network mean; Holm-corrected z-test p):
| dataset task | network | nearest tagger (bootstrap frequency) | cos to nearest | cos to plain mass window | Holm p (nearest) |
|---|---|---|---|---|---|
| hls4ml W | JEDI-linear | V_DDT (0.57) | 0.35 [0.33, 0.37]; perm 0.13 | 0.20 [0.15, 0.25]; perm 0.12 | 1.0 (z p 0.015) |
| hls4ml Z | JEDI-linear | mass_V (0.62) | 0.43 [0.38, 0.47]; perm 0.11 | (nearest) | 0.018 |
| hls4ml t | JEDI-linear | mass_top (0.66) | 0.28 [0.20, 0.35]; perm 0.10 | (nearest) | 1.0 |
| hls4ml q/g | JEDI-linear | LR(N, pT^D) (0.88) | 0.84 [0.80, 0.87]; perm 0.28 | – | 0.47 |
| JetClass W | ParT_full | V_D2 (1.00) | 0.52 [0.48, 0.55]; perm 0.26 | 0.36 [0.28, 0.43]; perm 0.37 | 0.49 (V_τ21: 0.40, Holm 1.6e-6) |
| JetClass Z | PN_full | V_D2 (1.00) | 0.59 [0.56, 0.61]; perm 0.24 | 0.49 [0.46, 0.52]; perm 0.23 | 0.010 |
| JetClass top | ParT_full | top_τ32 (1.00) | 0.51 [0.45, 0.56]; perm 0.32 | 0.32 [0.24, 0.38]; perm 0.33 | 1.0 |
| Top | EFN | top_τ32 (1.00) | 0.76 [0.70, 0.78]; perm 0.24 | 0.35 [0.20, 0.51]; perm 0.28 | 0.020 |
| Top | ParT_kin | top_τ32 (1.00) | 0.49 [0.44, 0.56]; perm 0.24 | 0.28 [0.19, 0.38]; perm 0.28 | 1.0 |
| QG | part_kin | LR(N, pT^D) (0.99) | 0.43 [0.37, 0.49]; perm 0.16 | – (LR girth 0.27; perm 0.48) | 1.0 |
| QG | pfn_k | LR(N, pT^D) (0.65) | 0.68 [0.61, 0.72]; perm 0.16 | – | 0.037 |
| QG | efn_k | LR(girth) (1.00) | 0.33 [0.29, 0.39]; perm 0.48 | – | 1.0 |

**Holm results across the 130 cells:** 36 have p < 0.05. All of them are in:
- JetClass W/Z (all four networks to V_τ21 and V_DDT; Z also to V_D2 and mass_V);
- hls4ml Z (Deep Sets, Linformer, JEDI-style float to V_τ21 / V_DDT / mass_V; JEDI-linear to mass_V);
- Top EFN to top_τ32;
- QG pfn_k to LR(N, pT^D).

No hls4ml top, hls4ml q/g or JetClass-top cell passes. The plain mass window never exceeds the permuted null on W (JetClass) or top (any dataset).

## Reading
- **What the fingerprint compares.** Weight-permuted networks change 10–25 % of their calls under most edits, including the count edits. Generic input sensitivity therefore accounts for much of the size of each response. The fingerprint comparison uses the shape of the response vector, not its size.
- **Two-prong and three-prong tasks.** Trained networks share three features of the classical substructure taggers:
  1. a large decay-axis response, which is 0 for a mass window;
  2. a gin ≠ gout asymmetry (JetClass, Top only);
  3. a small soft-particle response.
  They differ from all classical taggers in two respects:
  1. a nonzero exact-collinear-split response (1–5 %), with a definite direction set by the dataset;
  2. an axis response of 8–20 %, which is about the same for the WTA axis as for a random shift of the same size.
- **JEDI-linear.** Its fingerprint is no closer to the plain-mass window than the other hls4ml networks'; all five hls4ml networks have nearly the same fingerprint. Together with A4 and D1, this means the edit library does not separate a "mass window" from a "mass + shape" reader for hls4ml. The hls4ml networks' gin = gout response (they ignore grooming) and their decay response of 8–11 % both depart from a pure mass window, whose decay response is 0.
- **QG networks.** ParT, PN, PFN and LorentzNet change calls under collinear splits in the gluon direction, which is the LR(N) direction. The fraction changed is 1–4 %, compared with 4–10 % for LR(N). This agrees with the H18v finding that QG networks use particle count weakly. Their nearest classical tagger is the count-based LR(N, pT^D), not girth. The EFN, which is IRC-safe, matches neither: it has no split response but a large axis response.
- **Cosine scale.** Cosines are moderate (≤ 0.84) because every network has axis and boost components that no classical tagger in the panel has. The cosine therefore ranks taggers but does not indicate equivalence.

## Issues and limitations
- **Working points.** Classical taggers use published cuts (signal efficiency 0.35–0.93), while networks use 50 % signal efficiency. The fraction of calls changed depends on how many jets lie near the threshold, so chg magnitudes are not directly comparable across rows. The fingerprint standardises columns but not rows.
- **V_D2 CI.** The bootstrap CI of V_D2 cosines is asymmetric (e.g. 0.28 [0.26, 0.61]). Some columns (gin/gout, soft) have small RMS on hls4ml, so resampling changes the scaling.
- **Permuted nulls cover a subset of architectures.** On JetClass the permuted rows are only ParT_full and PN_full copies; there are no permuted LorentzNet copies. The z-test assumes approximately normal Δ.
- **Edit coverage.**
  - zboost is a beam-axis boost. The jet-axis boost of H12 was not repeated.
  - gin/gout are not defined for QG (R = 0.4).
  - The decay edit uses kT-2 prongs for every class, including top and QCD, where "decay axis" is only a geometric definition.
- **Sample biases.** hls4ml count edits exclude 22–25 % of jets (those with 64 particles). V_DDT uses CMS constants taken from a different jet algorithm and pT range.
- **Hbb.** Hbb vs QCD has no classical tagger in the panel; only network responses are stored.

## Explainer entry (for site/explain.py, H55_EXPLAINER)
1. **Question.** Which classical jet tagger does each trained network respond like, when both are given the same physics edits?
2. **Physics background.**
   - Classical taggers combine IRC-safe substructure observables: groomed mass $m_{SD}$, the N-subjettiness ratios $\tau_{21}=\tau_2/\tau_1$ and $\tau_{32}$, and the energy-correlation ratio $D_2=e_3/e_2^3$. These are unchanged by a collinear split or an infinitely soft emission.
   - Quark/gluon taggers often use the multiplicity $N$ or $p_T^D=\sqrt{\sum p_{T,i}^2}/\sum p_{T,i}$, which are IRC-unsafe and do change under a split.
   - DDT is a designed decorrelated tagger: $\tau_{21}' = \tau_{21}+0.063\ln(m_{SD}^2/p_T)$.
   - A working point is the cut, or the score threshold, that defines a tagger's yes/no call.
3. **Method.**
   - Nine edits, each described in words in the report, are applied to 10,000 hls4ml, 10,000 JetClass, 4,000 Top and 4,000 QG jets.
   - Each edit holds something exactly fixed: the four-momentum (split, decay), the particles (wta, ctrl), the invariant mass and Δy, Δφ (zboost), or the ungroomed mass change (gin vs gout).
   - For each tagger and edit: fraction of calls changed and net call flow.
   - Fingerprint $f=(\text{chg}_e,\text{flow}_e)_e$, with each component divided by its RMS. Cosine similarity $\cos(f_a,f_b)=f_a\cdot f_b/|f_a||f_b|$ (1 = same response pattern, 0 = unrelated).
   - Null: the same networks with their weights permuted.
   - Statistics: $z=\Delta/\sqrt{\sigma_{boot}^2+\sigma_{perm}^2}$, Holm correction.
4. **Result (figures `h55_heatmap.png`, `h55_edits.png`).**
   - IRC-safe classical taggers: exactly 0 % of calls change under collinear splits.
   - Networks: 1–5 % change.
   - Shifting the input axis changes 8–20 % of network calls and 0 % of classical calls.
   - On W/Z/top tasks every network's nearest classical tagger is an $m_{SD}$ + shape tagger ($\tau_{21}$, $D_2$ or $\tau_{32}$), not the plain mass window. Holm-corrected evidence above the permuted null is strongest on JetClass W/Z.
   - QG networks are nearest LR(N, $p_T^D$).
5. **Checks.**
   - Predicted zeros all observed: IRC-safe taggers under splits; mass window under decay; LR(N) under wta; hls4ml networks under zboost; EFN under splits; LorentzNet under axis changes.
   - Unedited accuracies reproduce the known values to within 0.02.
6. **Physical significance.** Trained taggers respond like groomed-mass + prong-shape taggers. In addition, they read IRC-unsafe information (particle count, particle energies under a beam boost) and the position of the input axis, which no classical tagger in this panel uses.

Glossary additions:
- **DDT:** designed decorrelated tagger; τ21 shifted by a linear term in ln(m²/pT) so that the cut does not sculpt the mass distribution.
- **D2:** e3/e2³, a three-point over two-point energy-correlation ratio. It is small for two-prong jets.
- **m_SD:** soft-drop groomed mass (C/A declustering, removing branches with z < 0.1; β = 0).
- **Working point:** the cut or score threshold defining a tagger's yes/no call, quoted by its signal and background efficiency.
- **Cosine similarity:** f_a·f_b/(|f_a||f_b|) between two response vectors. 1 means the same pattern, 0 means unrelated.

## Proposed claim (exploratory)
D-new (0.5): across hls4ml, JetClass, Top and QG, trained taggers' call responses to a common edit library are closer to m_SD + τ21 / D2 / τ32 taggers than to a plain mass window on two-prong and three-prong tasks. This is significant above a weight-permuted null after Holm on the JetClass W/Z and hls4ml Z cells. The trained taggers also respond to exact collinear splits (1–5 % of calls) and to the input-axis position (8–20 %), which IRC-safe classical taggers do not (0 %). QG networks are nearest LR(N, pT^D).
