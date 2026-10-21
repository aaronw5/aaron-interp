# H65 verification: soft-contamination density scan (fresh seeds, shrink control, particle-ID arm)

Verifier for node H65 (loop D). Script `interp/h65v_verify.py`, which imports the H65 code (`interp/h65_pileup_dose.py`) unchanged and replaces only the seeds. Jobs: `d-h65-verify` (GPU: JetClass, Top, QG), `d-h65-verify-hls` (CPU: hls4ml), `d-h65-verify-sum` (Holm), all Complete. Results: `results/physics/h65_verify_{hls,jc,top,qg}.json`, `h65_verify_summary.json`.

## Conclusion

The H65 numbers reproduce on new jets. A second test then showed that the large response of the JetClass full-input ParT to soft particles stacked on the jet axis comes mostly from those particles being labelled neutral hadrons. When the same particles at the same positions are labelled photons, the stacked response falls to about a quarter of the spread-out response (0.25 for W, against 1.19 with the neutral-hadron label). The count response at fixed mass in the Top Landscape ParT/PFN and in the JetClass ParticleNet_full survives both the label change and a spread of the stack out to about 0.1 rad, so it is not caused by the near-coincident placement. Not tested: how the label effect compares with real pileup composition in JetClass (Delphes), and charged ghosts with realistic impact parameters.

## Procedure

- **Samples (fresh seeds, random rows, never first-k):**
  - hls4ml: `jedi_core.sample(2000, 8651)`, 10,000 jets.
  - JetClass: first test file per class, QCD/Zqq/Wqq/Tbqq, 2,000 random rows each (seed 8652).
  - Top Landscape test: 2,000 + 2,000 (seed 8653).
  - QG Pythia test: 2,000 + 2,000 (seed 8654).
- **Edit:** the same as H65. Massless ghosts with pT uniform in 0.5–1 GeV are placed uniformly in the jet disc of radius R about the stored axis. The count is Poisson with mean ρπR²/0.75 GeV. Sets are nested by thinning from ρ = 20, and inputs stay about the stored axis.
- **Conditions repeated from H65:** pu_ρ for ρ = 2, 5, 10, 20 GeV per unit η–φ area; c1_10, c2_10, in10, out10.
- **New control 1, shrink sweep sh_f_10.** The ρ = 10 ghosts keep their pu10 pT values and angular pattern, but every ghost's (Δη, Δφ) from the axis is multiplied by f, for f ∈ {0.003, 0.03, 0.06, 0.125, 0.25, 0.5}. f = 1 is pu10, and f = 0.003 is a disc version of c2.
  - Count and summed pT stay fixed, while the added plain mass grows roughly as f².
  - Why this is the right comparison: c2 piles about 27 particles into 0.004 rad, a configuration the networks never saw in training. If the c2 response were an artefact of that stacking, Δs would change between f = 0.003 and f = 0.06–0.125. At f = 0.125 the ghosts form a physical soft cluster of radius 0.1 rad that adds only +1.9 GeV of mass in JetClass W.
- **New control 2, particle-ID arm (JetClass only).** The pu10, c2_10 and sh0.125_10 ghosts are relabelled, with positions and pT unchanged:
  - as photons (`_ph`);
  - as charged hadrons (`_ch`): charge ±1 at random, d0 = dz = 0, and d0err/dzerr equal to the jet's median over its real charged particles.

  H65 labelled every ghost a neutral hadron. The kinematics-only models (ParT_kin, PN_kin) do not read the labels, so they act as a built-in null: their label differences are exactly 0.000 in every test.
- **Metrics:**
  - mean Δ log-odds Δs over the signal + QCD rows of each task, with a 95 % bootstrap CI over jets (B = 1000);
  - share = Δs(condition)/Δs(pu10), with a bootstrap CI;
  - ε_S, the signal efficiency at the unedited 50 % threshold;
  - paired bootstrap p (normal approximation) for H0: Δs(sh0.003) = Δs(sh0.125) (stacking) and for H0: Δs(label) = Δs(neutral) (particle ID), with Holm over the 147 tests for the trained models.

## Unedited models on the fresh samples

- **hls4ml:** JEDI-linear 5-class accuracy 0.8060, against the known 0.8092. Redrawn JEDI 0.200.
- **JetClass, per-class accuracy QCD/Z/W/t:**
  - ParT_full 0.761 / 0.697 / 0.798 / 0.948;
  - ParT_kin 0.714 / 0.564 / 0.680 / 0.899;
  - PN_full 0.747 / 0.666 / 0.775 / 0.947;
  - PN_kin 0.686 / 0.524 / 0.671 / 0.894;
  - permuted and untrained ParT_full 0.00.
- **Top AUC:** ParT 0.9876, PN 0.9873, LorentzNet 0.9883, PFN 0.9830, EFN 0.9788. Permuted PFN 0.70.
- **QG AUC:** part_kin 0.9026, pnet_kin 0.9001, pfn_k 0.8908, efn_k 0.8820, lorentznet 0.9136. Permuted part_kin 0.545.

## Code review (interp/h65_pileup_dose.py)

- **Axis and coordinate conventions.**
  - JetClass ghosts go through `jc_table` → `add_parts` → `jc_maker`/`build`, which uses η = jet_eta + sign(jet_eta)·Δη. This is consistent.
  - Inputs are taken about the stored axis, and jet_pt/jet_energy are recomputed with the ghosts included.
  - Top and QG use `top_J` and the h36 axis patch, so inputs are about the stored axis.
  - No bug found.
- **Padded particles:**
  - `add_parts` writes pt = 0 for ghosts that are off.
  - `finalize` zeroes the padding.
  - `fin` counts a slot as a ghost only where its pT is above 0 and its column is ≥ W.
- **Sampling:** every dataset uses random indices.
- **Working point:** the threshold is the median unedited signal score, so ε_S = 0.50 by construction.
- **Observables:** computed on the particles each model sees, after the slot cut. The one exception: Top PFN/EFN see only the first 128 slots (`TopModels.score`), while the observables use up to 200. Ghosts are softer than most real particles, so at high ρ some ghosts the PFN/EFN never see are counted in m and N. This is minor.
- **QG:** ghosts get pid 130. QG part_kin, pnet_kin, pfn_k and efn_k are kinematics-only, so the label does not enter.
- **c2 control:** stacked within 0.004 rad. H65 kept PN_full's c2 reading (share 1.36) while excluding PN_kin as kNN-OOD, without a test that separates the two cases. This verification supplies that test (below).
- **Earlier issues:** the issues the worker listed (calls-changed saturation, win_msub under non-uniform controls, the hls4ml 64-slot cut) are correctly stated.
- **Figure** (`h65_pileup.png`): the left panel shows the real JetClass W jet with 27 ghosts at ρ = 10, m 80 → 101 GeV and m_SD 79 → 79 GeV. The right panel shows ε_S(ρ) for ParT_full pu and c2, win_m, win_msd, win_msub and LR(N), and matches the summary JSON. Axes and legends are correct.

## Fresh-seed reproduction (H65 → verify)

| quantity | H65 | verify |
|---|---|---|
| JetClass W ParT_full Δs(pu10) | −2.48 [−2.54, −2.42] | −2.48 [−2.53, −2.42] |
| JetClass W ParT_full c2 share | 1.17 | 1.19 [1.17, 1.20] |
| JetClass W ParT_full r_Δs | 0.53 | 0.53 [0.52, 0.53] |
| JetClass W ParT_full ε_S at ρ 2/5/10/20 | 0.31 / 0.19 / 0.10 / 0.02 | 0.33 / 0.18 / 0.09 / 0.02 |
| JetClass W win_msd ε_S at ρ 20 | 0.43 | 0.43 |
| JetClass W ParT_kin c2 share | 0.39 | 0.39 [0.38, 0.40] |
| JetClass W PN_full c2 share | 1.36 | 1.37 [1.36, 1.38] |
| hls4ml W JEDI Δs(pu10), c2 share, r_Δs | −2.20, 0.01, 3.22 | −2.24, 0.01 [0.01, 0.02], 3.11 [2.95, 3.28] |
| hls4ml W JEDI ε_S at ρ 2/5 | 0.12 / 0.00 | 0.12 / 0.003 |
| Top ParT Δs(pu10), c2 share, r_Δs | −0.60, 1.22, 0.39 | −0.62, 1.19 [1.15, 1.25], 0.39 [0.36, 0.43] |
| Top PFN c2 share | 1.9 | 1.83 [1.71, 1.98] |
| QG part_kin Δs(pu10), c2 share, r_Δs | −0.65, 0.16, 4.6 | −0.67, 0.16 [0.15, 0.17], 4.66 [4.50, 4.86] |
| QG pfn_k c2 share | 0.42 | 0.41 [0.40, 0.43] |

All are within the bootstrap intervals or within 0.02 of them.

## Shrink sweep: Δs at ρ = 10 as the ghosts are pulled toward the axis

Values are for f = 0.003 / 0.03 / 0.06 / 0.125 / 0.25 / 0.5 / 1 (pu10). JetClass W Δm, for reference: +1.4 / 1.4 / 1.5 / 1.9 / 3.2 / 8.0 / 24.3 GeV.

| model | Δs | reading |
|---|---|---|
| JEDI-linear hls4ml W | −0.03 / −0.04 / −0.16 / −0.65 / −1.65 / −2.74 / −2.24 | grows with the added mass. sh0.003 − sh0.125 = +0.63, p_Holm < 1e-10 |
| ParT_full JetClass W | −2.94 / −3.13 / −3.34 / −3.75 / −4.11 / −3.71 / −2.48 | not a stacking artefact: the 0.1-rad cluster moves it more (share 1.51) |
| ParT_kin JetClass W | −0.93 / −1.00 / −1.05 / −1.07 / −1.71 / −2.88 / −2.35 | share 0.39–0.46 up to f = 0.125 |
| PN_full JetClass W | −4.00 / −4.17 / −4.23 / −4.61 / −4.25 / −3.36 / −2.94 | not a stacking artefact (share 1.57 at f = 0.125) |
| PN_kin JetClass W | −24.1 / −15.6 / −8.2 / −4.6 / −3.4 / −3.0 / −2.5 | stacking artefact, as H65 stated |
| ParT Top | −0.74 / −0.75 / −0.75 / −0.74 / −0.75 / −0.71 / −0.62 | flat. sh0.003 − sh0.125 = +0.003, p_Holm = 1 |
| PFN Top | −1.03 / −1.03 / −1.04 / −1.06 / −1.01 / −0.79 / −0.57 | flat |
| PN Top | +1.82 / +1.25 / +0.68 / +0.20 / −0.23 / −0.32 / −0.28 | stacking artefact |
| LorentzNet Top | −3.10 / −2.53 / −1.54 / −0.67 / −0.30 / −0.02 / +0.13 | stacking artefact |
| part_kin QG | −0.11 / −0.11 / −0.11 / −0.15 / −0.26 / −0.50 / −0.67 | share 0.16, flat to f = 0.06 |
| lorentznet QG | −0.51 at every f / −0.48 | position-independent; c1 alone gives −0.24 |

## Particle-ID arm (JetClass)

Δs at ρ = 10. Each share uses the pu10 value with the same label as its denominator.

| model, class | label | pu10 | c2_10 (share) | sh0.125_10 (share) | ε_S pu10 |
|---|---|---|---|---|---|
| ParT_full W | neutral hadron (H65) | −2.48 | −2.94 (1.19) | −3.75 (1.51) | 0.089 |
| ParT_full W | photon | −2.03 | −0.51 (0.25) | −0.78 (0.38) | 0.111 |
| ParT_full W | charged hadron | −2.17 | −1.31 (0.60) | −1.55 (0.71) | 0.026 |
| ParT_full Z | neutral / photon / charged | −2.50 / −2.02 / −1.98 | −2.58 / −0.34 / −0.85 (1.03 / 0.17 / 0.43) | −3.37 / −0.63 / −1.17 | 0.04 / 0.06 / 0.02 |
| PN_full W | neutral / photon / charged | −2.94 / −1.94 / −2.47 | −4.01 / −2.23 / −3.98 | −4.61 / −1.61 / −1.39 (1.57 / 0.83 / 0.56) | 0.005 / 0.013 / 0.001 |
| ParT_kin, PN_kin | any | identical to neutral, difference 0.000 | | | |

Statistics:
- Photon minus neutral, ParT_full W:
  - c2: +2.43 [2.37, 2.49];
  - sh0.125: +2.97 [2.91, 3.03];
  - pu10: +0.45 [0.44, 0.46];
  - all p_Holm < 1e-20.
- Holm over 147 tests: 107 have p_Holm < 0.05. The 40 that do not are:
  - the 36 label tests on the kinematics-only models, which have a difference of exactly 0;
  - the Top ParT stacking tests (2);
  - Top EFN sh0.125 vs pu10;
  - PN_full W c2_ch vs c2.

## Competing hypotheses for the JetClass full-model count response, and what the new controls exclude

- **H_stack** (the c2 response is an artefact of near-coincident particles). Predicts Δs at f = 0.06–0.125 near 0. Excluded for ParT_full, ParT_kin, PN_full, Top ParT, Top PFN and QG part_kin/pfn_k: Δs at f = 0.125 equals or exceeds Δs at f = 0.003. Confirmed for PN_kin, Top PN and Top LorentzNet, the cases H65 had already set aside.
- **H_count-kin** (the network counts soft particles from their kinematics alone). Predicts the same share whatever the label. Excluded for ParT_full: the share falls from 1.19 to 0.25 (W) and from 1.03 to 0.17 (Z) when the label changes from neutral hadron to photon. For the photon label it is consistent with ParT_kin's 0.39/0.45. For PN_full the share falls less (1.57 → 0.83 at f = 0.125), so a kinematic count response remains there.
- **H_label** (most of ParT_full's fixed-mass response is to extra neutral-hadron-labelled soft particles). Consistent with every ParT_full number above.

## Corrected reading

Under uniform 0.5–1 GeV ghosts at a fixed 50 %-efficiency threshold:
- Every trained network loses signal calls faster than the soft-drop-mass and area-subtracted-mass windows. None gains QCD calls. This is reproduced on fresh seeds and holds for photon- and charged-labelled ghosts too: ParT_full W ε_S at ρ = 10 is 0.11 / 0.03, against win_msd 0.47.
- **JEDI-linear (hls4ml W/Z):** the response follows the added plain mass. The c2 share is 0.01, and Δs grows with f² in the shrink sweep.
- **Top Landscape ParT and PFN** (kinematic inputs): the same ghosts stacked at the axis, or clustered within 0.1 rad, reproduce 1.2 (ParT) or 1.8 (PFN) of the spread-out response, flat in cluster size. This is a count response at fixed mass and fixed summed pT, and not a stacking artefact.
- **JetClass ParT_full:** the on-axis share of 1.0–1.2 (W/Z) depends on the ghosts being labelled neutral hadrons. With photon labels it is 0.17–0.25, close to ParT_kin's 0.39–0.45. The "count at fixed mass" description applies to ParT_full only for neutral-hadron-labelled particles.
- **JetClass PN_full:** keeps a kinematic count response (share 0.83 at f = 0.125 with photon labels).
- **QG ParT/PN/PFN:** share 0.16–0.42; out-of-core ghosts dominate (r_Δs 2.7–4.7).
- The c2 control is invalid for PN_kin, Top PN and Top LorentzNet (stacking artefact, confirmed by the shrink sweep).

## Corrected Conclusion (for the H65 entry)

Adding a uniform haze of 0.5–1 GeV particles over the jet makes the hls4ml tagger respond only through the plain jet-mass increase. The Top Landscape ParT and PFN respond to the number of soft particles added even when they add no mass, and this holds whether the particles are stacked on the axis or spread over 0.1 rad. In JetClass, most of the full-input ParT's response to particles added without mass comes from their neutral-hadron label: relabelled as photons, the stacked particles produce about a quarter of the spread-out response. No network behaves like a groomed-mass or area-subtracted-mass tagger; those keep about 40 % W efficiency at 20 GeV per unit area while the networks keep almost none. Not yet known: which input path inside ParT_full carries the neutral-hadron effect, and how JetClass's real pileup composition compares.

## Glossary additions

- **shrink factor f:** the multiplier applied to each ghost's (Δη, Δφ) from the jet axis. Count and pT are fixed; f = 1 is the uniform haze.
- **label arm (_ph, _ch):** the same ghosts with the particle-type inputs set to photon, or to charged hadron.
- **share:** Δs(condition)/Δs(pu10 with the same label).
