# H49: groomed versus ungroomed mass at matched added count, pT and ungroomed mass change (new claim D21; refines D10)

Script: `interp/h49_groomed_mass.py`, with parts hls, hlss, jc, top and summary.
- **Jobs.** d-h49-hls (CPU, 1 CPU / 4 GiB, 448 s), d-h49-hlss (conda CPU), d-h49-jc and d-h49-top (GPU), d-h49-sum2 (statistics and figure). All exited 0.
- **Smoke runs.** d-h49-sm-*, d-h49-sm2-*.
- **Failed run.** d-h49-sum failed. The ratio to MLP(m) was undefined, because MLP(m) gives D = 0 by construction, and the bootstrap interval came back empty. I removed that ratio and reran as d-h49-sum2.
- **Results.** Written to `results/physics/`: `h49_{hls,hlss,jc,top}.{npz,json}`, `h49_summary.json` and the figure `h49_groomed.png`. Logs are in `logs/d-h49-*.log`.

## Conclusion
- **The test.** We added three soft particles (2 % of the jet pT in total) to simulated jets in two ways that give the same number of particles, the same added pT and the same rise in the ordinary jet mass. In one placement soft drop removes the particles (the groomed mass m_SD stays fixed). In the other, soft drop keeps them (m_SD rises by about the same 6–12 GeV). Soft drop is the standard grooming step that removes soft wide-angle radiation.
- **hls4ml.** JEDI-linear and the three small nets react almost equally to both placements on W and Z jets, as a plain-mass reference classifier does. Their W/Z difference is 0–5 % of the difference shown by a groomed-mass reference classifier.
- **JetClass.** ParT and ParticleNet react to both placements. They react more to the kept placement by roughly the amount the groomed-mass reference does: 0.4–1.4 times it, ParT_full 1.15× on W jets.
- **Top Landscape.** The top taggers also react more to the kept placement, but here the groomed-mass references move the other way. So that part is not a groomed-mass response.
- **Still uncertain.** The kept particles sit inside a prong, so they also change that prong's own mass and shape. This test cannot separate "follows m_SD" from "follows the mass of the subjet the particle joins".

## Procedure
- **Samples (random rows, never first-k).**
  - hls4ml: `jedi_core.sample(2000, 4901)` (the same rows as `h1_irc_panel.hls_sample`), 10,000 jets in 5 classes, 64 slots.
  - JetClass: QCD (ZJetsToNuNu), Wqq, Zqq and Tbqq, 2,000 random rows each of each class's first test file, seed 4902, 8,000 jets, 128 slots. The build uses η = jet_eta + sign(jet_eta)·Δη.
  - Top Landscape: `data_top.sample(3000, 4903)`, 3,000 top + 3,000 QCD test jets, 200 slots.
- **Models.**
  - hls4ml: JEDI-linear official n64 3-feature, plus three small nets at seed 0 (Deep Sets dsets-d64-n64, Lin-former linf-2b-d32-k8-n64, Transformer tf-2b-d32-n64).
  - JetClass: ParT_full, ParT_kin, PN_full, PN_kin.
  - Top Landscape: ParT and PN (kinematic inputs), LorentzNet, PFN, EFN.
  - Nulls: JEDI-linear with redrawn weights, ParT_full with every weight tensor's entries permuted (seed 7), PFN with permuted weights.
- **Score.** s = log-odds of signal class c (W, Z, t) against QCD.
  - hls4ml: log p(c) − log(p(q) + p(g)).
  - JetClass: logit_c − logit_QCD.
  - Top: the network's top log-odds.
- **Finding the prongs (per jet, on the capped particle list the model sees).**
  - Recluster with Cambridge/Aachen (C/A, angular-ordered pairwise clustering), E-scheme recombination, distance in rapidity–azimuth.
  - Apply soft drop with β = 0 and z_cut = 0.1. With β = 0 the R0 = 0.8 parameter drops out. Soft drop declusters, following the harder branch, until min(pT1, pT2)/(pT1 + pT2) > 0.1.
  - The two branches at that step are prongs A and B, and R_g is their angular separation.
  - F is the prong farther from the jet's E-scheme axis (the direction of the summed particle 3-momentum), at distance r_F.
  - On 20 synthetic jets this soft-drop code matches the repository's `jetdistill.observables.compute._softdrop` to 3e-13 GeV.
- **Added particles.** k = 3 massless neutral particles, each with pT q = 0.02·pT_J/3. JetClass fields: neutral hadron, d0 = dz = 0. All three sit on a circle of radius r* = r_F + d about the jet axis, with d = min(0.7 R_g, 0.78 − r_F) and d ≥ 0.3 R_g. They are spaced 0.15 R_g apart along the circle.
- **Edit G_out.** The particles are centred on one of 9 directions 60–300° from F (30° steps). Among the directions whose centre is ≥ 1.1 R_g from both prong centres, the one with the least unedited pT within 0.5 R_g is used. These particles join the jet after A and B have merged, so soft drop removes them.
- **Edit G_in.** The particles are centred on the direction of F, i.e. a distance d beyond F's centre on the outer side. C/A attaches them to F before A and B merge, so soft drop keeps them.
- **Why equal radius gives equal ungroomed Δm.** Δ(m²) = 2 P_J·p_added ≈ q pT_J (r² + ⟨r²⟩) to leading order in the angles, so equal radius gives equal ungroomed Δm. The G_in radius is then rescaled per jet by bisection so that m(G_in) = m(G_out) to 0.01 GeV. The median rescale factor is 1.000.
- **Deviation from the plan.** The plan put G_out at 0.5–0.75 from the axis. Matching the ungroomed Δm requires G_in at the same radius, and soft drop keeps a particle only within about R_g of a prong. So both edits sit at r* ≈ 1.4 R_g: median r* = 0.22 (hls4ml W), 0.37 (JetClass W), 0.70 (JetClass t), 0.73 (Top t).
- **Matched set (all statistics).** A jet enters only if all of these hold:
  - a split passing z_cut was found, with R_g ≥ 0.02;
  - r*, r_in ≤ 0.8;
  - |Δm_SD(G_out)| < 0.5 GeV;
  - Δm_SD(G_in) ≥ 0.5·Δm(G_out);
  - |m(G_in) − m(G_out)| < 1 GeV;
  - all added particles fall inside the slot limit.

  Pass fractions:

  | sample | W | Z | t | QCD |
  |---|---|---|---|---|
  | hls4ml | 79.7 % | 80.1 % | 58.0 % | 42.9 % (g), 36.2 % (q) |
  | JetClass | 82.5 % | 80.8 % | 62.6 % | 63.7 % |
  | Top Landscape | – | – | 56.0 % | 70.6 % |

  QCD jets fail mostly for having no split with R_g ≥ 0.02 (hls4ml q: 39 % pass), and tops mostly because r* would exceed 0.8.
- **Controls (same jets, same q).**
  - **c0 (count-only).** The 3 particles are placed at the jet axis (radius 0.002), and their total pT 3q is removed from the jet's softest particles: pT and mass are scaled by 1 − 3q/S over the softest set whose pT sum S reaches 6q. The count rises by 3, ΣpT is unchanged (to 1e-15 GeV on Top), and m changes by −1.8 to −6.6 GeV.
  - **rand.** The 3 particles are placed uniformly in the disc of radius 0.8 about the axis.
  - **TN (translation).** The unedited particles with inputs taken about the E-scheme axis of G_in or G_out. This is the axis shift alone.
- **Builds.**
  - st: inputs about the stored axis. That is (0, 0) for hls4ml, jet_eta / jet_phi for JetClass, and the unedited E-scheme axis for Top.
  - rc: inputs about each edited jet's own E-scheme axis.
  - Both are reported. Numbers below are rc unless marked.
- **References (on the same edited jets).**
  - Windows: W/Z, 65 < m < 105 GeV and the same on m_SD. Top: 105 < m_SD < 210 GeV and τ32 < 0.54, where τ32 = τ3/τ2 with exclusive-kT axes, β = 1, all particles. The same top window is also applied on ungroomed m.
  - MLP(m) and MLP(m_SD): numpy MLPs (1 → 32 → 32 → classes, tanh, Adam, 1,500 full-batch steps) trained on the unedited observable. They are 2-fold cross-fitted on the sample, so each jet is scored by the half not trained on it, and give log-odds in the same way as the networks.
- **Metrics (per model × signal class, on matched signal jets).**
  - Δs(X) = mean score change under edit X.
  - D = mean of Δs(G_in) − Δs(G_out) per jet, with a 95 % bootstrap interval over jets (B = 2,000) and a sign-flip permutation p (10,000 permutations).
  - Ratio r_SD = D_net / D_MLP(m_SD) on the same jets, with a paired bootstrap interval.
  - Signal efficiency at ε_B = 5 %, with the threshold re-set on each condition's QCD jets, as ε_S(G_in) − ε_S(G_out) with a bootstrap interval.
  - Exact McNemar test on the calls at the frozen unedited threshold (5 % QCD), G_in vs G_out.
  - Holm correction within each build over 12 networks (nulls included) × signal classes × datasets = 36 tests, separately for D and for McNemar.
- **Hypotheses.**
  - H0 (the call follows the ungroomed mass, the count or ΣpT): D ≈ 0, as for MLP(m) and the m window.
  - H1 (the call follows the groomed mass): D ≈ D of MLP(m_SD), i.e. r_SD ≈ 1.
  - H2 (count only, as in the H13v reading for JetClass/Top): Δs(G_in) ≈ Δs(G_out) ≈ Δs(c0).

## Unedited models on the samples (reproduction check; whole sample)
| model | sample value | known |
|---|---|---|
| JEDI-linear | 5-class acc 0.8091 | 0.8092 (260k test) |
| JEDI-linear redrawn | 0.200 | chance |
| Deep Sets d64 / Lin-former / Transformer | 0.8074 / 0.8103 / 0.7583 | 0.8064 / 0.8107 / 0.7619 |
| ParT_full (per-class acc QCD / Zqq / Wqq / Tbqq) | 0.775 / 0.697 / 0.799 / 0.955 | 0.783 / 0.689 / 0.805 / 0.956 (H18) |
| ParT_kin | 0.712 / 0.575 / 0.687 / 0.910 | 0.718 / 0.550 / 0.702 / 0.912 |
| PN_full | 0.766 / 0.655 / 0.773 / 0.952 | 0.764 / 0.652 / 0.784 / 0.949 |
| PN_kin | 0.673 / 0.545 / 0.665 / 0.911 | 0.693 / 0.528 / 0.680 / 0.913 |
| ParT_full permuted | 0.00–0.01 per class | chance |
| Top ParT / PN / LorentzNet / PFN / EFN (AUC) | 0.9871 / 0.9866 / 0.9876 / 0.9804 / 0.9753 | 0.9858 / 0.9858 / 0.9870 / 0.9819 / 0.975–0.979 |
| PFN permuted (AUC) | 0.706 | 0.70 (H13, H18) |

## What the edits do (median on matched jets)
| sample, class | R_g | r* | q [GeV] | Δm G_in = G_out [GeV] | Δm_SD G_out | Δm_SD G_in | c0 Δm / Δm_SD | rand Δm / Δm_SD | axis shift G_in / G_out [mrad] |
|---|---|---|---|---|---|---|---|---|---|
| hls4ml W | 0.157 | 0.215 | 6.8 | +6.7 | 0.0 | +7.0 | −1.8 / −1.1 | +36.4 / 0.0 | 4.5 / 4.5 (all classes) |
| hls4ml Z | 0.178 | 0.245 | 6.8 | +7.7 | 0.0 | +7.9 | −1.9 / −1.3 | +33.5 / 0.0 | |
| hls4ml t | 0.273 | 0.378 | 6.6 | +10.6 | 0.0 | +11.2 | −3.0 / −2.1 | +23.1 / 0.0 | |
| hls4ml QCD | 0.083 | 0.119 | 6.9 | +2.7 | 0.0 | +4.4 | −4.8 / −0.5 | +40.1 / 0.0 | |
| JetClass W | 0.270 | 0.371 | 3.9 | +6.3 | 0.0 | +6.8 | −2.9 / −1.1 | +13.2 / 0.0 | 8.0 / 8.1 |
| JetClass Z | 0.306 | 0.423 | 3.9 | +7.3 | 0.0 | +8.0 | −3.2 / −1.3 | +12.5 / +0.4 | |
| JetClass t | 0.494 | 0.700 | 4.1 | +11.3 | 0.0 | +11.8 | −3.6 / −2.3 | +8.9 / +3.9 | |
| JetClass QCD | 0.107 | 0.157 | 4.0 | +2.3 | 0.0 | +3.8 | −5.3 / −0.4 | +15.2 / 0.0 | |
| Top t | 0.512 | 0.726 | 4.0 | +11.8 | 0.0 | +12.3 | −4.0 / −2.7 | +8.4 / +4.5 | 9.8 / 9.7 |
| Top QCD | 0.133 | 0.194 | 4.0 | +2.6 | 0.0 | +4.1 | −6.6 / −0.1 | +13.8 / 0.0 | |

- The particle count seen rises by 3.0 on JetClass and Top. On hls4ml it rises by 2.1–2.9, because jets at the 64-slot limit drop their softest particles.
- ΣpT rises by 2 % of pT_J (G_in, G_out, rand) or is unchanged (c0).

## Numbers (rc build; Δs = mean change of the signal-vs-QCD log-odds on signal jets; 95 % CI)
| sample, model, class | Δs G_in | Δs G_out | Δs c0 | Δs rand | D = G_in − G_out | r_SD = D / D_MLP(m_SD) | Holm p (D) | ε_S(G_in) − ε_S(G_out) at ε_B 5 % | McNemar calls lost only under G_in / only under G_out (Holm p) |
|---|---|---|---|---|---|---|---|---|---|
| hls4ml MLP(m) W | −3.06 | −3.06 | −0.41 | −5.84 | 0.00 | – | – | 0.00 | 0 / 0 |
| hls4ml MLP(m_SD) W | −4.03 | 0.00 | −0.41 | −0.09 | −4.03 [−4.20, −3.87] | 1 | – | −0.68 [−0.73, −0.61] | 1104 / 59 |
| hls4ml JEDI-linear W | −3.38 | −3.28 | −0.13 | −3.55 | −0.10 [−0.14, −0.07] | 0.02 [0.02, 0.03] | 0.004 | −0.05 [−0.07, −0.01] | 93 / 58 (0.08) |
| hls4ml JEDI-linear Z | −3.23 | −3.27 | −0.16 | −4.17 | +0.04 [+0.01, +0.07] | −0.01 [−0.02, −0.00] | 0.11 | −0.01 | 75 / 51 (0.4) |
| hls4ml Deep Sets W / Z | −3.31 / −3.40 | −3.24 / −3.39 | −0.35 / −0.59 | −5.70 / −5.28 | −0.08 [−0.10, −0.06] / −0.01 [−0.03, +0.01] | 0.02 / 0.00 | 0.004 / 1 | −0.04 / −0.00 | |
| hls4ml Lin-former W / Z | −3.50 / −3.50 | −3.31 / −3.34 | −0.57 / −0.39 | −3.38 / −3.39 | −0.18 [−0.21, −0.16] / −0.15 [−0.18, −0.13] | 0.05 / 0.04 | 0.004 / 0.004 | −0.05 / −0.03 | |
| hls4ml Transformer W / Z | −3.12 / −2.59 | −3.33 / −2.87 | +0.15 / +0.27 | −2.75 / −2.42 | +0.20 [+0.17, +0.23] / +0.28 [+0.25, +0.30] | −0.05 / −0.07 | 0.004 / 0.004 | +0.02 / −0.01 | |
| hls4ml MLP(m_SD) t | −0.53 | 0.00 | −0.08 | −0.03 | −0.53 [−0.60, −0.45] | 1 | – | −0.26 [−0.32, −0.20] | |
| hls4ml JEDI-linear t | −0.59 | −0.25 | −0.07 | +0.10 | −0.34 [−0.37, −0.30] | 0.64 [0.55, 0.76] | 0.004 | −0.01 [−0.04, +0.02] | 58 / 8 (4e-9) |
| hls4ml Deep Sets / Lin-former / Transformer t | −0.69 / −0.35 / −0.29 | −0.32 / −0.14 / −0.12 | −0.04 / −0.25 / +0.28 | | −0.37 / −0.20 / −0.17 | 0.70 / 0.38 / 0.32 | 0.004 (each) | | |
| JetClass MLP(m) W / Z | −0.17 / −0.19 | −0.17 / −0.19 | −0.04 / −0.08 | −0.43 / −0.33 | 0.00 | – | – | 0.00 | |
| JetClass MLP(m_SD) W / Z | −0.46 / −0.40 | 0.00 / 0.00 | −0.08 / −0.07 | −0.04 / −0.04 | −0.46 [−0.51, −0.41] / −0.40 [−0.44, −0.36] | 1 | – | −0.14 [−0.24, −0.06] / −0.08 [−0.18, +0.04] | |
| JetClass ParT_full W | −2.07 | −1.54 | −0.46 | −1.04 | −0.53 [−0.59, −0.46] | 1.15 [1.00, 1.32] | 0.004 | −0.03 [−0.05, −0.01] | 137 / 34 (2e-14) |
| JetClass ParT_full Z | −1.85 | −1.30 | −0.35 | −1.00 | −0.55 [−0.61, −0.50] | 1.39 [1.23, 1.58] | 0.004 | −0.06 [−0.08, −0.02] | 184 / 29 (3e-27) |
| JetClass ParT_kin W / Z | −1.63 / −1.50 | −1.44 / −1.25 | −0.04 / −0.01 | −0.99 / −0.96 | −0.19 [−0.23, −0.15] / −0.26 [−0.29, −0.21] | 0.41 [0.33, 0.49] / 0.64 [0.55, 0.74] | 0.004 | −0.04 / −0.08 | |
| JetClass PN_full W / Z | −1.76 / −1.58 | −1.42 / −1.26 | −0.54 / −0.49 | −0.98 / −0.95 | −0.34 [−0.39, −0.29] / −0.32 [−0.37, −0.27] | 0.74 [0.63, 0.87] / 0.80 [0.69, 0.93] | 0.004 | −0.04 / −0.03 | |
| JetClass PN_kin W / Z | −1.83 / −1.70 | −1.52 / −1.36 | −0.17 / −0.10 | −1.19 / −1.12 | −0.31 [−0.35, −0.27] / −0.34 [−0.38, −0.30] | 0.67 [0.59, 0.76] / 0.85 [0.76, 0.96] | 0.004 | −0.05 / −0.05 | |
| JetClass MLP(m_SD) t | 0.00 | 0.00 | −0.07 | +0.03 | 0.00 [−0.04, +0.04] | 1 | – | −0.16 | |
| JetClass ParT_full / ParT_kin / PN_full / PN_kin t | −0.72 / −0.75 / −0.69 / −0.74 | −0.28 / −0.30 / −0.31 / −0.29 | −0.51 / −0.23 / −0.46 / −0.32 | −0.36 / −0.38 / −0.34 / −0.44 | −0.44 / −0.45 / −0.37 / −0.45 (CIs ±0.04) | undefined (D_MLP(m_SD) ≈ 0) | 0.004 | ≈ 0 (ε_S ≥ 0.92 at 5 % QCD) | |
| Top MLP(m_SD) t (st) | +0.11 | 0.00 | −0.09 | +0.08 | +0.11 [+0.08, +0.14] | 1 | – | −0.06 | |
| Top ParT t | −0.70 | −0.30 | −0.15 | −0.48 | −0.40 [−0.44, −0.36] | −3.7 [−5.5, −2.7] | 0.004 | +0.00 | 44 / 3 (6e-9) |
| Top PN t | −0.97 | −0.44 | −0.17 | −0.54 | −0.53 [−0.58, −0.49] | −4.9 | 0.004 | −0.01 | 48 / 9 (3e-6) |
| Top LorentzNet t | −0.95 | −0.46 | −0.34 | −0.55 | −0.49 [−0.54, −0.45] | −4.5 | 0.004 | −0.00 | 50 / 7 (9e-8) |
| Top PFN t | −0.43 | −0.39 | −0.21 | −0.30 | −0.04 [−0.06, −0.02] | −0.34 | 0.004 | 0.00 | 43 / 11 (2e-4) |
| Top EFN t | −0.50 | −0.85 | +0.10 | −0.56 | +0.35 [+0.32, +0.38] | +3.2 | 0.004 | +0.02 | 32 / 84 (3e-5) |
| nulls: JEDI redrawn (W, Z, t) | ≤ 0.004 | ≤ 0.004 | | | \|D\| < 0.005 (Z: Holm 0.004 at this size) | | | | |
| nulls: ParT_full permuted (W, Z, t) | +0.01, +0.05, −0.01 | +0.01, +0.08, −0.01 | | | 0.00, −0.03 [−0.05, −0.01], 0.00 | | 1, 0.053, 1 | | |
| nulls: PFN permuted (Top) | +0.05 | +0.09 | | | −0.03 [−0.04, −0.02] | | 0.004 | | |

- **Holm summary.** 29 of 36 D tests have Holm p < 0.05 in rc (30 of 36 in st); for McNemar it is 22 of 36 (rc) and 24 (st). With 10,000 sign flips the smallest attainable p is about 1e-4, so the smallest Holm p is 0.004.
- **st vs rc.** The st and rc builds differ by ≤ 0.10 in D for every model; for example, ParT_full W gives −0.52 (st) vs −0.53 (rc).
- **Translation control.** D − (TN_in − TN_out) equals D to within 0.06 for every model, so the axis shift each edit causes contributes nothing to D. G_in and G_out shift the axis by the same 4.5 / 8 / 10 mrad.
- **QCD jets.** D on QCD jets is −0.01 to −0.16 for the trained networks; Top PN and LorentzNet give +0.12 / +0.17.
- **Classical windows (ε_S change, G_in − G_out, 95 % CI).**

  | window | hls4ml | JetClass | Top Landscape |
  |---|---|---|---|
  | W on m | 0 | 0 | – |
  | W on m_SD | +0.02 [+0.01, +0.02] | −0.00 | – |
  | Z on m | 0 | 0 | – |
  | Z on m_SD | −0.10 [−0.11, −0.08] | −0.14 [−0.16, −0.12] | – |
  | top on m_SD with τ32 | +0.02 | +0.05 [+0.03, +0.07] | +0.06 [+0.04, +0.07] |
  | top on m with τ32 | +0.02 | +0.06 | +0.06 |

  The top windows differ between G_in and G_out because G_out raises τ32, which loses tops: Top ε_S is 0.485 → 0.437 (G_in) and → 0.380 (G_out).
- **Argmax calls changed (multiclass models, rc, matched jets), G_in / G_out / c0.**
  - JEDI-linear: 34.1 / 33.2 / 7.0 %
  - ParT_full: 18.7 / 12.2 / 5.7 %
  - PN_full: 18.9 / 13.3 / 4.7 %
  - ParT_kin: 28.6 / 21.4 / 9.5 %
  - PN_kin: 32.1 / 25.1 / 8.0 %

## Reading
- **hls4ml W/Z.**
  - On W and Z jets the hls4ml networks lower the W/Z log-odds by 3.2–3.5 under both placements, and the two placements differ by −0.18 to +0.28.
  - This is about MLP(m)'s pattern (−3.06 for both, D = 0). It is not MLP(m_SD)'s (−4.03 vs 0.00): r_SD is 0.00–0.05 for JEDI-linear, Deep Sets and the Lin-former, and −0.05/−0.07 for the Transformer.
  - The count-only control c0 gives −0.13 to −0.59 (the Transformer +0.15/+0.27), so the shared response is not the count.
  - c0 also keeps ΣpT fixed and lowers m by about 2 GeV. Therefore this test does not separate the ungroomed mass from ΣpT, which both rise equally under G_in and G_out (D1: JEDI-linear uses a ΣpT shortcut).
  - H1 is excluded for these models on W/Z. H0 (ungroomed m and/or ΣpT) is consistent. H2 (count only) is excluded, since c0 is 6–26× smaller than the shared response.
- **hls4ml top.** On hls4ml top jets the hls4ml networks separate G_in from G_out (D −0.17 to −0.37, r_SD 0.32–0.70).
- **JetClass W/Z.**
  - G_out alone lowers the W/Z log-odds by 1.25–1.54. Neither reference does this: MLP(m) gives −0.17/−0.19 and MLP(m_SD) gives 0. c0 gives −0.01 to −0.54.
  - G_in lowers it by a further 0.19–0.55 (r_SD 0.41–1.39). The size of this extra drop is in the range MLP(m_SD) predicts from the m_SD change, and largest for ParT_full (1.15 W, 1.39 Z).
  - H13v's reading "count only" (H2: G_in ≈ G_out ≈ c0) is excluded. Both edits move the score at least 2.8× more than c0 (ParT_full W: G_out −1.54 vs c0 −0.46), and G_in more than G_out.
  - H0 is excluded (D ≠ 0, Holm 0.004). H1 holds for the G_in − G_out part only: about half to two-thirds of the G_in response is shared with G_out, which grooming would have removed.
- **Top tagging.**
  - On JetClass Tbqq and Top Landscape tops, ParT, PN and LorentzNet lower the top log-odds more under G_in than G_out (D −0.37 to −0.55).
  - The references do not predict this sign:
    - MLP(m_SD) gives 0.00 on JetClass and +0.11 on Top, since the +12 GeV m_SD shift moves tops near the peak.
    - The m_SD + τ32 top window keeps more tops under G_in than G_out (+0.05/+0.06).
  - The top networks' G_in − G_out difference is therefore not the groomed-mass response of these references. One candidate is the change in the mass and shape of the prong the particles join (for top, often the b–W split), which the networks may read (D1/H23: W-pair mass). This was not tested here.
  - PFN gives D −0.04, the same as its permuted-weight null (−0.03). EFN gives D +0.35: it loses more under G_out, which spreads energy flow farther from the prongs.
- **Efficiency at fixed QCD rate.** The G_in − G_out differences in ε_S at 5 % QCD are small for every network (|Δ| ≤ 0.06). For comparison, MLP(m_SD) gives −0.14 (JetClass W) and −0.68 (hls4ml W). Most of the score change moves signal and QCD jets together.

## Issues and limitations
- **Not a pure m_SD edit.** G_in sits about 0.7 R_g beyond prong F. It raises that prong's own mass and changes the two-prong shape (τ21, D2), not only m_SD. G_out raises τ21/τ32 through wide-angle radiation. A control that keeps the prong masses fixed while moving m_SD does not exist in this design.
- **Selection.** The matched set selects jets where soft drop finds a split with R_g ≥ 0.02 and where G_out is groomed and G_in kept. It is 56–83 % of signal jets and 36–71 % of QCD jets. QCD quark jets are under-represented (36 % on hls4ml).
- **Placement radii.** The radii follow the prong geometry (r* ≈ 1.4 R_g), not 0.5–0.75 as planned: matching the ungroomed Δm is impossible otherwise (see Procedure).
- **hls4ml slot cap.** 64 slots: the added particles evict the softest originals in jets at the cap, so the seen count rises by 2.1–2.9, not 3.
- **The r_SD ratio** is undefined where MLP(m_SD)'s D is ≈ 0 (JetClass t) and of opposite sign on Top. There the top windows are the relevant references.
- **Nulls.** Top PFN permuted gives D −0.03 (Holm 0.004), the size of trained PFN's −0.04, so PFN's D is at its null level. JEDI redrawn has |D| < 0.005.
- **Debug attempts.** One debug fix (d-h49-sum: empty bootstrap for the undefined MLP(m) ratio). The design change after the first smoke (adaptive d and the G_out direction search) raised Top acceptance from 39 % to 57 %.

## Proposed claim
D21 (new, 0.45): under soft particles with matched count, ΣpT and ungroomed mass change, the taggers split by dataset.
- **hls4ml W/Z taggers** (JEDI-linear, Deep Sets, Lin-former, Transformer) do not distinguish particles soft drop keeps from particles it removes: r_SD = −0.07 to 0.05 of the groomed-mass reference.
- **JetClass ParT/ParticleNet W/Z taggers** respond to both. The extra response to kept particles is 0.4–1.4× the groomed-mass reference, and the response to removed particles is 1.3–1.5 log-odds, which no mass reference gives.
- **Top taggers** respond more to kept particles, in the direction opposite to the m_SD-based references.

Refines D10: "count only" (H13v) is excluded for JetClass W/Z, since c0 = −0.01 to −0.54 vs G_out −1.25 to −1.54.

Figure: `results/physics/h49_groomed.png`.
- Left panel: one JetClass W jet in the (Δη, Δφ) plane, showing both edits, the prongs and the circle r*. Both edits add Δm = +7.2 GeV; Δm_SD is +7.3 GeV for G_in and 0.0 for G_out.
- Right panel: Δs under G_in, G_out and c0 for every trained network and both MLP references.
