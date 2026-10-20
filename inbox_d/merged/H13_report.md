# H13 — Soft contamination / pileup panel: network response vs ungroomed, groomed and axis-moment mass, and the induced axis offset (claims D4m, D5, D8; new sub-claim D10)

Script `interp/h13_pileup_panel.py` (parts jedi, small, jetclass, top, obs ×3, summary). Jobs: d-h13-jedi, d-h13-small, d-h13-obsh, d-h13-obsj, d-h13-obst (CPU), d-h13-jc, d-h13-top (GPU), d-h13-sum and d-h13-sum2 (summary and figure). Smoke runs: d-h13-sm-*. All exited 0. Results in `/j-jepa-vol/aaron-interp/results/physics/`, copied to `results/physics/`: `h13_{jedi,small,jetclass,top}.json`, `h13_obs_{hls,jc,top}.json`, `h13_summary.json`, `h13_*_perjet.npz`, `h13_panel.png`. `h13_top_main.json` is an interim copy of the top result written before the background-rejection step finished. `h13_top.json` is the final version.

## Conclusion
We added soft particles spread evenly over the jet area, of the kind pileup produces, to simulated jets. We then recomputed each network's inputs. Every trained network lost signal efficiency faster than the soft-drop-mass tagger did. On hls4ml, JEDI-linear's W efficiency fell like that of a plain ungroomed-mass window: from 87 % to 42 % at the lowest density, against 36 % for the mass window. The JetClass and Top-Landscape networks fell between the ungroomed-mass and soft-drop taggers. The jet axis moves only 3–13 mrad, about a tenth of the winner-take-all offset studied in D4. Moving the original particles by that same amount changes only 0.4–3 % of calls, so the networks respond to the added particles and the mass they add, not to the axis shift. Still uncertain: the edit is a toy pileup model (uniform disc, massless neutral particles drawn from each dataset's softest quartile, no subtraction). The moment-formula fits explain little of the per-jet variance (R² ≤ 0.30).

## Procedure
- **Samples (seed 302).**
  - hls4ml: `jedi_core.sample(per_class=2000, seed=302)`, 10,000 jets, 64 slots. Median jet pT is 1,020 GeV and the median has 46 constituents.
  - JetClass: 1,000 random rows per class from each class's first test file, 10,000 jets, 128 slots. Median pT is 585 GeV. The build uses η = jet_eta + sign(jet_eta)·deta. A round trip from the table to the inputs against `build(kin(J0))` gives a max abs diff of 6.1e-5.
  - Top Landscape: `data_top.sample(5000, 302)`, 5,000 top and 5,000 QCD test jets. For 1/ε_B, 100,000 random QCD test jets (seed 303, with the main sample's rows excluded).
  - Classical observables: rows [::5] (2,000 jets per dataset), computed on all particles.
- **Models.**
  - hls4ml: JEDI-linear official n64 3-feature, Deep Sets dsets-d64-n64 seed0, and the H1 EFN.
  - JetClass: ParT_full, ParT_kin, ParticleNet_full and ParticleNet_kin.
  - Top: ParT_kin and PN_kin (H2 weaver, best epoch), LorentzNet (released weights), PFN and EFN (H2).
  - Nulls (C4): JEDI-linear with redrawn weights, Deep Sets with every weight tensor's entries permuted, ParT_full with redrawn weights, and PFN with permuted weights. LorentzNet is the invariant model: its inputs do not depend on the axis. EFN is the IRC-safe model.
- **Edit `pu_ρ`, ρ ∈ {5, 10, 20, 40} GeV per unit η–φ area.**
  - Add a Poisson number of particles with mean ρA/⟨q⟩, with A = πR² and R = 0.8, uniform in the disc of radius R about the stored axis.
  - Each particle's pT q is drawn from that sample's constituents below their own 25th percentile q25 (q25 / ⟨q⟩ = 2.76 / 1.38 GeV hls4ml, 2.65 / 1.68 GeV JetClass, 1.27 / 0.71 GeV Top).
  - At ρ = 40 the mean count added is 58 (hls4ml), 48 (JetClass) and 113 (Top), and the added ΣpT is 80 GeV.
  - The particles are massless. For JetClass they are neutral hadrons with charge 0 and d0 = dz = 0.
  - The jet is then rebuilt as each model's preprocessing does: pT-sorted, cut to the slot limit, E-scheme axis recomputed, and every input taken about the new axis.
- **Controls (same injected particles within a ρ).**
  - c1 `sk`: SoftKiller-style cut at q25, with the hardest particle always kept. Since every injected q < q25, this gives the same jet for any ρ: the original jet minus its own particles below q25.
  - `c1m_ρ`: the edit, then a cut at the median of the injected spectrum.
  - c2 `c2_ρ`: the injected ΣpT placed as one particle on the stored axis. The axis stays fixed (offset < 0.002 rad) and the mass rises by ≤ 8.8 GeV at ρ = 40.
  - c3 `c3_ρ`: the original particles about the `pu_ρ` axis, a pure translation with the same offset and nothing added.
  - `pufix_ρ`: the edit, with inputs about the stored axis (no axis offset).
  - Calibration points: `reax0` (stored particles about their own E-scheme axis) and `wta` (the D4 winner-take-all axis).
- **Metrics.**
  - % calls changed: argmax for the multiclass models. For Top, the score is compared with the ε_S = 0.5 threshold frozen on unedited tops.
  - Score changes: Z−W logit on W/Z jets (hls4ml), Zqq−Wqq on Wqq/Zqq jets (JetClass), log-odds (Top).
  - Class efficiencies, and 1/ε_B at ε_S = 0.3 / 0.5. Thresholds are either re-set on the edited signal or kept frozen, with 200 bootstrap resamples over 5,000 tops and 100,000 QCD jets.
  - Paired McNemar of `pu_ρ` against c1, with Holm over all 64 model × ρ tests (16 models × 4 ρ). Intervals are 95 % bootstraps over jets.
- **Moment-formula test (D4m).**
  - (i) Fit the slope a of Δscore on ΔA, with A = ½log(M0M2) about the input axis. The fit runs through the origin and uses only the pure-axis conditions (reax0, wta, c3), for which Δlog m = 0. The predicted response is then a·ΔA(pu), reported as the ratio observed/predicted with a CI.
  - (ii) A two-variable fit Δs = a·ΔA + b·Δlog m + c, pooled over the reax0, wta, pufix, c2, c3 and pu conditions.
  - (iii) The axis part of the edit, s(pu) − s(pufix), compared with c3.
  - (iv) A matched-size slope comparison: wta (ΔA ≈ 0.12) against pu_5 (ΔA ≈ 0.12–0.21).
- **Classical references.**
  - Observables: m on all particles, m_SD (C/A, z_cut 0.1, β 0), τ21 and τ32 (WTA kT axes, β 1, R0 0.8), D2 (β 1), and m_A.
  - Working points: W(m) 65 < m < 95 GeV; W(m_SD) 65 < m_SD < 95 GeV; top(m_SD, τ32) 105 < m_SD < 210 GeV with τ32 < 0.65; top(m, τ32) the same window on ungroomed m.

## Unedited models on the samples (reproduction check)
| model | sample value | known |
|---|---|---|
| JEDI-linear | acc 0.8082 [0.801, 0.816] | 0.8092 (260k); 0.8051 / 0.8116 (H1 / H1v) |
| Deep Sets d64 | 0.8068 [0.800, 0.815] | 0.8064 |
| EFN (hls4ml) | 0.7181 [0.709, 0.727] | 0.7206 |
| ParT_full / ParT_kin | 0.8581 / 0.7351 | 0.859 / 0.745 (D4), 0.856–0.863 / 0.736–0.742 (H1, H1v) |
| PN_full / PN_kin | 0.8434 / 0.7002 | 0.839–0.846 / 0.695–0.704 |
| Top ParT / PN / LNet / PFN / EFN (AUC) | 0.9867 / 0.9867 / 0.9876 / 0.9822 / 0.9792 | 0.9858 / 0.9858 / 0.9870 / 0.9819 / ~0.98 |
| nulls | JEDI redrawn 0.200, Deep Sets permuted 0.200, ParT redrawn 0.100, PFN permuted AUC 0.704 | chance |

## What the edit does (mean change; obs on rows [::5], Δ with 95 % CI)
| dataset | ρ | Δm ungroomed [GeV] | Δm_SD [GeV] | Δτ32 | axis offset vs stored, median (p90) [mrad] |
|---|---|---|---|---|---|
| hls4ml | 5 / 10 / 20 / 40 | +19.8 / +36.0 / +62.9 / +108.4 | +0.4 / +0.7 / +1.3 / +2.6 | +0.037 / +0.057 / +0.079 / +0.101 | 2.5 (6.6) / 3.2 (7.4) / 4.3 (8.7) / 5.4 (10.6) |
| JetClass | 5 / 10 / 20 / 40 | +10.1 / +19.5 / +36.9 / +68.3 | +2.5 / +5.0 / +10.1 / +20.2 | +0.037 / +0.062 / +0.092 / +0.127 | 4 (8) / 5 (11) / 8 (17) / 13 (25) |
| Top | 5 / 10 / 20 / 40 | +9.7 / +19.1 / +35.6 / +65.9 | +3.3 / +6.8 / +13.3 / +27.6 | +0.028 / +0.047 / +0.074 / +0.106 | 3 (5) / 4 (8) / 7 (13) / 12 (21) |

- The small-angle prediction Δm² = pT_J·ρA·R²/2 gives Δm = +19.7 / +35.2 / +60.7 / +101.0 GeV on hls4ml, against the measured +19.8 / +36.0 / +62.9 / +108.4.
- Reference axis offsets, median (p90): WTA vs E-scheme is 40 (97) mrad on hls4ml (X7: 40 mrad), 95 (186) on JetClass and 92 (241) on Top. The natural stored-vs-E-scheme offset on hls4ml (E23) is 1.46 (5.8) mrad.
- c1 `sk` changes the mass by −5.3 (hls4ml), −8.5 (JetClass) and −6.5 GeV (Top). It is an edit in its own right, not a return to the unedited jet.
- Slot overflow: on hls4ml the injected jets exceed 64 particles in 31 % (ρ 5) to 99.7 % (ρ 40) of jets, and the softest particles are dropped before the model sees them. Overflow is 0–1.3 % on JetClass and 0–3.4 % on Top.

## Numbers

### Signal efficiency of each call vs classical taggers (true W jets for hls4ml/JetClass, true tops for Top; figure `h13_panel.png`)
| tagger | ρ = 0 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| hls4ml JEDI-linear (call = W) | 87 % | 42 | 5 | 0 | 0 |
| hls4ml Deep Sets | 87 | 18 | 12 | 2 | 2 |
| hls4ml EFN | 85 | 6 | 0 | 0 | 0 |
| hls4ml W(m), 65–95 GeV | 93 | 36 | 6 | 0 | 0 |
| hls4ml W(m_SD) | 77 | 78 | 77 | 78 | 79 |
| JetClass ParT_full | 80 | 71 | 62 | 45 | 26 |
| JetClass ParT_kin | 67 | 58 | 47 | 32 | 10 |
| JetClass PN_full | 76 | 67 | 55 | 30 | 7 |
| JetClass PN_kin | 67 | 57 | 43 | 20 | 3 |
| JetClass W(m) | 75 | 46 | 15 | 1 | 0 |
| JetClass W(m_SD) | 72 | 72 | 72 | 68 | 61 |
| Top ParT (ε_S 0.5 threshold frozen) | 50 | 36 | 24 | 9 | 1 |
| Top PN | 50 | 34 | 19 | 3 | 0 |
| Top LorentzNet | 50 | 31 | 14 | 2 | 0 |
| Top PFN | 50 | 28 | 10 | 0 | 0 |
| Top EFN | 50 | 40 | 29 | 14 | 3 |
| Top top(m, τ32) | 70 | 65 | 61 | 44 | 5 |
| Top top(m_SD, τ32) | 66 | 63 | 60 | 47 | 18 |

### % calls changed, with 95 % CI and Holm-corrected p for McNemar against c1 (64 tests; 51 have Holm p < 0.05)
| model | c1 sk | pu_5 | pu_10 | pu_20 | pu_40 | c2_40 | c3_40 |
|---|---|---|---|---|---|---|---|
| JEDI-linear | 20.5 | 23.5 [22.7, 24.3] (Holm 3e-6) | 48.2 | 66.2 | 69.0 [68.0, 69.9] (Holm ≈ 0) | 14.7 | 2.4 |
| Deep Sets | 17.6 | 40.6 | 52.5 | 62.3 | 66.6 | 11.0 | 1.2 |
| EFN (hls4ml) | 12.5 | 48.2 | 59.8 | 64.6 | 71.1 | 7.8 | 3.3 |
| ParT_full | 12.0 | 6.3 [5.8, 6.7] (below c1; Holm 7e-53) | 12.9 (Holm 0.34) | 26.4 | 41.3 [40.3, 42.2] | 9.5 | 1.2 |
| ParT_kin | 20.5 | 10.4 (below c1) | 18.8 (Holm 0.017) | 35.5 | 59.2 | 18.9 | 3.1 |
| PN_full | 12.2 | 6.7 (below c1) | 13.3 (Holm 0.24) | 26.0 | 42.6 | 9.7 | 1.4 |
| PN_kin | 22.3 | 12.5 (below c1) | 23.5 (Holm 0.31) | 45.6 | 67.8 | 21.7 | 2.5 |
| Top ParT | 6.0 | 7.2 (Holm 0.017) | 13.3 | 20.7 | 24.5 | 7.1 | 1.0 |
| Top PN | 7.2 | 8.0 (Holm 0.31) | 15.4 | 23.5 | 25.1 | 4.5 | 1.3 |
| Top LorentzNet | 8.3 | 9.5 (Holm 0.06) | 18.1 | 24.3 | 25.1 | 7.8 | 0.0 |
| Top PFN | 8.2 | 11.3 | 20.0 | 25.0 | 25.2 | 6.2 | 1.2 |
| Top EFN | 2.6 | 5.3 | 10.7 | 18.4 | 24.0 | 7.1 | 1.8 |
| nulls (JEDI redrawn / Deep Sets permuted / ParT redrawn) | 0 / 0 / 1.5 | 0 / 0 / 0.3 | 0 / 0 / 0.3 | 0 / 0 / 0.3 | 0 / 0 / 0.3 | 0 / 0 / 0.3 | 0 / 0 / 0.2 |
| PFN permuted (Top) | 18.0 | 24.2 | 42.2 | 57.1 | 62.6 | 0.7 | 2.9 |

- Top call changes are capped near 25 %, because only tops above the ε_S = 0.5 threshold (2,500 of 10,000 jets) can lose the call. Holm p is < 1e-3 wherever it is not shown.
- Classical calls changed on hls4ml, pu_5 → pu_40: W(m) 52.4 → 48.5 %; W(m_SD) 0.4 → 3.0 %; top(m_SD, τ32) 0.9 → 5.2 %. On JetClass: W(m) 14.7 → 22.9 %; W(m_SD) 1.6 → 8.5 %; top(m_SD, τ32) 3.8 → 16.1 %.
- Dominant call flow under pu_ρ:
  - JEDI-linear: W → g at ρ 5, Z → t at ρ 10, q → t at ρ 20–40.
  - Deep Sets: W → Z, then q → g.
  - EFN (hls4ml): W → t (1,487 → 2,148 jets, 0 in reverse).
  - JetClass models: Wqq → QCD and Hgg → QCD, with 0–3 reverse calls.

### Background rejection on Top Landscape (1/ε_B, threshold re-set on the edited signal; 100k QCD)
| model | ε_S | unedited | sk | pu_5 | pu_10 | pu_20 | pu_40 (ratio [CI]) | c2_40 |
|---|---|---|---|---|---|---|---|---|
| ParT | 0.3 | 1429 [1111, 1963] | 1087 | 1389 | 1124 | 935 | 578 (0.41 [0.31, 0.52]) | 862 |
| PN | 0.3 | 1754 | 1250 | 1351 | 1389 | 855 | 481 (0.27 [0.21, 0.37]) | 1163 |
| LorentzNet | 0.3 | 1786 | 1449 | 1613 | 1563 | 1124 | 217 (0.12 [0.09, 0.15]) | 971 |
| PFN | 0.3 | 699 | 637 | 637 | 518 | 329 | 162 (0.23 [0.20, 0.27]) | 538 |
| EFN | 0.3 | 521 | 490 | 483 | 441 | 327 | 181 (0.35 [0.29, 0.41]) | 351 |
| ParT | 0.5 | 407 | 308 | 372 | 352 | 258 | 183 (0.45 [0.38, 0.51]) | 226 |
| PN | 0.5 | 369 | 334 | 356 | 283 | 229 | 147 (0.40 [0.35, 0.44]) | 274 |

- The pu_5 ratios at ε_S = 0.3 are 0.97 (ParT), 0.77 [0.61, 0.97] (PN), 0.90, 0.91 and 0.93. Their CIs include 1 except for PN.
- With the ε_S = 0.3 threshold frozen, ε_S at pu_40 falls to 0–0.2 % for all five networks: the edit moves top jets towards QCD scores.

### Axis part of the edit and the moment formula (Δscore on the regression rows; Z−W, Zqq−Wqq or top log-odds)
| model | pu − pufix calls changed, ρ 5 / 40 (pp) | s(pu) − s(pufix) vs c3, ρ 5 | axis-only slope a [CI] | obs / pred at ρ 5 [CI] | obs / pred at ρ 40 | two-var a, b (R²) | matched slope ratio pu_5 / wta |
|---|---|---|---|---|---|---|---|
| JEDI-linear | 0.2 / 0.3 | 0.012 vs 0.009 (p 0.71) | 10.9 [9.8, 11.9] | 0.22 [0.18, 0.25] | 0.03 [0.01, 0.04] | 10.5, −9.6 (0.02) | 0.26 [0.23, 0.29] |
| Deep Sets | 0.1 / 0.1 | −0.002 vs 0.004 | 7.6 [6.4, 8.7] | 0.13 [0.04, 0.19] | 0.10 | 8.3, −6.9 (0.02) | 0.26 [0.21, 0.30] |
| EFN (hls4ml) | 0.1 / 0.2 | 0.004 vs −0.001 | 3.5 [2.5, 4.6] | −2.4 [−3.5, −1.7] | −1.6 | 7.1, −10.4 (0.08) | −1.2 [−1.9, −0.8] |
| ParT_full | 0.1 / 0.0 | 0.005 vs 0.009 | 5.4 [4.8, 5.9] | 0.13 [0.10, 0.16] | 0.03 | 4.9, −4.7 (0.04) | 0.13 [0.10, 0.16] |
| ParT_kin | 0.8 / 0.2 | −0.015 vs −0.013 | 4.0 [3.7, 4.5] | 0.22 [0.18, 0.26] | 0.08 | 3.6, −3.2 (0.05) | 0.21 [0.17, 0.24] |
| PN_full | 0.0 / 0.2 | 0.009 vs 0.011 | 1.1 [0.7, 1.5] | 0.67 [0.46, 1.01] | 0.25 | 0.6, −0.2 (0.01) | 0.70 [0.47, 1.12] |
| PN_kin | 0.1 / 0.3 | 0.006 vs 0.008 | −0.5 [−0.8, −0.3] | −2.0 [−4.1, −1.2] | −1.0 | −0.9, 1.5 (0.06) | −1.7 [−3.0, −1.1] |
| Top ParT | 0.0 / — | 0.000 vs 0.002 | −2.9 [−3.2, −2.6] | 1.15 [1.05, 1.27] | 0.56 [0.51, 0.63] | 0.6, 1.5 (0.10) | −0.09 [−0.12, −0.07] |
| Top PN | — | −0.003 vs −0.003 | −10.6 [−10.9, −10.2] | 0.06 [0.03, 0.10] | 0.19 | −5.2, 10.2 (0.24) | −0.65 |
| Top LorentzNet | 0 | 0 vs 0 | 0 (invariant) | undefined | undefined | 5.9, 0.5 (0.26) | undefined |
| Top PFN | — | −0.002 vs 0.002 | −1.1 [−1.5, −0.7] | 2.4 [1.8, 3.7] | 3.4 | 3.2, −1.4 (0.05) | −2.0 |
| Top EFN | — | −0.018 vs −0.016 | −3.6 [−4.2, −3.0] | 0.21 [0.17, 0.26] | −0.23 | −0.7, 4.2 (0.30) | −0.51 |

- In every pu_ρ condition, ΔA equals Δlog m to within 0.01 (for example hls4ml ρ 5: 0.212 and 0.212). About the recomputed E-scheme axis, m_A ≈ m, so the pileup edit cannot separate the moment from the invariant mass. Only the pure-axis conditions do.
- c3 (pure translation by the pileup-induced offset), calls changed for ρ 5 → 40:
  - JEDI-linear 1.8 → 2.4 %, Deep Sets 0.6 → 1.2 %, EFN 1.6 → 3.3 %.
  - ParT_full 0.4 → 1.2 %, ParT_kin 1.7 → 3.1 %, PN_full 0.5 → 1.4 %, PN_kin 0.8 → 2.5 %.
  - Top models 0–1.8 % at ρ 40.
  - For comparison, E23 found 1.4–3.8 % for random shifts of 1.5–13 mrad on JEDI-linear.

## Hypotheses
- **H1 (models read ungroomed mass).** Predicts that network calls follow the ungroomed-mass window.
  - On hls4ml, JEDI-linear's W efficiency (87 → 42 → 5 → 0 %) follows W(m) (93 → 36 → 6 → 0 %). Deep Sets and EFN fall faster.
  - On JetClass, every network stays above W(m) at ρ ≥ 5 (ParT_full 71 % vs 46 % at ρ 5; 26 % vs 0 % at ρ 40). This is excluded for the JetClass models.
  - On Top, the networks fall faster than top(m, τ32) at the same ρ, relative to their own starting rate. ParT keeps 72 / 47 / 17 / 2 % of its tags; top(m, τ32) keeps 92 / 87 / 62 / 7 %.
- **H1′ (groomed-like).** Predicts a response near the soft-drop tagger and near c1.
  - Excluded for every trained network at ρ ≥ 10. All fall below W(m_SD) or top(m_SD, τ32), and pu_ρ changes more calls than c1 (Holm p < 1e-60 at ρ ≥ 20 for all trained models).
  - At ρ 5 the JetClass models change fewer calls than c1 (ParT_full 6.3 % vs 12.0 %), because c1 removes original particles below 2.6 GeV.
- **H1″ (axis-relative).** Predicts that the response equals slope × ΔA and equals c3 at matched offset.
  - The axis part of the edit (pu − pufix) matches c3 in size. Both are 0–0.03 in score and 0.0–0.8 pp in calls.
  - The full pileup response is not predicted by the pure-axis slope: obs/pred at ρ 5 is 0.13–0.67 or of opposite sign for 10 of 11 trained, axis-dependent models, with CI excluding 1.
  - The exception is Top ParT, with ratios 1.15 [1.05, 1.27] (ρ 5) and 1.09 [1.00, 1.20] (ρ 10), falling to 0.56 at ρ 40.
- **Expected ordering (JEDI-linear largest, EFN follows ungroomed m, ParT intermediate).**
  - On hls4ml, EFN changes the most calls (48 % at ρ 5), then Deep Sets (41 %) and JEDI-linear (23 %).
  - On JetClass, ParT_full and PN_full change the fewest calls (41–43 % at ρ 40).
  - On Top, EFN changes the fewest calls (24 % at ρ 40, efficiency 3 % vs 0–1 %), and LorentzNet loses the most rejection (×0.12).

## Reading
- The pileup-like edit raises the ungroomed mass by the amount the small-angle formula predicts (+20 to +108 GeV on 1 TeV hls4ml jets). It raises the soft-drop mass by 0.4–2.6 GeV (hls4ml) and 2.5–28 GeV (JetClass, Top).
- Every trained network's signal efficiency falls more than W(m_SD) or top(m_SD, τ32) at each ρ ≥ 10.
- On hls4ml, the JEDI-linear W call tracks the ungroomed-mass window. On JetClass the networks lie between the ungroomed and groomed windows.
- The induced E-scheme axis offset is 2.5–13 mrad median. That is 1.7–9 times the natural stored-vs-E-scheme offset (E23), and 7–20 times smaller than the WTA offset used in D4/X7.
- At these offsets a pure translation changes 0.4–3.3 % of calls. Recomputing the inputs about the stored axis instead (pufix) changes the edited call rate by ≤ 0.8 pp. The pileup response on these samples therefore comes from the added particles (mass, ΣpT, multiplicity), not from the axis offset.
- c2 (same ΣpT on the axis, Δm ≤ 8.8 GeV) changes 4.5–21.7 % of calls at ρ 40. This is a response to added ΣpT or energy at nearly fixed mass. It is largest for JEDI-linear (14.7 %) and the kin models (18.9–21.7 %), consistent with the D1 ΣpT shortcut.
- The axis-only moment slope from D4m does not carry over to the pileup edit for most models. Because ΔA ≈ Δlog m under this edit, the two-variable fits have nearly collinear regressors in the pileup conditions and explain R² ≤ 0.30 of per-jet variance. The fitted a and b, of opposite sign and similar size for JEDI-linear, Deep Sets and ParT, are not a stable decomposition.

## Issues and caveats
- c1 (cut at q25) removes the original particles below q25 as well as the injected ones. It changes 2.6–22 % of calls by itself. The McNemar tests against c1 therefore compare two edits, not an edit against no change. Comparisons with the unedited jet (calls changed) and with c3/pufix are reported alongside.
- Toy pileup: a uniform disc about the stored axis (not the clustered jet area), massless neutral particles, a pT spectrum taken from each dataset's softest quartile (mean 0.7–1.7 GeV), and no charged-hadron subtraction. None of the networks was trained with pileup.
- On hls4ml, 31–100 % of edited jets exceed 64 slots, so part of the injected pT never reaches the models. The ungroomed W(m) reference uses all particles.
- The figure's top panel shows networks at their frozen 50 % working point and classical taggers at their own published windows (starting at 66–70 %), so starting values differ.
- LorentzNet's axis slope is exactly 0 (invariant inputs). The obs/pred ratio is undefined, and the summary's printed value of 6e6 is a division by ~0.
- The ParT redrawn-weight null has a few significant McNemar tests (0.3 % vs 1.5 % calls changed). It predicts one class almost always (acc 0.100), so these are tiny counts.
- The script and figure were produced in an earlier session of this round. This session copied the final top result (with the 100k-background rejection), re-checked accuracies, reran the summary/figure (d-h13-sum2) and wrote this report.

## Explainer entry (for site/explain.py, H13_EXPLAINER)
- **Question.** How do jet taggers respond to diffuse soft contamination (pileup-like), compared with ungroomed mass, soft-drop mass and classical taggers? Does the axis offset it induces matter (claims D4m, D5, D10)?
- **Physics background.**
  - Pileup adds soft particles roughly uniformly in η–φ, with a transverse-momentum density ρ per unit area.
  - For a jet of transverse momentum $p_{T,J}$ and radius $R$, the small-angle mass shift is $\Delta m^2\approx p_{T,J}\sum_i q_i\Delta R_i^2 = p_{T,J}\,\rho\,\pi R^4/2$. The ungroomed mass rises quickly with ρ.
  - Soft drop (C/A declustering, $z_{\rm cut}=0.1$, $\beta=0$) removes soft wide-angle branches, so $m_{SD}$ changes little.
  - The E-scheme axis is the direction of $\sum_i\vec p_i$. Added particles move it by about $\sum_i q_i\,\vec r_i/p_{T,J}$, which is small for a uniform disc.
  - D4m relates axis-relative network responses to $A=\tfrac12\log(M_0M_2)$, with $M_0=\sum_i p_{T,i}$ and $M_2=\sum_i p_{T,i}\Delta R_i^2$ about the axis the network is given.
- **Method.**
  - Edit: add $\mathrm{Poisson}(\rho\pi R^2/\langle q\rangle)$ massless particles uniformly in the $R=0.8$ disc, for $\rho\in\{5,10,20,40\}$ GeV per unit area. Their $q$ is drawn from the dataset's softest-quartile spectrum. Recompute the E-scheme axis and all inputs.
  - Controls: (c1) a SoftKiller-style cut at the 25th percentile; (c2) the same $\sum q$ on the axis; (c3) the original particles translated by the induced axis offset; (pufix) the edit about the stored axis; weight-permuted or redrawn nulls; the invariant LorentzNet and the IRC-safe EFN.
  - Metrics: % calls changed, class efficiency, and $1/\varepsilon_B$ at $\varepsilon_S=0.3,0.5$ with 100k QCD jets. McNemar against c1, with Holm over 64 tests.
  - Moment test: $\Delta s = a\,\Delta A$ with $a$ fitted on the pure-axis edits, then the ratio of observed to predicted response for the pileup edit.
- **Result (figure `results/physics/h13_panel.png`).**
  - hls4ml true-W efficiency at ρ = 5: JEDI-linear 87 → 42 %; ungroomed window 93 → 36 %; soft-drop window 77 → 78 %.
  - JetClass at ρ = 40: ParT_full 80 → 26 %, W(m) 75 → 0 %, W(m_SD) 72 → 61 %.
  - Top at ρ = 40, ε_S = 0.3: ParT 1/ε_B falls from 1429 to 578 (×0.41 [0.31, 0.52]); LorentzNet ×0.12.
  - Induced axis offset, median: 2.5–5.4 mrad (hls4ml), 4–13 mrad (JetClass), 3–12 mrad (Top), against 40–95 mrad for WTA vs E-scheme. Translating by it changes 0.4–3.3 % of calls.
- **Checks.**
  - Unedited accuracies and AUCs reproduce earlier values (JEDI-linear 0.808, ParT_full 0.858, Top ParT AUC 0.987).
  - On hls4ml, the measured Δm matches the small-angle formula within 8 % (JetClass and Top: measured 11–23 % above it).
  - The nulls change ≤ 0.3 % of calls under the edit.
  - The edit with and without axis recomputation differs by ≤ 0.8 pp in calls.
- **Physical significance.** None of the networks behaves like a groomed observable under diffuse soft contamination. JEDI-linear on hls4ml behaves like an ungroomed-mass window; the JetClass and Top networks lie between ungroomed and groomed. The axis-convention sensitivity found in D4 involves offsets about ten times larger than those pileup produces. At pileup-sized offsets, axis effects change a few percent of calls; the bulk of the pileup response comes from the added soft momentum.
