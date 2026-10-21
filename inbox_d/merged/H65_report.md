# H65 report: soft-contamination density scan across four datasets (loop D, round D13)

Node H65, new D claim (links D21, D23 v2, D5). Script `interp/h65_pileup_dose.py`. Results in `results/physics/h65_{hls,jc,top,qg}_{obs,nets}.{npz,json}` and `h65_summary.json`. Figure: `results/physics/h65_pileup.png`. Status: done (exploratory).

## Conclusion

We sprayed soft particles of 0.5–1 GeV uniformly over each jet, as diffuse pileup would, and found that the taggers lose their signal calls in different ways on different datasets.

- **hls4ml JEDI-linear (W/Z):** reacts only to the plain-mass increase. The same particles stacked on the jet axis, which adds no mass, leave its output unchanged. It responds like a plain-mass window.
- **JetClass ParT_full / ParticleNet_full and Top Landscape ParT:** respond to how many particles were added. The on-axis stack, with no mass added, moves them as much as the spread-out haze or more.
- **Quark–gluon networks:** respond mainly to particles at wide angle, as a mass or energy-flow tagger does.

No trained network follows the groomed (soft-drop) or area-subtracted mass. Those keep about 40 % W efficiency at a density of 20 GeV per unit area, while the networks keep almost none.

Still uncertain:
- In JetClass "full" models the added particles are labelled neutral hadrons, so particle-ID inputs may carry part of the count response.
- The on-axis control is unphysical for ParticleNet and LorentzNet, so their count readings are not usable.

## What was done

**Question.** As the density of uniform soft contamination grows, how do each network's calls change? Is the change like that of a plain-mass window, a groomed-mass window, an area-subtracted-mass window, or a particle-count tagger?

**Edit (pu_ρ).** Massless "ghost" particles are added uniformly over the disc of radius R about the stored jet axis in (η, φ):
- R = 0.8 for hls4ml, JetClass and Top; R = 0.4 for quark–gluon (QG).
- Each ghost has pT drawn uniformly in [0.5, 1] GeV.
- The ghost count is Poisson with mean ρ·πR²/0.75 GeV, so the added pT per unit η–φ area is ρ, for ρ ∈ {2, 5, 10, 20} GeV. At ρ = 20 that is 53.6 ghosts (R = 0.8) or 13.4 ghosts (R = 0.4).
- Ghosts are neutral hadrons in JetClass (charge 0, d0 = dz = 0) and pid 130 in QG.
- One master set is drawn at ρ = 20. A lower ρ keeps each ghost with probability ρ/20, so the sets are nested and every model sees the same positions.
- Particles are then pT-sorted and cut to the model's slot limit: hls4ml 64, JetClass 128, Top 200, QG none.
- All inputs are taken about the stored, unedited axis, so the edit does not move the input axis.

Mean ghosts kept per jet after the slot cut, for ρ = 2/5/10/20:
- hls4ml: 4.4 / 10.2 / 17.2 / 21.4 of 5.4 / 13.4 / 26.8 / 53.7 (40–81 % kept). The 64-slot cut drops ghosts and pushes out the softest real particles.
- JetClass, Top, QG: all ghosts kept (53.3 of 53.6 at ρ = 20 in JetClass).

**Controls**, all using the same ghost pT values at each ρ:
- **c1_ρ** (summed pT fixed): one massless particle carrying the summed ghost pT, placed 0.002 rad from the axis. Count +1; mass change small. For JetClass W at ρ = 10, Δm = +1.4 GeV against +24.5 GeV for pu_10.
- **c2_ρ** (count and summed pT fixed): the same ghosts, same number and same pT values, all placed 0.002 rad from the axis. The mass change is the same as c1, so this isolates "number of extra particles" from "mass added".
- **in_ρ / out_ρ** (C3, the H58v r_g split as a density scan). Each base jet is reclustered with C/A and soft drop (z_cut 0.1, β 0) to find its two prongs and their separation R_g.
  - in_ρ: the same ghosts placed uniformly within R_g/2 of a prong centre. C/A attaches them to the prong and soft drop keeps them.
  - out_ρ: the same ghosts placed in the disc more than 1.1 R_g from both prong centres. Soft drop removes them.
  - Count and summed pT are matched between the two. Plain-mass change: in ≪ out. Groomed-mass change: in > out. Example, JetClass W at ρ = 10: Δm +3.4 vs +26.9 GeV; Δm_SD +3.3 vs +0.7 GeV.
  - Valid on jets with a soft-drop split, R_g ≥ 0.02, both prong centres inside R, and at least 5 % of the disc outside the core: hls4ml 80 %, JetClass 87 %, Top 87 %, QG 65 % of jets.
- **Nulls:** JEDI-linear with redrawn (untrained) weights; ParT_full with every weight tensor permuted (seed 7) and ParT_full untrained; Top PFN permuted; QG part_kin permuted.

**Samples** (random rows, never first-k):
- hls4ml: `jedi_core.sample(2000, 6501)` jets, 10,000 jets, 5 classes.
- JetClass: first test file per class, QCD (ZJetsToNuNu), Zqq, Wqq, Tbqq, 2,000 random rows each (seed 6502).
- Top Landscape test: 2,000 top + 2,000 QCD (seed 6503).
- QG Pythia test (data_qg): 2,000 quark + 2,000 gluon (seed 6504).

**Models:**
- hls4ml: JEDI-linear official n64 3-feature.
- JetClass: ParT_full, ParT_kin, PN_full, PN_kin (released).
- Top Landscape: ParT_kin, PN_kin (H2 weaver trainings), LorentzNet (released weights), PFN, EFN.
- QG: part_kin, pnet_kin (H5 weaver), pfn_k, efn_k (H2 energyflow), lorentznet (released).

**Scores:** log-odds of the signal class against QCD.
- hls4ml: log p(c) − log(p(q) + p(g)).
- JetClass: logit_c − logit_QCD.
- Top: top log-odds.
- QG: log p(quark) − log p(gluon), with quark as the signal.

**References.** All are computed on the particles the model sees (after the slot cut) and set to 50 % signal efficiency on unedited jets:
- **win_m:** plain jet mass m in the shortest interval that holds 50 % of unedited signal jets. For QG: m < the median quark m.
- **win_msd:** the same construction with the soft-drop mass m_SD (C/A, z_cut 0.1, β 0).
- **win_msub:** the same construction with the area-subtracted mass m_sub = |p_J − ρ_seen A_μ|. This is 4-vector area subtraction in the style of Soyez, with ρ_m = 0. A_μ is the 4-momentum of a uniform disc of unit pT density and radius R about the stored axis. ρ_seen = ρ × (ghost pT kept by the slot cut / ghost pT added), i.e. the true density of the ghosts in the input. The same ρ is subtracted under the controls, where the added pT is not uniform, so win_msub is not a reference for c1/c2/in.
- **LR(N):** a 20-quantile-bin histogram likelihood ratio of the particle count N the model sees. It uses a 2-fold cross-fit on the unedited sample, and N beyond the last bin edge takes the edge value.
- **EFN:** the trained IRC-safe (infrared- and collinear-safe) energy flow network, on Top and QG.

**Metrics**, per tagger × signal class × condition:
- Fraction of calls changed. A call is "score above the unedited 50 % signal-efficiency threshold", evaluated on the task's signal and QCD jets.
- Signal efficiency ε_S(ρ): the fraction of signal jets still called signal.
- Mean Δ log-odds (Δs).
- 95 % bootstrap intervals over jets: B = 1000, the same resamples for every tagger.
- Slope of Δs against ρ through the origin, per GeV of ρ.
- pu − c1 and pu − c2 at each ρ:
  - paired bootstrap of the difference in Δs (H0: difference = 0, normal approximation from the bootstrap standard error);
  - McNemar exact binomial p on the jets whose call changes under one edit but not the other.
- **r_core** = (calls changed by out_ρ) / (calls changed by in_ρ), and **r_Δs** = Δs(out)/Δs(in). Interpretation:
  - an m_SD window gives r_core ≈ 0.1–0.3;
  - a plain-mass window gives r_core ≈ 1.5–9;
  - a count-only reader gives 1.
- Nearest reference by L2 distance between curves over ρ ∈ {2, 5, 10, 20}, with bootstrap frequency of each reference being nearest. Computed (a) on the calls-changed curve, as planned, and (b) on the ε_S(ρ) curve.
- Holm correction over models × ρ × classes, all datasets pooled:
  - family "pu effect": H0 Δs = 0, 100 tests;
  - family "pu vs control": 200 McNemar tests.

**Reading the calls-changed metric.** At a 50 % signal-efficiency threshold the calls-changed fraction saturates at half of the signal share of the task rows: 0.25 for JetClass and Top, 0.167 for hls4ml. It reaches that ceiling when every signal call is lost while QCD calls are unchanged. A mass window reaches lower values non-monotonically, because it also pulls QCD jets into the window. For this reason ε_S(ρ) and Δs carry most of the reading below. The calls-changed nearest-reference result is reported but it is dominated by this saturation.

## Unedited accuracy on the samples used (check)

All models reproduce their known values.

| dataset | model | this sample | known |
|---|---|---|---|
| hls4ml | JEDI-linear | 5-class acc 0.8152 | 0.8092 (260k test), 0.8091 (H49) |
| hls4ml | JEDI redrawn | 0.200 | chance |
| JetClass | ParT_full | per-class acc QCD 0.772, Zqq 0.670, Wqq 0.811, Tbqq 0.958 | 0.783 / 0.689 / 0.805 / 0.956 |
| JetClass | ParT_kin | 0.713 / 0.544 / 0.687 / 0.897 | 0.718 / 0.550 / 0.702 / 0.912 |
| JetClass | PN_full | 0.763 / 0.632 / 0.789 / 0.947 | 0.764 / 0.652 / 0.784 / 0.949 |
| JetClass | PN_kin | 0.688 / 0.506 / 0.665 / 0.899 | 0.693 / 0.528 / 0.680 / 0.913 |
| JetClass | nulls | permuted and untrained ParT_full: per-class acc 0.00–0.01, W-vs-QCD AUC 0.51–0.56 | — |
| Top | ParT / PN / LorentzNet / PFN / EFN | AUC 0.9867 / 0.9874 / 0.9885 / 0.9816 / 0.9789 | 0.9858 / 0.9858 / 0.9870 / 0.9819 / 0.975–0.979 |
| Top | PFN permuted | AUC 0.705, acc 0.50 | — |
| QG | part_kin / pnet_kin / pfn_k / efn_k / lorentznet | AUC 0.9012 / 0.9009 / 0.8891 / 0.8841 / 0.9135 | 0.9013 / 0.9015 / ~0.88–0.90 / 0.8835 / 0.917 |
| QG | part_kin permuted | AUC 0.547 | — |

## Numbers

### Observable changes (mean over signal + QCD jets of the task)

| dataset, class | edit | Δm (GeV) | Δm_SD (GeV) | Δm_sub (GeV) | ΔN |
|---|---|---|---|---|---|
| JetClass W | pu 2 / 5 / 10 / 20 | +5.5 / +13.1 / +24.5 / +44.5 | +0.5 / +1.3 / +2.5 / +5.2 | −0.2 / −0.3 / −0.5 / −0.8 | +5.3 / +13.4 / +26.8 / +53.4 |
| JetClass W | c1_10 = c2_10 | +1.4 | +0.5 | −30.6 (subtraction of a non-uniform addition) | +1 / +26.8 |
| JetClass W | in10 / out10 | +3.4 / +26.9 | +3.3 / +0.7 | — | +26.8 / +26.8 |
| hls4ml W | pu 10 | +30.7 | +0.1 | −1.7 | +16.1 (64-slot cut) |
| Top t | pu 10 | +18.6 | +6.9 | −0.2 | +26.8 |
| QG | pu 10 (R 0.4) | +3.3 | +0.3 | −0.1 | +6.7 |

### Signal efficiency ε_S(ρ) at the fixed unedited threshold

Values at ρ = 2 / 5 / 10 / 20 (unedited = 0.50).

| dataset, class | trained networks | win_m | win_msd | win_msub | LR(N) |
|---|---|---|---|---|---|
| hls4ml W | JEDI 0.12 / 0.00 / 0.00 / 0.00 | 0.06 / 0.02 / 0.01 / 0.01 | 0.49 / 0.49 / 0.49 / 0.47 | 0.20 / 0.14 / 0.12 / 0.12 | 0.44 / 0.23 / 0.03 / 0.00 |
| JetClass W | ParT_full 0.31 / 0.19 / 0.10 / 0.02; ParT_kin 0.30 / 0.14 / 0.04 / 0.00; PN_full 0.26 / 0.09 / 0.01 / 0.00 | 0.44 / 0.25 / 0.05 / 0.00 | 0.49 / 0.48 / 0.46 / 0.43 | 0.48 / 0.46 / 0.43 / 0.39 | 0.51 / 0.31 / 0.03 / 0.00 |
| JetClass t | ParT_full 0.43 / 0.34 / 0.20 / 0.05 | 0.48 / 0.43 / 0.33 / 0.14 | 0.49 / 0.47 / 0.41 / 0.29 | 0.50 / 0.49 / 0.48 / 0.47 | 0.59 / 0.67 / 0.60 / 0.50 |
| Top t | ParT 0.44 / 0.36 / 0.24 / 0.08; PFN 0.41 / 0.30 / 0.13 / 0.01; EFN 0.46 / 0.41 / 0.30 / 0.15 | 0.49 / 0.45 / 0.33 / 0.13 | 0.49 / 0.47 / 0.40 / 0.25 | 0.49 / 0.49 / 0.48 / 0.47 | 0.52 / 0.50 / 0.27 / 0.01 |
| QG q | part_kin 0.45 / 0.39 / 0.29 / 0.16; pfn_k 0.47 / 0.42 / 0.36 / 0.27; efn_k 0.47 / 0.42 / 0.35 / 0.23 | 0.47 / 0.42 / 0.33 / 0.14 | 0.50 / 0.50 / 0.50 / 0.49 | 0.49 / 0.49 / 0.49 / 0.49 | 0.46 / 0.38 / 0.26 / 0.09 |

The trained networks keep the QCD efficiency at 0.00–0.03 at every ρ. They do not gain QCD calls. A plain-mass window does: JetClass W ε_B goes from 0.09 to 0.20 at ρ = 10.

### Δ log-odds under pu_10 and under the count-matched control c2_10

For c2, Δs(c2) = Δs(pu) − (pu − c2). Brackets are 95 % bootstrap intervals over jets. "Share" is c2/pu.

| model (dataset, class) | Δs(pu_10) | pu − c2 at ρ 10 | Δs(c2_10) (share of pu) | Δs slope per GeV of ρ |
|---|---|---|---|---|
| JEDI-linear (hls4ml W) | −2.20 [−2.30, −2.11] | −2.18 [−2.29, −2.09] | −0.02 (0.01) | −0.135 [−0.141, −0.129] |
| JEDI-linear (hls4ml Z) | −1.75 [−1.85, −1.65] | −1.78 [−1.88, −1.68] | +0.03 (−0.02) | −0.114 [−0.120, −0.108] |
| JEDI-linear (hls4ml t) | +2.52 [+2.42, +2.62] | +2.76 [+2.66, +2.86] | −0.24 | +0.156 [+0.150, +0.162] |
| ParT_full (JetClass W) | −2.48 [−2.54, −2.42] | +0.43 [+0.39, +0.47] | −2.91 (1.17) | −0.199 [−0.203, −0.194] |
| ParT_full (JetClass Z) | −2.45 [−2.52, −2.39] | +0.08 [+0.05, +0.12] | −2.53 (1.03) | −0.191 |
| ParT_full (JetClass t) | −0.50 [−0.57, −0.42] | +0.34 [+0.30, +0.38] | −0.84 (1.68) | −0.044 |
| ParT_kin (JetClass W) | −2.34 [−2.40, −2.28] | −1.42 [−1.46, −1.37] | −0.92 (0.39) | −0.194 |
| ParT_kin (JetClass Z) | −2.12 | −1.14 [−1.19, −1.08] | −0.98 (0.46) | −0.174 |
| PN_full (JetClass W) | −2.94 [−3.00, −2.87] | +1.06 [+1.03, +1.10] | −4.00 (1.36) | −0.246 |
| ParT (Top t) | −0.60 [−0.65, −0.56] | +0.13 [+0.11, +0.16] | −0.73 (1.22) | −0.047 [−0.050, −0.043] |
| PFN (Top t) | −0.53 [−0.59, −0.48] | +0.49 [+0.45, +0.52] | −1.02 (1.9) | −0.058 |
| EFN (Top t) | −0.10 [−0.13, −0.07] | +0.02 [−0.02, +0.05] | −0.12 (IRC safe: c2 ≡ c1) | −0.003 [−0.006, +0.000] |
| part_kin (QG q) | −0.65 [−0.67, −0.64] | −0.55 [−0.56, −0.53] | −0.10 (0.16) | −0.056 |
| pnet_kin (QG q) | −0.62 [−0.64, −0.61] | −0.51 [−0.53, −0.50] | −0.11 (0.18) | −0.054 |
| pfn_k (QG q) | −0.53 [−0.54, −0.52] | −0.31 [−0.32, −0.30] | −0.22 (0.42) | −0.047 |
| lorentznet (QG q) | −0.49 [−0.50, −0.47] | +0.04 [+0.03, +0.04] | −0.53 (1.08) | −0.037 |
| LR(N) (JetClass W), count reference | −1.08 [−1.11, −1.04] | 0 by construction | −1.08 (1.00) | −0.087 |

Controls that are out of distribution, not used for a count reading. The c2 control is about 27 particles within 0.004 rad of each other.
- PN_kin (JetClass): Δs(c2_10) about −24 (W) and −31 (Z).
- PN (Top): c2 +1.8.
- LorentzNet (Top): c2 −3.2.

c1 (one particle with the summed pT, on the axis) at ρ = 10:
- JEDI-linear W: Δs −0.04.
- JetClass ParT_full W: −0.04.
- QG part_kin: −0.03.
- QG lorentznet: −0.24.
- The Δs(pu) − Δs(c1) values are listed in `h65_summary.json` under `vs_ctrl`.

**McNemar, pu vs c1 and pu vs c2.** 176 of 200 tests have Holm p < 0.05. The 24 that do not:
- pu vs c2 for JetClass ParT_full Z (ρ 2–10) and t (ρ 5–20);
- PN_full and PN_kin (JetClass), at saturation;
- Top ParT (ρ 2, 5), Top PN (ρ 2), Top LorentzNet (ρ 2);
- QG lorentznet (both controls at low ρ).

For ParT_full and Top ParT, then, the full edit and the count-matched control change the same calls.

**pu effect, H0 Δs = 0.** 96 of 100 tests have Holm p < 0.05. The four that do not: JetClass t ParT_kin ρ 20, Top EFN ρ 20, Top LorentzNet ρ 20, Top PN ρ 5. All four lie where the mean Δs changes sign along ρ.

### Core split: r_core (calls) and r_Δs (log-odds) at ρ = 10

| tagger | r_core (calls) | r_Δs [95 % CI] |
|---|---|---|
| plain-mass window, JetClass W / hls4ml W / Top / QG | 3.5 / 1.3 / 2.6 / 19 | — |
| m_SD window, JetClass W / hls4ml W / Top / QG | 0.15 / 0.07 / 0.62 / 0.45 | — |
| LR(N) | 1.00 | 1.00 |
| JEDI-linear W / Z (hls4ml) | 1.30 / 1.27 (win_m 1.31 / 1.45) | 3.22 [3.06, 3.40] / 2.69 [2.55, 2.82] |
| ParT_full JetClass W / Z / t | 0.71 / 0.80 / 0.56 | 0.53 [0.52, 0.54] / 0.58 [0.57, 0.59] / 0.18 [0.13, 0.22] |
| ParT_kin JetClass W / Z | 1.10 / 1.15 | 1.18 [1.15, 1.21] / 1.34 [1.30, 1.39] |
| PN_full JetClass W | 0.97 (saturated) | 0.75 [0.74, 0.76] |
| Top ParT / PFN | 0.74 / 1.31 | 0.39 [0.35, 0.42] / 0.49 [0.45, 0.53] |
| Top EFN | 8.0 | 3.3 [2.9, 3.7] |
| QG part_kin / pnet_kin / pfn_k / efn_k / lorentznet | 4.6 / 2.4 / 2.8 / 14 / 0.91 | 4.6 / 4.0 / 2.7 / 24 / 0.91 |

At ρ = 2, JEDI-linear's r_core is 5.77 against 4.96 for win_m, and 2.47 vs 2.09 at ρ = 5. It tracks the plain-mass window at every ρ.

### Nearest reference (bootstrap frequency)

On the ε_S(ρ) curve:
- JEDI-linear W/Z: win_m (1.00).
- JetClass W networks: win_m (1.00).
- JetClass Z networks: win_m 0.64–0.84, LR(N) 0.16–0.36.
- JetClass t networks: win_m (1.00).
- Top ParT, PN, PFN, LorentzNet: EFN (0.68–0.98).
- QG part_kin: win_m (0.78).
- QG pnet_kin, pfn_k, lorentznet: efn_k (0.64–1.00).
- win_msd and win_msub are never nearest for a trained network.

On the calls-changed curve, as in the plan, the result follows the saturation ceiling:
- JEDI-linear and JetClass W/Z: win_msub (1.00).
- JetClass t: win_msd.
- Top: win_msd or EFN.

This is because win_msub's calls-changed fraction rises slowly toward the same 0.17–0.25 that the networks reach by saturating. Its ε_S (0.39–0.47 at ρ = 20) is far from theirs (0.00–0.08).

### Nulls

Mean Δs at ρ = 10:
- JEDI redrawn: 0.00.
- ParT_full permuted: W 0.00, Z +0.38.
- ParT_full untrained: W −0.38, against trained −2.48.
- QG part_kin permuted: −0.01.
- Top PFN permuted: +0.49, opposite in sign to the trained PFN's −0.53. Summing per-particle features gives a count response by architecture alone.

The calls-changed fractions of the nulls (0.04–0.45) are not a sensitivity measure. Their scores are nearly constant, so the 50 % threshold sits inside a dense cluster of scores.

## Reading

Competing hypotheses and their predictions, against what was measured:

- **H_mass** (the network reads the plain mass). Predicts:
  - c1 and c2 responses ≈ 0;
  - r_core like the plain-mass window (> 1);
  - ε_S(ρ) like win_m, plus some QCD jets gaining calls.

  Measured: JEDI-linear W/Z matches every prediction except the QCD gain, since its QCD efficiency at the threshold is ≈ 0. The c2 share is 0.01, r_core is within 0.01–0.81 of win_m's at each ρ, and win_m is the nearest ε_S curve with frequency 1.00. QG part_kin and pnet_kin partly match: c2 share 0.16–0.18 and r_Δs 4.0–4.6, with nearest ε_S curve win_m or efn_k. H_mass is excluded for JetClass ParT_full / PN_full and Top ParT / PFN, whose c2 share is 1.0–1.9.

- **H_groomed** (the network reads m_SD). Predicts ε_S staying near 0.45–0.49 to ρ = 20 and r_core ≈ 0.1–0.6. Excluded for every trained network: their ε_S at ρ = 10 is 0.00–0.36 (except QG lorentznet 0.37) against win_msd 0.40–0.50. The r_Δs of 0.18–0.58 for JetClass ParT_full and Top ParT/PFN shows that ghosts near the prongs move these networks more than ghosts outside the core, at equal count and summed pT.

- **H_subtracted** (the network reads area-subtracted mass with the true ρ). Predicts ε_S ≈ 0.39–0.49 at ρ = 20. Excluded for every trained network on the ε_S curve.

- **H_count** (the network reads the particle multiplicity, as LR(N) does). Predicts a c2 share ≈ 1 and r_core ≈ 1. Consistent with JetClass ParT_full (share 1.03–1.68, McNemar pu vs c2 not significant for Z/t), PN_full (share 1.36) and Top ParT (share 1.22, McNemar n.s. at ρ 2–5). Partly consistent with ParT_kin, where the share is 0.39–0.46: about 40–60 % of its response needs the ghosts spread out. Excluded for JEDI-linear (share 0.01).

These readings depend on the dataset, which is the same split D21 and D23 v2 found with fixed three-particle edits: the hls4ml net follows the plain mass, while JetClass/Top networks respond to soft particles beyond mass. H65 adds:
- the density scale: ε_S falls to about 0.1–0.3 by ρ = 2–5 GeV for JetClass W/Z and hls4ml, about 0.25–0.36 at ρ = 5–10 for Top;
- the count-at-fixed-mass control at each density;
- QG, where the networks respond mostly to wide-angle ghosts.

Overlap: H13 scanned ρ 5–40 on three datasets with a softest-quartile pT spectrum. H65 differs in fixed 0.5–1 GeV ghosts, the QG dataset, area-subtracted and LR(N) references, count-matched controls at every ρ, and the in/out core split.

Caveats:
1. In JetClass "full" models the ghosts are labelled neutral hadrons. Their count response may run through the particle-ID inputs, which ParT_kin (share 0.39) does not have. A charged-ghost or photon-ghost arm would separate this.
2. hls4ml's 64-slot cut keeps 40–81 % of the ghosts (fewer at higher ρ) and drops real soft particles, so the hls4ml edit is "contamination plus truncation".
3. c2 places ~27 particles within 0.004 rad, which is unphysical for kNN graph (ParticleNet) and Minkowski-product (LorentzNet) models. Their c2 numbers are not interpreted.
4. win_msub subtracts the nominal density even under the non-uniform controls.
5. Single sample per dataset. No fresh-seed rerun yet.

## Explainer entry (for site/explain.py)

**Question.** If a jet is covered by a uniform haze of soft particles, like pileup or underlying event, do neural taggers degrade like a plain-mass cut, like a groomed-mass cut, or like a particle counter?

**Physics background.** Diffuse contamination of transverse-momentum density ρ adds pT ≈ ρ·A to a jet of area A. It raises the jet mass roughly as
$\delta m^2 \approx \rho\,A\,p_{T,J}\,\langle\Delta R^2\rangle$.
Grooming with soft drop removes soft wide-angle particles: it declusters the C/A tree until
$\min(p_{T1},p_{T2})/(p_{T1}+p_{T2}) > z_\text{cut}$.
Area subtraction (Soyez) removes the expected contribution:
$p^\mu_\text{sub} = p^\mu_J - \rho A^\mu$.
A count-based tagger responds to $\Delta N \approx \rho A/\langle p_T\rangle$, independent of where the particles land.

**Method.**
- Add Poisson-distributed ghosts with $p_T\in[0.5,1]$ GeV uniformly in $|\Delta\eta,\Delta\phi| < R$, at $\rho\in\{2,5,10,20\}$ GeV per unit area.
- Hold the input axis fixed.
- Compare four versions of the same ghosts: spread out ($pu$); the same summed pT as one collinear particle ($c1$); the same particles stacked on the axis ($c2$: count up, mass ~unchanged); placed only where soft drop keeps or removes them ($in$/$out$).
- Measure:
  - the signal efficiency at a fixed threshold, $\varepsilon_S(\rho)$;
  - the mean log-odds change $\Delta s$;
  - the count share $\Delta s(c2)/\Delta s(pu)$;
  - the core ratio $r_{\Delta s} = \Delta s(out)/\Delta s(in)$.

**Result (figure `h65_pileup.png`).** Left: a JetClass W jet with 27 ghosts at ρ = 10 GeV. Its mass goes from 80 to 101 GeV while its soft-drop mass stays at 79 GeV. Right: ParT_full's W efficiency falls from 50 % to 10 % at ρ = 10. Stacking the same ghosts on the axis, which changes the mass by about 1 GeV, gives the same fall. Soft-drop and area-subtracted mass windows keep 43–46 %.

**Checks.**
- Unedited accuracies reproduce the known values.
- The IRC-safe EFN gives c2 = c1 exactly.
- Permuted and untrained nulls give Δs ≈ 0, except the permuted PFN, whose sum-pooled features carry a count response by architecture.

**Physical significance.** The hls4ml tagger's pileup sensitivity is the plain-mass shift. The JetClass and Top transformer and graph taggers lose signal efficiency in proportion to the number of soft particles added, wherever those land, and a grooming or area correction applied to the mass would not remove that.

Glossary additions:
- **ghost (soft-contamination particle):** a massless particle of 0.5–1 GeV added uniformly over the jet area.
- **ρ (rho):** the added pT per unit η–φ area, in GeV.
- **m_sub:** area-subtracted mass, |p_J − ρA_μ|.
- **A_μ:** the 4-momentum of a unit-density uniform disc over the jet area.
- **count share:** Δs(c2)/Δs(pu), the fraction of a network's response reproduced by the same particles stacked on the axis.
- **r_core / r_Δs:** response to ghosts outside the soft-drop core divided by the response to the same ghosts inside it.
- **ε_S(ρ):** the fraction of signal jets still called signal at the unedited 50 % threshold.

## Run record

| job | what | wall | peak memory |
|---|---|---|---|
| d-h65-hls | jax env, 1 CPU / 4 Gi | 870 s | 2.65 GiB |
| d-h65-gpu | GPU, jc → top → qg | 14 min | — |
| d-h65-obs-jc / -top / -qg | conda, 1 CPU, 6 / 4 / 4 Gi | 87 / 160 / 64 s | 1.15 / 1.51 / 0.55 GiB |
| d-h65-sum2 | jax env, 1 CPU / 2 Gi | 124 s | — |

- An earlier summary job (d-h65-sum) used bootstrap-floor p-values, which cannot pass Holm across 100 tests. It is superseded by d-h65-sum2, which uses normal-approximation p-values from the bootstrap standard error.
- Node issues during the run:
  - k8s-gpu-6.ucsc.edu: a CephFS D-state stall on the first hls smoke pod; the pod was deleted.
  - k8s-chase-ci-01.calit2.optiputer.net: "GPU is lost" admission error. The GPU job was relaunched with that node added to the exclusion list in the piped YAML only; part_job.sh itself was not edited. Suggest adding both nodes to the job scripts' exclusion lists.
- Smoke runs (`_smoke` files, 40 per class) were used only for debugging.
