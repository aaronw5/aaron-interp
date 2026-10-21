# H64: boost along the jet axis at fixed invariant mass. Does each tagger respond like a reader of the jet mass or of the jet's angular size?

Loop D, round D13. Claims: new D claim (links D1, D4m, A4, A14; extends H12/D9). Script `interp/h64_axis_boost_panel.py`. Status: done (exploratory).

## Conclusion

Boosting a jet along its own direction keeps its mass fixed but makes it narrower, the way a higher-momentum jet looks. Taggers trained on JetClass and Top Landscape (ParT, ParticleNet, LorentzNet) react to this boost with about a quarter or less of their reaction to an equal pure narrowing of the jet at fixed pT. At the size of this response they behave mostly like readers of the invariant mass, not of the angular size. Taggers trained on hls4ml, and ParT/ParticleNet trained on quark–gluon jets, react much more (up to about 90 % of the narrowing response when pT is lowered). Two points remain open. A reference classifier that sees only the mass and the jet pT also reacts strongly to the boost, so a large reaction can come from using the jet pT and does not by itself show that a network reads angles. And most boosted Top and all boosted quark–gluon jets lie outside the pT range those models were trained on.

## Question and terms

- **B(λ), the boost**: an exact Lorentz boost of every constituent four-vector along the jet three-momentum (E-scheme sum of the constituents). The rapidity is chosen so that |P_jet| → λ|P_jet|. The invariant jet mass m stays fixed (|Δm|/m ≤ 4e-12 on every jet with m > 1 GeV), the jet pT is multiplied by λ, and every constituent's angle to the axis shrinks by about 1/λ. The stored jet axis is boosted with the same transformation. λ ∈ {0.8, 0.9, 1.1, 1.25}. In logarithms, B changes (log m, log pT, log angle) by (0, +log λ, −log λ).
- **P(λ), pT-only control**: every four-vector × λ at fixed (η, φ). m × λ, pT × λ, angles fixed. Changes (+log λ, +log λ, 0).
- **A(λ), angle-only control**: every (Δη, Δφ) about the stored axis × 1/λ at fixed constituent pT. m × ≈1/λ, pT fixed. Changes (−log λ, 0, −log λ). A is the comparison that changes the angular size by the same factor as B but also changes the mass.
- **R(λ), equal-size random control**: a rotation of all constituents about the axis, with the same pT-weighted mean displacement as B on the same jet. m, pT and every infrared- and collinear-safe shape stay fixed.
- To first order B = P + A. **Predictions computed before measuring**:
  - A reader of the invariant mass alone gives Δ_B = 0.
  - A reader of the angular size at fixed pT normalisation, or of m/pT (opening angle ≈ 2m/pT), gives Δ_B = Δ_A and Δ_P = 0.
  - A reader of pT alone gives Δ_A = 0 and Δ_B = Δ_P.
  - A reader of m and pT together gives Δ_B = Δ_P + Δ_A, which is neither 0 nor Δ_A.
- **Response Δ_X,k(λ)**: for class k, the one-vs-rest log-odds is l_k = log p_k − log(1 − p_k), from the network's softmax. Δ_X,k(λ) is the mean over jets of true class k of l_k(edited by X) − l_k(unedited). It has a 95 % bootstrap interval (1,000 resamples over jets, stratified by class).
- **R_B,k = Δ_B,k / Δ_A,k**: the network's response to the boost divided by its response to an angle-only narrowing of the same factor. 1 means it responds like an angular-size (or m/pT) reader; 0 means it responds like an invariant-mass reader. It is reported only where the 95 % interval of Δ_A,k excludes 0.
- **R_m,k = Δ_B,k / Δ_P,k**: the boost response divided by the pT-only response.
- **R_B^w**, the class-weighted summary per model: Σ_k s_k Δ_B,k / Σ_k |Δ_A,k| with s_k = sign(Δ_A,k). Classes are weighted by the size of their A response. R_m^w is built the same way with P. A summary ratio is reported only when its denominator is ≥ 0.02 log-odds.
- **50 % working point (WP)**: for class k, the threshold on l_k that keeps 50 % of true-class-k unedited jets. A "call" is l_k above that threshold. "Calls changed" is the share of all jets whose call flips under an edit.
- **McNemar test**: an exact two-sided binomial test on the jets whose call flips under one edit but not the other (same jets, paired).
- **Holm correction**: a step-down family-wise correction over all model × λ × class tests in a family.
- **C2ST (classifier two-sample test)**: a gradient-boosted classifier tries to tell edited jets from real jets. AUC 0.5 means it cannot tell them apart. Here the real jets are reweighted to the edited jets' pT distribution, so the test asks whether a boosted jet looks like a real jet at its new pT.
- **dR_prongs**: the distance in (Δη, Δφ) between two pT-weighted 2-means axes. They start at the hardest particle and at the particle that maximises pT·ΔR² from it, then run 10 iterations.

## Procedure

**Samples**. Accuracies are of each unedited model on these jets.
- hls4ml: `interp.jedi_core.sample(per_class=2000, seed=6401)`, 10,000 jets (5 classes g, q, W, Z, t; 64 particle slots, massless constituents at η = Δη). Reference jets: 3,000 other test rows per class (rng 6411), 15,000 jets.
- JetClass: 5,000 random rows per class of each class's first test file (`h1_irc_panel.jc_sample`, seed 6402; random indices, never first-k). These were split at random (rng 6412) into 2,000 per class evaluated (20,000 jets) and 3,000 per class reference (30,000). Built with η = jet_eta + sign(jet_eta)·Δη.
- Top Landscape test split: 2,000 top + 2,000 QCD random rows (rng 6403). Reference: 10,000 + 10,000 disjoint rows (rng 6413).
- Quark–gluon (QG) test split: 2,000 quark + 2,000 gluon random rows (rng 6404). Reference: 10,000 + 10,000 disjoint rows (rng 6414).

**Models**
- hls4ml: JEDI-linear official 3-feature n64. Two small hls4ml nets, `dsets-d64-n64` (Deep Sets) and `tf-2b-d64-n64-e100` (transformer), seed 0.
- JetClass: ParT_full, ParT_kin, ParticleNet (PN)_full, PN_kin.
- Top: ParT_kin, PN_kin (H2 weaver checkpoints), LorentzNet (released), PFN, EFN (H2).
- QG: ParT_kin, PN_kin (H2 weaver checkpoints), LorentzNet (released QG model; an addition to the plan).

**Nulls**
- Untrained copies: every weight tensor redrawn from a Gaussian with the trained tensor's mean and spread, the project convention. These are jedi_n64_untr, dsets_untr, tf_d64_untr, ParT_full_untr (JetClass), PN_full_untr, Top ParT_kin_untr, Top EFN_untr and QG part_kin_untr.
- Weight-permuted copies: the entries of every weight tensor permuted. These are jedi_n64_perm, dsets_perm, tf_d64_perm, JetClass ParT_kin_perm and PN_kin_perm, Top ParT_kin_perm, PN_kin_perm and LorentzNet_perm, and QG part_kin_perm and pnet_kin_perm.
- The JetClass PN nulls give unbounded outputs (mean |Δ_B| up to 6e4 log-odds). They are reported but left out of the H0_null test.

**References** (sklearn MLP 32-32, trained on the disjoint reference jets of the same dataset):
- MLP(m, pT) on [log m, log pT]: reads invariant mass and pT, with no angular input.
- MLP(m/pT) on [log(m/pT)]: an angular-size reader.
- MLP(dR, pT) on [log dR_prongs, log pT].

**Tests** (trained models, per model × λ × class, 284 tests per family)
- H0_inv: Δ_B = 0.
- H0_ang: Δ_B = Δ_A, paired.
- H0_null: Δ_B equals Δ_B of the bounded null of the same dataset that gives the largest p (paired, same jets).
- McNemar of WP call changes, B vs A and B vs R.
- Holm is applied globally over all four datasets within each family.

**Training-range check**: the share of edited jets whose jet pT leaves each dataset's range:
- hls4ml: 0.5–99.5 % quantiles of the unedited sample.
- JetClass: 500–1000 GeV.
- Top: 550–650 GeV.
- QG: 500–550 GeV.

Then a pT-reweighted C2ST of B jets against the real reference jets. Features: log m, constituent count N, pT-weighted mean ΔR, log pT. 30 pT bins. Coverage = share of B jets in bins that hold ≥ 5 real jets.

**Jobs**: d-h64-jedi (CPU, jax, 1,178 s), d-h64-small (CPU, conda, 451 s), d-h64-gpu (GPU: Top 432 s, QG 366 s, JetClass 855 s), d-h64-sum and d-h64-sum2 (summary and figures).

## Unedited accuracy on these samples (reference in brackets)

| dataset | model | accuracy [95 % CI] | known |
|---|---|---|---|
| hls4ml | JEDI-linear n64 | 80.66 % [79.9, 81.5] | 80.68 % (E26), 81.00 % (H12) |
| hls4ml | Deep Sets | 80.19 % | 80.64 % (test.json) |
| hls4ml | transformer | 81.04 % | 81.44 % (test.json) |
| JetClass | ParT_full / ParT_kin | 85.83 / 74.16 % | 85.75 / 73.88 % (H12) |
| JetClass | PN_full / PN_kin | 83.84 / 70.40 % | 84.09 / 70.04 % (H12) |
| Top | ParT / PN / LorentzNet / PFN / EFN | 93.70 / 94.00 / 94.33 / 92.93 / 92.27 % (AUC 0.985 / 0.985 / 0.986 / 0.979 / 0.976) | 93.98 / 94.02 / 94.15 / 92.85 / 92.21 % (H12) |
| QG | ParT / PN / LorentzNet | 84.10 [83.0, 85.2] / 83.55 / 85.42 % (AUC 0.911 / 0.909 / 0.925) | 82.63 / 82.69 % (H12), 84.42 % (released LorentzNet) |
| refs | MLP(m,pT) / MLP(m/pT) / MLP(dR,pT) | hls4ml 74.6 / 65.9 / 50.0 %; JetClass 33.2 / 27.3 / 23.2 %; Top 89.1 / 89.2 / 67.8 %; QG 69.3 / 69.2 / 53.3 % | – |

Every unedited model reproduces its known accuracy within 0.4 points except the QG models. Those are 0.8–1.5 points above on this 4,000-jet sample, in the same direction for all three models, which is consistent with a sample fluctuation shared across models. The nulls sit at chance accuracy (8.7–20 % on 5 or 10 classes, 49–57 % on binary).

## Edits as applied (medians over jets)

- B: |Δm|/m ≤ 4e-12 on every jet with m > 1 GeV. One QG jet has m = 0 (a single massless constituent), so its ratio is undefined. Jet pT × λ exactly. Median dR_prongs −20 % (λ 1.25) to +24 % (λ 0.8). m_A = √(M0·M2) about the boosted stored axis has median change ≤ 0.14 %, but changes per jet by up to 1.4 % (hls4ml), 2.7 % (Top) and 40 % (JetClass, λ 0.8).
- P: m and pT × λ, angles unchanged.
- A: median m × 1/λ (per jet within 1 % on hls4ml and Top; up to 48 % instead of 25 % on some JetClass jets with massive constituents), pT unchanged, dR_prongs × 1/λ.
- R: median m, pT and dR_prongs unchanged. Per-jet |Δm|/m ≤ 0.7 % (hls4ml), 1.4 % (Top), and up to 19 % on JetClass λ 0.8, where the matched displacement is large.
- Figure `results/physics/h64_wjet_boost.png` shows one hls4ml W jet (m = 79.4 GeV, pT = 1173 GeV): B at λ = 1.25 gives pT 1466 GeV with m 79.4 GeV, and A at λ = 1.25 gives m 63.5 GeV with pT fixed. Both bring dR_prongs from 0.163 to 0.130.

## Results

### 1. Class-weighted ratio R_B^w (response to the boost / response to angle-only narrowing), 95 % bootstrap

| model | λ 0.8 | λ 0.9 | λ 1.1 | λ 1.25 | R_m^w at 0.9 / 1.1 |
|---|---|---|---|---|---|
| hls4ml JEDI-linear | 1.05 | 0.86 [0.84, 0.88] | 0.35 [0.34, 0.36] | 0.38 | 0.92 / 0.43 |
| hls4ml Deep Sets | 1.05 | 0.61 [0.60, 0.62] | 0.28 | 0.48 | 0.95 / 0.44 |
| hls4ml transformer | 1.19 | 0.93 [0.91, 0.94] | 0.21 | 0.23 | 1.38 / 0.52 |
| JetClass ParT_full | 0.24 | 0.24 [0.23, 0.25] | 0.16 [0.15, 0.16] | 0.07 | 0.10 / −0.12 |
| JetClass ParT_kin | 0.09 | 0.25 [0.23, 0.27] | −0.05 [−0.08, −0.03] | −0.01 | 0.13 / −0.05 |
| JetClass PN_full | 0.20 | 0.24 | 0.16 | 0.04 | 0.07 / −0.26 |
| JetClass PN_kin | 0.14 | 0.25 | −0.03 | 0.01 | 0.15 / 0.00 |
| Top ParT | 0.21 | 0.20 [0.19, 0.22] | −0.05 [−0.07, −0.03] | −0.01 | 0.09 / −0.16 |
| Top PN | 0.45 | 0.45 | 0.20 | 0.17 | −0.03 / −0.38 |
| Top LorentzNet | 0.25 | 0.29 [0.27, 0.31] | 0.06 [0.03, 0.08] | 0.09 | −0.03 / −0.11 |
| Top PFN / EFN | 0.98 / 0.96 | 0.97 / 0.95 | 0.99 / 0.99 | 0.98 / 0.97 | P response exactly 0 |
| QG ParT | 0.70 | 0.74 [0.72, 0.76] | 0.83 [0.81, 0.84] | 0.86 | −3.5 / −4.1 |
| QG PN | 0.58 | 0.61 | 0.64 | 0.64 | −1.5 / −1.8 |
| QG LorentzNet | −0.15 | −0.18 [−0.22, −0.15] | −0.22 | −0.25 | 0.13 / 0.21 |
| ref MLP(m/pT) (all four datasets) | 0.99–1.00 | 0.98–1.00 | 0.97–1.00 | 0.97–1.00 | P response exactly 0 |
| ref MLP(m,pT): hls4ml / JetClass / Top / QG | 1.67 / 0.29 / −0.08 / −2.94 | 1.42 / 0.33 / −0.29 / −1.28 | 0.71 / 0.18 / −0.29 / 1.11 | 0.35 / 0.10 / −0.23 / 1.68 | – |
| ref MLP(dR,pT): hls4ml / JetClass / Top / QG | 8.3 / 0.65 / 0.41 / 27.6 | 7.0 / 0.51 / 0.25 / 26.8 | 1.8 / 0.14 / −2.7 / −13.2 | 0.94 / 0.08 / −2.2 / – | – |

Figure: `results/physics/h64_RB.png` (R_B^w at λ 0.9 and 1.1 per model and reference, with 95 % intervals).

### 2. Per-class responses (mean Δ log-odds on true-class jets, 95 % CI; R = rotation control)

| model, class | λ | Δ_B | Δ_P | Δ_A | Δ_R | R_B,k |
|---|---|---|---|---|---|---|
| JEDI-linear, W | 0.9 | −3.62 [−3.70, −3.53] | −3.06 | −5.34 [−5.51, −5.17] | +0.01 | 0.68 [0.66, 0.69] |
| JEDI-linear, W | 1.1 | −0.50 [−0.53, −0.46] | −4.18 | −1.34 | −0.00 | 0.37 [0.35, 0.40] |
| JEDI-linear, Z | 0.9 / 1.1 | −3.32 / −0.73 | −4.87 / −2.15 | −2.53 / −5.68 | 0.00 | 1.31 / 0.13 |
| JEDI-linear, t | 0.9 / 1.1 | +0.42 / −1.35 | +0.08 / −1.40 | −0.29 / −0.90 | 0.00 | −1.46 / 1.49 |
| JEDI-linear, g | 0.9 / 1.1 | −0.71 / +0.76 | −0.62 / +0.71 | −0.10 / +0.02 | 0.00 | 7.0 / 35 (Δ_A ≈ 0) |
| JetClass ParT_kin, Wqq | 0.9 / 1.1 | −0.13 [−0.16, −0.10] / +0.06 | −0.43 / −1.01 | −1.21 / +0.24 | 0.00 | 0.11 / 0.23 |
| JetClass ParT_kin, Tbqq | 0.9 / 1.1 | −0.20 / +0.14 | −0.41 / −0.48 | −0.83 / −0.50 | 0.00 | 0.25 / −0.28 |
| JetClass ParT_full, Wqq | 0.9 / 1.1 | −0.35 / +0.13 | +0.17 / −0.99 | −1.51 / +0.17 | 0.00 | 0.23 / 0.78 |
| Top ParT, top | 0.9 / 1.1 | −0.16 [−0.18, −0.14] / +0.09 | −0.61 / −0.61 | −0.87 / −0.52 | −0.01 | 0.19 / −0.18 |
| Top LorentzNet, top | 0.9 / 1.1 | −0.22 / +0.04 | −0.70 / −0.79 | −1.10 / −0.57 | −0.04 / −0.05 | 0.20 / −0.07 |
| QG ParT, quark | 0.9 / 1.1 | −0.13 [−0.14, −0.12] / +0.13 | +0.06 / −0.07 | −0.21 / +0.18 | 0.00 | 0.61 / 0.71 |
| QG ParT, gluon | 0.9 / 1.1 | +0.13 / −0.16 | −0.01 / +0.00 | +0.14 / −0.16 | 0.00 | 0.94 / 0.97 |
| QG LorentzNet, quark | 0.9 / 1.1 | +0.008 / −0.013 | +0.11 / −0.11 | −0.10 / +0.09 | 0.00 | −0.08 / −0.14 |
| ref MLP(m,pT), hls4ml W | 0.9 | −3.60 | −3.11 | −3.46 | 0 | 1.04 |
| ref MLP(m/pT), hls4ml W | 0.9 | −2.76 | 0.00 | −2.77 | 0 | 1.00 |

**First-order additivity B = P + A**:
- It holds within 0.05 log-odds for the classes without a mass peak (hls4ml g and q, JetClass QCD, QG both classes, Top QCD).
- It fails for the resonance classes. B − (P+A) is +4.1 to +7.1 for JEDI W/Z, +0.6 to +1.5 for JetClass and Top signal classes, and +3.0 for the reference MLP(m,pT) on W.
- The reason: P and A move the mass in opposite directions, and either direction leaves the mass window, so the per-class R_B,k of a peaked class reflects the window shape as well as which quantity is read.

### 3. Calls changed at the 50 % working point (signal class; % of all jets; λ 0.9 / 1.1)

| model, class | B | P | A | R | McNemar B vs A, λ 0.9 |
|---|---|---|---|---|---|
| JEDI-linear, W | 9.7 / 2.9 | 10.9 / 10.2 | 10.3 / 16.6 | 1.4 / 1.2 | p 7e-9 |
| JEDI-linear, t | 6.6 / 4.7 | 4.7 / 6.5 | 5.3 / 4.8 | 0.9 / 0.7 | p 1e-6 |
| Deep Sets / transformer, W | 9.6 / 2.7; 9.8 / 3.9 | 10.2; 10.3 | 10.3; 10.4 | 0.6; 0.2 | p 1e-13; 5e-9 |
| JetClass ParT_full, Wqq / Tbqq | 1.1 / 0.7; 0.7 / 0.3 | 2.4; 0.9 | 3.5; 2.2 | 0.1; 0.1 | p 5e-131; 4e-79 |
| JetClass ParT_kin, Wqq / Tbqq | 1.0 / 0.8; 0.6 / 0.5 | 4.6; 1.5 | 4.8; 2.5 | 0.4; 0.2 | p 1e-176; 2e-87 |
| JetClass PN_kin, Wqq | 0.6 / 0.5 | 5.3 | 4.9 | 0.2 | p 5e-224 |
| Top ParT / PN / LorentzNet, top | 2.7; 3.1; 2.6 (λ 1.1: 2.2; 2.3; 1.6) | 7.0; 5.5; 7.3 | 11.1; 11.5; 12.0 | 1.1; 0.9; 1.3 | p ≤ 3e-74 |
| Top PFN / EFN, top | 7.5; 9.0 | 0; 0 | 7.9; 9.5 | 1.5; 2.7 | p 0.02; 0.005 |
| QG ParT / PN / LorentzNet, quark | 2.1; 1.8; 0.4 | 1.1; 0.8; 2.1 | 3.1; 3.0; 2.1 | 0.5; 0.4; 0.2 | p 8e-9; 1e-12; 5e-13 |

### 4. Nulls

Largest per-class |mean Δ_B| of each null copy at λ 0.9:
- JEDI untr/perm: 0.004 / 0.003.
- Deep Sets untr/perm: 0.47 / 0.61.
- Transformer untr/perm: 0.07 / 0.05.
- JetClass ParT_full_untr / ParT_kin_perm: 0.16 / 0.05. PN_full_untr and PN_kin_perm are unbounded (2e2 / 2e4), so they were left out of H0_null.
- Top ParT perm/untr: 0.07 / 0.05. PN perm: 0.06. LorentzNet perm: 0.001. EFN untr: 0.003.
- QG part perm/untr: 0.04 / 0.01. pnet perm: 0.06.

The nulls' own R_B^w values range from −4.1 to 2.6. R_B therefore says nothing on its own about a network whose response is near the null band. It is read here only for trained models whose |Δ| exceeds the band, which holds for 218 of 284 model × λ × class cells after Holm (next section).

### 5. Global tests (Holm over 284 model × λ × class tests per family, trained models)

| H0 | rejected at Holm p < 0.05 |
|---|---|
| H0_inv: Δ_B = 0 (invariant-mass reader) | 268 / 284 |
| H0_ang: Δ_B = Δ_A (angular-size reader) | 268 / 284 |
| H0_null: Δ_B = bounded-null Δ_B | 218 / 284 |
| McNemar B vs A call changes | 258 / 284 |
| McNemar B vs R call changes | 275 / 284 |

### 6. Training-range check

| dataset | share of B jets outside the training pT range, λ 0.8 / 0.9 / 1.1 / 1.25 | pT-reweighted C2ST AUC of B jets vs real jets (coverage) | same for A at λ 0.9 / unedited |
|---|---|---|---|
| hls4ml | 4.2 / 1.5 / 1.4 / 3.5 % | 0.69 (0.93) / 0.71 / 0.65 / 0.73 | 0.74 / 0.50 |
| JetClass | 63.9 / 35.7 / 2.5 / 8.1 % | 0.56 (0.37) / 0.55 (0.67) / 0.54 / 0.57 (0.93) | 0.65 / 0.50 |
| Top | 100 / 66.9 / 52.4 / 100 % | none (0 coverage) / 0.53 (0.33) / 0.55 (0.50) / none | 0.70 / 0.51 |
| QG | 100 / 100 / 100 / 100 % | none: no real QG jets at the boosted pT | 0.58 / 0.49 |

On JetClass and Top the boosted jets are closer to real jets at the same pT than the angle-only (A) or pT-only (P) edits are: C2ST AUC 0.53–0.57 for B against 0.60–0.77 for A and P. On hls4ml all three edits are separable (0.65–0.87).

## Reading (scoped to what was measured)

1. **Predictions against observations.**
   - The angular-size / m/pT prediction (R_B = 1, Δ_P = 0) is met only by MLP(m/pT) (0.97–1.00) and by Top PFN/EFN (0.95–0.99). For PFN/EFN this follows from their inputs: the pT fraction z_i = pT_i/ΣpT makes the P edit an exact symmetry, so B = A to first order. It is not a measurement of what they learned.
   - The invariant-mass prediction (R_B = 0) is approached by:
     - JetClass ParT/PN at λ > 1 (−0.05 to 0.16) and Top ParT at λ > 1 (−0.05, −0.01);
     - QG LorentzNet, with |Δ_B| ≤ 0.02 log-odds while Δ_A is 0.02–0.11.
   - At λ < 1 the JetClass and Top models sit at R_B^w 0.09–0.45.
2. **Hypothesis (A4/C3 extended): JEDI-linear and ParT_kin respond to B as to A (R_B ≈ 1).**
   - JEDI-linear gives R_B^w 1.05 (λ 0.8), 0.86 (λ 0.9), 0.35 (λ 1.1) and 0.38 (λ 1.25), so it is near 1 only when the boost lowers pT.
   - JetClass ParT_kin gives 0.09–0.25 at λ < 1 and −0.05 / −0.01 at λ > 1. Top ParT gives 0.20 / −0.05.
   - For ParT the alternative (near 0, LorentzNet-like) is closer to the measurement than R_B ≈ 1. For JEDI-linear neither prediction holds across λ.
3. **The m-and-pT reference limits what R_B can separate.**
   - MLP(m, pT) has no angular input, yet it gives R_B^w 1.42 (λ 0.9) and 0.71 (λ 1.1) on hls4ml. That is the same high-at-λ<1 / low-at-λ>1 pattern as JEDI-linear (0.86 / 0.35), and B − (P+A) on W is large in both.
   - So the hls4ml nets' boost response is reproduced by a reader of invariant mass plus jet pT. This is consistent with A14/D1, which say JEDI uses ΣpT as a class feature next to its mass.
   - H64 does not separate "reads angular size" from "reads mass and pT" for the hls4ml nets.
4. **QG ParT/PN.**
   - Their R_B^w of 0.58–0.86 at every λ, with a P response that is small and of opposite sign (R_m^w −1.4 to −4.1), are closer to the angular-size prediction than any other trained ParT/PN.
   - Every boosted QG jet is outside the 500–550 GeV training range, and no real QG jets exist at those pT for a C2ST. These responses are therefore responses to out-of-range inputs.
   - The QG reference MLP(m, pT), which also extrapolates in pT, gives R_B^w −2.9 to +1.7.
5. **LorentzNet is not a boost-invariant control here.**
   - Top LorentzNet changes 2.6 % of top calls under B at λ 0.9. That is about the same as Top ParT (2.7 %) and 2× its own rotation control (1.3 %).
   - Its rotation response (Δ_R −0.04 to −0.10) is the largest of all trained models.
   - The released LorentzNet adds two beam particles along the z axis to every jet. A boost along the jet axis and a rotation about it are therefore not symmetries of its inputs.
   - QG LorentzNet has a near-zero B response.
6. **Relation to H12.** The JetClass and Top call-change rates under B, P and A agree with H12's ordering (B several times below P and A). H64 adds the per-class ratios, the references, the nulls and the pT-reweighted C2ST.

## Issues and caveats

- The R_B ratio is not a clean mass-vs-angle readout when a network also uses the jet pT. The MLP(m, pT) reference shows this on every dataset.
- For peaked classes the first-order decomposition fails (B − (P+A) up to +7 log-odds), so per-class R_B,k for W/Z/top mixes in the window shape. R_B^w weights classes by their A response, which is dominated by those peaked classes.
- Out-of-range inputs: QG (all λ), Top (λ 0.8 and 1.25 entirely; 52–67 % at λ 0.9 and 1.1), JetClass λ 0.8 (64 %).
- JetClass PN nulls are unbounded. H0_null for JetClass uses only the two ParT nulls.
- The MLP references were trained on test-split rows disjoint from the evaluated jets, not on the training split.
- A summary ratio R^w with |denominator| < 0.02 is not reported: the P response of MLP(m/pT), PFN and EFN is exactly 0.
- One sample per dataset. No independent rerun yet (verification pending).

## Explainer entry (for site/explain.py)

- **Question.** When a tagger sees a heavier-looking jet, does it respond to the jet's invariant mass m, or to how wide the jet looks? For a two-prong decay, the opening angle is approximately $\Delta R \approx 2m/p_T$.
- **Physics background.** A Lorentz boost along the jet direction with rapidity $\zeta$ leaves $m^2 = E^2 - |\vec p|^2$ unchanged, multiplies $|\vec p_J|$ by $\lambda = \sinh(\zeta_0 + \zeta)/\sinh\zeta_0$, and shrinks constituent angles by $\approx 1/\lambda$. A network that computes $m$ is unchanged. One that computes an angular size at fixed pT normalisation, or $m/p_T$, changes as if the jet were narrower.
- **Method.** For each jet, apply $B(\lambda)$ (exact boost, $m$ fixed), $P(\lambda)$ ($p^\mu \to \lambda p^\mu$), $A(\lambda)$ ($(\Delta\eta, \Delta\phi) \to (\Delta\eta, \Delta\phi)/\lambda$) and a rotation $R$ of equal displacement. Measure $\Delta_{X,k} = \langle l_k(X) - l_k(0)\rangle_{y=k}$ with $l_k = \log p_k - \log(1-p_k)$, and form $R_B = \Delta_B/\Delta_A$ and $R_m = \Delta_B/\Delta_P$. Predictions: $R_B = 0$ for an $m$ reader, $R_B = 1$ for an angle or $m/p_T$ reader, and $\Delta_B = \Delta_P + \Delta_A$ for an $(m, p_T)$ reader.
- **Result** (figures h64_wjet_boost.png, h64_RB.png):
  - JetClass/Top ParT and ParticleNet: $R_B \approx 0.2$–$0.45$ for $\lambda < 1$ and $\approx 0$ for $\lambda > 1$.
  - hls4ml taggers: 0.6–0.9 and 0.2–0.35.
  - QG ParT/PN: 0.6–0.86.
  - MLP($m/p_T$): 1.00. MLP($m, p_T$): −2.9 to 1.7.
- **Checks.** Exact mass conservation ($|\Delta m|/m \le 4\cdot 10^{-12}$ for $m > 1$ GeV). Rotation control 0.0–0.1 log-odds. Untrained / weight-permuted nulls. pT-reweighted C2ST (B jets look like real jets at their new pT on JetClass/Top: AUC 0.53–0.57). Share of jets outside the training pT range.
- **Physical significance.** Large JetClass/Top taggers respond to a mass-preserving boost much less than to an equal narrowing at fixed pT, so their mass-like feature behaves closer to the invariant mass than to the jet's angular size. A large boost response in the small hls4ml taggers is matched by a classifier on invariant mass plus pT, so it is not evidence of angle reading by itself.

## Glossary additions

- **R_B**: response to the boost / response to the angle-only scaling at the same λ.
- **R_m**: response to the boost / response to the pT-only scaling.
- **R_B^w**: R_B weighted over classes by |Δ_A|.
- **B / P / A / R edits**: as defined above.
- **50 % working point**: the threshold on a class's log-odds that keeps half of that class's unedited jets.
- **pT-reweighted C2ST**: a two-sample classifier test of edited jets against real jets reweighted to the edited jets' pT spectrum.
- **dR_prongs**: the separation of two pT-weighted 2-means axes.

## Files

- Script: `interp/h64_axis_boost_panel.py`.
- Results (`results/physics/`, local copy; cluster `/j-jepa-vol/aaron-interp/results/physics/`): `h64_hls_jedi.json`, `h64_hls_small.json`, `h64_jetclass.json`, `h64_top.json`, `h64_qg.json` (+ `*_perjet.npz`, per-jet own-class Δ log-odds for every model × edit), `h64_summary.json` (R^w, Holm, null bands, unbounded-null list).
- Figures: `h64_wjet_boost.png`, `h64_RB.png`.
- Logs: `/j-jepa-vol/aaron-interp/logs/d-h64-{jedi,small,gpu,sum,sum2}.log`.
