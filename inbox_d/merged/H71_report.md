# H71 report: energy-scale panel, dimensionful vs dimensionless reading (loop D, round D15)

Node H71 (exploratory; new D claim). Script `interp/h71_energy_scale_panel.py`.
Results: `results/physics/h71_{hls_jedi,hls_obs,jetclass,top,qg}.json` + `.npz` (per-jet own-class log-odds for every model × edit), `results/physics/h71_summary.json` (all statistics, Holm).
Figure: `results/physics/h71_wjet_scale.png`.
Jobs: `anrunw-ai-d-h71-jedi` (hls4ml JEDI-linear, CPU, 1,173 s), `anrunw-ai-d-h71-hobs` (hls4ml observables and reference classifiers, conda CPU, 103 s), `anrunw-ai-d-h71-gpu` (JetClass, Top, QG; GPU, 28 min), `anrunw-ai-d-h71-sum2` (statistics and figure, 350 s). The first summary job `d-h71-sum` had a p-value floor bug (see Checks) and was replaced by `sum2`.
Logs: `logs/d-h71-{jedi,hobs,gpu,sum,sum2}.log`.
Status: done.

## Conclusion

We multiplied every particle's energy and momentum in a jet by the same factor (±10 %). This changes the jet mass and pT by that factor and leaves every shape variable (m/pT, τ21, τ32, D2) exactly unchanged. Every tagger whose inputs include absolute momenta changed its scores. A shape-only reference classifier did not. On JetClass, ParT and ParticleNet with full inputs responded about 0.6 times as strongly as to the same mass change at fixed pT. JEDI-linear on hls4ml responded about as strongly (1.1 times).

Physically, the taggers read the absolute energy scale of the jet, mostly through its mass, not only dimensionless shapes. A detector energy-scale error therefore shifts their outputs like a mass shift. Scaling only charged or only neutral particles gave small differences, up to 0.16 log-odds, beyond a random-particle control. These appeared mostly in the type-aware JetClass models.

Still uncertain:
- How much of the uniform-scale response comes from pushing jet pT outside the training range: 36 % of JetClass jets, 53–67 % of Top jets and 100 % of QG jets at λ = 0.9 or 1.1.
- Whether the charged/neutral differences come from the particle-type inputs or from the kinematics of charged particles: kinematics-only QG models show differences of similar size.

## Question

Do jet taggers use dimensionful quantities (absolute jet mass m, jet pT, constituent energies) or only dimensionless shapes (m/pT, τ21, τ32, D2, C2, momentum fractions)? A detector-like energy-scale edit answers this, because it changes only the dimensionful quantities.

Hypotheses and their predictions for ρ. ρ is defined under Metrics: the response to a uniform scale divided by the response to the same mass change at fixed pT.
- **H_shape (dimensionless reader):** the uniform-scale response Δs_U = 0 and ρ = 0. References: GBDT(shape), MLP(m/pT), and PFN/EFN, whose inputs are z_i = pT_i/ΣpT and angles.
- **H_mass (absolute-mass reader, pT unused):** Δs_U = Δs_UB in every class, so ρ = 1.
- **H_(m,pT) (reads mass and pT):** ρ equals the value of an MLP trained on (log m, log pT) for the same dataset: 1.51 on hls4ml, 0.77 on JetClass, 1.51 on Top, 3.21 on QG.
- **Type channel (links to H68):** a charged-only vs neutral-only scale difference beyond the random-subset control appears in ParT_full/PN_full and not in the kinematics-only (kin) models.

## Procedure

**Samples.** Rows are drawn at random, never first-k.
- hls4ml: `jedi_core.sample(per_class=1000, seed=7101)`, 5 classes, N = 5,000. Reference-classifier training jets: 3,000 other rows per class (rng 7111).
- JetClass: `h1_irc_panel.jc_sample(3000, 7102)`, random rows of each class's first test file. 1,000 per class are evaluated (N = 10,000) and 2,000 per class train the references (rng 7112).
- Top Landscape test: 2,000 top + 2,000 QCD (rng 7103). References: 5,000 + 5,000 disjoint rows (rng 7113).
- Quark–gluon (QG) test: 2,000 quark + 2,000 gluon (rng 7104). References: 5,000 + 5,000 disjoint rows (rng 7114).

**Models.**
- hls4ml: JEDI-linear (official, 3-feature, n64).
- JetClass: ParT_full, ParT_kin, PN_full and PN_kin (PN = ParticleNet). "full" models take the particle type, charge and impact parameters; "kin" models take kinematics only.
- Top: ParT_kin, PN_kin, LorentzNet, PFN, EFN.
- QG: ParT_kin, PN_kin, ParT_kinpid and PN_kinpid (with particle-ID inputs), LorentzNet, and torch PFN / EFN from H5.
- Nulls: weight-permuted copies (every weight tensor's entries permuted, seed 7) of JEDI-linear, all four JetClass nets, Top ParT / PN / PFN and QG ParT_kin / ParT_kinpid.

**Edits.** Each edit acts on the constituent four-momenta p_i = (E, px, py, pz) of the stored jet. λ ∈ {0.9, 0.95, 0.98, 1.02, 1.05, 1.1}.

| arm | what changes | what is held fixed | jet m | jet pT |
|---|---|---|---|---|
| U(λ), uniform energy scale | every p_i → λ p_i | all angles, z_i, m/pT, τ21, τ32, D2, C2, N | ×λ | ×λ |
| UB(λ), mass at fixed pT | U(λ), then an exact boost along the jet momentum that returns jet pT to its original value | jet pT; the invariant mass stays λ m | ×λ | ×1 |
| Ch(λ), tracker scale | p_i → λ p_i for charged particles only | neutral particles | ~×(1 + f_ch(λ−1)) | same |
| Nh(λ), calorimeter scale | p_i → λ p_i for neutral particles only (photons, neutral hadrons) | charged particles | | |
| Rh / Rc, random-subset control | λ applied to a random subset whose pT share matches the jet's charged share (Rh), or to its complement (Rc) | | | |
| S(a), resolution smearing | p_i → g_i p_i at fixed direction; g_i log-normal with mean 1 and relative spread a/√(E_i/GeV) + 0.03; a ∈ {0.5, 1.0} | particle directions, N | | |

**Charged definition.**
- JetClass: part_charge ≠ 0.
- QG: |PDG id| ∈ {211, 321, 2212, 11, 13}.
- hls4ml and Top have no particle IDs. There, Ch / Nh act on a random half of the constituents and its complement, so on these two datasets the Ch − Nh difference is itself a null measurement.

**Measured edit sizes.**
- Mean charged pT share: JetClass 0.596, QG 0.616.
- The Rh subset matches it to a median |Δshare| of 0.014–0.017 (95th percentile 0.11–0.12).
- U at λ = 1.1 gives Δlog m = Δlog pT = +0.0953 exactly.
- UB at λ = 1.1 gives Δlog m = +0.0953 and Δlog pT = 0.0000.
- τ21, τ32, D2 and C2 are unchanged under U to 1e-14 on JetClass, Top and QG. On hls4ml, D2 and C2 are unchanged to 1e-14, while τ21 changes by up to 2.9 % on 300 jets through ties in the clustering order. The references are therefore scored on the unedited shapes for U.

**Why these controls.**
- UB changes the jet mass by the same factor as U but keeps pT. Comparing U with UB separates "reads m" from "reads m together with pT, or m/pT".
- Rh / Rc move the same share of jet energy as Ch / Nh, but on randomly chosen particles. A difference between Ch and Rh must therefore come from which particles are charged, either their type or their kinematics.
- The kin models cannot see particle type. Any Ch/Rh difference in them is kinematic.

**References.** sklearn models trained on the disjoint real jets:
- MLP(m,pT) on [log m, log pT];
- MLP(m/pT) on [log(m/pT)];
- GBDT(shape): HistGradientBoosting on [τ21, τ32, log D2, log C2]. τ_N uses WTA kT axes with β = 1 and R0 = 0.8. D2 = e3/e2³ and C2 = e3/e2², with β = 1;
- LR(N): logistic regression on the constituent count N.

**Mass-window reference.** For each class, the window is the interquartile range of that class's unedited jet mass, which gives 50 % efficiency on unedited jets. The efficiency after each edit is then measured.

**Metrics.**
- Own-class log-odds l_k = log p_k − log(1 − p_k), from the softmax, on jets of true class k.
- **Δs** = l_k(edited) − l_k(unedited), averaged over the class's jets.
- **Odd part g** = (Δs(1.1) − Δs(0.9)) / log(1.1/0.9): the slope dΔs/dlog λ.
- **Even part h** = (Δs(1.1) + Δs(0.9)) / 2. A negative h means the score drops in both directions, which is the signature of a window around the nominal scale.
- **ρ** = Σ_k size_U,k / Σ_k size_UB,k, where size = (|Δs(0.9)| + |Δs(1.1)|)/2. It is the network's response to a uniform energy scale divided by its response to the same mass change at fixed pT.
- **Frozen 50 % working point.** For each class, the threshold on l_k gives 50 % efficiency on unedited jets of that class. We report the efficiency after the edit and the share of calls changed.

**Statistics.**
- 95 % bootstrap intervals over jets: 1,000 resamples, class-stratified for ρ.
- Paired sign-flip permutation tests with 2,000 flips (H0: mean per-jet difference = 0). When no flip reached the observed value, the normal tail of the sign-flip null (variance Σd²/n²) is used. The permutation floor of 1/2001 would otherwise block Holm over 138 tests.
- Exact McNemar tests on the call changes at the working point.
- Holm correction within each family, across all datasets × trained models × classes × λ ∈ {0.9, 1.1}. Families: Δs_U = 0; Δs_U = Δs_UB; asymmetry excess; Ch = Rh; Nh = Rc; S1.0 = 0; and three McNemar families.

**Asymmetry excess** = (Δs_Ch − Δs_Nh) − (Δs_Rh − Δs_Rc), per jet. It measures how much more the network separates charged-only from neutral-only scaling than it separates a random subset from its complement of the same energy shares.

## Numbers

**Unedited accuracy on these samples** (known value from earlier nodes in parentheses).

| dataset | model | accuracy (known) | AUC |
|---|---|---|---|
| hls4ml | JEDI-linear | 0.8082 (0.8066, H64) | – |
| JetClass | ParT_full | 0.8649 (0.858, H64 / 0.859, D4) | – |
| | ParT_kin | 0.7497 (0.742 / 0.745) | – |
| | PN_full | 0.8499 (0.838) | – |
| | PN_kin | 0.7086 (0.704) | – |
| Top | ParT_kin | 0.9390 (0.937 / 0.940) | 0.987 |
| | PN_kin | 0.9367 (0.940) | 0.985 |
| | LorentzNet | 0.9390 (0.943) | 0.986 |
| | PFN | 0.9335 (0.929) | 0.981 |
| | EFN | 0.9225 (0.923) | 0.977 |
| QG | ParT_kin | 0.8257 (0.841, H64 / 0.827, H2) | 0.898 |
| | PN_kin | 0.8237 (0.836) | 0.898 |
| | ParT_kinpid | 0.8325 | 0.905 |
| | PN_kinpid | 0.8233 | 0.901 |
| | LorentzNet | 0.8430 (0.854) | 0.914 |
| | PFN / EFN (torch) | 0.798 / 0.795 | 0.876 / 0.869 |

- Null accuracies: 0.09–0.11 (JetClass), 0.49–0.50 (Top, QG) and 0.200 (hls4ml; the permuted JEDI-linear gives a constant output).
- Reference accuracies:

| dataset | MLP(m,pT) | MLP(m/pT) | GBDT(shape) | LR(N) |
|---|---|---|---|---|
| hls4ml | 0.744 | 0.672 | 0.657 | 0.341 |
| JetClass | 0.331 | 0.265 | 0.411 | 0.214 |
| Top | 0.890 | 0.888 | 0.904 | 0.646 |
| QG | 0.669 | 0.669 | 0.761 | 0.755 |

**Uniform scale vs mass at fixed pT: ρ with 95 % CI.** size_U and size_UB are in log-odds, summed over classes.

| dataset | model | ρ | size_U | size_UB |
|---|---|---|---|---|
| hls4ml | JEDI-linear | 1.116 [1.097, 1.137] | 9.00 | 8.06 |
| | ref MLP(m,pT) | 1.506 [1.483, 1.529] | 7.82 | 5.19 |
| | ref MLP(m/pT), GBDT(shape) | 0 (exact) | 0 | 4.45 / 1.73 |
| JetClass | ParT_full | 0.621 [0.613, 0.633] | 3.59 | 5.78 |
| | PN_full | 0.669 [0.653, 0.682] | 3.80 | 5.67 |
| | ParT_kin | 0.921 [0.900, 0.951] | 4.41 | 4.79 |
| | PN_kin | 0.883 [0.867, 0.899] | 3.09 | 3.50 |
| | ref MLP(m,pT) | 0.772 [0.762, 0.783] | 1.98 | 2.57 |
| | ref MLP(m/pT), GBDT(shape) | 0 | 0 | 1.85 / 1.59 |
| | null ParT_full_perm | 0.22 [0.21, 0.37] | 0.11 | 0.49 |
| Top | ParT_kin | 0.863 [0.848, 0.878] | 0.86 | 1.00 |
| | PN_kin | 0.672 [0.651, 0.692] | 0.75 | 1.12 |
| | LorentzNet | 0.807 [0.793, 0.822] | 1.05 | 1.30 |
| | PFN, EFN | 0 (inputs are z_i and angles) | 0 | 0.91 / 0.80 |
| | ref MLP(m,pT) | 1.509 [1.494, 1.527] | 1.68 | 1.11 |
| | null ParT_kin_perm | 0.34 | 0.03 | 0.09 |
| QG | ParT_kin | 0.211 [0.197, 0.226] | 0.074 | 0.352 |
| | PN_kin | 0.370 [0.362, 0.379] | 0.132 | 0.356 |
| | ParT_kinpid | 0.394 [0.355, 0.433] | 0.074 | 0.188 |
| | PN_kinpid | 0.395 [0.359, 0.434] | 0.058 | 0.148 |
| | LorentzNet | 1.174 [1.150, 1.202] | 0.166 | 0.142 |
| | PFN, EFN | 0 | 0 | 0.110 / 0.495 |
| | ref MLP(m,pT) | 3.21 [3.17, 3.26] | 0.472 | 0.147 |

**Per-class curves, examples.** Mean Δs at λ = 0.9 / 1.1.
- JetClass Wqq:
  - ParT_full U +0.19 / −1.09, UB +0.15 / −1.44;
  - ParT_kin U −0.48 / −1.07;
  - PN_full U +0.25 / −1.03;
  - MLP(m,pT) U +0.05 / −0.36;
  - GBDT(shape) U 0 / 0.
- JetClass Hbb, ParT_full: U −0.83 / +0.24, UB −1.30 / +0.11.
- Top, top jets:
  - ParT_kin U −0.67 / −0.50, UB −0.70 / −0.61;
  - LorentzNet U −0.74 / −0.71.
- hls4ml W, JEDI-linear: U −3.06 / −4.22, UB −1.55 / −4.58 (MLP(m,pT): U −3.47 / −2.88).
- hls4ml gluon, JEDI-linear: U −0.61 / +0.71, UB +0.03 / −0.08. This is a pT response at fixed m/pT; MLP(m,pT) gives U −0.44 / +0.74 and UB +0.05 / −0.07.

**Holm, all datasets.**

| family | rejected / tests | breakdown |
|---|---|---|
| H0: Δs_U = 0 | 111 / 138 | PFN / EFN 0 / 8 (exact 0) |
| H0: Δs_U = Δs_UB | 120 / 138 | JetClass ParT_full 19 / 20, PN_full 18 / 20; JEDI 8 / 10 |
| McNemar, U vs UB call changes at λ = 1.1 | 39 / 69 | ParT_full 10 / 10, PN_full 8 / 10, ParT_kin 0 / 10, PN_kin 2 / 10 |

**Frozen 50 % working point.** Efficiency after U(0.9) / U(1.1) / UB(1.1), with the interquartile mass window at U(0.9) / U(1.1) in parentheses.

| dataset, class | model | U(0.9) | U(1.1) | UB(1.1) | mass window U(0.9) / U(1.1) |
|---|---|---|---|---|---|
| JetClass Wqq | ParT_full | 0.55 | 0.30 | 0.26 | (0.31 / 0.40) |
| | PN_full | 0.57 | 0.28 | 0.23 | |
| Top, top jets | ParT_kin | 0.40 | 0.41 | 0.39 | (0.23 / 0.28) |
| | LorentzNet | 0.39 | 0.38 | 0.35 | |
| hls4ml W | JEDI-linear | 0.02 | 0.04 | 0.03 | (0.01 / 0.03) |
| hls4ml Z | JEDI-linear | 0.01 | 0.14 | 0.13 | (0.01 / 0.02) |

**Jets pushed outside the dataset's pT range** (shares at U(0.9) / U(0.95) / U(1.05) / U(1.1); UB keeps pT fixed).

| dataset | pT range | U(0.9) | U(0.95) | U(1.05) | U(1.1) |
|---|---|---|---|---|---|
| JetClass | 500–1000 GeV | 0.36 | 0.19 | 0.01 | 0.03 |
| Top | 550–650 GeV | 0.67 | 0.34 | 0.26 | 0.53 |
| QG | 500–550 GeV | 1.00 | 0.59 | 0.46 | 1.00 |

**Charged-only vs neutral-only scale, asymmetry excess at λ = 1.1** (log-odds, 95 % CI, Holm p).
- JetClass ParT_full, Holm < 0.05 in 7 / 10 classes:

| class | excess | Holm p |
|---|---|---|
| Wqq | +0.155 [+0.129, +0.184] | 5e-23 |
| Tbl | +0.132 [+0.105, +0.158] | 2e-21 |
| Hgg | +0.080 | 2e-15 |
| QCD | +0.074 [+0.061, +0.088] | 1e-19 |
| Zqq | +0.071 | 6e-5 |
| Tbqq | +0.065 | 0.002 |
| Hcc | −0.053 | 0.005 |

  Components for Wqq: Ch − Rh = +0.077 [+0.062, +0.091] and Nh − Rc = −0.078 [−0.091, −0.064].
- JetClass PN_full, Holm < 0.05 in 7 / 10 classes: Wqq +0.125 [+0.101, +0.150], Tbl +0.126, QCD +0.063.
- JetClass kin models: ParT_kin 1 / 10 (Wqq −0.022 [−0.063, +0.015], QCD +0.032 [+0.002, +0.060]); PN_kin 3 / 10 (Hbb +0.077, Hqql +0.076, Tbl +0.095).
- hls4ml JEDI-linear (random halves): 0 / 5, all |excess| ≤ 0.055. Top, all five models (random halves): 0 / 10, all |excess| ≤ 0.018.
- QG:
  - ParT_kin: gluon −0.025 [−0.033, −0.018], quark +0.027; 2 / 2.
  - PN_kin: 2 / 2, similar sizes.
  - ParT_kinpid: gluon +0.024 (opposite sign to the kin model), quark −0.012; 1 / 2.
  - PN_kinpid: 1 / 2.
  - PFN (no ID input): 2 / 2, ±0.015.
  - EFN, LorentzNet: 0 / 2.
- At λ = 0.9 the excess is also large for JetClass ParT_kin (8 / 10 classes, up to −0.69). ParT_kin has a step in its U response between λ = 0.95 and 0.9 (Wqq +0.20 → −0.48; QCD +0.04 → −0.40) that no other model shows. 36 % of JetClass jets fall below 500 GeV at U(0.9). We therefore read the charged/neutral comparison at λ = 1.1.
- McNemar call-change tests, Ch vs Rh and Nh vs Rc at λ = 1.1: 3 / 69 and 2 / 69 after Holm. The shifts are 0.05–0.16 log-odds and move few jets across the 50 % threshold.

**Resolution smearing S(1.0)**, mean Δs (95 % CI).
- JetClass:

| model | QCD | Wqq | Tbqq | other classes |
|---|---|---|---|---|
| ParT_full | −0.118 | −0.330 | −0.365 | Hqql −0.469 |
| PN_full | −0.065 | −0.203 | −0.140 | |
| ParT_kin | −1.17 | −2.08 | | H4q −2.30, Hbb +0.96 |
| MLP(m,pT) | | | | \|Δs\| ≤ 0.075 |
| GBDT(shape) | | | | \|Δs\| ≤ 0.043 |

- QG:

| model | gluon | quark |
|---|---|---|
| ParT_kin | −0.316 [−0.335, −0.298] | +0.176 |
| LorentzNet | −0.298 | +0.107 |
| PFN | −0.060 | +0.026 |
| GBDT(shape) | −0.091 | +0.104 |

- Top: ParT_kin QCD −0.128, top −0.076; LorentzNet top −0.337; EFN −0.03.
- hls4ml JEDI-linear: W −0.82, Z −0.81. MLP(m,pT): W −0.83, Z −0.82.
- H0: S1.0 = 0 is rejected for 64 / 69 tests after Holm.

## Reading

- **Uniform scale.**
  - Every network with absolute-momentum inputs changes its scores under U, which leaves all dimensionless shapes unchanged; the shape-only references give exactly 0. H_shape is excluded for JEDI-linear and for the JetClass, Top and QG ParT / PN / LorentzNet models (Holm, Δs_U = 0 rejected in 111 / 138 tests). PFN and EFN (inputs z_i, Δη, Δφ) respond exactly 0 by construction.
  - On JetClass, the full-input models respond to U with 0.62–0.67 of their response to the same mass change at fixed pT. H_mass predicts 1 and is excluded: U ≠ UB in 10 / 10 classes for ParT_full and PN_full.
  - The kin models sit at 0.88–0.92, closer to H_mass. The MLP(m,pT) reference sits at 0.77.
  - JEDI-linear (1.12) lies between H_mass (1) and MLP(m,pT) (1.51). Class by class it follows the MLP(m,pT) pattern. W and Z respond about equally to U and UB, which is a mass window. Gluon and top respond to U but not to UB, which is a pT response at fixed m/pT, in line with the ΣpT shortcut of D1.
  - Top ParT / PN / LorentzNet: ρ = 0.67–0.86. The top-jet score falls on both sides of λ = 1 (even part h < 0), which is the window signature.
  - QG kin and kinpid models: ρ = 0.21–0.40, so they are closer to dimensionless. QG LorentzNet: 1.17.
- **Size relative to the mass-window reference.** At the frozen 50 % working point, JetClass ParT_full's Wqq efficiency moves 0.50 → 0.30 under U(1.1); the interquartile W mass window moves 0.50 → 0.40. Under U(0.9), ParT_full rises to 0.55 while the window falls to 0.31. ParT_full's W response is therefore asymmetric in λ, not a symmetric window about the nominal mass.
- **Training-range confound.** U moves jet pT by ±10 %. This puts 36 % (JetClass), 67 % (Top) and 100 % (QG) of jets outside the training pT range at λ = 0.9. UB does not. For Top and QG, part of the U response may be a response to out-of-range pT rather than to mass. The QG ρ values are the most exposed.
- **Charged vs neutral.**
  - The random-half datasets (hls4ml, Top) give no excess (0 / 15), so the control is calibrated.
  - On JetClass at λ = 1.1, the full models show an excess in 7 / 10 classes each. The kin models show it in 1 / 10 and 3 / 10.
  - Wqq gives the clearest contrast: ParT_full +0.155 [+0.129, +0.184] against ParT_kin −0.022 [−0.063, +0.015]; PN_full +0.125 against PN_kin +0.008. Scaling the charged particles up moves the W score less negative than scaling a random subset of the same energy share.
  - On QG, the kin-only models and PFN show excesses of similar size (±0.025) to the ID-input models, with the opposite sign in gluon jets. Charged and neutral particles differ kinematically by themselves, so on QG the asymmetry cannot be assigned to the type channel.
  - The effects are small in calls: 2–3 of 69 McNemar tests survive Holm.
- **Smearing.**
  - S(1.0) lowers the own-class scores of most JetClass classes for all four models. ParT_kin is 5–10× more sensitive than ParT_full.
  - On QG, all nets move gluon jets toward quark, which is the same sign as GBDT(shape), with 2–4× larger magnitude.
  - hls4ml JEDI-linear W / Z drops (−0.82 / −0.81) equal those of MLP(m,pT) (−0.83 / −0.82). They are consistent with the mass resolution widening out of the W / Z windows.

## Checks

- **Unedited accuracies** reproduce earlier nodes within sampling error on 4,000–10,000 jets: ParT_full 0.865 vs 0.858–0.859, JEDI-linear 0.808 vs 0.807, Top ParT 0.939 vs 0.937–0.940. QG ParT_kin 0.826 matches H2's full-test 0.827; H64's 4,000-jet sample gave 0.841.
- **Invariance of shapes under U:** relative changes ≤ 3e-14 on JetClass, Top and QG. On hls4ml, τ21 changes by up to 2.9 % (ties); D2 and C2 are unchanged to ≤ 1e-14.
- **Nulls.** Permuted models respond 30–50× less than trained models: JetClass ParT_full_perm size_U 0.11 vs 3.59. Permuted JEDI-linear gives a constant output, so its response is exactly 0 and it is a degenerate null. PN_full_perm and PN_kin_perm (JetClass) have unbounded outputs (|Δs| up to 10⁴); they are reported in the JSON but carry no information.
- **Additivity.** Δs_Ch + Δs_Nh − Δs_U is stored per class in `h71_summary.json` (`asym.<λ>.additivity_ch_plus_nh_minus_u`).
- **Code fix.** The first summary job (`d-h71-sum`) used a pure 2,000-flip permutation p. Its floor of 5e-4, times 138 tests, made every Holm-adjusted p ≥ 0.07. `sum2` uses the normal tail of the sign-flip null when no flip reaches the observed value. The effects and CIs are unchanged.

## Explainer entry (for site/explain.py, Paper D tab)

**Question.** Does a jet tagger read the absolute energy scale of a jet (its mass in GeV, its pT, the energies of its particles) or only dimensionless shapes (m/pT, τ21, τ32, D2)?

**Physics background.**
- A detector measures particle energies with a scale uncertainty of a few per cent. The tracker (charged particles) and the calorimeter (neutral particles) have separate scales.
- Multiplying every particle four-momentum by λ multiplies the jet mass and pT by λ. It leaves every angle and every ratio of energies unchanged, and so leaves every dimensionless substructure observable unchanged:
$$p_i^\mu \to \lambda p_i^\mu:\quad m\to\lambda m,\; p_T\to\lambda p_T,\; \frac{m}{p_T},\,\tau_{21},\,\tau_{32},\,D_2,\,C_2,\, z_i=\frac{p_{T,i}}{\sum_j p_{T,j}}\ \text{unchanged}.$$
- A tagger that uses only shapes does not respond. A mass-window tagger responds as if the jet mass had changed.

**Method.**
- Four edits:
  - U: uniform scale of all particles.
  - UB: U followed by a boost along the jet axis that restores the jet pT. The mass changes by λ at fixed pT.
  - Ch / Nh: scale charged-only or neutral-only.
  - Rh / Rc: scale a random subset with the same pT share, or its complement.
- Plus a per-particle energy smearing $\sigma_E/E = a/\sqrt{E} + 0.03$.
- Response: the change of the own-class log-odds $\Delta s = \ell_k^{\rm edit}-\ell_k$. Summary:
$$\rho = \frac{\sum_k \tfrac12\left(|\Delta s_{U,k}(0.9)|+|\Delta s_{U,k}(1.1)|\right)}{\sum_k \tfrac12\left(|\Delta s_{UB,k}(0.9)|+|\Delta s_{UB,k}(1.1)|\right)}$$
- Predictions: ρ = 0 for a shape reader and ρ = 1 for a pure mass reader.
- Charged/neutral excess: $(\Delta s_{Ch}-\Delta s_{Nh})-(\Delta s_{Rh}-\Delta s_{Rc})$.

**Result (figure `h71_wjet_scale.png`).**
- Left panel: one JetClass W jet (m = 79.9 GeV) as λ goes from 0.9 to 1.1. The W score falls steeply once the jet mass exceeds about 82 GeV, with or without the pT change.
- Right panel: mean over 1,000 W jets. All four JetClass networks respond (Δs ≈ −1.0 at λ = 1.1). The shape-only reference stays at 0. MLP(m,pT) responds −0.36.
- ρ by dataset:

| dataset | models | ρ |
|---|---|---|
| JetClass | ParT_full, PN_full | 0.62–0.67 |
| JetClass | kin models | 0.88–0.92 |
| hls4ml | JEDI-linear | 1.12 |
| Top | ParT, PN, LorentzNet | 0.67–0.86 |
| QG | ParT, PN | 0.21–0.40 |

**Checks.**
- Shapes are invariant under U to 1e-14.
- PFN / EFN, whose inputs are momentum fractions, respond exactly 0.
- Permuted networks respond 30–50× less.
- The random-half control on datasets without particle IDs gives no charged/neutral excess.

**Physical significance.** The taggers read the absolute energy scale, mostly through the jet mass, not only scale-free shapes. A jet-energy-scale error of a few per cent therefore shifts their outputs as much as a mass shift of the same size, two-thirds as much on JetClass with full inputs. Type-aware JetClass taggers respond somewhat differently to tracker-only and calorimeter-only scale shifts, up to 0.16 log-odds in W jets, and kinematics-only models mostly do not. On quark–gluon jets, kinematics-only models show a difference of similar size.

**New glossary terms (explain.GLOSSARY):**
- **Uniform energy scale U(λ)**: every constituent four-momentum multiplied by λ. Jet mass and pT scale by λ; all angles and dimensionless shapes stay fixed.
- **UB(λ)**: U(λ) followed by a boost along the jet axis that restores the jet pT. It is the same mass change at fixed pT.
- **Tracker / calorimeter scale (Ch / Nh)**: the energy scale applied only to charged or only to neutral constituents.
- **Rh / Rc**: a random subset of constituents with the same pT share as the charged particles, and its complement. It is the control for Ch / Nh.
- **ρ (scale ratio)**: the network's response to a uniform energy scale divided by its response to the same mass change at fixed pT. 0 means a reader of dimensionless shapes; 1 means a reader of absolute mass only.
- **Asymmetry excess**: (Δs_Ch − Δs_Nh) − (Δs_Rh − Δs_Rc). It is how much more the network separates charged-only from neutral-only scaling than a random split of the same energy shares.
- **Dimensionful / dimensionless observable**: an observable with units of energy (m, pT, E_i), or one unchanged by a common rescaling of all momenta (m/pT, τ_N ratios, D2, C2, z_i).
- **C2**: the energy-correlation ratio e3/e2² (β = 1). It is a two-prong discriminant like D2 = e3/e2³.
