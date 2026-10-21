# H42: jet-charge response nulls for the QG ParT / ParticleNet kinpid taggers (claim D16), part 1

Status: part 1 done (sample asymmetry, C2ST, Herwig arm, seed-0 repeat on new samples). Part 2 is running: the second-seed and shuffled-label kinpid trainings `d-h42-qg-part-kp-s1`, `d-h42-qg-pn-kp-s1`, `d-h42-qg-part-kp-shuf`, `d-h42-qg-pn-kp-shuf` (GPU, 40 GiB). They were first launched at 32 GiB and `pn-kp-shuf` was OOM-killed at about 32.3 GiB while loading data in memory. All four were relaunched at 40 GiB. At report time one was running and three were waiting for a GPU. The E1/E2/E3 rerun on those checkpoints, and the H0/H1 decision, have not been done yet.

## Conclusion

Quark jets in the quark–gluon Z+jet samples have a positive average jet charge (+0.097 e in Pythia, +0.067 e in Herwig). Gluon jets average zero. A network can therefore learn "positive charge means quark" from this sample. The cause is the sample's flavour mix (more u quarks than d quarks), not a difference in how quarks and gluons radiate. Both Pythia-trained particle-ID networks apply the same charge rule to Herwig jets as to Pythia jets. On Herwig that rule is about 20 % stronger than what a charge fit trained on Herwig gives. This run does not yet show whether the rule reproduces with a new training seed or disappears when the labels carry no class information. Those trainings are still queued.

## Procedure

- **Jet charge.** Q_κ = Σ_i q_i (pT_i / pT_J)^κ with κ = 0.3 (also 0.5). q_i is the particle's charge from its PDG id, and pT_J = |vector sum of the particle pT|.
- **Samples.**
  - Pythia: KMT Pythia 8 test split, `data_qg.sample(10000, 4201)`, giving 10,000 quark and 10,000 gluon random rows.
  - Herwig: Herwig 7.1 files 18–19, 10,000 per class, seed 4202.
  - For (a), also every jet: 199,998 Pythia test jets and 200,000 Herwig jets.
- **Models.** `part_kinpid` and `pnet_kinpid`, the H2 weaver trainings on Pythia, best epoch 17 of 20.
  - Score s = log p(quark) − log p(gluon).
- **References.** Logistic regressions on Q_0.3 alone:
  - `LR_Q03_py` was fitted on the 179,998 Pythia test jets outside the sample. Its coefficient is 0.372 log-odds per unit charge.
  - `LR_Q03_hw` was fitted on Herwig file 0 (100,000 jets). Its coefficient is 0.250.
- **Edits.** Only PDG ids change; every (pT, y, φ) and N are fixed and asserted. The code is `h40_qg_pid_charge.make_edits`.
  - **E1:** every charged particle becomes its antiparticle, so Q → −Q exactly (checked: max |Q_E1 + Q| = 0).
  - **E2k1 − C2:** flip the charge of the hardest charged particle, minus the control C2. C2 flips one + and one − particle together with |ΔQ_0.3|, |ΔQ_0.5| < 0.01. The difference is multiplied by the sign of the hardest particle's charge.
  - **E3g_4 / E3n_4 − C3_4:** 2 π+ and 2 π− below the jet's median particle pT are relabelled as photons or as K0L. The control C3_4 swaps the same four pions π+ ↔ π−.
- **(a) Sample asymmetry.** Mean Q per class with a 1,000-sample bootstrap 95 % CI, and the fraction of jets with Q > 0. Quark and gluon means are compared with a Welch t-test.
- **(b) C2ST (classifier two-sample test).**
  - Setup: for each class and generator, the jets where the edit applies are split at random into halves. One half is edited and the other is kept as is.
  - Classifier: a HistGradientBoosting classifier is trained on 50 % of the pooled jets to tell edited from unedited jets. It uses jet-level features: Q_0.3, Q_0.5, n+, n−, n_ch, f_ch, the counts of π, K, p, γ, K0L, n, e and μ, N, girth, C1^(0.5, 1, 2), pT^D and mass.
  - Scoring: AUC on the held-out 50 %, with a bootstrap CI.
  - Null: 50 label permutations, retrained each time. The permutation p has a floor of 1/51. A one-sided normal p from z = (AUC − null mean) / null sd is also given, with Holm correction over the 8 cells.
  - Analytic reference for E1: the AUC obtained by separating Q from −Q using Q alone.
- **(c) Model response.**
  - E1 × sign(Q_0.3): the score change under E1 times the sign of the original jet charge. Negative means positive-charge jets are scored as more quark-like.
  - Also: E1 split by class and sign, E2k1 − C2 signed, E3 − C3, and slopes of Δs per unit ΔQ_0.3.
  - Statistics: 95 % bootstrap (B = 1000); McNemar test at the threshold that gives 50 % quark efficiency on unedited jets; Holm over the 12 model cells (2 models × 2 generators × contrasts).
- **Jobs.** Two CPU jobs (`d-h42-run-py`, 5,712 s; `d-h42-run-hw`, 2,561 s; 6 CPU, 6.6 GiB peak), then `d-h42-merge` for Holm and the figure.

## Numbers

**Unedited models on the samples used.**

| model | Pythia AUC / acc | Herwig AUC / acc | earlier value |
|---|---|---|---|
| ParT kinpid | 0.9084 / 83.4 % | 0.8077 / 71.4 % | H40: 0.9104 Pythia, 0.8029 Herwig (other jets); 0.9123 on the full test set |
| PN kinpid | 0.9038 / 82.9 % | 0.8083 / 71.4 % | H40: 0.9058 Pythia, 0.8047 Herwig |

The AUC differences are within the sample-to-sample variation for 20,000 jets.

**(a) Jet-charge asymmetry, all test jets.**

| generator, class | mean Q_0.3 [95 % CI] | fraction Q > 0 |
|---|---|---|
| Pythia quark | +0.0969 [0.0938, 0.1002] | 57.9 % |
| Pythia gluon | −0.0007 [−0.0039, 0.0023] | 50.0 % |
| Herwig quark | +0.0672 [0.0640, 0.0704] | 55.6 % |
| Herwig gluon | +0.0025 [−0.0006, 0.0054] | 50.2 % |

- The quark-minus-gluon difference is +0.098 (t = 42.6) in Pythia and +0.065 (t = 28.6, p = 4 × 10⁻¹⁷⁹) in Herwig. The standard deviation of Q_0.3 is about 0.51 in every class.
- H40 quoted a gluon mean of +0.017 [0.007, 0.026] on its 10,000-jet sample. The full split gives −0.001, so H40's value was a sampling fluctuation.
- Predictions:
  - H_flavour predicted quark mean > 0 and gluon mean ≈ 0, which is what both generators show.
  - The hypothesis "quark and gluon jets are charge-symmetric" is excluded at t > 28.
- The Herwig-to-Pythia ratio of LR coefficients (0.250 / 0.372 = 0.67) equals the ratio of the quark-jet asymmetries (0.067 / 0.097 = 0.69).

**(b) C2ST, edited versus unedited jets.**

| cell | AUC [95 % CI] | permutation null | Holm p_z | analytic Q vs −Q |
|---|---|---|---|---|
| Pythia E1, quark | 0.556 [0.540, 0.572] | 0.500 ± 0.007 | 4 × 10⁻¹⁴ | 0.603 |
| Pythia E1, gluon | 0.506 [0.492, 0.523] | 0.499 ± 0.008 | 0.39 | 0.504 |
| Herwig E1, quark | 0.541 [0.525, 0.558] | 0.502 ± 0.008 | 1 × 10⁻⁶ | 0.586 |
| Herwig E1, gluon | 0.507 [0.491, 0.522] | 0.501 ± 0.009 | 0.39 | 0.505 |
| Pythia E3 π±→γ, quark / gluon | 0.839 / 0.758 | ≈ 0.50 | < 10⁻¹⁹⁰ | – |
| Pythia E3 π±→K0L, quark / gluon | 0.995 / 0.983 | ≈ 0.50 | < 10⁻¹⁹⁰ | – |

- The permutation p is 0.020 (the floor) for every distinguishable cell.
- E1 gluon jets cannot be told from real gluon jets. E1 quark jets differ from real quark jets only by the sample's charge asymmetry.
- The analytic Q-only AUC is higher than the GBDT's. The GBDT trains on 5,000 jets with 21 features, so it does not reach the one-feature optimum.
- The E3 relabels give jets with species counts unlike real jets: 4 extra K0L gives AUC 0.98–0.99. The H40 neutral-species responses are therefore measured on off-manifold jets, meaning jets unlike any in the training data.

**(c) Model response, Δs in log-odds of quark.**

| contrast | ParT kinpid | PN kinpid | LR_Q03_py | LR_Q03_hw |
|---|---|---|---|---|
| Pythia E1 × sign Q | −0.2515 [−0.2579, −0.2454] | −0.2401 [−0.2447, −0.2354] | −0.310 | −0.208 |
| Herwig E1 × sign Q | −0.2448 [−0.2510, −0.2387] | −0.2393 [−0.2438, −0.2348] | −0.302 | −0.203 |
| Pythia E1 split (q Q>0, q Q<0, g Q>0, g Q<0) | −0.374, +0.261, −0.174, +0.180 | −0.334, +0.254, −0.178, +0.182 | | |
| Pythia E2k1 − C2, signed by leader charge | −0.407 [−0.416, −0.398] | −0.415 [−0.419, −0.410] | −0.429 | −0.288 |
| Herwig E2k1 − C2, signed | −0.434 | −0.443 | −0.432 | −0.290 |
| slope per unit ΔQ_0.3, E1 (Pythia / Herwig) | 0.287 / 0.292 | 0.282 / 0.288 | 0.372 | 0.250 |
| slope per unit ΔQ_0.3, E2 (Pythia / Herwig) | 0.377 / 0.392 | 0.382 / 0.399 | 0.372 | 0.250 |
| Pythia E3 π±→γ − C3 | +0.149 [0.145, 0.153] | +0.148 | 0 | 0 |
| Pythia E3 π±→K0L − C3 | −0.240 [−0.247, −0.234] | −0.260 | 0 | 0 |

- McNemar at 50 % quark efficiency, E1, calls changed:
  - Pythia: ParT 5.19 % (413 to quark / 624 to gluon, p = 6 × 10⁻¹¹), PN 4.46 % (p = 2 × 10⁻¹⁴).
  - Herwig: ParT 6.27 % (p = 1 × 10⁻⁴), PN 5.46 % (p = 3 × 10⁻⁸).
- Holm over the 12 model cells: all 12 have p < 0.05 (normal approximation, n = 14,915–19,999).
- Ratio of the model response to the LR response:
  - Against the Pythia LR: 0.81 (ParT) and 0.78 (PN), on both generators.
  - Against the Herwig LR: 1.21 (ParT) and 1.16–1.18 (PN).
- Null band from H40v (12 weight-permuted networks), max |E1 × sign Q|: 0.019 for ParT and 0.112 for PN. The trained values are 13× and 2.1× those maxima.

## Reading

- **(a)** The quark-jet sample has a positive mean jet charge on both generators, and gluon jets are neutral within ±0.004. This is the pattern expected from qg → Zq with valence u quarks outnumbering d quarks. The Pythia asymmetry is 1.45× the Herwig one. The best charge-only fits scale the same way: the LR coefficient per unit charge is 0.37 on Pythia and 0.25 on Herwig.
- **(b)** E1 jets are indistinguishable from real jets for gluons (AUC 0.506–0.507, null-compatible). For quarks they are distinguishable at the level of the sample's charge asymmetry (0.54–0.56). E1 therefore stays on the data manifold, apart from the very charge asymmetry it tests. The E3 species relabels do not stay on the manifold (AUC 0.76–0.99).
- **(c)** On new samples the seed-0 networks reproduce H40/H40v: E1 × sign Q is −0.25 / −0.24.
  - The networks respond the same way on Herwig jets (−0.245 / −0.239) as on Pythia. Their per-charge slope stays at 0.28–0.29 (E1) and 0.38–0.40 (E2) on both generators. Pythia-trained networks are not expected to adapt to Herwig.
  - Relative to the charge rule that Herwig's own asymmetry supports (LR_hw), the networks over-weight charge by 16–21 % on Herwig jets. They under-weight it by about 20 % relative to the Pythia LR.
  - The H1 prediction "the magnitude tracks LR(Q) on each generator" applies to the sample (the LR ratio follows the asymmetry ratio), not to a fixed network. A network trained on Herwig would be needed to test it on the network side, and none is planned here.
- **Not yet tested.** Whether the rule comes from the labels and reproduces across seeds:
  - H0 (seed artefact): seed-1 E1 × sign Q outside [0.5, 1.5] × seed 0, or of opposite sign.
  - H1: seed 1 inside that range; shuffled-label networks within the permuted band (0.019 ParT, 0.112 PN).
  - The decision code is in `decisions()` of the script. It runs with `--models part_kinpid,pnet_kinpid,part_kinpid_s1,pnet_kinpid_s1,part_kinpid_shuf,pnet_kinpid_shuf --tag _retrained` once the trainings are final. Use the same seeds so the jets are the same; on CPU this takes about 3× the part-1 time, so run it per generator.

## Files

- Script: `interp/h42_qg_charge_null.py`. Training: `interp/h42_kinpid_train.sh`, a copy of `h39_weaver_train.sh` with the kinpid feature set. The shuffled-label check in the training log gives agreement of 0.498 with the true label on test_file_0.
- Results: `results/physics/h42_qg_charge_null.json`, `h42_qg_charge_null_{py,hw}.json`, `h42_perjet_py_py.npz`, `h42_perjet_hw_hw.npz`.
- Figure: `results/physics/h42_charge_asym.png` shows the Q_0.3 distributions by class and generator, with the E1 mirror image of the quark jets.
- Explainer: `site/explain.py` H42_EXPLAINER, plus 4 glossary entries: jet charge Q_κ, charge conjugation edit E1, E1 × sign Q, and LR(Q_0.3).
