# H48 — Leptonic decay endpoints in JetClass taggers: the m_bl endpoint in t→bℓν and the H→ℓνqq′ visible-mass edge

Node H48 (loop D, round D10; new claim D20, exploratory). Script `interp/h48_lepton_endpoint.py`. Jobs: `d-h48-smoke` (CPU 4 / 8 GiB, 289 s, 300 jets per class), `d-h48-run` (CPU 4 / 8 GiB, 943 s, peak memory 5.9 GiB, 3.4 cores on average). Results: `results/physics/h48_lepton_endpoint.json`, `h48_perjet.npz`; figures `results/physics/h48_jet_edit.png` (one edited jet) and `h48_response.png` (responses and data curve); logs `logs/d-h48-run.log`, `logs/d-h48-smoke.log`. Explainer entry `H48_EXPLAINER` and five glossary terms added to `site/explain.py`.

## Conclusion

When the lepton in an H→ℓνqq′ jet is turned away from the other particles so that the visible jet mass rises by 20 GeV across the Higgs mass (125 GeV), ParT and ParticleNet lower their Higgs score about 4 log-odds more than for the same 20 GeV rise below 125 GeV. That step is about twice what reference classifiers given only the jet mass and lepton isolation predict, and it holds with or without particle-ID inputs. For leptonic top jets (t→bℓν) ParT with full inputs shows no extra drop at the lepton–b mass endpoint (152.6 GeV, the largest lepton + b mass a top decay allows): its response follows a smooth mass-only reference. The four-momentum-only models do drop at that endpoint, but they do not rise when a jet is moved back below it, so this is not a two-way use of the endpoint. The H→ℓνqq′ edit also isolates the lepton from the quark pair, so it does not fully separate the mass edge from lepton–quark geometry.

## Question and hypotheses

Does a tagger use the kinematic endpoint of the visible decay products in leptonic jets? A **kinematic endpoint** is the largest value a visible-mass combination can take, set by the parent masses. For t → b W → b ℓ ν with massless b, ℓ, ν the lepton + b mass is m_bl ≤ sqrt(m_t² − m_W²) = 152.6 GeV (m_t 172.5, m_W 80.4 GeV). For H → W W* → ℓ ν q q′ the visible mass is ≤ m_H = 125 GeV. D2 tested only lepton isolation and the lepton–b charge link.

- **H0** (smooth mass dependence): the response to a +20 GeV mass shift is about the same whether or not the shift crosses the edge, as the reference MLP(m) trained on data predicts. The network's score-vs-mass curve does not change slope at the edge any more than MLP(m)'s does.
- **H1** (endpoint used): the "across" arm drop is larger than the "below" arm change and larger than the MLP(m) prediction. The plan expected E − C below −1 log-odds for the full models, smaller drops for the kin models, about 0 for permuted networks, and the below arm near MLP(m).

## Definitions

- **Lepton ℓ**: the hardest real particle with isElectron or isMuon set. **b system B**: every other real particle in the jet. The neutrino is invisible, so B holds the b quark's hadrons plus radiation, and for H jets also the q q′ pair.
- **m_vis**: the invariant mass of B + ℓ. Because B is all the other particles, this is exactly the jet's invariant mass. In t→bℓν without radiation, m_vis = m_bl.
- **θ**: the 3-D opening angle between the lepton momentum and the momentum of B.
- **I_ℓ** (relative isolation): the scalar pT sum of the other particles within ΔR < 0.2 of the lepton, divided by the lepton pT. Small means isolated. **f_ℓ** = lepton pT / jet pT.
- **Scores**: from the 10-class logits L, s_Tbl = L_Tbl − log(e^{L_QCD} + e^{L_Tbqq}) = log p(t→bℓν) / (p(QCD) + p(t→bqq′)). s_Hqql is defined the same way. t→bℓν jets are scored with s_Tbl and H→ℓνqq′ jets with s_Hqql. A secondary score, signal vs all nine other classes, is stored as `E-C_vs_rest`.
- **E − C**: for each jet, the change in score under edit E minus the change under control C (both measured against the unedited, rebuilt jet), averaged over jets.
- **Holm**: a step-down correction of p-values over the tested family that controls the family-wise error rate. **McNemar**: an exact binomial test on the numbers of jets whose call flips in each direction between C and E. The call is s above the threshold that passes 50 % of unedited signal jets.

## Sample

JetClass test_20M files TTBarLep_100, HToWW2Q1L_100, ZJetsToNuNu_100 (QCD) and TTBar_100 (hadronic top). From each, 3,000 rows were drawn at random (`np.random.default_rng(4801 + k)`, k = 0..3; never the first k rows), N = 12,000 in total. Every jet, the unedited one included, is rebuilt from its particle four-vectors about its own E-scheme axis, using the `part_core.build` convention η = jet_eta + sign(jet_eta)·Δη.

Data check, unedited jets:

| class | with a lepton | m_vis > 152.6 GeV (of jets with a lepton) | m_vis > 125 GeV | median f_ℓ | median I_ℓ |
|---|---|---|---|---|---|
| t→bℓν | 99.0 % | 16.1 % | 42.5 % | 0.354 | 0.008 |
| H→ℓνqq′ | 98.5 % | 3.2 % | 9.1 % | 0.319 | 0.576 |
| QCD | 11.3 % | – | – | 0.010 | 25.9 |
| t→bqq′ | 45.2 % | – | – | 0.029 | 7.0 |

On the 15,000 training jets per class the fractions above the edge are 16.3 % (t→bℓν, 152.6 GeV) and 8.9 % (H→ℓνqq′, 125 GeV). The t→bℓν m_vis quantiles (5, 25, 50, 75, 95 %) are 61, 96, 119, 141 and 202 GeV. Radiation and extra particles therefore smear the endpoint in these jets.

## Edits (what moved, what was held)

- **E(Δm)**: only the lepton moves. Its momentum is rotated in the plane spanned by p_B and p_ℓ to the opening angle θ′ that gives m_vis → m_vis + Δm exactly: cos θ′ = (m_B² + m_ℓ² + 2E_B E_ℓ − (m_vis + Δm)²) / (2|p_B||p_ℓ|). The lepton's energy, |p|, mass, charge, PID flags and impact parameters are kept, and B is untouched. With Δm = +20 GeV, θ goes from 0.306 to 0.378 rad on average in the top "below" arm and from 0.353 to 0.428 rad in the top "across" arm. For H jets, θ goes from 0.162 to 0.273 rad below and from 0.189 to 0.300 rad across. The lepton's lab pT changes by less than 2 GeV on average. The resulting mass change is +20.000 GeV, with every jet within 10⁻⁴ of the target (asserted).
- **C** (control): the lepton is rotated about the p_B direction by the azimuth φ with cos δ = cos²θ + sin²θ cos φ, where δ = |θ′ − θ| is E's angular displacement; the sign of φ is random (rng 4810). The lepton therefore moves by the same 3-D angle as in E (matched to 10⁻⁶ rad), while θ, m_vis (to 10⁻⁴ GeV), the lepton energy and B stay fixed. C is the right comparison because it removes the response to moving the lepton by that angle at all (position, axis shift, isolation change), leaving the response to the change in opening angle and so in mass.
- **TN** (translation control): the unedited particles rebuilt about the E-edited jet's axis. It gives the response to the axis shift that E causes.
- A jet enters an arm only if E and C both exist and the lepton stays within ΔR < 0.8 of the edited jet axis in both.
- **Arms** (windows on unedited m_vis, edge X = 152.6 GeV for t→bℓν and 125 GeV for H→ℓνqq′): **below** [X − 43, X − 23] → +20 GeV (top 110–130 → 130–150; H 82–102 → 102–122); **across** [X − 13, X − 3] → +20 GeV (top 140–150 → 160–170; H 112–122 → 132–142); **reverse** (exploratory) [X + 7, X + 27] with Δm = −20 GeV (back below the edge). Jets per arm: top below 700 (of 712 in the window), across 233 (of 240), reverse 117 (of 143); H below 811 (of 988), across 282 (of 321), reverse 23 (of 91).
- Side effects on the lepton features (mean change, E / C): top across I_ℓ −0.067 / −0.051, f_ℓ +0.001 / −0.000. H below I_ℓ −1.02 / +0.04; H across I_ℓ −0.61 / −0.05. In H jets, opening the lepton angle moves the lepton out of the q q′ particles and isolates it. MLP(m, lepton) sees this change.

## Models and references

- Trained networks: ParT_full and PN_full (they see the isElectron / isMuon flags); ParT_kin and PN_kin (four-momenta only, so the lepton is not identified for them).
- Null: ParT_perm1..6, ParT_full with every weight tensor's entries permuted (torch seeds 5..10).
- References: LR(m), MLP(m) and MLP(m, lepton). They are trained on unedited jets from JetClass train_100M `<sample>_000`, 15,000 random rows per class (rng 4821..4824), signal vs QCD + t→bqq′, with 80/20 train/held-out splits.
  - LR(m): logistic regression on m_vis. Its log-odds are linear in m, with no edge.
  - MLP(m): an MLP with hidden layers (32, 32) on m_vis.
  - MLP(m, lepton): the same MLP on (m_vis, f_ℓ, I_ℓ, has-lepton).
  - Held-out AUC: t→bℓν 0.560 / 0.794 / 0.995 for LR(m) / MLP(m) / MLP(m, lepton); H→ℓνqq′ 0.657 / 0.883 / 0.995.

## Unedited models on this sample (accuracy reproduces)

| model | 10-class acc (12,000 jets) | per class t→bℓν / H→ℓνqq′ / QCD / t→bqq′ | AUC t→bℓν vs QCD + t→bqq′ | AUC H→ℓνqq′ vs QCD + t→bqq′ |
|---|---|---|---|---|
| ParT_full | 0.926 | 0.986 / 0.982 / 0.779 / 0.955 | 0.99998 | 0.99990 |
| PN_full | 0.919 | 0.982 / 0.977 / 0.768 / 0.948 | 0.99998 | 0.99993 |
| ParT_kin | 0.859 | 0.941 / 0.870 / 0.723 / 0.903 | 0.99919 | 0.99771 |
| PN_kin | 0.834 | 0.915 / 0.826 / 0.697 / 0.898 | 0.99802 | 0.99528 |
| ParT_perm1..6 | 0.00–0.25 | – | 0.50–0.59 | 0.50–0.62 |

The ordering and size match the published JetClass results: ParT_full is above PN_full, both are above the kin models, and QCD has the lowest per-class accuracy. They are also in line with earlier loop-D runs (H41: ParT_full 79.8 % on W/Z/Hbb/QCD). Input path: rebuilding about the own E-scheme axis instead of using the stored inputs changes ParT_full's logits by 0.006 on average (99th percentile 0.17, max 1.87) and its 10-class call on 0.017 % of jets (2 of 12,000). Accuracy is 0.9256 either way.

## Numbers

E − C, mean change in signal log-odds [95 % bootstrap over jets, B = 1000]. Holm-adjusted p is given where the cell is in the family. "perm" is the range of the 6 permuted ParT.

**t→bℓν (edge 152.6 GeV)**

| arm (n) | ParT_full | PN_full | ParT_kin | PN_kin | LR(m) | MLP(m) | MLP(m, lep) | perm |
|---|---|---|---|---|---|---|---|---|
| below +20 (700) | −0.15 [−0.23, −0.09], p_H 0.007 | +1.57 [+1.48, +1.67], 0.007 | +0.01 [−0.09, +0.12], 1 | −0.02 [−0.10, +0.05], 1 | −0.04 | −0.91 | −1.00 | [0.00, +0.10] |
| across +20 (233) | −0.65 [−0.78, −0.53], 0.007 | +0.85 [+0.70, +0.99], 0.007 | −1.84 [−2.05, −1.62], 0.007 | −1.46 [−1.64, −1.27], 0.007 | −0.04 | −1.47 | −1.20 | [−0.05, +0.03] |
| reverse −20 (117) | −0.46 [−0.75, −0.17], 0.015 | −1.49 [−1.82, −1.18], 0.007 | +0.03 [−0.39, +0.43], 1 | −0.16 [−0.48, +0.11], 1 | +0.04 | +1.28 | +0.57 | [−0.18, +0.02] |
| across − below | −0.50 [−0.65, −0.35], p_H 0.015 | −0.72 [−0.91, −0.54], 0.015 | −1.85 [−2.11, −1.62], 0.015 | −1.44 [−1.64, −1.24], 0.015 | 0.00 | −0.56 [−0.58, −0.53] | −0.20 [−0.28, −0.12] | – |

- Network minus MLP(m) per jet, across arm: ParT_full +0.82 [+0.70, +0.95] (p_H 0.007), PN_full +2.32 [+2.19, +2.47], ParT_kin −0.37 [−0.59, −0.14] (p_H 0.015), PN_kin +0.01 [−0.17, +0.19] (p_H 1).
- Network minus MLP(m) per jet, below arm: +0.76, +2.49, +0.93 and +0.90.
- Difference of the two (across − below, beyond MLP(m); CI from the two independent bootstraps combined in quadrature): ParT_full +0.06 ± 0.15, PN_full −0.17 ± 0.17, ParT_kin −1.29 ± 0.24, PN_kin −0.89 ± 0.19.
- E alone and C alone, across arm: ParT_full −0.61 / +0.04, PN_full +0.97 / +0.12, ParT_kin −2.17 / −0.34, PN_kin −1.59 / −0.13.
- TN, across arm: ParT_full +0.00, PN_full −0.79 [−0.91, −0.67], ParT_kin −0.28, PN_kin −0.09.
- McNemar, across arm, pass rate at 50 % signal efficiency (C → E):
  - ParT_full 0.601 → 0.464 (39 pass→fail, 7 fail→pass; p 1.8e-6)
  - PN_full 0.725 → 0.803 (2 / 20; p 1.2e-4)
  - ParT_kin 0.541 → 0.210 (79 / 2; p 2.7e-21)
  - PN_kin 0.622 → 0.313 (77 / 5; p 1.2e-17)
- 10-class t→bℓν calls stay at 99–100 % under E and C in every top arm: these jets sit at s_Tbl ≈ 13–15, far above the class boundary.
- Signal-vs-rest score, E − C: below ParT_full +1.54, PN_full +3.01, ParT_kin +1.59, PN_kin +1.13; across −0.44, +0.84, −1.29, −1.01.
- Data curve, unedited t→bℓν jets with a lepton: slope of s_Tbl vs m_vis on [152.6, 192.6] minus that on [112.6, 152.6], in log-odds per GeV.
  - ParT_full −0.009 [−0.038, +0.021], PN_full −0.135 [−0.186, −0.083], ParT_kin −0.029 [−0.068, +0.005], PN_kin −0.033 [−0.059, −0.007].
  - MLP(m) +0.025 [+0.022, +0.028], ParT_perm1 +0.001.
  - Jets per side: 1,223 below, 292 above.

**H→ℓνqq′ (edge 125 GeV)**

| arm (n) | ParT_full | PN_full | ParT_kin | PN_kin | LR(m) | MLP(m) | MLP(m, lep) | perm |
|---|---|---|---|---|---|---|---|---|
| below +20 (811) | −1.51 [−1.67, −1.36], p_H 0.007 | −1.63 [−1.84, −1.42], 0.007 | −0.87 [−1.05, −0.66], 0.007 | −0.66 [−0.85, −0.48], 0.007 | −0.23 | −0.18 | −0.52 | [−0.03, +0.05] |
| across +20 (282) | −5.66 [−6.09, −5.20], 0.007 | −5.54 [−5.98, −5.06], 0.007 | −4.78 [−5.18, −4.36], 0.007 | −4.46 [−4.85, −4.03], 0.007 | −0.23 | −2.31 | −2.71 | [−0.03, +0.04] |
| reverse −20 (23, not in Holm) | +0.14 [−1.38, +1.88] | −0.72 [−2.21, +0.87] | −0.59 [−2.29, +1.10] | −0.44 [−1.91, +0.91] | +0.23 | +2.28 | +1.97 | [−0.45, +0.01] |
| across − below | −4.14 [−4.61, −3.69], p_H 0.015 | −3.90 [−4.45, −3.42], 0.015 | −3.91 [−4.36, −3.42], 0.015 | −3.80 [−4.26, −3.36], 0.015 | 0.00 | −2.13 [−2.16, −2.09] | −2.19 [−2.26, −2.12] | – |

- Network minus MLP(m) per jet, across arm: ParT_full −3.35 [−3.77, −2.89], PN_full −3.22 [−3.65, −2.77], ParT_kin −2.47 [−2.88, −2.05], PN_kin −2.15 [−2.55, −1.75]; all p_H 0.007.
- Network minus MLP(m, lepton), across arm: −2.94, −2.82, −2.07 and −1.75.
- Network minus MLP(m) per jet, below arm: −1.33, −1.45, −0.68 and −0.48.
- Across − below beyond MLP(m) (quadrature CI): ParT_full −2.02 ± 0.47, PN_full −1.78 ± 0.49, ParT_kin −1.79 ± 0.46, PN_kin −1.68 ± 0.44.
- E alone and C alone, across arm: ParT_full −5.70 / −0.05, PN_full −5.55 / −0.02; TN −0.17 and −0.39.
- McNemar, across arm, pass rate at 50 % signal efficiency (C → E): ParT_full 0.507 → 0.039 (135 / 3; p 2.5e-36), PN_full 0.518 → 0.050, ParT_kin 0.560 → 0.113, PN_kin 0.578 → 0.025. In the below arm, ParT_full goes 0.578 → 0.303 (p 5.6e-46).
- 10-class calls, across arm, under C → E:
  - ParT_full H→ℓνqq′ 0.982 → 0.794 and t→bℓν 0.007 → 0.181
  - PN_full H→ℓνqq′ 0.972 → 0.819 and t→bℓν 0.018 → 0.170
  - ParT_kin H→ℓνqq′ 0.904 → 0.291, with t→bℓν 0.383 and W→qq 0.252 under E
- Data curve, slope change at 125 GeV (on [85, 125] vs [125, 165]): ParT_full −0.085 [−0.122, −0.046], PN_full −0.086, ParT_kin −0.144, PN_kin −0.129, MLP(m) −0.088 [−0.094, −0.084], MLP(m, lepton) −0.033, ParT_perm1 +0.004.

Holm family: 36 tests (4 trained nets × 2 classes × {E − C below, across, reverse (n ≥ 30 only), across − below, across net − MLP(m)}). 31 have p_H < 0.05. The five that do not are t→bℓν below ParT_kin and PN_kin, t→bℓν reverse ParT_kin and PN_kin, and t→bℓν across net − MLP(m) PN_kin. Permutation p uses 5,000 sign flips (1,000 arm-label permutations for across − below), so the smallest reachable p_H values are 0.007 and 0.015.

## Reading

**H→ℓνqq′ jets.** All four networks lower the H score more when a +20 GeV mass shift crosses 125 GeV than when it stays below. The step from the below arm to the across arm is −3.8 to −4.1 log-odds. The mass-only reference gives −2.13, and the reference that also sees lepton pT fraction and isolation gives −2.19. The excess beyond MLP(m) is about −1.7 to −2.0 for every network, including the kin models without lepton identification. Permuted networks stay within ±0.05.

H0 (the response follows the MLP(m) or MLP(m, lepton) prediction) is excluded for both edited arms. H1's predicted pattern holds for the H edge: the across drop is larger than the below drop and larger than the references. Where the calls go: 17–18 % of the full models' edited jets become t→bℓν, whose visible-mass range extends to 152.6 GeV. The data curve of the unedited jets shows the same slope change at 125 GeV as MLP(m), so on natural jets the networks' mass dependence at the edge is not steeper than the data's.

Not separated by this edit: the H "E" edit opens the lepton away from the q q′ pair and isolates it (I_ℓ −0.61 across, −1.02 below). MLP(m, lepton) sees this through I_ℓ, but not through other lepton–quark geometry. The reverse H arm has 23 jets and its intervals span ±1.5.

**t→bℓν jets.** For ParT_full the across − below step (−0.50 [−0.65, −0.35]) equals the MLP(m) step (−0.56). Beyond MLP(m) the two arms differ by +0.06 ± 0.15, and its data curve has no slope change at 152.6 GeV (−0.009 [−0.038, +0.021]). This is the H0 pattern. The H1 expectation for full models (E − C < −1 across) is not met: ParT_full −0.65, PN_full +0.85.

PN_full raises its t→bℓν score when the lepton–b angle opens, both below (+1.57) and across (+0.85). It lowers the score when the angle closes (reverse −1.49). It also responds −0.79 to the axis shift alone (TN, not removed by E − C). It therefore responds to the opening angle in the direction opposite to an endpoint cut. Its unedited data curve, however, bends down at the edge (−0.135 per GeV).

The kin models (no PID) do not respond below the edge (+0.01, −0.02) and drop across it (−1.84, −1.46). Beyond MLP(m) that is −1.29 ± 0.24 (ParT_kin) and −0.89 ± 0.19 (PN_kin). This is the opposite of the plan's "kin smaller" expectation. When jets above the edge are moved 20 GeV back below it (reverse arm), the kin models do not rise (+0.03, −0.16), whereas MLP(m) predicts +1.28. So their step is one-directional. The reverse-arm jets are those that exceed the endpoint through radiation, and the reverse edit closes the lepton angle and de-isolates the lepton (I_ℓ +0.29).

10-class t→bℓν calls do not change in any top arm: these jets have s_Tbl ≈ 13–15. The responses measured here are changes in confidence, except at the 50 % signal-efficiency working point, where they appear as call changes (McNemar).

**Competing predictions:**

- "Smooth mass dependence (MLP(m))": holds for ParT_full on t→bℓν. It is excluded for all four networks on H→ℓνqq′, and for the kin models on t→bℓν (across arm).
- "Lepton presence and isolation only (MLP(m, lepton))": excluded on H→ℓνqq′, where the networks go beyond it by −1.75 to −2.94 across.
- "Endpoint used symmetrically": not supported on t→bℓν. The reverse arm does not rise for any network.

## Issues and limits

- Because B is all non-lepton particles, m_vis is the jet mass. The edit therefore changes "the mass of the visible decay products" and "the jet mass" together. Endpoint use and mass use are distinguished only through the comparison with the references and between arms.
- The E edit changes lepton isolation and the lepton–quark geometry (most for H jets). MLP(m, lepton) covers I_ℓ and f_ℓ but no other lepton–quark angles.
- E − C does not remove the axis-shift response. TN is reported separately and is largest for PN_full (−0.79 top across, +0.80 top reverse).
- The across − below "beyond MLP(m)" difference uses a quadrature combination of two independent bootstrap intervals, not a joint resample.
- The reverse H arm has n = 23 and is not in the Holm family. Only 23 of 91 window jets could be edited, because closing the lepton angle by 20 GeV often needs cos θ′ > 1.
- One seed and one edit size (|Δm| = 20 GeV). No C2ST (classifier two-sample test) was run on the edited jets, so how natural they look (on-manifold) is not measured.

## Figures

- `results/physics/h48_jet_edit.png`: one t→bℓν jet from the across arm (pT 543 GeV), in Δη–Δφ about the jet axis. Unedited: m_vis 144.4 GeV. E: opening angle 24.8° → 28.4°, m_vis 164.4 GeV. C: the lepton moved by the same 3.6° around the b-system direction, m_vis 144.4 GeV.
- `results/physics/h48_response.png`: left, E − C per arm and model with references and the permuted-network band; right, mean s_Tbl vs m_vis on unedited jets, with the 152.6 GeV line.
