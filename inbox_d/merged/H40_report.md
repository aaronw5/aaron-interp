# H40: quark–gluon taggers and particle ID — jet charge, charge conjugation, charged/neutral and kaon/proton relabels at fixed four-momenta (new claim D16, exploratory)

- **Status:** done.
- **Jobs:**
  - `anrunw-ai-d-h40-run`: GPU, 33 min, exit 0.
  - `d-h40-pz`: CPU, normal-approximation p and Holm.
  - `d-h40-fig`: CPU, figure.
  - Smoke run: `d-h40-smoke2`, 300 jets per class. The first smoke pod, `d-h40-smoke`, failed before it wrote a log and was resubmitted unchanged.
- **Script:** `interp/h40_qg_pid_charge.py`, with steps `run`, `pz` and `fig`.
- **Results (in `results/physics/`):**
  - `h40_qg_pid_charge.json`
  - `h40_perjet_py.npz`
  - `h40_perjet_hw.npz`
  - `h40_pid_edits.png`
- **Logs:** `logs/d-h40-run.log`, `logs/d-h40-pz.log`, `logs/d-h40-fig.log`.

## Conclusion
The quark–gluon ParticleTransformer and ParticleNet with particle-ID inputs use the sign of the jet charge. We replaced every particle by its antiparticle without changing any momentum. This lowered the quark score of positively charged jets and raised it for negatively charged ones, by about 0.25 in log-odds on average (taken with the sign of the jet charge; per jet, Δs × sign Q₀.₃). The size matches a logistic regression on jet charge alone. The networks learned that positive jet charge means "quark" because the quark jets in this Z+jet sample come mostly from u quarks. An antiquark jet is still a quark jet, so this part of the tagger relies on a property of the sample, not of quark versus gluon radiation.

The networks also respond to which neutral species a soft particle is labelled as. Relabelling soft charged pions as photons pushes toward "quark"; relabelling them as neutral hadrons (K⁰_L) pushes toward "gluon". Both relabels change the charged fraction by the same amount, so the charged fraction itself is not the quantity read. A boosted-decision-tree classifier on species counts moves the same way.

These networks cannot tell kaons or protons from pions; their inputs do not distinguish them. LorentzNet, which does see the full species, responds to kaon/proton relabels.

What is uncertain:
- The edits produce species–momentum combinations that the generator may not produce.
- A network with permuted weights also responds to these edits. Measured relative to each network's own score spread, it responds about as much as the trained ones. The trained networks stand apart by the direction of their response, which matches the reference classifiers. They do not stand apart by its size.
- How much of the Pythia-to-Herwig change in particle-ID tagger output comes from species content is estimated here only with a linear approximation.

## Procedure
**Samples** (random rows, both classes; never first-k):
- **Pythia:** `interp.data_qg.sample(per_class=10000, seed=4001)`, the Komiske–Metodiev–Thaler test split, through `h14_qg_generator.load_sample('py', 10000, 4001)`.
- **Herwig 7.1:** Zenodo 3066475 files 18–19, 10,000 jets per class, seed 4001, through `load_sample('hw', 10000, 4001)`.
- **Total:** N = 20,000 jets per generator.

**Score:** s = log p(q) − log p(g). A positive Δs means more quark-like.

**Models:**
- **ParT kinpid and ParticleNet kinpid:** H2 weaver trainings on Pythia, final checkpoints (best epoch 17 of 20). Their particle-ID inputs (the "exp" set):
  - charge;
  - isChargedHadron: |pid| ∈ {211, 321, 2212};
  - isNeutralHadron: {130, 2112};
  - isPhoton, isElectron, isMuon.

  The four-vectors are massless. A K±/p → π± relabel at the same charge therefore leaves their inputs identical (zero by construction).
- **LorentzNet** (released QG weights): a one-hot of |pid| over 8 species, plus PDG masses. It does not read the sign of the PDG id, so E1 and E2 are zero for it by construction. It sees E4.
- **Zero controls:**
  - ParT_kin, PN_kin and EFN (H2 energyflow): blind to particle ID.
  - ParT kinpid with every weight tensor's entries permuted, as the null band.

**References.** Trained on 50,000 + 50,000 random non-validation rows of the KMT train split and scored on the same sample jets. Q_κ = Σ_i q_i (p_T,i / p_T,J)^κ.
- **Logistic regressions:** LR(Q₀.₃), LR(Q₀.₅), LR(|Q₀.₃|), LR(n_ch/N), LR(N).
- **GB_kin:** GBDT on (N, log girth, log C₁^β for β ∈ {0.5, 1, 2}, p_T^D, m). This stands in for GBDT(EFP+N), because energyflow is not in the GPU environment.
- **GB_sp:** GB_kin plus 10 species counts (π, K, p, γ, K⁰_L, n, e, μ, n₊, n₋).
- **GB_spQ:** GB_sp plus Q₀.₃ and Q₀.₅.

**Edits.** Only PDG ids change. Every particle keeps its (p_T, y, φ), N is unchanged, and so is every kinematic or IRC-safe observable; the code asserts that (p_T, η, φ) are unchanged. LorentzNet recomputes E from the PDG mass, which changes E by a relative O(m²/2E²) amount.

| id | edit | control |
|---|---|---|
| E1 | charge conjugation: pid → −pid for all particles except γ and K⁰_L | none (C-symmetric models give 0); reported raw and as Δs·sign(Q₀.₃) |
| E2k, k = 1, 2, 3 | pid → −pid for the k hardest charged particles | C2: one + and one − charged particle flipped together, with \|ΔQ_κ\| < 0.01 for κ = 0.3 and 0.5 (the highest-p_T such pair among the 10 hardest of each sign); applied to 78 % of jets |
| E3g_m / E3n_m, m = 2, 4, 6 | m soft π± (p_T below the jet median; m/2 of each sign) → γ / K⁰_L | C3_m: the same pions π⁺ ↔ π⁻ (n_ch and total charge fixed) |
| E3r_m | m soft photons → π⁺ (m/2) and π⁻ (m/2) | C3r_m: the same photons → K⁰_L (also reported as its own edit, "γ→K⁰_L") |
| E4a | all K± and p/p̄ → π± of the same charge | C4: a no-op relabel through the same code path |
| E4b | m_j random π± → K± of the same charge, where m_j is the jet's K±/p count | C4 |
| E4n | all n/n̄ → K⁰_L | C4 |
| E5sh | PDG ids randomly permuted among the jet's particles (species counts and total charge fixed) | none |

**Metric.** The per-jet difference Δs(edit) − Δs(control).
- Its mean, with a 95 % bootstrap CI over jets (B = 1000).
- A two-sided sign-flip permutation p (P = 2000; floor 5×10⁻⁴).
- A normal-approximation p (n ≥ 9,900 jets per cell). Holm correction is applied over contrasts × {ParT kinpid, PN kinpid, LorentzNet, permuted ParT}: 96 tests on Pythia and 44 on Herwig.
- McNemar on calls (edit vs control) at the threshold that gives quark efficiency 0.5 on the unedited sample (the R50 working point).
- **Per-unit slopes** (OLS through the origin, bootstrap CI):
  - E1 and E2: Δs per unit ΔQ₀.₃.
  - E3: Δs per unit Δ(n_ch/N).

## Checks: unedited models on these samples (AUC / accuracy at s = 0 / R50)
| model | Pythia (20,000) | known | Herwig (20,000) | known |
|---|---|---|---|---|
| ParT kinpid | 0.9104 / 0.837 / 42.2 | 0.9123 test, 0.9121 published, R50 41.3 | 0.8029 / 0.712 / 12.2 | 0.8049 (H14) |
| PN kinpid | 0.9058 / 0.830 / 35.5 | 0.906 (H5v) | 0.8047 / 0.711 / 12.4 | – |
| ParT kin | 0.9009 / 0.825 / 34.2 | 0.9013 | 0.8065 / 0.716 / 11.5 | 0.8079 |
| PN kin | 0.9007 / 0.824 / 35.0 | 0.9015 | 0.8080 / 0.717 / 11.9 | 0.8080 |
| LorentzNet | 0.9157 / 0.842 / 45.5 | 0.9161 released, 0.917 (H5v) | 0.7977 / 0.697 / 11.3 | – |
| EFN | 0.8821 / 0.805 / 28.8 | 0.8831 (H2 test) | 0.8022 / 0.715 / 11.0 | 0.8023 |
| ParT kinpid permuted | 0.509 / 0.500 / 2.1 | – | 0.496 / 0.496 / 2.0 | – |
| LR(Q₀.₃) / LR(\|Q₀.₃\|) / LR(n_ch/N) / LR(N) | 0.544 / 0.521 / 0.503 / 0.840 | | 0.534 / 0.518 / 0.488 / 0.790 | |
| GB_kin / GB_sp / GB_spQ | 0.873 / 0.881 / 0.881 | | 0.806 / 0.803 / 0.804 | |

**Zero models:**
- Largest \|Δs\| over all edits: ParT_kin, PN_kin and EFN all 0.0 (Pythia and Herwig).
- E4a, E4b and E4n give exactly 0 for ParT kinpid, PN kinpid and permuted ParT (inputs identical).
- E1 and E2 give 0 for LorentzNet; its largest |Δs| is below 10⁻⁵.

**AUC from particle ID** (sample AUC, unedited):
- **kin → kinpid:** ParT 0.9009 → 0.9104 (+0.0095), PN 0.9007 → 0.9058 (+0.0051) on Pythia. On Herwig: ParT 0.8065 → 0.8029, PN 0.8080 → 0.8047.
- **GB_kin → GB_sp (species counts added):** +0.0080 on Pythia, −0.0028 on Herwig.

## Data checks (correlational only)
| | Pythia quark | Pythia gluon | Herwig quark | Herwig gluon |
|---|---|---|---|---|
| mean Q₀.₃ [95 % CI] | +0.096 [0.086, 0.106] | +0.017 [0.007, 0.026] | +0.064 [0.053, 0.074] | +0.004 [−0.005, 0.015] |
| fraction with Q₀.₃ > 0 | 0.573 | 0.509 | 0.549 | 0.500 |
| mean \|Q₀.₃\| | 0.425 | 0.406 | 0.417 | 0.399 |
| mean n_ch/N | 0.4915 | 0.4908 | 0.4906 | 0.4858 |
| particle fractions π / K / p | 0.401 / 0.048 / 0.030 | 0.399 / 0.050 / 0.031 | 0.413 / 0.041 / 0.024 | 0.416 / 0.035 / 0.023 |
| γ / K⁰_L / n | 0.461 / 0.024 / 0.030 | 0.458 / 0.025 / 0.031 | 0.473 / 0.019 / 0.024 | 0.479 / 0.017 / 0.022 |
| mean N | 33.4 | 53.2 | 33.4 | 48.7 |

- Quark jets carry a positive mean jet charge, as expected for u-valence-dominated Z+q.
- The per-class species fractions differ by ≤ 0.003 within each generator.
- Herwig has more π and γ and fewer K, p, K⁰_L and n than Pythia. The difference is 0.005–0.022 per species and is larger for gluon jets.

## Numbers (Pythia, 20,000 jets; mean Δs edit − control [95 % CI]; Holm-adjusted normal p)
| contrast | ParT kinpid | PN kinpid | LorentzNet | permuted ParT | GB_spQ | GB_sp | LR(Q₀.₃) |
|---|---|---|---|---|---|---|---|
| E1 raw | −0.060 [−0.066, −0.053] | −0.052 [−0.057, −0.046] | 0 | −0.002 | −0.034 | −0.009 | −0.042 |
| E1 × sign Q₀.₃ | **−0.246** [−0.253, −0.240], H < 1e−300 | **−0.239** [−0.243, −0.234] | 0 | −0.015 [−0.016, −0.014] | −0.243 | −0.069 | −0.308 |
| E2k1 − C2 (ΔQ₀.₃ −0.061) | +0.171 [+0.161, +0.181], H 2e−221 | +0.002 [−0.006, +0.009], H 1 | 0 | −0.001 | +0.097 | −0.001 | −0.023 |
| E2k2 − C2 | +0.131 [+0.120, +0.141] | −0.006 [−0.014, +0.002], H 1 | 0 | −0.002 | +0.051 | +0.001 | −0.031 |
| E2k3 − C2 | +0.149 [+0.137, +0.160] | −0.012 [−0.021, −0.004], H 0.09 | 0 | −0.001 | +0.088 | +0.004 | −0.030 |
| E3g_4 − C3 (π±→γ; Δn_ch/N −0.096) | +0.154 [+0.150, +0.158] | +0.147 [+0.144, +0.150] | +0.090 | +0.037 | +0.219 | +0.216 | 0.000 |
| E3n_4 − C3 (π±→K⁰_L; same Δn_ch/N) | −0.238 [−0.245, −0.232] | −0.262 [−0.268, −0.256] | −0.267 | +0.047 | −0.221 | −0.245 | 0.000 |
| E3r_4 − C3r (γ→π± minus γ→K⁰_L) | +0.210 [+0.204, +0.215] | +0.212 | +0.251 | −0.066 | +0.228 | +0.263 | 0.000 |
| E3r_4 raw (γ→π±) | −0.112 [−0.115, −0.109] | −0.113 | −0.071 | −0.046 | −0.207 | −0.207 | 0.000 |
| γ→K⁰_L, m = 4 (C3r_4 raw) | −0.322 [−0.329, −0.314] | −0.324 | −0.322 | +0.019 | −0.435 | −0.470 | 0 |
| E3g_6 − C3 / E3n_6 − C3 | +0.225 / −0.279 | +0.212 / −0.322 | +0.130 / −0.332 | +0.050 / +0.062 | +0.337 / −0.086 | +0.328 / −0.137 | 0 |
| E4a − C4 (K, p → π) | 0 (inputs identical) | 0 | +0.069 [+0.057, +0.081] | 0 | +0.368 | +0.386 | 0 |
| E4b − C4 (π → K) | 0 | 0 | −0.306 [−0.312, −0.300] | 0 | −0.145 | −0.135 | 0 |
| E4n − C4 (n → K⁰_L) | 0 | 0 | −0.390 [−0.407, −0.375] | 0 | +0.064 | +0.056 | 0 |
| E5sh (ids shuffled in jet) | +0.100 [+0.088, +0.113] | +0.117 | −0.251 | −0.035 | +0.026 | 0.000 | −0.009 |

**Holm.** On Pythia, 79 of 96 tests have Holm p < 0.05. Not rejected:
- the zero-by-construction cells: LorentzNet E1/E2 and kinpid/permuted E4a/b/n;
- PN kinpid E2k1–k3.

On Herwig, 29 of 44 tests are rejected. Not rejected: the zero-by-construction cells, PN kinpid E2k1, LorentzNet E4a (H 0.3) and permuted ParT E1 raw. Every other trained-network cell in the table has Holm p ≤ 1e−17.

**McNemar** (calls at ε_q = 0.5; edit vs control; to quark / to gluon):

| contrast | ParT kinpid | PN kinpid | LorentzNet | permuted ParT | GB_sp |
|---|---|---|---|---|---|
| E1 | 5.1 % (377/639, p 2e−16) | 4.4 % (342/531, p 2e−10) | – | 19.7 % (1872/2076, p 0.001) | – |
| E2k1 − C2 | 6.6 % (668/374, p 6e−20) | 5.7 % (433/455, p 0.48) | – | – | – |
| E3g_4 − C3 | 2.2 % (279/45) | 2.0 % (266/30) | – | 15.4 % (2106/202) | – |
| E3n_4 − C3 | 3.3 % (80/411) | 3.0 % (50/404) | – | 18.9 % (2739/85) | – |
| γ→K⁰_L, m = 4 | 5.3 % (84/931) | 5.9 % (120/1005) | 5.6 % (110/958) | – | 8.1 % (2/1560) |
| E4b − C4 | – | – | 5.3 % (88/887) | – | – |

The permuted network's scores sit in a narrow band around its threshold (score SD 0.13 vs 2.3–2.4 for the trained networks), so small shifts change many of its calls.

**E1 split by class and sign of Q₀.₃** (raw Δs, Pythia; ParT kinpid / PN kinpid / LR(Q₀.₃) / permuted):

| class | Q > 0 | Q < 0 |
|---|---|---|
| quark | −0.372 / −0.343 / −0.337 / −0.018 | +0.247 / +0.239 / +0.286 / +0.016 |
| gluon | −0.183 / −0.185 / −0.308 / −0.014 | +0.165 / +0.173 / +0.293 / +0.012 |

On Herwig, ParT kinpid gives quark Q > 0 −0.328 and Q < 0 +0.247, and gluon −0.192 / +0.197.

**E2k1 − C2 split by the charge of the flipped particle** (Pythia):
- **Leading charged particle positive** (flip to negative): quark −0.360 (ParT), −0.514 (PN), −0.461 (LR(Q)).
- **Leading charged particle negative:** quark +0.703 (ParT), +0.535 (PN), +0.449 (LR(Q)).
- Gluon jets: ParT −0.125 / +0.517, PN −0.299 / +0.356.

**Per-unit slopes** (Pythia; Herwig in brackets):

| | ParT kinpid | PN kinpid | LR(Q₀.₃) | LR(Q₀.₅) | GB_spQ | permuted |
|---|---|---|---|---|---|---|
| Δs / ΔQ₀.₃ under E1 | 0.286 [0.280, 0.292] (0.291) | 0.282 (0.287) | 0.370 | 0.364 | 0.264 | 0.018 |
| Δs / ΔQ₀.₃ under E2 | 0.321 (0.389) | 0.365 (0.398) | 0.370 | – | 0.282 | 0.015 |

The LR(Q₀.₃) slope of 0.370 is its training coefficient.

Under E3, the slope Δs / Δ(n_ch/N) has opposite signs for the photon and K⁰_L targets. LR(n_ch/N) gives −0.042 for both targets.

| E3 target | ParT kinpid | PN kinpid | LorentzNet | GB_sp | permuted |
|---|---|---|---|---|---|
| γ | −1.53 | −1.46 | −0.90 | −1.74 | −0.37 |
| K⁰_L | +2.15 | +2.36 | +2.43 | +3.00 | −0.48 |

**Herwig arm** (20,000 Herwig jets, Pythia-trained models; edit − control):

| contrast | ParT kinpid | PN kinpid | LorentzNet | permuted ParT | GB_sp |
|---|---|---|---|---|---|
| E1 × sign Q | −0.243 | −0.239 | – | −0.013 | – |
| E2k1 − C2 | +0.200 | +0.009 (H 0.5) | – | – | – |
| E3g_4 − C3 | +0.174 | +0.165 | +0.110 | +0.038 | +0.209 |
| E3n_4 − C3 | −0.331 | −0.347 | −0.348 | +0.044 | −0.319 |
| γ→K⁰_L, m = 4 | −0.436 | −0.442 | −0.417 | +0.016 | −0.529 |
| E4b − C4 | – | – | −0.298 | – | −0.190 |
| E4a − C4 | – | – | +0.014 (H 0.3) | – | +0.287 |

The response to each edit has the same sign on both generators and is 0–40 % larger on Herwig for E3.

**Generator gap.** Mean s(Herwig) − s(Pythia) per class:

| model | quark | gluon |
|---|---|---|
| ParT kinpid | −0.033 | +1.299 |
| ParT kin | +0.041 | +1.099 |
| PN kinpid | −0.017 | +1.184 |
| PN kin | +0.028 | +1.081 |
| LorentzNet | +0.004 | +1.489 |

The kinpid-minus-kin difference in this gap is −0.074 (quark) and +0.200 (gluon) for ParT, and −0.045 and +0.104 for PN.

A linear estimate from the Pythia E3 per-particle responses predicts the shift caused by Herwig's species content. It uses +1.53 per unit photon fraction and −2.15 per unit neutral-hadron fraction, both relative to charged, and the Herwig − Pythia fraction differences: Δf_γ +0.012 / +0.022 and Δf_(K⁰_L+n) −0.011 / −0.016. The predicted shift is +0.04 for quark jets and +0.07 for gluon jets. Jet charge adds about −0.01 for quark jets (0.29 × ΔQ̄ −0.032).

## Hypotheses
- **H0** (per edit × model): the kinpid networks' Δ (edit − control) equals the PID-blind zero (0) and lies within the permuted-network band.
- **H1:** Δ beyond control, with Holm correction over edits × models × k.

**Results:**
- **Equality with the zero models** is rejected for ParT and PN kinpid on E1, E3 (all m, both targets, both directions) and E5sh, and for ParT kinpid on E2k1–3 (Holm p < 1e−50 for each).
- **Not rejected:** PN kinpid E2 and every E4 cell for the kinpid networks, which are zero by construction.
- **Permuted band:**
  - In logit units, the trained-network effects are 16–19× the permuted network's on E1 × sign Q and 3–17× on E3.
  - Relative to each model's own score spread (Δ/SD of s), the permuted network's responses are as large: E1 × sign Q −0.116 SD vs −0.102 / −0.104 SD for ParT / PN kinpid.
  - The permuted network's direction disagrees with the trained networks and with GB_sp on E3n (+0.047 vs −0.24) and on E3r − C3r (−0.066 vs +0.21).
  - The "within the permuted band" part of H0 therefore depends on the units. In calibrated log-odds and in direction the trained networks are outside it; in score-SD units they are not.

**Competing predictions:**
- **(a) The networks read signed jet charge:** E1 splits by the sign of Q, and E2 follows LR(Q). Observed for both kinpid networks.
  - The E1 split is −0.37 / +0.25 for quark jets, and the E1 slope is 0.29 vs LR 0.37.
  - E2 per-unit slope: 0.32 (ParT) and 0.37 (PN) vs 0.37.
  - The E2 mean minus control for PN is ~0 because its response is symmetric in the sign of the flipped charge (−0.51 / +0.54). ParT's response is asymmetric (−0.36 / +0.70), so its mean is +0.17.
- **(b) The networks read only |Q| or the charged fraction:** predicts E1 = 0. Excluded (E1 × sign Q −0.25).
  - The charged fraction is also excluded as the quantity read: π→γ and π→K⁰_L change n_ch/N identically but move the score in opposite directions. LR(n_ch/N) carries no class information (AUC 0.503).
- **(c) Particle ID is used only through species counts (no charge):** predicts E1 = E2 = 0. Excluded.
  - The E3 directions do agree with GB_sp (species counts) on every arm and both generators, at 0.5–2.4× its size.
- **(d) Kaon/baryon content:** not testable for the kinpid networks (E4 is input-identical).
  - LorentzNet responds in GB_sp's direction on E4a (+0.07 vs +0.39) and E4b (−0.31 vs −0.14), and against GB_sp on E4n (−0.39 vs +0.06).

## Reading (scoped to what was measured)
- **Particle-ID use with kinematics fixed.** On Pythia and Herwig test jets, relabelling PDG ids at fixed (p_T, y, φ) moves the quark score of the Pythia-trained ParT and ParticleNet kinpid by up to 0.25–0.44 in log-odds and changes 2–9 % of their calls at the R50 working point. The PID-blind ParT_kin, PN_kin and EFN do not move.
- **Signed jet charge.** The response to charge conjugation and to flipping the hardest charged particles follows the sign of the jet charge. Its slope per unit Q₀.₃ is 0.29–0.40 log-odds, against 0.37 for a logistic regression on Q₀.₃. For PN kinpid it is symmetric in the direction of the flip; for ParT kinpid it is asymmetric.
- **Neutral species.** Soft-particle species enters with a direction that depends on which neutral species replaces a charged pion: photon is quark-ward, K⁰_L gluon-ward. A GBDT on species counts plus IRC-safe shapes and N has the same signs. Its AUC gain from species counts (+0.008) is about the size of the kin → kinpid gain (+0.005 to +0.010).
- **Herwig.** The Herwig responses have the same signs. Linearised, the per-particle responses predict that Herwig's higher photon and lower neutral-hadron fractions shift kinpid scores by +0.04 (quark) and +0.07 (gluon). The measured kinpid-minus-kin generator-gap differences are −0.07 / +0.20 (ParT) and −0.05 / +0.10 (PN). The signs agree for gluon jets and disagree for quark jets.

## Issues
- **Off-manifold combinations:** the edits create species–momentum combinations that Pythia and Herwig do not produce, for example a K⁰_L at a soft charged pion's momentum. No classifier two-sample test (C2ST) of edited vs natural jets was run.
- **Permuted-network comparison:** it depends on units (see Hypotheses). A retrained-seed or shuffled-label kinpid network would be a stronger null (H39-type).
- **GBDT stand-in:** GB_kin, GB_sp and GB_spQ use IRC-safe shapes plus N instead of EFPs.
- **Linear generator estimate:** the generator attribution is a linear estimate from Pythia edit responses. The kinpid and kin networks are separate trainings, so their gap difference also contains training differences.
- **Permutation p floor:** the sign-flip p has a floor of 5×10⁻⁴, which puts Holm-adjusted values at ≥ 0.022–0.048. The normal-approximation p of step `pz` (n ≥ 9,900) is used for Holm in the tables.
- **Coverage on Herwig:** E2k2, E2k3, E3 with m = 2 or 6, and E5sh were run on Pythia only.

## Explainer entry (for site/explain.py; to be merged by the ledger)
```python
H40_EXPLAINER = [
 ('Question', 'Quark–gluon ParT and ParticleNet trained with particle-ID inputs ("kinpid") reach AUC 0.906–0.910, about 0.005–0.010 above the same networks without particle ID. Which species information do they use: the sign of the jet charge, the charged fraction, the neutral species, or kaon/proton content? Each is tested by relabelling particles while every momentum stays fixed.', None),
 ('Physics background',
  'The jet charge $Q_\\kappa=\\sum_i q_i\\,(p_{T,i}/p_{T,J})^{\\kappa}$ ($\\kappa=0.3,0.5$) tracks the charge of the initiating parton: in $pp\\to Z+$jet the quark jets come mostly from $ug\\to Zu$, so quark jets have $\\langle Q_{0.3}\\rangle=+0.096$ and gluon jets $+0.017$ (Pythia). An antiquark jet is still a quark jet, so a tagger that prefers positive $Q$ uses the sample\'s u-quark excess, not quark/gluon radiation. '
  'Hadronisation fixes the species mix: about 40 % charged pions, 46 % photons (mostly from $\\pi^0$), 5 % kaons, 3 % protons, 2–3 % $K^0_L$ and neutrons. Pythia and Herwig differ by 0.5–2 points in these fractions. The kinpid inputs are charge, isChargedHadron ($\\pi,K,p$ together), isNeutralHadron ($K^0_L,n$), isPhoton, isElectron and isMuon, so kaons and protons look like pions to these networks.', None),
 ('Method',
  '20,000 Pythia test jets (seed 4001) and 20,000 Herwig 7.1 jets. Only PDG ids are changed; every $(p_T,y,\\phi)$ and $N$ is kept, so all kinematic and IRC-safe observables are unchanged. E1: every particle → its antiparticle. E2: the $k$ hardest charged particles flip charge, minus a pair flip that keeps $Q_\\kappa$ within 0.01. E3: $m$ soft $\\pi^\\pm$ → $\\gamma$ or $K^0_L$, minus a $\\pi^+\\leftrightarrow\\pi^-$ swap; $m$ soft photons → $\\pi^\\pm$, minus photons → $K^0_L$. E4: $K^\\pm,p\\to\\pi^\\pm$ and $\\pi^\\pm\\to K^\\pm$ at fixed charge. '
  'Score $s=\\log p(q)-\\log p(g)$, mean $\\Delta s$ (edit − control) with bootstrap CI, McNemar at $\\varepsilon_q=0.5$, Holm over 96 tests. Zeros: ParT_kin, PN_kin, EFN, and a weight-permuted ParT kinpid. References: LR($Q_\\kappa$), LR($n_{ch}/N$), and a GBDT on shapes + $N$ + species counts (+ $Q$).', None),
 ('Result',
  'Charge conjugation lowers the quark score of $Q>0$ quark jets by 0.37 and raises it for $Q<0$ by 0.25 (ParT kinpid; PN −0.34 / +0.24), i.e. 0.29 log-odds per unit $\\Delta Q_{0.3}$ vs 0.37 for LR($Q_{0.3}$); permuted network 0.02. Soft $\\pi^\\pm\\to\\gamma$ moves the score toward quark (+0.15 at $m=4$) and $\\pi^\\pm\\to K^0_L$ toward gluon (−0.24 to −0.26) at the same change of charged fraction; the species-count GBDT gives +0.22 / −0.25. Kaon/proton relabels give exactly 0 for the kinpid networks (identical inputs); LorentzNet, which sees all species, moves by up to −0.31. ParT_kin, PN_kin and EFN give 0 for every edit.', 'results/physics/h40_pid_edits.png'),
 ('Checks',
  'Unedited AUCs reproduce: ParT kinpid 0.9104 (0.9123 test), PN kinpid 0.9058, ParT_kin 0.9009, LorentzNet 0.9157; Herwig ParT kinpid 0.8029 (0.8049 in H14). The kinematics of every edited jet are asserted unchanged. PID-blind models: max $|\\Delta s|=0$. The same edits on Herwig jets give the same signs. In units of each model\'s own score spread the permuted network moves as much as the trained ones, so what separates the trained networks is the direction of the response, which matches the reference classifiers.', None),
 ('Physical significance',
  'Part of what particle ID adds to these quark–gluon taggers is the sign of the jet charge, which in this sample encodes the u-quark excess of $Z+q$ production rather than quark/gluon radiation, and should not carry over to samples with a different flavour mix (e.g. dijets, or $\\bar u$-rich processes). The rest acts through which neutral species the soft particles are (photons quark-ward, neutral hadrons gluon-ward), a hadronisation-model detail that differs between Pythia and Herwig by 1–2 points per species.', None),
]
```
Entry line for the `'d'` list: `('Result explained: quark–gluon taggers and particle ID — jet charge sign and neutral species at fixed momenta (claim D16; H40)', H40_EXPLAINER, False)`.

**Proposed claim D16 wording (exploratory, for the ledger):**

> Quark–gluon ParT and ParticleNet trained with particle-ID inputs use the sign of the jet charge (0.29 log-odds per unit Q₀.₃ under charge conjugation, vs 0.37 for LR(Q₀.₃)) and the neutral-species label of soft particles (photon quark-ward, K⁰_L gluon-ward, at a fixed charged fraction), with kinematics fixed. They cannot use K/p vs π (identical inputs).

Suggested confidence: 0.5, pending a C2ST of the edits and a retrained-seed or shuffled-label kinpid null.
