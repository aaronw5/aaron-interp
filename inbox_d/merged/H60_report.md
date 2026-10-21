# H60: hadronisation-only Pythia variants on top and QCD jets. Do top taggers follow particle count?

Node H60 (loop D, round D12, exploratory). It extends D22 (QG taggers respond to hadron-level particle count beyond IRC-safe energy flow, from H54/H54v) to top tagging.
Script: interp/h60_top_hadronisation.py.

## Conclusion

We changed only how Pythia turns partons into hadrons. The parton-level event stayed the same, and top and QCD jets ended up with about 9 % more or fewer particles while jet mass, τ32 and the W-pair mass stayed fixed to within 0.2 %. Every Top-Landscape and JetClass top tagger tested moved top jets toward "QCD" when the particle count rose, and toward "top" when it fell. The IRC-safe energy-flow network (EFN) did not move.

The direction is opposite to a classifier that uses the particle count alone (top jets carry more particles than QCD jets overall). It is the same direction as a boosted-tree classifier that reads the count on top of fixed IRC-safe substructure, GBDT(EFP+N). ParT and ParticleNet follow about 0.7–1.1 times that reference's response. In physical terms, the taggers treat extra soft hadrons at fixed three-prong kinematics as a sign of a radiation-rich QCD jet. Top jets lose about a fifth of their tag rate at a fixed working point: ParT's signal efficiency goes from 0.30 to 0.24 when N rises by 9 %.

Three things remain uncertain:
- The jets are particle-level while the networks were trained on Delphes jets, which have about 35 % fewer particles.
- Particle count is not separated from the softer momenta of the extra hadrons.
- On QCD jets the sign of the response differs between architectures.

## Question and hypotheses

On Top Landscape jets, do the taggers respond to the hadron-level particle count N when the IRC-safe content is held fixed?
- **H0 (IRC-safe only).** The net's shift Δs equals the shift of the IRC-safe references (EFN, GBDT(EFP)), so |κ| < 0.2 and Δs lies within the generator-noise band.
- **H1 (count read with the sign of LR(N)).** κ_N = (Δs_net − Δs_EFN)/(Δs_LR(N) − Δs_EFN) lies between 0.4 and 0.9 and is positive, so the net moves top-ward when N rises, because the LR(N) slope is positive for tops.
- **H1′ (count read with the opposite sign, as the H18 split edits suggested).** Δs has the opposite sign to LR(N).

Definitions used throughout:
- **s** is the top log-odds: log p(top) − log p(QCD). For the JetClass nets it is logit(Tbqq) − logit(QCD).
- **Δs** is the mean over event-paired jets of s(variant) − s(nominal V0).
- **LR(N)** is a logistic regression on N alone, a "count-only classifier". It is trained on Top Landscape val (25k + 25k jets, rows as in H19) and has slope +0.0436 per particle. Top Landscape mean N is 54.9 for tops and 43.3 for QCD.
- **EFP** = energy-flow polynomials, the IRC-safe basis: all 102 connected multigraphs up to degree 5, β = 1, normalised.
- **GBDT(EFP)** is a gradient-boosted tree classifier on log EFPs. It is IRC-safe.
- **GBDT(EFP+N)** is the same classifier with N added as an input. It is a count classifier conditional on IRC-safe substructure.
- **κ_EFP+N** = (Δs_net − Δs_GBDT(EFP))/(Δs_GBDT(EFP+N) − Δs_GBDT(EFP)). It is the share of the conditional count response that the net follows. This is the κ used in H54. The plan's κ_N with LR(N) is also reported.
- Each reference comes in two versions: one trained on Top Landscape (Delphes) jets, and one marked "_pl" trained on the independent particle-level nominal sample V0b (5k + 5k jets). The _pl version has N inside its training range.

## Procedure

1. **Generator.** Pythia 8.317 (pythia8mc, env-gen).
   - Collisions: pp at 14 TeV, Monash tune, MPI, ISR and FSR on.
   - Top sample: tt̄ with W → quarks.
   - QCD sample: HardQCD:all.
   - p̂T 450–800 GeV.
   - Jets: anti-kT R = 0.8 with Pythia's fjcore SlowJet, on all visible final-state particles (|η| < 5, real masses). Every jet with 550 < pT < 650 GeV and |η| < 2 is kept.
   - Truth matching for tops: the top quark and its b, q, q′ from the process record all lie within ΔR < 0.8 of the jet, as in the Top Landscape definition.
   - QCD jets carry the slot of the nearest outgoing hard parton.
   - Particle level, no detector. Delphes 3.5.1 exists on conda-forge (checked via the anaconda API in d-h60-check2) but was not used in this round.
2. **Pairing.** Before event i of worker w the generator is reseeded with 1 + (6060 + w)·50,000 + i, as in H54. The parton-level record was identical between V0 and Hz+ in 20/20 checked events per class.
   - Pair key: (worker, event, slot), where slot is t or t̄ for tops and the hard parton for QCD.
   - Pairs per class and setting: 3,816–3,892 for tops and 4,741–4,772 for QCD, out of 5,000 jets each.
3. **Settings.** Each is (StringZ:aLund, StringZ:bLund, StringPT:sigma). Monash is (0.68, 0.98, 0.335 GeV).
   - V0: Monash.
   - V0b: Monash with other seeds (independent events). This gives the generator-noise band and is the training set for the _pl references.
   - Hz+: (0.79, 0.88, 0.247). Hz−: (0.48, 0.98, 0.438). Both reused unchanged from H54.
   - Single-knob controls from H54v: σ only (0.235 / 0.45) and aLund only (1.28 / 0.18).
   - 5,000 top + 5,000 QCD jets per setting.
   - Jobs: d-h60-check2 and d-h60-gen-{v0, v0b, hzp, hzm, sigup, sigdn, aup, adn}, each 1 CPU / 2 GiB, 6–11 min.
4. **Network inputs.** Constituents are passed as massless four-vectors (E = |p|), as Top Landscape constituents are.
5. **Scoring** (job d-h60-score3, GPU).
   - Top Landscape-trained models: ParT_kin and ParticleNet_kin (H4 checkpoints), LorentzNet (released weights), PFN and EFN (H2).
   - JetClass ParT_kin and PN_kin, with s = L(Tbqq) − L(QCD).
   - LorentzNet is scored two ways: in batches of 32, where paired jets share an order key so they sit in the same batch positions (the H4 convention), and at batch size 1 (LNet_b1, an input-convention check).
   - Null: 3 weight-permuted copies of Top ParT (every tensor's entries shuffled, seeds 0–2).
   - Observables per jet: m, m_SD (soft drop, z_cut 0.1, β = 0), τ21, τ32 (WTA kT axes), mW (the exclusive-kT 3-subjet pair mass closest to 80.4 GeV), and N.
   - References: LR(N), LR3 = binned likelihood ratio in (m_SD, τ32, mW), and MLP3 = MLP(64, 64) on (m, τ32, mW).
6. **EFP references** (job d-h60-efp, 8 CPU, venv-ef, energyflow 1.4): GBDT(EFP) and GBDT(EFP+N), trained on Top Landscape val and on V0b.
7. **Statistics** (job d-h60-stats3).
   - Effects: paired Δs per class with a 95 % bootstrap CI over jet pairs (B = 2,000).
   - Tests: net − EFN and net − GBDT(EFP+N)_pl, each with a bootstrap-z p-value and Holm correction. The net − EFN family has 84 tests: 7 networks (the 6 nets plus LNet_b1) × 6 settings × 2 classes. The net − GBDT(EFP+N) family has 96 tests: the same plus EFN.
   - κ with bootstrap CI.
   - Working points: thresholds at ε_S = 0.3 and 0.5 taken from the V0 paired tops. At each, the variant's ε_S, the QCD rejection 1/ε_B, and a McNemar test on paired QCD calls.
   - Generator-noise band: V0b − V0, unpaired, with bootstrap CI.
   - C2ST of V0 vs Top Landscape test jets: HGB on (N, m, m_SD, τ21, τ32, mW), 2-fold AUC.

## Checks: unedited models

AUC / accuracy on Top Landscape test (5k + 5k, rng 6061) compared with known values, and on the particle-level V0 sample:

| model | Top Landscape test | known | V0 particle level |
|---|---|---|---|
| Top ParT | 0.9864 / 94.2 % | 0.9867 / 94.3 % (H25) | 0.9872 / 93.0 % |
| Top ParticleNet | 0.9859 / 94.2 % | 0.9868 / 94.0 % | 0.9848 / 93.2 % |
| LorentzNet | 0.9873 / 94.4 % | 0.9879 / 94.5 % | 0.9821 / 89.5 % |
| PFN | 0.9802 / 92.9 % | 0.9816 / 93.0 % | 0.9774 / 82.4 % |
| EFN | 0.9761 / 92.1 % | 0.9783 / 92.6 % | 0.9788 / 92.7 % |
| JetClass ParT_kin | 0.9826 / 92.8 % | 0.9830 / 92.7 % | 0.9750 / 91.5 % |
| JetClass PN_kin | 0.9832 / 93.3 % | 0.9847 / 93.5 % | 0.9870 / 92.6 % |
| permuted ParT (3) | 0.72–0.78 | 0.717 (H36) | 0.77–0.80 |
| LR(N) / LR3 / MLP3 / GBDT(EFP) / GBDT(EFP+N) | 0.704 / 0.958 / 0.967 / 0.969 / 0.975 | – | 0.670 / 0.970 / 0.975 / 0.974 / 0.972 (accuracy 0.75: N out of range) |

All Top Landscape values are within 0.003 AUC of the earlier samples.

On particle-level jets:
- The AUCs stay at 0.975–0.987.
- The accuracies at s > 0 fall for LorentzNet and PFN. Their score offset moves with the higher N.
- C2ST V0 vs Top Landscape is 0.977 (tops) and 0.974 (QCD) with N included, and 0.656 / 0.749 without N.
- Mean N is 83.4 vs 54.9 for tops and 69.9 vs 43.5 for QCD (particle level vs Delphes). The particle-level sample is clearly off the training distribution in N. All comparisons are paired variant − V0, so the mismatch is common to both arms.
- 2.5 % of V0 jets and 5.6 % of Hz+ jets have N > 128. Top ParT, ParticleNet and the EF nets see only the 128 hardest particles.

## Numbers (95 % CI; top-jet log-odds)

**Top jets, paired with V0.**

| setting | ΔN | Δm [GeV] | Δτ32 | ΔmW [GeV] | EFN | GBDT(EFP)_pl | MLP3 | LR(N) (TL / _pl) | GBDT(EFP+N)_pl |
|---|---|---|---|---|---|---|---|---|---|
| Hz+ | +7.2 (+8.6 %) | −0.14 [−0.44, 0.14] | −0.001 | +0.02 | +0.012 [−0.007, 0.031] | −0.041 | +0.001 | +0.316 / +0.181 | −0.461 [−0.499, −0.424] |
| Hz− | −7.5 (−8.9 %) | +0.04 | +0.001 | −0.10 | +0.003 | +0.007 | −0.012 | −0.325 / −0.186 | +0.438 |
| σ only, N up | +6.2 (+7.4 %) | −0.01 | −0.005 | +0.05 | +0.049 [0.029, 0.067] | −0.016 | +0.040 | +0.269 | −0.386 |
| σ only, N down | −6.0 (−7.2 %) | +0.22 | +0.005 | −0.25 | −0.033 | +0.018 | −0.026 | −0.264 | +0.394 |
| a only, N up | +5.8 (+7.0 %) | +0.32 | +0.010 (+1.9 %) | +0.03 | −0.071 [−0.090, −0.053] | −0.031 | −0.061 | +0.255 | −0.312 |
| a only, N down | −6.2 (−7.4 %) | −0.12 | −0.010 | +0.06 | +0.076 | +0.010 | +0.068 | −0.272 | +0.317 |

V0 per-jet standard deviations, for scale: N 19.3, m 21.7 GeV, τ32 0.15, mW 11.7 GeV.

| network | Hz+ Δs | κ_EFP+N (_pl) | κ_N (LR(N), TL) | Hz− Δs | κ_EFP+N | σ only N up | a only N up |
|---|---|---|---|---|---|---|---|
| Top ParT | −0.435 [−0.487, −0.380] | 0.94 [0.82, 1.06] | −1.47 [−1.66, −1.29] | +0.485 [0.430, 0.537] | 1.11 [0.99, 1.24] | −0.264 | −0.529 |
| Top ParticleNet | −0.417 [−0.486, −0.354] | 0.90 [0.75, 1.05] | −1.41 | +0.528 | 1.21 | −0.207 | −0.513 |
| LorentzNet (batch 32) | −0.818 [−0.887, −0.746] | 1.85 [1.68, 2.03] | −2.72 | +0.783 | 1.80 | −0.640 | −0.764 |
| PFN | −0.579 [−0.622, −0.534] | 1.28 [1.18, 1.39] | −1.94 | +0.650 | 1.49 | −0.453 | −0.469 |
| JetClass ParT_kin | −0.355 [−0.403, −0.305] | 0.75 [0.62, 0.86] | −1.20 | +0.445 | 1.02 | −0.265 | −0.355 |
| JetClass PN_kin | −0.345 [−0.389, −0.299] | 0.72 [0.61, 0.84] | −1.17 | +0.443 | 1.01 | −0.209 | −0.468 |
| permuted ParT (3) | −0.018 to +0.027 | – | – | −0.025 to +0.018 | – | ≤ 0.021 | ≤ 0.011 |

κ_EFP+N computed with the Top Landscape-trained GBDTs is 0.63–0.83 for ParT, PN, ParT_kin and PN_kin on Hz+. In the N-down settings every network moves top-ward. Every network's sign is the same in all six settings.

**QCD jets (Hz+, ΔN +7.2 = +10.3 %, Δm +0.02 GeV).**
- IRC-safe references: EFN +0.003 [−0.011, 0.018], GBDT(EFP)_pl +0.026.
- Count references: LR(N) +0.312, GBDT(EFP+N)_pl −0.133 [−0.155, −0.111].
- Networks:

| network | Δs | 95 % CI | κ_EFP+N |
|---|---|---|---|
| Top ParT | −0.087 | [−0.109, −0.065] | 0.71 |
| ParT_kin | −0.101 | | 0.80 |
| PFN | −0.273 | | 1.88 |
| Top ParticleNet | +0.217 | [0.165, 0.272] | −1.20 |
| LorentzNet | +0.135 | | −0.68 |
| PN_kin | +0.065 | | −0.24 |

On QCD jets the sign therefore differs by architecture: ParT-type nets and PFN go one way, ParticleNet-type nets and LorentzNet the other. Hz− mirrors this.

**LorentzNet input convention.** At batch size 1, LorentzNet's QCD shift is +0.824 (Hz+) and −1.072 (Hz−), compared with +0.135 and −0.163 in batches of 32. Its readout averages node features over the padded length of the batch, so its response to N depends on the batching.

**Generator-noise band (V0b − V0, independent events, unpaired).** Network |Δs| ≤ 0.07 with CIs inside [−0.15, 0.20], for example Top ParT +0.024 [−0.077, 0.131] on tops. N −0.14 [−0.91, 0.66].

**Tests (Holm).**
- Net − EFN: 76/84 tests have Holm p < 0.05.
- Net − GBDT(EFP+N)_pl: 82/96 have Holm p < 0.05. On top jets, ParT and PN are not separated from the conditional count reference in Hz+ (raw p 0.31 and 0.18) or Hz− (0.07 and 0.005). The same holds for ParT_kin and PN_kin in Hz− (0.80, 0.86).

**Working points** (thresholds from V0 tops; top-jet signal efficiency in the variant):

| model | ε_S 0.30 at Hz+ / Hz− | ε_S 0.50 at Hz+ / Hz− |
|---|---|---|
| Top ParT | 0.240 / 0.362 | 0.428 / 0.572 |
| PN | 0.243 / 0.363 | – |
| LorentzNet | 0.210 / 0.409 | – |
| EFN | 0.304 / 0.302 | – |

QCD rejection is not resolved with 5,000 QCD jets: 1–50 QCD jets pass, and the McNemar p-values on paired QCD calls are mostly > 0.05. One exception is PFN at Hz− with ε_S = 0.5: 33 calls up, 7 down, p = 4·10⁻⁵.

## Reading

- **H0 is excluded for all six networks on top jets.** Net − EFN reaches Holm significance in every top cell. The shifts are 5–15 times the IRC-safe shifts and lie outside the generator-noise band.
- **H1 is excluded on top jets.** With LR(N) as the reference, κ_N lies between −1.2 and −2.7.
- **H1′ holds in sign on top jets.** It is the expected sign once the count is read at fixed IRC-safe substructure. GBDT(EFP+N), trained on the same Top Landscape jets (or on particle-level V0b), also moves tops toward QCD when N rises. ParT, ParticleNet, ParT_kin and PN_kin follow 0.7–1.2 of it, and PFN and LorentzNet 1.3–1.9.
- **The single-knob controls rule out a cancellation artefact.** σ only moves the EFN slightly top-ward (+0.049) and a only moves it QCD-ward (−0.071). The networks move QCD-ward in both cases, by −0.21 to −0.76.
- **QCD jets.** The networks differ in sign. ParT-type networks and PFN follow the conditional count reference (κ_EFP+N 0.7–1.9). ParticleNet and LorentzNet move the other way (κ −0.2 to −1.2).
- **Relation to D22.** On QG, ParT_kin followed 0.34–0.62 of LR(N) (H54v). On top tagging, the count dependence appears with the sign of the conditional (fixed-substructure) count likelihood, not the marginal one.

## Issues and limits

- **Detector mismatch.** These are particle-level jets with no Delphes. The networks see about 50 % more particles than in training (C2ST 0.97). The variant − V0 pairing cancels common offsets but not a dependence of the response on where N sits. A Delphes ATLAS-card rerun is the next step; conda-forge Delphes 3.5.1 is available.
- **Count vs soft-hadron momentum.** As in H54, particle count is not separated from the softer momenta of the extra hadrons. The a-only setting also moves τ32 by +1.9 %.
- **LorentzNet** responses depend on batch padding.
- **Single training seed** per network.
- **Rejection** at ε_S 0.3 / 0.5 is unresolved with 5k QCD jets.
- **GBDT(EFP+N) trained on Top Landscape** reaches only 0.69–0.80 accuracy at particle level, because N is outside its training range. The particle-level (_pl) version is the main reference.
- **Debug attempts.**
  - d-h60-check was OOM-killed by `conda search`. Replaced with an API query.
  - d-h60-score was unschedulable on k8s-chase-ci-01. Rerun with the H54 GPU script that excludes that node.
  - d-h60-score2 failed because the WeaverNet deep-copy could not pickle a module. Fixed by fresh checkpoint loads.

## Files

- Script: /Users/anrunw/Documents/aaron-interp/interp/h60_top_hadronisation.py
- Results in /Users/anrunw/Documents/aaron-interp/results/physics/:
  - h60_run.json: all shifts, CIs, κ, tests, working points, noise, C2ST, accuracies.
  - h60_scores.npz, h60_efp.npz, h60_refs_info.json.
  - h60_gen/*.json: per-worker generator summaries.
- Data: /j-jepa-vol/aaron-interp/data/top/gen_h60/{V0,V0b,Hzp,Hzm,sigup,sigdn,aup,adn}.npz
- Figures:
  - results/physics/h60_shifts.png: paired Δs per model for Hz±, σ-only and a-only, top and QCD jets.
  - results/physics/h60_jet.png: one top jet in V0 and Hz+, same event, N 59 → 74, with b, q, q′ marked. The per-jet mass also changes for this jet (165 → 171 GeV), while the sample-mean Δm is −0.14 GeV.
- Jobs: d-h60-check2, d-h60-gen-*, d-h60-efp, d-h60-score3, d-h60-stats3.

## Explainer entry (for site/explain.py)

- **Question.** Do top taggers react to how many hadrons a jet contains when its IRC-safe substructure (mass, three-prong shape, W mass) is fixed?
- **Physics background.** IRC-safe observables do not change when a particle splits collinearly or a soft particle is added. Hadron multiplicity N is not IRC-safe and depends on the hadronisation model. In the Lund string model, StringZ:aLund and bLund set the longitudinal momentum sharing and StringPT:σ sets the hadron transverse momentum. Changing them changes N at an almost unchanged energy flow.
- **Method.** Same Pythia event, reseeded per event, with only the hadronisation parameters changed (Hz±, σ only, a only). The paired score shift is
  $$\Delta s=\frac1{n}\sum_{j}\big[s_j^{\rm variant}-s_j^{V0}\big],\qquad s=\log\frac{p(\text{top})}{p(\text{QCD})}.$$
  The share of the conditional count response is
  $$\kappa_{\rm EFP+N}=\frac{\Delta s_{\rm net}-\Delta s_{\rm GBDT(EFP)}}{\Delta s_{\rm GBDT(EFP+N)}-\Delta s_{\rm GBDT(EFP)}}.$$
- **Result.** See the figure h60_shifts.png. When N rises by about 9 % at fixed m, τ32 and m_W, all taggers lower the top score (ParT −0.44 log-odds), while the EFN stays at +0.01. The direction is opposite to a count-only classifier and matches a count-at-fixed-substructure classifier.
- **Checks.**
  - Permuted ParT: |Δs| ≤ 0.03.
  - Independent-seed noise band: |Δs| ≤ 0.07.
  - Single-knob controls: the EFN moves in opposite directions, the nets move the same way.
  - Model AUCs reproduce the known values on Top Landscape.
- **Physical significance.** Top taggers trained on Delphes jets read extra soft hadrons at fixed three-prong kinematics as evidence for a radiation-rich QCD jet. Their top efficiency therefore depends on the hadronisation model: ε_S 0.30 → 0.24 for ParT at +9 % N.
- **Glossary terms to add.** EFP (energy-flow polynomial), GBDT(EFP+N), κ_EFP+N, Lund string parameters aLund / bLund / σ (StringPT:sigma), event-paired generator variant, C2ST (classifier two-sample test: AUC of a classifier separating two samples; 0.5 = indistinguishable).
