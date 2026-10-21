# H61: detector-level, MPI-off rerun of the D24 top hadronisation test (Delphes ATLAS card)

Node H61 (loop D, round D13). Claim D24 (0.5): top taggers lower their top score on top jets when the hadron count rises at fixed mass, τ32 and W-pair mass, beyond the IRC-safe EFN. H60 and H60v showed this at particle level, with multi-parton interactions (MPI) on. This node repeats the test on Delphes detector-level jets with MPI off, which is how the Top Landscape training sample was made.

- Script: `interp/h61_top_delphes.py`, plus `interp/h61_install_delphes.sh` (pod-local Delphes install).
- Results: `$AI_RESULTS/physics/h61_run.json`; local copies in `results/physics/h61_run.json`, `h61_shifts.png`, `h61_jet.png`.
- Data: `/j-jepa-vol/aaron-interp/data/top/gen_h61/{V0,V0r,V0b,Hzp,Hzm,sigup,aup,split,merge}.npz`.

## Conclusion

We changed only Pythia's hadronisation settings and passed the events through the Delphes ATLAS detector simulation used for the Top Landscape sample, with MPI off. Top jets then had 6.7 % more reconstructed particles (energy-flow objects) at the same jet mass, τ32 and W-pair mass. The Top Landscape ParT moved its top score by −0.31 log-odds (95 % CI [−0.37, −0.25]) toward QCD, while the IRC-safe EFN did not move (+0.01). That is about 70 % of the particle-level size per jet. Top ParticleNet, LorentzNet, PFN and JetClass ParticleNet_kin behave the same way. JetClass ParT_kin moves only −0.07.

The size of the effect does not come from running at particle level or with MPI on. On their own type of input, the trained top taggers treat extra soft particles at fixed three-prong kinematics as evidence for QCD. Adding particles by an exact collinear split alone reproduces the shift (ParT 0.89 of it). Merging the extra particles away removes it.

Still uncertain:
- Our Delphes jets are not identical to the Top Landscape jets. A classifier separates them with AUC 0.74, so the target of 0.6 was not met: our jets have about 13 % more, softer constituents.
- The response of QCD jets still differs in sign between architectures.

## Question, hypotheses, predictions

**Question.** At detector level and with MPI off (the Top Landscape conditions), does a hadronisation-only rise in particle count on top jets still lower the networks' top score by more than the IRC-safe EFN's?

**H0.** Network minus EFN = 0 under Hz+ (paired shift).

**H1 (D24).** The predictions:
- The Hz+ top-jet shift of ParT, ParticleNet and PFN is negative.
- Each of these networks moves more than 0.1 log-odds further than the EFN.
- The count-only split reproduces most of the shift.
- The merge removes it.

**Alternative (particle-level artefact).** The particle-level effect came from N being outside the training range and from MPI. Then the detector-level shift should lie within the EFN band, as the H60v calorimeter proxy suggested for ParT_kin (+0.007).

## Definitions

- **s:** the top log-odds. For the Top Landscape networks it is log p(top) − log p(QCD); for the JetClass networks it is logit(Tbqq) − logit(QCD). Positive means more top-like.
- **Δs:** mean over event-paired jets of s(variant) − s(V0). The 95 % CI is a bootstrap over jet pairs, B = 2,000.
- **Network − EFN:** the paired difference of the two shifts on the same jets, with a bootstrap CI. Its p-value is the bootstrap-z value, Holm-corrected over 84 tests: 7 networks (ParT, ParticleNet, LorentzNet at batch 32 and at batch 1, PFN, ParT_kin, PN_kin) × 6 comparisons (Hz+, Hz−, σ-only, a-only, split, merge) × 2 classes.
  - Holm: a step-down multiple-test correction that controls the chance of any false positive.
  - Bootstrap-z p-values below about 1e−10 are normal-tail extrapolations. They are quoted as "< 1e−10".
- **EFN:** energy-flow network. Its inputs are energy-weighted angles only, so it is IRC-safe: blind to collinear splits and to soft particles.
- **EFP:** energy-flow polynomials, the IRC-safe substructure basis (54 polynomials, degree ≤ 5).
- **GBDT(EFP):** a gradient-boosted tree on log EFPs. **GBDT(EFP+N):** the same with the constituent count N added. It is the reference for "reads N at fixed IRC-safe substructure".
- **LR(N):** a logistic regression on N alone, the count-only reference. It has slope +0.042 per constituent, so higher N means more top-like.
- All three references were trained on Top Landscape train rows: 25k + 25k random rows (rng 6162), constituents from the stored 128 hardest (pT, Δη, Δφ) in float16, N from the jet record.
- **κ:** Δs_net / Δs_GBDT(EFP+N), the network's shift as a fraction of the conditional count reference's shift.
- **ε_S at the V0 50 % threshold:** the cut is set where 50 % of the paired V0 top jets pass. ε_S is the fraction of variant top jets above the same cut.
- **McNemar:** a test on paired calls. It counts top jets whose call is gained or lost under the variant and gives a binomial p-value.
- **C2ST:** classifier two-sample test. A HistGradientBoosting classifier is trained to separate two jet samples, with a 2-fold AUC. AUC 0.5 means the samples cannot be told apart.
- **N:** number of Delphes energy-flow objects in the jet. **N_pl:** visible stable generator particles within ΔR < 0.8 of the same jet axis.

## Procedure

### 1. Generator

Pythia 8.317 (pythia8mc, venv env-gen):
- pp at 14 TeV, Monash tune (Pythia's default).
- `PartonLevel:MPI = off`. The plan's "MultipartonInteractions:processLevel" is not a Pythia setting; this is the switch that turns MPI off.
- ISR, FSR and hadronisation on.
- Tops: gg/qq̄ → tt̄ with W → quarks. QCD: HardQCD:all. p̂T 450–800 GeV.
- Seeds: before event i of worker w and class c the generator is reseeded with 1 + (2w + c)·20,000 + i. Workers w = 6101–6110; V0b uses 6111–6120.
- Prefilter: events go to Delphes only if they have a particle-level anti-kT 0.8 jet with pT > 450 GeV and |y| < 2.5. 89–93 % of events pass.
- All stable particles are written as HepMC2.

### 2. Detector

Delphes 3.5.1 (conda-forge), the stock `delphes_card_ATLAS.tcl`. The same card family was used for Top Landscape, but in a 2017-era Delphes 3.4 version.
- `set RandomSeed` is fixed per (worker, class, chunk of 1,000 events).
- Installed per job into pod-local /tmp. Linking the conda prefix onto CephFS ran at about 4 packages per minute, and the d-h61-inst prefix on the volume is not used.
- Twice, a Delphes run exited with code 129 before reading any event. The script retries with the same seed, and the retry succeeded.

### 3. Detector-level jets

- Anti-kT R = 0.8 (Pythia SlowJet / fjcore) on EFlowTrack + EFlowPhoton + EFlowNeutralHadron, all massless.
- Kept: 550 < pT < 650 GeV and |η| < 2.
- Top truth-matching: the top and its b, q, q′ all within ΔR < 0.8 of the jet axis, as in Kasieczka et al.
- QCD jets: the nearest outgoing hard parton within 0.8 defines the pairing slot.
- 500 top + 500 QCD jets per worker, so 5,000 + 5,000 per setting.

**Checks.**
- Detector/particle leading-jet pT has median 0.994 (10–90 % range 0.96–1.02).
- In 90 % of V0 top jets each of b, q, q′ has a constituent above 5 GeV within ΔR 0.2.
- A first version mis-indexed constituents (Pythia's `Event.reset()` keeps the system entry) and lost two thirds of the jets. It was fixed before any production run.

### 4. Settings

Values are (StringZ:aLund, StringZ:bLund, StringPT:sigma); Monash is (0.68, 0.98, 0.335 GeV).

| setting | values | role |
|---|---|---|
| V0 | Monash | nominal |
| V0r | Monash; V0 particle events, Delphes RandomSeed shifted | detector-noise null, paired |
| V0b | Monash, independent seeds | generator-noise band, unpaired |
| Hz+ | (0.79, 0.88, 0.247) | N up |
| Hz− | (0.48, 0.98, 0.438) | N down |
| σ-only | (0.68, 0.98, 0.235) | N up |
| a-only | (1.28, 0.98, 0.335) | N up |

Pair key = (worker, class, event, slot). Pairs in both arms of the 550–650 GeV window: 3,491–3,524 top and 4,005–4,038 QCD jets per variant, and 4,241 / 4,059 for V0r.

### 5. Count-only edits on the Delphes jets (7,506 V0/Hz+ pairs)

For each pair, k = max(N_Hz+ − N_V0, 0). k = 0 in 35 % of pairs; mean k is 6.5 for tops and 6.0 for QCD.

- **split:** in the V0 jet, each of the k softest Delphes objects is replaced by two objects of half its pT. The two halves sit ±0.005 in (η, φ) about the parent, in a random direction (rng 6161).
  - N rises by k.
  - The jet four-momentum, mass, τ32, mW and EFPs are unchanged to O(10⁻⁴). Measured: Δm = 0.000 GeV, EFN Δs = −0.000.
- **merge:** in the Hz+ jet, the k nearest-in-ΔR pairs among its softer half are combined (three-momenta added, massless).
  - Merged jets return to the V0 count where k > 0. On average they end 2.3 objects below V0, because jets with N_Hz+ < N_V0 are untouched.
- Because of the max(·, 0), the split adds more objects on average (+6.5) than Hz+ does (+4.2). The split/Hz+ ratios below are therefore also given relative to LR(N)'s ratio. That count-normalised ratio is the share per added object.

### 6. Models and jobs

- Top Landscape: ParT, ParticleNet (H4 checkpoints); LorentzNet (released weights) in batches of 32 with paired jets in the same batch positions, and at batch 1 to show its padding dependence; PFN, EFN (H2).
- JetClass ParT_kin, PN_kin.
- One weight-permuted ParT: every tensor's entries shuffled, seed 0.
- Constituents are massless (E = |p|), as in Top Landscape.

Jobs:

| job | step | resources | wall time |
|---|---|---|---|
| d-h61-check…check5 | checks | | |
| d-h61-g-{v0, v0r, v0b, hzp2, hzm, sigup, aup} | generation | 1 CPU / 3 GiB | 10–19 min |
| d-h61-efp | references | venv-ef, 4 CPU / 8 GiB | 10 min |
| d-h61-score | scoring | GPU | 17 min |
| d-h61-stats | statistics | 1 CPU / 2 GiB | |
| d-h61-figs2 | figures | | |

## Unedited models

AUC / accuracy at s > 0, 5k + 5k jets each. Top Landscape test rows: rng 6061, the same rows as H60.

| model | Top Landscape test | known (H60) | Delphes V0 (this node) |
|---|---|---|---|
| Top ParT | 0.9864 / 94.2 % | 0.9864 / 94.2 % | 0.9882 / 94.4 % |
| Top ParticleNet | 0.9859 / 94.2 % | 0.9859 | 0.9877 / 94.2 % |
| LorentzNet (b32 / b1) | 0.9873 / 0.9871 | 0.9873 | 0.9887 / 0.9886 |
| PFN | 0.9802 / 92.9 % | 0.9802 | 0.9830 / 92.2 % |
| EFN | 0.9761 / 92.1 % | 0.9761 | 0.9796 / 92.8 % |
| JetClass ParT_kin | 0.9826 / 92.8 % | 0.9826 | 0.9879 / 93.2 % |
| JetClass PN_kin | 0.9832 / 93.3 % | 0.9832 | 0.9891 / 94.5 % |
| permuted ParT | 0.723 | 0.72–0.78 | 0.780 |
| LR(N), GBDT(EFP), GBDT(EFP+N) | 0.704 / 0.969 / 0.975 | 0.704 / 0.969 / 0.975 (val-trained) | 0.714 / 0.975 / 0.979 |

Every Top Landscape value reproduces H60. On the Delphes V0 jets, accuracy matches Top Landscape for all networks, unlike particle level, where LorentzNet fell to 89 % and PFN to 82 %.

## Manifold check: C2ST of V0 against Top Landscape test

Features: N, m, τ32, pT^D, the pT fraction of the constituents ranked 1, 2, 3, 5, 10, 20, 40, and the counts with pT < 1, 1–2 and 2–5 GeV.

| | top | QCD |
|---|---|---|
| C2ST AUC, all features | 0.738 | 0.741 |
| without the count features | 0.727 | 0.729 |
| N alone | 0.626 | 0.544 |

For comparison: H60 particle level 0.977 / 0.974; H60v proxy 0.70 / 0.73. **The target AUC ≤ 0.6 is not met.**

Means, Top Landscape → V0:

| observable | top | QCD |
|---|---|---|
| N | 54.9 → 61.8 | 43.5 → 47.7 |
| pT^D | 0.296 → 0.263 | 0.392 → 0.347 |
| leading-constituent pT fraction | 0.205 → 0.173 | 0.301 → 0.257 |
| τ32 | 0.546 → 0.506 | |
| m | 176.0 → 173.7 GeV | 95.5 → 84.3 GeV |

Our Delphes jets fragment more softly than the Top Landscape jets: more, softer constituents. Possible sources, none tested here: Pythia 8.317 against the 2017-era Pythia 8.2, Delphes 3.5.1 against 3.4 card details, and the unknown p̂T generation range.

MPI off lowers the particle-level count from 83.4 (H60) to 74.8 for tops.

## Numbers

### Hz+ on top jets (3,501 pairs)

Observable shifts:
- ΔN = +4.20 [3.87, 4.56] Delphes objects (+6.7 %); particle level +6.11 (+8.1 %).
- Δm = −0.36 [−0.75, 0.02] GeV; Δτ32 = +0.0009; ΔmW = +0.14 GeV; ΔpT^D −2.0 %.
- Per-jet standard deviations, for scale: N about 15, m about 22 GeV.

| model | Δs | net − EFN | Holm p | κ = Δs / Δs_GBDT(EFP+N) | ε_S (V0 50 % cut) | McNemar p (gained / lost) |
|---|---|---|---|---|---|---|
| Top ParT | −0.313 [−0.372, −0.253] | −0.327 [−0.386, −0.268] | 1e−24 | 0.73 [0.60, 0.86] | 0.500 → 0.459 | 6e−8 (274 / 417) |
| Top ParticleNet | −0.308 [−0.375, −0.241] | −0.322 | 5e−20 | 0.72 [0.58, 0.87] | 0.459 | 5e−8 |
| LorentzNet b32 | −0.415 [−0.489, −0.340] | −0.429 | 4e−29 | 0.96 | 0.445 | 1e−12 |
| LorentzNet b1 | −0.640 [−0.771, −0.514] | −0.654 | 1e−22 | 1.49 | 0.446 | 7e−13 |
| PFN | −0.334 [−0.376, −0.295] | −0.348 | < 1e−10 | 0.78 [0.72, 0.84] | 0.436 | 3e−19 |
| JetClass ParT_kin | −0.073 [−0.120, −0.028] | −0.087 [−0.136, −0.041] | 0.004 | 0.17 [0.07, 0.28] | 0.484 | 0.03 |
| JetClass PN_kin | −0.201 [−0.252, −0.151] | −0.215 | 5e−14 | 0.47 | 0.462 | 4e−7 |
| permuted ParT | +0.030 [+0.023, +0.037] | +0.016 | – | – | 0.534 | – |
| EFN | +0.014 [−0.015, +0.043] | – | – | – | 0.501 | 0.92 |
| GBDT(EFP) | +0.024 [−0.004, +0.051] | – | – | – | 0.508 | – |
| LR(N) | +0.177 [+0.163, +0.192] | – | – | – | 0.594 | – |
| GBDT(EFP+N) | −0.430 [−0.472, −0.389] | – | – | – | 0.395 | 3e−37 |

Particle level (H60, MPI on) → detector level (H61, MPI off), Hz+ top Δs:

| model | H60 | H61 |
|---|---|---|
| ParT | −0.435 | −0.313 |
| ParticleNet | −0.417 | −0.308 |
| LorentzNet | −0.818 | −0.415 |
| PFN | −0.579 | −0.334 |
| ParT_kin | −0.355 | −0.073 |
| PN_kin | −0.345 | −0.201 |

ParT per added object: −0.060 (particle level) and −0.075 (Delphes). The H60v detector proxy gave ParT −0.120 and ParT_kin +0.007; Delphes gives a larger ParT shift than that proxy.

### Other hadronisation settings, top jets

| setting | ΔN | EFN | ParT | ParticleNet | PFN | ParT_kin | PN_kin | GBDT(EFP+N) | LR(N) |
|---|---|---|---|---|---|---|---|---|---|
| Hz− | −4.87 (−7.8 %) | −0.034 | +0.327 [0.268, 0.382] | +0.396 | +0.434 | +0.128 | +0.290 | +0.505 | −0.205 |
| σ-only | +2.98 (+4.8 %) | +0.050 | −0.110 | −0.091 | −0.261 | +0.023 (net − EFN Holm 1.0) | −0.032 | −0.333 | – |
| a-only | +3.58 (+5.8 %); Δτ32 +0.0097 | −0.083 | −0.431 | −0.525 | −0.350 | −0.234 | −0.395 | −0.325 | – |

In the a-only setting part of the shift may come from the τ32 change, as the EFN's −0.083 shows.

### Count-only edits, top jets

**Split** (ΔN +6.54, +10.5 %; Δm 0.000; EFN −0.000):

| model | Δs | split / Hz+ | count-normalised (÷ LR(N)'s 1.56) |
|---|---|---|---|
| ParT | −0.278 [−0.294, −0.263] | 0.89 [0.75, 1.08] | 0.57 |
| ParticleNet | −0.342 | 1.11 [0.93, 1.39] | 0.71 |
| PFN | −0.350 | 1.05 [0.95, 1.16] | 0.67 |
| LorentzNet | −0.789 | 1.9 | 1.22 |
| ParT_kin | −0.296 | 4.1 [2.5, 11.4] | 2.6 |
| PN_kin | −0.530 | 2.6 | 1.7 |
| permuted ParT | +0.062 | – | – |
| GBDT(EFP+N) | −0.665 | 1.54 | 0.99 |

All network − EFN tests under the split have Holm p < 1e−10.

**Merge** (Hz+ jets brought back to the V0 count; ΔN vs V0 −2.34; Δm −0.43 GeV):

| model | Δs vs V0 | share of the Hz+ shift removed | count-normalised (÷ LR(N)'s 1.56) |
|---|---|---|---|
| ParT | +0.123 [0.067, 0.177] | 1.39 [1.18, 1.69] | 0.89 |
| ParticleNet | +0.140 | 1.46 | 0.94 |
| PFN | +0.088 | 1.26 | 0.81 |
| LorentzNet | +0.184 | 1.44 | – |
| ParT_kin | +0.056 | 1.8 [1.1, 4.8] | – |
| PN_kin | −0.070 | 0.65 | – |
| GBDT(EFP+N) | +0.229 | – | – |

The split result means ParT_kin does respond to the count on Delphes jets (−0.30 under the split). Its Hz+ shift is small because the hadronisation change also moves something it reads in the opposite direction: pT^D rises −2 % and the constituent spectrum shifts.

### QCD jets, Hz+ (4,005 pairs)

ΔN +4.29 (+8.9 %), Δm −0.30 GeV.
- IRC-safe and count references: EFN −0.007, GBDT(EFP+N) −0.158, LR(N) +0.181.
- QCD-ward (toward lower top score): ParT −0.160 [−0.193, −0.129], PFN −0.210, ParT_kin −0.057 (Holm 0.24).
- Top-ward: ParticleNet +0.170, LorentzNet +0.257, PN_kin +0.100.

Under the split, ParT gives −0.035 (split/Hz+ 0.22) and PFN −0.040 (0.19). For these two networks the QCD-jet Hz+ shift is mostly not a count response, as in H60v.

### Nulls and noise

- **Detector-noise null V0r** (same particles, Delphes reseeded, paired): top-jet |Δs| ≤ 0.024 for every network (ParT −0.016 [−0.042, 0.009]), and every net − EFN difference on tops has unadjusted p > 0.1. On QCD jets LorentzNet gives −0.058 and LorentzNet b1 −0.159 [−0.329, 0.009].
- **Generator-noise band V0b − V0** (independent events, unpaired): all networks |Δs| ≤ 0.064, with CIs within [−0.26, 0.25]; LorentzNet b1 QCD [−0.81, 0.95].
- **Permuted ParT:** +0.030 (Hz+ top), +0.062 (split top).

**LorentzNet padding dependence.** Its readout averages node features over the padded length of the batch, so its response to N depends on how the jet is batched:

| | batch 32 | batch 1 |
|---|---|---|
| Hz+ top | −0.415 | −0.640 |
| Hz+ QCD | +0.257 | +1.333 |
| split QCD | +0.555 | +2.625 |

The batch-1 response is 1.5–5× the batch-32 response.

**Holm.** 73 of 84 network − EFN tests have Holm p < 0.05. The 11 that do not:
- ParT_kin: QCD under Hz+, Hz−, σ-only, a-only; top under σ-only; top under merge.
- PN_kin: QCD under Hz− and a-only.
- QCD under merge: ParT, ParticleNet, PFN.

## Reading

**Which hypothesis survives.**
- **H0 (net − EFN = 0 under Hz+) is rejected** on top jets for all seven network entries (Holm p ≤ 0.004).
- **D24 (H1) predictions** for ParT, ParticleNet and PFN, all met on Delphes, MPI-off jets:
  - The Hz+ top shift is negative.
  - Net − EFN is −0.32 to −0.35, beyond 0.1.
  - The split reproduces 0.89–1.11 of the shift (0.57–0.71 per added object).
  - The merge removes 1.26–1.46 of it (0.81–0.94 per removed object).
- **The particle-level-artefact alternative** predicted shifts within the EFN band at detector level. It is excluded for ParT, ParticleNet, LorentzNet, PFN and PN_kin. For JetClass ParT_kin the Hz+ shift is small (−0.073), but not zero (Holm 0.004), and its split response is large.

**What changes from H60.** The size shrinks by 0.55–0.72× for the Top Landscape networks and by 0.2–0.6× for the JetClass networks. The working-point effect at the V0 50 % cut is ε_S 0.50 → 0.46 for ParT, ParticleNet and PFN.

**Scope.**
- The C2ST of 0.74 means our Delphes jets are not on the Top Landscape manifold: N +13 %, softer constituents. The measured sizes may differ on the exact training conditions.
- The pairing is exact at parton level. The Delphes smearing is not paired, which V0r shows contributes ≤ 0.024 on tops.
- Hz+ also shifts pT^D by −2 %. The count-only edits show the count alone is sufficient for the top-jet shift. They do not show that pT^D plays no role.

## Files

- `interp/h61_top_delphes.py` (steps check, gen, edits, efp, score, stats, figs)
- `interp/h61_install_delphes.sh`
- `results/physics/h61_run.json`
- `results/physics/h61_shifts.png`: paired shifts per model, top and QCD, with 95 % CIs, for Hz+, Hz−, split, merge and V0r.
- `results/physics/h61_jet.png`: one Delphes top jet in V0 (41 objects, m 179 GeV, ParT +7.93) and Hz+ (60 objects, m 176 GeV, ParT +6.72), with tracks, photons and neutral hadrons coloured and the b, q, q′ partons marked.
- `/j-jepa-vol/aaron-interp/env-delphes`: an unused, partly linked conda prefix from d-h61-inst, 1.5 GB. It can be deleted.
