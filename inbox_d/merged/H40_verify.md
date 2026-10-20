# H40 verify: quark–gluon particle-ID taggers, jet charge and neutral-species relabels

- **Verifier job:** `anrunw-ai-d-h40-verify` (GPU, V100, 25 min, exit 0). The first two submissions were placed on `k8s-haosu-09.sdsc.optiputer.net` and failed with `UnexpectedAdmissionError` (GPU admission) before writing a log. The third ran from a scratch copy of `part_job.sh` with that node added to the exclusion list. `part_job.sh` itself is unchanged; the loop owner may want to add haosu-09 to its list.
- **Script:** `interp/h40v_verify.py`. It imports `make_edits`, `train_refs`, `statistics` and `all_feats` from `interp/h40_qg_pid_charge.py` unchanged.
- **Results:** `results/physics/h40_verify.json` and `h40_verify_perjet.npz`. They are also on the volume under `/j-jepa-vol/aaron-interp/results/physics/`.

## Conclusion
The jet-charge part of H40 holds up on a fresh sample. Replacing every particle by its antiparticle, with momenta unchanged, moves the quark score of the particle-ID ParticleTransformer and ParticleNet by about 0.24 log-odds against the sign of the jet charge. That is about 85 % of what a logistic fit on jet charge alone does. Each of 12 networks with randomly permuted weights moves by at most 0.11 log-odds. The neutral-species part reproduces only in direction: soft pions relabelled as photons push toward "quark", and relabelled as neutral hadrons toward "gluon". Random-weight networks move by similar amounts (up to 0.30 log-odds) with random signs, so the evidence that this species use was learned rests on the shared direction, not on the size. H40 reported that ParticleNet does not respond to flipping the hardest charged particle. That came from averaging over both charge signs. Signed by the leader's charge, ParticleNet responds as strongly as ParT.

## What was checked in the code (`interp/h40_qg_pid_charge.py`)
- **Particle order.** `data_top.convert_files` pT-sorts momenta and PDG ids with the same permutation, and the Herwig loader sorts too. "Hardest charged" (E2) and the leading charge used for the split are therefore correct. `h5.from_list` re-sorts edited lists stably, and the pid travels with its row.
- **Padding.** Padding is excluded everywhere: `pt > 0` masks are used, and `to_list` keeps only the first n particles, which are the real ones after sorting.
- **Edits change only ids.** The (pT, η, φ) assertion holds for every edit. The weaver inputs (`h5.weaver_inputs`) use massless four-vectors. Charge is read from ±211/321/2212 and ∓11/13, isChargedHadron from |pid| ∈ {211, 321, 2212}, and isNeutralHadron from {130, 2112}. The E4 zeros for kinpid are therefore by construction, as stated. For LorentzNet, relabelling changes the PDG mass, so E changes slightly. "Every momentum held fixed" is therefore exact only for the kinpid networks.
- **Score sign.** For weaver (labels [isQ, isG]) and LorentzNet (label 1 = quark), the sign of s = log p(q) − log p(g) is confirmed by the AUCs.
- **Working point.** The threshold sits at quark efficiency 0.5 on the unedited jets, which is the R50 definition.
- **Sampling.** Random rows per class, never first-k rows.
- **Minor: references.** `train_refs` builds the reference features from `kin3` (float16, at most 128 particles), but they are scored on the float32 full arrays (up to 180 particles). The references are therefore fit and evaluated on slightly different feature definitions. The effect is small because the maximum multiplicity is 148.
- **Design issue: E2 contrast.** `E2k-C2` is averaged over jets without the sign of the flipped leader. A symmetric, sign-following response therefore averages to about zero. This is why H40 reports PN at +0.002 and why Holm does not reject it. The H40 class × sign split already shows −0.51 / +0.54. The figure's E2 rows carry the same issue.
- **Holm.** The sign-flip p has a floor of 5×10⁻⁴. With 165 tests in the verify run, no test is rejected under Holm on the permutation p. This is the same floor artefact that H40's `pz` step works around. All bootstrap CIs below exclude 0 by more than 30 standard errors.

## Procedure (verify run)
- **Sample:** 10,000 quark and 10,000 gluon jets, drawn at random (seed 4777) from the KMT Pythia test split. Rows in the H40 sample (`data_qg.sample(10000, 4001)`) are excluded, and the code asserts this. N = 20,000.
- **Edits:** `make_edits(full=False)`: E1, E2k1, C2, E3g_4, E3n_4, C3_4, E3r_4, C3r_4, E4a, E4b, E4n, C4.
- **Models:** ParT kinpid and PN kinpid (H2, best epoch 17 of 20), and LorentzNet (released).
- **New null ensemble:** 6 weight-permuted copies of ParT kinpid and 6 of PN kinpid (torch seeds 1–6). Each copy permutes every parameter tensor's entries. H40 used a single permuted ParT with seed 0.
- **References:** re-trained on 50,000 + 50,000 non-validation train rows, seed 4777.
- **Added contrast:** E2k1_signlead = (Δs(E2k1) − Δs(C2)) × sign(charge of the hardest charged particle).
- **Statistics:** mean with a 95 % bootstrap CI over jets (B = 1000); sign-flip p; per-unit slopes. For the ensemble comparison, a rank p = (1 + #{permuted nets with |mean| ≥ |trained mean|}) / (K + 1), with K = 6 per architecture or 12 pooled. The minimum possible p is 0.14 per architecture and 0.077 pooled.

## Unedited models on the verify sample (AUC)
| model | AUC | H40 sample |
|---|---|---|
| ParT kinpid | 0.9105 | 0.9104 |
| PN kinpid | 0.9053 | 0.9058 |
| LorentzNet | 0.9152 | 0.9157 |

- ParT permuted s1–s6: 0.48–0.68.
- PN permuted s1–s6: 0.33–0.61.
- LR(Q₀.₃): 0.552. GB_kin: 0.871. GB_sp: 0.880. GB_spQ: 0.880.

## Results: H40 vs. verify
| contrast (Pythia) | model | H40 | verify (95 % CI) | permuted nets, ParT / PN: range of means |
|---|---|---|---|---|
| E1 × sign Q₀.₃ | ParT kinpid | −0.246 | −0.249 [−0.255, −0.243] | −0.019 … +0.007 |
| | PN kinpid | −0.239 | −0.233 [−0.238, −0.229] | −0.032 … +0.112 |
| | LR(Q₀.₃) | −0.308 | −0.276 [−0.280, −0.274] | |
| E2k1 − C2, signed by leader | ParT kinpid | (unsigned +0.171) | −0.405 [−0.414, −0.396] | −0.021 … +0.006 |
| | PN kinpid | (unsigned +0.002) | −0.411 [−0.415, −0.406] | −0.025 … +0.147 |
| | LR(Q₀.₃) | | −0.389 | |
| E3g_4 − C3 (π± → γ) | ParT / PN / LorentzNet | +0.154 / +0.147 / +0.090 | +0.152 / +0.144 / +0.089 (CI ±0.004) | ParT −0.084 … +0.086; PN −0.298 … +0.037 |
| E3n_4 − C3 (π± → K⁰_L) | ParT / PN / LorentzNet | −0.238 / −0.262 / −0.267 | −0.238 / −0.257 / −0.265 (CI ±0.006) | ParT −0.159 … +0.082; PN −0.280 … +0.022 |
| E3r_4 − C3r (γ → π± vs γ → K⁰_L) | ParT / PN | +0.210 / +0.212 | +0.211 / +0.214 | ParT −0.076 … +0.191; PN −0.020 … +0.276 |
| γ → K⁰_L (m = 4) | ParT / PN / LorentzNet | −0.322 / −0.324 / −0.322 | −0.320 / −0.325 / −0.318 | ParT −0.189 … +0.144; PN −0.028 … +0.071 |
| E4a / E4b / E4n | LorentzNet | +0.069 / −0.306 / −0.390 | +0.074 / −0.307 / −0.372 | (kinpid and permuted: 0 by construction) |

**Per-unit slopes, Δs / ΔQ₀.₃.**

| model | E1 | E2 (k = 1) |
|---|---|---|
| ParT kinpid | 0.293 | 0.383 |
| PN kinpid | 0.280 | 0.382 |
| LR(Q₀.₃) (re-trained) | 0.338 | 0.338 |
| LR(Q₀.₅) | 0.338 | 0.378 |

- H40 quoted an LR(Q₀.₃) coefficient of 0.370; the re-trained LR gives 0.338.
- Permuted networks: |slope| ≤ 0.024, except PN s3 at −0.135.

**E1, class × sign split for ParT (verify).**
- Quark jets: Q > 0 −0.372, Q < 0 +0.254.
- Gluon jets: Q > 0 −0.177, Q < 0 +0.173.

The network applies the charge prior to gluon jets as well, as expected for a learned prior on Q.

**Calls changed at R50.**
- ParT kinpid: E1 5.0 %; E3 2.4–5.4 %.
- PN kinpid: E1 4.2 %; E3 2.1–6.2 %.
- Over all H40 contrasts, the maximum is 7.5 % (E5sh). H40's reading says 2–9 %; the measured range is 2–7.5 %.

## Hypotheses and tests
**E1/E2: are the networks using the signed jet charge?**
- H0: the response to charge relabels is no larger than that of a network with the same architecture and inputs but no training signal. Its sign matches the data's Q–label correlation only by chance.
- H1: the response follows sign(Q), with a slope close to the Bayes-like LR(Q) coefficient.
- Results: the trained |mean| exceeds all 12 permuted networks for E1 and for E2-signed (pooled rank p = 1/13 = 0.077, the minimum attainable with K = 12). The trained slopes are 0.83–1.13 × the LR slope, while permuted slopes are ≤ 0.14 in absolute value.
- Direction alone does not separate the trained networks from the null: 9 of 12 permuted networks also have the same E1 sign.
- In score-SD units the permuted networks move more than the trained ones (|mean|/SD up to 0.32 vs 0.10). Their scores are nearly constant (SD 0.09–0.35 vs 2.3–2.4), so the SD scale does not measure use here. The log-odds scale is the one that sets calls.
- H0 is excluded in log-odds size and in slope. A shuffled-label or second-seed kinpid is still missing.

**E3: do the networks use the photon versus neutral-hadron label of soft particles?**
- H0: random-weight networks show the same size and sign pattern.
- H1: the sign pattern (π→γ quark-ward, π→K⁰_L gluon-ward, γ→K⁰_L gluon-ward) follows the data, matching GB_sp, with sizes outside the null.
- Size: trained |mean| is outside the permuted range only for ParT E3g, E3n and γ→K⁰_L. For PN, permuted network s3 moves by 0.28–0.30 on E3g, E3n and E3r, which is at or above the trained values. Size is not distinctive (pooled rank p for PN E3g = 2/13 = 0.15).
- Pattern: the full sign pattern (E3g > 0, E3n < 0, γ→K⁰_L < 0) appears in 2 of 12 permuted networks. It appears in all 3 trained networks, including LorentzNet, which has different inputs, and in GB_sp and GB_spQ.
- Within a single network, the paired difference E3g − E3n at fixed Δ(n_ch/N) excludes a charged-fraction-only reading (H40's point). This is reproduced: +0.15 vs −0.24 for ParT.
- H0 is excluded for direction (pattern), not for size.

## Corrections to the H40 reading
1. "In log-odds and in direction, the trained networks are outside the permuted-network band" holds for E1/E2 in log-odds. It does not hold for direction alone (most permuted networks share the E1 sign), and it does not hold for E3 in log-odds (PN permuted s3 reaches 0.30).
2. The E2 statement "PN +0.002 (symmetric)" should read: signed by the leader's charge, PN −0.41 [−0.415, −0.406], equal to ParT (−0.405) and to LR(Q₀.₃) (−0.389). The ParT asymmetry (−0.36 / +0.70) is the part that differs between the two networks, and GB_spQ shows a similar asymmetry. The figure's E2 rows should use the signed contrast.
3. "Changes 2–9 % of calls" should read 2–7.5 %.
4. The "about as much as a fit on jet charge alone" comparison is 0.29 vs 0.34 per unit Q₀.₃ on the verify references (about 85 %), not 0.29 vs 0.37.
5. "With every momentum held fixed" is exact for the kinpid networks only. For LorentzNet, E changes through the PDG mass.
6. The u-quark attribution is an interpretation, consistent with mean Q₀.₃ = +0.096 for quark jets vs +0.017 for gluon jets. No edit or sample split tests it.

## Corrected reading
- **Jet charge.** With all particle momenta fixed, charge conjugation moves the quark score of the Pythia-trained ParT and PN kinpid by −0.249 and −0.233 log-odds × sign(Q₀.₃). Flipping the hardest charged particle moves it by −0.41 × sign(leader charge) for both networks. The per-unit-Q₀.₃ slopes are 0.28–0.38, compared with 0.34–0.38 for logistic fits on Q₀.₃ or Q₀.₅. All 12 weight-permuted networks move by ≤ 0.15 log-odds (ParT copies ≤ 0.02) and have slopes ≤ 0.14 in absolute value.
- **Neutral species.** Relabelling 4 soft π± as photons, as K⁰_L, or 4 soft photons as K⁰_L gives the sign pattern quark-ward / gluon-ward / gluon-ward in ParT, PN and LorentzNet, as in a GBDT on species counts. The pattern occurs in 2 of 12 permuted networks. The sizes (0.14–0.32 log-odds) are within the range of the permuted networks for PN.
- **Kaon/proton.** K/p ↔ π is invisible to the kinpid inputs. LorentzNet responds (+0.07, −0.31, −0.37).
- **Herwig.** The Herwig arm and the generator-gap estimate were not re-run.

## Remaining open
- A shuffled-label or second-seed kinpid training: the null that separates "learned from labels" from "architecture plus input encoding". Nothing on the volume provides it; `models/qg` has `efn_shuf` / `pfn_shuf`, which are PID-blind.
- A C2ST of the E3 relabels: at m = 4, the edit raises the K⁰_L count from about 1 to about 5 per jet, far outside the natural distribution.
- A LorentzNet null.
