# H60 verify: hadronisation-only Pythia variants on top and QCD jets

Skeptic pass on H60 (loop D). Script: interp/h60_verify.py. It imports the H60 generator and the H19/H4 model code unchanged.
Jobs:
- d-h60-verify-gv0, -ghzp, -ghzm: generation, 1 CPU / 2 GiB each, 5–8 min.
- d-h60-verify: GPU scoring and statistics, 6 min, exit 0.

Result: $AI_RESULTS/physics/h60_verify.json, with a local copy at results/physics/h60_verify.json. The fresh generator data are in /j-jepa-vol/aaron-interp/data/top/gen_h60_verify/{V0,Hzp,Hzm}.npz.

## Conclusion

On a fresh sample with new seeds, the H60 particle-level numbers come back within their 95 % intervals. When only the hadronisation settings add about 9 % more particles to a top jet, the Top Landscape ParT moves its top score by −0.45 log-odds toward QCD, and the IRC-safe EFN does not move. New input edits show that the extra particle count causes this shift on top jets:
- Splitting soft particles into collinear pairs changes the count and nothing IRC-safe. It moves the networks the same way.
- Merging the extra particles back into their nearest neighbours removes the shift.

When the jets are passed through a crude calorimeter-like proxy that brings the particle count into the range the networks were trained on, three things change: most of the added particles disappear, the shifts shrink to about a quarter, and JetClass ParT_kin no longer responds. So H60's efficiency statement (ParT signal efficiency 0.30 → 0.24) holds only for particle-level inputs. How large the effect is on detector-level jets is still open, and a real Delphes run is needed to settle it. On QCD jets the response is not explained by the count.

## What was checked

### 1. Code review of interp/h60_top_hadronisation.py

No bugs that change the numbers were found. Points checked:

- **Pairing.** The generator is reseeded per event with `p.rndm.init(seed(w, i))`. Variant settings use the same worker ids as V0, so they get the same seeds. The pair key is worker·10⁷ + event·10 + slot.
  - In the fresh sample, the stored truth (η, φ) of the t, b, q, q′ (tops) or of the hard parton (QCD) is bit-identical in 100 % of pairs: 6,894 pairs for Hz+ and 6,851 for Hz−.
  - Pairs are kept only when both settings give a jet in 550–650 GeV, so pairs make up about 77 % of the top jets. This selection is symmetric between the arms.
- **Padding.** `massless_P4` zeroes padded rows, and n counts real particles only (capped at 200).
  - The Top nets compute the jet axis from all n particles and read the first 128 (`jet_inputs`, `weaver_inputs`).
  - The EFN and PFN read the first 128.
  - The observables (`H._obs`) are computed on the same massless P4 the networks see.
- **Rows.** The Top Landscape rows are random (rng 6061 / 4194), not first-k. The generator samples are not class-ordered in a way that matters, because pairs are matched by key.
- **Working point.** The threshold is set at ε_S (top-jet efficiency) = 0.3 / 0.5 on the paired V0 tops and applied to the variant tops. This is correct.
- **Minor points.**
  - The setup sentence "mass, τ32, mW change by at most 0.2 %" holds for Hz± tops. Hz− QCD τ32 moves −0.56 %, and the a-only settings move τ32 by about 2 %.
  - Holm in H60 uses bootstrap-z p-values, which is fine.
  - Per the Top Tagging reference dataset description, the Top Landscape jets were generated without multiple interactions. H60 has MPI on, so H60 has one more source of extra soft particles than the training data. This adds to the particle-level mismatch and is not mentioned in H60.

### 2. Fresh-seed rerun of the key comparison

**Sample.** Seed base 7272 (H60: 6060). Settings V0 (Monash), Hz+ and Hz−, with 4,000 top + 4,000 QCD jets each. The generator, cuts and truth matching are those of H60.

**Unedited accuracy / AUC on fresh V0, particle level:**

| model | accuracy / AUC |
|---|---|
| Top ParT | 93.9 % / 0.990 |
| ParticleNet | 93.8 % / 0.988 |
| LorentzNet | 89.8 % / 0.984 |
| PFN | 82.3 % / 0.981 |
| EFN | 93.6 % / 0.983 |
| JetClass ParT_kin | 92.3 % / 0.978 |
| JetClass PN_kin | 92.4 % / 0.988 |
| permuted ParT | 69.1 % / 0.774 |

**Paired shift.** Δs is the mean over event-paired jets of s(Hz+) − s(V0). s is the top log-odds; positive means toward top. Intervals are 95 % bootstraps over pairs, B = 2000. Top jets, 3,097 pairs, ΔN = +7.58 (+9.0 %), Δm = +0.03 GeV [−0.29, 0.33]:

| model | H60 | fresh |
|---|---|---|
| Top ParT | −0.435 | −0.454 [−0.516, −0.394] |
| ParticleNet | −0.417 | −0.423 [−0.492, −0.358] |
| LorentzNet | −0.818 | −0.773 |
| PFN | −0.579 | −0.593 |
| ParT_kin | −0.355 | −0.353 |
| PN_kin | −0.345 | −0.352 |
| EFN | +0.012 | +0.003 [−0.019, 0.025] |
| permuted ParT | ≤ 0.027 | +0.028 |
| LR(N) | +0.316 | +0.331 |

**Hz− on top jets** (ΔN −9.1 %): ParT +0.502, PN +0.475, LorentzNet +0.849, PFN +0.635, ParT_kin +0.409, PN_kin +0.424, EFN −0.022.

**QCD jets, Hz+.** The H60 sign pattern reproduces: ParT −0.104, ParT_kin −0.128, PFN −0.322, ParticleNet +0.231, LorentzNet +0.057, PN_kin +0.052.

**Statistics.** Across the 210 tests (nets and null; mean = 0 and net − EFN = 0; all comparisons below), Holm with bootstrap-z p-values leaves 167 tests with Holm p < 0.05. The sign-flip permutation p (2,000 flips) has a floor of 5·10⁻⁴, so under Holm over 210 tests it cannot reach 0.05. For that reason the bootstrap-z Holm is used.

### 3. New controls

#### a. Detector proxy: is the shift still there when N is in the training range?

**Edit.** The same rule is applied to every setting:
- Charged particles with pT > 0.5 GeV are kept one by one.
- All other particles (photons, neutral hadrons, and charged particles below 0.5 GeV) are summed into 0.1 × 0.1 (η, φ) cells.
- A cell is kept as one massless particle if its pT exceeds 1 GeV.

This is not Delphes. Its purpose is to remove the main training-range mismatch.

**Effect on the sample.**
- Mean N in V0 becomes 49.0 (top) and 38.1 (QCD). Top Landscape has 54.9 and 43.5.
- The C2ST AUC against Top Landscape test jets on (N, m, m_SD, τ21, τ32, mW) falls from 0.979 / 0.972 to 0.697 / 0.728. C2ST is a classifier two-sample test: a classifier tries to separate the two samples, and AUC 0.5 means it cannot.
- Unedited accuracy rises toward the Top Landscape values: ParT 95.3 %, PN 95.2 %, LorentzNet 95.4 %, PFN 94.5 %.

**Top jets, Hz+ (3,097 pairs).** ΔN is only +3.05 (+6.2 %), because most of the extra hadrons are soft neutrals that fall into shared cells or below threshold. Δm = −0.51 GeV [−0.84, −0.21]. Shifts, with the net − EFN difference and its Holm p:

| model | Δs | net − EFN | Holm p |
|---|---|---|---|
| ParT | −0.120 [−0.172, −0.068] | −0.159 | 4·10⁻⁹ |
| ParticleNet | −0.181 [−0.250, −0.114] | – | – |
| LorentzNet | −0.280 | – | – |
| PFN | −0.260 | – | – |
| PN_kin | −0.103 | – | – |
| ParT_kin | +0.007 [−0.039, 0.057] | −0.033 | 1.0 |
| EFN | +0.040 [0.016, 0.062] | – | – |
| LR(N) | +0.133 | – | – |

**Top jets, Hz− under the proxy:**

| model | Δs | Holm p |
|---|---|---|
| ParT | +0.104 | – |
| ParticleNet | +0.085 | 0.52 |
| LorentzNet | +0.367 | – |
| PFN | +0.270 | – |
| PN_kin | +0.073 | 0.23 |
| ParT_kin | −0.083 | 0.038 (opposite in sign to its particle-level Hz− shift) |
| EFN | −0.059 | – |

**Reading.** Under the proxy, Top Landscape ParT keeps the QCD-ward sign, at about a quarter of its particle-level size. The JetClass ParT_kin response is absent (Hz+) or reversed (Hz−). Per added particle, ParT gives −0.039 under the proxy and −0.060 at particle level.

#### b. Exact collinear split: count only, at the Hz+ count change

**Edit.** In each V0 top jet, k_j = max(N_Hz+,j − N_V0,j, 0) distinct random particles of the softer half are each replaced by two pT/2 halves. The halves sit ±0.005 in (η, φ) about the parent, in a random direction (the H18v edit). This keeps every IRC-safe quantity fixed, so Δm = 0.000 and the EFN gives +0.001. Mean k = 9.7 per top jet.

**Top-jet results:**

| model | Δs | share of the same jets' Hz+ shift |
|---|---|---|
| ParT | −0.382 [−0.399, −0.364] | 0.84 [0.76, 0.95] |
| ParticleNet | −0.316 | 0.75 [0.65, 0.88] |
| PFN | −0.482 | 0.81 [0.76, 0.87] |
| LorentzNet | −1.469 | 1.9 |
| ParT_kin | −0.688 | 1.95 |
| PN_kin | −0.953 | 2.7 |
| permuted ParT | +0.043 | – |
| LR(N) | +0.424 | – |

Per added particle, ParT gives −0.039 under the split, against −0.060 for Hz+.

#### c. Merge: Hz+ jets returned to the V0 count

**Edit.** In each Hz+ top jet, the k_j nearest-in-ΔR pairs of its softer half are merged by adding their three-momenta, and the result is made massless. Δm = −0.06 GeV.

**Fraction of the Hz+ shift removed** (1 = all of it):

| model | fraction removed |
|---|---|
| ParT | 1.19 [1.07, 1.35] |
| ParticleNet | 1.08 [0.92, 1.27] |
| PFN | 0.74 [0.69, 0.79] |
| LorentzNet | 1.69 |
| ParT_kin | 1.85 |
| PN_kin | 1.22 |

Fractions above 1 occur partly because merged jets end 2.3 particles below V0 on average: jets where Hz+ had fewer particles were not adjusted.

#### d. QCD jets under the same edits

The split moves:
- ParT +0.005 [−0.002, 0.011]
- ParticleNet +0.091
- ParT_kin +0.189
- PN_kin −0.526
- LorentzNet −0.374

Merging Hz+ QCD jets leaves ParticleNet at +0.251 relative to V0, compared with +0.231 for Hz+. The QCD-jet Hz+ shifts therefore do not follow the count edits in sign or size.

### 4. Hypotheses and what is excluded (top jets)

- **H0, IRC-safe only** (the net moves like the EFN). Excluded at particle level and, for the Top Landscape nets, under the detector proxy (Holm p ≤ 4·10⁻⁷ for ParT, PN, LorentzNet, PFN, PN_kin). Not excluded for JetClass ParT_kin under the proxy.
- **H1, count read with the LR(N) sign** (more particles means more top-like). Excluded: every network moves opposite to LR(N) under the split, Hz+ and the proxy.
- **H1′, count at fixed IRC-safe content read QCD-ward.** Supported causally at particle level:
  - The split reproduces 0.75–0.84 of the ParT, PN and PFN shift at a larger count change.
  - The merge removes 0.7–1.2 of it.
  
  The JetClass nets and LorentzNet overshoot under the split. The split's collinear duplicates are probably off-manifold for them.
- **H2, the response is a training-range artefact of particle-level N.** Partly supported: under the proxy the shift shrinks to about 0.25 of particle level for ParT, and to 0 for ParT_kin.

### 5. Figures

- h60_shifts.png: the plotted points match h60_run.json, the axes and labels are correct, and the 95 % CIs are shown.
- h60_jet.png: same event and same truth partons, N 59 → 74. The title states the per-jet mass change (165 → 171 GeV), which the report also notes.

Both figures show particle-level jets only; no figure was made for the proxy result.

## Corrected reading

At particle level (fresh seed, reproduced), all six networks move top jets QCD-ward when hadronisation alone adds about 9 % particles, while the EFN stays at zero. On top jets, the count edits at fixed IRC-safe content (collinear split, nearest-pair merge) reproduce or remove most of this shift for ParT, ParticleNet and PFN, so the shift is caused by the extra particle count. The magnitude and the efficiency change (ParT ε_S 0.30 → 0.24) are particle-level numbers.

Under a calorimeter-like proxy that brings N into the training range (C2ST 0.70 instead of 0.98):
- the count change shrinks from +9 % to +6 %;
- the Top ParT shift shrinks to −0.12 log-odds;
- JetClass ParT_kin shows no response.

On QCD jets the response differs by architecture and does not follow the count edits.

## Corrected Conclusion (for the node)

When only Pythia's hadronisation settings change so that top jets carry about 9 % more particles at the same mass and three-prong structure, all six top taggers tested call top jets less top-like (ParT −0.45 log-odds), while an IRC-safe network does not move. Edits that change only the particle count reproduce this on top jets, and edits that undo it remove it. So these networks treat extra soft particles at fixed substructure as evidence for QCD, which is opposite to a count-only classifier. Particle-level jets are, however, far from the detector-level jets the networks were trained on. After a crude calorimeter-like merging of neutral particles, most of the extra particles disappear and the ParT effect falls to about a quarter (−0.12 log-odds); one JetClass network then shows none. The size of the effect on detector-level top tagging therefore needs a real Delphes run.

## Issues

1. The H60 conclusion's efficiency statement (ParT ε_S 0.30 → 0.24) and "all six networks" hold for particle-level inputs only. Under the detector proxy the effect is about 0.25× for ParT and absent for ParT_kin.
2. H60 has MPI on, while the Top Landscape training sample has no multiple interactions (per the dataset description). This is an extra source of soft particles that H60 does not mention.
3. The detector proxy is crude: thresholds and cell size were chosen, not tuned to the ATLAS card. It also shifts mass under Hz± (Δm −0.5 GeV for tops, −1.1 GeV for QCD) and moves the EFN by 0.04–0.06, so it is not exactly IRC-matched.
4. The split edit's δ = 0.01 collinear halves are off the data manifold for some networks: the JetClass nets and LorentzNet give 1.9–2.7× their Hz+ response. Per-particle comparisons should be taken as bounds.
5. The permuted-ParT null is not zero under the edits (+0.03 to +0.10; |Δs| up to 0.145 under merge on QCD). It stays below the network effects on top jets.
6. The κ_EFP+N comparison with GBDT(EFP+N) was not rerun; this verify uses the network shifts and causal edits instead.
7. The sign-flip permutation resolution (2,000) is too coarse for Holm over 210 tests, so the bootstrap-z Holm is reported.
