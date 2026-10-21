# H66 verify (independent verifier, round D16)

Script `interp/h66_verify.py` (steps `edits` GPU, `w` CPU, `data` CPU). Jobs: `anrunw-ai-d-h66-verify` (GPU, 6 min), `anrunw-ai-d-h66-verifyw` (1 CPU / 4 GiB, peak 0.48 GiB), `anrunw-ai-d-h66-verifyd` (AI_PY=conda, 1 CPU / 4 GiB, peak 2.5 GiB). Results: `$AI_RESULTS/physics/h66_verify_edits.json`, `h66_verify_w.json`, `h66_verify_data.json` (local copies in `results/physics/`). The worker's own self-check is kept below under "Worker self-check".

## Conclusion

On real Top Landscape top jets, the four Top Landscape taggers lower their top score when objects are added and raise it when objects are removed. This holds whether the new objects are near-duplicate collinear splits, wider splits, or soft particles, while the energy-only EFN does not move. So the networks read the object count itself, not a near-duplicate artefact. On the reweighted simulated jets, the hadronisation-change shift for ParT reproduces at −0.25 log-odds (top log-odds: log p(top) − log p(QCD)). Its size depends on how event pairs are weighted: weighting by the varied jet instead of the default jet gives −0.05, and only the two pair-symmetric weightings give the reported 0.59–0.87 retention. The data count slope reproduces on fresh training jets: −0.025 log-odds per object at fixed m, τ32 and mW. The response of ParT and PFN on QCD jets has no stable sign across edit types.

## What was checked and how

### 1. Code read (`interp/h66_top_count_onmanifold.py`)

- **Axis and coordinates.** Edits act on (pT, η, φ) rows of massless constituents. `jet_inputs` recomputes the jet axis as the constituent sum and uses Δη·sign(η_jet), following the Top Landscape conversion. Splits conserve the jet four-momentum to O(1e−5) relative; the measured |Δm| is ≤ 0.03 GeV on top jets.
- **Padding.** The multiplicity n comes from `jet_nparticles` (TL), stored n (generated), or `len(rows)` (edits). Padding is not counted. `weaver_inputs` truncates at the 128 hardest constituents; top jets have ⟨N⟩ ≈ 50–62, so this rarely binds.
- **Row selection.** TL rows are random draws per class (rng 6611 / 6612), never first-k.
- **Observables and inputs.** Both are computed from the same P4 arrays that are scored.
- **Event pairing.** Pairs are matched by event key.
- **Cross-fit.** Correct: weights for half h come from a model fitted on the other half.

**Defects and mislabels:**
- (a) The weight-permuted band quoted in the reading (−0.085 to +0.038) comes from the stats step under the biased w(V0) weights. Under w_sym the band is [−0.041, +0.029] (this verify).
- (b) The permuted copies are made from `h4_ckpt_part_kin.pt`, which is the **Top Landscape ParT** (`ParT_top`), not the JetClass ParT_kin. The report labels them "ParT_kin".
- (c) The working-point number ParT ε_S 0.500 → 0.444 is computed under w(V0), the weighting the node itself found biased. No w_sym working point exists.
- (d) The reading quotes C2ST 0.61 for the reweighted sample. That is the V0 member of each pair. The Hz+ member under the same weights is at 0.68–0.69, and the count-arm jets at 0.74–0.78 (h66_sym.json).
- (e) `per_object_ci` divides by the mean bootstrap ΔN, as the worker noted.
- (f) In the fixed-effect fits, the bootstrap intervals do not always contain the point estimate (e.g. fine cells: b −0.0272, CI [−0.0290, −0.0267]). This is the expected incidental-parameter effect of resampling with duplicates in 640–2,000 cells. The CI is shifted by about 0.001, which is small next to the effect.

### 2. Reweighting robustness (step `w`, on the H66 paired scores)

**Procedure.**
- The density-ratio model is refitted with the Top Landscape roles swapped: fitted V0 vs T2, with C2ST against T1.
- It uses different HistGB settings (seed 100 + class, 400 iterations, learning rate 0.04, 25 leaves, L2 2) and is cross-fitted over halves A and B.
- Five weightings of each event pair (V0 jet, Hz+ jet from the same parton-level event):
  - unweighted;
  - w(V0);
  - w(Hz+), the mirror image of w(V0), with the regression-to-the-mean bias in the opposite direction;
  - w_sym = √(w(V0) w(Hz+));
  - w_avg = (w(V0) + w(Hz+))/2.
- Metric: weighted mean Δs = s(Hz+) − s(V0), with a 95 % bootstrap CI over pairs (B = 1,000).
- Control: the six weight-permuted Top Landscape ParT copies under the same weights.

**Results, top jets (6,962 pairs):**

| weighting | ΔN | Δm (GeV) | ParT | ParticleNet | LorentzNet | PFN | EFN | permuted band |
|---|---|---|---|---|---|---|---|---|
| none | +4.48 | −0.19 | −0.300 [−0.342, −0.260] | −0.265 | −0.432 | −0.358 | +0.023 | [−0.035, +0.033] |
| w(V0) | +7.52 | −1.34 | −0.472 | −0.364 | −0.384 | −0.593 | +0.114 | [−0.085, +0.036] |
| w(Hz+) | +0.95 | +0.58 | −0.050 [−0.123, +0.026] | +0.013 [−0.102, +0.150] | −0.276 | −0.037 [−0.087, +0.016] | −0.097 | [−0.003, +0.041] |
| **w_sym** | +4.26 | −0.40 | **−0.248 [−0.310, −0.177]** | −0.155 [−0.256, −0.031] | −0.313 [−0.397, −0.231] | −0.308 [−0.357, −0.261] | +0.015 [−0.022, +0.049] | [−0.041, +0.029] |
| w_avg | +4.23 | −0.38 | −0.261 [−0.322, −0.191] | −0.175 [−0.272, −0.062] | −0.330 | −0.315 | +0.009 | [−0.044, +0.031] |

- **C2ST (top, halves A/B).**
  - V0 jets: unweighted 0.81, w(V0) 0.56, w_sym 0.61–0.62, w_avg 0.62.
  - Hz+ jets: unweighted 0.86, w(Hz+) 0.61–0.63, w_sym 0.68–0.69.
- **ESS (effective sample size, (Σw)²/Σw²):** w_sym 2,506, w_avg 2,665, w(V0) 2,216, w(Hz+) 1,891.

**Reading.**
- With the refitted weights, the H66 w_sym numbers reproduce within 0.01 for every network.
- w(V0) and w(Hz+) carry opposite regression-to-the-mean biases:
  - ΔN is +7.5 and +0.95, against +4.5 unweighted;
  - the EFN moves +0.11 and −0.10, against ≈ 0 unweighted.
- Only the symmetric weightings keep ΔN and the EFN at their unweighted values. Under them, ParT, LorentzNet and PFN lie outside the permuted band.
- ParticleNet's interval (upper end −0.031) overlaps the band (lower end −0.041).
- Per added object, ParT's response is −0.053 to −0.067 log-odds under every weighting, including w(Hz+). ParticleNet's per-object response changes sign under w(Hz+).
- Count-arm share of the Hz+ shift under w_sym: ParT 0.50, PFN 0.59, ParticleNet 1.23, LorentzNet 1.96. This reproduces H66.

### 3. Real-jet count edits on a fresh Top Landscape sample (step `edits`)

**Sample.** 4,000 top + 4,000 QCD Top Landscape test rows, rng 7766, with zero overlap with the H66 rows.

**Unedited accuracy / AUC on this sample:**

| model | accuracy | AUC |
|---|---|---|
| ParT | 0.944 | 0.985 |
| ParticleNet | 0.941 | 0.986 |
| LorentzNet | 0.946 | 0.987 |
| PFN | 0.930 | 0.980 |
| EFN | 0.925 | 0.977 |
| ParT_kin (JetClass) | 0.923 | 0.982 |
| PN_kin (JetClass) | 0.934 | 0.984 |
| permuted ParT copies | – | 0.55–0.78 |

**Edits** (8 constituents chosen at random per jet with rng 7767, unless noted):
- **sr8** (as in H66): each chosen constituent is replaced by two pT/2 halves 0.01 apart in (η, φ). ΔN +8, |Δm| ≤ 0.03 GeV.
- **sr8w** (new): the same, with the halves 0.05 apart, so they are no longer near-duplicates. ΔN +8, mean Δm +0.10 GeV.
- **jit8 / jit8w**: the same constituents moved by 0.005 / 0.025 with N fixed. These are the matched controls: same particles, same displacement, no count change.
- **mg8** (new): the 8 nearest-in-ΔR pairs of the softer half are merged (three-momenta added). ΔN −8, mean Δm −0.08 GeV on top jets.
- **add8** (new): 8 soft particles of 0.5 GeV each are added uniformly within ΔR < 0.4 of the jet axis. ΔN +8, Δm +1.1 GeV, ΔpT +4 GeV. This edit does not conserve momentum.
- **ss8**: the 8 softest constituents are split, as in H66.

**Top jets, Δs with 95 % CI (B = 2,000):**

| edit | ParT | ParticleNet | LorentzNet | PFN | EFN | permuted band |
|---|---|---|---|---|---|---|
| sr8 (+8) | −0.342 [−0.360, −0.325] | −0.351 | −0.514 | −0.598 | −0.001 | [−0.031, +0.061] |
| sr8 − jit8 | −0.357 [−0.374, −0.341] | −0.368 | −0.509 | −0.598 | 0.000 | – |
| sr8w (+8, wide) | −0.507 [−0.525, −0.489] | −0.559 | −0.695 | −0.593 | −0.033 | [−0.031, +0.062] |
| sr8w − jit8w | −0.486 [−0.502, −0.470] | −0.522 | −0.660 | −0.595 | −0.008 | – |
| add8 (+8 soft) | −0.429 [−0.439, −0.419] | −0.549 | −0.662 | −0.410 | −0.054 | [−0.160, +0.144] |
| mg8 (−8) | **+0.327 [+0.308, +0.346]** | +0.487 | +0.570 | +0.494 | +0.015 | [−0.144, +0.097] |
| ss8 (+8) | −0.266 | −0.329 | −0.727 | −0.435 | 0.000 | [−0.086, +0.184] |

- The H66 sr8 result reproduces on fresh jets: ParT −0.342 here against −0.334 in H66, per object −0.043 against −0.041.
- **Statistics.** H0: Δs = 0 for each network and edit. Bootstrap-z p < 1e−10 for all 4 TL networks × 5 count edits on top jets; all 20 tests survive Holm.
- **Competing hypotheses:**
  - **Near-duplicate-pair artefact.** The networks would react to two particles 0.01 apart, not to N. This predicts that wider splits and soft additions give smaller responses and merges give about zero. It is excluded: sr8w and add8 give larger negative responses, and mg8 reverses the sign.
  - **Count read.** Δs has the sign opposite to ΔN, in both directions, for every edit type. This is what is observed.

**QCD jets (same edits):**
- ParticleNet and LorentzNet respond positively to adding objects and negatively to merging. The sign is consistent with each other and opposite to the data slope:
  - ParticleNet: sr8 +0.716, add8 +0.332, mg8 −1.251;
  - LorentzNet: sr8 +0.991, add8 +0.975, mg8 −0.565.
- ParT's response on QCD jets is not monotone in N:
  - sr8 +0.128;
  - sr8w −0.008 [−0.025, +0.010];
  - add8 −0.019;
  - mg8 +0.113 (removing objects also raises it).
- PFN's response on QCD jets is −0.045 to +0.094 depending on the edit.
- The H66 statement "ParT +0.017 per object on QCD jets" is specific to the 0.01 split.

### 4. Data count slope on fresh Top Landscape training jets (step `data`)

**Sample.** 60,000 top + 60,000 QCD training rows, rng 7768, with zero overlap with the H66 rng-6612 rows. m, τ32 and mW are computed from the 128 hardest constituents; median |m − stored jet mass| is 0.006 GeV. N is `jet_nparticles`.

**Fit.** b_N is the coefficient on N in a logistic regression of the label, with one free intercept per cell (C = 1e4) and a 40-replicate bootstrap.

| cells | all masses | 140 < m < 200 GeV |
|---|---|---|
| H66: 10 m × 8 τ32 × 8 mW quantile bins | −0.0254 [−0.0273, −0.0239] | −0.0325 |
| finer: 20 × 10 × 10 | −0.0272 | −0.0334 |
| H66 cells × 4 jet-pT quartiles | −0.0263 | −0.0341 |
| null: N shuffled within H66 cells (20 shuffles) | −0.0002 (range −0.0018 to +0.0009) | – |

- H0 (no N dependence at a fixed cell) is rejected. The permutation p is ≤ 1/21, limited by the 20 shuffles; the observed value is more than 10 times the null range.
- The slope does not shrink with finer cells or with jet-pT conditioning, so residual mass or pT confounding inside the H66 cells does not produce it.
- The H66 value of −0.024 reproduces.

## Status of each part of the reading

**Holds:**
- On real Top Landscape top jets, the four TL networks' count response is negative per added object and positive per removed object, and the EFN stays at about 0.
- The data slope at fixed (m, τ32, mW) is negative and of about the same order (the networks' per-object response is 1.6–3.6 times larger, depending on edit and network).
- The w_sym numbers reproduce with an independently refitted weight model.

**Needs correction:**
- (i) The 0.59–0.87 retention holds under the pair-symmetric weightings (w_sym, w_avg) only. With the mirror weighting w(Hz+), ParT, ParticleNet and PFN go to zero within errors. That weighting carries the reverse regression bias (ΔN 0.95, EFN −0.10), so it does not measure the same hadronisation change. Still, the pair-level estimand is not identified without a parton-level weighting.
- (ii) Only the V0 member is matched at C2ST 0.61. The Hz+ member is at 0.68.
- (iii) Under w_sym, ParticleNet's interval overlaps the permuted band [−0.041, +0.029]. The band quoted in the reading (−0.085 to +0.038) is the biased w(V0) band.
- (iv) The ε_S 0.500 → 0.444 number uses the biased weights.
- (v) The permuted copies are of the TL ParT, not ParT_kin.
- (vi) On QCD jets, ParT's and PFN's sign depends on the edit type. ParticleNet and LorentzNet are consistently positive, which is opposite to the data.

---

## Worker self-check (kept)


### Worker self-check: conclusion

The two halves of the generated sample, each weighted by a model that never saw it, give the same sign and similar sizes for every Top Landscape tagger: ParT −0.27 vs −0.22 log-odds on top jets, and LorentzNet and PFN agree within 0.02. This makes it unlikely that the reweighted result is a fitting artefact of one half. One defect was found and handled: weighting each event pair by its V0 jet alone biases the comparison through regression to the mean. The pair-symmetric weights used for the main numbers remove it. The first figure (`h66_count_onmanifold.png`, left panel) still shows the biased V0-only weighting scan and should be read only as a C2ST trace.

### Code check (`interp/h66_top_count_onmanifold.py`)

- **Pairing.** V0, Hz+ and count-arm jets are matched by event key (`key|…`), and only pairs present in all three are used: 6,962 top and 8,000 QCD. The count arm reaches the exact Hz+ object count in 99.67 % (A) and 99.62 % (B) of pairs (score log). Its pooled ΔN of 4.343 equals the Hz+ ΔN of 4.345 to 0.002.
- **Cross-fit.**
  - In `step_stats`, the ratio model fitted on half h is applied to half ho and stored as `W[(ho, cls)]`. The paired loop reads `W[(h, cls)]` for half h, so each half uses the other half's model.
  - `step_sym` fits on `ho` and applies to `h`, with the same seeds and the same deterministic HistGB + CalibratedClassifierCV (cv = 3, unshuffled). It reproduces the stats-step w(V0) numbers exactly: ParT top Hz+ −0.465/−0.466, ESS 2,216.
  - The C2ST reference T2 is disjoint from the fitting rows T1 (alternate rows of the same rng-6611 draw). The T1-vs-T2 null is 0.495 and 0.506.
- **Weights.** Capped at 30. The fraction of jets at the cap is 0.0002 per half and class, and the 99th percentile is 6.3–7.8. The weighted means of N, τ32, pT^D and f1 land on the TL values within 1 %.
- **Edits on TL jets.**
  - split_rand k = 8 keeps the jet four-momentum: max |Δm| 0.025 GeV on top jets, mean Δm 0.004 GeV.
  - jitter moves the same objects by the same 0.005 without changing N.
  - EFN responses are ≤ 0.002 in every split arm.
- **Statistics.**
  - The weighted bootstrap resamples pairs and recomputes the weighted mean.
  - The sign-flip permutation for the rest arm uses w·D. Its smallest attainable p is 1/2001, so Holm floors at 0.007 for 14 tests.
  - `wquant` was patched against an index overflow at q → 1. The patch is not exercised at q = 0.5.
- **Known limitation (not a bug).** `per_object_ci` divides the bootstrap Δs by the mean bootstrap ΔN rather than per replicate, so it slightly understates the ratio's interval.
- **Bias found.** w(V0) depends on the V0 member of the pair only, so the weighted Hz+ − V0 contains a regression-to-the-mean term:
  - ΔN 4.48 → 7.58;
  - rest-arm Δm −0.17 → −1.21 GeV and Δτ32 +0.0007 → −0.0136;
  - EFN +0.02 → +0.12.

  w_sym = √(w(V0) w(Hz+)) removes it: ΔN 4.34, Δm −0.35 [−0.79, 0.10] GeV, EFN +0.018 [−0.017, 0.052].
- **NaN / OOM.** None in the score, stats or sym logs. Peak memory was 0.85 GiB in both CPU jobs.

### Unedited accuracy reproduced

- ParT on TL rows T1 and T2: AUC 0.9868 and 0.9858. The H60/H61 value is 0.9864.
- ParT on V0 halves A and B: 94.4 % and 94.5 % accuracy (H61: 94.4 %).
- All TL networks keep AUC ≥ 0.974 on the w(V0)-weighted V0 sample.

### Figure check

- `results/physics/h66_sym.png` (main figure). Two panels for top jets: Hz+ − V0 (left) and count arm − V0 (right). Each shows the 7 models under three weightings, with 95 % bootstrap bars and the y axis in top log-odds per jet pair. The legend gives ΔN per weighting (4.5, 7.6, 4.3), which marks the biased orange points. It is readable and carries one message: the TL networks stay negative under the unbiased weighting, and the EFN stays at 0.
- `results/physics/h66_count_onmanifold.png`.
  - Left panel: Δs against C2ST for the tempered w(V0)^α. It carries the regression-to-the-mean bias (the EFN drifts to +0.12), so it should not be used as evidence for or against H_alt.
  - Right panel: per-object network slopes on real TL top jets next to the data's b_N (red). It is correct and readable, with the axis in log-odds per added object.

### Cross-fit half consistency (top jets, w_sym; ESS A 1,279, B 1,281)

| model | half A | half B | pooled |
|---|---|---|---|
| ParT | −0.274 | −0.224 | −0.249 |
| ParticleNet | −0.206 | −0.106 | −0.156 |
| LorentzNet | −0.308 | −0.297 | −0.303 |
| PFN | −0.319 | −0.308 | −0.313 |
| EFN | +0.004 | +0.032 | +0.018 |
| ParT_kin | −0.034 | −0.040 | −0.037 |
| PN_kin | −0.137 | −0.180 | −0.158 |
| count arm, ParT | −0.140 | −0.123 | −0.132 |
| V0 C2ST under w_sym | 0.607 | 0.609 | – |

QCD jets under w_sym, half A vs half B:

| model | half A | half B |
|---|---|---|
| ParT | −0.196 | −0.185 |
| ParticleNet | +0.047 | +0.137 |
| LorentzNet | +0.162 | +0.195 |
| PFN | −0.129 | −0.150 |

Every network keeps its sign across halves. ParticleNet differs most between halves (0.10 log-odds on both classes), consistent with its wider bootstrap intervals.
