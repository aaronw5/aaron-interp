# H66: does the Top Landscape taggers' response to the object count survive on-manifold reweighting? (claim D24 v2)

Node H66 (loop D, round D16). Script `interp/h66_top_count_onmanifold.py` (steps `score` GPU, `stats` CPU, `sym` CPU). Jobs: `anrunw-ai-d-h66-score3` (GPU, 21 min), `anrunw-ai-d-h66-stats` (1 CPU / 4 GiB, peak 0.85 GiB), `anrunw-ai-d-h66-sym` (1 CPU / 2 GiB). Results: `$AI_RESULTS/physics/h66_scores.npz`, `h66_tltrain.npz`, `h66_run.json`, `h66_sym.json`; local copies `results/physics/h66_run.json`, `h66_sym.json`, `h66_count_onmanifold.png`, `h66_sym.png`, `h66_stats.log`, `h66_sym.log`.

## Conclusion

We reweighted our simulated (Delphes) top jets so that a classifier can barely tell them from the Top Landscape jets the taggers were trained on. A hadronisation change that adds about 4 reconstructed objects per top jet at fixed mass, τ32 and W-pair mass still lowers the ParT top log-odds by −0.25 (95 % CI −0.32 to −0.18), against −0.30 without reweighting; ParticleNet, LorentzNet and PFN also stay negative and the IRC-safe EFN stays at zero. Independently, the Top Landscape training data show the same sign: among jets in the same mass, τ32 and W-pair-mass bin, each extra object lowers the true top log-odds by 0.024. Still uncertain: the reweighted sample reaches a classifier AUC of 0.61, not the 0.60 target, and the networks' own score distributions remain distinguishable (AUC 0.61–0.68). The count-only edit explains about half of ParT's shift but more than all of ParticleNet's and LorentzNet's, so the non-count part of the hadronisation change is model-dependent.

(C2ST AUC: out-of-fold AUC of a classifier trained to separate two samples; 0.5 = indistinguishable. Log-odds s = log p(top) − log p(QCD).)

## Question and hypotheses

**Question.** H61/H61v measured that top taggers lower their top score when Pythia hadronisation adds about 7 % more reconstructed objects at fixed m, τ32, mW. The Delphes jets differ from Top Landscape jets (C2ST 0.74 in H61; 0.80 on the 24 features here). Does the response remain when the generated jets are reweighted to look like Top Landscape jets? Does the Top Landscape data favour QCD as N rises at fixed shape?

**H1 (D24 v2, count read on-manifold).** Predictions:
- Δs of ParT / ParticleNet / LorentzNet stays negative at C2ST ≤ 0.6, within 30 % of the unweighted value.
- The count arm reproduces most of it.
- EFN ≈ 0.
- The Top Landscape data favour QCD as N rises at fixed (m, τ32, mW).

**H_alt (off-manifold artefact).** Predictions:
- Δs shrinks toward the weight-permuted-network band as C2ST → 0.5.
- The data show no N dependence at fixed (m, τ32).

**H0 (decomposition).** Δs(Hz+) = Δs(count arm), i.e. the rest arm (Hz+ − count) is zero.

## Procedure

### Sample
- Generated jets: H61 (half A, workers 6101–6110) and H61v (half B, 6201–6210). Pythia 8.317, MPI off, Delphes 3.5.1 ATLAS card, anti-kT R = 0.8 on energy-flow objects, 550 < pT < 650 GeV. Each half has 5k top + 5k QCD jets per setting.
  - V0: default (Monash) hadronisation.
  - Hz+: aLund 0.79, bLund 0.88, σ 0.247 GeV, made from the same parton-level event as V0.
  - Event pairs present in V0, Hz+ and the count arm: 6,962 top and 8,000 QCD (A + B).
- Top Landscape (TL) test rows, rng 6611, 20k per class, split alternately:
  - T1 (10k per class) for fitting the weights;
  - T2 (10k per class) for the C2ST check;
  - every second T2 row (5k + 5k) for the real-jet edits.
- TL training rows, rng 6612, 60k per class, for the data check. N = `jet_nparticles`. m, τ32, mW, τ21 and mSD are computed from the 128 hardest constituents; m agrees with the stored jet mass to a median 0.006 GeV.

### Edits and arms (all on the same parton-level events)
- **Hz+ − V0:** the hadronisation change. ΔN = +4.5 objects on top jets (+7 %). Unweighted Δm −0.19 GeV, Δτ32 +0.0004, ΔmW +0.23 GeV.
- **Count arm − V0:** the V0 jet brought to exactly the Hz+ object count. For d = N(Hz+) − N(V0) > 0, the d softest objects are each split into two pT/2 halves ±0.005 apart in (η, φ). For d < 0, the |d| nearest soft pairs are merged. This changes N only: Δm −0.014 GeV, Δτ32 < 1e−4, EFN shift +0.003.
- **Rest = Hz+ − count arm:** the non-count part of the hadronisation change (ΔN ≈ 0).

### Weights
- Per class, a Platt-calibrated HistGB classifier separates V0 jets from T1 jets on 24 features:
  - N, m, τ32, mW, pT^D;
  - pT fractions of the objects ranked 1, 2, 3, 5, 10, 20 and 40;
  - counts with pT < 1, 1–2 and 2–5 GeV;
  - girth, pT fractions within ΔR 0.1, 0.2 and 0.4, and counts in ΔR bins;
  - jet pT and |η|.
- w = p/(1 − p) × n_V0/n_T1, capped at 30, mean 1.
- **Cross-fit:** the classifier is fitted on half A and applied to half B, and vice versa.
- **Two weightings:**
  - (i) **w(V0):** the weight of the V0 jet, as planned.
  - (ii) **Pair-symmetric w_sym = √(w(V0 jet) · w(Hz+ jet)):** used for the main result after (i) was found to bias the pairs (see Reading 1).
- **Tempered weights w^α** (α = 0, 0.25, 0.5, 0.75, 1) trace Δs against C2ST.

### Models
- Top Landscape ParT, ParticleNet, LorentzNet (batch 32, key-hashed order), PFN, EFN.
- JetClass ParT_kin and PN_kin (logit Tbqq − logit QCD).
- Six weight-permuted copies of ParT_kin (seeds 0–5) as the untrained-network band.
- Score s = top log-odds.

### Metric
- Δs = Σ_j w_j [s(variant_j) − s(V0_j)] / Σ_j w_j over pairs, with a 95 % bootstrap interval over pairs (B = 2,000).
- H0 for the rest arm: sign-flip permutation test (2,000 flips), Holm across 7 models × 2 classes = 14 tests.
- The stats step also gives network − EFN bootstrap-z p-values, Holm over 84 tests (6 networks × [3 matched arms + 4 TL edits] × 2 classes).
- Working point: the weighted median of the V0 top scores (ε_S = 0.50), with McNemar tests on paired calls.

### Data check (TL training jets, 120,000)
- Logistic regression of the label on N with one free intercept per (m, τ32, mW) cell: 10 × 8 × 8 quantile bins = 640 cells, C = 1e4. The slope b_N is the change in top log-odds per object at fixed cell.
  - Also inside 140 < m < 200 GeV.
  - Also per 20-GeV mass bin.
  - 100 bootstrap refits.
- Unconditional logistic regression of the label on N.
- Cross-fitted GBDTs on (m, τ32, mW, N) and on (m, τ32, mW, τ21, mSD, pT, N), with the finite difference [f(N + 5) − f(N − 5)]/10.

### Real-jet edits (TL test, 5k top + 5k QCD)
- split_rand k = 2, 4, 8: k random constituents, each split into two pT/2 halves 0.01 apart.
- split_soft k = 8: the 8 softest constituents, split the same way.
- jitter 8: the same 8 objects moved by 0.005, with N fixed.
- Slope through the origin of Δs on ΔN over k = 2, 4, 8.

## Numbers

### Unedited accuracy (accuracy at s > 0, AUC)

| sample | ParT | ParticleNet | LorentzNet | PFN | EFN | ParT_kin | PN_kin |
|---|---|---|---|---|---|---|---|
| TL T2 (20k) | 0.939 / 0.986 | 0.939 / 0.985 | 0.941 / 0.987 | 0.927 / 0.979 | 0.921 / 0.976 | 0.923 / 0.982 | 0.928 / 0.983 |
| V0 half A (10k) | 0.944 / 0.988 | 0.942 / 0.988 | 0.946 / 0.989 | 0.922 / 0.983 | 0.928 / 0.980 | 0.932 / 0.988 | 0.945 / 0.989 |
| V0, w(V0)-weighted (A + B) | 0.935 / 0.983 | 0.932 / 0.982 | 0.935 / 0.983 | 0.910 / 0.974 | 0.906 / 0.968 | 0.907 / 0.978 | 0.933 / 0.982 |

The TL ParT AUC (0.986) matches the H60/H61 value (0.9864). The weight-permuted copies score AUC 0.49–0.82.

### C2ST (top / QCD; mean of halves A and B)

| weighting | main 24 features | substructure not in the weights (τ21, mSD, subjet masses, m, τ32, mW) | network scores | N only | all features |
|---|---|---|---|---|---|
| none | 0.80 / 0.85 | 0.60 / 0.64 | 0.77 / 0.74 | 0.62 / 0.55 | 0.87 / 0.89 |
| w(V0), α = 1 | 0.545 / 0.573 | 0.58 / 0.61 | 0.68 / 0.61 | 0.49 / 0.50 | 0.72 / 0.73 |
| w_sym (V0 jets) | 0.608 / 0.621 | – | – | – | – |
| w_sym (Hz+ jets) | 0.680 / 0.670 | – | – | – | – |

- The T1-vs-T2 null C2ST is 0.495 (top) and 0.506 (QCD).
- ESS (effective sample size) under w_sym: 2,559 of 6,962 top pairs and 2,175 of 8,000 QCD pairs.
- Weighted means move to TL: top ⟨N⟩ 61.7 → 55.1 (TL 54.8), τ32 0.505 → 0.548 (0.548), pT^D 0.264 → 0.297 (0.297).

### Top jets: paired shifts Δs (95 % CI)

| model | Hz+, unweighted | Hz+, w(V0) | Hz+, w_sym | count arm, w_sym | rest, w_sym (Holm p, H0 rest = 0) |
|---|---|---|---|---|---|
| ParT (TL) | −0.300 [−0.342, −0.258] | −0.465 [−0.541, −0.392] | **−0.249 [−0.316, −0.182]** | −0.132 [−0.155, −0.107] | −0.117 [−0.183, −0.057] (0.010) |
| ParticleNet (TL) | −0.265 [−0.328, −0.186] | −0.357 [−0.453, −0.254] | **−0.156 [−0.247, −0.061]** | −0.197 [−0.230, −0.163] | +0.041 [−0.048, 0.133] (1.0) |
| LorentzNet (TL) | −0.432 [−0.487, −0.379] | −0.391 [−0.483, −0.296] | **−0.303 [−0.386, −0.227]** | −0.623 [−0.661, −0.582] | +0.320 [0.245, 0.392] (0.007) |
| PFN (TL) | −0.358 [−0.388, −0.330] | −0.596 [−0.648, −0.546] | **−0.313 [−0.359, −0.266]** | −0.185 [−0.207, −0.163] | −0.128 [−0.165, −0.090] (0.007) |
| EFN (TL, IRC-safe) | +0.023 [0.003, 0.044] | +0.120 [0.080, 0.161] | **+0.018 [−0.017, 0.052]** | +0.002 | +0.015 (1.0) |
| ParT_kin (JetClass) | −0.074 [−0.107, −0.042] | +0.005 [−0.059, 0.068] | −0.037 [−0.093, 0.015] | −0.242 [−0.260, −0.225] | +0.205 [0.150, 0.258] (0.007) |
| PN_kin (JetClass) | −0.203 [−0.243, −0.167] | −0.216 [−0.286, −0.142] | −0.158 [−0.226, −0.099] | −0.477 [−0.504, −0.453] | +0.319 [0.256, 0.377] (0.007) |
| ΔN, Δm (GeV), Δτ32 | +4.48, −0.19, +0.0004 | +7.58, −1.21, −0.0136 | +4.34, −0.35, +0.0007 | +4.34, −0.014, 0.0000 | 0.00, −0.34, +0.0009 |

- **Weight-permuted ParT band** (Hz+, top, w(V0)): −0.085 to +0.038, mean −0.011. Count arm: −0.074 to +0.140.
- **w_sym / unweighted ratio:** ParT 0.83, ParticleNet 0.59, LorentzNet 0.70, PFN 0.87.
- **Per object under w_sym:** ParT −0.057 [−0.073, −0.042], unweighted −0.067.
- **Count arm share of Hz+ under w_sym:** ParT 0.53, PFN 0.59, ParticleNet 1.26, LorentzNet 2.06.
- **Working point** (w(V0); ε_S = weighted fraction of V0 top jets above the weighted median):
  - Hz+: ParT 0.500 → 0.444 (McNemar p 3e−16); EFN 0.500 → 0.516 (p 0.07).
  - Count arm: ParT 0.500 → 0.453.
- **Tempered w(V0)^α, ParT top** (C2ST: Δs): 0.80: −0.300; 0.73: −0.347; 0.66: −0.390; 0.59: −0.429; 0.55: −0.465. EFN rises from +0.023 to +0.120 along the same path, and ΔN from 4.5 to 7.6 (see Reading 1).

### QCD jets (w_sym)

Hz+ ΔN = +3.96:

| model | Δs |
|---|---|
| ParT | −0.191 [−0.241, −0.139] |
| ParticleNet | +0.092 [−0.003, 0.187] |
| LorentzNet | +0.178 [0.093, 0.271] |
| PFN | −0.140 |
| EFN | +0.013 [−0.032, 0.059] |
| ParT_kin | −0.039 |
| PN_kin | +0.077 |

In the count arm, ParT moves −0.028, ParticleNet +0.060, LorentzNet +0.262 and ParT_kin +0.142.

### Real Top Landscape test jets: response per split object (slope over k = 2, 4, 8; 95 % CI)

| model | top jets | QCD jets |
|---|---|---|
| ParT | −0.041 [−0.043, −0.039] | +0.017 [0.014, 0.020] |
| ParticleNet | −0.043 [−0.046, −0.041] | +0.093 |
| LorentzNet | −0.064 | +0.122 |
| PFN | −0.074 | −0.004 |
| EFN | −0.0002 | +0.0002 |
| ParT_kin | −0.009 [−0.011, −0.007] | +0.067 |
| PN_kin | −0.028 | +0.080 |
| permuted ParT copies | −0.004 to +0.008 | −0.011 to +0.010 |

- split_rand k = 8 changes m by at most 0.025 GeV on top jets and N by +8.
- At the TL 50 % top-efficiency cut, ParT ε_S goes 0.500 → 0.442 (McNemar p 1e−53).
- jitter 8 (N fixed, the same 8 objects moved by 0.005) gives |Δs| ≤ 0.018 for the TL networks on top jets and ParT_kin +0.038.
- Network − EFN for split_rand and split_soft on top jets: Holm p < 1e−38 for every TL network.

### Top Landscape data: top log-odds per object (120,000 training jets)

| fit | b_N | 95 % CI |
|---|---|---|
| unconditional label ~ N | +0.043 | [0.042, 0.044] |
| fixed (m, τ32, mW) cell | −0.024 | [−0.026, −0.022] |
| fixed cell, 140 < m < 200 GeV (58,174 jets) | −0.029 | [−0.032, −0.026] |
| GBDT(m, τ32, mW, N), finite difference, top jets | −0.017 | [−0.018, −0.017] |
| GBDT(m, τ32, mW, N), finite difference, QCD jets | −0.020 | |
| GBDT with τ21, mSD, pT added, top jets | −0.029 | |
| GBDT with τ21, mSD, pT added, QCD jets | −0.017 | |

- Per 20-GeV mass bin (τ32 × mW cells), the slope is between −0.055 and −0.012 for 60 < m < 200 GeV and about 0 above 200 GeV. The 40–60 GeV bin contains no top jets, so its slope (+0.48) is not identified.

## Reading

1. **The V0-only weights bias the paired comparison; pair-symmetric weights remove the bias.**
   - Under w(V0), the Hz+ − V0 pairs have ΔN = 7.6 instead of 4.5, and the rest arm acquires Δm = −1.2 GeV and Δτ32 = −0.014. The count arm stays exact.
   - The cause is regression to the mean. The weights up-weight V0 jets that fluctuated toward TL (fewer objects; top τ32 higher, QCD mass higher), and the paired Hz+ jet of the same event does not share the fluctuation.
   - This bias makes the w(V0) Δs larger (ParT −0.47) and moves the EFN to +0.12. So the α-scan with w(V0) does not test the off-manifold alternative cleanly.
   - With w_sym, ΔN = 4.3, Δm = −0.35 GeV (CI includes 0) and the EFN shift is +0.02. The w_sym numbers are the main result. Their V0 C2ST is 0.61 (top) and 0.62 (QCD), slightly above the 0.6 target.
2. **Top jets under w_sym compared with each hypothesis.**
   - ParT, ParticleNet, LorentzNet and PFN remain negative, with CIs excluding 0 and outside the permuted band (−0.09 to +0.04). The EFN CI includes 0.
   - Relative to unweighted, the magnitudes are 0.83 (ParT), 0.59 (ParticleNet), 0.70 (LorentzNet) and 0.87 (PFN). ParT and PFN meet the "within 30 %" prediction, LorentzNet is at the edge, and ParticleNet does not.
   - None approaches the permuted band, the H_alt prediction for a matched sample.
3. **The count arm, which changes only N, gives a negative shift for every TL network under all three weightings.** Its share of the Hz+ shift is model-dependent: about half for ParT and PFN, more than all for ParticleNet and LorentzNet. H0 (rest = 0) is rejected for ParT, LorentzNet, PFN, ParT_kin and PN_kin (Holm p ≤ 0.01) and not for ParticleNet or EFN.
4. **On Top Landscape training jets, at fixed (m, τ32, mW) cell, more objects go with a lower top fraction (b_N −0.024 per object).**
   - The sign agrees in the top-mass window, in both GBDT estimates and in every 20-GeV mass bin from 60 to 200 GeV.
   - The H_alt data prediction (no N dependence at fixed shape) is excluded.
   - The networks' per-object response on real TL top jets (−0.041 to −0.074 for the TL networks) has the same sign and is 1.7–3.1 times b_N.
   - The two numbers are not the same quantity. b_N compares different jets with different N, which also differ in other properties. The split edit changes N alone.
5. **QCD jets.**
   - The networks disagree in sign: ParT and PFN move toward QCD, while ParticleNet, LorentzNet and the JetClass networks move toward top under splits.
   - The data slope for QCD jets is negative (GBDT −0.020).
   - The ParticleNet and LorentzNet QCD-jet split responses have the opposite sign to the data.
6. **Not matched.**
   - After weighting, a classifier on the networks' own scores still separates weighted V0 from TL (0.61–0.68), and so does one on all 31 features (0.71–0.73). The weights match the 24 chosen features, not everything the networks read.
   - ESS is about 2,200–2,600 pairs per class.

## Deviations from the plan

- The planned C2ST target ≤ 0.6 is met by w(V0) (0.55–0.57) but not by the unbiased pair-symmetric weights (0.61–0.62). The w_sym step (`--step sym`) was added after the stats step showed the regression-to-the-mean bias (debug attempt 1 of 2).
- The planned reference classifiers LR(N), GBDT(EFP), GBDT(EFP+N) and MLP(m, τ32, mW) were not rescored on the reweighted sample. The data's conditional count slope b_N, from logistic and GBDT fits on TL training jets, replaces them as the physics reference.
- Weight-permuted copies exist only for ParT_kin (six seeds), not for each architecture.
- The tempered-weight scan was run with w(V0) only, which carries the bias above. No α-scan with w_sym was run.
