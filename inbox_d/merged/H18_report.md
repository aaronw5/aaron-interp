# H18 — Soft-particle count response: edit difference or model difference (claims D7 vs D10; open question b)

## Conclusion

The same soft-particle edits were applied to quark–gluon (QG), JetClass, Top Landscape and hls4ml taggers. When the edit adds energy flow (k soft particles copied from the jet's own soft half), every network moves toward the high-multiplicity class. In QG this response is about as large as that of the IRC-safe EFN, a network that cannot count particles: slope −0.07 versus −0.06 per particle.

When the edit changes the count while leaving IRC-safe content almost fixed, the datasets differ. This holds both for one on-manifold particle and for the diffuse particles moved close to the axis:

- **QG networks:** stay within 0.03 per particle, against 0.08 for a count-only likelihood ratio.
- **JetClass W-vs-QCD networks:** move by 0.05–0.18 per particle.

So the D7/D10 contrast is reproduced under identical edits. It comes from the models and datasets, not from the edits. D7 ("QG networks do not count") holds only for count at fixed IRC-safe content.

Uncertain:
- The compact-core control (c4) is off the data manifold (C2ST ≥ 0.91).
- The k-particle edit changes IRC-safe content as well as the count.
- In the top tasks the networks move against the sign of the count-only reference. That is not explained here.

## Procedure

- **Script:** `interp/h18_softcount_cross.py`.
- **Jobs:** d-h18-{qg,jc,top} (GPU), d-h18-hls (jax), d-h18-hlss (conda), d-h18-ref-{qgpy,qghw,jc,top,hls} (venv-ef, 4 CPU), d-h18-stats2.
- **Results:** `results/physics/h18_{qgpy,qghw,jc,top,hls}.json`, `h18_summary.json`, `h18_*_model.json`, `h18_*_refs.json`, figure `results/physics/h18_softcount.png`. Per-jet npz files are on the volume only.

**Samples (seed 4181, random rows):**
- QG Pythia test: 5,000 q + 5,000 g.
- QG Herwig 7.1 (files 18–19): 5,000 + 5,000.
- JetClass first test file: 3,334 each of QCD, Wqq and Tbqq for the edits; 1,000 of each other class for the unedited 10-class accuracy.
- Top Landscape test: 5,000 + 5,000.
- hls4ml n64 test: 2,000 per class.

**Score.** Signal log-odds s, so Δs < 0 means a move toward background.

| Task | Score s |
|---|---|
| QG | log p(q) − log p(g) |
| JetClass | logit_W − logit_QCD; logit_Tbqq − logit_QCD |
| Top | top log-odds |
| hls4ml | log p(W) − log(1 − p(W)) |

**Inputs.**
- JetClass, Top and hls4ml: inputs are taken about the stored axis.
- QG: the input code recomputes the E-scheme axis. The median offset is ≤ 1.9 mrad for k ≤ 20 and 5 mrad for c5_20.

**Edits.**

| Edit | What it does |
|---|---|
| e1 | One particle as in the H5v generator: the jet's softest pT, at a random particle's position + 0.05, type copied |
| k1–k20 | k copies of random members of the jet's softer half (pT, mass, type), at the donor's distance from the axis and a random azimuth |
| pu5 / pu20 | The H13 diffuse disc (ρ in GeV per unit area; R = 0.4 for QG, 0.8 otherwise), neutral hadrons |
| c4 | The pu particles at r/20: count and ΣpT kept, Δm 0.1–4.6 GeV |
| c5 | One particle with the kept ΣpT and mass of pu |
| c0 | Positions of the softer half permuted (count fixed) |

**References.** All are trained on the dataset's own training split (60,000 jets; JetClass uses the first train file of each class):
- LR(N): histogram log-likelihood ratio of the multiplicity the model sees.
- EFPN: GBDT on EFPs (d ≤ 5, β = 1) plus N.
- EFP: GBDT on EFPs only (count-blind).
- Herwig sample only: an extra LR(N) built from the Herwig training files.

**Metrics.**
- Mean Δs with a 95 % bootstrap over jets (B = 1,000).
- Response per added particle: mean Δs / mean ΔN_seen.
- Slope of mean Δs in k = 1, 2, 5, 10, 20.
- % calls changed and signal efficiency at the threshold that gives ε_S = 0.5 on unedited jets (McNemar).
- Sign-flip p (normal limit; Monte Carlo P = 2,000 as a cross-check). Paired bootstrap p for slope_net − slope_ref.
- Holm correction over 512 tests (networks × edits × tasks, plus slope comparisons).
- C2ST (GBDT on low-level summaries; 4,000 jets).

## Unedited performance on the samples used (known values in brackets)

| Dataset | Model performance on this sample [known value] |
|---|---|
| QG Pythia (AUC) | ParT_kin 0.905 [0.901]; ParT (kinpid) 0.913 [0.912]; PN_kin 0.905 [0.902]; PN 0.911 [0.906]; LorentzNet 0.919 [0.917]; PFN 0.886 [0.883]; EFN 0.888 [0.884] |
| QG Herwig (AUC) | ParT_kin 0.809 [0.808]; ParT 0.806 [0.805]; PN_kin 0.809 [0.808]; EFN 0.803 [0.802] |
| JetClass (10-class balanced accuracy) | ParT_full 0.860 [0.859]; ParT_kin 0.736 [0.736–0.745]; PN_full 0.840 [0.839–0.846]; PN_kin 0.694 [0.695–0.704] |
| Top (AUC) | ParT 0.9856 [0.9858]; PN 0.9856 [0.9858]; LorentzNet 0.9862 [0.9870]; PFN 0.9794 [0.9819]; EFN 0.975 [~0.98] |
| hls4ml (5-class accuracy) | JEDI-linear 0.811 [0.809]; Deep Sets on its 500-jet smoke sample 0.810 [0.806] |

- Nulls: weight-permuted ParT_kin (QG) AUC 0.55; redrawn ParT_full 0.10; permuted PFN (Top) AUC 0.70.
- 22.5 % of hls4ml jets fill all 64 slots, so there ΔN_seen < k (k20: +12.9 seen particles).

## Numbers (mean Δs; per particle where stated; 95 % CIs in the JSON files)

| Dataset / task | Networks: k-slope per particle | EFN (IRC safe) | EFP (count-blind) | LR(N) | EFPN |
|---|---|---|---|---|---|
| QG Pythia | −0.068 to −0.074 | −0.060 | −0.045 | −0.077 | −0.090 |
| QG Herwig | −0.077 to −0.081 | −0.067 | −0.049 | −0.081 | −0.098 |
| JetClass W vs QCD | −0.109 to −0.121 | – | −0.011 | −0.041 | −0.057 |
| JetClass top vs QCD | −0.034 to −0.001 | – | +0.039 | +0.058 | +0.011 |
| Top | −0.035 to −0.004 | −0.007 | +0.018 | +0.033 | −0.032 |
| hls4ml W vs rest | JEDI −0.071, Deep Sets −0.063 | −0.023 | −0.021 | −0.050 | −0.046 |

Count changed with IRC-safe content about fixed (the EFP and EFN responses here are |Δ| ≤ 0.013 per particle):

| Dataset / task | e1, networks (one particle) | e1, LR(N) | c4_20 per particle, networks | c4_20, LR(N) |
|---|---|---|---|---|
| QG Pythia | +0.004 to +0.024 (PFN −0.056) | −0.095 | −0.031 to +0.011 | −0.079 |
| QG Herwig | +0.002 to +0.016 (PFN −0.055) | −0.105 | −0.045 to 0.000 | −0.085 |
| JetClass W vs QCD | −0.048 to −0.113 | +0.023 | −0.026 (ParT_kin), −0.108 to −0.177 | −0.043 |
| JetClass top vs QCD | −0.035 to +0.087 | +0.103 | −0.011 to −0.033; PN_kin −0.266 | +0.049 |
| Top | −0.026 to +0.029 | +0.073 | ParT −0.020, PN +0.005, PFN −0.042, LorentzNet −0.147 | +0.003 |
| hls4ml | JEDI −0.042, Deep Sets −0.041, EFN −0.004 | −0.033 | JEDI −0.019, Deep Sets −0.012 | −0.090 |

**Classification calls at the ε_S = 0.5 threshold:**
- e1 changes ≤ 2.0 % of calls for every trained network. LR(N) changes 2–15 %.
- k20 changes 15–24 % of calls.
- Signal efficiency after k20, starting from 0.50:

| Dataset | Signal efficiency after k20 |
|---|---|
| QG quark | 0.09–0.12 |
| JetClass W | 0.03–0.07 |
| JetClass top | 0.09–0.13 |
| Top | 0.17–0.21 |
| hls4ml W | 0.007–0.008 |

**C2ST AUC** (unedited vs unedited is 0.48–0.52 on these subsamples):

| Edit | C2ST AUC |
|---|---|
| e1 | 0.47–0.54 |
| k1–k2 | ≤ 0.55 |
| k5 | 0.54–0.62 |
| k20 | 0.84–0.92 |
| pu20 | 0.91–1.00 |
| c4_20 | 0.91–1.00 |
| c5 | 0.48–0.63 |
| c0 | at the null |

**Statistics:**
- 490 of 512 tests are below 0.05 after Holm; with 10,000 paired jets, effects of a few thousandths are resolved. The 22 above 0.05 include JetClass top PN_full / PN_kin at several k, Top PN at k ≤ 5, the QG Herwig PN (kinpid) pu20 / c4_20 responses and ParT_kin top-vs-QCD e1.
- Slope_net − slope_EFP has a CI excluding 0 for every network. For QG it is −0.023 to −0.032. The QG networks also exceed the EFN slope by 0.008–0.014.

## Reading

**Hypotheses from the plan:**
- **H1, edit difference.** It predicted QG networks at LR(N) size for k ≥ 5 and e3, and JetClass/Top networks near 0 for e1.
  - QG networks have k-slopes of 0.89–1.00 × LR(N), but the IRC-safe EFN also has 0.79–0.83 × LR(N).
  - The JetClass W networks are not near 0 for e1 (−0.05 to −0.11).
  - Top networks are within ±0.03 for e1.
  - So H1 is not supported as stated.
- **H0, model difference.** It predicted QG networks within ±0.02 for all k and e3. This is excluded (k-slope −0.07; no slope CI lies within ±0.02). Its second part, JetClass/Top networks following LR(N)'s sign, holds for W vs QCD and hls4ml. It does not hold for either top task: networks are negative, LR(N) is positive.
- **Third outcome** (networks respond to the diffuse edit e3 only): excluded. All networks respond to realistic k additions with C2ST ≤ 0.62 for k ≤ 5.

**Separating the two parts of the response:**
- **Energy-flow part.** The k edit adds 1.6–3.2 GeV of pT per particle at soft-particle radii. The IRC-safe EFN and the count-blind EFP-only GBDT respond to it. In QG this part is most of the networks' slope.
- **Count part.** The particle count is changed with IRC-safe content about fixed (e1 on the data manifold; c4 off it).
  - QG networks: 0 to 0.4 × LR(N) per particle, the same result as H5v and H14.
  - JetClass W-vs-QCD networks: 1–4 × LR(N), above the conditional count reference EFPN (except ParT_kin under c4, −0.026).
  - Top Landscape: small except LorentzNet (−0.147 per particle under c4) and PFN (−0.042).
  - hls4ml: JEDI and Deep Sets match LR(N) for e1 (−0.04 vs −0.03) and are below it for c4 (−0.02 vs −0.09).
- D10's count response is therefore specific to some models and datasets (the JetClass networks; Top LorentzNet and PFN). It is not a consequence of the H13 edit.

**Scope and caveats:**
- c4 and pu are off the data manifold. The k edit changes the count together with energy flow.
- QG inputs use a recomputed axis (offset ≤ 2 mrad for k ≤ 20).
- hls4ml slot saturation reduces ΔN_seen.
- The opposite sign of the networks and LR(N) in the top tasks is not explained. EFPN, which conditions on the EFPs, is also negative for Top (−0.032).

**Proposed follow-up:**
- A count edit on the data manifold at fixed IRC-safe content and fixed ΣpT: k soft particles added with a matched removal of pT from the same radii.
- A count-only regression of each network's Δs on ΔN at fixed EFPs.
