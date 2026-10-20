# H25 — Top tagging: τ32 (subjet width) at fixed jet mass and fixed W-pair mass (claim D6, new arm)

Script `interp/h25_top_tau32.py`. Results in `/j-jepa-vol/aaron-interp/results/physics/`, copied to `results/physics/`:
- `h25_top_tau32.json`: all numbers.
- `h25_tau32.png`: figure.
- `h25_perjet.npz`: per-jet scores and τ32; this file is on the volume only.

Jobs (GPU, `part_job.sh`):
- `d-h25-smoke`: pre-registered edit; debug attempt 1 (infeasible, see Procedure).
- `d-h25-smoke2` and `d-h25-smoke3`: the edit as run.
- `d-h25-run`: the full run, 17 min, exit 0.
- `d-h25-fig` and `d-h25-fig2` (CPU): figure redraws.

Logs are in `logs/d-h25-*.log`.

## Conclusion
We made the three subjets of each jet narrower or wider. The jet mass and the mass of the W-candidate subjet pair were held fixed, so the three-prong measure τ32 moved. All seven top taggers moved their top score in the direction that a classifier on (m, τ32, mW) predicts: wider subjets looked more like QCD, narrower subjets more like top. This held for the Top-Landscape ParT, ParticleNet, LorentzNet, EFN and PFN, and for JetClass ParT_kin and ParticleNet_kin, on both the Top Landscape and the JetClass jets. A rotation of the same size inside each subjet changed the scores much less. A likelihood ratio built on mass and W-pair mass alone did not change.

On Top Landscape tops at s = 1.4, the edit raises τ32 by about +0.10. It moved the networks by −0.7 to −1.5 in top log-odds, against −0.76 for the (m, τ32, mW) classifier. So the taggers use prong-shape information beyond m and mW.

The edit also changes the subjet masses, and a classifier can tell most edited jets from natural ones. These results therefore do not separate τ32 from subjet width or subjet mass. PFN's response is the weakest and is the least distinct from the side-effect reference.

## Procedure
**Samples.** Random rows, never first-k.
- **Top Landscape:** test split, 3,000 top + 3,000 QCD (`data_top.sample` rows, seed 9733).
- **JetClass:** test files `TTBar_100` (Tbqq) and `ZJetsToNuNu_100` (QCD). Stored 550 < pT < 650 GeV, |η| < 2, ≥ 3 particles; 3,000 per class drawn with rng 9733 / 9734 from 38,239 / 36,575 eligible rows.
- Every jet is first rebuilt through (pT, η, φ, m), the identity path used for all comparisons.

**Edit S(s), s ∈ {0.5, 0.7, 1.4, 2.0}.**
- Three exclusive-kT subjets (E-scheme).
- Every constituent's (η, φ) offset from its subjet's pT-weighted centroid is multiplied by s, so all angles within a subjet scale by s. Constituent pT and masses are kept.
- The subjets are then moved rigidly to restore the W-pair mass and the jet mass, using the H4 solver (`inner_W` with target = the jet's own mW):
  - The W pair is the subjet pair with mass closest to 80.4 GeV. It is opened about its pT-weighted centroid by a factor f.
  - The third subjet is moved along the line from that centroid by a factor g.
- **Acceptance**, recomputed after re-clustering: |Δm| < 0.5 GeV and |ΔmW| < 2 GeV.
- Every edited jet is rebuilt about its own constituent-sum axis for every network. For the JetClass nets this is the recentred jet dict from H19v.

**Debug attempt 1 (pre-registered version, not used).** The plan kept each subjet's four-momentum by reweighting the constituent pT inside the subjet (exponential tilt, w_i = exp(q_i·λ)). Smoke run `d-h25-smoke`, Top sample, 300 jets: 0/300 converged at s = 0.5, 1.4 and 2.0, and 42/300 at s = 0.7. At fixed subjet pT, the subjet mass fixes the pT-weighted angular second moment, so the width cannot be scaled by s at fixed subjet four-momentum. The version run keeps m and mW and lets the subjet masses change. This side effect is covered by the MLPx reference below.

**Controls** (same jets; same m/mW restore and acceptance):
- **c1:** each subjet's constituents rotated about its centroid (random sign per subjet). The angle is chosen so that the pT-weighted mean constituent displacement equals that of S on the same jet; the median displacement matches S to 10⁻⁴. Angles within each subjet are kept, so τ32 stays almost fixed.
- **c2:** the same scaling by s, about 3 random constituents instead of the kT subjet centroids. The centres are drawn with probability ∝ pT, and each constituent is assigned to the nearest centre.
- **c3 (TN):** the whole jet translated rigidly in (η, φ) by the S edit's mean displacement, in a random direction.
- The analysis set for each s is the set of jets where both S and c1 are accepted. c2 is reported on its accepted subset of those jets.

**Models.**
- Top Landscape: ParT_kin and PN_kin (H4 checkpoint snapshots), LorentzNet (released), PFN and EFN (H2).
- JetClass: ParT_kin and PN_kin (released). The score is L(Tbqq) − L(QCD).
- Every network is run on both samples. JetClass jets are passed to the Top nets as massless four-vectors, as in H19.

**References.** Fitted per dataset on disjoint jets: Top val split, 25,000 per class (rng 4194); JetClass `TTBar_101` / `ZJetsToNuNu_101` in-window jets, 25,000 per class.
- **LR3:** binned LR(m, τ32, mW), with 15 × 20 × 10 bins (positive control).
- **MLP3:** MLP(64, 64) on (m, τ32, mW) (positive control, smooth).
- **LR2:** LR(m, mW) (negative control).
- **MLPx:** MLP on the edit's side effects without any N-subjettiness. Inputs: m, mW, mSD, sorted pair masses, subjet masses, subjet pT fractions, pair ΔR, smallest pair mass, 3-subjet mass, multiplicity.
- **MLPxT:** MLPx plus τ1–3, τ21 and τ32.

**Metrics.**
- **Effect:** S − c1 = mean over jets of (score_S − score_c1), in top log-odds, with a 95 % bootstrap CI (B = 2,000 over jets).
- **Tests:**
  - H0 (S − c1 = 0): two-sided sign-flip permutation (5,000 permutations).
  - H0_side (network S − c1 = MLPx S − c1, per jet): sign-flip permutation.
  - Holm correction over 112 tests (7 networks × 4 s × 2 samples × 2 classes) for each family. The Holm floor is 112/5,001 = 0.022.
- **Ratios** to MLP3, LR3 and MLPx, with a paired bootstrap CI.
- **Δτ32 achieved.**
- **Calls:** % of calls changed at ε_S = 0.3 and 0.5 (thresholds from the sample's unedited tops), and McNemar S vs c1.
- **C2ST:** edited jets vs natural reference jets of the same class, matched (nearest neighbour) in standardized (m, τ32, mW). HGB on 20 substructure features, 2-fold AUC, up to 1,500 jets. The identity row gives the floor of the matching.
- **Natural half:** S − c1 on the half of the S jets that the C2ST scores as most natural.

## Checks: unedited models on these samples (6,000 jets each)
| model | Top sample AUC / acc. | JetClass sample AUC / acc. | known |
|---|---|---|---|
| Top ParT | 0.9867 / 94.3 % | 0.9867 / 94.3 % | 0.9858 published; 0.9868 (H19v) |
| Top PN | 0.9868 / 94.0 % | 0.9844 / 93.6 % | 0.9858 published |
| Top LorentzNet | 0.9879 / 94.5 % | 0.9881 / 94.5 % | 0.9868 published; 0.9878 (H19v) |
| Top PFN | 0.9816 / 93.0 % | 0.9792 / 92.4 % | – |
| Top EFN | 0.9783 / 92.6 % | 0.9793 / 93.0 % | – |
| JetClass ParT_kin | 0.9830 / 92.7 % | 0.9930 / 95.7 % | 0.9932 / 95.8 % (H19v, JetClass) |
| JetClass PN_kin | 0.9847 / 93.5 % | 0.9897 / 94.9 % | 0.9901 / 95.2 % (H19v) |
| LR3 / MLP3 / LR2 / MLPx | 0.966 / 0.967 / 0.946 / 0.982 | 0.974 / 0.977 / 0.957 / 0.982 | – |

- Rejection with 3,000 QCD jets is not resolved for the networks: 1–17 QCD jets pass at ε_S = 0.3 or 0.5. For example, Top ParT 1/ε_B = 750 at ε_S = 0.3 with n = 4 passing, and 375 at ε_S = 0.5 with n = 8. All values are in the JSON (`unedited.*.wp`).
- The translation control c3 moves every network by |Δ| ≤ 0.12 log-odds (LorentzNet on QCD) and all others by ≤ 0.07. So the rebuild about each jet's own axis is in place, and a rigid displacement of this size does not move the scores.

## What the edit does (analysis set, top jets; QCD jets in brackets)
| s | n (top / QCD), Top sample | n, JetClass sample | Δτ32 S (mean) | Δτ32 c1 | Δτ32 c2 | Δ subjet-mass sum S (median) | ΔmSD S (median) | C2ST AUC S / c1 / identity (Top tops) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1,708 / 1,307 | 1,905 / 1,490 | −0.196 (−0.247) | −0.003 (−0.031) | −0.139 | −26.5 GeV | +0.5 (+8.2) GeV | 1.00 / 0.58 / 0.54 |
| 0.7 | 1,875 / 1,460 | 2,120 / 1,635 | −0.110 (−0.135) | −0.002 (−0.014) | −0.080 | −16.2 GeV | +0.6 (+5.3) GeV | 0.90 / 0.53 / 0.51 |
| 1.4 | 625 / 220 | 910 / 338 | +0.097 (+0.085) | −0.003 (−0.019) | +0.089 | +16.1 GeV | −2.6 (−3.6) GeV | 0.81 / 0.55 / 0.54 |
| 2.0 | 172 / 34 | 389 / 82 | +0.192 (+0.086) | −0.005 (−0.044) | +0.196 | +33.2 GeV | −8.3 (−3.7) GeV | 0.95 / 0.61 / 0.55 |

- **Δm and ΔmW (S).** Median |Δm| ≤ 0.004 GeV and median |ΔmW| ≤ 0.9 GeV for S; for c1, ≤ 0.73 GeV.
- **Δτ21 (S).** −0.07 (s = 0.5) to +0.08 (s = 2.0) on tops, and up to −0.15 on QCD.
- **Acceptance.**
  - S is accepted for 92–95 % of jets at s ≤ 0.7, 19–26 % at s = 1.4 and 5–10 % at s = 2.0. The rigid restore often fails to bring mW back once the subjets overlap.
  - c1 is accepted for 54–65 % (the mW change after re-clustering is > 2 GeV).
- **C2ST.**
  - JetClass tops: 0.98 / 0.84 / 0.74 / 0.90 for s = 0.5 / 0.7 / 1.4 / 2.0 (identity 0.51–0.60).
  - QCD at s = 1.4: 0.57 (Top) and 0.56 (JetClass), against identity 0.64 / 0.62.
  - Top QCD at s = 2.0 (n = 34) is not informative (all variants 0.50).

## Numbers: S − c1 (top log-odds) [95 % CI]; × MLP3 = ratio to the MLP(m, τ32, mW) response
Top Landscape sample, top jets:

| model | s = 0.5 | s = 0.7 | s = 1.4 | s = 2.0 |
|---|---|---|---|---|
| Top ParT | +0.98 [+0.90, +1.06], × 0.75 | +0.62 [+0.57, +0.67], × 0.79 | −1.14 [−1.24, −1.05], × 1.50 | −2.54 [−2.82, −2.26], × 1.73 |
| Top PN | +2.01 [+1.92, +2.10], × 1.54 | +0.98 [+0.92, +1.04], × 1.25 | −1.48 [−1.59, −1.38], × 1.94 | −3.56 [−3.88, −3.24], × 2.43 |
| Top LorentzNet | +1.68 [+1.57, +1.79], × 1.29 | +0.79 [+0.73, +0.85], × 1.01 | −1.23 [−1.33, −1.13], × 1.62 | −2.61 [−2.94, −2.29], × 1.78 |
| Top PFN | +0.18 [+0.14, +0.22], × 0.14 | +0.17 [+0.15, +0.20], × 0.22 | −0.67 [−0.73, −0.60], × 0.88 | −1.87 [−2.11, −1.63], × 1.28 |
| Top EFN | +1.55 [+1.49, +1.61], × 1.19 | +1.04 [+1.00, +1.08], × 1.32 | −1.43 [−1.50, −1.36], × 1.88 | −3.22 [−3.47, −2.96], × 2.19 |
| JetClass ParT_kin | +0.65 [+0.56, +0.73], × 0.50 | +0.57 [+0.52, +0.62], × 0.72 | −1.16 [−1.25, −1.08], × 1.53 | −2.24 [−2.48, −2.00], × 1.53 |
| JetClass PN_kin | +1.41 [+1.33, +1.48], × 1.08 | +0.78 [+0.74, +0.82], × 0.99 | −1.04 [−1.12, −0.96], × 1.36 | −2.02 [−2.23, −1.79], × 1.37 |
| MLP3 (pos.) | +1.30 [+1.28, +1.33] | +0.79 [+0.77, +0.80] | −0.76 [−0.80, −0.73] | −1.47 [−1.58, −1.36] |
| LR3 (pos., binned) | +0.57 [+0.48, +0.66] | +0.64 [+0.60, +0.69] | −0.80 [−0.87, −0.74] | −1.61 [−1.80, −1.42] |
| MLPx (side effects) | +0.48 [+0.43, +0.53] | +0.32 [+0.29, +0.35] | −0.62 [−0.67, −0.56] | −1.55 [−1.70, −1.39] |
| MLPxT | −0.08 [−0.13, −0.03] | +0.01 [−0.02, +0.04] | −0.25 [−0.31, −0.20] | −0.83 [−0.97, −0.67] |
| LR2 (neg.) | +0.01 [−0.00, +0.01] | +0.00 [−0.00, +0.01] | −0.01 [−0.02, +0.00] | +0.00 [−0.01, +0.01] |

JetClass sample, top jets (the same columns; full table in the JSON):
- **Top PN:** +3.06 / +1.58 / −1.83 / −3.71.
- **Top LorentzNet:** +2.98 / +1.77 / −1.85 / −3.56.
- **Top ParT:** +1.04 / +0.68 / −1.14 / −2.56.
- **JetClass ParT_kin:** +0.47 / +0.60 / −1.38 / −3.04.
- **JetClass PN_kin:** +1.62 / +1.01 / −1.43 / −3.02.
- **Top EFN:** +1.25 / +0.82 / −1.20 / −2.72.
- **Top PFN:** +0.20 / +0.16 / −0.49 / −1.15.
- **References:** MLP3 +1.13 / +0.68 / −0.75 / −1.55; MLPx +0.23 / +0.22 / −0.43 / −1.06; LR2 0.00 throughout.

QCD jets (both samples):
- The networks move top-ward for s < 1 and QCD-ward for s > 1. MLP3 gives +1.28 / +0.74 / −0.65 / −0.78 (Top sample) and +1.02 / +0.62 / −0.45 / −1.02 (JetClass sample).
- For the networks, Top sample, s = 0.5: Top PN +1.80 [+1.69, +1.91], JetClass ParT_kin +0.93 [+0.84, +1.02], PFN +0.46 [+0.41, +0.50].
- For the networks, Top sample, s = 1.4: Top PN −0.64 [−0.84, −0.44], JetClass PN_kin −1.05 [−1.23, −0.87].

**Summary over the 112 network tests.**
- **H0 test.** Holm p < 0.05 for 109 of the 112 tests. The 3 that do not pass are Top ParT, LorentzNet and PFN on Top QCD at s = 2.0 (n = 34).
- **Sign.** 111 of the 112 have the sign of the H1 prediction (PFN, Top QCD, s = 2.0: +0.09 [−0.34, +0.51]).
- **Ratio to MLP3.**

  | model | range |
  |---|---|
  | Top ParT | 0.59–1.73 |
  | Top PN | 0.98–2.70 |
  | LorentzNet | 0.42–2.63 |
  | EFN | 0.55–2.19 |
  | JetClass ParT_kin | 0.42–1.96 |
  | JetClass PN_kin | 0.99–1.98 |
  | PFN | −0.11–1.28 |

  The ratio grows with |s − 1| for tops.
- **vs MLPx (H0_side).** S − c1 exceeds the MLPx response with Holm p < 0.05 in 90 of the 112 tests. Of the 22 that do not, 10 are PFN and 9 are QCD at s ≥ 1.4 with n ≤ 338.

**Controls.**
- **c1 (rotation).** Networks Δ_c1 = −0.23 to +0.03 on tops and +0.01 to +0.43 on QCD for s ≤ 1.4. At s = 2.0 the range is −0.77 to +0.45.
- **c2 (random centres).** It moves τ32 by 71–102 % as much as S does (tops) and moves the networks by 65–105 % as much as S. c2 is therefore a second τ32-changing edit, not a null.
- **Natural half** of the S jets: S − c1 within about ±0.4 of the full-set value. Examples: Top PN at s = 0.5 on Top tops +1.91 vs +2.01; ParT at s = 1.4 −1.10 vs −1.14. On this half, C2ST for s ≤ 0.7 is still high, so this is not an on-manifold subset.

**Calls at ε_S = 0.3, top jets, Top sample** (% passing: before → S / c1; McNemar S vs c1, c1-pass→S-fail / reverse):

| model | s = 0.7 | s = 1.4 |
|---|---|---|
| Top ParT | 36 → 48 / 35 (27 / 274, p = 10⁻⁵²) | 52 → 32 / 52 (134 / 11, p = 5·10⁻²⁸) |
| Top PN | 36 → 50 / 34 (24 / 325) | 53 → 29 / 52 (149 / 8) |
| JetClass ParT_kin | 35 → 48 / 34 (63 / 331) | 51 → 25 / 50 (161 / 5) |
| MLP3 | 36 → 61 / 36 (2 / 484) | 53 → 28 / 54 (169 / 5) |
| LR2 | 44 → 44 / 43 (18 / 28, p = 0.2) | 49 → 48 / 49 (19 / 16, p = 0.7) |

- Calls changed by S vs by c1 at s = 2.0: 47–62 % vs 9–20 %.
- QCD jets mostly stay below the ε_S = 0.3 threshold: S changes 0–2.4 % of QCD calls.

## Hypotheses
| hypothesis | prediction | observed | status |
|---|---|---|---|
| H0: m and mW suffice (D1-like) | S − c1 ≈ 0, like LR2 | S − c1 ≠ 0 in 109 / 112 tests (Holm < 0.05); LR2 within ±0.06 | excluded |
| H1: τ32 used | tops QCD-ward for s > 1, QCD top-ward for s < 1; size ≈ τ32 reference; beyond c1 | sign as predicted in 111 / 112; size 0.4–2.7 × MLP3 for six networks (PFN 0.14–1.28 ×); c1 smaller | consistent |
| Third: subjet width read directly (≫ τ32 LR) | response ≫ MLP3 / LR3 | at most 2.7 × MLP3 (Top PN, s = 2.0; Top PN and LorentzNet on JetClass tops, s = 0.5–0.7); grows with |s − 1| | not separable: subjet masses also change, and edited jets are off-manifold (C2ST 0.74–1.00) |
| Side effects only | network ≈ MLPx | six networks 1.1–14.7 × MLPx (lowest: EFN, JetClass QCD, s = 0.7), PFN −0.8–1.5 ×; 90 / 112 above MLPx at Holm < 0.05 | excluded for six networks; open for PFN |

## Reading (scoped to what was measured)
- **Direction.** For each of the seven networks, on both samples, the S − c1 response has the sign of the change that the (m, τ32, mW) references predict. Its size is of the same order as those references.
- **c2.** The same scaling about random centres moves both τ32 and the scores by 65–105 % of the S values, so the response follows the change in the angular spread of the constituents rather than the exact prong axes.
- **Side effects.** The edit is not τ32 alone. Subjet masses change by −27 to +33 GeV (median sum), τ21 changes, and mSD changes by up to 8 GeV on QCD. A classifier on those side effects without N-subjettiness (MLPx) also responds, at 0.2–1.6 log-odds. The networks' S − c1 is larger than MLPx's for the six non-PFN networks.
- **MLPxT** (side effects plus all τ's) responds weakly at s < 1 on tops (−0.08 to +0.53). That is the one reference whose prediction is not close to the networks'.
- **Cross-dataset.** The JetClass-trained ParT_kin and PN_kin and the Top-trained networks give responses of the same sign and similar size on both samples.

## Issues
- The pre-registered edit (subjet four-momenta kept) is infeasible (debug attempt 1). The edit as run keeps m and mW but changes the subjet masses.
- **Acceptance.** It is low for s > 1 (S 5–26 %) and for c1 (54–65 %), so the analysis sets at s ≥ 1.4 are selected subsets (n = 34–910).
- **Off-manifold.** Edited top jets are off-manifold (C2ST 0.74–1.00, identity floor 0.51–0.60). The natural-half subset is still distinguishable, so it does not give an on-manifold check.
- **LR3.** The binned LR3 is non-monotone at s = 0.5 on tops (+0.57 vs MLP3 +1.30): its sparse low-τ32 bins fall back to the prior. MLP3 is used as the main positive reference.
- **Rejection.** Background rejection is not resolved with 3,000 QCD jets per sample.
- **Top QCD at s = 2.0** has n = 34, and its C2ST is uninformative.

## Explainer entry (for site/explain.py; to be merged by the ledger)
```python
H25_EXPLAINER = [
 ('Question', 'Top taggers use the W-candidate pair mass inside top at fixed jet mass and $\\tau_{32}$ (H4, D6). Do they also use the three-prong shape, $\\tau_{32}$, at fixed jet mass and fixed W-pair mass?', None),
 ('Physics background',
  'N-subjettiness $\\tau_N=\\frac{1}{\\sum_i p_{T,i}R_0}\\sum_i p_{T,i}\\min_k\\Delta R_{i,k}$ measures how well $N$ axes describe the radiation; $\\tau_{32}=\\tau_3/\\tau_2$ is small for three-prong tops and near 1 for QCD. '
  'For a jet of three narrow subjets, $\\tau_3$ is set by the subjet widths, so making every subjet narrower lowers $\\tau_{32}$. The jet mass $m^2=\\sum_{a<b}m_{ab}^2-\\sum_k m_k^2$ (three subjets) and the W-pair mass $m_{ab}$ depend on the subjet four-vectors, which can be restored by moving the subjets apart.', None),
 ('Method',
  'Three exclusive-$k_T$ subjets; every constituent offset from its subjet centroid scaled by $s\\in\\{0.5,0.7,1.4,2.0\\}$; subjets then moved rigidly so that $m_W$ and $m$ return to their values ($|\\Delta m|<0.5$ GeV, $|\\Delta m_W|<2$ GeV). Controls: c1 rotation inside each subjet with the same constituent displacement ($\\tau_{32}$ kept), c2 the same scaling about 3 random constituents, c3 rigid translation. '
  'Samples: Top Landscape and JetClass ($p_T$ 550–650 GeV), 3,000 top + 3,000 QCD each (seed 9733). Models: Top ParT, PN, LorentzNet, PFN, EFN; JetClass ParT_kin, PN_kin; all on both samples, each jet rebuilt about its own axis. References: MLP3 / LR3 on $(m,\\tau_{32},m_W)$, LR2 on $(m,m_W)$, MLPx on the side effects without $\\tau$. Metric $S-c_1$ in top log-odds, bootstrap CI, sign-flip permutation, Holm over 112 tests.', None),
 ('Result',
  'Achieved $\\Delta\\tau_{32}$ on tops: $-0.20,-0.11,+0.10,+0.19$ for $s=0.5,0.7,1.4,2.0$ (c1: $|\\Delta\\tau_{32}|\\le0.005$). All seven networks move in the direction the $\\tau_{32}$ references predict (111 / 112 tests; 109 at Holm $p<0.05$); LR2 stays at 0. Top Landscape tops at $s=1.4$: ParT $-1.14\\,[-1.24,-1.05]$, PN $-1.48$, LorentzNet $-1.23$, EFN $-1.43$, PFN $-0.67$, JetClass ParT_kin $-1.16$, vs MLP3 $-0.76$ and MLPx $-0.62$. Ratio to MLP3 0.4–2.7 (PFN 0.14–1.28), growing with $|s-1|$.', 'results/physics/h25_tau32.png'),
 ('Checks',
  'Unedited AUC reproduce known values (Top ParT 0.9867, LorentzNet 0.9879, JetClass ParT_kin 0.9930 on JetClass). Translation c3 $|\\Delta|\\le0.12$. Random-centre scaling c2 moves $\\tau_{32}$ and the scores by 65–105 % of S. The pre-registered edit that keeps each subjet four-vector is infeasible (0 / 300 converged): a subjet mass at fixed $p_T$ fixes its angular second moment. The edit run here changes the subjet masses; edited jets are off-manifold (C2ST 0.74–1.00 on tops; 0.56–0.57 on QCD at $s=1.4$).', None),
 ('Physical significance',
  'Top taggers trained on either dataset use the prong shape (subjet width / $\\tau_{32}$) in addition to the jet mass and the W-pair mass, with responses comparable to a classifier that sees $\\tau_{32}$. Whether they read $\\tau_{32}$, the subjet widths or the subjet masses is not separated by this edit.', None),
]
```
Entry line for the `'d'` list: `('Result explained: top taggers and the prong shape τ32 at fixed jet mass and W-pair mass (claim D6; H25)', H25_EXPLAINER, False)`.
