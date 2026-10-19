# H4 — Top tagging: W-pair mass inside top at fixed jet mass and τ32, with positive controls (claim D6)

Script `interp/h4_top_wmass.py` (steps edit, classical, c2st, ef, nets, stats, c2stsub, fig). Results in `/j-jepa-vol/aaron-interp/results/physics/` and copied to `results/physics/`: `h4_results.json` (final), `h4_c2st.json`, `h4_c2stsub.json`, `h4_edit_info.json`, `h4_nets_info.json`, `h4_ef_info.json`, `h4_fig_wmass.png`. Interim results (ParT epoch 11, PN epoch 7) are kept in `h4_results_interim.json` and `h4_nets_info_interim.json`. The edited jets (`h4_edits.npz`, 420 MB) and the score arrays are on the volume only.
Jobs: d-h4-ctrl (edit, classical, c2st; CPU), d-h4-run (interim nets), d-h4-ef (EFN/PFN; CPU), d-h4-cls2 (classical with the τ21 controls added), d-h4-run3 (nets with final checkpoints; GPU), d-h4-fin (stats, c2stsub, fig; CPU). d-h4-run2 was deleted: its pod hit an image-pull failure.

## Conclusion
We changed the mass of the W-candidate subjet pair inside simulated top jets and held the total jet mass and the three-prong measure τ32 fixed. All five trained taggers (ParT, ParticleNet, LorentzNet, PFN, EFN) then lowered their top score. Moving a different subjet pair, or rotating the jet by the same amount, left the scores almost unchanged. For example, ParT's top-tag rate at 50 % signal efficiency fell from 69 % to 32 % when the W pair was moved to 60 GeV, against 65 % for the non-W control. Two positive controls, classifiers that see the W-pair mass (LR3 and MLP3), responded in the same direction and by similar amounts. The negative control, which sees only mass and τ32 (LR2), did not respond. So the networks do use the W-pair mass beyond total mass and τ32. This contradicts the D1 statement "no W-mass constraint" for these top-trained models; D1 was measured on JEDI-linear on hls4ml/JetClass. One caveat remains: a classifier can tell the edited jets from natural jets with the same (m, τ32, mW). The 80 → 60 GeV edit also lowers τ21 and the subjet masses. On the more natural-looking half of the 80 → 100 GeV edits, where that classifier is near chance (AUC 0.56), ParT and ParticleNet still respond, by about 0.8 of the response of the MLP3 positive control.

## Procedure
- **Sample.** `data_top.sample(5000, 601)`: 5,000 top and 5,000 QCD test jets (random rows) of the Top Landscape test split. For background rejection, an extra 100,000 random QCD test jets (`seed 602`).
- **Subjets and mW.** Three exclusive-kT subjets (E-scheme, all constituents). The W pair is the subjet pair whose mass is closest to 80.4 GeV, as defined in H2.
- **Eligible jets.** 105 < mSD < 210 GeV and at least 3 constituents.
  - Top jets also need 65 < mW < 95 GeV (3,592 jets).
  - QCD jets need 50 < mW < 70 or 90 < mW < 115 GeV (8,578 of 100k).
- **W⟨t⟩ edit.** The two W subjets are moved rigidly in (η, φ) about their pT-weighted centroid until the W-pair mass equals the target t. Targets: 60 and 100 GeV for top jets, 80.4 GeV for QCD jets.
  - The third subjet is moved along the line from the W centroid to restore the jet mass.
  - Every subjet's internal spread is scaled by a common factor h, solved so that τ32 returns to its original value.
  - Constituent pT and the subjet pT fractions are unchanged.
- **N⟨t⟩ control (non-W pair).** The pair (harder W daughter, third subjet) gets the same relative mass change m → m·t/mW0, through a 2-d shift of the third subjet. The jet mass is held and τ32 is restored through h; the W pair is not moved.
- **R⟨t⟩ control (rotation).** The whole jet is rotated about its axis by the angle that gives the same pT-weighted mean constituent displacement as the W edit.
- **Acceptance.** Recomputed after reclustering, on the edited jets.
  - W edit: |Δm| < 0.5 GeV, |Δτ32| < 0.02 and |mW − t| < 5 GeV.
  - N control: |Δm| < 0.5 GeV, |Δτ32| < 0.02 and |ΔmW| < 2 GeV.
  - The analysis set is the jets where both W and N are accepted (R is applied to the same jets).
  - Analysis sets: T60 556 / 3,248 eligible; T100 400 / 3,166; Q80 448 / 4,828.
- **Models.**
  - ParT (`part_kin`, final best epoch 16 of 20) and ParticleNet (`pnet_kin`, final best epoch 19 of 20): trained on Top Landscape kinematic features (H2 weaver jobs, finished).
  - LorentzNet: released top weights, with the authors' input path (two beam particles).
  - PFN and EFN: H2 energyflow models, finished. Run here by an exact numpy forward pass; the maximum |Δlogit| against the saved test probabilities is 3·10⁻⁶.
  - Positive controls: LR3 = LR(mSD, τ32, mW) and MLP3 = MLP(64, 64) on (mSD, τ32, mW).
  - Negative control: LR2 = LR(mSD, τ32).
  - Added τ21 controls: LR3t / MLP3t = LR / MLP on (mSD, τ32, τ21), with no mW.
  - All classical models are fitted on 100k validation jets.
- **Score.** Log-odds of top: logit difference for the networks, log likelihood ratio for the LRs.
- **Metrics.**
  - Δscore = mean over jets of (edited − unedited), with a 95 % bootstrap interval (2,000 resamples over jets).
  - Main contrast W − N, tested one-sided in the direction of the LR3 response (H0: W − N = 0; H1: W − N < 0 on top jets, > 0 on QCD), with a bootstrap p-value.
  - Holm correction over 10 models × 3 targets = 30 tests.
  - Exact McNemar test on the calls at ε_S = 0.3 and 0.5, using thresholds set on the 5,000 unedited tops.
  - Δ(1/ε_B) over the 100k QCD jets, with the edited rows substituted.
- **Off-manifold check (C2ST).** Edited top jets are compared with natural test top jets matched on (m, τ32, mW). Features: mSD, τ1–3, τ21, the smallest pair mass, the three-subjet mass, multiplicity, and the sorted pair masses, subjet pT fractions, subjet masses and pair ΔR. Classifier: gradient-boosted trees, 5-fold out-of-fold AUC, null = 10 label permutations. Follow-up (c2stsub): per-feature AUCs, and the W − N contrast re-measured on the half of edited jets with the lowest out-of-fold C2ST probability.

## Unedited models on the sample (10k jets; R from 5,000 tops + 100k QCD)
| model | AUC | acc | R30 [95 % CI] | R50 | published / own full test |
|---|---|---|---|---|---|
| ParT_kin (final) | 0.9860 | 94.17 % | 1493 [1205, 1887] | 394 | ParT 0.9858 / 1602 / 413 |
| ParticleNet_kin (final) | 0.9862 | 94.11 % | 1538 [1250, 1963] | 364 | PN 0.9858 / 1615 / 397 |
| LorentzNet (released) | 0.9871 | 94.56 % | 1754 [1369, 2441] | 476 | 0.9868 (paper), 0.9870 (released score file) / 2195 / 498 |
| PFN | 0.9806 | 92.82 % | 735 | 204 | own test 0.9801 / 748 / 200 |
| EFN | 0.9756 | 92.30 % | 515 | 136 | own test 0.9759 / 515 / 141 |
| LR2 / LR3 / MLP3 | 0.9545 / 0.9573 / 0.9604 | – / – / 89.5 % | 242 / 299 / 369 | 64 / 71 / 83 | H2: LR2 243, LR3 305 |
| LR3t / MLP3t | 0.9518 / 0.9580 | – / 89.3 % | 202 / 306 | 56 / 73 | – |

- ParT and ParticleNet R30 are within their 95 % intervals of the published values. LorentzNet's AUC reproduces the released score file.
- Rebuilding the jets through (pT, η, φ, m) changes 3 (ParT), 0 (PN) and 8 (LorentzNet) calls out of 10,000 compared with the stored arrays.
- Recomputed deta/dphi match the stored inputs to 5·10⁻⁷.

## What the edits do (median, analysis set)
| set | variant | jet m change | τ32 change | mW before → after | τ21 before → after | mean displacement D |
|---|---|---|---|---|---|---|
| T60 | W | 0.004 GeV | 0.002 | 79.2 → 60.0 | 0.51 → 0.40 | 0.056 |
| T60 | N | 0.001 | 0.002 | 79.2 → 79.0 | 0.51 → 0.46 | 0.149 |
| T60 | R | 0.48 | 0.000 | 79.2 → 79.2 | 0.51 → 0.51 | 0.056 |
| T100 | W | 0.004 | 0.002 | 85.3 → 100.0 | 0.50 → 0.50 | 0.045 |
| T100 | N | 0.001 | 0.001 | 85.3 → 85.2 | 0.50 → 0.49 | 0.080 |
| Q80 | W | 0.004 | 0.002 | 91.5 → 80.4 | 0.48 → 0.48 | 0.042 |
| Q80 | N | 0.001 | 0.002 | 91.5 → 91.3 | 0.48 → 0.47 | 0.048 |

The N control moves constituents 1.1–2.7 times further than the W edit (D column), so it is not smaller in angular size. The mSD change is ≤ 1.2 GeV for all variants.

## Numbers: Δ score (log-odds) — W − N [95 % CI], Holm p
| model | T60 (80 → 60) | T100 (80 → 100) | Q80 (QCD → 80.4) |
|---|---|---|---|
| ParT | −1.73 [−1.91, −1.55], 0.015 | −1.20 [−1.41, −1.00], 0.015 | +0.24 [+0.07, +0.42], 0.028 |
| ParticleNet | −1.80 [−1.99, −1.61], 0.015 | −1.38 [−1.61, −1.17], 0.015 | +0.24 [+0.08, +0.40], 0.016 |
| LorentzNet | −2.31 [−2.49, −2.11], 0.015 | −1.86 [−2.09, −1.65], 0.015 | +0.25 [+0.09, +0.41], 0.015 |
| PFN | −0.83 [−0.93, −0.73], 0.015 | −0.74 [−0.92, −0.56], 0.015 | +0.17 [+0.07, +0.29], 0.015 |
| EFN | −0.68 [−0.79, −0.56], 0.015 | −0.83 [−0.98, −0.68], 0.015 | +0.08 [−0.03, +0.19], 0.48 |
| LR2 (neg.) | +0.03 [−0.02, +0.07], 1 | +0.02 [−0.03, +0.09], 1 | −0.03 [−0.10, +0.03], 1 |
| LR3 (pos.) | −1.98 [−2.12, −1.83], 0.015 | −0.94 [−1.07, −0.81], 0.015 | +0.42 [+0.33, +0.51], 0.015 |
| MLP3 (pos.) | −1.52 [−1.59, −1.44], 0.015 | −0.97 [−1.05, −0.89], 0.015 | +0.38 [+0.30, +0.45], 0.015 |
| LR3t (τ21) | −0.29 [−0.38, −0.20], 0.015 | −0.24 [−0.36, −0.12], 0.015 | −0.03 [−0.14, +0.08], 1 |
| MLP3t (τ21) | −0.12 [−0.16, −0.08], 0.015 | −0.11 [−0.18, −0.04], 0.015 | −0.02 [−0.09, +0.05], 1 |

- The Holm floor is 0.015 because the smallest bootstrap p is 1/2001 and m = 30.
- Rotation control Δscore: |ΔR| ≤ 0.09 for every model and target.
- Score s.d. on the sample: ParT 6.2, PN 7.3, LNet 7.2, PFN 5.6, EFN 5.4, LR3 3.7.

## Numbers: calls at the working points (top jets; % passing; McNemar W vs N, exact)
| model | T60 ε_S = 0.5: before → W / N / R | T60 McNemar W vs N (N-pass→W-fail / reverse, p) | T100 ε_S = 0.5: before → W / N / R | T100 McNemar |
|---|---|---|---|---|
| ParT | 69.1 → 32.4 / 64.9 / 67.4 | 204 / 23, 2·10⁻³⁷ | 69.2 → 48.8 / 68.8 / 68.8 | 103 / 23, 3·10⁻¹³ |
| ParticleNet | 69.8 → 31.3 / 64.2 / 70.3 | 203 / 20, 3·10⁻³⁹ | 69.8 → 49.2 / 69.8 / 70.5 | 103 / 21, 4·10⁻¹⁴ |
| LorentzNet | 68.0 → 20.1 / 63.1 / 68.0 | 256 / 17, 6·10⁻⁵⁶ | 69.0 → 37.5 / 67.2 / 67.0 | 131 / 12, 2·10⁻²⁶ |
| PFN | 70.5 → 44.8 / 64.0 / 70.1 | 127 / 20, 3·10⁻²⁰ | 66.5 → 52.2 / 67.2 / 67.8 | 89 / 29, 3·10⁻⁸ |
| EFN | 71.8 → 56.1 / 66.9 / 70.5 | 96 / 36, 2·10⁻⁷ | 68.2 → 52.5 / 72.5 / 68.5 | 95 / 15, 2·10⁻¹⁵ |
| LR2 | 75.2 → 76.3 / 74.3 / 75.0 | 8 / 19, 0.05 | 80.2 → 79.5 / 79.5 / 80.5 | 12 / 12, 1 |
| LR3 | 75.9 → 25.0 / 75.4 / 75.2 | 282 / 2, 3·10⁻⁸¹ | 79.8 → 58.2 / 78.2 / 79.8 | 87 / 7, 10⁻¹⁸ |
| MLP3 | 74.8 → 36.9 / 74.6 / 74.5 | 211 / 1, 7·10⁻⁶² | 84.0 → 62.0 / 82.2 / 84.0 | 86 / 5, 4·10⁻²⁰ |
| MLP3t | 75.0 → 73.2 / 74.3 / 74.8 | 20 / 14, 0.39 | 80.5 → 78.2 / 79.8 / 80.5 | 21 / 15, 0.41 |

At ε_S = 0.3 the pattern is the same. Examples (T60): ParT 47.1 → 14.7 (W) / 46.4 (N); LR3 54.0 → 7.6 / 52.2; LR2 54.1 → 53.6 / 53.6.

**QCD (Q80; 448 edited jets out of 100,000).**
- Fraction of edited QCD jets passing ε_S = 0.5:
  - ParT 3.6 → 6.2 % (N 3.8 %), McNemar W vs N 8/19, p = 0.05.
  - LorentzNet 2.7 → 6.5 % (N 2.9 %), p = 0.005.
  - LR3 16.3 → 30.8 % (N 17.4 %), p = 2·10⁻¹⁰.
- 1/ε_B over all 100k QCD jets with the edited rows substituted, at ε_S = 0.3 (before → W / N):
  - ParT 1493 → 1351 / 1370; ParticleNet 1538 → 1389 / 1471; LorentzNet 1754 → 1538 / 1613; LR3 299 → 281 / 297.
  - Only 57–67 QCD jets pass this cut before the edit for the networks (N stated), and the edited jets are 0.45 % of the 100k. The Δ(1/ε_B) values are within the bootstrap intervals of the unedited R30 and are not resolved.
- At ε_B = 1 % (1,000 QCD jets passing), 1/ε_B is 100 → 99.0–99.6 (W) for the networks and 100 → 95.4 for LR3.

## Off-manifold check
**C2ST, top pool, AUC (permutation null mean / max).**

| target | identity | W edit | N control | R control |
|---|---|---|---|---|
| T60 | 0.644 (0.51 / 0.55) | 0.984 | 0.783 | 0.655 |
| T100 | 0.672 | 0.771 | 0.669 | 0.683 |

- The identity jets are not exactly matched (median match distance 0.2 in units of 2 GeV, 0.02 and 2 GeV), which sets a floor of about 0.65–0.67.
- The features that separate the W-edited jets most (per-feature AUC, edited vs natural) are the heaviest subjet mass (0.31 at T60, 0.37 at T100), τ21 (0.35, 0.39), τ2 and the middle subjet mass.
- At T60, τ21 is 0.51 (before) → 0.40 (edited), against 0.45 for natural tops with mW = 60 GeV. The h rescaling that restores τ32 shrinks the subjets.

**Natural-looking half (c2stsub).** The half of W-edited jets with the lowest C2ST probability. Its AUC against natural jets is 0.968 at T60 (n = 278) and 0.562 at T100 (n = 200).

| W − N on the natural-looking half [95 % CI] | T60 | T100 |
|---|---|---|
| ParT | −1.29 [−1.54, −1.03] | −0.66 [−0.91, −0.41] |
| ParticleNet | −1.19 [−1.42, −0.95] | −0.73 [−0.97, −0.48] |
| LorentzNet | −1.67 [−1.94, −1.41] | −1.26 [−1.52, −1.01] |
| PFN | −0.47 [−0.62, −0.33] | −0.09 [−0.27, +0.09] |
| EFN | −0.39 [−0.56, −0.23] | −0.40 [−0.58, −0.22] |
| LR3 / MLP3 (pos.) | −1.46 / −1.22 | −0.75 / −0.87 |
| LR3t / MLP3t (τ21) | −0.04 / −0.04 (CIs include 0) | −0.07 / +0.01 (CIs include 0) |
| LR2 (neg.) | +0.06 [+0.01, +0.12] | −0.00 [−0.08, +0.07] |

These sub-half p-values are not Holm-corrected; they are a follow-up to the main 30-test family.

## Hypotheses
- **H0 (D1 extended: mass + prongs only)** predicts W − N within the CI of 0, as for LR2. This is excluded for ParT, ParticleNet, LorentzNet, PFN and EFN on top jets at both targets (Holm p = 0.015, the floor). It is also excluded for ParT, ParticleNet, LorentzNet and PFN on QCD jets (Holm p ≤ 0.028). For EFN on QCD it is not excluded (Holm p = 0.48).
- **H1 (uses mW)** predicts a response in the LR3 direction, of the size of the positive controls.
  - Both positive controls respond, which validates the edit.
  - The network responses have the LR3 sign in all 15 model × target cells.
  - Relative to MLP3, the ParT / ParticleNet / LorentzNet responses are 1.1–1.9 times larger on top jets and 0.6–0.7 times on QCD jets.
- **Alternative (τ21 / subjet-mass shift).** The W edit at 60 GeV lowers τ21, and a τ21-using classifier responds. The MLP3t response (−0.12) is 7 % of the MLP3 response. On the natural-looking halves, where the τ21 controls are within the CI of 0, ParT, ParticleNet, LorentzNet and EFN still respond. The τ21 change explains a small part of the classical response. It is not excluded that the networks respond to other off-manifold features of the edit, such as the subjet masses.

## Reading
- With jet mass and τ32 held fixed, moving the W-candidate pair mass away from 80 GeV lowers every trained top tagger's score, and moving a QCD jet's best pair onto 80.4 GeV raises it. The size is comparable to or larger than that of classifiers built on mW.
- Moving a different subjet pair, which displaces constituents 1.1–2.7 times further, gives Δscore between −0.38 and +0.24 on top jets (W edit: −0.59 to −2.69). A rotation of equal displacement gives |Δscore| ≤ 0.09.
- LorentzNet has the largest response. EFN and PFN, which have no explicit subjet pairing, respond at about half the ParT size.
- EFN is IRC-safe and linear in the energy weights, so it was planned in H2 as a no-pairing control. It responds, so an energy-weighted angular sum can carry the W-pair mass information.
- The ParT / ParticleNet numbers use the final best-epoch checkpoints. In the interim run (epochs 11 / 7) the T60 W − N contrast was −1.18 / −1.09 (final −1.73 / −1.80).

## Issues and limits
- **Selection.** The analysis set is 12–17 % of eligible jets (both the W edit and the N control had to converge and pass acceptance). Jets with well-separated subjets are favoured.
- **Off-manifold edit.** The W edit at 60 GeV is strongly off-manifold (C2ST AUC 0.98); the 100 GeV edit much less so (0.77, and 0.56 on its natural-looking half).
- **Mass definitions.** The H2 mW is the exclusive-kT pair closest to 80.4 GeV. A network may define the W pair differently, for example by soft-drop subjets.
- **QCD statistics.** At the QCD working points the edited rows are a small fraction of the 100k sample, so Δ(1/ε_B) is not resolved.
- **Process.** `site/` is outside this loop's write scope, so the explainer text is given below for the ledger agent.

## Explainer entry (for site/explain.py, H4_EXPLAINER)
- **Question.** Inside a top jet, do trained taggers use the mass of the W-candidate subjet pair beyond the total mass and three-prong structure (claim D6)?
- **Physics background.** $t\to bW\to bq\bar q'$ gives three prongs with $m\approx m_t$ and one pair with $m_{jj}\approx m_W=80.4$ GeV. N-subjettiness $\tau_N=\frac{1}{R_0\sum_i p_{T,i}}\sum_i p_{T,i}\min_k\Delta R_{ik}$, so $\tau_{32}=\tau_3/\tau_2$ is small for three-prong jets. For three subjets, $m^2=m_{12}^2+m_{13}^2+m_{23}^2-\sum_k m_k^2$: one pair mass can change while $m$ is held, if the other pairs compensate.
- **Method.**
  - Three exclusive-kT subjets; W pair = the pair with mass closest to 80.4 GeV.
  - Edit: scale the W pair's opening about its axis until $m_W\to t$ ($t$ = 60 or 100 GeV for top jets, 80.4 GeV for QCD jets with $m_W$ away from 80). Move the third subjet to restore $m$ ($|\Delta m|<0.5$ GeV), and rescale the subjet sizes to restore $\tau_{32}$ ($|\Delta\tau_{32}|<0.02$).
  - Controls: the same relative mass change on a non-W pair ($m_W$ fixed), and a rotation of equal displacement.
  - Statistic: $\Delta_E=\langle s(E(j))-s(j)\rangle$ with $s$ = log-odds of top; $H_0:\Delta_W=\Delta_N$, one-sided bootstrap, Holm over 30 tests. McNemar on the calls at $\varepsilon_S=0.3,\,0.5$.
  - Positive controls: LR and MLP on $(m_{SD},\tau_{32},m_W)$. Negative control: LR on $(m_{SD},\tau_{32})$.
  - Off-manifold check: a classifier two-sample test against natural jets matched in $(m,\tau_{32},m_W)$.
- **Result (figure `results/physics/h4_fig_wmass.png`).**
  - Top jets, $80\to60$ GeV, at $\varepsilon_S=0.5$: ParT 69.1 → 32.4 % (non-W control 64.9 %); LorentzNet 68.0 → 20.1 % (63.1 %); MLP($m_W$) 74.8 → 36.9 % (74.6 %); LR($m_{SD},\tau_{32}$) 75.2 → 76.3 %.
  - In log-odds, $\Delta_W-\Delta_N$: ParT −1.73 [−1.91, −1.55], ParticleNet −1.80, LorentzNet −2.31, PFN −0.83, EFN −0.68, MLP3 −1.52, LR2 +0.03 [−0.02, +0.07].
  - QCD jets moved onto 80.4 GeV: ParT +0.24 [+0.07, +0.42].
- **Checks.**
  - Unedited AUC on the sample: ParT 0.9860, ParticleNet 0.9862, LorentzNet 0.9871 (released 0.9870), PFN 0.9806, EFN 0.9756.
  - The edited 60 GeV jets are distinguishable from natural ones (C2ST AUC 0.98), mainly through $\tau_{21}$ and the subjet masses. A $\tau_{21}$-using MLP responds at 7 % of the $m_W$ MLP.
  - On the half of the 100 GeV edits that look natural (C2ST AUC 0.56), ParT still responds (−0.66 [−0.91, −0.41]).
- **Physical significance.** Top taggers trained on Top Landscape check the internal W-pair mass, not only the total mass and three-prong shape. This differs from the hls4ml/JetClass JEDI-linear result (D1). The size of the response for the near-natural edits is about 0.8–1.7 times that of a classifier built on $m_W$.
