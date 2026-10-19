# H4 verify — W-pair mass inside top (claim D6)

Verifier script `interp/h4_verify.py`. It imports the H4 subjet, edit, control, observable and network functions unchanged from `interp/h4_top_wmass.py`. Job `d-h4-verify` (GPU pod, 4 CPU; 13 min; exit 0). Results: `results/physics/h4_verify.json` (also on the volume); log `logs/d-h4-verify.log`.

## Conclusion
On a fresh set of top jets, we again moved the W-candidate subjet-pair mass while holding jet mass and τ32 fixed. ParT, ParticleNet and LorentzNet lowered their top score by about as much as a classifier given the W-pair mass. That is 2 to 4 times what a classifier gets from the edit's side effects alone (smaller subjets, lower τ21) when it is given no pair mass. So the evidence that these three networks use the W-pair mass holds up. For PFN and EFN, the response is the same size as the side-effect-only classifier's, so this test does not show that they use the W-pair mass. The test also uses a single side-effect classifier as its reference, which is a limit on how far it can be pushed.

## Code check (`interp/h4_top_wmass.py`)
- **Rows.** Rows are drawn with random `rng.choice` per class (`_sample`), not first-k.
- **Padding.** Padded particles are excluded:
  - `nn = min(jet_nparticles, L)` is used everywhere.
  - `jet_inputs` masks the padded slots.
  - LorentzNet uses `am = E != 0`.
  - EFN/PFN use `pt > 0`.
- **Axes and network inputs.** The jet axis is recomputed from the edited constituents, as in the Top Landscape conversion (deta sign flip included). The stored-vs-rebuilt check gives max |Δdeta| 5·10⁻⁷.
- **Same jets.** Observables and scores are computed on the same edited arrays.
- **Bootstrap.** W, N and R use the same resample indices (paired). The Holm floor is right (1/2001 × 30 = 0.015).
- **Working points.** Thresholds at ε_S = 0.3 / 0.5 are set on the 5,000 unedited tops of the same sample. This is correct.
- **What the edit holds fixed.** The edit claims to hold m and τ32, and both are verified after reclustering (|Δm| median 0.004 GeV, |Δτ32| 0.002). It does not hold the subjet masses. The module docstring says "subjet masses do not change", but the τ32-restoring factor h scales each subjet's internal spread:
  - T60: h median is 0.78 for the W edit and 0.97 for the N control.
  - The summed subjet mass goes 50.6 → 39.7 GeV under W and → 47.9 GeV under N.
  - The report states this in its caveat, but the docstring is wrong.
- **Mislabelled count.** The report's "eligible" counts (3,248 / 3,166 / 4,828) are the jets for which W or N converged. The eligible counts are 3,592 (top) and 8,578 (QCD), as in the edit log.
- No bug was found that changes the numbers.

## Rerun on a fresh sample
- **Sample.** 5,000 top + 5,000 QCD test jets (seed 611) and 90,000 QCD test jets (seed 612). All are random rows of the Top Landscape test split with every H4 row (seeds 601 / 602) excluded; overlap 0 / 0. Only about 92k QCD test rows remain after the exclusion, hence 90k.
- **Models.** The same model snapshots: ParT / PN `h4_ckpt_*.pt`, the same md5 as H4; released LorentzNet; PFN / EFN final.
- **Acceptance.** As in H4.
- **Unedited models on this sample (10k).**

| model | AUC | accuracy |
|---|---|---|
| ParT | 0.9846 | 93.6 % |
| ParticleNet | 0.9842 | 93.8 % |
| LorentzNet | 0.9854 | 94.0 % |
| PFN | 0.9785 | 92.7 % |
| EFN | 0.9741 | 91.8 % |
| LR2 / LR3 / MLP3 | 0.953 / 0.954 / 0.959 | – |
| MLPx (new) | 0.9756 | 92.1 % |
| MLPxW (new) | 0.9770 | 92.2 % |

### Edits and controls run
- **T60 / T100** (replication). Top jets with 65 < mW < 95; the W pair is moved to 60 or 100 GeV.
  - Analysis sets: n = 545 and 398 (H4: 556 / 400).
- **T80** (new). Top jets with 6 < |mW − 80.4| < 15 GeV; the W pair is moved onto 80.4 GeV (|mW − 80.4| < 2 after the edit).
  - Analysis set: n = 416; mW median 88.4 → 80.4.
  - An mW-window reading predicts a rise. A reading where any deformation of a top jet lowers the score predicts a fall.
- **Q80** (replication), n = 374, split by direction:
  - Q80up: QCD jets with mW 50–70, pair opened to 80.4 GeV; n = 120.
  - Q80dn: QCD jets with mW 90–115, pair closed to 80.4 GeV; n = 254.
  - A window reading predicts a rise in both directions. A monotone dependence on the pair opening predicts opposite signs.
- **MLPx** (new side-effect control). MLP(64, 64) on (mSD, τ1, τ2, τ3, τ32, τ21, multiplicity, sorted subjet pT fractions, sorted subjet masses). It has no pair mass and no pair angle. It is fitted on 100,000 random val jets (seed 613).
  - It sees what the edit changes besides mW (subjet sizes and masses, τ21), but not mW.
- **MLPxW.** The same features plus mW: a positive control on the same feature set.
- **Statistics.** Metric is W − N Δscore (log-odds) with a paired 2,000-resample bootstrap 95 % CI.
  - H0: W − N = 0. H1: W − N in the mW-window direction (negative for T60 and T100; positive for T80, Q80up and Q80dn). One-sided bootstrap p.
  - Holm correction over 5 sets × 10 models = 50 tests; the floor is 0.025.

### W − N Δscore [95 % CI] (Holm p)
| model | T60 | T100 | T80 (onto 80.4) | Q80 all | Q80up | Q80dn |
|---|---|---|---|---|---|---|
| ParT | −1.79 [−1.98, −1.60] (0.025) | −1.14 [−1.35, −0.93] (0.025) | −0.04 [−0.19, +0.10] (1) | +0.40 [+0.23, +0.60] | +0.18 [−0.13, +0.51] (1) | +0.50 [+0.25, +0.75] (0.025) |
| ParticleNet | −2.04 [−2.24, −1.84] (0.025) | −1.28 [−1.50, −1.05] (0.025) | +0.01 [−0.13, +0.13] (1) | +0.35 [+0.19, +0.53] | −0.04 [−0.34, +0.27] (1) | +0.53 [+0.32, +0.75] (0.025) |
| LorentzNet | −2.58 [−2.79, −2.38] (0.025) | −1.63 [−1.85, −1.41] (0.025) | −0.07 [−0.21, +0.09] (1) | +0.47 [+0.30, +0.65] | +0.20 [−0.06, +0.48] (1) | +0.60 [+0.37, +0.83] (0.025) |
| PFN | −0.77 [−0.87, −0.67] (0.025) | −0.61 [−0.79, −0.44] (0.025) | −0.10 [−0.19, −0.01] (1) | +0.20 [+0.07, +0.34] | −0.14 [−0.36, +0.07] (1) | +0.36 [+0.19, +0.53] (0.025) |
| EFN | −0.63 [−0.74, −0.51] (0.025) | −0.66 [−0.83, −0.51] (0.025) | −0.05 [−0.15, +0.04] (1) | +0.09 [−0.04, +0.22] | −0.39 [−0.59, −0.18] (1) | +0.32 [+0.16, +0.48] (0.025) |
| LR2 (neg.) | −0.02 [−0.07, +0.04] (1) | +0.06 [−0.00, +0.13] (1) | −0.07 [−0.13, −0.01] (1) | −0.10 [−0.21, −0.00] | −0.03 (1) | −0.13 (1) |
| LR3 (pos.) | −2.18 (0.025) | −0.94 (0.025) | +0.21 [+0.12, +0.30] (0.025) | +0.43 | +0.50 (0.025) | +0.40 (0.025) |
| MLP3 (pos.) | −1.65 (0.025) | −0.96 (0.025) | +0.19 [+0.13, +0.25] (0.025) | +0.40 | +0.54 (0.025) | +0.33 (0.025) |
| **MLPx** (no mW) | −0.63 [−0.71, −0.55] (0.025) | −0.54 [−0.69, −0.40] (0.025) | −0.19 [−0.24, −0.12] (1) | +0.15 [+0.02, +0.29] | −0.05 (1) | +0.25 [+0.07, +0.43] (0.076) |
| **MLPxW** (with mW) | −1.89 [−1.98, −1.80] (0.025) | −1.11 [−1.29, −0.93] (0.025) | −0.06 [−0.13, +0.01] (1) | +0.54 [+0.39, +0.70] | +0.41 (0.025) | +0.60 (0.025) |

Q80 all is not in the Holm family; its two halves are.

### Calls at ε_S = 0.5, T60 (before → W / N; McNemar W vs N)
| model | before → W / N | N-pass→W-fail / reverse | p |
|---|---|---|---|
| ParT | 72.8 → 36.9 / 65.1 % | 189 / 35 | 10⁻²⁶ |
| LorentzNet | 73.4 → 20.6 / 66.2 % | 269 / 20 | 10⁻⁵⁶ |
| MLPx | 75.8 → 58.0 / 69.2 % | 87 / 26 | 7·10⁻⁹ |
| MLP3 | 78.9 → 37.2 / 79.4 % | 230 / 0 | 10⁻⁶⁹ |

### Per-jet dose response
Within each W-edit set, Δscore(W) was regressed on three terms: Δ|mW − 80.4|, Δ(sum of subjet masses) and Δτ21, with 300 bootstrap fits. β₁ is the Δ|mW − 80.4| coefficient, in log-odds per GeV.

| model | T60 β₁ | T100 β₁ |
|---|---|---|
| ParT | −0.085 [−0.115, −0.044] | −0.099 [−0.131, −0.070] |
| ParticleNet | −0.095 | −0.103 |
| LorentzNet | −0.124 | −0.121 |
| PFN | −0.022 [−0.038, −0.004] | −0.054 |
| EFN | −0.053 | −0.058 |
| MLP3 | −0.095 | −0.063 |
| MLPx | −0.003 [−0.020, +0.014] | −0.027 [−0.041, −0.014] |

The three regressors are correlated, so β₁ is supporting evidence only.

## Competing hypotheses and what the data exclude
- **H0: mass + τ32 only** (LR2 pattern).
  - Excluded for all five networks at T60 and T100.
  - For QCD, excluded for all five in Q80dn.
- **Hs: the networks respond to the side effects of the edit** (subjet shrinkage, τ21), not to mW. The prediction is the MLPx pattern: about −0.6 / −0.5 / −0.2 / +0.15 on T60 / T100 / T80 / Q80.
  - Excluded for ParT, ParticleNet and LorentzNet: their T60, T100 and Q80 CIs lie outside the MLPx values, by factors of 2.1–4.1 on top jets. On QCD jets, MLPx's W-edit response alone is dW = −0.02 [−0.12, +0.09], while ParT's is dW = +0.46 [+0.30, +0.64].
  - Not excluded for PFN and EFN: their W − N CIs overlap MLPx at T60 and T100, and EFN's overlaps at Q80.
- **Hm: mW plus side effects** (the MLPxW pattern: −1.89 / −1.11 / −0.06 / +0.54).
  - ParT and ParticleNet match it within CIs in all four columns. LorentzNet is larger at T60 and T100.
  - This also accounts for T80. Moving the W pair onto 80.4 GeV raises LR3 / MLP3, which see only (mSD, τ32, mW). It does not raise any network or MLPxW, because the side effects (MLPx −0.19) cancel the mW gain (+0.19).
  - So the T80 null does not exclude mW use. It means T80 cannot discriminate between the readings.
- **Hd: any deformation of a top jet lowers the score.** For the networks, dW ≈ dN ≈ −0.2 on T80. On QCD jets this predicts a fall, but the networks rise on Q80 (ParT dW +0.46). So it does not explain the W − N contrast.
- **Monotone opening dependence.** Not supported: T60 (pair closed) and T100 (pair opened) both lower the score. In QCD, Q80up has n = 120 and is not resolved for any network: ParT dW +0.43 [+0.17, +0.69], but dN is +0.24, so W − N has a CI that includes 0.

## Assessment of the H4 reading
- **Correct and reproduced.** The size and sign of the W − N contrast for all five networks on top jets, the LR2 null, and the QCD rise. On the fresh sample: ParT −1.79 vs H4 −1.73 at T60; −1.14 vs −1.20 at T100; +0.40 vs +0.24 at Q80.
- **Understated: the τ21 control.** The MLP3t comparison ("a τ21-using MLP responds at 7 % of the mW MLP") understates the side-effect channel.
  - A classifier with τ1–3, τ21 and the subjet masses, and no mW (MLPx), responds at 38 % (T60) and 56 % (T100) of MLP3.
  - That is 35 % and 48 % of ParT's response.
- **Overstated: PFN / EFN.**
  - "PFN and EFN respond to the W-pair mass" (thesis note D6) and "EFN … is not a zero control" are not established. Their responses equal MLPx's within CIs.
  - The T100 natural-looking-half PFN number in H4 (−0.09 [−0.27, +0.09]) points the same way.
- **Not mentioned in H4.** The networks do not respond to moving top jets onto 80.4 GeV from 6–15 GeV away (T80). This is consistent with mW use only once the side effects are counted (MLPxW also ≈ 0).
- **Untested:** the T60 off-manifold caveat (C2ST 0.98) and the c2stsub half.

## Corrected reading
At fixed jet mass (|Δm| < 0.5 GeV) and τ32 (|Δτ32| < 0.02), moving the exclusive-kT W-candidate pair mass from about 80 GeV to 60 or 100 GeV lowers the top log-odds of ParT, ParticleNet and LorentzNet, beyond the non-W control. This replicates on a fresh sample: ParT W − N −1.79 [−1.98, −1.60] at 60 GeV and −1.14 [−1.35, −0.93] at 100 GeV.

The response is 2.1–4.1 times that of a substructure classifier without mW (MLPx: −0.63, −0.54). It matches a classifier with the same features plus mW (MLPxW: −1.89, −1.11). On QCD jets, moving the best pair from 90–115 GeV onto 80.4 GeV raises their score (ParT +0.50 [+0.25, +0.75]). From 50–70 GeV, the 120 jets do not resolve it.

PFN and EFN also respond, but by the same amount as MLPx, so their mW use is not separated from the edit's side effects (subjet shrinkage, lower τ21). Moving the W pair onto 80.4 GeV from 6–15 GeV away changes no network's score beyond the control. MLPxW gives the same null, so this does not distinguish the readings.

Proposed D6 scope: ParT, ParticleNet and LorentzNet only, confidence 0.7. PFN / EFN: open.

## Explainer entry addendum (for H4_EXPLAINER, "Checks")
- **Side-effect control.** The edit restores $\tau_{32}$ by scaling subjet sizes. A classifier on $(m_{SD},\tau_{1,2,3},\tau_{32},\tau_{21},n,\text{subjet }p_T\text{ fractions},\text{subjet masses})$, with no $m_W$, responds to the 80 → 60 GeV edit with $\Delta_W-\Delta_N=-0.63$. ParT gives −1.79 and the same classifier with $m_W$ added gives −1.89 (fresh sample, $n=545$). PFN (−0.77) and EFN (−0.63) are at the side-effect level.
- **Toward-80 edit.** Moving the pair onto 80.4 GeV from 6–15 GeV away raises LR($m_{SD},\tau_{32},m_W$) by 0.21 and leaves the networks unchanged. A classifier with $m_W$ and the substructure features is also unchanged (−0.06), so this edit does not discriminate.
