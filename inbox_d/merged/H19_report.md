# H19: W-pair mass inside top, JetClass vs Top Landscape models on matched jets and edit (open question c; claims D6 vs D1 / H3v)

Script `interp/h19_wpair_cross.py`. GPU job `d-h19-run` (4 CPU, 16 Gi, GTX 1080 Ti; 21 min; exit 0). Smoke runs `d-h19-smoke` and `d-h19-smoke2` were also made. The figure was redrawn by CPU job `d-h19-fig`.
Results are in `results/physics/h19_wpair_cross.json`, `h19_perjet.npz` and `h19_wpair.png`, with copies on the volume. The log is `logs/d-h19-run.log`.

## Conclusion
We applied the same W-candidate pair-mass edit (jet mass and τ32 held fixed) to two sets of jets: JetClass top/QCD jets restricted to the Top Landscape window (550–650 GeV) and Top Landscape jets. We scored both sets with the JetClass-trained and the Top-trained networks.

- The Top-trained ParT, ParticleNet and LorentzNet respond to the W-pair mass on both jet samples, at 2.4–4.7 times the side-effect-only reference. Their response therefore does not depend on which sample the jets come from.
- The JetClass-trained networks respond strongly when the W pair is moved up to 100 GeV (3–4 times the reference, on both samples). When it is moved down to 60 GeV they stay at the reference level.
- The cause of the low-side result: the JetClass networks lose almost as much top score when the other subjet pair (b plus one W quark) is changed instead.

So the JetClass/Top contrast depends on the edit direction and on the control, not only on training. Whether the JetClass networks read the three subjet-pair masses as a set, rather than the W pair specifically, is not resolved here.

## Procedure
**Samples** (random rows, never first-k):
- **jc.** JetClass test files `TTBar_100.root` and `ZJetsToNuNu_100.root`. Selection: stored 550 < jet_pT < 650 GeV, |η| < 2, ≥ 3 particles (38,239 and 36,575 eligible of 100,000 each). From these, 5,000 Tbqq (default_rng 4191) and 5,000 QCD (rng 4192) were drawn.
- **top.** Top Landscape test split: 5,000 top and 4,360 QCD (rng 4192). The H4 rows (seeds 601/602) and H4v rows (611/612) are excluded; only 4,360 fresh QCD test rows remain.

**Edit.** `interp.h4_verify.edit_set`, which uses the H4 functions unchanged:
- Three exclusive-kT subjets. The W pair is the pair whose mass is closest to 80.4 GeV.
- The subjets are moved rigidly in (η, φ), with the internal spread scaled to restore τ32.
- Acceptance: |Δm| < 0.5 GeV and |Δτ32| < 0.02.
- Sets:
  - T60 and T100: tops with 65 < mW < 95, target ±5 GeV.
  - T80: tops with 6 < |mW − 80.4| < 15, moved onto 80.4 ± 2.
  - Q80: QCD in 105 < mSD < 210 with mW 50–70 or 90–115, moved to 80.4.

**Controls on the same jets:**
- N (non-W pair): the pair (b, harder W daughter) gets the same relative mass change, with the W pair, m and τ32 held fixed.
- R (rotation): a rotation about the jet axis with the same pT-weighted displacement.

**Models.** Score = log-odds of top.
- JetClass released ParT_full, ParT_kin, PN_full, PN_kin. The score is L(Tbqq) − L(QCD) of the 10-class logits.
  - JetClass jets are rebuilt about the stored axis (η = jet_η + sign(jet_η)·Δη). Constituent masses are taken from the stored particles.
  - On Top jets only the kin models are run. Top four-vectors (the 128 hardest) are passed about the constituent-sum axis, with track and PID features set to 0.
- Top-trained ParT_kin and PN_kin (the H4 checkpoint snapshots), LorentzNet (released weights), PFN and EFN (H2).
  - JetClass jets are passed to them as four-vectors with E = |p|. Top Landscape constituents are massless (fraction with m > 0.1 GeV: 0.13 %, against 48 % in JetClass). Before this conversion, LorentzNet on JetClass jets had AUC 0.24 in the smoke run.

**References, retrained per dataset.** All are evaluated on both samples.
- jc references: `TTBar_101` / `ZJetsToNuNu_101` in-window rows, 25,000 per class.
- top references: the Top val split, 25,000 per class (rng 4194).
- LR2 = binned LR(mSD, τ32), the negative control.
- MLPx = MLP(64, 64) on mSD, τ1–3, τ32, τ21, multiplicity, sorted subjet pT fractions and sorted subjet masses. It sees the side effects of the edit, not mW.
- MLPxW = MLPx plus mW.

**Metric.**
- W − N Δscore (log-odds) per jet, with a 2,000-resample paired bootstrap.
- Sign-flip permutation p (2,000 permutations), one-sided in the mW-window direction (T60/T100 down, T80/Q80 up).
- Ratio to the MLPx and MLPxW of the model's own training dataset.
- H0_side: per-jet (W − N)_net = (W − N)_MLPx (sign-flip, one-sided).
- Holm correction over 64 tests (networks × samples × sets). The floor is 0.032.
- McNemar on W vs N calls at ε_S 0.3 / 0.5. Thresholds come from each sample's unedited tops.

## Checks
**Unedited models.** The table gives AUC, binary accuracy (score > 0), and 10-class accuracy on the Tbqq + QCD jets.

| model | jc sample AUC / acc | top sample AUC / acc | known |
|---|---|---|---|
| JetClass ParT_full | 0.9987 / 98.4 % (10-class 86.5 %) | – | 85.9 % 10-class (D4) |
| JetClass ParT_kin | 0.9925 / 95.8 % (81.2 %) | 0.9816 / 92.8 % (68.8 %) | 74.5 % 10-class |
| JetClass PN_full | 0.9980 / 98.1 % (85.5 %) | – | 84.4 % published |
| JetClass PN_kin | 0.9890 / 95.1 % (79.8 %) | 0.9838 / 93.3 % (74.3 %) | – |
| Top ParT | 0.9852 / 94.3 % | 0.9866 / 94.2 % | 93.6 % (H4v) |
| Top PN | 0.9849 / 94.1 % | 0.9857 / 94.0 % | 93.8 % |
| Top LorentzNet | 0.9870 / 94.4 % | 0.9872 / 94.4 % | 94.0 % |
| Top PFN / EFN | 0.9797 / 0.9808 | 0.9805 / 0.9765 | 92.7 / 91.8 % |
| MLPx / MLPxW (jc refs) | 0.9793 / 0.9802 | 0.9767 / 0.9765 | – |
| MLPx / MLPxW (top refs) | 0.9788 / 0.9792 | 0.9780 / 0.9782 | – |

- Each model reproduces its known accuracy on its own sample. The cross-dataset AUCs fall by at most 0.011.
- **Background rejection.** At ε_S 0.5 on the jc QCD jets (n = 5,000):
  - ParT_full and PN_full pass 0 QCD jets (1/ε_B > 5,000).
  - Top ParT: 156. Top LorentzNet: 357.
  - On top-sample QCD jets (n = 4,360): Top ParT 484, LorentzNet 623, JetClass ParT_kin 229.
- **Domain shift.** C2ST (HGB on 17 substructure features, 2-fold AUC), jc vs top:
  - Tops: 0.649. Median m 168.8 vs 173.6 GeV, τ32 0.50 vs 0.55.
  - QCD: 0.735. Median mSD 22.9 vs 35.7 GeV.
- **Input path.** Stored JetClass jets vs the identity rebuild used for every variant:
  - Median |Δscore|: ≤ 0.001 for all four JetClass nets.
  - 99th percentile: 0.02 (ParT_full), 1.26 (ParT_kin), < 10⁻⁵ (PN).
  - Calls changed: 0.1 % (ParT_kin), 0 for the others.
  - ParT_kin is sensitive to float32 rounding of near-collinear pairs. The same rebuild path is used for id, W, N and R, so this adds per-jet noise, not bias.
- **Edit fidelity** (medians, T60): mW 79.5 → 60.0 (W) / 79.2 (N); m 171.44 → 171.44; τ32 0.430 → 0.431.
  - Side effects the edit does not hold: τ21 0.486 → 0.370 (W) / 0.437 (N), and subjet mass sum 44.6 → 37.0 / 42.5 GeV (the same as in H4v).
- **Rotation control.** In the T sets, |dR| ≤ 0.14 for every network, with all 95 % interval bounds within [−0.18, +0.05]. In Q80 (n ≈ 20), dR reaches −0.31 (JetClass-sample EFN).
- **Negative control.** LR2 has |W − N| ≤ 0.11 in all T sets.

## Numbers (W − N Δscore [95 % CI]; ratio to MLPx of the training dataset [CI]; Holm p for H0_side)
### T60 (W pair 80 → 60 GeV); n = 683 (jc), 611 (top)
| model | jc: dW / dN | jc: W − N | jc: × MLPx | top: dW / dN | top: W − N | top: × MLPx |
|---|---|---|---|---|---|---|
| JC ParT_full | −3.20 / −2.68 | −0.52 [−0.72, −0.33] | 1.18 [0.74, 1.65], p 1 | – | – | – |
| JC ParT_kin | −2.42 / −2.23 | −0.19 [−0.36, −0.04] | 0.43 [0.09, 0.81], p 1 | −1.93 / −1.88 | −0.06 [−0.19, +0.08] | 0.13 [−0.18, 0.41], p 1 |
| JC PN_full | −2.68 / −2.09 | −0.60 [−0.76, −0.43] | 1.34 [0.95, 1.74], p 1 | – | – | – |
| JC PN_kin | −2.13 / −1.69 | −0.44 [−0.57, −0.30] | 0.98 [0.65, 1.31], p 1 | −1.90 / −1.36 | −0.54 [−0.66, −0.42] | 1.22 [0.94, 1.51], p 1 |
| Top ParT | −1.64 / −0.23 | −1.40 [−1.57, −1.23] | 2.38 [2.00, 2.82], p 0.032 | −2.03 / −0.30 | −1.73 [−1.91, −1.56] | 2.98 [2.56, 3.48], p 0.032 |
| Top PN | −1.61 / −0.11 | −1.50 [−1.70, −1.29] | 2.53 [2.11, 3.03], p 0.032 | −2.09 / −0.24 | −1.85 [−2.05, −1.66] | 3.18 [2.75, 3.69], p 0.032 |
| Top LorentzNet | −3.12 / −0.38 | −2.73 [−2.98, −2.47] | 4.63 [3.95, 5.43], p 0.032 | −3.25 / −0.50 | −2.75 [−2.96, −2.54] | 4.73 [4.15, 5.44], p 0.032 |
| Top PFN | −1.15 / −0.40 | −0.74 [−0.84, −0.64] | 1.26 [1.08, 1.46], p 0.068 | −1.19 / −0.43 | −0.76 [−0.86, −0.66] | 1.30 [1.13, 1.50], p 0.032 |
| Top EFN | −0.92 / −0.33 | −0.59 [−0.72, −0.47] | 1.00 [0.80, 1.22], p 1 | −0.79 / −0.16 | −0.63 [−0.75, −0.52] | 1.09 [0.89, 1.32], p 1 |
| MLPx (jc / top refs) | | −0.45 / −0.59 | 1 | | −0.44 / −0.58 | 1 |
| MLPxW (jc / top refs) | | −1.65 / −1.66 | 3.7 / 2.8 | | −1.72 / −1.63 | 3.9 / 2.8 |

### T100 (W pair 80 → 100 GeV); n = 472 (jc), 436 (top)
| model | jc: W − N | jc: × MLPx | top: W − N | top: × MLPx |
|---|---|---|---|---|
| JC ParT_full | −1.79 [−2.07, −1.52] (dN −1.33) | 4.16 [3.34, 5.30], p 0.032 | – | – |
| JC ParT_kin | −1.35 [−1.57, −1.12] (dN −1.19) | 3.13 [2.54, 4.00], p 0.032 | −0.81 [−0.97, −0.64] | 3.59 [2.71, 5.18], p 0.032 |
| JC PN_full | −1.85 [−2.08, −1.60] (dN −0.92) | 4.29 [3.54, 5.37], p 0.032 | – | – |
| JC PN_kin | −1.40 [−1.59, −1.21] (dN −0.84) | 3.26 [2.67, 4.12], p 0.032 | −0.67 [−0.84, −0.50] | 2.99 [2.24, 4.28], p 0.032 |
| Top ParT | −1.60 [−1.82, −1.39] (dN +0.12) | 2.61 [2.14, 3.29], p 0.032 | −1.19 [−1.38, −1.00] | 3.38 [2.62, 4.73], p 0.032 |
| Top PN | −1.90 [−2.14, −1.66] | 3.10 [2.56, 3.93], p 0.032 | −1.31 [−1.51, −1.11] | 3.74 [2.90, 5.28], p 0.032 |
| Top LorentzNet | −2.70 [−2.99, −2.40] | 4.41 [3.58, 5.56], p 0.032 | −1.95 [−2.17, −1.73] | 5.57 [4.29, 7.95], p 0.032 |
| Top PFN / EFN | −0.91 / −0.97 | 1.48 / 1.59, p 0.032 | −0.60 / −0.62 | 1.71 / 1.78, p 0.032 |
| MLPx / MLPxW (jc refs) | −0.43 / −1.40 | 1 / 3.25 | −0.22 / −0.93 | 1 / 4.17 |
| MLPx / MLPxW (top refs) | −0.61 / −1.26 | 1 / 2.06 | −0.35 / −0.85 | 1 / 2.44 |

### T80 (W pair moved onto 80.4 GeV, from 6–15 GeV away; window direction = rise); n = 520 (jc), 496 (top)
- **JetClass nets.** On jc jets W − N is +0.31 to +0.54: ParT_full +0.40 [+0.20, +0.61], PN_kin +0.54 [+0.42, +0.66]; Holm 0.032.
  - On top jets the kin nets give +0.27 [+0.15, +0.40] (ParT_kin) and +0.28 [+0.19, +0.37] (PN_kin).
  - The positive sign comes mostly from N: dN is −0.63 to −0.92 for the JetClass nets, while dW is −0.18 to −0.52.
- **Top nets.** W − N is between −0.09 and +0.07 on both samples, all CIs containing 0. This matches H4v (−0.07 to +0.01).
- **References.** MLPxW_jc +0.03 [−0.05, +0.10], MLPx_jc −0.12.

### Q80 (QCD moved to 80.4 GeV)
- Analysis n is 20 (jc) and 17 (top), because few QCD jets pass both the W and N acceptance.
- All CIs span 0. MLPxW_jc gives +0.52 [−0.08, +1.11]. This set has too few jets to test either hypothesis.

### Calls at ε_S = 0.5, T60, jc jets (pass rate id → W / N; McNemar N-pass→W-fail / reverse)
- JC ParT_full: 63.4 → 21.1 / 33.4 %; 138 / 54; p 10⁻⁹.
- JC ParT_kin: 64.9 → 25.8 / 27.8 %; 93 / 79; p 0.32.
- Top ParT: 66.5 → 36.7 / 62.7 %; 220 / 43; p 10⁻³⁰.
- Top LorentzNet: 70.9 → 30.9 / 66.0 %; 281 / 41; p 10⁻⁴⁵.

For T100 on jc jets, JC ParT_full gives 67.6 → 28.0 / 49.8 % (136 / 33; p 10⁻¹⁶).

## Reading
**Predictions and outcomes.**
- **H1** (the edit/sample difference; JetClass ParT/PN at 2–4× MLPx on both samples):
  - Met at T100: 3.1–4.3× on jc jets and 3.0–3.6× (kin) on top jets, all Holm 0.032.
  - Not met at T60: 0.43–1.34× on jc jets and 0.13–1.22× on top jets, Holm 1.
- **H0** (the training difference; JetClass nets at the MLPx level, Top nets at the MLPxW level on JetClass jets):
  - The Top-net part is met at T60 and T100. Top ParT/PN on jc jets give 2.4–3.1× MLPx_top, against MLPxW_top at 2.1–2.8×. LorentzNet gives 4.4–4.6×.
  - The JetClass-net part is met at T60 and excluded at T100.
- **Sample-factor alternative** (Top nets lose the response on JetClass jets): excluded.
  - Top ParT, PN and LorentzNet ratios on jc jets are 0.80–0.98 of their ratios on top jets at T60, with overlapping CIs. PFN and EFN are at the same ratio on both samples.

**The non-W pair control.** For the JetClass nets, the N control moves the score almost as much as the W edit:
- T60: dN −1.7 to −2.7 against dW −2.1 to −3.2.
- T80: dN −0.6 to −0.9 for an 8 % change of the (b, harder W daughter) pair mass.
- For the Top nets and MLPx, dN is −0.5 to +0.3.

So for the JetClass nets the W − N contrast measures the difference between two responses that are both large. The low result at T60, and the positive W − N at T80 where MLPxW gives 0, both follow from the size of dN.

This is consistent with the H3v result (no W-pair effect beyond a W-fixed control on JetClass ParT) and with the H3v observation that ParT's top call falls under any rearrangement of the pair masses (proposed H6). It does not separate a "set of pair masses" reading from a "deviation from the training pair-mass configuration" reading.

**Limits.**
- One edit family, with τ21 and subjet-mass side effects (MLPx covers them only to the extent its features do).
- Q80 is underpowered (n ≈ 20).
- The rejection on 4,360–5,000 QCD jets is saturated for JetClass ParT_full/PN_full.
- ParT_kin input-path noise has a 99th percentile of 1.26 in |Δscore|.
- The jc QCD jets differ from the Top QCD jets (C2ST 0.73; lower mSD).

## Explainer entry (for site/explain.py, H19_EXPLAINER)
- **Question.** Do jet taggers trained on JetClass and on the Top Tagging Landscape use the W-boson mass inside a top jet in the same way, when tested with the same edit on the same jets?
- **Physics background.**
  - In $t\to bW\to bq\bar q'$ two of the three subjets reconstruct $m_W\approx80.4$ GeV, and all three reconstruct $m_t$.
  - At fixed $m_J$ and $\tau_{32}$, a tagger that uses the W constraint should lose top probability when the W-candidate pair mass $m_W^{\rm cand}$ moves away from 80 GeV.
  - A tagger that only reads $m_J$ and $\tau_{32}$ should not.
  - A rigid move of subjets also changes $\tau_{21}$ and the subjet sizes, so a reference that sees those side effects is needed.
- **Method.**
  - Edit: three exclusive-$k_T$ subjets; the pair with mass closest to 80.4 GeV is rescaled to $t$ = 60 or 100 GeV (or onto 80.4), with $|\Delta m_J|<0.5$ GeV and $|\Delta\tau_{32}|<0.02$.
  - Controls: N gives the (b, harder W daughter) pair the same relative mass change with the W pair fixed; R is a rotation.
  - Metric: $\Delta_{W-N}=\langle (s_W-s_{\rm id})-(s_N-s_{\rm id})\rangle$ in log-odds, divided by the same quantity for MLPx (no pair mass) of the model's training dataset.
  - Samples: JetClass top/QCD jets with $550<p_T<650$ GeV, $|\eta|<2$, and Top Landscape jets (5,000 per class). Every model is scored on both samples.
- **Result** (figure `h19_wpair.png`).
  - Top-trained ParT, ParticleNet and LorentzNet give $\Delta_{W-N}$ = 2.4–4.7 × MLPx on both samples, for both 60 and 100 GeV.
  - JetClass-trained ParT and ParticleNet give 3–4 × MLPx at 100 GeV, but 0.4–1.3 × at 60 GeV.
  - At 60 GeV the JetClass nets lose almost as much under the non-W control ($\Delta_N\approx-2$) as under the W edit.
- **Checks.**
  - Unedited accuracies are reproduced: JetClass ParT_full 86.5 % 10-class on these jets; Top ParT 94.2 %.
  - Cross-dataset AUC changes ≤ 0.011.
  - Rotation control $|\Delta_R|\le0.14$ in the top sets; LR(mSD, τ32) control $|\Delta_{W-N}|\le0.11$.
  - Holm over 64 tests.
- **Physical significance.**
  - The Top-trained networks' use of the W-pair mass holds on JetClass jets, so it is a property of the networks, not of the Top Landscape sample.
  - The JetClass networks are sensitive to every subjet-pair mass, not specifically to the W pair. With a non-W control that is itself large for them, a W-specific term appears only on the high-mass side.
