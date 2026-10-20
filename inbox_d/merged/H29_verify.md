# H29 verify: JEDI-linear W-pair edit with each edited jet rebuilt about its own axis

Script `interp/h29_verify.py`. It imports `interp/h29_jedi_wpair_rebuild.py` unchanged and replaces only the seed, the output tag and the sample exclusion. Job `anrunw-ai-d-h29-verify` (AI_PY=conda, 2 CPU / 6 GiB; parts edit, jedi (jax env), stats, refens; wall 767 s, exit 0). Results: `results/physics/h29_verify.json` (H29 statistics on the fresh sample, 78-test Holm family, JEDI-linear and the redrawn null only), `h29_verify_refens.json` (reference spread), `h29_verify_edit.json`, `h29_verify_wpair_rebuild.png`. Log: `logs/d-h29-verify.log`.

## Conclusion

Rebuilt about each edited jet's own axis, JEDI-linear lowers its top score under a W-pair-mass edit by an amount inside the range of five retrained classifiers that never see the pair mass. This holds at 60, 70 and 100 GeV on both the H29 sample and a fresh one. At 100 GeV the drop is smaller than for every retrained classifier given the pair mass. At 60 GeV it is smaller for five of six. The fixed-axis sign flip comes from the control edit moving the jet axis, and it reproduces. Not settled: the specific ratios quoted (1.01× and 0.40–0.42×) depend on which single reference fit is used. At 90 GeV JEDI-linear raises its top score (+0.15 log-odds) while all twelve references lower theirs. The side-effect reference therefore does not reproduce JEDI-linear's response at every target.

## Code check

- **Axis and coordinates.** st uses the stored (Δη, Δφ) as the coordinates. rc subtracts each variant's E-scheme axis (the sum of the massless four-vectors of its 64 real particles). The η shift is applied as a longitudinal boost and the φ shift as a rotation, with wrapping, and pT is unchanged. The identity rebuild reproduces the stored inputs to 6e-5 on the fresh sample. On unedited jets the rc axis lies 0.0015 rad (median) from (0, 0).
- **Padding.** Particles count as real only when pT > 0. The script asserts that padding is contiguous at the end. Observables, references and m_A all use the same `nn`.
- **Rows.** Every class is drawn with `rng.choice` over its rows, never by first-k. The fresh sample overlaps H3, H3v, H23 and H29 in 0 rows.
- **Same jets.** The analysis rows are W ∩ N ∩ R accepted. For each variant, the model inputs and the reference features come from the same edited P4 (`REFP`). `searchsorted` relies on `edit_set` returning sorted rows, which it does, because `rows` comes from `np.flatnonzero`.
- **Conservation.** The edit holds the jet invariant mass to |Δm| < 0.5 GeV and τ32 to 0.02 by acceptance cut, not exactly. Pair masses are set to within 5 GeV of the target. In rc, Δlog m_A is ≤ 4e-4 for W and N, so A4's mass variable stays fixed in the rc build.
- **Working point.** eS50 is the median score of the unedited tops in the same build.
- **No bug found.** One single-reference fragility, described below.

## Rerun on a fresh sample

- **Sample.** hls4ml test archive, 2,000 per class, `default_rng(2911)`, excluding H3 (501), H3v (601), H23 (9523) and H29 (2901) rows.
- **Unedited accuracy.** JEDI-linear 5-class accuracy is 81.20 % st / 81.13 % rc (80.92 % on the full 260k test set). Top efficiency is 83.3 % and AUC t vs q/g is 0.964. The redrawn-weight JEDI-linear scores 20.0 % and calls every jet top.
- **Analysis sets.** T60 249, T70 531, T90 506, T100 167, Q80 24, H3v_60 905, H3v_100 721.

JEDI-linear W − N, top-vs-QCD log-odds. Values are the mean [95 % bootstrap CI over jets, B = 2,000]; "H29" is the original sample.

| set | st (fresh) | rc (fresh) | rc (H29) | single MLPx / MLPxW (fresh) | rc ×MLPx [CI] | rc ×MLPxW [CI] | rc − MLPx, one-sided t p (Holm, 78 tests) |
|---|---|---|---|---|---|---|---|
| T60 | +0.97 [+0.79, +1.17] | −0.32 [−0.47, −0.16] | −0.32 | −0.25 / −0.71 | 1.27 [0.62, 2.38] | 0.46 [0.24, 0.67] | −0.07 [−0.25, +0.12], p 0.23 (1) |
| T70 | +0.37 | −0.19 [−0.27, −0.10] | −0.16 | −0.07 / −0.13 | 2.84 [1.27, 7.67] | 1.42 [0.70, 2.43] | −0.12 [−0.21, −0.02], p 0.007 (0.43) |
| T90 | +0.40 | +0.15 [+0.07, +0.22] | +0.15 | −0.12 / −0.27 | −1.25 | −0.53 | +0.26, p 1 |
| T100 | +0.13 [−0.14, +0.41] | −0.30 [−0.48, −0.10] | −0.34 | −0.27 / −0.99 | 1.10 [0.44, 2.00] | 0.30 [0.11, 0.47] | −0.03 [−0.21, +0.16], p 0.38 (1) |
| Q80 (n 24) | +0.11 | +0.28 [−0.06, +0.60] | +0.02 | +0.22 / +0.40 | 1.26 | – | p 0.36 (1) |
| H3v_60, E − B | −0.35 | −0.47 [−0.54, −0.40] | −0.47 | −0.64 / −0.78 | 0.73 | 0.60 | p 1 |

- **Two-sided test of rc against the single MLPxW.** T60 p 1e-5; T100 p 1e-10.
- **Axis term.** Residual (ΔN_st − ΔN_rc) − ΔTN_st, with ΔTN_st in brackets:
  - T60: −0.09 [−0.25, +0.09] (ΔTN_st −1.16)
  - T70: −0.14 [−0.26, −0.03] (−0.39)
  - T90: +0.03 (−0.31)
  - T100: −0.13 [−0.30, +0.05] (−0.31)
- **Axis shift.** At T60 the N control moves the axis by 0.074 rad (median) and W by 0.011 rad.
- **Redrawn-weight null.** |W − N| ≤ 0.003 in every cell.
- **Paired dose test** (tops accepted at all four targets, n 47). JEDI-linear rc: T60 − T70 = +0.26 (one-sided p 0.92); T100 − T90 = +0.32 (p 0.98). Both point opposite to the H1 ordering. The H29 exploratory Welch result (100 − 90 = −0.40, p 3e-4) is not reproduced in this paired set.
- **eS50 McNemar, W vs N, rc.** T60 34/18 (p 0.036; Holm 1); T70 49/23 (p 0.003; Holm 0.19).

## Missing control: spread of the reference classifiers

H29 used one MLPx and one MLPxW fit (sklearn MLP, `random_state=0`, one 50k subset), and its CIs cover jets only. Here MLPx and MLPxW were each retrained five times, each with `random_state=i` on its own random 50k subset of a 150k pool (75k t + 75k q/g train-archive jets, dev excluded, rng 2919 / 3000+i). A gradient-boosted pair (HGBx, HGBxW) was also fitted. Every member has unedited AUC 0.947–0.954.

W − N of the members (fresh sample; H29 sample in brackets), with JEDI-linear rc alongside:

| set | JEDI rc | MLPx members (5) | HGBx | MLPxW members (5) | HGBxW | JEDI ratio to MLPx members | JEDI ratio to MLPxW members |
|---|---|---|---|---|---|---|---|
| T60 | −0.32 [−0.32] | −0.18 to −0.51 [−0.22 to −0.56] | −0.14 [−0.21] | −0.32 to −1.06 [−0.44 to −1.10] | −1.10 [−1.19] | 0.63–1.76 [0.57–1.46] | 0.30–0.99 [0.29–0.72] |
| T70 | −0.19 [−0.16] | 0.00 to −0.20 [−0.03 to −0.17] | +0.01 [−0.03] | +0.06 to −0.26 [+0.02 to −0.23] | −0.38 [−0.37] | 0.9–55 [0.9–4.6] | – |
| T90 | +0.15 [+0.15] | −0.06 to −0.20 [0.00 to −0.15] | −0.07 [−0.05] | −0.15 to −0.47 [−0.12 to −0.45] | −0.40 [−0.38] | negative | negative |
| T100 | −0.30 [−0.34] | −0.34 to −0.58 [−0.17 to −0.41] | −0.19 [−0.11] | −0.56 to −1.30 [−0.52 to −1.15] | −0.88 [−0.82] | 0.51–0.87 [0.83–2.01] | 0.23–0.53 [0.29–0.65] |

JEDI rc minus each reference, with 95 % bootstrap CI:

- **T60.** Against the MLPx members and HGBx, 5 of 6 CIs contain 0 on the fresh sample (HGBx −0.18 [−0.36, −0.01]). On H29, 5 of 6 contain 0 (MLPx_3 +0.24 [+0.05, +0.43]). Against the MLPxW members, 5 of 6 CIs exclude 0 on both samples. The exception is MLPxW_4: +0.002 [−0.18, +0.18] fresh, +0.12 [−0.05, +0.29] on H29.
- **T100.** Against every MLPxW member and HGBxW, the difference is +0.18 to +1.01, with 11 of 12 CIs excluding 0 (MLPxW_3 on H29: +0.18 [−0.02, +0.40]). Against the MLPx members, 10 of 12 CIs contain 0.
- **T70.** JEDI rc lies 0.12–0.19 below MLPx_0, MLPx_4 and HGBx (CIs exclude 0) and within ±0.02 of MLPx_1–3.
- **T90.** JEDI rc lies above all 12 references on both samples: +0.15 to +0.34 above the MLPx-type members and +0.27 to +0.62 above the MLPxW-type members. Every CI excludes 0.

## Hypotheses

- **H0 (D1 null).** The rc W − N lies within the side-effect references' range at every target. Met at 60, 70 and 100 GeV. Not met at 90 GeV, where the departure is opposite in sign to the pair-mass references.
- **H1 (W-mass window).** The rc W − N approaches the pair-mass references and is ordered in |mW − 80|. Excluded at 100 GeV, where JEDI lies below every MLPxW member, and at 60 GeV against 5 of 6 members. The paired dose test shows no ordering on either sample (n 29, n 47).
- **H2 (axis-relative mass under a fixed axis).** The st − rc difference equals the translation-only response within 0.09 at 60 GeV and within 0.15 at 70–100 GeV. Reproduced.
- **Not tested here.** The small models and PFNs, and with them the H29 cross-model statements (D6 / D11), were not rerun.

## Issues with H29 as reported

1. **The quoted ratios depend on a single reference fit.**
   - "1.01× MLPx" and "0.40–0.42× MLPxW" come from one MLP fit each. Across five retrains plus an HGB, the ratio at 60 GeV runs 0.57–1.76× to MLPx members and 0.29–0.99× to MLPxW members.
   - The bootstrap CIs in H29 omit this source of variance.
   - The robust separation is at 100 GeV, where JEDI sits below all MLPxW-type references. At 60 GeV, one MLPxW retrain matches JEDI.
2. **The T90 departure is left out of the reading.**
   - At 90 GeV JEDI-linear's rc W − N is +0.15, against −0.00 to −0.47 for every reference, on both samples.
   - So MLPx does not model JEDI-linear's response to the W edit at every target, which weakens the reading of the 60 GeV match as "equals the side-effect reference".
   - The reading's sentence "no set is ordered in |mW − 80| beyond MLPx" is correct as stated.
3. **T70 is sensitive to the reference fit.** JEDI-linear exceeds the single MLPx in the same direction on both samples (2.04×, p 0.049; 2.84×, p 0.007, Holm 0.43). Against three of the five MLPx retrains the excess is 0.
4. **The exploratory Welch dose result does not reproduce.** The 100 − 90 GeV result (p 3e-4) is not reproduced in the fresh paired set (n 47, opposite sign).
5. **Figure.** In `h29_wpair_rebuild.png` the reference curves are single fits drawn without intervals, and the title states the interpretation. The JEDI-linear panel matches the numbers.

## Corrected reading

- **Fixed (0, 0) axis.** With the H4-type edit, JEDI-linear's W − N is positive (+0.92 on H29, +0.97 on the fresh sample at 60 GeV). The st − rc difference matches the translation-only response to the non-W control's axis shift (0.07 rad) within 0.09 at 60 GeV.
- **Each jet's own axis.**
  - JEDI-linear's W − N at 60, 70 and 100 GeV lies inside the range of five retrained side-effect MLPs.
  - At 100 GeV it lies below every pair-mass reference, at 0.23–0.65× their response.
  - At 60 GeV it lies below 5 of 6 pair-mass references (ratio 0.29–0.99×).
  - At 90 GeV JEDI-linear's score rises (+0.15) while every reference's falls.
  - No ordering in |mW − 80| appears in the paired dose sets.
- **Status of the ratios.** The H29 point ratios (1.01×, 0.40–0.42×) carry an unreported reference-fit spread of about ±50 %.
