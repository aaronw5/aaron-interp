# H23 verify: axis-rebuild audit of the W-pair edits, rerun on fresh JetClass rows

Jobs: d-h23-verify (GPU, `python -m interp.h23_verify`) and d-h23-vstat (CPU, `python -m interp.h23_verify_stats`). Scripts: `interp/h23_verify.py` (reuses `interp/h23_axis_audit.py` unchanged and adds two dose targets) and `interp/h23_verify_stats.py`. Results: `results/physics/h23_verify_jc.json`, `h23_verify_jc_perjet.npz`, `h23_verify_stats.json`.

## Conclusion

When each edited top jet is rebuilt about its own axis, the JetClass-trained ParT and ParticleNet lower their top score when the mass of the W-candidate subjet pair is moved away from 80 GeV. At 60 GeV the drop is about as large as that of a reference classifier that is given the pair mass directly. The drop also grows with distance from 80 GeV: it is larger at 60 than at 70 GeV, and larger at 100 than at 90 GeV, as expected if the networks use a W-mass window. This reproduces on fresh jets. Three statements in H23 do not hold, and the H13 and Top-Landscape arms were not rerun:
- At 100 GeV, three of the four networks do respond to the H3v edit.
- ParT_full is separated from the side-effect classifier under the H3v 60 GeV edit on the fresh rows.
- For ParT_full, the fixed-axis artefact is not a pure translation: on the fresh rows a residual of about 0.5 log-odds remains at 60 GeV.

## Code review of `interp/h23_axis_audit.py`

- **Axis and coordinate convention.** The rc axis is the E-scheme sum of the real constituents (`axis()`, masked by the particle count). `jc_dict` sets deta = sign(jet_eta)·(η − jet_eta) about the axis it is given, and `part_core.build` inverts this with the same sign, so the build is self-consistent. `build` recomputes jet_pt and jet_energy from the particles, so logptrel and logerel follow the edited jet. deltaR uses the rebuilt deta and dphi. No bug found.
- **Feature indices.** GROUPS (PID 5–10, tracks 11–14) match `FEATURES['full']`: KIN has 5 features, then charge plus 5 flags, then d0, d0err, dz, dzerr, deta, dphi. ParticleNet points are the last two features, deta and dphi. Correct.
- **Padded particles.** Every sum is masked by nn. `inputs()` fills padded slots with copies of real particles, but those slots are masked. No bug found.
- **Sampling.** Rows are random, never first-k. The H19 and H19v draws are excluded (overlap 0). Edits are applied to the same rows the models score.
- **References.** MLPx, MLPxW and LR2 use axis-independent features (exclusive-kT subjets, τN, mSD), so scoring them on the st path only is valid for both builds.
- **Working points.** The ε_S 0.3 and 0.5 thresholds come from the unedited tops of each path. Correct.
- **Reading errors in the report (data, not code).**
  - At H3v 100 GeV the MLPx side-effect response is positive (+0.36): the edit makes the jet look more top-like to MLPx. So the ratio to MLPx is negative and the one-sided "vs MLPx" test is not informative there. In `h23_summary.json` the weight-permuted PFN null also "passes" that test (Holm 9e-20, with E − B of −0.04).
  - The report read this cell as "no model responds at 100 GeV". The right comparisons are MLPxW (−0.04) and the null (−0.04). Against those, ParT_full gives −0.25 [−0.45, −0.04] and ParT_kin −0.15 [−0.29, −0.01] on the H23 rows.
- **Figure** (`results/physics/h19_wpair.png`, read):
  - The T60 and T100 fixed-axis points use the same grey "x" marker and colour, so the legend cannot tell them apart.
  - The EFN fixed-axis T60 point (−1.6×) lies below the y-range (−1.5), so only its error bar is visible.
  - The upper CI of ParT_full T100 is clipped at 9.
  - The right panel's legend says "fixed axis (as H19)". For Top Landscape the original H4 build was recentred, and the fixed-axis build was made here only for comparison.

## Procedure (verification run)

- **Sample.**
  - JetClass TTBar_100 and ZJetsToNuNu_100, 550 < pT < 650 GeV, |η| < 2, ≥ 3 particles.
  - 2,000 Tbqq (default_rng(6623)) and 2,000 QCD (default_rng(6624)), drawn as random rows.
  - Excluded: the H19 and H19v draws, and the H23 draws (rng 9523/9524, reconstructed with the same calls). Overlap 0.
- **Models.** ParT_full, ParT_kin, PN_full, PN_kin.
- **Unedited accuracy on this sample** (st build; rc differs by ≤ 0.1 point):

| model | AUC | binary acc. | 10-class acc. |
|---|---|---|---|
| ParT_full | 0.9983 | 98.1 % | 86.1 % |
| ParT_kin | 0.9909 | 95.0 % | 81.0 % |
| PN_full | 0.9974 | 97.8 % | 85.2 % |
| PN_kin | 0.9877 | 94.6 % | 79.2 % |

- **References.** MLPx_jc (AUC 0.9776), MLPxW_jc (0.9788) and LR2_jc, retrained exactly as in H23 (same iterations and validation accuracy).
- **Edits.** The H4 W-pair edit (h4_verify.edit_set), unchanged: mW → target at fixed m (|Δm| < 0.5 GeV) and τ32 (|Δτ32| < 0.02). Control N: non-W pair, same relative mass change. Control R: rotation. Translation-only control TN.
- **Targets.** 60 and 100 GeV as in H23, plus two added targets, 70 and 90 GeV, as a dose-response control.
  - A W-mass-window reading predicts |W − N| ordered 60 > 70 and 100 > 90.
  - A generic "any subjet rearrangement" reading predicts no ordering in |mW − 80|.
- **H3v edit.** k-means subjets, 60 and 100 GeV, with control B.
- **Builds.** st = stored axis; rc = each edited jet's own E-scheme axis.
- **Statistics.**
  - 95 % bootstrap CI over jets (B = 2,000).
  - Paired t-tests on per-jet differences; Welch t-test for the dose comparison, which uses different jet sets.
  - Holm and BH over a 72-test family; 50 tests have Holm p < 0.05.
- **Not rerun.** H13 soft contamination, the Top Landscape sample and the T80/Q80 sets.

## Numbers (fresh rows; H23 value in brackets where one exists)

### H4 edit, recentred (rc): W − N log-odds L_T − L_QCD [95 % CI], × MLPx and × MLPxW

MLPx and MLPxW reference values (W − N):

| set | n | MLPx | MLPxW |
|---|---|---|---|
| T60 | 265 | −0.44 [−0.54, −0.35] | −1.60 [−1.78, −1.43] |
| T70 | 591 | −0.17 | −0.40 |
| T90 | 563 | −0.11 | −0.33 |
| T100 | 170 | −0.48 [−0.67, −0.30] | −1.43 [−1.70, −1.19] |

Network W − N on the rc build (H23 value in brackets):

| model | T60 rc W − N | × MLPx [CI] | × MLPxW [CI] | T70 | T90 | T100 rc W − N | × MLPx | × MLPxW |
|---|---|---|---|---|---|---|---|---|
| ParT_full | −1.59 [−1.93, −1.25] | 3.59 [2.77, 4.76] (4.39) | 0.99 [0.80, 1.19] (1.24) | −0.34 | −0.64 | −3.00 [−3.44, −2.54] | 6.21 (6.25) | 2.10 |
| ParT_kin | −1.84 [−2.15, −1.55] | 4.16 [3.33, 5.34] (4.48) | 1.15 [1.00, 1.31] | −0.44 | −0.59 | −2.71 | 5.62 (5.84) | 1.90 |
| PN_full | −1.29 [−1.62, −0.97] | 2.91 [2.13, 3.90] (3.57) | 0.80 [0.62, 1.00] | −0.18 | −0.46 | −2.21 | 4.57 (5.42) | 1.55 |
| PN_kin | −1.47 [−1.74, −1.21] | 3.33 [2.60, 4.33] (3.50) | 0.92 [0.78, 1.06] | −0.26 | −0.45 | −2.06 | 4.26 (4.33) | 1.44 |

- **H0** (net W − N = MLPx W − N) at T60/T100 rc: excluded for all four nets, Holm ≤ 9e-6.
- **H0'** (net W − N = MLPxW W − N): not excluded at T60 for any net (Holm ≥ 0.6). Excluded at T100: the nets drop 0.6–1.6 more than MLPxW (Holm ≤ 1.3e-4).
- **Rotation control R at T60 (rc).** W − R is 3.6–5.6× MLPx, so the result does not depend on the N control alone.
- **Calls at ε_S 0.3 (rc, T60).** The W edit changes 37–41 % of top calls and N changes 15–21 %. McNemar W vs N p ≤ 5e-14.
- **Fixed axis (st), T60.** ×MLPx is 1.25 / 0.36 / 1.42 / 0.97 for ParT_full / ParT_kin / PN_full / PN_kin (H23: 1.31 / 0.80 / 1.61 / 0.72). The fixed-axis null reproduces.

### Dose-response control (new)

- **Rows used.** All four targets use the same eligible tops (65 < mW < 95 GeV). The accepted subsets differ by target: n = 265, 591, 563 and 170.
- **ParT_full |W − N| (rc).** 1.59 (60 GeV), 0.34 (70), 0.64 (90), 3.00 (100). MLPxW gives 1.60, 0.40, 0.33, 1.43.
- **Test.** W − N is more negative at 60 than at 70 GeV, and at 100 than at 90 GeV, for all four nets (Welch one-sided, Holm ≤ 4e-8). The ordering is the one a W-mass window predicts and that MLPxW shows.
- **Excluded hypothesis.** A response that is independent of |mW − 80| is excluded.
- **Asymmetry.** The upward side (90, 100 GeV) gives 1.3–2.1× MLPxW, the downward side 0.8–1.15×.

### Translation residual: (dN_st − dN_rc) − dTN_st [CI], rc

| model | T60 | T70 | T90 | T100 | H23 T60 |
|---|---|---|---|---|---|
| ParT_full | +0.48 [+0.23, +0.74] (Holm 0.007) | +0.20 (Holm 0.001) | +0.11 (0.07) | +0.13 [−0.06, +0.32] | +0.13 [−0.06, +0.33] |
| ParT_kin | +0.08 [−0.11, +0.27] | +0.06 | −0.00 | −0.01 | −0.01 |
| PN_full | +1.20 | +0.69 | +0.26 | +0.67 | +0.80 |
| PN_kin | +0.73 | +0.40 | +0.14 | +0.48 | +0.36 |

- The consistency check dTN on rc is ≤ 0.014 for every net.
- H23's statement "for ParT that response equals the fixed-minus-recentred difference within ±0.15" holds for ParT_kin at every target. It holds for ParT_full only at T90 and T100. At T60 on the fresh rows the ParT_full residual is +0.48, with a CI that excludes 0.

### H3v edit (k-means subjets), rc: E − B log-odds

60 GeV (n 1,055): MLPx −0.455 [−0.53, −0.38], MLPxW −0.82.

| model | E − B rc [CI] | × MLPx [CI] | H23 × MLPx | net − MLPx, Holm | net − MLPxW, Holm |
|---|---|---|---|---|---|
| ParT_full | −0.88 [−1.04, −0.71] | 1.93 [1.51, 2.39] | 1.58 (Holm 0.40) | −0.42, 1.9e-5 | −0.06, 1 |
| ParT_kin | −1.35 | 2.96 | 2.76 | −0.89, 3e-28 | −0.53, 7e-12 |
| PN_full | −0.63 | 1.38 [1.04, 1.73] | 1.04 | −0.17, 0.19 | +0.19, 0.14 |
| PN_kin | −1.38 | 3.04 | 2.68 | −0.93, 6e-41 | −0.57, 1e-18 |

- **ParT_full.** On the fresh rows it is separated from MLPx. On both samples it equals MLPxW (1.00× H23, 1.07× [0.88, 1.28] fresh).
- **Power of the H3v edit.** MLPxW itself is only 1.6–1.8× MLPx under this edit, so H3v at 60 GeV cannot separate W-pair use from side effects much beyond that level.
- **10-class top call, ParT_full rc, E − B.** +2.3 [+0.5, +3.9] pp (McNemar p 0.012). The sign is opposite to the log-odds, as in H23.

100 GeV (n 991): MLPx +0.31 [+0.23, +0.39], MLPxW 0.00 [−0.08, +0.08].

| model | E − B rc [CI] | net − MLPxW | Holm |
|---|---|---|---|
| ParT_full | −0.42 [−0.63, −0.22] | −0.42 | 5e-4 |
| ParT_kin | −0.24 [−0.38, −0.09] | −0.24 | 0.008 |
| PN_full | −0.26 [−0.41, −0.10] | −0.26 | 0.011 |
| PN_kin | +0.01 [−0.11, +0.13] | — | — |

- st and rc agree within 0.02. H23 already gave ParT_full −0.25 [−0.45, −0.04] and ParT_kin −0.15 [−0.29, −0.01].
- "No model responds beyond MLPx at 100 GeV" is excluded for ParT_full, ParT_kin and PN_full. Against the informative references (MLPxW at 0.00, permuted null at −0.04 in H23), these three lower the top score when the W pair is moved to 100 GeV.

### decomp (fresh, T60)

- ParT_full recentred dN is −1.59, against −0.84 for ParT_kin (H23: −1.32 vs −0.62).
- With PID inputs zeroed, dN is −0.19; with tracks zeroed, −0.43.
- This reproduces H23's pattern. The off-distribution caveat stands.

## Competing hypotheses

| hypothesis | prediction | verdict on fresh rows |
|---|---|---|
| (a) side effects only | net W − N = MLPx | excluded at 60 and 100 GeV (rc, all four nets), at 90 GeV, and at 70 GeV for ParT_kin |
| (b) generic sensitivity to rearranging subjets | no ordering in \|mW − 80\| | excluded by the dose control |
| (c) W-mass window | ordering, and ≈ MLPxW | ordering observed. ≈ MLPxW at 60 GeV. 1.3–2.1× MLPxW on the upper side, so the networks penalise mW above 80 GeV more steeply than MLPxW does |
| (d) fixed-axis artefact = pure translation | residual 0 | holds for ParT_kin. Fails for ParT_full at 60/70 GeV and for PN_full/PN_kin, where the edit and the translation do not add |

## Corrected reading

With every edited jet rebuilt about its own E-scheme axis:
- **H4 W-pair edit at fixed m and τ32.**
  - It lowers L_T − L_QCD of the JetClass ParT/PN models by 2.9–4.5× the MLPx side-effect response at 60 GeV (0.8–1.25× MLPxW) and 4.3–6.3× at 100 GeV (1.4–2.1× MLPxW). Ranges combine the H23 rows and the fresh rows.
  - The response is ordered in |mW − 80| (60 > 70, 100 > 90 GeV).
- **Earlier fixed-axis values (H19, and H3v for ParT).** They contain the networks' response to the axis shift that the N control induces.
  - For ParT_kin that response equals the fixed-minus-recentred difference within 0.1 at every target.
  - For ParT_full a residual of up to +0.5 log-odds remains (60 GeV, fresh rows).
  - For PN_full and PN_kin the residual is +0.1 to +1.2.
- **H3v k-means edit at 60 GeV.**
  - ParT_kin and PN_kin give 2.7–3.0× MLPx.
  - ParT_full gives 1.6–1.9×, equal to MLPxW on both samples. It is separated from MLPx on the fresh rows only.
  - PN_full gives 1.0–1.4×.
- **H3v at 100 GeV.** MLPx moves the opposite way (+0.31 to +0.36) and MLPxW does not move. ParT_full, ParT_kin and PN_full lower the top score by 0.15–0.42 on both builds and both samples, beyond MLPxW and the null.
- **Not re-tested here.** The H13 pu arms, the Top-network H4 numbers and the Top fixed-axis translation response.

## Issues

1. The report and thesis note state "at 100 GeV no model responds beyond MLPx (H3v)". Both H23's own summary and the fresh rows contradict this. The ratio to MLPx is not usable when MLPx changes sign. Use MLPxW and the null as the references.
2. "For ParT the translation equals st − rc within ±0.15" does not reproduce for ParT_full at T60 (+0.48 [+0.23, +0.74]). It is specific to ParT_kin.
3. H3v 60 GeV, ParT_full: "not separated from MLPx after Holm" does not reproduce (fresh Holm 1.9e-5). The stable statement is "≈ MLPxW".
4. The Conclusion's "about 4×" should be stated as a range and against MLPxW: about 1× MLPxW at 60 GeV, more at 100 GeV.
5. Figure: the fixed-axis T60 and T100 markers cannot be told apart; the EFN fixed-axis point is off scale; the right-panel legend "(as H19)" is wrong for the Top data.
6. Not checked by rerun: H13 (D10), Top Landscape (D6), T80/Q80.
