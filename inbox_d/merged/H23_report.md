# H23 — Axis-rebuild audit of the earlier W-pair (H3v, H4/H4v, H19) and soft-contamination (H13) edits (claims D1, D6, D10, D11; absorbs H20)

Jobs: d-h23-smj / d-h23-smt / d-h23-smh (smoke), d-h23-audit (JetClass, GPU), d-h23-top (Top Landscape, GPU), d-h23-hls (hls4ml, CPU), d-h23-decomp (dN decomposition, GPU), d-h23-sum and d-h23-fig (Holm across all arms, figure). Script `interp/h23_axis_audit.py` (`--part jc | top | hls | decomp | summary`). Results `results/physics/h23_{jc,top,hls,decomp,summary}.json`, `h23_{jc,top}_perjet.npz`, figure `results/physics/h19_wpair.png` (redrawn; the earlier fixed-axis version is kept as `h19_wpair_fixedaxis.png`).

## Conclusion

When every edited jet is rebuilt about its own energy-weighted (E-scheme) axis, the JetClass-trained ParT and ParticleNet lower their top score when the W-candidate pair mass is moved from 80 to 60 or 100 GeV at fixed jet mass and τ32, by 3.5–6.3 times as much as a reference classifier that sees only the side effects of the edit. The earlier fixed-axis numbers (H19; H3v for ParT) mixed this response with the networks' response to the jet's constituents being shifted relative to the axis. That shift alone accounts for the fixed-minus-recentred difference within ±0.15 log-odds for ParT. So the "JetClass ParT does not use the W-pair mass" part of D1 does not hold for the H4 edit. With H3v's own edit (k-means subjets, all pT), the recentred response at 60 GeV is 1.0–2.8× the reference, and no model responds at 100 GeV. The soft-contamination (H13) results and the Top-trained networks' W-pair results (H4, D6) do not change, because those builds already recomputed each jet's axis. What remains unexplained is why the full-input JetClass models have a larger response to the non-W control than the kinematic-only models. Zeroing the track or particle-ID inputs removes 40–47 points of accuracy, so that test cannot separate the two.

## Procedure

- **Two builds per edit.** st = fixed axis as in the original node (JetClass: stored jet_eta/jet_phi; Top: the unedited jet's E-scheme axis; hls4ml: the stored (0, 0) axis). rc = each edited jet rebuilt about its own E-scheme axis (sum of the constituent four-momenta; JetClass inputs via part_core.build convention η = jet_eta + sign(jet_eta)·deta, through h19_wpair_cross.jc_dict).
- **Translation-only control TN (TB for H3v).** The unedited jet moved rigidly in (η, φ) by the axis shift that the control edit N induces, and nothing else. It is scored on st and on rc. On rc it should be ≈ 0 by construction (consistency check).
- **Samples.**
  - JetClass: TTBar_100 and ZJetsToNuNu_100 rows with 550 < pT < 650 GeV, |η| < 2, ≥ 3 particles. 2,000 Tbqq (default_rng(9523)) + 2,000 QCD (rng 9524) are drawn as random rows (not first-k), with the H19 (rng 4191/4192) and H19v (4391/4392) draws excluded (overlap 0).
  - H13 JetClass arm: 1,000 Wqq + 1,000 Zqq random rows of WToQQ_100 / ZToQQ_100 (rng 9523).
  - Top Landscape test: data_top.sample(2000, 9523), 2,000 t + 2,000 QCD.
  - hls4ml: jedi_core.sample(per_class=1000, seed=9523), N 5,000.
- **Models.**
  - JetClass ParT_full, ParT_kin, PN_full, PN_kin.
  - Top ParT, PN, LorentzNet (four-vectors only, so st = rc), PFN, EFN, and a weight-permuted PFN (null).
  - JEDI-linear official n64 3-feature, and a redrawn-weight JEDI-linear (null).
  - References are retrained as in H19 (25,000 jets per class): MLPx (side effects only: mSD, τ1–3, τ32, τ21, N, subjet pT fractions and masses, no pair mass), MLPxW (MLPx + mW) and LR2(m, mW). They are fitted on JetClass TTBar_101 / ZJetsToNuNu_101 (MLPx_jc) and on the Top val split (MLPx_top).
- **Arms.**
  - H4 (h4_verify edit, exclusive-kT 3 subjets): T60 and T100 (tops with 65 < mW < 95 moved to mW 60 / 100 GeV at fixed m and τ32), T80 (tops 6–15 GeV off 80.4 moved onto it) and Q80 (QCD in the top window moved to 80.4). Controls: N (non-W pair with the same relative mass change, W pair, m and τ32 fixed) and R (rotation).
  - H3v (h3_verify.wshift_all, k-means subjets, all-pT Tbqq): W pair to 60 / 100 GeV at fixed jet mass. Control B: W pair held, non-W pair moved by the same dm. Control A: the original H3 control.
  - H13: pu_ρ diffuse soft particles at ρ = 5 / 20 / 40 GeV per unit η–φ area. Controls: c4 (same particles at r/20, so Δm ≈ 0), c5 (one wide particle with ΣpT and mass matched), c2k (that particle on the axis) and c3 (translation by the pu-induced offset).
  - decomp: dN on rc for ParT_full / PN_full with PID inputs (charge + 5 flags), track inputs (tanh d0, d0err, tanh dz, dzerr) or both set to 0 at the feature level, against the kin models.
- **Metric.** d = s(edit) − s(identity) in log-odds (JetClass: L_Tbqq − L_QCD; Top: top vs QCD). The W-pair effect is W − N (H4) or E − B (H3v). Reported with: 95 % bootstrap CI over jets (B = 2,000); ratio to the MLPx W − N with CI; one-sided sign-flip permutation p of (net − MLPx) in the mW direction; two-sided paired st − rc test; McNemar of W vs N calls at ε_S 0.3 / 0.5 (thresholds from the sample's unedited tops); % calls changed. H13: % of 10-class (JetClass) or binary (Top) calls changed, with McNemar pu vs control and st vs rc. Holm is applied across all 644 primary tests (arms × models × paths); 313 have Holm p < 0.05.
- **Hypotheses.**
  - H0: fixed and recentred builds agree, and the net W − N equals MLPx W − N.
  - H1a (axis artefact): st − rc ≈ TN response, JetClass nets at ≈ 3–5× MLPx after recentring, and H3v's ParT null reverses.
  - H1b: st and rc agree within CI for H4 (Top nets) and H13, so only the JetClass W-pair null needs revision.

## Checks: unedited models on these samples (st build; rc agrees to ≤ 0.1 point)

| sample | model | AUC | accuracy | earlier value |
|---|---|---|---|---|
| JetClass tops/QCD 550–650 GeV (N 4,000) | ParT_full | 0.9985 | binary 98.2 %, 10-class 85.2 % | H19 0.9987 / 86.5 % |
| | ParT_kin | 0.9932 | 95.8 / 79.5 % | H19 0.9925 |
| | PN_full | 0.9978 | 98.0 / 84.3 % | H19 0.9980 |
| | PN_kin | 0.9894 | 94.9 / 78.5 % | H19 0.9890 |
| | Top ParT / PN / LNet / PFN / EFN | 0.9869 / 0.9847 / 0.9875 / 0.9802 / 0.9787 | 94.1 / 93.6 / 94.5 / 92.5 / 93.3 % | H19 Top ParT 0.9852, LNet 0.9870 |
| | permuted PFN (null) | 0.767 | 50.0 % | chance |
| | MLPx_jc / MLPxW_jc / LR2_jc | 0.9779 / 0.9794 / 0.9622 | | |
| Top Landscape (N 4,000) | Top ParT / PN / LNet / PFN / EFN | 0.9853 / 0.9846 / 0.9856 / 0.9765 / 0.9710 | 94.3 / 94.1 / 94.4 / 92.5 / 91.7 % | H4v 0.9846 / 0.9842 / 0.9854 / 0.9785 / 0.9741 |
| | JC ParT_kin / PN_kin | 0.9816 / 0.9823 | 92.8 / 93.1 % | H19v 0.9813 / 0.9838 |
| | permuted PFN (null) | 0.706 | 50.0 % | chance |
| JetClass Wqq + Zqq (N 2,000) | ParT_full / ParT_kin / PN_full / PN_kin | | 10-class 75.0 / 63.7 / 72.5 / 61.3 %; W eff 80.2 / 69.7 / 79.0 / 69.1 % | H13 ParT_full W eff 80 % |
| hls4ml (N 5,000) | JEDI-linear | | 81.5 %, W eff 87.8 % | 81.1–81.3 % |
| | redrawn JEDI-linear (null) | | 20.0 % (constant call) | |

- Stored vs constituent-sum axis on the JetClass sample: median 0.39 mrad, 99th percentile 12.9 mrad.
- Consistency check: dTN on rc is |≤ 0.03| log-odds for every model and set, as it must be.

## Numbers

### H4 W-pair edit on JetClass tops (550–650 GeV); W − N log-odds [95 % CI], × = ratio to MLPx_jc W − N

T60 (n 300 of 1,478 eligible). Axis shift induced: W 0.021, N 0.130 (median). MLPx_jc −0.44 [−0.52, −0.35], MLPxW_jc −1.54, LR2_jc +0.06.

| model | st W − N (×MLPx) | rc W − N (×MLPx) [CI of ratio] | st − rc | dTN on st | residual (dN_st − dN_rc) − dTN | rc McNemar W vs N calls ε_S 0.3 |
|---|---|---|---|---|---|---|
| ParT_full | −0.57 (1.31×) | −1.92 [−2.25, −1.57] (4.39× [3.42, 5.59]) | +1.35 [+1.13, +1.57] | −1.49 | +0.13 [−0.06, +0.33] | 83 vs 6, p 2e-18 |
| ParT_kin | −0.35 (0.80×) | −1.96 (4.48× [3.65, 5.61]) | +1.60 | −1.54 | −0.01 [−0.19, +0.16] | p 2e-19 |
| PN_full | −0.70 (1.61×) | −1.56 (3.57× [2.85, 4.51]) | +0.86 | −2.00 | +0.80 [+0.56, +1.04] | p 3e-15 |
| PN_kin | −0.32 (0.72×) | −1.53 (3.50× [2.84, 4.41]) | +1.21 | −1.72 | +0.36 [+0.20, +0.51] | p 2e-21 |
| Top ParT | −0.14 (0.33×) | −1.28 (2.93×) | +1.14 | −1.08 | +0.28 | p 3e-15 |
| Top PN | +0.33 (−0.75×) | −1.43 (3.28×) | +1.76 | −2.56 | +0.46 | p 4e-8 |
| Top LorentzNet | = rc | −2.75 (6.29× [5.08, 7.97]) | — | — | — | p 1e-23 |
| Top PFN | +0.10 | −0.63 (1.44× [1.13, 1.80]) | +0.73 | −0.44 | −0.03 | p 3e-9 |
| Top EFN | +0.70 | −0.59 (1.34× [0.94, 1.79]) | +1.29 | −0.10 | −0.81 | p 3e-9 |
| permuted PFN | +0.13 | +0.00 [−0.06, +0.06] | +0.13 | −0.04 | −0.10 | p 0.92 |

- **T100 (n 194).** MLPx_jc −0.45 [−0.61, −0.29].
  - rc: ParT_full −2.83 [−3.25, −2.38] (6.25×), ParT_kin 5.84×, PN_full 5.42×, PN_kin 4.33×, Top ParT 3.91×, Top PN 4.43×, LorentzNet 6.50×, PFN 2.18×, EFN 2.59×.
  - st: ParT_full 3.97×, ParT_kin 2.59×, PN_full 4.35×, PN_kin 2.79×.
  - st − rc: +0.48 to +1.47 for the JetClass nets.
- **T80 (n 177; pair moved onto 80.4).** MLPx_jc −0.08, MLPxW +0.03.
  - st: JetClass nets +0.24 to +0.58 (dN −0.62 to −0.76), as H19 reported.
  - rc: +0.02 to +0.15, all CIs include 0.
  - st − rc +0.15 to +0.43 ≈ dTN_st (−0.39 to −0.74).
- **Q80 (n 15).** All CIs span 0 on both builds.
- Under Holm (644 tests), rc W − N vs MLPx is below 0.05 for all four JetClass nets at T60 and T100 (Holm ≤ 3e-7). st − rc is below 0.05 for 7 of 8 JetClass-net cells (PN_full T100 Holm 0.051).

### H4 on Top Landscape tops (n 215 T60, 163 T100, 177 T80, 14 Q80)

- **rc (this build matches the original H4/H4v build, h4_top_wmass.jet_inputs, which recomputes each jet's axis).**
  - T60: Top ParT −1.73 [−2.03, −1.43] (2.61× MLPx_top) vs H4v −1.79 [−1.98, −1.60]; Top PN 2.95×, LorentzNet 4.30×, PFN 1.15× [0.90, 1.45], EFN 0.85× [0.58, 1.14].
  - JetClass ParT_kin 3.94× [3.25, 4.81], PN_kin 3.26× (H19v fresh tops 3.76×, 3.11×).
  - T100: Top ParT 2.62×, PN 3.22×, LNet 4.06×, PFN 1.46×, EFN 1.59×; JC ParT_kin 5.48×, PN_kin 4.20×.
- **st (fixed axis, constructed here only for comparison).** Top ParT 0.45×, Top PN −0.62×, JC ParT_kin 0.29× at T60.
  - dTN_st: Top ParT −1.29, Top PN −2.85, JC ParT_kin −1.71.
  - So on a fixed axis the Top-trained ParT and PN respond to a pure translation as much as the JetClass nets do.

### H3v W-pair edit (k-means subjets, all-pT Tbqq; 60 GeV n 1,061, 100 GeV n 989); E − B log-odds, × MLPx_jc

60 GeV: jet mass fixed at 167.4 GeV, W pair 79.2 → 60.0 (E). Control B keeps W at 79.4 and moves the non-W pair to 109.2. Axis shift E 2.9, B 18.3 mrad. MLPx_jc −0.46 [−0.53, −0.37], MLPxW_jc −0.72, LR2_jc −0.35.

| model | st E − B (×MLPx) | rc E − B (×MLPx) [CI] | st − rc | residual − dTB | 10-class top call E − B (pp), st / rc (McNemar Holm) |
|---|---|---|---|---|---|
| ParT_full | −0.34 [−0.54, −0.14] (0.75×) | −0.72 [−0.89, −0.55] (1.58× [1.20, 2.04]; vs MLPx Holm 0.40) | +0.38 [+0.27, +0.50] | +0.08 | +3.1 [1.2, 5.1] (0.84) / +1.3 [−0.5, 3.1] (1) |
| ParT_kin | −0.93 (2.05×) | −1.26 (2.76× [2.34, 3.36]; Holm 3e-23) | +0.32 | +0.04 | +5.0 (0.015) / +1.2 (1) |
| PN_full | +0.06 (−0.14×) | −0.47 (1.04× [0.74, 1.39]; Holm 1) | +0.54 | +0.10 | +3.6 (0.11) / +0.8 (1) |
| PN_kin | −0.69 (1.52×) | −1.22 (2.68×; Holm 4e-29) | +0.53 | +0.02 | +6.5 (6e-5) / +2.5 (1) |
| Top ParT / PN / LNet / PFN / EFN (rc) | | 2.52× / 4.17× / 5.40× / 1.66× / 2.56× | +0.25 to +0.94 | −0.06 to −0.04 | |

- At 60 GeV, the ParT_full 10-class top call under E stays above that under B on both builds (st 93.2 vs 90.1 %). This is the opposite sign to the log-odds L_T − L_QCD.
- 100 GeV: W pair 82.6 → 100.0. MLPx_jc goes up, +0.36 [+0.28, +0.44] (LR2 +0.53), whereas MLPxW −0.04.
  - JetClass nets E − B: ParT_full −0.25 [−0.45, −0.04], ParT_kin −0.15, PN_full −0.07, PN_kin +0.07, on both builds.
  - st − rc ≤ 0.02 (axis shifts 3.3 / 8.3 mrad).
  - 10-class McNemar Holm = 1 for all.

### decomp: JetClass full models' residual recentred dN (T60 n 300, T100 n 194)

| model, inputs | unedited AUC / binary acc | T60 dW | T60 dN [CI] | dN/dW | T100 dN [CI] | dTN (rc) |
|---|---|---|---|---|---|---|
| ParT_full, all | 0.9985 / 98.2 % | −3.24 | −1.32 [−1.62, −1.01] | 0.41 | −0.76 [−1.04, −0.47] | −0.003 |
| ParT_full, PID = 0 | 0.864 / 57.7 % | −0.15 | −0.08 [−0.20, +0.04] | 0.52 | +0.01 | −0.012 |
| ParT_full, tracks = 0 | 0.852 / 51.8 % | −0.38 | −0.36 [−0.49, −0.23] | 0.94 | −0.10 | +0.014 |
| ParT_full, both = 0 | 0.786 / 51.8 % | −0.26 | −0.13 [−0.25, −0.01] | 0.51 | +0.15 | −0.020 |
| PN_full, all / PID 0 / tracks 0 / both 0 | 0.998 / 0.945 / 0.820 / 0.848 | −2.57 / −0.89 / −1.98 / −0.73 | −1.01 / −0.26 / −0.54 / −0.04 | 0.39 / 0.29 / 0.27 / 0.05 | −0.35 / −0.17 / +0.43 / +0.13 | ≤ 0.03 |
| ParT_kin / PN_kin | 0.993 / 0.989 | −2.58 / −1.94 | −0.62 / −0.41 | 0.24 / 0.21 | −0.37 / −0.13 | ≤ 0.02 |

- The translation part of the recentred dN is ≤ 0.02 log-odds.
- Zeroing input groups removes 40–47 points of binary accuracy for ParT_full, and all responses shrink with it. The ratio dN/dW does not move to the kin-model value (ParT_full 0.52 / 0.94 / 0.51 vs ParT_kin 0.24). PN_full with both groups zeroed gives dN/dW 0.05 at T60.

### H13 soft contamination: % calls changed, st / rc

| sample, model | pu_5 | pu_20 | pu_40 | c4_40 | c5_20 | c5_40 | c3_40 (translation) |
|---|---|---|---|---|---|---|---|
| JetClass W/Z, ParT_full | 13.6 / 13.8 | 47.0 / 46.9 | 70.0 / 70.2 | 87.9 / 87.9 | 24.9 / 28.7 | 20.2 / 24.8 | 2.5 |
| ParT_kin | 15.8 / 16.8 | 55.9 / 56.9 | 82.2 / 82.3 | 47.9 / 48.2 | 65.6 / 68.3 | 48.4 / 51.4 | 5.0 |
| PN_full | 13.7 / 13.4 | 55.4 / 55.9 | 85.0 / 84.8 | 82.1 / 82.0 | 30.6 / 36.1 | 26.7 / 36.0 | 2.6 |
| PN_kin | 19.4 / 19.2 | 67.0 / 66.8 | 90.8 / 91.0 | 94.3 / 94.3 | 70.0 / 74.4 | 59.6 / 59.9 | 3.1 |
| Top, ParT / PN / LNet / PFN / EFN (pu_40) | | | 24.6/24.6, 25.2/25.2, 25.1/25.1, 25.4/25.4, 23.7/23.9 | | | EFN 31.9 / 19.1 | 0.0–1.9 |
| hls4ml, JEDI-linear | 23.9 / 24.3 | 66.9 / 67.2 | 68.8 / 69.0 | 24.3 / 24.3 | 40.1 / 43.4 | 50.0 / 54.3 | 2.3 |

- Induced axis offset (median): pu ρ 5 → 40 is 3.6 → 13.6 mrad (JetClass), 2.5 → 12.0 (Top), 2.5 → 5.1 (hls4ml); c5 at ρ 40 is 70 mrad; c4 and c2k are ≤ 0.5 mrad.
- st vs rc McNemar under Holm: no pu, c4 or c2k cell differs for any trained network. The exception is the permuted PFN null at ρ 20/40 (≤ 1.5 pp). c5 differs at ρ 20/40 for 4/4 JetClass nets, 3/5 Top nets and JEDI-linear (by 3–13 pp).
- The pu-vs-control orderings that H13v reported are reproduced on both builds: ParT_full, PN_full and LorentzNet change more calls under c4 than under pu, and ParT_kin, PN_kin and JEDI-linear change fewer (pu vs c4 McNemar Holm < 0.05 on both builds).

## Hypotheses: what is excluded

- **H0 is excluded** for the JetClass nets and the Top ParT/PN under the fixed-axis build of H4 and H3v_60. st − rc is +0.86 to +1.76 at T60 and +0.25 to +0.94 for H3v_60 (Holm < 1e-5 for most cells).
- **H1a**:
  - Predicted "st − rc ≈ TN response": holds for ParT_full, ParT_kin and Top PFN (residual |≤ 0.17|, CI includes 0 or is near 0) and for H3v_60 (residual 0.02–0.10). For PN_full, PN_kin and Top PN the residual is +0.36 to +0.80, and for EFN it is −0.81: the translation and the edit do not add for these models.
  - Predicted "JetClass nets 3–5× MLPx after recentring": holds for the H4 edit (T60 3.5–4.5×, T100 4.3–6.3×). With the H3v edit, ParT_kin and PN_kin are at 2.7–2.8× and ParT_full at 1.58× (vs MLPx Holm 0.40), with PN_full at 1.04×.
  - Predicted "H3v's ParT null reverses": in log-odds, rc ParT_full E − B is −0.72 [−0.89, −0.55] (≠ 0) but not separated from MLPx after Holm. Its 10-class top call shows no E vs B change on either build after Holm.
- **H1b**: holds for H13 (pu, c4 and c2k arms; st ≈ rc within 1.5 pp) and for the Top nets on Top data in the sense that H4/H4v already used rc (H4v ParT −1.79 vs here −1.73). It is excluded as a statement that st and rc agree for Top ParT/PN, since they differ by 1.1–2.4 log-odds when their inputs are built on a fixed axis.

## Reading (scoped to what was measured)

- On 550–650 GeV JetClass tops rebuilt about their own axis, the H4 W-pair edit (mW 80 → 60 / 100 GeV at fixed m, τ32) lowers L_T − L_QCD of all four JetClass nets more than the non-W control N. The excess is 3.5–4.5× (T60) and 4.3–6.3× (T100) the MLPx side-effect reference, and 1.0–1.25× the MLPxW reference (which sees mW). At ε_S 0.3 it changes 33–37 % of top calls vs 17–18 % for N.
- The fixed-axis numbers of H19 (0.7–1.6× at T60) and the T80 positive values come from the N control's 0.13 axis shift. Translating the unedited jet by that shift alone gives −1.5 to −2.0 log-odds.
- D1's statement that JetClass ParT has no W-pair response beyond a W-fixed control rested on a fixed-axis build. With H3v's own edit rebuilt about each jet's axis, ParT_kin and PN_kin respond at 2.7× MLPx, ParT_full at 1.6× (not separated from MLPx after Holm) and PN_full at 1.0×. At 100 GeV the H3v edit gives no model response beyond MLPx, and MLPx moves in the opposite direction there.
- D6 (Top-trained nets) and D10 (H13) are unaffected by the axis build: their original builds recomputed each jet's axis, and recentring changes pu-arm call rates by ≤ 1.5 pp.
- The full-input JetClass models' larger dN (−1.32 vs −0.62 for ParT at T60) is not translation (≤ 0.02). Zeroing track or PID inputs at the feature level degrades the models so far (AUC 0.79–0.86) that the measurement does not attribute it to either group.

## Issues

- The Top Landscape sample does not exclude earlier H4/H4v/H19 rows (data_top.sample(2000, 9523)). The JetClass sample does exclude the H19/H19v draws.
- Q80 analysis sets are small (15 and 14 jets), and all their CIs span 0.
- The redrawn JEDI-linear null outputs a constant call (accuracy 20 %), so its 0 % calls-changed is degenerate. The permuted PFN null sits at 50 % accuracy with scores near threshold, so it changes up to 62 % of calls under pu_40. Its W − N is 0 on rc.
- The decomp arm uses feature-level zeroing, which is off-distribution. A retrained "full minus one group" model would be needed to attribute the residual dN.
- H3v at 60 GeV: the ParT 10-class top-call metric (H3v's original metric) and the binary log-odds disagree in sign for E − B. Both are reported.
- Holm across all 644 tests (H4, H3v, decomp, H13): 313 have Holm p < 0.05.

## Explainer entry (for site/explain.py; to be merged by the ledger)
```python
H23_EXPLAINER = [
 ('Question', 'Earlier W-pair edits (H3v, H19) built the network inputs of each edited jet about the axis of the unedited jet. The edit itself moves the energy-weighted jet axis, so the networks saw the edit plus a shift of every constituent relative to the axis. Do the earlier W-pair and soft-contamination conclusions hold when every edited jet is rebuilt about its own axis?', None),
 ('Physics background',
  'ParT and ParticleNet read constituent coordinates relative to the jet axis, $\\Delta\\eta_i=(\\eta_i-\\eta_J)\\,\\mathrm{sign}(\\eta_J)$, $\\Delta\\phi_i=\\phi_i-\\phi_J$, where the E-scheme axis is the direction of $\\sum_i p_i$. '
  'Moving two subjets apart at fixed jet mass changes $\\sum_i p_i$ only slightly, but a non-W pair moved by the same relative amount shifts the axis by $\\Delta R\\approx0.13$ (median). If the axis is not recomputed, every $\\Delta\\eta_i,\\Delta\\phi_i$ is translated by that amount.', None),
 ('Method',
  'Each edit is scored twice: fixed axis (as originally) and recentred (axis recomputed from the edited constituents). The translation-only control TN moves the unedited jet rigidly by the axis shift the non-W control induces. '
  'W-pair metric: $\\Delta_W-\\Delta_N$ in top log-odds ($\\Delta_k=s_k-s_{\\rm id}$), compared with a side-effect classifier MLPx (jet mass, $\\tau_{1..3}$, multiplicity, subjet masses and $p_T$ fractions, no pair mass). '
  'Samples: 2,000 + 2,000 JetClass tops/QCD at 550–650 GeV and 2,000 + 2,000 Top Landscape jets (seed 9523); 2,000 JetClass W/Z and 5,000 hls4ml jets for soft contamination. Bootstrap CIs over jets, sign-flip permutation tests, Holm over 644 tests.', None),
 ('Result',
  'JetClass ParT_full at $m_W\\to60$ GeV: $\\Delta_W-\\Delta_N=-0.57$ (1.3× MLPx) on the fixed axis, $-1.92\\,[-2.25,-1.57]$ (4.4× [3.4, 5.6]) recentred. The difference $+1.35$ equals the translation-only response ($-1.49$) to within $0.13\\,[-0.06,0.33]$. '
  'All four JetClass networks: 3.5–4.5× at 60 GeV and 4.3–6.3× at 100 GeV when recentred. Top-trained networks on Top data were already built recentred; their numbers reproduce H4v. Soft-contamination call changes differ by ≤ 1.5 points between builds.', 'results/physics/h19_wpair.png'),
 ('Checks',
  'Unedited AUCs reproduce earlier values (ParT_full 0.9985, Top ParT 0.9853, JEDI-linear 81.5 %). The translation control on the recentred path is $|\\Delta|\\le0.03$, as required. The original H3v edit (k-means subjets) gives 1.0–2.8× MLPx when recentred, and nothing beyond MLPx at 100 GeV. A permuted-weight PFN gives 0.00 [−0.06, 0.06].', None),
 ('Physical significance',
  'JetClass-trained ParT and ParticleNet use the mass of the W-candidate subjet pair inside a top jet, as Top-trained taggers do. The earlier null result came from an input build that also translated the jet relative to its axis. These networks respond strongly to such translations, so any edit that moves the jet axis must be rebuilt about the new axis before its effect is attributed to a physical quantity.', None),
]
```
Entry line for the `'d'` list: `('Result explained: W-pair edits rebuilt about each jet\'s own axis (claims D1, D11; H23)', H23_EXPLAINER, False)`.
