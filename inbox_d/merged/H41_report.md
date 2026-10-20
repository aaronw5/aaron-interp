# H41: colour flow (jet pull) in boosted W/Z/H vs QCD (new claim D17, exploratory)

Job: d-h41-run (GPU, V100, 741 s). Smoke: d-h41-smoke2 (200 jets per class). An earlier smoke job, d-h41-smoke, was lost to a node fault ("GPU is lost") before it started. Script: `interp/h41_colour_flow.py`. Results: `results/physics/h41_colour_flow.json`, `h41_perjet.npz` and figure `h41_colour_flow.png`; log in `logs/d-h41-run.log`.

## Conclusion

In W, Z and H→bb jets, every JetClass tagger (ParT and ParticleNet, full and kinematic inputs) lowered the signal-vs-QCD score when soft particles were mirrored about their prong along the prong–prong line. This mirror moves radiation that sat between the two prongs to the outside. The comparison edits were the same mirror across that line and a random rotation of equal size. With jet mass held within 1 GeV and τ21 within 0.01, the difference to the across-the-line mirror was −0.15 to −0.69 in log-odds. A network with permuted weights stayed within ±0.06, and a classifier that sees only mass, τ21 and multiplicity stayed near 0. A classifier that also sees the pull angles and the between-prong energy fraction moved in the same direction. With the looser match (mass within 3 GeV, τ21 within 0.03) the networks' change was 1.1–7 times the classifier's. The networks therefore respond to where soft radiation sits relative to the two prongs, which is the radiation pattern that colour connection sets. The reverse effect on QCD jets (a rise in signal score) was not resolved: +0.06 to +0.17 with intervals that include 0. Only 3–6 % of jets pass the strict mass/τ21 match, and other angular observables that the mirror changes (for example energy-correlation ratios such as D2) were not held fixed.

(Pull angle: angle between a subjet's pT-weighted radiation vector and the direction to the other subjet. f_btw: fraction of the soft pT that lies in the strip between the two prongs.)

## Procedure

- **Sample.** JetClass test files WToQQ_100, ZToQQ_100, HToBB_100 and ZJetsToNuNu_100. For each class, 2,000 random rows (default_rng 4101–4104, never first-k) with 500 < pT < 1000 GeV, |η| < 2 and ≥ 3 particles: N = 8,000, median pT 587 GeV. Secondary arm: Top Landscape test, 2,000 t + 2,000 QCD (h4_top_wmass._sample, seed 4105).
- **Geometry.**
  - JetClass: two exclusive-kT subjets (E-scheme); prong centres c_k are the (η, φ) of the subjet sums.
  - Top arm: three exclusive-kT subjets. The prongs are the pair with mass closest to 80.4 GeV. Particles nearest the third subjet are not edited.
  - Soft particle: pT < 2 % of the jet pT, ΔR > 0.05 from every subjet centre, and nearest centre is one of the two prongs. On average there are 18.0 per jet, carrying 11.9 % of the jet pT.
  - Frame per prong: û points toward the other prong; v̂ is perpendicular to it.
- **Edits.** Only soft particles move. pT, constituent mass, N, ΣpT and the distance to the nearest prong are unchanged.
  - M: mirror along û (u_rel → −u_rel).
  - C1: mirror along v̂ (v_rel → −v_rel).
  - C2: rotation about the nearest prong. The angle makes the displacement equal to the M displacement of a randomly chosen soft particle of the same jet; the sign is random.
  - Mean displacement: M 0.227, C1 0.268, C2 0.167. Soft particles moved beyond ΔR = 0.8: 1.2 % (M), 1.0 % (C1), 1.4 % (C2).
  - Mx / C1x / C2x: the same edits with one soft-four-vector scale per jet, so that m_inv equals the unedited value (residual ≤ 2×10⁻⁴ GeV; success 99.98 %; Mx scale median 1.017, 5–95 % range 0.65–1.51).
  - TN: the unedited jet translated by the axis shift that M induces (median 4.8 mrad).
  - Every variant is rebuilt about its own E-scheme axis (rc). M is also scored about the unedited axis (M_st).
- **Sets.**
  - matched (primary): |Δm| < 1 GeV and |Δτ21| < 0.01 for each of M, C1 and C2. N: W 119, Z 101, H 72, QCD 63. In the Top arm |Δτ32| < 0.01 is also required (t 38, QCD 19).
  - matched_loose: < 3 GeV and < 0.03. N: W 754, Z 737, H 604, QCD 447 (Top arm: t 537, QCD 287).
  - mass_exact: Mx − C1x and Mx − C2x on all jets (≈ 2,000 per class). τ21 is not held: mean |Δτ21| is 0.025 (Mx), 0.012 (C1x) and 0.020 (C2x).
  - all: every jet with ≥ 1 soft particle, plus an OLS intercept of the per-jet contrast on (Δm, Δτ21) between the two edits.
- **Models.**
  - JetClass: ParT_full, ParT_kin, PN_full, PN_kin. Score s_c = L_c − L_QCD for c = W, Z, Hbb.
  - Null: ParT_full with every weight tensor permuted (seed 5, as in H35).
  - Top arm: Top Landscape ParT, PN, LorentzNet, PFN and EFN, with a weight-permuted PFN as the null.
- **References.** Trained on train_100M `<class>_000` vs ZJetsToNuNu_000: 10,000 random rows per class (rng 4111–4114), 80/20 split.
  - LR3: logistic regression on (m, m², τ21, N).
  - GB3: HistGradientBoosting on (m, τ21, N).
  - LR5 / GB5: the same plus (θ_p1, θ_p2, f_btw).
  - Held-out AUC GB3 / GB5: W 0.922 / 0.927, Z 0.915 / 0.921, H 0.936 / 0.938.
- **Statistics.**
  - Per-jet contrasts M − C1 and M − C2: mean, 95 % bootstrap CI (B = 2,000) and a two-sided sign-flip permutation p (5,000).
  - Holm over 6 (sample, score) cells × 4 networks × 2 contrasts = 48 tests on the matched set.
  - A second family (48 tests): net minus GB3 per jet, with H0 "the response equals the (m, τ21, N) prediction".
  - McNemar test of M vs C1 calls at ε_S = 0.5 (threshold = median s_c on unedited class-c jets).
  - C2ST: HGB, 5-fold AUC; edited jets from one random half against unedited jets from the other half; with and without (θ_p1, θ_p2, f_btw).
- **Hypotheses.**
  - H0: M − C1 lies within the permuted-network band and equals the LR/GB(m, τ21, N) prediction.
  - H1: W/Z/H scores fall when between-prong radiation moves outside, and QCD scores rise under the same mirror.
  - Competing predictions:
    - colour-flow use: the sign of LR5/GB5 on both signal and QCD jets;
    - generic radial or shape reading: M ≈ C2;
    - axis-translation artefact: M ≈ TN.

## Checks

| model | 10-class acc. (this 4-class sample, rc / st) | AUC W / Z / H vs QCD |
|---|---|---|
| ParT_full | 79.8 / 79.8 % | 0.975 / 0.977 / 0.999 |
| ParT_kin | 65.6 / 65.6 % | 0.967 / 0.967 / 0.985 |
| PN_full | 77.8 / 77.8 % | 0.969 / 0.972 / 0.999 |
| PN_kin | 60.7 / 60.6 % | 0.961 / 0.960 / 0.973 |
| ParT_perm | 0.04 % | 0.52 / 0.49 / 0.45 |

- Reference values: H23 W+Z jets at 550–650 GeV gave 75.0 / 63.7 / 72.5 / 61.3 %. This sample also contains H→bb and QCD jets, so the accuracies are higher.
- Raw read_root input vs identity rebuild (ParT_full): accuracy 79.84 vs 79.81 %; maximum |Δlogit| over all jets and classes 1.79.
- Top arm, unedited AUC: ParT 0.9853, PN 0.9840, LorentzNet 0.9863, PFN 0.9791, EFN 0.9745, permuted PFN 0.712. The H23 values were 0.9853 / 0.9846 / 0.9856 / 0.9765 / 0.9710.
- Debug: one fix after the smoke run. With the soft pT scaled to 0, the particles kept their rest mass, so the Mx bracket failed for 24 % of jets. Fixed by scaling the whole soft four-vector, with the network masses scaled to match.

### Effect of the edits on the observables (all jets, mean change)

| sample | unedited f_btw / θ_p1 [rad] | M: Δm [GeV], Δτ21, Δf_btw, Δθ_p1 | C1: Δm, Δτ21, Δf_btw, Δθ_p1 |
|---|---|---|---|
| W | 0.50 / 1.66 | −0.80, −0.015, −0.181, +0.107 | −0.31, −0.003, +0.006, +0.018 |
| H→bb | — | +0.26, −0.012, −0.121, +0.171 | −0.46, −0.002, +0.009, −0.008 |
| QCD | 0.28 / 2.06 | −2.73, −0.018, −0.056, −0.037 | −0.44, −0.002, −0.001, +0.049 |

M lowers f_btw and turns the pull of W/H jets away from the other prong. On QCD jets M changes f_btw and θ_p1 much less, and it changes the mass more (−2.7 GeV).

## Numbers (Δ log p(c)/p(QCD); mean [95 % CI])

### Signal jets, M − C1

| set | sample (score) | ParT_full | ParT_kin | PN_full | PN_kin | ParT_perm | GB3 | GB5 |
|---|---|---|---|---|---|---|---|---|
| matched | W (n 119) | −0.34 [−0.53, −0.14] Holm 0.028 | −0.23 [−0.39, −0.09] 0.13 | −0.30 [−0.46, −0.14] 0.019 | −0.22 [−0.35, −0.10] 0.059 | −0.04 [−0.10, +0.04] | +0.02 [−0.02, +0.07] | −0.18 [−0.35, −0.01] |
| | Z (n 101) | −0.32 [−0.56, −0.10] 0.22 | −0.23 [−0.40, −0.06] 0.33 | −0.19 [−0.35, −0.02] 1 | −0.15 [−0.29, −0.02] 1 | +0.00 [−0.07, +0.07] | −0.04 [−0.11, +0.02] | −0.28 [−0.43, −0.13] |
| | H→bb (n 72) | −0.69 [−1.13, −0.29] 0.043 | −0.26 [−0.47, −0.04] 0.82 | −0.35 [−0.64, −0.04] 0.82 | −0.17 [−0.34, −0.01] 1 | −0.02 [−0.10, +0.06] | +0.03 [−0.03, +0.09] | −0.07 [−0.16, +0.04] |
| matched_loose | W (n 754) | −0.50 [−0.59, −0.39] | −0.42 [−0.49, −0.34] | −0.42 [−0.50, −0.34] | −0.33 [−0.40, −0.26] | −0.02 [−0.04, +0.01] | +0.04 [+0.01, +0.06] | −0.26 [−0.31, −0.20] |
| | Z (n 737) | −0.70 [−0.82, −0.59] | −0.45 [−0.54, −0.37] | −0.44 [−0.53, −0.36] | −0.27 [−0.34, −0.20] | −0.01 [−0.03, +0.02] | −0.00 [−0.03, +0.03] | −0.24 [−0.30, −0.18] |
| | H→bb (n 604) | −0.96 [−1.09, −0.82] | −0.40 [−0.48, −0.31] | −0.52 [−0.62, −0.41] | −0.19 [−0.26, −0.12] | +0.01 [−0.02, +0.04] | +0.02 [−0.01, +0.06] | −0.13 [−0.17, −0.10] |
| mass_exact (Mx − C1x) | W | −0.78 [−0.86, −0.71] | −0.70 | −0.61 | −0.41 | −0.00 | +0.07 | −0.21 |
| | Z | −0.91 [−0.99, −0.84] | −0.58 | −0.62 | −0.37 | −0.01 | +0.07 | −0.23 |
| | H→bb | −1.28 [−1.37, −1.19] | −0.47 | −0.79 | −0.33 | +0.00 | +0.09 | −0.14 |
| all, intercept at zero (Δm, Δτ21) | W / Z / H | −0.78 / −0.87 / −1.23 | −0.62 / −0.58 / −0.61 | −0.65 / −0.60 / −0.74 | −0.52 / −0.43 / −0.40 | 0.00 / −0.01 / 0.00 | −0.06 / −0.05 / −0.06 | −0.35 / −0.32 / −0.23 |

- matched_loose: permutation p ≤ 2×10⁻⁴ in all 12 network cells of the table, and also for M − C2.
- Holm, matched set (48 tests): 6 tests have Holm p < 0.05, all on signal jets: ParT_full W and H (M − C1), PN_full W (M − C1, M − C2), PN_kin W (M − C2) and ParT_full H (M − C2, Holm 0.010).
- Net − GB3 family: 5 of 48 tests have Holm p < 0.05 (W: ParT_full, PN_full ×2, PN_kin; H: ParT_full M − C2 −0.63).

### Signal jets, M − C2 (matched)

| sample | ParT_full | ParT_kin | PN_full | PN_kin | perm |
|---|---|---|---|---|---|
| W | −0.22 [−0.37, −0.08] | −0.15 | −0.24 [−0.37, −0.11] | −0.18 | −0.06 [−0.12, +0.01] |
| Z | −0.27 [−0.49, −0.08] | −0.13 | −0.18 | −0.09 | +0.04 |
| H→bb | −0.62 [−0.93, −0.34] | −0.27 | −0.28 | −0.11 | −0.03 |

In matched_loose, M − C2 for the networks ranges from −0.07 to −0.63, with p ≤ 0.016 in every cell.

### QCD jets, M − C1 (signal-score s_W / s_Z / s_H)

- matched (n 63): +0.06 to +0.17 for every network and score; every CI includes 0; Holm p = 1. GB5: +0.09 / +0.12 / +0.01.
- matched_loose (n 447): −0.06 to +0.03 for every network; permuted network −0.02 to +0.04; GB5 +0.02 / −0.04 / −0.05.
- mass_exact:
  - Mx − C1x: full models −0.03 to +0.04; kin models +0.05 to +0.32. Mx − C2x: +0.13 to +0.29 for every network.
  - GB3 +0.02, GB5 +0.03 to +0.08.
- all-jet intercept: networks −0.08 to −0.15. The m/τ21/N references give −0.14 to −0.22 under the same regression, so on QCD jets the intercept does not remove the mass side effect: M moves the mass by −2.7 GeV and the reference responses are nonlinear.

### Axis and translation controls (matched)

- TN (translation only, rc): |Δ| ≤ 0.03 for every network, against M of −0.24 to −1.09 on signal jets.
- M_st − M (scored about the unedited axis vs the recomputed axis): |Δ| ≤ 0.013 for every network.

### Calls at ε_S = 0.5

- All jets, signal efficiency under M vs C1:
  - ParT_full: W 32.2 vs 43.8 % (McNemar p = 2×10⁻³⁵), Z 28.8 vs 43.5 %, H→bb 15.4 vs 39.5 % (p = 3×10⁻⁹³).
  - PN_kin: W 36.9 vs 46.9 %.
  - Permuted network: 50.7 vs 50.6 %.
- Matched set: 1 of 12 network cells has McNemar p < 0.05 (ParT_kin Z, p = 0.012); the n are small.
- QCD mistag at this threshold is 0–0.6 %, which leaves too few calls to test.

### C2ST (edited vs natural jets, AUC; same-sample split gives 0.48–0.51)

| edit | W / Z / H→bb / QCD, all observables | without θ_p1, θ_p2, f_btw |
|---|---|---|
| M | 0.78 / 0.76 / 0.77 / 0.63 | 0.56 / 0.55 / 0.56 / 0.55 |
| Mx | 0.78 / 0.74 / 0.76 / 0.62 | 0.55 / 0.53 / 0.54 / 0.53 |
| C1 | 0.53 / 0.50 / 0.51 / 0.55 | 0.52 / 0.48 / 0.49 / 0.53 |
| C2 | 0.61 / 0.59 / 0.60 / 0.52 | 0.52 / 0.51 / 0.52 / 0.51 |

### Top Landscape arm (W pair inside top jets), M − C1

- matched (t 38, QCD 19): no test has Holm p < 0.05 (20 tests).
- matched_loose, tops (n 537):
  - ParT −0.18 [−0.25, −0.11], PN −0.10 [−0.18, −0.02], LorentzNet −0.25 [−0.33, −0.17];
  - PFN −0.00 [−0.03, +0.03], EFN +0.03 [−0.01, +0.06], permuted PFN −0.01 [−0.03, +0.01].
- matched_loose, QCD (n 287): −0.01 to +0.02 for every model.
- all-jet intercept, tops: ParT −0.23 [−0.28, −0.18], PN −0.20, LorentzNet −0.29, PFN −0.06 (Holm 0.01 each), EFN +0.00.
- No pull reference was trained for the Top arm.

## Reading

- **Against H0, on signal jets.** GB3, which sees only (m, τ21, N), stays at −0.06 to +0.09 in every set. The permuted network stays within ±0.06. All four JetClass networks lower W, Z and H→bb scores under M relative to C1 and to C2: by −0.15 to −0.69 in the strict matched set and by −0.19 to −1.28 in the larger sets.
- **Competing predictions.**
  - Axis-translation artefact (M ≈ TN): excluded, since TN ≈ 0 and M_st ≈ M.
  - Generic radial or shape reading (M ≈ C2): excluded on signal jets (M − C2 < 0, p ≤ 0.016 in matched_loose).
  - The sign agrees with GB5/LR5, which have pull and f_btw. Network-to-GB5 ratio of M − C1: 1.1–7.2 in matched_loose, 1.6–9 in mass_exact, 0.5–10 in the strict matched set (Z: GB5 −0.28 exceeds PN_full −0.19 and PN_kin −0.15). Examples: W matched_loose ParT_full −0.50 vs GB5 −0.26; H→bb mass_exact ParT_full −1.28 vs −0.14.
- **QCD direction.** In the strict and loose sets the QCD rise is not resolved. In the mass-exact set it appears against C2 and, for the kinematic-only models, against C1. GB5 also predicts only +0.03 to +0.12 on QCD, so the expected effect is small. The weaker change in f_btw and θ_p1 on QCD (−0.06, −0.04 rad vs −0.18, +0.11 rad on W) limits this test.
- **Open confounds.**
  - M changes angular structure beyond pull and f_btw: the soft particles' distances to the other prong change, which moves energy-correlation ratios (C2, D2) and τ32. None of these was in the references or held fixed. Part of the network response could therefore come through those observables rather than colour flow as such.
  - C2ST without the colour-flow observables gives AUC 0.53–0.56 for M, so a small residual change in the remaining observables exists.
  - "Soft" here means pT < 2 % of the jet pT, which includes about 12 % of the jet pT.
- **Top arm.** ParT, PN and LorentzNet show the same direction on the W pair inside top jets (−0.10 to −0.25 in the loose set). PFN and EFN show none (|Δ| ≤ 0.03).

## Explainer entry (for site/explain.py, H41_EXPLAINER)

- **Question.** A W, Z or H boson is a colour singlet: its two decay quarks are colour-connected to each other, so soft gluon radiation fills the region between them. A QCD jet that splits into two prongs is colour-connected to the rest of the event, and its soft radiation is spread more outside. Do jet taggers use this pattern beyond the jet mass, $\tau_{21}$ and multiplicity?
- **Physics background.** The jet pull vector of subjet $k$ is $\vec t_k=\sum_{i\in k}\frac{p_{T,i}|\vec r_i|}{p_{T,k}}\vec r_i$ with $\vec r_i$ relative to the subjet axis. The pull angle $\theta_p$ is the angle between $\vec t_k$ and the direction to the other subjet. For colour singlets $\theta_p$ tends toward 0. Here the unedited means are $\theta_{p1}=1.66$ rad for W and $2.06$ for QCD, and the between-prong soft-pT fraction is $f_{\rm btw}=0.50$ for W vs $0.28$ for QCD.
- **Method.** JetClass test jets, 500–1000 GeV, 2,000 each of W, Z, H→bb and QCD (random rows, seed 4101). Two exclusive-$k_T$ subjets. Soft particles ($p_T<0.02\,p_{T,J}$, $\Delta R>0.05$ from both prongs) are edited about their nearest prong:
  - M mirrors them along the prong–prong line, $u\to-u$, which moves between-prong radiation outside;
  - C1 mirrors them across that line, $v\to-v$;
  - C2 rotates them by a random angle of equal displacement.
  Every edited jet is rebuilt about its own E-scheme axis. Analysis on jets with $|\Delta m|<1$ GeV and $|\Delta\tau_{21}|<0.01$, plus a looser set, a mass-exact variant and a regression to zero mismatch. Metric: $\Delta\log p(c)/p({\rm QCD})$, $M-C1$ per jet, bootstrap CI, sign-flip permutation, Holm over 48 tests. Nulls: permuted-weight ParT and a BDT on $(m,\tau_{21},N)$; positive reference: a BDT that also sees $\theta_{p1},\theta_{p2},f_{\rm btw}$.
- **Result.** Matched jets (n 72–119): $M-C1=-0.34$ (W), $-0.32$ (Z), $-0.69$ (H) for ParT_full; ParticleNet and the kinematic-only models range from $-0.15$ to $-0.35$. Permuted network within $\pm0.06$; $(m,\tau_{21},N)$ BDT $\approx0$; pull BDT $-0.07$ to $-0.28$. Looser match (n 604–754): $-0.19$ to $-0.96$, every $p\le2\times10^{-4}$. QCD jets: $+0.06$ to $+0.17$, CIs include 0. Figure: results/physics/h41_colour_flow.png.
- **Checks.**
  - Unedited 10-class accuracy (79.8 / 65.6 / 77.8 / 60.7 %) and the Top Landscape AUCs reproduce earlier runs.
  - Translation-only control and fixed-axis scoring change the score by $\le0.03$.
  - C2ST of M-edited vs natural jets without the colour-flow observables: AUC 0.53–0.56.
- **Physical significance.** JetClass ParT and ParticleNet, including the versions that see only four-momenta, lower their W/Z/H score when the soft radiation between the two prongs is moved outside at fixed mass and $\tau_{21}$. This is the radiation pattern that colour connection produces. In the looser and mass-exact sets the networks respond to it 1.1–9 times more than a pull-angle BDT does. Not yet separated: whether the networks read colour flow itself or energy-correlation observables (D2) that the same edit changes.
