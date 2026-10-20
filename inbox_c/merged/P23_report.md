# P23: shape of the axis response: 2-D axis-offset surface, odd vs even terms, ParT vs ParticleNet (C3, D4) (agent report, 2026-10-20)

Job `anrunw-ai-c-p23-run` (NVIDIA GTX 1080 Ti, 3,338 s): `python -m interp.p23_axis_surface`. The first attempt of this job was
lost when its node went NotReady at chunk 31/36 (TaintManagerEviction); it was rerun unchanged.
- Smoke run `c-p23-smoke` (60 jets per class) passed.
- CPU follow-up `c-p23b-perjet` (1 CPU, 4 Gi, 81 s): `python -m interp.p23b_perjet`, per-jet vectors and calibration from the P23 npz.
- Figure redrawn by `c-p23-plot` (job.sh, 6 s).
- Scripts: `interp/p23_axis_surface.py`, `interp/p23b_perjet.py`.
- Results: `results/part/p23_axis_surface.json`, `results/part/p23b_perjet.json`; per-jet changes in
  `/j-jepa-vol/aaron-interp/results/part/p23_axis_surface.npz` (volume only).
- Figure: `results/part/p23_axis_surface.png`.
- Logs: `logs/c-p23-run.log`, `logs/c-p23-smoke.log`, `logs/c-p23b-perjet.log`, `logs/c-p23-plot.log`.

## Conclusion
Shifting all of a jet's particles by a small angle c leaves the jet's mass unchanged. The response of ParT's W-vs-Z score to such a
shift is almost entirely symmetric under c → −c. It grows like |c|², as the jet's second angular moment about the stored axis
(M2, the quantity behind the axis-moment mass) does. For a φ shift of 0.06 rad, 98 % of the mean response is this symmetric part.
Scaling it by each jet's own response to a real mass change accounts for 72 % of its jet-to-jet variation in ParT_full. The
direction-dependent (antisymmetric) part is 6–11 % of the response variance in trained ParT. Jet by jet, it does not point along
the jet's pT-weighted centroid any more than along a random other jet's centroid, so the JEDI-like centroid term is not seen in
ParT. ParticleNet weights the antisymmetric part more than the symmetric one (gains 0.52 vs 0.27 and 0.71 vs 0.15). Still
uncertain: the symmetric part depends on the shift's direction relative to the jet's long axis (10 % in ParT_full, 60 % in
ParticleNet_full), and this measurement does not name which jet quantity that dependence reads.

## Procedure
**Models.** All on the same jets:
- Released ParT_full and ParT_kin.
- ParT_full seeds s1 and s2 from P10 (2M-jet subset, best epoch).
- Released ParticleNet_full and ParticleNet_kin.
- Two untrained ParT_full (weaver initialisation, torch seeds 1 and 2).

**Sample.**
- JetClass test_20M WToQQ_105, ZToQQ_105 and ZJetsToNuNu_105 (QCD): 3,000 random rows each, `numpy default_rng(2305)`; no first-k.
- N = 9,000; no jet has fewer than 2 particles.
- Inputs are rebuilt with `part_core.build` (η = jet_eta + sign(jet_eta)·Δη). The baseline is the unedited particles, rebuilt.

**Edits.** Every edit acts on the stored (Δη, Δφ) of the real particles; per-particle features and four-vectors are rebuilt.
- **grid.** Rigid translation by c = (c_η, c_φ) on a 9 × 9 grid, each coordinate in {−0.08, −0.06, …, 0.08} rad: 80 non-zero
  points, forming 40 mirror pairs (+c, −c).
  - Relative change of the invariant mass: median 1.8e-6 / 2.5e-5 and max 5e-4 / 4.8e-3 at c = (0, 0.08) / (0.08, 0).
  - About the stored axis, M2(c) = M2 + 2 c·M1 + |c|² M0, with M0 = Σ pT, M1 = Σ pT (Δη, Δφ) and M2 = Σ pT ΔR².
- **rd0–3.** Per-jet random direction, |c| = 0.06. Held out from the fits.
- **rot90 / 180 / 270.** Rotation of (Δη, Δφ) about the stored axis. M0, M2 and all pair ΔR are kept; M1 is rotated. Relative
  invariant-mass change: median 0.65 % (rot90), max 39 %, because particles move in η.
- **Matched real change.** Per-jet dilation of the angles by k_j = √(M2(c)/M2). This gives the same ½Δlog M2 as the translation
  c ∈ {(0, ±0.06), (±0.06, 0)} (E23 / P11 method).

**Read-out.** d = per-jet change of logit_Z − logit_W, on the W and Z jets (pooled "WZ", N = 6,000; also W, Z and QCD alone). On
QCD jets also logit_QCD − (logit_W + logit_Z)/2.

**Decomposition.** Per mirror pair:
- odd(c) = (d(c) − d(−c))/2 and even(c) = (d(c) + d(−c))/2.
- Exact predictors: P(c) = ½ log(M2(c)/M2), with odd part Po and even part Pe. To first order, Po = c·M1/M2 and Pe = |c|² M0/(2 M2).

**Fits** (pooled over jets and grid points):
- odd = b_o Po + a_η c_η + a_φ c_φ, where a_η and a_φ are jet-independent direction terms.
- even = b_e Pe.
- Calibration κ = slope, through 0, of d(dilation) on log k_j over the 4 matched dilations.
- Gains g_o = b_o/κ and g_e = b_e/κ. A gain of 1 means the same response as a real change of equal ½Δlog M2.
- First-order η/φ split: separate coefficients for c_η M1η/M2, c_φ M1φ/M2, c_η² M0/2M2, c_φ² M0/2M2 and c_η c_φ M0/M2.

**Per jet (P23b).**
- Odd gradient g_j: least-squares fit of odd(c) ≈ g_j·c.
- Even quadratic form H_j.
- cos(g_j, v_j) for candidate vectors v: pT-weighted centroid M1/M0, unweighted, energy-weighted and log pT-weighted centroids,
  and the leading-particle position. Null: v taken from a random other jet.
- Anisotropy of H_j against the jet's pT-weighted second-moment tensor T = Σ pT r rᵀ/M0. Measured by the doubled-angle cosine and
  by the response along T's long axis vs its short axis.
- Per-jet calibration κ_j, used in the even part as g κ_j Pe.

**Locate (DAS).** P8 method, 'jm' interchange on the jet-mean residual at emb and b1–b8, one direction q per site.
- Models: ParT_full, ParT_kin and untrained ParT_full init1.
- Counterfactuals: per-jet random-direction translation ±c, |c| = 0.06. Per-particle route only: the four-vectors, hence the pair
  features, come from the unedited jet.
- q_odd is trained so that interchanged (I(+c) − I(−c)) reproduces (C(+c) − C(−c)) over 10 logits; q_even does the same for
  I(+c) + I(−c).
- 150 Adam steps of 128 jets; 1,500 training jets and 1,000 held-out W/Z jets.
- Recovery = 1 − Σ‖I_part − C_part‖² / Σ‖C_part‖², compared with 3 random q.

**Statistics.**
- 95 % bootstrap over jets (1,000 resamples; coefficients and κ resampled together).
- H0 odd = 0: sign-flip permutation of each jet's odd part (2,000).
- H0 g_o = g_e, H0 equal η/φ even gains and H0 a_η = a_φ: bootstrap two-sided p.
- Holm over the 8 models within each family. Holm over the 9 sites in DAS.
- Calls: W → Z (10-class argmax) on true W jets, with exact McNemar on W correctness.

## Numbers
**Unedited accuracy.**

| model | P1 sample (rows 3,000–4,000 of each _100 file, N = 10,000) | this sample (W/Z/QCD, 10-class argmax) |
|---|---|---|
| ParT_full | 85.93 % (known 85.9) | 75.8 % |
| ParT_kin | 74.83 % (known 74.5; P4 74.46) | 66.2 % |
| ParT_full s1 / s2 | 83.60 / 83.48 % | 72.3 / 72.2 % |
| PN_full | 84.57 % (P4 84.57) | 73.4 % |
| PN_kin | 70.34 % (P4 70.35) | 62.8 % |
| untrained 1 / 2 | 12.7 / 11.9 % | 3.0 / 19.2 % |

**Gains on WZ jets (pooled; 95 % CI).**

| model | κ [logit per unit ½Δlog M2] | g_odd | g_even | g_o − g_e (p, Holm) | odd share of variance | sign-flip p (Holm) |
|---|---|---|---|---|---|---|
| ParT_full | 9.87 [9.40, 10.36] | 0.68 [0.56, 0.79] | 0.66 [0.64, 0.67] | +0.02 [−0.09, 0.13] (0.71, 0.71) | 0.064 [0.062, 0.067] | 0.0005 (0.004) |
| ParT_kin | 7.91 [7.53, 8.29] | 0.02 [−0.12, 0.15] | 0.50 [0.49, 0.52] | −0.49 [−0.63, −0.35] (0.001, 0.008) | 0.080 | 0.81 (0.81) |
| ParT_full s1 | 9.29 | 0.33 [0.18, 0.49] | 0.59 [0.58, 0.61] | −0.26 [−0.42, −0.11] (0.001, 0.008) | 0.110 | 0.0005 (0.004) |
| ParT_full s2 | 9.34 | 0.63 [0.48, 0.78] | 0.74 [0.72, 0.76] | −0.11 [−0.26, 0.03] (0.14, 0.29) | 0.081 | 0.0005 (0.004) |
| PN_full | 9.39 | 0.52 [0.39, 0.65] | 0.27 [0.26, 0.29] | +0.24 [0.11, 0.37] (0.001, 0.008) | 0.184 | 0.0005 (0.004) |
| PN_kin | 7.29 | 0.71 [0.60, 0.82] | 0.15 [0.14, 0.16] | +0.56 [0.46, 0.67] (0.001, 0.008) | 0.160 | 0.0005 (0.004) |
| untrained 1 | −0.19 [−0.21, −0.18] | (κ ≈ 0: ratio undefined) | | | 0.567 | 0.0005 |
| untrained 2 | −0.03 [−0.04, −0.02] | (κ ≈ 0) | | | 0.669 | 0.0005 |

**Size of the response.**
- Mean |d| at c = (0, 0.06): ParT_full 1.08, ParT_kin 0.80, PN_full 0.79, PN_kin 0.51, untrained 0.056 / 0.042.
- By class, ParT_full: W g_o 0.97 [0.85, 1.12] vs g_e 0.73 [0.71, 0.75]; Z g_o 0.20 [−0.07, 0.47] vs g_e 0.45 [0.42, 0.48].
- The pooled equality of the two gains in ParT_full is an average of opposite class differences.

**Variance explained (ParT_full, WZ).**
- Odd part:
  - The centroid term Po explains R² 0.02; the fixed-direction terms explain 0.013.
  - A per-jet free linear gradient explains 0.66.
- Even part:
  - Pooled g Pe: R² 0.31.
  - Pe scaled by each jet's own κ_j: 0.72 (g 0.56).
  - Per-jet free quadratic form: 0.87.
- Other models, R² of the even part:

  | model | pooled g Pe | g κ_j Pe | per-jet quadratic |
  |---|---|---|---|
  | ParT_kin | 0.19 | 0.59 | 0.86 |
  | s1 | 0.27 | 0.68 | 0.89 |
  | s2 | 0.34 | 0.70 | 0.88 |
  | PN_full | 0.16 | 0.38 | 0.82 |
  | PN_kin | 0.06 | 0.17 | 0.81 |

- Held-out random directions (|c| = 0.06), ParT_full:
  - R² of the fitted surface 0.273; even part alone 0.269.
  - Within-jet s.d. over the 4 directions: measured 0.33, predicted by the fit 0.04.

**Per-jet odd gradient vs jet vectors (P23b, WZ, N = 6,000).** cos(g_j, v_j), with the random-other-jet null in parentheses.

| model | pT-weighted centroid M1 | diff vs null (CI) | unweighted centroid | leading particle |
|---|---|---|---|---|
| ParT_full | 0.132 (0.125) | [−0.0005, 0.016], p 0.07 | 0.064 (−0.001) | −0.060 (0.008) |
| ParT_kin | 0.081 (0.088) | [−0.015, 0.003], p 0.14 | −0.053 (−0.009) | 0.131 (0.005) |
| s1 | −0.012 (−0.011) | p 0.85 | 0.044 (−0.015) | −0.052 (0.014) |
| s2 | 0.049 (0.044) | p 0.21 | 0.009 (−0.014) | −0.009 (0.013) |
| PN_full | 0.232 (0.212) | [0.012, 0.027], p 0.001 | 0.213 (−0.003) | −0.225 (0.007) |
| PN_kin | 0.259 (0.238) | [0.012, 0.029], p 0.001 | 0.088 (−0.005) | −0.051 (0.005) |

- Centroid size, median |M1η/M0| = 5.5 mrad and |M1φ/M0| = 0.17 mrad.
- A centroid offset therefore exists almost only along η about the stored axis. The first-order φ odd coefficient
  (e.g. PN_full 4.98) has no support from the data and is not interpreted.

**Even part: anisotropy relative to the jet's long axis (P23b).**
- Measure: (H_long − H_short)/(H_long + H_short), where long = the principal axis of T.

| model | long vs short | 95 % CI |
|---|---|---|
| ParT_full | −0.098 | [−0.104, −0.092] |
| ParT_kin | +0.101 | [0.087, 0.116] |
| s1 | −0.182 | – |
| s2 | −0.056 | – |
| PN_full | −0.625 | [−0.667, −0.587] |
| PN_kin | −1.53 | – |
| untrained | −0.31 / −0.71 | – |

- Doubled-angle cosine with T: ParT_full −0.30 (null 0.01), PN_full −0.45, untrained +0.64 / +0.53.

**η vs φ.**
- Even gains: ParT_full η 0.453 vs φ 0.450, difference 0.002 [−0.003, 0.007]. ParT_kin difference 0.017 [0.009, 0.025] (Holm
  p 0.008). Other models: CI includes 0 or the difference is < 0.01.
- Model-free axis comparison of the even part: equal within 0.015 for every trained model.
- Fixed-direction odd term a_η [logit/rad]:

  | model | a_η | p | Holm |
  |---|---|---|---|
  | ParT_full | +0.37 [0.14, 0.58] | – | – |
  | ParT_kin | −0.44 | – | – |
  | s1 | +0.72 | – | – |
  | s2 | +0.85 | – | – |
  | PN_full | −0.73 | 0.0005 | – |
  | PN_kin | −0.13 | 0.10 | – |

  a_φ is within ±0.3 for all trained models.
- Ratio to the matched dilation, R(+c) − R(−c):
  - φ: ParT_full +0.033 [0.007, 0.057]; others in [−0.035, 0.013].
  - η: ParT_kin +0.24, s1 +0.20, s2 +0.18, PN_full −0.12, PN_kin −0.14 (all CIs exclude 0); ParT_full −0.016 [−0.039, 0.005].
- An η shift at fixed pT also changes every particle's energy (e_log and logerel inputs), so the ±η asymmetry is not attributable
  to the Δη input convention alone.

**Ratio R to the matched dilation, at c = (0, +0.06), (0, −0.06), (+0.06, 0), (−0.06, 0).**

| model | R |
|---|---|
| ParT_full | 0.82, 0.78, 0.81, 0.82 |
| ParT_kin | 0.73, 0.75, 0.88, 0.64 |
| s1 | 0.68, 0.71, 0.83, 0.63 |
| s2 | 0.85, 0.84, 0.97, 0.79 |
| PN_full | 0.54, 0.54, 0.47, 0.58 |
| PN_kin | 0.39, 0.40, 0.30, 0.44 |

**φ + 0.06 decomposition (WZ mean d).**

| model | total | even | odd | odd: centroid term | odd: fixed term |
|---|---|---|---|---|---|
| ParT_full | 0.760 | 0.744 | 0.015 | 0.000 | 0.007 |
| ParT_kin | 0.551 | 0.554 | −0.003 | – | – |
| PN_full | 0.475 | 0.478 | −0.003 | – | – |

**Calls: true W → Z, base → edit.**

| model | φ+0.06 | φ−0.06 | η+0.06 | η−0.06 | φ+0.08 | rot90 | rot180 | dilation matched to φ+0.06 |
|---|---|---|---|---|---|---|---|---|
| ParT_full | 10.4 → 31.0 | 30.1 | 27.5 | 34.8 | 44.2 | 13.2 | 14.1 | 36.3 |
| ParT_kin | 16.2 → 34.7 | 34.5 | 34.5 | 35.2 | 42.5 | 18.0 | 17.4 | 43.6 |
| PN_full | 11.4 → 24.4 | 24.6 | 20.9 | 28.2 | 25.4 | 14.8 | 14.5 | 38.1 |

- All McNemar p < 1e-20, except rot180 for ParT_kin (p 1e-7).
- Rotation at fixed M2, mean d: ParT_full +0.02, mean |d| 0.59 (vs 1.08 for φ+0.06); PN_full mean −0.28, mean |d| 0.95.

**QCD jets.**
- κ for logit_Z − logit_W is about 0: ParT_full 0.07 [−0.02, 0.16]. Gains are undefined there.
- QCD read-out (logit_QCD − mean of W, Z), ParT_full: κ −2.97; g_even 0.83 [0.80, 0.87]; g_odd −0.92 [−1.33, −0.51]; odd share 0.07.

**DAS (1-d, held-out recovery; random q ≤ 0.06 everywhere; all Holm p ≤ 0.009).** "zw" = share of the d response.

| model | even part: emb → b8 | odd part: emb → b8 |
|---|---|---|
| ParT_full | 0.77 → 0.79 (b2) → 0.82 (b4) → 0.84 (b8) | 0.12 → 0.16 (b2) → 0.07 (b3) → 0.19 (b5) → 0.24 (b8) |
| ParT_kin | 0.44 → 0.60 (b2) → 0.67 (b5) → 0.72 | 0.14 → 0.21 (b3) → 0.33 |
| untrained 1 | 0.73–0.76 at every site | 0.35–0.38 at every site |

- On the d read-out (zw), ParT_full: even 0.67 → 0.83, odd 0.13 → 0.20.
- |cos(q_odd, q_even)|: ParT_full 0.28–0.72; untrained 0.09–0.55.
- On the per-particle route alone, the odd part is 17 % of the logit response energy (ParT_full: 0.99 vs 4.72), against 6 % with
  the full inputs.

## Reading (scoped to these measurements)
- **Symmetric part.** In trained ParT (released and both seeds), 89–94 % of the response variance on the grid is symmetric under
  c → −c. It accounts for 98 % of the mean φ + 0.06 response.
  - Per jet, it is a quadratic in c (R² 0.86–0.89).
  - Its size follows each jet's own response to a real dilation of equal ½Δlog M2 (R² 0.59–0.72 with per-jet κ_j, vs 0.19–0.34
    pooled).
  - This is the behaviour expected of a function of the axis moment M2 about the stored axis, at 0.5–0.74 of a real change.
- **Antisymmetric part, ParT.** It is non-zero (sign-flip p < 0.001 in 5 of 6 trained models). It is a per-jet linear function of c
  (R² 0.64–0.70), but its direction is not this jet's pT-weighted centroid: alignment exceeds the other-jet null by at most 0.016.
  - The pooled centroid gain (0.68 in ParT_full) therefore reflects a direction shared across jets: centroids sit along η, and the
    ±η responses differ.
  - It does not reflect a per-jet c·M1 term. The JEDI-like reading (per-jet c·M1 weighted 1.3–1.8×) is not seen in any ParT model.
- **ParticleNet.**
  - It weights the antisymmetric part more: odd share 0.16–0.18; g_o > g_e.
  - Its per-jet gradient aligns slightly with the jet's own centroid (+0.02 over null) and with the unweighted centroid (0.21, PN_full).
  - Its symmetric part follows the matched real change less well (0.38 / 0.17).
  - Its response is much weaker along the jet's long axis (−0.63 / −1.5).
  - The shape of the axis response is therefore architecture-dependent.
- **Orientation.** Rotation about the axis at fixed M2 moves ParT_full's score by 0.59 per jet on average (mean 0.02) and changes W → Z
  by +2.8 to +3.7 points. Together with the long-vs-short-axis anisotropy of the even part (−0.10), this is the measured dependence on
  something besides M2. The rotation also changes particle energies through η.
- **Untrained nets.**
  - Their response to the shift is 20× smaller; it is mostly antisymmetric (0.57–0.67), and they do not respond to real dilations
    (κ ≈ 0).
  - Their 1-d DAS recoveries (even 0.73–0.76, odd 0.35–0.38) are as high as or higher than the trained nets'. As in P18v, the DAS
    recovery level is not evidence of training-specific structure.
  - The trained-specific DAS feature is the rise of even recovery with depth in ParT_kin (0.44 → 0.72).

## Predictions of competing hypotheses and which are excluded
- **Thesis (axis-moment mass; even term dominant, response symmetric, one gain).**
  - For ParT: the even term dominates (89–94 % of variance, 98 % of the φ + 0.06 mean), and the even part follows the per-jet
    real-change gain.
  - One pooled gain is not excluded for ParT_full (p 0.71) and s2 (p 0.29 Holm). It is excluded for ParT_kin and s1, whose odd gain
    is lower.
  - "Odd ≈ 0" holds only for ParT_kin's centroid coefficient. The odd part as a whole is small but non-zero.
- **Alternative 1 (centroid/direction-dependent as JEDI E23v, odd gains 1.3–1.8 > even).**
  - Excluded for every ParT model: all g_o ≤ 0.79 (upper CI), and per-jet alignment with the own centroid is at null.
  - ParticleNet has odd > even, but its gains are 0.52 / 0.71 (< 1.3), with weak per-jet centroid alignment.
- **Alternative 2 (input convention, η and φ treated differently).**
  - Even gains are equal within 0.002–0.017; only ParT_kin differs (5 %, Holm p 0.008).
  - ±η asymmetries exist (|R(+) − R(−)| 0.12–0.24 in five models) while ±φ is symmetric. The η shift also changes particle energies,
    so the convention is not isolated.
- **For C3 / P8v / P13v (why natural m_A variation predicts about 1/7 of the φ-shift logit change).**
  - The φ shift acts almost entirely through the symmetric |c|² M0/M2 term, at 0.66 of a matched real change.
  - No large odd or direction term was found that would add to the φ response.
  - The gap therefore is not explained by an odd/centroid quantity. Candidates left: orientation relative to the prong axis, and
    per-jet differences in sensitivity (κ_j spans −8 to +31 between the 10th and 90th percentiles).

## Issues and limits
- **ParT_kin accuracy.** On the P1 sample it is 74.83 % against 74.46 % in P4 and P18 on the same rows. ParT_full and both
  ParticleNets match earlier runs exactly. The 0.37-point difference is not explained (GPU type, GTX 1080 Ti here, is one candidate).
- **Linearised split.** The first-order η/φ split of the odd coefficients is ill-conditioned because M1φ ≈ 0 about the stored axis.
  Only the exact-predictor fits and the model-free axis comparisons are read.
- **Pooled fits.** The pooled g Pe fit explains only 0.31 of the even part. Gains are population averages; per-jet sensitivity varies
  widely.
- **Rotation and η edits.** Both change particle energies and, for wide jets, the invariant mass (rot90 max 39 %). They are not
  pure axis edits.
- **QCD read-out.** For logit_Z − logit_W on QCD jets, κ ≈ 0, so gains are undefined there.
- **DAS route.** The DAS counterfactuals use the per-particle route only, so DAS shares refer to that route.
- **Not done.** ParticleNet DAS (P15 covers ParticleNet internals); L-GATr.

## Explainer entry (for site/explain.py)
- **Question.** When all particles of a jet are moved by a small angle $\vec c$, the invariant mass stays the same but ParT's W/Z score
  changes (C3). Is that response a function of the axis moment $M_2$ (symmetric under $\vec c\to-\vec c$) or does it depend on the
  direction of the shift (antisymmetric), as found for JEDI-linear?
- **Background.**
  - About the stored axis, $M_2(\vec c)=M_2+2\,\vec c\cdot\vec M_1+|\vec c|^2M_0$, where $M_0=\sum p_T$,
    $\vec M_1=\sum p_T(\Delta\eta,\Delta\phi)$ and $M_2=\sum p_T\Delta R^2$.
  - So an axis-moment reading predicts an odd term $\propto\vec c\cdot\vec M_1$ and an even term $\propto|\vec c|^2M_0$ with one gain.
- **Method.**
  - 9 × 9 grid of shifts, $|c_i|\le0.08$ rad, on 3,000 W, Z and QCD jets; 8 models.
  - Split each jet's response: $\mathrm{odd}=\tfrac12[d(\vec c)-d(-\vec c)]$ and $\mathrm{even}=\tfrac12[d(\vec c)+d(-\vec c)]$.
  - Fit $\mathrm{odd}=b_oP_o+a_\eta c_\eta+a_\phi c_\phi$ and $\mathrm{even}=b_eP_e$, with $P=\tfrac12\log(M_2(\vec c)/M_2)$.
  - Gains $g=b/\kappa$, where $\kappa$ is the response per unit $\tfrac12\Delta\log M_2$ to a real dilation of the angles.
  - Per jet: compare the odd gradient's direction with the jet's centroid (null: another jet's centroid).
  - DAS: find the one-dimensional subspace at each block that carries the odd and the even parts.
- **Result (figure p23_axis_surface.png).**
  - ParT: the response is a bowl ($\propto|\vec c|^2$). The even part is 89–94 % of the variance and 98 % of the φ + 0.06 mean.
    Scaled by each jet's own real-change response, it explains 72 % of the even part (ParT_full).
  - The odd part does not follow the jet's own centroid (cos 0.132 vs null 0.125).
  - ParticleNet: odd gain > even gain (0.52 vs 0.27; 0.71 vs 0.15), and a much weaker response along the jet's long axis.
- **Checks.**
  - Unedited accuracy 85.93 / 74.83 % (ParT) and 84.57 / 70.34 % (ParticleNet) on the P1 rows.
  - Mass change of the φ shifts ≤ 5e-4.
  - Sign-flip permutation and Holm over models.
  - Held-out random directions.
  - Rotation at fixed $M_2$.
  - Untrained nets: κ ≈ 0 and a 20× smaller response.
- **Why it matters.**
  - ParT's response to moving the axis is, to good approximation, the axis-moment term, unlike JEDI-linear's direction-weighted
    response.
  - The remaining direction dependence is orientation relative to the jet's long axis, not the centroid.
  - ParticleNet's response has a different shape, so the shape is not shared across architectures.
