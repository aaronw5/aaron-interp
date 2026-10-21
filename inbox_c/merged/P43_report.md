# P43 — Spread about the stored axis vs centroid offset: compensated two-particle edits (ParT, ParticleNet)

Claims: C18 (P39v confidence 0.75), C3. Script `interp/p43_spread_vs_centroid.py`; job `c-p43-run` (GPU, smoke + full, exit 0);
results `results/part/p43_spread_vs_centroid.json`, per-jet logits `results/part/p43_logits.npz`, figure
`results/part/p43_spread_vs_centroid.png`, log `logs/c-p43-run.log`.

## Conclusion

When two hard particles are moved apart or together so that the jet's angular spread changes while its pT-weighted centre stays
put, ParT_full responds almost exactly as it does to a real mass change of the same size (ratio 0.95). When the same jet is
instead shifted as a whole so that its centre moves 0.01–0.04 rad off the stored jet axis, which changes the spread about that
axis by the same formula, ParT_full responds at only about one third of that size (ratio 0.31); ParticleNet_full gives the same
pattern. So "spread about the stored axis" (C18) describes the networks' response to a change in internal spread. It overstates
their response to an off-axis shift of the centre at these sizes. Smaller re-centring shifts of about 0.006 rad gave a ratio
near 0.7–0.8 (as in P39v, but without an interval here). The response to the offset may therefore fall as the offset grows,
but that dose dependence has not yet been measured directly.

## Definitions

- Coordinates: (Δη, Δφ) of each particle from the **stored jet axis** (jet_eta, jet_phi in the file), r = |(Δη, Δφ)|.
- M0 = Σ pT; **c** = Σ pT (Δη, Δφ) / M0 (pT-weighted centroid about the stored axis); **M2** = Σ pT r² (spread about the stored
  axis); **M2c** = Σ pT |r⃗ − c|² (spread about the centroid). Identity: M2 = M2c + M0 |c|². log M2 enters the pT-weighted
  angular mass proxy m_A = √(M0 M2).
- Read-out **d** = logit_Z − logit_W (primary; same read-out as P39v). Secondary **q** = log(p_W + p_Z) − log(p_QCD) (the
  W/Z-vs-QCD score named in the plan). On these W/Z jets a real dilation moves q very little, so q-based ratios have tiny
  denominators (values 1–550). They are reported in the JSON, but no reading is based on them.
- **R(edit)** = Σ_jets [d(edit) − d(unedited)] / Σ_jets [d(E4 matched) − d(unedited)]: the network's response to the edit
  divided by its response to a real angular dilation about the stored axis that changes log M2 by the same predicted amount on
  the same jets. Under "the network reads M2 about the stored axis" every edit has R ≈ 1, or R ≈ 0 when M2 is unchanged.
- **W → Z rate**: share of true-W jets whose 10-class argmax is Zqq. **McNemar**: exact binomial test on paired call flips
  (the same jets with and without the edit). **Holm**: step-down family-wise correction over the four trained models.

## Procedure

Sample: JetClass val_5M WToQQ_120.root and ZToQQ_120.root (test_20M stops at _119), 10,000 random rows each, numpy
default_rng(4343). Jets with < 2 particles were dropped. The edits below require a suitable particle pair, which 4,592 of
20,000 jets have (2,407 W, 2,185 Z). Kept jets: median pair pT fraction 0.29, median pair separation 0.30 rad, median
|c| 0.006 rad, median m_inv 89 GeV. Models: released ParT_full, ParT_kin, ParticleNet_full (PN_full), ParticleNet_kin (PN_kin).
Nulls: untrained ParT_full (weaver initialisation, torch seed 0) and weight-permuted ParT_full (entries of every tensor
permuted, seed 4343). Inputs were rebuilt with part_core.build. Invariant mass m_inv was recomputed after every edit.

Pair: among the 8 hardest particles, pairs with pT ratio ≥ 0.6. The admissible pair with the largest pT_a + pT_b is used for
every edit of that jet: in 3,197 jets it includes the hardest particle, and in 2,455 the second hardest. A pair is admissible
if it meets all of these conditions:
- every edit keeps the two particles at ≥ ½ of their separation;
- the E2 translation is ≤ 0.2 rad;
- the E5 direction exists.

Only the (Δη, Δφ) of particles a and b change. pT, mass, PID and track inputs are kept. In the list below, u is the unit vector
from b to a and d is the a–b distance.

- **E1 spread-only**, f = ±2, ±5, ±10 %. Particle a moves by +t u / pT_a and particle b by −t u / pT_b, i.e. in opposite
  directions along the line joining them, with the momentum-weighted sum of displacements kept at zero. Then c is unchanged
  (realised |Δc| = 0) and M2 and M2c both change by exactly f·M2 (realised Δlog M2c = Δlog M2 to 3·10⁻⁴). Median displacement
  at ±10 %: 0.021–0.032 rad per particle. Δlog m_inv ≈ ½ Δlog M2 (e.g. +0.047 at +10 %).
- **E2 centroid-only**, δ = 0.01, 0.02, 0.04 rad, in two steps:
  - Step 1 translates both particles by s = M0 δ e / (pT_a + pT_b). Direction e points from the pair's pT-weighted
    barycentre toward the centroid; this is the only direction for which step 2 is usually feasible. The centroid moves by
    exactly δ.
  - Step 2 is an E1-type move on the same pair that restores M2c exactly (realised Δlog M2c = 0.0000).
  - M2 changes by M0(|c + δe|² − |c|²): realised Δlog M2 +0.005, +0.019, +0.075.
  - Median displacement per particle is 0.033, 0.067, 0.13 rad. m_inv is unchanged to 10⁻⁴.
- **E2rec**: e = −c/|c|, δ = |c| (centroid moved onto the stored axis). Δlog M2 −0.0023, tiny.
- **E3** = E2 with the stored axis moved along with the centroid. jet_eta, jet_phi shift by δe, and every particle's (Δη, Δφ)
  is re-expressed about the new axis. The lab-frame particles, and so m_inv, are identical to E2 (max |Δlog m_inv| vs E2 in
  JSON). c, M2c and M2 about the stored axis equal those of the unedited jet. E3 therefore carries E2's two-particle
  displacement without its offset. **E2 − E3** is a rigid translation of all axis-relative coordinates by δ (M2c fixed, |c|
  and M2 changed), applied to the same deformed jet. This contrast isolates the offset term M0|c|².
- **E4 reference**: all particles' (Δη, Δφ) are multiplied by k, with k² = target M2 / M2 per jet, matched to each E1 / E2 dose.
- **E5 null**: same pair, same |t| as E1 ±10 %, so each particle is displaced by the same amount. The direction is rotated from
  u by θ with cos θ = −t(1/pT_a + 1/pT_b)/(2d), which keeps both c and M2 fixed.
- Pooled R values: E1 raise/lower = sums over the three doses. E2 = sum over δ. E2 − E3 = (ΣΔd_E2 − ΣΔd_E3)/ΣΔd_E4. E5 =
  (ΣΔd_E5+ + ΣΔd_E5−)/(ΣΔd_E4(+10 %) − ΣΔd_E4(−10 %)).

Statistics: 2,000 bootstrap resamples over jets, shared by all edits so that differences are paired. Bootstrap two-sided p
values have a resolution of 0.001. Sign-flip permutation tests (5,000) and exact McNemar tests were used, with Holm correction
over the 4 trained models. Hypotheses:
- H0: R(E1) = R(offset).
- H1a: R(offset) < R(E1), i.e. the network reads the spread about the centroid and the offset only partly.
- H1b: R(offset) > R(E1).

Predictions:
- C18: R(E1) ≈ R(E2 − E3) ≈ 1, E3 ≈ 0, E5 ≈ 0.
- Centroid-relative read-out: R(E1) ≈ 1, R(E2 − E3) ≈ 0.

## Numbers

Unedited accuracy (10-class argmax):
- P1 sample (10,000 jets): ParT_full 85.93 %, ParT_kin 74.47 %, PN_full 84.56 %, PN_kin 70.36 % (reproduces P1 and P39v);
  untrained 10.1 %, permuted 9.7 %.
- All 20,000 W/Z jets read: 74.3 / 62.0 / 71.6 / 59.8 %.
- 4,592 kept jets: 80.5 / 70.8 / 78.2 / 68.5 %.
- 15,408 dropped jets: 72.4 / 59.4 / 69.6 / 57.2 %.
- So the kept set is easier than the dropped set.

R (read-out d), 95 % bootstrap interval:

| model | E1 raise | E1 lower | E2 raw | E3 (axis moved) | **E2 − E3 (offset)** | E5 null | E1 raise − (E2−E3), Holm p |
|---|---|---|---|---|---|---|---|
| ParT_full | 0.95 [0.90, 1.00] | 0.92 [0.87, 0.97] | 0.39 [0.16, 0.62] | 0.08 [−0.16, 0.30] | **0.31 [0.28, 0.34]** | 0.03 [−0.01, 0.07] | +0.64 [0.59, 0.69], 0.002 |
| ParT_kin | 1.11 [1.07, 1.16] | 0.54 [0.50, 0.58] | 0.81 [0.65, 0.97] | 0.45 [0.29, 0.60] | **0.37 [0.32, 0.41]** | 0.37 [0.33, 0.40] | +0.75 [0.69, 0.80], 0.002 |
| PN_full | 0.91 [0.87, 0.96] | 0.99 [0.94, 1.04] | 0.12 [−0.10, 0.33] | −0.18 [−0.40, 0.03] | **0.30 [0.27, 0.34]** | −0.05 [−0.09, −0.01] | +0.61 [0.55, 0.67], 0.002 |
| PN_kin | 0.81 [0.79, 0.84] | 0.58 [0.55, 0.60] | −1.31 [−1.45, −1.19] | −1.14 [−1.26, −1.02] | **−0.18 [−0.20, −0.15]** | 0.12 [0.11, 0.14] | +0.99 [0.96, 1.02], 0.002 |
| ParT_full untrained | 0.12 | 0.14 | 0.16 | −0.73 | 0.88 [0.79, 0.98] | 0.002 | – |
| ParT_full permuted | 2.4 [1.4, 7.3] | 2.0 [−7, 12] | 2.0 [−8, 13] | 3.0 [−12, 20] | −1.1 [−11, 6] | 0.33 [−0.3, 1.2] | – |

- The plan's direct test, raw E2 vs E1 raise, gives Holm p = 0.002 for all four trained models. Differences: +0.57 (ParT_full),
  +0.30, +0.79, +2.13.
- Raw E2 combines the offset with a large two-particle displacement:
  - Mean |per-jet Δd| under E2_0.04 is 3.0× that of the matched dilation, and 2.7× under E3_0.04.
  - ParT_full calls changed: E2_0.04 51.7 %, E3_0.04 46.1 %, matched E4 10.7 %.
  - Hence the E2 − E3 contrast is the offset estimate used here.
- Per-dose offset ratio, point estimates R(E2_δ) − R(E3_δ) (shared denominator):

  | δ (rad) | ParT_full | ParT_kin | PN_full | PN_kin |
  |---|---|---|---|---|
  | 0.01 | 1.05 | 3.6 | 0.19 | −0.33 |
  | 0.02 | 0.38 | 0.29 | 0.34 | −0.31 |
  | 0.04 | 0.25 | 0.29 | 0.30 | −0.14 |

  At δ = 0.01 the denominator is small (Δlog M2 = 0.005).
- Small re-centring offset, point estimate R(E2rec) − R(E3rec) (δ = |c| ≈ 0.006, no interval; each term alone has an interval
  of about ±2): ParT_full 0.73, PN_full 0.76, PN_kin 0.83, ParT_kin −0.31. P39v measured 0.76–0.81 for rigid shifts of this
  size.
- E3 = 0 test (sign-flip, Holm): ParT_full p 0.51, PN_full 0.20, ParT_kin 0.0008, PN_kin 0.0008.
- E5 null (c and M2 fixed, same displacement as E1 ±10 %): ParT_full 0.03 [−0.01, 0.07] (p 0.22), PN_full −0.05
  [−0.09, −0.01]. ParT_kin 0.37 [0.33, 0.40] and PN_kin 0.12 [0.11, 0.14]: both E5 directions move the kin models toward Z.
- Raising vs lowering (E1, pooled R_raise − R_lower, Holm):

  | model | difference [95 % CI] | Holm p | per dose |
  |---|---|---|---|
  | ParT_full | +0.03 [−0.05, 0.11] | 0.43 | 2 %: 0.91 vs 1.02; 10 %: 0.98 vs 0.88 |
  | PN_full | −0.08 [−0.15, 0.00] | 0.10 | – |
  | ParT_kin | +0.57 [0.51, 0.64] | 0.002 | 10 %: 1.19 vs 0.38 |
  | PN_kin | +0.24 [0.20, 0.28] | 0.002 | – |

  P39v's direction (raising 0.71–0.84 < lowering 0.94–1.04, for ParT_full under rigid shifts) is not seen for E1. The
  dilation reference itself is symmetric to within 4 % (E4 raise/lower response ratio 0.96–1.00, ParT_full).
- Dose linearity, slope of R vs Δlog M2 across the six E1 doses: ParT_full +0.23 [−0.30, 0.73] (flat); PN_full −0.50
  [−1.00, 0.01]; ParT_kin +3.8 [3.4, 4.2]; PN_kin +1.6 [1.4, 1.9]. The E4 response is close to linear: quadratic term
  2.0 [0.7, 3.2] against a linear term of 6.1, ParT_full.
- W → Z rate (true W, ParT_full; baseline 11.1 %), McNemar:

  | edit | W → Z rate | McNemar p |
  |---|---|---|
  | E1 +10 % | 25.8 % | 5e-83 |
  | matched E4 for E1 +10 % | 24.7 % | – |
  | E1 −10 % | 6.2 % | – |
  | matched E4 for E1 −10 % | 5.0 % | – |
  | E2_0.04 | 16.9 % | – |
  | matched E4 for E2_0.04 | 21.6 % | – |
  | E3_0.04 | 14.0 % | 4e-4 |
  | E5 ± | 11.9–12.2 % | 0.08, 0.01 |

  Calls changed: E1 ±10 % 17.4–17.8 % vs E4 13.9–14.5 %; E5 6.6–8.4 %.
- Nulls:
  - The untrained net has an E4 response about 8× smaller than trained ParT_full (R(E1) 0.12, no calls changed), so its ratios
    divide by a near-zero response. Its E2 − E3 of 0.88 reflects a direct dependence of the untrained embedding on the
    axis-relative coordinates, not a mass reading.
  - Permuted-model intervals span about ±10.

## Reading

- E1 spread-only: for ParT_full and PN_full, a change of the spread about a fixed centroid produced 91–99 % of the read-out
  change of a matched dilation. The ratio was flat in dose and did not differ between raising and lowering. The E5 control
  (same particles, same displacement, M2 and c fixed) gave 3 % and −5 %. The read-out change is therefore carried by the M2 /
  M2c change, not by the displacement of two particles.
- Offset: moving the centroid 0.01–0.04 rad off the stored axis by a rigid shift (E2 − E3, M2c fixed) produced 30–31 % of the
  matched-dilation change in ParT_full and PN_full (37 % in ParT_kin). Over these doses H0 (R(offset) = R(spread)) is rejected
  in favour of H1a. The pure centroid-relative read-out (offset ratio 0) is also excluded for the full models.
- Re-centring shifts of about 0.006 rad gave point estimates of 0.73–0.76, near P39v's 0.76–0.81. This would be consistent with
  an offset response that weakens as the offset grows. A dose-controlled rigid-shift series with intervals is needed to
  establish this.
- Raw E2 is dominated by the 0.07–0.13 rad two-particle relocation. The E3 control carries most of that response, so raw E2
  alone does not separate the hypotheses.
- Kin models:
  - ParT_kin and PN_kin respond to E5 (c and M2 fixed) at 12–37 % of the dilation response.
  - Their E1 ratio depends on dose and direction: ParT_kin 0.38 at −10 % vs 1.19 at +10 %.
  - PN_kin's offset ratio is negative.
  - These models read more pair geometry than M2 alone.
- Scope:
  - The kept jets (23 %) have two hard similar-pT particles carrying a median 29 % of the jet pT, and are classified more
    accurately than the dropped jets.
  - Offsets above 0.04 rad were not tested.
  - The P39v raise/lower asymmetry was measured with rigid shifts, not with this edit.

## Suggested explainer entry (for site/explain.py; not written there: outside this node's write scope)

- **Question**: does ParT read the spread about the stored jet axis as one number, or the spread about the jet's centre plus
  the centre's offset separately?
- **Background**: $M_2=\sum_i p_{T,i} r_i^2=M_{2c}+M_0|\vec c|^2$, with $\vec c=\sum_i p_{T,i}\vec r_i/M_0$.
- **Method**:
  - E1 moves two similar-pT particles by $\pm t\hat u/p_{T}$, which fixes $\vec c$ and gives $\Delta M_2=\Delta M_{2c}=2td+t^2(1/p_{T,a}+1/p_{T,b})$.
  - E2 − E3 shifts all particles rigidly by $\delta$ relative to the axis, so that $\Delta M_{2c}=0$ and $\Delta M_2=M_0(|\vec c+\delta\hat e|^2-|\vec c|^2)$.
  - $R=\sum\Delta d_{\rm edit}/\sum\Delta d_{\rm dilation}$, with the dilation matched in $\Delta\log M_2$.
- **Result**: the figure above. $R_{\rm spread}=0.95$, $R_{\rm offset}=0.31$ (ParT_full).
- **Checks**: the E5 null, the untrained and permuted models, invariant mass reported for every edit.
- **Why it matters**: the mass the tagger "sees" is mostly the internal spread. An off-axis centre at the 0.02–0.04 rad scale
  counts about a third as much.
- **Glossary additions**: M2c (spread about the pT centroid), centroid offset c, compensated two-particle edit, E2 − E3 offset
  contrast.

## Figure

`results/part/p43_spread_vs_centroid.png`, three panels:
- Left: a real W jet (m_inv 85.0 GeV) under E1 +10 %: two particles at Δη ≈ ±0.2 rad move apart and the centroid stays fixed;
  m_inv becomes 89.2 GeV and Δlog M2 = +0.095.
- Middle: the same jet under E2 δ = 0.04: the two particles move toward the centroid, the centroid (red ×) moves 0.04 rad, m_inv
  stays at 85.0 GeV and Δlog M2c = 0.
- Right: pooled R per edit and model. The bar chart shows raw E2, not the E2 − E3 contrast. The E2 − E3 values are in the table
  above.

## Deviations from the plan

- The plan named the "_120 rows"; these exist only in val_5M, which was used. P15 used 2,000 other random rows of the 10-class
  _120 files.
- The analysed N is 4,592, not 20,000, because of the admissible-pair requirement. Unedited accuracies are given for both sets.
- E2 direction: toward the centroid (feasibility), not along c. For δ ≫ |c|, M2 rises in either direction.
- The primary read-out is d = logit_Z − logit_W (comparable to P39v), not the W/Z-vs-QCD score. The q-based numbers are in the
  JSON.
- The E2 − E3 offset contrast and E2rec were added. Two debug restarts:
  - The first fixed an array-shape bug.
  - The second changed the E2 direction (the outward direction admitted 14 % of jets), fixed the E5 pooled denominator, and
    added the E2 − E3 contrast and the logits file.
