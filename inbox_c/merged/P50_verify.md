# P50 verify: ParticleNet EdgeConv-1 edge function and the axis response

Verifier for node P50 (loop C). Script `interp/p50_verify.py` (imports the P50 functions, adds one feature set and a
fit-quality ladder). Job `anrunw-ai-c-p50-verify` (1 GPU, Quadro RTX 6000, 15 min, exit 0; first two submissions were rejected
at admission on node k8s-chase-ci-01 with "GPU is lost", so the third submission excluded that node through a scratch copy of
part_job.sh; the repo's part_job.sh is unchanged). Results: `/j-jepa-vol/aaron-interp/results/part/p50_verify.json`, copied to
`results/part/p50_verify.json`; log `logs/c-p50-verify.log`.

## Conclusion

The direction of P50's result replicates on new jets. When ParticleNet's first graph layer is limited to each particle's own
inputs plus rotation-invariant pair quantities, the network's response to a mass-preserving shift of the jet axis falls (from
0.36 to 0.26 of a real mass change in the kinematics-only model), so ParticleNet does not move toward ParT's larger response.
Two parts of P50's reading do not hold as stated. About half of that fall also appears when a layer that does see the pair
direction is fitted equally badly, so only part of it is specific to the removed information. And the information whose removal
lowers the response is the neighbour's distance to the jet axis: adding only that unsigned number restores the response as
fully as adding the signed pair direction. This is axis-relative position information, not the pair orientation as such.
Not tested here: the training control (untrained network) for the kinematics-only model.

## What was checked in the code (interp/p50_pn_edge_function.py)

- Layer and module: the substitution replaces `msg` of `md.edge_convs[0]` (EC1) and `[1]` (EC2) inside an own forward that
  matches weaver 0.4.17 to |Δlogit| = 0 (P50 log). The kNN of EC1 uses (Δη, Δφ) with padded slots pushed to 1e9, as in weaver.
- Padded particles: recorded centres are real particles only (`keep & mask`); padded neighbours get a flag and zeroed pair
  invariants and offsets in the fits; in the substituted network padded centres are masked after the layer as in the original.
- Row selection: random rows per class file (`default_rng`), no first-k.
- η build: `part_core.build` uses η = jet_eta + sign(jet_eta)·Δη (the fixed build).
- Edits: φ + 0.06 on every real particle; max |Δ ln m| 0.0006 (P50) and 0.0003 (verify); masks unchanged; the matched
  dilation multiplies (Δη, Δφ) of each jet by k_j so √(M0·M2) moves as under the shift (mean k 1.08). Neither edit changes the
  EC1 kNN graph (both are rigid or uniform in the (Δη, Δφ) plane). The shift changes each particle's own Δφ_i and ΔR_i, and the
  difference ΔR_j − ΔR_i; it does not change Δη_ij, Δφ_ij or the four pair invariants.
- Null arm: fitted on shuffled neighbours, applied to the true neighbours; the comment's argument (same mean over the 16) holds.
- No bug found that changes the numbers. One framing point: feature set B adds (Δη_ij, Δφ_ij); together with the centre's own
  (Δη_i, Δφ_i) already in A, this gives the neighbour's position about the jet axis, not only the pair's orientation. The verify
  adds arm D to separate the two.
- Missing control in P50: the untrained and weight-permuted networks were run for PN_full only. The headline B − A gain
  (+0.13) and the R result are for PN_kin, which has no training control. Not run here.

## Procedure (verify)

- Sample: val_5M `*_124` of every class (P50 used `*_123`). Evaluation: 1,000 random rows per class (default_rng(6060)),
  N 10,000. Fit: 500 random rows per class from the remaining rows (default_rng(6061)), N 5,000, split 80/20 by jet; 35 % of
  real particles per fit jet kept as centres with their 16 edges (69,274 centres per model).
- Models: ParticleNet_kin (PN_kin: 7 kinematic inputs) and ParticleNet_full (PN_full: 17 inputs), official weights.
- Arms: all EC1 edge messages $e_{ij}=h_1(x_i, x_j-x_i)$ replaced by an MLP fit (2 hidden layers), rest of the network intact,
  later kNN graphs recomputed.
  - Main fits (width 256, 12,000 steps, P50's primary setting): A = centre input $x_i$ + the 4 pair invariants (ln kT, ln z,
    ln ΔR_ij, ln m²_ij); B = A + signed (Δη_ij, Δφ_ij); C = A + non-angular input differences; F = exact inputs (ceiling).
  - **New arm D** = A + (ΔR_j − ΔR_i) in the network's normalised units: the neighbour's distance to the jet axis relative to
    the centre's, with no sign of direction in the (η, φ) plane.
  - **New control, fit-quality ladder**: B and F (which contain the neighbour's axis position) refitted with small or briefly
    trained MLPs (width 8–64, 150–3,000 steps) so that their held-out R² spans A's. Question: does any equally poor fit lower R
    as much as A does (then A's drop is generic fit error), or only fits that lack the neighbour's position?
- Metric: R = slope through 0, over true W and Z jets, of Δd under the φ + 0.06 rigid shift against Δd under the matched
  dilation, d = logit_W − logit_Z. In words: the network's W-vs-Z response to a shift that leaves the invariant mass unchanged,
  as a fraction of its response to a real mass change of the same axis-moment size. ΔR = arm − unedited. 95 % bootstrap over
  jets (2,000); paired permutation p (2,000 per-jet swaps, smallest p 5e-4); McNemar for call changes. H0: ΔR = 0;
  H1: ΔR ≠ 0. Comparability rule: accuracy within 1 point.

## Numbers

Unedited (verify sample): PN_kin 70.57 % [69.65, 71.42], R 0.361 [0.336, 0.386] (P50: 69.87 %, 0.378); PN_full 84.70 %
[83.98, 85.41], R 0.524 [0.496, 0.551] (P50: 84.66 %, 0.507). R_φ − R_η: PN_kin +0.050 [+0.012, +0.090], PN_full +0.060
[+0.016, +0.102].

PN_kin, EC1 replaced (held-out R² per edge; Δacc points [CI]; ΔR [CI]; permutation p):

| arm | R² | Δacc | R | ΔR vs unedited | p | within 1 pt |
|---|---|---|---|---|---|---|
| A (x_i + invariants) | 0.802 | −1.28 [−1.77, −0.78] | 0.261 | −0.100 [−0.113, −0.086] | 5e-4 | no (P50: −0.92, yes) |
| B (A + signed Δη_ij, Δφ_ij) | 0.927 | −0.26 | 0.347 | −0.014 [−0.019, −0.010] | 5e-4 | yes |
| C (A + non-angular diffs) | 0.826 | −1.17 | 0.276 | −0.085 [−0.096, −0.074] | 5e-4 | no |
| **D (A + ΔR_j − ΔR_i)** | 0.893 | −0.47 | **0.348** | −0.013 [−0.018, −0.007] | 5e-4 | yes |
| F (exact inputs) | 0.933 | −0.21 | 0.340 | −0.020 [−0.025, −0.016] | 5e-4 | yes |
| ladder B, width 64, 3,000 steps | 0.816 | −0.63 | 0.310 | −0.051 [−0.061, −0.040] | 5e-4 | yes |
| ladder F, width 64, 3,000 steps | 0.808 | −1.04 | 0.338 | −0.023 [−0.034, −0.011] | 5e-4 | no (by 0.04) |
| ladder A, width 64, 3,000 steps | 0.742 | −1.46 | 0.231 | −0.130 [−0.145, −0.114] | 5e-4 | no |

Between arms (PN_kin, paired): B − A +0.085 [+0.073, +0.097], p 5e-4 (P50: +0.091); D − A +0.087 [+0.075, +0.099], p 5e-4;
C − A +0.015 [+0.007, +0.022], p 5e-4; **B − D −0.002 [−0.006, +0.003], p 0.43**.

Matched fit quality (PN_kin): at R² ≈ 0.81 and accuracy within 0.65 points of A, the oriented fits lose 0.023–0.051 of R, A
loses 0.100. Of A's drop, 0.05–0.08 is not reproduced by an equally poor oriented fit; the rest is. A linear fit of R on R²
over all B/F arms gives an A excess of −0.052 [−0.064, −0.040] (bootstrap over jets), but that line includes ladder arms that
lost 16–28 points of accuracy, so the direct matched pairs above are the better comparison.

PN_full (none of A/B/C/D within 1 point of unedited, as in P50): A −4.44 points, ΔR −0.087 [−0.109, −0.066]; B −1.86, −0.030;
D −2.76, −0.047; F −0.45, −0.013 [−0.022, −0.005]. **Matched pair not available in P50**: ladder B (width 64, 3,000 steps)
has R² 0.500 vs A 0.523 and accuracy −4.79 vs −4.44 points (0.35 apart, so comparable with each other); its ΔR is −0.029
[−0.049, −0.010] vs A's −0.087. So in PN_full, too, the fit without the neighbour's axis position lowers R by about 0.06 more
than an oriented fit of the same quality and accuracy. In PN_full D recovers only part of B (B − D +0.018 [+0.005, +0.030],
p 0.003).

Direction dependence (change in R_φ − R_η vs unedited): PN_kin A +0.019 [+0.003, +0.035] (P50: +0.015 [−0.002, +0.031]);
PN_full A +0.025 [−0.004, +0.053]. A substitution does not remove the φ-vs-η difference (unedited +0.05), consistent with P50.

## Hypotheses and which are excluded

- H_fit (A's R drop is generic fit error): predicts equally poor oriented fits lose as much R. Partly excluded: B-ladder at
  matched R² loses about half, F-ladder about a quarter.
- H_orient (the signed 2-d pair direction keeps R up): predicts D (unsigned radial difference only) behaves like A. Excluded
  in PN_kin: D ≈ B (difference −0.002, p 0.43).
- H_axispos (the neighbour's position about the jet axis keeps R up): predicts D ≈ B ≈ F in R. Consistent in PN_kin; in
  PN_full D recovers about two thirds of B's R over A.
- H_inv-toward-ParT (invariant-only edges move PN's R toward ParT's ≈ 0.8): excluded in both samples (R falls).

## Corrected reading

- Replicates: invariant-only EC1 (A) lowers R in PN_kin (0.361 → 0.261 here; 0.378 → 0.274 in P50), and B restores it;
  PN_kin's B − A R² gain replicates (0.80 → 0.93).
- Correction 1 (comparability): on the fresh sample PN_kin A costs 1.28 points [0.78, 1.77], outside the 1-point rule; the
  ladder arm F (width 64) is within 0.24 points of A and has R 0.338 vs 0.261, which supplies the accuracy-matched comparison.
- Correction 2 (generic fit error): about half of A's R drop (0.04–0.05 of 0.10) also appears in an oriented fit of the same
  R² and accuracy; the part specific to the removed information is about 0.05–0.08.
- Correction 3 (what the information is): the R-restoring information is the neighbour's distance to the jet axis relative to
  the centre (D), not the signed pair direction; with x_i already containing the centre's (Δη_i, Δφ_i), B's "orientation" is
  equivalent to the neighbour's axis-relative position. "Putting the signed direction back restores the response" should read
  "giving the layer the neighbour's position about the jet axis (even only its radial distance) restores the response".
- Correction 4 (PN_full): an accuracy-matched comparison exists (A vs width-64 B, 0.35 points apart) and shows the same sign
  (A lower by ≈ 0.06).
- Unchanged: hinge (additive) fits reach about half of the variance (not re-run); direction dependence is not removed by A.
- Still missing: untrained / permuted controls for PN_kin; hinge result not re-tested on new jets.

## Figure check (results/particlenet/p50_pn_edge_s12000.png)

Panels match the JSON. In panel 3 the x-tick labels "B: + signed Δη, Δφ" and "C: + non-angular differences" overlap, and the
legends of panels 1–2 cover the top y tick; readable. Panel 3 labels arm B as "signed Δη, Δφ"; per correction 3 a caption
should say this also gives the neighbour's position about the axis.

## Glossary additions (for site/explain.py)

- **Arm D**: EC1 messages fitted from the centre input, the four pair invariants and ΔR_j − ΔR_i (difference of the two
  particles' distances to the jet axis); carries the neighbour's radial axis position but not the direction in the (η, φ)
  plane.
- **Fit-quality ladder**: the same feature set fitted with deliberately small or briefly trained MLPs, giving a series of
  substitutions with lower held-out R²; used to separate "the fit lacks information X" from "the fit is just worse".
