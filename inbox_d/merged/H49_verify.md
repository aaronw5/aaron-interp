# H49 verify: groomed vs ungroomed mass at matched added count, pT and ungroomed mass change

Verifier script: `interp/h49_verify.py`. It imports `interp/h49_groomed_mass.py` unchanged, so the edit code is the same; only the seeds differ, and two checks are added. Jobs:
- d-h49-verify (GPU, JetClass and Top Landscape, 987 s);
- d-h49-vhls (CPU 1/4 Gi, hls4ml JEDI-linear only, 209 s).

Both exited 0. Outputs:
- `results/physics/h49_verify_gpu.json`, `h49_verify_hls.json`: fresh-sample H49 summary plus the new tests;
- `h49_verify_summary.json`: the offset/slope decomposition below;
- `h49_verify_groomed_{gpu,hls}.png`: the H49 figure on the fresh samples.

## Conclusion
- **Reproduced.** On new random jets the H49 numbers come back. On hls4ml W/Z jets, JEDI-linear treats soft particles that soft drop keeps the same as ones it removes. On JetClass, ParT and ParticleNet lower the W/Z score by about 1.2–1.5 even for particles grooming removes, and lower it further for kept particles: ParT_full W, D = −0.53 in both samples.
- **Main correction: the extra response to kept particles is not a groomed-mass response.** We split jets by whether a groomed-mass-only classifier says the kept particles should raise or lower the score. The JetClass and top taggers follow that direction only about 0.4 times as strongly (top taggers 0.17–0.45). On top of that comes a penalty of up to −0.4 to −0.5 log-odds for kept particles that does not depend on direction. For ParT_full W jets, the score falls even where the groomed-mass classifier says it should rise.
- **Top taggers.** The claim that they respond "opposite to the groomed-mass references" is wrong. Jet by jet, they follow the groomed-mass direction weakly. Their mean is negative because of the same direction-independent penalty.
- **Still uncertain.** We do not know what the penalty reads. It is not the local particle density: groomed-away particles placed next to existing radiation are penalised less, not more. The prong-mass/shape change caused by the kept particles is still the leading candidate, and this run did not test it.

## What was checked in the code
- **Axes and coordinates.** The JetClass build uses η = jet_eta + sign(jet_eta)·Δη (`jc_table`), and the edited-jet build uses the sign of the new axis (`jc_maker`). hls4ml inputs are (pT, η − axis, φ − axis). Top inputs are taken about the given axis (`top_J`). No error found.
- **Samples.** All samples are random rows, not first-k: JetClass uses `rng.choice` per class file, hls4ml uses `hls_sample`, Top uses `data_top.sample`.
- **Observables.** Every observable is computed on the capped (finalized) particle list that is then passed to the model. Padded slots are excluded (pT > 0 masks) from soft drop, the mass, the count and ΣpT.
- **Edit matching.** The ungroomed mass is matched by bisection on the same capped list. On fresh JetClass W jets the medians are Δm(G_out) = +6.13 GeV and Δm_SD(G_out) = 0.0, with the G_in/G_out radius ratio 1.000.
- **c0.** c0 keeps ΣpT fixed by scaling the softest particles. It is not a 4-momentum-conserving edit, and it does not claim to be.
- **Two design issues (not bugs).**
  1. **G_out is a chosen placement.** It goes in the direction with the *least* existing pT. Since D = Δs(G_in) − Δs(G_out), D depends on that choice (control 3 below).
  2. **The bootstrap leaves out the reference fit.** The r_SD interval does not include refitting the MLP(m_SD) reference. On the fresh sample, MLP(m_SD)'s W-jet D is −0.35 against −0.46 in H49. ParT_full W r_SD becomes 1.52 [1.30, 1.76], outside the H49 interval [1.00, 1.32], although D_net itself is unchanged (−0.526 vs −0.529).
- **Figure.** `h49_groomed.png` shows the edit on a real JetClass W jet correctly: equal Δm = +7.2 GeV, Δm_SD +7.3 for G_in and 0.0 for G_out. The right panel's x-range is set by the hls4ml MLP(m_SD) bars (−4), which compresses the JetClass/Top differences.

## Definitions
- **s.** Log-odds of the signal class against QCD: logit_c − logit_QCD (JetClass), log p(c) − log(p(q) + p(g)) (hls4ml), or the network's top log-odds (Top).
- **Edits.** G_in and G_out are 3 massless neutral particles, each with q = 0.02·pT_J/3, on a circle of radius r* ≈ r_F + 0.7 R_g about the jet axis. R_g is the soft-drop prong separation (β = 0, z_cut = 0.1); r_F is the farther prong's distance from the axis.
  - G_in is placed just beyond the far prong, so soft drop keeps it: m and m_SD both rise.
  - G_out is placed ≥ 1.1 R_g from both prongs in the emptiest direction, so soft drop removes it: m rises, m_SD stays fixed.
  - Both edits add the same count, ΣpT and ungroomed Δm.
- **D.** Per-jet Δs(G_in) − Δs(G_out), averaged over signal jets in the H49 matched set.
- **MLP(m_SD), MLP(m).** Small classifiers that see only the groomed mass or only the plain jet mass. They are trained on unedited jets of the same sample (2-fold cross-fit) and evaluated on the edited jets.
- **D_ref.** MLP(m_SD)'s per-jet D: what a classifier that reads only m_SD would do.
- **r_SD.** D_net / D_ref, the network's G_in − G_out difference divided by the groomed-mass classifier's.

## Check 1: fresh-sample rerun
Seeds: hls4ml 4931 (H49 used 4901), JetClass 4932 (4902), Top 4933 (4903). The procedure is otherwise identical, with the same N: hls4ml 2,000 per class; JetClass 2,000 per class for QCD/Zqq/Wqq/Tbqq; Top 3,000 + 3,000.

Matched fractions:

| sample | W | Z | t | QCD |
|---|---|---|---|---|
| JetClass | 83.7 % | 80.2 % | 62.6 % | 63.6 % |
| Top Landscape | – | – | 55.0 % | 69.7 % |
| hls4ml | 78.3 % | 78.9 % | 57.6 % | 45.5 % / 34.9 % (g / q) |

| model, class (rc build) | D, H49 | D, fresh [95 % CI] | r_SD, H49 | r_SD, fresh [CI] | Δs G_out, fresh |
|---|---|---|---|---|---|
| hls4ml JEDI-linear W | −0.10 | −0.10 [−0.13, −0.06] | 0.02 | 0.03 [0.02, 0.04] | −3.21 |
| hls4ml JEDI-linear Z | +0.04 | +0.04 [+0.01, +0.07] | −0.01 | −0.01 | −3.21 |
| hls4ml JEDI-linear t | −0.34 | −0.36 [−0.40, −0.33] | 0.64 | 0.68 [0.58, 0.79] | −0.21 |
| JetClass ParT_full W | −0.53 | −0.53 [−0.59, −0.46] | 1.15 | 1.52 [1.30, 1.76] | −1.53 |
| JetClass ParT_full Z | −0.55 | −0.51 [−0.56, −0.46] | 1.39 | 1.31 [1.14, 1.52] | −1.29 |
| JetClass ParT_kin W / Z | −0.19 / −0.26 | −0.17 / −0.26 | 0.41 / 0.64 | 0.49 / 0.66 | −1.45 / −1.22 |
| JetClass PN_full W / Z | −0.34 / −0.32 | −0.33 / −0.31 | 0.74 / 0.80 | 0.95 / 0.80 | −1.42 / −1.23 |
| JetClass PN_kin W / Z | −0.31 / −0.34 | −0.28 / −0.32 | 0.67 / 0.85 | 0.79 / 0.83 | −1.52 / −1.31 |
| JetClass ParT_full t | −0.44 | −0.41 [−0.45, −0.37] | undefined | undefined | −0.24 |
| Top ParT / PN / LorentzNet t | −0.40 / −0.53 / −0.49 | −0.39 / −0.53 / −0.49 | – | – | −0.31 / −0.44 / −0.44 |
| Top PFN, PFN permuted (null) | −0.04, −0.03 | −0.06, −0.04 | – | – | – |
| Top EFN | +0.35 | +0.33 | – | – | −0.84 |
| reference MLP(m_SD) W / Z / t (JetClass) | −0.46 / −0.40 / 0.00 | −0.35 / −0.39 / −0.03 | 1 | 1 | 0 |
| reference MLP(m_SD) t (Top) | +0.11 | +0.05 [+0.01, +0.09] | 1 | 1 | 0 |

- **Holm.** Holm p < 0.05 for all 18 rc D tests (JetClass + Top, nulls included) and all 3 hls4ml JEDI tests.
- **Accuracy.** The unedited per-class accuracies and AUCs on the fresh samples match H49 and the known values (in the JSON).

## Check 2 (new): does the network follow the direction of the m_SD change?
- **Why this test is needed.** H49 compares mean D to mean D_ref. A mean ratio near 1 can come from following m_SD, or from a penalty on kept particles that does not depend on m_SD and happens to have the same sign as D_ref on average.
- **Using the sign of D_ref.** The sign of D_ref changes from jet to jet. A W jet with m_SD below the W peak moves toward the peak under G_in (+7 GeV), so MLP(m_SD) predicts a score rise. A jet above the peak moves away from it, so MLP(m_SD) predicts a fall.
- **Groups.** Signal jets are split by MLP(m_SD)'s own per-jet prediction: "rise" (D_ref > +0.05) and "fall" (D_ref < −0.05).
- **H0 (m_SD follower).** The network's D is positive in the rise group and negative in the fall group, in proportion to D_ref.
- **H1 (direction-independent penalty for kept particles).** D is negative in both groups.
- **Decomposition.** D_net = a + b·D_ref, fitted from the two group means. Here a is the part of D that does not depend on the m_SD direction, and b is the fraction of the groomed-mass classifier's response that the network follows.
- **Statistics.** Sign-flip p per group; Holm over 18 rise-group tests; Spearman ρ(D_net, D_ref) over all matched signal jets with a permutation p (2,000 permutations).

| model, class | D in rise group [CI] (D_ref +0.60 to +0.66) | D in fall group [CI] (D_ref −0.74 to −1.17) | a | b | Spearman ρ |
|---|---|---|---|---|---|
| JetClass ParT_full W | **−0.18 [−0.27, −0.09]** (Holm 0.007) | −0.86 [−0.95, −0.75] | −0.42 | 0.37 | 0.32 |
| JetClass ParT_full Z | **−0.14 [−0.22, −0.05]** (Holm 0.017) | −0.82 | −0.39 | 0.40 | 0.37 |
| JetClass ParT_kin W / Z | +0.23 / +0.14 | −0.51 / −0.55 | −0.04 / −0.12 | 0.40 / 0.40 | 0.49 / 0.48 |
| JetClass PN_full W / Z | +0.02 / +0.05 (n.s.) | −0.63 / −0.58 | −0.22 / −0.18 | 0.36 / 0.37 | 0.35 / 0.38 |
| JetClass PN_kin W / Z | +0.11 / +0.03 (n.s.) | −0.62 / −0.61 | −0.15 / −0.21 | 0.40 / 0.37 | 0.49 / 0.49 |
| JetClass ParT_full / PN_full t | −0.30 / −0.25 | −0.54 / −0.47 | −0.41 / −0.35 | 0.17 / 0.16 | 0.19 / 0.17 |
| Top ParT / PN / LorentzNet t | −0.26 / −0.34 / −0.24 | −0.59 / −0.82 / −0.88 | −0.40 / −0.54 / −0.51 | 0.24 / 0.34 / 0.45 | 0.21 / 0.23 / 0.31 |
| Top PFN / PFN permuted | +0.01 / −0.02 | −0.16 / −0.06 | −0.06 / −0.04 | 0.12 / 0.03 | 0.23 / 0.07 |
| hls4ml JEDI-linear W / Z | +0.06 / +0.12 | −0.13 / +0.03 | +0.02 / +0.10 | 0.03 / 0.01 | 0.17 / 0.18 |
| hls4ml JEDI-linear t | −0.27 | −0.45 | −0.32 | 0.08 | 0.14 |

All Spearman ρ have permutation p ≤ 5e-4.

The same picture in bins of the unedited m_SD, for ParT_full W, D_net against D_ref:

| m_SD bin [GeV] | 50–65 | 65–75 | 75–85 | 85–95 |
|---|---|---|---|---|
| D_net | −0.04 | −0.26 [−0.39, −0.14] | −0.78 | −0.90 |
| D_ref | +0.76 | +0.72 | −0.75 | −1.47 |

- **H0 excluded for every network as a full account.** All have b ≈ 0.4 or less, and all trained JetClass and Top networks except ParT_kin have a clearly nonzero offset a.
- **ParT_full W/Z.** The score falls where an m_SD reader's score rises. The H49 mean ratio r_SD ≈ 1.2–1.5 is a sum of about 0.4 of the m_SD response and a direction-independent offset of about −0.4. It does not show ParT_full reading m_SD "about like a groomed-mass classifier".
- **ParT_kin.** The closest to a partial m_SD follower (a ≈ 0, b = 0.40).
- **Top taggers.** Jet by jet they follow the m_SD direction (b 0.17–0.45, ρ 0.17–0.31, p ≤ 5e-4). H49's "opposite to the m_SD references" holds only for the mean, which is set by the offset a ≈ −0.4 to −0.54. On JetClass the m_SD reference mean is ≈ 0 (−0.03 [−0.08, +0.02]), so "opposite" does not apply there at all.
- **hls4ml JEDI-linear W/Z.** b ≤ 0.03, which confirms H49. The shared response to G_out follows the plain-mass classifier MLP(m): in jets where MLP(m) predicts a rise (+1.17) JEDI-linear gives +0.22, and where it predicts a fall (−3.41) it gives −3.72 (slope 0.86, offset −0.79). This supports "plain m (plus a pT/count offset)" over "count only".
- **hls4ml JEDI-linear t.** The top D of −0.36 (r_SD 0.68) is almost all offset (b = 0.08). H49's hls4ml-top r_SD of 0.64–0.70 should not be read as groomed-mass following either.

## Check 3 (new control): where the groomed-away particles go, at fixed m and m_SD
- **G_dense.** The same 3 particles are placed on the same circle r*, in the qualifying G_out direction (≥ 1.1 R_g from both prongs, so still groomed away) with the *most* unedited pT within 0.5 R_g, instead of the least. Count, ΣpT, radius and Δm are unchanged: median Δm 6.18 vs 6.13 GeV on JetClass W. Δm_SD = 0.
- **Jets used.** Those where the dense direction differs from the G_out direction and Δm matches to 1 GeV: JetClass W 574, Z 581, t 363; Top t 558.
- **Neighbourhood pT within 0.5 R_g (median).** G_out 0 GeV; G_dense 3.0 (W) / 3.3 (Z) / 3.7 (Top t) GeV; G_in 2.2 / 2.0 / 6.6 GeV.

| model, class | Δs(G_dense) − Δs(G_out) [CI] | D on the same jets |
|---|---|---|
| JetClass ParT_full W / Z | +0.21 [+0.15, +0.26] / +0.17 [+0.11, +0.21] | −0.60 / −0.49 |
| JetClass ParT_kin W / Z | +0.17 / +0.19 | −0.25 / −0.29 |
| JetClass PN_full W / Z | +0.14 / +0.16 | −0.42 / −0.34 |
| JetClass PN_kin W / Z | +0.27 / +0.26 | −0.37 / −0.36 |
| JetClass ParT_full t | −0.01 [−0.09, +0.06] | −0.36 |
| Top ParT / PN / LorentzNet | +0.15 / +0.26 / +0.19 | −0.37 / −0.50 / −0.43 |
| Top PFN / PFN permuted | +0.17 / −0.02 | −0.05 / −0.04 |
| hls4ml JEDI-linear W / Z / t | +0.00 / +0.01 / −0.05 | −0.10 / +0.02 / −0.28 |

- **What this shows.** At fixed m, m_SD, count and ΣpT, moving the groomed-away particles next to existing radiation raises the JetClass/Top scores by 0.14–0.27. That is 30–100 % of |D| for ParT_kin and the PN models.
- **H49's D depends on the G_out choice.** H49's D is measured against the most-penalised groomed-away placement. Against G_dense, D would be 0.14–0.27 more negative.
- **Local density does not explain the offset.** G_in's neighbourhood pT on JetClass (2 GeV) lies between G_out's and G_dense's, so local density predicts G_in to be penalised less than G_out. It is penalised more. The direction-independent offset a therefore comes from something else, for example the change of the joined prong's mass/width or of the two-prong shape, which was not tested here.
- **hls4ml JEDI-linear.** It does not care where the groomed-away particles go: +0.00 / +0.01.

## Corrected reading
- **hls4ml W/Z (JEDI-linear; small nets not rerun).** Confirmed. D is 0.03 or less of the groomed-mass reference, and the per-jet direction of the shared response follows MLP(m) (slope 0.86). The test still does not separate plain m from a ΣpT/count offset.
- **hls4ml t.** JEDI-linear's D of −0.36 is a direction-independent offset (b = 0.08), not groomed-mass following.
- **JetClass W/Z, response to groomed-away particles.** G_out alone lowers the score by 1.2–1.5 log-odds. That is far more than either mass reference (≤ 0.19) or c0. Confirmed.
- **JetClass W/Z, extra response to kept particles.** It is a partial m_SD response (b = 0.36–0.40 of MLP(m_SD), ρ 0.32–0.49) plus a direction-independent penalty a = −0.04 (ParT_kin W) to −0.42 (ParT_full W). The H49 ratios r_SD 0.4–1.4 (fresh 0.49–1.52) mix the two. They are also sample-dependent beyond their bootstrap intervals, because the MLP(m_SD) reference refits differ (W: −0.46 vs −0.35).
- **Top taggers (JetClass ParT/PN, Top ParT/PN/LorentzNet).** Same structure: b = 0.16–0.45 in the m_SD direction plus a = −0.35 to −0.54. They are not "opposite to the m_SD references".
- **D depends on the G_out placement.** Groomed-away particles next to existing radiation are penalised 0.14–0.27 less than in an empty region.

## Corrected Conclusion (for H49)
We added three soft particles to simulated jets in two ways with the same particle count, added pT and plain jet mass. Soft drop, the standard grooming step, removes the particles in one placement and keeps them in the other. The hls4ml W/Z taggers treat both placements alike and follow the plain mass, so they do not act as groomed-mass taggers. The JetClass and top taggers react strongly even to particles grooming would remove. To kept particles they add only a partial groomed-mass response, about 0.4 of what a groomed-mass-only classifier gives, plus an extra penalty of up to about 0.4–0.5 log-odds that does not depend on whether the groomed mass moves toward or away from the signal peak. What that penalty reads is not known; the change in the shape and mass of the prong the particles join is the leading untested candidate.

## Issues
- **Small nets not rerun.** The hls4ml small nets (Deep Sets, Lin-former, Transformer) were not rerun on the fresh sample: only JEDI-linear, since the conda/torch part needs the stored inputs.
- **Fitted groups.** The rise/fall groups are defined by the fitted MLP(m_SD), so a and b inherit its shape. The group means rest on 497–985 jets each.
- **Two-point decomposition.** a and b come from two group means, not a per-jet regression. The m_SD-bin table gives the same pattern.
- **Reduced G_dense set.** G_dense uses only jets where a second qualifying direction exists (about 35 % of matched W/Z jets).
- **Nothing committed.** One new script (`interp/h49_verify.py`); no changes to `h49_groomed_mass.py`.
