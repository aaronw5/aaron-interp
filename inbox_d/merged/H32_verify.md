# H32 verify: W decay angle inside top at fixed m_top and m_W

## Conclusion

The raw result replicates on a fresh sample. Rotating the W decay inside top jets to raise |cos θ*| (the W decay angle; the top mass and W mass stay fixed) lowers every kinematic-input tagger's top score more than the matched control does. The per-jet linear correction gives the Deep Sets models PFN and EFN a negative angle slope in every sample, seed, direction and model specification tested.

The H32 reading also says that ParT, ParticleNet and LorentzNet respond "mostly through τ32 / prong sharing". The verification does not support that. For these models, the same linear correction changes sign between raising and lowering the angle, so the pooled slope cannot be interpreted. A second check used the jets on which the control already matches τ32 and the prong pT fractions to within 0.01. On those jets, raising |cos θ*| by about 0.33 lowers the scores of JetClass ParT_kin, PN_kin, Top LorentzNet and Top ParT by 0.3 to 0.7 log-odds on JetClass jets. That is as much as PFN and EFN, and well beyond the angle-blind MLP (≤ 0.1). On Top Landscape jets the same check gives 0.2 to 0.35 log-odds. Still uncertain: that subset holds only 120 to 200 jets per sample, and it matches τ32 and the prong fractions but not every other substructure variable.

## Procedure checked

Code read: interp/h31_top_dalitz.py, interp/h32_top_dalitz_full.py, interp/h32_adjust.py, interp/h32_plot.py.

| check | result |
|---|---|
| Row sampling | Random rows (`H._sample` rng choice per class; `jc_read` rng among eligible rows). No first-k. |
| Padding | Every observable and subjet sum masks with `arange(L) < nn`. Edits act on `P4[j, :n]` only. |
| Conservation | D, cφ and c-side are Lorentz transformations of the constituents. The self-tests (rerun in the verify job) give a jet four-momentum change, m_top, m_W, intra-subjet masses and \|cos θ*\| all ≤ 1e-10. The mirrored c-side candidates move m_bq_hi/lo by about 0.9 GeV, as the report states. |
| Axis | Every variant is rebuilt about its own constituent-sum axis before scoring (`H25.axis`). Top-sample constituents are re-sorted by pT after editing. |
| Observables vs model inputs | Observables come from the same edited arrays the models score. On the Top sample, features are computed on the pT-sorted array; this is order-independent. |
| Working points | ε_S = 0.5 / 0.3 thresholds come from the unedited top jets of the same sample. |
| Adjustment columns | sorted-fraction column 13 is the largest fraction (labelled "hi"), column 11 the smallest. These are correct. |
| Regression reproduction | Refitting the H32 regression on h32_perjet.npz (job d-h32-vsplit) reproduces every H32 b_cos to the printed precision, e.g. PFN Top −1.82 [−2.22, −1.46]. |

No coding bug was found.

## Fresh-sample rerun (job d-h32-verify, GPU, exit 0, 27 min)

**Sample.** Seed 3301. Top Landscape test split, 2,000 top + 2,000 QCD jets, with no overlap with the H32 rows. JetClass TTBar_100 / ZJetsToNuNu_100 jets with stored pT between 550 and 650 GeV, 2,000 per class, with the H32 rows excluded. References: 5 retrains each of MLP3, MLP5, MLP5c and MLPx. Script: interp/h32_verify.py. Output: results/physics/h32_verify.json.

**Unedited models (AUC / accuracy), Top sample; JetClass sample.**
- Top ParT: 0.9855 / 94.3 %; 0.9874 / 94.5 %.
- Top PN: 0.9853 / 94.0 %; 0.9862 / 94.0 %.
- LorentzNet: 0.9865 / 94.5 %; 0.9891 / 94.7 %.
- PFN: 0.9799 / 92.6 %; 0.9816 / 93.0 %.
- EFN: 0.9762 / 92.3 %; 0.9811 / 93.5 %.
- ParT_kin: 0.9830 / 93.0 %; 0.9937 / 96.3 %.
- PN_kin: 0.9846 / 93.3 %; 0.9909 / 95.1 %.
- ParT_full (JetClass sample only): 0.9988 / 98.4 %.
- PN_full (JetClass sample only): 0.9983 / 98.0 %.
- MLPx: 0.9799 / 93.0 %; 0.9837 / 94.1 %.

**New control c-side2.** Same Lorentz family as c-side, with a wider top-frame tilt grid: ρ up to 0.7 rad and 12 azimuths, giving 3,504 candidates. Selection keeps the 30 closest candidates in prong fractions, then picks by a τ32-weighted cost (σ_τ32 = 0.005).

### Raw D − c-side, top jets (top log-odds, 95 % bootstrap CI, B = 2000)

| cell | n | networks (kinematic inputs) | MLPx (5 retrains) | c-r null (cr − cφ) |
|---|---|---|---|---|
| Top sample, Δc +0.3 | 411 | −0.08 (Top PN) to −0.85 | −0.37 to −0.50 | −0.20 to −0.03 |
| Top sample, Δc +0.6 | 196 | −1.61 to −2.18 | −1.19 to −1.50 | −0.22 to +0.63 |
| JetClass sample, Δc +0.3 | 549 | −0.87 to −1.26 | −0.44 to −0.55 | −0.21 to −0.11 |
| JetClass sample, Δc +0.6 | 260 | −1.98 to −2.57 | −1.26 to −1.55 | −0.29 to +0.33 |

- **τ32 match.** c-side reproduces 50 % (+0.3) and 37–40 % (+0.6) of D's τ32 change. c-side2 raises this only to about 59 % / 46 %. At +0.6 the c-side2 C2ST is 0.61 / 0.73 against D's 0.53 / 0.56.
- **Paired network − MLPx contrast at +0.3** (H0: network D−ctrl = MLPx D−ctrl; sign-flip permutation, maximum p over the 5 MLPx retrains, Holm over 16 network cells per family).
  - Top sample: 5 of 7 networks have Holm p ≤ 0.015. The exceptions are Top PN and PFN.
  - JetClass sample: 7 of 9 are more negative than MLPx with Holm p ≤ 0.015. PN_full also has Holm p 0.014, but in the opposite direction: its own D−ctrl is about −0.1.
  - So at +0.3 the angle-blind MLPx reaches about 40–60 % of most networks' contrast, not an equal amount.

### Linear adjustment, b_cos (log-odds per unit |cos θ*|)

The H32 design is used, pooled over all four Δc cells or split by direction. "ext" adds τ32², τ21, mSD, the middle fraction and the three sorted subjet masses as regressors.

| model | Top: fresh, all | Top: fresh, ext | Top: H32 sample, +Δc only | Top: H32 sample, −Δc only | JC: fresh, all | JC: fresh, ext | JC: H32 sample, +Δc only | JC: H32 sample, −Δc only |
|---|---|---|---|---|---|---|---|---|
| PFN | −1.19 [−1.51, −0.86] | −1.30 | −2.17 [−3.67, −0.92] | −0.98 [−1.57, −0.46] | −1.81 [−2.04, −1.57] | −1.93 | −1.17 [−2.33, 0.00] | −0.97 [−1.48, −0.45] |
| EFN | −1.24 [−1.44, −1.03] | −1.32 | −1.04 [−1.75, −0.38] | −1.13 [−1.47, −0.81] | −1.69 [−1.87, −1.50] | −1.73 | −1.68 [−2.40, −1.02] | −1.27 [−1.59, −0.97] |
| Top ParT | +0.07 [−0.23, 0.38] | −0.26 | −0.08 [−1.03, 0.84] | +0.42 [−0.01, 0.87] | −0.45 [−0.72, −0.18] | −0.73 | +0.93 [0.00, 1.79] | +0.53 [0.09, 0.97] |
| Top PN | +0.63 [−0.09, 1.34] | +0.23 | −1.56 [−3.39, 0.26] | +0.95 [0.19, 1.67] | −0.59 [−1.04, −0.14] | −0.82 | +0.52 [−1.79, 3.09] | +1.15 [0.42, 1.92] |
| LorentzNet | −0.13 [−0.43, 0.18] | −0.42 | −0.06 [−1.10, 0.89] | +0.28 [−0.16, 0.71] | −0.79 [−1.14, −0.44] | −0.89 | +0.05 [−1.16, 1.18] | +0.52 [−0.06, 1.10] |
| ParT_kin | −0.26 [−0.48, −0.03] | −0.38 | +0.13 [−0.71, 0.94] | +0.26 [−0.09, 0.60] | −0.52 [−0.78, −0.24] | −0.68 | +0.46 [−0.42, 1.29] | +1.11 [0.65, 1.59] |
| PN_kin | −0.14 [−0.38, 0.09] | −0.37 | −0.17 [−0.94, 0.57] | +0.14 [−0.19, 0.47] | −0.65 [−0.90, −0.41] | −0.81 | −0.14 [−0.96, 0.65] | +0.80 [0.38, 1.26] |
| ParT_full | — | — | — | — | −0.48 [−0.86, −0.08] | −0.85 | +0.94 [−0.29, 2.16] | +0.47 [−0.31, 1.22] |
| PN_full | — | — | — | — | −0.82 [−1.15, −0.51] | −1.18 | +0.93 [−0.12, 1.91] | −0.24 [−0.93, 0.43] |
| MLPx range | −0.25 to −0.01 | −0.27 to −0.13 | +0.10 to +0.45 | −0.22 to −0.04 | −0.35 to −0.10 | −0.36 to −0.18 | −0.08 to +0.16 | −0.18 to −0.02 |

The fresh-sample direction split gives the same pattern (results/physics/h32_verify.json, regression cside|pos and cside|neg).

**Network b_cos − MLPx b_cos.** Paired bootstrap, maximum p over the 5 retrains, Holm over 16 networks × samples.
- PFN and EFN: Holm p = 0.008 in every specification, on both samples.
- Other models, fresh pooled fit: LorentzNet on JetClass has Holm p 0.154 (0.008 with c-side2). PN_full on JetClass has 0.036. All others are ≥ 0.15.
- Other models, "ext" fit: PN_kin, LorentzNet and PN_full on JetClass reach Holm p ≤ 0.04.

**Per-model reproducibility of the H32 classification.**
- Top PN on the Top sample: −0.68 [−1.19, −0.16] in H32, +0.63 [−0.09, 1.34] fresh.
- PN_kin on the Top sample: −0.64 [−0.87, −0.42] in H32, −0.14 [−0.38, 0.09] fresh.
- The "just outside the leak range" call for PN_kin on the Top sample does not replicate. The call for LorentzNet on the JetClass sample replicates: −0.67 in H32, −0.79 fresh.

### Non-parametric check: jets with a small residual mismatch

Selection: top jets, +Δc cells, where the D − c-side residual satisfies |Δτ32| < 0.01, |Δf_largest| < 0.01 and |Δf_smallest| < 0.01. Metric: raw D − c-side contrast, 95 % CI.

| model | Top sample, H32 (n 123, Δ\|cos\| +0.33) | Top sample, fresh (n 116, +0.33) | JetClass sample, H32 (n 161, +0.35) | JetClass sample, fresh (n 175, +0.33) |
|---|---|---|---|---|
| Top ParT | −0.29 [−0.50, −0.08] | −0.10 [−0.40, 0.20] | −0.27 [−0.49, −0.05] | −0.50 [−0.72, −0.27] |
| Top PN | +0.11 [−0.36, 0.61] | +0.62 [0.18, 1.07] | −0.03 [−0.46, 0.41] | −0.38 [−0.68, −0.07] |
| LorentzNet | −0.22 [−0.46, 0.02] | −0.27 [−0.51, −0.04] | −0.61 [−0.92, −0.28] | −0.67 [−0.99, −0.35] |
| PFN | −0.24 [−0.50, 0.01] | +0.10 [−0.18, 0.39] | −0.58 [−0.79, −0.37] | −0.47 [−0.63, −0.29] |
| EFN | −0.59 [−0.80, −0.40] | −0.48 [−0.71, −0.25] | −0.60 [−0.80, −0.41] | −0.59 [−0.76, −0.41] |
| ParT_kin | −0.23 [−0.42, −0.05] | −0.28 [−0.47, −0.10] | −0.37 [−0.58, −0.15] | −0.46 [−0.66, −0.26] |
| PN_kin | −0.35 [−0.52, −0.18] | −0.17 [−0.39, 0.05] | −0.40 [−0.61, −0.19] | −0.44 [−0.62, −0.25] |
| ParT_full | — | — | +0.20 [−0.12, 0.54] | +0.15 [−0.14, 0.42] |
| PN_full | — | — | +0.29 [0.00, 0.62] | +0.04 [−0.20, 0.27] |
| MLPx range | −0.15 to −0.08 | −0.04 to +0.03 | −0.09 to 0.00 | −0.09 to +0.01 |

**Fresh JetClass sample, network − MLPx.** Holm p ≤ 0.008 for Top ParT, LorentzNet, PFN, EFN, ParT_kin and PN_kin (16-test family).

**Fresh Top sample, network − MLPx.** Holm p ≤ 0.05 for EFN (0.006) and ParT_kin (0.049) only.

**Lowering |cos θ*| (−Δc matched subsets, n 880–1120).**
- Most kinematic networks: |contrast| ≤ 0.1.
- PFN: +0.3 to +0.45.
- ParT_full / PN_full: +0.4 to +0.6, as in H32's c-side − cφ response.

## Hypotheses

- **H0:** at matched τ32 and prong fractions, the network's response to D equals the angle-blind MLPx's.
- **H1:** the network has an additional |cos θ*| dependence.

| prediction | data | status |
|---|---|---|
| "τ32 side effect only": b_cos for ParT / PN / LorentzNet stays within the MLPx range in both directions, and the matched-subset contrast is ≈ MLPx. | On JetClass jets the matched-subset contrasts are −0.27 to −0.67 against MLPx ≤ 0.09 (H32 sample and fresh sample). The direction-split b_cos is positive at −Δc. | Excluded on JetClass jets. Not excluded on Top Landscape jets for Top ParT, Top PN and PN_kin (fresh). |
| "Angle-specific for PFN/EFN only": PFN/EFN outside MLPx, all others inside, in every check. | ParT_kin and LorentzNet lie outside MLPx in the matched subsets on 3 of 4 sample × seed combinations. PFN is inside on the fresh Top sample. | Excluded as stated. |
| "Angle-specific for all kinematic networks at +Δc, of similar size." | Supported on JetClass jets for 6 of 7 kinematic networks (not Top PN on the H32 sample). On Top Landscape jets the effects are smaller and less consistent. | Not excluded. |

## Figure (results/physics/h32_dalitz.png)

- **What it shows.** The figure plots raw D − c-side with bootstrap CIs. Its references are MLP5c (the reference whose sign the report calls unstable) and MLPx; MLP5, the stable positive control, is not shown.
- **Title.** The title says MLPx "responds about as much as the networks at Δ|cos θ*| = +0.3". That holds for the Top Landscape panel. On the JetClass panel most networks sit at −0.85 to −1.25, against MLPx at about −0.5.
- **What it leaves out.** The figure does not show b_cos, which carries the model ranking.

## Corrected reading

- **Raw contrast.** Raising |cos θ*| at fixed m_top and m_W lowers every kinematic-input network's top score more than c-side does. This replicates on a fresh seed for both samples.
- **Angle-blind reference.** c-side matches about half of D's τ32 change. The angle-blind MLPx reaches about 40–60 % of most networks' raw contrast at +0.3.
- **PFN and EFN.** The per-jet linear adjustment gives b_cos of −1.0 to −2.2 for PFN and EFN on both samples, both seeds, both edit directions and with extra side-effect regressors. These values lie outside the MLPx leak.
- **Other networks.** For ParT, ParticleNet, LorentzNet and the full-input models, the adjustment gives b_cos ≈ 0 or positive within each edit direction. Their negative pooled b_cos comes from the difference between directions, so it does not measure an angle slope.
- **Matched jets.** A non-parametric check on jets whose residual τ32 and prong-fraction mismatch is below 0.01 gives a different ranking. Raising |cos θ*| by about 0.33 lowers ParT_kin, PN_kin, LorentzNet, Top ParT, PFN and EFN by 0.3 to 0.7 log-odds on JetClass jets, on both seeds, against MLPx ≤ 0.09. On Top Landscape jets the drops are 0.2 to 0.6 log-odds and reach significance against MLPx only for EFN and ParT_kin (fresh seed). Top ParticleNet and the JetClass full-input models show no response on this subset.
- **Overall.** The data support an angle-specific response for most kinematic-input taggers on JetClass-like jets, with PFN/EFN the only models for which every method agrees. They do not support "mostly a τ32 side effect for transformer and graph taggers".

## Issues

1. **Unsupported model ranking.** Model ranking: the claim that the "PFN/EFN only" difference comes from the linear adjustment rests on a model that changes sign between edit directions for the non-Deep-Sets networks. The matched-subset check does not reproduce that ranking on JetClass jets.
2. **Per-model calls not reproducible.** The per-model "just outside / inside the leak range" calls are not reproducible across seeds: Top PN on the Top sample and PN_kin on the Top sample. These calls also compare a CI with a range of MLPx point estimates rather than testing the paired difference.
3. **"About as much" overstates.** "MLPx drops about as much at +0.3" overstates the match. On the fresh sample, 12 of 16 network cells at +0.3 are more negative than MLPx with Holm p ≤ 0.015 (5 of 7 on the Top sample, 7 of 9 on the JetClass sample).
4. **Control limit.** c-side2 (a wider tilt grid with τ32 priority) does not close the τ32 gap: about 59 % / 46 % matched, with higher C2ST. The tilt/rotation family cannot reproduce D's τ32 change, so a non-rigid control at fixed prong mass is still needed (as H32 proposes).
5. **Matched-subset limits.** The matched subset is small (116–200 jets). It is selected on the residual mismatch, and it leaves the mSD, τ21 and subjet-mass mismatch uncontrolled.
6. **Acceptance asymmetry.** At +0.6, D is accepted for only 20–24 % of jets, against 50–63 % for cφ. The +Δc cells are therefore jets with a low starting |cos θ*| whose W pairing survives re-clustering. The report notes this.

## Jobs and files

- d-h32-verify (GPU, exit 0, 27 min): fresh seed 3301, c-side and c-side2, cells, regressions (all / pos / neg / ext / ext_pos), matched subsets. Smoke run: d-h32-verify-smk.
- d-h32-vsplit (CPU 1 / 2 GiB, 39 s, exit 0): direction-split regression and matched subset on the original H32 per-jet file.
- Scripts: interp/h32_verify.py, interp/h32_verify_split.py.
- Results: results/physics/h32_verify.json, h32_verify_split.json (h32_verify_smoke.json).
