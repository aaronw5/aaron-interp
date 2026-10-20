# H13 verify — soft contamination / pileup panel

Script `interp/h13_verify.py`. Jobs:
- `d-h13-verify` (hls4ml with classical observables, then topwp; the observable step runs slowly in the jax env and was still running when this report was written, see Status).
- `d-h13-verify-hls2` (the same hls4ml run without the observable step, then the summary).
- `d-h13-verify-jc` and `d-h13-verify-top` (GPU).
- Smoke runs: `d-h13-vsm-{hls,jc,top}`.

Results are in `results/physics/h13_verify_{hls_noobs,jc,top,topwp,summary}.json`.

## Conclusion
On fresh samples the pileup-like edit gives the same efficiency losses as H13. Shifting the jet axis by the small amount the edit causes still accounts for almost none of the response.

The added control changes how the response should be read. It keeps the same number of soft particles and the same added pT but puts them at the jet centre, so the jet mass barely changes.
- **Most networks.** For the JetClass ParT and ParticleNet models that use particle ID, and for the Top-Landscape ParT, LorentzNet and PFN, that control alone gives most or all of the response. One particle that adds the same mass and pT gives little. These networks therefore respond to the number of added soft particles, not to the mass the particles add.
- **JEDI-linear.** It is the opposite: the mass-matched single particle reproduces its loss of W calls, and the centred soft particles do not (W efficiency at ρ = 5: 39 % vs 88 %).
- **Top working point.** Compared at the same starting efficiency, the Top networks lose tags faster than the cut-based taggers at low and medium density. At ρ = 40 that no longer holds for every network against the ungroomed top(m, τ32) tagger.

Still uncertain:
- The centred-particle control also changes where the soft particles sit relative to the hard prongs.
- Each network's internal use of particle count is not identified.

## Code review of `interp/h13_pileup_panel.py`
- **Axis and η convention.**
  - JetClass uses η = jet_eta + sign(jet_eta)·Δη, through `jc_table` / `part_core.build`; H13 logs a round-trip max diff of 6e-5.
  - Added particles are placed in absolute (η, φ) about the stored axis, and the inputs are rebuilt about the axis passed in.
  - `build` recomputes jet_pt and jet_energy from the particles. No problem found.
- **Padding.** `eaxis`, `moments`, `jet4` and `mass_all` all mask pt > 0, and `finalize` zeroes padded slots. No problem found.
- **Row selection.** JetClass uses random rows per class. Observables use rows [::5] of the full sample, and the edit is generated on the full sample with the same `default_rng([seed, rho, chunk])` and the same spectrum. The observables are therefore computed on exactly the jets the models saw. No first-k subsampling.
- **Energy and momentum.** The edit adds particles and does not claim conservation. c2 adds the injected ΣpT on the axis as stated.
- **Working points.** The windows match those listed (W 65–95 GeV; top 105–210 GeV with τ32 < 0.65).
- **Issue: hls4ml all-particle W(m).** The W(m) reference uses the mass of all particles, but JEDI-linear sees at most 64 slots (overflow 31–99.7 %). The fresh run gives W(m_kept) of 37.6 % against W(m_all) of 36.9 % at ρ = 5, and 6.4 % against 5.4 % at ρ = 10, so this does not change the comparison.
- **Issue: c2 on hls4ml.** c2 adds the full injected ΣpT (80 GeV at ρ = 40), but in pu_40 JEDI-linear sees only 42.6 GeV of added pT after the slot cut. c2_40 on hls4ml is therefore not matched in pT to what the model sees. The new c2k control matches the kept pT.
- **Issue: Top panel working point.** The figure's Top panel compares networks at ε_S = 0.5 with cut-based taggers that start at 66–70 %. A tighter working point loses a larger fraction under the same shift. This is now tested below.
- No bug was found that changes a number in the H13 tables.

## Procedure
**Samples (H13 seed-302 rows excluded; overlap 0).**

| dataset | sample | N | jets per class |
|---|---|---|---|
| hls4ml | n64 test jets, seed 3131 | 10,000 | 2,000 |
| JetClass | first test file per class, seed 3132 | 5,000 | 500 |
| Top Landscape | test jets, seed 3133 | 5,000 | 2,500 top + 2,500 QCD |

**Edit and H13 conditions.** These use H13's own functions (`inject`, `finalize`, `eaxis`, `spectrum`):
- pu_ρ: the edit, with the axis recomputed.
- pufix_ρ: the edit, with inputs about the stored axis.
- c3_ρ: the original particles, translated by the pu axis offset.
- ρ ∈ {5, 10, 20, 40} GeV per unit η–φ area.

**New controls.** Inputs are taken about the stored axis, so each compares with pufix_ρ and carries no axis offset.
- **c4_ρ (centred particles).** The same injected particles (same count and pT), with their displacement from the axis divided by 20 (r ≤ 0.04). Multiplicity and added ΣpT are the same as pufix. The added mass shrinks to Δm_kept = 0.2 GeV (hls4ml, ρ 5) or 1.0 GeV (JetClass, ρ 5).
- **c5_ρ (one wide particle).** One massless particle at a random azimuth:
  - its pT is the ΣpT that pufix adds to the kept particles;
  - its radius is set per jet by bisection so that the kept-particle invariant mass equals pufix's (median |Δm| < 1e-12 GeV; 3–5 % of hls4ml jets reach r = 2.0).
  - It matches kept ΣpT and mass, with multiplicity +1.
- **c2k_ρ.** The c5 particle placed on the axis: the same kept ΣpT, mass change of 0.3–9 GeV, multiplicity +1.

**Predictions.**

| response driven by | c5 | c4 | c2k |
|---|---|---|---|
| mass | ≈ pufix | small | small |
| soft-particle count | small | ≈ pufix | — |
| ΣpT only | ≈ pufix | ≈ pufix | ≈ pufix |

**Metrics.**
- % calls changed versus the unedited jet, with a 95 % binomial bootstrap interval.
- Signal efficiency (argmax = W on true W jets). For Top, the tag rate on true tops at a threshold frozen on unedited tops.
- Paired McNemar between conditions. Holm correction over all 240 tests of the four parts.
- Per-jet concordance on true W jets: P(net calls W | 65–95 GeV window passes) minus P(net calls W | window fails), with a bootstrap interval, plus Cohen's κ.

**topwp.** Uses the H13 Top per-jet files (5,000 + 5,000 jets).
- Network thresholds are frozen on unedited tops at ε_S = 0.5 and at each cut-based tagger's own starting efficiency.
- Retention is the tag rate after the edit divided by the rate before, on true tops.
- The comparison is paired with the cut-based tagger on the same rows [::5] (1,036 tops), with 2,000 bootstrap resamples.

## Reproduction on the fresh samples (unedited accuracy on the sample used)

**Unedited accuracy or AUC.**
- hls4ml: JEDI-linear 0.811 [0.804, 0.818] (H13 0.808). The redrawn null scores 0.200.
- JetClass:
  - ParT_full 0.856 (H13 0.858)
  - ParT_kin 0.743 (H13 0.735)
  - PN_full 0.840 (H13 0.843)
  - PN_kin 0.707 (H13 0.700)
- Top AUC (H13 in brackets):
  - ParT 0.987 (0.987)
  - PN 0.986 (0.987)
  - LorentzNet 0.987 (0.988)
  - PFN 0.982 (0.982)
  - EFN 0.978 (0.979)
  - Permuted PFN 0.718

**Signal efficiency, % (ρ = 0 → 5 / 10 / 20 / 40).**

| tagger | fresh | H13 |
|---|---|---|
| hls4ml JEDI-linear, true W | 88.2 → 43.1 / 5.1 / 0.1 / 0.0 | 87 → 42 / 5 / 0 / 0 |
| hls4ml W(m), all particles | 93.6 → 36.9 / 5.4 / 0.1 / 0.0 | 93 → 36 / 6 / 0 / 0 |
| JetClass ParT_full, true Wqq | 80.0 → 72.0 / 63.6 / 45.8 / 26.2 | 80 → 71 / 62 / 45 / 26 |
| JetClass PN_full | 78.0 → 70.2 / 58.6 / 32.2 / 9.6 | 76 → 67 / 55 / 30 / 7 |
| JetClass W(m) | 74.4 → 49.4 / 17.2 / 1.4 / 0.0 | — |
| JetClass W(m_SD), rows [::5] | 74 → 74 / 74 / 68 / 65 | 72 → 72 / 72 / 68 / 61 |
| Top ParT, ε_S 0.5 frozen, retention | 0.71 / 0.44 / 0.14 / 0.01 | 0.72 / 0.47 / 0.17 / 0.02 |
| Top top(m_SD, τ32), retention | 0.95 / 0.88 / 0.73 / 0.29 (eff0 0.64) | — |
| Top top(m, τ32), retention | 0.92 / 0.84 / 0.63 / 0.07 (eff0 0.69) | — |

**Axis share of the response.**
- Axis offset, median (p90), ρ 5 → 40:
  - hls4ml 2.6 (6.7) → 5.5 (10.6) mrad
  - JetClass 3.6 (7.8) → 12.9 (25.2) mrad
  - Top 2.6 (5.0) → 12.2 (21.1) mrad
- c3 (pure translation) changes calls:
  - JEDI-linear 1.8–2.5 %
  - JetClass 0.4–2.5 %
  - Top 0.0–1.8 %
- pu vs pufix differ by ≤ 0.5 pp everywhere. McNemar Holm p ≥ 0.17 for all trained models.

These reproduce H13's efficiency, axis-offset and c3 numbers.

## New controls: which content of the edit drives the response

**JEDI-linear (hls4ml): W efficiency, % (calls changed, %).**

| ρ | pufix | c5 (mass + ΣpT, one particle) | c4 (count + ΣpT, Δm ≈ 0) | c2k (ΣpT on axis) |
|---|---|---|---|---|
| 5 | 43.3 (24.0) | 39.0 (29.5) | 87.5 (5.3) | 88.0 (3.0) |
| 10 | 5.2 (47.9) | 22.6 (37.6) | 86.5 (9.2) | 87.6 (4.3) |
| 40 | 0.0 (68.3) | 9.1 (49.9) | 77.2 (24.8) | 84.4 (8.4) |

- Unedited W efficiency is 88.2 %.
- McNemar on calls changed, ρ = 5: pufix vs c4 p < 1e-300; pufix vs c5 p = 8e-34 (c5 changes more calls).
- Mass-driven prediction: met. Count-driven prediction: excluded.
- Per-jet concordance at ρ = 5:
  - P(JEDI W | W(m) window passes) 0.64 against 0.31 when it fails (difference CI [0.29, 0.37]), κ = 0.32.
  - Of the true W jets not called W after the edit, 663 go to g/q and 475 to Z/t.
- The efficiency match with W(m) (43 vs 37 %) therefore comes with moderate per-jet agreement, not one-to-one agreement.

**JetClass: W efficiency, % at ρ = 5 / 40.**

| model | pufix | c4 (count) | c5 (mass) | c2k |
|---|---|---|---|---|
| ParT_full | 72.0 / 26.6 | 64.2 / 4.2 | 76.8 / 70.2 | 79.2 / 61.4 |
| PN_full | 70.2 / 8.8 | 62.8 / 11.6 | 75.0 / 56.8 | 76.2 / 59.2 |
| ParT_kin | 60.6 / 9.0 | 65.0 / 32.6 | 51.4 / 69.8 (ρ 10: 13.0 vs pufix 49.2) | 67.2 / 44.4 |
| PN_kin | 54.8 / 2.0 | 60.6 / 0.0 | 41.8 / 58.2 | 64.2 / 43.2 |

- Unedited W efficiency: ParT_full 80.0, PN_full 78.0, ParT_kin 68.2, PN_kin 66.2.
- ParT_full calls changed at ρ 5: pufix 6.3 %, c4 8.0 %, c5 3.1 %. At ρ 40: 41.3, 35.6 and 12.8 %.
- **Full-feature models.** c4, with Δm = 1–8 GeV, gives as much or more W loss than pufix (Δm = 10–69 GeV). c5 gives little. Count-driven prediction: met for ParT_full and PN_full. Mass-driven prediction: excluded.
- **kin models.** Both contents contribute. At ρ ≤ 20, c5 changes as many calls as pufix or more (ParT_kin ρ 10: 23.5 vs 18.4 %).
- Per-jet concordance with W(m) at ρ 5: κ = 0.15 (ParT_full), 0.17 (PN_full), 0.26 (ParT_kin), 0.31 (PN_kin).

**Top Landscape: retention at the ε_S = 0.5 threshold, ρ = 5 / 40.**

| model | pufix | c4 (count) | c5 (mass) | c2k |
|---|---|---|---|---|
| ParT | 0.71 / 0.01 | 0.71 / 0.03 | 0.92 / 0.77 | 0.94 / 0.84 |
| PN | 0.69 / 0.00 | 0.83 / 0.31 | 0.93 / 0.93 | 0.95 / 0.94 |
| LorentzNet | 0.59 / 0.00 | 0.50 / 0.17 (ρ 10: 0.11 vs 0.26) | 0.94 / 0.78 | 0.92 / 0.81 |
| PFN | 0.55 / 0.00 | 0.70 / 0.00 | 1.02 / 1.14 | 0.95 / 0.95 |
| EFN (IRC safe) | 0.79 / 0.06 | 0.98 / 0.87 | 0.84 / 0.71 | 0.98 / 0.82 |

- **ParT, LorentzNet, PFN:** count-driven (c4 ≈ pufix; ParT pufix vs c4 McNemar Holm p = 1 at ρ 5).
- **PN:** mostly count-driven at ρ ≤ 10.
- **EFN:** neither control alone reproduces the edit. As an IRC-safe model it treats the centred particles like one particle (c4 ≈ c2k at ρ ≤ 20, Holm p = 1). The single mass-matched particle gives part of the response. The rest needs soft radiation spread over the jet area.
- **Cut-based taggers on the same edits:**
  - top(m, τ32) loses 0.08 / 0.93 of its tags under the edit (ρ 5 / 40), 0.03 / 0.18 under c4 and 0.08 / 0.86 under c5.
  - top(m_SD, τ32) behaves the same way: count-insensitive and mass-sensitive.

**Top Landscape at a matched working point (H13 sample, topwp).** Each network's threshold is set to the cut-based tagger's own starting efficiency.

| network | ε_S | network retention ρ 5 / 10 / 20 / 40 | tagger retention ρ 5 / 10 / 20 / 40 | paired diff at ρ 40 [CI] |
|---|---|---|---|---|
| ParT | 0.70 (top(m, τ32)) | 0.83 / 0.68 / 0.40 / 0.15 | 0.92 / 0.87 / 0.62 / 0.07 | +0.09 [0.06, 0.12] |
| EFN | 0.70 (top(m, τ32)) | 0.89 / 0.78 / 0.56 / 0.33 | 0.92 / 0.87 / 0.62 / 0.07 | +0.30 [0.25, 0.34] |
| PN | 0.70 (top(m, τ32)) | 0.82 / 0.62 / 0.28 / 0.06 | 0.92 / 0.87 / 0.62 / 0.07 | −0.00 [−0.03, 0.02] |
| LorentzNet | 0.70 (top(m, τ32)) | 0.81 / 0.59 / 0.21 / 0.00 | as above | −0.06 [−0.08, −0.04] |
| PFN | 0.70 (top(m, τ32)) | 0.76 / 0.47 / 0.08 / 0.00 | as above | −0.06 [−0.08, −0.04] |
| ParT | 0.66 (top(m_SD, τ32)) | 0.82 / 0.62 / 0.34 / 0.10 | 0.95 / 0.91 / 0.72 / 0.28 | −0.18 [−0.21, −0.14] |

- At ρ 5–20, every network keeps fewer tags than either tagger (bootstrap p < 1e-3), except EFN vs top(m, τ32) at ρ 5 (p = 0.08) and ρ 20 (p = 0.50).
- Against top(m_SD, τ32), all networks keep fewer tags at every ρ, except EFN at ρ 40 (diff 0.00 [−0.05, 0.05]).

**Statistics.** 240 tests (McNemar and paired bootstrap) across hls, jc, top and topwp. 182 have Holm p < 0.05. Those that do not:
- most pu-vs-pufix tests (the axis part);
- ParT_kin and PN_kin pufix vs c5 at ρ 5 and 20;
- Top ParT pufix vs c4 at all ρ;
- EFN c4 vs c2k;
- 4 of the topwp comparisons above.

## Hypotheses
- **H_axis (the response comes from the induced axis offset).** Excluded for every model: c3 is 0–2.5 %, and pu ≈ pufix.
- **H_mass (the response follows the added ungroomed mass).**
  - Met for JEDI-linear: c5 ≈ pufix and c4 small.
  - Partly met for ParT_kin and PN_kin.
  - Excluded for ParT_full, PN_full, and Top ParT, LorentzNet and PFN, where c5 is small and c4 ≈ pufix.
- **H_count (the response follows the number of added soft particles at fixed ΣpT).**
  - Met for ParT_full, PN_full, and Top ParT, LorentzNet, PFN, and mostly PN.
  - Excluded for JEDI-linear.
- **H_ΣpT only.** Excluded for all models: c2k is small.
- **H_groomed (the networks act like m_SD).** Excluded for all trained models at ρ ≥ 10, as in H13.
- **H_faster (Top networks lose tags faster than both cut-based taggers).**
  - Met at a matched working point for ρ 5–20, except EFN vs top(m, τ32).
  - At ρ 40 excluded for ParT and EFN against top(m, τ32), which keep more tags. PN equals it there.

## Verdict on the H13 reading
- **Reproduced:** the efficiency losses, the axis-offset sizes, the c3 and pufix results, and "faster than the soft-drop windows at ρ ≥ 10".
- **Not supported:**
  - "the response comes from the added soft momentum and mass" (H13 conclusion and reading). This holds for JEDI-linear and partly for the kin models only.
  - "Top networks lost tags faster than both cut-based top taggers". This holds at ρ 5–20 at a matched working point but not at ρ 40 against top(m, τ32).
- **Qualified:** "JEDI-linear's W call tracked the ungroomed-mass window". It holds as an efficiency match and as a causal mass response (c5). Per-jet agreement is moderate (κ 0.32), and 58 % of the lost W calls go to g/q rather than to heavier classes.

**Corrected reading.** Under diffuse soft particles (ρ 5–40 GeV per unit area), every trained network's signal efficiency falls more than the soft-drop windows at ρ ≥ 10.
- The induced axis offset (2.6–13 mrad median) and a translation of that size change 0–2.5 % of calls, so the axis is not the cause.
- What the networks respond to differs by model:
  - JEDI-linear (hls4ml) loses W calls in proportion to the added ungroomed mass. One particle that adds the same mass and pT reproduces the loss (W efficiency 39 vs 43 % at ρ 5); the same soft particles placed at the jet centre do not (88 %).
  - The JetClass full-feature ParT and ParticleNet, and the Top-Landscape ParT, LorentzNet and PFN, respond to the number of added soft particles. The centred-particle control (Δm ≈ 1–9 GeV) gives as much or more loss than the full edit, and the mass-matched single particle gives little.
  - The JetClass kin models respond to both.
  - The IRC-safe EFN responds to neither alone.
- At a matched working point the Top networks lose tags faster than top(m, τ32) and top(m_SD, τ32) at ρ 5–20. At ρ 40, ParT and EFN keep more tags than top(m, τ32).

## Figure check
`results/physics/h13_panel.png` matches the H13 numbers.
- The third panel compares networks at ε_S = 0.5 with taggers starting at 66–70 %. The ordering at ρ 5–20 survives working-point matching. At ρ 40 the ParT and EFN curves would lie above top(m, τ32) at a matched working point.
- The panel title ("lower every network's signal efficiency faster than they lower the soft-drop mass tagger's") holds at a matched working point, except EFN at ρ 40.

## Status
`d-h13-verify` (the hls4ml run with W(m_SD) on rows [::5]) was still computing classical observables when this report was written. The observable step runs without parallel speed-up in that env, about 4–25 min per condition. Its JEDI-linear numbers are identical to `d-h13-verify-hls2` (same seed and code path; logged values agree, for example pu_5 24.2 % and c5_10 37.6 %). When it finishes it writes `h13_verify_hls.json` (W(m_SD) for id, pu, c4 and c5) and reruns topwp (deterministic, identical).

## Explainer entry (for site/explain.py, H13V_EXPLAINER)
- **Question.** When pileup-like soft particles lower a tagger's signal efficiency, is the network reacting to the mass they add or to how many there are?
- **Physics background.**
  - A soft particle of transverse momentum $q$ at angle $r$ from the jet axis raises the jet mass by $\Delta m^2\approx p_{T,J}\,q\,r^2$.
  - The same particle near the axis ($r\to0$) adds almost no mass, but still adds one constituent and $q$ of $\sum p_T$.
  - Ungroomed mass taggers respond to the first. Constituent-count observables respond to the second. IRC-safe observables ignore near-collinear splittings.
- **Method.** Add $\mathrm{Poisson}(\rho\pi R^2/\langle q\rangle)$ soft particles (as in H13). Compare against three edits, all with inputs about the stored axis:
  - (c4) the same particles with $r\to r/20$: same count and $\sum p_T$, $\Delta m\approx0$;
  - (c5) one particle with the same kept $\sum p_T$, at the radius that matches the kept invariant mass;
  - (c2k) that particle on the axis.
  - Metrics: signal efficiency and calls changed, McNemar between edits, Holm over 240 tests.
- **Result.**
  - JEDI-linear: c5 reproduces the W loss (39 % vs 43 % at $\rho=5$), c4 does not (88 %).
  - JetClass ParT_full: c4 gives 64 % vs 72 % for the edit, c5 77 % (unedited 80 %).
  - Top ParT: c4 retention 0.71 equals the edit's 0.71; c5 gives 0.92.
- **Checks.**
  - The fresh samples reproduce the H13 efficiencies and accuracies.
  - The axis offset (c3) changes ≤ 2.5 % of calls.
  - Nulls are unchanged.
  - The c5 mass match is exact to $10^{-12}$ GeV.
- **Physical significance.** Soft contamination can lower a network tagger's efficiency through two routes: the mass it adds, or the count of constituents. JEDI-linear follows the first. The particle-ID JetClass networks and the Top-Landscape ParT, LorentzNet and PFN follow the second, which no mass-based tagger shares.
