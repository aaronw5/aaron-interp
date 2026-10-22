# H80 verify (loop D): is the complete N-subjettiness basis enough?

Verifier: skeptical pass on `interp/h80_nsub_basis_nullspace.py`, `interp/h80_diag.py`, `results/physics/h80_stats.json`, `h80_{top,jc,qg}.json`, the figure `h80_nsub_basis.png` and `inbox_d/H80_report.md`.
Rerun: GPU Job `d-h80-verify` (exit 0, about 27 min after a 200-jet smoke step on Top). Script: `interp/h80_verify.py` (new). Outputs: `results/physics/h80_verify.json` (statistics), `h80_verify_smoke.json`, and work files in `results/physics/h80_verify_work/` (per-dataset npz/json; the cluster copy is in `/j-jepa-vol/aaron-interp/results/physics/`).

## Conclusion

On fresh jets, holding all 15 N-subjettiness values (plus jet mass, pT, axis and particle count) while every particle moves by about 1 % still leaves every trained tagger with most of its response; a classifier built on those 15 values gives exactly zero. So the main claim holds: none of these networks is a function of the five-body N-subjettiness basis. The quark–gluon number in the report ("about half the response is removed") is partly an artefact of the comparison edit. The edit that holds the basis moves the hardest particles less than the free edit does. When the free edit is given the same per-particle move sizes, the quark–gluon ratio rises from about 0.5 to about 0.7. Still open: the matched comparison edit is itself easier to tell apart from real jets than the original one, and JEDI-linear (hls4ml) was not rerun.

## Verdict on the reading

- Reproduced: κ₆ on the fresh samples falls within about 0.1 of the reported values on Top, JetClass and QG. Trained networks give 0.47–0.98, against 0.000 for GBDT/MLP on B₆ and 1.0–1.2 for weight-permuted networks.
- Not supported as stated: (i) that holding the basis "removes about half of the quark–gluon taggers' response", and (ii) the cross-dataset order "QG > JetClass ≈ Top". Both depend on a FREE control that is not matched in *which particles* move (details below). With the profile-matched control, QG κ_m is 0.62–0.78 at ε = 0.01. That overlaps the Top PFN/EFN values (0.65–0.82). JetClass κ_m is 0.87–0.99 and mostly consistent with 1 after Holm.
- Not rerun: hls4ml/JEDI-linear (κ₆ = 1.05). Scoring it needs the jax CPU env, so a separate job would be required.

## What was rerun (procedure, sample, edit, control)

**Samples (fresh rows, disjoint seeds).**
- Top Landscape test: 2,000 top + 2,000 QCD, random rows (rng 9101); 3,978 jets with N ≥ 13 particles. References: 5k + 5k disjoint rows (rng 9111).
- KMT Pythia quark–gluon test: 2,000 q + 2,000 g (rng 9103); 3,978 jets kept.
- JetClass: the first test file of TTBar, WToQQ, ZToQQ and ZJetsToNuNu, 1,000 random rows each (rng 9102; never first-k); 3,959 jets kept.

H80 used rng 8001/8002/8003.

**Unedited accuracy / signal-vs-background AUC on these samples.**
- Top: ParT_kin 0.938 / 0.983, PN_kin 0.939 / 0.983, PN_kin_s1 0.936, LorentzNet 0.938 / 0.985, PFN 0.929, PFN_s1 0.930, EFN 0.922 / 0.975; GBDT(B₆) 0.919 / 0.972.
- QG: part_kin 0.827 / 0.901, part_kin_s1 0.832, pnet_kin 0.827, lorentznet 0.845 / 0.916, pfn 0.822, efn 0.817 / 0.883; GBDT(B₆) 0.793 / 0.864.
- JetClass (10-class argmax on the 4 classes / signal-vs-QCD AUC): ParT_full 0.816 / 0.985, ParT_kin 0.725 / 0.977, seeds s0–s2 0.702–0.707, PN_full 0.796, PN_kin 0.702; GBDT(B₆) 0.701 / 0.946.

**Edits.** Exactly the H80 machinery (imported, not copied): the Solver, exclusive-kT re-clustering, acceptance at 0.5 %, and tie groups for coincident particles. Steps ε = 0.01 and 0.00125, where ε is the RMS per-particle ΔR move; the pT move is 3 % × ε/0.01. Arms:
- **NS6.** Hold B₆ = {τ_N^β : N = 1..5, β = ½, 1, 2} (15 values), log m, log pT, the jet rapidity y_J and azimuth φ_J (the E-scheme axis), and N. Here τ_N^β = Σ_i pT_i min_k ΔR_ik^β / Σ_i pT_i about N exclusive-kT axes.
- **FREE.** Hold only m, pT, the axis and N. It starts from the same random vector and has the same norm before the Gauss–Newton correction.
- **FREEM (new, profile-matched control).** For each jet, take the realised NS6 displacement du (per particle and per coordinate: log pT, η, φ, in normalised units) and keep its magnitudes |du|. Give every entry an independent random sign; tied particles keep a common angular move. Project onto the null space of the FREE constraints, rescale to the same total norm as du, and solve back onto m, pT and the axis.

  FREEM therefore moves the same particles by the same amounts in the same coordinates as NS6, but the basis is free to change. It isolates "the basis is held" from "a different set of particles is moved".
- **FLOOR.** The FREE edit at ε = 1e-5, used for the numerical floor.

**Metric.** Δs = change of the true-class log-odds. Two ratios are reported:
- κ = mean|Δs(NS6)| / mean|Δs(FREE)|, as in H80.
- κ_m = mean|Δs(NS6)| / mean|Δs(FREEM)|.

Both are computed on jets accepted in all three arms. Bootstrap 95 % CI over jets (B = 1000; ratio of jointly resampled means). Tests:
- H0: κ_m = 1, with a sign-flip permutation p on the per-jet |Δs_NS6| − |Δs_FREEM| (10⁴ flips). Holm over 40 tests: 20 trained models × 2 steps. The minimum attainable Holm p is 0.004.
- Calls at the frozen 50 % working point (as in H80). Changed calls, NS6 vs FREEM, compared with an exact McNemar test.

## Why the FREE control is not adequate as the only comparison

The NS6 null space excludes directions that change pT-weighted angular sums. Those are dominated by hard particles, so after projection and rescaling the NS6 move puts less of its size on the hardest particles. Two shares are used to show this:
- the pT-weighted share = N·Σ z_i |du_i|² / Σ |du_i|², with z_i = pT_i/ΣpT; it equals 1 when the move does not depend on pT;
- the share of the squared move carried by the 4 hardest particles.

Medians on the jets used, ε = 0.01:

| dataset | pT-weighted share NS6 / FREE / FREEM | top-4 share NS6 / FREE / FREEM |
|---|---|---|
| Top | 0.61 / 0.80 / 0.58 | 0.032 / 0.059 / 0.028 |
| QG | 0.64 / 0.81 / 0.61 | 0.042 / 0.069 / 0.035 |
| JetClass | 0.65 / 0.82 / 0.62 | 0.044 / 0.075 / 0.039 |

So FREE moves the hard particles about 1.7–1.8× more, in squared size, than NS6 does. A network that is more sensitive to hard-particle moves gives κ < 1 under the FREE comparison even with no use of the basis. The H80 report also showed that the two arms differ: |Δlog pT^D| was 1.4–1.9× larger under NS6 (pT^D = √(Σ pT_i²)/Σ pT_i).

## Results (fresh samples; κ, κ_m with 95 % CI; FREEM/FREE response ratio)

ε = 0.01 (n = jets accepted in all arms):

| dataset (n) | model | κ (NS6/FREE) | κ_m (NS6/FREEM) | FREEM/FREE | p_Holm(κ_m = 1) |
|---|---|---|---|---|---|
| Top (2230) | ParT_kin | 0.94 [0.89, 0.99] | 0.96 [0.91, 1.01] | 0.98 | 1.0 |
| | PN_kin / PN_kin_s1 | 0.98 / 0.94 | 0.92 [0.87, 0.97] / 0.95 [0.90, 1.00] | 1.07 / 0.98 | 0.048 / 1.0 |
| | LorentzNet | 0.88 [0.84, 0.93] | 0.98 [0.93, 1.03] | 0.90 | 1.0 |
| | PFN / PFN_s1 | 0.75 / 0.79 | 0.75 [0.69, 0.81] / 0.82 [0.77, 0.87] | 1.00 / 0.96 | 0.004 / 0.004 |
| | EFN | 0.65 [0.60, 0.70] | 0.65 [0.60, 0.69] | 1.00 | 0.004 |
| QG (607) | part_kin | 0.49 [0.43, 0.54] | 0.72 [0.65, 0.79] | 0.67 | 0.004 |
| | part_kin_s1 | 0.52 | 0.78 [0.70, 0.86] | 0.66 | 0.004 |
| | pnet_kin | 0.47 | 0.68 [0.61, 0.75] | 0.70 | 0.004 |
| | lorentznet | 0.49 | 0.73 [0.66, 0.81] | 0.67 | 0.004 |
| | pfn | 0.57 | 0.75 [0.67, 0.84] | 0.76 | 0.004 |
| | efn | 0.36 [0.32, 0.41] | 0.62 [0.56, 0.68] | 0.59 | 0.004 |
| JetClass (1727) | ParT_full | 0.80 [0.73, 0.87] | 0.89 [0.82, 0.97] | 0.89 | 0.091 |
| | ParT_kin | 0.78 [0.72, 0.85] | 0.90 [0.83, 0.97] | 0.87 | 0.090 |
| | ParT_kin s0 / s1 / s2 | 0.78 / 0.77 / 0.75 | 0.88 / 0.92 / 0.90 | 0.84–0.88 | 0.011 / 0.31 / 0.12 |
| | PN_full / PN_kin | 0.92 / 0.76 | 0.99 [0.92, 1.06] / 0.88 [0.83, 0.94] | 0.93 / 0.86 | 1.0 / 0.004 |

Controls, ε = 0.01:
- GBDT(B₆) and MLP(B₆): κ = κ_m = 0.000–0.004 on all three datasets. The solver holds the basis, and a function of B₆ alone does not respond.
- GBDT(B₇), which adds τ₆: κ 0.19–0.28, κ_m 0.25–0.30.
- Weight-permuted networks: κ 0.99–1.18, κ_m 0.96–1.13.
- LR(pT^D): κ 1.47–1.68, κ_m 1.20–1.26.

ε = 0.00125:
- QG: κ_m 0.73–0.94 (part_kin 0.94 [0.89, 0.99], p_Holm 0.19; efn 0.73; pfn 0.78), against κ 0.52–0.77.
- Top: κ_m 0.76–0.99 (EFN 0.76, PFN 0.77).
- JetClass: κ_m 0.87–1.01, with every p_Holm = 1 except ParT_full (0.091).

Changed calls, NS6 vs FREEM (McNemar): only pnet_kin (QG, 0.01; 3 vs 16 jets, p = 0.004) and ParT_kin_s2 (JetClass, 0.01; 49 vs 74, p = 0.03) differ before correction. Under the FREE comparison, H80 found QG part_kin 1.0 vs 4.1 %; here NS6 vs FREEM is 2.6 vs 3.1 % (8 vs 11 jets, p = 0.65).

## Caveat on the FREEM control (why it brackets rather than replaces FREE)

FREEM matches the per-particle move sizes but not their correlations. Randomising the signs of the NS6 move changes the jet observables more than FREE does, at the same norm:
- mean |Δlog τ₆| at ε = 0.01: Top 0.117 vs 0.070; QG 0.214 vs 0.107; JetClass 0.187 vs 0.102.
- |Δlog pT^D|: 5–15× larger than under NS6.
- C2ST at ε = 0.01: Top 0.57, JetClass 0.66, QG 0.75. C2ST is the AUC of a classifier separating edited from real jets; 0.5 means indistinguishable.

On QG, FREEM is above the H80 on-manifold threshold of 0.65. At ε = 0.00125 FREEM's C2ST is 0.48 / 0.53 / 0.58 (Top / JetClass / QG), and there QG κ_m is 0.73–0.94.

FREEM changes the basis and pT^D more than FREE, yet the QG networks respond to it 25–40 % less. That points to a dependence on *which particles* (hard vs soft) are moved, not on the basis values. The true basis-held fraction for QG lies between κ (FREE, unmatched in profile) and κ_m (FREEM, unmatched in correlation): about 0.5–0.75 at ε = 0.01 and 0.5–0.95 at ε = 0.00125.

H0 (QG networks are functions of B₆ + m + pT, κ_m = 0) is excluded: κ_m ≥ 0.62 with CI lower bounds ≥ 0.56. H0 (κ_m = 1, holding the basis changes nothing) is also excluded for all six QG nets at ε = 0.01, so some response reduction from holding the basis remains.

## Code review

No defect found that changes the H80 numbers. Checked:
- **Axis and coordinates.**
  - The basis uses rapidity y and the E-scheme kT axes, computed in numba from (px, py, pz, E). The solver uses the same definitions in torch.
  - ParT/JetClass inputs are rebuilt from the edited table relative to the stored jet axis, with η = jet_eta + sign(jet_eta)·Δη: `prep` passes sg·de, and `build` multiplies by sign again.
  - The held E-scheme axis of the constituents and the stored axis are separate things. Holding the constituent axis keeps the stored-axis offset fixed.
- **Padding.**
  - `table_P4`, `Solver.pack` (mask mk, mk3) and `c2st_feats` all mask padded slots.
  - `keepN` and the smoke subsetting keep `ext` (pid and track features) aligned.
  - `sortT` reorders `ext` with pT, so the network sees consistent per-particle features.
- **Four-momentum.** m, pT_J, y_J and φ_J are held, so the jet four-vector is held; acceptance rechecks this in numpy after re-clustering (0.5 %, axis 0.002). Particle masses are fixed. In QG they are set to 0, so y = η.
- **Same jets.** Model scores, reference-classifier scores, observables and C2ST features are all computed from the same edited table U for each condition.
- **Sampling.** All samples use random rows. JetClass uses `rng.choice` within each class file; there is no first-k.
- **Working point.** It is frozen at the median of the unedited signal (binary) or own-class (multi-class) score.
- **Statistics.** The bootstrap ratio CI, sign-flip and Holm (126 tests, minimum p_Holm 0.0126) are implemented as described.

Minor items in the reading and report:
1. "A tree on B₆ is within 0.002–0.04 AUC of the networks": on QG the gap is 0.016–0.050 (lorentznet 0.916 vs 0.866), both in H80 and here.
2. "κ does not rise with C2ST": the Top correlation is +0.51. All Top C2ST values are ≤ 0.54, so the correlation carries no information, but the wording is not literally true.
3. The numerical floor is a large share for Top LorentzNet (0.22 of |Δs| 0.40–0.50) and for JetClass ParT_kin (0.08–0.10 of 0.33–0.44). The floor-corrected values are reported; the source of that floor (nondeterminism or non-smoothness) is not identified.
4. NS6 acceptance is 15 % on QG at ε = 0.01 (fresh: 15.3 %). κ there is computed on 607 of 3,978 jets.

## Figure (results/physics/h80_nsub_basis.png)

The figure matches h80_stats.json:
- Left panel: a real Top jet (m = 180.8 GeV, pT = 567 GeV, N = 49). τ32 and τ21 are unchanged to 4 decimals after the NS6 edit.
- Right panels: floor-corrected κ_M versus M, with the GBDT(B_M) AUC on the right axis.

Issues:
- The LR(pT^D) curves are mostly above the plotted range (clipped at 1.65).
- The κ axis label says "floor-subtracted" for every curve, including the GBDT and LR references, whose floor is 0.
- The figure has no profile-matched control. A revised figure should add κ_m.

## Corrected reading

With B₆ (15 values), m, pT, the E-scheme axis and N held, every trained tagger on Top Landscape, JetClass and quark–gluon jets keeps a response to a random particle-level move of 1–3 %. A function of B₆ alone gives 0.000–0.004.
- Compared with a free move of the same size (FREE): κ₆ = 0.65–0.98 on Top, 0.75–0.92 on JetClass and 0.36–0.57 on QG (fresh samples).
- Compared with a free move that has the same per-particle displacement sizes (FREEM): κ_m = 0.65–0.98 on Top, 0.87–0.99 on JetClass and 0.62–0.78 on QG at ε = 0.01; QG is 0.73–0.94 at ε = 0.00125.

About half of the QG reduction reported in H80 is due to the basis-holding edit moving the hardest particles less, not to holding the basis. The cross-dataset order is model-dependent rather than "QG > JetClass ≈ Top": Top EFN/PFN (κ_m 0.65–0.82) overlap QG. The EFN keeps κ_m 0.62–0.76, so the leftover response includes IRC-safe structure outside B₆. JEDI-linear (κ₆ = 1.05) was not reverified.

## Suggested follow-up (not run)

- Report κ against both FREE and a profile- and correlation-matched control. One option: sample FREE moves from the NS6 per-particle covariance, i.e. the same Gaussian v with the per-particle scale profile of NS6 applied before the FREE projection, rather than random signs on the realised move.
- Rerun JEDI (hls4ml) with FREEM in the jax env: `./job.sh` after a hls run of `interp/h80_verify.py` with `--ds hls`. This needs the hls table save, which `H.run_ds` already does.
