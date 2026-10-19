# H3 — D3: full-statistics rerun of the D1 / D2 physics edits (corrected η build, random rows, matched controls)

Script `interp/h3_d3_fullstats.py` (figure `interp/h3_plot.py`). Results `results/physics/h3_jedi.json`, `h3_part.json`,
`h3_jedi_wshift.json`, `h3_part_wshift.json`, figure `results/physics/h3_effects.png`. Logs `logs/d-h3-run.log` (JEDI),
`logs/d-h3-part.log` (ParT), `logs/d-h3-wjedi.log`, `logs/d-h3-wpart.log` (matched W-shift reruns), `logs/d-h3-plot.log`.
Cluster jobs: anrunw-ai-d-h3-run (1 CPU / 2 Gi), anrunw-ai-d-h3-part (GPU, GTX 1080 Ti), anrunw-ai-d-h3-wjedi, anrunw-ai-d-h3-wpart,
anrunw-ai-d-h3-plot; smoke jobs d-h3-smoke, d-h3-psmoke (100 jets per class).

## Conclusion

On 10 times more jets than before, with the η bug fixed and random rows, every earlier physics-edit result was reproduced with the
same sign and, in nearly all cases, within the old rate's interval. One example: shrinking the momentum of W jets by 10 % changes JEDI-linear's
W call from 87.5 % to 27.0 % (before: 87 → 28 %). For ParT, removing the displaced tracks from H→bb jets drops the H→bb call to
3.6 %, while shrinking the track displacements without removing them leaves 87 %. Each edit now has a matched control.
The controls narrow three earlier statements. (1) The W-pair mass inside a top jet. JEDI-linear does not respond to it beyond
the control. ParT_full responds a little: 2.7 percentage points more than the control when the pair is raised to 100 GeV, and less
than the control when it is lowered to 60 GeV. (2) In H→ℓνqq′ jets, flipping only the lepton's charge moves ParT more than flipping a hadron's charge.
(3) In H→cc jets, adding photons inside the charm quark's "dead cone" (the small cone around a heavy quark where its radiation is
suppressed) moves ParT more than adding them just outside. All three effects are a few percentage points. It is not known
whether they reflect physics the network learned or side effects of how the edits are built.

## Procedure

Samples (unedited accuracy reproduced on each):
- JEDI-linear (official 64-particle 3-feature checkpoint): `jedi_core.sample(per_class=4000, seed=501)`, 20,000 hls4ml test jets.
  Accuracy 81.07 % (own-class g 78.7, q 72.7, W 87.5, Z 83.5, t 83.0 %); earlier samples 80.6–81 %.
- ParT_full / ParT_kin: 4,000 random rows (numpy `default_rng(502)`, drawn class by class without replacement, row indices up to
  99,998 of 100,000) of each class's first JetClass test file, 40,000 jets. Raw-file accuracy ParT_full 86.38 %, ParT_kin 74.55 %
  (known 85.9–86.2 % and 74.4–74.5 %).
- Build check (corrected η = jet_eta + sign(jet_eta)·Δη): rebuilding the unedited jets changes 0.025 % of ParT_full's calls and
  1.24 % of ParT_kin's (max |ΔE| 1.2·10⁻⁴ GeV; the residual is from jet_pt / jet_energy being recomputed from the ≤ 128 stored particles,
  which enter log pT_rel and log E_rel). With the old mirrored build: 1.14 % / 3.87 % changed (FINDINGS: 0.90 % / 4.47 % on 4,000 jets).
  Unedited calls are taken on the rebuilt jets, so edited and unedited jets share one convention (ParT_kin rebuilt accuracy 74.09 %).

Edits and controls. The tested quantity changes under the edit and is held fixed under the control. Exact definitions are in the script docstring.

| edit | control | applied to |
|---|---|---|
| boost k (pT × k, Δη, Δφ ÷ k), k = 0.8, 0.9, 1.1, 1.25 | rotation about the jet axis by θ = 2 arcsin(\|1/k − 1\|/2): same per-particle displacement, ΣpT kept | all classes |
| decay collapse: softer branch of the largest-kT primary Lund emission moved onto the harder branch (64 hardest particles) | same for the other primary emission with closest softer-branch pT (median pT 94 vs 161 GeV ParT, 149 vs 249 GeV JEDI) | all classes |
| W-pair mass in top → 60 / 80.4 / 100 / 120 GeV, jet mass kept (3 k-means subjets, pair within 15 GeV of 80.4) | non-W pair (b + harder W daughter) shifted by the same Δm, jet mass kept | top |
| remove b, recentre, boost back to original ΣpT | remove the W daughter of pT closest to the b (median pT ratio 0.81) | top (key: call W or Z) |
| ParT_full: displaced (\|d0\|/σ > 2) made prompt; lifetime × 0.35 / × 2.5; keep 2 most displaced | lifetime × 0.35 (count at S > 2 kept) | QCD, H→bb, H→cc, t→bqq′ |
| ParT_full: drop displaced tracks (mean 5.3 per jet) | drop as many prompt charged tracks, pT-matched (mean dropped pT 74 vs 116 GeV) | same |
| ParT_full: lepton de-isolation (4 nearest particles to ΔR 0.03 of the lepton) | same move around the charged hadron of closest pT; same particles moved in random directions | H→ℓνqq′, t→bℓν with a lepton |
| ParT_full: lepton charge and d0 sign flipped | charged hadron of nearest pT rank flipped | same |
| ParT_full: π⁰ merge (photon pairs with m_γγ 85–185 MeV merged, mean 4.4 per jet) | as many 250–450 MeV photon pairs merged | QCD, H→bb, W, t→bqq′ |
| ParT_full: dead-cone fill (6 × 1 GeV photons within ΔR 0.015 of each prong's displaced-track centroid) | same photons at ΔR 0.05–0.08 | H→bb, H→cc |

Metrics: key call rate before → after (own class unless stated), its change with a 95 % bootstrap interval (2,000 resamples of
jets), the control's change, edit − control with its interval. H0 per (edit, class): edit effect = control effect, tested with an exact
McNemar test on the paired key indicator (edit vs control) and a sign-flip permutation test (5,000 permutations, floor p = 2·10⁻⁴)
on Δ log p(true class). Holm correction across each model's edit-vs-control list: JEDI 30 tests, ParT_full 76, ParT_kin 55. The matched
W reruns form their own families of 5 (JEDI) and 4 (each ParT). Background rejection 1/ε_B is taken at ε_S = 50 % and at the
classical tagger's ε_S, with the discriminant p_S/(p_S + p_B). The bootstrap interval is on the ratio after/before. Classical taggers:
W/Z, 65 < m_SD < 105 GeV and τ21 < 0.5. Top, 140 < m_SD < 200 GeV and τ32 < 0.6. H→bb (ParT), 100 < m_SD < 150 GeV and ≥ 2 tracks with
|d0|/σ > 3. m_SD uses soft drop β = 0, z_cut 0.1 on the 40 hardest particles.

W-shift fix (debug attempt 1): in the first run the control pair reached only part of its target, because 5 iterations of
pair-spread then jet-rescale were not enough. At 100 GeV the W pair moved +19.1 GeV and the control pair +13.7 GeV; at 120 GeV,
+35.0 vs +22.8 GeV. The rerun (`wshift` mode) uses 30 iterations and keeps only jets where both pairs land within 2 GeV of target
(n = 3,288 / 3,183 / 2,084 / 723 ParT and 2,921 / 2,816 / 1,604 / 463 JEDI for 60 / 80.4 / 100 / 120 GeV). The jet mass is held to
< 0.01 GeV. The matched numbers are the ones read below; the first-run numbers are in `h3_*.json`.

## Numbers: old vs new (key call rate before → after, %; new interval is on the change)

| model, edit, class | old (N per class, rows) | new (N) | new change, 95 % CI | control change | edit − control, Holm p |
|---|---|---|---|---|---|
| JEDI boost ×0.9, W | 87 → 28 (2,000) | 87.5 → 27.0 (4,000) | −60.5 [−62.1, −59.0] | −0.3 | −60.2, p < 10⁻³⁰⁰ |
| JEDI boost ×0.9, Z | 85 → 48 | 83.5 → 47.3 | −36.3 [−37.8, −34.6] | −0.3 | −36.0, p < 10⁻³⁰⁰ |
| JEDI boost ×0.8, W | 87 → 4 | 87.5 → 3.3 | −84.2 | −0.2 | −84.0 |
| JEDI boost ×1.1, t | 83 → 64 | 83.0 → 65.6 | −17.4 [−18.6, −16.2] | −0.2 | −17.2, p 10⁻¹⁷⁸ |
| JEDI boost ×1.25, t | 83 → 42 | 83.0 → 44.9 | −38.0 [−39.5, −36.5] | −0.2 | −37.8 |
| JEDI decay collapse, W / Z / t | (not run before) | 87.5→15.3 / 83.5→5.1 / 83.0→30.1 | −72.2 / −78.4 / −52.9 | −3.8 / −4.5 / −12.2 | all p < 10⁻³⁰⁰ |
| JEDI W pair → 60, t (matched) | 88 → 86 (2,328) | 87.6 → 85.7 (2,921) | −1.9 [−2.7, −1.1] | −2.4 | +0.4 [−0.4, 1.3], p 1 |
| JEDI W pair → 100, t (matched) | 88 → 86 | 94.6 → 93.5 (1,604) | −1.1 [−2.0, −0.4] | −1.8 | +0.7 [−0.4, 1.9], p 1 |
| JEDI W pair → 120, t (matched) | 88 → 85 | 98.1 → 95.2 (463) | −2.9 [−4.3, −1.5] | −2.6 | −0.2 [−2.2, 1.7], p 1 |
| JEDI remove b, t (call W or Z) | 80 (W 63, Z 17; 1,533) | 6.3 → 79.7 (W 63, Z 16; 3,047) | +73.4 [71.7, 75.0] | +14.0 | +59.3, p < 10⁻³⁰⁰ |
| ParT_full boost ×0.8, W | 78 → 54 (400, rows 3,000–3,400) | 80.6 → 57.8 (4,000) | −22.8 [−24.2, −21.3] | −0.5 | p 10⁻¹⁹⁷ |
| ParT_full boost ×0.9, W | — | 80.6 → 72.5 | −8.1 [−8.9, −7.0] | −0.1 | p 10⁻⁶⁰ |
| ParT_full boost ×1.25, W / Z | 78 → 80 / 72 → 64 | 80.6 → 83.0 / 70.7 → 64.4 | +2.4 / −6.3 | ≈ 0 | p 10⁻⁹ / 10⁻²⁹ |
| ParT_full decay collapse, W / Z / H→bb / t→bqq′ | 78→10 / 72→6 / 92→20 / 96→24 | 80.6→10.5 / 70.7→6.7 / 92.3→16.6 / 95.1→27.6 | −70.1 / −63.9 / −75.7 / −67.5 | −3.6 / −3.6 / −3.6 / −26.8 | all p < 10⁻²⁹⁰ |
| ParT_full decay collapse, QCD | 77 → 88 | 77.6 → 88.0 | +10.3 [9.0, 11.8] | +0.5 | p 10⁻⁴¹ |
| ParT_full W pair → 60, t (matched) | 96 → 92 (1,653) | 95.8 → 93.3 (3,288) | −2.6 [−3.3, −1.8] | −5.3 | +2.8 [1.7, 3.7], p 3·10⁻⁷ |
| ParT_full W pair → 100, t (matched) | 96 → 90 | 96.4 → 90.4 (2,084) | −6.0 [−7.1, −4.9] | −3.3 | −2.7 [−4.0, −1.3], p 2·10⁻⁴ |
| ParT_full W pair → 120, t (matched) | 96 → 78 | 98.5 → 82.6 (723) | −15.9 [−18.7, −13.1] | −13.7 | −2.2 [−5.8, 1.0], p 0.45 |
| ParT_kin W pair → 60 / 100 / 120 (matched) | — | 91.8→84.9 / 93.2→83.9 / 97.0→69.6 | −6.9 / −9.4 / −27.4 | −10.1 / −8.6 / −26.6 | +3.3 (p 5·10⁻⁵) / −0.7 (p 0.89) / −0.8 (p 0.89) |
| ParT_full remove b, t (call W or Z) | W 70, Z 18 (331) | 0.7 → 85.8 (W 68, Z 18; 3,292) | +85.1 | +25.9 | +59.3, p < 10⁻³⁰⁰ |
| ParT_kin remove b | W 67, Z 27 | 0.9 → 93.6 (W 67, Z 26) | +92.6 | +30.0 | +62.6 |
| ParT_full displaced made prompt, H→bb / H→cc / t→bqq′ | 92→10 / 83→56 / 96→68 | 92.3→11.8 / 84.9→58.0 / 95.1→64.3 | −80.5 / −26.9 / −30.9 | (lifetime ×0.35) −5.0 / −5.7 / −2.0 | all p < 10⁻²²⁰ |
| ParT_full lifetime × 0.35, H→bb | 92 → 86 | 92.3 → 87.3 | −5.0 [−5.8, −4.3] | — | — |
| ParT_full lifetime × 2.5, H→cc | 83 → 81 | 84.9 → 81.6 | −3.3 [−4.2, −2.3] | — | — |
| ParT_full keep 2 most displaced, H→bb | 92 → 32 | 92.3 → 32.9 | −59.5 [−61.0, −57.9] | −5.0 (lifetime ×0.35) | −54.4, p < 10⁻³⁰⁰ |
| ParT_full drop displaced tracks, H→bb | (new) | 90.8 → 3.6 (2,698) | −87.2 [−88.5, −86.0] | −30.7 (drop prompt) | −56.5, p < 10⁻³⁰⁰ |
| ParT_full drop displaced tracks, H→cc / t→bqq′ / QCD | (new) | 85.0→24.2 / 95.3→36.2 / 74.2→75.3 | −60.8 / −59.2 / +1.1 | −21.3 / −9.9 / −0.8 | p < 10⁻³⁰⁰ / < 10⁻³⁰⁰ / 0.007 |
| ParT_full lepton de-isolated, t→bℓν | 99 → 71 (496) | 98.7 → 77.2 (3,311) | −21.6 [−23.0, −20.1] | −0.2 (hadron) / −2.3 (random) | −21.4, p 10⁻¹⁹⁹ |
| ParT_full lepton de-isolated, H→ℓνqq′ | 100 → 86 (493) | 98.8 → 84.4 (3,846) | −14.4 [−15.5, −13.2] | −1.4 / −3.0 | −12.9, p 10⁻¹⁰⁷ |
| ParT_full lepton charge flip, t→bℓν | 99 → 98 | 98.9 → 98.2 (3,972) | −0.7 [−1.1, −0.4] | −0.4 | −0.4 [−0.8, 0.0], p 1 |
| ParT_full lepton charge flip, H→ℓνqq′ | 100 → 96 | 98.7 → 93.7 (3,946) | −5.1 [−5.8, −4.4] | −2.5 | −2.6 [−3.3, −1.9], p 2·10⁻¹² |
| ParT_full π⁰ merge, QCD / H→bb / W / t→bqq′ | (attention scan, G202) | 75.2→74.5 / 92.0→91.2 / 80.0→80.9 / 95.1→95.1 | −0.8 / −0.8 / +0.9 / −0.1 | −0.7 / −0.7 / +0.7 / +0.2 | −0.1 … −0.3, all p 1 |
| ParT_full dead-cone fill, H→bb | 92 → 92 (control 92) | 92.9 → 93.9 (3,931) | +1.0 [0.6, 1.5] | +0.7 | +0.3 [0.0, 0.7], p 1 |
| ParT_full dead-cone fill, H→cc | 84 → 81 (control 82) | 86.3 → 81.9 (2,899) | −4.4 [−5.3, −3.4] | −2.3 | −2.1 [−2.8, −1.4], p 1.3·10⁻⁷ |

Track counts (mean tracks with |d0|/σ > 3, H→bb): unedited 7.77; lifetime × 0.35: 5.64; keep 2: 1.99; made prompt: 0. For H→cc:
3.27 → 2.11 / 1.73 / 0.

W-shift by sign of jet η (ParT_full, matched, edit − control): 100 GeV, η < 0 [−5.9, −2.1], η ≥ 0 [−3.3, +0.3]; 60 GeV, η < 0 [+2.3, +5.3],
η ≥ 0 [+0.6, +3.3]. The two halves' intervals overlap at every target.

Background rejection (1/ε_B; QCD = g + q for JEDI, QCD for ParT; bootstrap 95 % CI of after/before):
- Unedited, at the classical tagger's ε_S. W vs QCD: classical 1/0.0806 = 12.4 at ε_S 74.1 % (JEDI sample); JEDI 174 [134, 263].
  The same classical tagger on JetClass: 1/0.0645 = 15.5 at ε_S 63.8 %; ParT_full 200 [133, 364]; ParT_kin 114. Top vs QCD: classical
  1/0.0148 = 68 at ε_S 41.5 % (JEDI sample), JEDI 195; JetClass classical 1/0.0192 = 52 at ε_S 54.1 %, ParT_full ≥ 8,000 (no QCD jet
  passes; the bound is set by 4,000 QCD jets), ParT_kin 667. H→bb vs QCD: classical 1/0.0512 = 19.5 at ε_S 67.7 %, ParT_full ≥ 8,000,
  ParT_kin 235.
- JEDI, W vs QCD at ε_S = 50 %: boost ×0.9: 800 → 258, ratio [0.15, 0.53]; its rotation control: 800 → 800, ratio [0.67, 1.37].
  Top vs QCD, boost ×1.25: 119 → 55, ratio [0.36, 0.56]. The classical tagger's ε_S and ε_B do not change under the boost (m_SD, τ21, τ32
  are boost-invariant at this order).
- ParT_full, W vs QCD at ε_S = 50 %: boost ×0.9: 400 → 444, ratio [0.59, 2.18]; boost ×0.8: 400 → 267, ratio [0.33, 1.18]. At the classical
  ε_S (63.8 %), boost ×0.8: 200 → 100, ratio [0.33, 0.72]. ParT_kin boost ×0.8: ratio [0.71, 1.59].
- ParT_full, H→bb vs QCD at ε_S = 50 %: displaced made prompt ≥ 8,000 → 286 (ratio ≤ 0.07); keep 2 displaced → 667; lifetime × 0.35 →
  ≥ 8,000 (no change); drop displaced tracks 5,944 → 66 (pT-matched prompt-drop control → 2,972).
- Decay collapse (both models): W vs QCD rejection falls to 3–6 at ε_S = 50 % (ratio ≤ 0.02). Under the same edit the classical
  tagger keeps 1.4 % of W jets (m_SD falls below the window).

## Reading

- JEDI-linear: the boost edits change the calls in the same direction and by the same size as J27b at every k. The rotation control
  changes ≤ 0.5 points, so the response follows ΣpT, not particle displacement. Shifting the W pair inside a top at fixed jet mass
  changes the top call by −1.1 to −2.9 points. The matched non-W shift changes it by −1.8 to −2.6 (edit − control +0.7 to −0.2, Holm p = 1).
  The edit lowers log p(top) less than the control (−0.02 to −0.07 vs −0.06 to −0.12, Holm p ≤ 0.004). Removing the b gives a W/Z call in 80 %
  of tops; removing the pT-matched W daughter gives one in 20 %.
- ParT_full: the boost response is small next to JEDI's but not zero. W jets at ×0.9 lose 8.1 points (rotation control −0.1); at ×0.8,
  22.8. Shifting the W-pair mass at fixed top mass: lowering it to 60 GeV costs 2.8 points less than the matched control; raising it to
  100 GeV costs 2.7 points more; at 120 GeV the difference is −2.2 [−5.8, 1.0]. ParT_kin shows no edit − control difference for raising
  the pair (−0.7, −0.8; Holm p 0.89). The η-split shows no difference between jets with η < 0 and η ≥ 0 beyond overlapping intervals.
  So the old mirrored build did not hide a W-mass dependence at negative η.
- Heavy flavour (ParT_full): removing displaced tracks, or making them prompt, takes H→bb to 3.6 % / 11.8 %. Keeping only 2 displaced
  tracks takes it to 32.9 %. Shrinking lifetimes by 0.35 at fixed count (S > 2) leaves 87.3 %, even though the S > 3 count falls 27 %.
  The prompt-track drop control (less pT removed, 74 vs 116 GeV) takes H→bb to 60.2 %. Part of the drop effect is the lost momentum and
  mass: jet-mass median 103 vs 111 GeV.
- Leptons (ParT_full): de-isolation lowers t→bℓν by 21.6 points. The flow is to H→bb (13 %), i.e. a semileptonic b decay. The same move
  around a charged hadron changes it by 0.2 points. The t→bℓν rate after de-isolation is 77.2 %; the old 400-jet value was 71 %, outside
  the new interval but with the same sign. Flipping the lepton's charge leaves t→bℓν within 0.4 points of the control. In H→ℓνqq′ it costs
  2.6 points more than the control, with the flow going to t→bℓν (5 %).
- π⁰ merge: the edit and the control are equal within 0.3 points in all four classes (p = 1). Dead cone: equal for H→bb; for H→cc
  the inside fill costs 2.1 points more than the outside fill.
- Rejection at fixed efficiency: JEDI-linear and ParT_full reject 13–14× more QCD than the classical m_SD + τ21 tagger at the
  tagger's own signal efficiency (JEDI 174 vs 12.4; ParT_full 200 vs 15.5; ParT_kin 114, 7×). The decay-collapse edit removes most of that margin for both networks.

Hypotheses (D1/D2 as stated vs alternatives):
- D1, JEDI ΣpT shortcut: predicted that boost flows reproduce J27b. They do at all four k, and the rotation control rules out
  generic displacement (Holm p < 10⁻¹⁴ for W/Z/t at every k).
- D1, no W-mass constraint inside top: predicted edit ≤ control. JEDI: edit − control ∈ [−2.2, +1.9] points at all targets (p = 1),
  and the edit lowers log p(top) less than the control. The alternative "W-mass dependence hidden at η < 0" does not apply to JEDI
  (hls4ml has no η build). ParT_full: raising the pair to 100 GeV costs 2.7 points more than the control (p 2·10⁻⁴); lowering
  it to 60 GeV costs 2.8 points less. ParT_kin: no difference when the pair is raised.
- D2, multiplicity not lifetime: predicted that dropping displaced tracks changes b calls far more than d0 scaling. Drop: −87.2
  points; keep 2: −59.5; lifetime × 0.35: −5.0 (edit − control −80.6 and −54.4, Holm p < 10⁻³⁰⁰).
- D2, isolation used / π⁰ not used / dead cone not used / lepton–b charge link not used: isolation −21.4 points vs control;
  π⁰ edit = control; dead cone, H→bb edit = control; t→bℓν charge flip edit = control. Each is the same as before. Two new small
  effects: dead cone in H→cc (−2.1) and lepton charge flip in H→ℓνqq′ (−2.6).

## Issues and caveats

- W-shift control under-reached its target in the first run (5 iterations). Fixed in the `wshift` rerun with 30 iterations and a
  2 GeV tolerance. The matched subsets keep fewer jets as the target rises (2,084 at 100 GeV, 723 at 120 GeV for ParT). These are
  jets with higher original W-pair mass (median 82.7 and 85.5 GeV) and higher unedited top rates (96.4 %, 98.5 %).
- The decay-collapse control emission has a softer softer-branch than the edited emission (median pT ratio 0.69 ParT, 0.74 JEDI).
  An exact pT match is not always available among the primary emissions.
- The drop-prompt-tracks control removes less pT than the displaced-track drop (74 vs 116 GeV mean).
- Rejection values for ParT_full H→bb / top vs QCD are lower bounds (≥ 8,000, set by the 4,000 QCD jets). The edit ratios there are
  bounded the same way.
- Lund declustering and soft drop use the 64 / 40 hardest particles (cost). G192 used all particles for Lund.
- ParT_kin's own rebuild changes 1.24 % of its calls, which is the floor for ParT_kin edit effects.
- Compute: the GPU job (anrunw-ai-d-h3-part, ~50 min) spent most of its wall time in numpy (classical observables, Lund), so GPU
  utilisation was low. The JEDI job's peak memory was 2,047 MiB, at the 2 Gi request. H1's JetClass smoke job (d-h1-jc-smk)
  was running on a separate pod/GPU while d-h3-part ran.

## Explainer entry (for site/explain.py)

**Question.** Do the physics statements D1 (JEDI-linear: mass windows plus a ΣpT shortcut, no W-mass constraint inside top) and
D2 (ParT: displaced-track multiplicity rather than lifetime, lepton isolation, no π⁰ / dead cone / lepton–b charge link) still hold
with 10× more jets, random JetClass rows, the corrected η build, and a matched control for every edit?

**Physics background.** A longitudinal boost along the jet axis rescales momenta by $k$ and angles by $1/k$, leaving the jet mass
$m^2 \approx \sum_{i<j} p_{T,i} p_{T,j} \Delta R_{ij}^2$ and N-subjettiness ratios unchanged; only $\sum p_T$ changes. A top quark
decays as $t \to bW$, $W \to q\bar q'$, so classical top taggers require a subjet pair with $m_{jj} \approx m_W = 80.4$ GeV. A b hadron
travels $c\tau \approx 450\,\mu$m and its decay yields about 5 charged tracks; a c hadron 120–310 µm and 2–3 tracks. A prompt lepton
from $W \to \ell\nu$ is isolated, $I = \sum_{\Delta R<0.1} p_T / p_{T,\ell} \approx 0$. A heavy quark radiates little inside
$\theta \lesssim m_Q/E$ (dead cone). $\pi^0 \to \gamma\gamma$ gives photon pairs with $m_{\gamma\gamma} = 135$ MeV.

**Method.** For each edit $E$ and its control $C$ (same size of change, tested quantity held fixed), on jets of class $c$:
$\Delta_E = \Pr(\text{call}=c \mid E) - \Pr(\text{call}=c)$ with a 95 % bootstrap interval;
$H_0: \Delta_E = \Delta_C$, tested by an exact McNemar test on the paired calls, Holm-corrected over the edit list. Background
rejection $R = 1/\varepsilon_B$ at fixed signal efficiency $\varepsilon_S$ (50 % and the classical tagger's), discriminant
$p_S/(p_S+p_B)$. The W-pair shift spreads the pair's subjets about their centroid until $m_{jj} = t$ and rescales all angles so that
$m_{\rm jet}$ is unchanged; the control does the same for the non-W pair, by the same $\Delta m$.

**Result (figure `results/physics/h3_effects.png`).** JEDI-linear, W jets at $k = 0.9$: W call 87.5 → 27.0 % (rotation control
87.2 %). ParT_full, H→bb: dropping displaced tracks 90.8 → 3.6 %; lifetime × 0.35 at fixed count 92.3 → 87.3 %. W-pair mass at fixed top
mass: JEDI edit − control within ±2 points (p = 1); ParT_full −2.7 points at 100 GeV, +2.8 at 60 GeV.

**Checks.** Unedited accuracies: JEDI 81.07 %, ParT_full 86.38 %, ParT_kin 74.55 %. Corrected rebuild changes 0.025 % of ParT_full
calls (mirrored build 1.14 %). The η < 0 and η ≥ 0 halves have overlapping intervals. Old 400-jet rates fall within or near the new
intervals, with the same sign everywhere.

**Physical significance.** The networks' responses to boosts, heavy-flavour track edits and lepton isolation are measured at the
percent level and do not depend on the η sign convention. At the classical working point JEDI-linear and ParT_full reject 13–14× more QCD than
m_SD + τ21. Collapsing the decay splitting (removing the two-prong structure) removes most of that margin. For the
W-pair mass inside a top quark, the effect beyond a matched control is at most a few points.
