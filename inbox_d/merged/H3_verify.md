# H3 verify — D3 full-statistics rerun of the D1 / D2 physics edits

Script `interp/h3_verify.py`. Results `results/physics/h3_verify_part.json`, `results/physics/h3_verify_jedi.json`.
Logs `logs/d-h3-verify.log`, `logs/d-h3-verifyj.log`. Jobs: anrunw-ai-d-h3-verify (GPU, GTX 1080 Ti, 5 min) and anrunw-ai-d-h3-verifyj
(1 CPU / 2 Gi, 8 min, peak 2,047 MiB). The first launch of both jobs failed on an indexing bug in the verify script. The bug was fixed
and both jobs were relaunched under the same names.

## Conclusion

On a new set of jets, the main H3 results come back with the same size. For JEDI-linear, W jets get the W label 87 % of the time
before and 27 % after scaling the jet momentum by 0.9. A rotation of equal size has no effect. For ParT, the H→bb label depends on
how many tracks are displaced from the collision point, not on how far they are displaced. Pulling every strongly displaced track back
to just above the "displaced" threshold costs 3 points. Keeping only the two most displaced tracks costs 61 points. Two parts of H3 do
not hold. First, H3's control for the W-pair-mass edit inside top jets also moves the W-pair mass, by 10–30 GeV in the opposite
direction. With a control that keeps the W-pair mass fixed, neither network shows a W-pair-specific effect beyond the interval. The
2.7-point ParT_full effect that H3 reported was not significant on the fresh sample. Second, "lifetime × 0.35 at fixed S > 2 count" is
wrong: the count of tracks with S > 2 falls by 26 %. Not tested here: whether the networks respond to the set of three pair masses in
general, and the small lepton-charge (H→ℓνqq′) and dead-cone (H→cc) effects.

## Code check of `interp/h3_d3_fullstats.py` / `h3_plot.py`

- Rows: ParT rows are random (`default_rng(502)`, sorted, read in 25k chunks); no first-k subsampling. JEDI uses `jedi_core.sample`
  (random per class). The `searchsorted` mapping in `compare` / `rejection` needs sorted row arrays. Every variant's rows are sorted
  subsets of `arange(N)`, so the mapping is correct.
- Build: `part_core.build` uses η = jet_eta + sign(jet_eta)·Δη. Unedited calls are taken on the rebuilt jets, so the edited and
  unedited jets share one convention. Padded slots: `kin` multiplies by the mask, `sig` masks charge = 0 and padding.
- Boost: pT × k with Δη, Δφ ÷ k. This keeps m ≈ √(Σ pT_i pT_j ΔR_ij²) at small angles and changes ΣpT. The rotation control moves each
  particle by the same distance and keeps ΣpT and the mass. Implemented as described.
- **Lifetime edit mislabelled.** `scale_disp(J, 0.35)` multiplies d0 and dz by 0.35 but leaves σ(d0) unchanged, so S = |d0|/σ falls by
  2.9× and tracks with 2 < S < 5.7 drop below 2. On the fresh H→bb jets the mean count at S > 2 goes from 8.97 to 6.63. The variant
  name "count at S>2 kept" and the reading's "lifetime scaling at a fixed S>2 count" are therefore incorrect. The figure also labels
  lifetime × 0.35 as the "control" for keep-2, but it is not matched to keep-2.
- **W-pair control is not W-neutral.** `wshift` spreads the non-W pair (b + harder W daughter) about its centroid and then rescales
  every angle to restore the jet mass. The W pair shares a subjet with the control pair, and the jet mass is fixed. The control
  therefore moves the W-pair mass the other way (measured below). The original script logs the W-pair mass only for the edit.
- The matched-shift subsets keep only jets where both moves converge within 2 GeV. This selects jets with a higher W-pair mass and a
  higher unedited top rate (H3 noted this). At 120 GeV this is n = 723 (H3) and n = 508 here.
- Rejection (`rejection`): the threshold is set at quantile 1 − ε_S of the signal discriminant. Backgrounds are edited when the
  variant covers them. For variants that cover only part of the QCD rows (drop-tracks, π⁰), the background is that subset. The value
  is internally consistent before and after, but it is not a full-sample rejection. No working-point error was found.
- Figure (`h3_effects.png`, read): the values match the JSONs. The W-pair rows show only the W-free control. The keep-2 row's "control"
  is lifetime × 0.35. The new H→ℓνqq′ charge-flip and H→cc dead-cone effects named in the reading are not in the figure.

## Rerun on fresh samples

Samples:
- JEDI-linear: 2,000 jets per class drawn with seed 601 from the hls4ml test set, excluding H3's 20,000 seed-501 jets (overlap 0).
  N = 10,000; unedited accuracy 81.26 %.
- ParT: 3,000 random rows per class (`default_rng(702)`) of each class's second JetClass test file `*_101.root` (H3 used `*_100`).
  N = 30,000. ParT_full accuracy 85.90 % raw / 85.92 % rebuilt; ParT_kin 74.71 % / 74.23 %.

Statistics: 2,000-resample bootstrap 95 % CIs over jets; exact McNemar on paired key calls; sign-flip permutation on Δ log p(true).
Holm correction is applied across each model's family of tests (JEDI 11 tests, ParT 28 tests).

### Replications (key call before → after; edit − control)

| model, edit, class | H3 (N 4,000/class) | fresh sample | control | edit − control [95 % CI], Holm p |
|---|---|---|---|---|
| JEDI boost × 0.9, W | 87.5 → 27.0 | 87.4 → 26.9 (n 2,000) | rotation 87.7 | −60.8 [−62.9, −58.6], p < 10⁻³⁰⁰ |
| JEDI boost × 0.9, Z | 83.5 → 47.3 | 83.3 → 45.5 | rotation 83.2 | −37.6 [−39.8, −35.4], p 8·10⁻¹⁸⁸ |
| ParT_full boost × 0.9, Wqq | 80.6 → 72.5 | 78.8 → 72.3 (n 3,000) | rotation 78.6 | −6.3 [−7.4, −5.2], p 3·10⁻³¹ |
| ParT_kin boost × 0.9, Wqq | 69.0 → 66.5 | 68.3 → 65.8 | rotation 68.6 | −2.9 [−3.8, −2.0], p 2·10⁻⁹ |
| ParT_full keep 2 most displaced, Hbb | 92.3 → 32.9 | 92.4 → 31.5 | — | Δ −61.0 [−62.7, −59.2] |
| ParT_full displaced made prompt, Hbb | 92.3 → 11.8 | 92.4 → 11.9 | — | Δ −80.5 [−82.0, −79.1] |
| ParT_full lifetime × 0.35, Hbb | 92.3 → 87.3 | 92.4 → 86.9 | — | Δ −5.6 [−6.5, −4.7] |

### New control 1: W-pair mass inside top, with the W pair held fixed

Edit: the W pair is spread to target mass t at fixed jet mass (H3 procedure, 30 iterations). Control A is H3's control: the non-W pair
(b + harder W daughter) is shifted by the same Δm and the W pair is left free. Control B (new) shifts the non-W pair by the same Δm
while holding the W pair at its original mass and the jet mass fixed: 40 alternating iterations, with all three within 2 GeV
(0.5 GeV for the jet). All results are on the same jets, matched for the edit and both controls.

Median pair masses (GeV) on ParT's matched top jets (JEDI shows the same pattern):

| target | n | unedited W / C / third | edit W / C / third | control A W / C / third | control B W / C / third |
|---|---|---|---|---|---|
| 60 | 1,568 | 79.5 / 118.2 / 92.9 | 60.0 / 128.7 / 95.3 | **90.0** / 99.5 / 102.5 | 79.7 / 99.8 / 110.3 |
| 100 | 1,441 | 82.8 / 108.3 / 102.4 | 100.0 / 94.0 / 100.4 | **68.9** / 126.7 / 91.9 | 82.6 / 126.5 / 85.0 |
| 120 | 508 | 84.9 / 99.9 / 117.7 | 120.0 / 68.2 / 107.9 | **56.5** / 135.6 / 99.0 | 84.7 / 135.7 / 86.6 |

Under H3's control the W-pair mass moves by 10–28 GeV in the direction opposite to the edit.

H0: the top call does not depend on the W-pair mass beyond the general rearrangement of the pair masses (edit = control B).
H1: moving the W pair away from 80 GeV lowers the top call more than an equal Δm shift of another pair with the W pair fixed (edit < control B).
Predictions: H1 (a W-mass constraint) predicts edit − control B < 0 at 100 and 120 GeV and also at 60 GeV. A response to the overall
three-pair mass configuration predicts edit ≈ control B, with both negative.

Top-call change (percentage points), edit and control B, and edit − control B:

| model | target | edit Δ | control B Δ | edit − control B [95 % CI], Holm p | H3 edit − control A (H3 sample) | edit − control A, fresh |
|---|---|---|---|---|---|---|
| ParT_full | 60 | −2.1 | −3.7 | +1.6 [+0.3, +2.9], p 0.39 | +2.8 [1.7, 3.7] | +0.9 [−0.3, 2.0], p 1 |
| ParT_full | 100 | −5.3 | −5.6 | +0.3 [−1.2, +1.9], p 1 | −2.7 [−4.0, −1.3] | −1.1 [−2.6, 0.3], p 1 |
| ParT_full | 120 | −15.6 | −12.2 | −3.3 [−7.3, +0.6], p 1 | −2.2 [−5.8, 1.0] | −2.0 [−6.1, 2.0], p 1 |
| ParT_kin | 60 / 100 / 120 | −2.7 / −8.4 / −26.6 | −5.3 / −9.5 / −22.7 | +2.6 [0.8, 4.5] / +1.2 [−0.8, 3.1] / −3.9 [−8.7, 0.8]; p 0.13 / 1 / 1 | | |
| JEDI | 60 / 100 / 120 | −2.0 / −1.2 / −3.8 | −2.2 / −0.4 / −1.7 | +0.2 [−1.4, 1.8] / −0.8 [−2.5, 0.8] / −2.1 [−5.1, 0.4]; p 1 / 1 / 1 | | |

Δ log p(top), edit − control B, ParT_full: +0.037 [−0.002, 0.077] / +0.003 [−0.036, 0.043] / −0.137 [−0.234, −0.033]
(Holm p 0.84 / 1 / 0.11).

Reading: H0 is not rejected at any target after Holm correction. With the W pair held fixed, an equal Δm change of another pair lowers
ParT's top call about as much as the W-pair edit (e.g. −12.2 vs −15.6 points at 120 GeV). The 120-GeV point leans toward H1 (−3.3, CI
reaches +0.6; Δ log p CI excludes 0 before correction). H3's ParT_full asymmetry (+2.8 at 60, −2.7 at 100) has the same sign on the
fresh sample but is smaller (+0.9, −1.1) and its CIs include 0. The reported "small asymmetric W-mass effect of about 2.7 points" is
not replicated. JEDI's edit itself moves the top call by ≤ 4 points at every target, so the JEDI null does not depend on the control.

### New control 2: H→bb displaced-track count vs displacement size (ParT_full, n = 3,000 H→bb jets)

| edit | mean tracks S > 2 | mean tracks S > 3 | mean Σ S (S > 2) | H→bb call | Δ [95 % CI] |
|---|---|---|---|---|---|
| unedited | 8.97 | 7.89 | 1,598 | 92.4 | — |
| S capped at 3.5 (d0, dz × 3.5/S for S > 3.5) | 8.97 | 7.89 | 30 | 89.4 | −3.1 [−3.9, −2.2] |
| lifetime × 0.35 | 6.63 | 5.73 | 557 | 86.9 | −5.6 [−6.5, −4.7] |
| lifetime × 0.1 | 3.66 | 2.78 | 156 | 63.4 | −29.0 [−30.7, −27.3] |
| lifetime × 0.03 | 1.55 | 1.20 | 44 | 31.2 | −61.2 [−63.0, −59.4] |
| keep 2 most displaced (others d0, dz = 0) | 2.00 | 1.99 | 1,338 | 31.5 | −61.0 [−62.7, −59.2] |
| displaced made prompt | 0 | 0 | 0 | 11.9 | −80.5 [−82.0, −79.1] |

Paired tests: keep-2 − capped = −57.9 [−59.7, −56.1] (Holm p < 10⁻³⁰⁰); lifetime × 0.1 − capped = −25.9 [−27.6, −24.2] (Holm p 10⁻¹⁸²).
H0 (the call depends on the summed displacement) predicts that the capped edit (Σ S down 98 %) drops the call more than keep-2
(Σ S down 16 %). The result is the opposite (−3 vs −61), so H0 is excluded. H1 (the call depends on the number of tracks above a
significance of about 2–3) predicts that the call follows the count column: capped ≈ unedited, and keep-2 ≈ lifetime × 0.03 (both about
1.5–2 tracks). Both predictions hold. This control is stronger than H3's lifetime × 0.35, which changes the count as well as the size.

## Corrected reading

On fresh samples, the boost (JEDI W 87.4 → 26.9 under pT × 0.9, rotation control 87.7; ParT_full Wqq −6.3 points vs rotation) and
H→bb track results reproduce within about 2 points of H3. For ParT_full H→bb, capping every track's significance at 3.5 keeps the
displaced-track count fixed and costs 3.1 points. Reducing the count to 2 at full displacement costs 61 points. The call follows the
number of tracks with S ≳ 2–3, not the size of the displacement. H3's lifetime × 0.35 edit does not keep the S > 2 count fixed (−26 %).
For the W-pair mass inside top: H3's control also shifts the W-pair mass, by 10–28 GeV the other way. Against a control that keeps the
W pair at its original mass, the W-pair edit's effect on the top call does not differ beyond the 95 % intervals after Holm correction,
for JEDI-linear, ParT_full or ParT_kin. ParT's top call falls under both, by up to 16 points at 120 GeV. The ParT_full ±2.7-point
asymmetry is not replicated (fresh sample +0.9 [−0.3, 2.0] and −1.1 [−2.6, 0.3]). Not rerun: lepton isolation, lepton charge flip,
π⁰, dead cone, decay collapse, remove-b and the rejection numbers.

## Suggested changes to H3's thesis note

- D1: do not add "ParT_full depends a little on the W-pair mass inside top". Replace it with: ParT's top call responds to rearranging
  the three subjet-pair masses at fixed jet mass (up to −16 points). The response is the same whether or not the W pair is the one that
  moves (difference ≤ 3.3 points, CI includes 0).
- D2: replace "lifetime at fixed S > 2 count" with the capped-significance result.
- Figure: add control B to the W-pair rows, replace the keep-2 "control" with the capped edit, and add the H→ℓνqq′ charge-flip and
  H→cc dead-cone rows if they stay in the reading.
