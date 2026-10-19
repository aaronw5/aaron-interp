# H8 — EFP/ADO guided search (Faucett-style) vs causal edits on JEDI-linear and ParT (novelty test for D1/D4/D5)

## Conclusion
For each network and binary task we built the closest IRC-safe stand-in from standard jet shapes (energy-flow polynomials, jet and soft-drop mass, N-subjettiness ratios, C2/D2), choosing them by how well they rank jets the same way as the network (average decision ordering, ADO: the fraction of signal–background pairs ordered the same way). On unedited jets these stand-ins rank W/QCD and top/QCD pairs like the networks in 90–97 % of pairs, but when the particle coordinates are recomputed about the winner-take-all axis the networks change up to 37 % of their binary calls while the stand-ins change none, and the same holds for a 0.06 rad translation and (outside the 64-particle truncation) for an exact collinear split. A summary in terms of standard IRC-safe shapes therefore reproduces what these taggers decide on natural jets but not how they respond to the axis convention or to particle splitting; this difference follows from the surrogate being axis-free and IRC-safe by construction, so the measurement sizes the gap rather than discovering a new variable. Open: for ParT's Z vs W task the best stand-in reaches only ADO 0.66, so part of that network's natural-jet decision is not captured by this pool at all, and an untrained ParT also responds strongly to these edits (generic input sensitivity).

## Question and hypotheses
Is the information the networks use on natural jets fully described by IRC-safe, translation-invariant shapes (EFPs d ≤ 5, m, m_SD, τ21, τ32, C2, D2), and does such a description also predict their responses to the D4/A4/H3/H1 edits?
- H0 (per model × task × edit): % binary calls changed by the edit for the network = % changed for its ADO-chosen surrogate (paired, exact McNemar on discordant jets).
- H1: network change ≠ surrogate change.
- Predictions. "Networks use only standard IRC-safe shapes": ADO > 0.95 and matching edit responses (≈ 0 for WTA, translation, split). D4/D5 thesis: ADO > 0.9 on natural jets, surrogate ≈ 0 % under WTA/translation vs tens of % for the network, ≈ 0 vs 1–5 % under splits. Generic input sensitivity: a weight-redrawn network responds to the edits as much as the trained one.

## Procedure
Script `interp/h8_efp_ado.py`; cluster jobs `d-h8-jedi` (jax env, CPU), `d-h8-edit` (GPU, ParT), `d-h8-efp` (venv-ef with energyflow, 4 CPU / 6 GiB), `d-h8-diag` (invariance check + figure).
- Samples. hls4ml: `jedi_core.sample(per_class=2000, seed=801)`, N = 10,000, 64 slots (22.8 % of jets fill all 64). JetClass: 1,000 random rows per class (rng 802) from each class's first test file, N = 10,000 for accuracy; surrogates and edits on the 4,000 QCD (ZJetsToNuNu), Wqq, Zqq, Tbqq jets; build η = jet_η + sign(jet_η)·Δη.
- Models: JEDI-linear official n64 3-feature; ParT_full; ParT_kin. Nulls: JEDI-linear and ParT_full with all weights redrawn (seed 0).
- Tasks and score: W vs QCD, top vs QCD, Z vs W; score = log Σ_sig p − log Σ_bkg p (hls4ml QCD = g + q), binary call = score > 0.
- Observable pool "irc": all EFP multigraphs with 1 ≤ d ≤ 5 edges (101 graphs) at β ∈ {0.5, 1, 2} (hadronic measure, z_i = pT_i/ΣpT, θ_ij = ΔR_ij^β; 303 EFPs), m, m_SD (C/A, z_cut 0.1, β 0), τ21, τ32 (exclusive-kT axes), C2 = (2/3)·e3/e2², D2 = (4/3)·e3/e2³ (β 1), computed on the particles the network sees. Pool "ext" = irc + N, pT^D (IRC unsafe).
- Guided search (Faucett et al. 2021): jets split 50/50 into fit and test halves (rng 803); fit half split 2:1 into search A and stopping V (rng 804). Step 1: the observable with the largest orientation-free ADO with the network score on 150,000 random (sig, bkg) pairs of A. Step k: a HistGradientBoosting (HGB) regression of the network score on the chosen observables (2-fold out-of-fold on A); add the observable with the largest ADO on the A pairs the current surrogate orders differently from the network. Stop when ADO on V gains < 0.002 or at 8 observables. Final surrogates (HGB; MLP 32-32 on log-transformed inputs) fitted on the fit half, evaluated on the test half.
- Edits on the same jets: wta (D4: (Δη, Δφ) recomputed about the C/A WTA axis; four-vectors unchanged); translate (A4: Δφ + 0.06 rad for all real particles); boost (H3: pT × 0.9, angles / 0.9); split (H1 `split_r0_z0.05_dR0`: the hardest particle into 5 %/95 % daughters at ΔR = 0, re-sorted, cut to 64/128 slots).
- Metrics (test half: hls4ml about 3,000 jets for W or top vs QCD and 2,000 for Z vs W; JetClass about 1,000 per task): ADO over all test (sig, bkg) pairs with a 95 % bootstrap interval (jets resampled per class, B = 400); same-call %; per edit % calls changed for network and surrogate with 95 % bootstrap intervals (B = 1,000), paired difference interval, exact McNemar, Δscore; Benjamini–Hochberg over 36 tests (3 trained models × 3 tasks × 4 edits, irc pool, HGB).
- Invariance check (`h8_diag.json`): fraction of jets whose observables change by > 1e-2 (relative) per edit.

## Unedited accuracy on the samples (reference in brackets)
- JEDI-linear 80.86 % (5 classes, N 10,000) [80.6 % E17/D4; 81.1 % H3]; 5-class calls changed by WTA 30.50 % [30.3 % D4].
- ParT_full 86.05 % (10 classes, N 10,000) [85.9 % D4; 86.38 % H3]; ParT_kin 73.71 % [74.5 % D4; 74.55 % H3]. On the 4-class rows: 81.13 % / 71.17 %. WTA changes 17.75 % / 36.50 % of 10-class calls on the 4-class rows (D4's 11.7 % / 34.1 % was over all 10 classes).
- Redrawn nets: JEDI 20.00 % (one constant call), ParT_full 10.00 %.

## Invariance of the observables (h8_diag.json, fraction of jets changing > 1e-2)
- WTA, translate: ≤ 0.0001 for every family (m_SD tie-breaks only), both datasets.
- Split: JetClass ≤ 0.0003 (one overflow jet); hls4ml 0.5–2.7 % of jets, all from the 22.8 % that fill 64 slots (truncation); N and pT^D change in 77–100 %.
- Boost: EFPs and C2 change in all jets (θ^β scaling: IRC safe but not boost invariant); D2, τ21, τ32 unchanged; m, m_SD change in ≤ 1.3 %.

## Numbers (irc pool, HGB surrogate, test half)

| model | task | chosen observables (order) | ADO [95 % CI] | same-call % |
|---|---|---|---|---|
| JEDI | W vs QCD | τ21, EFP β2 #14, τ32, β2 #48, #15, #34, #29, #24 (8 = cap) | 0.933 [0.923, 0.942] | 91.2 |
| JEDI | top vs QCD | EFP β0.5 #58, m, β0.5 #94 | 0.956 [0.951, 0.961] | 91.9 |
| JEDI | Z vs W | m, C2 | 0.977 [0.972, 0.981] | 93.7 |
| ParT_full | W vs QCD | τ21, τ32, EFP β1 #3 | 0.896 [0.877, 0.914] | 83.7 |
| ParT_full | top vs QCD | EFP β1 #12 (triangle = e3), τ32 | 0.966 [0.956, 0.977] | 89.0 |
| ParT_full | Z vs W | m | 0.663 [0.631, 0.690] | 62.2 |
| ParT_kin | W vs QCD | τ21, τ32, β1 #4, β2 #43, β1 #12, β2 #24 | 0.915 [0.900, 0.929] | 86.7 |
| ParT_kin | top vs QCD | EFP β1 #12, τ32, β2 #17 | 0.969 [0.960, 0.978] | 91.1 |
| ParT_kin | Z vs W | m, β1 #69, β0.5 #1, β2 #14 | 0.754 [0.732, 0.775] | 75.6 |
| JEDI redrawn | 3 tasks | – | 0.525–0.594 | 100 (constant) |
| ParT_full redrawn | 3 tasks | – | 0.549–0.583 | 56–92 |

EFP graphs: #1 single edge (e2); #3, #4 two vertices with 3/4 parallel edges (Σ z_i z_j θ_ij^{3β}, ^{4β}); #12 triangle (e3); #14, #15 triangles with multiplied edges; #17, #34, #48 stars (one hard particle correlated with 3–5 others); #58 product of two two-point graphs (d 2 × d 3). The MLP surrogate gives ADO within 0.02 of HGB in every row. The "ext" pool adds N (JEDI top vs QCD, ADO 0.960) or pT^D (ParT_kin W vs QCD, ADO 0.906); elsewhere it chooses the same set.

**% binary calls changed: network / surrogate; difference [95 % CI]; BH q**

| model | task | WTA | translate 0.06 | boost ×0.9 | split |
|---|---|---|---|---|---|
| JEDI | W vs QCD | 17.3 / 0.0; [16.0, 18.6]; 4e-155 | 41.0 / 0.0; [39.2, 42.9]; <1e-300 | 9.7 / 6.1; [2.4, 4.7]; 6e-9 | 0.8 / 0.1; [0.4, 1.1]; 3e-5 |
| JEDI | top vs QCD | 8.1 / 0.0; [7.2, 9.1]; 3e-73 | 10.6 / 0.0; [9.5, 11.7]; 5e-95 | 9.7 / 9.5; [−1.1, 1.4]; 0.85 | 1.2 / 0.4; [0.4, 1.4]; 3e-4 |
| JEDI | Z vs W | 37.3 / 0.0; [35.2, 39.5]; 7e-224 | 52.8 / 0.0; [50.6, 55.0]; 9e-317 | 17.5 / 0.8; [15.2, 18.5]; 1e-89 | 2.5 / 0.0; [1.8, 3.2]; 1e-14 |
| ParT_full | W vs QCD | 3.4 / 0.0; [2.4, 4.6]; 2e-10 | 3.6 / 0.0; [2.5, 4.8]; 6e-11 | 1.9 / 6.6; [−6.4, −3.1]; 4e-8 | 0.6 / 0.0; [0.2, 1.1]; 0.034 |
| ParT_full | top vs QCD | 1.7 / 0.0; [0.9, 2.5]; 4e-5 | 0.9 / 0.0; [0.3, 1.6]; 0.0045 | 1.1 / 3.1; [−3.3, −0.6]; 0.0049 | 0.1 / 0.0; [0.0, 0.3]; 1.0 |
| ParT_full | Z vs W | 18.1 / 0.0; [15.8, 20.2]; 2e-56 | 14.4 / 0.0; [12.3, 16.7]; 5e-45 | 5.6 / 0.8; [3.4, 6.4]; 3e-10 | 3.5 / 0.0; [2.4, 4.6]; 6e-11 |
| ParT_kin | W vs QCD | 4.0 / 0.0; [2.8, 5.3]; 4e-12 | 3.8 / 0.0; [2.6, 5.0]; 2e-11 | 1.8 / 9.3; [−9.6, −5.6]; 9e-14 | 0.9 / 0.0; [0.4, 1.5]; 0.0045 |
| ParT_kin | top vs QCD | 4.6 / 0.0; [3.3, 6.0]; 3e-13 | 2.9 / 0.0; [1.9, 4.0]; 1e-8 | 0.8 / 2.7; [−3.1, −0.7]; 0.0036 | 0.2 / 0.0; [0.0, 0.5]; 0.53 |
| ParT_kin | Z vs W | 24.2 / 0.0; [21.6, 26.7]; 3e-75 | 19.3 / 0.0; [17.0, 21.8]; 3e-60 | 3.1 / 7.2; [−6.2, −2.3]; 3e-5 | 2.6 / 0.0; [1.6, 3.6]; 2e-8 |

- 33 of 36 tests have BH q < 0.05. Not significant: JEDI top vs QCD boost (9.7 vs 9.5), ParT_full and ParT_kin top vs QCD split (0.1 and 0.2 % vs 0).
- JEDI split restricted to jets without 64-slot overflow: surrogate 0.0 % in all three tasks (network 0.8 / 1.1 / 2.3 %).
- Nulls (redrawn weights, % calls changed WTA / translate / boost / split): JEDI 0 / 0 / 0 / 0 (constant call); ParT_full W vs QCD 27.4 / 33.3 / 22.2 / 17.7, top vs QCD 12.4 / 16.0 / 11.1 / 7.4, Z vs W 32.5 / 31.1 / 22.5 / 19.2; its surrogates (ADO ≤ 0.58) change 0 % except under boost.
- MLP surrogates: same pattern (WTA, translate 0.0 % in all 9 rows; split ≤ 0.1 %).

## Reading
- On natural jets the guided search reaches ADO 0.90–0.97 for W vs QCD and top vs QCD in all three networks and 0.977 for JEDI's Z vs W (m and C2); ParT_full's and ParT_kin's Z vs W surrogates reach 0.66 and 0.75 (network AUC 0.91 / 0.81, surrogate 0.64 / 0.69), so for Z vs W in ParT the pool does not describe the ranking.
- The first observables chosen are the standard ones: τ21 for W vs QCD in all three networks; τ32 with the triangle EFP (e3, the numerator of C2/D2) for top vs QCD in ParT; jet mass first for Z vs W in all three networks. Mass enters directly as m rather than as the β = 2 single-edge EFP. JEDI's W vs QCD set adds five β = 2 EFPs and hits the 8-observable cap.
- Under WTA re-axis and translation every network changes 0.9–52.8 % of its binary calls and every surrogate 0.0 %. Because the surrogate is translation-invariant by construction, this measures how much of each network's call depends on a non-translation-invariant quantity at a given natural-jet ADO, e.g. JEDI Z vs W: ADO 0.977 on natural jets, 37.3 % vs 0.0 % calls changed under WTA.
- Under the collinear split the surrogates change 0.0 % outside truncation; the networks change 0.6–3.5 % except top vs QCD in ParT (0.1–0.2 %, not significant).
- Under the boost the surrogates also change (EFPs scale as θ^β), so it is not a zero-prediction control. JEDI changes more than its surrogate (Z vs W 17.5 vs 0.8 %, consistent with the ΣpT dependence in D1/H3); ParT changes fewer calls than its surrogates (1.9 vs 6.6 % for ParT_full W vs QCD).
- Predictions. "Only standard IRC-safe shapes" (ADO > 0.95 and matching edit responses): excluded for WTA and translation in all 9 model × task rows and for the split in 7 of 9. D4/D5 thesis (ADO > 0.9, surrogate ≈ 0 vs network ≫ 0): consistent for W/QCD and top/QCD and for JEDI Z vs W; for ParT Z vs W the ADO precondition is not met. Generic sensitivity: the redrawn ParT_full changes more calls (7–33 %) than the trained one, so the size of a ParT response alone does not show learned use; the redrawn JEDI gives a constant call and is not an informative null.
- Scope: the binary call is score > 0 on a 2-class log-ratio, not the 10-class argmax; test halves are about 1,000–3,000 jets per task; ADO intervals resample jets, not the search.

## Files
- `interp/h8_efp_ado.py` (parts jedi, jetclass, efp, diag, plot)
- `results/physics/h8_efp.json` (search steps, ADO, edit stats, BH q), `h8_jedi_net.json`, `h8_jc_net.json`, `h8_diag.json`, `h8_fig.png`
- Volume only (large): `h8_obs_{jedi,jc}.npz`, `h8_{jedi,jc}_net.npz`
- Logs: `logs/d-h8-jedi.log`, `d-h8-edit.log`, `d-h8-efp.log`, `d-h8-diag.log`
