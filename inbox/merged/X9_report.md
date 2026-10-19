# X9 — JEDI-linear vs Energy Flow / Particle Flow Networks on the same 3-feature input (claims A3, A4, A5, D4, B2; review weakness 1)

Status: done (EFN labelled: it reaches 74.0 / 74.3 % dev, not the 79.9 % target).
Code: `interp/x9_train.py` (venv training), `interp/x9_efn_pfn.py` (analysis, base jax env), `interp/x9_plot.py` (figure, local).
Results: `results/jedi/x9_pfn_64-64-32_s0.json`, `x9_pfn_64-64-32_s1.json`, `x9_efn_100-100-128_s0.json`, `x9_efn_100-100-128_s1.json`,
`x9_summary.json` (Holm over all 56 model-vs-JEDI tests), `x9_models/<model>/{weights.npz, info.json, history.json, check.npz}`,
figure `results/jedi/x9_compare.png`. Logs: `logs/x9-*.log` (copied from /j-jepa-vol/aaron-interp/logs).

## Conclusion
A PFN trained to JEDI-linear's accuracy and an EFN (74 %) both change W calls to Z when every particle is translated by
0.06 rad with the invariant mass unchanged, and their mass read-outs move by about the amount a real mass change of the
predicted size moves them. So the un-recentred mass response is found in all three architectures on this input and is not
specific to JEDI-linear. JEDI-linear's rank-window response (5.2 times larger change for a swap across rank 14 than inside)
and its split of first-layer cuts into pure pT ramps and pure angular half-planes are not found in the PFN/EFN; the PFN/EFN
latents are named by the same radial-moment family less precisely (median held-out R² 0.73–0.87 vs 0.98). Open: the EFN is
6.6–6.9 points below the matched accuracy, and the IIA of the named formulas is above the permuted alignment but not above
random rotations for any of the five models, JEDI-linear included.

## Procedure
Environment. venv `/j-jepa-vol/aaron-interp/venv-ef` (tensorflow-cpu 2.15.1, energyflow 1.4.0, numpy 1.26.4, h5py 3.16; no system
site-packages), built in rounds before this one (x9-env, x9-env2 import check: ENVCHECK OK). Jobs extract `venv-ef.tar` to the pod's
/tmp (45 s) and run `/tmp/venv-ef/bin/python -m interp.x9_train`.

Training (`interp/x9_train.py`). energyflow.archs EFN (input z_i = pT_i / Σ_{j≤64} pT_j, (Δη_i, Δφ_i) in rad as stored;
latent L_k = Σ_i z_i Φ_k(Δη_i, Δφ_i)) and PFN (input (pT, Δη, Δφ) normalised as JEDI-linear / train_small, padding rows set to 0
and masked; L_k = Σ_i Φ_k). First 64 particles. Split as train_small: 595,000 train, dev 25,000 (splits.npz), test = val archive
(260,000; not used for selection). F = (100, 100, 100), Adam 1e-3, batch 500, early stopping on dev loss (patience 5, best weights).
1-epoch probe: 50–57 s/epoch, 2.0 cores, 4.2 GiB → jobs requested 3 CPU / 6 GiB (measured 2.0–2.6 cores, 3.7–5.9 GiB peak).
Φ size chosen on seed-0 dev accuracy: the smallest Φ within 1 point of 80.9 %.

| model | Φ | seed | dev acc | test acc (260k) | best epoch |
|---|---|---|---|---|---|
| PFN | (64,64,32) | 0 | 81.12 % | 81.23 % | 21 |
| PFN | (64,64,32) | 1 | 80.98 % | 81.11 % | 22 |
| PFN | (100,100,64) | 0 | 81.28 % | 81.50 % | 27 |
| PFN | (100,100,128) | 0 / 1 | 81.79 / 81.36 % | 81.61 / 81.35 % | 31 / 18 |
| EFN | (64,64,32) | 0 | 73.70 % | 73.89 % | 41 |
| EFN | (100,100,64) | 0 | 73.11 % | 73.00 % | 30 |
| EFN | (100,100,128) | 0 | 73.93 % | 74.00 % | 29 |
| EFN | (100,100,128) | 1 | 74.32 % | 74.38 % | 54 |
| EFN, exploratory: angles ÷ 0.07 rad | (100,100,128) | 0 | 74.41 % | 74.60 % | 41 |

Selected: PFN (64,64,32) seeds 0, 1; EFN (100,100,128) seeds 0, 1 (best EFN; gap to 80.9 %: 6.6–6.9 points, labelled). The
exploratory EFN with standardised angles (still IRC-safe) reaches 74.4 %, so the EFN gap is not from the angle scale of the input.

Analysis (`interp/x9_efn_pfn.py`, one 1 CPU / 2 GiB job per model, 12–44 min; frozen header in the module docstring). The exported
Dense weights run in an exact numpy forward pass, next to `interp.jedi_core.Jedi(64)`, so every statistic is paired over the same jets.
Samples: T = sample(per_class=2000, seed=313), N = 10,000 test jets; Fs = sample(per_class=6000, seed=0) minus the 1,110 jets in T
(28,890; fit 19,260 / dev 9,630). Steps:
0. accuracy on T; |TF − numpy| softmax (max < 1e-5 for all four models); JEDI S-override identity (0 exactly).
1. (E2 analog) first Φ-layer units: live = mean ablation over real particles changes ≥ 1 % of calls (T[::2]); typed by term
   spread w_k·sd(x_k): ramp |w_ang|/|w| < 0.05, angular |w_pT|/|w| < 0.05 (half-plane if the line passes < 0.01 rad from the axis
   at median pT), mixed otherwise; null (PFN): kernel rows permuted across units, 20,000 draws, statistic = pure fraction.
2. (E5 naming) latent unit cost = % calls changed when L_k takes another jet's value (one fixed derangement); live ≥ 0.1 %.
   Each live unit MARS-fitted (jetdistill.mars.fit_neuron, ≤ 15 terms, 6,000 selection jets) on the E11 library (380 Q
   observables + rank-window sums of pT, pT·r, pT·r², r, r², counts, hinges); held-out R² on T; top term by weight share.
   Exact radial share of Φ_k on an (r 0–0.4 rad × θ 0–355°) grid (× pT 1, 3, 10, 30, 100 GeV for PFN); first-layer units as
   control. Cross-readability: ridge R² of each live JEDI S unit from the latent and vice versa (fit Fs, test T).
3. Edits on T (padding untouched): (a) translation of every real particle by 0.06 rad along +Δη, +Δφ, and a random direction per
   jet; control: matched rotation about the axis (equal mean displacement, m and M2 unchanged). Calibration: angle scaling by
   k = exp(mean ½Δlog M2) (a real mass change of the predicted mean size). Read-outs: ridge of log m on L and on the last F layer h
   (JEDI: S and z), fitted on unedited Fs. (b) WTA axis (C/A, winner-take-all; E17 code), prediction ½Δlog(M0·M2).
   (c) rank swap: exchange the (Δη, Δφ) of ranks 13|14 vs the X6 inside pair 11|12 (and 0|1 vs 3|4); m = mean over the 10
   costliest latent units (JEDI: X6's S0, S43, S51, S52, S63) of |ΔL|/sd; ratio across / inside on common jets.
4. (E22 IIA) per named latent unit: H = the model's own F layers on L̂ (every named unit replaced by its MARS formula); interchange
   on 5,000 ordered (base, source) pairs of T; effective IIA = agreement on pairs where the swap changes the model's call.
   Baselines: 200 derangements of the unit–formula alignment (one-sided permutation p, floor 1/201), 200 random rotations in the
   standardised named-unit space. JEDI on the same pairs: S with the E11 (a) formulas (19 units).
Statistics: 1000 jet-bootstrap 95 % CIs (300 pair-bootstraps for IIA); H0: model statistic = JEDI statistic, paired bootstrap p
(two-sided, floor 1/1001); McNemar for paired call changes; Holm within each model (14 tests) and over all 56 (`--summary`).

## Results (T, N = 10,000; JEDI-linear on the same jets)

Step 0. Accuracy on T: JEDI-linear 80.23 % [79.44, 81.00] (Wilson); PFN s0 80.42 % [79.63, 81.19], PFN s1 80.37 %; EFN s0
73.43 % [72.56, 74.29], EFN s1 74.30 %. JEDI's T accuracy is 0.37 points below the 80.6–81.3 % check range; its interval
contains 80.9 % (official full-test accuracy). Call agreement with JEDI: PFN 92.3 / 92.5 %, EFN 81.0 / 81.2 %.

Step 1 — first Φ layer (JEDI X6: 13 live l1 units, 4 ramps + 9 half-planes, 0 mixed).
| | live | ramp | half-plane | angle-offset | mixed | pure fraction | null (shuffled kernel rows) |
|---|---|---|---|---|---|---|---|
| PFN s0 | 64/64 | 1 | 2 | 4 | 57 | 0.11 [0.05, 0.21] | median 0.11 [0.08, 0.14]; p = 0.83 |
| PFN s1 | 64/64 | 1 | 1 | 5 | 57 | 0.11 [0.05, 0.21] | median 0.09 [0.06, 0.11]; p = 0.11 |
| EFN s0 | 76/100 | – | 27 | 49 | – | angular by construction | – |
| EFN s1 | 77/100 | – | 25 | 52 | – | angular by construction | – |
PFN pT-weight share of the live units (quartiles): 0.31 / 0.67 / 0.84 (s0). H0 (PFN pure fraction = shuffled-kernel null) is not
rejected for either seed.

Step 2 — latent units.
| | live (≥ 0.1 %) | ≥ 1 % | top cost | Φ radial share median (≥ 0.9) | MARS test R² median (≥ 0.8) | top-term family: radial / ΣpT / multiplicity / mass+ECF / other |
|---|---|---|---|---|---|---|
| JEDI S (E11, X6) | 19 | 19 | 40.9 % | top-5 S: 0.98–1.00 | 0.98 (17/19) | 13 / 4 / 1 / 0 / 1 |
| PFN s0 | 30/32 | 27 | 30.5 % | 0.48 (4/30) | 0.85 (20/30) | 18 / 8 / 4 / 0 / 0 |
| PFN s1 | 30/32 | 29 | 24.9 % | 0.56 (6/30) | 0.87 (20/30) | 20 / 3 / 4 / 0 / 3 |
| EFN s0 | 113/128 | 92 | 15.0 % | 0.33 (16/113) | 0.82 (63/113) | 91 / 0 / 0 / 14 / 8 |
| EFN s1 | 110/128 | 89 | 26.4 % | 0.37 (18/110) | 0.73 (40/110) | 84 / 0 / 0 / 10 / 16 |
First-layer control radial share medians 0.17–0.23. MARS dev R² at 1 / 3 / 15 terms: JEDI 0.90 / 0.95 / 0.98; PFN 0.66–0.69 /
0.75–0.80 / 0.85–0.87; EFN 0.39–0.54 / 0.55–0.67 / 0.73–0.81. Top terms of the costliest units: PFN s0 L3 (30.5 %, R² 0.92)
ΣpT·r²[ranks 1–63]; L17 ΣpT·r²[top 39] hinge; L5 ΣpT·r[top 26] hinge; L13, L14 Σ pT. PFN s1 L31 (24.9 %) Σr; L18, L6 ΣpT·r²
hinges. EFN s0 L2 (15.0 %, R² 0.97) e2² (ECF) hinge; L47, L81 m/ΣpT² ; EFN s1 L99 (26.4 %, R² 0.999) girth² hinge; L102, L84
m/ΣpT². Rank-window terms are top terms of 15/30 and 11/30 PFN units; the PFN is permutation invariant, so these terms act as
correlated proxies (step 3c). Cross-readability (ridge, held-out R² median): JEDI S from PFN latent 0.93 / 0.95, from EFN latent
0.68 / 0.68; PFN latent from JEDI S 0.70 / 0.74; EFN latent from JEDI S 0.78 / 0.73.

Step 3a — 0.06 rad translation (m unchanged; predicted ½Δlog M2 = 0.305 mean). W → Z among true-W jets (unedited 1.8–3.5 %),
calibrated read-out ratio at h / z (1 = moves as a real mass change of the predicted size; 0 = invariant mass):
| edit | JEDI W→Z | PFN s0 | PFN s1 | EFN s0 | EFN s1 | cal. ratio JEDI z | PFN h | EFN h |
|---|---|---|---|---|---|---|---|---|
| rotation (control) | 2.0 % [1.5, 2.7] | 2.9 % | 1.5 % | 2.9 % | 2.9 % | – | – | – |
| random direction | 61.2 % [59.0, 63.3] | 57.2 % [55.0, 59.4] | 71.0 % | 55.3 % | 68.9 % | 1.07 [1.05, 1.10] | 1.02 / 1.16 | 1.15 / 0.97 |
| +Δη | 75.6 % [73.7, 77.4] | 54.2 % | 71.7 % | 37.4 % [35.3, 39.5] | 71.6 % | 0.98 | 1.01 / 1.20 | 1.10 / 1.05 |
| +Δφ | 76.5 % [74.6, 78.4] | 82.0 % | 50.9 % | 76.8 % | 70.6 % | 0.99 | 0.97 / 1.30 | 1.16 / 1.04 |
Calls changed by the translations: JEDI 56–60 %, PFN 60–68 %, EFN 73–76 %; rotation 4.4 % / 5.4–5.7 % / 6.4–8.9 %. McNemar of
call changes, translation vs rotation: p < 1e-200 for every model and direction. Calibrated ratio at the latent L: PFN 0.78–1.39,
EFN 0.77–0.88 (JEDI S 1.13–1.24). Paired W → Z difference model − JEDI (bootstrap 95 %): random direction PFN −4.0 [−6.5, −1.4] /
+9.8 [7.4, 12.3] points, EFN −5.8 [−8.4, −3.3] / +7.7 [5.3, 10.2]; +Δη PFN −21.4 / −3.9, EFN −38.2 / −4.0; +Δφ PFN +5.4 / −25.6,
EFN +0.3 [−1.7, 2.2] / −5.9. The direction asymmetry (+Δη vs +Δφ) varies in sign between PFN seeds (s0 54 vs 82 %, s1 72 vs
51 %) and is 1 point for JEDI. McNemar of call changes, axis-aligned vs random-direction translation: JEDI p = 1e-27 (+Δη) and
2e-15 (+Δφ), more calls changed by the axis-aligned edit; PFN s0 +Δη p = 0.26, s1 +Δη p = 0.36; EFN s1 p = 0.14 / 0.23.

Step 3b — WTA axis (½Δlog(M0·M2) = 0.120 mean). W → Z: JEDI 56.5 % [54.3, 58.7]; PFN 58.7 / 58.5 % (paired difference +2.2,
p = 0.053 / +2.0, p = 0.067); EFN 70.8 / 70.7 % (+14.3 / +14.2 points, p = 0.001). Calibrated ratio at h: JEDI z 0.88
[0.86, 0.91], PFN 0.89 / 0.64, EFN 0.35 / −0.37. At the latent stage the read-out moves opposite to the prediction for every model
(JEDI S −1.06, PFN L −1.12 / −1.44, EFN L −2.23 / −2.19).

Step 3c — rank-window swap, ranks 13|14 vs 11|12 (N = 9,960). Ratio across / inside of mean |ΔL|/sd: JEDI 5.25 [4.98, 5.54];
PFN 0.82 [0.78, 0.86] / 0.84 [0.80, 0.87]; EFN 0.79 [0.74, 0.85] / 0.81 [0.76, 0.86]; pT-share gap ratio of the swapped pairs
0.72. Ranks 0|1 vs 3|4 (N = 10,000): JEDI 0.16 [0.05, 0.30]; PFN 2.18 / 1.90; EFN 3.43 / 3.50; gap ratio 7.69. For the flow
models the ratio stays between the gap ratio and 1 at the rank-14 edge; JEDI's is 7.3× the gap ratio.

Step 4 — IIA (5,000 pairs; mean effective IIA [95 %]; permuted-alignment mean; random rotations mean).
| | units | named formulas | permuted (200) | p (perm.) | rotations (200) | p (rot.) | units > own perm. 97.5 % | paired model − JEDI |
|---|---|---|---|---|---|---|---|---|
| JEDI S (E11 formulas) | 19 | 0.701 [0.684, 0.720] | 0.251 | 0.005 | 0.705 | 0.60 | 19/19 | – |
| PFN s0 | 30 | 0.531 [0.505, 0.555] | 0.276 | 0.005 | 0.529 | 0.48 | 19/30 | −0.170 [−0.201, −0.140] |
| PFN s1 | 30 | 0.576 [0.554, 0.598] | 0.321 | 0.005 | 0.565 | 0.44 | 17/30 | −0.124 [−0.154, −0.096] |
| EFN s0 | 113 | 0.580 [0.542, 0.617] | 0.487 | 0.005 | 0.694 | 0.95 | 10/113 | −0.121 [−0.165, −0.076] |
| EFN s1 | 110 | 0.515 [0.484, 0.547] | 0.434 | 0.005 | 0.635 | 0.94 | 5/110 | −0.186 [−0.226, −0.151] |
Accuracy of H (all named units replaced by formulas): JEDI 80.1 %; PFN 74.7 / 74.6 % (agreement with the PFN 87.3 / 86.5 %);
EFN 56.5 / 55.3 % (agreement 67.4 / 63.9 %). Per-unit permutation p has floor 1/201, so no unit passes Holm at 0.05 in any model
(30–113 tests); the count above the unit's own 97.5 % null quantile is reported instead.

Model-vs-JEDI tests: 56 (14 per model). Per-model Holm: 49 of 56 below 0.05 (minimum 0.014). Holm over all 56: minimum 0.056,
because the bootstrap p floor (1/1001) × 56 exceeds 0.05; no test passes at 0.05 over all models with B = 1000.
Not below 0.05 within model (7): rotation W → Z (PFN s0 0.069, s1 0.134; EFN s0 0.074), WTA W → Z (PFN s0 0.106, s1 0.134),
EFN s0 +Δφ W → Z (0.79), PFN s0 WTA calibrated-ratio difference at h (0.88).

## Predictions and which are excluded
- JEDI-specific (EFN/PFN read-out ratio near 0 under translation, i.e. invariant mass): excluded — calibrated ratios at h
  0.97–1.30 (PFN) and 0.97–1.16 (EFN), W → Z 37–82 % vs rotation 1.5–2.9 %.
- Generic "translation W → Z ≥ 60 %, ratio 0.6–1.0": W → Z ≥ 60 % in 7 of 12 (model, direction) cells, below for PFN s0
  random/+Δη (57, 54 %), PFN s1 +Δφ (51 %), EFN s0 random/+Δη (55, 37 %); calibrated ratios at h have point estimates above 1.0
  in 10 of 12 cells (1.01–1.30; JEDI z 0.98–1.07).
- Generic "PFN latents radial, pT-weighted, R² ≥ 0.8 on ΣpT·r²-type terms": partly — radial-moment top families in 18–20/30
  units, R² ≥ 0.8 in 20/30, ΣpT·r / ΣpT·r² top terms in 8–10/30; Φ radial share median 0.48–0.56 (JEDI top S ≥ 0.98).
- Rank windows: absent in PFN and EFN (ratio 0.79–0.84 vs gap ratio 0.72; JEDI 5.25), as permutation invariance requires; the
  built-in negative control behaves as expected.
- First-layer cut types: JEDI's pure ramp / half-plane split (13/13 pure) is not reproduced by the PFN (0.11 pure = shuffled null).
- IIA "at the permuted level" (JEDI-specific prediction): excluded for all models (p = 0.005); IIA is 0.12–0.19 below JEDI's
  and no model, JEDI included, is above random rotations of the named-unit space.

## Reading (scoped to these measurements)
On this input and these test jets, the un-recentred mass response (a 0.06 rad translation moves W jets to Z and moves the mass
read-out like a real mass change of ½Δlog M2) is measured in JEDI-linear, in a PFN at matched accuracy (two seeds) and in an
EFN at 74 % (two seeds). The rank-window response, the pure ramp / half-plane first-layer cuts, the tight radial Φ maps and the
1–3-term formulas of JEDI's S units are not measured in the flow models; their latents are readable from the same observable
library (radial moments, ΣpT, multiplicity for PFN; ECF / girth / m/ΣpT² for EFN) at lower R². The direction asymmetry of the
translation response is seed dependent in the PFN. The EFN rows carry a 6.6–6.9-point accuracy gap; the exploratory
standardised-angle EFN (74.4 %) shows that the gap does not come from the input scale; whether it comes from the absence of
multiplicity / pT-scale information in an IRC-safe z-weighted latent is not tested here.

## Checks
- numpy forward vs TensorFlow softmax on T: max |Δp| < 1e-5 (4 models); JEDI S-override identity: 0.
- JEDI accuracy on T 80.23 % [79.44, 81.00] (0.37 below the requested 80.6–81.3 % check range; the full-test 80.9 % is inside
  the interval; same jets, same forward pass as E22).
- Smoke runs on 1-epoch models (`x9_*_probe_smoke.json`) exercised every step before the full runs.
- Two cluster pods were lost to a NotReady node (TaintManagerEviction) and rerun unchanged.
- Venv isolation: training in the venv only; analysis in the base env reads exported weights.

## Explainer entry (Paper A tab; site/explain.py)
**Question.** Are the elements found in JEDI-linear (pT-weighted radial objects over pT-rank windows; a mass variable that is not
recentred on the jet axis) a property of JEDI-linear, or of the task and the 3-feature input that any set network would learn?

**Background.** An Energy Flow Network computes $L_k=\sum_i z_i\,\Phi_k(\Delta\eta_i,\Delta\phi_i)$ with $z_i=p_{T,i}/\sum_j p_{T,j}$
(IRC safe); a Particle Flow Network computes $L_k=\sum_i \Phi_k(p_{T,i},\Delta\eta_i,\Delta\phi_i)$. Both are permutation invariant,
so they cannot use the pT-rank of a particle. Both see the angles about the stored jet axis, as JEDI-linear does. The un-recentred
mass variable is $\tfrac12\log(M_0M_2)$ with $M_0=\sum_i p_{T,i}$, $M_2=\sum_i p_{T,i}r_i^2$, $r_i^2=\Delta\eta_i^2+\Delta\phi_i^2$;
the invariant mass satisfies $m^2\approx M_0M_2-|M_1|^2$ with $M_1=\sum_i p_{T,i}(\Delta\eta_i,\Delta\phi_i)$.

**Method.** Train EFN and PFN on the same 595k jets (64 particles, two seeds). On the same 10,000 test jets as JEDI-linear:
(1) type first-layer cuts; (2) fit each latent unit with MARS on the E11 observable library; (3) translate every particle by
$\delta=0.06$ rad, which leaves $m$ unchanged and raises $\tfrac12\log M_2$ by 0.305 on average, and compare with a rotation of
equal displacement (both $m$ and $M_2$ unchanged); calibrate the mass read-out against a real angular scaling of the same
predicted size; swap the angles of ranks 13 and 14 vs 11 and 12; (4) interchange each named unit between jets in the network and
in its formula, and compare the agreement with permuted alignments.

**Result.** Figure `results/jedi/x9_compare.png`. Translation: W → Z 51–82 % (PFN), 37–77 % (EFN), 61–77 % (JEDI-linear);
rotation 1.5–2.9 %. Calibrated read-out ratio at the last hidden layer 0.97–1.30 (all three; 0 would be an invariant-mass
read-out). Rank-14 swap ratio: JEDI-linear 5.25, PFN/EFN 0.79–0.84 (pT-share gap ratio 0.72). Mean effective IIA:
JEDI-linear 0.70, PFN 0.53–0.58, EFN 0.52–0.58, each above its permuted alignment (p = 0.005).

**Checks.** numpy forward reproduces TensorFlow (|Δp| < 1e-5); two seeds; rotation control; paired bootstrap and McNemar tests;
EFN accuracy gap (74 % vs 80.9 %) stated; an EFN with standardised angles reaches the same 74.4 %.

**Why it matters.** The axis-dependent mass response is measured in three set architectures trained on the same input, so it
is attributable to the input representation (angles about the stored axis) and the training data rather than to JEDI-linear's
architecture; the rank-window objects and the pure cut types are measured only in JEDI-linear.
