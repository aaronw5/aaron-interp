# E31: are the loop-A input edits on the jet manifold? (claims A4, A4w, A15, A14; stage 4; review weakness "no C2ST on any loop-A edit")

Script `interp/e31_onmanifold_jedi.py` (job e31-run4, 1 CPU / 2 GiB, node nautilusg01.sci.cwru.edu, wall 1,682 s of which ~580 s was a slow CephFS import, peak memory 2,047 MiB, 0.54 cores average). Figure `interp/e31_plot.py` -> `results/jedi/e31_onmanifold.png` (rendered locally from the JSON). Results `results/jedi/e31_onmanifold.json`, example jet `results/jedi/e31_examples.npz`; run-1 output with the replaced subset rule kept as `results/jedi/e31_onmanifold_run1.json`. Logs `logs/e31-run2.log` (run 1), `logs/e31-run4.log` (final).

## Conclusion
A classifier separates the shifted and the azimuth-turned jets from real jets almost perfectly (AUC 1.00), because both edits move the jet's pT-weighted centroid off the jet axis, and real jets are centred on it. Real jets cannot supply the missing test either: their centroid offset changes the jet mass by at most 0.3 % (99th percentile), so in natural data the jet mass m and the axis moment $\sqrt{M_0 M_2}$ are the same quantity and no natural pair separates them. The claim that JEDI-linear reads M2 about the stored axis rather than m is therefore a statement about how the network responds to off-centre jets that do not occur in its data. The collinear-split and soft-particle results (A15) and the boost call change (A14) persist on edited jets that the classifier cannot tell from real ones. Open: the shift's mass response falls to 0.71 of a real mass change on the 7 % of shifted jets that look real once the centroid offset is ignored, and the boost's call-change rate falls from 33 % to 17–24 % on its look-real subsets.

## Definitions
- **Inputs.** Each jet is up to 64 particles, each with $(p_T, \Delta\eta, \Delta\phi)$ measured from the stored jet axis; $r_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$.
- **Moments.** $M_0=\sum p_T$ (summed pT), $\vec M_1=\sum p_T(\Delta\eta,\Delta\phi)$ (pT-weighted centroid times $M_0$; $|\vec M_1|/M_0$ is the centroid offset from the axis), $M_2=\sum p_T r^2$. For massless particles at small angles $m^2 \approx M_0M_2-|\vec M_1|^2$, so $m=\sqrt{M_0M_2}$ exactly when the centroid sits on the axis.
- **C2ST (classifier two-sample test).** A classifier is trained to tell edited jets from unedited jets. **AUC** = the probability that a randomly chosen edited jet gets a higher "edited" score than a randomly chosen unedited jet; 0.5 = cannot tell them apart, 1 = always tells them apart. Here: sklearn `HistGradientBoostingClassifier` (200 trees, learning rate 0.1, 31 leaves).
- **Feature sets.** `obs` = 9 jet observables: $m$, jet pT, particle count N, $\tau_{21}$, $\tau_{32}$ (N-subjettiness ratios, $\beta=1$; axes from hardest-particle + greedy $p_T\Delta R^2$ seeding and 8 pT-weighted Lloyd iterations), $D_2$ ($e_3/e_2^3$, $\beta=1$ energy correlators), girth $\sum p_T r/M_0$, $p_T^D=\sqrt{\sum p_T^2}/M_0$, and the centroid term $|\vec M_1|^2/(M_0M_2)$. `obs8` = the same without the centroid term. `plist` = $(p_T,\Delta\eta,\Delta\phi)$ of the 30 hardest particles (90 numbers).
- **R** = (mean change of the network's mass read-out under the edit) / (mean change under its matched real change, on the same jets). Mass read-out = ridge regression of $\log m$ on the 64 h4 neurons, fit on unedited jets (held-out $R^2$ 0.975). R = 1: the network responds as to a real mass change of that size; R = 0: no response.
- **Matched real change** = a dilation: every particle's $(\Delta\eta,\Delta\phi)$ multiplied by one constant per jet. A dilated jet has a larger (or smaller) mass the way heavier real jets do. Matched per edit: same $\Delta M_2$ (shift), same true $\Delta m$ (turn), same $\Delta\log M_2$ (boost), a fixed 5 % ($\times1.05$) for splits and soft additions, which change neither m nor M2.
- **Null edit** = an edit of equal size that should not change the call: a rigid rotation of the whole jet about the axis with the same pT-weighted mean particle displacement (shift, turn, boost); the split particle rotated about the axis without splitting (splits, as E28); 0.5 GeV added to the hardest particle (soft addition).
- **W→Z flow** = share of W-called jets called Z after the edit; **Z→W** likewise; **calls changed** = share of jets whose arg-max class changes. **McNemar** = exact binomial test on the jets where only one of two paired edits changes the call. **Holm** = step-down family-wise correction.

## Procedure
- Network: official JEDI-linear 3-feature n64, exact forward pass (`interp.jedi_core.Jedi`).
- Samples: edits on `sample(per_class=1600, seed=3101)` = 8,000 jets (1,600 per class); unedited accuracy **80.59 %** (check band 80.6–81.3 %; 0.01 pp below its lower edge, inside the ±0.44 pp sampling SE for 8,000 jets). Natural pool `sample(3000, 3102)` minus 1,455 jets also in the edit sample = 14,545 jets, accuracy 81.19 %.
- Edits (reimplemented or imported):
  - **shift** (E6/X8/E27w arm A): $\Delta\eta \to \Delta\eta+0.06$ rad for every real particle. Median $\Delta\log M_2$ = +0.44; true m unchanged (median $\Delta\log m$ 8e-9).
  - **turn5 / turn10** (E27w arm B 'hard14', `solve_B` imported): the 14 hardest particles are turned about the axis toward the azimuth of $\vec M_1$, each by the same fraction t of its angular distance to that azimuth, t solved per jet so the true mass falls 5 % / 10 %. Every particle keeps pT and r, so $M_0$, $M_2$, every $\sum p_T r^\beta$, ranks and slots are unchanged (max $|\Delta r|$ 1.9e-8 rad, median $\Delta\log M_2$ 4e-11); $|\vec M_1|$ grows. Feasible on 6,386 / 5,087 jets.
  - **split0 / split13 / split21** (E28w `split_ext` imported): the particle at pT rank k is replaced by two particles at the same $(\Delta\eta,\Delta\phi)$ with half its pT each, then re-sorted; $M_0$, $\vec M_1$, $M_2$ and the four-momentum unchanged. Applicable when N ≥ k + 2 (8,000 / 7,965 / 7,586 jets).
  - **soft** (E28 `soft_add` imported): one particle with pT 0.5 GeV at the r of a random real particle and a uniform azimuth (N ≤ 63: 6,238 jets).
  - **boost** (E26, `h12_boost_panel.boost`): exact boost along the jet momentum, $|P|\times0.9$ at fixed jet mass (max $|\Delta\log m|$ 6e-8; median $\Delta\log M_0$ −0.105, $\Delta\log M_2$ +0.105).
- C2ST per edit, per feature set: the jets where the edit applies are split at random into halves A, B. Classifier 1 separates A unedited from B edited, classifier 2 separates B unedited from A edited, each with 5-fold stratified cross-validation, so every jet gets an out-of-fold score $s_0$ for its unedited version and $s_E$ for its edited version, each from a classifier that never saw that jet. AUC = pooled AUC of $s_E$ vs $s_0$. 95 % CI: 1,000 bootstrap resamples of jets. H0: edited and unedited jets come from one distribution (AUC 0.5); H1: AUC ≠ 0.5; p from 1,000 random within-jet swaps of $(s_E, s_0)$ (floor 0.001). Retrain null (both halves unedited, 3 seeds): AUC 0.487–0.510 (obs), 0.492–0.506 (obs8), 0.501–0.504 (plist). ΔAUC vs the matched dilation and vs the null edit by paired bootstrap on the common jets.
- On-manifold subset (per edit, per feature set): jets whose edited version's score $s_E$ lies inside the central half of the unedited scores, $[q_{25}, q_{75}]$ of $s_0$ (by construction 50 % of unedited jets fall in this band, so an edit the classifier cannot detect keeps about 50 % of jets). `both` = inside for obs and plist; `obs8_plist` = inside for obs8 and plist. Headline numbers recomputed on each subset with 1,000 Poisson jet-bootstrap CIs; McNemar of the edit vs its null edit on the same subset, Holm over the 56 tests (8 edits × 7 jet sets).
- Natural pairs (pool, unedited jets; greedy unique nearest-neighbour pairs, same N, every matched variable within 1 % in log): arm `m_differs` (matched $M_0$, $M_2$, jet pT; $|\Delta\log m|$ ≥ 0.03, also ≥ 0.02); arm `M2_differs` (matched $M_0$, m, jet pT; $|\Delta\log M_2|$ ≥ 0.03); controls `all_matched` (everything matched) and `m_only` (m matched, nothing else). Per pair: Δ(read-out), compared with the prediction if the read-out follows $\sqrt{M_0M_2}$, $k\cdot\tfrac12(\Delta\log M_0+\Delta\log M_2)$, and if it follows m, $k\,\Delta\log m$, where k = 0.984 [0.974, 0.993] is the read-out change per unit $\Delta\log m$ under a real 5 % dilation of the pool jets.
- Run history: run 1 (`e31-run2`) used the subset rule $|s_E-0.5| <$ median $|s_0-0.5|$. With a near-perfect classifier both scores sit at $|s-0.5|\approx0.5$, so that rule kept every jet (e.g. 5,098 shifted jets at AUC 1.00). It was replaced by the $[q_{25},q_{75}]$ band and the obs8 set was added (debug attempt 1). C2ST AUCs for obs and plist are identical in both runs (same seeds).

## Results

### 1. C2ST AUC (95 % CI; N jets as in Procedure). H0: AUC = 0.5.
| edit | obs | obs8 (no centroid term) | plist | matched dilation obs / obs8 / plist | null edit obs / obs8 / plist |
|---|---|---|---|---|---|
| shift | 1.000 [1.000, 1.000] | 0.956 [0.953, 0.959] | 0.999 [0.999, 0.999] | 0.947 / 0.943 / 0.761 | 0.498 / 0.495 / 0.491 |
| turn5 | 1.000 [1.000, 1.000] | 0.883 [0.878, 0.889] | 0.942 [0.938, 0.945] | 0.739 / 0.741 / 0.512 | 0.494 / 0.497 / 0.496 |
| turn10 | 1.000 [1.000, 1.000] | 0.976 [0.973, 0.978] | 0.993 [0.992, 0.994] | 0.817 / 0.814 / 0.551 | 0.501 / 0.498 / 0.488 |
| split0 | 0.673 [0.666, 0.679] | 0.673 | 0.880 [0.874, 0.884] | 0.698 / 0.698 / 0.511 | 0.936 / 0.750 / 0.746 |
| split13 | 0.503 [0.493, 0.513] (p 0.58) | 0.508 (p 0.12) | 0.757 [0.752, 0.763] | 0.698 / 0.698 / 0.511 | 0.652 / 0.534 / 0.582 |
| split21 | 0.506 [0.496, 0.518] (p 0.27) | 0.510 (p 0.08) | 0.718 [0.711, 0.726] | 0.698 / 0.698 / 0.511 | 0.581 / 0.510 / 0.557 |
| soft | 0.518 [0.505, 0.529] (p 0.006) | 0.521 (p 0.002) | 0.641 [0.630, 0.653] | 0.698 / 0.698 / 0.511 | 0.499 / 0.498 / 0.505 |
| boost | 0.881 [0.875, 0.886] | 0.879 [0.873, 0.884] | 0.785 [0.779, 0.791] | 0.715 / 0.716 / 0.513 | 0.507 / 0.506 / 0.505 |

All other p = 0.001 (floor). Paired ΔAUC edit − matched dilation: shift +0.053 [0.050, 0.056] (obs), **+0.013 [0.009, 0.017] (obs8)**, +0.239 (plist); turn5 +0.26 / +0.14 / +0.43; turn10 +0.18 / +0.16 / +0.44; boost +0.165 / +0.163 / +0.272; split0 −0.025 (obs) / +0.369 (plist); split13 −0.195 / +0.246; split21 −0.194 / +0.209; soft −0.206 / +0.130.

Reading: (i) shift and turn are separated through the centroid offset: with the centroid term the AUC is 1.000; without it the shift is as separable as its matched dilation (+0.013), which is itself large (0.943) because the matched dilation is a 25 % mass increase (median $\Delta\log M_2$ = 0.44). The turn stays more separable than its matched dilation without the centroid term (+0.14 to +0.16). (ii) Splits at ranks 13 and 21 and the soft addition are not separated by jet observables (AUC 0.50–0.52); the particle list separates them (0.64–0.76), through two equal-pT particles at an identical position, or an extra low-pT particle. Rotating the hardest particle (null for split0) is more separable in observables (0.936) than splitting it (0.673), since rotating the hardest particle moves the centroid. (iii) The boost is separated (0.88 / 0.79), more than its M2-matched dilation (0.72 / 0.51).

### 2. Headline numbers on all jets and on the on-manifold subsets
Shares of jets kept in the band (indistinguishable edit ≈ 0.5): shift obs 0.000, obs8 0.073, plist 0.000; turn5 0 / 0.164 / 0.065; turn10 0 / 0.018 / 0.007; split0 0.48 / 0.48 / 0.13; split13 0.50 / 0.50 / 0.33; split21 0.51 / 0.50 / 0.38; soft 0.49 / 0.48 / 0.50; boost 0.13 / 0.13 / 0.21.

**Shift** (R vs dil_M2; W→Z among W-called jets; null = rigid rotation):
| jets | n | R | W→Z edit | W→Z null | McNemar edit vs null (Holm) |
|---|---|---|---|---|---|
| all | 8,000 | 1.190 [1.175, 1.206] | 0.809 [0.790, 0.828] | 0.023 [0.016, 0.030] | 1301 / 19, p < 1e-300 |
| obs8 band | 580 | **0.710 [0.670, 0.752]** | 0.935 [0.893, 0.972] | 0.014 [0.000, 0.037] | 128 / 0, 3e-37 |
| obs, plist, both | 0 | – | – | – | – |
Null-rotation R −0.001 (all), −0.012 (obs8). The obs8 subset is class-skewed (76 g, 38 q, 131 W, 159 Z, 176 t).

**Turn** (R vs dil_m; Z→W among Z-called jets; null = rigid rotation):
| edit, jets | n | R | Z→W edit | Z→W null | Z→W dil_m | McNemar edit vs null |
|---|---|---|---|---|---|---|
| turn5 all | 6,386 | −0.011 [−0.024, −0.000] | 0.028 [0.020, 0.037] | 0.025 | 0.372 | 29 / 25, p 0.68 |
| turn5 obs8 band | 1,050 | 0.016 [−0.015, 0.046] | 0.037 [0.014, 0.063] | 0.028 | 0.284 | 7 / 5, p 0.77 |
| turn5 plist band | 412 | 0.020 [−0.022, 0.064] | 0.033 [0.000, 0.071] | 0.011 | 0.385 | 3 / 1, p 0.63 |
| turn5 obs8 + plist | 93 | 0.057 [−0.037, 0.144] | 0.06 | 0.06 | 0.13 | 1 / 1 |
| turn10 all | 5,087 | −0.008 [−0.015, −0.001] | 0.033 [0.024, 0.042] | 0.027 | 0.915 | 31 / 23, p 0.34 |
| turn10 obs8 band | 94 | −0.008 [−0.067, 0.053] | 0.086 | 0.086 | 0.829 | 3 / 3 |
| turn10 plist band | 36 | −0.013 [−0.128, 0.094] | 0 | 0 | 1.0 | – |
obs band: 0 jets for both doses.

**Splits and soft addition** (calls changed, edit vs null; Holm-corrected McNemar):
| edit | all jets | obs band | obs8 band | plist band | both (obs + plist) |
|---|---|---|---|---|---|
| split0 | 4.49 % [4.03, 4.98] vs 1.10 % (n 8,000) | 4.85 % vs 1.07 % (3,815) | 4.79 % vs 1.10 % (3,817) | 5.13 % vs 1.24 % (1,052) | 6.51 % [4.67, 8.49] vs 0.67 % (599; Holm 2e-8) |
| split13 | 2.69 % [2.34, 3.03] vs 0.85 % | 2.82 % vs 0.83 % (3,974) | 2.94 % vs 0.68 % (3,985) | 3.01 % vs 1.07 % (2,628) | 3.35 % [2.40, 4.33] vs 0.99 % (1,313; Holm 6e-5) |
| split21 | 2.76 % [2.42, 3.14] vs 1.03 % | 2.88 % vs 1.11 % (3,885) | 2.75 % vs 1.10 % (3,818) | 2.98 % vs 0.98 % (2,855) | 3.28 % [2.41, 4.14] vs 1.03 % (1,462; Holm 6e-6) |
| soft | 1.89 % [1.58, 2.27] vs 0.24 % | 2.03 % vs 0.33 % (3,054) | 1.82 % vs 0.26 % (3,020) | 1.94 % vs 0.36 % (3,091) | 2.10 % [1.34, 2.90] vs 0.39 % (1,527; Holm 5e-5) |
Every split / soft row: Holm p ≤ 1.1e-3 vs the null edit. q→g vs g→q flows keep their direction on every subset (split0 all 105/9, both 10/0; soft all 34/7, both 12/3).

**Boost** (calls changed; null = rigid rotation of equal pT-weighted displacement):
| jets | n | edit | null | M2-matched dilation | edit − null | McNemar (Holm) |
|---|---|---|---|---|---|---|
| all | 8,000 | 32.7 % [31.7, 33.7] | 2.8 % | 10.0 % | +30.0 pp [28.9, 31.0] | 2473 / 76, p < 1e-300 |
| obs band | 1,045 | 16.6 % [14.2, 18.9] | 2.7 % | 7.7 % | +13.9 pp | 1e-28 |
| obs8 band | 1,056 | 16.7 % | 3.1 % | 7.7 % | +13.5 pp | 3e-28 |
| plist band | 1,687 | 24.5 % [22.4, 26.6] | 2.3 % | 8.8 % | +22.2 pp | 1e-90 |
| both | 509 | 16.9 % [13.7, 20.1] | 2.4 % | 6.7 % | +14.5 pp | 2e-14 |
The obs band is gluon-heavy (507 of 1,045 jets are gluons); boosted gluon jets change call less often. Mean read-out R vs dil_B: −0.108 (all), −0.06 to −0.11 on subsets.

### 3. Natural pairs (observational complement)
- Centroid offset in real jets: $|\vec M_1|^2/(M_0M_2)$ median 3.6e-4, 90th percentile 1.9e-3, 99th 6.0e-3. The mass deficit it causes, $-\tfrac12\log(1-|\vec M_1|^2/(M_0M_2))$: median 0.018 %, 99th percentile 0.30 %, maximum 0.76 % over 14,545 jets. $\log m - \log\sqrt{M_0M_2}$: 1st–99th percentile −0.58 % to +0.43 % (dominated by the small-angle approximation, not by the offset).
- `m_differs` (m differs ≥ 3 % or ≥ 2 % at matched $M_0$, $M_2$, pT, N): **0 pairs**. Such pairs need $|\vec M_1|^2/(M_0M_2)$ ≥ 0.06, 10 times the 99th percentile.
- `M2_differs` (86 pairs, from stacking of the 1 % tolerances, mean $\Delta\log M_2$ 0.032, $\Delta\log m$ 0.009, $\Delta\log M_0$ −0.004, 81 % same class): Δ read-out −0.0027 [−0.020, 0.013] (sign-flip p 0.76); prediction if it follows $\sqrt{M_0M_2}$: 0.0138; if it follows m: 0.0088. The two predictions differ by 0.005, below the CI width; observed − $\sqrt{M_0M_2}$ prediction [−0.034, −0.0005], observed − m prediction [−0.029, 0.005].
- Controls: `all_matched` (2,499 pairs): −0.0014 [−0.0042, 0.0014], p 0.33. `m_only` (7,229 pairs; mean $\Delta\log M_2$ +0.098 with $\Delta\log M_0$ −0.097): −0.0001 [−0.0025, 0.0020], p 0.92; both predictions ≈ 0.0002.
- Pooled regression of Δ read-out on ($\Delta\log M_0$, $\Delta\log M_2$, $\Delta\log m$) over 9,814 pairs, in units of k: −0.44, −0.42, +1.58 (CIs −0.83 to −0.05, −0.81 to −0.04, 0.58 to 2.58); expected (0.5, 0.5, 0) if the read-out follows $\sqrt{M_0M_2}$ and (0, 0, 1) if it follows m. The three regressors are nearly collinear on natural pairs, and the read-out was fit to $\log m$ on natural jets, so it is expected to track m wherever h4 allows; this regression does not test what the network computes.

## Hypotheses and which are excluded
- H_thesis (the edits isolate what the network reads and are not artefacts): predicted AUC ≤ 0.6 for shift, turn and boost, close to the matched dilation; subset R ≤ 0.02 (turn), 0.7–1.3 (shift), split rates 2.8–4.8 %. Observed: AUC ≈ 1.00 (shift, turn), 0.88 (boost); excluded for the AUC part at p 0.001 (each). Subset numbers: turn R 0.016–0.020 (CIs to 0.064), shift R 0.71, split / soft rates reproduced.
- H_artefact (results driven by unnatural jets): predicted AUC near 1 for the key edits, and subset numbers outside the full-sample CIs (turn's R toward 1). Observed: AUC near 1 (met); turn's R on look-real subsets stays at 0.02 [−0.02, 0.06], R toward 1 excluded on those jets; shift R moves from 1.19 to 0.71 (outside the full CI, inside 0.7–1.3); boost rate moves from 33 % to 17–24 % (outside the full CI, still 6–7× the null).
- Natural-data route (pairs separating m from M2): not available in this data set (0 pairs), so neither "the read-out follows m" nor "follows M2" is tested on natural jets.

## Reading (scoped)
- The centroid offset is the observable that makes shifted and turned jets separable: every real jet in 14,545 has $|\vec M_1|^2/(M_0M_2) < 0.016$ (99 % below 0.006), while both edits create offsets the classifier finds on every jet. The turn5 subset that looks real once the centroid term is removed (obs8 band, 1,050 jets) gives R 0.016 [−0.015, 0.046] and a Z→W rate equal to the rotation's (p 0.77): the network's mass read-out does not follow the jet mass on these jets. Those jets are still off-centre, so the test of "reads M2 rather than m" remains a test on off-centre inputs.
- On the jets where the shift looks real apart from its centroid (580 jets, 7 %), the read-out moves 0.71 [0.67, 0.75] of a real mass change of equal $\Delta M_2$, and 94 % of W-called jets go to Z, vs 1.4 % under rotation.
- The split, soft and boost call changes persist on jets the classifier cannot separate from real ones (split and soft at full rate; boost at about half the rate in the observable bands).

## Thesis effect
Refines A4 / A4w: the "M2 about the stored axis, not m" distinction is measured only on off-centre jets, which do not occur in the data (C2ST AUC 1.00 via the centroid term; natural $m$ and $\sqrt{M_0M_2}$ agree to 0.3 % at the 99th percentile). Supports A15 (splits and soft additions reproduce on look-real jets, AUC 0.50–0.52 in observables for ranks 13/21 and soft). Refines A14 (boosted jets are separable, AUC 0.88; the call-change effect remains at 17–24 % vs 2–3 % for the rotation on look-real subsets).

## Explainer draft (for site/explain.py, Paper A tab)
- **Question.** Do the edits behind the mass claims (shift, azimuth turn, collinear split, soft particle, boost) produce jets that look like real jets, and do the results hold on the edited jets that look real?
- **Background.** An input edit can make a jet unlike any jet the network was trained on ("off-manifold"); the network's response to such a jet may come from the jet being unusual rather than from the quantity changed. A classifier two-sample test (C2ST) measures this: a classifier is trained to tell edited from unedited jets, and its AUC ($P(s_E > s_0)$ for a random edited and a random unedited jet) is 0.5 when it cannot.
- **Method.** 8,000 jets (sample 3101). Edits: shift $\Delta\eta\to\Delta\eta+0.06$; turn of the 14 hardest particles in azimuth toward the centroid, $\theta_i\to\theta_i+t\,\mathrm{wrap}(\phi_{M_1}-\theta_i)$, at fixed $p_{T,i}$ and $r_i$ with $m\to0.95m$ or $0.9m$; split at rank $k$ into two halves at the same point; one 0.5 GeV particle; boost $|P|\to0.9|P|$ at fixed $m$. Each is compared with a dilation $(\Delta\eta,\Delta\phi)\to a(\Delta\eta,\Delta\phi)$ matched in size (a real mass change) and a null edit (rotation about the axis). Gradient-boosted classifiers on 9 jet observables (with and without $|\vec M_1|^2/(M_0M_2)$) and on the 30-particle list, out-of-fold scores for every jet; the look-real subset is the edited jets scoring in the central half $[q_{25},q_{75}]$ of the unedited scores; $R=\overline{\Delta\hat\ell}_{\rm edit}/\overline{\Delta\hat\ell}_{\rm dil}$ with $\hat\ell$ the h4 read-out of $\log m$. Natural pairs: unedited jets matched to 1 % in $M_0$, $M_2$, $p_T$, $N$ with $m$ differing by ≥ 3 %.
- **Result (figure `results/jedi/e31_onmanifold.png`).** Shift and turn: AUC 1.00 (0.96 / 0.88–0.98 without the centroid term), as against 0.74–0.95 for matched dilations and 0.49–0.50 for rotations. Split ranks 13/21 and soft: AUC 0.50–0.52 on observables. Boost 0.88. On look-real subsets: turn $R$ 0.016 [−0.015, 0.046]; shift $R$ 0.71 [0.67, 0.75]; split rates 2.7–6.5 % vs 0.7–1.2 % for the null; boost 17–24 % vs 2–3 %. Natural pairs: none exist, since $|\vec M_1|^2/(M_0M_2)\le0.006$ at the 99th percentile, so $m$ and $\sqrt{M_0M_2}$ differ by ≤ 0.3 %.
- **Checks.** Unedited accuracy 80.59 % (sample) / 81.19 % (pool); retrain null AUC 0.49–0.51; rotation nulls 0.49–0.51; turn exactness max $|\Delta r|$ 2e-8 rad; boost $|\Delta\log m|$ ≤ 6e-8.
- **Why it matters.** The mass result (the network reads $M_2$ about the stored axis, not $m$) is measured on off-centre jets, which real data do not contain; on real jets the two quantities coincide, so the distinction describes the network's extrapolation. The IRC-safety and boost results do not depend on off-manifold jets.
- **Conclusion.** As in the Conclusion section above.

## Glossary (C2ST and AUC entries exist, explain.GLOSSARY lines 143–144; additions)
- ('look-real subset (E31)', 'the edited jets whose out-of-fold C2ST score lies in the central half (25th–75th percentile) of the unedited jets\' scores; for an edit the classifier cannot detect it keeps about half of the jets'),
- ('centroid term |M1|²/(M0 M2) (E31)', 'the squared pT-weighted centroid offset from the stored axis relative to M2; below 0.016 in every real jet tested and below 0.006 in 99 %, so m and sqrt(M0 M2) agree to 0.3 % in 99 % of jets; rigid shifts and azimuth turns raise it on every jet'),
- ('obs8 (E31)', 'the 9 jet observables used for C2ST without the centroid term, to test whether the centroid offset is the only observable that separates an edit').

## Issues
- Three cluster nodes stalled at import in CephFS (process in D state, ceph_mdsc_wait_request / folio_wait_bit_common): ucm-fiona01.ucmerced.edu (e31-smoke), k8s-haosu-12.sdsc.optiputer.net (e31-run), rci-tide-gpu-04.sdsu.edu (e31-run3). They were added to the node exclusion list in `job.sh` (comment updated). e31-run4 also waited ~10 min at import on nautilusg01.sci.cwru.edu and then ran normally.
- Debug attempt 1: the planned subset rule (|score − 0.5| below the unedited median) is undefined for a perfectly separating classifier (kept all jets); replaced by the [q25, q75] band. One code fix before that: E28's `soft_add` returns the re-sorted array directly (smoke e31-smoke2).
- Peak pod memory 2,047 MiB at a 2 GiB request (cgroup peak, includes page cache); the job finished.
- The natural-pair arm planned in the node (m differs ≥ 3 % at matched M0, M2) is empty by construction of real jets; reported with the quantiles that show why.
