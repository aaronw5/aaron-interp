# E31 verify: fresh-sample rerun of the on-manifold check, plus three controls

Script `interp/e31_verify_jedi.py` (job e31-verify, 1 CPU / 2 GiB; wall 1,604 s, about 390 s of it import; peak pod memory 2,048 MiB, at the request; 0.69 cores on average). Results `results/jedi/e31_verify.json`, log `logs/e31-verify.log`. The script reuses `build_edits`, `featsets`, `c2st` and `dauc` from `interp/e31_onmanifold_jedi.py` unchanged.

## Conclusion
On a fresh sample, the E31 numbers reproduce: shifted and turned jets are separated from real jets only through the offset between the jet's pT-weighted centre and its stored axis, and no real jet has an offset that changes the mass by more than 0.7 %. Three statements in the E31 reading are not supported by the controls added here. First, the shift is not separable on observables that do not depend on where the axis is (classifier AUC 0.52, where 0.5 means no separation). The shift moves every particle by the same amount in pseudorapidity, which is a boost along the beam: the jet is physically unchanged and only its stored axis is displaced. Second, the drop of the shift's mass response from about 1.2 to about 0.7 real-mass-change units on the "look-real" subset comes mostly from which jets that subset contains (class, width, pT), not from those jets looking real. Third, boosted jets lie inside the region occupied by real jets: a nearest-neighbour test puts 1.4 % of them outside it, against 1.0 % of real jets. The classifier separates the boosted sample because its pT is 10 % lower, and the lower call-change rate on its look-real subset matches the rate of composition-matched jets outside that subset.

## Definitions (used below)
- Stored axis: the jet direction from which each particle's $(\Delta\eta, \Delta\phi)$ is measured in the input. $r_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$.
- $M_0=\sum p_T$, $\vec M_1=\sum p_T(\Delta\eta,\Delta\phi)$, $M_2=\sum p_T r^2$. Centroid term $|\vec M_1|^2/(M_0M_2)$: 0 when the pT-weighted centre is on the axis. For massless particles at small angles $m^2\approx M_0M_2-|\vec M_1|^2$.
- C2ST (classifier two-sample test): a gradient-boosted classifier (sklearn HistGradientBoosting, 200 trees) is trained to tell edited from unedited jets; AUC = probability that a random edited jet scores higher than a random unedited jet (0.5 = no separation, 1 = full separation). Out-of-fold scores, two swapped half-assignments, as in E31.
- Feature sets: `obs` = $m$, jet pT, particle count N, $\tau_{21}$, $\tau_{32}$ (N-subjettiness ratios), $D_2$ (energy-correlation ratio), girth $\sum p_T r/M_0$, $p_T^D=\sqrt{\sum p_T^2}/M_0$, centroid term. `obs8` = obs without the centroid term (girth is still measured about the stored axis). `plist` = $(p_T,\Delta\eta,\Delta\phi)$ of the 30 hardest particles. **New:** `inv` = $m$, jet pT, N, $\tau_{21}$, $\tau_{32}$, $D_2$, $p_T^D$: observables that do not depend on where the stored axis is.
- Look-real band: edited jets whose score lies inside the central half $[q_{25},q_{75}]$ of the unedited jets' scores (E31's rule).
- R = mean change of the h4 mass read-out (ridge fit of $\log m$ on the 64 h4 neurons) under the edit, divided by its mean change under the matched real change (a dilation $(\Delta\eta,\Delta\phi)\to a(\Delta\eta,\Delta\phi)$ of the same size). R = 1: the network responds as to a real mass change; R = 0: no response.
- kNN support test (new): each jet's features are log-transformed and standardised on unedited reference jets; a jet's distance = mean distance to its 10 nearest reference jets; "outside the support" = distance above the 99th percentile of held-out unedited jets (so 1 % of real jets are outside by construction). Unlike the C2ST, this test is not failed by a change that only moves jets to another region where real jets exist (e.g. heavier jets).
- Composition-matched complement (new): the jets not in the band, reweighted to the band's mix of strata (true class × quintile of unedited $\log M_2$ × tercile of unedited jet pT; 75 strata, coverage ≥ 99.5 % of band jets). If the band value equals this value, the band changes the number only through which jets it selects.

## Procedure
- Network: official JEDI-linear 3-feature n64, exact forward pass (`interp.jedi_core.Jedi`).
- Sample: `sample(per_class=1200, seed=4101)` = 6,000 jets (E31: `sample(1600, 3101)`); unedited accuracy 80.90 %. Edits from E31's `build_edits` with rng 4104: shift ($\Delta\eta+0.06$ rad on every particle), turn5 (14 hardest particles turned in azimuth at fixed pT and r so $m$ falls 5 %; feasible on 4,801 jets), split13 (rank-13 particle split into two halves at the same point), soft (one 0.5 GeV particle added), boost ($|P|\times0.9$ at fixed mass). Matched dilations: dil_M2, dil_m5, dil5, dil_B. Null edits: rigid rotation about the axis with the same pT-weighted displacement (shift, turn, boost), split particle rotated (split13), 0.5 GeV added to the hardest particle (soft).
- C2ST on obs / obs8 / plist / inv; 300 bootstrap resamples, 300 within-jet swaps (p floor 0.0033). H0: AUC = 0.5.
- kNN support test on obs / obs8 / inv: reference = unedited half A (3,000 jets), queries = half B (edited, dilated, null-edited, unedited).
- Band subsets and composition-matched complement for shift, turn5, split13, soft, boost; McNemar (exact) edit vs null on the band.
- Natural offsets on a fresh pool `sample(2000, 4102)` = 10,000 unedited jets.

## Results

### 1. Reproduction (E31 → verify)
| edit | obs AUC | obs8 AUC | plist AUC | matched dilation obs8 | ΔAUC vs dilation, obs8 |
|---|---|---|---|---|---|
| shift | 1.000 → 1.000 | 0.956 → 0.952 | 0.999 → 0.999 | 0.943 → 0.940 | +0.013 → +0.013 [0.007, 0.018] |
| turn5 | 1.000 → 1.000 | 0.883 → 0.865 | 0.942 → 0.932 | 0.741 → 0.719 | +0.142 → +0.147 |
| split13 | 0.503 → 0.518 | 0.508 → 0.518 (p 0.01) | 0.757 → 0.755 | 0.698 → 0.680 | −0.19 → −0.16 |
| soft | 0.518 → 0.529 | 0.521 → 0.532 (p 0.003) | 0.641 → 0.572 | 0.698 → 0.680 | −0.20 → −0.17 |
| boost | 0.881 → 0.882 | 0.879 → 0.880 | 0.785 → 0.785 | 0.716 → 0.703 | +0.163 → +0.177 |
Retrain nulls (unedited vs unedited) 0.500–0.505; rotation nulls 0.49–0.51.

Band headline numbers (E31 → verify): shift obs8 band kept 7.3 % → 8.0 %, R 0.71 → 0.72 [0.68, 0.77] (n 480), W→Z 0.94 → 0.89 vs null 0.01 (McNemar 93 / 1, p 1e-26). turn5 all jets R −0.011 → 0.007 [−0.007, 0.019]; obs8 band R 0.016 → 0.022 [−0.015, 0.056] (n 901); Z→W edit 0.065 vs null 0.071 (McNemar 7 / 8, p 1.0). split13 calls changed 2.7 % → 2.9 % (all), 2.8–3.6 % on bands vs null 1.0–1.2 %. soft 1.9 % → 1.8 % vs 0.15 %. boost 32.7 % → 33.2 % (all), 16.6–24.5 % → 18.9–26.9 % on bands, null 2.5–3.7 %.

Natural offsets (10,000 fresh jets): centroid term median 3.6e-4, 99th percentile 5.7e-3, maximum 0.014; mass deficit from the offset 0.29 % at the 99th percentile, 0.71 % at most (E31: 0.30 %, 0.76 %). Every shifted jet (median centroid term 0.35) and every turned jet (median 0.098) is above the natural maximum.

### 2. Axis-free observables (`inv`)
Under the shift, every `inv` feature is unchanged to float precision (99th percentile of relative change ≤ 2e-7; girth changes by up to a factor 8.5, the centroid term by up to 6.6e5). C2ST on inv:
| edit | inv AUC | matched dilation | null |
|---|---|---|---|
| shift | **0.520 [0.507, 0.534]** (p 0.003) | 0.933 | 0.507 |
| turn5 | 0.776 [0.766, 0.785] | 0.724 | 0.488 |
| split13 | 0.515 | 0.682 | 0.538 |
| soft | 0.534 | 0.682 | 0.513 |
| boost | 0.881 [0.874, 0.886] | 0.699 | 0.506 |
The obs8 separation of the shift (0.952) comes from girth measured about the stored axis, which is another function of the axis displacement. A rigid $\Delta\eta$ translation is a longitudinal boost, so the shifted jet is the same physical jet with a displaced axis.

### 3. Support test (share of half-B jets outside the 99 % real-jet region; unedited = 1.0 %)
| edit | obs (with centroid term) | obs8 | inv | matched dilation obs / obs8 / inv |
|---|---|---|---|---|
| shift | **87.1 % [85.9, 88.2]** | 9.1 % [8.1, 10.1] | 1.0 % | 1.9 / 1.8 / 1.6 % |
| turn5 | **73.3 % [71.7, 74.9]** | 0.8 % | 0.9 % | 1.0 / 1.0 / 1.0 % |
| split13 | 1.1 % | 1.0 % | 1.0 % | 1.1 / 1.0 / 1.0 % |
| soft | 1.1 % | 0.9 % | 0.9 % | 1.1 / 1.0 / 1.0 % |
| boost | 1.4 % [1.0, 1.8] | 1.1 % | 1.1 % | 1.1 / 1.0 / 1.0 % |
Null edits: 0.8–1.0 %. Reading: shift and turn leave the real-jet region only through the centroid term (and, for the shift, girth about the axis); without it the turned jets are inside the region (0.8 %). The boost and the matched dilations move jets within the region; their C2ST AUCs (0.70–0.94) measure a change of distribution (heavier or lower-pT jets), not jets unlike any real jet. The thesis-note criterion "AUC ≤ 0.6 for shift, turn and boost" is also failed by every matched real change (0.70–0.94), so it does not separate off-manifold edits from real changes.

### 4. Band selection vs composition
| edit, band | band value | complement, unweighted | complement, composition-matched |
|---|---|---|---|
| shift R, obs8 band (n 480) | 0.72 [0.68, 0.77] | 1.21 | **0.89** |
| shift R, inv band (n 2,990) | **1.15 [1.13, 1.18]** | 1.21 | 1.14 |
| shift W→Z, obs8 band | 0.89 | 0.80 | 0.80 |
| turn5 R, obs8 band (n 901) | 0.022 [−0.015, 0.056] | 0.003 | 0.019 |
| boost calls changed, obs band (n 791) | 19.2 % | 35.3 % | **20.7 %** |
| boost calls changed, inv band (n 819) | 19.0 % | 35.4 % | 21.4 % |
| boost calls changed, plist band (n 1,345) | 26.9 % | 35.0 % | 31.1 % |
| split13 calls changed, obs band | 2.8 % | – | 3.0 % |
| soft calls changed, obs band | 1.5 % | – | 2.3 % |
Shift: of the drop from 1.21 (complement) to 0.72 (obs8 band), the class / $M_2$ / pT mix accounts for 0.32 (to 0.89); the rest comes from selecting jets whose girth about the axis stays typical after the shift. On the jets that look real in axis-free observables (half the sample), R is 1.15. Boost: the band is gluon-heavy (403 of 791 jets); composition-matched jets outside the band change call at 20.7 %, close to the band's 19.2 %.

### 5. Script check (code reading)
- Padding: all edits and features mask on pT > 0; the shift and dilations leave padded slots at 0. No padded slot counted as a particle.
- Split edit: 22.4 % of jets have 64 particles; for those `split_ext` pushes the softest particle out of the 64 slots, so the split also removes one particle and $M_0$, $\vec M_1$, $M_2$ and the four-momentum change by that particle. Not relevant to the E31 claims; worth stating in A15.
- C2ST: out-of-fold scores, halves disjoint, no jet scored by a classifier that saw it. Band uses pooled $s_0$ quantiles across two classifiers (calibration may differ slightly between them; no effect on the conclusions above).
- Natural pairs `M2_differs` (E31 JSON): observed − $\sqrt{M_0M_2}$ prediction CI [−0.034, −0.0005] excludes 0, observed − m prediction [−0.029, 0.005] does not. The report's "cannot tell the two apart" is not exact; since the read-out was fit to $\log m$ on natural jets, the arm does not test the network either way (as E31 states).
- Figure (`results/jedi/e31_onmanifold.png`): readable; the AUC axis caption "0.5 = edited jets look like real jets" treats any AUC above 0.5 as off-manifold, while the matched dilations (real changes) reach 0.70–0.94. Suggest adding the `inv` row and the support-test shares.

## Hypotheses
- H_off (shift / turn are off the real-jet region only through the axis offset): predicts inv / obs8 support shares ≈ 1 % and obs support share ≫ 1 %. Observed: turn5 73 % (obs) vs 0.8 % (obs8); shift 87 % (obs), 9 % (obs8, girth about the axis), 1.0 % (inv). Consistent.
- H_lookreal (the shift's look-real R of 0.71 reflects look-real status): predicts the composition-matched complement R ≈ 1.2 and the inv-band R ≈ 0.7. Observed 0.89 and 1.15; excluded (inv-band CI [1.13, 1.18]).
- H_boost_off (boosted jets are unlike real jets): predicts support share ≫ 1 %. Observed 1.4 % [1.0, 1.8]; excluded. H_boost_comp (band rate set by composition): predicts complement-matched ≈ band. Observed 20.7 % vs 19.2 %; consistent.

## Corrected reading
Shifted and turned jets leave the region of real jets only through quantities measured about the stored axis (centroid term, girth about the axis, particle coordinates): with the centroid term 87 % / 73 % of them fall outside the 99 % real-jet region, against 1 % of real jets, and on axis-free observables the shift is not separable (AUC 0.52). Real jets have centroid offsets that change m by at most 0.7 %, so m and $\sqrt{M_0M_2}$ cannot be separated with natural jets (reproduced). The turn's near-zero mass response holds on every subset (R 0.02 [−0.02, 0.06]). The shift's response on its obs8 "look-real" band (0.72) is mostly a composition effect (composition-matched complement 0.89; axis-free band 1.15) and does not show weaker response on realistic jets. Splits at rank 13 and soft additions are inside the real-jet region and their call-change rates hold. Boosted jets are inside the real-jet region (1.4 % outside); their C2ST separation is the 10 % pT shift, and the band call-change rate (19–27 %) matches composition-matched jets outside the band (21–31 %), against 2.5–3.7 % for the rotation null.

## Explainer draft (verify step; for site/explain.py, Paper A, after E31)
- **Question.** Is the E31 "look-real" check measuring whether edited jets are unlike real jets, or only whether the sample changed?
- **Background.** A classifier two-sample test separates any two different distributions; a sample of slightly heavier real jets is also separable. Being unlike any real jet is a property of single jets (support), not of the sample.
- **Method.** Fresh 6,000 jets. (i) C2ST on axis-free observables $\{m, p_T, N, \tau_{21}, \tau_{32}, D_2, p_T^D\}$. (ii) For each jet, $d=\frac{1}{10}\sum_{j\in\mathrm{10NN}}\lVert z-z_j\rVert$ in standardised $\log|f|$ space against 3,000 real jets; outside if $d>q_{0.99}(d_{\rm real})$. (iii) Band numbers compared with the jets outside the band reweighted to the band's (class, $\log M_2$, $p_T$) mix.
- **Result.** Shift: AUC 0.52 axis-free, 87 % outside with the centroid term, 1 % without axis-dependent features. Turn: 73 % outside with the centroid term, 0.8 % without. Boost: 1.4 % outside. Shift R on the obs8 band 0.72 vs 0.89 for composition-matched jets and 1.15 on the axis-free band.
- **Checks.** Unedited jets 1.0 % outside (by construction); dilations 1.0–1.9 %; null rotations 0.8–1.0 %; retrain-null AUC 0.50.
- **Why it matters.** The off-manifold part of the mass edits is exactly the axis offset that separates m from $\sqrt{M_0M_2}$; the boost and the IRC edits stay among real jets.
- **Conclusion.** As in the Conclusion above.

Glossary additions: kNN support test; axis-free observables (`inv`); composition-matched complement.
