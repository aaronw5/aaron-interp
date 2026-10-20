# E23x: which centre carries JEDI-linear's direction-dependent shift response, and does it pass through S31 / S58 (2026-10-20)

Claims A4n, A4d, A3c (absorbs E24y's S31 link and its 180° question). Parent: E23wv, E24x / E24xv.

## Conclusion
When every particle of a jet is shifted by 30–60 mrad, the part of the network's response that depends on the direction of the shift follows the change in the plain (unweighted) spread of the particles about the jet axis. It does not follow the pT-weighted spread. The two context units S31 and S58 compute particle-count dipole and quadrupole sums from the jet-level context. Holding them at their unshifted values removes about a quarter of this direction-dependent response in the later layers (h4, h5, z) and none of it in the W-versus-Z score (W−Z logit margin). The thesis that these two units carry most of it (> 50 %) is therefore excluded. Moving only the 14 hardest particles changes the W−Z score about twice as much as moving only the soft particles, the opposite of what the thesis predicted. Still uncertain: which units carry the remaining three quarters and the margin's response. No other S unit responds to the shift as strongly as S31, so the random-unit control is not matched in size.

Terms: *odd response* = $\tfrac12[f(\vec c)-f(-\vec c)]$ for a rigid shift $\vec c$ (the direction-dependent part); *even response* = $\tfrac12[f(\vec c)+f(-\vec c)]-f(0)$; *freeze* = replace a unit by its value on the unshifted jet; *count second moment* = $\sum_i r_i^2$ over real particles (unweighted), against $M_2=\sum_i p_{T,i} r_i^2$.

## Procedure
- **Script and job.**
  - Script: `interp/e23x_centre_jedi.py`; figure: `interp/e23x_plot.py`.
  - Job `e23x-run`, 1 CPU / 2 GiB: wall 263 s, peak 1,812 MiB, 0.87 cores.
  - Smoke runs: `e23x-smoke` was evicted by a node taint (infrastructure, no output). `e23x-smoke2` ran with per 100.
  - Results: `results/jedi/e23x_centre.json`. Logs: `logs/e23x-run.log`, `logs/e23x-smoke2.log`.
- **Model.** Official JEDI-linear 3-feature n64, exact `jedi_core` forward.
  - S-unit freezes are applied by a jitted copy of the d4…d7 head, built from the same fused kernels and input quantizers.
  - Checks: max |logit(head(S)) − logit(full forward)| = 0.0 on the unedited jets. Calls agree on 100 % of 500 shifted jets.
  - The g freeze uses `run_jetwise` with g held at the unshifted jet's g. Check: S31 and S58 change by exactly 0.0 under it, which reproduces E24x.
- **Samples.**
  - Test: `sample(1500, 2311)`, N 7,500, accuracy 80.44 %.
  - Fit: `sample(1500, 2310)` minus 231 rows shared with the test sample, N 7,269, accuracy 80.55 %.
  - Both accuracies are 0.15–0.5 points below the 80.6–81.3 % reference range. That is within about 1 binomial SE (0.46 points at N 7,500) of the 80.9 % full-test value, and the head reproduces the full forward exactly.
- **Units.**
  - Stages: 19 live S (E1 list), 55 live h4, 32 live h5, 16 z (pre-ReLU), and the W−Z logit margin.
  - Each unit is standardised by its sd on the unedited test jets. A stage's odd power is $\sum$ over pairs, jets and units of $O^2$.
- **Edit (arms 1–4).**
  - Rigid translation of every real particle (padding untouched), $|\vec c|$ = 30 and 60 mrad at 8 azimuths, giving 8 mirror pairs.
- **Arms.**
  1. S31 and S58 frozen.
  2. All 64 g units frozen.
  3. S31 only; S58 only.
  4. Control: every pair of the 17 other live S units frozen. That is 136 pairs, all enumerated, since the plan's 200 draws exceed the number of distinct pairs. Also the 17 singles.
     - Matching on raw sd and mean |odd response| was not possible within ×0.5–2: S31's mean |odd ΔS| is 3.10, against at most 0.89 for any other live S unit (S29).
     - The "matched" sets are therefore the nearest 20 pairs and nearest 5 singles. The matched pairs reach only 0.20–0.43 of S31 + S58's response.
  5. Centre-moving edits, on 3,000 random fit jets and 3,000 random test jets, at the same 16 cells:
     - (5a) Move real particles with pT < 11 GeV and pT rank ≥ 14 by $\vec c$. Dilate all other real particles about the axis so that $M_2$ is unchanged exactly.
     - (5b) Move the 14 hardest by $\vec c$. Dilate the rest (ranks ≥ 14) to hold $M_2$.
     - A jet is kept only if the dilation factor satisfies $k^2\in[0.25,4]$ at all 16 cells.
     - Deviation from the plan: 5a rescales the complement of the moved set (the hard 14 plus any rank ≥ 14 particle with pT ≥ 11 GeV), not only the hard 14.
     - Measured per cell: change of the count centroid $\sum_i\Delta\vec x_i/N$ and of the pT centroid $\sum_i p_{T,i}\Delta\vec x_i/M_0$, relative $\Delta M_2$, $q_{\rm count}=\tfrac12\Delta\log\sum r^2$ and $q_{p_T}=\tfrac12\Delta\log M_2$.
     - Regression: per unit OLS without intercept, fitted on the fit jets across the rigid, 5a and 5b edits. Regressors are the odd parts of [$q_{\rm count}$, Δcount centroid] (count model), [$q_{p_T}$, ΔpT centroid] (pT model), or both. Held-out $R^2$ is computed on the test jets.
  6. 180° and 90° rigid rotations about the stored axis, with S58, S31, both, or g frozen. Single-unit null over the 17 other live S units.
  7. Untrained redraws, seeds 0–2, on 1,500 test jets.
- **Statistics.**
  - 95 % intervals: jet Poisson bootstrap, 2,000 resamples.
  - Null test: rank p against the 136-pair (or 17-single) band, and a normal p from the band mean and sd plus the bootstrap sd. The larger of the two is used.
  - Holm over 24 arm × stage tests: 3 freeze arms × 4 stages against the null, 4 tests of H1 > 50 %, and 2 rotations × 4 stages for S58.
  - Calls: exact McNemar on (jet, cell) pairs at 60 mrad, plus a jet-bootstrap CI of the difference in calls changed.
- **Hypotheses.**
  - H0: the S31 + S58 freeze removes no more odd power than the random-pair band.
  - H1: it removes > 50 % at h4, h5, z and the margin.

## Numbers
**Unfrozen response (test, N 7,500; standardised RMS).**
- Odd: S 0.37, h4 0.36, h5 0.35, z 0.26, margin 0.28.
- Even: 0.50 / 0.59 / 0.68 / 0.67 / 1.15.
- Mean |ΔS|/sd under the shift: S31 1.20, S58 0.51.

**Arms 1–3: odd power removed by the freeze, $1-P(O_{\rm frz})/P(O)$, pooled over both rings (95 % CI).**

| freeze | S | h4 | h5 | z | W−Z margin |
|---|---|---|---|---|---|
| S31 + S58 | 0.511 [0.506, 0.516] | **0.255** [0.249, 0.261] | **0.249** [0.245, 0.254] | **0.208** [0.201, 0.215] | **−0.009** [−0.016, −0.003] |
| S31 only | 0.452 | 0.208 [0.202, 0.214] | 0.229 | 0.200 | 0.005 [−0.001, 0.011] |
| S58 only | 0.059 | 0.040 [0.038, 0.043] | 0.015 | 0.017 | −0.012 |
| all g | 0.514 | 0.259 [0.253, 0.265] | 0.244 | 0.208 | 0.038 [0.028, 0.047] |
| null, 136 pairs: mean [2.5, 97.5 %], max | 0.058 [0.006, 0.126], 0.133 | 0.039 [−0.136, 0.183], 0.235 | 0.015 [−0.241, 0.199], 0.255 | 0.018 [−0.208, 0.234], 0.294 | −0.052 [−0.344, 0.173], 0.216 |
| null, 20 nearest pairs: [2.5, 97.5 %] | [0.025, 0.097] | [−0.037, 0.184] | [−0.194, 0.226] | [−0.180, 0.199] | [−0.209, 0.196] |
| null, 17 singles: [2.5, 97.5 %] | [0.000, 0.069] | [−0.110, 0.121] | [−0.221, 0.121] | [−0.182, 0.158] | [−0.212, 0.119] |

- **Path share.** $P(O-O_{\rm frz})/P(O)$, the response that differs when S31 + S58 are frozen: h4 0.43, h5 0.30, z 0.27, margin 0.09. The share removed is smaller than the path share because frozen and unfrozen parts partly cancel.
- **By ring (S31 + S58).**
  - 30 mrad: 0.248 / 0.236 / 0.213 / −0.007 (h4 / h5 / z / margin).
  - 60 mrad: 0.258 / 0.255 / 0.206 / −0.012.
- **All-g vs S31 + S58.** Freezing all of g removes the same share as freezing S31 + S58 at h4–z (difference ≤ 0.006). At the margin it removes 0.038 against −0.009.
- **Tests against the null (S31 + S58).**
  - h4: rank p 0.007, z = 2.7. h5: 0.015. z: 0.058. margin: 0.39.
  - After Holm (24 tests): 0.18, 0.75, 1, 1.
  - S31 alone against the singles: rank p 0.056 at h4–z (the minimum possible with 17 units), Holm 1. S58 alone: rank p 0.44–0.56.
- **H1 (> 50 %).** Excluded at every downstream stage: bootstrap P(share > 0.5) = 0 at h4, h5, z and margin.
- **Even response removed (prediction iii: |change| ≤ 10 %).**
  - S31 + S58: S +0.364, h4 **+0.201** [0.196, 0.207], h5 +0.063, z +0.048, margin −0.019.
  - S58 only: ≤ 0.09 at every stage. All g: h4 +0.216, z +0.086, margin −0.044.
- **Calls at 60 mrad (8 azimuths × 7,500 jets).**

| | calls changed vs unedited | W → Z (true W called Z) | McNemar (n10 / n01, p) | diff in calls changed [CI] |
|---|---|---|---|---|
| unfrozen | 59.9 % | 60.2 % (unedited 2.3 %) | – | – |
| S31 + S58 frozen | 56.7 % | 62.9 % | 3,157 / 1,209, p 1e−197 | 3.2 [2.9, 3.6] points |
| S31 frozen | 56.9 % | 62.1 % | 2,919 / 1,130, p 9e−180 | 3.0 [2.6, 3.3] |
| S58 frozen | 59.6 % | 61.1 % | 611 / 400, p 3e−11 | 0.35 [0.24, 0.46] |
| all g frozen | 56.1 % | 70.0 % | 3,217 / 959, p 6e−282 | 3.8 [3.4, 4.1] |

**Arm 5: centre-moving edits** (test jets; fit jets agree within 0.01).

| edit | jets kept | Δcount centroid / c (median) | ΔpT centroid / c | odd $q_{\rm count}$ RMS (%) | odd $q_{p_T}$ RMS (%) | max \|ΔM2\|/M2 |
|---|---|---|---|---|---|---|
| rigid | 3,000 (100 %) | 1.00 | 1.00 | 7.14 | 0.81 | – (not held) |
| 5a soft | 2,409 (80 %) | 0.56 | 0.10 | 5.59 | 5e−7 | 8e−8 |
| 5b hard 14 | 685 (23 %) | 0.25 | 0.71 | 8.41 | 3e−7 | 5e−8 |

- 5b is feasible only on jets where the particles outside the hard 14 carry enough $M_2$ to absorb the hard-core shift. The 5a vs 5b comparison below uses the 665 test jets feasible for both.
- **Odd RMS at 60 mrad, 5a / 5b / rigid:**
  - S 0.345 / 0.434 / 0.580; h4 0.299 / 0.470 / 0.569; z 0.161 / 0.320 / 0.351; margin 0.080 / 0.177 / 0.118.
  - Ratio 5a/5b: h4 0.64 [0.62, 0.66], z 0.50 [0.48, 0.52], **margin 0.45 [0.42, 0.49]**. P(5a > 5b) = 0 in the bootstrap.
- **S31 + S58 freeze share under each edit (h4 / h5 / z / margin):**
  - rigid 0.256 / 0.251 / 0.208 / −0.002;
  - **5a 0.502** [0.489, 0.514] / 0.297 / 0.238 / −0.075;
  - **5b −0.026** / −0.023 / −0.032 / −0.048.
- **Held-out $R^2$ of the odd response** (test jets, all three edits; fitted on the fit jets):

| model | S | h4 | h5 | z | margin |
|---|---|---|---|---|---|
| count moment $q_{\rm count}$ only | 0.376 [0.365, 0.386] | 0.176 | 0.143 | 0.154 | 0.001 |
| pT moment $q_{p_T}$ only | 0.002 | 0.001 | 0.001 | 0.003 | 0.001 |
| count ($q_{\rm count}$ + Δcount centroid) | 0.386 | 0.185 | 0.152 | 0.168 | 0.003 |
| pT ($q_{p_T}$ + ΔpT centroid) | 0.012 | 0.009 | 0.011 | 0.018 | 0.004 |
| both | 0.388 | 0.187 | 0.154 | 0.173 | 0.005 |
| centroid displacement only (no moment terms) | 0.012 | 0.009 | 0.010 | 0.015 | 0.004 |

- Count − pT difference in $R^2$: S 0.374 [0.363, 0.384], h4 0.175 [0.168, 0.183], h5 0.141, z 0.150 (all p < 0.001), margin −0.001 [−0.003, 0.000].
- Rigid edits only: count moment 0.41 / 0.22 / 0.17 / 0.20 at S / h4 / h5 / z, which reproduces E23wv's 0.16–0.35. pT moment ≤ 0.005.
- **Gains in the joint model** (stage RMS coefficient, sd per 1 % of odd $q$; fit-jet bootstrap):

| stage | $q_{\rm count}$ | $q_{p_T}$ | ratio |
|---|---|---|---|
| S | 0.036 | 0.026 | 1.37 [1.21, 1.48] |
| h4 | 0.024 | 0.024 | 0.99 [0.86, 1.08] |
| h5 | 0.020 | 0.024 | 0.83 [0.71, 0.93] |
| z | 0.015 | 0.028 | 0.54 [0.48, 0.60] |
| margin | 0.0016 | 0.0104 | 0.16 [0.09, 0.34] |

  Under a rigid shift the odd $q_{\rm count}$ varies 8.8× more than $q_{p_T}$ (RMS 7.1 % vs 0.8 %). Per unit of moment change, the two gains are similar at h4 and the pT moment's gain is 2–6× larger at z and the margin.
- **Count-moment slope per edit, relative to the rigid edit** (same jets; h4 / h5 / z):
  - 5a 0.62 [0.60, 0.64] / 0.43 / 0.45;
  - 5b 0.43 [0.39, 0.46] / 0.51 / 0.41;
  - margin undetermined (CI spans ±100).
  - A single function of the count moment would give 1. The response per unit of count-moment change depends on which particles move.

**Arm 6: rotations about the stored axis** (N 7,500; share of the response's power removed; S58 single-unit null rank p over 17 units).

| edit | RMS h4 / z / margin | calls changed | S58 frozen h4 / h5 / z / margin | S31 frozen h4 / z / margin | S31 + S58 h4 / z / margin | all g h4 / z / margin |
|---|---|---|---|---|---|---|
| 180° | 0.21 / 0.14 / 0.21 | 3.73 % | 0.145 [0.127, 0.165] / 0.082 / 0.052 / 0.010 (rank p 0.11, 0.11, 0.22, 0.39) | 0.203 / 0.163 / 0.038 | 0.349 / 0.212 / 0.042 | 0.358 / 0.236 / 0.117 |
| 90° | 0.27 / 0.16 / 0.24 | 4.48 % | 0.301 [0.285, 0.319] / 0.160 / 0.089 / 0.041 (rank p 0.056, 0.056, 0.11, 0.28) | 0.132 / 0.119 / 0.029 | 0.433 / 0.211 / 0.071 | 0.449 / 0.247 / 0.164 |

- 180° calls, unfrozen vs frozen:
  - S58 frozen: 3.67 %, McNemar 17 / 12, p 0.46.
  - S31 + S58 frozen: 3.19 %, 70 / 29, p 5e−5.
  - All g frozen: 2.93 %, 90 / 30, p 4e−8.
- Single-unit null band at 180° (h4): mean 0.008, max 0.159. None of the 8 S58 rotation tests passes Holm.

**Arm 7: untrained redraws** (1,500 jets).
- Seed 0: 1 live S unit (S58); no shift arm (fewer than 2 live S units).
- Seed 1: S58 and S60 live, S31 dead. Freezing S58 removes 0.76 / 0.97 / 0.97 / 0.94 (h4 / h5 / z / margin).
- Seed 2: S30 and S31 live, S58 dead. Freezing S31 removes 0.65 / 0.55 / 0.30 / −0.02.
- The share is not constant across redraws. With 1–2 live S units, any live unit carries most of the response, so the untrained network does not give a usable null for this share.

## Hypotheses
- **H0 (S31 + S58 freeze inside the random band).**
  - Not rejected after Holm at any stage. The h4 raw rank p is 0.007, above the band maximum (0.255 vs 0.235), with Holm p 0.18.
  - The control is weak in one direction: no other S unit has an odd response as large as S31 (the nearest pairs reach 20–43 % of it). A response-matched null would need random directions in S scaled to S31 + S58's response, which this node did not run.
- **H1 (> 50 % at h4–z and the margin).** Excluded: 0.21–0.26 at h4–z with upper CI ≤ 0.27, and −0.009 at the margin.
- **Competing hypothesis (i), odd term through S31 / S58.**
  - Its first prediction, that the freeze removes most of the odd response, is excluded.
  - Its second prediction, that 5a moves the margin more than 5b at matched $M_2$, is excluded: 5a/5b = 0.45 [0.42, 0.49].
- **Competing hypothesis (ii), odd term in the per-particle radial objects.**
  - Its prediction that the freeze removes no more than the random band is not excluded after Holm, but S31 + S58 remove 21–26 % at h4–z. That is a defined share, and the all-g freeze removes the same share. So the whole context round's contribution to the odd response at h4–z passes through S31 + S58.
  - Its prediction that 5b ≥ 5a holds (margin ratio 0.45, h4 0.64).
  - The remaining 74–79 % at h4–z, and the margin's odd response, do not pass through g or S31 / S58.
- **Prediction (iii), even response unchanged by the freeze (±10 %).**
  - Holds at h5, z and the margin for every freeze (S31 + S58: +0.063, +0.048, −0.019).
  - Fails at h4: +0.20 for S31 + S58, +0.22 for all g.
- **Which centre.**
  - At S–z, the odd response follows the odd change of the unweighted second moment $\sum r^2$: held-out $R^2$ 0.15–0.38, against ≤ 0.02 for the pT-weighted $M_2$ and the pT centroid. Count − pT is 0.14–0.37 with p < 0.001.
  - At the margin, neither centre explains the odd response ($R^2$ ≤ 0.005).
  - The count-moment advantage comes mainly from the size of the change: a rigid shift changes $\sum r^2$ about 9× more (relative) than $M_2$, while the per-unit gains are similar at h4 and larger for $M_2$ at z and the margin.
  - The count-moment slope differs between soft-only, hard-only and rigid moves (0.4–0.6 of rigid). A function of the count moment alone does not describe the response.
- **E24y 180° question.** Freezing S58 removes 14.5 % of the 180°-rotation response at h4 and 1 % at the margin, inside the single-unit range (rank p ≥ 0.11). S31 removes 20 % at h4 at 180°. Most of the 180° response downstream of S does not pass through S58.

## Corrected reading
- **What S31 + S58 carry.** The count dipole and count quadrupole context units carry about a quarter of the direction-dependent shift response at h4–z (0.21–0.26 of odd power; S31 0.20–0.23, S58 ≤ 0.04). That is the whole share carried by the context round.
- **What they do not carry.** They carry none of it at the W−Z margin (−0.009 [−0.016, −0.003]), and they do not carry the share that moves calls most. Freezing them lowers calls changed at 60 mrad by 3.2 points (59.9 → 56.7 %) and raises W → Z from 60.2 to 62.9 %.
- **Soft vs hard moves.** Moving only the soft particles (pT < 11 GeV) at fixed $M_2$ produces an odd response that is half carried by S31 + S58 at h4 (0.50). Moving only the hard 14 produces a larger odd response, 2.2× larger at the margin, and none of it passes through S31 + S58.
- **Centre.** The odd response at S–z follows the unweighted second moment rather than $M_2$, mainly because a rigid shift changes the unweighted moment much more. Its gain differs by which particles are moved.

## Explainer entry (for site/explain.py, Paper A, after E23w verify)
```python
E23X_EXPLAINER = [
 ('Question', 'When all particles of a jet are shifted, which part of the jet does the direction-dependent part of the network\'s response follow, and does it pass through the two context-only units S31 (count dipole) and S58 (count quadrupole)?', None),
 ('Background', 'For a rigid shift $\\vec c$ of every particle, the odd response $O=\\tfrac12[f(\\vec c)-f(-\\vec c)]$ is the part that depends on the shift direction; the even response $E=\\tfrac12[f(\\vec c)+f(-\\vec c)]-f(0)$ does not. '
  'To first order, the odd change of the pT-weighted moment $M_2=\\sum_i p_{T,i}r_i^2$ is $2\\vec c\\cdot\\vec M_1$, and that of the unweighted moment $\\sum_i r_i^2$ is $2\\vec c\\cdot\\sum_i\\vec x_i$. '
  'E23wv found that the odd response follows the unweighted centroid; E24x found that S31 and S58 read only the context round g and have thresholds set by count dipole and quadrupole sums.', None),
 ('Method', 'Test sample(1500, 2311), $N$ 7,500; shifts of 30 and 60 mrad at 8 azimuths. Freeze S31 + S58 (or S31, S58, or all of g) at the unshifted values and measure the odd power removed, $1-\\sum O_{\\rm frz}^2/\\sum O^2$, at h4, h5, z and the W−Z margin; null: all 136 pairs of the other live S units. '
  'Centre-moving edits at fixed $M_2$: (5a) move only soft particles ($p_T<11$ GeV) and dilate the rest to hold $M_2$; (5b) move only the 14 hardest and dilate the rest. Regress the odd response on the odd parts of $\\tfrac12\\Delta\\log\\sum r^2$ and $\\tfrac12\\Delta\\log M_2$ (fit sample(1500, 2310), held-out $R^2$ on the test jets). Also 180° / 90° rotations with S58 frozen.', None),
 ('Result', 'Freezing S31 + S58 removes 0.25 (h4), 0.25 (h5), 0.21 (z) and −0.01 (margin) of the odd power; freezing all of g removes the same at h4–z. The random-pair band reaches up to 0.24–0.29. '
  'The unweighted moment explains 15–38 % of the odd response at S–z, the pT-weighted moment ≤ 2 %; neither explains the margin. '
  'Moving only the hard 14 changes the W−Z margin 2.2× more than moving only the soft particles; S31 + S58 carry half of the soft-move response at h4 and none of the hard-move response. '
  'Freezing S58 removes 15 % of the 180° rotation response at h4.', 'e23x_centre.png'),
 ('Checks', 'The jitted head reproduces the full forward exactly (max logit difference 0). Holding g leaves S31 and S58 exactly unchanged. $M_2$ is held to $10^{-7}$ in 5a / 5b. The fit and test samples are disjoint. The untrained redraws have only 1–2 live S units, so they give no usable null for the share.', None),
 ('Why it matters', 'The context round (g → S31, S58) carries a defined quarter of the direction-dependent shift response, not most of it, and none at the decision score. The rest, including what moves the W−Z margin, comes from S units that read the particles, most of all when the hard core moves.', None),
]
```
Conclusion text for site/conclusions.py: the "Conclusion" paragraph above.

## Files
- `interp/e23x_centre_jedi.py`, `interp/e23x_plot.py`
- `results/jedi/e23x_centre.json`, `results/jedi/e23x_centre.png` (smoke: `results/jedi/e23x_centre_smoke.json`)
- `logs/e23x-run.log`, `logs/e23x-smoke2.log`

## Issues / limitations
- Random-unit control not matched in response size: no other S unit's odd response is within ×0.5 of S31's. A random-direction-in-S control scaled to S31 + S58's response would be the matched null.
- 5b is feasible on 23 % of jets, a subset with wide soft radiation. The 5a vs 5b comparison uses 665 jointly feasible test jets.
- The plan asked for 200 random pairs; only 136 distinct pairs exist.
- The untrained reference is degenerate: 1–2 live S units, and S31 and S58 are never both live.
- Test accuracy is 80.44 %, slightly below the 80.6–81.3 % reference range (sample variation; the head check is exact).
