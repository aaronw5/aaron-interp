# R1: statistics on the mass and jet-axis results (2026-10-17)

User questions:
- "When you shift the axis, do you also get the same mass shift you would expect?"
- "Always do hypothesis testing to see how likely it is that it's just luck."

## Runs and files
- **Script:** `interp/r1_mass_stats.py`. The docstring lists every definition.
- **Jobs:**
  - `anrunw-ai-r1-jedi`: CPU, 1 CPU / 2 GiB, 263 s, peak 1,136 MiB.
  - `anrunw-ai-r1-part`: GPU, TITAN Xp, 341 s.
  - `anrunw-ai-r1-plot` and `anrunw-ai-r1-plot2`: figures.
  - Smoke runs (cluster only): `r1-jedi-smoke` (100 jets/class), `r1-part-smoke` (30 jets/class, CPU).
- **Results:**
  - `results/stats/r1_mass_axis.json`: merged, Holm over all tests.
  - `results/stats/r1_mass_axis_{jedi,part}.json`.
  - `results/site/T13_mass_axis_stats.png`.
  - `results/site/F7_mass_axis_calibrated.png`. Another file, `F7_e8_conjunctions.png`, already uses the F7 prefix.
  - Logs: `logs/r1-{jedi,part}_cluster.log`.
  - Cluster copies are under `/j-jepa-vol/aaron-interp/{results/stats,results/site,logs}`.

## Procedure
**Samples**
- JEDI-linear (3-feature, n64, fixed point): 20,000 hls4ml test jets, 4,000 per class, seed 51.
- ParT_full and ParT_kin: 10,000 JetClass test jets, rows 8,000–9,000 of each class's first test file.
- Neither sample overlaps E6, E6v, E17, P1 or P1v.

**Read-out:** one ridge regression per model, from the last hidden layer to $\log m$.
- JEDI: $z$, 16 units. Test-half $R^2$ 0.941; $\alpha = 10^{-6}$, the lower edge of the grid, i.e. effectively unregularised.
- ParT_full: class token, 128 units. $R^2$ 0.890, $\alpha = 3\times10^{-4}$.
- ParT_kin: class token, 128 units. $R^2$ 0.927, $\alpha = 10^{-3}$.
- Fitted on a random half of the unedited jets (split seed 51); every $\Delta\log\hat m$ is a mean over the other half (JEDI 10,000 jets, ParT 5,000).

**Edits.** None changes the invariant mass. The true $\Delta\log m$ is at most $1.1\times10^{-7}$ for JEDI, and exactly 0 for ParT, whose 4-vectors are unchanged.
- **Rigid shift:** every real particle's $(\Delta\eta,\Delta\phi)$ moved by $\delta \in \{0.02, 0.03, 0.045, 0.06\}$ along $\eta$ or along $\phi$.
  - ParT: deta or dphi is shifted and deltaR is recomputed; the pair features are unchanged.
- **Recentred axes:** E-scheme, WTA, and hardest particle. Median displacement:
  - JEDI: 0.0015, 0.040 and 0.041 rad.
  - ParT: 0.0003, 0.095 and 0.102 rad.
- **Null edit:** every particle rotated by $90^\circ$ about the axis. $M_0$, $M_2$ and all the alternative quantities are unchanged exactly. The true $\Delta\log m$ (mean) is $-1.2\times10^{-4}$ for JEDI and $-1.4\times10^{-3}$ for ParT (4-vectors rebuilt).

**Calibration.** Angles are scaled by $k$, which is a real mass change. $k$ is solved so that the mean true $\Delta\log m$ equals each edit's mean prediction under each hypothesis; $k = 1.05$ and $1.1$ are added. This gives 46 scalings per model, each matched to within $10^{-6}$. The calibration factor is

$$c = \frac{\overline{\Delta\log\hat m}}{\overline{\Delta\log m}}.$$

Measured values:
- JEDI: 0.80–0.92 over $\Delta\log m \in [-0.13, 0.67]$.
- ParT_full: 0.81–0.85.
- ParT_kin: 0.75–0.89.

**Hypotheses.** Each predicts a per-jet change computed about the new axis:
- H0a, invariant mass: 0.
- H0b, $\tfrac12\Delta\log(M_0M_2)$.
- G1: $\Delta\log\sum p_T r$.
- $\sqrt{Q_2}$: $\tfrac12\Delta\log\sum p_T^2 r^2$. This is E6v's alternative, written $\sqrt{\Sigma r^2}$ in T2.
- $\sqrt{\Sigma r^2}$: unweighted, over real particles.

**Tests**
- Each hypothesis expects a read-out change $E_h = c_h\,\bar P_h$, with $c_h$ taken from its own matched calibration edit. The test is mean $\Delta\log\hat m = E_h$.
- Calibrated ratio $= \text{obs}/(c\,\bar P_{M_0M_2})$.
- Bootstrap: 2,000 resamples of the test-half jets, shared across all edits (paired). The bootstrap p has a floor of $1/2001$. Holm correction runs over 165 tests (5 hypotheses × 11 edits × 3 models) on the normal-approximation p.
- Per-jet least-squares slope and $R^2$ of the read-out change on each hypothesis' per-jet prediction.
- **Calls:** W → Z and calls changed with Wilson 95 % intervals. Exact McNemar tests, paired by jet: edit vs null edit, and edit vs the matched real mass change.

## Results: observed / predicted, calibrated ratio [95 % CI]
| edit | JEDI-linear | ParT_full | ParT_kin |
|---|---|---|---|
| shift η .02 / .03 / .045 / .06 | 0.56 [.54,.57] / 0.70 [.69,.71] / 0.90 [.88,.91] / 1.02 [1.00,1.03] | 0.46 [.35,.59] / 0.64 [.57,.72] / 0.76 [.71,.81] / 0.83 [.79,.88] | 2.58 [2.22,2.99] / 2.13 [1.95,2.32] / 2.00 [1.89,2.12] / 1.98 [1.90,2.06] |
| shift φ .02 / .03 / .045 / .06 | 0.61 [.59,.63] / 0.74 [.72,.75] / 0.92 [.90,.93] / 1.03 [1.01,1.04] | 0.80 [.73,.87] / 0.81 [.76,.86] / 0.84 [.80,.88] / 0.88 [.84,.91] | 1.12 [1.02,1.21] / 1.25 [1.17,1.32] / 1.42 [1.36,1.48] / 1.53 [1.47,1.58] |
| E-scheme (pred −.0004 / +.0002 / +.0002) | 1.20 [0.43,1.97] | 1.23 [0.95,1.65] | 0.17 [0.07,0.74] |
| WTA | 0.92 [.90,.94] | 1.32 [1.29,1.36] | 0.78 [.75,.81] |
| hardest particle | 0.97 [.95,.99] | 1.50 [1.46,1.54] | 0.85 [.82,.88] |
| null: rotate 90°, $\Delta\log\hat m$ | −0.0001 [−.0009,+.0007], p 0.77 | +0.0216 [.0203,.0231], z 30 | +0.0247 [.0234,.0261], z 36 |

**Raw ratios (observed / predicted, no calibration)**

| | rigid shifts | WTA | hardest |
|---|---|---|---|
| JEDI | 0.51–0.87 | 0.81 | 0.85 |
| ParT_full | 0.38–0.72 | 1.08 | 1.22 |
| ParT_kin | 0.95–2.25 | 0.60 | 0.63 |

These differ from the earlier P1/E17 values because the jets and the read-out are different.

## Which hypotheses are excluded (Holm-adjusted $p < 0.01$)
**H0a (invariant mass)**
- Excluded for every rigid shift and for the WTA and hardest-particle axes, in all three models: $|z| \ge 8$, Holm $p \le 10^{-14}$.
- E-scheme recentring: excluded for ParT_full ($+0.0002$, $z = 10$) and ParT_kin (Holm $p$ 0.008). Not excluded for JEDI (Holm $p$ 0.04).

**H0b ($\sqrt{M_0M_2}$): not excluded in these rows**
- JEDI shift η 0.06: ratio 1.02, raw $p$ 0.014.
- JEDI hardest-particle axis: ratio 0.97, raw $p$ 0.002, Holm $p$ 0.02.
- ParT_kin shift φ 0.02: ratio 1.12, raw $p$ 0.014.
- All three E-scheme rows.

**H0b: excluded in these rows**
- Every other edit.
- This includes JEDI shift φ 0.06 (1.03 [1.01, 1.04], Holm $p$ $10^{-3}$) and JEDI WTA (0.92, Holm $p$ $10^{-13}$).

**G1 and $\sqrt{Q_2}$**
- Excluded for every non-E-scheme edit of JEDI and ParT_full. They predict 1.7–4.3 × the $\sqrt{M_0M_2}$ change under shifts.
- They predict decreases under the WTA and hardest axes (JEDI G1 −0.13 and −0.10; ParT −0.11 and −0.05), while the read-out rises.
- ParT_kin shifts: G1 is not excluded at η .02, .03, .045 and φ .06. $\sqrt{Q_2}$ is not excluded at η .06.
- Both are excluded for ParT_kin WTA and hardest (Holm $p < 10^{-140}$).

**$\sqrt{\Sigma r^2}$ (unweighted)**
- Excluded everywhere except the JEDI and ParT_kin E-scheme rows and ParT_full η 0.02.
- Closest at the JEDI WTA and hardest axes: differences +0.005 and +0.009, Holm $p$ $3\times10^{-4}$ and $3\times10^{-10}$.

## Per-jet regression on the $\sqrt{M_0M_2}$ prediction (slope [CI], $R^2$)
- **JEDI shifts:** slope rises with $\delta$, from 0.29 [.27, .30] at $R^2$ 0.21 (δ = .02) to 0.78–0.80 at $R^2$ 0.73 (δ = .06).
- **JEDI axes:** WTA 0.93 [.91, .95], $R^2$ 0.47. Hardest 1.00 [.96, 1.03], $R^2$ 0.56.
- **ParT_full shifts:** 0.05–0.16, $R^2 \le 0.07$. Axes: WTA 1.06, $R^2$ 0.41; hardest 1.31, $R^2$ 0.66.
- **ParT_kin shifts:** 0.14–0.46, $R^2 \le 0.21$. Axes: WTA 0.30, $R^2$ 0.06; hardest 0.43, $R^2$ 0.19.
- In ParT_kin shifts, G1 has the highest per-jet $R^2$ (0.21–0.23 at δ = .06). In JEDI shifts, G1 and $\sqrt{Q_2}$ have $R^2$ equal to or above $\sqrt{M_0M_2}$, at slopes 0.06–0.37. The per-jet $R^2$ does not separate the hypotheses as clearly as the mean does.

## Calls (all jets; W → Z [Wilson CI]; McNemar exact)

**Unedited, and under the null edit (rotation)**

| model | W → Z unedited | W → Z null | calls changed by the null |
|---|---|---|---|
| JEDI | 0.026 | 0.026 | 4.0 % |
| ParT_full | 0.116 | 0.148 | 6.7 % |
| ParT_kin | 0.164 | 0.187 | 9.0 % |

**JEDI**
- Shift η .02 / .03 / .045 / .06: W → Z 0.067 / 0.258 / 0.845 / 0.768, vs null McNemar $p$ $10^{-37}$ to $10^{-953}$ (φ shifts: $10^{-33}$ to $10^{-963}$).
- WTA 0.581 [.565, .596]; hardest 0.504 [.488, .519].
- Matched real mass change of the same predicted size: 0.278 / 0.783 / 0.781 / 0.124; WTA 0.806; hardest 0.840.
- At 0.06 rad the mean read-out change equals the real-change one (ratio 1.02), but the calls differ: W → Z 0.77 vs 0.12. Under the real +36 % mass change W jets move past Z (McNemar $p$ $10^{-716}$).

**ParT_full**
- Shifts give W → Z 0.12–0.33.
  - δ ≤ 0.03: not above the null (0.121–0.154 vs 0.148; McNemar $p$ 0.002–0.6, fewer W → Z than the null at η .02).
  - δ ≥ 0.045: above the null ($p \le 6\times10^{-4}$).
- Shifts equal the matched real change: e.g. φ .06 gives 0.326 vs 0.316, $p \ge 0.2$ for every shift.
- WTA 0.348 and hardest 0.338: below the matched real change (0.489, 0.490; $p \le 2\times10^{-18}$).

**ParT_kin**
- Shifts give W → Z 0.19–0.36, within ±0.04 of the matched real change ($p$ 0.004–0.5).
- WTA 0.386 and hardest 0.362, vs matched 0.531 and 0.533 ($p$ $10^{-18}$).

## Measurements that qualify the ratios
1. **The ParT rotation null is not neutral for the read-out.**
   - The read-out moves +0.022 (full) and +0.025 (kin), with true $\Delta\log m$ −0.0016 and $M_2$ unchanged.
   - The rotation changes 6.7 % and 9.0 % of calls.
   - Against this null, the ParT read-out changes for δ ≤ 0.03 are smaller than or equal to the null's own change. For ParT_full η 0.045, the difference from the null is −0.0009 ($p$ 0.3).
   - JEDI's null is at 0.
2. **ParT η vs φ shifts.**
   - About the stored axis, $\sum p_T\,\Delta\eta \ne 0$ on average, so $\tfrac12\Delta\log(M_0M_2)$ is lower for +η than for +φ: 0.060 vs 0.073 at 0.06.
   - ParT_kin's read-out moves about equally under the two (0.096, 0.090), which gives calibrated ratios of 2.0 (η) and 1.5 (φ).
   - ParT_full moves less under η (0.040 vs 0.052).
3. **E-scheme rows.** The predicted change is $2\text{–}4\times10^{-4}$, so the matched calibration is noisy: $c$ 1.03, 0.91, 2.6, with wide intervals. The ratios in these rows carry no information beyond "near 0".
4. **JEDI shift dose-response.** The calibrated ratio rises with $\delta$ (0.56 → 1.02), as in E6v (raw 0.46–0.79 at h5). At 0.06 rad, 0.01 % of jets have a quantizer wrap.

## Answer to the user's question (scoped to these measurements)
With one read-out per model, calibrated on real mass changes of the same predicted size:
- **Invariant mass (no change) is excluded** for every axis displacement of 0.02 rad or more in all three networks.
- **JEDI-linear:**
  - Moves the read-out by the full $\sqrt{M_0M_2}$ amount for 0.06-rad shifts (1.02–1.03) and the hardest-particle axis (0.97).
  - Moves it by 0.92 for the WTA axis.
  - Moves it by only 0.56–0.74 for 0.02–0.03-rad shifts.
- **ParT_full:**
  - Moves it by 0.46–0.88 of the predicted amount for rigid shifts.
  - Moves it by 1.32–1.50 for the WTA and hardest-particle axes.
- **ParT_kin:**
  - Moves it by 1.1–2.6 for rigid shifts.
  - Moves it by 0.78–0.85 for the two axes.
- **Exact agreement with the predicted amount** (H0b not excluded at Holm $p < 0.01$) holds only for JEDI at 0.06 rad η, the JEDI hardest-particle axis, and ParT_kin φ 0.02.
- **Every other displacement is excluded**, with deviations from 3 % to 160 %. Most of them are statistically far from chance (Holm $p$ from $10^{-3}$ to $10^{-300}$).
- **The alternatives G1 and $\sqrt{Q_2}$** predict mass decreases under the WTA and hardest axes and are excluded there in every model.
- **Equal mean read-out shifts do not imply equal calls.** JEDI at 0.06 rad changes W → Z to 0.77 where the matched real mass change gives 0.12. ParT's axis edits give fewer W → Z than the matched real change.
- **ParT's read-out also moves (+0.02) under a 90° rotation** that changes neither the invariant mass nor $M_2$.
