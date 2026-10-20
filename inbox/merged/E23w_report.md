# E23w: functional form of JEDI-linear's mass read-out on natural off-centre jets (two gains, odd term) and the product test (E21w) (2026-10-20)

Claims A4 (mass variable, product of M0 and M2), A9 (stage where the product forms).

## Conclusion
We moved every particle of naturally off-centre jets sideways by 10–60 mrad and compared each network stage's response with its response to a real mass change of the predicted size. The part of the response that follows the jet's centroid offset (the "odd" term, linear in the shift) enters at about half to three quarters of the weight of the part quadratic in the shift (a1/a2 0.4–0.8). This holds already at the second-round sums S and at every later stage, on two independent samples. So the unequal weighting is present where the moments are first summed; it is not created in the h4/h5 layers, and its direction is the opposite of the E23v probe result (odd term 1.3–1.8 times the even term), which used a different read-out and smaller shifts. A test for a unit computing the product M0·M2 (slope 1 in a mixed-derivative test) finds no z unit or W−Z margin outside the band of 200 random unit pairs. That band is wide (±10–20), so this test can only exclude large effects. The h4/h5 units that do fall outside the band have slopes of about −30 to +30, not 1.

## Procedure
- **Script:** `interp/e23w_gains_jedi.py` (module docstring has the full plan). Figure: `interp/e23w_plot.py`, rendered locally → `results/jedi/e23w_gains.png`.
- **Jobs:**
  - `e23w-run`: `python -m interp.e23w_gains_jedi`, 1 CPU / 2 GiB, wall 761 s, 0.27 cores on average, peak 1,960 MiB.
  - `e23w-verify`: `--seed 2312 --exclude 2311 --tag verify`, wall 761 s, peak 2,047 MiB. Peak memory is at the 2 GiB limit (cgroup peak includes file cache from the test archive; no OOM kill). A rerun should request 1 CPU / 3 GiB or reduce `--nprod`.
  - Results: `results/jedi/e23w_gains.json`, `e23w_gains_verify.json`. Logs: `logs/e23w-run.log`, `logs/e23w-verify.log`. Earlier smoke run (`e23w-smoke`, per 500): `e23w_gains_smoke_500.json`.
- **Model:** official JEDI-linear 3-feature n64 (`jedi_core.Jedi(64)`). Units: 19 live S units (E1 list), 55 h4 and 32 h5 units with nonzero sd, all 16 z, and the W−Z logit margin (123 units in total).
- **Sample:**
  - Main: `sample(5000, 2311)`, N 25,000, unedited accuracy 80.70 % (expected band 80.6–81.3 %).
  - Replicate: `sample(5000, 2312)` with the 2,394 jets also drawn by seed 2311 removed, N 22,606, accuracy 80.59 % (0.01 points below the band; binomial 95 % ±0.5 points).
- **Moments** about the stored axis, real particles only: $M_0=\sum p_T$, $\vec M_1=\sum p_T(\Delta\eta,\Delta\phi)$, $M_2=\sum p_T r^2$, centroid offset $d=|\vec M_1|/M_0$.
- **Sets:**
  - Tail: top 5 % of $d$ within each class. Main: 1,250 jets (250 per class), $d$ median 7.4 mrad (p10–p90 3.0–14.0), accuracy 67.9 %. Replicate: 1,133 jets, $d$ median 7.5 mrad.
  - Bulk: 2,000 random jets with $d$ below the median, $d$ median 0.52 mrad.
  - Product set: 2,000 random jets.
- **Arm 1 edit (gains):**
  - Rigid translation $\vec c$ of every real particle; pT, slot order and padding unchanged. $|\vec c|\in\{0.01,0.02,0.04,0.06\}$ rad at 8 azimuths: 32 cells plus the unedited jet.
  - Prediction: $\tfrac12\Delta\log(M_0M_2)=\tfrac12\log\big(1+(2\vec c\cdot\vec M_1+|\vec c|^2M_0)/M_2\big)$. Measured vs actual: max deviation 3e-8; $M_0$ unchanged exactly.
  - Matched real change: the same jet dilated about the stored axis by $e^{\rm pred}$ (angles × $e^{\rm pred}$), which changes $m$ and $\sqrt{M_0M_2}$ by the same amount.
- **Arm 1 gains:**
  - Each unit's response is standardised by its sd on the sample. For each $(\vec c,-\vec c)$ pair, the odd part is $O=\tfrac12(\Delta(\vec c)-\Delta(-\vec c))$ and the even part is $E=\tfrac12(\Delta(\vec c)+\Delta(-\vec c))$, for the translation (T) and its matched real change (C).
  - $a_1$ (odd, centroid-linear gain) = coefficient of $O_C$ in the regression of $O_T$ on $[O_C, c_\eta, c_\phi]$. The two $c$ terms absorb any odd response along a fixed direction that does not follow $\vec M_1$ (e.g. half-plane cuts).
  - $a_2$ (even gain) = $\sum E_TE_C/\sum E_C^2$.
  - Pooled over the 16 pairs, the jets and the units of a stage.
  - Secondary (the plan's raw form): per jet, the response is regressed on $x_1=\vec c\cdot\vec M_1/M_0$ and $x_2=|\vec c|^2$ over the 32 cells, separately for T and for C. Reported as $(b_1/b_2)_T/(b_1/b_2)_C$.
- **Arm 1 controls:**
  - (i) Reflected tail jets, $(\Delta\eta,\Delta\phi)\to-(\Delta\eta,\Delta\phi)$. The odd response of the reflected jets is projected on the original jets' $O_C$ ("cross" gain); a term linear in $\vec M_1$ predicts cross ≈ −a1.
  - (ii) Bulk jets (small $\vec M_1$): the RMS of the M1-linear odd term must shrink.
  - (iii) Rotation about the stored axis with the same mean particle displacement as each ring ($M_2$ unchanged, prediction 0).
  - (iv) Untrained network: 3 redraws per sample with ≥ 2 live S units.
- **Arm 1 statistics:**
  - Jet bootstrap with Poisson weights, 1,000 draws.
  - Location test. H0: $a_1/a_2=1$ at a stage. H1: $a_1/a_2\ne1$. Normal p from the bootstrap sd, Holm over the 5 stages. The location is the first stage (S → margin) with Holm p < 0.05.
  - Competing predictions for $a_1/a_2$:
    - recentring-invariant mass $m$: $a_1=a_2=0$;
    - a single function of $M_0M_2$ about the stored axis: 1;
    - E23v probe values: odd 1.34–1.82, even 0.62–0.71, ratio about 1.9–2.9.
- **Arm 2 (product, E21w) edits**, step DL = 0.05 in log:
  - M2 edit: angles × $e^{DL/2}$.
  - M0 edit: pT of the 14 hardest real particles rescaled so that $M_0\times e^{DL}$, then a dilation that restores $M_2$.
  - Measured: $|\Delta\log M_0-{\rm target}|\le4$e-8; $|\Delta\log M_2-{\rm target}|\le1.1$e-7.
- **Arm 2 statistic:**
  - Mixed slope = $[\log|\sum_j g_2^{(+)}|-\log|\sum_j g_2^{(-)}|]/(2\,DL)$, where $g_2^{(\pm)}=U(\pm M_0,+M_2)-U(\pm M_0,-M_2)$, summed over jets.
  - A unit linear in $M_0M_2$ gives 1; a unit additive in $M_0$, $M_2$ (or in their logs) gives 0. CI from a 300-draw jet bootstrap.
  - The per-jet median slope is kept as a secondary statistic (see Issues).
- **Arm 2 null:**
  - 200 random ordered pairs (A, B) of live S units of the trained network. Each pair is patched at S, A with the S-space step size of the M0 edit and B with that of the M2 edit (median standardised S step: M0 edit 0.61 sd, M2 edit 0.27 sd). The same statistic is computed for each pair.
  - Per unit: z = (network − null mean)/null sd, normal p, BH and Holm over the 123 units; rank p (floor 1/201).
  - Untrained redraws: input-edit slope per stage.
  - H0: every unit inside the random-pair band. H1: some unit outside, with slope near 1.
- **Reference networks** (gain pattern only): Deep Sets d64 seed 0 (pooled latent and margin) and the X12 HGQ-150 retrain seed 0 (S, z, margin).

## Numbers
Main sample (seed 2311) / replicate (seed 2312). Brackets are 95 % jet-bootstrap intervals.

**Arm 1: stage gains on tail jets, all rings**

| stage | a1 (odd) | a2 (even) | a1/a2 | Holm p (ratio = 1) | raw (b1/b2)_T/(b1/b2)_C |
|---|---|---|---|---|---|
| S | 0.32 [0.23, 0.42] / 0.42 [0.32, 0.52] | 0.62 / 0.62 | 0.51 [0.37, 0.68] / 0.67 [0.52, 0.83] | 4e-9 / 8e-5 | 0.49 [0.40, 0.59] / 0.60 [0.48, 0.72] |
| h4 | 0.29 / 0.38 | 0.70 / 0.62 | 0.42 [0.33, 0.51] / 0.61 [0.50, 0.74] | 4e-33 / 3e-9 | 0.54 / 0.66 |
| h5 | 0.41 / 0.47 | 0.77 / 0.77 | 0.53 [0.45, 0.60] / 0.61 [0.53, 0.69] | 2e-32 / 3e-22 | 0.70 [0.52, 0.91] / 0.82 [0.67, 0.99] |
| z (16) | 0.66 / 0.71 | 0.92 / 0.91 | 0.71 [0.65, 0.77] / 0.78 [0.71, 0.84] | 1e-18 / 2e-11 | 1.01 [0.90, 1.12] / 1.00 [0.90, 1.09] |
| W−Z margin | 0.77 [0.65, 0.90] / 0.70 [0.58, 0.83] | 0.97 / 0.95 | 0.79 [0.66, 0.93] / 0.74 [0.61, 0.87] | 1.4e-3 / 8e-5 | 2.84 [1.69, 5.54] / 1.66 [0.91, 2.95] |

- **Location:** the first stage with Holm p < 0.05 for $a_1/a_2=1$ is S in both samples. Every stage has $a_1/a_2<1$.
- **Competing hypotheses:**
  - Recentring-invariant $m$ ($a_1=a_2=0$) is excluded at every stage: the a2 intervals are ≥ 0.53.
  - Single gain ($a_1/a_2=1$) is excluded at every stage on the odd/even estimator.
  - The E23v ratio of about 2 is excluded at every stage on both samples.
- **Per ring:** $a_1/a_2$ rises with $|\vec c|$. At z: 0.50, 0.75, 0.77, 0.72 (main) for 10, 20, 40, 60 mrad. At S: 0.19, 0.37, 0.57, 0.55.
- **Per unit:** 60 / 46 of 123 units have Holm p < 0.05 for ratio ≠ 1. Of these, 7 / 3 are above 1 (all S units, e.g. S13, S43, S60 on both samples); the rest are below.
- **Raw regression vs odd/even split:** the raw form is ≈ 1 at z and above 1 at the margin (main 2.84, CI excludes 1; replicate 1.66, CI includes 1). The two estimators therefore disagree from z onward. The raw form does not remove the fixed-direction odd response (amplitude 0.010–0.036 sd, comparable to the M1-linear amplitude 0.037–0.078 sd), and it takes a ratio of ratios.

**Arm 1 controls (main / replicate)**
- **Reflected jets:**
  - a1 measured with the reflected jets' own $\vec M_1$: S 0.32 / 0.47, z 0.65 / 0.72, margin 0.72 / 0.89.
  - Cross gain on the original jets' $O_C$: S −0.29 [−0.38, −0.20] / −0.42; h4 −0.25 / −0.39; h5 −0.39 / −0.48; z −0.63 [−0.69, −0.57] / −0.73; margin −0.67 [−0.81, −0.53] / −0.86.
  - The odd term flips sign with $\vec M_1$, with magnitude ≈ a1.
- **Bulk jets ($d$ median 0.52 mrad):** RMS of the M1-linear odd term, tail → bulk:
  - S 0.047 → 0.0033; h4 0.040 → 0.0046; h5 0.037 → 0.0045; z 0.052 → 0.0058; margin 0.078 → 0.022 (main). Replicate: same within 0.004.
  - Bulk a2 at S 0.12 and h4 0.29, against 0.62 and 0.70 on the tail.
- **Rotation of matched displacement** (pred 0), RMS standardised response rotation / translation:
  - at 10 mrad: S 0.15 / 0.18, z 0.11 / 0.14, margin 0.14 / 0.17;
  - at 60 mrad: S 0.30 / 0.81, z 0.21 / 0.87, margin 0.20 / 1.35.
  - At the smallest shift, a rotation that leaves $M_2$ unchanged moves the units almost as much as the translation.
- **Untrained redraws** (3 per sample, 2–3 live S units): a2 0.12–0.83 (mostly ≈ 0.5); a1 −0.20 to +0.86; a1/a2 −0.76 to +1.25. The trained values lie inside the untrained spread at most stages (figure, grey crosses).
- **Reference networks:**
  - Deep Sets d64 margin: a1/a2 0.96 [0.86, 1.07] / 0.82 [0.72, 0.93].
  - X12 HGQ-150 seed 0 (accuracy on sample 76.7 %): z 0.41 [0.24, 0.60] / 0.49; margin 0.47 [0.39, 0.56] / 0.58 [0.49, 0.67].

**Arm 2: product test** (N 2,000; null 200 random S pairs, median 811 / 875 jets per pair with S ≥ 0 after the patch)
- **z units and margin:**
  - Pooled slope, main: z0 1.41 [1.13, 1.67], z7 −3.40 [−4.11, −2.70], z10 −14.5 [−31.2, −8.0], z14 −0.06 [−0.71, 0.52], margin −0.60 [−1.46, 0.21]. Replicate: margin −1.11 [−2.08, −0.08], z7 −3.60.
  - Random-pair 95 % bands are about ±10 to ±20 for every z unit and the margin (margin: −22 to +22).
  - BH q = 1 for every z unit and the margin on the main sample, and q ≥ 0.57 on the replicate. Minimum rank p over all units 0.010.
- **h4/h5 units with BH q < 0.05:** 10 (main) and 12 (replicate). Six are common to both: h442, h444, h457, h463, h56, h525.
  - Their slopes are 12 to 20 (h442, h457, h56, h525), −2 to −4 (h444) and −27 to −32 (h463).
  - These units are active on 5–48 % of jets. Slopes of this size come from units crossing their ReLU threshold under the M0 edit, not from a product.
  - One unit near 1 with q < 0.05: h451, replicate only, 0.96 [0.62, 1.33] (not significant on the main sample).
  - Units near 1 above the band in both samples: none.
- **Secondary per-jet median slope:** S, h4, h5 medians and their nulls are 0.0 (exact ties, see Issues). z medians −1.15 to +1.41; margin 0.25 / 0.36; no normal p < 0.05 among z units.
- **S-level patch of S0 with S50** (the E21v pair): per-jet median slope z7 −0.11, margin −0.80; pooled z7 −3.75, margin −0.18. E21v's gradient-ratio slope (z7 0.94, margin 0.64) used a different statistic (regression of log(g50/g0) on log(S0/S50)) and is not comparable one-to-one.
- **Untrained redraws:** median pooled slope per stage −6.9 to +2.0; no stage near 1 in a consistent way.

## Reading (scoped to what was measured)
- Over 10–60 mrad translations of the 5 % most off-centre jets, each stage's response splits into an even part (quadratic in $\vec c$) and an M1-linear odd part. The even part has a gain of 0.62 at S, rising to 0.92–0.97 at z and the margin. The odd part has a gain of 0.3–0.4 at S and h4, rising to 0.7–0.8 at z and the margin.
- The odd part is linear in $\vec M_1$: it flips sign under reflection with matched magnitude, and its amplitude drops 8–14× on bulk jets at S to z.
- The odd/even ratio is below 1 from S onward. The departure from a single function of $M_0M_2$ is therefore measurable at the first stage where the moments are summed. It is not introduced at the S → h4 step, and the ratio moves toward 1 (0.4–0.6 → 0.7–0.8) between h4 and the margin.
- The plan's expected pattern ($a_1>a_2$ from h4 onward) is not found with this edit and these unit read-outs. The alternative "gains equal across stages" is partly consistent: the ratio's CIs overlap between S and the margin on both samples. The E23v values (odd 1.34–1.82) came from a ridge probe of log m on z, with recentring at $|\vec c|=d$ (≈ 7 mrad). Here the read-outs are the units themselves and $|\vec c|$ is 10–60 mrad. At the smallest ring the ratio is lowest, and a rotation moves the units nearly as much as the translation. The difference between the two results is therefore not resolved by this run.
- Untrained redraws give a2 ≈ 0.5 and a1 near 0 in most stages. That even-part agreement with a matched dilation appears without training, so a2 by itself is not specific to a mass computation.
- Product test: no z unit and not the margin departs from the random-S-pair band, so H0 is not rejected for z and the margin. The band is wide relative to the network's CIs, so the test excludes only large departures. The h4/h5 units that depart do so with slopes far from 1.

## Issues
- **Debug change from the smoke run (1 attempt):** per-jet finite differences of the quantized network tie exactly ($g_2^{(+)}=g_2^{(-)}$) on most jets. This pins per-jet medians and their nulls at 0 in S, h4 and h5. The primary statistic was changed to the jet-pooled slope; the median is kept as secondary.
- **Unequal jet sets:** the null for each pair is computed on jets with S ≥ 0 after the patch (median 811 jets), the network on all 2,000.
- **Pooled-slope instability:** when $\sum g_2^{(-)}$ is small, the pooled slope becomes large (null bands ±10–20; network values up to ±30 on rarely active units). A test with fewer degrees of freedom would narrow it, e.g. a regression of $\log|g_2|$ on $\log M_0$ over several M0 steps.
- **Replicate accuracy** 80.59 %, 0.01 points below the 80.6 % band edge (within binomial error).
- **Memory:** peak at the 2 GiB limit.
- **No site edits:** `site/explain.py` and `site/conclusions.py` have uncommitted edits by other workers, so the entries below are left for the ledger step.

## Explainer entry (for site/explain.py, PAPER A; figure results/jedi/e23w_gains.png)
```python
E23W_EXPLAINER = [
 ('Question',
  'How does JEDI-linear weigh the two parts of the change in $M_2$ under a rigid shift of a naturally off-centre jet: the part linear in the centroid offset, $2\\vec c\\cdot\\vec M_1$, and the part quadratic in the shift, $|\\vec c|^2M_0$? '
  'At which layer does any imbalance appear? And does any unit combine $M_0$ and $M_2$ as a product?', None),
 ('Background',
  'About the stored axis, a shift $\\vec c$ of every particle changes $M_2=\\sum p_T r^2$ by $2\\vec c\\cdot\\vec M_1+|\\vec c|^2M_0$, with $\\vec M_1=\\sum p_T(\\Delta\\eta,\\Delta\\phi)$, and leaves $M_0=\\sum p_T$ unchanged. '
  'A network that reads a single function of $M_0M_2$ responds to both parts with the same gain. A network that reads the invariant mass $m$ responds to neither, since a shift is a boost plus a rotation. '
  'Splitting the response to $+\\vec c$ and $-\\vec c$ into odd and even parts separates the two: the odd part is linear in $\\vec c\\cdot\\vec M_1$ and the even part quadratic in $|\\vec c|$. '
  'A unit $U=f(M_0M_2)$ that is linear in the product has mixed slope $\\partial\\log(\\partial U/\\partial\\log M_2)/\\partial\\log M_0=1$; a unit additive in $M_0$ and $M_2$ has slope 0.', None),
 ('Method',
  'Tail jets: the 5 % most off-centre per class (centroid offset $d=|\\vec M_1|/M_0$ median 7.4 mrad), 1,250 jets from sample(5000, 2311) and 1,133 from a disjoint replicate (seed 2312). '
  'Edits: translations with $|\\vec c|$ = 10, 20, 40, 60 mrad at 8 azimuths. Each is compared with a matched real change, a dilation by $e^{\\delta}$ with $\\delta=\\tfrac12\\Delta\\log(M_0M_2)$ predicted for that shift. '
  'Per unit (19 S, 55 h4, 32 h5, 16 z, W$-$Z margin; standardised): $a_1$ is the regression of the translation\'s odd part on the matched change\'s odd part, with fixed-direction terms in $c_\\eta, c_\\phi$ removed; $a_2$ is the same for the even parts. Jet bootstrap, Holm over stages. '
  'Controls: reflected jets, bulk jets ($d$ 0.5 mrad), a rotation of matched displacement, 3 untrained redraws. '
  'Product test: separate 5 % edits of $M_0$ (pT of the 14 hardest particles, $M_2$ restored) and $M_2$ (dilation) on 2,000 jets; jet-pooled mixed slope per unit; null of 200 random pairs of S units patched with the same step sizes; BH over 123 units.', None),
 ('Result',
  'The ratio $a_1/a_2$ is 0.51 [0.37, 0.68] at S, 0.42 at h4, 0.53 at h5, 0.71 at z and 0.79 [0.66, 0.93] at the W$-$Z margin. The replicate gives 0.67, 0.61, 0.61, 0.78 and 0.74. Every stage is below 1 (Holm $p\\le 1.4\\times10^{-3}$), and S is already below 1. '
  'The even gain rises from 0.62 at S to 0.97 at the margin; neither gain is 0, so the invariant mass is excluded as the read-out. '
  'Product test: no z unit and not the margin lies outside the random-pair band (BH $q\\ge0.57$), but that band spans about $\\pm$10 to $\\pm$20. The h4/h5 units outside the band have slopes of $-32$ to $+20$, not 1.', 'results/jedi/e23w_gains.png'),
 ('Checks',
  'Unedited accuracy 80.70 % / 80.59 %. Shift edits reproduce the predicted $\\Delta\\log M_2$ to $3\\times10^{-8}$; product edits hit their targets to $10^{-7}$. '
  'Reflecting the jets flips the odd part with the same magnitude (z: $-0.63$ vs $+0.66$). On bulk jets the odd part shrinks 8–14$\\times$. '
  'At 10 mrad a rotation that leaves $M_2$ unchanged moves the units by 0.8 of the translation; at 60 mrad, by 0.24. '
  'Untrained networks give an even gain of about 0.5, so agreement of the even part with a matched dilation is partly generic. '
  'A per-jet regression on $(\\vec c\\cdot\\vec M_1/M_0, |\\vec c|^2)$ that keeps the fixed-direction response gives a ratio of 1.0 at z and 1.7–2.8 at the margin, so the value from z onward depends on the estimator.', None),
 ('Why it matters',
  'The centroid-linear part of the shift response is weighted less than the quadratic part from the first summed stage, S, onward, so the two-gain form is not created in the dense layers h4/h5. '
  'The test for a product of $M_0$ and $M_2$ found no unit with the product\'s signature, but its null band is wide, so where (or whether) a product forms remains open.', None),
]
```
Conclusion text for site/conclusions.py (`'E23W_EXPLAINER'`): the "Conclusion" paragraph at the top of this report.
