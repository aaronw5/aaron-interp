# E23: naturally off-centre jets and on-manifold axis edits, JEDI-linear 3f n64 (2026-10-17)

**Claims:** A4 / D4 (thesis v3), review weakness 1. Does the network follow $\sqrt{M_0M_2}$ about the stored axis, or the invariant mass $m$, on unedited jets whose stored axis is off the pT centroid?

**Files**
- Script: `interp/e23_tail_jedi.py`. Jobs `e23-smoke` (500 per class) and `e23-run`: 1 CPU / 2 GiB, wall 1034 s, CPU 160 s, pod peak memory 2047 MiB. E17 also logged 2047 MiB, which points to the cgroup ceiling with page cache.
- Results: `results/jedi/e23_tail.json` (also `e23_tail_500.json`).
- Figure: `results/jedi/e23_tail.png`.
- Log: `logs/e23-run.log`.
- Explainer: `E23_EXPLAINER` in `site/explain.py`, added to Paper A as result 14.

## Procedure

**Sample**
- `sample(20000, 101)`: N = 100,000. Unedited accuracy 80.92 % (in the 80.6–81.3 % range); W→Z 0.024, Z→W 0.062.

**Read-out**
- Ridge of $\log m$ on $z$, using the E17 procedure.
- Fitted on `sample(4000, 103)` after removing the 7,645 jets it shares with the main sample: 12,355 jets, of which 6,177 are used for the fit. Alpha is chosen on the last quarter of the fit half.
- Held-out $R^2$: 0.942 on the read-out sample, 0.941 on the main sample. On the main sample, corr($\log m$, $\log\sqrt{M_0M_2}$) = 0.999994.

**Moments about the stored axis (real particles)**
- Definitions: $q = |\vec M_1|^2/(M_0M_2)$ and $d = |\vec M_1|/M_0$.
- $q$ quantiles: p50 0.00035, p90 0.0019, p99 0.0058, max 0.025.
- $d$ quantiles: p50 1.46, p90 5.8, p99 13.1, max 30.9 mrad.
- $|$stored axis − E-scheme axis of the 64 particles$|$ has the same quantiles (1.46 / 5.9 / 13.2 mrad).

**Sets**
- Tq1/2/5 and Td1/2/5: top 1/2/5 % of $q$ and of $d$.
- Tq1c/2c/5c: top 1/2/5 % of $q$ within each class. These were added after the smoke run because the global tails hold almost no W or Z jets.
- Bulk: $q$ below the median (50,000 jets, $q$ median 8.6e-5, $d$ median 0.51 mrad).

| set | n | g/q/W/Z/t | $q$ p50 | $d$ p50 [mrad] | acc | particles beyond train 99 % angle |
|---|---|---|---|---|---|---|
| Tq5 | 5000 | 1009/370/8/12/3601 | 0.0045 | 11.0 | 0.832 | 2.46 % |
| Td5 | 5000 | 731/296/8/6/3959 | 0.0044 | 11.0 | 0.854 | 2.61 % |
| Tq5c | 5000 | 1000 each | 0.0029 | 6.3 | 0.725 | 2.05 % |
| Tq1 | 1000 | 303/80/1/3/613 | 0.0068 | 14.3 | 0.800 | 2.39 % |
| bulk | 50000 | 11116/14438/9586/9119/5741 | 8.6e-5 | 0.51 | 0.801 | 1.95 % |

## (A) Observational matching (no edit)

**Matching**
- Pairs are formed within class (W, Z, t). Greedy 1:1 without replacement, calipers 0.02 on both match variables.
- Statistics: bootstrap over pairs (1,000), sign-flip permutation (4,000).

**Match 1: on $(\log m, \log M_0)$**
- Tail − bulk $\Delta\log\sqrt{M_0M_2}$ is +0.0017 to +0.0044. This is the H1 prediction; H0 predicts 0.
- Read-out differences:
  - Tq5: +0.0004 [−0.0048, 0.0062], 2,660 pairs, p_perm 0.88.
  - Tq5c: +0.0004 [−0.0044, 0.0050], 2,821 pairs, p 0.86.
  - Td1: +0.0101 [−0.0026, 0.0231], p 0.14.
- Z-call differences are 0 within ±0.008 (Tq5c −0.0007 [−0.0081, 0.0063]).

**Match 2: on $(\log\sqrt{M_0M_2}, \log M_0)$**
- $\Delta\log m$ is −0.0010 to −0.0032.
- Read-out differences are −0.0042 to +0.0042, and all intervals include 0.

**Holm over the 18 tests:** no adjusted p below 1.

**Reading:** the interval half-widths (0.005–0.013) exceed both predictions (0.002–0.004), so this arm does not separate H0 from H1 at N ≤ 2,821 pairs.

## (B) Causal recentring and controls

**Edit checks**
- After recentring, $|\vec M_1|/M_0 \le 7.4\times10^{-9}$.
- $|\Delta\log m| \le 4.4\times10^{-8}$.
- pT and padded slots are unchanged.
- Rotation: $|\Delta\log M_2| \le 7.5\times10^{-8}$.
- P freeze on unedited jets: max |Δlogit| = 0.

**Read-out change [×10⁻³] with 95 % CI.** "Real" is the matched real change (offsets × $e^{\rm pred}$); "obs/real" is the calibrated ratio.

| set | pred (recentre) | recentre Δread | real Δread | obs/real [CI] | reflect pred / Δread | reflect obs/real | rotate Δread | P-frozen Δread |
|---|---|---|---|---|---|---|---|---|
| Tq1 | −3.7 | −8.8 [−11.1, −6.5] | −4.7 | 1.86 [1.33, 2.46] | +10.9 / +15.6 | 1.16 [0.98, 1.34] | −1.9 [−4.1, 0.1] | −5.4 |
| Tq5 | −2.4 | −4.2 [−5.1, −3.4] | −2.7 | 1.56 [1.23, 1.91] | +7.3 / +7.9 | 1.06 [0.94, 1.18] | −0.4 [−1.1, 0.4] | −3.2 |
| Td5 | −2.3 | −4.2 [−5.1, −3.3] | −2.8 | 1.50 [1.19, 1.86] | +7.0 / +7.9 | 1.08 [0.95, 1.20] | −0.4 [−1.1, 0.4] | −3.3 |
| Tq5c | −1.8 | −3.4 [−4.1, −2.6] | −2.1 | 1.57 [1.22, 1.97] | +5.5 / +7.1 | 1.19 [1.07, 1.33] | −0.1 [−0.7, 0.6] | −2.2 |
| bulk (c1) | −0.06 | +0.02 [−0.25, 0.30] | — | — | — | — | — | — |

**Read-out, all nine tail sets**
- Calibrated ratio obs/real: 1.50–2.01. Every interval excludes 0 and 1.
- Raw ratio obs/pred: 1.73–2.77.
- Matched real change / pred: 1.11–1.53.
- H0 (Δread = 0) for recentring: Holm-adjusted normal p ≤ 10^−10.4; the permutation p sits at its floor, 10^−2.03 after Holm.
- Rotation: every interval includes 0 (Holm-adjusted normal p ≥ 0.66).
- P-unit freeze keeps 57–80 % of the recentring response.

**Even/odd decomposition (point estimates from the set means)**
- Recentre and reflect are $\mp\vec c$, with $\vec c = \vec M_1/M_0$.
- Prediction: even part ½q, odd part q (from the cross term $2\vec c\cdot\vec M_1$).
- Observed even / predicted even: 0.94, 0.75, 0.80, 1.04 (Tq1, Tq5, Td5, Tq5c).
- Observed odd / predicted odd: 1.66, 1.24, 1.28, 1.43.

**Per class, Tq5c recentring (Δread / pred / real, ×10⁻³)**
- t: −6.9 / −3.2 / −3.7.
- W: −0.86 / −0.74 / −1.20.
- Z: −0.38 / −0.92 / −0.84.

**Calls**
- Calls changed by recentring: Tq5 1.82 % [1.46, 2.20], Tq5c 2.92 % [2.44, 3.38]; 1.64–3.2 % over all tail sets.
- By rotation (same mean displacement): Tq5 1.60 % [1.28, 1.96], Tq5c 2.18 %; 0.85–2.5 % over all sets.
- Recentre − rotate: Tq5 +0.22 pp [−0.16, 0.60], Tq5c +0.74 pp [0.32, 1.20].
- Reflected edit: 1.8–4.2 %. Matched real change: 0.22–0.50 %. Bulk recentring (c1, mean displacement 0.65 mrad): 0.82 % [0.58, 1.06].

**Z→W against W→Z after recentring**
- Tail W/Z jets in Tq5c (n 2,000): 10 vs 5 (McNemar p 0.30).
- Tq1c: 2 vs 0. Global tail sets: 0 vs 0 (4–20 W/Z jets).
- No McNemar Z/W test survives Holm.

**Accuracy before → after recentring**
- Tq5: 83.24 → 83.46 % (p 0.27).
- Tq5c: 72.50 → 73.02 % (McNemar p 0.020, unadjusted).

## (C) Physical-size random-direction shift (20,000 bulk jets)

| size | ½Δlog M2 [×10⁻³] | Δread [×10⁻³] | ratio | calls changed | Z→W / W→Z (McNemar p) |
|---|---|---|---|---|---|
| 0 (identity) | 0 | 0 | — | 0 (max \|Δlogit\| 0) | 0 / 0 |
| d p50, 1.46 mrad | 0.60 | 0.36 [0.17, 0.55] | 0.59 [0.28, 0.91] | 1.43 % [1.26, 1.59] | 39 / 35 (0.73) |
| d p90, 5.83 mrad | 9.4 | 5.0 [4.7, 5.3] | 0.53 [0.50, 0.57] | 2.21 % [2.03, 2.43] | 36 / 70 (1.2e-3) |
| d p99, 13.1 mrad | 42.7 | 23.5 [23.0, 24.1] | 0.55 [0.54, 0.56] | 3.79 % [3.55, 4.09] | 22 / 171 (9e-30) |

Sizes taken from the $|$stored − E-scheme axis$|$ quantiles (1.46 / 5.9 / 13.2 mrad) give the same numbers within 0.05 pp.

## Hypotheses

**H0 (invariant mass): excluded for the read-out**
- H0 predicts ratio 0 under recentring. Measured: 1.50–2.01 of the matched real change, Holm p ≤ 10^−10.4.
- The sign reverses under the reflected edit.

**H1 (√(M0M2) about the stored axis, ratio about 0.8)**

| H1 component | result |
|---|---|
| Direction | Matches: recentre lowers the read-out and reflect raises it. |
| Magnitude | Above the stated 0.78–0.84 raw ratio: raw 1.7–2.8. Calibrated 1.5–2.0 for recentring, 1.06–1.19 for the reflected edit. |
| Odd term | The part linear in $\vec M_1$ is weighted 1.2–1.7× its prediction. The even part is at 0.75–1.04×. |
| c1 | ≈ 0: observed. |
| c3 | ≤ 0.03 of calls: observed (0.9–2.5 %). |
| Z→W flips in tail W/Z | Not observed: 10 vs 5, p 0.30. The predicted read-out change for tail W/Z jets is < 1×10⁻³, against 0.13 between the W and Z masses. |

**H2 (generic displacement / off-manifold artefact)**
- Excluded for the read-out:
  - The rotation of equal displacement gives Δread consistent with 0.
  - The sign follows ½Δlog(M0M2) between recentre and reflect.
  - Bulk recentring gives 0.
- Not excluded for calls:
  - Rotation changes 0.85–2.5 % of tail calls, against 1.6–3.2 % for recentring.
  - Random shifts of 1.5 mrad change 1.4 % of bulk calls without a Z/W direction.
  - Call changes on natural jets at these sizes are mostly not mass-specific.
  - A mass-specific W→Z excess appears only at ≥ 5.8 mrad (p90 and p99 sizes).

## Reading (scoped to these measurements)
- **Read-out:** on 1,000–5,000 unedited off-centre jets, an edit that keeps $m$ fixed to 4e-8 and sets $\vec M_1 = 0$ lowers the log m read-out on $z$ by 1.5–2.0× the response to a real mass change of the predicted size. A rotation of equal particle displacement does not change it.
- **Calls:**
  - The tails are mostly top (72 %) and gluon jets, with $d$ ≈ 6–14 mrad. The predicted change for tail W/Z jets is < 0.1 % in mass.
  - Recentring changes calls at about the rate of the rotation control, with no Z→W excess.
  - The mass-specific decision effect of the axis convention (W→Z) appears in random shifts at the 90th–99th percentile of natural centroid offsets.
- **Arm (A):** observational matching is underpowered at this N; it separates neither hypothesis.
