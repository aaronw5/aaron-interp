# E15 verification (skeptic agent, 2026-10-16): product or sum of M0 and M2

Scripts: interp/e15_verify.py (job e15-verify; first attempt stopped at a division by zero after section 3, log kept as
logs/e15-verify-fail1.log; rerun complete), interp/e15_verify_iso.py (job e15-verify2). Results: results/jedi/e15_verify.json,
results/jedi/e15_verify_iso.json. Logs: logs/e15-verify.log, logs/e15-verify2.log. Both jobs used 1 CPU / 2 GiB and exited 0, with
wall times of 247 s and 598 s (554 s of which was loading). Peak memory was 2047 MiB and 1624 MiB, so the 3 GiB request of E15 run 2
was not needed.

## Sample and checks
- sample(per_class=2000, seed=23). The 361 jets also drawn by seed 11 (E15's sample) were dropped, leaving N = 9,639.
- Accuracy 0.8105 (binomial sd 0.004). run_jetwise identity: max |diff| 0.0.
- Jets in range in all cells: 9,633 (part 1) and 9,636 (part 2).
- λ = 0.006432. Share of the M0 term by M2 third: 0.767 / 0.503 / 0.270. M2 third edges: 5.65 and 8.11 GeV.

## 1 Code review of interp/e15_product_jedi.py
I found no bug that changes a reported number:
- Padded slots are not edited (real = pT > 0).
- in_range is the same as e5_causal.in_range.
- The A2 moment ratios are exact (M0 0.8 / 1.25; M2 0.64 / 1.5625).
- Read-outs are fit on the unedited train half and evaluated on the test half, with no split leakage.

These points change the interpretation:
- a. **Sum references for the read-outs.** The sum's reference values assume the read-out reports $V_{sum} = \lambda M_0 + M_2$ linearly (m̂ GeV: β = 0) or reports $\log V_{sum}$ (log m̂: β_sum = 1). A linear read-out of log m on units that are linear in M0 and M2 is additive in (c, k), so it gives the product's value (β_sum = 0) on the log metric. Section 3 measures both cases.
- b. **Per-unit log null.** U_log = U(c,1)·U(1,k)/U(1,1) is exact for a product only when the unit has no offset. A ReLU unit with a bias fails it even when the unit is an exact function of $\sqrt{M_0 M_2}$ (section 3).
- c. **Propagation identity.** On a unit's on-region the unit is linear in the previous stage, so I = W·I(upstream) holds by construction.
- d. **The A2 c-edit is not "M0 alone" at the input.** pT × c with angles × c^−½ moves M0 at fixed M2. It also moves every pT-threshold feature: the a32 ramp at 11 GeV and the S19 count of pT > 1 GeV. "M0 alone" holds for the moments, not for the network's input.
- e. **The S0 patch moves one of several M0 carriers.** Its slope per M2 third combines the combination rule with S0's share of the M0 path, and that share changes from third to third (section 5).
- f. **Bootstrap se.** The se resamples jets with the read-out held fixed. Refitting the read-out on another sample moves the interaction by more than this se: m̂ I at S 0.0169 → 0.0198 sd and at h5 0.0336 → 0.0281 sd, against se 0.0005–0.0008. As β: S 0.33 → 0.42, h5 0.60 → 0.53.

## 2 Reproduction on the fresh sample
| quantity | E15 (seed 11) | verify (seed 23) |
|---|---|---|
| A2 m̂ (GeV) β at S / h4 / h5 / z | 0.33 / 0.70 / 0.60 / 0.32 | 0.42 / 0.70 / 0.53 / 0.32 |
| A2 exp(log m̂) β | 1.43 / 1.14 / 1.19 / 0.22 | 1.61 / 1.19 / 1.17 / 0.28 |
| log m̂ β_sum | −0.49 / −0.09 / −0.42 / +0.43 | −0.63 / −0.11 / −0.36 / +0.28 |
| T h4, M2-only ÷ pT×c (low / mid / high) | 0.755 / 1.146 / 1.061 | 0.751 / 1.148 / 1.089 (se ≤ 0.009) |
| T h4, M0-only ÷ pT×c | 0.629 / 0.449 / 0.477 | 0.633 / 0.435 / 0.467 (se ≤ 0.005) |
| T z, M0-only ÷ pT×c | 0.649 / 0.539 / 0.529 | 0.648 / 0.542 / 0.534 |
| S0 (e_c, e_k); S50 (e_c, e_k) | 9.31, 0.00; 0.05, 2.47 | 9.37, 0.00; 0.04, 2.45 |
| I of S56 / S1 / S29 / S13 (sd) | 0.642 / 0.333 / 0.077 / 0.144 | 0.632 / 0.328 / 0.076 / 0.142 |
| β of h4_18 / h5_0 / h5_28 | −0.29 / −0.04 / +0.58 | −0.26 / −0.05 / −1.06 (h5_28 I_pred +0.097 → −0.043) |
| S0 patch, h4 slope (low / mid / high) | 0.525 / 0.438 / 0.286 | 0.613 / 0.386 / 0.290 |
| S19 control, h4 slope | 0.041 / 0.002 / 0.008 | 0.099 / 0.003 / 0.007 |
| W→Z: A2 M0 × 1.25; M2 × 1.5625; S0 patch with Δlog S0 ≥ 0.5 (mid third) | 0.72; 0.71; 0.82 | 0.74; 0.71; 0.79 |

New check: (M0-only response) + ½ (M2-only response), both divided by the pT×c response, is 1.00–1.04 at S, h4, h5 and z in every third. A read-out that is locally a function of M0 and M2 alone gives exactly 1.

## 3 Calibration control: the same read-out procedure on populations whose combination rule is known
Each population has 64 units built from the true moments of the same jets in the same cells. Read-outs are ridge fits on the
unedited train half, evaluated on the test half, as in E15. Random populations were drawn with 3 seeds, listed in order.

| population | m̂ GeV β | exp(log m̂) β | log m̂ β_sum | T: M0-only ÷ pT×c (low / mid / high) | per-unit: log null closer (median); unit β |
|---|---|---|---|---|---|
| true sum λM0 + M2 | 0.00 | – | 1.00 | 0.754 / 0.496 / 0.294 | – |
| true product √(M0·M2) | 1.00 | – | 0.00 | 0.500 / 0.500 / 0.500 | – |
| lin: [M0, M2] | 0.00 | 1.00 | 0.00 | 0.790 / 0.573 / 0.322 | – |
| sumwin: ReLU windows on λM0 + M2 | −0.73, −0.76, −0.77 | −0.71, −0.72, −0.71 | 2.36, 2.43, 2.48 | 0.783 / 0.487 / 0.296 | 0.00; 0 |
| prodwin: ReLU windows on √(M0·M2) | 1.00, 1.00, 1.00 | 1.03–1.04 | −0.02, −0.02, −0.03 | 0.500 / 0.500 / 0.499 | 0.03; −0.89 … 0.67 |
| sepwin: ReLU on random directions in (z(M0), z(M2)) | 1.15, 0.90, 1.04 | 1.55, 0.45, 1.19 | −0.22, 0.86, −0.01 | 0.55 / 0.47 / 0.53; 0.59 / 0.22 / 0.41; 0.54 / 0.42 / 0.49 | 0.00; 0 |
| logsepwin: the same on (z(log M0), z(log M2)) | 1.16, 0.88, 0.99 | 1.00, 0.99, 0.98 | 0.00 | 0.50 / 0.51 / 0.50 | 0.00; 0 |
| rand1: 64 random ReLU units on the trained network's S | −0.01, 0.42, 1.14 | 0.55, 0.69, 1.89 | 0.46, −0.17, −0.97 | 0.66 / 0.45 / 0.46; 0.51 / 0.48 / 0.46; 0.49 / 0.44 / 0.50 | 0.43; −3.2 … 5.2 |
| rand2: 64 → 32 random ReLU on S | −3.33, −10.77, 0.09 (R² m 0.87–0.89) | | | | |
| rand1 on the trained h4 (32 units) | 0.60, 1.79, −0.03 | 1.57, 1.95, 0.28 | −0.33, −1.61, 0.28 | 0.69 / 0.40 / 0.42; 0.64 / 0.58 / 0.67; 0.66 / 0.47 / 0.44 | – |
| **network** S / h4 / h5 / z | 0.42 / 0.70 / 0.53 / 0.32 | 1.61 / 1.19 / 1.17 / 0.28 | −0.63 / −0.11 / −0.36 / +0.28 | S 0.62 / 0.58 / 0.55; h4 0.63 / 0.44 / 0.47; z 0.65 / 0.54 / 0.53 | h4: 0.47; −1.7 … 2.4 |

Target control: ridge read-outs of V_sum and of V_prod themselves from the network's units. The A2 β of each (true values 0 and 1),
with R² in brackets:

| stage | V_sum read-out β (R²) | V_prod read-out β (R²) |
|---|---|---|
| S | 0.09 (0.995) | 0.42 (0.977) |
| h4 | 0.03 (0.995) | 0.71 (0.991) |
| h5 | 0.02 (0.986) | 0.53 (0.986) |
| z | 0.17 (0.971) | 0.32 (0.968) |

## 4 Read-out-free test on the network's own units (interp/e15_verify_iso.py)
- **Edits** are matched in log moment, c ∈ {0.8, 1.25}:
  - M0 step: pT × c with angles × c^−½, so M0 × c.
  - M2 step: angles × c^½, so M2 × c.

  Both steps give the same $\sqrt{M_0 M_2}$.
- **Statistic** per unit and M2 third: $d = \sum_j (e_0 - e_2)\,\mathrm{sgn}(e_0+e_2) / \sum_j |e_0+e_2|$.
  - A unit that reads M0 and M2 only through $\sqrt{M_0 M_2}$ gives d = 0, whatever its nonlinearity.
  - A unit that reads them through $\lambda M_0 + M_2$ gives d ≈ 2s − 1.

Controls:
- Moment models: the sum gives +0.47 / −0.01 / −0.48 and the product 0.00.
- Calibration units: sumwin median +0.55 / −0.01 / −0.54; prodwin 0.00 for every unit. sepwin gives ≈ +1.0, because z(M0) moves by more sd than z(M2) for the same log step.

Network units (median d, low / mid / high third):
- **Single-moment S units:** S0 +1.00 in all thirds; S50 −0.71 / −1.05 / −0.90; S29 −3.82 / −2.95 / −3.04.
- **h4** (44–49 units with response > 0.1 sd): +0.61 / +0.15 / +0.02. IQR [−0.02, 1.43] / [−0.47, 1.17] / [−0.74, 1.10]; 10–16 % of units have |d| < 0.15.
- **h5:** +0.65 / +0.32 / +0.40.
- **z** (16 units): +0.70 / +0.03 / −0.00. IQR low [0.52, 1.02], mid [−0.26, 0.30], high [−0.20, 0.22].
- **E6 mass-window units:** z0 +0.79 / −0.40 / −0.25; z7 +0.55 / +0.06 / +0.19; z10 +0.95 / −0.01 / −0.18; z14 +0.59 / −0.46 / −1.02.
- **Logit differences:** Z−W +0.39 / −0.26 / −0.43; t−Z +1.06 / +0.38 / +0.97; W−q +0.68 / −0.38 / +0.44 (se ≤ 0.04).

Closure is |both-step response − M0 step − M2 step| in the same normalisation, and is 0 for a function of (M0, M2) at first
order. Median closure: z 0.03 / 0.25 / 0.19; logit differences 0.17 / 0.59 / 0.18.

Calls:
- The M0 × 1.25 and M2 × 1.25 cells give different calls for 28.6 / 25.1 / 47.0 % of jets by third (33.6 % overall).
- Alone, the M0 step changes 47.1 % of calls and the M2 step 29.9 %.
- W jets (mid third, n 1,459): W→Z 0.89 under the M0 step and 0.94 under the M2 step.

The M0 step also moves the pT-threshold features, so d ≠ 0 combines the combination rule with those features. The median d
falls from the low to the high third, the direction a sum predicts (h4 +0.61 → +0.02; z +0.70 → 0.00; logit Z−W +0.39 → −0.43).

## 5 S0's share of the M0 path (B)
- **Setup.** Donors were found for 91.9 % of jets; median |Δlog S0| 0.42. Calls changed 0.367; accuracy on patched jets after the patch 0.575 (whole sample unpatched 0.8105).
- **h4 slope** on ½Δlog S0: 0.613 ± 0.021 / 0.386 ± 0.021 / 0.290 ± 0.007. Divided by the pT×c gain: 0.638 / 0.375 / 0.320 (high/low 0.50; E15 0.595).
- **S0's share of the M0 path** (slope ÷ 2 × M0-only response; 1 if S0 carried all of M0):

  | patch | h4 | h5 | z |
  |---|---|---|---|
  | matched-donor S0 | 0.50 / 0.43 / 0.34 | 0.56 / 0.41 / 0.38 | 0.62 / 0.39 / 0.30 |
  | S0 × 1.25 on every jet | 0.59 / 0.48 / 0.41 | | |
  | S0 × 0.8 on every jet | 0.50 / 0.50 / 0.39 | | |

- **Input-level comparison.** At h4 the M0-only elasticity changes little from the mid to the high third (0.435 → 0.467), while the S0-patch slope falls (0.375 → 0.320). The fall in the high third goes with S0's smaller share of the M0 path.
- **S19 control** (h4): 0.099 / 0.003 / 0.007, with 13.5 % of calls changed. The low-third value is not 0 here (E15: 0.041).

## Status
- **Reproduced:** yes. The headline numbers fall within the sample-to-sample spread listed in section 2. The per-unit β of h5_28 is not reproduced.
- **A2 read-out β (0.60–0.70 at h4/h5) as "between sum and product":** not supported. The same procedure gives:
  - 0 for a sum held linearly and −0.73 to −0.77 for a sum followed by ReLU windows;
  - 0.88 to 1.16 for ReLU units on separate M0 and M2;
  - −0.01 / 0.42 / 1.14 for a random ReLU layer on the trained S, and −0.03 to 1.79 for one on h4.
- **log m̂ β_sum:** a linear sum gives 0, the product's value.
- **Per-unit "neither null":** exact product units give log null closer 0.03 and unit β −0.89 to 0.67 under the same test.
- **T ratios:** these separate the known rules. The network's h4 pattern is close to S's own read-out pattern and lies within the range of random ReLU layers on S.
- **Network's own units:** they do not respond as functions of $\sqrt{M_0 M_2}$ alone (per-unit d spread ≈ 1.5 at h4). This is confounded with the pT-threshold features that only the M0 step moves.
- Confidence that the E15 reading is correct as stated: 0.2.

## Corrected reading
- **Fresh-sample numbers.** On 9,639 fresh jets, E15's read-out numbers reproduce:
  - m̂ β 0.42 / 0.70 / 0.53 / 0.32 at S / h4 / h5 / z;
  - T M0-only ratio at h4 0.633 / 0.435 / 0.467 by M2 third.
- **What the read-out metrics give for known rules** (same ridge read-out):
  - m̂ β: sum held linearly 0; sum followed by ReLU windows −0.75; product 1; ReLU units on separate M0 and M2 0.9–1.15; a random ReLU layer on the trained S −0.01 / 0.42 / 1.14.
  - log β_sum: a linear sum gives 0, the product's value.
  - Read-outs of V_sum and V_prod from the same h4 units: β 0.03 and 0.71.
- **What the A2 interaction measures.** It is a property of the read-out target and of the generic nonlinearity on S, not of a combination rule learned in h4/h5.
- **T elasticities.** These separate the known rules. The read-out's M0 elasticity is:

  | read-out | low / mid / high M2 third |
  |---|---|
  | S | 0.62 / 0.58 / 0.55 |
  | h4 | 0.63 / 0.44 / 0.47 |
  | z | 0.65 / 0.54 / 0.53 |
  | product | 0.50 / 0.50 / 0.50 |
  | sum | 0.75 / 0.50 / 0.29 |
  | random ReLU layers on S | 0.49–0.66 / 0.44–0.48 / 0.46–0.50 |

- **Matched M0 and M2 steps on the network's own units.** Median d at h4 is +0.61 / +0.15 / +0.02 (IQR width about 1.5), and logit Z−W is +0.39 / −0.26 / −0.43; product 0, sum +0.47 / 0 / −0.48. This includes the pT-threshold features that the M0 step also moves.
- **What holds as stated:**
  - S holds M0 (S0) and M2 (S50) in separate units, and S56, S1, S29 and S13 carry interaction of the product's sign.
  - The read-out matches a product in the high-M2 third, from S onward.
- **What does not hold:**
  - "M0 and M2 combine with an interaction of 0.6–0.7 of a product's at h4/h5."
  - "Product formed in h4/h5" (A4/A9).
