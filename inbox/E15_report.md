# E15 — product or sum: how h4/h5 combine M0 = ΣpT and M2 = Σ pT r² (agent report, 2026-10-16; claims A4, A9)

Script interp/e15_product_jedi.py (run: `python -m interp.e15_product_jedi`; figure: `--plot`). Results results/jedi/e15_product.json,
figure results/jedi/e15_product.png, table results/jedi/e15_product_table.png, log logs/e15-run2.log. Run 1 (job e15-run, 20:22 UTC)
kept as results/jedi/e15_product_run1.json + logs/e15-run1.log (bug, see Issues). Run 2: job e15-run2, 1 CPU / 3 GiB, wall 845 s.

## Sample, network, read-outs
- sample(per_class=2000, seed=11) = 10,000 test jets; fixed-point JEDI-linear 3-feature n64; accuracy 0.8028 (expected 80.6–81.3 %;
  binomial sd 0.40 % at N = 10,000); untrained control Jedi(untrained=True, seed 0) 0.200. run_jetwise(S := own S) max |diff| 0.0 at S, h4,
  h5, z, logits. Only real particles edited (padded slots 28.5 %, moved by 0.0); pT order kept in 100 % of jets.
- Read-outs: standardised ridge of log m and of m (GeV) on S, h4, h5, z, fit on the unedited train half (5,000 jets), evaluated on the test
  half. Held-out R² log m | m: S 0.898 | 0.976, h4 0.973 | 0.991, h5 0.973 | 0.985, z 0.941 | 0.966 (untrained log m ≤ 0.15).
- Hypotheses on the true moments: sum V = λM0 + M2 with λ = median M2 / median M0 = 0.00646 (M2 6.61 GeV, M0 1022 GeV), share of the M0 term
  0.503 at the median (0.763 / 0.503 / 0.276 in the low / mid / high M2 third; third edges M2 = 5.69, 8.14 GeV); product V = √(M0·M2).

## A — factorial as planned: pT × c, angles × k, c, k ∈ {0.8, 1, 1.25} (9 cells; 9,997 jets in range in all cells)
pT × c multiplies M2 = Σ pT r² by c as well: M0 → c·M0, M2 → c·k²·M2. Both hypotheses are homogeneous of degree 1 in pT
(sum → c(λM0 + k²M2), product → c·k·√(M0·M2)), so both are exactly log-additive in (c, k). Model rows on the true moments (test half,
4,998 jets): sum, product and true m each give "log null closer" 1.00 and β = I/I_pred = 1.00; log-interaction 0 for all three.
This design does not separate the two hypotheses; its numbers are kept in the json (res['A']) and measure pT-homogeneity:
m̂ (GeV) β 0.75 (S), 0.77 (h4), 0.73 (h5), 0.76 (z) where both hypotheses give 1; projected log m̂ interaction −0.017 (S), +0.011 (h4),
+0.005 (h5), +0.011 (z) where both give 0 (this is the network's curvature not described by either hypothesis).

## A2 — separable factorial (added): pT × c, angles × k/√c (9 cells; 9,989 jets in range)
M0 → c·M0 and M2 → k²·M2 exactly (median moment ratios: M0 0.800/1.000/1.250, M2 0.640/1.000/1.562). Model rows: sum → log null closer
0.00, β 0.00; product and true m → 1.00, β 1.00. Log m̂ interaction projected on −sign((c−1)(k−1)): sum +0.0203, product 0.
Metric per unit/read-out on jets with value > 0 in all 9 cells; I in sd of the unit; se from 1,000 Poisson bootstrap resamples of jets.

| quantity | n | med |U−U_lin| | med |U−U_log| | frac log closer | I ± se | I pred (product) | β |
|---|---|---|---|---|---|---|---|
| m̂ GeV at S | 4995 | 0.037 | 0.046 | 0.42 | 0.017 ± 0.0005 | 0.051 | 0.33 |
| m̂ GeV at h4 | 4995 | 0.053 | 0.051 | 0.54 | 0.039 ± 0.0007 | 0.055 | 0.70 |
| m̂ GeV at h5 | 4995 | 0.077 | 0.079 | 0.51 | 0.034 ± 0.0008 | 0.056 | 0.60 |
| m̂ GeV at z | 4995 | 0.095 | 0.100 | 0.46 | 0.017 ± 0.0009 | 0.054 | 0.32 |
| h4_18 | 164 | 0.251 | 0.654 | 0.16 | −0.153 ± 0.011 | 0.523 | −0.29 |
| h5_0 | 74 | 0.357 | 0.841 | 0.18 | −0.027 ± 0.037 | 0.708 | −0.04 |
| h5_28 | 1474 | 0.283 | 0.386 | 0.35 | 0.056 ± 0.008 | 0.097 | 0.58 |
| z9 | 2760 | 0.142 | 0.144 | 0.48 | −0.109 ± 0.002 | 0.065 | −1.67 |
| z11 | 1983 | 0.042 | 0.043 | 0.46 | 0.007 ± 0.001 | −0.001 | – |
| z12 | 1955 | 0.093 | 0.095 | 0.45 | −0.013 ± 0.001 | 0.022 | −0.59 |

- Log m̂ interaction (log units, projected; sum +0.0203, product 0): S −0.0099 ± 0.0004, h4 −0.0018 ± 0.0008, h5 −0.0085 ± 0.0005,
  z +0.0086 ± 0.0005; β_sum = −0.49, −0.09 ± 0.04, −0.42 ± 0.02, +0.43 ± 0.03 (1 = sum, 0 = product). Per-corner values ±0.001–0.028.
- All live units: median "log null closer" S 0.50, h4 0.47, h5 0.45, z 0.48; median β over units with |I pred| > 0.05 sd: S 0.33 (12 units),
  h4 0.34 (33), h5 0.12 (16), z 0.39 (3); unit β ranges from −8 to +22.
- S control: S0 responds to c only (e_c 9.31 sd per unit log c, e_k 0.00) and S50 to k only (e_c 0.05, e_k 2.47): I = 0.000 and 0.014 ± 0.001
  with I pred 0.000 and 0.013. The other S hinge sums carry interaction: S56 0.642 ± 0.007 (β 0.64), S1 0.333 ± 0.003 (0.58), S29 0.077 ±
  0.001 (0.93), S13 0.144 ± 0.002 (1.39).
- Propagation (on a downstream unit's on-region, I = W·I(upstream) to a median residual of 0.008–0.27 sd): h4_18 −0.153 = −0.151 from S units
  on in all cells + −0.002 from S units that switch; h4_33 0.772 = 0.775 + 0.002; h5_10 0.510 = −0.029 + 0.541; h5_28 0.056 = −0.008 + 0.067;
  z10 0.248 = 0.079 + 0.169; z9 −0.109 = −0.016 − 0.093.
- Untrained network: live units S 2, h4 7, h5 17, z 11; all but one have |I pred| < 0.05 sd (no pT response: untrained P has no pT weight);
  the control does not test the interaction.
- Calls per cell (in range): A2 c = 1.25 (M0 × 1.25 at fixed M2, true Δlog m 0.112, h4 Δlog m̂ 0.113): W→Z 0.72; A2 k = 1.25 (M2 × 1.56,
  Δlog m 0.223, Δlog m̂ 0.198): W→Z 0.71; A pT × 1.25 (Δlog m 0.223): W→Z 0.62. A2 c = 0.8: Z→t 0.74; A pT × 0.8: 0.63. Unedited W→Z 0.03.

## T — one moment moved, by M2 third (test half, 4,995 jets in range in A and A2; per jet Δlog m̂ / Δlog c)
Each response divided by the same third's response to pT × c alone (design A, k = 1; both hypotheses give 1; measured h4 1.00/1.04/0.92,
h5 0.88/0.97/0.97, z 0.74/1.17/0.92). M2-only = angles × k at c = 1 (product 1, sum ≈ 2(1 − s)); M0-only = A2 c-edit (product 0.5, sum ≈ s).

| read-out | M2-only low / mid / high | high/low | M0-only low / mid / high | high/low | S0 patch ÷ gain low / mid / high | high/low |
|---|---|---|---|---|---|---|
| h4 | 0.755 / 1.146 / 1.061 | 1.405 ± 0.018 | 0.629 / 0.449 / 0.477 | 0.758 ± 0.009 | 0.524 / 0.423 / 0.312 | 0.595 ± 0.036 |
| h5 | 0.805 / 1.175 / 0.964 | 1.198 ± 0.015 | 0.626 / 0.408 / 0.546 | 0.873 ± 0.009 | 0.613 / 0.398 / 0.425 | 0.693 ± 0.040 |
| z | 0.764 / 0.949 / 0.991 | 1.297 ± 0.017 | 0.649 / 0.539 / 0.529 | 0.815 ± 0.009 | 0.605 / 0.487 / 0.319 | 0.526 ± 0.035 |
| sum model | 0.504 / 1.009 / 1.399 | 2.777 | 0.750 / 0.496 / 0.299 | 0.398 | 1.252 / 1.015 / 0.506 | 0.404 |
| product model | 1 / 1 / 1 | 1.000 | 0.5 / 0.5 / 0.5 | 1.000 | 1 / 1 / 1 | 1.000 |

- Position of the measured high/low ratio between product (0) and sum (1) on a log scale: M2-only 0.33 (h4), 0.18 (h5), 0.25 (z);
  M0-only 0.30, 0.15, 0.22; S0 patch 0.57, 0.41, 0.71.
- In the mid third the two hypotheses give the same values (1.0 and 0.5); measured there 0.95–1.18 and 0.41–0.54: deviations up to 0.18 and
  0.09 not described by either hypothesis.
- High third (M2 > 8.1 GeV, m ≳ 90 GeV): M2-only 0.96–1.06 (product 1.0, sum 1.40), M0-only 0.48–0.55 (0.50, 0.30). Low third (M2 < 5.7
  GeV): M2-only 0.76–0.81 (1.0, 0.50), M0-only 0.63–0.65 (0.50, 0.75).

## B — S0 patch at matched M2 (run_jetwise, S0 := donor's S0)
Donor: another sample jet with |Δlog S29|, |Δlog S50| < 0.05 and |Δlog S0| ≥ 0.25, random; found for 92.2 % of jets; median |Δlog S0| 0.417,
median |Δlog M2(true)| 0.142 (unmatched donors: 0.997). Test half with donor: 4,635 jets. Accuracy 0.804 → 0.589; calls changed 0.353.
Sum model in S0 units: λ_S = median M2 / median S0 = 0.38; share 0.773 / 0.501 / 0.266 by M2 third.
- Slope of Δlog m̂ on ½Δlog S0, M2 low / mid / high third (product: equal; sum prediction 1.252 / 1.015 / 0.506, high/low 0.404):
  h4 0.525 ± 0.027 / 0.438 ± 0.006 / 0.286 ± 0.009 (high/low 0.544); h5 0.542 ± 0.025 / 0.387 ± 0.011 / 0.413 ± 0.014 (0.762);
  z 0.452 ± 0.026 / 0.571 ± 0.010 / 0.294 ± 0.008 (0.651); m̂ GeV at h5 0.757 / 0.458 / 0.331 (0.437), at z 0.560 / 0.535 / 0.275 (0.490).
- Calls, mid M2 third (W/Z masses): Δlog S0 ≥ 0.5: W→Z 0.82 (141 W jets); 0.25–0.5: 0.44 (385); Δlog S0 < 0: W→Z 0.00; Δlog S0 < −0.5:
  Z→t 0.64 (56 Z jets), −0.5 to −0.25: 0.21 (184). High third: Z→t 0.79 (24) and 0.33 (81) for the two negative bins. Low third: W→Z 0.16 (31).
- Control S19 patched with the same donors' S19 (median |Δlog S19| 0.514): slope on the donor's ½Δlog S0 0.022 (h4), 0.005 (h5), 0.007 (z);
  slope on its own Δlog S19 −0.018, −0.007, +0.001; calls changed 0.138; W→Z ≤ 0.01, Z→t 0.00 in the mid third.
- Control donors with M2 not matched: h4 0.576 / 0.375 / 0.283 (0.49), h5 0.554 / 0.304 / 0.402 (0.73), z 0.492 / 0.452 / 0.288 (0.59);
  sum prediction ratio 0.392. Unmatched minus matched slope per third: −0.12 to +0.05 (largest in the mid third).
- Supplementary S0 × f (all jets): h4 high/low 0.54 (f 0.7), 0.59 (0.85), 0.49 (1.2), 0.43 (1.4).
- Untrained network: S0 not live (sd 0); with S0 set to the trained donors' S0, calls changed 0.000; read-out R² ≤ 0.15, slopes not interpretable.

## Against the plan's expectations
- Product expectation 1 (log null closer in ≥ 70 % for E6's mass-like units and the read-outs): A2 0.16–0.48 (units), 0.46–0.54 (m̂ GeV).
- Product expectation 2 (I ≠ 0 at > 5 se with the predicted sign): m̂ GeV at h4 / h5 / z: 56 / 42 / 19 se, sign as predicted, β 0.70 / 0.60 / 0.32.
- Product expectation 3 (S: linear null wins, I ≈ 0): S0 and S50 I ≈ 0 with I pred ≈ 0 (each responds to one moment); S56, S1, S29, S13 I 0.08–0.64 sd.
- Product expectation 4 (S0-patch slopes equal across M2 thirds within 20 %): high/low 0.54 (h4), 0.76 (h5), 0.65 (z).
- Sum expectation (linear null wins at h4/h5/z, I ≈ 0): m̂ β 0.32–0.70; sum expectation (S0 response falls with M2): high/low 0.54–0.76 vs 0.40.

## Reading (scoped to these measurements)
The planned factorial gives the same prediction for both hypotheses; with M0 and M2 moved separately (A2), the network's mass read-out takes
an interaction between the sum's (β 0) and the product's (β 1): β 0.60–0.70 at h4/h5 and 0.32 at z for m̂ in GeV; log m̂ interaction
−0.09 (h4) to +0.43 (z) of the sum's. Normalised by the pT × c response, the responses to M0 alone and to M2 alone change with M2 by 15–33 % of
the log distance from the product toward the sum, and match the product within 0.06 in the high-M2 third; the S0-only patch changes by 41–71 %
of that distance. Single h4/h5/z units do not follow either null (unit β −8 to +22); on each unit's on-region its interaction is W·I(upstream),
from S units in h4 and from upstream units switching in h5/z. S already carries M0 (S0) and M2 (S50) as separate single-moment units, and other
S hinge sums (S56, S1, S29, S13) already show interaction with the product's sign. The non-recentring (translation) result is not tested here.

## Issues
1. Run 1 had λ inverted (median S0 / median M2 = 2.63 instead of median M2 / median S0): share of the M0 term 0.87 instead of 0.5; its sum
   predictions (B slopes 1.29–1.74, A log interaction ≈ 0) were wrong. Fixed in run 2; measured network quantities are identical in both runs.
2. The planned factorial (pT × c) changes M2 with c; design A2 was added (angles × k/√c). Plan deviation, made so that the edit separates the
   hypotheses as the plan intends.
3. min_n lowered from 200 to 30 jets on in all 9 cells so that h4_18 (164 jets in A2, 53 in A) and h5_0 (74, 47) are reported; their se are large.
4. S units are sums over particles of hinge functions of (pT, r), not functions of M0 and M2 alone, so "sum over particles" does not imply
   additivity in the edit factors; only S0 and S50 isolate one moment each.
5. The S0 patch moves S0 alone; other S units also respond to pT (S43, S20, S51, S60 e_c 1.5–4.1 sd in A2), so the absolute slopes (0.29–0.57)
   are below 1 for either hypothesis; the comparison uses the variation across M2 thirds.
6. Untrained network has no pT response (control degenerate for this test); its read-out R² ≤ 0.15.
7. Accuracy 80.28 % on this sample is 0.3 % below the stated 80.6–81.3 % range (0.8 binomial sd); run_jetwise identity exact.
8. Cluster: job name e15-run existed, so e15-run2. Load took 655 s on node rci-tide-gpu-05.sdsu.edu (process in D state on CephFS); average CPU
   0.12 cores of the 1 requested (below NRP's 20 % threshold for a non-exempt pod); cgroup peak memory 2996 MiB of 3 GiB (run 1: 2047 of 2 GiB),
   both equal to the limit, consistent with page cache filling the limit. A rerun fits 1 CPU / 2 GiB (exempt).
