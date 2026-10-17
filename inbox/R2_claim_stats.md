# R2 — hypothesis tests and 95 % intervals for A1, A2, A3, A9, B2, B3 and the small taggers (agent report, 2026-10-17)

Script `interp/r2_stats.py` (`python -m interp.r2_stats --part <a1units|a1float|a2|a3|a9|b2|b3|small|table>`). Outputs
`results/stats/r2_{a1units,a1float,a2,a3,a9,b2,b3,small}.json`, `r2_a1units_vs_e1.json`, `r2_table_rows.json`; table
`results/site/T14_claim_stats.png`. Cluster jobs `r2-<part>` (logs `/j-jepa-vol/aaron-interp/logs/r2-<part>.log`; smoke runs `r2s-<part>`).
Sizes: 1 CPU / 2 GiB, except a3, b2, b3 at 1 CPU / 4 GiB (the 2 GiB smoke runs of a3 and b3 reached the limit and ran at 0.12 cores; at
4 GiB they peaked at 1.2–1.45 GiB, 0.97–0.99 cores). Nothing was trained; nothing refit except the A9 ridge probes.

## Sample and methods
- A1, A2, A3, A9: `jedi_core.sample(4000, seed=61)`, 20,000 jets. Overlap with earlier samples: E1 791, E4/E10 1,564, E11 fit+dev 2,297,
  E5v 805, E11v fresh 1,601 jets.
- B2: `sample(5000, 61)` minus every row of `sample(6000, 0)` (E11 fit/dev), `sample(2000, 11)`, `sample(4000, 31)`: 19,683 jets. B3: 2,000
  per class from these (rng 61).
- Small taggers: all 260,000 test jets; JEDI-linear calls from `jedi_core.Jedi(64)` (80.938 %; labels identical to `val_meta.h5`).
- Intervals 95 %. Bootstrap = resampling within each true class, 2,000 replicates (A3 objects 10,000); plain bootstrap for the rejection
  ratios (as E11v) and the 260,000 test jets. Wilson for proportions. McNemar = exact binomial test on the discordant pairs. Holm within the
  stated family. Verdicts at $\alpha = 0.01$.
- The network is deterministic: rerunning the same jets changes 0 of 20,000 calls (max |Δlogit| 0). Any edit with $k \ge 1$ changed calls
  rejects "the edit changes no call" with $p = 0$; the tests with content are against thresholds or control edits.

## A1 (units that matter)
Procedure: every unit of every stage whose replacement by its mean (over real particles, or over jets for pooled stages) changes some slot
value (186 units; the other units' replacement is the identity, 0 calls by construction; two per stage run: 0). Metric: fraction of calls
changed. Tests: H0 cost ≤ 1 % (and ≤ 0.1 %) vs H1 cost > threshold, and the reverse, exact one-sided binomial, Holm over the 186 units.
Control edit: resample ablation (the unit's value from another jet, one fixed derangement), McNemar per unit.

| stage | a | P | g | s | u | S | h4 | h5 | z |
|---|---|---|---|---|---|---|---|---|---|
| point count > 1 % | 13 | 10 | 7 | 9 | 19 | 19 | 40 | 29 | 14 |
| > 1 %, Holm 0.01 | 13 | 10 | 5 | 9 | 18 | 19 | 30 | 24 | 13 |
| < 1 %, Holm 0.01 | 0 | 1 | 1 | 0 | 0 | 0 | 8 | 2 | 2 |
| unresolved | 0 | 1 | 5 | 0 | 1 | 0 | 17 | 6 | 1 |
| > 0.1 %, Holm 0.01 | 13 | 12 | 11 | 9 | 19 | 19 | 53 | 32 | 16 |

- Unresolved at 1 %: P3 (0.95 %), g29, g32, g36, g44, g63 (0.77–1.26 %), u20 (1.17 %), 17 h4 units (0.78–1.19 %), h5:5, 6, 12, 27, 28, 30
  (0.90–1.26 %), z3 (1.03 %). Not > 0.1 %: h4:8 (0.135 %), h4:33 (0.045 %).
- Stated E1 counts (10,000 jets): P 11, g 7, h4 42, h5 30, z 14. E11 live set (> 0.1 %): P 12, g 11, S 19, h4 54, h5 32, z 16.
- Seed-61 vs E1 seed-0 cost per unit (two-proportion z, Holm over 186): 0 units differ at 0.05.
- Examples: a32 63.10 % [62.52, 63.70] (E1 64.2 %); S29 40.82 % [40.21, 41.42]; h4:49 19.63 % [19.10, 20.11].
- Mean vs resample ablation: 132 of 186 units differ (Holm 0.01), resample larger in 109; Spearman between the two costs 0.96.
  a32 63.1 vs 34.2 %; P15 8.3 vs 13.1 %; S0 12.2 vs 20.7 %; h4:49 19.6 vs 30.0 %.

## A1 (float + zero-bit masks vs fixed point)
E3m forward passes on the 20,000 jets (`e3m_masks` "mask" mode; `e4_e5_jedi.make_masked` gives identical calls).
- Same call: 95.58 % [95.29, 95.86] (4.42 % changed; stated 4.49 %). Fully float: 48.04 % same call, 43.52 % accuracy.
- Accuracy: fixed 80.66 %, float + masks 80.36 %; difference −0.29 [−0.55, −0.03] points; McNemar 335 vs 394, $p = 0.032$.
  Stated (E3m, 10,000 jets): 80.68 vs 81.29 % (−0.61).

## A2 (context)
- g → mean: 14.78 % [14.29, 15.28] of calls changed; accuracy 80.66 → 74.37 % (−6.29 [−6.77, −5.82]). With the E4-sample mean: identical calls.
  g → 0: 52.47 %; g → another jet's g: 27.94 %. Single g units: g49 10.62 % [10.21, 11.06], g4 4.28 %, g55 2.17 %, g39 1.72 %, g5 1.38 %, g63 1.26 %.
- S0, S30, S31, S58 → their mean-context values: 14.16 % [13.68, 14.65]; same call as g → mean on 98.08 % of jets and on 90.8 %
  [89.7, 91.8] of the 2,956 jets g → mean changes. All 19 live S: 14.78 %.
- Null 1 (H0: the four are exchangeable with any four of the other 15 live objects; all 1,365 subsets): cost mean 0.34 %, max 1.01 %;
  $p = 1/1366 = 7.3\times10^{-4}$ (the smallest attainable). Agreement with g → mean: null max 4.2 %, same $p$. On the 18,436 jets not in the
  E4 selection sample: 14.14 % vs null max 1.01 %.
- Null 2 (all 3,876 4-subsets of the 19 live S): 3 subsets change more calls — {S0, S17, S30, S31} 14.26 %, {S0, S13, S30, S31} 14.24 %,
  {S0, S30, S31, S43} 14.18 % — rank 4, $p = 0.00103$. On agreement with g → mean's calls the four are first of 3,876 ($p = 2.6\times10^{-4}$).

## A3 (rotation)
Edits on the 20,000 jets: rotations by 45°, 90°, 180°, reflections $\Delta\phi \to -\Delta\phi$, $\Delta\eta \to -\Delta\eta$ (every r unchanged),
per-particle azimuth scramble. Controls: rerun unedited (0 calls); matched per-unit spread — each live S unit shifted by Gaussian noise
with the edit's per-unit rms, or by its own edit shift taken from another jet (S clipped at 0), 10 draws each, patched into the network after S.

| edit | calls changed | masked float | Gaussian, matched rms | own shift, permuted |
|---|---|---|---|---|
| 45° | 4.01 % [3.75, 4.30] | 3.06 % | 9.28–9.99 % | 8.28–8.97 % |
| 90° | 4.13 % [3.86, 4.41] | 3.49 % | 9.51–10.30 % | 8.37–8.94 % |
| 180° | 3.94 % [3.68, 4.22] | 3.01 % | 7.79–8.34 % | 6.87–7.42 % |
| φ flip | 3.78 % [3.52, 4.05] | 2.96 % | 8.20–8.85 % | 7.37–7.87 % |
| η flip | 4.03 % [3.76, 4.31] | 3.49 % | 9.25–9.95 % | 8.24–8.84 % |
| scramble | 6.89 % [6.55, 7.25] | 6.35 % | 10.04–10.71 % | 8.51–8.95 % |

- Each edit vs its matched-noise draw: McNemar $p < 10^{-17}$ (fewer calls change under the edit). H0 rate ≤ 3 %: $p \le 2.7\times10^{-10}$ for
  every r-preserving edit. Cochran's Q across the five r-preserving edits: $p = 0.147$.
- Carriers (one rotated $S_c$ patched in, 180°): S1 2.51 %, S29 2.12 %, S56 1.73 %, S31 1.33 %, S17 1.08 %.
- Per-object rms $\Delta S$/SD($S$) (bootstrap 10,000; H0 ≤ 0.1 vs > 0.1 and H0 ≥ 0.1 vs < 0.1; Holm over 19 objects, 0.01):
  - 180°: < 0.1 for S0, S19, S20, S29, S30, S40, S43, S50, S51, S60 (the ten in A3); > 0.1 for S1, S13, S17, S31, S52, S56, S58, S62, S63.
  - Largest of the five r-preserving edits: < 0.1 only for S0, S19, S20, S29, S51; > 0.1 for 13 objects including S30 (0.116 at 90°),
    S40 (0.18–0.19 at 45°, 90°, both reflections; 0.075 at 180°), S43 (0.12–0.15), S50 (0.11); S60 0.101 at 90°.
  - Masked float, largest of five: < 0.1 for S0, S19, S20, S29, S30, S51, S60; S40 0.179 and S43 0.117 at 45° (above 0.1 without rounding).

## A9 (layers)
E10's ridge read-outs refit on the seed-61 jets (fixed 50/50 split, rng 0). First stage at R² ≥ 0.9 and ≥ 0.7 identical to seed 0 for
all 45 observables.
- R² ≥ 0.9: Spearman $\rho = 0.507$; 95 % interval over observables [0.256, 0.717], over held-out jets with probes fixed [0.491, 0.519],
  both levels [0.267, 0.708]; permutation of depth labels (10,000): one-sided $p = 5.0\times10^{-4}$, two-sided $6.0\times10^{-4}$.
  Without slot-order quantities: 0.561 [0.292, 0.785], $p = 3\times10^{-4}$.
- R² ≥ 0.7: $\rho = 0.383$ [0.099, 0.628]; one-sided $p = 0.0049$, two-sided 0.0095.
- Median first stage by depth 0/1/2/3 at 0.9: S/S/never/never (never: 4/5/7/7 of 14/14/8/9).

## B2 (program)
Stored E11 formulas on the 19,683 fresh jets (network 80.80 %).

| program | same call | accuracy − network | McNemar p | W-vs-rest rejection ratio |
|---|---|---|---|---|
| nested (b) | 94.10 % [93.76, 94.42] | −0.04 [−0.35, 0.26] | 0.845 | 1.13 [0.96, 1.42], p 0.165 |
| nested + half-planes | 94.60 % [94.28, 94.91] | +0.03 [−0.26, 0.30] | 0.865 | 1.09 [0.93, 1.30], p 0.313 |
| z on physics (direct) | 92.72 % [92.35, 93.07] | −0.27 [−0.63, 0.04] | 0.118 | 1.18 [0.94, 1.52], p 0.186 |

- Rejection at 50 % efficiency, 2,000 plain bootstrap resamples: network W rejection 263 (60 background jets pass); E11v (17,011 other jets)
  had 453 and ratio 0.70 [0.50, 0.95]. Other tasks for the nested program: g 0.96 [0.87, 1.06], q 0.92 [0.88, 1.00] (p 0.043), t 0.95
  [0.79, 1.12], Z 0.40 [0.14, 0.86] (6 background jets pass for the network, p 0.02); W and Z vs QCD: 1–5 background jets pass.
- Per class accuracy, nested − network: g −0.51, q −0.94 (p 0.010), W −0.13, Z +0.59, t +0.80 (p 0.024) points.

## B3 (agreement under edits)
10,000 jets; physics inputs recomputed from the edited particles. Same call with the network, nested / nested + half-planes / direct:
- unedited: 94.2 / 94.7 / 92.7 %; nested − direct +1.5 [1.1, 2.0] points, $p = 9\times10^{-10}$ (on the 19,683 B2 jets: +1.38 [1.05, 1.73], $p = 1.7\times10^{-15}$);
- η shift 0.06 rad (invariant mass unchanged): 74.4 [73.5, 75.2] / 81.1 / 51.5 [50.5, 52.5] %; nested − direct +22.9 [21.8, 24.0], $p = 10^{-276}$;
- pT × 0.8: 88.7 / 90.5 / 84.0 %; nested − direct +4.7 [4.0, 5.4], $p = 3\times10^{-35}$.
- W → Z under the shift: network 76.4 % [74.5, 78.2], nested 78.1 % (difference +1.7, McNemar $p = 0.080$), direct 0.0 %; unedited 2.9 / 3.6 / 2.3 %.

## Small taggers (N = 64, 260,000 test jets)
| run | accuracy [95 %] | − JEDI-linear (80.938 %) | McNemar p |
|---|---|---|---|
| linf-2b-d32-k8 seed 0 | 81.065 [80.912, 81.216] | +0.127 [0.033, 0.220] | 0.0096 |
| linf-2b-d32-k8 seed 1 | 80.763 [80.612, 80.918] | −0.175 [−0.273, −0.084] | 4.0e-4 |
| dsets-d64 seed 0 | 80.642 [80.486, 80.788] | −0.296 [−0.389, −0.197] | 2.7e-9 |
| dsets-d64 seed 1 | 80.815 [80.668, 80.962] | −0.124 [−0.218, −0.025] | 0.0115 |

- Seed 1 − seed 0 on the same jets: Linformer −0.302 [−0.382, −0.230] ($p = 6\times10^{-15}$); Deep Sets +0.172 [0.093, 0.252] ($p = 2.4\times10^{-5}$).
  Every architecture's two seeds differ at $p < 0.05$ except tf-2b-d32 (seed 0 vs the patience-30 seed 1, $p = 0.27$).
- Mean of the two seeds − JEDI (test-set bootstrap only): Linformer −0.024 [−0.115, 0.061]; Deep Sets −0.210 [−0.296, −0.123].
  Two-seed t interval (1 df, seed spread only): Linformer [79.00, 82.83] %, Deep Sets [79.63, 81.82] %; both contain 80.94 %.
- The other N = 64 runs (tf-*, linf-1b) are 3.2–11.8 points below JEDI-linear, $p < 10^{-300}$. The axhard runs (E18 inputs) are in the JSON.
  The 100-epoch runs had no test_probs.npy at run time.

## Effects not significant, or intervals that change the stated reading
1. B2 "W rejection 0.70× the network's": on these jets 1.13× [0.96, 1.42], $p = 0.165$; the E11v interval [0.50, 0.95] and this one do not overlap.
   No rejection ratio differs from 1 at 0.01.
2. A3 "ten objects move ≤ 0.1 SD under every r-preserving edit": holds at 180° only; under the largest of the five edits five objects
   (S0, S19, S20, S29, S51) are < 0.1 SD; S40, S43, S50 are > 0.1 SD at 45°/90°/reflections (S40, S43 also in masked float).
3. A2: the four-object count does not single out S58; {S0, S30, S31} with S17, S13 or S43 change as many calls (14.18–14.26 %). The four
   objects rank first only on agreement with g → mean's calls.
4. A1 float + masks: accuracy cost −0.29 [−0.55, −0.03] points, $p = 0.032$ (stated −0.61).
5. A1 counts at 1 %: g 5, h4 30, h5 24 units are significantly above 1 % (point counts 7, 40, 29); 17 h4 and 6 h5 units are unresolved.
   The 0.1 % counts (E11 live set) are resolved (h4 53 of 54).
6. Small taggers: the sign of Linformer − JEDI depends on the seed; the seed-to-seed difference (0.30 points) exceeds the test-set interval.
   Deep Sets seed 1 − JEDI: $p = 0.0115$.
7. B3: nested-program W → Z vs network, $p = 0.080$.
8. A9 at R² ≥ 0.7: two-sided $p = 0.0095$; lower interval bound 0.10.

Note: `interp/r2_stats.py` was included in commit 0081f53 by another process while this run was in progress; this agent made no commits.
