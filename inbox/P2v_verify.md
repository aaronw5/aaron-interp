# P2v — verification of P2 (ParT: pair route vs per-particle route under a real mass change), 2026-10-17

Job `anrunw-ai-p2v-run` (Tesla V100, 323 s), `python -m interp.p2v_verify_part`; local CPU smoke run with 20 jets/class before it.
Script `interp/p2v_verify_part.py` (table: `python interp/p2v_verify_part.py --table`, rendered on the helper pod). Results
`results/part/p2v_verify.json`, table `results/site/T12b_part_routes_stats.png`, log `logs/p2v_cluster.log` (copies of
/j-jepa-vol/aaron-interp/{results/part,results/site,logs/p2v-run.log}). No training.

Reading under test: under a real mass change (angles $\times k$ or $p_T \times c$), the response of ParT_full and ParT_kin goes mostly
through the per-particle (axis-relative) inputs: per-particle route 93–101 % and pair route 1–6 % of the read-out $\Delta\log\hat m$; pair route
10–21 % of $\Delta$(W→Z) for mass increases and 26–42 % for decreases.

## Procedure
- Samples: rows 6,000–7,000 of each class's first test file (10,000 jets, primary, not used by P2) and P2's rows 3,000–4,000
  (reproduction), same code.
- Edits (`part_core.build`): angles $\times$ 0.9, 1.1, 1.25; $p_T \times$ 0.8, 1.25; extra $p_T \times 0.9$ (dose point for item 5). Controls:
  angles $\times$ 1.0, $p_T \times$ 1.0, and a rigid $\phi + 0.06$ shift whose variant (b) is the float32 noise floor of the pair route (pair
  features invariant in exact arithmetic; P1v variant (d)).
- Variants as P2: (a) full edit $(x_e, v_e)$; (b) pair route only $(x_0, v_e)$; (c) per-particle route only $(x_e, v_0)$.
- Measures. Primary (no read-out): $\Delta(\ell_W - \ell_Z)$ on true W jets, $\Delta$(W→Z), $\Delta$(Z→W). Secondary: $\Delta\log\hat m$ from P2's
  ridge read-out, refitted on the unedited class token of the fresh sample (train half, seed-1 split; $R^2$ 0.890 full, 0.924 kin) and
  P2's read-out (fitted on rows 3,000–4,000) applied unchanged ($R^2$ 0.891 / 0.925).
- Shares: sufficiency share $(b)/(a)$ and, because the routes need not add, the removal share $1 - (c)/(a)$ (drop of the response when
  the pair route is removed). Sum $[(b)+(c)]/(a)$.
- Intervals: 2,000 bootstrap resamples over jets, stratified by class (rng seed 2), percentile [2.5 %, 97.5 %].
- Tests. H0: pair share = 0 ($\bar b = 0$); H0': pair share = 1 ($\bar b - \bar a = 0$); two-sided bootstrap percentile p (floor
  $2/2001 = 0.001$) and normal p from the bootstrap s.e.; Holm over 30 share tests per model. McNemar (exact binomial) of (b) vs the
  unedited network on correctness, on W→Z (W jets), and on "call changed" in (b) vs in the noise control; Holm over 15 per model.
- Item 5: pair-feature distributions on 17.55 M real–real pairs ($i \ne j$) of edited vs unedited jets; the pair-embedding
  BatchNorm running statistics (training data); jet $p_T$ vs the JetClass range [500, 1000] GeV; causal split of (b) at $p_T \times c$:
  unedited 4-vectors with only $\ln k_T \mathrel{+}= \ln c$, only $\ln m^2 \mathrel{+}= 2\ln c$, or both (applied above the eps clamp; "both"
  agrees with (b) on ≥ 99.9 % (full) / 95.9–98.8 % (kin) of calls, c = 0.8, 0.9, 1.25).

## Reproduction
Rows 3,000–4,000 reproduce P2 to the printed digits (e.g. ParT_full angles $\times$ 0.9: 10.98 % calls changed, pair share of
$\Delta\log\hat m$ 0.012; ParT_kin $p_T \times 0.8$: pair share of $\Delta(\ell_W-\ell_Z)$ 0.87, of $\Delta\log\hat m$ −5.34). Fresh rows vs P2's point estimates (P2 value outside the fresh 95 % interval):
- Logit and $\Delta$W→Z shares: 3 of 40 outside (one by 0.001) — ParT_kin angles $\times$ 0.9 pair share of $\Delta(\ell_W-\ell_Z)$ 0.31 vs [0.21, 0.29]; its
  per-particle share of $\Delta$W→Z 0.88 vs [0.93, 1.11]; ParT_full $p_T \times 1.25$ per-particle $\Delta$W→Z 0.966 vs [0.967, 1.069].
- Read-out shares: 14 of 20 outside (e.g. ParT_full angles $\times$ 0.9 pair 0.012 vs [−0.006, 0.005]; $p_T \times 0.8$ 0.050 vs [0.032, 0.043]).
  The bootstrap resamples jets with the read-out held fixed; refitting the read-out on another 10,000 jets moves the read-out shares by
  up to 0.02 (outside $p_T$ decreases of ParT_kin), more than the jet-resampling interval. With P2's read-out applied to the fresh rows the
  ParT_full values are 0.009, 0.009, 0.015, 0.053, 0.045 (P2: 0.01, 0.02, 0.02, 0.05, 0.05). The read-out-share intervals in this
  report therefore understate the uncertainty; the read-out-fit spread (≈ 0.01–0.02) is the larger term.

## Results, fresh rows (estimate [2.5 %, 97.5 %])
| model | edit (true $\Delta\log m$) | $\Delta(\ell_W-\ell_Z)$ on W, (a) | pair (b)/(a), logit | $1-(c)/(a)$, logit | sum | pair (b)/(a), $\Delta$W→Z | pair (b)/(a), $\Delta\log\hat m$ | per-particle (c)/(a), $\Delta\log\hat m$ | (b) calls changed % |
|---|---|---|---|---|---|---|---|---|---|
| full | ang × 0.9 (−0.104) | +0.34 | 0.43 [0.38, 0.50] | −0.03 [−0.07, 0.01] | 1.45 | 0.38 [0.27, 0.51] | 0.00 [−0.01, 0.01] | 0.99 [0.99, 1.00] | 2.5 [2.2, 2.8] |
| full | ang × 1.1 (+0.094) | −1.37 | 0.13 [0.12, 0.14] | 0.18 [0.17, 0.19] | 0.95 | 0.11 [0.07, 0.15] | 0.00 [−0.01, 0.01] | 1.00 [1.00, 1.01] | 2.1 [1.9, 2.4] |
| full | ang × 1.25 (+0.221) | −2.30 | 0.20 [0.18, 0.22] | 0.03 [0.01, 0.05] | 1.17 | 0.25 [0.19, 0.31] | 0.01 [0.00, 0.01] | 1.01 [1.00, 1.02] | 5.6 [5.1, 6.0] |
| full | $p_T$ × 0.8 (−0.223) | +0.23 | 0.58 [0.46, 0.80] | −0.12 [−0.24, −0.06] | 1.71 | 0.40 [0.28, 0.53] | 0.04 [0.03, 0.04] | 0.96 [0.96, 0.97] | 3.2 [2.9, 3.6] |
| full | $p_T$ × 0.9 (−0.105) | +0.33 | 0.24 [0.22, 0.27] | 0.04 [0.02, 0.06] | 1.20 | 0.28 [0.17, 0.39] | 0.03 [0.02, 0.04] | 0.97 [0.96, 0.97] | 1.4 [1.2, 1.7] |
| full | $p_T$ × 1.25 (+0.223) | −2.24 | 0.12 [0.11, 0.13] | 0.05 [0.04, 0.06] | 1.07 | 0.11 [0.07, 0.15] | 0.04 [0.03, 0.04] | 0.94 [0.94, 0.95] | 3.5 [3.1, 3.9] |
| kin | ang × 0.9 | +0.38 | 0.25 [0.21, 0.29] | 0.00 [−0.04, 0.04] | 1.25 | 0.26 [0.16, 0.37] | 0.06 [0.05, 0.06] | 0.93 [0.92, 0.93] | 4.9 [4.5, 5.3] |
| kin | ang × 1.1 | −1.02 | 0.10 [0.09, 0.12] | 0.15 [0.14, 0.16] | 0.96 | 0.14 [0.10, 0.20] | 0.06 [0.05, 0.07] | 0.96 [0.95, 0.97] | 4.9 [4.5, 5.3] |
| kin | ang × 1.25 | −1.98 | 0.14 [0.13, 0.16] | 0.05 [0.03, 0.06] | 1.09 | 0.25 [0.19, 0.32] | 0.07 [0.06, 0.07] | 0.97 [0.97, 0.98] | 9.2 [8.6, 9.8] |
| kin | $p_T$ × 0.8 | −3.08 | 0.87 [0.86, 0.88] | 1.10 [1.08, 1.13] | 0.77 | 0.28 [0.21, 0.34] | −3.15 [−3.38, −2.94] | 3.77 [3.59, 3.98] | 62.0 [61.2, 62.7] |
| kin | $p_T$ × 0.9 | −0.30 | 1.45 [1.25, 1.80] | 2.15 [1.83, 2.70] | 0.30 | 2.14 [1.41, 5.60] | −0.14 [−0.16, −0.12] | 1.22 [1.20, 1.25] | 12.6 [12.0, 13.3] |
| kin | $p_T$ × 1.25 | −2.03 | 0.13 [0.12, 0.15] | 0.04 [0.03, 0.06] | 1.09 | 0.18 [0.12, 0.25] | 0.05 [0.04, 0.05] | 0.97 [0.96, 0.97] | 13.9 [13.2, 14.5] |

With P2's read-out applied unchanged, the pair share of $\Delta\log\hat m$ is 0.009–0.053 (full) and 0.055–0.063 (kin, without $p_T$ decreases);
with the refitted read-out 0.000–0.038 (full) and 0.048–0.069 (kin). The read-out share moves by up to 0.015 with the read-out fit.

Tests. Pair share = 0: rejected in every row of the logit and $\Delta$W→Z measures (bootstrap p ≤ 0.001, Holm ≤ 0.03); for $\Delta\log\hat m$ of
ParT_full not rejected at angles $\times$ 0.9 and 1.1 with the refitted read-out (p 0.91, 0.85), rejected with P2's read-out (p 0.002, 0.007).
Pair share = 1: rejected in every row (p ≤ 0.003). Per-particle share = 1 (logit): not rejected at ParT_full angles $\times$ 0.9 (p 0.13) and
ParT_kin angles $\times$ 0.9 (p 0.87); rejected elsewhere.

McNemar, (b) vs unedited, W→Z on W jets (gained / lost): ParT_full 1/31, 30/3, 89/10, 6/40, 45/7 (angles 0.9, 1.1, 1.25, $p_T$ 0.8, 1.25;
p 2e-8 to 6e-17, Holm ≤ 2e-6); ParT_kin 7/35, 43/6, 95/13, 291/154, 87/25 (p 2e-5 to 1e-16). Correctness: ParT_full p 0.022, 0.40, 7e-11,
0.007, 3e-4; ParT_kin p 0.92, 0.50, 8e-5, ≈0 (502/4,198), 1e-42. Calls changed in (b) vs the noise control (jets changed only in (b) /
only in the control): ParT_full 213/0 to 556/0; ParT_kin 451/20 to 6,151/13; all p < 1e-60.
Noise floor ((b) of the rigid $\phi + 0.06$ shift): ParT_full 0.00 % [0.00, 0.00] (rows 3,000: 0.01 %), ParT_kin 0.60 % [0.46, 0.75] (rows
3,000: 0.62 %) of calls; it changes no W→Z call beyond 3/3 (kin) and correctness 36/13 (kin, p 0.0014). Factor-1.0 controls: 0.00 % calls
changed, max $|\Delta L|$ 5.7e-6 (full, rows 3,000) / 0 otherwise.

## Item 5 — ParT_kin $p_T \times 0.8$
- Pair-feature ranges: the shift is uniform, $\Delta\ln k_T = \ln 0.8 = -0.223$, $\Delta\ln m^2 = -0.446$, $\ln z$ and $\ln\Delta R$ unchanged. No
  edited pair lies below the unedited minimum or above the maximum of any feature; 0.16 % of edited $\ln k_T$ values lie below the
  unedited 0.1 % quantile (0.17 % for $\ln m^2$). In units of the pair BatchNorm's running s.d. (training data: $\ln k_T$ 3.41, $\ln m^2$
  3.71), the shifts are 0.065 and 0.120 s.d. The opposite shift ($p_T \times 1.25$: $\ln m^2$ +0.446) gives W→H→bb 0.004 in (b).
- Jet $p_T$: 63.5 % of $p_T \times 0.8$ jets fall below 500 GeV. (b) sends 0.49 of W jets to H→bb when the scaled jet stays in
  [500, 1000] GeV (397 W jets) and 0.54 when it falls below (603); (a) 0.15 / 0.19. The effect does not depend on leaving the jet-$p_T$ range.
- Causal split (unedited 4-vectors, fresh rows): $\ln m^2 - 0.446$ alone: 63.9 % calls changed, W→H→bb 0.53, $\Delta(\ell_W-\ell_Z)$ −2.95;
  $\ln k_T - 0.223$ alone: 4.8 %, 0.004, −0.06; both: 62.0 %, 0.52, −2.67 (= (b)). Dose: $\ln m^2 - 0.21$ ($p_T \times 0.9$): W→H→bb 0.05,
  16.4 % changed. ParT_full under the same $\ln m^2$-only shift: 10.5 % changed, W→H→bb 0.003.
- (b) is off the joint input distribution in every row (pair features inconsistent with the unedited per-particle $p_T$, $E$, $\Delta R$);
  for $p_T \times c$ the inconsistency is between pair $\ln m^2$ and per-particle pt_log / e_log.
- The full edit (a) at $p_T \times 0.8$ is itself dominated by the pair route: (a) moves $\Delta(\ell_W-\ell_Z)$ by −3.08 (toward Z, for a mass
  decrease), (c) by +0.31, (b) by −2.67; $\Delta\log\hat m$ (a) −0.044 vs true −0.223. The same sign pattern holds at $p_T \times 0.9$ ((a) −0.30,
  (b) −0.44, (c) +0.35) and on rows 3,000–4,000.
- Exclusion: the row is not an input-range artefact by marginal pair-feature range or jet-$p_T$ range, and it is reproduced on two row sets
  and at $c = 0.9$. Its ratio shares are not a decomposition (sum 0.77 logit, 0.63 read-out; read-out denominator 0.2 of the true change),
  so it is left out of the pooled share ranges below and reported as a separate result, not dropped.

## Corrected reading (fresh rows; these edits, read-outs, jets)
- Read-out $\Delta\log\hat m$ (secondary): per-particle route alone 0.94–1.01 (ParT_full), 0.93–0.97 (ParT_kin); pair route alone 0.00–0.04
  (full; 0.009–0.053 with P2's read-out) and 0.05–0.07 (kin); ParT_kin $p_T$ decreases excepted.
- Logits (primary): for mass increases the per-particle route alone gives 0.82–0.97 of $\Delta(\ell_W-\ell_Z)$ on W jets, the pair route alone
  0.10–0.20, and removing the pair route lowers the response by 0.03–0.18; the two add to 0.95–1.17. For mass decreases (angles $\times$ 0.9,
  ParT_full $p_T \times$ 0.8 / 0.9) the per-particle route alone gives 0.96–1.12, the pair route alone 0.24–0.58, the sum is 1.20–1.71, and
  removing the pair route changes the response by −0.12 to +0.04: the pair route alone moves the logits, but in the full edit its effect
  is not additive and its removal leaves the response unchanged within 0.12.
- $\Delta$W→Z, pair route alone: 0.11–0.25 at increases (intervals 0.07–0.32), 0.26–0.40 at decreases (intervals 0.16–0.53).
- ParT_kin, $p_T \times$ 0.8 and 0.9: the response goes mainly through the pair route (pair alone 0.87 and 1.45 of the logit change,
  per-particle alone −0.10 and −1.15); the cause is the $\ln m^2_{ij}$ shift with the per-particle inputs fixed.
