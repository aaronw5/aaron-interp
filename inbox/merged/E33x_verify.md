# E33x verification: recurrence of the squared-radius unit on fresh jets, with decoys of matched complexity

**Conclusion.** On a new set of jets, a second-round unit that tracks the momentum-weighted squared distance of particles from the jet axis (Σ pT r²) again appears in 8 of the 9 other trained networks, and E33x's comparison with random one-term formulas gives the same p-value (0.014). The E33x random formulas have one fitted term and the squared-radius formula has three. Against random three-term formulas the p-value is 0.011–0.046 depending on the formula set and the matching rule, so the recurrence is above chance at roughly the 1–5 % level, not below 1 %. The radial-move test shows a per-particle weight close to pT r² only for particles within r ≈ 0.2 of the axis, in the 7 quantized 3-feature networks. A single power r^β does not describe any network over the full range, and the fitted β moves by up to 0.10 between jet samples.

## What was checked
1. **Code review** of `interp/e33x_recurrence_null.py` and the E33 helpers it calls (`e33_program_recurrence_jedi`: `load`, `particles`, `role_feats`, `fit_set`, `single_fit`, `Model`, `to16`, `z_windows`).
2. **Rerun on fresh jets** (cluster job `e33x-verify`, 1 CPU / 2 GiB, 1,117 s, peak 2,047 MiB including page cache). Fit jets = `jedi_core.sample(2000, 5501)` and test jets = `sample(2000, 5502)`, 10,000 each (E33x used 4401/4402). The E33x routine `fits()` was rerun unchanged on all 10 trained networks, with the same 603 decoys.
3. **Missing control: decoys of matched complexity.** The r2 formula is a least-squares fit on three regressors. The E33x decoys are fitted on one regressor (the best of n rank windows). A three-regressor formula reaches R² ≥ 0.9 more easily, so the E33x null can be too easy to beat. Two three-regressor decoy sets were fitted exactly as the r2 formula (fit on [1, F1, F2, F3]).
4. **Ablations of the r2 formula.** Each sub-formula or analogue was fitted with the same procedure.
5. **Radial weight β** on fresh jets, with the goodness of fit of the power law.
6. **Untrained float redraws** on fresh jets.
7. **Smoke run** `e33x-verify-smoke2` (1,000 jets). The first smoke job (`e33x-verify-smoke`) was lost when node k8s-haosu-21.sdsc.optiputer.net became unreachable (process in D state, `folio_wait_bit_common`). That node is not in job.sh's exclusion list; I did not edit job.sh.

Script: `interp/e33x_verify.py`; figure script: `interp/e33x_verify_plot.py`. Results: `results/jedi/e33x_verify.json` (smoke: `e33x_verify_smoke.json`). Figure: `results/jedi/e33x_verify.png`. Log: `logs/e33x-verify.log`.

## Terms
- **S unit**: one of the 64 second-round pooled objects S = Σ_i u_i / 16 (the sum over particles of the second-round per-particle statements u). A unit is *live* if its s.d. over the fit jets is > 1e-9.
- **r2 formula**: S ≈ a + b Σ pT r² + c Σ r² + d N_soft. Here r = √(Δη² + Δφ²) is a particle's distance from the jet axis, and N_soft = number of particles with pT < 2 GeV and r > 0.1. It has three regressors.
- **Recurrence count**: the number of the 9 other trained networks in which the formula's unit reaches held-out (test-jet) R² ≥ 0.9. A formula *qualifies* if it reaches R² ≥ 0.9 in the reference (official 3-feature n64).
- **Hungarian (one unit per formula)**: the 4 named formulas (hard, r2, soft, core) are assigned to distinct units by maximising the summed fit R². Each decoy enters as a 5th row. **Unassigned**: the best unit by fit R², with no exclusivity.
- **p (decoy test)**: (1 + number of qualifying decoys whose recurrence is ≥ the r2 formula's) / (1 + number of qualifying decoys). H0: the r2 formula recurs no more often than a qualifying decoy. H1: it recurs more often.
- **D3h**: 300 decoys, each the linear fit on three random per-particle hinges summed over all real particles, F = Σ_i ReLU(cos θ z_pT,i + sin θ z_r,i + b). Here θ ~ U(0, 2π), b ~ N(0, 1) and z = pT or r standardised over the fit jets' particles (rng 4477). These are three-regressor analogues of the E33x hinge decoys.
- **D3L**: 300 decoys, each the linear fit on three random sums over all particles of the E33 feature library. The library terms are count, pT, ln pT, pT², r, r², pT r, pT r², ln pT·r, soft-wide count, (r − t)₊, (t − r)₊, 1[pT < t], (pT − t)₊, (t − pT)₊, 1[r > t] and ln pT (r − t)₊. The exact r2 triple is excluded (rng 4478). These are physics-like features of the same complexity as r2.
- **γ_b**: the response of the unit to a radial move, per unit of r and per GeV of the moved particle's pT, fitted in r bin b (E33x definition). A weight pT r^β gives γ_b ∝ β r_b^(β−1); for β = 2, γ_b / r_b is constant.
- **χ²/dof** of the β fit: the weighted residual sum (weights 1/se_b²) divided by (number of bins − 2). A value ≈ 1 means the power law describes the bins within their errors.
- **Exact McNemar**: a two-sided binomial test on the networks where exactly one of two formulas reaches R² ≥ 0.9.

## Code review
- **Stages and quantizers.** S is taken after the `qs1` pooling quantizer, through the same `Model` forward as E33. Live units and the fit/test split are as documented. Fit and test are separate `sample()` draws with different seeds from the same test file. They can share some jets (expected share ≈ 2,000 / (jets per class), about 1 %; not measured). I did not find a bug that changes the reported numbers.
- **Padded slots.** Hinge decoys, role features and competitors multiply by the real-particle mask (pT > 0). The soft formula counts empty slots on purpose, as documented. The fixed decoys `sum_r_prefix` and `sum_pT2_prefix` are zero on padded slots because r and pT are 0 there.
- **Edits.** The radial move rescales (Δη, Δφ) of one particle, so pT, rank and azimuth are kept. The azimuthal control rotates the particle about the axis at fixed r. In the 16-feature network the dependent columns are recomputed by `to16`. The pT-add keeps the rank because δ ≤ 0.9 × the gap to the next-harder particle. No edit changes more than the intended particle.
- **Statistical design.** The decoy null is not matched in complexity: the r2 formula has three regressors and the decoys have one. This is the control added below.
- **Reporting errors.** The E33x report speaks of "the 8 quantized 3-feature networks". There are 7 (official n32/n64/n128, perm-inv, HGQ s0–s2). The thesis note's "β close to 2 in 8 of 10 networks" includes float s0, whose radial response is not r²-shaped (see below).
- **β confidence interval.** The interval resamples particles within bins on one jet sample. On fresh jets β moved outside it in two networks (below).

## Numbers (fresh jets 5501/5502, N = 10,000 test jets)
**Accuracy.** Reference 80.94 % (E33x sample: 80.43 %). Other networks: n32 78.19, n128 81.28, perminv 82.12, 16f 81.74, HGQ s0 77.49, s1 56.49, s2 78.80, float s0 82.66, float s1 82.56 %.

**Replication of E33x (same 603 one-regressor decoys, threshold 0.9).**
- r2: reference test R² 0.977, unit S29 again; recurrence 8/9.
- Per-network R² (Hungarian): n32 0.964, n128 0.920, perminv 0.959, 16f 0.937, HGQ s0 0.954, s1 0.938, s2 0.957, float s0 0.897, float s1 0.904. These are the same networks as in E33x, and float s0 misses the cut again (0.897).
- Qualifying decoys: 145/603 (Hungarian), mean recurrence 2.63. Decoys ≥ 8: 1, so **p = 0.014** (E33x 0.014).
- Unassigned: 206 qualify, 1 ≥ 8, **p = 0.0097** (E33x 0.0095).

**Matched-complexity control (three-regressor decoys, threshold 0.9).**

| null set | matching | qualifying | mean recurrence (of 9) | decoys with ≥ 8 | p for r2 (8/9) |
|---|---|---|---|---|---|
| E33x hinges (1 regressor) | Hungarian | 145 / 603 | 2.63 | 1 | 0.014 |
| E33x hinges (1 regressor) | unassigned | 206 / 603 | 3.18 | 1 | 0.0097 |
| D3h random hinge triples | Hungarian | 180 / 300 | 3.00 | 1 | 0.011 |
| D3h random hinge triples | unassigned | 239 / 300 | 4.72 | 10 | 0.046 |
| D3L library triples (all) | Hungarian | 205 / 300 | 3.61 | 6 | 0.034 |
| D3L library triples (all) | unassigned | 260 / 300 | 4.86 | 21 | 0.084 |
| D3L triples without Σ pT r² | Hungarian | 182 | 3.29 | 2 | 0.016 |
| D3L triples without Σ pT r² | unassigned | 237 | 4.57 | 5 | 0.025 |

- The 23 qualifying D3L triples that contain Σ pT r² as one of their three terms recur in a mean of 6.2 (Hungarian) or 7.8 (unassigned) of 9 networks. The triples without it recur in 3.3 or 4.6. Of the 6 D3L triples with ≥ 8 (Hungarian), 4 contain Σ pT r². The other 2 contain Σ pT r together with ln pT·(r − 0.1)₊, a pT-weighted radial ramp.

**Dependence on the R² cut (Hungarian; p against E33x hinges / D3h / D3L).**
- Cut 0.85: r2 recurs 9/9; p = 0.0068 / 0.037 / 0.14.
- Cut 0.90: r2 recurs 8/9; p = 0.014 / 0.011 / 0.034.
- Cut 0.95: r2 recurs 4/9; p = 0.027 / 0.13 / 0.047.
- Unassigned at 0.85: p = 0.0056 / 0.075 / 0.17.

**Ablations of the r2 formula (threshold 0.9; recurrence of 9, Hungarian / unassigned; exact McNemar p against the full r2 formula).**
- Σ pT r² alone (one regressor): ref 0.934; 6 / 7; McNemar p 0.50 / 1.0.
- [Σ pT r², Σ r²]: ref 0.940 / 0.958; 7 / 8; p 1.0.
- [Σ r², N_soft] (no pT weighting): ref 0.909 / 0.920; 1 / 1; p 0.016 / 0.016.
- [Σ pT r, Σ r, N_soft] (β = 1 analogue): ref 0.921 / 0.959; 2 / 4; p 0.031 / 0.125.
- [Σ pT r³, Σ r³, N_soft] (β = 3 analogue): ref 0.916 / 0.924; 5 / 6; p 0.375 / 0.625.
- The pT-weighted r² term carries the recurrence. At the level of jet-formula fits, an r² weight is preferred over an r weight only with Hungarian matching, and is not distinguished from an r³ weight.

**Radial weight on each network's r2 unit (fresh jets; ±0.01 radial move of one particle at fixed pT, rank and azimuth; 500-draw bootstrap).**

| network | β fresh [95 % CI] | β E33x [95 % CI] | χ²/dof of power law (fresh / E33x) | γ_b fresh, bins 0.015–0.05 … 0.3–0.4 |
|---|---|---|---|---|
| official n64 | 1.82 [1.74, 1.90] | 1.90 [1.83, 1.96] | 24 / 25 | 0.018, 0.036, 0.056, 0.082, 0.069, −0.001 |
| n32 | 1.62 [1.57, 1.68] | 1.72 [1.67, 1.77] | 109 / 139 | 0.025, 0.050, 0.071, 0.087, 0.028, 0.001 |
| n128 | 1.87 [1.83, 1.92] | 1.89 [1.85, 1.94] | 1210 / 1592 | 0.015, 0.042, 0.063, 0.090, 0.083, −0.112 |
| perminv | 1.97 [1.93, 2.02] | 2.01 [1.96, 2.05] | 303 / 386 | 0.010, 0.031, 0.049, 0.062, 0.087, 0.083 |
| 16f | 1.25 [1.21, 1.28] | 1.26 [1.23, 1.30] | 482 / 664 | 0.100, 0.164, 0.164, 0.178, 0.180, 0.108 |
| HGQ s0 | 2.07 [2.00, 2.16] | 2.15 [2.08, 2.22] | 16 / 18 | 0.013, 0.027, 0.047, 0.068, 0.115, 0.165 |
| HGQ s1 | 1.85 [1.79, 1.93] | 1.92 [1.87, 1.98] | 25 / 19 | 0.023, 0.040, 0.063, 0.084, 0.127, 0.178 |
| HGQ s2 | 2.00 [1.93, 2.07] | 2.06 [2.00, 2.12] | 45 / 32 | 0.014, 0.026, 0.044, 0.066, 0.106, 0.150 |
| float s0 | 1.83 [1.80, 1.86] | 1.87 [1.84, 1.90] | 1597 / 1899 | 0.002, 0.011, 0.014, 0.013, 0.015, 0.018 |
| float s1 | 1.45 [1.42, 1.49] | 1.50 [1.47, 1.54] | 955 / 1203 | 0.005, 0.010, 0.016, 0.017, 0.013, 0.011 |

- **Fit quality.** The power law has χ²/dof of 16–1,600 in every network, so a single r^β does not describe the per-bin responses within their errors in any network. β is set mostly by the small-r bins, which have the smallest errors.
- **Stability across samples.** β decreased in all 10 networks on the fresh sample (by 0.01–0.10). n32 went from 1.72 to 1.62, outside both intervals, so the reported intervals do not cover sample-to-sample variation. The range over the 7 quantized 3-feature networks is 1.62–2.07 (E33x 1.72–2.15).
- **Where the r² shape holds.** For r < 0.2, γ_b / r_b is constant within ±25 % in the 7 quantized 3-feature networks, as a pT r² weight predicts (figure, right).
  - Beyond r = 0.2 the shapes differ. The reference, n32 and n128 drop to about 0 (n128 to a negative value at 0.3–0.4), perminv flattens, and HGQ s0–s2 keep rising.
- **Float networks.** float s0's γ_b is flat for r ≥ 0.07 (0.011–0.018), which is a weight closer to pT r¹ there. Its β of 1.83 comes from the first two bins.
- **Azimuthal control.** The azimuthal / radial response ratio is reproduced: 0.53 (ref), 0.24–0.55 in the 3-feature networks, 0.004 in 16f.

**Untrained float redraws (fresh jets; r2 R², Hungarian).**
- float s0 u0 0.821, u1 0.917, u2 0.917; float s1 u0 0.813, u1 0.743, u2 0.906. That is 4 of 6 ≥ 0.9 (E33x: 3 of 6).
- The trained float networks score 0.897 and 0.904, so in the float architecture the trained r2 fit is at the level of random redraws.

## Competing hypotheses
- **H1 (learned observable).** A unit computing approximately Σ pT r² (plus Σ r² and a soft count) recurs because training selects it. Prediction: recurrence above formulas of equal complexity, and the r² weight specifically. Observed: above all three null sets at p = 0.011–0.046 (cut 0.9). The pT r² term is needed (dropping the pT weighting: 1/9). r² is preferred over r with Hungarian matching (2/9, p 0.031) but not over r³ (5/9, p 0.375).
- **H0a (input format / any three-term radial formula).** Prediction: equally complex random formulas recur as often. Excluded at about the 1–5 % level, not at 1 %.
- **H0b (β 0.3–0.7, E33v slopes).** Excluded: β ≥ 1.25 in every network on both samples.
- **H0c (architecture alone, untrained).** Not tested for the quantized networks, because untrained quantized redraws have almost no live S units. In the float architecture it is not excluded: 4/6 redraws fit r2 at R² ≥ 0.9, against 1/2 trained.

## Corrected reading
- **Recurrence.** A unit fitting a + b Σ pT r² + c Σ r² + d N_soft recurs in 8 of 9 other trained networks on two independent jet samples. Against one-regressor rank-window hinge decoys, p = 0.014 (one unit per formula) or 0.0097 (unassigned) on both samples. Against three-regressor decoys of the same complexity, p = 0.011–0.046 (random hinge triples) or 0.016–0.025 (library triples without Σ pT r²). The p-value also depends on the R² cut (0.007–0.17 between cuts 0.85 and 0.95).
- **Which term recurs.** The pT-weighted r² term carries the recurrence: dropping the pT weighting gives 1/9, and three-term library formulas containing Σ pT r² recur in 6.2–7.8 of 9. At the formula level an r³ weight is not distinguished.
- **Radial weight.** One-particle radial moves show a weight close to pT r² for r < 0.2 in the 7 quantized 3-feature networks. Beyond r ≈ 0.2 the networks differ (drop to zero, flat, or rising). A single power law fits no network (χ²/dof 16–1,600). The fitted β (1.62–2.15 across the two samples) shifts by up to 0.10 between samples, more than its bootstrap interval.
- **Float networks.** float s0 has a flat radial response beyond r = 0.07. Its trained r2 fit (0.897–0.899) is no better than that of random float redraws (4/6 ≥ 0.9).
- **Thesis note.** "β is close to 2 in 8 of 10 networks" should read "the response is r²-shaped for r < 0.2 in the 7 quantized 3-feature networks". "Excludes the input-format alternative" should read "excludes it at p ≈ 0.01–0.05 against formulas of equal complexity".

## Issues for the ledger
1. The decoy null in E33x is not complexity-matched (3 regressors vs 1). With matched three-regressor decoys the p-value is 0.011–0.046, and 0.084 if library triples that contain Σ pT r² itself are counted as decoys.
2. The β bootstrap interval covers only within-sample resampling of particles in bins. On fresh jets β shifted by up to 0.10 (n32), outside both intervals. The power law has χ²/dof ≫ 1 everywhere.
3. "8 quantized 3-feature networks" in the E33x report should be 7.
4. Float s0's β of 1.87 is not r²-like behaviour (flat γ beyond r = 0.07).
5. Node k8s-haosu-21.sdsc.optiputer.net lost a pod (process in D state on CephFS, then the node became unreachable). It could be added to job.sh's exclusion list (I did not edit job.sh).

## Explainer entry (for site/explain.py)
- **Question.** Is the recurring squared-radius unit rarer than chance once the random comparison formulas have the same number of fitted terms?
- **Background.** E33x compared the three-term formula $a + b\sum_i p_{T,i} r_i^2 + c\sum_i r_i^2 + d\,N_{\rm soft}$ with one-term random formulas. More fitted terms make $R^2 \ge 0.9$ easier to reach.
- **Method.**
  - Fresh jets (seeds 5501/5502).
  - 300 random triples of hinge sums $\sum_i \mathrm{ReLU}(\cos\theta\, z_{p_T,i} + \sin\theta\, z_{r,i} + b)$ and 300 random triples of library sums, each fitted by least squares like the r2 formula.
  - $p = (1 + \#\{\text{decoys recurring} \ge 8\}) / (1 + \#\text{qualifying})$.
  - Radial weight from one-particle moves $r \to r \pm 0.01$: $\gamma_b = \partial^2 S / \partial r\, \partial p_T$ per bin, with $\gamma_b \propto r_b$ for a $p_T r^2$ weight.
- **Result.** r2 recurs 8/9 again. $p = 0.011$–$0.046$ against three-term decoys, compared with $0.010$–$0.014$ against one-term decoys. $\gamma_b / r_b$ is flat for $r < 0.2$ in the quantized 3-feature networks, and a single power law does not fit the full range (figure e33x_verify.png).
- **Checks.** Same decoys reproduce E33x exactly. Ablations: removing the $p_T$ weighting drops recurrence to 1/9. Untrained float redraws reach the r2 fit in 4/6.
- **Why it matters.** It sets the strength of the claim that JEDI-linear networks learn the same radial jet-shape observable: above chance at roughly the 1–5 % level, with the $p_T r^2$ shape limited to particles near the jet axis.
- **Glossary additions.** Matched-complexity decoy, χ²/dof (goodness of fit), exact McNemar test.
