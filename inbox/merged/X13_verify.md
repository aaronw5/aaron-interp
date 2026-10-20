# X13 verification: what sets the size of the axis (translation) response across networks

## Conclusion
On a fresh set of jets, each network's translation-to-dilation ratio R comes out almost the same as in X13 (rank correlation 0.99 across the 56 trained networks). The large seed differences in transformers, Linformers and float JEDI-linear also reproduce. However, "about 1 for the radial-only edit in every network" does not show that every network uses the distance from the stored axis: a read-out of the true jet mass, which ignores the axis, already gives 0.80 for this edit. The measured pattern matches a simple mix in which R is the share of the mass estimate that is tied to the stored axis. No separate cancelling mechanism is needed. Architecture does set R among the non-attention models (JEDI-linear, PFN, EFN, Deep Sets), but not among attention models. Two networks respond to the shift with the opposite sign.

## Procedure
- Script `interp/x13_verify.py`. It imports the edits, the network wrappers and the Ridge read-out from `interp/x13_axis_size.py` and drops the A′ edits.
- Jobs, each 1 CPU / 2 GiB:
  - x13-verify (jax group, 21 networks + oracles): 609 s, 2.0 GiB peak, 0.75 cores.
  - x13-verify-t (torch, 35 networks): 433 s, 1.16 GiB, 0.94 cores.
  - x13-verify-ana: 241 s, 0.19 GiB.
- Sample: `jedi_core.sample(per_class=1000, seed=2727)`, N = 5,000, with a fresh split rng(2727): 2,500 fit jets and 2,500 test jets. X13 used seed 1313.
- Ridge alpha was chosen on a random quarter of the fit half. X13 used the last quarter of the index-sorted fit half; its class mix was checked: [132, 114, 130, 111, 138], so it is not class-blocked.
- Networks: the same 56 trained networks. No untrained redraws.
- Edits: same as X13 (T_k at 8 azimuths, R_k, dil, rot), c = 0.06 rad.
- Jet bootstrap: B = 1000 over test jets, with the same indices for every network.
- New control: input-only oracle read-outs, computed on the n = 64 test jets with no network:
  - log m;
  - ½ log(M0 M2) about the stored axis;
  - recentred ½ log(M0 M2 − |M1|²);
  - linear mixtures w·log m + (1 − w)·½ log(M0 M2), w = 0.25 / 0.5 / 0.75.
- Additional analysis: OLS of R_rad on R across networks, compared with the two-route mixture prediction; variance decomposition with the Linformers and transformers removed; fresh-sample seed-pair differences for all 29 X13 pairs.

## Code review of x13_axis_size.py / x13_plot.py
- Edits are applied to real particles only. Padding stays unchanged (`padding_changed` = 0). T leaves m and pairwise distances unchanged; R_k reproduces T's r_i to 1e-7. The dilation formula M2(T_k) = M2 + 2c·M1 + M0c² is correct. hold_moments matches E24 A′. No split leakage: the read-out is fitted on the fit half and evaluated on the test half.
- Inputs and stages:
  - JEDI networks: last hidden layer = relu(z), pooled = S.
  - Torch networks: input of `head`, with padding zeroed after normalisation.
  - Sample accuracies match each network's reference test accuracy within about 1 %.
- Bootstrap CIs cover test-jet resampling only. They hold the fitted read-out fixed, so they leave out read-out-fit variance (see seed pairs below).
- Statement error: "d(T) exceeds d(rot) in all 56 trained networks". In two networks, tf-1b-d16-n32_s0 and tf-1b-d32-n32_s1, d(T) is below d(rot), because their mean d(T) is negative (X13: −0.037, −0.241). The test in the script is two-sided, so it shows that d(T) and d(rot) differ, not that d(T) is larger.
- Figure (`x13_axis_size.png`):
  - Readable; axes labelled.
  - No error bars.
  - No reference line for the read-out that ignores the axis (R_rad = 0.80 at R = 0). Without that line, R_rad ≈ 1 reads as more informative than it is.
  - Untrained crosses are clipped at R = 3.0 without notice (the X13 untrained range goes up to 4.6).

## Results (fresh sample, seed 2727)

### 1. Reproduction of per-network numbers
- Spearman between the X13 R and the fresh-sample R: 0.993 over 56 networks (Pearson 0.989).
  - Median |ΔR| 0.034.
  - Max |ΔR| 0.37: tf-1b-d32-n32_s1, −1.11 → −0.74.
- Spearman for R_rad between the two samples: 0.966.
- Summary statistics:
  - R median 0.95 (IQR 0.63–1.16), vs X13 0.94 (0.62–1.13).
  - R_rad median 1.07 (IQR 1.01–1.19), vs X13 1.05 (0.99–1.18).
- Spearman(R, R_rad) = +0.77 [0.62, 0.87], permutation p 1e-4. X13: +0.77 [0.61, 0.87].

### 2. Oracle control: what R_rad a read-out that ignores the axis gives
| read-out (input-only) | R | R_rad [95 % CI] |
|---|---|---|
| log m (true mass; translation-invariant) | 0.000 | 0.804 [0.797, 0.810] |
| ½ log(M0 M2 − \|M1\|²) (recentred; no explicit pairwise term) | 0.000 | 0.804 [0.798, 0.811] |
| ½ log(M0 M2) about stored axis | 1.000 | 1.000 |
| 0.25 log m + 0.75 axis | 0.750 | 0.951 |
| 0.5 / 0.5 | 0.500 | 0.902 |
| 0.75 log m + 0.25 axis | 0.250 | 0.853 |

The radial-only edit raises the true mass by 80 % of the dilation's change (Δlog m +0.241 vs +0.302). A two-route read-out with share α on the axis moment and 1 − α on the translation-invariant mass therefore has:
- R = α;
- R_rad = 0.80 + 0.20·R.

### 3. Networks against the mixture prediction
- H0: R_rad = 0.804 + 0.196·R, the two-route mixture with no extra pairwise sensitivity.
- H1: the networks deviate from this line.
- OLS across the 56 networks: R_rad = 0.804 [0.660, 0.896] + 0.328 [0.220, 0.482]·R, with a network-and-jet bootstrap.
  - Intercept: equal to the mixture value 0.804.
  - Slope: CI lies just above 0.196.
  - Mean residual from the mixture line: +0.118 [0.107, 0.130] (jet bootstrap); SD over networks 0.14.
  - The largest deviation is JEDI-linear n128: R 2.14, R_rad 2.11.
- Predictions of competing hypotheses:
  - "Every network reads r_i at full strength; pairwise use subtracts": R_rad ≈ 1 independent of R.
  - "R = share of the axis-referenced route": R_rad = 0.80 + 0.20 R.
  - The data follow the second form, with an offset of +0.12. The first form, a slope of 0, is excluded (slope CI 0.22–0.48).
- The recentred-moment oracle uses only M0, M1 and M2 and has no explicit pairwise term. It reproduces log m exactly for these edits. So the edits do not distinguish "pairwise distances" from "centroid (M1) recentring" as the route that is invariant under the shift.
- Negative R: 2 networks, both N = 32 transformers. In both, d(T) is below d(rot).
  - tf-1b-d16-n32_s0: d(T) − d(rot) CI [−0.045, −0.024].
  - tf-1b-d32-n32_s1: CI [−0.185, −0.161].
  - The other 54 have d(T) > d(rot).

### 4. Architecture vs seed
| quantity | between-family share | permutation q95 | p |
|---|---|---|---|
| log R, all (45 nets, 21 families) | 0.50 | 0.64 | 0.33 (X13: 0.48, 0.64, 0.38) |
| linear R, all (49 nets, 23 families) | 0.59 | 0.62 | 0.098 |
| log R by class (8 classes) | 0.20 | 0.26 | 0.15 |
| log R without Linformers / transformers (21 nets, 9 families) | 0.77 | 0.66 | 0.006 |

- H0: family labels explain no more variance than permuted labels.
- Pooled over all networks, H0 is not rejected, which reproduces X13.
- With the attention models removed, H0 is rejected (p 0.006). JEDI-linear, PFN, EFN and Deep Sets families differ in R more than their seeds do.

### 5. Seed-pair differences (fresh sample vs X13)
- Large differences reproduce:

| pair | X13 ΔR | fresh ΔR [95 % CI] |
|---|---|---|
| tf-1b-d32-n64 s0 − s1 | +1.72 | +1.69 [1.65, 1.73] |
| tf-1b-d32-n32 | +1.66 | +1.21 [1.15, 1.27] |
| linf-1b-d32-k4-n32 | −1.04 | −1.03 [−1.07, −0.99] |
| x12-float | −0.45 | −0.45 [−0.48, −0.42] |

- 2 of the 29 differences change sign on the fresh sample. X13 reported both with CIs that exclude 0:
  - pfn_64-64-32 s0 − s1: X13 +0.115 [0.094, 0.139]; fresh −0.005 [−0.026, +0.016].
  - x12-hgq s1 − s2: X13 −0.039 [−0.067, −0.008]; fresh +0.019.
- The jet-bootstrap CIs hold the read-out fixed, so they understate the uncertainty of differences of about 0.1. "28 of 29 CIs exclude 0" therefore overstates how many seed differences are established.

## Corrected reading
- R and R_rad per network reproduce on fresh jets and a fresh read-out fit (Spearman 0.99 / 0.97).
- R_rad near 1 is mostly fixed by the edit itself. A read-out of true mass gives 0.80, one of the axis moment gives 1.00, and any linear mix lies in between.
- Across networks, R_rad follows 0.80 + 0.33·R (mixture prediction 0.80 + 0.20·R; mean offset +0.12). R is therefore close to the share of the network's log-m read-out that is tied to the stored axis. The rest comes from quantities that a shift leaves unchanged.
- These edits do not separate pairwise distances from centroid recentring as that shift-invariant part. A cancellation stronger than the axis term is seen only in the 2 networks with R < 0.
- The axis-referenced response differs from the rotation null in all 56 networks: positive in 54, negative in 2.
- Family labels explain the R spread among JEDI-linear, PFN, EFN and Deep Sets families (share 0.77, p 0.006), but not across all networks (p 0.33). The seed dependence is concentrated in Linformers, transformers and float JEDI-linear.
- Seed differences below about 0.15 in R are not reproducible from these CIs.

## Files
- `interp/x13_verify.py`
- `results/jedi/x13_verify_jax.json`, `x13_verify_torch.json` (per-network rows with 1000 bootstrap replicates of R, R_rad; oracle rows), `x13_verify_summary.json`
- `logs/x13-verify.log`, `logs/x13-verify-t.log`, `logs/x13-verify-ana.log`
