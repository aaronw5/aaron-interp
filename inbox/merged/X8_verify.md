# X8 verification: functional nulls for the headline causal tests (JEDI-linear part, fresh sample)

Verifier node for X8. Script `interp/x8_verify.py`. It imports `interp/x8_nulls.py` unchanged: the edits, `measure()`,
the permutations and `ablation()`. Results are in `results/jedi/x8_verify.json`, the log in `logs/x8-verify.log`.
Cluster job `x8-verify`: 1 CPU / 2 GiB, code 83cd9c2 plus the new script, wall 1370 s, peak memory 2047 MiB, exit 0.

## Conclusion
On a new sample of 10,000 jets, every JEDI-linear number from X8 comes back within its interval:
- W→Z flow: 0.744 vs 0.752.
- Window ratio: 5.19 vs 5.22.
- Radial response: 0.205 vs 0.200.
- g, mean → resample ablation: 14.9 % → 27.9 % of calls.
- Mean vs resample unit ranking: Spearman 0.965.

Three parts of the reading need narrower wording:
1. The accuracy drop under WTA re-axing cannot separate trained from null networks. The nulls are at chance accuracy, so
   they have nothing to lose. A random-direction translation of the same length lowers accuracy more than the WTA re-axis
   does.
2. The radial-response gap comes mostly from the selected unit's correlation with $\sum p_T r^2$. Per unit of
   correlation, several networks with no trained function respond as much as the trained one.
3. The H3 null range quoted as 0.5–11 includes constant-output copies that used a fallback unit rule. The added
   per-unit control shows that the 13|14 mask edge produces the asymmetry. The trained network's own live units without
   that edge give a ratio of 1.45.

## Code review of `interp/x8_nulls.py`
- **Forward pass.** The own forward pass equals `Jedi.run`: max |Δlogit| = 0 before the permutations and after
  restoring them.
- **Stages and overrides.** The stages are those of `jedi_core`. The per-jet override reshapes `(J, u)` to `(J, 1, u)`
  for the 3-D P/g tensors.
- **Edits.** They touch only real particles (pT > 0) and keep pT order. Swaps are defined only when both ranks hold a
  real particle.
- **Read-out.** The ridge is fitted on the fit half (alpha chosen on its 3/4 vs 1/4 split) and evaluated on the test
  half, so there is no split leakage.
- **Wr and Kr null edits.** Wr uses the per-jet WTA displacement length. Kr's rotation angle $2\arcsin(0.05)$ gives the
  same per-particle displacement as K ($0.1\,r$).
- **Quantizer state.** In variants b and c, qs1 masks are recomputed after the quantizer tables are permuted. A new jit
  function per model avoids stale traced constants.
- No coding error was found that changes a reported number.
- **Design points.** These are not bugs, but they affect interpretation:
  - **Degenerate nulls.** A null whose S is constant, or that has no live S unit, falls back to "10 S units of largest
    s.d.". That is a different unit rule from the trained network's. Two such copies set the lower end (0.52) of the
    reported variant-a null range for H3.
  - **H4 statistic.** It is in s.d. units of the unit chosen by maximum |ρ| with $Q=\sum p_T r^2$. The expected response
    of a unit linear in Q is $|\rho|\,E_Q$, with $E_Q = \overline{0.21\,Q}/\mathrm{sd}(Q)$. The statistic therefore
    scales with the selected unit's correlation.

## Procedure
- **Sample.** `sample(per_class=2000, seed=7307)`, N = 10,000 jets (N_W = 2,000). The read-out split is as in X8 (rng
  7307): fit half 5,000, test half 5,000.
- **Models (JEDI-linear 3-feature n64).**
  - Trained.
  - Variant a, s0–4.
  - Variant c, s0–4.
  - Variant b, s0 only (all five were constant-output in X8).
  - **Added null: untrained redraw s0–2.** This is `jetdistill.network.redraw_untrained`: every kernel, bias and BN
    gamma is redrawn from a Gaussian with the trained mean and s.d. Number formats and zero-bit masks are kept. It is
    meant as a null that keeps per-layer scale.
- **Edits and metrics.** As in X8: T/Tr, W/Wr, K/Kr, and the 13|14 vs 5|6 angle swap. The statistics are H1–H4,
  accuracy drop under W, and the Kr control.
- **Degenerate model.** A model is degenerate if one class receives ≥ 99 % of calls.
- **Statistics.** 1,000 paired jet bootstraps (the same indices for every model). Per-null one-sided p. $p_{\rm IUT}$ is
  the maximum over all nulls and, separately, over the non-degenerate nulls. Holm correction over the 4 headline tests.
- **Added control 1 (H3 units).** For each model, the across/inside ratio of mean $|\Delta S_c|/\mathrm{sd}(S_c)$ is
  computed over:
  - the live S units whose own qs1 mask changes between slots 13 and 14 (edge units);
  - the live S units without that change (no-edge units).
- **Added control 2 (H4 normalisation).** response / ($|\rho|\,E_Q$), with $E_Q$ = 0.2253 on this sample.
- **Ablation arm (trained).** Mean vs resample ablation of every live P/S/z unit, with 10 derangements (X8 used 20).
  Context g: mean vs resample.

## Results (trained [95 % CI] vs null points)

| statistic | trained (seed 7307) | X8 (seed 307) | non-degenerate nulls (a, c; n = 8) | all nulls (n = 14) | p_IUT (non-deg.) → Holm |
|---|---|---|---|---|---|
| H1 W→Z flow under T | 0.744 [0.724, 0.765] | 0.752 | −0.083 … 0.109 | −0.083 … 0.109 | 0.001 → 0.004 |
| H2 calls changed under W | 0.306 [0.298, 0.315] | 0.298 | 0.034 … 0.628 | 0 … 0.628 | 1.0 → 1.0 |
| H3 A13/I5 ratio | 5.19 [4.96, 5.42] | 5.22 | 2.07 … 11.34, + 2 × ∞ | 0.52 … 11.34 | 1.0 → 1.0 |
| H4 radial response [s.d.] (S50) | 0.205 [0.200, 0.210] | 0.200 | 0.018 … 0.086 | 0 … 0.142 | 0.001 → 0.004 |
| accuracy drop under W | 0.218 [0.209, 0.227] | 0.219 | −0.012 … 0.017 | −0.012 … 0.017 | 0.001 |
| Kr control (radial unit) | 0.0001 [−0.001, 0.001] | 0.0006 | −0.002 … 0.004 | −0.002 … 0.008 | — |

Unedited trained accuracy on this sample: 80.42 % (X8 sample: 81.47 %). Null accuracies are 19.5–23.1 %. All three
untrained redraws, variant b s0, a4 and c4 give a constant call. The non-degenerate set includes c3, which gives 98.1 %
of calls to one class.

### Matched null edits on trained JEDI-linear (exact McNemar)
- **T vs Tr, W→Z calls among true W.** 362 vs 84 discordant, p = 4e-42. Flow 0.744 under T vs 0.605 under Tr.
- **W vs Wr, calls changed.** 0.306 vs 0.338; discordant 323 vs 641, p = 7e-25. Accuracy 0.804 → 0.586 under W and
  0.804 → 0.554 under Wr. The X8 sample gives the same pattern: 0.815 → 0.596 under W and 0.815 → 0.559 under Wr.
- **K vs Kr, calls changed.** 0.245 vs 0.023; discordant 2343 vs 122, p < 1e-300.

### Added control 1: H3 ratio by unit group (13|14 vs 5|6)

| model | edge units: ratio (units) | no-edge live units: ratio |
|---|---|---|
| trained | 5.19 (S0, S43, S51, S52, S63) | 1.45 (14 units) |
| a0 / a1 / a2 / a3 | 9.24 / 2.07 / 3.01 / 6.76 | 1.64 / 2.82 / 2.55 / 4.62 |
| c0 / c1 / c2 / c3 | ∞ / ∞ / 11.34 / 6.69 | 0.48 / 0.11 / 0.90 / ∞ (1 unit) |

None of the edge units of the trained network has a qs1 change at 5|6.

### Added control 2: H4 response / ($|\rho| E_Q$)

| model | ρ of selected unit | response / ($|\rho| E_Q$) |
|---|---|---|
| trained (S50) | 0.928 | 0.98 |
| non-degenerate nulls | 0.47–0.70 | 0.17–0.69 |
| degenerate nulls with non-constant S (a4, c4, untrained s1, s2) | 0.43–0.60 | 0.95–1.13 |

S29: ρ = 0.854, response 0.263 s.d.

### Ablation arm (trained, N = 10,000, 10 resamples)
- **Spearman, mean vs resample cost.** 0.965 [0.957, 0.973] over 44 live units (X8: 0.965 over 43). Per stage: P 0.99,
  S 0.99, z 0.93.
- **BH test.** 44/44 units have resample cost > 1 % at q < 0.05.
- **Context g.** Mean ablation 14.85 % [14.2, 15.6] → resample 27.88 % [27.3, 28.4] of calls. McNemar (mean vs first
  resample) 496 vs 1772, p = 2e-167.
- **S29.** Mean 41.8 %, resample 38.9 %.

## H0/H1 and competing hypotheses
- **H0 / H1.** H0: trained statistic ≤ null statistic, for every null. H1: trained > every null.
- **Rejected.** H0 is rejected for H1 (flow), H4 (radial) and the accuracy drop at p_IUT = 0.001 (Holm 0.004).
- **Not rejected.** H0 is not rejected for H2 or H3.
- **Architecture/mask artefact for H3.**
  - Not excluded. Mask-keeping permuted copies give 2.1–11.3.
  - In the trained network, live units without an edge at 13|14 give 1.45.
  - The ratio follows the presence of the mask edge.
- **"Learned" for the H4 gap.** The trained network's selected unit has ρ = 0.93, against ≤ 0.70 in the nulls. Per unit
  of correlation, its response (0.98) is matched by nulls with no trained function (0.95–1.13). The training-specific
  quantity measured is how closely one S unit tracks $\sum p_T r^2$. It is not the size of that unit's causal response
  per unit of correlation.
- **WTA specificity.** Nulls cannot show an accuracy drop because they are at chance. The matched edit Wr lowers
  accuracy more than W (0.250 vs 0.218), so the accuracy loss under re-axing is not specific to the WTA direction.
- **Untrained redraw as a scale-keeping null.** All three seeds give constant calls, so this null does not make H2
  informative either.

## Corrected reading
- **Translation (H1).** The W→Z flow (0.744; X8 0.752) exceeds every permuted and redrawn JEDI-linear null (≤ 0.109).
  The small-model part of the claim was not rerun here.
- **Accuracy loss under WTA re-axing.** It cannot be compared with chance-level nulls. The matched random-direction
  translation Wr lowers accuracy more (0.250 vs 0.218) and changes more calls (p = 7e-25). The measured loss therefore
  reflects the displacement size and is not specific to the WTA axis.
- **H2 (share of calls changed).** It does not separate trained from null.
- **H3.** Across/inside ratio 5.19; non-degenerate mask-keeping nulls give 2.1–11.3. The trained network's live units
  without a 13|14 mask edge give 1.45.
- **H4.** The radial response (0.205 s.d.) exceeds every null (≤ 0.142). This follows from the selected unit's
  correlation with $\sum p_T r^2$ (0.93 vs ≤ 0.70). Normalised by correlation, the response of networks with no trained
  function is 0.17–1.13, against 0.98 for the trained network.
- **Ablation.** Resample ablation reproduces the mean-ablation ranking (Spearman 0.965). g under resample changes
  27.9 % of calls.

## Issues
- The small models (dsets, jedis) were not rerun because they need the torch (conda) environment. Their numbers were
  checked only for consistency in `x8_dsets.json` / `x8_jedis.json`. In those files:
  - dsets s1 (ρ 0.87, response 0.133) and jedis s0 (0.181) lie inside the H4 null range (dsets perm2 0.159 at ρ 0.55;
    jedis perm1 0.200 at ρ 0.62, a 99.8 %-one-class model);
  - the radial selection rule picks different units in each model.
- Peak memory was 2047 MiB against a 2 GiB request. Later runs of this size should use fewer models per job.
- 10 resamples were used instead of 20; the Spearman CI is the same width as X8's.

## Explainer additions (for site/explain.py, ledger agent)
- **Checks.**
  - A fresh-sample rerun (seed 7307) reproduces the headline numbers.
  - The across/inside ratio drops to 1.45 on the trained network's own units that have no 13|14 mask edge.
  - The radial response divided by $|\rho|\,\overline{\Delta Q}/\mathrm{sd}(Q)$ is 0.98 for the trained unit and 0.95–1.13
    for null units with constant output.
  - A random-direction translation of the WTA displacement length lowers accuracy more than the re-axis.
