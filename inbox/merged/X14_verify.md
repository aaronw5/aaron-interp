# X14 verify — axis-tied share R over training and the init × data-order grid, recomputed on fresh jets

Script `interp/x14_verify.py`; job `x14-verify` (AI_PY=conda, 1 CPU / 2 GiB; 887 s, peak 721 MiB, 0.98 cores); log `logs/x14-verify.log`;
results `results/jedi/x14_verify.json` (volume: `$AI_RESULTS/jedi/x14_verify.json`). Figure checked: `results/jedi/x14_axis_training.png`.

## Conclusion
On a new set of 5,000 jets the main pattern of X14 holds: how strongly the last-layer mass code of these attention taggers depends on the
stored jet axis (R; 0 = only true mass, 1 = only the spread about the axis) differs between training runs, and neither the starting weights
nor the batch order explains a detectable part of that spread. Two statements in the reading need correcting: changing only the starting
weights or only the batch order moves R by at most about 0.4 (not 0.5–0.7; the 0.7 change comes from the schedule-length pair alone), and
the "interaction" share is the share expected when neither factor has any systematic effect, so it is not evidence of a joint mechanism.
Still uncertain: the network's own Z-vs-W output does not follow R in the one large schedule-length change (it moves the other way), so R
describes the read-out of the last layer rather than the decision.

## Procedure
- **Sample**: `interp.jedi_core.sample(per_class=1000, seed=2029)` (X14 used 1313), N 5,000; fit / test halves 2,500 / 2,500 by
  `np.random.default_rng(2029).permutation`; X14 code path otherwise unchanged (`interp.x13_axis_size.load_sample`, `make_edits`, `Ridge`,
  `torch_runner`). Classes in the quarter of the fit half where `Ridge` chooses alpha: 121 / 132 / 131 / 115 / 126 (balanced).
- **Edits**: T_k rigid 0.06 rad translation of the real particles (8 azimuths), dil matched dilation, rot rotation about the stored axis (null).
- **Runs / checkpoints**: 18 grid runs (`tf-2b-d32`, `linf-2b-d32-k8`, `-n64-x14d{0,1,2}/seed{0,1,2}`) at epochs 3, 10, 30; schedule pairs
  `tf-2b-d32-n64` s0 vs `-e100` s0, `-p30` s1 vs `-e100` s1 at epoch 30 and the last epoch.
- **Metric**: R = mean_test d(T, azimuth mean) / mean_test d(dil) of a ridge read-out of log m on the last hidden layer, refitted per checkpoint.
- **Added controls**: (i) R_shuf: alpha chosen on a random quarter of the fit half instead of the positional last quarter; (ii) R_logit:
  read-out-free version, d(logit_Z − logit_W) under T / under dil on all test jets (the network's own output); (iii) interaction null:
  probability under full permutation of the 9 grid cells of a residual share ≥ the observed one, and the null mean shares.
- **Statistics**: jet bootstrap B = 1000 (shared indices); permutation tests 10,000 (shares), 5,000 (Spearman).

## Code review (interp/x14_axis_training.py, x14_plot.py, train_small.py --data_seed, make_jobs.py set x14)
- Edits act on real particles only (mask `pT > 0`); padded slots are masked in `torch_runner`; N 32 runs see the first 32 particles. No
  fit / test leakage (read-out fitted on the unedited fit half, evaluated on the test half; both halves come from the test archive).
- `--data_seed` changes only the `torch.Generator` that draws `randperm` for batch order; `torch.manual_seed(--seed)` still sets the
  initialisation; the models have no dropout, so the init seed controls nothing else during training. Epoch-0 R is identical across data
  seeds for a fixed init (|Δ| ≤ 3e-7), consistent with this.
- Schedule pairs: `-e100` differs from the 30-epoch run in `--epochs 100` (cosine total_steps) and patience; learning rate differs from the
  first post-warm-up step, and total training length differs. "Schedule length" in the reading covers both.
- Grid Spearman at epoch 0 (linf −0.78, p 0.016) treats 9 runs as independent although epoch-0 R takes only 3 distinct values (one per
  init seed); that p-value is not valid.
- Per-run jet-bootstrap CIs hold the read-out fixed and resample only test jets; they do not include the read-out refit or the choice of
  sample (see replication below).
- No bug found that changes the X14 numbers on its own sample.

## Results
**Replication of R (60 run × epoch values, seed 2029 vs seed 1313).** Spearman 0.990; |ΔR| median 0.040, max 0.119 (linf cell (0,0):
1.371 vs 1.490). The X14 per-run 95 % jet CIs have half-width 0.01–0.05, so the sample-to-sample change is up to ≈ 2–3× the reported CI.
R_shuf − R: max |Δ| 0.024. Rotation null: max |rot / dil| 0.019; T vs rot bootstrap p 0.002 (floor) in every run.

**Grid, final R (epoch 30; rows init 0–2, columns data order 0–2).**
| model | R (seed 2029) | shares init / order / residual [jet CI] | p init within order / order within init | p(residual ≥ obs), full perm. | max change, order alone [CI] | max change, init alone [CI] |
|---|---|---|---|---|---|---|
| tf-2b-d32 | [[0.49, 0.73, 0.74], [0.67, 0.61, 0.31], [0.63, 0.91, 0.65]] | 0.27 [0.24, 0.30] / 0.26 [0.24, 0.27] / 0.48 [0.45, 0.50] | 0.45 / 0.45 | 0.54 | 0.36 [0.35, 0.37] | 0.43 [0.41, 0.45] |
| linf-2b-d32-k8 | [[1.37, 1.16, 1.40], [1.42, 1.31, 1.08], [1.21, 1.16, 1.17]] | 0.22 [0.15, 0.29] / 0.24 [0.20, 0.27] / 0.54 [0.48, 0.61] | 0.42 / 0.43 | 0.46 | 0.33 [0.31, 0.35] | 0.31 [0.28, 0.34] |
X14 (seed 1313): tf 0.21 / 0.23 / 0.56, linf 0.34 [0.26, 0.42] / 0.31 / 0.35; max single-factor change tf 0.32 (order) / 0.35 (init),
linf 0.40 / 0.41. Null mean shares under full permutation: 0.25 / 0.25 / 0.50. The linf init share 0.34 of X14 lies outside the fresh-sample
jet CI [0.15, 0.29].

**Grid correlations (9 runs per model, seed 2029).** ρ(R, val acc): tf −0.05 (p 0.90), linf +0.20 (p 0.60). ρ(R, val loss): tf +0.05
(p 0.92), linf −0.13 (p 0.75) (X14: −0.63, p 0.074). ρ(R epoch 3, final R): tf +0.48 (p 0.19), linf −0.28 (p 0.46); epoch 10: +0.58 (p 0.10),
+0.58 (p 0.11).

**Schedule-length pairs (same init and batch order).** Seed 0: R 0.49 (30 epochs) → 1.17 (100 epochs), Δ +0.68 [0.66, 0.70], p 0.002
(X14: 0.51 → 1.23); R at epoch 30 of the long run 0.74. Seed 1: 0.61 → 0.49, Δ −0.11 [−0.13, −0.09], p 0.002 (X14: 0.65 → 0.46).

**Read-out-free control (R_logit, network's own Z − W logit).** Schedule pair seed 0: R_logit 0.73 [0.66, 0.81] → 0.35 [0.23, 0.45]
(Δ CI [−0.51, −0.28]), opposite in sign to the read-out R change (+0.68). Seed 1: 1.00 → 0.95 (Δ CI [−0.16, 0.07]). Across the grid
ρ(R, R_logit) tf +0.63 (p 0.07), linf +0.50 (p 0.18). R_logit shares: tf init 0.14 / order 0.61 (p 0.36 / 0.13); linf 0.05 / 0.38
(p 0.83 / 0.33); linf R_logit is unstable at intermediate epochs (one value 4.99 at epoch 10, denominator near 0).

## Hypotheses
- H0 (no init effect and no order effect on final R; R exchangeable over the 9 cells) vs H1 (an additive init or order effect): H0 not
  rejected on either sample (restricted p 0.42–0.47 fresh; 0.25–0.47 X14). Under H0 the expected shares are 0.25 / 0.25 / 0.50; the observed
  residual share is at its null expectation (p 0.46–0.54). "Largest part is interaction" is therefore the H0 prediction, not a separate finding.
- H_init (init fixes R) predicts identical R along each grid column: changes of 0.31–0.43 at fixed init on both samples; excluded as sole
  determinant. H_data predicts identical R along each row: changes 0.33–0.36 at fixed order; excluded as sole determinant.
- H_acc: no correlation on the fresh sample (|ρ| ≤ 0.20); the X14 linf val-loss value (−0.63) did not replicate.

## Figure check
Panel (a): seed-0 30- and 100-epoch curves coincide to ≈ epoch 20 and separate after; seed-1 dashed curve ends below the solid one; matches the
numbers. Panel (b) mixes N 32 / 64, the early-stopped runs and the 100-epoch runs on a fraction-of-training axis (as labelled). Panel (c)
matches `x14_summary.json`; its error bars are jet-bootstrap only and are smaller than the fresh-sample changes (up to 0.12).

## Corrections to the X14 reading
1. "Changing any one of them changes R by up to about 0.5–0.7" → changing only the init seed or only the batch order changes final R by up to
   0.35 / 0.32 (tf) and 0.41 / 0.40 (linf) on seed 1313, 0.43 / 0.36 and 0.31 / 0.33 on seed 2029; changing only the schedule length changed
   R by +0.68 to +0.72 (seed 0) and −0.11 to −0.19 (seed 1).
2. "The largest part is their interaction" → the shares (0.21–0.34 per factor, 0.35–0.56 residual) are those expected when R varies
   without any additive init or order effect (null 0.25 / 0.25 / 0.50); no factor structure is detected with 9 runs per model.
3. Per-run CIs (half-width 0.01–0.05) understate the uncertainty of R: a fresh sample moves R by up to 0.12 (median 0.04); share CIs
   likewise (linf init share 0.34 → 0.22).
4. "R at epochs 1–3 does not predict final R" → no detectable rank correlation at n = 9 (epoch 3: CI −0.46 to 0.95 in X14; fresh +0.48, p 0.19).
5. R is a property of a refitted last-layer read-out of log m; the network's own Z − W logit changed in the opposite direction in the
   seed-0 schedule pair (0.73 → 0.35) and correlates with R at ρ 0.50–0.63 across the grid.

## Explainer entry (for site/explain.py, X14 verification)
**Question.** Do the X14 statements about the axis-tied share $R$ survive a new jet sample, and is the "interaction" share informative?
**Method.** Recompute $R=\langle\hat\ell(T_c x)-\hat\ell(x)\rangle/\langle\hat\ell(D_s x)-\hat\ell(x)\rangle$ for the 18 grid runs and the
schedule pairs on sample seed 2029; decompose $SS_{tot}=SS_{init}+SS_{order}+SS_{res}$ and compare with the permutation null, where
$E[SS_{init}/SS_{tot}]=E[SS_{order}/SS_{tot}]=2/8$ and $E[SS_{res}/SS_{tot}]=4/8$.
**Result.** $R$ ranks replicate ($\rho=0.99$); shares 0.27 / 0.26 / 0.48 (tf) and 0.22 / 0.24 / 0.54 (linf), at the null values;
single-factor changes ≤ 0.43; schedule pair seed 0 $\Delta R=+0.68$.
**Checks.** Rotation null ≤ 0.019 of dilation; alpha-selection change ≤ 0.024; read-out-free logit version reported alongside.
**Why it matters.** The A10s wording should describe run-to-run spread of $R$ without attributing it to an init × order interaction.
