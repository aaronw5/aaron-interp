# P11 — ParT seed replication: pre-registered S1–S4 on six retrained checkpoints (claims C3s, C3, C2b, C7), 2026-10-20

Status: **done**. Jobs `anrunw-ai-c-p11-full` and `anrunw-ai-c-p11-kin` (RTX 2080 Ti, 621 s and 660 s), smoke `c-p11-smoke`,
figure jobs `c-p11-plot` / `c-p11-plot2` (CPU), seed-identity check `c-p11-wcheck` (CPU).
Scripts: `interp/p11_seed_tests.py` (analysis, `--plot` for the figure), `interp/p11_weight_check.py` (weight correlations
between checkpoints); wrapper fix in `interp/p10_part_seeds_train.sh`.
Results: `results/part/p11_seed_tests_full.json`, `results/part/p11_seed_tests_kin.json` (attempt 1:
`*.attempt1.json`), figure `results/part/p11_seed_tests.png`; logs `logs/c-p11-{full,kin,smoke,wcheck,plot2}.log`
(attempt 1: `logs/c-p11-{full,kin}.attempt1.log`).

## Conclusion
We retrained the Particle Transformer six times from scratch (two input sets, three random seeds, 2 million training jets
each). Every retrained network still calls more true W jets "Z" when all particles are rotated by 0.06 rad about the jet
axis, an edit that leaves the jet's invariant mass unchanged. The shift is 14–19 percentage points, against 18–19 for the
released networks. So the axis-relative mass response is not an accident of one checkpoint. What does not carry over is
the internal route: in the released networks the first class-attention block carries about 97–98 % of the W/Z unit's
response to a real mass change, but in the retrained ones it carries 63–83 %, and the rest arrives through the second
block. The retrained networks saw 50 times less data, so this run cannot say whether the different route comes from the
seed or from the data scale.

## Procedure
1. **P10 wrapper fix.** The four "Failed" P10 jobs (full_s0, kin_s0/s1/s2) printed `line 54: port: command not found` and
   `line 56: syntax error near unexpected token ')'` after the final test step. bash reads a script as it runs, and
   `./sync.sh` rewrote `interp/p10_part_seeds_train.sh` (the header comment grew) while the jobs sat in `wait` for 9 h. The
   jobs started before that sync then resumed at a stale byte offset, in the middle of a line. full_s1 and full_s2 started
   later (queued 15–30 min) and exited 0. The training and the final test had already finished in all six jobs.
   - Fix: the body is now one function (`main() { ...; }; main "$@"; exit $?`), which bash parses completely before running.
   - The end-of-job summary now prints all ten per-class test metrics and their mean (weaver reports one accuracy per
     class file). `bash -n` passes.
2. **Models.** Per feature set (full, kin):
   - released ParT_<feat>;
   - seeds 0/1/2 (`/j-jepa-vol/aaron-interp/models/part_seeds/<feat>_s<k>/net_best_epoch_state.pt`, the best validation
     epoch of 50);
   - control "untrained": the same architecture from weaver's own initialisation with torch seed 0;
   - control "shuffled" (kin only): the P14 arm-s ParT (kin inputs, 2 outputs, labels permuted over the training jets,
     2 epochs). Only the read-out and attention statements apply to it.
3. **Seed identity check** (`p11_weight_check`). This is the Pearson r of flattened weight tensors between checkpoints, for 5
   tensors: particle embedding, pair embedding, particle-block-1 in_proj, class-block-1 in_proj, last layer.
4. **Samples** (JetClass test_20M, `<sample>_100.root`, never first-k):
   - P1 sample: rows 3,000–4,000 of each class file, N = 10,000. Used for 10-class accuracy, the confusion matrix, and the
     ridge fit.
   - WZ sample: 10,000 random rows each of WToQQ_100 and ZToQQ_100 (`default_rng(11)`; rows 3,000–3,999 excluded), N =
     20,000. No jet had fewer than 2 particles. Median mass 88.5 GeV.
   - S3 subset: 2,000 W + 2,000 Z of the WZ sample (`default_rng(12)`).
5. **Edits** (`part_core.build`, corrected η). Here $\tfrac12\Delta\log(M_0M_2)$ is the axis-moment change, with $M_0=\sum p_T$
   and $M_2=\sum p_T\Delta R^2$ about the stored axis.
   - phi: every Δφ + 0.06. Δlog m = 1.6e-8 (max |·| 1e-4); $\tfrac12\Delta\log(M_0M_2)$ = +0.0865.
   - match (matched dilation): angles × $k_j=\exp(\tfrac12\Delta\log(M_0M_2)_{\phi,j})$ per jet (mean 1.091; 5/50/95 % =
     1.039 / 1.081 / 1.175). It gives the same per-jet axis-moment change as phi (max difference 1e-7) but changes the real
     mass (Δlog m +0.085).
   - k1.1: angles × 1.1 (Δlog m +0.094).
   - rot: the whole jet including the axis rotated by 0.06 rad; this is the numerical null.
   - Route variants (P1v / P2): (x_base, v_phi) = pair route only; (x_phi, v_base) = per-particle route only; x_base with
     only deltaR, only dphi, or both from x_phi; the same pair / per-particle split for k1.1; (x_base, v_rot) = pair-route null.
6. **S1 (read-out).**
   - The read-out is P1's ridge regression of log m on the class token z (128 numbers), fitted on the P1 sample (fit 3/4,
     α chosen on the last 1/4, refit on all) and applied to the WZ sample.
   - Ratio = mean Δlog m̂(phi) / mean Δlog m̂(match), 1,000 bootstrap resamples over jets.
   - Competing predictions: H_axis (the network reads $\sqrt{M_0M_2}$ about the stored axis) gives 1; H_inv (invariant mass)
     gives 0. The pre-registered window is 0.7–1.4.
   - Also reported: the P10 calibrated form (Δm̂/prediction under phi) / (Δm̂/prediction under k1.1).
   - Logit level: R = OLS slope through 0, over true-W jets, of Δ(ℓ_W − ℓ_Z)(phi) on Δ(ℓ_W − ℓ_Z)(match).
   - W→Z = fraction of true W called Z (10-class argmax), before → after, with a paired bootstrap CI and an exact McNemar
     test. Holm correction over the 50 McNemar tests per job.
7. **S2 (input route).** Share = mean Δlog m̂(variant) / mean Δlog m̂(phi), and the same for Δ(ℓ_W − ℓ_Z) on W. Pre-registered:
   the deltaR share > 0.8 (full), the dphi + deltaR share > 0.8 (kin).
8. **S3 (block route).**
   - The W/Z unit u* is the z coordinate with the largest |W[u, W] − W[u, Z]| × sd(z_u). Its response is Δz_u*/sd under
     k1.1 on the S3 subset.
   - Removed share = 1 − Σ_j dP_j dE_j / Σ_j dE_j², where dE is the edited-run response and dP the response with the
     head set patched to its unedited-run values. This per-jet projection is the primary form because in full_s1 the W and Z
     responses have opposite signs. The pooled-mean form (P3v's), W-only, Z-only and the W/Z-logit form are also in the json.
   - Head sets:
     - class-attention block 1 (cls_blocks[0], 8 heads; the class token's query there is the constant cls token, so every
       particle → class-token path through class block 1 passes through these heads);
     - the same block by resample ablation (contexts from a random other jet of the batch, in both runs);
     - class block 2;
     - each of the 16 class heads;
     - 30 block-stratified random sets (4 heads per class block);
     - particle block 1 (blocks[0]; the literal wording of the P10 pre-registration);
     - each of the 64 particle heads;
     - 20 stratified random sets (one head per particle block);
     - all 64 particle heads.
   - Per-head bootstrap p (share = 0), Holm within the 16 class heads and within the 64 particle heads.
9. **S4 (switch scan).**
   - Per head, a per-jet statistic vs jet mass on the WZ sample:
     - class heads: the class-token attention share;
     - particle heads: the attention-weighted pair ΔR averaged over real query particles, divided by the jet's mean pair ΔR.
   - Logistic fit $y=a+c/(1+e^{(m-m_{50})/w})$ to 5 GeV binned means over 50–140 GeV (weighted least squares on a grid:
     m50 40–160 GeV, w 0.5–80 GeV).
   - Switch (pre-registered form) requires all of: m50 in 60–110 GeV; w < 15 GeV; |c| ≥ 0.2·max(|a|, |a+c|); the criterion
     met in ≥ 95 % of 200 jet bootstraps; BH q < 0.05 over the 80 heads.
   - The p-value for BH is a permutation test with 4,000 mass shuffles. H0: the statistic does not depend on mass; the test
     statistic is the span of the binned means.
   - Strong switch: in addition |c| ≥ 0.2, P3's 20-point rule.

## Numbers
**Seed identity** (r per tensor):
- between different seeds, or between a seed and the released model: |r| ≤ 0.16 for every tensor (most |r| < 0.05);
- epoch 0 vs best epoch of the same seed: 0.57–0.95.

The six checkpoints are distinct networks, so head indices are arbitrary labels across seeds.

**Accuracy.** P1 sample, N = 10,000, with 95 % bootstrap CI. The weaver test is the mean of 10 per-class accuracies on
test_20M `_100` (1M jets).

| model | P1-sample acc | weaver test (1M) | best val | own-class W / Z |
|---|---|---|---|---|
| released full | 85.93 % [85.25, 86.58] (ref 85.9) | – | – | 77.9 / 72.0 |
| full s0 / s1 / s2 | 83.71 / 83.62 / 83.49 % | 83.65 / 83.46 / 83.50 % | 0.838 / 0.837 / 0.837 | 77.4 / 65.4; 76.7 / 65.5; 76.9 / 64.5 |
| released kin | 74.47 % [73.54, 75.33] (ref 74.5) | – | – | 68.4 / 60.0 |
| kin s0 / s1 / s2 | 72.57 / 72.09 / 72.04 % | 72.28 / 72.22 / 72.34 % | 0.726 / 0.725 / 0.725 | 66.0 / 56.9; 63.4 / 57.9; 64.7 / 57.4 |
| untrained full / kin | 10.07 / 9.54 % | – | – | – |

Full 10×10 confusion matrices are under `p1_confusion` in the jsons.

**Calls under the edits.** WZ sample; 10,000 true W per model. Δ is in percentage points with a 95 % bootstrap CI. Holm
McNemar p for phi is ≤ 5e-245 in every trained model. The rot null is 0.0 to +0.1 points (Holm p = 1), and the pair-only
null is within ±0.1.

| model | W→Z base | phi: after (Δ) | match: after | k1.1: after | R (phi / match) [CI] | R_mean | R under rot |
|---|---|---|---|---|---|---|---|
| released full | 11.1 | 30.4 (+19.3 [18.4, 20.1]) | 35.4 | 37.0 | 0.787 [0.778, 0.796] | 0.806 | 0.000 |
| full s0 | 12.8 | 31.9 (+19.1 [18.2, 19.9]) | 35.7 | 37.0 | 0.805 [0.796, 0.815] | 0.844 | 0.000 |
| full s1 | 11.8 | 26.8 (+15.0 [14.2, 15.7]) | 34.6 | 36.0 | 0.650 [0.638, 0.663] | 0.686 | 0.000 |
| full s2 | 12.1 | 29.9 (+17.8 [17.0, 18.6]) | 34.9 | 35.6 | 0.786 [0.775, 0.798] | 0.832 | 0.000 |
| untrained full | 0.0 | 0.0 (0.0) | 0.0 | 0.0 | (0.38; every jet called Hcc) | 0.45 | 0.000 |
| released kin | 16.2 | 34.0 (+17.9 [16.9, 18.8]) | 42.8 | 43.9 | 0.722 [0.712, 0.732] | 0.755 | −0.001 |
| kin s0 | 16.8 | 32.3 (+15.5 [14.7, 16.3]) | 42.5 | 42.8 | 0.575 [0.563, 0.586] | 0.596 | 0.000 |
| kin s1 | 18.4 | 33.1 (+14.7 [13.9, 15.6]) | 44.1 | 44.7 | 0.542 [0.530, 0.555] | 0.548 | −0.001 |
| kin s2 | 17.6 | 31.2 (+13.6 [12.8, 14.4]) | 42.9 | 43.1 | 0.526 [0.513, 0.538] | 0.561 | −0.001 |
| untrained kin | 78.8 | 83.2 (+4.4 [3.8, 4.9]) | 77.6 | 77.3 | −0.23 | 0.58 | −0.002 |

Across seeds:

| | R mean (range) | W→Z shift under phi, mean (range) |
|---|---|---|
| full | 0.75 (0.65–0.81) | +17.3 (+15.0 to +19.1) points |
| kin | 0.55 (0.53–0.57) | +14.6 (+13.6 to +15.5) points |

At epoch 3 (P10) the kin seeds were at +4 to +9 points and R 0.17–0.35.

**S1, read-out ratio** Δlog m̂(phi) / Δlog m̂(match), with 95 % CI. The pre-registered window is 0.7–1.4.

| model | ratio | in window | calibrated (P10 form) | Δlog m̂ phi / match / k1.1 / rot | read-out R² (P1 val / WZ) |
|---|---|---|---|---|---|
| released full | 1.004 [0.997, 1.011] | yes | 1.024 | .0814 / .0811 / .0876 / .0000 | 0.880 / 0.752 |
| full s0 | 0.980 [0.969, 0.992] | yes | 1.045 | .0730 / .0744 / .0770 / .0000 | 0.830 / 0.632 |
| full s1 | 0.632 [0.622, 0.643] | no | 0.642 | .0505 / .0799 / .0867 / .0000 | 0.830 / 0.630 |
| full s2 | 0.877 [0.868, 0.887] | yes | 0.902 | .0665 / .0758 / .0813 / .0000 | 0.852 / 0.716 |
| untrained full | 0.930 [0.918, 0.944] | (yes) | 0.918 | .0632 / .0679 / .0758 / .0000 | 0.776 / 0.513 |
| released kin | 1.501 [1.485, 1.515] | no (above) | 1.506 | .1061 / .0707 / .0776 / −.0001 | 0.920 / 0.840 |
| kin s0 | 0.743 [0.733, 0.754] | yes | 0.768 | .0609 / .0820 / .0875 / −.0001 | 0.857 / 0.745 |
| kin s1 | 0.637 [0.625, 0.649] | no | 0.670 | .0523 / .0820 / .0860 / .0000 | 0.867 / 0.722 |
| kin s2 | 0.509 [0.499, 0.520] | no | 0.515 | .0395 / .0776 / .0846 / −.0001 | 0.872 / 0.761 |
| untrained kin | 0.019 [−0.001, 0.037] | no | 0.019 | .0012 / .0612 / .0669 / .0000 | 0.757 / 0.414 |
| shuffled kin | −1.031 [−1.048, −1.014] | no | −1.010 | −.0736 / .0713 / .0802 / .0010 | 0.862 / 0.679 |

**S2, input route** (share of Δlog m̂ under phi; logit share on W in brackets). The pair-only share under phi is ≤ 0.003 in
absolute value in every model, and the rot_pair null is ≤ 0.003 (trained models).

| model | deltaR only | dphi only | per-particle (all) | pre-reg. key > 0.8 | k1.1 pair-only share: read-out [logit] |
|---|---|---|---|---|---|
| released full | 0.980 [1.006] | 0.077 [0.031] | 1.000 | dR 0.980: yes | 0.010 [0.128] |
| full s0 / s1 / s2 | 0.998 / 1.054 / 1.015 | −0.002 / −0.041 / −0.010 | 1.000 | yes / yes / yes | 0.108 / 0.170 / 0.143 [0.110 / 0.129 / 0.127] |
| untrained full | 0.974 | 0.049 | 1.000 | (yes) | −0.003 |
| released kin | 0.349 [0.652] | 0.573 [0.386] | 1.000 | dφ+dR 1.000: yes | 0.101 [0.098] |
| kin s0 / s1 / s2 | 0.583 / 0.549 / 0.409 | 0.288 / 0.355 / 0.491 | 0.999 / 1.003 / 0.999 | yes / yes / yes | 0.201 / 0.125 / 0.160 [0.106 / 0.093 / 0.125] |
| shuffled kin | −0.596 | 1.326 | 1.000 | (yes) | 0.334 |

**S3, block route** of the W/Z unit u*: removed share of its response to angles × 1.1, as a per-jet projection with 95 % CI.
N = 4,000 W/Z jets.

| model | u* ((W_W−W_Z)·sd) | Δz_u*/sd (W, Z) | class block 1 | cb1 resample | class block 2 | particle block 1 | all 64 particle heads | random 4+4 class sets mean / max (p) | random 1-per-block particle sets mean / max (p) | largest single class head | W/Z-logit share, cb1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| released full | 97 (+2.72) | −0.30 (−0.41, −0.19) | 0.981 [0.980, 0.983] | 0.989 | 0.074 | 0.086 | 0.409 | 0.50 / 0.87 (0.032) | 0.05 / 0.08 (0.048) | c1h3 0.45 | 0.983 |
| full s0 | 80 (+0.60) | −0.22 (−0.26, −0.17) | 0.790 [0.784, 0.796] | 0.806 | 0.265 | 0.330 | 0.820 | 0.52 / 0.71 (0.032) | 0.12 / 0.21 (0.048) | c1h2 0.19 | 0.780 |
| full s1 | 96 (+0.51) | −0.04 (−0.15, +0.08) | 0.825 [0.816, 0.834] | 0.865 | 0.173 | 0.315 | 0.849 | 0.49 / 0.68 (0.032) | 0.11 / 0.21 (0.048) | c1h6 0.22 | 0.731 |
| full s2 | 19 (−0.67) | +0.31 (+0.39, +0.23) | 0.729 [0.724, 0.734] | 0.754 | 0.322 | 0.324 | 0.781 | 0.53 / 0.68 (0.032) | 0.08 / 0.18 (0.048) | c1h2 0.19 | 0.627 |
| released kin | 54 (−1.05) | +0.22 (+0.16, +0.29) | 0.968 [0.965, 0.972] | 0.983 | 0.054 | 0.115 | 0.440 | 0.53 / 0.74 (0.032) | 0.05 / 0.08 (0.048) | c1h7 0.46 | 0.913 |
| kin s0 | 4 (+0.37) | −0.33 (−0.32, −0.33) | 0.643 [0.634, 0.651] | 0.662 | 0.360 | 0.233 | 0.810 | 0.51 / 0.69 (0.097) | 0.08 / 0.16 (0.048) | c1h3 0.16 | 0.637 |
| kin s1 | 118 (+0.60) | −0.34 (−0.27, −0.41) | 0.675 [0.672, 0.679] | 0.699 | 0.346 | 0.256 | 0.744 | 0.54 / 0.68 (0.097) | 0.08 / 0.14 (0.048) | c1h7 0.21 | 0.473 |
| kin s2 | 31 (−0.45) | +0.31 (+0.24, +0.37) | 0.632 [0.626, 0.637] | 0.682 | 0.354 | 0.255 | 0.785 | 0.49 / 0.72 (0.097) | 0.09 / 0.14 (0.048) | c1h7 0.15 | 0.586 |
| untrained kin | 72 (−0.11) | +0.10 | 0.867 | 0.872 | 0.127 | 0.991 | 0.947 | 0.50 / 0.83 | 0.16 / 0.89 | c1h5 0.28 | – |

- The released-full row reproduces P3v: n98 is selected by the rule, and class block 1 gives 0.987 in the pooled form.
- Untrained full: u* responds 0.002 s.d., so the shares are undefined.
- Holm-significant single heads (share ≠ 0): 15–16 of 16 class heads in every model, 0 of 64 particle heads in every model.
- Largest single particle head: 0.04 (released), 0.07–0.11 (seeds).
- Permutation p values at their floor are 1/31 = 0.032 for the 30 class-head random sets and 1/21 = 0.048 for the 20
  particle-head random sets.

**S4, mass-switch scan** (80 heads; BH q ≤ 1e-3 for every listed head).

| model | heads passing the pre-registered form (count) | strong switches (|amplitude| ≥ 0.2): head m50 / w [GeV], statistic low-mass → high-mass |
|---|---|---|
| released full | 7 (c1h4, c1h6, c2h1, c2h3, c2h5, c2h6, c2h8) | c2h8 (= b2h8): 87.0 [86.5, 87.0] / 4.8 [4.3, 4.8], 0.87 → 0.08 (P3: 87.1 / 4.25) |
| full s0 | 9 | p8h6 (particle block 8 head 6): 76.0 / 9.2, 0.75 → 1.27; no class head (largest c2h2 0.21 → 0.05 at 75 GeV, \|c\| 0.15) |
| full s1 | 7 | c2h2: 105.0 / 3.1, 0.40 → 0.12; p7h5: 67.0 / 3.5, 1.22 → 0.88 |
| full s2 | 8 | c2h4: 68.0 / 4.8, 0.59 → 0.02; p2h8: 97.5 / 12.8; p8h3: 68.5 / 7.4 |
| untrained full | 0 | – |
| released kin | 8 | none (c2h8: 91.0 / 5.4, 0.01 → 0.21, \|c\| ≥ 0.2 in 66 % of bootstraps) |
| kin s0 | 4 | none |
| kin s1 | 6 | c2h1: 63.5 / 8.3, 0.77 → 0.04; c2h3: 84.5 / 3.5, 0.42 → 0.06 |
| kin s2 | 7 | c2h3: 85.0 / 5.4, 0.86 → 0.10 |
| untrained kin | 1 (c2h1, amplitude 0.005) | – |
| shuffled kin | 0 | – |

## Reading
- **Axis response at the call level (C3s).**
  - H_inv (invariant mass only) predicts no W→Z change under phi beyond the rot null (≤ 0.1 points) and R ≈ 0. It is
    excluded in all six seeds: Δ +13.6 to +19.1 points, McNemar Holm p ≤ 5e-245, R 0.53–0.81 with CIs far from 0.
  - H_axis (√(M0M2) about the stored axis only) predicts R = 1. It is also excluded in every model (upper CI ≤ 0.82), as in
    the released models (0.79, 0.72).
  - The full seeds lie at the released value: R 0.65–0.81 vs 0.79. The kin seeds lie below it: 0.53–0.57 vs 0.72, with a
    seed spread of 0.05 against a gap of 0.15 to the released kin.
  - The untrained networks do not reproduce the call-level response. The full one calls every jet Hcc. The kin one moves
    +4.4 points with R −0.23.
- **S1 (pre-registered window 0.7–1.4).**
  - Holds in full s0 (0.98), full s2 (0.88) and kin s0 (0.74). Fails in full s1 (0.63), kin s1 (0.64) and kin s2 (0.51).
    The released kin is above the window (1.50).
  - The controls show that this ratio does not need a trained network: the untrained full network gives 0.93 and the
    untrained kin 0.02, while the shuffled-label kin network gives −1.03.
  - The linear log-mass probe on any class token picks up whatever per-particle deltaR/dphi signal passes through. So the
    ratio's size depends on the input set and the weights, not only on training to classify. S1 does not separate trained
    from untrained ParT_full.
- **S2.** The pair-only share under phi is ≤ 0.003 in every model. The pair features are invariant under a rigid rotation
  (P1v), so the per-particle share is 1 by construction, including for the untrained and shuffled networks.
  - The pre-registered S2 therefore holds in all six seeds but carries no test of training.
  - The parts that differ between models are these:
    - The ΔR / Δφ split in kin. Released: 0.35 / 0.57. Seeds: 0.58 / 0.29, 0.55 / 0.36, 0.41 / 0.49.
    - The pair route under a real mass change (k1.1). In read-out units it is 0.010 in the released full and 0.11–0.17 in
      the full seeds. In W−Z logit units it is 0.13 in both.
  - The P2 statement "pair route 1–6 % of the read-out change" holds for the released checkpoints only.
- **S3 (pre-registered: class block 1 > 0.9).**
  - Fails in all six seeds: 0.73–0.83 (full), 0.63–0.68 (kin). The released models give 0.98 and 0.97.
  - In the seeds, class block 2 carries 0.17–0.36 (released 0.05–0.07) and particle block 1 carries 0.23–0.33 (released
    0.09–0.12).
  - Against block-stratified random 4+4 class-head sets, block 1 exceeds the random maximum in the full seeds (p 0.032).
    It does not in the kin seeds (p 0.097; random maxima 0.68–0.72).
  - No single class head carries more than 0.22 in any seed; the released c1h3 (full) and c1h7 (kin) carry 0.45–0.46.
  - The literal "particle block 1 > 0.9" fails everywhere, including the released models (0.09, 0.12).
  - Head indices: the largest class head is c1h2, c1h6, c1h2 (full) and c1h3, c1h7, c1h7 (kin). Seeds share no weights
    (|r| ≤ 0.16), so equal indices across seeds have no identity meaning. One shared index among three seeds has chance
    1 − (7/8)(6/8) = 0.34.
  - C2b's "first class-attention block carries the W/Z unit's mass response" is a property of the released checkpoints. In
    the retrained checkpoints the response is spread over both class blocks and the particle blocks.
- **S4 (pre-registered: a head with m50 60–110 GeV, w < 15 GeV).**
  - Holds in every trained model, with 4–9 heads each, mostly class block 2. Untrained full has 0 such heads, untrained kin
    1 (amplitude 0.005), shuffled 0.
  - With P3's amplitude rule (≥ 20 points of share), a class-token share switch exists in 3 of 6 seeds, each at a
    different head and m50:
    - full s1: c2h2 at 105 GeV;
    - full s2: c2h4 at 68 GeV;
    - kin s1: c2h1 at 64 GeV and c2h3 at 85 GeV;
    - kin s2: c2h3 at 85 GeV.
  - Full s0 has only a particle-head switch (p8h6, 76 GeV); kin s0 has none.
  - The released ParT_full's b2h8 (0.87 → 0.08 at 87 GeV) is the largest switch measured; the released kin has none at the
    20-point level.
- **Competing statements of the plan.**
  - "S1, S2 hold in all seeds; head indices differ": S2 holds but is structural, and S1 holds in 3 of 6. The call-level
    response holds in 6 of 6.
  - "S1/S2 fail in some seeds (axis response is a checkpoint accident)": the call-level data exclude it (all six seeds,
    +13.6 to +19.1 points).
  - "A fixed head index recurs": head labels are not comparable across seeds. The one-head dominance of the released models
    (0.45 of the response in one block-1 head; one 80-point switch head) is not reproduced.
- **What this run cannot separate.** The seeds differ from the released models in two ways at once: the random seed and the
  data (2M jets, 25.6 passes vs 100M). The seed-to-seed spread is smaller than the released-vs-seed gap for kin R (0.05 vs
  0.15) and for S3 (≤ 0.10 vs 0.15–0.34). So the route difference is more likely a data-scale or schedule effect than seed
  noise, but this design does not test that.

## Issues
- **Attempt 1** (`*.attempt1.json/log`): two analysis faults, fixed in the rerun (2 attempts in total).
  - S4 used p = max(bootstrap, permutation) with 200 resamples, so the minimum p was 0.005. BH over 80 heads then could not
    reach q < 0.05 even for b2h8. Fixed with 4,000 permutations and the bootstrap used as a ≥ 95 % stability requirement.
  - S3's pooled-mean share was unstable in full s1, where the W and Z responses have opposite signs (pooled 0.36
    [0.20, 0.46]). The per-jet projection is now the primary form.
  - S1 and S2 numbers are identical between attempts (same seeds).
- **Deviations from the pre-registration text.**
  - S3 tests class-attention block 1 (the meaning in C2b / P3v / P8; the P10 text says "particle block 1", which is also
    reported).
  - S4's amplitude criterion was not in the pre-registration. Both forms are reported.
  - S1 uses a per-jet matched dilation as the calibration edit; the P10 calibrated form is also reported.
- The shuffled-label control has 2 outputs, so it has no W/Z calls, R or S3.
- The untrained full network calls every jet Hcc, so its call-level numbers are degenerate.
- The S4 statistic for particle heads (relative attention-weighted pair distance) is new. The amplitude rule of 0.2 in its
  units is not calibrated against the class-share rule.
- Not run: verifier P11v on fresh rows, and C7's jet-mean direction per seed (the P8v DAS direction).

## Explainer entry (for site/explain.py)
- **Question:** Does every Particle Transformer trained on JetClass respond to the jet axis rather than to the invariant
  mass, and does it route that response through the same attention heads as the released network?
- **Background:** A rigid rotation of all particles by $\delta\phi=0.06$ rad about the stored axis keeps the invariant mass
  $m$ fixed but raises the axis moment $\sqrt{M_0M_2}$, with $M_0=\sum p_T$ and $M_2=\sum p_T\Delta R^2$. The released ParT
  turns true-W jets into Z under this edit (C3). In the released ParT_full, the first class-attention block carries 98 % of
  the W/Z unit's response to a mass change (C2b), and head b2h8 switches its attention at 87 GeV (C2a).
- **Method:**
  - Six networks retrained from scratch (P10: full and kin inputs, seeds 0–2, 2M jets).
  - On 20,000 true W/Z test jets, compare the edit $\phi+0.06$ with a matched dilation. The dilation scales each jet's
    angles by $k_j=\exp(\tfrac12\Delta\log M_0M_2)$, which gives the same axis-moment change but a real mass change. Measure
    $R=\sum_j \Delta g_j^{\phi}\Delta g_j^{k}/\sum_j(\Delta g_j^{k})^2$ with $g=\ell_W-\ell_Z$ over true W. An
    invariant-mass tagger gives $R=0$; a pure axis-moment tagger gives $R=1$.
  - Patch the 8 heads of class-attention block 1 back to their unedited values and measure the share of the W/Z unit's
    response to angles × 1.1 that disappears: $1-\sum_j dP_j dE_j/\sum_j dE_j^2$.
  - Scan all 80 heads for a logistic mass switch $y=a+c/(1+e^{(m-m_{50})/w})$.
- **Result (figure `results/part/p11_seed_tests.png`):**
  - Every seed turns more W into Z under the rotation: +14 to +19 points, against +18 to +19 for the released networks. $R$
    is 0.65–0.81 for full (released 0.79) and 0.53–0.57 for kin (released 0.72).
  - The class-block-1 share is 0.63–0.83 in the seeds against 0.97–0.98 in the released networks. No seed has a head as
    dominant as b2h8.
- **Checks:**
  - The released models reproduce 85.93 % / 74.47 % on the P1 sample, b2h8's switch (87.0 GeV, 4.8 GeV wide) and P3v's
    0.98 block-1 share.
  - The whole-jet rotation null gives $R=0.000$ and ≤ 0.1-point call changes.
  - The pair-only route under the rotation is ≤ 0.003.
  - The seeds share no weights (|r| ≤ 0.16).
  - An untrained network gives no call-level response, but it does give a read-out ratio of 0.93 (full), so the read-out
    ratio alone does not show training.
- **Why it matters:** The axis-relative mass response is a reproducible property of ParT trained on JetClass. The specific
  head-level circuit described for the released checkpoint (one block-1 head, one threshold head) is not reproduced by
  retraining at 2M jets. Head-level claims should be stated per checkpoint, or tested at the released data scale.
