# X14 — what sets the axis-tied share R in transformer-type networks (formation over training; init × data-order factorial)

Node X14, stage 3, claims A10s, A10, A10m, A4 (exploratory; checklist item 5 "formation over training").
Script `interp/x14_axis_training.py` (measure + analysis), figure `interp/x14_plot.py`; training flag `train/train_small.py --data_seed`
(opt-in, default None = previous behaviour), job set `x14` in `train/make_jobs.py`.
Results `results/jedi/x14_summary.json`, per-run `results/jedi/x14_runs/<run>__seed<k>.{npz,json}`, figure `results/jedi/x14_axis_training.png`,
analysis log `logs/x14-ana.log`.

## Conclusion
How strongly a transformer or Linformer's mass estimate depends on the stored jet axis (R; 0 = only the true, shift-independent mass, 1 = only the
spread about the axis) is not fixed when the network is initialised. It forms during the middle of training and stays stable once the
validation loss levels off. Training is exactly reproducible: the same starting weights and the same batch order give the same R to
seven digits. But R changes when only the initial weights, only the batch order, or only the length of the learning-rate schedule are
changed. Neither the starting weights nor the batch order alone accounts for most of the spread (about a fifth to a third each, not
distinguishable from chance with 9 runs per model); most of it depends on the specific combination. So R in these models is a product of the whole optimisation path and
is not tied to accuracy. What is still open: which internal parts carry the shift-independent share. Also open: whether 9 runs per
model are enough to see a factor that accounts for ≈ 30 % of the variance.

## Procedure
- **Sample**: `interp.jedi_core.sample(per_class=1000, seed=1313)` = X13's jets; X13's fit / test split (2,500 / 2,500, rng 1313); 1,000 true W
  jets for call flows. Networks with N = 32 see the first 32 particles.
- **R (X13 code imported)**: ridge read-out of log m (true invariant mass of the n particles) on the last hidden layer, **refitted at every
  checkpoint** on the unedited fit half; R = mean_test[mean_k d(T_k)] / mean_test[d(dil)], T_k = rigid 0.06 rad translation of the real
  particles at azimuth 2πk/8, dil = matched dilation (per-jet ½Δlog(M0 M2) equal to its azimuth mean under T). Per direction R_dir
  (k = 0 η+, 2 φ+, 4 η−, 6 φ−). Null edit: rigid rotation about the stored axis (rot / dil). Calls: W → Z fraction among true W under T
  (8-azimuth mean), dil, rot. R_rad (radial-only edit) at epoch 0 and the best epoch only.
- **Oracle (input-only, no network)**: true log m: R 1.5e-8, R_rad 0.797; stored-axis moment ½log(M0 M2): R 1.000, R_rad 1.000; rotation 0.000.
- **Arm 1 (no training)**: 35 existing runs, every saved epoch (e100 runs: 0–30 then every 5th + best): tf-1b/2b-d16/d32 (N 64, 32;
  seeds 0, 1), tf-2b-d16/d32-n64-p30 seed 1, tf-2b-d32/d64-n64-e100 seeds 0, 1, linf-1b-d32-k4 and linf-2b-d32-k8 (N 64, 32; seeds 0, 1),
  dsets-d64 (N 64 seeds 0–2, N 32 seeds 0, 1). Val loss / val acc per epoch from each run's `log.csv` (dev split, 25,000 jets).
- **Arm 2 (new comparison models; official checkpoints untouched)**: tf-2b-d32 and linf-2b-d32-k8, N 64, init seed {0,1,2} × data-order
  seed {0,1,2} (`--data_seed`), 30 epochs cosine, patience 30 (no stop occurred), dirs `models/small/<model>-n64-x14d<data>/seed<init>/`.
  Same per-epoch measurement.
- **Statistics**: jet bootstrap B = 1000 (the same test-jet indices for every run and epoch; W jets resampled separately); settle epoch =
  first evaluated epoch after which |R_t − R_last| < 0.1 at every later evaluated epoch (bootstrap CI); val-loss plateau = first epoch with
  val loss ≤ 1.01 × its minimum, drift = R_last − R_plateau; Spearman across runs with bootstrap over runs × jets and 10,000 permutations;
  grid: two-way decomposition (no replication) of final R into init, data-order, residual (= interaction, since training is deterministic,
  see below), permutation tests 10,000 both full and restricted (init labels permuted within each data-order column; data labels within
  each init row); Spearman of final R with val accuracy / val loss over the 9 runs per model (permutation p).
- **Jobs**: `x14-ckpt-{0..5}-b` (AI_PY=conda, 1 CPU / 2 GiB; 12–36 min; peak ≤ 1.65 GiB; 0.89–0.98 cores), GPU `train-x14-grid-{tf,linf}-s{0,1,2}`
  (+ `-b`/`-c` resubmits; 3 runs per job, 3 CPU / 9 GiB; 6 GiB was OOM-killed at start-up, steady 5.8 GiB, GPU 100 %, 2.8 cores),
  `x14-grid-*` measurements (1 CPU / 2 GiB, ≤ 1.5 GiB), `x14-ana` (137 s, 287 MiB). First submission of the ckpt jobs failed on a missing
  `inputs.axis` key in older `arch.json` (debug 1, fixed). Two GPU pods failed with `UnexpectedAdmissionError` (node side) and were resubmitted.

## Checks
- **Reproduction of X13** at each run's best epoch (33 runs present in X13): max |ΔR| 2.2e-7, max |Δ sample accuracy| 0.0
  (e.g. linf-2b-d32-k8-n64 s0 80.64 %, dsets-d64-n64 s0 80.54 %). The JEDI-linear network was not part of this node; its X13 value
  (80.46 % on this sample) stands.
- **Determinism of training (control for the factorial)**: grid cell (init s, order s) vs the earlier run with the same seed:
  tf-2b-d32 s0, tf-2b-d32 p30 s1, linf-2b-d32-k8 s0, s1: max |Δ val loss| over all epochs 0.0; final R identical (|ΔR| ≤ 7e-9);
  test acc of grid cell (0,0) 76.19 % = original 76.19 %. Also tf-2b-d16/d32-n64 s1 vs their p30 reruns: identical R trajectory to epoch 12/14.
  So the residual term of the grid contains no run-to-run noise; it is the init × order interaction.
- **Epoch 0**: the same init seed gives the same R at epoch 0 for all three data seeds (|Δ| ≤ 3e-7), as it should.
- **Rotation null**: |rot / dil| ≤ 0.038 at every epoch of every run (median 0.004), vs last-epoch R from −1.11 to 2.07 (|R| ≥ 0.14 in every run). Read-out T vs
  rot bootstrap p ≤ 0.002 (floor) for every run at its last epoch. W → Z flow under rotation −0.010 to +0.028 vs under T +0.008 to +0.74
  (largest p of T vs rot 0.30 in a run with flow ≈ 0).
- **Untrained (epoch 0)**: R 0.03–1.86 (random body + refitted read-out); epoch-0 R_rad 0.36–1.84.
- **R_rad at the best epoch vs R** (35 runs): OLS R_rad = 0.88 + 0.22 R (X13v: 0.80 + 0.33 R on 56 networks).

## Results (numbers)
**Formation over training (Arm 1, 35 runs; Arm 2, 18 runs).** Figure (b).
| group | n | settle epoch median [range] | settle / last epoch (median) | abs. drift after val-loss plateau, median (max) |
|---|---|---|---|---|
| Deep Sets d64 | 5 | 6 [4, 16] | 0.20 | 0.016 |
| Linformer (arm 1) | 8 | 17.5 [1, 20] | 0.58 | 0.025 |
| transformer (arm 1) | 22 | 12.5 [3, 65] | 0.47 | 0.017 (0.14, tf-2b-d32-n64 s1, early-stopped at epoch 12) |
| grid tf-2b-d32 | 9 | 16 [4, 21] | 0.53 | 0.022 |
| grid linf-2b-d32-k8 | 9 | 13 [5, 18] | 0.43 | 0.009 |
e100 runs settle at epochs 50–65 of 100, before their val-loss plateau (epochs 85–90). Drift CIs after the plateau exclude 0 in most runs
(bootstrap p at the floor 0.002) but are ≤ 0.14 in size; with B = 1000 and 53 runs Holm cannot go below 0.05 (0 of 53 after Holm).

**Early R vs final R (Spearman across runs).**
| set | n | epoch 0 | epoch 1 | epoch 3 | epoch 5 | epoch 10 |
|---|---|---|---|---|---|---|
| all attention runs (arm 1 + grid) | 48 | −0.16 [−0.46, 0.15] p 0.26 | +0.39 [0.08, 0.62] p 0.009 | +0.61 [0.38, 0.76] p 1e-4 | +0.73 [0.54, 0.85] | +0.80 [0.64, 0.90] |
| arm 1 attention | 30 | −0.19 p 0.30 | +0.20 p 0.28 | +0.44 [0.09, 0.73] p 0.016 | +0.66 | +0.78 |
| grid tf-2b-d32 | 9 | −0.36 p 0.33 | +0.47 p 0.21 | +0.40 [−0.46, 0.95] p 0.29 | +0.33 p 0.39 | +0.65 p 0.065 |
| grid linf-2b-d32-k8 | 9 | −0.78 p 0.016 | +0.03 p 0.95 | −0.28 [−0.94, 0.58] p 0.47 | −0.12 | +0.30 p 0.44 |
Pooled correlations include the difference between architectures (Linformer R 1.06–1.49 vs tf-2b-d32 0.42–0.97 in the grid); within one
architecture the epoch-3 value does not predict the final value (grid, both models).

**Learning-rate schedule length (same init, same batch order; tf-2b-d32-n64).** Figure (a). Trajectories coincide to ≈ epoch 20, then separate:
seed 0: 30-epoch schedule R_last 0.51 [0.49, 0.52] vs 100-epoch schedule 1.23 [1.21, 1.25] (at epoch 30 of the long run 0.78);
seed 1: 0.65 [0.63, 0.66] (p30 run) vs 0.46 [0.42, 0.49].

**Init × data-order grid (final R, epoch 30; rows init 0–2, columns data order 0–2).** Figure (c).
| model | R matrix | share init [jet CI], p (within order / full) | share order, p | residual (interaction) | SD of R | ρ(R, val acc), p | ρ(R, val loss), p |
|---|---|---|---|---|---|---|---|
| tf-2b-d32 | [[0.51, 0.71, 0.77], [0.74, 0.65, 0.42], [0.68, 0.97, 0.65]] | 0.21 [0.18, 0.24], 0.47 / 0.52 | 0.23 [0.22, 0.24], 0.42 / 0.46 | 0.56 [0.53, 0.58] | 0.16 | 0.00, 1.00 | +0.08, 0.85 |
| linf-2b-d32-k8 | [[1.49, 1.26, 1.46], [1.46, 1.26, 1.06], [1.26, 1.20, 1.19]] | 0.34 [0.26, 0.42], 0.25 / 0.30 | 0.31 [0.27, 0.34], 0.27 / 0.41 | 0.35 [0.29, 0.41] | 0.15 | +0.50, 0.18 | −0.63, 0.074 |
F (init) 0.75 / 1.94, F (order) 0.83 / 1.75 (df 2, 4). Under no structure each factor's expected share is 0.25 (2 of 8 df).
Per-run 95 % jet CIs of final R have half-width 0.01–0.05, so the 0.42–0.97 and 1.06–1.49 spreads are not jet noise.
Val accuracy over the grid: tf 75.9–76.5 %, linf 80.2–80.9 % (sample acc at the best epoch).

**Direction (phi+ / phi− / eta+ / eta−) at the last epoch**: |R_φ+ − R_φ−| median 0.38, max 2.14 (tf-1b-d32-n32 s1: φ+ −2.30, φ− −0.16;
tf-1b-d32-n64 s1: η+ 0.96, φ− 0.06); Deep Sets d64-n64 s0 0.72–0.91 across the four directions.

**Seed spread of final R (arm 1, 2–3 seeds)**: Deep Sets d64 SD 0.07 (N 32 and N 64); Linformer 0.15–0.73; transformers 0.06–1.22.

## Hypotheses
- **H0** (final R does not depend on init seed beyond data order): not rejected (restricted permutation p 0.47 tf, 0.25 linf; 9 runs per model;
  minimum detectable share with 216 distinct restricted permutations is not small, so a ≈ 0.3 init share is not excluded).
- **H_init** (init fixes R early; settle in the first few epochs; early R predicts final R with ρ > 0.7): predictions not met — epoch-0 R has
  ρ ≤ 0 with final R, settle median 13–17 of 30 epochs, within-architecture epoch-3 ρ 0.40 (tf) / −0.28 (linf); the same init with a
  different batch order or schedule length ends at R 0.42–0.74 (tf, init 1). Excluded as the sole determinant.
- **H_data** (data order dominates): order share 0.23 / 0.31, p 0.42 / 0.27; not supported.
- **H_drift** (R keeps moving after val loss stops improving): change after the plateau ≤ 0.14 (median 0.01–0.03), below the 0.1 settle
  tolerance in 51 of 53 runs (0.10 tf-1b-d32-n64 s0, 0.14 early-stopped tf-2b-d32-n64 s1); the large changes (schedule-length pair, panel a) happen while val loss is still falling. The flat-direction
  version is not supported; R is set during the mid-training, high-learning-rate phase, and depends on the path taken there.
- **H_acc** (R trades off against accuracy at fixed architecture): ρ(R, val acc) 0.00 (tf, p 1.0) and +0.50 (linf, p 0.18); not supported at
  this n; the linf val-loss correlation (−0.63, p 0.074) is in the direction "higher R, lower loss", opposite to a trade-off.
- **Set networks**: Deep Sets settle earlier (median epoch 6, 0.2 of training) with seed SD 0.07, consistent with X13 (family sets R among set networks).

## Reading (scoped)
In tf-2b-d32 and linf-2b-d32-k8 at N = 64 the final axis-tied share R is a deterministic function of (initial weights, batch order,
schedule): changing any one of the three changes R by up to ≈ 0.5–0.7, with no single factor carrying a majority of the grid variance
and the largest part in their interaction. R forms between roughly epoch 5 and 20 of a 30-epoch cosine schedule (epochs 40–65 of a
100-epoch schedule) and changes ≤ 0.14 after the val-loss plateau. Across architectures, R at epoch 5–10 predicts final R (ρ 0.73–0.80,
48 runs), partly through architecture-level differences. For the A10s wording: "varies with the seed" in attention models can be stated as
"varies with the training path (initialisation, batch order and schedule jointly), not with initialisation alone".
Limits: one N (64), two architectures, 9 runs per model (low power for a single-factor share ≈ 0.3); R uses a refitted ridge read-out
at each epoch (what the last layer encodes), not the network's own logits (call flows reported alongside); only two schedule-length
pairs.

## Explainer entry (for site/explain.py, Paper A tab, A10s)
**Question.** When a transformer is trained twice with different random seeds, its mass estimate can depend on the stored jet axis by very
different amounts (X13). Is that amount fixed by the random starting weights, by the order in which training jets are shown, or by
something that happens later in training?

**Background.** A rigid shift of all particles by $c = 0.06$ rad leaves the invariant mass $m$ unchanged but changes the second moment
about the stored axis, $M_2 = \sum_i p_{T,i} r_i^2$. We read the network's mass estimate with a linear read-out $\hat\ell$ of $\log m$ on its
last hidden layer and define
$$R = \frac{\langle \hat\ell(T_c x) - \hat\ell(x)\rangle}{\langle \hat\ell(D_s x) - \hat\ell(x)\rangle},$$
where $T_c$ is the shift (averaged over 8 directions) and $D_s$ a dilation with the same change of $\tfrac12\log(M_0 M_2)$. An
input-only read-out of the true $\log m$ gives $R = 0$; one of $\tfrac12 \log(M_0M_2)$ gives $R = 1$.

**Method.** (1) For 35 existing small-tagger runs we recompute $R$ from every saved epoch checkpoint on the same 2,500 jets (read-out
refitted per epoch). (2) We train tf-2b-d32 and linf-2b-d32-k8 nine times each on a $3\times3$ grid of initialisation seed × batch-order
seed and split the variance of the final $R$: $SS_{tot} = SS_{init} + SS_{order} + SS_{int}$, testing each share by permuting labels.
(3) Controls: the same seeds rerun (training is deterministic), rotation about the axis (no change of $M_2$), epoch-0 networks.

**Result.** Figure `x14_axis_training.png`: (a) identical start and batch order, only the schedule length differs: final $R$ 0.51 vs 1.23;
(b) $R$ forms in the first half of training and is stable after the validation loss levels off; Deep Sets settle earlier; (c) grid: neither
factor dominates (init 0.21 / 0.34, order 0.23 / 0.31 of the variance; $p \ge 0.25$), the rest is their interaction.

**Checks.** Final-epoch values equal X13's to $2\times10^{-7}$; reruns with the same seeds give identical $R$ and validation loss; rotation
response $\le 0.04$ of the dilation response; 95 % jet-bootstrap intervals on every $R$ (half-width ≤ 0.05).

**Why it matters.** The axis-tied share in attention models is a property of the individual training run, not of the architecture or of
the initial weights alone, so cross-network statements about it (A10s) are made at the level of architecture families for set networks and
as a distribution over training runs for transformers.
