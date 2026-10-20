# X9w: input representation vs training as the source of the axis-moment mass response (2 × 2 on PFN, EFN, Deep Sets, JEDI-linear)

Node X9w (stage 2), claims A10m / A4 / D4 (review weakness 1: novelty vs EFN/PFN). Date 2026-10-20.

## Conclusion
Set networks that see jets in the stored-axis format respond to sliding the whole jet sideways by 0.06 rad as if its mass had grown,
and this response disappears in PFN, EFN and Deep Sets when the same networks are trained either on jets re-centred on their
pT-weighted centre or with random sideways shifts (augmentation), costing at most about 0.9 points of accuracy. So for these three
architectures the response comes from the input format plus ordinary training, not from anything specific to one network. In
retrained JEDI-linear (quantization-aware HGQ training) augmentation removed the response in only 1 of 3 seeds (the other two still
send 33–47 % of W jets to Z) and cost 4–14 points of accuracy, so whether JEDI-linear's architecture/quantized training keeps an
axis-relative term, or the augmented HGQ runs are simply under-trained, is not separated by this run.

## Question
Is the un-recentred, M2-like mass response (W → Z when every particle is translated by 0.06 rad; M2 = Σ pT r² about the stored axis)
a consequence of the stored-axis input representation, shown by any set network trained on it, or specific to JEDI-linear's
architecture / HGQ training?

## Procedure
Training (opt-in flags; defaults reproduce the earlier runs; official checkpoints untouched):
- `--recentre`: every real particle's (Δη, Δφ) minus the jet's pT-weighted centroid c = Σ pT_i (Δη_i, Δφ_i) / Σ pT_i (first 64 slots),
  before normalisation, for train, dev and test jets (normalisation constants unchanged).
- `--aug 0.08`: training jets only; every real particle of a jet translated by the same vector, |c| ~ U(0, 0.08) rad, direction ~
  U(0, 2π), drawn per jet and redrawn every epoch (Deep Sets: every batch); applied after recentring in the recentred + aug cell
  (applied before, the two would cancel exactly).
- Files: `interp/x9_train.py` (PFN Φ 64-64-32, EFN Φ 100-100-128, F 100-100-100; epochs ≤ 60, ≤ 55 min wall), `train/train_small.py`
  (dsets-d64, N 64), `train/train_hgq_jedi.py` (JEDI-linear n64, arm hgq = HGQ-50: 200 epochs, zero-fraction ramp to 0.896 by epoch
  50), `train/gpu_job_x9w.sh`, `train/x9w_cpu_batch.sh`, `train/make_jobs.py` set `x9w`.
- Cells {stored, recentred} × {no aug, aug}; seeds 0, 1 for PFN, EFN, Deep Sets; JEDI-linear seed 0 in all cells plus seeds 1, 2 in
  the stored and aug cells (added after the seed-0 aug run ended at 64.4 % test accuracy; stored seeds 1, 2 = X12 checkpoints).
  Stored / no-aug cell = X9 (PFN, EFN), small-model (dsets-d64-n64) and X12 (jedi-n64-x12-hgq) checkpoints.
- Jobs: x9w-pfn (12 CPU / 30 GiB, 6 runs, 25 min; peak 18.6 GiB, 8.3 cores), x9w-efn (16 CPU / 36 GiB, 6 runs, 57 min; peak 17.4 GiB,
  13.3 cores), train-x9w (GPU: 6 Deep Sets + 3 JEDI-linear), train-x9w-c (JEDI rcaug rerun), train-x9w-d (JEDI aug seeds 1, 2).

Evaluation: `interp/x9w_repr_2x2.py` (job x9w-run, 1 CPU / 2 GiB, 654 s, peak 1.07 GiB; Deep Sets state_dicts exported to npz in the
torch env first). Sample `jedi_core.sample(2000, 909)`, N = 10,000 (2,000 true W). Edits on the raw jets (padding never moved); for
recentred models every edit is applied before the model's recentring step, so the reported translation response is the residual.

| edit | definition | Δlog m (mean) | Δlog M2 about stored axis (mean) | role |
|---|---|---|---|---|
| T | all real particles +0.06 rad in Δη | 0.000 | +0.610 | test edit |
| Tr | +0.06 rad, random direction per jet (rng 9091) | 0.000 | +0.610 | null (equal length) |
| rot | rotation about the axis, mean displacement 0.06 rad (X6 construction, rng 9092) | 0.000 | 0.000 | null |
| M2held | Tr, then angles × √(M2/M2(Tr)) (X9v) | −0.305 | 0.000 | control |
| dilM2 | angles × √(M2(T)/M2) (X12v), mean factor 1.414 | +0.305 | +0.610 | matched real change |

Metrics: W → Z = fraction of true-W jets called Z; flow = W → Z(edit) − W → Z(unedited), 2,000-draw bootstrap over W jets; calls
changed (all jets, Wilson); exact McNemar on paired Z indicators (W jets): edit vs unedited, T vs rot; calibration ratio
flow(T)/flow(dilM2) with paired bootstrap; Holm over the 33 networks per test. Seed-level: exact permutation of networks over cell
labels per architecture (8 networks, 40,320 permutations); stratified by architecture over PFN/EFN/Deep Sets (exact, 216
relabellings, for stored vs one other cell; 20,000 random draws for the 2 × 2 effects); JEDI stored vs aug exact 3 vs 3 (20
relabellings, minimum two-sided p 0.10).

Checks: numpy forward vs TensorFlow softmax (PFN/EFN, sample(2000, 313)) max |Δp| ≤ 6.5e-6; Deep Sets numpy vs torch max |Δlogit|
≤ 1.9e-5, vs stored test_probs argmax 100 %; retrained JEDI-linear vs the stored GPU test_probs argmax agreement 99.41–99.79 % (max
|Δp| 0.14–0.30, GPU float16 input rounding as in X12). Official JEDI-linear on this sample: 80.56 % [79.8, 81.3] (test archive
80.92 %; the requested 80.6–81.3 % band is missed by 0.04 points, within the interval).

## Results

### Translation T: W → Z flow [95 % CI] (rotation null flow in parentheses), test accuracy (260,000 jets)
| arch | seed | stored | stored + aug | recentred | recentred + aug |
|---|---|---|---|---|---|
| PFN | 0 | 0.522 [0.498, 0.547] (−0.002); 81.23 % | 0.012 [0.004, 0.019] (0.004); 80.88 % | 0 (residual; −0.003); 81.14 % | 0 (−0.007); 80.78 % |
| PFN | 1 | 0.697 [0.675, 0.718] (−0.003); 81.11 % | 0.007 [0.002, 0.013] (0.000); 81.29 % | 0 (0.001); 81.33 % | 0 (−0.001); 81.31 % |
| EFN | 0 | 0.356 [0.333, 0.379] (0.003); 74.00 % | 0.004 [−0.004, 0.011] (0.000); 73.40 % | 0 (−0.001); 73.84 % | 0 (0.004); 73.54 % |
| EFN | 1 | 0.667 [0.645, 0.689] (0.003); 74.38 % | 0.003 [−0.003, 0.007] (−0.003); 73.45 % | 0 (0.000); 73.95 % | 0 (−0.005); 73.45 % |
| Deep Sets | 0 | 0.638 [0.616, 0.659] (0.004); 80.64 % | −0.001 [−0.006, 0.005] (0.010); 80.03 % | 0 (0.011); 80.89 % | 0 (0.009); 80.42 % |
| Deep Sets | 1 | 0.683 [0.660, 0.705] (0.004); 80.81 % | −0.001 [−0.007, 0.005] (0.012); 79.68 % | 0 (0.009); 80.55 % | 0 (0.005); 80.02 % |
| JEDI-linear HGQ-50 | 0 | 0.577 [0.550, 0.601] (−0.001); 78.42 % | 0.325 [0.302, 0.348] (−0.030); 64.35 % | 0 (−0.002); 78.56 % | 0 (−0.001); 75.60 % |
| JEDI-linear HGQ-50 | 1 | 0.538 [0.512, 0.562] (0.002); 78.69 % | −0.019 [−0.030, −0.007] (0.000); 75.04 % | – | – |
| JEDI-linear HGQ-50 | 2 | 0.718 [0.695, 0.739] (−0.009); 80.21 % | 0.467 [0.445, 0.489] (−0.006); 74.24 % | – | – |
| official JEDI-linear | – | 0.721 [0.698, 0.742] (−0.001); 80.92 % | | | |

Absolute W → Z under T, stored cell: PFN 0.555 / 0.720, EFN 0.384 / 0.701, Deep Sets 0.663 / 0.709, JEDI-linear 0.652 / 0.599 /
0.766, official 0.751 (unedited 0.024–0.075). Stored + aug, PFN/EFN/Deep Sets: 0.020–0.040 (unedited 0.021–0.030); calls changed under
T 5.3–9.7 % vs rotation 4.3–6.9 %, although max |Δlogit| under T is 3.3–7.4 (the logits move, the calls mostly do not).
Recentred models: residual max |Δlogit| ≤ 7.3e-6 (flow models, Deep Sets) and 0 (JEDI-linear, fixed point); 0 calls changed.
Random-direction translation Tr (flow): stored JEDI 0.597 / 0.546 / 0.589; JEDI aug 0.273 / 0.022 / 0.232 (T: 0.325 / −0.019 /
0.467), so in aug seeds 0 and 2 the +Δη direction moves 0.05–0.24 more W jets to Z than a random direction of equal length.

### Controls
- M2-held translation, W → Z flow: stored cells PFN 0.076 / 0.038, EFN 0.076 / 0.029, Deep Sets 0.000 / 0.041, JEDI-linear 0.052 /
  0.128 / 0.056, official 0.013 (9 of 10 at or below 0.1; JEDI seed 1 0.128). Aug and recentred cells: −0.012 to −0.034 (PFN, EFN,
  Deep Sets), −0.018 to −0.170 (JEDI-linear). The same edit sends W jets to g/q (lower mass): stored 0.045–0.097 of W jets vs aug /
  recentred 0.30–0.95 (PFN 0.63–0.93, EFN 0.30–0.40, Deep Sets 0.37–0.55, JEDI-linear 0.49–0.95); its invariant-mass change is
  −0.305 in log m, so networks trained without axis information treat it as a real mass decrease.
- Matched real change dilM2 (same M0, M2 about the axis as T; invariant mass × 1.41), W → Z flow: every cell responds — stored PFN
  0.491 / 0.686, EFN 0.234 / 0.251, Deep Sets 0.447 / 0.483, JEDI 0.541 / 0.526 / 0.501; aug PFN 0.565 / 0.576, EFN 0.346 / 0.317,
  Deep Sets 0.472 / 0.448, JEDI 0.427 / 0.671 / 0.714; recentred PFN 0.538 / 0.629, EFN 0.260 / 0.191, Deep Sets 0.464 / 0.444,
  JEDI 0.715; recentred + aug PFN 0.521 / 0.576, EFN 0.320 / 0.382, Deep Sets 0.532 / 0.552, JEDI 0.729.
- Calibration ratio flow(T)/flow(dilM2): stored PFN 1.06 [1.02, 1.11] / 1.02 [0.98, 1.05], EFN 1.52 [1.42, 1.64] / 2.66 [2.47, 2.89],
  Deep Sets 1.43 [1.37, 1.49] / 1.41 [1.36, 1.47], JEDI 1.07 [1.03, 1.10] / 1.02 [0.98, 1.07] / 1.43 [1.38, 1.49], official 1.24
  [1.20, 1.28]; aug PFN 0.02 [0.01, 0.03] / 0.01 [0.00, 0.02], EFN 0.01 [−0.01, 0.03] / 0.01 [−0.01, 0.02], Deep Sets −0.00 [−0.01,
  0.01] / −0.00 [−0.02, 0.01], JEDI 0.76 [0.71, 0.81] / −0.03 [−0.05, −0.01] / 0.65 [0.62, 0.69]; recentred 0.

### Hypothesis tests
- Per network, H0: T moves no more W jets to Z than the rotation (McNemar, Holm over 33): stored cells and official p_Holm ≤ 3.6e-149
  (all 10 + official); aug PFN/EFN p_Holm 0.79–1; aug and recentred Deep Sets p_Holm 0.018–0.049 in the opposite direction (rotation
  moves more W to Z than T: 34–42 vs 12–17 discordant jets); JEDI aug seeds 0 and 2 p_Holm 3.6e-149 and 5.4e-252, seed 1 0.088
  (opposite direction).
- Seed level, PFN/EFN/Deep Sets stratified by architecture (exact, 216 relabellings; minimum attainable two-sided p 0.0093): flow(T)
  stored − aug 0.590, p 0.0093; stored − recentred 0.594, p 0.0093; aug − recentred+aug 0.004, p 0.056. Accuracy stored − aug 0.57
  points, p 0.028; stored − recentred 0.08 points, p 0.48.
- 2 × 2 effects (pooled, stratified, 20,000 draws), flow(T): representation 0.299 (p 0.0002), augmentation 0.295 (p 0.017),
  interaction 0.590 (p 0.018). Because three of the four cells are at 0, the three effects are not separable (each equals half or
  all of the stored-cell value); per architecture the exact 8-network test reaches p 0.029 for representation in PFN and EFN and
  0.17–0.43 otherwise (2 seeds per cell).
- JEDI-linear stored vs aug (seeds 0–2, exact 3 vs 3): flow(T) difference 0.353, p 0.10 (minimum attainable); accuracy difference
  7.9 points, p 0.10.

### Accuracy cost of augmentation (test archive, mean over seeds)
PFN −0.08 points, EFN −0.76 (the aug runs stopped at the 55-min wall at 35 epochs; at ≤ 35 epochs the stored runs' best dev
accuracy is 74.06 / 74.07 % vs aug 73.53 / 73.42 %: −0.59), Deep Sets −0.87, JEDI-linear −7.9 (seeds 1, 2 only: −4.8; recentred +
aug vs recentred, seed 0: −3.0). Recentring alone: PFN +0.07, EFN −0.30, Deep Sets −0.01, JEDI-linear +0.14.

### Predictions vs outcome
| prediction | outcome |
|---|---|
| Thesis: stored / no-aug W → Z 0.45–0.8 in every architecture | 10 of 11 networks (EFN seed 0: 0.384) |
| Thesis: aug alone → rotation level (< 0.05) at ≤ 1 point accuracy | PFN, EFN, Deep Sets: 6/6 (W → Z 0.020–0.040; cost 0.08–0.87 points); JEDI-linear: 1/3 seeds, cost 4–14 points |
| Thesis: recentred cells ~0 | 13/13 exactly 0 (invariant by construction) |
| Thesis: M2-held ≤ 0.1 in all cells | 22/23 networks (JEDI stored seed 1: 0.128) |
| Alternative: response persists in stored + aug | JEDI-linear seeds 0, 2 (0.325, 0.467); not in PFN, EFN, Deep Sets |
| Alternative: accuracy loss > 2 points under aug (stored axis carries used class information) | JEDI-linear only (4–14 points; also 3.0 points in recentred + aug, where the stored axis is removed) |

Excluded for PFN, EFN and Deep Sets: an architecture-specific mechanism (the response follows the training input format; p 0.0093
at the seed level). For JEDI-linear HGQ-50 two readings remain: (a) the architecture / quantized training keeps an axis-relative,
partly direction-specific (+Δη) term even when translations carry no class information; (b) the augmented HGQ runs did not reach a
translation-invariant solution within 200 epochs (accuracy 64–75 % vs 78–80 %; the recentred + aug run, which cannot see the
stored axis, also loses 3.0 points). The 3.0-point loss in recentred + aug, where no axis information exists, points to (b) for
the accuracy loss; it does not decide the source of the residual response.

Figure: `results/jedi/x9w_repr_2x2.png` ((a) T flow per cell with rotation and M2-held markers and jet-bootstrap CIs; (b) test
accuracy). Results: `results/jedi/x9w_repr_2x2.json`; log `logs/x9w-run.log`.

## Issues and deviations
- JEDI-linear aug seed 0 ended at 64.35 % (HGQ-50 runs oscillate strongly between epochs 30 and 150 in every cell, including the X12
  stored run: val 0.44 at epoch 55); seeds 1 and 2 were added for the stored and aug cells only. Recentred cells have seed 0 only.
- The first JEDI rcaug run ran out of GPU memory at epoch 52 (three HGQ runs sharing one GPU, a new 464 MB device array per epoch);
  rerun alone (train-x9w-c) with the old array freed explicitly. A host-resident array (train-x9w-b) made jax run the step on the
  CPU (46 s/epoch, GPU idle) and was stopped; `--aug_device` defaults to gpu:0. The Kubernetes retry of train-x9w duplicated the
  rcaug run for ~1 min before it was deleted (it wrote to the same directory; the kept run rewrote log.json and checkpoints).
  logs/train-x9w-jedi.log on the volume holds interleaved output of train-x9w, -b, -c, -d (same wrapper job name); per-run
  train.log / log.json in the model directories are complete.
- EFN aug / recentred+aug runs stopped at the 55-min wall (35–36 epochs, best epoch = last); EFN stored seed 1 ran 59 epochs in X9.
  EFN accuracy comparisons therefore also quote the matched-epoch dev accuracy.
- Deep Sets ran on the GPU via train/gpu_job.sh (train/ tooling) rather than CPU jobs. train/ files were edited as the plan
  specifies (opt-in flags only).

## Explainer entry (for site/explain.py, Paper A, claim A10m)
**Question.** Does a set network read the jet mass from the spread of particles about the stored jet axis because of the input
format, or because of something specific to JEDI-linear?

**Background.** The hls4ml jets store each particle's angles $(\Delta\eta_i, \Delta\phi_i)$ relative to a fixed jet axis.
Translating every particle by the same vector $\vec c$ leaves the invariant mass $m$ unchanged but raises
$M_2 = \sum_i p_{T,i}\, r_i^2$ about the stored axis: $M_2 \to M_2 + 2\vec c\cdot\sum_i p_{T,i}\vec r_i + |\vec c|^2 M_0$.
Earlier nodes (E6, X9, X12) showed that a $0.06$ rad translation sends 50–77 % of W jets to Z in JEDI-linear, PFN, EFN and Deep Sets.

**Method.** Retrain PFN, EFN, Deep Sets and JEDI-linear in a $2\times2$ design: inputs stored or re-centred on the
$p_T$-weighted centroid $\vec c_J = \sum_i p_{T,i}\vec r_i / \sum_i p_{T,i}$, and training with or without random translations
$|\vec c| \sim U(0, 0.08)$ rad. Apply the $0.06$ rad translation to 10,000 test jets and count W jets newly called Z; nulls:
rotation and random-direction translation of equal length; control: translation with $M_2$ held fixed; calibration: a real
widening with the same $M_0, M_2$.

**Result.** (figure x9w_repr_2x2.png) Stored-axis networks: flow 0.36–0.72. Augmented PFN/EFN/Deep Sets: flow ≤ 0.012 (rotation
level) at ≤ 0.9 points of accuracy. Recentred networks: exactly 0. Augmented JEDI-linear: 2 of 3 seeds keep 0.33–0.47, with a 4–14
point accuracy loss.

**Checks.** Every network still responds to a real widening of the jet (flow 0.19–0.73), so augmented and recentred networks still
read mass, from the spread about the jet's own centre: a translation-plus-shrink edit that holds $M_2$ about the axis sends 30–95 %
of their W jets to g/q, against 5–10 % for stored-axis networks.

**Why it matters.** The axis-moment mass reading is what any of these set networks learns from the stored-axis format without
augmentation; it is not evidence for a JEDI-specific mechanism. The JEDI-linear HGQ case is open (training quality vs architecture).
