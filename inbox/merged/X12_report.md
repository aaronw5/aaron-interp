# X12: retrained JEDI-linear n64 (HGQ seeds and a pipeline-matched float arm): which program elements recur

Node X12, round 6, stage 2. Claims A1, A1q, A3, A10; review weaknesses 2, 6, 7 (absorbs X5w and X8w).
Training: `train/train_hgq_jedi.py`, `train/gpu_job_x12.sh`, `train/make_jobs.py` set `x12` (jobs `train-x12`, `train-x12-r150`).
Battery: `interp/x12_retrained_jedi.py` (analysis plan frozen in the module header; additions after the first runs are listed
under "Changes to the plan"). Figure: `interp/x12_plot.py` → `results/jedi/x12_final.png`.
Results: `results/jedi/x12_<model>_final.json` (official, hgq-r150_s0–2, hgq_s0–2, float_s0–1), `results/jedi/x12_summary_final.json`.
Logs: `logs/x12-*.log`, `logs/train-x12*.log`. Models: `/j-jepa-vol/aaron-interp/models/small/jedi-n64-x12-{hgq-r150,hgq,float}/seed<k>/`
(the official checkpoints were not touched).

## Conclusion
Five JEDI-linear networks were retrained from scratch with the same architecture. Three used the official fixed-point
(quantization-aware) training. Two used the same script with quantization switched off. All retrained networks with
quantization-aware training reproduce the official network's mass behaviour: shifting every particle by 0.06 rad turns most
true W jets into Z calls (60–78 % in the five usable copies), and this disappears when the shift is undone in Σ pT r². The
networks trained without quantization respond much less (21–35 %). The official network's strongest rank-window effect (a
particle swap across the "14 hardest particles" boundary changes the pooled objects 5.3× more than a swap inside) is not
reproduced by any retrained copy, although all of them form zero-bit slot masks. The fraction of first-layer units that read pT
only or angles only is 100 % in every quantized network, but at that sparsity (13–20 nonzero weights) this follows from the
weight count alone. Uncertain: the retrained quantized copies are 1–4 points less accurate than the official network and
trained 200–300 instead of 2,868 epochs, and one of the three primary seeds collapsed to 57 % at its last epoch.

## Procedure
**Training.** All runs used GPU Jobs (RTX 2080 Ti) with the read-only keras 3 / jax / hgq2 0.1.9 env, code snapshot under
`models/small/_snapshots/x12-*`.
- Architecture: `train_hgq_jedi.build` reproduces the official `get_gnn` n64 3-feature network:
  - l1 → (qs pooling → dd = context g) + l2 → add → l3 → qs1 pooling (S) → d4, d5, d6, d7.
  - HGQ2 quantizer scopes as in the official `get_model`.
  - Check: the parameter count (142,437) and every layer's kernel and input-quantizer table shapes are identical to the
    official checkpoint's (battery `arch` record, `x12_official_final_smoke.json` vs `x12_hgq-smoke_s0_final_smoke.json`).
- Data and optimizer:
  - Inputs: `(x − shift)/scale` with `norm_stats.json['64']`, stored as float16.
  - Training rows: train archive minus `splits.npz['dev']` (595,000); validation: dev (25,000); test: val archive (260,000).
  - Optimizer: Adam, batch 2,790.
  - Learning rate: the official cosine decay with restarts (3e-3, period 100 epochs).
  - tf32 matmul, no early stopping. Every result below uses the last epoch.
- Arms:
  - **float** (seeds 0, 1): the same script with every quantizer replaced by hgq's identity quantizer and EBOPs off; 200 epochs.
  - **HGQ-50** (seeds 0, 1, 2; secondary). 200 epochs. Beta is set at each epoch start by a multiplicative controller (×/÷ 1.3)
    on the l1 zero fraction. The controller's target reaches 0.896 (the official fraction, 172 of 192 l1 entries zero) by
    epoch 50; beta ≤ 1e-4.
  - **HGQ-150** (seeds 0, 1, 2; primary). 300 epochs, target reached at epoch 150, beta ≤ 3e-6 (the official schedule's
    maximum). This arm was added after HGQ-50's train accuracy fell to 0.59–0.61 at epoch 58 with beta at 1–4e-5 (first
    training debug attempt). Primary vs secondary was fixed before any battery result of either arm (comment in `MODELS`).
- Final test accuracy (260,000 jets), with the l1 zero fraction in parentheses:

  | arm | seed 0 | seed 1 | seed 2 |
  |---|---|---|---|
  | float | 82.46 % (0.000) | 82.38 % (0.000) | – |
  | HGQ-50 | 78.42 % (0.911) | 78.69 % (0.932) | 80.21 % (0.901) |
  | HGQ-150 | 77.04 % (0.896) | **57.03 %** (0.901) | 78.10 % (0.896) |
  | official | 80.92 % (0.896, 2,868 epochs) | | |

  HGQ-150 seed 1 had validation accuracy 0.728 at its last logged epoch, then fell to 57 % at epoch 300. That network
  calls 49 % of unedited true-W jets Z.

**Battery.** One cluster Job per model, 1 CPU / 2 GiB.
- Sample: `interp.jedi_core.sample(per_class=5000, seed=0)`, N = 25,000 test jets (N_W = 5,000). Unit costs use x[::5]
  (5,000 jets).
- Forward: a generic jit forward (stages a, P, g, s, u, S, h4, h5, z, logits) with each layer's own input quantizer and
  fused quantized kernel. Exactness checks:
  - Official: identical to `Jedi.run`, max |Δlogit| = 0.
  - Retrained: identical to the Keras model run on CPU, max |Δlogit| = 0 (Keras checks on 2,000 jets in
    every run). The Keras model on float16-rounded inputs reproduces the stored GPU
    `test_probs` to ≤ 2e-7 for HGQ models and ≤ 0.007 for float models (tf32).
  - Input rounding: the battery feeds float32 inputs. Float16 rounding (the training dtype) changes 0.15–0.4 % of the
    HGQ calls.
- Steps (details in the module header):
  1. Accuracy: sample accuracy with Wilson CI, plus test accuracy.
  2. Pure cuts:
     - A unit is live if replacing it by its mean changes ≥ 1 % of calls.
     - Physical form α_pT pT + α_η Δη + α_φ Δφ + β. A unit is pure if ≥ 90 % of the input spread comes from pT alone or from
       the angles alone.
     - P_nz = the pure fraction among nonzero units.
     - Null: each feature column of α permuted over the 64 units, 2,000 draws.
  3. Zero-bit masks of the qs and qs1 input quantizers (bits ≤ 0) on live channels.
     - A window channel is partially kept and contiguous (Holm p < 0.01 vs random kept ranks, or one run).
  4. Rank-window swap test (E20):
     - Edges: the two edges k|k+1 with the largest summed S cost among live S units whose qs1 mask changes there (ranks
       held by ≥ 50 % of jets).
     - Inside pair: the nearest adjacent pair with no qs1 edge.
     - Swap the (Δη, Δφ) of the pair. m = mean over the edge's units of |ΔS_c| / sd(S_c).
     - Statistics: ratio across/inside with a 2,000-draw jet bootstrap, paired sign-flip p, McNemar on call changes.
     - Float models have no masks, so they use ranks 13|14 vs 5|6 and the 10 live S units of largest cost.
     - Null (X8w): mask-keeping reassignment copies. A permutation of the live S channels is applied to the l3 output
       columns, the d4 input rows and the d4 input-quantizer tables, with all masks unchanged. A draw is kept if accuracy is
       > 30 % and ≥ 3 classes are each called ≥ 1 % on x[::5]. Ten kept draws per model. Trained and null ratios on x[::2].
  5. Translation:
     - T = every real particle shifted 0.06 rad along +Δη.
     - Tr = 0.06 rad in a random direction per jet (control of equal length).
     - M2-held = Tr followed by a per-jet angle scaling that restores Σ pT r² about the stored axis (X9v).
     - Rotation control matched in displacement.
     - W→Z = fraction of true W called Z. Flow = edited − unedited, with jet bootstrap CI. Exact McNemar T vs Tr and T vs
       M2-held.
  6. Naming of S objects:
     - Up to 20 live S units by cost.
     - Probe: a single particle at the kept slot nearest rank 5 of an otherwise empty jet, context g frozen at its mean.
     - Grid: pT ∈ {1, 3, 10, 30, 100} GeV × r ∈ [0, 0.4] (41 points) × 72 azimuths.
     - Radial if the azimuthal variance share is ≤ 0.017; hinge if a ≤ 6-term (pT, r) hinge fit has R² ≥ 0.9.
- Element calls (frozen):
  - Pure cuts: column-permutation p < 0.05.
  - Windows: ≥ 1 window channel, ratio > null p90 and one-sided sign-flip p < 0.05.
  - Mass: W→Z(T) > 0.5 and W→Z(M2-held) < 0.15.
- Holm over {pure-cut p, window sign-flip p, mass McNemar T vs M2-held p} × (HGQ-150 seeds + float seeds) = 15 tests.

**Changes to the plan (after the first runs).**
- HGQ-150 arm added, as described above.
- Keras-CPU exactness check added.
- Sparsity-matched purity added. This is a weights-only check: each float model's |α_k|·sd(x_k) matrix is cut to its k
  largest entries, with k = the nonzero l1 entry count of each HGQ model.
- Pooled Fisher test on the radial-hinge object counts per arm.
- The battery's naming step now evaluates the probe grid in batches; the first official smoke job was OOM-killed at 2 GiB.

## Results (N = 25,000 jets, seed 0; 95 % intervals)

| model | test acc | l1 nonzero / live | P_nz (null median, p) | qs1 zero-bit frac; window ch. | swap ratio at top edge [CI] (edge) | null p90 (kept/tried) | W→Z: unedited / T / Tr / M2-held | radial hinge objects |
|---|---|---|---|---|---|---|---|---|
| official | 80.92 | 20 / 13 | 1.00 (1.00, 0.50) | 0.842; 12 | 5.28 [5.11, 5.48] (13\|14 vs 11\|12) | 2.85 (10/17) | 0.028 / 0.761 / 0.619 / 0.041 | 11/16 |
| HGQ-150 s0 | 77.04 | 20 / 13 | 1.00 (1.00, 0.64) | 0.914; 5 | 1.08 [0.99, 1.19] (24\|25 vs 6\|7) | 0.49 (10/20) | 0.147 / 0.689 / 0.732 / 0.147 | 4/8 |
| HGQ-150 s1 * | 57.03 | 19 / 12 | 1.00 (1.00, 0.65) | 0.940; 6 | 0.00 (39\|40 vs 37\|38) | 1.42 (10/11) | 0.492 / 0.151 / 0.179 / 0.560 | 5/5 |
| HGQ-150 s2 | 78.10 | 20 / 13 | 1.00 (1.00, 0.62) | 0.939; 10 | 0.59 [0.55, 0.64] (30\|31 vs 9\|10) | 2.27 (10/32) | 0.064 / 0.714 / 0.736 / 0.080 | 7/7 |
| HGQ-50 s0 | 78.42 | 17 / 10 | 1.00 (1.00, 0.66) | 0.842; 0 | 1.36 [1.32, 1.40] (43\|44 vs 36\|37) | 0.99 (10/10) | 0.079 / 0.666 / 0.674 / 0.127 | 6/11 |
| HGQ-50 s1 | 78.69 | 13 / 8 | 1.00 (1.00, 0.70) | 0.894; 3 | 1.31 [1.26, 1.36] (42\|43 vs 4\|5) | 3.04 (10/11) | 0.058 / 0.605 / 0.617 / 0.187 | 5/8 |
| HGQ-50 s2 | 80.21 | 19 / 10 | 1.00 (1.00, 0.66) | 0.861; 4 | 1.48 [1.35, 1.62] (43\|44 vs 22\|23) | 3.61 (10/10) | 0.040 / 0.780 / 0.640 / 0.113 | 2/9 |
| float s0 | 82.46 | 192 / 63 | 0.58 (0.47, 0.013) | none; 0 | 0.34 [0.33, 0.35] (13\|14 vs 5\|6) | – | 0.022 / 0.212 / 0.238 / 0.034 | 3/20 |
| float s1 | 82.38 | 192 / 64 | 0.53 (0.44, 0.013) | none; 0 | 0.49 [0.47, 0.50] (13\|14 vs 5\|6) | – | 0.021 / 0.346 / 0.315 / 0.036 | 4/20 |

\* degenerate network (57 % accuracy, 49 % of unedited W called Z); its translation numbers do not measure a W→Z response.

Further numbers:
- Translation flows (edited − unedited W→Z, jet bootstrap CI):
  - official T 0.733 [0.720, 0.746], Tr 0.591 [0.576, 0.605], M2-held 0.013 [0.008, 0.018].
  - HGQ-150 s0 / s2: T 0.542 [0.525, 0.559] / 0.651 [0.635, 0.666]; M2-held 0.000 [−0.009, 0.011] / 0.016 [0.008, 0.024].
  - HGQ-50 s0–s2: T 0.587 / 0.547 / 0.740; M2-held 0.048 / 0.129 / 0.073.
  - float s0 / s1: T 0.189 [0.177, 0.201] / 0.325 [0.311, 0.338]; Tr 0.216 / 0.294; M2-held 0.012 / 0.015.
  - Rotation flow: |flow| ≤ 0.01 in every model.
  - McNemar T vs M2-held (paired Z calls on true W): p < 1e-150 in every usable model.
- Directional (+Δη) excess, T vs Tr:
  - official: 954 vs 240 discordant, p 5e-101.
  - HGQ-50 s2: 895 vs 197, p 1e-106.
  - HGQ-150 s0 / s2: Tr > T (253 vs 472, p 3e-16; 301 vs 411, p 4e-5).
  - HGQ-50 s0 / s1: no difference (p 0.21, 0.11).
  - float: s0 Tr > T (476 vs 608), s1 T > Tr (737 vs 583).
- Pure cuts:
  - In every quantized network all live l1 units are pure (4–5 pT ramps + 3–9 angular half-planes or angle offsets), and
    the column-permutation null is also pure at that sparsity (median 1.00; p 0.50–0.70).
  - Sparsity-matched null: each float model's l1 cut to its k = 13–20 largest entries gives P = 1.00 for every k.
  - Float networks: 0.53–0.58 pure vs null median 0.44–0.47 (p 0.013 each; Holm 0.14).
- Masks and windows:
  - All quantized networks have zero-bit slot masks at qs1 (zero-bit fraction 0.84–0.94; official 0.842) and at qs
    (0.82–0.89; official 0.82).
  - The official network's top edge is 13|14, held by almost all jets (N = 24,874). In the retrained networks, the
    largest-cost mask edges sit at ranks 24–45 (N = 13,000–22,700 jets).
  - At those edges the across/inside ratio is 0.0–1.48. The official ratio is 5.28. The official network's second edge
    (0|1) gives 0.27.
  - Mask-keeping null: 10 kept draws per model (10–32 tried; kept null accuracies 0.30–0.75), so this null has
    non-constant output as required (X8w).
  - Only HGQ-150 s0 (1.08 vs null p90 0.49, sign-flip p 0.11, Holm 0.51) and HGQ-50 s0 (1.36 vs 0.99, but 0 window
    channels) exceed the null p90. Neither meets the element rule.
- Radial (pT, r)-hinge S objects, pooled:
  - HGQ-150: 16/20 vs float 7/40, Fisher p 7.8e-6. Float restricted to its top 7 units by cost: 3/14, p 0.0013.
  - HGQ-50: 13/28 vs 7/40 (p 0.015; top-7 p 0.18).
  - Official: 11/16.
- Holm (15 tests): mass p_Holm < 1e-150 in HGQ-150 s0, s2 and both float seeds (in the float arm, the T response is
  removed by holding M2, but its size 0.21–0.35 is below the 0.5 element threshold). All pure-cut and window p_Holm ≥ 0.14.
- Element calls per model (pure cuts / windows / mass):
  - official: no / yes / yes.
  - HGQ-150: s0 no / no / yes; s1 no / no / no (degenerate); s2 no / no / yes.
  - HGQ-50: s0 no / no / yes; s1 no / no / no (M2-held 0.187 > 0.15); s2 no / no / yes.
  - float: s0 yes / no / no; s1 yes / no / no.

## Reading (scoped to these measurements)
- Thesis prediction "HGQ seeds reproduce ≥ 2 of 3 element types": not met. No retrained seed meets more than one element
  (mass). The official network meets two (windows, mass).
- Alternative "elements are idiosyncratic to the official checkpoint": for the rank-window swap effect, the retrained copies
  agree with it. Quantized retraining reproduces the zero-bit masks but not a strong across-edge response at a commonly
  occupied rank. The retrained copies differ from the official one in training length (200–300 vs 2,868 epochs) and accuracy
  (77–80 % vs 80.9 %), so a long-training origin of the official window effect is not excluded.
- Pure cuts:
  - Quantized training produces the 0.90–0.93 l1 zero fraction, and at that sparsity every unit is pure in all seven
    quantized networks.
  - Both nulls (column permutation; float weights cut to the same entry count) are also fully pure. The purity of a 13–20-entry
    first layer therefore carries no information beyond its sparsity.
  - Without quantization the first layer is dense and 53–58 % pure, slightly above its permutation null (p 0.013,
    Holm 0.14).
  - For A1q this reads as: sparsity comes from HGQ training; purity follows from sparsity.
- Mass response:
  - The un-recentred M2 response (T removed by holding Σ pT r²) is present in every usable network of both arms.
  - Its size is 0.60–0.78 W→Z with quantization vs 0.21–0.35 without. The thesis prediction "float keeps the mass response"
    holds for the M2-held contrast, but not for the preregistered size threshold (> 0.5).
  - The +Δη-specific excess over a random-direction translation (official 0.76 vs 0.62) recurs only in HGQ-50 s2.
- Radial hinge objects: more frequent with quantization (HGQ-150 16/20 vs float 7/40, p 7.8e-6). This is a pooled
  comparison over units of different cost ranks; the top-7 matched comparison gives p 0.0013.

## Issues and limits
- The retrained quantized networks do not reach the official accuracy (best 80.21 %). HGQ-150 seed 1 collapsed at its last
  epoch (57 %). No checkpoint selection was done (frozen: last epoch). A checkpoint-selected or longer (≥ 1,000-epoch,
  official beta schedule) rerun is the open follow-up for the window question.
- The window-edge selection rule (largest summed S cost) picks rank edges 24–45 in the retrained networks. A swap test at
  the official 13|14 edge on the retrained networks was not run, because it is outside the frozen plan.
- Element thresholds (0.5, 0.15) are preregistered cut-offs. The float arm's M2-held response is significant but below the
  size threshold.
- hgq_s0's rerun with the Keras check was OOM-killed at 2 GiB (2,000-jet Keras call). The original hgq_s0 result (no Keras
  check) is used; the check passed exactly on the other seven retrained models.
- The source was later changed to run the Keras check on 500 jets, but no model was rerun with that version. The
  hgq-r150_s0 job ran the 2,000-jet check while its log records `git 9c8260b`. The cluster mirror is synced
  uncommitted, so the logged hash does not identify the exact code the job ran.
- The official battery peaks at 2,047 MiB of a 2 GiB request (page cache included); the jobs completed.
- Training job sizing (for train/STATUS.md):
  - `train-x12` (5 runs, 4 CPU / 20 Gi): 0.81 cores average, 12.3 GiB peak, GPU 100 %, 40 min. The first attempt at
    14 Gi lost all five runs, consistent with OOM (no traceback); the data loader was changed to block-wise reads.
  - `train-x12-r150` (3 runs, 3 CPU / 10 Gi): 0.44 cores average (below the 20 % CPU floor), 54 min.
  - Imports of the shared env take 4–10 min on some nodes.

## Explainer entry (for site/explain.py, Paper A tab)
**Question.** Are the parts of the JEDI-linear "program" a property of this architecture plus its training, or of the one
official checkpoint? Those parts are the sparse pT-only and angle-only first-layer cuts, the rank-window masks, and the
un-recentred mass response. Which of them require quantization-aware training?

**Background.**
- JEDI-linear n64 is trained with HGQ (high-granularity quantization). HGQ learns a bit width for every weight and every
  per-particle activation, and a penalty $\beta\cdot\mathrm{EBOPs}$ pushes bits to zero.
- A weight with zero bits is a zero weight. An activation slot with zero bits is a mask: in the pooling $S_c=\sum_i u_{ic}$,
  particles at masked pT ranks are dropped, which makes a rank window.
- The official network has 20 nonzero first-layer weights out of 192, 12 window channels in the second pooling, and a W→Z
  response to a rigid shift that follows $\sum_i p_{T,i} r_i^2$ about the stored axis.

**Method.**
- Retrain the same architecture three times with HGQ and twice with quantizers replaced by the identity (same optimizer,
  batch-norm folding, schedule, initialisation).
- Run the same battery on each:
  - purity of live first-layer units $a_c=\alpha_{pT}p_T+\alpha_\eta\Delta\eta+\alpha_\phi\Delta\phi+\beta$ (pure if
    $\ge 90\%$ of the spread comes from $p_T$ alone or the angles alone), against a column-permutation null;
  - zero-bit masks;
  - an angle swap of two adjacent pT ranks across a mask edge vs inside a window, against weight-permuted copies that keep
    the masks and still classify above chance;
  - a $0.06$ rad translation of all particles, compared with a random-direction translation of equal length and with a
    translation followed by a rescaling that holds $M_2=\sum_i p_{T,i}r_i^2$ fixed;
  - single-particle maps of the pooled objects.

**Result.** Figure `results/jedi/x12_final.png`.
- (a) Every quantized network has only pure cuts, but so do its sparsity nulls.
- (b) Only the official network shows a strong across-edge response (5.3× vs null p90 2.8). The retrained networks give
  0–1.5× at their largest mask edges.
- (c) The translation turns 60–78 % of true W into Z calls in the usable quantized copies and 21–35 % in the float copies.
  Holding $M_2$ removes it in all of them.

**Checks.**
- The battery forward equals `Jedi.run` (official) and the Keras model on CPU (retrained) exactly.
- The retrained architecture matches the official parameter and quantizer-table shapes.
- The mask-keeping null has non-constant output (10 draws per model, accuracy 0.30–0.75).
- Holm over 15 element tests.

**Why it matters.**
- The mass response is a property of the architecture and training, not of one checkpoint.
- The rank-window effect of the official network was not reproduced by shorter quantized trainings.
- The "pure cuts" statement reduces to "HGQ makes the first layer sparse".
