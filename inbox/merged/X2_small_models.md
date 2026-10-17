# X2 — JEDI-linear's main tests on the small taggers trained on the same inputs (agent report, 2026-10-17)

## Procedure

**Models** (N = 64 particles, `best.pt`, seeds 0 and 1; `/j-jepa-vol/aaron-interp/models/small/<run>/seed<k>/`):
- `dsets-d64-n64`: Deep Sets. phi = Linear(3→64) ReLU Linear(64→64) ReLU per particle; masked sum S = Σ_i phi2_i; rho 64→64→32→16 ReLU (h1, h2, h3); Linear(16→5).
- `linf-2b-d32-k8-n64`: Linformer. 2 pre-LN blocks, width 32, 2 heads, K/V projected from 64 slots to 8 by one learned 64×8 matrix.
- `tf-2b-d32-n64`: plain pre-LN transformer. 2 blocks, width 32, 2 heads; 30 epochs, patience 5.
- `tf-2b-d32-n64-e100`: the same transformer trained for 100 epochs.

Untrained control: the same architecture at initialisation (`ckpt/epoch_0.pt` of each seed). The e100 runs share their initialisation with `tf-2b-d32` (same seed, same architecture). Their untrained unit scan was therefore not repeated.

**Inputs.** As in training (`train/train_small.py`):
- (pT, Δη, Δφ) of the first 64 particles, (x − shift)/scale with arch.json's shift/scale (= `norm_stats.json['64']`);
- padded rows (raw pT = 0) set to 0, mask = raw pT > 0.

**Sample.** `interp.jedi_core.sample(per_class=4000, seed=0)`: 20,000 test jets (the val archive, = the training code's test set), the same jets as E10. Read-outs use E10's split: rng(0) permutation, first 10,000 train, last 10,000 test.

**Layers.**
- Deep Sets: per particle phi1, phi2; pooled S, h1, h2, h3.
- Transformers: per token emb (Linear 3→32), b⟨k⟩.attn (attention output after its projection), b⟨k⟩.mlp (GELU hidden, 64), b⟨k⟩.res (residual stream after block k); pooled: pool (masked mean of the final LayerNorm output, 32) → logits.
- The explicit forward used for all edits reproduces the module's logits exactly (max |Δlogit| 0 on 500 jets, every run).

**Tests.**
1. Accuracy on the sample vs `test.json`. The expectation for the balanced sample is the mean of test.json's per-class accuracies; s.e. is binomial with the finite-population factor. Per-jet comparison with `test_probs.npy` at the same rows.
2. Units. Each unit is replaced by its mean (over real particles / tokens; over jets for pooled layers) and the % of the 20,000 calls that change is recorded.
   - Dead: pre-activation never > 0 (ReLU, GELU layers).
   - Live: cost ≥ 1 % and ≥ 0.1 %. Brackets give the count of units whose 95 % Wilson interval lies entirely above the threshold, and the count whose upper bound reaches it.
   - Every unit of a layer at its mean at once: accuracy.
   - Untrained control: the same scan on epoch 0, seed 0.
3. Deep Sets first layer. Each unit is written in physical units: ReLU(a_pT·pT + a_η·Δη + a_φ·Δφ + b), with a = W1/scale and b = b1 − Σ a·shift.
   - Identity checked against the network's phi1 summed over real particles: Σ_i ReLU(·) = b·|A| + a_pT Σ_A pT + a_η Σ_A Δη + a_φ Σ_A Δφ.
   - Type of each unit, by its share of pre-activation variance over real particles: pT ramp (pT term ≥ 0.9), angular (angle terms ≥ 0.9), mixed.
   - For the angle part, the signed offset of the line from the axis, c(pT) = −(b + a_pT·pT)/|a_ang|, at the median real-particle pT (7.66 GeV), and dc/dpT.
   - Note: the network sums phi2, not phi1. Σ phi1 is the E3-style object, not a network stage.
4. Mass shift.
   - Edit: every real particle moved by δ ∈ {0, .01, .02, .03, .045, .06, .09} rad along Δη and, separately, along Δφ. Padding untouched, axis fixed. The invariant mass is unchanged (max |Δlog m| ≤ 1e-7).
   - Read-out: ridge fit of log m at every pooled layer on the unedited train half, as in E6v (alpha chosen on the last quarter of the train half). The last hidden layer (h3; pool) is the headline. The mean shift is taken on the test half.
   - Hypotheses for the shift of the read-out:
     - H0 (invariant mass): 0.
     - H1 (mass about the given axis, √(M0·M2)): ½Δlog(M0·M2).
     - Alternatives: Δlog Σ pT r, and ½Δlog Σ pT² r² (E6v's G1 and √Q2).
   - Calibration. Δη, Δφ × k is a real change of m and of √(M0M2) by log k. The response of the read-out on a grid log k ∈ {−0.2 … 0.9} gives each hypothesis's expected read-out shift, and a matched edit with log k = the mean H1 prediction gives the calibrated ratio.
   - Nulls: rotation by 90° about the axis (r and M2 unchanged; Δlog m mean −4.7e-5, max |Δlog m| 0.085, because a rotation in the (Δη, Δφ) plane is not an exact symmetry of the four-vectors) and the reflection Δφ → −Δφ (m exactly unchanged). Positive control: angles × 1.1.
   - Statistics: 95 % intervals by bootstrap over jets (1,000 resamples; test half for read-outs, class subsets for W → Z / Z → W, all jets for calls changed). Exact McNemar test on the paired W → Z calls. z = (observed − expected)/bootstrap s.d. Multiple comparisons: 12 shifts × 4 hypotheses per model and seed (48 tests); every test is held to |z| ≥ 3.34 (two-sided α = 0.05/48, the strictest Holm step).
5. Variables over layers (E10 light).
   - 12 observables: Σ pT, n particles, Σ pT r², pT of slot 0 (depth 0); girth, mass, e2 (1); C2, D2 (2); τ21, τ32, soft-drop mass (3).
   - E10's probe: RidgeCV on standardised live units; target clipped to the train half's 0.5–99.5 %. Held-out R² with bootstrap 95 % CI; untrained control.
   - Deep Sets read at Σ phi1, S, h1, h2, h3. Transformers read each per-token layer through [masked mean, masked sum] over the real tokens, and pool as is.

Code: `interp/x2_small_models.py` (modes: model run, `--plot`, `--report`). Jobs: `AI_PY=conda ./job.sh x2-<…>` (1 CPU, 2 GiB). Outputs: `/j-jepa-vol/aaron-interp/results/small/x2_<run>[__s0|__s1|__u0].json`; tables of tests 1–5 from `--report` (`results/small/x2_report.md`, log `logs/x2-report3.log` on the volume). PNGs: `results/site/T8_small_mass_shift.png`, `T9_small_layers.png`. JEDI-linear numbers: E1 (`e1_e3.json`), E6v (`e6_verify.json`), E6 (`e6_mass.json`), E10 (`e10_layers.json`), E5v (rotation).

## Results

### Test 1: accuracy (20,000 test jets)

| model | seed | acc (sample) | test.json acc (260k) | balanced expectation ± s.e. | z | max abs Δp vs test_probs | argmax agreement |
|---|---|---|---|---|---|---|---|
| dsets-d64-n64 | 0 | 0.8034 | 0.8064 | 0.8059 ± 0.0027 | -0.93 | 2.9e-06 | 1.0000 |
| dsets-d64-n64 | 1 | 0.8069 | 0.8081 | 0.8075 ± 0.0026 | -0.22 | 3.6e-06 | 1.0000 |
| linf-2b-d32-k8-n64 | 0 | 0.8055 | 0.8106 | 0.8100 ± 0.0026 | -1.72 | 7.3e-06 | 1.0000 |
| linf-2b-d32-k8-n64 | 1 | 0.8022 | 0.8076 | 0.8071 ± 0.0027 | -1.85 | 7.5e-06 | 1.0000 |
| tf-2b-d32-n64 | 0 | 0.7607 | 0.7619 | 0.7612 ± 0.0029 | -0.20 | 5.4e-06 | 1.0000 |
| tf-2b-d32-n64 | 1 | 0.7355 | 0.7342 | 0.7336 ± 0.0029 | +0.63 | 4.1e-06 | 1.0000 |
| tf-2b-d32-n64-e100 | 0 | 0.7979 | 0.8008 | 0.8002 ± 0.0027 | -0.85 | 1.2e-05 | 1.0000 |
| tf-2b-d32-n64-e100 | 1 | 0.7970 | 0.7991 | 0.7985 ± 0.0027 | -0.57 | 1.7e-05 | 1.0000 |

H0 (the sample reproduces test.json): |z| ≤ 1.85 in all 8 runs. The per-jet softmax outputs agree with `test_probs.npy` to ≤ 1.7e-5, and the argmax agrees on every jet.

### Test 2: units (mean ablation; % of the 20,000 calls changed)

Live ≥ 1 % (≥ 0.1 %), seed 0 / seed 1; dead units for ReLU / GELU layers; untrained = epoch 0 of seed 0.

| model | layer (width) | dead | live ≥ 1 % | live ≥ 0.1 % | largest cost [%] | untrained live ≥ 1 % (≥ 0.1 %) |
|---|---|---|---|---|---|---|
| JEDI-linear (E1, 10k jets) | a / P / g / s / u / S / h4 / h5 / z | 49 / 50 / 49 / 55 / 45 / 45 / 9 / 0 / 0 | 13 / 11 / 7 / 9 / 19 / 19 / 42 / 30 / 14 | 13 / 12 / 11 / 9 / 19 / 19 / 54 / 32 / 16 | 64.2 / 8.1 / 10.4 / 64.9 / 61.2 / 40.8 / 19.7 / 16.1 / 21.6 | – |
| Deep Sets | phi1 (64) | 1 / 0 | 63 / 64 | 63 / 64 | 23.8 / 27.1 | 0 (0) |
| | phi2 (64) | 5 / 5 | 53 / 51 | 59 / 59 | 16.8 / 20.8 | 0 (0) |
| | S = Σ phi2 (64) | 5 / 5 | 54 / 50 | 59 / 59 | 14.9 / 18.7 | 0 (0) |
| | h1 (64) | 11 / 15 | 52 / 49 | 52 / 49 | 23.9 / 12.5 | 1 (1) |
| | h2 (32) | 6 / 4 | 25 / 28 | 26 / 28 | 43.9 / 36.0 | 0 (0) |
| | h3 (16) | 2 / 5 | 13 / 11 | 13 / 11 | 28.1 / 65.0 | 0 (1) |
| Linformer | emb (32) | – | 32 / 32 | 32 / 32 | 41.0 / 42.5 | 32 (32) |
| | b0.attn (32) | – | 32 / 32 | 32 / 32 | 16.6 / 23.5 | 27 (32) |
| | b0.mlp (64) | 0 / 0 | 62 / 61 | 64 / 64 | 36.5 / 34.1 | 33 (63) |
| | b0.res (32) | – | 32 / 32 | 32 / 32 | 29.3 / 42.0 | 32 (32) |
| | b1.attn (32) | – | 31 / 32 | 32 / 32 | 9.6 / 11.2 | 26 (32) |
| | b1.mlp (64) | 0 / 0 | 49 / 54 | 64 / 64 | 29.9 / 38.5 | 25 (63) |
| | b1.res (32) | – | 29 / 28 | 32 / 32 | 4.2 / 4.4 | 32 (32) |
| | pool (32) | – | 29 / 27 | 32 / 32 | 3.7 / 3.6 | 32 (32) |
| Transformer | emb / b0.attn / b0.mlp / b0.res | – / – / 0, 0 / – | 32/32, 32/32, 64/62, 32/32 | all | 62.0/57.5, 25.6/9.4, 29.7/31.0, 62.4/34.6 | 28 (32), 16 (31), 8 (56), 24 (32) |
| | b1.attn / b1.mlp / b1.res / pool | – / 0, 0 / – / – | 32/32, 59/50, 32/28, 32/26 | all | 19.7/11.4, 69.3/25.8, 9.9/4.7, 10.7/3.7 | 14 (30), 7 (55), 23 (32), 25 (31) |
| Transformer 100 ep | emb / b0.attn / b0.mlp / b0.res | – / – / 0, 0 / – | 32/32, 32/32, 64/64, 32/32 | all | 72.1/62.1, 40.3/41.5, 48.8/51.0, 62.5/50.6 | same initialisation as Transformer |
| | b1.attn / b1.mlp / b1.res / pool | – / 0, 0 / – / – | 32/32, 63/64, 31/27, 28/25 | all | 24.9/15.6, 73.2/71.6, 10.1/7.5, 12.9/6.9 | |

- Wilson-interval ranges for each count are in `results/small/x2_report.md`. For example, Deep Sets phi2 live ≥ 1 % is 53 [51, 56] (seed 0).
- The untrained Deep Sets calls t on all 20,000 jets (seed 0), so ablations change no calls. The untrained Linformer and transformer calls vary (accuracy 0.18 and 0.20), and the largest single-unit ablation per layer changes 2.2–13.5 % of calls.
- Setting every unit of a layer to its mean gives accuracy 0.200 at every pooled and residual-stream layer and at the transformer emb. Other layers: Deep Sets phi1 / phi2 0.20–0.22; Linformer emb 0.31; attention outputs 0.30–0.60; GELU MLP layers 0.20–0.48.

### Test 3: Deep Sets first layer as cuts

- Identity Σ_i phi1_i = b|A| + a_pT Σ_A pT + a_η Σ_A Δη + a_φ Σ_A Δφ over the real particles: max abs error 9.8e-6 (seed 0) and 1.2e-5 (seed 1); relative 1.2e-7 and 1.4e-7 (float32). Untrained: 1.3e-5.
- Kinds over all 64 units (pT ramp / angular / mixed):
  - seed 0: 4 / 21 / 39;
  - seed 1: 3 / 17 / 44;
  - untrained seed 0: 0 / 17 / 47.
  - JEDI-linear's 13 live a units for comparison: 4 pT-only (a32 pT < 11 GeV, a56 < 31, a21 < 93, a15 always on), 9 angle-only half-planes through the axis within 0.01 rad, 0 mixed.
- Live units: 63 (seed 0) and 64 (seed 1).
- Angular units whose line passes within 0.01 rad of the axis at the median particle pT (7.66 GeV): 6 of 21 (seed 0), 5 of 17 (seed 1). Within 0.03 rad: 7 and 8.
- Mixed units: 38 live (seed 0) and 44 (seed 1). Of these, 16 and 18 have a_pT < 0 (on for soft particles on one side of a line), and the rest a_pT > 0. Their line offset moves with pT by |dc/dpT| = 0.0008–0.0066 rad/GeV (median 0.0028, 0.0030), and |offset| at the median pT is > 0.03 rad for 30 of 38 and 35 of 44.
- pT ramps, seed 0: a1 (on for pT < 17.1 GeV; angle weights 0.07, 0.00 rad⁻¹; sum shares count +1.39 / ΣpT −0.40). a58, a43 and a53 are ramps with a 6–10 % angular variance share.
- The ten costliest live units (seed 0):

| unit | cost | kind | a_pT [1/GeV] | a_η | a_φ | b | pT share | pass frac | pT at axis [GeV] | normal [deg] | offset c [rad] | dc/dpT [rad/GeV] | shares cnt / pT / η / φ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a57 | 23.8 % | mixed | -0.0188 | +5.04 | +1.62 | +0.28 | 0.70 | 0.48 | 14.8 (pT below) | +18 | -0.025 | +0.0036 | +0.26 / -0.13 / +0.79 / +0.08 |
| a59 | 23.6 % | angular | -0.0003 | +5.70 | -4.93 | +0.02 | 0.00 | 0.51 | 65.6 (pT below) | -41 | -0.002 | +0.0000 | +0.01 / +0.00 / +0.57 / +0.41 |
| a17 | 21.6 % | mixed | +0.0164 | +2.46 | +1.17 | -0.13 | 0.87 | 0.62 | 7.8 (pT above) | +26 | +0.001 | -0.0060 | -0.12 / +0.37 / +0.61 / +0.14 |
| a10 | 20.9 % | mixed | -0.0160 | +4.89 | -3.71 | -0.21 | 0.56 | 0.22 | -13.0 (pT below) | -37 | +0.054 | +0.0026 | -0.23 / -0.13 / +0.87 / +0.49 |
| a33 | 20.6 % | mixed | -0.0182 | +5.50 | +3.34 | +0.52 | 0.60 | 0.62 | 28.6 (pT below) | +31 | -0.059 | +0.0028 | +0.39 / -0.10 / +0.52 / +0.19 |
| a12 | 18.7 % | mixed | -0.0185 | +4.88 | +2.23 | -0.07 | 0.69 | 0.26 | -4.0 (pT below) | +25 | +0.040 | +0.0035 | -0.09 / -0.16 / +1.03 / +0.21 |
| a50 | 17.0 % | mixed | -0.0179 | -4.65 | -1.29 | +0.07 | 0.72 | 0.34 | 3.9 (pT below) | -164 | +0.014 | +0.0037 | +0.08 / -0.15 / +1.00 / +0.08 |
| a58 | 15.9 % | pT ramp | -0.0235 | +3.30 | -0.11 | +0.03 | 0.90 | 0.26 | 1.4 (pT below) | -2 | +0.045 | +0.0071 | +0.06 / -0.24 / +1.18 / +0.00 |
| a4 | 15.5 % | mixed | -0.0205 | -3.30 | -2.35 | -0.00 | 0.83 | 0.26 | -0.1 (pT below) | -145 | +0.039 | +0.0051 | -0.00 / -0.20 / +0.80 / +0.40 |
| a23 | 15.2 % | mixed | +0.0138 | -4.02 | -2.30 | -0.18 | 0.62 | 0.55 | 13.1 (pT above) | -150 | +0.016 | -0.0030 | -0.16 / +0.05 / +0.83 / +0.27 |
| a8 | 14.6 % | angular | +0.0002 | -0.06 | +5.93 | -0.01 | 0.00 | 0.50 | 59.4 (pT above) | +91 | +0.002 | -0.0000 | -0.01 / -0.00 / -0.00 / +1.01 |
| a37 | 14.6 % | mixed | +0.0068 | -4.07 | +5.53 | -0.27 | 0.16 | 0.42 | 39.9 (pT above) | +126 | +0.032 | -0.0010 | -0.20 / -0.03 / +0.43 / +0.81 |

### Test 4: mass shift

Hypotheses for the shift of the log m read-out:
- H0 (invariant mass): 0.
- H1 (√(M0·M2), mass about the given axis): ½Δlog(M0·M2) = 0.016, 0.055, 0.109, 0.201, 0.299, 0.492 at δ = .01 … .09.
- Alternatives: Σ pT r (0.064 … 0.749) and √Σ pT² r² (0.092 … 0.899).

The invariant mass changes by ≤ 1e-7 under every shift. The table below holds the T8 contents. Ratios are Δlog m̂ ÷ ½Δlog(M0M2). The calibrated ratio divides the shift response by the response to the matched angle scale (Δη, Δφ × k, log k = mean prediction).

| run (seed), layer | R² log m [CI] | W → Z % δ = 0/.03/.06/.09 (η) | Z → W % δ = 0/.06 | calls changed % η .03/.06 | Δlog m̂ η .06 [CI] | raw ratio η .01/.02/.03/.045/.06/.09 | calibrated ratio η .03/.06/.09 [CI at .06] | φ .06 raw / calibrated | rot 90°: Δlog m̂ [CI]; calls % | × 1.1: Δlog m̂ ÷ log 1.1; W → Z % | exponent η (δ ≤ .06) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsets-d64-n64 (0), h3 | 0.953 [0.950, 0.956] | 2/42/69/4 | 5/4 | 16.6/61.5 | 0.219 [0.215, 0.223] | -0.10/0.26/0.41/0.58/0.73/0.82 | 0.45/0.86/1.03 [0.85, 0.88] | 0.93 / 1.10 | +0.0038 [+0.0028, +0.0049]; 4.9 | 0.90; 76.8 | 2.46 |
| dsets-d64-n64 (1), h3 | 0.921 [0.918, 0.924] | 2/44/74/4 | 5/4 | 17.9/61.9 | 0.188 [0.184, 0.192] | 0.19/0.33/0.39/0.48/0.63/0.71 | 0.39/0.70/0.92 [0.69, 0.72] | 0.48 / 0.54 | +0.0026 [+0.0015, +0.0037]; 4.4 | 1.01; 76.8 | 2.28 |
| linf-2b-d32-k8-n64 (0), pool | 0.910 [0.905, 0.913] | 3/60/21/3 | 6/2 | 21.3/54.4 | 0.239 [0.233, 0.245] | 1.33/1.05/0.96/0.86/0.80/0.69 | 1.82/2.03/1.72 [1.97, 2.10] | 1.00 / 2.54 | +0.0014 [+0.0001, +0.0028]; 2.4 | 0.55; 76.0 | 1.37 |
| linf-2b-d32-k8-n64 (1), pool | 0.915 [0.911, 0.919] | 3/50/60/6 | 6/5 | 18.1/59.0 | 0.262 [0.255, 0.269] | 0.45/0.62/0.72/0.83/0.88/0.81 | 1.00/1.62/1.59 [1.58, 1.66] | 0.84 / 1.55 | -0.0003 [-0.0019, +0.0013]; 2.5 | 0.74; 75.3 | 2.03 |
| tf-2b-d32-n64 (0), pool | 0.911 [0.907, 0.915] | 3/36/66/21 | 12/5 | 18.6/52.4 | 0.091 [0.088, 0.095] | 0.05/0.12/0.17/0.24/0.31/0.42 | 0.20/0.41/0.65 [0.39, 0.42] | 0.53 / 0.71 | +0.0003 [-0.0012, +0.0019]; 5.7 | 0.86; 68.4 | 2.6 |
| tf-2b-d32-n64 (1), pool | 0.900 [0.895, 0.905] | 5/31/60/48 | 14/14 | 18.6/48.5 | 0.132 [0.128, 0.137] | 0.31/0.29/0.31/0.37/0.44/0.56 | 0.40/0.59/0.82 [0.57, 0.61] | -0.15 / -0.20 | -0.0027 [-0.0047, -0.0008]; 6.5 | 0.79; 64.9 | 1.85 |
| tf-2b-d32-n64-e100 (0), pool | 0.921 [0.917, 0.925] | 2/64/57/5 | 6/5 | 25.7/65.6 | 0.225 [0.220, 0.230] | -0.52/0.01/0.25/0.52/0.75/0.92 | 0.29/1.01/1.35 [0.99, 1.03] | 1.33 / 1.79 | +0.0007 [-0.0007, +0.0021]; 3.7 | 0.87; 74.6 | 5.54 |
| tf-2b-d32-n64-e100 (1), pool | 0.863 [0.858, 0.869] | 2/72/27/7 | 7/4 | 30.8/69.0 | 0.012 [0.007, 0.018] | -0.43/-0.13/-0.06/-0.01/0.04/0.09 | -0.08/0.07/0.18 [0.04, 0.10] | 0.81 / 1.41 | -0.0012 [-0.0031, +0.0007]; 3.1 | 0.81; 74.2 | None |

JEDI-linear (E6v, 10,000 jets, read-out at z; E6 for × 1.1):
- R² 0.942. W → Z 3 / 28 / 77 / 4 %; Z → W 6 / 3 %.
- Δlog m̂ at η .06: 0.251 ± 0.007.
- Raw ratio 0.25 / 0.49 / 0.61 / 0.75 / 0.84 / 0.86; φ .06 0.85.
- Exponent at z 2.31 (S 2.14, h4 2.17, h5 1.94).
- × 1.1: 0.89; W → Z 71 %.
- Rotation 90°: 4.1 % of calls changed (E5v, fixed point).
- Calibrated ratio, Z → W and calls changed over all jets: not recorded (n/r).

Calibrated ratio at η + 0.06 for every read-out layer:

| run (seed) | layer: ratio [CI] |
|---|---|
| dsets-d64-n64 (0) | Σ phi1 0.45 [0.43, 0.46]; S 1.11 [1.08, 1.13]; h1 0.94 [0.93, 0.96]; h2 0.86 [0.85, 0.88]; h3 0.86 [0.85, 0.88] |
| dsets-d64-n64 (1) | Σ phi1 0.78 [0.76, 0.80]; S 0.66 [0.64, 0.69]; h1 0.89 [0.87, 0.90]; h2 0.88 [0.87, 0.90]; h3 0.70 [0.69, 0.72] |
| linf-2b-d32-k8-n64 (0) | b0 MLP 0.82 [0.81, 0.84]; b0 0.86 [0.84, 0.87]; b1 MLP 0.99 [0.97, 1.01]; b1 1.27 [1.24, 1.30]; pool 2.03 [1.97, 2.10] |
| linf-2b-d32-k8-n64 (1) | b0 MLP 0.67 [0.66, 0.69]; b0 0.74 [0.72, 0.75]; b1 MLP 1.09 [1.06, 1.11]; b1 1.51 [1.48, 1.55]; pool 1.62 [1.58, 1.66] |
| tf-2b-d32-n64 (0) | b0 MLP 1.61 [1.59, 1.64]; b0 1.42 [1.40, 1.44]; b1 MLP 1.01 [0.99, 1.03]; b1 0.59 [0.57, 0.61]; pool 0.41 [0.39, 0.42] |
| tf-2b-d32-n64 (1) | b0 MLP 0.93 [0.91, 0.95]; b0 0.69 [0.68, 0.71]; b1 MLP 0.82 [0.80, 0.84]; b1 1.00 [0.98, 1.02]; pool 0.59 [0.57, 0.61] |
| tf-2b-d32-n64-e100 (0) | b0 MLP 1.85 [1.82, 1.87]; b0 1.70 [1.67, 1.72]; b1 MLP 1.30 [1.28, 1.33]; b1 1.01 [0.98, 1.04]; pool 1.01 [0.99, 1.03] |
| tf-2b-d32-n64-e100 (1) | b0 MLP 0.92 [0.91, 0.94]; b0 0.29 [0.28, 0.30]; b1 MLP 1.23 [1.20, 1.25]; b1 0.99 [0.97, 1.02]; pool 0.07 [0.04, 0.10] |

The emb layer is omitted: its read-out does not respond to the angle scale (Δlog m̂ ≤ 1e-4), so the ratio is undefined.

Hypothesis tests at the read-out layer (12 shifts × 4 hypotheses = 48 per run; Holm threshold |z| ≥ 3.34). Tests that are not rejected:
- Deep Sets s0, s1: none. The response lies between 0 and the calibrated √(M0M2) response, with these exceptions:
  - above it: η .09 and φ .06, .09 (seed 0; z +7 to +34);
  - below 0: η .01 (seed 0) and φ .01, .02 (seed 1).
- Linformer s0: η .03 Σ pT r (z +1.2); φ .045 √Σ pT² r² (+2.7).
- Linformer s1: η .03 √(M0M2) (+0.7); φ .03 √(M0M2) (+2.8); η .06 Σ pT r (−0.0).
- Transformer s0: η .01 invariant (+3.0).
- Transformer s1: φ .09 invariant (−1.6). Its φ response is negative at every δ (−0.022, −0.039, −0.051, −0.056, −0.045, −0.006 for δ = .01 … .09).
- Transformer 100 ep s0: η .02 invariant (+0.7); η .06 √(M0M2) (+2.0); η .09 Σ pT r (+2.7).
- Transformer 100 ep s1: η .045 invariant (−1.2); φ .06 Σ pT r (+3.1) and √Σ pT² r² (+0.7). Its η read-out moves ≤ 0.042 up to δ = .09 while W → Z reaches 72 % at δ = .03 and calls change 31–69 %.

Nulls at the read-out layer:
- Rotation 90°: |Δlog m̂| ≤ 0.004 in every run. The CI excludes 0 for Deep Sets s0 / s1 (+0.004, +0.003), Linformer s0 (+0.001) and Transformer s1 (−0.003). Calls changed 2.4–6.5 %, against 17–31 % at η .03 and 49–69 % at η .06. W → Z under rotation: McNemar p ≥ 0.25 except Deep Sets (3.9 vs 2.4 %, p 5e-8).
- Reflection Δφ → −Δφ: |Δlog m̂| ≤ 0.002, calls changed 2.1–5.7 %.

Positive control (angles × 1.1):
- Read-out response ÷ log 1.1 = 0.55–1.01.
- W → Z 65–77 %, against 71 % for JEDI.

### Test 5: variables over layers (T9 contents; held-out linear R², mean of seeds)

- Bootstrap 95 % half-widths ≤ 0.042 for every small-model cell.
- Per-seed values with intervals are in `results/small/x2_report.md`. The largest seed difference is 0.12 (Transformer C2 at pool: 0.70 / 0.58).

JEDI-linear (E10)

| observable | depth | P | g | S | h4 | h5 | z |
|---|---|---|---|---|---|---|---|
| Σ pT | 0 | 0.98 | 0.88 | 0.96 | 0.96 | 0.87 | 0.77 |
| n particles | 0 | 0.89 | 0.88 | 0.99 | 0.98 | 0.94 | 0.83 |
| Σ pT r² | 0 | 0.74 | 0.78 | 0.99 | 0.99 | 0.98 | 0.97 |
| pT of slot 0 | 0 | 0.65 | 0.59 | 0.90 | 0.82 | 0.57 | 0.45 |
| girth | 1 | 0.66 | 0.68 | 0.99 | 0.99 | 0.97 | 0.95 |
| mass | 1 | 0.74 | 0.76 | 0.97 | 0.99 | 0.99 | 0.97 |
| e2 | 1 | 0.49 | 0.47 | 0.95 | 0.97 | 0.96 | 0.94 |
| C2 | 2 | 0.48 | 0.44 | 0.71 | 0.80 | 0.76 | 0.64 |
| D2 | 2 | 0.21 | 0.15 | 0.63 | 0.82 | 0.79 | 0.68 |
| τ21 | 3 | 0.09 | 0.16 | 0.68 | 0.75 | 0.74 | 0.65 |
| τ32 | 3 | 0.24 | 0.23 | 0.36 | 0.44 | 0.44 | 0.39 |
| soft-drop mass | 3 | 0.38 | 0.40 | 0.88 | 0.91 | 0.88 | 0.84 |
| R² ≥ 0.9 (of 12) | | 1 | 0 | 7 | 7 | 5 | 4 |
| untrained: median R² | | 0.34 | 0.25 | 0.00 | 0.00 | 0.05 | 0.08 |
| untrained: R² ≥ 0.9 | | 0 | 0 | 0 | 0 | 0 | 0 |

dsets-d64-n64

| observable | depth | Σ phi1 | S | h1 | h2 | h3 |
|---|---|---|---|---|---|---|
| Σ pT | 0 | 0.98 | 0.98 | 0.97 | 0.96 | 0.88 |
| n particles | 0 | 1.00 | 1.00 | 1.00 | 0.97 | 0.90 |
| Σ pT r² | 0 | 0.97 | 0.99 | 0.99 | 0.99 | 0.98 |
| pT of slot 0 | 0 | 0.93 | 0.94 | 0.90 | 0.70 | 0.51 |
| girth | 1 | 0.95 | 0.98 | 0.98 | 0.98 | 0.96 |
| mass | 1 | 0.94 | 0.96 | 0.99 | 0.99 | 0.98 |
| e2 | 1 | 0.86 | 0.93 | 0.95 | 0.95 | 0.92 |
| C2 | 2 | 0.60 | 0.68 | 0.74 | 0.69 | 0.55 |
| D2 | 2 | 0.41 | 0.55 | 0.68 | 0.61 | 0.52 |
| τ21 | 3 | 0.46 | 0.58 | 0.70 | 0.67 | 0.62 |
| τ32 | 3 | 0.31 | 0.34 | 0.40 | 0.41 | 0.36 |
| soft-drop mass | 3 | 0.80 | 0.86 | 0.87 | 0.86 | 0.83 |
| R² ≥ 0.9 (of 12) | | 6 | 7 | 6 | 6 | 5 |
| untrained: median R² | | 0.85 | 0.86 | 0.77 | 0.59 | 0.39 |
| untrained: R² ≥ 0.9 | | 6 | 6 | 1 | 1 | 0 |

linf-2b-d32-k8-n64

| observable | depth | emb | b0 MLP | b0 | b1 MLP | b1 | pool |
|---|---|---|---|---|---|---|---|
| Σ pT | 0 | 0.98 | 0.95 | 0.84 | 0.94 | 0.90 | 0.82 |
| n particles | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.92 |
| Σ pT r² | 0 | 0.30 | 0.96 | 0.95 | 0.98 | 0.97 | 0.93 |
| pT of slot 0 | 0 | 0.36 | 0.95 | 0.86 | 0.87 | 0.86 | 0.79 |
| girth | 1 | 0.30 | 0.97 | 0.94 | 0.98 | 0.97 | 0.94 |
| mass | 1 | 0.32 | 0.96 | 0.94 | 0.98 | 0.97 | 0.93 |
| e2 | 1 | 0.13 | 0.93 | 0.88 | 0.96 | 0.96 | 0.92 |
| C2 | 2 | 0.24 | 0.70 | 0.66 | 0.78 | 0.72 | 0.65 |
| D2 | 2 | 0.19 | 0.70 | 0.66 | 0.82 | 0.77 | 0.68 |
| τ21 | 3 | 0.07 | 0.62 | 0.59 | 0.73 | 0.68 | 0.62 |
| τ32 | 3 | 0.12 | 0.35 | 0.34 | 0.44 | 0.41 | 0.38 |
| soft-drop mass | 3 | 0.09 | 0.82 | 0.78 | 0.89 | 0.87 | 0.85 |
| R² ≥ 0.9 (of 12) | | 2 | 7 | 4 | 6 | 5 | 5 |
| untrained: median R² | | 0.27 | 0.71 | 0.81 | 0.69 | 0.72 | 0.51 |
| untrained: R² ≥ 0.9 | | 2 | 4 | 2 | 1 | 2 | 1 |

tf-2b-d32-n64

| observable | depth | emb | b0 MLP | b0 | b1 MLP | b1 | pool |
|---|---|---|---|---|---|---|---|
| Σ pT | 0 | 0.98 | 0.89 | 0.91 | 0.75 | 0.63 | 0.21 |
| n particles | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.87 |
| Σ pT r² | 0 | 0.30 | 0.95 | 0.93 | 0.98 | 0.96 | 0.93 |
| pT of slot 0 | 0 | 0.36 | 0.92 | 0.93 | 0.88 | 0.90 | 0.84 |
| girth | 1 | 0.30 | 0.96 | 0.94 | 0.98 | 0.98 | 0.96 |
| mass | 1 | 0.32 | 0.95 | 0.93 | 0.98 | 0.96 | 0.92 |
| e2 | 1 | 0.13 | 0.92 | 0.89 | 0.97 | 0.95 | 0.93 |
| C2 | 2 | 0.24 | 0.71 | 0.66 | 0.80 | 0.76 | 0.64 |
| D2 | 2 | 0.19 | 0.71 | 0.67 | 0.82 | 0.77 | 0.66 |
| τ21 | 3 | 0.07 | 0.64 | 0.59 | 0.75 | 0.72 | 0.66 |
| τ32 | 3 | 0.12 | 0.38 | 0.36 | 0.45 | 0.44 | 0.38 |
| soft-drop mass | 3 | 0.09 | 0.83 | 0.80 | 0.89 | 0.86 | 0.84 |
| R² ≥ 0.9 (of 12) | | 2 | 6 | 6 | 5 | 5 | 4 |
| untrained: median R² | | 0.27 | 0.75 | 0.86 | 0.78 | 0.83 | 0.60 |
| untrained: R² ≥ 0.9 | | 2 | 4 | 4 | 4 | 3 | 0 |

tf-2b-d32-n64-e100

| observable | depth | emb | b0 MLP | b0 | b1 MLP | b1 | pool |
|---|---|---|---|---|---|---|---|
| Σ pT | 0 | 0.98 | 0.93 | 0.92 | 0.83 | 0.74 | 0.50 |
| n particles | 0 | 1.00 | 1.00 | 1.00 | 0.99 | 0.98 | 0.90 |
| Σ pT r² | 0 | 0.30 | 0.96 | 0.95 | 0.99 | 0.97 | 0.93 |
| pT of slot 0 | 0 | 0.36 | 0.97 | 0.96 | 0.94 | 0.91 | 0.87 |
| girth | 1 | 0.30 | 0.97 | 0.96 | 0.99 | 0.97 | 0.95 |
| mass | 1 | 0.32 | 0.96 | 0.94 | 0.99 | 0.95 | 0.91 |
| e2 | 1 | 0.13 | 0.93 | 0.91 | 0.97 | 0.95 | 0.91 |
| C2 | 2 | 0.24 | 0.74 | 0.73 | 0.84 | 0.79 | 0.66 |
| D2 | 2 | 0.19 | 0.72 | 0.69 | 0.83 | 0.77 | 0.63 |
| τ21 | 3 | 0.07 | 0.64 | 0.60 | 0.77 | 0.73 | 0.63 |
| τ32 | 3 | 0.12 | 0.39 | 0.37 | 0.50 | 0.47 | 0.42 |
| soft-drop mass | 3 | 0.09 | 0.84 | 0.81 | 0.89 | 0.86 | 0.83 |
| R² ≥ 0.9 (of 12) | | 2 | 7 | 7 | 6 | 6 | 4 |
| untrained: median R² | | 0.27 | 0.75 | 0.86 | 0.78 | 0.83 | 0.60 |
| untrained: R² ≥ 0.9 | | 2 | 4 | 4 | 4 | 3 | 0 |

Spearman ρ (depth vs first layer with R² ≥ 0.9; never = number of layers), same 12 observables:
- JEDI-linear 0.85 (p 5e-4);
- Deep Sets 0.90 / 0.90 (p 6e-5);
- Linformer 0.90 / 0.90 (p 6e-5);
- Transformer 0.87 / 0.85 (p 3e-4, 5e-4);
- Transformer 100 ep 0.90 / 0.90.
- At R² ≥ 0.7: JEDI 0.78; the small models 0.70–0.78.

Untrained controls:
- JEDI-linear: R² ≥ 0.9 for 0 of 12 observables at every stage.
- Deep Sets: 6 of 12 at Σ phi1 and S (the depth-0/1 sums; median R² 0.85), falling to 0 at h3.
- Linformer and Transformer: 0–4 of 12 at b0 MLP … pool (median 0.51–0.86); 2 of 12 at emb.

## Notes on the runs

- Jobs (`AI_PY=conda`, 1 CPU, 2 GiB): x2-dsets-d64 (14 min); x2-linf-s0/s1/u0 and x2-tf-s0/s1/u0 (per seed, untrained units separately; 13–56 min); x2-tfe100-s0/s1 (39–43 min).
  - Peak pod memory 1.6–2.0 GiB, the cgroup peak including page cache; Deep Sets and Linformer reached 2047 MiB with exit 0.
  - The first full Linformer / transformer jobs were stopped and replaced by the per-seed jobs. Their JSONs (`x2_linf-…json`, `x2_tf-…json`) agree with the per-seed ones: identical except one Linformer pool unit differing by 1 of 20,000 calls and read-out shifts differing by ~1e-8. The loader uses the per-seed files.
- `anrunw-ai-x2-plot-smoke` failed (KeyError 'calibrated'). It was a trial render that read a smoke JSON written before the calibration fields existed. It was re-run with a run filter (`x2-plot-smoke2`, exit 0). The first `x2-tfe100-*` submissions failed on argument parsing (`--tag -e100`) and were resubmitted with `--tag=-e100`.
- JEDI reference JSONs (`e1_e3.json`, `e6_verify.json`, `e6_mass.json`, `e10_layers.json`) were copied from the local `results/jedi/` to `/j-jepa-vol/aaron-interp/results/jedi/` for the plot / report jobs.
- Local copies: `results/small/x2_*.json`, `results/small/x2_report.md` (all tables of tests 1–5), `results/site/T8_small_mass_shift.png`, `results/site/T9_small_layers.png`.
