# X10 — Distributed alignment search (DAS) for M0, M2, Δ and the W−Z margin at S, h4, h5 of JEDI-linear (claims A4, A9, B2m; review weakness 3)

Status: done (two debug attempts used; see "Debug history").
Code: `interp/x10_das_jedi.py` (cluster, jax CPU), `interp/x10_plot.py` (figure, local).
Results: `results/jedi/x10_trained.json`, `x10_untrained{0,1,2}.json`, `x10_pfn.json`, `x10_summary.json` (Holm over 36 tests);
superseded runs kept as `x10_*_v0floattail.json` and `x10_*_v1ste.json`. Figure `results/jedi/x10_das.png`.
Logs: `logs/x10-tr3.log`, `x10-u0c.log`, `x10-u1c.log`, `x10-u2c.log`, `x10-sum3.log`, `x10-pfn.log`, smoke `x10-smoke3.log`.

## Conclusion
For each of four quantities, the method looks for a few directions (one, two or four) inside a layer of JEDI-linear such that
swapping only those directions between two jets changes the tagger's decision the same way as editing that quantity in the
input. The quantities are total pT, the pT-weighted squared radius about the stored axis, their combination ½ log(M0·M2), and the
W-versus-Z score. A single learned direction in the second-round object layer reproduces 87 % of the decision changes for total
pT. The squared-radius quantity and the W-versus-Z score are reproduced about as well, the latter mainly in the last two hidden
layers. The combination is reproduced at most at 0.62, in the first dense layer, and it shares directions with the squared-radius
quantity. Every learned subspace beats random directions of the same size. Two things are still uncertain: whether the
combination exists as one separate variable, and how far the method separates trained from untrained networks, which also
beat their own random directions in many cells, though at lower values.

## Procedure
Model: official JEDI-linear 3-feature n64 (`interp.jedi_core.Jedi(64)`), exact fixed-point forward pass for every reported number.
Samples: fit F = `sample(per_class=4000, seed=0)` (20,000 jets); test T = `sample(per_class=4000, seed=17)` minus the 1,590 jets also
in F (18,410 jets). Pairs: 8,000 ordered (base, source) pairs from F (seed 2; last 20 % for model selection); test "random arm"
5,000 ordered pairs from T (pair seed 1); "on-manifold arm" 5,000 pairs from T with the source's M0 within ±5 % of the base's (seed 3;
sd of log(M0_s/M0_b) 0.028 vs 0.145 in the random arm).

High-level model H (variables about the stored jet axis, real particles only):
$M_0=\sum_i p_{T,i}$, $M_2=\sum_i p_{T,i}r_i^2$ ($r_i^2=\Delta\eta_i^2+\Delta\phi_i^2$), $\Delta=\tfrac12\log(M_0M_2)$, and W−Z = $l_W-l_Z$
(the d7 logit margin). H's counterfactual for a pair = the exact network's call on the base jet after an input edit that sets V to
the source's value: do($M_0$): $p_T\times c$, angles $\times c^{-1/2}$, $c=M_{0,s}/M_{0,b}$ ($M_2$ unchanged); do($M_2$): angles
$\times\sqrt{M_{2,s}/M_{2,b}}$ ($M_0$ unchanged); do($\Delta$): $p_T\times\sqrt{M_{0,s}M_{2,s}/(M_{0,b}M_{2,b})}$; do(W−Z): base logits with
$(l_W,l_Z)$ set to their mean $\pm$ the source margin/2. Fraction of test pairs whose counterfactual call differs from the base call:
M0 0.186, M2 0.563, Δ 0.570, W−Z 0.261 (these pairs are $E_H$).

DAS: for stage L ∈ {S (64), h4 (64), h5 (32)}, each V and k ∈ {1, 2, 4}: $Q=\mathrm{qr}(W)$, $W\in\mathbb{R}^{d\times k}$; interchange
$h'=\Pi\big(h_b+(h_s-h_b)QQ^\top\big)$, where $\Pi$ clips each unit to the representable range of the next layer's fixed-point input
quantizer, widened to include the base value. The loss is the weighted cross-entropy of the network's own downstream layers,
which have exact fixed-point forward values and a straight-through gradient through the quantizers, against H's counterfactual
call. The two classes of pairs, call changes and no change, each carry half the weight. Adam with lr 0.01 runs for 800
full-batch steps, and the model is selected every 50 steps on the held-out training pairs.
Metric: effective IIA = fraction of $E_H$ test pairs on which the intervened network's call equals H's counterfactual call (primary);
plain IIA (all pairs) and the E22-style effective IIA (pairs where the intervention changes the call) are in the JSON. 95 % CI:
1,000 bootstrap resamples of test pairs (each pair has its own random base jet).
Controls: (i) 1,000 random orthonormal k-dim subspaces of the raw stage coordinates, same pairs and same clipping;
$p_\text{rot}=(1+\#\{\text{random}\ge\text{DAS}\})/1001$. (ii) DAS trained on the counterfactual labels of a deranged pair, evaluated on
the true labels; paired bootstrap difference. (iii) the same pipeline on `Jedi(untrained=True)` seeds 0–2 (200 random subspaces each;
see the note on calls below). (iv) X9 PFN (64,64,32) seed 0, pooled latent L (32 dims), 1,000 random subspaces.
Calibration (trained, V ∈ {M0, M2, Δ}, k ∈ {1, 2}): 5,000 jets of T (seed 4). The real edit is a 0.06 rad translation of every
real particle in a random direction per jet; it leaves m unchanged and changes $\tfrac12\log M_2$ by +0.303 on average. The matched
real mass change is an angle scaling by $e^{0.303}$. Patch: $h=\Pi(h_c+(h_\text{edit}-h_c)QQ^\top)$ at the stage, exact tail. Read-out:
ridge of log m on z, fitted on F. Reported: (patched Δlog m̂)/(full-edit Δlog m̂), W → Z among true-W jets (N_W = 1,021),
1,000 random subspaces, 1,000 jet-bootstrap CIs.
Tests: H0: DAS effective IIA = random-subspace effective IIA ($p_\text{rot}\ge0.05$); H1: higher. Holm over 4 V × 3 stages × 3 k = 36.

Untrained networks: each `Jedi(untrained=True)` seed gives one class for every jet (calls [0, 0, 0, 0, 18410] for seed 0 in the
first run), so with raw calls no pair changes the call. Their calls are taken as argmax(logits − mean logits on F). After centring,
the fraction of call-changing pairs is M0 0.111 / 0.002 / 0.057, M2 0.451 / 0.009 / 0.379, Δ 0.018 / 0.003 / 0.000 and
W−Z 0.452 / 0.005 / 0.138 (seeds 0 / 1 / 2). Cells with fewer than 20 such pairs are reported as "–".

## Debug history (two attempts, both before the reported runs)
- v0 (`x10_*_v0floattail.json`): DAS was trained through a float copy of d4–d7 (99.1–99.5 % call agreement with the exact tail on
  unedited states). On interchanged states the float copy agreed with the exact tail on only 21–37 % of calls (diagnostic added
  in v1), and v0's held-out training scores (0.53–0.89 balanced) did not carry over to the exact test (effective IIA 0.05–0.54).
- Debug 1 (`x10_*_v1ste.json`): training tail = exact fixed-point tail with a straight-through gradient; untrained calls centred.
  DAS then fell to 0.05–0.39 with balanced validation agreement 0.17–0.43, below the 0.5 an inert subspace gives.
- Debug 2 (reported): the next layer's input quantizers are unsigned fixed point and wrap modulo $2^i$ (`jetdistill/network.py`
  `round_wrap`). An interchange that drives a post-ReLU unit below 0, or above $2^i-2^{-f}$, wraps to an unrelated value. The
  clip $\Pi$ is applied to DAS training, DAS evaluation, the random null and the calibration patches. After it, the float copy
  agrees with the exact tail on 90.0 / 99.0 / 99.1 % of interchanged calls (S / h4 / h5), and the training tail on 100 %. Share of
  interchanged unit values clipped (random 4-dim swaps): S 74 %, h4 35 %, h5 27 %. At S, 41 of 64 units have an empty d4 input range
  (upper bound below 0, so they always pass 0); this gives the 41/64 = 64 % "unedited outside range" entries. At h4 the count is 7 of 64.
  The PFN has no quantizer, so its v0 run (`x10_pfn.json`) is unchanged by either fix (the float tail is its exact tail).

## Results (test pairs N = 5,000; effective IIA on $E_H$ with 95 % CI)
Checks: JEDI-linear accuracy on T 80.79 % (N 18,410; F 80.58 %), within 80.6–81.3 %. The exact tail from S / h4 / h5
reproduces `run` and `run_jetwise` overrides (max |Δlogit| 0). PFN accuracy on T 81.00 %, tail |Δlogit| 0.

| stage | V | k = 1 | k = 2 | k = 4 | random mean (k = 1/2/4) | permuted-label DAS (k = 1/2/4) | on-manifold arm (k = 1/2/4) | untrained s0 / s1 / s2 (k = 2) |
|---|---|---|---|---|---|---|---|---|
| S | $M_0$ | 0.87 [0.84, 0.89] | 0.86 [0.84, 0.89] | 0.85 [0.83, 0.87] | 0.10/0.14/0.17 | 0.16/0.14/0.25 | 0.71/0.69/0.70 | 0.06 (n 556) / – (n 9) / 0.19 (n 287) |
| S | $M_2$ | 0.66 [0.64, 0.68] | 0.75 [0.73, 0.76] | 0.55 [0.53, 0.57] | 0.04/0.07/0.10 | 0.16/0.07/0.16 | 0.67/0.75/0.55 | 0.09 (n 2254) / 0.14 (n 43) / 0.31 (n 1893) |
| S | $\Delta$ | 0.51 [0.49, 0.53] | 0.46 [0.44, 0.48] | 0.49 [0.47, 0.51] | 0.03/0.05/0.07 | 0.14/0.04/0.17 | 0.51/0.43/0.45 | 0.23 (n 88) / – (n 13) / – (n 0) |
| S | W−Z margin | 0.43 [0.40, 0.45] | 0.35 [0.33, 0.38] | 0.36 [0.34, 0.39] | 0.11/0.16/0.21 | 0.20/0.14/0.21 | 0.43/0.35/0.35 | 0.52 (n 2262) / 0.04 (n 23) / 0.26 (n 689) |
| h4 | $M_0$ | 0.66 [0.63, 0.69] | 0.83 [0.81, 0.85] | 0.84 [0.82, 0.86] | 0.07/0.09/0.13 | 0.16/0.16/0.26 | 0.47/0.63/0.63 | 0.37 (n 556) / – (n 9) / 0.37 (n 287) |
| h4 | $M_2$ | 0.68 [0.66, 0.69] | 0.74 [0.72, 0.76] | 0.78 [0.76, 0.79] | 0.02/0.03/0.04 | 0.08/0.15/0.14 | 0.68/0.75/0.80 | 0.17 (n 2254) / 0.16 (n 43) / 0.22 (n 1893) |
| h4 | $\Delta$ | 0.48 [0.46, 0.50] | 0.58 [0.56, 0.59] | 0.62 [0.61, 0.64] | 0.02/0.02/0.04 | 0.17/0.30/0.31 | 0.47/0.56/0.60 | 0.30 (n 88) / – (n 13) / – (n 0) |
| h4 | W−Z margin | 0.66 [0.63, 0.69] | 0.70 [0.67, 0.72] | 0.74 [0.72, 0.76] | 0.03/0.05/0.08 | 0.09/0.16/0.14 | 0.70/0.73/0.76 | 0.28 (n 2262) / 0.04 (n 23) / 0.34 (n 689) |
| h5 | $M_0$ | 0.34 [0.31, 0.38] | 0.72 [0.69, 0.75] | 0.74 [0.72, 0.77] | 0.08/0.12/0.18 | 0.17/0.20/0.28 | 0.36/0.55/0.61 | 0.12 (n 556) / – (n 9) / 0.32 (n 287) |
| h5 | $M_2$ | 0.26 [0.24, 0.27] | 0.68 [0.66, 0.69] | 0.73 [0.71, 0.74] | 0.03/0.05/0.08 | 0.09/0.11/0.15 | 0.24/0.72/0.78 | 0.23 (n 2254) / 0.23 (n 43) / 0.27 (n 1893) |
| h5 | $\Delta$ | 0.20 [0.19, 0.22] | 0.46 [0.44, 0.47] | 0.52 [0.50, 0.54] | 0.03/0.05/0.08 | 0.17/0.21/0.33 | 0.16/0.43/0.47 | 0.32 (n 88) / – (n 13) / – (n 0) |
| h5 | W−Z margin | 0.30 [0.27, 0.32] | 0.77 [0.74, 0.79] | 0.82 [0.80, 0.84] | 0.04/0.07/0.12 | 0.14/0.14/0.21 | 0.33/0.79/0.84 | 0.28 (n 2262) / 0.04 (n 23) / 0.69 (n 689) |
| PFN L | $M_0$ | 0.82 [0.80, 0.85] | 0.82 [0.80, 0.85] | 0.80 [0.77, 0.83] | 0.13/0.18/0.24 | 0.16/0.27/0.31 | 0.73/0.72/0.67 | – |
| PFN L | $M_2$ | 0.71 [0.69, 0.73] | 0.72 [0.71, 0.74] | 0.74 [0.73, 0.76] | 0.05/0.08/0.12 | 0.09/0.10/0.15 | 0.71/0.74/0.75 | – |
| PFN L | $\Delta$ | 0.58 [0.56, 0.59] | 0.60 [0.58, 0.61] | 0.63 [0.61, 0.65] | 0.04/0.07/0.10 | 0.20/0.20/0.26 | 0.60/0.62/0.65 | – |
| PFN L | W−Z margin | 0.31 [0.28, 0.33] | 0.31 [0.28, 0.33] | 0.57 [0.54, 0.60] | 0.08/0.12/0.18 | 0.07/0.14/0.10 | 0.28/0.29/0.58 | – |

Transfer (trained, k = 2, random arm): effective IIA of the subspace learned for V (rows) scored against the counterfactuals of V' (columns)
| stage | learned for | $M_0$ | $M_2$ | $\Delta$ | W−Z |
|---|---|---|---|---|---|
| S | $M_0$ | 0.86 | 0.07 | 0.12 | 0.08 |
| S | $M_2$ | 0.18 | 0.75 | 0.28 | 0.36 |
| S | $\Delta$ | 0.20 | 0.26 | 0.46 | 0.18 |
| S | W−Z margin | 0.19 | 0.13 | 0.09 | 0.35 |
| h4 | $M_0$ | 0.83 | 0.07 | 0.11 | 0.08 |
| h4 | $M_2$ | 0.17 | 0.74 | 0.27 | 0.43 |
| h4 | $\Delta$ | 0.24 | 0.39 | 0.58 | 0.39 |
| h4 | W−Z margin | 0.14 | 0.31 | 0.19 | 0.70 |
| h5 | $M_0$ | 0.72 | 0.08 | 0.11 | 0.10 |
| h5 | $M_2$ | 0.20 | 0.68 | 0.27 | 0.42 |
| h5 | $\Delta$ | 0.24 | 0.38 | 0.46 | 0.35 |
| h5 | W−Z margin | 0.14 | 0.21 | 0.13 | 0.77 |

| stage | patched subspace | share of translation read-out shift [95 %] | random null 2.5/50/97.5 % | p_rot | share of scaling shift | W→Z after patch | random W→Z median [97.5 %] | p_rot (W→Z) |
|---|---|---|---|---|---|---|---|---|
| S | $M_0$, k = 1 | -0.00 [-0.01, -0.00] | -0.08/0.06/0.39 | 0.865 | -0.01 | 2 % | 5 % [41 %] | 0.939 |
| S | $M_2$, k = 1 | 1.45 [1.44, 1.47] | -0.08/0.06/0.39 | 0.001 | 1.64 | 49 % | 5 % [41 %] | 0.019 |
| S | $\Delta$, k = 1 | 1.32 [1.31, 1.33] | -0.08/0.06/0.39 | 0.001 | 1.34 | 56 % | 5 % [41 %] | 0.006 |
| S | $M_0$, k = 2 | 0.01 [0.01, 0.01] | -0.08/0.11/0.50 | 0.850 | 0.04 | 2 % | 9 % [54 %] | 0.984 |
| S | $M_2$, k = 2 | 1.42 [1.41, 1.44] | -0.08/0.11/0.50 | 0.001 | 1.25 | 37 % | 9 % [54 %] | 0.094 |
| S | $\Delta$, k = 2 | 0.56 [0.55, 0.57] | -0.08/0.11/0.50 | 0.019 | 0.59 | 63 % | 9 % [54 %] | 0.005 |
| h4 | $M_0$, k = 1 | 0.01 [0.01, 0.01] | -0.07/0.00/0.12 | 0.351 | 0.01 | 2 % | 2 % [7 %] | 0.431 |
| h4 | $M_2$, k = 1 | 0.61 [0.60, 0.62] | -0.07/0.00/0.12 | 0.001 | 0.49 | 78 % | 2 % [7 %] | 0.001 |
| h4 | $\Delta$, k = 1 | 0.66 [0.65, 0.67] | -0.07/0.00/0.12 | 0.001 | 0.54 | 82 % | 2 % [7 %] | 0.001 |
| h4 | $M_0$, k = 2 | 0.02 [0.02, 0.02] | -0.08/0.01/0.16 | 0.407 | 0.00 | 2 % | 3 % [13 %] | 0.661 |
| h4 | $M_2$, k = 2 | 0.66 [0.65, 0.67] | -0.08/0.01/0.16 | 0.001 | 0.58 | 82 % | 3 % [13 %] | 0.001 |
| h4 | $\Delta$, k = 2 | 0.73 [0.72, 0.74] | -0.08/0.01/0.16 | 0.001 | 0.59 | 82 % | 3 % [13 %] | 0.001 |
| h5 | $M_0$, k = 1 | -0.01 [-0.01, -0.01] | -0.10/-0.00/0.16 | 0.564 | -0.03 | 3 % | 2 % [18 %] | 0.313 |
| h5 | $M_2$, k = 1 | 0.08 [0.08, 0.08] | -0.10/-0.00/0.16 | 0.124 | 0.11 | 43 % | 2 % [18 %] | 0.007 |
| h5 | $\Delta$, k = 1 | -0.01 [-0.01, -0.00] | -0.10/-0.00/0.16 | 0.587 | -0.16 | 10 % | 2 % [18 %] | 0.058 |
| h5 | $M_0$, k = 2 | -0.00 [-0.00, -0.00] | -0.11/0.01/0.24 | 0.612 | -0.01 | 4 % | 4 % [35 %] | 0.460 |
| h5 | $M_2$, k = 2 | 0.78 [0.77, 0.80] | -0.11/0.01/0.24 | 0.001 | 0.78 | 78 % | 4 % [35 %] | 0.001 |
| h5 | $\Delta$, k = 2 | 0.34 [0.33, 0.35] | -0.11/0.01/0.24 | 0.009 | 0.26 | 88 % | 4 % [35 %] | 0.001 |

The table above has three parts:
- The main table: effective IIA of DAS for each stage and variable, plus the random-subspace, permuted-label, on-manifold and
  untrained controls.
- The transfer table.
- The calibration table: the DAS subspace patched from translated jets. The full translation gives a calibrated read-out ratio
  of 1.07 and sends 61 % of W jets to Z; the matched scaling sends 13 %; unedited, 2 % of W jets are called Z.

Tests (`x10_summary.json`): all 36 trained cells have $p_\text{rot}$ = 0.001 (the floor, 1/1001); Holm-adjusted 0.036 for each; 36 / 36
below 0.05. DAS − permuted-label DAS (paired bootstrap): +0.03 to +0.72 across the 36 trained cells, p = 0.001 in every cell.
On-manifold arm (source M0 within ±5 %): values within 0.06 of the random arm for M2, Δ and W−Z; M0 lower (0.69–0.71 at S vs
0.85–0.87), on 296 call-changing pairs. E22-style effective IIA (pairs where the intervention changes the call): S M0 k=1 0.77,
S M2 k=2 0.71, h4 Δ k=4 0.71, h5 W−Z k=4 0.70.

Untrained networks (k = 2, see table): effective IIA 0.04–0.69 in the cells with ≥ 20 call-changing pairs, against 0.35–0.86 for the
trained network. In 41 of the 72 seed-0 / seed-2 cells the untrained DAS also has $p_\text{rot}\le0.05$ against its own random
subspaces (200 draws), so "DAS above random subspaces" by itself does not separate trained from untrained networks; the absolute
values do in most cells. Exceptions: W−Z at S for seed 0 (0.48–0.52 vs trained 0.35–0.43) and W−Z at h5 for seed 2 (0.69
vs 0.77–0.82). Δ has 0–88 call-changing pairs in the untrained networks, so the Δ control is not informative.

## Predictions and which are excluded
- Alternative "no causal subspace" (DAS within the random band at all k ≤ 4): excluded for every V and stage (36 / 36, Holm 0.036).
- Thesis "Δ aligned at h4/h5 with effective IIA ≥ 0.8 and $p_\text{rot}$ < 0.01": the $p_\text{rot}$ part holds; the ≥ 0.8 part does not:
  maximum 0.62 [0.61, 0.64] at h4, k = 4 (0.58 at k = 2). Δ rises from S (0.46–0.51) to h4 (0.48–0.62) and falls at h5 (0.20–0.52).
- Thesis "M0 and M2 aligned already at S, k = 1–2": M0 0.87 [0.84, 0.89] at S with k = 1; M2 0.75 [0.73, 0.76] at S with k = 2.
- Alternative "Δ only at S with the same value as M2": not supported. At S, Δ (0.46–0.51) is below M2 (0.55–0.75), and Δ is
  higher at h4 than at S for k = 2 and 4. The Δ subspace and the M2 subspace overlap partly: the Δ subspace scores 0.26 / 0.39 / 0.38
  on the M2 counterfactuals (S / h4 / h5, k = 2), and the M2 subspace scores 0.27–0.28 on Δ.
- Thesis "untrained at chance": partly. Untrained values are lower in most cells, but untrained DAS beats its own random subspaces
  in 41 / 72 cells.
- The class variable (W−Z margin) is located later than the mass moments: 0.35–0.43 at S, 0.66–0.74 at h4, 0.77–0.82 at h5 (k ≥ 2).

## Reading (scoped to these measurements)
On 5,000 test pairs, the exact fixed-point JEDI-linear network has learned linear subspaces of S, h4 and h5. When those subspaces
are swapped between jets, the network's call reproduces the effect of input edits of M0 and M2 about the stored axis on
0.68–0.87 of the call-changing pairs (best k ≤ 2 per stage), against 0.02–0.14 for random subspaces of the same size. The M0 subspace carries
none of the 0.06 rad translation's mass read-out shift (share 0.00–0.02), as expected because a translation leaves M0 unchanged.
The Δ subspace at h4 (k = 2) carries 0.73 [0.72, 0.74] of the shift (0.59 of a matched real mass change) and sends 82 % of W jets
to Z, more than the full translation (61 %). The M2 subspace at S carries 1.42–1.45 of the shift. Δ = ½ log(M0·M2) is captured at
most at 0.62 by a ≤ 4-dim subspace, and its subspace overlaps with the M2 subspace. The PFN latent gives similar values for M0
(0.80–0.82), M2 (0.71–0.74) and Δ (0.58–0.63). The X9 comparison (formula-replacement IIA 0.701 vs random rotations 0.705) used
rotations of the named-unit space without the range clip. Whether out-of-range wrapped values also affected that baseline was not
tested here.

## Issues / limits
- The range clip $\Pi$ is part of the intervention. Without it, swapped states outside the quantizer range wrap (v0/v1 runs), and
  DAS and random subspaces both stay at 0.05–0.54.
- Δ's counterfactual (pT scaling) also changes the pT-rank-window objects through the l1 pT ramps, so the Δ edit is not a pure
  change of Δ inside the network. An angle-only Δ edit was not run.
- 7.6 % of pairs need an angle scale > 3 for do(M2) (off-manifold edits); the on-manifold arm matches M0 only.
- Untrained control: centred-logit calls (raw calls are constant); seed 1 has < 50 call-changing pairs for every V.
- Bootstrap is over test pairs; 24 % of pairs share their base jet with another pair.
- Resources: x10-tr3 1 CPU / 3 GiB, 640 s, peak 1,812 MiB, 0.63 cores; untrained 524–529 s, peak 1.7 GiB.

## Explainer entry (Paper A tab; site/explain.py)
**Question.** Does JEDI-linear hold the mass ingredients $M_0$, $M_2$ and $\Delta=\tfrac12\log(M_0M_2)$, and the W-versus-Z score, in a few
linear directions of a layer? If it does, where does each of these first appear?

**Background.** Interchange intervention: take two jets, a base and a source, copy part of a layer's activations from the source
into the base, and run the rest of the network. If that part is the variable V, the network should then call the base jet as if
its V had been set to the source's value by an input edit. Distributed alignment search (DAS; Geiger et al. 2023) learns which
directions to copy instead of fixing them by hand. Earlier, X9 swapped units named by fixed formulas and found that this matched
the input edits no better than random rotations of the same units.

**Method.** For a layer $h\in\mathbb{R}^d$ ($S$, $h_4$, $h_5$) and an orthonormal $Q\in\mathbb{R}^{d\times k}$ ($Q=\mathrm{qr}(W)$), the
interchanged state is
$$h'=\Pi\!\left(h_b+(h_s-h_b)\,QQ^\top\right),$$
with $\Pi$ the clip to the fixed-point range of the next layer's input. $Q$ is trained by gradient descent on
$$\mathcal{L}(Q)=-\sum_{(b,s)}w_{bs}\,\log\operatorname{softmax}\!\big(f_{>L}(h')\big)_{y^H_{bs}},$$
where $f_{>L}$ is the network's own downstream layers and $y^H_{bs}$ is the network's call on the base jet after the input edit
do($V=V_s$). The edits are: $p_T\to c\,p_T$ with angles $\times c^{-1/2}$ for $M_0$; angles $\times\sqrt{M_{2,s}/M_{2,b}}$ for $M_2$; $p_T\times\sqrt{M_{0,s}M_{2,s}/M_{0,b}M_{2,b}}$ for $\Delta$.
The score is effective IIA, $\frac{1}{|E_H|}\sum_{(b,s)\in E_H}\mathbf 1\!\left[\arg\max f_{>L}(h')=y^H_{bs}\right]$, over the test pairs
$E_H$ whose counterfactual call differs from the base call. It is compared with 1,000 random $Q$.

**Result.** Figure `results/jedi/x10_das.png`. $M_0$: 0.87 at $S$ with one direction (random 0.10). $M_2$: 0.75 at $S$ and 0.78 at $h_4$.
W−Z margin: 0.43 at $S$ and 0.82 at $h_5$. $\Delta$: 0.51 at $S$, 0.62 at $h_4$, 0.52 at $h_5$. All 36 cells are above random subspaces
(Holm 0.036). Patching the 2-direction $\Delta$ subspace at $h_4$ from 0.06 rad translated jets reproduces 73 % of the translation's
mass read-out shift and sends 82 % of W jets to Z. The $M_0$ subspace reproduces 0–2 %, since a translation does not change $M_0$.

**Checks.** The exact fixed-point network is used throughout (accuracy 80.79 % on the test jets). Controls: random subspaces of the
same size, DAS trained on mismatched labels, three untrained networks, an on-manifold arm with $M_0$ matched to ±5 %, and the PFN
latent from X9. Two pipeline faults were found and corrected before the reported run (float-copy mismatch; wrap of out-of-range values).

**Why it matters.** The mass moments about the stored axis are causal variables in low-dimensional subspaces of JEDI-linear: $M_0$ and
$M_2$ from the second-round objects on, and the W/Z decision mainly at $h_4$–$h_5$. The combination $\Delta$ is reproduced less
completely (≤ 0.62) and shares directions with $M_2$. So the network's mass response runs through these subspaces, but a single
separable $\Delta$ variable is supported only in part.
