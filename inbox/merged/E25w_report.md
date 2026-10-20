# E25w — 16-feature JEDI-linear: where the mass variables are held, W-call matrix over translation dose, non-constant baselines

Claims A1f / A4 / A9 (exploratory, second official family). Script `interp/e25w_jedi16_mass.py` (parts `form`, `dose`, `base`),
figure `interp/e25w_plot.py` -> `results/jedi/e25w.png`. Cluster jobs (1 CPU / 2 GiB each, git 6fe40dc): `e25w-run`
(form, 500 random subspaces), `e25w-run2` (form, `--nrand 5000`, so that Holm over 30 cells can go below 0.05), `e25w-dose`,
`e25w-base`; smoke runs `e25w-smoke-*`. Measured use: peak 1.3–1.6 GiB, 0.5–0.93 CPU on average; wall 12–15 min each.
Results: `results/jedi/e25w_form.json`, `e25w_form_nrand5000.json` (DAS values identical; only the random null is larger),
`e25w_dose.json`, `e25w_base.json`; logs in `logs/e25w-*.log`.

## Conclusion

In the 16-feature network the jet's summed pT (M0) and its pT-weighted squared spread about the stored axis (M2) can each be
swapped between jets through a one- or two-dimensional direction already at the second pooled sum S, the earliest dense
stage tested, and this swap works less well in later layers; the 3-feature expectation of a late formation step (h4/h5) is
not what was measured. Moving all particles off the axis sends W calls first to Z (87 % at 0.045 rad) and then to top and
quark/gluon, and a change of only the ΔR column that gives the same M2 change reproduces this at every dose up to 0.06 rad.
Ten non-trained networks with a refitted read-out that classify above 40 % give smaller W→Z responses than the trained
network, but with only ten such baselines the permutation p-value cannot go below 0.09, and the formation between the first
pooled sum P and S was not resolved by interchange.

## Sample

`sample(per_class=1500, seed=2521)` minus the rows of sample(3000, 2501), sample(1500, 2502), sample(1500, 2507) (E25, E25
fresh, E25v): N = 6,694 jets (g 1344, q 1330, W 1330, Z 1354, t 1336). All 16 `val_c16_f16.npy` columns at these rows; the
(pt, etarel, phirel) columns equal `sample()` x exactly (max difference 0). Fit / test halves 3,347 / 3,347 by
`rng(2522).permutation`.

Unedited accuracy (all 6,694 jets): 16f n64 82.07 % [81.14, 82.97]; 16f-perminv n64 82.64 % [81.72, 83.53]; 16f n32 81.40 %;
16f n128 82.22 %. The 80.6–81.3 % range in the task is the 3-feature model's; the 16f n64 value is inside the E25 range on
other samples (81.62 % [80.99, 82.23] main, 81.88 % fresh). The script's own forward pass equals `Jedi.run` (max |Δlogit| 0) and
the exact tails from S, h4, h5, z reproduce the logits (max |Δlogit| 0).

## (1) Formation stage

**Procedure.** Variables about the stored axis: M0 = Σ pT, M2 = Σ pT ΔR², D = ½ log(M0·M2). Interchange do(V := source value)
on real jets: do(M0) scales pT by c = M0_s/M0_b and angles by c^−1/2 (M2 fixed); do(M2) scales angles by √(M2_s/M2_b);
do(D) scales pT. All 16 columns recomputed. Counterfactual label = the network's exact call on the edited base jet (calls change
in 19 % (M0), 55 % (M2), 56 % (D) of test pairs). 3,000 training pairs from the fit half, 2,000 test pairs from the test half.
DAS: k-dim orthonormal Q at S, h4, h5, z, logits, h' = clip(h_b + (h_s − h_b)QQᵀ), Q trained by Adam through straight-through
quantizers, evaluated with the exact fixed-point tail. Metric: effective IIA (agreement on test pairs whose counterfactual call
differs from the base call), 1,000-resample bootstrap. Controls: 5,000 random k-subspaces per cell (p_rot), full-stage swap
(source call), paired bootstrap DAS − full swap, the same DAS on baseline networks (part 3). Holm over 5 stages × 3 variables × 2 k.

**Numbers, 16f n64, k = 2** (effective IIA [95 % CI]; random-subspace mean / 97.5 % quantile; full swap):

| stage | M0 | M2 | D |
|---|---|---|---|
| S | 0.715 [0.673, 0.762]; rnd 0.10/0.16 | 0.707 [0.679, 0.733]; rnd 0.03/0.09 | 0.581 [0.550, 0.611]; rnd 0.03/0.06 |
| h4 | 0.728 [0.684, 0.775] | 0.659 [0.631, 0.687] | 0.491 [0.463, 0.522] |
| h5 | 0.572 [0.524, 0.623] | 0.622 [0.593, 0.651] | 0.405 [0.378, 0.436] |
| z | 0.452 [0.399, 0.503]; rnd q975 0.30 | 0.585 [0.555, 0.614]; rnd q975 0.24 | 0.305 [0.278, 0.331] |
| logits | 0.410 [0.359, 0.460]; rnd q975 0.43 | 0.378 [0.350, 0.406]; rnd q975 0.39 | 0.237 [0.212, 0.262] |
| full swap | 0.350 | 0.577 | 0.320 |

- k = 1 at S: M0 0.697, M2 0.675, D 0.471; at h5: 0.339, 0.364, 0.256.
- Holm (5,000 random subspaces): every S, h4, h5 cell and z k=2 have p_rot = 0.0002, Holm p = 0.006; z k=1 M0 Holm 0.29;
  all logits cells Holm ≥ 0.29.
- DAS − full swap for M2: S +0.131 [0.096, 0.165], h4 +0.082 [0.047, 0.114], h5 +0.046 [0.009, 0.080], z +0.008
  [−0.025, 0.041], logits −0.199. For M0: S +0.366 [0.300, 0.429], h4 +0.379, h5 +0.222, z +0.102, logits +0.06 [−0.00, 0.12].
- Specificity (transfer): the M0 subspace at S reproduces M2 counterfactuals at 0.06 and the M2 subspace reproduces M0 at
  0.18; the D subspace reproduces M0 0.31, M2 0.31.
- P (first pooled sum) full swap, effective IIA: M0 0.269, M2 0.184, D 0.131 (DAS was not run at P).
- 16f-perminv, k = 2: S M0 0.816, M2 0.721, D 0.361; h4 0.754 / 0.707 / 0.437; h5 0.577 / 0.635 / 0.352; z 0.506 / 0.584 /
  0.275; full swap 0.359 / 0.579 / 0.319.
- Ridge read-out (test half R²; calibration = read-out shift under the ΔR-only dilation dilR_0.06 / true shift; read-out
  shift under translation T_0.06 / under dilR_0.06), 16f n64, log M2:

| stage | P | g | S | h4 | h5 | z | logits |
|---|---|---|---|---|---|---|---|
| R² | 0.827 | 0.917 | 0.913 | 0.986 | 0.963 | 0.949 | 0.792 |
| calibration | 0.68 | 1.11 | 0.84 | 0.93 [0.92, 0.94] | 0.90 | 0.84 | 0.42 |
| T / dilR | 1.21 | 0.85 | 1.21 | 1.09 [1.08, 1.10] | 1.07 | 1.17 | 1.55 |

  log M0 R²: P 0.990, S 0.934, h4 0.943, h5 0.86, z 0.78, logits 0.37.

**Reading.** Separate low-dimensional causal subspaces for M0 and for M2 are present at S (the earliest stage where DAS was
run) and their IIA decreases from S/h4 to logits; M2 exceeds the full-stage swap at S, h4, h5 and equals it at z. The
log-product D has lower IIA than M2 at every stage. The thesis prediction (IIA rising at h4/h5 above S) is not observed; the
alternative "IIA ≈ full swap at every stage" is excluded for S–h5. The linear read-out of log M2 is highest at h4 (R² 0.986),
which matches the 3-feature result on readability, but the interchange result places the causal M2 direction at S already.
Whether M2 is formed between P and S (the u = ReLU(W3(s+g)) layer) or is already at P was not resolved: P was tested only by
full swap and ridge.

## (2) Dose and call matrix

**Procedure.** For d ∈ {0.015, 0.03, 0.045, 0.06, 0.09, 0.12} rad: T_d rigid translation Δη += d (all 16 columns recomputed);
dilR_d ΔR column × s_d with s_d = √(M2(T_d)/M2) (same M2 change, nothing else moved); dil_d all angle columns × s_d;
R_d rotation about the axis by the angle that moves a particle at the pT-weighted mean r by d (control). All 6,694 jets;
5×5 transition (unedited → edited call) per arm; true-W new-call fractions with 2,000-resample bootstrap; McNemar (Z call on
true W) vs R_d, Holm over 4 models × 6 doses × 3 arms. Median s_d: 1.02, 1.07, 1.14, 1.25, 1.50, 1.80.

**Numbers, 16f n64, true W jets (N 1,330), new-call fractions under T_d** (unedited: W 0.875, Z 0.014, t 0.015, q 0.080, g 0.016):

| d (rad) | Z | t | q+g | W | W→Z flow T [CI] | flow dilR | T/dilR [CI] | Z under R |
|---|---|---|---|---|---|---|---|---|
| 0.015 | 0.038 | 0.020 | 0.089 | 0.853 | +0.024 [0.016, 0.032] | +0.024 | 1.00 [0.74, 1.32] | 0.014 |
| 0.03 | 0.517 | 0.029 | 0.073 | 0.382 | +0.502 [0.476, 0.530] | +0.505 | 1.00 [0.96, 1.04] | 0.014 |
| 0.045 | 0.883 | 0.021 | 0.066 | 0.029 | +0.869 [0.849, 0.888] | +0.861 | 1.01 [1.00, 1.02] | 0.014 |
| 0.06 | 0.503 | 0.214 | 0.260 | 0.023 | +0.489 [0.461, 0.517] | +0.453 | 1.08 [1.03, 1.14] | 0.014 |
| 0.09 | 0.048 | 0.645 | 0.305 | 0.002 | +0.034 [0.020, 0.047] | +0.002 | undefined | 0.014 |
| 0.12 | 0.002 | 0.783 | 0.216 | 0.000 | −0.013 [−0.020, −0.006] | +0.006 | undefined | 0.014 |

- Under dilR at 0.09: Z 0.017, t 0.518, q+g 0.454. At 0.09 the translation sends more W to top (0.645 [0.620, 0.670]) than the
  matched dilation (0.518).
- Transition of W-called true-W jets under T_0.09: 764 → t, 344 → q, 46 → g, 10 → Z (of 1,164). Under T_0.06: 639 → Z, 249 → q,
  237 → t, 38 → g.
- McNemar Z(T) vs Z(R): Holm p ≤ 1e-7 at 0.015–0.06, 7.6e-6 at 0.09, 0.0068 at 0.12 (where Z under T is below R).
- Mean M2 change expressed as W mass × exp(½ Δlog M2): 83, 90, 99, 109, 132, 158 GeV for the six doses.
- 16f-perminv: same pattern; Z peak 0.890 at 0.045; T/dilR 1.06, 0.95, 1.005, 0.985 at 0.015–0.06; at 0.09 Z 0.038, t 0.650.
- 16f n32 (checkpoint present): Z peak 0.811 at 0.045; T/dilR 1.90 [1.42, 2.63] at 0.015, 0.95, 0.92, 0.94 [0.89, 1.01] at
  0.03–0.06; at 0.09 t 0.611, Z 0.037.
- 16f n128 (checkpoint present): Z peak 0.868 at 0.045; T/dilR 0.63 [0.36, 0.97], 0.91, 1.00, 1.24 [1.17, 1.31] at 0.015–0.06;
  at 0.09 t 0.547, q 0.378, Z 0.049.

**Reading.** The non-monotone W→Z noted in E25v is a passage through the Z window: W→Z peaks at 0.045 rad (M2-equivalent
mass ≈ 99 GeV) and at larger doses W jets are called top and q/g. A ΔR-only change with the same M2 change gives the same
W→Z flow within 0–8 % at 0.015–0.06 rad for n64 and perminv, within 5–24 % for n32 / n128 at 0.03–0.06; rotation of matched
displacement gives no flow. At 0.09 rad translation and matched dilation differ in the top / q-g split (0.645 vs 0.518 top).

## (3) Non-constant baselines

**Procedure.** Read-out d7 replaced by a multinomial logistic regression on iq(ReLU(z)) fitted on the fit half for every
baseline. Variants: trained_refit (trained body); init0–4 (trained l1, later fused kernels redrawn Glorot-uniform, biases 0);
gauss0–4 (same layers redrawn from N(mean, sd) of the trained kernel/bias); perm0–9 (non-zero entries of every fused kernel
and bias of l1..d6 permuted within the layer, masks and quantizers kept). Qualify: test accuracy > 0.40 and > 1 distinct call.
Metric on the test half, true W (N 687): flow = Δ(fraction called Z) under T_0.06, ratio = flow(T_0.06)/flow(dilR_0.06).
H0: the trained network's flow / ratio is inside the qualifying-baseline distribution; p_flow = (1 + #{b: flow_b ≥
flow_trained})/(1+n); p_ratio = (1 + #{b: |ratio_b − 1| ≤ |ratio_trained − 1|})/(1+n), undefined ratios counted as farther.
DAS k = 2 (50 random subspaces, 400 steps) on every qualifying baseline at S, h4, h5, z.

**Numbers.** 10 of 20 qualify (init 4/5, gauss 5/5, perm 1/10; all 20 give 5 distinct calls).

| network | test acc | W→Z unedited | flow T_0.06 [CI] | flow dilR | ratio [CI] |
|---|---|---|---|---|---|
| trained | 0.816 | 0.017 | +0.498 [0.459, 0.539] | +0.454 | 1.10 [1.02, 1.18] |
| trained, refit read-out | 0.817 | 0.020 | +0.384 [0.344, 0.425] | +0.367 | 1.05 [0.96, 1.15] |
| qualifying baselines (10) | 0.40–0.50 | 0.19–0.37 | −0.001 to +0.231 (median 0.07) | −0.023 to +0.199 | 1.16–19 (7 defined), 3 undefined |

- Largest baseline flow: init1 +0.231 [0.182, 0.278], ratio 1.16 [0.92, 1.47]; gauss0 +0.143, ratio 1.51 [1.09, 2.49];
  gauss4 +0.127, ratio 1.34 [0.88, 2.33]. Rotation flow in all networks within ±0.012.
- p_flow = 0.091 and p_ratio = 0.091 (n = 10; 1/11 is the smallest attainable value).
- Baseline DAS (k = 2) for M2 at S: 0.18–0.38 (random-subspace q97.5 0.14–0.30; full swap 0.22–0.34), vs trained 0.70
  (trained_refit 0.68).

**Reading.** Non-trained bodies with a trained l1 and a fitted read-out respond to the translation in the same direction and
with a T/dilR ratio near or above 1 where it is defined, but the flow is 0–0.23 against 0.50 for the trained network, and
their low-dimensional M2 subspace interchange is near their random-subspace level. The prediction "random body + trained
read-out reproduces the trained ratio and size" is excluded for the size of the flow (all 10 below); the ratio itself is not
distinguishable (baseline ratios ≥ 1 with wide intervals). With 10 qualifying baselines the permutation test cannot reach
p < 0.05. Baselines call 19–37 % of true W jets Z before any edit, which limits their available flow.

## (4) Other sizes

16f n32 and n128 official checkpoints exist and were run in part (2) (numbers above).

## Issues / limits

- DAS was not run at P or at u (per-particle stage), so formation between P and S is unresolved.
- The baseline test has 10 qualifying networks; p ≥ 1/11.
- The perm variant qualifies in 1/10 draws (accuracies 0.32–0.40).
- The 82.07 % accuracy is above the 80.6–81.3 % range given in the task; that range belongs to the 3-feature model.
- site/explain.py and site/conclusions.py were not edited (shared files; proposed entries below for the ledger agent).

## Proposed explainer entry (E25W_EXPLAINER)

- **Question.** In the official 16-feature JEDI-linear, which reads angles only through ΔR, at which layer can the mass
  variables be swapped between jets, where do W jets go under increasing off-axis shifts, and is the response specific to
  trained weights?
- **Background.** $M_0=\sum_i p_{T,i}$, $M_2=\sum_i p_{T,i}\Delta R_i^2$ about the stored axis; for small angles
  $m^2\approx M_0M_2$. E25/E25v: a 0.06 rad shift turns about half of W jets into Z calls; untrained redraws gave constant calls.
- **Method.** Distributed alignment search (DAS): learn a $k$-dimensional subspace $Q$ at a stage $h$ such that
  $h'=h_b+(h_s-h_b)QQ^\top$ gives the call the network makes when $V$ of the base jet is set to the source's value; effective
  interchange accuracy on pairs where that call changes; random subspaces and full-stage swap as controls. Dose: shift
  $d\in[0.015,0.12]$ rad vs a ΔR-only dilation with the same $M_2$ and a rotation. Baselines: random bodies (trained first
  layer) and permuted weights, read-out refitted by logistic regression.
- **Result.** Figure `results/jedi/e25w.png`. M2 IIA 0.71 at S, 0.66 h4, 0.62 h5, 0.59 z, 0.38 logits (full swap 0.58, random
  ≤ 0.09 at S). W→Z peaks at 0.88 for 0.045 rad, then top 0.65 / q-g 0.31 at 0.09 rad; dilation matches to 0–8 %. Trained flow
  0.50 vs 10 baselines −0.00 to 0.23 (p 0.09, the minimum for 10).
- **Checks.** perminv, n32, n128 give the same dose pattern; 5,000 random subspaces, Holm over 30 cells; exact tails
  reproduce the logits.
- **Why it matters.** The 16-feature net holds M0 and M2 separately from the second pooled sum on, rather than forming the
  product late as in the 3-feature net; the un-recentred response passes W through the Z window into top as M2 grows.

## Proposed conclusion (site/conclusions.py, EXPLAINER['E25W_EXPLAINER'])

Same text as the Conclusion section above.
