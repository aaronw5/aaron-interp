# E25x — where $M_0$ and $M_2$ first appear between the first layer and $S$ (3-feature and 16-feature JEDI-linear n64)

Script `interp/e25x_p_to_s.py` (imports the E25w routines `pair_edits`, `Net`, `das_fit`, `eval_Q`, `variant_KB`, `refit_d7`
unchanged; adds an exact per-particle forward `PNet` with interventions at `a`, `g`, `u`). Figure script `interp/e25x_plot.py`.
Cluster jobs (all 1 CPU / 2 GiB, exit 0): `e25x-{3f,16f}-{m0,m2,arm3}` (trained net, one variable per job; Arm 3 + probe),
`e25x-{3f,16f}-b{0..4}` (baseline shards), `e25x-comb` (combine), smokes `e25x-smoke-tr`, `e25x-smoke-b16`. Process peak RSS
1.38–1.58 GiB; pod cgroup peak 2,047 MiB including page cache from the memory-mapped inputs; longest wall 5,814 s (16f M0).
Results: `results/jedi/e25x_3f.json`, `e25x_16f.json` (combined cells, Holm, baselines), `e25x_{3f,16f}_trained_{M0,M2,arm3}.json`,
`e25x_{3f,16f}_base{0..4}.json`; figure `results/jedi/e25x.png`; logs `logs/e25x-*.log`. Nothing committed.

## Conclusion

In both official networks, the total jet pT ($M_0$) and the pT-weighted squared spread about the axis ($M_2$) can already be swapped
between jets through a single direction in each particle's first-layer vector: replacing that one number for every particle with
its value after an input edit reproduces the edited jet's call far more often than in networks without a trained function (for
$M_2$: 0.53 in the 3-feature net and 0.72 in the 16-feature net, against ≤ 0.45 for the non-trained networks). So the mass inputs
are not first created after the second pooling; per-particle precursors exist from the first layer on. In the 3-feature net the
$M_2$ swap accuracy keeps rising through the second per-particle layer to the second pooled layer ($S$), while in the 16-feature net
(which receives ΔR directly) it is already at its $S$ level at the first layer. What remains open: the first-layer direction is not
a linear read-out of $p_{T,i}\,r_i^2$ in the 3-feature net, and some fully random networks also carry $M_0$ this way, so part of the
$M_0$ result reflects the architecture (pT enters each particle linearly) rather than training.

## Sample and exactness

- Fit: `jedi_core.sample(2000, 2551)` (10,000 jets); test: `sample(2000, 2552)` minus 374 rows also in the fit sample (9,626 jets).
  All 16 jetConstituentList columns from `val_c16_f16.npy`, max difference to `sample()` x = 0. The 3f net reads the pt / etarel /
  phirel columns of the same (edited) arrays, so both nets see identical pairs and edits.
- Pairs (base, source): 3,000 from the fit sample (rng 2553; DAS training, last 20 % for model selection), 3,000 from the test sample
  (rng 2554; all reported IIA). Test-pair edit spread: sd log c (M0) 0.157, sd log a (M2) 0.732; 13.8 % of M2 pairs scale angles
  outside [1/3, 3].
- Accuracy (unedited): 3f n64 fit 80.55 %, test 80.38 % [79.57, 81.16]; 16f n64 fit 81.62 %, test 81.33 % [80.54, 82.10]
  (the 3f test value is 0.2 points under the 80.6–81.3 % band; its Wilson interval covers it; E25wv's 3f sample gave 80.33 %).
- Own forwards vs `Jedi.run`: max |Δlogit| 0 (E25w `Net` and the new `PNet`, both models). PNet on base pairs vs Net: 0; PNet call on
  the donor jet = counterfactual label in 100 % of checked pairs.
- Counterfactual call changes (test pairs): 3f M0 0.206 (n 619), M2 0.566 (n 1,698), D 0.580; 16f M0 0.188 (n 563), M2 0.573
  (n 1,719), D 0.579.

## Procedure

Edits on the base jet (E25w `pair_edits`): do(M0): $p_T \times c$, angles $\times c^{-1/2}$, $c = M_{0,s}/M_{0,b}$ (every $p_{T,i}$
changes, every $p_{T,i} r_i^2$ fixed); do(M2): angles $\times \sqrt{M_{2,s}/M_{2,b}}$ ($p_{T,i}$ fixed, every $p_{T,i} r_i^2$ changes).
Label = the network's exact call on the edited base jet. Intervention $h' = \mathrm{clip}(h_b + (h_x - h_b) Q Q^\top)$, $Q$
orthonormal $k$ columns, $k \in \{1, 2\}$.
- Design **don** (all stages): $h_x$ = the same stage of the edited base jet. At the per-particle stages `a` (l1 output; its pool is P)
  and `u` (l3 output; its pool is S) the same $Q$ acts on every particle, and the exact network runs on from there (pool to P, g,
  l2, add, l3, pool to S, h4, h5, z, d7). At `g` only g is mixed (l2 path from the base jet); at `S` the E25w routine.
- Design **src** (g, S): $h_x$ = the source jet's stage vector (E25w / X10 design).
- Clip: per-unit range over the fit sample widened to the base value (a, g, u); E25w quantizer range at S.
- Training: Adam 0.01, straight-through estimator through every quantizer, 600 steps (trained net; minibatch 512 at a / g / u, full
  batch at S), balanced class-change weights, selection on balanced held-out agreement every 50 steps.
- Metric: effective IIA = agreement with the label on test pairs whose label differs from the base call; 1,000 bootstrap resamples
  over pairs (identical resample indices in every cell, so stage differences are paired).
- Controls: 1,000 random orthonormal $k$-subspaces per trained cell ($p_\mathrm{rot}$, Holm over 24 cells; floor 0.024); full swap
  at the stage; 45 non-trained networks per model with a logistic-regression refit of d7 on the fit sample: untr 0–9 (`jedi_core`
  untrained: every kernel, bias, BN scale redrawn), init 0–14 (l1 trained; l2, dd, l3, d4–d6 Glorot re-init), gauss 0–14 (same
  layers N(mean, sd)), perm 0–4 (X8 variant a). Qualify: test accuracy > 0.30 and ≥ 3 distinct calls. (The E25w bar of 0.40 was met
  by no 3f baseline in the first run of this script; it was lowered before any baseline DAS ran.) Qualifying: 3f 32 (untr 2,
  init 15, gauss 12, perm 3), 16f 41 (untr 6, init 15, gauss 15, perm 5). Each baseline: same pairs and edits, its own labels,
  250 steps, minibatch 384, no random subspaces.
- H0 per cell (stage × variable × $k$ × design): trained IIA ≤ the 95th percentile (q95) of the qualifying baselines; H1: above.
  $p_{q95} = (1 + \#\{\text{bootstrap replicates of trained IIA} \le q95\})/1001$; Holm over the 24 cells (floor 0.024). Also the rank
  $p = (1 + \#\{b: \mathrm{IIA}_b \ge \mathrm{IIA}_{tr}\})/(1+n_b)$.
- Per-particle probe (correlational, with edit calibration): ridge read-out of $p_{T,i}$ and $p_{T,i} r_i^2$ from $a_i$ and $u_i$
  (real particles of 2,000 fit jets; R² on 137,117 test-pair particles); calibration = slope of read-out change on true change
  under the do(M0) / do(M2) donor edits.
- Arm 3 (D without a pT rescale): the $k = 1$ src DAS directions $q_0$ (M0) and $q_2$ (M2) at S; calibration
  $[q_0\cdot S, q_2\cdot S] = \alpha + B\,[\log M_0, \log M_2]$ (OLS on fit jets). do$_S$(D): $S' = S_b + Q_m (Q_m^\top Q_m)^{-1} B
  (\delta, \delta)$, $\delta = D_s - D_b$ (M0 × λ and M2 × λ in the S coordinates, λ = e^δ); fixed-D control $B(\delta, -\delta)$
  (M0 × λ, M2 / λ). Labels = exact call from $S'$. DAS (src design, $k$ 1, 2) for these labels at S and at h4; paired bootstrap
  IIA(h4) − IIA(S), one-sided p for h4 > S; the same with the input do(D) labels (pT × λ, E25w).

## Results — Arms 1 and 2 (design don; trained effective IIA [95 % CI]; baseline q95; Holm $p_{q95}$)

| stage | 3f M0 k1 | 3f M0 k2 | 3f M2 k1 | 3f M2 k2 | 16f M0 k1 | 16f M0 k2 | 16f M2 k1 | 16f M2 k2 |
|---|---|---|---|---|---|---|---|---|
| a (per particle) | 0.806 [0.774, 0.840]; q95 0.396; 0.024 | 0.821 [0.791, 0.853]; 0.346; 0.024 | 0.529 [0.506, 0.553]; 0.395; 0.024 | 0.560 [0.535, 0.583]; 0.406; 0.024 | 0.858 [0.829, 0.887]; 0.562; 0.024 | 0.865 [0.834, 0.891]; 0.685; 0.024 | 0.723 [0.700, 0.743]; 0.453; 0.024 | 0.718 [0.696, 0.738]; 0.486; 0.024 |
| g (jet, from P) | 0.635 [0.599, 0.676]; 0.295; 0.024 | 0.632; 0.312; 0.024 | 0.125 [0.110, 0.139]; 0.340; 1.0 | 0.123; 0.354; 1.0 | 0.506 [0.464, 0.548]; 0.764; 1.0 | 0.513; 0.764; 1.0 | 0.167 [0.150, 0.186]; 0.397; 1.0 | 0.170; 0.365; 1.0 |
| u (per particle) | 0.822 [0.793, 0.855]; 0.299; 0.024 | 0.835 [0.807, 0.866]; 0.372; 0.024 | 0.554 [0.530, 0.580]; 0.459; 0.024 | 0.623 [0.598, 0.647]; 0.445; 0.024 | 0.742 [0.703, 0.778]; 0.493; 0.024 | 0.751 [0.713, 0.785]; 0.493; 0.024 | 0.457 [0.434, 0.479]; 0.493; 1.0 | 0.553 [0.529, 0.576]; 0.497; 0.024 |
| S (jet) | 0.869 [0.844, 0.896]; 0.337; 0.024 | 0.858 [0.831, 0.884]; 0.383; 0.024 | 0.741 [0.720, 0.762]; 0.481; 0.024 | 0.787 [0.767, 0.806]; 0.520; 0.024 | 0.835 [0.802, 0.865]; 0.474; 0.024 | 0.840 [0.808, 0.869]; 0.480; 0.024 | 0.681 [0.658, 0.703]; 0.569; 0.024 | 0.725 [0.704, 0.747]; 0.595; 0.024 |

- Random $k$-subspaces (97.5 % quantile): ≤ 0.13 at a, u, g; ≤ 0.22 at S; Holm $p_\mathrm{rot}$ = 0.024 (floor) in all 48 trained cells.
- Full swap (don): 1.0 at a, u, S by construction (the donor jet's own activations) except a/M2 0.985 (3f), 0.990 (16f) and S/M2
  0.978 / 0.946 (clip); at g 0.59 (3f M0), 0.03 (3f M2), 0.29 (16f M0), 0.18 (16f M2).
- Design src (E25w reproduction): S M0 k2 3f 0.817, 16f 0.732; S M2 k2 3f 0.727, 16f 0.676 (E25wv: 3f 0.833 / 0.715, 16f
  0.778 / 0.668). g src: 3f M0 0.52–0.53 (Holm 0.024), M2 0.14 (Holm 1.0); 16f M0 0.35 / 0.41 (Holm 0.72 / 0.024), M2 0.17 (1.0).
- Baselines by family at a (median, max of IIA, k 1): 16f M0 untr 0.48 / 0.87, init 0.34 / 0.41, gauss 0.36 / 0.47, perm 0.19 / 0.27
  (one fully random net exceeds the trained value; rank p 0.048); 16f M2 untr 0.35 / 0.82; 3f M0 untr 0.44 / 0.46, others ≤ 0.37.
- Paired stage differences (don, IIA difference [95 % CI], one-sided p for an increase):
  - 3f M2: u − a +0.025 [−0.001, 0.049] (k1, p 0.027), +0.063 [0.037, 0.089] (k2, p 0.001); S − u +0.187 [0.163, 0.211] (k1),
    +0.164 [0.139, 0.188] (k2), p 0.001.
  - 3f M0: u − a +0.016 [−0.019, 0.050] (p 0.19); S − u +0.047 [0.021, 0.073] (k1, p 0.001).
  - 16f M2: u − a −0.265 [−0.291, −0.237] (k1), −0.165 (k2); S − a −0.041 [−0.065, −0.020] (k1), +0.007 [−0.015, 0.029] (k2, p 0.26).
  - 16f M0: u − a −0.115 [−0.152, −0.077]; S − a −0.023 [−0.055, 0.009].
- Per-particle probe (R² on test particles; edit slope): a: $p_{T,i}$ 1.000 / 0.999 (3f), 1.000 / 0.991 (16f); $p_{T,i} r_i^2$
  0.365 / 0.307 (3f), 0.796 / 0.503 (16f). u: $p_{T,i}$ 0.980 / 1.38 (3f), 0.990 / 0.78 (16f); $p_{T,i} r_i^2$ 0.961 / 0.737 (3f),
  0.973 / 0.667 (16f). Jet-sum correlation of read-out change with true $\Delta M_2$: a 0.93 (3f), 0.96 (16f); u 0.98, 0.96.
  (The jet-sum ratio for $M_2$ is not reported: the mean true $\Delta M_2$ over pairs is near zero, so the ratio is unstable.)

## Results — Arm 3 (D)

- S-coordinate calibration: $R^2$ test 0.943 ($q_0$ on log M0, log M2) and 0.743 ($q_2$) in 3f; 0.889, 0.769 in 16f;
  $\cos(q_0, q_2)$ = −0.005 (3f), −0.081 (16f).
- Call changes on test pairs: do$_S$(D) 0.548 (3f), 0.571 (16f); fixed-D control (M0 × λ, M2 / λ in the same coordinates)
  0.495 (3f), 0.478 (16f); input do(D) 0.580, 0.579.

| labels | k | 3f S | 3f h4 | 3f h4 − S [CI], p(h4>S) | 16f S | 16f h4 | 16f h4 − S [CI], p(h4>S) |
|---|---|---|---|---|---|---|---|
| do$_S$(D) | 1 | 0.444 [0.420, 0.468] | 0.516 [0.493, 0.540] | +0.072 [0.049, 0.096], 0.001 | 0.467 [0.443, 0.489] | 0.241 [0.221, 0.261] | −0.226 [−0.254, −0.197], 1.0 |
| do$_S$(D) | 2 | 0.543 [0.518, 0.567] | 0.503 [0.479, 0.528] | −0.040 [−0.063, −0.015], 1.0 | 0.555 [0.532, 0.577] | 0.546 [0.523, 0.570] | −0.009 [−0.033, 0.015], 0.78 |
| input do(D) (pT × λ) | 1 | 0.362 [0.340, 0.386] | 0.451 [0.428, 0.475] | +0.089 [0.064, 0.114], 0.001 | 0.503 [0.480, 0.526] | 0.237 [0.218, 0.258] | −0.266 [−0.294, −0.238], 1.0 |
| input do(D) | 2 | 0.533 [0.510, 0.556] | 0.485 [0.461, 0.510] | −0.048 [−0.071, −0.024], 1.0 | 0.609 [0.586, 0.632] | 0.538 [0.515, 0.561] | −0.071 [−0.097, −0.044], 1.0 |

Random 1–2-dim subspaces at S / h4: q97.5 ≤ 0.09. Full swap (source call): 0.34 (3f), 0.30–0.34 (16f).

## Competing predictions and which are excluded

- Thesis ("M0 present at P via the pT ramps; M2 first above baseline at u or only at S"): the M0 part holds (a: 0.81 / 0.86,
  above q95 in both nets, Holm 0.024). The M2 part is excluded on these pairs: M2 at a is above the baseline q95 in both nets
  (3f 0.53 vs 0.39, 16f 0.72 vs 0.45; Holm 0.024 in all four a cells).
- Alternative 1 ("both M0 and M2 linearly present per particle at P"): consistent with the 16f net (M2 at a 0.72 = its S level,
  S − a +0.007 [−0.015, 0.029] at k 2). For the 3f net, M2 is above baselines at a but rises by +0.21 [0.19, 0.24] (k1) from a to S,
  and its first-layer direction is not a linear $p_{T,i} r_i^2$ read-out (probe R² 0.36): in 3f, part of M2 is built after a.
- Alternative 2 ("neither at g; M0/M2 only after the second pooling"): excluded (both above baselines at a and u). At g, M2 is below
  the baseline q95 in both nets (Holm 1.0), and M0 is above it in 3f only.
- D (thesis: "first appears at h4 in 3f, not in 16f"): holds for a 1-dim subspace (3f +0.072, p 0.001; 16f −0.226); at k = 2 the
  rise is absent in 3f (−0.040) and 16f (−0.009). The fixed-D coordinate edit changes calls almost as often as the D-changing one
  (3f 0.495 vs 0.548; 16f 0.478 vs 0.571), so the call is not a function of D alone at these S coordinates.

## Issues / limits

- At a, the init and gauss baselines share the trained l1 (identical a); they test the downstream reading. Fully random nets
  (untr) exceed the trained M0 IIA at a in 1 of 6 16f cases (max 0.87–0.88) and M2 0.82–0.87: a monotone-in-pT per-particle
  direction exists by construction when pT enters l1 linearly.
- The 1-dim direction at a is whatever per-particle feature reproduces the donor call; that it equals $p_{T,i}$ (or
  $p_{T,i} r_i^2$) is supported only by the probe read-outs, not by a direct comparison of $Q$ with the probe weights (not run).
- Qualification bar lowered from 0.40 to 0.30 after the first 3f baseline pass (no 3f baseline reached 0.40), before baseline DAS.
- One DAS seed per cell; baselines trained with 250 instead of 600 steps (could lower baseline IIA slightly).
- Range clip at a, g, u = empirical fit-sample range (not the quantizer range).
- Holm floor 0.024 = 24 × 1/1001, so p-values at the floor indicate the bootstrap bound, not the size of the margin.

## Explainer entry (site/explain.py, Paper A)

**Question.** The network's mass variable is built from $M_0 = \sum_i p_{T,i}$ and $M_2 = \sum_i p_{T,i} r_i^2$ ($r_i$ = distance to the
stored jet axis). Earlier work found both as 2-dim causal subspaces at the second pooled layer $S$. Where between the input and $S$
do they first appear?

**Background.** JEDI-linear processes each particle separately ($a_i$ = first-layer vector of particle $i$; $u_i$ = third-layer vector)
and sums over particles twice: $P = \frac{1}{64}\sum_i a_i$ and $S = \frac{1}{16}\sum_i u_i$; $g$ is a jet-level context computed from
$P$. Because a sum of per-particle numbers is the only way information leaves the particle level, a per-particle direction $v$ with
$v\cdot a_i \propto p_{T,i}$ would make $v\cdot P \propto M_0$.

**Method.** Distributed alignment search (DAS) with interchange interventions. For a jet $b$ we make an edited copy $b'$ that changes
one variable: $M_0$ by $p_{T,i} \to c\,p_{T,i}$, $\Delta\eta_i,\Delta\phi_i \to c^{-1/2}(\Delta\eta_i, \Delta\phi_i)$ (so $M_2$ is fixed), or
$M_2$ by scaling all angles by $\sqrt{M_{2,s}/M_{2,b}}$. We learn an orthonormal $Q$ ($k = 1$ or 2 columns) and replace, for every
particle, $a_i \to a_i + (a'_i - a_i) Q Q^\top$, then run the exact network. The interchange accuracy (IIA) is the fraction of
call-changing pairs where this reproduces the call of $b'$. The same is done at $g$, $u$ and $S$, and on 32–41 networks without a
trained function (re-initialised or permuted weights, read-out refit).

**Result.** (figure `e25x.png`) Both $M_0$ and $M_2$ swap through one per-particle direction already at $a$, above the 95th
percentile of the non-trained networks (3-feature: 0.81 and 0.53; 16-feature: 0.86 and 0.72). The jet-level context $g$ carries
$M_2$ below the non-trained level. In the 3-feature net the $M_2$ IIA rises to 0.79 at $S$; in the 16-feature net it is already at
its $S$ value at $a$.

**Checks.** Exact forward ($|\Delta\text{logit}| = 0$); 1,000 random subspaces per cell (all far below); paired bootstrap over pairs;
Holm over 24 cells; a per-particle linear read-out gives $R^2 = 1.00$ for $p_{T,i}$ at $a$ and 0.36 (3f) / 0.80 (16f) for
$p_{T,i} r_i^2$.

**Why it matters.** It places the start of the mass chain at the first layer: per-particle precursors of $M_0$ and $M_2$ exist before
any pooling, and in the 3-feature net $M_2$ is completed over the second per-particle layer and the second pooling.

**Conclusion.** (as above)
