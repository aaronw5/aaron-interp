# E25x verification — first-layer (stage $a$) swaps of $M_0$ and $M_2$ against fully random networks, matched step budget, and an $M_0$/$M_2$-preserving edit

Script `interp/e25x_verify.py` (imports `interp/e25x_p_to_s.py` unchanged: `PNet`, `fit_pp`, `boot_reps`; and the E25w routines
`pair_edits`, `Net`, `das_fit`, `eval_Q`, `refit_d7`). Cluster jobs (all 1 CPU / 2 GiB, exit 0): `e25x-verify` (3f trained),
`e25x-verify-16f` (16f trained), `e25x-verify-u3f` (3f untrained seeds 10–39), `e25x-verify-u16a` / `-u16b` (16f untrained seeds
10–25), smoke `e25x-verify-smoke`. Results: `results/jedi/e25x_verify_{3f,16f}_trained.json`, `e25x_verify_3f_untr10_40.json`,
`e25x_verify_16f_untr10_18.json`, `e25x_verify_16f_untr18_26.json`, `e25x_verify_summary.json`; logs `logs/e25x-verify*.log`.

## Conclusion

On a fresh sample, the one-direction first-layer swap of the pT-weighted angular spread ($M_2 = \sum_i p_{T,i} r_i^2$) works about as
well, or better, in networks with every weight drawn at random as in the trained networks (3-feature: trained 0.48, random-network
median 0.57). In the 16-feature net, the same first-layer swap also reproduces call changes from an edit that keeps $M_0$ and $M_2$
fixed and only moves particles radially, nearly as well as for the $M_2$ edit. So the first-layer result for $M_2$ does not show that
training put an $M_2$ precursor there; it reflects how any network of this shape processes per-particle angles. The total-pT
($M_0$) first-layer swap and the second-pooled-layer ($S$) swaps stay above most random networks, but with 6–15 random networks per
model none of these comparisons reaches p < 0.05 after Holm correction.

## Code review (interp/e25x_p_to_s.py, interp/e25x_plot.py)

- Stages: `PNet` reproduces `Jedi.run` exactly (logged |Δlogit| 0); the intervention at `a` is applied to the l1 output before the
  `qs` pool, at `u` before `qs1`, at `g` after `dd` with the base `s`; the same $Q$ is applied to every slot. Correct as described.
- Padding: edits act only on real slots (`np.where(rl, ...)`); padded slots have identical base and donor activations, so
  $(h_x - h_b)QQ^\top = 0$ there. Padded slots enter the clip range of `a`/`u` (range over all slots), which only widens the clip.
- Split: DAS fit on fit-sample pairs, all reported IIA on test-sample pairs (overlapping rows removed); probe fit on fit jets, scored on
  test pairs; baseline read-outs refit on the fit sample. No leakage found.
- Edits: realised changes on the fresh test pairs: do(M0) |Δ log M2| ≤ 1e-4; do(M2) |Δ log M0| = 0. The do(M0) edit also changes
  every angle by $c^{-1/2}$, $\sum p_T r$ by $c^{1/2}$ and $\sum r$ by $c^{-1/2}$; DAS identifies a subspace that carries the call
  change of this edit family, not $M_0$ to the exclusion of these co-varying quantities.
- Baseline comparison at `a`: 30 of 41 (16f) and 27 of 32 (3f) qualifying baselines are `init`/`gauss`, which keep the trained l1, so
  their `a` activations are identical to the trained net's. The pooled q95 used for the stage-`a` test is dominated by networks whose
  stage-`a` representation is the trained one; only `untr` (and `perm`) vary l1. In the original run, fully random (`untr`) nets
  already exceeded the trained value at `a` for 16f M0 (0.87 > 0.858) and 16f M2 (0.82 > 0.723).
- Step budget: baselines ran 250 steps / minibatch 384, the trained nets 600 / 512 (asymmetric; measured below).
- Figure `results/jedi/e25x.png`: the black q95 markers are the same for M0 and M2 (distinguishable only by a small x-offset); the
  random-subspace line named in the plot docstring is not drawn; the baseline marker pools families that share the trained l1; the
  title is a declarative statement. Lines join a → g → u → S although g is a side branch (P → g), not a step between a and u.

## Procedure (rerun)

- Sample: fit `sample(2000, 2561)` (10,000 jets), test `sample(2000, 2562)` minus 389 rows in the fit sample (9,611 jets); 791 fit
  and 718 test rows also occur in the E25x samples. All 16 columns from `val_c16_f16`, max difference to `sample()` x = 0.
- Pairs: 3,000 train (rng 2563; last 20 % for model selection), 3,000 test (rng 2564).
- Edits on the base jet: do(M0), do(M2) as E25x. Two control edits:
  - **Rr** (radial redistribution): per-particle angle scale $s_i = \kappa\, r_i^{\gamma - 1}$, i.e. $r_i \to \kappa r_i^{\gamma}$,
    $\kappa = \sqrt{M_2 / \sum_i p_{T,i} r_i^{2\gamma}}$, $\log\gamma \sim U(\log 0.5, \log 2)$ per pair. Every $p_{T,i}$, $M_0$ and $M_2$
    unchanged (realised |Δ log M2| ≤ 3e-4); every $p_{T,i} r_i^2$ changes.
  - **Rot**: rotation of $(\Delta\eta, \Delta\phi)$ about the stored axis by $U(-\pi, \pi)$ (every $p_{T,i}$, $r_i$ unchanged).
- Interventions: E25x design "don", $k = 1$, at `a` (600 steps, minibatch 512, 100 random subspaces), `S` (E25w `das_fit`, 600 steps),
  `u` (M2 only); and at `a` with the E25x baseline budget (250 steps, minibatch 384).
- Control networks: `jedi_core` untrained (every kernel, bias, BN scale redrawn) seeds 10–39 (3f) and 10–25 (16f), new seeds not used
  in E25x; d7 refit by logistic regression on the fit sample; qualify at test accuracy > 0.30 and ≥ 3 distinct calls (E25x rule):
  15 of 30 (3f) and 6 of 16 (16f). Each: same pairs and edits, its own labels, `a` and `S` for M0 and M2, $k = 1$, **600 steps /
  minibatch 512 (the trained budget)**.
- H0 per cell: trained effective IIA is inside the fully-random-network distribution; H1: above it. Rank p = (1 + #{untr ≥ trained}) /
  (1 + n_untr); Holm over the 8 cells (2 models × a, S × M0, M2). Also whether the trained 95 % bootstrap CI lies above the untr q95.

## Results

Accuracy (unedited, fresh test): 3f 80.61 % [79.80, 81.38]; 16f 81.64 % [80.85, 82.40]. Call changes on test pairs: 3f M0 0.196,
M2 0.565, Rr 0.086, Rot 0.044; 16f M0 0.190, M2 0.582, Rr 0.087, Rot 0.002 (7 pairs; no IIA).

Replication of the trained cells (effective IIA, k 1, [95 % CI]; E25x value):

| cell | 3f fresh | 3f E25x | 16f fresh | 16f E25x |
|---|---|---|---|---|
| a, M0 | 0.820 [0.788, 0.849] | 0.806 | 0.801 [0.767, 0.834] | 0.858 |
| a, M2 | 0.481 [0.456, 0.503] | 0.529 [0.506, 0.553] | 0.721 [0.701, 0.741] | 0.723 |
| u, M2 | 0.536 [0.514, 0.559] | 0.554 | 0.451 [0.428, 0.474] | 0.457 |
| S, M0 | 0.867 [0.839, 0.894] | 0.869 | 0.821 [0.789, 0.852] | 0.835 |
| S, M2 | 0.764 [0.746, 0.785] | 0.741 | 0.701 [0.679, 0.725] | 0.681 |
| a, M0, 250 steps | 0.803 [0.769, 0.835] | — | 0.726 [0.684, 0.760] | — |
| a, M2, 250 steps | 0.479 [0.455, 0.502] | — | 0.667 [0.646, 0.689] | — |

Random 1-dim subspaces (q97.5): ≤ 0.09 (a, M0/M2), ≤ 0.15 (S). Paired differences: 3f S − a (M2) +0.284 [0.256, 0.311], p 0.001;
3f u − a (M2) +0.055 [0.034, 0.078]; 16f S − a (M2) −0.020 [−0.043, 0.004], p 0.095; 16f u − a (M2) −0.270 [−0.296, −0.245].
Step budget: 250 instead of 600 steps lowers the trained a-cell IIA by 0.075 (16f M0) and 0.054 (16f M2); 3f unchanged (≤ 0.02).

Trained vs fully random networks (fresh untr, same budget; untr values sorted):

| cell | trained | untr (n) median / q95 / max | # untr ≥ trained | rank p | Holm |
|---|---|---|---|---|---|
| 3f a, M0 | 0.820 | 0.40 / 0.72 / 0.77 (15) | 0 | 0.062 | 0.50 |
| 3f a, M2 | 0.481 | 0.57 / 0.75 / 0.77 (15) | 10 | 0.69 | 1.0 |
| 3f S, M0 | 0.867 | 0.40 / 0.69 / 0.71 (15) | 0 | 0.062 | 0.50 |
| 3f S, M2 | 0.764 | 0.47 / 0.72 / 0.91 (15) | 1 | 0.125 | 0.75 |
| 16f a, M0 | 0.801 | 0.38 / 0.63 / 0.64 (6) | 0 | 0.14 | 0.75 |
| 16f a, M2 | 0.721 | 0.78 / 0.86 / 0.86 (6) | 3 | 0.57 | 1.0 |
| 16f S, M0 | 0.821 | 0.42 / 0.51 / 0.51 (6) | 0 | 0.14 | 0.75 |
| 16f S, M2 | 0.701 | 0.47 / 0.76 / 0.82 (6) | 1 | 0.29 | 0.86 |

(E25x's own untr nets, original sample and 250 steps: 16f a M0 0.82 and 0.87, a M2 0.82, above the trained values; pooled with the
fresh ones, 2 / 12 (16f a M0) and 4 / 12 (16f a M2) fully random nets reach the trained value.)

Edit-specificity control (trained nets, k 1):

| cell | 3f | 16f |
|---|---|---|
| a, Rr (M0, M2 fixed) | 0.300 [0.245, 0.359] (n 257; random q97.5 0.16) | 0.681 [0.623, 0.736] (n 260; 0.18) |
| a, M2 − a, Rr | +0.181 [0.118, 0.240], p 0.001 | +0.040 [−0.017, 0.098], p 0.19 |
| S, Rr | 0.576 [0.517, 0.634] | 0.565 [0.504, 0.627] |
| S, M2 − S, Rr | +0.189 [0.126, 0.253], p 0.001 | +0.136 [0.072, 0.196], p 0.001 |
| a, Rot | 0.282 [0.203, 0.366] (n 131; random q97.5 0.42) | n 7 (not evaluated) |

The Rr and M2 IIA are computed on different call-changing subsets (n 257–260 vs 1,694–1,747), so the difference compares edits of
different call-change rate, not matched pairs.

## Competing hypotheses on the fresh sample

- H_train ("training put a one-dimensional $M_2$ precursor into the per-particle first-layer vector"): predicts trained a-M2 IIA above
  fully random nets and above an $M_2$-preserving edit. Observed: 3f trained below the untr median (10 / 15 untr ≥ trained); 16f
  3 / 6 untr ≥ trained; 16f a-direction reproduces the Rr edit (M2 fixed) at 0.68 vs 0.72 for M2 (difference not resolved).
  Not supported in either net.
- H_arch ("any network with this l1 / pooling structure has a per-particle direction at a that carries the call change of an angular
  edit"): predicts untr ≈ trained at a for M2 and comparable IIA for non-M2 angular edits. Consistent with 16f and with the untr data
  in 3f. In 3f, the M2 edit is reproduced better than Rr (+0.18), so the 3f a-direction is not edit-agnostic.
- M0 at a: trained above every fresh untr net in both models (3f 0.82 vs max 0.77; 16f 0.80 vs max 0.64), but not above E25x's two
  16f untr nets (0.82, 0.87), and the rank p cannot go below 0.06 / 0.14 with 15 / 6 nets (Holm 0.50 / 0.75).
- 3f M2 rise from a to S (+0.28 fresh, +0.21 E25x) and 16f M2 flat from a to S (−0.02) and dip at u (−0.27): replicated.

## Corrected reading

The DAS numbers replicate within about 0.05 on a fresh sample (3f a-M2 0.48 vs 0.53; 16f a-M0 0.80 vs 0.86). The stage-a comparison
with "networks without a trained function" does not hold for $M_2$: against fully random networks evaluated with the same step
budget, the trained a-M2 IIA is below the median in 3f (0.48 vs 0.57) and inside the distribution in 16f (0.72 vs median 0.78). The
E25x q95 used for this test is set mainly by `init`/`gauss` nets that keep the trained l1 (identical a) and were trained 250 instead of
600 steps. In 16f the a-direction also reproduces calls under an $M_0$- and $M_2$-preserving radial edit (0.68). The statements
"M2 first appears at u or S is excluded" and "Alternative 1 (both present at P) fits 16f" are not supported by these data. $M_0$ at a
and $M_0$, $M_2$ at S remain above most fully random nets, with rank p 0.06–0.29 (Holm ≥ 0.5) given 6–15 such nets.

## Explainer entry (site/explain.py, Paper A; check on E25x)

**Question.** Is the first-layer swap of $M_2$ found in E25x a property of the trained network, or of the architecture?

**Background.** E25x replaced, for every particle, one direction of the first-layer vector $a_i$ by its value after an input edit
($a_i \to a_i + (a'_i - a_i)qq^\top$) and found that this reproduces the edited jet's call. The network computes $a_i$ from the
particle's own $(p_T, \Delta\eta, \Delta\phi)$ (or 16 features), so any such network has per-particle activations that move with the
particle's angles.

**Method.** Same swap on a fresh sample, for the trained nets and for 21 qualifying networks with all weights redrawn (read-out refit),
all with 600 optimisation steps; plus a control edit $r_i \to \kappa r_i^\gamma$ that keeps $M_0$ and $M_2 = \sum p_{T,i} r_i^2$
fixed.

**Result.** 3-feature: trained 0.48, random-network median 0.57; 16-feature: trained 0.72, median 0.78. The 16-feature control edit
gives 0.68 at the first layer.

**Checks.** Exact forward; 100 random directions per trained cell (q97.5 ≤ 0.09); paired bootstrap for differences; rank tests with
Holm over 8 cells.

**Why it matters.** The start of the mass chain cannot be placed at the first layer from this test; the second pooled layer remains
where trained $M_2$ swaps exceed most random networks.

**Conclusion.** (as above)
