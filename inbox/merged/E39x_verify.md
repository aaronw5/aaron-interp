# E39x verification: fresh-jet rerun plus the missing class-score and unit-choice controls

**Conclusion.** On new jets the E39x numbers come out the same. A joining map that passes 4 numbers from network A's
second-round object layer S gives network B's decision on 87 % of jets when it is trained on B's classification loss. The new
control shows that a map trained the same way on A's 5 output class scores also gives 87 %. So the 4 shared dimensions are not
shown to carry anything beyond A's class scores; only the full 64-unit map adds a little (about 2 points). The named-unit
advantage replicates, but about half of it comes from which of B's units are kept free, not from A's named units matching B's.
The Z-jet prong-sharing response at S again follows the receiving network, now also separated from the untrained-donor control.

## Terms
- **S**: the 64 second-round object sums, $S=\sum_i u_i/16$. **h4**: the first dense layer after S (64 units).
- **Stitch**: run network A up to S or h4, apply an affine map into B's layer, clip each unit to B's [min, max] on the fit jets,
  then run B's exact fixed-point remaining layers ("tail"). **Same-call %**: share of test jets on which the stitched network's
  class equals B's own class on the unedited jet. **Native agreement**: the same share for unedited A against unedited B.
- **Ridge map / rank-r map (RRR)**: an affine map fitted by least squares (ridge) to reproduce B's layer from A's. The rank-r
  version keeps the r output directions that carry most of the fitted variance: $W_r=WV_rV_r^\top$. It passes r numbers.
- **Task-loss map**: a rank-r map started at the RRR map and trained (Adam, 300 steps) to minimise the cross-entropy of B's frozen
  tail on the true labels; gradients go through a float copy of the tail (agrees with the exact tail on 96.7–100 % of calls).
- **Class-score (logit) donor**: the same maps, but the input is A's 5 output logits instead of A's S. Softmax depends only on
  logit differences, so 5 logits carry 4 class-relevant numbers.
- **Inheritance index** $I=(r_{\rm stitch}-r_B)/(r_A-r_B)$, $r_X$ = network X's mean response to an edit; 1 = like donor A,
  0 = like recipient B, 0.5 = value for a response unrelated to either (the null).
- **Network bootstrap**: resample the networks with replacement (2,000 draws) and recompute the pooled number; intervals reflect
  which networks were trained.

## Code review of `interp/e39x_stitch_rank_jedi.py`, `e39x_plot.py`, `e39_plot.py`
- Stages: maps act on S (after the qs1 pool) and h4; the tails start at d4 / d5 with B's own input quantizer. Tail exactness
  (max |Δlogit| 0) reproduced here for all 8 networks.
- RRR: `W @ Vt[:r].T @ Vt[:r]` with `Vt` from the SVD of the centred fitted values; this is reduced-rank regression of the ridge
  prediction. Correct. The random-map control uses the singular values of $W_r$ (rank r). Correct.
- Task-loss init `U = W Vt_r^T`, `V = Vt_r` reproduces $W_r$. The float-tail activations are taken from the first network for
  all networks; harmless here (all tails are ReLU / linear; agreement check 96.7–100 %).
- Split: maps, clip ranges, task training on fit jets; all metrics on test jets with the 398 (here 357) shared rows removed. No
  leakage found. Untrained donors: `redrawn(seed)` patches `keras.models.load_model`, which both the official (via `Jedi`) and
  the X12 loaders call; untrained accuracy 20.0 % (constant call) confirms the redraw.
- Edits: reused from E39 (`build_edits`); padding untouched for T, D at fixed four-momentum (max relative ΔP4 4e-13), N inserted at
  its pT rank. Nothing found that changes more than intended.
- `e39_plot.py` panel (d): pooled I over eligible pairs of the 7 non-degenerate edit networks with the network bootstrap and the
  0.5 line, as described. The figure renders correctly. Panel (d) of `e39x_rank.png` is consistent with the JSON.
- No coding bug found. The issues are with two comparisons (below), not with the code.

## Procedure of the rerun (job `e39x-verify`, 1 CPU / 2 GiB, 840 s, exit 0)
- Script `interp/e39x_verify.py`; it imports the ridge, RRR, clip, exact tail, float tail, task trainer, network bootstrap and
  inheritance pooling from the E39x / E39 modules unchanged.
- **Networks**: 8 of the 10 (official 3f n64, n32, perm-inv n64, 16f n64, HGQ-r150 s0 / s2, float s0 / s1): every
  non-degenerate edit network plus 16f. n128 and HGQ s1 are left out to stay within 2 GiB. 56 ordered pairs.
- **Jets**: fit `sample(2000, 6941)`, N 10,000; test `sample(2000, 6942)` minus 357 shared rows, N 9,643 (E39x used 4941/4942).
- **Untrained donors**: redraw seed 1 (E39x used seed 0).
- **New control 1 (class scores under B's loss)**: the task-loss training applied to A's 5 logits, as a full 5→64 map
  (`task_logit5`) and at rank 4 (`task_logit4`); also to the untrained A's S at rank 4 and full rank.
  - H0: 4 directions of A's S and A's class scores give the same same-call % under B's loss. H1: S rank 4 > logits.
- **New control 2 (choice of B's units)**: in the subset stitch (only k S units passed, the rest of B's S at its fit mean),
  besides named→named and random→random (E39x), also random A units → B's named units, and A's named units → random B units;
  100 random draws each. Shows whether the named gap comes from A's named units matching B's, or only from which B units are kept.
- Edits T, D, N through the full-rank, untrained-donor and logit-donor stitches, as in E39x (rank-4 edit stitches not rerun).

## Results (network-bootstrap 95 % intervals; 56 pairs)
### Replication of E39x (same-call % with B)
| | E39x (seeds 4941/4942, 72 pairs) | rerun (6941/6942, 56 pairs) |
|---|---|---|
| S ridge r 1 / 2 / 4 / 8 / full | 49.9 / 59.7 / 75.3 / 83.9 / 86.6 | 49.6 / 60.4 / **75.2 [72.2, 77.8]** / 84.0 / 86.7 |
| S logit ridge | 66.0 | 65.6 [63.0, 68.3] |
| S ridge r4 − logit ridge | +9.2 [+6.8, +11.4] | +9.5 [+7.1, +11.7], p 5e-4; Holm: r4 higher in 56/56 pairs |
| S task-loss r 4 | 87.3 | **87.4 [85.0, 89.6]** |
| h4 ridge r 4 / logit ridge | 77.8 / 78.6 | 77.8 / 78.2; difference −0.4 [−2.4, +1.5], p 0.69 |
| native agreement | 89.5 | 89.4 |

### New control 1: A's class scores trained on B's loss
| layer | S rank 4, task | logits 5→64, task | logits rank 4, task | S full, task | untrained S rank 4, task |
|---|---|---|---|---|---|
| S | 87.4 [85.0, 89.6] | **87.0 [84.3, 90.1]** | 87.0 [84.0, 90.2] | 89.2 [87.0, 91.5] | 37.1 [26.7, 47.9] |
| h4 | 88.8 [86.9, 90.8] | 89.2 [86.9, 91.7] | 89.1 [86.9, 91.6] | 89.3 [87.3, 91.3] | 36.9 [27.4, 46.4] |

- **S rank 4 − logits (task loss), S: +0.4 points [−1.3, +1.9], p 0.62.** Per pair, exact McNemar with Holm over 56 pairs:
  S rank 4 higher in 23, logits higher in 16, no difference in 17. Against logits at rank 4: +0.4 [−1.6, +2.5], p 0.66.
- h4: −0.4 [−1.1, +0.2], p 0.25.
- Stitched accuracy, S: task S rank 4 78.0 %, task logits 78.5 %.
- **S full rank − logits (task loss): +2.2 points [+0.9, +3.4], p 0.0015** at S; +0.0 [−0.5, +0.6] at h4.
- Untrained A's S under the same training: 37 % same-call (accuracy 35 %), so the task-loss training does not reach high
  agreement from any input. Caveat: the untrained redraws give a constant call, and their S may be close to degenerate.
- **Competing hypotheses.** "Content = class scores" predicts S rank 4 ≈ logits under the same map family; "content beyond class
  scores in 4 directions" predicts S rank 4 > logits. Under B's loss the first is not excluded (difference +0.4 points, upper
  bound +1.9) and the second is not supported. The E39x exclusion of "content = class scores" at S (+9.2 points) holds only for
  maps fitted to copy B's layer: a least-squares map from 5 logits copies B's S less well than a map from 4 directions of A's S,
  while giving B's tail the same decision when trained for it.

### New control 2: subset stitch with the choice of B's units separated (S; 54 pairs, k = 1, 3, 4)
| condition (mean same-call %) | all pairs | k = 4 (6 pairs) | k = 3 (24 pairs) |
|---|---|---|---|
| A's named → B's named | 30.1 | 42.2 | 37.1 |
| random A → B's named | 26.5 | 35.6 | 31.1 |
| A's named → random B | 24.9 | 28.0 | 29.3 |
| random A → random B (E39x control) | 23.3 | 25.6 | 26.5 |
| B's own named units, no stitch | 33.7 | 44.9 | 42.8 |

- Named − random A/B: +6.7 [+1.8, +12.9], p 0.0055 (E39x +6.6, p 0.0045): replicated.
- Of that gap, keeping B's named units free with random A inputs gives +3.1 [+0.5, +6.3], p 0.012; A's named units over random A
  units into the same B named units give **+3.6 [+0.8, +7.1], p 0.012**.
- k = 1 pairs (24, into float s0 / s1 single-role units) are at the all-at-mean value (~20 %) in every condition.

### Inheritance index (7 edit networks, 40–42 pairs; null 0.5)
| cell | E39x | rerun |
|---|---|---|
| S DZ full | 0.04 [−0.36, 0.22], p 0.011 | **0.06 [−0.29, 0.26], p 0.012** |
| S DZ untrained | 0.34 [0.14, 0.64] | 0.40 [0.18, 0.70] |
| S DZ full − untrained | −0.30 [−0.96, +0.05], p 0.066 | **−0.35 [−0.83, −0.10], p 0.018** |
| S DW full | 0.95 [0.50, 1.56], p 0.053 | 0.94 [0.35, 1.82], p 0.16 |
| S T full | 0.36 [0.03, 0.89] | 0.38 [0.10, 0.81] |
| S N full / untrained | 0.68 / 0.86 | 0.69 [0.42, 0.82] / 0.98 [0.70, 1.32] |
| h4 DW full − untrained | −0.39 [−0.98, −0.26] | −0.33 [−1.11, −0.14], p 0.018 |

- Uncorrected over cells; the S DZ full cell would survive Holm over the 8 full-stitch cells only at p ≤ 0.00625 (it does not).
- The rank-4 stitch cells (E39x S T 0.85, p 0.021) were not rerun.

Figure: `results/jedi/e39x_verify.png` (panel a: same-call by map input and training; panel b: subset conditions by k).

## Assessment of the E39x reading
- Numbers: all headline numbers replicate within about 0.5 points on fresh jets, a fresh untrained redraw and 8 of the 10 networks.
- **Not supported as stated**: "rank 4 passes more of B's decision than A's 5 logits" and thesis note A1s "the content B's later
  layers use from A's S is low-dimensional ... and is more than A's class scores at S". Under the map family in which the 87 %
  was measured (trained on B's loss), A's logits give 87.0 % against 87.4 %. The +9-point gap exists only between maps fitted to
  copy B's layer, which measures how well B's S is reconstructed, not which information B's decision uses. The low dimension (4)
  equals the number of class-relevant numbers in 5 logits, so it is also what the class-score account predicts.
- **Refined**: the named-unit advantage (+6.6) splits into +3.1 from which B units are kept and +3.6 [+0.8, +7.1] from A's named
  units corresponding to B's.
- **Strengthened**: S DZ recipient-like response; the difference from the untrained donor is now p 0.018 (was 0.066).

## Corrected reading
At S, a rank-4 map trained on B's loss gives B's call on 87 % of test jets (native agreement 89 %), and a map trained the same
way from A's 5 output logits gives the same 87 % (difference +0.4 points [−1.3, +1.9]); a full-rank map from S adds +2.2 points
[+0.9, +3.4] over the logits. Maps fitted to reproduce B's layer need about 8 dimensions to reach 84 %, and at rank 4 they
reproduce B's decision better from A's S than from A's logits (+9.5 points), a difference of reconstruction, not of the
information the decision uses. Passing only the named S role units gives 30 % against 23 % for random units in both networks;
+3.6 points [+0.8, +7.1] of that are due to A's named units matching B's, the rest to which B units are kept. The Z-jet
prong-sharing response at S follows the recipient (I 0.06, p 0.012 against 0.5; −0.35 [−0.83, −0.10] against the untrained
donor); other inheritance cells are not separated from 0.5 with 7 networks.

## Issues
- 8 networks instead of 10 (n128 and HGQ s1 out); no k = 2 subset pairs as a result.
- Untrained donors are degenerate (constant call); the untrained task-loss control bounds what the training does from such
  inputs, not from an arbitrary random-feature network with spread activations.
- Task-loss maps were trained on the true labels (as E39x); training on B's calls was not tried.
- The pod's cgroup memory peak read 2,047 MiB of 2 GiB (includes page cache); the run exited 0.

## Explainer entry (Paper A tab, for site/explain.py)
- **Question**: when two separately trained JEDI-linear networks are joined at S, is what the second network needs from the
  first more than the first network's class scores?
- **Background**: E39x found that 4 directions of A's S, with the map trained on B's loss, give B's decision on 87 % of jets.
  Five logits also carry 4 class-relevant numbers ($\mathrm{softmax}$ depends on logit differences only).
- **Method**: train the same map on B's loss from A's 5 logits ($\ell_A\in\mathbb{R}^5\to S_B$) and compare same-call % with the
  rank-4 map from $S_A$; network bootstrap over 8 networks, McNemar per pair with Holm.
- **Result** (`results/jedi/e39x_verify.png`): 87.4 % from $S_A$ rank 4 and 87.0 % from $\ell_A$; full rank from $S_A$ 89.2 %.
- **Checks**: fresh jets, fresh untrained redraw (37 %), exact tails, float-tail agreement 96.7–100 %.
- **Why it matters**: the stitching result does not by itself show shared intermediate content beyond the decision; the shared
  part beyond class scores is at most about 2 points of same-call at full rank.

Files: `interp/e39x_verify.py`, `interp/e39x_verify_plot.py`, `results/jedi/e39x_verify.json`, `results/jedi/e39x_verify.png`,
`logs/e39x-verify.log`.
