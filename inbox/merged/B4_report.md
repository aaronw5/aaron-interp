# B4: baselines for the nested if-statement program, fidelity vs description length, edit agreement (claims B2, B3, B4)

Script: `interp/b4_baselines_jedi.py` (modes `mars K`, `direct`, `trees`, `eval`, `plot`).
Jobs (each 1 CPU / 2 GiB, exit 0): b4-mars2, b4-mars4, b4-mars8, b4-mars16, b4-mars30, b4-direct, b4-trees, b4-eval. Logs are in `logs/b4-*.log`.
Results: `results/jedi/b4.json` (all metrics, CIs, McNemar tests), `results/jedi/b4_pred_*.npz/json` (per-jet calls), `results/jedi/b4m{2,4,8,16}_fit_*.json` (refitted nested programs), `results/jedi/b4_direct_terms.json`.
Figure: `results/jedi/b4_fidelity_dl.png`.

## Procedure

- **Splits (same as E11).**
  - Fit: `sample(per_class=6000, seed=0)`, permuted with rng 0. The first 2/3 (20,000 jets) are used for fitting and the last 1/3 (10,000 jets) as dev, for MARS stopping and model selection.
  - Test: `sample(2000, seed=11)` minus the 1,152 jets shared with seed 0, leaving N = 8,848.
  - Confirmation: `sample(2000, seed=91)` minus the 1,481 jets shared with seed 0 or seed 11, leaving N = 8,519.
- **Targets.** The network's call (argmax) and its 5 logits.
- **Inputs.** All physics surrogates use the E11 physics set: 475 inputs (the Q observable library plus 95 hand-built rank-window sums H).
- **Methods.** DL is the number of fitted terms, internal tree nodes, leaves or non-zero weights. Bits = DL × (log2 #inputs + 16).
  - (a) Nested MARS program (E11 b). Refitted with max_terms per unit K ∈ {2, 4, 8, 16, 30}; K = 30 is the stored E11 fit. Every stage is evaluated on the previous stage's surrogate values, end to end from particles to z. The logits come from the network's own d7. DL is the total number of hinge terms over the 144 live units.
  - Reference zphys30 (E11 a): the 16 z neurons fitted directly on physics, then d7.
  - (b) Direct MARS from physics to each logit. Up to 100 terms per logit; prefixes k ∈ {2, …, 100} per logit, each refitted by least squares.
  - (c) Decision tree, depth 2–12. One version is fitted on the call (`tree`) and one on the logits (`treereg`).
  - (d) HistGradientBoosting with leaves ∈ {4, 8, 16, 32} × iterations ∈ {4, …, 128} and learning rate 0.2. One version is fitted on the call (`gbdt`) and one per logit (`gbdtreg`). DL = total leaves.
  - (e) L1 logistic regression on the call (C path) and Lasso per logit (alpha path). DL = non-zero weights.
  - (f) PySR and energyflow are not installed in the job env (checked at run time). PySR per neuron was therefore not run. As the substitute for EFP+ADO, energy-flow polynomials up to degree 3 were computed in numpy from (pT, Δη, Δφ): all 8 connected multigraphs with ≤ 3 edges, β ∈ {1, 2}, plus their logs. An L1 logistic regression was fitted on these 32 features. ADO (sign agreement of the log-softmax differences over 200k jet pairs) is reported for every model in b4.json as `ado_test`.
- **Edits (on the test jets).** Physics inputs were recomputed from the edited particles. On the unedited jets the recomputed inputs match the stored ones (max relative difference 3.0e-9).
  - `trans`: Δη += 0.06 rad for all real particles. Control `trans_c`: each particle moved 0.06 rad in an independent random direction.
  - `rot`: rotation by 45° about the axis. Control `rot_c`: each particle moved 2r·sin 22.5° in a random direction.
  - `A13`: swap of (Δη, Δφ) between pT ranks 13 and 14, in jets with ≥ 15 particles (N 8,800). Control `A13_c`: ranks 13 and 14 each moved by their mutual distance in random directions.
  - `a32x`: one particle in the 9–13 GeV range set from 10 to 12 GeV with its rank kept (crosses the a32 edge at 11 GeV; N 3,452). Control `a32c`: one particle set from 13 to 15 GeV (N 4,121).
- **Metrics.**
  - Same-call %, accuracy and logit R² on test, with 95 % bootstrap CIs over jets (B = 1000; R² uses B = 200).
  - Edit agreement per jet: the surrogate flips if and only if the network flips, and when both flip the new calls match. The reference is a surrogate that never flips, which scores 1 − (network flip rate).
  - Recall: the share of network-flipped jets where the surrogate flips to the same new call.
- **Paired test.** Each nested program K is compared with the baseline that has the highest dev same-call among baselines with DL ≤ DL(K). The baseline is selected on dev, not on test. The test is McNemar on per-jet agreement.
  - H0: nested agreement = baseline agreement. H1: nested is higher.
  - BH correction over 5 K × 4 main edits = **20 cells**.
  - Also reported: the "oracle" baseline, which is the best baseline on that edit's test agreement with DL ≤ DL(K). It is selected on test, so its comparison favours the baseline.

## Network check

- Test: 80.42 % [79.63, 81.22], N 8,848. This is identical to E11 on this split (E11 logs: test 0.8042, fit 0.8058).
- Confirmation: 80.10 % [79.28, 80.95], N 8,519.
- The point estimate is 0.2 points below the requested 80.6–81.3 % range; the CI contains the full-test 80.9 %.

Network flip rates under the edits:

| edit | network flip rate | control | control flip rate |
|---|---|---|---|
| trans | 55.84 % | trans_c | 52.15 % |
| rot | 3.94 % | rot_c | 40.02 % |
| A13 | 1.31 % (115 jets) | A13_c | 7.99 % |
| a32x | 0.98 % (34 jets) | a32c | 0.53 % |

Under trans, the share of true-W jets called Z goes from 2.9 % to 75.4 %.

## Same call vs DL (test; confirmation in brackets after)

| model | DL | bits | same call test | conf | logit R² |
|---|---|---|---|---|---|
| mars2 | 286 | 6,719 | 84.45 [83.67, 85.15] | 83.26 | 0.801 |
| mars4 | 570 | 13,390 | 88.13 [87.44, 88.78] | 87.25 | 0.884 |
| mars8 | 1,123 | 26,381 | 90.45 [89.81, 91.04] | 90.21 | 0.927 |
| mars16 | 2,219 | 52,128 | 92.92 [92.36, 93.42] | 92.52 | 0.954 |
| mars30 | 3,824 | 89,833 | 94.02 [93.50, 94.48] | 93.49 | 0.965 |
| zphys30 | 477 | 11,873 | 92.13 [91.56, 92.72] | 92.05 | 0.945 |
| direct8 / direct16 / direct32 / direct100 | 40 / 80 / 160 / 485 | | 88.48 / 90.72 / 91.79 / 92.91 [92.36, 93.45] | 87.21 / 90.06 / 90.89 / 91.83 | 0.863 / 0.903 / 0.928 / 0.944 |
| tree_d8 (best tree) | 202 | | 89.64 [88.97, 90.29] | 88.51 | |
| gbdt_L16_T16 | 1,280 | 31,861 | 92.88 [92.30, 93.40] | 92.64 | |
| gbdt_L4_T128 | 2,560 | 63,723 | 93.12 | 92.58 | |
| gbdt_L32_T128 (largest) | 20,454 | 509,137 | 93.86 [93.34, 94.35] | 93.27 | |
| gbdtreg_L32_T128 | 20,480 | | 93.64 [93.08, 94.16] | 92.93 | 0.962 |
| l1logit_C1 | 2,114 | | 87.17 [86.44, 87.83] | 87.09 | |
| lasso, best | 1,471 | | 75.24 | 75.36 | 0.706 |
| EFP deg ≤ 3 + L1, best | 160 | | 80.73 [79.87, 81.53] | 79.13 | |

Lowest baseline DL that reaches each nested program's test same-call:

| nested program | direct MARS | GBDT on call | GBDT on logits | tree on call |
|---|---|---|---|---|
| mars2 | 40 | 80 | | 15 |
| mars4 | 40 | 160 | | 61 |
| mars8 | 80 | 320 | | none |
| mars16 | none (max 92.91) | 1,280 (gbdt_L8_T32) | 5,120 | none |
| mars30 (94.02) | none | none (max 93.87) | none (max 93.64) | none |

For mars30, the largest GBDT (93.86 [93.34, 94.35]) has a CI that overlaps mars30's.

## Edit agreement, paired at matched DL (test; McNemar p, BH q over 20 cells)

Nested program minus the dev-selected baseline with DL ≤ nested DL, in agreement points:

| nested (DL) | baseline (DL) | same call diff | trans | A13 | a32x | rot |
|---|---|---|---|---|---|---|
| mars2 (286) | direct32 (160) | −7.35 [−8.11, −6.62] | +2.89, q 0.001 | −0.39, q 0.003 | −0.03, q 1 | +0.17, q 0.0015 |
| mars4 (570) | direct100 (485) | −4.78 [−5.41, −4.09] | +17.67, q 3e-205 | −0.22, q 0.094 | +0.26, q 0.26 | −0.23, q 0.099 |
| mars8 (1,123) | direct100 (485) | −2.46 [−3.01, −1.91] | +20.89, q 2e-265 | −0.55, q 4e-5 | +0.20, q 0.42 | −0.67, q 9e-6 |
| mars16 (2,219) | gbdt_L16_T16 (1,280) | +0.05 [−0.47, +0.57] | +15.03 [13.95, 16.18], q 4e-131 | −0.62, q 8e-6 | −0.41, q 0.064 | −0.92, q 7e-9 |
| mars30 (3,824) | gbdt_L4_T128 (2,560) | +0.90 [+0.41, +1.42] | +15.64 [14.42, 16.83], q 2e-128 | −0.14, q 0.42 | −0.03, q 1 | −0.24, q 0.24 |

The same comparisons on the matched random-direction controls:

| nested | trans_c | rot_c | A13_c | a32c |
|---|---|---|---|---|
| mars2 | −10.92 | −3.40 | −1.69 | −0.32 |
| mars4 | −3.45 | −2.12 | −0.59 | −0.07 |
| mars8 | +2.94 | +0.51 | −0.76 | −0.05 |
| mars16 | +0.49 (p 0.32) | +0.94 (p 0.031) | −0.55 | −0.39 |
| mars30 | +3.81 (p 9e-14) | +4.67 (p 3e-25) | +0.10 | −0.05 |

**Translation against the oracle baseline** (the best baseline on test trans agreement among DL ≤ nested DL):

| nested | nested trans agreement | oracle baseline | oracle trans agreement | McNemar p |
|---|---|---|---|---|
| mars2 | 54.69 | gbdtreg_L8_T4 | 63.07 | 3e-46, oracle higher |
| mars4 | 56.78 | gbdtreg_L16_T4 | 64.32 | 3e-45, oracle higher |
| mars8 | 60.00 | gbdtreg_L32_T4 | 65.57 | 1e-21, oracle higher |
| mars16 | 69.83 [68.87, 70.77] | gbdtreg_L32_T8 | 67.43 | 3.4e-6, nested +2.40 |
| mars30 | 72.39 [71.43, 73.36] | gbdtreg_L32_T16 | 68.93 [67.94, 69.84] | 9.4e-10, nested +3.46 |

On trans_c, mars30 is +3.03 above its oracle (gbdt_L16_T32, 73.35; p 3.1e-9). mars16 is +0.49 above its oracle (gbdt_L16_T16; p 0.32).

**Translation, jets the network flips** (N 4,941). Recall, and the share of true-W jets called Z after the edit:

| model | recall | W → Z |
|---|---|---|
| network | | 75.4 % |
| mars30 | 69.0 % | 77.5 % |
| mars16 | 59.8 % | 86.1 % |
| gbdtreg_L32_T16 | 52.3 % | 76.3 % |
| gbdtreg_L32_T128 | 51.5 % | 76.5 % |
| treereg_d9 | 44.1 % | 91.3 % |
| gbdt_L4_T128 (on call) | 26.3 % | 41.5 % |
| gbdt_L16_T16 (on call) | 23.2 % | 20.9 % |
| direct100 | 23.7 % | 0.0 % |
| zphys30 | 18.9 % | 0.0 % |
| l1logit_C1 | 33.7 % | 0.1 % |
| EFP | 0.0 % | 2.5 % |

These recall values have no CIs. The binomial s.e. at N 4,941 is about 0.7 points.

**Rotation, A13 and a32x (network flip rate 1–4 %).**
- Every surrogate's agreement lies at or below the never-flip reference: 96.06 / 98.69 / 99.02 %.
- Recall on network-flipped jets is ≤ 12 % for every model. mars30: rot 4.6 %, A13 7.8 % (about 9 of 115), a32x 5.9 % (2 of 34).
- On these three edits, the agreement metric measures how rarely a surrogate makes spurious flips, not whether it reproduces the network's flips.

## Reading (scoped to these measurements)

**Same call.**
- At K ≥ 16, the nested program's same-call is within 1 point of the dev-selected baseline at DL ≤ its own:
  - mars16: +0.05 [−0.47, 0.57] vs GBDT on call at DL 1,280.
  - mars30: +0.90 [0.41, 1.42] vs GBDT at DL 2,560.
- At K = 2–8 it is 2.5–7.4 points below direct MARS, which uses DL 160–485.
- Direct MARS and GBDT reach the same-call of mars2/4/8/16 at 4–14× lower DL.
- No baseline in the grid reaches mars30's 94.02 %. The largest GBDT reaches 93.86 % with an overlapping CI, at DL 20,454.

**Translation.**
- mars16 and mars30 agree with the network's call changes more often than any baseline at DL ≤ theirs: +2.4 and +3.5 points over the oracle baseline, and +15 points over the dev-selected GBDT on call.
- They also reproduce more of the flipped calls: recall 60 % and 69 %, against ≤ 52 % for the baselines.
- The physics surrogates fitted directly to z or the logits (zphys30, direct MARS) do not produce W → Z calls under translation (0.0 %). The nested program and GBDT on logits reproduce the network's 75 % to within 1–11 points.
- For mars30, the advantage over the oracle baseline on the matched random-direction displacement (+3.0, p 3e-9) is about as large as on the translation (+3.5). The advantage is therefore not specific to the coherent translation at this K.
- For mars16, the translation advantage (+2.4, p 3e-6) is present and the control advantage (+0.5, p 0.32) is not.

**Rank swap, rotation, a32 crossing.**
- No higher agreement for the nested program. It is lower by 0.2–0.9 points for A13 and rot at K = 8, 16 (q ≤ 1e-5) and equal at K = 30 (q 0.24–0.42).
- The network changes 1–4 % of calls under these edits, and all surrogates miss ≥ 88 % of those changes.

**Predictions.**
- Thesis prediction:
  - (i) Same-call within 2 points at equal DL: observed for K = 16, 30; not observed for K = 2–8.
  - (ii) Higher edit agreement for translation (p < 0.01): observed for K = 16, 30 against all baselines. For K = 30 the matched-control advantage is of similar size.
  - (iii) Higher agreement for the rank swap: not observed.
- Alternative prediction:
  - Same or higher same-call at lower DL: observed for K ≤ 16 (direct MARS, GBDT).
  - Equal edit agreement: not observed for translation at K = 16, 30; observed for A13, a32x and rot, where all methods sit at the never-flip floor.
- Neither prediction is excluded in full.

## Issues / limits

- **PySR / energyflow** are not in the job env. PySR per neuron was not run; the EFP numpy substitute is used.
- **DL units differ** across families (a MARS hinge term, a tree split, a GBDT leaf, a linear weight). The bits column applies the same per-unit cost to all. The nested program's DL counts terms over 144 chained units.
- **Code provenance.** The mars/direct/trees jobs started at 08:10Z from code synced before the last edit; the final script was synced at 08:49Z, before the eval job. The volume code now equals the local file. The fit-mode code at 08:10Z cannot be re-checked; the eval read all prediction files without error.
- **Memory.** Every job's pod peak was 2,021–2,047 MiB against the 2 GiB request (page cache included). There were no OOM kills.
- **Plot change.** After the run, the figure was re-rendered locally from b4.json with an added GBDT-on-logits line (leaves 32). This was a plot-only change, synced to the volume.
- **Rotation control.** The control (random displacement 2r·sin 22.5°) changes 40 % of calls, against 3.9 % for the rotation, so it is not matched in effect size.

## Explainer entry (for site/explain.py; Paper B tab)

```python
B4_EXPLAINER = [
 ('Question', 'Does the nested if-statement program (E11) describe JEDI-linear better than standard compact surrogates of the same size, both in reproducing the calls and in reproducing how the calls change when the input jet is edited?', None),
 ('Background',
  'Description length (DL) counts the fitted pieces of a surrogate: hinge terms for MARS, internal nodes for a tree, leaves for boosted trees, non-zero weights for linear models. '
  'Same call is the share of jets where the surrogate\'s argmax equals the network\'s. Edit agreement on a jet is 1 if the surrogate changes its call exactly when the network does, to the same new class: '
  '$A = \\mathbb{1}[f_s(x\') \\ne f_s(x) \\Leftrightarrow f_n(x\') \\ne f_n(x)]\\cdot\\mathbb{1}[f_n(x\') \\ne f_n(x) \\Rightarrow f_s(x\') = f_n(x\')]$. '
  'A surrogate that never changes its call scores $1 - $ (network flip rate).', None),
 ('Method',
  'Fit on 20,000 jets (seed 0), select on 10,000, test on 8,848 jets (seed 11) and confirm on 8,519 (seed 91). Surrogates: the nested program with 2–30 terms per unit, MARS from 475 physics inputs straight to the 5 logits, '
  'decision trees, gradient-boosted trees (on the call and on the logits), L1 logistic and Lasso, and energy-flow polynomials up to degree 3 with L1 logistic. '
  'Edits: shift every particle by 0.06 rad, rotate by 45°, swap the angles of pT ranks 13 and 14, move one particle from 10 to 12 GeV; each with a random-direction or 13→15 GeV control. '
  'Each nested program is compared with the best baseline of equal or smaller DL (McNemar on per-jet agreement, BH over 20 cells).', None),
 ('Result',
  'At 16 and 30 terms per unit the program matches the best baseline on same call (92.9 % vs 92.9 %; 94.0 % vs 93.1 %), but boosted trees and direct MARS reach the smaller programs\' same call with 4–14 times fewer pieces. '
  'Under the 0.06 rad shift the network changes 55.8 % of calls (75 % of W jets become Z). The 30-term program reproduces 69 % of these changes and agrees on 72.4 % of jets, against 68.9 % for the best boosted-tree model at smaller DL ($p = 9\\times10^{-10}$); '
  'MARS fitted straight from physics to the logits produces no W → Z changes. Under rotation, rank swap and the 2 GeV crossing the network changes only 1–4 % of calls and no surrogate reproduces more than 12 % of them.', 'results/jedi/b4_fidelity_dl.png'),
 ('Checks',
  'The network gives 80.42 % on the test jets, as in E11. Recomputing the physics inputs on unedited jets reproduces the stored ones to $3\\times10^{-9}$. '
  'On the random-direction control the 30-term program also leads the best baseline by 3.0 points, about as much as on the shift; at 16 terms the shift lead is 2.4 points and the control lead 0.5 (not significant). PySR and energyflow are not installed in the cluster env.', None),
 ('Why it matters',
  'The staged program is not the most compact way to reproduce the calls, but at its larger sizes it reproduces more of the network\'s response to the mass-shifting translation than any baseline of equal size. '
  'For the edits that change few calls, the measurement does not separate the surrogates.', None),
]
# register in PAPER_EXPLAINERS['b']:
#   ('Result explained: the if-statement program vs standard surrogates of the same size (claims B2-B4; B4)', B4_EXPLAINER, False)
```
