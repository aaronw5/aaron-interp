# E22: interchange-intervention accuracy (IIA) of the E11 (b) program on JEDI-linear (claims A5, A6, B2, B3)

Node E22, stage 4, round 3. Script `interp/e22_iia_jedi.py` (job `e22-run`, 1 CPU / 2 GiB, wall 711 s, of which 398 s was data
loading; pod peak memory 2047 MiB, at the request); figure `interp/e22_plot.py`. Results `results/jedi/e22_iia.json`,
`results/jedi/e22_iia.png`, log `logs/e22-run.log`.

## Procedure
- **High-level model H**: the E11 (b) nested program (`e11_fit_a_P.json`, `e11_fit_b_{g,S,h4,h5}.json`, `e11_fit_b_z.json`). It runs
  $\hat P \leftarrow$ physics, then $\hat g \leftarrow \hat P$, $\hat S \leftarrow \hat g$ + physics, $\hat h_4 \leftarrow \hat S$, $\hat h_5 \leftarrow \hat h_4$ and
  $\hat z \leftarrow \hat h_5$. The network's own last layer d7 on $\mathrm{relu}(\hat z)$ makes the call. Variables: the 19 live S formulas,
  54 live h4, 32 h5 and 16 z formulas.
- **Low-level model L**: the network. Interventions at S, h4, h5 or z run the network's own layers from that stage on. This tail
  reproduces `Jedi.run` and `run_jetwise` exactly: max |Δlogit| = 0 on the base jets, on one S-unit swap and on one z-unit swap.
- **Sample**: E11 held-out test jets, `sample(2000, 11)` minus the 1152 jets that are also in `sample(6000, 0)`, leaving N = 8848.
  From these, 5000 random ordered (base, source) pairs with base ≠ source (pair seed 0).
- **Unedited check**: the network's accuracy on these jets is 80.42 %, the same as in E11 on the same sample. This is 0.2 points
  below the 80.6–81.3 % range in the brief; the binomial SE at N = 8848 is 0.42 points. The program's accuracy is 80.72 %. It gives
  the network's call on 94.02 % of the 8848 jets, and on 94.42 % [93.76, 95.04] of the 5000 base jets.
- **Interchange on unit k of stage s**: L gets unit k set to the network's value on the source jet. H gets formula k set to its
  chain value on the source jet, and the downstream formulas are recomputed.
  - IIA = fraction of pairs where L and H make the same argmax call.
  - Effective IIA (primary) = the same fraction on E(k), the pairs where the swap changes the network's call on the base jet.
  - 95 % CI: bootstrap over pairs, 1000 resamples.
- **Baselines**:
  - (i) Permuted alignment. H's variable k is paired with L's unit π(k), where π is a random derangement of the stage's live units
    (200 draws); the score is on E(π(k)). A second form, "fixed-L", keeps L's unit k and E(k) and averages H's other variables j ≠ k
    on the same pairs.
  - (ii) Random directions. 200 random unit directions r in the standardized live-unit space of each stage. The swap replaces the
    component along r: $V' = V_b + \sigma\, r\, (r\cdot(V_s-V_b)/\sigma)$, in L and in H, clipped at 0 for S, h4 and h5.
  - (iii) Probe alignment. Formula k is replaced in H, for both base and source, by a ridge probe from the 380 finite Q observables.
    The probe is fitted on the E11 fitting jets (20000 fit, 10000 dev for λ). Formula and probe are compared with a McNemar test on E(k).
  - (iv) Untrained network (`Jedi(untrained=True, seed=0)`) with its E11 control fits: single-stage fits on physics, then the
    untrained network's own downstream layers. The trained network gets the same construction with its E11 (a) fits.
- **Joint interventions**: all live units of a stage; S and z together; random subsets over 200 draws (each live S and z unit with
  p = 0.5, and each live unit of all four stages with p = 0.25). The subsets are compared against H swapping a different random
  subset of the same size per stage.
- **Tests**: H0: effective IIA(k) = permuted-alignment IIA; H1: greater.
  - One-sided permutation p over the 200 draws, floor 1/201.
  - Paired p for IIA(k) − fixed-L null on the same pairs (normal approximation).
  - Holm over the 118 per-unit variables with |E| ≥ 20 (S 19, h4 51, h5 32, z 16).

## Numbers

### Per stage, mean effective IIA over live units

| stage | program (unit-aligned) | permuted (95 % of draws) | random direction (95 % of draws) | probe-aligned | formula on physics, network downstream (trained (a)) |
|---|---|---|---|---|---|
| S | 0.544 | 0.240 [0.219, 0.267] | 0.580 [0.419, 0.728], p_rot 0.70 | 0.533 | 0.704 |
| h4 | 0.529 | 0.341 [0.327, 0.357] | 0.517 [0.375, 0.670], p_rot 0.43 | 0.518 | 0.607 |
| h5 | 0.557 | 0.288 [0.271, 0.306] | 0.486 [0.351, 0.621], p_rot 0.14 | 0.533 | 0.617 |
| z | 0.661 | 0.220 [0.192, 0.253] | 0.445 [0.263, 0.651], p_rot 0.025 | 0.576 | 0.592 |

- The stage-level permutation p is 0.005 (the floor) for every stage.
- The per-unit effective IIA ranges are: S 0.334–0.764, h4 0.187–0.828, h5 0.252–0.828, z 0.485–0.873.
- On pairs where the swap leaves the network's call unchanged, IIA is 0.87–0.96 per unit, the level of the base agreement.

### Named units

The thesis predicted an effective IIA of at least 0.8 for each of these units.

| unit | E1 cost | IIA (all pairs) | E(k) | effective IIA [95 % CI] | permuted (95 % of draws) | fixed-L null [CI] | probe effective IIA | McNemar formula / probe correct-only |
|---|---|---|---|---|---|---|---|---|
| S0 | 12.25 % | 0.911 | 983 | 0.764 [0.738, 0.790] | 0.246 [0.098, 0.442] | 0.118 [0.105, 0.131] | 0.727 | 53 / 17, p 1.9e-5 |
| S1 | 20.61 % | 0.818 | 1142 | 0.384 [0.356, 0.414] | 0.247 [0.079, 0.517] | 0.114 [0.102, 0.126] | 0.371 | 48 / 33, p 0.12 |
| S29 | 40.78 % | 0.778 | 2009 | 0.545 [0.523, 0.567] | 0.286 [0.110, 0.425] | 0.073 [0.066, 0.079] | 0.543 | 52 / 48, p 0.76 |
| S50 | 12.02 % | 0.841 | 1004 | 0.743 [0.715, 0.769] | 0.341 [0.098, 0.626] | 0.143 [0.130, 0.156] | 0.749 | 14 / 20, p 0.39 |
| S19 | 11.52 % | 0.922 | 816 | 0.756 [0.727, 0.783] | 0.240 [0.043, 0.465] | 0.143 | 0.748 | 19 / 12, p 0.28 |
| z0 | 15.10 % | 0.922 | 567 | 0.746 [0.711, 0.783] | 0.216 [0.052, 0.388] | 0.110 [0.089, 0.132] | 0.638 | 84 / 23, p 2.4e-9 |
| z1 | 21.63 % | 0.934 | 1093 | 0.873 [0.854, 0.891] | 0.215 [0.080, 0.607] | 0.072 | 0.809 | 94 / 24, p 5.6e-11 |
| z7 | 3.42 % | 0.924 | 352 | 0.682 [0.631, 0.727] | 0.211 [0.059, 0.425] | 0.129 [0.104, 0.154] | 0.318 | 130 / 2, p 3.2e-36 |
| h4 49 | 19.69 % | 0.829 | 1519 | 0.607 [0.583, 0.633] | 0.358 [0.174, 0.623] | 0.059 | 0.588 | 72 / 43, p 0.009 |
| h5 26 | 16.12 % | 0.862 | 856 | 0.415 [0.382, 0.449] | 0.319 [0.140, 0.514] | 0.089 | 0.383 | 45 / 18, p 9e-4 |

- **Holm (118 variables)**:
  - Permutation test: with 200 draws the smallest p is 1/201 = 0.005, so the smallest Holm-adjusted p is 0.587. No variable passes.
  - Paired test against the fixed-L null: 94 of 118 have Holm p < 0.05 (S 17/19, h4 34/51, h5 27/32, z 16/16). S20 and S51 do
    not pass.
- **Units whose effective IIA lies inside the 95 % range of their permuted-alignment draws**: S1, S40, S43, S51 (S: 4/19);
  40/51 h4; 10/32 h5 (including h5 26); 0/16 z.
- **Effective IIA against E1 cost**: Spearman 0.24 over the 118 units (within stages: S 0.26, h4 0.12, h5 0.06, z 0.81).
- **Probe (McNemar, Holm within stage)**:
  - Formula above probe: S0; h4 53; h5 25 and 26; z0, z1, z4, z5, z7, z8, z10, z14.
  - Probe above formula: none.
  - Probe test R² for S is 0.63–1.00, close to the formulas'. Formula and probe do not differ for S29 or S50.
- **Formula with the network downstream**: here only the stage's own formulas are substituted, and the network computes the
  rest. Effective IIA is S29 0.811, S1 0.741, S0 0.803, S50 0.800, S19 0.855, against 0.545, 0.384, 0.764, 0.743, 0.756 in the
  full program. At z both versions use only the network's d7, and the (a) fits on physics give 0.592 against 0.661 for the chain fits.

### Joint and whole-graph interventions

| intervention | IIA [95 % CI] | effective IIA | control |
|---|---|---|---|
| all live S, h4, h5 or z (each), or S + z | 0.9394 [0.933, 0.946] (identical for all five) | 0.9395 (E = 4003) | none (see note) |
| random half of S and of z (200 draws) | median 0.847 (0.761–0.917) | mean 0.829 (0.715–0.915), median E 3607 | misaligned subsets 0.615 (0.337–0.844); aligned > misaligned in 200/200 draws, p 0.005 |
| random quarter of S, h4, h5 and z (200 draws) | median 0.807 (0.709–0.880) | mean 0.775 (0.650–0.883), median E 3389 | misaligned 0.544 (0.281–0.744); 200/200, p 0.005 |
| no intervention | 0.9442 [0.9376, 0.9504] | — | — |

Note: swapping all live units of any stage makes L equal to the network on the source jet and H equal to the program on the
source jet. Its IIA is therefore the same-call rate on the source jets, by construction, and it is not an independent test.

### Untrained control
The untrained network calls class t for all 8848 jets (accuracy 19.9 %). No swap changes its call, so E is empty for every
variable and this control gives no effective pairs.

## Reading (scoped to these measurements)
- **Above the permuted alignment.** For a one-unit swap, the program gives the network's new call in 33–87 % of the pairs where
  the call changes. The mean over a stage's units is 0.53–0.66. With misaligned variables the figure is 0.18–0.41 per unit and
  0.22–0.34 per stage.
- **Thesis prediction (effective IIA ≥ 0.8) not met for any of S0, S29, S1, z0, z7.** S0 comes closest: 0.764, with CI upper
  bound 0.790. Among the costliest units:
  - S1 (cost 20.6 %) is at 0.384, inside its permuted-draw range.
  - S29 (cost 40.8 %) is at 0.545.
  - The paired test separates both from the fixed-L null.
- **Where the shortfall is measured.** With the same S variables, putting the network's layers downstream in place of the h4/h5/z
  formulas raises the effective IIA of S29 from 0.545 to 0.811 and of S1 from 0.384 to 0.741. For these two units, most of the
  shortfall therefore lies in how the downstream formulas respond to S values combined from two jets, not in the S formula itself
  (R² 0.99 and 0.97).
- **Whole graph.** Under random multi-unit swaps across stages the effective IIA is 0.78–0.83, below the 0.944 agreement without
  intervention and above misaligned swaps (0.54–0.62).
- **Basis.** For S, h4 and h5, random directions in the live-unit space give the same mean effective IIA as the unit basis
  (p_rot 0.14–0.70). For z the unit basis gives more: 0.661 against 0.445 [0.263, 0.651], p 0.025.
- **Probe contrast.** Replacing a formula by a linear probe on 380 observables lowers the effective IIA for 8/16 z units, most for
  z7 (0.682 → 0.318), and for S0. It does not change the effective IIA of the mass-family S units S29 and S50.
  - So the expected pattern "probe lower for the mass-family units" holds at z7 (a Z window unit), not at S29 or S50.
- **Untrained control.** It is degenerate: the untrained network calls t for every jet.

## Predictions vs outcome
- **Thesis**:
  - Effective IIA ≥ 0.8 for S0, S29, S1, z0, z7: not observed (0.764, 0.545, 0.384, 0.746, 0.682).
  - Whole-graph IIA close to 94 %: the all-stage swap gives 0.939, but this equals the source same-call rate by construction; the
    random-subset whole-graph IIA is 0.78–0.83.
  - Permuted baseline near the base rate: observed (0.18–0.41).
  - Random directions near the base rate: not observed for S, h4 and h5, which are at the unit-aligned level.
  - Probe lower for the mass-family units: observed for z7, not for S29 or S50.
- **Alternative** (effective IIA within the permuted CI; whole-graph IIA well below 94 %):
  - Excluded for all 16 z units and 15/19 S units (permuted-draw range).
  - Not excluded for S1, S40, S43, S51 and for 40/51 h4 units.
  - Whole-graph random-subset IIA (0.78–0.83) is below 0.944 and above misaligned (0.54–0.62).
- **Statistics caveat**: the per-variable permutation p cannot pass Holm at 118 tests with 200 draws (floor 0.005, adjusted 0.587).
  The paired test passes for 94/118.

## Issues
- The joint per-stage swaps (all S, all h4, all h5, all z, S + z) are degenerate by construction (see the note under the joint
  table); the random-subset swaps are the informative whole-graph test.
- The untrained control has no effective pairs, because the untrained network calls t for every jet.
- The probe R² column labelled "formula_r2_test_chain" in the JSON is the raw pre-ReLU R² for z, which is low (0.03 for z7).
  E11's post-ReLU chain R² is 0.92–0.99 for z0, z1 and z7 (`e11_mars_all.json`).
- The figure on the volume (`/j-jepa-vol/aaron-interp/results/jedi/e22_iia.png`) is the in-job version; the local
  `results/jedi/e22_iia.png` is the redrawn version from `interp/e22_plot.py`.
- Memory peaked at 2047 MiB on a 2 GiB request, at the limit. A rerun with more draws should request 3 GiB (mostly the probe fit
  on 30000 jets × 380 observables).

## Explainer entry (for site/explain.py, Paper B / Paper A cross-reference)
```python
E22_EXPLAINER = [
 ('Question', 'When one unit of the network is set to its value on another jet, does the if-statement program, with the same variable swapped, make the same new call?', None),
 ('Background',
  'The program of E11 reproduces 94 % of the network\'s calls on ordinary jets. Agreement on ordinary jets can come from correlations between the program\'s variables and the network\'s units. '
  'An interchange intervention tests the variables one at a time: unit $k$ of the network gets its value on a source jet $s$, the program\'s formula $\\hat V_k$ gets its value on $s$, and the two calls are compared. '
  'IIA is the fraction of pairs where the calls agree; on pairs where the swap does not change the network\'s call, agreement is mostly the 94 % base rate, so the measure is taken on the pairs where it does (effective IIA).', None),
 ('Method',
  'On 8,848 held-out jets, 5,000 random (base, source) pairs. For each of the 121 live units of $S$, $h_4$, $h_5$, $z$: the network with $V_k \\leftarrow V_k(s)$ and the program with $\\hat V_k \\leftarrow \\hat V_k(s)$ and its later formulas recomputed. '
  'Controls: the program\'s variable $k$ paired with another unit $\\pi(k)$ (200 random derangements); swaps along 200 random directions $r$, $V\' = V_b + \\sigma r\\,(r\\cdot(V_s-V_b)/\\sigma)$; a linear probe on 380 observables in place of the formula; the untrained network. Holm over 118 variables.', None),
 ('Result',
  'Effective IIA per unit is 0.33–0.76 for $S$ and 0.49–0.87 for $z$ (stage means 0.54 and 0.66), against 0.24 and 0.22 for the permuted alignment. '
  '$S_0$ 0.76 [0.74, 0.79], $S_{29}$ 0.55, $S_1$ 0.38, $z_0$ 0.75, $z_7$ 0.68. Random multi-unit swaps across all stages give 0.78 (misaligned 0.54).', 'results/jedi/e22_iia.png'),
 ('Checks',
  'The network tail used for the swaps reproduces the network exactly. With the network\'s own layers after $S$ in place of the $h_4$–$z$ formulas, $S_{29}$ rises to 0.81 and $S_1$ to 0.74: the shortfall is in the downstream formulas\' response to swapped values. '
  'Random directions in $S$, $h_4$, $h_5$ give the same IIA as the unit basis; in $z$ less (0.45 vs 0.66). The probe gives lower IIA than the formula for 8 of 16 $z$ units ($z_7$: 0.32 vs 0.68), not for $S_{29}$ or $S_{50}$. The untrained network calls one class for every jet and gives no effective pairs.', None),
 ('Why it matters', 'The 94 % same-call figure measures agreement on ordinary jets; under single-unit swaps the program gives the network\'s new call for about half to two thirds of the changed calls, above misaligned variables, and least for $S_1$ and $S_{29}$, two of the costliest objects.', None),
]
```

## Addendum (round-3 re-dispatch, 2026-10-17)
- The E22 job had already completed (`logs/e22-run.log`, exit 0, wall 711 s) and the E22v skeptic rerun exists (`inbox/E22_verify.md`). No new cluster job was submitted; results above are unchanged.
- Corrections carried from E22v:
  - The "formula with the network downstream" arm (`controls.trained_a`, figure bar "trained (a)") uses the single-stage (a) physics fits, not the program's (b) chain formulas. With the (b) formulas (E22v `fnet`) the numbers are close: S29 0.826, S1 0.742 (seed 91).
  - The one-unit effective IIA range over all stages is 0.19–0.87 (h4 30 at 0.187); 0.33–0.87 holds for S and z only.
  - A swap-free predictor (the program's runner-up class on the base jet) scores 0.50–0.54 per stage on the same effective pairs. The program exceeds it at S (+0.063 [0.027, 0.099]) and z (+0.117 [0.070, 0.161]), not at h4 or h5; per unit after Holm, 16/119 units are higher (incl. S0, S19, S29, S50, z0, z1, z7, h4 49), S1 and h5 26 are not.
