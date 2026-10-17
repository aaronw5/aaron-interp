# E22v: skeptic rerun of E22 (interchange-intervention accuracy of the E11 (b) program)

Script `interp/e22_verify.py` (job `e22-verify`, 1 CPU / 2 GiB, wall 251 s, pod peak memory 2047 MiB at the request, exit 0;
log `logs/e22-verify.log`). Results `results/jedi/e22_verify.json` (also on the volume). Figure `results/jedi/e22_verify.png`
(`interp/e22_verify_plot.py`, drawn locally from the JSON).

## 1. Code review of `interp/e22_iia_jedi.py`
- **Stage and tail.** The tail (`make_tail`) applies d4 → d5 → d6 → d7 with the network's own input quantizers. It replaces a whole
  stage at the entry point. E22 checked it against `Jedi.run` and `run_jetwise` (max |Δlogit| 0). The rerun builds the tail
  independently and gets max |Δlogit| = 0 against the network at S, h4, h5 and z, and against `run_jetwise` for one S-unit swap.
- **Padded slots, masks, quantizers.** The swaps act at S and later stages, after pooling, so padded slots do not enter. The
  S/h4/h5/z quantizers are the network's own.
- **Split.** `sample(2000, 11)` minus `sample(6000, 0)`. The E11 fits and the probe use `sample(6000, 0)`, so there is no
  fit/test overlap.
- **Edits.** Each per-unit L swap changes one column of one stage. Each H swap changes one formula value, and the downstream
  formulas are recomputed. No edit changes more than stated.
- **Joint swaps.** The all-units-of-a-stage swaps are degenerate. E22 says so.
- **Mislabelled arm (substantive).** The "formula with network downstream" arm (`controls.trained_a`) uses the single-stage
  **(a)** fits on physics, not the program's **(b)** chain formulas. The report describes this arm as "with the same S
  variables" and "keeping the S formula", but the formulas it uses are not the program's. The figure bar "formula, network
  downstream" has the same issue, and at z it compares (a) fits with the program's (b) fits. The rerun adds the arm with the
  program's own (b) formulas (below). The conclusion holds with that arm.
- **Inconsistent range.** The reading gives "33–87 % of effective pairs" for one-unit swaps. The per-unit minima are h4 0.187
  (h4 30) and h5 0.252, and the figure title gives "19–87 %". The 33–87 % range holds for S and z only.
- **Effective set E.** E is defined relative to the network's base call. On bases where the program's base call already
  differs from the network's, H counts as agreeing when the network moves to the program's unchanged call. The report does not
  separate these pairs (quantified below).

## 2. Rerun on a fresh sample
- **Sample.** `sample(2000, 91)` minus the jets of `sample(6000, 0)` and `sample(2000, 11)`, N = 8519 jets. 5000 ordered
  (base, source) pairs, pair seed 1.
- **Baseline accuracy.** Network 80.10 %; program 80.02 %. The program gives the network's call on 93.49 % of jets and on 93.54 %
  of bases (E22: 94.02 / 94.42 %).
- **Statistics.** 95 % CIs from 500 bootstrap resamples over pairs. McNemar (exact) on E(k) per unit, Holm over the 119 units
  with |E| ≥ 20.

### Arms (all scored on E(k), the pairs where the network's call changes under the one-unit swap)
| arm | definition |
|---|---|
| prog | E22's arm: program with formula k set to its chain value on the source jet |
| fnet | program's (b) chain formulas at stage s (unit k from the source), the network's layers downstream (new) |
| fnet_a | as fnet with the (a) physics fits (E22's `trained_a`) |
| nform | the network's stage-s vector after L's swap, then the program's formulas downstream (new) |
| ru_prog | swap-free: the runner-up class of the program's logits on the base jet (new) |
| ru_net | swap-free: the runner-up class of the network's logits on the base jet (new; uses L's own logits) |
| src | swap-free: the network's call on the source jet (new) |
| perm | permuted alignment, 200 derangements, as in E22 |

### Reproduction (named units, effective IIA [95 % CI], |E|)
| unit | E22 (seed 11) | E22v (seed 91) |
|---|---|---|
| S0 | 0.764 [0.738, 0.790] (983) | 0.761 [0.735, 0.786] (1006) |
| S1 | 0.384 [0.356, 0.414] (1142) | 0.407 [0.379, 0.437] (1085) |
| S29 | 0.545 [0.523, 0.567] (2009) | 0.572 [0.550, 0.593] (1919) |
| S50 | 0.743 (1004) | 0.760 [0.735, 0.788] (1004) |
| S19 | 0.756 (816) | 0.764 [0.734, 0.791] (872) |
| z0 | 0.746 [0.711, 0.783] (567) | 0.759 [0.721, 0.795] (544) |
| z1 | 0.873 (1093) | 0.877 [0.858, 0.895] (1106) |
| z7 | 0.682 [0.631, 0.727] (352) | 0.632 [0.579, 0.682] (348) |
| h4 49 | 0.607 (1519) | 0.600 [0.574, 0.622] (1511) |
| h5 26 | 0.415 (856) | 0.426 [0.394, 0.462] (859) |

Stage means (program / permuted):

| stage | E22 | E22v |
|---|---|---|
| S | 0.544 / 0.240 | 0.562 [0.543, 0.581] / 0.259 [0.235, 0.287] |
| h4 | 0.529 / 0.341 | 0.542 [0.511, 0.576] / 0.349 |
| h5 | 0.557 / 0.288 | 0.547 [0.523, 0.573] / 0.291 |
| z | 0.661 / 0.220 | 0.655 [0.630, 0.678] / 0.222 |

- Per-unit effective IIA lies inside the permuted-draw range for S1, S17, S40, S43, S60 (E22: S1, S40, S43, S51), for 29/52 h4
  units (E22: 40/51), 12/32 h5 units (E22: 10/32) and 0/16 z units.
- Random subsets: half of S and z 0.831 (E22 0.829), misaligned 0.604; quarter of all four stages 0.774 (E22 0.775), misaligned
  0.537.

### New control 1: swap-free predictors of the network's new call
- **H0:** the program's effective IIA equals that of a predictor that does not use the swapped value. **H1:** it differs.

Stage means:

| stage | program | ru_prog | program − ru_prog [95 % CI] | ru_net | program − ru_net | src |
|---|---|---|---|---|---|---|
| S | 0.562 | 0.499 | +0.063 [0.027, 0.099] | 0.746 | −0.184 [−0.205, −0.164] | 0.209 |
| h4 | 0.542 | 0.514 | +0.029 [−0.031, 0.095] | 0.897 | −0.354 [−0.385, −0.323] | 0.197 |
| h5 | 0.547 | 0.540 | +0.007 [−0.043, 0.059] | 0.864 | −0.316 [−0.345, −0.289] | 0.249 |
| z | 0.655 | 0.538 | +0.117 [0.070, 0.161] | 0.794 | −0.139 [−0.169, −0.115] | 0.321 |

Per unit, McNemar with Holm over 119 units:

| comparison | program higher | program lower | no difference |
|---|---|---|---|
| vs ru_prog | 16 | 3 | 100 |
| vs ru_net | 8 | 100 | 11 |
| vs src | 103 | 0 | 16 |

Named units against ru_prog (discordant pairs program-only / ru_prog-only, Holm p):

| unit | discordant pairs | Holm p | outcome |
|---|---|---|---|
| S0 | 374 / 106 | 5e-34 | program higher |
| S29 | 682 / 348 | 1e-23 | program higher |
| S50 | 444 / 108 | 3e-47 | program higher |
| S19 | 258 / 102 | 1e-14 | program higher |
| z0 | 235 / 57 | 9e-25 | program higher |
| z1 | 320 / 44 | 9e-51 | program higher |
| z7 | 127 / 65 | 9e-4 | program higher |
| h4 49 | 542 / 188 | 2e-38 | program higher |
| S1 | 263 / 248 | 1.0 | no difference |
| h5 26 | 169 / 187 | 1.0 | no difference |

- For z7, the network's call on the source jet predicts the new call as often as the program does (0.632 vs 0.632, 63/63).
- The E22 permuted null (0.22–0.35) is close to 0.25, the rate for one of the four other classes picked at random.

### New control 2: composition of E
- **Share of E from disagreeing bases.** 17.5 % (S), 27.4 % (h4), 25.3 % (h5) and 18.5 % (z) of pooled effective pairs come from
  bases where the program's base call already differs from the network's.
- **Agreement on those pairs.** It is 0.83–0.88. The program mostly keeps its base call there, and the network moves to it.
- **Pairs where the base calls agree.** Effective IIA, as pooled and as a unit mean:

  | stage | pooled | unit mean |
  |---|---|---|
  | S | 0.54 | 0.447 |
  | h4 | 0.43 | 0.314 |
  | h5 | 0.48 | 0.380 |
  | z | 0.69 | 0.556 |

- **Named units on base-agree pairs.** They change little: S0 0.732, S29 0.560, S1 0.354, S50 0.756, z0 0.737, z7 0.602.
- **Does the program's call change on E?** In 41.5 % (S), 25.2 % (h4), 32.0 % (h5) and 49.0 % (z) of E pairs (unit means).

### New control 3: where the shortfall sits (program's own (b) formulas)
Effective IIA:

| unit | prog | fnet (S formulas, network downstream) | nform (network S, formulas downstream) | fnet_a (E22's arm) |
|---|---|---|---|---|
| S29 | 0.572 | 0.826 | 0.596 | 0.833 |
| S1 | 0.407 | 0.742 | 0.427 | 0.745 |
| S0 | 0.761 | 0.814 | 0.797 | 0.792 |
| S50 | 0.760 | 0.795 | 0.781 | 0.786 |
| h4 49 | 0.600 | 0.860 | 0.607 | 0.862 |
| h5 26 | 0.426 | 0.789 | 0.469 | 0.773 |

- **fnet − prog (S29, S1).** 551 / 63 and 407 / 44 discordant pairs, Holm p 3e-96 and 1e-72.
- **nform − prog (S29, S1).** 159 / 113 and 112 / 91, Holm p ≥ 0.58.
- **Stage means.**

  | stage | fnet − prog | nform − prog |
  |---|---|---|
  | S | +0.150 [0.134, 0.166] | +0.027 [0.009, 0.044] |
  | h4 | +0.087 | −0.005 |
  | h5 | +0.114 | +0.078 [0.051, 0.107] |

  At h5, part of the shortfall is in the h5 formulas' own response to the swap.

## 3. Corrected reading
- **Reproduction.** Single-unit swaps reproduce on a fresh sample (seed 91, 8519 jets, 5000 pairs). The program gives the
  network's changed call in 0.20–0.88 of effective pairs per unit (S 0.37–0.76, h4 0.20–0.73, h5 0.30–0.79, z 0.37–0.88).
  Stage means are 0.54–0.66; permuted alignment gives 0.22–0.35.
- **Swap-free baseline.** The permuted-alignment null is close to the 0.25 rate for one of the four other classes picked at
  random. A predictor that ignores the swap and outputs the program's runner-up class on the base jet reaches 0.50–0.54 per
  stage. The program exceeds it at S (+0.063 [0.027, 0.099]) and z (+0.117 [0.070, 0.161]), not at h4 or h5 (CIs include 0).
  Per unit, after Holm, the program is higher for 16/119 units, including S0, S19, S29, S50, z0, z1, z7 and h4 49. It is not
  distinguishable for 100/119, including S1 and h5 26.
- **Network runner-up baseline.** The network's runner-up class on the base jet predicts the new call in 0.75–0.90 of effective
  pairs, above the program at every stage.
- **Disagreeing bases.** 18–27 % of effective pairs are bases where the program already disagrees with the network. Without them
  the stage-mean effective IIA is 0.45 (S), 0.31 (h4), 0.38 (h5) and 0.56 (z).
- **Where the shortfall sits.** The localisation of the S29/S1 shortfall in the downstream formulas holds with the program's
  own (b) S formulas. With the network downstream: S29 0.826, S1 0.742. With the network's S and the formulas downstream: 0.596,
  0.427. The E22 report reached the same result with the (a) physics fits.
- **Whole graph.** Random multi-stage swaps give 0.77–0.83 (misaligned 0.54–0.60). The network's call on the source jet gives
  0.71–0.75, and the program is higher in 86–88 % of draws.

## 4. Figure check
- `results/jedi/e22_iia.png` is readable.
- The title gives "19–87 %" but the report text gives "33–87 %".
- The green bar "formula, network downstream" is the (a) physics fits, and the label does not say so. At z it differs from the
  program only because the (a) and (b) fits differ.
- The grey band shows the range of the stage-mean permuted nulls. Per-unit permuted ranges reach 0.5–0.6 (e.g. S50
  [0.10, 0.59]).
- The figure has no swap-free baseline.
- `results/jedi/e22_verify.png` adds the swap-free predictors and the fnet arm per stage.

## 5. Issues
- Memory peaked at 2047 MiB on the 2 GiB request in both runs. It is the cgroup peak and likely includes the page cache of the
  memory-mapped test file. The run was not killed.
- Not rerun: random directions, probe alignment, the untrained control (degenerate in E22: calls t for every jet).
- The bootstrap is over pairs, with 500 resamples. Jets recur across pairs (5000 pairs from about 8.5k jets), so the CIs are
  slightly narrow.

## Explainer entry (for site/explain.py; to be merged by the ledger step)
```python
E22V_EXPLAINER = [
 ('Question', 'Does the program\'s agreement with the network after a one-unit swap exceed what a predictor that ignores the swapped value already gets?', None),
 ('Background', 'E22 scored the program on pairs where swapping unit $k$ from a source jet changes the network\'s call (effective IIA) and compared it with a permuted alignment. '
  'With five classes, a changed call goes to one of four others, so a predictor that picks one at random scores about $0.25$, close to the permuted null.', None),
 ('Method', 'Fresh sample (seed 91, 8,519 jets, 5,000 pairs). Same swaps as E22. Swap-free predictors: the runner-up class of the program\'s logits on the base jet, the runner-up of the network\'s logits, the network\'s call on the source jet. '
  'Split arms: the program\'s stage-$s$ formulas with the network\'s layers downstream, and the network\'s stage-$s$ values with the program\'s formulas downstream. McNemar per unit, Holm over 119 units; 95 % bootstrap over pairs.', None),
 ('Result', 'Stage means of effective IIA: program $0.56/0.54/0.55/0.66$ (S/h4/h5/z), program runner-up $0.50/0.51/0.54/0.54$, network runner-up $0.75/0.90/0.86/0.79$. '
  'Program minus program runner-up: S $+0.063\\,[0.027, 0.099]$, z $+0.117\\,[0.070, 0.161]$, h4 and h5 intervals include 0. For $S_{29}$: program 0.57, formulas with the network downstream 0.83, network $S$ with formulas downstream 0.60.', 'results/jedi/e22_verify.png'),
 ('Checks', 'E22\'s per-unit numbers reproduce within 0.05 ($S_0$ 0.761, $S_{29}$ 0.572, $S_1$ 0.407, $z_0$ 0.759, $z_7$ 0.632). 18–27 % of effective pairs are bases where the program already disagrees with the network; without them the stage means are 0.45/0.31/0.38/0.56.', None),
 ('Why it matters', 'The swap-specific agreement of the program with the network is measured at S and z (16 units after Holm, including $S_0$, $S_{29}$, $S_{50}$, $z_0$, $z_7$); at h4 and h5 a swap-free predictor scores the same.', None),
]
```

---

# E22v2: second skeptic rerun (seed 137): rotation test, matched-target probe, jet bootstrap

Script `interp/e22_verify2.py` (job `e22-verify2`, 1 CPU / 2 GiB, wall 1111 s, exit 0, pod peak memory 2048 MiB at the
request, not killed; log `logs/e22-verify2.log`). Results `results/jedi/e22_verify2.json` (also on the volume). The name
`e22-verify` was taken by the E22v job, so this run uses `e22-verify2`.

## 1. Code review (additions to section 1)
- Re-read `interp/e22_iia_jedi.py`. Tail, masks, quantizers, split and per-unit edits are as described in section 1; no new defect
  in the swap code.
- **Rotation test (`rotation.p_rot`).** One one-sided test per stage (4 tests), reported without correction. E22 z: p_rot 0.025,
  Holm over the 4 stages 0.10.
- **Probe target (`probe`).** The ridge probe is fitted to the raw stage value. For z this is the pre-ReLU value. The E11
  formulas for z were fitted to z clipped below at −0.2 sd of its positive part (`jetdistill/mars.py`, `fit_neuron`), and the
  program reads relu(z). The JSON field `formula_r2_test_chain` is the pre-ReLU R² (z7 0.027); the probe's R² is not reported
  post-ReLU. So formula and probe were fitted to different targets and compared on different R² scales.
- **Bootstrap unit.** CIs resample pairs. 5000 pairs are drawn from about 8.5k jets (5782 distinct jets on seed 137), and each
  jet can be base or source in several pairs.

## 2. Rerun
- **Sample.** `sample(2000, 137)` minus the jets of `sample(6000, 0)`, `sample(2000, 11)` and `sample(2000, 91)`; N = 8149 jets;
  5000 ordered pairs, pair seed 3.
- **Baseline.** Network 80.86 %; program 80.64 %. Same call: 93.86 % of jets, 94.06 % of bases. Tail vs network: max |Δlogit| 0
  (S, h4, h5, z, and one S-unit swap against `run_jetwise`).
- **Statistics.** CIs from 500 resamples, two ways: (a) pairs (as E22); (b) jets, with multinomial weights $w_j$ and pair weight
  $w_{\text{base}}\,w_{\text{source}}$. McNemar (exact) per unit with Holm.

### Reproduction (effective IIA, |E|, CI over pairs / over jets)
| unit | E22 (seed 11) | E22v (seed 91) | E22v2 (seed 137) | CI pairs | CI jets |
|---|---|---|---|---|---|
| S0 | 0.764 | 0.761 | 0.755 (1033) | [0.729, 0.779] | [0.706, 0.796] |
| S1 | 0.384 | 0.407 | 0.400 (1164) | [0.373, 0.429] | [0.350, 0.451] |
| S29 | 0.545 | 0.572 | 0.571 (1872) | [0.549, 0.595] | [0.532, 0.612] |
| S50 | 0.743 | 0.760 | 0.767 (1015) | [0.742, 0.791] | [0.721, 0.809] |
| z0 | 0.746 | 0.759 | 0.771 (560) | [0.738, 0.809] | [0.702, 0.833] |
| z7 | 0.682 | 0.632 | 0.723 (311) | [0.671, 0.772] | [0.630, 0.806] |
| h4 49 | 0.607 | 0.600 | 0.639 (1505) | [0.615, 0.662] | [0.598, 0.679] |
| h5 26 | 0.415 | 0.426 | 0.441 (895) | [0.410, 0.477] | [0.382, 0.501] |

- **Stage means (units with |E| ≥ 20).** S 0.533, h4 0.518, h5 0.538, z 0.662. Per-unit ranges: S 0.35–0.77, h4 0.20–0.75,
  h5 0.25–0.78, z 0.40–0.86. The z minimum differs across samples: 0.49 (seed 11), 0.37 (seed 91), 0.40 (seed 137).
- **CI width.** The jet-bootstrap CI is 1.79× the pair-bootstrap CI (median over 118 units; 5–95 %: 1.64–1.95).

### Swap-free predictor (program runner-up on the base), stage level, jet bootstrap
| stage | program | ru_prog | program − ru_prog, 95 % CI over jets |
|---|---|---|---|
| S | 0.533 | 0.508 | [−0.043, 0.098] |
| h4 | 0.518 | 0.516 | [−0.132, 0.129] |
| h5 | 0.538 | 0.530 | [−0.087, 0.111] |
| z | 0.662 | 0.515 | [0.071, 0.231] |

- **H0:** program effective IIA = ru_prog. At the stage level the jet-bootstrap CI excludes 0 at z only. E22v reported
  S +0.063 [0.027, 0.099] with the pair bootstrap.
- **Per unit.** McNemar, Holm over 118 units: program higher for 17/118 (E22v: 16/119). The named units reproduce: S0 378/79,
  S29 644/339, S50 438/115, z0 265/34, z7 135/56 discordant pairs; S1 283/279 (p 0.90), h5 26 197/191 (p 0.80). McNemar treats
  pairs as independent, so these p-values have the same pair-reuse issue as the pair bootstrap.

### Random directions vs unit basis (1000 directions per stage)
| stage | directions, mean [2.5, 97.5 %] | unit basis | one-sided p (unit > directions) | Holm over 4 stages |
|---|---|---|---|---|
| S | 0.586 [0.408, 0.731] | 0.533 | 0.72 | 1.0 |
| h4 | 0.512 [0.389, 0.653] | 0.518 | 0.45 | 1.0 |
| h5 | 0.505 [0.345, 0.657] | 0.538 | 0.34 | 1.0 |
| z | 0.425 [0.242, 0.642] | 0.662 | 0.019 | 0.076 |

- E22 (200 directions): z p 0.025, Holm over 4 stages 0.10.
- Both samples give unit basis above random directions at z, with p ≈ 0.02 per stage. After Holm over the 4 stages, p is 0.08–0.10.

### Probe alignment with matched target (new control)
- **Arms.** `probe_raw` is E22's probe (target: raw z). `probe_matched` has the formulas' target (z clipped below at
  −0.2 sd(z⁺); S, h4, h5 unchanged because they are post-ReLU). Both are ridge regressions on 380 observables, fitted on
  `sample(6000, 0)`, with λ chosen on a 1/3 dev split.
- **Post-ReLU test R²** (the E11 scale), and the mean effective IIA over units with |E| ≥ 20:

  | stage | formula R² | matched-probe R² | IIA formula | IIA matched | IIA raw |
  |---|---|---|---|---|---|
  | S | 0.54–1.00 | 0.63–1.00 | 0.533 | 0.526 | 0.526 |
  | h4 | 0.74–1.00 | 0.63–0.98 | 0.518 | 0.509 | 0.509 |
  | h5 | 0.65–0.99 | 0.33–0.95 | 0.538 | 0.515 | 0.515 |
  | z | 0.83–0.99 | 0.36–0.89 | 0.662 | 0.572 | 0.586 |

- **z units, R² formula / matched probe, effective IIA formula / matched probe:**

  | unit | R² formula | R² matched probe | IIA formula | IIA matched probe | McNemar |
  |---|---|---|---|---|---|
  | z7 | 0.92 | 0.41 | 0.723 | 0.367 | 113/2 |
  | z0 | 0.95 | 0.44 | 0.771 | 0.530 | 146/11 |
  | z1 | 0.97 | 0.84 | 0.863 | 0.799 | 92/26 |
  | z8 | 0.99 | 0.89 | 0.737 | 0.668 | 42/11 |

- **Holm within stage, formula higher.** raw: S 1, h4 2, h5 5, z 6/16. Matched: S 1 (S0), h4 2, h5 5, z 8/16. The formula is
  lower for no unit.
- **Holm over all 118 units (matched).** 14 units: S0; h4 39, 53; h5 20, 24, 25, 26; z0, 1, 4, 7, 8, 9, 10.
- **Fit-quality confound.** Across the 118 units, the formula − probe IIA gap correlates with the formula − probe post-ReLU R² gap
  (Spearman 0.43, permutation p < 1e-4; z only, 15 units: 0.46, p 0.08).
- **S29 and S50.** R² are matched within 0.01 (0.992 / 0.988, 0.983 / 0.977). Effective IIA: S29 0.571 / 0.557 (73/47, p 0.022),
  S50 0.767 / 0.750 (31/13, p 0.010). Neither passes Holm within the stage.
- **Reading.** The z probe contrast compares the chain formulas (post-ReLU R² 0.83–0.99) with linear probes that fit relu(z) much
  less well (0.36–0.89). It is not a fit-quality-matched test. On matched units (S29, S50) the formula−probe difference is
  0.01–0.02.

## 3. Corrected reading (E22 + E22v + E22v2)
- **Per-unit swaps (three samples).** Seeds 11, 91 and 137 reproduce the per-unit effective IIA within 0.05 (z7 within 0.09):
  S0 0.76, S29 0.55–0.57, S1 0.38–0.41, z0 0.75–0.77, z7 0.63–0.72.
- **Stage means.** 0.52–0.66. Permuted alignment 0.22–0.35.
- **Per-unit ranges.** S 0.33–0.77; z 0.37–0.88 across samples (the 0.49 lower end for z is seed 11 only).
- **CIs.** Pair-bootstrap CIs are about 1.8× narrower than jet-bootstrap CIs.
- **Swap-free baseline.** The program's runner-up class gives 0.51–0.54 per stage. Stage-level difference, jet bootstrap: z
  +0.15 [0.07, 0.23]; S +0.03 [−0.04, 0.10]; h4 and h5 include 0. Per unit (McNemar, Holm, pairs treated as independent): 17/118
  units higher, including S0, S19, S29, S50, z0, z1, z7, h4 49. The network's own runner-up class scores 0.75–0.90, above the
  program at every stage.
- **Unit basis vs random directions.** At z, 0.66 vs 0.43–0.45 on two samples, one-sided p 0.019–0.025. Holm over the 4 stages:
  0.076–0.10. At S, h4 and h5 the unit basis and random directions give the same effective IIA.
- **Probe vs formula.** At z, the formula's effective IIA exceeds the linear probe's for 6–8/16 units (z7 0.72 vs 0.37). The z
  probes have post-ReLU R² 0.36–0.89 against 0.83–0.99 for the formulas, with or without the formulas' clipped target. Across
  units, the IIA gap tracks the R² gap (Spearman 0.43). For S29 and S50 (R² within 0.01) the difference is 0.01–0.02, not
  significant after Holm.
- **Unchanged from E22v.**
  - Shortfall localisation (S formulas with the network downstream: S29 0.83, S1 0.74).
  - h4/h5 equal to the swap-free predictor.
  - Random multi-stage swaps: 0.78–0.83 effective; 0.81–0.85 all-pair IIA vs 0.944 with no intervention.
  - The untrained control is degenerate.

## 4. Figure check (`results/jedi/e22_iia.png`)
- The S0, S19 and S50 labels overlap, and "h449" reads as one number (h4 49).
- The right panel has no error bars. Its legend lists "untrained (a)", but no bar is drawn (E is empty).
- "trained (a)" is the (a) physics-fit arm (see section 1).
- The figure has no swap-free baseline or jet-bootstrap CIs. No new figure is drawn here; the numbers are in `e22_verify2.json`.

## 5. Issues
- Memory peaked at 2048 MiB on the 2 GiB request in all three runs. No run was killed. A rerun should request 3 GiB.
- The pigeonhole jet bootstrap is mildly conservative for pair data. The pair bootstrap is anti-conservative.
- The probe is linear in 380 observables. A probe matched to the formulas' R² (e.g. MARS on observables without the chain) was
  not run.

## Explainer entry (E22v2; for site/explain.py, to be merged by the ledger step)
```python
E22V2_EXPLAINER = [
 ('Question', 'Do the E22 contrasts (unit basis vs random directions, formula vs probe, program vs swap-free predictor) hold on a new sample with corrected statistics?', None),
 ('Background', 'E22 compared the program with three alternatives on pairs where a one-unit swap changes the network\'s call. The pairs reuse jets, and the probe at $z$ was fitted to the raw pre-ReLU value while the formulas were fitted to $\\max(z, -0.2\\,\\sigma_{z^+})$.', None),
 ('Method', 'Fresh sample (seed 137, 8,149 jets, 5,000 pairs). 95 % CIs from a jet bootstrap: weights $w_j\\sim\\mathrm{Mult}$, pair weight $w_b w_s$. 1,000 random directions per stage, Holm over 4 stages. Ridge probes on 380 observables fitted to the raw and to the formulas\' clipped target; post-ReLU $R^2$; McNemar on $E(k)$.', None),
 ('Result', 'Unit basis vs directions at $z$: $0.662$ vs $0.425$, $p = 0.019$, Holm $0.076$. Program minus program runner-up (jet CI): $z$ $[0.071, 0.231]$, $S$ $[-0.043, 0.098]$. Probe at $z_7$: $R^2$ 0.41 vs formula 0.92; effective IIA 0.37 vs 0.72. IIA gap vs $R^2$ gap over 118 units: Spearman 0.43.', None),
 ('Checks', 'Named units reproduce within 0.05 ($S_0$ 0.755, $S_{29}$ 0.571, $S_1$ 0.400, $z_0$ 0.771); $z_7$ 0.723 (0.63–0.68 on the other samples). Jet-bootstrap CIs are 1.8× the pair-bootstrap CIs.', None),
 ('Why it matters', 'At $z$ the program exceeds a swap-free predictor. The $z$ probe contrast is not matched in fit quality, and the unit-vs-direction contrast at $z$ is $p \\approx 0.02$ per stage, 0.08–0.10 after Holm over stages.', None),
]
```
