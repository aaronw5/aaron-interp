# X5: program elements in Deep Sets trained as float, with L1 on the first layer, and with HGQ (claims A1, A3, A10; review weakness 7)

## Conclusion (plain terms)
- **Found:** The same Deep Sets (dsets-d64, n64, 3-feature inputs) was trained three ways, with 3 seeds each. Every regime responds to the 0.06 rad shift; the read-out moves 0.72 to 1.26 times the $\sqrt{M_0M_2}$ prediction. No regime forms pT-rank windows, and no regime forms count-weighted radial objects. Pure pT-only / angle-only first-layer units exceed a weight-permutation null only under HGQ (2 of 3 seeds in each HGQ arm). Even there they are 30–50 % of live units, against 13 of 13 in JEDI-linear. With L1 at JEDI-linear's sparsity, 100 % of live units are pure, but permuting the weights at that sparsity gives the same purity.
- **Means:** In this architecture the shift response comes from the task and inputs, not from fixed-point training. The purity of a sparse first layer (L1 arm) is what random placement gives at that sparsity. HGQ training of 10 epochs did not reproduce JEDI-linear's first-layer sparsity (≤ 0.094 vs 0.896 zero weights) or its pooling slot masks (≤ 1 % zero-bit slots). So this run does not test whether long HGQ training produces rank windows.
- **Uncertain:** HGQ was capped at 10 epochs, and its sparsity was not matched. No regime-label permutation test survives Holm (smallest Holm p = 0.095). 3 models per regime limit the exact test (smallest attainable p = 1/84).

## Procedure
- **Models** (`models/small/`, dsets-d64, N = 64; inputs (pT, Δη, Δφ) normalised with norm_stats.json['64']):
  - **float:** `dsets-d64-n64/seed{0,1,2}`, from train/train_small.py with AdamW, 30 epochs.
  - **L1:** `dsets-d64-n64-l1_3e-2/seed{0,1,2}`, using `--l1_phi1 3e-2`. After training, phi1 weights with |w| < 1e-3 are set to 0 and the model is re-evaluated.
    - λ was chosen on seed 0 so that the phi1 zero fraction matches JEDI-linear's fused l1 kernel (1 − 20/192 = 0.896).
    - The planned grid {1e-5 … 3e-4} reached ≤ 0.063 zero weights, so the grid was extended to 3e-2. λ = 3e-2 gives 0.885 / 0.906 / 0.896 for seeds 0 / 1 / 2.
  - **HGQ:** train/train_hgq_dsets.py (hgq2 0.1.9, keras/jax, CPU), with the same architecture.
    - One 30-epoch run takes more than 1 h (170–250 s per epoch), so all runs were capped at 10 epochs: batch 2048, lr 2e-3, patience 5.
    - β sweep on seed 0, with phi1 zero fraction and test accuracy at each β:

      | β | 1e-8 | 3e-8 | 1e-7 | 3e-7 | 1e-6 | 3e-6 | 1e-5 | 1e-4 | 3e-4 | 1e-3 |
      |---|---|---|---|---|---|---|---|---|---|---|
      | phi1 zero fraction | 0.016 | 0.031 | 0.042 | 0.036 | 0.063 | 0.078 | 0.094 | 0.089 | 0.094 | 0.089 |
      | test accuracy | 79.5 % | 79.1 % | 79.3 % | 77.9 % | 78.3 % | 78.5 % | 77.9 % | 74.7 % | 77.1 % | 76.6 % |

      The β = 3e-5 seed-0 run failed (node). No β approaches 0.896, so two HGQ arms were analysed:
    - **HGQ sparse, β = 3e-4:** the highest seed-0 zero fraction. It ties β = 1e-5 at 0.094; β = 3e-4 was taken because its seeds 1 and 2 were already trained. Seed 2 comes from the duplicate run `seed2b`, because the first run's node failed.
    - **HGQ accuracy-matched, β = 1e-8:** test accuracy 79.37–79.51 %.
  - **Accuracy window:** float seed mean on the 260k test archive is 80.38 %. The models outside the 1-point window are flagged in the combined JSON:
    - L1 seed 1: 78.43 %
    - HGQ 3e-4, all seeds: 77.10 / 76.77 / 75.45 %
    - HGQ 1e-8 seed 2: 79.37 % (1.01 points below)
- **Analysis:** interp/x5_dsets_sparsity.py. The analysis plan is frozen in the module docstring and was written before the first run.
  - Sample: `sample(per_class=2000, seed=311)`, 10,000 jets. Each model ran as its own job (1 CPU / 2 GiB).
  - Element definitions (E2/X2/X6v/E20/E14v):
    - **(a)** Share of live phi1 units that are pure, either a pT ramp (pT variance share ≥ 0.9) or angular (≥ 0.9). Null: each feature column permuted over units, 200 draws, statistic computed on nonzero units.
    - **(b)** Angle swap of ranks k, k+1 with pT order kept, for k = 0–16. Edge contrast c_k at k = 1–15, sign-flip p over 2,000 draws, Holm over 15. A window is present iff some k has Holm p < 0.05 and c_k ≥ 1.5. Controls: whole-row swap, and HGQ's pooling zero-bit masks.
    - **(c)** Single-particle grid at slot 5 (pT 1–100 GeV, r 0–0.4, 72 angles) on each live pooled unit. The unit counts if its within-pT radial share R beats the null (2,000 random ReLU combinations of the live phi1 maps; Holm) and its radial profile fits a count model better than a pT-weighted model.
    - **(d)** Translation of every particle by 0.06 rad along Δη. Measures: W → Z calls, and the ridge read-out of log m on h3, divided by the read-out change under the matched angular rescale with log k = ½ Δlog(M0 M2). Controls: a rotation with equal mean displacement (McNemar, frozen criterion), and a random-direction translation of 0.06 rad (McNemar, reported).
  - Per model: 1,000 jet bootstraps, and Holm over the 4 element p-values.
  - Across regimes: an element counts as present when it reaches Holm 0.05 in ≥ 2 of 3 seeds. Regime test: H0 = equal element statistic across regimes; statistic |mean_HGQ − mean_(float, L1)|; exact p over the 1,680 relabellings; Holm over the 4 elements.
- **Results files:**
  - `results/jedi/x5_{float,l1,hgq,hgq_acc}_s{0,1,2}.json`
  - `x5_combined.json` (HGQ = β 3e-4)
  - `x5_combined_hgq_acc.json` (HGQ = β 1e-8)
  - figure `results/jedi/x5_dsets_sparsity.png`
  - logs `/j-jepa-vol/aaron-interp/logs/x5-*.log`

## Step 0: unedited networks
Each model's explicit forward reproduces its stored 260k-archive test probabilities on the 10,000 sample rows, with argmax agreement 1.000 for all 12 models:

| regime | max \|Δp\| |
|---|---|
| float / L1 | ≤ 5.6e-6 |
| HGQ (fixed point) | ≤ 2.1e-7 |

Sample accuracy is 0.4–1.0 points above test.json accuracy because the sample is class-balanced (z = 2.2–2.6 against the balanced expectation). JEDI-linear itself was not run in X5.

## Results

| element | statistic per seed (0 / 1 / 2) | float | L1 (λ 3e-2) | HGQ β 3e-4 | HGQ β 1e-8 |
|---|---|---|---|---|---|
| — | phi1 zero fraction | 0 / 0 / 0 | 0.885 / 0.906 / 0.896 | 0.094 / 0.078 / 0.047 | 0.016 / 0.026 / 0.031 |
| — | live phi1 / live S units | 63,64,64 / 56,52,48 | 14,12,12 / 49,48,49 | 64,64,64 / 64,61,63 | 64,64,64 / 63,62,64 |
| (a) | pure share P | 0.381 / 0.312 / 0.391 | 1.00 / 1.00 / 1.00 | 0.500 / 0.297 / 0.422 | 0.484 / 0.312 / 0.406 |
| (a) | null 95 % range | [.36,.50] [.28,.41] [.31,.44] | [.89,1] [.86,1] [.89,1] | [.28,.44] [.22,.33] [.23,.36] | [.28,.42] [.22,.31] [.25,.34] |
| (a) | p (one-sided) | 0.96 / 0.88 / 0.47 | 0.74 / 0.55 / 0.56 | 0.005 / 0.21 / 0.005 | 0.005 / 0.090 / 0.010 |
| (a) | ramps + angular / live | 3+21, 3+17, 4+21 | 3+11, 5+7, 4+8 | 12+20, 5+14, 11+16 | 12+19, 5+15, 10+16 |
| (b) | max edge contrast c_k | 1.034 / 1.040 / 1.049 | 1.053 / 1.051 / 1.042 | 1.030 / 1.030 / 1.039 | 1.044 / 1.026 / 1.044 |
| (b) | row swap / angle swap | 0 / 0 / 0 | 0 / 0 / 0 | 0.079 / 0.062 / 0.198 | 0.204 / 0.179 / 0.153 |
| (b) | pooling zero-bit slot fraction | – | – | 0.0046 / 0.0066 / 0.0032 | 0.0002 / 0.0100 / 0.0054 |
| (c) | count-weighted radial objects / live S | 0/56, 2/52, 1/48 | 1/49, 2/48, 1/49 | 0/64, 1/61, 0/63 | 0/63, 0/62, 0/64 |
| (c) | jet-level held-out R² median, count / pT-weighted | .48/.40, .51/.42, .47/.43 | .61/.48, .64/.42, .59/.42 | .54/.38, .54/.31, .55/.36 | .55/.35, .55/.35, .58/.36 |
| (d) | W→Z unedited → translated | .021→.679, .025→.715, .028→.531 | .030→.826, .026→.818, .032→.801 | .068→.872, .082→.627, .069→.812 | .046→.706, .022→.491, .037→.735 |
| (d) | W→Z rotation / random-direction translation | .033/.557, .029/.586, .037/.566 | .030/.774, .027/.766, .036/.680 | .068/.866, .089/.796, .070/.829 | .052/.693, .023/.678, .040/.798 |
| (d) | calibrated read-out ratio [95 % CI] | .843 [.825,.863], .719 [.700,.738], .877 [.857,.897] | .946 [.926,.966], .856 [.832,.880], .819 [.800,.838] | 1.035 [1.009,1.061], .977 [.950,1.004], .939 [.915,.964] | 1.255 [1.218,1.292], .916 [.887,.944], 1.054 [1.030,1.078] |

More detail on the results:
- **(a)** P has no bootstrap spread here: P_ci [P, P] in 11 of 12 models. The live set and the variance shares do not change under jet resampling.
- **(b)** No k has c_k ≥ 1.5 in any model. The largest contrast is at k = 11 in most models; the HGQ accuracy-matched seed 0 has Holm p = 0.007 there with c = 1.044. The angle-swap response falls monotonically with rank, for example 0.014 → 0.003 (|ΔS|/sd) in HGQ seed 0. The row-swap response is 0 for float and L1 because both are permutation-invariant. It is nonzero for HGQ because HGQ uses per-slot input quantizers.
- **(c)** The median within-pT radial share R per live S unit:

  | regime | seed 0 / 1 / 2 |
  |---|---|
  | float | 0.35 / 0.33 / 0.30 |
  | L1 | 0.82 / 0.76 / 0.55 |
  | HGQ β 3e-4 | 0.21 / 0.21 (seed 2 not tabulated) |
  | HGQ β 1e-8 | 0.21 / 0.17 / 0.23 |

  The null R has median 0.50–0.67 and 95th percentile 0.75–0.91.
- **(d)** Translation vs rotation, McNemar on the Z calls of 4,000 true-W jets: p < 1e-200 in all 12 models, with translation larger.
  - Translation along Δη vs a random-direction translation of equal size:

    | regime | seed 0 / 1 / 2 |
    |---|---|
    | float | p 2e-32, 3e-29 (Δη larger); 0.004 (random direction larger) |
    | L1 | 3e-13, 4e-10, 2e-35 (Δη larger) |
    | HGQ β 3e-4 | 0.40; 6e-61 and 0.046 (random direction larger) |
    | HGQ β 1e-8 | 0.21; 5e-56 and 8e-12 (random direction larger) |

  - Ridge read-out R² of log m on h3 on test jets is 0.92–0.96.
  - The raw read-out shift relative to ½Δlog(M0M2) = 0.301 is 0.65–1.23.

**Presence by the frozen criterion (Holm 0.05 in ≥ 2 of 3 seeds)**

| element | float | L1 | HGQ β 3e-4 | HGQ β 1e-8 |
|---|---|---|---|---|
| (a) pure cuts above permutation null | absent (0/3) | absent (0/3; P = 1.0 = null) | present (2/3) | present (2/3) |
| (b) rank windows | absent (0/3) | absent (0/3) | absent (0/3) | absent (0/3) |
| (c) count-weighted radial objects | absent (0/3) | absent (0/3) | absent (0/3) | absent (0/3) |
| (d) shift response, calibrated ratio CI excludes 0 | present (3/3) | present (3/3) | present (3/3) | present (3/3) |

**Regime test.** H0: equal statistic across regimes. H1: the HGQ mean differs from the float/L1 mean. Exact p over 1,680 relabellings (84 distinct HGQ triplets, smallest p 0.012), with Holm over 4 elements:

| HGQ arm | element | statistic (HGQ mean − float/L1 mean) | p | Holm p |
|---|---|---|---|---|
| β 3e-4 | (a) P | 0.41 vs 0.68 | 0.345 | 0.345 |
| β 3e-4 | (b) | 1.033 vs 1.045 | 0.036 | 0.143 |
| β 3e-4 | (c) | 0.005 vs 0.024 | 0.095 | 0.190 |
| β 3e-4 | (d) | 0.984 vs 0.843 | 0.036 | 0.143 |
| β 1e-8 | (a) | 0.40 | 0.357 | 0.595 |
| β 1e-8 | (b) | — | 0.298 | 0.595 |
| β 1e-8 | (c) | — | 0.083 | 0.250 |
| β 1e-8 | (d) | 1.075 vs 0.843 | 0.024 | 0.095 |

## Predictions vs measurements

| hypothesis | prediction | measured |
|---|---|---|
| elements come from HGQ | (a), (b) HGQ only; (c), (d) all | (a) above null HGQ only; (b) absent in HGQ; (c) absent in all; (d) all |
| elements come from task/inputs | (a)–(d) in float and L1 as well | (d) in float and L1; (a) not above null in float/L1; (b), (c) absent everywhere |
| purity from sparsity, windows from masks (E3m) | L1 purity matches JEDI, no windows | L1 purity 1.0, equal to its own random-placement null (p 0.55–0.74); no windows in L1; HGQ formed no masks, so the mask part is not tested |

The "task/inputs" hypothesis is excluded for (a) in float, and for (b) and (c) in all regimes, at this architecture and training length. The "HGQ" hypothesis is excluded for (b) at 10 HGQ epochs. For (d), both hypotheses predict presence in every regime, and that is what was measured.

## Reading (scoped to dsets-d64, n64, these 12 models, and the sample)
- **(d) Shift response.** The translation response is produced in all three regimes. Under the 0.06 rad translation, 49–87 % of W jets are called Z. Under the matched rotation, 2–9 % are. The calibrated read-out ratio is 0.72–1.26 of the $\sqrt{M_0M_2}$ prediction in every regime, and no CI includes 0. In HGQ the response does not depend on the translation direction: the random-direction translation gives an equal or larger W→Z fraction. In float and L1, Δη gives the larger fraction in 5 of 6 models. The HGQ-arm mean ratio is 0.98–1.08 against 0.84 for float/L1; the regime difference has Holm p 0.095–0.14.
- **(a) Pure cuts.** Pure first-layer cuts above the random-placement level appear only in HGQ: 30–50 % of 64 live units, 2 of 3 seeds per arm. This holds in the accuracy-matched arm too, where only 1.6–3.1 % of phi1 weights are zero, so in these models the effect is not carried by zero weights. The fixed-point fraction is far below JEDI-linear's 13/13. The L1 model reproduces JEDI-linear's count (12–14 live units, 3–5 pT ramps, 7–11 angular units, all pure). At that sparsity, permuting the weights gives the same purity, as in X6v for JEDI-linear (p 0.16–0.48).
- **(b) Rank windows.** No regime shows rank windows (max contrast 1.03–1.05, threshold 1.5). Float and L1 are permutation-invariant by construction. HGQ at 10 epochs set ≤ 1 % of pooling-input slots to zero bits, so there were no slot masks for windows to follow. JEDI-linear's masks came from a longer HGQ training; this run does not reproduce that training.
- **(c) Count-weighted objects.** In no regime do count-weighted radial objects reach Holm significance in more than 2 live pooled units (0–4 %). At the jet level, held-out R² of pooled units on counts in r bins exceeds R² on pT-weighted moments in all 12 models (median 0.47–0.64 vs 0.31–0.48). This is a correlational read-out and was not part of the presence criterion.

## Issues
- **HGQ training length.** HGQ was capped at 10 epochs by the plan's 1-hour rule. Neither the sparsity match (≤ 0.094 vs 0.896) nor zero-bit pooling masks were reached. The HGQ-only predictions for (b) are therefore tested only for short HGQ training. A longer HGQ run is the open test: about 30–100 epochs with a β ramp as in JEDI-linear, which is several hours per seed on 3 CPUs.
- **β choice.** β = 3e-4 and 1e-5 tie at seed 0 (0.094). β = 3e-4 was used because its seeds were already trained; β = 1e-5 has 0.8 points higher accuracy. The β = 3e-5 seed-0 run died on node ucm-fiona01 and was not rerun. On the same node, the first β = 3e-4 seed-2 run also died and the first β = 1e-8 seed-1 run (4 GiB) died with no log; they were rerun as `seed2b` and `s1b` (6 GiB).
- **L1 grid.** The planned λ grid did not reach the target sparsity; the extended grid was chosen before this session (see logs train-x5-*).
- **Accuracy matching.** Flagged models: L1 seed 1 (−1.95 points); all HGQ β 3e-4 seeds (−2.7 to −4.9); HGQ β 1e-8 seed 2 (−1.01).
- **Bit-width.** JEDI-linear's l1 mean bit-width was not computed. Matching used the zero-bit fraction only.
- **(d) frozen criterion.** The presence criterion for (d) uses McNemar against the rotation, as written in the frozen docstring. The task text names a random-direction translation; both are reported above.
- **Analysis job resources.** HGQ analysis jobs peaked at 2047 MiB of 2 GiB (page cache) with 0.13–0.37 cores on average. They completed with exit 0.
- **Code changes in this session.**
  - opt-in `--label` / `--hgq_label` in interp/x5_dsets_sparsity.py: output file names only, no change to any definition;
  - interp/x5_plot.py: four arms.
- **Debugging.** No debug attempts were needed in this session. Earlier debug 1 (null definitions) is recorded in the docstring.

## Explainer entry (for site/explain.py, Paper A; not inserted)
```python
X5_EXPLAINER = [
 ('Question', 'Do the elements found in JEDI-linear (pure pT-only or angle-only first-layer cuts, pT-rank windows, count-weighted radial objects, the response to a rigid shift) come from its fixed-point (HGQ) training, or from the task and inputs?', None),
 ('Background',
  'JEDI-linear is a Deep-Sets-like network trained with HGQ, which learns a bit-width per weight and per input slot; zero bits remove a weight or a slot. '
  'Its first layer keeps 20 of 192 weights, and every live unit depends on $p_T$ only or on $(\\Delta\\eta,\\Delta\\phi)$ only. A permutation null (X6v) showed that random placement of 20 nonzero weights gives this purity often (p 0.16–0.48). '
  'Training the same network without quantization, or with an L1 penalty that only produces sparsity, separates the three sources.', None),
 ('Method',
  'Deep Sets $S_c=\\sum_i \\phi_2(\\phi_1(x_i))_c$, $\\rho(S)$ with widths 64/64/64/32/16, trained on the same 3-feature, 64-particle inputs, 3 seeds each: float; L1 on $\\phi_1$ with $\\lambda=3\\times10^{-2}$ (0.89–0.91 zero weights, JEDI-linear 0.896); HGQ for 10 epochs at $\\beta=3\\times10^{-4}$ (largest zero fraction reached, 0.05–0.09) and $\\beta=10^{-8}$ (accuracy within 1 point). '
  'On 10,000 test jets: (a) share of live first-layer units with $\\ge 90\\,\\%$ of their variance from $p_T$ or from the angles, against weights permuted across units; (b) swap the angles of ranks $k,k+1$ and look for an edge in the response; (c) single-particle scans of each pooled unit for a radial, count-weighted profile; (d) translate every particle by 0.06 rad and compare the mass read-out change with $\\tfrac12\\Delta\\log(M_0M_2)$ and with a rotation of equal displacement.', None),
 ('Result',
  'The shift response appears in every regime: 49–87 % of W jets become Z (rotation: 2–9 %) and the read-out moves 0.72–1.26 of the $\\sqrt{M_0M_2}$ prediction. '
  'Pure cuts above the permutation null appear only with HGQ (30–50 % of units, 2 of 3 seeds); L1 gives 100 % pure units, but so does its null. No regime has rank windows (edge contrast $\\le 1.05$, threshold 1.5) or more than 2 count-weighted radial objects.', 'results/jedi/x5_dsets_sparsity.png'),
 ('Checks',
  'Each network\'s forward reproduces its stored test probabilities (argmax agreement 1.000). HGQ at 10 epochs set $\\le 1\\,\\%$ of pooling-input slots to zero bits, so it formed no slot masks; whether longer HGQ training forms windows is not tested. Regime differences do not survive Holm (smallest 0.095).', None),
 ('Why it matters', 'The mass-like shift response is a property of the task and inputs in this architecture, not of fixed-point training; the purity of a very sparse first layer is what sparsity alone gives.', None),
]
```
