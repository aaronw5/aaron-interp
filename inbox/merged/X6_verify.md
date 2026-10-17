# X6 verification: program elements on other official JEDI-linear checkpoints

Script `interp/x6_verify.py`. Jobs:
- `x6-verify` (1 CPU / 2 GiB): n64, perminv, the combine step and n32 completed. During n128 step V5 the pod reached the 2 GiB limit and
  ran at about 0.03 cores; the job was deleted.
- `x6-verify2` (1 CPU / 2 GiB, `--only56`): n128 steps V5 and V6 only, grid batch 250. Exit 0; wall 842 s; peak 2047 MiB; 0.06 cores.

Results:
- `results/jedi/x6_verify_3-feature_n{32,64,128}.json`
- `results/jedi/x6_verify_3-feature-perminv_n64.json`
- `results/jedi/x6_verify_combined.json`

Logs: `logs/x6-verify.log`, `logs/x6-verify2.log`.

Fresh sample: V = `sample(per_class=1500, seed=4242)` with every jet of X6's M removed, N = 6,950. Per-jet calls were also computed on
X6's M (N = 20,000) and C (N = 9,223) for the paired accuracy test. The live sets, edge plans, E5 units and slots, and g mean were read
from X6's JSON files and not refitted.

## Code review of `x6_models_jedi.py` / `x6_plot.py`
- **Stages and quantizers:**
  - The forward pass is `jedi_core.Jedi`.
  - The float program matches `e3m_masks.py`.
  - The pool scale `L[k].scale` is applied.
  - The l1 input masks have shape [1, n, 3] and are 0 % zero in all models, so they impose no slot window.
- **No split leakage:** C and V exclude every M jet.
- **E1 edit scope:** the mean ablation of `a`, `s` and `u` writes the per-particle mean into padded slots as well as real ones. This does
  not change E2, because in every model the live l1 set equals the set of units with any nonzero weight (13/13, 12/12, 15/15, 17/17).
- **"Masks removed" lets padded slots into the sums:** the input of a padded slot is $(0-\text{shift})/\text{scale}\neq 0$, so l1
  outputs at padded slots are nonzero. `ones_masks` sets the slot masks to 1 at padded slots as well as real ones (measured in V4 below).
- **Selective reporting:**
  - The post hoc n128 edge 61|62 has the largest edge mass under the relaxed rule (0.665, units S6, S21, S39, S48, S54, S60). Its X6 results are
    network ratio 0.41 [0.34, 0.49] on M and 0.31 on C. They are stored in the JSON but are not in the report table, the figure or the reading.
  - Second frozen edges are in the report text but not in the reading: n64 0|1 gives 0.14 and n32 5|6 gives ≈ 0.
- **Radial share includes pT variance:** the X6 radial share puts the pT dependence into the total variance. The within-pT version is
  measured in V5 below.
- **Figure** (`x6_models.png`, opened):
  - Panel (a) title "no mixed cuts in any model" is a declarative statement and shows no null.
  - Panel (b) shows only the first frozen edge plus n128 19|20; 61|62 and the second edges are absent.
  - Axes and units are readable.

## V1 accuracy on V (N = 6,950)
| model | X6 M | V |
|---|---|---|
| n64 | 80.59 % | 81.25 % |
| n32 | 77.86 % | 77.57 % |
| n128 | 80.30 % | 80.94 % |
| perminv | 81.48 % | 81.74 % |

Paired perminv − n64 test (H0: P(perminv right, n64 wrong) = P(n64 right, perminv wrong); exact McNemar; 4,000-draw bootstrap over jets):

| sample | N | difference | 95 % CI | discordant n10 / n01 | p |
|---|---|---|---|---|---|
| M | 20,000 | +0.90 points | [0.58, 1.22] | 432 / 611 | 3.3e-8 |
| C | 9,223 | +1.34 points | [0.89, 1.80] | 170 / 294 | 9.3e-9 |
| V | 6,950 | +0.49 points | [−0.04, 1.02] | 154 / 188 | 0.074 |

## V2 null for "pure" l1 cuts (H0: nonzero kernel entries placed at random)
- Statistic: the fraction of l1 units with any nonzero fused weight that are pT-only or angle-only.
- Null: each feature's zero/nonzero column of the 3 × 64 fused kernel permuted independently over the 64 units (per-feature count of nonzero weights kept); 20,000 draws.
- p = P(null ≥ observed). The observed fraction is 1.0 in every model.

| model | nonzero weights (pT, Δη, Δφ) | units with any nonzero | null fraction 2.5/50/97.5 % | p |
|---|---|---|---|---|
| n64 | 4, 8, 8 | 13 | 0.81 / 0.94 / 1.00 | 0.34 |
| n32 | 3, 8, 7 | 12 | 0.86 / 0.94 / 1.00 | 0.48 |
| n128 | 3, 10, 10 | 15 | 0.88 / 0.95 / 1.00 | 0.35 |
| perminv | 7, 8, 7 | 17 | 0.78 / 0.95 / 1.00 | 0.16 |

- All four models together (zero mixed units in each): product of the per-model probabilities 0.009; Fisher combination p = 0.31.
- At the observed sparsity (3–7 nonzero pT weights among 64 units), random placement gives "no mixed unit" in 16–48 % of draws per model.
- The untrained redraw does not keep the trained zero pattern (n64 redraw: 1, 7, 7 nonzero weights), so it is not a control for E2.

## V3 E20 window test on V (ratio of mean |ΔS|/sd(S), across ÷ inside, bootstrap 95 %, paired sign-flip p, 10,000)
Two mask-removal variants were tested:
- X6's "ones": slot layers set to 1.
- "Slot-uniform": each channel is kept at every slot if it is kept at any slot. This keeps channel pruning and removes only the slot dependence.

Both variants give the same ratios to 3 decimals.

| model, edge vs inside (units) | network | float + masks | masks removed (ones / slot-uniform) |
|---|---|---|---|
| n64 13\|14 vs 11\|12 (S0, S43, S51, S52, S63) | 5.37 [5.01, 5.75], p < 1e-4 | 26.0 | 0.81 [0.75, 0.87] |
| n64 0\|1 vs 3\|4 (S1), frozen 2nd | 0.27 [0.10, 0.48] | 1.79 | 0.66 |
| n32 9\|10 vs 7\|8 | 2.36 [2.22, 2.51], p < 1e-4 | 3.66 | 0.71 |
| n32 5\|6 vs 28\|29 (S34, S45, S63), frozen 2nd | 0.001 | ≈ 0 | ≈ 0 |
| n128 23\|24 vs 20\|21 (S10), frozen | 0.75 [0.69, 0.81] | 1.35 (p 0.40) | 1.41 / 1.18 |
| n128 25\|26 vs 20\|21 (S10), frozen | 1.00 [0.93, 1.07], p 0.97 | 0.64 | 1.75 / 1.70 |
| n128 61\|62 vs 59\|60 (S6, S21, S39, S48, S54, S60), post hoc | 0.32 [0.27, 0.38], p < 1e-4, N 1,707 | 0.68 [0.54, 0.84] | 0.81 |
| n128 19\|20 vs 16\|17 (S43, S57, S63), post hoc | 4.59 [4.32, 4.89], p < 1e-4 | 8.85 | 0.82 [0.76, 0.88] |
| perminv 13\|14 vs 5\|6 (top-10 S) | 0.34 [0.32, 0.36] | 0.34 | 0.34 |

The across > inside result reproduces at the hard-particle window edge of each sorted model: 5.37, 2.36 and 4.59. At the other five
edges the ratio is ≤ 1.00 for the network: 0.27, ≈ 0, 0.75, 1.00 and 0.32. Two of these edges carry units whose single-particle maps
depend on angle: n64 0|1 (S1) and n128 61|62 (S21, S48, S54).

## V4 float-program accuracy on V: what the mask-removal cost contains
| model | float + masks | X6 "ones" | slot-uniform | ones on real slots only (padded slots keep masks) | slot-uniform on real slots only |
|---|---|---|---|---|---|
| n64 | 80.76 % | 43.11 % | 43.11 % | 62.33 % | 62.33 % |
| n32 | 77.21 % | 74.14 % | 74.14 % | 74.56 % | 74.56 % |
| n128 | 76.78 % | 23.93 % | 26.98 % | 56.12 % | 57.96 % |
| perminv | 80.46 % | 80.46 % | 80.46 % | 80.46 % | 80.46 % |

- Removing the masks at real-particle slots only costs 2.7 (n32), 18.4 (n64) and 20.7 (n128) points.
- Letting padded slots into the sums as well raises the cost to 3.1, 37.7 and 52.9 points.
- In n64 and n128 about half or more of X6's reported mask-removal cost comes from padded slots, not from rank windows over real particles.

## V5 radial share (single particle at X6's slot, g frozen at the M mean)
- X6's share uses a total variance that includes the pT dependence.
- The within-pT share removes each pT's mean: $1-\sum_{p_T,r}\sum_\theta (f-\bar f_{ring})^2 / \sum_{p_T}\sum_{r,\theta}(f-\bar f_{p_T})^2$.

| unit | X6 share | within-pT share | pT share of the total variance |
|---|---|---|---|
| n64 S29 / S1 / S50 | 0.995 / 0.978 / 0.997 | 0.994 / 0.977 / 0.994 | 0.05 / 0.02 / 0.45 |
| n32 S5 | 0.996 | 0.995 | 0.20 |
| n128 S48 / S21 / S54 / S12 | 0.999 / 0.999 / 0.971 / 0.999 | 0.999 / 0.999 / **0.941** / 0.999 | 0.37 / 0.39 / 0.51 / 0.17 |
| perminv S54 / S13 / S48 | 0.997 / 0.998 / 0.992 | 0.994 / 0.995 / 0.989 | 0.46 / 0.53 / 0.24 |

Null: random combinations of the model's own live angle-only l1 maps at the same slot, 2,000 draws each. X6's control was single half-planes, 0.14–0.17.

| model | equal-weight sum | Dirichlet(1) 2.5/50/97.5 % | Dirichlet draws ≥ 0.97 | ReLU(random signed sum − median), 97.5 % |
|---|---|---|---|---|
| n64 | 0.859 | 0.32 / 0.62 / 0.93 | 0.45 % | 0.83 |
| n32 | 0.848 | 0.31 / 0.62 / 0.92 | 0.40 % | 0.81 |
| n128 | 0.964 | 0.41 / 0.73 / 0.95 | 1.0 % | 0.80 |
| perminv | 0.974 | 0.38 / 0.71 / 0.95 | 0.85 % | 0.82 |

- 11 of the 12 position-dependent units lie above the 97.5 % quantile of the Dirichlet null under both definitions.
- n128 S54 within pT (0.941) lies below that quantile (0.955) and below the equal-weight sum of n128's angle cuts (0.964).
- For perminv the unweighted sum of its angle cuts already has share 0.974.

## V6 translation 0.06 rad along +Δη vs matched rotation on V (N_W = 1,384, Wilson 95 %, McNemar)
| model | unedited | +Δη | rotation | n10 / n01 | log10 p |
|---|---|---|---|---|---|
| n64 | 0.022 | 0.760 [0.737, 0.782] | 0.023 | 18 / 1038 | −279 |
| n32 | 0.029 | 0.451 [0.425, 0.477] | 0.023 | 26 / 618 | −147 |
| n128 | 0.029 | 0.736 [0.712, 0.759] | 0.029 | 22 / 1001 | −263 |
| perminv | 0.021 | 0.561 [0.534, 0.587] | 0.026 | 23 / 763 | −192 |

The n32 value on V (0.451) lies outside X6's M interval [0.484, 0.515]. The other three are within 0.03 of M.

## Hard-particle window sizes
"10 / 14 / 20 ranks at n = 32 / 64 / 128" names one window per model. The other qs1 prefix windows in X6's own tables are:
- n32: 0–15, 0–20
- n64: 0–2, 0–9, 0–10, 0–25
- n128: 0–3, 0–7, 0–17, 0–23

## Corrected reading
- **Present in all four checkpoints:**
  - Every l1 unit with a nonzero weight has exactly zero weight on either pT or angle (57/57). At the observed kernel sparsity, random
    placement of the nonzero weights gives this outcome with p = 0.16–0.48 per model (product over the four models 0.009; Fisher 0.31).
  - Radial share ≥ 0.97 for 11 of 12 position-dependent top-5 S units; n128 S54 has 0.941 within pT. Random non-negative combinations of the
    same model's angle cuts exceed 0.97 in 0.4–1.0 % of draws. The equal-weight sum reaches 0.85–0.97.
  - W→Z under a 0.06 rad shift: 0.45–0.76 on V, against 0.02–0.03 for the matched rotation.
- **Sorted-input models:**
  - At the edge of the shared hard-particle window (9|10, 13|14, 19|20; the n128 edge chosen post hoc), an angle swap across the edge changes
    those S units 2.4–5.4 times more than a swap inside. With masks removed the ratio is 0.71–0.82.
  - At the other five edges tested (n64 0|1, n32 5|6, n128 23|24, 25|26, 61|62) the network ratio is 0.27–1.00 or ≈ 0.
  - Removing slot masks at real-particle slots only costs 2.7 / 18.4 / 20.7 points (n32 / n64 / n128). Admitting padded slots as well
    costs 3.1 / 37.7 / 52.9.
- **perminv vs n64:** perminv is more accurate on the same jets: paired +0.90 [0.58, 1.22] on M (p = 3e-8) and +1.34 on C (p = 9e-9);
  +0.49 [−0.04, 1.02] on V (p = 0.07).
