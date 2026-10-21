# X15 verify: Linformer rank-projection E, re-run on a fresh sample and smooth rank-structure checks

Script `interp/x15_verify.py` (imports `interp.x15_linformer_ranks` unchanged); results `results/jedi/x15_verify.json` (parts A–C and a
summary of the re-run) and `results/jedi/x15_verify_small.json` (full re-run output, same schema as `x15_small.json`); log `logs/x15-verify.log`.
Job `x15-verify` (AI_PY=conda, 1 CPU / 2 GiB, 1,152 s, peak 1.22 GiB, 0.97 cores).

## Conclusion
On a fresh set of 5,000 jets, every X15 number reproduces: swapping two neighbouring particles' directions matters more where the
Linformer's rank table E jumps, random untrained tables show the same, and the jump positions do not recur between training seeds.
X15 only looked for sharp jumps, though. Looking at the overall size of each rank's weight instead, all four trained Linformers give the
5 hardest particles about 2.3 times the typical weight in their second block. A random table never reaches that. Scaling those 5 weights
down to the typical size changes about 40 % of the network's decisions. So these networks do learn a reproducible, graded preference for
the hardest few particles, a soft version of JEDI-linear's "sum over the hardest 14" window. What is still open is how much of the
causal effect comes from that learned preference, and how much from hard particles carrying most of the jet's momentum in any network.

## Code check (x15_linformer_ranks.py)
- E enters as `einsum("bhnd,nk->bhkd", k * m, E)` with padded tokens zeroed (train/models.py, l. 57–59); the script reads
  `blocks[b].attn.E`, the matrix used in the forward pass. No positional encoding, so E is the only rank-dependent weight.
- Swap edit (`swap`): it exchanges columns 1:3 (Δη, Δφ) of ranks i and i+1, and only in jets with ≥ i+2 real particles. pT, the pT order
  and the padding are unchanged. Padded slots are never edited or counted: means run over eligible jets only.
- Hardest-axis inputs are recentred after the edit. Rank 0 is never moved, so the axis is unchanged (numpy vs torch recentring max |diff| 0.0).
- `ckpt/epoch_0.pt` is the initial E: a fresh build under `torch.manual_seed(seed)` reproduces it exactly (max |diff| 0.0).
- The test split only (`full_test.npz`). The unedited outputs match the stored `test_probs.npy` (max |Δp| ≤ 6×10⁻⁶ on the new sample).
- Edge score: rows are compared with the 2-norm over the 8 slots. That is invariant to permuting the slots, which is a symmetry of the layer.
- Limitation of the design, not a bug: the edge score is a first difference between adjacent ranks. In a random initial E it has a
  typical size of about 0.5, so it hides any smooth (multi-rank) structure that training adds. Parts A–B below test that.
- No bugs found that change the X15 numbers.

## Part 0: re-run on a fresh sample
**Procedure:** the same code and settings as X15, but sample `interp.jedi_core.sample(per_class=1000, seed=2602)` (X15 used seed 1501),
N = 5,000. The bootstrap and derangement seeds are unchanged. The sample accuracies are now 80.6–81.3 % (full test 80.6–81.1 %).

| quantity | X15 (seed 1501) | verify (seed 2602) |
|---|---|---|
| R at own top-5 E edges vs matched flats (L-st-s0 / L-st-s1 / L-ax-s0 / L-ax-s1) | 1.19 / 1.76 / 2.08 / 1.81 | 1.18 [1.12, 1.25] / 1.75 [1.67, 1.82] / 2.14 [2.02, 2.27] / 1.80 [1.71, 1.89] |
| permutation p (252 labellings) | 0.37 / 0.052 / 0.052 / 0.16 | 0.37 / 0.048 / 0.048 / 0.15 |
| untrained R (U0–U4) | 1.89, 2.60, 1.18, 2.58, 1.57 | 1.86, 2.60, 1.33, 2.58, 1.56 |
| R at L-st-s0's pairs: other trained / Deep Sets | 0.99–1.09 / 0.81 | 1.00–1.13 / 0.76 |
| swap response vs edge score, Spearman raw (trained) | 0.42–0.64 | 0.46–0.63 |
| step 3, L-st-s0 across/inside at edges 60, 3, 32, 9 | 2.88, 1.82, 1.07, 0.58 | 2.83, 1.79, 1.09, 0.61 |
| step 3, Deep Sets at the same blocks | 0.98/0.95, 1.75/1.73, 0.92/0.91, 0.62/0.63 | 0.94/0.92, 1.67/1.67, 0.95/0.99, 0.68/0.69 |

R = mean |Δlogit| at the top-5 edge boundaries ÷ mean |Δlogit| at 5 matched flat boundaries. |Δlogit| = mean over the 5 classes of the
absolute change in the logit. The flat partner of edge 32 moved from 34 to 30. The weight-only results (top edges, recurrence: stored-axis
0/5 overlap, ρ −0.03; hardest-axis 2/5, p 0.048) do not depend on the sample and are identical.

## Part A: smooth, slot-invariant structure of E (weights)
**Definitions.** Row norm r_b(n) = ‖E_b[n]‖₂, the overall weight of the rank-n particle in block b. LRE (low-rank emphasis) =
mean r_b(0..4) ÷ median r_b(5..39): how much more the 5 hardest particles weigh than a typical particle. The smoothed edge score is
‖mean E[i+1..i+4] − mean E[i−3..i]‖, a jump between 4-rank averages. The rank Gram matrix G = E Eᵀ (ranks 0–39) holds which ranks get
similar slot weightings, and it is invariant to permuting the 8 slots.
**Null (H0: E has no rank structure beyond a random draw):** 4,000 random E with i.i.d. N(0, 1/64) entries, the PyTorch initialisation of E.
Cross-net statistics use 2,000 pairs of independent random E. The 5 untrained Linformers give a second null.
**H1:** training produces rank structure that recurs across seeds with different initial E.

- **LRE, block 2:** L-st-s0 2.34, L-st-s1 2.43, L-ax-s0 2.48, L-ax-s1 2.23. Random-E null: median 1.01, q99 1.34, max 1.53 (n = 8,000).
  p < 1.3×10⁻⁴ for each net. Untrained 0.93–1.20. The nets' own initial E 1.00 / 1.17, and the learned change ΔE alone 1.80–2.43.
  Block 1: 1.05 / 1.25 / 1.38 / 1.43 (p 0.37 / 0.038 / 0.005 / 0.002), so it is clearly raised only in the hardest-axis nets.
- **Agreement between nets with different seeds** (and so different initial E), mean over the 2 blocks:

| pair | row-norm ρ (ranks 0–39) | row-norm ρ (ranks 5–39) | smoothed-edge ρ | Gram r |
|---|---|---|---|---|
| L-st-s0 vs L-st-s1 | 0.44 | 0.30 | 0.55 | 0.46 |
| L-ax-s0 vs L-ax-s1 | 0.53 | 0.40 | 0.49 | 0.34 |
| L-st-s0 vs L-ax-s1, L-st-s1 vs L-ax-s0 | 0.44, 0.47 | 0.38, 0.34 | 0.50, 0.49 | 0.26, 0.24 |
| random-E null q99 | 0.27 | 0.27 | 0.36 | 0.065 |
| 10 untrained pairs | −0.19 to 0.03 | −0.18 to 0.11 | −0.09 to 0.29 | −0.02 to 0.05 |

  All four cross-seed pairs exceed the null q99 on all four statistics (p ≤ 0.0055). The learned change ΔE alone agrees less across seeds.
  Its Gram r is 0.04–0.23 (p 0.001–0.05) and its row-norm ρ 0.16–0.24 (p 0.02–0.09).
- **Max X15 edge score under the random-E null:** q95 1.63, q99 1.73. The hardest-axis nets reach 1.90 and 1.96 at boundary 2|3, above q99,
  and the stored-axis nets 1.56 and 1.60. In L-ax-s0, boundaries 1|2, 2|3 and 3|4 are all high (1.78, 1.90, 1.81). Its block-1 rows 1–3
  are also heavier than typical (LRE 1.38). So the 2|3 jump is the edge of a raised low-rank weighting in block 1, not an isolated jump.

## Part B: causal test of the low-rank emphasis (unit intervention on E)
**Edit:** in one block of a trained Linformer, rows 0–4 of E (the 5 hardest ranks) are rescaled to that block's median row norm over
ranks 5–39. The direction of each row is kept, so only the extra weight of the hardest particles is removed. Factors are 0.28–0.61 in
block 2 and 0.53–1.50 in block 1. The jets are not edited. Sample: seed 2602, N = 5,000.
**Controls (20 draws each):** (i) the same 5 factors applied to 5 random rows from ranks 5–25 (ranks occupied in ≥ 90 % of jets);
(ii) random-direction perturbations with the same per-row norm as the real edit, added to 5 random rows of ranks 5–25.
**Metric:** mean |Δlogit|, % calls changed, accuracy change (95 % jet-bootstrap CI), and McNemar on per-jet correctness (unedited vs edited).
p = (1 + # controls ≥ real) / 21; 0.048 is the smallest value 20 draws can give.

| net | block | Δacc real (CI) | calls changed | |Δlogit| real | ctrl (i) Δacc / |Δlogit| | ctrl (ii) Δacc / |Δlogit| |
|---|---|---|---|---|---|---|
| L-st-s0 | 2 | −31.5 [−33.0, −30.0] pts | 43.2 % | 1.63 | −1.3 / 0.33 | −1.3 / 0.27 |
| L-st-s1 | 2 | −27.0 [−28.5, −25.6] | 40.2 % | 1.42 | −7.0 / 0.61 | −7.0 / 0.51 |
| L-ax-s0 | 2 | −30.9 [−32.4, −29.4] | 43.8 % | 1.44 | −3.0 / 0.40 | −3.8 / 0.39 |
| L-ax-s1 | 2 | −23.6 [−25.0, −22.3] | 33.9 % | 1.21 | −1.7 / 0.30 | −2.4 / 0.31 |
| L-st-s0 / L-st-s1 / L-ax-s0 / L-ax-s1 | 1 | −3.1 / −6.7 / −24.0 / −12.7 | 8.1 / 14.0 / 35.3 / 23.7 % | 0.33 / 0.59 / 1.19 / 0.80 | −0.0 to −0.8 | −0.2 to −1.2 |

The real edit exceeds every control draw in all 8 cases (p = 0.048 each, the floor). McNemar p < 10⁻¹⁶ in all 8 cases. Holm over the 8
leaves 0.38 for the control comparison, because 20 draws limit the attainable p; the effect sizes are 4–6× the control means and 3–5× the control maxima.
**Competing explanation not excluded:** the hardest 5 particles carry most of the jet's pT, so any change to their rows of E may matter
more than the same change at ranks 5–25, whether or not the weight there was learned. The weight-level result in part A (LRE far above
random E and above each net's own initial E) is what shows the emphasis is learned. Part B shows those rows carry a large share of the output.

## Part C: cross-seed agreement of the causal swap-response profiles
**Procedure:** the Spearman correlation between two nets' 62-boundary swap-response profiles (from part 0). Raw, and partial given a
quadratic in rank and the Deep Sets D-s0 response (the data-only part, set by the pT gap and angular distance of the swapped pair).
Permutation p (2,000).
- Partial ρ: L-st-s0 vs L-st-s1 0.16 (p 0.19); L-ax-s0 vs L-ax-s1 0.35 (p 0.005); cross-axis different seed 0.29, 0.32 (p 0.02, 0.01);
  same seed 0.70, 0.67 (p < 5×10⁻⁴). The 10 untrained pairs give −0.15 to 0.36, with 2 of 10 at p < 0.05.
- Once rank and the data-only response are removed, the per-boundary causal profile agrees across seeds at about the level of
  untrained-net pairs. It is shared within a seed. This matches X15's statement on edge positions.

## Assessment of the X15 reading
- Reproduced: all step 1–3 numbers; edge-response coupling present equally in untrained nets (H_lin); no cross-seed recurrence of
  sharp edge positions; edge positions follow the initial E; Deep Sets reproduces the step-3 asymmetries at edges 3 and 9; edge 60 differs.
- Needs correction: "these networks do not form a reproducible rank window" and the thesis note ("supports ... window is specific to
  the official checkpoint rather than a general element"). The edge statistic only detects sharp jumps. Smooth, slot-invariant statistics
  show reproducible rank structure across seeds (row-norm ρ 0.44–0.53, Gram r 0.24–0.46 vs null q99 0.27 / 0.065). All four nets learn a
  graded 2.2–2.5× weighting of the 5 hardest ranks in block 2 (random-E max 1.53), and removing it changes 34–44 % of calls (control means 5–16 %).
  This is not a hard-edged window like JEDI-linear's S0 (ranks 0–13), but it is a reproducible learned emphasis on the hardest particles.
- Refinement: the hardest-axis 2|3 jump exceeds the random-E q99 (1.73) in both nets, so the evidence is stronger than "above 5 untrained
  nets". It sits inside a raised block-1 low-rank weighting (LRE 1.38, 1.43; p 0.005, 0.002).

Corrected reading: sharp rank edges in E are set by the random initialisation, and their causal effect follows E's roughness in trained and
untrained nets alike. Training does produce a reproducible graded rank weighting, strongest as a 2.2–2.5× emphasis on the 5 hardest
particles in block 2. It recurs across seeds and axis conventions, and removing it costs 24–31 accuracy points. Whether that cost reflects
the learned emphasis or the general importance of hard particles is not separated.

## Explainer entry (for site/explain.py)
- **Question:** besides sharp jumps, does a Linformer's rank table E carry a rank weighting that recurs across training seeds?
- **Background:** E ∈ ℝ^{64×8} weights each pT rank in 8 summaries; JEDI-linear's S0 is the hard-edged sum over the 14 hardest particles.
- **Method:** $\mathrm{LRE}_b=\overline{\|E_b[0..4]\|}/\mathrm{median}\,\|E_b[5..39]\|$; rank Gram matrix $G=EE^\top$; null = random $E_{nj}\sim\mathcal N(0,1/64)$.
  Causal edit: rescale rows 0–4 of block-2 E to the median row norm; control = same factors on random rows of ranks 5–25.
- **Result:** LRE 2.2–2.5 in all four trained nets vs random max 1.53. The edit changes 34–44 % of calls (control means 5–16 %).
- **Checks:** fresh-sample re-run of X15 reproduces all numbers; untrained nets show no LRE or cross-net agreement.
- **Why it matters:** a graded "hardest few particles" weighting is learned reproducibly; the hard-edged window is not.
- **Glossary additions:** LRE (low-rank emphasis); rank Gram matrix; smoothed edge score.
