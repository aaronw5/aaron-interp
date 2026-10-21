# X15: learned rank windows in a Linformer (claims A1, A3, A10; exploratory)

Script `interp/x15_linformer_ranks.py`; results `results/jedi/x15_small.json`, `x15_jedi.json`, `x15_summary.json`; figure
`results/jedi/x15_linformer_ranks.png`; logs `logs/x15-run.log`, `logs/x15-jedi.log`. Jobs: `x15-run` (AI_PY=conda, 1 CPU / 2 GiB,
727 s, peak 1.16 GiB, 0.97 cores) and `x15-jedi` (jax env, 1 CPU / 2 GiB, 630 s, peak 1.30 GiB). Two debug fixes before the final run
(the untrained nets read `arch.json` from a seed folder that does not exist; a boolean mask was built from a float array).

## Conclusion
A Linformer summarises its pT-sorted particles with a learned table E that weights each pT rank. We looked for "rank windows"
like JEDI-linear's, meaning places where this weighting jumps between neighbouring ranks. Swapping the directions of two
neighbouring particles changes the output more where E jumps than where it is flat. Untrained Linformers with random E show the same
effect, and as strongly (trained 1.2–2.1x, untrained 1.2–2.6x). The jump positions do not recur between the two stored-axis training
seeds, and they mostly follow each network's random starting E. So in this Linformer the rank weighting behaves like a random lookup
table that training changes a lot (by about its own size), not like a reproducible window. One exception is still uncertain: both
networks trained with the hardest particle as the axis develop a jump at ranks 2|3, above every untrained network.

## What E is
In each of the two Linformer blocks (`linf-2b-d32-k8`, width 32, 2 heads), the keys $K_n$ and values $V_n$ of the $N=64$ pT-sorted
particle tokens are compressed into $k=8$ summaries, $\tilde K_j=\sum_n E_{nj}K_n$ and $\tilde V_j=\sum_n E_{nj}V_n$, with one learned
$64\times8$ matrix $E$ shared by keys, values and heads. Padded tokens are zeroed before the sum. Row $E_n$ is the weight the particle of pT rank $n$
(rank 0 = hardest) gets in each of the 8 summaries, so E is a learned weighting of particles by their pT rank. JEDI-linear's zero-bit masks
are the hard-edged version of the same thing: S0 keeps ranks 0–13 (S0 = ΣpT of the 14 hardest) and S19 keeps ranks 13–63 (soft multiplicity).
Panel (a) of the figure shows E's rank profile with these two masks.

## Procedure
- **Networks** (existing checkpoints, no training): Linformers `linf-2b-d32-k8-n64` seeds 0, 1 with stored-axis inputs (L-st-s0/s1)
  and with inputs measured about the hardest particle (`-axhard`, L-ax-s0/s1). Controls: Deep Sets `dsets-d64-n64` seeds 0, 1
  (D-s0/s1). Deep Sets sums over particles, so its output cannot depend on which rank a particle has. Also 5 untrained Linformers
  (U0–U4, PyTorch default init under `torch.manual_seed(5000+k)`, so E has i.i.d. entries $\mathcal N(0,1/64)$), and each trained Linformer's
  own initial E (`ckpt/epoch_0.pt`). Reference: official JEDI-linear 3-feature n64 (`interp.jedi_core.Jedi`).
- **Sample**: `interp.jedi_core.sample(per_class=1000, seed=1501)`, N = 5,000 test jets (1,000 per class). Particles are pT-sorted (the
  stable re-sort is the identity). Each network gets its own input convention, with (x − shift)/scale from `arch.json`. Hardest-axis inputs are
  recentred with a numpy copy of `train_small.recentre_hardest` (max |difference| from the torch version 0.0).
- **Unedited check**: the Linformers and Deep Sets give the same probabilities as their stored `test_probs.npy` rows (max |Δp| ≤ 8×10⁻⁶,
  argmax agreement 1.000). Accuracy on this sample: L 80.20 / 79.94 / 80.20 / 79.78 %, Deep Sets 79.60 / 80.02 %, JEDI-linear 80.14 %.
  Full test-set accuracy is 80.6–81.1 %; the sample values are 0.6–1.0 points lower, within about 2 binomial standard errors
  (0.56 points at N = 5,000), and the forward pass matches the stored outputs exactly.
- **Candidate boundaries** $i|i{+}1$: $1\le i\le62$ with at least 500 jets having a real particle at rank $i+1$. All 62 qualify. Rank 0
  is never moved, so the hardest-axis recentring is never touched.
- **Step 1 (weights)**: edge score $e(i)=\frac12\sum_{b=1}^{2}\|E_b[i{+}1]-E_b[i]\|_2/\mathrm{median}_i\|E_b[i{+}1]-E_b[i]\|_2$. This is the size of
  the jump in the 8-slot weighting between neighbouring ranks, relative to the network's typical jump. We take the top 5 boundaries per network.
  Recurrence is the overlap of the top-5 sets between networks (exact, and within ±1 rank). The chance overlap is hypergeometric (exact)
  or comes from 20,000 random 5-subsets (±1). We also report the Spearman correlation of whole profiles, with a permutation p (5,000).
  The same is done for the learned change $\Delta E=E-E_{\rm init}$.
- **Step 2 (causal angle swap)**: at boundary $i|i{+}1$, the rank-$i$ and rank-$(i{+}1)$ particles exchange $(\Delta\eta,\Delta\phi)$ and keep
  their pT. The pT order, the pT multiset and the set of occupied slots stay the same; only which angle sits at which rank changes. The edit is
  applied to jets with at least $i+2$ real particles, and the mean is taken over those jets. All 62 boundaries are scanned.
  - **Metric**: mean over jets of $\frac15\sum_c|\Delta\,\mathrm{logit}_c|$, and % of calls changed.
  - **Flat pairs**: one per top edge, taken from boundaries with $e\le$ median and at least 2 ranks from any top edge. Each minimises
    $|i-j|/4+|\log(\mathrm{gap}_i/\mathrm{gap}_j)|+|\log(\Delta R_i/\Delta R_j)|$. Here gap is the median relative pT gap of the two particles
    and $\Delta R$ is the median angular distance between them, so an edge and its flat partner move particles of similar pT spacing and
    similar angular separation in a similar rank region.
  - **Ratio R** = mean response at the 5 edge pairs ÷ mean response at the 5 flat pairs. R = 1 means swapping across an E jump does the
    same as swapping where E is flat. We give a 95 % CI from 1,000 jet-bootstrap resamples and an exact permutation test over the 252
    edge/flat labellings of the 10 pairs (smallest possible p = 0.004). Each edge is also tested against its own flat pair: a paired bootstrap
    for |Δlogit| and exact McNemar for calls, Holm-corrected over the 5 edges (Holm = step-down multiple-comparison correction).
  - **Rank-region control**: the Spearman correlation over all 62 boundaries of swap response vs edge score, raw and partial given rank
    (residuals after a quadratic in rank).
- **Step 3 (window shuffle)**: for each top edge e, the "across" block is ranks e−2…e+3 (3 on each side of the jump). The "inside" block is
  the nearest 6 consecutive ranks with no top-5 edge between them. In both blocks the angles are permuted by the same random derangement
  per jet (rng 150100+e), with pT kept per rank, so 6 particles move in each. The read-out is ratio across/inside with a bootstrap CI,
  paired bootstrap p and McNemar, Holm over edges.
- **Controls in steps 2–3**: untrained Linformers at their own top-5 edges (same procedure). Deep Sets and the other trained Linformers
  at L-st-s0's pairs and blocks ("transfer"). JEDI-linear's edge 13|14 vs the flats 15|16, 19|20, 20|21, 22|23, 23|24 (boundaries of
  no S or P zero-bit window in E20).

## H0 / H1 and predictions
- **H0 (no learned rank windows)**: trained R ≈ untrained R and ≈ Deep Sets R. E edges do not recur across seeds beyond chance. Across-edge
  shuffles do not exceed inside shuffles any more than in Deep Sets.
- **H1 (rank windows are a general learned element)**: trained R above the untrained q95. Edges recur across seeds and axis
  conventions. Across-edge shuffles move more calls than inside shuffles, and Deep Sets does not show this.
- **Mechanical alternative (H_lin)**: swapping two tokens changes the projected keys/values by about $(E_i-E_{i+1})\cdot(\text{token change})$, so R > 1
  for any E, trained or random, in proportion to its roughness. This predicts R > 1 in untrained nets as well.

## Numbers
**Step 1: E profile and recurrence**

| comparison | top-5 exact overlap (chance 0.40) | within ±1 (chance) | profile Spearman (perm p) |
|---|---|---|---|
| L-st-s0 vs L-st-s1 (seeds) | 0 (p 1.0) | 2 (1.05; p 0.28) | −0.03 (0.83) |
| L-ax-s0 vs L-ax-s1 (seeds) | 2 (p 0.048) | 3 (0.89; p 0.035) | 0.24 (0.059) |
| L-st-s0 vs L-ax-s0 (same seed, other axis) | 2 (p 0.048) | 3 (0.65; p 0.013) | 0.73 (< 2×10⁻⁴) |
| L-st-s1 vs L-ax-s1 (same seed, other axis) | 4 (p < 10⁻³) | 4 (0.89; p 0.003) | 0.90 (< 2×10⁻⁴) |
| L-st-s0 vs L-ax-s1, L-st-s1 vs L-ax-s0 | 1, 1 (p 0.35) | 1, 1 | 0.06, 0.12 (0.64, 0.36) |
| each trained L vs its own initial E | 2–3 | 2–5 | 0.63–0.80 (all < 2×10⁻⁴) |
| 10 untrained pairs | 0–1 | 0–3 | −0.12 to 0.12 |

- Top-5 edges: L-st-s0 {62, 60, 3, 32, 9}; L-st-s1 {23, 24, 10, 2, 28}; L-ax-s0 {2, 3, 1, 60, 61}; L-ax-s1 {2, 23, 24, 3, 28}.
- Trained E vs its own initial E: correlation of entries 0.42–0.71 (per block). ‖ΔE‖/‖E_init‖ = 0.79–1.05, so training changed E by about
  its own size, but the jump pattern still follows the initial one. Cross-axis agreement holds only for the same seed, and the same seed means the
  same initial E and batch order. The learned change ΔE agrees the same way (ρ 0.74 and 0.89 same seed; −0.07 to 0.01 across seeds).
- Largest edge score: untrained max 1.57 (all 5 nets, all boundaries). Trained L-st 1.56 / 1.60. **L-ax-s0 1.90 and L-ax-s1 1.96, both at
  boundary 2|3** (L-ax-s0 also 1.78 at 1|2 and 1.81 at 3|4). There are 3 boundaries above the untrained max in each hardest-axis net, and 0–1
  in the stored-axis nets. At these boundaries, the jump in ΔE alone is not large (0.87–1.19). We did not test whether the 2|3 jump changes the output beyond
  what the general E-roughness relation predicts (see step 2).

**Step 2: angle swap at top-5 E edges vs matched flats** (R for mean |Δlogit|, 95 % jet-bootstrap CI; permutation p over 252 labellings)

| network | edges / flats | R (|Δlogit|) | perm p | calls changed edge vs flat (R) |
|---|---|---|---|---|
| L-st-s0 | own | 1.19 [1.13, 1.25] | 0.37 | 0.61 vs 0.57 % (1.07 [0.79, 1.43]) |
| L-st-s1 | own | 1.76 [1.69, 1.84] | 0.052 | 0.67 vs 0.41 % (1.65 [1.31, 2.12]) |
| L-ax-s0 | own | 2.08 [1.97, 2.19] | 0.052 | 0.98 vs 0.37 % (2.64 [2.04, 3.46]) |
| L-ax-s1 | own | 1.81 [1.72, 1.90] | 0.16 | 0.74 vs 0.41 % (1.82 [1.48, 2.31]) |
| U0–U4 (untrained) | own | 1.89, 2.60, 1.18, 2.58, 1.57 (q95 2.59) | 0.012–0.27 | R 1.03–3.17 |
| L-st-s1 / L-ax-s0 / L-ax-s1 | L-st-s0's | 0.99 [0.94, 1.04] / 1.08 [1.02, 1.14] / 1.09 [1.04, 1.15] | 0.41–0.52 | |
| D-s0 / D-s1 (Deep Sets) | L-st-s0's | 0.81 [0.74, 0.89] / 0.81 [0.74, 0.88] | 0.60–0.62 | 0.32 vs 0.54 % |
| U0–U4 | L-st-s0's | 0.73–2.16 | 0.07–0.87 | |
| JEDI-linear | 13|14 vs 5 flats | 1.94 [1.88, 2.01] | (edge ranks 1st of 6) | 1.41 vs 0.64 % (2.19 [1.69, 2.79]) |

- Per-edge tests (Holm over 5): in the trained Linformers, 4 or 5 of the 5 edges differ from their flat partner on |Δlogit| (Holm p < 0.001).
  In the untrained nets all 5 do. For JEDI-linear, 13|14 exceeds each of the 5 flats: |Δlogit| difference 0.019–0.040, Holm p < 0.001;
  McNemar on calls Holm p 0.024 (vs 15|16) and ≤ 7×10⁻⁶ (others).
- Swap response vs edge score over all 62 boundaries (Spearman, raw / partial given rank): trained L 0.63/0.68, 0.42/0.48, 0.64/0.66,
  0.49/0.52 (all perm p ≤ 0.002). Untrained 0.22–0.37 / 0.24–0.44. At L-st-s0's edge scores, Deep Sets gives 0.16 / 0.00–0.10 (p ≥ 0.19).
  With ΔE instead of E: trained 0.02–0.22 / 0.20–0.40.

**Step 3: shuffle across an E edge vs inside a window** (6 particles moved in both; ratio of mean |Δlogit|, across/inside)
- L-st-s0 (its edges 60, 3, 32, 9): 2.88 [2.75, 3.02], 1.82 [1.72, 1.92], 1.07 [1.03, 1.10], 0.58 [0.54, 0.61]. Calls
  1.54/0.51, 8.86/4.24, 1.11/0.94, 2.17/4.13 %.
- Deep Sets at the same blocks: D-s0 0.98, 1.75, 0.92, 0.62; D-s1 0.95, 1.73, 0.91, 0.63.
- Only the high-rank edge 60 differs from Deep Sets (L-st-s0 2.88, L-ax-s0 2.27 vs 0.95–0.98). At edges 3 and 9 the Deep Sets ratio is the same, so
  those differences come from the rank region (where the blocks sit), not from E.
- L-st-s1 (23, 24, 10, 28): 0.98, 0.96, 0.65, 0.86. L-ax-s1 (23, 24, 3, 28): 0.89, 0.91, 1.75, 1.06. Untrained: 0.56–1.69.

## Reading (scoped to what was measured)
- Swapping two neighbouring particles' angles changes the Linformer output more where E jumps (R 1.19–2.08 in the trained nets, response
  correlated with the edge score over all 62 boundaries). The same holds in randomly initialised Linformers (R 1.18–2.60). This is what the
  mechanical relation H_lin predicts. No trained R exceeds the untrained q95 (2.59). H1's first prediction is not met. H0's "trained ≈ untrained"
  holds, but its "R ≈ 1" does not, because random E is also rough.
- Edge positions do not transfer between seeds. At L-st-s0's pairs the other trained Linformers give R 0.99–1.09 and Deep Sets 0.81, and the
  stored-axis top-5 sets do not overlap across seeds (0 of 5; profile ρ −0.03). Same-seed agreement across axis conventions (ρ 0.73, 0.90)
  goes with a shared initial E, and trained E stays correlated with its initial E (0.42–0.71).
- The window-shuffle asymmetries at edges 3 and 9 are reproduced by Deep Sets, which has no rank dependence. They reflect where the blocks
  sit in rank, not E. Edge 60 in the seed-0 nets is the one case where across exceeds inside and Deep Sets does not show it.
- Both hardest-axis Linformers have their largest E jump at 2|3 (1.90, 1.96), above every untrained boundary (max 1.57). The axhard seed
  overlap (2/5, p 0.048) is this boundary plus its neighbour. With the hardest particle at the origin of the input coordinates, the first
  few ranks form a recurring rank region in E. This was not followed up causally beyond the step-2 scan.
- JEDI-linear's 13|14 swap response is 1.94× its flats on this sample, consistent with E20. This lies inside the untrained-Linformer range of
  R. The two architectures differ, so this comparison only places the numbers side by side.

## Competing hypotheses
- **H1 (general learned windows)**: excluded at this test's resolution for stored-axis Linformers. Trained R is not above the untrained q95,
  there is no cross-seed recurrence (0/5, ρ −0.03), and the step-3 effects match Deep Sets except at edge 60.
- **H_lin (swap response follows E roughness, trained or random)**: consistent. R > 1 in 9/9 Linformers. Per-boundary response correlates
  with the edge score in trained and untrained nets, and not with another net's edge score (transfer R ≈ 1).
- **"Windows only through shared initialisation"**: consistent with same-seed cross-axis agreement and with E–E_init correlation.
- **Hardest-axis low-rank jump (2|3)**: open. It recurs in 2/2 axhard seeds above the untrained max, with p 0.048 for the top-5 overlap.

## Issues / caveats
- The per-network permutation test over 252 labellings has low power (min p 0.004). Most per-network p are 0.05–0.4, including for untrained nets.
- There are only 5 untrained nets, so q95 ≈ the max.
- Untrained call changes use a random head (accuracy 16–21 %). Their call rates (1–17 %) are not comparable to trained nets; the |Δlogit|
  ratio is scale-free.
- Sample accuracy is 0.6–1.0 points below full-test accuracy (sampling; the outputs match the stored test probabilities).
- The inside blocks in step 3 can sit next to the edge (e.g. 55–60 for edge 60), and they lie in a different rank region than the across
  block. That is why Deep Sets is the needed reference.

## Explainer entry (for site/explain.py; Paper A, claim A1 / A10)
- **Question**: is a pT-rank window (JEDI-linear's S0 = ΣpT of ranks 0–13) something networks that see pT-sorted particles learn in general?
- **Background**: a Linformer compresses its 64 particle tokens into 8 summaries with a learned matrix $E\in\mathbb R^{64\times8}$,
  $\tilde K_j=\sum_n E_{nj}K_n$. Row $E_n$ is the weight of the rank-$n$ particle. A "window edge" is a boundary where $E_{n+1}$ differs strongly from $E_n$.
- **Method**: edge score $e(i)=\|E_{i+1}-E_i\|/\mathrm{median}$. Causal edit: swap the $(\Delta\eta,\Delta\phi)$ of the rank-$i$ and rank-$(i{+}1)$
  particles, keeping their pT (pT ordering unchanged). $R$ = mean $|\Delta\text{logit}|$ at the 5 largest-$e$ boundaries ÷ that at 5 matched
  flat boundaries. Controls: 5 untrained Linformers, Deep Sets, the other seed's edges.
- **Result (figure x15_linformer_ranks.png)**: trained $R$ = 1.19–2.08, untrained 1.18–2.60. Edge positions do not recur between seeds.
  The two hardest-axis networks share a large jump at 2|3.
- **Checks**: exact match of unedited outputs to the stored test probabilities. Deep Sets reproduces the rank-region part of the window-shuffle effect.
- **Why it matters**: JEDI-linear's rank windows come from its zero-bit masks (X8) and are specific to the official checkpoint (X12). In a
  Linformer the rank weighting also exists, but its jump positions follow the random initialisation, not a reproducible learned window.
- **Glossary additions**: Linformer projection E; edge score; angle swap; edge/flat ratio R.
