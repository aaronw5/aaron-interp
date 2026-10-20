# P9 verification (P9v): ParT_full W-vs-Z whole-network circuits, 2026-10-20

Jobs:
- `anrunw-ai-c-p9-verify` (V100, 757 s): script `interp/p9v_verify_part.py`. This job was launched by an earlier verifier
  instance of this node, and its report was never written. Its outputs are used here after the script was read in full.
  Results: `results/part/p9_verify.json`, `p9_verify.png`, `logs/c-p9-verify.log`.
- `anrunw-ai-c-p9-verify2` (RTX 2080 Ti, 1,988 s): script `interp/p9v2_verify_part.py`. Results: `results/part/p9_verify2.json`,
  `p9_verify2.png`, `logs/c-p9-verify2.log`.

Both jobs freeze P9's graph, MLP-unit grouping and EAP-IG scores (`p9_eap_scores.npz`; the graph and node ranking are asserted
to match). Neither job uses any of P9's rows.

## Conclusion
The search for a small circuit that keeps ParT_full's W-versus-Z decision reproduces on new jets, and attribution-ranked
circuits recover the same share of the decision as in P9. The search method matters, though. A mask trained directly to keep
the decision, using 10 % of the connections, scores 0.98 on P9's faithfulness measure (a summed logit difference), but it
calls W vs Z correctly on only 63 % of jets, against 83 % for the model. "Faithfulness ≥ 0.8 needs over a third of the
network" is therefore not a property of the network: it depends on the search method. The supported statement is narrower:
no circuit found with up to 10 % of the connections reproduces the model's W/Z calls. A better search could still change this.

## Code review of `interp/p9_eap_wz_part.py`
- **Layer and head indexing.** The class blocks use an un-normalised query and normalised keys over [cls, particles].
  w_resid is applied after the attention residual and before the MLP output. c_attn is folded into the out_proj rows with
  the 'tbhd->tbdh' shuffle. Clean logits match weaver to 7.3e-6 (V100) and 7.9e-6 (2080 Ti). Masks = 1 reproduce the clean
  logits to 5.8e-6. The 4.6e-3 in P9 comes from the Ada GPU only.
- **Padding.** Each reader uses the clean jet's key-padding mask. Donor activations come from the donor's own forward pass
  with the donor's mask. Padded slots are never scored.
- **Rows.** All samples use random rows (sorted indices from `default_rng`). eval[::10] is taken on a W-then-Z concatenation,
  so it contains 500 jets of each class. The data are not class-ordered first-k.
- **η build.** The jets go through `part_core.build` with the sign(jet_eta) fix. The P1-sample accuracy of 85.93 % matches.
- **Weak checks and controls found in the code.**
  - The "all masks = 0 reproduces the donor (0.0)" check is trivial: the z coordinates are masked directly
    (`z = cz['z'] + M('z')·(z − cz['z'])`). The check therefore never exercises the upstream masks.
  - P9's matched random circuits draw z coordinates at random, so they almost never contain n98 (z97). Their F of about 0 is a
    control for "contains the read-out coordinate", not for the internal nodes. See control 3 below.
  - The 3.6× "underestimate" in the single-edge fidelity test is expected. Integrated gradients split the credit of a serial
    path among its edges, while removing a single edge from the full model takes away the whole path. For example, ∂D/∂m on
    z97→logits is proportional to z97(α) − z97^corr, which is small at small α. The ratio describes the attribution method.
    It does not show that the method under-measures.
- No indexing, hook or sampling bug was found.

## Re-run 1: fresh sample, P9 ranking frozen (job c-p9-verify)
- **Sample.** WToQQ_120 / ZToQQ_120.root, 3,000 + 3,000 random rows, rng 9090.
- **Donors.** B uses P9's rule (rng 31). Bm is opposite-class, chosen among the 30 nearest in pT as the 3 nearest in
  multiplicity (median |Δmult| 1 vs 9 for B). Bm tests P9 issue 4, the slot pairing between jets of different multiplicity.
- **Unedited model.** W/Z 83.17 %; everything resampled 17.6 %.

| edges kept (top-k by P9 EAP-IG) | P9 (files _123) F | fresh B: F [95 % CI] | fresh Bm: F | fresh B W/Z acc |
|---|---|---|---|---|
| 10 | 0.098 | 0.097 [0.091, 0.103] | 0.092 | 23.1 % |
| 100 | 0.238 | 0.238 [0.230, 0.247] | 0.215 | 34.6 % |
| 1,075 (5 %) | 0.513 | 0.512 [0.502, 0.522] | 0.497 | 55.3 % |
| 2,000 | 0.604 | 0.602 | 0.594 | 63.0 % |
| 4,000 | 0.722 | 0.719 | 0.716 | 70.3 % |
| 6,000 | – | 0.788 [0.780, 0.796] | 0.791 | 75.6 % |
| 7,000 | – | 0.823 | 0.826 | 77.6 % |
| 8,000 | 0.856 | 0.856 [0.849, 0.863] | 0.860 | 79.3 % |

- **F per doubling of the edge count.** Least squares over 10–8,000 edges gives 0.082 [0.081, 0.083], not 0.12. The slope
  rises with size: 0.043 per doubling for 10–100 edges, 0.079 for 100–1,000 and 0.118 for 1,000–8,000. The curve is convex
  in log-size.
- **Size for F = 0.8.** Log-linear interpolation gives 6,318 edges [6,096, 6,542], which is 29 % of the edges, not 37 %. By
  nodes it is 182 nodes (42 %), not 256 (59 %). P9 quoted the first grid point above 0.8.
- **B vs Bm (paired bootstrap).** ΔF = −0.015 [−0.023, −0.006] at 1,075 edges and +0.004 [−0.001, 0.009] at 8,000. The
  multiplicity-matched donors leave the curve unchanged within 0.02.
- **Ranking controls.**
  - Top-k by plain EAP is below EAP-IG: F 0.42, 0.63 and 0.76 at 1,075, 4,000 and 8,000 edges.
  - k uniformly random edges give F ≤ 0.007 at 1,075 and 4,000 edges. At 8,000 edges, one draw of three gives 0.11.
- **ACDC literal circuits reproduce.** The 89-, 27- and 10-node circuits give F 0.561, 0.272 and 0.168 (P9: 0.564, 0.273,
  0.169), with W/Z 58.4 %, 36.2 % and 28.8 %.

## Re-run 2: attribution stability, a stronger search, anchored random controls (job c-p9-verify2)
- **Sample.**
  - Search set T: WToQQ/ZToQQ_121.root, 2,500 + 2,500 random rows, rng 6161.
  - Evaluation set V: WToQQ/ZToQQ_122.root, 2,500 + 2,500 random rows, rng 6262.
  - Donors B: rng 63 on T, rng 64 on V.
- **Unedited model on V.** W/Z 82.90 %; everything resampled 17.5 %.

**1. Attribution stability.**
- Procedure: EAP-IG recomputed on T with P9's 5 α steps.
- Spearman of |IG_T| against |IG_P9| over 21,504 edges: 0.912. Overlap of the top-k sets: 100 % at k = 10, 97 % at 100,
  93 % at 1,075 and 92 % at 4,000. Completeness 1.025.
- Top-k by IG_T on V: F 0.517 at 1,075 edges, 0.712 at 4,000 and 0.857 at 8,000. P9's ranking gives 0.513, 0.719 and 0.854
  on V.

**2. Stronger search at fixed size (missing control for "no small circuit").**
- **Procedure.**
  - Soft top-K edge mask \(m_e = \sigma((\theta_e - \tau_K)/T)\), with \(\tau_K\) the K-th largest \(\theta\) (detached).
  - \(\theta\) starts at the standardised \(\log|\mathrm{IG}_{P9}|\). T is annealed from 0.5 to 0.02.
  - Adam, lr 0.02, 800 steps on T. Loss \(= -\sum_j s_j(d_j(m) - d_j^{corr}) / \sum_j s_j(d_j - d_j^{corr})\), which is P9's F,
    over 4 random batches per step.
  - After training, the binary top-K edges by \(\theta\) are kept and evaluated on V.
- **Results.** P9's ranking at the same K is in brackets.

| K | F on V [95 % CI] | W/Z acc on V | F on T[::2] | overlap with P9 top-K | literal (node-induced) circuit: nodes / F / W/Z acc |
|---|---|---|---|---|---|
| 1,075 (5 %) | 0.715 [0.693, 0.737] (0.513) | 58.2 % [57.0, 59.7] (53.5 %) | 0.750 | 49 % | 179 / 0.715 / 69.2 % |
| 2,150 (10 %) | 0.977 [0.932, 1.020] (0.612) | 63.1 % [61.9, 64.5] (63.0 %) | 1.042 | 50 % | 269 / 0.888 / 79.1 % |

- **Paired bootstrap of ΔF against P9's top-K.**
  - K = 1,075: +0.202 [0.184, 0.221], p = 0.001 (bootstrap floor).
  - K = 2,150: +0.365 [0.325, 0.403], p = 0.001.
- **Tests at F = 0.8.** At K = 1,075, F ≥ 0.8 is excluded (bootstrap fraction ≥ 0.8 = 0.001). At K = 2,150, F ≥ 0.8 is not
  excluded (fraction 1.0).
- **Accuracy at K = 2,150.** The W/Z accuracy of the optimised circuit equals that of P9's top-2,150 (63.1 vs 63.0 %;
  intervals overlap fully). F rises from 0.61 to 0.98 with no gain in correct calls. The optimised mask raises the summed
  s·Δd mostly by moving d further on jets already called correctly, not by changing calls. This is inferred from the
  aggregate numbers; a per-jet breakdown was not saved. The F gap between T (1.04) and V (0.98) is small, so this is not
  overfitting to T.

**3. Anchored random controls for the ACDC literal circuits.**
- **Procedure.**
  - Anchor = E plus every class-stage node (C\*) and z coordinate of the circuit.
  - The particle-stage nodes (pair biases, particle heads, particle MLP groups) are replaced by random nodes of the same type
    and block, 20 draws.
  - "z97rand" keeps only E and z97 and draws every other node at random, per type and block.
  - Permutation p = (1 + #draws ≥ circuit)/21; the floor is 0.048.

| circuit | F on V | anchor alone | particle nodes random: mean [range of 20] | E + z97 kept, rest random: mean (max) | p |
|---|---|---|---|---|---|
| 89-node ACDC | 0.560 [0.547, 0.571] | 0.249 | 0.384 [0.363, 0.416] | 0.166 (0.324) | 0.048 / 0.048 |
| 27-node ACDC | 0.276 [0.267, 0.286] | 0.235 | 0.250 [0.239, 0.271] | – | 0.048 |

- With the read-out kept, random particle-stage nodes recover F 0.38. The 89-node circuit's own particle nodes add a further
  0.18 above that.
- For the 27-node circuit, most of the F (0.235 of 0.276) comes from the read-out anchor alone. Its 6 particle heads and 1
  pair bias add 0.04 over the anchor and 0.03 over random replacements.
- P9's random control (F mean 0.014) did not keep n98 and so measured mostly the presence of n98.

## Hypotheses and which are excluded
- **H_loc (P9's H1: F ≥ 0.8 and W/Z within 3 points at ≤ 5 % of edges).** Excluded by both searches. At 5 % the best F is
  0.715 [0.693, 0.737], with W/Z 58 %.
- **H_dist as P9 states it (F rises smoothly at ~0.12 per doubling; 80 % needs > 1/3 of the network).**
  - The slope is 0.08 and grows with size.
  - F = 0.8 is reached at 29 % of the edges with attribution ranking, and at ≤ 10 % with an optimised mask (F 0.98).
  - The "> 1/3" part is excluded for the F metric.
- **H_call (no circuit with ≤ 10 % of edges reproduces the W/Z calls).** Not excluded by any search run here.
  - Best edge circuits: 58 % (5 %) and 63 % (10 %), against 83 %.
  - Literal node circuits need 269 nodes (62 %) for 79 %.
- **H_overshoot (summed F can be raised without more correct calls).** Supported at K = 2,150: F 0.61 → 0.98 while accuracy
  stays at 63.0 → 63.1 %.

## Assessment of the P9 reading, item by item
- "rises by about 0.12 per doubling of the edge count, from 10 to 8,000 edges": incorrect. The least-squares slope is 0.079
  on P9's own numbers and 0.082 [0.081, 0.083] on the fresh sample. Segment slopes rise from 0.04 to 0.12.
- "F ≥ 0.8 is first reached at 8,000 edges (37 %) or 256 nodes (59 %)": grid points. The interpolated values are 6,300 edges
  (29 %) and 182 nodes (42 %), and these hold only for attribution ranking. An optimised 2,150-edge mask reaches F 0.98.
- "At 5 % of edges, F is 0.51 and W/Z accuracy 28.6 points below the model": reproduces (0.512–0.513) for the attribution
  ranking. The optimised 5 % mask gives F 0.715 and W/Z 24.7 points below the model.
- "Largest attributions at the read-out (n98 direct path 18 %, C0H1, C0H2, class MLP group 0)": reproduces. On T, z97 → logits
  is 0.180, C0M0 → OUT 0.060 and C0H1 → OUT 0.054.
- "ACDC circuits well above matched random circuits": holds against P9's control (F 0.56 vs 0.01). Against random circuits
  that keep the read-out anchor, the margin is 0.56 vs 0.38 for the 89-node circuit and 0.28 vs 0.25 for the 27-node one.
- "Edge attributions rank single-edge effects correctly but underestimate their size by 3.6×": the rank part is supported
  (Spearman 0.87). The ratio follows from how integrated gradients split credit along serial paths and is not a measurement
  error.
- The Conclusion's "ParT_full has no small internal circuit" is an existence claim. The evidence is limited to the searches
  run: attribution ranking, ACDC from a top-2,000 start, and here a gradient-trained top-K mask.

## Corrected reading
Under opposite-class, pT-matched resampling, the share of ParT_full's W-vs-Z logit difference recovered by the top-k
attribution-ranked edges grows convexly in log k. The slope is 0.04 per doubling at 10–100 edges and 0.12 at 1,000–8,000
(0.08 overall). F = 0.8 is reached near 6,300 edges (29 %) or 182 nodes (42 %). This reproduces on fresh files (_120, _122)
and with multiplicity-matched donors (|ΔF| ≤ 0.02). The curve depends on the search method. A gradient-trained top-K mask
reaches F 0.72 at 5 % and 0.98 at 10 % of the edges. Its W/Z accuracy stays at 58 % and 63 % (model 83 %), the same accuracy
as the attribution-ranked circuit at 10 %. Summed F therefore overstates how well a circuit reproduces the calls. The
attributions are stable across samples (Spearman 0.91) and concentrate at the read-out: n98's direct path, the block-1 class
heads and the first MLP group of each class block. Against random circuits that keep this read-out, the 89-node ACDC
circuit's particle-stage nodes add 0.18 in F, and the 27-node circuit's add 0.03–0.04.

## Remaining caveats
- The optimised masks ran for 800 steps from the IG initialisation, with one temperature schedule. Longer training or an
  accuracy-based objective could give smaller circuits.
- The per-jet distribution of Δd for the optimised mask was not saved, so the overshoot statement rests on aggregate F
  against accuracy.
- The 10-class read and corruption A were not re-run.

## Explainer additions (for site/explain.py)
- **Check: does the curve depend on the search?** Train a soft top-K mask
  \(m_e=\sigma((\theta_e-\tau_K)/T)\) to maximise \(F=\frac{\sum_j s_j(d_j(m)-d_j^{corr})}{\sum_j s_j(d_j-d_j^{corr})}\). With 10 % of
  edges, F = 0.98, but W/Z accuracy stays at 63 % (model 83 %). F is a sum over jets and can grow by enlarging \(d\) on jets
  already called correctly. Report accuracy next to F.
- **Check: is the circuit better than random once the read-out is kept?** Keep E, the class-stage nodes and the z coordinates,
  and draw the particle-stage nodes at random. The 89-node circuit scores 0.56, random draws 0.36–0.42, and the read-out alone
  0.25 (figure p9_verify2.png).
