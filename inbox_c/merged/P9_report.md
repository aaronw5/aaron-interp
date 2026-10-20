# P9: whole-network W-vs-Z circuit discovery in ParT_full (EAP-IG + ACDC), agent report 2026-10-19

Job `anrunw-ai-c-p9-run` (NVIDIA RTX 5000 Ada, 3,715 s), `python -m interp.p9_eap_wz_part`. Smoke run `c-p9-smoke` (RTX 2080 Ti):
the first attempt ran out of GPU memory in the gradient pass, and the second passed after the batch size for gradient passes was
reduced. Post step `c-p9-post` (job.sh, CPU) recomputes the fidelity p-values (see Issues) and renders the figure.
Script `interp/p9_eap_wz_part.py`. Results `results/part/p9_eap_wz.json`, scores `results/part/p9_eap_scores.npz`, figure
`results/part/p9_eap_wz.png`, logs `logs/c-p9-run.log` and `logs/c-p9-smoke.log`.

## Conclusion
We searched the whole of ParT_full for the smallest set of internal connections that reproduces its W-versus-Z decision. The
search covered attention heads, MLP-unit groups, pair-bias inputs and class-token coordinates. Everything outside a candidate
set was replaced by values from a jet of the other class. No small set works. The top 5 % of connections (1,075 of 21,504)
recover 51 % of the full model's W-vs-Z score difference, and 80 % needs about 8,000 connections (37 %). This growth follows
the "distributed decision" alternative, not a localized circuit. The found circuits are well above random circuits of the same
size and layout. They contain the first class-attention block's heads, a few MLP groups and the W/Z coordinate n98. Open
questions: whether coarser units (whole blocks, learned features) would give a short description, and how much the steady
growth depends on the resample corruption chosen.

## Procedure
- **Model.** ParT_full, official weights, weaver-core 0.4.17.
- **Re-implementation.** The forward pass is re-implemented as a graph of 433 writer nodes and 21,504 edges, each edge with a
  mask in [0, 1].
  - Nodes: E (particle embedding); 64 pair-bias inputs P{k}B{h}; 64 particle heads P{k}H{h}; 128 particle MLP groups P{k}M{g};
    16 class heads C{b}H{h}; 32 class MLP groups C{b}M{g}; 128 coordinates z{i} of the final class token.
  - Each MLP group is 32 hidden units (post_fc_norm outputs). In each block the units are ranked by |gradient × activation
    difference| on 1,000 discovery jets, and consecutive ranks form a group (group 0 = top 32).
  - Every reader (head, MLP group, final LayerNorm, logits) gets its own copy of the residual stream.
  - post_attn_norm mixes a block's heads. A reader therefore sees, for each earlier block,
    \(\mathrm{LN}(\sum_h [m\,y_h + (1-m)\,y_h^{corr}] + b)\). The w_resid factors are applied exactly.
- **Checks on the re-implementation.**
  - All masks = 1 reproduces clean logits to 3.8e-6.
  - All masks = 0 reproduces the donor's logits exactly (0.0).
  - Against the weaver model: 4.6e-3 max |Δlogit| on the Ada GPU in the run (4.8e-6 on the 2080 Ti in the smoke).
- **Sample.** JetClass val_5M, random rows. Rows are never split by first-k.
  - Discovery: 5,000 W + 5,000 Z from WToQQ/ZToQQ_123.root (rng 2026).
  - Evaluation: 5,000 + 5,000 from the same files, disjoint from discovery (rng 2027; overlap 0).
  - All-class set: 2,000 per class from <sample>_124.root (rng 2028).
  - P1 sample: test files, rows 3,000–4,000, accuracy check only.
- **Corruptions (resample ablation).** Ablated edges carry the donor jet's clean value, slot by slot for particle nodes.
  - B (main): a donor of the other class (W↔Z), one of the 5 nearest in jet pT. Median |ΔpT|/pT 1.2e-4.
  - A: a donor of the same class with |Δm| < 5 GeV and nearest multiplicity. Median |Δm| 2.5 GeV, median Δmultiplicity 0.
  - U: a uniform random other jet of the all-class set. Same class 9.8 % of the time; this is P3's uniform_S convention.
- **Metric.** \(d = \ell_W - \ell_Z\), with \(s = \pm 1\) for true W/Z.
  - Faithfulness \(F = \sum_j s_j (d^{circ}_j - d^{corr}_j) / \sum_j s_j (d^{full}_j - d^{corr}_j)\).
  - W/Z accuracy = argmax over (\(\ell_W, \ell_Z\)) on true W/Z jets.
- **Step 1, EAP-IG over edge masks.** \(\mathrm{score}_e = \frac{1}{5}\sum_{\alpha \in \{.1,.3,.5,.7,.9\}} \partial D/\partial m_e |_{m=\alpha}\),
  where \(D = \sum_j s_j d_j\), on all 10,000 discovery jets. Plain EAP (\(\alpha = 1\)) is also computed. Node score = the sum
  of the node's outgoing edge scores.
- **Step 2, ACDC.**
  - Start from the top 2,000 edges by |score|, with every other edge ablated.
  - On 300 discovery jets with cached donors, visit edges in reverse topological order and remove an edge if F drops by less
    than \(\tau\).
  - Nested passes at \(\tau\) = 0.001, 0.003 and 0.01.
- **Evaluation on the 10,000 evaluation jets (B donors).**
  - F and W/Z accuracy for the top-k edges, and for the top-k nodes as node-induced literal circuits (every edge between kept
    nodes kept, all else resampled).
  - The same for the ACDC circuits, both as edge circuits and as their literal node circuits.
  - Reference circuits from P3/P3v, both literal (whole network) and in P3's scope (particle stage, embedding and pair bias
    intact).
- **Controls.**
  - 100 random circuits per ACDC circuit on eval[::10] (1,000 jets). Edge circuits are matched per (writer type+block,
    reader type+block) category; node circuits per node type+block.
  - Permutation p = (1 + #random ≥ circuit)/101.
  - McNemar test of each circuit's W/Z correctness against the full model.
  - Attribution fidelity: single-edge resample of the top 50 edges and of 50 random edges (ranks 51+) on eval[::10], compared
    with the EAP-IG and EAP predictions. Normal-approximation p from the bootstrap s.e., Holm over 100.
- **10-class read.** The 20,000 all-class jets with U donors: accuracy, confusion and W/Z binary accuracy.
- **Statistics.** Bootstrap over jets, 1,000 resamples, 95 % intervals.
- **Hypotheses.**
  - H0 (distributed): no circuit with ≤ 5 % of edges reaches F ≥ 0.8.
  - H1 (localized): such a circuit reaches F ≥ 0.8 with W/Z accuracy within 3 points of the model.

## Numbers
**Unedited model.**
- P1 sample 85.93 % (expected 85.9). Evaluation W/Z binary 83.25 %. All-class 10-class 86.13 % [85.66, 86.65], W/Z binary 83.85 %.
- With everything resampled from B donors, W/Z accuracy is 16.8 %. Calls follow the donor, so this is the floor under B, not
  50 %.

**EAP-IG.** D(1) − D(0) = 4.88 per jet. The IG scores sum to 4.998 (completeness ratio 1.024). Share of total |score|: top 100
edges 0.57, top 1,000 0.78, top 5,000 0.93. Spearman of |IG| against |EAP| over all edges is 0.72. Largest edges, as a share of
D(1) − D(0):

| edge | share |
|---|---|
| z97 → logits | +0.182 |
| C0M0 → OUT | +0.059 |
| C0H1 → OUT | +0.058 |
| C1M0 → OUT | +0.050 |
| z81 → logits | +0.041 |
| E → C0H2 (b1h3) | +0.029 |

The largest node scores are z97 0.18, E 0.16, C0H1 (b1h2) 0.070, C0M0 0.052, C1M0 0.050 and C0H2 (b1h3) 0.043. The largest
particle-stage node is P7M0 at 0.012.

**Faithfulness vs size (evaluation set, B donors; F with 95 % CI, W/Z accuracy).**

| edges kept | F | W/Z acc |
|---|---|---|
| 10 | 0.098 [0.094, 0.102] | 23.5 % |
| 100 | 0.238 [0.231, 0.244] | 34.3 % |
| 1,000 | 0.501 [0.494, 0.508] | 53.5 % |
| 1,075 (5 %) | 0.513 [0.506, 0.520] | 54.7 % |
| 2,000 | 0.604 [0.597, 0.612] | 63.3 % |
| 4,000 | 0.722 [0.715, 0.730] | 70.6 % |
| 8,000 | 0.856 [0.851, 0.861] | 79.6 % |
| 15,000 | 0.985 [0.983, 0.988] | 82.8 % |

| nodes kept (literal) | 8 | 16 | 32 | 64 | 128 | 256 | 433 |
|---|---|---|---|---|---|---|---|
| F | 0.148 | 0.213 | 0.322 | 0.492 | 0.651 | 0.954 | 1 |
| W/Z acc | 27.2 % | 31.4 % | 36.6 % | 50.8 % | 65.3 % | 82.2 % | 83.25 % |

- From 10 to 8,000 edges, F rises by about 0.12 per doubling of the edge count.
- First size reaching F ≥ 0.8: 8,000 edges, or 256 nodes.
- At 5 %: F 0.513, bootstrap p(F ≤ 0.8) = 1.0. The W/Z accuracy drop is 28.6 points. H0 is not rejected; H1's prediction is
  excluded by every interval.

**ACDC circuits** (start: top 2,000 edges, F = 0.60 on the 300 ACDC jets).

| τ | edges | nodes | F edge circuit (eval) | F literal (eval) | W/Z acc literal | random literal F mean / max | p |
|---|---|---|---|---|---|---|---|
| 0.001 | 223 | 89 | 0.385 [0.379, 0.392] | 0.564 [0.556, 0.572] | 58.9 % | 0.014 / 0.250 | 0.0099 |
| 0.003 | 48 | 27 | 0.241 | 0.273 | 36.0 % | 0.003 / 0.169 | 0.0099 |
| 0.01 | 15 | 10 | 0.151 | 0.169 | 28.9 % | 0.001 / 0.049 | 0.0099 |

- All edge circuits also have p = 0.0099 against matched random circuits (random F mean ≤ 0.007, max ≤ 0.10).
- McNemar against the full model on W/Z correctness, 89-node literal circuit: 2,922 jets right → wrong, 484 wrong → right,
  p < 1e-300.
- The 10-node circuit is {E, P6H7, C0H1, C0H2, C0H5, C0M0, C1M0, z81, z97, z124}. P6H7 is particle block 7, head 8, one of
  G101's far-side heads.
- The 27-node circuit adds all 8 block-1 class heads (b1h1–b1h8), b2h8, MLP groups C0M0, C0M1, C1M0 and C1M2, six particle
  heads (one each in blocks 1, 2, 5, 6, 7, 8), one pair bias (P4B7) and z25, z52, z81, z97, z124.
- The 89-node circuit contains 38 particle heads, 13 pair biases and 10 particle MLP groups spread over all 8 blocks, plus 19
  class-stage nodes, 8 z coordinates and E.
- Node-ranked literal circuits of the same size give nearly the same values: F 0.586 at 89 nodes, 0.292 at 27, 0.172 at 10.

**Reference circuits (P3/P3v).**

| circuit | B donors, eval: F / W/Z acc | U donors, all-class: W/Z acc / 10-class acc |
|---|---|---|
| literal {b2h8, b1h3, n98} | 0.006 / 17.1 % | 50.4 % / 9.8 % |
| literal 16 class heads + n98 | 0.015 / 17.6 % | 50.1 % / 9.8 % |
| P3 scope {b2h8, b1h3, n98} | 0.047 / 19.3 % | 52.1 % / 9.9 % |
| P3 scope 16 heads + n98 | 0.367 / 44.3 % | 63.6 % / 11.0 % |

- The two P3-scope rows reproduce P3's V_a (52.9 %) and P3v's 16-head circuit (63.0 %).
- The literal references sit above size-matched random sets of the same node types (p 0.03 and 0.02), but within 0.5 points.

**10-class read (all-class set, U donors).**

| circuit | 10-class acc | W/Z binary |
|---|---|---|
| 89-node ACDC literal | 13.1 % [12.6, 13.6] | 62.4 % |
| top 1,075 edges | 18.4 % | 61.0 % |
| top 128 nodes | 16.2 % | 64.6 % |
| everything resampled | 9.7 % | 49.9 % |

- With the 89-node circuit, calls follow the donor's class on 75.4 % of jets. The per-class diagonal is 9–20 %, and true Wqq is
  predicted only 986 times out of 20,000. The confusion matrices are in the json.

**Corruption A (eval[::5], same class, mass-matched donors).**
- W/Z accuracy: unedited model 83.2 %, fully resampled model 84.3 %.
- Jet-level faithfulness \(1 - \langle|d^{circ} - d|\rangle / \langle|d^{corr} - d|\rangle\): top 1,075 edges 0.29, top 2,000
  0.35, 89-node literal 0.30, 10-node circuit 0.05, P3 references ≤ 0.29.
- A same-class, mass-matched donor already carries the W/Z information. Corruption A therefore cannot test W/Z faithfulness
  through accuracy; it measures only how much jet-specific detail a circuit carries.

**Attribution fidelity (eval[::10], single-edge resample).**
- Spearman of predicted against measured change: top 50 edges 0.87 (EAP-IG) and 0.75 (EAP). All 100 edges: 0.94 (EAP-IG) and
  0.85 (EAP).
- The sign agrees for 50 of 50 top edges.
- Measured / predicted, median 3.6. Example: z97 → logits, predicted −0.89 and measured −3.34 per jet.
- Holm-significant (normal p): 49 of 50 top edges, 9 of 50 random edges.

## Reading
- Under the resample corruption that swaps W for Z information (B), the share of the W-vs-Z logit difference that a kept subset
  recovers grows smoothly with its size. The growth is about 0.12 in F per doubling, from 10 edges to 8,000 edges.
- 80 % recovery needs 37 % of the edges or 59 % of the nodes. Neither the localized prediction (> 0.8 at < 5 %) nor a sharp knee
  is seen. This matches the stated alternative ("faithfulness grows with size; the decision is distributed"). The growth is
  close to linear in log-size, not linear in size.
- Attribution concentrates at the read-out end. The largest single contributions are n98's direct path (18 %) and the block-1
  class heads with the class MLP group 0 of each block. The literal {b2h8, b1h3, n98} keeps 0.6 % of the logit difference. b2h8
  enters the ACDC circuits only at the 27-node size, and drops out at 10 nodes.
- The ACDC circuits recover far more than random circuits of the same size and layout (p at the floor 0.0099). As literal
  circuits they keep 17–56 % of the score difference and 29–59 % W/Z accuracy, against the model's 83 %. In the 10-class read
  they stay at 10–13 %, because 120 of 128 class-token coordinates come from the donor.
- EAP-IG ranks single-edge effects well (Spearman 0.87 on the top 50) but underestimates their size by 3.6×. The sum over edges
  is complete (1.02), so edges interact and do not add.

## Issues and caveats
- **Corruption choice.** The main metric uses opposite-class donors, which is the strictest corruption. Under P3's uniform donors
  the same circuits score higher in W/Z accuracy (62 % vs 59 % for the 89-node circuit), because "chance" is 50 %, not 17 %.
- **Edge granularity.** MLP units are grouped in 32s by attribution rank. Finer units could change the curve.
  - Heads are separated before post_attn_norm. Ablating one head of a block shifts the LayerNorm statistics seen by its other
    heads.
  - Particle-slot resampling pairs particles by pT rank between two jets of different multiplicity. For B donors, multiplicity
    is not matched.
- **ACDC subset size.** ACDC ran on 300 discovery jets. Its circuits' F on the evaluation set (0.385 / 0.241 / 0.151) is close
  to the subset values (0.43 / 0.28 / 0.18). ACDC started from a top-2,000 set with F = 0.60, which caps what it can find.
- **Fidelity p-values.** The in-run fidelity p-values use the bootstrap floor 1/1000. Holm over 100 edges can then never fall
  below 0.05, so the logged "0 Holm-significant" is an artefact. `--post` recomputes normal-approximation p-values from the
  bootstrap s.e. (`p_normal_holm` in the json). Random-circuit p-values are at their floor 1/101.
- **Logit agreement with weaver.** On the Ada GPU the re-implementation differs from weaver's logits by up to 4.6e-3, and by
  4.8e-6 on the 2080 Ti. Call changes against the model were not counted in this run. All circuit numbers are relative to the
  re-implementation's own clean logits, which have mask = 1 exactness of 3.8e-6.
- **Time budget.** Random controls and fidelity used eval[::10] (1,000 jets) to fit the GPU budget.

## Explainer entry (for site/explain.py)
- **Question.** Is there a small circuit inside ParT_full that reproduces its W-vs-Z decision when everything else is replaced?
- **Background.** P3 found that the circuit {b2h8, b1h3, n98} kept alone separates W from Z at 53 %, against 83 % for the model.
  All 16 class heads plus n98 give 63 %. P3 searched only the class-attention stage. Here the search covers the whole network.
- **Method.**
  - Write the network as a graph of 433 nodes and 21,504 edges with masks \(m_e\). Ablated edges carry a donor jet's value: a
    jet of the other class with matched \(p_T\).
  - Faithfulness \(F = \frac{\sum_j s_j (d^{circ}_j - d^{corr}_j)}{\sum_j s_j (d_j - d^{corr}_j)}\), with \(d = \ell_W - \ell_Z\).
  - Rank edges by integrated gradients over the masks, \(\mathrm{score}_e = \int_0^1 \partial D/\partial m_e\,d\alpha\) (5
    steps). Then prune greedily (ACDC).
- **Result (figure p9_eap_wz.png).** F rises steadily with size: 0.51 at 5 % of edges, 0.86 at 37 %. Literal circuits of 10, 27
  and 89 nodes give 29 %, 36 % and 59 % W/Z accuracy. Random circuits of the same layout give 17 %.
- **Checks.**
  - All masks on reproduces the model; all off reproduces the donor.
  - Integrated-gradient completeness 1.02.
  - Single-edge resampling ranks match the attributions (Spearman 0.87).
  - The P3 circuits reproduce (52 %, 64 %).
  - 100 random circuits per size; McNemar against the full model.
- **Why it matters.** Under the user's literal circuit definition, ParT_full's W/Z decision has no small circuit at the
  head/MLP-group level. Paper C can report circuit faithfulness against size, with this ceiling, rather than a single named
  circuit.
