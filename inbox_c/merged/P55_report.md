# P55: L-GATr literal attention-head circuits for W vs Z and t vs QCD; claim C26 (agent report, round C12, 2026-10-21)

## Conclusion
No small set of L-GATr's attention heads works on its own. When only the 16 top-ranked heads (of 96) are kept and the others are replaced by their output on a different jet, the network recovers none of its W-vs-Z ability and about a fifth of its top-vs-QCD ability. Even with 32 heads kept, the share recovered stays below 0.6 for both tasks. The heads that matter are all in the first three or four blocks. For top vs QCD, random heads taken from those same blocks do about as well as the ranked heads, and random heads taken from any block do much worse. So what matters is that the early blocks keep most of their heads, not which particular heads are kept. Still uncertain: replacing the removed heads by their average output instead of another jet's output gives clearly different numbers (up to 0.65 at k = 16 for top vs QCD), so the exact recovered fractions depend on how heads are removed.

## Terms
- **Head output O_bh**: what attention head h of block b passes on for each token, softmax(q·k/√72)·V. It has 72 numbers per token: 4 multivector channels × 16 components, plus 8 auxiliary scalars. It is taken before the per-head scale and the output projection.
- **Multivector grades**: grade 0 = scalar (component 0); grade 1 = four-vector (1–4); grades 2–4 = bivector, axial vector, pseudoscalar (5–15).
- **Resample ablation (main)**: O_bh of a jet is replaced by O_bh of a donor jet, computed with the unedited network on the unedited donor.
  - Donor: chosen at random among the 5 jets nearest in jet pT, any class. The median |Δ ln pT| is 6·10⁻⁵, and the donor has the same class in 9.9 % of cases.
  - Token mapping: the global token and the two reference ("spurion") tokens map to themselves; particle k maps to donor particle k mod n_donor.
- **Mean ablation (secondary)**: O_bh is replaced by its mean over a 10-class reference set (200 jets per class, file _119). Separate means are used for the global token, the beam spurion, the time spurion and the particles.
- **Circuit of size k**: the k top-ranked heads are computed normally; the other 96 − k are ablated together. Every MLP is kept.
- **Read-out d**: the logit difference, l_W − l_Z (task WZ) or l_t − l_QCD (task tQCD).
- **Binary accuracy**: the fraction of the task's true jets for which the sign of d is correct.
- **Faithfulness F** = (acc_circuit − acc_floor) / (acc_unedited − acc_floor), where floor = all 96 heads ablated. F = 1 means the circuit gives the full binary accuracy; F = 0 means it gives no more than the floor. F_d is the same ratio computed on the mean signed d.
- **Greedy ranking**: start from the full model. Remove each live head in turn and measure M = mean of s_y·d (s_y = ±1 by true class). Remove the heads whose removal keeps M highest: 16 at a time down to 32 live heads, then 4 at a time. The ranking is the reverse removal order.
- **EAP-IG**: integrated gradients of M with respect to a per-head mixing weight w, where O = w·O_live + (1 − w)·O_donor. The path runs from all w = 0 to all w = 1, with 16 midpoint steps.
- **Block-matched random set**: random heads with the same number of heads per block as the ranked circuit.
- **Rank p** = (1 + #random sets with accuracy ≥ circuit) / (1 + n_random).
- **Holm**: step-down correction over the 10 (task, k) tests of each family.
- **E4 dilation**: every particle's (Δη, Δφ) about the stored jet axis is multiplied by 1.1. This is a real change of the jet mass and of the pair masses (mean Δ ln m_jet = +0.094).
  - **R_c** = Σ_jets Δd_circuit / Σ_jets Δd_full, the fraction of the full network's response to this real mass change that the circuit reproduces. Here Δd = d(dilated) − d(unedited).
  - The ablated heads keep the donor outputs from the unedited donors, so they do not see the edit.

## Procedure
**Model.** L-GATr final checkpoint `models/lgatr/runs/jc2m/jc2m_s0a/models/model_run0.pt` (authors' jctagging config; 12 blocks × 8 heads). Control model: the same architecture with random weights (torch seed 0).

**Samples.** All from JetClass test_20M. Neither file is used by L-GATr's training (train_100M) or validation (val_5M _120–_124) data, nor anywhere earlier in loop C. val_5M has no _125/_126.
- **Ranking**: `<class>_119.root`, `default_rng(5555)`, 2,000 random rows per class for W, Z, t→bqq′ and QCD.
  - Reference/donor pool: 200 other random rows per class, all 10 classes.
  - Greedy subset: 400 of the 2,000 per class.
- **Evaluation**: `<class>_113.root`, `default_rng(5556)`, 1,000 random rows per class, N = 10,000, shuffled.
  - Task jets: 2,000 for W/Z and 2,000 for t/QCD.
  - Donors come from this sample, never the jet itself.
- **Freezing**: rankings were written to `results/lgatr/p55_ranking.json` before any evaluation-sample run.

**Edits and controls.**
- Circuits at k = 4, 8, 16, 24, 32 for both rankings.
- Null sets per (task, k), evaluated on the task's 2,000 jets:
  - 100 random sets block-matched to the greedy circuit;
  - 30 random sets block-matched to the EAP-IG circuit;
  - 50 unrestricted random sets.
- Per-head roles (task jets):
  - each of the 96 heads removed alone from the full model, and kept alone;
  - for each head of the greedy k = 16 circuit: the circuit minus that head, and the circuit with only that head's grade-0, grade-1, grade-2–4 or scalar part ablated;
  - McNemar tests with Holm over the heads.
- E4 dilation on the W/Z jets: greedy circuits, the floor, and 20 block-matched random sets per k.
- Untrained L-GATr: EAP-IG ranking and circuits.
- Bootstrap: 1,000 resamples over jets.

**Jobs.**
- `anrunw-ai-c-p55-smoke` (`--smoke`, GTX 1080 Ti). Found NaN greedy scores: jets outside the greedy subset entered the mean. Fixed.
- `anrunw-ai-c-p55-run` (GTX 1080 Ti). Ran out of GPU memory in the greedy step: 94k tokens per forward pass. Fixed by capping tokens per pass at 24k.
- `anrunw-ai-c-p55-run2` (TITAN Xp, 10,601 s, exit 0). Run with `/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p55_lgatr_head_circuit`. All sections completed.

**ParT reference.** ParT_full was not rerun under this protocol: there was no GPU time left after the 2.9 h job. P37 is quoted as-is below.

## Checks
- **Unedited accuracy**:
  - P1 sample (N 3,000): 83.80 % (expected 83.73 %).
  - Evaluation sample, 10-class: 84.62 % [83.87, 85.40].
  - W/Z binary 82.35 % [80.80, 84.05]; t/QCD binary 97.85 % [97.25, 98.50].
- **Re-implemented attention vs the model's own (xformers) forward**: same call on 100 % of jets, max |Δlogit| 0.15 (as in P49). With w = 1 for every head the logits are identical to the unedited ones (max |Δ| = 0).
- **Floors (all heads ablated)**:
  - resample: 10-class 10.1 %; binary W/Z 50.40 %, t/QCD 49.25 %;
  - mean: 10.0 %; binary 50.0 % for both.
- **Untrained L-GATr**: 10-class 9.8 %; W/Z binary 50.0 %; t/QCD binary 61.75 %.
- **EAP-IG completeness is poor.**
  - Sum of scores: 0.25 vs M(1) − M(0) = 2.22 (W/Z); 3.32 vs 9.31 (t/QCD).
  - M along the path stays small until w ≈ 1. For t/QCD, M = 2.25 at w = 0.97 and 9.38 at w = 1, so the 16 midpoints miss most of the change.
  - The EAP-IG ranking is therefore a weak secondary ranking; the greedy ranking is primary.

## Numbers (evaluation sample; binary accuracy over each task's 2,000 jets; [95 % bootstrap])
| task | k | greedy F | greedy binary acc | 10-class acc (circuit kept) | block-matched random F: mean [2.5, 97.5 %], max | rank p (Holm) | unrestricted random F mean (max) | EAP-IG F | mean-ablation F (greedy) |
|---|---|---|---|---|---|---|---|---|---|
| W/Z | 4 | 0.063 [−0.007, 0.121] | 52.4 % | 10.4 % | −0.013 [−0.061, 0.044], 0.052 | 0.010 (0.099) | −0.015 (0.022) | −0.025 | 0.000 |
| W/Z | 8 | 0.034 [−0.034, 0.093] | 51.5 % | 10.4 % | −0.012 [−0.056, 0.050], 0.078 | 0.079 (0.48) | −0.019 (0.031) | −0.019 | 0.000 |
| W/Z | 16 | −0.016 [−0.091, 0.054] | 50.0 % | 13.7 % | −0.012 [−0.067, 0.045], 0.067 | 0.55 (1.0) | −0.006 (0.039) | −0.002 | 0.005 |
| W/Z | 24 | 0.146 [0.081, 0.205] | 55.1 % | 18.7 % | 0.044 [−0.016, 0.114], 0.136 | 0.010 (0.099) | 0.003 (0.083) | 0.049 | 0.516 |
| W/Z | 32 | 0.186 [0.122, 0.248] | 56.4 % | 23.9 % | 0.050 [−0.020, 0.151], 0.175 | 0.010 (0.099) | 0.004 (0.089) | 0.028 | 0.334 |
| t/QCD | 4 | 0.071 [0.041, 0.100] | 52.8 % | 11.6 % | 0.040 [0.011, 0.098], 0.101 | 0.14 (0.69) | 0.018 (0.146) | 0.010 | 0.000 |
| t/QCD | 8 | 0.133 [0.101, 0.164] | 55.8 % | 11.9 % | 0.120 [0.028, 0.255], 0.296 | 0.43 (1.0) | 0.024 (0.113) | 0.248 | 0.000 |
| t/QCD | 16 | 0.193 [0.159, 0.227] | 58.7 % | 13.2 % | 0.230 [0.042, 0.452], 0.483 | 0.60 (1.0) | 0.042 (0.131) | 0.203 | 0.650 |
| t/QCD | 24 | 0.568 [0.527, 0.604] | 76.9 % | 15.2 % | 0.260 [0.052, 0.539], 0.599 | 0.020 (0.14) | 0.050 (0.210) | 0.123 | 0.684 |
| t/QCD | 32 | 0.418 [0.380, 0.454] | 69.6 % | 17.7 % | 0.273 [0.023, 0.580], 0.620 | 0.22 (0.87) | 0.051 (0.282) | 0.175 | 0.334 |

**Logit-difference faithfulness F_d (greedy).**
- W/Z: 0.035, 0.024, −0.041, 0.185, 0.272 (k = 4 … 32).
- t/QCD: 0.249, 0.325, 0.358, 0.702, 0.865.

**Placement of the greedy circuits (heads per block, blocks 0–11).**
- W/Z k = 16: 7, 7, 2, 0, … These are b0h0, b0h5, b2h4, b2h3, b0h1, b1h6, b1h5, b0h4, b0h7, b1h7, b1h3, b1h0, b0h6, b0h2, b1h2, b1h1.
- t/QCD k = 16: 6, 7, 2, 1, 0, … (15 of 16 heads shared with the W/Z set; b3h4 in place of b0h1).
- At k = 24 and 32, 91–100 % of the heads are in blocks 0–4.
- The EAP-IG sets are less concentrated: 53–62 % in blocks 0–4 at k = 32.

**10-class behaviour with the circuit kept.** Every circuit gives 10–24 % 10-class accuracy (unedited 84.6 %). Calls pile up on QCD and H→4q. For example, with the t/QCD k = 32 circuit, QCD recall is 0.56 and H→4q recall 0.47, while t→bqq′ recall is 0.13.

**Per-head roles (task jets, binary accuracy).**
- **Removed alone from the full model.** Most heads in blocks 0–5 are Holm-significant for both tasks. Examples:
  - W/Z: b0h0 −32.9 points, b0h5 −31.5, b1h3 −30.1;
  - t/QCD: b0h5 −47.9, b0h2 −47.8, b1h2 −47.9, b2h3 −47.9.
- **Kept alone (all other 95 ablated).** Every head gives 48.9–51.8 %, which is the floor. No head is Holm-significant against the floor.
- **Circuit minus one head** (t/QCD k = 16 circuit, baseline 58.7 %):
  - b0h0 −7.6, b0h2 −6.1, b0h6 −6.1, b1h6 −4.8;
  - removing b2h4 raises accuracy by +11.0 and removing b2h3 by +8.3.
  - In the W/Z k = 16 circuit (baseline at the floor), every change is within ±2 points.
- **Grade split** (t/QCD circuit; ablating only one part of the head's output):
  - Ablating the grade-1 part or the grade-2–4 part costs similar amounts, e.g. b0h2 −7.3 (grade 1) and −7.2 (grades 2–4); b0h5 −6.7 and −6.5; b0h0 −3.5 and −6.8.
  - Grade 0 and the auxiliary scalars cost ≤ 1.5.
  - This matches where the output sits: these heads put ≤ 5 % of their output sum of squares into grade 0 and scalars, and the rest into grade 1 and grades 2–4.

**E4 dilation (W/Z jets, angles × 1.1, Δ ln m = +0.094).**
- Full network: mean Δd = −0.91 [−0.97, −0.85] (W −1.35, Z −0.48); true-W → Z calls 11.6 % → 37.6 %.
- All heads ablated: R = 0.000.
- Greedy circuits: R_c = 0.038 [0.032, 0.045] (k = 4), 0.044 (k = 8), −0.146 [−0.175, −0.121] (k = 16), 0.072 [0.043, 0.100] (k = 24), 0.114 [0.084, 0.141] (k = 32).
- Block-matched random sets: R between −0.025 and 0.007 on average; 95 % ranges reach ±0.29 at k = 32.
- Rank p of the circuit ≥ 0.095 at every k.

**Untrained L-GATr (EAP-IG circuits).**
- W/Z binary: 50.0 % at every k (unedited also 50.0 %).
- t/QCD binary: 53.5–64.9 % (unedited 61.75 %, floor 49.15 %).

## Hypotheses
- **H1 (sparse early circuit)** predicted F ≥ 0.8 with k ≤ 16 for both tasks, heads mostly in blocks 0–4, and the chosen set above every random set.
  - Excluded: at k ≤ 16, F ≤ 0.06 (W/Z) and ≤ 0.19 (t/QCD), with upper CI bounds 0.12 and 0.23.
  - The placement part holds: 100 % of the k = 16 heads are in blocks 0–3.
  - The "beats every random set" part fails after Holm in every (task, k). The smallest Holm p is 0.099 (W/Z k = 4, 24, 32).
- **H2 (distributed)** predicted F rising roughly with k, block-matched random sets reaching a similar F, and no small set exceeding 0.5.
  - Consistent for t/QCD: block-matched random sets reach the circuit's F (97.5 % points 0.45–0.58 for k = 16–32).
  - For W/Z, F stays ≤ 0.19 for every set.
  - One exception: the t/QCD greedy k = 24 set reaches 0.57, above 0.5, though not at k ≤ 16.
- **Unrestricted vs block-matched random sets** (not pre-registered as a hypothesis): at k ≥ 8, unrestricted random sets give F ≤ 0.05 on average for t/QCD, against 0.12–0.27 for block-matched ones. Which blocks the kept heads come from changes F more than which heads within those blocks are chosen.
- **Mass read-out through the W/Z circuit**: at most 11 % of the response to a real mass change (k = 32) passes through the kept heads.

## Reference: ParT_full (P37, quoted as-is; mean ablation, file _111, different protocol)
- Five heads (b1h8, b1h4, b1h3, b1h1 in the first class-attention block, plus the lepton head b2h1) carry 58 % of the drop in the top-vs-QCD score that comes from ablating all 80 heads. Random 5-head sets carry about 5 %.
- Literal circuits of 3–10 heads, with the two class-block MLPs kept, reach balanced accuracy 0.28–0.39 (full 0.87) and top recall ≤ 0.14.
- Difference in placement: ParT's top-relevant heads sit in its class-attention blocks, i.e. the read-out stage, after all particle blocks. L-GATr's are in its first three or four of 12 blocks.
- Similarity: in neither model does a small literal head set reproduce the binary or 10-class call.
- The numbers are not computed on the same footing (ablation type, metric, sample), so they are not compared directly.

## Issues
- F depends on the ablation type. For example, W/Z k = 24 gives 0.15 with resample ablation and 0.52 with mean ablation; t/QCD k = 16 gives 0.19 and 0.65. Mean ablation gives every ablated head a class-neutral constant, while resample ablation injects a different jet's information.
- EAP-IG with 16 steps misses most of M(1) − M(0) (completeness 11 % and 36 %), so its ranking is weak.
- The greedy ranking is non-monotone in k: t/QCD F is 0.57 at k = 24 and 0.42 at k = 32. In the k = 16 circuit, b2h4 and b2h3 lower accuracy when present. Greedy removal in chunks of 16/4 is an approximation.
- R_c at k = 16 is negative (−0.15); the circuit's response to the dilation has the opposite sign to the full network's. Its binary accuracy at k = 16 is at the floor.
- ParT_full was not rerun under this protocol.

## Files
- Script: `interp/p55_lgatr_head_circuit.py`.
- Results: `results/lgatr/p55_lgatr_head_circuit.json`, `p55_ranking.json` (frozen rankings), `p55_arrays.npz`, `p55_lgatr_head_circuit.png`.
- Logs: `logs/c-p55-run2.log` (also `c-p55-smoke.log`, `c-p55-run.log`).

## Figure
`results/lgatr/p55_lgatr_head_circuit.png`:
- (a), (b): F against k for the greedy and EAP-IG circuits, with the block-matched random band and the unrestricted random mean.
- (c): blocks of the k = 16 greedy heads.

## Explainer entry (for site/explain.py; glossary additions: resample ablation, faithfulness F, EAP-IG, block-matched random set)
- **Question**: does a small set of L-GATr attention heads, kept alone, reproduce its W-vs-Z and top-vs-QCD decisions?
- **Background**: P49 found that removing single heads hurts only in blocks 0–4. A literal circuit, in which only the circuit's heads compute, had not been tested.
- **Method**:
  - Rank the 96 heads on one sample (greedy removal; integrated gradients).
  - On a fresh sample, keep the top k heads and replace every other head's output $O_{bh}$ by its output on a pT-matched donor jet.
  - Measure $F=\frac{\mathrm{acc}_{\rm circ}-\mathrm{acc}_{\rm floor}}{\mathrm{acc}_{\rm full}-\mathrm{acc}_{\rm floor}}$.
  - Compare with random head sets drawn from the same blocks and from any block.
- **Result** (figure p55_lgatr_head_circuit.png):
  - $F\le0.19$ for W/Z up to $k=32$; for top/QCD, $F=0.19$ at $k=16$ and at most $0.57$ at $k=24$.
  - The ranked heads all sit in blocks 0–3.
  - Random heads from the same blocks reach a similar F; random heads from any block do not.
- **Checks**: unedited 84.6 % (P1 sample 83.8 %); $w=1$ reproduces the logits exactly; the floor is at chance; untrained model; mean-ablation variant; real-mass-dilation response $R_c\le0.11$.
- **Why it matters**: in L-GATr the early blocks act as a group. Most early heads are needed, but no small subset of them is enough. Unlike ParT, the decision is not concentrated in a few read-out heads.
