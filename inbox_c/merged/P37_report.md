# P37: top-quark head circuit and an 80-head × 10-class knock-out map of ParT; claim C17 (agent report, round C9, 2026-10-20)

## Conclusion
In ParT_full, a few attention heads carry most of the top-quark decision. Five heads together carry 58 % of the drop in the top-vs-QCD score that comes from averaging out all 80 heads; five random heads carry about 5 %. Four of the five are class-token heads in the first class-attention block, and the fifth is the head P32/P34 found reading the lepton flag. Those five heads also carry two thirds of the network's response when the b-quark prong is merged into one particle. However, they carry a similar share (53–57 %) when a W prong is merged instead, so they respond to prong structure in general and not to the b quark specifically.

Two parts of C17 do not hold:
- **Circuit kept alone.** The few-head circuit does not reproduce the top call: top recall is at most 14 %, against 96 % for the full network.
- **Class-specific head sets.** Each class's call depends mostly on its own weighting of one shared pool of eight class-block-1 heads. The sets are not disjoint. The top and lepton head sets overlap by half; top vs W/Z overlap 0.2–0.3.

Still uncertain:
- single-head knock-outs change recall by at most 11 %, so the per-class head sets measured by recall rest on small effects;
- the weight-permuted null gives more significant recall cells than the trained model, because its output scores are nearly tied.

## Runs
| job | command | node / GPU | wall | output |
|---|---|---|---|---|
| `anrunw-ai-c-p37-smoke4` | `--smoke` | A10 | 66 s | failed in the head-description step (array shape); fixed |
| `anrunw-ai-c-p37-smoke5` | `python -m interp.p37_top_circuit --smoke && … --smoke --plot` | V100 | 170 s | smoke JSONs (all four models) |
| `anrunw-ai-c-p37-run` | `python -m interp.p37_top_circuit --models full,untrained,perm` | A10 (hcc-nrp-shor-c6009) | 1,447 s | `results/part/p37_top_circuit_{full,untrained,perm}.json`, `logs/c-p37-run.log` |
| `anrunw-ai-c-p37-runk` | `python -m interp.p37_top_circuit --models kin` | RTX 2080 Ti | 1,006 s | `results/part/p37_top_circuit_kin.json`, `logs/c-p37-runk.log` |
| `anrunw-ai-c-p37-plot` | `python -m interp.p37_top_circuit --plot` (CPU, job.sh) | – | 110 s | `results/part/p37_top_circuit.png` |

Script: `interp/p37_top_circuit.py`, rewritten from the round-C8 version, which had never run. Changes from the C8 version:
- fresh file `_111`, seed 3725;
- a second reference mean (the noise floor);
- the 80 × 10 recall matrix with McNemar and BH;
- k = 2 and 40 added to the circuit sizes;
- a 'class-block MLPs kept' variant;
- the t-vs-QCD AUC;
- joint knock-outs;
- a whole-W-subjet merge control;
- a displaced-track check of the b-prong;
- the weight-permuted null;
- head descriptions from attention weights.

**File choice.** The plan asked for `_125`, which does not exist: test_20M holds `_100–_119` and val_5M holds `_120–_124`. `_111` was used instead. TTBar and seven other classes of `_111` were never used before. P13v2 used 3,000 of the 100,000 W and Z rows.

## Unedited accuracy (reference check)
- P1 sample (rows 3,000–4,000 of each class's `_100` file), 10-class accuracy:
  - ParT_full **0.8593**, ParT_kin **0.7447**, both as in P1;
  - untrained null 0.102, permuted null 0.088.
- P37 evaluation sample (10,000 jets, 1,000 per class, `_111`), 10-class accuracy with 95 % bootstrap interval:
  - ParT_full 0.871 [0.864, 0.877]: t→bqq′ recall 0.958, t-vs-QCD accuracy 0.987, t-vs-QCD AUC 0.9992;
  - ParT_kin 0.754 [0.745, 0.762]: t→bqq′ recall 0.905, AUC 0.992.
- **Head-output check.** The sum of the per-head outputs plus the output bias reproduces the attention-module output within 6·10⁻⁵ in every block.
- **Mean-ablation check.** With all 80 heads and all 10 MLPs at their mean, every jet gets the same call (balanced accuracy 0.100).

## Procedure
**Models.**
- ParT_full and ParT_kin: released weights.
- Nulls, both built from ParT_full:
  - **untrained**: every weight tensor redrawn from a Gaussian with the trained tensor's mean and spread (seed 37);
  - **perm**: the output rows of every weight matrix permuted, with the matching biases permuted alike (P33 recipe, seed 3725). This keeps every weight value but destroys the trained wiring.

**Sample.**
- JetClass test_20M `<class>_111.root`, numpy `default_rng(3725)`, random rows (never the first k).
- Split per class into four disjoint sets:
  - **EVAL**: 1,000 per class, N 10,000;
  - **REF**: 200 per class, used for the mean outputs;
  - **REF2**: 200 per class, a second mean used as the noise floor;
  - **SCAN**: a further 1,000 t→bqq′ jets used to rank the heads, so that ranking and circuit evaluation use different jets.
- **SUB**: 100 per class of EVAL, used for the random-set comparisons.
- Every jet passes once through `part_core.build`.

**Heads.**
- ParT has 8 particle blocks (p1–p8, attention among particles) and 2 class-attention blocks (b1, b2), in which one extra vector, the class token, attends to the particles. Its final state gives the 10 class scores.
- Each block has 8 heads, 80 in total. Naming: `b1h8` = head 8 of class block 1. P34 calls the class blocks c1, c2, so its `c2h1` is `b2h1` here.
- A head's output is its term $A_hV_hW_O^{(h)}$ in the attention output.
- **Mean ablation** of a head: its output at every token is replaced by its average over the REF jets' real tokens (class blocks: the class token). The head then carries no information about the jet.
- **Mean ablation of an MLP** (the two-layer feed-forward network applied to every token after attention): its output is replaced by its REF average in the same way.

**Read-outs.**
- $\ell_t=\log\frac{p_t}{1-p_t}$: the top log-odds.
- $d_{QCD}=\Delta(\text{logit}_t-\text{logit}_{QCD})=\Delta\log(p_t/p_{QCD})$, and $d_W$ likewise against W→qq′.
- **Recall** of class k: the share of true-k jets called k.
- **Balanced accuracy**: the mean of the 10 recalls. 0.1 means one call for every jet. It equals the 10-class accuracy here, because the classes are balanced.
- **AUC**: the probability that a random top jet has a higher logit_t − logit_QCD than a random QCD jet.

**Step 1a (top scan, SCAN jets).**
- Each head is mean-ablated alone, and $d_{QCD}$ and $d_W$ are read per jet.
- Statistics:
  - the mean with a 95 % bootstrap interval (1,000 resamples of jets);
  - a sign-flip permutation p (20,000 flips; H0: the mean change is 0);
  - Holm correction over 160 tests (80 heads × 2 contrasts). The smallest attainable Holm p is 0.008.
- **Noise floor**: the same ablation with the REF2 mean.
- **Joint knock-out**: the top-k ranked heads ablated together, divided by the change when all 80 are ablated ("share of the drop"), against 20 random k-head sets.
- **Ranking**: by the mean of $d_{QCD}$ and $d_W$, Holm-significant lowering heads first.

**Step 1b (80 × 10 map, EVAL jets).**
- Each head is ablated alone. Read-outs per class:
  - the change in recall, with a bootstrap interval over that class's 1,000 jets;
  - McNemar on the paired correct/incorrect calls;
  - Benjamini–Hochberg (BH) correction over 800 tests;
  - the change of the class's own log-odds on its own jets.
- **Class head set (BH)**: heads with a BH-significant recall drop (q < 0.05).
- **Class head set (top-5)**: the 5 heads with the most negative own-class log-odds change. This fixed size does not depend on statistical power.
- Groups: top = {t→bqq′}, W/Z = {W→qq′, Z→qq}, lepton = {t→bℓν, H→ℓνqq′}.
- **Jaccard** overlap = |A∩B|/|A∪B|. Null: random sets of the same sizes, mean 0.04, 95th percentile 0.20.
- ParT_full only: the whole map repeated with the REF2 mean.

**Step 2 (literal circuits, EVAL; user circuit definition).**
- Circuit k = the first k ranked heads, k = 0, 1, 2, 3, 5, 10, 20, 40, 80. Three variants:
  - **literal**: only those heads compute; every other head and all 10 MLPs are mean-ablated;
  - **cls_mlp**: as literal, but the two class-block MLPs are kept;
  - **heads_only**: all MLPs are kept.
- Null: 100 random k-head sets, the same sets for every variant, evaluated on SUB.
- Percentile p = (1 + number of random sets ≥ circuit) / 101.

**Step 3 (input edit, the 1,000 EVAL true t→bqq′ jets).**
- **Prong finding:**
  - exclusive kT clustering to 3 subjets (distance min(p_T,i², p_T,j²)·ΔR², E-scheme recombination, rapidity–φ);
  - the W pair = the subjet pair with invariant mass nearest 80.4 GeV;
  - the **b-prong** = the third subjet.
- **Edits.** Each one replaces a set of particles by one particle at their summed four-momentum, so the jet four-momentum is unchanged; the merged particle carries the prong's mass.
  - **a**: merge the b-prong;
  - **b (control)**: merge a W-prong cluster with the same summed p_T;
  - **c (control)**: merge a whole W subjet (the one with p_T closer to the b-prong's);
  - **b2 (control)**: merge a W-prong cluster with the b-prong's particle count.
- 705 jets admit a, b and c; 836 admit b2.
- **Displacement check**: is the b-prong the subjet with the most charged tracks with |d0|/σ_d0 > 3 and |d0| < 2?
- **Patching**: the unedited jet is run with the top-k heads' outputs replaced by their outputs in the edited run (tokens aligned by particle).
  - share = mean patched Δℓ_t / mean full Δℓ_t;
  - null: 100 random k-head sets.

**Step 4 (head description).** Attention weights of every head on the 1,000 SCAN top jets:
- the share of attention weight on b-prong / W-prong particles, charged hadrons, neutral hadrons, photons, leptons and the leading particle, against the share of particles in each category;
- the mean ΔR of the attended particle to the jet axis (class heads) or to the querying particle (particle heads);
- the share of a particle's attention that stays within its own prong ("same-prong");
- the effective number of keys exp(entropy).

## Numbers

### Heads the top call depends on (ParT_full, SCAN, N 1,000 true t→bqq′; unedited top recall 0.946, mean ℓ_t 5.83)
| head | what it attends to (top jets) | d_QCD [95 % CI] | d_W [95 % CI] | Holm p |
|---|---|---|---|---|
| b1h8 | class token → b-prong 54 % of weight (38 % of particles), leptons 8 % (1.3 %), charged 59 %; 22 effective keys | −1.51 [−1.56, −1.46] | −2.33 [−2.40, −2.26] | 0.008 |
| b1h4 | class token, broad (43.5 effective keys), no category above its particle share | −1.58 [−1.62, −1.54] | −1.60 [−1.64, −1.55] | 0.008 |
| b1h3 | class token, broad (39 keys) | −0.62 [−0.65, −0.59] | −1.08 [−1.10, −1.04] | 0.008 |
| b1h1 | class token, broad (45 keys) | −0.42 [−0.44, −0.40] | −0.48 [−0.50, −0.46] | 0.008 |
| b2h1 | class token → leptons 15.5 % (1.3 %), b-prong 58 % (38 %); 11 keys (the P32/P34 lepton head) | −0.37 [−0.38, −0.35] | −0.51 [−0.52, −0.49] | 0.008 |
| p7h8 | particle block 7, cross-prong: 14 % same-prong, query–key ΔR 0.50 | −0.39 [−0.41, −0.37] | −0.40 [−0.42, −0.38] | 0.008 |
| p8h6 | particle block 8, 61 % same-prong, ΔR 0.28 | −0.31 | −0.32 | 0.008 |
| p1h5, p3h3 | particle blocks 1/3, local: 89 %/82 % same-prong, ΔR 0.13/0.18 | −0.22 / −0.22 | −0.19 / −0.18 | 0.008 |

- **Significant heads.** 57 of 80 heads lower the top contrast significantly after Holm, and 22 raise it (ParT_kin: 67 and 18). Most heads have a small but measurable effect.
- **Concentration of single-head effects.** The largest single head holds 15 % of the summed negative single-head effects (ParT_kin 9 %); 7 heads hold 50 % (ParT_kin 10). The alternative's "no head above 5 %" is excluded.
- **Joint knock-out**, share of the all-80 drop (all-80: d_QCD −9.77 full, −7.71 kin):

| k heads ablated together | ParT_full top-ranked | ParT_full random (mean of 20) | ParT_kin top-ranked | ParT_kin random |
|---|---|---|---|---|
| 1 | 0.15 [0.150, 0.158] | 0.01 | 0.10 | 0.01 |
| 3 | 0.47 [0.462, 0.470] | 0.04 | 0.27 | 0.04 |
| 5 | **0.58 [0.577, 0.587]** (top recall 0.946 → 0.525) | 0.05 | 0.38 | 0.06 |
| 10 | 0.64 [0.634, 0.646] (top recall 0.395) | 0.16 | **0.81 [0.801, 0.812]** (top recall 0) | 0.11 |
| 20 | 0.85 | 0.28 | 0.91 | 0.32 |

- **Noise floor.** With the mean taken from the other 2,000 reference jets, the 80 single-head d_QCD values correlate at 0.9998 with the main ones. The median |difference| is 0.001, against effects up to 1.6. All Holm-significant lowering heads keep their sign (ParT_kin: 91 %).
- **Nulls.** Untrained: the largest single-head d_QCD is −0.28, and all-80 ablation changes d_QCD by +0.01, so it has no top signal to remove. Permuted: largest single-head d_QCD −0.21, all-80 +0.68.

### 80 × 10 knock-out map (EVAL, 1,000 jets per class)
- **ParT_full.**
  - 39 of 800 cells are BH-significant (24 of them drops); the largest |Δrecall| is 0.113 (b1h3 on W→qq′).
  - The largest single-head recall drop per class is 0.005–0.113. Own-class log-odds drops are −0.3 to −1.8, and −3.8/−4.7 for b2h1 on H→ℓνqq′ and t→bℓν.
  - The REF2 repeat: median |Δrecall difference| 0, 95th percentile 0.001; every significant cell keeps its sign.
- **BH head sets (ParT_full).**
  - top = {b1h4};
  - W/Z = {p4h5, p5h6, b1h3, b1h4, b1h6, b1h8};
  - lepton = {} (single heads do not move the lepton-class recall: b2h1 changes H→ℓνqq′ recall by −0.005, q 0.87, although its own-class log-odds change is −3.78).
  - Jaccard: top–W/Z 0.17, top–lepton 0, W/Z–lepton 0.
- **Top-5 sets (ParT_full).**
  - top = {b1h1, b1h3, b1h4, b1h8, b2h1};
  - W/Z = {p1h5, p8h4, b1h2–b1h6};
  - lepton = {b1h1, b1h2, b1h3, b1h5, b1h8, b2h1, b2h6}.
  - Jaccard: top–W/Z **0.20**, top–lepton **0.50**, W/Z–lepton 0.27 (random mean 0.04, 95th percentile 0.20).
  - 48 of the 50 top-5 slots (10 classes × 5) are class-block heads. b1h2 and b1h3 are in 8 of the 10 class sets. The mean off-diagonal Jaccard across the 10 classes is 0.33.
- **ParT_kin.**
  - 72 significant cells (48 drops), largest |Δrecall| 0.142.
  - BH sets: top {p1h6, b1h1, b1h3, b1h7, b2h3, b2h7}, W/Z 9 heads, lepton {p8h3, b1h2, b1h6, b2h4}. Jaccard top–W/Z 0.15, top–lepton 0, W/Z–lepton 0.08.
  - Top-5 sets: Jaccard 0.30, 0.22, 0.27; 39/50 slots in class blocks; mean off-diagonal 0.20.
- **Earlier heads.** b2h8 (the 85 GeV threshold head of G21–G22) changes no class recall significantly in either model (W→qq′ −0.008, q 0.11). b2h1 does not enter ParT_kin's lepton set: ParT_kin has no lepton flag input.
- **Nulls.** Untrained: 4 significant cells, largest |Δrecall| 0.025. Permuted: **91** significant cells, largest 0.166. The permuted net calls 56 % of all jets W→qq′, with near-tied scores, so single ablations flip many calls. The number of significant recall cells is therefore not a measure of trained structure; log-odds effect sizes are (trained up to −1.6 on the top contrast, nulls up to −0.28).

### Literal circuits (ParT_full; full network balanced accuracy 0.871, top recall 0.958)
| variant | k | balanced acc. (EVAL) [CI] | random sets mean / 95th pct | p | t-vs-QCD AUC (SUB) | random AUC mean | p | top recall | where true top jets go |
|---|---|---|---|---|---|---|---|---|---|
| literal | 5 | 0.180 [0.172, 0.188] | 0.103 / 0.120 | 0.010 | 0.924 | 0.724 | 0.149 | 0.000 | 99 % H→gg |
| literal | 10 | 0.183 | 0.114 / 0.165 | 0.020 | 0.932 | 0.843 | 0.208 | 0.000 | 99 % H→gg |
| cls_mlp | 3 | 0.281 [0.272, 0.290] | 0.115 / 0.186 | 0.040 | 0.960 | 0.672 | 0.040 | 0.030 | 76 % QCD |
| cls_mlp | 5 | 0.342 [0.333, 0.352] | 0.118 / 0.188 | 0.020 | 0.971 | 0.723 | 0.020 | 0.084 | 83 % H→gg |
| cls_mlp | 10 | **0.386 [0.376, 0.395]** | 0.161 / 0.319 | **0.010** | 0.977 | 0.839 | 0.020 | **0.139** | 74 % H→gg |
| heads_only | 10 | 0.483 [0.474, 0.493] | 0.208 / 0.397 | 0.010 | 0.982 | 0.878 | 0.020 | 0.188 | 70 % H→gg |
| cls_mlp | 80 | 0.747 | – | – | – | – | – | 0.874 | – |

- Circuits up to k = 40 stay below the full network. The cls_mlp k = 40 circuit gives top recall 0.663 and balanced accuracy 0.644 (random sets 0.453, p 0.01).
- **ParT_kin.** Top recall is 0 for every k ≤ 20 in the literal and cls_mlp variants. cls_mlp balanced accuracy: k = 5 0.225 (random 95th percentile 0.160, p 0.02); k = 10 0.261 (0.245, p 0.05); k = 3 0.187 (p 0.08). True top jets go mostly to H→cc / H→gg.

### Prong-merge edit (705 true t→bqq′ jets with all three edits)
- **Edit checks.**
  - |Δ jet mass| ≤ 0.005 GeV and |Δ jet p_T| ≤ 1.2·10⁻⁴ GeV.
  - The b-prong holds a median of 17 particles (30 % of jet p_T); the merges remove on average a 17.5, b 13.5, c 14.0 particles.
  - τ32 (= τ3/τ2, kT axes) moves from 0.450 by a −0.127 [−0.133, −0.120], b −0.022, c −0.112.
  - The b-prong is the most-displaced subjet in 61 % of the 627 jets without a tie (chance 33 %); by largest single-track significance, 45 %.

| model | edit | Δℓ_t [95 % CI] | top calls before → after (McNemar lost/gained) | top-5 heads patched: share [CI] | random 5-sets mean / 95th pct, p | top-10 share |
|---|---|---|---|---|---|---|
| ParT_full | a: b-prong | **−6.57 [−6.95, −6.21]** | 0.948 → 0.460 (354/10, p 5·10⁻⁹¹) | **0.66 [0.65, 0.67]** | 0.06 / 0.29, 0.01 | 0.71 |
| ParT_full | b: pT-matched W cluster | −3.15 [−3.45, −2.84] | 0.948 → 0.767 (130/3) | 0.57 [0.55, 0.59] | 0.06 / 0.22, 0.01 | 0.61 |
| ParT_full | c: whole W subjet | −3.25 [−3.53, −2.97] | 0.948 → 0.766 (136/8) | 0.53 [0.51, 0.55] | 0.06 / 0.22, 0.01 | 0.57 |
| ParT_kin | a | −2.00 [−2.22, −1.78] | 0.904 → 0.712 (160/25) | 0.37 [0.34, 0.40] | 0.06 / 0.24, 0.02 | 0.69 |
| ParT_kin | b | −1.28 [−1.46, −1.08] | 0.904 → 0.776 | 0.23 | 0.05 / 0.22, 0.04 | 0.65 |
| ParT_kin | c | −0.43 [−0.55, −0.31] | 0.904 → 0.887 (33/21, p 0.13) | −0.39 | 0.04 / 0.48, 0.97 | 0.16 |

- **Paired differences** (sign-flip p over jets):
  - ParT_full: a − b −3.43 [−3.87, −2.99] (p 5·10⁻⁵); a − c −3.32 [−3.76, −2.89].
  - Count-matched control b2 (836 jets): −4.12 vs a −6.33 on the same jets, difference −2.21 [−2.64, −1.78].
  - ParT_kin: a − b −0.72 [−1.00, −0.46], a − c −1.57, a − b2 −0.47 [−0.72, −0.22].
- **Displacement subset.** On jets where the b-prong is also the most-displaced subjet, Δℓ_t(a) is −9.02 [−9.38, −8.62] (ParT_full) and −3.19 (ParT_kin, which has no impact-parameter inputs). The subset therefore also differs in kinematics.
- **Re-sorting check.** Re-sorting the edited particles by p_T changes the logits by a median of 1.4·10⁻⁴ (full) and 2.5·10⁻⁴ (kin); the maximum over jets is 0.53 and 3.8 (a few jets).

## Reading (scoped to what was measured)
**Pre-registered expectations.**
- (i) ≤ 10 heads carry > 50 % of the top drop, Holm-significant: **met**. Five heads carry 58 % in ParT_full; ten carry 81 % in ParT_kin; random sets carry 5–16 %.
- (ii) A k ≤ 10 literal circuit with the class MLPs kept beats ≥ 95 % of random sets on balanced accuracy: **met in rank** (ParT_full k = 3–10, p 0.01–0.04; ParT_kin k = 5 and 10). But the circuit's top recall is 0.03–0.14 in ParT_full and 0 in ParT_kin, so the circuit does not make the top call.
- (iii) Block-structured map with top / W/Z / lepton Jaccard < 0.3: **met for top vs W/Z** (0.15–0.30). **Not met for top vs lepton** on fixed-size sets (0.50 in ParT_full: b1h8, b1h1, b1h3 and b2h1 serve both top classes and H→ℓνqq′).

**Competing hypotheses.**
- **Excluded:**
  - "no head above 5 % of the drop" (largest 15 %);
  - "literal circuits indistinguishable from random sets" (p ≤ 0.04 for k = 3–10 with class MLPs).
- **Not excluded (partly supported):** "high overlap across classes". Every class's top-5 set is drawn mostly from the same 8 class-block-1 heads (48/50 slots), so the classes share heads and differ in how strongly each head matters.
- **The b-prong edit.**
  - It acts more strongly than matched W-prong merges (a − b, a − c, a − b2 all negative, p ≤ 3·10⁻⁴), so ParT_full's top call depends on b-prong substructure beyond particle count, τ32 and jet four-momentum.
  - The same five heads carry a similar share (53–66 %) of the response to every prong merge, so patching shows where prong-merge information enters the class token. It does not show a b-specific route.

**Physics meaning.**
- ParT recognises a hadronic top through a few class-token heads in the first class-attention block. These heads pool over the whole jet. One of them (b1h8) and the lepton head b2h1 weight the b-prong and its leptons and charged particles above their particle share.
- Collapsing the b prong into one particle lowers the top probability about twice as much as collapsing a W prong of the same p_T.

## Issues / caveats
- `_125` does not exist; `_111` was used (see Runs). The W and Z rows of `_111` overlap at most 3 % with the rows P13v2 used.
- **Single-head recall effects are small** (≤ 0.11 in ParT_full), so the BH head sets are tiny (top = 1 head) and depend on statistical power. The fixed-size top-5 sets on log-odds are the main overlap read-out.
- **The permuted null gives more significant recall cells than the trained model** (91 vs 39): near-tied scores make recall flips cheap. A future map should use log-odds effects with a matched null threshold instead of McNemar counts.
- **The b-prong is the non-W subjet by mass**, which matches the displaced-track assignment in 61 % of jets. Edit a removes 4 particles more than b and c; b2 controls the count.
- **The merged particle is massive** (the prong's mass, not massless), so the jet four-momentum is exact.
- A few jets change by up to 3.8 logits when the edited particles are re-sorted (ParT_kin), against medians of order 10⁻⁴. This does not affect the means reported.
- Head descriptions use attention weights averaged over top jets. They are descriptive, not interventions.

## Explainer entry (for site/explain.py, Paper C)
```python
P37_EXPLAINER = [
 ('Question', 'Does ParT decide that a jet is a hadronic top quark ($t\\to bq\\bar q\'$) through a few attention heads, and are the heads each class depends on different for different classes (top vs $W/Z$ vs leptonic classes)?'),
 ('Background', 'ParT has 8 particle blocks and 2 class-attention blocks, each with 8 attention heads (80 heads). In the class blocks one extra vector, the class token, attends to all particles; its final state gives the 10 class scores. Mean ablation replaces a head\'s output $h(x)$ by its average over 2,000 reference jets, $\\bar h=\\frac{1}{N}\\sum_j h(x_j)$, so the head no longer carries information about the jet. Read-outs: $d_{QCD}=\\Delta\\log(p_t/p_{QCD})$; recall = share of a class\'s jets called that class; balanced accuracy = mean recall over the 10 classes (0.1 = constant call); Jaccard overlap $|A\\cap B|/|A\\cup B|$ of two head sets. Holm and Benjamini-Hochberg (BH) correct p-values for testing many heads; McNemar tests paired call changes.'),
 ('Method', 'Fresh JetClass test jets (files _111, default_rng(3725)): 1,000 per class for evaluation, 2 x 200 per class for two independent reference means, 1,000 extra top jets to rank heads. (1) Each head mean-ablated alone: change of $d_{QCD}$ and $d_W$ on top jets (Holm over 160 tests); change of every class\'s recall (80 x 10 map, McNemar, BH over 800); the k top-ranked heads ablated together vs all 80. (2) Literal circuits: only k heads compute, every other head and every MLP at its mean (variant: the two class-block MLPs kept), against 100 random k-head sets. (3) Edit on true top jets: the b-quark prong (third exclusive-$k_T$ subjet, the other two forming the pair nearest 80.4 GeV) merged into one particle with the prong\'s four-momentum, so the jet\'s four-momentum is unchanged; controls merge a W-prong cluster of equal $p_T$, a whole W subjet, or a W cluster with equal particle count. Patching: only the top-5 heads see the edited jet. Nulls: untrained and weight-permuted ParT.'),
 ('Result', 'Five heads (b1h8, b1h4, b1h3, b1h1 in the first class block and the lepton head b2h1) ablated together remove 58 % [57.7, 58.7] of the top-vs-QCD drop caused by ablating all 80 heads; random 5-head sets remove about 5 %. Merging the b prong lowers $\\log\\frac{p_t}{1-p_t}$ by 6.6 and halves the top calls (94.8 to 46.0 %); W-prong merges lower it by 3.1-3.2. The top-5 heads carry 66 % of the b-prong response and 53-57 % of the W-prong responses (random sets 6 %). Literal circuits of 3-10 heads with the class MLPs kept beat 96-99 % of random sets on balanced accuracy but reach only 0.28-0.39 (full network 0.87) and top recall at most 0.14. The class head sets are drawn from the same eight class-block-1 heads (48 of 50 top-5 slots); top vs W/Z overlap 0.20, top vs leptonic classes 0.50.', 'p37_top_circuit.png'),
 ('Checks', 'Unedited accuracy 85.9 % (ParT_full) and 74.5 % (ParT_kin) on the P1 reference sample, 87.1 % and 75.4 % on this sample. Sum of head outputs reproduces the attention output within $6\\cdot10^{-5}$. A second reference mean gives single-head effects correlated 0.9998 with the first. Edits keep the jet mass within 0.005 GeV. Untrained net: single-head effects at most 0.28 and no top signal to remove. Limits: single-head recall changes are at most 0.11, so significance-based head sets are small; the weight-permuted net, whose scores are nearly tied, gives more significant recall cells (91) than the trained net (39).'),
 ('Why it matters', 'ParT\'s top decision is concentrated in a few class-token heads that pool over the whole jet and weight the b prong, but these heads are shared with the other classes and respond to any prong merge; a circuit of them alone gives a top score, not the top call. Claim C17 is supported for concentration and refuted for class-specific head sets.'),
]
```
Glossary additions:
- **mean ablation**: a unit's output replaced by its average over reference jets.
- **joint knock-out share**: the drop from ablating k heads together divided by the drop from ablating all 80.
- **Jaccard overlap**: |A∩B|/|A∪B|.
- **patching share**: the change when only the named heads see the edited jet, divided by the change when the whole network sees it.
- **b-prong (P37)**: the third exclusive-kT subjet of a top jet, the other two forming the pair with mass nearest 80.4 GeV.

## Suggested ledger update
- **C17** (pre-registered 0.0): split it.
  - Concentration: "≤ 5–10 heads, mostly class-block-1 class-token heads, carry 58–81 % of ParT's top-vs-QCD drop and two thirds of the prong-merge response". Suggested confidence 0.6, pending a verifier.
  - Class-specific head sets: the head sets overlap and are drawn from one shared pool. Refuted at 0.2.
- **Verifier suggestions:**
  - a fresh file (`_113` TTBar);
  - resample ablation instead of mean ablation;
  - log-odds-based class maps with a matched null threshold;
  - a b-prong merge on jets with a truth-matched b (if JetClass exposes it) or with tighter displaced-track agreement.
