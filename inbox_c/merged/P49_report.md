# P49: what L-GATr's attention heads compute: pair invariants vs ParT's pair-bias hinge (loop C, round C11, exploratory)

## Conclusion
In our L-GATr, a fixed formula of particle-pair features reproduces only a small part of what the attention heads compute. The typical head has about 20 % of its pattern explained by such a formula, against about 80 % for ParT's pair bias (C1). The heads whose removal costs accuracy are all in blocks 0–4, and these are more formula-like: their Minkowski-product part alone explains about 78 % of their pattern. When every head's particle-to-particle attention is replaced by its fitted formula, accuracy falls from 84.2 % to 41–49 %; when attention is made uniform it falls to 18 %. When the formula also has to set how much each particle attends to the global and reference tokens, accuracy falls to that same uniform floor. Still open: a formula substitution limited to the decisive early heads was not run, so it is not known whether the early-block attention alone acts as a geometric prior the way ParT's does.

## Setup
- Model: final L-GATr checkpoint `model_run0.pt` of anrunw-ai-c-p5-lgatr2, written at 10:40 UTC on 2026-10-21 after iteration 199,999. The training config has `es_load_best_model: true`, so this file may hold the best-validation weights rather than the last iteration; that was not checked. Sanity check: on the P1 sample (rows 3000–3300 of each `_100` test file, N 3,000) accuracy is 83.73 %, against 84.27 % for it154999 (P39/P44).
- Control model: untrained L-GATr (random init, torch seed 0, same config).
- Main (held-out) sample: JetClass val_5M `<class>_122.root`, 1,000 random rows per class, `default_rng(4949)`, N = 10,000, shuffled once. **Deviation from the plan:** the plan named `_123`, but that file is L-GATr's validation file for QCD, Hcc and Tbqq (data/jetclass2m/selection.json). `_122` is in neither its training nor its validation files.
- Fit split: the same files, 500 other random rows per class, `default_rng(4950)`, N = 5,000. The fit rows and main rows do not overlap (checked with an assert).
- Unedited accuracy on the main sample: **84.24 % [83.54, 84.97]**. W→Z (share of true W jets called Z) is 11.1 %.
- Plumbing: the attention is re-implemented densely per jet, using the model's own q, k and v and the same 1/√72 scale. Against the model's own xformers forward it gives max |Δlogit| 0.30 (0.16 in a second job) and changes no call in 125 jets. P44 found a run-to-run floor of max 1.0 for the model's own forward. The particle order in the tokens matches the jet arrays to 8e-8 relative.
- Jobs: c-p49-run (GTX 1080 Ti, 112 min), c-p49-extra2 (follow-up circuits, Quadro RTX 6000, 29 min), c-p49-smoke3 (smoke test on it174999). c-p49-smoke, -smoke2 and -extra were rejected at admission (UnexpectedAdmissionError, "GPU is lost") on k8s-chase-ci-01 and k8s-haosu-21. The launcher `interp/p49_job.sh` is part_job.sh with those two nodes excluded.
- Script: `interp/p49_lgatr_attention.py`. Results: `results/lgatr/p49_lgatr_attention.json`, `p49_arrays.npz` (per-jet calls and per-jet R² sums), `p49_lgatr_attention_circuit2.json`, figure `p49_lgatr_attention.png`. Logs: `logs/c-p49-run.log`, `logs/c-p49-extra2.log`.

## Definitions
- **Multivector, grade**: an element of the 16-dimensional space-time algebra. Grade 0 is the scalar (1 component), grade 1 a four-vector (t, x, y, z), grade 2 a bivector (6), grade 3 an axial vector (4), grade 4 the pseudoscalar (1).
- **L-GATr attention logit** of head h, from query token i to key token j: $a_{ij} = \big[\sum_{c=1}^{4}\langle q_i^c, k_j^c\rangle + \sum_{s=1}^{8} q_i^s k_j^s\big]/\sqrt{72}$. Here $\langle\cdot,\cdot\rangle$ is the algebra's invariant inner product. Its **grade-1 part is the Minkowski product** $q_t k_t - q_x k_x - q_y k_y - q_z k_z$. We split $a_{ij} = A_0 + A_1 + A_{234} + A_S$ by grade, where $A_S$ comes from the 8 auxiliary scalar channels.
- **Tokens per jet**: 0 = global token (the class is read out from it), 1 = beam reference (x–y-plane bivector), 2 = time reference (1,0,0,0), 3… = the particles in pT order. The **read-out row** is the global token's attention over the jet.
- **Row-centred**: softmax over keys j does not change when a constant is added to a whole query row. All logits, parts and formulas are therefore compared after subtracting their mean over the particle keys of the row. The **pp block** is particle query rows × particle keys.
- **Variance share** of part P: $\mathrm{Cov}(P, a)/\mathrm{Var}(a)$, computed on row-centred values pooled over pp pairs. The shares sum to 1 per head.
- **Pair features**: $m^2_{ij} = (p_i+p_j)^2$; $p_i\cdot p_j$ (Minkowski product of the four-momenta); $E_iE_j$ and $E_j$ (lab-frame energies, which are not boost-invariant); ParT's $k_T = \min(p_{T,i},p_{T,j})\,\Delta R$, $z = \min(p_T)/(p_{T,i}+p_{T,j})$, and $\Delta R$ from rapidity and φ. Feature sets: **inv** = {ln m², ln p·p}; **invE** = inv + {ln E_iE_j, ln E_j}; **part** = {ln kT, ln z, ln ΔR, ln m²}, the C1 inputs.
- **Hinge formula** (C1 form): for each feature, x plus 12 terms ReLU(x − knot), with knots at the fit-split quantiles, plus a diagonal (i = j) indicator. Fitted by row-centred least squares. **MLP**: one per head, input → 16 → 16 → 1 (GELU), trained 3,000 steps on row-centred targets from 2,000 fit jets.
- **Held-out R²**: $1 - \sum(\hat a - a)^2/\sum a^2$ over row-centred pp logits of the 10,000 main jets, with a 95 % bootstrap over jets. **Null**: the same hinge fit after the keys of every row are randomly permuted among that row's particles. This destroys the pair structure and keeps the row statistics.
- **Uniform ablation** of a head: all its logits set equal, so it attends uniformly to the jet's tokens. **Resample**: the head's logits are taken from the jet nearest in jet pT within the batch; particle k takes the donor's particle k mod n.
- **McNemar**: exact binomial test on the jets whose correctness changes (lost vs gained). **Holm**: step-down correction over the 96 heads.
- **Substitution families**, every head of every block replaced at once, with values, MLPs and everything else live:
  - **F1 (pp block)**: particle-row logits to particle keys come from the formula, plus the live row mean. The logits to the global and reference keys and the rows of tokens 0–2 stay live. Floor F1-0: formula = 0.
  - **F2 (pure)**: particle rows come from the formula, plus fitted per-head constants for the 3 non-particle keys. The rows of tokens 0–2 are uniform. Floor: uniform attention everywhere.
  - **F2g**: F2 with the read-out row live.
  - Follow-up **F1c**: F2g with the reference-token rows also live. Only the logit to the non-particle keys differs from F1 (fitted constant vs live).

## Numbers
Step 1, logit decomposition (pp block, median over 96 heads): grade 0 0.026, grade 1 (Minkowski) 0.241, grades 2–4 0.325, auxiliary scalars 0.383. The grade-1 share is 0.50 (median) in blocks 0–3 and 0.20 in blocks 4–11. The auxiliary-scalar share is 0.56 in blocks 8–11. Read-out row: grade 1 0.16, scalars 0.58. Untrained L-GATr: grade 1 0.000, scalars 0.82.

Step 2, held-out R² per head (96 heads; the bootstrap intervals are ±0.01 or less per head):

| formula | median R² | heads ≥ 0.7 | heads < 0.3 | median, blocks 0 / 2 / 6 / 11 |
|---|---|---|---|---|
| inv, MLP | 0.12 | 3 | 66 | 0.39 / 0.50 / 0.08 / 0.02 |
| invE (+ ln E), MLP | 0.22 | 5 | 61 | 0.62 / 0.54 / 0.27 / 0.07 |
| part, hinge (C1 form) | 0.19 | 3 | 61 | 0.52 / 0.53 / 0.25 / 0.06 |
| part, MLP | 0.22 | 5 | 61 | 0.61 / 0.55 / 0.26 / 0.07 |
| grade-1 part alone (not a pair formula) | 0.36 | 21 | 38 | 0.39 / 0.79 / 0.55 / 0.17 |
| null (keys permuted), all sets | 0.000 | 0 | 96 | – |
| untrained: invE MLP / part MLP / inv MLP | 0.20 / 0.19 / 0.03 | 1 / 1 / 0 | – | – |

ParT reference (G107/G108): the pair bias explains a median of 0.79 of each head's logit variance, and its hinge formula reproduces the bias with R² 0.86–0.99.

Step 3(i), single-head knock-outs (N 10,000, McNemar, Holm over 96):
- Uniform ablation: 21 heads are significant, all in blocks 0–4. Largest effects: b0h2 −66.9 points [−67.9, −65.8]; b0h5 −57.3; b1h2 −45.0; b1h1 −35.4; b0h6 −33.3; b2h5 −30.7. No head in blocks 5–11 changes accuracy by more than 1.1 points.
- Resample ablation: 20 heads significant, the same set plus b4h4 and b9h1.
- What the 21 uniform-significant heads compute: R² 0.56–0.57 (median; part MLP / invE MLP) and grade-1 part alone 0.78. b0h2: R² 0.79, Minkowski share 0.50. b0h5: R² 0.89.
- Set control: the 16 largest-drop significant heads ablated together give 9.6 % [9.0, 10.2]. 30 random sets of 16 heads give a mean of 40.1 % (2.5–97.5 %: 14.5–74.8 %); rank p = 0.032.

Step 3(ii), formula substitution in all 96 heads (10-class accuracy [95 % CI]; recovered = (acc − floor)/(84.24 − floor); McNemar lost/gained vs unedited, p < 1e-300 for every row):

| run | acc % | same call | recovered | W→Z |
|---|---|---|---|---|
| unedited | 84.24 | 1 | – | 0.111 |
| uniform everywhere (F2 floor) | 7.89 [7.35, 8.41] | 0.076 | – | 0.000 |
| read-out row uniform only | 9.84 | 0.098 | – | 0.001 |
| F1 floor (pp uniform) | 18.18 [17.46, 18.94] | 0.185 | 0 | 0.002 |
| F1 inv hinge / MLP | 44.36 / 44.48 | 0.458 / 0.457 | 0.40 / 0.40 | 0.205 / 0.243 |
| F1 invE hinge / MLP | 40.64 / 45.08 | 0.422 / 0.472 | 0.34 / 0.41 | 0.189 / 0.207 |
| F1 part hinge / MLP | **48.65 [47.57, 49.63]** / 46.06 | 0.517 / 0.479 | 0.46 / 0.42 | 0.144 / 0.065 |
| F2 (pure), 6 variants | 11.35–13.85 | 0.12–0.14 | 0.05–0.08 | ≤ 0.02 |
| F2g floor / 6 variants | 17.76 / 13.84–17.02 | 0.14–0.17 | −0.06 to −0.01 | ≤ 0.05 |
| F1c floor / 6 variants (follow-up) | 16.42 / 14.41–19.82 | 0.14–0.20 | – | ≤ 0.07 |
| only the 2 reference-token rows uniform | 60.03 [59.15, 61.02] | 0.632 | – | 0.338 |
| F1 + reference rows uniform, 6 variants | 29.92–36.64 | 0.30–0.38 | – | 0.08–0.19 |

- Per-class recall under F1 part hinge, against unedited: QCD 0.49 (0.74), Hbb 0.46 (0.91), Hcc 0.34 (0.82), Hgg 0.42 (0.78), H4q 0.25 (0.83), Hqql 0.73 (0.98), Zqq 0.29 (0.66), Wqq 0.47 (0.79), Tbqq 0.57 (0.94), Tbl 0.84 (0.99). The full 10 × 10 confusion matrices are in `circuit.runs.<run>.confusion`.
- Invariant-only vs with ln E: F1 inv MLP 44.5 % vs invE MLP 45.1 %; in hinge form, inv 44.4 % vs invE 40.6 %.

Step 3(iii), untrained L-GATr: it calls almost every jet the same class (9.91 %). Formula substitution changes 0.2–4 % of its calls. Mean |Δlogit| is 0.007–0.09, against a logit spread of 1.52; for the trained model under F1 part hinge it is 6.3, against a spread of 4.7. Untrained accuracy therefore carries no information about the formula. Its R² (invE/part MLP 0.19–0.20) is close to the trained model's.

## Reading
- **H0** (shared pair program, A8/C1 carry over): held-out R² ≥ 0.7 for most heads, and substitution within about 2 points of 84.2 %. This is excluded: 5 of 96 heads reach 0.7, and the best substitution is 48.7 %.
- **H1** (logits depend on learned multivector content; R² < 0.3; substitution near uniform-attention accuracy): this holds for the late blocks (blocks 5–11, R² 0.02–0.33) and for the pure formula F2 (11–14 %, near the 8 % floor). It does not describe the early blocks: blocks 0–2 have R² 0.5–0.6, and their single-head knock-outs carry the accuracy loss.
- Measured: the class decision depends on attention in blocks 0–4 (21 Holm-significant heads; the joint set at rank p 0.032 against random sets). In those heads the Minkowski grade-1 product is the largest part of the logit, and 56–89 % of the pattern is a function of pair features (b0h5 0.89).
- Late-block attention (blocks 5–11) is mostly auxiliary-scalar content that no pair formula fits. It is also individually dispensable: |Δacc| ≤ 1.1 points per head.
- The F1 → F1c comparison isolates one thing. The amount of attention each particle sends to the global and reference tokens, compared with the other particles, is content-dependent. Fixing it to a per-head constant removes all the formula's recovery (48.7 → 16.5 % for part hinge). The two reference tokens' own query rows also matter: making only those rows uniform gives 60.0 %.
- ln E (frame-dependent) adds 0.10 to the median R² over invariants alone. It adds 0.6 points of accuracy under the MLP and costs 3.7 points under the hinge.
- Physics: the attention that matters in L-GATr's first blocks weights particle pairs mainly by their Minkowski product, an invariant-mass-like quantity, much as ParT's pair bias weights them by m², kT and ΔR. Unlike ParT, the attention in later blocks and the attention to the reference tokens depend on the learned per-particle state, and L-GATr loses most of its accuracy without them.

## Caveats
- The substitution replaces all 96 heads, including the 75 non-significant late heads that the formula fits poorly. A substitution limited to blocks 0–4 (or to the 21 significant heads) was not run and is the direct test of the early-block prior.
- Pair features exclude particle identity (charge, PID) and displacement, which L-GATr also receives as scalars. Part of the unexplained variance may be a fixed function of those (ParT's G110 analogue).
- The final checkpoint may be the best-validation model (es_load_best_model). Holm and McNemar p-values are at the float floor for the large effects.
- The untrained control is informative for R² only, not for accuracy.

## Explainer draft (for site/explain.py, Paper C)
- **Question**: Is L-GATr's attention, like ParT's, a fixed formula of particle-pair geometry?
- **Background**: In ParT each attention logit is a learned content term plus a pair bias $U_h(\ln k_T, \ln z, \ln\Delta R, \ln m^2)$, which is a hinge formula (C1). L-GATr has no pair bias. Its logit is an inner product of query and key multivectors, $a_{ij} \propto \sum_c \langle q_i^c,k_j^c\rangle + q_i^s\cdot k_j^s$, whose grade-1 part is a Minkowski product. When $q$ and $k$ are proportional to the four-momenta, this part equals $p_i\cdot p_j = (m_{ij}^2 - m_i^2 - m_j^2)/2$.
- **Method**: (1) Split every head's logit by grade and measure each part's share of the row-centred variance. (2) Fit $\tilde a_{ij} = f_h(\text{pair features})$ on 5,000 jets and measure held-out $R^2$ on 10,000 others, with a key-permuted null. (3) Knock out each head (uniform or resampled attention) with McNemar and Holm, and compare the top set with random sets. Then replace every head's attention with $\mathrm{softmax}_j f_h$ and compare with uniform attention (floor) and the unedited model (ceiling).
- **Result** (figure p49_lgatr_attention.png): median $R^2$ is about 0.2 against ParT's 0.8; blocks 0–2 reach 0.5–0.6, blocks 8–11 stay below 0.2. The heads that matter (blocks 0–4) are dominated by the Minkowski part. With all heads replaced by their formulas, accuracy is 44–49 % (floor 18 %, unedited 84.2 %).
- **Checks**: the re-implemented attention reproduces the model to max |Δlogit| 0.3 with no call changed; the null gives R² 0.000; random head sets of the same size cost much less than the top set (40 % vs 9.6 % accuracy).
- **Why it matters**: the ParT finding that "attention is mostly a geometric prior" (C1) holds in L-GATr only for its first blocks. A shared pair stage across big taggers is at most an early-block property.
- Glossary terms to add: multivector grade, Minkowski product, row-centred logit, variance share, held-out R², uniform / resample head ablation, formula substitution (F1/F2), McNemar, Holm.
