# P63 verify: is the true jet mass present in ParT_full's layers, and does the decision use it?

Node P63v (loop C, skeptic of P63). Script `interp/p63v_verify.py` (imports P63's edit, capture, read-out and interchange functions unchanged). Jobs: c-p63-vsmoke (smoke), c-p63-verify (GPU, RTX 2080 Ti, 495 s). Results: `results/part/p63_verify.json`, log `logs/c-p63-verify.log`. Confidence that P63's reading is correct as stated: 0.55. Most of the reading holds. Four parts of it are wrong or unstable (list below).

## Conclusion
In ParT_full's particle blocks, read-outs can recover how the true invariant mass differs from the axis-based mass m_A, both across jets and within one jet when only m is changed. Untrained networks fed the same inputs give much less. Part of this information goes beyond what m_A and the particles' pT-weighted offset from the jet axis provide (about half of what is left after both). The W-vs-Z decision responds to a change of the true mass at fixed m_A only about a quarter to a third as much as to a real mass change. Still open: whether the blocks compute m or pass on what they get from the pair-feature input, which determines m almost exactly on its own. Also open: how much of the decision's response a small learned subspace of the class token carries. Two runs of this test gave 0.12 and 0.25.

## Corrections to P63's reading
1. **"On jets where only m was changed (prong turns) the trained blocks recover 0.37–0.54 and the untrained ones nothing" is misread.** The T-row partial R² is scored on T1 and T2 rows from different jets. Most of the variance it measures is the between-jet spread of the natural Δ = log m − log m_A (s.d. 0.026). The turn itself changes log m by a median of only 0.009. The test that isolates a true-mass change at fixed m_A compares T2 with T1 *within* each jet; P63 did not run it. Run here (difference probe, below), it gives trained b1–b8 0.60–0.73 and untrained 0.21–0.44. That is a clear gap, but the untrained networks do not recover "nothing". At the embedding (emb), trained and untrained give the same value (0.50 vs 0.53).
2. **"Linear read-outs on turned jets are about 0"** holds only for the ridge fitted on the pooled rows. That fit is dominated by the rigid-shift rows (s.d. of Δ is 0.093 on A rows vs 0.026–0.030 elsewhere). A ridge fitted on within-jet differences X(T2) − X(T1) predicts the held-out dlog m with R² 0.49–0.65 at b1–b8 (untrained 0.17–0.30).
3. **"The read-outs cannot separate 'computes m' from 'encodes m_A plus the centroid offset M1'"** is partly resolved. After conditioning on m_A *and* the 5 moment features (log M0, M1/M0 as two components, log M2/M0, |M1|²/(M0·M2)), the blocks still explain 0.40–0.56 of the remaining log m variance (untrained 0.05–0.12). So the blocks carry more than m_A + M1. They may get it from the pair path: the pair features alone fix m (oracle 0.995), and the pair-embedding hidden layer explains 0.73–0.77 beyond the moments in trained ParT and in untrained init 1. "Computes m" vs "relays the pair input" remains open.
4. **DAS numbers.** At b5, DAS-4 replicates (ρ 0.064 vs 0.067). Its fitting loss falls only from 0.96 to 0.90 here (1.15 to 0.89 in P63), so the value is a lower bound set by an under-converged optimiser. It does not measure what a 4-d subspace can carry. At cls1, DAS-4 does not replicate: ρ 0.246 [0.207, 0.284] vs P63's 0.116 [0.096, 0.135], with no overlap. On this sample that is 79 % of the input turn's effect, not 48 %. ρ_input itself is 0.31 [0.24, 0.38] here vs 0.24 [0.19, 0.29] in P63.

Parts that replicate: the pooled MLP partial R² (b1–b8 0.67–0.74 vs P63's 0.68–0.77; class tokens 0.45 vs 0.49–0.53; untrained 0.03–0.19); the trained-over-untrained gap at every block (Holm p at its floor); the profile over depth (highest at b1–b5, lower at b8, lowest at the class tokens); the pair path at the untrained level on the pooled partial; the 1-d probe swap having no effect; the shift result (true W called Z 11.3 → 30.9 % at 0.06 rad); the unedited accuracy.

## Definitions
- m: invariant mass of the summed particle four-vectors. M0 = Σ pT_i, M1 = Σ pT_i (Δη_i, Δφ_i), M2 = Σ pT_i ΔR_i², all about the stored jet axis. m_A = √(M0·M2) (axis-moment mass). For small angles, m² ≈ M0·M2 − |M1|². Δ = log m − log m_A.
- Partial R²(log m | X): 1 − SSE(model using X and the conditioning variables) / SSE(conditioning variables alone), on held-out evaluation rows. It is the share of log m variance left after the conditioning variables that site X explains. "Pool" = rows N, A, T1, T2. "noA" = rows N, T1, T2 only. "Beyond-mom" = the conditioning set is a cubic in log m_A plus the 5 moment features, with an MLP as the reduced model.
- Difference probe: the target is dlog m = log m(T2) − log m(T1) of the *same* jet. m_A is the same in T1 and T2 (max |dlog m_A| 3.7e-8), so a held-out R² > 0 can only come from true-mass information at fixed m_A, with no between-jet part. Ridge on X(T2) − X(T1). MLP (2 × 64 ReLU) on [X(T2) − X(T1), X(T1)]. Both are fitted on fit-split jets with an antisymmetric copy (−ΔX, −dlog m) and scored on evaluation jets.
- R_turn: slope of the pool-fitted log m read-out's change on the true dlog m under the turn, divided by the same slope under the matched dilation (all angles × k, m and m_A move together). corr_turn: correlation of the read-out change with the true dlog m over evaluation jets.
- ρ: change of the W−Z logit margin under an intervention, divided by s_dil · dlog m. s_dil is the margin slope per dlog m under the matched dilation (a real mass change); it is −11.17 here. ρ_input applies the same ratio to the turned jet itself.
- DAS-4: a learned 4-d orthonormal subspace at one site. Its component is copied from the donor run (T2) into the base run (T1) and trained to reproduce the donor's logits. IIA-informative: share of pairs, among those where donor and base calls differ, whose swapped call equals the donor's.

## Procedure
**Script review.** No indexing or hook errors were found. Particle-site features are jet means over real particles only (mask `mp`). The pair pooling masks padded particles and the diagonal. Rerun-from-site indices: 0 = after the embedding, l = after block l, 9 = class token after class block 1. The script's own forward pass matches weaver to |Δlogit| ≤ 1.8e-5 (TF32 off). Rows are random (`rng.choice`, sorted), never first-k. Every edit goes through `part_core.build`, η = η_J + sign(η_J)·Δη. Edit checks on this sample: the rigid shift changes |dlog m| by at most 6.4e-5 (5–95 %); the turn changes |dlog m_A| by at most 3.7e-8. The interchange adds the swapped component to padded slots too. This is harmless, because padded slots are masked as keys. Minor issues: the interchange site was chosen on the evaluation split's ridge partial R². The figure's panel (c) title ("not in untrained") and its hatched "prong-turn rows" bars carry the misreading in item 1. The figure also omits the moment baseline (0.81), which is above every block.

**Sample.** JetClass test_20M WToQQ_114, ZToQQ_114, ZJetsToNuNu_114, TTBar_114 (P63 used _112). 5,000 random rows per file (numpy `default_rng(6399)`), split at random per class into halves. 19,995 of 20,000 jets kept: 9,997 fit, 9,998 evaluation. Unedited ParT_full 10-class accuracy: 85.93 % on the P1 sample (expected 85.9 %) and 80.6 % on this 4-class sample.

**Edits** (same as P63). A: every particle shifted by 0.02, 0.04 or 0.06 rad in a random direction (m fixed, log m_A +0.026 median). T1, T2: the softer of 2 exclusive-kT subjets rotated about the stored axis by 10–40° (m_A fixed; dlog m median −0.009, 5 % quantile −0.040). D: all angles scaled so that m(D) = m(T1).

**Models.** ParT_full. Controls: untrained ParT_full, inits 0 and 1 (P63's). The weight-permuted net and ParT_kin were not rerun.

**Statistics.** H0: the statistic at a site is ≤ the untrained value. H1: it is larger. Paired Poisson bootstrap over evaluation jets (1,000 resamples); p = the larger of the two one-sided p (one per init); Holm over the 8 network sites (emb, b1, b3, b5, b8, cls1, cls2, pairH). The p floor is 1/1001, so the Holm floor is 0.008.

## Numbers (ParT_full vs untrained init 0 / init 1; 95 % bootstrap CI over evaluation jets)
| site | pool partial (MLP) | noA partial (MLP) | beyond-mom partial (MLP) | within-jet difference probe, MLP R² | difference probe, ridge R² | R_turn (pool MLP) |
|---|---|---|---|---|---|---|
| inp (input mean) | 0.078 | 0.118 | 0.128 | 0.448 | 0.386 | 1.10 |
| mom (5 moments) | 0.810 [0.758, 0.848] | 0.248 | – | 0.853 | 0.822 | 0.95 |
| emb | 0.529 / 0.360 / 0.441 | 0.214 / 0.155 / 0.198 | 0.196 / 0.105 / 0.144 | 0.500 / 0.526 / 0.538 (Holm p 0.98) | 0.377 / 0.317 / 0.294 | 0.26 / 1.05 / 0.69 |
| b1 | 0.742 [0.716, 0.765] / 0.180 / 0.190 | 0.507 / 0.163 / 0.142 | 0.516 [0.474, 0.568] / 0.124 / 0.089 | 0.695 [0.675, 0.712] / 0.441 / 0.421 | 0.590 / 0.299 / 0.287 | 0.36 / 1.19 / 0.57 |
| b3 | 0.732 / 0.115 / 0.105 | 0.513 / 0.136 / 0.097 | 0.561 / 0.099 / 0.060 | 0.725 [0.706, 0.742] / 0.349 / 0.332 | 0.645 / 0.262 / 0.237 | 0.30 / 1.23 / 0.39 |
| b5 | 0.733 / 0.083 / 0.081 | 0.462 / 0.102 / 0.085 | 0.469 / 0.079 / 0.059 | 0.679 / 0.278 / 0.274 | 0.605 / 0.222 / 0.206 | 0.31 / 1.39 / 0.22 |
| b8 | 0.668 / 0.070 / 0.029 | 0.388 / 0.082 / 0.057 | 0.401 / 0.065 / 0.047 | 0.602 / 0.214 / 0.218 | 0.492 / 0.173 / 0.175 | 0.25 / 1.25 / −0.10 |
| cls1 | 0.456 / 0.060 / 0.035 | 0.123 / 0.044 / 0.032 | 0.166 / 0.043 / 0.027 | 0.313 / 0.157 / 0.147 | 0.270 / 0.136 / 0.127 | 0.11 / 1.42 / −0.14 |
| cls2 | 0.450 / 0.033 / 0.026 | 0.218 / 0.039 / 0.031 | 0.151 / 0.036 / 0.023 | 0.284 / 0.131 / 0.128 | 0.214 / 0.118 / 0.106 | 0.13 / 1.67 / −0.11 |
| pairH | 0.456 / 0.350 / 0.614 (Holm p 1) | 0.688 / 0.295 / 0.621 | 0.773 / 0.180 / 0.729 (p 0.08) | 0.967 / 0.867 / 0.967 (p 0.95) | 0.796 / 0.778 / 0.802 | 0.87 / 0.83 / 0.96 |

Holm p = 0.008 (floor) for trained > untrained at b1–b8, cls1 and cls2 in every column except where shown, and for emb in the pool and beyond-mom columns (noA emb p 0.086). Differences trained − untrained init 0, within-jet MLP probe: b1 +0.25 [0.24, 0.27], b5 +0.40 [0.38, 0.43]. The untrained R_turn values above 1 come with low correlation (corr_turn 0.07–0.22): the untrained read-outs move 3–10 times more than the true dlog m, mostly as noise. The trained pool read-out correlates with the true turn change at only 0.18–0.30 (b1–b8) and 0.06 (class tokens). The pool-fitted direction mostly does not follow the within-jet change, even though a direction fitted to the differences does.

**Output and interchange** (ParT_full, 20,000 W/Z turn pairs, both orders; 624 informative pairs):
| intervention | ρ [95 % CI] | P63 | IIA-informative | calls changed | McNemar (right→wrong / wrong→right, p) |
|---|---|---|---|---|---|
| input turn (whole jet) | 0.311 [0.244, 0.381] | 0.241 [0.192, 0.291] | – | 3.12 % | – |
| 1-d probe swap at b5 | 0.0014 [0.0009, 0.0020] | 0.0008 | 0.005 | 0.04 % | 4 / 3, p 1.0 |
| DAS-4 at b5 (loss 0.96 → 0.90) | 0.064 [0.045, 0.081] | 0.067 [0.053, 0.082] | 0.215 | 0.91 % | 59 / 75, p 0.19 |
| 1-d probe swap at cls1 | 0.000 | 0.000 | 0 | 0 | 0 / 0 |
| DAS-4 at cls1 (loss 1.10 → 0.46) | 0.246 [0.207, 0.284] | 0.116 [0.096, 0.135] | 0.540 | 1.96 % | 137 / 160, p 0.20 |

Random-subspace null at b5 (100 random 1-d and 100 random 4-d swaps, 4,000 pairs): the 1-d probe ρ lies inside the null (rank p 0.46). DAS-4 ρ 0.073 vs the random-4-d 97.5 % quantile 0.017 (rank p 0.0099). Caveat on the 1-d test: the probe is fitted on pooled rows, and its read-out follows only about 0.3 of a within-jet true-mass change (ridge R_turn at b5 0.28 in P63). So even full use of that direction by the network would give ρ of at most about 0.3; the observed 0.001 is well below that bound.

## Competing hypotheses
- H_thesis (m never formed; true-mass information at the untrained level): excluded at b1–b8 by all four read-out tests, including the within-jet difference probe and the beyond-moment partial.
- H_moments (blocks carry only m_A and the centroid offset M1): excluded as a full account. Beyond-moment partial 0.40–0.56 vs untrained ≤ 0.12.
- H_relay (blocks carry the true-mass information that the pair input provides; the pair features fix m alone, oracle 0.995): not excluded. These tests cannot separate it from "blocks compute m".
- H_used (the decision follows m at fixed m_A, ρ ≈ 1): excluded. ρ_input is 0.24–0.31 over two samples, with no CI reaching 0.4.
- H_partly used through a low-dimensional class-token subspace: not excluded. DAS-4 at cls1 reaches 0.25 on this sample, but the size is unstable between runs.

## Output files
`results/part/p63_verify.json`, `results/part/p63_verify_smoke.json`, `logs/c-p63-verify.log`, script `interp/p63v_verify.py`.
