# P37 verification (P37v): top-quark head circuit and 80-head knock-out map of ParT, 2026-10-20

Jobs:
- `anrunw-ai-c-p37-verify` (V100, 1,404 s): `python -m interp.p37_verify --smoke && python -m interp.p37_verify`.
  Outputs: `results/part/p37_verify_{full,kin}.json`, `logs/c-p37-verify.log`.
- `anrunw-ai-c-p37-verify2` (1,168 s): `python -m interp.p37_verify_b`. This reruns the class-specificity test with 500 random sets.
  Outputs: `results/part/p37_verify_B_{full,kin}.json`, `logs/c-p37-verify2.log`.

Scripts: `interp/p37_verify.py` and `interp/p37_verify_b.py`. Both import P37's own functions (head hooks, prong finder,
merge edit, patching) and freeze P37's head ranking, read from `p37_top_circuit_<model>.json`. No P37 row is reused.

## Conclusion
The main numbers reproduce on new jets. Five attention heads of the class-token stage (the last two blocks, where a summary
token reads all particles) account for 59 % of the top-quark score drop, against 58 % in P37. They also carry two thirds of the
response when the b-quark prong is merged into one particle.

P37's comparison with random heads overstates this concentration. Every piece of information about the jet reaches the output
through the 16 class-token heads, so removing all 16 removes 100 % of the drop. Five random class-token heads carry 21 % on
average, not the 5 % P37 quotes for random heads drawn from all 80. The five ranked heads still beat every one of 200 such sets.

Removing these heads lowers the top-quark score about 3.5 standard deviations more than matched random sets do. No other class
moves by more than about 2 standard deviations. So, unlike P37's conclusion, the joint effect of this head set is largely
specific to the top class. That test is marginal after correcting for 10 classes (Holm p 0.04–0.06).

## Code review of `interp/p37_top_circuit.py`
- **Head decomposition.** Each head's output is computed at the attention-module output, before ParT's per-channel scaling
  (`c_attn`) and `post_attn_norm`:
  - the head's contribution is A_h·V_h·W_O[:, h-block];
  - the value is projected from the module's own value input (the normed sequence; in the class blocks, the normed
    [cls, particles]);
  - the `out_proj` weight is split with `view(C, H, HD)`, which matches the column order h·16 + d.
  The sum over heads plus the bias reproduces the module output within 6·10⁻⁵ (`recon_max`, both models and both runs).
  Block indices: modules 0–7 are `model.blocks` (particle blocks p1–p8) and modules 8–9 are `model.cls_blocks` (b1, b2). Correct.
- **Mean ablation and padding.** Head and MLP means are averaged over real query tokens only, using `~key_padding_mask` in the
  particle blocks and the class token in the class blocks. Padded keys stay masked. Means come from REF jets, which are disjoint
  from the ablated jets. The REF2 noise floor gives a correlation of 0.9998.
- **Network output.** `CFG` sets `for_inference=False`, so the model returns logits, not softmax outputs. The log-odds and the
  contrasts logit_t − logit_QCD are therefore what they claim to be.
- **Rows.** Rows are random (`rng.choice` then sorted), split by a permutation into EVAL, REF, REF2 and SCAN. There is no
  first-k subsampling. The SUB subset is drawn at random per class.
- **η build.** Every jet goes through `part_core.build` (η = jet_eta + sign(jet_eta)·Δη). `merge_edit` writes the merged
  Δη with the same sign convention. P1-sample accuracy reproduces in both runs (0.8593 and 0.7447).
- **Edits.**
  - The jet four-momentum is conserved: the largest |Δm_jet| is 0.006 GeV on the new sample.
  - Edit a (b-prong merge) removes 17.5 particles on average, against 14.5 for b and 14.6 for c. P37's count-matched control b2
    covers this difference.
  - The merged particle keeps the PID, charge and impact parameters of the hardest particle in the prong. This is a stated
    choice, not a bug.
- **In-sample ranking.** P37 ranks the heads on the SCAN jets and computes the joint knock-out on the same jets. On fresh jets,
  the share for the frozen top-5 set is 0.590 against P37's 0.582, so the selection bias is negligible.
- **Missing control (substantive).** The random head sets in steps 1a, 2 and 3 are drawn from all 80 heads. In ParT, the class
  token starts as a learned constant, and the class-block MLPs see only the class token. The 16 class-attention heads are
  therefore the only path from the particles to the output. Measured on the new sample:
  - mean-ablating all 16 gives 1.000 of the all-80 drop in both models;
  - patching all 16 with edited-run outputs carries 1.000 of every edit response.

  A random 5-of-80 set contains one class head on average, so the "random ≈ 5 %" baseline partly measures "class heads vs
  particle heads". This verification adds two stricter controls:
  - **class-matched**: the same number of class heads drawn from the 16 and particle heads drawn from the 64;
  - **block-matched**: the same number of heads drawn from each block.
- **Figure** (`p37_top_circuit.png`, read). Panel A correctly shows a real jet before and after the b-prong merge, at fixed
  jet mass of 159 GeV. Panels C, D and E draw their random bands from all-80 sets, which repeats the inflated baseline. Panel C's
  random curve is the mean of only 20 sets.

## Sample and procedure
- **Data.** JetClass test_20M `<class>_118.root` (P37 used `_111`), numpy `default_rng(3790)`, random rows. Per class:
  - EVAL: 300 jets per class (N 3,000);
  - REF: 200 jets per class, a fresh mean for the ablations;
  - TOPX: 1,000 extra true t→bqq′ jets, of which 704 admit all three merge edits.
- **Models.** ParT_full and ParT_kin (released weights).
- **Unedited accuracy.**
  - P1 reference sample: 0.8593 / 0.7447.
  - New EVAL sample: 10-class accuracy 0.853 / 0.739; t→bqq′ recall 0.950 / 0.900.
- **Read-outs.**
  - d_QCD = change of (logit_t − logit_QCD) on true top jets after mean ablation. Mean ablation means a head's output is replaced
    by its average over REF jets.
  - "Share" = mean d_QCD(set) / mean d_QCD(all 80 heads ablated).
  - ℓ_k = log p_k/(1−p_k) is class k's own log-odds.
  - Patch share = the change in ℓ_t when only the chosen heads receive their outputs from the edited jet (all else unedited),
    divided by the change when the whole network sees the edited jet.
- **Statistics.**
  - 95 % intervals come from 1,000 bootstrap resamples of jets.
  - Empirical p = (1 + number of random sets at least as extreme) / (1 + number of sets).
  - Edit differences use sign-flip permutation p values (20,000 flips).
  - Holm correction over the 10 classes in the specificity test.

## Results
### 1. Ranking reproduces (fresh TOPX, frozen P37 ranks)
- Spearman correlation of the 80 single-head scores with P37's: 0.991 (full) and 0.992 (kin).
- The top-5 sets are identical in both models: ParT_full b1h8, b1h4, b1h3, b1h1, b2h1; ParT_kin b1h7, b2h7, b1h5, b1h6, p4h5.

### 2. Joint knock-out share of the top-vs-QCD drop
Mean d_QCD with all 80 heads ablated: −9.58 (full), −7.40 (kin).

| set | share [95 % CI] | random from 80: mean (5–95 %) | class-matched: mean (5–95 %), max, p | block-matched: mean (5–95 %), p |
|---|---|---|---|---|
| full top-5 (5 class heads) | **0.590** [0.585, 0.595] (P37 0.582) | 0.065 (0.009–0.193) | 0.208 (0.062–0.402), max 0.590, p 0.005 | 0.330 (0.177–0.498), p 0.010 |
| full top-10 (6 class) | 0.647 [0.641, 0.653] (P37 0.640) | 0.135 (0.039–0.286) | 0.294 (0.127–0.515), p 0.015 | 0.372 (0.191–0.580), p 0.010 |
| kin top-5 (4 class) | 0.381 [0.375, 0.386] (P37 0.380) | 0.060 (0.012–0.141) | 0.171 (0.068–0.295), p 0.005 | 0.224 (0.136–0.323), p 0.010 |
| kin top-10 (8 class) | **0.808** [0.802, 0.813] (P37 0.807) | 0.138 (0.055–0.249) | 0.388 (0.239–0.572), p 0.005 | 0.558 (0.424–0.694), p 0.010 |

- All 16 class heads: 1.000 (both models). All 8 class-block-1 heads: 0.898 (full), 0.839 (kin). All 64 particle heads: 0.946
  (full), 1.112 (kin).
- **Hypotheses.** H0 = the ranked set is no more effective than a random set with the same class/particle composition.
  - H0 is rejected for all four sets (p ≤ 0.015 class-matched).
  - "Any five class heads would do" is excluded: the class-matched mean is 0.21, against 0.59.
  - "Uniform over class heads" predicts ≈ 5/16 ≈ 0.31 for five heads. This is also below 0.59.
  - The best of 200 random class-matched sets reaches 0.590 in ParT_full. So the ranked set sits at the top of the distribution,
    but other class-head subsets come close.

### 3. Class specificity of the joint knock-out (EVAL; 500 class-matched random sets; `p37_verify_B`)
Change of each class's own log-odds ℓ_k on its own jets, circuit vs class-matched random sets. z = (circuit − random mean) /
random s.d.

| model / set | t→bqq′ Δℓ (random mean ± s.d.), z, Holm p | largest other |z| | Δrecall t→bqq′ (random) |
|---|---|---|---|
| full top-5 | −6.80 (−2.06 ± 1.31), z −3.6, Holm 0.060 | t→bℓν z −2.0 (Holm 0.22); Z→qq +2.1 (score rises); H→gg −1.7 | −0.370 (−0.015), Holm 0.060 |
| kin top-5 | −3.68 (−1.32 ± 0.73), z −3.3, Holm 0.040 | H→bb −1.9 | −0.313 (−0.057), Holm 0.040 |
| kin top-10 | −6.88 (−3.33 ± 1.13), z −3.2, Holm 0.040 | H→bb −2.0 | −0.900 (−0.289), Holm 0.140 |

- **Hypotheses.** H0 = the set's knock-out lowers class k no more than matched random sets do.
  - For t→bqq′, H0 is rejected at Holm 0.04 in ParT_kin and is marginal (0.06) in ParT_full.
  - It is not rejected for any other class (Holm ≥ 0.22).
- The heads were ranked on top jets, so a top-leaning effect is expected by construction. The test shows that the effect holds on
  new jets and does not spread to the other classes beyond chance.
- This does not contradict P37's finding that the per-class top-5 head sets overlap (Jaccard 0.2–0.5). Those sets come from
  single-head effects, and the joint effect of this particular set is still concentrated on the top class.
- W→qq′ recall drops by 0.227 under the ParT_full top-5 knock-out (random sets 0.099), but its log-odds change is within the
  random spread (z −0.6).

### 4. Prong-merge edits (704 true t→bqq′ jets, fresh)
**Whole-network response.** Δℓ_t, edited − unedited:

| model | a: b-prong merge | b: pT-matched W-prong cluster | c: one W subjet | a − b | a − c |
|---|---|---|---|---|---|
| full | −6.10 [−6.47, −5.75] | −3.17 [−3.49, −2.85] | −3.17 [−3.48, −2.87] | −2.93 [−3.45, −2.48] (p 5·10⁻⁵) | −2.92 [−3.38, −2.49] |
| kin | −2.03 [−2.25, −1.83] | −1.32 [−1.52, −1.14] | −0.42 [−0.55, −0.29] | −0.71 [−0.98, −0.45] (p 5·10⁻⁵) | −1.61 |

- ParT_full top-call rate: 0.955 before; 0.490 after edit a; 0.743 after edits b and c. These match P37's −6.57, −3.15 and −3.25.

**Patch shares.** Only the listed heads see the edited jet:
- **ParT_full top-5.**
  - Edit a: 0.663 [0.653, 0.673]; class-matched 0.209 (5–95 %: 0.035–0.558), p 0.01; block-matched 0.353, p 0.02.
  - Edit b: 0.574 [0.556, 0.591]; class-matched 0.193 (p95 0.423), p 0.01.
  - Edit c: 0.537 [0.518, 0.555]; class-matched 0.190 (p95 0.421), p 0.01.
  - All 16 class heads: 1.000; class block 1: 0.835.
- **ParT_kin.**
  - Top-5 share for edit a is 0.371, which does not differ from class-matched sets (mean 0.196, p 0.18).
  - Top-10 share is 0.704 (class-matched 0.400, p 0.05).
  - For edit c, the top-5 share is −0.38, against a whole-network response of only −0.42.

Readings:
- The b-prong merge acts beyond matched W-prong merges.
- In ParT_full, the top-5 heads carry each merge response above class-matched random sets, with similar shares across the three
  merges (0.54–0.66).

### 5. Literal circuits vs class-matched random sets (SUB, 100 per class)
- **ParT_full, cls_mlp variant** (the ranked heads plus the two class-block MLPs kept, everything else at its mean):
  - k = 5: balanced accuracy 0.338 vs 0.221 (p95 0.315), p 0.04; top recall 0.08; t-vs-QCD AUC 0.981 vs 0.936 (p 0.02).
  - k = 10: 0.383 vs 0.258, p 0.03; top recall 0.12.
- **ParT_full, literal variant** (all MLPs at their mean):
  - k = 5: 0.174 vs 0.142 (p95 0.178), p 0.12;
  - k = 10: p 0.22.

  P37's literal p of 0.01 (against all-80 sets) does not survive the class-matched control.
- **ParT_kin, cls_mlp.** p 0.05–0.07; top recall 0 at k = 5 and 10.

## Issues
1. **Control.** The "random 5-head sets ≈ 5 %" baseline (steps 1a, 2, 3 and the figure) is drawn from all 80 heads. Because of
   the class-head bottleneck, the matched baseline is 0.21 (class-matched) to 0.33 (block-matched) for the joint share, and about
   0.2 for the patch shares. The concentration result survives this control, with a smaller margin.
2. **Specificity.** Under matched controls, the joint knock-out of the top-ranked set is largely top-specific (z −3.2 to −3.6
   for t→bqq′, |z| ≤ 2.1 for the other classes). The conclusion's "These heads are not specific to the top class" should be
   limited to the overlap of the per-class single-head sets.
3. **Literal circuits.** The literal variant (all MLPs ablated) is not distinguishable from class-matched random sets
   (p 0.12–0.22, ParT_full). Only the cls_mlp variant ranks above them (p 0.03–0.04).
4. **Not re-tested.**
   - The attention-weight descriptions: b1h8 and b2h1 put 54–58 % of their weight on the b-prong and 8–15 % on leptons.
     These are not interventions.
   - The 80 × 10 recall map and its Jaccard overlaps.
5. **Statistical power.** Specificity p values are limited by 500 random sets. The ParT_full t→bqq′ result is Holm 0.060.

## Corrected reading
- **Concentration.** In ParT_full, mean-ablating five ranked class-token heads (b1h8, b1h4, b1h3, b1h1, b2h1) removes 59 % of the
  top-vs-QCD drop caused by ablating all 80 heads.
  - All 16 class-token heads together remove 100 %.
  - Random sets of five class-token heads remove 21 % (5–95 %: 6–40 %); block-matched sets remove 33 %.
  - In ParT_kin, ten ranked heads remove 81 %, against 39 % (class-matched) and 56 % (block-matched).
- **Specificity.** Ablating the ranked set lowers the t→bqq′ log-odds 3.2–3.6 s.d. beyond class-matched random sets. No other
  class moves beyond about 2 s.d. (Holm 0.04–0.06 for top).
- **Circuits kept alone.** With the class-block MLPs kept, the ranked heads beat class-matched sets on balanced accuracy
  (p 0.03–0.04, ParT_full), but reach 0.34–0.38 and top recall ≤ 0.12.
- **Edits.** Merging the b-prong at fixed jet four-momentum lowers ℓ_t by 6.1, twice the 3.2 of either W-prong merge. The top-5
  heads carry 54–66 % of each merge response, against about 20 % for class-matched sets.

## Suggested explainer addition (Checks, P37)
('Checks', 'Fresh jets (files _118, seed 3790) with the head ranking frozen: top-5 share 0.590 (P37 0.582), ranking Spearman 0.99. All information reaches the output through the 16 class-token heads (ablating all 16 removes 100 % of the drop; patching all 16 carries 100 % of any edit), so the fair null draws the same number of class-token heads: random 5-sets of class-token heads remove 21 % (95th percentile 40 %), against 59 % for the ranked set (p 0.005). Ablating the ranked set lowers the top log-odds $\\ell_t=\\log\\frac{p_t}{1-p_t}$ 3.6 standard deviations more than matched random sets; no other class moves beyond about 2.')

Glossary additions: **class-matched random set**: a random head set with the same number of class-token heads (drawn from the 16)
and particle-block heads (drawn from the 64) as the tested set. **block-matched random set**: the same number of heads drawn from
each of the 10 blocks. **patch share**: the change in the top log-odds when only the chosen heads see the edited jet, divided by
the change when the whole network sees it.
