# P3v: verification of P3 (ParT_full W/Z mass circuit: b2h8 switch, head routes into n98, literal circuit), 2026-10-19

Job `anrunw-ai-c-p3-verify` (GTX 1080 Ti, 331 s), `python -m interp.p3v_verify_part`. Smoke run `anrunw-ai-c-p3-vsmoke` (TITAN Xp, passed).
Script `interp/p3v_verify_part.py`; it imports P3's `Stage`, `base_pass` and `eval_S`, so the class-stage re-implementation is the one
under test. Results `results/part/p3_verify.json` (smoke: `p3_verify_smoke.json`); log `logs/c-p3-verify.log`. No figure.
Supplement (section E, added in the second verifier pass): job `anrunw-ai-c-p3-verify2` (Tesla V100, 324 s), smoke
`anrunw-ai-c-p3-v2smoke` (passed), script `interp/p3v2_farside_part.py`, results `results/part/p3_verify2_farside.json`, log
`logs/c-p3-verify2.log`. It re-runs the two P3 parts that the first pass did not re-run: the particle far-side head patch, now
with a block-stratified null and 60 draws, and the G21b attention/value split.

## Conclusion
One attention head in ParT_full's last class-token layer (b2h8) moves its attention off the class token as the jet mass passes 87 GeV, no other head does this, and this reproduces on new jets. Under a real change of jet mass, however, this head carries only about 5 % of the change in the output unit that separates W from Z (n98); the heads of the first class-token layer carry nearly all of it, about half through one head (b1h3), and the circuit {b2h8, b1h3, n98} kept alone separates W from Z barely above chance. In the particle layers, the heads P3 named carry about twice the share that random heads carry, but only on W jets, only for a real mass change (not for a shift of the jet axis), and mostly through one head per layer. Still open: how the second class-token layer turns the first layer's output into n98.

## Code review of `interp/p3_wz_circuit_part.py`
- Head and layer indices: b2h8 = `cls_blocks[1]`, MHA head 7 (context dims 112–127 before out_proj); n98 = z[97] after the final
  LayerNorm. The query is the raw class token and keys/values are `pre_attn_norm(cat(cls, particles))`. c_attn is applied after
  out_proj with the `tbhd,h->tbdh` reshuffle. These match weaver-core 0.4.17. Re-implementation on fresh jets: max |Δlogit| 9.5e-6
  on S′ and 6.7e-6 on T′, 0 calls changed.
- Padding: masked with −inf in the class-attention softmax. The attention-weighted ΔR sums only over real particles.
- Rows: random rows (`rng.choice`, sorted for reading) from single-class files. No first-k split of class-ordered data. The P1
  sample (rows 3,000–4,000 of each class's test file) is used only for the accuracy check.
- Build: `part_core.build` with the sign(jet_eta) convention. Edits keep masks identical (checked). φ+0.06 shifts only real
  particles (max |Δm| 0.0065 GeV on T′). Angles ×1.1 gives Δm +9.06 GeV.
- Particle-block hook: per-head outputs `(A_h V_h) W_o,h` with weights unaveraged. The patch replaces only the selected heads.
- No indexing bug was found. The following are interpretation issues, not code errors:
  1. The resample "cal" in (b) is an observational slope between jets of the same class, divided by a causal edit slope. Resampling
     all 16 heads replaces the class token's whole input (the class token reads the particles only through these 16 context
     vectors), so z equals the donor's z. Yet `cal(all16)` is only 0.15 (P3) and 0.12 (here). So cal does not measure a head's
     share of the mass route. P3's "b1h3 carries 19 % of the slope" is a lower bound, not a share.
  2. The G21b decomposition separates attention weights from values, but per head only the attention part. No per-head test covered
     a head's total output under the real edit. Section B adds it.
  3. Particle embeddings reach the class token only through the 16 class-attention heads. The phrase "goes through b1h3 and the
     particle embeddings themselves" lists two routes that are nested, not parallel.

## Procedure
- Samples: JetClass val_5M, `numpy.default_rng(97)`, random rows, files not used by P3.
  - T′: 5,000 rows each of `WToQQ_122.root` and `ZToQQ_122.root` (10,000 jets; 8,240 with base mass 70–110 GeV).
  - S′: 1,000 rows from each class's `<sample>_123.root` (10,000 jets; 2,000 W/Z).
- Unedited accuracy: ParT_full 85.93 % on the P1 sample (same as P3); 86.04 % on S′; W/Z binary 82.85 % on S′.
- A. Threshold: as P3 (a), with 500 bootstraps and 100 logistic refits.
- B. Per-head activation patching under the real edits (new). Edits on T′: angles ×0.9, ×1.1, φ+0.06. For each head set G:
  - removal: the edited jet with ctx_G set to the unedited jet's values; removed share $=1-\overline{\Delta z_{97}^{\rm patched}}/\overline{\Delta z_{97}^{\rm edit}}$;
  - sufficiency: the unedited jet with ctx_G set to the edited values; share $=\overline{\Delta z_{97}^{\rm suff}}/\overline{\Delta z_{97}^{\rm edit}}$.
  - Sets: each of the 16 heads, block 1, block 2, all 16 (= 1 by construction, used as a check), {b2h8, b1h3}, all heads except b2h8.
  - Metrics: Δz97 in units of the s.d. of z97 on S′, and Δ(lW − lZ). Bootstrap with 500 resamples over jets. H0: share = 0, bootstrap
    two-sided p, Holm over 16 heads. Thesis test H0′: b2h8 removal share ≥ 0.5.
  - Control: a random 16-dim direction with the same per-jet norm as the head's edit change, added to the unedited ctx (5 draws per
    head). Compared on mean |Δz97|.
  - Calls: W/Z binary calls the edit changes, the fraction that each removal patch restores, and exact McNemar on correctness
    (edited vs patched), Holm over 16.
- C. P3 (b) repeated on S′: resample (same-class donor with |Δm| ≥ 10 GeV) for all 16, b1h3 and b2h8; mean ablation with McNemar.
- D. Circuits on S′, with one uniform donor per jet (excluding itself):
  - V_a {b2h8, b1h3} + n98 with MLP resampled, against 30 random head pairs plus a random active z coordinate;
  - all 16 heads + n98 with MLP resampled (new upper reference for V_a);
  - V_e n98 alone;
  - V_d heads only (MLP and read-out intact), run for all 120 head pairs and all 16 single heads (new).

## Results

**A. Threshold (T′, 9,598 W/Z jets in 50–140 GeV).** "Inside" means the P3 value lies inside the fresh 95 % interval.

| quantity | P3 | fresh |
|---|---|---|
| m50 (W+Z) | 87.10 [86.79, 87.39] | 86.64 [86.31, 87.01]; P3 point 0.09 GeV above the fresh interval |
| width | 4.25 | 4.29 [3.89, 4.68] |
| share hi → lo | 0.864 → 0.081 | 0.864 → 0.091 |
| m50 W only / Z only | 90.3 / 84.3 | 90.1 [89.5, 90.9] / 83.2 [82.1, 84.1] |
| span 50–130, b2h8 | 82.5 pt | 80.6 [79.3, 86.6] pt, Holm p 0.032 (floor) |
| next head span | b2h1 16.4 | b2h1 15.7 (Holm p 1) |
| Δshare ×0.9 / ×1.1 / φ+0.06 | +0.201 / −0.208 / −0.171 | +0.202 / −0.207 / −0.168 |
| curve prediction | +0.178 / −0.158 / 0 | +0.174 / −0.153 / 0 |

The W-only and Z-only midpoints are 7 GeV apart, so b2h8's share depends on more than mass. The real edit moves the share by
more than the observational curve predicts (−0.207 vs −0.153), and the φ shift moves it by −0.168 with no change in mass.

**B. Per-head patching under the real edit (T′, 8,240 W/Z jets with mass 70–110 GeV).** Δz97 under the edit: angles ×1.1
−0.340 s.d. [−0.348, −0.332]; φ+0.06 −0.285; angles ×0.9 +0.338.

| head set | removal ×1.1 | sufficiency ×1.1 | removal φ+0.06 | removal ×0.9 | removal of Δ(lW − lZ), ×1.1 |
|---|---|---|---|---|---|
| all 16 (check) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| block 1 (8 heads) | 0.987 [0.986, 0.988] | 0.994 | 0.990 | 0.983 | 0.987 |
| all but b2h8 | 1.001 | 1.007 | 0.999 | 1.000 | 1.003 |
| block 2 (8 heads) | 0.061 [0.059, 0.064] | 0.056 | 0.064 | 0.035 | 0.072 |
| b1h3 | 0.466 [0.461, 0.470] | 0.519 | 0.530 | 0.427 | 0.410 |
| b1h6 | 0.132 | 0.179 | 0.149 | 0.120 | 0.154 |
| b1h2 | 0.110 | 0.108 | 0.074 | 0.095 | 0.133 |
| b1h4 | 0.084 | 0.104 | 0.093 | 0.089 | 0.068 |
| b2h8 | 0.046 [0.044, 0.047] | 0.041 | 0.054 | 0.030 | 0.048 |
| {b2h8, b1h3} | 0.470 | 0.531 | 0.539 | 0.435 | 0.414 |

- Every single-head removal share is ≠ 0 (Holm p 0.032 at the floor for the listed heads).
- b2h8, H0′ share ≥ 0.5: p 0.002 (floor) for every edit.
- b1h3 − b2h8: +0.42 [0.41, 0.43] (×1.1), p 0.002.
- By class (×1.1): b2h8 removes 0.052 on W and 0.041 on Z; b1h3 removes 0.50 on W and 0.37 on Z.
- Random-direction control: the edit's own change in ctx moves |Δz97| more than all 5 random directions of the same norm for every
  head (×1.1: b1h3 0.196 vs random max 0.047; b2h8 0.019 vs 0.008). p = 1/6 = 0.17 is the floor for 5 draws.
- Size of the change in each head's output under ×1.1 (mean ‖Δctx_h W_o,h‖): b1h3 65, b1h6 57, b1h2 19, b2h8 8.5, b2h1 3.2.
  The b2h8 switch moves its context vector, but after out_proj the change is about 1/8 of b1h3's.
- Calls: ×1.1 changes 21.97 % of W/Z calls (2,197 jets). Restored by b1h3 43.4 %, b1h6 27.5 %, b1h2 9.3 %, b2h8 4.6 %, block 1
  97.5 %, block 2 6.3 %. McNemar (Holm) on correctness, edit vs patched: b1h3 p 1e-107, b2h8 p 4e-6. φ+0.06 and ×0.9 give the
  same order of restored fractions.

**C. P3 (b) on S′ (2,000 W/Z jets).** Real-edit slope Δz97 −0.296 s.d. per 10 GeV (P3 −0.295).

| | cal P3 | cal fresh | mean-ablation W/Z calls changed P3 | fresh |
|---|---|---|---|---|
| all 16 | 0.150 | 0.122 [0.082, 0.166] | – | 53.3 % |
| b1h3 | 0.191 [0.177, 0.207] | 0.185 [0.165, 0.209] | 12.1 % | 11.75 % (McNemar p 2e-6) |
| b2h8 | 0.006 [0.005, 0.008] | 0.0049 [0.0032, 0.0067] | 0.85 % | 0.50 % (5 vs 5 flips, p 1) |

The P3 values lie inside the fresh intervals. cal(all16) = 0.12 for a replacement that carries 100 % of the class token's input.
The resample slope is therefore diluted roughly 8× relative to the edit, and cal is not a share.

**D. Circuits on S′.** Unedited W/Z binary 82.85 %.

| circuit | W/Z binary fresh | P3 | 10-class |
|---|---|---|---|
| V_a {b2h8, b1h3} + n98, MLP resampled | 50.2 % (random pairs mean 48.6, max 49.5, p 0.03) | 52.9 % | 10.5 % |
| all 16 heads + n98, MLP resampled (new) | 62.95 % | – | 11.5 % |
| V_e n98 alone | 71.7 % | 71.6 % | 12.3 % |
| V_d {b2h8, b1h3}, MLP + read-out intact | 52.3 %, rank 24 of 120 pairs (median 50.5) | 52.6 % | 13.1 % |
| best V_d pairs (new) | b1h2+b1h3 65.7 %, b1h2+b1h8 65.5 %, b1h2+b1h5 65.1 % | – | – |
| best single heads, V_d (new) | b1h2 62.4 %, b1h3 52.0 %, b2h8 < 50.3 % | – | – |

- Even with all 16 heads kept, resampling the MLP units and the other 127 z coordinates leaves 63 %. The strict V_a family is
  capped by that before head choice matters.
- The two-head circuits that work best all contain b1h2. b1h2 carries 7–11 % of the mass-edit response (B), so its circuit value
  comes from information other than the mass response measured here.

## Assessment of P3's reading, item by item
| P3 statement | fresh measurement |
|---|---|
| b2h8 is the only head with the class-token switch; m50 87 GeV, width 4 GeV | m50 86.6, width 4.3, span 80.6 vs ≤ 15.7 pt |
| b2h8 follows the φ shift that leaves the mass unchanged | −0.168 (P3 −0.171) |
| b2h8's resample changes n98 by 0.6 % of the edit slope; mean ablation 0.85 % of calls | 0.49 %; 0.50 % |
| b2h8 carries < 1 % of the route | direct patch under the real edit: 3.0–5.4 % removal, 3.5–4.3 % sufficiency (< 1 % holds only for the resample metric) |
| b1h3 carries 19 % of the slope | 19 % is the diluted resample metric; direct patch 43–53 % (removal), 52–54 % (sufficiency); block 1 together 98–99 % |
| attention re-weighting 34 %, changed particle embeddings 68 % | E: 34.0 % [33.0, 34.9] and 67.8 %; b2h8's attention part 5.5 %, b1h3's 15.7 % (P3 5.6 %, 16 %). The particle embeddings are read through the block-1 heads, not beside them |
| circuit {b2h8, b1h3, n98} 53–54 % | 50.2 % (V_a), 52.3 % (V_d), rank 24/120 among head pairs |
| n98 alone 72–77 % | 71.7 % |
| far-side particle heads remove 2× the random share | E: on W jets under ×1.1, 0.259 vs 0.127 for block-stratified random sets (p 0.016, floor); on Z jets and under φ+0.06 not separated from random. The head-4 column carries 0.18 of the 0.26 |

Competing hypotheses for the head route of the mass response into n98, with the predicted b2h8 removal share:
- H_thesis: b2h8 is the main route, ≥ 0.5. Excluded (p 0.002 for every edit).
- H_distributed-block-1: block 1 ≈ 1 and b2h8 small. Matches (block 1 0.98–0.99, b2h8 0.03–0.05).
- H_block-2-other: another block-2 head carries it, so block 2 ≫ b2h8. Excluded (block 2 together 0.04–0.06).

**E. Supplement: particle far-side patch with a stratified null, and G21b (fresh sample).**
- Sample: JetClass val_5M `WToQQ_124.root`, `ZToQQ_124.root`, 2,000 random rows each, `numpy.default_rng(113)` (4,000 jets; files not
  used by P3 or P3v). ParT_full on this sample: 10-class accuracy 73.7 % (only two classes are present), W/Z binary 83.15 %. Class-stage
  re-implementation max |Δlogit| 5.7e-6; per-head particle-attention reconstruction max error 3.4e-5. Edits: angles ×1.1 (Δm +9.05 GeV),
  φ+0.06 (max |Δm| 0.006 GeV); masks identical.
- Patch: the edited jet is run with the per-head outputs $(A_{ih}V_{ih})W_{o,ih}$ of a set of particle-block heads replaced by the
  unedited jet's values (P3's hook). Removed share $=1-\overline{\Delta z_{97}^{\rm patched}}/\overline{\Delta z_{97}^{\rm edit}}$.
- Sets: far side = heads 4 and 8 of each of the 8 particle blocks (16 heads, 2 per block); all 64; the 8 head-index columns (head h in
  all 8 blocks); 60 block-stratified random sets (2 random heads per block, as the far-side set); 60 unstratified 16-of-64 sets (P3's
  null). H0: the far-side removal is drawn from the stratified null. p = (1 + #null ≥ far)/(1 + 60), floor 0.016. Bootstrap over jets
  (500) for the far-side interval.

| edit, class | Δz97 (s.d.) | far side removed | stratified null mean (sd, max) | p | all 64 | head-4 column | head-8 column |
|---|---|---|---|---|---|---|---|
| ×1.1, W | −0.440 | 0.259 [0.252, 0.265] | 0.127 (0.035, 0.236) | 0.016 | 0.470 | 0.180 | 0.080 |
| ×1.1, Z | −0.177 | 0.079 [0.057, 0.100] | 0.045 (0.023, 0.096) | 0.082 | 0.237 | 0.063 | 0.031 |
| ×1.1, W+Z | −0.308 | 0.208 [0.200, 0.215] | 0.103 (0.030, 0.192) | 0.016 | 0.403 | 0.146 | 0.066 |
| φ+0.06, W | −0.361 | 0.061 [0.048, 0.075] | 0.071 (0.025, 0.130) | 0.61 | 0.321 | 0.041 | 0.018 |
| φ+0.06, Z | −0.176 | 0.027 [0.003, 0.049] | 0.011 (0.019, 0.056) | 0.23 | 0.267 | 0.005 | 0.017 |

- P3's value on W under ×1.1 (0.257) lies inside the fresh interval. P3's unstratified null gives the same mean as the stratified null
  (0.126 vs 0.127), so block stratification does not change the comparison.
- Head-index columns under ×1.1 on W, ranked: h4 0.180, h8 0.080, h5 0.069, h2 0.052, h1 0.044, h3 0.041, h6 0.033, h7 0.026. Head 4
  carries 70 % of the far-side removal. Head 8 is second, but by 0.011 over head 5, so the {h4, h8} pairing is not singled out by
  this test beyond h4.
- All 64 particle heads together remove 0.47 (W, ×1.1) and 0.32 (W, φ): at least half of the edit's effect on n98 enters through
  the particle blocks' non-attention paths (embedding, pair-bias input, residual, MLPs), which the head patch does not touch.
- Competing hypotheses for the far-side heads: H_far (they carry the mass response beyond random heads; predicts far > null) holds on
  W under ×1.1 (z = 3.7 null s.d.), not on Z (z = 1.5, p 0.08). H_axis (they carry the axis-relative φ response) predicts far > null
  under φ+0.06: excluded on W (far 0.061 vs null 0.071, p 0.61).
- G21b on the 3,293 jets with base mass 70–110 GeV (decomposition into attention-weight change and value change, class stage):

| edit | attention | value | interaction | b2h8 attention part | b1h3 attention part |
|---|---|---|---|---|---|
| ×1.1 | 0.340 [0.330, 0.349] | 0.678 [0.669, 0.686] | −0.017 | 0.055 [0.052, 0.057] | 0.157 [0.152, 0.164] |
| φ+0.06 | 0.110 [0.094, 0.125] | 0.606 [0.589, 0.622] | 0.285 [0.271, 0.301] | 0.049 | 0.016 |

  P3's ×1.1 values (0.34, 0.68, 0.009, b2h8 0.056, b1h3 0.16) lie inside or at the edge of the fresh intervals; the interaction term
  differs in sign (−0.017 vs +0.009) and is small in both. Under φ+0.06 the split is not additive (interaction 29 %).

## Corrected reading
Within ParT_full's class-attention stage, b2h8 is the only head whose class-token share follows jet mass on W/Z jets. Its share
switches from 0.86 to 0.09 at 86.6–87.1 GeV with a 4 GeV width. Its midpoints differ for W and Z jets (90 vs 83 GeV). It moves
under a φ shift that leaves the mass unchanged (−0.17). Under real mass edits, patching b2h8's output alone removes 3–5 % of n98's
response and restores 4.6 % of the W/Z calls the edit changes. The eight block-1 heads together remove 98–99 %: b1h3 about half
(43–53 %), then b1h6 12–15 %, b1h2 7–11 % and b1h4 8–9 %. The block-2 heads together remove 4–6 %. The resample metric (cal)
understates head shares by about 8×: resampling all 16 heads gives 0.12–0.15, although it replaces the class token's entire input.
The literal circuit {b2h8, b1h3, n98} gives 50–53 % W/Z accuracy. The {b2h8, b1h3} pair ranks 24th of 120 head pairs. The best
pairs contain b1h2 (65–66 %). Under real mass edits the attention re-weighting is 34 % of n98's response and the value change 68 %
(reproduced). In the particle blocks, heads 4 and 8 of every block remove 0.26 of n98's response on W jets, against 0.13 for random
sets of 2 heads per block (p 0.016); on Z jets and under the φ shift they are not separated from random sets. Head 4 carries 0.18 of
the 0.26.

## Issues
- The random-direction control uses 5 draws (p floor 0.17). The effect ratios are 2–4×, but the p-value cannot go lower.
- Section E's p-values sit at the 60-draw floor (0.016) for the W/×1.1 rows. Holm over the 4 primary rows (W/Z × two edits) gives 0.064
  for W/×1.1; the null's max (0.236) is below the far-side value (0.259) by 0.023. The far-side interval resamples jets only; it does not
  carry the null's spread.
- P3 figure panel (b) plots cal, labelled "% of a real mass edit". Its "all 16 heads" line (15 %) lies below the single head b1h3 (19 %),
  so the panel cannot be read as shares (see code-review item 1). Panel (c)'s title says "rest resampled", but its fourth group (heads
  only) keeps the rest intact.
- Node k8s-haosu-06 was not hit. It is still not on part_job.sh's NotIn list.
- Bootstrap floors: 0.002 before Holm, 0.032 after.

## Explainer entry (for site/explain.py)
- **Question.** Which class-attention heads carry ParT_full's response to a real jet-mass change into the W/Z coordinate
  $z_{97}$ ("n98"), and is the 87 GeV switch head b2h8 one of them?
- **Background.** The class token reads the particle embeddings only through 16 head context vectors
  $c_h=\sum_p a_{hp} v_{hp}$ (8 heads in each of 2 blocks). P3 found that b2h8's class-token weight $a_{h0}$ switches at 87 GeV.
  P3 measured the heads' roles with a resample slope between different jets, which mixes mass with every other difference between
  jets.
- **Method.** Scale all particle angles by $k$ (mass $\times k$), or shift $\phi$ by 0.06 (mass fixed). For a head set $G$,
  removal share $R_G = 1-\overline{\Delta z_{97}[c_G\leftarrow c_G^{\rm base}]}/\overline{\Delta z_{97}}$ and sufficiency
  $S_G=\overline{\Delta z_{97}[c_G\leftarrow c_G^{\rm edit}]}/\overline{\Delta z_{97}}$. $R_{\rm all\,16}=1$ by construction. Bootstrap
  over jets, Holm over heads, random-direction control of equal norm $\|c_h^{\rm edit}-c_h^{\rm base}\|$.
- **Result.** $R_{\rm b2h8}=0.046$ [0.044, 0.047]; $R_{\rm b1h3}=0.47$; $R_{\rm block\,1}=0.99$; $R_{\rm block\,2}=0.06$ (angles $\times1.1$).
  The switch head b2h8 m50 = 86.6 GeV [86.3, 87.0].
- **Checks.** Re-implementation $|\Delta\ell|<10^{-5}$. $R_{\rm all\,16}=1.000$. Three edits agree. Fresh files and seed. P3's resample
  numbers fall inside the fresh intervals. Particle far-side heads on fresh jets: $R_{\rm far}=0.259$ vs block-stratified random
  $0.127\pm0.035$ (W, angles $\times1.1$); not separated on Z or under $\phi+0.06$. G21b attention share 0.34 reproduced.
- **Why it matters.** The switch exists but sits beside the route. The mass response reaches $z_{97}$ through the first
  class-attention block, about half of it through one head (b1h3). So thesis step 3 should name block-1 heads, not b2h8.
