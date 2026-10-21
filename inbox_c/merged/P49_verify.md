# P49 verify: L-GATr attention heads vs a fixed pair formula (fresh sample, shuffled-formula control, block-limited substitution)

## Conclusion
In our L-GATr, a fixed formula of particle-pair kinematics accounts for about a fifth of a typical attention head's pattern. On a fresh sample this replicates: median held-out R² 0.19–0.22, and single-head knock-outs remove up to 67 points, only in blocks 0–4. P49 reported that the formula restores 41–49 % accuracy (unedited 84–85 %). That overstates what the pair geometry contributes, for two reasons. First, the restored accuracy depends on keeping the network's own (live) attention to the global and reference tokens; when those logits are fixed, nothing is restored. Second, about 13 of the roughly 30 restored points remain when the formula's values are shuffled among the keys of each row, which removes all pair geometry. The geometry itself is worth 17 points [16, 18]. Substituting the formula only in blocks 0–4, with blocks 5–11 left live, also gives 50 % (the open question in P49), so the early-block attention does not act as a fixed geometric prior the way ParT's pair bias does (ParT keeps 85.0 % with block 1 driven by its bias alone, G107). Still uncertain: whether adding particle-identity scalars to the formula would change this.

## What was checked
- Code review of `interp/p49_lgatr_attention.py` and the launcher `interp/p49_job.sh`:
  - Head index = block·8 + head throughout (`pp_targets`, `sec_eval`, `sec_scan`, `_edit`).
  - The patched attention computes the logits from the model's own q, k (inner-product factors checked against the algebra: grade 1 (+,−,−,−), bivectors 0i −1 and ij +1, trivectors +1,+1,+1,−1, pseudoscalar −1), with scale (4·16+8)^−½.
  - Padded tokens are masked with −inf, and the pair sets use `pv` (valid × valid).
  - Token order is asserted (global first, count = particles + 3).
  - Rows are random per class and shuffled (no first-k).
  - `build` uses the corrected η sign (η = jet_eta + sign(jet_eta)·Δη). L-GATr reads the rebuilt E, px, py, pz.
  - Each edit touches only the stated logits.
  - No bug changes the numbers.
- Issues in how the results are presented, not in the code:
  1. The F1 family ("pp block") is not a formula-only circuit. It keeps the live row mean, the live logits from particles to the global and reference tokens, and the live query rows of tokens 0–2. The 41–49 % figure therefore measures formula plus live content. The circuit made only of formula and constants (F2, F1c) sits at its floor.
  2. P49 had no control for the formula's *pair structure* inside F1. Any spread in the particle-key logits changes the softmax normaliser, and with it the share of attention that goes to the reference tokens, which is live in F1. That control was missing; it is added below.
  3. The top-16 knock-out set was compared with random 16-head sets drawn from all 96 heads, most of which are in the dispensable late blocks. A block-matched control is added below.
  4. The ParT reference "about 80 %" (G107) is the share of ParT's logit variance carried by its pair bias, a learned function of pair features that a hinge formula reproduces with R² 0.86–0.99 (G108). The two numbers are not the same quantity, but they are comparable within about 0.1. A more direct comparison is substitution: ParT with every block's attention computed from the bias alone keeps 69.4 % against 85.4 % (G107), and with block 1 alone keeps 85.0 %.
  5. The untrained L-GATr reaches about the same median R² (invE MLP 0.20 vs 0.22 trained; in blocks 5–11 roughly 0.15–0.2 for both). The R² level alone therefore does not show what training learned. Only the early-block rise (0.5–0.6) and the grade-1 share (0.00 untrained) differ.
- Figure `results/lgatr/p49_lgatr_attention.png`, read:
  - Panel (a) stacks per-block medians, which exceed 1 in blocks 4–6 (noted by the worker).
  - The legend in panel (b) covers points in blocks 0–2.
  - Panel (c) is titled "every head's attention replaced by its fitted pair formula", but its green bars are F1 (formula plus live row mean and live reference-token logits). Its "uniform attention everywhere" floor (7.9 %) is a different floor from the "pp block uniform" floor (18.2 %) that the green bars should be compared with.

## Rerun (cluster Job c-p49-verify, GTX 1080 Ti, 71 min; `interp/p49_verify.py`, which imports P49's code unchanged and adds the controls)
- Model: same final checkpoint `model_run0.pt` (anrunw-ai-c-p5-lgatr2).
- Fresh sample: JetClass **test_20M `<class>_105.root`**. This file is not among L-GATr's training, validation, P1 or P49 files (data/jetclass2m/selection.json). 1,000 random rows per class, `default_rng(5949)`, N = 10,000. Fit split: 500 other random rows per class, `default_rng(5950)`, N = 5,000, with no overlap (asserted).
- Formulas were refitted on the new fit split. Hinge = additive ReLU-knot formula (12 knots per feature) on ParT's pair features part = {ln kT, ln z, ln ΔR, ln m²}, or on inv = {ln m², ln p_i·p_j}. MLP = one small network per head on the same features.
- Statistics: 95 % bootstrap over jets (1,000 resamples). McNemar = exact binomial test on the jets whose correctness differs between two runs, paired on the same jets.

### Replication (P49 → verify)
| quantity | P49 (val _122) | verify (test _105) |
|---|---|---|
| unedited accuracy | 84.24 % [83.54, 84.97] | 84.88 % [84.15, 85.55] |
| patched vs model forward, max \|Δlogit\| (125 jets) | 0.30 (0 calls changed) | 0.05 (0 calls changed) |
| median grade-1 (Minkowski) share of the row-centred pp logit | 0.241 | 0.244 |
| median held-out R²: part hinge / part MLP / invE MLP / inv MLP | 0.19 / 0.22 / 0.22 / 0.12 | 0.188 / 0.216 / 0.221 / 0.120 |
| same, P49's 21 Holm-significant heads | 0.55 / 0.57 / 0.56 / 0.50 | 0.547 / 0.571 / 0.559 / 0.503 |
| grade-1 part alone, those 21 heads | 0.78 | 0.782 |
| key-permuted null | 0.000 | 0.000 |
| knock-out b0h2 / b0h5 / b1h2 (uniform), Δacc points | −66.9 / −57.3 / −45.0 | −67.4 [−68.4, −66.3] / −57.1 [−58.2, −56.0] / −45.7 [−46.8, −44.7] |
| knock-out b6h0 / b11h3 | within ±1.1 | +0.04 [−0.14, 0.23] / −0.16 [−0.35, 0.03] |
| F1 part hinge, all blocks | 48.65 % | 48.21 % [47.18, 49.22] |
| F1 floor (particle keys uniform, rest live) | 18.18 % | 18.11 % |
| F1c part hinge / F1c floor | 16.54 / 16.42 % | 16.92 / 16.80 % (Δ +0.12 [−0.84, +1.14], McNemar p 0.84) |
| F2 part hinge (pure) / uniform everywhere | 11.35 / 7.89 % | 11.99 / 8.06 % |

### New control 1: is F1's recovery due to the pair geometry? (F1s = shuffled formula)
- Edit: every head's particle-to-particle logits are set to the fitted formula values (row-centred) plus the live row mean, as in F1. In F1s, the formula values are then permuted at random among the off-diagonal particle keys of each query row; the self-pair value stays in place.
- What this holds fixed: the set of values in each row, hence the softmax normaliser and the share of attention that goes to the global and reference tokens.
- What it destroys: which particle pair gets which value. If F1's gain over its floor came from pair geometry, F1s would fall to the floor. If it came from the logit spread and the attention share to the reference tokens, F1s would equal F1.

| run (all 12 blocks) | acc % [95 % CI] | Δ vs shuffled, points [CI]; McNemar |
|---|---|---|
| F1 floor | 18.11 [17.42, 18.89] | – |
| F1s part hinge (shuffled) | 31.07 [30.18, 31.97] | +12.96 [11.86, 14.05] vs floor; p 2e-111 |
| F1 part hinge | 48.21 [47.18, 49.22] | **+17.14 [15.97, 18.27]** vs F1s; lost 1,074 / gained 2,788, p 4e-173 |
| F1s / F1 inv hinge | 31.58 / 44.26 | +12.68 [11.45, 13.88] |
| F1s / F1 part MLP | 32.45 / 45.82 | +13.37 [12.13, 14.63] |
| F1cs / F1c part hinge (reference-token logits fixed) | 11.49 / 16.92 | +5.43 [4.65, 6.22] |

- H0 (the pair geometry adds nothing beyond the logit spread): F1 = F1s. Excluded (p 4e-173).
- H1 (all of F1's recovery is pair geometry): F1s = floor. Also excluded (+13.0 points, p 2e-111).
- Measured split of the floor-to-unedited gap (66.8 points): pair geometry 0.26, row spread with live reference share 0.19, not recovered 0.55. P49's "recovers 0.34–0.46" counts the 0.19 as formula recovery.

### New control 2: formula substitution limited to blocks 0–4, or to blocks 5–11 (other blocks fully live)
| run | blocks 0–4 replaced | blocks 5–11 replaced |
|---|---|---|
| uniform attention (all rows and keys) | 8.71 % | 61.37 % |
| F1 floor | 15.89 % | 81.21 % |
| F1c floor | 16.52 % | 81.43 % |
| F1s part hinge (shuffled) | 27.43 % | 81.34 % |
| F1 part hinge | **49.63 % [48.66, 50.58]** | 81.90 % |
| F1 part MLP / inv hinge | 49.59 / 43.00 % | – |
| F1c part hinge / MLP | 15.72 / 17.90 % | 81.80 % |
| F2 part hinge (pure) | 15.55 % | – |

- Blocks 0–4: the fitted formula in the 40 early heads, with every late block live, keeps 49.6 %. This is the same as replacing all 96 heads (48.2 %): the 35-point loss sits entirely in the early blocks.
  - With the attention to the reference tokens fixed (F1c), the early-block formula is at its floor: 15.7 vs 16.5 %, Δ −0.80 [−1.76, +0.09], McNemar p 0.11.
  - Pair geometry over the shuffled control: +22.2 points [21.0, 23.4].
  - Prediction under "early attention is a ParT-like geometric prior": substitution within a few points of 84.9 %, as ParT's block-1 bias-only run (85.0 vs 85.4 %, G107). Excluded.
- Blocks 5–11: making the late blocks fully uniform costs 23.5 points, although no single late head moves accuracy by more than 1.1 points. Keeping their non-particle rows and logits live and only flattening the particle-to-particle block (F1 floor) costs 3.7 points. The fitted formula adds 0.7 points [0.3, 1.1] over that floor. The late blocks' joint contribution therefore runs through the global and reference-token rows and keys, not through their particle-pair pattern.

### New control 3: block-matched random sets for the top-16 knock-out
P49's top-16 set (uniform ablation) gives 9.63 % [9.03, 10.23]. Ten random 16-head sets drawn only from blocks 0–4 give 12.4–47.8 % (mean 20.2 %), rank p = 0.091 (the smallest p possible with 10 sets). Against all 96 heads (P49) the rank p was 0.032. Most random early-block sets collapse the model about as much. The "21 decisive heads" are therefore mainly a statement about blocks 0–4, not a sparse head-level circuit; with 10 sets this does not separate them from other early heads.

## Corrections to the P49 reading
- Replicated and correct: H0 ("shared pair programme", R² ≥ 0.7 for most heads) is excluded. The median R² is about 0.2, and the decisive knock-outs are in blocks 0–4. In the significant heads the grade-1 (Minkowski) part dominates, and a pair formula explains 0.55–0.57 of their row-centred logits.
- To correct: "Replacing every head's attention with such a formula keeps 41–49 % accuracy". This holds only with the live attention to the global and reference tokens and the live tokens-0–2 rows kept. With those fixed, accuracy is at the floor (16.9 vs 16.8 %). Of the 30 recovered points, 17 [16, 18] come from the pair geometry and 13 from the logit spread alone.
- To correct: "The later heads and the attention to the reference tokens depend on learned per-particle content". The late blocks' particle-to-particle attention is nearly dispensable for accuracy (−3.7 points when flattened), even though no formula fits it. The late blocks matter through the global and reference-token rows and keys (−23.5 points when uniform).
- Open question now answered: the early-block formula substitution alone keeps 49.6 %, not near 85 %.
- "About 78 % … is a Minkowski product of the particles' states" is the grade-1 part of the network's own learned q·k, not a fixed formula of the momenta. The fixed formula reaches 0.55–0.57 on those heads, and substituting it loses 35 points.
- The comparison with ParT is better made on substitution: ParT keeps 69.4 % with all attention from its bias formula (85.0 % with block 1 only, G107). L-GATr keeps 16.9 % with formula plus constants (F1c), and 48.2 % with the live reference-token share.

## Files
- Script: `interp/p49_verify.py`. Results: `$AI_RESULTS/part/p49_verify.json` (local copy `results/part/p49_verify.json`; smoke run `p49_verify_smoke.json`). Log: `logs/c-p49-verify.log`.
- Glossary terms to add: F1s / F1cs (shuffled-formula control), block-limited substitution, block-matched random head set.
