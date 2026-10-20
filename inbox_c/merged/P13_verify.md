# P13 verify: feature circuit in ParT_full on fresh jets, decomposition of the live paths, and m_A steering dose and joint controls (2026-10-20)

Jobs:
- `anrunw-ai-c-p13-verify` (exit 0, 16 min; smoke then full in one job), `python -m interp.p13_verify`. Script `interp/p13_verify.py`
  was written by an earlier verify attempt of this node; this attempt reviewed it line by line before using its output. It imports
  P13's runner, mask hook, dictionaries and data helpers, and re-implements only the sample choice and the added controls.
- `anrunw-ai-c-p13-verify2` (GTX 1080 Ti, 83 s, exit 0; smoke then full), `python -m interp.p13_verify2`. Script `interp/p13_verify2.py`.
  It adds one control: which kept sets carry the "main" accuracy when particle attention and the skip maps stay live.

Results: `results/part/p13_verify.json` and `results/part/p13_verify2.json` (cluster: `$AI_RESULTS/part/`). Logs:
`/j-jepa-vol/aaron-interp/logs/c-p13-verify.log` and `c-p13-verify2.log`. Dictionaries: P13's `p13_dicts.pt`, reused unchanged.

## Conclusion
On fresh jets, the P13 numbers reproduce within 1–2 points:
- 50 learned features kept alone give 51 % W-vs-Z accuracy; the model gives 83 %.
- About 1,000–2,000 features are needed for 74–78 %.
- With the attention outputs and the transcoders' linear bypass also replaced, every variant stays near 43 %.

A decomposition shows how the gain splits. All ~1,300 class-token features together, with every particle-level feature replaced, give
59 %; the particle-level features add about 15 points. The feature most correlated with the axis-moment mass m_A does change the calls
when pushed far enough (4–8 standard deviations), but toward W rather than Z. Pushing all features of one class-token layer together
reproduces how the W/Z score varies with m_A across natural jets of the same class. That shift is about one seventh of the shift caused
by the φ + 0.06 rad edit (the rigid φ shift that leaves the invariant mass unchanged).

Still uncertain:
- whether a trained (not attribution-ranked) feature or feature-plus-head selection gives a short circuit;
- what carries the remaining six sevenths of the φ response.

## Code review (interp/p13_transcoder_part.py)
- **Forward pass.** `Runner.block` follows weaver-core 0.4.17's `Block.forward`:
  - class attention uses the un-normed class token as query and `pre_attn_norm(cat(cls, x))` as keys and values, with the padding
    mask extended by one slot;
  - then the `c_attn` head scale, `post_attn_norm`, `pre_fc_norm`, fc1 + GELU, `post_fc_norm`, fc2, and the `w_resid` residual.

  `max |Δlogit|` against `model(...)` is 0.0 in P13 and again in the verify run. Hook sites:
  - P0–P7: the 8 particle-block MLPs, input `pre_fc_norm` output, target fc2 output;
  - C0, C1: the 2 class-block MLPs;
  - R0, R1: the class token after each class block.

  These match the stated sites.
- **Padding.** Particle-site training, FVU and atlas use only real tokens (`u[real]`, `tok`). Feature masks at particle sites are shared
  over slots. Resampling is slot-wise by pT rank. Where the clean jet has more particles than its donor, the "donor value" at those
  slots is the activation of a padded donor slot, inherited from P9's corruption B. Padded clean slots are masked as keys in every
  attention, so they do not reach the read-out.
- **Rows.** All samples use `rng.choice` over the file, sorted, with no first-k selection. Training pools use random rows of random
  files.
- **Build.** `prepset` → `part_core.build` uses η = η_jet + sign(η_jet)·Δη (the fixed version, not the G192 mirror). The φ + 0.06 edit
  shifts Δφ of real particles only, so the invariant mass is unchanged.
- **Variants.** "strict" resamples the skip terms and the particle-block attention outputs. It leaves live:
  - the token embeddings on the residual stream;
  - the pair-bias input for blocks whose attention output is replaced, where it has no effect;
  - both class-attention layers.

  So "strict" is still not the literal circuit definition. The report states the class-attention part.
- **Steering hook.** `Steer` adds `s_out · Δa · W_dec[f]` to the site output. This is correct for a transcoder (C sites) and for an SAE
  (R sites).
- **Figure.** The x-axis counts only the kept features' parameters (257 each). The live attention and skip parameters (~0.84 M, per the
  P13 issues list) are not counted, while the P9 literal circuits count all of theirs. The left-shift of the blue curve relative to
  P9 therefore depends on this choice of count.
- **No coding bug found** that changes a reported number.

## Procedure
- **Fresh samples** (test_20M, random rows, never first-k):
  - verify 1, seed 29:
    - ALL10: `<sample>_110.root`, 500 per class, N = 5,000;
    - DISC: W/Z `_108`, 1,500 + 1,500;
    - EVAL: W/Z `_109`, 5,000 + 5,000.
  - verify 2, seed 41: EVAL2: W/Z `_111`, 3,000 + 3,000.

  None of these files was used by P13 (105/106/107).
- **Rerun (verify 1).**
  - Unedited accuracy and all-12-site splice.
  - Fresh integrated-gradients (IG) ranking on DISC, with overlap against P13's ranking.
  - Circuits kept alone on EVAL: main, errlive and strict at K = 5 … 5,000, and P13's original ranking (main).
  - Random equal-size feature sets (per-site counts matched, live features): 40 sets at K = 20, 50, 200, 1,000, 2,000 on EVAL[::5]
    (2,000 jets).
  - 10-class read with uniform donors.
- **Added control A (verify 1): single-feature dose.**
  - The top m_A feature (again C0#831, r = −0.759) is moved toward higher m_A by 0.16, 1, 2, 4 and 8 standard deviations on the 5,000
    EVAL W jets.
  - Read-outs: W→Z flow (share of true W called Zqq minus the unedited share) and Δ(ℓ_Z − ℓ_W).
- **Added control B (verify 1): joint steering per site.** Every live feature i of one class-level site (C0, R0, C1 or R1) is moved by
  β_i·Δlog m_A, where:
  - Δlog m_A is each jet's own φ + 0.06 value;
  - β_i is the natural same-class pT-matched pair slope of feature i on log m_A (DISC).

  Scales are ×1, ×3 and ×10, plus a version with only the BH-significant features. Control: β permuted among the site's live features
  (20 permutations).

  Calibration of the read-out: the slope of (ℓ_Z − ℓ_W) on log m_A over the same natural pairs, times the φ edit's Δlog m_A, gives the
  read-out change that the edit's m_A change predicts.
- **Added control C (verify 2): what the live paths carry.** Same corruption (opposite-class pT-matched donor, errors resampled), P13's
  original ranking. Kept sets:
  - all live R1 latents (the bottleneck to the read-out);
  - all live R0 + R1;
  - all live class-level features (C0, R0, C1, R1; 1,308), with every particle-site feature resampled;
  - top-50; its 49 class-level members only; top-50 plus all R1;
  - top-1,000; its 371 class-level members only.

  Each set is run as main and as strict.
- **Statistics.**
  - Bootstrap over jets (1,000) for 95 % intervals.
  - McNemar for paired call changes.
  - Permutation p against random sets (40, so the minimum p is 1/41 = 0.024) and against permuted β (20, minimum 1/21 = 0.048).

## Numbers
**Unedited (verify 1, ALL10, N = 5,000)**
- 10-class accuracy: 85.96 % [85.10, 86.88]; P13 had 86.16 %.
- All-12-site splice: drop 0.28 points [−0.02, +0.62], McNemar 39 vs 25, p = 0.10. P13: 0.03 points.

**Attribution (fresh DISC, N = 3,000)**
- Completeness 1.001; error nodes 0.52 % of |score| (P13: 0.5 %); top-50 share 68.5 % (P13: 68.7 %).
- 49 of the top 50 features are class-level.
- Overlap of the top-K sets with P13: 0.90 (K = 10), 0.96 (50), 0.93 (200), 0.86 (1,000).
- Score correlation over all features: Pearson 0.998, Spearman 0.69 (the Spearman value is driven by near-zero scores).

**Circuits kept alone (fresh EVAL, N = 10,000)**

Unedited W/Z 83.47 % [82.77, 84.19]; donor 16.83 %. Exactness: all kept vs clean 4.8e-6, none kept vs donor 2.9e-6.

| K | main (fresh rank) | main (P13 rank) | errlive | strict | P13 main / strict |
|---|---|---|---|---|---|
| 20 | 34.1 % | 35.9 % | 33.1 % | 23.9 % | 34.6 / 24.4 |
| 50 | 51.4 % [50.4, 52.4] | 50.8 % [49.8, 51.8] | 51.7 % | 34.0 % | 50.0 / 32.0 |
| 200 | 65.7 % | 66.2 % | 65.4 % | 38.9 % | 66.3 / 38.0 |
| 1,000 | 73.8 % [73.0, 74.7] | 74.5 % | 74.8 % | 41.7 % | 74.1 / 40.4 |
| 2,000 | 78.3 % [77.5, 79.2] | 78.8 % | 78.7 % | 42.2 % | 79.0 / 40.3 |
| 5,000 | 80.9 % | 80.9 % | 82.9 % | 43.4 % [42.4, 44.4] | 81.4 / 41.7 |

- All live features kept (main): 81.1 % [80.3, 81.9]; P13: 81.6 %.
- Strict with every feature and error kept: 43.1 % [42.1, 44.0]; P13: 41.9 %.
- H1 (P13): a ≤ 50-feature circuit reaches ≥ 75 %. Fresh K = 50 gives 51.4 % [50.4, 52.4]; the interval excludes 75 %.
- **Random sets** (2,000 jets, 40 sets each):

  | K | circuit | random mean | random max | p (permutation) |
  |---|---|---|---|---|
  | 20 | 34.5 % | 18.5 % | 19.0 % | 0.024 |
  | 50 | 51.3 % | 18.5 % | 19.3 % | 0.024 |
  | 200 | 65.2 % | 18.7 % | 21.4 % | 0.024 |
  | 1,000 | 73.7 % | 19.6 % | 24.9 % | 0.024 |
  | 2,000 | 78.8 % | 21.6 % | 27.9 % | 0.024 |

- **10-class read**, uniform donors, N = 5,000: K = 50 11.8 % [11.0, 12.7]; K = 1,000 50.5 %; K = 5,000 79.2 %. P13: 12.0 / 51.7 / 78.8 %.

**Control C: what the live paths carry** (EVAL2, N = 6,000; unedited 83.3 %, donor 17.0 %)

| kept set (n features) | main | strict |
|---|---|---|
| all live R1 (142) | 17.0 % [16.1, 18.1] | 17.2 % |
| all live R0 + R1 (644) | 28.3 % | 22.2 % |
| all live class-level (1,308), particle features resampled | 59.2 % [58.0, 60.4] | 36.4 % |
| top-50 (50) | 50.8 % [49.5, 52.2] | 32.4 % |
| top-50, class-level members only (49) | 50.1 % | 32.4 % |
| top-50 plus all R1 (177) | 55.3 % | 34.2 % |
| top-1,000 (1,000) | 74.9 % [73.8, 76.0] | 40.7 % |
| top-1,000, class-level members only (371) | 60.0 % [58.8, 61.3] | 36.5 % |

Paired comparisons:
- top-1,000 minus its class-level members: +14.8 points [13.8, 16.0], McNemar 1,099 vs 209, p < 1e-140.
- all class-level minus top-1,000: −15.6 points [−16.8, −14.5].
- all R1 minus top-50: −33.8 points [−35.3, −32.5].

**m_A steering (EVAL W jets, N = 5,000; fresh DISC calibration)**
- Scan: 1,251 class-level features, 820 BH-significant. Top feature C0#831, r = −0.759; β = −7.41 per unit log m_A; s.d. 2.17.
- φ + 0.06 edit:
  - Δlog m_A = 0.093;
  - W→Z 10.3 → 31.0 %, flow +20.8 % [19.5, 22.0];
  - Δ(ℓ_Z − ℓ_W) = +1.077 [1.041, 1.112].
- Natural same-class pairs: slope of (ℓ_Z − ℓ_W) on log m_A = 1.64 [1.08, 2.15] (pair correlation 0.11). The predicted read-out change
  for the edit's Δlog m_A is 0.152, which is 14 % of the edit's 1.077.
- **Control A, single-feature dose** (toward higher m_A):

  | step (s.d.) | flow | Δ(ℓ_Z − ℓ_W) | McNemar on W calls (lost / gained), p |
  |---|---|---|---|
  | 0.16 | −0.02 % [−0.10, +0.06] | +0.0006 | — |
  | 1 | −0.02 % | −0.004 | 37 / 6, p 1.6e-6 |
  | 2 | −0.40 % [−0.68, −0.12] | −0.036 | — |
  | 4 | −2.2 % [−2.6, −1.8] | −0.186 | — |
  | 8 | −8.0 % [−8.8, −7.3] | −0.880 [−0.893, −0.868] | 101 / 315, p 1e-26 |

  The sign is opposite to the φ edit (toward W).
- **Control B, joint steering per site** (×1 = each jet's own Δlog m_A):

  | site (live, BH-significant) | Δ(ℓ_Z − ℓ_W) ×1 | flow ×1 | flow ×10 | permuted-β flow mean [95 %] | p |
  |---|---|---|---|---|---|
  | C0 (375, 255) | 0.017 | +0.50 % | +4.9 % | −0.27 % [−1.4, +0.6] | 0.38 |
  | R0 (457, 287) | 0.160 [0.158, 0.163] | +1.90 % [1.54, 2.28] | +23.6 % | +0.30 % [−0.4, +0.9] | 0.048 |
  | C1 (277, 184) | 0.010 | +0.12 % | +1.4 % | −0.02 % | 0.67 |
  | R1 (142, 94) | 0.149 [0.147, 0.151] | +1.28 % [0.96, 1.62] | +13.9 % | +0.08 % [−0.5, +0.7] | 0.048 |

  R0 and R1 at ×1 match the natural-pair prediction (0.152). Their ×1 flows are 9 % and 6 % of the φ flow. At ×10, R0 reaches 1.13 of
  the φ flow and 1.72 of its logit change.

## Checks and caveats
- **Hypotheses for the circuit size.**
  - H1 (≤ 50 features reach ≥ 75 %): excluded on fresh jets, as in P13.
  - H0 (feature circuits cap near the P9v 63 % ceiling): excluded for "main" from K = 200 (65.7 %).
  - For "strict" (≤ 43.4 %), H0 is not excluded.
- **Live paths (control C).**
  - The live particle attention and skip maps alone carry no W/Z information to the read-out: with all R1 latents kept and all
    upstream features resampled, accuracy is 17.0 %, equal to the donor.
  - All class-level features with every particle feature resampled give 59 %.
  - The particle-site members of the top-1,000 add 15 points.
  - The main-to-strict gap (≈ 30–35 points at K ≥ 200) shows how much of the main accuracy needs the particle attention outputs and
    skip maps computed from the real jet. The strict variant resamples both at once, so the split between attention and skip is not
    measured.
- **Steering test as designed in P13.** Its H1 asked a single-feature steer to give ≥ 0.6 of the φ flow, with the step calibrated on
  natural same-class pairs. Under the same calibration, the whole read-out changes by only 14 % of the φ edit's logit change, so even a
  perfect m_A feature steered by this rule could not meet the 0.6 criterion. The ablation part (C0#831 set to its mean under the edit,
  flow unchanged) is a valid single-feature test.
- **"No effect on the calls when steered."** This holds at the P13 step (0.16–0.31 s.d.). At 4–8 s.d. the feature changes calls, toward W.
- **"Spread over many features."** Supported in a limited sense: joint steering of all R0 or all R1 latents reproduces the natural-pair
  m_A read-out slope, and permuting β among the features removes it. The φ edit's response is about 7 times larger than this slope
  predicts.
- **Strict variant.** P13's strict maximum (41.9 %) was 43.4 % [42.4, 44.4] on fresh jets, so "below 42 %" is sample-dependent; "about
  42–43 %" holds on both.
- **Not rerun:** the atlas (the conditional-mean R² metric, by construction, favours dense coordinates) and the weight-permuted
  dictionaries.

## Corrected reading
Sparse dictionaries reproduce ParT_full's MLPs and class token (10-class accuracy 86.0 % unedited, splice drop 0.3 points
[−0.0, +0.6]).

Under W↔Z pT-matched resampling, with particle attention, transcoder skip maps and embeddings computed from the real jet:
- the top-K feature set's W/Z accuracy rises smoothly with K: 51 % at 50 features, 74 % at 1,000, 78 % at 2,000;
- random sets of the same size stay at 18–22 %;
- the fresh ranking and P13's ranking agree (top-50 overlap 0.96).

The small circuits are almost entirely class-level features. Within the main variant:
- all class-level features alone give 59 %;
- particle-level features add about 15 points at K = 1,000;
- the live attention and skip paths alone, with the R1 bottleneck fully open, give donor-level accuracy (17 %).

When particle attention and skip maps are also resampled, every variant stays at 42–43 %, even with all features and error terms kept.

The class-level feature most correlated with log m_A (C0#831, r = −0.76):
- leaves the calls and the φ + 0.06 response unchanged when steered by 0.16–0.31 s.d. or ablated to its mean;
- moves calls toward W (flow −8 % at 8 s.d.) when steered toward higher m_A.

Steering all features of the R0 or R1 SAE together by their natural m_A slopes reproduces the natural-pair m_A read-out slope (permuted
control null). That is 14 % of the φ edit's logit change and 6–9 % of its W→Z flow.

## Corrected conclusion
Learned sparse features, ranked by attribution, did not give a small circuit for ParT_full's W-versus-Z decision. Fifty features kept
alone give about 51 % W/Z accuracy against the model's 83 %, and 75 % needs 1,000–2,000 features. That holds only while the attention
layers and the transcoders' linear bypass terms are computed on the real jet; when they are also replaced, accuracy stays near 43 %.
The feature most correlated with the axis-moment mass m_A does not mediate the response to a mass-preserving φ shift. Moving all
features of one class-token layer together reproduces how the score varies with m_A in natural jets, but that accounts for only about a
seventh of the φ-shift response. Untested: trained (rather than ranked) feature or feature-plus-head circuits, and what carries the rest
of the φ response.

## Explainer entry (for site/explain.py)
- **Question.** Do learned sparse features of ParT_full's own activations give a small circuit for the W-vs-Z decision, and is the
  mass-like quantity m_A held in one such feature?
- **Background.** A skip-transcoder replaces an MLP by \(\hat y = \mathrm{TopK}_{32}(u W_{enc}^\top + b_{enc}) W_{dec} + b_{dec} +
  u W_{skip}^\top\); an SAE does the same for a residual vector without the skip term. With the error \(\varepsilon = y - \hat y\) added
  back, the network is unchanged. A feature "circuit" keeps a set of features on the real jet and replaces every other feature (and
  the errors) by its value on a donor jet of the opposite class with the same pT. Axis-moment mass:
  \(m_A = \sqrt{\textstyle\sum_i p_{T,i}\,\sum_i p_{T,i}\,\Delta R_i^2}\).
- **Method.**
  - Rank features by integrated gradients of \(D = \sum_j s_j(\ell_W - \ell_Z)\) over feature masks.
  - Keep the top K, measure W/Z accuracy on fresh jets, and compare with random sets of the same size.
  - Decompose the kept set into class-level and particle-level features, and vary which non-feature paths stay live.
  - Steer the top m_A feature by 0.16–8 s.d., and all features of one site by \(\beta_i\,\Delta\log m_A\) with \(\beta_i\) from
    natural same-class pairs.
- **Result** (figure `results/part/p13_feature_circuit.png`; fresh numbers in `p13_verify.json`, `p13_verify2.json`).
  - 50 features: 51 %; 1,000: 74 %; 2,000: 78 %; random sets: 18–22 %.
  - With attention and skip maps also replaced: ≤ 43 %.
  - Joint R0 steering: \(\Delta(\ell_Z-\ell_W) = 0.160\) versus the natural-slope prediction 0.152 and the φ edit's 1.077.
- **Checks.**
  - Exactness: \(< 5\times10^{-6}\) logit.
  - Fresh files, seed, and ranking.
  - Random-set and permuted-β controls.
  - A decomposition showing that the live attention/skip paths alone give donor-level accuracy.
- **Why it matters.** Learned features do not shorten the W/Z circuit under the literal definition. The m_A coupling of the φ response
  is not held by one feature, and the m_A calibration explains only a seventh of that response.
