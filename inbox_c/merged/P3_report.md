# P3: ParT_full W/Z mass circuit at full statistics, corrected build (agent report, 2026-10-19)

Job `anrunw-ai-c-p3-run` (NVIDIA A10, 1,293 s), `python -m interp.p3_wz_circuit_part`. Smoke runs: `c-p3-smoke` (passed);
`c-p3-smoke2` failed with a CUDA error on node k8s-haosu-06, which reports "GPU is lost". That node also refused two
later admissions, so the full run went through without a second smoke pass. Figure re-rendered by `c-p3-plot` (job.sh, CPU).
Script `interp/p3_wz_circuit_part.py`; results `results/part/p3_wz_circuit.json` (smoke: `p3_wz_circuit_smoke.json`);
figure `results/part/p3_wz_circuit.png`; logs `logs/c-p3-run.log`, `logs/c-p3-smoke.log`.
Infrastructure note for the ledger: node k8s-haosu-06.sdsc.optiputer.net needs to go on part_job.sh's NotIn list. I did not edit
part_job.sh because it is outside this loop's write scope.

## Conclusion
In ParT_full, one class-attention head (b2h8) behaves as a switch. It puts about 86 % of its attention on the class token below
the W mass and about 8 % above the Z mass, with the midpoint at 87 GeV, and no other head comes close to this. However, the
switch carries little of the mass information that reaches the W/Z neuron n98. Replacing b2h8's output with the output from a
jet of different mass moves n98 by under 1 % of what a real mass change does. Most of the response goes through a different
head (b1h3) and through the changed particle embeddings themselves. The proposed three-part circuit {b2h8, b1h3, n98}, kept alone
with everything else in the class stage replaced, sorts W from Z at 53–54 % against 83 % for the full model. So steps 2 and 3
of the thesis hold as a description of b2h8's switch, but not as the route by which the mass reaches the decision. That route
is spread over many units. Open questions: how the class-stage MLP passes the head outputs into n98, and whether a circuit that
includes MLP units chosen by a better criterion would do better.

## Label mapping (G-series → here)
Class-attention head bXhY = `cls_blocks[X-1]`, head Y−1 (dimensions 16(Y−1)…16Y−1 of the in-projection); b2h8 = (1, 7).
Neuron nK = index K−1 of z, the class token after the final LayerNorm (logits = zW + b). n98 = z[97].
Particle-block "far-side" heads (G101) = heads 4 and 8 = indices (i, 3), (i, 7) in each of the 8 particle blocks. These are stored
in the json as `label_map`.

## Procedure
- **Samples.** All jets come from the JetClass validation split (`/j-jepa-vol/JetClass/Pythia/val_5M`), using random rows from
  `numpy.default_rng(31)`, never the first k rows.
  - S: 2,000 rows from each class's `<sample>_120.root`, 20,000 jets in total. Used for head roles, circuits and the all-class span.
  - T: 10,000 rows each from `WToQQ_121.root` and `ZToQQ_121.root`, 20,000 W/Z jets. Used for the threshold, the edits and the
    decomposition.
  - T[::5]: 4,000 jets. Used for the particle-block patch.
  - Build: `part_core.build` with the corrected η convention. Jet mass = invariant mass of the built jet.
- **Unedited accuracy.**
  - ParT_full: 85.93 % on the P1 sample (rows 3,000–4,000 of the test files), 86.26 % on S, W-vs-Z binary on S's W/Z jets 83.25 %.
  - ParT_kin: 74.46 % on the P1 sample, 74.29 % on S, W/Z binary 75.60 %.
- **Model access.** The particle blocks run exactly as in weaver. The two class-attention blocks are re-implemented exactly as
  weaver-core 0.4.17 does them, including the post-out_proj c_attn reshuffle.
  - Agreement with the model: max |Δlogit| 7.6e-6 on S and 4.8e-6 on T, with 0 calls changed.
  - Particle-block per-head outputs o_ih = (A_ih V_ih)W_o,h. Their sum plus b_o equals the attention output to within 3.4e-5.
- **Edits on T.**
  - Angles × 0.9 / × 1.1 (every particle's Δη and Δφ scaled): true Δm −9.08 / +9.11 GeV.
  - φ + 0.06 rad (P1): max |Δm| 0.01 GeV.
  - Masks are identical to the unedited jets in every edit.
- **(a) Threshold.**
  - Class-token share = attention weight of the class-token query on the class-token key, for each of the 16 class-attention heads.
  - Binned in 5 GeV mass bins, with 1,000 bootstrap resamples over jets.
  - Logistic fit lo + (hi − lo)/(1 + e^{(m − m50)/w}), per jet, over 50–140 GeV, with 300 bootstrap refits.
  - Span = max − min of the binned mean over the 50–130 GeV bins.
  - H0: span ≤ 20 pt; H1: span > 20 pt. Bootstrap p, Holm-corrected over 16 heads.
- **(b) Heads → n98.** Each class-attention head's 16-dimensional context vector was replaced in three ways:
  - (i) resample: taken from a jet of the same class whose mass differs by ≥ 10 GeV (one donor per jet; median |Δm| on W/Z jets
    in the json);
  - (ii) mean ablation: set to its mean over S;
  - (iii) control: shifted by a random direction with the same norm as the change in (i), 20 draws per head.
  - Metrics on the 4,000 W/Z jets of S: the slope of Δz97 (in units of z97's standard deviation) and of Δ(lW − lZ) against the
    donor-minus-own mass difference (per 10 GeV).
  - Calibration "cal" = that slope divided by the slope under the real angles × 1.1 edit on T: Δz97 −0.295 s.d. per 10 GeV
    [−0.302, −0.288]; Δ(lW − lZ) −0.951 per 10 GeV.
  - Holm over 16 heads; McNemar test on W/Z binary calls under mean ablation.
  - Rule for naming further heads, set before the run: Holm p < 0.05 and |slope| ≥ 0.10 of the real-edit slope, excluding b2h8.
- **G21b decomposition at full statistics.** 16,514 W/Z jets of T with base mass 70–110 GeV.
  - Δz97 under each edit = attention change (every head's attention weights from the edited run, values from the base run)
    + value change (base attention, edited values) + interaction.
  - The same for each head alone (only that head's attention taken from the edit).
- **Particle-block far-side patch** (G103 style). Under each edit, the outputs of heads 4 and 8 in all 8 particle blocks were
  taken from the unedited jet. Compared with 20 random sets of 16 of the 64 particle heads, and with all 64.
- **(c) Literal circuits on S.**
  - Ablated components take a donor jet's clean values. One donor per jet, the same donor for every component and every
    circuit (paired).
  - Two donor distributions: uniform over all 20,000 S jets (`uniform_S`), and uniform over the 4,000 W/Z jets of S (`WZ_only`).
  - Scope: particle blocks intact; the circuit is stated inside the class-attention stage and the read-out.
  - MLP unit = an output unit of a class block's post_fc_norm (512 per block).
  - Variants, with H_c = {b2h8, b1h3} (b1h3 is the only head the naming rule selected):
    - V_a: H_c + z97 only; every MLP unit and every other z coordinate resampled.
    - V_b: V_a plus the 16 MLP units per block with the largest mean |∂z97/∂h · (h − h̄)| on W/Z jets.
    - V_c: H_c + all MLP units + z97 only.
    - V_d: H_c only, with MLP and read-out intact.
    - V_e: n98 alone, with the whole class stage intact.
  - Null: 100 random circuits of equal size per variant (random heads among the 16, random MLP units, a random other active z
    coordinate). Permutation p = (1 + #random ≥ circuit)/(1 + 100).
- **ParT_kin.** Steps (a) and (b) only. Index 97 is not ParT_kin's W/Z coordinate, so (lW − lZ) is the metric there.

## Numbers (ParT_full unless stated)

**(a) Threshold (T, 19,174 W/Z jets in 50–140 GeV).**
- b2h8 logistic fit: m50 = 87.10 GeV [86.79, 87.39], width 4.25 GeV [4.00, 4.51], share 0.864 → 0.081 (hi [0.853, 0.876],
  lo [0.073, 0.090]).
- By class: W only, m50 90.26 [89.79, 90.78]; Z only, m50 84.25 [83.53, 85.00]. The two classes switch at midpoints 6 GeV apart.
  The b2h8 switch therefore does not depend on mass alone.
- Span over 50–130 GeV: b2h8 82.5 pt [79.6, 86.9], Holm p = 0.016 (bootstrap floor). The next largest heads are b2h1 16.4,
  b2h5 10.5 and b2h3 9.1, each with Holm p = 1. b2h8 is the only head rejecting H0.
- All-class span on S (40–240 GeV, 10 GeV bins): b2h8 0.90; the others are ≤ 0.24 (b2h7 0.24, b2h4 0.16, b1h3 0.14). This
  reproduces G22.
- Above 140 GeV, b2h8's share rises again on W/Z jets (to 20–45 %, with wide bands).

**Edits, Δ share of b2h8 on T with 95 % CI; "largest other" = largest |Δ| among the other 15 heads.**

| edit | b2h8 Δ share | prediction of the observed share-vs-mass curve | largest other |
|---|---|---|---|
| angles × 0.9 | +0.201 [0.198, 0.205] | +0.178 | 0.028 |
| angles × 1.1 | −0.208 [−0.213, −0.204] | −0.158 | 0.046 |
| φ + 0.06 rad (invariant mass unchanged) | −0.171 [−0.174, −0.167] | 0.000 | 0.038 |

- Most of the other heads' Δ values are also non-zero after Holm correction (14 of 16 heads, including b2h8), but all are ≤ 4.6 pt.
- The φ shift moves b2h8 by 82 % of what angles × 1.1 does, with zero change in invariant mass. b2h8 is therefore axis-relative,
  as C3 found for the model as a whole.

**(b) Heads → n98 (S, 4,000 W/Z jets).** Resample slope as a fraction of the real-edit slope ("cal"):

| head(s) resampled | cal (Δz97) | cal (Δ(lW − lZ)) |
|---|---|---|
| all 16 | 0.150 | 0.099 |
| block 1 (8 heads) | 0.146 | 0.109 |
| block 2 (8 heads) | 0.015 | −0.002 |
| b1h3 | 0.191 [0.177, 0.207] | 0.150 [0.138, 0.163] |
| b2h8 | 0.006 [0.005, 0.008] | 0.004 [0.003, 0.005] |
| any other single head | ≤ 0.009 | – |

- b2h8: Holm p 0.016 for slope ≠ 0. Its share of the all-16 resample slope is 0.043 [0.036, 0.053]. For the thesis question
  "does b2h8 carry ≥ 50 %?", the bootstrap p for share ≥ 0.5 is 0.001.
- b1h3 alone (0.191) exceeds all 16 together (0.150), so head contributions do not add.
- Random-direction control: for every head, the real resample moves |Δz97| more than all 20 random directions of the same norm
  (p = 0.048, the floor for 20 draws). Example for b2h8: 0.026 vs a random maximum of 0.014.
- Mean ablation, W/Z binary calls changed, with McNemar p before and after Holm:
  - b1h3: 12.1 %, p 1.1e-10 (Holm 1.7e-9). Δz97 on W −0.243 [−0.251, −0.234] s.d.; on Z −0.175.
  - b1h2: 7.0 %, p 0.005 (Holm 0.07).
  - b1h6: 4.3 %, p 0.04 (Holm 0.53).
  - b2h8: 0.85 %, p 0.86 (Holm 1). Δz97 on W −0.020, on Z +0.015.
- Attention-weighted ΔR to the jet axis of the particles each head reads (W jets): b2h8 0.33, b1h3 0.19, b1h6 0.22,
  other block-1 heads 0.17–0.22.

**G21b decomposition (T, 16,514 jets with mass 70–110 GeV).** Δz97 in s.d. units.

| edit | total | attention | value | interaction | attention share |
|---|---|---|---|---|---|
| angles × 1.1 | −0.345 [−0.351, −0.339] | −0.118 [−0.120, −0.116] | −0.236 [−0.241, −0.232] | +0.009 | 0.34 [0.34, 0.35] |
| φ + 0.06 | −0.295 | −0.031 | −0.177 | −0.088 | 0.10 [0.10, 0.11] |

- Angles × 1.1, attention part by head: b1h3 −0.055 (0.16 of the total), b1h6 −0.035, b2h8 −0.019 (0.056 [0.055, 0.057]).
- φ + 0.06, attention part by head: b1h6 −0.015, b2h8 −0.015 (0.050).
- For Δ(lW − lZ) under angles × 1.1: total −1.10 = attention −0.39 + value −0.75; b2h8's attention part −0.065 (0.059).
- G21b reported attention −0.458 of −0.478, with b2h8 as the largest head. That came from the class-token formula fit at
  400 jets per class. ParT's own network at full statistics gives attention 34 % and value 68 %, with b2h8 third.

**Particle-block far-side heads (h4, h8 in all 8 blocks; T[::5], 4,000 jets).** Share of the edit's Δz97 removed by patching:

| edit | jets | far-side heads | random 16-head sets | all 64 heads | p |
|---|---|---|---|---|---|
| angles × 1.1 | W | 0.257 [0.251, 0.264] | mean 0.125, max 0.179 | 0.464 | 0.048 (floor) |
| angles × 1.1 | Z | 0.085 [0.066, 0.102] | mean 0.043, max 0.072 | 0.244 | 0.048 |
| φ + 0.06 | W | 0.065 | mean 0.079, max 0.135 | 0.335 | 0.76 |

**(c) Literal circuits (S, 20,000 jets).** Unedited: 10-class 86.26 % [85.79, 86.72], W/Z binary 83.25 %.

| circuit | 10-class acc | W/Z binary (uniform_S donors) | random: mean / max | p | W/Z binary (WZ_only donors) |
|---|---|---|---|---|---|
| V_a {b2h8, b1h3, n98}, MLP resampled | 10.27 % [9.88, 10.71] | 52.85 % | 49.67 / 50.30 | 0.010 | 50.88 % |
| V_b + 32 MLP units | 10.41 % | 54.05 % | 49.68 / 50.22 | 0.010 | 53.05 % |
| V_c + all MLP units | 10.49 % | 53.02 % | 49.67 / 50.48 | 0.010 | 53.87 % |
| V_d heads only (MLP, read-out intact) | 12.73 % | 52.58 % | 52.71 / 65.77 | 0.24 | 54.97 % (p 0.15) |
| b2h8 alone (MLP, read-out intact) | 10.13 % | 50.08 % | 51.37 / 62.52 | 0.48 | – |
| V_e n98 alone (class stage intact) | 12.03 % | 71.57 % | 50.77 / 55.30 (other active coordinates) | 0.010 | 76.52 % |

- Reading the table: the circuits sit about 3 points above equal-size random circuits (p 0.01), 30 points below the full model,
  and agree with the unedited 10-class call on about 10 % of jets.
- The 32 MLP units in V_b hold 59 % (block 1) and 37 % (block 2) of the total attribution.
- Single read-out coordinate with all others resampled: n98 71.6 %, rank 1 of 128. Next are n82 55.3 % and n125 52.0 %.
  Permutation p vs the other 127 coordinates: 0.008.
- Confusion matrices for every variant are in the json (`models.full.circuits.variants.*.*.confusion`). With all class-token
  coordinates but one resampled, calls follow the donor's class.

**ParT_kin.**
- Threshold: b2h8's class-token share rises with mass (0.219 → 0.015 in the fit's lo/hi terms, m50 91.9 GeV [91.3, 92.5]).
  Spans: b2h7 24.6 pt and b2h8 23.2 pt (both Holm p 0.016).
- Edits: b2h8 Δ share +0.063 (angles × 1.1) and +0.049 (φ + 0.06); b2h7 moves more (0.093).
- Δ(lW − lZ) resample, cal: all 16 heads 0.144; b1h7 0.094 [0.082, 0.106]; b1h2 0.046; b2h8 0.002.
  Mean ablation of b1h7 changes 14.1 % of W/Z calls (McNemar Holm p 3.2e-6).
- No head was named under the rule, because the z97 slope does not apply to ParT_kin and the rule is defined on it. b1h7 plays
  the role that b1h3 plays in ParT_full.
- Decomposition of Δ(lW − lZ) under angles × 1.1: attention 0.41, value 0.60. b2h8's part is 0.003.

## Reading
- b2h8's class-token share is a sharp, monotonic function of jet mass across W/Z jets: m50 87.1 GeV, 4 GeV wide. It is the only
  class-attention head with such a span. It follows real mass edits about as the observed curve predicts (+0.20 vs +0.18;
  −0.21 vs −0.16). It also follows the φ translation, which leaves the invariant mass unchanged (−0.17).
- Changing what b2h8 writes, by resample or by its mean, changes n98 and the W/Z call by < 1 % of a real mass edit and
  < 1 % of calls.
- The class-attention route into n98 is mainly block-1 head b1h3: 19 % of the real-edit slope for Δz97, 12 % of W/Z calls under
  mean ablation. Under the edit, 68 % of n98's response comes from changed values (particle embeddings), not from the
  re-weighting of attention.
- Particle-block far-side heads 4 and 8 carry about twice the random share of the real mass edit on W jets (26 % vs 13 %). They
  do not carry it for the φ translation.
- The literal circuit {b2h8, b1h3, n98} keeps W/Z separation statistically above equal-size random circuits, but close to 50 %.
  n98 alone, read from the intact class stage, keeps 71.6–76.5 %.

Competing predictions from the task, compared with the measurements:

| | thesis prediction | measured |
|---|---|---|
| m50 | 80–90 GeV | 87.1 |
| b2h8 span vs others | much larger | 82.5 pt vs ≤ 16.4 pt |
| b2h8 carries most of Δn98 and Δ(lW − lZ) | yes | 4 % of the all-head resample, 6 % of the edit's attention decomposition; ≥ 50 % rejected, p 0.001 |
| literal circuit well above random | yes | 3 points above random, 30 below the model |

The alternative's predictions (the n98 route is one of several, b2h8 < 50 %, the circuit near random-circuit accuracy) match the
measurements for steps 2–3. Its prediction that the threshold would move or be shared does not match: the threshold is held by
b2h8 alone.

## Issues / caveats
- The resample slope is set by the mass association between jets of the same class. The calibration against the real edit shows
  that resampling all 16 heads reproduces only 15 % of the edit's n98 slope. The per-head "cal" values are therefore lower bounds
  on what a head can carry, not shares of the edit response. The edit-based decomposition is the direct share measure.
- Head contributions are not additive (b1h3 alone 0.19 > all 16 together 0.15). This comes from post_attn_norm's LayerNorm over
  all heads and from the block-2 query depending on block 1.
- The strict circuits resample 127 of 128 class-token coordinates from the donor, so the 10-class accuracy (~10 %) mainly reflects
  that. V_e isolates the read-out part. The MLP-unit selection (top-16 by gradient × deviation) is one choice; other selections
  were not tried.
- The bootstrap p floor is 0.001 before Holm (0.016 after). The random-direction and particle-set p floors are 1/21 = 0.048.
- The json keys `share_*` (relative to the all-16 resample) are unstable when the denominator is small (ParT_kin z97). Use `cal_*`.

## Explainer entry (for site/explain.py)
- **Question.** Does ParT's class-attention head b2h8 act as a W/Z mass switch, and does it carry the mass into neuron n98?
- **Background.** ParT ends with two class-attention blocks. A learned class token attends over the particle embeddings, and its
  final value z gives the scores through \(\ell = zW + b\). Earlier fits (G21) suggested that head b2h8 turns its attention away
  from the class token as the mass passes 85 GeV, and that this lowers the W/Z coordinate \(z_{97}\) ("n98").
- **Method.**
  - Class-token share \(a_{0}\) of the head, binned against mass, fitted with
    \(s(m)=s_{lo}+\frac{s_{hi}-s_{lo}}{1+e^{(m-m_{50})/w}}\).
  - Causal edits: scale all particle angles by \(k\) (mass \(\times k\)), or shift \(\phi\) by 0.06 (mass unchanged).
  - Head patching: replace the head's context \(c_{h}=\sum_p a_{hp}v_{hp}\) with a same-class donor's, and measure
    \(\beta_h=\frac{\sum_j \Delta z_{97,j}\,\Delta m_j}{\sum_j \Delta m_j^2}\), compared with the real-edit slope.
  - Decomposition \(\Delta z_{97}=\Delta_{attn}+\Delta_{value}+\Delta_{int}\).
  - Literal circuit: only \(\{b2h8, b1h3, z_{97}\}\) kept, everything else in the class stage taken from a random donor jet.
- **Result (figure p3_wz_circuit.png).**
  - \(m_{50}=87.1\) GeV [86.8, 87.4]; b2h8 span 82 pt, others ≤ 16 pt.
  - b2h8's \(\beta\) is 0.6 % of the real-edit slope; b1h3's is 19 %.
  - The attention change is 34 % of \(\Delta z_{97}\).
  - Circuit W/Z accuracy 53–54 % (random 50 %, full model 83 %).
- **Checks.**
  - Re-implementation reproduces the logits to \(<10^{-5}\).
  - Random-direction patches, random head and unit sets, two donor distributions, Holm correction.
  - The unedited accuracy of 85.9 % on the P1 sample is reproduced.
- **Why it matters.** The switch exists, but the decision does not route through it. The W/Z mass reaches the score mainly
  through block-1 attention and the particle embeddings, which rewrites thesis steps 2–3 as a distributed read-out.
