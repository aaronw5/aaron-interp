# P11v — verification of P11 (ParT seed replication, pre-registered S1–S4), 2026-10-20

Status: **done** (two verifier passes).
- Pass 1: job `anrunw-ai-c-p11-verify` (GPU, 28 min, exit 0), script `interp/p11_verify.py` (calls the unchanged
  `interp.p11_seed_tests.prepare` / `analyse` on a fresh W/Z sample, plus an epoch-trajectory control); results
  `results/part/p11_verify_full.json`, `results/part/p11_verify_kin.json`; log `logs/c-p11-verify.log`.
- Pass 2 (added control): job `anrunw-ai-c-p11-verify2` (GPU, 20 min, exit 0), script `interp/p11_verify_untr.py`: the same
  unchanged analysis on untrained ParT_full / ParT_kin from three further initialisation seeds (torch seeds 1, 2, 3) on the
  pass-1 fresh sample; results `results/part/p11_verify_untrained_full.json`, `results/part/p11_verify_untrained_kin.json`;
  log `logs/c-p11-verify2.log`.

## Conclusion
Six Particle Transformers retrained from scratch all keep the released network's main behaviour on two separate test
samples: rotating every particle about the jet axis, which leaves the jet mass unchanged, turns 14–20 % more true W jets
into Z, while untrained networks of the same design barely respond. Inside the network the retrained models differ from
the released ones: the released checkpoints send almost all of the W/Z output unit's mass response through the first
class-attention block (the layer where the summary token first reads the particles), while the retrained ones split it,
at levels that untrained networks already reach; sharp mass-switching attention heads like the released b2h8 (a head whose
attention changes abruptly near the W/Z mass) do appear in some retrained models, at different heads and masses. The
retrained models saw 50 times fewer distinct jets and less training than the released ones, so seed and training scale
are not yet separated.

## Code review (interp/p11_seed_tests.py, p11_weight_check.py, p10_part_seeds_train.sh)
- Layer / head indices: S3 patches `cls_blocks[0]` (class block 1) and `cls_blocks[1]` per head on the 16-dim head
  contexts before `out_proj`; the re-implemented class stage matches the network's logits to 2.9e-6 (log
  `reimpl max |dlogit|`), and the particle-head record/patch reconstructs the block output to 1.0e-5. No index error found.
- Padded particles: the class-attention softmax masks padded keys; the S4 particle-head statistic averages only over real
  query particles (`qm`) and zeroes ΔR at padded slots. No padding leak found.
- Sampling: the W/Z sample is random rows (`default_rng(11)`), the S3 subset random (`default_rng(12)`); the P1 sample is
  rows 3,000–4,000 of each single-class file (the project's P1 convention, not a first-k split of class-ordered rows).
- Build: `part_core.build` uses η = jet_eta + sign(jet_eta)·Δη (corrected form).
- Edits: φ + 0.06 changes log m by 1.6e-8 on average (max 1e-4); the matched dilation reproduces the per-jet axis-moment
  change to 1e-7. The rot null (whole jet incl. axis rotated) gives R 0.000 in every model. No edit changes more than stated.
- Weight check: correlations of flattened tensors; adequate for "distinct initialisations".
- Wrapper: the `main() {...}; main "$@"; exit $?` form is the standard fix for bash reading a script incrementally.
- No bug found that changes a number.

## Procedure of the rerun
- Models: released ParT_full / ParT_kin; P10 seeds s0, s1, s2 (best epoch); untrained (weaver init, torch seed 0).
- Fresh sample: 10,000 random rows each of `WToQQ_101.root` and `ZToQQ_101.root` (the second test file of each class;
  P11 used `_100`), numpy `default_rng(21)`; N = 20,000 (no jet dropped). S3 subset: 2,000 W + 2,000 Z random jets of this
  sample. The ridge read-out is fitted on the P1 sample as in P11 (disjoint from the new jets).
- Same edits, statistics and settings as P11 (1,000 jet bootstraps; S4 with 200 bootstraps and 4,000 permutations; BH over
  80 heads; Holm within class / particle heads; 30 random 4+4 class-head sets, 20 random particle-head sets).
- Added control (epoch trajectory): seed 0 (full and kin) at epochs 0, 3, 9, 19, 29, 49 and best, same sample; S3 limited
  to class heads, block sets, two particle heads, 10 random class sets; S4 with 50 bootstraps and 500 permutations.
  Each P10 epoch = 1.024M training samples (0.5 pass over the 2M jets); the released model saw 512M samples from 100M jets.
  - H0 (training amount at fixed data explains the gap): the class-block-1 share rises with epochs toward the released
    0.97–0.98. H1: it stays near its final value.

## Numbers (fresh rows vs P11)
Unedited accuracy on the P1 sample (same as P11): released full 85.93 %, kin 74.47 %; seeds full 83.49–83.71 %, kin
72.04–72.57 %.

**Calls and R** (10,000 true W; Δ in points, 95 % bootstrap CI; exact McNemar p).

| model | W→Z base → φ (Δ) fresh | P11 Δ | R fresh [CI] | P11 R | McNemar p (fresh) |
|---|---|---|---|---|---|
| released full | 10.9 → 30.4 (+19.6 [18.8, 20.3]) | +19.3 | 0.785 [0.776, 0.794] | 0.787 | < 1e-300 |
| full s0 | 12.2 → 32.0 (+19.8 [18.9, 20.6]) | +19.1 | 0.802 [0.792, 0.811] | 0.805 | < 1e-300 |
| full s1 | 11.6 → 26.8 (+15.2 [14.5, 16.0]) | +14.9 | 0.646 [0.634, 0.659] | 0.650 | 9e-317 |
| full s2 | 11.5 → 30.5 (+19.1 [18.2, 19.9]) | +17.8 | 0.785 [0.774, 0.796] | 0.786 | < 1e-300 |
| untrained full | 0.1 → 0.0 (all jets called Hcc) | 0.0 | 0.374 (logits only) | 0.379 | 0.25 |
| released kin | 16.2 → 34.6 (+18.4 [17.5, 19.4]) | +17.8 | 0.725 [0.716, 0.735] | 0.722 | < 1e-300 |
| kin s0 | 17.0 → 32.5 (+15.5 [14.6, 16.4]) | +15.4 | 0.585 [0.573, 0.597] | 0.575 | 2e-278 |
| kin s1 | 18.3 → 33.3 (+15.0 [14.1, 15.8]) | +14.7 | 0.546 [0.532, 0.560] | 0.542 | 1e-262 |
| kin s2 | 17.6 → 32.1 (+14.5 [13.7, 15.3]) | +13.6 | 0.535 [0.522, 0.547] | 0.526 | 6e-265 |
| untrained kin | 79.4 → 83.3 (+4.0 [3.4, 4.5]) | +4.3 | −0.234 [−0.303, −0.171] | −0.234 | 2e-42 |

Rot null: R within ±0.002 and W→Z change 0.0 in every model.

**S1 / S2** (fresh; P11 in brackets): S1 ratio full rel 1.001 (1.004), s0 0.982 (0.980), s1 0.631 (0.632), s2 0.879
(0.877), untrained 0.930 (0.930); kin rel 1.507 (1.501), s0 0.749 (0.743), s1 0.625 (0.637), s2 0.512 (0.509), untrained
−0.001 (0.019). S1 window passes in the same 3 of 6 seeds. S2 pre-registered share: full 0.98–1.06, kin 1.00 in every
model including untrained (full untrained ΔR share 0.970). k1.1 pair share (read-out): released full 0.011, seeds
0.108–0.172; released kin 0.101, seeds 0.124–0.200 (all within 0.003 of P11).

**S3** (removed share of u*'s response to angles × 1.1, per-jet projection, N 4,000 fresh jets).

| model | class block 1 fresh [CI] | P11 | class block 2 | particle block 1 | cb1, W−Z logit form | largest class head | random 4+4 mean / max |
|---|---|---|---|---|---|---|---|
| released full | 0.982 [0.981, 0.984] | 0.981 | 0.075 | 0.090 | 0.983 | c1h3 0.46 | 0.50 / 0.88 |
| full s0 / s1 / s2 | 0.798 / 0.836 / 0.730 | 0.790 / 0.825 / 0.729 | 0.27 / 0.16 / 0.32 | 0.33 / 0.32 / 0.34 | 0.79 / 0.73 / 0.63 | 0.20 / 0.23 / 0.19 | 0.50–0.54 / 0.69–0.72 |
| untrained full | 0.535 (u* response 0.001 sd: undefined) | 0.519 | 0.48 | 0.98 | – | – | – |
| released kin | 0.966 [0.962, 0.970] | 0.968 | 0.057 | 0.114 | 0.922 | c1h7 0.46 | 0.53 / 0.74 |
| kin s0 / s1 / s2 | 0.632 / 0.674 / 0.627 | 0.643 / 0.675 / 0.632 | 0.36 / 0.35 / 0.35 | 0.23 / 0.26 / 0.26 | 0.65 / 0.50 / 0.59 | 0.16 / 0.21 / 0.15 | 0.49–0.54 / 0.68–0.72 |
| untrained kin | 0.868 [0.858, 0.879] | 0.867 | 0.126 | 0.994 | (logit degenerate) | c1h5 0.28 | 0.50 / 0.83 |

Untrained kin: u* responds 0.100 sd [0.097, 0.103] to angles × 1.1 (trained models 0.05–0.34 sd), so its share is defined.

**S4** (fresh rows). Heads passing the pre-registered form: full rel 7, s0 8, s1 6, s2 8, untrained 1; kin rel 9, s0 5,
s1 4, s2 7, untrained 2. Strong class-token-share switches (|amplitude| ≥ 0.2):

| model | head: m50 / w [GeV], share low-mass → high-mass (fresh) | P11 |
|---|---|---|
| released full | c2h8 (b2h8): 87.0 / 4.3, 0.87 → 0.08 (amplitude 0.79) | 87.0 / 4.8, 0.87 → 0.08 |
| full s1 | c2h2: 104.5 / 3.5, 0.40 → 0.10 | 105.0 / 3.1 |
| full s2 | c2h4: 68.0 / 4.8, 0.59 → 0.02 | 68.0 / 4.8 |
| kin s1 | c2h3: 84.0 / 3.9, 0.44 → 0.06; c2h1 62.5 / 9.2, 0.78 → 0.04 (amplitude 0.74) passes in only 78 % of bootstraps | c2h1 and c2h3 both strong |
| kin s2 | c2h3: 85.0 / 4.8, 0.85 → 0.11 (amplitude 0.74) | 85.0 / 5.4, 0.86 → 0.10 (0.76) |
| full s0, kin s0, released kin | none | none |

Particle-head strong switches of P11 that do not pass on fresh rows: full s1 p7h5, full s2 p8h3.

**Epoch trajectory (seed 0; class-block-1 share, R, W→Z Δ under φ, P1-sample accuracy).**

| epoch (samples seen) | full: acc / Δ / R / cb1 [CI] / cb2 | kin: acc / Δ / R / cb1 [CI] / cb2 |
|---|---|---|
| 0 (1M) | 76.0 % / +22.0 / 0.93 / 0.726 [0.716, 0.736] / 0.35 | 64.5 % / −7.1 / −0.29 / 0.656 [0.651, 0.661] / 0.36 |
| 3 (4M) | 80.3 % / +22.2 / 0.92 / 0.884 [0.879, 0.889] / 0.21 | 69.8 % / +8.0 / 0.37 / 0.752 [0.747, 0.757] / 0.31 |
| 9 (10M) | 81.7 % / +17.5 / 0.82 / 0.818 [0.814, 0.821] / 0.16 | 71.1 % / +12.8 / 0.49 / 0.618 [0.604, 0.629] / 0.38 |
| 19 (20M) | 83.7 % / +20.8 / 0.83 / 0.801 [0.796, 0.805] / 0.19 | 72.1 % / +13.1 / 0.57 / 0.634 [0.623, 0.643] / 0.34 |
| 29 (31M) | 83.4 % / +18.1 / 0.79 / 0.811 [0.807, 0.815] / 0.23 | 71.2 % / +10.7 / 0.58 / 0.663 [0.653, 0.672] / 0.33 |
| 49 (51M) | 83.8 % / +20.0 / 0.80 / 0.797 [0.790, 0.802] / 0.28 | 72.4 % / +15.7 / 0.60 / 0.623 [0.609, 0.635] / 0.37 |
| released (512M, 100M jets) | 85.9 % / +19.6 / 0.79 / 0.982 / 0.08 | 74.5 % / +18.4 / 0.73 / 0.966 / 0.06 |

H0 (rise toward 0.97–0.98 with epochs) is not supported: from epoch 9 to 49 the share stays at 0.80–0.82 (full) and
0.62–0.66 (kin), every CI excludes 0.9; the full maximum (0.88) is at epoch 3. ParT_full's R and W→Z response are
already at the final level after epoch 0 (R 0.93), while ParT_kin's grow over training (R −0.29 → 0.60).

## Pass 2: untrained-initialisation control (added)
Question: P11 and pass 1 each used one untrained network (torch seed 0). Are the untrained values of the class-block-1
share (S3), R and the W→Z call change stable across initialisations, and where do the retrained seeds sit relative to them?
- Procedure: `interp/p11_verify_untr.py` builds ParT_<feat> from weaver's initialisation with torch seeds 1, 2, 3 and runs
  the unchanged `p11_seed_tests.analyse` (all P11 settings: 1,000 bootstraps, S3 with 30 random 4+4 class-head sets and 20
  particle-head sets, S4 with 200 bootstraps and 4,000 permutations, BH over 80 heads).
- Sample: the pass-1 fresh sample (WToQQ_101 / ZToQQ_101, 10,000 random rows each, numpy default_rng(21); N 20,000; S3
  subset 2,000 W + 2,000 Z, default_rng(12)); read-out fitted on the P1 sample.
- H0: the retrained seeds' class-block-1 share and R differ from what an untrained network gives (seeds outside the
  untrained range). H1: they lie inside it.

| model | P1 acc | W→Z Δ under φ (points) [CI]; McNemar p | R [CI] | S1 ratio | u* response (sd) | class block 1 share [CI] | class block 2 | largest class head |
|---|---|---|---|---|---|---|---|---|
| full init0 (pass 1) | 10.1 % | 0.0 (all jets Hcc) | 0.374 [0.346, 0.402] | 0.930 | 0.001 (share undefined) | 0.535 | 0.48 | – |
| full init1 | 12.8 % | −0.1 [−0.2, −0.0]; 0.012 | 0.422 [0.376, 0.467] | 0.644 | +0.101 | 0.315 [0.299, 0.330] | 0.69 | c2h1 0.26 |
| full init2 | 11.9 % | 0.0 [−0.05, 0.05]; 1.0 | 0.530 [0.473, 0.587] | 0.517 | −0.147 | 0.577 [0.572, 0.582] | 0.43 | c1h8 0.42 |
| full init3 | 11.8 % | −0.1 [−0.1, 0.0]; 0.18 | 0.051 [0.012, 0.088] | 0.642 | +0.059 | 0.843 [0.828, 0.858] | 0.15 | c1h4 0.28 |
| kin init0 (pass 1) | 9.5 % | +4.0 [3.4, 4.5]; 2e-42 | −0.234 [−0.303, −0.171] | −0.001 | +0.100 | 0.868 [0.858, 0.879] | 0.13 | c1h5 0.28 |
| kin init1 | 11.1 % | +2.1 [1.6, 2.6]; 6e-15 | 0.254 [0.201, 0.306] | −0.027 | −0.110 | 0.665 [0.656, 0.674] | 0.28 | c1h5 0.24 |
| kin init2 | 10.6 % | 0.0 (no W called Z); 1.0 | 0.506 [0.478, 0.536] | 0.248 | +0.106 | 0.826 [0.817, 0.835] | 0.18 | c2h6 0.28 |
| kin init3 | 9.0 % | −0.3 [−0.6, 0.0]; 0.09 | 0.775 [0.727, 0.821] | 0.270 | +0.063 | 0.692 [0.682, 0.701] | 0.30 | c1h7 0.26 |
| reference: full seeds / released | 83.5–83.7 / 85.9 % | +15.2 to +19.8 / +19.6 | 0.65–0.80 / 0.785 | 0.63–0.98 / 1.00 | 0.05–0.32 / 0.31 | 0.73–0.84 / 0.982 | 0.16–0.32 / 0.08 | ≤ 0.23 / c1h3 0.46 |
| reference: kin seeds / released | 72.0–72.6 / 74.5 % | +14.5 to +15.5 / +18.4 | 0.53–0.59 / 0.725 | 0.51–0.75 / 1.51 | 0.33 / 0.22 | 0.63–0.67 / 0.966 | 0.35–0.36 / 0.06 | ≤ 0.21 / c1h7 0.46 |

S4 in the untrained inits: 0 or 1 pre-registered switch heads (full init3 c2h2, kin init1 c2h7, amplitudes 0.09 and
0.02); no strong (≥ 0.2) switch in any of the eight untrained networks.

Read-out of the control:
- Calls: untrained networks change W→Z by −0.3 to +4.0 points (8 inits), trained networks by +13.6 to +19.8 points. The
  call-level response separates trained from untrained in every pair.
- R: untrained values span −0.23 to 0.78 (every CI excludes both 0 and 1). R strictly between 0 and 1 is
  therefore what any network fed (Δη, Δφ, ΔR) about the stored axis gives; it excludes the invariant-mass hypothesis for
  the trained models, but its value (0.53–0.81 in seeds) is not by itself a trained signature. The φ shift alters the
  per-particle inputs, so every network with those inputs responds to it; what training adds is the size of the response
  at the class decision.
- Class-block-1 share: untrained 0.32–0.87 over the seven networks where u* responds (median 0.69). Full seeds 0.73–0.84
  and kin seeds 0.63–0.67 lie inside this range; H0 (seeds outside the untrained range) is not supported. The released
  0.966–0.982 exceeds every untrained value with a defined share (3 of 3 full, 4 of 4 kin; one-sided rank p = 1/4 (full) and 1/5 (kin), 1/20 = 0.05 jointly).
- Largest single class head: untrained 0.24–0.42, seeds ≤ 0.23, released 0.46. One untrained full network (init2, c1h8
  0.42) comes close to the released single-head share.
- S1: untrained full 0.52–0.93 (one of four inside the 0.7–1.4 window), untrained kin −0.03 to 0.27. S1 does not separate
  trained from untrained ParT_full (confirms P11).

## Assessment of the reading
- Call level (C3s): reproduced on fresh rows to within 1.3 points and 0.01 in R. H_inv (R = 0, no call change beyond the
  rot null) and H_axis (R = 1) are excluded in every trained model, as stated.
- "The untrained networks show no such call response": partly incorrect. Untrained full calls every jet Hcc, so its call
  test is degenerate (its logits give R 0.37). Untrained kin changes W→Z by +4.0 [3.4, 4.5] points with R −0.23: a small
  response of opposite sign in R, not none.
- S1 (3/6) and S2 (structural, 6/6): reproduced. Additional point: the full untrained network also has ΔR share 0.97, so
  the ΔR-vs-Δφ split in full carries no training information either.
- S3: reproduced (seed shares within 0.011 of P11). Not stated in the reading: the untrained kin network gives 0.868
  [0.858, 0.879], between the seeds (0.63–0.67) and the released kin (0.97). A first-block share near 0.9 is therefore
  close to what the architecture gives without training; the retrained seeds lie below the untrained value. "C2b holds
  only for the released checkpoints" is accurate as a measurement, but C2b's > 0.9 is not a learned signature by itself.
  In the kin seeds, class block 1 does not exceed random 4+4 head sets (p 0.097).
- S4 "no seed has a head as dominant as b2h8": not supported. Kin s2's c2h3 switches 0.85 → 0.11 at 85 GeV (w 4.8 GeV),
  amplitude 0.74 vs b2h8's 0.79, on both samples. The statement holds for the S3 single-head dominance (largest single
  class head ≤ 0.23 in seeds vs 0.46 in the released models), not for the switch scan. The count "strong class-share switch
  in 3 of 6 seeds" is a miscount of P11's own list: full s1 (c2h2), full s2 (c2h4), kin s1 (c2h1, c2h3) and kin s2 (c2h3)
  are 4 of 6 seeds, on both samples (on fresh rows kin s1 keeps c2h3; its c2h1 fails the bootstrap stability rule, 78 % <
  95 %). Two particle-head switches (full s1 p7h5, full s2 p8h3) do not reproduce.
- Data scale vs seed: the epoch trajectory shows that more training steps on the 2M jets (1M → 51M samples) do not move
  the first-block share toward the released value. If the gap is a data effect, it is the number of distinct jets (2M vs
  100M) or the authors' schedule, not training length on the 2M subset. Seed vs data scale is still confounded.
- Pass 2 (untrained inits): the seeds' class-block-1 shares (0.63–0.84) and R (0.53–0.81) lie inside the range of
  untrained networks (0.32–0.87; R −0.23 to 0.78). "The internal wiring is not reproduced" is accurate as a statement
  that the released concentration (0.97–0.98, above all untrained values) is absent in the seeds; it should not be read
  as the seeds having learned a different route, since their block-level share is indistinguishable from an untrained one.
  The call-level response is the statement that separates trained from untrained (≥ +13.6 vs ≤ +4.0 points).
- The P11 conclusion's sentence "no retrained network has a single head as dominant as the released head b2h8" mixes two
  statistics. b2h8 (c2h8) is the S4 switch head; its own S3 share is 0.055 (both samples). In S4, kin s2 c2h3 matches it.
  The S3 single-head statement holds for c1h3 (full, 0.45–0.46) and c1h7 (kin, 0.46): seeds ≤ 0.23.
- Training amount is a second confound besides data scale: P10 seeds ran 51.2M training samples (1/10 of the authors'
  512M samples per run). P11 names only the 50× data-scale difference.

## Corrected reading
Call level: in all six retrained seeds, on two disjoint samples, the mass-preserving φ shift moves true W to Z by +13.6 to
+19.8 points; eight untrained networks move −0.3 to +4.0 points. R lies between the invariant-mass prediction (0) and the
axis-moment prediction (1) and both are excluded; full seeds match the released R (0.65–0.80 vs 0.79), kin seeds are
below it (0.53–0.59 vs 0.72); untrained networks also give R strictly between 0 and 1 (−0.23 to 0.78), so the value of R
is not a training signature. S1 holds in 3/6 and does not separate trained from untrained ParT_full; S2 is structural.
S3 (class block 1 > 0.9) fails in 6/6 (0.63–0.84); untrained networks give 0.32–0.87, so the seeds lie in the untrained
range and the released 0.97–0.98 (above every untrained value) is a property of the released checkpoints. The released
single first-block head carrying 0.46 (c1h3 full, c1h7 kin) is not reproduced (seeds ≤ 0.23). S4 holds as pre-registered
in every trained model; a strong class-share switch exists in 4/6 seeds (P11 counted 3) at different heads and masses,
and one seed (kin s2 c2h3, 85 GeV, 0.85 → 0.11) has a switch as large as b2h8. Continuing training on 2M jets leaves the
first-block share flat (0.80 full, 0.62–0.66 kin from epoch 9 on); seed, distinct-data scale (2M vs 100M jets) and total
training (51M vs 512M samples) remain confounded.

## Issues
- The epoch-trajectory S4 used 500 permutations and 50 bootstraps (p floor 0.002; BH over 80 heads can still pass); S3
  for epochs used 10 random class sets (p floor 0.091). Main-model rows use P11's full settings.
- Holm across the McNemar tests of the job is not recomputed by the wrapper; the unadjusted p values above are all below
  1e-40 for trained models, so the adjustment does not change any call.
- No new figure; P11's figure `results/part/p11_seed_tests.png` matches the json values (S1, R, S3 class-block-1
  panels). It shows the untrained kin bar at 0.87 in the S3 panel without comment in the reading; it shows no class
  block 2 and no S4 panel. The R panel's y-axis starts at 0, so the untrained kin R (−0.23) is not drawn and the untrained
  full R (0.38, logits only) is omitted; the footer gives N = 20,000 for R, which uses the 10,000 true-W jets.
- Pass 2 used four untrained initialisations per feature set (three with a defined S3 share for full); the untrained
  range is from 7 networks and its edges are uncertain.

## Explainer entry (for site/explain.py)
- **Question:** Do P11's seed-replication results hold on jets the analysis never saw, does training longer on the same
  2M jets move a retrained ParT's internal route toward the released one, and where do untrained networks sit on the same
  statistics?
- **Background:** P11 found that six retrained ParTs keep the released network's response to a mass-preserving rotation
  $\phi\to\phi+0.06$ (C3s), but route less of it through class-attention block 1 (share 0.63–0.83 vs 0.97–0.98).
- **Method:** Rerun the unchanged P11 code on 20,000 W/Z jets from the second test file (`_101`), on seed 0's
  checkpoints after epochs 0–49, and on untrained networks from four initialisation seeds per feature set. Statistics: $R=\sum_j\Delta g_j^{\phi}\Delta g_j^{k}/\sum_j(\Delta g_j^{k})^2$ with
  $g=\ell_W-\ell_Z$; removed share $1-\sum_j dP_j\,dE_j/\sum_j dE_j^2$; switch fit $y=a+c/(1+e^{(m-m_{50})/w})$.
- **Result:** All numbers reproduce within 1.3 points (calls), 0.01 (R) and 0.011 (block shares). The first-block share
  stays at 0.80 (full) and 0.62–0.66 (kin) from epoch 9 to 49. Untrained networks give first-block shares 0.32–0.87 and
  $R$ from −0.23 to 0.78, but change W→Z calls by at most 4 points. One retrained kin seed has an 85 GeV attention switch
  of b2h8's size.
- **Checks:** re-implemented class stage matches logits to 3e-6; the whole-jet rotation null gives R 0.000; released
  models reproduce 85.93 % / 74.47 % and b2h8 (87.0 GeV, 0.87 → 0.08).
- **Why it matters:** The axis-relative mass response is a robust property of trained ParTs. Head-level and block-level
  statements are properties of individual checkpoints; the seeds' block-1 shares and $R$ values lie inside the untrained
  range, so only the call-level response and the released checkpoints' 0.97–0.98 concentration exceed it.
