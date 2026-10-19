# X8: functional nulls for the headline causal tests (shuffled-label and weight-permuted networks; resample ablation)

Node X8, round 5, stage 2. Claims A1, A3, A4, D4, A10; round-4 review weakness 12.
Script `interp/x8_nulls.py` (analysis plan frozen in the module header). Results `results/jedi/x8_jedi.json`,
`x8_jedi_abl.json`, `x8_dsets.json`, `x8_jedis.json`, `x8_summary.json` (per-jet arrays `.npz` on the volume only), figure
`results/jedi/x8_nulls.png`. Logs `logs/x8-{jedi,jabl,dsets,jedis,comb}.log`.

## Conclusion
The translation W→Z flow, the accuracy loss under WTA re-axing, and the radial response of the pooled Σ pT r² unit are
much larger in trained JEDI-linear than in 15 weight-permuted copies of it. The rank-window across-edge test does not
separate trained from permuted: copies with the trained zero-bit masks kept in place give across/inside ratios of 0.5–11
(trained 5.2), so this test measures the mask layout, not trained weights. Resample ablation ranks live units the same way
as mean ablation (Spearman 0.97). Replacing the context round g with another jet's g changes 27.8 % of calls (mean ablation:
14.6 %). Open question: permuted networks mostly give constant or near-random output, which makes them a weak null for the
"calls changed" statistics.

## Procedure
- Sample: `interp.jedi_core.sample(per_class=2000, seed=307)`, N = 10,000 test jets (2,000 per class; N_W = 2,000). Seed 307
  was not used by any earlier node. Read-out split: rng(307) permutation, fit half 5,000 / test half 5,000.
- Models.
  - JEDI-linear 3-feature n64 (exact fixed-point network). The script's own jit forward is checked equal to `Jedi.run` (max
    |Δlogit| = 0, before and after the permutation loop). Weight-permuted copies, no training, rng seeds 0–4:
    - variant a: non-zero kernel entries permuted among the non-zero positions, non-zero biases within the layer; all input
      quantizers (bit widths, zero-bit masks incl. the qs/qs1 rank windows) in place.
    - variant c: kernels as in variant a; slot-dependent quantizer tables of qs, l2, add, l3, qs1 permuted along the slot
      axis per channel (rng 200+s). This variant was added after the smoke run and before the full run.
    - variant b: all kernel entries permuted within the matrix and every quantizer's k/i/f tables permuted (rng 100+s).
  - dsets-d64 and jedis-d64 (float; jedis has a context round g = masked mean of the first per-particle MLP). For each
    family: trained seeds 0 and 1 (best.pt; test accuracy 80.6/80.8 % and 79.7/79.3 % on 260k jets); shuffled-label seeds 0
    and 1 (training labels permuted once with rng(seed); final epoch-30 checkpoint, no model selection on true labels; test
    accuracy 21.2/23.6 % and 20.2/30.6 %); 5 weight-permuted copies of trained seed 0 (every nn.Linear: weights within the
    matrix, bias within the layer). Training was the GPU job `train-n64-x8-b` (2026-10-17, 4 concurrent runs, RTX 2080 Ti,
    6.6 min, code d4692a5). It used the opt-in `--shuffle_labels` flag in `train/train_small.py` and the `jedis-d64` model in
    `train/models.py`.
- Edits (raw particles, padding untouched, pT order kept), each with a matched null edit:
  - T: translation 0.06 rad along +Δη (E6v). Null edit Tr: 0.06 rad in a random direction per jet.
  - W: re-axis about the WTA axis (E17; mean displacement 0.046 rad). Null edit Wr: translation by the same per-jet distance
    in a random direction.
  - K: (Δη, Δφ) → 1.1 (Δη, Δφ) at fixed pT (E23 calibration edit; mean Δlog m = 0.0953). Null edit Kr: rotation about the
    axis with equal per-particle displacement.
  - Angle swap of ranks 13|14 (across the E20 edge) vs 5|6 (inside). Also 13|14 vs 11|12 and 0|1 vs 3|4 (X6).
- Metrics:
  - H1: W→Z flow = mean over true W of [call_T = Z] − [call_0 = Z].
  - H2: share of calls changed under W. Secondary: accuracy drop under W.
  - H3: across/inside ratio of the mean |ΔS|/sd(S) over the live S units with a qs1 mask edge at k|k+1. Models without masks
    use the 10 live S units of largest cost.
  - H4: response under K, in s.d., of the S unit with the largest |Spearman ρ| with Σ pT r² on unedited jets (selection rule
    frozen).
  - Secondary: calibrated read-out ratio = (ΔR_T / pred_T) / (ΔR_K / Δlog m_K). R is a ridge read-out of log m at the last
    hidden layer; pred = ½ Δlog M2.
- Statistics:
  - 1,000 jet-level bootstrap resamples, using the same indices for every model (paired), with 95 % percentile intervals.
  - Null set per family: JEDI-linear 15 permuted copies; dsets/jedis 5 permuted + 2 shuffled-label models.
  - H0: trained ≤ null; H1: trained > null. p_IUT is the maximum over null models of the paired-bootstrap one-sided p
    (intersection-union test). The floor is 1/1001; the rank p floor is 1/16 (JEDI) or 1/8 (small models).
  - Holm correction across 4 headline tests × 3 families (trained seed 0 for small models).
  - McNemar (exact) for paired call changes, edit vs matched null edit.
- Ablation arm (trained JEDI-linear; dsets/jedis seed 0):
  - Every P, S, z unit (dsets: S, h3; jedis: g, S, h2) is set to its mean over jets. Units with ≥ 1 % calls changed are live.
  - Each live unit is resample-ablated: its value is taken from another jet, 20 random derangements (rng 3074).
  - Spearman correlation of mean vs resample cost, with a bootstrap CI.
  - Per unit, H0: resample cost ≤ 1 % (one-sided z), with BH correction.
  - Context round: all 64 g units to their mean vs to a random other jet's g (20 resamples).
- Compute: 5 cluster CPU jobs at 1 CPU / 2 GiB. Peak memory 0.55–1.6 GiB; wall times 12 min (jedi), 26 min (jabl),
  31 min (dsets), 115 min (jedis), 1 min (combine).

## Unedited networks on this sample
- JEDI-linear trained: accuracy 81.47 % (N = 10,000; bootstrap s.e. about 0.4 pp; expected range 80.6–81.3 %).
- Permuted copies: accuracy 19.5–23.1 %.
  - variant a: s4 gives a constant call.
  - variant c: s4 gives a constant call.
  - variant b: all 5 seeds give a constant call. Their S is constant, so the window ratio and the read-out are undefined (nan).
- Small models: trained 80.1–81.0 %; shuffled-label 16.2–22.2 %; permuted 14.1–23.7 % (dsets s3/s4 and jedis s1/s4 give a
  constant call). Torch forward checks: max |Δlogit| = 0.

## Results: headline tests (trained point [95 % CI] vs null points; p_IUT → Holm)

| test | JEDI-linear trained | JEDI-linear nulls (15) | dsets s0 / s1 | dsets nulls (7) | jedis s0 / s1 | jedis nulls (7) |
|---|---|---|---|---|---|---|
| H1 W→Z flow under T | 0.752 [0.730, 0.771] | −0.107 … 0.135 | 0.676 / 0.724 | −0.086 … 0.016 | 0.660 / 0.734 | 0 … 0.0005 |
| H2 calls changed under W | 0.298 [0.289, 0.307] | 0 … 0.627 | 0.398 / 0.387 | 0 … 0.363 | 0.393 / 0.365 | 0 … 0.138 |
| H3 A13/I5 ratio | 5.22 [4.99, 5.48] | a: 0.52–9.27; c: 2 × ∞, 3.2–11.1; b: undefined | 0.47 / 0.64 | 0.21–0.58 | 0.42 / 0.41 | 0.17–0.45 |
| H4 radial response [s.d.] | 0.200 [0.196, 0.205] (S50) | 0 … 0.129 | 0.202 / 0.133 | 0.021–0.159 | 0.181 / 0.209 | 0.070–0.200 |

Holm-adjusted p (trained seed 0; 12 tests):

| family | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| JEDI-linear | 0.012 | 1.0 | 1.0 | 0.012 |
| dsets | 0.012 | 0.012 | 1.0 | 0.012 |
| jedis | 0.012 | 0.012 | 1.0 | 1.0 |

The minimum attainable p is 0.012 = 12 × 1/1001. For JEDI-linear H3, p is computed over the defined (non-nan) nulls only.

Secondary statistics:

| statistic | JEDI-linear trained | JEDI-linear nulls | dsets s0 | dsets nulls | jedis s0 | jedis nulls |
|---|---|---|---|---|---|---|
| accuracy drop under W | 0.219 [0.210, 0.228] | −0.012 … 0.020 | 0.304 | −0.011 … 0.003 | 0.288 | −0.020 … 0.008 |
| calls changed under T | 0.562 | 0 … 0.645 | 0.623 | 0 … 0.51 | 0.672 | 0 … 0.333 |
| calibrated read-out ratio, T | 0.943 [0.924, 0.962] | −9.99 … 5.23 | 0.810 | −0.36 … 3.04 | 0.709 | −5.65 … 1.74 |
| calibrated read-out ratio, W | 0.919 [0.894, 0.946] | −11.8 … 2.16 | 0.587 | −14.7 … 3.57 | −0.246 | −4.99 … 3.0 |
| read-out R² (test half) | 0.939 | −0.0 … 0.43 | 0.953 | −9.3 … 0.55 | 0.895 | 0.13 … 0.53 |
| A0/I3 ratio | 0.25 [0.11, 0.43] | 0.08 … 25.9, 1 × ∞ | 2.56 | 1.14–1.83 | 2.30 | 1.07–1.82 |
| radial response under Kr [s.d.] | 0.0006 | −0.003 … 0.005 | 0.0001 | −0.001 … 0.002 | 0.0002 | −0.001 … 0.001 |

Matched null edits on trained JEDI-linear (McNemar, exact):
- T vs Tr, W→Z calls among true W: 409 vs 81 discordant, p = 1e-53. W→Z flow is 0.752 under T vs 0.588 under Tr (Tr also
  moves W to Z: a translation in any direction raises M2 about the fixed axis).
- W vs Wr, calls changed: 0.298 vs 0.337; discordant 328 vs 711, p = 4e-33. Under Wr, a random-direction translation of the
  same per-jet length changes more calls than the re-axis does.
- K vs Kr, calls changed: 0.244 vs 0.024; discordant 2303 vs 104, p < 1e-300.
- Swap 13|14 vs 5|6, calls changed: 1.23 % vs 0.97 %; p = 0.061.
- Swap 13|14 vs 11|12, calls changed: p = 1.4e-4.

Radial unit:
- The frozen rule selected S50 (ρ = 0.932), not S29. S29 has ρ = 0.849 and responds to K with 0.261 s.d.
- Per-unit A13 response on trained JEDI-linear: S52 0.075 vs 0.006 inside; S63 0.068 vs 0.013; S43 0.043 vs 0.011. S0 and S51
  have 0 across.

## Results: ablation arm

| model | live units | Spearman mean vs resample [95 % CI] | per stage | BH q < 0.05 (resample > 1 %) | context g: mean → resample (% calls) |
|---|---|---|---|---|---|
| JEDI-linear | 43 (P 11, S 19, z 13) | 0.965 [0.954, 0.973] | P 0.96, S 0.98, z 0.92 | 43/43 | 14.6 [13.9, 15.3] → 27.8 [27.3, 28.3]; McNemar p = 7e-174 |
| dsets s0 | 67 (S 54, h3 13) | 0.988 [0.983, 0.990] | S 0.99, h3 0.96 | 67/67 | n/a |
| jedis s0 | 99 (g 38, S 48, h2 13) | 0.977 [0.969, 0.980] | g 0.98, S 0.97, h2 0.74 | 99/99 | 31.2 [30.3, 32.2] → 46.6 [46.2, 47.1] |

JEDI-linear S units, mean / resample cost (% calls changed):

| S29 | S1 | S0 | S50 | S19 | S30 |
|---|---|---|---|---|---|
| 40.4 / 38.4 | 21.0 / 23.2 | 12.6 / 20.5 | 11.5 / 20.0 | 11.0 / 16.3 | 7.9 / 17.1 |

## Reading (scoped to these measurements)
- H1 (translation W→Z) is far above every null on all three architectures, with Holm p = 0.012 (the floor). For trained
  networks, the calibrated read-out ratio of the translation is 0.71–0.94. For null networks, the read-out of log m is poor
  (R² ≤ 0.55) and the ratios scatter widely in both signs (−10 to +5), so the nulls are not near 0 as predicted. The read-out
  ratio therefore separates trained from null by its closeness to 1, not by a one-sided test (p_IUT = 1).
- H2 as frozen (share of calls changed under the WTA re-axis) does not separate JEDI-linear from its permuted copies.
  Permuted copies with non-constant output change 21–63 % of their calls under any edit, and those calls are near chance. The
  accuracy drop under the same edit is 0.219 for trained JEDI-linear vs ≤ 0.020 for every null, and 0.29–0.30 vs ≤ 0.008
  for the small models.
- H3 (rank-window across/inside ratio) on JEDI-linear lies inside the range of the permuted copies whose zero-bit masks are
  kept in place:
  - variant a gives 0.5–9.3; variant c gives 3.2–11.1 plus two ∞ (inside change = 0).
  - The selection rule (units whose qs1 mask changes at 13|14) guarantees by construction that a unit has a window edge at
    that rank, whatever the weights.
  - Variant b, the mask-permuted copy needed for the "masks vs weights" split, gives constant-output networks, so that
    comparison is not available from this run.
  - The measured window asymmetry is consistent with the masks alone. It does not by itself show trained function.
  - Float models (no masks) give ratios < 1 for trained and nulls alike at 13|14. At 0|1 vs 3|4, trained 2.3–2.6 vs nulls
    1.1–1.8, p_IUT = 0.001.
- H4 (radial response) on JEDI-linear: S50 0.200 s.d. vs a null maximum of 0.129. The Kr control is ≈ 0 for every model.
  - For the small models the results split: dsets s0 0.202 and jedis s1 0.209 are above all nulls, while dsets s1 0.133 and
    jedis s0 0.181 are inside the null range. A permuted jedis copy reaches 0.200.
  - A pooled sum of a ReLU MLP over particles responds to an angular scale-up at fixed pT even without training.
- Resample ablation gives the same unit ordering as mean ablation (Spearman 0.965 JEDI-linear, 0.92–0.98 per stage). Every
  live unit keeps a resample cost > 1 % (BH). The context round g changes about twice as many calls under resample (27.8 %)
  as under mean ablation (14.6 %; earlier sample 14.9 %).

Competing hypotheses:
- "Architecture/input artefact" is excluded for H1 and for the accuracy drop under W on all three architectures.
- It is excluded for H4 on JEDI-linear only.
- It is not excluded for the rank-window ratio H3: masks alone reproduce it.
- "Resample reorders units / g < 5 %" is excluded.

## Issues and deviations
- Permuted networks are often degenerate (constant call; all of variant b). "Calls changed" metrics are then 0 for those
  nulls and 21–63 % for the non-degenerate ones. A null that keeps the per-layer output scale, such as a re-initialised but
  trained-statistics network, would make H2 informative.
- The frozen radial rule picked S50, not S29 (S29 numbers above).
- The JEDI-linear accuracy on this sample (81.47 %) is 0.17 pp above the stated 80.6–81.3 % range, within 1 bootstrap s.e.
- The training and the four measurement jobs ran on 2026-10-17 (code d4692a5); the combine step ran on 2026-10-19 (code
  aa5dfbe; `x8_nulls.py` unchanged in between). The shuffled-label and jedis models were trained in that earlier GPU job; no
  new training was run in this session.
- The holdout for null comparisons uses intersection-union p (maximum over nulls). With 7–15 nulls, the rank-p floor is
  0.06–0.125.

## Explainer entry (for site/explain.py)
- **Question.** Do the headline input-edit responses of JEDI-linear require its trained weights, or do networks with the
  same architecture, inputs and quantizer masks but no trained function give the same responses?
- **Background.** An edit response can come from the geometry of the inputs and layers rather than from learned function.
  For example, any pooled ReLU sum grows when particles move outward. Functional nulls remove the learned function while
  keeping the architecture:
  - shuffled-label training: same data, no label information;
  - weight permutation: same weight values, rearranged.
- **Method.** Four edits on N = 10,000 jets (seed 307), each with a matched null edit, applied to trained models and to
  15 (JEDI-linear) or 7 (Deep Sets, JEDI-style float) nulls. The headline statistic for the translation is
  $\Delta_{W\to Z} = \frac{1}{N_W}\sum_{j\in W}\big([c_T(j)=Z]-[c_0(j)=Z]\big)$. The test is $H_0: \theta_{\rm trained}\le
  \theta_{\rm null}$, with $p_{\rm IUT}=\max_i p_i$ over nulls from a paired jet bootstrap, and Holm correction over 12 tests.
  Resample ablation replaces unit $c$ by $h_c(\pi(j))$ for random derangements $\pi$, instead of $\bar h_c$.
- **Result.** Figure `results/jedi/x8_nulls.png`. Trained vs null maximum:
  - W→Z flow: 0.75 vs ≤ 0.14.
  - Radial response: 0.20 vs ≤ 0.13 s.d.
  - Accuracy drop under WTA: 0.22 vs ≤ 0.02.
  - The window ratio 5.2 lies inside the range of mask-preserving permuted copies (0.5–11).
  - Resample vs mean ablation unit ranking: Spearman 0.97. g → resample changes 27.8 % of calls.
- **Checks.**
  - Own forward = `Jedi.run` exactly, before and after the permutations.
  - Each matched null edit gives ≈ 0 radial response.
  - Small-model forward = the torch module exactly.
  - Shuffled-label models are at chance accuracy on true labels.
- **Why it matters.** It separates the translation/mass and WTA results (A1/A10, D4), which need the trained weights, from
  the rank-window asymmetry (A3/A4 window part), which the zero-bit mask layout produces with any weights.
