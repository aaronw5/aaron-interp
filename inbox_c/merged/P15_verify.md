# P15 verify — round 2: mirror-sign test, third sample, atlas rotation control (2026-10-20)

Job `anrunw-ai-c-p15-verify2` (RTX 2080 Ti, 9 min, exit 0), `python -m interp.p15_verify2` (the name `c-p15-verify` was used by
round 1, below). Script `interp/p15_verify2.py` (imports the P15 functions; adds the opposite-sign edits, a both-sign DAS fit and
the standardised-rotation atlas control). Results `results/part/p15_verify2.json` (cluster `$AI_RESULTS/part/p15_verify2.json`);
log `logs/c-p15-verify2.log`. Round 1 (job `c-p15-verify`, file `_122`) is kept unchanged further down.

## Conclusion
On a third set of jets, a φ shift of −0.06 rad changes ParticleNet's W-vs-Z score exactly as much as a shift of +0.06 rad, yet the
first-layer direction that P15 fitted to the +0.06 shift moves the opposite way under −0.06 and reproduces −0.29 of that response
(full model; −0.22 kin). So the first-layer direction records which way the particles were moved, not a quantity that the network's
response depends on; the second-layer direction does respond to both signs (0.80 and 0.74 of the response). Two further P15
statements change on re-test: the statement that channels align less with physics observables than random directions reverses when
the random directions are drawn after putting all channels on the same scale, and freezing the neighbour graphs removes up to about
4 % of the response, not under 1.8 %. The small pooled-channel set that matches the W/Z accuracy reproduces on this sample.

## Code review (additions to round 1)
- No new bug found in `interp/p15_particlenet_internal.py`. Round 1's points stand (layer-1 graph unchanged by construction for
  rigid shifts, scalings and rotations; shuffled network keeps trained BatchNorm statistics; share is a ratio of sums).
- The per-jet comparison in part (5) is not like for like: the DAS prediction writes the counterfactual jet's own site value (a
  partial forward pass of the edited jet), while the gradient and LRP predictions use only the unedited jet. A higher per-jet r for
  DAS (0.61 against 0.45 / 0.40) is therefore expected from the construction and does not rank the methods.
- P15's "pT scaling hardly moves it" holds for the ec1 direction (0.018); for the ec2 direction the P15 ratio is 0.32 (full) and
  0.19 (kin), the same size as the angle-scaling ratio.

## Procedure
- **Sample.** JetClass `val_5M`, file `<sample>_124.root` of each class (P15 `_120`, round 1 `_122`), 2,000 random rows per class
  (`default_rng(45)`), N = 20,000; fit/test halves `default_rng(46)`; W/Z set 4,000 jets (1,987 fit, 2,013 test). Build with
  η = η_jet + sign(η_jet)·Δη.
- **Edits.** As P15, plus φ − 0.03 / 0.06 / 0.09 and η − 0.06 (every real particle). Checks: |Δlog m| ≤ 1.3e-7 (mean) for all φ
  shifts; Δlog m_A = +0.0862 (φ + 0.06) and +0.0862 (φ − 0.06), positive for 100 % of jets in both.
- **(a) Mirror-sign test.** DAS fitted exactly as P15 (φ + 0.03 / 0.06 / 0.09 counterfactuals, 300 Adam steps, 10-logit MSE, W/Z
  fit half) at ec1, ec2, ec3 (= pooled), fc; evaluated on the W/Z test half for φ ± 0.06, η ± 0.06, angk, pT × 1.25: share of the
  change in d = ℓ_W − ℓ_Z (ratio of sums), 10-logit IIA, per-jet Pearson r, and the mean change Δs of the direction's jet projection.
  - H0: share(φ − 0.06) = share(φ + 0.06); H1: they differ. Paired bootstrap over test jets (1,000 resamples), Holm over the 4 sites
    per model.
  - Predictions. A direction that encodes a sign-even quantity (|axis offset|, m_A, or any variable the network's response reads)
    gives Δs of the same sign for ± shifts and share(−) ≈ share(+). A direction that encodes the signed mean Δφ gives Δs of opposite
    sign and share(−) < 0 when the network's response has the same sign for both shifts.
- **(b) Both-sign fit.** The same DAS fitted on φ ± 0.03 / 0.06 / 0.09; share for each sign; |cos| with the (a) direction.
- **(c) Random directions.** 5 unit vectors per site, share for each sign.
- **(d) Graph freezes** for φ + 0.06 and φ − 0.06 (layer l frozen to the unedited graph; all; graphs only).
- **(e) Atlas control.** Pooled live channels against 256 random orthonormal directions of (i) the centred channels (P15's control)
  and (ii) the standardised channels (each channel scaled to unit variance before rotation); median over channels of max |Spearman ρ|
  over the 20 observables, conditional R², Mann–Whitney p. Dependence of max |ρ| on the channel's fraction of exact zeros; channels
  with fewer than 40 % zeros against the standardised rotation.
- **(f) Pooled-channel circuits** K = 8–64, 100 random live-channel sets, as P15.

## Numbers
**Unedited models on this sample.** Accuracy 83.95 % (full), 69.74 % (kin) on the 20,000 jets (P15 84.50 / 70.14 %; round 1 84.35 /
70.00 %).

**Model response to mirror-image shifts** (W/Z test half, N = 2,013; mean Δd with 95 % bootstrap interval; true W called Z).

| model | φ + 0.06 | φ − 0.06 | per-jet r(Δd₊, Δd₋) |
|---|---|---|---|
| full | Δd −0.483 [−0.524, −0.442]; W→Z 11.1 → 24.1 % | Δd −0.487 [−0.529, −0.445]; W→Z 24.1 % | 0.53 [0.49, 0.57] |
| kin | Δd −0.278 [−0.304, −0.253]; W→Z 17.8 → 26.7 % | Δd −0.274 [−0.300, −0.248]; W→Z 26.6 % | 0.58 [0.55, 0.62] |

**P15-style direction (fitted on + shifts only), evaluated on both signs.**

| model / site | share φ + 0.06 | share φ − 0.06 | Δs (+) | Δs (−) | share(−) − share(+), Holm p |
|---|---|---|---|---|---|
| full ec1 | 1.13 [1.06, 1.21] (IIA 0.31, r 0.65) | −0.29 [−0.33, −0.26] (IIA −0.14, r −0.08) | −1.99 [−2.01, −1.96] | +0.53 [0.50, 0.55] | −1.42 [−1.51, −1.33], 0.004 |
| full ec2 | 0.75 [0.70, 0.81] (0.29, 0.53) | 0.80 [0.74, 0.86] (0.29, 0.50) | +1.95 | +2.03 | +0.05 [−0.03, 0.13], 0.76 |
| full ec3 | 0.43 [0.39, 0.47] (0.30, 0.19) | 0.42 [0.38, 0.47] | −0.036 | −0.037 | 1.0 |
| full fc | 0.04 [0.03, 0.06] | 0.05 [0.03, 0.06] | +0.021 | +0.022 | 1.0 |
| kin ec1 | 1.25 [1.16, 1.37] (0.38, 0.52) | −0.22 [−0.26, −0.18] (−0.08, 0.01) | −2.04 [−2.08, −2.01] | +0.36 [0.33, 0.39] | −1.47 [−1.61, −1.37], 0.004 |
| kin ec2 | 0.73 [0.68, 0.80] (0.39, 0.52) | 0.74 [0.69, 0.81] (0.39, 0.54) | +2.50 | +2.50 | +0.01 [−0.06, 0.08], 1.0 |
| kin ec3 | 0.18 [0.15, 0.20] | 0.17 [0.15, 0.20] | −0.020 | −0.019 | 1.0 |
| kin fc | 0.17 [0.16, 0.19] | 0.18 [0.16, 0.19] | +0.171 | +0.173 | 1.0 |

- Random directions (5 per site): shares −0.10 to 0.26 (full ec1), −0.06 to 0.08 (kin ec1), |share| ≤ 0.07 elsewhere, either sign.
- Both-sign fit at ec1: full 0.82 [0.75, 0.89] (+) and 0.94 [0.86, 1.02] (−), |cos| with the + fit 0.78; kin 0.83 [0.76, 0.92] and
  0.98 [0.91, 1.09], |cos| 0.83. At ec2 the both-sign direction is the + fit direction (|cos| 0.94 full, 0.99 kin).
- Held-out shares of the + fit direction at full ec1: η + 0.06 0.78, η − 0.06 0.21, angk 0.20, pT × 1.25 0.02 (P15 0.85, –, 0.21,
  0.02). The model's own response is not symmetric in η: Δd −0.38 (η + 0.06) and −0.61 (η − 0.06), full.
- Edit-response ratios of the ec2 direction (Δs per Δlog m_A relative to φ + 0.06): angk 0.34, pT × 1.25 0.35 (full); 0.13, 0.18 (kin)
  (P15 0.33 / 0.32 and 0.15 / 0.19).
- Third replicate of the post-pooling share for φ + 0.06: full ec3 0.43 (P15 0.11, round 1 0.57); kin ec3 0.18 (0.29, 0.18); full fc
  0.04 (0.04, 0.05); kin fc 0.17 (0.25, 0.21).

**Graphs** (share of Δd left when the graphs are frozen to the unedited ones).

| model, edit | edges changed L1 / L2 / L3 | freeze L3 | freeze all | graphs only |
|---|---|---|---|---|
| full, φ + 0.06 | 0 / 5.3 / 8.6 % | 0.969 [0.957, 0.981] | 0.962 [0.942, 0.980] | −0.017 [−0.033, −0.002] |
| full, φ − 0.06 | 0 / 5.3 / 8.7 % | 0.975 [0.961, 0.988] | 0.966 [0.947, 0.986] | −0.015 [−0.030, 0.001] |
| kin, φ + 0.06 | 0 / 6.0 / 8.5 % | 0.993 [0.981, 1.005] | 1.000 [0.983, 1.019] | −0.033 [−0.048, −0.018] |
| kin, φ − 0.06 | 0 / 6.0 / 8.5 % | 0.998 [0.985, 1.012] | 1.007 [0.989, 1.027] | −0.040 [−0.057, −0.025] |

Freezing all graphs removes 3.8 % [2.0, 5.8] of the full model's response here, 4.0 % [2.1, 6.0] in round 1 and 0.7 % in P15.

**Atlas: pooled live channels against random directions** (median over channels / directions; Mann–Whitney p).

| model | metric | channels | rotation of centred channels (P15) | rotation of standardised channels |
|---|---|---|---|---|
| full (220 live) | max \|ρ\| | 0.311 | 0.394 (p 1.8e-7) | 0.266 (p 1.0e-4) |
| full | conditional R² | 0.070 | 0.192 (p 5e-24) | 0.103 (p 6e-5) |
| kin (191 live) | max \|ρ\| | 0.359 | 0.396 (p 2.0e-3) | 0.280 (p 1.4e-5) |
| kin | conditional R² | 0.108 | 0.212 (p 1e-16) | 0.118 (p 0.25) |

- The centred-channel rotation weights each direction by the channels' variances; the high-variance channels are the ones most tied to
  mass, pT and multiplicity, so P15's control favours the rotated directions. With equal weights the channels have the larger single-
  observable correlation.
- Median fraction of jets on which a pooled channel is exactly 0: 0.69 (full), 0.71 (kin). Spearman(max |ρ|, fraction of zeros)
  −0.22 (full), −0.17 (kin). Channels with < 40 % zeros (50 full, 41 kin): median max |ρ| 0.35 / 0.31 against 0.26 / 0.27 for the
  standardised rotation (p 1.2e-3, 7.2e-3).

**Pooled-channel circuits** (W/Z test N = 2,013; 10-class read N = 5,000 test jets).

| model, K | W/Z acc. (model) | random sets mean (95th pct) | McNemar right→wrong / wrong→right, p | 10-class (model) |
|---|---|---|---|---|
| full, 16 | 63.2 % (70.9 %) | 17.8 % (23.3 %) | 315 / 160, 1e-12 | 20.3 % (83.4 %) |
| full, 32 | 69.3 % [67.2, 71.2] | 20.7 % (27.2 %) | 181 / 150, 0.099 | 35.0 % |
| full, 64 | 73.4 % | 28.5 % (40.5 %) | 87 / 138, 8e-4 | 42.6 % |
| kin, 8 | 54.7 % (58.2 %) | 22.4 % (25.6 %) | 300 / 230, 0.003 | 14.9 % (69.4 %) |
| kin, 16 | 59.9 % [57.5, 61.9] | 23.5 % (27.9 %) | 155 / 189, 0.075 | 24.0 % |
| kin, 32 | 60.0 % | 26.0 % (34.6 %) | 110 / 147, 0.025 | 35.8 % |

Permutation p against random sets at the 1/101 floor for every K.

## Competing readings and which are excluded
- "The ec1 direction is a variable the response passes through" predicts share(φ − 0.06) ≈ share(φ + 0.06). Measured −0.29 / −0.22
  against 1.13 / 1.25 (Holm p 0.004 each): excluded for the P15 ec1 direction in both models.
- "The ec1 direction encodes the signed mean Δφ" predicts Δs of opposite sign under ± shifts: measured −1.99 / +0.53 (full),
  −2.04 / +0.36 (kin). Consistent; the − response is 3–6 times smaller, so the direction also has a sign-even part.
- "The ec2 direction encodes a sign-even quantity" predicts equal Δs and shares for ± shifts: measured Δs +1.95 / +2.03, shares 0.75 /
  0.80 (full); 2.50 / 2.50, 0.73 / 0.74 (kin). Not excluded. An m_A copy is excluded at ec2 (angle- and pT-scaling ratios 0.13–0.35
  against a prediction of 1).
- "Pooled channels are less aligned with observables than random directions" depends on the rotation control: the sign of the
  difference reverses between the centred and the standardised rotation, so it is not supported as a property of the channel basis.

## Corrected reading
- **Single direction.** One jet-level direction at EdgeConv 1–2 reproduces the summed φ + 0.06 change in d (0.73–1.43 over three
  samples), with 10-logit IIA 0.27–0.39 and per-jet r 0.48–0.65; random directions do not. The same match is reached by an unfitted
  mean-difference direction, by a freshly initialised network and, in most cases, by the weight-shuffled network (round 1).
  - At EdgeConv 1 the direction fitted on + shifts encodes the shift's sign: under φ − 0.06, which changes d by the same amount, it
    moves the opposite way and reproduces −0.29 (full) / −0.22 (kin) of the response. A direction fitted on both signs (|cos| 0.78–0.83
    with it) reproduces 0.82–0.98 of each.
  - At EdgeConv 2 the direction responds equally to both signs and reproduces 0.74–0.80 of each.
  - After pooling (ec3, fc), one direction reproduces 0.04–0.57 depending on the sample (full ec3: 0.11, 0.57, 0.43 on three samples),
    with per-jet r ≤ 0.22.
- **Edit responses.** Per unit Δlog m_A, angle scaling moves the ec1–ec2 directions 0.11–0.39 of their φ response; pT scaling moves
  the ec1 direction −0.05 to 0.02 and the ec2 direction 0.15–0.36. The kin FC direction responds about equally to all four edits per unit
  m_A (0.91–0.98) and reproduces 0.17–0.25 of the response.
- **Graphs.** The layer-1 graph cannot change under these edits (by construction). Freezing the later graphs removes 0.7–4.0 % of the
  full model's response over three samples (−1.3 to 0.1 % for kin).
- **Literal circuits.** At the pooled level, 32 channels (full) and 16 channels (kin) reach the model's W/Z accuracy within two
  points (McNemar p 0.075–1.0 at full K = 32 and kin K = 16 over three samples), while 15–17 % of W/Z jets change correctness in
  either direction. Their 10-class accuracy is 20–45 % against 69–85 %; the channels were ranked by W-vs-Z attribution. At ec2 and fc,
  sets of up to 32 channels stay below the model, except kin fc K = 32 (58.5 % against 60.1 %, McNemar p 0.15).
- **Atlas.** Relative to random directions, pooled channels have a lower single-observable correlation under P15's centred rotation
  and a higher one under a standardised rotation; the comparison does not establish that the channels are less aligned with physics
  observables.
- **Baselines.** EFP and Lund-plane probes do not move under a φ shift by construction. Gradient and LRP linearisations from the
  unedited jet give per-jet r 0.31–0.45 with the φ + 0.06 response; the DAS prediction (0.51–0.61 at ec1–ec2) uses the edited jet's
  site value and is not comparable.

## Explainer entry (mirror-sign test)
**Question.** Does the EdgeConv-1 direction that reproduces ParticleNet's response to a +0.06 rad φ shift also carry the response to
the mirror-image −0.06 rad shift?

**Background.** A rigid shift of every particle by $\pm\delta$ in φ leaves the invariant mass $m$ unchanged and raises the axis-moment
mass $m_A=\sqrt{\sum_i p_{T,i}\sum_i p_{T,i}\Delta R_i^2}$ by the same amount for either sign. ParticleNet's score
$d=\ell_W-\ell_Z$ falls by the same amount for both signs (full: $-0.483$ and $-0.487$). A variable that the response is computed from
must therefore change the same way for $+\delta$ and $-\delta$.

**Method.** DAS at EdgeConv 1 fits a unit vector $q$ so that writing the counterfactual's jet-mean component along $q$,
$h_p\leftarrow h_p+\big((\bar h_{cf}-\bar h)\cdot q\big)q$, reproduces the counterfactual logits for $+0.03,+0.06,+0.09$ rad. The
fitted $q$ is then used with the $-0.06$ rad counterfactual. Share
$S=\sum_j(d^{int}_j-d_j)/\sum_j(d^{cf}_j-d_j)$; projection change $\Delta s=\overline{(\bar h_{cf}-\bar h)\cdot q}$. Test of
$H_0: S_-=S_+$ by paired bootstrap over jets; control: 5 random $q$; second fit on both signs.

**Result.** Full model: $S_+=1.13$, $S_-=-0.29$; $\Delta s_+=-1.99$, $\Delta s_-=+0.53$ (Holm $p=0.004$). Kin: $S_+=1.25$,
$S_-=-0.22$. At EdgeConv 2: $S_+=0.75$, $S_-=0.80$ (full). A both-sign fit at EdgeConv 1 gives $S_\pm=0.82,\ 0.94$.

**Checks.** $|\Delta\log m|\le1.3\times10^{-7}$ for every shift; $\Delta\log m_A=+0.086$ for both signs on 100 % of jets; random
directions $|S|\le0.26$; third independent file and seed.

**Why it matters.** Reproducing one edit's averaged response with one direction does not identify a variable the network uses: the
EdgeConv-1 direction tracks the sign of the shift, which the network's output ignores. Mirror-image edits with equal output effect are
a cheap test that a candidate variable must pass.

---

## Round 1 (job c-p15-verify) — P15 verify — ParticleNet internals on a fresh sample, with untrained-network and mean-difference controls (2026-10-20)

Job `anrunw-ai-c-p15-verify` (GTX 1080 Ti, 21 min, exit 0), `python -m interp.p15_verify`. Script `interp/p15_verify.py` (imports the
P15 functions from `interp/p15_particlenet_internal.py`; nothing re-implemented except the file choice, the BatchNorm re-estimation and
the extra controls). Results `results/part/p15_verify.json` (cluster: `$AI_RESULTS/part/p15_verify.json`); log `logs/c-p15-verify.log`.

### Conclusion
On a fresh set of jets, one direction in ParticleNet's first two graph layers again matches the average response of the W-vs-Z score to
a rigid φ shift that leaves the jet mass unchanged. It does not behave like the axis-moment mass, and fixing the neighbour graphs removes
only a few percent of the response. But the plain average-change direction, with no fitting, matches the average equally well, and so
does the same search in a freshly initialised, untrained ParticleNet, so matching the average says nothing about a trained variable.
Per jet, the early-layer direction tracks the response only partly (correlation about 0.5–0.65), and after pooling no single direction
tracks it (correlation at most about 0.35). The P15 numbers for the pooled layer (0.11) and for the graphs (at most 1.8 %) did not
reproduce exactly (0.57 and 4 %).

### Code review (interp/p15_particlenet_internal.py)
- Layers and hooks. `fw` follows weaver's ParticleNet forward (bn_fts, three EdgeConv blocks with k = 16, masked mean, fc[0] = Linear +
  ReLU + Dropout, fc[1] = Linear); the P15 check gives max |ΔL| = 0 against `model(...)`. ec1/ec2/ec3 hooks act on the block outputs and
  are re-masked; the fc hook acts after fc[0] (post-ReLU), the input of the last Linear. Class indices: Zqq = 6, Wqq = 7 (asserted).
- Padding. Hooks multiply by the mask; jet means divide by the real-particle count; the kNN coordinate shift of 1e9 on padded slots is
  as in weaver; graph-overlap counts use real particles only. No padded particle enters a metric.
- Rows. 2,000 random rows per class from one file (`rng.choice`, sorted); no first-k selection. Fit/test halves from a random
  permutation.
- Build. `pc.build` uses η = η_jet + sign(η_jet)·Δη and recomputes jet pT and energy. Edit checks on the fresh sample: φ shifts
  |Δlog m| ≤ 5e-8 (mean); η + 0.06 4.5e-6; rotation 0.3 −1.8e-4; angk Δlog m_A = 0.0873 against φ + 0.06's 0.0873.
- Edits change only what is stated, with three points to scope the reading:
  - The layer-1 kNN graph uses (Δη, Δφ) as coordinates. A common φ shift, a common angle scale and a rotation all leave every pairwise
    distance ordering unchanged, so "0 edges changed at layer 1" holds by construction and is not a measured property of the network.
  - The weight-shuffled control keeps the trained BatchNorm statistics. Its logits are 20–100 times larger than the trained network's
    (mean |d| 110 full, 63 kin on this sample, against 2.4 and 1.1), so its response to the edit is on another scale.
  - The share metric is a ratio of sums over jets, Σ(d_int − d)/Σ(d_cf − d). At fc, where d is linear in the units, the mean-difference
    direction reproduces this ratio by construction (= 1 on the jets used to define it).
- The circuit "W/Z accuracy" is accuracy, not call agreement: McNemar counts show which jets change correctness.
- No other bug found.

### Procedure
- **Sample.** JetClass `val_5M`, file `<sample>_122.root` of each class (P15 used `_120`), 2,000 random rows per class
  (`default_rng(25)`), N = 20,000; fit/test halves `default_rng(26)`; W/Z set 4,000 jets (2,042 fit, 1,958 test). Same edits as P15.
- **Rerun.** Edit response; one-direction DAS at ec1, ec2, ec3 (= pooled), fc (300 Adam steps, φ + 0.03/0.06/0.09 counterfactuals,
  10-logit MSE); 5 random unit directions; the weight-shuffled network (now shuffle seed 25); P8v ratios and edit regression; graph
  freezes for φ + 0.06; pooled-channel literal circuits K = 8–64 with 100 random live-channel sets (W/Z test half, 10-class read on
  5,000 test jets with uniform donors).
- **Added controls.**
  - (a) DAS restarted from a second random init: |cos| between the two directions and the second's share.
  - (b) Mean-difference direction $q=\overline{\Delta h}/|\overline{\Delta h}|$, $\overline{\Delta h}$ = mean over W/Z fit jets of the
    jet-level site value change under φ + 0.06; no optimisation.
  - (c) Two untrained networks with data-scaled activations: `perm_bn` = the weight-shuffled network with every BatchNorm's running
    statistics re-estimated on 20 batches of 500 jets of this sample; `reinit_bn` = a freshly initialised ParticleNet (PyTorch default
    init, `torch.manual_seed(25)`) with the same BatchNorm re-estimation. The same DAS search is fitted on each.
- **Metrics.** Share of the φ + 0.06 change in d = ℓ_W − ℓ_Z reproduced (ratio of sums); 10-logit IIA; per-jet Pearson r between the
  interchange's Δd and the observed Δd. 1,000 bootstrap resamples over the 1,958 test jets (percentile 95 % intervals).
- **Hypotheses.** H0: the trained network's direction reproduces the same share as the control network's direction (bootstrap of the
  paired difference, two-sided, floor 1/1001). H1: the shares differ. Competing predictions for the main P15 claim "one early-layer
  direction carries the response": (i) a trained, specific variable predicts share ≈ 1 with high per-jet r in the trained network and
  share ≈ 0 or low r in untrained networks and for the untrained mean-difference direction; (ii) a generic low-rank response predicts
  share ≈ 1 in untrained networks and for the mean-difference direction as well.

### Numbers
**Unedited model on this sample.** Accuracy 84.35 % (full), 70.00 % (kin) on the 20,000 jets (P15: 84.50 / 70.14 %). True W called Z:
10.4 → 22.9 % (full), 18.0 → 28.4 % (kin) under φ + 0.06 (P15: 10.8 → 25.4, 17.4 → 28.0). Mean Δd on the W/Z test half −0.471 (full),
−0.267 (kin) (P15: −0.47, −0.26).

**Share of the φ + 0.06 change in d reproduced by one direction** (test N = 1,958; IIA and per-jet r in brackets).

| model / site | P15 DAS | DAS, fresh sample | DAS restart (|cos|) | mean-difference direction | random (5) |
|---|---|---|---|---|---|
| full ec1 | 1.16 | 1.19 [1.12, 1.28] (IIA 0.31, r 0.64) | 1.22 (0.977) | 1.34 (IIA 0.16, r 0.62) | −0.06 to 0.13 |
| full ec2 | 0.86 | 0.86 [0.80, 0.94] (0.29, 0.54) | 0.90 (0.978) | 1.03 (0.17, 0.53) | −0.01 to 0.07 |
| full ec3 / pooled | 0.11 | 0.57 [0.52, 0.63] (0.31, 0.15) | 0.53 (0.869) | 1.27 (−0.03, 0.29) | −0.01 to 0.04 |
| full fc | 0.04 | 0.05 [0.04, 0.06] (0.55, 0.08) | 0.05 (0.906) | 0.96 (0.20, 0.34) | −0.02 to 0.04 |
| kin ec1 | 1.43 | 1.28 [1.18, 1.40] (0.36, 0.48) | 1.28 (0.994) | 1.91 (0.13, 0.46) | −0.06 to 0.11 |
| kin ec2 | 0.84 | 0.79 [0.73, 0.87] (0.39, 0.49) | 0.87 (0.990) | 1.35 (0.19, 0.43) | 0.00 to 0.04 |
| kin ec3 / pooled | 0.29 | 0.18 [0.15, 0.20] (0.44, −0.05) | 0.08 (0.897) | 1.45 (0.17, −0.17) | −0.01 to 0.06 |
| kin fc | 0.25 | 0.21 [0.19, 0.23] (0.48, 0.14) | 0.19 (0.953) | 0.99 (0.31, 0.30) | −0.03 to 0.00 |

**Same DAS search on untrained networks** (share; IIA; per-jet r). Mean |Δd| under φ + 0.06 of each network in the header.

| model / site | trained | shuffled, trained BN (|Δd| 23 / 43) | shuffled, BN re-estimated (3.3 / 10) | fresh init, BN re-estimated (0.017 / 0.033) |
|---|---|---|---|---|
| full ec1 | 1.19; 0.31; 0.64 | 1.04; 0.62; 0.33 | 0.04; 0.12; 0.08 | 1.02; 0.81; 0.68 |
| full ec2 | 0.86; 0.29; 0.54 | 0.86; 0.64; 0.24 | −0.21; 0.17; 0.16 | 1.03; 0.77; 0.56 |
| full ec3 | 0.57; 0.31; 0.15 | 0.85; 0.92; 0.35 | −0.08; 0.77; 0.37 | 1.00; 0.75; 0.43 |
| full fc | 0.05; 0.55; 0.08 | 0.77; 0.95; 0.34 | −0.12; 0.82; 0.44 | 0.89; 0.72; 0.03 |
| kin ec1 | 1.28; 0.36; 0.48 | 1.00; 0.53; 0.50 | 0.94; 0.30; 0.50 | 0.79; 0.54; 0.65 |
| kin ec2 | 0.79; 0.39; 0.49 | 0.07; 0.58; 0.17 | 0.94; 0.23; 0.49 | 1.09; 0.50; 0.57 |
| kin ec3 | 0.18; 0.44; −0.05 | 0.24; 0.84; 0.28 | 0.28; 0.65; 0.08 | 1.00; 0.44; 0.35 |
| kin fc | 0.21; 0.48; 0.14 | 0.20; 0.86; 0.22 | 0.21; 0.68; 0.05 | 0.59; 0.38; −0.33 |

- Trained minus fresh-init share at ec1: +0.17 [0.09, 0.26] (full), +0.49 [0.37, 0.63] (kin); at ec2: −0.17 [−0.24, −0.09] (full),
  −0.30 [−0.42, −0.17] (kin); bootstrap p ≤ 0.001 each (Holm over the 8 site × model tests: ≤ 0.008). The fresh-init direction
  reproduces at least 0.79 of its own network's response at ec1–ec2 in both models.
- The shuffled network with trained BN statistics gives 0.86–1.04 at full ec1–ec2 (P15: 0.95, 1.23) but 0.07 at kin ec2 with this
  shuffle seed (P15: 2.29).

**Direction readings (DAS direction, fresh sample).**
- P8v ratios (response per Δlog m_A relative to φ + 0.06; an m_A variable predicts 1): full ec1 angk 0.310 [0.304, 0.316], pT × 1.25
  0.018 [0.017, 0.020], η + 0.06 0.604; full ec2 0.345 / 0.363 / 1.24; kin ec1 0.331 / −0.051 / 0.571; kin ec2 0.114 / 0.150 / 1.29.
  P15: 0.31 / 0.018 / 0.62 (full ec1). Kin fc: 0.911 / 0.958 / 0.993 (P15 0.92–0.98).
- Edit-regression ratio b_A/b_inv: −1.13 (full ec1), −1.56 (full ec2), −1.06 (kin ec1), −1.17 (kin ec2); kin fc 2.95 [2.35, 3.82]
  (P15 2.49).
- Held-out shares at full ec1: η + 0.06 0.91, angk 0.22, pT × 1.25 0.03 (P15: 0.85, 0.21, 0.02).

**Graphs (φ + 0.06; share of Δd left when graphs are frozen to the unedited ones).**

| | edges changed L1 / L2 / L3 | freeze L2 | freeze L3 | freeze all | graphs only |
|---|---|---|---|---|---|
| full | 0 / 5.3 % / 8.9 % | 0.980 [0.962, 0.997] | 0.984 [0.970, 0.998] | 0.960 [0.940, 0.979] | −0.024 [−0.041, −0.007] |
| kin | 0 / 6.1 % / 8.7 % | 0.995 [0.976, 1.012] | 0.982 [0.970, 0.995] | 0.999 [0.980, 1.018] | −0.043 [−0.059, −0.028] |

P15: freeze all 0.993 (full), 1.013 (kin); freeze L3 0.982 / 0.987. On the fresh sample, freezing all graphs removes 4.0 % [2.1, 6.0]
of the full model's response.

**Pooled-channel literal circuits** (W/Z test N = 1,958; 10-class read N = 5,000 test jets, uniform donors).

| model, K | W/Z acc. (model) | random sets mean (95th pct) | F | McNemar right→wrong / wrong→right, p | 10-class (model) |
|---|---|---|---|---|---|
| full, 16 | 62.5 % (71.4 %) | 16.1 % (21.5 %) | 0.70 | 327 / 152, p 9e-16 | 20.5 % (83.8 %) |
| full, 32 | 71.2 % [69.1, 73.2] (71.4 %) | 19.2 % (26.8 %) | 0.86 | 155 / 152, p 0.91 | 35.7 % |
| full, 64 | 74.6 % | 29.2 % (42.7 %) | 0.95 | 75 / 137, p 2.5e-5 | 43.6 % |
| kin, 16 | 61.2 % (60.5 %) | 22.4 % (28.1 %) | 0.88 | 161 / 175, p 0.48 | 21.5 % (70.1 %) |
| kin, 32 | 61.5 % | 24.7 % (32.2 %) | 0.93 | 113 / 133, p 0.23 | 34.1 % |

Permutation p against random sets at the 1/101 floor for every K. At full K = 32, 307 of 1,958 W/Z jets (15.7 %) change correctness
in one direction or the other, balanced, so the accuracy matches while the calls differ on those jets.

**Not rerun:** atlas, LRP and EFP/Lund probes (no code issue found; caveat below).

### Checks and caveats
- The random-rotation atlas control rotates the centred but unstandardised channels; a random direction then weights the
  high-variance channels, which are the ones most tied to mass, pT and multiplicity. "Channels align less than random rotations" may
  reflect this weighting; a rotation of the standardised channels was not run.
- The circuits are ranked by W-vs-Z attribution, so a low 10-class accuracy is expected from the selection; it does not test whether a
  10-class circuit of this size exists.
- The fresh-init network's absolute response is 15–30 times smaller than the trained network's; the share is scale-free, the
  conclusion from it is only that a one-direction match is reachable without training.

### Corrected reading
- At EdgeConv 1–2, one jet-level direction reproduces the summed φ + 0.06 change in d (0.79–1.28 on the fresh sample; 0.84–1.43 in P15),
  in both models, with 10-logit IIA 0.29–0.39 and per-jet r 0.48–0.64. Random directions do not (≤ 0.13). The DAS fit converges to the
  same direction from two inits (|cos| 0.98–0.99).
- The match of the summed change does not need the fit or a trained network: the unfitted mean-difference direction gives 1.03–1.91
  at ec1–ec2 with similar per-jet r (0.43–0.62), and the same search in a freshly initialised ParticleNet with data-estimated BatchNorm
  gives 0.79–1.09. The weight-shuffled network with trained BatchNorm gives such a direction in 3 of 4 (model, site) cases; with
  re-estimated BatchNorm in 2 of 4.
- After pooling, the DAS direction's share depends on the sample (full ec3 0.11 in P15, 0.57 here), while the mean-difference direction
  reproduces the summed change (0.96–1.45; at fc ≈ 1 by construction). No single direction at ec3 or fc tracks the per-jet response
  (r ≤ 0.35 for any direction; DAS 0.08–0.15 full, −0.05–0.14 kin).
- Edit responses at ec1–ec2 replicate: per Δlog m_A, angle scaling moves the direction 0.11–0.35 of the φ response and pT scaling
  −0.05 to 0.36; the η shift 0.57–1.29. The kin fc direction responds equally to all four edits per unit m_A (0.91–0.99) and carries
  0.21 of the summed response with per-jet r 0.14.
- The layer-1 graph cannot change under these edits (by construction). Freezing the layer-2 and layer-3 graphs removes 0–4 % of the
  response (P15: ≤ 1.8 %); the edited graphs alone reproduce −2 to −6 %.
- Pooled-channel circuits of 16–32 channels match the W/Z accuracy (McNemar p 0.23–0.91), while ~13–16 % of W/Z jets change
  correctness and F = 0.86–0.93; their 10-class accuracy is 20–36 % against 70–84 %.

### Explainer entry
**Question.** Does a single direction inside ParticleNet carry its response to a mass-preserving φ shift, and is that a property of the
trained network?

**Background.** Shifting every particle by the same Δφ keeps the invariant mass $m$ but moves the particles away from the stored jet
axis, which raises the axis-moment mass $m_A=\sqrt{\sum_i p_{T,i}\,\sum_i p_{T,i}\Delta R_i^2}$. ParticleNet's W-vs-Z score
$d=\ell_W-\ell_Z$ drops under this shift (true W called Z: 10 → 23 %). Distributed alignment search (DAS) finds a unit vector $q$ at a
layer such that copying only the component along $q$ from the shifted jet into the unshifted jet reproduces the shifted output.

**Method.** At site $s$ with jet-level mean $\bar h$, the interchange is
$h_p \leftarrow h_p + \big((\bar h_{cf}-\bar h)\cdot q\big)\,q$ for every particle $p$. Share of the change reproduced:
$S=\sum_j (d^{int}_j-d_j)\big/\sum_j (d^{cf}_j-d_j)$; per-jet agreement $r=\mathrm{corr}_j(d^{int}_j-d_j,\;d^{cf}_j-d_j)$. Controls:
random $q$; the unfitted $q\propto\overline{\bar h_{cf}-\bar h}$; and the same fit in untrained ParticleNets whose BatchNorm statistics are
re-estimated on the same jets.

**Result.** Table "Same DAS search on untrained networks" above; at EdgeConv 1, $S=1.19$ (trained, full) and $S=1.02$ (fresh init).

**Checks.** Fresh file and seed; two DAS inits; graph freezes; edit sanity ($|\Delta\log m|\le 5\times10^{-8}$ under the φ shift).

**Why it matters.** A single-direction match of an averaged response is reached without training, so it cannot by itself identify a
learned physics variable; the per-jet agreement and the edit-ratio tests (which exclude an $m_A$ copy at EdgeConv 1–2) carry the
information.
