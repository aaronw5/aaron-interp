# P13: learned-feature circuit in ParT_full (transcoders + class-token SAEs), agent report 2026-10-20

Job `anrunw-ai-c-p13-run` (RTX 2080 Ti, 2,723 s), `python -m interp.p13_transcoder_part`. Smoke `c-p13-smoke` (2 launches: the
first stopped on an empty training-history log line, fixed). First full launch OOM-killed at 16 Gi (two jet pools plus a CPU cache
of class-token activations); fixed by building jets file by file and keeping the class-token cache on the GPU (debug attempt 1 of
2). Figure re-rendered by `c-p13-plot` (job.sh, CPU). Script `interp/p13_transcoder_part.py`. Results
`results/part/p13_transcoder.json`, IG scores `results/part/p13_scores.npz`, figure `results/part/p13_feature_circuit.png`, logs
`logs/c-p13-run.log`, `logs/c-p13-smoke.log`, `logs/c-p13-plot.log`. Dictionaries (not committed):
`/j-jepa-vol/aaron-interp/results/part/p13_dicts.pt` (`--reuse` loads them).

## Conclusion
We trained sparse dictionaries ("features") on ParT_full's own activations and searched for a small set of features that alone
reproduces its W-versus-Z calls, with everything else in the dictionaries replaced by values from a Z (or W) jet of the same pT.
The dictionaries replace the network's MLPs with no loss of accuracy, but 50 features give 50 % W/Z accuracy (model 84 %), and 75 %
needs between 1,000 and 2,000 features. These numbers hold only while all attention layers and the transcoders' linear bypass
terms stay computed on the real jet; when those are also replaced, even all features together give 42 %. Steering or removing the
single feature most correlated with the axis-moment mass m_A does not change the calls, so the m_A information is not carried by
one feature. The prediction of a ≤ 50-feature circuit at ≥ 75 % is excluded; whether a joint search over features and attention
heads would give a short circuit is still open.

## Procedure
- **Model.** ParT_full, released weights, weaver-core 0.4.17. The forward pass is re-implemented with hook points (`Runner`);
  max |Δlogit| against weaver's own forward on 64 jets: 0.0.
- **Dictionaries (units).**
  - P0–P7: skip-transcoders on the 8 particle-block MLPs. Input u = pre_fc_norm output (128), target y = fc2 output (128).
    \(\hat y = \mathrm{TopK}_{32}(u W_{enc}^T + b_{enc}) W_{dec} + b_{dec} + u W_{skip}^T\), 1,024 latents (8×).
  - C0, C1: the same on the 2 class-block MLPs (class token).
  - R0, R1: TopK SAEs (2,048 latents, 16×, k = 32) on the class-token residual after class block 1 / 2.
  - Loss: normalised MSE plus AuxK (1/32) on latents dead for > 50 (particle) / 500 (class) steps; unit-norm decoder rows; Adam
    1e-3 with warm-up and a linear decay over the last 20 %.
- **Training data.** JetClass train_100M, 4 rounds × 10 classes; each round uses one random file per class and 25,000 random rows
  of it (numpy default_rng(13) for files, 1000·round + class for rows): 1,000,000 jets, 40 files. Particle sites trained online
  (2 passes per 250,000-jet pool, batch 256 jets, all real particle tokens; 7,816 steps). Class sites trained from a GPU cache of
  the 1,000,000 class tokens (10 epochs, batch 1,024; 9,760 steps).
- **Controls trained the same way.** (i) Weight-permuted ParT_full (each weight tensor's entries permuted, seed 5): dictionaries at
  P0, P7, C0, C1, R0, R1. (ii) An SAE on a random orthogonal rotation Q of the R1 residual (rotR1). (iii) Untrained units scored
  with the same atlas metric: each MLP's 512 hidden neurons (post-GELU), the raw 128 input coordinates, and the Q-rotated input
  coordinates.
- **Samples (never first-k, seed 13).**
  - ALL: test_20M `<sample>_105.root`, 2,000 random rows per class, N = 20,000. Accuracy, FVU, splice, atlas, 10-class read.
  - DISC: test_20M WToQQ/ZToQQ_106.root, 1,500 + 1,500 random rows. Attribution, m_A scan, steering calibration.
  - EVAL: test_20M WToQQ/ZToQQ_107.root, 5,000 + 5,000 random rows. Circuits kept alone, steering and ablation.
  - None of these files was used by P1–P9 (they used test _100–_102 and val _120–_124).
- **Replacement model.** At every site, out = \((a W_{dec} + b_{dec} + \mathrm{skip} + \varepsilon)\, s_{out}\) with the error
  \(\varepsilon\) = target − reconstruction, so the replacement model is exact.
- **Resample ablation (P9's corruption B).** Donor = an opposite-class jet (W↔Z) with matched jet pT (one of the 5 nearest;
  median |ΔpT|/pT 1.2e-4). A feature not kept carries the donor's value (slot by slot in pT order at particle sites); error terms
  likewise. Variants:
  - main: features outside the circuit and all error terms resampled; attention layers and skip terms computed live.
  - errlive: as main, error terms live.
  - strict: as main, plus skip terms and particle-block attention outputs resampled. Class attention stays live (it is the only
    path from the particles to the class token).
  - With nothing kept, the R1 SAE carries only donor values, so the logits equal the donor's exactly (check below).
- **Attribution graph.** \(D = \sum_j s_j(\ell_W - \ell_Z)\), \(s = \pm 1\) for true W/Z. Node score = mean over
  \(\alpha \in \{0.1, 0.3, 0.5, 0.7, 0.9\}\) of \(\partial D / \partial m\) at all masks = \(\alpha\) (integrated gradients over
  masks interpolating donor → clean). One mask per feature (shared over particle slots) and one per site error node (14,336
  features + 12 error nodes). Circuits = top-K live features by |score|.
- **Atlas.** For each unit, the conditional-mean \(R^2\) (correlation ratio, 20 quantile bins or the categories) against 20 fixed
  observables:
  - particle sites: log pT, log pT/pT_jet, log E, ΔR, Δη, Δφ, pT rank, 5 particle-ID flags, charge, tanh d0, tanh dz, jet log m,
    jet log m_A, jet log pT, multiplicity, ΔR to the leading particle;
  - class sites: log m, log m_A, log pT, multiplicity, M2/M0, |M1|/M0, pTD, leading-pT fraction, ΔR12, log m12, e2, N95, charged /
    neutral-hadron / photon / lepton counts, max |tanh d0|, pT-weighted jet charge, |η|, leading-particle ΔR.
  - Monosemanticity = max of the 20. Live = fires at least once on ALL.
  - Cards for the top-50 circuit features: top-3 observables, class profile, top-100-activating jets' classes, response profiles
    (ΔR, pT, m_A, PID / m_A, m, pT, multiplicity, M2/M0), negative and positive pole, upstream cosine (encoder row vs earlier
    circuit features' decoder rows), and for R1 the linearised direct effect on \(\ell_W - \ell_Z\).
- **Steering.**
  - Scan: Pearson r of every live class-level feature (C0, R0, C1, R1; 1,233 features) with log m_A on DISC, where
    \(m_A = \sqrt{\sum p_T \cdot \sum p_T \Delta R^2}\); BH over the scan. f* = the largest |r|.
  - Calibration: natural same-class pairs (partner among the 6 nearest in jet pT), \(\beta = \sum \Delta a\,\Delta\log m_A /
    \sum (\Delta\log m_A)^2\).
  - On the 5,000 EVAL W jets: add \(\beta \log 1.05\) to f* (a 5 % m_A increase), or \(\beta\,\Delta\log m_A\) of the jet's own
    φ + 0.06 edit, and compare the W→Z flow (share of true W called Zqq, minus the unedited share) with the φ + 0.06 edit's flow.
  - Ablate f* to its DISC mean and re-run the φ + 0.06 test.
  - Control: 20 random live features of the same site, steered by the same amount in standard-deviation units (sign of their own
    m_A correlation) and ablated to their mean.
- **Statistics.** Bootstrap over jets (1,000), 95 % intervals; McNemar for paired call changes; permutation p against 100 random
  feature sets of equal size (per-site counts matched, drawn from live features) on EVAL[::10] (1,000 jets); BH over the m_A scan;
  Mann-Whitney (one-sided) and a bootstrap of the mean difference for the atlas comparisons.
- **Hypotheses.**
  - H1 (thesis): a ≤ 50-feature circuit (errors resampled) reaches ≥ 75 % W/Z (model ~83 %); it contains an m_A feature whose
    steering gives ≥ 0.6 of the φ-edit W→Z flow; atlas features are more monosemantic than rotated-basis controls.
  - H0 (alternative): feature circuits cap near the P9v 63 % ceiling; error nodes carry most of the attribution; atlas scores match
    the rotation control.

## Numbers
**Unedited model.**
- ALL (N = 20,000, seed 13): 10-class accuracy 86.16 % [85.69, 86.66] (expected ~86 %).
- EVAL: W/Z accuracy 84.23 %; everything resampled (donor) 15.81 %.
- Exactness: all masks = 1 vs clean logits 3.8e-6; all masks = 0 vs donor logits 2.9e-6.

**Dictionaries (on ALL).**

| site | FVU | dead fraction | spliced 10-class acc (this site only) |
|---|---|---|---|
| P0 … P7 | 0.027, 0.033, 0.040, 0.044, 0.051, 0.050, 0.050, 0.045 | 0.3–1.4 % | 86.14–86.21 % |
| C0 / C1 | 0.004 / 0.001 | 59.9 % / 68.0 % | 86.15 % / 86.17 % |
| R0 / R1 | 0.002 / 0.001 | 72.0 % / 92.6 % (151 live of 2,048) | 86.16 % / 86.11 % |
| permuted model P0 / P7 / C0 / C1 / R0 / R1 | 0.028 / 0.082 / 0.032 / 0.032 / 0.062 / 0.076 | – | – |
| rotR1 | 0.002 | 95.4 % | – |

- All 12 sites spliced at once (no error terms): 86.13 % vs 86.16 %. Drop 0.03 points [−0.14, +0.19]; McNemar 123 right → wrong,
  118 wrong → right, p = 0.80. The target (within 1 point) is met.

**Atlas: mean monosemanticity (max conditional-mean R² over the 20 observables), live units.**

| site | dictionary | MLP neurons | raw coords | rotated coords | permuted-model dictionary |
|---|---|---|---|---|---|
| P0 | 0.068 | 0.110 | 0.290 | 0.217 | 0.081 |
| P3 | 0.040 | 0.056 | 0.148 | 0.142 | – |
| P7 | 0.034 | 0.044 | 0.117 | 0.099 | 0.020 |
| C0 | 0.112 | 0.152 | 0.284 | 0.329 | 0.016 |
| C1 | 0.093 | 0.116 | 0.177 | 0.213 | 0.011 |
| R0 | 0.076 | – | 0.184 | 0.272 | 0.008 |
| R1 | 0.151 | – | 0.266 | 0.253 | 0.005 |

- Dictionary minus rotated coordinates: −0.07 to −0.22 at every site, all bootstrap intervals below 0 (Mann-Whitney p for
  "dictionary > rotated" = 1.0).
- Dictionary minus permuted-model dictionary: P7 +0.014 [0.011, 0.016], C0 +0.096 [0.080, 0.112], C1 +0.081, R0 +0.068, R1 +0.146
  [0.125, 0.169] (p ≤ 4e-16); P0 −0.013 [−0.019, −0.006].
- R1 SAE vs the SAE trained on the rotated R1 residual: −0.031 [−0.068, +0.007] (no difference resolved).
- Most frequent best observable: C0 log m_A (183 of 411 live), R0 log m_A (205 of 574), R1 log m_A (57 of 151); P0 tanh d0 (221),
  neutral-hadron flag (163); P7 jet log m_A (172), ΔR (132). The permuted-model dictionaries' class features align with counts
  (charged, photon, multiplicity), not with mass.
- For the class features log m_A and log m give equal R² to two decimals (e.g. R0#2037 0.34 / 0.34): on natural jets the atlas
  does not separate the axis-moment mass from the invariant mass.

**Attribution (DISC, 3,000 jets).**
- D(1) − D(0) = 4.857 per jet; sum of IG scores / (D(1) − D(0)) = 1.001.
- Error nodes: 0.5 % of total |score|; signed share −0.3 %.
- Share of |score|: top 10 features 37.2 %, top 50 68.7 %, top 200 82.9 %, top 1,000 92.4 %.
- By site: R1 34.8 %, R0 27.3 %, C0 11.8 %, C1 9.6 %, P7 4.0 %, P0–P6 1.4–2.1 % each.
- Largest features (score per jet; card summary):
  - R0#2037 +0.370: top-100 jets 97 Wqq; best observable log m_A / log m (R² 0.34).
  - R1#46 +0.307: top-100 jets 94 Tbl, 6 Wqq; direct effect +1.05 on \(\ell_W - \ell_Z\) per unit.
  - R1#584 +0.295: top-100 jets 92 Hbb; best observable max |tanh d0| (0.13); direct effect −1.03.
  - R0#1588 +0.289: top-100 jets 91 Wqq; log m_A / log m (0.47).
  - R1#1043 +0.260: top-100 jets 48 Hcc, 50 Zqq; direct effect −0.57.
- The top-50 set has 49 class-level features (C0 9, R0 16, C1 9, R1 15) and 1 particle feature (P2). The top-200 set has 43
  particle features (23 of them in P7).

**Circuits kept alone (EVAL, 10,000 W/Z jets, donors B).** W/Z accuracy [%] with 95 % CI; F as in P9.

| K features | params (257 each) | main: F | main: W/Z | errlive: W/Z | strict: W/Z |
|---|---|---|---|---|---|
| 0 | 0 | 0.000 | 15.8 | 15.9 | – |
| 5 | 1.3 k | 0.064 | 18.0 | 17.4 | 16.6 |
| 10 | 2.6 k | 0.123 | 22.0 | 20.0 | 19.0 |
| 20 | 5.1 k | 0.253 | 34.6 [33.7, 35.5] | 33.6 | 24.4 |
| 50 | 12.9 k | 0.475 [0.469, 0.481] | 50.0 [49.0, 51.1] | 50.0 | 32.0 |
| 100 | 25.7 k | 0.562 | 56.2 | 55.9 | 33.1 |
| 200 | 51.4 k | 0.674 | 66.3 [65.4, 67.2] | 65.8 | 38.0 |
| 500 | 129 k | 0.719 | 70.4 | 70.7 | 39.9 |
| 1,000 | 257 k | 0.783 | 74.1 [73.2, 75.0] | 74.4 | 40.4 |
| 2,000 | 514 k | 0.864 | 79.0 [78.2, 79.8] | 80.2 | 40.3 |
| 5,000 | 1.29 M | 0.915 | 81.4 [80.6, 82.2] | 83.5 | 41.7 |
| all 9,595 live | 2.47 M | 0.914 | 81.6 [80.8, 82.3] | – | – |

- strict with all features and all errors kept: W/Z 41.9 % [41.0, 42.9], F 0.371.
- McNemar vs the full model (main): K = 50, 3,852 right → wrong, 431 wrong → right (p < 1e-300); K = 1,000, 1,500 / 485
  (p 4e-120); K = 2,000, 912 / 388 (p 5e-49).
- Random feature sets (EVAL[::10], 1,000 jets, 100 sets each): K = 20 circuit 34.7 % vs random 16.3 % (max 17.6 %); K = 50 49.8 % vs
  16.3 % (max 18.4 %); K = 200 65.9 % vs 16.7 % (max 19.7 %). Permutation p = 0.0099 (floor) for accuracy and F at each K.
- Comparison with P9 (head / 32-unit-MLP-group literal circuits, same corruption, val files; one P9 node ≈ 8,200 parameters):
  8 nodes (≈ 66 k parameters) 27.2 %, 16 nodes (≈ 132 k) 31.4 %, 64 nodes (≈ 0.53 M) 50.8 %, 128 nodes (≈ 1.05 M) 65.3 %. P9 edge
  circuits: 1,075 edges 54.7 %; P9v trained mask at 10 % of edges 63 %.
  - These parameter counts leave out what the main variant keeps live: 10 attention blocks (≈ 0.67 M parameters) and 10 skip maps
    (0.16 M). Counting them, K = 50 uses ≈ 0.84 M parameters, where P9's node curve gives about 55–60 %.
- Hypothesis checks: W/Z at K = 50 is 50.0 %; bootstrap p(acc ≤ 0.75) = 1.0, so H1's circuit-size prediction is excluded. The
  smallest K reaching 75 % is 2,000 (main and errlive); strict never reaches it. The main variant exceeds the 63 % P9v ceiling from
  K = 200, so H0's "cap near 63 %" is excluded for main; strict caps at 42 %. H0's "error nodes carry most of the attribution" is
  excluded (0.5 %).

**10-class read (ALL, 20,000 jets, uniform random donors).**

| circuit | 10-class acc | W/Z binary | calls equal to the donor's class |
|---|---|---|---|
| nothing kept | 10.0 % | 49.6 % | 86.5 % |
| K = 50 | 12.0 % [11.5, 12.4] | 62.2 % | 79.2 % |
| K = 200 | 30.4 % | 65.8 % | 46.2 % |
| K = 1,000 | 51.7 % | 73.8 % | 32.1 % |
| K = 5,000 | 78.8 % [78.2, 79.3] | 81.8 % | 11.4 % |

Confusion matrices are in the json (`circuits.ten_class`).

**m_A feature: steering and ablation.**
- Scan: 834 of 1,233 class-level features BH-significant for correlation with log m_A. Top: C0#831 (class block 1 MLP transcoder),
  r = −0.757 (falls as m_A rises), mean activation 4.45, s.d. 2.18. Next: C0#340 −0.698, C0#833 −0.670, C0#219 −0.633, C0#656
  +0.615.
- Calibration on natural pairs: β = −7.29 [−7.58, −6.99] per unit log m_A; 5 % m_A = Δa −0.36 (0.16 s.d.).
- φ + 0.06 on the 5,000 EVAL W jets: mean Δlog m_A +0.093; W→Z 9.72 % → 30.82 %, flow +21.1 points [19.9, 22.3].
- Steering C0#831 by 5 % m_A: W→Z 9.70 %, flow −0.02 points [−0.10, +0.06]; ratio to the φ flow −0.001 [−0.005, +0.003].
  McNemar on W calls: 8 right → wrong, 0 wrong → right (p 0.008).
- Steering by each jet's own φ-edit Δlog m_A: W→Z 9.72 %, ratio 0.000 [−0.004, +0.004].
- Ablating C0#831 to its mean: φ flow 21.28 points [20.06, 22.50] vs 21.10 intact; difference +0.18 [−0.12, +0.50].
- 20 random C0 features: steering flows −0.02 to +0.12 points, ablated φ flows 21.06–21.26 points; permutation p 1.0
  for both.
- H1's steering prediction (≥ 0.6 of the φ flow) is excluded by the interval [−0.005, +0.003].

## Reading
- The dictionaries reconstruct every MLP and both class-token residuals with FVU ≤ 0.052. Splicing all 12 in leaves the 10-class
  accuracy unchanged within 0.2 points. The class token is low-rank for these dictionaries: 151 of 2,048 R1 latents fire on 20,000
  jets.
- Under the W↔Z pT-matched resample, the W/Z accuracy of the top-K feature set rises smoothly with K: 34.6 % at 20, 50.0 % at 50,
  66.3 % at 200, 74.1 % at 1,000, 79.0 % at 2,000, against 16–20 % for random sets of the same size. Error terms carry 0.5 % of the
  attribution and keeping them live changes the curve by at most 2 points below K = 5,000.
- The small circuits are almost entirely class-level features (R0, R1, C0, C1) that sit just upstream of the logits. Their top
  activating jets are mostly one class (e.g. R0#2037: 97 of 100 Wqq) and their best observables are log m_A / log m.
- The main-variant curve depends on the attention layers and skip terms being computed on the real jet. With skip terms and
  particle-block attention also resampled, the accuracy stays at 32–42 % for K = 50–5,000, and 42 % with every feature and error
  term kept. Under the user's literal definition (only the circuit's units computed from the jet), these feature circuits therefore
  stay below P9v's 63 %.
- Per parameter of the kept units, the main-variant feature circuits reach higher W/Z accuracy than P9's head/MLP-group circuits
  (50 % at 13 k vs 27 % at 66 k). Once the live attention and skip parameters are counted, the two are within a few points at
  equal size.
- By the conditional-mean R² score, the dictionary features are less tied to any single one of the 20 observables than the
  network's own neurons and than raw or rotated coordinates, at every site. They score above the weight-permuted model's
  dictionaries at five of six sites. The score favours dense units: a sparse feature is zero on most jets, which caps its R².
  The rotation-trained R1 SAE gives the same scores as the R1 SAE. The "more monosemantic than rotated-basis controls"
  prediction is not supported by this metric.
- The class-level feature most correlated with m_A (C0#831, r −0.76) has no measurable effect on the W/Z calls. This holds when
  steered by a calibrated 5 % m_A change, by each jet's own φ-edit m_A change, or when ablated during the φ + 0.06 edit. With
  834 m_A-correlated features at the class level, the m_A response that P1–P8 measured is spread over many features, not
  carried by this one.

## Issues and caveats
- **Main vs literal definition.** The main variant keeps all attention and the transcoders' linear skip maps live. This is the
  standard transcoder-circuit convention, but not the user's literal definition. The strict variant is closer to the literal
  definition, but it resamples attention as a whole, not as units, so a joint feature + head circuit was not tested.
  Class-block attention is live in every variant.
- **Parameter comparison.** P9's parameters per node (8,224) is an approximation. Its z-coordinate and pair-bias nodes are
  smaller, so the P9 x-positions in the figure are upper bounds.
- **Steering size.** The calibrated 5 % step is 0.16 s.d. of C0#831; the φ-matched step is about 0.31 s.d. Larger steps, and
  joint steering of the top m_A-correlated features (or of a learned m_A direction, as in P8/P8v), were not run. Only class-level
  features were scanned for steering; particle-level m_A features (P7: 172 features with jet log m_A as best observable) were not
  steered.
- **Monosemanticity metric.** The conditional-mean R² with 20 bins cannot tell m_A from m on natural jets (their R² match). It
  also penalises sparse units. A metric conditioned on firing (R² among active tokens) was not computed.
- **Dead latents.** 60–93 % of class-level latents never fire, despite AuxK. The class token has few active directions, so the
  effective class-level dictionary is 151–574 features per site.
- **Random sets** were drawn from live features only, with matched per-site counts. They were evaluated on 1,000 jets for
  K = 20, 50, 200 only.
- **ParT_kin replication** not run (time).
- **Data reading.** Training files are read whole and subsampled at random rows; each 250,000-jet pool is shuffled before
  batching.

## Explainer entry (for site/explain.py)
- **Question.** P9 found no small circuit of attention heads and MLP-unit groups that reproduces ParT_full's W-vs-Z calls. Do
  learned sparse features give one?
- **Background.** A transcoder replaces an MLP by a sparse dictionary,
  \(\hat y = \mathrm{TopK}(u W_{enc}^T + b_{enc}) W_{dec} + b_{dec} + u W_{skip}^T\). An SAE does the same for a residual
  vector. The features are the dictionary latents. Claim C8 says no circuit with up to 10 % of P9's edges reproduces the W/Z
  calls.
- **Method.**
  - Train transcoders on the 8 particle-block MLPs and the 2 class-block MLPs, and SAEs on the class token after each class
    block, on 1M training jets.
  - Write each site as features + error, \(y = (aW_{dec} + b_{dec} + \text{skip} + \varepsilon)s\), so the network is unchanged.
  - Score each feature by integrated gradients of \(D = \sum_j s_j(\ell_W - \ell_Z)\) over masks that interpolate from a
    pT-matched opposite-class donor to the jet.
  - Keep the top-K features and replace every other feature and error term by the donor's values.
- **Result (figure p13_feature_circuit.png).** W/Z accuracy rises from 16 % (nothing kept) to 50 % at 50 features, 74 % at 1,000
  and 79 % at 2,000 (model 84 %; random sets 16–20 %). If attention outputs and skip terms are also taken from the donor, it
  stays below 42 %. The feature most correlated with \(m_A = \sqrt{\sum p_T \sum p_T\Delta R^2}\) does not change the calls when
  steered or removed.
- **Checks.**
  - Spliced dictionaries keep the 10-class accuracy (86.13 % vs 86.16 %).
  - All masks on reproduces the model to 4e-6; all off reproduces the donor.
  - IG completeness 1.001.
  - 100 random feature sets per size (p = 0.0099).
  - Atlas against neuron, raw, rotated, rotation-trained and weight-permuted controls.
- **Why it matters.** Changing the unit from heads to features does not give a short circuit for the W/Z decision under the
  user's definition. The decision passes through many class-level features and needs the attention layers computed on the real
  jet. C8 holds at the feature level, with this qualification.
