# P40: class-token MLP neuron atlas of ParT with random-rotation and untrained controls; new claim C19 (agent report, 2026-10-20)

## Conclusion
ParT has two class-token blocks, each with a 512-neuron feed-forward layer (MLP). As individual readable units, these neurons are not distinguished from random rotations of the same 512-dimensional space:
- About half of the neurons and about half of the rotated directions respond selectively to one physics quantity or one class (ParT_full 49 % vs 49–50 %). In ParT_kin the neurons are less selective than raw rotated directions (31 % vs 51 %).
- Steering one neuron by 2 standard deviations moves its predicted class the right way in 53 % of 300 neurons, against 47–52 % for the control directions. For most neurons the effect is close to zero.
- Among the neurons whose steering does change the class score visibly, 75 % (ParT_full) and 86 % (ParT_kin) move it the way their label predicts. The rotated controls give 44–47 %.
- The 18 neurons that track the jet mass together carry 17–30 % of the network's W-score response to a real 5 % mass increase. Random directions carry about 0 %.

The thesis as stated (neurons at least twice as readable as rotated directions, steering correct in more than 70 % of neurons) is not supported. The neuron basis is useful only for a minority of high-impact units. Still uncertain:
- the subset analysis was not pre-registered;
- the "clean" criterion saturates in trained networks (any trained unit separates some class), so it cannot separate the two bases.

## Runs
| job | command | GPU | wall | output |
|---|---|---|---|---|
| `anrunw-ai-c-p40-run` (3rd submission) | `python -m interp.p40_neuron_atlas --smoke && python -m interp.p40_neuron_atlas` | RTX 3090 | 80 s smoke + 702 s full | `results/part/p40_neuron_atlas.json` (+ `_smoke`), `logs/c-p40-run.log` |
| `anrunw-ai-c-p40-plot` | `python -m interp.p40_plot` (CPU, job.sh) | – | 116 s | `results/part/p40_neuron_atlas.png` |

Debug notes:
- **Attempt 1** failed: torch 1.13 in the image has no `with_kwargs` for forward pre-hooks. Fixed by wrapping `cls_blocks[0].forward`.
- **Attempt 2** was stopped by me after its smoke step. The smoke showed that the pre-registered "clean" criterion saturates. I then:
  - added a pre-defined family-selectivity measure;
  - drew the 300 steered units at random instead of taking the 300 highest scores, which were all t→bℓν units;
  - enlarged the reference sample to the 1,000-per-class P1 sample.

**Sample file:** the plan named `_126`, but test_20M has only `_100`–`_119`. I used `<sample>_110.root`, one of the least-used files.

## Unedited accuracy (reference check)
| model | P1 sample (rows 3,000–4,000 of each `_100` file, 10,000 jets) | labelling half (10,000) | held-out half (10,000) |
|---|---|---|---|
| ParT_full | 85.93 % (expected ≈ 85.9) | 85.93 % | 86.11 % |
| ParT_kin | 74.47 % (expected ≈ 74.5) | 73.38 % | 73.80 % |

- Untrained nets: 10.0 % (full), 10.1 % (kin). Weight-permuted nets: 10.2 % (full), 9.5 % (kin).
- The class-token path was re-run from cached class-block inputs, and the result was checked against the full forward pass: max |Δlogit| = 0.

## Procedure
**Sample.** JetClass test_20M `<sample>_110.root`:
- 2,000 random rows per class (numpy `default_rng(4040)`; never first-k), N = 20,000;
- every jet is rebuilt through `part_core.build`;
- within each class a random half is the *labelling half* and the other the *held-out half* (10,000 each); jet order is shuffled.

**Units.** In each class block b1, b2 the class token passes through `fc1` (128 → 512), GELU, a 512-d LayerNorm and `fc2` (512 → 128). A "neuron" is one of the 512 coordinates:
- *pre-GELU*: the `fc1` output;
- *post-GELU*: the GELU output, before the LayerNorm.

That gives 2 × 512 = 1,024 units per site type.

**Bases compared.**
1. The neurons themselves.
2. `rot raw`: the activations multiplied by a Haar-random orthogonal 512 × 512 matrix Q (10 seeds per block). Undoing Q before `fc2` leaves the network unchanged, so rotated coordinates describe the same computation in another basis.
3. `rot std`: the same rotation applied after standardising each neuron on the labelling half. P15v showed that the comparison can flip between 2 and 3.
4. Controls:
   - an *untrained* ParT of each feature set: every weight tensor redrawn from a Gaussian with the trained tensor's mean and s.d. (seed 0);
   - a *weight-permuted* ParT: each tensor's entries shuffled (seed 5).

**(1) Read.**
- *Labels.* 22 labels per unit:
  - 12 jet observables: N (particle count); log pT; log m (invariant mass); log m_A, the axis-moment mass (m_A = √(M0·M2) with M0 = Σ pT and M2 = Σ pT r², r the distance from the stored axis); τ21 = τ2/τ1 and τ32 = τ3/τ2 (N-subjettiness ratios from the file); D2 = e3/e2³ (energy-correlation ratio, β = 1, all particles); the leading-particle pT fraction; the numbers of charged particles, neutral particles and leptons; and the largest |d0| (transverse impact parameter) among charged particles;
  - 10 classes, each one-vs-rest.
- *Score.* S = Spearman ρ for an observable and 2(AUC − 0.5) for a class, so both lie in [−1, 1].
- *Labelling.* A unit's label is the argmax |S| on the labelling half. On the held-out half, v = (sign on the labelling half) × S_heldout[label].
- *Significance.* p-values come from the held-out half (t-test for ρ, normal approximation to Mann–Whitney for AUC), with Benjamini–Hochberg (BH) over 512 × 22 per site.
- *Clean:* v ≥ thr and BH q < 0.05. thr is the 95th percentile of v over the untrained net's 1,024 units, so the untrained net passes 5 %:
  - ParT_full: 0.276 (post), 0.315 (pre);
  - ParT_kin: 0.359 (post), 0.413 (pre).
- *Selective* (added; defined before the full run): clean, and the best label family leads the second family by at least 0.1 in held-out |S|. The best family must be the label's own. The families are:
  - count (N, n_charged, n_neutral);
  - mass (m, m_A; correlation 0.99 on natural jets);
  - pT; shape (τ21, τ32, D2); leading fraction; leptons; d0;
  - each class alone.
- *Interval.* Wilson 95 % over the 1,024 units. Against rotations: rank among the 10 seeds (empirical p).

**(2) Write** (trained models; post-GELU units; 300 randomly chosen clean units per model; held-out half).
- *Target class c\* and predicted sign s:*
  - a class-labelled unit has c\* = its class and s = the sign of its score;
  - an observable-labelled unit (observable o) has c\* = argmax_c sign(ρ)·(AUC_c(o) − 0.5), the class in which o is most shifted in the direction the unit follows, and s = +1.
- *Steering:* add k × sd to the unit (k = ±1, ±2, ±4; sd is the unit's s.d. on the labelling half).
- *Mean ablation:* set the unit to its labelling-half mean.
- *Metrics:*
  - Δℓ_c\*, where ℓ_c = log p_c − log(1 − p_c) is the class log-odds, taken per jet against the unedited pass;
  - specificity = Δℓ_c\* − (mean Δℓ of the other 9 classes);
  - 95 % bootstrap over jets (1,000 resamples);
  - exact McNemar test on "called c\*" (steering) or on "called correctly" among class-c\* jets (ablation), with BH over units.
- *Direction-correct:* s·Δℓ_c\*(+2 sd) > 0 with its interval excluding 0, and s·Δℓ_c\*(−2 sd) < 0.
- *Controls per unit:*
  - **rotated**: the same-index column of Q₀ in the same block, steered by k × its own projection s.d.;
  - **matched**: the unit of the same block with the closest s.d. whose label and target class both differ, steered by k × its own s.d. and read on the first unit's c\*.

**(3) Calibration.**
- The real mass change is the P1 matched dilation: every particle's Δη, Δφ × 1.05 at fixed pT and particle mass, giving median Δlog m = +0.048.
- Each mass-labelled clean post-GELU unit (label log m or log m_A) is *patched*: set per jet to its value on the dilated jet, with everything else from the unedited jet.
- R = mean Δℓ_c\*(patch) / mean Δℓ_c\*(dilation), i.e. the share of the network's response to the real mass change that passes through this unit. 95 % bootstrap ratio.
- Joint patch of all mass units against the same number of random rotated directions (10 draws).

**(4) Neuron circuits.** For each class c, the post-GELU units labelled c are:
- ablated to their means (all other units intact); or
- kept alone, with all other 1,024 − k post-GELU units at their means.

The particle blocks and attention are intact in both cases. Each variant is compared with 20 random unit sets of the same per-block size (empirical p).

This is a hidden-layer-restricted circuit, not a literal circuit in the user's sense. A strictly literal version that also mean-ablates the 16 class-attention heads gives one constant call for every jet: the class token receives jet information only through class attention.

## Numbers
### (1) Atlas: share of units that are clean / selective (post-GELU, 1,024 units, threshold from the untrained net)
| | clean | selective [95 % Wilson] |
|---|---|---|
| ParT_full neurons | 0.868 | 0.488 [0.458, 0.519] |
| ParT_full rot raw (10 seeds) | 1.000 (0.999–1.000) | 0.492 (max 0.519); p(neuron ≥ rot) 0.73 |
| ParT_full rot std | 0.985 (0.974–0.994) | 0.501 (max 0.530); p 0.73 |
| ParT_full untrained / permuted | 0.051 / 0.024 | 0.004 / 0.003 |
| ParT_kin neurons | 0.574 | 0.312 [0.285, 0.342] |
| ParT_kin rot raw | 0.983 (0.977–0.986) | 0.512 (max 0.524) |
| ParT_kin rot std | 0.728 (0.714–0.746) | 0.347 (max 0.368) |
| ParT_kin untrained / permuted | 0.051 / 0.023 | 0.005 / 0.008 |

**Pre-GELU.**
- ParT_full: clean 0.995 vs 1.000 / 1.000; selective 0.517 vs 0.490 (rot raw, p 0.18) / 0.535 (rot std, p 0.91).
- ParT_kin: selective 0.455 vs 0.452 / 0.462.

**Selective with a family gap ≥ 0.2** (post-GELU, neurons vs rot raw vs rot std):
- ParT_full: 0.256 vs 0.212 vs 0.228;
- ParT_kin: 0.149 vs 0.271 vs 0.119.

**Fixed thresholds** (post-GELU clean, ParT_full): 0.952 / 0.839 / 0.501 at thr 0.2 / 0.3 / 0.5, against rot std 0.999 / 0.975 / 0.649.

**Labels of clean post-GELU neurons, ParT_full:**
- t→bℓν 243, H→bb 126, t→bqq′ 110, H→ℓνqq′ 100, QCD 75, H→cc 59, H→gg 56, W 46, H→4q 43, Z 7;
- m_A 15, m 3, N 6.

The jet observables label only 24 of 889 clean units; almost all are labelled by a class.

**Named units** (ParT_full unless noted; class means of the post-GELU activation on held-out jets):
- **b2n153**: fires for t→bℓν (mean 9.9; ≤ 0.9 in every other class); v 0.998.
- **b2n249**: fires for H→ℓνqq′ (13.3 vs ≤ 1.2); v 0.984. **b1n138** does the same in block 1 (5.9 vs ≤ 0.2).
- **b2n419, b2n144**: fire for both lepton classes (H→ℓνqq′ 9–11, t→bℓν 18–23, others < 1). These are "lepton present" units, consistent with P34.
- **b2n104**: lower for t→bℓν, t→bqq′ and H→ℓνqq′ (15.8 / 23.8 / 24.2 vs 26–32).
- **b1n92, b1n108**: m_A-labelled. They fall as m_A rises (−0.08 to −0.09 s.d. under the 5 % dilation). Steering by +2 s.d. raises the W log-odds by +0.25 and +0.42 (McNemar p 3e-25 and 2e-4). These are W-side mass units in the class MLP.
- **b2n473** (t→bℓν): steering by +2 s.d. moves ℓ_t→bℓν by +1.65 [CI excludes 0]; −2 s.d. moves it by −1.65.
- **ParT_kin b1n106**: fires for t→bqq′ (15.2 vs ≤ 2.7). **ParT_kin b2n75**: low for t→bqq′ (3.9 vs 12–16).

Several high-v units have s.d. ≈ 0.05: they are nearly silent and differ only slightly between classes, e.g. b2n408, b2n252 (labelled t→bℓν). They read cleanly but cannot act downstream.

### (2) Write (300 random clean post-GELU units per model)
| | direction-correct [Wilson] | median s·Δℓ_c\*(+2 sd) | median abs Δℓ (+2 sd) | McNemar BH < 0.05 |
|---|---|---|---|---|
| ParT_full neurons | 0.533 [0.477, 0.589] | +7.6e-5 | 0.0011 | 3.3 % |
| ParT_full rotated | 0.470 [0.414, 0.526] | −2.3e-3 | 0.021 | 4.7 % |
| ParT_full matched | 0.523 [0.467, 0.579] | +4.3e-5 | 0.0013 | 3.7 % |
| ParT_kin neurons | 0.537 [0.480, 0.592] | +3.4e-5 | 0.0030 | 7.0 % |
| ParT_kin rotated | 0.523 [0.467, 0.579] | +1.8e-3 | 0.019 | 18.3 % |
| ParT_kin matched | 0.453 [0.398, 0.510] | −5.8e-6 | 0.0034 | 11.7 % |

**Paired comparison** of s·Δℓ(+2 sd) per unit, neurons against controls:
- against rotated: Wilcoxon p 0.10 (full), 0.008 (kin);
- against matched: Wilcoxon p 0.54 (full), 2e-4 (kin).

**Calls changed.** At +4 s.d. the median share of changed calls is 0 % (full) and 0.14 % (kin). The largest single-unit value at +2 s.d. is 3.3 % (full).

**Ablation of one unit:**
- the median signed Δℓ on class-c\* jets is about 0;
- 54 % of units lower their target class (rotated controls 47–51 %);
- 0.3–1 % of units are significant by McNemar after BH;
- at most 1.1 % of calls change.

**Post-hoc subset:** units whose +2 s.d. step changes |Δℓ_c\*| by ≥ 0.05.
| model | neurons | rotated | matched | neurons vs chance |
|---|---|---|---|---|
| ParT_full | 18/24 = 0.75 | 24/54 = 0.44 | 8/24 = 0.33 | binomial p 0.023 |
| ParT_kin | 31/36 = 0.86 | 21/45 = 0.47 | 21/36 = 0.58 | binomial p 1.3e-5 |

The rotated directions have a larger raw step than the neurons (median s.d. 2.02 vs 0.41 in ParT_full), because a rotated direction mixes in high-variance neurons.

### (3) Calibration: 5 % angular dilation
**Response of the intact network.**
- Δℓ_W: −0.144 (full), −0.170 (kin).
- Δℓ for t→bqq′: +0.30 (full), +0.35 (kin).
- Calls changed: 5.5 % (full), 9.9 % (kin).

**Single mass-labelled units, R for the W log-odds:**
- ParT_full: b1n92 0.117 [0.106, 0.129]; b1n108 0.089; b1n253 0.052; b1n492 0.028; the other 14 ≤ 0.006;
- ParT_kin: b1n8 0.054; b1n120 0.039; b1n449 0.017; b1n306 0.015; b2n31 0.011.

**Joint patch of all 18 mass units vs the same number of random rotated directions (10 draws):**
| model | R_W, mass units | R_W, random directions |
|---|---|---|
| ParT_full | 0.303 [0.278, 0.335] | −0.03 (−0.14 to +0.06) |
| ParT_kin | 0.173 [0.160, 0.191] | −0.01 (−0.04 to +0.01) |

For the other classes R is small or of opposite sign: QCD −0.09 / −0.17, Z −0.11 / −0.22, t→bqq′ +0.02 / −0.05 (full / kin).

**Steering vs the real change.** The mass units shift by only about 0.08 s.d. under the dilation. One +2 s.d. step of b1n92 (Δℓ_W +0.25) is therefore larger than the whole dilation response carried by that unit (−0.017).

### (4) Hidden-layer circuits
**All 1,024 post-GELU units at their mean:**
- ParT_full 10-class accuracy 86.11 % [85.4, 86.8] → 82.31 % [81.6, 83.1];
- the W recall falls 0.793 → 0.445; the other recalls change by −0.11 to +0.05;
- ParT_kin 73.80 % → 70.94 %.

**Per class (ParT_full):**
- *Ablating that class's labelled units* changes 10-class accuracy by at most 1.5 points (H→bb units: 86.1 → 84.6 %, while H→bb recall rises 0.916 → 0.951). Only QCD (0.777 → 0.755, p 0.048 vs random sets) and Z (p 0.048, 7 units, recall 0.713 → 0.712) fall below the random-set range.
- *Keeping only that class's units:*
  - W, 46 units: W recall 0.566 against 0.486 for random sets (max 0.640, p 0.19);
  - H→ℓνqq′: 0.985 (random 0.972, p 0.095);
  - no class exceeds every random set.
- **ParT_kin:**
  - keeping only the QCD-labelled units: QCD recall 0.735 against random 0.667 (max 0.729, p 0.048);
  - ablating the t→bqq′ units: t→bqq′ recall 0.904 → 0.866 (random 0.902, p 0.048).
- Full 10 × 10 confusions are in the JSON (`models.<m>.circuits`).

## Reading
**H0 / H1.**
- H1 (thesis C19): the neuron basis is privileged.
  - clean or selective fraction ≥ 2 × rotations;
  - steering direction-correct in > 70 % of labelled units, more than in the controls.
- H0 (alternative): the neurons are no more readable than rotated directions; steering effects are not class-specific.

**Prediction 1 (clean fraction ≥ 2 × rotations).**
- *Clean* fraction: neurons ≤ rotations in both models (0.87 vs 0.99–1.00; 0.57 vs 0.73–0.98).
- *Selective* fraction:
  - ParT_full: neurons within the rotation-seed range (0.488 vs 0.49–0.52 / 0.50–0.53);
  - ParT_kin: neurons below raw rotations (0.312 vs 0.51) and below standardised rotations (0.347).
- This prediction of H1 is excluded on this sample; the atlas measurements agree with H0.

**Prediction 2 (steering).**
- Direction-correct in 53 % of units, below 70 %.
- The neurons do not differ from the rotated control in ParT_full (Wilcoxon p 0.10).
- In ParT_kin they differ from both controls (p 0.008, 2e-4), but the effect sizes are of order 1e-4 log-odds.
- Most clean units have little downstream effect at ±2 s.d. The 512-d LayerNorm after GELU and small `fc2` weights mean that a low-variance unit's step barely changes the class token. This is measured, not inferred: median |Δℓ| is 0.001–0.003.

**Minority subset.** The subset of units with a visible effect (8–12 % of the steered units) does move its labelled class in the predicted direction (75 % / 86 % vs 44–47 % rotated). This subset is post hoc: it was defined after the run, and conditions on the outcome magnitude.

**Mass.**
- A handful of block-1 units that fall with m_A carry 17–30 % of the W-score response to a real mass change. Random directions carry ≈ 0.
- Their labels are causally meaningful at the real-change scale.
- The rest of the response does not pass through the class-MLP hidden layer: P2 placed 93–101 % of the mass read-out change in the per-particle route, which reaches the class token through attention.

**Physics meaning.** ParT's class-token MLPs mostly compute class-identity units (lepton present, top-like, H→bb-like), not physics observables.

**Size of the role.** With the whole hidden layer at its mean, ParT_full loses 3.8 points of accuracy, almost all from W recall.

The class-MLP hidden layer thus carries:
- a W-versus-rest correction that partly reads m_A;
- a small number of strong class-specific units.

## Issues / caveats
- **Saturating criterion.** The "clean" criterion as planned (max |S| above the untrained 95th percentile) is reached by 87–100 % of trained units in every basis. In a trained classifier's last layers nearly every direction separates some class, so this measure does not test basis alignment. The selectivity measure is the informative read.
- **Rotation controls disagree.** The raw and standardised rotations disagree in ParT_kin (0.51 vs 0.35 selective), as P15v found for ParticleNet. A basis statement depends on this choice. The conclusion "neurons ≤ rotations" holds under both.
- **Steered-unit selection.** The 300 steered units are a random subset of the clean units (889 / 588). The 70 % criterion is about this population.
- **Dose scale.** The steering dose is per-unit s.d. The rotated control is steered by its own s.d., which is about 5 × larger in ParT_full. A same-absolute-step rotated control was not run.
- **Calibration target.** The calibration uses each unit's target class. For all 18 mass units that class is W or t→bqq′.
- **Circuit definition.** The hidden-layer circuits keep attention and the particle blocks intact; this is not a literal circuit (see Procedure).
- **File change.** `_126` does not exist in test_20M; `_110` was used.

## Thesis effect
**Weakens C19 as stated.** On the planned measures the neuron basis is not privileged: readability equals or falls below random rotations, and steering is not more class-specific in > 70 % of units.

**Refinement (post hoc, to verify):**
- a minority of high-impact units are interpretable and causal: lepton-class units; W-side m_A units that carry 17–30 % of the real mass response;
- for later circuit work, a neuron-level analysis restricted to high-impact units is supported;
- a full atlas needs feature-level methods (P13 transcoders).

**Suggested verification (P40v):**
- fresh file;
- pre-register the |Δℓ| ≥ 0.05 subset and the rotated control at equal absolute step;
- n98-style W/Z read of b1n92 / b1n108 under the φ + 0.06 shift (axis-relative test, C3).

## Explainer entry (for site/explain.py)
```python
P40_EXPLAINER = [
 ('Question', 'ParT has two class-token blocks, each with a 512-neuron feed-forward layer. Are these neurons individually readable (each responds to one physics quantity or one class) and causally meaningful (pushing a neuron moves the class it is labelled with), more than arbitrary directions of the same 512-dimensional space?', None),
 ('Background', 'A random orthogonal rotation $Q$ of a hidden layer, undone before the next weight matrix, leaves the network unchanged; so if the neuron coordinates are special, they must beat the coordinates of $Q h$. Neuron $i$ of block $b$ is $h_i = \\mathrm{GELU}(W_1 x + b_1)_i$; the next steps are a LayerNorm over the 512 values and $W_2$. Score of a unit against an observable: Spearman $\\rho$; against a class: $2(\\mathrm{AUC}-\\tfrac12)$.', None),
 ('Method', '20,000 fresh JetClass test jets (2,000 per class, file _110, seed 4040), half for labelling and half held out. Each unit is labelled with the best of 22 labels (12 observables such as $N$, $\\log m$, $\\log m_A$ with $m_A=\\sqrt{M_0M_2}$, $\\tau_{21}$, $D_2$; 10 classes) on the labelling half and scored on the held-out half. A unit is clean if its held-out score exceeds the 95th percentile of an untrained network, and selective if its label family also leads the next family by 0.1. Same analysis for 10 random rotations (raw and standardised), an untrained and a weight-permuted ParT. Steering: add $\\pm1,2,4$ s.d. to one unit and measure the change of the target class log-odds $\\ell_c=\\log\\frac{p_c}{1-p_c}$. Calibration: patch mass-labelled units to their values on jets whose particle angles are scaled by 1.05 (a real 5 % mass increase) and compute $R=\\Delta\\ell(\\text{patch})/\\Delta\\ell(\\text{dilation})$.', None),
 ('Result', 'About half of the neurons are selective (ParT_full 0.49), the same as rotated directions (0.49–0.50); in ParT_kin the neurons are less selective than raw rotations (0.31 vs 0.51). Steering one neuron by 2 s.d. moves its class the predicted way in 53 % of 300 neurons (rotated control 47–52 %), and usually by less than 0.003 in log-odds. Among the units with a visible effect, 75 % (full) and 86 % (kin) move the predicted way, against 44–47 % for rotated directions. 18 units that fall with $m_A$ carry $R=0.30$ (full) and 0.17 (kin) of the W-score response to the real mass increase; random directions carry about 0.', 'results/part/p40_neuron_atlas.png'),
 ('Checks', 'Unedited accuracy 85.9 % / 74.5 % on the P1 sample. Re-running the class blocks from cached inputs reproduces the logits exactly. Untrained and weight-permuted nets: 2–5 % clean, under 1 % selective. Setting all 1,024 hidden units to their mean lowers ParT_full accuracy from 86.1 % to 82.3 %, mostly through W recall (0.79 to 0.45).', None),
 ('Why it matters', 'The neuron basis of ParT\'s class MLPs is not privileged as a whole, so a neuron-by-neuron atlas is not a complete description; feature-level methods (transcoders) remain needed. A small set of high-impact units (lepton-class units and W-side mass units) can still be studied one at a time.', None),
]
```

## Glossary additions (explain.GLOSSARY)
- **Random-rotation control (basis control):** the same hidden layer read in the coordinates of a random orthogonal rotation of its activations. The rotation, undone before the next layer, changes nothing in the network. If neurons are special, they must be more readable than these rotated directions. 'raw' rotates the activations as they are; 'std' first standardises each neuron.
- **Clean / selective unit (P40):**
  - clean: the unit's best label (one of 12 observables or 10 classes) scores above the 95th percentile of an untrained network on held-out jets;
  - selective: clean, and that label's family (count, mass, pT, shape, leading fraction, lepton, d0, or one class) leads the second family by at least 0.1.
- **Steering (unit):** adding a fixed multiple of a unit's standard deviation to its activation for every jet, then reading the change in the class log-odds.
- **Patching to a real edit:** setting a unit to the value it takes on the edited jet while everything else stays at the unedited jet's values. R is the share of the network's response to the edit that passes through that unit.

## Files
- Scripts: `interp/p40_neuron_atlas.py`, `interp/p40_plot.py`.
- Results:
  - `results/part/p40_neuron_atlas.json` (+ `_smoke.json`);
  - `results/part/p40_neuron_atlas.png`.
- Logs: `logs/c-p40-run.log`, `logs/c-p40-plot.log`.
