# H56 verify: fresh-sample rerun of the seed tests for the axis direction (D4m) and count (D7, H26) results, with larger untrained bands

Job `d-h56-verify` (GPU, 3,123 s, exit 0; smoke stage in the same job passed first). Script `interp/h56_verify.py`. Results `results/physics/h56_verify.json` (smoke: `h56_verify_smoke.json`), log `logs/d-h56-verify.log`. The extra EFN/PFN epoch-0 weights the job drew are in `/j-jepa-vol/aaron-interp/results/physics/h56v_ef_init/` on the volume.

## Conclusion

On new jets, with three times as many untrained networks to compare against, the second training seed still gives the same result as the first. The direction part of the axis response is no larger than what untrained networks of the same architecture show. This holds in all 82 trained-network tests (smallest uncorrected p = 0.08).

Two statements in H56 need correcting:
- **The QG PFN cells.** H39 flagged these as near the edge of the untrained band, and H56 said they "move to 0 with seed 1". On new jets the seed-0 network is itself near 0 (+0.17 pp, against +1.16 pp on the H56 jets). So that flag came from the particular jet sample, not from the training seed.
- **The Herwig–Pythia shift.** The shift in the particle-count response between Herwig- and Pythia-trained networks is about 4–5 times the seed difference for ParT, not 8 times.

Two things remain open:
- The untrained bands are wide (SD 0.3–4.5 pp), so a learned use of the axis direction smaller than about two band widths cannot be excluded.
- The jet-bootstrap intervals at fixed signal efficiency understate how much D changes from one jet sample to another.

## What was checked in the code

- **Axis edit.** Recomputing the inputs about the WTA (winner-take-all) axis or the hardest-particle (HP) axis was compared with shifting the stored arrays by the same offset. They agree exactly: max |Δlogit| = 0 on both new samples (`max_abs_axis_minus_TN`). The torch port of EFN/PFN matches numpy energyflow to ≤ 4.8 × 10⁻⁶.
- **Sampling.** Rows are drawn at random within each class (`data_top.sample` / `data_qg.sample` use `rng.choice` per class). There is no first-k selection.
- **Provisional checkpoints.** The provisional seed-1 ParT checkpoint (`s1p`, epoch 16) is paired with the seed-0 checkpoint at the same epoch (`s0e`, `net_epoch-16`), which is correct. The count step used the final QG ParT s1 checkpoint (best epoch). For QG ParT s1, `predict_output` now exists, so this verify uses the final checkpoint in both steps.
- **Count edits.** The count step reproduces H24v on the H56 sample: ParT_kin +0.0079, ParticleNet_kin +0.0042. The EFN split response is exactly 0.
  - In the verify sample the longest jet after five splits has 116 particles, so nothing is truncated at the 180-particle padding.
- **Statistics: bugs.** No bug was found that changes the numbers.
- **Statistics: Holm family (H56).** The 283 tests include about 100 McNemar and shuffled-minus-s0 tests on constant (degenerate) networks, with p ≈ 0. These enlarge the Holm family without testing anything. They do not change any conclusion, because no vs-init test reached p < 0.047 even before correction.
- **Wording in H56.** "Inside the band in every cell" and "the D4m prediction (outside the band, same sign across seeds) is not observed in any cell" hold only as "not rejected by the t-prediction test". By the plain [min, max] of the 6 (or 2) untrained values, several cells are outside, with the same sign in both seeds:
  - Top PFN, sign call, WTA: s0 −1.47, s1 −1.35, band [+0.28, +0.49], n = 2, z ≈ −12, p = 0.06.
  - QG ParticleNet, ε_S 0.3, WTA: s0 +0.42, s1 +0.54, band [−0.10, +0.30], n = 6.

  With n = 2–6 untrained networks the test has almost no power. These are exactly the cells a larger band was needed for, which is what this verify adds.

## Procedure

- **Samples.**
  - Top Landscape test: `data_top.sample(3000/class, seed 5691)`, 6,000 jets.
  - QG Pythia test: `data_qg.sample(3000/class, seed 5692)`, 6,000 jets.
  - None of these seeds was used before.
- **Edit (unchanged H39 protocol).** Every particle's (Δη, Δφ) is recomputed about the WTA axis or about the hardest particle. This is a rigid translation of the jet by the axis offset δ_j.
- **Control.** The same per-jet length |δ_j| applied in a uniformly random direction, 10 directions per jet (20 in H56).
- **D (percentage points, pp).** D = mean over jets of [1(call changes about the axis) − mean over directions of 1(call changes under the random shift)]. It is the part of the call change that comes from the shift pointing at the hardest particle rather than in an arbitrary direction.
- **Calls.**
  - The sign of the log-odds.
  - Fixed thresholds at signal efficiency ε_S = 0.3 and 0.5, where signal is top for Top and quark for QG. Each threshold is set on this sample's unedited scores.
- **Networks.**
  - Trained seed 0 (s0) and seed 1 (s1) for ParticleNet_kin, PFN and EFN (Top and QG), and for ParT_kin (QG only; Top ParT has no s1).
  - Untrained (epoch-0) bands with new seeds:
    - ParT: 12 networks, `torch.manual_seed(2000+k)`.
    - ParticleNet: 20 networks, same seeding.
    - EFN and PFN: 20 networks each, energyflow's default initialiser. Kernels are drawn from U(±√(6/fan_in)) (he_uniform) with zero biases, with the tensor shapes of the stored energyflow inits.
    - Check of the EFN/PFN draw against the stored inits: max |w| / √(6/fan_in) = 0.98–1.00, std / expected = 0.94–1.00, biases 0.
  - Shuffled-label networks were left out, because they are constant (see H56).
- **Count step (H24v protocol).**
  - Sample: Pythia `h14.load_sample('py', 1000/class, seed 5693)`, with H24 edits seeded 5693. The edits apply to 1,998 of 2,000 jets.
  - sp_k: an exact collinear split of k = 1–5 soft particles into two halves at the same (y, φ). N rises by k and every IRC-safe quantity is unchanged.
  - ct_k: the ΔN = 0 control with pT^D matched.
  - Metric: per jet, the slope over k of Δlog-odds toward gluon under sp_k, minus the same slope under ct_k. Units are log-odds per added particle.
  - Herwig-trained ParT_kin and ParticleNet_kin (`models/qg_herwig/*/net_best_epoch_state.pt`) were added, so the generator shift and the seed difference are measured on the same jets.
- **Statistics.**
  - 95 % jet bootstrap (B = 2000).
  - Paired sign-flip p (P = 20,000) for s1 − s0 and Herwig − Pythia.
  - Trained vs untrained: z and the two-sided t prediction-interval p (df = n − 1).
  - Holm over all 140 tests of this run.
- **Hypotheses.**
  - H0: trained D is drawn from the untrained distribution.
  - H1 (D4m formula-driven reading): trained D lies outside it, with the same sign in s0 and s1.

## Accuracies of the unedited networks on the samples used

| sample | ParT s0 / s1 | ParticleNet s0 / s1 | PFN s0 / s1 | EFN s0 / s1 | untrained (range) |
|---|---|---|---|---|---|
| Top, 6k | 94.03 / – | 93.90 / 94.08 | 93.13 / 93.20 | 92.33 / 92.23 | 38–58 % |
| QG Pythia, 6k | 82.88 / 82.97 | 82.87 / 82.65 | 81.72 / 81.38 | 81.07 / 81.12 | 42–62 % |
| QG count, 2k | 82.55 / 82.20 (Herwig-trained 79.40) | 82.55 / 82.40 (Herwig-trained 78.45) | 81.20 / 80.65 | 80.80 / 80.45 | |

## Numbers

### Axis direction, ε_S = 0.5, WTA, D in pp [95 % CI]

The untrained band is [min, max] over n networks. p is the prediction-interval p against that band.

| sample | arch | s0 | s1 | s1 − s0 [CI], p | untrained band (n, SD) | H56 sample (s0 / s1) |
|---|---|---|---|---|---|---|
| Top | ParT | −1.79 [−2.41, −1.21], p 0.26 | – | – | [−1.90, +4.97] (12, 2.06) | −2.12 / – |
| Top | ParticleNet | −0.49 | −0.71 | −0.22 [−0.69, +0.21], 0.32 | [−4.73, +4.87] (20, 2.28) | −0.58 / −0.56 |
| Top | PFN | −0.97 | −1.73 | −0.76 [−1.56, −0.05], 0.052 | [−6.31, +2.01] (20, 2.20) | −1.12 / −1.34 |
| Top | EFN | −0.61 | −1.63 | −1.01 [−1.95, −0.12], 0.025 (Holm 1) | [−4.96, +0.55] (20, 1.52) | −1.13 / −1.35 |
| QG | ParT (s1 final) | +0.14 | +0.31 | +0.17 [−0.20, +0.55], 0.38 | [−0.65, +0.41] (12, 0.35) | +0.17 (best) / +0.35 (ep 16) |
| QG | ParticleNet | +0.44 | +0.50 | +0.06 [−0.34, +0.46], 0.77 | [−0.46, +0.55] (19, 0.29) | +0.35 / +0.77 |
| QG | PFN | −0.05 | −0.16 | −0.12 [−0.65, +0.46], 0.69 | [−6.64, +0.34] (20, 1.95) | +0.68 / −0.02 |
| QG | PFN, HP | +0.17 [−0.29, +0.62] | −0.22 | −0.40 [−0.96, +0.18], 0.18 | [−7.25, +0.34] (20, 2.11) | +1.16 / +0.08 |
| QG | EFN | +4.13 [+3.40, +4.88], z 1.4 | +3.82, z 1.3 | −0.30 [−1.17, +0.55], 0.49 | [−1.98, +4.01] (20, 1.95) | +3.88 / +3.61 |

- **Against the untrained band.** 82 network × cell tests (2 datasets × 4 architectures × 3 calls × 2 axes, s0 and s1; the 4 ParticleNet sign-call cells have no non-degenerate untrained network). None has a prediction-interval p < 0.05, even before correction.
  - Smallest: QG ParticleNet s1, ε_S 0.3, WTA: +0.80, z 1.9, p 0.078, against an untrained band now [−0.74, +0.96] with n = 19. The H56 band was [−0.10, +0.30] with n = 6.
  - Top PFN at the sign call, WTA: s0 −0.70, s1 −0.87, against a band [−7.01, +0.52] with n = 5. In H56 the band was [+0.28, +0.49] with n = 2.
  - The two cells that looked outside the range with n ≤ 6 are inside once the band has 5–20 networks.
- **Same sign in both seeds and toward the band edge.** QG EFN at ε_S 0.5 (WTA and HP) sits at the upper edge of the band in both seeds and on both jet samples: +3.6 to +4.6 pp, z 1.3–1.6. QG ParticleNet is positive in all four working-point cells in both seeds (+0.29 to +0.82). Neither reaches p < 0.05.
- **Seed differences that reproduce H56 (Holm p of this run):**
  - Top EFN, sign call: WTA −1.57 [−2.26, −0.89], HP −1.80, Holm 0.007. H56: −1.66 / −1.97.
  - Top ParticleNet, sign call, HP: −1.07, Holm 0.038. H56: −0.94.
  - QG PFN, sign call: +1.22 / +1.19, Holm 0.09 / 0.21. H56: +1.05 / +1.09.
- **Seed difference that does not reproduce.** QG PFN at ε_S 0.5, HP: H56 had −1.08 [−1.52, −0.63] (Holm 0.012); here it is −0.40 [−0.96, +0.18], p 0.18.
  - The reason is that the same seed-0 network gives D = +1.16 [+0.79, +1.53] on the H56 jets (seed 3911) and +0.17 [−0.29, +0.62] on the new jets (seed 5692). That difference, 1.0 pp, is larger than the two jet-bootstrap CIs allow (z ≈ 3.3).
  - The jet bootstrap holds the working-point threshold fixed, but the threshold is re-estimated on each sample. So the bootstrap CIs at fixed signal efficiency understate the sample-to-sample spread.
  - The H39/H56 statement that the "near-edge" QG PFN cells "move to near 0 with seed 1" is better described this way: the seed-0 value on the H39 sample was high, and on new jets seed 0 and seed 1 both lie near 0.

### Count response, exact split − ΔN = 0 control, log-odds toward gluon per added particle (1,998 new Pythia jets)

| network | s0 | s1 | s1 − s0 [CI], p | Herwig-trained | Herwig − s0 / Herwig − s1 | untrained band (n) |
|---|---|---|---|---|---|---|
| ParT_kin | +0.0083 [0.0069, 0.0097] | +0.0058 [0.0043, 0.0073] | −0.0024 [−0.0036, −0.0014], 1.5 × 10⁻⁴ (Holm 0.019) | −0.0040 | −0.0123 / −0.0098 | [−0.035, +0.059] (12) |
| ParticleNet_kin | +0.0036 [0.0019, 0.0053] | +0.0037 [0.0019, 0.0055] | +0.0001 [−0.0014, +0.0015], 0.93 | −0.0087 | −0.0123 / −0.0123 | [−0.0013, +0.0010] (20) |
| PFN | +0.0209 | +0.0252 | +0.0043 [0.0034, 0.0053] | | | [−0.22, +0.30] (20) |
| EFN | −0.0035 (split part 0.0000) | −0.0032 | +0.0003 [0.0001, 0.0006] | | | [−0.00006, +0.00007] (20) |
| LR(N) | +0.1014 [0.0998, 0.1031] | | | | | |

- **Size relative to a pure particle counter.** For ParT_kin and ParticleNet_kin the response is 3.5–8 % of LR(N). LR(N) is the negative-binomial likelihood ratio in the particle number N alone. H56 gave 4–8 %.
- **Trained vs untrained.**
  - ParticleNet_kin (both seeds) lies outside its untrained band: z ≈ 6, p ≈ 10⁻⁵, Holm 0.002.
  - ParT_kin lies inside its band (z 0.3), so its count response cannot be told apart from what an untrained ParT shows.
- **Herwig shift vs seed difference.**
  - ParT: the Herwig-trained minus Pythia-trained shift (−0.0098 to −0.0123) is 4–5 times the seed difference (0.0024). H56 said 8 times, with a seed difference of 0.0016.
  - ParticleNet: the seed difference is 0.0001 with CI [−0.0014, +0.0015]. Against the CI bound, the shift (0.0123) is at least 8 times the seed difference. The H56 "> 100×" divides by a point estimate that is consistent with zero.
  - The Herwig − Pythia difference is about the same whichever Pythia seed is used.

## Adequacy of the controls

- **Untrained networks.** These remain the only usable null for the axis test, and n = 6 (n = 2 at the sign call) was too small to locate the band edge. With 12–20 networks the bands are wider than H56's for most cells, and every trained value lies inside them. The test is therefore conservative. The power limit is that a trained D within about two untrained SDs (0.6–9 pp depending on the cell) cannot be detected.
- **Shuffled-label networks** are constant and test nothing (H56 says so). A non-degenerate "no class information" network is still missing.
- **One extra seed.** A single s1 − s0 difference does not estimate the seed variance. The paired jet-bootstrap CI covers only the estimation on these jets, and at fixed-efficiency calls it also leaves out threshold re-estimation.

## Corrected reading

For the second seed and on a new jet sample, the direction part of the axis response (call change about the WTA or HP axis minus the call change under shifts of the same length in random directions) cannot be told apart from untrained networks of the same architecture in any cell. This uses 12–20 untrained networks per architecture; the smallest uncorrected p is 0.078. The H56 cells that lay outside the [min, max] of 2–6 untrained networks lie inside the larger bands.

Seed-to-seed differences of 1–2 pp reproduce at the sign call (Top EFN, Top ParticleNet HP, QG PFN). At ε_S 0.5 the largest H56 seed difference (QG PFN HP, −1.08 pp) does not reproduce (−0.40 pp, p 0.18). That cell's seed-0 value differs by 1.0 pp between the two jet samples, so the H39 "near-edge" QG PFN values reflect the jet sample, and the jet-bootstrap CIs at fixed efficiency understate between-sample spread.

The count response at fixed IRC-safe energy flow reproduces: ParT_kin +0.008 / +0.006 and ParticleNet_kin +0.004 / +0.004 per particle (seed 0 / seed 1), 3.5–8 % of LR(N). ParticleNet's is outside its untrained band; ParT's is not. The Herwig-minus-Pythia shift is 4–5 times the seed difference for ParT and at least 8 times it for ParticleNet (not 8 and > 100 times). The statement "not a learned use of the axis direction" is supported only down to the width of the untrained band: an effect smaller than about two untrained SDs is not excluded.

## Corrected Conclusion

On a second training seed and on new jets, the extra response of trained taggers to recentring the jet on its hardest particle (beyond a shift of the same length in a random direction) stays within the range shown by 12–20 untrained networks of the same type. So training has not added a detectable use of the axis direction, though the untrained range is wide enough (up to several percentage points) to hide a small one. The response to adding particles at fixed IRC-safe energy flow is small: 0.004–0.008 log-odds per particle, against about 0.10 for a pure particle counter. It repeats across seeds, and the change between Herwig- and Pythia-trained networks is 4 to more than 8 times the seed-to-seed difference. Still uncertain: there is only one extra seed; the shuffled-label networks learned a constant and test nothing; the error bars at fixed signal efficiency understate how much the axis numbers change from one jet sample to another; and the jet-charge test is still waiting for its networks.

## Explainer entry (verify step, for site/explain.py H56)

```
H56V_EXPLAINER = [
 ('Question', 'Does the H56 seed result hold on new jets and against a larger set of untrained networks, and is the Herwig-vs-Pythia count shift larger than the seed difference when both are measured on the same jets?', None),
 ('Method', 'Fresh Top (seed 5691) and QG Pythia (seed 5692) test jets, 6,000 each; untrained bands of 12 ParT, 20 ParticleNet, 20 PFN and 20 EFN networks (new seeds; EFN/PFN drawn with energyflow\'s he\\_uniform initialiser, checked against the stored inits). $D$ = call change about the WTA/HP axis minus the mean call change under 10 shifts of the same length $|\\delta_j|$ in random directions. Count: 1,998 fresh Pythia jets (seed 5693), slope of $\\Delta$log-odds over $k$ exact collinear splits minus that under the $\\Delta N=0$ control; Pythia seeds 0 and 1 and the Herwig-trained ParT\\_kin / ParticleNet\\_kin on the same jets.', None),
 ('Result', 'No trained cell outside the untrained distribution (82 network-cell tests, smallest $p=0.078$). Seed differences at the sign call reproduce (Top EFN $-1.6$ pp, Holm $0.007$); the H56 QG PFN $\\varepsilon_S=0.5$ seed difference does not ($-0.40$ pp, $p=0.18$), because the seed-0 value moves by 1.0 pp between jet samples. Count: ParT\\_kin $+0.0083/+0.0058$, ParticleNet\\_kin $+0.0036/+0.0037$ per particle against LR(N) $+0.101$; Herwig $-$ Pythia $-0.012$, i.e. 4-5$\\times$ (ParT) and $\\geq 8\\times$ (ParticleNet) the seed difference.', None),
 ('Checks', 'Axis edit equals the stored-array shift to 0; EF port $\\leq 4.8\\times10^{-6}$; random rows per class; QG ParT seed 1 final checkpoint.', None),
 ('Physical significance', 'Training on top or quark/gluon labels does not add a detectable dependence on where the jet axis points beyond generic sensitivity to translating the jet; the small residual dependence on particle number at fixed energy flow is a reproducible property of ParticleNet training, and its sign flips with the generator used in training.', None),
]
```

Glossary additions: **untrained (epoch-0) band**: the [min, max] and SD of a response over networks built with the training code but not trained; **between-sample spread**: change of a measured quantity for the same network on two independent random jet samples.
