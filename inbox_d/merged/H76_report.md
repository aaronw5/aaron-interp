# H76: Which generalized angularity quark/gluon taggers follow, measured with edits that move one angularity at fixed others (claims D7, D22)

## Conclusion

On Pythia and Herwig quark/gluon jets we edited each jet so that one of five classical quark/gluon observables changed and the other four did not. The five observables are particle count, momentum sharing (pT^D), and three angular-spread measures: LHA, width and λ(1,2) ≈ m²/pT². Every trained network (ParT, ParticleNet, LorentzNet, PFN, EFN; Pythia- or Herwig-trained) responded almost only to the angular-spread edits. Raising width at fixed LHA, or lowering LHA at fixed width, moved them toward "gluon". Their five-number response pattern matches the IRC-safe EFN's (cosine 0.98–1.00), not the pattern of a boosted-tree classifier trained on the five observables, which is led by the count (cosine 0.33–0.61). The count edit (±10 % particles by exact splits / merges) moved ParT_kin by −0.013 log-odds per +10 %, smaller than a weight-permuted ParT (−0.05). This does not support a counting reading or a pT^D reading at fixed angular shape. What remains uncertain: the LHA and width edits are clearly off the natural-jet distribution (a classifier tells edited from real jets with AUC 0.86–0.91). They also fail on 4–20 % of jets, and the response to them is not linear. So the size of the LHA/width response, as opposed to its direction, is not established for real jets.

Terms: angularity λ^κ_β = Σ_i z_i^κ (ΔR_i/R)^β (z_i = pT_i/Σ pT, ΔR_i = distance of particle i from the jet axis, R = 0.4). Log-odds s = log p(gluon) − log p(quark). EFN = energy-flow network, IRC-safe by construction (its output cannot depend on the particle count at fixed energy flow).

## Question

Earlier QG nodes (H5, H24, H30, H54, H57) showed that the networks respond to count edits and to where soft particles sit, but those results were never placed on one map. The generalized angularities λ^κ_β (Larkoski, Thaler, Waalewijn 2014) put the classical QG observables on one axis set:
- (κ, β) = (0, 0): multiplicity N
- (2, 0): (pT^D)²
- (1, ½): LHA (Les Houches angularity)
- (1, 1): width / girth (units of R)
- (1, 2): ≈ m²/pT² (units of R²)

Question: when one of them moves and the others are held fixed, which one moves each network?

Competing predictions (from the plan):
- **Count reading:** the N direction dominates (|g_N| > 2× every other component), and the Herwig-trained models show a different N slope.
- **IRC-safe-shape reading:** LHA / width dominate, the N direction is ≈ the control, and the cosine with the all-five GBDT is > 0.9.
- **pT^D reading:** the pT^D direction dominates at fixed N.

## Procedure

**Sample.** KMT Pythia 8.226 test jets (interp.data_qg) and Herwig 7.1 test jets (Zenodo 3066475 files 18–19, interp.h14_qg_generator.load_sample). For each generator: 4,000 quark + 4,000 gluon jets, random rows, seed 7601, jets with N ≥ 8 (0.06 % / 0.08 % of drawn jets dropped). Verification rerun: seed 7611, 2,000 + 2,000 per generator (H76_verify.md).

**Angularities.** All are computed about the E-scheme axis: the rapidity and φ of the four-vector sum of the massless constituents. They are recomputed from the edited particle list in every case, and the networks also rebuild their inputs about the edited jet's own axis.

**Edit (IRC-safe directions t ∈ {LHA, width, pT^D, λ(1,2)}).** Each particle i has two variables:
- a_i = log pT_i (a "z move": pT is exchanged between particles);
- b_i = r_i / R (a "dR move": the particle slides radially toward or away from the unedited axis along its own azimuth θ_i about that axis; θ_i is unchanged).

A batched Gauss–Newton solver takes minimum-norm steps. It minimises Σ_i (δ log pT_i)² + (δ r_i / R)², at most 100 iterations with each step capped at 0.2. It drives the constraint vector C = (log λ_pT^D, log λ_LHA, log λ_width, log λ_(1,2), log Σ pT, Δy_J/R, Δφ_J/R) to its unedited values, except that the target angularity is multiplied by 1 ± 0.03. N is unchanged because no particle is added or removed.

The main step is ±3 %; responses are reported per +10 % by multiplying by 10/3. ±10 % (the planned size) was run as a side arm. With the other three fixed, it fails on most jets for LHA and width (see Issues). The reason is that log λ^1_β is convex in β (Lyapunov's inequality), so the middle one of three moments has little room to move when its neighbours are fixed.

**Acceptance per jet.** All of the following must hold:
- the solver converged (max |residual| < 10⁻⁶);
- an independent numpy recomputation shows each kept angularity, Σ pT, the vector jet pT and N within 0.5 %, and the jet η and φ within 0.5 % of R;
- the achieved target change is ±3 % within 0.015 %.

**Edit (N direction).** k = max(1, round(0.1 N)) particles are changed:
- N+: the k softest particles are each split exactly into two halves at the same position. This leaves every IRC-safe angularity unchanged and lowers (pT^D)², which is then restored by the solver.
- N−: k times, the softest particle is merged into its nearest-in-angle neighbour (pT summed, pT-weighted position); then all four angularities, Σ pT and the axis are restored.
- Nsp: N+ before the pT^D restore. This is the exact-split check, on which the EFN must give 0.

Slope per +10 %: (s(N+) − s(N−))/2 × 0.1/(k/N).

**Controls.**
- **crand:** a random edit of the same per-particle size. Each particle gets a random 2D direction in (a_i, b_i) with the size |δ_i| it had in the t+ edit. The vector is projected onto the null space of the constraint Jacobian, rescaled to the t+ edit's total size, applied with + and − signs, and each version is restored to the unedited constraints by the solver. The crand slope is (s(r+) − s(r−))/2 with the same scaling. For N, crand is a null-space move of the size of the N+ restore step, applied to the split jet.
- **crot:** the 3 particles the t+ edit moved most are rotated in azimuth about the axis (random sign). The angle is chosen so that the arc length equals that particle's radial displacement in the edit. Then the unedited constraints are restored. This gives position changes of matched size at fixed angularities.
- **Weight-permuted** ParT_kin and PN_kin.
- **EFN** (IRC-safe).

**Metric.** Per jet and direction: d_t = (s(t+) − s(t−))/2 per +10 %, effect = d_t − crand_t.
- Means are taken over jets accepted in every main condition. Accepted fraction: Pythia 84.6 %, Herwig 79.8 %; the losses are LHA+ and width± failures.
- 95 % CIs come from 1,000 bootstrap resamples over jets.
- p is a sign-flip test with 20,000 sign vectors.
- Response vector g = (effect_N, effect_LHA, effect_width, effect_pTD, effect_mass), unit vector g/|g|.
- Cosine with each reference's g, with a bootstrap CI over jets (500 resamples).
- H0 "network unit vector = all-five-GBDT unit vector": sign-flip test on per-jet differences of the two vectors, each scaled by its model's own mean norm.
- Holm correction over 152 tests: 11 networks × 5 directions × 2 generators, 22 H0 tests, and 20 Herwig-minus-Pythia-trained differences.

**Models.**
- Pythia-trained: ParT_kin, ParticleNet_kin (H2 weaver), LorentzNet (released), PFN and EFN (H2 energyflow, ported), PFN_t and EFN_t (H5 torch, second training runs).
- Herwig-trained: ParT_kin and PN_kin (H14/H24).
- Weight-permuted ParT_kin and PN_kin.

**References.** Trained here on 30k + 30k train-split jets of each generator:
- GBDT on each single angularity (5);
- GBDT on all five ("g5");
- LR(N), the negative-binomial likelihood ratio (interp.h14_qg_generator.lrn_tables).

**C2ST.** A gradient-boosted classifier separates edited jets (one half of the sample) from unedited jets (the other half), using 8,000 jets per generator. Features: N, minimum pair distances, softest pT, z of the 5 leading particles, radii of the 3 leading particles, the four IRC-safe angularities and jet mass. The value reported is the held-out AUC; 0.5 means the two sets cannot be told apart.

## Unedited models on the sample (8,000 jets per generator; AUC / accuracy at s = 0)

| model | Pythia jets | Herwig jets | earlier (H24 / H57) |
|---|---|---|---|
| ParT_kin (Py) | 0.907 / 0.827 | 0.812 / 0.720 | 0.904 / 0.833 ; 0.815 / 0.721 (H24); acc 0.827 (H57) |
| PN_kin (Py) | 0.907 / 0.830 | 0.812 / 0.721 | 0.905 / 0.832 ; 0.817 / 0.720 |
| LorentzNet (Py) | 0.923 / 0.849 | 0.802 / 0.699 | acc 0.845 (H57) |
| PFN (Py) | 0.898 / 0.820 | 0.816 / 0.729 | 0.896 ; 0.818 |
| EFN (Py) | 0.888 / 0.809 | 0.804 / 0.716 | 0.886 ; 0.809 |
| PFN_t / EFN_t | 0.888 / 0.870 (AUC) | 0.811 / 0.796 | – |
| ParT_kin (Hw) | 0.889 / 0.800 | 0.834 / 0.756 | 0.888 / 0.803 ; 0.836 / 0.760 |
| PN_kin (Hw) | 0.886 / 0.795 | 0.836 / 0.760 | 0.884 / 0.796 ; 0.838 / 0.762 |
| ParT_kin / PN_kin permuted | 0.548 / 0.459 | 0.523 / 0.463 | 0.539 |
| GBDT all five (own generator) | 0.870 / 0.788 | 0.807 / 0.734 | – |
| LR(N) | 0.846 / 0.768 | 0.794 / 0.727 | – |

Every network reproduces its earlier AUC within 0.006.

## What the edits do (median over jets; Pythia ; Herwig)

| edit | accepted | size ‖δ‖ (in log pT and r/R) | Δ jet mass | C2ST AUC |
|---|---|---|---|---|
| pT^D ±3 % | 100 % | 0.081–0.088 | 0 | 0.509–0.518 ; 0.511–0.525 |
| λ(1,2) ±3 % | 99.8–100 % | 0.051–0.061 | ±1.5 % | 0.564–0.644 ; 0.568–0.637 |
| LHA +3 % / −3 % | 84.6 / 97.1 % ; 79.9 / 96.4 % | 0.375 / 0.109 ; 0.475 / 0.132 | 0 | 0.886 / 0.905 ; 0.874 / 0.904 |
| width +3 % / −3 % | 96.0 / 93.4 % ; 94.6 / 91.3 % | 0.113 / 0.270 ; 0.137 / 0.335 | 0 | 0.870 / 0.869 ; 0.875 / 0.859 |
| N +10 % (splits) / −10 % (merges) | 100 % | restore 1.3e-5 / 0.006 | 0 | 1.00 / 0.955 ; 1.00 / 0.950 |
| crand (width, pT^D) | 100 % | matched | 0 | 0.545, 0.528 ; 0.547, 0.512 |
| crot (width, λ(1,2)) | 100 % | matched | 0 | 0.531, 0.509 ; 0.539, 0.492 |
| ±10 % side arm: LHA+ / LHA− / width+ / width− | 27 / 81 / 78 / 53 % ; 23 / 77 / 74 / 45 % | 0.42–1.69 | | LHA+10 0.974, width+10 0.979 |

In accepted jets, the kept quantities moved by a median of 10⁻⁶–10⁻⁴ relative (99th percentile < 0.1 %).

Figure results/physics/h76_angularity.png, left: a Pythia gluon jet (N = 58) under width +10 %. Width goes 0.162 → 0.179 while LHA (0.362), λ(1,2) (0.0609) and (pT^D)² (0.071) are unchanged. The solver moves the hard near-axis particles outward by up to about 0.05, pulls several wide-angle soft particles inward, and leaves the jet axis fixed. Middle: (pT^D)² +10 % (0.071 → 0.078) is made mostly by pT exchange among the hard core, with positions nearly unchanged.

## Numbers: effect = edit − crand, gluon log-odds per +10 % move [95 % CI] (Pythia jets; Herwig jets in the second line)

| model | N | LHA | width | (pT^D)² | λ(1,2) | unit vector g/|g| |
|---|---|---|---|---|---|---|
| ParT_kin (Py) | −0.013 [−0.015, −0.010] | −0.93 [−0.97, −0.88] | +0.71 [0.67, 0.75] | −0.060 [−0.078, −0.043] | −0.22 [−0.24, −0.20] | (−0.01, −0.78, 0.60, −0.05, −0.18) |
| | −0.010 | −0.74 | +0.58 | −0.053 | −0.22 | (−0.01, −0.77, 0.60, −0.05, −0.23) |
| PN_kin (Py) | −0.041 | −0.96 | +0.76 | −0.042 | −0.24 | (−0.03, −0.77, 0.61, −0.03, −0.19) |
| LorentzNet | +0.011 | −0.48 | +0.33 | −0.031 | −0.12 | (0.02, −0.81, 0.55, −0.05, −0.20) |
| PFN | +0.101 | −0.75 | +0.50 | −0.045 | −0.066 | (0.11, −0.82, 0.55, −0.05, −0.07) |
| PFN_t | +0.305 | −0.79 | +0.56 | −0.041 | −0.074 | (0.30, −0.78, 0.55, −0.04, −0.07) |
| EFN | +0.003 | −1.58 | +1.33 | −0.039 | −0.25 | (0.00, −0.76, 0.64, −0.02, −0.12) |
| ParT_kin (Hw-trained) | −0.099 | −0.69 | +0.67 | −0.039 | −0.23 | (−0.10, −0.69, 0.67, −0.04, −0.23) |
| PN_kin (Hw-trained) | −0.169 | −0.80 | +0.80 | −0.030 | −0.29 | (−0.14, −0.68, 0.68, −0.03, −0.24) |
| ParT_kin permuted | −0.050 | +0.001 | −0.013 | +0.003 | +0.005 | (−0.96, …) |
| PN_kin permuted | −0.093 | +0.021 | −0.040 | −0.006 | +0.022 | (−0.88, …) |
| GBDT all five (Py) | +0.315 | −0.20 | +0.060 | −0.072 | +0.020 | (0.82, −0.52, 0.16, −0.19, 0.05) |
| GBDT single angularity (Py) | +0.364 | +0.26 | +0.15 | −0.21 | +0.11 | (axis) |
| LR(N) (Py) | +0.402 | 0 | 0 | 0 | 0 | (1, 0, 0, 0, 0) |

**Control slopes.** The crand slopes are between −0.03 and +0.03 for every model and direction. On Herwig jets the unit vectors of every network agree with the Pythia-jet vectors within 0.05 per component.

**Unsigned sensitivity** (mean |s(edit+) − s0| vs crand vs crot; Pythia, ParT_kin):

| direction | edit | crand | crot |
|---|---|---|---|
| LHA | 0.76 | 0.28 | 0.27 |
| width | 0.48 | 0.28 | 0.23 |
| λ(1,2) | 0.18 | 0.10 | 0.07 |
| pT^D | 0.039 | 0.20 | 0.031 |

The pT^D edit moves ParT less than a random null-space edit of the same size.

**Cosine with the EFN unit vector** (computed from the mean vectors):

| model | Pythia jets | Herwig jets |
|---|---|---|
| ParT_kin | 0.997 | 0.994 |
| PN_kin | 0.996 | 0.991 |
| LorentzNet | 0.992 | 0.980 |
| PFN | 0.987 | 0.982 |
| PFN_t | 0.95 | 0.94 |
| Herwig-trained ParT / PN | 0.985 / 0.977 | 0.984 / 0.976 |
| permuted ParT | −0.19 | −0.18 |

**Cosine with the all-five GBDT** (same generator) [95 % CI]:

| model | Pythia jets | Herwig jets |
|---|---|---|
| ParT_kin | 0.487 [0.473, 0.500] | 0.394 |
| PN_kin | 0.462 | 0.364 |
| LorentzNet | 0.519 | 0.470 |
| PFN | 0.608 | 0.523 |
| PFN_t | 0.736 | 0.673 |
| EFN | 0.491 | 0.407 |
| Herwig-trained ParT / PN | 0.377 / 0.332 | 0.342 / 0.299 |

Cosine with LR(N): −0.14 to +0.13 for all networks except PFN_t (0.30–0.33).

**Restricted to the four IRC-safe components** (N dropped), the cosine with the GBDT is 0.79–0.91. The GBDT has the same sign pattern (LHA −, width +, pT^D −) but a far smaller size: |LHA| 0.20 vs 0.48–0.96 for the networks.

**H0** (network unit vector = all-five-GBDT unit vector): rejected for all 11 networks on both generators (sign-flip p = 5 × 10⁻⁵, the floor; Holm < 0.01). Overall, 141 of 152 tests have Holm p < 0.05. The 11 that do not:
- permuted ParT LHA;
- EFN_t N on Herwig jets;
- 8 of 20 Herwig-minus-Pythia differences in the IRC-safe directions.

**Generator dependence** (Herwig-trained minus Pythia-trained, paired over the same Pythia jets):

| direction | ParT_kin | PN_kin |
|---|---|---|
| N | −0.086 [−0.089, −0.083] | −0.127 [−0.131, −0.124] |
| LHA | +0.24 [0.21, 0.28] | +0.16 |

Herwig jets give −0.062 / −0.091 for N. The Herwig-trained networks move further toward quark when particles are added, in line with H24.

**Linearity** (jets where the ±10 % edit also solved; Pythia, ParT_kin; ±10 % slope vs the ±3 % slope × 10/3 on the same jets):

| direction | ±10 % | ±3 % × 10/3 |
|---|---|---|
| LHA | −0.42 | −0.61 |
| width | +0.49 | +0.81 |
| λ(1,2) | −0.11 | −0.22 |
| pT^D | −0.064 | −0.061 |

Other networks show the same pattern: the responses to the angular edits grow more slowly than linearly, while pT^D is linear.

**Per class** (ParT_kin, Pythia, quark / gluon jets):

| direction | quark jets | gluon jets |
|---|---|---|
| LHA | −0.71 | −1.16 |
| width | +0.72 | +0.71 |
| λ(1,2) | −0.23 | −0.20 |
| N | −0.021 | −0.004 |

## Reading (scoped to these edits and samples)

- **Count prediction:** not met. On both generators every trained ParT, ParticleNet and LorentzNet has |g_N| ≤ 0.04 per +10 %. The permuted networks give −0.05 to −0.09, while LR(N) gives +0.40 and the five-angularity GBDT +0.32. The trained ParT / PN / LorentzNet N components (−0.04 to +0.03) are smaller in magnitude than those of the permuted networks. Only PFN (+0.10) and PFN_t (+0.30) have a sizeable positive (count-like) N component. The Herwig-trained nets do differ from the Pythia-trained ones in N slope, by −0.09 to −0.13, but toward quark.
- **pT^D prediction:** not met. At fixed N, LHA, width and λ(1,2), a +10 % (pT^D)² move changes every network by −0.03 to −0.06, about 0.15–0.3 of the single-variable GBDT(pT^D). Its unsigned effect is below the size-matched random edit.
- **IRC-safe-shape prediction:**
  - Met in direction: LHA and width dominate, with opposite signs (width up or LHA down → gluon), λ(1,2) up → quark, and the N component is ≈ 0.
  - Not met in its "cosine with GBDT > 0.9" part. The cosine is 0.33–0.61, because the GBDT, trained on natural jets, puts most of its partial response on N.
  - The networks' vectors coincide with the IRC-safe EFN's (cosine ≥ 0.94; ≥ 0.98 for every network except PFN_t).
- **What the dominant direction is.** Width up at fixed LHA and λ(1,2), or LHA down at fixed width and λ(1,2), changes the shape of the radial energy profile Σ z_i (ΔR_i/R)^β as a function of β while keeping the overall spread. The networks read that shape; no single classical angularity captures it, and the single-variable GBDTs give the opposite sign for LHA.
- **Scope.** These are partial derivatives at fixed other angularities. In natural jets all angularities rise together from quark to gluon, so this does not say which angularity carries the networks' discrimination on real jets. The LHA and width edits are off-manifold (C2ST 0.86–0.91), so the size of their responses is not a statement about natural-jet variations.

## Issues / limits

- The planned ±10 % edits for LHA and width are infeasible on 19–77 % of jets, because log λ^1_β is convex in β (Lyapunov). The main analysis therefore uses ±3 % and linear scaling. The ±10 % arm shows responses about 0.5–0.7 of the linear extrapolation for LHA, width and λ(1,2), so "per +10 %" for these directions is an upper-scale number.
- Solver acceptance depends on the jet. At ±3 %, 3–20 % of jets are rejected, mostly LHA+ and width−. Rejected jets are excluded from every direction (common jet set: 84.6 % Pythia, 79.8 % Herwig). The selection is not random: jets close to the moment bound are dropped.
- On-manifold check, against the target C2ST < 0.6:
  - met for pT^D, crand and crot;
  - borderline for λ(1,2) (0.56–0.64);
  - failed for LHA and width (0.86–0.91) and for the count edits (exact splits 1.00, merges 0.95; H24v found near-collinear splits at ΔR 0.01 give C2ST 0.52 with the same response within 0.0003).
- For N, the crand control is very small (the pT^D restore after splitting soft particles is about 10⁻⁵), so the N effect equals the raw split-minus-merge slope. The permuted networks are the effective null.
- crot rotates the three most-moved particles with a matched arc length, not a full random azimuth. A first smoke version with full random rotations moved the networks 0.2–0.6 (much larger than the edits) and was replaced.
- The GBDT references are piecewise constant. Their ±3 % responses come from jets near bin edges, so a single-angularity GBDT vector is exactly axis-aligned by construction and its cosines are trivial.
- The kinpid networks were not included (species labels unchanged by these edits; not in the plan).
- Debug history: smoke 1 used log r as the radial variable, which gave very large relative moves of near-axis particles (‖δ‖ 3–6, C2ST 0.95). Smoke 2 switched to r/R and smoke 3 added the ±3 % main step. The numbers reported here all come from the final code.

## Files / jobs

- Script: interp/h76_angularity_qg.py (steps run, stats, fig).
- Jobs: d-h76-smoke, d-h76-smoke2, d-h76-smoke3 (GPU smoke, 100 + 100 jets per generator); d-h76-run (GPU, 2,180 s, exit 0); d-h76-stats (CPU 1 / 2 Gi, 369 s); d-h76-vrun (GPU) and d-h76-vstats (CPU), seed 7611.
- Logs: logs/d-h76-*.log.
- Results: results/physics/h76_run.json (models, per-condition solver and tolerance stats, base AUCs, C2ST), h76_stats.json, h76_py.npz, h76_hw.npz (per-jet logits, observables, ok flags), h76_example_py.npz, and the _v7611 counterparts.
- Figures: results/physics/h76_angularity.png, h76_angularity_v7611.png.

## Glossary additions (explain.GLOSSARY)

- **generalized angularity λ^κ_β**: Σ_i z_i^κ (ΔR_i/R)^β over a jet's particles, with z_i = pT_i/Σ pT and ΔR_i the (y, φ) distance from the jet axis. κ weights momentum and β weights angle. (0, 0) is the multiplicity, (2, 0) is (pT^D)², (1, 1) is width, and (1, 2) ≈ m²/pT². Larkoski, Thaler, Waalewijn 2014.
- **LHA**: Les Houches angularity, λ^1_{1/2}; a quark/gluon observable intermediate between multiplicity-like and mass-like.
- **pT^D**: √(Σ pT_i²)/Σ pT_i; large when a few particles carry most of the jet momentum.
- **crand**: control edit of the same per-particle size as the tested edit, in a random direction that leaves every tested angularity, the jet pT and the axis unchanged.
- **crot**: control that rotates the most-moved particles about the jet axis by an angle of matched arc length, at fixed angularities.

## Explainer entry (for site/explain.py; H76_EXPLAINER under Paper D)

**Question.** Which classical quark/gluon observable does each network follow, when the observables are moved one at a time?

**Physics background.** Gluon jets radiate more than quark jets (colour factor C_A = 3 vs C_F = 4/3). They therefore have more particles, softer fragmentation and broader angular spread. The generalized angularities

$$\lambda^{\kappa}_{\beta} = \sum_{i\in \rm jet} z_i^{\kappa}\left(\frac{\Delta R_i}{R}\right)^{\beta},\qquad z_i=\frac{p_{T,i}}{\sum_j p_{T,j}}$$

cover all three effects: $(0,0)=N$, $(2,0)=(p_T^D)^2$, $(1,\tfrac12)=$ LHA, $(1,1)=$ width, $(1,2)\approx m^2/p_T^2$. For κ = 1 they are IRC safe, and $\log\lambda^1_\beta$ is convex in β.

**Method.** For each jet we solve for the smallest change of particle momenta, $\delta\log p_{T,i}$, and radial positions, $\delta r_i/R$, that multiplies one angularity by $1\pm0.03$ and keeps the other angularities, $\sum p_T$ and the jet axis fixed:

$$\min \sum_i (\delta\log p_{T,i})^2+(\delta r_i/R)^2\quad\text{s.t.}\quad C(\text{jet}') = C(\text{jet}) + \Delta_t .$$

The count direction uses exact collinear splits (N up) or merges (N down) of the softest particles, followed by the same restore. The slope is $d_t=\tfrac{s(t^+)-s(t^-)}{2}$ per +10 %, with $s=\log p_g-\log p_q$, minus the same quantity for a random null-space edit of equal size.

**Result (figure h76_angularity.png).**
- Every trained network's response vector points along "width up / LHA down" with a small λ(1,2) part.
- It matches the IRC-safe EFN's (cosine 0.98–1.00, PFN_t 0.94–0.95).
- The count component is ≤ 0.04 in magnitude per +10 %, smaller than that of weight-permuted networks (−0.05 to −0.09); a count classifier gives 0.40.
- The pT^D component is −0.03 to −0.06.
- A boosted tree on the five angularities instead points mainly along N (cosine with the networks 0.33–0.61).

**Checks.**
- Kept quantities are fixed to < 0.1 % (99th percentile).
- Random null-space and matched-rotation controls give signed slopes |·| ≤ 0.03.
- The fresh seed 7611 reproduces every unit vector within 0.05.
- LHA and width edits are off-manifold (C2ST 0.86–0.91) and partly infeasible at ±10 %.

**Physical significance.** At fixed values of the classical angularities, the networks do not count particles and barely use momentum sharing. They respond to how energy is spread in angle, in a combination (the β-dependence of the radial profile) that no single angularity measures, as an IRC-safe energy-flow network does.
