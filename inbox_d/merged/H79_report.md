# H79 report (loop D, round D17; claims D30, D7 v2): which radial variable do the quark–gluon networks read — r_1, D or λ(1,0.1)?

Jobs: d-h79-run (GPU, 69 min), d-h79-stats (CPU), d-h79-fig (CPU); smoke jobs d-h79-smoke, d-h79-smoke2 (CPU, tag _smoke).
Script: interp/h79_radial_variable_qg.py (steps run / stats / fig). Results: results/physics/h79_run.json, h79_stats.json,
h79_radial.png, h79_example_{py,hw}.npz (volume: /j-jepa-vol/aaron-interp/results/physics/h79_{py,hw}.npz, per-jet logits).

## Conclusion

We moved the distance of a jet's hardest particle from the jet axis (r_1) by ±10 %, holding seven angular-spread observables, the momentum sharing, the particle count and the jet axis fixed. All nine trained quark/gluon networks responded, in both directions, on both generators, on edited jets that a classifier can barely tell from real ones. But they responded with the opposite sign to a classifier trained on r_1 alone: moving the hardest particle outward makes the networks call the jet more quark-like (ParT −0.042 gluon log-odds per +10 %, against +0.051 for the r_1 tree). So the networks do not read r_1 as the data-trained r_1 classifier does. They do use radial information that the seven angularities do not contain, since a tree trained on those angularities gives exactly 0 under this edit.

The radial dispersion D and the soft-insensitive moment λ(1,0.1) could not be moved by more than 1 % and 0.3 % at fixed other observables without edits that look unlike real jets and that also move λ(1,3) by 5–7 %. At those sizes the networks move toward quark for both signs, and the direction-dependent part is not stable across models or generators. The data therefore do not single out one radial variable, and D30 stays at "radial energy profile".

In the count arm, an edit that splits and merges particles with no net change in count moves the networks toward quark by about twice the direction-independent part of the count response. That part of the count response therefore does not need a change in count.

## Terms

- **Generalized angularity.** λ(κ,β) = Σ_i z_i^κ (ΔR_i/R)^β, where z_i = pT_i/Σ_j pT_j, ΔR_i is the (rapidity, azimuth) distance of particle i from the E-scheme jet axis (the four-vector sum), and R = 0.4.
  - LHA = λ(1,½); width = λ(1,1); λ(1,2) ≈ m²/pT²; (pT^D)² = Σ z_i².
  - "The seven angularities" are λ(1,β) for β = 0.1, 0.25, 0.5, 1, 1.5, 2 and 3.
- **The three targets.**
  - **r_1** is ΔR/R of the hardest particle (largest pT).
  - **D** = λ(1,1)/λ(1,½)² is the radial dispersion. D ≥ 1, with D = 1 when all of the jet's pT sits at one radius.
  - **λ(1,0.1)** is an angular moment that weights all radii almost equally (β close to 0).
- **s** is the network's gluon log-odds, log p(g) − log p(q). Δs = s(edited) − s(unedited), per jet.
- **crand_big** is the control edit. It is a random per-particle change of (log pT, r/R) with:
  - the same per-particle size profile and the same total size (norm) as the larger of the two signed edits;
  - its direction projected into the null space of the constraints, so it leaves all nine constrained quantities at their unedited values;
  - both signs applied, then the constraints restored exactly.

  It tests "any rearrangement of this size". "Excess" means Δs(edit) − mean Δs(crand_big±).
- **Odd part** = ½[Δs(+) − Δs(−)] minus the same for crand_big. This is the part that follows the direction of the change; a reader of the variable gives a non-zero odd part.
- **Even part** = ½[Δs(+) + Δs(−)] − crand_big. This is the part common to both directions: a response to being edited, or to edit artefacts.
- **Per reference unit** = the network's odd part divided by the odd part of the single-variable tree trained on that target. +1 means "moves like the r_1 (or D, λ(1,0.1)) classifier"; −1 means the same size, opposite sign.
- **C2ST** (classifier two-sample test) is the AUC of a boosted tree separating edited jets (one random half of the sample) from unedited jets (the other half). 0.5 means the two cannot be told apart.
- **Reweighted excess.** The same excess with each edited jet weighted by a cross-fitted, isotonic-calibrated density ratio real/edited (from the C2ST tree), so the edited sample is reweighted to look like real jets. ESS is the effective sample size fraction of those weights.
- **McNemar.** At a frozen working point (threshold = median unedited s, so half the sample is called gluon), it compares the number of jets whose call the edit flips but crand_big does not, against the reverse.
- **Holm** is the step-down correction across all 168 tests: targets × signs × 12 models × 2 generators at the main step, plus the count test.

## Procedure

### Sample

KMT quark/gluon jets (pT 500–550 GeV, R = 0.4):
- **Pythia 8:** test split.
- **Herwig 7.1:** files 18–19.

For each generator: 3,000 quark + 3,000 gluon jets, random rows, seed 7901, N ≥ 8 particles (dropped share of N < 8: 0.000 Pythia, 0.009 Herwig).

### Edit

Per jet, the variables are a_i = log pT_i and b_i = r_i/R, where r_i is the signed distance from the unedited axis along the particle's fixed azimuth about that axis. The edit is the minimum-norm Gauss–Newton step (H76 solver, batched in torch, float64) that minimises Σ(δ log pT_i)² + (δ r_i/R)² and reaches a target of

C = (log r_1, log D, log λ(1,0.1), log[LHA·width²], log λ(1,2), log (pT^D)², log ΣpT, axis y/R, axis φ/R).

One of the first three components is moved; all other components stay at their unedited values.

**Deviation from the plan (the plan's version is impossible).** D is a function of LHA and width, so D cannot move while both LHA and width are held. The D arm instead holds LHA·width², the log-space combination orthogonal to log D: D +1 % then means width +0.2 % and LHA −0.4 %. The r_1 and λ(1,0.1) arms hold LHA and width, and hence D, exactly.

**Step sizes (changed after the smoke run).** On 80 Pythia jets, D ±3 % and λ(1,0.1) +3 % at fixed others were solved for 7 % and 0 % of jets. The steps used are therefore:

| target | L (main) | M | S |
|---|---|---|---|
| r_1 | ±10 % | ±3 % | ±1 % |
| D | ±1 % | ±0.3 % | ±0.1 % |
| λ(1,0.1) | ±0.3 % | ±0.1 % | ±0.03 % |

The arms are compared through the "per reference unit" ratio, which removes the step size.

### Acceptance (numpy recomputation of every quantity on the edited particle list)

An edited jet is accepted only if all of the following hold:
- every kept quantity is within 0.1 % (LHA, width, LHA·width², λ(1,2), (pT^D)², ΣpT, jet pT) and the axis is within 0.1 % of R;
- the other two targets are within 10 % of their own L step;
- N is unchanged and the hardest particle is the same one;
- the achieved target change is within 5 % of the requested one.

In accepted jets, kept quantities move by a median of 10⁻⁶ (r_1 arm) to 2 × 10⁻⁵ (D, λ(1,0.1) arms).

### Controls

- **crand_big**, per target and per step.
- **crot**: the 3 particles moved most by the larger L edit are rotated about the axis by the azimuth whose arc length equals their displacement, then the constraints are restored.
- **Null networks:**
  - weight-permuted ParT_kin and PN_kin;
  - PN_kin trained on hash labels (pnet_kin_shuf), whose output is constant. It gives Δs ≡ 0 and is listed only for completeness.
  - The shuffled-label EFN and PFN (efn_shuf, pfn_shuf) are also constant (test logit std 0) and were not loaded.

### Count arm (as in H76, then restored to all nine constraints)

With k = max(1, round(0.1 N)):
- **N+**: the k softest particles are each split exactly into two halves at the same position.
- **N−**: k times, the softest particle is merged into its nearest-in-angle neighbour.
- **Nsm (artefact control)**: the same k splits, then k merges among particles not involved in a split. The net ΔN is 0.

Under additive footprints, Δs(N+) = b·k + a_split, Δs(N−) = −b·k + a_merge and Δs(Nsm) = a_split + a_merge. The even count part is then expected to equal Δs(Nsm)/2.

### Models and references

**Networks.** Pythia-trained:
- ParT_kin (part_kin), ParT_kin seed 1 (part_kin_s1);
- ParticleNet_kin (pnet_kin), pnet_kin_s1;
- LorentzNet;
- PFN and EFN (H2 energyflow ports).

Herwig-trained: ParT_kin (hw_part_kin), PN_kin (hw_pnet_kin).

**References.** HistGB trees trained per generator on 20k + 20k train-split jets with N ≥ 8 (seed 7902; disjoint from the test sample):
- one tree each on r_1, D and λ(1,0.1);
- "g7", a tree on (pT^D)² plus the seven angularities, without N.

### Statistics

- Bootstrap 95 % CI over jets (B = 1000), on jets accepted in all four conditions of an arm (edit±, crand_big±).
- Sign-flip p-value: 10,000 sign vectors, so the smallest attainable p is 1e-4, which becomes 0.0168 after Holm over 168 tests. H0: excess = 0. H1: excess ≠ 0.
- McNemar test at the frozen working point.
- Holm correction over all 168 tests.

## Unedited models on this sample (AUC for quark vs gluon; accuracy at s = 0)

| model | Pythia jets | Herwig jets | known value |
|---|---|---|---|
| ParT_kin | 0.905 / 0.831 | 0.802 / 0.712 | H76v 0.905 / 0.829; weaver test acc 0.827 |
| ParT_kin s1 | 0.905 / 0.829 | 0.801 / 0.713 | weaver test acc 0.827 |
| PN_kin | 0.904 / 0.831 | 0.803 / 0.716 | H76v 0.905; weaver 0.827 |
| PN_kin s1 | 0.905 / 0.828 | 0.802 / 0.714 | weaver 0.827 |
| LorentzNet | 0.919 / 0.849 | 0.792 / 0.691 | H76v 0.925 / 0.856 |
| PFN | 0.896 / 0.826 | 0.805 / 0.722 | H76v 0.898 |
| EFN | 0.886 / 0.809 | 0.796 / 0.709 | H76v 0.889 |
| ParT_kin (Herwig-trained) | 0.887 / 0.803 | 0.829 / 0.760 | H76v 0.888 / 0.834 |
| PN_kin (Herwig-trained) | 0.886 / 0.797 | 0.828 / 0.756 | H76 0.885 / 0.831 |
| ParT_kin permuted / PN_kin permuted / PN_kin shuffled | 0.530 / 0.443 / 0.500 | 0.536 / 0.467 / 0.500 | chance |
| GBDT r_1 / D / λ(1,0.1) / g7 | 0.674 / 0.647 / 0.722 / 0.846 | 0.678 / 0.636 / 0.709 / 0.789 | H76v 0.666 / 0.638 / 0.715 / 0.846 |

Every model reproduces its known value within 0.008 AUC.

## Numbers

### 1. Feasibility, edit size and distance from real jets

C2ST is the edited half vs the real half; the C2ST of real vs real is 0.504 (Pythia) and 0.505 (Herwig).

| arm (Pythia; Herwig similar) | accepted (+ / −) | median ‖δ‖ (+ / −) | C2ST + / − | C2ST crand_big |
|---|---|---|---|---|
| r_1 ±10 % | 99.7 / 97.7 % | 0.012 / 0.014 | 0.559 / 0.579 | 0.50–0.51 |
| r_1 ±3 %, ±1 % | ≥ 99.7 % | 0.004, 0.001 | 0.50–0.52 | 0.50 |
| D ±1 % | 38.5 / 91.0 % | 0.15 / 0.10 | 0.874 / 0.844 | 0.57–0.59 |
| D ±0.3 % | 75.7 / 97.7 % | – | 0.718 / 0.600 | 0.53 |
| D ±0.1 % | 91.6 / 99.4 % | – | 0.606 / 0.541 | 0.52 |
| λ(1,0.1) ±0.3 % | 16.6 / 92.0 % | 0.16 / 0.07 | 0.891 / 0.883 | 0.60 |
| λ(1,0.1) ±0.1 % | 59.2 / 96.9 % | – | 0.776 / 0.669 | 0.56 |
| λ(1,0.1) ±0.03 % | 87.2 / 99.4 % | – | 0.626 / 0.534 | 0.51 |

The largest step with C2ST ≤ 0.7 for both signs is r_1 ±10 %, D ±0.1 % and λ(1,0.1) ±0.03 %; these are the on-manifold versions. Reweighting the L edits to real jets leaves ESS 0.87–0.95 for r_1, but only 0.18–0.25 for D and λ(1,0.1).

### 2. Movement of unconstrained quantities (median relative change, accepted jets, Pythia; Herwig agrees within 0.2 %)

| edit | λ(1,¼) | λ(1,1.5) | λ(1,3) | z_1 |
|---|---|---|---|---|
| r_1 +10 % / −10 % | 0.000 / 0.000 % | 0.000 / 0.000 % | −0.003 / +0.007 % | −0.002 / +0.001 % |
| D +1 % / −1 % | −0.13 / +0.14 % | +1.27 / −1.03 % | −7.1 / +4.7 % | −0.10 / 0.00 % |
| λ(1,0.1) +0.3 % / −0.3 % | +0.38 / −0.36 % | +1.09 / −0.63 % | −7.0 / +3.1 % | −0.13 / 0.00 % |

The r_1 edit moves the hardest particle by 10 % of its radius and moves no other listed observable by more than 0.01 %. The D and λ(1,0.1) edits move λ(1,3) by 3–7 %.

### 3. r_1 ±10 % (Pythia; Herwig in brackets). Δs excess over crand_big (95 % CI); odd part; per-r_1-tree unit

| model | excess + | excess − | odd part | even part | per r_1-tree unit |
|---|---|---|---|---|---|
| ParT_kin | −0.021 [−0.026, −0.017] | +0.062 [0.056, 0.067] | −0.042 [−0.046, −0.037] (−0.047) | +0.020 | −0.82 [−0.91, −0.73] (−0.93) |
| ParT_kin s1 | −0.022 | +0.066 | −0.043 (−0.047) | +0.022 | −0.85 (−0.92) |
| PN_kin | −0.031 | +0.062 | −0.048 (−0.048) | +0.016 | −0.93 (−0.95) |
| PN_kin s1 | −0.032 | +0.065 | −0.050 (−0.051) | +0.016 | −0.98 (−1.02) |
| LorentzNet | −0.024 | +0.043 | −0.034 (−0.039) | +0.009 | −0.67 (−0.76) |
| PFN | −0.040 | +0.029 | −0.035 (−0.037) | −0.006 | −0.69 (−0.74) |
| EFN | −0.017 | +0.027 | −0.021 (−0.021) | +0.005 | −0.41 (−0.41) |
| ParT_kin (Hw-trained) | −0.014 | +0.039 | −0.026 (−0.025) | +0.013 | −0.52 (−0.50) |
| PN_kin (Hw-trained) | −0.025 | +0.043 | −0.035 (−0.037) | +0.009 | −0.68 (−0.72) |
| ParT_kin / PN_kin permuted | 0.000 | 0.000 | +0.0003 / 0.000 | 0.000 | 0.01 / 0.00 |
| GBDT r_1 | +0.051 | −0.051 | +0.051 (+0.050) | 0.000 | +1 |
| GBDT g7 (7 λ + pT^D) | +0.001 | +0.000 | +0.0002 | +0.0005 | 0.00 |

Other measurements on this arm:
- **crand_big:** moves ParT_kin by +0.013 and the EFN by +0.003.
- **crot:** moves ParT_kin by +0.037.
- **Reweighted excess (ESS 0.95):** ParT −0.022 / +0.066, unchanged.
- **Linearity:** the odd part per 1 % of r_1 is −0.0042 (10 % step), −0.0037 (3 %) and −0.0040 (1 %) for ParT_kin.
- **Calls changed (ParT_kin):** +10 % flips 2.2 % of calls vs 1.3 % for crand_big (McNemar p 1e-6); −10 % flips 2.5 % vs 1.0 % (p 2.5e-12).
- **Holm:** all 36 trained-network r_1 tests have p_Holm = 0.017 (the floor).

### 4. D and λ(1,0.1) arms (ParT_kin; excess over crand_big; odd part; per-unit of that target's tree)

| arm | C2ST + / − | excess + | excess − | even part | odd part | target-tree odd part | per unit |
|---|---|---|---|---|---|---|---|
| D ±1 %, Py | 0.87 / 0.84 | −0.077 [−0.10, −0.05] | −0.147 [−0.17, −0.13] | −0.112 | +0.041 | −0.044 | −0.94 [−1.38, −0.51] |
| D ±1 %, Hw | 0.87 / 0.85 | +0.002 [−0.02, +0.03] | −0.135 | −0.066 | +0.072 | −0.037 | −1.97 |
| D ±0.1 %, Py (on-manifold) | 0.61 / 0.54 | −0.011 | −0.023 | −0.017 | +0.007 [0.002, 0.013] | −0.005 | −1.5 [−3.4, −0.5] |
| D ±0.1 %, Hw (on-manifold) | 0.60 / 0.52 | 0.000 | −0.035 | −0.018 | +0.019 | −0.002 | −12 [−72, −4] |
| λ(1,0.1) ±0.3 %, Py | 0.89 / 0.88 | −0.138 [−0.17, −0.10] | −0.084 [−0.11, −0.05] | −0.111 | −0.032 | +0.051 | −0.62 [−1.18, −0.08] |
| λ(1,0.1) ±0.3 %, Hw | 0.89 / 0.88 | −0.081 | −0.077 | −0.079 | −0.001 | +0.048 | −0.03 [−0.68, +0.55] |
| λ(1,0.1) ±0.03 %, Py (on-manifold) | 0.63 / 0.53 | −0.026 | −0.019 | −0.023 | −0.004 [−0.010, +0.003] | +0.005 | −0.85 [−2.4, +0.6] |
| λ(1,0.1) ±0.03 %, Hw (on-manifold) | 0.64 / 0.53 | −0.010 | −0.029 | −0.020 | +0.008 [0.001, 0.015] | +0.004 | +1.9 [0.2, 4.1] |

The other networks follow the same pattern: the even part is negative (quark-ward) in every trained network, and the odd part changes sign between models, steps and generators. Examples:
- **D ±1 % (Py):** PN_kin odd −0.003, against ParT +0.041.
- **λ(1,0.1) ±0.03 %:** EFN odd +0.017 (Py) and +0.028 (Hw), against PN_kin −0.012 (Py).

**crand_big at these edit sizes.** It moves the networks gluon-ward: ParT +0.050 (D ±1 %), +0.026 (λ(1,0.1) ±0.3 %), +0.088 / +0.063 on Herwig jets.

**Reweighted to real jets** (ESS 0.18–0.25):

| arm | excess + | excess − |
|---|---|---|
| D ±1 %, Py | −0.027 [−0.073, +0.019] | −0.064 |
| D ±1 %, Hw | +0.018 [−0.046, +0.086] | −0.120 |
| λ(1,0.1) ±0.3 %, Py | −0.014 [−0.091, +0.051] | −0.087 |

**Holm.** Of 168 tests, 134 have p_Holm < 0.05. The non-significant ones are:
- the null networks;
- D+ for several networks (Herwig ParT, PN, PFN, EFN; Pythia LorentzNet, Herwig-trained ParT);
- λ(1,0.1)+ for LorentzNet and Herwig-trained ParT, and λ(1,0.1)− for the EFN.

### 5. Count arm (Δs per edit, 95 % CI; Pythia, Herwig in brackets)

| model | N+ | N− | even ½(N+ + N−) | Nsm (ΔN = 0) | Nsm/2 | even − Nsm/2 | odd per +10 % N |
|---|---|---|---|---|---|---|---|
| ParT_kin | −0.058 | −0.042 | −0.050 [−0.052, −0.048] (−0.052) | −0.206 | −0.103 [−0.106, −0.100] (−0.101) | +0.053 [0.051, 0.056] (+0.049) | −0.009 (−0.003) |
| ParT_kin s1 | −0.070 | −0.036 | −0.053 | −0.218 | −0.109 | +0.056 | −0.017 |
| PN_kin | −0.103 | −0.026 | −0.064 | −0.246 | −0.123 | +0.059 | −0.039 |
| PN_kin s1 | −0.096 | −0.020 | −0.058 | −0.247 | −0.123 | +0.066 | −0.039 |
| LorentzNet | −0.055 | −0.081 | −0.068 | −0.228 | −0.114 | +0.046 | +0.013 |
| PFN | +0.099 | −0.114 | −0.008 | −0.061 | −0.030 | +0.023 | +0.106 |
| EFN | 0.000 | −0.007 | −0.004 | −0.022 | −0.011 | +0.008 | +0.004 |
| ParT_kin (Hw) | −0.147 | +0.046 | −0.051 | −0.225 | −0.113 | +0.062 | −0.097 |
| PN_kin (Hw) | −0.266 | +0.073 | −0.097 | −0.335 | −0.167 | +0.071 | −0.170 |
| ParT_kin permuted | −0.049 | +0.047 | −0.001 | −0.050 | −0.025 | +0.024 | −0.048 |
| PN_kin permuted | −0.121 | +0.061 | −0.030 | −0.119 | −0.059 | +0.029 | −0.092 |

C2ST is 1.00 for N+ and Nsm and 0.96 for N−. Every "even − Nsm/2" test on a trained or permuted network has p_Holm = 0.017. Δs(Nsm) is about twice Δs(N+) + Δs(N−) for the trained networks (ParT −0.206 vs −0.100), so the footprints are not additive. Even the permuted networks respond to Nsm (−0.05), so the control is not a pure "footprint copy".

## Reading (scoped to what was measured)

**r_1 arm.** At fixed λ(1,β) for β = 0.1…3, (pT^D)², ΣpT, N and the axis (all within 0.01 %), moving the hardest particle radially by ±10 % changes all nine trained networks in a direction-dependent way. The response is linear across 1, 3 and 10 % steps, close to the data distribution (C2ST 0.56–0.58), unchanged by reweighting, and the same on Pythia and Herwig jets. Its sign is opposite to the r_1-only tree: −0.41 to −1.02 r_1-tree units.

Two predictions follow from this:
- "The networks are a function of the seven angularities and pT^D" predicts 0 (as g7 gives 0). It is excluded.
- "The networks read r_1 as the data r_1 classifier does" predicts +1. It is excluded.

To hold the angularities while r_1 grows, the solver moves other particles' radii and pT. The networks' response may be to that compensating change of the subleading particles rather than to r_1 itself; this arm does not separate the two.

**D and λ(1,0.1) arms.** At fixed r_1, the other angularities and the axis, these variables are stiff:
- A 1 % (D) or 0.3 % (λ(1,0.1)) change needs edits 10× larger in norm than the r_1 edit.
- The positive-sign edits succeed for only 17–39 % of jets.
- The edits move λ(1,3) by 3–7 % and sit far from real jets (C2ST 0.84–0.89).

At these sizes the trained networks move quark-ward for both signs (even part −0.05 to −0.15; ParT −0.11). The same-size random null-space edit moves them gluon-ward (+0.03 to +0.09). In the on-manifold small steps (C2ST ≤ 0.64), the even part is −0.02 and the odd part is ≤ 0.02 in size, with signs that differ between networks and generators.

The prediction "T2 dominates" (D read) and the prediction "T3 dominates" (soft-insensitive radial profile read) are not supported. "T2 and T3 at fixed r_1 give ≈ crand_big" is not supported either: they give a direction-independent quark-ward excess. Per unit of each target's own tree, ParT_kin gives −0.8 (r_1), about −1 (D, wide CI) and −0.6 to −0.9 (λ(1,0.1), wide CI, sign unstable on Herwig). These are similar magnitudes with the same sign, so no single variable is identified; this is the plan's "D30 stays at 'radial profile'" outcome.

**Count arm.** The direction-independent part of the count response (ParT −0.050) is smaller than half the response to the net-zero split-and-merge control (−0.103). The plan's alternative, "the even response exceeds the control (networks read split/merge footprints beyond an artefact)", is excluded in its stated form. The control is non-additive, though, and also moves weight-permuted networks, so it bounds the artefact rather than matching it. The odd (counting) part is ≤ 0.04 per +10 % N for Pythia-trained ParT/PN and ≤ 0.017 for ParT_kin. As in H76, it is larger for the Herwig-trained networks (−0.10 / −0.17) and for the PFN (+0.11).

## Issues

1. **Step sizes deviate from the plan.** The plan's ±3 % / ±1 % steps were infeasible for D and λ(1,0.1) (smoke run). The plan's "hold LHA and width" in the D arm is mathematically impossible; LHA·width² was held instead.
2. **Selection bias in the main D and λ(1,0.1) arms.** Their main (L) steps keep only 38 % and 16 % of Pythia jets. Statistics use jets accepted in all four conditions of an arm (n = 2,257 and 973), a selected subset (jets far from the moment bound).
3. **crand_big is not neutral at large norm.** It moves the networks gluon-ward by up to +0.09 for the D and λ(1,0.1) edits. The excess therefore mixes "edit vs random" with a random-rearrangement effect. Raw Δs are in h79_stats.json.
4. **Weak reweighting.** For the off-manifold edits the density-ratio reweighting has ESS 0.18–0.25, so the reweighted numbers have wide intervals.
5. **Minimum sign-flip p.** With 10,000 permutations the minimum Holm-adjusted p is 0.017, so many tests sit at that floor.
6. **Compensating moves not separated.** The r_1 arm does not separate "leading-particle radius" from "the compensating move of the other particles". A follow-up could move r_1 with the compensation confined to particles at large or small radius.
7. **Null networks.** pnet_kin_shuf (and the unloaded efn_shuf, pfn_shuf) are constant-output networks, so their zero response carries no information beyond "untrained".
8. **Figure.** The example jet in results/physics/h79_radial.png has a very collimated core (r_1 = 0.009 R). Its integrated profile changes in steps of up to 30 % of jet pT as the hardest particle crosses radii. The right panel shows the per-sign excess for Pythia jets at the main step.
