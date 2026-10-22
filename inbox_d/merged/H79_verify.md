# H79 verify (loop D): which radial variable the quark–gluon networks read

Verifier: skeptic pass on interp/h79_radial_variable_qg.py, results/physics/h79_stats.json, h79_run.json, h79_radial.png and inbox_d/H79_report.md.

**Status:** the code review and control review are done. **The fresh-sample rerun did not run**:
- The first launch of `d-h79-verify` landed on node hcc-chase-shor-c4715.unl.edu. CUDA failed to initialise there ("CUDA unknown error"), so the job fell back to CPU while holding an idle GPU. I deleted that job after 4 min.
- The relaunch landed on the same node and failed at admission (UnexpectedAdmissionError).
- Further cluster launches (a launcher copy that excludes that node, and a plain `./part_job.sh d-h79-verify2`) were denied by the session's permission classifier.

The verification script is ready: interp/h79_verify.py (synced to the volume). Run it with:
1. `./part_job.sh d-h79-verify2 'python -m interp.h79_verify --step run'`
2. then `./job.sh d-h79-verst 'python -m interp.h79_verify --step stats'`

That run uses 2000 q + 2000 g per generator, seed 7931, random rows, N ≥ 8, and writes `$AI_RESULTS/physics/h79_verify.json`. Recommendation: add hcc-chase-shor-c4715.unl.edu to the node exclusion list in part_job.sh.

## Conclusion

Moving the hardest particle 10 % further from the jet axis, while holding the seven angularities, (pT^D)², the particle count, ΣpT and the axis, shifts every trained quark/gluon network in a consistent direction (toward quark; ParT −0.04 gluon log-odds per +10 %). So the networks depend on more than those angular-spread observables; the code reproduces this correctly. Any function of the held observables alone must give exactly zero under this edit.

The report's further reading does not follow from the data shown. It says the networks "do not read the leading-particle radius as the data trend does", but it compares against a tree trained on r_1 alone. That tree carries the *marginal* trend (wider jets also have larger r_1). The edit holds the angularities, so the trend that applies is r_1's effect *at fixed angularities*. Its sign was not measured and may well be negative. Whether the networks' negative sign agrees or disagrees with the data is therefore open, as is whether they respond to the hardest particle or to the compensating move of the other particles.

## Code review (no defect that changes the numbers)

Checked items:
- **Axis and coordinates.**
  - Observables (`ext`) and the solver (`Solver.C`, H76 `_axis`) use the same E-scheme axis: the rapidity and azimuth of the massless four-vector sum.
  - The model build (`h5_qg_counting.kinematics`) uses the jet pseudorapidity, which is offset from the rapidity by a per-jet constant that depends on m/pT. λ(1,2) and jet pT are held in the r_1 arm, so the offset changes negligibly between the unedited and edited jet.
  - ParT's sign(η) flip of Δη is a reflection, so it leaves every angularity unchanged.
- **Padding.** Padded slots get pT = 0 through the mask `m` in pack/coords. Every sum is multiplied by `m`, and `unpack` returns only the real particles. No padded particle enters an observable or the model.
- **Same jets.** Model logits, reference scores and observables are all computed from the same edited lists `Le`.
- **Sampling.** H76.load_sample takes random rows with `rng.choice` per class from N ≥ 8; there is no first-k. The KMT test split is not class-ordered by row in a way that matters here, because selection is random.
- **Conservation claims.** ΣpT, jet pT and the axis are constrained and rechecked in numpy, as are λ(1,0.1), LHA, width, λ(1,2) and (pT^D)².
  - Accepted r_1 jets: kept quantities change by median 10⁻⁶. The unconstrained λ(1,0.25), λ(1,1.5), λ(1,3) and z_1 move ≤ 7 × 10⁻⁵ relative (h79_stats.json `arms.r1L.shift`).
  - Energy is not conserved, and the report does not claim it (massless E changes with the rapidity moves).
- **Working point.** The McNemar threshold is the median unedited s over all jets, frozen before the edits. This is consistent with the report.
- **Odd and even parts.** crand_big's ± signs share one random vector, so its odd part has mean 0 by construction; the odd-part test is valid. Holm runs over 168 tests, with a 1e-4 sign-flip floor that gives 0.0168 after Holm, as reported.
- **Minor: hardest-particle identity.** `i1` is fixed at pack time, and acceptance requires the hardest particle to stay the same. Correct.
- **Minor: generator offsets in the references.** The Pythia train lists (efp_train_lists) are (pT, Δη, Δφ) relative to the stored axis, while the test lists are absolute. A constant rapidity shift is a longitudinal boost for massless particles, so the angularities are unchanged. Correct.

## Control review

1. **Missing conditional reference (main issue).**
   - The "per r_1-tree unit" (−0.82 for ParT_kin) divides by the odd response of s_r1, a HistGB on r_1 alone. Under the r_1 edit, the held angularities are exactly the variables that make r_1's marginal and conditional effects differ: r_1 is positively correlated with width and LHA.
   - The prediction of "the network reads r_1 the way the data does" at fixed angularities is the odd response of a classifier on (angularities + r_1). That is g7r1 or gfull in h79_verify.py. Its sign is unknown.
   - So "opposite to the data trend" and the exclusion of "+1 tree unit" exclude only "the networks follow the marginal r_1 trend", and that is not the physically relevant prediction.
   - Comparing the arms "in tree units" (r_1 vs D vs λ(1,0.1)) uses three different marginal trees, so equal magnitudes in those units carry no information about which variable is read.
2. **Leading particle vs compensation.** This is acknowledged in the report as issue 6. h79_verify.py adds two splits of the edit:
   - Decomposition: hard-only (only the hardest particle's δ applied) and rest-only (only the others' δ), with networks and g7 on each.
   - An r_2 ±10 % arm: the second-hardest particle moved at fixed r_1 and the other constraints, with its own crand_big.

   Both are cheap and ready to run.
3. **Generic sensitivity.** crot (a matched-arc rotation at fixed constraints) moves ParT_kin by +0.037, the same size as the r_1 odd part (−0.042). The networks respond to any rearrangement at fixed angularities. The odd part is the specific signal, and its sign is consistent across the 1 / 3 / 10 % steps and both generators. The raw per-sign shifts are asymmetric:
   - ParT_kin Δs(+10 %) = −0.008 and Δs(−10 %) = +0.075 on Pythia; the crand_big mean is +0.013.
   - So "outward → quark-like" is the linear (odd) part. The raw outward shift alone is close to zero.
4. **D and λ(1,0.1) arms.** The report's reading is appropriately limited: off-manifold, selected subsets, odd parts that change sign. No change.
5. **Count arm.** The control is non-additive and moves permuted networks; the report says so. The statement "even part < Nsm/2" is a bound, not a match. No change.
6. **Null networks.** Weight-permuted ParT_kin and PN_kin give an odd part of +0.0003 and 0.000, so they are adequate. The shuffled-label network is constant and uninformative (stated).

## Figure (results/physics/h79_radial.png)

- **Left panel.** It shows the cumulative pT-share change for one Pythia quark jet with N = 18 and r_1 = 0.009 R. The curves are step functions of up to ±30 % of jet pT, caused by the hardest particle crossing the radius grid. The legend values (r_1 0.009 → 0.010 → 0.008; D and λ(1,0.1) unchanged in the r_1 edit) match the edit. The panel shows that the r_1 edit is a reshuffle at fixed moments, but at this r_1 it is hard to read. A jet near the median r_1 (0.045 R) would be clearer.
- **Right panel.** It shows the per-sign excess over crand_big at the main (L) steps, including the off-manifold D ±1 % and λ(1,0.1) ±0.3 % arms. The title does not say that those two arms are off-manifold (C2ST 0.84–0.89).
- The numbers in the figure match h79_stats.json.

## Corrected reading

**r_1 arm.** At fixed λ(1,β) for β = 0.1…3, (pT^D)², ΣpT, N and the E-scheme axis, moving the hardest particle radially by ±10 % gives a nonzero direction-dependent (odd) response in all nine trained networks:
- ParT_kin −0.042 [−0.046, −0.037] per +10 % (Pythia), −0.047 (Herwig);
- linear across 1, 3 and 10 % steps;
- C2ST 0.56–0.58.

This excludes "the networks are a function of the seven angularities, (pT^D)², N and ΣpT" (prediction 0).

The sign is opposite to a tree trained on r_1 alone, which excludes only "the networks follow the marginal r_1 trend". Whether the networks agree with the data's trend at fixed angularities is untested: the reference that is needed is a classifier on angularities + r_1 (h79_verify.py, not yet run). The arm also does not separate the hardest particle's move from the compensating moves of the other particles.

**D and λ(1,0.1) arms** (unchanged). They can only be moved alone by off-manifold edits on selected jets. The odd parts change sign across models, steps and generators, and the per-tree-unit comparison across arms does not identify a variable.

**Count arm** (unchanged, with its caveats).

## Confidence

0.5 that the reading is correct as stated:
- The existence and direction of the r_1 response, and the D, λ(1,0.1) and count statements, are supported by the code and the numbers.
- The interpretive clause "opposite to the data trend / do not read r_1 as the data does" uses a marginal reference for a conditional edit and is not supported.
- The fresh-sample rerun was not executed.
