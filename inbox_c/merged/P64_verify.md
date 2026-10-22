# P64 verification: ParticleNet's learned kNN graphs and prongs (loop C)

Verifier job `c-p64-verify` (complete, 13 min, NVIDIA A10). Script `interp/p64v_verify.py` (imports P64's own `run`, `purity`, `Graph`, `excl_kt`, `flips`, `mcnemar`, `holm` from `interp/p64_pn_knn_prongs.py`, so the code under test is the same). Results `$AI_RESULTS/part/p64_verify.json` (local copy `results/part/p64_verify.json`; smoke `p64_verify_smoke.json`).

## Conclusion

ParticleNet's learned neighbour graphs (layers 2 and 3, rebuilt from learned features) link a particle to its own prong less often than the plain (η, φ) graph of layer 1: about 0.63 against 0.79 of the neighbours for W/Z jets, and this reproduces on a fresh sample. They link it more often than an untrained network does, and that excess is mostly specific to the prong split, not to any spatial split of the jet. P64's intervention reading does not hold once a better control is used. Forcing the learned graphs to use only cross-prong neighbours changes W-vs-Z calls more often than cutting the same graphs along a rotated or angular split of the same jet does. For top-vs-QCD calls the excess over random rewiring is reproduced by cutting along any spatial split. So the prong split does matter for the W/Z decision; P64's random-rewiring control hid this. Whether this amounts to a JEDI-linear-like "object" stage is still open, and the new neighbours in the prong cut are somewhat farther away in feature space than in the matched control.

## What was checked in the code

- Layer indices: EC2 = `edge_convs[1]` and EC3 = `edge_convs[2]`. The graph is replaced before `get_graph_feature_v1`, so the trained edge MLP recomputes the messages. The own forward pass equals weaver's (max |Δlogit| 0.0, P64 log).
- Padded particles: these are the zero-padding slots up to 128. They get a 1e9 coordinate shift as in weaver. `purity` counts only real neighbours, excludes self, and averages only over real particles. The rewiring tiers place padded slots last.
- Rows: `rng.choice(N, 3000)` gives random rows (sorted only for reading), not the first k. The fresh sample also uses random rows.
- Build: `part_core.build` uses η = jet_eta + sign(jet_eta)·Δη, so the η-mirror bug is not present.
- Random-graph expectation (n_s − 1)/(n − 1): this is the exact mean purity of a uniformly random 16-neighbour set (n = real particles, n_s = size of the particle's subjet). Correct.
- Edits that change more than intended:
  - R2 ("cross-prong only") puts the 16 nearest cross-prong particles in the list. When a particle's other prong(s) hold fewer than 16 particles, the rest of the list is filled with same-prong particles. This is common for W/Z jets: mean subjet sizes are 17.7 / 14.4 for W. So R2 graphs are not purely cross-prong. R1 has the mirror case.
  - The separation edit scales jet pT by λ (0.73–1.39). The report says so.
- No bug changes the numbers. The problems are in reporting and in the control, below.

## Reporting errors in P64

1. The P64 summary says the W/Z ratio R2/R3r2 is "0.99–1.07 in every arm and both models (Holm ≥ 0.33)". R2/R3r2 is the fraction of calls changed by cross-prong-only rewiring divided by the fraction changed by random prong-blind rewiring of the same number of edges. The statement is false for PN_kin with only the EC2 graph rewired: P64's own JSON has W/Z 1.35 [1.28, 1.42], Holm 1.1e-32. The fresh sample gives 1.32 [1.25, 1.40].
2. The report's Conclusion says "1.5 to 3 times" for top/QCD. The 3× figure is against the nearest-prong-blind control R3n. Against the random control the range is 1.5–1.9, as the summary says. The two texts disagree.
3. Figure panels (d) and (e) show EC3-only and EC2+EC3. They leave out the EC2-only arm, which is where the PN_kin W/Z effect is.

## Fresh-sample rerun (same code, new rows)

- **Sample.** val_5M `*_123` (P64 used `*_124`): QCD, Z, W, top. 2,500 random rows per class, `default_rng(6490)`, N = 10,000.
- **Prong labels.** Exclusive-kT as in P64: N = 2 subjets for W/Z, 3 for top/QCD.
- **Unedited 10-class accuracy on this 4-class sample.** PN_full 78.57 %, PN_kin 70.44 % (P64's sample: 79.46 / 70.50; P1 reference sample: 84.56 / 70.36).

Edge purity (unweighted mean over particles, then jets; [95 % bootstrap over jets, 2,000 resamples]):

| | W/Z | top | QCD |
|---|---|---|---|
| random graph | 0.543 | 0.370 | 0.381 |
| EC1 | 0.795 [0.792, 0.799] | 0.816 [0.812, 0.819] | 0.599 |
| EC3 PN_full | 0.625 [0.623, 0.628] | 0.587 [0.584, 0.591] | 0.484 |
| EC3 PN_kin | 0.624 | 0.554 | 0.476 |
| EC3 untrained PN_full (seed 6464) | 0.569 | 0.444 | – |

- EC3 − EC1 (PN_full): W/Z −0.170 [−0.173, −0.167], top −0.229 [−0.232, −0.225]. P64: −0.169, −0.227.
- Calls changed relative to random rewiring, R2/R3r2 [95 % CI]:

| model | layers rewired | W/Z | top/QCD | argmax |
|---|---|---|---|---|
| PN_full | EC2 + EC3 | 0.95 [0.91, 1.00] | 1.56 [1.30, 1.90] | 1.23 |
| PN_full | EC2 only | 1.00 | 1.82 [1.55, 2.19] | 1.66 |
| PN_full | EC3 only | 1.06 | 1.25 [0.97, 1.60], Holm 1 | 1.02 |
| PN_kin | EC2 + EC3 | 1.01 | 1.62 [1.44, 1.83] | 1.49 |
| PN_kin | EC2 only | 1.32 [1.25, 1.40], Holm 9e-22 | 1.58 | 2.08 |
| PN_kin | EC3 only | 1.04 | 0.93 | 0.99 |

- R1/R3r1 for W/Z, i.e. within-prong-only rewiring divided by count-matched random rewiring: 0.56–0.88. Within-prong-only graphs change fewer W/Z calls than random rewiring, as in P64.
- Every P64 purity and R2/R3r2 number reproduces within its interval.

## Missing control: a geometry-matched fake partition

**Why it is needed.** R3r replaces the same number of edges with random particles anywhere in the jet. R2 instead cuts the graph along a spatial boundary and brings in the nearest particles across it. R2 vs R3r therefore mixes two things: (i) cutting along the prong boundary specifically, and (ii) cutting along any spatially coherent boundary. The right reference for (i) is the same R2/R3r2 procedure applied with a partition that is spatially coherent but does not follow the prongs.

**Fake partitions.** Both are computed per jet from the unedited (Δη, Δφ) positions relative to the jet's pT centroid.

- *rot*: each particle gets the exclusive-kT label of the original particle nearest to its own position rotated by −90° about the centroid. This is the prong partition turned by 90°. Its region sizes are not matched to the prongs: the mean largest-group share is 0.71, against 0.56 for the prongs.
- *sec*: particles are ordered by angle about the centroid, starting at the direction of the highest-pT subjet. They are then cut into consecutive angular sectors with exactly the true subjet sizes. For W/Z this is the half-plane split along the prong axis, which cuts each prong through its centre.
- Agreement of the particle labels with the prong labels: 0.50 (rot) and 0.44 (sec), compared with 0.5 for a random 2-way split (W/Z).

**Arms.** R1x, R2x, R3r1x and R3r2x are P64's R1, R2, R3r1 and R3r2 run with the fake labels (x = rot or sec); each random arm is count-matched to its cut arm.

**Statistic.**

$$\rho=\frac{f(\mathrm{R2})/f(\mathrm{R3r2})}{f(\mathrm{R2}x)/f(\mathrm{R3r2}x)}$$

Here $f$ is the fraction of calls changed relative to the unedited network. ρ is the ratio of ratios, with a paired bootstrap over jets (2,000 resamples) and a two-sided bootstrap p. Holm runs over the 48 ratio-of-ratios tests: 2 models × 3 layer sets × 4 call types × 2 fake partitions.

**Hypotheses.**

- H0: the extra effect of R2 over random rewiring comes from cutting along any spatial boundary, so ρ = 1.
- H1 (prong-specific): ρ > 1.

**Predictions of P64's reading.** If prong identity mattered only for top/QCD, ρ(top/QCD) > 1 and ρ(W/Z) = 1.

**Result.**

- Fake-partition cuts relative to their random controls: W/Z 0.67–0.89 (below 1 in all 12 cases).
- ρ for W/Z calls: 1.14–1.72. Holm < 0.05 in 11 of 12 tests. For example, PN_full EC2+EC3: 1.21 [1.12, 1.32] (rot) and 1.20 [1.13, 1.28] (sec). PN_kin EC2+EC3: 1.35 [1.26, 1.45] and 1.43 [1.35, 1.52].
- ρ for argmax: 1.13–1.84 (12 of 12, Holm 0.024; this is the bootstrap floor).
- ρ for top/QCD: 0.86–1.31. Holm < 0.05 in 1 of 12 (PN_kin EC2, rot: 1.21 [1.09, 1.34]). PN_full EC2+EC3: 1.31 [1.07, 1.62], Holm 0.14 (rot), and 1.17 [0.93, 1.49] (sec). PN_kin EC2+EC3: 1.03 and 1.11.
- So the pattern P64's reading predicts is excluded. The W/Z pattern is H1's, and the top/QCD pattern is mostly H0's.

**Purity excess against the same partitions.** Trained minus untrained EC3 purity (PN_full):

| partition | W/Z | top |
|---|---|---|
| prong | +0.057 [0.055, 0.058] | +0.143 [0.140, 0.146] |
| sec | +0.009 [0.008, 0.010] | +0.057 [0.056, 0.059] |
| rot | +0.015 | +0.070 |

So most of the trained-over-untrained excess at W/Z is specific to the prong partition (+0.05 of +0.06). For top it is about +0.08 of +0.14. PN_kin is similar: prong +0.056 / +0.110, sec +0.010 / +0.044.

**Caveats of this control.**

- *Edge count.* The fake cuts replace fewer edges than R2. With EC2+EC3 rewired: R2 0.53/0.55, sec 0.42/0.43, rot 0.38/0.39. This happens because the graphs are less pure with respect to the fake partitions. The count-matched random arm normalises this only if the effect of rewiring grows in proportion to the number of edges replaced.
- *Feature distance.* We measured the mean feature-space rank of the inserted neighbours (rank 1 = nearest other particle; the original lists average 8.5) in jets with ≥ 40 particles, PN_full, EC2+EC3. R2 gives 21.7 against 25.6 for its random control, a gap of 3.9. For sec the figures are 18.0 against 22.1 (gap 4.1), and for rot 20.4 against 22.9 (gap 2.5). The gap between each cut and its random control is similar across partitions, but R2's absolute ranks are higher than R2sec's. Part of ρ > 1 can therefore still be feature distance. An exactly distance-matched control was not run.

## Corrected reading

- **Purity.** P64's purity reading reproduces on a fresh sample. EC2 and EC3 graphs are less prong-pure than EC1, more than random, and more than untrained or channel-permuted PN. Measured against size-matched angular sectors, most of the trained-over-untrained excess is specific to the prong partition.
- **Against random rewiring.** Cross-prong-only rewiring changes argmax and top/QCD calls 1.2–2.1× as often as random rewiring of the same number of edges. For W/Z the ratio is 0.95–1.07, except with only EC2 rewired in PN_kin, where it is 1.32–1.35.
- **Random rewiring is not a neutral reference.** Cutting the same graphs along a spatially coherent non-prong partition changes W/Z calls only 0.67–0.89× as often as random rewiring. Relative to these partitions, the prong cut changes W/Z calls 1.14–1.72× and argmax calls 1.13–1.84× more often. For top/QCD the prong cut's excess is mostly reproduced by the fake cuts (ρ 0.86–1.31; 1 of 12 tests after Holm).
- **Revised claim.** The statement "prong identity matters only for top/QCD and not for W/Z" is not supported. The data show a prong-specific dependence for W/Z and argmax calls, and a mostly generic spatial-cut dependence for top/QCD.

## Terms

- Edge purity, exclusive-kT subjets, R1/R2/R3r/R3n/R4, McNemar and Holm are as defined in P64_report.md.
- New terms for the glossary:
  - **Fake partition (rotated / angular-sector).** A split of a jet's particles into regions that are spatially coherent but not aligned with the prongs. Used as the control for "does cutting along prong boundaries matter".
  - **Ratio of ratios ρ.** The changed-call ratio of the prong cut to its random control, divided by the same ratio for a fake-partition cut.
  - **Feature-space rank.** The position of a particle in another particle's distance-sorted list in the layer's learned features.
