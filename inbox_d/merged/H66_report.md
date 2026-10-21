# H66: top-tagger object-count response on generated jets matched to Top Landscape (not run: no GPU job admitted)

Node H66 (loop D, round D14). Claim D24 v2. Script `interp/h66_top_count_onmanifold.py` (written; import and edit-helper check passed). No result exists yet.

## Conclusion

There are no measurements for H66. The analysis script is written and its jet-editing helpers pass a local check. The GPU scoring job never started:
- Twice, Kubernetes placed the pod on node k8s-chase-ci-01.calit2.optiputer.net, which refused it with "GPU is lost" (UnexpectedAdmissionError).
- My attempt to exclude that node in a third launch was blocked by the session's permission system.
- I could not confirm the state of the cluster jobs afterwards. Job `anrunw-ai-d-h66-score` was deleted. A delete of `anrunw-ai-d-h66-score2` was in the blocked command, so that job may still exist in Failed state.

D24 v2 keeps its open caveat: the H61 Delphes jets can be told apart from Top Landscape jets (C2ST AUC 0.74). The question of whether Top Landscape's own data favours QCD as N rises at fixed (m, τ32, mW) is still unanswered.

## What is needed to run it

1. Add `k8s-chase-ci-01.calit2.optiputer.net` to the node exclusion list in `part_job.sh`, or relaunch until another node is chosen. `part_job.sh` is outside this loop's write scope.
2. `./sync.sh && ./part_job.sh d-h66-score3 'python -m interp.h66_top_count_onmanifold --step score'`. This writes `h66_scores.npz` and `h66_tltrain.npz`. Expected wall time is about 40 min.
3. `./job.sh d-h66-stats 2 4 'python -m interp.h66_top_count_onmanifold --step stats'`. This writes `h66_run.json` and `h66_count_onmanifold.png`.

## Procedure implemented (for the next worker)

### Samples

No new generation is needed. Half A is H61 (`data/top/gen_h61`, workers 6101–6110) and half B is H61v (`gen_h61v`, workers 6201–6210). Both have event-paired V0 (Monash) and Hz+ (aLund 0.79, bLund 0.88, σ 0.247 GeV) samples of 5k top + 5k QCD jets. These two independent halves replace the planned fresh workers 6601–6610.

### Count arm (cnt) and decomposition

For each V0/Hz+ pair, the cnt arm brings the V0 jet to exactly the Hz+ object count:
- d > 0: the d softest objects are each split into two pT/2 halves, 0.01 apart (±0.005 about the parent) in (η, φ).
- d < 0: the |d| nearest pairs in the softer half are merged.

The Hz+ shift is then decomposed as Hz+ = count part (cnt − V0) + rest (Hz+ − cnt). The ParT_kin decomposition comes from this.

### Matching

Density-ratio weights w = p_TL/p_V0 per class come from a Platt-calibrated HistGB classifier. Its features are:
- N, m, τ32, mW, pT^D;
- the pT fractions of the constituents ranked 1, 2, 3, 5, 10, 20 and 40;
- the counts with pT < 1, 1–2 and 2–5 GeV;
- girth, the pT fractions within ΔR 0.1, 0.2 and 0.4, the counts in ΔR bins, and jet pT and |η|.

The weights are capped at 30 and cross-fitted: the classifier is fit on one half against TL test rows T1 (rng 6611, 10k per class) and applied to the other half.

The check is a weighted C2ST on the held-out half against a disjoint set of TL rows, T2 (10k per class). C2ST (classifier two-sample test) is the out-of-fold AUC of a classifier trained to separate the two samples, so 0.5 means they cannot be told apart. It is computed on four feature sets:
- the weighting features;
- substructure not used in the weights (τ21, mSD, the subjet pair masses);
- the networks' own scores;
- N alone.

A T1-vs-T2 null C2ST is computed alongside. The tempered weights w^α (α = 0, 0.25, 0.5, 0.75, 1) give Δs as a function of C2ST, which tests the off-manifold alternative.

### Metrics on the matched set

- Weighted Δs, the mean change in top log-odds for Hz+, cnt and rest, with a 95 % bootstrap CI over jet pairs (B = 2,000).
- Network − EFN for each network, with bootstrap-z p-values.
- Weighted accuracy and AUC of every unedited model on matched V0.
- At the weighted 50 % signal-efficiency threshold: ε_S before and after, exact McNemar (unweighted), and a weighted Δε_S with a bootstrap p.

### Edits on real Top Landscape test jets

The edits use 5k + 5k of the T2 rows. Each edit is applied to the same jets as its control.
- **split_rand k = 2, 4, 8:** the first k of 8 randomly drawn constituents, each split into two collinear halves. N rises by k; jet four-momentum and IRC-safe content stay fixed.
- **jit8 (control):** the same 8 constituents moved by 0.005 in the same directions, with N fixed.
- **ss8:** the 8 softest constituents split.

For each edit the script reports Δs, the dose slope per added object, and McNemar at the TL 50 % top-efficiency threshold. Controls are the EFN and six weight-permuted ParT copies (seeds 0–5). Holm correction runs across 6 networks × all arms × 2 classes.

### Physics reference

The data's d log-odds/dN at fixed substructure is computed on Top Landscape train rows (rng 6612, 60k per class; N = jet_nparticles) in four ways:
1. Unconditional logistic regression on N.
2. Fixed-effect logistic regression: label ~ cell(10 m × 8 τ32 × 8 mW quantile bins) + βN. It is fit on all jets and inside 140 < m < 200 GeV, with bootstrap CIs.
3. A cross-fitted GBDT on (m, τ32, mW, N), read as the finite-difference slope in N on held-out top and QCD jets.
4. The same GBDT with τ21, mSD and pT added.

The figure compares these slopes with the networks' per-object slopes.

The second generation arm (Delphes 3.4-era card) is not implemented. It was conditional on weighting failing to reach C2ST ≤ 0.6, which is not yet known.

## Unedited model accuracy

Not measured (no run).
