# P4 — ParticleNet: translation, WTA-axis and route tests (claims C3, D4), 2026-10-19

Job `anrunw-ai-c-p4-run` (RTX 3090, 13 min), `python -m interp.p4_particlenet && python -m interp.p4_particlenet --plot`;
smoke job `anrunw-ai-c-p4-quick` (30 jets/class) before it. Script `interp/p4_particlenet.py`; network code `interp/pn_net.py`
(verbatim copy of weaver-core 0.4.17 `ParticleNet.py` plus the official `example_ParticleNet.get_model` settings); data configs
`interp/pn_data/JetClass_{kin,full}.yaml` (copies of the official ones, checked field by field against the input code at start).
Results `results/particlenet/p4_particlenet.json`; figures `results/particlenet/p4_wz_shift.png`, `results/particlenet/p4_routes.png`;
logs `logs/c-p4-run.log`, `logs/c-p4-quick.log`. No training.

## Conclusion
ParticleNet reacts to a rigid shift of all particles relative to the jet axis almost as much as ParT does, even though the shift leaves
the jet's invariant mass unchanged: true W jets called Z rise from 12 % to 28 % (ParT_full: 12 % to 31 %). The response enters through
the per-particle distance-to-axis input ΔR and through the parts of the first graph layer that read each particle on its own. The
neighbour graph that ParticleNet builds from particle positions carries almost none of it. The axis-relative mass response is
therefore a property of the shared JetClass input format (per-particle coordinates measured from the jet axis), not of ParT's attention.
Still uncertain: why the network's mass read-out moves by only about two-thirds of what an axis moment ∝ Σ pT r² predicts, and whether a
model without axis-relative per-particle inputs (L-GATr, P5) behaves differently.

## Procedure
- Models: ParticleNet_full (17 per-particle features) and ParticleNet_kin (7), official released JetClass weights
  (`/j-jepa-vol/aaron-interp/models/particlenet`), loaded `strict=True`. ParticleNet input: points (Δη, Δφ) for the layer-1 kNN graph
  (k = 16), per-particle features (same list and scaling as ParT, including deltaR = √(Δη² + Δφ²), deta, dphi) through a BatchNorm,
  3 EdgeConv blocks (layer-1 message $h_\Theta(x_j - x_i) + h_\Phi(x_i)$, then 2 more conv+BN+ReLU, mean over the 16 neighbours, plus
  a shortcut $\mathrm{BN}(W_{sc} x_i)$; layers 2–3 build their kNN graph in the learned feature space), mean pooling, FC 256 → 10.
  Padded to 128 slots as weaver. The split forward used for the route tests reproduces weaver's `ParticleNet.forward` exactly (max |ΔL| 0.0).
- ParT_full / ParT_kin run on the same jets and edits (call metrics only) for a paired comparison.
- Samples: P1 rows 3,000–4,000 of each class's first JetClass test file (10,000 jets, `part_core.sample`, rebuilt with `part_core.build`;
  primary) and fresh rows 6,000–7,000 (10,000 jets, as P2v).
- Edits (`part_core.build`, pT and particle masses fixed): `phi±0.06` every real particle Δφ ± 0.06 rad (exact rotation, true Δlog m
  0.00000, max |·| 1e-4); `rand0.06` per jet a random direction θ (seed 3), (Δη, Δφ) + 0.06(cos θ, sin θ) (true Δlog m 0.0000);
  angles × 1.05 / × 1.1 (true Δlog m +0.048 / +0.094); matched calibration angles × k, k = 1.0766 (rows 3,000) / 1.0773 (rows 6,000),
  chosen by secant so that the mean true Δlog m on the read-out test half equals the shift's mean ½Δlog(M0·M2) (0.0729 / 0.0735).
- Read-out: ridge regression of log m from the 256-d last hidden layer (input of the last Linear), fitted on the unedited jets
  (seed-1 50/50 split, α on the last quarter of the train half); held-out R² 0.837 (full) / 0.880 (kin) on rows 3,000; 0.833 / 0.881 on
  rows 6,000. Δlog m̂ = mean over the test half.
- Route split (edits phi0.06, rand0.06, ang1.1): (a) full edit; kNN graph of layer 1 / of all 3 layers fixed to the unedited graph; only
  the kNN points edited; feature groups deltaR, (deta, dphi), all three edited alone ("only", points unedited) or fed unedited
  ("removed"); layer-1 terms: centre $h_\Phi(x_i)$, edge $h_\Theta(x_j - x_i)$, shortcut $W_{sc} x_i$, each edited alone (rest and
  points unedited) or each kept unedited (rest edited). Share = variant / (a).
- WTA test (D4): (Δη, Δφ, ΔR) recomputed about the stored axis (identity), the E-scheme axis, the hardest particle and the WTA axis
  (P1v's code; 4-vectors unchanged); rows 3,000–4,000.
- Controls: re-run of the unedited inputs (same batches) and with batch size 97 (float noise floor); calls changed on the 8 classes other
  than W/Z (untouched-class control); the random-direction shift of equal size; angles × 1.1 real-mass control; ParT on the same jets.
- Statistics: 2,000 bootstrap resamples over jets stratified by class (seed 2), percentile 95 % intervals. H0inv (invariant mass): mean
  Δlog m̂ = 0. H0M2 (pure axis moment): mean Δlog m̂ = c · mean ½Δlog(M0M2), c = read-out gain under the matched calibration edit;
  calibrated ratio = obs / (c · pred). Two-sided bootstrap p (normal p from the bootstrap s.e. when at the 1/2001 floor), Holm over the 24
  ratio tests. McNemar (exact) for W→Z edit vs unedited and calls changed edit vs noise. Route shares: H0 share = 0 and share = 1, Holm per
  model/sample/edit (60 tests each).

## Numbers
Step 0 — accuracy (unedited, rebuilt jets; raw jets differ by ≤ 0.02 points):

| model | rows 3,000–4,000 | rows 6,000–7,000 | reference |
|---|---|---|---|
| ParticleNet_full | 84.57 % | 84.47 % | 84.4 % (ParT paper, JetClass test, full) |
| ParticleNet_kin | 70.35 % | 69.56 % | no released number found in the clone README |
| ParT_full | 85.93 % | 86.31 % | 85.9 % on P1 rows (P1) |
| ParT_kin | 74.46 % | 73.99 % | 74.5 % on P1 rows (P1) |

Noise floor: re-run 0 calls changed (max |ΔL| 0); batch size 97: ≤ 0.04 % of calls (one jet with max |ΔL| 0.78, a kNN tie).

Step 1 — rows 3,000–4,000 (rows 6,000–7,000 in brackets where they differ by more than 1.5 points). W→Z = fraction of 1,000 true W jets
called Z; Δ(ℓW − ℓZ) on true W jets.

| model | edit | calls changed % [CI] | other 8 classes changed % | W→Z % unedited → edited [CI] | McNemar W→Z gained/lost, p | Δ(ℓW−ℓZ) on W |
|---|---|---|---|---|---|---|
| PN_full | φ + 0.06 | 8.1 [7.6, 8.6] | 4.7 | 12.3 → 28.5 [25.8, 31.3] (12.8 → 26.7) | 172/10, 3e-39 | −0.70 [−0.8, −0.6] |
| PN_full | φ − 0.06 | 8.5 | 5.1 | 12.3 → 26.7 | 156/12, 4e-33 | −0.7 |
| PN_full | random dir. 0.06 | 8.4 | 4.9 | 12.3 → 27.1 [24.4, 29.9] | 166/18, 4e-31 | −0.7 |
| PN_full | angles × 1.1 | 13.0 | 7.6 | 12.3 → 41.1 [38.2, 44.1] | 306/18, 9e-69 | −1.27 |
| PN_kin | φ + 0.06 | 12.1 [11.5, 12.7] | 9.7 | 18.4 → 29.0 [26.3, 31.8] (20.7 → 29.1) | 137/31, 4e-17 | −0.44 |
| PN_kin | random dir. 0.06 | 12.5 | 9.7 | 18.4 → 29.1 | 146/39, 1e-15 | −0.47 |
| PN_kin | angles × 1.1 | 20.6 | 15.5 | 18.4 → 45.9 [42.9, 49.1] | 320/45, 3e-52 | −0.93 |
| ParT_full | φ + 0.06 | 8.1 [7.6, 8.6] | 4.3 | 11.7 → 31.0 [28.3, 33.9] (12.2 → 33.5) | 203/10, 7e-48 | −1.1 |
| ParT_full | random dir. 0.06 | 8.3 | 4.6 | 11.7 → 29.7 | 190/10 | −1.1 |
| ParT_full | angles × 1.1 | 12.6 | 7.8 | 11.7 → 37.1 | 272/18 | −1.5 |
| ParT_kin | φ + 0.06 | 19.6 [18.8, 20.3] | 15.8 | 16.2 → 37.3 [34.4, 40.3] | 242/31, 1e-41 | −0.8 |
| ParT_kin | angles × 1.1 | 18.7 | 13.5 | 16.2 → 45.5 | 324/31 | −1.1 |

Z→W falls under every mass-unchanged shift (PN_full 18.9 → 11.6 %, PN_kin 23.0 → 16.3 %). Every edit's calls-changed fraction differs
from the noise control (McNemar p < 1e-180).

Read-out and ratio tests (prediction ½Δlog(M0M2) = +0.073 for every shift):

| model, rows | edit | Δlog m̂ [CI] | gain c (calibration) | raw ratio | calibrated ratio [CI] | Holm p, H0inv / H0M2 |
|---|---|---|---|---|---|---|
| PN_full, 3,000 | φ + 0.06 | +0.0317 [0.0300, 0.0334] | 0.682 [0.668, 0.696] | 0.435 | 0.637 [0.602, 0.673] | 4e-284 / 5e-55 |
| PN_full, 3,000 | φ − 0.06 | +0.0317 | 0.682 | 0.434 | 0.637 [0.600, 0.674] | 1e-265 / 9e-54 |
| PN_full, 3,000 | random dir. | +0.0316 | 0.682 | 0.435 | 0.638 [0.602, 0.675] | 6e-279 / 8e-53 |
| PN_full, 6,000 | φ + 0.06 | +0.0318 | 0.654 | 0.433 | 0.662 [0.625, 0.704] | 2e-258 / 8e-44 |
| PN_kin, 3,000 | φ + 0.06 | +0.0355 [0.0335, 0.0374] | 0.714 [0.701, 0.728] | 0.487 | 0.681 [0.641, 0.722] | 5e-283 / 5e-40 |
| PN_kin, 3,000 | φ − 0.06 | +0.0368 | 0.714 | 0.505 | 0.707 [0.669, 0.747] | 5e-309 / 1e-34 |
| PN_kin, 6,000 | φ + 0.06 | +0.0336 | 0.709 | 0.457 | 0.645 [0.604, 0.685] | 1e-237 / 7e-49 |

All 24 tests reject at Holm p < 1e-33. For reference (R1, rows 8,000–9,000, per-particle-only φ edit): ParT_full calibrated ratio 0.88
[0.84, 0.91], ParT_kin 1.53 [1.47, 1.58].

Step 2 — share of the full edit's Δ(ℓW − ℓZ) on true W jets, rows 3,000–4,000 [95 % CI] (rows 6,000 within 0.05 unless noted):

| variant | PN_full φ + 0.06 | PN_full angles × 1.1 | PN_kin φ + 0.06 | PN_kin angles × 1.1 |
|---|---|---|---|---|
| layer-1 kNN graph fixed to unedited | 1.000 (graph identical: overlap 1.000) | 1.000 (overlap 1.000) | 1.000 | 1.000 |
| all 3 kNN graphs fixed | 0.975 [0.959, 0.990] | 0.999 [0.995, 1.003] | 0.991 [0.976, 1.007] | 0.995 |
| only kNN points edited | 0.000 | 0.000 | 0.000 | 0.000 |
| only deltaR edited | 1.095 [1.066, 1.123] | 0.939 [0.929, 0.949] | 1.369 [1.314, 1.426] | 1.011 |
| only (deta, dphi) features edited | 0.000 [−0.015, 0.016] | 0.032 | −0.036 [−0.063, −0.012] | −0.010 |
| deltaR fed unedited (rest edited) | 0.000 | 0.026 | −0.036 | −0.017 |
| only shortcut $W_{sc}x_i$ edited | 1.023 [0.989, 1.062] | 0.871 | 1.012 [0.973, 1.055] | 0.852 |
| only centre term $h_\Phi(x_i)$ edited | 0.179 [0.158, 0.202] | 0.151 | 0.293 [0.266, 0.322] | 0.234 |
| only edge term $h_\Theta(x_j - x_i)$ edited | 0.038 [0.028, 0.050] | −0.027 | 0.084 [0.070, 0.101] | −0.069 |
| shortcut kept unedited | 0.193 [0.172, 0.217] | 0.110 | 0.321 [0.292, 0.352] | 0.133 |
| centre term kept unedited | 1.064 | 0.830 | 1.099 | 0.756 |
| edge term kept unedited | 0.975 | 1.043 | 0.956 | 1.113 |

kNN-graph overlap with the unedited graph under φ + 0.06 (fraction of the 16 neighbours kept, real particles): layer 1 1.000, layer 2
0.947, layer 3 0.912 (full); 0.937 / 0.913 (kin). Random-direction shifts give the same shares within 0.02 (kin deltaR-only 1.33). For
φ + 0.06, "only (deta, dphi)" equals "deltaR fed unedited" to all digits, as it must when the points do not matter. Holm-adjusted
p < 0.001 for share = 0 on every row with |share| > 0.05; share = 1 is not rejected for the layer-1 graph, all-graphs-fixed (except
PN_full rows 6,000, 0.963 [0.946, 0.979]) and shortcut-only rows under the shifts.

Step 3 — calls changed with (Δη, Δφ, ΔR) about another axis, rows 3,000–4,000 [CI] (McNemar vs noise p < 1e-300 for hardest/WTA):

| axis | PN_full | PN_kin | ParT_full | ParT_kin |
|---|---|---|---|---|
| stored (identity) | 0.00 | 0.00 | 0.00 | 0.00 |
| E-scheme (recomputed) | 0.15 [0.08, 0.23] | 0.26 [0.17, 0.36] | 0.08 [0.03, 0.14] | 0.17 [0.09, 0.26] |
| WTA | 19.0 [18.3, 19.8] | 25.8 [25.0, 26.6] | 11.7 [11.2, 12.3] | 34.1 [33.3, 35.0] |
| hardest particle | 21.2 [20.5, 22.0] | 28.4 [27.6, 29.3] | 14.0 [13.3, 14.6] | 38.0 [37.2, 38.9] |

Accuracy about the WTA axis: PN_full 84.6 → 74.2 %, PN_kin 70.3 → 59.9 %. The WTA axis equals the hardest particle in 58 % of jets.

## Reading (scoped to these measurements)
- Competing predictions: (input format) PN W→Z rises by a similar amount to ParT and the response goes through the absolute-coordinate
  per-particle features and the $x_i$ terms, not the graph; (architecture) PN's relative edges make the response small (ratio near 0)
  and route it through the graph. Measured: PN_full W→Z +16 points (ParT_full +19), PN_kin +11 (ParT_kin +21); calibrated read-out ratio
  0.63–0.71 (H0 "ratio 0" rejected); the graph route (layer-1 graph identical under any rigid shift or scaling; all three graphs fixed keeps
  0.96–1.01 of the response) carries ≤ 4 %. The architecture-specific prediction is excluded on all three measures.
- Within the per-particle inputs, ΔR carries the response (deltaR alone 0.94–1.41 of it; with deltaR fed unedited |share| ≤ 0.06). The deta and
  dphi features, which shift rigidly, carry |share| ≤ 0.06: ParticleNet does not use the absolute position along η or φ for this response, only
  the distance from the axis. Same input as ParT_full (P1v: deltaR unedited removes 92–100 %); ParT_kin used dphi as well (0.65).
- Within layer 1, the terms that read each particle alone (shortcut, centre term) carry the response; the shortcut alone gives
  0.85–1.05, the centre term 0.15–0.30, and the two do not add exactly (sum 1.0–1.3). The translation-invariant edge difference carries ≤ 0.10
  (non-zero because ΔR differences are not shift-invariant). The prediction named the centre term; the measurement places most of it on the
  shortcut, which is also a per-particle (x_i) term.
- The read-out moves by 0.63–0.71 of the axis-moment prediction, both "invariant mass" and "pure M2 axis moment" rejected; ParT_full was
  0.88 in R1 on other rows. Under the WTA axis, ParticleNet changes 19 % (full) / 26 % (kin) of calls, between ParT_full (12 %) and
  ParT_kin (34 %).

## Issues
- First launch failed at start (files.json read empty, a transient on the volume, presumably a concurrent sync); relaunched unchanged.
- No released ParticleNet_kin accuracy found locally to compare with; 70.3 % is 4 points below ParT_kin on the same rows.
- The read-out is a ridge on one hidden layer; its gain c is 0.65–0.71, so raw ratios (0.43–0.51) understate the calibrated ones.
- The route-share intervals resample jets with the read-out fixed (P2v caveat applies to the m̂ column, not to the logit column used here).

## Explainer entry (for site/explain.py, Paper C)
**Question.** ParT's sense of jet mass moves when every particle is shifted relative to the jet axis, although the invariant mass does
not change (C3). Is that because of the transformer, or because every JetClass tagger receives the same axis-relative inputs?

**Background.** JetClass gives each particle its offsets from the jet axis, $\Delta\eta_i$, $\Delta\phi_i$ and
$\Delta R_i = \sqrt{\Delta\eta_i^2 + \Delta\phi_i^2}$. A rigid shift $\Delta\phi_i \to \Delta\phi_i + \delta$ leaves
$m^2 = (\sum_i p_i)^2$ unchanged but raises the axis moment $M_2 = \sum_i p_{T,i}\,\Delta R_i^2$; a network that estimates mass from
$\sqrt{M_0 M_2}$ predicts $\Delta \log \hat m = \tfrac12 \Delta\log(M_0 M_2) \approx 0.073$ at $\delta = 0.06$. ParticleNet sees the same
per-particle features but mixes particles by EdgeConv on a $k$-nearest-neighbour graph:
$x_i' = \mathrm{ReLU}\big(W_{sc}x_i + \tfrac1k\sum_{j\in\mathcal N(i)} h(\,h_\Phi(x_i) + h_\Theta(x_j - x_i)\,)\big)$; the graph and
the difference $x_j - x_i$ of the shifted coordinates are unchanged by a rigid shift.

**Method.** 10,000 JetClass test jets (1,000 per class); edits: $\Delta\phi + 0.06$, a random direction of the same size, and angles
$\times 1.1$ (a real 9.9 % mass increase). Measure W→Z calls, the change of $\ell_W - \ell_Z$, and a calibrated read-out of $\log m$
from the last hidden layer. Then feed the edit to one input route at a time (ΔR, Δη/Δφ features, graph points, and the three layer-1
terms) and measure the share of the response it carries.

**Result.** Figure `results/particlenet/p4_wz_shift.png`: true-W jets called Z go from 12 % to 28 % in ParticleNet_full (ParT_full 12 %
to 31 %). Figure `results/particlenet/p4_routes.png`: ΔR alone carries the response; the graph carries ≤ 4 %; the per-particle shortcut
term carries most of it. The read-out moves by 0.64 of the $\sqrt{M_0M_2}$ prediction (95 % CI 0.60–0.67).

**Checks.** The split forward pass equals weaver's own forward exactly; re-running unchanged inputs changes no call; fresh rows
6,000–7,000 give W→Z rates, calibrated ratios and route shares inside the rows-3,000 intervals or within 0.05 of them; random-direction shifts match the φ shift.

**Why it matters.** The axis-relative mass response is shared by a transformer and a graph network that use the same inputs, so it comes
from the JetClass input format rather than from attention. A model without axis-relative per-particle inputs (L-GATr) is the next test.
