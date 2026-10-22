# P64 — ParticleNet's learned kNN graphs and prongs (loop C, round C14, exploratory; claims A8 / C5, proposes C24)

Status: done (job c-p64-run, one debug fix to a log format string). Script `interp/p64_pn_knn_prongs.py` (the figure is drawn by the same module, `--plot`; no separate p64_plot.py).
Results: `results/particlenet/p64_pn_knn.json`, figure `results/particlenet/p64_pn_knn.png` (smoke: `p64_pn_knn_smoke.*`).

## Conclusion

ParticleNet's second and third neighbour graphs, which are rebuilt from learned features, link a particle to its own subjet (prong) less often than the first graph, which is built from (η, φ) distance: about 0.63 of the neighbours against 0.79 for W/Z jets. The learned graphs are still more prong-pure than random or untrained graphs. Rewiring them so every neighbour comes from another prong changes top-vs-QCD calls 1.5 to 3 times more often than prong-blind rewiring of the same number of edges, but changes W-vs-Z calls about as often. So we did not find a prong-grouping "object stage" like JEDI-linear's in ParticleNet's learned graphs: the graphs lose prong structure layer by layer, and prong identity matters to the call only for top/QCD and only moderately. The size of the cross-prong effect may partly come from the new neighbours being far away in feature space rather than from their being in another prong; this was not separated.

## Question and design

- **Question.** ParticleNet (PN) has three EdgeConv layers (EC1–EC3). Each one updates every particle from its k = 16 nearest neighbours. EC1 picks neighbours by distance in (Δη, Δφ). EC2 and EC3 pick them by distance in the previous layer's learned features (64-d and 128-d). If the learned graphs gather particles prong by prong (the two quarks of a W/Z, the three of a top), they would act as an object stage like the one found in JEDI-linear (loop A/B, claim A8).
- **Predictions.** Object-stage hypothesis H1: EC2/EC3 edge purity above both the random graph and EC1, no excess in untrained PN, and cross-prong-only rewiring (R2) changes W/Z/top calls by more than 3× a prong-blind control (R3). Alternative H0: purity at the EC1 or untrained level, and R1, R2, R3 change calls by similar amounts.
- **Terms.** *Subjet / prong*: we cluster the jet's particles with exclusive-kT into exactly N subjets, N = 2 for W and Z and N = 3 for top and QCD. Exclusive-kT repeatedly merges the pair with the smallest d_ij = min(pT,i², pT,j²)·ΔR_ij², with four-vector (E-scheme) addition and ΔR in rapidity and φ. There is no beam distance, so every particle ends up in a subjet. *Edge purity* of particle i in layer l: the fraction of i's real neighbours (self excluded) that are in i's own subjet. Jet purity is the mean over particles (unweighted, or pT-weighted). *Random graph*: the exact expectation for a uniformly random graph with the same degree, (n_s − 1)/(n − 1), where n_s is the size of i's subjet and n the number of particles in the jet. *McNemar*: an exact binomial test on the jets where only one of two paired arms changed the call. *Holm*: step-down family-wise correction.

## Procedure

- **Models.** We used ParticleNet_full (17 inputs) and ParticleNet_kin (7 inputs) with the official JetClass weights and code from `interp/pn_net.py`. Our forward pass matches weaver's ParticleNet.forward to max |Δlogit| 0.0. Controls for purity: untrained PN_full with two initialisations (torch seeds 6464, 6465), and channel-permuted PN_full (seed 6466). In the channel-permuted model, the input-channel axis of every 1×1-conv/linear weight is shuffled, which keeps the weight values but destroys the learned wiring between layers.
- **Sample.** val_5M files `*_124` of QCD (ZJetsToNuNu), Z (ZToQQ), W (WToQQ) and top (TTBar), with 3,000 random rows per class (numpy `default_rng(6464)`), so N = 12,000. The plan named `_126`, but val_5M only has `_120` to `_124`. Every jet has ≥ N real particles, so all 12,000 are valid. Mean particles per jet: QCD 35.0, Z 33.7, W 32.1, top 51.0. Mean subjet sizes: W 17.7 / 14.4, top 18.4 / 17.1 / 15.4.
- **Unedited accuracy.** On the P1 reference sample (N 10,000): PN_full **84.56 %**, PN_kin **70.36 %**, which reproduces P50's 84.56 / 70.36. On this 4-class sample, 10-way argmax: PN_full 79.46 %, PN_kin 70.50 %.
- **Purity observation.** The graphs are read in the real forward pass. We report a 95 % bootstrap over jets (2,000 resamples) and paired bootstrap p-values for EC_l − EC1, EC_l − random and trained − control. We also report the overlap: the fraction of EC_l's neighbours that are also EC1 neighbours.
- **Interventions.** All weights are fixed. Only the neighbour index of EC2, of EC3, or of both is replaced; the messages are recomputed by the trained edge network.
  - R1: within-prong only. Every cross-prong neighbour is replaced by the nearest same-prong particle in that layer's feature distance.
  - R2: cross-prong only. Every within-prong neighbour is replaced by the nearest cross-prong particle.
  - R3r1 / R3r2: prong-blind random rewiring. For each particle, the same number of edges as R1 / R2 is replaced: that many randomly chosen original neighbours are dropped and that many random other particles of the jet are added.
  - R3n1 / R3n2: as R3r, but the added particles are the nearest non-neighbours in feature space. This matches R1/R2's "nearest" rule without regard to prong.
  - R4: the EC1 (η, φ) graph is used in place of the EC2/EC3 graph.
  - Edges replaced: R1 0.22–0.27, R2 0.47–0.55, R4 0.33–0.43 of all edges.
- **Metrics.** We count the fraction of calls changed relative to the unedited network: argmax over the 10 classes (all jets), W vs Z (the sign of l_W − l_Z on true W/Z jets), top vs QCD (the sign of l_t − l_QCD on true top/QCD), and W vs QCD. Accuracy and the 4×10 confusion matrix are in the JSON. We test R1 vs R3r1/R3n1 and R2 vs R3r2/R3n2 with McNemar on the per-jet "call changed" indicator, with Holm over 96 tests (arms × layer sets × call types × models). Ratios of changed-call fractions get a bootstrap CI.
- **Physics edit.** On true W jets (N 3,000), each of the two kT prongs is moved rigidly so that its centroid's distance from the jet's pT centroid scales by s ∈ {0.6, 0.8, 1.25, 1.5}; the pT centroid stays fixed. Then every particle's pT and mass are scaled by λ = m₀/m′, which keeps the invariant mass fixed (max |Δ ln m| 4.5e-4). As a result, jet pT scales by λ (mean 1.39 at s = 0.6, 0.73 at s = 1.5). A 90° rotation of all particles about the pT centroid serves as the separation-fixed control. We measure the change in EC1/EC3 purity using the unedited jet's subjet labels; particle order is unchanged (checked).

## Numbers

Purity (unweighted; [95 % CI]), PN_full. pT-weighted values differ by ≤ 0.03.

| layer | W, Z | top | QCD |
|---|---|---|---|
| random graph | 0.543 | 0.369 | 0.383 |
| EC1 (η–φ) | 0.794 [0.791, 0.797] | 0.815 [0.812, 0.818] | 0.594 [0.588, 0.601] |
| EC2 | 0.665 [0.662, 0.667] | 0.654 [0.651, 0.657] | 0.504 [0.499, 0.509] |
| EC3 | 0.625 [0.622, 0.627] | 0.588 [0.585, 0.592] | 0.482 [0.478, 0.486] |
| EC3 PN_kin | 0.624 | 0.553 | 0.476 |
| EC3 untrained s6464 / s6465 | 0.568 / 0.566 | 0.443 / 0.436 | – |
| EC3 channel-permuted | 0.571 | 0.464 | – |

- **EC3 − EC1, PN_full.** W/Z −0.169 [−0.172, −0.167], top −0.227 [−0.230, −0.224]; p < 0.001.
- **EC3 − random.** W/Z +0.081 [0.079, 0.083], top +0.219 [0.216, 0.222].
- **Trained − untrained s6464, EC3.** W/Z +0.056 [0.055, 0.058], top +0.145 [0.143, 0.148]. The other controls are similar: +0.053 to +0.152.
- **Overlap with the EC1 graph.** EC2 0.65–0.75, EC3 0.56–0.70.
- **Robustness.** With the other N (W/Z clustered into 3 subjets, top/QCD into 2), EC3 is still above random but below EC1 in the same way: W/Z 0.470 vs random 0.377; top 0.715 vs 0.558.

Calls changed, EC2 + EC3 both rewired (% [95 % CI]):

| arm | PN_full W/Z | PN_full top/QCD | PN_full argmax | PN_kin W/Z | PN_kin top/QCD |
|---|---|---|---|---|---|
| R1 within-prong | 6.8 [6.2, 7.4] | 1.8 | 9.7 | 9.2 | 4.5 |
| R2 cross-prong | 17.5 [16.6, 18.5] | 3.1 [2.6, 3.5] | 23.0 | 26.5 | 6.3 |
| R3r2 random (R2 count) | 16.9 [16.0, 17.9] | 2.1 [1.7, 2.4] | 17.8 | 25.7 | 4.0 |
| R3n2 nearest blind (R2 count) | 14.6 | 1.6 | 15.2 | 22.2 | 3.5 |
| R4 EC1 graph | 14.4 | 2.8 | 17.4 | 18.2 | 10.3 |

Accuracy (PN_full): R1 77.5 %, R2 69.2 %, R3r2 73.6 %, R3n2 75.2 %, R4 74.6 % (unedited 79.5 %).

Ratios R2 / R3r2, i.e. the changed-call fraction under cross-prong rewiring divided by that under random rewiring of the same number of edges; McNemar and Holm p-values:

- **PN_full, both.** W/Z 1.04 [0.99, 1.08] (Holm 1). Top/QCD 1.50 [1.28, 1.80] (Holm 4e-5). Argmax 1.29 [1.25, 1.33] (Holm 1e-58).
- **PN_full, EC2 only.** W/Z 0.99 [0.92, 1.07]. Top/QCD 1.90 [1.63, 2.26] (Holm 2e-14). Argmax 1.61 [1.55, 1.67].
- **PN_full, EC3 only.** W/Z 1.07 [1.02, 1.13] (Holm 0.33). Top/QCD 1.49 [1.14, 2.03] (Holm 0.25).
- **PN_kin, both.** W/Z 1.03 [0.99, 1.07] (Holm 1). Top/QCD 1.57 [1.40, 1.76]. Argmax 1.49 [1.45, 1.52].
- **PN_kin, EC3 only.** Every ratio is between 0.84 and 1.04 (Holm ≥ 0.96).
- **R2 / R3n2 (nearest prong-blind control).** 1.19–3.0. The largest is PN_full EC2 top/QCD at 2.99 [2.46, 3.73]; W/Z is 1.20–1.66.
- **R1 / R3r1, W/Z.** 0.65–0.90. Within-prong graphs change fewer W/Z calls than random rewiring (PN_full both: Holm 2e-6).

Physics edit (W jets, PN_full; change in unweighted purity vs unedited [95 % CI]):

| edit | EC1 | EC3 PN_full | EC3 PN_kin | EC3 untrained | Δ(l_W − l_Z) PN_full |
|---|---|---|---|---|---|
| s = 0.6 (prongs closer) | −0.075 [−0.077, −0.073] | −0.024 [−0.025, −0.022] | −0.026 | −0.008 | +0.21 [0.17, 0.25] |
| s = 0.8 | −0.030 | −0.011 | −0.012 | −0.005 | +0.28 |
| s = 1.25 | +0.022 | +0.012 | +0.011 | +0.007 | −0.94 [−0.97, −0.90] |
| s = 1.5 (prongs apart) | +0.034 [0.033, 0.035] | +0.023 [0.021, 0.024] | +0.019 | +0.016 | −1.87 [−1.93, −1.82] |
| rotation 90° | 0.000 | +0.001 | −0.000 | 0.000 | −0.41 [−0.44, −0.38] |

## Reading

- **Purity.** The learned EC2 and EC3 graphs keep a smaller share of same-subjet neighbours than the (η, φ) graph they start from, in both trained models and for every class and both subjet counts. The share drops from EC2 to EC3. It stays above the random-graph level and 0.05–0.15 above untrained and channel-permuted PN, so training adds some prong structure relative to an untrained feature map of the same inputs. It does not add enough to reach the geometric EC1 level. This is H0's pattern for the comparison with EC1, and H1's pattern for the comparison with untrained networks.
- **Interventions.** Restricting the learned graphs to cross-prong neighbours changes more argmax and top-vs-QCD calls than prong-blind rewiring of the same number of edges (×1.3–1.9 vs random, up to ×3.0 vs the nearest-blind control). It changes W-vs-Z calls by the same amount as random rewiring (ratio 0.99–1.07). The planned H1 threshold (> 3× R3) is not reached against the random control in any arm. R1 (within-prong) changes fewer calls than random rewiring.
- **Possible confound.** R2's nearest cross-prong particles are farther in feature space than the dropped neighbours, so part of the R2 > R3n gap can come from feature distance rather than from prong membership. R3r, which adds far particles, is the closer match on this point, and against R3r the W/Z ratio is about 1.
- **Physics edit.** Moving the W prongs closer at fixed mass lowers EC1 purity by 0.075. EC3 purity falls by 0.024, and in the untrained network by 0.008. EC3 purity tracks the η–φ geometry more weakly than EC1 in both trained and untrained networks, so this damping does not by itself show that training made the graph follow prongs. The edit also changes jet pT by the factor λ, and the W-vs-Z score moves with both edits. A 90° rotation leaves purity unchanged but moves l_W − l_Z by −0.41. This is consistent with the orientation (axis) sensitivity found in P15/P50.

## Files

- `interp/p64_pn_knn_prongs.py` (module; `--smoke`, `--plot`).
- `results/particlenet/p64_pn_knn.json`: all purity rows, the alternative-N rows, arms with confusion matrices, McNemar/Holm tests, controls, separation scan and figure-jet per-particle purity.
- `results/particlenet/p64_pn_knn.png`: panels (a, b) per-particle purity on one top jet (pT 701 GeV, m 162 GeV; the first top jet in sample order with ≥ 40 particles and ≥ 6 per subjet) for EC1 and EC3; (c) purity per layer vs random / untrained / channel-permuted; (d, e) calls changed for EC3-only and EC2 + EC3 rewiring; (f) purity vs prong separation. In (f), the EC1 curves of all three models coincide, since EC1 is the same η–φ graph.
- Log: `/j-jepa-vol/aaron-interp/logs/c-p64-run.log`.

## Explainer entry (for site/explain.py; P64_EXPLAINER)

- **Question.** Do ParticleNet's learned neighbour graphs group the particles of a jet by prong, which would make them an "object" stage like JEDI-linear's?
- **Background.** ParticleNet updates each particle $i$ in each EdgeConv layer as
  $x_i^{(l+1)}=\mathrm{ReLU}\big(\mathrm{sc}(x_i^{(l)})+\tfrac1k\sum_{j\in\mathcal N_l(i)}h_l(x_i^{(l)},x_j^{(l)}-x_i^{(l)})\big)$, with $k=16$.
  $\mathcal N_1(i)$ = the 16 nearest particles in $(\Delta\eta,\Delta\phi)$; $\mathcal N_{2,3}(i)$ = the 16 nearest in the learned features $x^{(1)}$, $x^{(2)}$.
  We define prongs by exclusive-$k_T$ clustering into $N$ subjets ($N=2$ for W/Z, $3$ for top/QCD): merge the pair with the smallest $d_{ij}=\min(p_{T,i}^2,p_{T,j}^2)\,\Delta R_{ij}^2$ until $N$ remain.
- **Method.** Edge purity is $\pi_i^{(l)}=\frac{|\{j\in\mathcal N_l(i): s(j)=s(i)\}|}{|\mathcal N_l(i)|}$, averaged over particles and jets. It is compared with the random-graph expectation $(n_{s(i)}-1)/(n-1)$, with EC1, and with untrained and channel-permuted networks.
  Interventions replace $\mathcal N_{2,3}$ by within-prong-only (R1) or cross-prong-only (R2) neighbour sets, or by prong-blind rewiring of the same number of edges (R3r random, R3n nearest), or by $\mathcal N_1$ (R4). All weights are fixed. We count changed calls and test them with McNemar and Holm.
  Physics edit: the W prongs are moved apart or together about the pT centroid by a factor $s$, with all four-momenta rescaled by $m_0/m'$ to hold the jet mass fixed.
- **Result (figure p64_pn_knn.png).** For W/Z jets, $\bar\pi$ = 0.79 (EC1) → 0.67 (EC2) → 0.63 (EC3), against 0.54 for a random graph and 0.57 for untrained PN. For top jets: 0.82 → 0.65 → 0.59, against 0.37 random and 0.44 untrained. Cross-prong-only graphs change top-vs-QCD calls 1.5–1.9× and W-vs-Z calls 1.0× as often as random rewiring of the same number of edges.
- **Checks.** The own forward pass equals weaver's (0.0). P1 accuracy reproduces 84.56 % / 70.36 %. The purity pattern holds with the other number of subjets and with pT weighting. Untrained networks have two seeds, and the channel-permuted network is included. Under the separation edit the jet mass is held to 4.5e-4 in ln m.
- **Why it matters.** The loop asks whether big taggers share JEDI-linear's cuts → objects → mass stages (claim A8). In ParticleNet, the learned graphs carry less prong structure than the plain η–φ graph, and the prong partition matters for the call only partly (top/QCD, not W/Z). So an object stage in ParticleNet, if one exists, is not found as a prong-wise neighbour graph.
- **Glossary additions.** Edge purity; exclusive-kT subjet (E-scheme, no beam distance); degree-matched random graph; R1/R2/R3r/R3n/R4 graph rewiring; channel-permuted network.
