# P44: L-GATr internals: channel grades, spurion route, mass read-out (loop C, round C10, exploratory)

## Conclusion
In our L-GATr, a network built so that its internal channels transform properly under Lorentz transformations, the class decision does not rest mainly on the pure-scalar part (grade 0) of its multivector channels. Blanking that part at any block costs at most 3 accuracy points. The decision rests on the separate auxiliary scalar channels and, in the first three blocks, on the vector and axial-vector channels; blanking the auxiliary and grade-0 channels together after block 3 drops accuracy from 84.8 % to 25–38 %. A boost along the beam that leaves every pair mass unchanged moves about 2 % of calls, and only through the two inputs that fix the lab frame: the time reference vector and the authors' frame-dependent per-particle features (log E). When those are boosted too, the change falls to the floating-point level. What is still uncertain: the mass is linearly readable from the invariant channels just as well in an untrained copy, and the one-direction mass swap (DAS) does not separate the trained network from the untrained one.

## Setup
- Model: L-GATr from job anrunw-ai-c-p5-lgatr2 (authors' jctagging config, 12 blocks, 2M JetClass jets). The training job was still running at 06:00 UTC on 2026-10-21, so all three analysis jobs used the **interim checkpoint model_run0_it154999.pt**, which is also the latest checkpoint on disk. The training job was not touched. Check: accuracy on the P1 sample (rows 3000–3300 of each `_100` file, N 3,000) is **84.27 %**, the same as P39.
- Sample: JetClass val_5M `<class>_122.root`, which is not among L-GATr's training or validation files (selection.json). 1,000 random rows per class (`default_rng(4444)`), N = 10,000, shuffled once so that every batch mixes classes. Inputs were rebuilt with `part_core.build` (as in P5/P39).
- Unedited: accuracy **84.80 %** [84.08, 85.50] (stored inputs 84.82 %). Per-class recall: QCD 0.772, Hbb 0.918, Hcc 0.824, Hgg 0.771, H4q 0.853, Hqql 0.971, Zqq 0.669, Wqq 0.776, Tbqq 0.945, Tbl 0.981. The largest confusions are Z → W 185/1000, W → Z 122, H4q → Hgg 77, Hgg → H4q 96 and QCD → Hgg 74. Full 10×10 matrix: `base.confusion` in p44_main.json.
- Plumbing checks: the split forward (blocks run one at a time with hooks) matches the wrapper's own forward to max |Δlogit| 0.20. Identical inputs run twice give mean |Δlogit| 0.0036, max 1.0, and 0.02 % calls changed. This run-to-run floor comes from GPU non-determinism in the xformers block-diagonal attention and is the float floor quoted below.
- Jobs: c-p44-run5 (main, GTX 1080 Ti, 60 min), c-p44-ctl7 (controls, 86 min), c-p44-das (DAS + untrained, L4, 41 min), c-p44-an2 (CPU analysis). c-p44-run/-ctl/-run3/-run4/-ctl3/-ctl5/-ctl6 were rejected at admission on k8s-chase-ci-01, where the device plugin reported "GPU is lost". c-p44-run2/-ctl2 were evicted when k8s-haosu-12 was tainted. None of these failures came from the code. Loading the model takes 20–25 min per job (imports and hydra from CephFS).
- Script: interp/p44_lgatr_grades.py. Results: results/lgatr/p44_main.json/.npz, p44_ctl.npz, p44_das.json, p44_lgatr_grades.json, figure results/lgatr/p44_lgatr_grades.png.

## Definitions
- **Multivector**: an element of the 16-dimensional space-time (Clifford) algebra. **Grades**: 0 = scalar (1 component), 1 = four-vector (t, x, y, z; 4), 2 = bivector (oriented planes tx … yz; 6), 3 = axial vector (trivector; 4), 4 = pseudoscalar (1). Each L-GATr token has 16 hidden multivector channels (256 numbers) plus 32 **auxiliary scalar channels s**. Grade 0 and s are Lorentz invariant ("invariant", 48 per token). Grades 1–4 change under boosts and rotations ("covariant", 240 per token).
- **Equivariance**: if every input multivector, particles and references alike, is transformed by the same Lorentz transformation Λ, every hidden multivector transforms by Λ and the invariant channels and logits stay the same: $f(\Lambda p, \Lambda s_{\rm ref}) = f(p, s_{\rm ref})$.
- **Spurion**: a fixed reference multivector, fed in as an extra token, that breaks the symmetry. Here there are two: the x–y-plane bivector (the beam reference, unchanged by rotations about the beam and by boosts along it) and the time direction $t = (1,0,0,0)$, which is unchanged by rotations but not by boosts. A third symmetry-breaking route is the authors' per-particle scalar inputs: log pT, log E, log pT/pT_jet, log E/E_jet, Δφ, Δη, ΔR. Of these, log E and log E/E_jet change under a longitudinal boost.
- **Mean / resample ablation**: replace the chosen components of every token with their sample mean, computed per token type (global, beam, time, particle), or with the values from another jet (particle k takes the donor's particle k mod n).
- **DAS (distributed alignment search) / IIA (interchange-intervention accuracy)**: learn one unit direction u in a 48-dimensional channel space. Run the network on a base jet, but replace the base's projection on u with the projection from a source jet, here the same jet with all angles scaled by 1.1 or 1/1.1. IIA is the fraction of jets whose call changes under that dilation for which the intervened run gives the source's call.
- **McNemar**: exact binomial test on jets whose correctness changes (lost vs gained). **Holm**: step-down family-wise correction.

## (a) Grade ablations
Procedure: at the output of block k (k = 0…11), one component group of all tokens is mean-ablated or resampled. The groups are s (32), g0 (16), g1 (64), g2 (96), g3 (64), g4 (16), inv = g0 + s (48) and cov = g1–g4 (240), all on N 10,000.

Control: on the first 2,000 jets of the shuffled sample (unedited 85.1 %), the same number of components is drawn at random from the 288 per token and mean-ablated, 50 draws per block and size. The effect is control accuracy minus group accuracy per jet, with a bootstrap 95 % CI. The p-value is a paired z-test, Holm-corrected over 96 tests (12 blocks × 8 groups). The 2,000-permutation sign-flip p is also stored. Its resolution limit makes the Holm floor 0.048.

Accuracy after mean ablation (%), N 10,000 (control mean for the same block and size in brackets, N 2,000):

| group | b0 | b2 | b3 | b5 | b8 | b11 |
|---|---|---|---|---|---|---|
| s (32) | 53 (17) | 50 (30) | 40 (51) | 55 (68) | 62 (78) | 70 (84) |
| g0 (16) | 84 (18) | 84 (36) | 84 (54) | 84 (69) | 83 (79) | 82 (85) |
| g1 (64) | 14 (16) | 24 (26) | 57 (43) | 73 (64) | 81 (74) | 85 (82) |
| g2 (96) | 83 (16) | 83 (20) | 84 (35) | 84 (57) | 85 (69) | 85 (78) |
| g3 (64) | 19 (16) | 21 (26) | 58 (43) | 70 (64) | 83 (74) | 85 (82) |
| g4 (16) | 84 (18) | 84 (36) | 84 (54) | 83 (69) | 83 (79) | 83 (85) |
| inv (48) | 47 (16) | 42 (29) | 25 (48) | 31 (66) | 29 (77) | 26 (83) |
| cov (240) | 16 (16) | 24 (10) | 47 (10) | 64 (12) | 77 (13) | 83 (31) |

Reading:
- **g0**: mean ablation costs 0.7–3.2 points at every block (McNemar p ≤ 2e-4). Calls changed 5.6–9.5 %. This is far less than a random 16-component set, which leaves 18–85 %: extra drop −65.7 points [−67.5, −63.9] at block 0 and −15.0 [−16.5, −13.6] at block 5 (Holm-z p < 1e-10). At blocks 9–11 g0 and the control do not differ (Holm-z p 0.17–0.43). Resampling g0 costs more at late blocks: 66.8 % at block 11, 73.0 % at block 9.
- **s (auxiliary scalars)**: 40–70 % after ablation at every block, with 27–58 % of calls changed. From block 3 on this is more than the random-32 control: extra drop +12.4 [10.4, 14.4] at b5 and +13.5 [11.6, 15.3] at b11.
- **inv = g0 + s**: 24–47 %. From block 3 on this is more than random-48: extra drop +35.5 [33.2, 37.7] at b5 and +56.1 [53.8, 58.4] at b11 (Holm-z p < 1e-10). Resampling gives 13–38 %.
- **g1 and g3 (vector, axial vector)**: 14–33 % at blocks 0–2. This recovers to 57 % at block 3, ≥ 79 % from block 6 and ≥ 84 % at block 10. g1 ablation at block 0 removes the input four-momenta that still sit in the residual stream. Both are about equal to the random-64 control at blocks 0–2 (extra drop +1.6 [0.7, 2.5] at b0) and below it afterwards (−10.2 [−11.4, −9.1] at b5).
- **g2 and g4**: ≤ 2 points at every block. The exception is g2 resampling at block 0 (76.0 %).
- **cov (240)**: 16 % at block 0, 64 % at b5, 83 % at b11. The random-240 control leaves 10–31 %, so per component the covariant channels carry less than a random mix.
- **Code check**: at block 11 the read-out layer takes only grade 0, s and (through multiplication by the pseudoscalar) grade 4. Ablating g1, g2 or g3 at block 11 changes 0.1 % of calls (float floor), as required. g4 at block 11 changes 6.8 % of calls (83.1 %).
- **Prediction check**: "grade 4 and odd grades carry little" holds for g4 and g2 but not for g1/g3 at blocks 0–2. "Grade 0 carries most" does not hold. The invariant information the decision uses sits mostly in the 32 auxiliary scalar channels.

## (b) Spurion route (longitudinal boost; jet mass and every pair mass $m_{ij}$ unchanged)
Edit: every particle's rapidity is shifted by y (exact boost of E and pz; pT and particle mass fixed). In float32 the jet mass changes by at most 0.2 %. The largest pair-mass change is 7 %, for nearly collinear high-energy pairs, from float32 cancellation in $E^2 - p^2$. Calls changed are over 10,000 jets, with bootstrap CI and McNemar on correctness:

| edit | y 0.1 | y 0.3 | y 0.5 |
|---|---|---|---|
| A: particles boosted (physical edit) | 0.65 % [0.50, 0.82] | 1.70 % [1.45, 1.97] | 2.58 % [2.27, 2.89] (McNemar 113/102, p 0.50) |
| D: particles boosted, spurions and features held (spurion route) | 1.74 % | 4.78 % [4.37, 5.22] | 7.89 % (473/179, p 2e-31) |
| E: spurion swap (time reference boosted by −y, nothing else changed) | 1.73 % | 4.73 % | 7.85 % |
| F: only the scalar features taken from the boosted jet | 1.44 % | 4.62 % | 8.39 % (507/208) |
| B: particles + time reference boosted, features recomputed | 1.50 % | 4.63 % | 8.40 % |
| C: particles + time reference boosted, features held (exact symmetry) | 0.13 % [0.06, 0.20] | 0.17 % [0.09, 0.25] | 0.11 % [0.05, 0.18] |
| floors: identical inputs 0.02 %; rigid φ rotation 0.06 / 0.3 rad 0.07 % / 0.09 %; P1-style dφ + 0.06 0.08 % (W → Z 12.2 → 12.1 %); P1-style Δη + 0.1 0.64 % | | | |

Reading:
- Boosting particles and the time reference together, with the features held (C), changes 0.11–0.17 % of calls, mean |Δlogit| 0.02–0.03. This is the same size as the exact-rotation floor (0.07–0.09 %, mean |Δlogit| 0.016). The prediction "zero after boosting the spurions along, to float precision" holds.
- The equivariance identities hold at the level of calls: D ≈ E (1.74 / 1.73, 4.78 / 4.73, 7.89 / 7.85 %) and B ≈ F (1.50 / 1.44, 4.63 / 4.62, 8.40 / 8.39 %).
- The boost response does not come only through the spurion. The scalar-feature route (F, through log E and log E/E_jet) is as large as the spurion route (D). The physical edit A is 2–3 times smaller than either route alone, so the two routes partly cancel. W → Z: A 12.2 → 12.6 % at y 0.5, D → 18.0 %, F → 22.5 %.
- Compared with ParT: a rigid φ shift of 0.06 (W → Z 12 → 31 % in ParT_full, P1) is an exact symmetry for L-GATr (0.07 % calls, W → Z unchanged). The explicit prediction "L-GATr has no axis dependence beyond the spurions (and the frame-dependent scalar features)" is consistent with every edit here.

## (c) Mass read-out
Ridge probe of log m_inv (GeV) per block. Features: the global token's 48 invariant channels and the mean of the particle tokens' 48 (96 features). Fit on 5,000 jets with m > 1 GeV (α chosen on a validation split), test on the other 5,000. Calibration edit: all Δη, Δφ scaled by 1.1 about the stored axis, which gives a true Δlog m of +0.094. Mass-fixed edits are expressed as (probe change) / (probe change under the dilation), with bootstrap CI over test jets.
- R²: 0.977 (b0), 0.981 (b1), 0.968 (b4), 0.950 (b7), 0.915 (b11).
- Probe change / true change under the dilation (gain): 0.98–1.03 at blocks 0–9, 0.94 [0.93, 0.95] at b10, 0.92 [0.92, 0.94] at b11.
- Rigid φ + 0.06: 0.000 at all blocks (the edit is an exact symmetry). Boost y 0.3, physical edit (A): −0.003 to +0.016. Boost y 0.3 through the spurion route only (D): −0.062 to −0.081 at all blocks (CI excludes 0, e.g. b11 −0.081 [−0.094, −0.067]).
- Untrained L-GATr (random init, torch seed 0, same config; accuracy 9.84 %): R² 0.88 → 0.78, gain 0.83–1.00, boost (A) −0.06 to +0.01. A linear mass read-out from the invariant channels exists without training. R² and gain therefore do not show that training built a mass variable.

DAS (blocks 0, 2, 4, 6, 8, 10, 11): u is trained on 3,000 jets (KL to the source logits, 6 epochs) and tested on 3,000 other jets. 343–344 test jets change their call under the dilation. Learned 1-d IIA: 0.24 [0.19, 0.28] (b0), 0.28 (b2), 0.35 (b4), 0.35 (b6), 0.39 [0.34, 0.45] (b8), 0.22 (b10), 0.21 [0.17, 0.26] (b11). 20 random directions: mean 0.005–0.017, max 0.055 (binomial p vs random mean < 1e-50). Swapping all 48 invariant channels gives 0.37 (b0) to 0.71 (b11). The remaining 29–63 % of the dilation's call changes need the covariant channels at that block. Squared weight of u on grade 0 (vs s): 0.23–0.26 at blocks 0–8 and 0.50–0.55 at blocks 10–11; a uniform direction would give 0.33. Untrained L-GATr: only 44 test jets flip, and learned IIA is 0.45–0.89 with all-48 swap 0.61–0.84. The untrained control is therefore not lower than the trained network. Its few flips are low-margin and easy to steer, so this control does not separate the two networks.

## Competing hypotheses
- H0 (vector/bivector channels carry substantial class information): holds for g1/g3 at blocks 0–2 and is excluded for g2 and g4.
- C21 as stated ("mass read-out built in grade-0 channels"): mean ablation of grade 0 costs ≤ 3.2 points, so the stated form is excluded. A refined form fits: the invariant read-out uses the auxiliary scalar channels, and grade 0 matters only under resampling at late blocks.
- "The η shift changes calls only through spurions": excluded. The log-E scalar features carry an equal share. "The change is zero once all symmetry-breaking inputs are boosted along": holds to the float floor.
- "Implementation not equivariant": excluded by C and the D ≈ E and B ≈ F identities.

## Issues and limits
- Interim checkpoint (it154999), not the final one.
- Mean |Δlogit| has a GPU non-determinism floor (0.004). The max |Δlogit| floor (1.0) is large, so max-|Δlogit| equivalence checks are not informative. Call-level checks are used instead.
- Controls cover N 2,000, not 10,000. The random-component control mixes channel types, so it does not measure "any 16 components". Equal-size random sets that hit s or g1/g3 are more damaging than g0.
- DAS uses 6 epochs (training loss 0.13 → 0.06–0.10). It may be under-trained.
- No explainer entry or glossary edits were made, because site/ is outside the write scope. Glossary terms to add: multivector, grade, spurion, equivariance, auxiliary scalar channels, DAS, IIA.

## Explainer draft (for site/explain.py, Paper C)
**Question.** Does L-GATr, a Lorentz-equivariant tagger, decide from invariant quantities, and where does any frame dependence enter? **Background.** L-GATr represents each particle as a multivector and uses layers that commute with Lorentz transformations; two fixed reference multivectors (beam plane, time direction) and per-particle log E features tell it the lab frame. ParT's mass response depends on a stored jet axis (C3), so an equivariant model is the natural contrast. **Method.** (i) Blank one grade of all hidden channels after each block and compare with blanking the same number of random components; (ii) boost every particle along the beam by rapidity $y$ (all $m_{ij}$ fixed) and route the boost separately through particles, the time reference $t \to (\cosh y, 0, 0, \pm\sinh y)$ and the scalar features; equivariance requires $f(\Lambda p, \Lambda t, \text{features held}) = f(p, t)$; (iii) ridge probe of $\log m$ from the invariant channels calibrated by $\Delta\eta, \Delta\phi \to 1.1\,\Delta\eta, 1.1\,\Delta\phi$. **Result.** Figure p44_lgatr_grades.png: auxiliary scalars plus grade 0 carry the decision after block 3, grade 0 alone does not; the boost moves ≤ 2.6 % of calls, through both the time reference and log E, and 0.1 % when both are boosted along. **Checks.** Exact rotation 0.07–0.09 %; read-out-layer check at block 11; untrained network for probe and DAS. **Why it matters.** A network whose inputs are four-vectors and fixed references shows no axis dependence beyond those references, so ParT's axis-relative mass is tied to its axis-relative inputs, not to the jet-tagging task.
