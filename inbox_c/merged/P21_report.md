# P21: which prong observable ParT uses: τ21, τ32 and D2 edited one at a time, located in blocks 5–8 (claim C4, prong stage) (agent report, 2026-10-20)

Jobs:
- `anrunw-ai-c-p21-smoke` (GTX 1080 Ti, 6 min, 150 jets per class).
- `anrunw-ai-c-p21-run`: evicted at 1,304 s when its node failed (events: NodeNotReady, TaintManagerEviction; not a code error). Partial JSON kept as `results/part/p21_prong_part_run1_partial.json`.
- `anrunw-ai-c-p21-run2` (complete, 3,097 s): the full run, with one added arm (`raw`, below).
  - Behaviour numbers are identical to the evicted run's partial output. For example, ParT_full E1 at s = 0.6, edit − rotation d_WQ = −0.144 in both.

Files:
- Script: `interp/p21_prong_part.py`.
- Results: `results/part/p21_prong_part.json`.
- Figure: `results/part/p21_prong_part.png`.
- Logs: `logs/c-p21-run2.log`, `logs/c-p21-run.log`, `logs/c-p21-smoke.log`.

## Conclusion
When one of the three standard prong-shape variables is changed with the other two and the jet mass held fixed, ParT does not follow that variable the way a simple classifier built on it does. A boosted-decision-tree classifier on (mass, τ21) predicts that lowering τ21 (making the jet look more two-pronged) makes it more W-like. ParT_full's W-versus-QCD score instead moves the other way: −0.14 in logit relative to a matched control, where the reference predicts +0.25. ParT's response is also spread over the same blocks and the same attention heads whichever variable is edited, so the late-block "prong" stage found in P18 is not specific to τ21, τ32 or D2. The largest and most consistent response, shared by both ParT and both ParticleNet models, is to moving the soft particles radially at fixed τ21 and τ32. That move changes D2 by only 4 % but moves the W-versus-QCD logit by +0.73. Still uncertain: an edit that changes one variable at fixed others also changes things these variables do not describe, such as subjet masses and where the soft radiation sits. The matched rotation controls also move the logits. So what the network does read is not yet identified.

## Procedure
**Models.**
- Released ParT_full and ParT_kin; released ParticleNet_full and ParticleNet_kin (behaviour only).
- Controls with the ParT_full architecture:
  - `init`: weaver default initialisation, torch seed 18, untrained.
  - `perm`: the released ParT_full with the output rows of all 48 weight tensors permuted, seed 18 (as P18).

**Sample.**
- JetClass test_20M `WToQQ_103`, `ZToQQ_103`, `TTBar_103` and `ZJetsToNuNu_103` (Wqq, Zqq, Tbqq, QCD).
- 4,000 random rows per class (`numpy.default_rng(2103)`, never first-k). Jets with ≥ 4 particles: N = 15,998 of 16,000.
- Every jet (base and edited) is rebuilt about its own E-scheme axis (4-vector sum of its particles; `part_core.build`, η = η_jet + sign(η_jet)·Δη).
- Edited jets are then dilated in angle about that axis until the invariant mass equals the base mass. A uniform dilation leaves τ21, τ32 and D2 unchanged.
- Particle pT, masses, PID and impact-parameter fields, and N are unchanged in every edit and control. So the planned |ΔpT_i| matching is trivially met (ΔpT_i = 0).

**Observables.** As P18 `substructure`:
- τ_k with β = 1, k-means axes, R0 = 0.8.
- D2 with β = 1, from the 24 hardest particles.
- The torch re-implementation matches P18's numpy code to 3e-16 (τ21, τ32) and 1e-15 relative (D2).
- Rebuilding the base jet about its own axis changes τ21, τ32 and D2 by 1e-8 relative and m by 8e-6.

**Edit components**, in the (Δη, Δφ) plane. Clusters are the k-means 2- or 3-subjet assignment of the base jet, i.e. the τ axes. Exclusive-kT subjets were not used (a deviation from the plan).
- A(s2): every particle's offset from its 2-subjet pT centroid is multiplied by s2.
- B(s3): the same about the 3-subjet centroids.
- C(u): soft particles (pT fraction z < 0.02) moved radially about the jet pT centroid by the factor u.
- Applied in the order A, B, C.

**Edits** at nominal dose s ∈ {0.6, 0.8, 1.25}:
- E1 (τ21 edit) = A(s), with (s3, u) chosen per jet to hold τ32 and D2.
- E2 (τ32 edit) = B(s), with (s2, u) chosen to hold τ21 and D2.
- E3 (D2 edit) = C(s), with (s2, s3) chosen to hold τ21 and τ32.
- The search is a grid of 11 × 11 values on [0.5, 2], then two 5 × 5 refinements, minimising the larger relative change of the two held observables.
- A jet is accepted if both held observables are within 2 % and the mass within 0.5 % after the rebuild.

**Controls** (same jets, same rebuild and mass restore):
- `rot`: each cluster's particles are rotated about the cluster centroid (E1: 2-subjet clusters; E2: 3-subjet clusters; E3: only the soft particles, about their own 3-subjet centroid). The angle gives the same pT-weighted mean particle displacement as the edit, with a random sign per cluster.
- `rc`: the primary component at the same dose, applied about clusters built around 2 (E1) or 3 (E2) random constituents drawn with probability ∝ pT. For E3, the radial move is about a random constituent. No compensation.
- `raw` (added in run2): the primary component alone at the same dose, with no compensation (A(s), B(s) or C(s)). This separates the primary component from the compensating ones.
- `scrF`: the P18 full-jet azimuth scramble about the base jet's axis, as the positive reference.

**Metrics.**
- d_WQ = logit(Wqq) − logit(QCD) and d_TQ = logit(Tbqq) − logit(QCD), each as the change against the base jet. Also the share of 10-class calls that change.
- Edit − control as a paired per-jet difference over the accepted jets:
  - 95 % bootstrap CI over jets, 1,000 resamples.
  - Sign-flip permutation p (5,000 permutations; floor 2e-4).
  - McNemar test on call changes.
  - Holm correction over 4 networks × 3 edits × 3 doses for the primary score (E1, E3: d_WQ; E2: d_TQ).

**References.**
- HistGradientBoosting classifiers on (log m, τ21), (log m, τ32, log D2) and (log m, τ21, τ32, log D2).
- Two tasks: W/Z vs QCD and top vs QCD. Each is 2-fold cross-fitted on this sample's base jets.
- Held-out AUC: W/Z vs QCD 0.890 / 0.899 / 0.910; top vs QCD 0.947 (m, τ21) / 0.964 / 0.971.
- Predicted change = f(edited observables) − f(base observables).

**Location** (ParT_full, ParT_kin, init; dose 0.6 for E1–E3, plus scrF):
- Jets: 3,000 DAS-training and 1,000 evaluation jets, drawn at random from the 6,108 jets accepted for all three edits.
- (i) 1-d DAS per site, by the P18 method:
  - Representation: the pooled pT-weighted particle mean; the class token at c1.
  - Counterfactual: the full edited jet.
  - Fit: 150 Adam steps × 128 jets.
  - Score: IIA = 1 − Σ‖L_int − L_cf‖² / Σ‖L_base − L_cf‖².
  - Null: 1,000 linearised random directions.
  - Trained vs init: paired bootstrap difference-in-differences of IIA(b7) − IIA(emb).
- (ii) Head patching in blocks 5–8:
  - In the edited jet's forward pass, one head's attention output (A_h V_h W_o,h) is replaced by its value on the unedited jet. As a block-level variant, all 8 heads of a block are replaced together.
  - The per-head decomposition reproduces the attention output to 3e-5.
  - Restoration R = 1 − Σ‖L_patch − L_base‖² / Σ‖L_edit − L_base‖².
  - One-sided bootstrap p for H0: R ≤ 0, with Holm over the 64 heads of ParT_full + ParT_kin per edit.
  - The same patching is run on init and perm.

**Hypotheses.**
- H0: the response to an edit equals the response to its matched rotation control.
- H1: the network responds to the edited observable.
- Thesis: E1 moves d_WQ with a slope comparable to GBDT(m, τ21); E2 moves d_TQ; the response is located in the heads of blocks 6–7, above init and perm.
- Alternatives:
  - (a) Equal response to E1–E3, i.e. a joint prong shape.
  - (b) Response equal to the rotation control.
  - (c) An untrained net shows the same late-block share.

## Numbers
**Unedited accuracy (10 classes).**

| sample | ParT_full | ParT_kin | init | perm | PN_full | PN_kin |
|---|---|---|---|---|---|---|
| P1 rows (10 × 1,000, rebuilt) | 85.93 % | 74.47 % | 13.58 % | 11.34 % | 84.56 % | 70.36 % |
| this sample (4 classes, N = 15,998) | 79.83 % | 71.12 % | 34.0 % | 22.1 % | 77.97 % | 69.95 % |

- References: 85.9 / 74.5 %.
- ParT_full per class on this sample: W 79.5 %, Z 68.7 %, top 94.5 %, QCD 76.7 %.

**Edits achieved** (mean relative change on accepted jets; per-jet s.d. in brackets).

| edit, dose | accepted | primary | held (each) | rot: τ21 / τ32 / D2 | raw (no compensation): τ21 / τ32 / D2 |
|---|---|---|---|---|---|
| E1, 0.6 | 9,210 (57.6 %) | τ21 −10.6 % (12.6) | τ32 +0.03 %, D2 −0.04 % (s.d. 0.6–0.7 %) | −1.2 / +0.8 / −0.3 % (s.d. 6–9 %) | −32.8 / +1.0 / −18.4 % |
| E1, 1.25 | 9,227 (57.7 %) | τ21 +8.9 % (14.6) | ≤ 0.14 % | −0.7 / +0.7 / −0.3 % | +13.5 / +1.3 / +7.4 % |
| E2, 0.6 | 13,674 (85.5 %) | τ32 −12.7 % (14.2) | ≤ 0.03 % | −0.04 / −2.0 / −0.4 % | −15.0 / −25.4 / −10.6 % |
| E2, 1.25 | 14,568 (91.1 %) | τ32 +9.9 % (15.9) | ≤ 0.11 % | −0.07 / −0.6 / −0.1 % | +8.7 / +9.0 / +5.7 % |
| E3, 0.6 | 12,193 (76.2 %) | D2 −4.0 % (10.2) | ≤ 0.01 % | −2.0 / −3.6 / −0.8 % | +3.9 / +3.6 / −6.6 % |
| E3, 1.25 | 15,005 (93.8 %) | D2 +4.6 % (8.1) | ≤ 0.01 % | −0.9 / −1.8 / −0.4 % | +4.2 / +3.9 / +9.1 % |

- Median compensation parameters for E1: (s3, u) = (1.00, 1.32) at s = 0.6 and (0.89, 0.83) at s = 1.25.
  - So holding D2 at s = 0.6 moves the soft particles outward by about 32 %. This is the same kind of move as E3 at u > 1.
- E1 at s = 0.6 accepts 37.6 % of top jets against 61–69 % of the other classes.
- Rotation controls: only 15–52 % of jets keep each observable within 1 %. The planned "< 1 %" holds for the means only, not per jet.
- scrF: τ21 +94.6 %, τ32 +31.6 %, D2 +28.1 %, m −2.4 %.

**Edit − rotation control, mean Δlogit** (95 % CI; permutation p = 2e-4 in every cell unless stated; Holm over the 36 primary tests p ≤ 0.007 unless stated). Reference = the GBDT-predicted edit − rotation difference.

| edit, dose | score | ParT_full | ParT_kin | PN_full | PN_kin | reference |
|---|---|---|---|---|---|---|
| E1, 0.6 | d_WQ | −0.144 [−0.177, −0.111] | −0.110 [−0.139, −0.080] | +0.006 [−0.022, 0.036] (p 0.67) | −0.014 [−0.040, 0.012] (p 0.29) | +0.254 (m, τ21) |
| E1, 0.8 | d_WQ | −0.104 [−0.128, −0.080] | −0.071 | +0.021 (Holm 0.22) | −0.018 (Holm 0.22) | +0.100 |
| E1, 1.25 | d_WQ | +0.047 [0.020, 0.072] | +0.067 | +0.076 | +0.055 | −0.229 |
| E2, 0.6 | d_TQ | −0.246 [−0.279, −0.213] | +0.170 [0.147, 0.195] | +0.114 | +0.366 | +0.598 (m, τ32, D2) |
| E2, 0.8 | d_TQ | −0.207 | +0.014 [−0.003, 0.031] (Holm 0.33) | −0.030 (Holm 0.03) | +0.085 | +0.256 |
| E2, 1.25 | d_TQ | −0.334 [−0.356, −0.312] | −0.337 | −0.226 | −0.392 | −0.575 |
| E3, 0.6 | d_WQ | +0.729 [0.703, 0.756] | +0.719 | +0.671 | +0.672 | −0.054 (all four) |
| E3, 0.8 | d_WQ | +0.304 | +0.315 | +0.294 | +0.259 | −0.019 |
| E3, 1.25 | d_WQ | −0.215 [−0.232, −0.197] | −0.187 | −0.138 | −0.207 | −0.138 |

- Secondary scores for ParT_full:
  - E1 also moves d_TQ: −0.447 (0.6), −0.453 (0.8), −0.215 (1.25). The reference (m, τ21) predicts −0.21 / −0.09 / +0.14.
  - E2 moves d_WQ by +0.136 (0.6) and −0.072 (1.25).
- Calls changed, ParT_full, edit / rotation:
  - E1 0.6: 20.7 / 7.1 %. McNemar 1,545 edit-only vs 285 control-only, p < 1e-200.
  - E2 0.6: 15.7 / 6.9 %.
  - E3 0.6: 17.5 / 13.0 %.
  - scrF: 51.2 %.
- E1, d_WQ by true class, ParT_full, s = 0.6 (edit; rotation in brackets): QCD +0.283 (+0.015), Wqq −0.780.
  - Same pattern in all four networks: QCD +0.28 to +0.33; Wqq −0.36 to −0.78.
- Rotation controls alone move the logits. ParT_full d_WQ: −0.161 (E1 rot, 0.6), −0.151 (E2 rot), −0.430 (E3 rot).
- Untrained and permuted nets respond with |mean Δd| ≤ 0.08. Their mean squared logit change is 0.03 (init) and 3.2 (perm), against 12.6–16.9 for ParT_full.

**Uncompensated components (`raw`), ParT_full, Δ vs base** (rotation-control value in brackets):
- A(0.6), τ21 −33 %: d_WQ −0.218 [−0.246, −0.187] (−0.161); d_TQ −0.437 (−0.151). Reference (m, τ21) raw − rot = +0.95.
- B(0.6), τ32 −25 %: d_TQ +0.099 [0.076, 0.120] (−0.149); d_WQ +0.140. Reference (m, τ32, D2) raw − rot = +1.22.
  - ParT_kin: d_TQ +0.630. PN_full: +0.387. PN_kin: +1.024.
- C(0.6), D2 −6.6 %: d_WQ +0.194 (−0.430). C(1.25): −0.415 (−0.194).
- E − raw for E3 is ≤ 0.13 in d_WQ for all models (0.07–0.13 at u = 0.6, ≤ 0.04 at u = 0.8 and 1.25), so the E3 response comes from the soft-particle move itself and not from its compensation.
- E − raw for E1 at s = 0.6 is −0.087 (ParT_full, d_WQ). Narrowing the 2-subjets alone also lowers d_WQ against the rotation control (raw − rot = −0.06).

**Location, DAS 1-d IIA** at emb / b2 / b4 / b5 / b6 / b7 / b8 / c1 (1,000 evaluation jets; 95 % CI ≈ ±0.02–0.04).

| edit | ParT_full | ParT_kin | init |
|---|---|---|---|
| E1 (τ21) | 0.01 / 0.07 / 0.12 / 0.18 / 0.24 / 0.31 / 0.32 / 0.38 | 0.01 / 0.04 / 0.09 / 0.12 / 0.14 / 0.18 / 0.19 / 0.22 | 0.34 / 0.46 / 0.46 / 0.44 / 0.42 / 0.44 / 0.42 / 0.41 |
| E2 (τ32) | 0.03 / 0.06 / 0.11 / 0.17 / 0.22 / 0.28 / 0.30 / 0.35 | 0.01 / 0.04 / 0.09 / 0.12 / 0.15 / 0.18 / 0.20 / 0.21 | 0.36 / 0.51 / 0.52 / 0.52 / 0.50 / 0.50 / 0.48 / 0.47 |
| E3 (D2) | 0.03 / 0.06 / 0.12 / 0.16 / 0.20 / 0.25 / 0.26 / 0.31 | 0.03 / 0.05 / 0.08 / 0.11 / 0.13 / 0.17 / 0.22 / 0.20 | 0.46 / 0.62 / 0.61 / 0.60 / 0.59 / 0.59 / 0.57 / 0.55 |
| scrF | 0.02 / 0.40 / 0.63 / 0.69 / 0.70 / 0.71 / 0.70 / 0.59 | 0.11 / 0.26 / 0.60 / 0.69 / 0.73 / 0.76 / 0.75 / 0.69 | 0.27 / 0.45 / 0.46 / 0.46 / 0.46 / 0.46 / 0.46 / 0.43 |

- Linearised random-direction maximum (1,000 draws) per site: ≤ 0.08 for E1–E3 and ≤ 0.13 for scrF (ParT_full). DAS exceeds it at every site from b2 on.
- ParT_full E1, E2 and E3 agree within 0.07 at every site.
- Difference-in-differences against init, (b7 − emb)_trained − (b7 − emb)_init:
  - ParT_full: E1 +0.195 [0.157, 0.232]; E2 +0.118 [0.069, 0.163]; E3 +0.083 [0.052, 0.114]; scrF +0.509 [0.488, 0.532]. All p = 0.001.
  - ParT_kin: E1 +0.061 [0.033, 0.088]; E2 +0.033 [−0.003, 0.065] (p 0.064); E3 +0.009 [−0.016, 0.033] (p 0.52); scrF +0.455 [0.428, 0.480].
- Per-site trained − init, ParT_full: E1–E3 are negative at every site (−0.56 to −0.04). scrF is +0.17 to +0.25 at b4–b8.
- Call IIA for E1–E3 (ParT_full): 0.04–0.15 at every site.

**Location, head patching** (1,000 jets; block = all 8 heads restored).

| edit | ParT_full block restoration b5 / b6 / b7 / b8 | ParT_kin | init | perm |
|---|---|---|---|---|
| E1 | 0.299 / 0.336 / 0.266 / 0.206 | 0.210 / 0.240 / 0.254 / 0.215 | 0.046 / 0.179 / 0.204 / 0.145 | 0.183 / 0.172 / 0.102 / 0.079 |
| E2 | 0.305 / 0.336 / 0.268 / 0.217 | 0.213 / 0.231 / 0.251 / 0.208 | 0.027 / 0.170 / 0.230 / 0.124 | 0.192 / 0.188 / 0.115 / 0.108 |
| E3 | 0.296 / 0.318 / 0.274 / 0.223 | 0.187 / 0.210 / 0.243 / 0.196 | −0.019 / 0.193 / 0.238 / 0.135 | 0.170 / 0.174 / 0.087 / 0.098 |
| scrF | 0.118 / 0.141 / 0.170 / 0.220 | 0.079 / 0.090 / 0.107 / 0.137 | 0.294 / 0.068 / 0.218 / −0.096 | 0.134 / 0.093 / 0.096 / 0.115 |

- Block CIs are about ±0.02.
- ParT_full − init by block, E1: b5 +0.254 [0.225, 0.281], b6 +0.157, b7 +0.063 [0.036, 0.090], b8 +0.061. E2 and E3 are within 0.04 of these.
- ParT_full − perm: +0.11 to +0.19 at every block for E1–E3.
- Single heads, ParT_full:
  - The three highest restorations are b6h5, b5h5 and b7h5 (head 5 of blocks 5–7) for all three edits:
    - E1: 0.078 [0.070, 0.087], 0.067, 0.072.
    - E2: 0.086, 0.080, 0.078.
    - E3: 0.074, 0.087, 0.080.
  - The sum over the 32 single heads is 1.07–1.13.
  - scrF: b8h8 0.063, b7h4 0.054; sum 0.67.
- Single heads, ParT_kin: b7h6, b8h3 and b5h3 lead for all three edits (0.047–0.057).
- Statistics: raw one-sided p = 0.001 (the bootstrap floor) for 29–30 of 32 heads per edit in both models. With 1,000 resamples and 64 tests, the smallest attainable Holm p is 0.064, so no head reaches Holm 0.05 (see Issues).

## Reading (scoped to these measurements)
**Edits.**
- The edits change their target observable by −13 % to +10 % (τ21, τ32) and by ±4 % (D2) at the 0.6 / 1.25 doses.
- The held observables stay within 0.14 % on average, and the mass within 0.5 % per jet.
- E1 holds D2 mainly by moving soft particles outward or inward (u = 1.32 / 0.83). So E1 contains part of the move E3 makes.

**H0 (edit = rotation control).**
- Excluded for E1 (both ParT models), E2 (all four models at doses 0.6 and 1.25) and E3 (all four models), with Holm p ≤ 0.007 in 30 of 36 primary cells and Holm p = 0.029 in one more (PN_full E2 at 0.8).
- Not excluded for PN_full and PN_kin E1 d_WQ at 0.6 and 0.8 (Holm 0.22–0.67), or for ParT_kin E2 at 0.8 (Holm 0.33).

**Thesis, sign and size of the response.**
- τ21 (E1) for W/Z vs QCD: ParT_full and ParT_kin move d_WQ opposite to the GBDT(m, τ21) prediction at both doses (−0.14 vs +0.25 at s = 0.6; +0.05 vs −0.23 at s = 1.25).
  - By class, QCD jets move toward W (+0.28) and W jets away from W (−0.78) when τ21 is lowered.
  - The uncompensated subjet narrowing (τ21 −33 %) also gives raw − rot d_WQ of −0.06 against a reference +0.95.
  - The thesis prediction "E1 slope comparable to GBDT(m, τ21)" is excluded for ParT on this sample.
- τ32 (E2) for top vs QCD:
  - ParT_full: d_TQ falls at both doses (−0.25 at 0.6, −0.33 at 1.25), so it is not monotone in τ32.
  - ParT_kin, PN_full and PN_kin move d_TQ in the reference direction at doses 0.6 and 1.25, at 0.2–0.7 times the reference size.
  - The uncompensated 3-subjet narrowing raises d_TQ in all four networks (+0.10 ParT_full to +1.02 PN_kin; reference +1.22 relative to rotation).
- D2 (E3): all four networks move d_WQ by +0.67 to +0.73 for a 4 % D2 decrease, where the GBDT on (m, τ21, τ32, D2) predicts −0.05.
  - The response is in the same direction for the opposite dose and of similar size to its reference (−0.14 to −0.22 vs −0.14).
  - The E3 response is carried by the soft-particle radial move itself: E − raw ≤ 0.13.

**Alternative (a), joint prong shape.**
- In behaviour, the three edits give different sizes and signs: E3 ≫ E1 in d_WQ, and E2 only moves d_TQ.
- In location, the three edits are indistinguishable: identical DAS depth profiles within 0.07, identical block restoration within 0.04, and the same leading heads (head 5 of blocks 5–7 in ParT_full).
- Within this measurement, the internal location does not resolve which observable was edited.

**Alternative (b), response = rotation.** Excluded in the cells listed under H0. The rotation controls themselves move d_WQ by −0.15 to −0.43 (ParT_full) while changing the observables by ≤ 3.6 % on average. So ParT responds to particle placement inside subjets beyond what τ21, τ32 and D2 record.

**Alternative (c), the untrained net gives the same late-block share.**
- ParT_full: the depth rise of the 1-d share is training-specific for all three edits (difference-in-differences +0.08 to +0.20, CIs exclude 0). The share level stays below the untrained net's at every site (−0.04 to −0.56).
- ParT_kin: the rise exceeds init only for E1 (+0.06); it does not for E2 and E3.
- Block patching in ParT_full exceeds init by +0.25 to +0.32 at b5 and by +0.04 to +0.06 at b7. The largest restoration is at b5–b6, not b6–b7.
- The single-observable edits do not reproduce the full scramble's training-specific stage: 0.71 at b7 vs 0.25–0.31, with difference-in-differences +0.51 vs +0.08 to +0.20.

**For C4.**
- The P18v late-block stage (1-d share 0.02 → 0.69–0.71 at b5–b7, reproduced here) belongs to the full scramble.
- Edits that move τ21, τ32 or D2 one at a time each produce a smaller, shared, rising share (0.25–0.31 at b7). Their responses do not follow the observables' classifier-predicted sign (τ21) or monotonicity (τ32 in ParT_full).
- None of the three named observables is identified as the variable behind the stage.

## Issues
- **Off-manifold side effects.** Narrowing subjets at fixed mass reduces subjet masses, and moving soft particles changes their radial profile. Neither is held, and both may be what the networks read. The `raw` and `rot` arms show these moves alone shift the logits by up to 0.4.
- **Rotation controls are not neutral per jet.** Per-jet s.d. of the observable change is 3–9 %. The planned "< 1 %" holds for means only.
- **Acceptance differs by class.** E1 at s = 0.6 keeps 38 % of top jets and 61–69 % of the others, so the analysed jets differ between edits and doses.
- **Subjet assignment.** k-means axes were used instead of exclusive-kT subjets (same axes as the τ definition).
- **Head statistics.** The Holm floor for 64 heads with 1,000 bootstrap resamples is 0.064, so no head can reach Holm 0.05. A rerun needs ≥ 2,000 resamples, or the normal approximation from the bootstrap s.e.
- **Untrained and permuted controls barely respond** (mean squared logit change 0.03 and 3.2, against 13–17). Their restoration and IIA are fractions of a near-zero response.
- **Score share.** The primary-score share in head patching is unstable for E2 at dose 0.6, because the mean d_TQ change over the 1,000 jets is +0.01 (mixed signs). Restoration (vector) is the reported quantity.
- **4-class sample accuracy** (79.8 % ParT_full) is a 10-class argmax over 4 classes and is not comparable to 85.9 %. The P1-row reference reproduces exactly.
- **Not done.** The planned `c-p21-verify` job (fresh rows, second seed) has not been run; it is for the verifier. Suggested additions for it:
  - an edit holding subjet masses as well;
  - per-class GBDT predictions;
  - NB ≥ 2,000 for the head Holm correction.

## Explainer entry (for site/explain.py)
- **Question.** P18 found that one direction in ParT's late blocks carries the response to scrambling a jet's prong structure, and that this depends on training. The scramble changes three standard prong variables at once: $\tau_{21}=\tau_2/\tau_1$, $\tau_{32}=\tau_3/\tau_2$ and $D_2=e_3/e_2^3$. Which of them does ParT use?
- **Background.**
  - $\tau_N=\frac{1}{R_0}\sum_i z_i \min_k \Delta R_{i,k}$ measures how well N axes describe the jet's energy, with $z_i=p_{T,i}/\sum_j p_{T,j}$.
  - A W jet has two prongs (low $\tau_{21}$) and a top jet three (low $\tau_{32}$).
  - $D_2$ uses energy correlation functions $e_2=\tfrac12\sum_{ij}z_iz_j\Delta R_{ij}$ and $e_3=\tfrac16\sum_{ijk}z_iz_jz_k\Delta R_{ij}\Delta R_{jk}\Delta R_{ik}$.
- **Method.**
  - Edits: E1 scales each particle's offset from its 2-subjet centroid by $s$, with a 3-subjet scaling and a soft-particle radial factor solved per jet so that $\tau_{32}$ and $D_2$ stay within 2 %. E2 and E3 are built the same way for the other two variables. The jet is then rebuilt about its own axis and dilated until $m$ is unchanged.
  - Control: rotations inside each subjet with the same mean particle displacement.
  - Network response: $\Delta d_{WQ}=\Delta(\ell_W-\ell_{QCD})$, compared with the change predicted by gradient-boosted trees trained on $(m,\tau_{21})$.
  - Location: the 1-d interchange share $\mathrm{IIA}=1-\sum\lVert L^{int}-L^{cf}\rVert^2/\sum\lVert L^{base}-L^{cf}\rVert^2$ per block, and head patching $R_h=1-\sum\lVert L^{patch}_h-L^{base}\rVert^2/\sum\lVert L^{edit}-L^{base}\rVert^2$.
- **Result (figure `results/part/p21_prong_part.png`).**
  - Panel (a): lowering $\tau_{21}$ by 11 % at fixed $\tau_{32}$, $D_2$ and $m$ moves ParT_full's W-vs-QCD logit by $-0.14\,[-0.18,-0.11]$ relative to the rotation control, where the $(m,\tau_{21})$ trees predict $+0.25$. Moving soft particles 40 % closer to the axis ($D_2$ −4 %) moves it by $+0.73$ in all four networks.
  - Panel (b): the one-direction share rises with depth to 0.25–0.31 at block 7 for all three single-variable edits, against 0.71 for the full scramble.
  - Panel (c): restoring the 8 heads of one block removes 0.21–0.34 of the response, nearly the same for the three edits.
- **Checks.**
  - The unedited model reproduces 85.93 % / 74.47 %.
  - The held variables move ≤ 0.14 % on average.
  - The untrained net's share is flat (0.42–0.62).
  - The full-scramble profile reproduces P18v (0.02 → 0.71).
- **Why it matters.** The training-specific late stage of ParT is not explained by any one of the textbook prong variables. ParT reacts to how particles are placed within and around subjets in ways these variables do not capture.
