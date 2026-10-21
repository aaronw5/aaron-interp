# E38: the leading-particle (hardness) channel in JEDI-linear: rank-to-rank pT transfers at fixed jet four-momentum

Claims A18, A3, A5 (exploratory: a quantity beyond mass). Script `interp/e38_hardness_channel_jedi.py`; figure `interp/e38_plot.py` → `results/jedi/e38_hardness.png`.
Results `results/jedi/e38.json` (all numbers), `e38_maps.npz` (cell maps), `e38_fig_jet.npz` (figure jet), `e38_margin_sd.json` (spread of the trained network's margins, job `e38-sd`).
Logs `logs/e38-run3.log` (final run), `e38-smoke*.log`; on the volume `/j-jepa-vol/aaron-interp/logs/`.

## Conclusion
We moved a fixed share of pT from one of the hardest particles to another (or to/from the soft particles), then nudged all particle angles by the smallest amount that keeps the jet's mass, pT and axis exactly as before. The W-vs-Z and quark-vs-gluon scores move in almost the same direction whichever way the pT is moved. A random reshuffle of the same number of GeV among the soft particles moves them as much or more (in 99 of 126 edit types the soft reshuffle moves the W-vs-Z score further, Holm p < 0.05). Holding the three "leading-particle" internal objects (S20, S51, S43: hinge functions of the pT of the 1–5 hardest particles) at their unedited values removes 3 % of the score change behind the earlier minimal call flips (E36). Holding random triples removes a median 11 %, and holding the three radial objects (S1, S29, S30) removes 38 %. So the network does not treat the share of pT carried by the hardest particles as a separate named channel. The response to these edits looks like a general sensitivity to perturbations at fixed four-momentum, mostly Z jets moving toward W. Still open: every edit here also moves particle angles by 3–47 mrad (pT-weighted), and that angle part cannot be separated from the pT part while the mass is held fixed.

## 1. Question and design
E36/E36v found that the smallest edits that flip the network's call at fixed jet mass, pT and axis mostly move pT among the hardest particles. They raise pT^D by 1.6 sd and move the internal objects S20, S51, S43 by 1.2–2.1 sd. The radial objects S1, S29, S30 move ≤ 0.27 sd under the same edits. E38 asks whether these leading-particle objects form a separate, named "hardness" channel. The test has three parts: a direct map of the response to pT moved between specific pT ranks, a naming fit of the three objects, and a hold test of how much of E36's flips they carry.

- **Network**: official JEDI-linear 3-feature n64 (inputs pT, Δη, Δφ about the stored jet axis); exact fixed-point forward pass `jedi_core.Jedi.run`; unit holds with `Jedi.run_jetwise` (override of the S layer after pooling).
- **Samples**: main `sample(1000, 3801)` (5,000 jets, 1,000 per class); replicate `sample(1000, 3802)` minus overlaps (4,900 jets); pool `sample(1000, 3804)` minus both (4,790 jets; reference MLP training only). All 5,000 main jets have ≥ 8 particles and enter.
- **Accuracy check (unedited)**: 81.86 % on main (N 5,000; binomial 95 % interval ±1.1 points, overlapping the 80.6–81.3 % reference); 80.57 % on replicate (N 4,900).
- **Edit T(k→j, f)**: a fraction f ∈ {0.05, 0.10, 0.20} of the pT of the particle at pT rank k is removed from it and added to the particle at rank j. Here k, j ∈ {0, 1, 2, 3, 4, 5, rest}, with "rest" = ranks 6–63 pooled:
  - as a source, every rest particle gives the fraction f of its own pT;
  - as a destination, the moved pT is shared among the rest particles in proportion to their pT.
  This gives 42 ordered (k, j) cells × 3 fractions = 126 edits per jet. ΣpT is unchanged by construction.
- **What is held fixed and how**: the summed four-momentum (massless particles, axis at (0, 0), as in E36). Moving pT between particles at different angles changes the jet's invariant mass and E-scheme axis (the E-scheme axis is the direction of the summed four-momentum). To restore them, the angles (Δη, Δφ) of all particles are changed by the smallest amount, in the E36 angle unit 0.02 rad, that brings Σp^μ back to its unedited value. This is a Gauss–Newton minimum-norm step in angle space only, using E36's Jacobian (`e36_counterfactual_flip_jedi.jac/fourmom`), with the pT change ramped in 2 steps, pseudo-inverse of the 4 × 4 normal matrix and step halving. Particle pT values are not touched by the projection, and the particle count is unchanged. Particles are re-sorted by pT before the network sees them (slot = pT rank).
- **Achieved constraints**: an edit is kept if |Δm|/m < 10⁻³, the axis shift is < 1 mrad and |ΔpT|/pT < 10⁻³ ("compliant"). Compliance is 98.9 % of jet-edits overall, and ≥ 92.1 % in every cell; the lowest cells are those with rank 0 as source at f = 0.2. M2 = Σ pT r² changes by ≤ 0.006 sd, as expected with m, ΣpT and the axis fixed.
- **Residual angular change** (median over jets of the pT-weighted mean |Δangle|): at f = 0.2 it is 24–31 mrad for rank 0 as source, 3–9 mrad for ranks 3–5, and 40–47 mrad for "rest" as source. The median largest single-particle move is 88–111 mrad for rank 0 as source and 131–152 mrad for rest as source. These angle moves are part of every T edit (§2.3).
- **Margins**: W−Z margin = logit(W) − logit(Z); q−g margin = logit(q) − logit(g). Each is reported as its change from the unedited jet, in logits, on W + Z jets (W−Z) or q + g jets (q−g), and per true class. Spread over main jets (job `e38-sd`): sd of the W−Z margin 3.11 logits over all jets (4.74 on W + Z jets); sd of the q−g margin 2.11 (2.62 on q + g jets).
- **Controls**
  1. *Soft-sector null*: a random zero-sum pT reshuffle among ranks 6–63 with the same L2 norm of the pT change (in GeV) as the T edit, followed by the same angle projection. A random half of the rank 6–63 particles each give the same fraction a ≤ 0.9 of their pT to 2 other random rank 6–63 particles (Dirichlet(1, 1) split); a is set by bisection. Achieved / target L2 has median 1.00 in every cell. This is the comparison for "does it matter that the pT moved belongs to the hardest particles?". An earlier version, which scaled Gaussian noise ∝ pT, reached only 5–13 % of the target size and was replaced before the final run.
  2. *Matched rank distance*: transfer of the same fraction f of the source particle's pT between two particles with the same rank distance |k − j| and the same direction, placed at random inside ranks 14–40. Same projection. Defined for the 30 single-rank cells; coverage 98–99.5 % of jets.
  3. *Rotation null*: rigid rotation of the whole particle pattern about the jet axis by a random angle (no pT change), followed by the same projection; 3 draws per jet.
  4. *Untrained*: 10 redraws (`Jedi(untrained=True, seed 0–9)`) on 2,000 main jets; all 42 cells at f = 0.2; the change in each redraw's W−Z and q−g margins is expressed in units of that redraw's margin sd.
  5. *Decomposition* (f = 0.2): the pT transfer alone, without the angle projection (the four-momentum, and so the mass, is not restored), and the projection's angle moves alone (no pT transfer).
- **Naming of S20, S51, S43** (and S1, S29, S30 for comparison). Basis: per feature, x, (x − q25)₊, (x − q50)₊, (x − q75)₊, (q25 − x)₊, with knots at the quartiles of unedited jets; ridge regression, α by cross-validation.
  - Leading-particle features: pT_0, pT_0 + pT_1, pT_0 − pT_1, pT_0 + pT_1 + pT_2, pT^D = √(ΣpT²)/ΣpT, z_lead = pT_0 / pT_J.
  - Radial features: Σ_{rank≥1}(r − 0.2)₊, M2 = Σ pT r², Σ r², Σ_{r<0.165} r, Σ_{top14} pT r, Σ_{top14} r, N(pT < 2 GeV).
  - The combined family uses both feature sets.
  - (A) fitted on unedited jets of a random half of main; predicts each edit's change.
  - (B) fitted directly on the edit changes of that half.
  - Test: OLS slope of observed on predicted change (in sd of the unit) over all edits of the held-out half; 95 % CI by resampling held-out jets. Pass if the CI is inside [0.8, 1.2].
  - Note: under (B), a slope near 1 on held-out data is expected for any linear fit. Its R² is the informative number.
- **Causal share on E36's minimal flips**. `results/jedi/e36_edits.npz` stores the unedited jets x0 and the edits v, so no rerun with seed 3803 was needed. Recomputing the edited jets reproduces the stored exact calls on 100 % of compliant edits. 1,142 primary (W/Z/q/g) jets flip to the target class.
  - Hold: the chosen S units are set to their unedited values in the edited jet's forward pass. Identity check: holding all 19 live S units at their *edited* values reproduces the edited logits exactly (max |Δlogit| 0).
  - Metrics: the fraction of flips undone (call no longer the target), and the margin removed = 1 − mean(m_hold − m_0) / mean(m_edit − m_0), where m = logit(target) − logit(unedited call).
  - Comparisons: the radial triple S1 + S29 + S30, and the 500 live-S triples (from the 19 live units) closest to the named triple in displacement norm (mean over flipped jets of √Σ(ΔS_u/sd_u)²).
  - Rank p = (1 + number of random triples ≥ the set) / 501. Holm correction over the 2 sets × 2 metrics.
- **Holds on T cells**: the same three holds (plus 30 matched random triples) on the 4 cells with the largest W−Z response at f = 0.2, on W + Z jets.
- **References**: 5-class MLPs (2 × 32, sklearn) trained on the true labels of the pool on (log m, log pT, N), (log m, log pT, N, pT^D) and (log m, log pT, N, z_lead). They are scored on the same edited jets (features recomputed), with margins = differences of log-probabilities. Accuracy on main: 78.1 %, 79.1 %, 78.8 %.
- **Statistics**: 1,000-sample jet bootstrap (95 % CI) for every cell mean; paired bootstrap for T − null. The bootstrap p has a floor of 0.001, which Holm over 126 cells inflates to 0.126. Cell p-values are therefore also given from the normal approximation to the bootstrap distribution, (mean / se with se = CI width / 3.92), with Holm over the 126 cells per margin. McNemar for paired call changes (T vs null), Holm over 126 cells. S units: paired bootstrap T − null for 19 units × 126 cells (2,394 tests, Holm).
- **Hypotheses.**
  - H0 (hold): holding S20/S51/S43 removes no more of the E36 flips than matched random triples.
  - H0′ (map): the T map equals the soft-sector null, cell by cell.
  - Thesis (hardness is a named rank-window object set) predicts:
    - S20/S51/S43 fit leading-particle hinges with edit slope 0.8–1.2;
    - the triple hold removes ≥ 0.5 of the E36 flips, while random triples remove a median ≤ 0.15;
    - the W−Z margin responds to transfers among ranks 0–2, matching the pT^D reference ratio within 0.8–1.2.
  - Alternative (hardness enters diffusely, through g or through context) predicts:
    - the fits fail the edit-slope test;
    - the hold is at the random level;
    - the map is smooth over ranks, like a total-pT reweighting (S0/g route, E30).

## 2. Results (main sample unless stated)

### 2.1 Rank × rank map, f = 0.2 (mean change in logits; 95 % CI in `e38.json`)
W−Z margin on W + Z jets (rows: source rank k, columns: destination j; "rest" = ranks 6–63):

| k \ j | 0 | 1 | 2 | 3 | 4 | 5 | rest |
|---|---|---|---|---|---|---|---|
| 0 | | +0.32 [0.25, 0.38] | +0.28 | +0.26 | +0.26 | +0.30 | +0.29 |
| 1 | +0.11 [0.07, 0.16] | | +0.11 | +0.13 | +0.16 | +0.24 | +0.18 |
| 2 | +0.14 | +0.12 | | +0.06 | +0.09 | +0.13 | +0.17 |
| 3 | +0.09 | +0.10 | +0.08 | | +0.10 | +0.10 | +0.12 |
| 4 | +0.07 | +0.07 | +0.07 | +0.08 | | +0.09 | +0.08 |
| 5 | +0.02 | +0.02 | +0.01 | +0.03 | +0.05 | | +0.04 |
| rest | +0.37 [0.30, 0.43] | +0.37 | +0.27 | +0.28 | +0.27 | +0.32 | |

- **Sign**: every cell is ≥ 0. The two directions of the same pair have the same sign (0→1 +0.32, 1→0 +0.11; 0→rest +0.29, rest→0 +0.37), although pT^D moves in opposite directions (0→1: −0.10 sd; 1→0: +0.08 sd; rest→0: +0.48 sd; 0→rest: −0.28 sd).
- **Per class**: the pooled W−Z change comes from Z jets.
  - Z jets: +0.03 to +0.89 per cell (mean +0.35), i.e. toward W.
  - W jets: −0.22 to +0.04 (mean −0.04), i.e. slightly toward Z.
  - Both classes therefore move toward the decision boundary.
- **Ranks 0–2 vs the rest of the map**: the 6 cells inside ranks 0–2 average +0.18, the other 36 cells +0.15. The largest rows are source = rank 0 and source = rest.
- **Scaling with f**: 0→1 +0.04 / +0.11 / +0.32 at f = 0.05 / 0.10 / 0.20; rest→0 +0.02 / +0.09 / +0.37.

q−g margin on q + g jets: positive (toward quark) in 41 of 42 cells:
- single-rank cells: +0.06 to +0.16;
- rest → k: +0.37 to +0.52;
- 0 → rest: −0.01 [−0.04, 0.01].

On gluon jets rest→0 gives +0.77, on quark jets +0.27.

Call changes (any change of the argmax) at f = 0.2: 2.3–10.4 % per cell. Largest: rest → k 9.2–10.4 %, then source rank 0 6.6–7.3 %.

### 2.2 Controls

| comparison (f = 0.2) | W−Z on W + Z | q−g on q + g | calls changed |
|---|---|---|---|
| T edits, range over 42 cells | +0.01 to +0.37 | −0.01 to +0.52 | 2.3–10.4 % |
| (i) soft-sector null, same GeV L2, range | +0.11 to +0.44 | +0.15 to +0.45 | 5.8–11.4 % |
| (ii) matched rank distance in ranks 14–40, range (30 cells) | −0.003 to +0.03 | −0.02 to −0.004 | 1.1–1.3 % |
| (iii) rotation null (no pT change) | +0.22 [0.17, 0.27] (mean \|Δ\| 0.72) | +0.04 [0.02, 0.05] | 5.5 % [5.0, 5.9] |

- **T − soft-sector null (H0′)**: the soft-sector null moves the margins *more* than T.
  - W−Z: negative in 126 of 126 cells (f = 0.2: −0.31 to −0.01). Holm < 0.05 in 100/126 cells (normal-approx p), none with T > null.
  - q−g: Holm < 0.05 in 119/126 cells. In 7 of them T > null: rest → 0, 1, 2, 3 at f = 0.10 and rest → 0, 1, 2 at f = 0.20 (e.g. rest→0 at f = 0.20: +0.09 [0.06, 0.12]). In the other 112, T < null.
  - McNemar (calls changed by T only : by null only): null-only exceeds T-only in all 126 cells, Holm < 0.05 in all. Examples: 0→1 146 : 268, 1→0 121 : 327, rest→0 145 : 208.
  - H0′ is rejected in the direction opposite to the thesis: per GeV moved, shuffling soft pT changes the network more than moving pT among the hardest particles.
- **T − matched rank distance**: T moves the margins more, e.g. 0→1 W−Z +0.28 [0.21, 0.36], 2→1 +0.10 [0.06, 0.14]. This control moves the same *fraction* of a much softer particle, so it moves far fewer GeV and needs far smaller angle corrections. Per fraction of the source particle's pT, transfers at ranks 0–5 matter more than at ranks 14–40.
- **Untrained (10 redraws)**: every redraw calls one class for all jets. The largest |cell mean| is 0.00–0.25 sd of the redraw's own W−Z margin (0.00–0.21 for q−g); the mean |cell| is 0.00–0.07 sd. The trained network's all-jet cell means are at most 0.06 sd (W−Z) and 0.17 sd (q−g), with mean |cell| 0.02 / 0.03 sd. In these units the trained response is not larger than the untrained one. The untrained margins have small spread, so sd units inflate their changes; this comparison is a coarse one.

### 2.3 Decomposition of T at f = 0.2 (W−Z on W + Z jets; q−g on q + g jets)

| cell | full T | pT transfer alone (mass not restored) | angle moves alone |
|---|---|---|---|
| 0→1 | +0.32 / +0.16 | −0.16 [−0.21, −0.11] / −0.01 | +0.45 [0.36, 0.54] / +0.17 |
| 1→0 | +0.11 / +0.15 | +0.13 [0.10, 0.16] / +0.02 | −0.04 [−0.09, 0.01] / +0.13 |
| 0→5 | +0.30 / +0.11 | −0.57 [−0.66, −0.47] / −0.06 | −0.06 / +0.20 |
| 0→rest | +0.29 / −0.01 | −0.79 [−0.87, −0.70] / −0.12 | +0.64 / +0.11 |
| rest→0 | +0.37 / +0.52 | +0.99 [0.89, 1.09] / +0.26 | +0.06 / +0.24 |

- **pT part alone**: a signed, rank-ordered response.
  - Moving pT out of rank 0 lowers the W−Z margin, increasingly with destination rank: 0→1 −0.16, 0→5 −0.57, 0→rest −0.79.
  - Moving pT into rank 0 raises it (rest→0 +0.99).
  - This edit changes the jet mass, so it does not separate hardness from mass.
- **Angle moves alone**: the angle corrections that restore the mass shift the q−g margin toward quark by +0.03 to +0.24 in every cell. Their W−Z effect varies by cell. In the full edit, the angle part and the pT part do not add linearly.

### 2.4 S units under T (mean ΔS/sd over all jets; f = 0.2)
- The largest |cell mean| per unit: S13 0.58, S43 0.43, S40 0.30, S20 0.29, S51 0.27, S29 0.26, S56 0.21. S0 (ΣpT of the 14 hardest) ≤ 0.02, S1 ≤ 0.07, S30 ≤ 0.10, S19 ≤ 0.08.
- Example cells:
  - S20: 0→1 −0.22, 0→rest −0.28, rest→0 +0.11.
  - S51: 0→1 −0.14, 0→rest −0.27, rest→0 +0.11.
  - S43: rest→0 +0.43, 0→rest −0.11.
- The soft-sector null moves S20 and S51 by −0.04 to −0.11 and S43 by +0.08 to +0.16.
- Paired T − null per unit and cell: 2,394 tests, Holm-corrected p in `e38.json` (`S_T_minus_null_p_holm`; bootstrap floor as above).
- Under T, S20/S51/S43 move 0.1–0.4 sd. Under E36's minimal flips the same units moved 1.2–2.0 sd (recomputed here: S20 2.00, S51 1.27, S43 1.25).

### 2.5 Naming (held-out half: 2,500 jets × 126 edits)

| unit | family | (A) fitted on unedited jets: edit slope [95 % CI], R² of changes | (B) fitted on changes: slope, R² | R² on unedited held-out jets |
|---|---|---|---|---|
| S20 | leading-particle | **0.97 [0.94, 1.00]**, 0.26 | 0.97, 0.31 | 0.93 |
| S20 | radial | 0.54 [0.51, 0.57], 0.37 | 1.01, 0.45 | 0.48 |
| S20 | leading + radial | 1.20 [1.16, 1.24], 0.60 | 1.01, 0.66 | 0.96 |
| S51 | leading-particle | 0.43 [0.40, 0.45], 0.06 | 0.96, 0.13 | 0.80 |
| S51 | radial | 0.67 [0.65, 0.69], 0.42 | 0.99, 0.63 | 0.69 |
| S51 | leading + radial | **1.03 [1.01, 1.05]**, 0.69 | 0.98, 0.76 | 0.97 |
| S43 | leading-particle | 0.41 [0.37, 0.45], 0.03 | 0.99, 0.21 | 0.12 |
| S43 | radial | 0.72 [0.69, 0.76], 0.23 | 0.99, 0.73 | 0.85 |
| S43 | leading + radial | 1.22 [1.20, 1.25], 0.62 | 0.99, 0.79 | 0.97 |
| S1 / S29 / S30 | radial | 1.09 / 1.35 / 0.97 (CIs in json); R² 0.90 / 0.81 / 0.78 | ≈ 1.0; 0.90 / 0.89 / 0.84 | 0.96 / 0.98 / 0.98 |

- **Pass/fail (A, slope CI inside [0.8, 1.2])**:
  - S20 passes with leading-particle formulas alone, but those explain 26 % of the variance of its changes.
  - S51 passes only with the combined formula.
  - S43 fails with every family. Its unedited values are named by the radial family (R² 0.85), not by the leading-particle family (0.12).
- **Variance explained under (B)**: the radial family explains more of the change of all three units than the leading-particle family (S20 0.45 vs 0.31, S51 0.63 vs 0.13, S43 0.73 vs 0.21). Under T, part of each unit's change comes from the angle corrections.

### 2.6 Causal share on E36's minimal flips (N 1,142 flipped jets; edit raised the target margin by +3.53 [3.40, 3.65] logits)

| held at unedited values | flips undone [95 % CI] | margin removed [95 % CI] | rank p (flips; margin), Holm |
|---|---|---|---|
| S20 + S51 + S43 (leading-particle) | 0.476 [0.446, 0.505] | 0.034 [0.009, 0.061] | 0.60; 0.80 → 1.0; 1.0 |
| S1 + S29 + S30 (radial) | 0.576 [0.548, 0.604] | 0.384 [0.351, 0.423] | 0.31; 0.032 → 0.92; 0.13 |
| 500 matched random live-S triples: median / 95th pct / max | 0.505 / 0.667 / 0.817 | 0.112 / 0.361 / 0.457 | |

- **Matching**: the named triple has the largest displacement of any triple (2.82). The random triples' norms are 0.92 / 1.52 / 2.51 (5th / 50th / 95th percentile), so the matched set moves less than the named one.
- **Why "flips undone" is a weak metric here**: the minimal edits leave the flipped jets close to the decision boundary, so any hold undoes about half of the flips. The margin-removed column is the informative one.
- **By flip group** (flips undone):
  - named triple: W→Z 0.20, Z→W 0.25, q→g 0.59, g→q 0.77;
  - radial triple: W→Z 0.70, Z→W 0.93, q→g 0.56, g→q 0.33.
- **H0 (named triple removes no more than random)** is not rejected (rank p 0.80 on margin removed). The named triple removes less margin than the median random triple.

**Holds on T cells** (f = 0.2, W + Z jets; share of the W−Z change removed):

| cell | ΔW−Z | S20+S51+S43 | S1+S29+S30 | 30 random triples: median / max | rank p (named; radial) |
|---|---|---|---|---|---|
| rest→0 | +0.37 | −1.37 [−1.69, −1.11] | 2.99 | −1.50 / 1.46 | 0.32; 0.03 |
| rest→1 | +0.36 | −1.53 | 3.57 | 0.12 / 4.02 | 0.90; 0.06 |
| rest→5 | +0.32 | −1.21 | 3.79 | 0.03 / 4.06 | 0.94; 0.06 |
| 0→1 | +0.32 | −0.51 [−0.72, −0.35] | 1.67 | −0.79 / 1.09 | 0.39; 0.03 |

- A negative share means that holding the leading-particle triple *increases* the W−Z change: its own change under T pushes the W−Z margin the other way.
- A share above 1 means that holding the radial triple reverses the sign of the change.
- The random triples spread from about −1.5 to +4, so neither set is outside that range.

### 2.7 References (MLPs on jet-level features)
- The (m, pT, N) reference changes by exactly 0, because T preserves m, pT and N. Its ratio is undefined.
- The pT^D and z_lead references change by at most 0.03 (W−Z) and 0.16 (q−g) log-probability. Network / reference ratios:
  - W−Z: 10–280 in magnitude, of either sign; for 0→1 the CI spans 0.
  - q−g: rest→0 3.5 [3.3, 3.8] (pT^D) and 3.2 [3.1, 3.4] (z_lead); 1→0 7.7; 0→1 −9.1.
- None is inside 0.8–1.2. A jet-level classifier that knows pT^D or z_lead barely moves under these edits; the network moves 3–280 times more.

### 2.8 Replication (`sample(1000, 3802)` minus overlaps, N 4,900; f = 0.2)
Accuracy 80.57 %. Correlation of the 42 cell means with main: W−Z 0.958, q−g 0.998. Examples: 0→1 +0.27 / +0.18, rest→0 +0.31 / +0.50, 0→rest +0.29 / −0.01 (W−Z / q−g). T − null W−Z ranges from −0.25 to +0.03.

## 3. Reading (scoped to these edits)
- **Thesis predictions vs. measurements**:
  - Edit slope near 1 for leading-particle hinges: met for S20 only, and with low R². S51 and S43 are not named by leading-particle formulas under these edits.
  - Triple hold ≥ 0.5 of E36 flips with random median ≤ 0.15: not met. Margin removed is 0.03 vs a random median of 0.11, and flips undone 0.48 vs 0.51.
  - W−Z response concentrated in ranks 0–2 and matching the pT^D reference: not met. The response has the same sign for both transfer directions, sits mostly in the rows with source rank 0 or rest, and exceeds the reference by 10–280×.
- **Alternative predictions**:
  - Hold at the random level: met.
  - Fits failing the slope test: met for S51 and S43.
  - "Smooth map like total-pT reweighting through S0/g": not supported in that form, since S0 moves ≤ 0.02 sd. The map does vary smoothly with source rank, and soft-sector reshuffles of the same GeV give larger responses.
- **Physics reading**: at fixed jet four-momentum, these perturbations move W and Z jets toward each other's class, mostly Z → W, and quark and gluon jets toward quark. This happens whether the moved pT goes into or out of the hardest particles. The edit's direction in pT^D does not set the sign; the size of the perturbation does. The signed hardness dependence appears only in the pT-transfer-alone edit, which also changes the jet mass.
- **E36 link**: S20/S51/S43 move most under E36's minimal flips but carry ~3 % of the flip margin. The radial triple, which moves 0.49 sd in norm, carries 38 % (rank p 0.03, Holm 0.13). The size of an object's move under an edit is not a measure of its causal share.

## 4. Issues and limits
1. **Angle moves are part of every edit.** The angle corrections needed to keep the mass fixed are 3–47 mrad (pT-weighted), and up to ~150 mrad for single particles when "rest" is the source. The decomposition shows they matter on their own, especially for q−g. A pT-only edit at fixed four-momentum is not possible.
2. **Soft-sector null size.** The null matches the L2 in GeV. In relative terms it is a much larger edit for the soft particles (a = up to 0.9 of their pT), so "same size" is metric-dependent. The matched-rank-distance control matches the fraction instead and gives the opposite ordering. Both are reported.
3. **Matched random triples.** The named triple's displacement norm is larger than every random triple's (95th pct 2.51 vs 2.82), so the matched null is weaker in displacement than the named set. This favours the named set, and it still ranks below the median.
4. **p-values.** The bootstrap p floor (0.001) times 126 cells gives a Holm floor of 0.126. The normal-approximation p-values are reported for the cell tests, and McNemar is exact.
5. **Untrained comparison.** It is in sd units of the untrained margins. Those networks call one class for every jet and have tiny margin spreads, so it is a coarse control.
6. **Compute.** The first full run (`e38-run`) was stopped to replace the soft-sector null (it reached only 5–13 % of the target size). The second (`e38-run2`) was OOM-killed at 2 GiB during the f = 0.2 loop. The final run `e38-run3` (forward batch 1,000) used 1 CPU / 3 GiB: wall 1,621 s, average 0.71 cores, pod memory peak 3,071 MiB including page cache. `e38-sd` (margin spread) used 1 CPU / 2 GiB.

## 5. Explainer entry (draft for site/explain.py; Paper A tab)
- **Question**: does JEDI-linear have a separate "hardness" channel, a few internal objects that read how pT is shared among the hardest particles and carry the network's response to it?
- **Background**: E36 searched for the smallest edits that flip the network's call at fixed jet mass, pT and axis. They mostly moved pT among the hardest particles and moved three internal second-round objects most: S20 and S51 (hinge functions of the summed pT of the 2–5 hardest particles) and S43 (a rank-13|14-window object). pT^D $= \sqrt{\sum_i p_{T,i}^2}/\sum_i p_{T,i}$ measures how unevenly pT is shared among particles; $z_{\rm lead} = p_{T,0}/p_{T,J}$ is the leading particle's momentum fraction.
- **Method**: edit $T(k\to j, f)$: $p_{T,k} \to (1-f)\,p_{T,k}$, $p_{T,j} \to p_{T,j} + f\,p_{T,k}$. Then the minimal change of all particle angles restores $\sum_i p_i^\mu$, so $m$, $p_T$ and the E-scheme axis are unchanged.
  - Measure the change of $\ell_W - \ell_Z$ and $\ell_q - \ell_g$ (logit margins) per cell.
  - Compare with a random reshuffle of the same GeV among ranks 6–63 (soft-sector null), the same fraction moved between ranks 14–40, a rigid rotation, and untrained networks.
  - Hold S20 + S51 + S43 at unedited values on E36's flipped jets. Margin removed $= 1 - \overline{(m_{\rm hold}-m_0)}/\overline{(m_{\rm edit}-m_0)}$, compared with 500 random triples (rank p) and with the radial triple S1 + S29 + S30.
- **Result**: figure `results/jedi/e38_hardness.png`.
  - (a) The edit on a real W jet.
  - (b) W−Z margin change map: positive in every cell (toward W for Z jets) for both transfer directions; the soft-sector null of the same GeV gives +0.11 to +0.44.
  - (c) Margin removed on E36 flips: leading-particle triple 0.03, radial triple 0.38, random median 0.11.
- **Checks**:
  - holding all live S units at their edited values reproduces the edited logits exactly;
  - the stored E36 calls are reproduced on 100 % of jets;
  - the replicate map correlates 0.96 / 1.00 with main;
  - constraint compliance is 98.9 %.
- **Why it matters**: the objects that move most under a minimal flip are not the ones that cause it. The hardness direction found by E36 is not carried by a separate named channel. The network's response to sharing pT differently at fixed mass is not specific to the hardest particles.

## 6. Glossary additions
- **T(k→j, f)**: move the fraction f of the pT of the rank-k particle to the rank-j particle (rank = position in pT order, 0 = hardest; "rest" = ranks 6–63 pooled), then restore the jet four-momentum by the smallest change of particle angles.
- **Soft-sector null**: random pT reshuffle among ranks 6–63 with the same L2 norm of the pT change, in GeV, as the edit it is compared with.
- **Margin removed**: the share of an edit's mean margin change that disappears when chosen internal units are held at their unedited values.
- **Rank p**: (1 + number of random unit sets that do at least as well) / (1 + number of random sets).
- **z_lead**: the leading particle's share of jet pT.
- **pT^D**: the pT dispersion √(ΣpT²)/ΣpT.

## 7. Conclusion (for site/conclusions.py)
Moving pT among the hardest particles at fixed jet mass, pT and axis shifts the network's W-vs-Z and quark-vs-gluon scores in the same direction whichever way the pT goes. Reshuffling the same number of GeV among soft particles shifts them more. The three internal objects that respond most to the hardest particles' pT (S20, S51, S43) carry about 3 % of the score change behind E36's minimal call flips, less than random object triples (median 11 %). The hardness direction is therefore not a separate named channel in this network. Uncertain: the angle corrections that keep the mass fixed are part of every edit and cannot be separated from the pT change.
