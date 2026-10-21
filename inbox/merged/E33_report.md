# E33 — Recurrence of the JEDI-linear program across official checkpoints and retrained copies

Node E33, round 15, stage 3, claims A1, A3, A10, A16 (review weakness "multi-seed / retrained stability"; Paper B transfer of the
naming method). Status: done (no verifier yet). Script `interp/e33_program_recurrence_jedi.py` (`--model <m>`, `--combine`, `--plot`,
`--smoke`). Jobs `e33-n64, e33-n32, e33-n128, e33-perminv, e33-f16, e33-hgq0, e33-hgq1, e33-hgq2, e33-float0, e33-float1, e33-untr`
(each 1 CPU / 2 GiB; 0.6–1.0 cores used, peak 1.1–2.0 GiB incl. page cache; wall 9–52 min) and `e33-combine`. Results
`results/jedi/e33_<model>.json`, `results/jedi/e33_summary.json`, figure `results/jedi/e33_recurrence.png`, logs `logs/e33-*.log`.
Smoke runs (`*_smoke.json`, 1,000 jets) are kept for reference only.

## Conclusion

Second-round jet quantities with the same formulas as the reference network's "sum of pT of the hardest particles", "pT-weighted
squared radius", and "soft / empty-slot count" are found (held-out R² ≥ 0.9, and a one-particle edit moves them more than a matched
control edit) in 5, 9 and 7 of the 10 trained networks, against about 0.1, 1.2 and 0.05 networks expected from random rotations of the
same unit space; the core-radius sum is found by fit in only 3 networks and fails its causal control in all of them. Under the strict
plan criterion (edit slope interval covering 1) only 2 networks pass, for the hardest-particle sum, because the formulas are fitted on
whole jets and over- or under-predict single-particle changes (e.g. the pT-weighted squared-radius formula over-predicts a 10 %
radial move by about 1.5× in every network). The two quantization-free (float) retrained copies are the exception: their second-round
layer is spread over all 64 units, few units are nameable, and pruning keeps 90–98 % of their units. Uncertain: the per-model
rotation tests are too weak to pass Holm on their own, and the mass-window test as defined finds window edges at many masses in every
trained network, so it does not separate the reference windows from others.

## Question

The named elements of JEDI-linear 3-feature n64 (13 l1 cuts of two kinds; rank-window radial hinge objects S0, S29, S19, S30; z mass
windows) were found on one checkpoint. Are they a property of training this architecture on this task (recurring in other official
checkpoints and in independently retrained copies), or of one checkpoint?

## Background (definitions)

- **Stages / units**: `a` (64 per-particle l1 cuts), `P` (pooled cuts), `g` (jet context), `s`, `u` (per-particle), `S` (64
  second-round jet quantities, "objects"), `h4`, `h5`, `z` (16 pre-ReLU neurons), 5 logits. A per-particle stage counts one unit per
  channel. **Live unit**: activation s.d. > 1e-9 over the 10,000 fit jets.
- **Rank**: the slot index of a particle after sorting by pT (rank 0 = hardest). **r** = $\sqrt{\Delta\eta^2+\Delta\phi^2}$ to the
  stored jet axis (rad).
- **Corruption**: *mean* = a removed unit set to its fit-sample mean (per slot for per-particle stages); *resample* = set to its value
  on a donor jet of the partner class (W↔Z, q↔g, t→W or Z) among the 5 nearest in jet ΣpT.
- **F** = $(\mathrm{acc}_{\rm circ} - 0.2)/(\mathrm{acc}_{\rm full} - 0.2)$: share of the network's above-chance accuracy that a
  circuit keeps (every unit outside the circuit corrupted). **circuit95** = smallest pruned circuit with F ≥ 0.95.
- **Roles** (fixed formula families with fitted parameters; each fitted on the fit jets, R² on the held-out test jets):
  - hard: $a + b\sum_{\rm rank<k} p_T$ (k = 1..n) — reference S0 (ΣpT of the 14 hardest, here fitted k = 55);
  - r2: $a + b\sum p_T r^2 + c\sum r^2 + d\,N(p_T<2\,\mathrm{GeV}, r>0.1)$ — reference S29;
  - soft: $a + b\sum_{\rm slots\ \ge k}(t-p_T)_+$, an empty slot counting $t$ — reference S19 (ReLU(2.86 − 1.39 pT) on slots 13–63);
  - core: $a + b\sum_{\rm slots}(t-r)_+$, an empty slot counting $t$ — reference S30 (ReLU(3.47 − 21 r), 3.38 per empty slot).
  Roles are given to distinct units (Hungarian assignment of 4 roles to live S units maximising summed fit R²).
- **Edit slope**: per test jet one particle is edited; predicted ΔS = fitted formula(edited) − formula(unedited); measured ΔS from the
  full network (g recomputed). Slope = $\sum \Delta S_{\rm obs}\Delta S_{\rm pred}/\sum \Delta S_{\rm pred}^2$; 1 = the formula's
  prediction is met. **Control ratio** = mean |ΔS| under a control edit the formula predicts to be 0, divided by mean |predicted ΔS|
  of the in-window edit on the same jets.
- **Rotation null (X11v-style)**: the live S activations are standardised and rotated by a random orthogonal matrix (250 draws); the
  same role fit is applied to the rotated directions. Null rate = share of rotations with a direction reaching test R² ≥ 0.9.
- **Hungarian match**: one-to-one assignment of the model's live S units to the reference's 19 live S units maximising mean
  |Pearson correlation| on the shared test jets.
- **Holm**: step-down multiple-comparison correction. **Poisson-binomial tail**: probability that independent models with the given
  null rates produce at least the observed number of hits.

## Procedure

Models: official 3-feature n64 (reference), n32, n128, 3-feature-perminv n64, 16-feature n64 (official checkpoints via
`jedi_core.Jedi`); X12 retrained HGQ copies (`hgq-r150` seeds 0–2) and float copies (seeds 0–1), final checkpoints; control row:
`Jedi(64, untrained=True, seed=0)`.
Samples: fit = `sample(2000, 3301)`, test = `sample(2000, 3302)`, N = 10,000 each (n32: first 32 slots; n128: same rows of
`val_ptetaphi_f16.npy`; 16-feature: same rows of `val_c16_f16.npy`; both checked equal to `sample()`, max difference 0).
1. Forward with a per-stage override for all nine stages; checked against `Jedi.run` / `x12_retrained_jedi.Net.run`: max |Δlogit| 0,
   0 calls differ (1,000 jets) for every model.
2. Node-level ACDC-style pruning on fit[::10] (1,000 jets): receivers z, h5, h4, S, u, g, s, P, a; units within a stage in ascending
   single-unit-ablation KL; a unit is dropped if KL(full ‖ circuit) rises by less than τ; τ = 1e-5 … 3e-2 (8 values, nested).
   F of each ladder circuit on the 10,000 test jets; circuit95; null = 500 random circuits with the same units per stage, F on
   test[::10]; p = (1 + #null ≥ obs)/501.
3. Naming of each live S unit: greedy forward selection (≤ 8 terms) over 665 rank-window sums of per-particle hinge functions
   of (pT, ln pT, r); named iff test R² ≥ 0.9. l1 cut types by spread-weighted weight share (≥ 0.9 pT: ramp; ≥ 0.9 angle: angular;
   16-feature: largest E25 feature group).
4. Role fits, Hungarian role assignment, rotation null; Hungarian match to the reference S space.
5. Edits (test jets): hard — one particle of rank < k pT × 1.1 (capped at the pT of rank − 1 so ranks are kept); control: same on a
   rank ≥ k particle. r2 — one particle radially × 1.1 at fixed pT, azimuth and rank; control: same particle rotated 0.1 rad about the
   axis (same displacement, formula change 0). soft — the hardest particle at rank ≥ max(k, 1) with 0.5 t ≤ pT ≤ t raised to
   min(1.2 t, pT of the next harder particle) (crosses the fitted pT edge); control: the same pT step on a particle with pT > 1.5 t.
   core — a particle with r < t/1.1 moved radially × 1.1; controls: a particle with r > t moved radially × 1.1, and the azimuthal move.
   1,000-draw jet bootstrap for slopes and ratios; paired sign-flip p (10,000) for E|ΔS_in| = E|ΔS_ctrl|.
6. z windows: all real particles' (Δη, Δφ) × 1.05 and × 0.95 (jet mass ± 5 %); per jet and z unit R = Δz / Δ ln(scale); jets in 10 GeV
   mass bins 20–250 GeV (≥ 40 jets); window centre = mass where the bin mean of R crosses from + to − with |mean/se| ≥ 3 within
   30 GeV on each side; 200-draw bootstrap CI.

Changes after the smoke runs (before the main run, recorded in the docstring): soft was a step count N(pT > t, rank ≥ k): a single
particle crossing the step edge moved S19 by 0.07 of the count prediction (the unit is a pT ramp); core was a + b Σ(t − r)₊ + c N, and
the N term let the soft-count unit win the core role; the four roles were made to go to distinct units. Secondary criteria
`found_equiv` (slope CI inside [0.8, 1.25]) and `found_dir` (R² ≥ 0.9, slope CI > 0, in-window |ΔS| > every control's |ΔS|, sign-flip
p < 0.05) were added then and are reported beside the plan's criterion. The untrained network calls one class for all jets (accuracy
20.0 %), so its circuit search was skipped (F undefined; first attempt crashed on the division, fixed).

## Results per model

Reference check: official n64 test accuracy 80.82 % [80.04, 81.58] (Wilson 95 %), inside the expected 80.6–81.3 %.

Columns: accuracy; live units; circuit95 size / live (mean | resample corruption), F of circuit95 vs random circuits of equal size on
test[::10] (mean corruption, p); named S / live S; pure l1 / live l1; roles hard / r2 / soft / core: test R² and edit slope [95 % CI]
(✓ = plan criterion, * = found_dir); Hungarian mean |corr| with the reference vs rotation null (p).

| model | acc | live | circuit95 (share) | F95 vs random | named S | pure l1 | hard | r2 | soft | core | match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| official n64 (ref) | 80.8 % | 186 | 126 (0.68) \| 154 (0.83) | 0.98 vs 0.19, p 0.002 | 17/19 | 13/13 | 0.93, 1.16 [1.11, 1.20] * | 0.97, 0.70 [0.66, 0.74] * | 1.00, 1.07 [1.05, 1.09] * | 0.97, 0.88 [0.83, 0.93] | – |
| official n32 | 78.1 % | 181 | 142 (0.78) \| 155 (0.86) | 0.97 vs 0.37, p 0.002 | 17/19 | 12/12 | 1.00, 0.99 [0.99, 1.00] * | 0.96, 0.71 [0.68, 0.75] * | 0.96, 0.45 [0.40, 0.51] * | 0.98, 0.87 [0.83, 0.92] | 0.75 vs 0.62, p 0.004 |
| official n128 | 80.8 % | 192 | 145 (0.76) \| 159 (0.83) | 0.98 vs 0.26, p 0.002 | 16/19 | 15/15 | 0.97, 1.06 [1.05, 1.07] (no control: k = 128) | 0.92, 0.30 [0.24, 0.37] * | 0.76, 0.21 | 0.77, 2.26 | 0.83 vs 0.65, p 0.004 |
| perm-inv n64 | 81.6 % | 206 | 146 (0.71) \| 169 (0.82) | 0.99 vs 0.39, p 0.002 | 16/21 | 17/17 | 0.93, 1.04 [1.02, 1.06] (no control: k = 64) | 0.96, 0.64 [0.58, 0.71] * | 1.00, 1.28 [1.27, 1.29] * | 0.94, 1.34 | 0.86 vs 0.67, p 0.004 |
| 16-feature n64 | 81.6 % | 219 | 154 (0.70) \| 174 (0.79) | 1.00 vs 0.15, p 0.002 | 16/23 | 13/17 | 0.93, 1.03 [1.03, 1.04] * | 0.93, 0.42 [0.39, 0.45] * | 0.97, 0.65 [0.62, 0.67] * | 0.85, 1.12 | 0.81 vs 0.65, p 0.004 |
| retrained HGQ s0 | 76.8 % | 146 | 104 (0.71) \| 85 (0.58) | 1.01 vs 0.29, p 0.002 | 9/11 | 13/13 | 0.99, 1.01 [0.99, 1.02] ✓ | 0.95, 0.62 [0.57, 0.68] * | 0.99, 0.38 [0.28, 0.49] * | 0.83, 1.41 | 0.92 vs 0.72, p 0.004 |
| retrained HGQ s1 | 57.4 % | 128 | 115 (0.90) \| 106 (0.83) | 1.00 vs 0.64, p 0.052 | 6/7 | 12/12 | 0.81, 0.77 | 0.94, 0.67 [0.61, 0.73] * | 0.65, 0.14 | 0.54, 1.09 | 0.83 vs 0.74, p 0.008 |
| retrained HGQ s2 | 78.1 % | 143 | 114 (0.80) \| 117 (0.82) | 0.99 vs 0.45, p 0.002 | 8/10 | 13/13 | 0.98, 1.00 [1.00, 1.01] ✓ | 0.95, 0.65 [0.60, 0.71] * | 0.99, 0.79 [0.75, 0.83] * | 0.81, 1.38 | 0.92 vs 0.72, p 0.004 |
| retrained float s0 | 82.7 % | 496 | 483 (0.97) \| 448 (0.90) | 1.01 vs 0.90, p 0.002 | 12/64 | 37/64 | 0.63, 1.09 | 0.90, 0.46 [0.42, 0.50] * | 0.94, 0.09 [0.09, 0.09] * | 0.88, 0.94 | 0.81 vs 0.69, p 0.004 |
| retrained float s1 | 82.4 % | 496 | 485 (0.98) \| 459 (0.93) | 1.00 vs 0.92, p 0.004 | 8/64 | 34/64 | 0.69, 0.97 | 0.90 (0.8996), 0.43 | 0.71, 0.03 | 0.81, 1.02 | 0.79 vs 0.69, p 0.004 |
| untrained redraw | 20.0 % | 69 | – (constant call) | – | 0/1 | 9/9 | none | none | 0.00 | none | 0.06 vs 0.06 |

Row conclusions (plain terms):
- **official n64 (reference)**: The four reference quantities are found again on a new sample with R² 0.93–1.00, and the pruning keeps
  all 13 cuts and 15–17 of the 17 hand-named objects (E29's names). Their single-particle edit responses match the formula for the soft
  count (slope 1.07) but not exactly for the other three (S0 1.16 through the context route; the squared-radius formula over-predicts
  a radial move by about 1.4×).
- **official n32**: Every role has a unit with R² ≥ 0.96; with 32 slots the hardest-particle sum uses the 10 hardest particles and
  responds as predicted (slope 0.99). The soft-count unit reads particles above 5 GeV beyond rank 20 and responds to an edge crossing at
  about half the formula's size.
- **official n128**: The hardest-particle sum (all 128 particles) and the squared-radius quantity recur; no unit fits the soft or core
  forms at R² ≥ 0.9 (best 0.76, 0.77). Its z units show no window crossing below 95 GeV in this test.
- **perm-inv n64**: Hard, squared-radius and soft roles recur (soft R² 1.00); having no rank-dependent masks, its hardest-particle
  unit is a sum over all 64 particles (k = 64), so the rank-window part of the reference S0 does not recur.
- **16-feature n64**: All four roles have units (R² 0.85–0.97); its "hardest-particle" unit is the leading particle's pT alone
  (k = 1, slope 1.03).
- **retrained HGQ s0, s2**: Same set of objects as the reference (Hungarian mean |corr| 0.92 vs 0.72 for rotations), and the hardest-
  particle sum meets its formula exactly (slope 1.01, 1.00; the only plan-criterion passes); their soft-count units respond to an edge
  crossing at 0.4 and 0.8 of the formula.
- **retrained HGQ s1** (57 % accuracy, a poorly trained seed): only the squared-radius quantity recurs at R² ≥ 0.9.
- **retrained float s0, s1**: No quantization or zero-bit masks; all 64 S units are live, 8–12 are nameable, about half the l1 cuts are
  mixed (pT and angle), and pruning keeps 90–98 % of the units (random circuits of that size keep F ≈ 0.9 under mean corruption). The
  hardest-particle sum is absent (R² 0.63, 0.69); the squared-radius quantity sits at R² 0.90.
- **untrained redraw**: constant call; one live S unit, no role fitted. Control for "objects appear without training".

## Recurrence statistics (10 trained models; null from the rotation rates, floored at 1/251 per model)

| role | plan criterion (R² ≥ 0.9, slope CI ∋ 1) | found_dir (R² ≥ 0.9, edit > control) | R² ≥ 0.9 only | expected under rotation null | p (found_dir), Holm-relevant |
|---|---|---|---|---|---|
| hard | 2 (HGQ s0, s2); p 0.0025, Holm 0.010 | 5 | 7 | 0.08 | 3.8e-9 |
| r2 | 0; p 1 | 9 | 9 | 1.18 | 1.0e-8 |
| soft | 0 (1 by slope CI in [0.8, 1.25]: reference) | 7 | 7 | 0.05 | 5.4e-15 |
| core | 0 | 0 (outer-particle control moves the unit 2.1–8.3× the in-window prediction) | 3 | 0.08 | 1.0 (R² only: 5.3e-5) |

Per-model Holm over 4 roles × 10 models of the rotation p (min attainable 1/251): no single (model, role) passes (smallest Holm p
0.10). H0 (roles recur at the rotation rate) is rejected for hard, r2 and soft by the across-model count; for core only by R².
Competing predictions: thesis expected all four roles in ≥ 6 of 9 models — met by r2 (9) and soft (7) under found_dir, hard reaches
5 (7 by R²), core 3 by R²; the single-checkpoint alternative (roles only in official n64 and 3-feature relatives) is excluded for hard,
r2 and soft (present in retrained HGQ copies and the 16-feature network). The float copies decompose differently (no hardest-particle
unit, few named units), as the alternative predicted for them.

Circuit sizes: circuit95 / live = 0.68–0.80 (mean corruption) in the 8 quantized models vs the expected 0.50–0.65; 0.97–0.98 in the
float copies. Every circuit95 beats 500 equal-size random circuits (p 0.002–0.004) except HGQ s1 under mean corruption (p 0.052).
The hand-named set (pure cuts + named S) lies inside circuit95 in every model, but random circuits of the same size contain nearly
all of it too (p 0.19–1.0), so this containment does not distinguish the names.

z windows: every trained model shows 9–16 crossings (+ to −) between 30 and 180 GeV; every model except n128 has crossings in each of
the three reference bands (65–85, 85–110, 120–160 GeV). The reference reproduces z7 100, z14 101, z10 147, z9 145 GeV; its z0 shows no
crossing at 75 GeV under this definition (z11 76, z15 68 do). Because crossings occur at many masses in every model and no null for
crossing positions was run, band hits are not evidence of recurrence of specific windows.

## Reading (scoped to what was measured)

- The hardest-particle sum, the pT-weighted squared-radius quantity and the soft/empty-slot ramp each appear as a single second-round
  unit with held-out R² ≥ 0.9 and a causal single-particle response above control in most quantized JEDI-linear networks (official
  n32/n64/n128, perm-inv, 16-feature, two of three retrained HGQ seeds), at rates far above random rotations of the same S spaces.
- The jet-level formulas do not predict single-particle edits exactly: r2 slopes are 0.30–0.71 in every model (the true per-particle
  function is a hinge, E5), and the core sum responds to particles outside its fitted radius in every model (consistent with the core
  unit's context input, E4), so "core" as a formula does not recur causally.
- The float copies do not show the sparse object structure (dense, mostly unnamed S layer); the structure tracks quantization / HGQ
  pruning more than the task alone.

## Issues / limitations

- Strict plan criterion (slope CI covers 1) is very tight at N = 10,000 (CIs ± 0.01–0.05); secondary criteria were defined after the
  smoke run and are labelled as such.
- Rotation null tests R² only (an upper bound on the joint rate); per-model resolution 1/251 cannot pass Holm over 40 tests.
- 16-feature edits recompute 13 columns; rotated-frame columns (etarot, phirot, costhetarel) are unchanged for one-particle edits.
- Hand-named-containment test is uninformative (named units are nearly all units of their stages).
- z-window definition is lenient; a crossing-position null (e.g. untrained-weights-with-trained-head, or shuffled mass) is missing.
- Node cph-dgx-node6.humboldt.edu stalled CephFS reads (process in D state); added to `job.sh`'s node exclusion list.
- No verifier run yet; suggested: fresh samples, per-particle hinge (E5-style) predictions instead of jet-level formulas for the r2 and
  core edits, and a z-window crossing null.

## Explainer entry (for site/explain.py)

- **Question**: Are the objects named in one JEDI-linear checkpoint present in other checkpoints?
- **Background**: second-round objects $S_c = \frac1{16}\sum_i u_c(p_{T,i}, r_i)$ over rank windows; reference roles S0
  ($\approx\sum_{\rm rank<14}p_T$), S29 ($\approx\sum p_T r^2+\sum r^2$), S19 ($\approx\sum_{\rm slots\ge 13}(2.06-p_T)_+$), S30
  ($\approx\sum(0.165-r)_+$).
- **Method**: same pipeline on 10 networks (pruning, naming, role fits with Hungarian assignment, one-particle edits with controls,
  rotation null, mass dilation for z).
- **Result**: figure `results/jedi/e33_recurrence.png` (left: role R² and edit slope per network; middle: pruned-circuit share; right:
  z window crossings). Hard / r2 / soft roles in 5 / 9 / 7 of 10 networks vs ≤ 1.2 expected; float copies differ.
- **Checks**: forward = reference implementation (0 calls differ); reference accuracy 80.8 %; untrained control has no objects.
- **Why it matters**: objects are not a single-checkpoint artefact in quantized JEDI-linear networks, but their formulas are jet-level
  approximations, and float training gives a different decomposition.
- New glossary terms: **circuit95**, **role (E33)**, **rotation null**, **found_dir**.
