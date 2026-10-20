# X10 verify — fresh-sample rerun of the DAS run, plus a source-call baseline, a class-transfer subspace, a no-clip evaluation and a representation check

Status: done. Confidence that the X10 reading is correct as stated: 0.6 (numbers reproduce; three qualifications below change how
Second verification (2026-10-19, x10-verify3 / x10-verify3u, section "Second verification" at the end): angle-only Δ edit and
the full-swap baseline for the untrained control; confidence after it: 0.45.
the comparison with random subspaces and the Δ result should be read).
Code: `interp/x10_verify.py` (imports X10's `JNet`, `das_fit`, `proj`, `scores`, `boot_eff`, `boot_diff`, `rand_Q`, `do_edit`,
`wz_call`, `calibrate` unchanged from `interp/x10_das_jedi.py`; same hyper-parameters and DAS init seeds).
Results: `results/jedi/x10_verify.json` (main run + calibration), `x10_verify_paired.json` (same run, deterministic, with paired
bootstrap differences), `x10_verify_smoke.json`. Logs: `logs/x10-verify.log`, `logs/x10-verify2.log`, `logs/x10-verify-smoke.log`.
Cluster jobs: x10-verify (1 CPU / 2 GiB, 893 s, peak 2,047 MiB incl. page cache of the memory-mapped jet file), x10-verify2
(921 s, same peak), x10-verify-smoke (602 s).

## Conclusion
On a new set of jets, the learned directions give the same results as in X10. For total pT (M0) they also behave like a stored
variable: an input edit of M0 moves the jet along them almost exactly to the source jet's value. For the angular moment M2, and
for the W-versus-Z score in the first pooled layer, most of the match already follows from a simpler rule, "the edited jet takes
the source jet's class", which holds for 60 % of the M2 decision changes; the M2 directions add at most 0.15 to that rule. For
Δ = ½ log(M0·M2) the directions reproduce about half the decision changes, but the Δ edit moves the jet only part of the way to
the source's value along them, so they show a stored Δ only partly. Still uncertain: in the dense layers every swapped state has
to be clipped back into the range the next layer can represent, and without that clip most of the results disappear.

(Terms: "effective IIA" = fraction of base/source pairs whose input edit changes the network's call, on which the swapped network
gives the same call as the edit; "full swap" = copy the whole layer from the source, which gives the source's own call.)

## Code review of `interp/x10_das_jedi.py`
- Stages: S = `pool('qs1', u)`, h4 = `dense('d4', S)`, h5 = `dense('d5', h4)`, as `jedi_core.Jedi`. The exact tail from each stage
  reproduces `run` and `run_jetwise` (max |Δlogit| 0; reproduced here on T'). No mask or quantizer change.
- Padded slots: `scale`, `translate` edit only pT > 0 slots; `moments` and `mass` sum padded slots with pT = 0 (contribute 0). No
  padded slot is counted as a particle.
- Split: training pairs from F, test pairs from T minus F (`indices` reproduces `sample`'s rng exactly). No leakage found. The ridge
  read-out for the calibration is fitted on F.
- Edits: do(M0) multiplies pT by c (preserves pT ranks) and angles by c^-1/2; through the absolute pT ramps of l1 (a32 on for
  pT < 11 GeV) it also changes which particles are soft, so it is "the network's response to a global pT rescale at fixed M2",
  not a pure M0 change inside the network. do(Δ) = pure pT scaling (stated in X10). do(M2) uses angle scales > 3 for 7.6 % of
  pairs (stated).
- Range clip Π: range = [−k·2^i, 2^i − 2^−f] of the next layer's input quantizer, widened to the base value; matches the wrap in
  `jetdistill/network.py:round_wrap`. At S, 41 of 64 units have an empty range (f = −24 bits; the next layer reads them as 0),
  so X10's "interchanged_clipped 74 %" at S is mostly those inert units. Applied identically to DAS, random nulls and calibration.
- Bootstrap: over pairs; 24 % (X10) / 15 % (here) of test pairs share a base jet. Not corrected here either.
- Figure `results/jedi/x10_das.png` (read): matches the JSON and the report table. It shows only the random-subspace band as a
  reference; the source-call baseline (below), which is 0.32 / 0.60 / 0.32 / 0.50 for M0 / M2 / Δ / W−Z, is not shown and would
  sit inside the range of the DAS dots for M2 and W−Z at S.
- No bug found that changes a reported number.

## Procedure (verify)
- Network: JEDI-linear 3-feature n64, `Jedi(64)`, exact fixed-point tail (X10's `JNet`).
- Jets: F' = `sample(3000, 23)` minus every jet of X10's F (seed 0) and T (seed 17) → 12,729; T' = `sample(3000, 29)` minus X10's F, T
  and F' → 11,972. Accuracy on T' 80.35 %, F' 80.75 %.
- Pairs: 6,000 training pairs from F' (seed 32; last 20 % for model selection); 4,000 random test pairs from T' (seed 31).
  Fraction of test pairs whose counterfactual call differs from the base call: M0 0.202, M2 0.556, Δ 0.575, W−Z 0.251 (X10: 0.186,
  0.563, 0.570, 0.261).
- DAS: X10's `das_fit` (exact tail, straight-through gradient, clip, Adam lr 0.01, 800 steps, init seed 10 + k), stages S / h4 / h5,
  k = 1, 2, 4, V ∈ {M0, M2, Δ, W−Z}. 1,000 random orthonormal k-dim subspaces per stage and k (rng 1000 + k), p_rot, Holm over 36.
- Added controls (the X10 run did not have them):
  (a) Source-call baseline: on the call-changing pairs E_H, the fraction whose counterfactual call equals the source jet's own
      call. This is the effective IIA of the full swap (Q = identity), i.e. of "the base jet takes the source's class".
  (b) Class-transfer DAS: the same `das_fit`, trained with label = the source jet's call (no variable), init seed 40 + k; scored
      on each V's counterfactual calls. Paired bootstrap (1,000 resamples of pairs; X10's `boot_diff`) of V-DAS minus
      class-transfer DAS, and V-DAS minus full swap. H0: difference = 0; H1: V-DAS higher.
  (c) Clip dependence: the trained DAS subspace evaluated without Π (exact tail, values wrap); share of E_H pairs in which Π changes
      at least one live unit (unit with a non-empty quantizer range); median ‖Π(h′) − h′‖ / ‖swapped component‖ over E_H; the same
      for 50 random subspaces.
  (d) Representation check (V with an input edit): with h_e = the stage state of the edited base jet, gap closed =
      1 − Σ‖Qᵀ(h_e − h_s)‖² / Σ‖Qᵀ(h_b − h_s)‖² (1 = the edit moves the jet's subspace coordinates exactly to the source's),
      and the share of the edit's displacement Σ‖Qᵀ(h_e − h_b)‖² / Σ‖h_e − h_b‖² inside span(Q); 200 random subspaces as null.
  (e) Calibration at S and h4: X10's `calibrate` unchanged, on 5,000 jets of T' (rng 4).

## Results

### Reproduction (effective IIA on 4,000 fresh test pairs; X10 value in brackets)
| stage | M0 k=1/2/4 | M2 k=1/2/4 | Δ k=1/2/4 | W−Z k=1/2/4 |
|---|---|---|---|---|
| S | 0.86 / 0.86 / 0.85 (0.87 / 0.86 / 0.85) | 0.65 / 0.71 / 0.51 (0.66 / 0.75 / 0.55) | 0.48 / 0.54 / 0.50 (0.51 / 0.46 / 0.49) | 0.46 / 0.32 / 0.38 (0.43 / 0.35 / 0.36) |
| h4 | 0.68 / 0.85 / 0.84 (0.66 / 0.83 / 0.84) | 0.67 / 0.72 / 0.75 (0.68 / 0.74 / 0.78) | 0.46 / 0.53 / 0.62 (0.48 / 0.58 / 0.62) | 0.62 / 0.65 / 0.72 (0.66 / 0.70 / 0.74) |
| h5 | 0.36 / 0.71 / 0.74 (0.34 / 0.72 / 0.74) | 0.27 / 0.69 / 0.73 (0.26 / 0.68 / 0.73) | 0.20 / 0.44 / 0.50 (0.20 / 0.46 / 0.52) | 0.30 / 0.75 / 0.80 (0.30 / 0.77 / 0.82) |
CI half-widths 0.02–0.03 (e.g. S M0 k=1 0.862 [0.838, 0.887], n_EH 807; h4 Δ k=4 0.616 [0.596, 0.636], n_EH 2,300). Two cells
fall outside X10's CI: S Δ k=2 (0.54 vs 0.46 [0.44, 0.48]) and S M2 k=2 (0.71 vs 0.75 [0.73, 0.76]). Random-subspace means 0.015–0.21.
All 36 cells p_rot = 0.001 (floor); Holm-adjusted 0.036; 36 / 36 below 0.05.
Calibration (T', 5,000 jets, N_W ≈ 1,000): full translation read-out ratio 1.07, W→Z 62 % (scaling 13 %, unedited 3 %).
h4 Δ k=2 patch: share of translation shift 0.70 [0.69, 0.71] (random 97.5 % 0.16), W→Z 85 % (X10: 0.73, 82 %). h4 M2 k=2 patch:
0.69 [0.68, 0.70], W→Z 82 %. S M2 k=1/2: 1.47 / 1.38. M0 subspaces: −0.00 to 0.02.

### (a, b) Source-call baseline and class-transfer subspace (paired differences, 95 % CI, bootstrap p)
Source-call baseline (full swap) on E_H: M0 0.32 (n 807), M2 0.60 (n 2,222), Δ 0.32 (n 2,300), W−Z 0.50 (n 1,005).
| cell | V-DAS | class-transfer DAS on V | V-DAS − class-transfer | V-DAS − full swap |
|---|---|---|---|---|
| S M0 k=1 | 0.86 | 0.16 | +0.70 [+0.67, +0.74] p 0.001 | +0.54 [+0.50, +0.58] p 0.001 |
| h4 M0 k=2 | 0.85 | 0.25 | +0.60 [+0.56, +0.64] p 0.001 | +0.53 [+0.49, +0.57] p 0.001 |
| S M2 k=1 | 0.65 | 0.62 | +0.03 [+0.01, +0.05] p 0.003 | +0.05 [+0.03, +0.07] p 0.001 |
| S M2 k=2 | 0.71 | 0.36 | +0.35 [+0.33, +0.37] p 0.001 | +0.11 [+0.08, +0.13] p 0.001 |
| S M2 k=4 | 0.51 | 0.60 | −0.08 [−0.11, −0.06] | −0.09 [−0.12, −0.06] |
| h4 M2 k=1 | 0.67 | 0.63 | +0.04 [+0.02, +0.05] p 0.001 | +0.07 [+0.04, +0.09] p 0.001 |
| h4 M2 k=4 | 0.75 | 0.61 | +0.14 [+0.12, +0.16] p 0.001 | +0.15 [+0.13, +0.17] p 0.001 |
| h5 M2 k=4 | 0.73 | 0.62 | +0.11 [+0.09, +0.13] p 0.001 | +0.13 [+0.11, +0.15] p 0.001 |
| h4 Δ k=4 | 0.62 | 0.32 | +0.30 [+0.27, +0.33] p 0.001 | +0.29 [+0.27, +0.32] p 0.001 |
| S W−Z k=1 | 0.46 | 0.46 | −0.00 [−0.04, +0.04] p 0.94 | −0.04 [−0.08, +0.01] p 0.13 |
| S W−Z k=2/4 | 0.32 / 0.38 | 0.20 / 0.48 | +0.12 / −0.10 | −0.18 / −0.12 (both p 0.001, below) |
| h4 W−Z k=4 | 0.72 | 0.49 | +0.22 [+0.19, +0.26] p 0.001 | +0.22 [+0.18, +0.26] p 0.001 |
| h5 W−Z k=4 | 0.80 | 0.49 | +0.31 [+0.27, +0.35] p 0.001 | +0.30 [+0.26, +0.34] p 0.001 |
All 36 cells are in `x10_verify_paired.json`. The class-transfer subspace reproduces the source's call on 0.25–0.91 of the pairs
where the source and base calls differ (0.25–0.43 at k = 1; 0.87 / 0.91 at h4 / h5 k=4).

### (c) Clip dependence
| stage | share of E_H pairs where Π changes a live unit (DAS; random) | median clip norm / swap norm (DAS; random) | effective IIA without Π |
|---|---|---|---|
| S | 0.85–0.99; 0.91–0.94 | M0 0.005–0.016; M2, Δ, W−Z 0.11–0.38; random 0.23–0.27 | M0 0.45–0.64; M2 0.21–0.34; Δ 0.22–0.34; W−Z 0.16–0.32 |
| h4 | 1.00; 1.00 | 0.25–0.39; 0.45–0.47 | M0 0.22–0.27; M2 0.13–0.14; Δ 0.42–0.53; W−Z 0.03–0.08 |
| h5 | 1.00; 1.00 | 0.27–0.45; 0.44–0.48 | M0 0.17–0.22; M2 0.11–0.18; Δ 0.37–0.55; W−Z 0.10–0.32 |
At h4 and h5 no call-changing pair is free of clipping, so IIA on unclipped pairs cannot be computed there. At S, M0 k=1 gives
0.88 on the 119 unclipped pairs and 0.86 on the clipped ones.

### (d) Representation check (gap closed by the input edit inside the DAS subspace; random 2.5 / 50 / 97.5 %)
| cell | gap closed | random | edit displacement inside span(Q) | random median |
|---|---|---|---|---|
| S M0 k=1 / h4 M0 k=2 | 0.98 / 0.96 | −0.02 / 0.10 / 0.75; 0.01 / 0.11 / 0.39 | 0.89 / 0.68 | 0.01 / 0.03 |
| S M2 k=1 / h4 M2 k=2 / h5 M2 k=4 | 0.94 / 0.90 / 0.83 | median 0.12 / 0.17 / 0.40 | 0.13 / 0.32 / 0.52 | 0.01 / 0.03 / 0.11 |
| S Δ k=1 / h4 Δ k=2 / h4 Δ k=4 / h5 Δ k=4 | 0.80 / 0.66 / 0.29 / 0.09 | median −0.88 / −0.58 / −0.59 / −0.49 | 0.11 / 0.19 / 0.21 / 0.41 | 0.01 / 0.03 / 0.06 / 0.11 |

## Hypotheses and which are excluded
- H0 "DAS = random subspaces": excluded in 36 / 36 cells on fresh jets (Holm 0.036), as in X10.
- H0' "the V subspace only transfers the source's class" (V-DAS = class-transfer DAS / full swap): excluded for M0 (all stages,
  k ≥ 1 at S/h4, k ≥ 2 at h5), Δ (all cells vs class-transfer), W−Z at h4/h5 (k ≥ 2). For M2 the excess is +0.03 to +0.15 (p ≤ 0.003)
  in the cells X10 reports as best, and negative at S k=4 and h5 k=1. For W−Z at S, H0' is not excluded (k=1: −0.00, p 0.94; full swap higher at
  k = 2, 4).
- "The Δ subspace stores Δ" (DAS premise: the Δ edit moves the subspace coordinate to the source's): at S k=1 gap closed 0.80; at
  h4 k=1 / 2 / 4 0.77 / 0.66 / 0.29 (k=4 is the best Δ cell) and at h5 0.09–0.56. Partly supported at h4 k ≤ 2; not in the best
  Δ cell or at h5 k=1, 4. The M0 subspaces satisfy it (0.90–0.98 at S/h4), M2 mostly
  (0.58–0.95).
- "The subspace swap alone (linear) reproduces the edits": at h4/h5, without Π effective IIA falls to 0.03–0.55; Π changes at
  least one unit in every call-changing pair and its correction is 25–45 % of the swapped component's norm. Only S M0 is close to a
  linear swap (clip norm ≤ 2 % of the swap; 0.64 without Π at k=1).

## Corrected reading
On fresh jets (11,972 test, 12,729 fit), the X10 numbers reproduce within about 0.05 (two cells at S outside X10's CI), and all 36
learned subspaces beat random subspaces (Holm 0.036). The intervention is a k-dim swap followed by the range clip Π; at h4 and h5
the clip acts on every call-changing pair (correction 25–45 % of the swap norm) and the results depend on it (0.03–0.55 without it).
Random subspaces are a weak reference for M2 and the W−Z margin: on 60 % of M2 call changes and 50 % of W−Z call changes the
counterfactual call equals the source jet's own call (full swap). Against that baseline, M0 subspaces exceed it by +0.53 (S k=1,
h4 k=2), Δ by +0.12 to +0.29 (k ≥ 2), M2 by +0.05 to +0.15 (below it at S k=4 and h5 k=1), and the W−Z margin by +0.15 to
+0.30 at h4/h5 (k ≥ 2) and not at all at S (−0.18 to −0.04).
The M0 subspaces satisfy the DAS premise (the M0 edit moves the subspace coordinate 0.90–0.98 of the way to the source's value
at S/h4; 0.68–0.89 at h5); the M2 subspaces mostly (0.58–0.95); the Δ subspaces at h4/h5 only partly (0.09–0.77; 0.29 in the
best Δ cell, h4 k=4). Calibration reproduces
(h4 Δ k=2 carries 0.70 of the translation shift, W→Z 85 %), and the h4 M2 subspace gives the same (0.69, 82 %), so the translation
calibration does not separate the Δ subspace from the M2 subspace. "The W−Z margin aligns at S (0.43)" should read: at S the
W−Z subspace does not exceed the full-swap baseline.

## Issues / limits of this verification
- Untrained-network and PFN controls were not rerun (they were outside the cheap budget); the source-call baseline and class-
  transfer subspace were not run on the untrained networks or the PFN.
- Clip decomposition into the lower (0, ReLU-like) and upper (saturation) bound was not measured.
- The class-transfer DAS reproduces the source's call on only 0.25–0.43 of differing pairs at k = 1, so at k = 1 it is a weak version
  of the full swap; both are reported.
- Bootstrap over pairs; 15 % of test pairs share a base jet.
- Peak memory 2,047 MiB = the 2 GiB request (page cache of the memory-mapped jet file is counted); the jobs were not killed.

## Explainer entry (Paper A tab; site/explain.py)
**Question.** Do the DAS subspaces found in X10 hold up on new jets, and are they more specific than "copy the source jet's class"?

**Background.** In an interchange intervention, part of a layer is copied from a source jet into a base jet. The target is the
network's call on the base jet after an input edit that sets a variable $V$ to the source's value. If the edit often makes the
base jet take the source's own class, a subspace that only carries the class decision would already score well. That sets a
baseline above random subspaces.

**Method.** Same DAS as X10, $h'=\Pi\big(h_b+(h_s-h_b)QQ^\top\big)$, on 4,000 new test pairs. Baselines: (i) full swap, $Q=I$,
whose call is the source's call; its score is $\Pr[y^H_{bs}=\hat y_s\mid y^H_{bs}\neq\hat y_b]$; (ii) a class-transfer subspace
trained with label $\hat y_s$. Representation check: with $h_e$ the state of the edited base jet,
$$\text{gap closed}=1-\frac{\sum\|Q^\top(h_e-h_s)\|^2}{\sum\|Q^\top(h_b-h_s)\|^2},$$
which is 1 when the edit moves the subspace coordinate exactly to the source's value.

**Result.** The X10 numbers reproduce within about 0.05. Full-swap baselines are 0.32 (M0), 0.60 (M2), 0.32 (Δ) and 0.50 (W−Z).
Above the full swap, the margins are: M0 +0.53, M2 +0.05 to +0.15, Δ +0.12 to +0.29 (k ≥ 2), W−Z +0.15 to +0.30 at h4/h5 and
below it at S.
Gap closed: M0 0.90–0.98 at S/h4; M2 0.58–0.95; Δ at h4/h5 0.09–0.77.

**Checks.** The same code, exact tail and seeds as X10 on disjoint jets. 1,000 random subspaces per cell, Holm over 36. Paired
bootstrap for each difference. Without the range clip, the h4/h5 results fall to 0.03–0.55.

**Why it matters.** M0 is stored in a few directions that the input edit moves as DAS requires. For M2 and the W−Z margin at S,
most of the reported alignment is the class decision itself. The Δ directions reproduce call changes, but the Δ edit sets them
to the source's value only partly, so they show a stored Δ only partly.


---

# Second verification (x10-verify3, x10-verify3u): angle-only Δ edit, full-swap baseline for the untrained networks

Code: `interp/x10_verify3.py` (imports X10's `JNet`, `das_fit`, `proj`, `scores`, `boot_eff`, `boot_diff`, `rand_Q`, `do_edit`, `scale`,
`wz_call` unchanged; same DAS hyper-parameters and init seeds 10 + k). Results: `results/jedi/x10_verify3.json`,
`results/jedi/x10_verify3_untrained.json`. Logs: `logs/x10-verify3.log`, `logs/x10-verify3u.log`.
Jobs: x10-verify3 (1 CPU / 2 GiB, 606 s, 0.45 cores mean), x10-verify3u (1 CPU / 2 GiB, 740 s, 0.10 cores mean); pod peak memory
2,047 MiB in both (includes page cache of the memory-mapped jet file; not killed). The job name x10-verify was already taken by the
first verification (completed Job, not touched), so these runs use x10-verify3 / x10-verify3u.

## Conclusion
The directions X10 labels "Δ = ½ log(M0·M2)" reproduce the tagger's response to scaling every particle's pT, but not its
response to a change of the same Δ made by scaling the angles instead. For the angle version, the tagger's decision follows
the directions found for the angular moment M2, and the "Δ" directions do worse than simply giving the edited jet the source jet's
class (0.42 vs 0.61 of decision changes in X10's best Δ cell). So the evidence does not support Δ as one quantity inside the
network: the "Δ" directions describe a response to a pT rescale. A second point concerns the untrained control. Untrained
networks score below the "take the source jet's class" baseline for M0 and M2 in every testable cell, while the trained network
scores above it. That baseline separates trained from untrained networks; comparison with random directions does not.

(Terms: "effective IIA" = fraction of base/source pairs whose input edit changes the network's call, on which the swapped
network gives the same call as the edit. "Full-swap / source-call baseline" = the same fraction for "the base jet takes the source
jet's own call", which is the result of swapping the whole layer.)

## Procedure
- Network: JEDI-linear 3-feature n64, exact fixed-point tail; tail check max |Δlogit| 0 (S and h5 overrides via `run_jetwise`).
- Jets (trained run): F3 = `sample(2500, 41)`, T3 = `sample(2500, 43)`. Removed from both: every jet of X10's F (seed 0) and T
  (seed 17) and of the first verify's F′ (seed 23) and T′ (seed 29). F3's jets are also removed from T3. That leaves F3 9,499 jets
  and T3 8,993 jets. Accuracy: T3 80.83 %, F3 80.39 %. Pairs: 6,000 training pairs from F3 (seed 52; last 20 % for model
  selection) and 4,000 test pairs from T3 (seed 51); 19 % of test pairs share a base jet.
- New counterfactual Da (angle-only Δ edit): all angles × c with c = √(M0_s M2_s / (M0_b M2_b)), the same c as X10's do(Δ). It leaves
  M0 and every pT unchanged and multiplies M2 by c², so ½log(M0 M2) is set to the source's value as in do(Δ). The difference is
  that do(Δ) does this by pT × c, which moves M0, M2, the pT ranks' absolute values and the soft-particle pT ramps of l1.
  Angle scale c: 2.5 / 50 / 97.5 % = 0.23 / 1.00 / 4.77. 15 % of pairs have c > 3 or c < 1/3, against 7.6 % above 3 for X10's
  do(M2).
- DAS for V ∈ {M0, M2, Δ, W−Z, Da} at S, h4, h5 with k = 1, 2, 4, using X10's `das_fit`: exact tail, straight-through gradient,
  clip Π, Adam lr 0.01, 800 steps. 200 random orthonormal subspaces per stage and k (rng 2000 + k).
- Transfer: every subspace is scored on every counterfactual. On the Da call-changing pairs, paired bootstrap (1,000 resamples
  of pairs) of:
  - Δ-DAS minus Da-DAS;
  - Δ-DAS minus the full swap;
  - M2-DAS minus Δ-DAS;
  - Δ-DAS scored on Δ minus the same subspace scored on Da (joint resampling).
- Subspace overlap: ‖Q₁ᵀQ₂‖²_F / k, which is 1 for identical subspaces and k/d on average for random ones.
- Gap closed in a subspace Q by an edit: 1 − Σ‖Qᵀ(h_e − h_s)‖² / Σ‖Qᵀ(h_b − h_s)‖².
- Untrained run, on X10's own T (18,410 jets) and its 5,000 test pairs (pair seed 1, reconstructed exactly: n_eff matches X10 in
  every cell):
  - networks: trained and `Jedi(untrained=True)` seeds 0–2. For the untrained networks, calls are the argmax of the logits
    centred on the F mean, as in X10.
  - quantities: the source-call baseline on E_H with a 1,000-resample CI; the exact full swap (Q = I, with Π) at each stage;
    and X10's stored DAS effective IIA per cell minus that baseline.

## Results

### Reproduction on T3 (effective IIA, DAS / random-subspace mean; 4,000 test pairs)
| stage | k=1: M0 / M2 / Δ / W−Z | k=2 | k=4 |
|---|---|---|---|
| S | 0.85/0.09, 0.67/0.04, 0.50/0.03, 0.49/0.10 | 0.86/0.13, 0.72/0.06, 0.45/0.05, 0.34/0.14 | 0.85/0.17, 0.53/0.10, 0.47/0.07, 0.39/0.21 |
| h4 | 0.62/0.06, 0.67/0.02, 0.45/0.02, 0.62/0.03 | 0.82/0.09, 0.72/0.02, 0.56/0.03, 0.65/0.05 | 0.81/0.12, 0.76/0.04, 0.61/0.04, 0.70/0.09 |
| h5 | 0.38/0.08, 0.26/0.03, 0.20/0.03, 0.30/0.05 | 0.66/0.11, 0.66/0.05, 0.43/0.05, 0.75/0.07 | 0.71/0.17, 0.71/0.08, 0.51/0.08, 0.80/0.11 |

- All 36 cells are within 0.06 of X10's values. The largest differences are S W−Z k=1 (0.49 here vs 0.43 in X10), h5 M0 k=2
  (0.66 vs 0.72) and h4 W−Z k=2 (0.65 vs 0.70).
- Examples with 95 % CI: S M0 k=1 0.85 [0.83, 0.87] (n_EH 794); h4 Δ k=4 0.61 [0.59, 0.63] (n_EH 2,258); h5 W−Z k=4
  0.80 [0.77, 0.82] (n_EH 1,062).
- DAS exceeds all 200 random subspaces in 45 of 45 cells, including Da. With 200 nulls p_rot is floored at 0.005, so Holm over
  45 cannot reach 0.05 in this run. The 1,000-null Holm result is the first verification's (36 of 36, 0.036).
- Source-call baseline on T3: M0 0.34, M2 0.60, Δ 0.34, W−Z 0.51, Da 0.61.

### Angle-only Δ edit: does the Δ subspace carry Δ?
Network level: the calls after do(Δ) and after Da agree on 51 % of all test pairs and on 33 % of Da's call-changing pairs.
| cell | Δ-DAS on Δ [CI] | Δ-DAS on Da | on-Δ minus on-Da | Δ-DAS minus full swap, on Da | Da-DAS on Da [CI] | M2-DAS on Da | overlap Δ–Da / M2–Da / M2–Δ (random) | gap closed in Q_Δ by Δ edit / by Da edit (random median) |
|---|---|---|---|---|---|---|---|---|
| S k=1 | 0.50 [0.48, 0.52] | 0.38 | +0.12 [+0.10, +0.15] p 0.001 | −0.23 [−0.26, −0.20] p 0.001 | 0.66 [0.64, 0.68] | 0.65 | 0.55 / 0.96 / 0.59 (0.016) | 0.80 / 0.92 (−0.66 / 0.14) |
| S k=2 | 0.45 [0.43, 0.47] | 0.27 | +0.18 [+0.16, +0.20] p 0.001 | −0.34 [−0.37, −0.32] p 0.001 | 0.71 [0.69, 0.73] | 0.71 | 0.40 / 0.94 / 0.36 (0.031) | 0.70 / 0.73 (−0.89 / 0.15) |
| S k=4 | 0.47 [0.45, 0.49] | 0.27 | +0.19 [+0.17, +0.21] p 0.001 | −0.33 [−0.36, −0.31] p 0.001 | 0.53 [0.51, 0.55] | 0.52 | 0.45 / 0.70 / 0.42 (0.062) | 0.60 / 0.74 (−1.09 / 0.22) |
| h4 k=1 | 0.45 [0.43, 0.47] | 0.41 | +0.04 [+0.01, +0.07] p 0.005 | −0.19 [−0.22, −0.17] p 0.001 | 0.67 [0.65, 0.69] | 0.66 | 0.60 / 0.95 / 0.58 (0.016) | 0.78 / 0.92 (−0.67 / 0.10) |
| h4 k=2 | 0.56 [0.54, 0.58] | 0.40 | +0.16 [+0.14, +0.19] p 0.001 | −0.21 [−0.24, −0.18] p 0.001 | 0.72 [0.70, 0.74] | 0.70 | 0.40 / 0.95 / 0.40 (0.031) | 0.72 / 0.86 (−0.54 / 0.20) |
| h4 k=4 | 0.61 [0.59, 0.63] | 0.42 | +0.19 [+0.16, +0.22] p 0.001 | −0.19 [−0.22, −0.17] p 0.001 | 0.75 [0.73, 0.77] | 0.73 | 0.27 / 0.68 / 0.23 (0.062) | 0.42 / 0.73 (−0.76 / 0.20) |
| h5 k=1 | 0.20 [0.18, 0.22] | 0.19 | +0.01 [−0.01, +0.03] p 0.24 | −0.42 [−0.45, −0.39] p 0.001 | 0.27 [0.25, 0.28] | 0.26 | 0.53 / 0.99 / 0.51 (0.031) | 0.14 / 0.71 (−0.46 / 0.33) |
| h5 k=2 | 0.43 [0.41, 0.45] | 0.41 | +0.02 [−0.01, +0.04] p 0.20 | −0.19 [−0.22, −0.17] p 0.001 | 0.69 [0.67, 0.71] | 0.67 | 0.53 / 0.97 / 0.52 (0.062) | 0.60 / 0.83 (−0.54 / 0.38) |
| h5 k=4 | 0.51 [0.49, 0.53] | 0.43 | +0.08 [+0.05, +0.11] p 0.001 | −0.18 [−0.20, −0.15] p 0.001 | 0.73 [0.71, 0.75] | 0.72 | 0.38 / 0.93 / 0.32 (0.125) | 0.23 / 0.75 (−0.54 / 0.40) |

On the Da pairs, M2-DAS minus Δ-DAS is +0.07 to +0.44 in every cell, with p = 0.001 in all of them.

### Untrained control against the full-swap baseline (X10's T and test pairs; X10's stored DAS values)
| network | V (n_EH) | source-call baseline [95 % CI] | X10 DAS, 9 cells (S/h4/h5 × k = 1, 2, 4) | cells above the baseline |
|---|---|---|---|---|
| trained | M0 (929) | 0.35 [0.32, 0.38] | 0.34–0.87 | 8 / 9 (h5 k=1 −0.01) |
| trained | M2 (2,816) | 0.61 [0.59, 0.63] | 0.26–0.78 | 7 / 9 (S k=4 −0.06, h5 k=1 −0.36); excess +0.05 to +0.17 |
| trained | Δ (2,852) | 0.33 [0.32, 0.35] | 0.20–0.62 | 8 / 9 (h5 k=1 −0.13) |
| trained | W−Z (1,303) | 0.51 [0.48, 0.54] | 0.30–0.82 | 6 / 9 (all three S cells and h5 k=1 below) |
| untrained 0 | M0 (556) | 0.45 [0.41, 0.49] | 0.02–0.37 | 0 / 9 |
| untrained 0 | M2 (2,254) | 0.39 [0.37, 0.42] | 0.05–0.24 | 0 / 9 |
| untrained 0 | Δ (88) | 0.20 [0.12, 0.29] | 0.05–0.32 | 3 / 9, within the baseline CI or at most +0.12 on 88 pairs |
| untrained 0 | W−Z (2,262) | 0.84 [0.82, 0.85] | 0.10–0.52 | 0 / 9 |
| untrained 2 | M0 (287) | 0.44 [0.38, 0.50] | 0.18–0.37 | 0 / 9 |
| untrained 2 | M2 (1,893) | 0.49 [0.47, 0.51] | 0.12–0.34 | 0 / 9 |
| untrained 2 | W−Z (689) | 0.35 [0.32, 0.39] | 0.13–0.69 | 4 / 9 (h5 k = 1, 2, 4 at 0.69; S k=4 0.38) |
Untrained seed 1 has fewer than 50 call-changing pairs for every V; Δ for seed 2 has 0 such pairs. The exact full swap (Q = I with
Π) gives the source's call on 98.9–100 % of pairs at every stage, so the source-call baseline is the full-swap score.
Untrained seed 0 at S scores 0.48 / 0.52 on W−Z (k = 1 / 2), against 0.43 / 0.35 for the trained network. Untrained seed 2 at h5
k=1 scores 0.69 on W−Z, against 0.30 for the trained network.

## Hypotheses and which are excluded
- H1 "Δ is stored as one variable in span(Q_Δ)". It predicts that Q_Δ reproduces Da as well as do(Δ). Da sets Δ with M0 held
  at the base value, which is exactly what a pure Δ swap does, so under H1 Q_Δ should do at least as well on Da.
  - Observed: Q_Δ scores lower on Da than on do(Δ) by +0.12 to +0.19 at S, +0.04 to +0.19 at h4 and +0.01 to +0.08 at h5.
  - Observed: on Da, Q_Δ is below the full-swap baseline by 0.18–0.42 in all 9 cells (p = 0.001).
  - H1 is excluded at S and h4. At h5, Q_Δ is below the full swap on Da in every cell.
- H2 "Q_Δ is a pT-rescale response (M0 and M2 moved together, plus pT-threshold effects)". It predicts that Q_Δ reproduces
  do(Δ) better than Da, that Da is carried by the M2 subspace, and that Q_Δ overlaps Q_M2 only partly.
  - Observed: Da-DAS ≈ M2-DAS (0.53–0.75 vs 0.52–0.73 on Da; overlap 0.68–0.99).
  - Observed: overlap of Q_Δ with Q_M2 is 0.23–0.59 and with Q_M0 0.01–0.46.
  - H2 is not excluded.
- The Da edit closes 0.71–0.92 of the gap in Q_Δ, more than do(Δ) itself closes (0.14–0.80). The swapped coordinates move
  with M2. Setting them reproduces Da's calls less often than giving the base jet the source's class does.
- H0 "a subspace beats random subspaces only in a trained network". Excluded by X10's own untrained results. Untrained seed 2
  also exceeds the full-swap baseline for W−Z at h5, by +0.34.
- H0′ "trained = untrained relative to the full-swap baseline", for M0 and M2. Excluded:
  - trained: above the baseline in 15 / 18 cells;
  - untrained (seeds 0 and 2): above it in 0 / 36 cells.
  For W−Z, H0′ is not excluded at S or at h5 k=1.

## Corrected reading (X10 reading, with this and the first verification)
- On three disjoint jet samples, learned subspaces of 1–4 dimensions at S, h4 and h5 reproduce the call changes of the M0 and M2
  input edits on 0.66–0.87 of call-changing pairs (best k ≤ 2), against 0.02–0.14 for random subspaces. The swap is followed by
  the range clip Π, and at h4/h5 the results depend on Π.
- Against the full-swap baseline (the base jet takes the source jet's class):
  - M0 subspaces exceed it by about +0.5 at S and h4.
  - M2 subspaces exceed it by +0.05 to +0.17.
  - The W−Z subspaces exceed it by +0.15 to +0.30 at h4 and h5 (k ≥ 2), and do not exceed it at S.
- The "Δ" subspaces reproduce the response to a global pT rescale (0.45–0.61 at S/h4). They do not reproduce the response to an
  angle-only edit that sets the same Δ: 0.27–0.43, below the 0.61 full-swap baseline. That response is reproduced by the M2
  subspace (0.65–0.73) and by a subspace trained on it (0.66–0.75), which overlaps the M2 subspace (0.68–0.99). The data do not
  show Δ = ½log(M0 M2) as one variable at any stage. "Δ forms most clearly at h4" should read: the response to a pT rescale is
  reproduced best at h4 k=4 (0.61–0.62).
- Untrained networks beat their own random subspaces in many cells. For M0 and M2 they stay below the full-swap baseline in all
  36 testable cells, while the trained network is above it in 15 of 18. For W−Z an untrained network exceeds the trained one at S
  (seed 0: 0.48–0.52 vs 0.35–0.43) and at h5 k=1 (seed 2: 0.69 vs 0.30). "At lower absolute values" therefore does not hold for
  W−Z.

## Issues / limits of this second verification
- 200 random subspaces per cell (p floor 0.005). The 1,000-null Holm test is the first verification's.
- The Da edit is off-manifold for 15 % of pairs (angle scale > 3 or < 1/3). It was not repeated with those pairs excluded or
  with an on-manifold arm.
- do(Δ) and Da differ in M0. A model in which the call depends on Δ and M0 jointly predicts that a pure Δ swap matches Da, which
  holds M0 at the base value, and not do(Δ). The observed ordering is the opposite. A three-variable alternative (Δ, M0 and the
  soft-particle count moved by the pT ramps) was not tested.
- The untrained comparison uses X10's stored DAS values against a baseline computed here on the same pairs (unpaired
  differences, no bootstrap of the difference). The PFN was not rechecked.
- Bootstrap over pairs; 19 % of test pairs share a base jet.

## Explainer entry (Paper A tab; site/explain.py)
**Question.** Do the directions X10 found for $\Delta=\tfrac12\log(M_0M_2)$ carry $\Delta$ itself, or only the network's response
to the specific edit used to set it (scaling every pT)?

**Background.** DAS learns a $k$-dimensional subspace $Q$ such that swapping it from a source jet into a base jet,
$h'=\Pi\big(h_b+(h_s-h_b)QQ^\top\big)$, gives the call that an input edit setting a variable $V$ to the source's value gives.
For $\Delta$, X10 set the variable by $p_T\to c\,p_T$ with $c=\sqrt{M_{0,s}M_{2,s}/(M_{0,b}M_{2,b})}$, which multiplies both $M_0$
and $M_2$ by $c$. The same change of $\Delta$ can be made with angles: $(\Delta\eta,\Delta\phi)\to c\,(\Delta\eta,\Delta\phi)$
leaves $M_0$ unchanged and multiplies $M_2$ by $c^2$. If the network holds $\Delta$ in $Q_\Delta$, both edits should be
reproduced by swapping $Q_\Delta$.

**Method.** On 4,000 fresh test pairs, train DAS for the pT-scale edit ($\Delta$) and for the angle edit (Da). Score each subspace
on both. Compare with the full-swap baseline $\Pr[\text{edit call}=\hat y_s\mid\text{edit call}\ne\hat y_b]$. Measure the overlap
$\|Q_1^\top Q_2\|_F^2/k$ and the gap closed,
$$1-\frac{\sum\|Q^\top(h_e-h_s)\|^2}{\sum\|Q^\top(h_b-h_s)\|^2}.$$

**Result.** The two edits give the same network call on only 33 % of the angle edit's call changes. $Q_\Delta$ reproduces the
pT-scale edit on 0.45–0.61 of call changes at S/h4, and the angle edit on 0.27–0.42, below the full-swap baseline (0.61). The angle
edit is reproduced by the $M_2$ subspace (0.65–0.73), and a subspace trained on it overlaps $Q_{M_2}$ by 0.68–0.99.
Separately, untrained networks stay below the full-swap baseline for $M_0$ and $M_2$ in all 36 testable cells, while the trained
network is above it in 15 of 18.

**Checks.** Same code and seeds as X10, on jets disjoint from X10 and the first verification. 200 random subspaces per cell, all
below DAS. Paired bootstrap for every difference. The exact full swap gives the source's call on 98.9–100 % of pairs.

**Why it matters.** The "Δ" directions encode how the network reacts to rescaling pT, not a combined mass-like quantity that it
would use whichever way Δ changes. The full-swap baseline separates the trained network from untrained ones; random subspaces
do not.
