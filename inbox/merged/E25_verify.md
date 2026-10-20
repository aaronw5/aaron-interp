# E25 verification: 16-feature JEDI-linear, l1 inputs, radial objects, axis dependence of the mass response

Script `interp/e25_verify.py` (edits re-implemented from scratch, not imported from `e25_jedi16.py`). Job `e25-vfy` (1 CPU / 2 GiB;
peak 2,047 MiB, 0.17 cores, wall 1,041 s; the name `e25-verify` was taken by E25's own failed job, which was left untouched).
Results `results/jedi/e25_verify.json`; log `logs/e25-vfy.log`.

## Conclusion
The main E25 result reproduces on new jets: the 16-feature network reads particle angles only through the distance ΔR from the stored jet
axis, and moving all particles 0.06 rad off that axis turns about 48 % of true W jets into Z calls, with the whole effect carried by the ΔR
column. A new control shows that the response follows the pT-weighted spread Σ pT ΔR² (M2) rather than any change of ΔR: scrambling the ΔR
values by the same average amount while holding M2 fixed moves only about 2 % of W jets to Z. One side claim does not hold: the first-layer
cuts are not "more mixed than random placement"; that came from a null that included the 47 units with all-zero weights. The untrained
baseline remains unusable (all six extra redraws give one constant call).

## Code review of `e25_jedi16.py`
- Stages, masks: S1 resamples `a` only at real slots; edits touch only real slots (pt > 0); padded slots unchanged (checked in the
  re-implementation: `pad_changed` false for every arm). Labels: class 2 = W, 3 = Z (`jetdistill/config.py`). Loader equality with
  `sample()` checked (max difference 0 on the new sample as well).
- Edit columns (`S4_cols_changed` in `e25_main.json`): T changes pz, e, erel, eta, etarel, deltaR, costheta; T_dRonly only deltaR;
  T_dR all of T except deltaR; Tabs no relative column. As described.
- **Purity null (S2).** `stat()` is evaluated on the 17 live units, but the null permutes each feature column over all 64 units, of which
  47 have all-zero effective weight. Under that null the 30 nonzero entries spread over 64 units, most live units receive 0–1 entries and
  come out "pure". The comparison therefore tests unit-level sparsity, not a preference for mixed cuts.
- **dilM2 for perminv** also scales etarot, phirot, costhetarel and recomputes absolute coordinates; perminv has nonzero weight on
  etarot/costhetarel (a12, a14) and px/py/pz. Its ratio 1.16 mixes the lead-frame dilation into the comparison (see below).
- The worker's summary lists nonzero units per feature as deltaR 10, pt 9, ptrel 5, phirot 1; the JSON also has pz 2, erel 2, px 1
  (the report lists them).
- No split leakage found: the new sample excludes both E25 samples.

## Re-run on a new sample
- **Sample:** rows of `sample(1500, 2507)` minus every E25 main row (`sample(3000, 2501)`) and fresh row (`sample(1500, 2502)`);
  N = 6,864, N_W = 1,377. Bootstrap: 2,000 resamples over jets; RNG seed 2508.
- **Edits (independent implementation):** T (+0.06 in η about the stored axis, all columns recomputed), T_dRonly, T_dR, R (matched
  rotation, median-displacement 0.06), dil_dR (only ΔR × s_T, s_T = √(M2(T)/M2), median 1.246). New:
  - jit_dR: only ΔR changed, r_i → c·r_i·exp(±σ), random sign per particle, c per jet so that M2 is unchanged (max relative change
    8e-16); σ = 1.15 chosen so the pT-weighted mean |Δr| on W jets equals T's (0.0372).
  - con_dR: ΔR ÷ s_T (contraction).
  - dose: T_dRonly and the matched dil_dR at d = 0.03 and 0.09.
- **H0** (call invariant under T beyond the rotation rate) vs **H1** (W→Z(T) > W→Z(R)); McNemar on the Z indicator vs R, Holm over 22 tests.

| model | arm | W→Z [95 % boot] | flow [95 %] | McNemar vs R n10/n01, Holm p |
|---|---|---|---|---|
| 16f (acc 80.9 %) | unedited | 0.013 | | |
| | R | 0.013 [0.007, 0.020] | 0.000 [0.000, 0.000] | |
| | T | 0.476 [0.448, 0.502] | 0.463 [0.436, 0.491] | 13/650, 5e-172 |
| | T_dRonly | 0.479 [0.451, 0.505] | 0.466 [0.438, 0.493] | 13/654, 4e-173 |
| | T_dR | 0.012 [0.006, 0.018] | −0.001 [−0.004, 0.000] | 2/0, 1.0 |
| | dil_dR | 0.449 [0.421, 0.476] | 0.436 [0.408, 0.463] | 17/617, 4e-157 |
| | **jit_dR (M2 held)** | 0.031 [0.023, 0.041] | 0.018 [0.009, 0.027] | 5/30, 2e-4 |
| | con_dR | 0.004 [0.001, 0.007] | −0.009 [−0.016, −0.003] | 18/5, 0.07 |
| | T_dRonly 0.03 / dil_dR 0.03 | 0.535 / 0.544 | 0.522 / 0.531 | |
| | T_dRonly 0.09 / dil_dR 0.09 | 0.044 / 0.024 | 0.031 / 0.011 | |
| 16f-perminv (81.5 %) | R / T / T_dRonly / T_dR | 0.029 / 0.489 / 0.495 / 0.031 | | T 35/668, 1e-151; T_dR 1.0 |
| | dil_dR / jit_dR | 0.493 / 0.038 | 0.466 / 0.011 [0.000, 0.022] | |
| 3f (80.6 %) | R / T | 0.031 / 0.743 | 0.001 / 0.713 | T 27/1007, 4e-257 |

- **Paired comparisons (16f):** T vs T_dRonly 7/11, p = 0.48. jit_dR vs T_dRonly 23/639, p = 2e-157 (perminv 31/661, p = 7e-155).
- **Flow ratio T_dRonly / dil_dR (paired bootstrap):** 16f 1.07 [1.01, 1.13] at d = 0.06, 0.98 [0.95, 1.02] at 0.03, 2.87 [1.44, 12.3]
  at 0.09 (both flows ≤ 0.03). perminv 1.00 [0.95, 1.06] at 0.06 (E25's 1.16 used the full dilM2, which also dilates the lead frame),
  0.94 [0.91, 0.97] at 0.03.
- **Dose:** W→Z is not monotone in the shift: 0.53 (0.03), 0.48 (0.06), 0.04 (0.09). Under T, W→q/g also rises from 0.116 to 0.272;
  the remaining changed W calls at 0.09 go to other classes (calls changed 69 %). The W→Z flow at 0.06 is past its peak, so the flow ratio
  at 0.06 is a weak test of matched magnitude; the 0.03 ratio (0.98) is the cleaner one.
- 16f / 3f flow(T) = 0.463 / 0.713 = 0.65 (E25: 0.63 main, 0.65 fresh).

Competing hypotheses for the axis response (16f): (a) M2-like Σ pT ΔR² read from the ΔR column — predicts T ≈ T_dRonly ≈ dil_dR,
T_dR ≈ 0, jit_dR ≈ 0: observed. (d) any perturbation of ΔR values — predicts jit_dR ≈ T_dRonly: excluded (23/639, p = 2e-157).
(b) half-planes, (c) absolute coordinates: E25's T_rel/Tabs arms; not rerun (zero l1 weights confirmed from the JSON).

## Purity null restricted to active units (missing control)
Each feature's column permuted among the units with any nonzero effective l1 weight, 20,000 draws, two-sided p.

| model | active units / nonzero entries | observed exact purity | null [2.5 %, 50 %, 97.5 %] | p lower / upper / two-sided |
|---|---|---|---|---|
| 16f | 17 / 30 | 0.47 | [0.27, 0.47, 0.63] | 0.59 / 0.43 / 0.85 |
| 16f-perminv | 32 / 100 | 0.28 | [0.06, 0.16, 0.28] | 0.99 / 0.033 / 0.065 |

With the dead units removed from the null, 16f's 8/17 mixed cuts equal the random-placement expectation, and perminv is at the upper edge
(more pure, not more mixed). E25's statement "more mixed than random placement" (numbers and thesis_note) is not supported.

## Untrained control
16-feature redraws seeds 3–8: every one gives a single constant call (accuracy 19.7–20.1 %), 0 % calls changed under T_dRonly and R.
With E25's seeds 1–2 that is 8 of 9 redraws constant; the untrained arm gives no baseline for this family. The R, T_dR and jit_dR arms
are the operative controls.

## Figure
`e25_edits.png` matches `e25_main.json` (bars and bootstrap intervals; the dotted line is the 16f unedited rate). The title states
"goes through the ΔR column only"; the 3-feature bars for "only ΔR edited" sit at baseline because that model has no ΔR input, which the
legend does not say.

## Corrected reading
In the official 16-feature JEDI-linear the first layer reads angles only through ΔR (10 of 17 live cuts, 8 of them mixed with pT) and one
rotated-frame coordinate, with zero weight on Δη, Δφ, η, φ. The proportion of mixed cuts equals the expectation from random placement of
the nonzero weights among the active units (p = 0.85). A 0.06 rad translation off the stored axis turns 48–49 % of true W jets into Z
calls (R: 1.3–1.5 %), carried entirely by the ΔR column, and close to an M2-matched ΔR dilation (ratio 0.98 at 0.03 rad, 1.05–1.07 at
0.06 rad); an M2-preserving ΔR scramble of matched size gives 1.8 %. The W→Z rate is not monotone in the shift size (0.53, 0.48, 0.04 at
0.03, 0.06, 0.09 rad). Untrained redraws give constant calls (8/9) and provide no baseline.

## Explainer entry (for site/explain.py)
**Question.** Does the 16-feature JEDI-linear's W→Z response to moving particles off the jet axis track the pT-weighted spread
$M_2=\sum_i p_{T,i}\,\Delta R_i^2$, or any change of the ΔR column?
**Method.** On 6,864 new jets, edit only the ΔR column: (i) as under a 0.06 rad translation, (ii) a dilation $\Delta R_i\to s_T\Delta R_i$ with
the same $M_2$, (iii) a random scramble $\Delta R_i\to c\,\Delta R_i e^{\pm\sigma}$ with $c$ chosen so $M_2$ is unchanged and $\sigma$ so the
mean $|\delta\Delta R|$ equals the translation's.
**Result.** W→Z: 0.479 (translation), 0.449 (dilation), 0.031 (scramble), 0.013 (rotation control).
**Checks.** McNemar scramble vs translation 23/639, $p=2\times10^{-157}$; dose 0.03/0.06/0.09 rad.
**Why it matters.** The axis response follows $M_2$ about the stored axis, not ΔR changes in general.
