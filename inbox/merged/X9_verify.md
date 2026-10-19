# X9 verification (skeptic, 2026-10-19): fresh sample seed 2027, edit block rerun, two added controls

Script `interp/x9_verify.py` (imports Flow, edits, Ridge, bootstrap and McNemar helpers from `interp/x9_efn_pfn.py`). Job
`anrunw-ai-x9-verify`, 1 CPU / 2 GiB, wall 863 s, peak memory 2047 MiB (at the request), 0.19 cores mean. Results
`results/jedi/x9_verify.json`; log `logs/x9-verify.log`.

## Conclusion
On 10,000 new test jets, a 0.06 rad translation again changes most W calls to Z in JEDI-linear, both PFNs and both EFNs, and
the last-layer mass read-out moves like a real mass increase of the predicted size. A new control that translates the jet but
then shrinks its angles so that M2 about the stored axis is unchanged removes almost all of that W to Z change (4-11 % left,
rotation 2-3 %) and the read-out change in every model. This means the translation response in all three architectures runs
through M2 about the stored axis rather than through a generic reaction to an off-centre jet. The rank-14 swap ratio of 5.25
for JEDI-linear depends on X6's choice of units: with JEDI's units picked the same way as the flow models' (top 10 by cost) it
is 2.96, still well above the PFN/EFN 0.84-0.87.

## Code review of x9_efn_pfn.py / x9_train.py / x9_plot.py
- Edits: padding untouched and pT untouched in every edit (checked on the fresh sample: `padding_changed` and `pt_changed`
  false for all 8 edits). Translation: max |Δlog m| 1e-7 (m invariant), mean ½Δlog M2 0.305. Rotation: M2 change ≤ 8e-8,
  |Δlog m| ≤ 0.010 (small-angle invariance only).
- PFN numpy mask (`any(input != 0)`) matches energyflow's zero-row masking; the TF check (|Δp| < 1e-5) passes. EFN z uses
  real slots only. No split leakage: training uses the train archive minus `splits.npz['dev']`; T and Fs are drawn from the val
  archive (`full_test.npz`), asserted equal to the val archive in x9_train.
- Class indices W = 2, Z = 3 match `jetdistill.config.CLASSES = ['g', 'q', 'W', 'Z', 't']`.
- No bug found that changes the reported numbers. Points that affect interpretation, not arithmetic:
  1. Rank swap unit selection is not matched: JEDI uses X6's units [0, 43, 51, 52, 63] (chosen in X6 as the units at the
     rank-14 edge), the flow models their 10 costliest latent units.
  2. The calibration edit is matched to the mean ½Δlog M2 over all classes (k = 1.357), not per W jet; on W jets the uniform
     scaling sends 41-55 % of calls to g/q and only 6-23 % to Z, while translation sends 39-81 % to Z. The read-out ratio is
     matched on the mean; the call changes of the two edits differ.
  3. The WTA calibrated ratio uses its own calibration (k = exp(0.120)); in this verification all ratios use the translation
     calibration (k = exp(0.305)), so WTA ratios here are not on the same scale as X9's (rescaled by 0.305/0.121 they agree:
     JEDI 0.90 vs 0.88, PFN 0.85 / 0.74 vs 0.89 / 0.64, EFN 0.33 / −0.36 vs 0.35 / −0.37).
  4. JEDI-linear's pure first-layer cuts come from HGQ zero-bit weights; the PFN/EFN are float networks with no sparsity
     pressure, so step 1 compares quantised-pruned vs float, not only architecture.
  5. The absence of rank windows in PFN/EFN holds by construction (both are permutation invariant); `3-feature-perminv`
     JEDI-linear (supported by jedi_core) is the architecture-matched comparison and was not run.
- x9_plot.py: panel (b) does not state that JEDI's bar uses X6-selected units and the flow bars the top-10 cost units.

## Procedure
T' = sample(per_class=2000, seed=2027), N = 10,000 (401 jets shared with X9's T). Read-out: ridge of log m on h (flow) / z
(JEDI), fitted on sample(6000, 0) minus T' (28,800; fit 19,200, dev 9,600). Edits as X9 (matched rotation seed 66, random
direction seed 67, +Δη, +Δφ, WTA axis), plus:
- M2-held translation: random-direction translation, then per-jet angle scaling s = √(M2/M2′) about the stored axis. M0, M2
  unchanged (|Δlog M2| ≤ 7e-8); centroid offset median 0.048 rad; true Δlog m = −0.305 on average.
- scale up / scale down: angles × exp(±0.305) (real mass change ±0.305; the "up" one is X9's calibration).
- Rank swap 13|14 vs 11|12 and 0|1 vs 3|4 with JEDI's units chosen as the top-10 S units by resample cost on T' ([29, 1, 0,
  50, 30, 19, 43, 13, 60, 56]) next to X6's units.
Statistics: Wilson 95 % CIs for rates; 1000 jet bootstraps for read-out means and ratios; exact McNemar for paired calls.

Hypotheses for the M2-held translation (W jets): H_M2 (response carried by M2 about the stored axis): W→Z and calls changed
near rotation, read-out ratio ≈ 0. H_m (invariant-mass reader): ratio ≈ −1, W calls to g/q as under scale down (JEDI 99 %).
H_off (generic off-centre response): W→Z and calls changed near the plain translation.

## Results (T', N = 10,000; nW = 2,000)
Accuracy: JEDI 80.31 % [79.52, 81.08]; PFN s0 / s1 80.72 / 80.87 %; EFN s0 / s1 74.03 / 74.00 %.

W→Z % [95 %] ; calls changed % ; calibrated read-out ratio at h / z [95 %]
| edit | JEDI | PFN s0 | PFN s1 | EFN s0 | EFN s1 |
|---|---|---|---|---|---|
| rotation | 2.5 [1.9, 3.2]; 4.2; −0.00 | 2.9; 5.0; 0.00 | 1.9; 5.8; 0.00 | 1.8; 6.8; −0.00 | 2.9; 9.3; −0.00 |
| translation random | 62.5 [60.4, 64.6]; 59.0; 1.08 [1.05, 1.10] | 58.4; 65.0; 1.00 | 73.1; 61.9; 1.14 | 57.2; 75.8; 1.16 | 69.9; 75.1; 0.98 |
| +Δη | 76.6; 55.9; 0.98 | 53.4; 65.2; 1.01 | 73.0; 61.9; 1.17 | 39.3; 72.5; 1.11 | 73.1; 75.2; 1.05 |
| +Δφ | 76.7; 56.8; 0.99 | 80.8; 60.7; 0.96 | 50.4; 68.3; 1.27 | 76.6; 73.0; 1.17 | 72.4; 74.2; 1.04 |
| M2-held translation | 4.0 [3.2, 4.9]; 16.9; 0.00 [−0.01, 0.01] | 11.1 [9.7, 12.5]; 18.7; −0.18 [−0.19, −0.17] | 6.9 [5.8, 8.0]; 21.3; −0.13 [−0.13, −0.12] | 9.2 [8.0, 10.5]; 25.5; 0.01 [−0.00, 0.02] | 5.0 [4.1, 6.0]; 30.7; 0.02 [0.01, 0.03] |
| scale down (real m −0.305) | 0.2; 53.7; −1.04 | 0.3; 52.8; −1.35 | 0.3; 52.9; −1.10 | 0.2; 50.1; −1.04 | 0.3; 49.0; −0.97 |

W→g/q under M2-held translation: 8.5 / 9.0 / 9.2 / 6.4 / 3.8 % (rotation 9.3 / 9.3 / 10.4 / 11.1 / 11.0 %; scale down 99.3 /
98.1 / 98.8 / 34.8 / 32.4 %).

Tests (McNemar, exact): translation vs rotation, W→Z: p < 1e-185 for all 5 networks and 3 directions. M2-held vs rotation,
W→Z: JEDI p = 0.0018, PFN 9e-31 / 2e-18, EFN 5e-36 / 1e-5. M2-held vs random translation, calls changed: p < 1e-300 for all 5.
Holm over these 25 tests: all adjusted p ≤ 0.0018. M2-held W→Z model − JEDI (paired bootstrap): PFN +7.1 [5.7, 8.5] /
+2.9 [1.6, 4.2] points, EFN +5.2 [3.9, 6.6] / +1.0 [−0.1, 2.0].

Predictions vs observed (M2-held translation): H_off predicted W→Z near 58-77 %: observed 4-11 % (excluded, p < 1e-300 on
calls). H_m predicted ratio ≈ −1 and W→g/q ≈ 99 % (JEDI, PFN): observed ratio 0.00 / −0.18 / −0.13 and W→g/q at rotation level
(excluded). H_M2 predicted ratio 0 and rotation-level calls: ratio 0.00-0.02 for JEDI and EFN, −0.13 to −0.18 for PFN (an
invariant-mass-like component of 13-18 % in the PFN read-out); W→Z exceeds rotation by 1.5-9 points and calls changed exceed
rotation by 13-21 points (residual response to the displaced jet that does not go through M2).

Re-run of X9 numbers on T': translation W→Z (all within 1-2 points of X9; PFN +Δη/+Δφ asymmetry again opposite between seeds:
s0 53 vs 81 %, s1 73 vs 50 %); calibrated ratios 0.96-1.27 (X9 0.97-1.30); WTA W→Z JEDI 58.5 %, PFN 58.7 / 58.4 %, EFN 70.7 /
71.1 % (X9 56.5, 58.7 / 58.5, 70.8 / 70.7 %). EFN WTA: W→Z 71 % while the h read-out ratio is 0.33 / −0.36 of the predicted
mass change, so the EFN's W→Z under the axis change is not tracked by its mass read-out.

Rank swap (ratio across / inside, mean |ΔL|/sd):
| | 13|14 vs 11|12 (N 9,949) | 0|1 vs 3|4 (N 10,000) |
|---|---|---|
| JEDI, X6 units | 5.24 [4.95, 5.53] | 0.13 [0.03, 0.25] (unit 1) |
| JEDI, top-10 cost S units (matched selection, new) | 2.96 [2.88, 3.05] | 1.93 [1.81, 2.07] |
| PFN s0 / s1 | 0.85 [0.81, 0.89] / 0.87 [0.84, 0.91] | 2.20 / 1.97 |
| EFN s0 / s1 | 0.85 [0.79, 0.91] / 0.84 [0.78, 0.89] | 3.50 / 3.68 |
| pT-share gap ratio | 0.77 | 7.55 |

## Differences from the X9 reading
- Reproduced on fresh jets: the translation W→Z and calibrated read-out numbers in all five networks; rotation W→Z ≤ 3 %; the
  seed-dependent PFN direction asymmetry; the WTA rows; the flow models' rank-swap ratio near the gap ratio.
- Added: the M2-held control places the translation response on M2 about the stored axis in all three architectures (W→Z
  drops from 39-81 % to 4-11 %); the PFN read-out keeps a −0.13 to −0.18 invariant-mass-like component; 13-21 points of call
  changes remain that do not go through M2.
- Changed: JEDI's rank-14 ratio is 2.96 [2.88, 3.05] with unit selection matched to the flow models (5.24 with X6's units);
  at the 0|1 edge, matched selection gives JEDI 1.93, inside the PFN range (1.97-2.20).
- Scope: "a property of the input representation and the task" is not tested (no model with recentred inputs or translation
  augmentation); measured: the response is not specific to JEDI-linear among three set architectures trained on stored-axis
  inputs. The cut-type and rank-window differences confound architecture with HGQ quantisation and permutation invariance.

## Explainer entry (Paper A tab)
**Question.** When a set network changes W calls to Z under a jet translation, is that response carried by the un-recentred
moment $M_2=\sum_i p_{T,i}(\Delta\eta_i^2+\Delta\phi_i^2)$, or by some other reaction to an off-centre jet?

**Background.** A rigid translation $(\Delta\eta_i,\Delta\phi_i)\to(\Delta\eta_i+\delta_\eta,\Delta\phi_i+\delta_\phi)$ leaves the
invariant mass $m$ unchanged and raises $M_2$ by $2\,\vec\delta\cdot\vec M_1+\delta^2 M_0$. X9 found that JEDI-linear, a PFN and
an EFN all change W calls to Z under $\delta=0.06$ rad.

**Method.** On 10,000 new test jets (seed 2027), translate by 0.06 rad in a random direction and then scale every angle by
$s=\sqrt{M_2/M_2'}$, so $M_0$ and $M_2$ are exactly unchanged while the jet stays displaced and $\log m$ falls by $\log s\approx-0.305$.
Compare with the plain translation, the matched rotation, and a real angular scaling by $e^{\pm0.305}$.

**Result.** W→Z: plain translation 39-81 %, $M_2$-held translation 4-11 %, rotation 2-3 % (all five networks). Mass read-out
ratio under the $M_2$-held edit: 0.00 (JEDI-linear), −0.18 / −0.13 (PFN), 0.01 / 0.02 (EFN); a reader of $m$ would give −1.
Rank-14 swap ratio with matched unit selection: JEDI-linear 2.96, PFN/EFN 0.84-0.87.

**Checks.** Edits leave padding and $p_T$ untouched; $M_2$ held to $7\times10^{-8}$; McNemar tests with Holm over 25 tests.

**Why it matters.** It ties the shared translation response to $M_2$ about the stored axis in all three architectures, and
shows that part of the size of JEDI-linear's rank-window response depends on which units are measured.
