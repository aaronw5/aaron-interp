# H36 verify: axis convention on Top Landscape and quark–gluon taggers (claims D4, D4m)

## Conclusion

On two fresh samples, recomputing each tagger's inputs about the winner-take-all (WTA) axis or the hardest-particle axis instead of the momentum-sum axis again changes 3–25 % of the calls of Top and quark–gluon taggers. LorentzNet changes none, but that holds by construction: its inputs are four-vectors, which do not depend on the axis, so it gives exactly the same input and is not a measured control. Moving the inputs toward the hardest particle has a direction-specific effect beyond a random shift of the same size, from −1.7 to +4.0 percentage points (pp) in trained networks. Untrained networks with permuted weights show effects of either sign up to about ±3 pp, so only the top-tagging EFN (+2.6 pp WTA, +4.0 pp hardest particle) lies near or above the edge of that range. With only 6–8 null networks, even that excess cannot be separated from the null (empirical p = 0.14). What is still uncertain is whether the EFN effect is learned, because permuted EFNs give a constant output and no architecture-matched null exists.

## Verdict on the H36 reading

- **Numbers:** reproduced on two fresh samples (tables below). The share of calls changed and the model ranking are within about 1 pp of H36 for every network.
- **Corrections:**
  1. **LorentzNet = 0 is by construction.** `TopModels.score` passes the unedited `P4` to `LNet.score`, and the QG `lnet_inputs` reads raw $(p_T, y, \phi)$, so neither path takes the axis condition. The zero tests code determinism, not invariance. The same holds for LR(N).
  2. **"No axis-specific excess beyond the permuted-network null, except Top EFN HP"** is too narrow on one side and too strong on the other.
     - Top EFN WTA is also in excess of a direction-averaged random shift: +2.56 [+1.89, +3.21] pp in verify 2, +2.48 pp in verify 1 and +1.93 pp in H36.
     - H36 used one permuted ParT as the null. An ensemble of 6–8 permuted networks shows direction effects of both signs, with a spread (SD) of 1.5–2.0 pp and a range of −3.0 to +3.2 pp.
     - Against that ensemble, no trained cell, Top EFN included, is significant (empirical two-sided p ≥ 0.11).
  3. **The phrase "beyond a translation"** is empty as written. Recomputing the inputs about a new axis is exactly a translation (H36's own identity), so what can be tested is only whether the direction of the axis offset matters. The corrected reading states this.
  4. **The Top ParT deficit** (axis < random shift: −1.71 [−2.11, −1.31] pp in verify 2, −1.22 in verify 1) reproduces. Permuted ParTs give −2.65 to +1.06 pp, though. The deficit therefore does not need a learned mechanism, and H36's "why ParT changes fewer calls is untested" should say so.
- **o/p (observed / predicted):** not rerun, because the calibration step was not repeated. H36's own R / pred numbers already show that the prediction method is biased by a factor of 0.57–1.22, and the worker reports this.

## Code check of interp/h36_axis_top_qg.py

- **Axis and sign convention:**
  - The `ax` path sets $\Delta\eta = (\eta_i-\eta_a)\,\mathrm{sign}(\eta_J)$ and $\Delta\phi = \mathrm{wrap}(\phi_i-\phi_a)$.
  - TN shifts by $(-\mathrm{sign}(\eta_J)\,\delta\eta,\ -\delta\phi)$, which is algebraically the same translation.
  - $\eta_J$ is the stored E-scheme axis, with the same definition as `h4.jet_inputs` and `h5.kinematics`. In QG, rapidity is used throughout, consistently, as in `convert_qg_datasets`.
- **ΔR is not stale:**
  - Top: `weaver_inputs` recomputes $\Delta R$ from `J['deta']`, `J['dphi']`.
  - QG: `h5.weaver_inputs` and `ef_inputs` call the patched `kinematics`, which is a module attribute looked up at call time.
  - The axis change therefore reaches every axis-dependent input. Pair features use four-vectors and are unaffected, as stated.
- **Padding:** padded slots are masked in both paths (`np.where(m, …, 0)`, `* real`). In EF inputs, $p_T = 0$ zeroes them.
- **WTA:** C/A with WTA recombination is implemented correctly. The merged object keeps the harder branch's direction, so distances need no update. Verify 1 checked it against an independent brute-force C/A + WTA on 300 jets per sample: 0 mismatches.
- **E-scheme axis recompute:** the unedited score is recovered to ≤ 3e-6 (verify 1).
- **Rows:** random choice per class (`_sample`, `load_sample`). The calibration/main split is a random permutation. There is no first-k selection.
- **Working point:** calls are sign(log-odds), i.e. the argmax, used consistently across conditions. This is not a published working point, and the report does not claim one.
- **Top LR($m_{SD}$, $\tau_{32}$) = 0 by construction:** correct. C/A soft drop and exclusive-$k_T$ $\tau_N$ axes do not use the jet axis.
- No coding bug found.
- **Figure** (`h36_axis.png`): the markers match the JSON. Open markers are R. It shows no null band, and a reader cannot see that LorentzNet's zero holds by construction.

## Procedure (verify)

Two independent fresh-sample reruns, both GPU cluster Jobs:

### Verify 1: job `d-h36-verify` (script `interp/h36_verify.py`)
- **Samples:** Top seed 3611, QG Pythia 3612, Herwig 3613; 3,000 jets per class. 134, 274 and 302 jets overlap the H36 main sample.
- **Conditions:**
  - E (unedited), WTA, HP;
  - R1 and R2: two independent random directions at the same per-jet offset size. R1 − R2 is the floor of the contrast itself.
  - ANTI: the axis translation reversed;
  - ROT: the axis translation rotated by 90°.
- **Nulls:** permuted ParT (seeds 0 and 1), PN, PFN and EFN.
- **Statistics:** McNemar test with Holm correction over 76 cells per contrast; bootstrap CIs with B = 2000.
- **Output:** `results/physics/h36_verify.json`.

### Verify 2: job `d-h36-verify2` (script `interp/h36_verify2.py`, 21 min, exit 0)
- **Samples:** Top seed 3621, Pythia 3622, Herwig 3623; 2,600 jets per class. Jets that were in the H36 main sample are dropped, leaving 5,089, 4,948 and 4,933 jets.
- **Conditions:** E, WTA, HP, R (one random direction), M (mirror, the same as ANTI) and Rbar.
  - Rbar: 6 equally spaced directions with a random per-jet phase at the jet's own offset size, giving the per-jet direction-averaged flip probability.
- **Null ensemble:** Top ParT permuted with seeds 0–5. QG: ParT_kin seeds 0–5, PN_kin seeds 0–2 and PFN seeds 0–2.
  - A null is called degenerate if fewer than 0.5 % of its calls change under Rbar.
  - The non-degenerate nulls form the band: 6 on Top, 8 on each QG sample.
- **Statistics:**
  - axis − Rbar: the per-jet paired difference, with a bootstrap CI (B = 2000) and a normal p.
  - Trained − permuted: the paired per-jet difference of (axis − Rbar).
  - Holm correction over 96 cells (axis vs Rbar, R, M) and over 32 cells (trained vs permuted).
  - Position of each trained network in the null band: z-score and empirical two-sided p.
- **Output:** `results/physics/h36_verify2.json`. Per-jet arrays are in `h36_verify2_perjet.npz` on the volume.

### Hypotheses
- **H0:** the direction-specific part of the axis response (axis − Rbar) of a trained network is drawn from the same distribution as that of untrained, weight-permuted networks on the same jets.
- **H1:** a trained network responds to the direction of the axis offset (toward the hardest particle) beyond that distribution.

## Reproduction: % calls changed (WTA) [95 % CI]

| model | H36 (10k) | verify 1 (6k) | verify 2 (~5k) |
|---|---|---|---|
| Top ParT | 3.86 [3.49, 4.23] | 3.62 [3.15, 4.08] | 3.34 [2.85, 3.85] |
| Top PN | 9.31 | 9.30 | 9.22 [8.47, 9.98] |
| Top PFN | 8.65 | 9.03 | 9.67 [8.86, 10.45] |
| Top EFN | 14.72 | 15.08 | 15.48 [14.48, 16.43] |
| Top LorentzNet | 0 | 0 (by construction) | not rescored |
| QG Py jets: ParT_kin (Py) | 3.00 | 2.97 | 2.83 [2.38, 3.27] |
| QG Py jets: PFN | 11.35 | 11.57 | 12.05 [11.14, 12.91] |
| QG Py jets: EFN | 22.50 | 21.62 | 23.54 [22.35, 24.72] |
| QG Hw jets: PFN | 14.55 | 14.25 | 14.35 [13.36, 15.37] |
| QG Hw jets: Hw ParT_kin | 4.30 | 4.37 | 4.89 [4.30, 5.51] |

Unedited AUCs, verify 2: Top ParT 0.9853, PN 0.9851, PFN 0.9783, EFN 0.9744; Py ParT_kin 0.907 (Py jets) and 0.814 (Hw jets); Hw ParT_kin 0.834 (Hw jets).
- LR(girth) accuracy before → after with the WTA axis: 68.6 → 71.1 % (Py) and 67.3 → 68.4 % (Hw). This reproduces the H36 gain.

## Direction contrasts, verify 2 (axis − Rbar, pp [95 % CI], Holm p)

| model | WTA | HP | trained − permuted ParT/PN/PFN, WTA | null band z (WTA / HP) |
|---|---|---|---|---|
| Top ParT | −1.71 [−2.11, −1.31] (7e-16) | −0.66 [−1.06, −0.25] (0.1) | −2.06 [−2.94, −1.23] | −0.60 / 0.00 |
| Top PN | +0.48 [−0.05, +1.04] (1) | −0.20 (1) | +0.13 | +0.91 / +0.29 |
| Top PFN | −0.57 [−1.10, −0.01] (1) | +1.19 [+0.56, +1.84] (0.02) | −0.92 | +0.18 / +1.16 |
| Top EFN | +2.56 [+1.89, +3.21] (5e-12) | +3.98 [+3.27, +4.71] (9e-26) | +2.20 [+1.21, +3.17] | +2.33 / +2.92 (emp p 0.14) |
| Py jets: ParT_kin | +0.21 (1) | +0.37 (1) | −1.59 | +0.32 / +0.34 |
| Py jets: PFN | +2.37 [+1.70, +3.05] (4e-10) | +2.43 (2e-10) | +2.89 [+1.73, +4.08] | +1.64 / +1.48 (emp p 0.11 / 0.22) |
| Py jets: EFN | +1.20 [+0.50, +1.87] (0.04) | +1.34 (7e-3) | +1.71 | +0.92 / +0.87 |
| Hw jets: PFN | +1.63 [+0.88, +2.38] (8e-4) | +2.02 (5e-6) | +2.70 | +0.96 / +1.21 |
| Hw jets: Hw ParT_kin | +1.36 [+0.92, +1.80] (1e-7) | +1.37 (5e-7) | −0.52 | +0.82 / +0.89 |
| Py jets: permuted ParT (seed 0, H36's null) | +1.80 [+1.33, +2.27] (2e-11) | +1.88 (4e-11) | – | – |

**Null bands** (axis − Rbar of the non-degenerate permuted networks):

| sample | n | WTA: range (mean, SD), pp | HP: range, pp |
|---|---|---|---|
| Top | 6 | −2.65 to +1.06 (−0.84, 1.46) | −2.97 to +1.31 |
| QG Py jets | 8 | −2.42 to +2.36 (−0.31, 1.64) | −2.56 to +2.82 |
| QG Hw jets | 8 | −2.81 to +3.24 (−0.27, 1.98) | −2.88 to +2.90 |

Permuted EFNs (Top and QG), permuted Top PN and PFN, and some ParT seeds give a constant output with 0 calls changed, so they are degenerate.

**Mirror control M** (axis − M, pp):
- QG PFN and EFN: +3.3 to +5.0.
- Permuted ParT seed 0 on QG: +2.9 to +3.75.
- LR(girth): −2.2 to −2.75.
- Top ParT: −2.75.
- The axis-vs-mirror asymmetry is as large in an untrained ParT as in trained networks. It is also present in a fixed classical function (girth).

**Floor:** R1 − R2 in verify 1 is within ±0.6 pp in every trained cell, and up to ±1.1 pp in permuted nets.

### Competing hypotheses
- **H0 not excluded:** the per-jet CIs exclude 0 for several trained cells (Top ParT, Top EFN, QG PFN), but the between-network spread of untrained functions is of the same size.
- **Top EFN lies furthest out:** z = 2.3 (WTA) and 2.9 (HP) against the ParT-permuted band, with empirical p = 0.14 (smallest attainable with 6 nulls: 0.14).
- **H1 is neither established nor excluded** for Top EFN and QG PFN.

## Issues
- **Script overwrite and restore:** this verifier first wrote its script to `interp/h36_verify.py`, which overwrote the script of an earlier verify attempt that ran job `d-h36-verify` and then stopped at a session limit without writing a report. The earlier script was restored byte-for-byte from that agent's transcript (one Write, no later edits) and re-synced to the cluster. The new script is `interp/h36_verify2.py`.
- **Null bands:** they come from one architecture on Top (ParT) and three on QG. No non-degenerate EFN-architecture null exists, so the Top EFN comparison is cross-architecture.
- **Calibration and o/p:** not repeated.
- **Other networks:** no second training seed exists for any trained network, so trained-network variance is unknown. A trained-seed ensemble would be the matched null for "learned".

## Corrected reading
- **Identity:** for every Top and QG model tested, the axis enters only through $(\Delta\eta,\Delta\phi)$ and $\Delta R$, so recomputing about the WTA or HP axis is an exact rigid translation. LorentzNet, LR(N) and LR($m_{SD}$, $\tau_{32}$) take no axis-dependent input, so their 0 % holds by construction.
- **Size of the effect:** the share of calls changed (Top 3.3–15.5 %, QG 2.8–24 %) reproduces on two fresh samples. It ranks the models as their response to random translations of the same size ranks them.
- **Direction-specific component:** the part that depends on the offset pointing toward the hardest particle (axis − direction-averaged shift) ranges from −1.7 to +4.0 pp in trained networks. It is comparable to the range in weight-permuted networks (−3.0 to +3.2 pp, SD 1.5–2.0 pp), so no trained cell differs from the untrained-network null at p < 0.1.
- **Largest outlier:** Top EFN (+2.6 / +4.0 pp, z 2.3 / 2.9), for which there is no architecture-matched null.
- **Top ParT:** its smaller axis response than a random shift (−1.7 pp) lies within the permuted-ParT range.
