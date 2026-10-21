# B8: program fidelity vs description length, scored on edits and interchange interventions (Paper B: B4, B2m, B2o; stage 2)

**Script:** `interp/b8_fidelity_dl_iia.py`. It has three modes, each run as its own cluster job:
- `pool` (jax env): builds the jets and edits and runs the network.
- `efp` (volume `venv-ef`, energyflow 1.4): computes the EFPs. **energyflow is not installed in the job env**, so the EFPs were computed in this separate env.
- `main` (jax env): fits the baselines, evaluates the stored nested programs, and computes the scores, IIA and tests.

**Jobs** (all exit 0):

| job | request | wall | peak memory | notes |
|---|---|---|---|---|
| b8-pool | 1 CPU / 3 GiB | 311 s | 1.39 GiB | |
| b8-efp | 1 CPU / 2 GiB | 500 s | 0.30 GiB | |
| b8-run | 1 CPU / 3 GiB | 751 s | 2.50 GiB | |

Smoke runs: b8-smk-pool, b8-smk-efp, b8-smk-main and b8-smk-main2. b8-smk-main stopped on a closure bug in the baseline predictor; it was fixed before the full run.

**Files:**
- Results: `results/jedi/b8.json` (every program, score, CI and test), `b8_pool.json` (jets, edits, accuracies, flip rates), and `b8_perjet.npz`, `b8_pool.npz`, `b8_efp.npz` (volume only).
- Figure: `results/jedi/b8.png`.
- Logs: `logs/b8-*.log`.

## Conclusion

We compared several compact programs (sets of explicit formulas) that imitate the jet-tagging network. Each was scored on how well it reproduces the network's changes of call when the jets are edited, and when an internal quantity is swapped between two jets. Given enough bits (≈ 60 kbit or more), the program built in the network's own staged structure follows the network's response to a small rigid shift of the jet much more closely than formulas built on standard jet observables, energy-flow polynomials or boosted trees of the same size (Cohen κ 0.46–0.67 vs ≤ 0.35). Its internal pieces can also be swapped jet-to-jet with the network's matching units with the same outcome in about 0.6–0.7 of the swaps that change the network's call, against about 0.25 when the program pieces are paired with the wrong units. Three questions remain open:
- On adding one soft particle and on swapping the pT of the 13th and 14th particles, the network changes too few calls (1.9 % and 0.4 %) to tell the programs apart.
- The swap test for the program uses the very units it was fitted to.
- For the baselines, our learned one-direction alignment did not beat a random direction, so no baseline passed or failed that test.

## Procedure

**Network.** Official JEDI-linear 3-feature n64, exact fixed point (`jedi_core.Jedi`). Every program is fitted to the network's outputs, not to truth labels. A call is the argmax of the 5 logits (g, q, W, Z, t).

**Jets.**
- Fit sample: `sample(4000, 808)` = 20,000 jets, split by rng 0 into 13,333 fit and 6,667 dev jets. The dev jets are used only to choose GBDT members, baseline variables, and the DAS step and stage.
- Test sample: `sample(2000, 809)` minus every jet that is in the fit sample or in `sample(6000, 0)` (the jets the nested programs were fitted on). This leaves N = 8,234 jets.

**Unedited-network check.** Accuracy is 80.88 % on the test jets (in the 80.6–81.3 % band) and 81.28 % on sample 808. A copy of the network's last layers (S onward) reproduces the network's logits exactly (max |Δ| = 0).

**Description length (DL).** DL is the number of bits needed to write the program down. Each stored number costs its bit width, and each hinge factor adds structure bits: log2(#inputs) + 2 (B6 rule). A hinge is max(0, x − t) or max(0, t − x) for an input x and threshold t.

**Programs and their DL levels.**
- **P: nested program (E11 / B4).** Every live unit (a unit whose mean-ablation changes ≥ 0.1 % of calls, E1) has its own hinge formula:
  - P, g and S units are formulas on rank-window (pT, r) sums and observables;
  - h4 is a formula on S, h5 on h4, and z on h5;
  - z goes through the network's own last layer.
- **P levels** (DL in bits). "10-bit rounding" means every coefficient and knot is rounded to 10 significant bits and stored in 18 bits.

  | level | terms per unit K | number rounding | DL (bits) |
  |---|---|---|---|
  | P30 | 30 | 32-bit | 367,369 |
  | P16 | 16 | 32-bit | 207,611 |
  | P8 | 8 | 32-bit | 101,800 |
  | P8r10 | 8 | 10 significant bits | 62,516 |
  | P4r10 | 4 | 10 significant bits | 30,917 |
  | P2r10 | 2 | 10 significant bits | 15,537 |
  | P2r8d1 | 2 | 8 significant bits | 12,354 |

  In P2r8d1, the 21 live units with mean-ablation cost < 1 % of calls are also replaced by their mean over 3,000 fit jets ("drop objects by cost"). Merging rank windows was not implemented.
- **STD:** MARS (multivariate adaptive regression splines, forward-stepwise hinge regression; `jetdistill.mars`, B6/B7 settings) on 10 standard observables: m, m_SD, τ21, τ32, D2, N, ΣpT, ΣpT r, ΣpT r², ΣpT r³. Here τ21 and τ32 are N-subjettiness ratios, D2 is the energy-correlation two-prong variable, and r is the distance to the jet axis. Sizes K = 2, 4, 8, 16, 30 terms per logit; DL 0.85–13.9 kbit.
- **EFP:** MARS on the 54 connected energy-flow polynomials of degree ≤ 5 plus m, ΣpT and N. EFPs are the complete linear basis of IRC-safe jet observables (hadronic measure, β = 1, normed). They are translation invariant, so they cannot respond to a rigid shift of the jet. Sizes K = 2 to 50; DL 0.9–21.9 kbit.
- **GBDT:** histogram gradient-boosted regressors on STD ∪ EFP inputs quantised to 4 or 8 bits, with 4, 8 or 16 leaves and 3, 10, 30 or 100 iterations (24 members, DL 4.4–368 kbit).
- **Shuffled-label controls:** STD K30, EFP K30 and a GBDT (q 8, 8 leaves, 30 iterations), fitted to the fit jets' logits permuted (rng 8099).

**Edits** (on test jets; padding never moved; every program recomputes its inputs from the edited particles):
- **trans:** every particle's Δη + 0.06 rad. Invariant mass and every EFP are unchanged (this is the "W → Z flow" edit). N = 8,234.
- **soft:** one extra particle in the first empty slot. Its pT is 0.5 × the softest particle's pT (median 0.4 % of ΣpT), at r = 0.4·√U with uniform azimuth. Jets with N < 64 only (N = 6,377).
- **swap:** the pT values in slots 13 and 14 are exchanged (N = 8,203; jets with ≥ 15 particles). The particle set is the same as in B6's A13 angle swap, but the softer of the two now sits in slot 13, the edge of S0's window over the 14 hardest particles.
- **D:** prong-sharing edit dc = +0.3 (E35 / H45). It rotates the decay axis of the two exclusive-kT prongs in their rest frame, holding the jet four-momentum, the pairwise masses and N. It is applied on a random subset C of 3,000 test jets (rng 8091), with inputs rebuilt about the E-scheme axis; the base is the same jets rebuilt the same way. N = 2,982. Control **cn**: the same angle applied about the prong axis, so the energy sharing is held.
- **cf:** E36 minimal call flip on C. This is the smallest per-particle pT and angle change, at fixed jet four-momentum, that makes the float program call the partner class (W ↔ Z, q ↔ g). 1,723 edits pass the constraint check; median ε = 6.1. Control **cfr**: a random edit of the same size under the same constraints.

**Scores.**
- **Natural same call:** % of test jets where the program's call equals the network's.
- **Edit fidelity:**
  - Cohen κ between the program's and the network's call-change indicators. κ = 0 is the "call never changes" rule; κ = 1 is identical changes.
  - R = mean change of the program's margin divided by the mean change of the network's margin. The margin is the logit of the network's base call minus the largest other logit.
- **Interchange-intervention accuracy (IIA)** on 3,000 random (base, source) test pairs (rng 8093):
  - Set one program variable V on the base jet to its value on the source jet and record the program's call. Do the same at the aligned location in the network and record the network's call.
  - IIA = the fraction of pairs where the two calls agree.
  - Effective IIA = the same fraction over only the pairs where the network's call changes.
- **P variables and their aligned network locations** (the identical units):
  - S29: ΣpT r² + Σr² + soft count;
  - S1: Σ over ranks ≥ 1 of (r − 0.2)₊;
  - S0: ΣpT of the 14 hardest particles;
  - S19: soft multiplicity;
  - S30: Σr in the core;
  - all 19 live S units together;
  - the mass variable: h4 units 3, 9, 18 (the M2-following and m-like units of E6; M2 = ΣpT r²);
  - the z mass windows z0 (≈ 75 GeV), z7 and z14 (94–98 GeV), z10 (135–148 GeV), and the four together.
- **Baseline variables and their aligned network locations.**
  - For each baseline, V is each of the two inputs whose swap changes the program's call most often on dev pairs. These were mostly m, ΣpT r², N and single EFPs.
  - The aligned network location is a learned 1-dimensional DAS direction w (distributed alignment search) at S or at h4, whichever matches better on dev pairs. Interchange along w means S′ = S_b + ((S_s − S_b)·w) w.
  - w is trained for 300 Adam steps on 4,000 fit-jet pairs. The loss is the cross-entropy of a float copy of the network tail, which reproduces 99.2 % of the exact calls, against the program's counterfactual call.
  - w is evaluated with the exact network.

**Controls.**
- **Permuted alignment.** For P, V is paired with random other live units of the same stage and count (5 draws). This is not possible for "all 19 S units", because no disjoint set exists. For the baselines, V is paired with a random direction in the live span (5 draws).
- Shuffled-label programs.
- GBDT at matched DL.

**Statistics.**
- 95 % bootstrap over jets (over pairs for IIA), B = 1,000.
- Matched DL: for each P level and baseline family, the member with bits ≤ P's bits and the highest dev same call. No comparison needed the "above P's bits" fallback.
- Each comparison uses a paired bootstrap of P − baseline (same resamples) and an exact McNemar test on paired per-jet indicators. Benjamini–Hochberg (BH) is applied over the 147 tests (7 P levels × 3 families × 7 scores). P's aligned − permuted tests get their own BH over 77 tests.
- H0: the score of P equals that of the matched baseline (respectively, aligned IIA = permuted IIA). H1: they differ.

## Results

**Network call changes** (% of jets, 95 % CI; Δm = mean margin change):

| edit | trans | soft | swap | D | cf | cn (control) | cfr (control) |
|---|---|---|---|---|---|---|---|
| % calls changed | 56.3 [55.2, 57.3] | 1.90 [1.57, 2.24] | 0.37 [0.23, 0.50] | 20.3 [18.9, 21.8] | 49.6 [47.3, 52.0] | 6.4 | 6.3 |
| Δm | −3.58 | −0.09 | −0.003 | −0.97 | −2.78 | −0.13 | −0.45 |

**Selected programs.** Same call and κ are on test jets, with 95 % CI. R is the margin-response ratio for trans. IIA_eff is the mean effective IIA over variables, with the permuted-alignment value after the slash.

| program | DL (kbit) | same call % | κ trans | κ soft | κ swap | κ D | κ cf | R trans | IIA_eff / permuted |
|---|---|---|---|---|---|---|---|---|---|
| P30 | 367 | 94.0 [93.5, 94.5] | 0.669 [0.652, 0.685] | 0.140 [0.081, 0.206] | 0.03 | 0.567 [0.529, 0.603] | 0.218 [0.175, 0.259] | 0.97 | 0.683 [0.662, 0.701] / 0.245 |
| P16 | 208 | 93.1 | 0.625 | 0.069 | 0.00 | 0.552 | 0.173 | 0.90 | 0.655 / 0.249 |
| P8 | 102 | 90.4 | 0.464 | 0.047 | 0.03 | 0.526 | 0.093 | 0.85 | 0.605 / 0.258 |
| P8r10 | 62.5 | 90.5 | 0.462 | 0.060 | 0.03 | 0.532 | 0.088 | 0.85 | 0.607 / 0.256 |
| P4r10 | 30.9 | 87.9 | 0.365 | 0.074 | 0.00 | 0.493 | 0.001 | 0.87 | 0.508 / 0.254 |
| P2r10 | 15.5 | 83.3 | 0.413 | 0.010 | 0.09 | 0.408 | 0.004 | 0.68 | 0.502 / 0.269 |
| P2r8d1 | 12.4 | 68.0 | 0.052 | 0.033 | 0.08 | 0.200 | 0.079 | 0.33 | 0.436 / 0.253 |
| STD30 | 13.9 | 90.4 [89.7, 90.9] | 0.281 | 0.030 | 0.18 | 0.381 | 0.035 | 0.37 | 0.142 / 0.163 |
| EFP50 | 21.9 | 91.0 [90.3, 91.6] | 0.000 | 0.025 | 0.15 | 0.514 | 0.047 | 0.00 | 0.086 / 0.130 |
| GBDT q8 L8 T100 | 196 | 91.2 | 0.035 | 0.014 | 0.11 | 0.421 | 0.042 | 0.14 | 0.115 / 0.082 |
| GBDT q8 L4 T100 | 97.5 | 91.0 | 0.062 | 0.076 | 0.08 | 0.411 | 0.056 | 0.13 | 0.081 / 0.096 |
| STD30 shuffled | 3.4 | 19.5 | −0.001 | 0.000 | 0.00 | 0.006 | 0.028 | 0.01 | 0.455 / 0.365 |
| EFP30 shuffled | 2.4 | 19.7 | 0.000 | −0.002 | 0.00 | −0.002 | 0.002 | 0.00 | 0.497 / 0.373 |
| GBDT shuffled | 68.3 | 20.2 | 0.066 | 0.004 | 0.01 | 0.124 | 0.035 | 0.03 | 0.267 / 0.215 |

Every EFP program has κ trans = 0 and R = 0 by construction: EFPs are translation invariant, so they predict no response. This is a stated baseline property, not a finding.

**Matched DL: P − baseline** (κ or same-call points as a fraction; 95 % CI; BH q):
- **P30**, against STD30, EFP50 and GBDT q8 L8 T100 respectively:

  | score | vs STD30 | vs EFP50 | vs GBDT q8 L8 T100 |
  |---|---|---|---|
  | same call | +0.036 [0.030, 0.042] | +0.030 | +0.028 |
  | κ trans | +0.39 | +0.67 | +0.63 |
  | κ soft | +0.11 (q 0.009) | +0.12 (q 0.005) | +0.13 (q 0.002) |
  | κ swap | −0.15 (q 0.009) | −0.12 (q 0.015) | −0.08 (q 0.15) |
  | κ D | +0.19 | +0.05 (q 0.005) | +0.15 |
  | κ cf | +0.18 | +0.17 | +0.18 |

  All entries without a q value have q ≤ 0.002.
- **P8r10 (62.5 kbit):**
  - Same call is −0.5 to +0.8 points against all three families (within ±1 point).
  - κ trans: +0.18 (STD30), +0.46 (EFP50), +0.40 (GBDT q8 L16 T10); all q 0.002.
  - κ soft: +0.03 to +0.04 (q ≥ 0.26).
  - κ D: +0.15 (STD), +0.02 (EFP, q 0.44), +0.17 (GBDT).
  - κ cf: +0.05 (STD, q 0.04), +0.04 (EFP, q 0.15), +0.04 (GBDT, q 0.21).
- **P4r10 (30.9 kbit):**
  - Same call: −1.3 to −3.1 points (q 0.002).
  - κ trans: +0.08 to +0.37.
  - κ cf: −0.05 to −0.02 (n.s.).
- **P2r10 (15.5 kbit):**
  - Same call: −4.4 to −7.1 points.
  - κ trans: +0.13 to +0.41.
  - κ D: +0.03 (STD, n.s.), −0.03 (EFP, n.s.), +0.12 (GBDT).
- **P2r8d1 (12.4 kbit):**
  - Same call: −19 to −22 points.
  - κ trans: −0.30 against STD16 and −0.08 against GBDT.
- **Effective IIA:** P minus the matched baseline is +0.33 to +0.60 at every level (q 0.002). See Issues 1 and 2: this difference does not measure baseline quality.

**IIA of P30 by variable.** Columns:
- IIA (all) is over all 3,000 pairs; effective IIA is over the n pairs where the network's call changes.
- "No-swap" is the agreement of the unswapped program's call with the network's swapped call on those n pairs. It is the value a program that ignores V would get.

| variable | IIA (all) | effective IIA [95 % CI] | n | permuted alignment | no-swap |
|---|---|---|---|---|---|
| S29 | 0.80 | 0.557 [0.530, 0.585] | 1157 | 0.360 | 0.05 |
| S1 | 0.82 | 0.399 [0.364, 0.435] | 707 | 0.247 | 0.08 |
| S0 | 0.91 | 0.762 [0.729, 0.795] | 631 | 0.249 | 0.08 |
| S19 | 0.91 | 0.727 [0.688, 0.767] | 484 | 0.251 | 0.11 |
| S30 | 0.88 | 0.607 [0.568, 0.646] | 532 | 0.227 | 0.11 |
| all 19 S units | 0.94 | 0.938 [0.929, 0.948] | 2448 | n/a | 0.02 |
| mass variable h4 {3, 9, 18} | 0.90 | 0.620 [0.545, 0.692] | 184 | 0.267 | 0.22 |
| z0 window | 0.91 | 0.759 [0.711, 0.806] | 345 | 0.195 | 0.11 |
| z7 window | 0.92 | 0.715 | 200 | 0.182 | 0.14 |
| z10 window | 0.93 | 0.719 | 235 | 0.224 | 0.12 |
| z14 window | 0.93 | 0.548 [0.394, 0.692] | 42 | 0.298 | 0.19 |
| z windows together | 0.91 | 0.843 [0.819, 0.865] | 970 | 0.196 | 0.06 |

- Aligned − permuted is significant (BH q < 0.05) in 69 of 77 (level × variable) tests.
- The 8 non-significant tests are all at K ≤ 8 or at the pruned level: S1 at P8 and P8r10 (−0.03); S29 at P2r10 (−0.04); S30 and z10 at P2r10; z14 at P4r10; z10 and z14 at P2r8d1.
- At P30 the aligned − permuted difference is +0.20 [0.15, 0.24] for S29 and +0.56 [0.50, 0.62] for z0.

**Baselines with DAS alignment.**
- Effective IIA is 0.03–0.30, and the random-direction controls are 0.05–0.23. For example, STD30 gives 0.142 against 0.163 permuted.
- The learned direction was chosen at h4 in most fits. Its dev match with the program's counterfactual call was 0.62–0.90, but over all pairs, most of which do not change.
- The shuffled-label programs reach 0.46–0.50, which is higher than every real baseline.

## Pre-stated predictions

The B thesis predicted that, at matched DL:
1. P matches the baselines on natural calls within ±2 points;
2. P exceeds them by ≥ 0.2 κ on trans and soft;
3. P's effective IIA is ≥ 0.6 against a permuted value ≤ 0.3.

The alternative was that GBDT matches P on every edit family except where its inputs are invariant by construction, and that P's IIA does not exceed the permuted alignment.

**How each prediction fared.**
- **(1) Natural calls:** met at P8r10 and P8 (−0.6 to +0.8 points). P16 and P30 lie +1.9 to +3.6 points above the baselines. P4r10 is −1.3 to −3.1 and P2r10 is −4.4 to −7.1, so (1) fails below ≈ 30 kbit.
- **(2) trans:** met from 15.5 kbit up against EFP and GBDT, and from 62.5 kbit up against STD30 (+0.18 [0.16, 0.21] at P8r10, +0.39 at P30).
- **(2) soft:** not met. The differences are +0.02 to +0.13; only P30's differences pass BH. The network changes 1.9 % of calls, so κ is noisy (CI widths ≈ 0.12).
- **(3) IIA:** met at P30, P16, P8 and P8r10 (0.605–0.683 against 0.245–0.258). Not met at P4r10, P2r10 and P2r8d1 (0.44–0.51).
- **Alternative:** excluded where it makes a testable prediction.
  - GBDT's inputs (ΣpT rⁿ, m_SD, τ) are not translation invariant, yet the matched GBDT has κ trans 0.03–0.30 against P's 0.41–0.67, and κ D 0.13–0.30 below P at ≥ 30 kbit (q 0.002).
  - P's IIA exceeds the permuted alignment in 69 of 77 tests.
  - For EFP the alternative and B make the same prediction on trans (0 by construction). The informative EFP comparisons are D, where EFP50 is within 0.02–0.05 of P at ≥ 62 kbit, and cf, where EFP50 is 0.04–0.17 below P.

## Issues and limits

1. **P's IIA alignment holds by construction.** P's S, h4 and z formulas were fitted to exactly the network units used as the aligned location. High IIA therefore measures how well the unit-level formulas reproduce the network's downstream response to a swapped unit value. It does not test an alignment discovered independently. The permuted control (wrong units) is the comparison that separates this from a generic effect of swapping.
2. **The baseline IIA with 1-dim DAS is not informative.**
   - The learned direction does no better than a random direction.
   - The shuffled-label programs score higher than the real ones (0.46–0.50). Their calls are almost constant, so on pairs where a disruptive patch moves the network to the majority class they agree by default.
   - A 1-dim patch at S changes the network's call on 50–80 % of pairs (n_eff ≈ 1,500–2,400 of 3,000), so the patch moves S off the natural jet distribution.
   - The B-thesis comparison of P's IIA against the baselines' IIA therefore cannot be read as a property of the baselines. A better test would need a DAS that preserves the distribution (clipped to the observed range, multi-dimensional, or trained only on pairs that change the program) and a no-op calibration.
3. **swap** (0.37 % of calls change, 30 jets) and **soft** (1.9 %) do not have the statistical power to separate the programs. The B6x request for an edit with > 5 % flips at fixed M0 and M2 is still open.
4. **Pruning is limited.** It was done by K, number rounding and dropping objects only; merging rank windows was not implemented. Dropping the 21 objects with cost < 1 % (P2r8d1) costs 15 points of same call. The P frontier does not reach below 12 kbit with useful fidelity, while STD and EFP reach 85–88 % same call at 1.7–3.4 kbit.
5. **No shuffled-label P.** Refitting the 144 unit formulas on shuffled values was not done (hours of MARS). The shuffled controls exist for STD, EFP and GBDT only.
6. **Fitting data differ.** P was fitted on `sample(6000, 0)` (30,000 jets, unit targets); the baselines on 13,333 fit jets (logit targets). The test jets exclude both fitting sets.
7. **The P program variables were chosen from earlier results** (E4/E5/E6/E7), not from this sample.

## Explainer entry (for site/explain.py, Paper B tab)

**Question.** A short program can imitate a network's calls on ordinary jets. Does it also respond to changes of the jet, and to swaps of its internal quantities, the way the network does? And is a program built in the network's own staged structure better at this than standard compact descriptions of the same size?

**Background.**
- *Description length* $\mathrm{DL}$ = bits to write the program: each stored number costs its width ($32$, or $b+8$ after rounding to $b$ significant bits), and each hinge factor $\max(0, x-t)$ costs $\log_2(\#\text{inputs}) + 2$ structure bits.
- *EFPs* are the energy-flow polynomials $\mathrm{EFP}_G=\sum_{i_1..i_N} z_{i_1}\cdots z_{i_N}\prod_{(k,l)\in G}\theta_{i_k i_l}$, with $z_i = p_{T,i}/\sum p_T$ and $\theta_{ij}=\Delta R_{ij}$. They form a complete linear basis of IRC-safe observables, and they are unchanged by a rigid translation of the jet.
- *Cohen's* $\kappa = (p_o - p_e)/(1-p_e)$ for the "call changed" indicators of program and network. Here $p_o$ is the observed agreement and $p_e = p_n p_s + (1-p_n)(1-p_s)$ is the agreement expected by chance.
- *IIA* (interchange-intervention accuracy): for a base jet $b$ and source jet $s$, replace program variable $V$ by $V(s)$ and the aligned network quantity by its value on $s$; IIA $=\Pr[\text{program call} = \text{network call}]$. *Effective IIA* is IIA restricted to pairs where the network's call changes.
- *DAS* (distributed alignment search) learns a direction $w$ with $S' = S_b + ((S_s - S_b)\cdot w)\,w$.
- *BH* (Benjamini–Hochberg) controls the false-discovery rate over many tests.

**Method.**
- Programs: the nested unit-level program P (7 sizes, 12–367 kbit) against MARS on standard observables, MARS on EFPs, and gradient-boosted trees (0.9–368 kbit). All are fitted to the network's outputs.
- Edits: 0.06 rad translation, one added soft particle, a pT swap of ranks 13/14, prong-sharing $dc=+0.3$, and minimal call flips.
- IIA: 3,000 random test-jet pairs, with permuted-alignment and shuffled-label controls.

**Result** (figure `b8.png`).
- Under the 0.06 rad translation (56 % of calls change), P reaches $\kappa = 0.46$–$0.67$ at ≥ 62 kbit. The best baselines reach $\kappa \le 0.35$, and EFPs reach $0$ by construction.
- On prong sharing, P reaches $\kappa = 0.53$–$0.57$ against $0.38$–$0.51$ for the baselines.
- Swapping single S objects or z windows between jets gives effective IIA $0.40$–$0.84$ against $0.18$–$0.36$ for wrong-unit pairings.

**Checks.**
- The unedited network has 80.88 % accuracy on the test jets.
- The exact tail reproduces the network's logits to $0$.
- The float tail matches 99.2 % of the exact calls.

**Why it matters.** The network's staged program carries information about how the network responds to edits, beyond what a same-size formula on standard or IRC-safe observables carries. This holds at large size (≥ 60 kbit). Below ≈ 30 kbit the standard-observable formulas reproduce natural calls better.

**Glossary additions:**
- **DL:** description length in bits.
- **EFP:** energy-flow polynomial.
- **κ (Cohen's kappa):** chance-corrected agreement of the program's and the network's call-change indicators.
- **IIA:** interchange-intervention accuracy.
- **Effective IIA:** IIA over the pairs where the network's call changes.
- **DAS:** distributed alignment search.
- **BH:** Benjamini–Hochberg false-discovery-rate correction.
- **MARS:** multivariate adaptive regression splines.
- **R (margin-response ratio):** mean change of the program's margin divided by the mean change of the network's margin under the same edit.
