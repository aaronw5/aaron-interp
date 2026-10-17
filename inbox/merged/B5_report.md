# B5: compact direct hinge programs on axis-relative vs invariant observables (thesis v3, B2–B4, review weakness 3)

- Scripts: `interp/b5_compact_jedi.py` (modes `fit`, `eval`, `all`) and `interp/b5_plot.py` (summary figure, rendered locally from `b5.json`).
- Jobs (1 CPU / 2 GiB each, exit 0):
  - b5-smoke: 300 jets per class, prefix b5smoke; 829 s.
  - b5-fit: 679 s.
  - b5-eval: 546 s.
  - Logs are in `logs/b5-*.log`.
  - The pod peak memory read 2047 MiB in all three jobs. This is the same reading as in B4; the cgroup peak includes the page cache.
- Results:
  - `results/jedi/b5.json`: all metrics, CIs and tests.
  - `results/jedi/b5_programs.json`: the 50-step paths and the programs with T ∈ {5, 10, 20, 30, 50}, with LaTeX.
  - `results/jedi/b5_calls.npz`: per-jet calls of every model on every set.
- Figures:
  - `results/jedi/b5_summary.png`: the main figure.
  - `results/jedi/b5.png`: 6 panels, including the GBDT, the hardest-particle axis and the trans_c control.

## Procedure

- **Splits (identical to B4 / E11).**
  - Fit: sample(6000, seed 0) after the rng-0 permutation, 20,000 jets.
  - Dev: the remaining 10,000 jets.
  - Test: sample(2000, seed 11) minus the seed-0 jets, N 8,848.
  - Confirmation: sample(2000, seed 91) minus the seed-0 and seed-11 jets, N 8,519.
- **Target.** The network's 5 logits; calls are the argmax.
- **Pools.** Keys come from the E11 physics set plus extras: 510 inputs in total. 128 invariant non-mass observables are in no pool: N-subjettiness, ECFs, pair distances and pair masses, subjets, soft drop, mass of the top-k particles.
  - **pT-only (in both pools), 123 inputs:** ΣpT, pT_i, z_i, multiplicities, and ΣpT and N over pT-rank windows.
  - **P_axis, 347 inputs:** pT-only plus 221 observables measured about the stored axis, plus explicit $|\vec M_1|^2$, $M_0 M_2$ and $|\vec M_1|^2/M_0$. The axis observables include the radial window sums $\Sigma p_T r^2$, $\Sigma r^2$, $\Sigma r$, $\Sigma(r-0.2)_+$ and $\Sigma r[r<0.165]$, and also $|\vec M_1|$, $\sqrt{M_0M_2}$, girth, the $(\Delta\eta,\Delta\phi)$ tensor, rings, Ψ, and Δη_i, Δφ_i, ΔR_i.
  - **P_inv, 158 inputs:** pT-only plus $m$, $m/M_0$, $(m/M_0)^2$ and the 32 B4 EFP features (β = 1, 2, with their logs).
  - **P_all, 382 inputs:** P_axis ∪ P_inv.
- **Fitter.** Forward-stepwise MARS with the candidate rules of `jetdistill.mars.fit_neuron`:
  - Candidates: 19 quantile knots per input, and products with the 10 inputs most correlated with the residual.
  - Gain is computed on a 6,000-jet subsample (rng 1). The target is the raw logit.
  - One term budget is shared across the 5 logits. At each step, each logit proposes its best term, refitted by least squares on the fit jets. The term goes to the logit with the largest reduction of its dev MSE.
  - The programs with T ∈ {5, 10, 20, 30, 50} are prefixes of the 50-step path, each refitted.
  - DL in bits = T (log2 #pool inputs + 16).
- **References.**
  - mars30: stored E11 nested program, 3,824 terms.
  - mars16: B4 refit, 2,219 terms.
  - GBDT on the call (B4 settings, E11 physics set) with 10–80 leaves.
- **Edits on the test jets.** Physics inputs are recomputed from the edited particles. On the unedited jets, the recomputed inputs match the stored ones to a maximum relative difference of 5.7e-8, and the network calls are identical on 100 % of jets.

| edit | definition | matched control |
|---|---|---|
| trans | every real particle Δη += 0.06 rad | trans_c: each particle moved 0.06 rad in a random direction (rng 101) |
| rot | 45° rotation about the axis | none |
| wta | re-axis on the WTA axis (E17 code); median displacement 0.040 rad | wta_c: axis moved by the same distance in a random direction (rng 104) |
| hardest | re-axis on the hardest particle | hardest_c: same construction as wta_c |
| jump | X4 rule on the test jets, all classes (details below); N 212 of 5,876 two-prong jets | jump_null: a random other particle of the same subjet raised by the same ΔpT |
| A13 | swap of (Δη, Δφ) between pT ranks 13 and 14 | A13_c |

The jump rule:
- exclusive kT clustering to two subjets, with z_A in (0.2, 0.8);
- the hardest particle of the other subjet is raised to 1.005 pT_hardest, a raise of at most 5 % of its own pT;
- the slots are then re-sorted by pT, so the raised particle moves to slot 0.

- **Metrics.** For each model and edit:
  - % calls changed;
  - W→Z after the edit;
  - Δaccuracy;
  - per-jet edit agreement (B4 definition).

  Each has a 95 % jet-bootstrap CI (B 1000).
- **Paired tests.** The per-jet call-change indicator is compared between surrogate and network as a paired difference (bootstrap CI) and with McNemar. Holm correction is applied over 253 cells (23 models × 11 edits).
- **"Reproduces".** The surrogate's call-change CI overlaps the network's CI, and the paired-difference CI contains 0.

## Network check

- Test accuracy: 80.42 % [79.63, 81.22]. Confirmation: 80.10 % [79.28, 80.95]. These match B4 / E11 on the same splits. The point estimate is 0.2 points below the requested 80.6–81.3 %, and the CI contains 80.9 %.
- Network calls changed:

| edit | calls changed | control | control calls changed |
|---|---|---|---|
| trans | 55.84 % [54.81, 56.82] | trans_c | 52.15 % [51.11, 53.21] |
| wta | 30.48 % [29.55, 31.37] | wta_c | 34.11 % [33.14, 35.04] |
| hardest | 31.97 % [31.02, 32.87] | hardest_c | 35.39 % [34.40, 36.31] |
| jump | 1.89 % [0.47, 3.77] (4 of 212) | jump_null | 3.30 % [0.94, 6.13] (7 of 212) |
| rot | 3.94 % | | |
| A13 | 1.31 % | A13_c | 7.99 % |

- Paired McNemar tests of the network:
  - jump vs jump_null: 1 vs 4 discordant jets, p 0.375.
  - wta vs wta_c: p 1.9e-27. The random-direction re-axis of the same length changes more calls than the WTA re-axis.
- W→Z rate among true-W jets:
  - unedited: 2.9 %;
  - after trans: 75.4 %;
  - after wta: 59.2 %;
  - after hardest: 51.8 %;
  - after the jump: 0 of 40 W jets.

## Results (test; confirmation in brackets)

| program | DL (bits) | same call % | trans | wta | hardest | jump | trans_c |
|---|---|---|---|---|---|---|---|
| network | | | 55.8 [54.8, 56.8] | 30.5 [29.6, 31.4] | 32.0 [31.0, 32.9] | 1.9 [0.5, 3.8] | 52.1 [51.1, 53.2] |
| P_axis T 5 | 5 | 61.8 (62.2) | 48.5 | 14.8 | 17.9 | 0.0 | 51.7 ✓ |
| P_axis T 10 | 10 | 68.6 (67.8) | 36.0 | 12.7 | 15.7 | 0.0 | 36.0 |
| P_axis T 20 | 20 (489) | 83.2 [82.3, 83.9] (82.3) | 51.2 [50.3, 52.3] | 26.8 [25.9, 27.7] | 28.0 | 0.0 | 51.1 ✓ |
| P_axis T 30 | 30 | 87.6 (86.4) | 68.1 | 33.0 | 34.7 | 0.0 | 63.3 |
| P_axis T 50 | 50 (1,222) | 89.5 [88.8, 90.1] (88.2 [87.6, 88.9]) | 68.1 [67.1, 69.0] | 32.5 [31.5, 33.5] | 33.7 | 0.5 ✓ | 63.8 |
| P_inv T 5–50 | 5–50 (T 50: 1,165) | 72.7 → 88.8 [88.1, 89.4] (87.8) | 0.0 | 0.0 | 0.0 | 0–0.9 | 53.4–60.5 |
| P_all T 50 | 50 | 88.2 (87.1) | 44.6 | 15.8 | 16.7 | 0.0 | 62.7 |
| GBDT L4 T2 | 40 leaves | 86.3 (85.5) | 55.5 ✓ | 33.0 | 34.5 | 1.4 ✓ | 59.3 |
| mars16 | 2,219 | 92.9 (92.5) | 51.4 | 25.9 | 27.6 | 0.5 ✓ | 51.2 |
| mars30 | 3,824 | 94.0 [93.5, 94.5] (93.5) | 59.7 [58.7, 60.6] | 29.8 ✓ | 31.2 ✓ | 1.4 ✓ | 54.6 |

Entries in the edit columns are % calls changed. ✓ marks "reproduces" as defined above.

**Smallest T at which all three of trans, WTA and jump are reproduced:** none in any pool (P_axis, P_inv, P_all) at T ≤ 50. mars30 also does not reproduce trans: it changes 59.7 % of calls against the network's 55.8 %, a paired difference of +3.8 [+3.0, +4.6] points, Holm p 2.6e-16.

### P_axis, T = 50
- trans: paired difference +12.2 [+11.3, +13.1] points, Holm p 5.5e-136.
- wta: paired difference +2.0 [+1.3, +2.8] points, Holm p 6.6e-5.
- hardest: paired difference +1.7 [+0.9, +2.5] points, Holm p 0.002.
- wta_c and hardest_c: reproduced. The paired differences are −0.0 [−0.7, +0.7] and within the CI.
- Per-jet agreement minus the never-flip reference: trans +7.3, wta +3.4, hardest +4.1 points.
- Same-call comparisons:
  - vs mars30: −4.5 [−5.1, −3.9] points, McNemar p 2.4e-48.
  - vs P_inv T 50: +0.7 [+0.2, +1.3] points, p 0.014.

### P_axis, T = 20
- trans: −4.6 [−5.8, −3.4] points.
- wta: −3.6 [−4.6, −2.6] points.
- Both are below the network's call changes, and Holm p ≤ 2.3e-10 for both.

### P_inv, every T
- The programs change 0 calls under trans, wta, hardest, wta_c and hardest_c. Their inputs are exactly invariant under these edits, so this result holds by construction.
- Paired difference on trans: −55.8 [−56.8, −54.8] points. On wta: −30.5 points. Holm p < 1e-300 for both.
- Same call reaches 88.8 % at T 50, which is 0.7 points below P_axis T 50.
- P_inv changes 53.4–60.5 % of calls under the control trans_c. The network changes 52.1 %. P_inv T 5 is closest at +1.3 points, and its paired-difference CI excludes 0.

### Direction of the call changes under trans (W→Z among true-W jets; network 75.4 %)

| program | W→Z, % | difference from network, points |
|---|---|---|
| P_axis T 10 | 64.3 | −11.1 [−13.9, −7.7] |
| P_axis T 20 | 22.5 | −52.8 [−55.5, −50.1] |
| P_axis T 50 | 40.9 | −34.5 [−36.9, −31.8] |
| P_all T 50 | 78.0 | +2.7 [+0.7, +5.0] |
| mars30 | 77.5 | +2.1 [+0.3, +4.1] |
| GBDT L4 T2 | 1.7 | −73.6 |

- GBDT L4 T2 matches the network's call-change rate under trans (55.5 %, paired difference −0.3 [−1.3, +0.6]). Its W→Z rate is 1.7 %, and its per-jet agreement is 4.1 points below the never-flip reference.
- Under wta (network 59.2 %): P_axis T 50 gives 47.9 % and mars30 gives 71.2 %.

### Axis jump
- In the network, the axis jump changes 4 of 212 calls and the null edit changes 7. The paired test gives p 0.375.
- Every program changes 0–3 of the 212 calls under the jump.
- Whether a program counts as reproducing the jump depends on 4 jets. For example, P_axis T 30 changes 0 calls and its paired CI is [−3.8, −0.5].

### Readable program: P_axis, T = 20 (20 hinge terms, 489 bits)

Same call on test is 83.2 % and on confirmation 82.3 %. Inputs:
- $M_0=\Sigma p_T$;
- $M_2=\Sigma p_T r^2$ about the stored axis;
- window sums over pT ranks [a-b].

$$\begin{aligned}\ell_{g} &= 1.764 - 1.582\,(0.8221-\mathrm{planar\_flow})_+ + 0.1872\,(0.8221-\mathrm{planar\_flow})_+\,(N_{[13\text{-}63]}-35)_+ + 1.373\,(1.905-\Sigma r_{[0\text{-}25]})_+ - 2.892\,(0.9942-\Sigma r^2_{[13\text{-}63]})_+ + 35.21\,(\log M_0-6.92)_+ - 0.02069\,(\Sigma p_{T[0\text{-}49]}-890.6)_+ + 0.4843\,(5.6-\Sigma p_T r^2_{[0\text{-}25]})_+ \\ \ell_{q} &= 0.4687 + 0.0009145\,(6361-M_0 M_2)_+ - 0.354\,(7.371-\Sigma r_{[all]})_+ \\ \ell_{W} &= -2.992 + 0.7265\,(10.26-\Sigma r_{[all]})_+ - 0.01541\,(1008-\Sigma p_{T[0\text{-}49]})_+ - 621.8\,(0.003486-\langle \Delta R^2\rangle_{[0\text{-}49]})_+ \\ \ell_{Z} &= -0.7783 - 0.2427\,(\Sigma r_{[all]}-3.878)_+ - 0.01811\,(1008-\Sigma p_{T[0\text{-}49]})_+ - 5354\,(0.007844-\langle \Delta R^2\rangle_{[0\text{-}49]})_+ + 0.06044\,(120.4-\sqrt{M_0 M_2})_+ + 2663\,(0.006294-\langle \Delta R^2\rangle_{[0\text{-}39]})_+ + 2206\,(0.009618-\langle \Delta R^2\rangle)_+ \\ \ell_{t} &= 4.175 - 254.1\,(0.02557-\langle \Delta R^2\rangle)_+ + 0.6202\,(0.02557-\langle \Delta R^2\rangle)_+\,(1017-M_0)_+\end{aligned}$$

Here $\langle\Delta R^2\rangle_{[w]} = \Sigma_{i\in w} p_{T,i}\Delta R_i^2/\Sigma_{i\in w}p_{T,i}$, which is `girth2` (`girth2_topk`), the pT-weighted mean ΔR² about the stored axis. The 50-term P_axis program is listed in `b5_programs.json` (`programs.axis50`). Its additional terms are further hinges on $M_0M_2$, $\Sigma p_T r^2$ over rank windows, $\langle\Delta R^2\rangle$ and Σr.

## Hypotheses

- **H_thesis.** P_axis reproduces all three edits at T ≤ 50 with same call ≥ 90 %, and P_inv shows near-zero response that is rejected against the network.
  - The P_axis part is not met. No T meets the criterion on trans or wta: T 20 is below the network, and T 30 and T 50 are above it. Same call at T 50 is 89.5 % [88.8, 90.1].
  - The P_inv part is met: zero response, Holm p < 1e-300. This holds by construction, because the inputs are invariant. Same call is within 0.7 points of P_axis.
- **H_alt1** (no compact account; only mars16/30 reproduce).
  - The first half is consistent with the data: no P_axis program meets the criterion.
  - mars16 does not reproduce trans or wta either.
  - mars30 reproduces wta, hardest and jump, but not trans.
- **H_alt2** (the response is a generic sensitivity).
  - The clause "P_inv also reproduces" is excluded for trans, wta and hardest (p < 1e-300).
  - The clause "trans_c is reproduced equally by all pools" is not observed: P_axis T 5 and T 20 reproduce trans_c, P_inv does not at any T, and P_all does not.
- **Network control.** The network changes more calls under the random-direction re-axis of the same length than under the WTA re-axis (34.1 vs 30.5 %). Under the √(M0·M2)-about-the-axis description, a re-axis by d adds about $M_0 d^2$ to $M_2$ whatever its direction, so wta_c is not a null for that description. P_axis T 50 gives 34.1 vs 32.5 % on the same pair of edits.

## Reading (scoped to these measurements)

- **Pool comparison.** At equal term budgets, programs restricted to axis-relative observables change calls under the 0.06 rad shift and under the WTA and hardest-particle re-axis at rates that bracket the network's. T 20 gives 51 / 27 / 28 % and T 50 gives 68 / 33 / 34 %, against the network's 56 / 30 / 32 %. Programs on invariant observables change none, and same call on unedited jets differs by ≤ 1.2 points between the two pools at T ≥ 20.
- **Strict criterion and direction.** None of the ≤ 50-term programs meets the strict criterion on the shift. The direction of their call changes (W→Z 23–64 %) does not match the network's 75 %, which mars30 and P_all T 50 match within 3 points.
- **Axis jump.** The jump on this network gives no response distinguishable from its null (4 vs 7 of 212 jets). It therefore does not separate the programs here.

## Explainer entry (for site/explain.py, Paper B tab; not written there: site/ is outside this worker's write scope)

- **Question.** Can a short formula on observables measured about the jet axis reproduce how JEDI-linear's calls change when the jet is shifted or re-axised, and can a formula on axis-free observables of equal length do the same?
- **Background.**
  - Earlier results showed the network's mass variable is formed about the given axis, close to $\sqrt{M_0M_2}$ with $M_2=\Sigma p_T r^2$, rather than the invariant mass $m$. A 0.06 rad shift of all particles leaves $m$ unchanged and turns 75 % of W calls into Z.
  - The full nested hinge program (3,824 terms) reproduces the network's calls on 94 % of jets.
  - $m$ and the energy-flow polynomials depend only on pairwise angles, so they do not change under a shift or a re-axis.
- **Method.**
  - We built the 5 logits as sums of hinge terms $(x-t)_+$, $(t-x)_+$ and products of two hinges, chosen one at a time and shared across logits, up to $T$ = 50 terms.
  - Three input pools: (i) observables about the axis plus pT sums; (ii) $m$, EFPs plus pT sums; (iii) both.
  - Edits on 8,848 held-out jets: a shift of $+0.06$ in Δη; re-axis on the WTA axis and on the hardest particle, each with a random-direction control of equal length; the X4 axis jump with its null; the rank-13/14 swap. The physics inputs were recomputed after each edit.
  - Tests: per-jet paired bootstrap and McNemar against the network, with Holm correction over 253 cells.
- **Result (figure `b5_summary.png`).**
  - The axis-pool programs change 51 % (T 20) to 68 % (T 50) of calls under the shift, against the network's 56 %. Under the WTA re-axis they change 27–33 %, against 30 %.
  - The invariant-pool programs change 0 %, at the same same-call rate (88.8 vs 89.5 % at T 50).
  - No program with ≤ 50 terms matches the shift response within its CI, and none matches its W→Z direction (23–64 % vs 75 %). The 3,824-term program and the 50-term union program match the direction within 3 points.
  - The 20-term axis program is shown above.
- **Checks.**
  - The recomputed physics matches the stored physics to 5.7e-8.
  - Network accuracy is 80.42 % [79.63, 81.22].
  - GBDT at equal size matches the shift's call-change rate (55.5 %) but sends W to classes other than Z (W→Z 1.7 %).
  - The confirmation sample gives same-call rates within 1.3 points of test.
  - The axis jump is uninformative for this network: 4 vs 7 changed calls in 212 jets for jump vs null.
- **Why it matters.** Same-call accuracy on unedited jets does not distinguish an axis-based from an invariant-mass-based program: the two differ by ≤ 1.2 points. The edits do distinguish them, at p < 1e-300. Reproducing the size and direction of the shift response needed more than 50 terms in this fitter, or the union pool.

## Issues / notes

- **Network control.** The 0.2-point accuracy offset is the same as in B4 on these splits.
- **Greedy allocation.** Terms are allocated by dev logit MSE, not by call agreement. P_all T 50 has a lower same call than P_axis T 50 (88.2 vs 89.5 %) despite its larger pool.
- **Jump selection.** The FMAX = 5 % rule selects only 212 test jets. Every jump comparison is limited by the 4 network call changes.
- **Strictness of "reproduces".** At N 8,848 the call-change CIs are about ±1 point wide, and the nested 3,824-term reference fails the definition on trans. A tolerance-based reading is not part of the pre-registered plan and is not applied.
- **sklearn and scipy.** Both are present in the job env. scipy is not in the local system python; the paired post-hoc numbers above (network jump vs null, wta vs wta_c, P_axis vs P_inv same call) were computed locally from `b5_calls.npz` with the jedi-distill venv python, in seconds.
