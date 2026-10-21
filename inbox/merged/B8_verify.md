# B8 verification: program fidelity vs description length, rerun on a fresh sample with same-input baselines added

**Script:** `interp/b8_verify.py` (new; it imports B8's helpers from `interp/b8_fidelity_dl_iia.py`, and that file is unchanged).

**Job:** b8-verify, 1 CPU / 2 GiB. It ran three stages in one Job: `pool && efp && main`.
- The first pod stalled on node patternlab.calit2.optiputer.net. Its python process sat in D state in `ceph_mdsc_wait_request`, the same CephFS stall listed in job.sh. That Job was deleted and resubmitted, and the second pod ran on k8s-gpu-6.ucsc.edu.
- **Suggestion for job.sh:** add patternlab.calit2.optiputer.net to the node exclusion list. I did not edit job.sh, because it is outside the directories I may write to.
- Exit 0; wall 1,124 s; pod peak memory 2,047 MiB, which is at the 2 GiB limit; 0.42 cores on average.
- The "Jax plugin configuration error … cuInit" lines in the log are JAX's CUDA probe failing on a CPU node. JAX then runs on CPU.

**Files:**
- `$AI_RESULTS/jedi/b8_verify.json`, copied to `results/jedi/b8_verify.json`.
- Figure: `results/jedi/b8_verify.png`, rendered locally from the JSON.
- Log: `logs/b8-verify.log`.
- The intermediate npz files were deleted at the end of the run.

## Conclusion

On a new set of jets, the B8 numbers for the nested program reproduce:
- same call 94.4 % at full size (367 kbit);
- κ = 0.66 for agreement with the network's call changes under the 0.06 rad shift.

B8 lacked one comparison: formulas or boosted trees built on the **same rank-window inputs as the program**, but fitted straight to the network's outputs without the program's layer-by-layer structure. These baselines follow the network's response to the shift as well as the program does, or better, at the same or fewer bits. A 6.6-kbit direct formula reaches κ 0.61, against the program's 0.45 at 62 kbit. So the shift advantage that B8 reports against standard observables, EFPs and trees comes from the program's input variables, not from its staged structure. Two other edges are larger and not traced to the inputs:
- Under the prong-sharing edit, the program leads the same-input baselines by 0.10–0.26 κ, but an EFP formula of 22 kbit matches or exceeds it.
- The program's unit-swap test (IIA) reproduces at full size. Below 100 kbit its mean falls under the 0.6 target once the uncontrolled "all 19 S units" variable is set aside.

## What was checked in the code (interp/b8_fidelity_dl_iia.py)

No defect was found that changes the B8 numbers.
- **Samples:** `indices()` reproduces `jedi_core.sample` exactly. The test exclusion of the fit sample and of `sample(6000, 0)` is correct.
- **Padding:** real particles are contiguous at the front of every jet (checked: `contiguous_real = true`). The soft particle therefore goes into the first padding slot, and the trans edit moves real particles only.
- **Edits:** the EFP translation invariance holds numerically, with max relative difference 5e-7 on 500 shifted jets.
- **Stages and quantizers:**
  - The exact tail reproduces the network's logits with max |Δ| = 0.
  - The P program goes S → h4 → h5 → z and then through the network's own d7 with its input quantizer (`e11_verify.Last`).
  - IIA swaps use the same unit indices in program and network.
- **κ and bootstrap:** the code is correct, with paired resamples for P − baseline.

These are design issues in B8, not bugs:
1. **The STD and EFP baselines are not size-matched at ≥ 62 kbit.**
   - Their largest members are 13.9 and 22 kbit, against P's 62–367 kbit.
   - Rerun with STD MARS extended to K = 60 and 100 (29 and 49 kbit): κ trans drops to 0.11–0.13. Larger STD formulas do not close the translation gap, so this point does not change the STD comparison.
2. **The missing control is a baseline on P's own input set.**
   - B8's GBDT inputs are 67 standard observables and EFPs, of which only ΣpT·rⁿ (n = 1–3) depend on the jet axis. B8 states that "GBDT's inputs are not translation invariant", but in practice they are nearly invariant.
   - Earlier nodes B4v and B6/B6v already found that rank-window trees and GBDT follow the shift. The rerun adds these baselines; see below.
3. **The "mean effective IIA" includes "all 19 live S units", which has no permuted control.**
   - Its value is 0.83–0.94, so it raises the mean.
   - Without it, B8's own P8r10 mean is 0.580, not 0.607, and P8 is 0.578. Prediction (3), IIA ≥ 0.6, is then met at P30 and P16 only.
4. **κ trans has a high reference value that uses no jet information.**
   - Under the shift, the network changes 99 % of its W calls and 88 % of its Z calls, against 6–47 % for g, q and t.
   - The rule "the call changes if and only if the network's base call is W or Z" reaches κ = 0.574. That is above P8r10 (0.454), and below P16 (0.605) and P30 (0.660).
   - κ trans therefore mostly measures whether a program knows which jets the network calls W or Z.
5. **The figure (`b8.png`) suptitle makes comparative claims that the same-input baselines contradict.** The swap panel title also says "0 %" where 0.37 % is meant (0.4 % in the suptitle).

## Procedure of the rerun

**Network.** JEDI-linear 3-feature n64, exact fixed point. Accuracy on the fresh test sample is 81.20 %, inside the 80.6–81.3 % band.

**Fresh test sample.** `sample(2000, 8810)` minus every jet in:
- `sample(4000, 808)`, B8's fit and dev jets;
- `sample(6000, 0)`, the jets the nested program was fitted on;
- `sample(2000, 809)`, B8's test sample.

This leaves N = 7,855 jets.

**Fitting data.** Unchanged from B8: the fit and dev rows of sample 808 (13,333 and 6,667 jets), with their network logits, standard observables and EFPs read from B8's `b8_pool.npz` and `b8_efp.npz`. EFPs for the fresh jets were computed with energyflow 1.4 in `venv-ef`, with the same EFPSet as B8.

**Edits on the fresh jets.** Same code and seeds as B8:
- **trans:** Δη + 0.06 rad on every real particle;
- **soft:** one extra particle at 0.5 × the softest pT, at radius 0.4·√U with uniform azimuth, in the first empty slot (N = 6,032);
- **swap:** the pT values of slots 13 and 14 exchanged (N = 7,834);
- **D:** prong-sharing dc = +0.3 on 1,500 random jets (1,497 accepted), with control **cn**, the same angle about the prong axis.

The E36 minimal-flip search (cf) was not repeated, to keep the run within 1 CPU and 2 GiB.

**Programs.**
- **B8's own programs:** the seven P levels, STD MARS (K = 2–30, plus the new K = 60 and 100), EFP MARS (K = 2–50), and GBDT on STD ∪ EFP (the q = 8 grid).
- **New baseline DIRECT:** MARS from the nested program's own physics set (the observable library plus the rank-window sums H:…, 475 inputs) straight to the 5 logits.
  - The term list is B4's stored forward path (`b4_direct_terms.json`, selected on `sample(6000, 0)`).
  - Coefficients for each prefix K = 8, 16, 32, 64, 100 per logit are refitted by least squares on B8's fit rows.
  - DL uses the B8 bit rule: 32 bits per number, plus log2(#inputs) + 2 structure bits per hinge factor.
- **New baseline PGBDT:** HistGradientBoosting regressors, one per logit, on the same 475 physics inputs quantised to 8 bits. Settings (leaves, iterations): (8, 30), (8, 100), (16, 100). DL uses B6w's `hgb_bits_q`.

**Scores.** As in B8:
- same call;
- Cohen κ between the program's and the network's call-change indicators (0 = never-change rule, 1 = identical changes);
- margin ratio R (program's mean margin change / network's mean margin change);
- IIA for P30, P8r10 and P2r10 on 3,000 fresh pairs (rng 8813), with 5 permuted-unit draws.

**Statistics.**
- Bootstrap B = 1,000 over jets (over pairs for IIA).
- Matched DL: within each family, the member with bits ≤ P's bits and the highest dev same call.
- Paired bootstrap of P − baseline, with BH over 210 tests.
- H0: P's score equals the matched baseline's. H1: they differ.

## Results

**Network call changes** (fresh sample vs B8):

| edit | trans | soft | swap | D | cn |
|---|---|---|---|---|---|
| fresh | 55.7 [54.6, 56.8] % | 1.89 % | 0.49 % | 20.6 % | 7.3 % |
| B8 | 56.3 % | 1.90 % | 0.37 % | 20.3 % | 6.4 % |

**Reproduction of B8's P and baseline numbers** (fresh / B8):

| program | DL (kbit) | same call % | κ trans | κ D | R trans |
|---|---|---|---|---|---|
| P30 | 367 | 94.42 / 94.0 | 0.660 / 0.669 | 0.586 / 0.567 | 0.97 / 0.97 |
| P8r10 | 62.5 | 91.05 / 90.5 | 0.454 / 0.462 | 0.468 / 0.532 | 0.86 / 0.85 |
| P2r10 | 15.5 | 83.91 / 83.3 | 0.408 / 0.413 | 0.391 / 0.408 | 0.68 / 0.68 |
| STD30 | 13.9 | 90.96 / 90.4 | 0.276 / 0.281 | 0.410 / 0.381 | 0.37 / 0.37 |
| EFP50 | 22.0 | 91.66 / 91.0 | 0 / 0 | 0.554 / 0.514 | 0 / 0 |
| GBDT q8 L8 T100 | 196 | 91.78 / 91.2 | 0.029 / 0.035 | 0.461 / 0.421 | 0.14 / 0.14 |

**Same-input baselines (new control).** CIs are 95 % bootstrap.

| program | DL (kbit) | same call % | κ trans | κ D | κ soft |
|---|---|---|---|---|---|
| DIRECT8 | 3.1 | 88.5 | 0.400 [0.383, 0.419] | 0.157 | 0.051 |
| DIRECT16 | 6.6 | 91.0 [90.4, 91.7] | 0.613 [0.596, 0.628] | 0.181 | 0.093 |
| DIRECT32 | 13.3 | 92.0 | 0.539 | 0.263 | 0.055 |
| DIRECT64 | 27.3 | 92.7 | 0.254 | 0.303 | 0.084 |
| DIRECT100 | 43.2 | 92.9 [92.3, 93.4] | 0.516 [0.498, 0.535] | 0.329 | 0.124 |
| PGBDT L8 T30 | 91.1 | 91.9 | 0.588 [0.570, 0.606] | 0.369 | 0.154 |
| PGBDT L8 T100 | 284 | 92.8 | 0.599 [0.581, 0.616] | 0.453 | 0.187 |
| PGBDT L16 T100 | 511 | 92.6 | 0.467 | 0.503 | 0.131 |

κ trans of DIRECT is not monotone in K (0.61, 0.54, 0.25, 0.52). Its response to the shift depends on which hinge terms enter, so DIRECT's κ trans is not a stable function of size.

**Matched DL, P − baseline** (95 % CI, BH q over 210 tests):

| P level | vs DIRECT (≤ P bits) | vs PGBDT | vs EFP50 | vs STD (largest ≤ P bits) |
|---|---|---|---|---|
| P30, 367 kbit | same call +1.6 [1.1, 2.1] pts; κ trans +0.145 [0.126, 0.163]; κ D +0.26 | L8 T100 (284 kbit): same call +1.6; κ trans +0.061 [0.040, 0.083] (q 0.002); κ D +0.13 (q 0.007) | κ D +0.032 [−0.023, 0.087] (q 0.37) | STD100: κ trans +0.53 |
| P16, 208 kbit | DIRECT100: same call +0.5 (q 0.13); κ trans +0.09; κ D +0.22 | L8 T30 (91 kbit): κ trans +0.018 [−0.003, 0.040] (q 0.18); κ D +0.18 | κ D −0.002 (q 0.99) | κ trans +0.47 |
| P8, 102 kbit | DIRECT100: same call −1.8 [−2.3, −1.2]; κ trans −0.060 [−0.081, −0.041]; κ D +0.15 | L8 T30 (91 kbit): same call −0.9; κ trans −0.132; κ D +0.11 | κ D −0.079 [−0.138, −0.022] (q 0.007) | STD100: κ D +0.05 (q 0.27) |
| P8r10, 62.5 kbit | DIRECT100: same call −1.8 [−2.4, −1.2]; κ trans −0.062 [−0.083, −0.043]; κ soft −0.11; κ D +0.14 | no member ≤ 62.5 kbit; L8 T30 (91 kbit) gives κ trans −0.134 | κ D −0.085 [−0.141, −0.027] (q 0.007) | STD100: same call −0.8 (q 0.03); κ D +0.04 (q 0.34) |

**Reference rule for κ trans** (reads no jet input; uses only the network's base call):

| rule: "the call changes iff the network's base call is …" | W | Z | W or Z |
|---|---|---|---|
| κ trans | 0.325 | 0.226 | 0.574 |

The network's flip rate under the shift by base call: g 47 %, q 40 %, W 99 %, Z 88 %, t 7 %.

**IIA on fresh pairs.** Effective IIA is the agreement over the pairs where the network's call changes. Each cell gives aligned / permuted; for the means, n counts variables.

| | P30 | P8r10 | P2r10 |
|---|---|---|---|
| mean over 11 variables (all 19 S units excluded) | 0.657 / 0.250 | 0.557 / 0.271 | 0.459 / 0.276 |
| B8, same 11 variables | 0.660 | 0.580 | 0.472 |
| S0 | 0.778 / 0.254 | 0.703 / 0.293 | |
| S19 | 0.730 / 0.197 | 0.721 / 0.260 | |
| z windows together | 0.847 / 0.180 | 0.754 / 0.184 | |
| S1 | 0.382 / 0.250 | 0.228 / 0.253 (n.s., p 0.27) | |
| S29 | | | 0.235 / 0.308 (below permuted) |
| S30 | | | 0.224 / 0.264 (n.s.) |
| z10 | | | 0.285 / 0.308 (n.s.) |

At P30, every one of the 11 aligned − permuted differences has a bootstrap p of 0.001.

## Competing hypotheses on the translation κ

- **H_struct:** the staged structure (P → g → S → h4 → h5 → z) is what lets P follow the network under the shift. It predicts that same-input, non-staged baselines fall well below P at matched bits.
  - **Excluded at ≤ 102 kbit.** DIRECT100 (43 kbit) is above P8r10 and P8 by 0.06 κ, with q 0.002. DIRECT16 (6.6 kbit) is above P8r10 by 0.16.
  - At 208 kbit, P16 equals PGBDT at 91 kbit (+0.018, q 0.18).
  - Only at 367 kbit does P30 exceed the same-input GBDT, by 0.06 [0.04, 0.08].
- **H_inputs:** the advantage comes from the rank-window (pT, r) inputs, which respond to a shift of the axis. It predicts that same-input baselines match P and that observable-pool baselines do not.
  - **Consistent with the rerun.** It also agrees with the earlier B4v and B6/B6v results.
- **Prong sharing (D):** P leads the same-input baselines by 0.10–0.26 κ (q ≤ 0.007), but EFP50 (22 kbit) equals or exceeds P at every level. The D lead is therefore not specific to P's structure either. It shows that the rank-window input set handles D worse when it is not staged, while EFPs handle D well.

## Corrected reading

- **What reproduces on the fresh sample:**
  - At ≥ 62 kbit the nested program follows the network's call changes under the 0.06 rad translation more closely than MARS on standard observables (up to 49 kbit), MARS on EFPs, and boosted trees on standard observables plus EFPs, by +0.32 to +0.66 κ. The EFP value of 0 holds by construction.
- **What the added baselines change:**
  - Baselines on the program's own rank-window inputs, without its staged structure, match or exceed this κ at fewer bits:
    - direct MARS: 0.61 at 6.6 kbit and 0.52 at 43 kbit;
    - trees: 0.59 at 91 kbit;
    - P8r10: 0.45 at 62 kbit.
  - P exceeds them only at 367 kbit (+0.06).
  - A rule that flips exactly the network's W and Z calls reaches κ 0.57.
- **Natural same call:**
  - At 62–102 kbit P is 0.8 points below the 49-kbit STD formula and 1.8 points below the 43-kbit direct formula.
  - P exceeds every baseline only at ≥ 208 kbit.
- **Prong sharing:** P's κ (0.47–0.59) exceeds the same-input baselines and STD/GBDT at ≥ 208 kbit, but EFP50 (22 kbit, κ 0.55) equals it at P30 and P16 and exceeds it at P8 and P8r10.
- **IIA:**
  - Swapping a single S object or z window between jets at the same network units reproduces the network's new call in 0.38–0.85 of call-changing swaps at P30, against 0.18–0.33 for wrong-unit pairings.
  - The mean over variables that have a permuted control is 0.66 at P30 and 0.56–0.58 at P8r10.
  - This alignment holds by construction, since the formulas were fitted to those units.
- **Soft and swap edits:** they change 1.9 % and 0.5 % of calls and do not separate the programs.

**Thesis effect:** weakens the B4 refinement proposed by B8.
- The translation-κ lead is an input-set effect (H_inputs).
- The prong-sharing lead does not hold against EFPs.
- The IIA ≥ 0.6 prediction is met at ≥ 208 kbit only.

## Issues of the verification

- cf was not rerun. B8's κ cf comparison (+0.17 at P30) was not tested against the same-input baselines.
- DIRECT's terms were selected by B4 on `sample(6000, 0)` with logit targets, and only the coefficients were refitted. PGBDT used three settings only. Neither family was tuned for κ.
- No DAS rerun for the baselines; B8's issue 2 still stands.
- The pod's peak memory reached the 2 GiB limit. The run completed, but a larger rerun would need a 3 GiB request.

## Explainer additions (for the B8 entry in site/explain.py)

**Checks** (append to B8's checks):
- Fresh sample $N = 7{,}855$ (seed 8810): P's same call and κ reproduce within 0.06.
- **Same-input control.** Formulas and trees on the program's own rank-window inputs, fitted directly to the logits, reach the program's translation $\kappa$ at fewer bits:
  - $\kappa = 0.61$ at 6.6 kbit;
  - P reaches $0.45$ at 62 kbit.
- The rule "flip iff the network's base call $\in\{W, Z\}$" gives $\kappa = 0.57$.

**Glossary additions:**
- **DIRECT (same-input MARS):** hinge formulas from the nested program's input variables straight to the 5 logits, with no intermediate stages.
- **PGBDT:** gradient-boosted trees on those same inputs.
- **Class-rule reference κ:** the κ of a rule that predicts a call change from the network's unedited call alone.
