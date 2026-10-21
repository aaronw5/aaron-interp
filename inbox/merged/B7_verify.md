# B7 verification: counterfactual-guided program extraction (fresh seeds + leave-one-in control)

Script: `interp/b7_verify.py` (reuses the pool and fit code of `interp/b7_counterfactual_programs.py` unchanged). Job
`b7-verify` (1 CPU / 2 GiB, wall 1,259 s, pod peak 1,915 MiB, 0.81 cores). Node hcc-nrp-pki-c1705.unl.edu.
Results: `results/jedi/b7_verify.json` (all rows and tests), `b7_verify_pool.json` (jets, flip rates, padding check),
figure `results/jedi/b7_verify.png`, log `logs/b7-verify.log`.

## Conclusion

The fresh-sample rerun confirms two parts of B7. First, the programs do not learn the network's own smallest call-flipping edits,
even when those edits are in the fit (kappa about 0.2). Second, no surrogate can respond to rigid rotations. The headline
prong-sharing result also reappears: kappa for the 50-term program goes from 0.30 to 0.44. But nearly the same rise (to 0.42) comes
from adding rotated copies of about 1,800 ordinary jets. These rows look like ordinary jets to the program's inputs and carry no
edit information. So the rise reflects how sensitive the program fit is to any added data, not information carried by the
network-labelled counterfactuals. The B7 statements that the program gains more than a GBDT of equal size, and that holding out
the 0.06 rad shift lowers agreement, are not reproduced. (Kappa = Cohen's kappa, defined below: how well the program's
call changes match the network's, where 0 is the rule "the call never changes".)

## What was checked

**Code review** of `interp/b7_counterfactual_programs.py`:
- No wrong stage: logits come from `jedi_core.Jedi.run`, the exact fixed-point forward pass.
- No split leakage:
  - Base jets C are split by jet, and every edit inherits its base jet's half.
  - The natural fit, natural test and C samples are de-duplicated with `b6w_symreg.indices`.
- Padding:
  - *trans* and *trans_c* move only real particles (pT > 0).
  - *dil* scales all angles and *rot* rotates all angles. Both are safe only if padded slots carry zero angles. Checked on the
    fresh samples: 364,769 / 183,716 / 73,327 padded slots, max |angle| on padding = 0.0. So padding is unchanged.
- *cf* edits are kept only if |dm|/m < 1e-3 and the axis moves < 1 mrad (E36 check). D(dc) uses the E35 acceptance.
- Shuffled control: `build_condition('shuf')` permutes logits *within each family* across all its sets (e.g. trans, trans_c and
  dil together). It therefore keeps each family's label distribution and destroys only the jet-by-jet pairing.

No bug in the code changes the numbers. Three problems with how the numbers are read:
1. **Matched-bit GBDT member switching.** `gbdt@K` is re-chosen in every fit condition. The members are tiny: 3–10
   boosting iterations, at most 14 kbit. In B7, gbdt@30 is `gbdt8_L8_T3` under nat but `gbdt4_L4_T10` under lofo_D. So
   "program gain minus GBDT gain" compares changes of different models. With one fixed GBDT member, B7's own per-jet file gives a
   Dmin agreement change lofo_D − nat of +1.34 points for the 72-kbit `gbdt8_L8_T30`. That is the same size as prog50 (+1.24).
2. **Confidence intervals cover jet sampling only.** The bootstrap resamples test base jets for fixed fitted surrogates. It does
   not include fit-to-fit variability of forward-stepwise MARS, which changes when the fit rows change. The leave-one-in control
   below shows that this variability is as large as the effects being tested.
3. **Wording.** "Holding the family out lowers the program's agreement by 2–3 points" compares lofo_trans with nat. That fit
   *adds* the other families' edits, with the shift family absent from both fits. It is not a comparison of held-out with
   included. Also, "does not come from the network's labels" (trans, shuf) should read "does not need the jet-by-jet pairing of
   labels; the family's label distribution, which shuffling keeps, is enough".

## Procedure (rerun)

- **Network:** official JEDI-linear 3-feature n64, exact forward pass. Every jet, natural or edited, is labelled by its 5
  logits, and the call is the argmax.
- **Jets:** the same construction as B7, with every `jedi_core.sample` / `indices` seed offset by +1000.
  - natural fit: sample(4000, 4901) = 20,000 jets, split into 13,333 fit and 6,667 dev.
  - natural test: sample(2000, 4902) minus overlaps = 9,173.
  - base C: sample(800, 4903) minus overlaps = 3,572, split by jet into a 1,786 fit half and a 1,786 test half.
  - Edit rng seeds (3904–3910) are as in B7.
- **Unedited accuracy:** 80.89 % (4901, N 20,000) and 80.57 % (4902, N 9,173). Both are in the 80.6–81.3 % band within one binomial
  SE (0.28 / 0.41 points).
- **Edit families** (identical code). Network flip rates on the test half:
  - *cf*, minimal flip edits (E36 gradient search at fixed jet four-momentum): 49.6 % (N 999; 1,947 edits pass the check).
  - *cfr*, random edits of the same size: 5.9 %.
  - *D(dc)*, rest-frame rotation of the two kT prongs' decay axis at fixed jet four-momentum, which changes the energy sharing
    between the prongs, dc = ±0.15, ±0.3: 12.5 % (N 7,136 rows on 1,786 base jets).
  - *cn* controls (the same rest-frame angle about the prong axis, sharing held): 4.3–7.3 %.
  - *trans*, deta += 0.06 rad on every real particle: 56.0 %. *trans_c*, each particle moved 0.06 rad in its own random
    direction: 51.9 %. *dil*, angles scaled to the same M2 = Σ pT r² increase: 44.8 %.
  - *rot*, rigid rotation about the axis by 22.5–90°: 4.8 % (B7: 3.4 %).
- **Fit conditions:**
  - The B7 conditions: nat, lofo_D, lofo_trans, lofo_cf, all, shuf (logits permuted within family) and rnd (+7,144 random
    non-targeted edits).
  - New, leave-one-in: **only_rot**, **only_cf** and **only_trans** = natural jets plus the fit-half rows of that one family
    (1,786 / 1,894 / 4,000 rows).
  - **Why only_rot is a control.** Both observable pools are invariant under rotation about the axis. A rotated jet therefore
    gives the program the same inputs as its unrotated base jet, which is an ordinary jet from sample 4903. The label is the
    network's output on the rotated jet, and it differs from the base call on 4.8 % of jets. only_rot thus adds about 1,800
    ordinary jets with the network's labels. It contains no edit the program can see, and no D, shift or flip-edit information.
- **Surrogates:**
  - prog K = 8, 16, 30, 50: forward-stepwise MARS hinge program on the 98 rank-window inputs, with the B7 settings. K = 50 is
    about 22 kbit.
  - The B7 quantised-GBDT grid (16 members), with gbdt@K chosen as in B7.
  - mlp32. hprog, cart and mlp8 were dropped to save time.
- **Metrics:** computed on test-half rows, as in B7.
  - *Network flip:* the exact call on the edited jet differs from the call on its base jet. *Surrogate flip:* the same, using the
    surrogate's own calls.
  - *Agreement:* the fraction of rows where the surrogate's flip / no-flip matches the network's.
  - *Cohen's kappa:* $\kappa = (p_o - p_e)/(1 - p_e)$, where $p_o$ is the agreement and $p_e$ is the agreement expected from the
    two flip rates alone. The rule "the call never changes" has $\kappa = 0$.
- **Statistics:**
  - 5,000 cluster-bootstrap resamples over base jets (all edits of one base jet move together).
  - Paired differences in agreement and in kappa. H0: no difference between the two fit conditions on the same rows; H1: a
    difference.
  - Holm over the 40 pre-listed tests, for agreement and kappa separately.
  - Two-sided bootstrap p, floor 2e-4, so the Holm floor is 0.008.

## Results

**Minimal flip edits (cfmin, 999 base jets).** Kappa [95 % CI]:

| fit | prog30 | prog50 | mlp32 |
|---|---|---|---|
| nat | 0.179 [0.13, 0.23] | 0.166 [0.11, 0.22] | 0.237 [0.18, 0.29] |
| lofo_cf | 0.171 | 0.157 | 0.160 |
| all (in fit) | 0.192 | 0.202 | 0.181 |
| shuf | 0.202 | 0.235 | 0.216 |

- lofo_cf − nat, prog30: agreement −0.40 [−2.6, +1.8] points, Holm 1.
- all − nat, prog50: +1.80 [−0.8, +4.5], Holm 1.
- B7 reported prog 0.15–0.17 and every surrogate in 0.07–0.24. Reproduced.

**Prong-sharing edit D(dc) (Dmin, 7,136 rows on 1,786 base jets).** prog50 kappa [95 % CI], with Δ = difference from nat:

| fit | kappa | Δ kappa vs nat [CI], Holm | Δ agreement vs nat (points) [CI], Holm |
|---|---|---|---|
| nat | 0.302 [0.26, 0.34] | | |
| rnd (+7,144 random edits) | 0.369 | +0.067 [+0.027, +0.106], 0.022 | +0.25 [−0.48, +0.95], 1 |
| only_rot (+1,786 ordinary jets, rotated) | 0.424 | +0.122 [+0.082, +0.164], 0.008 | +1.16 [+0.45, +1.89], 0.075 |
| only_cf | 0.396 | +0.094 [+0.049, +0.138], 0.008 | +0.60 [−0.17, +1.37], 1 |
| only_trans | 0.431 | +0.129 [+0.083, +0.175], 0.008 | +1.22 [+0.42, +2.05], 0.12 |
| lofo_D (B7's held-out condition) | 0.441 [0.40, 0.48] | +0.139 [+0.095, +0.184], 0.008 | +1.36 [+0.57, +2.16], 0.008 |
| all (D in fit) | 0.392 | all − lofo_D −0.049 [−0.085, −0.012], 0.18 | −0.74 [−1.43, −0.07], 0.64 |
| shuf | 0.336 | | |

- B7 reported 0.317 → 0.447. Reproduced in size: 0.302 → 0.441.
- only_rot reproduces 88 % of the kappa rise (0.122 of 0.139). Its rows show the program only ordinary jets.
  - Test, lofo_D vs only_rot: Δ kappa ≈ +0.017, well inside the CIs of both conditions.
- Including D edits in the fit (all) gives a lower held-out kappa than leaving them out (lofo_D), in both B7 and the rerun.
- mlp32: lofo_D − nat Δ kappa +0.042 [+0.001, +0.082], Holm 0.70. B7 had +0.09.
- Program gain minus matched-bit GBDT gain (lofo_D − nat):
  - K 30: −0.24 [−1.27, +0.78] points.
  - K 50: −0.69 [−1.78, +0.43].
  - B7 had +1.25 / +1.55 with CIs excluding 0. Not reproduced.
- A fixed 72-kbit GBDT (`gbdt8_L8_T30`) moves 0.343 → 0.452 under lofo_D and 0.330 under only_rot.

**0.06 rad shift family (trans, 5,358 rows).**

| prog30 kappa | nat | lofo_trans | rnd | only_rot | only_trans | all | shuf |
|---|---|---|---|---|---|---|---|
| B7 | 0.609 | 0.563 | 0.611 | – | – | 0.655 | 0.677 |
| verify | 0.537 | 0.529 | 0.609 | 0.555 | 0.692 | 0.689 | 0.640 |

- lofo_trans − nat agreement:
  - prog30: −0.41 [−1.34, +0.49], Holm 1.
  - prog50: +1.53 [+0.56, +2.48], Holm 0.032.
  - B7 had −2.35 and −3.42, so the B7 drop is not reproduced. For prog50 the sign is reversed.
- all − shuf (label specificity of the in-fit gain):
  - prog30: +2.43 [+1.27, +3.62], Holm 0.020.
  - prog50: +3.06 [+1.89, +4.22], Holm 0.008.
  - B7 had −1.05 / +0.64.
- On the fresh sample the shuffled fit keeps about 2/3 of the in-fit gain (kappa 0.537 → 0.640, against 0.689 with true labels).
  The rest depends on the jet-by-jet labels.
- The natural-only program's kappa on this family differs between the two samples: 0.609 vs 0.537, with no overlap of CIs. That
  is the size of the fit-to-fit variation.

**Rigid rotation.** The network flips 4.8 % of calls. Every rank-window surrogate has kappa 0, as expected by construction.

**Natural same call** (prog30, fraction of 9,173 natural test jets where program and network calls agree):
- 91.07 % (nat), 90.59 % (lofo_D), 90.97 % (only_rot), 87.49 % (shuf).
- Within 0.5 points for every non-shuffled fit. Reproduced.

## Competing hypotheses for the D(dc) rise

- **H_cf: the network-labelled edits carry information about the sharing response.**
  - Predicts: lofo_D > only_rot ≈ nat, and the rise grows when D edits are in the fit.
  - Observed: only_rot ≈ lofo_D, and all < lofo_D. Not supported.
- **H_data: any added rows perturb the forward-stepwise MARS term selection.**
  - Predicts: rises of similar size from any added set, including sets with no visible edit, and noise-level differences between
    them.
  - Observed: only_rot, only_trans, only_cf and lofo_D all fall in 0.40–0.44; rnd gives 0.37. Consistent.
  - The rnd rows contain small random edits that also change the base pT-rank structure, so rnd is not the null of
    "ordinary jets added".
- **H_shift: off-axis coverage from the shifted jets.**
  - Predicts: only_trans > only_rot.
  - Observed: 0.431 vs 0.424. Not separated.

## Corrected reading

- **Minimal flip edits.** Every surrogate stays at kappa 0.13–0.24 on both samples, whether the edits are absent, held out,
  included, or included with shuffled labels. This part of B7 holds.
- **Rigid rotation.** The network flips 3.4–4.8 % of calls, and no surrogate on these rotation-invariant pools registers it. This
  part holds.
- **D(dc).** The rise in held-out kappa from 0.30 to 0.44 (50-term program) reproduces. Adding about 1,800 ordinary jets in
  rotated form gives 0.42, and D edits inside the fit give 0.39. So the rise is not attributable to network-labelled
  counterfactuals, and it does not show that the sharing response is expressible in the rank-window pool. The
  program-vs-matched-GBDT difference does not reproduce: B7's gbdt@K changes member between conditions, and a fixed 72-kbit
  GBDT gains as much.
- **0.06 rad shift.**
  - The B7 drop for lofo_trans vs nat (−2.35 points) does not reproduce: −0.41 (prog30) and +1.53 (prog50).
  - On the fresh sample, about 2/3 of the in-fit gain survives label shuffling within family; the rest is specific to the
    network's labels.
- **Fit variability.** The natural-only program's kappa differs between samples by up to 0.07 (0.609 vs 0.537 on trans).
  Condition-to-condition differences of about 0.1 in kappa are within what refitting on a different set of rows produces. A
  test of fit conditions needs repeated fits (several fit subsamples or seeds per condition), not only a bootstrap over test
  jets.

## Remaining issues

- The CIs and Holm p values here also cover test-jet sampling only. Repeated MARS fits per condition were not run, because each
  costs about 85 s per condition on a fast node.
- McNemar on rows ignores the clustering of edits per base jet. The bootstrap is used as the primary test.
- hprog, cart and mlp8 were not rerun.
- `job.sh` (node exclusions) and `interp/b7_verify.py` are uncommitted. The ledger agent commits them.
- During sync, several `tar` processes from concurrent syncs were in D state on the helper pod (CephFS). The verify files reached
  the volume (md5 checked), and the job was submitted directly.

## Glossary additions

- **only_<family> (leave-one-in):** natural jets plus a single edit family in the fit.
- **only_rot as a data-only control:** rotated jets, which a rotation-invariant program sees as ordinary jets.
- **fit-to-fit variability:** the change in a surrogate's score when it is refitted on a different set of rows.
