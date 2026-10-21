# B7: counterfactual-guided program extraction (Paper B, claims B2 / B2o, B4 / B6; exploratory method)

Script: `interp/b7_counterfactual_programs.py` (modes `pool`, `fit`). Jobs: `b7-run` (pool + first fit, 1 CPU / 3 GiB, wall 2,010 s,
peak 1.9 GiB), `b7-fit2` (fit rerun, 1 CPU / 2 GiB, wall 3,062 s, peak 0.6 GiB). Results: `results/jedi/b7.json` (all rows and
tests), `b7_pool.json` (jets, edit statistics, accuracies), `b7_perjet.npz` (per-row agreement indicators), figure `b7.png`.
Logs: `logs/b7-run.log`, `logs/b7-fit2.log`. Smoke run: `b7smoke.*`.

## Conclusion

We tried to make the extracted programs follow the network's call changes better by adding edited jets to the fit, each labelled
with the network's own output. Programs fitted this way do not do better on a kind of edit that was left out of the fit. On the
network's own smallest call-flipping edits, agreement stays low even when those edits are included in the fit (kappa about 0.15).
The one exception is the prong-sharing edit: with edits from the other families added, the 50-term program reaches kappa 0.45,
up from 0.32. Part of that rise also appears with random, non-targeted edits, so how much of it comes from the network-targeted
edits specifically is not settled. Kappa (Cohen's kappa) measures how well the program's call changes match the network's,
where 0 is the rule "the call never changes" and 1 is a perfect match.

## Question

Programs extracted from JEDI-linear (rank-window hinge programs, B6 / B6w) match the network's calls on natural jets (~91 % same
call at 30 terms), but on the network's own call-flipping edits they barely beat the rule "the call never changes" (E36v: Cohen
kappa 0.11). The candidate explanation is that natural jets under-constrain the program. If so, fitting on extra jets placed where
the network's decision changes (counterfactuals), labelled by the network, should help, including on edit types not used in the
fit. The alternative is that the network's edit responses depend on structure absent from the observable pools, or need many
more terms. In that case adding edited jets helps only on the families that were in the fit (memorisation), or not at all.

## Procedure

**Network.** Official JEDI-linear, 3-feature, n64, exact fixed-point forward pass (`jedi_core.Jedi`). Every jet, natural or
edited, is labelled with the network's 5 logits (g, q, W, Z, t). The network's call is the argmax.

**Jets** (`jedi_core.sample`, test file, overlaps removed):
- natural fit: sample(4000, seed 3901) = 20,000 jets. Permutation rng 0: 13,333 fit rows and 6,667 dev rows (dev is used to pick
  the matched-bit GBDT by natural same call).
- natural test: sample(2000, seed 3902) minus the 3901 jets = 9,183 jets.
- counterfactual base C: sample(800, seed 3903) minus the 3901 and 3902 jets = 3,551 jets, split by jet (rng 3904) into a fit
  half and a test half. Every edited jet inherits its base jet's half, so no base jet behind a test row appears in any fit.

**Edit families.** Particles are given as pT, deta, dphi about the stored jet axis, and padding is never moved.
- **cf** (E36, regenerated on C):
  - *cf (minimal flip):* the smallest edit (per real particle pT_i -> pT_i exp(u_i); angles + 0.02 rad x w_i;
    $\epsilon^2=\sum_i (\Delta p_{T,i}/p_{T,i})^2 + |\Delta\vec r_i|^2/0.02^2$) found by 200 Adam steps on the float + masks
    program that makes it call the partner class (W <-> Z, q <-> g). The jet four-momentum (pT, mass, axis) is held by a
    Gauss-Newton projection, and edits are kept if |dm|/m < 1e-3 and the axis moves < 1 mrad.
  - Searched: 2,761 jets (called g, q, W or Z by both the exact network and the float program). The float program flips on
    74 % of them, and 2,044 edits pass the constraint check. Median epsilon is 5.96. On the test half the exact network flips on
    50.0 % of these jets (E36: 41 %).
  - *cfr:* one random edit per jet with the same v-space norm, tangent to the same constraints. The network flips on 6.6 %.
- **D** (E35 / H45):
  - *D(dc):* rest-frame rotation of the two exclusive-kT prongs' decay axis at fixed jet four-momentum, with all pairwise
    invariant masses and N held. dc = -0.3, -0.15, +0.15, +0.3 (dc > 0 means more asymmetric energy sharing between the
    prongs). Inputs are rebuilt about each jet's E-scheme axis, and the base is rebuilt the same way.
  - *cn(dc):* control with the same rest-frame angle about the prong axis (sharing and opening angle held).
  - Network flip rates on the test half: D 6.8–20.6 % (dc +0.3: 20.6 %), cn 4.1–6.9 %. Pooled D-only rows ("Dmin"): 12.3 %.
- **trans:**
  - *trans:* deta += 0.06 rad for every real particle. Flips 56.4 %.
  - *trans_c:* each particle moved 0.06 rad in its own random direction. Flips 53.7 %.
  - *dil:* all angles scaled per jet by $k=\sqrt{M_2(\mathrm{trans})/M_2}$, where $M_2=\sum p_T r^2$. This gives the same M2
    increase as the translation without moving the centroid (median k 1.19). Flips 46.3 %.
- **rot:** rigid rotation about the axis by |theta| ~ U(22.5°, 90°) with random sign (the A13 rigid-rotation edit). Every r, pT,
  M0, M2 and N is held. Flips 3.4 %.
- **rnd** (control pool, fit only): every real particle of a fit-half base jet moved 0.03 rad in a random direction, pT x
  exp(0.1 N(0,1)), re-sorted by pT; 4 draws per jet. This is extra data that is not aimed at the network's decision boundary.

**Observable pools** (B6w), recomputed from each edited jet:
- RW: 97 pT-rank-window energy-flow sums plus M0 = ΣpT.
- HAND: mass, soft-drop mass, tau21, tau32, D2, N, ΣpT, ΣpT·r, ΣpT·r², ΣpT·r³.

Both pools are invariant under rotation about the axis, so no surrogate can respond to *rot*. That family is a built-in null for
expressibility.

**Fit conditions.** Each family's fit-half rows are capped at 4,000:
- *nat (i):* natural jets only, 13,333 rows.
- *lofo_f (ii):* natural plus every family except f (+7,833 to 10,058 rows).
- *all (iii):* natural plus all four families (+11,833 rows).
- *shuf (control):* as *all*, but the logits are permuted among each family's fit rows.
- *rnd (control):* natural plus 7,100 rnd rows.

**Surrogates** (all regress the 5 logits; call = argmax):
- *prog:* forward-stepwise MARS on RW, prefixes K = 4, 8, 16, 30, 50 terms per logit, refitted by least squares. Description
  length (DL) uses the B6 rule; K 30 is ≈ 13 kbit.
- *hprog:* the same on HAND.
- *gbdt:* HistGBDT on RW quantised to 4 / 8 bits, with 4 / 8 leaves, 3 / 5 / 10 / 30 iterations, lr 0.5. "gbdt@K" is the member
  with bits ≤ prog-K bits and the highest natural dev same call (natural jets only decide).
- *cart:* CART regression tree of depth 4 / 6 / 8.
- *mlp8, mlp32:* small MLPs on standardised RW, 32 bits per parameter (mlp32 = 113 kbit, not size-matched).

**Metrics** (family test-half rows):
- *Network flip* = the exact call on the edit differs from the call on the base. *Surrogate flip* = the same with the surrogate's
  own calls.
- *Edit agreement* = fraction of rows where the surrogate's flip / no-flip decision matches the network's.
- *Cohen kappa* of the two flip indicators: agreement corrected for chance. Never-flip has kappa 0 and agreement 1 − flip rate.
- *r(Δmargin)* = Pearson correlation of the margin change between network and surrogate (margin = logit of the network's base
  call minus the largest other logit).
- *Natural same call* = % of the 9,183 natural test jets where surrogate and network calls agree.

**Statistics.**
- 95 % CIs: 1,000 bootstrap resamples of base jets (all edits of one base jet move together).
- Primary test, per family view f × surrogate: H0 agreement(lofo_f) = agreement(nat) on f's held-out rows; H1 they differ.
  Paired cluster-bootstrap CI and p, exact McNemar on per-row agreement (rows of one base jet are not independent, so the
  bootstrap is the primary p), and Holm over the 60 tests (6 family views × 10 surrogates).
- The views are cf, cfmin (minimal-flip rows only), D, Dmin (D rows only, no cn), trans and rot. cfmin and Dmin are subsets of
  cf and D.
- With B = 1,000 the smallest bootstrap p is 0.001, so the smallest Holm-adjusted bootstrap p is 0.06. The Holm-adjusted
  McNemar p is also given.
- Secondary tests: all vs nat (gain on a family that is in the fit), all vs shuf (label specificity), lofo vs rnd (targeted vs
  random extra data), prog gain minus matched-GBDT gain, and natural same-call change.

**Accuracy check (unedited network).**
| Sample | N | Accuracy |
|---|---|---|
| 3901 | 20,000 | 80.25 % |
| 3902 | 9,183 | 80.74 % |
| C | 3,551 | 81.92 % (rebuilt-axis C: 81.89 %) |

Sample 3901 is 0.35 points below the 80.6–81.3 % reference band; the binomial SE at N 20,000 is 0.28 points.

## Results (Cohen kappa with 95 % CI; held-out = family not in the fit)

| family view (network flip rate) | surrogate | nat (i) | lofo, held out (ii) | all, in fit (iii) | shuf | rnd |
|---|---|---|---|---|---|---|
| cfmin, minimal flips (50.0 %) | prog30 | 0.147 [0.101, 0.191] | 0.126 [0.084, 0.170] | 0.147 | 0.135 | 0.114 |
| | prog50 | 0.165 [0.118, 0.210] | 0.128 [0.087, 0.173] | 0.171 | 0.127 | 0.110 |
| | mlp32 | 0.238 [0.187, 0.291] | 0.177 [0.128, 0.228] | 0.196 | 0.301 | 0.185 |
| Dmin, prong sharing D(dc) (12.3 %) | prog30 | 0.283 [0.244, 0.324] | 0.370 [0.328, 0.409] | 0.366 | 0.320 | 0.294 |
| | prog50 | 0.317 [0.274, 0.358] | 0.447 [0.410, 0.485] | 0.416 | 0.320 | 0.365 |
| | mlp32 | 0.380 [0.341, 0.416] | 0.469 [0.431, 0.504] | 0.476 | 0.334 | 0.398 |
| trans (52.1 %) | prog30 | 0.609 [0.583, 0.634] | 0.563 [0.534, 0.593] | 0.655 | 0.677 | 0.611 |
| | prog50 | 0.618 [0.590, 0.644] | 0.551 [0.522, 0.582] | 0.672 | 0.660 | 0.605 |
| | mlp32 | 0.583 [0.554, 0.612] | 0.610 [0.582, 0.637] | 0.713 | 0.600 | 0.650 |
| rot (3.4 %) | every RW surrogate | 0 | 0 | 0 | 0 | 0 |

**Primary test, edit agreement lofo − nat (points, 95 % cluster-bootstrap CI, Holm over 60):**
- **cfmin.**
  - prog30 −1.08 [−3.35, +1.08]; prog50 −1.87 [−4.24, +0.50]; gbdt@30 −0.10 [−2.27, +2.27]; mlp32 −3.05 [−5.81, −0.39].
  - All have Holm p = 1. No surrogate or size gains.
- **Dmin.**
  - prog50 +1.24 [+0.44, +2.01] (bootstrap Holm 0.16, McNemar Holm 0.019); prog30 +0.73 [−0.11, +1.48].
  - mlp32 +2.16 [+1.30, +3.08] (Holm 0.06, McNemar Holm 1e-6); gbdt@30 −0.52 [−1.39, +0.39].
- **trans.**
  - prog30 −2.35 [−3.36, −1.33]; prog50 −3.42 [−4.54, −2.29] (both bootstrap Holm 0.06, McNemar Holm ≤ 4e-6).
  - hprog16 +4.49 [+3.45, +5.54]; gbdt@30 +0.17 [−1.09, +1.35].
- **rot:** 0 for every RW surrogate. The pools cannot represent a rotation.

**Secondary tests.**
- **Extra data not aimed at the network (lofo − rnd, agreement).**
  - Dmin: prog50 +1.04 [+0.31, +1.79] (Holm 0.27), mlp32 +1.75 [+0.87, +2.72] (Holm 0.06).
  - cfmin: prog30 +0.59 [−1.67, +2.86].
  - trans: prog30 −2.40 [−3.42, −1.37].
  - Of the prog50 rise on Dmin (kappa 0.317 → 0.447), the random pool gives 0.365.
- **In-fit gain (all − nat).**
  - cfmin: prog30 +0.00 [−2.27, +2.27]. Including the minimal-flip edits in the fit does not raise agreement on held-out jets
    of the same family.
  - Dmin: prog50 +1.04 [+0.25, +1.85].
  - trans: prog30 +2.21 [+1.13, +3.25], prog50 +2.63 [+1.56, +3.62].
- **Label specificity (all − shuf).**
  - trans: prog30 −1.05 [−2.31, +0.21], prog50 +0.64 [−0.58, +1.84]. The trans in-fit gain of the program is equally present
    with the labels shuffled within family, so it is not specific to the network's labels. mlp32 +5.61 [+4.37, +6.89] is
    specific.
  - Dmin: prog50 +4.35 [+3.34, +5.47], specific.
  - cfmin: prog30 +0.59 [−2.46, +3.55].
- **Program gain minus matched-bit GBDT gain (lofo − nat).**
  - Dmin: +1.56 [+0.49, +2.68] (K 16), +1.25 [+0.04, +2.41] (K 30), +1.55 [+0.48, +2.69] (K 50).
  - D (with cn): +0.9 to +1.0, CI excludes 0.
  - trans: −2.5 to −3.0, CI excludes 0.
  - cfmin: −3.05 [−5.71, −0.30] (K 16), −0.99 [−4.24, +2.36] (K 30).
- **Natural same call (cond − nat, prog30).**
  - lofo_cf −0.48 [−0.91, −0.03]; all +0.32 [−0.12, +0.76]; rnd +0.04; the other conditions are within ±0.5 points.
  - shuf −2.93 [−3.56, −2.28], the expected cost of wrong labels.
  - Every non-shuffled condition stays within 1 point.
- **Margin-change correlation (prog30).**
  - cfmin 0.21 (nat) → 0.23 (lofo); Dmin 0.51 → 0.51; trans 0.78 → 0.81.

**Competing predictions.**
- **H_compact** (natural data under-constrain a compact program): held-out kappa rises from ~0.1 to ≥ 0.4 at K ≤ 30 with < 1
  point natural same-call loss, and the rank-window program gains more than GBDT.
  - Not met on cfmin: kappa ≤ 0.13 held out and ≤ 0.17 even in the fit.
  - Not met on trans: held out lowers agreement.
  - On Dmin it is met at K 50 (0.447 [0.410, 0.485]) but not at K ≤ 30 (0.370 [0.328, 0.409]). The program gains 1.2–1.6
    points more than matched GBDT, and natural same call is unchanged.
- **H_absent** (structure absent from the pools, or many terms needed): kappa on held-out families near natural-only fits,
  gains only in-fit, and shuffled fits equal to real ones.
  - Matches cfmin and trans for the programs.
  - rot is excluded from the comparison by construction (the pools are rotation invariant).
  - On Dmin the held-out rise (+1.24 points agreement) and the label specificity (+4.35 over shuffled) do not follow H_absent.
    About 40 % of the kappa rise (0.317 → 0.365 of 0.447) is also produced by random edits.

## Reading (scoped to what was measured)

- On the network's own minimal flip edits, every surrogate stays at kappa 0.07–0.24 whether the edits are absent, held out, or
  included with their labels. That includes the 113-kbit MLP; the program is at 0.15–0.17. The shuffled-label fit gives the same
  kappa.
  - With the rank-window and standard-observable pools, these edits are not learnable from ~1,000 labelled examples.
  - What the edits change (re-sorting of pT ranks under small per-particle pT and angle changes at fixed four-momentum) is not
    captured by the pool sums at the resolution the fits reach.
- On the prong-sharing edit D(dc), adding the other families' edited jets raises held-out agreement for the program at K 50
  (kappa +0.13) and for the MLP (+0.09), with no natural same-call loss.
  - The program's gain is larger than matched-bit GBDT's (+1.2–1.6 points).
  - The rnd control reproduces part of it (kappa 0.365 vs 0.447), so the share due to network-targeted edits is about 0.08 in
    kappa, with a lofo − rnd agreement CI of [+0.31, +1.79] points.
  - The sharing response is partly expressible in the rank-window pool (kappa up to 0.45 at 50 terms per logit, about 22 kbit).
- On the 0.06 rad shift, holding the family out lowers the program's agreement by 2–3 points relative to natural-only fits. The
  in-fit gain (+2.2 to +2.6) is equally present with shuffled labels, so for the program it does not come from the network's
  labels.
- The network changes its call on 3.4 % of rigid rotations, and no surrogate on these pools can register that.

## Checks and caveats

- Unedited accuracy: 80.25 % (3901) is 0.35 points below the 80.6–81.3 % band; 80.74 % (3902) is inside it.
- **Debug attempt 1** (`b7-fit2`): the first fit's GBDT grid had no member below 13.5 kbit, so gbdt@16 / gbdt@30 were
  undefined. Iterations 3 and 5 were added and the fit was rerun on the same pool. MARS rows reproduce to within ±0.02 in
  kappa (prog50 nat cf 0.247 → 0.266 because of the refit). The first fit also lacked the cfmin / Dmin views and the per-jet
  file.
- **Two node stalls:**
  - `b7-smoke` stalled in CephFS (D state, d_alloc_parallel) on k8s-chase-ci-01.calit2.optiputer.net. That node was added to the
    job.sh exclusion list (job.sh edited; ledger to commit).
  - The `b7-fit2` node ran ~3.5× slower than `b7-run`.
- The minimal-flip test half has 1,015 base jets, so per-family CIs on cfmin are ±2–3 points of agreement.
- Fit-half rows are capped at 4,000 per family. D and cn share one cap, so ~2,000 D rows enter a fit.
- Holm floor 0.06 (bootstrap p resolution 0.001 with 60 tests). McNemar on rows ignores the clustering of up to 8 edits per base
  jet.
- MARS candidate gains are computed on 8,000 random fit rows (in *all*, about half are edited jets).
- The rot family cannot be fitted by construction. A half-plane pool (e11 handbuilt(halfplanes=True)) would be needed to test it.

## Explainer entry (for site/explain.py, Paper B tab)

- **Question.** Programs extracted from JEDI-linear agree with it on ordinary jets, but on the edits that flip the network's own
  call they barely beat "never change the call". Does fitting the program on edited jets, labelled by the network, fix that for
  edits it was not shown?
- **Background.**
  - A *counterfactual* here is an edited jet with the network's output on it.
  - The *never-flip rule* predicts that no edit changes the call.
  - Cohen's kappa $\kappa = (p_o - p_e)/(1 - p_e)$ compares the observed agreement $p_o$ of flip decisions with the agreement
    $p_e$ expected by chance from the two flip rates. Never-flip has $\kappa = 0$.
  - Description length (DL) counts the bits needed to write the program (32 bits per number plus the choice of inputs).
- **Method.** Edit families:
  - minimal flip edits found by gradient search at fixed jet four-momentum;
  - the prong-sharing rotation $D(\Delta c)$ at fixed four-momentum and pair masses;
  - a 0.06 rad shift with a random-direction and a matched-dilation control;
  - a rigid rotation.

  Hinge programs (MARS on 98 pT-rank-window sums), quantised GBDT, CART and MLPs are fitted to the network's logits on natural
  jets alone, on natural jets plus all edit families but one (leave-one-family-out), and on everything. Agreement is measured on
  held-out jets of the left-out family. Controls: labels shuffled within family, and the same number of random non-targeted
  edits.
- **Result** (figure b7.png). Held-out kappa on minimal flips is 0.15 → 0.13 (prog30). On the 0.06 rad shift it is 0.61 → 0.56.
  On the prong-sharing edit it is 0.32 → 0.45 (prog50); random edits give 0.37.
- **Checks.** Network accuracy 80.3 / 80.7 %. Natural same call is unchanged within 1 point. The rotation family is invisible to
  rotation-invariant inputs (kappa 0 by construction).
- **Why it matters.** Paper B compares programs with baselines at equal description length on edit agreement. B7 shows that the
  low agreement on the network's own flip edits is not removed by training on counterfactuals in these observable pools, so the
  limit lies in the pools or the program form, not in the amount of natural data.

**Glossary additions:**
- Cohen's kappa
- counterfactual (network-labelled edited jet)
- leave-one-family-out (LOFO)
- never-flip rule
- matched-bit GBDT (gbdt@K)
- cfmin / Dmin (the minimal-flip rows only; the D(dc) rows without the cn controls)
