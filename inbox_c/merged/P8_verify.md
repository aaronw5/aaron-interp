# P8 verify: per-block patching, DAS mass direction, substitution (ParT_full) (verifier, 2026-10-19)

Job `anrunw-ai-c-p8-verify` (TITAN Xp, 390 s), `python -m interp.p8v_verify_part`; smoke `c-p8v-smoke` passed.
Script `interp/p8v_verify_part.py` (reuses P8's edit, hook, patching and DAS functions). Results
`results/part/p8_verify.json` (smoke: `p8_verify_smoke.json`). Logs `logs/c-p8-verify.log`, `logs/c-p8v-smoke.log`.

## Conclusion
ParT does not form its axis-relative mass at a single layer. The response to a φ shift at fixed invariant mass builds up a little
in every particle layer, and about half of it reaches the class token directly from the input embedding. One direction in the
jet-averaged particle representation is present from the embedding onward and carries about two thirds of the response. Across real
jets, this direction's value follows the axis-moment mass $\sqrt{\sum p_T\cdot\sum p_T\Delta R^2}$: donor-minus-recipient differences
correlate at 0.87. Copying its value from real jets moves the W/Z output in the direction the φ shift predicts. P8 read the opposite
because its substitution donors had equal invariant mass, which also fixes the axis-moment mass. The direction does not copy that
quantity exactly: per unit change of axis-moment mass it responds less to a pT rescaling than to a φ shift. Real-jet substitution
also gives under half of the effect calibrated on the edit.

## What was checked
**Code review of `interp/p8_mass_formation_part.py`:**
- Hook sites are correct. The attention-sublayer site is the `post_attn_norm` output, which comes after the head scaling `c_attn`
  and before dropout and the residual add. The MLP site is the `fc2` output. The block site is the whole block output.
  `cls_blocks[0]` output `[0]` is the class token.
- In-place residual adds do not corrupt the records or the replacements, because hook outputs are cloned.
- Padding: jet means use the mask. The `emb` hook runs before `masked_fill`. pp writes into padded slots are masked out as keys.
- Rows: random rows, sorted, never first-k. The build uses $\eta = \eta_{jet}+\mathrm{sign}(\eta_{jet})\Delta\eta$.
- Edits: φ/η/angle edits change only (Δη, Δφ). pT × c changes only `pt_log` and `e_log` (`logptrel`, `logerel` and ΔR are
  unchanged). DAS counterfactuals keep the unedited pair features, as the report states.
- No indexing bug found.

**Missing control, and the reason P8's main negative reading does not hold.** P8 defines $M_2 = S_2/S_0$, where
$S_0=\sum p_T$ and $S_2=\sum p_T\Delta R^2$. That is the normalised moment. The C3 quantity, E6's axis-moment mass, is
$\mathrm{mom}=\sqrt{S_0 S_2}$, which is approximately the invariant mass $m$ for natural jets. This has two consequences:
- **pT × 1.25 leaves $S_2/S_0$ fixed but raises mom by the same factor as $m$** (Δlog mom = +0.223, in P8's own `truth` block).
  So "the M2 direction also carries a pT rescaling" is what a mom-like variable predicts. It is not evidence against an
  axis-moment variable.
- **P8's substitution donors (same class, |Δm| < 3 GeV, |Δlog M2| ≥ 0.15) also hold mom fixed.** On the fresh sample,
  mean |Δlog mom| = 0.022 while mean |Δlog M2| = 0.46. The slope on Δlog M2 therefore tests $S_2/S_0$ at fixed mom, not the
  C3 quantity.

## Procedure (verify run)
- Model: ParT_full. Fresh sample: JetClass test_20M `WToQQ_102.root` + `ZToQQ_102.root` (P8 used `_101`), 5,000 random rows each,
  `default_rng(7)`, P8's build and edits. Fit/test halves come from a seed-7 permutation.
- Accuracy on the P1 sample.
- Patching on 2,000 random jets, edits φ + 0.06 and angles × $k_{match}$ (1.090): P8's `patching`, with resample and random-half
  controls; 1,000 bootstraps.
- DAS: 1-dim jet-mean subspaces at emb, b1, b4, b8, trained on φ 0.03/0.06/0.09 ("M2") and on pT × 0.8/1.25 ("M0"). Each used
  300 Adam steps (P8 used 150). Evaluation on 2,000 test-half jets, with 3 random directions per site.
- New, edit calibration of the direction. $s = \bar h\cdot q$ is the jet-mean projection on the 1-dim direction $q$, computed on
  10,000 jets through the per-particle route only (unedited pair features).
  - Quantity: $R_e = (\overline{\Delta s}/\overline{\Delta\log\mathrm{mom}})_e \,/\, (\overline{\Delta s}/\overline{\Delta\log\mathrm{mom}})_{\phi+0.06}$,
    with a paired bootstrap.
  - Predictions for $R_{pT\times1.25}$: normalised $M_2$ gives 0; log mom gives 1; invariant mass alone gives no φ response
    ($R$ unbounded).
  - The same ratio is computed for n98 under the full edits.
  - H0: $R=1$. H1: $R\neq1$. Holm over 8 tests (4 sites × 2 directions).
- New, natural-jet regression: $s \sim 1 + [Z] + \beta_0\log S_0 + \beta_2\log(S_2/S_0)$ on 10,000 base jets. log mom (or $m$)
  predicts $\beta_0/(2\beta_2)=1$; normalised $M_2$ predicts 0. 200 bootstrap resamples.
- New, substitution redesign at emb, b1 and b8 (k = 1), on 4,000 test-half recipients. Natural donors come from the same
  multiplicity batch and class.
  - Set A uses P8's rule. Set B requires |Δlog mom| ≥ 0.10, with no mass constraint.
  - The recipient's jet-mean projection is set to the donor's. Reference: self-interchange from the jet's own φ + 0.06
    counterfactual.
  - Metrics: slopes through 0 of Δz97 [s.d.] on Δs and on Δlog mom, and the ratio natural / self (paired bootstrap). Controls:
    3 random directions, and the observed (unpatched) z97 difference between donor and recipient. McNemar on 10-class correctness.

## Numbers
**Replication (fresh sample):**
- Accuracy: P1 sample 85.93 %; WZ 10-class 75.67 %.
- φ + 0.06: Δz97 −0.265 [−0.271, −0.259], W → Z 0.098 → 0.309 (P8: −0.272, 0.102 → 0.311).
- Patching (2,000 jets), φ + 0.06:
  - Largest single sublayer: MLP b3, 0.055 (P8 0.056).
  - All 16 sublayers: 0.420 [0.401, 0.439] (P8 0.418).
  - Blocks 1..l: 0.089, 0.192, 0.266, 0.313, 0.340, 0.371, 0.394.
  - Noise allsub: 0.325.
- Patching, angles × $k_{match}$: largest single sublayer attention b1, 0.090; allsub 0.490.
- DAS, 1-dim jm "M2" (logit recovery):

  | site | φ + 0.06 | pT × 1.25 | random | M0 direction on φ | cos(M2 dir, M0 dir) |
  |---|---|---|---|---|---|
  | emb | 0.677 [0.659, 0.694] | 0.576 | ≈0.000 | 0.429 | 0.64 |
  | b1 | 0.613 [0.593, 0.632] | 0.714 | 0.002 | 0.473 | 0.78 |
  | b4 | 0.629 | 0.792 | 0.032 | 0.548 | 0.85 |
  | b8 | 0.679 | 0.807 | 0.002 | 0.579 | 0.79 |

  - vs random at b1: +0.611 [0.590, 0.630], p_Holm 0.004 (floor).
  - M2 − M0 on φ at b1: +0.141 [0.135, 0.147].

**Output calibration:** n98 change per unit Δlog mom, relative to φ + 0.06:
- η + 0.06: 1.01 [0.99, 1.03].
- angles × $k_{match}$: 1.09 [1.07, 1.11].
- pT × 1.25: 0.743 [0.727, 0.758].

**Direction calibration** $R_e$ for the M2 direction:

| site | angles × $k_{match}$ | η + 0.06 | pT × 1.25 | natural $\beta_0/(2\beta_2)$ | natural $R^2$ (log mom alone / log M2 alone / log S0 alone) |
|---|---|---|---|---|---|
| emb | 0.517 | 0.513 | 0.399 [0.395, 0.404] | 0.981 [0.956, 1.003] | 0.59 / 0.37 / 0.05 |
| b1 | 0.878 [0.866, 0.891] | 0.881 | 0.714 [0.704, 0.725] | 1.022 [0.997, 1.045] | 0.60 / 0.36 / 0.06 |
| b4 | 0.931 | 0.995 | 0.801 [0.789, 0.813] | 0.942 [0.914, 0.967] | 0.61 / 0.42 / 0.09 |
| b8 | 0.940 | 1.003 | 0.847 [0.837, 0.859] | 0.805 [0.762, 0.848] | 0.41 / 0.35 / 0.16 |

- H0 $R_{pT}=1$ is rejected at every site (p_Holm 0.008, the floor). $R_{pT}=0$ is far outside every interval.
- Natural $R^2$ with log $m$ alone is within 0.01 of log mom at every site, so natural jets do not separate $m$ from mom. The
  φ response (mass fixed, non-zero, ≥ the matched real edit at b1–b8) does.
- At emb the direction's φ response per unit Δlog mom is not linear: −96, −68 and −61 for φ 0.03, 0.06 and 0.09. At b1 it is
  close to linear: 83, 78, 79.

**Substitution, block 1, k = 1:**

| donor set | N | mean abs Δlog mom | slope Δz97 / Δlog mom: natural | self (φ) | ratio natural / self | slope / Δlog M2 | corr(Δs, Δlog mom) | calls changed | random dirs: slope / Δlog mom |
|---|---|---|---|---|---|---|---|---|---|
| A (P8 rule) | 3,786 | 0.022 | −0.66 [−1.02, −0.29] | −1.96 | 0.34 [0.15, 0.53] | +0.007 [−0.013, 0.029] | 0.13 | 16.2 % | −0.009, +0.048, +0.017 |
| B (mom differs) | 4,000 | 0.272 | −0.79 [−0.86, −0.72] | −1.89 | 0.42 [0.38, 0.46] | −0.284 | 0.87 | 32.5 % | −0.010, −0.011, +0.007 |

- Per unit of the projection difference Δs, the natural/self slope ratio is 0.87 [0.82, 0.93] (A) and 0.53 [0.48, 0.58] (B).
- Observed (unpatched) z97 difference per Δlog mom in set B: −0.49 [−0.59, −0.39].
- McNemar (B): right → wrong 867, wrong → right 288 (p 9e-68). Random directions: 0.7–1.5 % of calls changed.
- emb and b8 give the same pattern. Set B ratio per Δlog mom: emb 0.26 [0.24, 0.29], b8 0.40 [0.37, 0.43]. corr(Δs, Δlog mom):
  emb −0.84, b8 −0.69; the sign of $q$ is arbitrary.
- H0: ratio = 0, rejected for all 6 site × set tests (p 0.001, Holm 0.006 floor). H0: ratio = 1, rejected for all 6.
- Set A replicates P8's null on Δlog M2 (P8: −0.007 [−0.025, 0.011]).

**Additional control:** the donor-resample control can exceed the jet's own clean value. Under φ + 0.06, the resample patch at
MLP b3 restores 0.130 against the jet's own clean value 0.055. In P8's figure this happens at attention b2 (0.077 vs 0.043) and
attention b7. The single-sublayer shares are therefore not separated from a mass-matched donor control. This does not change the
"no single sublayer > 0.084" statement.

## Competing hypotheses for the 1-dim direction
- (H_M2) It encodes the normalised moment $S_2/S_0$. This predicts $R_{pT}=0$, natural $\beta_0/(2\beta_2)=0$, and a set-A slope on
  Δlog M2 equal to the φ slope. **Excluded:** $R_{pT}$ 0.40–0.85, natural ratio 0.81–1.02, set-A Δlog M2 slope ≈ 0.
- (H_m) It encodes the invariant mass only. This predicts no φ response. **Excluded:** the φ response is 1.0–1.9 × the matched real
  edit's response.
- (H_mom) It encodes log mom (axis-moment mass) linearly. This predicts $R_{pT}=1$, natural ratio 1, and natural/self substitution
  ratio 1.
  - Supported in direction and sign: natural ratio 0.94–1.02 at emb–b4; the value follows Δlog mom (r 0.87 at b1); natural
    substitution moves n98 with the φ sign.
  - Not exact: $R_{pT}$ is 0.71 at b1 (the output itself gives 0.74), and the natural/self ratio is 0.42.
- (H_scale) A generic mass-evidence or energy-scale direction with no axis-moment structure. This predicts $\log S_0$ dominating
  natural variation. **Not supported:** $R^2$ with log $S_0$ alone is 0.05–0.16, against 0.41–0.61 with log mom.

## Issues found in P8
1. **Main issue.** The "not M2-specific" argument (pT × 1.25 reproduced 0.55–0.85) and the substitution null (ratio 0.004) both
   test the normalised moment $S_2/S_0$. They do not test the axis-moment mass $\sqrt{S_0S_2}$ that C3 and E6 use. P8's donors
   hold that quantity fixed (|Δlog mom| 0.022). With donors that differ in it, the direction's value tracks it (r 0.87) and
   substitution moves n98 with the φ-predicted sign (0.42 of the self slope; random ≈ 0). The conclusion "no linear jet-level
   variable equal to the axis moment was found" therefore does not hold for the axis-moment mass.
2. The thesis note says the subspace is "not M2-specific". This is correct only for $S_2/S_0$. The M2 and M0 directions are
   largely the same direction (|cos| 0.64–0.85), as one mass-like variable predicts.
3. The donor-resample control exceeds the own-clean restoration at some single sublayers (see above). This is not discussed in
   P8's report or figure caption.
4. The probe reading ("the invariant-mass probe reads φ + 0.06 as 0.86–0.99 of the matched real edit") matches this verify: the
   direction reads φ at 1/0.88–1/0.94 of the matched real edit at b1–b8.
5. Unchanged and replicated: the patching numbers, the 0.58 share carried by the residual path of the edited embedding, the DAS
   recoveries and random controls, and the flat-in-depth recovery. The site rule (block 1) is not an issue: emb and b8 give the
   same substitution pattern.

## Corrected reading
- Patching (replicated): no single sublayer restores more than about 0.09 of n98's change, and single-sublayer shares are within
  the range of a mass-matched donor control. Restoration accumulates by 0.03–0.10 per block. About 0.58 of the φ response
  bypasses all 16 particle-block sublayers through the residual path of the edited embedding.
- DAS (replicated): a 1-dim jet-mean direction reproduces 0.61–0.68 of the per-particle-route φ logit change at emb–b8
  (random ≤ 0.03).
- Correction to P8: this direction behaves as a linear read-out of a mass-like jet quantity that follows the axis-moment mass
  $\sqrt{\sum p_T\cdot\sum p_T\Delta R^2}$ rather than the invariant mass:
  - it responds to the mass-fixed φ shift at least as strongly as to a matched real mass change at b1–b8;
  - across natural jets it depends on $\log S_0$ and $\log(S_2/S_0)$ in the mom proportion (0.94–1.02 at emb–b4);
  - writing its value from real donors that differ in axis-moment mass moves n98 with the φ-predicted sign: 0.42 [0.38, 0.46] of
    the edit-calibrated slope at b1; random directions about 0.
- It is not an exact log-mom variable: a pT rescaling moves it 0.71 as much per unit Δlog mom as φ does at b1 (n98 itself: 0.74).
- It is already present at the embedding output, so the "formed at block 1–3" part of thesis C v2 step 3 remains unsupported.
- The P7 path shares (block-1 class heads 0.99) were not re-run here.

## Explainer entry (for site/explain.py)
**Question.** Does ParT carry an axis-moment mass $\mathrm{mom}=\sqrt{\sum_i p_{T,i}\cdot\sum_i p_{T,i}\Delta R_i^2}$ as a linear
jet-level variable, or only the normalised moment $M_2=\sum p_T\Delta R^2/\sum p_T$?

**Background.** For massless particles about the $p_T$-weighted axis, at small angles, $m^2 \approx \sum p_T\cdot\sum p_T\Delta R^2$.
A rigid φ shift raises mom at fixed $m$. A pT rescaling raises mom and $m$ together and leaves $M_2$ fixed. Equal-mass natural jets
have nearly equal mom.

**Method.**
1. Take P8's 1-dim jet-mean direction $q$ (DAS on φ shifts). Measure $\Delta s/\Delta\log\mathrm{mom}$ for $s=\bar h\cdot q$ under
   each edit. $R_{pT}=(\Delta s/\Delta\log\mathrm{mom})_{pT}/(\Delta s/\Delta\log\mathrm{mom})_{\phi}$ is 0 for $M_2$ and 1 for mom.
2. Regress $s$ on $\log S_0$ and $\log(S_2/S_0)$ across real jets.
3. Write $s$ from real donors with different mom: $h_p\leftarrow h_p+(\bar h_d-\bar h_r)qq^\top$. Compare
   $\mathrm{d}\Delta z_{97}/\mathrm{d}\Delta\log\mathrm{mom}$ with the jet's own φ interchange.

**Result.** At block 1:
- $R_{pT}=0.71$ (the network output gives 0.74); the natural regression ratio is 1.02.
- Donor substitution follows Δlog mom (r = 0.87) and gives 0.42 of the edit-calibrated slope. Random directions give about 0.
- P8's donors held mom fixed (|Δlog mom| = 0.022), which explains its null.

**Checks.**
- Fresh files (`_102`); unedited accuracy 85.93 %.
- P8's patching and DAS numbers replicate within 0.03.
- Three random directions per site.
- Holm over the sites.

**Why it matters.** The thesis step "ParT forms an axis-moment variable" survives in a linear jet-level form, though not as an exact
copy. What fails is the "forms at block 1–3" part: the variable is already readable from the input embedding.
