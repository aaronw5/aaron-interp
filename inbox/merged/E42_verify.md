# E42 verify: W-pair edit on JEDI-linear top jets, fresh sample and a side-effect reference

Script `interp/e42_verify.py` (imports the E42 / H4 edit code `h4_verify.edit_set`, `h4_top_wmass`, `h29_jedi_wpair_rebuild` and the E42 statistics helpers unchanged). Jobs: `anrunw-ai-e42-vsmoke` (150 jets per class), `anrunw-ai-e42-verify` (AI_PY=conda edit part, then the jax env; 1 CPU / 2 GiB; wall 956 s; exit 0). Results: `results/jedi/e42_verify.json` (smoke: `e42_verify_smoke.json`). Log: `logs/e42-verify.log` on the volume. The per-jet inputs stayed in the pod's /tmp.

## Conclusion

On a fresh set of real top jets, moving the W-candidate subjet pair to 60 or 100 GeV at fixed jet mass and three-prong shape again lowers JEDI-linear's top score slightly more than the control edit (by about 0.2–0.35 in log-odds). E42 found the same. A reference classifier that is never given the W-pair mass, only the jet mass, soft-drop mass, N-subjettiness values and particle count, shows a difference of the same size under the same edits. The reason is that the W edit also shrinks the subjets and lowers tau21, and the control edit does not do this. So this measurement cannot attribute the network's response to the W-pair mass. The earlier verified H29 result on the same network and edit says the same. One small hint remains open: moving off-peak W pairs onto 80 GeV raises the score slightly (+0.13, p 0.08), in the direction a W-mass reader would predict, but it is not significant.

## What was checked in the code (interp/e42_top_wpair_jedi.py)

- **Stages, masks, padding.** Scores come from `jedi_core.Jedi.run` (the logits). A particle counts as real only when pT > 0. Padding is asserted to be contiguous. Every variant is rebuilt about its own E-scheme axis (the axis is the sum of the massless four-vectors of the real particles). The identity rebuild matches the stored inputs to 6e-5. Accuracy is 80.81 % on the stored inputs and 80.59 % recentred (N = 13,845). No bug was found that changes the reported numbers.
- **Holds.** The S and z overrides go through `run_jetwise`. The identity hold reproduces the score exactly (max |d| 0). The random-direction null has a negative mean (−0.94 for S), so its p-values are lenient. E42 states this.
- **Calibration choice.** k_DIL is the network's response to a real total-mass change (all angles scaled by λ) per unit of the reference classifier's response to the same change. E42 switched from the through-origin slope to the antisymmetric pair ratio after the smoke run. On the fresh sample the three estimators give 2.54 (antisymmetric, [2.18, 3.08]), 1.12 (through-origin) and 1.33 (even part). R_WN is the network's W − N divided by k_DIL times the reference's W − N. It therefore spans 0.06–0.13 at T60 and 0.10–0.23 at T100 depending on the estimator. E42's "5–9 %" depends on this post-smoke choice.
- **The control does not match the W edit's side effects.** This is the main issue.
  - The W edit restores tau32 by scaling every subjet's internal spread by a common factor h. The N edit hardly needs to.
  - Fresh sample, median h at T60: 0.78 for W, 1.00 for N. At T100: 0.94 for W, 1.04 for N.
  - Median tau21 at T60 (tau21 = tau2/tau1, the two-prong vs one-prong N-subjettiness ratio): unedited 0.464, after W 0.363, after N 0.430.
  - E42's own C2ST shows the same thing. C2ST is a classifier two-sample test: an AUC of 0.5 means the edited jets cannot be told apart from natural ones. On features that exclude every pair mass (m, mSD, tau1, tau21, tau32, n), the AUC is 0.79 for W60 jets and 0.61 for N jets.
  - So W − N measures mW together with subjet shrinkage and tau21. E42 has no reference that sees these side effects without seeing mW.
  - The earlier H29 / H29v runs (inbox_d/merged/H29_verify.md) used the same network and the same rebuilt edit. They found JEDI-linear's W − N inside the range of five such side-effect MLPs (MLPx) at 60, 70 and 100 GeV, and of the wrong sign at 90 GeV. E42 does not cite them.

## Procedure of this verification

- **Sample.** `jedi_core.sample(3000, 4302)`. The 1,155 jets whose full_test row was also drawn by E42's samples (seeds 4201, 4202; 2,000 per class) were removed, leaving N = 13,845.
- **Top window.** Same as E42: 105 < mSD < 210 GeV, with mSD the soft-drop mass. 1,631 eligible tops had 65 < mW < 95 GeV, where mW is the mass of the exclusive-kT subjet pair closest to 80.4 GeV.
- **Edits.** Exactly E42's. Each edit moves whole subjets rigidly in (η, φ). The pT of every particle and the subjet masses are unchanged. Jet mass is restored to within 0.5 GeV and tau32 to within 0.02.
  - **W⟨t⟩:** the opening of the W pair is rescaled so that mW = t.
  - **N⟨t⟩:** the pair (harder W daughter, third subjet) gets the same relative mass change, and the W pair is left in place.
  - **R⟨t⟩:** the whole jet is rotated about its axis.
- **Sets.**
  - T60 and T100 (replication).
  - **T80 (new).** Tops with 6 < |mW0 − 80.4| < 15 GeV, with the W pair moved onto 80.4 GeV (accepted if |mW − 80.4| < 2). n = 317 / 211 / 145.
- **Score and statistics.**
  - Score: top log-odds s = L_t − logsumexp(the other four logits). The response to edit v is Δ_v = s_v − s_unedited. W − N is the per-jet ΔW − ΔN.
  - Intervals: 95 % jet bootstrap (2,000 draws).
  - H0: mean W − N = 0. Tested with a two-sided sign-flip permutation (10,000 draws), Holm-corrected over the 3 sets.
  - McNemar (exact binomial on the jets whose top call differs) compares the calls after W and after N.
- **Reference classifiers.** Each is an MLP 32-32 trained on t vs q/g, using 25,000 + 25,000 train-archive jets (dev split excluded, rng 4205, as E42).
  - **REF:** (m, tau32, mW). E42's mW reader.
  - **MLPx:** (m, mSD, tau1, tau2, tau3, tau21, tau32, n). It sees no pair mass, only the side effects of the edits.
  - **MLPxW:** MLPx plus mW.
  - The test AUCs are 0.913, 0.951 and 0.953. Each reference is calibrated to the network with its own k_DIL.
- **Side-effect regression.** Per jet, (ΔW − ΔN) is regressed on the W − N differences of these input sums: Σr, Σr², ΣpT·r, Σ over ranks ≥ 1 of (r − 0.2)₊, log ΣpT·r², tau21, tau1 and mSD. Here r is the particle's angular distance from the jet axis and rank is its pT order. The covariates are standardised. The intercept, with a bootstrap CI, is the part of W − N not linearly accounted for by these changes. This is descriptive, not causal: the covariates move together with mW.
- **Other checks.** z reads (z0, z7, z10, z14) and single-unit S holds (S58, S60, S13) at T60, T100 and T80, as E42. Three untrained redraws.

## Results (fresh sample, seed 4302)

### Network responses

| set | n | ΔW | ΔN | ΔR | W − N [95 % CI] | p perm (Holm, 3) | top calls unedited / W / N; McNemar p |
|---|---|---|---|---|---|---|---|
| T60 | 317 | −0.700 | −0.470 | −0.009 | −0.230 [−0.373, −0.094] | 0.0018 (0.0036) | 0.984 / 0.953 / 0.962; p 0.51 |
| T100 | 211 | −0.299 | +0.047 | −0.042 | −0.346 [−0.539, −0.166] | 0.0006 (0.0018) | 0.981 / 0.981 / 0.976; p 1 |
| T80 | 145 | +0.060 | −0.070 | +0.015 | +0.131 [−0.012, +0.274] | 0.082 (0.082) | 0.924 / 0.952 / 0.945; p 1 |

The E42 T60 / T100 numbers replicate. E42 reported −0.178 / −0.293 on its test sample and −0.386 / −0.325 on its fit sample. Untrained redraws: |W − N| ≤ 0.012.

### References

Each cell gives the reference's own W − N, then the network's W − N divided by k_DIL × the reference's W − N (antisymmetric k, with its bootstrap CI). A ratio of 1 means the network responds like that reference, scaled to the network's own mass sensitivity.

| set | REF (mW reader) | MLPx (no pair mass) | MLPxW |
|---|---|---|---|
| T60 | −1.55; 0.058 [0.024, 0.094] | −0.276; 0.47 [0.19, 0.78] | −1.26; 0.080 [0.033, 0.131] |
| T100 | −1.36; 0.100 [0.049, 0.156] | −0.141; 1.38 [0.71, 3.04] | −0.93; 0.164 [0.083, 0.247] |
| T80 | +0.40; 0.13 [−0.01, 0.27] | −0.022 [−0.067, +0.024]; – | +0.127; 0.45 [−0.06, 1.26] |

MLPx is never given a pair mass, yet its unscaled W − N (−0.28 at 60 GeV, −0.14 at 100 GeV) is about the size of the network's (−0.23 and −0.35).

### Side-effect regression

| set | intercept [95 % CI] | raw mean W − N | R² |
|---|---|---|---|
| T60 | +0.006 [−0.140, +0.157] | −0.230 | 0.27 |
| T100 | −0.141 [−0.322, +0.025] | −0.346 | 0.39 |
| T80 | −0.009 [−0.156, +0.133] | +0.131 | 0.30 |

### Holds

Each hold fixes one second-round object S at its unedited value during the edit. The share is the fraction of the W response it removes.

- **T60:**
  - S58: share of the W60 response 0.84 [0.76, 0.91]. E42 reported 0.84 / 0.83. With S58 held, W − N becomes +0.34 [+0.23, +0.45], i.e. the sign flips.
  - S60: 0.17 [0.12, 0.23].
  - S13: 0.03 [0.00, 0.07].
- **T100:** S58 0.15 [−0.14, +0.34]; S60 0.38 [0.18, 0.60].

### z reads

z units are the 16 last-hidden-layer neurons. These are read-outs, not interventions.

- At T60, z0 W − N is −0.176 [−0.262, −0.093], Holm 0.0012 over 12 tests. E42 reported −0.187.
- z10 W − N: −0.066 [−0.145, +0.015] at T60 and −0.017 at T100.

## Hypotheses

- **H0: W − N = 0.** Rejected at T60 and T100 (Holm 0.0036, 0.0018). Not rejected at T80 (p 0.082).
- **Hm: the network reads mW like a reference classifier, scaled to its own mass response.** Predicted W − N (MLPxW-scaled): −2.9 at T60, −2.1 at T100, +0.29 at T80.
  - Excluded at T60 and T100: the ratios are 0.08 and 0.16, and their CIs lie far below 1.
  - E42 reached the same exclusion with REF.
- **Hs: the network responds only to the edits' side effects (subjet shrinkage, tau21, radial sums) and not to mW.** Predicted (MLPx-scaled): −0.49 at T60, −0.25 at T100, −0.04 at T80.
  - At T100 it is not excluded (ratio 1.38 [0.71, 3.04]).
  - At T60 the network responds less than the MLPx scaling predicts (0.47 [0.19, 0.78]), so it does not reach even the side-effect reference's size.
  - At T80 the network's +0.13 is above Hs's −0.04 by 0.17. This is about 2 SE and is not significant on its own.
  - The linear side-effect covariates account for the whole T60 W − N (intercept +0.006).
- **H_weak: a small mW term at about 5–10 % of REF on top of the side effects (E42's reading).**
  - At T60 and T100 it cannot be separated from Hs with this design.
  - The T80 sign points weakly toward it (p 0.08).
- **Excluded by the data:** Hm. **Not excluded:** Hs and H_weak. E42's "W-pair-specific term" assumes that Hs is excluded, and it is not.

## Assessment of the E42 reading

- **Correct and reproduced.**
  - The sign and size of W − N at T60 and T100 at fixed m, tau32 and axis.
  - No measurable change in the top calls.
  - The untrained null.
  - S58 removing about 84 % of the W60 response, and holding S58 flipping the sign of W − N.
  - The z0 read.
  - W − N lying far below a calibrated mW reader.
- **Not supported: "W-pair-specific" and "a small W-pair-specific term remains … at R_WN 0.05–0.09".**
  - The N control does not reproduce the W edit's subjet shrinkage (h 0.78 vs 1.00) or its tau21 drop.
  - A classifier with no pair-mass input shows a W − N of the same size.
  - H29 / H29v had already shown this for the same network and edit.
  - R_WN also depends by a factor of about 2 on which k_DIL estimator is used.
- **Scope.** "Does not go through z10" and "z0 responds to it" are read-outs of a difference that is not specific to the W pair. The S58 routing is target-dependent: 0.84 at 60 GeV, but 0.15 at 100 GeV, where S60 carries 0.38.
- **Figure (`e42_wpair.png`).**
  - Panel b compares the network only with the scaled mW reader (−3.8 / −3.5 in the title). No side-effect reference is shown, so the panel cannot show that an mW-blind classifier gives a W − N of the same size.
  - Panel a is labelled correctly but does not show the N control.
  - Panel c is readable and matches the JSON.

## Corrected reading

At fixed jet mass, tau32 and axis, moving the W-candidate pair mass of real top jets to 60 or 100 GeV lowers JEDI-linear's top log-odds by 0.18–0.39 more than the non-W control edit. This replicates on a fresh sample: −0.23 [−0.37, −0.09] at 60 GeV and −0.35 [−0.54, −0.17] at 100 GeV. No top call changes measurably. That excess is about the size of the response of an MLP that is never given the pair mass (MLPx: −0.28, −0.14). The W edit also shrinks the subjets (h 0.78 vs 1.00) and lowers tau21, and the control edit does neither. So the difference is not attributable to the W-pair mass. It is 6–16 % (antisymmetric k) of the response of a calibrated mW reader, which excludes mW reading at the reference's strength. Moving off-peak W pairs onto 80.4 GeV gives +0.13 [−0.01, +0.27] (p 0.08), in the mW-reader direction, where MLPx gives −0.02. This is the only result left that could separate a weak mW term from the side effects. Holding S58 removes 84 % of the 60 GeV response, but S58 carries the W and the control responses together.

## Suggested next control (not run)

A matched-side-effect control: an N edit that applies the same internal subjet scale h as the W edit, or a W edit with h fixed at 1 and tau32 left free. A larger T80 set would also help (e.g. sample(6000, ·) gives about 300 jets).

## Glossary additions

- **MLPx / MLPxW.** Reference t-vs-q/g MLPs. MLPx uses (m, mSD, tau1, tau2, tau3, tau21, tau32, n) and no pair mass. MLPxW uses the same features plus mW.
- **h (subjet-size factor).** The common factor that scales every subjet's internal angular spread in the H4 edit so that tau32 is restored.
- **T80.** Top jets whose W-pair mass is 6–15 GeV from 80.4 GeV, edited onto 80.4 GeV.
