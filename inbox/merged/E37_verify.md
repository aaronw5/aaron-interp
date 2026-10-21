# E37 verification: learned interchange direction for the prong-sharing edit, re-run on fresh jets with added controls

Verifier for node E37. Script `interp/e37_verify.py` (parts a, b, c; it reuses E37's own functions from `interp/e37_sharing_das_jedi.py` and the stored E37 subspaces `results/jedi/e37_das_Q.npz` and `e37_das_M.npz`).
Jobs: e37-verify (part a, 1 CPU / 2 GiB, wall 217 s, peak 1,964 MiB), e37-verify-b (part b, 505 s, peak 2,047 MiB), e37-verify-c (part c, 482 s, peak 2,047 MiB). Parts b and c reached the 2 GiB limit and still finished. Their CPU use was 0.06–0.08 cores because about 450 s of each run went to start-up (matplotlib font cache, jax plugin probe).
Results: `results/jedi/e37_verify.json`, `e37_verify_b.json`, `e37_verify_c.json`; logs `/j-jepa-vol/aaron-interp/logs/e37-verify*.log`.

## Conclusion
On new jets, one learned direction in JEDI-linear's fourth layer reproduces about 0.85 of the average W-versus-Z change caused by a more lopsided two-prong momentum split. That direction carries almost none of a real mass change, so the average-size result and the mass-specificity result reproduce. Two parts of E37's reading do not hold. First, adding a fixed amount along this direction to unedited jets pushes W jets toward Z and Z jets toward W, and reversing the sign reverses both. That is the edit's class-opposite pattern, and a fixed offset along a random direction produces it less often (p 0.01). The claim that the direction "only works when taken from the jet itself" came from a different, two-dimensional direction. Second, "carries most of the response" holds for copying the direction in, not for removing it: holding the direction at its unedited value removes about half of the W-jet response and none of the Z-jet response. A plain difference-of-means direction reproduces the average change just as well, and the network's mass subspace also reproduces much of the Z-jet part.

## What was checked
**Code review** of `e37_sharing_das_jedi.py` and `e37_plot.py`:
- Stages: `JNet.exact` is the fixed-point tail from S, h4 and h5, with the next quantizer's range clip. It reproduces `Jedi.run` exactly, as `tail_check` shows.
- Padding: `to_p4` builds four-vectors only for slots with pT > 0. `model_xn` zeroes the padded slots after re-sorting by pT.
- Split: test jets exclude the fit rows. Model selection uses 20 % of the fit jets. The read-out weights w are fit on fit pairs.
- Edit: D keeps the jet four-momentum fixed to about 4e-13 in relative terms, again on fresh jets.
- Bootstrap: resampling is over jets, with each jet's pairs kept together.

No coding bug was found that changes the numbers. Two scoping problems were found in the report:
1. The steering test (adding a fixed offset to h4) was run only along the h4 k = 2 read-out direction. E37's own read-out showed that direction is not monotonic in dc. The reading then attributes the steering result to "the direction" that carries 0.83, which is the h4 k = 1 direction.
2. In the recurrence panel (c) of `results/jedi/e37_das.png`, the untrained networks are drawn in units of their own margin s.d. That s.d. is 0.002–0.12 log-odds, so untrained1 appears at about −1.4 s.d. In absolute terms its change is under 0.003 log-odds. The axis label gives this range. The report text quotes 0.002–0.027 instead, which is inconsistent.

**Fresh sample.**
- Fit set F′: W+Z jets of `sample(1500, 3801)`, 2,757 jets.
- Test set T′: W+Z jets of `sample(1500, 3802)`, 2,724 jets.
- Every row in E37's fit or test sample (seeds 3701 and 3702) was removed from both sets, and the F′ rows were removed from T′.
- All-class accuracy on stored inputs of 3802: 80.37 %. That is 0.2 points below the 80.6–81.3 % band; the binomial s.e. is 0.46 %.

**Edits** (same code as E37):
- D(dc): every constituent is boosted to the jet rest frame, rotated so that |cos θ*| changes by dc, and boosted back. |cos θ*| is the cosine between the two-prong kT axis and the jet's flight direction; dc > 0 means a more lopsided momentum split. Jet four-momentum, both prong masses and N are held fixed.
- cn: a rotation by the same angle about the prong axis. The sharing is unchanged.
- dil(f): every constituent angle about the jet axis is multiplied by f = 0.95 or 1.05. This is a real mass change.

**Definitions.**
- Margin: W logit minus Z logit.
- Interchange (DAS patch): $h' = h_b + QQ^\top(h_s-h_b)$ at h4. The unedited jet's coordinates inside subspace Q are replaced by those of the edited jet, and the exact tail is run.
- Hold: $h' = h_s + QQ^\top(h_b-h_s)$. The edited jet's coordinates in Q are set back to their unedited values. "Removed" = 1 − (held change / full change), per class.
- frac: the share of the average margin change reproduced by the patch, pooled over (dc, class) cells with the sign of each cell's target.
- IIA (interchange intervention accuracy): among pairs whose W/Z call the edit changes, the share where the patched call equals the edited call.
- Steering: on unedited jets, add c × 2.87 × v to h4. Here v is the stored h4 k = 1 direction, and 2.87 is the mean |Δu| under D(+0.3), where u = h4 · v.
- Antisymmetric contrast: A = [(Z−W)(+c) − (Z−W)(−c)]/2, where (Z−W)(c) is the mean Z margin change minus the mean W margin change at offset c. A perturbation that only pushes jets toward the W/Z boundary gives A ≈ 0.

## Results (test T′, N 2,724 W+Z jets; 95 % intervals from 1,000 jet bootstraps)
**Edit response (reproduces E37).**

| edit | W Δmargin (log-odds) | Z Δmargin (log-odds) |
|---|---|---|
| D(+0.3) | −1.57 | +1.52 |
| cn(+0.3) | +0.04 | +0.12 |
| D(−0.3) | +0.68 | −0.09 |
| dil × 1.05 | −1.93 | −1.02 |

**(1) Stored E37 subspaces on fresh jets.** n = 1,156 pairs whose call the edit changes.

| subspace | frac [CI] | IIA [CI] | slope | dilation frac [CI] |
|---|---|---|---|---|
| h4 k = 1 | 0.85 [0.82, 0.89] | 0.37 [0.33, 0.40] | 0.55 | 0.004 [0.002, 0.007] |
| h4 k = 2 | 0.84 | 0.31 | 0.57 | 0.055 |
| h5 k = 1 | 0.69 | 0.49 | 0.52 | 0.000 |
| S k = 1 | 0.65 | 0.46 | 0.47 | −0.003 |

- 300 random live h4 subspaces: k = 1 frac mean 0.05 (q95 0.21), IIA mean 0.08 (q95 0.15). k = 2 frac mean 0.12 (q95 0.33). Rank p = 0.003, the floor for 300 draws.
- Per dc at h4 k = 1: frac 0.94 at dc +0.3 and +0.15; 0.58 and 0.57 at −0.3 and −0.15.

**(2) Refit on F′** (h4, k = 1, same code):
- Learned fit: frac 0.84 [0.81, 0.87], IIA 0.37 [0.34, 0.41], dilation frac 0.003. |cos| with the stored E37 direction 0.75.
- Shuffled-target fit: frac 0.47 [0.40, 0.54], IIA 0.49 [0.46, 0.53], |cos| 0.37.
- As in E37, IIA does not separate the learned fit from the shuffled-target fit.

**(3) Non-learned k = 1 directions at h4** (built from F′ only):

| direction | frac [CI] | IIA [CI] | dilation frac | \|cos\| with DAS |
|---|---|---|---|---|
| mean displacement (D(+0.3) minus D(−0.3)) | 0.96 [0.91, 1.00] | 0.20 [0.18, 0.23] | 0.03 | 0.63 |
| PC1 of the displacements | 0.66 [0.63, 0.69] | 0.41 [0.38, 0.45] | 0.02 | 0.55 |
| margin read-out (least squares of margin on h4) | 0.06 | 0.10 | 0.13 | 0.02 |

The average-change share is matched without learning. DAS is higher than the mean-displacement direction on IIA (0.37 vs 0.20) but not on frac.

**(4) Positive control for the mass-specificity metric.**
- On dilation pairs: the stored M2 DAS subspace (k = 2) gives dilation frac 0.88 [0.88, 0.89]; the stored dilation direction 0.88 [0.88, 0.89]; the sharing direction 0.004.
- So the metric detects a mass carrier, and the sharing direction is not one.
- The same M2 subspace also reproduces 0.55 [0.52, 0.58] of D's average change (random k = 2 q95 0.33), and the dilation direction reproduces 0.41 [0.36, 0.46] (random k = 1 q95 0.21).
- Per cell, the M2 interchange at dc +0.3 gives Z +1.77 (target +1.52) but W only −0.46 (target −1.57). At dc < 0 it gives nothing (W +0.13 vs +0.68).
- So under interchange the Z-jet part of the D(+0.3) response is reproduced through the mass subspace.

**(5) Hold (necessity) on D(+0.3) pairs; share of the D response removed.**

| held subspace | W | Z |
|---|---|---|
| DAS h4 k = 1 | 0.47 | −0.08 |
| DAS h4 k = 2 | 0.50 | 0.05 |
| M2 (k = 2) | 0.12 | 0.03 |
| dilation direction | −0.06 | 0.02 |
| random k = 1, 200 draws (q95) | 0.07 | 0.08 |
| random k = 2, 200 draws (q95) | 0.09 | 0.06 |

- Positive control on dilation pairs: holding M2 removes 0.90–1.06 of the dilation response, and holding the sharing direction removes 0.01–0.09.
- For M2, this matches E35 (holding the mass direction removes ≤ 0.08 of D's response).
- No bootstrap was run for the hold numbers; the random quantiles are the null.

**(6) Steering along the h4 k = 1 direction** (not done in E37):

| offset | W Δmargin [CI] | Z Δmargin [CI] |
|---|---|---|
| ×2.87 in D(+0.3)'s direction | −2.25 [−2.31, −2.19] | +1.49 [1.40, 1.57] |
| ×2.87 opposite to D(+0.3)'s direction | +0.48 [0.41, 0.55] | −0.95 [−1.01, −0.88] |
| D(+0.3) itself | −1.57 | +1.52 |

- Antisymmetric contrast A: 2.58 at c = 1 and 1.55 at c = 0.5.
- 200 random live h4 directions of the same length: |A| q95 1.86 and 1.07; p = 0.01 at both sizes. That is H0 "|A| along the DAS direction ≤ random" rejected; the floor for 200 draws is 0.005.
- The symmetric part S (1.16) is inside the random range (median 1.38). A large offset in any h4 direction pushes W and Z jets toward the boundary: 94.5 % of random directions give W down and Z up for one of the two signs. Only the part that flips with the sign is specific to this direction.
- Along E37's k = 2 read-out direction, E37's own result reproduces: W +0.50 and Z +0.17 per unit u, the same sign for both classes.

**(7) Recurrence** (code only; not re-run). The pattern test, Fisher test and McNemar test are coded correctly. The untrained control is weak: the redrawn networks give nearly constant outputs (W+Z accuracy 0 or 0.5; two redraws exactly constant), so "0/10 untrained" shows that the response needs training, not that it is W/Z-specific. A shuffled-label network would be the matched control.

## Competing hypotheses
- **H1, a signed one-dimensional sharing coordinate at h4.** Predicts: a large interchange frac; an antisymmetric steering response with D's sign; a hold that removes most of the response; no dilation transfer.
  - Observed: frac 0.85, A p = 0.01 and no dilation transfer agree.
  - The hold removes 0.47 (W) and none (Z), so H1 holds only partially.
- **H2, the response is generic h4 perturbation toward the boundary.** Predicts A ≈ 0 and random directions as good. Excluded for the antisymmetric part (p 0.01); it does describe the symmetric part.
- **H3, E37's "not additive, works only with the jet's own coordinates."** Predicts that a fixed offset gives same-sign W/Z changes. Excluded for the k = 1 direction; it was observed only for the k = 2 read-out direction.
- **H4, the Z part of the response also runs through the mass subspace.** Supported under interchange (M2 reproduces Z +1.77 of +1.52). Not supported under hold (0.03). The Z response is therefore carried redundantly, or the h4-to-margin map is not additive across subspaces.

## Corrected reading
- At h4, the learned one-dimensional DAS direction, copied from the edited jet, reproduces 0.83–0.85 of the average W−Z margin change caused by D on two disjoint test samples. Random directions reproduce 0.05–0.07.
- A non-learned difference-of-means direction reproduces 0.96, so concentration of the average change in one direction is a property of D's h4 displacement. DAS adds per-jet call agreement: 0.37 vs 0.20, though no better than a shuffled-target fit (0.49).
- The direction transfers 0.004 of a real mass change, against 0.88 for the M2 subspace (positive control).
- A fixed offset along it, in D(+0.3)'s direction, moves W −2.25 and Z +1.49; the opposite offset reverses both (antisymmetric contrast vs random p 0.01). It therefore acts as a signed coordinate whose effect on the margin has opposite sign for W and Z jets.
- Holding it fixed removes 0.47 of the W response and none of the Z response. Under interchange, the mass subspace also reproduces the Z response at dc +0.3.
- The response recurs in 9/10 trained networks. The untrained comparison networks have nearly constant output.

## Explainer entry (for site/explain.py)
- **Question:** Does the direction in JEDI-linear's fourth layer found by E37 reproduce on new jets, does it act like a stand-alone coordinate, and is it needed for the response?
- **Background:** E37 found one h4 direction whose copy from an edited jet reproduces about 0.8 of the W/Z response to a more lopsided prong split. The jet four-momentum stays fixed under this edit.
- **Method:** On jets not used by E37, the following were evaluated:
  - the stored direction, and a refit with the same code;
  - non-learned directions (mean displacement, first principal component);
  - the mass subspace as a positive control;
  - holding the direction at its unedited value: $h' = h_D + QQ^\top(h_0 - h_D)$;
  - adding $\pm c\,\delta\,v$ to h4 on unedited jets, with the antisymmetric contrast $A=\tfrac12[(\Delta_Z-\Delta_W)(+c)-(\Delta_Z-\Delta_W)(-c)]$ compared against 200 random directions.
- **Result:** frac 0.85 [0.82, 0.89]; dilation transfer 0.004 (M2 subspace 0.88); steering W −2.25, Z +1.49, reversed by the opposite sign (A 2.58 vs random q95 1.86, p 0.01); hold removes 0.47 (W) and none (Z); a mean-displacement direction reproduces 0.96.
- **Checks:** fresh disjoint samples; the refit direction has |cos| 0.75 with the stored one; positive control for the specificity metric; random and shuffled-target nulls.
- **Why it matters:** the sharing response has a signed coordinate at h4 that is separate from the mass coordinate. That coordinate is needed for only part of the W response, and the Z response has other carriers.
- **Glossary additions:**
  - **hold (necessity patch):** set the edited jet's coordinates in a subspace back to their unedited values and measure the share of the response removed.
  - **antisymmetric steering contrast A:** the part of the W/Z response to a fixed offset that reverses with the offset's sign.
