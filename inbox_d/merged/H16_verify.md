# H16v: skeptic check of H16 (boost sensitivity, training dataset vs architecture/inputs; claim D9)

Job: d-h16-verify (CPU 2 / 6 Gi, conda, exit 0, 17 min; peak 3.7 GiB, 1.75 cores). Script: `interp/h16_verify.py`.
Results: `results/physics/h16_verify.json` (also on the volume), log `logs/d-h16-verify.log`. No figure was made; the H16 figure was read.

## Conclusion
On a fresh sample, the same small top-vs-quark/gluon taggers again change 4–7 % of their calls under a 10 % mass-preserving boost, against under 1 % for a matched rotation. Training on hls4ml or on Top Landscape changes this fraction by about 1 percentage point or less for pT-fraction inputs. With absolute-pT inputs, the training sample flips the direction of the change: hls4ml-trained nets move jets toward "top" when pT is lowered, and Top-trained nets move them toward "QCD". The H16 statement that the nets read moments about the jet axis (rather than the invariant mass) is not supported. A new control holds that moment fixed and moves the jet off its axis by the same amount, and it changes about as many calls. Most of H12's 21–38 % came from W and Z jets: on top, quark and gluon jets, the same H12 checkpoints change their "top" call on 4–11 % of jets.

## What was checked
1. **Code review** of `interp/h16_boost_dataset.py` and `interp/h16_train.py`.
   - Rows are random (seed 16 for training, 4161 for the battery), never first-k.
   - Padded slots are masked: the features are zeroed and masked in `features`/`Net`, and `rel`/`set_rel` keep the padding at 0.
   - The inputs are the 64 hardest particles after re-sorting on edited pT, measured about the edited stored axis, which is the same convention as training.
   - The working point is the threshold at ε_S = 0.5 on the unedited jets of the same sample.
   - Top inputs use the η sign of the unedited axis, as in the conversion. Round trip: centroid ≤ 8e-5.
   - The hls4ml q/g rows are drawn from q ∪ g. W/Z were excluded from validation after the 2026-10-20 fix.
   - No bug that changes a number was found.
2. **Independent boost.** An explicit 4×4 Lorentz matrix along the jet 3-momentum, written separately from `h12_boost_panel.boost`.
   - Per particle it agrees with h12.boost to 8e-13 (pT, η) and 1e-15 (φ); the axis agrees to 7e-16; |P′|/|P| = λ to 3e-15.
   - The battery used this independent boost.
3. **Fresh sample.** Seed 9917, random rows, 5,000 t + 5,000 bkg on each of hls_all, hls_win and top. Each of the 30 nets was scored on its native sample.
4. **New controls for the M edit** (k = 1/0.9):
   - **TC.** Shrink the particle positions about the pT centroid by a per-jet factor c, then apply the same translation t·u as M. c is set so that m_A = √(M0·M2) about the axis is unchanged. The centroid moves exactly as in M, and the invariant m goes to ≈ c·m.
   - **AC.** The same shrink without the translation. The centroid stays fixed, and m and m_A both go to ≈ c·m.
   - Here c = 0.875 (5–95 %: 0.872–0.875; none clipped).
5. **Dataset contrasts.** Seed-mean B0.9 % changed, with a bootstrap over jets (1,000 resamples, independent samples), the range over the four seed pairs, and Holm over the 12 contrasts.
6. **H12 five-class checkpoints on the top call.** From the H12 per-jet file (seed 301, 20,000 hls4ml jets, 12,000 of them t/q/g): the % of t/q/g jets whose "top" call (argmax = t) changes under B0.9.

## Numbers (fresh sample, seed 9917)
- **Unedited accuracy.** Within 0.44 pp of test.json for all 30 nets.
  - hlsA 88.0–90.2 %, hlsB 89.6–90.1 %, Top 90.6–92.6 %.
- **Edit fidelity.**
  - B0.9: |Δm/m| ≤ 7e-13 (p99), |Δm_A/m_A| ≤ 0.13 % (hls) / 0.57 % (Top) (p99), ΔpT_J = −10.00 %.
  - M: m_A +11.11 %, with m and pT_J unchanged.
  - TC: |Δm_A/m_A| ≤ 2e-15 (p99), m −12.5 %, pT_J +0.1 to +0.4 %.
  - AC: m −12.5 %, m_A −12.5 %.
  - Median centroid offset |Σ pT r| / Σ pT: unedited 0.0017 (hls) / 0.012 (Top). After M and after TC: 0.049 (hls) / 0.124 (Top). Unchanged by RM and AC.

**% binary calls changed (s0 / s1 range over the configs on each native sample)**

| arm | sample | B0.9 | A0.9 | P0.9 | R0.9 | B1.1 | A1.1 |
|---|---|---|---|---|---|---|---|
| frac | hlsA → hls_all | 4.70–5.31 | 4.75–5.37 | 0 | 0.36–0.72 | 4.00–5.06 | 4.03–5.11 |
| frac | hlsB → hls_win | 4.09–5.82 | 4.13–5.88 | 0 | 0.33–0.66 | 3.75–4.40 | 3.76–4.38 |
| frac | top → top | 3.85–4.76 | 3.84–4.87 | 0 | 0.43–0.87 | 4.76–5.44 | 4.80–5.67 |
| abs | hlsA → hls_all | 4.33–7.48 | 3.77–4.48 | 1.25–5.43 | 0.49–0.71 | 4.34–9.05 | 3.69–4.03 |
| abs | top → top | 3.66–5.48 | 4.11–5.68 | 3.52–4.68 | 0.40–0.86 | 2.96–3.59 | 4.98–5.68 |

- **B vs R.** B changes more calls in all 30 nets. McNemar, Holm over 30: p ≤ 2e-53.
- **B vs A, frac nets.** |B − A| ≤ 0.11 pp on hls and ≤ 0.26 pp on Top. This reproduces H16 (B/A 0.95–1.00).
- **Direction of the changed calls at B0.9 (→ top / → bkg, % of jets).**
  - hlsA-frac: 4.1–4.5 / 0.5–1.2.
  - hlsB-frac: 3.4–3.8 / 0.6–2.0.
  - top-frac: 2.7–3.4 / 1.1–1.9.
  - **hlsA-abs: 4.0–6.6 / 0.3–0.9.**
  - **top-abs: 1.0–1.5 / 2.5–4.0.**
  - At B1.1, top-abs gives 1.0–1.5 / 1.8–2.2.
  - With absolute-pT inputs the sign of the response to the boost depends on the training sample. This matches H16's ε_S shift: hlsA-abs 0.50 → 0.54–0.58, top-abs 0.50 → 0.26–0.39.

**Dataset contrasts, seed-mean B0.9 (pp), bootstrap 95 % CI, Holm over 12; [seed-pair range]**

| contrast | jedis | tf | pfn |
|---|---|---|---|
| frac hlsA − top | +0.06 [−0.46, +0.58], 0.84 [−0.06, +0.17] | +1.17 [+0.69, +1.65], 2e-5 [+0.97, +1.36] | +0.55 [+0.04, +1.08], 0.22 [+0.44, +0.66] |
| frac hlsB − top | −0.48 [−0.98, 0.00], 0.22 | +0.51 [+0.05, +0.96], 0.22 | +0.86 [+0.31, +1.39], 0.019 |
| frac hlsA − hlsB | +0.54 [+0.02, +1.03], 0.22 | +0.65 [+0.14, +1.15], 0.089 | −0.31 [−0.84, +0.26], 0.82 |
| abs hlsA − top | +0.75 [+0.18, +1.30], 0.089 | +0.26 [−0.25, +0.76], 0.82 | +3.56 [+3.00, +4.10], 2e-34 |

- Every frac contrast is within 1.2 pp. That reproduces H16, which reported −0.50 to +0.88 pp.
- The abs-pfn contrast reproduces (H16: +2.97 pp).

**M edit and the new controls (k = 1/0.9), % calls changed**

| sample (nets) | M | RM | TC (m_A held, centroid moved as M) | AC (m, m_A × 0.875, centroid fixed) |
|---|---|---|---|---|
| hls_all (12 hlsA) | 6.69–9.04 | 1.65–2.77 | 4.86–9.39 | 5.44–7.30 |
| hls_win (6 hlsB) | 6.10–8.86 | 1.31–2.20 | 4.43–9.76 | 5.20–6.49 |
| top (12 top) | 7.71–12.05 | 1.91–2.59 | 6.07–13.84 | 7.36–9.23 |

- **TC vs RM.** TC changes more calls in all 30 nets (McNemar, Holm over 30: p ≤ 3e-38). Moving the jet off its axis changes calls whether or not m_A changes.
- **TC vs AC.**
  - All 10 transformer nets: TC > AC (+1.2 to +5.4 pp, Holm ≤ 1.1e-3).
  - jedis and PFN: TC − AC from −3.0 to +1.0 pp; 9 of 20 Holm < 0.05, all 9 with TC < AC.
- **M vs A0.9.** M > A0.9 in all 30 nets (Holm ≤ 2e-9). A0.9 moves m_A by the same +11 % but does not move the centroid.

**H12 five-class hls4ml checkpoints, B0.9 (seed 301, H12 per-jet file)**

| model | five-class calls changed, all jets | five-class, t/q/g jets | "top" call changed, t/q/g jets [95 % CI] |
|---|---|---|---|
| JEDI-linear n64 / n32 / n16 | 32.5 / 37.7 / 38.4 | 17.4 / 18.3 / 20.0 | 10.7 [10.1, 11.2] / 9.5 / 8.5 |
| dsets / linf / tf_d64 / jedis | 25.9 / 36.6 / 36.9 / 34.2 | 14.1 / 17.6 / 16.0 / 13.0 | 8.9 / 9.8 / 8.6 / 7.7 [7.2, 8.2] |
| h12 pT-normalised retrain (2 seeds) | 31.2 / 33.1 | 10.7 / 11.0 | 5.7 / 6.1 |
| h12 pT-flattened retrain (2 seeds) | 21.1 / 25.0 | 8.4 / 9.0 | 3.6 / 3.6 |
| EFN | 36.8 | 15.0 | 5.4 |

- Of the five-class calls that change, 67–77 % are on W or Z jets.
- On t/q/g jets the "top" call changes for 7.7–10.7 % of jets in the original H12 models. The H16 binary hlsA nets give 4.3–7.5 %.
- The "top" call here also counts t ↔ W/Z changes on top jets, so it is not the same decision as the binary taggers'.

## Statistics: hypotheses and outcomes
- **H0 (architecture/inputs) vs H1 (dataset), on the fraction of calls changed.**
  - Prediction of H1: Top-trained small nets near R level, with B ≪ P and B ≪ A.
  - Measured: Top-trained nets give B0.9 3.7–5.5 % vs R0.9 0.4–0.9 % (Holm ≤ 2e-53), and B ≈ A for frac inputs.
  - H1 is excluded for the fraction of calls changed. H16's second reason ("the hls nets are not at 20–35 %") is not a valid exclusion criterion, because those levels came from the five-class task.
- **H1′ (dataset sets the direction of the response, abs inputs).**
  - Prediction: opposite-signed flow for hlsA-abs vs top-abs nets.
  - Measured: → top / → bkg = 4.0–6.6 / 0.3–0.9 % (hlsA) vs 1.0–1.5 / 2.5–4.0 % (top).
  - Not excluded; this is a dataset effect that the H16 Conclusion omits.
- **H_mA (nets read m_A about the axis) vs H_pos (nets read positions relative to the axis or centroid) vs H_m (nets read invariant m / pair geometry only).**
  - Predictions:
    - H_mA: TC ≈ RM.
    - H_pos: TC ≫ RM and TC ≥ AC.
    - H_m: M ≈ RM and TC ≈ AC.
  - Measured: M ≫ RM, which excludes H_m. TC ≫ RM in 30 of 30 nets, which excludes H_mA as the sole reading.
  - TC vs AC is architecture-dependent: TC > AC in the transformers, TC < AC in jedis/PFN. The data are consistent with a response to both the angular size and the off-axis position, in proportions that differ by architecture.

## Issues in H16
1. **The M reading.** The report says "every net therefore responds to moments about the given axis rather than to the invariant mass", inferred from M > A.
   - The inference does not hold: A moves m_A by the same factor as M, so M − A is the effect of the centroid shift, not of m_A.
   - The TC control (m_A held, same shift) changes 4.4–13.8 % of calls.
   - What is supported: none of the nets is invariant under a rigid translation of all particles relative to the axis (a translation that keeps m). Which axis-relative quantity they read is not determined.
2. **The abs-arm direction.** The Conclusion says "the training sample makes almost no difference". That holds for the fraction of calls changed. For absolute-pT inputs the training sample reverses the direction of the response (signal efficiency at fixed threshold 0.50 → 0.54–0.58 for hlsA vs 0.26–0.39 for Top), so the nets read pT differently.
3. **The hlsA vs hlsB contrast is weak.** 65 % of the hls4ml t + q/g pool already lies inside the ±8 % window (16–84 % of pT_J: 953–1151 GeV). "Narrowing the window gives no effect" therefore tests a modest narrowing.
4. **Contrast statistics ignore seed variance.** The binomial-SE normal approximation leaves out seed-to-seed variance, which is of the same size as the frac contrasts (seed spread up to 0.85 pp vs contrasts ≤ 1.2 pp). With 2 seeds, the Holm-significant frac contrasts (in H16 and here) cannot be separated from seed variation. abs-pfn (+3.0 to +3.6 pp, seed pairs +3.3 to +3.8) is outside it.
5. **H1 exclusion criterion.** H16 uses "B/P ≤ 0.5 not met" and "hls nets not at 20–35 %". The second is not a test, because of the task mismatch (the report notes this). The valid exclusion is B ≫ R together with B ≈ A for the Top-trained nets.
6. **The task-effect part of D9 can now be partly quantified** (item 6 above): 67–77 % of H12's changed five-class calls are on W/Z jets. On t/q/g jets the "top" call changes for 7.7–10.7 % (original H12 models), 1.2–2× the H16 binary small nets.
7. **No bug was found** in the edits, the axis and coordinate conventions, padding, row sampling or the working point. All values reproduce on the fresh sample.

## Corrected reading
- Small binary top taggers trained on hls4ml (full spectrum or ±8 % window) or on Top Landscape change 3.7–7.5 % of their calls under B0.9, against 0.3–0.9 % for a matched rotation. This was reproduced on a fresh sample with an independently coded boost.
- For pT-fraction inputs the boost is an angle-only rescaling (B = A within 0.3 pp). The training sample changes the fraction of calls changed by ≤ 1.2 pp, which is within the range of seed variation.
- For absolute-pT inputs the fraction changed is similar, but the direction depends on the training sample:
  - hls4ml-trained nets move jets toward top when pT is lowered, with B > P and B > A.
  - Top-trained nets move them toward QCD, with B ≈ P < A.
  - abs-PFN differs by 3–3.6 pp.
- None of the nets is invariant under a translation that keeps m and pT_J (M 6–12 % vs RM 1.3–2.8 %). Holding m_A fixed during the same translation (TC) still changes 4–14 %. The response is therefore to the particles' position relative to the axis and to the angular size, not specifically to m_A.
- 67–77 % of H12's five-class changes are on W/Z jets. On t/q/g jets the H12 checkpoints change their top call on 3.6–10.7 % of jets.

## Explainer entry (for site/explain.py, H16v)
- **Question.** Does the H16 result (the boost sensitivity of small taggers does not depend on the training sample) survive a fresh sample, an independent boost and a control that separates the axis moment from the jet's position?
- **Physics background.**
  - A boost along the jet momentum keeps $m^2=E^2-|\vec P|^2$ and scales the angles by $\approx 1/\lambda$.
  - The axis moment is $m_A=\sqrt{M_0M_2}$, with $M_0=\sum p_{T,i}$ and $M_2=\sum p_{T,i}\,r_i^2$, where $r_i$ is the particle's distance from the axis.
  - For massless particles at small angles, $m^2 \approx M_0M_2-|\vec S|^2$, with $\vec S=\sum p_{T,i}\vec r_i$. A translation $\vec r_i\to\vec r_i+t\hat u$ keeps $m$ and changes $m_A$ and the centroid $\vec S/M_0$.
- **Method.**
  - Same 30 nets; 10,000 fresh jets per sample (seed 9917).
  - Boost $B$ coded as an explicit Lorentz matrix $\Lambda(\zeta\hat n)$ with $\zeta=\operatorname{asinh}(\lambda|\vec P|/m)-\operatorname{asinh}(|\vec P|/m)$.
  - New control TC: $\vec r_i\to \bar{\vec r}+c(\vec r_i-\bar{\vec r})+t\hat u$, with $c^2=1-(k^2-1)M_2/(M_2-M_0|\bar{\vec r}|^2)$, so that $m_A$ is fixed and the centroid moves as in M.
  - AC: the same shrink without the translation.
  - Metrics: McNemar with Holm correction; bootstrap over jets for the dataset contrasts.
- **Result.**
  - $B_{0.9}$ changes 3.7–7.5 % of calls (rotation 0.3–0.9 %).
  - Frac-input nets differ by dataset by ≤ 1.2 pp.
  - Abs-input nets flip in opposite directions depending on the training sample.
  - TC changes 4–14 % of calls (RM 1–3 %).
- **Checks.**
  - The independent boost agrees with H12's to $10^{-12}$.
  - Accuracies are within 0.44 pp of the test values.
  - TC keeps $m_A$ to $2\times10^{-15}$.
- **Physical significance.**
  - These small taggers are not boost-invariant whatever they were trained on.
  - With absolute pT they learn a sample-dependent use of the jet pT.
  - They respond to where the particles sit relative to the axis, not to a single mass variable.
