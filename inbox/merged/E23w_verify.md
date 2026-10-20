# E23w verify: estimator check of the odd/even gain ratio on a fresh sample (2026-10-20)

Claims A4, A9. Node verified: E23w (`interp/e23w_gains_jedi.py`, report `inbox/E23w_report.md`).

## Conclusion
On a fresh, disjoint sample, the E23w estimator gives the same pattern: the gain of the shift-response part that is linear in the jet's centroid offset is about half to three quarters of the gain of the part quadratic in the shift. Most of that gap comes from the estimator, not the network. E23w measures the gain by regressing on the network's own response to a real mass change, and that response is noisy. When the gain is measured against the predicted change in $\log M_2$ instead, the ratio is 0.97 at h4 and about 0.87 at h5 and z. Only at the S sums is it clearly below 1 (0.77). Separately, only 2–4 % of the network's odd response to a shift follows the pT-weighted centroid (the $\vec M_1$ term); much more follows the unweighted particle centroid. So the "two gains" describe a small part of the response. Simple radial sums of the kind found earlier in S (for example $\sum r$, or $M_2$ without the hardest particle) also give ratios of about 0.7, so the shortfall at S is what such sums produce.

## Procedure
- **Script:** `interp/e23w_verify.py`. It imports the E23w edit functions (`moments`, `shift`, `scale_ang`, `cells`, `jedi_feat`, `LIVE_S`) unchanged.
- **Job:** `e23w-vfy2`, 1 CPU / 2 GiB.
  - Wall 787 s, 0.16 cores. Peak memory 2,047 MiB, at the limit with no OOM kill; the cgroup peak includes file cache from the test archive.
  - The name `e23w-verify` was already used by the worker's replicate job, so a unique name was chosen to keep that job and its log untouched.
  - `e23w-vfy` was a first attempt that stopped on an einsum subscript error in my script; its log is `logs/e23w-vfy.log`.
  - Results: `results/jedi/e23w_verify.json`; log: `logs/e23w-vfy2.log`.
- **Model:** official JEDI-linear 3-feature n64. Units as in E23w: 19 live S, 55 h4, 32 h5, 16 z, and the W−Z logit margin (K = 123), standardised by their sd on the sample.
- **Sample:** `sample(5000, 2313)` minus the 4,584 jets also drawn by seeds 2311 or 2312.
  - N 20,416; accuracy 80.99 %.
  - Tail: top 5 % of $d=|\vec M_1|/M_0$ per class, 1,024 jets, $d$ median 7.4 mrad.
  - Bulk: 1,000 random jets with $d$ below the median ($d$ median 0.52 mrad).
- **Edit (same as E23w):**
  - Rigid translation $\vec c$ of the real particles, $|\vec c|$ = 10/20/40/60 mrad at 8 azimuths.
  - Matched change: a dilation by $e^{\rm pred}$ with ${\rm pred}=\tfrac12\Delta\log(M_0M_2)$.
  - For each $(\vec c,-\vec c)$ pair: odd and even parts $O_T, E_T$ (translation) and $O_C, E_C$ (dilation).
- **Estimators:**
  - **Projection (E23w):** $a_1=\sum O_T\tilde O_C/\sum\tilde O_C^2$, where $\tilde O_C$ is $O_C$ with the fixed-direction terms ($c_\eta, c_\phi$) regressed out; $a_2=\sum E_TE_C/\sum E_C^2$.
    - $O_C$ is itself a network response. Any part of it that is not the predicted $\log M_2$ change acts as error in the regressor, which biases $a_1$ and $a_2$ toward 0 (errors-in-variables).
  - **IV (instrumented):** the instrument is the analytic predicted change.
    - Per unit, $O_T$ and $O_C$ are each regressed on $[P_1, c_\eta, c_\phi]$, with $P_1=\tfrac12({\rm pred}(\vec c)-{\rm pred}(-\vec c))$.
    - Stage gain: $a_1^{IV}=\sum_u\beta^T_u\beta^C_u/\sum_u(\beta^C_u)^2$.
    - Even: the same with $P_2=\tfrac12({\rm pred}(\vec c)+{\rm pred}(-\vec c))$. A variant adds a jet-independent $|\vec c|^2$ term.
    - A unit equal to any function of $M_0M_2$ plus response unrelated to the predicted change gives $a_1^{IV}=a_2^{IV}=1$ in expectation.
- **Statistics:**
  - Jet Poisson bootstrap, 1,000 draws (200 per ring).
  - H0: $a_1/a_2=1$ (IV). H1: $a_1/a_2\neq1$. Normal p from the bootstrap sd, Holm over the 5 stages.
  - Paired bootstrap of the difference (IV − projection) in the ratio.
- **Controls:**
  - (i) Analytic functions, no network, on the same jets and edits. Positive controls: $\log(M_0M_2)$; $\log M_2$ quantised in steps of 0.02; $\log M_2$ plus a pT-weighted half-plane ReLU sum. Radial sums of the kind found in earlier S-object fits: $\log M_2$ over ranks ≥ 1; $\log M_2$ over the 14 hardest; $\sum_{\rm rank\ge1}(r-0.2)_+$; $\sum r$; $\sum r^2$.
  - (ii) Variance of $O_T$ explained by the fixed direction alone, by $[P_1,\vec c]$, and by alternative centres: $\vec c\cdot\vec v$ for $\vec M_1$ of the 14 hardest, $\vec M_1$ of ranks ≥ 1, the unweighted centroid, and the leading particle.
  - (iii) Bulk jets.
  - (iv) 2 untrained redraws.

## Numbers (tail, N 1,024; brackets 95 % bootstrap)
| stage | projection a1/a2 (E23w estimator) | IV a1 | IV a2 | IV a1/a2 | IV a1/a2 with $\lvert c\rvert^2$ term | IV − projection | Holm p (IV ratio = 1) | $R^2$ of $O_T$ by $[P_1,\vec c]$ |
|---|---|---|---|---|---|---|---|---|
| S | 0.65 [0.50, 0.81] | 0.62 [0.51, 0.73] | 0.81 | 0.77 [0.64, 0.91] | 0.54 [0.44, 0.66] | +0.12 [0.02, 0.23] (p 0.02) | 0.003 | 0.025 |
| h4 | 0.47 [0.36, 0.58] | 0.95 [0.85, 1.06] | 0.98 | 0.97 [0.87, 1.08] | 0.90 [0.80, 1.00] | +0.50 [0.37, 0.63] (p < 0.001) | 0.63 | 0.019 |
| h5 | 0.58 [0.50, 0.66] | 0.92 [0.84, 1.01] | 1.07 | 0.86 [0.79, 0.95] | 0.93 [0.85, 1.01] | +0.28 [0.19, 0.38] (p < 0.001) | 0.002 | 0.022 |
| z | 0.77 [0.71, 0.84] | 0.96 [0.88, 1.03] | 1.10 | 0.87 [0.80, 0.94] | 0.95 [0.87, 1.02] | +0.10 [0.04, 0.16] (p 0.004) | 0.0007 | 0.042 |
| W−Z margin | 0.87 [0.73, 1.02] | 1.65 [0.76, 2.93] | 1.03 | 1.61 [0.73, 2.87] | 1.58 [0.73, 2.82] | +0.74 [−0.16, 1.95] (p 0.11) | 0.63 | 0.004 |

**Reproduction.** The projection values on the fresh sample are close to E23w's two samples: 0.51 / 0.67 at S, 0.42 / 0.61 at h4, 0.71 / 0.78 at z, and 0.79 / 0.74 at the margin. The margin interval now includes 1.

**Per ring (projection → IV):**
- At 10 mrad: S 0.22 → 0.95, h4 0.03 → 0.94, h5 0.17 → 1.04, z 0.44 → 1.10.
- At 60 mrad: S 0.75 → 0.75, h4 0.60 → 0.98, z 0.83 → 0.82.
- The rise of the projection ratio with $|\vec c|$ reported in E23w is not seen with IV, where the ratio is near 1 at small shifts. The largest projection deficit is where the odd signal is smallest compared with the noise in the dilation response.

**Analytic controls (projection / IV a1/a2):**
- Positive controls:
  - $\log(M_0M_2)$: 1.000 / 1.000.
  - Quantised $\log M_2$: 1.000 / 1.000.
  - $\log M_2$ + half-plane: 1.06 [0.39, 1.76] / 1.04 [0.87, 1.22].
  - The estimators recover 1 when the function is $M_0M_2$, including under quantisation. With an added non-mass term, the projection interval widens 3.5× compared with IV.
- Radial sums:
  - $\log M_2$ over ranks ≥ 1: 0.69 / 0.69.
  - $\sum r$: 0.71 / 0.69.
  - $\log M_2$ of the 14 hardest: 1.93 / 1.93.
  - $\sum_{\rm rank\ge1}(r-0.2)_+$: −0.33 / −1.08.
  - $\sum r^2$: 0.17 / −0.08.
  - Ratios between −1 and 2 come from summing over a different set of particles or with a different weight; none of these is a gain applied to $M_0M_2$.
  - For these functions the two estimators agree, because their dilation response is exactly the predicted one ($R^2$ of $O_C$ ≈ 1).
  - For the network, $R^2$ of $O_C$ by $[P_1,\vec c]$ is 0.31 (S), 0.14 (h4), 0.16 (h5), 0.24 (z) and 0.003 (margin). That is the regressor noise behind the projection's downward bias.

**What the odd response follows (stage-pooled $R^2$ of $O_T$):**

| stage | fixed $\vec c$ only | $[P_1,\vec c]$ (pT-weighted $\vec M_1$) | $\vec M_1$ of 14 hardest | $\vec M_1$ of ranks ≥ 1 | unweighted centroid | leading particle | all centres |
|---|---|---|---|---|---|---|---|
| S | 0.011 | 0.025 | 0.193 | 0.103 | 0.348 | 0.135 | 0.389 |
| h4 | 0.007 | 0.019 | 0.096 | 0.059 | 0.172 | 0.070 | 0.206 |
| h5 | 0.007 | 0.022 | 0.092 | 0.062 | 0.160 | 0.071 | 0.195 |
| z | 0.010 | 0.042 | 0.119 | 0.069 | 0.178 | 0.081 | 0.229 |
| W−Z margin | 0.002 | 0.004 | 0.008 | 0.002 | 0.002 | 0.002 | 0.011 |

**Bulk vs tail, RMS of the odd translation response (standardised):**
- Total odd response:
  - S 0.38 → 0.26; h4 0.38 → 0.25; h5 0.33 → 0.26; z 0.26 → 0.18; margin 0.22 → 0.22.
  - This shrinks by 1.0–1.5×.
- Its $\vec M_1$-linear (IV) part:
  - S 0.045 → 0.005; z 0.046 → 0.005; margin 0.009 → 0.0002.
  - This shrinks by about 9× (S, z) and more at the margin.
  - The shrinkage follows from the construction: the predicted odd change is proportional to $\vec M_1$, which is small on bulk jets.
- E23w's "shrinks 8–14× on bulk jets" applies to the $\vec M_1$-projected part only, not to the odd response.

**Untrained redraws (seeds 1, 2; projection / IV a1/a2):**
- Seed 1: S 0.00 / 0.93; h4 0.00 / 3.21. h5 and z are unstable (IV a2 near 0 or −35).
- Seed 2: 0.17–0.84 / 2.1–2.4 at every stage.
- The untrained range includes the trained IV values. Seed 2 gives about 2 at every stage, the value E23v reported for the trained network's probe. The untrained redraws do not give a stable null for this ratio.

## Checks on the E23w script
- Edits: translation and dilation act on real particles only, and padding is unchanged (checked in the E23w log: padding_unchanged True; shift prediction max deviation 3e-8; $M_0$ unchanged exactly).
- Units and margin: `jedi_feat` takes stages S/h4/h5/z (z pre-ReLU) and logits W−Z (indices 2, 3).
- Odd/even pairing: $(\vec c,-\vec c)$ pairs are cells $k$ and $k+4$. Correct.
- FWL residualisation of $O_C$ on $\vec c$: correct for pooled OLS.
- No stage, mask or padding bug found.
- Arm 1 estimator:
  - The projection regresses on a noisy network response. This biases $a_1$ more than $a_2$ (shown above).
  - The bulk control as computed (`m1_linear_rms` = $a_1\cdot{\rm rms}(\tilde O_C)$) shrinks by construction, since $\tilde O_C\propto\vec M_1$.
  - The reflection control flips every centroid at once (pT-weighted, unweighted, hardest-14). So it cannot show that the odd term follows $\vec M_1$ rather than another centre.
- Arm 2 (product): not rerun.
  - The 200 null pairs are drawn with replacement, so pairs repeat (the log shows (52, 17) at draws 50 and 150).
  - Each null pair is evaluated only on jets with $S\ge$ step after the patch (242–1,015 jets), while the network is evaluated on all 2,000.
  - Both make the null band wider and less matched. E23w's reading for the product test is a null result with a wide band, and these points do not change it.

## Hypotheses (H0: one function of $M_0M_2$, IV ratio 1; H1: two gains)
- **S:** IV ratio 0.77 [0.64, 0.91], Holm p 0.003, so H0 is rejected. The analytic radial sums ($\sum r$, $\log M_2$ over ranks ≥ 1) give 0.69, inside this interval. The value at S matches what S-type radial sums produce; a separate read-out gain is not needed to produce it.
- **h4:** 0.97 [0.87, 1.08], Holm p 0.63, so H0 is not rejected. E23w's 0.42–0.61 at h4 is not reproduced with IV.
- **h5, z:** 0.86 and 0.87, Holm p 0.002 and 0.0007. These are 13–14 % below 1. With the $|\vec c|^2$ term in the even fit they are 0.93 and 0.95, and the intervals include 1. The departure therefore depends on the even-part specification.
- **Margin:** not identified. $P_1$ explains 0.4 % of the margin's odd response, and the IV interval spans 0.73–2.87.
- **E23v's ratio of about 2:** excluded at S–z with IV, as it is with the projection.
- **Invariant mass ($a_1=a_2=0$):** excluded (IV a2 0.81–1.10).

## Corrected reading
- **Projection reproduced, but biased.** E23w's projection estimator reproduces on a third, disjoint sample (a1/a2 0.47–0.87). It is biased toward 0 for the odd part, because the dilation response it regresses on carries 70–86 % variance unrelated to the predicted $\log M_2$ change.
- **IV gains.** Instrumented on the analytic predicted change, the ratio is:
  - 0.77 at S (below 1; equal to the value for simple radial sums of the S-object type);
  - 0.97 at h4 (not different from 1);
  - 0.86–0.87 at h5 and z (below 1 by 13–14 %, and within 1 when the even fit includes a $|\vec c|^2$ term);
  - undetermined at the margin.
- **"Ratio 0.4–0.8 at every stage" does not hold.** E23w's statements that "the ratio is 0.4–0.8 at every stage", that "it moves toward 1 between h4 and the margin", and that "it rises with $|\vec c|$" are properties of the projection estimator, not of the network.
- **The odd response mostly does not follow $\vec M_1$.** The $\vec M_1$-linear term explains 2–4 % of the variance of the odd translation response at S–z and 0.4 % at the margin. The unweighted particle centroid explains 16–35 %. On bulk jets the total odd response shrinks only 1.0–1.5×.
- **What the evidence supports.** A single function of full-jet $M_0M_2$ about the stored axis is excluded at S. Beyond that, the measured odd/even ratio is near 1 at h4–z once the estimator bias is removed. A two-gain functional form in $(\vec c\cdot\vec M_1, |\vec c|^2)$ describes only a small part of the shift response.
- **Product test.** E23w's product-test reading (no unit outside a wide random-pair band) is unchanged; it was not rerun.

## Explainer entry (for site/explain.py, Paper A, after E23w)
```python
E23W_VERIFY_EXPLAINER = [
 ('Question', 'Is the E23w odd/even gain ratio a property of JEDI-linear or of the estimator used to measure it?', None),
 ('Background', 'E23w regressed the odd part of the response to a shift, $O_T$, on the odd part of the response to a matched dilation, $O_C$. '
  '$O_C$ is itself a network response, so its parts unrelated to the predicted change $P_1=\\tfrac12[\\mathrm{pred}(\\vec c)-\\mathrm{pred}(-\\vec c)]$ act as error in the regressor and bias the gain toward 0.', None),
 ('Method', 'Fresh sample (seed 2313, jets of seeds 2311 and 2312 removed), 1,024 tail jets, same edits as E23w. '
  'Instrumented gain: $a_1^{IV}=\\sum_u\\beta^T_u\\beta^C_u/\\sum_u(\\beta^C_u)^2$, with $\\beta$ the coefficients of $O_T$ and $O_C$ on $[P_1,c_\\eta,c_\\phi]$; the same for the even part with $P_2$. '
  'Controls: analytic functions ($\\log M_0M_2$, quantised $\\log M_2$, radial sums), the variance of $O_T$ explained by alternative centres, bulk jets, untrained redraws.', None),
 ('Result', 'Projection a1/a2 reproduces (0.47–0.87). IV a1/a2: S 0.77 [0.64, 0.91], h4 0.97 [0.87, 1.08], h5 0.86, z 0.87 [0.80, 0.94], margin undetermined. '
  '$\\vec c\\cdot\\vec M_1$ explains 2–4 % of the odd response; the unweighted centroid explains 16–35 %. '
  'Radial sums such as $\\sum r$ give 0.69 with both estimators.', None),
 ('Checks', 'Both estimators return exactly 1 for $\\log M_0M_2$ and for quantised $\\log M_2$. Untrained redraws give IV ratios of 0.9–3.2.', None),
 ('Why it matters', 'The two-gain description of the mass read-out depends on the estimator. With the bias removed, the ratio is near 1 at h4–z and below 1 at the S sums, where it matches simple radial sums.', None),
]
```
Conclusion text for site/conclusions.py: the "Conclusion" paragraph above.
