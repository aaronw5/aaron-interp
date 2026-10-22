# E42x: W-pair edit with rigid subjets vs a matched non-W edit, on JEDI-linear and 7 retrained networks (claim A20t)

Script `interp/e42x_top_wpair_matched_jedi.py`; figure re-render `interp/e42x_plot.py`. Jobs: `anrunw-ai-e42x-smoke` (300 jets per class, end-to-end check) and `anrunw-ai-e42x-run`. The run used the conda env for the edit part and the jax env for the networks, with 1 CPU / 2 GiB. Wall time was 616 s, pod peak memory 1,257 MiB, and the exit code 0.

Results: `results/jedi/e42x.json` (all statistics), `e42x_edit.json` (edit counts, side effects, references), `e42x_scores.npz` (per-jet official scores, reference scores, rows, example jet), `e42x_wpair_rigid.png`. Smoke outputs are `e42x_smoke*.json`. Logs: `logs/e42x-run.log`, `logs/e42x-smoke.log`.

## Conclusion

We moved the W-candidate pair inside real top jets by sliding whole subjets, without resizing any of them. JEDI-linear's top score then drops more than under a matched edit of a different subjet pair. The same happens when an off-peak W pair is moved *onto* 80 GeV. A classifier that reads the W-pair mass does the opposite in that case: its top score rises. So the network's extra response depends on the W-pair geometry being disturbed, not on how close the pair mass is to the W mass. After adjusting for the remaining shape changes (tau21, tau32, soft-drop mass), the network's W − N is +0.10 at 60 GeV and −0.19 for the move onto 80 GeV. The pair-mass-aware reference gives −0.33 and +0.29. The signature of a W-mass reader is therefore absent in the official network and in all seven retrained networks. Two things remain uncertain. Only 12–28 % of eligible tops pass the strict rigid-edit acceptance. And the control edit cannot hold tau32 fixed (see Procedure), so the comparison relies on reference classifiers and regression adjustment, not on exact matching.

## Question and why now

Claim A20t (0.2) says moving the W-candidate subjet-pair mass of top jets lowers JEDI-linear's top log-odds by 0.18–0.39 more than a control edit. E42v showed that E42's W edit also shrinks every subjet. The common internal-spread factor $h$ had median 0.78 at 60 GeV, against 1.00 for the control. The edit also lowers tau21. A classifier blind to the pair mass reproduced the difference. E42x removes that side effect: every edit here translates whole subjets with $h = 1$, so no subjet changes width.

## Definitions

- **Subjets.** Three exclusive-$k_T$ subjets of the 64 stored constituents, taken as massless four-vectors (E-scheme recombination, rapidity–azimuth distances; `h4_top_wmass`).
  - **W pair** $(a, b)$: the subjet pair whose invariant mass $m_W$ is closest to 80.4 GeV.
  - **$c$**: the third subjet (normally the b-quark subjet).
- **$m$, $m_{SD}$.** $m$ is the jet invariant mass. $m_{SD}$ is the soft-drop mass ($z_{cut} = 0.1$, $\beta = 0$, C/A).
- **$\tau_{21} = \tau_2/\tau_1$, $\tau_{32} = \tau_3/\tau_2$.** N-subjettiness ratios ($\beta = 1$, WTA $k_T$ axes, $R_0 = 0.8$). Low $\tau_{32}$ means three-prong-like; low $\tau_{21}$ means two-prong-like.
- **Subjet width $w_k$.** The pT-weighted mean $\Delta R$ of a subjet's constituents from the subjet axis. $w_1, w_2, w_3$ are sorted by subjet pT.
- **$p_T^D = \sqrt{\sum p_{T,i}^2}/\sum p_{T,i}$.** Unchanged by every edit here, because no particle's pT changes.
- **Displacement $D$.** The pT-weighted mean distance in $(\eta, \phi)$ that the constituents move, after removing the common shift of the whole jet.
- **E-scheme axis.** The direction of the summed four-momentum. Every jet variant is re-expressed about its own E-scheme axis before it enters a network, so no edit moves the axis the network sees.
- **Score $s$.** The network's top log-odds, $s = \ell_t - \log\sum_{k \ne t} e^{\ell_k}$ ($\ell$ = logits). Per jet, $\Delta_v = s_v - s_{\rm id}$.
- **W − N.** $\overline{\Delta_W - \Delta_N}$: how much more the top score changes under the W-pair edit than under the matched non-W edit.
  - An $m_W$ reader predicts W − N < 0 when $m_W$ moves 65–95 → 60 or 100 GeV.
  - It predicts W − N > 0 when an off-peak $m_W$ (60–70 or 90–100 GeV) moves onto 80.4 GeV.
- **BLIND / AWARE.** Histogram gradient-boosted trees (HistGBDT), trained top vs the other four classes on the 25,000 natural sample jets.
  - **BLIND** sees $(m, m_{SD}, \tau_{21}, \tau_{32}, N, w_1, w_2, w_3, p_T^D)$ and no pair mass. It predicts the W − N that comes from side effects alone.
  - **AWARE** adds the three pair masses (the $m_W$ candidate and the other two, sorted). It is the positive control: what an $m_W$-reading classifier does under the same edits.
  - Both use 2-fold cross-fitting: each jet's variants are scored by the model not trained on that jet. Out-of-fold AUC (top vs rest) is 0.941 for BLIND and 0.941 for AWARE.
- **McNemar.** An exact binomial test on the jets whose top call (argmax = top) differs between the W and N variants.
- **Holm.** Step-down family-wise correction, applied over 8 trained networks × 3 sets (24 tests per statistic).

## Procedure

**Sample.** `jedi_core.sample(5000, 4421)`: 25,000 hls4ml test jets, 5,000 per class.
- 3,198 rows overlap the E42/E42v samples. They were not removed; the seed is new.
- The identity rebuild matches the stored inputs to $1.2\times10^{-4}$.
- Top window: $105 < m_{SD} < 210$ GeV (3,617 tops).
- T60 / T100: 3,002 tops with $65 < m_W < 95$ GeV, W pair moved to 60 / 100 GeV.
- T80 (onto-peak): all 610 tops with $60 \le m_W < 70$ or $90 < m_W \le 100$ GeV, W pair moved to 80.4 GeV.

**Edits.** Every subjet is translated as a whole in $(\eta, \phi)$. Each particle's pT, each subjet's internal shape and width, and $p_T^D$ are unchanged.

- **Wr (W-rigid).**
  1. The W-pair opening is scaled by $f$ about the pair's pT-weighted centroid $c_W$, with bisection on $f$ so that $m_{ab} = t$.
  2. Subjet $c$ is placed at $c_W + g\,R(\theta)(r_c - c_W)$. For each $\theta$, $g$ restores the jet mass.
  3. $\theta = 0$ is kept when $\tau_{32}$ is already within 0.5 %. Otherwise the smallest $|\theta|$ (scan to ±π) that returns $\tau_{32}$ to its unedited value is used.
  4. Medians: $f$ = 0.79 / 1.25 / 1.17 and $g$ = 1.15 / 0.85 / 0.92 (T60 / T100 / T80). The 10–90 % range of $\theta$ is −0.3 to +0.33 rad.
  - Why the extra coordinate: rescaling $c$'s separation fixes only the jet mass. With rigid subjets, the three-subjet triangle has three degrees of freedom. $m_W = t$, $m = m_0$ and $\tau_{32} = \tau_{32,0}$ use all three, so the rotation $\theta$ is needed.
- **ND (N-matched, same displacement; primary control).**
  1. The non-W pair $(d, c)$ is edited, with $d$ the harder W prong (the softer one if that fails).
  2. Subjet $c$ is put on a circle of radius $f'|r_c - r_d|$ about $d$, at the azimuth nearest the original that keeps the jet mass.
  3. $f'$ is chosen by bisection so that the displacement $D$ equals Wr's for the same jet.
  4. The W pair does not move, so $m_W$ stays fixed.
  - Why $\tau_{32}$ is not restored: once $m$ and $m_W$ are fixed, the triangle has one degree of freedom left. The only configuration that also keeps $\tau_{32}$ is the unedited jet. So the N control cannot restore $\tau_{32}$, and its $\tau_{32}$ change is measured and treated as a side effect (BLIND, regression, matched subset).
- **Nf (N-matched, same opening factor; secondary).** As ND, but with $f' = f$, so the opening change matches Wr's instead of the displacement.
- **Ws / Ns.** E42's edit (subjet sizes scaled by $h$) and its control, for reference.

**Acceptance.** Observables are recomputed after reclustering.
- Wr: $|\Delta m|/m < 1\,\%$, $|\Delta\tau_{32}|/\tau_{32} < 1\,\%$, $|m_W - t| < 5$ GeV (2 GeV for T80).
- ND / Nf: $|\Delta m|/m < 1\,\%$, $|m_W - m_{W,0}| < 2$ GeV.
- The analysis set for each pair is the jets where both edits are accepted.

**Real-mass reference (DIL).** All angles are scaled by λ = 0.92 and 1.08 on the 3,002 eligible tops.

**Networks.**
- The official 3-feature n64 network. Unedited accuracy on its sample is 80.81 % (stored axis) and 80.80 % (recentred), within the 80.6–81.3 % check.
- The 7 E39 networks: official n32, perm-inv n64, HGQ-r150 s0/s1/s2, float retrains s0/s1.
- One untrained redraw (Jedi(untrained=True, seed 0)).

**Statistics.**
- Bootstrap over jets (2,000 draws); sign-flip permutation (10,000 draws).
- H0: $(W-N)_{\rm net} = (W-N)_{\rm BLIND}$, tested on the per-jet paired difference.
- Per-jet regression of $(\Delta_W - \Delta_N)$ on the W−N differences in $\tau_{21}$, $\tau_{32}$, $m_{SD}$ and $D$ (standardised). The intercept is the W − N at zero side-effect difference; its CI comes from 2,000 bootstraps.
- Holm over 24 tests.

**Changes after the smoke run** (decided before the full run; the smoke network numbers had N = 11–26 and CIs of ±0.5–2):
- The primary control was switched from Nf to ND. On the full sample, Nf moved $\tau_{32}$ by +0.095 and $m_{SD}$ by +13 GeV relative to Wr at 60 GeV. ND moved them by +0.024 and +3.6 GeV.
- The $\theta$ search was widened from ±0.8 rad to ±π, and several brackets are now tried. In the smoke run Wr was solved for only 40 / 186 tops.

## Results

### Edit counts (accepted Wr / ND; analysis set)

| set | eligible | Wr solved → accepted | ND solved → accepted | Wr−ND analysis set | Wr−Nf | Ws−Ns |
|---|---|---|---|---|---|---|
| T60 | 3,002 | 818 → 348 | 474 → 306 | 138 | 178 | 584 |
| T100 | 3,002 | 2,257 → 806 | 980 → 763 | 295 | 473 | 390 |
| T80 | 610 | 337 → 214 | 197 → 114 | 84 | 96 | – |

### Side effects, W minus N (Wr − ND, mean with 95 % CI)

| set | $\Delta m_W$ (GeV) | $w_1$ | $w_3$ | $\tau_{21}$ | $\tau_{32}$ | $m_{SD}$ (GeV) | $D$ |
|---|---|---|---|---|---|---|---|
| T60 | −14.1 | 0.0000 [−0.0005, 0.0006] | −0.0026 [−0.0054, −0.0002] | −0.039 [−0.052, −0.027] | +0.024 [+0.008, +0.040] | +3.6 [+1.6, +5.8] | 0 (matched; 0.035) |
| T100 | +17.2 | +0.0002 | −0.0009 | −0.031 [−0.036, −0.027] | +0.052 [+0.047, +0.057] | −0.07 [−1.25, +0.95] | 0 (0.036) |
| T80 | +3.4 (mean of up / down moves) | −0.0003 | +0.0002 | −0.015 [−0.027, −0.004] | +0.035 [+0.024, +0.045] | +1.7 [+0.1, +3.5] | 0 (0.030) |

- Subjet widths change by at most 0.003. The typical widths are 0.016–0.036, so this is under 10 % of the narrowest width.
- For comparison, the E42 edit (Ws) on the same jets narrows the leading subjet by −0.0036 (median $h$ 0.78) and changes tau21 by −0.054 relative to Ns.
- In ND, $\tau_{32}$ falls ($\Delta\tau_{32}$ = −0.024 / −0.051 / −0.034, unedited medians 0.47–0.55), so the N jets become more three-prong-like. Wr holds $\tau_{32}$ to within 0.0003.
- The sign of the remaining $\tau_{21}$ difference (W lower than N) is the same as in E42v, and its size is 0.73× E42's at 60 GeV.

### Official JEDI-linear (n64), Wr − ND (primary)

| set | n | $\Delta_W$ | $\Delta_N$ | W − N [95 % CI] | perm p (Holm) | BLIND W − N | AWARE W − N | net − BLIND [CI], p (Holm) | top-call changes W / N, McNemar p |
|---|---|---|---|---|---|---|---|---|---|
| T60 | 138 | −0.224 | −0.024 | −0.200 [−0.375, −0.027] | 0.026 (0.32) | −0.024 [−0.159, +0.114] | −0.541 [−0.692, −0.381] | −0.176 [−0.371, +0.017], 0.066 (0.93) | 5 / 4, 1.0 |
| T100 | 295 | −0.042 | +0.395 | −0.437 [−0.539, −0.335] | 1e-4 (0.0024) | −0.316 [−0.377, −0.261] | −0.676 [−0.746, −0.611] | −0.121 [−0.224, −0.020], 0.019 (0.29) | 4 / 4, 0.125 |
| T80 | 84 | +0.030 | +0.244 | −0.214 [−0.367, −0.067] | 0.0054 (0.086) | −0.098 [−0.196, +0.002] | **+0.121** [+0.015, +0.233] | −0.116 [−0.293, +0.044], 0.18 (1.0) | 2 / 1, 1.0 |

- **Side-effect-adjusted W − N** (regression intercept):

  | set | official network | BLIND | AWARE |
  |---|---|---|---|
  | T60 | +0.103 [−0.059, +0.280] | +0.122 [+0.023, +0.220] | −0.332 [−0.472, −0.188] |
  | T100 | −0.168 [−0.357, −0.006] | +0.076 [−0.008, +0.146] | −0.315 [−0.411, −0.216] |
  | T80 | −0.195 [−0.378, −0.037] | +0.069 [−0.032, +0.161] | **+0.291** [+0.163, +0.412] |

- **Matched subset** ($|\Delta\tau_{21}| < 0.02$, $|\Delta\tau_{32}| < 0.02$, $|\Delta m_{SD}| < 3$ GeV between W and N): n = 3 / 20 / 7, too few to use. T100 gives −0.36 [−0.62, −0.08].
- **Secondary control Wr − Nf:**
  - W − N: −0.525 (T60), −0.355 (T100), −0.213 (T80).
  - Net − BLIND: −0.10 [−0.30, +0.11], −0.17 [−0.28, −0.08] (Holm 0.013), −0.03 [−0.21, +0.15].
  - Nf has large $\tau_{32}$ / $m_{SD}$ side effects, and BLIND's W − N at T60 (−0.42) is close to the network's.
- **E42 edit on the same jets (Ws − Ns):** W − N = −0.230 / −0.276 (T60 / T100), reproducing E42 / E42v. BLIND gives −0.002 / −0.087, so net − BLIND = −0.23 / −0.19 (p ≤ 0.004).
  - This BLIND sees the subjet widths directly. Unlike E42v's MLPx, it does not reproduce W − N for the scaled edit.
  - The rigid edits, which remove the width change, give the T80 result above.
- **Calibration by $k_{\rm DIL}$ is not usable.** It is the ratio of the network's to BLIND's antisymmetric response under all-angle scaling. BLIND's top score falls under both λ = 0.92 (−0.42) and λ = 1.08 (−0.50), so the antisymmetric part is near zero and $k_{\rm DIL}$ = −5.8 [−13.8, −3.4]. The rescaled comparisons in the JSON (`net_minus_kref`) are therefore not interpreted. The raw comparison is the primary one.

### All networks, Wr − ND, W − N [95 % CI] (Holm p for net − BLIND)

| network | acc. (stored) | T60 | T100 | T80 (onto peak) |
|---|---|---|---|---|
| official n64 | 80.81 % | −0.200 [−0.375, −0.027] (0.93) | −0.437 [−0.539, −0.335] (0.29) | −0.214 [−0.367, −0.067] (1.0) |
| official n32 | 77.87 % | −0.144 [−0.289, +0.003] (1.0) | −0.471 [−0.573, −0.375] (0.056) | −0.163 [−0.309, −0.018] (1.0) |
| perm-inv n64 | 81.81 % | −0.124 [−0.298, +0.047] (1.0) | −0.482 [−0.595, −0.370] (0.054) | −0.191 [−0.365, −0.025] (1.0) |
| HGQ-r150 s0 | 76.98 % | −0.083 [−0.167, −0.003] (1.0) | −0.057 [−0.117, +0.001] (net above BLIND, 0.0024) | −0.017 [−0.092, +0.057] (1.0) |
| HGQ-r150 s1 | 57.03 % | +0.042 [−0.040, +0.126] (1.0) | −0.014 [−0.065, +0.034] (above, 0.0024) | −0.110 [−0.192, −0.041] (1.0) |
| HGQ-r150 s2 | 78.14 % | −0.032 [−0.103, +0.036] (1.0) | −0.027 [−0.075, +0.018] (above, 0.0024) | −0.044 [−0.111, +0.027] (1.0) |
| float s0 | 82.39 % | −0.488 [−0.709, −0.260] (0.0024) | −0.666 [−0.773, −0.560] (0.0024) | −0.221 [−0.410, −0.050] (1.0) |
| float s1 | 82.13 % | −0.414 [−0.648, −0.197] (0.0036) | −0.700 [−0.828, −0.583] (0.0024) | −0.178 [−0.328, −0.030] (1.0) |
| untrained | 20.0 % | −0.001 [−0.007, +0.005] | +0.001 [−0.003, +0.005] | −0.004 [−0.015, +0.006] |
| BLIND GBDT | – | −0.024 [−0.159, +0.114] | −0.316 [−0.377, −0.261] | −0.098 [−0.196, +0.002] |
| AWARE GBDT | – | −0.541 [−0.692, −0.381] | −0.676 [−0.746, −0.611] | +0.121 [+0.015, +0.233] |

- **T80, onto-peak arm (8 networks).** No trained network gives a positive T80 W − N. Six of eight have a CI below 0; the three HGQ-r150 networks are within [−0.11, −0.02].
- **Side-effect-adjusted T80 intercepts.** Official −0.19 [−0.38, −0.04], float s0 −0.12 [−0.35, +0.13], float s1 −0.01 [−0.21, +0.18], HGQ s0 −0.01. AWARE gives +0.29 [+0.16, +0.41].
- **Networks that exceed BLIND after Holm.** Only the two float retrains, at T60 and T100 (net − BLIND −0.35 to −0.46, Holm ≤ 0.004). Their T80 W − N is also negative (−0.22, −0.18).
- **HGQ-r150 networks.** Their W − N lies above BLIND's at T100 (Holm 0.0024).
- **Top calls.** Calls change for at most 12 of 295 jets per variant. No Wr − ND McNemar test passes Holm; the smallest raw p is 0.016, for n32 at T100.

![E42x](../results/jedi/e42x_wpair_rigid.png)

Figure: `results/jedi/e42x_wpair_rigid.png`.
- **(a)** One real top jet (unedited $m_W$ 74 GeV, $m$ 163 GeV, $\tau_{32}$ 0.50).
  - Unedited particles: filled circles.
  - W-rigid with the W pair moved to 60 GeV: triangles. Reclustered $m_W$ 63 GeV, $m$ 163 GeV, $\tau_{32}$ 0.50.
  - N-matched (same displacement): squares. $m_W$ 75 GeV, $\tau_{32}$ 0.56.
  - Each subjet's particle cloud moves as a whole and keeps its shape.
- **(b)** W − N per network and for the BLIND / AWARE references, for the three sets.

## Reading (scoped to what was measured)

- **Mass-reader signature.** Under rigid W-pair edits that hold every subjet's width, an $m_W$-reading classifier (AWARE) has W − N < 0 for moves away from the peak and W − N > 0 for moves onto it. This holds both raw and after the side-effect regression. All 8 trained networks lack the positive onto-peak response (T80), and the official network's adjusted T80 intercept is negative (−0.19 [−0.38, −0.04]).
  - The prediction "JEDI-linear's top score reads $m_W$" requires a positive T80. On this sample it is excluded for the official network: AWARE's adjusted +0.29 [+0.16, +0.41] and the network's −0.19 [−0.38, −0.04] do not overlap.
- **Excess over BLIND.** H0 ($(W-N)_{\rm net} = (W-N)_{\rm BLIND}$) is not rejected for the official network after Holm (raw p 0.066 / 0.019 / 0.18). The excess over BLIND is −0.12 to −0.18 in the same negative direction for all three sets, including T80. That pattern fits a dependence on "W-pair geometry changed relative to the third subjet", not on the mass value.
- **What differs from E42.** The E42 edit itself (Ws − Ns) still gives −0.23 / −0.28 on this sample, beyond a width-aware BLIND (≈ 0). Part of E42's effect therefore comes from the subjet-size change in a way that the GBDT on $(w_1, w_2, w_3)$ does not capture. This step does not identify which input property carries it.
- **Competing hypotheses.**

  | hypothesis | prediction | outcome |
  |---|---|---|
  | (i) $m_W$ reader | T60 / T100 negative, T80 positive | Excluded for the official network: T80 is negative |
  | (ii) Side effects only | W − N equal to BLIND | Not excluded for the official network after Holm |
  | (iii) Sensitivity to any rigid change of the W-pair opening relative to $c$ | Negative in all three sets | Matches the sign pattern; not separated from (ii) with significance for the official network; separated for the float retrains at T60 / T100 |

## Checks

- **Unedited accuracy.** 80.81 % on the 25,000-jet sample, in range.
- **Untrained redraw.** |W − N| ≤ 0.004.
- **Identity rebuild.** Matches to 1.2e-4. Padding is asserted contiguous.
- **Width check.** Subjet widths are held to under 0.003 by construction; this is a check that the rigid edit works.
- **Positive control.** AWARE responds with the $m_W$-reader signs in all three sets.
- **Overlap.** Of the 25,000 sample rows, 3,198 were also in E42 / E42v samples (not removed).

## Issues and limits

- **Acceptance.** Strict acceptance (1 % on $m$ and $\tau_{32}$ after reclustering) keeps 12 % (T60), 27 % (T100) and 35 % (T80, Wr) of eligible tops, and fewer once both W and N must pass. The analysis sets favour jets whose triangle can be re-solved, and T80 has n = 84.
- **Unmatched $\tau_{32}$.** The N control cannot hold $\tau_{32}$ (a degree-of-freedom limit, shown above). W − N is therefore compared at matched side effects only through BLIND and the regression, not by exact matching. The exactly matched subset is too small (n 3–20).
- **Calibration.** $k_{\rm DIL}$ is not interpretable here (negative, because BLIND's response to all-angle scaling is even in λ).
- **Changes after the smoke run.** The primary control was switched from Nf to ND because Nf's side effects were larger. Both are reported, and Holm is computed separately for each.

## Explainer entry (for site/explain.py, A20t / Paper A tab)

**Question.** Does JEDI-linear's top score use the mass of the W-candidate subjet pair, once the subjets are kept the same size?

**Background.**
- A top jet has three prongs. Two of them come from the W and have pair mass $m_W \approx 80$ GeV.
- E42 changed $m_W$ at fixed jet mass and $\tau_{32}$, and the top score dropped more than under a control edit.
- E42v found that this edit also shrank the subjets (factor $h \approx 0.78$) and lowered $\tau_{21}$, and that a classifier without $m_W$ showed the same drop.

**Method.**
1. In each top jet, cluster three subjets.
2. Slide whole subjets without resizing them:
   - Scale the W-pair opening by $f$ so that $m_{ab} = t$.
   - Place the third subjet at $c_W + g R(\theta)(r_c - c_W)$, so that $m$ and $\tau_{32}$ return to their unedited values.
3. Control: move the third subjet relative to one W prong so that the constituents move by the same pT-weighted distance $D$, with $m$ and $m_W$ unchanged.
4. Score $s = \ell_t - \log\sum_{k\ne t} e^{\ell_k}$. Compare $W-N = \overline{(s_W - s_0) - (s_N - s_0)}$ with two gradient-boosted classifiers, one without pair masses (BLIND) and one with them (AWARE).

**Result** (figure `e42x_wpair_rigid.png`).
- AWARE gives W − N = −0.54 / −0.68 / +0.12 (to 60 GeV / to 100 GeV / onto 80 GeV).
- JEDI-linear gives −0.20 / −0.44 / −0.21. Its move onto 80 GeV lowers the score instead of raising it.
- After regressing out the remaining $\tau_{21}$, $\tau_{32}$ and $m_{SD}$ differences, the onto-peak values are +0.29 (AWARE) and −0.19 (JEDI).

**Checks.**
- Untrained network |W − N| ≤ 0.004.
- Subjet widths unchanged to < 0.003.
- The 7 retrained networks show no positive onto-peak response.

**Why it matters.** It removes the remaining candidate for a W-mass term inside JEDI-linear's top call. What the network responds to under these edits is a disturbance of the W-pair geometry relative to the third subjet, in either direction, not closeness to 80 GeV.

**New glossary terms** (for explain.GLOSSARY):
- **W-rigid edit:** subjets translated as wholes (no resizing) so that $m_W = t$ at fixed $m$ and $\tau_{32}$.
- **N-matched (same displacement):** non-W pair edited so that the pT-weighted particle displacement equals the W edit's, with $m$ and $m_W$ fixed.
- **BLIND / AWARE GBDT:** top-vs-rest gradient-boosted trees without / with the three subjet-pair masses.
- **Onto-peak arm (T80):** off-peak W pairs moved onto 80.4 GeV.
- **Subjet width:** pT-weighted mean $\Delta R$ to the subjet axis.
- **$p_T^D$:** $\sqrt{\sum p_T^2}/\sum p_T$.

## Effect on claim A20t

A20t currently reads: "W-pair term not established, side effects explain the difference." E42x refines it. With subjet widths held, an $m_W$-reading signature (a positive onto-peak response) is excluded for the official network on this sample and is absent in all 8 trained networks. The network's remaining extra response is negative in every direction of the W-pair edit and is not distinguishable from the pair-blind reference after Holm (official). Suggested wording: "JEDI-linear's top score does not read the W-pair mass; rigid W-pair deformations lower it in either direction, by 0.1–0.2 beyond a pair-blind reference (not significant after Holm for the official network)."
