# E42x verify: rigid W-pair edit on a fresh sample, and a toward-vs-away control that does not use the non-W edit

Script `interp/e42x_verify.py`. It imports E42x's edit code unchanged (`w_rigid`, `n_matched`, `Fast`, `disp`, `allobs` from `interp/e42x_top_wpair_matched_jedi.py`). Jobs:
- `anrunw-ai-e42x-vsmoke`: 300 jets per class, end-to-end check.
- `anrunw-ai-e42x-verify`: AI_PY=conda edit part, then the jax env; 1 CPU / 2 GiB; wall 1,664 s; exit 0.

Results: `results/jedi/e42x_verify.json` (all statistics) and `e42x_verify_edit.json` (counts, side effects, reference AUCs); smoke versions `e42x_verify_smoke*.json`. Logs: `logs/e42x-verify.log`, `logs/e42x-vsmoke.log`.

## Conclusion

We tested whether JEDI-linear's top score depends on how close the W-candidate subjet pair mass is to 80 GeV. On a fresh sample, the same off-peak W pair was slid onto 80 GeV in one copy of each jet and an equal distance away from 80 GeV in another copy, with the jet mass and three-prong shape held fixed in both. The network's score difference between the two copies matches a classifier that never sees any pair mass (difference +0.02 log-odds). A classifier that does see the pair masses gives +0.61. So the network does not use the W-pair mass as strongly as a pair-mass-aware classifier does. A weaker dependence, up to about 0.4 of that classifier's, is not excluded.

Two parts of E42x's reading are not supported:
- The negative onto-peak W − N in E42x (−0.21) comes mostly from the control edit raising the network's score, not from the W edit lowering it. On the fresh sample it shrinks to −0.14 (p 0.12) and equals the pair-blind classifier's value.
- "Rigid W-pair deformations lower the top score in either direction" does not hold for the W edit alone. Beyond the pair-blind classifier, the W edit alone changes the score by −0.13 to −0.19 at 60 GeV, about 0 at 100 GeV, and +0.05 to +0.07 when moved onto 80 GeV.

## Definitions (as in E42x)

- **Subjets.** Three exclusive-$k_T$ subjets per jet (massless constituents, E-scheme). The **W pair** $(a, b)$ is the pair whose invariant mass $m_W$ is closest to 80.4 GeV. $c$ is the third subjet.
- **$m$, $m_{SD}$, $\tau_{21}$, $\tau_{32}$.**
  - $m$: jet invariant mass.
  - $m_{SD}$: soft-drop mass.
  - $\tau_{21} = \tau_2/\tau_1$ and $\tau_{32} = \tau_3/\tau_2$: N-subjettiness ratios (WTA $k_T$ axes, $\beta = 1$). A low $\tau_{32}$ means the jet looks three-pronged.
- **Wr (W-rigid edit).** Whole subjets are translated in $(\eta, \phi)$, and none is resized.
  - The W-pair opening is scaled by a factor $f$ until $m_{ab} = t$.
  - The third subjet is then rotated and rescaled about the W-pair centroid until $m$ and $\tau_{32}$ return to their unedited values.
- **ND (N-matched control).** The pair (harder W prong, $c$) is opened, with $m$ and $m_W$ held fixed. The opening is chosen so that the particles move the same pT-weighted distance $D$ as in Wr.
  - With rigid subjets, this edit cannot also hold $\tau_{32}$. Its $\tau_{32}$ drops by 0.02–0.05, so the N jets look more top-like.
- **Score.** $s = \ell_t - \log\sum_{k\ne t} e^{\ell_k}$, the top log-odds ($\ell$ = logits).
  - $\Delta_W = s_W - s_0$ and $\Delta_N = s_N - s_0$, where $s_0$ is the unedited jet's score.
  - $W - N = \overline{\Delta_W - \Delta_N}$.
- **BLIND / AWARE.** Gradient-boosted trees (HistGBDT) trained top vs the other four classes, with 2-fold cross-fitting on the natural sample.
  - BLIND sees $(m, m_{SD}, \tau_{21}, \tau_{32}, N, w_1, w_2, w_3, p_T^D)$ and no pair mass, where $w_k$ is the subjet width (pT-weighted mean $\Delta R$ to the subjet axis).
  - AWARE adds the three pair masses. It is the positive control: what a W-mass-reading classifier does under the same edits.
- **"net − BLIND".** The per-jet paired difference between the network's response and BLIND's response to the same edited jets. This is the part of the network's response that the side effects (the changes in $m_{SD}$, $\tau_{21}$, $\tau_{32}$ and widths) do not account for.
- **TA (toward vs away), new here.** Each off-peak top gets two Wr edits of the same W pair:
  - toward: $t = 80.4$ GeV;
  - away: $t = 2m_{W,0} - 80.4$, the same $|\Delta m_W|$ in the other direction. A pair at 60–70 GeV goes to 40–60 GeV; a pair at 90–100 GeV goes to 100–120 GeV.
  - Per jet, $C = s_{\rm toward} - s_{\rm away}$.
- **Balanced mean $C_{\rm bal}$.** $(\bar C_{\rm low} + \bar C_{\rm high})/2$. On the low side ($m_{W,0} < 80.4$), toward opens the pair and away closes it; on the high side it is the reverse. Averaging the two sides cancels any response that depends only on the opening direction.
- **Holm.** Step-down family-wise correction; family sizes are given with each result.
- **McNemar.** Exact binomial test on the jets whose top call differs between two variants.

## What was checked in the E42x code

- **Inputs, padding, axis.**
  - A particle is real when pT > 0, and padding is asserted contiguous.
  - Every variant is rebuilt about its own E-scheme axis (`model_x(P, n, axis(P, n))`). The identity rebuild matches the stored inputs to 1.2e-4.
  - Particle order (pT rank) is unchanged by the edits, because no pT changes. So JEDI-linear's rank windows see the same particles.
- **Stages.** Only logits are used: `Jedi.run` for the official network and the untrained redraw, and `e33.Model.run` for the 7 retrained networks. The score uses class index 4 (top), matching labels y == 4.
- **Edits.**
  - `w_rigid` translates subjets only: shifts are per subjet and $h = 1$. The reported width changes (≤ 0.003) confirm this.
  - `n_matched` with a displacement target starts its bisection at $q = 1$ with $G = -D$, which is correct, because no move gives zero displacement.
  - `disp` is computed at subjet level, which is exact for rigid moves.
- **Acceptance.** It is computed from reclustered observables. The analysis set is the intersection of accepted W and accepted N. The same selection applies to both arms, so it cannot bias W − N in one direction.
- **References.** The cross-fit uses `fold[rows]` on the 25,000-row index. A jet's variants are scored by the model that was not trained on that jet. No split leakage was found.
- **Bugs.** None found in the code.
- **Analysis gap.** The report and the figure give only the W − N difference. On the E42x sample, the components are as follows (from `e42x_scores.npz`, 2,000-draw jet bootstrap):

| set | n | official $\Delta_W$ | official $\Delta_N$ | $\Delta_W$ − BLIND [95 % CI] | $\Delta_N$ − BLIND [95 % CI] | AWARE $\Delta_W$ − BLIND |
|---|---|---|---|---|---|---|
| T60 | 138 | −0.224 | −0.024 | −0.186 [−0.349, −0.023] | −0.009 [−0.146, +0.127] | −0.509 |
| T100 | 295 | −0.042 | +0.395 | −0.001 [−0.093, +0.094] | +0.119 [+0.016, +0.221] | −0.307 |
| T80 (onto peak) | 84 | +0.030 | +0.244 | +0.068 [−0.049, +0.192] | +0.183 [+0.059, +0.308] | +0.182 |

- **How to read the T80 row.**
  - Moving the off-peak W pair onto 80 GeV changes the network's score by +0.03. Beyond BLIND this is +0.07, which is 0.37 of AWARE's +0.18, in the W-mass-reader direction, and not significant.
  - The negative T80 W − N (−0.21) comes from the control: the N edit raises the network's score by 0.18 more than it raises BLIND's.
  - So E42x's exclusion of a W-mass reader at T80 rests on the network's response to the N edit, not on its response to the W edit.

## Procedure of this verification

- **Sample.** `jedi_core.sample(7000, 4521)`: N = 35,000 jets (7,000 per class).
  - 3,399 rows are shared with E42x (seed 4421), and 4,534 with E42 / E42v; they were not removed.
  - The identity rebuild matches to 1.2e-4.
  - Official accuracy on this sample is 80.60 % (the expected range is 80.6–81.3 %).
- **Eligible jets.** Tops with $105 < m_{SD} < 210$ GeV: 4,969.
  - T60 / T100: 4,117 tops with $65 < m_W < 95$ GeV.
  - T80: 856 tops with $m_W$ in 60–70 or 90–100 GeV.
- **(1) Replication of Wr − ND at T60 / T100 / T80.** Same edits and acceptance as E42x. Each edit response is also compared with BLIND separately ($\Delta_W$ − BLIND, $\Delta_N$ − BLIND).
- **(2) TA.** Run on the T80-eligible tops whose kt3 W pair satisfies $|m_{W,0} - 80.4| > 8$ GeV.
  - Both edits must pass $|\Delta m|/m < 1\,\%$ and $|\Delta\tau_{32}|/\tau_{32} < 1\,\%$, and one reclustered pair mass must lie within 2 GeV of $t$.
  - Both copies hold $m$, $\tau_{32}$ and every subjet width, and have the same $|\Delta m_W|$. So $C$ has no N-type control in it.
  - The remaining side-effect difference (toward − away) is $m_{SD}$ −4.1 GeV, $\tau_{21}$ +0.002, $\tau_{32}$ 0.0000, $D$ +0.007. BLIND covers it, and so does a regression on $(\tau_{21}, m_{SD}, D, \text{side})$.
  - n = 83 (28 low side, 55 high side). 231 of the 402 solved away edits were accepted.
- **Statistics.**
  - Intervals: 2,000-draw jet bootstrap, stratified by side for TA.
  - p-values: 10,000-draw two-sided sign-flip permutation; exact McNemar on top calls.
  - Holm over the 8 trained networks (TA) and over 8 × 3 (T sets).
  - H0 for TA: $C_{\rm bal,net} = C_{\rm bal,BLIND}$.
  - H1 (W-mass reader): $C_{\rm bal,net} - C_{\rm bal,BLIND} > 0$, with size up to AWARE's.
  - Competing hypothesis (deformation only): $C_{\rm bal,net} = C_{\rm bal,BLIND}$.
- **Networks.** The official n64, the 7 E39 networks (official n32, perm-inv n64, HGQ-r150 s0/s1/s2, float s0/s1) and an untrained redraw.

## Results (fresh sample 4521)

### TA: toward minus away (no N edit)

| classifier | $C_{\rm bal}$ [95 % CI] | − BLIND [95 % CI], p (Holm, 8) | low side $C$ (n 28) | high side $C$ (n 55) | regression intercept [CI] |
|---|---|---|---|---|---|
| official n64 | −0.219 [−0.496, +0.054] | **+0.022 [−0.213, +0.267]**, p 0.86 (1.0) | −0.547 | +0.108 | −0.042 [−0.295, +0.216] |
| BLIND | −0.241 [−0.440, −0.079] | 0 | −0.398 | −0.085 | −0.098 [−0.193, +0.002] |
| AWARE | +0.369 [+0.095, +0.604] | **+0.610 [+0.473, +0.744]**, p 1e-4 | +0.276 | +0.462 | +0.546 [+0.388, +0.716] |
| official n32 | −0.163 | +0.079 [−0.185, +0.334] (1.0) | | | +0.003 |
| perm-inv n64 | −0.108 | +0.133 [−0.168, +0.449] (1.0) | | | +0.101 |
| HGQ-r150 s0 / s1 / s2 | −0.124 / −0.151 / −0.111 | +0.118 / +0.091 / +0.131 (all 1.0) | | | +0.022 / −0.080 / −0.068 |
| float s0 / s1 | +0.101 / −0.255 | +0.342 [+0.007, +0.697] (0.41) / −0.013 (1.0) | | | +0.160 / +0.103 |
| untrained | +0.007 | – | | | +0.006 |

- **Official network.** Its TA response equals BLIND's within ±0.24. AWARE's excess over BLIND is +0.61.
- **Ratio to AWARE.** Network excess / AWARE excess = 0.04. The upper end of the network's CI (+0.27) is 0.44 of AWARE's +0.61.
- **Top calls.** Calls change for 2 of 83 jets under each edit (McNemar p 1.0).
- **All 8 trained networks.** Each excess over BLIND lies between −0.01 and +0.34, and none passes Holm.

### Replication of Wr − ND and the per-edit components (official network)

| set | n | W − N [CI], p (Holm, 24) | W − N − BLIND [CI], p (Holm) | $\Delta_W$ | $\Delta_N$ | $\Delta_W$ − BLIND [CI], p (Holm) | AWARE $\Delta_W$ − BLIND |
|---|---|---|---|---|---|---|---|
| T60 | 150 | −0.350 [−0.515, −0.171], 4e-4 (0.006) | −0.252 [−0.411, −0.089], 0.0034 (0.051) | −0.205 | +0.145 | −0.125 [−0.264, +0.017], 0.088 (1.0) | −0.656 |
| T100 | 496 | −0.432 [−0.514, −0.345], 1e-4 (0.002) | −0.120 [−0.207, −0.031], 0.0052 (0.073) | −0.145 | +0.287 | −0.075 [−0.159, +0.010], 0.086 (1.0) | −0.717 |
| T80 | 103 | −0.136 [−0.303, +0.030], 0.12 (1.0) | −0.010 [−0.182, +0.164], 0.91 (1.0) | −0.032 | +0.103 | +0.049 [−0.098, +0.207], 0.53 (1.0) | +0.322 |

- **BLIND / AWARE W − N.** BLIND: −0.098 / −0.311 / −0.125. AWARE: −0.754 / −1.105 / +0.189.
- **Side effects, W − N (Wr − ND).**
  - $\tau_{21}$: −0.051, −0.031, −0.006.
  - $\tau_{32}$: +0.031, +0.044, +0.020.
  - $m_{SD}$: +5.0, −0.9, +0.5 GeV.
  - $D$: matched (difference < 3e-5).
  - Widths: the W − N difference in $w_1$ is at most 0.0007. The largest single-edit width change is −0.0047 ($w_3$ under Wr at T60), against typical widths of 0.016–0.036.
- **Comparison with E42x.**
  - T60 and T100 W − N replicate (E42x −0.200 / −0.437).
  - T80 W − N is −0.136 with p 0.12 (E42x −0.214, p 0.0054). Its excess over BLIND is −0.01 (E42x −0.116).
- **T80 W − N across the 8 trained networks.** −0.136, −0.033, −0.127, −0.057, −0.019, −0.013, −0.118, −0.107. All are negative, as in E42x; none passes Holm (all Holm p 1.0, except perm-inv at 0.88).
- **Float retrains.** They again exceed BLIND at T60 / T100 (W − N − BLIND −0.39 to −0.53, Holm ≤ 0.009). Their W edit alone exceeds BLIND at T60 (−0.34 / −0.26, Holm 0.0024 / 0.0115), in the W-mass-reader direction. At T80, their W edit alone vs BLIND is +0.075 / +0.104 (n.s.), also in the reader direction.
- **Untrained redraw.** |W − N| ≤ 0.007 and |C| ≤ 0.007.

### Competing hypotheses

| hypothesis | prediction for TA ($C_{\rm bal}$ − BLIND) | prediction for $\Delta_W$ − BLIND | observed (official) | status |
|---|---|---|---|---|
| (i) W-mass reader at AWARE strength | ≈ +0.61 | T60/T100 ≈ −0.66/−0.72, T80 ≈ +0.32 | +0.02 [−0.21, +0.27]; −0.13 / −0.08 / +0.05 | excluded (TA upper CI 0.27 < AWARE lower CI 0.47) |
| (i′) weaker W-mass term (≤ 0.4 × AWARE) | 0 to +0.27 | small, reader signs | signs of $\Delta_W$ − BLIND match the reader on 3/3 sets on both samples, none significant | not excluded |
| (ii) side effects only (= BLIND) | 0 | 0 | 0.02; −0.13 / −0.08 / +0.05 | not excluded |
| (iii) E42x's "any rigid W-pair deformation lowers the score" | 0 (both copies are deformed equally) | negative in all three sets | $\Delta_W$ − BLIND is +0.05 / +0.07 at T80 | not supported: the extra negative W − N in E42x comes from $\Delta_N$ |

## Checks

- **Fresh sample.** Seed 4521, accuracy 80.60 %, identity rebuild 1.2e-4.
- **Untrained network.** |W − N| and |C| ≤ 0.007.
- **Positive control.** AWARE gives the reader signature in every contrast: TA +0.61 over BLIND; $\Delta_W$ − BLIND −0.66 / −0.72 / +0.32.
- **Widths.** Every edit changes subjet widths by at most 0.005 (typical widths are 0.016–0.036), because subjets are translated without resizing.
- **Memory.** Pod peak memory was 2,047 MiB against the 2 GiB limit, as in the smoke run. While it was at the limit, the official network step ran slowly (577 s at 0.1 cores). The run completed and the accuracy is in range; a rerun at this size should request more memory or use fewer jets.

## Limits

- **TA sample size.** n = 83, unbalanced (28 low, 55 high). The low side alone gives a network C of −0.55 [−0.99, −0.12] against BLIND −0.40, driven by the $m_{SD}$ drop of the toward edit.
- **Away targets.** Away moves reach 40–60 or 100–120 GeV, outside the natural W window. A reader whose response saturates far from 80 GeV would give a smaller C than one that responds linearly. AWARE, which saturates the same way, still gives +0.61.
- **Calibration.** There is no calibration of the network's overall gain against AWARE's ($k_{\rm DIL}$ was not interpretable in E42x). "Fraction of AWARE" therefore compares raw log-odds responses.
- **Sample overlap.** 3,399 rows are shared with E42x.

## Corrected reading

- **W-pair mass.** With subjet widths held, an off-peak W pair moved onto 80 GeV versus away by the same amount changes JEDI-linear's top log-odds by the same amount as a pair-blind GBDT: +0.02 [−0.21, +0.27] beyond BLIND, n 83, fresh sample. A pair-mass-aware GBDT gives +0.61 [+0.47, +0.74]. A W-pair-mass dependence of the pair-aware GBDT's strength is therefore excluded; one up to about 0.4 of it is not. The same holds for all 8 trained networks: excess over BLIND −0.01 to +0.34, none significant after Holm.
- **What W − N measures.** In E42x's W − N, the negative onto-peak value and the excess over BLIND come from the non-W control edit raising the network's score (by +0.05 to +0.18 more than BLIND), not from the W edit lowering it. On the fresh sample, the onto-peak W − N beyond BLIND is −0.01.
- **The W edit alone.** Beyond BLIND it gives −0.13 to −0.19 (60 GeV), −0.08 to 0.00 (100 GeV) and +0.05 to +0.07 (onto 80 GeV) on the two samples, none significant after Holm for the official network.
- **A20t wording.** "The top score does not read the W-pair mass at the strength of a pair-mass classifier; the remaining W − N comes from the non-W control edit and from subjet-shape side effects." The wording "rigid W-pair deformations lower it in either direction" is not supported.

## Explainer entry (for site/explain.py, Paper A tab, A20t)

**Question.** If a top jet's off-peak W pair is moved onto the W mass, does JEDI-linear's top score rise more than when the same pair is moved the same distance away?

**Background.**
- In a top jet, two of the three subjets come from the W, so their pair mass $m_W$ is near 80 GeV.
- E42x moved the W pair at fixed jet mass, $\tau_{32}$ and subjet widths, and compared the result with an edit of a different pair (N). That comparison mixes two things: the network's response to the W edit, and its response to the N edit.

**Method.**
1. Take tops whose W pair is off-peak: 60–70 or 90–100 GeV.
2. Make two copies of each jet. Slide whole subjets, with none resized, so that $m_W = 80.4$ in one copy and $m_W = 2m_{W,0} - 80.4$ in the other. Restore $m$ and $\tau_{32}$ in both.
3. Per jet, compute $C = s_{\rm toward} - s_{\rm away}$, with $s = \ell_t - \log\sum_{k \ne t} e^{\ell_k}$.
4. Average the low and high sides with equal weight.
5. Compare with gradient-boosted trees that do not see the pair masses (BLIND) and that do (AWARE).

**Result.**
- $C_{\rm bal}$ − BLIND = +0.02 [−0.21, +0.27] for JEDI-linear, against +0.61 [+0.47, +0.74] for AWARE (n 83).
- The 7 retrained networks give −0.01 to +0.34; the untrained network gives 0.007.

**Checks.**
- The rerun on a fresh sample reproduces E42x at 60 and 100 GeV.
- Split into its parts, E42x's W − N shows that its onto-peak sign comes from the control edit.

**Why it matters.** A three-prong top tagger could check whether two subjets form a W. Under these edits, JEDI-linear does not do that at the strength of a classifier given the pair masses. What it responds to is accounted for by jet-shape variables that contain no pair mass.

**New glossary terms** (for explain.GLOSSARY):
- **Toward-vs-away (TA) contrast:** the score difference between moving a W pair onto 80.4 GeV and moving it the same distance away, both at fixed $m$, $\tau_{32}$ and subjet widths.
- **Balanced mean:** the equal-weight mean over the low-$m_W$ and high-$m_W$ sides, which cancels any response that depends only on whether the pair was opened or closed.
