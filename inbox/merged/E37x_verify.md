# E37x verify — fresh-sample rerun and three added controls

Verifier report for node E37x ("What the h4 prong-sharing direction computes, and what carries the Z-jet response").
- Script: `interp/e37x_verify.py`. It imports E37x's code unchanged (edits, read-outs, interaction model, holds) and adds controls A–C below.
- Results: `results/jedi/e37x_verify.json`; smoke run `results/jedi/e37x_verify_smoke.json`.
- Logs: `logs/e37x-verify2.log` (final), `logs/e37x-verify.log` (first attempt), `logs/e37x-vsmoke.log`.
- Jobs:
  - e37x-vsmoke: 1 CPU / 2 GiB, 150 jets per class, passed.
  - e37x-verify: 1 CPU / 2 GiB. It was OOM-killed (exit 137) at about 1,590 s, after the edited jets were built.
  - e37x-verify2 (final): memory-trimmed, with a fixed malloc mmap threshold, at 1 CPU / 3 GiB. Wall 1,511 s, peak 3,071 MiB, 0.79 cores on average. The 3 GiB request follows the measured 2 GiB OOM; E37x itself ran at 3–4 GiB.

## Conclusion
On a new set of jets, most of E37x's numbers come back:
- The layer-4 "sharing" coordinate moves strongly when the momentum split between the two prongs is changed.
- It barely moves when the prong separation is changed at the same jet-mass change.
- Holding or copying it under the fixed-four-momentum edit gives the same pattern as before.

Two parts of the reading do not hold as stated:
- **Part of the response is prong width.** The split edit also changes how wide each prong is, because each prong's mass is held fixed. Widening the soft prong alone, with the split and the separation unchanged, moves the coordinate by about 0.4 of the full split-edit response.
- **The sign change near 85 GeV holds for Z jets only.** Pushing the coordinate directly inside the network flips its effect on the W/Z call near 85 GeV for Z jets. For W jets the flip comes only above about 95 GeV, so it is not a function of jet mass alone.

Still uncertain: the W jets above 85 GeV are few (73 and 13 jets in the two bins). In the 85–95 GeV bin the W slope is no larger in size than that of random directions.

## Definitions (used below)
- **h4**: the 64-unit fourth layer of JEDI-linear (3-feature, n64, fixed point).
- **v**: the 1-dim DAS direction at h4. DAS (distributed alignment search) is a learned direction whose interchange between two jets reproduces the network's output change.
- **u**: the coordinate $u=h_4\cdot v$. Its sign is set so that it correlates positively with $z$.
- **z**: the vector-$p_T$ fraction of the harder of the two exclusive-$k_T$ prongs.
- **$\Delta R_{12}$**: the rapidity-azimuth distance between the two prongs.
- **Margin**: $\ell_W-\ell_Z$, the W logit minus the Z logit. Positive means W-like.
- **The E37x edits**:
  - $E_z$: change $z$ at fixed $\Delta R_{12}$ and prong masses.
  - $E_{\Delta R}$: change the prong separation at fixed $z$, matched to $E_z$'s jet mass.
  - $E_m$: a uniform angular dilation, matched to $E_z$'s jet mass.
  - $E_{\rm rot}$: a rotation about the jet axis.
  - D(±0.3): a rest-frame decay-axis rotation at fixed jet four-momentum.
- **$R_z$**: $\langle\Delta u(E_z)-\Delta u(E_m)\rangle/(\beta_z\langle\Delta z\rangle)$. Here $\beta_z$ is the natural slope of $u$ on $z$ at fixed $(m, p_T)$. $R_z=1$ means $u$ moves as natural jets of the same mass and $p_T$ imply.
- **Hold / copy**: keep a subspace of h4 at the unedited value (hold), or take only that subspace from the edited jet (copy). The removed or reproduced share is relative to the free margin change.
- **Holm**: step-down family-wise correction.
- **Rank p**: $(1+\#\{\text{random}\ge\text{observed}\})/(1+n_{\rm random})$.

## Procedure
**Samples.**
- Test T: the W+Z jets of `jedi_core.sample(3000, 4747)`, N = 6,000.
- Fit F: the W+Z jets of `sample(3000, 4748)` minus the T rows, N = 5,639.
- E37x used seeds 3737/3738. Over all five classes, 868 of the 15,000 T rows fall in E37x's T and 855 in E37x's F.

**Checks.**
- 5-class accuracy on the 15,000 stored T jets: 80.40 %.
- The exact h4 tail reproduces the logits (max difference 0).

**Refitted $v$.** E37x code, 400 steps.
- Interchange frac on T: 0.809 (E37x 0.824).
- Dilation-pair frac: 0.021.
- $|\cos|$ with the stored E37 direction: 0.81.
- $|\cos|$ with M2: at most 0.08.

**Edits.** E37x's `build_variants`, unchanged.
- Acceptance for $\delta z$ = −0.15 / −0.075 / +0.075 / +0.15: 0.818 / 0.874 / 0.556 / 0.364 (E37x: 0.82 / 0.87 / 0.55 / 0.36).
- Invariant errors: $\Delta R_{12}$ 6e-16; prong mass 5e-9 of $m$.

**Added controls.**
- **(A) Direct steering of $u$.**
  - Edit: $h_4' = \mathrm{clip}(h_4+\delta\,v)$ on unedited T jets, with $\delta\in\{-1,-0.5,+0.5,+1\}$ in $u$ units. The clip is to the next layer's representable range. The tail is exact.
  - Per jet: slope = least-squares $\Delta$margin per unit $\delta$ through 0.
  - Read-out: the mean per unedited-jet mass bin and class, with a 2,000 jet-bootstrap CI.
  - Null: 200 random unit directions in the 55 live h4 units, with the same $\delta$.
  - Statistic for the sign change: $d$ = slope(85–95) − slope(75–85).
  - Why: E37x's per-bin slopes $a_{\rm bin}$ are regressions of $\Delta$margin on $\Delta u$ across edit types. Those edits change many h4 directions and the jet mass together. Steering changes only $u$.
- **(B) Decomposition of $E_z$**, at $\delta z=\pm0.15$ on $E_z$-accepted jets.
  - Why: $E_z$ holds each prong's mass while changing its $p_T$, so it must rescale each prong's angular width.
    - At +0.15, the median width factors are 0.83 and 1.65 for prong labels 0 and 1. Label 0 is usually the harder prong.
    - At −0.15, they are 1.21 and 0.64.
  - **$E_{pT}$**: only $E_z$'s prong-$p_T$ rescaling. $z$ changes as in $E_z$. Prong angular shapes are kept, so prong masses scale with prong $p_T$. $\Delta R_{12}$ is exact.
  - **$E_{\rm wid}$**: only $E_z$'s width rescaling. Each prong's offsets about its own axis are scaled by the factor $E_z$ used, with constituent $p_T$ unchanged. $z$ and $\Delta R_{12}$ are exact; prong masses change.
  - Each comes with its own dilation matched to its jet-mass change ($E_{mP}$, $E_{mW}$), the same construction as $E_m$.
- **(C) Dilation scan.** All angles about the jet axis are scaled by $f$ = 0.80 … 1.20 on all T jets. This gives the shape of $u$ against angular scale. E37x's $E_m$ rows raise $u$ for both signs of the mass change.

**E37x read-outs rerun.** Calibrated slopes, naming, interaction model, and holds under D(+0.3) and $E_z$(+0.15). The holds use 500 random subspaces per dimension, so the Holm floor is 16/501 = 0.032.

## Results
**1. Replication of E37x.** W / Z, mean [95 % CI]; E37x values in parentheses.

$\Delta u$ at $\delta z$ = +0.15:

| edit | W | Z |
|---|---|---|
| $E_z$ | +1.44 [1.32, 1.55] (1.34) | +1.77 [1.66, 1.88] (1.47) |
| $E_{\Delta R}$ | +0.34 | +0.33 |
| $E_m$ | +0.24 | +0.20 |
| $E_{\rm rot}$ | 0.00 | 0.00 |

$\Delta u$ at $\delta z$ = −0.15:

| edit | W | Z |
|---|---|---|
| $E_z$ | −0.92 (−0.80) | −1.03 (−0.88) |
| $E_{\Delta R}$ | +0.38 | +0.50 |
| $E_m$ | +0.48 | +0.59 |

$\Delta$margin at +0.15:

| edit | W | Z |
|---|---|---|
| $E_z$ | −2.18 | +5.89 |
| $E_{\Delta R}$ | −2.07 | +6.39 |
| $E_m$ | −1.96 | +6.53 |

Calibrated slopes:
- $\beta_z$ = 5.40 [5.16, 5.63] (5.32).
- $R_z$ = 2.14 [2.02, 2.26] (1.96 [1.85, 2.08]). Per bin, 65–75 to 95–105 GeV: 3.96, 1.91, 1.79, 2.50.
- $R_{\Delta R}/R_z$ = 0.077 [0.071, 0.083] (0.079).
- $R_{\rm rot}$ = 0.006 [0.000, 0.011].

Naming ($s_q$ pooled; per edit type $E_z$ / $E_{\Delta R}$ / D; held-out partial $R^2$):

| $q$ | $s_q$ pooled | $E_z$ | $E_{\Delta R}$ | D | partial $R^2$ |
|---|---|---|---|---|---|
| $z$ | 1.56 | 0.99 | – | 2.14 | 0.38 |
| $z(1-z)$ | 1.78 | 1.13 | – | 2.34 | 0.48 |
| $\Delta R_{12}$ | 1.80 | – | 1.65 | 1.80 | 0.39 |
| $D_2$ | 1.29 [1.21, 1.36] | 1.01 | −0.78 | 1.53 | 0.39 |
| $\tau_{21}$ | 2.28 | 1.69 | −4.53 | 4.48 | 0.24 |

- The pre-set rule again selects $D_2$.
- $D_2$ and $\tau_{21}$ again have the wrong sign on $E_{\Delta R}$.
- Per-edit $s_q$ values for edits that do not move $q$ have denominators near 0 and are not interpretable. The json prints values up to $10^{13}$ there; the table shows them as –.

Holds under D(+0.3) (free $\Delta$margin W −1.52, Z +1.61):

| set | W | Z | rank p W / Z | Holm W / Z |
|---|---|---|---|---|
| hold $v$ | 0.43 [0.41, 0.45] | −0.09 [−0.11, −0.06] | – | 0.032 / 1.0 |
| hold M2 | 0.10 | 0.04 | – | – |
| hold $v$+M2 | 0.61 | 0.21 [0.18, 0.24] | 0.002 / 0.010 | 0.032 / 0.050 |
| copy $v$+M2 | 0.95 | 1.33 | – | – |
| copy $v$ (outside the Holm family) | 0.90 | 0.95 [0.92, 0.98] | – | – |

- Random 3-dim hold medians: W 0.01, Z −0.06.
- McNemar on Z call changes, hold $v$+M2 against free: 292 / 303 (p 0.68).

Holds under $E_z$(+0.15) (free W −2.18, Z +5.89):

| set | W | Z |
|---|---|---|
| hold $v$ | 0.25 | −0.07 |
| hold M2 | 0.42 | 0.67 |
| hold $v$+M2 | 0.84 | 0.70 |
| copy $v$+M2 | 1.02 | 0.98 |

All Holm p = 0.032 (the floor) except hold $v$ for Z.

Interaction model (all edits pooled):
- W: $c$ = −0.023 [−0.082, 0.009], p 0.19 (E37x −0.032, p 0.044).
- Z: $c$ = +0.059 [0.012, 0.129], p 0.014 (E37x +0.14, p < 0.0005).
- Per-bin $a_{\rm bin}$ (65–75 / 75–85 / 85–95 / 95–105):
  - W: −0.16 / −0.30 / +0.19 / +0.20.
  - Z: −0.13 / −0.10 / +0.52 / +0.38.
  - The same sign pattern as E37x.

**2. (A) Direct steering of $u$.** $\Delta$margin per unit $\Delta u$; positive = toward W.

| unedited mass bin | W: n, slope [CI] | rank p vs random | Z: n, slope [CI] | rank p vs random |
|---|---|---|---|---|
| 65–75 GeV | 202, −0.53 [−0.58, −0.48] | 0.015 | 51, −0.28 [−0.35, −0.21] | 0.035 |
| 75–85 GeV | 2,495, −0.73 [−0.74, −0.71] | 0.015 | 163, −0.45 [−0.52, −0.39] | 0.23 |
| 85–95 GeV | 73, **−0.21 [−0.33, −0.09]** | 0.73 | 2,437, +0.65 [0.63, 0.67] | 0.27 |
| 95–105 GeV | 13, +0.61 [0.43, 0.79] | 0.005 | 141, +0.87 [0.80, 0.93] | 0.005 |
| $d$ = (85–95) − (75–85) | +0.51 [0.40, 0.64] | 0.095 | +1.10 [1.03, 1.17] | 0.005 |

- Random-direction slopes: the 95 % range is wide, e.g. [−0.87, +0.95] for W 85–95. For $d$ it is [−0.65, +0.53] (W) and [−0.76, +0.60] (Z).
- The stored E37 direction gives the same pattern: W 85–95 −0.22 [−0.32, −0.11]; Z 75–85 −0.45, Z 85–95 +0.55.
- H0: the steering slope has the same sign in 75–85 and 85–95 GeV. H1: the sign changes.
  - Z: the sign changes; both CIs exclude 0, and $d$ against random has rank p 0.005.
  - W: H0 is not rejected. The 85–95 slope is negative with a CI excluding 0, and $d$ against random has rank p 0.095. The sign changes only in 95–105 GeV (13 jets).
- At equal mass the classes differ. In 85–95 GeV the slopes are W −0.21 and Z +0.65, with non-overlapping CIs. The effect of $u$ on the call therefore depends on more than jet mass.
- The regression $a_{\rm bin}$ depends on which edit rows enter it:
  - Without the D rows: W 85–95 +0.31; Z 75–85 +0.08 [−0.02, 0.17].
  - D rows alone: W 85–95 +0.02 [−0.14, 0.20]; Z 75–85 −0.29.
  - E37x's positive W 85–95 slope therefore comes from the mass-changing edit rows, where $\Delta u$ and $\Delta\log m$ move together. It is not reproduced by changing $u$ alone.

**3. (B) Decomposition of $E_z$.** Mass-matched contrasts in $\Delta u$, W+Z, mean [CI].

| $\delta z$ | $E_z-E_m$ (full edit) | $E_{pT}-E_{mP}$ ($z$ only, prong shapes kept) | $E_{\rm wid}-E_{mW}$ (widths only, $z$ and $\Delta R_{12}$ fixed) |
|---|---|---|---|
| +0.15 (n 2,187) | +1.39 [1.31, 1.47] | +0.60 [0.57, 0.63] | +0.53 [0.48, 0.58] |
| −0.15 (n 4,906) | −1.51 [−1.57, −1.46] | −1.41 [−1.46, −1.36] | −0.14 [−0.15, −0.13] |

- Per class at +0.15:
  - $E_{\rm wid}-E_{mW}$: W +0.49, Z +0.57.
  - $E_{pT}-E_{mP}$: W +0.48, Z +0.72.
- $E_{\rm wid}$ alone (unmatched) changes $\Delta\log m$ by +0.06 at +0.15 and changes the margin by W −1.46 / Z +0.59.
- The $z$-only edit moves $u$ with the sign of $\Delta z$ in both directions.
- The width-only edit moves $u$ in the same direction as $E_z$. At +0.15 it accounts for 0.38 of the full contrast; at −0.15, for 0.09.
- With the realised $\Delta z$ of E37x (+0.129 / −0.125, taken as approximately equal here), the pooled $R$ from the $z$-only edit is ≈ 1.47. The full $E_z$ gives ≈ 2.1.
- Interpretation scoped to these edits: part of the factor ≈ 2 in $R_z$ is the width rescaling that $E_z$ needs to hold the prong masses. $u$ also responds to how widely the soft prong's particles are spread at fixed $z$ and $\Delta R_{12}$.

**4. (C) Dilation scan.** Mean $\Delta u$ for $f$ = 0.80 / 0.90 / 0.95 / 1.05 / 1.10 / 1.20:
- W: +0.21 / +0.08 / +0.03 / −0.03 / −0.04 / −0.03.
- Z: +0.15 / +0.05 / +0.01 / +0.02 / +0.06 / +0.21.

$u$ is non-monotone in angular scale for Z jets and responds with opposite sign to expansion in W and Z jets. The responses are small (≤ 0.21 for a 20 % dilation) against $E_z$'s 1.4–1.8.

E37x's $E_m$ rows (+0.18 to +0.59) are larger than the scan suggests at the same mean $\Delta\log m$. $E_m$'s per-jet factor is solved to $E_z$'s mass, which gives large dilations for lopsided jets.

## Code review of `interp/e37x_sharing_physics_jedi.py` and `e37x_plot.py`
No stage, mask, quantizer, padding or split-leakage bug was found:
- The exact tail clips to the d5 input range.
- F rows that are also in T are excluded (873 in this run).
- Padded slots are excluded via `nn` in `model_xn`, `axis` and `obs1`.
- Bootstrap weights are drawn over jets, with F resampled jointly for $\beta$.

Points that affect the reading:
- **$E_z$ changes more than $z$.** To hold the prong masses it rescales each prong's angular width; at +0.15 the median factor is 1.65 for one prong and 0.83 for the other. The report lists $\Delta R_{12}$ and the prong masses as held, but not this change. Control B shows that it moves $u$.
- **$a_{\rm bin}$ is not a causal estimate.** It is a regression across edit rows whose $\Delta u$ and $\Delta\log m$ are collinear. Its values change with the set of edits included, and in W 85–95 it has the opposite sign to direct steering.
- **The example jet in the figure** (panels a, b) has $z$ 0.94 → 0.97. The soft-prong fraction is clipped at 0.03, so the realised $\Delta z$ is 0.03, not 0.15. Its call change (margin +1.97 → −0.02) comes with a 79.0 → 58.5 GeV mass drop, which $E_m$ alone also produces on average. The jet does not illustrate a $z$ effect.
- **Panel (d)** plots the regression $a_{\rm bin}$ under the title "effect of the coordinate on the W/Z call". The steering slopes in section 2 measure that quantity directly and differ for W in 85–95 GeV.
- **The suptitle label "ΔR12 edit 0.15"** is $R_{\Delta R}$ = 0.155, not an edit size.
- **Per-edit naming slopes** for edits that do not move $q$ are printed in the json with near-zero denominators.

## Corrected reading
- **Response to the split.** The h4 coordinate $u$ moves with the sign of a change in $z$.
  - This holds with $\Delta R_{12}$ and the prong masses held: $R_z$ = 2.14 [2.02, 2.26] on a fresh sample.
  - It also holds with the prong angular shapes held instead: the $z$-only edit gives +0.60 / −1.41 after mass matching at $\delta z$ = ±0.15.
  - At the same jet-mass change, a separation change and a uniform dilation move $u$ by 0.077 of the $z$ effect, and a rotation does not move it.
- **Response to prong width.** $u$ also responds to the width of the prongs at fixed $z$ and $\Delta R_{12}$. The width rescaling that $E_z$ needs to hold the prong masses gives +0.53 at +0.15, 0.38 of the full $E_z$ contrast. Part of $R_z\approx2$ is this width response; with the $z$-only edit, $R\approx1.5$.
- **Naming.** As in E37x:
  - $z(1-z)$ has the largest natural partial $R^2$ (0.48).
  - $D_2$ and $\tau_{21}$ have the wrong sign on the separation edit.
  - No single observable fits every edit type.
- **Effect on the call.** Under direct steering of $u$:
  - **Z jets:** the effect flips from toward Z (75–85 GeV, −0.45) to toward W (85–95 GeV, +0.65).
  - **W jets:** the effect stays toward Z through 85–95 GeV (−0.21 [−0.33, −0.09]) and turns toward W only in 95–105 GeV (13 jets).
  - At equal mass, W and Z jets differ (85–95 GeV: −0.21 against +0.65), so the effect depends on more than jet mass.
  - E37x's "sign change between 75–85 and 85–95 GeV in both classes" comes from the edit-pooled regression and is not reproduced for W by changing $u$ alone.
- **Carriers.** Replicated.
  - Under D(+0.3), copying $v$ reproduces 0.95 of the Z response.
  - Holding $v$ removes −0.09 of it, and holding $v$+M2 removes 0.21 (Holm p 0.050 with 500 nulls).
  - Under $E_z$(+0.15), holding M2 removes 0.67 of the Z response.

## Explainer note (for site/explain.py; site/ is outside this verifier's write scope)
- **Question:** Does the h4 coordinate's effect on the call change sign with jet mass, and is its $z$ response a response to $z$ alone?
- **Method:**
  - Add $\delta v$ to h4 ($\delta=\pm0.5,\pm1$) and read the margin per mass bin, with random-direction controls.
  - Split $E_z$ into its $p_T$ part and its width part, each matched to its own jet-mass change by a dilation.
- **Result:** Tables 2 and 3 above.
- **Checks:** fresh seeds 4747/4748; tail exact; acceptance and invariants as in E37x.
- **Why it matters:** the A4s claim text should say "signed $z$-and-prong-width coordinate; effect on the call flips near 85 GeV for Z jets and above 95 GeV for W jets".
- **New glossary terms:**
  - *Steering slope*: margin change per unit of $u$ added directly at h4.
  - *$E_{pT}$ / $E_{\rm wid}$*: the two parts of $E_z$.
