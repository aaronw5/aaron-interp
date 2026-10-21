# E37x — What JEDI-linear's h4 prong-sharing coordinate computes, and what carries the Z-jet response

Node E37x (stage 3; claim A4s v2, D18; parent E37v). Status: done. There was one rerun, made to resolve the statistics rather than to fix a crash. With 200 random subspaces per null, the smallest possible rank p is 1/201 = 0.005, and Holm over 16 tests then cannot go below 0.08. The holds were therefore rerun with 2,000 random subspaces per dimension. Everything else in the rerun matches run 1 to the 4th decimal, which is jax floating-point variation between nodes.
- Script: `interp/e37x_sharing_physics_jedi.py`.
- Figure script: `interp/e37x_plot.py`, rendered locally.
- Jobs:
  - e37x-smoke: 1 CPU / 2 GiB.
  - e37x-run: 1 CPU / 3 GiB; wall 1,291 s; peak 3,071 MiB; 0.87 cores on average.
  - e37x-run2 (final): 1 CPU / 4 GiB; wall 1,650 s; peak 3,991 MiB; 0.60 cores on average.

  In both full runs the cgroup peak equals the request. This pattern suggests page cache filling the limit rather than heap use. Neither run was killed. The request was raised above the 2 GiB exempt size because E37's identical DAS code measured 2.8 GiB.
- Results: `results/jedi/e37x.json` (final) and `results/jedi/e37x_run1.json` (200-null holds).
- Figure: `results/jedi/e37x.png`.
- Logs: `logs/e37x-run2.log`, `logs/e37x-run.log`, `logs/e37x-smoke.log`.

## Conclusion
The sharing coordinate in the network's fourth layer follows how unevenly the jet's momentum is split between its two prongs. When only the split is changed (prong separation and prong masses held), the coordinate moves strongly. When the jet mass is changed by the same amount through the prong separation or a uniform stretch, it barely moves. Shape variables that mix split and separation (D2, τ21) predict the wrong sign for the separation edit, so the coordinate is closer to the split variable z than to them. Its numerical slope is about twice what natural jets imply, so it is not a calibrated z read-out.

How the coordinate pushes the W/Z call depends on jet mass. Below about 85 GeV, a more lopsided split moves the call toward Z; above 85 GeV it moves it toward W, and this holds in both W and Z jets.

Under the fixed-four-momentum edit, holding the coordinate together with the mass subspace removes only a quarter of the Z-jet response. Copying the coordinate alone into an unedited jet reproduces all of it. The Z response therefore reaches the output through this coordinate, but other directions in the same layer carry it as well. Which other directions those are, and why the slope is about 2, is still open.

## Question
E37/E37v found a 1-dim direction $v$ at h4 (the 64-unit fourth layer) whose interchange carries about 0.8 of the W−Z margin response to the prong-sharing edit D. Holding it removes half of the W response and none of the Z response. Three questions follow:
1. Which physics quantity does the coordinate $u = h_4\cdot v$ compute: $z$, $z(1-z)$, $\Delta R_{12}$, the prong-mass ratio, or a shape variable (D2, τ21)?
2. How does its effect on the call depend on jet mass?
3. What carries the Z-jet response?

Predictions:
- **Thesis (coordinate = $z$):**
  - $E_z$ moves $u$ with calibrated slope $R_z\approx 1$, and $E_{\Delta R}$ gives $\approx 0$.
  - Holding $v$ and M2 jointly removes ≥ 0.7 of the Z response, against a random 2-3-dim median ≤ 0.2.
  - Copying both reproduces ≥ 0.7.
- **Alternative A (shape variable mixing $z$ and $\Delta R_{12}$):** $E_{\Delta R}$ moves $u$ about as much as $E_z$.
- **Alternative B (another route):** the joint hold removes no more than random subspaces do.

## Definitions
- **$z$**: the vector-$p_T$ fraction of the harder of the two exclusive-$k_T$ prongs, $z=p_{T,\rm hard}/(p_{T,\rm hard}+p_{T,\rm soft})$. The value 0.5 is a symmetric split; values near 1 are lopsided.
- **$\Delta R_{12}$**: the rapidity-azimuth distance between the two prong four-vectors.
- **$m_{\rm hi}$, $m_{\rm lo}$**: the prong masses.
- **E-scheme axis**: the direction of the summed four-momentum.
- **Margin**: $\ell_W-\ell_Z$, the difference of the W and Z output logits. Positive means W-like.
- **$u$**: the coordinate $h_4\cdot v$. Its SD over unedited test W+Z jets is 1.40; the margin SD is 4.67.
- **DAS** (distributed alignment search): a learned subspace whose interchange between two inputs reproduces the network's output change.
- **IIA** (interchange-intervention accuracy): the fraction of call-changing pairs where the patched network gives the target's call.
- **Hold of a subspace $Q$**: $h' = h_e + (h_0-h_e)QQ^\top$, i.e. the subspace is kept at the unedited jet's value while the rest of h4 comes from the edited jet.
- **Removed share**: $1 - \langle\Delta\text{margin}_{\rm held}\rangle/\langle\Delta\text{margin}_{\rm free}\rangle$, per class.
- **Copy (sufficiency)**: $h'=h_0+(h_e-h_0)QQ^\top$, i.e. only the subspace is taken from the edited jet.
- **Reproduced share**: $\langle\Delta\text{margin}_{\rm copy}\rangle/\langle\Delta\text{margin}_{\rm free}\rangle$.
- **Holm**: step-down family-wise correction.
- **McNemar**: exact binomial test on paired call changes; it counts jets whose call changes under one condition only.

## Procedure
**Network.** Official JEDI-linear, 3-feature n64, fixed point. The tail from h4 is exact and clips to the next layer's representable range (x10 JNet, as in E37).

**Samples.**
- Test T: the W+Z jets of `jedi_core.sample(per_class=3000, seed=3737)`, 6,000 jets.
- Fit F: the W+Z jets of `sample(3000, 3738)` minus the rows in T, 5,657 jets.
- The 5-class accuracy is computed on all 15,000 stored T jets.

**Prongs.** Two exclusive-$k_T$ subjets (p = 1, E-scheme, every constituent assigned), as in E35/E37. Constituents are massless. Edited jets keep each particle's prong label. Network inputs are rebuilt about each jet's own axis and $p_T$-sorted, for edited and unedited jets alike.

**The coordinate.**
- $v$ is refitted on F with E37's code: 1-dim DAS at h4 on the D(±0.15, ±0.3) self pairs and cross pairs, using the MSE of the margin, 400 Adam steps and model selection on 20 % of jets.
- Alternative direction $v_{dm}$: the difference of means of $h_4$(D+0.3) and $h_4$(D−0.3) on F (E37v).
- Both directions are signed so that $u$ correlates positively with $z$ on F.

**Edits on T.** Each uses $\delta z\in\{-0.15,-0.075,+0.075,+0.15\}$.
- **$E_z(\delta z)$ — change the split at fixed separation and prong masses.**
  - Every constituent of the softer prong has its $p_T$ multiplied by $(1-z_1)W/w_{\rm soft}$, and every constituent of the harder prong by $z_1W/w_{\rm hard}$. Here $z_1=z+\delta z$, $w$ is a prong's vector $p_T$, and $W=w_{\rm soft}+w_{\rm hard}$ is unchanged.
  - Each prong's angular offsets from its own axis are then rescaled (bisection) until its mass equals the original.
  - Each prong is translated back onto its original axis, so $\Delta R_{12}$ is exact.
  - The whole jet is translated in $(y,\phi)$ so the E-scheme axis is where it was.
  - Held exactly: $\Delta R_{12}$ (max error 7e-16), both prong masses (5e-9 of $m$), and $W$.
  - **Deviation from the plan:** the jet mass cannot also be held. At fixed prong masses, $\Delta R_{12}$ and $p_T$ it is fixed by $m^2\simeq m_a^2/z+m_b^2/(1-z)+z(1-z)p_T^2\Delta R_{12}^2$.
  - Mean $\Delta\log m$ is +0.139 / +0.091 / −0.090 / −0.171 for $\delta z$ = −0.15 / −0.075 / +0.075 / +0.15. The realised $\Delta z$ is −0.125 / −0.069 / +0.070 / +0.129 (clipped at $z_{\rm soft}\ge 0.03$ and $\le 0.5$).
  - All controls are therefore matched to $E_z$'s jet-mass change on each jet.
- **$E_{\Delta R}(\delta z)$ — the prong-separation control.** Both prongs are moved rigidly along the line joining them, which keeps $z$ and the prong masses exact. The separation scale $\lambda$ is solved so the jet mass equals $m(E_z)$ (mass-match error 4e-13). This gives the same $\Delta m$ as $E_z$ with $\Delta R_{12}$ moved instead of $z$.
- **$E_m(\delta z)$ — a real mass change.** All angles are dilated about the axis by $k$ (bisection) so the mass equals $m(E_z)$. $z$ is unchanged; $\Delta R_{12}$ and the prong masses scale by $k$. Also used: fixed dilations $E_{mf}$ by factors 0.95 / 0.98 / 1.02 / 1.05 for calibration.
- **$E_{\rm rot}(\delta z)$ — the matched no-information edit.** A rigid rotation of the jet about its axis in $(\eta,\phi)$ by $\pm\theta$, with $2\sin(\theta/2)\langle r\rangle_{p_T}$ equal to $E_z$'s $p_T$-weighted mean constituent displacement. $z$, $\Delta R_{12}$ and the prong masses are unchanged up to 4e-4, and $m$ up to 0.6 %.
- **D(±0.3)** (E35): a rest-frame decay-axis rotation at fixed jet four-momentum. $z$ and $\Delta R_{12}$ move together (D+0.3: $\Delta z$ +0.11, $\Delta R_{12}$ +0.06–0.07).
- **Acceptance.** Reclustering the $E_z$ and $E_{\Delta R}$ jets must move less than 1 % of the jet scalar $p_T$ between prongs (the H50 rule). Both prong centres must lie within $\Delta R$ 0.75 of the axis. A jet enters a $\delta z$ cell only if $E_z$, $E_{\Delta R}$ and $E_m$ are all accepted.
  - Accepted fractions for $\delta z$ = −0.15 / −0.075 / +0.075 / +0.15: 0.82 / 0.87 / 0.55 / 0.36.
  - Making the hard prong harder moves soft particles across the reclustering boundary, so the +0.15 cell is a selected subset of 2,173 jets.

**Calibrated slope.**
$$R_z=\frac{\langle \Delta u(E_z)-\Delta u(E_m)\rangle}{\beta_z\,\langle\Delta z\rangle},\qquad R_{\Delta R}=\frac{\langle \Delta u(E_{\Delta R})-\Delta u(E_m)\rangle}{\beta_z\,\langle\Delta z(E_z)\rangle},\qquad R_{\rm rot}=\frac{\langle\Delta u(E_{\rm rot})\rangle}{\beta_z\,\langle\Delta z(E_z)\rangle}$$
- $\beta_z$ is the natural slope of $u$ on $z$ among unedited F jets at fixed $(m,p_T)$. Both $u$ and $z$ are taken minus their means over the 50 nearest neighbours in standardised $(\log m,\log p_T)$. It is computed per mass bin for the per-bin $R$.
- In words, $R_z$ is the change of the coordinate under a pure-$z$ edit, after subtracting its change under a real mass change of the same size, divided by the change that natural jets of the same mass and $p_T$ show for the same $\Delta z$. $R_z=1$ means a calibrated $z$ read-out.
- Values are pooled over $\delta z$ with the sign of $\delta z$. CIs come from 2,000 jet-bootstrap resamples of T, with F jets resampled jointly for $\beta$.

**Naming.**
- For each $q\in\{z, z(1-z), \Delta R_{12}, m_{\rm lo}/m_{\rm hi}, D_2, \tau_{21}\}$, fit an OLS on unedited F jets: $u\sim 1+q+\log m+(\log m)^2+\log p_T$.
- Over all edit rows ($E_z$, $E_{\Delta R}$, $E_m$, $E_{mf}$, D), $\Delta u_{\rm net}=\Delta u-$ (the model's mass and $p_T$ part).
- The calibrated slope is $s_q=\sum\Delta u_{\rm net}\,\beta_q\Delta q/\sum(\beta_q\Delta q)^2$. Also reported: $s_q$ per edit type, and the held-out partial $R^2$ of $q$ beyond $(m,p_T)$ on T.
- Pre-set rule: the quantity whose $s_q$ CI lies above 0 and whose $|\log s_q|$ is smallest.

**Interaction model.**
- Rows are (jet, edit) over all edits, 48,680 W rows and 49,636 Z rows.
- $\Delta\text{margin}=c_0+a\,\Delta u+b\,\Delta\log m+c\,\Delta u\,(m_0-85\,{\rm GeV})/10$, fitted per class. Per-mass-bin slopes $a_{\rm bin}$ are fitted with $b$.
- 2,000 jet bootstraps. H0: $c=0$ (two-sided bootstrap p).

**Carriers.**
- Holds are applied at h4 on T under D(+0.3) (6,000 jets) and $E_z$(+0.15) (2,173 jets).
- The four sets:
  - (i) hold $v$;
  - (ii) hold M2. This is E37's refitted M2 DAS subspace at h4: k = 2, from X10/E25w-style do(M2) interchange, stored in `e37_das_M.npz`. **Deviation:** it is the stored E37 refit, not new E25w directions.
  - (iii) hold $v$ + M2 (3 dims);
  - (iv) copy $v$ + M2.
- Null: 2,000 random subspaces of matching dimension in the 55 live h4 units (E26v rule).
- Rank p = (1 + #random ≥ observed)/2,001. Holm over 4 sets × 2 classes × 2 edits = 16 tests.
- McNemar compares free and held W/Z call changes.

**Floor.** An untrained redraw, `Jedi(untrained=True, seed=0)`, with calls on logits minus the mean logits. It is run with its own $v_{dm}$ through the same read-out code.

## Results
**Checks.**
- 5-class accuracy on 15,000 stored T jets: 80.71 %; F: 80.88% (both within 80.6–81.3 %).
- The exact h4 tail reproduces the logits (max difference 0).
- The refitted $v$ reproduces E37 on T: interchange frac 0.824 (E37 0.825), IIA 0.33 (E37 0.41), dilation-pair frac 0.018.
- $|\cos|$ with the stored E37 direction is 0.67. With $v_{dm}$ it is 0.54; with M2 the maximum is 0.06.

**1. Effect of each edit on $u$ and on the margin** (all T jets accepted at $\delta z$ = ±0.15; mean, 95 % CI):

| edit ($\delta z$ = +0.15; $\Delta\log m=-0.17$) | $\Delta u$ W | $\Delta u$ Z | $\Delta$margin W | $\Delta$margin Z | W→Z | Z→W |
|---|---|---|---|---|---|---|
| $E_z$ ($z$ +0.129, $\Delta R_{12}$ fixed) | +1.34 [1.24, 1.43] | +1.47 [1.37, 1.56] | −2.23 [−2.34, −2.12] | +6.08 [5.85, 6.31] | 0.5 % | 48.5 % |
| $E_{\Delta R}$ ($\Delta R_{12}$ −0.03, $z$ fixed) | +0.28 [0.25, 0.31] | +0.25 [0.20, 0.28] | −2.10 [−2.21, −1.99] | +6.53 [6.30, 6.77] | 0.4 % | 55.0 % |
| $E_m$ (dilation, same $\Delta m$) | +0.18 [0.14, 0.21] | +0.12 [0.07, 0.16] | −1.98 [−2.09, −1.87] | +6.65 [6.43, 6.89] | 0.2 % | 56.2 % |
| $E_{\rm rot}$ | +0.00 | −0.01 | +0.02 | +0.01 | 0.8 % | 1.3 % |

| edit ($\delta z$ = −0.15; $\Delta\log m=+0.14$) | $\Delta u$ W | $\Delta u$ Z | $\Delta$margin W | $\Delta$margin Z | W→Z |
|---|---|---|---|---|---|
| $E_z$ ($z$ −0.125) | −0.80 [−0.84, −0.76] | −0.88 [−0.92, −0.84] | −3.98 [−4.16, −3.81] | +0.49 [0.38, 0.62] | 35.1 % |
| $E_{\Delta R}$ ($\Delta R_{12}$ +0.044) | +0.38 [0.33, 0.42] | +0.45 [0.41, 0.50] | −4.10 [−4.27, −3.95] | +0.44 [0.34, 0.54] | 38.2 % |
| $E_m$ | +0.45 [0.41, 0.50] | +0.53 [0.49, 0.58] | −3.94 [−4.10, −3.79] | +0.54 [0.44, 0.64] | 37.4 % |

The coordinate moves with the sign of $\delta z$ under $E_z$. Under $E_{\Delta R}$ and $E_m$, which have the same $\Delta m$ and fixed $z$, it moves by a similar small amount in the same direction.

The network's margin is set mainly by the mass change. At matched $\Delta m$, $E_z$, $E_{\Delta R}$ and $E_m$ give margins within 0.6 of each other, against mass-driven changes of 2–7. McNemar on call changes, $E_z$ against $E_{\Delta R}$ (jets changed only by $E_z$ / only by $E_{\Delta R}$):
- +0.15 W: 75 / 26 (p 1e-6).
- +0.15 Z: 57 / 41 (p 0.13).
- −0.15 W: 91 / 90 (p 1.0).
- −0.15 Z: 124 / 82 (p 0.004).

$E_z$ against $E_{\rm rot}$: 475 / 5 (W, +0.15) and 897 / 13 (Z, +0.15), both p < 1e-10.

**2. Calibrated slopes** (W+Z, pooled over $\delta z$; $\beta_z$ = 5.32 [5.08, 5.56] $u$ per unit $z$):

| jet-mass bin | $R_z$ | $R_{\Delta R}$ | $R_{\Delta R}/R_z$ | $R_{\rm rot}$ |
|---|---|---|---|---|
| all | 1.96 [1.85, 2.08] | 0.155 [0.143, 0.168] | 0.079 [0.074, 0.085] | −0.003 [−0.008, 0.003] |
| 65–75 GeV | 2.74 [2.16, 3.52] | 0.12 [0.02, 0.22] | 0.045 [0.009, 0.077] | 0.002 |
| 75–85 GeV | 1.81 [1.65, 1.98] | 0.14 [0.12, 0.15] | 0.075 | 0.000 |
| 85–95 GeV | 1.67 [1.54, 1.81] | 0.13 [0.11, 0.14] | 0.076 | −0.004 |
| 95–105 GeV | 1.79 [1.27, 2.60] | 0.28 [0.20, 0.42] | 0.156 [0.124, 0.200] | −0.009 |

- H0 $R_{\Delta R}=0$: p < 0.0005 (bootstrap). H0 $R_z=R_{\Delta R}$: p < 0.0005.
- By class: W $R_z$ 1.81 [1.68, 1.95], Z 2.10 [1.96, 2.25].
- With the difference-of-means direction $v_{dm}$: $R_z$ 1.80 [1.72, 1.89], $R_{\Delta R}$ 0.129 [0.120, 0.138].
- Caveat on $E_m$: the $E_{\Delta R}-E_m$ contrast moves $\Delta R_{12}$ by only 0.002–0.007 beyond $E_m$, because $E_m$ also stretches $\Delta R_{12}$. The comparison that separates $z$ from $\Delta R_{12}$ at equal $\Delta m$ is therefore $E_z$ against $E_{\Delta R}$ in table 1.
- $R_z$ is about 2 rather than 1. A pure-$z$ edit moves $u$ about twice as much as the natural $z$–$u$ relation at fixed $(m,p_T)$ predicts. With the natural-regression mass correction instead of $E_m$, the $E_z$ rows give $s_z$ = 0.84 and $s_{z(1-z)}$ = 0.99 (see 3). The magnitude therefore depends on how the jet-mass change is subtracted.

**3. Naming the quantity** ($v$; calibrated slope $s_q$ over all edit rows, per edit type, and held-out partial $R^2$ on T beyond $(m,p_T)$):

| $q$ | $s_q$ (all edits) | $s_q$ on $E_z$ | on $E_{\Delta R}$ | on D | edit $R^2$ | held-out partial $R^2$ |
|---|---|---|---|---|---|---|
| $z$ | 1.43 [1.39, 1.47] | 0.84 | – ($\Delta z$ = 0) | 2.01 | 0.08 | 0.33 |
| $z(1-z)$ | 1.65 [1.61, 1.68] | 0.99 | – | 2.20 | 0.20 | 0.42 |
| $\Delta R_{12}$ | 1.70 [1.67, 1.73] | – ($\Delta\Delta R$ = 0) | 1.55 | 1.70 | 0.43 | 0.37 |
| $m_{\rm lo}/m_{\rm hi}$ | 38 [−104, 176] | – | – | – | −0.11 | 0.07 |
| $D_2$ | 1.05 [0.99, 1.11] | 0.84 | −0.56 | 1.24 | 0.38 | 0.32 |
| $\tau_{21}$ | 2.11 [2.03, 2.18] | 1.53 | −4.05 | 4.13 | 0.12 | 0.19 |

"–" means the edit does not move $q$, so the per-edit slope is undefined. The untrained floor's held-out partial $R^2$ is ≤ 0.01 for every $q$.
- The pre-set rule selects $D_2$, with pooled $s$ 1.05.
- Per edit type, $D_2$ and $\tau_{21}$ predict the opposite sign for $E_{\Delta R}$ (−0.56, −4.05). These variables change strongly when the prongs are separated at fixed $z$, while $u$ does not, so neither is a consistent read-out.
- $z(1-z)$ has the largest natural partial $R^2$ (0.42; with $v_{dm}$, 0.50) and slope 0.99 on the pure-$z$ rows. On the D rows, where $z$ and $\Delta R_{12}$ co-move at fixed mass, its slope is 2.2.
- No quantity has slope ≈ 1 on every edit type.

**4. Mass dependence of the coordinate's effect.** Interaction model, slope of $\Delta$margin per unit $\Delta u$ ($\Delta\log m$ held in the fit):

| class | $a$ | $c$ (per 10 GeV of $m_0$) | p($c$=0) | $a_{65-75}$ | $a_{75-85}$ | $a_{85-95}$ | $a_{95-105}$ |
|---|---|---|---|---|---|---|---|
| W | −0.29 [−0.33, −0.26] | −0.032 [−0.084, −0.001] | 0.044 | −0.26 [−0.34, −0.18] | −0.34 [−0.37, −0.32] | +0.26 [0.13, 0.41] | +0.28 [0.10, 0.70] |
| Z | +0.44 [0.41, 0.48] | +0.14 [0.10, 0.19] | < 0.0005 | −0.20 [−0.31, −0.11] | −0.19 [−0.27, −0.11] | +0.57 [0.54, 0.60] | +0.52 [0.39, 0.66] |

In both classes the per-bin slope is negative below 85 GeV (a larger $u$ moves the call toward Z) and positive above 85 GeV (toward W). The linear term $c$ underfits this step; the per-bin slopes are the direct measurement.

This pattern is consistent with the opposite W and Z responses to D(+0.3), which leaves mass fixed: W jets sit mostly below 85 GeV and Z jets above. It is also consistent with $E_z$(+0.15) moves Z jets slightly toward Z beyond $E_m$ (Δmargin $E_z-E_m$ = −0.58): the edit lowers their mass by 17 %, into the region where the slope is negative.

Caveat: the per-bin slopes pool all edit types. The D rows have the largest $\Delta u$ at fixed mass (+2.1 to +2.4 at dc = +0.3), so they weigh heavily in $a_{\rm bin}$.

**5. Carriers of the response** (removed share for holds, reproduced share for the copy; 95 % CI; rank p against 2,000 random subspaces of the same dimension; Holm over 16):

| edit | set | W | Z | random median W / Z | Holm p W / Z |
|---|---|---|---|---|---|
| D(+0.3) (free Δmargin W −1.56, Z +1.58) | (i) hold $v$ | 0.49 [0.47, 0.51] | −0.10 [−0.12, −0.07] | 0.00 / −0.02 | 0.008 / 1.0 |
| | (ii) hold M2 | 0.10 [0.08, 0.12] | 0.03 [0.01, 0.06] | 0.01 / −0.05 | 0.13 / 0.37 |
| | (iii) hold $v$ + M2 | 0.58 [0.57, 0.60] | 0.25 [0.22, 0.28] | 0.01 / −0.07 | 0.008 / 0.018 |
| | (iv) copy $v$ + M2 | 0.97 [0.95, 0.99] | 1.28 [1.25, 1.32] | 0.18 / 0.28 | 0.018 / 0.018 |
| $E_z$(+0.15) (free W −2.23, Z +6.08) | (i) hold $v$ | 0.24 [0.22, 0.25] | −0.07 [−0.08, −0.06] | 0.00 / 0.00 | 0.012 / 1.0 |
| | (ii) hold M2 | 0.43 [0.41, 0.45] | 0.69 [0.67, 0.70] | 0.01 / 0.01 | 0.008 / 0.008 |
| | (iii) hold $v$ + M2 | 0.76 [0.74, 0.78] | 0.76 [0.74, 0.78] | 0.03 / 0.01 | 0.008 / 0.008 |
| | (iv) copy $v$ + M2 | 1.02 [1.01, 1.03] | 0.98 [0.97, 0.98] | 0.07 / 0.09 | 0.008 / 0.008 |

Outside the Holm family:
- **Copy $v$ alone under D(+0.3):** W 0.90 [0.88, 0.93], Z 0.96 [0.93, 0.99] (rank p 0.004 / 0.029).
- **Copy M2 alone under D(+0.3):** W 0.32, Z 1.17.
- **$v_{dm}$ under D(+0.3):** hold W 0.46, Z 0.12; with M2, W 0.71, Z 0.38.
- **$v_{dm}$ + M2 under $E_z$(+0.15):** W 0.76, Z 0.87.
- **Random 3-dim hold, 97.5 % quantile:** under D, W 0.13 / Z 0.13; under $E_z$, 0.18 / 0.09.

McNemar on W/Z call changes (jets changed under free only / under hold only):
- D(+0.3), hold $v$: W 580 / 85 (p < 1e-10); Z 143 / 492 (p < 1e-10). Holding $v$ increases the Z→W flips from 14.8 % to 26.4 %.
- D(+0.3), hold $v$ + M2: Z 278 / 282 (p 0.90).
- $E_z$(+0.15), hold $v$ + M2: Z 748 / 53 (p < 1e-10).

The holds under D(+0.3) replicate E37v (hold $v$: W 0.47, Z −0.08) on new jets.

**6. Untrained floor.**
- 7 live h4 units.
- $\beta_z$ = 0.045 [−0.003, 0.085]. $R_z$ and $R_{\Delta R}$ are not resolved (CI −4.6 to 10.5). $R_{\Delta R}/R_z$ = 0.94 [0.55, 2.25], so the untrained net shows no separation of $z$ from $\Delta R_{12}$.
- Partial $R^2$ ≤ 0.01 for every quantity.
- Margin responses in margin-SD units, untrained against trained:
  - $E_z$(+0.15): W +0.10 / Z +0.18 untrained; W −0.48 / Z +1.30 trained.
  - $E_{\Delta R}$(+0.15): W +0.06 / Z +0.09 untrained; W −0.45 / Z +1.40 trained.

## Hypotheses: which predictions hold
- **$E_z$ slope ≈ 1 (thesis): not met as pre-specified.** $R_z$ = 1.96 [1.85, 2.08]. With the natural-regression mass correction the $E_z$ slope is 0.84 ($z$) or 0.99 ($z(1-z)$). The sign follows $z$ in every mass bin.
- **$E_{\Delta R}\approx 0$ (thesis): small, not zero.** $R_{\Delta R}$ = 0.155 [0.143, 0.168], which is 0.079 [0.074, 0.085] of the $E_z$ effect.
- **Alternative A (shape variable; $E_{\Delta R}$ ≈ $E_z$): excluded.** The ratio is 0.079 with a CI upper bound of 0.085, and the natural slopes of $D_2$ and $\tau_{21}$ predict the wrong sign for $E_{\Delta R}$.
- **Joint hold ≥ 0.7 of the Z response (thesis):**
  - Not met under D(+0.3): 0.25 [0.22, 0.28].
  - Met under $E_z$(+0.15): 0.76 [0.74, 0.78], but M2 alone gives 0.69 there because $E_z$ changes the jet mass by −17 %.
- **Copy ≥ 0.7 (thesis): met under both edits.** D: 0.97 / 1.28; $E_z$: 1.02 / 0.98. Random 3-dim medians are ≤ 0.28.
- **Alternative B (joint hold ≤ random): excluded under both edits** (Holm p ≤ 0.018). However, 0.75 of the Z response to D(+0.3) remains with $v$ and M2 held, while copying $v$ alone reproduces 0.96 of it.
- **H1 $c\neq0$ (mass dependence): supported.**
  - Z: $c$ = +0.14, p < 0.0005.
  - W: $c$ = −0.032, p 0.044.
  - The per-bin slope changes sign between 75–85 and 85–95 GeV in both classes.

## Reading (scoped to these measurements)
- The h4 coordinate responds to the prong momentum fraction at fixed prong separation and prong masses. At the same jet-mass change, it does not distinguish a change of separation from a uniform angular stretch.
- Its size per unit $z$ is about twice the natural relation at fixed mass and $p_T$. One possible source is that the $E_z$ edit also reshapes the individual constituent $p_T$ (rank windows); this was not tested here.
- Of the named observables, $z(1-z)$ explains the most natural variance and matches the pure-$z$ edit. No observable matches every edit type.
- The coordinate's push on the W/Z margin changes sign near 85 GeV. Raising $u$ moves jets with mass below about 85 GeV toward Z and jets above it toward W. The same sign change appears in both classes, so it is a function of jet mass, not of the true class.
- Under the fixed-four-momentum edit, the Z response is reproduced by copying $v$ but is not removed by holding $v$, or $v$ with M2. Other h4 directions carry it in parallel.

## Explainer entry (for site/explain.py, Paper A tab; math in LaTeX)
**Question.** JEDI-linear's fourth layer (h4) has one learned direction $v$ that carries most of the network's response when the momentum split between a jet's two prongs becomes more lopsided. What does $u=h_4\cdot v$ measure? And how does it change the W/Z decision?

**Background.**
- A W or Z decaying to two quarks gives a jet with two prongs. Let $z$ be the harder prong's share of the prong transverse momentum, and $\Delta R_{12}$ the angular distance between the prongs.
- At fixed prong masses, the jet mass obeys $m^2\simeq m_a^2/z+m_b^2/(1-z)+z(1-z)\,p_T^2\Delta R_{12}^2$. So in natural jets of a given mass, $z$ and $\Delta R_{12}$ are tied together (correlation 0.79 at fixed $m$ and $p_T$).
- Shape variables such as $D_2$ and $\tau_{21}$ mix both.
- Separating "the network reads $z$" from "the network reads separation or shape" therefore needs edits that move one at a time.

**Method.**
1. Split each W/Z test jet into two $k_T$ prongs.
2. **$E_z$:** rescale the constituent $p_T$ of each prong so $z\to z+\delta z$. Restore each prong's mass by rescaling its internal spread, and put the prongs back at their original positions, so $\Delta R_{12}$ and the prong masses are exact. The jet mass changes as a consequence (formula above).
3. **$E_{\Delta R}$:** move the prongs apart or together at fixed $z$ and prong masses, until the jet mass equals $m(E_z)$.
4. **$E_m$:** stretch all angles until the mass equals $m(E_z)$.
5. **$E_{\rm rot}$:** rotate the jet about its axis; nothing physical changes.
6. Compare the change of $u$ across the four edits on the same jets. Calibrate it against natural jets of the same mass and $p_T$:
$$R_z=\frac{\langle\Delta u(E_z)-\Delta u(E_m)\rangle}{\beta_z\langle\Delta z\rangle},\qquad \beta_z=\frac{\partial u}{\partial z}\Big|_{m,p_T}\ \text{(natural jets)}.$$
7. Measure the coordinate's effect on the margin $\ell_W-\ell_Z$ per jet-mass bin.
8. Hold (keep at the unedited value) or copy (take from the edited jet) $v$ and the mass subspace M2 in h4, against 2,000 random subspaces of the same dimension.

**Result** (figure `results/jedi/e37x.png`). Panels (a, b) show one real W jet before and after $E_z$(+0.15). This jet is very lopsided ($z$ 0.94 → 0.97, $\Delta R_{12}$ fixed at 0.311), its mass drops from 79.0 to 58.5 GeV, and its call changes from W to Z. Panel (c) shows $\Delta u$ per mass bin for the four edits; panel (d) shows the per-bin margin slope $a$.
- At equal mass change, $E_z$ moves $u$ by +1.3 to +1.5 (W/Z) for $\Delta z$ = +0.13. $E_{\Delta R}$ and $E_m$ move it by +0.1 to +0.3, and $E_{\rm rot}$ by 0.00.
- $R_z$ = 1.96 [1.85, 2.08]. The separation edit gives 0.08 of the $z$ effect.
- $D_2$ and $\tau_{21}$ predict the wrong sign for the separation edit. $z(1-z)$ explains the most natural variance (partial $R^2$ 0.42).
- Per unit $u$, the W−Z margin moves toward Z for jets below about 85 GeV and toward W above, in both W and Z jets.
- Under the fixed-four-momentum edit, copying $v$ alone reproduces 0.96 of the Z response. Holding $v$ together with M2 removes 0.25.

**Checks.**
- Untrained network: no $z$ read-out (partial $R^2$ ≤ 0.01).
- Rotation edit: zero response.
- The refitted direction reproduces E37's interchange frac (0.824).
- The holds replicate E37v on new jets.
- Accuracy 80.7 %.

**Why it matters.**
- The network's sharing variable is a momentum-fraction variable, not a separation or $D_2$-type shape variable.
- Its use is conditional on mass: a lopsided split moves the call away from the class that the mass alone would choose. This is consistent with the opposite W and Z responses seen in E35.
- The Z response passes through this coordinate but is not carried by it alone.

**Glossary additions.**
- **$E_z$, $E_{\Delta R}$, $E_m$, $E_{\rm rot}$:** the four matched edits defined above.
- **$R_z$:** the calibrated slope (read-out change under a pure-$z$ edit, minus that under a matched real mass change, divided by the natural-jet slope at fixed $m$ and $p_T$ times $\Delta z$).
- **$z(1-z)$:** the prong-sharing factor in the two-prong mass formula.
- **Hold / copy share:** the fraction of the margin change removed when a subspace is kept at its unedited value, or reproduced when only that subspace is taken from the edited jet.

## Issues / limitations
- Jet mass is not fixed under $E_z$ (physically impossible together with the other constraints). All $z$-specific statements rest on the matched-$\Delta m$ comparisons with $E_{\Delta R}$ and $E_m$.
- $E_m$ also stretches $\Delta R_{12}$, so $E_{\Delta R}-E_m$ is a weak separation contrast. The $R_{\Delta R}$ values in units of $\Delta R_{12}$ (`R_dR_Edr` in the json) have denominators of 0.002–0.007 and are not interpretable.
- Acceptance at $\delta z$ = +0.15 is 36 % (reclustering migration), so that cell is a selected subset.
- The per-edit-type naming slopes have no CIs for individual edit types. Only the pooled $s_q$ is bootstrapped, and $\beta_q$ is held fixed at its fit-sample value there.
- M2 is E37's stored refit, not new E25w directions.
- The run needed 3–4 GiB; the cgroup peak equals the request in both runs.
