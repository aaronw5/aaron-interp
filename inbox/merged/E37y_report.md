# E37y — class gate on the h4 sharing coordinate's effect at equal mass; prong width at fixed prong mass; redundant Z carrier

Node E37y (parent E37xv; stage 3; claims A4s v3, D18). Script `interp/e37y_sharing_gate_jedi.py`; figure script `interp/e37x_plot.py`
(redraw of `results/jedi/e37x.png`). Results `results/jedi/e37y.json` (smoke: `e37y_smoke.json`); log `logs/e37y-run.log`.
Job `e37y-run`, 1 CPU / 5 GiB, wall 2,099 s, peak memory 1,859 MiB (37 % of request), 0.96 cores average. Smoke job `e37y-smoke`
(500 jets per class) ran first; after it, arm 3's calibration was changed from a sign-pooled to a per-edit-size ratio (reason in arm 3).

## Conclusion

Pushing the network's "momentum-sharing" coordinate in its fourth layer moves W jets toward the Z call and Z jets toward the W
call even when the W and Z jets have the same jet mass, so the effect depends on the true class and not only on mass. About
two-thirds of that W/Z difference passes through three last-layer neurons that switch on in particular jet-mass windows: holding
them fixed removes 0.63 of it, more than random holds of the same size, but the margin over random holds does not survive the
correction for seven tested holds (corrected p 0.10), and a third of the difference remains. Separately, making the prongs wider
while keeping their masses fixed lowers the coordinate about as much as the natural width–coordinate relation predicts, the
opposite sign to the earlier width edit in which prong masses rose with width. A different one-dimensional direction, orthogonal to
both the coordinate and the mass subspace, reproduces the Z-jet response to a more lopsided split on average (0.95 of it).

## Terms

- **h4**: the 64-unit fourth layer of JEDI-linear (after the second pooling S). **z**: the 16 pre-ReLU units of the last hidden
  layer; logits $=$ d7(ReLU($z$)). **W−Z margin** $= \ell_W - \ell_Z$ (log-odds between the W and Z outputs); > 0 toward W.
- **v**, **u**: $u = h_4 \cdot v$. $v$ = 1-dim **DAS** direction (distributed alignment search: a learned unit vector in the 55 live
  h4 units such that copying the component of h4 along $v$ from a jet edited by D(dc) into the unedited jet reproduces the
  edited jet's W−Z margin; E37 code, 400 Adam steps), refitted here on F, oriented so that corr($u$, $z_h$) > 0 on F (0.60).
  $|\cos|$ with E37's stored direction 0.82; largest cosine with the mass subspace M2 0.03. s.d. of $u$ on T: $\mathrm{sd}_u = 1.63$.
- **D(dc)**: E35's edit: rest-frame rotation of the decay axis so that $|\cos\theta^*|$ (0 = equal momentum sharing) changes by
  dc, jet four-momentum and both prong masses fixed. **$z_h$** = vector-pT fraction of the harder of the two exclusive-kT prongs.
- **Ez** (E37x): harder-prong fraction $z_h \to z_h + \delta z$ at fixed prong separation $\Delta R_{12}$ and prong masses.
- **M2**: E37's stored 2-dim DAS subspace at h4 that carries the network's M2-like mass variable. **M0**: same for M0.
- **IIA** (interchange-intervention accuracy): among pairs whose W/Z call changes under the edit, the share in which the patched
  network gives the edited jet's call. **frac**: mean patched margin change / mean full margin change.
- **$W_p$** (prong width) $= \sum_i p_{T,i} r_i / \sum_i p_{T,i}$, $r_i$ = distance in $(y, \phi)$ of particle $i$ to the
  four-vector axis of its own prong. **lead** = leading-particle pT / jet scalar pT.
- **R_w** = (mean change of $u$ under the width edit minus under its control) / ($\beta_w$ × mean change of $W_p$, edit minus
  control); $\beta_w$ = natural partial slope of $u$ on $W_p$ across unedited F jets with $\log m$, $(\log m)^2$, $\log p_T$, $z_h$,
  $z_h^2$, $\Delta R_{12}$ and both prong masses as regressors. R_w = 1: the network responds to the edited width as it does to the
  same width difference between real jets of equal $z_h$, $\Delta R_{12}$ and prong masses.
- **Holm**: step-down correction of $p$-values for the number of tests in a family. **McNemar**: exact binomial test on paired
  call changes (jets whose call changes under one edit and not the other, and vice versa).
- **rank p**: $(1 + \#\{\text{null} \ge \text{observed}\}) / (1 + n_{\text{null}})$.

## Samples and checks

- T = W+Z jets of `sample(3000, 5757)` (6,000; 3,000 W, 3,000 Z); F = W+Z jets of `sample(3000, 5758)` minus the 914 rows also in T
  (5,654). Inputs rebuilt about each jet's own E-scheme axis, pT-sorted.
- 5-class accuracy on the 15,000 stored T jets **81.03 %** (F 80.22 %), inside the 80.6–81.3 % band.
- Exact tail from h4 written for this node (d5 → d6 → z override → ReLU → d7, the official fixed-point kernels and quantizers):
  max |Δlogit| vs `JNet.exact` 0.0, vs the stored forward pass 0.0; holding all 16 z units at their unsteered values returns the
  unsteered logits exactly (0.0) while h4 is shifted. All 16 z units and 55 of 64 h4 units are live.
- F interchange check of the refitted $v$ (self D pairs): frac 0.83, IIA 0.35 (E37: 0.83 / 0.41).
- Jet-mass counts on T (W / Z): 80–85 GeV 1,185 / 105; 85–90 78 / 943; 90–95 11 / 1,508; 95–100 5 / 116. The matched comparison
  rests mainly on the 80–90 GeV bins.

## Arm 1 — steering $u$ at equal mass

**Procedure.** For every T jet, h4 is replaced by $h_4 + c\,\mathrm{sd}_u\,v$, $c \in \{\pm0.5, \pm1, \pm2\}$, clipped to the next
layer's input range (units cannot go below 0), and the exact tail gives the new margin. Per jet, slope = least squares through 0 of
the margin change on $c\,\mathrm{sd}_u$ (log-odds per unit of $u$). Mass matching: the W−Z difference of the mean slope in 5-GeV bins
80–100 GeV, averaged with weights $n_W n_Z/(n_W+n_Z)$ — equivalent to reweighting W and Z jets to a common mass distribution.
H0: $\Delta = 0$ (equal W and Z slopes at matched mass). H1: $\Delta \ne 0$. Competing reading "it was mass": $\Delta \to 0$ under
finer matching. Control: 200 random unit directions in the live h4 units steered by the same multiples of their own s.d.

**Numbers** (95 % CI from 2,000 jet bootstrap resamples):

| 80–100 GeV, matched | W slope | Z slope | $\Delta$ = W − Z |
|---|---|---|---|
| 5-GeV bins (primary) | −0.226 [−0.260, −0.192] | +0.129 [+0.090, +0.167] | **−0.355 [−0.408, −0.303]** |
| per bin 80–85 (n 1,185 / 105) | −0.448 | −0.173 | −0.276 [−0.350, −0.201] |
| per bin 85–90 (n 78 / 943) | −0.086 | +0.415 | −0.501 [−0.584, −0.412] |
| per bin 90–95 (n 11 / 1,508) | +0.544 | +0.665 | −0.121 [−0.229, −0.004] |
| per bin 95–100 (n 5 / 116) | +0.393 | +0.689 | −0.297 [−0.517, −0.084] |
| 1-GeV bins (12 bins used) | | | −0.207 [−0.262, −0.156] |
| linear model, W−Z at 90 GeV | | | −0.404 [−0.477, −0.324] |

- Permutation test (W/Z labels shuffled within 1-GeV bins, 2,000 permutations): p = 0.0005 (the floor 1/2001). Bootstrap p < 0.0005.
- Random directions (per s.d. step of each coordinate): $v$ gives $\Delta = -0.58$; random 2.5–97.5 % range [−0.30, +0.40], median
  |Δ| 0.11; rank p (|Δ|) 0.015. 12 % of random directions move matched W and Z jets in opposite directions.
- The response is not linear in $c$ (matched W / Z mean margin change): $c = +2$: −1.79 / +0.44; $+1$: −0.95 / +0.12; $+0.5$: −0.51 /
  −0.02; $-0.5$: +0.15 / −0.06; $-1$: −0.03 / −0.29; $-2$: −0.49 / −0.46. The through-zero slope is dominated by $c = \pm 2$, where every
  jet has at least one h4 unit at the clip (0 floor). Positive pushes (toward a more lopsided split) carry the class difference.
- Per-z-unit decomposition (margin = $\sum_k c_k\,\mathrm{ReLU}(z_k)$ + const, $c_k$ = d7 weight to W minus to Z; descriptive,
  input quantizer residual +0.001): contributions to $\Delta$: z0 −0.122 [−0.144, −0.099], z11 −0.078, z14 −0.057, z7 −0.047, z15 −0.030,
  others |·| < 0.012.

**Reading.** At matched mass the steering slope differs between W and Z jets: H0 rejected (p 0.0005). The 1-GeV matching shrinks the
difference from −0.36 to −0.21, so part of the 5-GeV value is within-bin mass difference and part is not; the "it was mass"
alternative predicts 0 at fine matching and is excluded at the measured precision (CI [−0.26, −0.16]).

## Arm 2 — z holds (gate test)

**Procedure.** Arm 1 repeated with chosen z units held at their unsteered values (per-unit override of z before the ReLU; the rest of
the network unchanged). Removed share = 1 − Δ(held)/Δ(free). Named holds: z0 (mass-window unit, ~75 GeV; E11 MARS: hinges on
mass < 79.7 and < 91.1 GeV; active (ReLU > 0) in 98 % of W and 40 % of Z jets; d7 weight W−Z +2.12), z7 (94–98 GeV window; active
28 % W / 85 % Z; weight −1.53), z14 (94–98 GeV; 11 % W / 87 % Z; weight −1.38), singly, in pairs and jointly (7 tests, Holm).
Nulls: 200 random $d$-dim holds in the z space ($z' = z + (z_0 - z)P$, $P$ a random rank-$d$ projector); and every hold of $d$ other
units (13 singles, 78 pairs, 286 triples). H1 (gate): a z-window hold removes > 0.5 of Δ, more than random holds. Alternatives:
the difference persists under all z holds (gate upstream in h4/h5), or vanishes with mass matching (excluded in arm 1).

| hold | removed share [95 % CI] | Δ held | p (share ≤ 0.5) | rank p vs random $d$-dim | Holm (7) | rank p vs other units |
|---|---|---|---|---|---|---|
| z0 | 0.34 [0.31, 0.38] | −0.233 | 1.0 | 0.040 | 0.17 | 0.071 (floor 1/14) |
| z7 | 0.13 [0.04, 0.20] | −0.310 | 1.0 | 0.20 | 0.45 | 0.14 |
| z14 | 0.16 [0.13, 0.19] | −0.298 | 1.0 | 0.15 | 0.45 | 0.14 |
| z0+z7 | 0.47 [0.39, 0.54] | −0.188 | 0.81 | 0.035 | 0.17 | 0.013 |
| z0+z14 | 0.50 [0.46, 0.54] | −0.177 | 0.44 | 0.025 | 0.15 | 0.013 |
| z7+z14 | 0.29 [0.20, 0.36] | −0.253 | 1.0 | 0.16 | 0.45 | 0.025 |
| **z0+z7+z14** | **0.63 [0.55, 0.70]** | −0.131 [−0.148, −0.114] | **0.002** | 0.015 | **0.10** | 0.0035 |

- Random-hold quantiles (median / 95 %): 1-dim 0.03 / 0.28; 2-dim 0.08 / 0.42; 3-dim 0.15 / 0.52. Best other-unit hold: single z11
  0.22 (a mass-threshold unit; E11: hinges on $\sqrt{M_0 M_2}$ < 81 and < 93 GeV), pair 0.30, triple 0.34.
- Under the joint hold the matched slopes are W −0.18, Z −0.05 (free: −0.23 / +0.13): the Z-side push toward W disappears.

**Reading.** Holding the three named mass-window units removes 0.63 of the W−Z difference (> 0.5 at bootstrap p 0.002; larger than
every one of the 286 holds of three other units). Against random 3-dim holds in z space the rank p is 0.015, and 0.10 after Holm over
the 7 named holds, so the comparison with random holds of the same dimension is not significant after correction. 0.37 of the
difference (Δ −0.13) remains with all three held, which is the prediction of the "partly upstream" alternative; the decomposition
assigns part of it to z11 and z15. These are holds of the last hidden layer: every effect on the logits passes through z, so the
measurement locates where the class dependence is expressed (which z units' on/off state differs between W and Z jets of equal
mass), not where it is first computed.

## Arm 3 — prong width at fixed prong mass

**Procedure** (T jets; 0.65 / 0.69 accepted for α = −0.5 / +0.5; rejections are kT-2 reclustering moving ≥ 1 % of jet pT).
- **Ws(α)**: inside each prong with ≥ 3 constituents, each constituent's pT × $\exp(\alpha (r_i - \langle r\rangle)/r_{\mathrm{rms}})$
  ($r_i$ = distance to the prong axis; pT-weighted mean / rms over the prong): α = +0.5 moves pT from the prong core to its periphery,
  α = −0.5 the reverse. Then 4 iterations of: rescale the prong's pT to its original vector pT ($z_h$ exact), translate the prong
  back onto its original axis ($\Delta R_{12}$ exact), scale its offsets about the axis to restore its mass (bisection). Measured
  max changes: $z_h$ 0, $\Delta R_{12}$ 0, prong mass 1.6e-4 (relative), jet mass and pT 0 (prong four-vectors unchanged). Mean
  $W_p$ change: α = +0.5 +0.0042 (+33 %); α = −0.5 +0.0012 (+11 %) — at fixed prong mass $W_p$ rises for both signs.
- Controls: **Erot** = rigid rotation of the jet about its axis with the same pT-weighted particle displacement (no physics
  change); **Eshuf** = the same pT factors randomly permuted within each prong, then the same re-solving (pT spectrum changed,
  no systematic width change). Comparison edit **Ewid** = each prong's offsets scaled by a common factor chosen so $W_p$ moves as in
  Ws (prong masses rise with it, +33 % / +11 %; jet mass +4.4 % / +1.1 %), with **EmW** = dilation matched to Ewid's jet mass.
- $\beta_w$ = −83.9 [−91.5, −76.5] $u$ per rad of $W_p$ (F jets, 500 bootstraps; s.d. of $W_p$ 0.0094, so −0.79 $u$ per s.d.); kNN
  (log m, log pT only) slope −8.4. Joint natural fit with lead: $\beta_{W_p}$ −85.9, $\beta_{\mathrm{lead}}$ −0.26.
- Per α (not pooled over signs, because $W_p$ rises for both). After the smoke run the pooled-by-sign ratio was replaced by this
  per-α ratio; no other change.

| edit vs control | α | Δu [95 % CI] | Δ$W_p$ | R_w [95 % CI] |
|---|---|---|---|---|
| Ws vs Erot | +0.5 (pT outward) | −0.284 [−0.304, −0.266] | +0.0042 | **0.80 [0.72, 0.89]** |
| Ws vs Eshuf | +0.5 | −0.354 [−0.389, −0.313] | +0.0036 | 1.17 [1.02, 1.34] |
| Ws vs Erot | −0.5 (pT inward) | +0.520 [+0.468, +0.569] | +0.0012 | −5.3 [−6.1, −4.6] |
| Ws vs Eshuf | −0.5 | +0.548 [+0.494, +0.595] | +0.0010 | −6.3 [−7.3, −5.5] |
| Ewid vs EmW | +0.5 | +0.359 [+0.336, +0.381] | +0.0035 | −1.22 [−1.36, −1.10] |
| Ewid vs EmW | −0.5 | +0.118 [+0.108, +0.127] | +0.0011 | −1.30 [−1.46, −1.17] |

- Raw Δu: Erot +0.000 (both α); Eshuf −0.03 / +0.07 (α −/+), although Eshuf(+0.5) raises lead by +0.148; Ws changes lead by
  +0.024 (α −0.5) / −0.016 (+0.5). The joint (W_p, lead) prediction does not account for Ws(−0.5) (observed/predicted −4.9).
- W−Z margin change, Ws minus Eshuf (W jets / Z jets): α = +0.5: +0.88 / −0.94 (both classes move toward their own label);
  α = −0.5: −0.25 / +0.76 (both move toward the other label), the same pattern as a push of $u$ down / up. McNemar on call changes Ws vs Eshuf: α +0.5 W 47:103 (p 6e-6), Z 69:133 (p 8e-6); α −0.5 W 117:33, Z 184:41
  (p < 1e-6).

**Reading.** Moving pT outward inside each prong at fixed $z_h$, $\Delta R_{12}$, prong masses and jet four-momentum lowers $u$ by
0.80 of what the natural width–$u$ relation predicts for the realised width change (rotation control ≈ 0). The prong-width
comparison edit in which prong masses rise with width (the E37xv "width-only" edit) moves $u$ the other way (+0.36), so the
upward width response seen in E37xv follows the co-moving prong masses (or the jet-mass-matched remainder), not width at fixed
prong mass. Pulling pT inward (α = −0.5) raises $u$ by +0.52 while $W_p$ barely changes and lead changes little; neither $W_p$ nor
lead accounts for it, and the quantity responsible is not identified.

## Arm 4 — redundant Z carrier

**Procedure.** DAS at h4 for $k$ = 1, 2, 3 restricted to the complement of span($v$, M2) (projector applied before orthonormalising),
trained on D(+0.3) self pairs of 2,815 F Z jets (target: edited margin), tested on 3,000 T Z jets (free margin change +1.61; 422
call-changing pairs). Nulls: 1,000 random $k$-dim subspaces of the same complement (Holm over 3 on IIA). Holds on T: the edited jet's
component in the subspace set back to the unedited value.

| subspace | frac [95 % CI] | random frac 50 / 95 % | IIA [95 % CI] | random IIA 50 / 95 % | Holm p (IIA) |
|---|---|---|---|---|---|
| copy $v$ (reference) | 0.92 | | 0.27 | | |
| copy $v$ + M2 (reference) | 1.23 | | 0.49 | | |
| $Q_1$ ⊥ ($v$, M2) | 0.95 [0.92, 0.98] | 0.06 / 0.26 | 0.12 [0.09, 0.16] | 0.04 / 0.08 | 0.010 |
| $Q_2$ | 0.94 [0.91, 0.97] | 0.11 / 0.32 | 0.19 [0.15, 0.23] | 0.06 / 0.10 | 0.008 |
| $Q_3$ | 1.02 [0.99, 1.05] | 0.16 / 0.44 | 0.37 [0.32, 0.41] | 0.07 / 0.13 | 0.003 |

- Hold $v$ + M2 removes 0.07 of the Z response. Hold $v$ + M2 + $Q_k$ removes 0.26 / 0.37 / 0.61 ($k$ = 1/2/3); random holds of the same
  dimension: 95 % quantile 0.10 / 0.11 / 0.14 (rank p ≤ 0.01). Largest singular value of $Q_k^\top$M0 (cosine with E37's M0 subspace):
  0.08 / 0.17 / 0.22.

**Reading.** The average Z-jet response to a more lopsided split can be copied in full through a direction orthogonal to both $v$ and
the M2 mass subspace (and nearly orthogonal to M0), but per jet it reproduces the call change in only 0.12 (k = 1) to 0.37 (k = 3) of
call-changing pairs. Removing $v$, M2 and a 3-dim complement together removes 0.61 of the response; with $v$ + M2 alone, 0.07. The Z
response is carried by at least two sets of h4 directions that can each produce it.

## Figure

`results/jedi/e37x.png` (redrawn): (a, b) a real W jet (81.9 GeV, 64 / 36 % prong-pT split) before and after Ez(+0.15): harder prong
64 → 79 % of prong pT, $\Delta R_{12}$ 0.135 unchanged, jet mass 81.9 → 82.3 GeV, $u$ +0.69 → +2.04, W−Z margin +3.97 → +2.71 (call
stays W); chosen as the accepted 75–85 GeV W jet with realised $\Delta z_h \ge 0.12$ and the smallest mass change among 1,516
candidates. (c) E37x's Δu by mass bin for Ez, Edr, Em, Erot (unchanged). (d) E37y arm 1: steering slope per 5-GeV bin for W and Z jets
with 95 % intervals and jet counts; shaded the 80–100 GeV matched window. Checked: readable, physical units, one message.

## Hypotheses summary

- H0 (equal W and Z slopes at matched mass): rejected, Δ −0.355 [−0.408, −0.303], permutation p 0.0005; at 1-GeV matching −0.21.
- "It was mass" (Δ → 0 with matching): excluded at 1-GeV matching (CI excludes 0).
- H1 (gate: a z-window hold removes > 0.5, more than random): the joint z0+z7+z14 hold removes 0.63 (> 0.5, p 0.002; above all 286
  other-unit triples, p 0.0035); vs random 3-dim holds rank p 0.015, Holm 0.10 — not significant after correction. No single named unit
  removes > 0.5 (z0 0.34).
- "Gate upstream / difference persists under all z holds": 0.37 of Δ remains under the joint hold; partly supported.
- Width (D18 / A4s v3): at fixed prong mass, widening lowers $u$ with R_w 0.80 [0.72, 0.89]; the E37xv width response (u up)
  has the sign of the prong-mass co-movement. Pulling pT inward raises $u$ by an amount not explained by width or leading fraction.

## Limits

- The steering slope is a through-zero fit over $c = \pm0.5..\pm2$ s.d.; the response is asymmetric in the sign of $c$ and at $c = \pm2$
  every jet has h4 units at the 0 floor. The W side of the 90–100 GeV bins has 16 jets.
- Holds in z cannot separate "gate" from "readout": every path to the logits passes through z.
- z-unit activity fractions above are over all jets, not at matched mass.
- One seed pair (5757 / 5758); no untrained or retrained-network comparison in this node.

## Explainer entry (for site/explain.py, paper A tab)

- **Question.** When the network's sharing coordinate $u$ is pushed, why do W and Z jets of the same mass respond differently, and is
  the coordinate's width sensitivity a width effect or a prong-mass effect?
- **Background.** $u = h_4\cdot v$ tracks the harder prong's momentum fraction $z_h$ (E37x). E37xv found that pushing $u$ moves Z jets
  toward W above about 85 GeV but W jets only above 95 GeV. The last hidden layer z contains units that switch on in jet-mass windows
  (E11): z0 near 75 GeV, z7 and z14 near 94–98 GeV.
- **Method.** (1) Add $c\,\mathrm{sd}_u\,v$ to h4 and record the change in $\ell_W - \ell_Z$; compare W and Z jets in the same 5-GeV (and
  1-GeV) mass bins: $\Delta = \sum_b w_b(\bar s_{W,b} - \bar s_{Z,b})/\sum_b w_b$, $w_b = n_W n_Z/(n_W+n_Z)$. (2) Repeat with z0, z7,
  z14 frozen at their unpushed values; removed share $= 1 - \Delta_{\text{held}}/\Delta$. (3) Widen each prong by moving pT outward,
  then restore $z_h$, $\Delta R_{12}$ and prong masses exactly; $R_w = \overline{\Delta u}/(\beta_w\,\overline{\Delta W_p})$.
  (4) Search for a direction orthogonal to $v$ and M2 that copies the Z-jet response.
- **Result.** $\Delta = -0.36$ log-odds per unit $u$ (1-GeV matching −0.21); freezing z0+z7+z14 removes 0.63 of it; widening at fixed
  prong mass gives $R_w = 0.80$; a direction ⊥ ($v$, M2) copies 0.95 of the mean Z response. Figure: `e37x.png` panel (d).
- **Checks.** Permutation within 1-GeV bins; 200 random h4 directions; 200 random z holds and all other-unit holds; Holm over 7;
  rotation and pT-shuffle controls; 1,000 random complement subspaces.
- **Why it matters.** The coordinate's effect on the call is not a single mass-dependent rule: which last-layer mass-window units are
  on differs between W and Z jets of equal mass, and that state sets much of the response.

New glossary terms for `explain.GLOSSARY`: **prong width $W_p$** (pT-weighted mean distance of particles to their own prong axis);
**R_w** (as above); **z hold** (last-hidden-layer units frozen at their unpushed values during steering); **matched contrast Δ**.
