# E37y verify: class gate on the h4 sharing-coordinate slope, prong width at fixed prong mass, Z carrier

Verifier node for E37y. Script `interp/e37y_verify.py` (imports the E37y functions unchanged). Job `e37y-verify`, 1 CPU / 2 GiB,
wall 511 s; smoke job `e37y-verify-smoke` (300 jets per class) ran first. Results `results/jedi/e37y_verify.json`
(`$AI_RESULTS/jedi/e37y_verify.json`; smoke `e37y_verify_smoke.json`), log `logs/e37y-verify.log`.

## Conclusion

On new jets, W and Z jets of the same mass again respond differently when the network's "momentum-sharing" coordinate is pushed.
When W and Z jets are also matched on how strongly the network already favours W over Z before the push (or on which of four
last-layer mass-window neurons are switched on), the difference disappears (−0.15 → +0.003 log-odds per unit). So the different
response follows the network's own state, not the true class beyond that state. Holding three mass-window neurons fixed removes
0.59 of the difference, but that is no longer clearly more than holding random directions of the same size. The prong-width
result does not hold up: two edits that both widen the prongs at fixed prong mass move the coordinate in opposite directions, and
the "natural relation" it was compared with changes size six-fold depending on how it is fitted.

## Terms

- **h4**: 64-unit fourth layer of JEDI-linear; **z**: the 16 pre-ReLU units of the last hidden layer; logits = d7(ReLU(z)).
- **W−Z margin** $\ell_W - \ell_Z$ (log-odds between the W and Z outputs). **Baseline margin** $\mu_0$: this margin for the
  unedited, unsteered jet.
- $u = h_4\cdot v$, $v$ = 1-dim DAS direction (distributed alignment search: a learned unit vector in h4 such that copying the
  component along $v$ from a D-edited jet into the unedited jet reproduces the edited jet's W−Z margin). D(dc): rest-frame
  decay-axis rotation that makes the prong momentum split more lopsided at fixed jet four-momentum and prong masses.
- **Steering slope**: per jet, least-squares slope through 0 of the W−Z margin change against the push $c\,\mathrm{sd}_u$,
  $c\in\{\pm0.5,\pm1,\pm2\}$ (log-odds per unit $u$). **Small-step slope**: the same with $c = \pm0.5$ only.
- **Matched contrast Δ**: W−Z difference of the mean slope inside cells (mass bins, or mass × state cells), averaged with weights
  $n_W n_Z/(n_W+n_Z)$, i.e. W and Z jets reweighted to a common distribution over the cells.
- **z pattern**: the on/off state (pre-ReLU value > 0) of z0, z7, z11, z14 (16 combinations); z0 is active in a window near
  75 GeV, z7/z14 near 94–98 GeV, z11 a mass-threshold unit (E11).
- **Removed share** = 1 − Δ(held)/Δ(free) when z units are frozen at their unsteered values during steering.
- **$W_p$** prong width = $\sum_i p_{T,i} r_i/\sum_i p_{T,i}$, $r_i$ = distance of particle $i$ to its own prong axis.
  **R_w** = (mean Δu under the width edit minus its control) / ($\beta_w$ × mean Δ$W_p$, edit minus control); $\beta_w$ = slope of
  $u$ on $W_p$ across unedited jets with controls (OLS: log m, (log m)², log pT, $z_h$, $z_h^2$, ΔR12, both prong masses; kNN:
  residuals after 50-nearest-neighbour averaging in (log m, log pT)).
- **IIA**: share of call-changing pairs whose patched call equals the edited jet's call. **frac**: mean patched margin change /
  mean full change. **Holm**: step-down multiple-test correction. **Rank p**: $(1+\#\{\text{null}\ge\text{obs}\})/(1+n)$.

## Code review (interp/e37y_sharing_gate_jedi.py, interp/e37x_plot.py)

- Stage and tail: the h4 → d5 → d6 → z override → ReLU → d7 tail uses the fused fixed-point kernels and input quantizers;
  reproduced here on the new sample (max |Δlogit| vs the stored forward pass 0.0). The z override $z + (z_v - z)P$ is applied
  before the ReLU as described. No wrong stage, mask or quantizer found.
- Padding: mass, prong labels and observables use only the first `nn` slots; edited jets are rebuilt with zero padding. No
  padded slots counted.
- Split: F rows that also appear in T are removed before fitting $v$ (here 585 of 12,500 rows); $\beta_w$ is fitted on F, effects
  measured on T. No leakage found.
- Steering clip: `proj` clips h4 to the next layer's input range including the 0 floor. On the new sample every jet has at least one
  h4 unit clipped already at $c = \pm0.5$ (fraction 1.00), not only at $\pm2$: units at 0 with a negative $v$ weight cannot go lower.
  The small-step slope is therefore not clip-free; it is reported as the least-extrapolated version.
- Arm 3 Erot control: the rotation sign is drawn at random per jet, so its mean Δu ≈ 0 holds by construction; it controls the
  mean, not the per-jet magnitude.
- Figure `results/jedi/e37x.png` (read): readable, axes in GeV and log-odds, jet counts shown. Panel (d) shows the per-bin slope
  rising from about −0.6 to +0.6 between 77 and 92 GeV; with a gradient of this size, 5-GeV bins leave W and Z jets at different
  masses inside a bin (from the neighbouring-bin counts, W jets in 80–85 GeV are expected to lie toward 80 GeV and Z jets toward 85 GeV), which is why the 1-GeV value is smaller.
  The suptitle ("moves W and Z jets of the same mass differently") is accurate as a population statement but, per the
  state-matching result below, the difference follows the network's baseline state.

## Rerun on fresh samples

Procedure as E37y. T = W+Z jets of `sample(2500, 6161)` (5,000 jets; 80–100 GeV: W 1,109, Z 2,197); F = W+Z jets of
`sample(2500, 6162)` minus rows in T (4,772). 5-class accuracy on the 12,500 stored T jets 80.46 %. Refitted $v$: |cos| with E37's
stored direction 0.83, largest cosine with M2 0.08, corr($u$, $z_h$) on F 0.63, F interchange frac 0.84 / IIA 0.34 (E37y: 0.83 /
0.35); $\mathrm{sd}_u$ = 1.63. Bootstrap 2,000 jet resamples (500 for regressions and $\beta$; 1,000 for arm 3).

### Arm 1 — steering slope, W vs Z at matched mass

| quantity | E37y (5757/5758) | verify (6161/6162) |
|---|---|---|
| Δ, 5-GeV bins 80–100 GeV | −0.355 [−0.408, −0.303] | −0.301 [−0.353, −0.249] |
| matched W / Z slope | −0.226 / +0.129 | −0.282 / +0.019 |
| permutation p (within 1-GeV bins) | 0.0005 | 0.001 (floor, 1,000 perms) |
| Δ, 1-GeV bins | −0.207 [−0.262, −0.156] | −0.145 [−0.198, −0.093] |
| linear model, W−Z at 90 GeV | −0.40 [−0.48, −0.32] | −0.43 [−0.50, −0.36] |
| Δ per s.d. step vs 100 random h4 directions | −0.58; rank p 0.015 | −0.49; random 95 % range [−0.23, +0.33]; rank p 0.010 |
| small-step slope ($c=\pm0.5$), 5-GeV / 1-GeV | – | −0.374 [−0.439, −0.310] / −0.197 [−0.271, −0.128] |

H0 (equal W and Z slopes at matched mass) is rejected again; the "mass only" alternative (Δ → 0 at 1-GeV matching) is excluded
again (CI excludes 0). The Z-side positive slope is smaller on the new sample (+0.02 vs +0.13).

**Added control: matching on the network's baseline state.** W and Z jets of equal mass differ in where they sit in the network
before any push: at matched 1-GeV mass the baseline margin $\mu_0$ is higher for W by 3.18 [2.87, 3.50] log-odds and $u_0$ by
−0.76 [−1.02, −0.51]. Two hypotheses: H_class — the slope depends on the true class beyond the network's state (predicts Δ ≠ 0 inside
state cells); H_state — the slope is a function of the baseline state, which differs between W and Z jets of equal mass (predicts
Δ → 0 inside state cells).

| matching (80–100 GeV jets) | jets in matched cells | Δ [95 % CI] | permutation p (within cells) |
|---|---|---|---|
| 1-GeV mass | 2,062 | −0.145 [−0.198, −0.093] | 0.001 |
| 5-GeV mass × $\mu_0$ (1 log-odds bins) | 885 | −0.033 [−0.095, +0.028] | 0.28 |
| 1-GeV mass × $\mu_0$ (1 log-odds bins) | 255 | +0.027 [−0.055, +0.108] | 0.55 |
| 1-GeV mass × z pattern (z0, z7, z11, z14 on/off) | 862 | +0.002 [−0.054, +0.056] | 0.95 |
| 1-GeV mass × $u_0$ (0.5 s.d. bins) | 620 | −0.186 [−0.257, −0.119] | 0.001 |

Regression on all 3,306 jets at 80–100 GeV, slope ~ isW + 1-GeV mass dummies + controls (isW coefficient [95 % CI], bootstrap p):

| controls | isW |
|---|---|
| mass only | −0.149 [−0.206, −0.094], p < 0.002 |
| + $\mu_0$, $\mu_0^2$ | −0.045 [−0.102, +0.015], p 0.13 |
| + $u_0$, $u_0^2$ | −0.177 [−0.226, −0.126], p < 0.002 |
| + $\mu_0$, $\mu_0^2$, $u_0$, $u_0^2$, $\mu_0 u_0$ | +0.003 [−0.040, +0.046], p 0.90 |
| + the above and ReLU(z) of all 16 units | −0.008 [−0.039, +0.027], p 0.67 |

H_state predicts these zeros; H_class (a class difference of the size of the 1-GeV value, −0.15) is excluded by the CIs of the
z-pattern match and the full regression. Matching on $u_0$ alone does not remove the difference, so the state that matters is the
baseline W/Z margin or z on/off pattern, not the coordinate's own baseline value. Reading: the slope difference between W and Z
jets of equal mass is the dependence of the slope on which z units are on (equivalently on the network's baseline W−Z margin);
"depends on the true class" is a population description, and the class carries no further dependence at this precision.

### Arm 2 — z holds

| hold | E37y share [CI] | verify share [CI] | verify p(share ≤ 0.5) | verify rank p vs random d-dim | verify Holm (7) | verify rank p vs other units |
|---|---|---|---|---|---|---|
| z0 | 0.34 [0.31, 0.38] | 0.41 [0.36, 0.46] | 1.0 | 0.040 | 0.28 | 0.071 (floor) |
| z7 | 0.13 [0.04, 0.20] | 0.04 [−0.07, 0.12] | 1.0 | 0.49 | 0.70 | 0.29 |
| z14 | 0.16 [0.13, 0.19] | 0.14 [0.10, 0.17] | 1.0 | 0.23 | 0.69 | 0.14 |
| z0+z7 | 0.47 | 0.45 [0.35, 0.52] | 0.90 | 0.070 | 0.30 | 0.013 |
| z0+z14 | 0.50 | 0.55 [0.50, 0.60] | 0.032 | 0.040 | 0.28 | 0.013 |
| z7+z14 | 0.29 | 0.17 [0.06, 0.27] | 1.0 | 0.35 | 0.70 | 0.16 |
| z0+z7+z14 | 0.63 [0.55, 0.70] | 0.59 [0.49, 0.66] | 0.043 | 0.060 | 0.30 | 0.0035 |

Random 3-dim z holds (200): median 0.16, 95 % quantile 0.59 (E37y 0.52). Best of 286 other-unit triples 0.36. With the small-step
slope the joint hold removes 0.56. The joint hold again removes more than every triple of other units, but its margin over random
3-dim holds is not significant before correction (0.060) or after (0.30), and its excess over 0.5 is marginal (p 0.043 vs E37y
0.002). The z7 contribution does not replicate (0.04, CI includes 0); most of the removed share comes from z0 (+ z14). Together
with arm 1's state result: the slope difference is a function of the z on/off pattern, and freezing z0 and z14 removes about half of it.

### Arm 3 — prong width at fixed prong mass (random 1,200 T jets; accepted 0.66 at α −0.5, 0.70 at α +0.5)

| edit vs control | α | E37y Δu | verify Δu [95 % CI] | verify Δ$W_p$ (relative) | R_w with OLS β | R_w with kNN β |
|---|---|---|---|---|---|---|
| Ws vs Erot | +0.5 (pT outward) | −0.284 | −0.337 [−0.383, −0.293] | +0.0044 (+33 %) | 0.89 | 5.8 |
| Ws vs Eshuf | +0.5 | −0.354 | −0.606 [−0.710, −0.509] | +0.0037 | 1.89 | 12.2 |
| Ws vs Erot | −0.5 (pT inward) | +0.520 | +0.651 [+0.529, +0.782] | +0.0012 (+12 %) | −6.4 | −42 |
| Ewid vs EmW | +0.5 | +0.359 | +0.437 [+0.383, +0.493] | +0.0036 | −1.4 | −9.1 |

- $\beta_w$: OLS −86.2 [−95.4, −76.6] (E37y −83.9); kNN −13.3 (E37y −8.4; partial correlation −0.06 in E37y). The natural relation
  that R_w is calibrated against changes by a factor 6–10 with the choice of controls, so R_w ≈ 0.8–0.9 is specific to the OLS model.
- Both Ws edits widen the prongs at fixed $z_h$, ΔR12, prong masses and jet four-momentum ($W_p$ rises in 88 % of jets at α +0.5 and
  75 % at α −0.5), yet $u$ falls at α +0.5 and rises at α −0.5. A width effect at fixed prong mass predicts the same sign for both;
  the signs are opposite. Within each α, the per-jet correlation between Δ$W_p$ and Δu is −0.37 (α +0.5) and +0.17 (α −0.5).
- Reading: $u$ follows the direction in which pT is moved inside the prongs (toward the core raises $u$, toward the periphery lowers
  it), not $W_p$. The E37y statement that widening at fixed prong mass lowers $u$ "in line with the natural width–$u$ relation"
  rests on one sign of α and one regression model. The Ewid result (u rises when prong masses rise with width) replicates, but no
  edit in E37y or here changes prong mass at fixed width, so "the earlier width response followed the prong masses" is an
  inference, not a measurement.

### Arm 4 — Z-jet response to D(+0.3) through directions orthogonal to $v$ and M2 (2,500 T Z jets, 428 call changes; fit on 2,395 F Z jets)

| quantity | E37y | verify |
|---|---|---|
| free margin change | +1.61 | +1.60 |
| copy $v$: frac / IIA | 0.92 / 0.27 | 0.95 / 0.23 |
| $Q_1$: frac / IIA (random 95 %) | 0.95 / 0.12 (0.26 / 0.08) | 0.92 [0.88, 0.95] / 0.18 [0.15, 0.22] (0.25 / 0.10); rank p 0.005 |
| $Q_3$: frac / IIA (random 95 %) | 1.02 / 0.37 (0.44 / 0.13) | 0.96 [0.93, 0.99] / 0.43 [0.38, 0.48] (0.37 / 0.12); rank p 0.005 |
| hold $v$ + M2: share removed | 0.07 | 0.46 |
| hold $v$ + M2 + $Q_1$ / $Q_3$ | 0.26 / 0.61 | 0.88 / 0.91 (random 95 % 0.08 / 0.18) |

The copy result replicates: a direction orthogonal to $v$ and M2 reproduces ≥ 0.9 of the mean Z response, with low per-jet call
agreement. The hold numbers do not replicate: holding $v$ + M2 removes 0.46 here vs 0.07 in E37y, and $v$ + M2 + $Q_1$ 0.88 vs 0.26.
"Holding $v$ + M2 alone removes 0.07" and the size of the redundancy are therefore seed-dependent.

## Verdict on the reading as stated

- Arm 1: replicated (H0 rejected; mass-only excluded). Corrected: the class difference at equal mass is fully accounted for by the
  network's baseline state (baseline W−Z margin, or the on/off pattern of z0, z7, z11, z14); with that state matched, Δ = +0.002
  [−0.054, +0.056]. The thesis-note wording "depends on the true class at matched mass" should read "differs between W and Z jets of
  equal mass because they sit in different z on/off states".
- Arm 2: partly replicated (0.59 vs 0.63; above all other-unit triples), weaker against random holds (rank p 0.06 uncorrected,
  Holm 0.30) and only marginally above 0.5; z7's share does not replicate.
- Arm 3: not supported as stated. The R_w 0.80 agreement depends on one α sign and one regression model; the other α widens the
  prongs and moves $u$ the opposite way. The correction of E37xv's width attribution ("followed the prong masses") is not
  established by a measurement.
- Arm 4: copy result replicated; hold shares (0.07 for $v$ + M2) not replicated (0.46).

Suggested corrected reading: "At equal jet mass, the steering slope of $u$ differs between W and Z jets (Δ −0.30 at 5-GeV, −0.15 at
1-GeV matching, two seed pairs); the difference vanishes when W and Z jets are also matched on the network's baseline W−Z margin or on
the on/off pattern of z0, z7, z11, z14, so it is a dependence on the network's last-layer state. Freezing z0+z7+z14 removes 0.59–0.63
of it, more than any triple of other units but not significantly more than random 3-dim z holds. Under the fixed-prong-mass width
edit, $u$ moves with the direction of the pT redistribution inside the prongs (outward lowers, inward raises), not with prong width;
the cause of the E37xv width response is not identified. A direction orthogonal to $v$ and M2 copies the mean Z-jet response."

## Limits of this verification

- One new seed pair; arm 3 on a 1,200-jet random subset (CIs about twice as wide as E37y's).
- The state-matched cells contain fewer jets (255–885 of 2,062 for the cell matches; all 3,306 for the regression).
- Matching on the baseline state removes any class dependence that acts through that state by construction; the test shows there
  is none left beyond it, not where upstream of z the state difference arises.
- New glossary terms: baseline margin $\mu_0$; z pattern; state matching.
