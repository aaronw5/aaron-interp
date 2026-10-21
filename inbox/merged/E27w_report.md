# E27w: jet mass $m$ vs the pT-weighted spread $M_2$ in JEDI-linear (azimuth-only edits), and the power of $r$ (claims A4w / A4; round 13; agent report 2026-10-21)

## Conclusion
We turned some particles of real jets around the jet axis. Each particle kept its pT and its distance from the axis, so the
pT-weighted spread $M_2=\sum p_T r^2$ did not change, but the particles bunched to one side and the jet mass fell by 2–10 %.
The network's mass read-out did not move (about 0 % of the response to a real mass change of the same size), and Z-tagged jets
were relabelled W no more often than under a control turn that left the mass unchanged (3 % vs 93 % for a real 10 % mass drop).
So the network's mass variable follows $M_2$ about the stored axis, not the jet mass. A second test, which reshaped the radial
profile with $M_2$ held, fits best with a single power of $r$ slightly above 2 ($\beta\approx2.15$). Still uncertain: five
random smooth radial sums predict those responses as well as the $\beta$ family, so the $\beta$ value is only weakly pinned.
The softer-half version of the edit was possible on too few jets to read.

## Terms used
- $r_i$, $\theta_i$: distance and direction (azimuth) of particle $i$ around the stored jet axis in the $(\Delta\eta,\Delta\phi)$ plane.
- $M_0=\sum_i p_{T,i}$ (scalar pT sum). $\vec M_1=\sum_i p_{T,i}(\Delta\eta_i,\Delta\phi_i)$: the pT-weighted offset of the
  particles from the axis. $|\vec M_1|/M_0$ is how far the pT-weighted centre lies from the axis (median 0.0016–0.003 rad here).
  $M_2=\sum_i p_{T,i}r_i^2$: the pT-weighted squared spread about the stored axis ("un-recentred").
- Jet mass: $m^2\approx M_0M_2-|\vec M_1|^2$. All "true mass" numbers below are the exact four-vector mass of the edited particles.
- $\lambda_\beta=\sum_i p_{T,i}r_i^\beta$ (pT-weighted spread with power $\beta$ of $r$; $\lambda_2=M_2$, $\lambda_1/M_0$ = girth).
  $C_1=\sum_i r_i$: count-weighted (each particle counts 1) sum of distances.
- **Azimuth-only edit (arm B):** chosen particles are turned around the axis, $\theta_i\to\theta_i+t\,\mathrm{wrap}(\phi_{M_1}-\theta_i)$,
  i.e. a fraction $t$ of the way toward the direction of $\vec M_1$. pT, $r$, pT rank and input slot of every particle stay
  exactly fixed (checked: max $|\Delta r|\le 2\times10^{-8}$ rad, float32 rounding), so $M_0$, $M_2$, every $\lambda_\beta$ and $C_1$
  are unchanged; $|\vec M_1|$ grows, so $m$ falls. $t$ is solved per jet so the true mass falls by exactly 2, 5 or 10 %.
- **Control (M1 held):** the same particles are turned by the same angles, but each clockwise or anticlockwise at random, then a
  small minimum-norm azimuth correction on the same particles restores the vector $\vec M_1$ (to $10^{-5}M_0$), alternated with a
  common rescale so the total displacement $D=\sqrt{\sum_i|\Delta\vec x_i|^2}$ equals the arm-B edit's (within 2 %). Same
  particles moved by a similar amount, every radial quantity fixed, $m$ fixed (median $|\Delta\log m|\le 3\times10^{-5}$). It is
  the comparison for "turning particles around the axis changes the angular (half-plane) statements", which both edits do.
- **dil_m:** all angles multiplied by one factor per jet, solved so the true mass falls by the same amount as in the arm-B edit on
  that jet (max difference $6\times10^{-8}$ in $\log m$). A real mass change of the same size; $M_2$ falls with it.
- **Freeze control (B_frz, ctrl_frz, dil_frz):** the same inputs, with the context vector g (all 64 units of the jet-level layer
  that feeds back into every particle) and the second-round objects S31 and S58 (context-gated count multipoles, E24x) held at
  their unedited values in the forward pass.
- **Mass read-out:** ridge fit of $\log m$ on the 64 values of layer h4 (and the 32 of h5), fitted on unedited jets of half A
  of the fit sample; responses evaluated on half B. Held-out $R^2$ 0.974 (h4), 0.973 (h5).
- **R_m** (arm B) = mean change of the read-out under the edit ÷ mean change under dil_m on the same jets. 1 = the network
  reacts as to a real mass change of that size; 0 = no reaction. **R_marg_Z** = the same ratio for the W–Z logit margin
  ($\mathrm{logit}_Z-\mathrm{logit}_W$) on jets called Z before the edit.
- **Z → W flow** = share of jets called Z before the edit that are called W after. **W → Z** analogous.
- **McNemar test:** exact paired test on the same jets' call changes (edit vs control, edit vs dil_m).
- **Holm:** step-down correction over the 6 arm-B variants per test, and over the 6 terms of the arm-C fit.
- **R_marg (arm C)** = mean change of the W–Z margin on W-called jets ÷ the same for a 5 % $M_2$ dilation ($r\to\sqrt{1.05}\,r$).
- **Freedman–Lane permutation:** test of one regression coefficient by permuting the residuals of the model without that term.

## Files
- Script: `interp/e27w_mass_vs_m2_jedi.py`; figure: `interp/e27w_plot.py` (rendered locally from the json/npz).
- Jobs: `./job.sh e27w-smoke 1 2 '... --smoke 60'` (code check; wall 649 s, peak 1,177 MiB);
  `./job.sh e27w-run 1 2 'python -m interp.e27w_mass_vs_m2_jedi'` (wall 701 s, of which ~360 s is import from the volume;
  pod peak 1,878 MiB; 0.46 CPU on average).
- Results: `results/jedi/e27w_mass_vs_m2.json`, `results/jedi/e27w_example.npz` (example jet before / after),
  `results/jedi/e27w_mass_vs_m2.png`; logs `logs/e27w-run.log`, `logs/e27w-smoke.log`.
- Debug: one change after the smoke run (counts as debug attempt 1). The first control (turn the subset toward another direction,
  solved so $|\vec M_1|$ is unchanged) was feasible on 1–9 of 300 jets, because $|\vec M_1|$ is near 0 for these jets and any
  one-sided turn raises it. It was replaced by the random-sign turn with an $\vec M_1$ correction described above.

## Procedure
- **Network:** official JEDI-linear 3-feature n64, exact float forward pass (`interp.jedi_core.Jedi`, `run` / `run_jetwise`).
- **Samples:** fit = `sample(per_class=1000, seed=2701)`, N 5,000; accuracy **0.8130** (expected 0.806–0.813). Test =
  `sample(1000, 2702)` minus 95 jets also in the fit sample, N 4,905; accuracy 0.8088 (arm C held-out scoring only).
- **Exact checks:** run_jetwise with g, S31, S58 overridden by their own unedited values reproduces the logits exactly
  (max difference 0).
- **Arm A (reference: $M_2$ up at fixed $m$):** every real particle shifted by +0.06 in $\Delta\eta$; median $\Delta\log M_2$ +0.44,
  true $\Delta\log m$ $<10^{-8}$. Calibration dil_M2: angles $\times\sqrt{M_2'/M_2}$ per jet (same $M_2$ change, $m$ changes with it).
- **Arm B variants:** subset "hard14" = the 14 highest-pT particles; subset "soft" = the lower-pT half of the real particles;
  mass target −2, −5, −10 %. A jet is used when the arm-B edit reaches the target with $t\le1$, the control meets its
  constraints, and the control's $|\Delta\log m|\le0.1\,|\log(1-f)|$.
- **Arm C (power of r):** each edit multiplies every real particle's $r$ by $\exp(0.2\,(f(r_i)-\lambda_J))$ (direction about the
  axis and pT fixed). $f$ is one fixed smooth function per edit, a random unit-weight combination of
  $\{r, r^2, \log r, (r-0.2)_+, (0.1-r)_+\}$ (each divided by its spread over particles), centred and scaled so its
  99.5th-percentile $|f|$ is 1. $\lambda_J$ is solved per jet so $M_2$ is unchanged (set H2) or the true mass is unchanged
  (set Hm). 60 shapes × 2 sets = 120 edits on the fit sample, 20 new shapes × 2 = 40 on the test sample; 99 % of W-called
  jets kept (max radius factor ≤ 2). Per edit, target R_marg and regressors $X_\beta=\overline{\Delta\log\lambda_\beta}/\log1.05$
  (the 5 % dilation has $X_\beta=\beta/2$, so a pure $\lambda_\beta$ reader has $R=c\,X_\beta$), $X_{C1}$, and
  $X_m=2\overline{\Delta\log m}/\log1.05$. Ridge fits without intercept, $\alpha$ by leave-one-edit-out CV on the fit edits,
  scored by $R^2$ on the 40 test edits (new jets, new shapes).
- **Statistics:** 1,000 Poisson jet-bootstrap resamples (95 % intervals; in arm C both samples are resampled and all fits
  recomputed); sign-flip permutation (10,000) on per-jet paired differences; exact McNemar; Holm. Arm-C shape-matched null:
  the same ridge on 5 regressors $\sum p_T h(r)$ with $h$ random smooth functions (random weights on 12 Gaussian bumps in $r$),
  200 draws.

## Hypotheses
- **H0 (thesis, network reads $M_2$ about the stored axis):** arm B gives R_m ≈ 0, the same as the control; Z → W flow equal to
  the control's.
- **H1 (network reads the jet mass $m$):** arm B gives R_m ≈ 1 and Z → W flow at the dil_m rate (8.9 / 39.2 / 92.8 %).
- Arm C, thesis prediction (from E27v): weight on $\beta=2$ plus a $\beta=1$ (girth) term. Alternatives: one $\beta$ near 1.5;
  count-weighted terms.

## Results

### Arm A (reference)
| metric | 0.06 rad translation | dil_M2 (same $\Delta M_2$) |
|---|---|---|
| W → Z flow | 80.7 % [78.2, 83.1] | 64.5 % [61.4, 67.5] |
| R (h4 read-out) vs dil_M2 | 1.19 [1.16, 1.22] | 1 |
| R (W–Z margin, W-called) | 1.17 [1.15, 1.18] | 1 |
McNemar translation vs dil_M2 on W-called jets: 203 vs 35 jets, p = 6e-30. The translation leaves $m$ unchanged and moves the
network more than a dilation with the same $M_2$ change, as in E6/X9v/E25w.

### Arm B, hardest 14 particles turned (N jets kept: 4,546 / 3,935 / 3,143 of 5,000; all classes)
| mass change | | arm B | control (M1 held) | dil_m (real mass change) | B with g, S31, S58 frozen | dil_m frozen |
|---|---|---|---|---|---|---|
| −2 % | R_m (h4) | −0.005 [−0.044, 0.036] | −0.035 [−0.066, −0.004] | 1 | 0.009 [−0.027, 0.046] | 0.94 |
| | R_marg_Z | 0.076 [0.024, 0.126] | 0.044 [0.010, 0.079] | 1 | 0.030 | 0.94 |
| | Z → W | 1.5 % [0.8, 2.5] | 1.4 % [0.7, 2.3] | 8.9 % [7.1, 10.8] | 1.1 % | 7.9 % |
| | calls changed | 2.6 % | 2.1 % | 3.7 % | 2.0 % | 3.4 % |
| −5 % | R_m (h4) | −0.001 [−0.022, 0.020] | −0.022 [−0.037, −0.006] | 1 | 0.008 [−0.008, 0.025] | 0.94 |
| | R_marg_Z | 0.040 [0.019, 0.061] | 0.027 [0.014, 0.041] | 1 | 0.015 | 0.94 |
| | Z → W | 1.9 % [1.1, 2.8] | 1.6 % [0.8, 2.4] | 39.2 % [36.0, 42.6] | 2.0 % | 36.5 % |
| | calls changed | 3.0 % | 2.6 % | 13.1 % | 2.9 % | 12.1 % |
| −10 % | R_m (h4) | −0.013 [−0.026, 0.000] | −0.026 [−0.034, −0.017] | 1 | 0.000 [−0.011, 0.012] | 0.95 |
| | R_marg_Z | 0.037 [0.024, 0.050] | 0.019 [0.011, 0.027] | 1 | 0.019 | 0.96 |
| | Z → W | 3.1 % [2.0, 4.3] | 1.7 % [0.9, 2.6] | 92.8 % [90.8, 94.4] | 2.5 % | 89.1 % |
| | calls changed | 3.9 % | 3.0 % | 33.7 % | 3.4 % | 32.3 % |

- B minus control, in dilation units of the h4 read-out: +0.030 [−0.009, 0.071], +0.021 [0.003, 0.041], +0.013 [0.001, 0.026]
  (sign-flip p 0.13 / 0.025 / 0.030; Holm over 6 variants 0.51 / 0.15 / 0.15).
- B vs dil_m: read-out sign-flip p ≤ 1e-4 for all three (Holm 6e-4, the permutation floor); Z → W McNemar B vs dil_m:
  2 vs 69, 0 vs 337, 0 vs 742 jets (Holm p 9e-18, 4e-101, 5e-223).
- B vs control: Z → W McNemar 8 vs 7, 12 vs 9, 23 vs 11 (Holm p 1.0, 1.0, 0.35); calls changed 68 vs 43, 62 vs 48, 78 vs 47
  (Holm p 0.11, 0.64, 0.042).
- h5 read-out: R_m(B) 0.023 / −0.022 / −0.031; control −0.059 / −0.036 / −0.047.
- W → Z flow (the opposite direction to a mass drop): B 2.1 / 2.1 / 3.3 %, control 1.1 / 1.7 / 1.1 %, dil_m 0 / 0 / 0 %.
- How hard each edit drives the angular objects (mean |change| / sd over jets; −2 / −5 / −10 %):
  nine half-plane first-round objects P: B 0.078 / 0.110 / 0.125, control 0.048 / 0.066 / 0.075, dil_m 0.028 / 0.059 / 0.093;
  S31: B 0.14 / 0.18 / 0.19, control 0.08 / 0.09 / 0.10; S58: B 0.26 / 0.35 / 0.40, control 0.23 / 0.32 / 0.38;
  the five radial S objects (S29, S1, S0, S19, S30): B 0.008, control 0.006–0.007, dil_m 0.031 / 0.064 / 0.098.
- Edit sizes: median turn fraction $t$ 0.24 / 0.36 / 0.48; median displacement $D$ 0.11 / 0.18 / 0.25 rad (14 particles);
  median $|\vec M_1|/M_0$ 0.0016–0.002 → 0.016 / 0.027 / 0.038 rad (control unchanged).

### Arm B, softer half turned
Reaching the mass target by turning the softer half was possible on 696 / 70 / 5 of 5,000 jets (−2 / −5 / −10 %); after the
control constraints 469 / 29 / 0 were kept, mostly gluon and top jets (−2 %: 241 g, 53 q, 26 W, 27 Z, 122 t). At −2 %:
R_m(h4) B −0.19 [−0.53, 0.13], control −0.08 [−0.33, 0.15] (Holm p 1.0); 22 Z-called jets, Z → W 0 (B) vs 1 (control) vs 1
(dil_m); calls changed B 8.1 % vs control 5.3 % vs dil_m 1.9 % (B: 17 g → q and 12 g → t). This variant has too few W/Z jets to
test the hypotheses.

### Arm C (power of $r$; 120 fit edits, 40 held-out edits on new jets with new shapes; 1,037 / 1,023 W-called jets)
- Holding $M_2$ also holds $m$ here: because $|\vec M_1|\approx0$, set H2 changes $m$ by median $|X_m|$ 0.011 and set Hm changes
  $M_2$ by median $|X_2|$ 0.011 (dilation units). Both sets are therefore "radial reshaping at fixed $M_2$ and $m$".
- Response range at fixed $M_2$: R_marg −0.30 to +0.46 (dilation units); calls changed median 3.0 %.

| model (regressors) | held-out $R^2$ (40 test edits) | coefficients (95 % CI) |
|---|---|---|
| $M_2$ only | −0.007 [−0.053, −0.004] | 0.20 |
| $m$ only | −0.002 [−0.054, 0.000] | ≈ 0 |
| $M_2$ + $\beta=1$ (girth) | 0.74 [0.55, 0.81] | $\beta$2: 1.15 [0.76, 1.53]; $\beta$1: −0.23 [−0.26, −0.19] |
| single power, $\beta$ chosen by CV | 0.85 [0.72, 0.94] | $\beta$ = 2.15 (bootstrap 2.10–3.20); slope 1.11 [0.13, 1.71] |
| $\beta$ family {0.5, 1, 1.5, 2, 3} | 0.93 [0.79, 0.95] | −0.29, 0.02, 0.38, 0.48, 0.17 |
| family + $C_1$ (count-weighted $\sum r$) | 0.92 [0.75, 0.95] | ... $C_1$: −0.17 [−0.46, 0.15] |
| null: 5 random smooth $\sum p_T h(r)$ | median 0.93, 95th pct 0.953, max 0.963 (200 draws) | – |

- Family vs null: p = 0.65 (the $\beta$ family predicts no better than random smooth radial sums).
- Freedman–Lane per term (family + $C_1$, Holm over 6): $\beta=2$ p 0.003, $\beta=3$ p 0.045; $\beta$ 0.5, 1, 1.5 and $C_1$ p = 1.0.
- CV error by single power: minimum at $\beta$ = 2.15 (MSE 0.00138), 2.2 (0.00143), 2.25, 2.1; $\beta$ = 1.5 and below are worse.
- Under the 5 % dilation, $X(2.15)=1.075$, so the single-power model predicts R = 1.19 for a real dilation.

## Reading
- Arm B (hardest 14): turning particles around the axis so that $m$ falls by 2–10 % while $M_2$ and every per-particle
  $(p_T, r)$ stay fixed moves the h4 mass read-out by −0.013 to −0.001 of a matched real mass change. This lies within 0.03
  of the control that moves the same particles without changing $m$, and is 0.94–1.00 dilation units below H1's prediction. Z → W
  flow is 1.5–3.1 % against 1.4–1.7 % for the control and 8.9–92.8 % for the matched real mass change. The measurement agrees with
  H0 and excludes H1 at every dose (McNemar Holm p ≤ 9e-18).
- The control turns drive the half-plane objects and S31 about 60 % as hard as the arm-B turns (S58 about equally). The small
  B − control differences (read-out +0.013 to +0.030; calls changed +0.5 to +0.9 points; W → Z +1 to +2 points) have the size and
  direction expected from that extra angular drive, not from a mass reading: W → Z is the opposite direction to a mass drop.
  With g, S31 and S58 frozen, arm B gives R_m 0.000–0.009, while dil_m keeps 0.94–0.95 of its response through the same frozen
  network. The context route therefore does not hide a mass response.
- Arm C: at fixed $M_2$ (and fixed $m$), smooth radial reshaping moves the W–Z margin by up to ±0.3–0.46 dilation units. $M_2$
  alone and $m$ alone predict none of it (held-out $R^2\approx0$). A single pT-weighted spread $\sum p_T r^{\beta}$ with
  $\beta\approx2.15$ predicts it with held-out $R^2$ 0.85. $M_2$ plus girth (the E27v-based prediction) gives 0.74, with a
  negative girth coefficient. This direction matches E27v, where at fixed $M_2$ raising $\beta\ge2.5$ members moved the output
  toward Z and raising $\beta\le1.5$ members moved it toward W: a variable weighting large $r$ slightly more than $r^2$ responds
  that way. The five-term $\beta$ family reaches 0.93, but so do five random smooth radial sums, so its individual coefficients
  are not interpretable as weights on specific powers.
- Physics meaning: on these jets the network's mass-like quantity is the pT-weighted squared spread of the particles about the
  stored jet axis. It does not subtract the offset of the pT-weighted centre from that axis, which the invariant mass does.
  A jet whose particles are bunched to one side of the stored axis gets the same "mass" as an evenly spread jet with the same
  radial profile. The radial weighting is close to $r^2$, possibly slightly steeper.

## Issues and limits
1. The softer-half variant was planned but is feasible on only 14 % / 1.4 % / 0.1 % of jets (soft particles carry too little
   pT·r to raise $|\vec M_1|$ enough), and the kept jets are mostly g/t. Only the hardest-14 variant tests the hypotheses.
2. The control is matched in which particles move and in displacement $D$, not in angular drive: it moves the half-plane objects
   and S31 by about 60 % of arm B. The freeze control covers g, S31 and S58, but not the half-plane statements entering s/u
   per particle directly.
3. Arm C cannot separate $M_2$ from $m$ (both stay fixed because $|\vec M_1|\approx0$); that separation comes from arm B only.
4. Arm C regressors are collinear in $\beta$. The family fit is no better than the random-smooth null (p 0.65), so only the
   model comparison (single power vs $M_2$ + girth vs $M_2$) is informative. The single-power fit has no null of its own, and its
   $\beta$ bootstrap interval is skewed (2.10–3.20).
5. Arm C uses the exponential form $r\to r\,e^{0.2(f-\lambda)}$ (equal to the planned $r(1+ef)$ to first order). Max radius
   factor ≤ 2; 1 % of jets dropped.
6. One seed per sample. The R_marg on W-called jets for arm B is unstable (a mass drop barely moves W-called jets' Z–W margin:
   e.g. −0.62 [−1.09, −0.32] at −2 %), so R_marg_Z and the read-out are the reported margin metrics.
7. Deviation from the plan: the |M1|-held control uses random-sign turns plus an $\vec M_1$ correction (the one-sided version was
   infeasible, see Files). The "count-weighted terms" are represented by $C_1=\sum r$; girth equals $\lambda_1/M_0$ and is the
   $\beta=1$ column at fixed $M_0$.

## Explainer entry (site/explain.py, Paper A tab)
- **Question:** JEDI-linear's mass variable behaves like $\sqrt{M_0M_2}$ with $M_2=\sum_ip_{T,i}r_i^2$ about the stored axis. The
  jet mass is $m^2\approx M_0M_2-|\vec M_1|^2$. Earlier edits (E27, E27v) held $\vec M_1$, so $M_2$ and $m$ always moved together. Does
  the network follow $M_2$ or $m$ when they are separated, and which power of $r$ does it use?
- **Background:** $\vec M_1=\sum_ip_{T,i}(\Delta\eta_i,\Delta\phi_i)$ is the pT-weighted offset of the particles from the axis.
  Turning particles around the axis keeps each $r_i$, so $M_2$ is unchanged. Bunching them on one side raises $|\vec M_1|$, so
  $m$ falls.
- **Method:** turn the 14 hardest particles of each jet part of the way toward the direction of $\vec M_1$ so that $m$ falls by
  2, 5 or 10 %. Compare with: (i) a control that turns the same particles by the same amounts in random directions with
  $\vec M_1$ restored ($m$ fixed); (ii) scaling all angles so $m$ falls by the same amount (a real mass change). Measure the
  ratio $R_m=\overline{\Delta\log\hat m}(\text{edit})/\overline{\Delta\log\hat m}(\text{real change})$ and the share of Z-called jets
  that become W. Second part: reshape the radial profile smoothly, $r_i\to r_ie^{0.2(f(r_i)-\lambda)}$ with $M_2$ held, and fit
  the margin response with $\sum_\beta c_\beta\,\Delta\log\sum p_Tr^\beta$.
- **Result** (figure `results/jedi/e27w_mass_vs_m2.png`): $R_m$ = −0.013 to −0.001 (control −0.035 to −0.022). Z → W:
  1.5–3.1 % vs control 1.4–1.7 % vs real mass change 8.9 / 39.2 / 92.8 %. Radial reshaping: $M_2$ alone predicts nothing
  (held-out $R^2\approx0$); a single power $\beta\approx2.15$ gives $R^2$ 0.85; $M_2$ + girth gives 0.74.
- **Checks:** pT and $r$ unchanged to $2\times10^{-8}$. Real-mass calibration matched to $6\times10^{-8}$ in $\log m$. With g, S31,
  S58 frozen, arm B $R_m\le0.009$ while the real change keeps 0.94. Held-out shapes on a fresh sample. Random-smooth null
  for the radial fit (equal $R^2$ to the 5-term family, so only the single-power comparison is used).
- **Why it matters:** the network's "mass" is a pT-weighted spread about the stored axis, not the invariant mass: it does not
  correct for the offset of the jet's pT-weighted centre. The radial weighting is close to $r^2$.

## Glossary entries (explain.GLOSSARY)
- ('azimuth-only edit', 'an input edit that turns chosen particles around the jet axis; each keeps its pT and its distance r to the
  axis, so every radial quantity (M0, M2, pT rank) is unchanged and only the direction around the axis moves')
- ('M1 (vector)', '$\\vec M_1=\\sum_i p_{T,i}(\\Delta\\eta_i,\\Delta\\phi_i)$, the pT-weighted offset of the particles from the stored
  axis; $m^2\\approx M_0M_2-|\\vec M_1|^2$')
- ('R_m (E27w)', 'network response (mass read-out or logit margin) to an edit divided by its response to scaling all angles so the
  true jet mass changes by the same amount on the same jets; 1 = reacts as to a real mass change, 0 = no reaction')
- ('Z → W flow', 'share of jets called Z before an edit that are called W after it')
- ('Freedman–Lane permutation', 'test of one regression coefficient: fit the model without that term, permute its residuals,
  refit the full model, and compare the coefficient with its permutation distribution')
