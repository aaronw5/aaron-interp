# E27w verification: jet mass $m$ vs pT-weighted spread at fixed $M_2$, fresh samples and a drive-matched control (verifier, 2026-10-21)

## Conclusion
On a new set of jets, turning the hardest particles around the jet axis lowered the true jet mass while leaving every
particle's pT and distance from the axis unchanged. The network did not respond as it does to a real mass change. Its
mass read-out moved by about 1 % of the response to a real mass change of the same size, and at a 10 % mass drop about
5 % of Z-tagged jets became W, against 94 % for the real mass change. So the network's mass variable depends only on each
particle's pT and distance from the axis, not on the jet mass. Radial reshaping at fixed $M_2$ again fits one power of the
distance slightly above 2 ($\beta\approx2.1$), and this time that power predicts better than a matched random-shape null.
Two statements in the E27w reading did not reproduce. First, the "shift toward Z" under the edit: on the new jets the
small extra relabelling goes in both directions. Second, the claim that drive explains the B–control differences: a
control matched in angular drive accounts for only part of them.

## Terms
- $r_i$, $\theta_i$: distance and azimuth of particle $i$ about the stored jet axis in the $(\Delta\eta,\Delta\phi)$ plane.
  $M_0=\sum p_T$, $\vec M_1=\sum p_T(\Delta\eta,\Delta\phi)$, $M_2=\sum p_T r^2$ (about the stored axis, not re-centred),
  $\lambda_\beta=\sum p_T r^\beta$; $m$ = exact four-vector mass of the particles.
- **Arm B edit:** each of the 14 hardest particles is turned partway around the axis toward the direction of $\vec M_1$. pT and $r$ stay fixed, so
  $M_0$, $M_2$ and every $\lambda_\beta$ are unchanged (checked: max $|\Delta\log M_2|\le 6\times10^{-8}$, max $|\Delta r|\le 2\times10^{-8}$).
  $t$ is solved per jet so that $m$ falls by 2, 5 or 10 %.
- **ctrl** (E27w control): the same particles turned by the same angles with random signs, then corrected so that $\vec M_1$
  (and hence $m$) is unchanged, with the total displacement $D=\sqrt{\sum_i|\Delta\vec x_i|^2}$ matched to the B edit's.
- **ctrl_drv** (new, drive-matched control): the same construction, but with $D$ scaled by a factor $k$. $k$ is chosen
  from 1–3 so that the mean change of the nine angular half-plane first-round objects (layer P units 29, 6, 40, 57, 3, 43,
  30, 36, 46, each a sum over particles of a cut on direction in the $(\Delta\eta,\Delta\phi)$ plane) matches the B edit's,
  in units of their spread over jets ("HP drive"). $m$ stays fixed (median $|\Delta\log m|\le 3\times10^{-4}$).
- **dil_m:** all angles scaled by one factor per jet, chosen to give the same true $\Delta\log m$ as B (a real mass change).
- **Freeze:** the context vector g (all 64 units) and second-round objects S31 and S58 are held at their unedited values.
- **$R_{h4}$:** mean change of the h4 ridge read-out of $\log m$ under the edit ÷ the same under dil_m, on the same jets.
  The read-out is fitted on half of the unedited jets; its held-out $R^2$ is 0.970. **$R_{\rm marg,Z}$:** the same ratio for the
  W–Z logit margin on Z-called jets.
- **Z→W / W→Z:** share of Z-called (W-called) jets called W (Z) after the edit. **McNemar:** exact paired test on those indicators.
- **$R_{\rm marg}$ (arm C):** mean change of the W–Z margin on W-called jets under a radial reshaping ÷ that under a
  5 % $M_2$ dilation ($r\to\sqrt{1.05}\,r$). **$X_\beta$:** the edit's mean $\Delta\log\lambda_\beta$ ÷ $\log 1.05$.

## Code review of `interp/e27w_mass_vs_m2_jedi.py`
- Stages, masks and quantizers: the script uses `Jedi.run` / `run_jetwise` from jedi_core unchanged. Overrides are g (all
  units) and S (units 31 and 58). The freeze identity on unedited input is 0. Padded slots ($p_T=0$) are excluded from every
  edit and sum through `real`.
- The true mass uses $(\Delta\eta,\Delta\phi)$ about the stored axis. This equals the jet mass for massless constituents.
  `mass_p` and `x10_das_jedi.mass` agree.
- The arm-B edit and controls change only $\theta$ of the chosen particles. pT ranks and slots are unchanged. The arm-C
  H2 hold is exact by construction: $\sum p_T r^2 e^{2\epsilon(f-\lambda)}=M_2$.
- No split leakage. The read-out is fitted on half A and $R_{h4}$ is evaluated on half B. The arm-C test sample drops the
  95 jets that also appear in the fit sample.
- No bugs found. Two design points:
  1. The arm-B result constrains only "radial (azimuth-blind) vs $m$". Any function of the per-particle (pT, $r$) gives the
     same prediction as $M_2$.
  2. The figure's left panel shows the soft-half variant, which was feasible on only 14 % of jets. The analysed variant is
     hardest-14.

## Procedure (job e27w-verify, 1 CPU / 2 GiB, wall 325 s, peak 1,001 MiB)
Script `interp/e27w_verify.py` imports the E27w edit functions and was run in two processes (`--arm AB`, then `--arm C`).
- Arms A and B: fit = `sample(per_class=1000, seed=2741)`, N 5,000, accuracy 0.8112. E27w used seeds 2701 and 2702.
  Arm A as in E27w. Arm B hardest-14 only, at −2/−5/−10 %, with B, ctrl, ctrl_drv, dil_m, B_frz, dil_frz and ctrl_drv_frz.
  - Kept jets: B, ctrl and ctrl_drv all feasible and both controls with $|\Delta\log m|\le0.1|\log(1-f)|$.
    This gives N 3,970 / 3,402 / 818.
  - At −10 % the selected $k=3$ control was feasible on few jets. The kept set is mostly W, Z and t: 28 g, 64 q, 300 W, 301 Z, 125 t.
  - Statistics: 1,000 Poisson jet-bootstrap resamples; sign-flip permutation (10,000) on per-jet read-out differences;
    exact McNemar; Holm over the 3 doses per test.
- Arm C: fit `sample(1000, 2741)`, test `sample(1000, 2742)` minus overlap, N 4,909 (accuracy 0.8209). As in E27w, there are
  60 fit and 20 test random smooth radial profiles, each with H2 and Hm holds (120 / 40 edits). W-called jets: 1,017 fit,
  1,043 test. Added:
  - a single power with the coefficient fixed at $2/\beta$. A network that reads exactly $\lambda_\beta$ must give
    $R_{\rm marg}=1$ for the dilation, where $X_\beta=\beta/2$;
  - a single-regressor null. Each of 200 draws makes 66 random smooth $\sum p_T h(r)$, as many candidates as the $\beta$
    grid. The one with the lowest fit MSE is scored on the test edits.
  - Bootstrap: 300 jet resamples over both samples.

## Results
**Arm A** (rigid +0.06 rad shift in $\Delta\eta$ vs a dilation matched in $M_2$):
| | E27w (seed 2701) | verify (seed 2741) |
|---|---|---|
| W→Z, translation | 80.7 % | 80.8 % [78.2, 83.2] |
| W→Z, dilation | 64.5 % | 63.5 % [60.6, 66.5] |
| McNemar | 203:35, p 6e-30 | 221:45, p 4e-29 |
| $R_{h4}$ | 1.19 | 1.20 [1.18, 1.23] |

**Arm B, hardest 14** (verify sample; dil_m = 1 by definition):
| $m$ change | −2 % | −5 % | −10 % |
|---|---|---|---|
| kept N (Z-called) | 3,970 (894) | 3,402 (839) | 818 (303) |
| $R_{h4}$ B | −0.014 [−0.059, 0.028] | −0.012 [−0.034, 0.009] | +0.001 [−0.023, 0.023] |
| $R_{h4}$ ctrl / ctrl_drv | −0.016 / −0.037 | −0.013 / −0.056 | −0.026 / −0.051 |
| $R_{h4}$ B_frz / dil_frz | +0.014 / 0.93 | +0.008 / 0.93 | +0.023 / 0.93 |
| $R_{\rm marg,Z}$ B / ctrl / ctrl_drv | 0.155 / 0.083 / 0.117 | 0.065 / 0.034 / 0.052 | 0.049 / 0.013 / 0.038 |
| Z→W B / ctrl / ctrl_drv / dil_m | 2.8 / 2.2 / 2.1 / 7.9 % | 3.0 / 1.6 / 1.9 / 38.4 % | 4.6 / 0.7 / 2.3 / 94.1 % |
| W→Z B / ctrl / ctrl_drv | 1.6 / 1.5 / 1.7 % | 2.0 / 1.7 / 1.9 % | 2.9 / 1.3 / 1.6 % |
| HP drive B / ctrl / ctrl_drv ($k$) | 0.078 / 0.047 / 0.082 (1.75) | 0.131 / 0.077 / 0.132 (1.75) | 0.132 / 0.079 / 0.193 (3.0) |

Tests (Holm over 3 doses):
- H0 "B acts on the network as the real mass change does" (read-out): sign-flip B vs dil_m, Holm p 3e-4 at every dose (the
  permutation floor). Z→W, McNemar B vs dil_m: 3:49, 0:297, 0:271 (Holm p 1e-11, 2e-89, 1e-81). Excluded at every dose.
- B vs ctrl: read-out Holm p 1.0 / 1.0 / 0.09. Z→W Holm p 0.38 / 0.07 / 0.005: at −10 %, 13 jets flip only under B and 1 only under ctrl.
- B vs ctrl_drv: Z→W Holm p 0.31 / 0.28 / 0.29. The read-out differs: B − ctrl_drv = +0.023 [−0.021, 0.065] / +0.045
  [0.023, 0.066] / +0.051 [0.024, 0.079], Holm p 0.31 / 6e-4 / 6e-4. The drive-matched control moves the mass read-out
  slightly down (−0.04 to −0.06); B does not.
- B $R_{\rm marg,Z}$ is above 0 at every dose (95 % CI lower bounds 0.105, 0.044, 0.025). The margin moves toward W by 5–16 % of
  the real mass change, compared with 1–8 % under ctrl and 4–12 % under ctrl_drv.

**Arm C** (verify samples; R² on 40 held-out edits from new jets and new shapes):
| model | E27w | verify |
|---|---|---|
| $M_2$ alone | −0.007 | −0.005 |
| $m$ alone | −0.002 | −0.005 |
| $M_2$ + $\lambda_1$ (girth) | 0.74 | 0.88 [0.73, 0.91] (coef 1.45, −0.19) |
| $\beta$ family (5 terms) | 0.93 | 0.93 [0.72, 0.95]; coefficients change sign between samples |
| single power, free coefficient | 0.85, $\beta$ 2.15 (2.10–3.20) | 0.92 [0.77, 0.94], $\beta$ 2.10 (2.05–2.20), coef 1.31 (implied dilation $R$ 1.38) |
| single power, coefficient $2/\beta$ | — | 0.93 [0.72, 0.94], $\beta$ 2.15 (2.10–2.15) |
| null, 5 random smooth sums | median 0.93 (family p 0.65) | median 0.88, 95th pct 0.93 (family p 0.065) |
| null, best of 66 single random smooth sums | — | median 0.87, 95th pct 0.906; single power p 0.015 |

- Single power minus $M_2$+girth: ΔR² 95 % CI [0.006, 0.069].
- $R_{\rm marg}$ range at fixed $M_2$: −0.18 to +0.34.
- At fixed $M_2$ the true $m$ still moves a little: $X_m$ goes down to −0.10.

## Assessment of the E27w reading
Reproduced on fresh jets:
- the azimuth-only edit leaves the mass read-out at $|R_{h4}|\le0.014$ and Z→W at 2.8–4.6 %, against 7.9–94 % for dil_m. The
  jet-mass reading is excluded at every dose;
- the freeze gives B ≤ 0.023 and dil_m 0.93;
- arm A reproduces;
- in arm C, $M_2$ alone and $m$ alone fail, and a single power slightly above 2 fits.

Corrections:
1. "B shifts W→Z, the opposite direction to a mass drop" does not reproduce. On the new jets B adds 0.1–3.9 percentage points
   of call changes in both directions over ctrl. The Z→W excess at −10 % (4.6 vs 0.7 %, Holm p 0.005) is in the mass-drop
   direction, at 0.05 of the dil_m rate.
2. "The B–control differences match the larger angular drive" holds only in part. With HP drive matched (ctrl_drv), the
   Z→W excess loses significance (Holm p ≈ 0.3) and the $R_{\rm marg,Z}$ gap narrows. The read-out difference reverses sign,
   though: ctrl_drv −0.05, B −0.01. Neither control reproduces B exactly, and all of these responses are ≤ 0.06 of a real
   mass change on the read-out.
3. $R_m\approx0$ holds for the h4 read-out. The W–Z margin on Z-called jets moves by 0.05–0.16 of the real mass change
   (CIs exclude 0). That is small but nonzero, and of the same size as the drive-matched control.
4. "Network reads $M_2$ about the stored axis" is too narrow as a reading of arm B. Arm B fixes every radial quantity, so it
   separates "radial (pT, $r$) only" from $m$, not $M_2$ from other radial sums. Arm C shows the radial weighting is not
   exactly $r^2$.
5. "$\beta$ only weakly pinned" is sample-dependent. On the new samples:
   - $\beta$ = 2.10 (2.05–2.20);
   - a single power with the dilation-consistent coefficient $2/\beta$ fits as well (R² 0.93, $\beta$ 2.15);
   - the single power beats a best-of-66 random single radial sum (p 0.015).

   The 5-term family still has unstable coefficients. A power near 2 at fixed $M_2$ acts mostly through the first-order
   term $\sum p_T r^2\log r$, so $\beta$ is identified as "slightly above 2", not to two digits.

Files:
- `interp/e27w_verify.py`
- `results/jedi/e27w_verify_AB.json`, `results/jedi/e27w_verify_C.json`
- `logs/e27w-verify.log`

Figure: the left panel shows the soft-half example, a variant that could not be analysed. A hardest-14 example would match
the right panel and the result.
