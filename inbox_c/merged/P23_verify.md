# P23v: verification of P23 (shape of the axis response; C3, D4) (verifier report, 2026-10-20)

## Conclusion
When all of a jet's particles are shifted by a small angle (invariant mass unchanged), ParT's W-versus-Z response is mostly the
same for a shift and its mirror image: about 93 % of each jet's response variance along φ. The size of this symmetric part
follows that jet's own response to a real widening of its angles. This is what a read-out of the jet's angular spread about the
stored axis (the second moment M2) would give. ParticleNet has a larger mirror-asymmetric part, about 18 % of the variance. In
both networks that part follows other per-jet directions, the leading particle and the unweighted particle centroid, and not the
pT-weighted centroid that JEDI-linear uses. Two things are still uncertain. At a fixed shift size, a jet's response varies
considerably with the shift's direction, and neither M2 nor the jet's long axis accounts for most of that variation. P23's pooled
"odd gain" cannot be read as a centroid effect, because the centroid predictor follows the jet's |η| almost exactly.

## What was checked
**Code review.**
- Scripts read: `interp/p23_axis_surface.py`, `interp/p23b_perjet.py` and `interp/part_core.py` (build, kin, moments).
- No bug was found that changes the reported numbers:
  - **Build.** Edits use `part_core.build`, with η = jet_eta + sign(jet_eta)·Δη, so the G192 mirror is absent.
  - **Padding.** Moments use pt > 0 slots only, and padded input slots are masked.
  - **Row sampling.** Rows are random (`default_rng`), not first-k. The P1 sample is per-class files at rows 3,000–4,000, as
    defined in P1.
  - **Edit order.** Translations keep pT, so particle order and mask are unchanged; P23v2 asserts this.
  - **DAS hooks.** They sit on `model.embed` and `model.blocks[l]` with (P, N, C) outputs, and the jet mean is taken over real
    particles.
  - **Calibration.** κ uses the same ½Δlog M2 for every dilation.
- Labelling caveats:
  - In P23, +c_η means "toward larger |η|", because Δη is stored with the jet's η sign factored out.
  - The figure's "odd part (direction-dependent)" series is the gain on the centroid predictor Po only, not the whole odd part
    (see 4 below).

**Jobs.**
- `c-p23-verify` (`interp/p23v_verify.py`; GTX 1080 Ti, 814 s), started earlier in this verification and analysed here.
  - Sample: a fresh sample from the WToQQ_107, ZToQQ_107 and ZJetsToNuNu_107 files, 2,000 random rows each
    (`default_rng(2399)`), N = 6,000. P23 used the *_105 files.
  - Models: ParT_full, ParT_kin, PN_full, PN_kin and untrained ParT_full (init1).
  - Content: P23's full edit set and `analyse()`, plus two controls (B and C below).
  - Output: `results/part/p23_verify.json`.
- `c-p23-verify2` (`interp/p23v2_axisonly.py`; RTX 2080 Ti, 648 s), with smoke run `c-p23-verify2-smoke`.
  - Sample: WToQQ_108 and ZToQQ_108, 2,000 random rows each (`default_rng(2311)`), N = 4,000 W/Z jets.
  - Models: the same five.
  - Output: `results/part/p23_verify2.json`.
  - Added control: three input modes for every one of the 80 grid translations.
    - **raw** is P23's version: inputs fully rebuilt.
    - **pv** takes every per-particle feature from the edited jet and the four-vectors from the unedited jet, so ParT's pair
      features are those of the unedited jet.
    - **ax** takes only Δη, Δφ and ΔR from the edited jet. Energies, log pT, PID, impact parameters and four-vectors are all
      from the unedited jet.
  - Read-out: change of logit_Z − logit_W. Statistics: jet bootstrap (1,000), sign-flip permutation (2,000), and Holm over the
    5 models.

## Numbers
**Unedited accuracy.**
- P1 sample, both runs: ParT_full 85.93 % and ParT_kin 74.83 %.
- Fresh samples: ParT_full 76.3 % on W/Z/QCD _107 and 74.6 % on W/Z _108 (10-class argmax).
- ParT_kin batching check:
  - Multiplicity-sorted batches (`ParTNetwork.run`) and P23's unsorted chunks give identical logits (max difference 8e-6) and
    74.83 % on both GPU types.
  - So the 0.37-point gap to P4/P18 (74.46 %) does not come from batching or the GPU. P4 reports raw and rebuilt-jet accuracies
    separately; raw vs rebuilt inputs is the remaining candidate and was not checked.

**1. Reproduction on fresh jets (_107, W/Z N = 4,000; P23 values in parentheses).**

| model | κ | odd share | g_odd | g_even | g_o − g_e (p) |
|---|---|---|---|---|---|
| ParT_full | 10.3 (9.87) | 0.063 [0.059, 0.066] (0.064) | 0.68 [0.54, 0.81] (0.68) | 0.65 [0.63, 0.67] (0.66) | +0.04 [−0.10, 0.16] (0.64) |
| ParT_kin | 8.5 | 0.082 (0.080) | 0.05 [−0.10, 0.18] (0.02) | 0.49 (0.50) | −0.44 (0.001) |
| PN_full | 9.8 | 0.188 (0.184) | 0.53 (0.52) | 0.26 (0.27) | +0.27 (0.001) |
| PN_kin | 7.7 | 0.157 (0.160) | 0.73 (0.71) | 0.14 (0.15) | +0.59 (0.001) |
| untrained | −0.20 | 0.563 (0.567) | – | – | – |

Other quantities, ParT_full unless stated:
- **φ + 0.06 mean.** Total 0.755 [0.715, 0.793]; even part 0.749.
- **Calls.** W → Z on true W: 9.9 → 29.3 % (McNemar p 9e-96). Rot90: 13.4 %.
- **Ratio R to the matched dilation.** 0.79–0.84 at the four reference shifts (P23: 0.78–0.82).
- **Per-jet cos(odd gradient, M1).** 0.130 against a null of 0.128, p 0.64 (P23: 0.132 vs 0.125).
- **Even anisotropy, long vs short axis.** −0.100 [−0.107, −0.093]; PN_full −0.64; PN_kin −1.63.
- **Held-out random directions.** R² 0.28. Within-jet s.d. over 4 directions at |c| = 0.06 is 0.33, against 0.04 from the fit;
  the mean response is 0.75.
- **Even η − φ gain.** ParT_full 0.006 [−0.000, 0.013]. ParT_kin 0.031 [0.021, 0.040] (P23: 0.017).

**2. Is the per-jet κ_j improvement specific to a real change? (control C of c-p23-verify, W/Z, b = 2).**
- Measure: R² of the even part under (per-jet scale) × Pe, with one pooled factor.

| per-jet scale | ParT_full | ParT_kin | PN_full |
|---|---|---|---|
| own κ_j (matched real dilation) | 0.72 [0.71, 0.73] | 0.59 | 0.37 |
| κ of a random other jet (null) | 0.10 [0.08, 0.11] | 0.07 | 0.05 |
| rms response to the 3 rotations at fixed M2 | 0.11 | 0.09 | 0.06 |
| none (P23 pooled) | 0.32 | 0.21 | 0.15 |
| rms of the 4 held-out random translations (same edit family; upper reference) | 0.44 | 0.28 | 0.26 |

- H0: κ_j carries no jet-specific information; it predicts the other-jet null. This is excluded: the CIs do not overlap.
- The per-jet scale from the real dilation predicts the symmetric response better than a scale taken from the same family of
  translations.

**3. Which angular moment the even part follows (control B; per-jet free scale, so only the c-dependence is tested).**
- Predictor family: S_b = Σ pT |r + c|^b.

| b | ParT_full | ParT_kin | PN_full | untrained |
|---|---|---|---|---|
| 0 (log) | 0.45 | 0.41 | 0.42 | 0.68 |
| 0.5 | 0.64 | 0.60 | 0.60 | 0.84 |
| 1 | 0.82 | 0.79 | 0.74 | 0.94 |
| **2 (M2)** | **0.88** | **0.85** | **0.77** | 0.92 |
| 3 | 0.85 | 0.82 | 0.73 | 0.88 |

- b = 2 is best for every trained model (Holm p 0.004 against each alternative).
- The untrained net also reaches R² 0.92 with b = 2. A quadratic-in-c symmetric part is the leading Taylor term of any smooth
  response, so "grows with |c|²" is not evidence for M2 on its own. The M2-specific evidence is the κ_j calibration (2).

**4. P23's g_odd is collinear with jet |η| (c-p23-verify2, _108, W/Z N = 4,000).**
- M1η < 0 in 98.3 % of jets: the pT-weighted centroid sits on the low-|η| side of the stored axis.
- Correlation of Po(c_η = 0.06) with tanh|η_jet|: −0.944. Correlation of M1η/M2 with tanh|η_jet|: −0.956.
- Po (P23's "centroid" regressor) is therefore about 95 % determined by the jet's |η|. A gain on it measures how the η-direction
  odd response scales with |η|, not a per-jet c·M1 term.
- Removing the energy and pair-feature changes (ax) lowers g_odd by:
  - ParT_full: 0.69 → 0.59, difference 0.10 [0.09, 0.12].
  - ParT_kin: −0.05 → −0.17, difference 0.12 [0.09, 0.16].
  - PN_full: 0.66 → 0.57, difference 0.09.
  - PN_kin: 0.74 → 0.54, difference 0.20.
  - All Holm p 0.005.
- The odd variance share moves by at most 0.009 for trained models; g_even changes by less than 0.004.
- H1 ("the odd part is the energy change of η shifts") is excluded. That route carries 13–27 % of g_odd and less than 1 point of
  odd share.
- With an explicit c_η·tanh|η_jet| regressor added to the raw odd fit, ParT_full's g on Po drops 0.69 → 0.46 and ParT_kin's
  −0.05 → −1.09.

**5. Per-jet odd share along φ (energies and pair features exactly unchanged in exact arithmetic).**

| model | odd share on φ-axis pairs [95 % CI] | on η-axis pairs |
|---|---|---|
| ParT_full | 0.075 [0.071, 0.081] | 0.082 |
| ParT_kin | 0.100 [0.093, 0.108] | 0.104 |
| PN_full | 0.180 [0.169, 0.190] | 0.205 |
| PN_kin | 0.169 | 0.186 |
| untrained | 0.650 | 0.706 |

- P23's "98 % of the mean φ + 0.06 response is even" is a mean over jets. Because M1φ ≈ 0 and the jet population is symmetric
  under φ → −φ, the odd part largely cancels in a mean whatever the network does. The per-jet figure (92.5 % even for ParT_full)
  is the informative one.

**6. Per-jet odd gradient vs per-jet vectors (cos, with the random-other-jet null; _108).**

| model | pT centroid | unweighted centroid | leading particle |
|---|---|---|---|
| ParT_full | 0.108 (0.108), p 0.93 | 0.053 (0.007), p 0.004 | −0.048 (−0.026), p 0.16 |
| ParT_kin | 0.062 (0.070) | −0.054 (−0.001), p 0.002 | 0.130 (−0.021), p 0.001 |
| PN_full | 0.239 (0.219), p 0.001 | 0.185 (−0.002), p 0.001 | −0.201 (0.015), p 0.001 |

- These match P23b on the _105 jets: ParT_kin leading particle 0.131; PN_full unweighted 0.213 and leading −0.225.
- In c-p23-verify, the ParT_kin odd gradient also aligns with V_1 = Σ pT r/|r| (0.18 vs 0.005), and PN_full's with
  V_3 = Σ pT |r| r (0.33 vs 0.02).

**7. Float32 rounding of ParT's pair features under φ-only shifts (raw − pv).**
- The pair features are rotation-invariant in exact arithmetic. Mean |Δd|:
  - ParT_full 0.002 against a response of 0.85.
  - ParT_kin 0.018 against 0.67; 3 % of jet-edit cases exceed 0.1.
  - ParticleNet 0, since it does not use four-vectors.
- This is not material to P23's numbers.

## Assessment of P23's statements
- **Reproduced on fresh rows and files:**
  - the even-dominated variance (6–8 % odd in trained ParT);
  - g_even (0.65 / 0.49);
  - ParticleNet's higher odd share and g_o > g_e;
  - untrained nets odd-dominated, with κ ≈ 0;
  - the per-jet null for the pT-weighted centroid in ParT_full;
  - the long-vs-short anisotropy;
  - η/φ even-gain equality for ParT_full;
  - the W → Z calls.
- **Corrections:**
  - (a) "98 % of the φ + 0.06 response" should be replaced by the per-jet φ-axis even share: 92.5 % ParT_full, 90 % ParT_kin.
  - (b) "Grows with |c|² as M2 does": the |c|² shape is generic (untrained R² 0.92). The M2 link rests on the κ_j calibration,
    which passes its controls.
  - (c) "At about 0.66 of a matched real mass change" is the pooled-grid gain. At the φ + 0.06 point itself the measured ratio
    is 0.79–0.82.
  - (d) "ParticleNet weights direction-dependent terms more (gains 0.52 vs 0.27)": the g_odd figure is the gain on a regressor
    that is −0.94 correlated with jet |η|. The supported statement is the odd variance share (0.18 vs 0.06) plus per-jet
    alignment with the unweighted centroid and against the leading particle.
  - (e) "The direction-dependent centroid term is not seen in ParT" holds for the pT-weighted centroid. ParT_kin's odd gradient
    follows the leading particle (0.13 vs −0.02), and ParT_full's weakly follows the unweighted centroid (0.05 vs 0.01).
  - (f) The remaining direction dependence is larger than "10 %". Within-jet s.d. over directions at |c| = 0.06 is 0.33 against
    a mean of 0.75, and the per-jet even anisotropy median is 0.15 (ParT_full) and 0.47 (PN_full). The −0.10 long-vs-short value
    is a population average.
  - (g) The thesis note "one gain not excluded for ParT_full" is an average of opposite W (0.97 vs 0.73) and Z (0.20 vs 0.45)
    differences, so it should not be read as one gain per jet class.
  - (h) "Untrained nets respond 20× more weakly" compares raw logit changes between nets with different logit scales. No
    normalisation was applied.
- **Not affected:** the DAS reading, that recovery levels are not training evidence (untrained 0.73–0.76).

## Corrected reading
In trained ParT, about 92 % of each jet's response variance to a rigid φ shift of the constituents is symmetric under c → −c. Over
the whole 2-D grid the figure is 92–94 %. The symmetric part's c-dependence is best described by the pT-weighted second moment M2
among Σ pT |r + c|^b (b = 0–3). Its per-jet size follows that jet's response to a matched real dilation: R² 0.72 against 0.10 for
another jet's calibration.

The antisymmetric part is 6–10 % of the variance. Per jet, it is not aligned with the pT-weighted centroid. It aligns weakly with
other per-jet directions: the leading particle in ParT_kin and the unweighted centroid in ParT_full. P23's pooled g_odd is a gain on
a regressor that is 94 % determined by jet |η|. 13–27 % of g_odd comes from the particle-energy change of η shifts. It is not a
per-jet centroid gain.

ParticleNet's antisymmetric share is 16–19 %. Its per-jet gradient aligns with the unweighted centroid and away from the leading
particle. Its symmetric part follows the matched real dilation less closely (R² 0.37 / 0.18).

At a fixed |c|, the per-jet response varies with direction (s.d. 0.33 against a mean of 0.75 in ParT_full). This variation is not
explained by M2 or by the jet's long axis.

## Statistics summary
- **H0 odd = 0.** Sign-flip p 0.0005 for ParT_full, PN_full, PN_kin and untrained. ParT_kin p 0.48–0.53 for the Po coefficient,
  while its φ-axis odd share is 0.10.
- **H0 κ_j uninformative.** Excluded: 0.72 [0.71, 0.73] against 0.10 [0.08, 0.11].
- **H0 b = 2 not better than b ≠ 2** (even part, per-jet scale). Excluded for every trained model (Holm p 0.004).
- **H1 odd part = energy change of η shifts.** Excluded for the odd share (Δ ≤ 0.009). It carries 13–27 % of g_odd (Holm p 0.005).
- **H0 odd gradient ∥ pT centroid beyond null.** Not rejected for ParT_full (p 0.93) and ParT_kin (p 0.13). Rejected for
  ParticleNet (+0.02).

## Files
- Scripts: `interp/p23v_verify.py`, `interp/p23v2_axisonly.py`.
- Results: `results/part/p23_verify.json`, `results/part/p23_verify2.json`.
- Logs: `/j-jepa-vol/aaron-interp/logs/c-p23-verify.log`, `c-p23-verify2.log`, `c-p23-verify2-smoke.log`.
- **Figure check** (`results/part/p23_axis_surface.png`):
  - The left panel (mean bowl, true W) matches the data.
  - The right panel's red series is labelled "odd part (direction-dependent)". It is the gain on Po, which is collinear with jet
    |η| (4), so it is better labelled "gain on Po (|η|-collinear)".

## Explainer entry (verification)
- **Question.** Does the P23 picture hold on new jets? Is ParT's response to an axis offset a function of the angular spread $M_2$,
  with a small direction-dependent part?
- **Method.**
  - Fresh jets were drawn from other files.
  - Per-jet calibration controls: own $\kappa_j$, another jet's $\kappa$, rotation response.
  - A family of moment predictors $S_b(\vec c)=\sum_i p_{T,i}|\vec r_i+\vec c|^b$ was compared.
  - In an input-route control, only $(\Delta\eta,\Delta\phi,\Delta R)$ move.
  - The correlation of the centroid predictor $P_o\approx \vec c\cdot\vec M_1/M_2$ with $\tanh|\eta_{\rm jet}|$ was measured.
- **Result.**
  - The symmetric part is about 92 % of the per-jet variance. It follows $b=2$ and the jet's own real-dilation response.
  - $P_o$ has correlation −0.94 with $\tanh|\eta_{\rm jet}|$, so its gain is not a centroid effect.
- **Why it matters.** The $M_2$ reading of C3 is supported per jet. The comparison with JEDI-linear's centroid term has to use per-jet
  alignment, not the pooled $g_{\rm odd}$.
