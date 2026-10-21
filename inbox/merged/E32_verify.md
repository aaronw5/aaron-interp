# E32 verification: JEDI-linear on trigger-style axis offsets

Script: `interp/e32_verify.py`. Cluster job `e32-verify`: 1 CPU / 2 GiB, 309 s wall, 1,410 MiB peak, exit 0. A smoke job `e32-vsmoke` ran first.
Results: `$AI_RESULTS/jedi/e32_verify.json` (local copy `results/jedi/e32_verify.json`). Log: `logs/e32-verify.log`.

## Conclusion
On a new set of jets, the trigger-style axis (E-scheme axis computed from particles snapped to 0.087-wide calorimeter towers) again moves the network's W-versus-Z decision about 0.9 as far as a real mass change of the same predicted size would.

A new control confirms that this response comes from the pT-weighted spread about the axis. If the jet is translated but its spread about the new axis is then restored, the W-to-Z call changes fall back to the level seen for a plain rotation.

The reading's statement that "no direction term is seen" at this offset size does not hold. The response to an offset c differs from the response to −c. That direction-dependent part is about half the size of the direction-independent part, about three times what the spread formula predicts.

The earlier test that compared the trigger offset with an offset in a random direction could not detect this, because both offset directions are effectively random. The sign-match statistic is also uninformative, because 99 % of W-called jets have a larger spread after the edit.

## What was checked

### 1. Code review
Files read: `interp/e32_seeded_axis_jedi.py` (parts main / c2st34 / rep) and `interp/e32_plot.py`.

- **Edits.** `reaxis` shifts only real particles (pT > 0) and wraps Δφ. Padding slots (pT = 0) are left at their stored values. pT and slots are unchanged.
- **Jet mass.** The four-vector jet mass is unchanged by T to 4e-8 in log (checked in the verify job), as it should be: an η shift is a boost and a φ shift is a rotation.
- **E-scheme axis.** `escheme` sums massless four-vectors in the stored relative frame. Rapidity differences are boost-invariant, so this is legitimate.
- **Tower quantiser.** Each particle goes to the centre of its tower, with a random grid phase per jet. The largest per-particle shift is t/2, as expected.
- **Matched dilation.** C1 matches Δlog M2 to 8e-8.
- **Bootstrap and tests.** Bootstrap weights, McNemar and Holm are implemented correctly.

No code bug was found that changes the reported numbers. The design issues are these:

1. **Sign-match column uninformative.** "100 % of T1 W→Z flows have ΔM2 > 0" carries no information. Among W-called jets, the base rate of ΔM2 > 0 under T1 is 99.2 % (verify sample). Under T0 the base rate of ΔM2 < 0 is 98.7 % (main sample).
2. **C2 cannot detect a direction term.** C2 (a translation by the same |c| in a uniformly random azimuth) can only show that T1's direction is not special. T1's offset direction comes from a random tower grid phase, so it is itself close to random. A per-jet term that is odd in c has random sign under both T1 and C2. It drops out of the projection R (defined in section 2) and of the call counts in the same way. Calls under T1 against C2 (p 0.41) therefore do not test for a direction term.
3. **Selection leakage in the freezes.** E32's freeze ranked the h4/h5 units on half A and then evaluated on the full sample, half A included. The verify run evaluates on half B only.
4. **R as a projection.** R = Σ e·d / Σ d² is a projection coefficient. It ignores the part of the response that is not along the dilation response. The share of the response that lies along the dilation direction was not reported in E32. The verify run adds it.
5. **"Error an L1 trigger would make" is an approximation.** T1 is an E-scheme axis computed on L1-sized (0.087) towers, with no threshold, no pileup and no energy resolution. CMS Phase-1 L1 jets place the jet at the seed-tower centre (a 9×9 sliding window), which gives a grid-quantised offset of up to about 44 mrad per coordinate. So "the error an L1 calorimeter trigger would make" and the figure title "the axis error a trigger makes" overstate how close the model is to a trigger.

The figure (`results/jedi/e32_seeded_axis.png`) was checked. The left panel shows one W jet with the T1 axis at 18 mrad and the T3 axis at 15 mrad. The right panel shows R against the quintile of |c|, with the C3 null and the R = 1 line, for both samples, and matches the JSON. Only the title wording in point 5 is affected.

### 2. Rerun on a fresh sample, plus the missing controls

**Network.** Official JEDI-linear 3-feature n64, exact `jedi_core` forward pass.

**Sample.** `sample(5000, 3233)` minus 4,584 jets that are also in `sample(5000, 3201)` or `sample(5000, 3202)`. This leaves N = 20,416 jets, disjoint from E32's main and replicate samples. Unedited accuracy is 80.80 %.

**Edits.** Built with E32's own `build()` (rng seed 3234):
- T1: E-scheme axis of particles moved to the centres of 0.0873 towers.
- T2: the same with 0.0436 towers.
- T3: axis on the hardest particle.

**Controls.**
- C1: matched dilation. Every real particle's (Δη, Δφ) is multiplied by sqrt(M2'/M2), so Δlog M2 matches the edit's.
- C2: translation by the same |c| in a random azimuth.
- C3: random rotation about the stored axis.

**Symbols.**
- M2 = Σ pT r², with r measured about the axis the network sees.
- e = the change of the W−Z logit margin under the edit; d = its change under C1.
- R = Σ e·d / Σ d² over jets called W or Z (7,988 jets). R is the network's response to the axis change, measured along its response to a real mass change of the same predicted size. R = 1 means the same response; R = 0 means no response along that direction.
- 95 % intervals: 500 Poisson bootstrap resamples over jets.

**New controls.**
- **TC (translation with the spread restored).** Apply T, then scale every real particle's angles about the new axis by sqrt(M2/M2'). This makes M2 about the axis the network sees exactly equal to its unedited value (to 8e-8). The centroid stays displaced, and the true (recentred) mass drops by Δlog m = −0.072 (mean, T1).
  - Predictions for R_TC under A4 (network reads M2 about the given axis): 0. Under a recentred or true-mass reader: −1. Under a mixture w·A4 + (1−w)·recentred: R_T = w and R_TC = −(1−w), so R_T − R_TC = 1.
- **Tm (reflected offset).** The axis is placed at −c instead of +c, and gets its own matched dilation C1m.
  - Odd part e_odd = (e(+c) − e(−c))/2; even part e_even = (e(+c) + e(−c))/2.
  - A pure function of r² about the axis has an odd part only through the small c·M1 term (M1 = Σ pT (Δη, Δφ), the pT-weighted centroid offset). The dilation responses predict that odd part.
- **Explained share.** 1 − Σ(e − R d)² / Σ e² is the fraction of the margin response that lies along the dilation response.
- **Freeze without leakage.** Rank the 4 h4 and 4 h5 live units with the largest standardised response to a 5 % M2 dilation on half A. Then evaluate on half B only: 855 T1-changed jets plus random jets, n = 2,666. Hold these units at their unedited per-jet values while T1 is applied. Null: 100 random holds of 8 live h4/h5 units. Report the rank p for both calls and margin.

### 3. Results (verify sample 3233; E32 main / replicate in brackets)

| Quantity | T1 (0.087 towers) | T2 (0.044 towers) | T3 (hardest particle) |
|---|---|---|---|
| \|c\| median, mrad | 18.3 [18.3 / 18.3] | 7.8 [7.9 / 8.0] | 41.4 [41.2 / 41.5] |
| calls changed, T | 8.53 % [8.14, 8.89] [8.69 / 8.49] | 2.88 % [2.90] | 31.4 % [31.2 / 31.8] |
| calls changed, C1 / C2 / C3 | 6.16 / 8.82 / 4.33 % | 1.18 / 2.91 / 4.24 % | 33.0 / 35.2 / 4.21 % |
| W→Z, T vs C1 (McNemar) | 716 vs 718 (191:193, p 0.96) [921 vs 858 / 823 vs 852] | 136 vs 94 (p 3e-5) | 2142 vs 2142 (p 1.0) |
| R_margin, T | 0.898 [0.877, 0.920] [0.91 / 0.89] | 0.794 [0.731, 0.848] [0.76 / 0.74] | 0.877 [0.869, 0.884] [0.88 / 0.89] |
| R_margin, C2 / C3 null | 0.924 / 0.056 | 0.800 / 0.226 | 0.952 / 0.014 |
| explained share by dilation, T (C3) | 0.70 [0.68, 0.71] (0.011) | 0.21 [0.18, 0.24] (0.010) | 0.91 [0.91, 0.91] (0.006) |
| **TC: R** | **0.041 [0.023, 0.058]** | 0.229 [0.165, 0.281] (≈ C3 0.226) | 0.126 [0.120, 0.133] |
| TC: calls changed; W→Z / Z→W | 5.31 % (C3 4.33 %, McNemar p 3e-8); 104 / 107 (C3 112 / 114) | 2.75 %; 71 / 71 | 11.3 %; 110 / 478 |
| R_T − R_TC (mixture predicts 1) | 0.857 [0.837, 0.879] | 0.565 [0.534, 0.597] | 0.750 [0.741, 0.760] |
| R_margin, Tm (own matched dilation) | 0.925 [0.902, 0.947] | 0.777 | 0.975 [0.968, 0.982] |
| **rms(e_odd)/rms(e_even), network** | **0.46 [0.45, 0.48]** | 0.81 [0.79, 0.84] | 0.21 [0.20, 0.21] |
| rms odd/even predicted from M2 (dilations) | 0.14 [0.13, 0.15] | 0.32 [0.30, 0.33] | 0.05 [0.05, 0.05] |
| T vs Tm: calls (McNemar) | 801:921, p 0.004 | 363:416, p 0.06 | 992:1781, p 3e-51 |
| T vs Tm: W→Z (McNemar) | 280:349, p 0.007 | 73:68, p 0.74 | 436:476, p 0.20 |
| odd part ~ a_η c_η + a_φ c_φ, margin per rad | a_η −17.2 [−18.9, −15.5], a_φ −11.1 [−12.7, −9.4], R² 0.10 | −15.0, −9.6 | +0.9, +1.0 |
| W-called jets with ΔM2 > 0 (sign-match base rate) | 99.2 % | 96.4 % | 100 % |

#### Freeze of the 8 most M2-responsive h4/h5 units under T1, evaluated on half B only
Units ranked on half A: h4 47, 6, 50, 37 and h5 20, 26, 7, 0. E32 ranked h4 47, 6, 50, 59 and h5 20, 26, 7, 31 (main) or 0 (replicate).

| Measure | Removed by the freeze | Random holds q50 / q97.5 | Rank p |
|---|---|---|---|
| Calls | 47.1 % | 16.6 / 42.1 % | 0.0099 (the floor with 100 nulls) |
| Margin | 77.2 % | 7.3 / 51.3 % | 0.0099 |

Holm over the 4 holds of E32: 4 × 0.0099 = 0.040 for this hold.

### 4. Hypotheses and which are excluded

#### Mass variable on the T1 axis
- H0 (recentred / true mass): R_T = 0 and R_TC = −1. Excluded: R_T 0.898, R_TC 0.041, bootstrap p 0.004 for each.
- H1 (A4, M2 about the given axis): R_T = 1 and R_TC = 0.
  - R_TC = 0.041 is close to the A4 prediction, and TC's W→Z flows (104) are at the rotation-null level (112).
  - R_T = 0.898 is below 1 (p 0.004).
  - R_T − R_TC = 0.86 rather than 1, so the mixture model is not exactly additive. The 0.10–0.14 shortfall is not explained by a recentred-mass component.

#### Direction term on the T1 axis
- H0 (no odd term beyond what M2 predicts): rms(e_odd)/rms(e_even) = 0.14 (the dilation prediction).
- H1 (direction-dependent term): the ratio is above 0.14.
- Observed: 0.46 [0.45, 0.48]. H0 is excluded (the intervals do not overlap). The paired call test agrees: T against Tm, McNemar p 0.004 for calls and p 0.007 for W→Z.
- The global linear part of the odd term has the same sign pattern as E6v (larger response toward −η and −φ). It explains only 10 % of the odd variance, so most of the odd part depends on the individual jet.

#### Freeze
H0: the M2-responsive units are no more important than random h4/h5 units. Excluded at p 0.0099 for both calls and margin on held-out jets. Holm over 4 gives 0.04 (the other three E32 holds have p ≥ 0.10).

## Statements in the E32 reading

| Statement | Verify result |
|---|---|
| R ≈ 0.89–0.91 for T1 on disjoint samples | Reproduced: 0.898 [0.877, 0.920]. |
| W→Z flow equals the dilation's within 7 % | Reproduced: 716 vs 718, p 0.96. The E32 main sample had p 0.006 (921 vs 858). |
| W→Z flows are sign-matched to ΔM2 | True but uninformative (base rate 99.2 %). Should be dropped as evidence. |
| Random-direction translation gives the same calls and R, "so no direction term is seen at this offset size" | **Refuted.** The reflected-offset control finds an odd, direction-dependent margin response of 0.46 × the even part. That is 3.3 × the M2-predicted odd part (paired p 0.004 for calls). C2 cannot detect such a term. The g-freeze margin result (3 %) concerns the context layer g, not the direction term. |
| "Changes its W/Z call almost exactly as it would for a real mass change" | Holds for W→Z counts and the projection R. Overall, T1 changes 8.5 % of calls against 6.2 % for C1 (McNemar p 4e-42). The excess is mostly g→q (283 vs 52) and is shared with C2. Only 70 % of the T1 margin-response variance lies along the dilation response. |
| Mass variable = pT-weighted spread about whatever axis it is given | Supported more directly by TC: restoring M2 about the new axis removes the W→Z response (R 0.04; flows at the rotation level). |
| 8 h4/h5 freeze: margin passes, calls do not pass Holm | On held-out jets both pass at the p floor 0.0099. The call-level Holm-4 result is 0.04 here, against 0.10 / 0.20 in E32. |
| T2 R 0.74–0.76 | Reproduced: 0.79. Only 21 % of the T2 response lies along the dilation, and R_TC (0.23) equals the rotation null (0.23). At 8 mrad the mass-like part is a minority of the response. |
| T3 R 0.88 | Reproduced: 0.877. Direction matters here: the reflected axis gives 0.975 and the random direction 0.952. |
| "The error an L1 calorimeter trigger would make" | Overstated. The model is an E-scheme axis on L1-sized towers, not the L1 seed-tower-centre position, and has no threshold, pileup or resolution. |

Not rerun: T0 and T4, C2ST and support, untrained redraws, the R_h4 read-out, and the g and S0+S29 freezes.

## Corrected reading
On an axis computed E-scheme from particles quantised to 0.087 towers (median offset 18 mrad), JEDI-linear's W−Z margin moves 0.90 [0.88, 0.92] along its response to a matched dilation. This holds on three disjoint samples (0.91, 0.89, 0.90), and the W→Z flow equals the dilation's (716 vs 718, p 0.96).

Translating the jet but restoring M2 about the new axis gives R 0.04 [0.02, 0.06], with W→Z flows at the rotation-null level. So the W/Z response to the axis error passes through M2 about the given axis.

On top of this mass-like part, the response has a direction-dependent (odd in c) part. Its rms is 0.46 × the even part, against 0.14 predicted from M2 alone. It shows in the calls (McNemar +c vs −c, p 0.004). C2 (random-direction translation) cannot detect it, because T1's offset direction is itself close to random.

The mass-like part accounts for 70 % of the T1 margin-response variance, and T1 changes 2.4 percentage points more calls than the matched dilation, mostly g→q. At 8 mrad (0.044 towers) only 21 % of the response is mass-like.

Holding the 8 most M2-responsive h4/h5 units at their unedited per-jet values removes 77 % of the margin response and 47 % of the call changes on held-out jets. The random-hold q97.5 values are 51 % and 42 %, p 0.0099 for each.

The tower model is an E-scheme axis on L1-sized towers without threshold or pileup. It is not the CMS L1 seed-tower position.
