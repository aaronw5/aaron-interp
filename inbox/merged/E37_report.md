# E37 — Learned interchange subspace for JEDI-linear's response to two-prong momentum sharing, and its recurrence across networks

Node E37 (stage 3; claims A4s, A4, A10 interchange weakness, A1x-type recurrence). Status: done (one debug attempt: the first full DAS runs were OOM-killed at 2 GiB; fixed a jax compile-cache leak and reran at 1 CPU / 3 GiB).
Script `interp/e37_sharing_das_jedi.py` (arms `--arm das`, `--arm untr`, `--arm recur`), figure `interp/e37_plot.py`.
Jobs: e37-das2 (1 CPU / 3 GiB, wall 2,531 s, peak 2,782 MiB, 0.74 cores), e37-untr2 (1 CPU / 3 GiB, 1,739 s, peak 2,666 MiB), e37-recur (1 CPU / 2 GiB, 1,435 s). The superseded runs e37-das / e37-untr were OOM-killed.
Results: `results/jedi/e37_das.json`, `e37_das_Q.npz` (all learned subspaces Q, stored), `e37_das_M.npz` (refitted M0/M2 subspaces, dilation directions), `e37_untr.json`, `e37_recur.json`; figure `results/jedi/e37_das.png`; logs `logs/e37-das2.log`, `logs/e37-untr2.log`, `logs/e37-recur.log`.

## Conclusion
A single learned direction in the network's 64-unit fourth layer (h4) carries most of JEDI-linear's response to a more lopsided momentum split between the two prongs. Copying that one coordinate from the edited jet into the unedited jet reproduces about 0.8 of the average change in the network's W-versus-Z preference, while random directions reproduce under 0.1, and the same direction carries almost none of the response to a real mass change. The per-jet agreement is weaker: the patched network makes the same W/Z call as on the edited jet in only about 40 % of the jets whose call the edit changes, and a subspace fitted to shuffled targets reaches the same call agreement, so this direction is not a complete stand-alone "sharing variable". The response itself (W jets pushed toward Z, Z jets toward W) appears in 9 of 10 independently trained JEDI-linear networks and in none of 10 untrained ones; which physics observable the direction encodes is unresolved (no observable gives a calibrated read-out).

## Question
E35/E35v found that JEDI-linear's W−Z logit margin moves by about 1.5 log-odds when the two-prong momentum split becomes more lopsided at exactly fixed jet four-momentum, that holding the network's mass direction removes almost none of this, and that no named unit, pair or 3-dim hold carries it. E37 asks: (1) is there a low-dimensional learned subspace (DAS) at S, h4 or h5 whose interchange reproduces the response, and is it specific (not the mass route)? (2) what physics quantity does it read? (3) does the response recur across trained networks?

Thesis prediction (a second, sharing-specific variable in a low-dimensional subspace): k ≤ 3 at h4 or h5 with IIA ≥ 0.7 and ≥ 0.6 of the margin change on held-out jets; random ≤ 0.15; ≤ 0.2 of the dilation response transferred; principal angles to M2 near random; read-out calibrated (slope ≈ 1) for z_min or D2; response in ≥ 7/10 trained networks, not untrained.
Alternative (sharing spread over many directions): IIA rises slowly with k, the subspace also transfers dilation, no calibrated read-out; recurrence ≤ 4/10 would mean a property of the official checkpoint only.

## Definitions
- **Margin** = $\ell_W - \ell_Z$, the difference of the network's W and Z logits (= log of the ratio of W to Z probability). **W/Z call** = sign of the margin.
- **D(dc)** (E35 / H45 edit): every constituent is treated as massless; the jet is split into two exclusive-kT prongs (kT clustering, p = 1, E-scheme); all constituents are boosted to the jet rest frame, where $|\cos\theta^*|$ = cosine between the prong axis and the jet flight direction (0 = equal momentum sharing, near 1 = one prong takes nearly all); every constituent is rotated so that $|\cos\theta^*|$ goes from $c_0$ to $c_1 = \mathrm{clip}(c_0 + dc, 0, 0.95)$, then boosted back. Because one Lorentz transformation is applied to all constituents, the jet four-momentum (pT, η, φ, mass), both prong masses and N are exactly unchanged (measured max relative four-momentum change 6e−13); the prong momentum fractions and lab angles change. dc ∈ {−0.3, −0.15, +0.15, +0.3}; dc > 0 = more lopsided. Mean $c_0$ on test W+Z jets 0.46.
- **cn** (control): rotation of the same rest-frame angle about the prong axis: both prong four-momenta, z and $|\cos\theta^*|$ fixed. It moves constituents by a comparable amount without changing the sharing.
- **dil(f)** (positive control for the mass route): all constituent angles about the axis × f, f = 0.95, 1.05 (a real mass change of about ∓5 %).
- **DAS** (distributed alignment search): learn an orthonormal k-dim subspace Q of a layer; the **interchange** $h' = h_b + QQ^\top(h_s - h_b)$ replaces the base jet's coordinates in Q by the source's and runs the rest of the network (exact fixed-point tail, values clipped to the next quantizer's range). Trained to make the patched margin equal the target margin (mean squared error), Q restricted to live units.
- **Pairs.** Self: base = unedited jet, source = its own D(dc) version, target = network on D(dc). Cross: base b, source = a different jet s, target = network on b edited to $|\cos\theta^*| = c_0(s)$ (the counterfactual "b with s's sharing").
- **IIA** (interchange intervention accuracy) = among pairs where the target W/Z call differs from the base call, the fraction where the patched call equals the target call.
- **frac** = $\sum_c s_c\,\overline{(m_{patch}-m_{base})} / \sum_c s_c\,\overline{(m_{target}-m_{base})}$ over cells c = (dc, class), $s_c$ = sign of the target mean: the share of the average margin change reproduced by the patch.
- **Holm**: Holm–Bonferroni correction over the 15 (stage, k) tests.
- **D2** (β = 1) = $e_3/e_2^3$ (energy correlation functions on all constituents); **τ21** = τ2/τ1 with axes approximated by the hardest particle of the jet / of each kT prong (WTA-like); **z_min** = min(pT1, pT2)/(pT1 + pT2) of the kT prongs; **pT^D** = $\sqrt{\sum p_T^2}/\sum p_T$; **lead_frac** = max pT / Σ pT.
- **Principal angles** between subspaces: arccos of the singular values of $Q^\top M$.

## Procedure and sample
- Network: JEDI-linear official n64, 3-feature, fixed point (`interp.jedi_core`, via `x10_das_jedi.JNet`; exact tail check max |Δlogit| 0 at S, h4, h5).
- Accuracy check (all five classes, stored inputs): 80.59 % on sample(2000, 3702), 80.21 % on sample(2000, 3701) (expected 80.6–81.3 %; 3701 is 0.4 points below the band, 10,000 jets, binomial s.e. 0.4 %). W+Z accuracy on inputs rebuilt about each jet's own axis: 84.9 %.
- Fit F = W+Z jets of sample(2000, 3701) (4,000 jets; 16,000 self pairs, 7,946 cross pairs). Test T = W+Z jets of sample(2000, 3702) minus the 369 rows also in F (3,846 jets; 15,384 self pairs, 3,820 cross pairs). Rows shared by chance with E35/E35v samples: about 370–410 per sample (reported in the JSON).
- DAS: 400 Adam steps (lr 0.01), selection on 20 % of the fit jets. k ∈ {1, 2, 3, 4, 6} at S (19 live of 64), h4 (64), h5 (32).
- Nulls: 1,000 random k-dim subspaces of the live span per (stage, k); DAS with deranged (shuffled) targets; DAS on 3 untrained redraws (Jedi(untrained=True, seed 0–2)) with their own counterfactuals (100 random subspaces each).
- Specificity: same Q on cn and dilation pairs; principal angles between Q and X10-style refitted M0 / M2 subspaces (k = 2; cross-entropy DAS on calls after do(M0) / do(M2) input edits; held-out effective IIA M2 0.82 (S), 0.87 (h4), 0.85 (h5); M0 0.81, 0.83, 0.68) and the dilation direction, vs 1,000 random subspaces.
- Read-out: scalar coordinate u = h·v (v = Q w normalised; w = least squares of margin change on Q-coordinate change, fit pairs); Δu under D regressed on Δobservable (test jets, all dc); natural slope = partial slope of u on the observable across unedited test jets controlling log m and log pT; calibrated slope = edit slope / natural slope; matched-change ratio = mean Δu(+0.3) / (natural slope × mean Δobservable(+0.3)).
- Steering: on unedited test jets, h4 + δ v with δ = c × 1.06 (mean |Δu| under D(+0.3)), c ∈ {±0.5, ±1, ±2}; slope of Δmargin per unit δ vs D's Δmargin per unit Δu; 200 random live h4 directions; dilation direction.
- Recurrence: the 10 trained networks of E33 (official n32 / n64 / n128, perminv n64, 16-feature n64, HGQ r150 s0–s2, float s0–s1) and 10 untrained redraws, on test W+Z jets of sample(2000, 3702) (N 4,000). Per network: mean over jets of Δmargin(D) − Δmargin(cn) for W and Z, 1,000-jet bootstrap CI, sign-flip p, McNemar on per-jet W/Z call changes (D vs cn), Holm over 20. n128: the edit is applied to the 128-slot jets. 16-feature: the 16 columns recomputed from the edited (pT, η, φ) with E33's to16; rotated-frame columns (9, 12, 15) keep unedited values and the absolute η, φ columns carry the small stored-vs-E-scheme axis offset (limitation).
- 95 % intervals: 1,000 jet-bootstrap resamples unless stated.

## Results

### Edit and controls (official n64, test jets)
| edit | W Δmargin | Z Δmargin |
|---|---|---|
| D(+0.3) | −1.58 | +1.63 |
| D(+0.15) | −0.80 | +0.74 |
| D(−0.15) | +0.50 | −0.11 |
| D(−0.3) | +0.65 | −0.06 |
| cn(+0.3) | +0.07 | +0.11 |
| dil × 1.05 | −1.96 | −0.89 |
| dil × 0.95 | −0.20 | +3.72 |
Margin s.d. on test W+Z jets 4.72 log-odds. Reproduces E35 (−1.48 / +1.67) and E35v (−1.54 / +1.55).

### Arm 1: learned interchange subspace (self pairs, held-out jets; n = 1,610 pairs whose W/Z call differs)
| stage | k | frac [CI] | random frac mean / q95 | IIA [CI] | random IIA mean / q95 | p (rank) | Holm | dilation frac (random mean) | shuffled-target frac / IIA |
|---|---|---|---|---|---|---|---|---|---|
| h4 | 1 | 0.83 [0.80, 0.85] | 0.07 / 0.23 | 0.41 [0.38, 0.43] | 0.10 / 0.18 | 0.001 | 0.015 | 0.004 (0.014) | 0.43 / 0.46 |
| h4 | 2 | 0.82 [0.80, 0.84] | 0.11 / 0.32 | 0.32 [0.29, 0.35] | 0.13 / 0.21 | 0.004 | 0.044 | 0.05 (0.03) | 1.13 / 0.24 |
| h4 | 3 | 0.84 [0.81, 0.86] | 0.16 / 0.40 | 0.41 [0.38, 0.44] | 0.15 / 0.25 | 0.002 | 0.026 | 0.05 (0.04) | 1.08 / 0.53 |
| h4 | 6 | 0.84 [0.82, 0.86] | 0.27 / 0.55 | 0.46 [0.43, 0.48] | 0.19 / 0.33 | 0.005 | 0.050 | 0.07 (0.09) | 0.96 / 0.34 |
| h5 | 1 | 0.66 [0.64, 0.68] | 0.08 / 0.25 | 0.49 [0.45, 0.51] | 0.10 / 0.19 | 0.001 | 0.015 | −0.002 (0.04) | 0.66 / 0.22 |
| h5 | 3 | 0.75 [0.73, 0.77] | 0.19 / 0.39 | 0.42 [0.39, 0.45] | 0.18 / 0.33 | 0.020 | 0.14 | 0.09 (0.10) | 0.54 / 0.67 |
| S | 1 | 0.64 [0.61, 0.67] | 0.38 / 1.23 | 0.49 [0.46, 0.52] | 0.35 / 0.57 | 0.19 | 0.93 | −0.001 (0.10) | 0.15 / 0.42 |
| S | 3 | 0.77 [0.74, 0.80] | 0.54 / 1.48 | 0.47 [0.44, 0.50] | 0.45 / 0.62 | 0.46 | 1.0 | −0.02 (0.26) | 0.14 / 0.52 |
(Full table for k = 1, 2, 3, 4, 6 at each stage in `e37_das.json`.)
- Per dc at h4, k = 1: frac 0.91 (+0.3), 0.90 (+0.15), 0.50 (−0.15), 0.54 (−0.3); IIA at +0.3: 0.43.
- Per-jet size: at h4 k = 1 the patch moves the margin by 0.82 log-odds on average (|Δ|) against 1.37 for the full edit (regression slope of patched on full change 0.55); on cn pairs 0.18 vs 0.63; on dilation pairs 0.12 vs 2.13.
- Cross pairs (source = another jet, target = base edited to the source's $|\cos\theta^*|$; n = 328 changed calls): h4 k = 1 IIA 0.29 vs random 0.15 (q95 0.24, p 0.014), frac 0.38; h5 k = 1 IIA 0.32 vs 0.17 (p 0.006).
- IIA does not rise with k (h4: 0.41, 0.32, 0.41, 0.45, 0.46 for k = 1, 2, 3, 4, 6). At S, DAS is within the random band at every k (random S subspaces reproduce 0.4–0.6 of the change on average).
- Principal angles: h4 k = 1 max cosine with the M2 subspace 0.086 (random median 0.16, 95 % range 0.04–0.34; p(closer than random) 0.81), with the dilation direction 0.045 (random median 0.094); h4 k = 3 vs M2 0.22 (random median 0.29). Cosine of the read-out direction v (h4, k = 2) with the dilation direction 0.21.
- Untrained redraws (arm untr): D(+0.3) moves the margin by 0.001–0.004 log-odds (margin s.d. 0.002–0.027); interchange metrics there are computed on sign changes of a near-zero margin and range from 0.0 to 0.99 with no consistent pattern (e.g. seed 0 h4 k = 4 IIA 0.99, seed 2 h4 k = 1 IIA 0.01); they do not provide a usable baseline beyond showing that untrained networks have no comparable response.

### Arm 2: what the coordinate reads
Primary subspace by the pre-set rule (h4, k ≤ 3, best fit-sample validation score): h4, k = 2. Its scalar u (weights on h4 units 24, 9, 12, 11, 4, 56: 0.49, 0.24, −0.24, 0.24, −0.23, 0.23) increases for both dc signs (mean Δu W: +0.70 at −0.3, +0.21 at +0.3), so it is not monotonic in the sharing change; observables explain 27 % of Δu (all five moved observables together).
| observable (h4, k = 2) | Δ at dc +0.3 | edit R² | calibrated slope [CI] | matched-change ratio at +0.3 [CI] |
|---|---|---|---|---|
| z_min | −0.112 | 0.08 | 0.74 [0.70, 0.79] | −0.66 [−0.76, −0.55] |
| D2 | +2.04 | 0.05 | −0.32 [−0.36, −0.27] | −0.47 |
| τ21 | +0.125 | 0.004 | 0.29 [0.18, 0.40] | −1.20 |
| pT^D | +0.031 | 0.06 | 2.12 [1.92, 2.33] | −2.78 |
| lead_frac | +0.033 | 0.06 | 2.49 [2.24, 2.75] | −3.68 |
| prong masses | 0 (fixed by D) | – | – | – |
The h4, k = 1 coordinate is monotonic in dc (mean Δu W −0.73 at −0.3, +2.49 at +0.3; s.d. over jets 1.61) and is better described by the observables (joint R² 0.65; D2 edit R² 0.51, τ21 0.55, z_min 0.30), but no observable is calibrated: D2 calibrated slope 1.26 [1.11, 1.45] with matched-change ratio 2.19 [1.89, 2.56]; z_min 1.94 / 3.32; τ21 3.82 / 4.89; pT^D 8.2 / 16.1. A calibrated read-out would give both numbers ≈ 1.

Steering (h4, k = 2 direction, on unedited jets): Δmargin per unit u: W +0.50, Z +0.18 (both signs equal), while D gives W +0.18, Z +0.48 per unit u with opposite-sign total changes; ratio steer / D: W 2.8 [2.1, 3.9], Z 0.38 [0.32, 0.44]. Random h4 directions of the same length: |slope| q95 0.51 (W), 0.88 (Z); p(|random| ≥ steer) 0.06 (W), 0.71 (Z). The dilation direction gives −1.12 (W), −1.72 (Z). Adding a fixed offset along this direction does not reproduce D's class-dependent (W down, Z up) response.

### Arm 3: recurrence over trained networks (dc +0.3, D minus cn, test W+Z jets, N 4,000)
| network | all-class acc | W: D−cn [CI] (s.d. units) | Z: D−cn [CI] (s.d. units) | McNemar W (D only : cn only) | McNemar Z |
|---|---|---|---|---|---|
| official n64 | 80.6 % | −1.59 [−1.68, −1.51] (−0.34) | +1.51 [1.41, 1.63] (+0.32) | 475 : 42 | 238 : 69 |
| official n32 | 77.3 % | −1.17 (−0.27) | +1.32 (+0.31) | 224 : 48 | 358 : 62 |
| official n128 | 80.5 % | −1.58 (−0.30) | +1.93 (+0.37) | 471 : 45 | 262 : 57 |
| perminv n64 | 81.6 % | −1.63 (−0.30) | +1.84 (+0.34) | 375 : 44 | 187 : 47 |
| 16-feature n64 | 81.3 % | −1.70 (−0.29) | +2.05 (+0.35) | 454 : 29 | 181 : 45 |
| HGQ r150 s0 | 76.3 % | −1.19 (−0.36) | +0.84 (+0.26) | 508 : 114 | 157 : 46 |
| HGQ r150 s1 | 56.7 % | +0.22 [0.16, 0.29] (+0.13) | +0.71 (+0.42) | 396 : 124 | 176 : 66 |
| HGQ r150 s2 | 77.4 % | −0.81 (−0.25) | +1.09 (+0.34) | 272 : 78 | 263 : 41 |
| float s0 | 81.8 % | −2.23 (−0.34) | +1.97 (+0.30) | 610 : 61 | 167 : 34 |
| float s1 | 82.1 % | −1.60 (−0.24) | +2.67 (+0.41) | 314 : 45 | 412 : 41 |
- Pattern (W CI < 0 and Z CI > 0): 9/10 trained (the exception, HGQ r150 s1, is a poorly trained network at 56.7 % accuracy, 32.7 % on W+Z) vs 0/10 untrained; Fisher exact one-sided p = 6e−5; permutation test on the Z−W contrast in s.d. units p = 1e−4 (10,000 permutations). All 20 trained McNemar tests Holm p < 1e−10.
- dc −0.3: the reversed pattern (W > 0, Z < 0, CIs excluding 0) in the same 9/10 trained networks (W +0.41 to +0.77, Z −0.14 to −0.59); HGQ r150 s1 W −0.16, Z −0.23.
- Untrained redraws: |D − cn| ≤ 0.021 log-odds, no W/Z call changes (McNemar 0 : 0); two redraws (7, 8) have constant output.

### Verification and identities
- Full replacement (Q = identity) reproduces the target margin exactly (max |Δ| 0) at every stage, for self, cn and dilation pairs.
- D vs H45 edit_one: E35's edit_jet reused unchanged; jet four-momentum fixed to 6e−13; prong masses Δ = 0 (to numerical precision).
- First full DAS runs were killed for memory (2 GiB); cause: each fit compiled a new jitted function with the training arrays embedded as constants; fixed by passing them as arguments and clearing caches (debug 1).

## Reading (scoped to what was measured)
- Thesis predictions met: ≥ 0.6 of the D margin change reproduced by a k ≤ 3 subspace at h4 (0.83 at k = 1) and h5 (0.66 at k = 1), with random subspaces at 0.07–0.16 (mean) / ≤ 0.40 (q95); ≤ 0.2 of the dilation response transferred (0.004 at h4, k = 1); principal angles to the M2 and dilation subspaces not smaller than random; response in 9/10 trained networks and 0/10 untrained (Fisher p 6e−5).
- Thesis predictions not met: IIA ≥ 0.7 (observed 0.32–0.49 at k ≤ 3; flat in k); a calibrated read-out by z_min or D2 (no observable has both calibrated slope and matched-change ratio near 1).
- Alternative predictions met: IIA flat in k; no calibrated read-out. Alternative predictions excluded: overlap with the mass route (dilation transfer 0.004 vs the 0.2 threshold); low recurrence (9/10, not ≤ 4/10).
- The shuffled-target control reaches the same IIA as the learned subspace at h4 (0.46 vs 0.41 at k = 1) and overshoots the average margin change at k ≥ 2 (frac 0.96–1.13), so IIA at h4 measures whether the patch moves margin-sensitive directions rather than whether Q isolates a sharing variable. The learned Q separates from the shuffled one on average size (0.83 vs 0.43 at k = 1) and on cross-jet transfer.
- The patch reproduces the average change (frac 0.83) but per jet only about half of it (slope 0.55, |Δ| 0.82 vs 1.37), and steering along the direction moves W and Z margins in the same direction, unlike D. This is consistent with the coordinate's effect on the margin depending on the jet's other coordinates (e.g. its mass), not with a single additive sharing variable.
- The response is weaker for less lopsided edits (dc < 0: Z margin −0.06 to −0.11, frac at h4 0.50–0.54).

## Issues
- Sample 3701 accuracy 80.21 %, 0.4 points below the stated band (3702: 80.59 %, inside); same network and code as E35 (80.68 %); attributed to sampling (s.e. 0.4 %).
- Untrained-network DAS is uninformative because the untrained response is 500× smaller and the margin is near-constant.
- 16-feature recurrence keeps the rotated-frame columns (9, 12, 15) at unedited values; the network still shows the pattern.
- τ21 uses hardest-particle axes as a WTA approximation (numba WTA clustering is not available in the jax env).
- Fit and test samples share 369 jets with each other (removed from test) and ~400 each with E35/E35v samples (not removed).

## Explainer entry (for site/explain.py; Paper A tab)
- Question: Where in JEDI-linear does the response to a more lopsided two-prong momentum split (at fixed jet mass) live, and is it the same in other trained JEDI networks?
- Background: E35 showed that making one prong of a W or Z jet carry more of the momentum, with the jet's four-momentum exactly unchanged, pushes W jets toward Z and Z jets toward W by about 1.5 log-odds, and that the network's mass direction does not carry this.
- Method: distributed alignment search: learn a k-dimensional subspace Q of a layer such that copying the edited jet's coordinates in Q into the unedited jet, $h' = h_b + QQ^\top(h_s - h_b)$, reproduces the edited jet's margin; compare with 1,000 random subspaces, shuffled-target fits, the same Q on a real mass change (all angles × 0.95 / 1.05), and principal angles to the mass subspaces; regress the coordinate on jet-shape observables with calibration; repeat the edit on 10 trained and 10 untrained networks.
- Result (figure `results/jedi/e37_das.png`): one h4 direction reproduces 0.83 of the average margin change (random 0.07) and 0.004 of the mass change; per-jet call agreement 0.41 (random 0.10, shuffled-target 0.46); the response recurs in 9/10 trained, 0/10 untrained networks.
- Checks: full replacement exact; Holm over 15; replicate of E35's response on a new sample; cross-jet interchange 0.29 vs 0.15.
- Why it matters: it shows the sharing response runs through a direction separate from the mass route and is a property of trained JEDI-linear networks in general, while leaving open which jet-shape variable that direction computes.
- New glossary terms: DAS, IIA, interchange, frac (share of the margin change reproduced), principal angle, z_min, pT^D, steering.
