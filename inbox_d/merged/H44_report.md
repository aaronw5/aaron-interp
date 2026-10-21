# H44 — b vs c in JetClass H→bb / H→cc: track-displacement size and displaced-system mass at fixed displaced-track counts

Node H44 (loop D, round D9; extends D2 / D14, exploratory). Script `interp/h44_bc_lifetime.py`. Jobs: `d-h44-smoke2` (CPU, smoke run, 549 s), `d-h44-run` (CPU 8 / 16 GiB, 3,180 s, peak memory 4.9 GiB, 7.5 cores on average), `d-h44-fig` and `d-h44-fig2` (figure only, 72 s). The GPU smoke job `d-h44-smoke` stayed Pending (no free GPU) and was deleted before it started. Results: `results/physics/h44_bc_lifetime.json`, `h44_perjet.npz`; figures `results/physics/h44_jet_edit.png` and `h44_response.png`; logs `logs/d-h44-run.log`, `logs/d-h44-smoke2.log`, `logs/d-h44-fig.log`.

## Conclusion

Both full-input JetClass taggers, ParT and ParticleNet, use how far the displaced tracks are from the collision point to separate H→bb from H→cc, not only how many displaced tracks there are. Shrinking each displaced track's impact parameter while keeping it inside its significance bin, so that every count above 1, 2, 3, 5 and 10σ stays the same, moves H→bb jets toward H→cc. The shift is about 15–18 % of the shift from shrinking the impact parameters without that constraint, it grows with the size of the edit, and networks with randomly permuted weights do not show it. This is the direction expected from the longer B-hadron lifetime, so D2's "multiplicity, not lifetime" describes bb vs QCD but not bb vs cc. The test of the displaced-track mass (a proxy for the vertex mass) gave no single direction across networks and jet classes. The control for that test, which moves the displacement onto other tracks at fixed mass, moved the score by about one log-odds unit by itself, so the vertex-mass question is still open.

## Question and hypotheses

Do ParT_full and ParticleNet_full (PN_full) use the size of the track displacement (lifetime) or the mass of the displaced-track system (vertex mass) to separate H→bb from H→cc, beyond the number of displaced tracks? Earlier work (D2, 400 jets per class) found "multiplicity, not lifetime", but only for bb vs QCD. b vs c is the physical case where lifetime and vertex mass differ: cτ is 455–490 µm for B hadrons and 123–312 µm for D hadrons, and the b vertex mass reaches about 5 GeV while the c vertex mass stays below about 2 GeV.

- H0 (count only, D2 extends): the count-preserving edits Lc and V − C1 give responses inside the permuted-network band and ≤ 10 % of the response to L(λ); the L response equals the prediction of LR_count.
- H1a (lifetime): Lc(0.4) moves H→bb jets toward cc, ordered in λ, by about the LR_life prediction. The planner expected a shift of order −0.5 log-odds.
- H1b (vertex mass): lowering the displaced-system mass (Vm − C1m) moves H→bb jets toward cc, by about the LR_vm prediction.

## Definitions

- **Charged track**: a real particle with charge ≠ 0.
- **S** = |d0val| / d0err: the transverse impact-parameter significance, i.e. the distance of closest approach of the track to the beam line in the transverse plane, divided by its uncertainty. d0val is a JetClass input; the networks see tanh(d0val), d0err, tanh(dzval) and dzerr. They do not see S itself.
- **Displaced track**: a charged track with S > 2. **Prompt track**: a charged track with S ≤ 2. **n(S>s)**: the number of charged tracks with S > s.
- **S bins**: [0,1], (1,2], (2,3], (3,5], (5,10], (10,∞). The upper edge belongs to its bin, so each n(S>s) is a sum of bins.
- **Subjets**: two exclusive-kT subjets of each jet (`h41_colour_flow.ktn_batch`).
- **m_D,k**: the invariant mass of the summed stored four-momenta of the displaced tracks in subjet k.
- **m_SV**: max_k m_D,k, used as a proxy for the secondary-vertex mass. Its subjet is called s\*. The proxy is per subjet because an H→bb jet holds two B hadrons, and a jet-wide displaced mass would mix both vertices. The jet-level displaced mass is also computed (unedited means 36.1 GeV for bb, 16.5 GeV for cc, 3.5 GeV for QCD).
- **IP tuple**: the four impact-parameter inputs of one track, (d0val, d0err, dzval, dzerr).
- **s_bc** = log(p_bb / p_cc) = L_Hbb − L_Hcc, from the 10-class logits L. **s_bq** = log(p_bb / p_QCD). The **bb-vs-cc call** is s_bc > 0.
- **Response**: the change of the score for each jet (edited minus unedited, both rebuilt the same way), averaged over jets. A contrast "V − C1" is the per-jet difference of the two responses, averaged over jets.
- **McNemar test**: compares the numbers of jets whose bb-vs-cc call flips in each direction (bb → cc against cc → bb) between two paired versions of the same jets; exact binomial p.
- **Holm correction**: a step-down correction of p-values for multiple tests that controls the family-wise error rate.
- **Permuted band**: the [min, max] of the mean response of 6 copies of ParT_full in which every weight tensor's entries are randomly permuted (torch seeds 5–10). These networks have the same architecture and weight values but no trained function.

## Procedure

**Sample.** JetClass test_20M `HToBB_100`, `HToCC_100` and `ZJetsToNuNu_100` (QCD): 2,000 random rows each (rng 4401 / 4402 / 4403; no first-k rows), N = 6,000. Every variant, the identity included, was rebuilt with `part_core.build` (η = jet_eta + sign(jet_eta)·Δη). No four-momentum was changed. Only the IP tuples of charged tracks were changed.

**Edits.**
- **L(λ)**, λ ∈ {0.4, 0.7, 1.4, 2.0}: d0val and dzval of every displaced track multiplied by λ, errors kept, so S → λS. λ < 1 mimics the shorter charm lifetime. This also changes n(S>s); for example, L(0.4) lowers n(S>2) of bb jets by 2.08 on average.
- **Lc(λ)**, count-preserving lifetime scaling (displaced tracks only). In a finite bin (lo, hi], set u = ln(S/lo)/ln(hi/lo), then u' = u^(1/λ) and S' = lo·(hi/lo)^u'. In the open bin (10, ∞), S' = 10 + λ(S − 10). d0val and dzval are multiplied by S'/S, and errors are kept. The map is monotone and keeps every track in its S bin, so n(S>1, 2, 3, 5, 10) are all unchanged (asserted for every jet). λ < 1 pushes values toward the lower bin edge and shrinks the tail above 10 by exactly λ; λ > 1 does the opposite. In 37 tracks (out of all displaced tracks in 6,000 jets) the float32 rounding of Lc(0.4) would have changed the bin; these tracks were left unedited. The planner's wording was "the same scaling followed by a monotone remap". An affine remap of a scaled bin back onto the same bin is the identity in ln S, so the in-bin power map above was used instead.
  - Effect on H→bb jets: Lc(0.4) lowers mean median ln S by 0.28 and mean max ln S by 0.72, with every count unchanged. Lc(2.0) raises them by 0.28 and 0.59. For comparison, L(0.4) changes them by −0.38 and −0.91 and also removes 2.1 displaced tracks.
- **Vm / Vp**, displaced-system mass edit: inside subjet s\*, k ∈ {1, 2} displaced tracks exchange their IP tuple with k prompt tracks of the same subjet, chosen from its 15 hardest prompt tracks. The exchange is chosen so that m_D,s\* falls by ≥ 1 GeV (Vm) or rises by 1–4 GeV (Vp). Among valid exchanges, the one with the smallest pT mismatch |ln(ΣpT in / ΣpT out)| is taken. The S multiset, every n(S>s) and both subjets' displaced counts are unchanged (asserted).
- **C1m / C1p** (control): the same subjet and the same k as Vm / Vp, but an exchange with |Δm_D,s\*| ≤ 5 %, again with the smallest pT mismatch. C1 is the comparison that holds the count, the S multiset and m_SV fixed while still moving displacement between tracks.
- **Cr** (control): k random displaced ↔ prompt exchanges inside s\* (rng 4410), with the mass change unconstrained.
- The V sets contain only jets where both V and C1 exist: Vm 1,333 / 927 / 520 jets and Vp 1,427 / 1,146 / 674 jets (bb / cc / QCD).
- On bb jets: mean Δm_SV is −2.38 GeV for Vm, +0.004 for C1m, +1.88 for Vp and +0.009 for C1p; mean k is 1.1. The mean pT mismatch is 0.71 for Vm against 0.34 for C1m, 0.54 for Vp against 0.44 for C1p, and 1.35 for Cr.

**Models.**
- ParT_full and PN_full (with IP inputs) are the networks tested.
- ParT_kinpid, PN_kinpid, ParT_kin and PN_kin have no IP inputs, so their response is zero by construction.
- 6 weight-permuted copies of ParT_full give the null band.

**References.** sklearn models trained on random JetClass train_100M rows, `HToBB_000` / `HToCC_000`, 20,000 each (rng 4404 / 4405), with an 80/20 split (rng 4406), labelled bb = 1:
- LR_count: logistic regression on n(S>2) and n(S>3).
- LR_life: logistic regression on n(S>2), the median ln S of displaced tracks and the max ln S.
- LR_vm: logistic regression on n(S>2) and m_SV.
- GBDT (HistGradientBoosting): on n(S>1, 2, 3, 5, 10), median and max ln S, Σ ln(S/2), m_SV and the jet-level displaced mass.

Each reference was scored on the same edited IP tuples, with its features recomputed. Its change in bb-vs-cc log-odds is the response that its hypothesis predicts.

**Statistics.**
- Each cell (model × edit × class × score) gives the mean over jets with a 95 % bootstrap interval (2,000 resamples).
- For ParT_full and PN_full, each tested contrast (L×4, Lc×4, Vm − C1m, Vp − C1p, Cr − C1m) also gets a two-sided sign-flip permutation p (5,000 flips; H0: mean = 0).
- Holm correction runs over all 132 tested cells (2 networks × 11 contrasts × 3 classes × 2 scores). The smallest attainable Holm p is about 0.03.
- McNemar tests on bb-vs-cc calls compare each edit with the identity (L, Lc) and V with C1.

## Unedited accuracy on this sample (N = 6,000; 10-class argmax; bb-vs-cc = s_bc > 0 on the 4,000 bb + cc jets)

| model | 10-class acc | recall Hbb / Hcc / QCD | bb-vs-cc acc | bb-vs-cc AUC |
|---|---|---|---|---|
| ParT_full | 84.9 % | 92.0 / 84.3 / 78.5 % | 98.0 % | 0.998 |
| PN_full | 82.6 % | 91.0 / 80.1 / 76.8 % | 97.1 % | 0.996 |
| ParT_kinpid | 72.6 % | 71.4 / 70.3 / 76.1 % | 86.2 % | 0.942 |
| PN_kinpid | 67.7 % | 66.5 / 63.2 / 73.3 % | 82.3 % | 0.907 |
| ParT_kin | 65.3 % | 62.8 / 60.3 / 72.8 % | 80.0 % | 0.885 |
| PN_kin | 58.8 % | 53.4 / 53.2 / 69.9 % | 74.7 % | 0.821 |
| ParT_perm1–6 | 0.0–31.8 % | — | 49.2–50.3 % | 0.44–0.53 |

The orderings full > kinpid > kin and ParT > PN match the published JetClass results (ParT_full 86.1 % over all 10 classes). ParT_kin's 65.3 % matches H41's 65.6 % on a different 4-class sample. The kin/kinpid networks gave max |Δlogit| = 0.000 for every edit, as required. Scoring the raw stored jets instead of the rebuilt identity changed ParT_full's logits by at most 0.53 (H35: 1.05). Every response here is edited-rebuilt minus identity-rebuilt.

References, held-out bb-vs-cc accuracy (train rows) and accuracy on this sample's bb + cc jets:

| reference | held-out acc | acc on this sample |
|---|---|---|
| LR_count | 81.5 % | 82.4 % |
| LR_life | 81.4 % | 82.8 % |
| LR_vm | 79.8 % | 81.1 % |
| GBDT | 84.9 % | 85.4 % |

Standardised coefficients:
- LR_count: n(S>2) +0.59, n(S>3) +1.86.
- LR_life: n(S>2) +2.66, median ln S +0.25, max ln S −0.71.
- LR_vm: n(S>2) +2.45, m_SV −0.31.

At fixed n(S>2), a larger maximum S and a larger m_SV both lower these fits' bb log-odds. One possible cause is that the displaced tracks of a bb jet are shared between two B hadrons. This was not tested.

Unedited means (bb / cc / QCD):

| observable | bb | cc | QCD |
|---|---|---|---|
| n(S>2) | 8.96 | 4.21 | 2.23 |
| n(S>10) | 5.30 | 1.92 | 1.22 |
| median ln S | 2.66 | 2.51 | 2.59 |
| max ln S | 5.28 | 4.44 | 3.54 |
| m_SV | 6.36 GeV | 3.56 GeV | 1.72 GeV |

## Numbers (Δ s_bc = change in log(p_bb/p_cc); negative = toward cc; 95 % bootstrap CI; Holm p for the networks)

### H→bb jets (N = 2,000)

| edit | ParT_full | PN_full | permuted band | LR_count | LR_life | LR_vm | GBDT |
|---|---|---|---|---|---|---|---|
| L(0.4) | −1.50 [−1.58, −1.42] (0.03) | −1.81 [−1.90, −1.71] (0.03) | [−0.06, +0.04] | −1.35 | −1.32 | −1.31 | −1.48 |
| L(0.7) | −0.41 [−0.44, −0.38] (0.03) | −0.52 [−0.56, −0.48] (0.03) | [−0.04, +0.03] | −0.52 | −0.63 | −0.61 | −0.53 |
| L(1.4) | +0.08 [+0.05, +0.11] (0.03) | +0.18 [+0.13, +0.22] (0.03) | [−0.02, +0.01] | +0.44 | −0.04 | 0 | +0.38 |
| L(2.0) | −0.04 [−0.10, +0.01] (1) | +0.18 [+0.11, +0.26] (0.03) | [−0.03, +0.03] | +0.57 | −0.08 | 0 | +0.50 |
| Lc(0.4) | −0.226 [−0.249, −0.202] (0.03) | −0.323 [−0.367, −0.275] (0.03) | [−0.041, +0.031] | 0 | +0.172 | 0 | +0.176 |
| Lc(0.7) | −0.070 [−0.081, −0.059] (0.03) | −0.119 [−0.143, −0.094] (0.03) | [−0.018, +0.010] | 0 | +0.068 | 0 | +0.089 |
| Lc(1.4) | +0.033 [+0.019, +0.044] (0.03) | +0.078 [+0.052, +0.104] (0.03) | [−0.030, +0.002] | 0 | −0.064 | 0 | −0.104 |
| Lc(2.0) | +0.030 [+0.004, +0.054] (0.7) | +0.125 [+0.075, +0.173] (0.03) | [−0.042, +0.014] | 0 | −0.132 | 0 | −0.267 |
| Vm − C1m (n 1,333) | −0.05 [−0.17, +0.07] (1) | +0.29 [+0.14, +0.44] (0.03) | [−0.004, +0.059] | 0 | 0 | +0.125 | +0.125 |
| Vp − C1p (n 1,427) | −0.52 [−0.63, −0.39] (0.03) | −0.10 [−0.24, +0.04] (1) | [−0.02, +0.15] | 0 | 0 | −0.106 | −0.216 |
| Cr − C1m (n 1,333) | −0.50 [−0.62, −0.38] (0.03) | −0.25 [−0.39, −0.11] (0.04) | [−0.03, +0.07] | 0 | 0 | −0.014 | −0.080 |
| C1m alone | −1.06 [−1.15, −0.96] | −1.67 [−1.81, −1.54] | [−0.09, +0.01] | 0 | 0 | 0 | 0 |
| C1p alone | −1.11 [−1.20, −1.01] | −1.71 [−1.85, −1.59] | [−0.13, +0.01] | 0 | 0 | 0 | 0 |

Ratio of the Lc response to the L response at the same λ:

| network | λ = 0.4 | λ = 0.7 | λ = 1.4 |
|---|---|---|---|
| ParT_full | 0.15 [0.14, 0.16] | 0.17 [0.15, 0.19] | 0.42 [0.29, 0.66] |
| PN_full | 0.18 [0.16, 0.20] | 0.23 [0.19, 0.27] | 0.44 [0.32, 0.57] |

The ratio at λ = 2.0 is undefined for ParT, whose L(2.0) response is about 0.

McNemar tests for ParT_full on bb jets (bb → cc against cc → bb call flips):
- L(0.4): 80 against 1 (p 7e−23). The bb call rate falls from 97.1 % to 93.1 %.
- Lc(0.4): 6 against 5 (p 1). The call rate stays at 97.1 %.
- Vm against C1m: 28 against 22 (p 0.5).

So on bb jets Lc moves the score but changes almost no calls; 97 % of these jets are called bb with a large margin.

### H→cc jets (N = 2,000)

| edit | ParT_full | PN_full | permuted band | LR_count | LR_life | LR_vm | GBDT |
|---|---|---|---|---|---|---|---|
| L(0.4) | −0.24 (0.03) | −0.27 (0.03) | [−0.02, +0.02] | −0.80 | −0.90 | −1.00 | −0.84 |
| L(2.0) | +1.07 [+1.00, +1.14] (0.03) | +0.90 (0.03) | [−0.01, +0.01] | +0.55 | −0.08 | 0 | +0.58 |
| Lc(0.4) | −0.019 [−0.036, 0.000] (1) | −0.027 (0.2) | [−0.015, +0.035] | 0 | +0.121 | 0 | +0.141 |
| Lc(0.7) | −0.029 (0.03) | −0.035 (0.03) | [−0.006, +0.003] | 0 | +0.048 | 0 | +0.057 |
| Lc(1.4) | +0.057 (0.03) | +0.063 (0.03) | [−0.003, +0.011] | 0 | −0.046 | 0 | −0.050 |
| Lc(2.0) | +0.151 [+0.134, +0.169] (0.03) | +0.161 [+0.142, +0.180] (0.03) | [−0.007, +0.014] | 0 | −0.096 | 0 | −0.100 |
| Vm − C1m (n 927) | −0.31 [−0.48, −0.14] (0.03) | −0.12 [−0.23, 0.00] (1) | [−0.006, +0.016] | 0 | 0 | +0.122 | −0.043 |
| Vp − C1p (n 1,146) | −0.11 [−0.26, +0.04] (1) | −0.11 [−0.22, 0.00] (1) | [−0.05, +0.01] | 0 | 0 | −0.104 | +0.119 |
| C1m alone | +1.31 | +0.68 | — | — | — | — | — |

### QCD jets (N = 2,000)

- Lc responses are ≤ 0.03 in |Δ s_bc|. Lc(0.4) gives ParT −0.004 [−0.012, +0.006].
- V − C1 CIs include 0 for both networks.
- L(2.0): ParT +0.39, PN +0.38, LR_count +0.39.

### Score bb vs QCD (Δ s_bq) on H→bb jets

| edit | ParT_full | PN_full |
|---|---|---|
| L(0.4) | −0.47 | −0.46 |
| Lc(0.4) | −0.034 [−0.045, −0.024] | −0.039 |
| Vp − C1p | −0.13 | −0.11 |

The lifetime response of the bb-vs-QCD score at fixed counts is about 7 % of the bb-vs-cc one.

Holm: 96 of the 132 tested cells have Holm p < 0.05 (most at the floor, about 0.03). By contrast: L 45 of 48, Lc 39 of 48, Vm − C1m 4 of 12, Vp − C1p 4 of 12, Cr − C1m 4 of 12. Each cell's value is in the JSON (`results[...]["p_holm"]`).

### Added post-hoc analysis: pT mismatch in the V edit

This analysis was not in the plan. It was computed in under 2 s from `h44_perjet.npz` and is not in the JSON. The exchanges differ in pT mismatch: on bb jets the Vm exchanges are mismatched by 0.71 and the C1m exchanges by 0.34. The random exchange Cr (mismatch 1.35) lowers s_bc by 0.50 more than C1m. To separate the mass effect from the mismatch effect, the per-jet V − C1 contrast was regressed on (mismatch_V − mismatch_C1). The intercept is the contrast at equal mismatch (95 % bootstrap CI, 1,000 resamples).

| class / edit | ParT_full | PN_full | slope (ParT) |
|---|---|---|---|
| bb, Vm | +0.17 [+0.04, +0.30] | +0.55 [+0.39, +0.73] | −0.61 per unit mismatch |
| bb, Vp | −0.47 [−0.60, −0.36] | −0.05 [−0.19, +0.10] | — |
| cc, Vm | −0.37 [−0.55, −0.19] | −0.14 [−0.26, −0.02] | — |
| cc, Vp | −0.15 [−0.31, +0.02] | −0.14 [−0.25, −0.03] | — |
| QCD | CIs include 0 | CIs include 0 | — |

## Reading (scoped to these edits, this sample and these two networks)

1. **L(λ)**, where count and size change together.
   - On bb jets the network response to λ < 1 is close to the counting reference: ParT/LR_count = 1.11 at λ = 0.4 and 0.78 at λ = 0.7.
   - For λ > 1 the networks move bb jets much less than LR_count does (ParT +0.08 against +0.44 at λ = 1.4; −0.04 against +0.57 at λ = 2.0). On cc jets they move more (ParT +1.07 against +0.55 at λ = 2.0).
   - The H0 statement "L response = LR_count prediction" holds only for λ < 1 on bb jets.
2. **Lc(λ)**, size at fixed counts.
   - On H→bb jets both networks respond outside the permuted band at λ = 0.4, 0.7 and 1.4 (PN also at λ = 2.0). The response is ordered in λ: ParT −0.23, −0.07, +0.03, +0.03; PN −0.32, −0.12, +0.08, +0.13.
   - The Lc response is 15 % (ParT) and 18 % (PN) of the L response at λ = 0.4. This exceeds the 10 % bound of H0 and lies outside the permuted band, so the count-only hypothesis H0 is excluded for bb vs cc on H→bb jets.
   - On H→cc jets, lengthening displacements at fixed counts (Lc(2.0)) moves them toward bb by +0.15 / +0.16, outside the band [−0.007, +0.014].
   - The direction (shorter displacement → more cc-like, longer → more bb-like) is the one H1a predicts from the B/D lifetime difference. The size, −0.23 / −0.32 at λ = 0.4, is about half of the planner's order −0.5.
   - The LR_life and GBDT references, fitted on the median, maximum and sum of ln S, predict the opposite sign (+0.17 / +0.18 at λ = 0.4). So the networks' size response is not the response of a fit to those summary statistics of S. The networks read the displacement through tanh(d0val) and tanh(dzval) per track, and the quantity they use was not identified here.
   - On QCD jets the Lc response is ≤ 0.03.
   - The bb-vs-QCD score moves by less than 0.04 under Lc. This fits D2's finding for bb vs QCD.
3. **Vm / Vp**, the m_SV proxy at fixed counts and fixed S multiset.
   - H1b predicts that lowering m_SV moves bb jets toward cc. On bb jets: ParT Vm − C1m is −0.05 [−0.17, +0.07] (not resolved); PN is +0.29 [+0.14, +0.44], toward bb, the opposite sign.
   - Raising m_SV (Vp − C1p) moves ParT toward cc by −0.52 [−0.63, −0.39]. PN is not resolved.
   - After the post-hoc mismatch adjustment, both networks on bb jets follow the sign of the data-fitted LR_vm: higher m_SV at fixed counts is more cc-like. ParT does so in the Vp direction (−0.47) and PN in the Vm direction (+0.55). Each is larger than LR_vm's ±0.11–0.13.
   - On cc jets the sign is reversed: lowering m_SV moves them toward cc (ParT −0.31 raw, −0.37 adjusted).
   - So the measured V responses do not follow one monotone function of m_SV across bb and cc jets. H1b in its physical direction (lower mass → cc) is excluded for PN on bb jets and is consistent with ParT on cc jets.
   - The control exchange alone (C1, at fixed m_SV, counts and S multiset) moves s_bc by −1.06 (ParT) and −1.67 (PN) on bb jets, and by +1.31 / +0.68 on cc jets. This is larger than any mass contrast. Which tracks carry the displacement (their pT and position inside the subjet) has a larger effect on the score than m_SV. The m_SV contrast is measured on top of that effect.
4. **Controls.** The no-IP models give exactly 0 for every edit. The permuted-network means stay within about ±0.06 for Lc and L, and within −0.13 to +0.15 for the V-type cells.

**Which hypotheses are excluded.**
- H0 (count only) is excluded for bb vs cc on H→bb jets (Lc) and on H→cc jets (Lc(1.4), Lc(2.0)).
- H1a is consistent in sign and ordering, with a size about half of the planner's expectation. The LR_life and GBDT summary-feature predictions are excluded as a description of the networks, because their sign is opposite.
- H1b (physical direction) is excluded for PN on bb jets and unresolved for ParT on bb jets. The V responses fit no single m_SV direction across classes.

**Open points.**
- There is one sample per class (file _100).
- The Lc map is one particular in-bin map. A second map, such as uniform redistribution within bins, would test whether the result depends on the map.
- The V edit's pT-mismatch confound is handled only post hoc. A V/C1 pair matched in pT mismatch would test the mass effect directly.
- m_SV is a two-subjet proxy, not a fitted secondary vertex.
- The bb jets sit at a 97 % bb call rate, which limits the room for responses toward bb (L(λ > 1)).

## Figures

- `results/physics/h44_jet_edit.png`: one real H→bb jet (pT 618 GeV) in the (Δη, Δφ) plane. Each displaced track's marker area grows with log S and its colour gives log10 S.
  - Unedited: n(S>2) = 7, median S 8.9, max S 879, m_SV 4.95 GeV.
  - Lc(0.4): n(S>2) still 7, median S 7.7, max S 358, m_SV unchanged.
  - Vm: n(S>2) still 7, S multiset unchanged, the IP tuples moved to other tracks, m_SV 3.91 GeV.
- `results/physics/h44_response.png`: Δ s_bc on H→bb jets for every edit. Bars show ParT_full, PN_full and the three logistic references; dotted lines show the permuted band.

## Explainer entry (for site/explain.py, H44_EXPLAINER; to be merged by the ledger)

```python
H44_EXPLAINER = [
 ('Question', 'Do the JetClass taggers separate $H\\to b\\bar b$ from $H\\to c\\bar c$ using how far the displaced tracks are from the collision point (lifetime) or the mass of the displaced-track system (vertex mass), beyond the number of displaced tracks? Earlier (D2) the answer for $b\\bar b$ versus QCD was "multiplicity, not lifetime"; $b$ versus $c$ is where lifetime and vertex mass differ.', None),
 ('Physics background', 'B hadrons fly $c\\tau\\approx455$–$490\\,\\mu$m before decaying, D hadrons $123$–$312\\,\\mu$m, and the $b$ vertex mass reaches about 5 GeV while the $c$ vertex mass stays below about 2 GeV. A track\'s transverse impact-parameter significance is $S=|d_0|/\\sigma_{d_0}$ (distance of closest approach to the beam line in the transverse plane over its uncertainty); a displaced track has $S>2$. The vertex-mass proxy is $m_{SV}=\\max_k\\big[(\\sum_{i\\in D_k}p_i)^2\\big]^{1/2}$, the invariant mass of the displaced tracks $D_k$ in each of the two exclusive-$k_T$ subjets.', None),
 ('Method', 'JetClass test jets, 2,000 each of $H\\to b\\bar b$, $H\\to c\\bar c$ and QCD (random rows, seeds 4401–4403). Four-momenta are never changed; only the impact-parameter inputs $(d_0,\\sigma_{d_0},d_z,\\sigma_{d_z})$ of charged tracks. L($\\lambda$): $d_0,d_z\\to\\lambda d_0,\\lambda d_z$ for displaced tracks ($S\\to\\lambda S$; counts change). Lc($\\lambda$): within each bin $(lo,hi]$ of $S$ (edges 0, 1, 2, 3, 5, 10, $\\infty$), $u=\\ln(S/lo)/\\ln(hi/lo)$, $S\'=lo\\,(hi/lo)^{u^{1/\\lambda}}$, and $S\'=10+\\lambda(S-10)$ above 10, so every count $n(S>1,2,3,5,10)$ is unchanged and only the size inside bins moves. V: inside the subjet with the larger $m_{SV}$, one or two displaced tracks exchange impact parameters with prompt tracks so that $m_{SV}$ falls by $\\ge1$ GeV (Vm) or rises by 1–4 GeV (Vp), at fixed counts and fixed multiset of $S$; control C1 makes the same kind of exchange with $|\\Delta m_{SV}|\\le5\\,\\%$. Score $s_{bc}=\\log(p_{bb}/p_{cc})$. Nulls: six weight-permuted ParT copies; networks without impact-parameter inputs (exactly 0). References: logistic fits on counts (LR_count), counts plus median/max $\\ln S$ (LR_life), counts plus $m_{SV}$ (LR_vm).', 'results/physics/h44_jet_edit.png'),
 ('Result', 'On $H\\to b\\bar b$ jets, Lc(0.4) lowers $s_{bc}$ by $-0.23\\,[-0.25,-0.20]$ (ParT) and $-0.32\\,[-0.37,-0.28]$ (ParticleNet), 15–18 % of the full scaling L(0.4) ($-1.50$, $-1.81$), ordered in $\\lambda$; permuted networks $[-0.04,+0.03]$. On $H\\to c\\bar c$ jets Lc(2.0) raises it by $+0.15$/$+0.16$. LR_life predicts the opposite sign ($+0.17$). The mass edit gives mixed signs: on $b\\bar b$ jets ParT Vm $-$ C1 $=-0.05$ (CI includes 0) and Vp $-$ C1 $=-0.52$; ParticleNet $+0.29$ and $-0.10$; the control exchange alone moves $s_{bc}$ by about $-1.1$ (ParT).', 'results/physics/h44_response.png'),
 ('Checks', 'Unedited 10-class accuracy on the sample 84.9 % (ParT_full), 82.6 % (PN_full); bb-vs-cc AUC 0.998 / 0.996. Counts above every threshold and the multiset of $S$ asserted unchanged for Lc and V. Kin/kinpid networks: max $|\\Delta\\text{logit}|=0$. Holm over 132 cells. The V exchanges have a larger pT mismatch than the control (0.71 vs 0.34 on bb jets); a post-hoc regression on the mismatch gives ParT $+0.17$ (Vm) and $-0.47$ (Vp) at equal mismatch.', None),
 ('Physical significance', 'For $b$ versus $c$ the taggers use the size of the track displacement, not only the number of displaced tracks, in the direction of the longer B-hadron lifetime; for $b$ versus QCD the same edit moves the score by less than 0.04, as D2 found. Whether they use the displaced-system mass is not resolved: the responses change sign between $b\\bar b$ and $c\\bar c$ jets and are smaller than the response to moving displacement between tracks at fixed mass.', None),
]
```

Entry line for the `'d'` list: `('Result explained: b versus c — displacement size and displaced-track mass at fixed displaced-track counts (claims D2, D14; H44)', H44_EXPLAINER, False)`.

Glossary additions (explain.GLOSSARY):
- `('impact-parameter significance S', 'S = |d0| / σ(d0): a track\'s distance of closest approach to the beam line in the transverse plane divided by its uncertainty; tracks from long-lived b and c hadrons have large S')`
- `('displaced track', 'a charged track with S > 2; prompt track: S ≤ 2')`
- `('Lc(λ) (count-preserving lifetime scaling)', 'an edit that shrinks (λ < 1) or stretches (λ > 1) each displaced track\'s impact parameter inside its significance bin (0, 1, 2, 3, 5, 10, ∞), so the number of tracks above every threshold stays the same')`
- `('m_SV (displaced-system mass)', 'the invariant mass of the displaced tracks in a subjet, maximum over the two subjets; a proxy for the secondary-vertex mass, which is up to about 5 GeV for b and below about 2 GeV for c hadrons')`
- `('s_bc', 'log(p_bb / p_cc), the network\'s log-odds of H→bb against H→cc from its 10-class output')`
