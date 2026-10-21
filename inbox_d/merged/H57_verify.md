# H57 verify: particle count vs momentum sharing (pT^D) in quark/gluon taggers

Script: interp/h57v_verify.py (steps gen, score, stats, deccheck).
Jobs:
- d-h57-verify: CPU 8 / 3 Gi, env-gen, 462 s.
- d-h57-verify-sc: GPU, scoring.
- d-h57-verify-st: CPU 1 / 2 Gi, peak 1.7 GiB.
- d-h57-verify-dc: GPU, validation of the hand decay.

Results:
- results/physics/h57_verify.json
- h57_verify_gen.json
- h57_verify_pamdec.json
- h57_verify_deccheck.json
- on the volume only: h57_verify_scores.npz and data/qg/gen/h57v/

## Conclusion

On new events, the H57 shifts repeat to within about 0.03 log-odds. The total particle count N is ruled out as what ParT and ParticleNet follow. Measured against an IRC-safe tagger, they take up only about 0.1–0.3 of a count classifier's extra shift. The test does not show that the networks read "momentum sharing rather than how many particles". A count of only the harder particles (pT above 1–4 GeV) predicts the network shifts as well as pT^D does, or better. Across the 38 settings, the shifts of these two observables are almost identical (correlation 0.97–0.99). The data therefore support "fragmentation hardness, not the total or angle-resolved particle count". They do not decide between pT^D and a count of hard hadrons.

Terms:
- pT^D = √(Σ pT,i²)/Σ pT,i.
- "IRC-safe": insensitive to soft or collinear emissions.
- Log-odds = log p(quark) − log p(gluon).

## What was checked

**Code read** (interp/h57_qg_count_vs_ptd.py, interp/h57_cpu_job.sh):
- Coordinates: (pT, y, φ, pid) are used as KMT stores them, with massless kinematics and η = y, as in training.
- Padded slots are excluded (real = pT > 0). The network input truncation at 128 particles does not bind, because N falls in NA− and PA−.
- Event pairing: every pair is keyed by worker × 100,000 + event, and pairs are matched within each class.
- Working point: the median V0 quark score of each model (50 % quark efficiency).
- κ and Holm: as stated.
- Training rows: the LR training jets are a random choice from the KMT train set, not the first k rows.
- Observables come from the same particle lists that the networks scored.
- h57_cpu_job.sh differs from job.sh only in the excluded-node list.
- No bug changes a reported number.

**Points in the report that are misleading or need care:**
1. **Call-change fractions** (e.g. "PA− quark 20.6 %") mostly measure per-jet hadronisation noise, not the edit. In the paired design, every setting re-hadronises each event, so the per-jet scores scatter. The IR+ control (N and pT^D fixed) also changes 21.3 % of ParT_kin quark calls, and Hz+ changes 20.8 %. The edit is in the asymmetry between the two flip directions. For PA− quark this is (1079 − 866)/n ≈ 2.3 % of jets net toward gluon (McNemar p 1.5e-6). Report the net flip, not the flip fraction.
2. **κ_pTD under NA−** (−0.38 to −0.74) has a denominator near zero, because LR(pT^D) − EFN ≈ −0.09. It carries no information.
3. **Figure h57_jet.png:** the x-axis label (Δy) is cut off at the bottom. h57_shifts.png and h57_settings.png match h57_run.json.
4. **Wording:** "barely moves" is a verdict word. Under NA− in quark jets, ParT_kin − GBDT(EFP) = +0.081, twice the +0.043 measured against the EFN.

## Procedure

**Fresh regeneration (d-h57-verify):**
- The same Pythia 8.317 setup and the same settings, read from h57_pilot.json: V0, NA− (R2:NAm#3), PA− (R2:PAm#2) and IR+ (R2:IRp#3).
- New worker ids 5761–5764, not used before. 6,000 quark + 6,000 gluon jets per setting.
- The parton level was identical to V0 in 400/400 hashed events for every setting.

**Scoring (d-h57-verify-sc):**
- Networks: ParT_kin, ParticleNet_kin, PFN (H2), EFN (H2), Herwig-trained ParT_kin, and one weight-permuted ParT_kin.
- Unedited V0 accuracy on the 12,000 fresh jets: ParT_kin 0.8296, PN_kin 0.8291, PFN 0.8139, EFN 0.8066, Herwig ParT_kin 0.7957, permuted 0.4983. LR(N) 0.7611, LR(pT^D) 0.7371.

**π⁰ split of PA− (new control).** PA− combines two changes: softer string fragmentation and stable π⁰. A stable π⁰ is a merged collinear photon pair.
- Every stable π⁰ in the fresh PA− jets (155,528 of them, 13.0 per jet) was decayed by hand into two photons, giving the sample PAmdec. The decay is isotropic in the π⁰ rest frame and then boosted, so four-momentum is conserved to 1e-13 GeV.
- V0 → PAmdec is the string-knob part (N +24 / +25 %, pT^D −8.2 / −9.1 %, q / g).
- PAmdec → PAm is the π⁰-merge part, measured on exactly the same jets (N −24 %, pT^D +5.4 / +4.9 %).

**Validation of the hand decay (d-h57-verify-dc).**
- The pilot D:pi0 jets (π⁰ stable) were decayed by hand and compared with the pilot nominal jets, which are the same events with π⁰ decayed by Pythia.
- If the hand decay is equivalent, every model shift is about 0. Paired mean shift, q / g:

| model | shift (q / g) | consistent with 0? |
|---|---|---|
| ParT_kin | −0.001 [−0.015, 0.014] / +0.003 [−0.013, 0.019] | yes |
| PFN | +0.010 / −0.036 | approximately |
| EFN | −0.035 / −0.039 | approximately |
| N | +1.4 / +1.5 % | (photons outside R = 0.4 are kept) |
| ParticleNet_kin | +0.059 / **+0.294** | no |
| Herwig-trained ParT_kin | 0.000 / +0.036 | approximately |

- ParticleNet_kin reacts strongly to the hand decay itself, so its decomposition numbers are not used. Its NA− and PA− numbers come from Pythia-clustered jets and are not affected.

**Competing "count" observables (d-h57-verify-st).** Histogram likelihood ratios were trained on 30k + 30k KMT train jets (the same jets as H57's LR(pT^D)) for:
- N(pT > 1, 2, 4 GeV): the number of constituents above a pT threshold;
- N_res(δ): the number of single-linkage clusters of constituents at ΔR < δ, for δ = 0.01 and 0.03. This is a count at finite angular resolution.

Each was scored on the fresh settings and on all 38 H57 fit settings, reusing H57's network scores from h57_scores.npz. Alignment was asserted per sample: the keys match and the recomputed N equals the stored N.

The fit is Δs_net = γ Δs_EFN + β Δs_LR(X), weighted by 1/SE², over the 38 settings (ndf 36). The χ² values are compared across the choices of X.

**Statistics:**
- 95 % CIs: 1,000 bootstrap resamples over jet pairs.
- p-values for net − reference come from z = difference / bootstrap SE.
- Holm correction over 270 tests: 3 networks × 5 comparisons × 2 classes × 9 references. 232 tests pass at 0.05.

## Numbers

### Replication on fresh events (Δs, quark ; gluon; H57 values in parentheses)

| setting | ParT_kin | PN_kin | PFN | EFN | LR(N) | LR(pT^D) |
|---|---|---|---|---|---|---|
| NA− (N −7.4 / −6.3 %, pT^D 0.0 / −0.8 %) | +0.143 [0.114, 0.172] ; +0.053 [0.021, 0.089] (+0.143 ; +0.057) | +0.166 ; +0.052 | +0.170 ; +0.066 | +0.110 ; +0.028 | +0.286 ; +0.258 | +0.022 ; −0.023 |
| PA− (N −5.8 / −4.7 %, pT^D −3.3 / −4.7 %) | −0.078 [−0.108, −0.048] ; −0.201 [−0.235, −0.169] (−0.097 ; −0.228) | −0.030 ; −0.211 | +0.028 ; −0.140 | +0.015 ; −0.063 | +0.217 ; +0.185 | −0.121 ; −0.194 |
| IR+ (N −0.9 / +0.1 %, pT^D +0.3 %) | +0.190 ; +0.182 (+0.172 ; +0.196) | +0.190 ; +0.189 | +0.164 ; +0.129 | +0.262 ; +0.271 | +0.040 ; +0.001 | +0.008 ; +0.017 |

The permuted ParT_kin moves by at most 0.039 in absolute value. κ_N on fresh events:
- NA−: ParT_kin 0.19 [0.08, 0.29] ; 0.11 [0.02, 0.20]; PN_kin 0.32 ; 0.10; PFN 0.34 ; 0.17.
- PA−: ParT_kin κ_pTD 0.68 [0.53, 0.90] ; 1.05 [0.84, 1.38].
- Net − LR(N), ParT_kin: NA− −0.143 ; −0.204 and PA− −0.295 ; −0.386, all Holm < 1e-20.
- H0_N ("follows the total count") is excluded again.

### Competing count observables (fresh PA−, ParT_kin − reference, q ; g)

| reference | its Δs (q ; g) | ParT_kin − reference |
|---|---|---|
| LR(pT^D) | −0.121 ; −0.194 | +0.043 (p 0.007) ; −0.007 (p 0.68) |
| LR(N, pT > 2 GeV) | −0.078 ; −0.138 | **+0.000 (p 0.98)** ; −0.063 (p 4e-5) |
| LR(N, pT > 4 GeV) | −0.109 ; −0.170 | +0.031 (p 0.06) ; −0.031 (p 0.06) |
| LR(N_res, 0.01) | +0.172 ; +0.118 | −0.250 ; −0.318 (p < 1e-60) |
| LR(N) | +0.217 ; +0.185 | −0.295 ; −0.386 |

Under PA−, the total N falls by 5–6 %, but the number of particles above 2 GeV rises by 2.7 / 5.1 %. A count of hard particles therefore predicts the same "toward gluon" move as pT^D.

### Fit over the 38 H57 settings: χ² for Δs_net = γ Δs_EFN + β Δs_LR(X) (ndf 36), quark / gluon

| X | ParT_kin | ParticleNet_kin | PFN |
|---|---|---|---|
| N (total, NB) | 338 / 644 | 176 / 649 | 139 / 623 |
| N_res 0.01 / 0.03 | 350, 368 / 680, 741 | 188, 207 / 682, 743 | 148, 172 / 699, 820 |
| pT^D | **81** / 107 | 159 / 128 | 422 / 908 |
| N(pT > 1 GeV) | 133 / 152 | **42** / 149 | **53** / **56** |
| N(pT > 2 GeV) | 114 / **49** | 58 / **50** | 125 / 143 |
| N(pT > 4 GeV) | 110 / 64 | 98 / 78 | 238 / 716 |

- With the total N added as a third regressor, all pT^D-like choices reach χ² 22–109. The coefficient of the total N stays between −0.27 and +0.14 for ParT_kin and ParticleNet_kin.
- Across settings, the correlation of Δs_LR(pT^D) with Δs_LR(N, pT > 2 or 4 GeV) is 0.97–0.99. With the total N it is 0.79–0.86.
- For PFN, the "count component" that H57 reports (α 0.24–0.58 on the total N) is described better by a count above 1 GeV: χ² 53 / 56, against 46 / 21 for H57's three-term N + pT^D form.

### π⁰ split of PA− (ParT_kin only; Δs, q ; g)

| part | ParT_kin | EFN | LR(N) | LR(pT^D) | κ_N | κ_pTD |
|---|---|---|---|---|---|---|
| string knobs (V0 → PAmdec) | −0.393 ; −0.512 | −0.150 ; −0.229 | −0.864 ; −0.922 | −0.356 ; −0.386 | 0.34 ; 0.41 | 1.18 ; 1.80 |
| π⁰ merge (PAmdec → PAm, same jets) | +0.316 ; +0.311 | +0.165 ; +0.164 | +1.080 ; +1.107 | +0.237 ; +0.191 | 0.17 ; 0.16 | 2.1 ; 5.4 |

- The PA− result is the small difference of two shifts of opposite sign, each about 0.3–0.5 log-odds.
- In each part, ParT follows 0.16–0.41 of the count classifier's extra shift.
- In the π⁰-merge part, ParT moves 2–5 times more than LR(pT^D) predicts beyond the EFN. Merging collinear photon pairs moves ParT by more than its pT^D change accounts for. This agrees with H5/H14, where collinear structure among hard particles is read directly.

## Competing hypotheses

| hypothesis | prediction | status |
|---|---|---|
| H0_N: follows the total N | κ_N ≈ 1 under NA−; toward quark under PA− | excluded (κ_N 0.1–0.3; PA− toward gluon; Holm < 1e-20) |
| H_res: follows the count at angular resolution 0.01–0.03 | like H0_N | excluded (χ² 350–740) |
| H_pTD: follows pT^D | PA− ≈ LR(pT^D); good fit | not excluded |
| H_hard: follows the number of hadrons above about 1–4 GeV | PA− ≈ LR(N, pT > 2–4 GeV); good fit | not excluded; fits ParT gluon and ParticleNet better than pT^D |
| pure IRC-safe | Δs = γ EFN | excluded (H57: GBDT(EFP) and the EFN fall short under PA− and NA−) |

## Corrected reading

ParT_kin and ParticleNet_kin (Pythia-trained, kinematic inputs) do not follow the total constituent count or the count at angular resolution 0.01–0.03:
- κ_N is 0.1–0.3 when N falls at fixed pT^D.
- They move toward gluon in PA−, where the total count classifier moves toward quark.
- Both results replicate on fresh events.

Their response beyond the IRC-safe energy flow follows the hardness of fragmentation. Two observables measure it equally well in these settings: pT^D, and the number of hadrons above about 1–4 GeV. The two are 0.97–0.99 correlated across the settings, and neither is preferred consistently:
- pT^D fits ParT quark jets best.
- N(pT > 1–2 GeV) fits ParT gluon jets and ParticleNet best.

Further points:
- PFN's extra count component is better described as a count of particles above about 1 GeV.
- The PA− result is a near-cancellation between a string-fragmentation part and a π⁰-merge part. In the π⁰-merge part, ParT moves more than pT^D predicts.
- The statement "how hard the hadrons are, not how many there are" should be narrowed to "not the total number of hadrons".

## Issues

- Fresh sample: 6,000 + 6,000 jets per setting. The CIs are about 1.3× wider than H57's.
- The hand π⁰ decay keeps photons that fall outside R = 0.4 (N +1.5 %). ParT_kin is validated against it; ParticleNet_kin is not. ParticleNet_kin reacts to the hand-decayed jets by +0.29 in gluon jets, for a reason not identified, so its decomposition is not reported.
- The candidate observables were chosen after seeing the H57 result, and only one family of thresholds was tried. The χ² ranking is descriptive: the fit residuals exceed the statistical errors (χ²/ndf > 1 for most choices).
- All other H57 limits remain: hand-picked settings, PA− far from the nominal jets (C2ST 0.84–0.90), no NA+ or PA+ setting, and one training seed per network.
