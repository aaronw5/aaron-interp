# H57 — Particle count N vs momentum sharing pT^D in quark/gluon taggers, separated with Pythia 8 hadronisation and decay settings (claims D22, D7)

## Conclusion

We regenerated quark and gluon jets with Pythia 8 from the same parton-level events. We changed only the hadronisation and decay settings, and chose the settings so that the particle count N and the momentum sharing among the hadrons (pT^D) no longer move together as they did in H54. In one setting, N falls by 7 % while pT^D is unchanged. The kinematic-input networks (ParT, ParticleNet, PFN) then move little beyond the IRC-safe EFN: ParT_kin moves +0.04 log-odds more than the EFN in quark jets and +0.01 in gluon jets, while a count-only classifier moves +0.18 to +0.23 more. In another setting, N falls (−5 %) and pT^D also falls (−3 to −5 %). The kinematic networks then move toward "gluon", following the pT^D classifier and opposite to the count classifier (ParT_kin gluon jets −0.23 vs LR(pT^D) −0.20 vs LR(N) +0.18). A fit over 38 settings gives ParT_kin a pT^D weight of 0.65–0.95 and a count weight of 0.04–0.10. So the hadron-level information that H54 found is mostly how the jet momentum is shared among its hadrons, not how many hadrons there are. Some points remain uncertain:
- PFN keeps a larger count part (count weight 0.24–0.58).
- The networks that also read particle identity are confounded, because the settings change the mix of hadron species.
- The decisive setting makes π⁰ mesons stable. A classifier separates those jets easily from nominal ones (C2ST AUC 0.84–0.90).
- No setting raised N at fixed pT^D, and Herwig was not available.

## Question and hypotheses

H54 (verified 0.8) found that Pythia-trained QG taggers respond to a hadronisation change that moves N by ±11 % at fixed IRC-safe energy flow. Under that change pT^D moved by ∓4.5 %. In every H54/H54v setting, N and pT^D moved together, so H54 could not say which of the two the networks read.

- **pT^D** = √(Σ pT,i²) / Σ pT,i over the jet constituents. This is the h5/H54/CMS definition. It is large when a few hadrons carry most of the jet momentum and small when the momentum is spread over many soft hadrons. The task text writes the squared form Σ pT,i² / (Σ pT,i)², which changes by twice the relative amount.
- **H0_N** (the networks read the count): under a setting that changes N at fixed pT^D, the networks move like LR(N), with κ_N near the H54 value of 0.5–0.9. Under a setting that changes pT^D, they follow the N change, not the pT^D change.
- **H0_pTD** (the networks read momentum sharing): the opposite pattern. At fixed pT^D they move like the EFN (κ_N ≈ 0), and they follow LR(pT^D) when pT^D moves.
- **Mixed**: both parts are nonzero. We fit Δs_net = γ·Δs_EFN + α·Δs_LR(N) + β·Δs_LR(pT^D) over all settings.

## Procedure

**Generator** (H54 setup; module interp/h54_qg_generator_knobs.py reused):
- Pythia 8.317 (pythia8mc wheel, env-gen), Monash 2013 tune, pp at 14 TeV.
- Processes: q q̄ → Z(νν)+g (gluon jets) and q g → Z(νν)+q with q ∈ {u, d, s} (quark jets). MPI on.
- Jets: anti-kT R = 0.4 on visible final-state particles, hardest jet only, pT 500–550 GeV, |y| < 1.7.
- Per-event reseeding: event i of worker w starts from seed(w, i) in every setting. The parton-level record (hard process, shower, MPI) was identical to V0 in 400/400 hashed events for every final setting, and in 99–100 % of the pilot settings. Each setting − V0 difference is therefore a paired, same-event change of hadronisation and decays only.
- Worker ids: 5701–5704 (pilot), 5711–5714 (final), 5721–5724 (second nominal V0b).

**Knobs** (Monash values read back from Pythia at run time):
- String fragmentation: `StringZ:aLund` 0.68, `StringZ:bLund` 0.98 (the Lund z-distribution f(z) ∝ z⁻¹(1−z)^a exp(−b m_T²/z), which sets the light-cone momentum fraction each new hadron takes), and `StringPT:sigma` 0.335 GeV (the transverse-momentum width at each string break).
- Flavour: `StringFlav:probStoUD` 0.217 (s-quark suppression), `probQQtoQ` 0.081 (diquark/baryon rate), `mesonUDvector` 0.5 (vector-to-pseudoscalar ratio for light mesons).
- Decay switches: `<pid>:mayDecay = off` for ρ (113, 213), ω (223), K* (313, 323), η/η′ (221, 331), K_S plus hyperons (310, 3122, 3112, 3222, 3312, 3322, 3334), all light vectors, or π⁰ (111). A particle whose decay is switched off stays in the final state as one particle. Its decay products (2–3 particles sharing its momentum) are no longer produced.

**Pilot** (job d-h57-run for round 1, d-h57-run2 for round 2; 2,000 q + 2,000 g jets per setting; paired with the pilot nominal run):
- Round 1: 19 single-knob or single-decay settings plus a second nominal run.
- A linear response model (dN/N, dpT^D/pT^D, ΔEFN per class) was fitted to round 1. The settings were then solved with box-constrained least squares for these targets: NA± (N ±9 %, pT^D and EFN fixed), PA± (pT^D ±4.4 %, N and EFN fixed) and IR+ (EFN +0.25, N and pT^D fixed).
- Round 2: 15 solved settings (3 per target).
- Measured on every setting: dN/N, dpT^D/pT^D, the KMT-trained EFN shift, and girth, m/pT and C1^(1) in units of their SD.

**Final** (jobs d-h57-gen-a, -b2, -c2; 10,000 q + 10,000 g jets per setting, fresh seeds). Six settings:

| setting | Pythia settings (changed from Monash) | ΔN/N q / g | ΔpT^D/pT^D q / g | Δgirth q / g |
|---|---|---|---|---|
| V0 | nominal | – | – | – |
| V0b | nominal, different seeds (generator-noise control) | −0.5 / −0.5 % | +0.7 / +0.1 % | −2.0 / −1.0 % (unpaired) |
| **NA−** (count down, pT^D unchanged) | aLund 1.214, σ 0.261, probStoUD 0.198, probQQtoQ 0.032, mesonUDvector 0.15; η, η′, ω, K_S, Λ, Σ, Ξ, Ω stable | −7.2 / −6.7 % | +0.1 / −0.65 % | −0.8 / −0.7 % |
| **PA−** (pT^D down while N also falls) | aLund 0.826, σ 0.200, probStoUD 0.117, probQQtoQ 0.020, mesonUDvector 0.90; π⁰ stable | −5.3 / −4.5 % | −3.0 / −4.7 % | −0.0 / +0.1 % |
| IR+ (IRC-safe content only; control) | aLund 0.630, σ 0.242, probStoUD 0.190, probQQtoQ 0.112, mesonUDvector 0.61; K_S + hyperons stable | −0.1 / −0.1 % | +0.4 / +0.4 % | −1.4 / −1.2 % |
| Hz+ (H54's setting, N and pT^D together) | aLund 0.79, bLund 0.88, σ 0.247 | +11.2 / +10.8 % | −4.5 / −4.5 % | −0.4 / −0.5 % |

**Scoring** (d-h57-score, GPU, interp/h54_gpu_job.sh) and **references** (d-h57-refs, CPU, venv-ef):
- Networks: the H54 set. ParT_kin, ParT_kinpid, ParticleNet_kin, ParticleNet_kinpid, LorentzNet, PFN (H2) and PFN 2nd run (H5 torch), EFN (H2) and EFN 2nd run, all trained on KMT Pythia jets. The Herwig-trained ParT_kin and ParticleNet_kin. Two weight-permuted ParT_kin.
- References trained on 30k + 30k KMT train jets:
  - LR(N): negative-binomial likelihood ratio of the multiplicity.
  - LR(pT^D): 40-bin histogram likelihood ratio.
  - LR(N, pT^D): gradient-boosted decision tree (GBDT) on both.
  - GBDT(EFP): 54 IRC-safe energy-flow polynomials, degree ≤ 5, β = 1.
  - GBDT(EFP+N) and GBDT(EFP+pT^D): the same EFPs plus one extra input.
- Stable resonances are outside the KMT particle vocabulary. For scoring they were relabelled by charge: π⁰ and η → photon (22), other neutral mesons → neutral hadron (130), charged mesons → ±211, baryons → ±2212 / ±2112. Relabelled shares: 12.2 % of particles in NA−, 31.5 % in PA−, 4.1 % in IR+, 0 % otherwise. No pid outside the vocabulary remained. Kinematic-only networks do not read the label.

**Stats** (d-h57-stats; figures d-h57-fig4):
- Δs = mean over event-paired jets of (s_setting − s_V0) per class. s is the quark log-odds log p(q) − log p(g).
- 95 % CIs come from 1,000 bootstrap resamples over jet pairs (V0b is unpaired: independent bootstrap of both samples).
- Network − reference differences use the same resample indices; p-values are two-sided from z = difference / bootstrap SE. Holm correction is applied over 288 tests: 9 networks × {NA−, PA−, IR+, Hz+} × 2 classes × {EFN, LR(N), LR(pT^D), GBDT(EFP)}.
- κ_N = (Δs_net − Δs_EFN) / (Δs_LR(N) − Δs_EFN). This is the share of the count classifier's extra shift that the network follows (0 = like the IRC-safe EFN, 1 = like the count classifier). κ_pTD is the same with LR(pT^D).
- Call changes use a threshold set per model at 50 % quark efficiency (the median score of V0 quark jets). We report the fraction of jet pairs whose call flips. McNemar's exact test (a binomial test on discordant pairs) checks whether q→g flips and g→q flips are equally frequent.
- Fit over 38 settings: 33 pilot settings paired with the pilot nominal, plus the 4 final non-nominal settings and Hz+. The fit is Δs_net = γ Δs_EFN + α Δs_LR(N) + β Δs_LR(pT^D), weighted by 1/SE², with no intercept. CIs come from 2,000 resamples of settings.
- C2ST (classifier two-sample test): a GBDT on 14 per-jet observables is trained to tell two samples apart; we report its held-out AUC (0.5 = indistinguishable).

## Unedited accuracy (V0, 10,000 + 10,000 jets; threshold s = 0)

| model | V0 accuracy (H57) | H54 V0 (30k + 30k) |
|---|---|---|
| ParT_kin | 0.8270 | 0.8305 |
| ParT_kinpid | 0.8391 | 0.8411 |
| ParticleNet_kin | 0.8285 | 0.8303 |
| ParticleNet_kinpid | 0.8347 | 0.8365 |
| LorentzNet | 0.8451 | 0.8466 |
| PFN | 0.8169 | 0.8193 |
| PFN 2nd run | 0.8024 | 0.806 |
| EFN | 0.8072 | 0.8112 |
| EFN 2nd run | 0.7918 | 0.795 |
| Herwig-trained ParT_kin | 0.7933 | 0.7967 |
| Herwig-trained ParticleNet_kin | 0.7902 | 0.7926 |
| permuted ParT_kin | 0.499 / 0.500 | – |
| LR(N) | 0.7645 | 0.766 |
| LR(pT^D) | 0.7358 | 0.740 |
| GBDT(EFP) | 0.7745 | 0.775 |
| GBDT(EFP+N) | 0.7827 | 0.785 |
| GBDT(EFP+pT^D) | 0.7797 | – |
| LR(N, pT^D) | 0.7703 | 0.773 |

Every model reproduces its H54 V0 accuracy within 0.004. C2ST V0 vs KMT test jets: AUC 0.493 [0.482, 0.504] (q), 0.507 [0.495, 0.517] (g).

## Numbers (Δs quark log-odds [95 % CI], quark ; gluon jets)

**NA− (N −7 %, pT^D unchanged, EFN +0.10 / +0.05)**

| model | Δs | − EFN | κ_N |
|---|---|---|---|
| ParT_kin | +0.143 [0.121, 0.168] ; +0.057 [0.032, 0.082] | +0.043 [0.028, 0.058] (Holm 1.6e-6) ; +0.011 [−0.005, 0.027] (Holm 1) | 0.24 [0.16, 0.33] ; 0.05 [−0.02, 0.11] |
| ParticleNet_kin | +0.151 ; +0.056 | +0.050 (6.5e-10) ; +0.010 (1) | 0.28 ; 0.04 |
| PFN | +0.157 ; +0.093 | +0.056 ; +0.046 | 0.32 ; 0.20 |
| PFN 2nd run | +0.173 ; +0.151 | +0.072 ; +0.105 | 0.41 ; 0.46 |
| Herwig ParT_kin / PN_kin | +0.102 / +0.109 ; +0.027 / +0.004 | +0.001 / +0.009 ; −0.020 / −0.042 | 0.01 / 0.05 ; −0.09 / −0.19 |
| ParT_kinpid / PN_kinpid / LorentzNet | −0.070 / −0.067 / −0.096 ; −0.217 / −0.229 / −0.075 | −0.17 to −0.28 | (species confound, see Issues) |
| permuted ParT_kin | −0.004 ; −0.003 | | |
| EFN / EFN 2nd run | +0.101 / +0.074 ; +0.046 / +0.035 | | |
| LR(N) | +0.277 ; +0.273 | | |
| LR(pT^D) | +0.023 ; −0.021 | | |
| GBDT(EFP) / GBDT(EFP+N) / GBDT(EFP+pT^D) | +0.062 / +0.229 / +0.061 ; +0.034 / +0.284 / +0.005 | | |

Network − LR(N), Holm: ParT_kin −0.133 (5e-32) ; −0.216 (2e-86). Calls changed at the 50 %-quark-efficiency threshold, ParT_kin: 21.5 % of quark-jet pairs (q→g 856, g→q 1,175, McNemar p 1.5e-12) and 2.9 % of gluon-jet pairs (123 vs 141, p 0.30). The EFN changes 18.1 % (p 2e-6) and 3.4 % (p 0.57); LR(N) changes 20.0 % (552 vs 1,339) and 6.1 % (207 vs 348, p 2e-9).

**PA− (N −5.3 / −4.5 %, pT^D −3.0 / −4.7 %, EFN +0.004 / −0.074)**

| model | Δs | − EFN | − LR(pT^D) | κ_pTD |
|---|---|---|---|---|
| ParT_kin | −0.097 [−0.122, −0.076] ; −0.228 [−0.253, −0.203] | −0.101 (Holm 9e-35) ; −0.155 (7e-74) | +0.030 (0.67) ; −0.032 (0.47) | 0.77 [0.63, 0.95] ; 1.26 [1.04, 1.53] |
| ParticleNet_kin | −0.047 ; −0.243 | −0.051 ; −0.170 | +0.080 ; −0.047 | 0.39 ; 1.38 |
| PFN | +0.001 ; −0.152 | −0.003 (1) ; −0.078 | +0.128 ; +0.045 | 0.03 ; 0.63 |
| PFN 2nd run | +0.073 ; −0.023 | +0.069 ; +0.051 | | −0.53 ; −0.41 |
| Herwig ParT_kin / PN_kin | −0.037 / −0.017 ; −0.090 / −0.085 | −0.040 / −0.021 ; −0.017 / −0.011 | | 0.31 / 0.16 ; 0.14 / 0.09 |
| ParT_kinpid / PN_kinpid / LorentzNet | −0.020 / −0.031 / +0.122 ; −0.057 / −0.132 / +0.170 | | | (confound) |
| permuted ParT_kin | −0.030 ; −0.033 | | | |
| EFN / EFN 2nd run | +0.004 / +0.016 ; −0.074 / −0.029 | | | |
| LR(N) | +0.206 ; +0.176 | | | |
| LR(pT^D) | −0.127 ; −0.197 | | | |
| GBDT(EFP) / GBDT(EFP+N) / GBDT(EFP+pT^D) | −0.005 / +0.147 / −0.075 ; −0.014 / +0.173 / −0.169 | | | |

Network − LR(N), Holm: ParT_kin −0.303 (2e-170) ; −0.405 (3e-293). Calls, ParT_kin: quark 20.6 % (q→g 1,079 vs g→q 866, p 1.5e-6); gluon 2.5 % (140 vs 82, p 1.2e-4).

**IR+ control (N and pT^D fixed within 0.5 %, girth −1.3 %, EFN +0.25 / +0.27).** ParT_kin +0.172 ; +0.196, ParticleNet_kin +0.165 ; +0.208, PFN +0.135 ; +0.137, Herwig-trained ParT_kin +0.174 ; +0.156, GBDT(EFP) +0.158 ; +0.141. LR(N) +0.007 ; +0.008, LR(pT^D) +0.008 ; +0.015, permuted +0.001. The kinematic networks follow an IRC-safe-only change with 0.6–0.8 of the EFN shift, with N and pT^D held.

**Hz+ (H54 setting, fresh jets).** ParT_kin −0.146 ; −0.217 (H54: −0.136 ; −0.215), EFN +0.014 ; −0.005, LR(N) −0.426 ; −0.422, LR(pT^D) −0.200 ; −0.191. κ_pTD 0.75 ; 1.14, κ_N 0.36 ; 0.51. The H54 numbers reproduce within 0.01.

**Generator noise (V0b vs V0, unpaired).** Every model |Δs| ≤ 0.027 with CIs containing 0. Network − EFN ≤ 0.034.

**Fit over 38 settings** (γ, α, β [95 % CI over settings]; χ²/ndf):

| model | quark jets | gluon jets |
|---|---|---|
| ParT_kin | γ 0.69, α 0.04 [−0.02, 0.11], β 0.65 [0.40, 0.78]; 72/35 | γ 0.69, α 0.10 [0.09, 0.14], β 0.95 [0.85, 1.00]; 22/35 |
| ParticleNet_kin | α 0.13 [0.08, 0.19], β 0.44 [0.24, 0.54]; 49/35 | α 0.11 [0.08, 0.14], β 0.95 [0.82, 1.03]; 30/35 |
| PFN | α 0.24 [0.21, 0.28], β 0.35 [0.22, 0.45] | α 0.32 [0.30, 0.34], β 0.89 [0.83, 0.94] |
| PFN 2nd run | α 0.41 [0.35, 0.45], β 0.19 [0.06, 0.39] | α 0.58 [0.55, 0.60], β 0.58 [0.52, 0.67] |
| Herwig ParT_kin | α 0.06 [0.05, 0.10], β 0.36 [0.28, 0.39]; 18/35 | α 0.01 [−0.02, 0.04], β 0.21 [0.08, 0.31] |
| Herwig PN_kin | α 0.07, β 0.27 | α −0.06, β 0.12 |
| permuted ParT_kin | α −0.05, β 0.12 | α −0.04, β 0.11 |
| ParT_kinpid / PN_kinpid / LorentzNet | χ² 550–1160/35 (fit does not describe them) | |

Without the EFN term (Δs = α LR(N) + β LR(pT^D), the form in the plan), ParT_kin gives α 0.06 / 0.12 and β 1.06 / 1.49, with χ² 828 / 748 for 36 degrees of freedom: the IR+ type settings need the IRC-safe term. Correlation of the LR(N) and LR(pT^D) shifts across the 38 settings: 0.86 (q), 0.79 (g). NA− and PA− are the settings that break this collinearity.

**Holm**: 247 of 288 net − reference tests pass at 0.05. The ones that do not pass are mainly:
- the Herwig-trained nets vs EFN;
- ParT_kin and ParticleNet_kin gluon NA− vs EFN and GBDT(EFP);
- the kinematic nets PA− vs LR(pT^D).

**C2ST vs V0** (AUC, q / g): NA− 0.607 / 0.627, PA− 0.838 / 0.899, IR+ 0.534 / 0.526, Hz+ 0.634 / 0.650, V0b 0.496 / 0.499. C2ST vs KMT is within 0.02 of each.

## Pilot findings (h57_pilot.json; figure h57_settings.png)

- Every string knob changes pT^D by −0.39 to −0.62 % per +1 % of N: aLund, bLund, σ, probQQtoQ, mesonUDvector (median −0.56). The decay switches give ρ −0.53, K* −0.56, ω −0.40, K_S/hyperons −0.36, η −0.28 and π⁰ −0.20. Only the π⁰ and η decays move N with relatively little pT^D change. This is the lever used for NA− (η/ω/K_S stable, compensated by more string fragmentation) and PA− (π⁰ stable, compensated by softer string fragmentation, σ 0.200).
- The linear solve did not predict combined settings well. For example, NA−#1 was predicted at N −10.2 %, pT^D −0.2 % and EFN 0.00, and measured at N −11.1 %, pT^D −0.3 % and EFN −0.15 / −0.10. No setting met all pre-set criteria (|ΔN/N| 8–10 %, |ΔpT^D| < 0.1 × co-movement, |ΔEFN| ≤ 0.03).
- NA+ and PA+ cannot be reached with these knobs. Raising N with a smaller pT^D drop than the string knobs give would require turning on more decays than nominal.
- The final NA− and PA− were chosen by hand from the round-2 measurements. This choice is recorded in h57_pilot.json (`picked_by_hand`). NA−#3 had the smallest EFN shift among the N-down, pT^D-flat settings. PA−#2 moves pT^D opposite to its usual co-movement with N, so LR(N) and LR(pT^D) predict opposite signs.
- Herwig: no Herwig executable, no python module and no Herwig install on /j-jepa-vol, so the Herwig arm was not run.

## Reading (scoped to these jets and settings)

- **Kinematic Pythia-trained networks (ParT_kin, ParticleNet_kin):**
  - At fixed pT^D (NA−), N −7 % moves them beyond the EFN by +0.04–0.05 log-odds in quark jets and +0.01 in gluon jets (κ_N 0.24–0.28 and 0.04–0.05). H0_N predicted κ_N of 0.5–0.9 from H54.
  - When pT^D falls while N also falls (PA−), they move toward gluon with LR(pT^D). They do not move toward quark as LR(N) would: net − LR(N) is −0.25 to −0.42, Holm p < 1e-100, and net − LR(pT^D) is within ±0.08.
  - H0_N (count) is excluded for these two networks in both classes.
  - H0_pTD with no count term is excluded only for quark jets: NA− net − EFN +0.04–0.05, Holm < 2e-6. The fit puts their pT^D weight at 0.44–0.95 and their count weight at 0.04–0.13.
- **PFN** keeps a larger count component (α 0.24–0.32; the 2nd PFN run 0.41–0.58). At fixed pT^D it moves 0.20–0.46 of the count increment.
- **Herwig-trained ParT_kin and ParticleNet_kin** move like the EFN under NA− (κ_N −0.19 to 0.05). Their pT^D weight is smaller (0.12–0.36).
- **Particle-identity networks** (ParT_kinpid, ParticleNet_kinpid, LorentzNet) move toward gluon under NA− (−0.07 to −0.23), opposite to both LR(N) and LR(pT^D). LorentzNet moves toward quark under PA−. NA− and PA− change the hadron-species mix (12 % and 31 % of particles are stable resonances relabelled as photons, neutral hadrons or pions), so these responses are not interpretable as count or pT^D responses.
- **IR+**: with N and pT^D fixed, a change of IRC-safe content moves the kinematic nets by 0.6–0.8 of the EFN shift. Shifts of the IRC-safe content are part of what these networks read (γ ≈ 0.7 in the fit).

## Issues / limits

- **Targets not met as specified.** NA− has |ΔN/N| 6.7–7.2 % (target 8–10 %), gluon ΔpT^D −0.65 % (criterion < 0.4 %) and ΔEFN +0.10 / +0.05 (criterion ±0.03). PA− has ΔN/N −5 % (criterion < 1 %). NA+ and PA+ were not produced. The κ and fit analyses use the EFN as a covariate to absorb the residual IRC-safe shifts. GBDT(EFP) moved +0.06 / +0.03 (NA−) and −0.005 / −0.014 (PA−).
- **PA− is far from nominal** (C2ST 0.84–0.90). It contains stable π⁰ (a merged photon pair stored as one "photon") and σ = 0.200 GeV. The networks never saw such jets in training. The PA− response could partly come from features specific to π⁰ merging, such as the leading photon fractions, rather than from pT^D as such. GBDT(EFP+pT^D) predicts −0.075 / −0.169 under PA−, close to ParT_kin's −0.097 / −0.228, which is consistent with but does not prove the pT^D reading.
- Manual pick of NA− and PA− after round 2. The rule-based choice reported no feasible setting. Round-2 pilot statistics are 2,000 jets per class.
- Species relabelling of stable resonances is my choice. It matters only for the pid networks, which are excluded from the count/pT^D reading.
- The fit's regressors are correlated (0.79–0.86). χ²/ndf for ParT_kin quark jets is 72/35, so the linear three-term model is approximate.
- One training seed per network (H56 seed replicates not used here). The permuted ParT_kin moves −0.03 under PA−. This is small next to the trained nets, but it is not zero.
- Run issues: the first pilot job was stopped and the solver rewritten from clip-after-solve to box-constrained least squares. One CPU node (hcc-nrp-shor-c6021.unl.edu) ran Pythia about 8× slower (112 vs 14 ms/event), so interp/h57_cpu_job.sh is a job.sh copy that excludes it. d-h57-gen-a returned exit 127 after both files were written: the `--pick_note` text contained `;` and the shell ran the remainder as a command. The data and the pick were saved before that point. None of these changes a number.

## Files / jobs

- Script interp/h57_qg_count_vs_ptd.py (steps pilot, gen, score, refs, stats, fig); interp/h57_cpu_job.sh.
- Jobs: d-h57-run (stopped), d-h57-run2 (pilot), d-h57-gen-a / -b2 / -c2 (8 CPU, 3 Gi, about 12 min each), d-h57-score (GPU, 30 min), d-h57-refs (8 CPU, 8 Gi, 19 min, peak 4.7 GiB), d-h57-stats, d-h57-fig4.
- Data: /j-jepa-vol/aaron-interp/data/qg/gen/h57/{pilot,final}.
- Results (results/physics/): h57_pilot.json, h57_gen_*.json, h57_gen_parton.json, h57_models.json, h57_valid.json, h57_run.json. On the volume only: h57_scores.npz, h57_refs.npz.
- Figures: h57_settings.png (pilot settings in the ΔN/N–ΔpT^D plane, coloured by EFN shift), h57_shifts.png (Δs per model and setting with 95 % CIs), h57_jet.png (one gluon event, x axis Δy, under V0, NA−, PA−).

## Glossary additions

- **pT^D**: √(Σ pT,i²)/Σ pT,i over jet constituents; high when a few hadrons carry the jet momentum. It is IRC-unsafe, because a collinear split lowers it.
- **NA− / PA− / IR+**: H57 generator settings. NA− lowers the count at unchanged pT^D. PA− lowers pT^D while N also falls. IR+ changes IRC-safe content at fixed N and pT^D.
- **κ_N, κ_pTD**: (Δs_net − Δs_EFN)/(Δs_LR(X) − Δs_EFN). This is the share of a single-observable classifier's extra shift that a network follows.
- **mayDecay off**: the Pythia switch that keeps a particle species stable, so its decay products are not produced.

## Explainer entry (for site/explain.py; paste as H57_EXPLAINER under Paper D)

```python
H57_EXPLAINER = [
 ('Question', 'H54 changed Pythia hadronisation so that quark/gluon jets got about 11 % more particles at nearly fixed IRC-safe energy flow, and the networks moved toward "gluon". But the same change also spread the jet momentum over softer hadrons, lowering $p_T^D=\\sqrt{\\sum_i p_{T,i}^2}/\\sum_i p_{T,i}$ by 4.5 %. Do the networks read the particle count $N$ or the momentum sharing $p_T^D$?', None),
 ('Physics background', 'In Pythia every string-fragmentation knob moves $N$ and $p_T^D$ together: $\\Delta p_T^D/p_T^D \\approx -0.56\\,\\Delta N/N$. Hadron decays offer a different lever. If a $\\pi^0$ is kept stable, its two nearly collinear photons become one particle: $N$ falls by about 21 % while $p_T^D$ rises only by about 4 %, a ratio of $-0.20$. Combining such decay switches with string knobs separates the two quantities.', None),
 ('Method', 'Same parton-level events (per-event seeds), only hadronisation and decays changed. NA−: $\\eta,\\omega,K_S$ and hyperons stable, with more string fragmentation, giving $N$ −7 % and $p_T^D$ unchanged. PA−: $\\pi^0$ stable with softer fragmentation, giving $N$ −5 % and $p_T^D$ −3 to −5 %. Here a count classifier predicts "more quark-like" and a $p_T^D$ classifier predicts "more gluon-like". IR+: IRC-safe content changed at fixed $N$ and $p_T^D$. 10,000 quark + 10,000 gluon jets each; $\\Delta s$ = paired mean change in quark log-odds; $\\kappa_X=(\\Delta s_{net}-\\Delta s_{EFN})/(\\Delta s_{LR(X)}-\\Delta s_{EFN})$.', 'results/physics/h57_jet.png'),
 ('Result', 'Under NA− ParT_kin moves only +0.04 (quark) and +0.01 (gluon) beyond the IRC-safe EFN ($\\kappa_N$ 0.24 / 0.05), whereas LR(N) moves +0.18 / +0.23 beyond it. Under PA− ParT_kin moves toward gluon, −0.10 / −0.23, with LR($p_T^D$) at −0.13 / −0.20, against LR(N) at +0.21 / +0.18. A fit over 38 settings gives $\\Delta s_{ParT} \\approx 0.69\\,\\Delta s_{EFN} + (0.04\\text{–}0.10)\\,\\Delta s_{LR(N)} + (0.65\\text{–}0.95)\\,\\Delta s_{LR(p_T^D)}$.', 'results/physics/h57_shifts.png'),
 ('Checks', 'Second nominal run: all $|\\Delta s|\\le0.03$. Permuted ParT: $|\\Delta s|\\le0.03$. IR+ moves the nets by 0.6–0.8 of the EFN and leaves LR(N) and LR($p_T^D$) at 0. Hz+ reproduces H54 within 0.01. Unedited accuracies match H54 within 0.004. PA− jets are easy to tell from nominal (C2ST 0.84–0.90). Networks that read particle identity are confounded by the changed hadron species.', 'results/physics/h57_settings.png'),
 ('Physical significance', 'The hadron-level information that the Pythia-trained kinematic taggers use beyond IRC-safe energy flow is mainly how the jet momentum is shared among its hadrons ($p_T^D$-like fragmentation hardness), with only a small dependence on the number of hadrons. PFN keeps a larger count component. Herwig-trained taggers use less of either.', None),
]
```
