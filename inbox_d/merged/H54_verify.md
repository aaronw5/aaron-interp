# H54 verify: generator-level hadronisation and shower knobs on quark/gluon jets

## Conclusion

When Pythia's hadronisation settings alone are changed so that quark and gluon jets carry about 10 % more or fewer particles, with jet mass, girth and the IRC-safe energy-flow network (EFN) output nearly unchanged, the Pythia-trained ParT, ParticleNet, LorentzNet and PFN taggers move their quark/gluon scores in the direction a particle-count classifier predicts. A rerun with new random seeds gives the same result. The same happens when only one knob, the string transverse-momentum width, is changed, and in that setting the IRC-safe EFN moves the opposite way. So the networks use hadron-level information that IRC-safe energy flow does not carry. The kinematic-only networks follow roughly a third to two-thirds of a count classifier's response, depending on which count classifier is the reference, and the networks with particle identity follow most or all of it. It is still open whether they read the number of particles or the softer momenta of the extra hadrons, because every hadronisation knob tested changes both together. Networks trained on Herwig jets respond much less.

## What was checked

1. **Code review** of interp/h54_qg_generator_knobs.py, covering generation, pairing, observables, scoring signs, padding and truncation, statistics and figures.
2. **Fresh-sample rerun** (job d-h54-verify, CPU, then d-h54-verify-sc, GPU) of the H54 Hz+ and Hz− settings. The rerun uses new per-event seeds and an independent V0 (nominal Monash run).
3. **A control H54 did not have.** Hz+ and Hz− were chosen in the pilot so that the EFN output does not move, and that zero comes from a cancellation. The Lund a/b knobs move the EFN toward gluon. The string-pT width σ moves it toward quark. If the networks read only IRC-safe information but weight these two IRC-safe changes differently from the EFN, they would move at the chosen point without reading anything beyond IRC-safe energy flow. The control uses single-knob settings with about the same |ΔN/N| (8–9 %) but EFN shifts of opposite sign:
   - σ only: N up via σ 0.335 → 0.235 GeV, N down via σ → 0.45 GeV.
   - a only: N up via aLund 0.68 → 1.28, N down via aLund → 0.18.

   It also adds a dose-response fit over all 30 hadronisation settings: the 24 pilot settings from H54 plus the 6 fresh ones.
4. **Figures** h54_variants.png and h54_shifts.png were read.

## Code review

No error that changes a reported number was found.

- **Seeds and pairing.** Before event i of worker w, the generator is reseeded with seed 1 + (5401+w)·50,000 + i. The fresh run uses workers 6001–6004, so its seeds are 1 + w·50,000 + i. The seed ranges of the main run, the pilot and the fresh run do not overlap, and all stay below Pythia's limit of 9×10⁸. Pairing uses the key widx·100,000 + event with event < 50,000, so keys are unique. In the fresh run, the parton-level record (shower, MPI, beam remnants) is identical to V0 in 399/400 checked events for every hadronisation setting.
- **Score signs.** M.logit returns log p(g) − log p(q) for every model kind: weaver models label [isQ, isG], and LorentzNet and EF models label 1 = quark. H54 negates it for every model, so the sign convention is consistent across models. Accuracies of about 0.83 confirm the signs. The permuted network sits at 0.50.
- **Reference signs.** The sign of LR(N) = −T['py'] is confirmed by its accuracy of 0.765. The GBDT decision_function is the quark log-odds.
- **Truncation.** Weaver and EF inputs are cut to the 128 hardest particles and LorentzNet to 148. The KMT sample has 6 of 60,000 jets with N > 128, and Hz+ gluon jets have a few more. This is negligible for the means.
- **Observables.** Observables use the same lists the models see, with massless four-vectors built from (pT, y, φ) as in the dataset conversion. Axis: E-scheme. N counts only real particles (pt > 0).
- **Choice of setting and reference.** The pilot chose Hz± by minimising the shift of efn_k, and efn_k is then the H0 reference. Two things limit selection on noise: the pilot used other seeds (5501/5502), and the independent IRC-safe references (efn_t, GBDT(EFP)) also stay near zero. The cancellation concern is not ruled out by this, which is why the single-knob control (item 3 above) was run.
- **H1 for ParticleNet_kinpid.** H54's statement "full H1 not excluded for ParticleNet_kinpid on gluon jets" rests on the Holm z-test of network − GBDT(EFP+N): p = 0.28 in Hz+ and 1.0 in Hz−. In Hz+ the bootstrap κ interval [0.90, 0.98] excludes 1. The two criteria disagree for this one case.
- **Figures.**
  - h54_shifts.png matches h54_run.json.
  - In h54_variants.png the legends cover the top y-axis tick labels.
  - The single example event shows ParT_kin moving toward gluon in Hz− (s −0.21 → −0.70), which is the opposite of the mean shift. One event is not representative, and the caption does not say so.

## Fresh-sample rerun

### Procedure

- Generator: Pythia 8.317 (env-gen) with the H54 settings: Monash tune, 14 TeV, Z(νν)+g and Z(νν)+uds, MPI on, anti-kT R = 0.4, jet pT 500–550 GeV, |y| < 1.7.
- Sample: 8,000 quark + 8,000 gluon jets per setting, 7 settings, worker ids 6001–6004.
- Pairing: about 7,150–7,560 jet pairs per class share an event with V0.
- Metric: Δs = mean over event-paired jets of (s_variant − s_V0), where s = log p(quark) − log p(gluon). The 95 % CI comes from 1,000 bootstrap resamples over pairs.
- Network − EFN: the difference of the two Δs on the same resamples, with a two-sided z-test p-value. Holm correction is applied over the 96 net − EFN tests (8 trained nets × 6 settings × 2 classes).
- κ_LR = (Δs_net − Δs_EFN)/(Δs_LR(N) − Δs_EFN). This is the share of the gap between the IRC-safe EFN and the multiplicity likelihood ratio LR(N) that a network covers: 0 means it moves like the EFN, 1 means it moves like LR(N). The GBDT(EFP) references were not retrained here. H54's κ uses GBDT(EFP) → GBDT(EFP+N), whose N increment is smaller (0.30–0.50 vs LR(N) 0.42–0.46).
- Unedited accuracy on the fresh V0 (16,000 jets): ParT_kin 0.825, ParT_kinpid 0.837, PN_kin 0.825, PN_kinpid 0.832, LorentzNet 0.842, PFN 0.815, EFN 0.805, EFN (2nd run) 0.785, Herwig-trained ParT_kin 0.795 and PN_kin 0.789, permuted ParT_kin 0.499, LR(N) 0.765, LR(pT^D) 0.738.

### Hz± replication

Δs is in quark log-odds. Each cell gives the fresh value and the H54 value.

| model | Hz+ quark | Hz+ gluon | Hz− quark | Hz− gluon |
|---|---|---|---|---|
| ΔN/N | +10.9 % ; +11.0 % | +11.1 % ; +10.7 % | −10.6 % ; −10.4 % | −10.4 % ; −10.7 % |
| ParT_kin | −0.142 [−0.168, −0.114] ; −0.136 | −0.256 [−0.285, −0.224] ; −0.215 | +0.152 ; +0.140 | +0.301 ; +0.328 |
| ParticleNet_kinpid | −0.275 ; −0.249 | −0.445 ; −0.403 | +0.229 ; +0.213 | +0.464 ; +0.491 |
| LorentzNet | −0.216 ; −0.207 | −0.409 ; −0.360 | +0.206 ; +0.188 | +0.419 ; +0.453 |
| PFN | −0.196 ; −0.182 | −0.356 ; −0.314 | +0.184 ; +0.179 | +0.361 ; +0.398 |
| EFN (IRC-safe) | +0.018 [−0.005, 0.041] ; +0.025 | −0.030 [−0.056, −0.003] ; −0.004 | +0.001 ; −0.003 | +0.050 [0.022, 0.075] ; +0.070 |
| Herwig ParT_kin | −0.064 ; −0.059 | −0.006 ; +0.005 | +0.123 ; +0.121 | +0.119 ; +0.129 |
| permuted ParT_kin | −0.007 | −0.009 | +0.002 | +0.004 |
| LR(N) | −0.417 ; −0.422 | −0.438 ; −0.421 | +0.450 ; +0.440 | +0.456 ; +0.471 |

Other observables in Hz+ (fresh): girth, m/pT and C1^(1) change by at most 0.5 %, and pT^D changes by −4.5 %. Network − EFN for the six Pythia-trained nets in Hz± is between −0.16 and −0.42 in Hz+ and between +0.14 and +0.41 in Hz−, with all Holm p < 1e-50. The κ_LR values are:

| network | κ_LR range across Hz±, quark and gluon |
|---|---|
| ParT_kin | 0.34–0.62 |
| ParticleNet_kin | 0.30–0.62 |
| ParT_kinpid | 0.44–0.86 |
| ParticleNet_kinpid | 0.51–1.02 |
| LorentzNet | 0.46–0.93 |
| PFN | 0.41–0.80 |
| Herwig-trained ParT/ParticleNet | −0.19 to 0.27 |

### Single-knob control

Paired Δs in quark log-odds, quark ; gluon.

| setting | ΔN/N | ΔpT^D | EFN | LR(N) | ParT_kin | ParticleNet_kinpid | LorentzNet | Herwig ParT_kin | permuted |
|---|---|---|---|---|---|---|---|---|---|
| σ only, N up (σ 0.235) | +8.5 / +8.7 % | −3.0 / −3.8 % | **+0.144 ; +0.125** | −0.325 ; −0.346 | **−0.008 ; −0.091** | −0.131 ; −0.295 | −0.085 ; −0.256 | +0.039 ; +0.061 | −0.005 ; −0.006 |
| σ only, N down (σ 0.45) | −8.7 / −8.8 % | +4.2 / +4.1 % | **−0.128 ; −0.097** | +0.362 ; +0.381 | **+0.030 ; +0.155** | +0.132 ; +0.315 | +0.089 ; +0.265 | +0.017 ; +0.028 | +0.005 ; +0.003 |
| a only, N up (a 1.28) | +7.9 / +7.5 % | −4.2 / −4.6 % | −0.318 ; −0.338 | −0.314 ; −0.311 | −0.353 ; −0.429 | −0.344 ; −0.475 | −0.335 ; −0.488 | −0.286 ; −0.203 | −0.004 ; −0.008 |
| a only, N down (a 0.18) | −8.7 / −7.7 % | +5.7 / +5.0 % | +0.256 ; +0.358 | +0.370 ; +0.381 | +0.277 ; +0.451 | +0.281 ; +0.488 | +0.276 ; +0.503 | +0.281 ; +0.230 | +0.000 ; +0.003 |

Girth, m/pT and C1 change by 1–2 % in these single-knob settings.

- **σ-only settings.** The IRC-safe EFN (and the second EFN run) moves toward quark when N rises. Every Pythia-trained network instead moves toward gluon or stays at zero: ParT_kin on quark jets is −0.008 [−0.034, 0.020]. The direction is mirrored when N falls. Network − EFN ranges from −0.15 to −0.42 (N up) and from +0.15 to +0.41 (N down), all Holm p < 1e-20. The CI of ParT_kin quark Δs, [−0.034, 0.020], excludes the EFN prediction of +0.144.
- **a-only settings.** The EFN and LR(N) predict nearly the same shift, so these settings cannot tell H0 from H1. The networks follow both and exceed the EFN by 0.07–0.15 on gluon jets.
- **Herwig-trained networks.** In the σ-only settings they move by |Δs| ≤ 0.12, between the EFN and LR(N) predictions.

### Dose-response fit over 30 hadronisation settings

Data: the 24 pilot settings from H54 (1,000 + 1,000 jets each, paired with the pilot nominal run) plus the 6 fresh settings.

Model: Δs_net = α·Δs_EFN + β·Δs_LR(N), with no intercept and weights 1/SE². Coefficient CIs come from 2,000 resamples of the settings.

Under H0 (the network responds like an IRC-safe classifier) the prediction is α ≈ 1 and β = 0. Under H1 (counting) it is β > 0.

| network | α (q ; g) | β (q ; g) | χ²/ndf two-term | χ² EFN-only (ndf 29) |
|---|---|---|---|---|
| ParT_kin | 0.73 [0.66, 0.80] ; 0.75 [0.70, 0.81] | 0.34 [0.31, 0.36] ; 0.57 [0.54, 0.60] | 37/28 ; 41/28 | 739 ; 1496 |
| ParticleNet_kin | 0.68 ; 0.78 | 0.33 [0.29, 0.37] ; 0.54 [0.51, 0.58] | 70/28 ; 41/28 | 792 ; 1373 |
| ParT_kinpid | 0.57 ; 0.60 | 0.49 ; 0.83 | 78/28 ; 28/28 | 1324 ; 2523 |
| ParticleNet_kinpid | 0.46 ; 0.47 | 0.55 ; 0.98 [0.96, 1.01] | 71/28 ; 21/28 | 1835 ; 3895 |
| LorentzNet | 0.53 ; 0.58 | 0.46 ; 0.88 | 51/28 ; 21/28 | 1263 ; 2836 |
| PFN | 0.58 ; 0.53 | 0.43 ; 0.76 | 78/28 ; 30/28 | 1395 ; 2857 |
| Herwig ParT_kin | 0.71 ; 0.52 | 0.24 [0.21, 0.27] ; 0.10 [0.03, 0.17] | 24/28 ; 144/28 | 629 ; 274 |
| Herwig ParticleNet_kin | 0.80 ; 0.58 | 0.21 ; −0.02 [−0.10, 0.07] | 18/28 ; 178/28 | 439 ; 182 |
| permuted ParT_kin | −0.01 ; 0.00 | 0.01 ; 0.01 | | |

- **H0 excluded.** For every Pythia-trained network, β > 0 with CIs that exclude 0, and adding the LR(N) term lowers χ² by a factor of 11 to 185.
- **Size of the IRC-safe term.** The networks respond to the IRC-safe part with α = 0.46–0.78 of the EFN's response. They are a weighted mixture of the EFN's response and a count response, not the EFN's response plus a count term.
- **Count versus pT^D.** Adding a third regressor, Δs_LR(pT^D), gives LR(pT^D) coefficients with opposite signs on quark and gluon jets, and every CI except one spans zero. In these settings N and pT^D move together (ΔpT^D/pT^D = −0.35 to −0.66 times ΔN/N in the fresh settings), so the fit cannot separate counting from soft-hadron momenta. This matches H54's stated limit.

## Assessment of the reading

The reading survives with these corrections and additions:

1. **The H0 exclusion does not depend on the tuned EFN cancellation.** With σ alone the EFN moves toward quark and the networks toward gluon, and over 30 settings an EFN-only description fails for every Pythia-trained network.
2. **"About half" depends on the reference.** The kinematic ParT/ParticleNet follow 0.49–0.65 of the GBDT(EFP)→GBDT(EFP+N) increment (H54) but 0.30–0.62 of the EFN→LR(N) gap (this rerun), and β = 0.33–0.57 in the fit. "A third to two-thirds" is the scoped range.
3. **The IRC-safe term is not the EFN's full response.** In the fit the networks follow the EFN's IRC-safe response with α = 0.46–0.78, not α = 1. Shift ratios in log-odds also depend on each classifier's separation power, so these fractions are descriptive.
4. **ParticleNet_kinpid on gluon jets in Hz+.** H54's κ CI [0.90, 0.98] excludes 1 while its Holm test against GBDT(EFP+N) does not (p = 0.28). The fresh κ_LR is 1.02 [0.95, 1.08].
5. **Not rerun:** the shower-α_s arm (S±) and the reweighting check. The count versus soft-momentum limit stands.

## Files and jobs

- Script: interp/h54_verify.py.
- Jobs:
  - d-h54-verify: generation, 8 CPU / 4 Gi, 22 min, 7.2 cores on average. Peak memory was 465 MiB, which is below the 20 % memory floor of the request; later jobs of this size should request 1–2 Gi.
  - d-h54-verify-sc: GPU, via interp/h54_gpu_job.sh, 13 min.
- Data: /j-jepa-vol/aaron-interp/data/qg/gen/verify/{V0,Hzp,Hzm,sig_Nup,a_Nup,sig_Ndn,a_Ndn}.npz and gen_info.json.
- Results: results/physics/h54_verify.json and h54_verify_scores.npz, on the volume and copied locally.

## Explainer addition (for the H54 entry in site/explain.py, "Checks")

> Verification (H54v): a rerun with new seeds reproduced the Hz± shifts within 0.05 log-odds. Control: changing only the string-pT width $\sigma$ raises $N$ by 8.5 % while the IRC-safe EFN moves toward quark ($+0.14$); ParT_kin does not move ($-0.008$) and ParticleNet_kinpid moves toward gluon ($-0.13$ quark, $-0.30$ gluon jets). The fit over 30 hadronisation settings, $\Delta s_{\rm net}=\alpha\,\Delta s_{\rm EFN}+\beta\,\Delta s_{{\rm LR}(N)}$, gives $\beta=0.33$–$0.98$ for the Pythia-trained networks and $\beta\approx0$–$0.24$ for the Herwig-trained ones. The EFN-only description ($\beta=0$) fails ($\chi^2$ 739–3895 for 29 degrees of freedom).

Glossary addition: ('κ_LR, α/β dose-response (H54v)', 'κ_LR = (Δs_net − Δs_EFN)/(Δs_LR(N) − Δs_EFN): the share of the gap between the IRC-safe EFN and the multiplicity likelihood ratio that a network covers under a generator change; α, β: weights of the EFN shift and the LR(N) shift in a fit of a network\'s shift over many hadronisation settings (α = 1, β = 0 means the network moves like the IRC-safe EFN)').
