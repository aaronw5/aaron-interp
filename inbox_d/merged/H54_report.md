# H54 — Generator-level intervention on quark/gluon jets: Pythia 8 hadronisation and shower-α_s knobs (claim D7, restated)

## Conclusion

We regenerated the quark/gluon jet sample with Pythia 8 and changed only how partons turn into hadrons. This raised or lowered the number of particles per jet by about 11 %, while jet mass, girth and the IRC-safe EFN output barely moved. The Pythia-trained ParT, ParticleNet, LorentzNet and PFN taggers all moved their quark/gluon score in the direction a particle-count classifier predicts. The kinematic-only ParT and ParticleNet moved by about half of the count-only increment, and the versions that see particle identity moved by 0.7–1.0 of it (ParT_kin gluon jets −0.22 vs EFN 0.00 log-odds). So, on realistic jets, these networks respond to hadron-level structure beyond IRC-safe energy flow, unlike their near-zero response to the hand-made count edits in H5/H14/H24. The hadronisation change also softens the hadron momentum spectrum (pT^D −4 %). This experiment therefore cannot say whether the networks read the particle count itself or the soft hadrons' momenta. Networks trained on Herwig jets respond less (0–0.45 of the increment).

## What was changed, what was held fixed, and why V0 is the control

- **Generator**: Pythia 8.317 (PyPI `pythia8mc` 8.317.2, wheel; venv /j-jepa-vol/aaron-interp/env-gen over the image's conda python; fastjet 3.4.1 installed for a cross-check). The Komiske–Metodiev–Thaler (KMT) settings come from the energyflow documentation: Pythia 8.226 with default tune (Monash 2013, `Tune:pp = 14`), pp at √s = 14 TeV, q q̄ → Z(→νν)+g (gluon sample) and q g → Z(→νν)+q with q ∈ {u, d, s} (quark sample), hadronisation and multi-parton interactions (MPI, the underlying event) on, all visible final-state particles clustered with anti-kT R = 0.4, jet pT ∈ [500, 550] GeV, |y| < 1.7. Jets are stored as (pT, y, φ, PDG id) with φ within π of the jet, as in KMT. The following choices are not stated by KMT and were made here: `PhaseSpace:pTHatMin/Max` = 450/700 GeV; only the hardest jet of each event is kept; quark sample with `PDFinProcess:nQuarkIn = 3`, plus a check that the outgoing quark is u, d or s (0 rejections); clustering with Pythia's SlowJet in fjcore mode, which gave the same leading-jet pT as fastjet on the events checked. Version 8.317 instead of 8.226 is a deviation; the validation below found it indistinguishable on the jet level.
- **Seeds**: before event i of worker w the generator is reseeded with seed(w, i) = 1 + (5401 + w)·50,000 + i. The workers are w = 0–3 for gluon and 4–7 for quark, with 7,500 jets each. Every variant therefore starts every event from the same random state. In a check job, 10/10 events had an identical parton-level record (shower, MPI, beam remnants) when only `StringZ:bLund` differed. In the pilot this held for 99.5–100 % of 200 events per setting. **A hadronisation-only variant therefore changes only the parton → hadron step of the same parton-level event.** A shower variant keeps the hard process and seeds but changes the shower.
- **Variants** (30,000 quark + 30,000 gluon jets each; S+ has 29,815 gluon jets because some workers reached the 50,000-event cap):
  - **V0** nominal Monash: `StringZ:aLund` 0.68, `StringZ:bLund` 0.98, `StringPT:sigma` 0.335 GeV, `TimeShower:alphaSvalue` 0.1365 (all read back from Pythia at run time).
  - **Hz+** (hadronisation, N up): aLund 0.79, bLund 0.88, σ 0.247 GeV. **Hz−** (N down): aLund 0.48, bLund 0.98, σ 0.438 GeV. Physics: the Lund symmetric fragmentation function f(z) ∝ z⁻¹(1−z)^a exp(−b m_T²/z) gives the light-cone momentum fraction z each new hadron takes from the string. Larger a or smaller b makes hadrons take smaller fractions, which gives more and softer hadrons. σ is the width of the Gaussian transverse momentum given to each quark–antiquark pair created in a string break. A smaller σ lowers the hadron transverse masses and lets more hadrons fit. The a/b and σ changes move the EFN output in opposite directions (pilot round 1), so they were combined to cancel the EFN shift.
  - **S+ / S−** (final-state shower): `TimeShower:alphaSvalue` 0.1465 / 0.1265, i.e. the strength of gluon emission in the final-state shower ±0.01. This moves N and the IRC-safe shapes together.
- **Held fixed**: hard process and random seeds per event, jet algorithm and pT/y window, MPI and ISR settings, all hadronisation parameters except the three named.
- **Why V0 is the control**: V0 shares code version, seeds, selection and clustering with every variant, so a variant − V0 difference isolates the parameter change. V0 also matches the KMT test sample (C2ST below), but all shifts are taken relative to V0, not KMT.

## Pilot (choice of Hz±), 1,000 q + 1,000 g jets per setting, events paired with the nominal run

Rule: the mean of the quark and gluon paired ΔN/N must lie in 0.07–0.14 in the wanted direction. Among those settings, the one with the smallest max-over-class |paired EFN shift| / |ΔN/N| was chosen. The EFN shift is the change in quark log-odds of the KMT-trained EFN. Two rounds were run (round 2 combined a/b with σ, using linear solutions from round 1). Files: results/physics/h54_pilot.json and h54_pilot_round1.json. Jobs: d-h54-pilot, d-h54-pilot2.

| setting | aLund, bLund, σ | ΔN/N % q / g | ΔEFN q / g | Δgirth/SD q / g | Δ(m/pT)/SD q / g |
|---|---|---|---|---|---|
| a+0.6 | 1.28, 0.98, 0.335 | +9.4 / +7.7 | −0.355 / −0.330 | +0.012 / +0.016 | +0.023 / +0.023 |
| b−0.45 | 0.68, 0.53, 0.335 | +9.2 / +8.1 | −0.396 / −0.383 | +0.016 / +0.024 | +0.031 / +0.035 |
| s−0.10 | 0.68, 0.98, 0.235 | +10.2 / +9.0 | +0.062 / +0.156 | −0.011 / −0.010 | −0.013 / −0.014 |
| a+0.3,b−0.2,s−0.04 | 0.98, 0.78, 0.295 | +12.4 / +10.9 | −0.290 / −0.316 | +0.006 / +0.010 | +0.020 / +0.015 |
| **a+0.11,b−0.1,s−0.088 (Hz+)** | 0.79, 0.88, 0.247 | +10.9 / +10.4 | +0.023 / −0.013 | −0.010 / −0.000 | −0.015 / −0.001 |
| a+0.15,s−0.10 | 0.83, 0.98, 0.235 | +12.1 / +10.8 | +0.023 / +0.043 | −0.010 / −0.003 | −0.014 / −0.006 |
| a−0.5 | 0.18, 0.98, 0.335 | −8.9 / −8.6 | +0.240 / +0.412 | −0.016 / −0.013 | −0.030 / −0.023 |
| b+0.8 | 0.68, 1.78, 0.335 | −8.0 / −7.9 | +0.251 / +0.345 | −0.013 / −0.010 | −0.023 / −0.023 |
| s+0.115 | 0.68, 0.98, 0.45 | −8.7 / −8.2 | −0.151 / −0.074 | +0.007 / +0.018 | +0.012 / +0.020 |
| a−0.3,b+0.3,s+0.04 | 0.38, 1.28, 0.375 | −10.5 / −10.3 | +0.213 / +0.333 | −0.008 / −0.006 | −0.014 / −0.011 |
| **a−0.2,s+0.103 (Hz−)** | 0.48, 0.98, 0.438 | −10.1 / −11.0 | −0.028 / +0.065 | +0.000 / +0.004 | −0.000 / +0.003 |
| a−0.25,s+0.095 | 0.43, 0.98, 0.43 | −10.1 / −11.1 | −0.005 / +0.104 | +0.001 / +0.001 | +0.002 / −0.001 |

(The full 24-row table is in h54_pilot.json.) Changing a or b alone moves the IRC-safe EFN output by 0.2–0.4 log-odds per 10 % of N, even though girth and mass move by ≤ 0.03 SD. Lund-string parameters therefore shift the fine-angle energy flow that the EFN reads, and only combinations with σ cancel it.

## Validation: V0 vs the KMT test file

Sample: 30,000 q + 30,000 g KMT test jets (data_qg random rows, seed 5402) vs V0. C2ST means classifier two-sample test: a gradient-boosted classifier is trained to tell V0 jets from KMT jets. Its features are N, N_charged, pT^D, log girth, m/pT, log C1^(β) for β = 0.5, 1, 2, jet pT and the five leading pT fractions. The AUC is measured on a held-out half, with a 95 % bootstrap CI over 300 resamples. Results: AUC **0.503 [0.497, 0.510] (quark), 0.505 [0.497, 0.511] (gluon)**, below the 0.55 requirement. Means V0 vs KMT: quark N 33.69 vs 33.54, pT^D 0.3810 vs 0.3813; gluon N 53.7 vs 53.2 (KS statistic ≤ 0.007 for N, pT^D, girth, mass in quark jets). Variant vs V0 C2ST: Hz+ 0.661 / 0.680, Hz− 0.658 / 0.690, S+ 0.566 / 0.565, S− 0.542 / 0.558 (quark / gluon). The variants are realistic jets (generator output) but distinguishable from V0, mostly through N.

Observable shifts per variant (relative to V0, quark / gluon): Hz+ N +11.0 / +10.7 %, pT^D −4.4 / −4.4 %, girth −0.4 / −0.6 %, m/pT −0.4 / −0.5 %, C1^(1) −0.5 / −0.6 %. Hz− N −10.4 / −10.7 %, pT^D +4.9 / +5.8 %, girth +0.5 / +0.2 %. S+ N +8.4 / +6.9 %, girth +6.9 / +5.0 %, m/pT +5.0 / +3.8 %. S− N −6.5 / −6.1 %, girth −6.6 / −6.0 %.

## Models and accuracy (unedited, on each sample used; threshold s = 0)

Scores are s = log p(quark) − log p(gluon) (quark log-odds). Accuracy / AUC on V0 ; on KMT (30k + 30k):
ParT_kin 0.8305 / 0.903 ; 0.8263 / 0.900 (H24 KMT 0.833). ParT_kinpid 0.8411 / 0.913 ; 0.8397 / 0.911. ParticleNet_kin 0.8303 ; 0.8265. ParticleNet_kinpid 0.8365 ; 0.8327. LorentzNet 0.8466 / 0.919 ; 0.8436 / 0.916 (released 0.8442 / 0.9161). PFN (H2) 0.8193 ; 0.8149. EFN (H2, IRC-safe) 0.8112 / 0.885 ; 0.8084 / 0.882 (H2 test 0.809 / 0.883). PFN/EFN second runs (H5 torch) 0.806 / 0.795 ; 0.803 / 0.791. Herwig-trained ParT_kin 0.7967 ; 0.7961, ParticleNet_kin 0.7926 ; 0.7893. Six weight-permuted ParT_kin (seeds 0–5) 0.497–0.500. Every model reproduces its known KMT accuracy within 0.007.

References were trained on 30,000 + 30,000 KMT **training** jets (not on generated jets), with V0 ; KMT accuracy given. LR(N) is a likelihood ratio from negative-binomial fits to the KMT-train multiplicity (0.766 ; 0.764). LR(pT^D) is a 40-bin histogram likelihood ratio (0.740 ; 0.737). LR(N, pT^D) is a gradient-boosted classifier on (N, pT^D) (0.773 ; 0.770). GBDT(EFP) uses the 54 IRC-safe energy-flow polynomials (EFPs) with degree ≤ 5 and β = 1 (0.775 ; 0.773). GBDT(EFP+N) adds N to the same EFPs (0.785 ; 0.783). The EFN reference is the H2 EFN above.

## Metric and statistics

Δs = ⟨s⟩_variant − ⟨s⟩_V0 per class and model. The 95 % CI comes from 1,000 bootstrap resamples over jets, with the two samples resampled independently (unpaired). A network-minus-reference difference uses the same resample indices for both, because both are scored on the same jets. Its p-value is two-sided from z = difference / bootstrap SE. Holm correction is applied over 344 tests: 11 non-null models × 4 variants × 2 classes × references {EFN, GBDT(EFP), LR(N), GBDT(EFP+N)}, excluding self-comparisons. κ = (Δs_net − Δs_GBDT(EFP)) / (Δs_GBDT(EFP+N) − Δs_GBDT(EFP)) is the share of the "N increment" a network follows: 0 = like the IRC-safe reference, 1 = like the same reference with N added. Its CI comes from the bootstrap. H0 (D7 restated): Δs_net = the EFN / GBDT(EFP) prediction. H1 (counting): Δs_net = the LR(N) / GBDT(EFP+N) prediction.

## Numbers

**Hadronisation-only variants** (Δs, quark log-odds, [95 % CI]; quark jets ; gluon jets):

| model | Hz+ (N +11 %) | κ Hz+ | Hz− (N −10.5 %) | κ Hz− |
|---|---|---|---|---|
| ParT_kin | −0.136 [−0.164, −0.109] ; −0.215 [−0.237, −0.192] | 0.49 ; 0.51 | +0.140 ; +0.328 [0.305, 0.352] | 0.52 ; 0.65 |
| ParticleNet_kin | −0.136 ; −0.206 | 0.49 ; 0.49 | +0.134 ; +0.323 | 0.50 ; 0.64 |
| ParT_kinpid | −0.225 ; −0.331 | 0.77 ; 0.78 | +0.192 ; +0.430 | 0.71 ; 0.86 |
| ParticleNet_kinpid | −0.249 ; −0.403 | 0.85 ; 0.94 [0.90, 0.98] | +0.213 ; +0.491 | 0.79 ; 0.99 [0.96, 1.03] |
| LorentzNet | −0.207 ; −0.360 | 0.71 ; 0.84 | +0.188 ; +0.453 | 0.70 ; 0.91 |
| PFN | −0.182 ; −0.314 | 0.63 ; 0.74 | +0.179 ; +0.398 | 0.66 ; 0.80 |
| EFN (IRC-safe) | +0.025 [−0.003, 0.052] ; −0.004 [−0.025, 0.020] | −0.02 ; 0.03 | −0.003 ; +0.070 [0.048, 0.092] | −0.01 ; 0.12 |
| ParT_kin, Herwig-trained | −0.059 ; +0.005 [−0.011, 0.021] | 0.24 ; 0.01 | +0.121 ; +0.129 | 0.45 ; 0.24 |
| ParticleNet_kin, Herwig-trained | −0.039 ; +0.067 | 0.18 ; −0.13 | +0.117 ; +0.089 | 0.44 ; 0.16 |
| permuted ParT_kin ×6 | \|Δs\| ≤ 0.018 (all variants) | | | |
| GBDT(EFP) | +0.018 ; +0.010 | | −0.000 ; +0.014 | |
| GBDT(EFP+N) | −0.298 ; −0.428 | | +0.270 ; +0.495 | |
| LR(N) | −0.422 ; −0.421 | | +0.440 ; +0.471 | |
| LR(N, pT^D) | −0.383 ; −0.317 | | +0.385 ; +0.395 | |
| LR(pT^D) | −0.188 ; −0.185 | | +0.197 ; +0.235 | |

Network − EFN in Hz±: −0.16 to −0.27 (quark), −0.20 to −0.40 (gluon) for the Pythia-trained nets in Hz+, mirrored in Hz−. Holm-significant in 37 of 40 Hz tests against EFN, 37/44 against GBDT(EFP), 41/44 against LR(N) and 42/44 against GBDT(EFP+N). The exceptions where a network equals a counting reference are ParticleNet_kinpid gluon Hz− − GBDT(EFP+N) −0.005 [−0.020, 0.012], LorentzNet gluon Hz− − LR(N) −0.018 [−0.039, 0.004] and ParticleNet_kinpid gluon Hz+ − LR(N) +0.017 [−0.001, 0.038]. Event-paired shifts (55,515 Hz+ and 55,139 Hz− events with a selected jet in both runs) agree with the unpaired ones within 0.01. For example, ParT_kin gluon Hz+ gives −0.219 [−0.234, −0.204], EFN −0.006 [−0.019, 0.007] and LR(N) −0.427.

**Shower variants** (quark ; gluon). S+: ParT_kin −0.374 ; −0.263, EFN −0.363 ; −0.248, LR(N) −0.325 ; −0.277, GBDT(EFP) −0.300 ; −0.169, GBDT(EFP+N) −0.361 ; −0.274. Ratio net/EFN for Pythia-trained nets 0.99–1.06 (q) and 1.03–1.15 (g). Ratio net/LR(N) 1.10–1.19 (q) and 0.92–1.03 (g). S−: ParT_kin +0.281 ; +0.252, EFN +0.288 ; +0.246, LR(N) +0.267 ; +0.256. Net/EFN 0.96–0.99 ; 1.00–1.06, net/LR(N) 1.04–1.07 ; 0.96–1.01. In S± the EFN, LR(N) and GBDT(EFP+N) predictions lie within 0.05 of each other, so this arm does not separate H0 from H1. All Pythia-trained nets follow both predictions within 0.06. Herwig-trained nets on gluon jets: S+ −0.158 / −0.141 (net/EFN 0.64 / 0.57), S− +0.165 / +0.163 (0.67 / 0.66).

**Reweighting check** (V0 jets reweighted to the variant's N distribution within deciles of the EFN output, 15 N bins, effective sample 23,000–25,000 of 30,000; 200 bootstrap resamples). Hz+ predicts ParT_kin −0.089 [−0.095, −0.084] (q) and −0.124 (g), ParT_kinpid −0.096 / −0.106, LorentzNet −0.110 / −0.110, EFN −0.015 / −0.019 and LR(N) −0.423 / −0.408. It also predicts −0.184 / −0.125 for GBDT(EFP), so at fixed EFN output N still correlates with other IRC-safe information, and this is not a pure-count prediction. The reweighting accounts for about 0.6 of ParT_kin's measured Hz+ shift (q −0.089 vs −0.136; g −0.124 vs −0.215) and about 0.3–0.45 of the kinpid / LorentzNet shifts.

## Reading (scoped)

- Under a change of hadronisation alone (same parton-level events, N ±11 %, IRC-safe shapes ≤ 0.6 %, EFN and GBDT(EFP) within ±0.07), every Pythia-trained network moves in the direction of the counting references. The kinematic ParT/ParticleNet move by 0.49–0.65 of the GBDT(EFP) → GBDT(EFP+N) increment. Models with particle-identity inputs (ParT_kinpid, ParticleNet_kinpid) and LorentzNet (which also uses identity) move by 0.70–0.99, and PFN by 0.63–0.80. H0 ("no response to N beyond IRC-safe energy flow") is excluded for all six Pythia-trained non-IRC-safe networks in both directions and both classes (Holm p < 1e-8). H1 in its full form (Δs = GBDT(EFP+N)) is excluded for the kinematic models (κ CI 0.44–0.68) but not for ParticleNet_kinpid on gluon jets (κ 0.94 and 0.99).
- The hadronisation change also lowers pT^D by 4–6 %, softens the hadron momentum spectrum and changes hadron masses (σ). LR(pT^D) alone predicts −0.19. This experiment does not separate "reads N" from "reads soft-hadron momenta / pT^D". H24v found that removing soft particles at fixed N moves the networks, consistent with the second.
- The contrast with H5/H14/H18/H24, where the networks responded at most about a tenth of LR(N) to added or split particles, means that the earlier hand-made count edits do not reproduce the networks' response to a realistic multiplicity change. They add particles at fixed momenta of the others, whereas hadronisation redistributes momentum over more, softer hadrons.
- Shower α_s variants move the networks with the IRC-safe EFN (ratio about 1.0). Here N and IRC-safe shapes move together, and the counting references predict almost the same shift.
- Herwig-trained ParT/ParticleNet respond about 2–4 times less to the Pythia hadronisation change (κ −0.13 to 0.45) and about 0.6 of EFN to the gluon-jet shower change. The size of the hadron-level response depends on the training generator.

## Issues / limits

- Pythia 8.317 instead of 8.226, and pTHat window, hardest-jet-only and nQuarkIn are my choices. The C2ST of 0.50 shows no detectable jet-level mismatch, but older-version differences (e.g. shower details) are not excluded at higher precision.
- Hz± move N together with pT^D and the hadron species and mass composition, so "count vs soft momenta" is not separated. A follow-up could hold pT^D fixed by combining knobs, or use `StringFlav` (species) at fixed N.
- κ for S− quark jets has a wide CI because the denominator GBDT(EFP+N) − GBDT(EFP) is only 0.024.
- Log-odds shifts of classifiers with different separation power are compared directly. The κ uses two references with the same feature set, plus N, to reduce this.
- One training run per network (no seed spread). Seed replicates part_kin_s1 / pnet_kin_s1 exist and are being analysed in H56, so not used here.
- site/explain.py is outside this worker's write scope. The explainer entry and glossary entries are given below, ready to paste.
- GPU job: the first two attempts were rejected on node k8s-chase-ci-01 ("GPU is lost", UnexpectedAdmissionError). interp/h54_gpu_job.sh is a copy of part_job.sh that excludes that node. The first venv build (pythia8mc 8.318.0, built from source) was stopped and pinned to the 8.317.2 wheel. Neither issue changes a number.

## Files / jobs

Script interp/h54_qg_generator_knobs.py (steps check, pilot, gen, score, refs, stats, fig); interp/h54_gpu_job.sh. Jobs: d-h54-check3, d-h54-pilot, d-h54-pilot2, d-h54-gen-{v0,hzp,hzm,sp,sm} (8 CPU / 4 Gi, 10–19 min each), d-h54-score3 (GPU), d-h54-refs (8 CPU / 12 Gi, venv-ef), d-h54-stats (1 CPU / 2 Gi). Data: /j-jepa-vol/aaron-interp/data/qg/gen/{V0,Hzp,Hzm,Sp,Sm}.npz (+ .json with per-worker summaries). Results: results/physics/h54_run.json, h54_scores.npz, h54_refs.npz, h54_valid.json, h54_models.json, h54_pilot.json, h54_pilot_round1.json, h54_gen_*.json. Figures: results/physics/h54_variants.png (N and EFN-output distributions per variant, and one gluon-jet event in all five variants), results/physics/h54_shifts.png (Δs per model and variant with 95 % CI).

## Explainer entry (for site/explain.py; paste as H54_EXPLAINER and list under Paper D)

```python
H54_EXPLAINER = [
 ('Question', 'Hand-made edits that add or split particles barely move the quark–gluon taggers (H5, H14, H24), which led to claim D7: the taggers do not count particles. A reviewer asked for an on-manifold test: change the jets at the generator level so that the particle count $N$ changes while the IRC-safe energy flow stays nearly fixed. Do the networks then move like an IRC-safe classifier (no response) or like a counting classifier?', None),
 ('Physics background', 'Gluons radiate with colour factor $C_A=3$, quarks with $C_F=4/3$, so gluon jets have more particles. In Pythia the multiplicity also depends on hadronisation: the Lund fragmentation function $f(z)\\propto z^{-1}(1-z)^a\\exp(-b\\,m_T^2/z)$ sets the momentum fraction $z$ each new hadron takes from the colour string (larger $a$ or smaller $b$: more, softer hadrons), and $\\sigma$ (StringPT:sigma) sets the transverse momentum of each string break. Infrared- and collinear-safe (IRC-safe) observables such as jet mass, girth and energy-flow polynomials (EFPs) depend weakly on hadronisation; $N$ and $p_T^D=\\sqrt{\\sum_i p_{T,i}^2}/\\sum_i p_{T,i}$ depend on it directly. The final-state shower coupling $\\alpha_s$ (TimeShower:alphaSvalue) changes how much the partons radiate, so it moves $N$ and the IRC-safe shapes together.', None),
 ('Method', 'Pythia 8.317 with the Komiske–Metodiev–Thaler settings (Monash tune, $\\sqrt s$ = 14 TeV, $Z(\\nu\\nu)$+g and $Z(\\nu\\nu)$+uds, MPI on, anti-$k_T$ $R=0.4$, $p_T$ 500–550 GeV, $|y|<1.7$). Five variants with the same per-event random seeds: V0 nominal; Hz+ (aLund 0.79, bLund 0.88, $\\sigma$ 0.247 GeV) and Hz− (aLund 0.48, $\\sigma$ 0.438 GeV), chosen in a pilot so that $N$ changes by about ±10 % with the smallest change of the IRC-safe EFN output; S± with final-state $\\alpha_s$ = 0.1365 ± 0.01. Hadronisation variants share the parton-level event exactly. 30,000 quark + 30,000 gluon jets per variant. Metric $\\Delta s=\\langle s\\rangle_{\\rm variant}-\\langle s\\rangle_{V0}$ of the quark log-odds $s=\\log p(q)/p(g)$, per class; 95 % bootstrap CI; references trained on KMT training jets: LR($N$), LR($p_T^D$), LR($N,p_T^D$), EFN, GBDT(EFP), GBDT(EFP+$N$). $\\kappa=(\\Delta s_{\\rm net}-\\Delta s_{\\rm EFP})/(\\Delta s_{\\rm EFP+N}-\\Delta s_{\\rm EFP})$ is the share of the multiplicity increment a network follows (0 = IRC-safe, 1 = counting). Holm over 344 tests.', 'results/physics/h54_variants.png'),
 ('Result', 'Hz+ raises $N$ by 11 % and changes girth and mass by ≤ 0.6 %; EFN moves by $+0.025$ (quark) / $-0.004$ (gluon) and GBDT(EFP) by $\\le0.02$, while LR($N$) moves by $-0.42$. ParT_kin moves by $-0.136\\,[-0.164,-0.109]$ / $-0.215\\,[-0.237,-0.192]$ ($\\kappa\\approx0.5$), ParticleNet_kinpid by $-0.249$ / $-0.403$ ($\\kappa$ 0.85 / 0.94), LorentzNet $\\kappa$ 0.71 / 0.84; Hz− mirrors this. Herwig-trained ParT/ParticleNet: $\\kappa$ −0.13 to 0.45. Shower variants: every Pythia-trained network follows the EFN prediction (ratio 0.96–1.15), which here is close to the LR($N$) prediction.', 'results/physics/h54_shifts.png'),
 ('Checks', 'V0 vs the KMT test file: C2ST AUC 0.503 (quark) / 0.505 (gluon); model accuracies on V0 and KMT agree within 0.007 and with earlier runs (ParT_kin 0.826 on KMT, LorentzNet 0.844). Six weight-permuted ParT_kin: $|\\Delta s|\\le0.018$. Event-paired shifts agree with unpaired ones within 0.01. Reweighting V0 jets to the Hz+ multiplicity within EFN-output deciles predicts about 0.6 of ParT_kin\'s shift but also moves GBDT(EFP), so it is not a pure-count prediction.', None),
 ('Physical significance', 'On realistic jets whose particle count changes through hadronisation, the Pythia-trained taggers respond beyond what IRC-safe energy flow predicts — about half of a counting classifier\'s response for the kinematic models and up to all of it for models with particle identity. The earlier conclusion that the taggers ignore $N$ holds only for hand-made edits. Because the same hadronisation change also softens the hadrons ($p_T^D$ −4 %), whether the networks read the count or the soft hadrons\' momenta is not separated.', None),
]
```

Glossary additions (explain.GLOSSARY):
- ('Lund string parameters a, b, σ (H54)', 'Pythia hadronisation parameters: StringZ:aLund (a) and StringZ:bLund (b) shape the Lund fragmentation function f(z) ∝ z⁻¹(1−z)^a exp(−b m_T²/z), the probability that a new hadron takes light-cone momentum fraction z from the string (larger a or smaller b: more, softer hadrons); StringPT:sigma (σ) is the width of the transverse momentum given to each quark–antiquark pair created when the string breaks. Monash values a = 0.68, b = 0.98 GeV⁻², σ = 0.335 GeV')
- ('Hz±, S± variants (H54)', 'Pythia 8 resimulations of the quark/gluon sample: Hz± change only hadronisation (a, b, σ) so that the particle count N moves by about ±10 % at nearly fixed IRC-safe shapes; S± change the final-state shower coupling α_s by ±0.01, which moves N and IRC-safe shapes together; V0 is the nominal run with the same per-event seeds')
- ('κ (count share, H54)', 'κ = (Δs_net − Δs_GBDT(EFP)) / (Δs_GBDT(EFP+N) − Δs_GBDT(EFP)): the fraction of the multiplicity-driven shift of a classifier on IRC-safe energy-flow polynomials plus N that a network follows; 0 = responds like the IRC-safe classifier, 1 = like the classifier that also counts')
- ('EFN, PFN', 'energy-flow network (EFN): Σ_i z_i Φ(Δy_i, Δφ_i), IRC-safe by construction because each particle enters linearly in its momentum fraction z_i; particle-flow network (PFN): Σ_i Φ(z_i, Δy_i, Δφ_i), not IRC-safe') (if not already present)
- ('pT^D', 'pT^D = sqrt(Σ_i pT_i²)/Σ_i pT_i: large when a few hard particles carry the jet (quark-like), small for many soft particles (gluon-like); not IRC-safe') (if not already present)
- ('Casimir factors C_F, C_A', 'colour charges that set how strongly quarks (C_F = 4/3) and gluons (C_A = 3) radiate; the ratio C_A/C_F = 9/4 sets the leading-order gluon/quark multiplicity ratio') (if 'Casimir scaling' does not already cover it)
