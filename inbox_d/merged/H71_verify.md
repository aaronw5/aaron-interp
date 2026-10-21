# H71 verify: energy-scale panel, dimensionful vs dimensionless reading (loop D)

Verifier node for H71. Script `interp/h71_verify.py` (new; H71's script and results not modified).
Jobs: `anrunw-ai-d-h71-verify` (GPU, JetClass fresh sample, 6 min), `anrunw-ai-d-h71-vst` (CPU 1/2Gi, statistics, 25 s).
Results: `results/physics/h71_verify_jetclass.{json,npz}` (per-jet own-class log-odds per model × edit), `results/physics/h71_verify.json` (ρ with and without the pT-range cut, the new B control, per-class tests, Holm). Logs: `logs/d-h71-verify.log`, `logs/d-h71-vst.log` on the volume.

## Conclusion

On a new set of JetClass jets, scaling every particle's energy and momentum by the same factor again changed the scores of ParT and ParticleNet. The change was 0.62 (ParT, full inputs) to 0.92 (ParT, kinematics only) times the change from the same mass shift at fixed jet pT, as H71 found. Keeping only jets whose scaled pT stays inside the training range changes this ratio by at most 0.04 on JetClass and by about 0.01 on Top and quark–gluon jets. Leaving the training range therefore does not explain the result, and H71's main open point is closed. Two parts of the reading are weaker than stated. A pT change at fixed mass also moves the scores, by about a quarter as much as the mass change, and the two effects do not add. The charged-versus-neutral scaling difference also appears in kinematics-only models (6 of 10 classes for ParticleNet_kin), so it is not mainly a particle-type effect, except in W jets.

## What was checked

**Code review of `interp/h71_energy_scale_panel.py`.**
- U: `scale_mask` multiplies pT and particle mass by λ at fixed (η, φ), so E, px, py, pz all scale by λ. Padded slots (pT = 0) stay 0. Correct.
- UB: `h12_boost_panel.boost(U, 1/λ)` is an exact Lorentz boost along the jet three-momentum, so the jet invariant mass is preserved and |P| returns to its original value. The stored axis is boosted with it. Correct. The H71 report's correction ("UB keeps pT, not m/pT") is right.
- JetClass `build` recomputes jet_pt and jet_energy from the edited particles. The ratio inputs log(pT/pT_jet) and log(E/E_jet) are therefore unchanged by U, as the reading assumes. The η convention (η = jet_eta + sign(jet_eta)·Δη) is applied consistently in `prep` (the sign factors cancel).
- `sortT` sorts the extra per-particle arrays (charge, d0, …) with pT, so the type inputs stay aligned with their particles after an edit.
- Samples: JetClass rows are random within each class file (`jc_sample`). Evaluated and reference jets are disjoint. Top and QG use random rows (`_rows`) with disjoint references. No first-k subsampling.
- The observables (m, pT, mass window) are computed on the same edited tables the networks see.
- Working point: per-class median of the unedited own-class log-odds (50 % efficiency), frozen. Correct.
- No bug found that changes a reported number. Two remarks follow.
  - The "H_(m,pT)" prediction for Top and QG comes from MLP(m,pT), which is itself evaluated outside its training pT range under U. Its ρ depends strongly on λ on QG (3.21 at λ = 0.9/1.1, 1.44 at 0.98/1.02, 1.89 on in-range jets). The QG value of 3.21 is not a stable prediction.
  - The ParT_kin "step" at λ = 0.9 (Wqq −0.48) is also present under UB (fixed pT) and on jets that stay in range (fresh sample: U(0.9) −0.42 [−0.52, −0.32], UB(0.9) −0.38 [−0.49, −0.28]). It is a response to the lower W mass, not to leaving the pT range.

**Figure `h71_wjet_scale.png`.** Axes, labels and curves agree with the npz means (right panel, Wqq: ParT_full U(1.1) −1.09). The example jet in the left panel has pT = 547 GeV, so at λ = 0.9 it is at 492 GeV, below the 500 GeV training range. This does not change the panel's message (see the in-range result below).

## Procedure of the rerun

**Sample.** `h1_irc_panel.jc_sample(2000, seed 7192)`: 2,000 random rows from each class's first JetClass test file (H71 used seed 7102). 1,000 per class are evaluated (N = 10,000) and 1,000 per class train the reference MLP(log m, log pT) (rng 7202).

**Unedited accuracy on this sample.**

| model | accuracy (H71) |
|---|---|
| ParT_full | 0.8631 (0.8649) |
| ParT_kin | 0.7431 (0.7497) |
| PN_full | 0.8458 (0.8499) |
| PN_kin | 0.7017 (0.7086) |
| MLP(m,pT) | 0.3338 (0.3305) |

**Edits** (constituent four-momenta p_i = (E, px, py, pz); directions about the stored axis):
- U(λ): every p_i → λ p_i. Jet m and pT both scale by λ; m/pT and all angles stay fixed.
- UB(λ): U(λ) then a boost along the jet by 1/λ. Jet m → λ m at the original pT.
- **B(λ), new control:** a boost along the jet only, with |P| → λ|P|. Jet pT → λ pT at the original mass; opening angles shrink by about 1/λ. U equals B applied after UB exactly (the two boosts along the same axis cancel), so B isolates the pT part of U at fixed mass. For a linear response, Δs_U = Δs_UB + Δs_B.
- Ch / Nh / Rh / Rc at λ = 0.9 and 1.1, as in H71. Ch scales the charged particles (part_charge ≠ 0) and Nh the neutral ones. Rh scales a random subset whose pT share matches the charged share, and Rc its complement.
- λ ∈ {0.9, 0.95, 1.05, 1.1} for U, UB and B.

**New control for the training-range confound.** ρ is recomputed only on jets whose scaled pT stays inside the dataset's pT range at both λ of the pair. The ranges are JetClass 500–1000 GeV, Top 550–650 GeV and QG 500–550 GeV, using the jet pT recomputed from the constituents. This is done on the fresh JetClass sample and on H71's own npz files for JetClass, Top and QG; those files are read only. On Top and QG no jet stays in range at λ = 0.9/1.1. There the cut uses λ = 0.95/1.05 (Top, 40 % of jets kept) and 0.98/1.02 (QG, 58 % kept).

**Metrics.**
- Δs = change of the own-class log-odds l_k = log p_k − log(1 − p_k) of jets of true class k.
- ρ = Σ_k size_U,k / Σ_k size_UB,k, with size = (|mean Δs(a)| + |mean Δs(b)|)/2 for the λ pair (a, b). This is the response to a uniform energy scale divided by the response to the same mass change at fixed pT.
- β_B = the same ratio with B in the numerator: the response to a pT change at fixed mass, divided by the response to a mass change at fixed pT.
- Additivity residual = mean over jets of |Δs_U − Δs_UB − Δs_B|, at λ = 1.1.
- Asymmetry excess = (Δs_Ch − Δs_Nh) − (Δs_Rh − Δs_Rc) at λ = 1.1.

**Statistics.**
- 95 % bootstrap intervals over jets, with 1,000 resamples stratified by class for ρ and β_B.
- ρ_in − ρ_all is computed from two independent bootstraps (not paired, so the interval is wide). Its p-value is the two-sided bootstrap tail.
- Per class, a paired sign-flip test (2,000 flips, with the normal tail when the flips saturate).
- Holm correction within each family over 4 models × 10 classes (× λ ∈ {0.9, 1.1} where both are tested).

**Hypotheses.**
- H0 (range confound): ρ on jets that stay in range equals ρ on all jets. H1: the out-of-range jets drive the gap between U and UB, which would move ρ_in toward 1.
- H0: Δs_U = Δs_UB per class, on jets that stay in range.
- H0: Δs_B = 0 per class. H_mass-only predicts Δs_B = 0, and so does a reader of mass alone at fixed pT.
- H0: asymmetry excess = 0. The type-channel hypothesis predicts an excess in ParT_full/PN_full and not in the kin models. The kinematic hypothesis predicts an excess in both.

## Numbers

**ρ, fresh JetClass sample (seed 7192), λ = 0.9/1.1** (95 % CI). H71's values on seed 7102 are in parentheses.

| model | ρ all jets | ρ in-range jets (61 %) | ρ_in − ρ_all | β_B in-range | additivity residual / mean\|Δs_U\| at 1.1 |
|---|---|---|---|---|---|
| ParT_full | 0.616 [0.607, 0.626] (0.621) | 0.636 [0.624, 0.645] | +0.020 [+0.003, +0.032] | 0.22 [0.21, 0.24] | 0.36 / 0.79 |
| PN_full | 0.680 [0.667, 0.696] (0.669) | 0.693 [0.677, 0.711] | +0.014 [−0.010, +0.036] | 0.23 [0.21, 0.24] | 0.32 / 0.76 |
| ParT_kin | 0.915 [0.891, 0.946] (0.921) | 0.954 [0.918, 1.002] | +0.039 [−0.009, +0.090] | 0.22 [0.20, 0.24] | 0.30 / 0.85 |
| PN_kin | 0.874 [0.865, 0.889] (0.883) | 0.909 [0.891, 0.930] | +0.035 [+0.011, +0.058] | 0.28 [0.26, 0.29] | 0.14 / 0.69 |
| MLP(m,pT) | 0.843 [0.832, 0.853] (0.772) | 0.815 [0.801, 0.831] | −0.028 | 0.28 | 0.04 / 0.40 |

At λ = 0.95/1.05 (80 % of jets in range), ρ_in − ρ_all is at most +0.03, and every interval includes 0.

**ρ on H71's own npz, all jets → jets that stay in range.**

| dataset (λ pair) | model | ρ all | ρ in-range [95 % CI] | difference [95 % CI] |
|---|---|---|---|---|
| JetClass (0.9/1.1) | ParT_full / PN_full / ParT_kin / PN_kin | 0.621 / 0.669 / 0.921 / 0.883 | 0.643 / 0.682 / 0.941 / 0.904 | +0.01 to +0.02 |
| Top (0.95/1.05, 40 % kept) | ParT_kin | 0.857 | 0.846 [0.814, 0.879] | −0.011 [−0.049, +0.027] |
| | PN_kin | 0.597 | 0.602 [0.555, 0.647] | +0.006 [−0.049, +0.060] |
| | LorentzNet | 0.784 | 0.794 [0.746, 0.841] | +0.011 [−0.049, +0.068] |
| QG (0.98/1.02, 58 % kept) | ParT_kin | 0.209 | 0.205 [0.181, 0.235] | −0.004 [−0.038, +0.031] |
| | PN_kin | 0.375 | 0.367 [0.353, 0.381] | −0.008 [−0.026, +0.009] |
| | ParT_kinpid / PN_kinpid | 0.394 / 0.392 | 0.384 / 0.396 | ±0.01 |
| | LorentzNet | 1.189 | 1.179 [1.147, 1.219] | −0.010 [−0.052, +0.035] |
| | MLP(m,pT) (reference) | 1.443 | 1.893 [1.729, 2.051] | +0.45 [+0.22, +0.67] |

The QG ρ values do not depend on the λ pair: ParT_kin gives 0.211, 0.212 and 0.209 at 0.9/1.1, 0.95/1.05 and 0.98/1.02.

**Per-class means on in-range jets, fresh sample, λ = 0.9 / 1.1** (95 % CI). Wqq, n = 618:
- ParT_full: U +0.17 / −1.10 [−1.20, −1.01]; UB +0.15 / −1.44 [−1.56, −1.33]; B −0.32 [−0.35, −0.28] / +0.13 [+0.11, +0.16].
- PN_full: U +0.25 / −1.05; UB +0.26 / −1.33; B −0.27 / +0.15.
- ParT_kin: U −0.42 / −1.06; UB −0.38 / −1.09; B −0.05 / +0.08.

Hbb, ParT_full: U −0.87 / +0.22; UB −1.34 / +0.10; B +0.16 / −0.20.

**Holm (fresh sample).**

| family | ParT_full | PN_full | ParT_kin | PN_kin |
|---|---|---|---|---|
| H0: Δs_U = Δs_UB, in-range jets | 18 / 20 | 18 / 20 | 10 / 20 | 15 / 20 |
| H0: Δs_B = 0, in-range jets | 14 / 20 | 15 / 20 | 13 / 20 | 18 / 20 |
| H0: asymmetry excess = 0 at λ = 1.1 (family of 40) | 8 / 10 | 7 / 10 | 2 / 10 | 6 / 10 |

H71 found 7 / 10, 7 / 10, 1 / 10 and 3 / 10 in a larger family of 138 tests.

**Asymmetry excess at λ = 1.1, fresh sample** (95 % CI).

| class | ParT_full | PN_full | ParT_kin | PN_kin |
|---|---|---|---|---|
| Wqq | +0.168 [+0.139, +0.198] | +0.139 [+0.114, +0.162] | +0.021 [−0.039, +0.084] | +0.008 [−0.012, +0.027] |
| Tbl | +0.122 [+0.097, +0.146] | +0.134 [+0.096, +0.175] | +0.005 [−0.057, +0.066] | +0.125 [+0.102, +0.151] |
| QCD | +0.081 [+0.066, +0.098] | +0.069 | +0.080 [+0.045, +0.117] | +0.049 [+0.032, +0.064] |
| Hqql | +0.060 | +0.076 | −0.100 | +0.089 [+0.057, +0.117] |
| Hbb | −0.043 | −0.027 | +0.039 | +0.054 [+0.037, +0.071] |

## Reading

- **Reproduced.** On a fresh JetClass sample, ρ = 0.62 / 0.68 (ParT_full / PN_full) and 0.92 / 0.87 (kin) matches H71 within 0.01. H_shape (ρ = 0) is excluded. U ≠ UB holds in 18 / 20 class × λ tests for each full model.
- **Training-range confound.** Restricting to jets that stay inside the training pT range moves ρ by +0.01 to +0.04 on JetClass, and by at most 0.011 on Top and QG at the λ pairs where in-range jets exist. Every interval on Top and QG includes 0. The gap between U and UB is therefore not produced by jets leaving the training pT range. H71's main open point is resolved. Only the MLP(m,pT) reference is range-sensitive (QG +0.45).
- **"Mainly through the jet mass."** This is only partly supported.
  - A pT change at fixed mass (B), with opening angles shrinking by 1/λ, also moves the scores. β_B = 0.22–0.28 of the response to a mass change at fixed pT, and Δs_B ≠ 0 in 13–18 / 20 tests per model.
  - The responses are not additive at λ = 1.1: the mean |Δs_U − Δs_UB − Δs_B| is 0.36 against a mean |Δs_U| of 0.79 for ParT_full.
  - ρ < 1 for the full models therefore reflects a joint, non-linear dependence on mass and on pT or angular size. It is not a mass reading diluted by a separate additive pT term.
  - The statement that holds is narrower: a uniform scale changes the scores by 0.6–0.9 of the change from the same mass shift at fixed pT.
- **Charged vs neutral.**
  - The full-model excess replicates: Wqq +0.17 for ParT_full and +0.14 for PN_full.
  - The contrast with the kin models holds only for Wqq (and Tbl for ParT). In QCD the kin models show the same excess as the full models: ParT_kin +0.080 against ParT_full +0.081. In Tbl, PN_kin shows +0.125 against PN_full +0.134.
  - PN_kin reaches 6 / 10 classes (H71: 3 / 10).
  - The random subset Rh matches the energy share of the charged particles but not their kinematics (pT spectrum, angular spread). The excess is therefore largely kinematic. The particle-type channel is supported only where the full − kin contrast is large, which is Wqq.
- **Not rechecked:** hls4ml JEDI-linear (ρ = 1.12) and the smearing results; the code paths are the same and no bug was found in them.

## Corrected reading

Uniform scale U: as H71 states, every network with absolute-momentum inputs responds, shape-only references give 0, and ρ = 0.62–0.67 (JetClass full models), 0.87–0.92 (JetClass kin), 0.60–0.86 (Top) and 0.21–0.40 (QG ParT/PN). These values do not change by more than 0.04 when only jets that stay inside the training pT range are used, so the training-range confound is excluded. A pT change at fixed mass (B) gives 0.22–0.28 of the fixed-pT mass response, and the responses are not additive at λ = 1.1. The reduction of U relative to UB is therefore a joint, non-linear dependence on mass and on pT or angular size, not "mainly mass". Charged-only versus neutral-only scaling, beyond a random subset of equal energy share, is present in the full JetClass models (7–8 / 10 classes). On a fresh sample the kin models show it too (ParT_kin 2 / 10, PN_kin 6 / 10, with equal size in QCD and in PN Tbl). Only the Wqq contrast (+0.17 full against +0.02 kin) points to the particle-type inputs; the rest is consistent with kinematic differences between charged and neutral particles.

Proposed confidence: 0.65 for the uniform-scale part (higher than H71's 0.5, because the range confound is excluded and the result replicates). The charged/neutral type-channel statement is limited to Wqq.
