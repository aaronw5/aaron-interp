# H5 verify: quark–gluon taggers and constituent counting

Script `interp/h5_verify.py` (imports the edits, models and statistics from `interp/h5_qg_counting.py`). Cluster job
`anrunw-ai-d-h5-verify` (GPU, GTX 1080 Ti, 1050 s, exit 0). Results `results/physics/h5_verify.json`,
`results/physics/h5_verify_perjet.npz`, log `logs/d-h5-verify.log`.

## Conclusion

On a new set of 10,000 test jets the H5 numbers come out the same. The networks move toward "gluon" when the hardest particle
is split into two collinear pieces. They move just as far when the particle count is held fixed, and they barely move when a
particle is removed. A new check adds a soft particle at a realistic position; a classifier cannot tell these jets from real
ones. That edit leaves ParT, ParticleNet and LorentzNet within 0.03 in logit, while a count-only likelihood ratio moves 0.1.
So these networks do not count soft particles, and this result does not depend on unrealistic inputs. It is still open which
property of the hard-particle energy sharing they respond to. Moving energy between particles at fixed count also changes the
IRC-safe jet shapes (an IRC-safe network responds to that edit as strongly as the taggers), so that edit cannot isolate p_T^D.

## Script review

- Particle order: `data_top.convert_files` sorts constituents by pT (line 59), so rank 0 is the hardest and `a[:-1]` removes the
  softest. Padding is excluded (`pt > 0`). Sampling is random per class (`rng.choice`), not first-k.
- Coordinates: Δη uses the sign(η_jet) flip used in the conversion; the rebuild check passes (max |Δ| 7.5e-7). Weaver inputs are rebuilt
  from the edited lists with the jet axis recomputed, so observables and model inputs come from the same edited jets.
- Momentum: the split keeps the pT-weighted centroid and the parent's pT; the jet four-momentum is kept up to the daughter
  masses (jet-mass change 1e-4 GeV). `merge_split` and `merge_soft` place the merged particle at the pT-weighted position
  as a massless particle, which changes the jet mass by 0.32 GeV on average. The report states this.
- Working points: thresholds frozen on unedited jets at ε_q = 0.3 / 0.5, quark = signal. Correct.
- "IRC-safe shapes fixed": this holds for girth and C1^(β=1,2). At ΔR = 5e-4, log C1^(β=0.5) shifts enough that the classical
  shapes-only regression `lr_shape` responds +0.063 to the hardest split (H5 run json, `edits.split_r0_z0.5.models.lr_shape`).
  The report does not mention this. An IRC-safe scorer can therefore move at this finite ΔR when it weights small angles
  strongly. The EFN (+0.0002) does not.
- Minor: the per-epoch scan averages the `ksplit_1` Δlogit over all jets, including the 2 jets left unedited (dilution 2e-4).
  Classical regressions are fitted on `kin3`, which keeps the 128 hardest particles, so N is capped at 128 in training. Neither changes a number in the reading.
- Figure `h5_counting.png`: the panels and bars match the json. The left panel does not show the per-k CI bars for the
  networks (they are smaller than the markers).

No bug that changes a reported number was found.

## Procedure

Sample `data_qg.sample(5000 per class, seed 9137)` (10,000 Pythia test jets, quark fraction 0.500). Models: ParT kin and kinpid
(final, best epochs 14/17 of 20); PN kin (interim, best epoch 17 of 18 written) and PN kinpid (interim, 14 of 17); LorentzNet
(released); the H5 torch PFN and EFN; ParT kin with permuted weights; count-only LR(N) (1.6 M training jets). Metric:
Δlogit = Δ[log p_g − log p_q], mean over jets with a 95 % bootstrap CI (2,000 resamples). H0: mean Δlogit = 0 (paired t-test). For
paired edit-vs-control comparisons, H0: equal means (paired t-test and sign-flip permutation, 2,000 flips). Calls are taken at the
frozen ε_q = 0.5 threshold, with an exact McNemar test. Holm correction over 162 tests: 153 have p_Holm < 0.05.

Rerun edits: split hardest z = 0.5 (ΔR = 5e-4); merge-2-softest + split-hardest (ΔN = 0); remove softest; merge two softest;
split softest; k = 1..3 random splits.

New controls:
1. `split_r0_dr0.005`, `split_r0_dr0.02`: split the hardest particle with the daughters 5e-3 or 0.02 apart. This tests whether
   the response depends on the 5e-4 pair, i.e. on off-manifold pair features.
2. `add_soft` (ΔN = +1), `add_soft3` (ΔN = +3): add a particle with the jet's softest pT at the position of a random existing
   particle, offset by 0.05 in a random direction. This creates no collinear pair and is the IR counterpart of `rmsoft`.
3. `ptd_match` (ΔN = 0, no new particle): scale the hardest particle's pT by f and every other particle by g, keeping Σ pT. f is
   solved per jet so that Σ pT² equals its value after the z = 0.5 hardest split. A solution exists for 2,829 jets.
4. `add_med_scaled`: add a particle with the median pT (same placement), then rescale all pT to keep Σ pT.

## Numbers

Unedited AUC: ParT kin 0.902, ParT kinpid 0.912, PN kin 0.902, PN kinpid 0.906, LorentzNet 0.917, PFN 0.883, EFN 0.866,
LR(N) 0.841, permuted 0.560.

**Reproduction** (H5 seed 701 → this sample seed 9137, mean Δlogit):

| edit | ParT kin | ParT kinpid | PN kin | PN kinpid | LorentzNet | PFN | EFN | LR(N) |
|---|---|---|---|---|---|---|---|---|
| split hardest | 0.160 → 0.161 [0.157, 0.165] | 0.142 → 0.135 | 0.099 → 0.083 | 0.204 → 0.204 | 0.199 → 0.191 | 0.165 → 0.167 | 0 → 0 | 0.098 → 0.097 |
| ΔN = 0 control | 0.177 → 0.179 | 0.158 → 0.151 | 0.085 → 0.064 | 0.207 → 0.209 | 0.192 → 0.186 | 0.084 → 0.082 | −0.001 → −0.004 | 0 → 0 |
| remove softest | 0.001 → 0.000 | 0.003 → 0.003 | −0.005 → 0.003 | 0.002 → 0.003 | −0.011 → −0.010 | −0.089 → −0.091 | −0.013 → −0.013 | −0.099 → −0.100 |
| split softest | −0.039 → −0.039 | −0.048 → −0.049 | −0.038 → −0.054 | −0.054 → −0.055 | −0.011 → −0.011 | 0.075 → 0.077 | 0 → 0 | 0.098 → 0.097 |
| random-split slope | 0.030 → 0.033 [0.031, 0.035] | 0.061 → 0.062 | 0.031 → 0.024 | 0.057 → 0.058 | 0.061 → 0.062 | 0.056 → 0.055 | 0 → 0 | 0.0945 → 0.0963 |

The PN kin changes (0.099 → 0.083, 0.031 → 0.024) come with the later checkpoint (best epoch 11 → 17). Restore fraction
1 − Δctrl/Δsplit on 9,999 common jets: −0.11, −0.12, +0.23, −0.03, +0.02 for the five networks (PN kin was +0.14 in H5). The
split − control difference is ≤ 0.019 in magnitude for the five networks, against 0.085 for the PFN and 0.097 for LR(N).

**LR(N) vs Poisson.** Training-jet multiplicity: quark mean 33.4, sd 13.3 (Poisson 5.8); gluon mean 53.1, sd 15.7 (Poisson 7.3).
The sample mean of lrN[N+1] − lrN[N] is 0.097, while the Poisson value is log 1.589 = 0.463. The data distributions are 2.2–2.3 times wider
than Poisson, which accounts for the factor of 4.8. The network slopes are 0.024–0.062, i.e. 7–19 times below 0.463 (H5 stated 8–15; PN kin
at epoch 17 gives 19).

**New control 1, split separation** (split hardest, mean Δlogit at ΔR = 5e-4 / 5e-3 / 0.02):
ParT kin 0.161 / 0.163 / 0.443; ParT kinpid 0.135 / 0.147 / 0.327; PN kin 0.083 / 0.124 / 0.397; PN kinpid 0.204 / 0.212 / 0.260;
LorentzNet 0.191 / 0.189 / 0.399; PFN 0.167 / 0.173 / 0.223; EFN 0.000 / 0.021 / 0.277.
From 5e-4 to 5e-3 the network responses change by −0.002 to +0.041 and the EFN by +0.021. At 0.02 the split changes girth and
C1 (EFN +0.28), so that point is not a collinear edit. C2ST AUC: 1.000 at both 5e-4 and 0.02.

**New control 2, add a soft particle** (on-manifold: C2ST AUC 0.502 [0.485, 0.514]; unedited vs unedited 0.5):

| model | add 1 soft (ΔN +1) [CI] | add 3 soft (ΔN +3) | remove softest (ΔN −1) | McNemar q→g/g→q, add 1 (p) |
|---|---|---|---|---|
| ParT kin | −0.004 [−0.005, −0.003] | −0.003 | 0.000 | 21/18 (0.75) |
| ParT kinpid | −0.016 [−0.018, −0.015] | −0.034 | +0.003 | 31/18 (0.085) |
| PN kin | −0.007 [−0.009, −0.006] | −0.016 | +0.003 | 30/30 (1.0) |
| PN kinpid | −0.024 [−0.026, −0.022] | −0.071 | +0.003 | 52/46 (0.61) |
| LorentzNet | −0.012 [−0.014, −0.010] | −0.044 | −0.010 | 54/48 (0.62) |
| PFN | +0.057 [0.055, 0.058] | +0.169 | −0.091 | 3/112 (1e-29) |
| EFN | +0.005 | +0.014 | −0.013 | 0/11 |
| permuted ParT kin | −0.028 | −0.071 | +0.015 | 907/39 |
| LR(N) | +0.097 [0.096, 0.098] | +0.290 | −0.100 | 0/209 (2e-63) |

Hypotheses for this edit: counting at the data LR(N) rate predicts +0.097 per particle; Poisson counting predicts +0.46; no use
of soft multiplicity predicts ≈ 0 (within the permuted-weight null scale of ±0.03). Both counting predictions are excluded for the five
networks: their response has the wrong sign, and the CIs exclude +0.097. The PFN follows the counting prediction at 0.59 of LR(N).

**New control 3, p_T^D matched without a split** (2,829 jets): Δp_T^D −0.072 (vs −0.045 average for the split over all jets),
but girth +0.0058, C1^(1) +0.0042, jet mass +3.0 GeV, and the EFN responds +0.497 [0.475, 0.519]. The networks give +0.38 to +0.59,
compared with +0.15 to +0.27 for the hardest split on the same jets. Because the EFN response is as large as the network responses, this edit
does not isolate p_T^D, and no p_T^D-only control without a collinear split was found: at fixed positions, any reallocation of
energy changes the IRC-safe shapes. `add_med_scaled` has the same problem (EFN +0.122, networks +0.12 to +0.14).

## Assessment of the H5 reading

- Reproduced on a fresh sample: rank dependence of the split response, ΔN = 0 control ≈ split response, remove-softest ≈ 0,
  EFN 0, permuted ≈ 0, LR(N) slope 0.097 vs Poisson 0.463, factor 7–19 for the networks.
- Strengthened: the "does not count soft particles" part now has an on-manifold test (add_soft, C2ST 0.50), so for this part
  the off-manifold caveat no longer applies.
- Not supported as written in the thesis note: "the response follows the energy sharing of hard particles (p_T^D-like)". The
  edits show a dependence on which particle is split (hard vs soft) and on particle type. They do not separate p_T^D from other
  non-IRC-safe functions of the hard-particle momentum fractions, or from the near-collinear pair itself. The H5 report's own
  hypothesis (d) paragraph says this, but the thesis note and suggested rewording state the p_T^D attribution.
- "IRC-safe shapes fixed" is approximate at ΔR = 5e-4 for β = 0.5 (`lr_shape` +0.063).
- Range correction: network slope / Poisson 0.05–0.13 (PN kin at epoch 17: 0.051).

## Corrected conclusion

The trained quark–gluon taggers (ParT, ParticleNet, LorentzNet) do not count particles. A soft particle added at a realistic
position moves them by −0.024 to −0.004 in logit, while a count-only likelihood ratio moves +0.097; removing the softest
particle also leaves them almost unchanged. Splitting the hardest particle into two collinear pieces moves them toward gluon by
0.08–0.20, and they move as much when the count is held fixed. Splitting the softest particle moves them slightly toward quark.
The hard-split response is a response to inputs a classifier separates from real jets, and these edits do not identify which
function of the hard-particle energy sharing the networks use. The Poisson "Casimir" estimate (0.46 per particle) is about 5 times the slope
of the measured count likelihood ratio, because the measured multiplicity distributions are about twice as wide as Poisson.
