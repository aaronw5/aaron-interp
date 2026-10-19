# H5: Quark–gluon taggers and constituent counting (collinear split, Casimir-meets-Poisson), claim D7

Script `interp/h5_qg_counting.py` (steps `run`, `herwig`, `hw`, `fig`). Results `results/physics/h5_run.json`,
`results/physics/h5_perjet.npz` (per-jet Δlogit for every edit × model), `results/physics/h5_herwig.json`, figure
`results/physics/h5_counting.png`. Logs `logs/d-h5-run3.log` (main run, RTX 2080 Ti, 1832 s), `logs/d-h5-hw3.log` (Herwig, final
checkpoints), `logs/d-h5-hwrun.log` (Herwig, earlier interim checkpoints), `logs/d-h5-hw2.log` (download), `logs/d-h5-run2.log`
(crashed run, see Issues), `logs/d-h5-fig3.log`. Cluster jobs: anrunw-ai-d-h5-run3 (GPU), anrunw-ai-d-h5-hw3 (GPU), anrunw-ai-d-h5-hw2 (CPU download),
anrunw-ai-d-h5-fig3 (CPU).

## Conclusion

The trained quark–gluon taggers (ParT, ParticleNet, LorentzNet) move toward "gluon" when one particle is split into two
collinear pieces. The size of the move depends on which particle is split, not on the extra particle. Merging the two softest particles while
splitting the hardest leaves the number of particles unchanged, yet it moves the networks as much as the split alone. Removing the softest particle leaves them
almost unchanged, while the count-only likelihood ratio moves by −0.10 in logit. A pure particle count under the Poisson
(Casimir) assumption predicts +0.46 per added particle; the networks move 0.03–0.06 per random split, and even the count-only
likelihood ratio built from the data moves only 0.095, because the measured multiplicity distributions are wider than Poisson.
The split edits leave a pair of particles closer than any typical real pair, so a classifier tells edited from real jets
(two-sample AUC ≈ 1). The responses are therefore measured off the training distribution, and the PN models are interim checkpoints.

Terms: "logit" = log p(gluon) − log p(quark). "IRC-safe" observables are those unchanged by a collinear split or by adding an
infinitely soft particle. The energy-flow network (EFN) is IRC-safe by construction.

## Procedure

**Sample.** `data_qg.sample(5000 per class, seed 701)`, test split of the Komiske–Metodiev–Thaler Pythia set (random rows),
10,000 jets, quark fraction 0.500. Rebuild check against the stored arrays: max |Δ(Δη)| 3.9e-7, max |Δ(Δφ)| 7.5e-7, max relative
jet-pT difference 2.5e-7.

**Models and unedited performance on this sample** (quark = signal; R50/R30 = 1/ε_g at ε_q 0.5 / 0.3; AUC with 95 % bootstrap CI over jets):

| model | checkpoint | AUC [95 % CI] | acc | R30 | R50 | reference |
|---|---|---|---|---|---|---|
| ParT kin | final, best epoch 14 of 20 | 0.9044 [0.8988, 0.9101] | 0.830 | 104 | 39.1 | own full test AUC 0.9016 |
| ParT kinpid | final, best epoch 17 of 20 | 0.9138 [0.9086, 0.9186] | 0.844 | 116 | 45.0 | own full test AUC 0.9121; published ParT 0.9121 |
| PN kin | interim, best epoch 11 of 15 written | 0.9026 [0.8966, 0.9078] | 0.832 | 96 | 39.4 | – |
| PN kinpid | interim, best epoch 14 of 15 written | 0.9085 [0.9027, 0.9136] | 0.834 | 116 | 42.0 | published PN 0.9116 |
| LorentzNet | released weights | 0.9190 [0.9141, 0.9239] | 0.846 | 119 | 44.2 | released test AUC 0.9161; published 0.9156 |
| PFN (torch, trained in H5, 400k jets, 15 epochs) | stand-in for H2 PFN | 0.8881 [0.8820, 0.8939] | 0.815 | 78 | 29.1 | – |
| EFN (torch, trained in H5; IRC-safe) | stand-in for H2 EFN | 0.8716 [0.8661, 0.8777] | 0.796 | 85 | 30.1 | – |
| ParT kin, weights permuted (null) | – | 0.5450 | 0.499 | 4.5 | 2.3 | – |
| classical LR(N, log girth, log C1^β) | logistic, 100k train jets | 0.8634 | 0.797 | 51 | 20.8 | – |
| classical LR(shapes) | same without N | 0.8519 | 0.788 | 52 | 22.6 | – |
| count-only LR(N) | histogram ratio, 1.6 M train jets | 0.8461 [0.8391, 0.8537] | 0.775 | 53 | 16.2 | H2: 0.840 |

The H2 EFN/PFN QG trainings (energyflow) have not produced QG checkpoints (d-h2-pfn failed; models/qg/efn empty), so torch
EFN (Σ z_i Φ(Δη_i, Δφ_i)) and PFN (Σ Φ(z_i, Δη_i, Δφ_i)) were trained in this job. PN kin/kinpid were still training at run
time (epoch 15 of 20); the per-epoch scan below covers that.

**Edits** (pT-ordered particle lists; daughters at ΔR = 5·10⁻⁴ from each other about the parent's position with the
pT-weighted centroid kept; jet four-momentum kept up to the massless-daughter approximation; edits that cannot apply leave the jet unchanged and the jet is excluded from that edit's statistics):
1. `split_r{0,1,5,soft}_z{0.5,0.2}`: split the particle of pT rank r with momentum fractions z, 1 − z (ΔN = +1). Kinpid
   models: both daughters keep the parent's PID.
2. `ksplit_k`, k = 1..5: k random particles each split at z = 0.5 (ΔN = +k; jets with N ≥ 5, 9,998 jets).
3. `pid_ch_copy`: split the hardest charged hadron into two charged hadrons; `pid_ch_photon`: same, second daughter a photon
   (N_ch unchanged); `pid_ph_copy`: split the hardest photon.
4. `rmsoft`: remove the softest particle (ΔN = −1). `merge_soft`: merge the two softest (ΔN = −1).
5. Count-fixed control `merge_split`: merge the two softest particles and split the hardest at z = 0.5 (ΔN = 0; compared
   with `split_r0_z0.5` on the 9,998 jets where both apply).
6. Nulls: EFN (zero response expected), ParT kin with permuted weights.

Shape changes under the splits (mean |Δ| over jets, `split_r0_z0.5`): girth 6.3e-7, C1^(β=1) 8.8e-6, C1^(β=2) 1.4e-8, jet
mass 1.0e-4 GeV; pT^D changes by 0.045 (not IRC-safe). Random split (`ksplit_1`): girth 5.0e-8, pT^D 2.8e-3. The count-fixed control
changes pT^D by 0.045 (the same as `split_r0_z0.5`) and C1^(β=2) by 1.3e-4.

**Predictions.** (a) Casimir-meets-Poisson: N_q ~ Poisson(λ_q), N_g ~ Poisson(λ_g), λ_g/λ_q = 1.589 (H2) → Δlog LR = log 1.589
= 0.463 per added particle, linear in k, the same for every rank, −0.46 for removal, 0 under the count-fixed control.
(b) Empirical counting: the slope of the data log LR(N) (0.095 per particle here). (c) IRC-safe shapes only: 0 for every
split (as the EFN). (d) Hard-particle energy sharing (pT^D-like): response largest for the hardest particle, falling with
rank, the same under the count-fixed control, ≈ 0 for adding/removing the softest.

**Statistics.** Mean Δlogit with 95 % bootstrap CI (2,000 resamples of jets). H0: mean Δlogit = 0 (paired t-test, and
sign-flip permutation, 5,000 flips); for the control, H0: Δlogit(split hardest) = Δlogit(count-fixed control) (paired t-test on
per-jet differences). Holm correction over 220 tests (11 models × 19 edits + 11 control tests); the t-test p is used for Holm because
the permutation p floor (2·10⁻⁴) is above the Holm threshold for 220 tests. Call changes at the ε_q = 0.5 threshold frozen on unedited
jets, with exact McNemar. Linearity: least-squares slope of mean Δlogit vs k (k = 1..5) and quadratic coefficient, bootstrap CI.
Off-manifold: gradient-boosted classifier two-sample test (C2ST), edited half vs unedited half, held-out AUC.

## Numbers

**Random collinear splits, slope per added particle** (k = 1..5; ratio to 0.463 and to the empirical LR(N) slope 0.0945):

| model | slope [95 % CI] | quadratic coeff [CI] | / Poisson 0.463 | / LR(N) 0.0945 |
|---|---|---|---|---|
| ParT kin | 0.0302 [0.0293, 0.0312] | −0.0004 [−0.0011, 0.0003] | 0.065 | 0.32 |
| ParT kinpid | 0.0612 [0.0591, 0.0633] | −0.0008 [−0.0022, 0.0007] | 0.132 | 0.65 |
| PN kin | 0.0310 [0.0301, 0.0320] | −0.0001 [−0.0007, 0.0005] | 0.067 | 0.33 |
| PN kinpid | 0.0573 [0.0555, 0.0590] | −0.0003 [−0.0016, 0.0009] | 0.124 | 0.61 |
| LorentzNet | 0.0606 [0.0586, 0.0629] | −0.0015 [−0.0029, 0.0000] | 0.131 | 0.64 |
| PFN | 0.0557 [0.0547, 0.0568] | +0.0008 [0.0001, 0.0015] | 0.120 | 0.59 |
| EFN | 1.9e-5 | 0 | 0.000 | 0.000 |
| ParT kin permuted | −0.0002 [−0.0004, 0.0000] | – | – | – |
| classical LR(N, shapes) | 0.0552 | 0.0000 | 0.119 | 0.58 |
| classical LR(shapes) | 0.0032 | 0.0000 | 0.007 | 0.03 |
| count-only LR(N) | 0.0945 [0.0935, 0.0955] | −0.0007 | 0.204 | 1 |

**Split by rank (z = 0.5), remove softest, count-fixed control**, mean Δlogit [95 % CI]:

| model | split hardest (ΔN +1) | split rank 5 | split softest (ΔN +1) | remove softest (ΔN −1) | merge 2 softest + split hardest (ΔN 0) | split hardest − control [CI], Holm p | restore fraction [CI] |
|---|---|---|---|---|---|---|---|
| ParT kin | +0.160 [0.156, 0.164] | +0.050 | −0.039 [−0.040, −0.038] | +0.001 [−0.000, 0.002] | +0.177 [0.172, 0.181] | −0.017 [−0.018, −0.015], 4e-88 | −0.10 [−0.11, −0.09] |
| ParT kinpid | +0.142 [0.135, 0.149] | +0.088 | −0.048 | +0.003 [0.001, 0.004] | +0.158 [0.151, 0.165] | −0.016 [−0.018, −0.014], 1e-54 | −0.11 [−0.13, −0.10] |
| PN kin | +0.099 [0.096, 0.103] | +0.042 | −0.038 | −0.005 [−0.006, −0.004] | +0.085 [0.079, 0.091] | +0.014 [0.010, 0.019], 2e-7 | +0.14 [0.09, 0.19] |
| PN kinpid | +0.204 [0.199, 0.210] | +0.105 | −0.054 | +0.002 [−0.000, 0.003] | +0.207 [0.201, 0.213] | −0.003 [−0.006, 0.000], 0.70 | −0.02 [−0.03, −0.00] |
| LorentzNet | +0.199 [0.191, 0.206] | +0.101 | −0.011 [−0.012, −0.009] | −0.011 [−0.013, −0.009] | +0.192 [0.184, 0.200] | +0.007 [0.004, 0.009], 7e-5 | +0.03 [0.02, 0.05] |
| PFN | +0.165 [0.162, 0.168] | +0.062 | +0.075 | −0.089 [−0.091, −0.087] | +0.084 [0.080, 0.088] | +0.081 [0.079, 0.084], 0 | +0.49 [0.47, 0.51] |
| EFN | +0.0002 | 0.000 | 0.000 | −0.013 [−0.013, −0.012] | −0.001 [−0.003, 0.000] | +0.002, 0.36 | – |
| classical LR(N, shapes) | +0.093 | +0.054 | +0.062 | −0.059 | +0.030 | +0.054, 0 | +0.64 [0.63, 0.65] |
| count-only LR(N) | +0.098 | +0.098 | +0.098 | −0.099 [−0.100, −0.097] | 0 | +0.098, 0 | 1 |
| ParT kin permuted | −0.003 | +0.004 | −0.020 | +0.015 | +0.011 | −0.014 | – |

Restore fraction = 1 − Δ(control)/Δ(split hardest): 1 if the response is carried by ΔN, 0 if the control reproduces the
response. z = 0.2 splits give the same ordering (hardest +0.09 to +0.21, softest −0.010 to −0.071 for the five networks).

**Call changes and working points** (`split_r0_z0.5`, threshold frozen at ε_q = 0.5 on unedited jets): calls changed 1.8 %
(PN kin) to 3.7 % (LorentzNet), McNemar quark→gluon vs gluon→quark 248/9 (ParT kin, p = 1e-61), 275/77 (ParT kinpid, 3e-27),
166/11 (PN kin), 330/34 (PN kinpid), 309/65 (LorentzNet, 4e-39), 221/0 (LR(N)). Δ(1/ε_g) at ε_q 0.3: +15 (PN kin) to +56
(PN kinpid); at ε_q 0.5: +4.1 to +10.6. Count-fixed control: 2.1–4.1 % calls changed, Δ(1/ε_g) at ε_q 0.5 +5.7 to +10.6.
`rmsoft`: 0.33–1.14 % changed for the networks, McNemar p 0.05–1.0 (LR(N): 2.16 %, 0/216, p = 2e-65).

**Particle type (kinpid models)**, hardest charged hadron split, second daughter charged vs photon (same kinematics, so kin
models are identical by construction: ParT kin +0.135, PN kin +0.087 in both):

| model | ch → ch + ch | ch → ch + γ | difference |
|---|---|---|---|
| ParT kinpid | +0.108 | +0.023 [0.020, 0.027] | 0.085 |
| PN kinpid | +0.187 | +0.048 [0.046, 0.052] | 0.139 |
| LorentzNet | +0.190 | +0.067 [0.063, 0.070] | 0.123 |

**Training-stage spread** (`ksplit_1` and `rmsoft` at every written epoch, same 10,000 jets): ParT kin k = 1 response 0.020–0.034
over 20 epochs (AUC 0.896–0.904); ParT kinpid 0.054–0.078 (AUC 0.905–0.914); PN kin 0.013–0.039 (16 epochs); PN kinpid
0.049–0.069. `rmsoft` stays within −0.019 to +0.022 at every epoch of every model (LR(N): −0.099). The kin vs kinpid factor of about 2 in the
random-split response holds at every epoch.

**Off-manifold (C2ST AUC, held-out, [CI])**: unedited vs unedited 0.515 [0.499, 0.531]. split hardest 1.000; split softest
1.000 (angle features only 0.9997; without angle features 0.669); ksplit_1 0.9999 (without angles 0.935); ksplit_5 1.000;
merge_split 1.000; rmsoft 0.706 [0.692, 0.719]. In unedited jets 4.95 % have a particle pair with ΔR < 5·10⁻⁴ and 17.2 % with
ΔR < 10⁻³ (median minimum pair ΔR 0.0020, 2,000 jets).

**Herwig 7.1 (Zenodo 3066475, energyflow `herwig`, files 0–1, 200,000 jets downloaded into data/qg/herwig; 5,000 per class,
seed 701)**, same models (PN at best epoch 16 of 17 / 14 of 16 at that time):

| model | Pythia AUC | Herwig AUC [CI] | Herwig R50 | ksplit_1 Pythia → Herwig |
|---|---|---|---|---|
| ParT kin | 0.9044 | 0.8111 [0.8024, 0.8198] | 11.4 | +0.031 → +0.028 |
| ParT kinpid | 0.9138 | 0.8070 [0.7985, 0.8148] | 12.2 | +0.063 → +0.060 |
| PN kin | 0.9042 | 0.8098 [0.8014, 0.8183] | 11.5 | +0.023 → +0.024 |
| PN kinpid | 0.9085 | 0.8092 [0.8012, 0.8180] | 12.2 | +0.055 → +0.053 |
| LorentzNet | 0.9190 | 0.7992 [0.7904, 0.8073] | 10.6 | +0.062 → +0.058 |
| PFN | 0.8881 | 0.8094 [0.8005, 0.8182] | 12.7 | +0.052 → +0.050 |
| EFN | 0.8716 | 0.7966 [0.7875, 0.8050] | 11.4 | 0 → 0 |
| LR(N) | 0.8461 | 0.7916 [0.7836, 0.8011] | 9.3 | +0.098 → +0.103 |

⟨N⟩: Pythia q 33.3, g 53.6; Herwig q 33.3, g 48.7. Mean network logit on gluon jets drops from +1.4–1.8 (Pythia) to
+0.16–0.41 (Herwig); on quark jets it is unchanged (−1.7 → −1.7).

## Reading

- Hypothesis (a), Casimir-meets-Poisson counting at 0.46 per particle: every network's random-split slope is 0.07–0.13 of it, and
  the count-only LR(N) built from the same data has slope 0.095 (0.20 of it). The Poisson model predicts a log-LR slope 5 times
  the one in the data; the measured N distributions are wider than Poisson. Excluded for all models and for the data LR.
- Hypothesis (b), empirical counting: random splits move the networks by 0.32–0.65 of the LR(N) slope, splitting the softest
  particle moves the five networks toward quark (−0.011 to −0.054), removing the softest particle moves them by −0.011 to +0.003
  (LR(N) −0.099), and the count-fixed control reproduces the split-hardest response (restore fraction −0.11 to +0.14; LR(N) 1.0,
  classical LR(N, shapes) 0.64, PFN 0.49). Excluded as the description of ParT, PN and LorentzNet on these edits. PFN and the classical
  LR(N, shapes) show a count component (restore 0.49 / 0.64, remove-softest −0.089 / −0.059).
- Hypothesis (c), IRC-safe shapes only: the EFN responds 0 to every split (|Δlogit| ≤ 2·10⁻⁴), as built; the five networks respond
  +0.10 to +0.20 to splitting the hardest particle. Excluded for the networks.
- Hypothesis (d), sharing of energy among hard particles (pT^D-like): response falls with the rank of the split particle (hardest
  +0.10–0.20, rank 5 +0.04–0.11, softest −0.01 to −0.05), and the ΔN = 0 control with the same pT^D change gives the same response.
  Not excluded by these edits. The edits do not separate pT^D from other functions of the hard-particle momentum fractions.
- Particle type: in the kinpid models and LorentzNet, adding a photon instead of a charged hadron as the second daughter lowers the
  response by 0.085–0.139 with pT^D held at the same value, so these models give weight to the charged-particle content of the hard core.
- The split edits are detectable by a classifier (AUC ≈ 1), mainly through the ΔR = 5·10⁻⁴ pair; the network responses are
  responses to inputs outside the training distribution. The remove-softest edit is less detectable (AUC 0.71).
- Herwig: every model's AUC drops by 0.06–0.12 and the drop comes from gluon jets (fewer constituents in Herwig: 48.7 vs 53.6). The
  per-split response is the same in both generators within 0.005.

## Issues

- First full run (d-h5-run2) crashed: `rmsoft`/`merge` on jets with < 3 particles produced empty jets (NaN shapes). Fixed by
  leaving those jets unedited and excluded (2 jets); the control comparison now uses the jets where both edits apply.
- PN kin/kinpid are interim checkpoints (best epoch 11/14 of 15 in the main run; 16/14 in the Herwig run). The per-epoch scan
  bounds the stage dependence. A rerun on the final PN checkpoints is cheap (`--step run --no_epochs`).
- EFN/PFN are torch stand-ins trained here, not the H2 energyflow models (not available for QG).
- EFN rmsoft −0.013: removing the softest particle at finite z changes Σ z_i Φ; the EFN is IR-safe only in the z → 0 limit.
- Effect sizes are 0.03–0.2 in logit (calls changed ≤ 4 %); with 10,000 jets, 200 of 220 tests pass Holm. The size comparisons
  above carry the result more than the p-values do.

## Explainer

1. **Question.** Gluon jets have more particles than quark jets. Do trained quark–gluon taggers decide by counting particles,
   or by something else? If one particle is split into two collinear pieces, so that every IRC-safe jet shape stays fixed and only the
   count changes, do the taggers move toward "gluon" by the amount that counting predicts?
2. **Physics background.** At leading order, the mean multiplicity scales with the colour charge, $C_A/C_F = 9/4$. In this
   sample $\langle N_g\rangle/\langle N_q\rangle = 1.589$ (H2). If $N_q \sim \mathrm{Poisson}(\lambda_q)$ and
   $N_g \sim \mathrm{Poisson}(\lambda_g)$, the optimal count-only discriminant is
   $\log \mathrm{LR}(N) = N\log(\lambda_g/\lambda_q) - (\lambda_g-\lambda_q)$, so one more particle adds
   $\log 1.589 = 0.463$. A collinear split $p \to (z p, (1-z) p)$ at $\Delta R \to 0$ leaves every IRC-safe observable
   (e.g. $C_1^{(\beta)} = \tfrac12\sum_{ij} z_i z_j \Delta R_{ij}^\beta$, girth $\sum_i z_i \Delta R_i$) unchanged. It changes $N$ by +1, and it changes
   $p_T^D = \sqrt{\sum_i p_{T,i}^2}/\sum_i p_{T,i}$ by an amount set by the parent's momentum fraction.
3. **Method.** 10,000 Pythia test jets (5,000 per class, seed 701). Nine scorers: ParT and ParticleNet with and without PID inputs, LorentzNet (released), PFN, EFN,
   classical likelihood ratios, the count-only $\mathrm{LR}(N)$ from the training histograms, and permuted-weight ParT. Edits: split the particle of
   rank 0/1/5/softest; split $k = 1..5$ random particles; remove the softest; and a count-fixed control (merge the two softest and split the
   hardest: $\Delta N = 0$, same $p_T^D$ change). Metric: mean $\Delta\,\mathrm{logit} = \Delta[\log p_g - \log p_q]$ with 95 % bootstrap CI;
   slope in $k$; restore fraction $1 - \Delta_{\rm control}/\Delta_{\rm split}$.
4. **Result** (figure `results/physics/h5_counting.png`). The networks' slope per random split is 0.030–0.061, against 0.463 for
   Poisson counting and 0.095 for the count-only LR(N) built from the data. Splitting the hardest particle gives +0.10 to +0.20, and the
   count-fixed control gives the same (restore fraction −0.11 to +0.14). Removing the softest particle gives −0.011 to +0.003 (LR(N): −0.099). Splitting
   the softest particle moves the networks toward quark.
5. **Checks.** Unedited AUC reproduces the full-test and published values (ParT kinpid 0.914 vs 0.9121; LorentzNet 0.919 vs 0.9161).
   EFN response 0 (IRC-safe by construction); permuted ParT ≈ 0. The ordering holds at every training epoch and on Herwig jets.
   Limitation: a classifier tells split jets from real ones (AUC ≈ 1) because the daughters are closer than typical real pairs.
6. **Physical significance.** The trained taggers do not implement "count the particles": the soft end of the spectrum barely
   matters to them, and the response to a split is carried by how the energy of the hardest particles is shared. That quantity is not IRC-safe either,
   so the networks are not IRC-safe (as found in H1/D5), but the unsafety enters through the hard core, not through the
   multiplicity. The Casimir-meets-Poisson number overstates the information in a single extra particle by a factor of 5 in this sample.
