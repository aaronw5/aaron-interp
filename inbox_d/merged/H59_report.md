# H59: jet-charge null on the retrained QG kinpid networks (D16), and which jet charge Q_κ the networks follow (D16, D4m)

Status: done (Part 1 and Part 2, run seed 5901, verify seed 5911). The label-noise null trainings are launched: `d-h59-qg-pn-ln30` is running (2 of 20 epochs validated, validation accuracy against the noisy labels 0.624, so the network is not constant); `d-h59-qg-part-ln30b` is queued for a GPU. The first ParT attempt `d-h59-qg-part-ln30` failed at start-up on an A10 node with a cuDNN error (`CUDNN_STATUS_MAPPING_ERROR` in weaver's FLOP counter, before any training step), so it was relaunched with A10 excluded.

## Conclusion

A second training seed of both quark–gluon particle-ID networks (ParT and ParticleNet with particle-ID inputs) shows the same jet-charge rule as the first seed: positively charged jets are scored as more quark-like, with the size of the effect within 7 % of seed 0. The networks trained on class-independent labels output a constant, so they cannot serve as a null. When the charge signs of hard or soft charged particles are flipped while all momenta stay fixed, the score follows the standard jet charge Q_κ with κ ≈ 0.33–0.37. In this weighting the hardest particles count most, but soft particles still contribute. A pure count of + minus − particles (κ = 0), a strongly hard-weighted charge (κ ≥ 0.5), and a leading-particle-only charge are all excluded. Still uncertain: a single linear Q_κ explains only 40 % of ParT's per-jet response, against 80 % for ParticleNet, and the non-constant label-noise null is still training.

## Terms

- **Q_κ (jet charge):** Q_κ = Σ_i q_i (p_T,i / p_T,J)^κ.
  - q_i is the particle's electric charge, read from its PDG id.
  - p_T,J is the magnitude of the vector sum of the particle p_T.
  - κ = 0 gives the net number of positive minus negative particles. Larger κ gives more weight to the hardest particles.
- **PDG id:** the Particle Data Group integer code of the particle species (211 = π⁺, −211 = π⁻, 22 = photon, 130 = K⁰_L, ...). Flipping its sign turns a particle into its antiparticle.
- **kinpid:** the input set "kinematics + particle ID". Per particle it has 7 kinematic features plus charge and the flags isChargedHadron, isNeutralHadron, isPhoton, isElectron and isMuon. K± and p therefore look like π± of the same charge to these networks.
- **s:** the score, log p(quark) − log p(gluon), in log-odds. **Δs** is its change under an edit.
- **Holm:** step-down correction of p-values over a family of tests.
- **Permuted network:** a seed-0 network with the entries of every weight tensor randomly permuted (the H40v construction, seeds 1–6). It has the same architecture and weight distribution but no learned function.

## Procedure

**Samples.** Fresh random rows were drawn:
- **Pythia:** `data_qg.sample(5000, 5901)`, the KMT Pythia 8 test split, giving 5,000 quark and 5,000 gluon jets.
- **Herwig:** Herwig 7.1 files 18–19, 5,000 per class, seed 5902.
- **Verify:** seeds 5911 and 5912, with every step rerun.

**Models.** All were trained on Pythia; the shuffled ones use weaver checkpoints.
- `part_kinpid`, `pnet_kinpid`: seed 0, best epoch 17 of 20.
- `part_kinpid_s1`, `pnet_kinpid_s1`: torch seed 1, best epochs 16 and 18.
- `part_kinpid_shuf`, `pnet_kinpid_shuf`: label = hash(jet p_T, η, φ) mod 2.
- 12 permuted networks (6 from ParT, 6 from ParticleNet).

**References.** For each κ ∈ {0, 0.3, 0.5, 1, 2}, a logistic regression LR_κ on (p_T,J, N, Q_κ) was fitted on the 189,998 Pythia test-split jets outside the sample.
- The edits below keep p_T,J and N fixed, so each LR's response is exactly c_κ ΔQ_κ.
- This makes the LRs positive controls with a known κ.
- Fitted Q coefficients c_κ: 0.048 (κ = 0), 0.346 (0.3), 0.494 (0.5), 0.814 (1) and 1.60 (2) log-odds per unit charge.

**Edits.** All edits change PDG ids only. For every edited jet, the code asserts that every particle's (p_T, y, φ), the multiplicity N and the charged count n_ch are unchanged.

Part 1 (the H40/H42 relabels, `h40_qg_pid_charge.make_edits`):
- **E1:** every charged particle → its antiparticle, so Q_κ → −Q_κ. Read as Δs × sign(Q_0.3).
- **E2k1 − C2:** the hardest charged particle → its antiparticle, minus the control C2. C2 flips one + and one − particle together with |ΔQ_0.3|, |ΔQ_0.5| < 0.01, which is the same kind of relabel with the jet charge held fixed. Read × sign of the hardest particle's charge.
- **E3g_4 / E3n_4 − C3_4:** 2 π⁺ and 2 π⁻ below the jet's median particle p_T are relabelled as photons, or as K⁰_L. The control C3_4 swaps the same four pions π⁺ ↔ π⁻, which keeps the charged count and the total charge.

Part 2 (charge-sign flips, pid → −pid; four-momenta, species and n_ch fixed):
- **Fh_k:** flip the k hardest charged particles (k = 1, 2, 3).
- **Fs:** flip every charged particle whose z = p_T/p_T,J is below the median z of that jet's charged particles. This is the soft half, on average 9.6 particles.
- **FrS:** flip the same number of charged particles as Fs, chosen at random among all charged particles. This is the size-matched control.
- **Fr_k:** flip k random charged particles, size-matched to Fh_k.

**Part 1 metric.** Mean Δs (edit − control), signed as above, with a 95 % bootstrap CI over jets (B = 1000) and a normal-approximation p.
- **H0 (seed artefact):** the seed-1 effect is 0, or lies outside [0.5, 1.5] × the seed-0 effect.
- **H1:** the seed-1 effect is nonzero and the same as seed 0 within that band.
- **Shuffled networks:** their effect is compared with the 12 permuted networks. Rank p = (1 + #perm with |e| ≥ |e_shuf|)/7.
- **Seed spread:** the between-seed spread is the paired s1 − s0 difference over jets.
- **Holm:** applied over trained models × 4 contrasts × 2 generators. The constant networks are excluded from the Holm family (see Issues).

**Part 2 metric.** Quark jets from both generators are pooled; for each (jet, edit), ΔQ_κ and Δs are computed (79,952 rows, 9,999 jets).
- **(i) Held-out R² of Δs = a_κ ΔQ_κ.** The fit goes through the origin, with a 2-fold cross-fit split by jet. The interval is a bootstrap over jets that uses the same draws for every κ. The best κ is tested against the second best with a paired bootstrap p, Holm over the 4 trained networks.
- **(ii) Slope ratio** r(κ) = slope(Δs on ΔQ_κ | Fs rows) / slope(Δs on ΔQ_κ | Fh1–3 rows). If the network reads Q_κ*, then r(κ*) = 1. κ* is found by linear interpolation on the κ grid, with a jet-bootstrap CI (300 draws).
- **(iii) Leading-only test:** [Δs(Fh2) − Δs(Fh1)] and [Δs(Fh3) − Δs(Fh2)], each multiplied by the sign of the ΔQ that the added flip causes.
- **(iv) Per unit net charge:** mean Δs × sign(ΔQ_0), divided by the mean |ΔQ_0|/2, for each edit.

**Jobs.**
- `d-h59-smoke` (60 jets per class).
- `d-h59-run` (GPU RTX 2080 Ti, 1,964 s).
- `d-h59-verify` (seed 5911, 2,898 s).
- Training script: `interp/h59_kinpid_ln_train.sh`, a copy of `h42_kinpid_train.sh` with variant ln30. 30 % of the jets, chosen by a fixed hash of (φ, p_T, η) and independent of the class, have their label flipped. The check on test_file_0 gives a flipped fraction of 0.302 (0.302 for quarks, 0.302 for gluons).

## Numbers

**Unedited models on the samples used** (AUC / accuracy at s = 0):

| model | Pythia 5901 | Herwig 5902 | Pythia 5911 (verify) | Herwig 5912 (verify) | earlier value |
|---|---|---|---|---|---|
| ParT kinpid s0 | 0.909 / 83.6 % | 0.813 / 72.2 % | 0.907 / 83.5 % | 0.804 / 71.6 % | H42: 0.908 / 83.4 %, Hw 0.808; weaver test acc 84.0 % |
| PN kinpid s0 | 0.904 / 82.9 % | 0.815 / 72.2 % | 0.902 / 82.5 % | 0.805 / 71.3 % | H42: 0.904 / 82.9 %, Hw 0.808; weaver 83.4 % |
| ParT kinpid s1 | 0.909 / 83.7 % | 0.812 / 71.8 % | 0.907 / 83.3 % | 0.805 / 71.5 % | weaver test acc 84.0 % |
| PN kinpid s1 | 0.905 / 83.3 % | 0.815 / 72.1 % | 0.903 / 82.7 % | 0.807 / 71.5 % | weaver 83.6 % |
| ParT / PN kinpid shuf | 50.0 % acc, score SD < 10⁻⁶ | same | same | same | weaver 50.0 % (H56: constant) |
| LR_κ (κ = 0…2) | 0.835–0.837 | 0.798–0.799 | | | |

- ParT shuf shows an AUC of 0.56–0.57 although its output is constant: its scores differ only at the float-rounding level (SD < 10⁻⁶), and those differences produce the ranking.

**Part 1: second seed versus seed 0** (Δs in log-odds, 95 % CI; Pythia / Herwig):

| contrast | ParT s0 | ParT s1 | PN s0 | PN s1 | LR_0.3 | permuted max abs (ParT / PN) |
|---|---|---|---|---|---|---|
| E1 × sign Q_0.3 | −0.246 [−0.255, −0.237] / −0.246 | −0.249 [−0.259, −0.240] / −0.252 | −0.238 / −0.237 | −0.254 [−0.261, −0.247] / −0.249 | −0.289 / −0.280 | 0.020 / 0.113 |
| E2k1 − C2 × sign lead | −0.403 / −0.433 | −0.401 / −0.427 | −0.415 / −0.437 | −0.436 / −0.460 | −0.399 / −0.400 | 0.021 / 0.152 |
| E3g_4 − C3 (π± → γ) | +0.150 / +0.172 | +0.133 / +0.151 | +0.144 / +0.161 | +0.140 / +0.148 | 0.000 | 0.085 / 0.323 |
| E3n_4 − C3 (π± → K⁰_L) | −0.233 / −0.327 | −0.222 / −0.288 | −0.256 / −0.340 | −0.261 / −0.344 | 0.000 | 0.159 / 0.301 |

- **Seed ratio s1/s0.**
  - E1: ParT 1.01 / 1.03, PN 1.07 / 1.05.
  - E2: 0.99–1.05.
  - E3: 0.88–1.02.
  - All are inside [0.5, 1.5] with the same sign, so H0 (seed artefact) is excluded for all 16 seed-1 cells.
- **Paired s1 − s0.**
  - E1: ParT −0.003 [−0.010, 0.006] (Pythia) and −0.007 [−0.016, 0.002] (Herwig); PN −0.015 [−0.020, −0.011] and −0.012 [−0.017, −0.007].
  - E3n on Herwig: ParT +0.039 [0.032, 0.045].
  - The between-seed spread is therefore ≤ 0.023 log-odds for the charge contrasts and ≤ 0.039 for the species contrasts. The effects are 0.15–0.46.
- **Holm.** Over the 32 tests of the trained non-constant networks (s0, s1 × 4 contrasts × 2 generators), every p_Holm < 10⁻¹⁰⁰.
- **Against the permuted band.** For E1 and E2, |trained| exceeds all 6 permuted values in every cell; the rank p is 1/7 = 0.14, the floor for 6 permutations. For E3, the ParticleNet permuted copy s3 reaches −0.30 / +0.32, so PN E3 lies inside or at the edge of its permuted band (rank p 0.29). This repeats H40v.
- **Shuffled-label networks.** Constant output, so every Δs is 0 within 10⁻⁶. They lie inside the permuted band trivially (rank p 1.0), and this does not test H1 (see H56). The label-noise trainings above are the non-collapsing replacement.
- **Verify (seed 5911).** E1 × sign Q: ParT s0 −0.249 / −0.243, s1 −0.254 / −0.248; PN s0 −0.237 / −0.243, s1 −0.249 / −0.256. The s1/s0 ratios are 0.98–1.05.

**Part 2: which Q_κ** (quark jets, Pythia + Herwig pooled; run seed 5901, verify 5911 in brackets).

Mean |ΔQ_κ| per edit: Fh1 has 2.0 at κ = 0 and 0.46 at κ = 1; Fs has 3.29 at κ = 0 and 0.026 at κ = 1. The soft flips therefore change the count-like charge a lot and the hard-weighted charge almost not at all.

| quantity | ParT s0 | ParT s1 | PN s0 | PN s1 | LR_0 | LR_0.3 | LR_0.5 | LR_1 |
|---|---|---|---|---|---|---|---|---|
| held-out R², κ = 0 / 0.3 / 0.5 / 1 / 2 | 0.26 / **0.40** / 0.40 / 0.29 / 0.13 | 0.26 / 0.41 / **0.41** / 0.31 / 0.15 | 0.51 / **0.79** / 0.77 / 0.55 / 0.23 | 0.50 / **0.81** / 0.81 / 0.59 / 0.25 | 1 at κ = 0 | 1 at 0.3 | 1 at 0.5 | 1 at 1 |
| best − second R² [CI], Holm p | +0.004 [0.002, 0.008], 0.018 | +0.003 [−0.000, 0.006], 0.14 | +0.018 [0.015, 0.020], 0.008 | +0.000 [−0.002, 0.003], 0.80 | | | | |
| κ* where slope ratio Fs/Fh = 1 [CI] | 0.32 [0.31, 0.33] (0.33) | 0.34 [0.33, 0.35] (0.35) | 0.34 [0.33, 0.34] (0.34) | 0.36 [0.36, 0.37] (0.37) | 0.00 | 0.30 | 0.50 | 1.00 |
| fitted a_0.3 (log-odds per unit Q_0.3) | 0.36 | 0.36 | 0.41 | 0.43 | | 0.346 | | |
| Fh2 − Fh1, signed [CI] | +0.318 [0.308, 0.327] | +0.317 | +0.426 [0.422, 0.430] | +0.447 | +0.095 | +0.352 | +0.326 | +0.190 |
| Fh3 − Fh2, signed | +0.280 | +0.278 | +0.367 | +0.376 | +0.095 | +0.304 | +0.255 | +0.117 |
| Δs per unit net flipped charge: Fh1 / Fs / FrS | 0.53 / 0.118 / 0.18 | 0.54 / 0.114 / 0.18 | 0.53 / 0.118 / 0.22 | 0.57 / 0.112 / 0.22 | 0.095 / 0.095 / 0.095 | 0.43 / 0.127 / 0.20 | 0.46 / 0.063 / 0.14 | 0.38 / 0.009 / 0.063 |

- **κ* by generator** (quark jets): Pythia 0.32, 0.33, 0.34, 0.36; Herwig 0.33, 0.35, 0.33, 0.36 (ParT s0, ParT s1, PN s0, PN s1).
- **κ* on gluon jets:** ParT 0.39 / 0.41, PN 0.34 / 0.35.
- **Permuted networks.**
  - |a_0.3| is 0.001–0.04 for 11 of 12; the outlier is ParticleNet permuted s3 with −0.20.
  - Their R² ranges from 0 to 0.74, and their κ* is scattered: no crossing, 0.003, 0.08, 0.11, 0.26, 0.32, 0.60.
  - R² is scale-free, so a permuted network with a tiny but charge-linear response can reach a high R². The trained networks differ from the permuted ones in the size of a_κ (10–200× larger than 11 of the 12), not in R².
- **Verify (seed 5911).**
  - Held-out R² repeats to ±0.01 (ParT 0.40 / 0.41, PN 0.79 / 0.80 at κ = 0.3).
  - The best κ by R² is 0.3 for s0 and 0.5 for s1 (best − second 0.002–0.016).
  - κ* = 0.33, 0.35, 0.34, 0.37.
  - Fh2 − Fh1 = +0.319 / +0.322 (ParT), +0.428 / +0.444 (PN).
  - Fs per unit net charge: 0.117–0.120.

**Predictions of the competing hypotheses, and outcome.**

| hypothesis | prediction | outcome |
|---|---|---|
| H_hard: hard-weighted charge (κ ≥ 0.5) | Fh strong, Fs weak, κ* ≥ 0.5 | Fs is weak (0.22 × Fh1 per unit net charge), but κ* = 0.32–0.37 with upper CI ≤ 0.38, so κ ≥ 0.5 is excluded in all 4 networks and both samples |
| H_count: soft-weighted or count-like (κ ≤ 0.3, κ = 0) | Fs per unit net charge ≈ Fh1 (LR_0: ratio 1.0); κ* ≈ 0 | ratio 0.21–0.22 (LR_0.3 gives 0.29); κ* > 0.31, so κ = 0 is excluded |
| H_lead: leading particle only | Fh2 − Fh1 = Fh3 − Fh2 = 0 | +0.32 to +0.45 (CI excludes 0, p < 10⁻¹⁰⁰): excluded |
| H_Q0.3–0.35 | κ* between 0.3 and 0.5, R²(0.3) ≈ R²(0.5) | observed |

## Reading

- **Part 1.**
  - The jet-charge response of the QG kinpid networks reproduces with a second training seed. E1 × sign Q_0.3 is −0.25 for both seeds and architectures, and the seed difference is ≤ 7 %.
  - The species responses (π± → γ toward quark, π± → K⁰_L toward gluon) also reproduce, with a seed difference up to 12 %.
  - These measurements bound the seed spread from one extra seed per architecture; they do not determine its distribution.
  - The shuffled-label control gives no information, because the networks are constant.
- **Part 2.**
  - At fixed four-momenta, multiplicity and charged count, the networks' response to flipping charges is described best among the tested definitions by the jet charge with κ ≈ 0.32–0.37. Soft flips count about 4–5× less per unit of net charge than flipping the leader.
  - The second and third hardest charged particles add almost as much as the first.
  - The single-κ linear model explains 79–81 % of ParticleNet's Δs variance but 40–41 % of ParT's.
  - So ParT's response has a large component that is not a fixed linear function of any one Q_κ, for example a dependence on where the flipped particle sits or on the jet's other properties. This was not resolved here.
  - The R² comparison cannot separate κ = 0.3 from κ = 0.5 (differences ≤ 0.018). The slope-ratio estimate κ* separates them, with a seed-to-seed difference of 0.02–0.03, which is larger than its jet-bootstrap CI.
- **Scale.** The fitted slope per unit Q_0.3 (0.36–0.43) is close to the coefficient of the Pythia logistic regression on (p_T, N, Q_0.3), which is 0.346.

## Files

- Script: `interp/h59_charge_kappa.py` (`--step run [--seed] [--tag]`, `--step fig`).
- Training: `interp/h59_kinpid_ln_train.sh`.
- Results:
  - `results/physics/h59_charge_kappa.json`, `h59_charge_kappa_v.json`.
  - Per jet, on the volume: `h59_perjet_{py,hw}.npz`, `h59_perjet_{py,hw}_v.npz`.
  - Logs: `logs/d-h59-run.log`, `d-h59-verify.log`, `d-h59-smoke.log`.
- Figure: `results/physics/h59_kappa.png`.
  - Left: held-out R² against κ for the 4 networks and the range of the 12 permuted networks.
  - Right: slope ratio (soft / hard flips) against κ for the networks and the LR_0, LR_1 controls; the ratio crosses 1 at the κ the score follows.

## Issues

- **Deviation from the plan.** Part 1 was run with `interp/h59_charge_kappa.py`, not with `h42_qg_charge_null.py --tag _retrained`, so that it could use fresh seeds (5901 / 5911, 5,000 per class), add the 12 permuted networks and run the Part 2 edits in the same pass. The edits are H42's (`make_edits`). The C2ST and full-split asymmetry of H42 were not repeated.
- **Holm family.** In the run JSON, Part 1 counts 40 of 48 rejections. 8 of those come from ParT shuf, whose Δs are float noise (|mean| ≤ 10⁻⁶, SD ≈ 10⁻⁷), so the normal-approximation p is meaningless. The script now sets p = 1 for constant networks and drops them from Holm. That fix was made after the runs; the numbers above apply it by excluding them, and they are unchanged otherwise.
- **Degeneracy threshold.** The first Part 2 degeneracy threshold (var 10⁻¹⁴) let ParT shuf through in the smoke run; it was fixed before the full run.
- **Truncation.** ΔQ_κ is computed over all particles, while the networks see the first 128 particles. QG jets rarely have > 128 particles, so this affects only the softest particles of a few jets.
- **ln30 is not a no-class null.** 30 % symmetric label noise keeps the class ordering: p(label = q | jet) = 0.3 + 0.4 p(q | jet). The expected outcome is a network with the same features and compressed log-odds, so the charge response should be the trained one times a compression factor, not 0. That makes it a non-collapsing control of the training procedure, not a zero-information null.
- **Grid interpolation.** κ* is interpolated linearly between grid points 0.3 and 0.5. A finer κ grid would narrow it.

## Explainer entry (for site/explain.py, to be integrated by the ledger agent)

**H59_EXPLAINER**

**Question.** Quark–gluon ParT and ParticleNet with particle-ID inputs score positively charged jets as more quark-like (D16). Does this reproduce with a new training seed? And which definition of jet charge does the score follow: a count of charges, a hard-weighted charge, or only the leading particle?

**Physics background**
- The jet charge is
  $$Q_\kappa=\sum_{i\in \rm jet} q_i\left(\frac{p_{T,i}}{p_{T,J}}\right)^{\kappa}.$$
- $\kappa=0$ counts $n_+-n_-$. $\kappa\to\infty$ keeps only the hardest particle. Analyses usually use $\kappa\approx0.3$–$0.5$.
- In the Z+jet sample, quark jets come mostly from $u$ quarks (valence $u>d$), so $\langle Q_{0.3}\rangle_q=+0.097$ while gluons have $\approx 0$ (H42). Charge is therefore a valid quark-versus-gluon cue in this sample.

**Method**
- Only PDG ids are changed ($\pi^+\leftrightarrow\pi^-$ and so on). Every $(p_T,y,\phi)$, $N$ and $n_{ch}$ is fixed, so every kinematic observable is unchanged.
- Part 1: charge conjugation of the whole jet ($Q\to-Q$), the leader flip, and soft $\pi^\pm\to\gamma/K^0_L$, applied to seed-0, seed-1 and shuffled-label networks and to 12 weight-permuted networks.
- Part 2: flip the $k$ hardest charged particles (Fh$_k$), the softer half (Fs), or random particles of the same number (Fr).
  - For each edit, compute $\Delta Q_\kappa$ for $\kappa\in\{0,0.3,0.5,1,2\}$ and fit $\Delta s=a_\kappa\,\Delta Q_\kappa$.
  - The network follows $Q_{\kappa^*}$ where the slope from soft flips equals the slope from hard flips: $r(\kappa^*)=a^{\rm Fs}_{\kappa^*}/a^{\rm Fh}_{\kappa^*}=1$.
  - Controls: logistic regressions on $(p_T,N,Q_\kappa)$ with known $\kappa$, which each recover their own $\kappa$ exactly.

**Result**
- Seed 1 reproduces seed 0: E1 $\times$ sign $Q_{0.3}$ = $-0.25$ for both seeds and both architectures (ratio 1.01–1.07). The shuffled-label networks are constant.
- $\kappa^*=0.32$–$0.37$ for all four networks, on Pythia and on Herwig, in both the run and the verify sample.
- Adding the 2nd and 3rd hardest particles moves $s$ by +0.28 to +0.45 each.
- Soft flips move $s$ 0.22× as much as the leader flip per unit of net charge. A count-like charge would give 1.0; LR($Q_{0.3}$) gives 0.29.
- Figure: `results/physics/h59_kappa.png`.

**Checks**
- Unedited AUCs reproduce: ParT 0.909, PN 0.904–0.905 (Pythia); 0.81 (Herwig).
- Momenta, $N$ and $n_{ch}$ are asserted unchanged; E1 gives $Q\to-Q$ exactly.
- The LR controls recover $\kappa=0,0.3,0.5,1,2$.
- The verify run (fresh jets) repeats $\kappa^*$ within 0.01.
- Caveat: a single $Q_\kappa$ explains 80 % of ParticleNet's response but only 40 % of ParT's.

**Physical significance.** Both networks have learned a jet-charge observable close to the one used in flavour-tagging analyses ($\kappa\approx0.3$). It is neither a plain particle count nor the leading particle's charge. The cue exists in these samples because of the $u/d$ quark mix of Z+jet events, so its use is sample-specific rather than a quark-versus-gluon radiation property.

**Glossary entries (new)**
- **Q_κ (jet charge):** $Q_\kappa=\sum_i q_i (p_{T,i}/p_{T,J})^\kappa$, the $p_T$-weighted sum of particle charges in a jet. κ sets how strongly hard particles dominate: κ = 0 counts charges, and large κ keeps the leading particle.
- **PDG id:** the Particle Data Group's integer code for a particle species (e.g. 211 = π⁺, −211 = π⁻, 22 = photon, 130 = K⁰_L, 2212 = proton). A negative sign denotes the antiparticle.
- **kinpid:** the input set "kinematics + particle identification": per-particle momentum features plus charge and five species flags (charged hadron, neutral hadron, photon, electron, muon). The QG ParT/ParticleNet "kinpid" models are trained on it.
