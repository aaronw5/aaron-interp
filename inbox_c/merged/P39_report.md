# P39: axis dependence on unedited jets: natural off-centre vs centred W/Z jets; recentring with the stored axis rebuilt; the φ-mirror input split; the pre-registered P31 final analysis (claims C18, C3; D4) (agent report, 2026-10-21)

Status: **done**. Exploratory: no. Parts (b), (c) and (d) gave clean measurements. Part (a), the natural comparison, does not identify the effect; the reason is given below.

**Jobs**
- `anrunw-ai-c-p39-smoke`: first smoke. Nets ran; the analysis stopped on a key-slicing bug, which was fixed. `c-p39-ansm` re-ran the analysis on that output, and `c-p39-smoke2` / `c-p39-ansm2` checked the revised matching and the figure.
- `anrunw-ai-c-p39-run` (RTX-class GPU, 922 s): `python -m interp.p39_natural_axis --stage nets`. Runs ParT_full, ParT_kin, PN_full and PN_kin.
- `anrunw-ai-c-p39-lg` (RTX 2080 Ti): `env-lgatr/bin/python -m interp.p39_natural_axis --stage lgatr --smoke && … --stage lgatr`. Uses L-GATr checkpoint `model_run0_it154999.pt`, 77 % of the 200k-iteration schedule; the P5 job `anrunw-ai-c-p5-lgatr2` is still running, so this is an interim checkpoint.
- `anrunw-ai-c-p39-an2` (job.sh, 2 CPU / 8 Gi, 632 s): `--stage analyze`. This is the final analysis. `c-p39-an0` and `c-p39-an` were earlier versions of it.
- `anrunw-ai-c-p39-p31` (GPU, 534 s): the pre-registered P31 command (part d).

**Files**
- Script: `interp/p39_natural_axis.py`.
- Results: `results/part/p39_natural_axis.json` and `results/part/p39_nets_meta.json`. Per-jet logits are on the volume only: `/j-jepa-vol/aaron-interp/results/part/p39_nets.npz` and `/j-jepa-vol/aaron-interp/results/lgatr/p39_lgatr.npz`.
- Figures: `results/part/p39_natural_axis.png` (parts a and b) and `results/part/p39_mirror.png` (part c).
- P31: `results/part/p31_classes_steps_final.json` and `.png`.
- Logs: `logs/c-p39-{run,lg,an2,p31,smoke2}.log`.

## Conclusion
Conclusion: we tested whether ParT's habit of measuring jet mass about the stored jet axis matters for jets as they occur, or only for artificially shifted ones. Comparing unedited off-centre jets with centred jets of the same mass does not answer this. In real jets the off-centre ones are almost all at large |η|, and every network shows the same large score gap between the two groups, including L-GATr, which cannot see the stored axis. Edits answer a narrower question. If the jet is shifted so that its momentum direction moves away from the stored axis but the spread about the axis stays the same, ParT's W/Z score moves only about 0.2 times as much as it does under recentring (shifting the particles so their pT-weighted centre lands on the axis). So P36's response comes from the change of the axis moment M2, not from the momentum direction leaving the axis. The natural offset cannot be removed while keeping the stored axis on the momentum direction. Whether this term changes ParT's calls on real jets therefore remains an inference from local linearity, not a direct measurement.

## Definitions used in this entry
- **Stored axis.** The jet direction (jet_η, jet_φ) stored in JetClass. ParT and ParticleNet receive each particle's Δη, Δφ and ΔR relative to it.
- **Momentum axis.** The direction of the constituents' summed three-momentum. In unedited jets it lies on the stored axis to within 0.0002 rad (median; 99th percentile 0.0098, N = 200,000 W/Z jets).
- **Centroid offset $\vec c$.** $\vec c = \sum_i p_{T,i}(\Delta\eta_i,\Delta\phi_i)/\sum_i p_{T,i}$, the pT-weighted centre of the particles relative to the stored axis. Median |c| is 0.0055 rad, and the top quintile starts at 0.0097 rad.
- **Moments.** $M_0=\sum p_T$; $M_2=\sum p_T r^2$ about the stored axis; $M_{2c}=M_2-|\vec c|^2M_0$, the same moment about the centroid.
- **Mass-like quantities.**
  - $m_A=\sqrt{M_0M_2}$: the axis-moment mass.
  - $m_c=\sqrt{M_0M_{2c}}$: the centroid-moment mass. For massless particles at small angles it equals the invariant mass $m_{inv}$.
  - For an off-centre jet, $m_A$ exceeds $m_c$ by the offset term $\tfrac12\log(1+|\vec c|^2M_0/M_{2c})$.
- **Read-out.** $d=\ell_Z-\ell_W$, the difference between the Z and W logits; positive means more Z-like.
- **Local slope.** $s_j=[d(k{=}1.03)-d(k{=}1/1.03)]/(2\log1.03)$, where k scales every particle's (Δη, Δφ) about the stored axis. This is the jet's score change per unit log-mass under a real mass change; Δlog m_A = log k exactly.
- **R (edits).** Mean Δd under an edit divided by the H1 prediction, mean $s_j\,\Delta\log m_A$. R = 1 means the network responds as it would to a real mass change of the same size in m_A; R = 0 means no response.
- **$R_{nat}$ (part a).** $\sum(y-P_I)/\sum(P_A-P_I)$ over matched pairs, where:
  - $y$ is the score gap (off-centre minus centred jet);
  - $P_I=\bar s\,\Delta\log m_{inv}$ is the prediction if only the invariant mass matters (H0);
  - $P_A=\bar s\,\Delta\log m_A$ is the prediction if the network follows m_A (H1).
  - So R_nat = 0 under H0 and 1 under H1.
- **McNemar.** An exact binomial test on the jets whose call changes under one edit but not the other.
- **Holm.** Step-down correction of a family of p-values.
- **CEM.** Coarsened exact matching: jets are paired only within identical bins of the matching variables.

## Unedited accuracy
- P1 sample (rows 3,000–4,000 of each `_100` file, N = 10,000): ParT_full **85.93 %**, ParT_kin **74.47 %**, PN_full 84.56 %, PN_kin 70.36 %. These are the known values.
- L-GATr it154999: 84.27 % (rows 3,000–3,300, N = 3,000).
- This sample (N = 28,000; class-balanced 10-class accuracy): ParT_full 85.6 %, ParT_kin 73.3 %, PN_full 84.1 %, PN_kin 69.8 %.

## Sample
- JetClass test_20M `WToQQ_109.root` and `ZToQQ_109.root`. The plan named `_127` files, which do not exist: test_20M holds `_100`–`_119` and val_5M `_120`–`_124`. No loop-C node had used `_109`.
- Rows: numpy `default_rng(3939)`, a permutation of all 100,000 rows per class.
  - The first 10,000 per class form the pre-registered sample **S** (20,000 W/Z jets).
  - All 200,000 rows form the **extension**, added for statistical power.
- For part (c): 1,000 random rows of each of the 8 other classes' `_109` files.
- Jets with fewer than 2 particles were dropped (none in practice). Inputs were rebuilt with `part_core.build`.

## (a) Natural matched pairs: no edit
**Procedure.**
- Within W and within Z, take the top quintile of |c| and the bottom quintile.
- Pair each top jet with a bottom jet, with replacement (`default_rng(3940)`), under three matching designs:
  1. **cem4**, the plan's literal design: 4 quartile bins each of log m_inv, log pT, N particles and log M2c.
     - I matched on M2c, the moment about the centroid, rather than on M2 about the stored axis. Matching M2 would equalise m_A itself, and the test would then have no treatment.
  2. **nn**: exact quartile cells of log pT and N, then the nearest bottom jet in (log m_inv, log M2c), with caliper |Δlog m_inv| ≤ 0.01.
     - This was added because cem4 left a mean Δlog m_inv of +0.045, about 7× the treatment.
  3. **nn+|η|**: as nn, with |η_jet| also matched (quartile cells, nearest neighbour, caliper 0.05).
- **Statistics.**
  - Each design is also fitted by OLS of $y-P_I$ on $(P_A-P_I)$ plus the pair differences in log pT, N, |η_jet|, pT_D ($\sum p_T^2/M_0^2$), leading-particle pT fraction, and eccentricity of the pT-weighted (Δη, Δφ) ellipse.
  - Intervals: 95 % bootstrap over pairs (2,000 resamples).
  - Sign-flip permutation tests (10,000) for H0 (R_nat = 0) and H1 (R_nat = 1). Holm correction over the 5 models for H0 on S.

**Numbers** (nn design unless stated; 95 % CI in brackets).

| model | set | pairs | score gap y | H1 prediction P_A | H0 prediction P_I | R_nat | R_nat, shape-adjusted OLS |
|---|---|---|---|---|---|---|---|
| ParT_full | S | 2,320 | −0.36 [−0.52, −0.21] | +0.025 | −0.001 | −14.0 [−23.2, −7.5] | 0.37 [−0.51, 1.27] |
| ParT_full | all 200k | 34,414 | −0.31 [−0.35, −0.27] | +0.029 | −0.000 | −10.6 [−12.2, −9.1] | 0.05 [−0.20, 0.29] |
| ParT_kin | all | 34,414 | −0.21 | +0.024 | −0.000 | −8.4 [−9.3, −7.5] | 0.20 [0.04, 0.34] |
| PN_full | all | 34,414 | −0.31 | +0.026 | −0.000 | −12.0 [−13.7, −10.4] | 0.42 [0.13, 0.67] |
| PN_kin | all | 34,414 | −0.23 | +0.020 | −0.000 | −10.9 [−11.9, −10.0] | 0.44 [0.31, 0.58] |
| **L-GATr** | S | 2,320 | −0.43 | +0.023 | −0.001 | **−18.2 [−29.1, −10.8]** | 0.39 [−0.52, 1.36] |
| ParT_full | cem4, all | 39,984 | −0.32 | +0.42 | +0.35 | −10.4 [−11.2, −9.6] | −0.40 [−0.61, −0.20] |
| ParT_full | nn+\|η\|, all | 718 | +0.95 | +0.21 | −0.003 | +4.5 [3.2, 5.7] | 1.51 [0.72, 2.28] |
| PN_kin | nn+\|η\|, all | 718 | +0.97 | +0.15 | −0.003 | +6.2 [5.4, 7.0] | 2.42 [1.88, 2.93] |

- H0 and H1 are each rejected with permutation p ≤ 1e-4 for every model on S and on all jets in the nn design; on S the H0 test is Holm p 5e-4 (5 models).
- In the nn design the matched pairs agree in m_inv (standardised mean difference 0.000), M2c (0.003) and pT (−0.004). They differ in |c| by +0.012 rad, which gives Δlog m_A = +0.0031, of which the offset term is +0.0026.
- They also differ strongly in **|η_jet|: standardised mean difference +4.45**. They differ somewhat in pT_D (+0.18) and leading fraction (+0.17).
- Matching |η_jet| as well leaves only 718 of about 40,000 top-quintile jets with a partner (5 in S). In those 718 pairs the sign of the gap reverses.

**Reading.**
- The natural pT-centroid offset is largely set by |η_jet|. For massive jets, the pT-weighted centroid in η and the momentum direction separate as |η| grows, because η is not linear in p_z.
- Off-centre jets therefore sit at larger |η|, where every network is more W-like at fixed mass: y is −0.21 to −0.45 in every model.
- L-GATr shows the same gap. It never reads the stored axis, and its own axis-relative scalars are computed about the momentum axis.
- So in this design the score gap is not an axis-moment effect. The observed y is about 10× the size of the H1 prediction (+0.02–0.03) and of opposite sign.
- **The answer flips with the matching choice:** R_nat ≈ −10 without |η| matching, +4 to +6 with it. Shape-adjusted OLS gives 0.05–2.4.
- No design gives an R_nat interval that covers either 0 or 1 in all models.
- Within these data, the offset term cannot be varied independently of |η_jet| and shape. The observational test does not separate H0 from H1, so the expected pattern of both the thesis (R_nat 0.7–1.2, interval excluding 0) and the alternative (interval containing 0) is unmet.

## (b) Edits on the 20,000 S W/Z jets: what drove P36's R ≈ 1
**Edits.** pT and particle masses are kept; four-vectors and features are rebuilt.

| edit | what changes in the jet | centroid offset after (top 20 % median) | momentum-axis offset after | Δ½log M2 (mean) |
|---|---|---|---|---|
| rec | every particle shifted by −c (P36); stored axis kept | 0.0000 | 0.0130 rad | −0.0029 |
| iso | shifted by \|c\| at ±120° from c; 2s·c + \|s\|² = 0 keeps M2 exactly | 0.0122 | 0.0126 rad | 0.0000 |
| rs | shifted by \|c\| in a random direction | 0.0178 | 0.0123 | +0.0029 |
| ref | shifted by +c | 0.0245 | 0.0118 | +0.0086 |
| rec_rb | rec, then the stored axis rebuilt on the momentum-sum direction and Δη/Δφ recomputed | **0.0130** | **0.0000** | +0.0003 |
| base_rb | the axis rebuild applied to the unedited jet | 0.0129 | 0.0000 | +0.0003 |
| dil_rec | angles × √(M2_rec/M2): a real mass change matched to rec | 0.0122 | 0.0004 | −0.0029 |

Invariant mass is unchanged under rec, iso, rs, ref and rec_rb (mean Δlog m ≤ 1e-5).

**Competing predictions for iso**, as the ratio iso/rec:
- 0 if the network reads M2 (equivalently the centroid offset) about the stored axis;
- 1 if it reacts to how far the momentum direction is displaced from the stored axis;
- 0.5 if it reacts linearly to displacement along c.

**Numbers** (top 20 % of |c|, N = 4,000 W/Z jets; mean Δd [95 % CI], H1 prediction in brackets; calls = 10-class calls changed).

| model | rec | iso | rs | rec_rb | base_rb | dil_rec | iso / rec | p(iso = rec) | p(iso = rec/2) |
|---|---|---|---|---|---|---|---|---|---|
| ParT_full | −0.031 [−0.035, −0.028] (−0.030), R 1.04 [0.92, 1.17], calls 2.1 % | −0.006 [−0.010, −0.002] (0), 2.1 % | +0.020 (+0.030), R 0.68 | −0.0035 (+0.003) | +0.003 (+0.0025) | −0.032 (−0.030), R 1.05 | **0.19 [0.07, 0.32]** | 2e-4 | 2e-4 |
| PN_full | −0.026 (−0.027), R 0.97 [0.78, 1.18] | −0.004 [−0.009, 0.001] | +0.021, R 0.79 | −0.0033 | +0.0018 | R 1.03 | 0.16 [−0.04, 0.34] | 2e-4 | 6e-4 |
| PN_kin | −0.019 (−0.022), R 0.85 [0.68, 1.03] | −0.001 | +0.015, R 0.70 | −0.0050 | +0.0014 | R 1.00 | 0.08 [−0.13, 0.25] | 2e-4 | 2e-4 |
| ParT_kin | −0.008 [−0.020, 0.002] (−0.025), calls 6.9 % | +0.003, calls 6.6 % | +0.006 | −0.022 | −0.011 | R 1.76 | undetermined [−9.7, 8.7] | 0.012 | 0.09 |
| L-GATr (it154999) | −0.0045 (−0.026), R ≈ 0.17 | −0.0022 | +0.0001 | (= rec) | 0.0000 | −0.027, R 1.04 | 0.49 [0.20, 0.76] | 0.002 | – |

- On all 20,000 W/Z jets: ParT_full rec R 0.83 [0.75, 0.92], iso/rec 0.06 [−0.06, 0.17]; PN_full rec R 0.80, iso/rec −0.02.
- The P36 ParT_full R(rec) ≈ 1 is reproduced: 1.04 [0.92, 1.17] on the top 20 %.

**Reading.**
- Displacing the momentum direction from the stored axis by the same 0.013 rad while keeping M2 fixed (iso) moves ParT_full's W/Z log-odds by 0.19 [0.07, 0.32] of the recentring response.
  - "Momentum-axis displacement" (iso/rec = 1) is excluded (p 2e-4).
  - "Linear displacement along c" (0.5) is excluded (p 2e-4).
  - The M2 reading (0) is not excluded on all jets (0.06 [−0.06, 0.17]). On the top 20 % a small residual of 0.19 remains.
  - PN_full and PN_kin give the same pattern.
- **The rebuilt-axis recentring is degenerate.** Once the stored axis is reset to the momentum direction of the shifted jet, the centroid offset returns to 0.0130 rad (it was 0.0122 before the edit). Under a rigid shift the gap between pT centroid and momentum direction is fixed by the jet's own particle configuration, so no rigid translation can put both on the axis.
  - rec_rb therefore changes ½log M2 by only +0.0003. ParT_full moves by −0.0035 against an H1 prediction of +0.003.
  - The same rebuild on the unedited jet (base_rb) moves it by +0.003, the prediction.
  - The plan's predictions for rec_rb (thesis: R ≈ 1; alternative: R < 0.3) do not apply, because the edit leaves the offset in place.
  - The rec_rb − base_rb difference (−0.0065 [−0.008, −0.005], p 2e-4) is a residual of the η part of the shift: an η translation at fixed pT is not an exact boost for massive particles. It is not an axis effect.
- **L-GATr** responds to recentring at about 0.17 of the m_A prediction, although its response to the real mass change is normal (R 1.04). This agrees with P36 (0.2–0.3 at it139999). For L-GATr, rec and rec_rb are the same input, because it ignores the stored axis.
- **ParT_kin.** Recentring, iso, rs and the axis rebuild all change 3–7 % of its calls. The W/Z log-odds response is not resolved, as in P36 and P36v.

## (c) φ-mirror input split
**Procedure.**
- mirp: Δφ → −Δφ for every particle. It keeps the mass, M2, every pT, every energy and the momentum axis.
- Variants additionally flip the sign of charge, of d0 (transverse impact parameter), of dz (longitudinal impact parameter), of charge and d0 together, or of all three.
- Flips without the mirror are also included.
- Sample: all 28,000 S jets, 10 classes; calls changed are class-balanced; 1,000-resample bootstrap.
- Test: exact McNemar on per-jet "call changed under mirp" versus "under the variant", with Holm correction over the 2 full models × 5 variants (m = 10).
- The kin models see none of charge, d0 or dz, so their variants equal mirp. They are the reference: ParT_kin 1.70 %, PN_kin 1.56 %.

| variant | ParT_full calls changed [%] | McNemar vs mirp (mirp-only / variant-only), p_Holm | PN_full [%] | McNemar, p_Holm |
|---|---|---|---|---|
| mirp (φ only) | 7.82 [7.35, 8.27] | – | 11.83 [11.31, 12.42] | – |
| + charge | 11.15 | 908 / 2,249, 5e-129 | 12.29 | 993 / 1,297, 9e-10 |
| + d0 sign | 9.20 | 1,606 / 1,870, 2e-5 | 5.84 | 2,853 / 969, 6e-212 |
| + dz sign | 7.76 | 126 / 120, 1.0 | 11.89 | 184 / 191, 1.0 |
| **+ charge, d0** | **2.80 [2.56, 3.07]** | 2,419 / 997, 1e-133 | **3.36 [3.08, 3.64]** | 3,347 / 931, 1e-315 |
| + charge, d0, dz | 2.92 | 2,418 / 1,014, 1e-129 | 3.30 | 3,320 / 934, 3e-309 |
| charge only, no mirror | 10.21 | – | 6.51 | – |
| d0 only, no mirror | 9.98 | – | 11.21 | – |
| dz only, no mirror | 0.57 | – | 0.95 | – |

Accuracy: ParT_full 78.6 → 74.1 % (mirp) → 78.0 % (+ charge, d0). PN_full 76.4 → 69.5 → 75.9 %.

**Reading.**
- A φ mirror is a physical symmetry of the event only if each track's charge and the sign of its transverse impact parameter d0 are flipped with it: in a fixed magnetic field the mirror image of a positive track bends like a negative one, and d0 is signed by the track's angular momentum about the beam.
- Flipping both with the mirror cuts the change from 7.8 to 2.8 % (ParT_full) and from 11.8 to 3.4 % (PN_full).
  - That is within 1.1–1.8 points of the kin-model reference.
  - It restores accuracy to within 0.6 points.
- Neither flip alone does this. For ParT_full, flipping charge alone with the mirror raises the change to 11.2 %. For PN_full, flipping d0 alone lowers it to 5.8 %.
- The dz sign plays no role (p_Holm 1.0).
- So the "orientation dependence" of the full models found in P36v is the networks using the joint sign pattern of Δφ, charge and d0. This pattern is physical: the correctly mirrored jet is treated almost like the original. This extends P4v (PN_full, charge + d0: 3.1 %) to ParT_full.

## (d) Pre-registered P31 analysis (final m10 and L checkpoints)
**Command:**

```
python -m interp.p31_classes_steps_part --models m10,m10s,L,t,t2,released_kin,released_full,init0 \
  --epochs 'L:0,4,…,49;m10:0,2,4,6,9' --tag final
```

The rows are as in P31 (N = 15,000, test_20M `*_114`).

**Definitions.**
- R: the per-jet slope of the W/Z log-odds change under a rigid 0.06 rad shift about the stored axis, against a matched real dilation. 95 % bootstrap over jets.
- Route share: the fraction of the response to the matched real dilation carried when only the per-particle inputs are edited.

| model | 3-class acc | R φ+ | R φ− | R η+ | per-particle route share | W → Z under φ + 0.06 |
|---|---|---|---|---|---|---|
| m10 (10 classes, 5.0M jets, 10 epochs) | 0.759 (10-class held-out 0.735) | 0.350 [0.340, 0.361] | 0.411 [0.399, 0.423] | 0.420 [0.407, 0.432] | 0.36 | 20.5 → 31.3 % |
| m10s (10 classes, 1.5M jets) | 0.752 | 0.316 | 0.428 | 0.303 | 0.38 | 21.0 → 30.5 % |
| L (3 classes, 50 epochs) | 0.729 | 0.293 [0.276, 0.310] | 0.309 [0.293, 0.324] | 0.351 [0.332, 0.369] | 0.27 | 23.2 → 31.3 % |
| t / t2 (3 classes, 10 epochs, two seeds) | 0.747 / 0.745 | 0.215 / 0.254 | 0.472 / 0.491 | 0.218 / 0.228 | 0.29 / 0.31 | – |
| released ParT_kin / ParT_full | 0.764 / 0.825 | 0.721 / 0.790 | 0.743 / 0.771 | 0.871 / 0.807 | 0.83 / 0.84 | 18.9 → 42.5 % / 11.7 → 33.8 % |

- Arm L over training (φ+ / φ− / η+): epoch 34, the end of the flat learning rate, gives 0.28 / 0.41 / 0.40; epochs 39–49 (the decay phase) give 0.28–0.30 / 0.31–0.32 / 0.34–0.36.
- m10 over epochs 2–9: φ+ 0.35, η+ 0.42–0.47.
- The rotation null is ≤ 0.0096 for every model.
- The released models reproduce 74.47 % and 85.93 % on the P1 sample.
- **Permutation tests**, H0: R_arm = R_t (2,000 permutations; family of 15, Holm). Every comparison gives p_Holm 0.0075. Differences:
  - m10 − t: +0.135, −0.061, +0.202;
  - L − t: +0.078, −0.163, +0.133;
  - m10 − m10s: +0.035, −0.017, +0.117;
  - seed gap t2 − t: +0.039, +0.019, +0.010.
- **Pre-registered outcome.**
  - Thesis "class count": R(m10) ≥ 0.5 is not met; R(m10) is 0.35–0.42.
  - Alt 1 "steps": R(L) did not rise toward 0.7 over 50 epochs. It stayed at 0.29–0.35, and φ− fell from 0.43–0.48 to 0.31.
  - Alt 2 "data volume": m10 > m10s holds by +0.035 in φ+ (about 1× the seed gap) and +0.117 in η+ (about 12× the seed gap).
  - **Alt 3 "data scale" (no arm reaches 0.5): holds.**
- Reading: neither 3.3× more data with 10 classes nor 5× more optimisation steps brings our ParT_kin trainings above R ≈ 0.42. The released models are at 0.72–0.87. The gap is not closed by class count, step count or 5M-jet data volume within the P22 training code path.

## Issues and deviations
- **No `_127` file exists**, so `_109` was used.
- **Matching.** The plan's 4-bin CEM on m_inv, pT, N and M2 left mass imbalances 7× the treatment, and matching on M2 about the stored axis would have removed the treatment entirely. The pre-registered design is reported (cem4), with the two refinements beside it. None of the three identifies the effect, because |c| is collinear with |η_jet|.
- **L-GATr checkpoint** is interim (it154999). The P5 run is still going. The L-GATr stage idled the GPU about 18 min on CephFS imports, as in P36.
- **Rebuilt-axis edit** is degenerate, as explained in (b). The iso edit was added as the test that separates M2 from momentum-axis displacement.

## Explainer entry (for site/explain.py, Paper C)
**Question.** ParT measures jet mass partly as the spread of the particles about the stored jet axis (C3). In P36, recentring off-centre jets moved its W/Z score like a real mass change. But recentring also moves the jet's momentum direction off the stored axis, which never happens in real data. Is ParT's response due to the spread or to that inconsistency? And can it be seen on unedited jets?

**Background.**
- About the stored axis, $M_2=\sum_ip_{T,i}r_i^2=M_{2c}+|\vec c|^2M_0$, where $\vec c$ is the pT-weighted centre of the particles.
- An off-centre jet therefore has a larger axis-moment mass $m_A=\sqrt{M_0M_2}$ than the same jet centred, at equal invariant mass.
- In real jets the stored axis is the momentum direction, and $|\vec c|$ is a few thousandths of a radian. It arises because η is not linear in $p_z$.

**Method.**
1. Natural pairs: off-centre (top 20 % of $|\vec c|$) vs centred (bottom 20 %) W/Z jets with the same $m_{inv}$, $p_T$, particle count and $M_{2c}$. Compare their score gap with $\bar s\,\Delta\log m_A$ (H1) and $\bar s\,\Delta\log m_{inv}$ (H0), where $s$ is each jet's score slope under a real ±3 % mass change.
2. Edits on 20,000 W/Z jets:
   - recentring (shift by $-\vec c$);
   - a shift of the same length at 120° to $\vec c$, which keeps $M_2$ exactly: $2\vec s\cdot\vec c+|\vec s|^2=0$;
   - recentring followed by rebuilding the stored axis on the new momentum direction.
3. Mirror: $\Delta\phi\to-\Delta\phi$, with or without flipping the signs of charge, $d_0$ and $d_z$.

**Result** (figure `results/part/p39_natural_axis.png`, `p39_mirror.png`).
- Natural pairs: the off-centre jets differ from their partners by 4.5 standard deviations in |η_jet|. Every network, including L-GATr, gives them a W/Z gap about 10× the H1 prediction and of opposite sign, so the comparison does not isolate $m_A$.
- Edits: the 120° shift, which moves the momentum direction as far as recentring does but keeps $M_2$, gives $0.19\,[0.07, 0.32]$ of the recentring response in ParT_full. Recentring itself gives $R=1.04\,[0.92,1.17]$.
- Rebuilding the axis after recentring puts the offset back (0.0130 rad). No rigid shift removes it.
- Mirror: flipping charge and $d_0$ with the mirror lowers ParT_full's call changes from 7.8 % to 2.8 % (PN_full: 11.8 to 3.4 %).

**Checks.**
- Unedited accuracies reproduced (85.93 / 74.47 %).
- Invariant mass unchanged by the rigid edits (|Δlog m| ≤ 1e-5).
- The real-mass-change calibration gives R = 1.00–1.05 in every model.
- dz flip: no effect (Holm p 1.0).

**Why it matters.** The axis-moment reading of ParT comes from the particles' spread about the stored axis, not from the momentum direction leaving the axis. On real jets the extra spread from the natural offset cannot be separated from jet |η| with the data alone, so its effect on calls is inferred from the edit response (a few % of calls among the most off-centre 5 %), not observed directly.

**Glossary additions.**
- **momentum axis**: the direction of the constituents' summed three-momentum.
- **iso shift**: a rigid shift of length $|\vec c|$ at 120° to $\vec c$; it keeps $M_2$ about the stored axis while moving the momentum direction off the axis.
- **axis rebuild**: the stored axis reset to the momentum direction, with Δη/Δφ recomputed.
- **$R_{nat}$**: the natural-pair ratio $(y-P_I)/(P_A-P_I)$: 0 if only the invariant mass matters, 1 if the network follows $m_A$.
- **$m_c$**: the centroid-moment mass $\sqrt{M_0M_{2c}}$.
- **CEM**: coarsened exact matching.
- **local slope $s_j$**: a jet's score change per unit log-mass under a real ±3 % angular dilation.
- **physical mirror**: Δφ, charge and $d_0$ sign flipped together.
