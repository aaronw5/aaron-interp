# H81 — training dataset vs architecture: which one sets a top tagger's physics-edit response? (claim D32, exploratory)

## Conclusion
We applied the same seven edits to the same top and QCD jets and scored them with ParT and ParticleNet, each trained either on JetClass or on Top Landscape. On the variation between models, the architecture accounted for more of the differences in response than the training dataset did. On JetClass jets the architecture share was 0.56 and the dataset share 0.12. ParT models responded to added particles in one direction and ParticleNet models in the other, whichever dataset they were trained on. Physically, how a tagger reacts to extra soft or collinear particles and to a boost follows mainly from how the network is built. The common features of the response were shared by every tagger, including a cut on soft-drop mass and τ32: the strong response to the W decay angle and to removing the jet periphery. Several things are uncertain. Only 2–3 seed replicates exist per cell, and the JetClass ParticleNet has one. Against this small seed sample, only two of the 42 per-edit factor effects survive Holm correction. On Top Landscape jets the seed spread is as large as either factor. Which jet set is edited changes the responses about as much as the training dataset does.

Status: done for the analysis on the existing checkpoints. Training job **d-h81-tpart-s1** (second Top Landscape ParT seed, torch seed 8111) is **running**. It passed the smoke check: epoch 0 train loss 0.2026 and accuracy 0.917, validation accuracy 0.9299, against seed 0's epoch 0 at 0.2008 / 0.918. It takes about 17 min per epoch for 20 epochs, so it should finish around 2026-10-22 03:30 UTC. When it ends, run `./part_job.sh d-h81-s1 'python -m interp.h81_dataset_vs_arch --step run --models Top_ParT_s1'` and then the stats/plot job again. The code includes Top_ParT_s1 in the Top–ParT cell and in the seed pool automatically.

## Question
Paper D compares physics responses across models and datasets, but earlier comparisons mixed three things at once: training data, input set and seed. Here the jets, the inputs (kinematics only) and the edits are held fixed. Only the training dataset (JetClass vs Top Landscape) and the architecture (ParT vs ParticleNet) vary, with seed replicates where they exist. The question is how much of the variation in the response vector each factor accounts for.

## Procedure
**Script:** `interp/h81_dataset_vs_arch.py` (steps `run`, `stats`) and `interp/h81_plot.py`.
**Jobs:** d-h81-smoke (GPU, 60 jets per class, exit 0), d-h81-run (GPU, 9 min, exit 0), d-h81-stats (CPU 1/2Gi, 39 s, peak 273 MiB, exit 0), and d-h81-tpart-s1 (GPU training, running; the first launch with 16 Gi was OOM-killed and was relaunched with H2_MEM=24, peak use 21.7 GiB).

**Jet sets.** Both are random rows, never the first k.
- **top:** Top Landscape test split, 2,000 top + 2,000 QCD jets (rng 8101). Anti-kT R = 0.8, 550 < pT < 650 GeV, |η| < 2, Delphes (ATLAS card), massless constituents.
- **jc:** JetClass test TTBar_100 (TbqqP) and ZJetsToNuNu_100 (QCD), restricted to stored 550 < pT < 650 GeV, |η| < 2 and at least 3 particles. There are 38,239 and 36,575 eligible rows; 2,000 were drawn from each (rng 8102).
- Every jet is rebuilt per particle from (pT, η, φ, m).
- The E-scheme axis computed from the JetClass particles is 0.39 mrad (median) from the stored axis.

**Scores s** are top-vs-QCD log-odds, with Δs > 0 meaning more top-like.
- JetClass models: s = logit(TbqqP) − logit(QCD) = log p(TbqqP)/p(QCD).
- Top Landscape models: s = the two-class top log-odds.
- Top-trained networks receive massless constituents (E = |p|), which is how they were trained (as in H19 and H31). JetClass-trained networks receive the stored constituent masses, which are zero for Top Landscape jets.

**Models.** All use kinematic inputs only: per particle log pT, log E, log(pT/pT_jet), log(E/E_jet), ΔR, Δη, Δφ. ParT also uses pairwise features.

| model | training data / recipe | role |
|---|---|---|
| JC_ParT_rel | released ParT_kin, full JetClass (authors) | cell JC–ParT (primary analysis) |
| JC_ParT_s0, s1, s2 | ParT_kin retrained from scratch on the 2M JetClass subset, torch seeds 0–2, 1/10 of the authors' sample budget (P10) | seed replicates; cell JC–ParT in the sensitivity analysis |
| JC_PN_rel | released ParticleNet_kin | cell JC–PN (single model) |
| Top_ParT_s0 | Top Landscape ParT_kin, authors' recipe (H2/H4) | cell Top–ParT (s1 still training) |
| Top_PN_s0, s1 | Top Landscape ParticleNet_kin, seeds 0 and 1 (H39) | cell Top–PN |
| Top_LNet, Top_PFN, Top_EFN | released LorentzNet; H2 energyflow PFN/EFN | references |
| CMS | call 1[105 < m_SD < 210 GeV and τ32 < 0.54] (m_SD: soft drop with z_cut 0.1, β = 0, R = 0.8; τ32 = τ3/τ2 with β = 1) | classical reference |
| LR(N) | histogram log-likelihood ratio of the constituent count N, 2-fold cross-fit on the unedited jets of the same set | classical count-only reference |

**Edit library.** Each edit E_k has a control C_k. The response uses the per-jet difference s(E_k) − s(C_k). Each edited jet is given to every model about its own E-scheme axis, except for E5 and C5.

| edit | what changes in the jet | control and why it is the comparison |
|---|---|---|
| E1 collinear split | Each of the 10 softest particles is replaced by two particles at the same (η, φ), each with half its pT and half its mass. N goes up by 10. The jet four-momentum, m, m_SD and τ32 are unchanged exactly (measured Δm = 0, Δm_SD = 0, Δτ32 = 0). | C1: the same 10 particles each moved by ΔR = 10⁻³ in a random direction, with no particle added. This separates the count change from any sensitivity to touching those particles. |
| E2 soft add | 5 massless particles of pT 0.5 GeV each, placed uniformly in the disc ΔR < 0.8 about the axis. N goes up by 5 and m by about 2.5–2.9 GeV. | C2: 2.5 GeV added to the hardest particle's pT. Same added pT, no new particle, m up by 0.4 GeV. |
| E3 energy scale U(1.1) | Every constituent four-momentum is multiplied by 1.1, so m and pT both go up 10 % and m/pT and all angles stay fixed. | C3 UB(1.1) (H71): U(1.1) and then a boost back to the original jet momentum, giving m × 1.1 at fixed pT. E3 − C3 is therefore a 10 % pT change at the same edited mass. |
| E4 boost B(1.1) | An exact boost along the jet momentum: pT × 1.1, m unchanged, opening angles × ~1/1.1 (H12). | C4 P(1.1): pT and m × 1.1 at fixed angles (the same jets as E3). E4 − C4 is a 10 % mass reduction carried by the angles at fixed pT. |
| E5 WTA axis | The particles are unchanged; the network inputs (Δη, Δφ) are computed about the winner-take-all axis. The WTA axis comes from C/A clustering where each merge keeps the harder branch's direction; its median offset from the E-scheme axis is 0.087 (jc) and 0.091 (top). | C5: inputs about an axis displaced from the E-scheme axis by the same length in a random direction. This separates "the WTA direction" from "any off-centre axis of that size". |
| E6 W decay angle | The jet is split into three exclusive-kT subjets, and the W pair is the pair with mass closest to 80.4 GeV. The W-pair constituents are rotated rigidly in the W rest frame so that \|cos θ*\| increases by 0.3 (median 0.42 → 0.72 on top jets). The rotation angle has median 0.36 rad. m_top and m_W are fixed exactly; m_SD changes by +1.5 GeV and τ32 by +0.04. (H31 'D'.) | C6: the W system rotated in its rest frame about the b-subjet direction by the same angle. \|cos θ*\| and all pair masses stay fixed, and the constituents move by a comparable amount (H31 'cphi'). |
| E7 periphery removal | All particles beyond ΔR 0.4 of the axis are removed (on average 17 per top jet and 10 per JetClass jet, about 10 % of pT). The original m and pT are then restored with one momentum scale and one angular scale (H72 Rm+m). It is valid for 2,003 / 4,000 top jets and 3,047 / 4,000 JetClass jets. | C7: the same number of pT-matched random inside particles is removed and then the same m and pT restore is applied (H72 Cr+m). This gives the same N and pT lost, removed from inside instead of outside. |

**Response vector.** For model M on jet set X: r_k = mean over valid jets of [s(E_k) − s(C_k)] / SD(s on unedited jets of X).
- r is in units of the standard deviation of the model's own score over the mixed top + QCD sample. For trained networks that SD is 5.2–9.4 log-odds units, so r = 0.1 corresponds to about 0.5–0.9 in log-odds.
- The normalisation makes models with different logit scales comparable.

**Statistics.**
- Bootstrap: B = 1000 resamples of jets, with the same resampled jets for all models, used for r, cosines and shares.
- Sign-flip permutation test of mean s(E) − s(C) = 0 per model × edit × set (2,000 sign flips; p floor 5·10⁻⁴). Holm correction over 13 models × 7 edits × 2 sets = 182 tests.
- **Variance decomposition (two-way), per edit and per jet set.** Each cell is (dataset, architecture), and the cell mean μ is the average of the models in it. The half-contrasts are:
  - dataset effect a = [μ(JC,ParT) + μ(JC,PN) − μ(Top,ParT) − μ(Top,PN)]/4
  - architecture effect b = [μ(·,ParT) − μ(·,PN)] summed over datasets / 4
  - interaction c
  - seed variance σ² = the pooled within-group variance of r over the replicate groups {JC_ParT_s0,s1,s2} and {Top_PN_s0,s1}, with df = 3.
- **Variance components.** a² − kσ², b² − kσ² and c² − kσ², each floored at 0, where k = Σ_cells(1/n_cell)/16 removes the seed noise in the cell means; plus σ² itself. The variance share is each component divided by their sum. "Pooled" means the components are summed over the 7 edits before the shares are taken.
- **Primary analysis:** JC–ParT = the released ParT_kin, which matches the released PN_kin in data and recipe. **Sensitivity analysis:** JC–ParT = the mean of the three 2M-subset retrains.
- **Tests per factor and edit:**
  - F = (effect)²/(kσ²) on (1, 3) degrees of freedom, which asks whether the effect is larger than seed spread.
  - A per-jet sign-flip test of the contrast for these fixed models, which asks whether the effect is non-zero for these particular networks.
  - Holm correction over 7 edits × 3 factors × 2 sets × 2 variants (F tests) and over 42 tests per variant (sign-flip tests).
- **Label permutation test** (H0: within-dataset cosine = within-architecture cosine). Statistic T = mean cosine over (same dataset, different architecture) pairs minus mean cosine over (same architecture, different dataset) pairs. Cosines are taken between the 7-component r vectors of the 8 trained main models. The (dataset, architecture) labels are permuted over the models 10,000 times; the p-value is two-sided.
- **Jet-set effect.** For each edit, the cell-balanced mean over models of [r(JetClass jets) − r(Top jets)]/2. Its pooled squared size V_jetset is compared with V_dataset = the mean over the two jet sets of Σ_k a_k².
- Caveat: the bootstrap intervals on shares and cosines vary only the jets. They do not include seed-to-seed uncertainty, which enters only through σ² and the F tests.

## Checks
**Unedited models on these samples** (AUC / accuracy at s > 0):

| model | Top jets | JetClass jets | known (H32 samples) |
|---|---|---|---|
| Top ParT s0 | 0.9877 / 94.5 % | 0.9873 / 94.2 % | 0.9862 / 93.7; 0.9849 / 94.0 |
| Top PN s0 | 0.9869 / 94.3 % | 0.9855 / 94.1 % | 0.9845 / 93.8; 0.9836 / 94.0 |
| Top PN s1 | 0.9870 / 94.6 % | 0.9851 / 94.0 % | test acc 0.9403 (H39) |
| JC ParT_kin (rel) | 0.9844 / 93.5 % | 0.9927 / 96.1 % | 0.9815 / 92.4; 0.9920 / 95.8 |
| JC ParT s0 / s1 / s2 | 0.980–0.981 / 93.0–93.5 % | 0.990 / 95.0–95.2 % | 10-class 0.708–0.715 (P10) |
| JC PN_kin (rel) | 0.9857 / 93.3 % | 0.9898 / 95.3 % | 0.9821 / 92.3; 0.9885 / 94.9 |
| LorentzNet / PFN / EFN | 0.9875 / 0.9821 / 0.9776 | 0.9870 / 0.9807 / 0.9800 | 0.9867 / 0.9782 / 0.9728 (Top) |
| CMS m_SD + τ32 | ε_S 0.43, ε_B 0.014 | ε_S 0.53, ε_B 0.015 | – |
| LR(N) | AUC 0.717 | AUC 0.790 | – |

All trained networks reproduce their known values to within 0.004 AUC.

**Built-in checks of the edits:**
- The IRC-safe EFN gives r(E1) = 0.000 exactly, as it should for an exact collinear split.
- The Lorentz-invariant LorentzNet gives r(E5) = 0.000 exactly, since its inputs carry no axis.
- The CMS tagger gives r(E1) = r(E5) = 0, and LR(N) is non-zero only for E1 and E2.
- The C1 jitter moves every network by |Δs| ≤ 0.01.
- E1 and C1 leave m, m_SD and τ32 unchanged; ΔN is +9.98 to +9.99, because a few jets have fewer than 10 particles.

## Numbers
**Response vectors r** (95 % bootstrap CI over jets in the JSON):

| model | E1 split | E2 soft | E3 E-scale | E4 boost | E5 WTA | E6 W angle | E7 periph. |
|---|---|---|---|---|---|---|---|
| *JetClass jets* | | | | | | | |
| JC ParT rel | +0.009 | −0.006 | +0.014 | +0.030 | +0.003 | −0.099 | −0.178 |
| JC ParT s0 / s1 / s2 | −0.015 / +0.009 / −0.026 | −0.032 / −0.024 / −0.025 | +0.009 to +0.013 | +0.032 to +0.047 | +0.003 to +0.018 | −0.107 to −0.114 | −0.194 to −0.199 |
| JC PN rel | +0.037 | +0.013 | +0.011 | +0.014 | +0.016 | −0.079 | −0.181 |
| Top ParT s0 | −0.052 | −0.024 | +0.008 | +0.030 | +0.000 | −0.120 | −0.236 |
| Top PN s0 / s1 | +0.054 / +0.041 | +0.063 / +0.062 | +0.033 / +0.034 | +0.024 / +0.022 | +0.003 / −0.002 | −0.079 / −0.084 | −0.192 / −0.178 |
| LorentzNet / PFN / EFN | −0.016 / −0.041 / 0.000 | +0.032 / −0.020 / +0.001 | +0.005 / −0.026 / +0.005 | +0.013 / −0.074 / −0.046 | 0.000 / −0.054 / −0.066 | −0.094 / −0.105 / −0.105 | −0.194 / −0.088 / −0.133 |
| CMS / LR(N) | 0.000 / +0.353 | −0.012 / +0.188 | 0.000 / 0 | +0.037 / 0 | 0 / 0 | −0.169 / 0 | −0.335 / 0 |
| *Top Landscape jets* | | | | | | | |
| JC ParT rel | −0.005 | +0.006 | +0.012 | +0.038 | +0.017 | −0.093 | −0.149 |
| JC ParT s0 / s1 / s2 | −0.030 / −0.022 / −0.061 | −0.021 / −0.010 / −0.019 | +0.008 to +0.011 | +0.034 to +0.037 | −0.001 to +0.014 | −0.097 to −0.102 | −0.152 to −0.172 |
| JC PN rel | −0.031 | +0.008 | +0.009 | −0.001 | +0.020 | −0.073 | −0.129 |
| Top ParT s0 | −0.029 | −0.020 | +0.007 | +0.022 | +0.023 | −0.103 | −0.189 |
| Top PN s0 / s1 | +0.007 / −0.022 | +0.015 / +0.006 | +0.003 / +0.013 | +0.009 / +0.012 | +0.013 / +0.019 | −0.067 / −0.087 | −0.147 / −0.152 |
| CMS / LR(N) | 0.000 / +0.323 | −0.016 / +0.179 | 0 / 0 | +0.039 / 0 | 0 / 0 | −0.201 / 0 | −0.353 / 0 |

Every trained network has r(E6) between −0.07 and −0.12 and r(E7) between −0.13 and −0.24. Every 95 % CI excludes 0. The sign-flip p of each is at the floor of 2,000 flips, 1/2001; after Holm over 182 tests that floor becomes 0.091, so the per-model Holm values are limited by the number of flips, not by the data. Raising \|cos θ*\| at fixed masses, or replacing the periphery by a rescaled core, makes every tagger less top-like, and the classical m_SD + τ32 call responds in the same direction with a larger magnitude (−0.17 to −0.35).

**Variance shares, 7 edits pooled** (jet-bootstrap 95 % CI):

| analysis | jet set | dataset | architecture | interaction | seed | bootstrap fraction dataset > arch. |
|---|---|---|---|---|---|---|
| primary (JC ParT = released) | JetClass | 0.12 [0.09, 0.16] | 0.56 [0.52, 0.58] | 0.22 [0.19, 0.24] | 0.10 [0.09, 0.13] | 0.000 |
| primary | Top | 0.15 [0.06, 0.26] | 0.34 [0.24, 0.42] | 0.09 [0.06, 0.14] | 0.41 [0.34, 0.52] | 0.024 |
| sensitivity (JC ParT = s0–s2) | JetClass | 0.09 [0.07, 0.11] | 0.73 [0.69, 0.75] | 0.10 | 0.09 | 0.000 |
| sensitivity | Top | 0.08 [0.02, 0.16] | 0.49 [0.37, 0.57] | 0.04 | 0.40 [0.33, 0.50] | 0.000 |

On Top jets the seed share is at least as large as both factor shares in 91 % of bootstrap resamples (primary analysis); on JetClass jets this happens in 0 %.

**Per edit** (primary analysis; architecture effect b > 0 means ParT is above PN):
- **E1 split, JetClass jets:** b = −0.032 [−0.034, −0.030], a = +0.013. The F test against seed spread gives Holm p = 0.65, because the three JetClass ParT seeds spread from −0.026 to +0.009 (σ² = 2.3·10⁻⁴). The fixed-model sign-flip test gives Holm p = 0.021.
- **E2 soft add, JetClass jets:** b = −0.026 [−0.028, −0.025]. F test Holm p = 0.024, one of the two factor effects that exceed the seed spread.
- **E4 boost, Top jets:** b = +0.013 [+0.012, +0.014]. F test Holm p = 0.022, the other one.
- **E6, both sets:** b = −0.014 and −0.012, meaning ParT responds more to the W angle than PN. F test Holm p ≥ 0.15.
- **E7:** dataset and architecture effects are similar in size (a = +0.016, b = −0.012 on JetClass jets; +0.015 / −0.015 on Top jets). F test Holm p ≥ 0.4.
- All other factor effects have F-test Holm p > 0.1. The fixed-model sign-flip tests give p_Holm = 0.021 for most a and b. That value is the floor of 2,000 flips times 42 tests: each fixed contrast is measured precisely over 4,000 jets.

**Counting edits (E1, E2) by architecture on JetClass jets.**
- ParT, both datasets (4 JetClass + 1 Top model): r(E1) from −0.052 to +0.009 and r(E2) from −0.032 to −0.006.
- ParticleNet, both datasets (3 models): r(E1) from +0.037 to +0.054 and r(E2) from +0.013 to +0.063.
- The sign follows the architecture. On Top jets the separation is weaker: PN r(E1) is −0.031 to +0.007, and r(E2) is +0.006 to +0.015 for PN against −0.021 to +0.006 for ParT.

**Cosine similarities and the label permutation test.** All trained-network r vectors have cosine 0.90–0.998, because the shared E6/E7 components dominate them.

| jet set | within-dataset cos (different arch.) | within-architecture cos (different dataset) | T [95 % CI] | permutation p |
|---|---|---|---|---|
| JetClass | 0.914 | 0.974 | −0.060 [−0.074, −0.048] | 0.056 |
| Top | 0.967 | 0.982 | −0.015 [−0.021, −0.010] | 0.146 |
| both (14-dim) | 0.933 | 0.977 | −0.044 [−0.052, −0.036] | 0.045 |

Comparison of the released JetClass ParT with the two other models:
- On JetClass jets, cos(JC ParT, JC PN) = 0.977 and cos(JC ParT, Top ParT) = 0.968. The difference is +0.008 [+0.002, +0.015].
- On Top jets, the same two cosines are 0.959 and 0.978. The difference is −0.019 [−0.027, −0.012].
- With both jet sets combined, the difference is −0.004 [−0.009, +0.001].

**Jet-set effect.** The same models were scored on JetClass jets and on Top Landscape jets.
- The cell-balanced half-difference per edit is +0.014 for E1, +0.005 for E2, +0.004 for E3, +0.003 for E4, −0.007 for E5, −0.004 for E6 and −0.020 for E7.
- V_jetset = 7.4·10⁻⁴ [4.3, 13.1]·10⁻⁴, against V_dataset = 4.1·10⁻⁴ [3.2, 5.2]·10⁻⁴.
- The ratio is 1.8 [1.0, 3.4], and V_jetset > V_dataset in 97.7 % of bootstrap resamples.
- The cosine between a model's JetClass-jet and Top-jet r vectors is 0.97–0.995 for the ParT models and LorentzNet, and 0.90–0.95 for the ParticleNets.

## Reading (hypotheses)
- **H_data** (the training set determines the response) predicted three things: a dataset share above the architecture share; within-dataset cosines above within-architecture cosines; and JC ParT closer to JC PN than to Top ParT. On both jet sets the dataset share is below the architecture share (0.12 vs 0.56; 0.15 vs 0.34). The cosine ordering goes the other way (T < 0, permutation p 0.045 on the combined vectors). The third prediction holds only on JetClass jets (+0.008) and is reversed on Top jets (−0.019). With these models and edits, the measurements do not show the pattern H_data predicts.
- **H_arch** (the inductive bias determines the response) predicted an architecture share above the dataset share, within-architecture cosines above within-dataset cosines, and E1/E2 grouped by architecture. All three are observed on JetClass jets. On Top jets the first two hold with a smaller margin, and the E1 grouping is weaker. Measured against the seed spread (df = 3), only E2 on JetClass jets and E4 on Top jets exceed it after Holm correction.
- **H_seed** (the differences are training noise) predicted both factor shares at or below the seed share. On Top jets the seed share, 0.41, is at least as large as both factor shares. On JetClass jets the seed share is 0.10, below the architecture share of 0.56. H_seed is not excluded for the Top-jet responses and is excluded for the JetClass-jet architecture component, though only with jet-bootstrap intervals and a 3-df seed estimate.
- **Jet-set effect.** The difference between editing JetClass and Top Landscape jets (V_jetset) is 1.8 [1.0, 3.4] times the dataset effect (V_dataset). Which jets are edited moves the response at least as much as which data the model was trained on.
- **Earlier differences.** The earlier gap between JetClass ParT_kin and Top ParT, quoted from earlier nodes as −0.69 vs −0.38 for the count response, is not reproduced as a dataset effect here. In this design, on the E1 exact split, JC ParT_kin gives +0.009 and Top ParT −0.052 SD on JetClass jets, and −0.005 and −0.029 on Top jets. The edits and the normalisation differ from those earlier nodes.

## Limitations
- There is one JetClass ParticleNet with no seed replicate. The JetClass ParT seeds were trained on a 2M subset with 1/10 of the sample budget, so their cell mean mixes seed with training budget. The primary and sensitivity analyses agree in direction: architecture share 0.56 vs 0.73 on JetClass jets.
- The seed variance has 3 degrees of freedom. Top_ParT_s1 (running) will add a fourth and give the Top–ParT cell its own replicate.
- The share and cosine intervals vary only the jets. The cosines are dominated by the shared E6/E7 component, so the decomposition (which works on the differences between models) is the more direct measure.
- Top-trained networks get massless constituents and JetClass-trained networks get massive ones, so on JetClass jets the input four-vectors differ slightly between the two families.
- E7 is valid on 50 % (Top) and 76 % (JetClass) of jets, and its analysis set differs from the other edits.
- E1's control is a sub-resolution jitter (10⁻³), not a same-angle split.

## Files
- `/Users/anrunw/Documents/aaron-interp/interp/h81_dataset_vs_arch.py`, `/Users/anrunw/Documents/aaron-interp/interp/h81_plot.py`
- `/Users/anrunw/Documents/aaron-interp/results/physics/h81_stats.json`, `h81_run.json`, `h81_example.npz`, `h81_dataset_vs_arch.png`
- On the volume: `$AI_RESULTS/physics/h81_s_<model>.npz` (per-jet scores), `h81_edits.npz` (validity masks, classical observables).
- Logs: `/j-jepa-vol/aaron-interp/logs/d-h81-{smoke,run,stats,tpart-s1}.log`. The checkpoint will be written to `/j-jepa-vol/aaron-interp/models/top/part_kin_s1/`.

## Glossary additions
- **response vector r:** for one model on one jet set, the list over the seven edits of mean[s(edit) − s(control)] divided by the SD of the model's unedited score s on that jet set. It is in units of the score's spread, so that taggers with different logit scales can be compared.
- **variance share:** the fraction of the variation of r between models that is attributed to the training dataset, the architecture, their interaction, or seed-to-seed differences. It comes from a two-way decomposition of the cell means with the seed noise subtracted; the shares sum to 1.
- **jet-set effect:** half the difference of one model's r between JetClass jets and Top Landscape jets, averaged over models.
