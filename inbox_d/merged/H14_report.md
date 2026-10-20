# H14: Quark–gluon generator dependence (Pythia 8 vs Herwig 7.1) and the hard-core function under the hardest split (claims D7, D5; absorbs H10)

Script `interp/h14_qg_generator.py` (steps `hwdata`, `eval`, `efp`, `stats`, `fig`), Herwig retrain wrapper `interp/h14_hw_train.sh`.
Results `results/physics/h14_run.json` (all statistics), `h14_baselines.json`, `h14_baselines_x.json` (LR(N) fits, C2ST),
`h14_models_*.json` (checkpoint status), per-jet logits `h14_logits_<gen>_<set>[_x].npz` (on the volume only), figures
`results/physics/h14_generator.png`, `results/physics/h14_hardcore.png`. Logs `logs/d-h14-{hwdata,run-py,run-hw,x,efp,efp-x,keras,stats,fig}.log`.

Status: **running**. The Pythia-trained arm, baselines and H10 arm are complete. The Herwig-trained ParT_kin / PN_kin retrains
(jobs `d-h14-hw-part3`, `d-h14-hw-pnet3`) and the H2 PFN rerun (`d-h14-pfn2`) were still running when this report was written.
Hypothesis H1' (Herwig-trained models count) is therefore untested.

## Conclusion

We tested quark–gluon taggers trained on Pythia jets on both Pythia and Herwig jets. On both, adding one realistic soft particle
moves ParT, ParticleNet and LorentzNet by −0.024 to −0.002 in logit, while a classifier that uses only the particle count moves by
+0.09 to +0.11. So the "no counting" result of H5 does not depend on the generator used for testing. Physically, these taggers do
not count soft particles. They respond to how the jet's momentum is shared among its hardest particles and to how that momentum
is spread in angle. At fixed particle positions, the leading-particle fraction z1 has almost no effect once p_T^D is held fixed.
A local change of p_T^D has a small effect of the right size to account for the split response. But an IRC-safe network responds
just as much to that local change, so p_T^D is not isolated from angular energy flow. Still open: whether models trained on Herwig
behave the same way (their retrains are running), and which single quantity carries the split response.

Terms: logit = log p(gluon) − log p(quark). p_T^D = √(Σ z_i²), with z_i = p_T,i / Σ p_T. IRC-safe = unchanged by a collinear
split or by adding an infinitely soft particle. LR(N) = log-likelihood ratio of the particle count alone. C2ST = classifier
two-sample test AUC between edited and unedited jets (0.5 = indistinguishable).

## Procedure

**Samples.** Pythia: `interp.data_qg.sample(20000 per class, seed 9311)`, from the test split of the Komiske–Metodiev–Thaler set
(random rows), 40,000 jets, quark fraction 0.500, ⟨N⟩ q 33.48 / g 53.32. Herwig 7.1: Zenodo 3066475 files 18–19, which form the
test split of the 80/10/10 file split used for the Herwig retrains and are disjoint from H5/H5v, which used files 0–1. 20,000 random
jets per class, seed 9311, ⟨N⟩ q 33.55 / g 48.63. The Herwig files were also converted to particle_transformer parquet
(`--step hwdata`, columns as in `convert_qg_datasets.py`): files 0–17 for training and validation, files 18–19 for testing.

**Models** (logit = log p(g) − log p(q)):
- Pythia-trained: ParT_kin, ParT_kinpid, PN_kin, PN_kinpid (H2 weaver trainings, all final, 20 epochs; best epochs 14 / 17 / 17 / 17);
  LorentzNet (released weights); EFN_k = H2 energyflow EFN (`d-h2-efn`, finished 23:27 UTC, 30 epochs), ported into the torch EF
  module (Keras Dense kernels transposed, angle rows × 0.4 for the module's 1/0.4 input scaling); PFN_t and EFN_t = H5 torch
  stand-ins (400k jets), since the H2 PFN was OOMKilled after epoch 5 at 18 GiB (`d-h14-pfn`) and has been resubmitted at 28 GiB
  (`d-h14-pfn2`); ParT_kin with permuted weights (null).
- Herwig-trained: ParT_kin and PN_kin, trained with the authors' `train_QuarkGluon.sh` (20 epochs, batch 512, ranger, in-memory,
  1.6M samples per epoch). The first launch (`d-h14-hw-part`, `d-h14-hw-pnet`, 40 GiB) was OOMKilled while loading
  (38 GB at 5 min). It was relaunched at 48 GiB as `d-h14-hw-part3` and `d-h14-hw-pnet3`.
- Baselines: LR(N) from negative-binomial (NB) fits to the multiplicity per class on each generator's training jets (Pythia 1.6M,
  Herwig 1.8M); GBDT on log EFPs (d ≤ 5, β = 1, 54 EFPs) + N ("EFP+N", IRC-safe shapes plus count) and on EFPs only ("EFP"), each
  trained on 60k training jets of each generator.

**Edits** (deterministic per generator and edit name, so every model sees the same edited jets; jets an edit cannot be applied to
are left unedited and excluded from that edit's mean):
- `add_soft` (ΔN +1): a particle with the jet's softest p_T, placed at a random existing particle's position + 0.05 in a random
  direction (on-manifold, C2ST 0.518 Pythia / 0.517 Herwig; unedited-vs-unedited reference 0.510 / 0.507).
- `add_soft_swap` (ΔN 0 control): the same added particle, with the original softest particle removed.
- `rmsoft` (ΔN −1): remove the softest particle (C2ST 0.70).
- `split_hard` (ΔN +1): the hardest particle split into two, z = 0.5, ΔR = 0 (C2ST 1.0).
- `merge_split` (ΔN 0 control): the two softest particles merged (p_T-weighted position) and the hardest split.
- `ksplit_k`: k = 1, 2, 3 random particles split (z 0.5, ΔR 0); the slope per added particle is reported.
- H10 (i) `ptd_up/dn`: p_T^D ± 0.01 with z1, z2 and all positions fixed. The z_i of particles 3..N are set ∝ z_i^α, renormalised,
  with α solved per jet (applied to 24k / 37k Pythia jets). This also changes jet girth by −0.0045 and mass by −4.2 GeV
  (C2ST 0.75 / 0.85).
- H10 (i, local) `ptd34_up/dn`: p_T^D ± 0.005 by moving momentum between particles 3 and 4 only. The ordering constraints limit it
  to 2,366 / 1,228 Pythia jets (262 with both). Girth change < 0.0002 (C2ST 0.547 / 0.520).
- H10 (ii) `z1_up/dn`: z1 ± 0.015 at fixed p_T^D and z_{≥5}, with particles 2–4 re-shared (11k / 24k jets; C2ST 0.60 / 0.73).
- H10 (iii) `pair_dr{0, 0.005, 0.02}`: the leading pair moved along its own axis about its p_T-weighted centroid to separation ΔR,
  with the p_T list fixed; `pair_rot` (control): the pair rotated 90° about its centroid at fixed separation (C2ST 0.52).

**Metric and statistics.** The metric is the mean Δlogit per jet, with a 95 % bootstrap CI over jets (1000 resamples, shared across
tests). The H0 for each edit is Δ = 0 (one-sample t-test p, sign-flip permutation p with 2000 flips). For paired controls, the H0 is
edit = control (per-jet difference). For H1, the H0 is network Δ = LR(N) Δ (per-jet difference). Holm correction is applied across
all 766 (generator × edit/comparison × model) tests, and 592 pass at 0.05. The H10 coefficients are β = Σ(Δlogit_up − Δlogit_dn) /
Σ(Δx_up − Δx_dn) over jets where both edits apply, with a bootstrap CI.

## Numbers

### Unedited performance on each sample (quark = signal; AUC 95 % bootstrap CI)

| model | trained on | Pythia AUC [CI] | Pythia acc | Herwig AUC [CI] | Herwig acc | reference |
|---|---|---|---|---|---|---|
| ParT kin | Pythia | 0.9013 [0.8981, 0.9042] | 0.827 | 0.8079 [0.8038, 0.8122] | 0.718 | own H2 test acc 0.827 |
| ParT kinpid | Pythia | 0.9123 [0.9097, 0.9151] | 0.841 | 0.8049 [0.8006, 0.8093] | 0.715 | H2 test AUC 0.9121 |
| PN kin | Pythia | 0.9015 [0.8982, 0.9047] | 0.827 | 0.8080 [0.8038, 0.8122] | 0.717 | H2 test acc 0.827 |
| PN kinpid | Pythia | 0.9078 [0.9051, 0.9106] | 0.834 | 0.8051 [0.8010, 0.8092] | 0.712 | published PN 0.9116 |
| LorentzNet | Pythia | 0.9175 [0.9147, 0.9202] | 0.845 | 0.7997 [0.7955, 0.8035] | 0.698 | released 0.9161 |
| EFN_k (H2, IRC-safe) | Pythia | 0.8835 [0.8802, 0.8871] | 0.809 | 0.8023 [0.7983, 0.8068] | 0.715 | H2 test AUC 0.8831, acc 0.809 |
| PFN_t (H5) | Pythia | 0.8811 | 0.804 | 0.8033 | 0.713 | H5 0.888 (10k jets) |
| GBDT EFP+N | Pythia | 0.8624 | 0.784 | 0.7949 | 0.714 | – |
| GBDT EFP+N | Herwig | 0.8516 | 0.769 | 0.8000 | 0.732 | – |
| GBDT EFP | Pythia / Herwig | 0.8441 / 0.8337 | 0.773 / 0.758 | 0.7860 / 0.7879 | 0.715 / 0.725 | – |
| LR(N) | Pythia / Herwig fit | 0.8403 | 0.765 / 0.767 | 0.7860 | 0.711 / 0.718 | – |
| ParT kin, permuted | – | 0.548 | 0.499 | 0.526 | 0.499 | null |
| ParT kin / PN kin | Herwig | running | | running | | `d-h14-hw-part3`, `d-h14-hw-pnet3` |

Every unedited Pythia-trained model reproduces its known test accuracy or AUC on the Pythia sample (within 0.003). On Herwig jets,
the mean gluon logit of ParT kin falls from +1.46 to +0.35, while the mean quark logit stays at −1.64 / −1.69. The NB fits give
⟨N⟩_g 53.1 (Pythia) vs 48.7 (Herwig) and ⟨N⟩_q 33.4 vs 33.4.

### Count edits (Δlogit, 95 % CI; all network values have Holm p < 1e-4 against 0 unless noted)

| model | add_soft Pythia | add_soft Herwig | add_soft − LR(N) same-gen, Pythia | same, Herwig | rmsoft Pythia | rmsoft Herwig | random-split slope Pythia | Herwig |
|---|---|---|---|---|---|---|---|---|
| ParT kin | −0.0040 [−0.0043, −0.0037] | −0.0021 | −0.111 [−0.112, −0.111] | −0.091 | +0.0007 (Holm 1) | −0.0017 | +0.032 [0.031, 0.033] | +0.030 |
| ParT kinpid | −0.0155 [−0.0162, −0.0148] | −0.0107 | −0.123 | −0.099 | +0.0030 | −0.0010 (Holm 1) | +0.056 | +0.055 |
| PN kin | −0.0066 [−0.0073, −0.0061] | −0.0018 | −0.114 | −0.090 | +0.0028 | −0.0017 | +0.023 | +0.024 |
| PN kinpid | −0.0236 [−0.0247, −0.0224] | −0.0163 | −0.131 | −0.105 | +0.0015 (Holm 0.47) | −0.0030 | +0.061 | +0.057 |
| LorentzNet | −0.0138 [−0.0148, −0.0127] | −0.0045 | −0.121 | −0.093 | −0.0119 | −0.0163 | +0.061 | +0.061 |
| PFN_t | +0.0554 | +0.0544 | −0.052 | −0.034 | −0.0897 | −0.0953 | +0.055 | +0.053 |
| EFN_k (IRC-safe) | +0.0052 | +0.0056 | −0.102 | −0.083 | −0.0144 | −0.0166 | 0.000 | 0.000 |
| GBDT EFP+N (Pythia) | +0.0698 | +0.0756 | −0.038 | −0.013 | −0.080 | −0.078 | +0.073 | +0.075 |
| GBDT EFP+N (Herwig) | +0.0549 | +0.0601 | −0.053 | −0.029 | −0.058 | −0.065 | +0.049 | +0.056 |
| GBDT EFP (Pythia) | +0.0013 | +0.0044 | −0.106 | −0.084 | −0.016 | −0.016 | 0.000 | 0.000 |
| LR(N) Pythia fit | +0.1074 | +0.1116 | – | – | −0.110 | −0.115 | +0.103 | +0.107 |
| LR(N) Herwig fit | +0.0846 | +0.0887 | – | – | −0.087 | −0.092 | +0.081 | +0.084 |
| ParT kin permuted | −0.0279 | −0.0292 | – | – | +0.015 | +0.016 | 0.000 | 0.000 |

ΔN = 0 control `add_soft_swap`: networks −0.024 to −0.003 on both generators. The paired difference add_soft − swap for the five
networks is −0.002 to +0.006 (Pythia) and +0.002 to +0.010 (Herwig); on Pythia it is not distinguishable from 0 for PN kinpid
(Holm 1). For comparison, the same difference is +0.107 / +0.085 for LR(N) (Pythia / Herwig fit) and +0.076 / +0.058 for EFP+N.

### Hardest collinear split and its ΔN = 0 control (Δlogit, Pythia / Herwig)

| model | split_hard | merge_split (ΔN 0) | split − control [CI], Pythia | split − control, Herwig |
|---|---|---|---|---|
| ParT kin | +0.176 / +0.158 | +0.190 / +0.169 | −0.014 [−0.015, −0.014] | −0.011 |
| ParT kinpid | +0.169 / +0.156 | +0.188 / +0.168 | −0.019 | −0.013 |
| PN kin | +0.084 / +0.075 | +0.099 / +0.085 | −0.015 | −0.010 |
| PN kinpid | +0.217 / +0.187 | +0.236 / +0.199 | −0.019 | −0.011 |
| LorentzNet | +0.195 / +0.181 | +0.192 / +0.171 | +0.003 [+0.002, +0.004] | +0.011 |
| PFN_t | +0.169 / +0.158 | +0.076 / +0.058 | +0.093 | +0.101 |
| EFN_k | 0.000 / 0.000 | −0.010 / −0.012 | +0.010 | +0.012 |
| GBDT EFP+N (Pythia) | +0.072 / +0.073 | −0.010 / −0.010 | +0.081 | +0.084 |
| GBDT EFP (Pythia) | 0.000 / −0.001 | −0.017 / −0.020 | +0.017 | +0.019 |
| LR(N) (Pythia fit) | +0.107 / +0.112 | 0 / 0 | +0.107 | +0.112 |

Observables changed by split_hard (Pythia): ΔN +1, Δp_T^D −0.0454, Δz1 −0.078, Δz12 −0.086, Δgirth 0, Δmass 0.

### H10 arm: hard-core edits at fixed N and positions (β = Δlogit per unit; shown per +0.01, Pythia; Herwig in brackets)

| model | p_T^D, particles 3..N re-shared (n 22,863) | p_T^D, particles 3–4 only (n 262) [95 % CI] | z1 at fixed p_T^D (n 6,183) [95 % CI] | pair ΔR→0.02 | pair ΔR→0 − rotation control |
|---|---|---|---|---|---|
| ParT kin | −0.59 (−0.58) | −0.022 [−0.041, −0.003] (−0.025) | −0.0001 [−0.0017, +0.0016] (+0.0019) | +0.355 (+0.349) | −0.012 (−0.001, Holm 1) |
| ParT kinpid | −0.51 (−0.49) | −0.010 [−0.031, +0.010] (0.000) | −0.0003 [−0.0021, +0.0014] (+0.0011) | +0.286 | −0.007 (Holm 0.024) |
| PN kin | −0.64 (−0.61) | −0.032 [−0.050, −0.015] (−0.028) | +0.0029 [+0.0017, +0.0043] (+0.0031) | +0.311 | −0.075 |
| PN kinpid | −0.55 (−0.53) | −0.033 [−0.054, −0.014] (−0.029) | +0.0003 [−0.0010, +0.0019] (+0.0040) | +0.126 | −0.011 |
| LorentzNet | −0.46 (−0.43) | −0.029 [−0.055, −0.004] (−0.024) | +0.0038 [+0.0023, +0.0057] (+0.0055) | +0.270 | +0.003 (Holm 1) |
| PFN_t | −0.25 | −0.058 [−0.073, −0.045] | +0.0076 | +0.077 | +0.022 |
| EFN_k (IRC-safe) | −0.68 (−0.67) | −0.082 [−0.102, −0.062] (−0.054) | +0.0170 [+0.0156, +0.0183] | +0.405 (+0.388) | −0.164 |
| GBDT EFP+N (Pythia) | −0.24 | −0.036 [−0.068, −0.007] | +0.0071 | +0.119 | −0.037 |
| GBDT EFP (Pythia) | −0.60 | −0.054 [−0.088, −0.021] | +0.0149 | +0.213 | −0.122 |
| ParT kin permuted | −0.058 | +0.002 | −0.0006 | 0.000 | 0.000 |

Linear prediction for split_hard (Pythia), computed as β × the split's observed change: from the local p_T^D coefficient (Δp_T^D −0.0454),
ParT kin +0.10 (measured +0.18), ParT kinpid +0.04 (+0.17), PN kin +0.14 (+0.08), PN kinpid +0.15 (+0.22), LorentzNet +0.13 (+0.20),
EFN_k +0.37 (measured 0.00). From the z1 coefficient (Δz1 −0.078): −0.03 to +0.002 for the five networks. From the global p_T^D
coefficient: +2.1 to +2.9, 11–35 × the measured value.

## Reading

- **Generator shift.** The Pythia-trained networks lose about 0.10 in AUC on Herwig jets (0.90–0.92 → 0.80–0.81). The IRC-safe
  EFN (0.883 → 0.802), EFP baselines and LR(N) (0.840 → 0.786) lose a similar amount. A GBDT on EFP+N trained on Herwig reaches
  0.800 on Herwig jets, compared with 0.795 for the same GBDT trained on Pythia. Within the baselines, most of the drop therefore
  comes from Herwig jets being less separable on these inputs, not from a train/test mismatch. The shift sits in the gluon jets
  (mean gluon logit +1.46 → +0.35), consistent with Herwig gluon jets having 4.5 fewer particles on average. The networks' own
  Herwig-trained reference is pending.
- **H1 (D7 across generators).** On Herwig jets, the add-soft response of all five Pythia-trained networks lies between −0.016
  and −0.002. The Herwig LR(N) gives +0.089, and the network-minus-LR(N) differences are −0.09 to −0.105, all with Holm p ≈ 0. The
  removal and random-split results repeat the Pythia pattern within 0.01. Across the two test generators, the H1 prediction "~0 vs
  LR(N) ≈ +0.1" holds for Pythia-trained models. H1' needs the Herwig-trained models and is untested.
- **IRC-safe + N baseline.** EFP+N responds to the count: add_soft +0.055 to +0.076, split_hard +0.053 to +0.073, and ΔN = 0
  controls ≈ −0.01. The networks give the opposite pattern: add_soft ≈ 0, and split_hard ≈ its ΔN = 0 control. The hypothesis
  "networks = counting + IRC-safe shapes", with the EFP+N GBDT as its model, predicts a split-minus-control gap of +0.06 to +0.08.
  The networks give −0.019 to +0.011, so this prediction is excluded for the five networks on both generators. The PFN
  stand-in is the exception: its split − control gap of +0.09 to +0.10 and its rmsoft −0.09 match the counting baseline.
- **H10 (ii) z1 at fixed p_T^D.** For the five networks the effect is ≤ 0.006 per +0.01 of z1 in either generator. z1_up and
  z1_dn both give small positive shifts for ParT (non-monotone). The z1 term predicts ≤ 0.03 of the 0.08–0.22 split response.
- **H10 (i) p_T^D at fixed z1.** The global edit changes the angular energy distribution (girth −0.0045 per +0.01 of p_T^D). The
  IRC-safe EFN and the EFP-only GBDT respond as much as the networks (−0.60 to −0.68 per +0.01), so this arm measures
  sensitivity to energy flow, not to p_T^D specifically. It over-predicts the split by 11–35 ×. The local edit (particles 3–4,
  girth change < 0.0002, near on-manifold) gives −0.010 to −0.033 per +0.01 for the networks, with wide CIs (n = 262 jets with
  both directions). Its linear prediction for the split (+0.04 to +0.15) is the same order as the measured +0.08 to +0.22. It is
  not specific, though: the IRC-safe EFN responds −0.082 per +0.01 to the same edit and has split response 0, so a local momentum
  transfer at different positions also moves IRC-safe quantities. The edits used here cannot separate p_T^D from the angular
  energy flow.
- **H10 (iii) near-collinear leading pair.** Making the leading pair exactly collinear (ΔR → 0) minus the rotation control gives
  −0.075 to +0.011 for the networks, with PN kin the largest, and is off-manifold (C2ST 1.0). Setting ΔR to 0.02 gives +0.13 to
  +0.36 for the networks and +0.39 to +0.40 for the IRC-safe EFN, an angular response shared with the IRC-safe models. Exact-collinear
  pair proximity does not account for the split response.
- **Competing hypotheses.** (a) "Counting" (LR(N)-like): excluded for the networks on both generators. (b) "Counting + IRC-safe
  shapes" (EFP+N): excluded (see above). (c) "z1 at fixed p_T^D": excluded as the main term. (d) "p_T^D, through momentum sharing
  at the hard core": consistent in size for the local arm, but not separated from angular energy flow. (e) "Pair collinearity":
  excluded as the main term.

## Issues

- The Herwig retrains OOMKilled at 40 GiB on the first launch (`d-h14-hw-part`, `d-h14-hw-pnet`, 38 GB during in-memory
  loading). They were relaunched at 64 GiB (did not schedule within 15 min), then at 48 GiB as `d-h14-hw-part3` / `d-h14-hw-pnet3`.
  Smoke check (00:28 UTC): `d-h14-hw-part3` got through in-memory loading (peak 32.7 GB of 48 GiB) and is training on an RTX 3090
  (GPU 85 %, 15 GB, pod memory 19.7 GB). `d-h14-hw-pnet3` started at about 00:28 UTC and is loading. Expected time to finish is 4–6 h.
  When they finish: `./part_job.sh d-h14-run-hwm 'python -m interp.h14_qg_generator --step eval --set herwig --gen py && python -m
  interp.h14_qg_generator --step eval --set herwig --gen hw && python -m interp.h14_qg_generator --step eval --set herwig --gen py
  --edits extra && python -m interp.h14_qg_generator --step eval --set herwig --gen hw --edits extra'`, then `--step stats` and
  `--step fig`. `load_gen` already merges `herwig` logit files.
- The H2 PFN was OOMKilled after epoch 5 at 18 GiB and was resubmitted as `d-h14-pfn2` (28 GiB, 300 min cap). The PFN rows use
  the H5 400k-jet stand-in. When `models/qg/pfn/info.json` exists, `--set keras` adds `pfn_k` automatically.
- The `split_hard` and `pair_dr0` edits are off-manifold (C2ST 1.0). The global p_T^D edit (0.75–0.85) and z1_dn (0.71–0.73) are
  partly off-manifold. The add_soft, swap, local p_T^D and rotation edits are near on-manifold (0.51–0.55).
- The local p_T^D arm applies to few jets (262 with both directions), because ordering and positivity limit the momentum
  transfer between particles 3 and 4. Its up and down responses are asymmetric (up ≈ 0, down +0.02–0.04).
- The EFN_k port was checked against its own H2 test numbers: acc 0.809, AUC 0.8835 on this sample vs 0.809 / 0.8831.
- `h14_run_smoke.json` is from an earlier smoke run (1000 jets per class) and is superseded.

## Explainer

1. **Question.** Do quark–gluon taggers trained on one event generator (Pythia) rely on particle counting when tested on jets from
   another generator (Herwig)? Which property of the hardest particles makes them respond to a collinear split?
2. **Physics background.** Gluons radiate more than quarks (colour factors $C_A = 3$ vs $C_F = 4/3$), so gluon jets have more
   particles and spread their momentum more evenly. Particle multiplicity $N$ is IRC-unsafe and its modelling differs between
   generators: here $\langle N\rangle_g$ = 53.1 (Pythia) vs 48.7 (Herwig). $p_T^D = \sqrt{\sum_i z_i^2}$ with $z_i = p_{T,i}/\sum_j
   p_{T,j}$ is also IRC-unsafe and is large for quark jets.
3. **Method.** We apply input edits to fixed samples of 40,000 jets per generator and record the change in logit
   $\ell = \log p(g) - \log p(q)$. A count-only reference is $\mathrm{LR}(N) = \log \mathrm{NB}_g(N) - \log \mathrm{NB}_q(N)$
   (negative-binomial fits). An IRC-safe-plus-count reference is a GBDT on EFPs ($d \le 5$) and $N$. The H10 coefficients are
   $\beta = \sum_j(\Delta\ell^{\uparrow}_j - \Delta\ell^{\downarrow}_j)/\sum_j(\Delta x^{\uparrow}_j - \Delta x^{\downarrow}_j)$
   for $x \in \{p_T^D, z_1\}$, with the other quantity, $N$ and all particle positions held fixed.
4. **Result.** See `h14_generator.png`. Adding one realistic soft particle changes the networks by −0.024 to −0.002 on both
   generators, while $\mathrm{LR}(N)$ changes by +0.085 to +0.112. In `h14_hardcore.png`, $z_1$ at fixed $p_T^D$ has
   $|\beta| \le 0.6$ (per unit). Local $p_T^D$ changes give $\beta \approx -1$ to $-3$ per unit, which predicts a split response of
   +0.04 to +0.15 against a measured +0.08 to +0.22.
5. **Checks.** Each unedited model reproduces its known test accuracy. The ΔN = 0 controls (swap, merge+split) separate count
   effects from position or sharing effects. The permuted-weight ParT is the null. C2ST AUCs show which edits stay on the data
   manifold. The ported EFN matches its own test metrics.
6. **Physical significance.** The networks' insensitivity to an extra soft particle does not change with the test generator. A
   generator-dependent multiplicity is therefore not the quantity these networks use. Their split response comes from how
   momentum is shared among the hardest particles. Our edits do not separate this from the jet's angular energy flow, and models
   trained on Herwig are still to be measured.
