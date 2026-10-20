# P14 — Planted ground truth for the ParT methods: ParT trained on known mass labels (claims C3, C8; review weaknesses 2, 8), 2026-10-20

Status: **done** (arm analysis run in P16 as `c-p16-run` / `c-p16-eap`, 2026-10-20; see `inbox_c/P16_report.md`). Training jobs `anrunw-ai-c-p14-a` (arm a, then arm s), `anrunw-ai-c-p14-b` and `anrunw-ai-c-p14-c` completed.
Analyses done this round: the released ParT_full / ParT_kin control (`c-p14-ctl`) and the arm analysis (P16: `c-p16-run`, `c-p16-eap`).
Arms a, b and s: analysis next round with `./part_job.sh c-p14-run 'python -m interp.p14_planted_part --models a,b,c,s,released_full,released_kin'`.
That run also gives the permutation tests R_a = R_b, R_a = R_s and R_a = R_c.

Scripts:
- `interp/p14_planted_train.py` (labels side file and training)
- `interp/p14_planted_part.py` (tests 1–3, figure)
- `interp/p14_eap_part.py` (test 4: P9's EAP-IG graph reused for 2-output kin models)

Results: `results/part/p14_planted_released.json` and `.png`, `p14_planted_p16.json`, `p14_planted_p16eap.json`, `p14_planted_p16.png`, `p14_planted_smoke.json`.
Side file: `/j-jepa-vol/aaron-interp/data/p14/labels.npz`.
Checkpoints: `/j-jepa-vol/aaron-interp/models/p14/<arm>/epoch-<e>.pt`, `final.pt`, `log.json`.
Logs: `/j-jepa-vol/aaron-interp/logs/c-p14-{a,b,c,ctl,run-c,smoke4,smoke5}.log`.

## Conclusion
We trained copies of the Particle Transformer (ParT) to flag jets whose mass falls in a fixed window.
- One copy was trained with a label defined by the true invariant mass.
- One copy was trained with a label defined by the axis-relative mass m_A, a mass-like quantity built from each particle's angular distance to the stored jet axis.

Both copies respond to a rotation that leaves the invariant mass unchanged at about 0.2 of their response to an equal real mass change. The released ParT models respond at 0.72–0.79. Both copies compute most of the mass from the pairwise particle features, while the released models use the per-particle angle inputs.

This measurement excludes two explanations: that the architecture and inputs alone force an axis-relative reading, and that a label alone sets which mass the network reads (the m_A-label copy does not learn m_A). The released models' larger response therefore comes from their training task or data: 10 classes and 100M jets, against 3 classes and 1.5M jets here. This experiment does not separate those two factors.

## Procedure
1. **Labels** (`p14_planted_train.make_labels`). Two masses are computed from the stored constituents of each jet:
   - m_inv: the invariant mass of the four-vector sum.
   - m_A = √(Σ pT · Σ pT ΔR²), with ΔR measured from the stored jet axis.

   The labels are:
   - **Arm a:** 1[|m_inv − 85| < 10 GeV].
   - **Arm b:** 1[|m_A − c| < w]. c = 84.62 GeV is the median m_A of the arm-a-positive training jets. w = 9.98 GeV is found by bisection so that the training positive rate equals arm a's (0.4137).
   - **Arm c:** the arm-a labels. The per-particle channels that carry position are set to 0 after scaling: deltaR, deta, dphi, and also e_log and logerel, which depend on η through E = pT cosh η. Only pt_log and logptrel stay, so geometry reaches the network only through the pair features (ln kT, ln z, ln ΔR, ln m²) built from the four-vectors.
   - **Arm s:** the arm-a labels permuted over the training jets (`default_rng(14)`), trained for 2 epochs.

   Positive rate by class (QCD / Z / W) is 0.132 / 0.499 / 0.611 for a and 0.128 / 0.503 / 0.610 for b. Labels a and b agree on 97.5 % of the training jets, and corr(m_inv, m_A) = 0.999. The two labels therefore differ only near the window edges; the tests below separate them by edits that move m_A but not m_inv.
2. **Training data.** JetClass `train_100M`, 5 random files per class (`default_rng(14).choice(100, 5)`) of WToQQ, ZToQQ and ZJetsToNuNu, 1.5M jets, held in memory.
   - 30,000 random pool rows are held out for per-epoch validation.
   - Held-out test set: 20,000 random rows (`default_rng(14)`) of the union of `test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_100.root` (6,666 / 6,659 / 6,675). This is the analysis sample.
3. **Model and optimiser.** weaver-core 0.4.17 ParticleTransformer with the authors' `example_ParticleTransformer.py` configuration and num_classes = 2. Settings follow `train_JetClass.sh` for ParT: Ranger, lr 1e-3, batch 512, AMP, CrossEntropy, and the flat+decay schedule (×0.01^(1/3) at epochs 7, 8, 9 of 10). Training uses the network's own random trimming.
   - **Deviation from the brief:** the kin input set (7 per-particle features plus four-vectors) is used instead of the full set. The labels are kinematic, and the 17-feature arrays would double the memory. The released ParT_full and ParT_kin are both in the control.
4. **Jobs.**
   - `interp/p10_part_seeds_job.sh` was used with P10_CPU=3, P10_MEM=16 and GPUs ≥ 22 GB.
   - A batch-512 smoke run on an 11 GB GTX 1080 Ti ran out of GPU memory (`c-p14-smoke3`). The training needs 18–21 GiB allocated.
   - **Smoke checks:** `c-p14-smoke4` (batch 128, 21k jets, train a and c, then the analysis on a, c and released_kin) and `c-p14-smoke5` (analysis with the EAP step) both exited 0.
   - Two bugs were fixed before the main launch: Ranger's `zero_grad` takes no `set_to_none`, and the matched dilation divided by M0M2 = 0.
   - **Measured throughput:**
     - Arm c: 6.0–6.5 steps/s, GPU 92–100 %, 7.7 min per epoch.
     - Arms a and b: 3.7 steps/s, GPU 82–86 %, 12.5 min per epoch.
5. **Analysis sample and edits** (`p14_planted_part.py`; all edits go through `part_core.build`). The sample is the 20,000 held-out jets; the baseline is these jets rebuilt unedited.
   - **phi:** every particle's Δφ + 0.06 rad about the stored axis. Mean Δlog m_inv = −1e-8 and Δlog m_A = +0.125. The arm-b label flips for 26.1 % of jets and the arm-a label for 0 %.
   - **dil:** per-jet matched dilation, (Δη, Δφ) × k_j with k_j = √(M0M2(phi)/M0M2(base)). Δlog m_A is +0.125 per jet as for phi, and Δlog m_inv = +0.121.
   - **k1.1:** angles × 1.1.
   - **rot:** the whole jet including the axis rotated by 0.06 rad (null).
6. **Tests.**
   - **Read-out:** planted arms use d = logit_1 − logit_0; released models use d = logit_W − logit_Z.
   - **Test 1:** R = per-jet least-squares slope of Δd(phi) on Δd(dil), with a 2,000-resample bootstrap over jets. H_inv (the network reads m_inv) predicts R = 0; H_axis (it reads m_A) predicts R = 1. The null is the same slope for the rot edit.
     - The probe version, R_probe, uses a ridge read-out of log m_inv from the class token (P1 procedure, fitted on the unedited train half) and takes mean Δ(phi) / mean Δ(dil) on the test half.
     - Calls changed are given with Wilson CIs, and McNemar tests compare the positive (or W) call before and after each edit (Holm-corrected).
   - **Test 2:** per-particle route (x_edit, v_base) and pair route (x_base, v_edit). The share is the per-jet slope of each route's Δd on the full Δd, with bootstrap CIs. For phi, the pair route is compared with the rot null, because both are float-level rotations of the four-vectors.
   - **Test 3 (arms only):** DAS with a 1-dimensional direction u on the jet-mean token at the outputs of particle blocks 2 and 8. The base jet's mean projection is set to the source jet's. u is trained with Adam, 400 steps of 128 pairs, so that the intervened call equals the window of the source's m_inv or m_A. IIA is measured on all held-out pairs and on the discriminating pairs (where window_minv(src) ≠ window_mA(src)). The controls are 4 random directions and u = 0.
   - **Test 4:** P9's graph (433 nodes, 21,504 edges) with P9's corruption B (donor = one of the 5 nearest-pT jets with the other label). EAP-IG uses 1,500 discovery jets (train half) and evaluation uses 3,000 jets (test half). Outputs are the edges and nodes needed for F = 0.8 (log-interpolated), the smallest top-k node-induced literal circuit with F ≥ 0.8 (accuracy, confusion, McNemar vs the full model), and 20 random circuits with the same node-category counts.

## Numbers
### Unedited models on the analysis sample
| model | metric | value |
|---|---|---|
| released ParT_full | 10-class accuracy, P1 sample (rows 3,000–4,000, N = 10,000) | 0.8593 (P1: 0.859) |
| released ParT_kin | same | 0.7447 (P1: 0.745) |
| released ParT_full | 10-class / argmax over {QCD, Z, W} on the 20,000 W/Z/QCD jets; true W called Z | 0.757 / 0.824; 0.106 |
| released ParT_kin | same | 0.659 / 0.764; 0.157 |
(Arm rows filled from P16, `inbox_c/P16_report.md`; released rows above are from c-p14-ctl and were reproduced exactly by c-p16-run.)

| model | sample | accuracy [95 % CI] |
|---|---|---|
| released ParT_full | P1 sample, 10-class (N = 10,000) | 0.8593 (P1: 0.859) |
| released ParT_kin | same | 0.7447 (P1: 0.745) |
| released ParT_full / ParT_kin | 20,000 W/Z/QCD jets, 10-class; argmax over {QCD, Z, W} | 0.757 / 0.659; 0.824 / 0.764 |
| arm a (m_inv window) | 20,000 jets, own label | 0.9805 [0.978, 0.982] (QCD 0.990, Z 0.973, W 0.978); vs label b: 0.970 |
| arm b (m_A window) | same | 0.9805 [0.978, 0.982] (0.991, 0.971, 0.979); vs label a: 0.966 |
| arm c (m_inv, pair route only) | same | 0.9775 [0.975, 0.979]; vs label b: 0.968 |
| arm s (shuffled) | same | 0.588. Calls "outside" for every jet, a constant output. |

### Test 1: φ + 0.06 rad (invariant mass unchanged) vs matched real dilation
| model | R [95 % CI] | rot null R | R_probe [CI] | calls changed phi / dil / rot | positive (W) call before → after phi, McNemar |
|---|---|---|---|---|---|
| released ParT_full | 0.790 [0.781, 0.799] | −0.0001 | 0.753 [0.737, 0.769] | 18.5 / 20.6 / 0.03 % | 0.338 → 0.243, n10 2,175 vs n01 278, p_Holm < 1e-300 |
| released ParT_kin | 0.723 [0.714, 0.733] | 0.006 | 1.559 [1.525, 1.590] | 31.0 / 26.8 / 0.81 % | 0.320 → 0.203, 2,645 vs 298, p_Holm < 1e-300 |
Planted arms and released models on the same jets (P16):

| model | R [95 % CI] | rot null R | R_probe [CI] | calls changed phi / dil / rot (%) | positive call before → after phi; McNemar n10 / n01, p_Holm | own-label accuracy after phi |
|---|---|---|---|---|---|---|
| a | **0.185 [0.181, 0.189]** | 0.0001 | 0.158 [0.146, 0.171] | 5.85 [5.53, 6.18] / 25.7 / 0.02 | 0.418 → 0.415; 620 / 549, p_Holm 0.49 | 0.9805 → 0.9505 (labels unchanged) |
| b | **0.215 [0.211, 0.220]** | 0.0002 | 0.176 [0.165, 0.187] | 6.25 [5.93, 6.60] / 25.7 / 0.02 | 0.419 → 0.418; 640 / 611, p_Holm 1 | 0.9805 → 0.7726 (against the edited b labels, 26.1 % flipped) |
| c | 0.0000 [−0.0000, 0.0000] | 0.0002 | −0.000 | 0.03 / 25.5 / 0.04 | 0.422 → 0.422; 5 / 0, p_Holm 0.69 | 0.9776 → 0.9776 |
| s | −0.13 [−0.18, −0.08] on mean \|Δd\| 3.5×10⁻⁴ | 0.044 | −0.87 | 0 / 0 / 0 | constant call | – |
| released ParT_full | 0.790 [0.781, 0.799] | −0.0001 | 0.753 [0.737, 0.769] | 18.5 / 20.6 / 0.03 | W 0.338 → 0.243; 2,175 / 278, p_Holm < 1e-300 | – |
| released ParT_kin | 0.723 [0.714, 0.733] | 0.006 | 1.559 [1.525, 1.590] | 31.0 / 26.8 / 0.81 | W 0.320 → 0.203; 2,645 / 298, p_Holm < 1e-300 | – |

- **Calls under dil, all three arms:** about 3,240 calls go positive → negative and about 1,890 go negative → positive, p_Holm < 1e-76.
- **Permutation tests** (2,000 permutations, difference of R):

  | comparison | difference of R | p |
  |---|---|---|
  | R_a − R_b | −0.031 | 0.0005 |
  | R_a − R_c | +0.185 | 0.0005 |
  | R_a − R_s | +0.315 | 0.0005 |
  | R_b − R_c | +0.215 | 0.0005 |
  | R_a − R_released_kin | −0.538 | 0.0005 |
  | R_a − R_released_full | −0.605 | 0.0005 |

  0.0005 is the floor of the test.

On true-W jets only, R = 0.787 (full) and 0.721 (kin), consistent with P10's 0.76 / 0.69.

### Test 2: route split for the matched dilation (and angles × 1.1)
| model | per-particle share [CI] | pair share [CI] | sum | additivity R² | phi: mean abs Δd pair-only vs full vs rot null |
|---|---|---|---|---|---|
| released ParT_full | 0.834 [0.829, 0.840] (k1.1: 0.850) | 0.153 [0.148, 0.158] (0.146) | 0.987 | 0.93 | 0.0016 vs 0.80 vs 0.0015 |
| released ParT_kin | 0.832 [0.826, 0.839] (0.848) | 0.168 [0.162, 0.175] (0.160) | 1.001 | 0.95 | 0.016 vs 0.65 vs 0.015 |
Planted arms (P16):

| model | per-particle share [CI] | pair share [CI] | sum | additivity R² | k1.1 per-particle / pair | phi: mean \|Δd\| pair-only / full / rot null; per-particle slope |
|---|---|---|---|---|---|---|
| a | 0.172 [0.170, 0.175] | 0.830 [0.827, 0.833] | 1.003 | 0.94 | 0.204 / 0.810 | 0.0033 / 1.20 / 0.0031; 1.000 |
| b | 0.211 [0.208, 0.214] | 0.796 [0.792, 0.799] | 1.007 | 0.92 | 0.243 / 0.777 | 0.0055 / 1.37 / 0.0052; 1.000 |
| c | −0.005 [−0.005, −0.005] | 1.005 [1.005, 1.005] | 1.000 | 1.00 | −0.008 / 1.007 | 0.0068 / 0.0068 / 0.0063; 0.000 |
| s | 0.63 [0.61, 0.64] | 0.37 [0.35, 0.38] | 0.997 | 0.99 | 0.60 / 0.40 | responses ≤ 3.5×10⁻⁴ |
| released ParT_full | 0.834 [0.829, 0.840] | 0.153 [0.148, 0.158] | 0.987 | 0.93 | 0.850 / 0.146 | 0.0016 / 0.80 / 0.0015; 1.000 |
| released ParT_kin | 0.832 [0.826, 0.839] | 0.168 [0.162, 0.175] | 1.001 | 0.95 | 0.848 / 0.160 | 0.016 / 0.65 / 0.015; 0.986 |

### Test 3: DAS (arms only)
The table shows IIA on all pairs (10,000) and, on the discriminating pairs (234), the rate at which the call follows the m_inv-window label of the source.

| arm, block | u trained for m_inv: IIA all; disc → m_inv | u trained for m_A: IIA all; disc → m_A | random u / u = 0: IIA all; disc → m_inv | paired McNemar (own target, disc) | cos(u_minv, u_mA) |
|---|---|---|---|---|---|
| a, 2 | 0.566 [0.556, 0.576]; 0.504 | 0.523; 0.496 | 0.505 / 0.505; 0.479 | 105 / 103, p 0.94 | 0.32 |
| a, 8 | 0.729 [0.720, 0.738]; 0.573 [0.509, 0.637] | 0.769; 0.397 | 0.517 / 0.518; 0.534 | 97 / 56, p 0.0012 (Holm over 8 tests: 0.009) | 0.56 |
| b, 2 | 0.567; 0.513 | 0.568; 0.479 | 0.520 / 0.520; 0.526 | 120 / 112, p 0.65 | −1.00 |
| b, 8 | 0.768; 0.449 | 0.729; 0.513 | 0.517 / 0.515; 0.440 | 58 / 73, p 0.22 | 0.60 |
| c, 2 | 0.520; 0.449 | 0.522; 0.547 | 0.520 / 0.520; 0.474 | 105 / 128, p 0.15 | 1.00 |
| c, 8 | 0.665; 0.491 | 0.820; 0.500 | 0.514 / 0.513; 0.466 | 74 / 76, p 0.93 | 0.46 |
| s, 2 and 8 | 0.585 (equal to u = 0) | 0.585 | 0.585 | 126 / 108, p 0.27 | – |

On the discriminating pairs, u = 0 already gives 0.44–0.53 against the m_inv target. A direction that carried only m_inv or only m_A would give values near 1.

### Test 4: EAP-IG circuits
| model | unedited acc (eval jets) | corrupted acc | edges at F = 0.8 | nodes at F = 0.8 | smallest literal node circuit with F ≥ 0.8 | its F [CI], accuracy [CI] | random circuits (20): F, accuracy | McNemar vs full |
|---|---|---|---|---|---|---|---|---|
| released ParT_kin (W vs Z, 3,000 jets) | 0.754 | 0.255 | 7,287 (33.9 %) | 197 | top-256 nodes, 12,321 edges | 0.898 [0.887, 0.909], 0.732 [0.715, 0.747] | 0.139 [0.003, 0.393], 0.307 (max 0.455); p_perm 0.048 | right→wrong 166, wrong→right 101, p 8e-5 |
Planted arms (P16):

| arm | unedited acc | corrupted acc | edges at F = 0.8 | nodes at F = 0.8 | literal top-256-node circuit: edges, F [CI], accuracy [CI] | 20 random circuits: F, accuracy (max) | McNemar vs full (right→wrong / wrong→right) | IG completeness |
|---|---|---|---|---|---|---|---|---|
| a | 0.984 | 0.016 | 6,423 (29.9 %) | 196 | 9,901; 0.948 [0.944, 0.951]; 0.932 [0.922, 0.940] | 0.229 [0.069, 0.428], 0.176 (0.556); p_perm 0.048 | 164 / 8, p 6e-39 | 1.07 |
| b | 0.981 | 0.018 | 7,012 (32.6 %) | 196 | 10,078; 0.946 [0.942, 0.950]; 0.943 [0.935, 0.951] | 0.210 [0.072, 0.341], 0.141 (0.277); p_perm 0.048 | 120 / 7, p 1e-27 | 1.06 |
| c | 0.977 | 0.021 | 6,079 (28.3 %) | 210 | 9,559; 0.939 [0.933, 0.944]; 0.923 [0.913, 0.932] | 0.206 [0.085, 0.341], 0.137 (0.343); p_perm 0.048 | 187 / 26, p 3e-31 | 1.06 |
| released ParT_kin (P14, W vs Z) | 0.754 | 0.255 | 7,287 (33.9 %) | 197 | 12,321; 0.898; 0.732 | 0.139, 0.307 | 166 / 101, p 8e-5 | 1.05 |

- Literal circuit calls by class: arm a calls "in window" for 13.8 % of QCD, 53.3 % of Z and 61.2 % of W jets; arm b: 13.4 / 53.4 / 61.0 %; arm c: 14.2 / 48.3 / 60.9 %.
- Top-40 node categories for arms a and b: 15 class-token coordinates (z), 7 head nodes of class block 0 (C0H), 2 MLP-group nodes of class block 0 (C0M), the embedding (E) and 4 head nodes of particle block 0 (P0H). The rest are particle-block MLP groups.
- Arm c has no E and no P0H among its top 40. It has 5 head nodes of class block 1 (C1H).

For released ParT_kin, the literal circuit calls W for 75.7 % of true W and 29.4 % of true Z, against 75.4 % accuracy for the full model on these jets. The completeness of EAP-IG (ΣIG / ΔD) is 1.05.

## Reading
- **Hypotheses.**
  - H_data predicted R_a ≈ 0, R_b ≈ 1 and R_c ≈ 0.
  - H_arch (architecture and inputs force an axis-moment reading) predicted R_a ≈ 0.5–0.9, near the released 0.72.
  - Both predicted R_s at the rot null.
- **Observed R.** R_a = 0.185, R_b = 0.215, R_c = 0.000.
  - **H_arch:** its prediction for a is excluded. R_a lies 0.54 below R_released_kin, permutation p = 0.0005.
  - **H_data:** its prediction R_b ≈ 1 is excluded.
  - **Arms a and b:** R_a and R_b differ by 0.03. With N = 20,000 the difference is significant (p = 0.0005), but it is small relative to the 0 vs 1 spread the test was designed for.
  - **Arm s:** learned a constant output (all jets "outside"), so its R is the ratio of two numerical-noise responses (mean |Δd| ≤ 3.5×10⁻⁴). It gives no information about R.
- **Arm b and its own label.** Arm b does not follow its own label under phi: own-label accuracy drops from 0.98 to 0.77 when 26 % of b labels flip, and only 6.3 % of its calls change. Labels a and b agree on 97.6 % of training jets. The network fits label b to 0.98 with a mostly rotation-invariant mass. Its accuracy on label a (0.966) is close to its accuracy on label b (0.980).
- **Arm a residual.** Arm a keeps a residual response to the mass-invariant shift: 5.9 % of calls change, in both directions (620 out of the window, 549 into it). Own-label accuracy falls from 0.9805 to 0.9505 although no label changes. The phi response of arm a runs entirely through the per-particle inputs (per-particle slope 1.000; pair-only response equal to the rot null).
- **Route of a real mass change.** In arms a and b, 80–83 % of the matched dilation's read-out change comes through the pair features (ln m² etc.) and 17–21 % through the per-particle angles. In the released models the shares are reversed (per-particle 0.83, pair 0.15–0.17).
  - The per-particle share of arms a and b (0.17–0.21) is close to their R (0.19–0.22). This is consistent with the per-particle route carrying an axis-relative quantity and the pair route an invariant one.
  - Arm c (no per-particle position) shows R = 0 and pair share 1.00, as constructed. Its phi response equals the rot null (0.0068 vs 0.0063), so the pair features are rotation-invariant up to float rounding.
- **Test 3 (DAS).** The 1-dimensional jet-mean DAS reaches IIA 0.67–0.82 at block 8 and 0.52–0.57 at block 2, against 0.51–0.52 for random directions.
  - On the 234 discriminating pairs it does not separate m_inv from m_A in arms b and c (p ≥ 0.15).
  - In arm a at block 8, the m_inv-trained direction reaches its target on 57 % of discriminating pairs and the m_A-trained direction on 40 %, p_Holm 0.009. The u = 0 baseline is 53 %, so this is a 4-point gain over no intervention.
  - The tool did not recover the planted variable at this sample size.
- **Test 4 (EAP-IG).** The tool finds a literal circuit of 256 of 433 nodes (about 46 % of edges) that keeps F ≈ 0.94–0.95 and accuracy 0.92–0.94 in all three arms, against F ≈ 0.21–0.23 for random circuits of the same composition.
  - The node rankings of arms a and b are almost identical in composition. Arm c lacks the embedding and particle-block-0 head nodes, consistent with its zeroed per-particle positions.
  - EAP-IG at this resolution does not distinguish an m_inv-label network from an m_A-label network.

## Issues
- Arms a, b, c and s were analysed in P16 (R_a 0.185, R_b 0.215, R_c 0.000; arm s learned a constant output). See P16_report Issues for the open points.
- The kin input set replaces the full set (see Procedure 3).
- The arm-a and arm-b labels agree on 97.5 % of jets. Accuracy against the arm's own label separates them only weakly; the phi edit, which flips 26 % of the b labels and 0 % of the a labels, is the discriminating test.
- ParT's pair features are not rotation-invariant at float32 precision. A rigid rotation changes 0.03 % (full) and 0.8 % (kin) of calls. For an arm with only the pair route, the phi response therefore has a floor equal to the rot null rather than 0.
- R_probe for released ParT_kin (1.56) differs from its logit R (0.72). The log m_inv ridge probe on the class token responds more to phi than to the matched dilation. The logit R is the primary measure.
- Test 4 uses 1,500 / 3,000 jets and 20 random circuits. The p_perm floor is 1/21 = 0.048.

## Explainer entry (for site/explain.py)
- **Question.** ParT's mass response is partly axis-relative: a rigid φ shift that keeps the invariant mass fixed moves W calls to Z. Is this forced by the architecture and inputs, or does it come from what the network was trained on?
- **Background.**
  - The invariant mass is $m^2=(\sum_i p_i)^2$.
  - The axis-moment mass is $m_A=\sqrt{\sum_i p_{T,i}\,\sum_i p_{T,i}\Delta R_i^2}$, where $\Delta R_i$ is measured from the stored jet axis.
  - They agree for real jets (correlation 0.999), but a rotation of all particles about the axis by $\delta\phi$ changes $m_A$ and not $m$.
  - Released ParT responds to that rotation at 0.72–0.79 of its response to a real mass change of equal $\Delta\log m_A$ (P1, P14 control).
- **Method.**
  - Train ParT from scratch on W/Z/QCD jets with labels that are known functions of one mass:
    - a: $1[|m-85|<10]$;
    - b: $1[|m_A-84.6|<10]$;
    - c: label a with the per-particle angles removed;
    - s: shuffled labels.
  - For each network, compare $\Delta d$ under the rotation (φ + 0.06 rad) with $\Delta d$ under a per-jet dilation scaled so that $\Delta\log m_A$ is equal: $R=\frac{\sum_j \Delta d_j(\phi)\,\Delta d_j(\mathrm{dil})}{\sum_j \Delta d_j(\mathrm{dil})^2}$. $R=0$ means the network reads $m$; $R=1$ means it reads $m_A$.
  - Split each response into the part through the per-particle inputs and the part through the pairwise features.
- **Result (figure p14_planted_p16.png).**
  - $R_a=0.185$ and $R_b=0.215$, against 0.79 / 0.72 for the released models. $R_c=0$, and rigid rotation gives $R\approx 10^{-4}$.
  - The trained-from-scratch networks take 80–83 % of a real mass change through the pair features. The released ones take 83 % through the per-particle angles.
  - The $m_A$-label network does not follow $m_A$: when the rotation flips 26 % of its labels, its accuracy on them falls from 0.98 to 0.77.
- **Checks.**
  - The released models reproduce their P1 accuracies (0.8593 / 0.7447).
  - Arm accuracies on held-out jets are 0.98.
  - The rotation changes $\log m$ by $10^{-8}$.
  - Route shares sum to 1.00 (additivity $R^2$ ≥ 0.92).
  - EAP-IG completeness is 1.06–1.07.
  - Permutation tests between arms give $p=0.0005$.
- **Why it matters.**
  - ParT with these inputs can learn a nearly rotation-invariant mass. The released models' strong axis-relative response is therefore not forced by the architecture and inputs.
  - Labels alone did not steer the network toward $m_A$ either: the label-b network also read mostly $m$.
  - The difference must come from the 10-class task or the training data, which a follow-up has to separate.
  - The DAS and EAP tools did not distinguish the two planted variables at this sample size.
