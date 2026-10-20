# P14 — Planted ground truth for the ParT methods: ParT trained on known mass labels (claims C3, C8; review weaknesses 2, 8), 2026-10-20

Status: **running**. Training jobs `anrunw-ai-c-p14-a` (arm a, then arm s), `anrunw-ai-c-p14-b` and `anrunw-ai-c-p14-c` were launched and smoke-checked.
Analyses done this round: the released ParT_full / ParT_kin control (`c-p14-ctl`) and ARMC_STATUS.
Arms a, b and s: analysis next round with `./part_job.sh c-p14-run 'python -m interp.p14_planted_part --models a,b,c,s,released_full,released_kin'`.
That run also gives the permutation tests R_a = R_b, R_a = R_s and R_a = R_c.

Scripts:
- `interp/p14_planted_train.py` (labels side file and training)
- `interp/p14_planted_part.py` (tests 1–3, figure)
- `interp/p14_eap_part.py` (test 4: P9's EAP-IG graph reused for 2-output kin models)

Results: `results/part/p14_planted_released.json` and `.png`, ARMC_FILES, `p14_planted_smoke.json`.
Side file: `/j-jepa-vol/aaron-interp/data/p14/labels.npz`.
Checkpoints: `/j-jepa-vol/aaron-interp/models/p14/<arm>/epoch-<e>.pt`, `final.pt`, `log.json`.
Logs: `/j-jepa-vol/aaron-interp/logs/c-p14-{a,b,c,ctl,run-c,smoke4,smoke5}.log`.

## Conclusion
CONCLUSION_TEXT

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
ARMC_UNEDITED

### Test 1: φ + 0.06 rad (invariant mass unchanged) vs matched real dilation
| model | R [95 % CI] | rot null R | R_probe [CI] | calls changed phi / dil / rot | positive (W) call before → after phi, McNemar |
|---|---|---|---|---|---|
| released ParT_full | 0.790 [0.781, 0.799] | −0.0001 | 0.753 [0.737, 0.769] | 18.5 / 20.6 / 0.03 % | 0.338 → 0.243, n10 2,175 vs n01 278, p_Holm < 1e-300 |
| released ParT_kin | 0.723 [0.714, 0.733] | 0.006 | 1.559 [1.525, 1.590] | 31.0 / 26.8 / 0.81 % | 0.320 → 0.203, 2,645 vs 298, p_Holm < 1e-300 |
ARMC_T1

On true-W jets only, R = 0.787 (full) and 0.721 (kin), consistent with P10's 0.76 / 0.69.

### Test 2: route split for the matched dilation (and angles × 1.1)
| model | per-particle share [CI] | pair share [CI] | sum | additivity R² | phi: mean abs Δd pair-only vs full vs rot null |
|---|---|---|---|---|---|
| released ParT_full | 0.834 [0.829, 0.840] (k1.1: 0.850) | 0.153 [0.148, 0.158] (0.146) | 0.987 | 0.93 | 0.0016 vs 0.80 vs 0.0015 |
| released ParT_kin | 0.832 [0.826, 0.839] (0.848) | 0.168 [0.162, 0.175] (0.160) | 1.001 | 0.95 | 0.016 vs 0.65 vs 0.015 |
ARMC_T2

### Test 3: DAS (arms only)
ARMC_T3

### Test 4: EAP-IG circuits
| model | unedited acc (eval jets) | corrupted acc | edges at F = 0.8 | nodes at F = 0.8 | smallest literal node circuit with F ≥ 0.8 | its F [CI], accuracy [CI] | random circuits (20): F, accuracy | McNemar vs full |
|---|---|---|---|---|---|---|---|---|
| released ParT_kin (W vs Z, 3,000 jets) | 0.754 | 0.255 | 7,287 (33.9 %) | 197 | top-256 nodes, 12,321 edges | 0.898 [0.887, 0.909], 0.732 [0.715, 0.747] | 0.139 [0.003, 0.393], 0.307 (max 0.455); p_perm 0.048 | right→wrong 166, wrong→right 101, p 8e-5 |
ARMC_T4

For released ParT_kin, the literal circuit calls W for 75.7 % of true W and 29.4 % of true Z, against 75.4 % accuracy for the full model on these jets. The completeness of EAP-IG (ΣIG / ΔD) is 1.05.

## Reading
READING_TEXT

## Issues
- Arms a, b and s are not analysed this round, so the decisive test (R_a) and the permutation tests are open.
- The kin input set replaces the full set (see Procedure 3).
- The arm-a and arm-b labels agree on 97.5 % of jets. Accuracy against the arm's own label separates them only weakly; the phi edit, which flips 26 % of the b labels and 0 % of the a labels, is the discriminating test.
- ParT's pair features are not rotation-invariant at float32 precision. A rigid rotation changes 0.03 % (full) and 0.8 % (kin) of calls. For an arm with only the pair route, the phi response therefore has a floor equal to the rot null rather than 0.
- R_probe for released ParT_kin (1.56) differs from its logit R (0.72). The log m_inv ridge probe on the class token responds more to phi than to the matched dilation. The logit R is the primary measure.
- Test 4 uses 1,500 / 3,000 jets and 20 random circuits. The p_perm floor is 1/21 = 0.048.

## Explainer entry (for site/explain.py)
EXPLAINER_TEXT
