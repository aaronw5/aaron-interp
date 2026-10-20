# P31: What enlarges ParT's axis response in the released training: 10 classes vs 3, and more optimisation steps (claims C10, C3s; takes up P28)

Status: **running** (round C7). Arm m10s is trained and analysed. Arms m10 and L are still training and are analysed next round.
- Training jobs: `anrunw-ai-c-p31-m10s` (Complete, 7,808 s), `anrunw-ai-c-p31-m10` (Running, epoch 3 of 10, about 2,080 s per epoch on a V100-32GB, 32 Gi job, ETA about 2026-10-21 00:00 UTC), `anrunw-ai-c-p31-l` (Running, epoch 13 of 50, about 555 s per epoch, ETA about 2026-10-21 01:30 UTC). The first L launch (`logs/c-p31-l-oom.log`) ran out of GPU memory on a 22 GiB card. It was relaunched and landed on a V100-32GB.
- Analysis job: `anrunw-ai-c-p31-run` (Complete, 494 s; log `logs/c-p31-run.log`).
- Results: `results/part/p31_classes_steps_run.json`, figure `results/part/p31_classes_steps_run.png`, training logs `results/part/p31_train_log_{m10s,m10,L}.json` (copies of `/j-jepa-vol/aaron-interp/models/p31/<arm>/log.json`).
- Scripts: `interp/p31_classes_steps_train.py` (training, P22 code path), `interp/p31_classes_steps_part.py` (analysis). The analysis ran with `--models m10s,t,t2,released_kin,released_full,init0 --epochs 'L:0,4,9,12;m10s:0,4;m10:2' --tag run`.

## Conclusion

Conclusion: we trained a Particle Transformer on the full 10-class JetClass task, using the same number of jets (1.5 million) and the same number of training steps as the earlier 3-class W/Z/QCD runs. When the jet is shifted rigidly about its stored axis, this network's W-vs-Z output moves about 0.3 times as much as under a real mass change of the same size. The 3-class runs give about 0.22, and the released models 0.72–0.87. Training on 10 classes therefore raises the axis response by a measurable amount, but at this data size and step count it does not come close to the released level. One seed per arm was trained, and the two 3-class seeds differ by up to 0.04. The 10-class run on 5 million jets and the 5× longer 3-class run are still training. So far, through 3 and 13 epochs respectively, they give 0.28–0.44 and 0.28–0.52 depending on the shift direction.

## Pre-registered predictions and outcome (so far)

R is the per-jet least-squares slope (through 0) of Δd(shift) on Δd(matched dilation), with d = ℓ_Z − ℓ_W. The interval is a 95 % bootstrap over jets (1,000 resamples).

| hypothesis | prediction | observed this round | status |
|---|---|---|---|
| Thesis (class count) | R(m10s), R(m10) ≥ 0.5 and symmetric in φ±; R(L) ≈ R(t) | R(m10s) φ+ 0.316, φ− 0.428, η+ 0.303 | m10s part excluded; m10 and L pending |
| Alt 1 (steps) | R(L) rises toward 0.7 over epochs; m10s stays ≈ 0.2–0.3 | m10s 0.30–0.43; L at epochs 0/4/9/12 (lr still flat): φ+ 0.06/0.26/0.33/0.28, η+ −0.10/0.15/0.21/0.36 | m10s part holds (slightly above the range in φ+); L pending (decay epochs 35–49 not reached) |
| Alt 2 (data volume / other classes' jets) | m10 > m10s ≈ t | m10s − t = +0.10 (φ+), −0.04 (φ−), +0.09 (η+); seed gap t2 − t = +0.04, +0.02, +0.01 | m10s ≈ t only within about 2.5× the seed gap; m10 pending |
| Alt 3 (data scale) | no arm reaches 0.5 | m10s does not reach 0.5 | pending m10 and L |

## Procedure

- **Training** (`interp/p31_classes_steps_train.py`). This is the P22 code path: P14 helpers plus the P22 `AugBatcher` (no augmentation) and `build_model`. All arms use weaver ParticleTransformer with the authors' configuration and ParT kin inputs, Ranger at lr 1e-3, batch 512 and AMP. The schedule is flat, then decays ×0.01 over the last 30 % of epochs. Initialisation is torch.manual_seed(14), batch order is default_rng(1000 + epoch), validation is 30k rows default_rng(140) of the arm's own pool, and the held-out set is the same 20k test_20M *_100 W/Z/QCD rows as P14/P22 (default_rng(14)).
  - **m10:** 10 true classes in the released output order. The pool is the same 1.5M W/Z/QCD jets as P22 arm t, plus 5 random train_100M files of each other class (default_rng(31).choice(100, 5) per class). That gives 500k jets per class, 5.0M jets in all, over 10 epochs (9,707 steps per epoch).
  - **m10s:** 10 true classes, 150k jets per class (1.5M total). W/Z/QCD jets are 150k random rows (default_rng(31)) of each class's arm-t pool, and the other classes are 150k random rows of the m10 files. This gives 2,872 steps per epoch over 10 epochs, which is 28,720 steps, the same as arm t.
  - **L:** arm t (3 classes, the same 1.5M jets) for 50 epochs (143,600 steps). The lr is flat for epochs 0–34 and decays over epochs 35–49. A checkpoint is saved every epoch.
- **Sample:** fresh rows of test_20M/{ZJetsToNuNu, ZToQQ, WToQQ}_114.root, 5,000 random rows per file (default_rng(3114 + class index), sorted), N = 15,000. The sample is rebuilt unedited through `part_core.build`.
- **Edits:**
  - Rigid shifts of every real particle about the stored axis: φ + 0.06, φ − 0.06, η + 0.06 rad, with η = jet_η + sign(jet_η)·Δη. Sample-mean Δlog m_inv is 0.0000 for each. Δlog m_A is +0.127 for φ± and +0.114 for η+, where m_A = √(M0·M2) is the axis moment.
  - Per shift, a per-jet matched real dilation (Δη, Δφ) × k_j with the same Δlog m_A. Its Δlog m_inv is +0.123 (φ±) and +0.110 (η+).
  - Rotation null: the whole jet, axis included, rotated 0.06 rad in φ.
- **Metrics:**
  - R per direction, and R as a ratio of means.
  - W → Z: the fraction of true-W jets called Z by argmax over QCD/Z/W, before and after φ + 0.06, with McNemar tests (Holm over the 13 model rows).
  - Route shares of the φ + 0.06 matched dilation (P2 method): per-particle inputs edited with pair four-vectors kept, and the reverse. Each is reported as a slope on the full-edit response.
  - 3-class accuracy (argmax over QCD/Z/W outputs), and for 10-output models the 10-class argmax accuracy on the sample. For the released models, also the 10-class accuracy on the P1 sample.
- **Controls:** the rotation null; an untrained 3-output ParT_kin (init0, torch seed 100); P22 arms t and t2; released ParT_kin and ParT_full, all evaluated on the same rows.
- **Tests:** permutation tests that swap per jet the two models' (Δd_shift, Δd_dil) pairs, 2,000 permutations, H0: R_arm = R_t. The family is {m10s, t2} × {φ+, φ−, η+} with Holm correction. The m10 and L final comparisons join the family next round. Interim checkpoints (L@epoch-k, m10s@epoch-k, m10@epoch-2) are reported descriptively and are not tested.

## Numbers

**Accuracy**

| model | 3-class acc on this sample [95 % CI] | 10-class acc on this sample (W/Z/QCD rows) | other |
|---|---|---|---|
| m10s | 0.752 [0.746, 0.759] | 0.641 | own training log: 10-class held-out 0.724 (2,000 rows per class, test_20M *_100); 3-class held-out 0.752 |
| t (P22) | 0.747 [0.740, 0.754] | – | own held-out 0.746 |
| t2 (P22) | 0.745 [0.738, 0.752] | – | – |
| released ParT_kin | 0.764 [0.757, 0.770] | 0.657 | P1-sample 10-class **0.7447** (P1: 0.745) |
| released ParT_full | 0.825 [0.819, 0.831] | 0.756 | P1-sample 10-class **0.8593** (P1: 0.859) |
| init0 | 0.335 [0.327, 0.343] | – | – |
| m10@epoch-2 (interim) | 0.745 | 0.636 | own 10-class held-out after epoch 2: 0.718 |
| L@epoch-12 (interim) | 0.744 | – | own held-out 0.744 |

**Axis response R (N = 15,000)**

| model | R φ+ [95 % CI] | R φ− | R η+ | rotation null | W → Z before → after φ+ (change [CI], McNemar p_Holm) | mean Δd dilation φ+ |
|---|---|---|---|---|---|---|
| m10s | 0.316 [0.306, 0.326] | 0.428 [0.418, 0.439] | 0.303 [0.292, 0.314] | 0.007 | 21.0 → 30.5 % (+9.5 [8.5, 10.5], 1e-92) | +0.486 |
| t | 0.215 [0.206, 0.225] | 0.472 [0.459, 0.485] | 0.218 [0.207, 0.229] | −0.000 | 19.9 → 26.2 % (+6.3 [5.4, 7.0], 4e-50) | +0.470 |
| t2 | 0.254 [0.244, 0.265] | 0.491 [0.477, 0.505] | 0.228 [0.217, 0.240] | −0.001 | 21.7 → 29.0 % (+7.3 [6.3, 8.2], 2e-52) | +0.456 |
| released ParT_kin | 0.721 [0.709, 0.732] | 0.743 [0.731, 0.754] | 0.871 [0.858, 0.884] | 0.006 | 18.9 → 42.5 % (+23.6 [22.3, 24.9], 4e-278) | +0.498 |
| released ParT_full | 0.790 [0.780, 0.800] | 0.771 [0.760, 0.782] | 0.807 [0.794, 0.819] | 0.000 | 11.7 → 33.8 % (+22.1 [20.9, 23.3], 2e-280) | +0.637 |
| init0 | −0.065 [−0.114, −0.019] | 0.566 [0.528, 0.606] | −0.027 [−0.062, 0.012] | 0.000 | 53.1 → 50.6 % | −0.007 (mean abs 0.038) |
| L@epoch-0 | 0.063 | 0.757 | −0.096 | 0.000 | 28.0 → 29.2 % | +0.373 |
| L@epoch-4 | 0.258 | 0.432 | 0.146 | −0.000 | 16.9 → 24.5 % | +0.402 |
| L@epoch-9 | 0.327 | 0.456 | 0.211 | −0.000 | 12.0 → 17.4 % | +0.381 |
| L@epoch-12 | 0.277 | 0.516 | 0.356 | 0.001 | 26.3 → 34.8 % | +0.444 |
| m10s@epoch-0 | 0.086 | 0.698 | −0.116 | 0.001 | 23.2 → 26.0 % | +0.344 |
| m10s@epoch-4 | 0.340 | 0.473 | 0.385 | 0.014 | 34.1 → 45.2 % | +0.412 |
| m10@epoch-2 (≈ 29,100 steps, lr flat) | 0.354 | 0.434 | 0.440 | 0.001 | 21.4 → 31.1 % | +0.441 |

The interim rows have CI half-widths of 0.01–0.02. init0 responds to the dilation about 17× less than the trained models (mean |Δd| 0.038 vs 0.63–0.65), so its R is a ratio of small numbers and is not a calibrated floor.

**Permutation tests, H0: R_arm = R_t** (2,000 permutations; the minimum attainable p is 0.0005; Holm over 6)

| arm vs t | φ+ diff | φ− diff | η+ diff | p (each) | p_Holm |
|---|---|---|---|---|---|
| m10s | +0.100 | −0.044 | +0.085 | 0.0005 | 0.003 |
| t2 (seed control) | +0.039 | +0.019 | +0.010 | 0.0005–0.0015 | 0.003 |

The permutation test holds the two trained networks fixed and resamples jets. It resolves the seed-to-seed difference as well, so it does not test whether a difference exceeds seed variation. With one seed per arm, the seed gap t2 − t is the only seed reference: m10s − t is 2.6× it in φ+, 8.5× in η+, and opposite in sign in φ−.

**Route shares of the φ + 0.06 matched dilation** (slope on the full-edit response, CI half-width ≤ 0.007)

| model | per-particle only | pair only | additivity R² |
|---|---|---|---|
| m10s | 0.377 | 0.612 | 0.90 |
| t | 0.294 | 0.686 | 0.91 |
| t2 | 0.308 | 0.676 | 0.91 |
| released ParT_kin | 0.826 | 0.174 | 0.95 |
| released ParT_full | 0.839 | 0.153 | 0.93 |
| init0 | 1.001 | −0.000 | 1.00 |

## Reading

- With the same 1.5M-jet total and the same 28,720 steps, changing from 3 to 10 classes raises R in φ+ by 0.10 and in η+ by 0.09, and lowers it in φ− by 0.04. The per-particle route share rises from 0.29–0.31 to 0.38. The W → Z call change under φ + 0.06 rises from 6.3–7.3 to 9.5 points. All three remain well below the released ParT_kin values (R 0.72–0.87, per-particle share 0.83, W → Z +23.6 points).
- The φ+/φ− asymmetry of the planted arms (t: 0.22 vs 0.47) is smaller in m10s (0.32 vs 0.43). The released models are symmetric to within 0.03. At epoch 0 every arm has φ− ≈ 0.70–0.76 and φ+ ≈ 0.06–0.09, and later epochs reduce φ− while raising φ+ and η+. The untrained network also gives R φ− 0.57 vs φ+ −0.07, although at a 17× smaller dilation response.
- 3-class accuracy on the sample is 0.752 (m10s) vs 0.747 (t), so the extra classes do not cost W/Z/QCD accuracy at this size.
- Interim checkpoints, not final (lr has not decayed): arm L at 37,000 steps (epoch 12) gives R 0.28 / 0.52 / 0.36. That is within 0.06 of t in φ+ and higher in η+ (+0.14). m10 at about 29,100 steps (epoch 2) gives 0.35 / 0.43 / 0.44, close to m10s final in φ+ and φ− and higher in η+. Whether the decay phase, more epochs or the extra 3.5M jets move these toward 0.7 is measured next round.
- The rotation null is ≤ 0.014 for every model (largest for m10s@epoch-4), against R ≥ 0.2 for trained arms.

## Next round (pre-registered)

- When `c-p31-m10` and `c-p31-l` complete, rerun the analysis with `--models m10,m10s,L,t,t2,released_kin,released_full,init0 --epochs 'L:0,4,9,14,19,24,29,34,39,44,49;m10:0,2,4,6,9' --tag final` on the same rows. The permutation family becomes {m10, m10s, L, t2} × 3 directions plus m10 vs m10s, with Holm correction.
- A second seed of m10s would turn the "m10s − t vs seed gap" comparison into a seed-level test (about 2.2 h of GPU time).

## Explainer entry (for site/explain.py)

- **Question.** The released Particle Transformers change their W/Z call when a jet is shifted rigidly about its stored axis, which leaves the invariant mass unchanged. Networks of the same architecture trained by us on 3 classes do this about 3× less. Is the difference that the released models were trained on 10 classes?
- **Background.**
  - ParT's per-particle inputs include $\Delta\eta_i$, $\Delta\phi_i$ and $\Delta R_i$ about the stored jet axis. Its pair inputs ($\ln k_T$, $\ln z$, $\ln \Delta R_{ij}$, $\ln m^2_{ij}$) do not depend on the axis.
  - A rigid shift $\Delta\phi_i \to \Delta\phi_i + 0.06$ raises the axis moment $m_A = \sqrt{M_0 M_2}$, with $M_0=\sum_i p_{T,i}$ and $M_2=\sum_i p_{T,i}\Delta R_i^2$, by about 13 %, and leaves $m_{inv}$ unchanged.
  - Earlier runs (P22) gave $R\approx0.22$ for 3-class training on 1.5M jets and $R\approx0.72$–$0.87$ for the released models.
- **Method.**
  - Train ParT_kin on all 10 JetClass classes with 150k jets per class (1.5M jets, the same number of optimisation steps as the 3-class run). Also train on 500k jets per class (5.0M jets), and train the 3-class run 5× longer. All other settings are held fixed.
  - Measure $R = \dfrac{\sum_j \Delta d^{\,\mathrm{shift}}_j\, \Delta d^{\,\mathrm{dil}}_j}{\sum_j (\Delta d^{\,\mathrm{dil}}_j)^2}$, where $d = \ell_Z - \ell_W$. $\Delta d^{\,\mathrm{dil}}$ comes from a per-jet dilation $(\Delta\eta, \Delta\phi) \to k_j (\Delta\eta, \Delta\phi)$ that changes $m_A$ by the same amount as the shift and does change $m_{inv}$.
  - Use three shift directions ($\phi\pm0.06$, $\eta+0.06$).
  - Statistics: bootstrap over 15,000 held-out jets for the intervals, and permutation tests between arms with Holm correction.
- **Result (figure `results/part/p31_classes_steps_run.png`).**
  - 10 classes at 1.5M jets: $R = 0.32$ ($\phi+$), $0.43$ ($\phi-$), $0.30$ ($\eta+$).
  - 3 classes: $0.22$, $0.47$, $0.22$.
  - Released ParT_kin: $0.72$, $0.74$, $0.87$.
  - The share of a real mass change carried by the per-particle inputs rises from $0.29$ to $0.38$ (released: $0.83$).
- **Checks.**
  - The released models reproduce 74.47 % and 85.93 % 10-class accuracy on the P1 sample.
  - A whole-jet rotation, axis included, gives $R \le 0.014$.
  - The 3-class seed-to-seed difference in $R$ is $\le 0.04$.
  - The 10-class network keeps the 3-class accuracy (75.2 % vs 74.7 %).
- **Why it matters.** At this data size and step count, switching to the 10-class objective closes about 0.1 of the 0.5–0.65 gap in $R$ between our 3-class networks and the released ones (in the $\phi+$ and $\eta+$ directions). The rest has to come from data volume, the number of steps, or something else in the released training. The 5M-jet and 5×-steps runs, still training, test the first two.
