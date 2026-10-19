# H2: one-signal datasets (Top Tagging Landscape, Quark–Gluon): released weights, loaders, classical references, trainings launched

## Conclusion
On the full Top Landscape test set, the best classical top tagger built here reaches a QCD rejection of 305 at 30 % top efficiency. It is a likelihood ratio of soft-drop mass, $\tau_{32}$ and the best subjet-pair mass $m_W$. That is about 5 times below the published ParT (1602), so the networks to be edited next round use information beyond these three observables. Adding $m_W$ to soft-drop mass and $\tau_{32}$ raises the classical rejection by about 28 %, so the W mass inside the top is useful information that a tagger could use. Whether ParT or ParticleNet actually use it is the D6 edit planned for next round. On the quark–gluon test set, gluon jets have 1.59 times as many constituents as quark jets, below the leading-order Casimir ratio $C_A/C_F = 9/4$. Constituent multiplicity alone separates quarks from gluons with AUC 0.840. The ParT, ParticleNet, EFN and PFN trainings are running; no trained-model result exists yet.

Status: classical references done; the model trainings are running (job names below). Node type: setup (no edit on a network this round).

## What was produced
- **Released-weight search** (2026-10-19, GitHub API and READMEs):

| model | dataset | released weights | URL | downloaded to |
|---|---|---|---|---|
| ParT, ParticleNet (JetClass pre-trained only: `ParT_{full,kin,kinpid}.pt`, `ParticleNet_*.pt`) | JetClass | yes; no TopLandscape or QG fine-tunes released | https://github.com/jet-universe/particle_transformer (models/) | not downloaded (JetClass weights already in models/part, models/particlenet) |
| weaver-core | – | no weights (training framework) | https://github.com/hqucms/weaver-core | cloned to ext/ |
| LorentzNet | Top Landscape | yes: `best-val-model.pt`, `score.npy`, `test-result.json` | https://github.com/sdogsq/LorentzNet-release/tree/main/logs/top/pretrained | /j-jepa-vol/aaron-interp/models/top/lorentznet |
| LorentzNet | QG (full PID) | yes, same files | .../logs/qg/pretrained | /j-jepa-vol/aaron-interp/models/qg/lorentznet |
| PELICAN | top | no (checkpoints only as training output) | https://github.com/abogatskiy/PELICAN | – |
| L-GATr | top / QG | no (configs `toptagging`, `qgtagging`; no weights, no releases) | https://github.com/heidelberg-hepml/lorentz-gatr | – |
| EFN / PFN (energyflow) | – | no pre-trained top/QG models | energyflow 1.4.0 | trained here |

  - Clones in /j-jepa-vol/aaron-interp/ext/: particle_transformer 2925bdb, weaver-core 98ca934, LorentzNet-release 14bbe78.
  - The released LorentzNet top `score.npy` (404,000 rows: label, p0, p1) gives AUC 0.98700 [0.98675, 0.98725], R30 2193 [1806, 2690], R50 534 [492, 598] (95 % bootstrap, B = 200). These match its `test-result.json` (0.9870, 2192.75, 533.7) and the published 0.9868 / 2195 / 498.
  - Its row order is not ours: label agreement with our test rows is 0.499 at offsets 0 and 1, i.e. chance. So the released scores cannot be paired per jet with our test rows. The network itself has not been run here yet: LorentzNet uses its own conversion of the original h5 with two beam particles, and that input path still has to be built.
- **Herwig sample.** The QuarkGluon.tar listing (`tar t`, nothing extracted) contains only the 20 Pythia parquet files (train_file_0–17, test_file_0–1). There is no Herwig sample on the volume.
- **Loaders.**
  - `interp/data_top.py` and `interp/data_qg.py`, with `sample(per_class, seed, split)`: random rows per class (`rng.choice`, sorted). `load_split(split)` gives memory-mapped arrays.
  - Converted arrays, written only to /j-jepa-vol/aaron-interp/data/{top,qg}/<split>/:
    - `label`, `jet` (pt, eta, phi, E, mass, n);
    - `kin3` (pT, Δη, Δφ of the 128 hardest constituents, float16);
    - `full` (px, py, pz, E, Δη, Δφ, all constituents, float32; val/test only);
    - `pid` (QG);
    - `is_val` (QG train: the last 11.11 % of each file, as weaver's `--train-val-split 0.8889`).
  - Sizes:
    - top: 1,210,999 / 402,999 / 403,999 jets; max multiplicity 160 / 156 / 165.
    - QG: 1,799,982 train (199,980 of them is_val) / 199,998 test; max multiplicity 148 / 135.
    - Label 1 = top or quark.
  - Job d-h2-conv: 252 s, peak 8.0 GiB of 8 GiB. The peak was page cache from the memory-mapped writes; exit 0.

## Procedure (classical references)
- **Script:** `interp/h2_onesignal_setup.py`. Steps: top, crosscheck, qg, lnet, stats, fig.
- **Jobs:**
  - d-h2-class (conda env, 4 CPU / 6 GiB; 470 s, peak 4.7 GiB, 1.14 cores on average).
  - d-h2-stats: died before writing (job BackoffLimitExceeded, pod removed, no traceback). Rerun as d-h2-stats2 (2 CPU / 6 GiB, 188 s, peak 0.55 GiB, exit 0); the numbers equal d-h2-class's.
  - d-h2-fig: redraws the rejection figure with a shorter title.
- **Results:** `results/physics/h2_classical.json`, `h2_lnet.json`, `h2_crosscheck.json`, `h2_qg_obs.npz` (per-jet QG observables), and the figures `h2_fig_observables.png` and `h2_fig_top_rejection.png`. The per-jet top observables are on the volume (`/j-jepa-vol/aaron-interp/results/physics/h2_top_obs_{test,val}.npz`).
- **Sample:** all test jets, with no subsampling.
  - Top: 403,999 jets (signal fraction 0.500).
  - QG: 199,998 jets (quark fraction 0.500).
  - The likelihood-ratio histograms are filled on 100,000 random val jets (seed 0), disjoint from test.
- **Observables (top):**
  - **Clustering:** all constituents, using rapidity–azimuth distances.
  - **$m$:** invariant mass of the constituent sum. It agrees with the stored `jet_mass` to 0.055 GeV; 2 test jets have a NaN stored mass.
  - **$m_{SD}$:** Cambridge/Aachen reclustering, declustered from the top, $z_{cut} = 0.1$, $\beta = 0$. A groomed stop is found for 99.5 % of jets.
  - **$\tau_N$:** $\beta = 1$, $R_0 = 0.8$, axes from exclusive-$k_T$ clustering with winner-take-all $p_T$ recombination; $\tau_{32} = \tau_3/\tau_2$, $\tau_{21} = \tau_2/\tau_1$.
  - **$m_W$:** three exclusive-$k_T$ subjets (E-scheme); $m_W$ is the pair mass closest to 80.4 GeV.
  - **Cross-check:** fastjet 3.4.1 on 300 random test jets agrees for all 300: $\tau_{32}$ to 1.2e-6, $m_{SD}$ to 3e-11 GeV, $m_W$ to 0.1 GeV.
- **Observables (QG):**
  - $N$: all constituents.
  - $N_{ch}$: PDG id in {e, μ, π±, K±, p}.
  - $p_T^D = \sqrt{\sum p_{T,i}^2}/\sum p_{T,i}$.
  - Girth: $\sum p_{T,i}\Delta R_i/\sum p_{T,i}$, about the stored axis.
  - $C_1^{(\beta=0.2)} = \sum_{i<j} z_i z_j \Delta R_{ij}^{0.2}$.
- **Scores:**
  - Window observables use $-|x - x_0|$, with $x_0$ = 172.5 GeV ($m$, $m_{SD}$) or 80.4 GeV ($m_W$).
  - "window + τ32" is $-\tau_{32}$ inside 105 < $m_{SD}$ < 210 GeV and is undefined outside.
  - LR2 and LR3 are binned log-likelihood ratios of ($m_{SD}$, $\tau_{32}$) and ($m_{SD}$, $\tau_{32}$, $m_W$). Bins: 30 × 25 (× 20) over [0, 300] GeV × [0, 1] (× [0, 200] GeV), with +0.5 per bin.
  - QG signal = quark.
- **Metrics:**
  - AUC and background rejection $1/\varepsilon_B$ at $\varepsilon_S$ = 0.3 and 0.5, with 95 % bootstrap intervals over test jets (B = 200).
  - Cut-based working points: $\varepsilon_S$, $\varepsilon_B$, rejection.
  - Test 1. H0: adding $m_W$ to LR2 does not change the rejection. H1: it does. Paired bootstrap (same resamples) of $\log R_{LR3} - \log R_{LR2}$; two-sided p; Holm correction over the 2 working points.
  - Test 2. H0: $\langle N_g\rangle/\langle N_q\rangle = 9/4$. H1: the ratio differs from 9/4. Bootstrap, 1,000 resamples.

## Numbers
### Top Landscape (N = 403,999 test jets)
| observable / tagger | AUC [95 % CI] | $1/\varepsilon_B$ at $\varepsilon_S$ = 0.3 | at $\varepsilon_S$ = 0.5 |
|---|---|---|---|
| $m$ | 0.9273 [0.9264, 0.9281] | 43.1 [41.7, 44.4] | 23.0 [22.5, 23.5] |
| $m_{SD}$ | 0.9253 [0.9245, 0.9261] | 46.2 [44.9, 47.6] | 24.1 [23.6, 24.6] |
| $\tau_{32}$ | 0.8511 [0.8500, 0.8523] | 44.4 [43.4, 45.8] | 16.0 [15.7, 16.3] |
| $\tau_{21}$ | 0.6569 | 4.48 [4.43, 4.52] | 3.10 |
| $m_W$ | 0.7720 [0.7705, 0.7735] | 9.77 [9.65, 9.93] | 5.54 [5.49, 5.61] |
| 105<$m_{SD}$<210 + $\tau_{32}$ scan | – | 158 [150, 168] | 48.7 [47.3, 50.4] |
| 105<$m_{SD}$<210 + 65<$m_W$<95 + $\tau_{32}$ scan | – | 181 [171, 194] | 48.4 [46.8, 50.2] |
| LR($m_{SD}$, $\tau_{32}$) | 0.9535 [0.9529, 0.9541] | 243 [227, 258] | 64.2 [61.9, 72.2] |
| LR($m_{SD}$, $\tau_{32}$, $m_W$) | 0.9559 [0.9553, 0.9564] | 305 [282, 350] | 73.4 [70.7, 76.3] |
| LorentzNet, released scores (its own test ordering, 404,000 rows) | 0.9870 [0.9868, 0.9873] | 2193 [1806, 2690] | 534 [492, 598] |
| published ParT / ParticleNet / PFN (arXiv:2202.03772 Table 5) | 0.9858 / 0.9858 / 0.9819 | 1602 / 1615 / 888 | 413 / 397 / 247 |

The AUC of the window-based scans is not a full ROC: jets outside the window share one score. Their rejection values are the defined quantity.

Cut-based working points (CMS-like):

| cut | $\varepsilon_S$ [CI] | $1/\varepsilon_B$ [CI] |
|---|---|---|
| 105<$m_{SD}$<210 | 0.871 | 5.46 [5.41, 5.50] |
| 105<$m_{SD}$<210, $\tau_{32}$<0.54 | 0.439 [0.436, 0.441] | 68.3 [66.2, 70.8] |
| 105<$m_{SD}$<210, $\tau_{32}$<0.65 | 0.662 [0.661, 0.664] | 21.1 [20.8, 21.6] |
| + 65<$m_W$<95, $\tau_{32}$<0.54 | 0.391 [0.389, 0.393] | 96.9 [92.8, 100.9] |
| + 65<$m_W$<95, $\tau_{32}$<0.65 | 0.575 [0.572, 0.577] | 30.6 [29.9, 31.3] |
| 65<$m_W$<95 alone | 0.804 | 2.69 |

**$m_W$ beyond ($m_{SD}$, $\tau_{32}$)**, LR3 vs LR2, paired bootstrap:
- $\Delta\log R_{30}$ = +0.244 [+0.172, +0.335] (rejection × 1.28), p = 0.005, Holm 0.01.
- $\Delta\log R_{50}$ = +0.130 [+0.023, +0.161], p = 0.01, Holm 0.01.
- Both p-values are at the bootstrap floor of 1/200.
- H0 (no added rejection) is excluded at both working points.

**Distributions** (median, interquartile range):

| observable | top | QCD |
|---|---|---|
| $m_{SD}$ | 170 GeV [153, 181] | 36 GeV [17, 93] |
| $m_W$ | 82.7 GeV [76.9, 89.5] | 60.4 GeV [44.4, 77.2] |
| $\tau_{32}$ | 0.549 [0.447, 0.651] | 0.747 [0.673, 0.803] |

### Quark–gluon (N = 199,998 test jets, signal = quark)
| observable | AUC [95 % CI] | gluon rejection $1/\varepsilon_g$ at $\varepsilon_q$ = 0.5 | at $\varepsilon_q$ = 0.3 |
|---|---|---|---|
| $N$ | 0.8397 [0.8380, 0.8413] | 15.98 [15.63, 16.27] | 50.8 [48.7, 52.9] |
| $N_{ch}$ | 0.8285 [0.8267, 0.8301] | 12.24 [12.01, 12.47] | 34.8 [33.7, 35.9] |
| $p_T^D$ | 0.8070 [0.8049, 0.8090] | 10.42 [10.18, 10.64] | 26.8 [25.8, 27.7] |
| girth | 0.7395 [0.7375, 0.7419] | 8.38 [8.12, 8.60] | 48.1 [45.4, 49.9] |
| $C_1^{(0.2)}$ | 0.7960 [0.7941, 0.7980] | 16.62 [16.09, 17.13] | 56.4 [53.6, 59.4] |
| published ParT / ParticleNet (exp PID) / LorentzNet (full PID) | 0.9121 / 0.9116 / 0.9156 | 41.3 / 39.8 / 42.4 | 101.2 / 98.6 / 110.2 |

**Casimir check.**
- $\langle N_g\rangle$ = 53.19 and $\langle N_q\rangle$ = 33.47, so the ratio is 1.589 [1.584, 1.594].
- Charged only: 25.93 / 16.25, ratio 1.596 [1.591, 1.601].
- H0 ratio = 9/4: p < 0.001, at the bootstrap floor, for both $N$ and $N_{ch}$.
- The measured ratio lies at the ~1.5 expected once energy conservation, the running coupling and the non-perturbative offset are included. The leading-order value 9/4 is the asymptotic double-log limit.

## Trainings launched (status: running)
Authors' code throughout: particle_transformer `train_TopLandscape.sh` / `train_QuarkGluon.sh` with weaver-core 0.4.17. The authors' settings are kept: 20 epochs, batch 512, Ranger, in-memory loading. Only `--model-prefix`, `--log` and `--tensorboard` are redirected to /j-jepa-vol/aaron-interp/models/<ds>/<name>/.
- **Launcher.** Jobs use `interp/h2_gpu_job.sh`, a copy of part_job.sh with CPU, memory and GPU-product arguments, and `interp/h2_weaver_train.sh`.
- **Debug history.**
  - The first top ParT job ran out of GPU memory on a TITAN Xp (12 GB). Jobs are now restricted to ≥ 24 GB GPUs.
  - The QG kin+kinpid two-process jobs on 48 GB GPUs did not schedule. They were split into one job per feature set.
  - The QG kinpid jobs at 28 GiB host memory were OOM-killed. They were relaunched at 44 GiB.

| job | model / features | GPU | smoke check (epoch 1) | expected (published) |
|---|---|---|---|---|
| d-h2-top-part | ParT, top kin | A10 | train loss 0.201 (epoch 0 avg) → 0.175 running avg in epoch 1; val acc 0.9312, val AUC 0.9812; GPU 98–100 %, host mem 21 GiB / 24 | AUC 0.9858, R30 1602 |
| d-h2-top-pnet | ParticleNet, top kin | A10 | 0.192 → 0.174; val acc 0.9312 (different confusion matrix), val AUC 0.9811; GPU 98–100 % | AUC 0.9858, R30 1615 |
| d-h2-qg-part | ParT, QG kinpid (exp PID) | Quadro RTX 6000 | epoch 0 running avg loss 0.425, acc 0.810; GPU 93 % | AUC 0.9121 |
| d-h2-qg-part-kin | ParT, QG kin | A10 | 0.434 → 0.416; val acc 0.8179; GPU 82–100 % | – |
| d-h2-qg-pnet | ParticleNet, QG kinpid | A10 | epoch 0 running avg loss 0.432; GPU 100 % | AUC 0.9116 |
| d-h2-qg-pnet-kin | ParticleNet, QG kin | A10 | running avg loss 0.434 at 2,842 / 3,125 steps; GPU 98 % | – |
| d-h2-efn | EFN (energyflow, CPU 4 / 10 GiB), top then QG | – | top epoch 1: train loss 0.256, val_loss 0.238, val acc 0.904 (data loading from CephFS ~25 min before epoch 1) | top AUC ~0.98 |
| d-h2-pfn | PFN (z, Δη, Δφ), top then QG | – | val_loss 0.222 → 0.196 (epochs 1–2), val acc 0.906 → 0.920 | top AUC 0.9819 |

- **Expected wall times:**
  - top ParT about 25 min/epoch, so about 8.5 h.
  - QG ParT kin about 20 min/epoch.
  - QG ParticleNet kin about 1 it/s, about 55 min/epoch, so about 18 h. This is within the 24 h job deadline; weaver writes a checkpoint every epoch.
  - The EFN/PFN jobs stop each dataset after at most 40 epochs, patience 5, or 330 min.
- **Weaver AUC.** Weaver's own `roc_auc_score` prints "y should be a 1d array" (sklearn version); the AUC above is its `roc_auc_score_matrix` entry. Test-set AUC and rejection come from `predict_output/pred.root` at the end of each run.
- **Accuracy reproduction (rule check).** Not possible this round: no trained model exists yet, and released LorentzNet scores cannot be paired with our rows. Next round: each trained model's test AUC / R30 on the full test set vs the published values above, and the LorentzNet network run on our arrays vs its released AUC 0.9870.

## Pre-registered edits for the next round
**Top (D6): W-pair opening-angle edit.**
- Sample: `data_top.sample(5000, seed)`, top jets that pass the $m_{SD}$ window and have three subjets.
- Edit: rescale the angular separation of the two subjets that form $m_W$ about their common axis. The scale is chosen so that $m_W$ moves from about 80 GeV to a target (e.g. 60 or 100 GeV). The remaining subjet is moved so that the total jet mass is restored to within 0.5 GeV. Subjet $p_T$ fractions and the three-prong structure ($\tau_{32}$ within 0.02) are held fixed.
- Prediction of the D1 hypothesis (total mass + three prongs only): the network output stays the same within the control.
- Alternative (uses $m_W$): the output falls by the amount the $m_W$-window classical tagger falls. The LR3 − LR2 difference above (rejection × 1.28 at 30 %) gives the scale of the information available.
- Controls:
  - an edit of the same angular size applied to the non-W pair, which leaves $m_W$ fixed;
  - EFN/PFN, trained here, as models with no subjet pairing.
- Metrics: Δ output with bootstrap CI; McNemar on calls at the 30 % working point.

**QG (D7): collinear split and soft emission.**
- Edits:
  - collinear split: split one constituent into two with $p_T$ fractions $z$ and $1-z$ at $\Delta R < 10^{-3}$, so $N \to N+1$ and all IRC-safe shapes are unchanged;
  - soft emission: add a particle with $p_T$ = 0.1–1 GeV at a random position inside $R$ = 0.4.
- Multiplicity hypothesis: the output moves toward gluon by the amount predicted from $N$ alone. The calibration is the response to a natural change of N by +1, from the classical $N$ ROC.
- IRC-safe-shape hypothesis: the response is ≈ 0, as for the EFN.
- Zero control: EFN (IRC-safe by construction). Second control: PFN (not IRC-safe).

## Reading
- The classical references span rejections of 4.5 ($\tau_{21}$) to 305 (LR3) at 30 % top efficiency. Published networks are at 1600–2200.
- The single best classical observables are the mass variables ($m$, $m_{SD}$; AUC 0.93), with $\tau_{32}$ adding rejection at fixed mass.
- $m_W$ alone separates weakly (AUC 0.77): QCD jets with three subjets also produce a pair near 80 GeV. Combined with $m_{SD}$ and $\tau_{32}$, it adds rejection (paired test above).
- QG: $N$ has the highest AUC among the five observables. $C_1^{(0.2)}$ gives the same rejection at 50 % quark efficiency (16.6 vs 16.0, CIs overlap).
- The gluon/quark multiplicity ratio is 1.59 for all particles and for charged particles.

## Explainer entry (for site/explain.py, Paper D tab)
1. **Question.** Before editing top and quark–gluon taggers, what do classical observables reach on the same test jets? Does the W mass inside a top jet carry information beyond total mass and three-prong structure? Does constituent multiplicity follow Casimir scaling?
2. **Physics background.**
   - A hadronic top decays as $t \to bW \to bq\bar q'$. The jet then has three hard prongs, total mass $m \approx m_t$, and one pair of prongs with mass $m_W \approx 80$ GeV.
   - $N$-subjettiness $\tau_N = \frac{1}{R_0\sum_i p_{T,i}}\sum_i p_{T,i}\min_k \Delta R_{i,k}$ measures how well $N$ axes describe the radiation, so three-prong jets have small $\tau_{32} = \tau_3/\tau_2$.
   - Soft drop removes soft wide-angle radiation: decluster until $\min(p_{T1},p_{T2})/(p_{T1}+p_{T2}) > z_{cut}$.
   - Gluons radiate with colour factor $C_A = 3$ and quarks with $C_F = 4/3$. At leading (double-log) order the mean multiplicity ratio is $\langle N_g\rangle/\langle N_q\rangle \to C_A/C_F = 9/4$. Energy conservation and the running coupling lower it to about 1.5 at these energies.
3. **Method.** All test jets (403,999 top-landscape, 199,998 QG). Observables are computed from the constituents and cross-checked against fastjet. ROC metrics have bootstrap intervals. The likelihood ratio of ($m_{SD}$, $\tau_{32}$) with and without $m_W$ is filled on separate validation jets.
4. **Result (figures `h2_fig_observables.png`, `h2_fig_top_rejection.png`).**
   - At $\varepsilon_S = 0.3$: $R$ = 158 for the $m_{SD}$ window + $\tau_{32}$, 243 for LR($m_{SD}$, $\tau_{32}$), 305 for LR($m_{SD}$, $\tau_{32}$, $m_W$), against 1602 for published ParT.
   - QG: $N$ gives AUC 0.840.
   - $\langle N_g\rangle/\langle N_q\rangle = 1.589\,[1.584, 1.594]$, below 9/4 with p < 0.001.
5. **Checks.**
   - fastjet agreement on 300 jets.
   - Invariant mass vs stored mass within 0.055 GeV.
   - Released LorentzNet scores reproduce its reported AUC 0.9870.
6. **Physical significance.** These are the reference points for the edits: a network that uses only $m$, $m_{SD}$ and $\tau_{32}$ would sit near the classical LR2 rejection. The $m_W$ information exists (rejection × 1.28), and the next-round edit tests whether the networks use it. Multiplicity carries most of the classical quark–gluon separation, and the Casimir ratio measured in this sample (1.59) sets the expected size of a multiplicity-driven response.

Conclusion (explainer): the same as the report Conclusion above.

## Issues
- No unedited trained-model accuracy is reported this round, because the trainings are running. LorentzNet is checked only through its released score file.
- All bootstrap p-values are at the resampling floor (1/200 or 1/1000). Exact values need more resamples.
- The top-landscape val set (402,999) is sampled 409,600 times per validation epoch by the authors' script, so some jets repeat. This affects only the validation metric.
- Volume: converted arrays about 7 GB in /j-jepa-vol/aaron-interp/data/{top,qg} (QG train full four-vectors not stored).
