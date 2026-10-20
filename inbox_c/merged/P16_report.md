# P16: Planted-label ParT arms. Does a ParT trained on an invariant-mass label still respond to the mass-preserving φ shift? (claims C3, C10; method validation), 2026-10-20

Status: **done** (exploratory). Jobs `anrunw-ai-c-p16-run` (tests 1–3, all six models, 8 min) and `anrunw-ai-c-p16-eap` (test 4 on arms a, b and c, 41 min) both exited 0. `c-p16-plot2` re-rendered the figure.
A verifier (P16v) is planned on fresh rows: `test_20M/*_101.root`, seed 16.

Script: `interp/p14_planted_part.py`. It is unchanged except for shorter figure labels. The EAP step is `interp/p14_eap_part.py`.
Commands (run from the repo root after `./sync.sh`):
- `./part_job.sh c-p16-run 'python -m interp.p14_planted_part --models a,b,c,s,released_full,released_kin --no-eap --tag p16'`
- `./part_job.sh c-p16-eap 'python -m interp.p14_planted_part --models a,b,c --tag p16eap --das-steps 5'`
  - This job also re-ran tests 1–2, with the same numbers. Only its EAP block is used.

Results:
- `results/part/p14_planted_p16.json`
- `results/part/p14_planted_p16eap.json`
- figure `results/part/p14_planted_p16.png`

Logs: `/j-jepa-vol/aaron-interp/logs/c-p16-{run,eap,plot2}.log`.
Checkpoints: `/j-jepa-vol/aaron-interp/models/p14/{a,b,c}/final.pt` (epoch 9) and `s/final.pt` (epoch 1).

## Conclusion
We trained copies of the Particle Transformer (ParT) to flag jets whose mass falls in a fixed window.
- One copy was trained with a label defined by the true invariant mass.
- One copy was trained with a label defined by the axis-relative mass m_A, a mass-like quantity built from each particle's angular distance to the stored jet axis.

Both copies respond to a rotation that leaves the invariant mass unchanged at about 0.2 of their response to an equal real mass change. The released ParT models respond at 0.72–0.79. Both copies compute most of the mass from the pairwise particle features, while the released models use the per-particle angle inputs.

This measurement excludes two explanations: that the architecture and inputs alone force an axis-relative reading, and that a label alone sets which mass the network reads (the m_A-label copy does not learn m_A). The released models' larger response therefore comes from their training task or data: 10 classes and 100M jets, against 3 classes and 1.5M jets here. This experiment does not separate those two factors.

## Procedure
- **Models.** Three groups were analysed:
  - **Planted arms.** ParT (weaver-core 0.4.17, authors' configuration, kin inputs, 2 outputs), trained in P14 for 10 epochs on 1.5M W/Z/QCD jets from `train_100M`.
    - a: label 1[|m_inv − 85| < 10 GeV].
    - b: label 1[|m_A − 84.62| < 9.98 GeV], with m_A = √(Σ pT · Σ pT ΔR²) about the stored axis.
    - c: the label of a, with the per-particle position channels zeroed. Geometry enters only through the pair features ln kT, ln z, ln ΔR and ln m².
    - s: the label of a shuffled over jets, 2 epochs.
  - **Released models.** ParT_full and ParT_kin (JetClass, 10 classes).
- **Sample.** N = 20,000 random rows (`numpy.random.default_rng(14)`) of the union of `test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_100.root`, with 6,666 QCD, 6,659 Z and 6,675 W jets. None of these rows was used in training.
  - Planted labels on this sample: positive rate 0.412 for both a and b. Labels a and b agree on 97.6 % of jets. The labels recomputed here match the training side file on 100 % of jets.
  - Released-model reference: the P1 sample (rows 3,000–4,000 per class of `test_*_100`, N = 10,000).
- **Edits** (all through `part_core.build`; baseline = the same jets rebuilt unedited):
  - **phi:** every particle's Δφ + 0.06 rad about the stored axis. Mean Δlog m_inv = −1×10⁻⁸ and Δlog m_A = +0.125. This flips 0 % of the a labels and 26.1 % of the b labels.
  - **dil:** a per-jet dilation of (Δη, Δφ) by k_j, chosen so that Δlog m_A equals the phi value for every jet. Δlog m_inv = +0.121. Labels flip for 25.9 % (a) and 26.1 % (b) of jets.
  - **k1.1:** all angles × 1.1.
  - **rot:** the whole jet, including its axis, rotated by 0.06 rad. This is the null: the inputs are equal up to float rounding.
- **Read-out d.**
  - Planted arms: logit₁ − logit₀ (inside minus outside the window).
  - Released models: logit_W − logit_Z.
- **Test 1.**
  - **R:** the per-jet least-squares slope through the origin of Δd(phi) on Δd(dil), with a 95 % CI from a bootstrap over jets (2,000 resamples).
    - H_inv (the network reads m_inv) predicts R = 0. H_axis (it reads m_A) predicts R = 1. The null is the same slope for rot.
  - **R_probe:** a ridge read-out of log m_inv from the class token, fitted on the unedited train half. R_probe = mean Δ(phi) / mean Δ(dil) on the test half, with a 1,000-resample bootstrap.
  - **Calls:** the fraction of calls changed, with a Wilson CI. McNemar tests compare the positive (or W) call before and after each edit, with Holm correction over the 24 model × edit tests.
  - **Permutation test** of H0: R_x = R_y. The two models' per-jet (Δd(phi), Δd(dil)) pairs are swapped with probability ½, 2,000 permutations; p ≥ 1/2001.
- **Test 2: route split.** Two variants of each edit:
  - per-particle only: edited per-particle inputs, unedited pair four-vectors.
  - pair only: unedited per-particle inputs, edited four-vectors.

  Each route's share is the per-jet slope of its Δd on the full Δd, with a bootstrap CI.
- **Test 3: DAS** (arms only). A 1-dimensional direction u acts on the jet-mean particle token at the outputs of particle blocks 2 and 8.
  - The base jet's projection on u is replaced by the source jet's projection.
  - u is trained with Adam, 400 steps of 128 pairs, so that the intervened call equals the window label of the source jet's m_inv, or of its m_A.
  - IIA (interchange-intervention accuracy) is measured on 10,000 held-out pairs and on the 234 discriminating pairs, where the m_inv label and the m_A label of the source differ.
  - Controls: 4 random directions and u = 0.
  - Paired McNemar test on the discriminating pairs. Holm correction over the 8 arm × block tests.
- **Test 4: EAP-IG.** P9's graph (433 nodes, 21,504 edges) with P9's corruption B.
  - Data: 1,500 discovery jets (train half) and 3,000 evaluation jets (test half), random with seed 2014.
  - Outputs:
    - the number of edges and nodes at F = 0.8 (log-interpolated);
    - the literal circuit made of the top 256 nodes, in which every edge not between kept nodes is resampled: its F, own-label accuracy, confusion and McNemar test against the full model;
    - 20 random circuits with the same node-category counts.

## Numbers
### Unedited models
| model | sample | accuracy [95 % CI] |
|---|---|---|
| released ParT_full | P1 sample, 10-class (N = 10,000) | 0.8593 (P1: 0.859) |
| released ParT_kin | same | 0.7447 (P1: 0.745) |
| released ParT_full / ParT_kin | 20,000 W/Z/QCD jets, 10-class; argmax over {QCD, Z, W} | 0.757 / 0.659; 0.824 / 0.764 |
| arm a (m_inv window) | 20,000 jets, own label | 0.9805 [0.978, 0.982] (QCD 0.990, Z 0.973, W 0.978); vs label b: 0.970 |
| arm b (m_A window) | same | 0.9805 [0.978, 0.982] (0.991, 0.971, 0.979); vs label a: 0.966 |
| arm c (m_inv, pair route only) | same | 0.9775 [0.975, 0.979]; vs label b: 0.968 |
| arm s (shuffled) | same | 0.588. Calls "outside" for every jet, a constant output. |

### Test 1: φ + 0.06 rad vs matched real dilation
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

### Test 2: route split of the matched dilation
| model | per-particle share [CI] | pair share [CI] | sum | additivity R² | k1.1 per-particle / pair | phi: mean \|Δd\| pair-only / full / rot null; per-particle slope |
|---|---|---|---|---|---|---|
| a | 0.172 [0.170, 0.175] | 0.830 [0.827, 0.833] | 1.003 | 0.94 | 0.204 / 0.810 | 0.0033 / 1.20 / 0.0031; 1.000 |
| b | 0.211 [0.208, 0.214] | 0.796 [0.792, 0.799] | 1.007 | 0.92 | 0.243 / 0.777 | 0.0055 / 1.37 / 0.0052; 1.000 |
| c | −0.005 [−0.005, −0.005] | 1.005 [1.005, 1.005] | 1.000 | 1.00 | −0.008 / 1.007 | 0.0068 / 0.0068 / 0.0063; 0.000 |
| s | 0.63 [0.61, 0.64] | 0.37 [0.35, 0.38] | 0.997 | 0.99 | 0.60 / 0.40 | responses ≤ 3.5×10⁻⁴ |
| released ParT_full | 0.834 [0.829, 0.840] | 0.153 [0.148, 0.158] | 0.987 | 0.93 | 0.850 / 0.146 | 0.0016 / 0.80 / 0.0015; 1.000 |
| released ParT_kin | 0.832 [0.826, 0.839] | 0.168 [0.162, 0.175] | 1.001 | 0.95 | 0.848 / 0.160 | 0.016 / 0.65 / 0.015; 0.986 |

### Test 3: DAS (1-dimensional direction on the jet-mean token)
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

### Test 4: EAP-IG circuits (3,000 evaluation jets, own label)
| arm | unedited acc | corrupted acc | edges at F = 0.8 | nodes at F = 0.8 | literal top-256-node circuit: edges, F [CI], accuracy [CI] | 20 random circuits: F, accuracy (max) | McNemar vs full (right→wrong / wrong→right) | IG completeness |
|---|---|---|---|---|---|---|---|---|
| a | 0.984 | 0.016 | 6,423 (29.9 %) | 196 | 9,901; 0.948 [0.944, 0.951]; 0.932 [0.922, 0.940] | 0.229 [0.069, 0.428], 0.176 (0.556); p_perm 0.048 | 164 / 8, p 6e-39 | 1.07 |
| b | 0.981 | 0.018 | 7,012 (32.6 %) | 196 | 10,078; 0.946 [0.942, 0.950]; 0.943 [0.935, 0.951] | 0.210 [0.072, 0.341], 0.141 (0.277); p_perm 0.048 | 120 / 7, p 1e-27 | 1.06 |
| c | 0.977 | 0.021 | 6,079 (28.3 %) | 210 | 9,559; 0.939 [0.933, 0.944]; 0.923 [0.913, 0.932] | 0.206 [0.085, 0.341], 0.137 (0.343); p_perm 0.048 | 187 / 26, p 3e-31 | 1.06 |
| released ParT_kin (P14, W vs Z) | 0.754 | 0.255 | 7,287 (33.9 %) | 197 | 12,321; 0.898; 0.732 | 0.139, 0.307 | 166 / 101, p 8e-5 | 1.05 |

- Literal circuit calls by class: arm a calls "in window" for 13.8 % of QCD, 53.3 % of Z and 61.2 % of W jets; arm b: 13.4 / 53.4 / 61.0 %; arm c: 14.2 / 48.3 / 60.9 %.
- Top-40 node categories for arms a and b: 15 class-token coordinates (z), 7 head nodes of class block 0 (C0H), 2 MLP-group nodes of class block 0 (C0M), the embedding (E) and 4 head nodes of particle block 0 (P0H). The rest are particle-block MLP groups.
- Arm c has no E and no P0H among its top 40. It has 5 head nodes of class block 1 (C1H).

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
- **Training differences.** The planted arms differ from the released models in task (2 vs 10 classes), data (1.5M W/Z/QCD jets vs 100M JetClass jets) and training length. This experiment does not identify which of these produces the released R of 0.72–0.79.
- **Labels a and b barely differ.** They agree on 97.6 % of jets, so the training signal that separates them is about 2.4 % of jets. A label b defined to decorrelate m_A from m_inv (for example, a window on m_A / m_inv) would test H_data directly.
- **Arm s** learned a constant output after 2 epochs. It provides no shuffled-label R; its route shares are computed on numerical noise.
- **DAS power.** DAS has 234 discriminating pairs. A 1-dimensional jet-mean intervention reaches at most 0.82 IIA, so it may not capture the variable.
- **EAP random baseline.** EAP uses 20 random circuits (p floor 0.048), not the 100 in the node plan. It runs on 1,500 / 3,000 jets.
- **Window-edge accuracy.** The H_arch prediction "arm-a accuracy limited near the window edges" was not measured as a function of |m_inv − 85|. Per-jet calls were not saved. The verifier should add this.
- **Seeds.** One training seed per arm.

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
