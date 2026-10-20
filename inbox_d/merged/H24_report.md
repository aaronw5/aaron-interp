# H24 — Quark–gluon taggers trained on Pythia vs Herwig: do they count constituents? (claim D7)

Script `interp/h24_qg_generator_models.py` (steps `eval` GPU, `base` CPU, `stats` CPU, `fig` CPU). Results in `/j-jepa-vol/aaron-interp/results/physics/`, copied to `results/physics/`:
- `h24_run.json`: all statistics (unedited metrics, edits, Herwig − Pythia paired tests, swap, Holm list).
- `h24_base.json`: LR(N) fits, Pythia − Herwig multiplicity gap, C2ST.
- `h24_models.json`: checkpoints used (epoch, path).
- `h24_generator_models.png`: figure.
- `h24_logits_{py,hw}.npz`, `h24_base_{py,hw}.npz`: per-jet logits per condition (volume only).

Jobs: `d-h24-smk-ev`, `d-h24-smk-b`, `d-h24-smk-st` (smoke, 300 jets/class), `d-h24-gen` (GPU eval, 8 min, exit 0), `d-h24-base` (references + C2ST, 4 CPU, 40 min, exit 0), `d-h24-stat` (stats; debug attempt 1, see Issues), `d-h24-stat2` (stats + figure as reported, exit 0). Logs `logs/d-h24-*.log`.
Herwig PN_kin (`d-h14-hw-pnet3`) finished at 06:58 UTC (best epoch 19 of 20, weaver test accuracy 0.7604) before `d-h24-gen` loaded it, so it is included as a final checkpoint.

## Conclusion
We added extra soft particles to quark and gluon jets in a way that leaves every infrared- and collinear-safe property of the jet unchanged, and measured how much each tagger moved toward "gluon" per added particle. Networks trained on Herwig jets respond to the particle count no more than networks trained on Pythia jets; both stay within about a tenth of what a classifier using only the particle count does. The networks' differences between Pythia and Herwig jets therefore do not come mainly from the multiplicity difference between the two generators. The two generators do shift the networks' count response by a small, measurable amount (about 0.01 in log-odds per particle). Whether that amount exceeds the spread between two training runs of the same network is not known, because only one training run per generator exists for ParT and ParticleNet.

## Procedure
**Samples.** Quark–gluon test jets (Komiske–Metodiev–Thaler, pT 500–550 GeV, |y| < 1.7, R = 0.4):
- Pythia 8.226: `interp.data_qg` test split.
- Herwig 7.1: Zenodo 3066475 files 18–19, the Herwig test split of the retrains (`interp.h14_qg_generator.load_sample`).

Each sample has 4000 quark + 4000 gluon jets, drawn as random rows with seed 9641.
- Mean N: Pythia quark 33.1, gluon 53.5; Herwig quark 33.4, gluon 48.6.
- Mean pT^D: Pythia quark 0.384, gluon 0.272; Herwig quark 0.375, gluon 0.285.

**Models.** The logit l = log p(gluon) − log p(quark), so a positive change is toward gluon.
- Pythia-trained:
  - ParT_kin and PN_kin (H2 weaver, final checkpoints).
  - PFN (`pfn_k`, H2 energyflow, the d-h14-pfn2 rerun, ported to torch).
  - PFN 2nd run (`pfn_t`, H5 torch, 400k jets).
  - EFN (`efn_k` and `efn_t`): IRC-safe, so its response to an exact split is 0 by construction.
  - ParT_kin with permuted weights (`part_kin_perm`): null model.
- Herwig-trained:
  - ParT_kin (d-h14-hw-part3, best epoch 12 of 20).
  - PN_kin (d-h14-hw-pnet3, best epoch 19 of 20).

**References.** Each is trained on each generator's training split, 30k jets per class, and evaluated on inputs rebuilt from the edited particle list.
- LR(N): negative-binomial likelihood ratio, fitted on the full training split.
- LR(pT^D) and LR(girth): 40-bin histogram likelihood ratios.
- GBDT(EFP + N) and GBDT(EFP): EFPs with d ≤ 5 (54 EFPs).

**Edits.** Every edited jet is rebuilt about its own E-scheme axis. Edits are deterministic from (generator, seed). The split family uses only jets with N ≥ 10: 7994 of 8000 Pythia jets and 7984 of 8000 Herwig jets.
- **add_soft (ΔN = +1).** One particle is added with the jet's softest pT, at a random existing particle's position + 0.05 in a random direction (donor radius).
- **add_swap (control, ΔN = 0).** The same particle is added and the jet's softest particle is removed.
- **sp_k, k = 1…5.** Exact collinear split (z = ½, ΔR = 0) of k distinct random particles from the softer half of the jet. N increases by k. Girth, mass, EFPs and the EFN latent are unchanged, and pT^D is lowered.
- **ct_k (control, ΔN = 0).** For the same k particles, momentum is moved from each one's nearest harder neighbour so that Σ pT² drops by the same amount as in sp_k. This gives the same pT^D with N fixed. Mean change per k = 1: girth +2 × 10⁻⁵, mass +0.025 GeV.
- **Observable swap.** On gluon jets, N is moved by the Pythia − Herwig mean training multiplicity gap (+4.48), at fixed pT^D. The quark gap is +0.04, so quark jets are almost never edited.
  - Herwig jets, moved toward Pythia: exact splits of soft particles, then Σ pT² restored by one momentum transfer between two existing particles, with the pair chosen to minimise the girth change. Residual mean |Δmass| = 0.00016 GeV, |Δgirth| = 2.5 × 10⁻⁶.
  - Pythia jets, moved toward Herwig: the softest particle is merged into its nearest-in-angle neighbour, then Σ pT² is restored in the same way. Residual Δmass = −0.046 GeV, Δgirth = −1 × 10⁻⁵.

**Metrics.**
- add_soft − add_swap: Δlogit per added particle.
- k-split: the per-jet least-squares slope of Δlogit on k, minus the same slope for ct_k (split − control).
- Swap: Δlogit on gluon jets.
- Statistics: 95 % bootstrap CI over jets (1000 resamples). p-values from a sign-flip permutation test with 20000 sign vectors. Herwig-trained − Pythia-trained is tested per jet on the same jets (paired). Holm correction over 122 tests: 3 edit metrics × 18 scorers × 2 generators, plus 4 Herwig − Pythia contrasts × 2 generators.

## Checks: unedited models on these samples (AUC [95 % CI] / accuracy)
| model | Pythia test jets | Herwig test jets | known value |
|---|---|---|---|
| ParT_kin, Pythia-trained | 0.904 [0.898, 0.912] / 0.833 | 0.815 [0.805, 0.824] / 0.721 | H14: 0.901 / 0.827 (Py), 0.808 / 0.718 (Hw) |
| PN_kin, Pythia-trained | 0.905 / 0.832 | 0.817 / 0.720 | H14: 0.902 / 0.828, 0.808 / 0.717 |
| PFN (pfn_k) | 0.896 / 0.824 | 0.818 / 0.730 | H2 test: 0.891 / 0.817 |
| PFN 2nd run (pfn_t) | 0.886 / 0.813 | 0.810 / 0.716 | H14: 0.881 / 0.804 |
| EFN (efn_k) | 0.886 / 0.817 | 0.809 / 0.722 | H2 test: 0.883 / 0.809 |
| ParT_kin, Herwig-trained | 0.888 [0.882, 0.896] / 0.803 | 0.836 [0.828, 0.845] / 0.760 | weaver Herwig test acc 0.7602 |
| PN_kin, Herwig-trained | 0.884 / 0.796 | 0.838 / 0.762 | weaver Herwig test acc 0.7604 |
| ParT_kin, permuted weights | 0.539 / 0.498 | 0.510 / 0.498 | H14: 0.548 / 0.499 |
| LR(N), own generator | 0.846 / 0.772 | 0.790 / 0.723 | H14: 0.840, 0.786 |
| GBDT(EFP + N), own generator | 0.868 / 0.791 | 0.803 / 0.728 | H14: 0.862, 0.800 |

The 2 × 2 table (training generator × test generator) for ParT_kin is Py→Py 0.904, Py→Hw 0.815, Hw→Py 0.888, Hw→Hw 0.836. On the matching test set each network is 0.04–0.05 AUC above GBDT(EFP + N).

On-manifold check: C2ST AUC for edited vs unedited jets, identity floor 0.499 (Py) and 0.486 (Hw).
- add_soft: 0.516 [0.498, 0.535] (Py), 0.495 (Hw).
- add_swap: 0.505 (Py), 0.498 (Hw).
- ct1 / ct5: 0.49–0.50.
- sp1 / sp5: 1.00, because exact zero-ΔR duplicates are detectable.
- swap: 0.756 (Py), 0.796 (Hw).

## Numbers (Δlogit per added particle, toward gluon; 95 % CI)
k-split slope minus ct-control slope (sp − ct):
| scorer | Pythia jets | Herwig jets |
|---|---|---|
| ParT_kin, Py-trained | +0.0083 [0.0076, 0.0090] | +0.0091 [0.0084, 0.0098] |
| PN_kin, Py-trained | +0.0045 [0.0036, 0.0054] | +0.0058 [0.0049, 0.0067] |
| PFN | +0.0209 [0.0204, 0.0215] | +0.0221 [0.0216, 0.0226] |
| PFN 2nd run | +0.0527 [0.0517, 0.0536] | +0.0532 [0.0523, 0.0541] |
| EFN (split exactly 0; control −) | −0.0033 | −0.0035 |
| ParT_kin, permuted weights | −0.0055 [−0.0058, −0.0053] | −0.0056 |
| **ParT_kin, Hw-trained** | −0.0041 [−0.0047, −0.0035] | +0.0013 [0.0008, 0.0019] |
| **PN_kin, Hw-trained** | −0.0083 [−0.0091, −0.0074] | −0.0014 [−0.0023, −0.0006] |
| LR(N) Pythia / Herwig | +0.101 / +0.079 | +0.105 / +0.082 |
| GBDT(EFP+N) Pythia / Herwig | +0.079 / +0.047 | +0.082 / +0.053 |
| GBDT(EFP) Pythia / Herwig | −0.002 / −0.001 | −0.003 / −0.003 |
| LR(pT^D) | 0.0000 | 0.0000 |

Raw split slopes before subtracting the control:
- Py-trained ParT_kin: +0.0107 (Py jets), +0.0117 (Hw jets).
- Hw-trained ParT_kin: −0.0024 (Py jets), +0.0032 (Hw jets).
- Mean Δlogit at k = 5: Py-trained ParT +0.056 / +0.062; Hw-trained ParT −0.009 / +0.017.

add_soft − add_swap:
| scorer | Pythia jets | Herwig jets |
|---|---|---|
| ParT_kin, Py-trained | +0.0045 [0.0034, 0.0056] | +0.0074 [0.0063, 0.0086] |
| PN_kin, Py-trained | −0.0026 [−0.0039, −0.0013] | +0.0022 [0.0008, 0.0036] |
| PFN / PFN 2nd run | +0.037 / +0.089 | +0.040 / +0.095 |
| EFN | +0.015 | +0.016 |
| ParT_kin, permuted | −0.011 | −0.011 |
| **ParT_kin, Hw-trained** | −0.0168 [−0.0177, −0.0157] | −0.0102 [−0.0112, −0.0092] |
| **PN_kin, Hw-trained** | −0.0304 [−0.0319, −0.0289] | −0.0212 [−0.0227, −0.0197] |
| LR(N) Pythia / Herwig | +0.108 / +0.085 | +0.112 / +0.089 |
| GBDT(EFP+N) Pythia / Herwig | +0.083 / +0.057 | +0.088 / +0.066 |
| GBDT(EFP) Pythia / Herwig | +0.013 / +0.009 | +0.014 / +0.010 |

Herwig-trained − Pythia-trained, paired on the same jets (sign-flip p = 5 × 10⁻⁵ for all eight contrasts; Holm < 0.05):
| contrast | Pythia jets | Herwig jets |
|---|---|---|
| ParT_kin, sp − ct slope | −0.0124 [−0.0131, −0.0117] | −0.0077 [−0.0084, −0.0070] |
| PN_kin, sp − ct slope | −0.0128 [−0.0138, −0.0118] | −0.0072 [−0.0083, −0.0062] |
| ParT_kin, add − swap | −0.0213 [−0.0221, −0.0205] | −0.0176 [−0.0186, −0.0167] |
| PN_kin, add − swap | −0.0279 [−0.0291, −0.0266] | −0.0234 [−0.0247, −0.0221] |

Spread references for the same quantity (sp − ct slope; add − swap):
- Same training data, different architecture (Py ParT − Py PN): +0.0038 / +0.0033; +0.0071 / +0.0052.
- Two PFN trainings on different-size Pythia sets and frameworks (pfn_k − pfn_t): −0.032 / −0.031; −0.052 / −0.055.
- LR(N) Pythia − Herwig: +0.023 (slope).

**Observable swap, gluon jets** (N moved by ±4.48 at fixed pT^D):
| scorer | Hw jets → Pythia N (+4.48) | Py jets → Herwig N (−4.49) |
|---|---|---|
| ParT_kin Py-trained | +0.065 [0.061, 0.069] | −0.055 [−0.061, −0.049] |
| PN_kin Py-trained | +0.044 | −0.027 |
| PFN | +0.115 | −0.122 |
| EFN | −0.001 | −0.002 |
| ParT_kin permuted | −0.025 | +0.041 |
| ParT_kin Hw-trained | −0.003 [−0.007, +0.0002] (Holm 1.0) | +0.060 [0.054, 0.066] |
| PN_kin Hw-trained | −0.023 | +0.101 |
| LR(N) Pythia / Herwig | +0.394 / +0.296 | −0.397 / −0.298 |
| GBDT(EFP+N) Pythia / Herwig | +0.375 / +0.212 | −0.394 / −0.213 |

Generator logit gap on gluon jets (mean logit, Pythia gluons − Herwig gluons) and the fraction of it reproduced by the Herwig-jet swap:
| scorer | gap | fraction reproduced |
|---|---|---|
| Py ParT_kin | 1.085 [1.015, 1.151] | 6 % |
| Py PN_kin | 1.067 | 4 % |
| PFN | 0.896 | 13 % |
| EFN | 0.942 | 0 % |
| Hw ParT_kin | 0.543 | −1 % |
| Hw PN_kin | 0.539 | −4 % |
| GBDT(EFP + N), Pythia / Herwig | 0.577 / 0.307 | 65 % / 69 % |

**Holm.** 107 of 122 tests are below 0.05. The 15 that are not:
- every LR(pT^D) slope and swap test (response exactly 0);
- the swap test for GBDT(EFP), LR(girth) and EFN (Pythia);
- the Herwig-jet swap for Herwig-trained ParT_kin;
- the GBDT(EFP, Herwig) slope on Pythia jets.

## Hypotheses
| hypothesis | prediction | observed | status |
|---|---|---|---|
| H0: architecture sets the strategy (D7 holds for both generators) | ParT/PN of both generators: count response −0.02 to 0, against LR(N) ≈ +0.08–0.10, on both test generators | 4 nets × 2 test generators: sp − ct between −0.008 and +0.009 (≤ 9 % of the same generator's LR(N)); add − swap between −0.030 and +0.007 | consistent; the Pythia-trained ParT/PN values are slightly above 0 (+0.002 to +0.009), the Herwig-trained ones are −0.030 to +0.001 |
| H1: the training generator sets the strategy, with the sign of that generator's LR(N) | Herwig-trained slope ≠ Pythia-trained by more than the seed spread; sign = sign(LR(N)_Hw) = + | Herwig − Pythia = −0.007 to −0.028 (p = 5 × 10⁻⁵), but the Herwig-trained responses are ≤ 0 or +0.001, opposite to LR(N)_Hw (+0.08) | sign prediction excluded; a generator dependence of about 0.01 per particle is present. Its size relative to a same-architecture seed spread is not measured |
| Herwig-trained nets count more (Herwig gluons have lower N, so N might be more useful) | Hw-trained slope > Py-trained | Hw-trained lower in all 8 contrasts | excluded |
| Networks' Pythia/Herwig gluon gap is carried by N | swap reproduces most of the gap, as for GBDT(EFP+N) (65–69 %) | ParT/PN reproduce −4 to 6 % of their gap (PFN 13 %) | excluded for ParT/PN |

## Reading (scoped to what was measured)
- **Size of the count response.** At fixed IRC-safe content, all four ParT/PN models change their gluon logit by at most 0.009 per added particle on either test generator. LR(N) changes by 0.079–0.105, and GBDT(EFP + N) by 0.047–0.082. GBDT(EFP), which has no N input, gives −0.003 to 0.
- **Effect of the training generator.** The Herwig-trained ParT_kin and PN_kin respond 0.007–0.013 per particle less on the k-split slope than the Pythia-trained ones, and 0.018–0.028 less on add − swap. The sign is the same on both test generators. The difference is 2–4 × the ParT − PN spread at fixed training data, and 3–4 × smaller than the gap between the two PFN trainings.
- **Size relative to the null.** The weight-permuted ParT_kin gives −0.0055 (slope) and −0.011 (add − swap). These are of the same size as the Herwig-trained networks' responses.
- **Swap on gluon jets.** Moving gluon multiplicity by the full generator gap at fixed pT^D moves Pythia-trained ParT/PN by 4–6 % of their Pythia-vs-Herwig gluon logit gap. It moves the Herwig-trained models by −1 to −4 %. GBDT(EFP + N) moves by 65–69 %.
- **Swap direction depends on the jet sample.** On Pythia jets, the merge-based swap (Py jets → Herwig N) moves the Herwig-trained models toward gluon (+0.06, +0.10), and the permuted null moves +0.041. That swap changes particle positions, so it is not a pure count edit.
- **PFN.** PFN is the only network that responds clearly to count: 0.021 (keras run) and 0.053 (torch run) per particle, which is 20–50 % of LR(N). The two PFN trainings differ by a factor of 2.5.

## Issues
- **No seed replicate for ParT/PN.** There is one training run per generator. The "seed spread" in the H1 test is replaced by the ParT − PN architecture spread (0.003–0.004) and the PFN two-run spread (0.03). A second-seed Herwig and Pythia ParT_kin would settle whether the −0.01 generator shift exceeds the run-to-run variation.
- **Off-manifold splits.** Exact collinear splits put two particles at ΔR = 0, and a classifier detects them perfectly (C2ST 1.00). add_soft and the controls are on-manifold (C2ST 0.49–0.52).
- **The control is not EFN-null.** ct_k moves momentum between particles, which changes the IRC-safe content slightly: the EFN control slope is −0.0035 and GBDT(EFP) is +0.002. Raw split slopes are given above for this reason.
- **Unclean Pythia-jet swap.** The merge-based swap on Pythia jets changes positions and mass (−0.046 GeV) and the permuted null responds to it (+0.041). The clean swap is the Herwig-jet one (|Δmass| 0.00016 GeV).
- **Herwig PN_kin.** d-h14-hw-pnet3 finished before evaluation, so PN_kin is final and not interim.
- **Debug attempt 1.** In `d-h24-stat`, the sign-flip test from H14's `Boot` uses 2000 permutations. Its p floor of 5 × 10⁻⁴ makes the Holm floor over 122 tests 0.061, so no test could pass. `d-h24-stat2` uses 20000 sign vectors (p floor 5 × 10⁻⁵). The numbers are unchanged; only the p-values changed.
- This script was written and its eval/base steps were run earlier in round D5. This worker ran the stats and figure steps, the fix, and wrote the report.

## Explainer entry (for site/explain.py; to be merged by the ledger)
```python
H24_EXPLAINER = [
 ('Question', 'Quark–gluon networks trained on Pythia jets barely respond to the constituent count at fixed IRC-safe content (D7). Is that a property of the networks or of Pythia jets? Herwig 7.1 jets have a different multiplicity and $p_T^D$ distribution; does a network trained on Herwig count more, or less?', None),
 ('Physics background',
  'Gluon jets radiate more (colour factor $C_A=3$ vs $C_F=4/3$), so they have more constituents $N$ and a smaller $p_T^D=\\sqrt{\\sum_i p_{T,i}^2}/\\sum_i p_{T,i}$. $N$ and $p_T^D$ are not infrared- and collinear (IRC) safe: an exact collinear split $p\\to(p/2,p/2)$ raises $N$ by one and lowers $p_T^D$, while every IRC-safe observable (girth $\\sum_i z_i\\Delta R_i$, mass, energy-flow polynomials) is unchanged. '
  'Pythia and Herwig differ mainly in these IRC-unsafe quantities: mean gluon $N$ is 53.1 in Pythia and 48.7 in Herwig at $p_T$ 500–550 GeV.', None),
 ('Method',
  '4000 quark + 4000 gluon test jets per generator (seed 9641). Models: ParT_kin, ParticleNet_kin trained on Pythia and on Herwig; PFN (two runs), EFN (IRC-safe), permuted-weight ParT. Edits about each jet\'s own axis: exact split of $k=1\\ldots5$ soft particles vs a $\\Delta N=0$ control that lowers $\\sum p_T^2$ by the same amount (same $p_T^D$); one added soft particle vs the same particle with the softest removed; a swap that moves gluon $N$ by the Pythia − Herwig gap (4.48) at fixed $p_T^D$. '
  'Metric: $\\Delta\\ell$ per added particle, $\\ell=\\log p_g-\\log p_q$; references LR($N$), LR($p_T^D$), LR(girth), GBDT(EFP+$N$), GBDT(EFP) per generator. Bootstrap CI, sign-flip permutation (20000), Holm over 122 tests.', None),
 ('Result',
  'Split − control slope: Pythia-trained ParT $+0.008$ to $+0.009$, PN $+0.005$ to $+0.006$; Herwig-trained ParT $-0.004$ to $+0.001$, PN $-0.008$ to $-0.001$; LR($N$) $+0.079$ to $+0.105$; GBDT(EFP) $\\approx 0$. Herwig − Pythia (paired) $-0.007$ to $-0.013$ per particle, $p=5\\times10^{-5}$. The $N$-swap on Herwig gluon jets reproduces 4–6 % of the Pythia-trained networks\' Pythia-vs-Herwig gluon logit gap (Herwig-trained: −1 to −4 %), against 65–69 % for GBDT(EFP+$N$).', 'results/physics/h24_generator_models.png'),
 ('Checks',
  'Unedited AUCs reproduce known values (Pythia ParT_kin 0.904 on Pythia, 0.815 on Herwig; Herwig ParT_kin 0.888 / 0.836; Herwig accuracy 0.760 = weaver test value). Added-particle and control edits are on-manifold (C2ST 0.49–0.52); exact splits are detectable (C2ST 1.00). Permuted-weight ParT responds $-0.006$ per particle, the same size as the Herwig-trained responses. Only one training run per generator for ParT/PN.', None),
 ('Physical significance',
  'For ParT and ParticleNet, the small response to an IRC-unsafe change in the particle count is found with either generator\'s training jets, so it is not specific to Pythia. Their Pythia/Herwig difference on gluon jets is not carried by the multiplicity difference. The training generator shifts the count response by about 0.01 per particle, toward less counting for Herwig; whether that exceeds training-run variation is open.', None),
]
```
Entry line for the `'d'` list: `('Result explained: do Herwig-trained quark–gluon taggers count constituents? (claim D7; H24)', H24_EXPLAINER, False)`.
