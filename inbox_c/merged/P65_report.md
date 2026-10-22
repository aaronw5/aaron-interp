# P65: H→bb vs H→cc vs Z→qq flavour circuit in ParT_full (new claim C29; agent report, 2026-10-22)

## Conclusion
ParT_full separates H→bb from H→cc jets almost entirely through the tracks' impact parameters (how far each charged track misses the collision point). Replacing these values with those of light-quark tracks drops b-vs-c accuracy from 98 % to 61 %. Two attention heads in the first class-attention block (the stage where a single summary token reads all particles) carry most of the flavour difference: swapping just their outputs for those of a c-jet erases the separation. However, a literal circuit (these heads plus 32 class-MLP neurons, everything else replaced) reaches only 66 % b-vs-c accuracy even with 32 heads. It needs the per-particle MLP layers (the feed-forward layers that act on each particle separately) to recover 85 %.
So the flavour read-out is concentrated in a few heads, but the computation that produces the flavour feature runs through the per-particle MLPs and cannot be isolated as a head-only circuit.

Still uncertain:
- about three quarters of the input effect also appears when the impact parameters are replaced by values from other jets of the same class, so much of the response is to any change in the impact-parameter pattern rather than specifically to b-like displacement (as H44v found);
- which particle-MLP units compute the flavour feature is not located.

## Runs
| job | command | GPU / node | wall | output |
|---|---|---|---|---|
| `anrunw-ai-c-p65-smoke` | `python -m interp.p65_flavour_circuit_part --smoke && python -m interp.p65_plot --smoke` | TITAN Xp | 4 min | `results/part/p65_flavour_circuit_smoke.json`, `logs/c-p65-smoke.log` |
| `anrunw-ai-c-p65-run` | `python -m interp.p65_flavour_circuit_part && python -m interp.p65_plot` | GTX 1080 Ti | 1,463 s | `results/part/p65_flavour_circuit.json`, `results/part/p65_flavour_circuit.png`, `logs/c-p65-run.log` |
| `anrunw-ai-c-p65-plot` | `python -m interp.p65_plot` (CPU, job.sh; shortened axis label only) | – | <1 min | `results/part/p65_flavour_circuit.png` |

Scripts: `interp/p65_flavour_circuit_part.py`, `interp/p65_plot.py`. Smoke and full runs completed on the first attempt (no debug rounds).

Deviation from the plan: the planned files `<sample>_127.root` do not exist. test_20M holds `_100`–`_119`, so `_117` was used; it has not been used before for a flavour study (H44 used `_100`/`_101`).

## Unedited accuracy (reference check)
**P1 sample** (rows 3,000–4,000 of each class's `_100` file), run through this script's runner (trimmer off, fixed padding, manual attention when hooks act):
- ParT_full 0.8593 and ParT_kin 0.7447, the same as P1/P32/P34;
- untrained ParT 0.1011.

**P65 evaluation sample** (15,000 jets: 3,000 each of H→bb, H→cc, Z→qq, QCD; 500 each of the other six classes):
- ParT_full 10-class accuracy 0.8316. It is lower than on P1 because the class mix differs: H→cc, Z→qq and QCD, the classes most often confused, make up 60 % of this sample. On the four main classes it is 0.8167.
- b-vs-c accuracy 0.9827 [0.9792, 0.9858], AUC 0.9982.
- b-vs-light accuracy 0.9703, AUC 0.9956.

**ParT_kin** (no displacement inputs; 1,000 per class of H→bb, H→cc, Z→qq): b-vs-c accuracy 0.7905, AUC 0.878; b-vs-light 0.926.

**Untrained:** b-vs-c accuracy 0.4965.

**Exact checks:**
- Manual attention (needed for per-head control) vs torch's attention: max |Δ| 1.2·10⁻⁴.
- Donor logits computed under the longer common padding vs the clean logits: max |Δ| 6.4·10⁻⁶.
- Class-block re-run used for the neuron scan vs the full forward: 0.
- All units resample-ablated: output identical to the donor jet's own output (max |Δ| 0).
- All units kept through the hooks: identical to the clean network (max |Δ| 7.6·10⁻⁶).
- The edits change only the four impact-parameter columns of charged tracks: four-momenta, flags and charge are bit-identical, and neutral particles are unchanged.

## Procedure
**Sample.**
- JetClass test_20M `<sample>_117.root`, numpy `default_rng(6565)`, random rows (never first-k).
- H→bb, H→cc, Z→qq, QCD (ZJetsToNuNu): 3,000 rows each form the ranking set and a disjoint 3,000 each the evaluation set.
- Other six classes: 500 evaluation rows each.
- All 10 classes: 300 further rows each form the donor/reference pool (N = 3,000).
- QCD: 2,000 further rows form the light-track pool (34,191 charged tracks).
- Every jet passes once through `part_core.build`. The impact-parameter edits do not touch kinematics, so edited and unedited jets share the same build pass.
- Mean multiplicity of charged tracks: H→bb 20.3, H→cc 18.7, Z→qq 16.6, QCD 17.1.
- Tracks with S > 3 per jet: H→bb 7.9, QCD 1.4. S is defined under "Displacement edit" below.
- 100 % of H→bb jets and ≥ 97 % of the others have ≥ 6 tracks, so the top-3 and bottom-3 track sets below are disjoint.

**Scores.**
- s_bc = logit(H→bb) − logit(H→cc) = log(p_bb/p_cc), the network's b-vs-c log-odds.
- s_bl = logit(H→bb) − logit(Z→qq), the b-vs-light log-odds.
- Signed margin u_bc = +s_bc on H→bb jets and −s_bc on H→cc jets. Its mean over the pair's jets is 8.5 (ParT_full).
- An intervention's effect Δu is the mean change of u over those jets. Negative means flavour separation is removed. "Share of margin" is Δu / mean u.
- b-vs-c accuracy is the share of H→bb and H→cc jets with the right sign of s_bc (0.5 = chance).
- Note: JetClass Z→qq contains Z→bb and Z→cc. These are about 22 % and 17 % of hadronic Z decays, so "b-vs-light" is diluted.

**Displacement edit (input check).**
- A "track" is a particle with charge ≠ 0.
- S = |d0|/σ(d0) is the transverse impact-parameter significance: the distance of closest approach to the beam line in the transverse plane, divided by its uncertainty. It is large for tracks from b- or c-hadron decays, which fly mm before decaying.
- Edit: a track's (d0, σ(d0), dz, σ(dz)) are set to those of a light-pool track with the same charge and nearest pT (random among the 5 nearest). dz is the longitudinal impact parameter. Everything else stays as it was.
- (a) All tracks edited.
- (b) The 3 largest-S tracks edited.
- (c) Control for (b): the 3 smallest-S tracks edited. It changes the same number of tracks but removes almost no displacement information.
- (d) Control for (a): all tracks replaced by tracks from other jets of the same class (pool = the ranking jets of that class), pT- and charge-matched. It changes every impact-parameter value but keeps the class's displacement distribution. The light-specific part of (a) is therefore a − d.

**Single-head scans** (ranking rows, 9,000 jets).
- The 80 heads are p1h1–p8h8 (particle-attention blocks 1–8, heads 1–8) and b1h1–b2h8 (class-attention blocks 1–2, where the class token attends to all particles).
- Head output = its 128-dim contribution after the output projection, before the block's per-head scale and LayerNorm.
- **Mean ablation:** the output is replaced by its mean over the donor pool's real tokens.
- **Resample ablation:** the output is replaced, token by token in pT order, by the same head's output in a pT-matched jet of the other flavour (H→bb ↔ H→cc for b-vs-c; H→bb ↔ Z→qq for b-vs-light). Tokens beyond the donor's multiplicity get the donor's mean.
- Statistics: 1,000-resample bootstrap over jets; p from the normal approximation with the bootstrap s.e.; Holm correction over 160 tests (80 heads × 2 ablations) per metric. H0: Δu = 0.
- Frozen ranking = resample Δu_bc, most negative first.

**Key masking.**
- For one head, the key projection reads the residual stream of the edit-(a) jet. Queries and values read the unedited stream. In block 1 this is exactly the embedding recomputed with light-track IP values entering only that head's keys; in later blocks it is the edit-(a) residual.
- Displacement share through keys = Δu_key / Δu_input(a).
- Joint key masking of the top-k heads, the 64 particle heads, the 16 class heads and all 80 heads is measured on the evaluation rows.

**Class-token MLP neurons.**
- 2 blocks × 512 GELU outputs of fc1 (b1n0–b2n511).
- Each is mean-ablated (set to its donor-pool mean) one at a time; Holm over 1,024 per metric.
- Top neurons = the 32 Holm-significant neurons with the most negative Δu_bc.

**Held-out check.** The top-8 heads' single resample effects are re-measured on the evaluation rows. Joint resample ablation of the top-k heads is compared with 20 block-matched random sets (same number of heads per block) on 400 per class.

**Literal circuit** (user definition, evaluation rows).
- *Kept:* the top-k heads of the frozen ranking (k = 1, 2, 4, 8, 16, 32) and the 32 top class-MLP neurons, plus the input and pair embeddings, LayerNorms, class token and last layer.
- *Resample-ablated* from a pT-matched random jet of the 10-class donor pool: every other head, all 8 particle-block MLP outputs, and the other 992 class-MLP neurons.
- *Metrics:* b-vs-c and b-vs-light accuracy/AUC, 10-class accuracy and 10×10 confusion.
- *Controls:* 100 block-matched random head sets with the same neurons, on 400 per class of H→bb, H→cc, Z→qq. Rank p = (1 + #random ≥ selected)/(1 + 100); the minimum possible is 0.0099.
- *Floor:* everything ablated.
- *Variants:* neurons only; heads without neurons; all class neurons kept; particle MLPs kept (not literal).

**References.** ParT_kin and the untrained ParT get the unedited metrics and edits a–d (1,000 per class). The untrained ParT also gets the single-head key-masking scan. ParT_kin has no displacement inputs, so its edited input is identical to the unedited one.

## Numbers

### Input check (ParT_full, evaluation rows; Δu_bc with 95 % CI)
| edit | Δu_bc | share of margin | b-vs-c acc. | AUC_bc | Δs_bc H→bb / H→cc | Δu_bl | b-vs-light acc. | McNemar (b-vs-c correct lost / gained) |
|---|---|---|---|---|---|---|---|---|
| none | – | – | 0.983 | 0.998 | – | – | 0.970 | – |
| a all tracks → light | −6.43 [−6.59, −6.26] | 0.75 | 0.611 | 0.848 | −12.14 / +0.71 | −2.66 | 0.780 | 2,268 / 39 |
| b 3 largest-S → light | −1.73 [−1.82, −1.64] | 0.20 | 0.926 | 0.987 | −3.62 / −0.15 | −0.21 | 0.945 | 371 / 28 (p 1e−77) |
| c 3 smallest-S → light (control) | −0.33 [−0.36, −0.30] | 0.04 | 0.979 | 0.997 | −0.24 / +0.42 | −0.23 | 0.969 | 32 / 9 |
| d all tracks → same class (control) | −4.77 [−4.89, −4.65] | 0.56 | 0.851 | 0.935 | −7.13 / +2.41 | −1.87 | 0.925 | 858 / 70 |

- b − c = −1.40 [−1.50, −1.30]; McNemar b vs c 374 / 54, p 6·10⁻⁶⁰.
- a − d = −1.66 [−1.78, −1.55]: the light-specific part of edit a is 26 % of its effect.
- ParT_kin: every edit gives exactly 0.
- Untrained: a −0.016 [−0.049, +0.014]; all edits are within ±0.03 and none is distinguishable from 0.

### Single-head scans (ranking rows; margin 8.50, input-a effect −6.47)
- b-vs-c, Holm-significant heads: mean ablation 72/80, resample 74/80, key masking 70/80.
- Heads with |share of margin| > 5 % after Holm: mean 4, resample 7, key masking 1.

| head | resample Δu_bc [CI] | share of margin | held-out Δu_bc | mean-abl. Δu_bc | key Δu_bc (share of input-a) | resample Δu_bl |
|---|---|---|---|---|---|---|
| b1h8 | −5.33 [−5.37, −5.28] | 0.63 | −5.37 [−5.41, −5.33] | −2.67 | −1.20 (0.185) | −2.46 |
| b1h2 | −4.41 [−4.45, −4.36] | 0.52 | −4.47 | −1.25 | −0.14 (0.022) | −1.78 |
| b1h3 | −2.11 | 0.25 | −2.15 | −1.07 | −0.12 (0.019) | −2.19 |
| b1h7 | −1.67 | 0.20 | −1.69 | −1.03 | −0.13 (0.020) | −0.69 |
| p7h5 | −0.62 | 0.073 | −0.62 | −0.34 | −0.04 | −0.25 |
| b1h1 | −0.55 | 0.065 | −0.56 | −0.25 | −0.08 | −0.28 |
| p8h4 | −0.52 | 0.061 | −0.52 | −0.19 | −0.05 | −0.21 |
| b2h1 | −0.40 | 0.047 | −0.40 | −0.18 | −0.02 | −0.20 |

- All Holm p < 10⁻¹⁰⁰.
- Ranking top 16: b1h8, b1h2, b1h3, b1h7, p7h5, b1h1, p8h4, b2h1, p1h5, p6h3, p7h6, p6h6, p3h3, p6h5, p7h8, p5h5.
- Block mix of the top 8: 5 heads from class block 1, 1 from class block 2, and 2 from particle blocks 7–8.
- No early particle block is in the top 8; the highest-ranked first-block head, p1h5, is 9th (share 4.6 %).

**Joint resample ablation of the top-k heads** (evaluation rows; share of margin):

| k | share of margin | b-vs-c accuracy |
|---|---|---|
| 1 | 0.63 | 0.854 |
| 2 | 1.16 | 0.270 |
| 4 | 1.75 | 0.033 |
| 8 | 1.88 | – |
| 32 | 1.99 | – |
| all 80 | 2.01 | – |

- A share above 1 means the call follows the donor's flavour: resample ablation with an other-flavour donor swaps the flavour information, it does not only remove it.
- Against 20 block-matched random sets: the selected set is more negative than all of them at every k (p = 0.048, the minimum possible with 20 sets). For k = 2: random mean −4.08, selected −9.90.

**Joint key masking** (share of the input-a effect):

| heads whose keys are masked | share [CI] |
|---|---|
| top 1 (b1h8) | 0.18 [0.18, 0.19] |
| top 2 | 0.21 |
| top 4 | 0.27 |
| top 8 | 0.30 |
| top 16 | 0.40 |
| top 32 | 0.45 |
| 64 particle heads | 0.35 |
| 16 class heads | 0.30 |
| all 80 | 0.55 [0.54, 0.55] (b-vs-c acc. 0.913) |

- So about 45 % of the displacement effect enters other than through the attention keys: through values, queries and the per-particle MLPs.
- Untrained ParT, single-head key masking: max |Δu| 0.032 per head, against its input-a effect of −0.016 (not different from 0).

**Class-MLP neurons** (b-vs-c):
- 914 of 1,024 are Holm-significant, but none moves more than 5 % of the margin.
- The largest are b2n116 −0.193 (2.3 %), b2n267 −0.189, b2n22 −0.168.
- b-vs-light: 863 significant, 1 above 5 % (b2n267 −0.186, 2.9 %).

### Literal circuit (evaluation rows, 15,000 jets; 32 class-MLP neurons kept)
| circuit | b-vs-c acc. [CI] | AUC_bc | b-vs-light acc. | AUC_bl | 10-class acc. |
|---|---|---|---|---|---|
| k = 1 | 0.504 [0.491, 0.517] | 0.523 | 0.505 | 0.513 | 0.095 |
| k = 2 | 0.505 | 0.522 | 0.509 | 0.521 | 0.096 |
| k = 4 | 0.509 | 0.542 | 0.530 | 0.565 | 0.106 |
| k = 8 | 0.514 [0.501, 0.527] | 0.561 | 0.539 | 0.587 | 0.111 |
| k = 16 | 0.565 | 0.649 | 0.601 | 0.679 | 0.115 |
| k = 32 | 0.658 [0.645, 0.670] | 0.750 | 0.702 | 0.802 | 0.152 |
| neurons only (k = 0) = floor | 0.4985 | 0.504 | 0.495 | 0.497 | 0.094 |
| k = 8, no neurons | 0.508 | 0.547 | 0.538 | 0.590 | 0.111 |
| k = 8, all 1,024 class neurons | 0.514 | 0.561 | 0.539 | 0.608 | 0.132 |
| k = 8 + particle MLPs kept (not literal) | 0.647 | 0.741 | 0.706 | 0.805 | 0.138 |
| k = 32 + particle MLPs kept (not literal) | 0.850 | 0.952 | 0.856 | 0.915 | 0.340 |
| all 80 heads + 32 neurons, particle MLPs ablated | 0.706 | 0.832 | 0.798 | 0.897 | 0.277 |
| all 80 heads + all neurons, particle MLPs ablated | 0.718 | 0.833 | 0.834 | 0.923 | 0.434 |
| full network | 0.983 | 0.998 | 0.970 | 0.996 | 0.832 |

- Fraction of the full network's above-chance b-vs-c accuracy, (acc − 0.5)/(0.983 − 0.5): k = 8 3 %, k = 32 33 %.
- Against 100 block-matched random head sets (subsample of 400 per class):

| k | selected b-vs-c acc. | random mean | random q95 | rank p | AUC p |
|---|---|---|---|---|---|
| 1 | 0.499 | 0.492 | – | 0.12 | – |
| 4 | – | – | – | – | 0.04 |
| 8 | 0.5075 | 0.4965 | 0.5025 | 0.0099 | 0.0099 |
| 16 | 0.543 | 0.499 | – | 0.0099 | – |
| 32 | 0.643 | 0.511 | 0.526 | 0.0099 | – |

- Subsample floor: 0.4925.
- 10-class confusion of the k = 32 circuit (in the JSON): calls are spread over all 10 classes for every true class (diagonal 0.10–0.20 for the main classes, e.g. H→bb 455 of 3,000 called H→bb). The donors are a random class, so most of the call follows the donor.

### Attention of b1h8 (class token → particles) on 2,789 correctly called H→bb jets with ≥ 6 tracks
| tracks | unedited | after edit a |
|---|---|---|
| 3 largest-S tracks | 0.210 | 0.130 |
| 3 smallest-S tracks | 0.033 | 0.079 |
| all tracks | 0.75 | 0.63 |

- The count share of 3 particles is 0.077.
- The figure jet is row 3,000 of the evaluation set (H→bb, pT 781 GeV, s_bc = +11.8).

## Reading
**Predictions of the localized hypothesis:**
- (i) Fewer than 8 heads carry more than 50 % of the b-vs-c log-odds: observed. b1h8 alone 63 %, and two heads jointly 116 % under resample ablation (31 % for b1h8 under mean ablation).
- However, these are class-attention heads in class block 1, not early particle blocks.
- (ii) Key masking of those heads reproduces most of the displacement-input effect: not observed. b1h8's keys carry 18 %, the top 8 30 %, all 80 heads 55 %.
- (iii) The k = 8 literal circuit reaches more than 80 % of full b-vs-c accuracy: not observed (3 %; k = 32: 33 %). Random sets do sit at the floor (rank p 0.0099).

**Predictions of the distributed alternative:**
- Flat single-head effects: excluded (7 heads above 5 % after Holm).
- The circuit needs more than 32 heads: observed.
- Class-MLP neurons carry the separation: not observed at the single-neuron level (max 2.3 %), and keeping all 1,024 class neurons with 8 heads adds nothing (0.514).
- ParT_kin and untrained show no displacement share: observed.

**What limits the circuit is the particle-block MLPs.**
- With all 80 heads and all class neurons kept, ablating only the 8 particle MLPs drops b-vs-c accuracy from 0.983 to 0.718.
- Keeping them with the top-32 heads gives 0.850.

**Measured picture, scoped to these interventions:**
- the per-particle MLPs build a flavour feature from the impact parameters on each track;
- class-block-1 heads, mainly b1h8 and b1h2, collect it into the class token. b1h8 puts 2.7× the count share of its attention on the 3 most displaced tracks, and this preference halves when the displacement values are made light-like;
- the 2 class blocks turn it into the call.
This matches P34's lepton result: heads carry the read-out, and MLPs are required for the decision. Under the user's literal definition, no head-plus-class-neuron circuit reproduces the b-vs-c call.

**Input level:**
- the 3 most displaced tracks carry 27 % of the all-track effect, against 5 % for the 3 least displaced;
- the same-class resampling control (d) moves u by 74 % of edit a. So most of the response to replacing impact parameters is a response to changing the jet's impact-parameter pattern at all; the light-specific part (a − d) is −1.66.
- d still lowers H→bb s_bc by 7.1. This suggests the network uses the consistency between each track's impact parameter and its own kinematics, not only the marginal distribution. Untested.

## Issues / caveats
- **Resample ablation with an other-flavour donor measures "swap the head's flavour information", not "remove it".** Joint shares above 1 follow from this. Mean-ablation shares are about half as large (b1h8 31 %).
- **The literal circuit's donors are random-class jets**, so the ablated parts carry no fixed flavour. The floor equals the donor's own call (checked exactly).
- **The neuron set (32) is fixed across k.** The random head sets keep the same neurons.
- **Joint-ablation random controls used 20 sets per k**, so p ≥ 0.048.
- **Z→qq contains b and c quarks**, so b-vs-light numbers are a lower bound on b-vs-light separation.
- **Key masking at blocks after the first** uses the whole edit-a residual stream at that head's key input. This includes displacement effects routed through earlier heads' values and MLPs, so it is "displacement hidden from this head's keys", not "the raw inputs hidden".

## Suggested claim text (new C29, exploratory, confidence ~0.5 pending a verifier)
"In ParT_full the H→bb vs H→cc call depends on the tracks' impact parameters (light-matched replacement: b-vs-c accuracy 98 → 61 %). The flavour difference is read into the class token mainly by two class-block-1 heads (b1h8, b1h2; swapping either with a c-jet's output removes 63 % / 52 % of the b-vs-c log-odds). The feature itself requires the particle-block MLPs: a literal circuit of the top 32 heads plus 32 class-MLP neurons reaches 66 % b-vs-c accuracy (random head sets 51 %), and 85 % only when the particle MLPs are kept."

Verifier suggestions:
- fresh file (`_118`) and seed;
- mean-ablation-based circuit;
- resample from same-flavour donors as a null for the head scan (should give ≈ 0);
- particle-MLP neuron scan (8 × 512) to locate the feature;
- per-head value masking (impact parameters hidden from values only) to account for the 45 % not entering through keys.

## Explainer entry (for site/explain.py; to be added by the ledger agent)
**Question.** Which parts of ParT_full read the track displacement that separates H→bb from H→cc jets, and can a small circuit of them do the job alone?

**Background.**
- b and c hadrons travel a few mm before decaying, so their decay tracks miss the collision point.
- The transverse miss distance d0 divided by its uncertainty is the significance $S = |d_0|/\sigma(d_0)$. b jets have more tracks with large $S$ than c jets.
- ParT_full receives $d_0$, $\sigma(d_0)$, $d_z$, $\sigma(d_z)$ for every charged track; ParT_kin does not.

**Method.**
1. *Input edit.* Each track's impact parameters are replaced by those of a light-jet (QCD) track with the same charge and nearest $p_T$. Controls:
   - replace the 3 least-displaced tracks;
   - replace with tracks from other jets of the same class.
2. *Head ablations.* Score $s_{bc} = \log(p_{bb}/p_{cc})$ and margin $u = \pm s_{bc}$ (+ for H→bb jets, − for H→cc jets). For each of 80 heads, replace its output by its mean (mean ablation) or by its output in a $p_T$-matched jet of the other flavour (resample ablation), and record $\Delta \bar u$. Holm correction over 160 tests.
3. *Key masking.* Only the head's keys see the jet with light-like impact parameters. Share $=\Delta\bar u_{\text{key}}/\Delta\bar u_{\text{input}}$.
4. *Literal circuit.* Keep the top-$k$ heads and 32 class-MLP neurons; replace every other head, every particle-block MLP and every other class neuron by a random donor jet's value. Compare with 100 random head sets drawn from the same blocks.

**Result** (figure `results/part/p65_flavour_circuit.png`).
- The input edit drops b-vs-c accuracy from 98 % to 61 %.
- Head b1h8 alone carries 63 % of the margin, and its attention favours the most displaced tracks.
- Its keys carry 18 % of the displacement effect; all 80 heads' keys carry 55 %.
- The literal circuit reaches 51 % (k = 8) and 66 % (k = 32), against 50 % for random sets and 98 % for the full network.
- Keeping the per-particle MLPs as well raises k = 32 to 85 %.

**Checks.**
- Exact identities: all units ablated reproduces the donor's output; all units kept reproduces the clean network.
- P1 accuracy reproduced (85.93 %).
- ParT_kin's response is exactly 0, and the untrained network's response is 0 within errors.

**Why it matters.** Flavour tagging, the most physics-relevant per-track computation, is read out by a few class-attention heads but computed in the per-particle MLPs. Head-only circuits, which worked as the read-out for leptons and mass, do not suffice here.

**Glossary additions.**
- Impact parameter (d0, dz): the distance by which a track misses the primary vertex, transverse (d0) or along the beam (dz).
- Significance S: the impact parameter divided by its uncertainty.
- Resample ablation: replacing a unit's output by its output on another (donor) jet.
- Key masking: only the key projection of a head sees an edited input.
- Margin u: the signed log-odds, positive when correct.
