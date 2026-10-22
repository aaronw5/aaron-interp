# B9: one program-extraction recipe on ten JEDI-linear networks (fidelity vs description length, recurrence, transfer, edit agreement)

Node B9, stage 2, Paper B (claims B2, B4, B6t) and Paper A (A1x / A1r, multi-seed stability). Exploratory / rigour.
Script `interp/b9_programs_across_nets.py` (plan in the module docstring, fixed before the first run). Jobs b9-a, b9-b, b9-c, b9-d (networks; 1 CPU / 2 GiB), b9-agg (aggregation); smoke jobs b9-smoke / b9-smoke2 / b9-smoke3.
Results: `results/jedi/b9.json` (all numbers), `results/jedi/b9_net_<network>.json/.npz` (per-network programs, terms, per-jet calls), figure `results/jedi/b9.png`. Logs `logs/b9-*.log`.

## Conclusion
The same recipe gives programs of similar size and accuracy on all nine trained networks that reach normal accuracy. With 16 terms per output it reproduces the network's call on 87–90 % of jets. Most of what these programs encode is common to the task rather than specific to one network. A program fitted only to the truth labels, never to any network, matches a network's calls and its responses to the shift edit about as well as another network's program does. A network's own program does better by about 1 percentage point on natural jets and 3 points on shift-edit call changes. Uncertain: the azimuth-only "turn" edit cannot be described by this feature pool at all, and the recurrence counts depend on the knot-matching tolerance, which was not varied.

## Question
Does the rank-window MARS program extraction used in B6/B7 (fitted so far only to official n64) behave the same way on every trained JEDI-linear network? The run checks three things:
- whether it gives programs of similar size and fidelity on each network;
- whether the same program statements recur across seeds and checkpoints;
- whether one network's program predicts another network's calls and edit responses as well as that network's own program does.

## Definitions
- **Program**: MARS (multivariate adaptive regression splines; forward selection of hinge terms $\max(0, x-t)$, $\max(0, t-x)$ or $x$, and products of two hinges). It is fitted to each of the network's 5 logits (g, q, W, Z, t) on the fit jets, and its call is the argmax of the 5 fitted logits.
- **Feature pool**: the same for every network. It holds the 97 rank-window sums of `e11_mars_all.handbuilt` plus $M_0=\sum p_T$, 98 inputs in all. The sums are $\sum p_T$, $N$, $\sum p_T r^2$, $\sum r^2$, $\sum r$, $\sum p_T r$, $\sum (r-0.2)_+$, $N(p_T>1)$, $N(p_T<2)$, $\sum r\,[r<0.165]$ and $N(p_T<2, r>0.1)$, taken over $p_T$-rank windows. A rank window is a range of particles ordered by $p_T$: all, the hardest 3/11/14/26/39, or ranks 1–63, 8–63 and 13–63. $r$ is the angular distance of a particle from the jet axis.
- **K**: number of terms per logit. The nested programs use K = 2, 4, 8, 16, 32 and 64; each is the first K terms of one forward path, with coefficients refitted by least squares.
- **DL (description length, bits; B6 definition)**: per logit, 32 bits for the intercept, plus 32 bits per coefficient, plus $\log_2 98 + 2 + 32$ bits per hinge factor. The last 32 bits (the knot) are left out for linear factors.
- **Same call**: share of test jets on which the surrogate's call equals the network's call. Wilson 95 % interval.
- **Cohen kappa** $\kappa = (p_o - p_e)/(1-p_e)$: agreement corrected for chance agreement. For edits it is computed on the per-jet "call changed" indicators: 0 means no better than the rule "never change the call", and 1 means the same set of changed jets.
- **Edit agreement (per jet)**: 1 if neither the network nor the surrogate changes its call, or if both change to the same new call.
  - **Recall**: share of the network's call changes that the surrogate reproduces with the same new call.
  - **Precision**: share of the surrogate's call changes that are network call changes to the same call.
- **Matched DL**: within each baseline family, the member with DL ≤ the program's DL and the highest same-call on dev jets.
- **McNemar**: exact binomial test on the jets where exactly one of two paired indicators is 1.
- **Holm**: step-down multiple-comparison correction, applied here over the 9 non-degenerate trained networks.

## Networks
No network was trained for this node.
- **Trained (10)**:
  - official 3-feature n64, n32 and n128;
  - perm-inv n64;
  - 16-feature n64 (16f);
  - X12 retrains: HGQ-r150 seeds 0, 1 and 2, and float seeds 0 and 1.
- **Degenerate**: hgq150_s1 (56.3 % test accuracy) is reported but left out of the pooled summaries, which therefore cover 9 networks.
- **Floor**: 3 untrained redraws of official n64 (`Jedi(64, untrained=True, seed 0/1/2)`). Each one calls the same class for every jet (accuracy 19.9–20.1 %), so every surrogate matches them trivially (100 %) and none of their calls change under any edit. They serve only as a floor for the recurrence counts.

## Samples
All samples are drawn with `jedi_core.sample`, and every network sees the same jets.

| Set | Draw | Jets |
|---|---|---|
| fit | sample(1000, 4901) | 5,000 |
| dev | sample(1000, 4902), minus 99 rows shared with fit | 4,901 |
| test | sample(1000, 4903), minus 187 rows shared with fit or dev | 4,813 |

- n32 sees the first 32 slots, and its program inputs are computed from those same 32 slots.
- n128 sees the same rows with 128 slots, and 16f the same rows with 16 columns; the loader checks both copies against `sample()` (max |difference| = 0). Their programs use the 64-slot jet.

**Reproduction check (official n64)**: 80.55 % on test (Wilson [79.4, 81.6]), 81.72 % on fit, 80.43 % on dev. The pooled value over all 14,714 jets is about 80.9 %, inside the expected 80.6–81.3 %. Test accuracies of the other networks:

| Network | Test accuracy |
|---|---|
| n32 | 77.71 % |
| n128 | 80.34 % |
| perminv | 81.92 % |
| 16f | 81.74 % |
| hgq s0 | 76.50 % |
| hgq s1 | 56.29 % |
| hgq s2 | 78.04 % |
| float s0 | 81.55 % |
| float s1 | 81.94 % |

## Edits
All edits are applied to the test jets. Padding is never moved, and every surrogate recomputes its inputs from the edited particles.
- **shift**: every real particle's $\Delta\eta$ is increased by 0.06 rad, a rigid translation (the E6 W→Z flow edit). It is applied to all 4,813 jets.
- **turn** (E27w arm B, subset hard14, dose 10 %): the 14 hardest particles are each turned about the jet axis toward the direction of the $p_T$-weighted centroid $\vec M_1$, by a common fraction of their angle gap. The fraction is solved per jet so that the jet mass drops by 10 %. Each particle's $p_T$ and $r$ are kept exactly, so $M_0$, $M_2=\sum p_T r^2$, every rank-window sum and every pT rank are unchanged. Measured: max $|\Delta M_2/M_2| = 7\times10^{-8}$, median $\Delta\ln m = -0.105$. 3,040 jets are feasible.
- **split** (E40 C1): the hardest particle is split into two halves of equal $p_T$ at the same $(\Delta\eta, \Delta\phi)$, and the list is re-sorted. The jet four-momentum, $M_0$, $M_1$ and $M_2$ are exact, and N increases by 1. It is applied to the 3,695 jets with N ≤ 63.
- **soft** (E40 C2): one particle of 0.5 GeV is added at $r = 0.3$ with a uniform random azimuth (rng 4905). All other particles get one common scale and one common Lorentz boost so that the jet four-momentum is unchanged (max relative deviation 2e-12). It is applied to 3,695 jets.
- **Not run on edits**: the n128 and 16f networks, as in E39, because their inputs carry slots or columns that these edits do not define. Their programs are still used as transfer donors.

How often each network's call changes (shift / turn / split / soft, % of edited jets):

| Network | shift | turn | split | soft |
|---|---|---|---|---|
| official n64 | 56.3 | 4.6 | 4.5 | 2.9 |
| official n32 | 65.7 | 7.2 | 5.1 | 1.3 |
| perminv n64 | 61.8 | 3.2 | 3.1 | 2.5 |
| hgq150 s0 | 60.0 | 5.8 | 7.0 | 4.1 |
| hgq150 s2 | 60.3 | 4.2 | 7.3 | 3.4 |
| float s0 | 64.3 | 6.6 | 2.3 | 1.9 |
| float s1 | 59.0 | 5.4 | 2.6 | 1.9 |

## Baselines and controls
Every baseline is fitted to the network's calls or logits on the fit jets, never to the labels.
- **CART**: decision tree, depth 2–12, on the pool.
- **HistGBDT**: gradient-boosted trees on the pool quantised to 4, 6 or 8 bits; 4/8/16 leaves × 3/10/30/100 iterations; lr 0.5.
- **Lasso**: one per logit, 10 regularisation strengths (alphas).
- **MLP**: inputs (m, ΣpT, N), one hidden layer of 2–64 units.
- **MARS-STD**: MARS on 10 standard observables: m, m_SD, $\tau_{21}$, $\tau_{32}$, $D_2$, N, ΣpT and $\sum p_T r^n$ (n = 1, 2, 3). Here $\tau_{21}$ and $\tau_{32}$ are N-subjettiness ratios (2- vs 1-prong and 3- vs 2-prong likeness), $D_2$ is the energy-correlation 2-prong discriminant, and m_SD is the soft-drop groomed mass.

Controls:
- **shuffled-logit program**: MARS K16 fitted to the fit rows' logits after a random permutation (rng 4906).
- **random-feature program**: MARS K16 restricted to 6 pool inputs drawn at random (rng 4907).
- **truth program**: MARS K16 fitted to the truth labels (target $\log(0.95\,\mathrm{onehot}+0.01)$). It never sees any network, and it serves as the "class-score-free" baseline in the transfer test.

## Results

### 1. Fidelity vs DL is similar on all networks (figure, left panel)
Program K16: DL 6,470–6,870 bits on every network. Same call on test jets:

| Network | Same call (95 % interval) |
|---|---|
| official n64 | 90.30 % [89.4, 91.1] |
| official n128 | 90.03 % |
| perminv | 89.99 % |
| 16f | 89.72 % |
| hgq s2 | 89.80 % |
| float s1 | 88.22 % |
| hgq s0 | 87.85 % |
| float s0 | 87.70 % |
| official n32 | 86.91 % |
| hgq s1 (degenerate) | 84.85 % |

- Kappa (5-class): 0.836–0.879 on the 9 non-degenerate networks.
- K2 / K8 / K64 range: 60.7–73.2 % / 85.3–89.1 % / 88.8–91.9 %.
- Controls on official n64: shuffled-logit program 18.8 % (κ = −0.01), random-feature program 65.5 %.

Matched-DL comparison at K16 on natural jets: the program is above every baseline family on all 9 networks (Holm p < 0.05 in 9/9 for each family). Mean differences:

| Family | Mean difference |
|---|---|
| MARS-STD | +2.6 pp |
| CART | +2.7 pp |
| GBDT | +3.2 pp |
| MLP | +3.1 pp |
| Lasso | +25.0 pp |

Example, official n64 vs MARS-STD K8 (3,455 bits): 90.30 vs 87.49 %, difference CI [+2.0, +3.6] pp, McNemar 264 vs 129 discordant jets.

Frontier area (mean over a common log2-DL grid; official n64; program minus baseline):

| Comparison | Natural jets | Shift edit |
|---|---|---|
| vs GBDT | +1.1 pp [0.5, 1.7] | +6.8 pp [5.8, 7.7] |
| vs MARS-STD | −1.4 pp [−1.9, −0.8] | +11.3 pp [10.4, 12.1] |
| vs CART | −4.4 pp [−5.2, −3.7] | +5.1 pp [3.9, 6.2] |

CART reaches a given same-call at fewer bits than the program, but not the same shift agreement.

### 2. Edit agreement
**Shift.** Program K16 recall of the network's call changes, by network:

| Network | Recall | 95 % interval |
|---|---|---|
| official n64 | 57.3 % | [55.4, 59.1] |
| official n32 | 53.5 % | |
| perminv | 48.3 % | |
| hgq s0 | 64.0 % | |
| hgq s2 | 66.0 % | |
| float s0 | 34.2 % | |
| float s1 | 40.7 % | |

- Flip kappa: 0.45–0.67.
- MARS-STD K16 recall on the same networks: 13.0–44.0 %.
- Matched-DL K16: the program has higher per-jet shift agreement than GBDT (7/7 networks, Holm p < 0.05), MLP (7/7), Lasso (7/7) and CART (6/7). Against MARS-STD it is higher on 6/7; on float s0 MARS-STD is higher (45.2 vs 48.4 %, CI [−4.7, −1.6] pp).

**Turn.** Program recall is exactly 0 on every network. The turn edit leaves every input in the pool unchanged, so the program's call cannot change. The program's "agreement" on turn therefore equals the never-flip rate (95.4 % on official n64). It beats MARS-STD and MLP at matched DL on 7/7 networks only because those baselines change their calls when the network does not:
- MARS-STD K32 on official n64: recall 19.3 %, precision 4.5 %.
- MLP h64 on official n64: recall 22.1 %, precision 3.0 %.

The prediction "the program beats matched-DL baselines on the turn edit" cannot be tested with this feature pool.

**Split and soft.**
- Program K16 recall: 6–11 % (split) and 0–7 % (soft) of the network's call changes (κ 0.0–0.16).
- At matched DL, MARS-STD and MLP have higher split agreement on 7/7 and 6/7 networks, by about 1 pp (mean −1.0 and −1.1 pp).
- GBDT and CART are lower than the program on split (5/7 and 3/7 networks with Holm p < 0.05).
- No family differs from the program on soft except Lasso (lower, 7/7).

### 3. Recurrence of statements
Matching rule: two terms match if they use the same pool input(s) (same sum, same rank window) and the same hinge type, and their knots differ by at most 0.05 in quantile units. Each term's recurrence count is the number of the other 8 non-degenerate trained networks whose K16 program contains a match.

Null hypothesis H0: recurrence occurs at the rate expected if each term's input were redrawn uniformly from the 98, with hinge type and knot kept. Under H0 the count follows an exact Poisson-binomial distribution; the test is one-sided and Holm-corrected over each network's 80 terms.

- **Mean count, K16**: 2.7–3.2 of 8 (null 0.36–0.44) on 8 networks. Official n32 is lower: 1.2 (null 0.31).
- **Terms above the null**: 29–36 of 80 per network (Holm p < 0.05); 9 of 80 on n32.
- **Terms recurring in at least 7 of 8 networks**: 12–23 % (n32: 4 %).
- **Untrained floor**: the 3 untrained-network programs contain a match for 1–3 % of terms.
- **K8**: mean count 3.1–3.7 of 8 (null about 0.3); n32 0.9.

The most recurrent terms in official n64's program (count 7–8 of 8) are mostly hinges below a knot in $\sum p_T r^2$ over all particles, which is $M_2$, the radial moment the mass variable is built from (E6). The others are $M_0 = \sum p_T$ above its 40 % quantile and $\sum p_T$ over the 39 hardest particles. 12 of the 16 terms recurring in at least 7 of 8 networks also appear in the truth-label program. Pooled over the 9 networks' K16 terms, terms that are also in the truth program recur in 5.0 of 8 networks on average, versus 1.8 of 8 for the other terms.

Core families. "Present" means some term of the program uses that kind of input.
- **S0-like hard $p_T$ sum** ($\sum p_T$ over a top-k window, or $M_0$):
  - K2 (the first two terms per logit): present in 9/9 trained programs, against a null presence probability of 0.47 (Poisson-binomial p = 0.001), and in 0/3 untrained programs.
  - K16: 19–35 % of a trained program's terms, against a null share of 7 %.
- **S29-like radial** ($\sum p_T r^2$, $\sum r^2$, $N(p_T<2, r>0.1)$):
  - K2: 9/9, but also present under the null (probability 0.96, p = 0.69).
  - K16: 40–60 % of a trained program's terms, against a null share of 32 % (untrained 20–34 %).
- **Soft count** ($N(p_T<2)$; N over ranks 1–63 / 8–63 / 13–63):
  - K2: 8/9, against a null probability of 0.73 (p = 0.25).
  - K16: 1–5 % of terms, below the null share of 15 %.

The thesis prediction was that the core terms recur in at least 7 of 9 networks. The hard $p_T$ sum does, above the null. Radial terms are present in every program, but presence is not informative because the null also predicts it. The soft count is not over-represented.

### 4. Transfer
For each ordered pair of networks (A, B), the K16 program of A is scored against B's calls. The 9 non-degenerate trained networks give 72 pairs on natural jets; edits use 56 pairs, with B among the 7 edit-capable networks. The interval is a network-level bootstrap (2,000 draws).

| Quantity (mean over pairs) | Value | 95 % CI | p |
|---|---|---|---|
| natural same-call: own − transfer | +1.25 pp | [+0.40, +2.24] | 0.0005 |
| natural same-call: transfer − truth program | +0.47 pp | [−0.26, +1.07] | 0.23 |
| shift agreement: own − transfer | +2.49 pp | [+0.93, +4.22] | 0.007 |
| shift recall: own − transfer | +2.72 pp | [+0.31, +5.11] | 0.03 |
| shift recall: transfer − truth program | −0.01 pp | [−2.23, +2.25] | 1.0 |
| D (shift) = agreement loss − natural loss | +1.23 pp | [+0.16, +2.33] | 0.03 |
| split / soft agreement loss | +0.09 / +0.10 pp | [−0.18, 0.25] / [−0.01, 0.30] | 0.32 / 0.07 |
| turn | 0 (no program responds) | | |

- On natural jets, own minus transfer stays within 1.5 pp for all pairs except those involving n32. For n32's program scored on official n64, transfer is 83.5 % against 90.3 % for n64's own program; for official n64's program scored on n32, transfer is 86.7 % against n32's own 86.9 %.
- On shift, official n64's own program reproduces 57.3 % of its call changes. Other networks' programs reproduce 45.3–55.8 %, and the truth program 52.3 %.
- In the other direction, n64's program reproduces more of perminv's shift changes (54.1 %) than perminv's own program does (48.3 %).

### Hypotheses: what each predicted and what the run shows
Thesis B (programs are a faithful, compact, network-specific description):

| Prediction | Result |
|---|---|
| (a) Beats matched-DL baselines on shift agreement on every trained network | 7/7 vs GBDT, MLP and Lasso; 6/7 vs CART and MARS-STD. Float s0 has MARS-STD higher. |
| (a) Beats matched-DL baselines on turn agreement | Untestable: the pool is blind to turn. |
| (b) Core terms recur in at least 7/9 networks | The hard pT sum does, above the null. The radial and soft-count families are not separable from the null at the family level. |
| (c) Transfer loses more on edits than on natural jets | Holds for shift (D = +1.2 pp, CI excludes 0) but not for split, soft or turn, where D is negative because the edit losses are about 0. |

The alternative (the description is of the task, not the network) also had three parts:
- **"The advantage is specific to official n64"**: excluded. The fidelity curves and the shift-agreement advantage hold on the retrains.
- **"Terms do not recur above the null"**: excluded for about 30 of 80 terms per network.
- **"Any network's program transfers as well as B's own"**: excluded at p ≤ 0.03 on natural jets and on shift. The gap is small (1–3 pp), and a program fitted to the truth labels does as well as a transferred one.

## Checks and caveats
- **Reproduction**: official n64 accuracy matches, as shown above. The edits were built once on the 64-slot test jets and are identical across jobs; the aggregation asserts that the labels and edit rows match across all 13 files.
- **Turn edit**: by construction it is invisible to the (pT, r) rank-window pool. A program would need azimuthal inputs, such as the E14 half-plane sums, to describe these call changes.
- **Recurrence tolerance**: depends on the 0.05 knot tolerance and on matching by exact pool input. Neighbouring rank windows, such as top26 vs all, count as different statements. Neither choice was varied.
- **n32**: its program inputs are computed from 32 slots, so its features and knots differ from those of the 64-slot networks. This lowers its recurrence counts.
- **Untrained floor**: uninformative for fidelity and edits (constant calls). It is used only for the recurrence counts.
- **Smoke-stage change**: K = 2 recurrence and the family term shares were added after the smoke run and before the main aggregation, because at K = 8/16 every family is also present under the null.
- **Memory**: per-network jobs reached about 2.0 GiB pod peak, which includes page cache, at a 2 GiB request (the NRP-exempt size). They were not OOM-killed.
- **Data-label correlations**: none are used as mechanism evidence. All program comparisons are against network calls on real or edited inputs.

## Explainer entry (for site/explain.py, Paper B tab)
- **Question**: is the extracted program a description of one network, or of the jet-tagging task that all the networks learn?
- **Background**: a program is a short sum of hinge functions of particle sums ($\sum p_T$, $\sum p_T r^2$, counts) taken over $p_T$-rank windows, fitted to the network's 5 outputs. Its size is counted in bits (DL).
- **Method**: fit the same recipe to 10 trained networks and 3 untrained ones on the same 5,000 jets, then score on 4,813 test jets:
  - same call as the network;
  - the share of the network's call changes reproduced under 4 input edits (shift, turn, split, soft emission);
  - whether the same terms appear in other networks' programs;
  - how well program A predicts network B, compared with B's own program and with a program fitted to the truth labels.
- **Result** (figure `b9.png`):
  - the fidelity-vs-size curves overlap, with K16 at about 6,600 bits and 87–90 % same call;
  - each network's own program beats matched-size baselines on the shift edit;
  - about 30 of 80 terms per network recur above chance;
  - the transfer gap is 1.3 pp on natural jets and 2.7 pp in shift recall.
- **Checks**:
  - official n64 accuracy is reproduced;
  - the shuffled-logit program reaches 19 % and the random-feature program 62–66 %;
  - recurrence is tested against a key-relabel null and an untrained floor;
  - Holm correction is applied over networks and over terms.
- **Why it matters**: the extraction transfers to retrained networks, so it is not tuned to one checkpoint. But most of a program's content is shared with a program fitted to the truth labels, so only a 1–3 pp margin of it is specific to the network. The pool also cannot express the network's response to an azimuth-only change.
- **New glossary terms**:
  - **Recurrence count**: the number of other networks whose program contains the same statement.
  - **Key-relabel null**: the same programs with each term's input redrawn at random.
  - **Truth program**: the same recipe fitted to the labels instead of to a network.
  - **Transfer**: one network's program scored against another network's calls.
