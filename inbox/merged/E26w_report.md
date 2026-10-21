# E26w: is total pT (S0) a specific class feature under the mass-preserving boost? (claims A14 / A3; round 12, stage 4)

Script `interp/e26w_boost_specificity.py` (plot `interp/e26w_plot.py`). Job `e26w-run` (1 CPU / 2 GiB, wall 2,427 s, exit 0,
0.66 cores on average; the 2,047 MiB peak includes the page cache from reading the 558 MB training archive). Smoke job
`e26w-smoke2` passed. A first smoke pod stalled in CephFS `ceph_mdsc_wait_request` on node k8s-gpu-2.ucsc.edu. I deleted it and
added the node to the `job.sh` exclusion list (one line, commented). No code bugs. Results: `results/jedi/e26w_boost_specificity.json`,
figure `results/jedi/e26w_boost_specificity.png`, log `logs/e26w-run.log`.

## Conclusion
At the 10 % boost that lowers jet pT, holding the network's internal total-pT quantity at its pre-boost value removes about 56 % of
the changed decisions. That is far more than holding the direction of largest data variance (3 %), and more than random directions
that carry the same data variance (raw p = 0.018). Random directions that the boost moves by the same amount remove as much or
more, so the effect follows the size of the boost displacement in that layer rather than anything special about total pT. Most of
total pT's boost change (68 %) reaches it through a context unit computed from the whole jet, not through the direct per-particle
sum. In the training data, total pT separates classes at fixed jet mass (W vs top AUC 0.85, W vs Z 0.63), so using it is
consistent with the training sample and is not an artefact of training. Whether that separation is physics or a property of how the
dataset's jet pT window was filled per class is not tested here.

## Question and setup
- **Network:** JEDI-linear 3-feature n64 (official checkpoint). Layer letters as in `interp/jedi_core.py`:
  - a: first-layer per-particle cuts;
  - P: pooled first-round objects;
  - g: context ("jet statement") units, computed from P and added to every particle's second-round input;
  - S: pooled second-round objects, 64 coordinates, 19 of them live;
  - h4, h5: dense layers after S.
- **Units named:**
  - S0 = sum of pT of the 14 hardest particles (the network's total-pT object);
  - S29 ≈ pT-weighted sum of r² plus soft terms (an M2-like angular-spread object);
  - g49 = context unit fed by summed pT;
  - a15 = first-layer cut that is linear in particle pT.
- **Sample:** `interp.jedi_core.sample(per_class=2000, seed=1203)`, N = 10,000 test jets (2,000 per class g, q, W, Z, t).
  - Unedited accuracy 0.8143. The binomial 95 % range around 0.809 at N = 10,000 is 0.801–0.817; E26 / E26v gave 0.8068 / 0.8066 on other seeds.
  - The exact fixed-point S / h4 / h5 tail reproduces the network (max |Δlogit| = 0).
  - Directions used for the nulls are built on a separate sample(2000, seed 1204), so the evaluation jets do not determine them (the two samples share 398 of 10,000 jets, 4 %, because both are random draws from the same test set):
    - principal components (PCs) of S;
    - the S covariance Σ;
    - the boost-displacement second moment C_D = E[D Dᵀ], with D = S(boosted) − S(unboosted).
  - DAS (distributed alignment search: a learned k-dimensional subspace of S that, when swapped between two jets, transfers an
    M0-only or M2-only edit's effect on the call) was refit exactly as in E26 / E26v: sample(4000, 0), 8,000 pairs, seed 10 + k.
    Validation balanced agreement: M0 k1 0.90, M2 k1 0.70, M0 k2 0.91, M2 k2 0.76.
    Here M0 = Σ pT and M2 = Σ pT r² about the jet axis.
- **Edit B (imported from `interp.h12_boost_panel`, as in E26):** the exact Lorentz boost of every particle along the jet momentum.
  - It sets |P_jet| → λ|P_jet| with the jet invariant mass unchanged: max |Δm|/m = 6e-12 (λ 0.9) and 1e-11 (λ 1.1).
  - At λ = 0.9, jet ΣpT falls by 9.95 % and particle angles to the axis grow by about 1/λ. λ = 1.1 is the opposite.
  - B changes 32.1 % [31.1, 33.1] of calls at λ 0.9 and 14.9 % [14.2, 15.7] at λ 1.1.
  - The network's S0 moves by −5.30 on average at λ 0.9 (unboosted mean 18.33, i.e. −29 %) and by +5.42 at λ 1.1.
- **Hold:** the network sees the boosted jet, but the projection of S onto a chosen k-dimensional subspace Q is reset to its unboosted
  value: S_hold = S_B + (S_0 − S_B) Q Qᵀ. The result is clipped to the next layer's representable range, as in E26v.
- **Metric:** removed share = 1 − #(calls that differ from the unboosted call with the hold) / #(calls that differ under B alone).
  - The share is net, so it can be negative when the hold creates more changes than it removes.
  - Intervals are 1,000 Poisson bootstraps over jets, with the same weights for every hold.
  - McNemar = exact binomial test on the paired call changes.
  - Rank p = (1 + #null ≥ observed) / (1 + #null).
  - Holm = step-down family-wise correction.

## Arm 1: S-level holds vs matched live nulls
Null families, all drawn only on the 19 live S coordinates, with 1,000 draws per family and λ:
- **iso:** isotropic random k-dimensional subspaces.
- **var (variance-matched):** Q = orth(Σ^{p/2} Z), with Z Gaussian and the tilt p bisected per draw so that tr(QᵀΣQ) equals the
  variance the named hold captures. For S0 that is Var(S0) = 29.7, the second-largest unit variance; the top PC variances are
  128.7, 31.2, 20.6 and 15.2.
- **shift (boost-shift-matched):** the same construction with C_D in place of Σ. A null direction then moves as much under the boost
  as the named direction does.

Geometry: S0 lies mostly on PC2 (cos² 0.80), and PC1 is mostly S29 (cos² 0.57). The variance-matched null draws overlap the top PC
by 0.14 on average, so they do not collapse onto PC1.

| hold (k) | λ 0.9 removed [95 % CI] | iso 97.5 % / max | var 95 % / 97.5 % | p iso / var / shift | λ 1.1 removed | p iso / var / shift |
|---|---|---|---|---|---|---|
| S0 (1) | 55.9 % [54.0, 58.0] | 32.3 / 50.8 | 43.2 / 51.8 | 0.001 / 0.018 / 0.72 | 0.1 % [−5.8, 6.0] | 0.13 / 0.15 / 0.72 |
| DAS M0 (1) | 61.5 % [59.6, 63.6] | 32.3 / 50.8 | 43.1 / 50.8 | 0.001 / 0.009 / 0.57 | 3.9 % [−2.0, 10.1] | 0.05 / 0.10 / 0.75 |
| S29 (1) | 4.1 % [2.6, 5.7] | | 39.9 / 45.4 | 0.25 / 0.45 / – | −7.2 % | 0.33 / 0.07 / – |
| DAS M2 (1) | 16.0 % [14.2, 17.8] | | 57.8 / 71.2 | 0.12 / 0.26 / – | −40.8 % | 0.71 / 0.57 / – |
| PC1 (1) | 2.6 % [0.0, 5.0] | | – | 0.27 / – / – | −129.5 % | 0.91 |
| S0+S29 (2) | 77.7 % [76.0, 79.2] | 37.9 / 53.2 | 60.9 / 68.2 | 0.001 / 0.004 / 0.18 | 51.0 % [47.4, 54.3] | 0.001 / 0.001 / 0.16 |
| DAS M0+M2 (2) | 84.3 % [82.9, 85.7] | 37.9 / 53.2 | 51.6 / 60.4 | 0.001 / 0.001 / 0.001 | 54.8 % [51.1, 58.4] | 0.001 / 0.001 / 0.007 |
| PC1-2 (2) | 48.9 % [46.6, 51.3] | | – | – | −5.5 % | – |
| DAS M0+M2 (4) | 84.3 % [83.0, 85.7] | 36.8 / 54.3 | 59.3 / 63.7 | 0.001 / 0.001 / – | 64.8 % [61.9, 67.8] | 0.001 / 0.001 / – |
| PC1-4 (4) | 78.5 % [76.9, 80.0] | 36.8 / 54.3 | – | 0.001 / – / – | 53.0 % [49.7, 56.5] | 0.001 |

The shift-matched null quantiles at λ 0.9 (median / 95 % / 97.5 %) are:
- S0: 62.6 / 78.0 / 79.3 %;
- S0+S29: 70.1 / 81.2 / 82.2 %;
- DAS M0+M2 (2): 54.9 / 73.7 / 75.8 %.

Holm-adjusted p values:

| null family (number of holds) | hold | λ 0.9 | λ 1.1 |
|---|---|---|---|
| var (14) | S0 | 0.13 | 0.60 |
| var (14) | DAS M0 (1) | 0.07 | – |
| var (14) | S0+S29 | 0.036 | 0.014 |
| var (14) | DAS M0+M2 (2 and 4 dims) | 0.014 | 0.014 |
| iso (18) | S0 | 0.018 | 0.86 |
| shift (8) | S0 | 1.0 | 1.0 |
| shift (8) | DAS M0+M2 (2) | 0.008 | 0.049 |

Paired comparisons (difference in removed share, bootstrap CI, McNemar on the call changes that remain):
- **S0 − PC1:** +53.4 pp [50.2, 56.4] at λ 0.9 (remaining changes: 687 only with S0, 2,400 only with PC1; p 1e-220) and
  +129.5 pp [121, 138] at λ 1.1.
- **S0 − DAS M0 (1):** −5.5 pp [−6.7, −4.3] at λ 0.9.
- **S0+S29 − PC1-2:** +28.8 pp [26.4, 31.2].
- **DAS M0+M2 (4) − PC1-4:** +5.9 pp [4.3, 7.6] at λ 0.9 and +11.7 pp [8.1, 15.1] at λ 1.1.

Class flow with the S0 hold at λ 0.9 (unboosted call → call under the hold):
- Of the 2,015 jets called W, 806 still go to Z and only 22 go to t. In E26, B alone sent 1,167 of 2,034 W calls to t.
- The S0 hold removes −24 % of the W-jet l_W − l_Z shift, i.e. it adds to that shift. The S0 hold therefore acts on the
  W/Z → t changes and not on the W ↔ Z part; DAS M0+M2 (2) removes 97 % of the l_W − l_Z shift.

Reading:
- At λ 0.9, S0 removes more than PC1, more than isotropic live directions, and more than the 95th percentile of variance-matched
  directions (raw p 0.018; Holm over 14 holds 0.13).
- It removes less than the median random direction with the same boost displacement (p 0.72).
- At λ 1.1, S0 removes 0.1 %, inside every null.
- The 2-dimensional DAS M0+M2 subspace exceeds all three nulls at both λ.

## Arm 2: single h4 / h5 units vs live random-direction nulls
Each live unit was held alone (55 h4 and 32 h5 units). Nulls: 4,000 isotropic random 1-dimensional directions on the live h4 or h5
coordinates, with Holm over the units of that layer. Variance-matched 1-dimensional nulls (500 draws) were run for the E26-named
units and the top 3 units per λ.

| λ | unit | removed | iso null 97.5 % / max | Holm iso | Holm var |
|---|---|---|---|---|---|
| 0.9 | h4 53 | 47.6 % | 8.8 / 25.6 % | 0.014 | 0.008 |
| 0.9 | h4 39 | 36.7 % | 8.8 / 25.6 % | 0.014 | 0.008 |
| 0.9 | h5 24 | 45.0 % | 20.1 / 47.6 % | 0.016 | 0.048 |
| 0.9 | h5 1 | 19.9 % | | 0.81 | 0.016 |
| 1.1 | h4 39 | 26.2 % | 10.4 / 26.5 % | 0.027 | 0.008 |
| 1.1 | h4 3 | 23.7 % | 10.4 / 26.5 % | 0.027 | 0.040 |
| 1.1 | h5 25 | 30.7 % | 12.0 / 30.4 % | 0.008 | 0.012 |
| 1.1 | h5 1 | 18.5 % | | 0.18 | 0.006 |
| 1.1 | h4 42 | 12.0 % | | 0.93 | 0.018 |

E26's S-level-null Holm results for h4 53 / h5 24 (λ 0.9) and h4 39 / h4 3 / h5 25 (λ 1.1) hold against their own layer's live
null. What these h4 / h5 units compute is not named yet.

## Arm 3: path-specific patch a15 → g49 → S0
S0 under the boost was split by patching g49 alone:
- S0_dir = S0(boosted jet, g49 at its unboosted value): the context route through g49 is reverted, the direct route is kept.
- S0_ctx = S0(unboosted jet, g49 at its boosted value): only the context route is active.

Decomposition of the mean boost change of S0 at λ 0.9 (−5.30):
- via g49: −3.61 (68 %);
- direct: −1.65 (31 %);
- interaction: −0.05.

g49 rises by 1.41 (unboosted mean 4.04). λ 1.1 gives the same 69 / 32 % split.

Removed share at λ 0.9 / λ 1.1 (other S units at their boosted values):

| intervention | λ 0.9 | λ 1.1 |
|---|---|---|
| S0 held at its unboosted value | 55.9 % [54.0, 58.0] | 0.1 % |
| S0, route via g49 reverted (S0 := S0_dir) | 50.0 % [48.1, 51.9] | 5.8 % [1.5, 10.5] |
| S0, direct route reverted (S0 := S0_ctx) | 20.6 % [19.1, 22.3] | 7.2 % [4.0, 10.0] |
| g49 held (all routes) | 50.0 % [48.1, 51.9] | 5.8 % |
| g49 held, S0 kept at its boosted value | 0.0 % (exactly; McNemar p 1) | 0.0 % |
| a15 held (per particle, all routes) | 41.1 % [38.9, 43.3] | −22.6 % |
| a15 effect via g only (g set to its value under the a15 hold) | 46.2 % [44.3, 48.2] (= via g49 only) | 6.2 % |
| a15 effect not via g (a15 held, g kept boosted) | 8.4 % [6.7, 10.2] | −9.4 % |
| a15 effect via S0 only | 55.5 % [53.5, 57.6] | 0.7 % |

Holm over the 10 path interventions: all λ 0.9 entries p < 1e-19, except g49 with S0 boosted (p 1).

With a15 held, S0's boost change shrinks from −5.30 to −0.31 and g49's from +1.41 to +0.11. a15 is therefore the source of about
94 % of the S0 change.

Reading:
- g49 acts on the calls only through S0.
- Most of S0's boost change, and the larger removed share, runs through the context route a15 → P → g49 → (second-round
  particle statements u) → S0, not through the direct pooled per-particle route.
- The two partial reversions sum to more than the full hold (50.0 + 20.6 > 55.9 %) because the call is a nonlinear function of S0.

## Arm 4: data check (training archive, N = 620,000; replicate on the n64 test set, N = 260,000)
Method:
- Jet mass from the particle four-vectors (massless particles).
- Stratified AUC = probability that a jet of the first class has the larger value than a jet of the second class in the same
  5 GeV jet-mass bin (bins 0–300 GeV with at least 20 jets of each class).
- 200 Poisson bootstraps.
- Within-bin label-permutation null, 200 permutations: the 95 % band is about [0.495, 0.506] for every pair, and p = 1/201 (the
  floor) for every pair and both sum-pT definitions.

| pair | Σ pT all particles | Σ pT 14 hardest (S0 definition) | multiplicity (reference) | jet mass within bin (residual) |
|---|---|---|---|---|
| W vs Z | 0.630 [0.624, 0.636] | 0.562 [0.557, 0.569] | 0.410 | 0.319 |
| q vs g (= 1 − g vs q) | 0.321 [0.319, 0.324] | 0.550 [0.547, 0.552] | 0.207 | 0.475 |
| W vs t | 0.852 [0.847, 0.857] | 0.737 [0.732, 0.742] | 0.342 | 0.422 |
| Z vs t | 0.834 [0.829, 0.839] | 0.737 [0.732, 0.742] | 0.297 | 0.502 |
| q vs t | 0.706 | 0.627 | 0.525 | 0.477 |

Test-set replicate:
- W vs Z: 0.642 (all particles), 0.584 (14 hardest).
- W vs t: 0.845 (all particles), 0.731 (14 hardest).
- q vs g: 0.323 (all particles), 0.555 (14 hardest).

Mean Σ pT (all particles / 14 hardest), training archive:

| class | all particles | 14 hardest |
|---|---|---|
| g | 1,135 GeV | 842 GeV |
| q | 1,019 GeV | 911 GeV |
| W | 1,043 GeV | 895 GeV |
| Z | 1,042 GeV | 889 GeV |
| t | 1,000 GeV | 734 GeV |

Reading:
- At fixed mass, W and Z jets have larger Σ pT of the 14 hardest particles than top jets (AUC 0.74).
- A boost that lowers Σ pT at fixed mass therefore moves W / Z jets toward the top region. This matches the W/Z → t flow that the
  S0 hold removes.
- Caveat: a 5 GeV bin does not fully fix the mass. Within-bin mass alone still gives W vs Z AUC 0.32, so a within-bin Σ pT–m
  correlation could contribute to the W vs Z number. The W vs t and Z vs t numbers (within-bin mass AUC 0.42 / 0.50) are less
  affected.

## Predictions: thesis vs alternative
**Thesis predictions:**

| prediction | result |
|---|---|
| S0 > 95th percentile of matched live nulls | Met at λ 0.9 for the iso and var nulls (raw p 0.001 / 0.018; Holm 0.018 / 0.13). Not met for the shift-matched null (p 0.72). Not met at λ 1.1. |
| S0 > PC1 at k = 1 | Met (+53 pp at λ 0.9). |
| Most of the effect through the direct route | Not met (context route 68 % of ΔS0; 50.0 vs 20.6 % removed). |
| Total pT has class information at fixed m (AUC > 0.55 for W vs Z or q vs g) | Met. All-particle Σ pT: W vs Z 0.63, q vs g 0.32. 14-hardest: W vs Z 0.562, q vs g 0.550. Strongest: W/Z vs t 0.74–0.85. |

**Alternative predictions:**

| prediction | result |
|---|---|
| S0 inside the live-null band | Excluded at λ 0.9 for iso / var. Not excluded for the shift-matched null. |
| S0 equal to PC1 | Excluded (McNemar p 1e-220). |
| Share runs through g49 | Supported (68 % of ΔS0 via g49; g49 acts only through S0). |
| No class information at fixed m | Excluded (permutation p 0.005 for all pairs). |

## Statistics summary
| item | value |
|---|---|
| N, seeds | N 10,000 (seed 1203); fit sample seed 1204, which shares 398 of its 10,000 jets with the evaluation sample (both are random draws from the same test set; 4 %) |
| Bootstrap | 1,000 Poisson draws |
| S-level nulls | 1,000 per family per λ |
| h-unit nulls | 4,000 iso per layer and λ; 500 variance-matched per unit |
| Holm families | S holds per null family over both λ (iso 18, var 14, shift 8); units per layer and λ (55 h4, 32 h5); path interventions per λ (10) |
| Data check | 200 bootstraps and 200 within-bin permutations per pair; p floor 1/201 |
| Controls | random live directions (three matchings); PC subspaces of the same dimension; untrained-network controls were not rerun (E26: thesis-set hold 0–21 % on untrained redraws) |

## Explainer entry (for site/explain.py, Paper A tab)
**Question.** When a jet is boosted along its own axis so that its mass stays the same but its pT drops by 10 %, about a third of
JEDI-linear's calls change. E26 found that holding S0 (the network's sum of pT of the 14 hardest particles) at its pre-boost value
undoes about half of these changes. Is that special to total pT, or would any high-variance internal direction do the same?

**Background.** The network pools per-particle statements into 64 numbers $S$, of which 19 vary. A *hold* replaces the component of
$S$ along a subspace $Q$ by its unboosted value:
$S_{\rm hold} = S_B + (S_0 - S_B)\,QQ^\top$.

The *removed share* is
$1 - N_{\rm changed}(\text{hold})/N_{\rm changed}(B)$.

A random direction is a fair comparison only if it is matched on what could make any direction matter:
- its data variance, $\mathrm{tr}(Q^\top\Sigma Q)$;
- how much the boost moves it, $\mathrm{tr}(Q^\top C_D Q)$ with $C_D = \mathbb{E}[DD^\top]$ and $D = S_B - S_0$.

**Method.**
1. Random subspaces on the live coordinates are tilted toward high variance, $Q=\mathrm{orth}(\Sigma^{p/2}Z)$, with $p$ chosen per
   draw so that the captured variance equals that of S0 (or so that the boost displacement matches, using $C_D$).
2. The S0 hold is compared with 1,000 such draws, with the top principal directions, and with DAS mass-moment subspaces.
3. A path patch splits S0's boost change into the part that arrives through context unit g49 and the direct part.
4. The training data are checked for whether $\sum p_T$ separates classes at fixed mass. Here the stratified AUC is the probability
   that a jet of one class has the larger $\sum p_T$ than a jet of the other class in the same 5 GeV mass bin.

**Result (figure `e26w_boost_specificity.png`).**
- At $\lambda=0.9$, S0 removes 56 % of the changes.
- PC1 removes 3 %; isotropic random directions remove at most 51 %; variance-matched directions remove 52 % at their 97.5th percentile.
- Directions that move as much under the boost remove 63 % at the median.
- 68 % of $\Delta S_0$ arrives via g49, and g49 acts only through S0.
- At fixed mass, W/Z jets have larger $\sum_{14} p_T$ than top jets (AUC 0.74), which matches the W/Z → t flow the hold removes.

**Checks.**
- Exact tail reproduces the network (max |Δlogit| = 0).
- Boost preserves the mass to 1e-11.
- The fit sample is drawn separately from the evaluation sample (398 shared jets, 4 %).
- Nulls were drawn only on live coordinates.
- Test-set replicate of the data check.

**Why it matters.** The boost sensitivity follows the size of the boost displacement in $S$. The total-pT object is one large
component of that displacement, not a uniquely privileged one. The dependence on total pT has a counterpart in the training data
(classes differ in $\sum p_T$ at fixed mass), so it is learned from the sample's kinematics rather than produced by training noise.

**Conclusion.** Same as the report's Conclusion above.

### Glossary additions (explain.GLOSSARY)
- **Variance-matched null:** random directions in the live coordinates, tilted so that each carries the same data variance as the
  tested direction.
- **Boost-shift-matched null:** random directions tilted so that the boost moves each by the same mean-square amount as the tested
  direction.
- **Stratified AUC (fixed-mass AUC):** the probability that a jet of class A has a larger value than a jet of class B drawn from the
  same jet-mass bin.
- **Path patch:** setting one unit to the value it would take if only one upstream route had changed, to split an effect between
  routes.
- **PC / PCk:** principal component(s) of the unedited S distribution, i.e. the directions of largest variance.

## Files
- `interp/e26w_boost_specificity.py`, `interp/e26w_plot.py`
- `results/jedi/e26w_boost_specificity.json`, `results/jedi/e26w_boost_specificity.png`, `logs/e26w-run.log`
- `job.sh`: node k8s-gpu-2.ucsc.edu added to the CephFS-stall exclusion list.
