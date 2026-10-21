# E36: smallest call-flipping edits at fixed jet pT, mass and axis (JEDI-linear 3f n64)

Claims A4, A5, B (exploratory method). Script `interp/e36_counterfactual_flip_jedi.py`; figure `interp/e36_plot.py` → `results/jedi/e36_edit.png`.
Results `results/jedi/e36.json` (final run e36-run3), `e36_edits.npz` (all edits of the main sample), `e36_examples.npz` (30 W→Z examples);
first full run with a defective random control kept as `results/jedi/e36_run1.json`. Logs `logs/e36-run.log`, `e36-run2.log` (crash), `e36-run3.log` (final).

## Conclusion
We asked the network to show, jet by jet, the smallest change that turns its W call into Z (or Z into W, quark into gluon, gluon into quark) while the jet's pT, mass and axis stay exactly the same. The changes it is most sensitive to are not the radial moves that the extracted program is built from: they mostly move pT between particles, pushing pT into the one to three hardest particles (pT^D, the pT-dispersion of the jet, rises by 1.6 standard deviations). Random changes of the same size almost never change the call (1.7 % vs 41 %). Two caveats limit the result. The float copy of the network used for the search and the real fixed-point network agree on only 41 % of the found changes. The changed jets also leave the region where real jets lie more often than random ones do. The extracted programs predict the network's flips on these changes no better than a small classifier on mass, pT, τ21, D2 and multiplicity (about 61 % vs 59 % agreement).

## 1. Question and design
Earlier edits were chosen by hand (shift, dilation, split, boost). Here a constrained search finds, for each jet, the smallest input edit that changes the call. The question is which input directions the network responds to most once mass, pT and axis are fixed, and whether the extracted programs (E11 nested hinge program, B6 rank-window program) predict these flips.

- **Network**: official JEDI-linear 3-feature n64 (inputs pT, Δη, Δφ about the stored jet axis); exact fixed-point forward pass `jedi_core.Jedi.run`.
- **Search model**: the float + masks program (as `interp/e3m_masks.py`: HGQ zero-bit input entries kept as 0/1 masks, no rounding, no wrap), written in jax so gradients exist.
- **Samples**: main `sample(1000, 3601)` (5,000 jets); replicate `sample(1000, 3602)` minus 91 overlaps (4,909 jets); pool `sample(2000, 3603)` minus both (9,622 jets; matching, kNN reference, MLP baseline).
- **Accuracy check (unedited)**: exact network 81.78 % on main (N 5,000; binomial 95 % interval about ±1.1 points, overlaps the 80.6–81.3 % reference), 81.36 % on replicate. Float + masks 81.22 % / 81.08 %; same call as exact 95.4 % / 96.0 %. Only jets on which float and exact agree before editing enter (4,770 main, 4,711 replicate).
- **Edit parametrisation**: each real particle i gets pT_i → pT_i·exp(u_i) (pT stays ≥ 0), Δη_i → Δη_i + 0.02·w_i, Δφ_i → Δφ_i + 0.02·w'_i. Padding slots and the particle count are unchanged. After every step the particles are re-sorted by pT (slot = pT rank, as in the data).
- **Edit size**: $\varepsilon^2 = \sum_i [(\Delta p_{T,i}/p_{T,i})^2 + (\Delta\eta_i^2 + \Delta\phi_i^2)/0.02^2]$. So ε = 1 means one particle's pT changed by 100 %, or one particle moved by 0.02 rad. Note that cutting a particle's pT to zero costs at most 1, while doubling it also costs 1. The metric is relative per particle, so moving absolute pT into the hardest particle is cheap.
- **Constraints**: the jet four-momentum $\sum_i p_i$ (massless particles, axis at (0, 0)) is held at its unedited value. This fixes jet pT, invariant mass and E-scheme axis (E-scheme axis = direction of the summed four-momentum). After every Adam step a Gauss–Newton minimum-norm projection in (u, w, w′) space is applied in float64. Achieved (99th percentile over found edits): |Δm|/m 7.8e-13, axis shift 2.5e-14 mrad, |ΔpT|/pT 1e-15 (tolerances 0.1 %, 1 mrad). Consequence: with m, M0 = ΣpT and the axis (so M1 ≈ 0) fixed, $M_2 = \sum p_T r^2$ is fixed to first order ($m^2 \approx M_0 M_2 - |M_1|^2$). Measured ΔM2 of the found edits is −1e-5 sd. So the radial M2 route of claim A4 is closed by design, and the search explores the directions left over.
- **Search**: Adam, 200 steps, lr 0.05, loss ε² + λ·relu(max_{k≠t} logit_k − logit_t + 0.05). λ starts at 0.3 and is multiplied by 1.1 per step while the target is not called, by 0.9 when it is. The found edit is the smallest-ε projected point at which the float program calls the target t. Targets: called W → Z, Z → W, q → g, g → q (primary); called top → its second-highest class (secondary).
- **Transfer** = share of found edits on which the exact fixed-point network also calls t. A secondary "robust" edit, the smallest edit with float margin below −0.5 logit, is also recorded.
- **Controls**
  1. Random edits: 100 per jet on main, 20 on replicate, each of the same v-space size. A Gaussian direction is projected onto the tangent space of the four constraints, scaled to the found edit's size, then projected back onto the constraint surface (30 Gauss–Newton steps). Flip rate to t on the exact network. All draws met the tolerances. Their median ε is 1.14× the found ε, so this control is not smaller than the found edits. In the first run, 4 projection steps were used and some draws violated the mass constraint (random flip rate 4.2 %). That run is superseded.
  2. Untrained network: 5 redraws (`Jedi(untrained=True, seed 0–4)`), same search on 400 jets each.
  3. On-manifold check. C2ST (classifier two-sample test): a HistGBDT trained to tell two jet sets apart, scored out-of-fold; AUC 0.5 means it cannot tell them apart. It is applied to edited jets vs real jets of the target class matched one-to-one in (log m, log pT) from the pool, using the E31 observable set (m, pT, N, τ21, τ32, D2, girth, pT^D, centroid term). References: unedited jets and random-edited jets against the same matched set, plus edited vs unedited. The kNN support test of E31v uses the mean distance to the 10 nearest pool jets in standardised log-feature space. "Outside" means above the 99th percentile of held-out real jets, so 1 % of real jets are outside by construction. It is run on the obs8 and axis-free "inv" feature sets.
- **Programs**
  - E11 nested hinge program: stored `e11_fit_*.json`, physics recomputed from the edited particles.
  - B6 rank-window program prog30: stored terms; coefficients refit by least squares on B6's fit split, reproducing 90.8 % same call there.
  - Baseline MLP(log m, log pT, τ21, log D2, N): 2 × 32, trained on the exact network's calls on the pool, validation same call 0.887.
  - Per jet, agreement = the program flips to t iff the network flips to t (program flips counted only where its unedited call equals the network's). McNemar compares network-flip vs program-flip, and pairs of predictors on agreement.
- **Statistics**: 95 % intervals by 1,000-sample bootstrap over jets; paired bootstrap p (floor 0.001); McNemar exact binomial; Holm over the 19 live S units.
- **H0**: found edits flip the exact network no more often than random same-size edits, and load on pT ranks, radii and coordinates like random edits. **H1 (thesis)**: transfer ≥ 70 %, random < 5 %, edit size concentrated on the radial coordinate of particles in the rank windows of S1/S30/S0, program agreement ≥ 70 %.

## 2. Results (main sample; replicate in brackets where it was run)

### Search and transfer (primary W/Z/q/g jets)
| quantity | main | replicate |
|---|---|---|
| float program flipped within 200 steps | 74.4 % of 4,770 | 74.8 % of 4,711 |
| primary jets with a found edit inside tolerances (N) | 2,783 | 2,752 |
| median ε (q25–q75) | 6.0 (2.8–14.0) | 6.2 (2.8–14.6) |
| **transfer to the exact network** | **41.1 % [39.3, 42.8]** | 41.5 % [39.5, 43.2] |
| random same-ε edits: flip to target (mean of 100 / 20 draws) | **1.65 % [1.39, 1.94]** | 1.82 % [1.54, 2.13] |
| random same-ε edits: any call change | 6.2 % [5.6, 6.9] | 7.2 % [6.5, 7.9] |
| found − random flip rate | +0.394 [0.376, 0.411], p 0.001 | +0.396 [0.377, 0.414] |
| McNemar found vs first random draw (found only : random only) | 1,106 : 16, p 1e-302 | 1,115 : 18, p 2e-302 |
| robust edit (float margin < −0.5): found on / transfer / ε ratio | 29.6 % / 67.5 % [65.0, 70.2] / 1.15× | 30.7 % / 67.3 % [64.8, 69.9] / 1.16× |

Per group (main), transfer / random flip / median ε:
- W→Z: 50.4 % / 2.7 % / 16.3
- Z→W: 32.8 % / 1.1 % / 3.7
- q→g: 8.8 % / 0.6 % / 6.8 (float program flipped only 41 % of q jets)
- g→q: 54.1 % / 1.7 % / 5.1
- top→2nd (secondary): 17.6 % / 0.6 % / 7.8

The float search often stops right at the float decision boundary (margin 0.05). The 8-bit-scale rounding of the real network then decides 59 % of these cases the other way. Edits 15 % larger with margin 0.5 transfer at 67 %.

### Where the edit size goes (share of ε², found vs random same-ε; paired difference with 95 % CI, all p 0.001 unless stated)
| part of the edit | found | random | difference |
|---|---|---|---|
| pT changes | 0.670 | 0.414 | +0.256 [0.241, 0.272] |
| radial moves | 0.256 | 0.295 | −0.039 [−0.050, −0.028] |
| azimuthal moves (about the particle's own position) | 0.073 | 0.291 | −0.218 [−0.224, −0.212] |
| particles of pT rank 0–13 (34 % of particles) | 0.601 | 0.263 | +0.339 [0.328, 0.349] |
| radial moves of ranks 0–13 | 0.117 | 0.092 | +0.025 [0.020, 0.030] |
| radial moves of ranks 14–63 | 0.139 | 0.203 | −0.064 [−0.072, −0.055] |
| r < 0.1 / 0.1–0.2 / ≥ 0.2 | 0.601 / 0.219 / 0.180 | 0.608 / 0.247 / 0.145 | −0.005 (p 0.35) / −0.027 / +0.033 |
| particles with pT < 11 GeV (the a32 low-pT ramp region) | 0.346 | 0.657 | −0.311 [−0.321, −0.300] |
| the 3 particles with the largest share | 0.471 | 0.337 | +0.13 [0.12, 0.14] |

Replicate: same pattern to within 0.02 (pT 0.663 vs 0.417; azimuthal 0.074 vs 0.293; ranks 0–13 0.596 vs 0.264).

From the saved edits (main, N 2,783):
- The found edits change the pT of the hardest particle by a median +83 % (random +0.8 %).
- The summed |ΔpT| over particles is a median 74 % of the jet pT (random 40 %).
- On average 5.1 particles per jet lose > 90 % of their pT (random 1.9); their median original rank is 15.

The typical found edit takes pT from mid-rank particles and gives it to the leading one to three particles at fixed total four-momentum. In physical terms these edits are large, not small. Signed radial moves differ by direction:
- W→Z: rank 14–63 particles move out by +12 mrad on average; 59 % of jets move the hard particles outward.
- Z→W: hard particles move in by −8 mrad; 8 % outward.

### S units of the exact network (|ΔS| in units of the unit's SD over unedited jets; found vs random mean of 10 draws; Holm over 19, all Holm p 0.019 = bootstrap floor except S58)
| unit (what it computes, E11 fit) | found | random | ratio |
|---|---|---|---|
| S20: hinge on leading-particle pT (pT_0 > 253 GeV; slots 0–2) | 1.94 | 0.27 | 7.2 |
| S51: pT of the 2–3 hardest | 1.26 | 0.24 | 5.3 |
| S43: pT of the 2 hardest + ΣpT·r over ranks 0–13 (rank 13/14 window object) | 1.18 | 0.22 | 5.4 |
| S17: ΣpT·r of the 11 hardest | 0.55 | 0.22 | 2.6 |
| S13: ΣpT·r² of the 11 hardest | 0.54 | 0.11 | 4.8 |
| S52, S40 (radial sums over ranks 0–13) | 0.54, 0.42 | 0.11, 0.08 | 4.9, 5.3 |
| S0: ΣpT of the 14 hardest (via g) | 0.26 | 0.05 | 5.3 |
| S19: soft multiplicity (N over ranks 13–63) | 0.27 | 0.15 | 1.8 |
| S29: ΣpT r² + Σr² + soft count (largest ablation cost, 40.8 %) | 0.24 | 0.09 | 2.7 |
| S1: Σ_{ranks ≥ 1}(r − 0.2)₊ | 0.22 | 0.13 | 1.6 |
| S30: Σr (core) | 0.20 | 0.13 | 1.5 |
| S58 (g-only) | 0.32 | 0.30 | 1.04 (Holm p 0.11) |

Replicate ratios agree within 0.4. The named radial objects (S1, S29, S30, S19) move 1.5–2.7× more than under random edits. The largest moves are in the leading-particle pT objects S20, S51 and S43.

### Jet observables (Δ / SD over the sample; found vs random)
| observable | found | random | note |
|---|---|---|---|
| pT^D | +1.57 [1.49, 1.64] | +0.09 | per group W→Z +1.69, Z→W +1.33, g→q +2.52, q→g −0.52 |
| N(pT < 2 GeV) | +0.85 [0.78, 0.93] | +0.47 | |
| τ21 | +0.10 [0.06, 0.14] | +0.53 | W→Z +0.86; real Z − W gap is −0.06 sd |
| τ32 | −0.23 | +0.13 | |
| D2 | +0.08 | −0.06 | |
| girth | −0.05 | +0.004 | |
| m, pT, N | 0 (fixed) | 0 (fixed) | |

The found edits do not move τ21, D2 or girth toward the target class's mean (W–Z gaps ≤ 0.25 sd). For q↔g the pT^D direction matches the class gap (g → q: pT^D +2.5 sd; real q − g gap +1.45 sd).

### Programs on the found edits (main; replicate in brackets)
| predictor | same call unedited | agreement with network flips | recall (flips the network made) | specificity | agreement on random edits |
|---|---|---|---|---|---|
| E11 nested hinge program | 0.954 | 0.608 [0.589, 0.625] (0.609) | 0.24 (0.26) | 0.86 | 0.973 |
| B6 rank-window prog30 | 0.920 | 0.605 [0.587, 0.623] (0.609) | 0.17 (0.17) | 0.90 | 0.979 |
| MLP(m, pT, τ21, D2, N) | 0.906 | 0.593 [0.575, 0.611] (0.587) | 0.03 (0.02) | 0.99 | 0.976 |

- E11 − MLP agreement: +0.014 [−0.003, 0.031], McNemar p 0.088 (replicate +0.021 [0.004, 0.039], p 0.013).
- B6 − MLP: +0.011 [−0.003, 0.025], p 0.11 (replicate +0.022 [0.009, 0.035], p 0.001).
- E11 − B6: +0.003, p 0.79.
- Network flips vs program flips, McNemar: E11 866 : 226 (p 1e-88), B6 942 : 158.
- Agreement is highest where the network rarely flips (q→g 0.86–0.90) and lowest where it flips most (W→Z 0.50–0.52, g→q 0.51–0.54).

### On-manifold check (main, N 2,782; permutation p 0.0033 = floor with 300 swaps)
| comparison | C2ST AUC [CI] |
|---|---|
| edited vs matched real target-class jets | 0.959 [0.954, 0.964] |
| unedited vs the same matched jets | 0.887 [0.877, 0.896] |
| random same-ε vs matched jets | 0.934 [0.927, 0.940] |
| edited vs unedited (same jets) | 0.900 [0.891, 0.908] |
| random same-ε vs unedited | 0.811 [0.800, 0.821] |

kNN support, share outside the 99 % real-jet region:
- obs8: edited 16.5 % [15.1, 17.8], random 1.0 %, unedited 0.8 %.
- inv: edited 17.0 %, random 1.1 %, unedited 0.8 %.

The edited jets are further from the target class than the unedited jets are, and 16× more often outside the real-jet region than random same-size edits.

### Untrained network
All 5 redraws call one class for all 400 jets (top, W, Z, top, q). Logit SD across jets is 0.0004–0.008 (trained network: O(1)). The search found no flip in 200 steps for any jet. The edit-size comparison with the trained network (median ε 6.7 on the same jets, 72 % found) could not be made. The untrained control here only shows that the constant-output redraws have no call boundary within reach.

## 3. Reading (scoped to this measurement)
- **H0, flip part**: excluded. Found edits flip the exact network on 41 % of jets vs 1.7 % for random edits of the same size and constraints (McNemar p 1e-302, both samples).
- **Thesis numbers**:
  - transfer ≥ 70 %: not met (41 %; 67 % for margin-0.5 edits on the 30 % of jets that have one);
  - random < 5 %: met (1.7 %);
  - radial concentration in the S1/S30/S0 rank windows: not met. Radial share 0.26 vs 0.30 random. The excess is in the pT coordinate (0.67 vs 0.41) of the hardest particles (rank 0–13 share 0.60 vs 0.26; leading-particle pT median +83 %);
  - program agreement ≥ 70 %: not met (0.61 for both programs; not separated from the MLP baseline on the main sample, +0.02 on the replicate).
- **Alternatives**:
  - "azimuthal half-plane / odd direction term": excluded for these edits (azimuthal share 0.07 vs 0.29 random);
  - "a few soft particles (count)": not supported as the main route (share on pT < 11 GeV particles 0.35 vs 0.66 random). Soft multiplicity does rise (N(pT < 2 GeV) +0.85 sd) as pT is drained from mid-rank particles;
  - "flips do not transfer / programs disagree": partly supported (59 % of float flips do not transfer; programs recall 17–26 % of the network's flips).
- **Scope**:
  - Fixing m, pT and axis also fixes M2 to first order, so the search cannot use the A4 mass route. The result describes the network's sensitivity in the directions that remain.
  - The ε metric treats relative pT changes of every particle equally, which makes large absolute pT transfers into the hardest particle cheap. Another metric (absolute pT, energy-flow distance) could select different directions.
  - The found jets are 16 % off-support and look less like the target class than before the edit, so they probe the network away from the data.
- **Physics**: at fixed mass, pT and axis, the cheapest way to change JEDI-linear's W/Z or q/g call is to change how jet pT is shared among the leading one to three particles (pT^D). The network reads this through its leading-particle pT objects S20, S51 and S43. This is a hardness-sharing variable, not a radial-shape variable, and the extracted programs capture it only partly.

## 4. Explainer entry (for site/explain.py, Paper A tab; the ledger step edits site/)
- **Question**: if we let the network choose the smallest change that flips its call, at fixed jet mass, pT and axis, what does it change, and does our extracted program predict those flips?
- **Background**: earlier tests used hand-chosen edits (shift, dilation, split, boost). A counterfactual search lets the network show its own most sensitive directions. With mass, pT and axis fixed, the M2 = Σ pT r² route is closed to first order, so the search looks for what else the network uses.
- **Method**: per jet, minimise $\varepsilon^2=\sum_i[(\Delta p_{T,i}/p_{T,i})^2+(\Delta\eta_i^2+\Delta\phi_i^2)/(0.02)^2]$ subject to $\sum_i p_i^\mu$ fixed, using 200 Adam steps on the differentiable float + masks program with projection after each step. Then test on the exact network, against 100 random edits of equal ε, an untrained network, C2ST/kNN on-manifold checks, and the E11/B6 programs plus an MLP baseline.
- **Result (figure `results/jedi/e36_edit.png`)**: 41 % of the found edits flip the exact network vs 1.7 % for random edits. The edit size goes into pT changes of the hardest particles (rank 0–13 share 0.60 vs 0.26 random; azimuthal share 0.07 vs 0.29). pT^D rises 1.6 sd. The leading-particle pT units S20/S51/S43 move 5–7× more than under random edits. The programs agree with the network on 61 % of jets, about the same as an MLP on (m, pT, τ21, D2, N).
- **Checks**:
  - constraints held to 1e-12;
  - random edits all inside the tolerances;
  - replicate sample reproduces every number within 0.02–0.4;
  - edited jets 16 % off the real-jet support (random 1 %).
- **Why it matters**: it tests the program on edits it was not built for and finds a direction (hard-particle pT sharing) that the radial-object description does not cover.

## 5. Glossary additions
- **ε (edit size)**: $\sqrt{\sum_i[(\Delta p_T/p_T)^2+(\Delta\eta^2+\Delta\phi^2)/0.02^2]}$ over real particles; ε = 1 is one particle's pT changed by 100 % or moved 0.02 rad.
- **Transfer rate**: share of edits found on the float + masks program that also flip the exact fixed-point network.
- **Counterfactual flip search**: gradient search for the smallest input edit that changes the network's call to a chosen class.
- **pT^D**: $\sqrt{\sum p_T^2}/\sum p_T$, the pT dispersion of the jet; high when few particles carry most of the pT.
- **E-scheme axis**: direction of the summed four-momentum of the particles.
- **C2ST**: classifier two-sample test; AUC of a classifier separating two jet sets (0.5 = indistinguishable).
- **McNemar test**: exact binomial test on the discordant pairs of two paired yes/no outcomes.
- **Holm correction**: step-down multiple-testing correction of p-values.

## 6. Runs, resources, issues
- Jobs:
  - e36-smoke (crash: C2ST on 7 jets);
  - e36-smoke2/3 (200 jets per sample, lr 0.02 vs 0.05; lr 0.05 kept);
  - e36-run (1 CPU / 3 GiB; peak 3,071 MiB, at the limit; random-edit projection with 4 Gauss–Newton steps left some draws off the mass constraint, so its random numbers are superseded, file e36_run1.json);
  - e36-run2 (crash: variable used before assignment in the compliance fix);
  - **e36-run3** (final; 1 CPU / 4 GiB, peak 2,051 MiB, 0.89 cores, wall 1,020 s; earlier runs spent 15 min on a slow node at start-up).
  - Two debug attempts used.
- jax prints a CUDA plugin error at start-up in the CPU env (harmless; the run uses CPU).
- Not done:
  - untrained edit-size comparison (no flips exist in the redraws);
  - C2ST on the replicate;
  - the McNemar of network vs program per group (by-group agreement is in the JSON);
  - λ/lr schedules other than one were not scanned, so ε values are those of this optimiser (an upper bound on the true minimum).
- Packages: sklearn, scipy, matplotlib present (figure rendered locally from the JSON).
