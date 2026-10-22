# P63: do later ParT_full layers compute the true invariant mass?

Node P63 (loop C round C14; claims C3, C7, C4; user question). Status: done (one run, no verifier yet).
Script `interp/p63_true_mass_layers_part.py` (main run and `--plot`). Jobs: c-p63-smoke, c-p63-smoke2, c-p63-smoke3 (smoke and debug), c-p63-run (GPU, 39 min), c-p63-fig (CPU, figure).
Results: `results/part/p63_true_mass_layers.json`, `results/part/p63_example.npz`, figure `results/part/p63_true_mass_layers.png`, log `logs/c-p63-run.log`. Explainer `P63_EXPLAINER` and four glossary entries added to `site/explain.py`.

## Conclusion
ParT_full's particle representation holds, from the first block on, information that fixes the true invariant mass beyond the axis-moment mass $m_A$. Nonlinear read-outs recover about three quarters of the true-mass variation left after $m_A$ (untrained networks: at most 0.30), and about half of it on jets where only the true mass was changed. The decision makes little use of it. When the true mass changes at fixed $m_A$, the W-vs-Z score moves about a quarter as much as for a real mass change of the same size, and copying the best linear true-mass direction from one jet into another changes nothing. Open: a 5-number summary of the particle moments already contains this information, so these read-outs cannot tell whether ParT forms $m$ itself or only carries $m_A$ together with the pT-weighted offset of the particles from the axis. Answer to the user question: the true mass is readable in later layers (more weakly in the class token), but it is mostly not used ("Alternative 1"); the pair path adds nothing beyond untrained ParT.

## Definitions
- $m$: invariant mass of the summed particle four-vectors [GeV].
- $M_0=\sum_i p_{T,i}$, $\vec M_1=\sum_i p_{T,i}(\Delta\eta_i,\Delta\phi_i)$, $M_2=\sum_i p_{T,i}\Delta R_i^2$, all about the stored JetClass jet axis.
- $m_A=\sqrt{M_0M_2}$ (axis-moment mass). For small angles $m^2\approx M_0M_2-|\vec M_1|^2$, so at fixed $m_A$ the true mass differs from $m_A$ only through the centroid offset $\vec M_1$. $\Delta=\log m-\log m_A$.
- Held-out $R^2$: $1-\mathrm{SSE}/\mathrm{SST}$ on evaluation jets that were not used for fitting.
- Partial $R^2$ of $\log m$ given $\log m_A$: $1-\mathrm{SSE}(\text{read-out on site features}+\text{cubic in }\log m_A)/\mathrm{SSE}(\text{cubic in }\log m_A\text{ alone})$ on evaluation rows. It is the share of the true-mass variation left after $m_A$ that the site explains. The reverse ($\log m_A$ given $\log m$) is defined the same way.
- $R_\text{turn}$: slope of a $\log m$ read-out's change against the true $\Delta\log m$ under a prong turn ($m_A$ fixed), divided by the same slope under the matched dilation ($m$ and $m_A$ change together). 1 means the read-out follows a true-mass change at fixed $m_A$ as fully as a real mass change.
- $\rho$: change of the W−Z logit margin $(\ell_W-\ell_Z)$ under an intervention, divided by the change expected if the network used $m$, i.e. $s_\text{dil}\cdot\Delta\log m$. Here $s_\text{dil}$ is the margin slope per $\Delta\log m$ under the matched dilation on W/Z evaluation jets (ParT_full: −11.37). $\rho_\text{input}$ is the same ratio for the turned jet itself.
- DAS (distributed alignment search): a learned orthonormal 4-d subspace whose jet-mean (or class-token) component is copied from a donor run into a base run. It is trained so that the base run reproduces the donor run's 10 logits. IIA (interchange-intervention accuracy): the share of pairs where the swapped run gives the donor run's call. Informative pairs are those where the donor and base calls differ.
- C2ST (classifier two-sample test): out-of-fold AUC of a gradient-boosted classifier that tries to tell edited jets from natural ones. 0.5 means they cannot be told apart.
- Holm: step-down correction over the 13 sites. McNemar: exact test on paired call outcomes.

## Procedure
**Models.** ParT_full and ParT_kin (released JetClass weights, weaver-core 0.4.17). Controls: untrained ParT_full, inits 0 and 1 (every tensor redrawn from a Gaussian with the trained tensor's mean and s.d.), and a weight-permuted ParT_full (output rows of every weight tensor permuted, torch seed 6363). The script uses its own forward pass with site capture. It matches weaver's forward to max |Δlogit| ≤ 2.1e-5 once TF32 is switched off (debug 1).

**Sample.** test_20M WToQQ_112, ZToQQ_112, ZJetsToNuNu_112, TTBar_112: 8,000 random rows per file (`default_rng(6363)`, never first-k). The planned index _125 does not exist (test_20M holds _100–_119, all used before; _112 is the least used in this loop). Each class was split at random into 4,000 fit and 4,000 evaluation jets. 31,987 of 32,000 jets were kept (soft subjet found and dilation matched): 15,994 fit, 15,993 evaluation.

**Unedited accuracy.** On the P1 sample (`part_core.sample(1000, 3000)`, 10 classes), ParT_full gives 85.93 % and ParT_kin 74.83 % (expected 85.9 / 74.5). On this 4-class sample the 10-class accuracy is 81.1 % (full) and 72.2 % (kin). Untrained and permuted networks: ≤ 0.04 %.

**Edits** (all through `part_core.build`, $\eta=\eta_J+\mathrm{sign}(\eta_J)\Delta\eta$; pT and particle masses kept; px, py, pz, E, jet pT and jet energy rebuilt). Changes are evaluation medians, with 5–95 % ranges.
| version | what is changed | held fixed | $\Delta\log m$ | $\Delta\log m_A$ |
|---|---|---|---|---|
| N | nothing (rebuilt) | – | 0 | 0 |
| A (rigid shift) | every particle's $(\Delta\eta,\Delta\phi)$ + $d(\cos a,\sin a)$, $d\in\{0.02,0.04,0.06\}$ rad per jet, $a$ uniform | pair distances, so $m$ | 0 (|·| ≤ 6e-5; max 0.004) | +0.026 (−0.0002 to +0.185) |
| T1, T2 (prong turn) | 2 exclusive-kT subjets (kT distance, E-scheme, pairwise merging); every particle of the softer subjet rotated about the stored axis by $\theta$, $|\theta|$ uniform 10–40°, random sign, drawn separately for T1 and T2 (median 13 particles moved) | each particle's pT and distance from the axis, so $M_0$, $M_2$, $m_A$ (to 1e-7) | −0.0087 (−0.040 to +0.004; max −0.16 to −0.21) | 0 |
| D (matched dilation, calibration only) | all angles × k, k solved per jet so that $m(D)=m(T1)$ | direction pattern | = T1 | ≈ T1 (−0.0088) |

The turn changes $m$ less than the plan expected ("up to about 15 %"). Since $|\vec M_1|^2/m^2\approx 2z(1-z)(1-\cos\theta)$, 40° changes $m$ by about 5 % for a soft-prong fraction $z=0.3$.

**Pool (N, A, T1, T2).** corr($\log m$, $\log m_A$) is 0.9989 on natural jets and 0.9937 on the pool. The planned target of < 0.6 is not reachable with edits of this size, because the spread of masses between jets dominates. The parts the read-outs must separate are uncorrelated: within-jet deviations correlate 0.13, and corr($\Delta$, $\log m_A$) on the pool is −0.03. s.d. of $\Delta$: N 0.025, A 0.089, T 0.029.

**C2ST** (4,000 natural vs 4,000 edited, other half of the fit jets). AUC [95 % CI]:
| family | axis-free jet features | axis-relative ($|\vec M_1|^2/M_0M_2$, $|\vec M_1|/M_0$, girth, $M_2/M_0$) | 8 hardest particles (pT share, $r$) | all |
|---|---|---|---|---|
| A | 0.507 | 0.981 | 0.570 | 0.980 [0.978, 0.982] |
| T1 | 0.500 | 0.946 | 0.504 | 0.945 [0.941, 0.950] |
| D | 0.507 | 0.507 | 0.500 | 0.498 |
| N vs N (null) | 0.505 | 0.506 | 0.504 | 0.499 |

A and T are flagged (AUC > 0.7), only through the axis-relative features. This is built into the question: changing $m$ at fixed $m_A$ requires a centroid offset $\vec M_1$, which natural jets about their stored axis do not have. Read-outs are therefore also reported on N, A and T rows separately.

**Sites.** inp (jet mean of the scaled inputs); mom (baseline: $\log M_0$, $\vec M_1/M_0$, $\log M_2/M_0$, $|\vec M_1|^2/M_0M_2$); emb, b1–b8 (jet mean over real particles of the residual stream); cls1, cls2 (class token); pairF (ln kT, ln z, ln ΔR, ln m², plain and pT_i·pT_j-weighted pooling over i ≠ j); pairH (64-channel hidden layer of the pair embedding, both poolings); pairU (pair bias per head, both poolings); pairSum (oracle $\tfrac12\log\sum_{i<j}m_{ij}^2$).

**Read-outs.** Ridge (α by 3-fold grouped CV) and MLP (2 × 64 ReLU, Adam, early stopping). Fitted on fit pool rows (63,976) and scored on evaluation pool rows (63,972). 95 % CIs come from 1,000 Poisson bootstraps over evaluation jets. Partial $R^2$ fits the residual after the cubic in the other mass (debug 2: fitting $\log m$ directly gave unstable MLP partials in the smoke run).

**Statistics.** H0: partial $R^2$ at a site ≤ untrained. Paired bootstrap against each untrained init; the larger p is kept; Holm over 13 sites. The p floor is 1/1001, so the Holm floor is 0.013.

**Interchange.** Pairs (base T1 ← donor T2) and (T2 ← T1) of the evaluation jets: 31,986 pairs, 16,000 of them W/Z. Site b5 has the largest ridge partial $R^2$ among residual sites; cls1 was also tested. Two interventions were used: (1) the 1-d unit gradient of the ridge read-out of $\log m-\text{cubic}(\log m_A)$, swapped at every particle; (2) DAS 4-d (200 Adam steps, 128 fit pairs per step, MSE to the donor run's logits). Nulls: 200 random 1-d directions and 200 random 4-d subspaces at b5 on 2,000 W/Z pairs (both directions, 4,000 pairs). The same probe swap and DAS fit were also run in untrained init 0.

## Numbers
**Partial $R^2$ of $\log m$ given $\log m_A$, MLP** (all evaluation rows; T = prong-turn rows only):
| site | ParT_full [95 % CI] | ParT_full, T rows | untrained 0 / 1 | permuted | ParT_kin | Holm p (full vs untrained) |
|---|---|---|---|---|---|---|
| inp | 0.118 | −0.18 | (same inputs) | – | 0.089 | – |
| mom (moments) | 0.823 [0.797, 0.843] | 0.47 | – | – | – | – |
| emb | 0.592 [0.562, 0.618] | 0.18 | 0.389 / 0.473 | 0.422 | 0.692 | 0.013 |
| b1 | 0.760 [0.742, 0.778] | 0.50 | 0.300 / 0.254 | 0.336 | 0.826 | 0.013 |
| b2 | 0.765 [0.748, 0.782] | 0.54 | 0.243 / 0.247 | 0.260 | 0.850 | 0.013 |
| b3 | 0.759 [0.740, 0.777] | 0.51 | 0.141 / 0.148 | 0.179 | 0.839 | 0.013 |
| b5 | 0.750 [0.732, 0.769] | 0.47 | 0.107 / 0.143 | 0.117 | 0.846 | 0.013 |
| b8 | 0.681 [0.656, 0.705] | 0.37 | 0.095 / 0.061 | 0.124 | 0.768 | 0.013 |
| cls1 | 0.526 [0.493, 0.561] | 0.12 | 0.050 / 0.044 | 0.074 | 0.669 | 0.013 |
| cls2 | 0.491 [0.454, 0.526] | 0.08 | 0.039 / 0.038 | 0.057 | 0.665 | 0.013 |
| pairF (raw) | 0.258 [0.214, 0.302] | −0.13 | (same inputs) | – | – | – |
| pairH | 0.544 [0.510, 0.581] | 0.20 | 0.420 / 0.551 | 0.530 | 0.504 | 1.0 (vs pairF +0.29 [0.25, 0.32]) |
| pairU | 0.342 [0.298, 0.384] | −0.24 | 0.264 / 0.395 | 0.397 | 0.322 | 1.0 (vs pairF +0.08 [0.05, 0.12]) |
| pairSum (oracle) | 0.995 | 0.98 | – | – | – | – |

Blocks b4, b6 and b7 lie between their neighbours (0.757 / 0.728 / 0.701). On turn rows the untrained values are below 0 at every residual site (−0.02 to −0.73).

Ridge (linear) partial $R^2$, ParT_full: emb 0.164, b1–b8 0.23–0.33 (b5 0.331 [0.315, 0.344]), cls1 0.106, cls2 0.100. Untrained 0.02–0.07 (Holm 0.013 at all sites, including pairH 0.202 vs 0.163/0.190 and pairU 0.089 vs 0.041/0.070). On turn rows the ridge values are −0.13 to +0.03: the true-mass information on turned jets is not linearly readable.

Reverse partial ($\log m_A$ given $\log m$, MLP), ParT_full: b1 0.844, b5 0.777, cls1 0.553, cls2 0.579; untrained 0.03–0.59. Both masses are readable separately in the particle blocks.

**Calibrated read-out response to the turn ($R_\text{turn}$, $\log m$ read-out fitted on the pool).**
- ParT_full, ridge: emb 0.04, b1 0.30, b3 0.29, b5 0.28 [0.25, 0.30], b8 0.14, cls1 0.01 [−0.03, 0.06], cls2 0.04; pairH 0.75, pairU 0.74.
- ParT_full, MLP: b1 0.39, b5 0.28, b8 0.23, cls1 0.06, cls2 0.11; pairH 0.82, pairU 0.78.
- Untrained pair path: 0.60–0.99. Untrained residual sites: −0.4 to 1.5 with wide CIs (the read-outs fit poorly, $R^2$ 0.56–0.85).
- References: moments 1.05; pairSum 1.00.
- ParT_kin, ridge: b1–b8 1.09–1.42, cls1/cls2 0.33/0.36.
- Transfer read-out (fitted on N + A + D rows only), ParT_full: b5 0.29, cls1 0.01.

**Network output.**
- ParT_full: true W called Z 10.1 % (N) → 19.7 % (A, all doses); 9.5 → 29.1 % at 0.06 rad on the same jets (C3 reproduced). T1 10.0 %, D 8.4 %. Calls changed: A 7.6 %, T 2.9 %, D 2.3 %.
- ParT_kin: W → Z 15.8 → 34.0 % at 0.06 rad.
- $\rho_\text{input}$ (turned jet vs the other turn, W/Z jets): ParT_full 0.241 [0.192, 0.291], ParT_kin 0.460 [0.403, 0.517]. Untrained and permuted networks have $s_\text{dil}$ ≈ 0 (0.27, −0.16, 0.55), so their ratio is not defined.

**Interchange** (ParT_full; 31,986 pairs, 1,142 informative pairs where the turn changes the call; the input turn changes 3.6 % of calls):
| swap | ρ [95 % CI] | calls changed | IIA, informative [CI] | change of the cls2 true-m read-out / donor−base | McNemar (correctness, base vs swapped) |
|---|---|---|---|---|---|
| input turn (reference) | 0.241 [0.192, 0.291] | 3.6 % | 1 | 1 | – |
| probe 1-d at b5 | 0.0008 [0.0004, 0.0013] | 0.03 % | 0.004 [0.001, 0.007] | 0.002 | 2 vs 4, p 0.69 |
| DAS 4-d at b5 | 0.067 [0.053, 0.082] | 0.97 % | 0.206 [0.184, 0.228] | 0.084 | 126 vs 126, p 1.0 |
| probe 1-d at cls1 | 0.000 | 0 | 0 | 0.048 | p 1.0 |
| DAS 4-d at cls1 | 0.116 [0.096, 0.135] | 1.86 % | 0.437 [0.409, 0.464] | 0.037 | 230 vs 247, p 0.46 |

Random nulls at b5 (4,000 W/Z pairs; values are 2.5 / 50 / 97.5 % quantiles):
- 200 random 1-d directions: ρ −0.005 / 0.000 / 0.009 (probe 0.0009, rank p 0.60); informative IIA 0 / 0.005 / 0.025 (probe 0, p 1.0).
- 200 random 4-d subspaces: ρ −0.008 / 0.002 / 0.016 (DAS 0.061, p 0.005); informative IIA 0.005 / 0.025 / 0.050 (DAS 0.238, p 0.005).
- Untrained init 0: the turn changes 0.3 % of calls; DAS informative IIA 0.26 (b5) / 0.41 (cls1) on 100 informative pairs.

## Reading
Each hypothesis's predictions, compared with what was measured:
- **Thesis (m never formed: partial $R^2$ at the untrained level in every block; swap moves neither logit nor call).** Excluded for the read-out part. Trained exceeds untrained by 0.20 (emb) to 0.64 (b5) in MLP partial $R^2$ at every residual site (Holm p 0.013, the test's floor), including turned jets alone. The interchange part holds for the 1-d probe direction ($|\rho|$ < 0.002, inside the random null).
- **Alternative 1 (m computed, not used: a site with partial $R^2$ > 0.3 above untrained; swap changes the read-out, not the call).** The read-out condition holds at b1–b8 and cls1–cls2. The 1-d swap changes neither the call nor the downstream cls2 read-out (0.002 of the donor−base difference): the linearly fitted direction is not the coordinate that later layers carry.
- **Alternative 2 (m computed and used: swap moves the margin > 0.3 of the m-based prediction, McNemar p < 0.01).** Not supported. No swap reaches 0.3. The input turn itself gives 0.24 [0.19, 0.29], DAS 4-d 0.07–0.12, and correctness McNemar p ≥ 0.46.
- **Pair path.** It decodes true-mass information (partial $R^2$ 0.34–0.54, $R_\text{turn}$ ≈ 0.75) at the level of untrained and permuted networks. The gain over the raw pooled pair features (+0.08 to +0.29) is also present without training. It comes from the pair-embedding architecture applied to ln m²_ij, not from training.
- **Depth profile (C4).** In the particle blocks the true-mass-beyond-$m_A$ information peaks at b1–b5 (0.75–0.77) and falls to 0.68 at b8 and 0.49–0.53 in the class tokens (0.08–0.12 on turned jets). It does not rise in later layers.

Scope:
- Read-outs measure decodability. The 5-number moment baseline reaches 0.82, so "encodes $m_A$ and the centroid offset $\vec M_1$" and "computes $m$" predict the same read-outs. Only the output tests (ρ) separate use from presence.
- The turned jets are off-manifold for axis-relative features (C2ST 0.95).
- The turn's mass changes are small (median 0.9 %), which limits ρ's precision for individual jets but not the pooled ratio.
- One sample and one seed; no verifier run yet.

Comparison with E41 (JEDI-linear): there no stage after the first layer tracked the true-mass change on turns. ParT_full and ParT_kin carry it (on turn rows MLP partial 0.37–0.54 and 0.47–0.65), and ParT_kin's linear read-out follows it (ridge $R_\text{turn}$ 1.1–1.4).

## Deviations and issues
- File _125 does not exist; _112 was used.
- Pool correlation target < 0.6 not reachable (0.994); the within-jet correlation (0.13) is reported instead.
- Turn mass change is smaller than planned: median 0.9 %, 95th percentile about 4 %, not "up to about 15 %".
- Debug 1: TF32 made the own forward differ from weaver by 6e-3; it was switched off.
- Debug 2: the MLP partial $R^2$ now fits the residual after the cubic in the other mass. A leftover empty list in the interchange bookkeeping was also removed.
- The best interchange site was chosen on evaluation ridge partial $R^2$ (a mild selection on the evaluation split). cls1 was tested as a fixed second site.
- The pair-path sites were not intervened on: a pooled pair-bias read-out direction has no unique inverse in the P×P bias matrices.
