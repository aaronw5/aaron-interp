# H80 — Is the complete N-subjettiness basis enough? Edits in the null space of the Datta–Larkoski M-body basis, four datasets

Loop D, round D17, exploratory (new claim D31; context D5, D8, D13, D30). Script `interp/h80_nsub_basis_nullspace.py` (diagnostic `interp/h80_diag.py`). Jobs: d-h80-run2-top, d-h80-run3-qg, d-h80-run3-jc, d-h80-run3-hls (GPU), d-h80-jedi2 (CPU, JEDI scoring), d-h80-stats, d-h80-fig (CPU). Smoke and debug jobs: d-h80-smoke, d-h80-smoke2, d-h80-jsmoke, d-h80-ssmoke, d-h80-diag, d-h80-pstats, d-h80-pstats2. The first full run, d-h80-run, was stopped and replaced (see "Debugging").
Results: `results/physics/h80_{top,jc,qg,hls}.json/.npz`, `h80_hls_jedi.npz`, `h80_stats.json`, `h80_example_top.npz`, figure `results/physics/h80_nsub_basis.png`.

## Conclusion

We held every N-subjettiness value of the jet fixed, together with the jet mass, the jet pT, the jet axis and the particle count, and moved every particle by a small random amount. Top-quark, W/Z and hls4ml taggers then changed their scores almost as much as they did under a random move of the same size that was free to change those values. The ratio of the two responses, κ₆, was 0.8–1.0 on Top Landscape and JetClass and about 1.0 for JEDI-linear. Quark–gluon taggers lost about half to two thirds of their response (κ₆ ≈ 0.43 at the largest step). In physical terms, the 15 numbers that the Datta–Larkoski construction uses to describe five-body phase space do not fix what these networks respond to, even though a classifier on those numbers reaches within 0.01–0.04 AUC of the networks. Even the IRC-safe energy-flow network kept a sizeable response (κ₆ 0.65 on top jets, 0.32 on quark–gluon jets). So the leftover response is not only IRC-unsafe information. Part of it can come from IRC-safe structure that B₆ leaves free: τ₆ itself still moves, and a tree classifier that also sees τ₆ keeps κ ≈ 0.2–0.3. Three things are still open. κ depends on the step size for quark–gluon jets (0.43 → 0.74 between ε = 0.01 and 0.00125). The edits with the full basis held succeed on only 15–60 % of jets. A matched random edit is not a zero baseline: weight-permuted networks give κ ≈ 1.0–1.3.

(Terms: B_M = {τ_N^β : N = 1..M−1, β = ½, 1, 2}, the Datta–Larkoski N-subjettiness basis. κ_M = mean |Δs| when B_M is held / mean |Δs| for the matched free edit, where Δs is the change in the network's own-class log-odds. C2ST = classifier two-sample test, the AUC of a classifier that tries to tell edited jets from real jets.)

## Question and hypotheses

If B_M (3M−3 numbers), m, pT, the E-scheme axis and N are all held fixed, does a tagger still change its score?
- **H1** (the tagger is a function of B_M plus m and pT): κ_M falls to the permuted-net level at the M where a GBDT on B_M reaches the network's AUC. The EFN also falls.
- **H2** (the tagger reads information outside any finite IRC-safe basis): κ₆ stays clearly above zero (≥ 0.3) while GBDT(B₆) is already close to the network AUC.
- **H3** (the edit is off-manifold and the response is generic): κ tracks C2ST, and permuted nets show the same κ.

## Procedure

**Basis.** τ_N^β = Σ_i pT_i min_k ΔR_{ik}^β / Σ_i pT_i, with ΔR in (rapidity y, φ). The axes are exclusive-kT subjets: pairwise kT clustering with d_ij = min(pT_i², pT_j²)ΔR_ij² and E-scheme recombination, undone to N subjets, implemented in numba. We compute τ_N for N = 1..6 and β ∈ {½, 1, 2} on every jet; that is B₇, 18 numbers.

**Edit (per jet; jets with N ≥ 13 particles: 3980/4000 Top, 3959/4000 JetClass, 3973/4000 QG, 1995/2000 hls).**
- Variables per particle: log pT_i, η_i and φ_i. Particle masses are fixed.
- The variables are normalised by (0.03, 0.01/√2, 0.01/√2).
- Each jet gets one Gaussian vector v, shared by all arms and all step sizes.
- For each arm, v is projected onto the null space of the Jacobian of the held quantities at the unedited jet. It is then scaled so that the RMS per-particle move is 3 % × ε/0.01 in pT and ε in ΔR.
- A Gauss–Newton minimum-norm correction then returns the held quantities to their unedited values.
- The arms differ in what is held:
  - **NS_M** (M = 3, 4, 6): B_M (6, 9 or 15 values), log m, log pT, y_J and φ_J. N is unchanged; no splits or merges.
  - **FREE**: only m, pT, the axis and N. The basis moves; for example, at ε = 0.0025 on JetClass, mean |Δlog τ_N| = 0.0015, 0.013, 0.026, 0.035, 0.043, 0.047 for N = 1..6.
  - **FLOOR**: the FREE edit at ε = 1e-5. It measures each model's numerical floor.
- Subjet membership comes from the unedited jet and is held during the solve; the axes are the E-scheme directions of the members.
- After the solve the jet is re-clustered from scratch. Where membership changed, the solve is repeated with the new membership, up to three times.
- A jet is accepted if the solver residual is < 1e-7, every held τ, m and pT is within 0.5 % after re-clustering, the axis is within 0.002, and N is unchanged.
- Step sizes: ε ∈ {0.01, 0.005, 0.0025, 0.00125}. The realised RMS at ε = 0.01 is 3.0–3.3 % in pT and 0.0095–0.0100 in ΔR in every arm.
- Particles at the same position (ΔR < 1e-5) move rigidly together (see Debugging).
- The figure shows a real Top Landscape top jet before and after NS₆ at ε = 0.01: m = 180.8 GeV, pT = 567 GeV, τ32 (β = 1) 0.6672 → 0.6672, τ21 0.4082 → 0.4082.

**Response.** s is the own-class log-odds, log p_y − log(1 − p_y), where y is the true class (10-class softmax for JetClass, 5-class for hls). Δs = s(edited) − s(unedited).
- κ_M = mean|Δs(NS_M)| / mean|Δs(FREE)|, computed on jets accepted in both arms.
- The floor-corrected version κ_M^c = (mean|Δs_NS| − f)/(mean|Δs_FREE| − f), where f = mean|Δs(FLOOR)|.
- Calls are taken at a frozen 50 % working point. Binary datasets: the signal log-odds at the median of the unedited signal jets. Multi-class: the own-class log-odds at the median of that class's jets. Calls changed by NS_M only vs by FREE only are compared with an exact McNemar test.
- Bootstrap 95 % CIs over jets (B = 1000; κ as a ratio of jointly resampled means).
- Sign-flip permutation p (10⁴) for two hypotheses:
  - H0a: κ = 1, using per-jet |Δs_NS| − |Δs_FREE|.
  - H0b: κ equals the κ of the architecture-matched weight-permuted net, using per-jet |Δs_NS|/mean|Δs_FREE| minus the same for the permuted net.
- Holm correction over trained models × M × datasets × both tests, at each arm's primary step.
- Primary step = the largest ε at which that arm and FREE both have C2ST ≤ 0.65. This is 0.01 for Top, JetClass and QG, and 0.005 for hls.

**Controls.**
- (a) FREE, described above.
- (b) Weight-permuted nets: Top ParT_kin, PN_kin, PFN, EFN; JetClass ParT_full/kin, PN_full/kin; QG part_kin, pnet_kin, pfn; JEDI. Also the shuffled-label nets pfn_shuf, efn_shuf, pnet_kin_shuf (Top and QG).
- (c) GBDT_B6 (HistGradientBoosting on log τ of B₆, log m, log pT), trained on disjoint jets.
- (d) EFN (Top, QG).
- (e) LR(pT^D), a logistic regression on pT^D = √(Σ pT_i²)/Σ pT_i.
- Also GBDT_B7 (adds τ₆) and MLP_B_M.

**C2ST.** A GBDT is trained on accepted edited jets of one half against unedited jets of the other half; both halves are drawn from accepted jets. Features: N, m, pT, 18 log τ, pT^D, two minimum pair distances, and (log pT_i/pT_J, Δy_i, Δφ_i) of the 16 hardest particles. The same test is also run on each feature block separately.

**Samples and references.**
- Top Landscape test: 2k top + 2k QCD (rng 8001); references 5k + 5k (rng 8011).
- JetClass first test file per class, random rows: 1k each of TTBar (Tbqq), WToQQ, ZToQQ, ZJetsToNuNu (rng 8002); references 2k per class (rng 8012).
- KMT Pythia QG test: 2k q + 2k g (rng 8003); references 5k + 5k (rng 8013).
- hls4ml: jedi_core.sample(400, 8004); references 3k per class (rng 8014).

## Unedited models on these samples (accuracy; signal-vs-QCD AUC)

| dataset | model | acc (known) | AUC |
|---|---|---|---|
| Top | ParT_kin / PN_kin / PN_kin_s1 | 0.936 (0.937–0.940) / 0.935 (0.940) / 0.934 | 0.985 / 0.984 / 0.984 |
| Top | LorentzNet / PFN / PFN_s1 / EFN | 0.937 (0.943) / 0.922 (0.929) / 0.922 / 0.915 (0.923) | 0.985 / 0.978 / 0.978 / 0.974 |
| Top | permuted ParT / PN / PFN / EFN; shuffled PN, PFN, EFN | 0.49–0.50; shuffled nets output a constant | 0.52 / 0.72 / 0.69 / 0.68; 0.50 |
| JetClass (4 classes, 10-class argmax) | ParT_full / ParT_kin / ParT_kin s0,s1,s2 / PN_full / PN_kin | 0.805 / 0.713 / 0.696, 0.696, 0.709 / 0.785 / 0.702 | 0.983 / 0.974 / 0.970, 0.970, 0.969 / 0.979 / 0.969 |
| QG | part_kin / part_kin_s1 / pnet_kin / lorentznet / pfn / efn | 0.824 (0.827, AUC 0.907 in H76) / 0.829 / 0.825 / 0.841 (0.854) / 0.810 / 0.802 | 0.900 / 0.902 / 0.902 / 0.916 / 0.890 / 0.882 |
| hls | JEDI-linear n64 | 0.804 (0.807–0.810) | 0.972 |

The known JetClass accuracies (0.858 / 0.742 / 0.838 / 0.704) are 10-class figures over all classes; the numbers here use 4 of the 10 classes. The shuffled-label nets output a constant (Δs = 0 for every edit), so they cannot serve as a null here.

**Saturation curve (GBDT(B_M) AUC, signal vs QCD, M = 2, 3, 4, 5, 6, 7):**
- Top: 0.931, 0.949, 0.967, 0.971, 0.973, 0.973 (nets 0.974–0.985).
- JetClass: 0.907, 0.933, 0.941, 0.945, 0.945, 0.944 (nets 0.969–0.983).
- QG: 0.758, 0.826, 0.856, 0.863, 0.866, 0.867 (nets 0.882–0.916).
- hls: 0.952, 0.961, 0.967, 0.967, 0.968, 0.968 (JEDI 0.972).
- MLP(B_M) is within 0.01 of the GBDT in every case.

## Numbers (primary step; κ₆ [95 % CI]; floor-corrected κ₆^c where the floor matters; matched permuted-net κ₆)

**Solver checks.**
- GBDT_B6 under NS₆: κ₆ = 0.000–0.004 on all datasets and steps. Its calls changed are 0 vs 0.4–7.8 % under FREE (McNemar, e.g. Top 0/68, p 7e-21).
- GBDT_B7 κ₆ = 0.27 (Top), 0.28 (JetClass), 0.20 (QG), 0.33 (hls), because τ₆ moves under NS₆ (mean |Δlog τ₆| 0.011–0.026 at ε = 0.0025).
- LR(pT^D) κ₆ = 1.3–1.9 (H0a p ≤ 7e-4). The IRC-unsafe reference moves more when the basis is held.

**C2ST at the primary step** (none = unedited vs unedited, 0.48–0.50):
- Top: NS₃/₄/₆/FREE 0.49/0.50/0.48/0.52.
- JetClass: 0.48/0.53/0.53/0.58.
- QG: 0.55/0.58/0.57/0.61.
- hls (ε 0.005): 0.53/0.56/0.55/0.55.
- At ε = 0.01 on hls, FREE has C2ST 0.69 (jet-level block 0.70).

**Acceptance (NS₆, ε 0.01 → 0.00125):**
- Top: 56 → 88 %.
- JetClass: 43 → 80 %.
- QG: 15 → 63 %.
- hls: 18 → 60 % (30 % at the primary 0.005).
- NS₃ acceptance is 79–96 %.

**Top Landscape (ε 0.01, n = 2228):**

| model | κ₃ / κ₄ / κ₆ [CI] / κ₆^c | permuted κ₆ | |Δs| FREE | calls changed NS₆ / FREE | floor share |
|---|---|---|---|---|---|
| ParT_kin | 1.01 / 0.97 / 0.94 [0.89, 1.00] / 0.93 | 1.10 | 0.226 | 2.2 / 2.5 % (24/30, p 0.5) | 0.21 |
| PN_kin | 1.08 / 1.08 / 0.98 [0.93, 1.03] / 0.97 | 1.14 | 0.212 | 1.8 / 1.8 % | 0.23 |
| PN_kin_s1 | 1.04 / 1.08 / 1.02 [0.96, 1.08] / 1.02 | 1.14 | 0.235 | 1.8 / 1.5 % | 0.23 |
| LorentzNet | 1.01 / 0.96 / 0.89 [0.85, 0.94] / 0.77 [0.68, 0.87] | 1.10 | 0.402 | 3.2 / 2.7 % | 0.54 |
| PFN / PFN_s1 | 0.88 / 0.91 / 0.79 [0.74, 0.85]; 0.79 [0.74, 0.85] | 1.14 | 0.062 | 0.4 / 0.7 % | 0.00 |
| EFN | 0.88 / 0.79 / 0.65 [0.60, 0.71] | 0.78 | 0.083 | 0.5 / 1.2 % (6/22, p 0.004) | 0.00 |

**JetClass (ε 0.01, n = 1683):**

| model | κ₃ / κ₄ / κ₆ [CI] / κ₆^c | permuted κ₆ | calls changed NS₆ / FREE |
|---|---|---|---|
| ParT_full | 0.88 / 0.86 / 0.89 [0.80, 0.98] | 1.01 | 5.1 / 6.2 % |
| ParT_kin | 0.88 / 0.84 / 0.85 [0.79, 0.92] / 0.81 | 1.04 | 6.4 / 7.9 % (51/77, p 0.03) |
| ParT_kin s0 / s1 / s2 | κ₆ 0.86 / 0.86 / 0.84; κ₆^c 0.82 / 0.82 / 0.79 | 1.04 | s1 6.1 / 8.9 % (47/93, p 1e-4) |
| PN_full / PN_kin | 0.93 [0.86, 1.01] / 0.81 [0.75, 0.87] | 1.07 / 1.20 | 6.4 / 6.6 %; 4.4 / 5.1 % |

At ε ≤ 0.0025 every JetClass ParT/PN κ₆ is 0.95–1.25. Floor share for ParT_kin is 0.24 at ε 0.01 and 0.58 at 0.0025.

**Quark–gluon (ε 0.01, n = 604 accepted in both arms; ε 0.00125, n = 2505):**

| model | κ₃ / κ₄ / κ₆ [CI] at ε 0.01 | κ₆ at ε 0.005 / 0.0025 / 0.00125 | permuted κ₆ | calls changed NS₆ / FREE (ε 0.01) |
|---|---|---|---|---|
| part_kin | 0.69 / 0.65 / 0.43 [0.38, 0.48] | 0.57 / 0.69 / 0.74 | 1.05 | 1.0 / 4.1 % (4/23, p 3e-4) |
| part_kin_s1 | 0.68 / 0.66 / 0.45 [0.41, 0.50] | 0.55 / 0.63 / 0.72 | 1.05 | 1.2 / 4.3 % (p 2e-5) |
| pnet_kin | 0.71 / 0.68 / 0.45 [0.40, 0.50] | 0.52 / 0.60 / 0.68 | 1.05 | 1.0 / 4.1 % (p 7e-5) |
| lorentznet | 0.72 / 0.73 / 0.43 [0.38, 0.49] | 0.49 / 0.54 / 0.58 | 1.05 | 0.8 / 3.8 % |
| pfn | 0.75 / 0.78 / 0.49 [0.44, 0.55] | 0.50 / 0.61 / 0.60 | 1.14 | 0.8 / 2.8 % |
| efn | 0.59 / 0.54 / 0.32 [0.29, 0.36] | 0.38 / 0.46 / 0.49 | 1.14 | 1.3 / 5.6 % (p 2e-5) |

At ε ≤ 0.0025, κ₃ for QG is 1.0–1.3 (above 1), while κ₆ stays at 0.46–0.74.

**hls4ml (ε 0.005, n = 591):** JEDI κ₃/κ₄/κ₆ = 1.02 / 1.06 / 1.05 [0.97, 1.12]. Permuted JEDI 1.13 [0.85, 1.58]. Calls changed 3.0 / 3.6 % (5/8, p 0.58). Floor share 0.03.

**Numerical floors.** mean|Δs| for the 1e-5 edit:
- Top: LorentzNet 0.219, ParT_kin 0.044, PN_kin 0.044 (PFN, EFN ≤ 0.0003).
- JetClass: ParT_kin and seeds 0.088–0.103, ParT_full 0.005, PN ≤ 0.001.
- QG: part_kin 0.016–0.020.
- hls: JEDI 0.007.

**Tests.** Holm over 126 tests (trained nets × M × datasets × {H0a, H0b} at each arm's primary step): 96 with p_Holm < 0.05. The permutation resolution is 1e-4, so the smallest attainable p_Holm is 0.0126.
- H0b (κ_net = κ_permuted) at M = 6: rejected (p_Holm 0.013) for every trained QG and Top net and for JetClass ParT_kin, its three seeds and PN_kin; PN_full p_Holm 0.040. Not rejected for JetClass ParT_full (raw p 0.008, p_Holm 0.21) or JEDI (raw p 0.75).
- H0a (κ = 1) at M = 6: rejected (p_Holm 0.013) for every QG net, JetClass ParT_kin/seeds/PN_kin and Top LorentzNet/PFN/PFN_s1/EFN. Not rejected for Top ParT_kin (raw p 0.04, p_Holm 0.85), PN_kin, PN_kin_s1, JetClass ParT_full, PN_full or JEDI.

## Reading (scoped to these edits)

- **Top, JetClass, hls4ml.** Holding B₆, m, pT, the axis and N lowers the trained networks' response by 0–20 % relative to FREE: κ₆ 0.81–1.02, with JEDI at 1.05. The permuted-net level is 1.01–1.20. The GBDT on B_M saturates by M = 4–5 at 0.002–0.04 AUC below the networks. H1 predicted κ_M near the permuted level at the saturation M. What we see is the reverse: the trained nets sit slightly *below* the permuted level, and their κ does not fall toward zero between M = 4 and M = 6. The H1 prediction "κ falls toward zero" is excluded for these datasets.
- **Quark–gluon.** κ₆ = 0.43–0.49 (EFN 0.32) at ε 0.01, against 1.05–1.14 for the permuted nets. Holding B₆ removes about half of the response. That is more than on the other datasets, but less than H1 requires, since 0.29–0.50 of it remains.
- **The size pattern across datasets** (the cross-dataset fact asked for): the QG response depends most on the basis, then JetClass and Top (κ₆ 0.8–1.0), and JEDI-linear least (≈ 1.0).
- **EFN.** The EFN is an IRC-safe network, yet it keeps κ₆ = 0.65 (Top) and 0.32 (QG). So the leftover response is not purely IRC-unsafe. The NS₆ edit still moves IRC-safe structure outside B₆: τ₆ moves (GBDT_B7 gives κ₆ 0.2–0.3), and other energy-flow polynomials are not constrained. H2 (κ₆ ≥ 0.3 with GBDT near the AUC) holds numerically for every trained net. H2's specific reading, that the remainder is IRC-unsafe, is not isolated by this design.
- **H3 (off-manifold or generic response).** At the primary steps the C2ST values are 0.48–0.61. Across arms × ε, the correlation of trained-net κ with C2ST is +0.51 (Top), −0.58 (JetClass), −0.61 (QG) and −0.36 (hls), so κ does not rise with C2ST. Permuted nets give κ ≈ 1.0–1.3 rather than the trained values on QG, on JetClass (except ParT_full) and on the Top nets, so H3 is excluded for these. For JEDI and JetClass ParT_full the trained and permuted κ are not separated (JEDI permuted CI 0.85–1.58).
- **Step-size dependence.** For QG, κ₆ rises from 0.43 (ε 0.01) to 0.74 (ε 0.00125), and κ₃ exceeds 1 at small ε. The accepted subset also changes with ε (15 → 63 %), so step size and jet selection are confounded. For ParT on Top and JetClass at small ε, the numerical floor is 50–70 % of the FREE response, and κ there carries a large floor component.

## Debugging

1. **Coincident constituents.** In the first smoke run, the Top C2ST was 0.94 at every ε, and ParT/PN responses did not shrink with ε. Diagnostic d-h80-diag found that 80 % of Top Landscape test jets contain a pair of constituents at ΔR < 1e-4 (median minimum pair distance 1.8e-8; 0.16 % of pairs). The edit separated these pairs. QG jets have such pairs in 0.3 % of jets. Fix: particles within ΔR < 1e-5 share their angular move and keep their original sub-1e-5 offset. After the fix, Top C2ST is 0.48–0.52 at every step, and the pair-distance block is 0.51–0.53.
2. **Low NS₆ acceptance from subjet-membership changes.** Fix: re-cluster and re-solve, up to 3 times. NS₆ acceptance on Top went from 34 % to 56 % at ε 0.01.
3. **C2ST on non-accepted jets.** The first version computed C2ST over all edited jets, including failed solves. It is now restricted to accepted jets in both halves.
4. **A JetClass smoke subset was class-ordered** (first 2,000 reference rows). This was replaced by a random subset; the full run never used first-k rows.
5. Three parallel GPU jobs (d-h80-run2-qg/jc/hls) failed before writing a log and were relaunched as run3-*.

## Caveats

- **Basis definition.** τ_N uses exclusive-kT E-scheme axes, not the WTA or one-pass axes of some Datta–Larkoski implementations. Held values are exact to 0.5 % after re-clustering.
- **Acceptance.** NS₆ edits are feasible on only 15–56 % of jets at ε 0.01. κ is computed on the jets accepted in both arms, which may not be representative; for QG they are likely higher-multiplicity jets.
- **Size matching.** FREE is matched to NS_M in total normalised size and per-particle RMS, not particle by particle. Permuted-net κ > 1 shows that the null-space direction is, on average, slightly more effective per unit norm for random functions.
- **Shuffled-label nets** output a constant, so they give no information.
- **LorentzNet (Top)** has a large numerical floor (0.22 log-odds for a 1e-5 move).
- **No verifier run yet.**

## Proposed follow-ups

- Hold a larger IRC-safe set (B₈ plus EFPs up to degree 5), so that the EFN's κ goes to zero and the IRC-unsafe remainder is isolated.
- Make the QG ε-dependence and selection separable by fixing the accepted jet set across ε.
- Measure κ against an untrained-architecture band rather than one permuted copy.
