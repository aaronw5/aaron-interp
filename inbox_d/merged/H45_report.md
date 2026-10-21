# H45 — Two-prong momentum sharing: decay-axis rotation in the prong-pair rest frame (claim D18, exploratory)

Status: done (one run; not yet verified). Script `interp/h45_prong_sharing.py`. Results `results/physics/h45_prong_sharing.json`,
figures `results/physics/h45_edit.png` (the edit on one real W jet), `results/physics/h45_response.png` (response vs Δc).
Cluster jobs: d-h45-smoke2 (CPU smoke), d-h45-edit (edits, observables, references, small nets, JEDI), d-h45-nets (GPU, JetClass
networks), d-h45-stats3 (statistics), d-h45-fig (figures). Per-jet arrays stay on the volume:
/j-jepa-vol/aaron-interp/results/physics/h45_{jc,hls}_{inputs,obs,logits*}.npz (inputs ~0.3 GB; delete after H45v).

## Conclusion

When the momentum of a W, Z or QCD jet is redistributed between its two prongs while the jet's four-momentum, its mass and every
pairwise invariant mass stay exactly fixed, all eight taggers (JetClass ParT and ParticleNet, hls4ml JEDI-linear and three small
nets) move their W/Z score in the direction a mass-drop tagger would: more balanced prongs look more like W/Z, more asymmetric
prongs less (for example up to −2.6 log-odds on W jets at Δ|cos θ*| = +0.3), while rotating the prongs by the same angle without
changing the sharing does nothing. A reference classifier that is never given the momentum fraction — only jet mass, pT, τ21, D2,
multiplicity and prong separation ΔR — predicts responses of the same sign and of comparable size (median network/reference ratio
0.9–1.4), because at fixed jet mass the sharing z, the separation ΔR and the prong masses are tied together; this experiment
therefore shows that the taggers respond to the momentum-sharing configuration, but does not separate "uses z" from "uses ΔR,
τ21, D2". JEDI-linear's response appears with its input moments ΣpT and m_A = √(M0·M2) unchanged to < 1 %, so its W/Z decision
uses more than the mass window and ΣpT of D1.

Terms: z = pT fraction of the softer of the two prongs; |cos θ*| = decay angle of the prong pair in its rest frame relative to the
jet direction (z ≈ (1 − |cos θ*|)/2); log-odds = log p(W)/p(QCD) etc. from the network's class scores.

## Question and hypotheses

Do two-prong taggers respond to the momentum sharing between the two prongs beyond what its side effects on τ21, D2 and the prong
separation predict? Physics: QCD 1→2 splittings follow P(z), which grows as z → 0 (asymmetric); W/Z → qq̄ decays give more
balanced splits. Mass-drop / soft-drop taggers cut on z (z_g > 0.1) for this reason. At fixed jet mass,
m² ≈ m_a²/z + m_b²/(1−z) + z(1−z) pT² ΔR², so changing z moves ΔR, τ21 and D2 too.

- H0 (side effects only): the network's response to the edit minus the matched control (c-side) lies in the permuted-network band
  and equals MLPx (no z) within its retrain spread.
- H1 (sharing used): balanced prongs (Δc < 0) move QCD and W/Z jets toward W/Z beyond c-side, by about MLPz − MLPx, ordered in Δc.
- W-vs-Z cell: predicted ≈ 0 under H0 and under H1 without polarisation information.

## Procedure

Samples (random rows, never first-k).
- JetClass: test files WToQQ_100, ZToQQ_100, ZJetsToNuNu_100, stored 500 < pT < 1000 GeV, |η| < 2, ≥ 3 particles; 2,000 jets per
  class (rng 4501 / 4502 / 4503). Median jet pT 586 GeV, median 32 particles.
- hls4ml: rows of `jedi_core.sample(per_class=2000, seed=4504)` (10,000 jets; tops only for the 5-class accuracy); analysis on W,
  Z, q, g (2,000 each). Massless constituents, 64 slots.

Prongs: two exclusive-kT subjets (p = 1, E-scheme, rapidity–azimuth distance) of all constituents, so the pair four-momentum is the
jet four-momentum.

Edit D(Δc), Δc ∈ {−0.3, −0.15, +0.15, +0.3}. Boost all constituents to the jet rest frame (β = p_jet/E_jet). n₊ = rest-frame
direction of the forward prong, c₀ = |cos θ*| = n₊·β̂. Target c₁ = clip(c₀ + Δc, 0, 0.95). Rotate every constituent by the angle
α = arccos c₀ − arccos c₁ about k = n₊ × β̂, boost back. Every constituent receives the same Lorentz transformation
Λ = B(−β) R B(β), which leaves the jet four-momentum fixed. Fixed exactly: jet mass, jet pT and direction, every p_i·p_j (hence each
prong mass and all intra-prong structure in invariant terms), multiplicity, particle features (charge, PID, d0, dz). Changed: the
lab-frame energies (z), angles (ΔR, τ21, D2). Self-test on a synthetic jet: jet four-momentum changed by ≤ 3·10⁻¹² GeV, p_i·p_j by
≤ 4·10⁻¹² GeV², |cos θ*| reached the target exactly.

Acceptance: both prong centres within ΔR 0.75 of the jet axis after the edit (fails for 0–9 % of jets at Δc = +0.3).

Controls (same jets):
- cφ: rotation by the same |α| about β̂ in the rest frame (random sign). |cos θ*| and both prong energies are exactly fixed; the
  prongs move around the jet axis. Tests "moving the prongs" without changing the sharing.
- c-side: a per-jet edit at fixed constituent pT (so z fixed): prongs moved rigidly along their joining line to D's separation ΔR,
  constituent offsets inside prong k scaled by s_k (14 ratios s_b/s_a from 0.12 to 8; s_a solved so the jet invariant mass is
  unchanged); the candidate whose reclustered τ21 is closest to D's is kept. Matched set: |Δm| < 1 GeV, |τ21 − τ21_D| ≤ 5 % of
  τ21_D, |ΔR − ΔR_D| ≤ 5 % of ΔR_D. It isolates z at matched ΔR and τ21, but it changes the prong masses and D2 (D does not).
- TN: translation of the unedited jet by D's pT-weighted mean displacement in a random direction.
- Every jet (edited and unedited) is rebuilt about its own E-scheme axis for every network (round-D5 rule).

Models. JetClass: ParT_full, ParT_kin, PN_full, PN_kin; nulls: 6 copies of ParT_full with every weight tensor permuted (seeds 5–10).
hls4ml: JEDI-linear official n64 3-feature, dsets-d64-n64, jedis-d64-n64, tf-2b-d32-n64 (seed 0); nulls: JEDI with redrawn weights
(seeds 0–2), dsets with permuted weights (seeds 7–9).

Scores (log-odds): JetClass WQ = L_W − L_QCD, ZQ = L_Z − L_QCD, WZ = L_W − L_Z; hls4ml WQG = L_W − logsumexp(L_q, L_g),
ZQG likewise, WZ = L_W − L_Z.

References, retrained on train rows (JetClass train_100M *_000, 10,000 per class, rng 4506–4508; hls4ml train archive without the
dev split, 10,000 per class, rng 4509), 5 retrains each, per score pair:
- MLPx = MLP(64, 64) on (m, pT, τ21, D2, N, ΔR) — the edit's side effects, no momentum-sharing variable;
- MLPz = MLPx + z_g (soft-drop momentum fraction: C/A declustering, z_cut 0.1, β = 0, R0 0.8);
- LRzg = logistic regression on (z_g, z_g²).

Metric per cell (sample class × score) × Δc × model: per-jet Δs = s(variant) − s(unedited); D − cφ on jets accepted for D and cφ;
D − c-side on jets also matched by c-side. Mean with 95 % bootstrap CI over jets (2,000 resamples); two-sided sign-flip permutation
p (5,000) and one-sample t-test p; Holm over 4 nets × 14 cells × 4 Δc × 2 contrasts = 448 tests (on the t-test p, since the
permutation p is floored at 1/5,001); per-jet net − MLPx (retrain 0), Holm over 448; null band = min–max of the null models; McNemar
on calls at the threshold that keeps 50 % of unedited signal jets; dose = Spearman of the means against Δc; C2ST = 2-fold
gradient-boosting classifier AUC separating edited jets from natural train jets of the same class matched (nearest neighbour) in
(m, |cos θ*|), on (pT, m, m_SD, τ21, τ32, D2, N, ΔR, z, z_g) (0.5 = indistinguishable).

## Unedited accuracy on these samples

| model | sample accuracy | per-class efficiency / known |
|---|---|---|
| ParT_full | 0.756 (10-class on W/Z/QCD) | W 0.805, Z 0.691, QCD 0.773 (H41 sample, same files: W 0.797, Z 0.697) |
| ParT_kin | 0.668 | W 0.707, Z 0.576 (H41: 0.698, 0.569) |
| PN_full | 0.728 | W 0.774, Z 0.661 (H41: 0.770, 0.669) |
| PN_kin | 0.630 | W 0.664, Z 0.539 (H41: 0.664, 0.539) |
| JEDI-linear | 0.8085 (5-class) | known 0.8092 (260k test) |
| dsets / jedis / tf | 0.8069 / 0.7988 / 0.7620 | known 0.8064 / 0.7968 / 0.7619 |
| permuted ParT ×6 | 0.002–0.278 | null |
| redrawn JEDI ×3, permuted dsets ×3 | 0.18–0.20 | null (nearly constant outputs) |

Stored-axis vs own-axis build changed accuracy by ≤ 0.1 pp for every model. Reference held-out AUC: JetClass MLPx W-vs-QCD 0.947,
MLPz 0.947, LRzg 0.589; hls4ml MLPx W-vs-q/g 0.974, MLPz 0.974, LRzg 0.585.

## Numbers

What the edit changed (mean per jet, D on the c-side-matched rows; JetClass W jets, Δc = −0.3, n = 585): achieved Δ|cos θ*| −0.15
(requested −0.3; jets with c₀ < 0.3 are clipped at 0), Δz +0.056 (c-side +0.008, cφ −0.001), Δz_g +0.050 (c-side +0.011),
Δτ21 −0.031 (c-side −0.031), ΔD2 −0.13 (c-side +0.10), Δm 0.000. c-side matched 25 % / 38 % / 10 % / 4 % of JetClass jets at
Δc = −0.3 / −0.15 / +0.15 / +0.3 (hls4ml 23 / 34 / 9 / 4 %): at fixed m, pT and z, ΔR can only grow by shrinking the prongs to zero
width, so for Δc > 0 a mass-preserving solution exists for about half the jets only.

JEDI-linear's input moments under D (hls4ml, about each jet's own axis): mean Δlog m_A ≤ 0.0014, 90th percentile of |Δlog m_A|
≤ 0.0085; ΣpT (M0) changes by ≤ 4·10⁻⁵ relative.

D − cφ (Δ log-odds, 95 % CI), selected cells; MLPx / MLPz = range over 5 retrains; null = permuted/redrawn band:

| cell | Δc | ParT_full | ParT_kin | PN_full | PN_kin | MLPx | MLPz | null |
|---|---|---|---|---|---|---|---|---|
| JetClass QCD, W-vs-QCD | −0.3 | +0.95 [0.88, 1.01] | +1.49 [1.41, 1.58] | +0.94 [0.87, 1.01] | +1.34 [1.27, 1.42] | +0.99…+1.15 | +0.95…+1.14 | −0.12…+0.09 |
| JetClass QCD, W-vs-QCD | +0.3 | −0.15 [−0.23, −0.08] | −0.19 | −0.20 | −0.31 | −0.39…−0.30 | −0.38…−0.33 | −0.06…+0.06 |
| JetClass W, W-vs-QCD | −0.3 | +0.23 [0.18, 0.28] | +0.56 | +0.44 | +0.60 | +0.44…+0.60 | +0.49…+0.61 | −0.06…+0.04 |
| JetClass W, W-vs-QCD | +0.3 | −2.61 [−2.73, −2.49] | −1.94 | −2.48 | −1.84 | −1.42…−1.20 | −1.39…−1.30 | −0.06…+0.08 |
| JetClass W, W-vs-Z | +0.3 | −1.60 [−1.68, −1.53] | −0.55 | −1.37 | −0.45 | −0.34…−0.21 | −0.32…−0.25 | −0.10…+0.04 |

| cell | Δc | JEDI-linear | dsets | jedis | tf | MLPx | MLPz | null |
|---|---|---|---|---|---|---|---|---|
| hls4ml W, W-vs-q/g | −0.3 | +0.49 [0.44, 0.55] | +0.28 | +0.43 | +0.74 | +0.24…+0.43 | +0.27…+0.53 | −0.15…+0.22 |
| hls4ml W, W-vs-q/g | +0.3 | −1.76 [−1.85, −1.66] | −2.08 | −2.22 | −1.97 | −1.93…−1.33 | −2.01…−1.45 | −0.74…+0.51 |
| hls4ml q, W-vs-q/g | −0.3 | +0.03 [−0.01, 0.07] | +0.52 | +0.42 | +0.52 | +0.01…+0.09 | −0.09…+0.21 | −0.17…+0.17 |
| hls4ml Z, W-vs-Z | +0.3 | +1.56 [1.45, 1.66] | +1.46 | +1.35 | +1.83 | −0.06…+0.47 | −0.13…+0.78 | 0.00…+0.35 |

Over all cells: D − cφ differs from 0 after Holm in 94/96 JetClass and 113/128 hls4ml network cells; cφ alone moves scores by
|Δs| ≤ 0.13. Dose: Spearman(D − cφ, Δc) = −0.8 to −1.0 in all 24 JetClass network × cell series and in 28 of 32 hls4ml series
(the 4 exceptions: dsets on q/g jets, and the Z-jet W-vs-Z cell where all nets give +0.8 to +1.0, i.e. asymmetric Z jets move
toward W). Ratio network / MLPx (cells with |MLPx| ≥ 0.2): median ParT_full 0.88, ParT_kin 1.38, PN_full 0.90, PN_kin 1.25,
JEDI 1.19, dsets 1.13, jedis 1.24, tf 1.19. Net − MLPx after Holm: JetClass 66/96 cells differ (23 above, 43 below MLPx),
hls4ml 105/128 (73 above, 32 below). Calls (W jets, W-vs-QCD, Δc = +0.3): ParT_full passes 50.8 % → 25.1 % under D vs 50.7 % under
cφ (McNemar p 8·10⁻¹¹⁷).

D − c-side (z changed at matched ΔR and τ21), selected cells:

| cell | Δc | ParT_full | ParT_kin | PN_full | PN_kin | MLPx | MLPz | null |
|---|---|---|---|---|---|---|---|---|
| JetClass W, W-vs-QCD (n 585) | −0.3 | +0.44 [0.32, 0.57] | +0.29 | +0.30 | +0.36 | +0.17…+0.24 | +0.22…+0.28 | −0.08…+0.05 |
| JetClass QCD, W-vs-QCD (n 420) | −0.3 | +0.24 [0.11, 0.37] | +0.57 | +0.15 | +0.57 | +0.25…+0.35 | +0.32…+0.43 | −0.07…+0.02 |
| JetClass Z, W-vs-Z (n 500) | −0.3 | −0.44 [−0.57, −0.31] | −0.01 | −0.37 | +0.46 | −0.09…−0.07 | −0.07…−0.04 | −0.01…+0.02 |
| hls4ml W, W-vs-q/g (n 519) | −0.3 | JEDI +0.22 [0.14, 0.32] | dsets +0.17 | jedis +0.24 | tf +0.23 | +0.10…+0.15 | +0.09…+0.26 | −0.06…+0.15 |

D − c-side differs from 0 after Holm in 43/96 JetClass and 51/128 hls4ml cells (most Δc > 0 cells have < 300 matched jets).
c-side alone moves W-jet W-vs-QCD scores by −0.8 to −1.4 (JetClass) and −0.7 to −1.1 (hls4ml), and MLPx by −0.5 / −1.2: the
control has its own large effect through the wider prongs and higher D2. In the W-vs-Z cells D − c-side has the same sign at
Δc < 0 and Δc > 0 (Spearman −0.4 to 0 for the full models), i.e. it follows the control, not Δc.

C2ST AUC (edited vs natural jets at matched m and |cos θ*|; unedited jets give 0.54–0.61): Δc = −0.3: D 0.75–0.81 (JetClass),
0.75–0.90 (hls4ml); cφ 0.55–0.62; c-side 0.61–0.69. Δc = +0.3: D 0.51–0.66 (≈ unedited). Balanced-prong edited jets are
distinguishable from natural jets with the same |cos θ*|; the Δc = −0.3 responses include an off-manifold component.

## Reading (scoped to the measurement)

- The sharing-changing rotation D moves all eight trained networks' W/Z scores, ordered in Δc, while the rotation of the same size
  that keeps the sharing (cφ) and the permuted/redrawn networks stay near 0. This holds on QCD, W and Z jets and on both datasets.
- MLPz − MLPx ≈ 0 in every cell (|difference| ≤ 0.1 in D − cφ): given m, pT, ΔR, τ21, D2 and N, z_g adds no information, so the
  H1 prediction "≈ MLPz − MLPx" is ≈ 0 and cannot be separated from H0 by this design. The networks' D − cφ is of the size MLPx
  predicts from the side effects (ratios 0.9–1.4), larger on W/Z jets at Δc = +0.3 for the JetClass full models (≈ 1.8×), and
  smaller for ParT_full / PN_full on balanced W jets.
- The c-side control, which holds z and matches ΔR and τ21, changes prong masses and D2 and produces network responses of its own
  (up to −1.4); D − c-side therefore mixes z with the prong-width change. Neither H0 (D − c-side = MLPx) nor H1 holds across cells:
  networks lie above MLPx in some cells and below in others.
- W-vs-Z: D − cφ is not ≈ 0. JetClass full models move asymmetric W jets toward Z by −1.4 to −1.6 (MLPx −0.2 to −0.3); in hls4ml,
  asymmetric W jets move toward Z and asymmetric Z jets toward W (+1.4 to +1.8 vs MLPx ≤ +0.5), i.e. the W-vs-Z separation shrinks
  in both directions. A polarisation-specific use is not separated from a general loss of W/Z-likeness.
- JEDI-linear: responds to D on W/Z jets (+0.49 / −1.76) with m_A and ΣpT unchanged to < 1 %; on q/g jets its response stays at the
  MLPx level (q jets, Δc = −0.3: +0.03 vs MLPx +0.01…+0.09), while the three small hls4ml nets respond 5–10× more (+0.42 to +0.52).

Competing hypotheses: "networks ignore the prong configuration at fixed m" — excluded (D − cφ ≠ 0 in 207/224 cells after Holm,
nulls ≈ 0). "Networks use z beyond its correlates" vs "networks use ΔR/τ21/D2 at fixed m" — not separated (MLPz ≈ MLPx; the matched
control has its own effect). "JEDI-linear = mass window + ΣpT only" — excluded for W/Z jets (response with both fixed).

## Issues / caveats

- c-side cannot be built at fixed m, pT and z for most jets with Δc > 0 (physics constraint above); Δc > 0 D − c-side cells rest on
  52–303 jets that are a selected subset.
- c-side changes D2 with the opposite sign to D and inflates prong masses; a control that also holds D2 and prong masses would need
  a second free direction (e.g. internal rotations of each prong); not available at fixed m, pT, z, ΔR.
- Δc = −0.3 edited jets are off-manifold (C2ST 0.75–0.90 vs 0.54–0.61 unedited).
- Null models are near-degenerate (permuted / redrawn hls4ml models output nearly constant classes); their band is narrow by
  construction on hls4ml.
- One sample per dataset, one seed; not yet verified on fresh jets.
- Two debug fixes: JetClass parts need the ParT env (JETDISTILL_TAGGER=part, env-part python); a dict-access bug in stats.
  Holm moved from permutation p to t-test p (permutation floor 1/5,001 × 448 = 0.09).

## Suggested next steps

H45v (skeptic): fresh seeds; replace c-side by a natural-jet reweighting control (reweight unedited jets so (m, pT, ΔR, τ21, D2, prong
masses) match the D jets, compare network score shifts); an MLP that also gets prong masses (MLPxs) as the side-effect reference;
W-vs-Z cell with a polarisation-labelled sample (truth cos θ* of the W/Z decay) to test polarisation use; JEDI-linear: locate the
z-sensitive units (loop A/B handoff).

## Explainer entry (for site/explain.py; to be merged by the ledger agent)

```python
H45_EXPLAINER = [
 ('Question', 'Do jet taggers use how the momentum is shared between the two prongs of a two-prong jet (the quantity mass-drop and soft-drop taggers cut on), beyond the changes of prong separation, $\\tau_{21}$ and $D_2$ that come with it?', None),
 ('Physics background', 'A W or Z boson decaying to two quarks gives two prongs; a QCD jet with two prongs comes from a gluon or quark splitting. Let $z = \\min(p_{T,a}, p_{T,b})/(p_{T,a}+p_{T,b})$ be the softer prong\'s momentum fraction. In the rest frame of the prong pair the two prongs are back to back at angle $\\theta^*$ to the jet direction; for a fast jet $z \\approx (1 - |\\cos\\theta^*|)/2$. QCD splittings follow $P(z) \\propto 1/z$ (asymmetric), boson decays are more balanced. At fixed jet mass $m^2 \\approx m_a^2/z + m_b^2/(1-z) + z(1-z)\\,p_T^2\\,\\Delta R^2$, so changing $z$ also changes the prong separation $\\Delta R$, $\\tau_{21}$ and $D_2$.', None),
 ('Method', 'Edit D: boost all particles into the jet rest frame ($\\vec\\beta = \\vec p_{jet}/E_{jet}$), rotate them by $\\alpha = \\arccos c_0 - \\arccos c_1$ about $\\hat n_+ \\times \\hat\\beta$ so that $|\\cos\\theta^*|$ goes from $c_0$ to $c_1 = c_0 + \\Delta c$, boost back. All particles get the same Lorentz transformation $\\Lambda = B(-\\vec\\beta) R B(\\vec\\beta)$, which leaves the jet four-momentum unchanged: jet mass, pT, direction and every $p_i\\cdot p_j$ are exactly the same; only the lab-frame energies and angles change. Controls: c$\\varphi$ (rotation by the same angle about $\\hat\\beta$: sharing unchanged, prongs moved), c-side (prongs moved and widened at fixed particle pT so that $z$ is fixed and $\\Delta R$, $\\tau_{21}$ match D), and a translation. References: MLPx on $(m, p_T, \\tau_{21}, D_2, N, \\Delta R)$ (never sees $z$), MLPz = MLPx + $z_g$, 5 retrains each.', 'results/physics/h45_edit.png'),
 ('Result', 'All eight trained taggers (JetClass ParT/ParticleNet, hls4ml JEDI-linear and three small nets) raise their W/Z score when the prongs are made more balanced and lower it when they are made more asymmetric (up to −2.6 log-odds on W jets), ordered in $\\Delta c$; c$\\varphi$ and permuted networks give about 0. MLPx, which never sees $z$, predicts the same sign and a similar size (network/MLPx median 0.9–1.4), and MLPz ≈ MLPx.', 'results/physics/h45_response.png'),
 ('Checks', 'Edit invariants exact to $10^{-12}$; accuracies reproduce the known values; Holm over 448 tests; C2ST shows strongly balanced edited jets are distinguishable from natural jets (AUC 0.75–0.90). JEDI-linear\'s input moments $M_0$ and $m_A=\\sqrt{M_0M_2}$ change by < 1 % under the edit.', None),
 ('Physical significance', 'The taggers respond to the momentum sharing between the prongs in the direction QCD splitting physics suggests, but at fixed jet mass $z$, $\\Delta R$ and the prong masses are tied, so this test does not tell whether a network reads $z$ itself or $\\Delta R$, $\\tau_{21}$, $D_2$. JEDI-linear responds with its mass variable and $\\Sigma p_T$ fixed, so it uses more than mass windows and $\\Sigma p_T$ on W/Z jets.', None),
]
# glossary additions
('z (momentum sharing), z_g', '$z$ = pT fraction of the softer of the two prongs (exclusive-kT subjets). $z_g$ = the same fraction at the first Cambridge/Aachen declustering that passes the soft-drop condition $z > 0.1$ (β = 0, R0 = 0.8); soft-drop taggers cut on it'),
('cos θ* (prong-pair decay angle)', 'angle between one prong and the jet direction, measured in the rest frame of the two-prong system; $|\\cos\\theta^*| \\approx 1 - 2z$ for a fast jet: 0 = balanced prongs, 1 = all momentum in one prong'),
('c-side (side-effect-matched control)', 'a control edit built per jet to reproduce the tested edit\'s changes of the side-effect observables (here $\\Delta R$ and $\\tau_{21}$, within 5 %) while keeping the tested quantity (here $z$) fixed; the response to the edit minus the response to c-side isolates the tested quantity only if c-side has no other effects'),
```
