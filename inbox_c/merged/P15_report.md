# P15 — ParticleNet internals: layer directions, dynamic graphs, channel atlas, literal channel circuits, LRP / EFP / Lund baselines (2026-10-20)

Job `anrunw-ai-c-p15-run` (GTX 1080 Ti, 40 min), `python -m interp.p15_particlenet_internal`; smoke job `anrunw-ai-c-p15-smoke`
(60 jets/class) before it; figure re-plot `anrunw-ai-c-p15-plot` (CPU, `--plot`). Script `interp/p15_particlenet_internal.py`
(reuses `interp/pn_net.py`, `interp/p4_particlenet.pn_inputs` / `check_config`, `interp/_lund.py`). Results
`results/particlenet/p15_particlenet_internal.json` (smoke: `p15_particlenet_internal_smoke.json`); figures
`results/particlenet/p15_das_layers.png`, `results/particlenet/p15_circuits.png`; logs `logs/c-p15-run.log`, `logs/c-p15-smoke.log`,
`logs/c-p15-plot.log`. No training. Claims C5 and a C7 analogue; review weaknesses 6 (cross-architecture) and 7 (prior-method baselines).

## Conclusion
Inside ParticleNet, the response to a rigid φ shift of all particles, which leaves the jet mass unchanged, is carried by one direction
in the first two graph layers. The same search also finds such a direction in a copy of the network with shuffled weights, so finding
it does not show that training built a mass variable. The direction does not behave like the axis-moment mass: a real mass increase of
the same size moves it about a third as much, and a pT rescaling hardly moves it. The neighbour graphs that ParticleNet rebuilds at each
layer carry under 2 % of the response, so the response travels through per-particle terms, as P4 found. A few dozen pooled channels
kept alone reproduce ParticleNet's W/Z calls (ParT has no small circuit of this kind) but not its 10-class calls. EFP and Lund-plane
probes, which do not use the jet axis, predict no response to the shift. Still open: whether a multi-dimensional subspace in the later
layers carries a mass-like variable.

## Procedure
- **Models.** ParticleNet_full (17 per-particle features) and ParticleNet_kin (7), official JetClass weights, `interp/pn_net.py`
  (weaver-core 0.4.17 code; 3 EdgeConv blocks with k = 16 and 64 / 128 / 256 channels, no fusion, masked mean pooling, FC 256 + ReLU → 10).
  - The forward is re-implemented with hooks (`fw`). It matches weaver's forward exactly (max |ΔL| = 0).
  - Sites:
    - ec1, ec2, ec3: outputs of EdgeConv blocks 1–3.
    - Pooled 256-d feature: the jet mean of ec3. A jet-level interchange at ec3 is therefore the same as one on the pooled feature.
    - fc: the 256 FC hidden units after the ReLU.
  - Weight-shuffled control: every conv and linear weight tensor has its entries permuted (seed 15); BatchNorm is unchanged.
- **Sample.** JetClass `val_5M`, file `<sample>_120.root` of each class: 2,000 random rows per class (`default_rng(15)`), so
  N = 20,000. Jets are rebuilt with `part_core.build` (η = η_jet + sign(η_jet) Δη).
  - One random fit/test split (seed 16). The W/Z set is the 4,000 Wqq + Zqq jets: 1,966 fit, 2,034 test.
  - Unedited-accuracy check: P4's P1 rows (test files, rows 3,000–4,000 per class).
- **Edits** (W/Z jets; 4-vectors rebuilt). m is the invariant mass. $m_A=\sqrt{\sum p_T\cdot\sum p_T\Delta R^2}$ is the axis-moment
  mass about the stored axis (P8v's "mom").

  | edit | what changes | measured mean change |
  |---|---|---|
  | φ + d, d = 0.03 / 0.06 / 0.09 | m fixed | Δlog m 0.0000 (max 1e-4); Δlog $m_A$ +0.024 / +0.087 / +0.176 |
  | η + 0.06 | m fixed | Δlog $m_A$ +0.074 |
  | angles × k, k = 1.0910 ("angk") | Δlog $m_A$ set equal to that of φ + 0.06 | Δlog m +0.086, Δlog $m_A$ +0.087 |
  | angles × 1.1 | real mass change | – |
  | pT and particle mass × 1.25 | m and $m_A$ × 1.25 | – |
  | rotation of (Δη, Δφ) by 0.3 rad | ΔR and all pairwise distances kept | Δlog $m_A$ 0, Δlog m −0.0001 |
- **(1) DAS per site.**
  - Fit: one unit vector q, 300 Adam steps, batch 128, lr 0.02, on the W/Z fit half. Jet-level interchange
    $h_p \leftarrow h_p + ((\bar h'_{cf}-\bar h)\cdot q)\,q$ (fc: $h \leftarrow h + ((h'_{cf}-h)\cdot q)\,q$). The loss is the MSE to
    the counterfactual's 10 logits, with counterfactuals φ + 0.03 / 0.06 / 0.09 of the same jet.
  - Metrics on the W/Z test half, with $d=\ell_W-\ell_Z$:
    - share of the edit's Δd reproduced, $\sum(d_{int}-d)/\sum(d_{cf}-d)$;
    - IIA $=1-\sum|L_{int}-L_{cf}|^2/\sum|L-L_{cf}|^2$;
    - held-out edits η + 0.06, angk and pT × 1.25.
  - Controls: 5 random unit vectors per site; the same DAS fit on the weight-shuffled network.
  - Direction readings, using the jet projection $s=\bar h\cdot q$:
    - (a) Edit regression $\Delta s=b_A\,\Delta\log m_A+b_{inv}\,\Delta\log m$ over the 8 edits.
    - (b) P8v ratios $(\Delta s/\Delta\log m_A)_e/(\Delta s/\Delta\log m_A)_{\phi+0.06}$. An $m_A$ variable predicts 1.
    - (c) Natural-jet regression.
    - (d) Natural-donor substitution (P8v set B: same class, |Δlog $m_A$| ≥ 0.10): slope of Δd on Δs, relative to the self-interchange
      slope from the jet's own φ + 0.06 counterfactual.
- **(2) Dynamic graphs** (W/Z test half; φ + 0.06, angles × 1.1, rotation 0.3).
  - Fraction of kNN edges changed per layer, over real particles.
  - Variants: layer l's graph frozen to the unedited graph (l = 1, 2, 3, all); "graphs only" = unedited features with the edited run's
    graphs.
  - Share = Δ(variant) / Δ(full edit) for Δd and ΔW→Z.
- **(3) Channel atlas** (all 20,000 jets; pooled 256 channels, and FC units).
  - Per channel: Spearman ρ with 20 observables. The observables are log m, log $m_A$, log $p_T$, log($S_2/S_0$), girth, $N$,
    $N_{ch}$, τ1, τ21, τ32, D2, $p_TD$, $z_{lead}$, ΔR12, the charged / photon / neutral-hadron $p_T$ fractions, the lepton count,
    the axis offset $|M_1|/M_0$, and the maximum log(1 + |d0|/σ).
    - τ_N uses a one-pass axis approximation: the N hardest particles plus one $p_T$-weighted k-means step.
    - D2 uses β = 1 on the 60 hardest particles.
  - Also per channel: class composition of the top and bottom 1 %; R² from class alone; conditional R² (within-class linear fit on the
    20 observables, fit half → test half).
  - Control: 256 random orthonormal directions of the same space. Mann–Whitney p and a bootstrap interval for the difference of
    medians. Fisher-z p per (channel, observable), Holm over all pairs.
- **(4) Literal channel circuits** at three levels:
  - ec2 (128 per-particle channels; donor values slot-aligned with wrap as weaver's padding; block 3's kNN graph is rebuilt from the
    mixed features);
  - pool (256);
  - fc (256).

  Ranking and evaluation:
  - Ranking: |mean s · IG attribution| to d, with IG from the donor value (α = 0.1–0.9; s = +1 for W, −1 for Z). Donors are
    $p_T$-matched opposite-class jets from the W/Z fit half. IG completeness 0.999–1.009.
  - Kept alone: the top K (K = 8, 16, 32, 64). Every other channel at that level takes the donor's value. W/Z test half: $p_T$-matched
    opposite-class donors. 10-class read: 5,000 test-half jets (every 2nd) with uniform random donors.
  - Metrics: W/Z accuracy (10-class call = label on true W/Z jets), binary W/Z accuracy (sign of d),
    $F=\sum s(d_c-d_{corr})/\sum s(d-d_{corr})$, 10-class accuracy and confusion.
  - Controls: 100 random sets of live channels per K; permutation p and normal p from the random distribution; McNemar against the
    full model; Holm over 24 tests.
  - With all channels kept, the circuit reproduces the model (max |ΔL| ≤ 4e-6).
- **(5) Baselines** (W/Z test half).
  - LRP-ε: ε = 0.1 at every ReLU pre-activation and at d (backward rescaling $g\,z/(z+\epsilon\,\mathrm{sign}z)$). Relevance is taken
    at the normalised input features and summed per particle.
  - Relevance-weighted $\langle\Delta R\rangle_R=\sum|R_i|\Delta R_i/\sum|R_i|$.
  - Ridge probes of d, fitted on the fit half:
    - 35 EFPs (all 20 connected multigraphs with ≤ 4 edges, β = 1, 60 hardest particles, plus their products up to degree 4);
    - primary Lund plane (C/A on the 60 hardest particles; 6 × 8 bins in (ln 1/Δ, ln $k_T$) plus the emission count);
    - (log m, log $m_A$, log $p_T$) as a known-observable reference.
  - Per-jet prediction of the φ + 0.06 change Δd from: the DAS interchange; the LRP-ε linearisation $\sum g_{LRP}\cdot\Delta x$; the
    plain gradient · Δx; each probe's change. Each prediction is calibrated by its response to the angk real-mass edit.
- **Statistics.** 1,000 bootstrap resamples over test jets, percentile 95 % intervals. Two-sided bootstrap p (floor 1/1001; normal p
  from the bootstrap s.e. where at the floor). Holm over the 8 DAS-vs-random tests and the 24 circuit tests.

## Numbers
**Unedited accuracy.**

| model | 20,000 val jets | P1 rows | P4, P1 rows |
|---|---|---|---|
| ParticleNet_full | 84.50 % | 84.56 % | 84.57 % |
| ParticleNet_kin | 70.14 % | 70.36 % | 70.35 % |

W/Z test half, full model: W/Z accuracy 71.9 % (full) / 60.1 % (kin); binary 81.7 / 72.5 %; 10-class on the 5,000-jet read 85.4 / 70.1 %.

**Edit responses** (true W jets called Z, all 2,000 W).

| model | unedited | φ + 0.06 | η + 0.06 | angk | angles × 1.1 | pT × 1.25 | rotation 0.3 |
|---|---|---|---|---|---|---|---|
| full | 10.8 % | 25.4 % | 21.4 % | 35.3 % | 38.1 % | 53.1 % | 11.1 % |
| kin | 17.4 % | 28.0 % | 25.4 % | 42.7 % | 45.3 % | 51.4 % | 17.8 % |

Mean Δd on the W/Z test half under φ + 0.06: −0.47 (full), −0.26 (kin).

**(1) One-direction DAS: share of the φ + 0.06 change in d reproduced (95 % CI), with IIA.**

| site | full | kin | random 5 directions, full (kin) | weight-shuffled network, full (kin) |
|---|---|---|---|---|
| ec1 | 1.16 [1.08, 1.24], IIA 0.30 | 1.43 [1.31, 1.56], IIA 0.38 | −0.07…0.05 (−0.05…0.30) | 0.95 [0.88, 1.03], IIA 0.46 (2.47, IIA 0.82) |
| ec2 | 0.86 [0.80, 0.93], IIA 0.27 | 0.84 [0.77, 0.92], IIA 0.40 | −0.02…0.00 (−0.05…0.09) | 1.23 [1.14, 1.33], IIA 0.54 (2.29, IIA 0.91) |
| ec3 = pooled | 0.11 [0.08, 0.15], IIA 0.22 | 0.29 [0.26, 0.33], IIA 0.47 | −0.01…0.04 (−0.01…0.09) | 0.98, IIA 0.86 (2.15, IIA 0.98) |
| fc | 0.04 [0.02, 0.05], IIA 0.54 | 0.25 [0.22, 0.27], IIA 0.47 | 0.00…0.02 (−0.02…0.02) | 0.90, IIA 0.94 (2.03, IIA 0.98) |

- DAS vs the mean random direction: H0 equal share. Holm p ≤ 6e-9 at every site except full fc (p = 1.4e-4).
- The weight-shuffled networks respond to φ + 0.06 with mean Δd −9.6 (full) and +11.4 (kin). That is 20–45 times the trained
  networks' response. The shuffled kin network calls every W/Z jet the same class.

Held-out edits (share of each edit's Δd reproduced by the φ-trained direction):

| site | η + 0.06 | angk (real mass, same Δ$m_A$) | pT × 1.25 |
|---|---|---|---|
| full ec1 | 0.85 [0.76, 0.93] | 0.21 [0.21, 0.23] | 0.02 [0.02, 0.03] |
| full ec2 | 1.27 | 0.17 | 0.25 |
| kin ec1 | 1.13 | 0.22 | 0.02 |
| kin ec2 | 1.34 | 0.05 | 0.09 |

Response of s per unit Δlog $m_A$, relative to φ + 0.06 (P8v ratios; an $m_A$ variable predicts 1 for every edit):

| site | angk / φ | pT × 1.25 / φ | η / φ |
|---|---|---|---|
| full ec1 | 0.31 [0.31, 0.32] | 0.018 [0.016, 0.019] | 0.62 |
| full ec2 | 0.33 | 0.32 | 1.28 |
| kin ec1 | 0.38 | 0.012 | 0.60 |
| kin ec2 | 0.15 | 0.19 | 1.31 |
| kin fc | 0.92 [0.88, 0.97] | 0.98 [0.94, 1.02] | 0.98 |

- Full fc and ec3 change s by < 0.03 per edit, so their ratios are noise-dominated.
- Edit regression of Δs on (Δlog $m_A$, Δlog m), ratio $b_A/b_{inv}$:
  - full: ec1 −1.13 [−1.14, −1.12], ec2 −1.48, ec3 −0.10, fc −0.48;
  - kin: ec1 −1.15, ec2 −1.22, ec3 −10.7 [−61, 35], fc 2.49 [1.71, 3.54] (share $b_A/(b_A+b_{inv})$ 0.71 [0.63, 0.78]; 88 % of
    bootstraps above 2).
  - A negative ratio means that real mass changes move s less, or in the other direction, than the same Δ$m_A$ from a φ shift.
- Natural jets: corr(log $m_A$, log m) = 0.998 on the W/Z sample. The natural and donor two-variable regressions therefore cannot
  separate the two (intervals span 0). |corr(s, log $m_A$)| = 0.62 / 0.79 (full ec1 / ec2) and 0.63 / 0.67 (kin).
- Natural-donor substitution (2,034 recipients; donors differ by |Δlog $m_A$| = 0.28 on average). Slope of Δd on Δs relative to the
  self-interchange slope:
  - full: ec1 0.44 [0.36, 0.51], ec2 0.58 [0.49, 0.69], ec3 0.83 [0.73, 0.96];
  - kin: ec1 0.34 [0.28, 0.41], ec2 0.44 [0.32, 0.57], ec3 0.57 [0.49, 0.66].
  - fc gives 1.00 by construction, because d is linear in the fc units. Random-direction slope (full ec1): 0.023, against DAS 0.120
    and self 0.275.

**(2) Dynamic graphs.**

kNN edges changed per layer (layers 1 / 2 / 3), full (kin):
- φ + 0.06: 0 / 5.3 % / 8.6 % (0 / 6.0 % / 8.5 %);
- angles × 1.1: 0 / 2.2 % / 3.7 % (0 / 2.1 % / 3.6 %);
- rotation: 0 / 1.8 % / 4.0 % (0 / 1.2 % / 2.4 %).

Layer 1's graph cannot change under these edits, because a rigid shift, a uniform scaling and a rotation keep every neighbour order in
(Δη, Δφ). Re-run noise: edge overlap 1.000.

Share of the full edit's Δd that remains when graphs are frozen (full model, φ + 0.06):

| variant | share of Δd | removed | share of ΔW→Z |
|---|---|---|---|
| layer 2 frozen | 1.007 [0.990, 1.024] | – | – |
| layer 3 frozen | 0.982 [0.970, 0.994] | 1.8 % [0.6, 3.0] | – |
| all three frozen | 0.993 [0.975, 1.014] | – | 0.97 [0.92, 1.03] |
| graphs only (unedited features, edited graphs) | −0.04 [−0.06, −0.02] | – | 0.01 [−0.04, 0.06] |

- Kin: all frozen 1.013 [0.996, 1.032]; graphs only −0.06 [−0.08, −0.04].
- Angles × 1.1, all frozen: 0.998 (full), 0.994 (kin).
- Rotation shares are undefined in practice, because the full Δd is ≈ 0 (0.013 / −0.010).

**(3) Channel atlas** (pooled feature).
- Live channels: 220 / 256 (full), 191 / 256 (kin). Holm-significant (channel, observable) pairs: 3,533 / 4,400 and 3,203 / 3,820.
- Best-matching observable, count of channels:
  - full: lepton count 63, log $m_A$ 47, N 27, D2 19, log m 17;
  - kin: log $m_A$ 56, N 33, D2 18, $p_TD$ 18.
  - Per channel, ρ(log $m_A$) and ρ(log m) differ by ≤ 0.007, so "$m_A$" vs "m" as top pattern is not a distinction this sample makes.
- Channels against random rotations of the same space (full; kin in parentheses):

  | metric (median) | channels | rotations | difference [95 % CI] | Mann–Whitney p |
  |---|---|---|---|---|
  | max \|ρ\| | 0.32 (0.35) | 0.37 (0.38) | −0.047 [−0.093, −0.020] | 1.2e-5 (5.6e-3) |
  | conditional R² | 0.068 (0.11) | 0.19 (0.19) | – | 2e-26 (2e-17) |
  | R² from class alone | 0.18 (0.14) | 0.37 (0.31) | – | – |

  - FC units against rotations: max |ρ| 0.35 vs 0.35, p 0.59.
- Most single-observable channels (full):
  - channel 233: ρ(N) −0.94, conditional R² 0.76; top 1 % mostly QCD;
  - channel 87: ρ($z_{lead}$) 0.90;
  - channel 1: ρ(log $m_A$) 0.80, ρ(log m) 0.79; top 1 % top-quark jets.
- The DAS ec3 direction: ρ(log $m_A$) −0.55 / ρ(log m) −0.55 (full), −0.78 / −0.77 (kin). The kin fc direction: +0.83 / +0.83.

**(4) Literal channel circuits** (W/Z test half N = 2,034; 10-class read N = 5,000).

| model / level | K = 8 | 16 | 32 | 64 | full model | all resampled |
|---|---|---|---|---|---|---|
| full / pool: W/Z acc. | 44.6 % [42.6, 46.9] | 63.0 % | 70.9 % | 73.1 % | 71.9 % | 15.3 % |
| full / pool: random mean (95th pct) | 16.2 (19.9) | 17.9 (23.7) | 20.5 (26.8) | 29.7 (44.6) | | |
| full / pool: F | 0.50 | 0.69 | 0.85 | 0.95 | | |
| full / pool: 10-class | 13.3 % | 20.1 % | 36.6 % | 45.1 % | 85.4 % | 9.3 % |
| full / fc: W/Z | 24.5 % | 32.3 % | 55.3 % | 76.6 % | 71.9 % | 15.3 % |
| full / fc: 10-class | 10.3 % | 11.3 % | 13.1 % | 24.4 % (random 25.2) | | |
| full / ec2: W/Z | 28.3 % | 41.0 % | 50.0 % | 61.2 % (random 44.5) | 71.9 % | 19.3 % |
| full / ec2: 10-class | 12.3 % | 22.8 % | 34.3 % | 65.2 % | | |
| kin / pool: W/Z | 57.1 % (random 22.6) | 60.0 % | 61.4 % | 60.7 % | 60.1 % | 21.7 % |
| kin / pool: 10-class | 13.8 % | 26.9 % | 38.0 % | 52.5 % | 70.1 % | 9.1 % |
| kin / fc: W/Z | 28.2 % | 41.9 % | 58.5 % | 65.9 % | 60.1 % | |
| kin / ec2: W/Z | 39.1 % | 45.4 % | 40.5 % | 50.7 % (random 43.9, perm p 0.15) | 60.1 % | 25.3 % |

- McNemar against the full model (W/Z correctness):
  - full pool K = 32: p 0.29 (171 right→wrong, 151 wrong→right); K = 64: p 0.10;
  - kin pool K = 16: p 1.0;
  - full fc K = 64: p 6e-9 (more right than the model);
  - all K ≤ 16 at fc and ec2: p < 1e-27.
- Random-set p: 23 of 24 tests are at the permutation floor 1/101, so Holm-adjusted p = 0.24. With normal p from the random
  distribution, Holm p ≤ 0.015 for 23 of 24; kin ec2 K = 64 gives 0.10.
- Top-ranked pooled channels (full): 169, 153, 224 (IG scores 0.47, 0.45, 0.39). They are not among the single-observable channels
  above.

**(5) Baselines** (full; kin in parentheses).
- LRP-ε conservation, Σ relevance / d: median −0.38 (−0.14), IQR [−1.10, 0.30]. Bias and BatchNorm shifts absorb most relevance.
- Mean relevance per particle against ΔR:
  - W jets: +0.036 to +0.038 at ΔR 0.02–0.1, −0.038 at ΔR 0.2–0.3;
  - Z jets: +0.19 to +0.22 at ΔR < 0.1, −0.055 at 0.3–0.4.
  - Against $p_T$: within ±0.03 in all bins.
- $\langle\Delta R\rangle_R$ against log $m_A$: r 0.384 [0.33, 0.43]; against log m: 0.385. Difference −0.0008 [−0.004, 0.003], p 0.6
  (kin: 0.387 both).
- Probes of d on the test half:

  | probe | R² | call agreement with the model |
  |---|---|---|
  | EFP linear | 0.06 (0.19) | 0.61 (0.67) |
  | Lund primary | 0.03 (0.08) | 0.49 (0.62) |
  | (m, $m_A$, $p_T$) | 0.03 (0.13) | 0.58 (0.75) |

- Prediction of the per-jet φ + 0.06 change Δd. Observed mean Δd −0.47 (−0.26); 197 W jets flip to Z.

  | predictor | per-jet r [95 % CI] | mean predicted Δd | W→Z flips predicted (also observed) | angk gain → calibrated φ ratio |
  |---|---|---|---|---|
  | DAS ec1 | 0.61 [0.58, 0.64] (0.51) | −0.54 | 187 (129) | – |
  | DAS ec2 | 0.51 (0.53) | −0.40 | 133 (92) | – |
  | DAS ec3 | 0.21 (0.11) | −0.05 | 15 | – |
  | DAS fc | 0.01 (0.19) | −0.02 | 16 | – |
  | gradient · Δx | 0.45 [0.41, 0.49] (0.37) | −0.51 | 134 (105) | 1.11 → 0.98 |
  | LRP-ε · Δx | 0.40 [0.35, 0.44] (0.31) | −0.14 | 35 (30) | 0.31 → 0.97 |
  | EFP linear | undefined: the prediction is constant (\|Δ\| ≤ 1e-8; EFP change under the shift ≤ 5e-7 relative) | 0 | 0 | 0.44 → 0 |
  | Lund primary | undefined: the features are unchanged by the shift | 0 | 0 | 0.09 → 0 |
  | (m, $m_A$, $p_T$) probe | 0.13 [0.09, 0.18] (0.07) | −0.26 | 55 (38) | 0.35 → 1.60 |

## Reading
- **H0 "no 1-dim direction" (thesis alternative, recovery < 0.2 at all layers).** Excluded at ec1 and ec2 in both models (lower CI ≥ 0.77;
  random directions ≤ 0.30). Not excluded at ec3 / fc for the full model (0.11, 0.04).
- **H1 "the direction is a trained, $m_A$-specific variable" (thesis: ≥ 0.5 from ec1 on, $m_A/m$ slope ratio > 2).**
  - The ≥ 0.5 recovery holds at ec1–ec2 but not after the pooling layer.
  - It is matched by the same search on the weight-shuffled network (0.9–2.5). One direction recovering this single-parameter edit is
    therefore not specific to training.
  - The ec1–ec2 directions respond to an equal-$m_A$ real mass change at 0.15–0.38 of their φ response, and to pT scaling at
    0.01–0.32. An $m_A$ variable predicts 1 for both.
  - The edit-regression ratio $b_A/b_{inv}$ is negative at ec1–ec2 (−1.1 to −1.5). The thesis prediction > 2 is not met.
  - Only the kin fc direction responds per unit $m_A$ equally to all four edits (ratios 0.92–0.98; $b_A/b_{inv}$ 2.49 [1.71, 3.54]). It
    carries 0.25 of the φ response.
  - Natural-donor substitution gives 0.34–0.58 of the self slope at ec1–ec2. ParT P8v gave 0.42.
- **Graphs.** The thesis prediction "< 10 % at every layer" is met. Freezing any layer's graph removes ≤ 1.8 % of Δd. Edited graphs alone
  carry ≤ 0 of Δd and 0.01–0.03 of ΔW→Z (intervals include 0). The alternative "later dynamic graphs carry the response" is excluded.
- **Literal circuits.** The thesis prediction "cap well below the model, as in ParT" is not met for W/Z accuracy at the pooled level:
  - pooled K = 32 (full) and K = 16 (kin) reach the full model's W/Z accuracy (McNemar p 0.29, 1.0);
  - the same circuits give 37 % / 27 % 10-class accuracy against 85 % / 70 %;
  - at ec2 and fc, K ≤ 32 stays below the model.
  - ParT's literal node circuits (P9): 29–59 % W/Z.
- **Atlas.** The pooled channels are less aligned with single observables, and less class-specific, than random rotations of the same
  space (Mann–Whitney p ≤ 6e-3 for max |ρ|; 2e-17 for conditional R²). The channel basis is therefore not privileged for these
  20 observables. log $m_A$ and log m cannot be separated on natural jets (r 0.998).
- **Baselines.** All read-outs are on the W/Z test half.
  - EFP-linear and Lund probes predict the calls weakly (R² ≤ 0.19). They predict zero φ response by construction (axis-free inputs),
    which matches the thesis prediction.
  - The gradient and LRP-ε linearisations predict the per-jet φ response at r 0.31–0.45. Calibrated on the angk edit, their mean
    response is 0.97–1.5 of the observed. The calibrated ratios are near 1 for the full model; for kin both are about 1.5.
  - The early DAS directions predict it at r 0.51–0.61.
  - LRP-ε's raw magnitude is 0.30 of the observed (gain 0.31), and its relevance does not sum to d (median conservation −0.38).
  - Relevance-weighted ΔR follows log $m_A$ and log m equally (difference p 0.6).

## Predictions of competing hypotheses and which are excluded
| hypothesis | prediction | observed | status |
|---|---|---|---|
| T1 shared mechanism with ParT: one $m_A$ direction from ec1 | recovery ≥ 0.5 at ec1+; per-dlog-$m_A$ ratios ≈ 1; $b_A/b_{inv}$ > 2 | recovery 0.84–1.43 at ec1–ec2 (shuffled 0.95–2.47); ratios 0.01–0.38; $b_A/b_{inv}$ < 0 | recovery met but not specific; $m_A$ form excluded at ec1–ec2 (CIs exclude 1) |
| T2 invariant-mass variable | φ response 0 | φ share 0.84–1.43; W→Z 11 → 25 % | excluded (as P4) |
| T3 axis-translation / offset variable (responds to rigid shifts, not to angle or pT scaling) | φ ≈ η per Δ$m_A$; angk, pT ≪ φ | η/φ 0.60–1.31; angk/φ 0.15–0.38; pT/φ 0.01–0.32 | not excluded at ec1–ec2 |
| A1 no 1-dim direction | < 0.2 everywhere | ec1–ec2 ≥ 0.77 (CI) | excluded |
| A2 later dynamic graphs carry the response | freeze removes ≫ 10 % | ≤ 1.8 % | excluded |
| literal circuit cap as in ParT | pooled K ≤ 64 ≪ model | pooled K = 32 = model on W/Z; 10-class ≪ model | excluded for W/Z at the pooled level |

## Issues and limits
- DAS was fitted only with k = 1 and 300 steps. Deeper sites may need more dimensions (P8 used k = 1, 2, 4). The low ec3 / fc values say
  only that no single direction carries the response there.
- The DAS loss is the 10-logit MSE (P8 convention). At ec1, IIA is 0.30, while the d share overshoots (1.16 / 1.43): the interchange
  reproduces the W−Z part more strongly than the other logits.
- The weight-shuffled control has a response 20–45 times larger than the trained networks'. Its shares are relative to its own edit
  response.
- Natural-jet and donor regressions cannot separate $m_A$ from m (r 0.998). Only the edit-based ratios separate them.
- τ_N use a one-pass axis approximation. D2, the EFPs and the Lund plane use the 60 hardest particles.
- Random-set permutation p sits at the 1/101 floor; normal p are reported alongside.
- The LRP-ε rule (ε = 0.1) leaves relevance largely unconserved because of biases and BatchNorm shifts. Other LRP rules (z+, γ) were
  not tried.

## Explainer entry (for site/explain.py)
- **Question.** Does ParticleNet carry the axis-relative mass response through one internal variable, as ParT does (C7)? Do its graph
  layers or a small set of channels carry the W/Z decision? How do standard tools (LRP, EFP and Lund-plane probes) read the same
  response?
- **Background.** P4: shifting every particle by 0.06 rad in φ leaves the invariant mass unchanged. It turns 12 → 28 % of ParticleNet's
  W calls into Z, through the per-particle ΔR input. ParT has one direction, from its embedding on, that follows the axis-moment mass
  $m_A=\sqrt{\sum p_T\cdot\sum p_T\Delta R^2}$ (P8/P8v).
- **Method.**
  - Fit one unit vector $q$ per layer so that writing a counterfactual jet's value of $\bar h\cdot q$ into the jet reproduces the
    shifted jet's logits (DAS). Compare with random $q$ and with the same fit on a weight-shuffled network.
  - Test the direction with edits of equal $\Delta\log m_A$: $R_e=\frac{(\Delta s/\Delta\log m_A)_e}{(\Delta s/\Delta\log m_A)_{\phi}}$,
    which is 1 for an $m_A$ variable.
  - Freeze each layer's kNN graph to the unedited one.
  - Keep the top-K channels by integrated gradients and resample the rest from $p_T$-matched opposite-class jets, against 100 random
    sets.
  - Compare per-jet predictions of the φ response: $\Delta d_{LRP}=\sum g_{LRP}\cdot\Delta x$, gradient, EFP / Lund / mass probes.
- **Result (figure p15_das_layers.png; p15_circuits.png).**
  - EdgeConv 1–2: one direction reproduces 0.84–1.43 of the φ response. A weight-shuffled network gives the same (0.95–2.47).
  - The direction responds to equal-$m_A$ angle scaling at $R\approx0.3$ and to $p_T$ scaling at $R\approx0.02$.
  - Frozen graphs remove ≤ 2 % of the response.
  - 32 pooled channels kept alone match the model's W/Z accuracy (70.9 vs 71.9 %), but give 37 % 10-class accuracy (model 85 %).
  - EFP and Lund probes predict exactly no φ response.
- **Checks.**
  - The re-implemented forward equals weaver's (max |ΔL| = 0).
  - Accuracy on P1 rows 84.56 / 70.36 % (P4: 84.57 / 70.35 %).
  - Mass-preserving edits: |Δlog m| ≤ 1e-4.
  - All-kept circuits reproduce the model (≤ 4e-6). IG completeness 0.999–1.009.
  - Re-run kNN overlap 1.000.
  - Holm over the DAS and circuit tests.
- **Why it matters.**
  - ParticleNet reaches ParT's input-level behaviour through per-particle terms, with the graphs inert for this response.
  - Its early-layer "mass direction" is a translation-type variable, not an $m_A$ copy, and its existence alone is not evidence of a
    trained mechanism: the shuffled-weight control is needed.
  - Standard EFP / Lund baselines cannot see this response at all, because they do not use the jet axis.
