# P18: ParT stage pipeline: where mass, prong and count variables become readable and causal, block by block (C4, A8) (agent report, 2026-10-20)

Job `anrunw-ai-c-p18-run` (NVIDIA RTX 2080 Ti, 3,382 s): `python -m interp.p18_stage_pipeline_part --models full,kin`.
- Smoke run `c-p18-smoke` (2,000 jets) passed.
- Figure re-rendered with shorter titles by `c-p18-plot` (job.sh, CPU, 14 s).
- Script: `interp/p18_stage_pipeline_part.py`.
- Results: `results/part/p18_stage_pipeline.json` (smoke: `p18_stage_pipeline_smoke.json`).
- Figure: `results/part/p18_stage_pipeline.png`.
- Logs: `logs/c-p18-run.log`, `logs/c-p18-smoke.log`, `logs/c-p18-plot.log`.

## Conclusion
ParT does not build its jet variables in the order "count, then prongs, then mass". The jet's multiplicity and pT can be read from the
input embedding, and its mass from the first block on. All three can be read just as well from an untrained network of the same
shape, so being readable does not show that the network learned them. Prong-shape variables (N-subjettiness ratios; τ21 compares how
well a jet looks two-pronged rather than one-pronged) are where the trained network differs from the controls: they become readable
in blocks 1–4. A single learned direction carries the network's response to a prong-destroying edit, and the share of that response
it carries rises from 0.05 at the embedding to 0.37–0.41 by block 7 (ParT_full). The response to mass and pT edits is carried best
by one direction at the embedding (0.63–0.95), and this share falls with depth. Still uncertain: the "prong" variable is defined
here only by one edit (random azimuths about the axis), which changes τ21, τ32 and D2 together, so which prong observable the network
uses is not resolved.

## Procedure
**Models.**
- Released ParT_full and ParT_kin (weaver-core 0.4.17).
- Controls, all with the ParT_full architecture and inputs:
  - perm: each Linear / Conv1d / attention in-projection weight of the released ParT_full has its output rows permuted, with the
    matching bias entries. One independent permutation per layer (seed 18); 48 tensors. Norm layers and cls_token are not changed.
  - init: weaver's default initialisation (`torch.manual_seed(18)`), not trained.
  - rot: the ParT_full features multiplied by a random orthogonal 128 × 128 matrix before the probe.

**Sample.**
- JetClass test_20M, `<sample>_100.root` of each of the 10 classes. 5,000 random rows per class (`numpy.default_rng(18)`, never
  first-k), N = 50,000.
- Train and test halves (25,000 each) come from a permutation with seed 18. Ridge alpha and the MLP early stop are chosen on the last
  quarter of the train half.
- Jets are rebuilt with `part_core.build` ($\eta=\eta_{jet}+\mathrm{sign}(\eta_{jet})\Delta\eta$).
- Unedited accuracy is also measured on the P1 rows (rows 3,000–4,000 of each class's `_100` file).

**Targets.** All are about the stored jet axis, with $\Delta R=\sqrt{\Delta\eta^2+\Delta\phi^2}$:
- $\log m_{inv}$: the invariant mass.
- $\log m_A=\tfrac12\log(M_0M_2)$, with $M_0=\sum p_T$ and $M_2=\sum p_T\Delta R^2$.
- $\log\langle\Delta R^2\rangle=\log(M_2/M_0)$: the per-particle radial moment.
- $\tau_{21}$, $\tau_{32}$: β = 1, k-means axes, as in `jetdistill.observables.compute`.
- $\log D_2$: β = 1, from the 24 hardest particles.
- N: the number of particles.
- $\log p_T$: the jet pT.

**Edits.** Each edit is applied to every real particle; px, py, pz, E, jet_pt and jet_energy are rebuilt.
- φ + 0.06: $m_{inv}$ and the pair features are unchanged; $\Delta\log m_A=+0.074$.
- Matched dilation (dil): $\Delta\eta,\Delta\phi\times k$ with $k=1.0766$, chosen so that the mean $\Delta\log m_A$ equals that of
  φ + 0.06. Result: $\Delta\log m_{inv}=+0.073$, $\Delta\log m_A=+0.074$.
- Azimuth scramble (scr, E24-style): each particle gets $\theta_i\sim U(0,2\pi)$ about the stored axis, with $p_T$, r and rank fixed.
  So $M_0$, $M_2$ and N are fixed, while $\Delta\tau_{21}=+0.290$, $\Delta\tau_{32}=+0.095$, $\Delta\log D_2=+0.277$ and
  $\Delta\log m_{inv}=-0.034$.
- Soft particles (soft5, H18-style): 5 on-manifold soft particles. Each copies a random member of the jet's softer half (pT, mass,
  PID, impact-parameter fields) at its donor's ΔR and a random azimuth. Result: ΔN = +5.00, $\Delta\log p_T=+0.026$,
  $\Delta\log m_A=+0.052$.
- pT × 1.1: particle pT and mass × 1.1. Result: $\Delta\log p_T=\Delta\log m=\Delta\log m_A=+0.095$.

**Calibration edit per target** (the matched real change):
- $m_{inv}$, $m_A$, $\langle\Delta R^2\rangle$: dil.
- τ21, τ32, D2: scr.
- N: soft5.
- $\log p_T$: pT × 1.1.

**Sites.**
- emb (embedding output) and b1–b8 (particle-block outputs): read as the pT-weighted jet mean of the particle tokens.
- c1, c2 (class-attention block outputs): read as the class token. The class blocks do not update the particle tokens.

**(a) Readability.**
- Probes: ridge, and a 2-layer MLP (128-64-1, GELU, Adam, best-validation of 600 steps). Both are fitted on the base jets of the
  train half.
- $R^2$ is measured on the test half, with 1,000 jet-bootstrap resamples.
- Indices:
  - onset: the first site where $R^2$'s lower bound exceeds the larger control (perm / init) upper bound.
  - t90: the first site where $R^2\ge 0.9\times$ the target's maximum over sites, with a bootstrap distribution paired across targets.
  - rise: $R^2_{best}-R^2_{emb}$, with Holm over 8 targets.
- Ordering tests: paired bootstrap of the group-mean t90 differences. The groups are low = {N, log pT, ⟨ΔR²⟩}, prong = {τ21, τ32,
  D2} and mass = {m_A}.

**(b) Calibrated edit response.**
- Per site, target and edit: the probe read-out change Δp on the test half, against the true change Δt.
- Gain $g=\sum\Delta p\,\Delta t/\sum\Delta t^2$, computed under the target's calibration edit. A gain of 1 means the read-out follows
  a real change of the variable at full size.
- For $m_A$, the ratio Δp(φ)/Δp(dil): 1 means the probe reads φ as an $m_A$ change.

**(c) One-dimensional interchange (DAS) per site and variable.**
- The counterfactual is the same jet with the edit applied to the per-particle inputs, keeping the base 4-vectors (pair features), as
  in P8. For soft5 it is the whole edited jet.
- Interchange rules:
  - Particle sites: $h_p\leftarrow h_p+((\bar h^{cf}_w-\bar h_w)\cdot q)\,q$ for every particle, where $\bar h_w$ is the pT-weighted
    mean.
  - Class sites: the same rule applied to the class token.
- Fitting: q (unit) is fitted by Adam (200 steps × 128 jets, lr 0.01) on 4,000 train-half jets. The loss is the MSE to the
  counterfactual's 10 logits.
- Evaluation on 1,000 test-half jets:
  - IIA (logit recovery) $=1-\sum\lVert L_{int}-L_{cf}\rVert^2/\sum\lVert L_{base}-L_{cf}\rVert^2$.
  - Call IIA: the share of call-changed jets that get the counterfactual call.
- Directions compared:
  - DAS q.
  - The ridge-probe direction of the variable's target.
  - 10 random q, run exactly.
  - 1,000 random q through the first-order (Jacobian) approximation of the interchange.
- Controls:
  - Random-direction null: p = (1 + #{linearised random ≥ linearised DAS}) / 1,001.
  - DAS refitted on the perm network at the same site, compared by a paired bootstrap over jets.
  - Holm over 11 sites × 5 variables per model.
- Variable → target: φ → $m_A$ at fixed $m_{inv}$; dil → $m_{inv}$ (and $m_A$); scr → τ21 (τ32 and D2 move with it); soft5 → N;
  pT × 1.1 → log pT.

**Hypotheses.**
- H0: readability and IIA are flat across depth.
- H1 (C4, JEDI-like): the low-depth variables peak in b1–2, the prong variables later, and $m_A$ last, with an IIA step.
- H_axis-alt: $m_A$ is readable from b1, with no later increase.

## Numbers
**Unedited accuracy.**

| | ParT_full | ParT_kin | perm | init |
|---|---|---|---|---|
| P1 rows, rebuilt jets | 85.93 % | 74.47 % | 11.34 % | 13.58 % |
| P1 rows, raw file jets | 85.93 % | 74.83 % | | |
| P18 sample (50,000) | 85.84 % | 73.88 % | 11.68 % | 13.73 % |

- References: 85.9 / 74.5 % (P1) and 85.93 / 74.47 % (P8).
- φ + 0.06, W → Z on true W (full): 10.98 → 31.64 % (P1: 12 → 31 %).
- ParT_kin: 16.1 → 35.2 %.

**(a) Ridge $R^2$ (test half, 95 % CI ≈ ±0.003–0.01), ParT_full; init and perm at the same site in brackets.**

| target | emb | b1 | b4 | b8 | c2 | t90 [boot CI] | onset vs controls |
|---|---|---|---|---|---|---|---|
| N | 0.989 (init 0.993) | 0.986 (0.979) | 0.977 (0.966) | 0.958 (0.957) | 0.885 | emb [emb, emb] | b1 |
| log pT | 0.989 (0.992) | 0.981 (0.980) | 0.945 (0.977) | 0.879 (0.974) | 0.073 | emb [emb, emb] | none |
| log ⟨ΔR²⟩ | 0.819 (0.820) | 0.941 (0.898) | 0.946 (0.892) | 0.929 (0.883) | 0.764 | b1 [b1, b1] | b1 |
| log m_inv | 0.825 (0.809) | 0.932 (0.891) | 0.931 (0.883) | 0.922 (0.869) | 0.889 | b1 [b1, b1] | emb |
| log m_A | 0.827 (0.809) | 0.937 (0.892) | 0.935 (0.883) | 0.925 (0.869) | 0.892 | b1 [b1, b1] | emb |
| τ21 | 0.537 (0.421; perm 0.448) | 0.729 (0.394) | 0.799 (0.377) | 0.787 (0.373) | 0.704 | b1 [b1, b1] | emb |
| τ32 | 0.055 (0.092) | 0.373 (0.114) | 0.505 (0.106) | 0.527 (0.101) | 0.469 | b4 [b4, b4] | b1 |
| log D2 | 0.802 (0.727) | 0.848 (0.703) | 0.866 (0.684) | 0.856 (0.672) | 0.756 | emb [emb, emb] | emb |

- ParT_kin, ridge t90: N, log pT and D2 at emb; ⟨ΔR²⟩, $m_{inv}$, $m_A$ and τ21 at b1; τ32 at b3.
- ParT_kin $m_A$ $R^2$: emb 0.815, b1 0.966, b8 0.938 (init 0.892 at b1). τ21 b1 0.810 (init 0.394). τ32 b4 0.568 (init 0.106).
- MLP probes give the same t90 or one block earlier: full $m_{inv}$, $m_A$ and ⟨ΔR²⟩ reach t90 at emb (MLP $R^2$ emb 0.96–0.97);
  τ32 at b3; τ21 at b1.
- The rot control equals the full model to ≤ 0.02 at the particle sites. Exceptions: ridge log ⟨ΔR²⟩ at c2 is 0.65 vs 0.76, and
  log D2 at c1–c2 is 0.52–0.55 vs 0.76–0.79.
- Ordering tests (ridge, 1,000 paired bootstraps; t90 in units of sites):
  - ParT_full:
    - t90(prong) − t90(low) = +1.33 [1.33, 1.33], p(H1 order violated) = 0.001.
    - t90(mass) − t90(prong) = −0.66 [−0.67, −0.67], p = 1.0. H1 predicts > 0.
    - t90(mass) − t90(low) = +0.67, p = 0.001.
  - ParT_kin: +1.13 [1.0, 1.33]; −0.46 [−0.67, −0.33], p = 1.0; +0.67.
  - MLP probes, ParT_full: mass − prong = −1.33 and mass − low = 0.0 (p = 1.0).
- Rise ($R^2_{best}-R^2_{emb}$), ridge, ParT_full: ⟨ΔR²⟩ +0.132, $m_{inv}$ +0.117, $m_A$ +0.120, τ21 +0.265, τ32 +0.474, D2 +0.064,
  each with p_Holm = 0.008 (the floor for 8 tests × 1,000 resamples). N and log pT: 0.
- H_axis-alt test for $m_A$ (ridge, max $R^2$ after b1 minus $R^2$(b1)): ParT_full +0.010 [0.009, 0.012]; ParT_kin −0.006
  [−0.006, −0.005]. $R^2$(b8) − $R^2$(b1): −0.012 [−0.014, −0.010] (full), −0.028 (kin).

**(b) Calibration gain (ridge read-out change / true change under the matched real edit; CI ≈ ±0.01).**

| target (edit) | model | emb | b1 | b4 | b8 | c2 | init b1 / b8 | perm b1 / b8 |
|---|---|---|---|---|---|---|---|---|
| m_A (dil) | full | 0.76 | 0.96 | 0.82 | 0.83 | 0.82 | 0.99 / 0.95 | 0.73 / 0.50 |
| m_A (dil) | kin | 0.77 | 0.95 | 0.90 | 0.84 | 0.84 | | |
| N (soft5) | full | 0.96 | 0.96 | 0.99 | 0.95 | 0.96 | 1.00 / 0.96 | 0.65 / 0.36 |
| log pT (pT × 1.1) | full | 0.96 | 0.93 | 0.79 | 0.70 | 0.03 | 0.98 / 0.97 | |
| τ21 (scr) | full | 0.00 | 0.77 | 0.77 | 0.79 | 0.72 | 0.01 / 0.01 | 0.03 / 0.00 |
| τ21 (scr) | kin | 0.00 | 1.00 | 0.79 | 0.80 | 0.74 | | |
| τ32 (scr) | full | 0.00 | −0.07 | 0.27 | 0.34 | 0.44 | 0.00 / 0.00 | 0.00 / 0.00 |
| τ32 (scr) | kin | 0.00 | 0.02 | 0.54 | 0.37 | 0.40 | | |
| log D2 (scr) | full | 0.00 | 0.59 | 0.80 | 0.81 | 1.15 | 0.03 / 0.01 | |

- φ vs dil on the $m_A$ ridge probe, Δp(φ)/Δp(dil):
  - ParT_full: 1.04–1.14 at emb–b8, 0.88–0.91 at c1–c2.
  - ParT_kin: varies between 0.16 and 1.69 across sites.
- The $m_{inv}$ ridge probe (ParT_full) moves under φ + 0.06 by +0.054 to +0.076 at every site, while the true $\Delta\log m_{inv}$ is
  0. Under dil it moves by +0.055 to +0.070, for a true change of +0.073.

**(c) DAS 1-d IIA (logit recovery; 1,000 test-half jets; 95 % CI ≈ ±0.02–0.04).**

| variable (edit) | model | emb | b1 | b2 | b4 | b6 | b7 | b8 | c1 | c2 | t90 [CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m_A (φ + 0.06) | full | 0.63 | 0.57 | 0.52 | 0.48 | 0.45 | 0.44 | 0.40 | 0.37 | 0.37 | emb |
| m (dil) | full | 0.89 | 0.88 | 0.87 | 0.83 | 0.77 | 0.75 | 0.75 | 0.54 | 0.54 | emb |
| pT (× 1.1) | full | 0.95 | 0.95 | 0.94 | 0.90 | 0.81 | 0.79 | 0.79 | 0.55 | 0.54 | emb |
| N (soft5) | full | 0.31 | 0.31 | 0.32 | 0.32 | 0.37 | 0.39 | 0.44 | 0.44 | 0.43 | b8 [b7, b8] |
| τ21 (scr) | full | 0.05 | 0.06 | 0.15 | 0.22 | 0.32 | 0.37 | 0.37 | 0.41 | 0.33 | b7 [b7, c1] |
| m_A (φ + 0.06) | kin | 0.87 | 0.74 | 0.71 | 0.61 | 0.60 | 0.61 | 0.71 | 0.66 | 0.77 | emb |
| m (dil) | kin | 0.89 | 0.88 | 0.87 | 0.84 | 0.81 | 0.80 | 0.81 | 0.55 | 0.49 | emb |
| pT (× 1.1) | kin | 0.97 | 0.95 | 0.94 | 0.91 | 0.88 | 0.86 | 0.86 | 0.54 | 0.50 | emb |
| N (soft5) | kin | 0.33 | 0.32 | 0.33 | 0.35 | 0.41 | 0.45 | 0.56 | 0.59 | 0.57 | b8 |
| τ21 (scr) | kin | 0.22 | 0.41 | 0.49 | 0.69 | 0.74 | 0.72 | 0.70 | 0.61 | 0.57 | b4 [b4, b4] |

- Largest increase between consecutive sites (selected as the maximum, so its CI is optimistic):
  - ParT_full: scr at b2, +0.097 [0.083, 0.113]; soft5 at b8, +0.042. φ, dil and pT have none (≤ 0.003).
  - ParT_kin: scr at b1, +0.194 [0.170, 0.219]; soft5 at b8, +0.111. φ: +0.107 at c2, after a fall from 0.87 to 0.60.
- IIA(b8) − IIA(emb):
  - ParT_full: scr +0.32 [0.28, 0.37]; soft5 +0.12 [0.10, 0.15]; φ −0.23 [−0.25, −0.20]; dil −0.14; pT −0.17.
  - ParT_kin: scr +0.48, soft5 +0.23, φ −0.16, dil −0.08, pT −0.11.
- Random directions:
  - Exact random q (10 per cell): mean IIA 0.000–0.028, max 0.091.
  - Linearised random q (1,000 per cell): 95th percentile 0.002–0.064, max 0.178.
  - DAS (linearised) exceeds the maximum of the 1,000 random draws in 54 of 55 cells for each model; the exception is scr at emb.
  - Raw p = 1/1,001. With 55 tests the smallest attainable Holm p is 0.055, so no cell reaches Holm 0.05 against this null (see Issues).
- Weight-permuted network: DAS IIA 0.00–0.30, with its largest values at c1–c2 (0.14–0.30).
  - Trained − perm, ParT_full: φ +0.08 to +0.49, dil +0.26 to +0.83, pT +0.27 to +0.86, soft5 +0.16 to +0.35, scr +0.03 to +0.21.
  - Bootstrap p_Holm = 0.055 (floor). A normal approximation from the bootstrap SE gives Holm < 0.05 in 54 of 55 cells (full; the
    exception is scr at c2) and 55 of 55 (kin).
- Ridge-probe direction used as the interchange direction: IIA between −0.08 and +0.05 at every site and variable. Cosine to the
  DAS q: 0.00–0.26.
- Linearisation check: on the exact random draws, linear vs exact correlation 0.999, max |Δ| 0.006. On the DAS q, the linearised
  IIA is on average 0.08 lower (correlation 0.98 full, 0.78 kin; ParT_kin φ at emb: exact 0.87 vs linear 0.13).

## Reading (scoped to these measurements)
- **Count and scale (N, log pT).** Readable at the embedding ($R^2$ 0.99), with a calibration gain of 0.96. The untrained network
  gives the same $R^2$ (0.99) and gain (0.98–1.00). log pT readability falls in the trained network after b4 and is near 0 at the
  ParT_full class token (0.07); in the untrained network it stays at 0.96.
- **Mass ($m_{inv}$, $m_A$, ⟨ΔR²⟩).** The ridge read-out reaches t90 at b1 (0.94), and the gain at b1 is 0.96. Readability does not
  increase after b1 (+0.010 full, −0.006 kin).
  - The untrained network reaches 0.89 at b1, with gain 0.99. The margin of the trained network over it is 0.04–0.07 $R^2$.
  - $m_{inv}$ and $m_A$ are read identically, with the same $R^2$. The $m_{inv}$ probe also moves under φ + 0.06, which leaves
    $m_{inv}$ unchanged, by 0.9–1.1 × its response to the matched dilation. In this sample these probes cannot separate $m_{inv}$
    from $m_A$.
- **Prong (τ21, τ32, D2).** This is the only group where the trained network exceeds both controls by a large margin.
  - τ21 at b4: $R^2$ 0.80 vs 0.38 (init) and 0.34 (perm).
  - τ32 at b4: 0.51 vs 0.11 and 0.03.
  - The calibration gain under the azimuth scramble is 0 at the embedding and 0.01 in the controls at every site.
  - In the trained network the gain is 0.77 (τ21) from b1, and rises from b2 to b8 for τ32 (0.13 → 0.34 full, 0.22 → 0.54 at b4 kin).
- **Causal 1-d interchange.**
  - pT, dilation and φ responses: one direction reproduces 0.63–0.97 of the logit change at the embedding, and less at each later
    block. The variable is most concentrated in one dimension at the input and is spread across more dimensions with depth.
  - Azimuth-scramble response: 1-d IIA rises from 0.05 / 0.22 (emb; full / kin) to 0.37–0.41 / 0.74 (b6–c1).
  - Soft-particle response: rises from 0.31–0.33 to 0.44–0.59 by b8.
  - The readable (probe) direction and the causal (DAS) direction are different directions: interchange along the probe direction
    gives IIA ≈ 0, with cosine ≤ 0.26.
- **Hypotheses.**
  - H0 (flat): excluded for readability of τ21, τ32, ⟨ΔR²⟩ and $m$ (rise p_Holm 0.008), and for the IIA of scr, soft5, φ, dil and
    pT (b8 − emb CIs exclude 0).
  - H1 (low → prong → mass last, with an $m_A$ IIA step): excluded in its mass-last part. Mass t90 precedes prong t90 (−0.66 sites,
    p = 1.0 for the H1 direction). No IIA step for $m_A$: the φ IIA falls with depth in full; in kin it falls and then partly
    recovers at b8 / c2.
  - The order that fits these data, for both models: count and scale at the input; mass at b1; prong (τ32) at b3–b4 in readability,
    with the prong 1-d variable forming over b2–b7.
  - H_axis-alt ($m_A$ readable from b1, no later increase): consistent with the readability data. The φ / dil read-out ratio at b1
    (1.11 full) means the b1 probe reads φ as an $m_A$-size change.
- **For A8 (JEDI-linear stages).** ParT shows an early mass variable (b1) and a later prong variable. This is a different order from
  JEDI-linear's "objects → axis-moment mass", if "objects" corresponds to prong structure. The mass-type readability in ParT is also
  reached by an untrained network.

## Predictions of competing hypotheses and which are excluded
| hypothesis | prediction | observed | status |
|---|---|---|---|
| H0 flat | t90 = emb for all, IIA flat | t90 spread emb → b4; IIA(b8) − IIA(emb) from −0.23 to +0.48 | excluded |
| H1 (C4, JEDI-like) | t90: low < prong < mass; $m_A$ IIA step | low < mass ≤ prong; $m_A$ IIA highest at emb | excluded (mass-last part) |
| H_axis-alt | $m_A$ readable from b1, no later rise | max after b1 − b1 = +0.010 (full), −0.006 (kin) | consistent |
| readability = trained computation | trained ≫ controls | holds for prong only; mass and count ≈ untrained | refines |

## Issues and limits
- **Holm floor.** With 1,000 random directions and 55 tests, the smallest Holm-adjusted p is 0.055. The same holds for the paired
  bootstrap against the permuted net. 54–55 of 55 DAS cells lie above the maximum of all 1,000 random directions, and above the
  permuted-net DAS by 10–40 bootstrap SE. A rerun with 10,000 linearised draws would bring the floor to 0.0055; the Jacobians were
  not saved.
- **Linearised null.** The random-direction null is evaluated in the first-order approximation. It matches exact runs for random q
  (max |Δ| 0.006), but underestimates the DAS q's IIA (ParT_kin φ at emb: 0.13 linear vs 0.87 exact). The p-values compare linear
  with linear.
- **Prong edit.** The azimuth scramble changes τ21, τ32, D2 and $m_{inv}$ together, so the DAS "prong" variable is not specific to
  one observable. The τ-specific statements rest on the readability and gain measurements only.
- **Soft5 edit.** soft5 also raises log pT (+0.026) and $m_A$ (+0.052); the N calibration is not mass-free.
- **pT × 1.1 edit.** It moves $m_{inv}$ and $m_A$ by the same +0.095, so its DAS variable is "scale", not pT alone.
- **Pair features in the counterfactuals.** DAS counterfactuals keep the base pair features (except soft5). The scr and dil
  responses through the pair route (P2: 1–17 %) are not in the counterfactual.
- **Index resolution.** The t90 index is discrete (11 sites), so its bootstrap CIs are often a single site. Ordering-test CIs are
  therefore narrow by construction.
- **Selected step.** The "largest step" of IIA is the maximum over sites, so its CI is optimistic.
- **MLP outliers.** The ParT_kin MLP probe at emb produced outlier predictions in some bootstrap resamples: rise CI for τ21
  [0.12, 2.04]. Ridge numbers are the primary ones.
- **Readout pooling.** Particle sites are read through the pT-weighted mean only. A variable carried per particle and not in the
  weighted mean (for example, carried by the soft particles) would not be counted at b1–b8.

## Explainer entry (for site/explain.py)
- **Question.** Does ParT compute jet variables in stages: the particle count first, then prong structure, then the jet mass (as
  JEDI-linear does: cuts → objects → axis-moment mass)? At which block does each variable become (i) readable and (ii) used?
- **Background.**
  - ParT's mass response follows the axis-moment mass $m_A=\sqrt{\sum p_T\cdot\sum p_T\Delta R^2}$ (C3).
  - P8 found one jet-mean direction carrying the φ-shift effect at every depth.
  - A probe can read a variable that the network does not use; an untrained network's features are a baseline for this.
- **Method.**
  - Fit ridge and MLP probes for 8 jet variables at the embedding, the 8 particle blocks and the 2 class blocks, on 25,000 jets;
    test on 25,000.
  - Calibrate each probe with a real edit of known size: $g=\frac{\sum\Delta p\,\Delta t}{\sum\Delta t^2}$, where $g=1$ means the
    read-out follows the change at full size. The edits are a matched dilation for mass, an azimuth scramble for τ21 / τ32 / D2,
    5 soft particles for N, and pT × 1.1 for pT.
  - Fit one direction $q$ per block so that writing the edited jet's value of $\bar h\cdot q$ into the unedited jet reproduces the
    edited jet's logits:
    $\mathrm{IIA}=1-\frac{\sum\lVert L_{int}-L_{cf}\rVert^2}{\sum\lVert L_{base}-L_{cf}\rVert^2}$.
  - Compare with 1,000 random directions, the probe direction, a weight-permuted network and an untrained network.
- **Result (figure p18_stage_pipeline.png).**
  - N and pT are readable at the embedding ($R^2$ 0.99), and mass from block 1 ($R^2$ 0.94). An untrained network gives 0.99 and 0.89.
  - τ21 and τ32 are readable only in the trained network: τ32 $R^2$ 0.06 → 0.51 by block 4, untrained 0.11; the scramble
    calibration gain is 0.77 vs 0.01.
  - One direction carries 0.63–0.95 of the mass / pT edit response at the embedding, falling with depth. For the prong edit the
    share rises from 0.05 to 0.41 by block 7 (ParT_full).
  - Probe directions carry ≈ 0 of any response.
- **Checks.**
  - Unedited accuracy 85.93 / 74.47 % on the P1 rows; 85.84 / 73.88 % on the sample.
  - φ + 0.06 W → Z 11 → 32 %.
  - Mass-preserving edit: $\Delta\log m_{inv}=0$.
  - Linearised random-direction null checked against exact runs (max |Δ| 0.006).
  - Random-rotation basis gives the same $R^2$.
  - Holm over blocks × variables.
- **Why it matters.**
  - ParT's order is: scale and count at the input, mass at block 1, prong structure over blocks 2–7. This is not JEDI-linear's
    mass-last order.
  - Being readable is not evidence of computation here: mass and count are as readable in an untrained ParT. The prong variables
    and the 1-d causal directions are the parts that depend on training.
