# P8: where ParT's axis-relative mass forms (residual patching, DAS, substitution into n98) (agent report, 2026-10-19)

Job `anrunw-ai-c-p8-run` (NVIDIA GTX 1080 Ti, 1,934 s), `python -m interp.p8_mass_formation_part`. Smoke runs `c-p8-smoke`
and `c-p8-smoke2` passed. The second smoke run was needed because cumulative patches were added after the first smoke showed that
single sublayers restore only 1–2 %. The figure was re-rendered by `c-p8-plot` (job.sh, CPU, 27 s).
Script `interp/p8_mass_formation_part.py`; results `results/part/p8_mass_formation.json` (smoke: `p8_mass_formation_smoke.json`);
figure `results/part/p8_mass_formation.png`; logs `logs/c-p8-run.log`, `logs/c-p8-smoke.log`, `logs/c-p8-smoke2.log`, `logs/c-p8-plot.log`.

## Conclusion
ParT does not compute its axis-relative mass at one layer. When every particle is shifted in φ (the true jet mass is unchanged), the
effect of the shift on the W-versus-Z output builds up a little in each of the eight particle layers. More than half of it reaches the
class token directly from the per-particle input embedding, without passing through any attention or MLP layer. One direction in the
jet-averaged particle embedding carries about two thirds of the effect at every depth, including the raw embedding. However, the same
direction also carries the effect of a pT rescaling, which leaves the axis moment unchanged. (The axis moment is the pT-weighted mean
squared distance of the particles from the jet axis.) Copying this direction's value from a real jet with a different axis moment
does not move the output in proportion to that difference. So the network has a low-dimensional direction that collects mass-related
evidence, but we found no variable equal to the axis moment. A nonlinear or per-particle form of such a variable remains untested.

## Procedure
- Model: ParT_full (released JetClass weights, weaver-core 0.4.17). Replication arm: ParT_kin. Network control: untrained ParT_full
  (`ParTNetwork(untrained=True, seed=2026)`, each weight tensor redrawn from a Gaussian with the trained tensor's mean and s.d.).
- Samples: JetClass test_20M, random rows, `numpy.default_rng(2026)`, never first-k. The build is `part_core.build`
  ($\eta = \eta_{jet} + \mathrm{sign}(\eta_{jet})\,\Delta\eta$).
  - WZ: 10,000 random rows of `WToQQ_101.root` + 10,000 of `ZToQQ_101.root` (20,000 jets). Fit/test halves come from a permutation
    with seed 2026; ridge alpha is chosen on the last quarter of the fit half.
  - T10: 500 random rows (from rows < 50,000) of each class's `<sample>_100.root` (5,000 jets, 10 classes).
  - P1 sample (rows 3,000–4,000 of each class's first test file): unedited-accuracy check only.
- Observables, about the stored jet axis: $m$ = invariant mass; $M_2 = \sum p_T \Delta R^2 / \sum p_T$ (axis moment);
  $\mathrm{mom} = \sqrt{\sum p_T \cdot \sum p_T \Delta R^2}$; girth $= \sum p_T \Delta R / \sum p_T$. "M0" in the task text is
  invariant mass squared, written here as $\log m^2$.
- Edits (all real particles; px, py, pz, E, jet_pt, jet_energy rebuilt):
  - φ + δ (δ = 0.03, 0.06, 0.09): invariant mass unchanged (|Δlog m| < 1e-7), $\Delta\log M_2$ = +0.047 / +0.173 / +0.348.
  - η + 0.06: Δlog m 6e-6, ΔlogM2 +0.145.
  - Angles × 1.1: Δlog m +0.094.
  - Angles × $k_{match}$ with $k_{match} = \exp(\overline{\tfrac12\Delta\log M_2})$ under φ + 0.06 = 1.0902: the matched-calibration
    real-mass edit (same predicted axis-moment change as φ + 0.06; Δlog m +0.085).
  - pT × c (c = 0.8, 1.25): Δlog m ∓0.223, $M_2$ unchanged.
- (a) Residual patching on 6,000 random WZ jets (2,979 W, 3,021 Z), edits φ + 0.06, angles × 1.1, angles × $k_{match}$, with the
  edited 4-vectors. Sites per particle block $l$: the attention sublayer output (post_attn_norm output, before the residual add),
  the MLP output (fc2 output, before the residual add), and the block output. Patches are applied at all particle slots,
  position by position (the edits keep the pT order).
  - Denoise: in the edited run, the site takes its clean value. Restored share $= 1 - \overline{d_P}/\overline{d_E}$.
  - Noise: in the clean run, the site takes its edited value. Carried share $= \overline{d_P}/\overline{d_E}$.
  - Controls: (i) resample, where the clean value comes from a donor jet in the same multiplicity batch, same class, |Δm| < 5 GeV,
    with slots beyond the donor's multiplicity wrapped; (ii) subset, where the clean value is used at a random 50 % of the real
    particles.
  - Cumulative patches: all 8 attention outputs (allA), all 8 MLP outputs (allM), all 16 (allsub), and blocks 1..$l$ (upto $l$).
  - Metric $d$: per-jet change of n98 ($z_{97}$, the W/Z class-token coordinate after the final LayerNorm, in units of its s.d.
    over WZ, 1.681) and of $\ell_W - \ell_Z$; W → Z rate on true W.
  - Statistics: 1,000 bootstrap resamples over jets. H0: share = 0 vs H1: share ≠ 0, two-sided, Holm over 16 sublayers × 2 edits
    (φ + 0.06, angles × 1.1). Second test H0′: share ≤ 0.5.
  - ParT_kin: the same on 4,000 jets (φ + 0.06, angles × 1.1; metric $\ell_W - \ell_Z$), without the resample and subset controls.
- (b) Representation search at sites emb (embedding output), b1..b8 (particle block outputs) and cls1 (class token after class
  block 1).
  - Probes: ridge on the masked particle mean (cls1: the token) for $\log m^2$, $\log M_2$, log mom and log girth. Reported: $R^2$
    on the test half, and the read-out change under each edit vs the true change. Calibration ratio = read-out(φ + 0.06) /
    read-out(angles × $k_{match}$); 1 means the probe reads the two edits the same, 0 means it reads invariant mass only.
  - DAS: an orthonormal $Q \in \mathbb{R}^{128\times k}$ (k = 1, 2, 4) is fitted on the fit half (150 Adam steps × 128 jets,
    lr 0.02). The objective is that the interchanged run reproduce the counterfactual run's 10 logits (MSE). The counterfactual is
    the same jet with the edit applied to the per-particle inputs (4-vectors from the unedited jet, so the whole edit flows through
    the residual stream).
    - Variable "M2": counterfactuals φ + 0.03/0.06/0.09.
    - Variable "M0": counterfactuals pT × 0.8/1.25 (k = 4).
    - Interchange modes: jm $h_p \leftarrow h_p + (\bar h' - \bar h)QQ^\top$ (jet-mean value written to every particle);
      pp $h_p \leftarrow h_p + (h'_p - h_p)QQ^\top$; tok at cls1.
  - Evaluation on 3,000 test-half jets for counterfactuals φ + 0.06, η + 0.06 (held-out edit), angles × $k_{match}$ and pT × 1.25.
    - Logit recovery ("IIA" here) $= 1 - \sum\lVert L_{int}-L_{cf}\rVert^2 / \sum\lVert L_{base}-L_{cf}\rVert^2$.
    - Call IIA = the share of jets whose 10-class call the edit changes that receive the counterfactual call.
    - n98 recovery $= \overline{\Delta z_{97,int}}/\overline{\Delta z_{97,cf}}$.
  - Controls: 3 random orthonormal Q of equal k; DAS on the untrained ParT_full (k = 4); the M0 subspace evaluated on the M2
    counterfactuals. Paired bootstrap over jets; Holm over the 48 trained M2 subspaces.
- (c) Substitution at the earliest particle block whose jm M2 k = 4 φ recovery is ≥ 0.8 × the best block's, using the smallest k
  that reaches ≥ 0.9 × k = 4 there. Result: block 1, k = 1, because recovery is flat across blocks.
  - Setup: 6,000 test-half W/Z jets; 5,634 have a natural donor in the same multiplicity batch with the same class,
    |Δm| < 3 GeV (mean 1.5) and |ΔlogM2| ≥ 0.15 (mean 0.46). The recipient's jet-mean projection is set to the donor's.
  - Variants: DAS Q, 3 random Q, the M0 Q (k = 4), and all 128 dims.
  - Metrics: slope through 0 of Δz97 on $\Delta\log M_2$ (donor − recipient), against the per-jet slope of the φ + 0.06 edit;
    W → Z; McNemar on 10-class correctness.
  - Path patching (P7) uses P3's class-stage re-implementation (Stage; max |Δlogit| vs the model 4.8e-6). A changed particle
    embedding reaches n98 only through: block-1 head h (its context vector from the changed run, everything else from the base
    run); b1h3 + b1h6; all block-1 heads; or block-2 heads directly.
  - 10-class read on T10: a self-donor φ + 0.06 interchange through the block-1, k = 1 subspace, compared with the full edit and a
    random Q.

## Numbers (ParT_full unless stated)
Unedited accuracy:
- P1 sample: ParT_full 85.93 %, ParT_kin 74.47 % (P1: 85.9 / 74.5 %).
- WZ: 10-class 74.82 %, W/Z binary 83.67 %, W → Z 0.102.
- T10: 86.02 %.

Edits on WZ (20,000 jets; Δz97 in s.d.):

| edit | Δz97 | W → Z |
|---|---|---|
| φ + 0.06 | −0.272 | 0.102 → 0.311 |
| η + 0.06 | −0.229 | 0.273 |
| angles × 1.1 | −0.318 | 0.381 |
| angles × 1.090 | −0.293 | 0.350 |
| pT × 1.25 | −0.511 | 0.513 |

**(a) Patching, 6,000 jets.** φ + 0.06: Δz97 −0.271 [−0.280, −0.263], Δ(ℓW − ℓZ) −0.78 [−0.81, −0.75], W → Z 0.102 → 0.307.

| restored share of Δz97 (denoise) | φ + 0.06 | angles × 1.1 | angles × 1.090 |
|---|---|---|---|
| largest single sublayer | MLP b3 0.056 [0.053, 0.059] | attn b1 0.081 [0.078, 0.083] | attn b1 0.084 |
| sum of the 16 single sublayers | 0.39 | 0.56 | 0.59 |
| all attention outputs (allA) | 0.291 [0.280, 0.301] | 0.388 | 0.395 |
| all MLP outputs (allM) | 0.310 [0.299, 0.320] | 0.286 | 0.290 |
| all 16 sublayers (allsub) | 0.418 [0.406, 0.428] | 0.484 [0.476, 0.492] | 0.490 |
| blocks 1..l, l = 1, 2, 3, 4, 5, 6, 7 | 0.084, 0.180, 0.257, 0.305, 0.331, 0.365, 0.389 | 0.096, 0.204, 0.295, 0.345, 0.393, 0.438, 0.456 | similar |
| noise: all 16 sublayers edited in the clean run | 0.325 | 0.530 | 0.524 |
| block output (whole stream), b1 / b8 | 1.000 / 1.000 | 0.873 / 1.000 | 0.881 / 1.000 |

- W → Z restored by allsub: 0.51 (φ), 0.59 (angles × 1.1).
- Holm (32 tests; the smallest attainable Holm p with 1,000 resamples is 0.032): 13/16 sublayers (φ) and 15/16 (angles × 1.1)
  have p_Holm = 0.032 for H0 share = 0. H0′ share ≤ 0.5 is not rejected for any sublayer (one-sided p = 1).
- Controls, attention b1 under φ: resample −0.080 [−0.107, −0.053]; random-half subset 0.028 (all particles 0.048 [0.041, 0.054]).
- ParT_kin (4,000 jets): φ + 0.06 Δ(ℓW − ℓZ) −0.57, W → Z 0.165 → 0.347; allsub restores 0.380 [0.358, 0.402]; blocks 1..1
  −0.001 [−0.022, 0.019]; largest single 0.044 (MLP b8). Angles × 1.1: allsub 0.499 [0.488, 0.510].

**(b) Probes (test half, 10,000 jets).**
- $R^2$: 0.88–0.96 for all four targets at emb–b8; 0.80–0.85 at cls1.
- The invariant-mass probe ($\log m^2$; true change under φ + 0.06 = 0) reads φ + 0.06 as +0.131 (emb), +0.148 (b1), +0.171 (b4),
  +0.150 (b8), +0.149 (cls1). Its calibration ratio φ / matched is 0.87 [0.85, 0.88], 0.86, 0.99 [0.98, 1.01], 0.91, 0.96.
- $\log M_2$ probe ratio: 0.88, 0.84, 1.01, 0.93, 0.84. log mom: 0.89–1.01. Girth: 1.08–1.46.
- Under pT × 1.25, the $\log m^2$ probe reads +0.42 to +0.45 at emb–b4 (true +0.446) and the $\log M_2$ probe −0.002 to +0.09
  (true 0).

**(b) DAS, 3,000 test jets** (logit recovery, 95 % CI):

| site | M2 jm k=1, φ | M2 jm k=4, φ | k=4, η (held out) | k=4, angles × 1.090 | k=4, pT × 1.25 | random k=4, φ | untrained k=4, φ | M0 subspace k=4, φ |
|---|---|---|---|---|---|---|---|---|
| emb | 0.686 [0.669, 0.701] | 0.700 [0.684, 0.715] | 0.453 | 0.653 | 0.552 | 0.055 | 0.020 | 0.452 |
| b1 | 0.596 | 0.628 [0.610, 0.645] | 0.534 | 0.783 | 0.725 | 0.022 | 0.048 | 0.486 |
| b2 | 0.626 | 0.640 | 0.572 | 0.812 | 0.771 | 0.011 | 0.055 | 0.526 |
| b3 | 0.642 | 0.658 | 0.601 | 0.817 | 0.783 | 0.025 | 0.058 | 0.558 |
| b4 | 0.621 | 0.667 | 0.607 | 0.825 | 0.786 | 0.021 | 0.087 | 0.564 |
| b5 | 0.638 | 0.668 | 0.606 | 0.824 | 0.800 | 0.020 | 0.103 | 0.581 |
| b6 | 0.584 | 0.685 | 0.626 | 0.814 | 0.784 | 0.026 | 0.135 | 0.599 |
| b7 | 0.610 | 0.690 | 0.628 | 0.828 | 0.822 | 0.054 | 0.174 | 0.582 |
| b8 | 0.666 | 0.718 [0.706, 0.730] | 0.660 | 0.837 | 0.845 | 0.023 | 0.197 | 0.628 |
| cls1 (tok) | 0.561 | 0.723 [0.711, 0.733] | 0.706 | 0.798 | 0.792 | 0.063 | 0.310 | 0.725 |

- Random 1-dim subspaces: −0.003 to 0.017. All 48 trained M2 subspaces vs random have p_Holm = 0.048 (the bootstrap floor).
  Trained vs untrained has p = 0.001 at every site.
- M2 − M0 subspace on φ (k = 4): +0.248 [0.237, 0.258] (emb), +0.142 (b1), +0.086 to +0.113 (b2–b8), −0.002 [−0.007, 0.003]
  (cls1, p 0.37).
- Call IIA on the jets whose call the φ edit changes (k = 4): emb 0.70, b1 0.56, b4 0.59, b8 0.66, cls1 0.68. n98 recovery: 0.90,
  0.66, 0.72, 0.83, 0.88.
- pp mode k = 4: 0.82–0.92 on φ, 0.80–0.90 on η, 0.81–0.87 on pT × 1.25 (random ≤ 0.05).
- ParT_kin, jm/tok k = 4: φ recovery emb 0.807, b1 0.688, b4 0.692, b8 0.807, cls1 0.719. Held-out η: emb 0.104, b1 0.334,
  b4 0.645, b8 0.787, cls1 0.697. Random ≤ 0.06.

**(c) Substitution** (block 1, k = 1; 5,634 recipients):
- Slope of Δz97 [s.d.] per unit $\Delta\log M_2$:
  - φ + 0.06 edit, same jets: −1.585 [−1.637, −1.536].
  - DAS subspace: −0.007 [−0.025, 0.011]; ratio to φ 0.004 [−0.007, 0.015].
  - Random subspaces: −0.002, +0.003, +0.010. DAS vs random: p 0.28.
  - M0 subspace: +0.357 [0.336, 0.380]. All 128 dims: +0.102 [0.080, 0.124].
- Calls changed: DAS 16.9 %, random 0.6–0.8 %, M0 25.1 %, all 128 dims 21.4 %. W → Z on W recipients whose donor has the larger
  $M_2$ (1,218 jets): 0.101 → 0.137 (DAS), 0.101 (random).
- McNemar on 10-class correctness (DAS):
  - donor $M_2$ larger: right → wrong 260, wrong → right 127, p 1.2e-11;
  - donor $M_2$ smaller: 341 vs 151, p 6e-18;
  - random subspaces: p ≥ 0.18.
- Path shares of the φ + 0.06 Δz97 on the same jets: all block-1 heads 0.991 [0.988, 0.993]; b1h3 0.532 [0.523, 0.543];
  b1h6 0.152 [0.147, 0.157]; b1h4 0.107; b1h2 0.072; b1h3 + b1h6 0.699 [0.691, 0.708]; block-2 heads directly 0.012. The same
  shares for the natural-donor substitution are undefined, because its total slope is ≈ 0 (CIs span −1.8 to 3.9).

**10-class read (T10, 5,000 jets).**
- Accuracy: base 86.02 %, φ + 0.06 83.30 %, DAS self-donor interchange (block 1, k = 1) 84.38 %, random 86.02 %.
- The edit changes 7.84 % of calls. The interchange gives the edit's call on 54.3 % of those (random 0.5 %).
- W → Z: 0.102 → 0.298 (edit), 0.218 (DAS).
- Own-class rate, base / edit / DAS: W 0.808 / 0.598 / 0.688; Z 0.704 / 0.672 / 0.662; QCD 0.774 / 0.730 / 0.766; other classes
  within 0.02.
- McNemar on 10-class correctness: DAS 155 vs 73 (p 5.9e-8); edit 231 vs 95 (p 3.2e-14).

## Reading (scoped to these measurements)
Competing predictions and what the data exclude:
- **Thesis C v2 step 3** (an M2-like quantity forms at an identifiable block 1–3 and is read by the block-1 class heads). It predicts
  (i) a block or sublayer whose clean patch restores > 0.5; (ii) a low-dimensional subspace with recovery well above random, above an
  M0 subspace, and generalising to η; (iii) natural-donor substitution reproducing the φ slope.
  - (i): no single sublayer exceeds 0.084 (H0′ share ≤ 0.5 not rejected for any). Cumulative restoration grows about 0.02–0.10
    per block. Restoring all 16 sublayers gives 0.42 (φ) and 0.48 (real mass edit), so 0.58 / 0.52 of n98's response reaches the
    class stage through the direct residual path of the edited per-particle embedding. ParT_kin: 0.62 / 0.50.
  - (ii): a 1-dim jet-mean subspace reproduces 0.58–0.69 of the φ logit change at every particle site (random ≤ 0.02; untrained
    network at k = 4: 0.02–0.20) and 0.46–0.60 of the held-out η change. It is present at the embedding output and roughly flat across depth, not
    formed at blocks 1–3. The M2 subspace exceeds the pT-trained subspace on φ by 0.09–0.25 at the particle sites and by
    −0.002 at cls1. The M2 subspace also reproduces 0.55–0.85 of the pT × 1.25 change, under which $M_2$ is unchanged.
  - (iii): the ratio to φ is 0.004 [−0.007, 0.015]. The substitution changes 17 % of calls and lowers accuracy whichever way the
    donor's $M_2$ differs.
  - Excluded for this linear jet-mean family: (i) and (iii); (ii) holds only in the non-specific form described.
- **Distributed alternative** (no localizable variable; restoration additive and small per block; IIA near random). The first part
  matches (a). The "IIA near random" part is excluded: recovery 0.6–0.7 vs random ≤ 0.06, p_Holm 0.048 at the bootstrap floor.
- **Third reading consistent with all three measurements:** one direction in the jet-mean particle embedding, already present in
  the input embedding, collects the output-relevant change produced by several different kinematic edits (φ, η, angle scaling,
  pT scaling). Its value across natural jets at fixed invariant mass does not track $M_2$. The probe calibration says the same:
  every linear mass-like probe, including the invariant-mass probe, reads φ + 0.06 as 0.84–1.01 of the matched real mass change
  at every site. So the linear read-out of "mass" in the particle stream is already axis-relative at the embedding.
- **C2b / P7 path, φ + 0.06:** block-1 class heads carry 0.99 of n98's change (b1h3 0.53, b1h6 0.15); block-2 heads directly 0.01.
  This reproduces P3v's split under the mass-fixed edit.

## Issues / caveats
- The site-selection rule chose block 1 and k = 1 because recovery is nearly flat across blocks (0.63–0.72 at k = 4). Substitution
  at the embedding or block 8 was not run.
- Natural-donor path shares are undefined (total slope ≈ 0). The P7 path numbers above come from the φ edit only.
- DAS used 150 Adam steps (loss e.g. emb k = 4: 11.4 → 2.2). Longer fits could raise every recovery, including that of the
  pT-trained subspace. Random and untrained controls used the same budget.
- The "M0" counterfactual (pT × c through the per-particle route only) changes pt_log and e_log. It is a jet-energy-scale
  subspace, not an invariant mass computed from pair features; the pair route is held at the unedited 4-vectors in all DAS
  counterfactuals.
- The jm interchange tests a jet-level variable written equally to every particle. A per-particle (pp) subspace reaches 0.82–0.92
  but is not a jet-level variable. A natural-donor pp substitution is not defined without particle correspondence.
- Bootstrap p-values have a floor of 0.001, so Holm values of 0.032 (a) and 0.048 (b) are floors. The intervals are far from 0.
- T10 rows were drawn from rows < 50,000 of the `_100` files and may overlap the P1 rows (3,000–4,000) used only for the accuracy
  check.

## Explainer entry (for site/explain.py)
**Question.** ParT's W/Z decision reacts to a φ shift that leaves the invariant mass unchanged (C3). Where inside the network does
the per-particle $\Delta R$ input turn into the mass-like quantity that the class token reads (C2b)?

**Background.** ParT has 8 particle blocks. Each block adds an attention output and an MLP output to every particle's 128-dim
residual vector. Two class-attention blocks then read the particles into one class token, whose coordinate $z_{97}$ ("n98")
separates W from Z. The axis moment is $M_2 = \sum_i p_{T,i}\,\Delta R_i^2 / \sum_i p_{T,i}$, where $\Delta R_i$ is the particle's
distance from the jet axis. A rigid shift $\Delta\phi_i \to \Delta\phi_i + 0.06$ leaves the invariant mass $m$ unchanged but raises
$M_2$ by about $0.06^2$.

**Method.**
1. Patching: run the shifted jet, put back the unshifted value of one sublayer's output (or of all sublayers in blocks $1..l$), and
   measure the restored share $1 - \overline{\Delta z_{97}^{patched}}/\overline{\Delta z_{97}^{edit}}$.
2. Distributed alignment search: learn an orthonormal $Q\in\mathbb{R}^{128\times k}$ such that writing only the shifted jet's value
   of $Q^\top \bar h$ (the jet-mean residual) into the unshifted jet, $h_p \leftarrow h_p + (\bar h' - \bar h)QQ^\top$, reproduces
   the shifted jet's logits. Score: $1 - \sum\lVert L_{int}-L_{cf}\rVert^2/\sum\lVert L_{base}-L_{cf}\rVert^2$, compared with random
   $Q$, an untrained network, and a $Q$ trained on pT rescaling (which changes $m$ but not $M_2$).
3. Substitution: copy $Q^\top\bar h$ from a real jet with the same class and mass but a different $M_2$, and compare the slope
   $\mathrm{d}\Delta z_{97}/\mathrm{d}\Delta\log M_2$ with the φ shift's slope.

**Result (figure `results/part/p8_mass_formation.png`).**
- Left panel: no single sublayer restores more than 6 % of $\Delta z_{97}$ under the φ shift. Restoring all 16 sublayers restores
  42 %, so 58 % travels from the shifted input embedding straight to the class token.
- Right panel: a 1-dim jet-mean direction reproduces 56–69 % of the logit change at every depth, from the embedding to class block 1
  (random directions ≤ 6 %, untrained network 2–31 %). It also reproduces most of the pT-rescaling change.
- Substituting the direction's value from real jets gives a slope of $-0.007$ against $-1.585$ for the φ shift.

**Checks.**
- Unedited accuracy 85.93 % on the P1 sample.
- The class-stage re-implementation matches the model to 5e-6.
- The held-out η shift is reproduced at 45–71 %.
- ParT_kin shows the same pattern: whole-stream restoration 0.38, DAS recovery 0.69–0.81.
- Path patching gives block-1 class heads 0.99 (b1h3 0.53) of the φ response.

**Why it matters.** The thesis that ParT forms an $M_2$ variable at an early block is not supported in this linear, jet-mean form.
The mass evidence is carried by one output-aligned direction present from the input embedding onward, and most of it enters the
class token without processing by the particle blocks. The ParT-specific mechanism is therefore in the class-attention read-out of
per-particle $\Delta R$ embeddings (C2b), not in an intermediate $M_2$ variable.
