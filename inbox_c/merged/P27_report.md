# P27: per-particle kernel of the axis response in ParT and ParticleNet (A8, C12, P12; exploratory) (agent report, 2026-10-20)

Job `anrunw-ai-c-p27-run2` (RTX 2080 Ti, 400 s, smoke stage then full run): `python -m interp.p27_particle_kernel --smoke && python -m interp.p27_particle_kernel`.
A first launch (`c-p27-run`) was stopped after its smoke stage to add the inward move (per-particle odd/even split); no result from it is used.
Analysis re-run as CPU job `c-p27-ana` (1 CPU, 2 Gi, 76 s; `AI_PY=conda ./job.sh c-p27-ana 'env JETDISTILL_TAGGER=part python -m interp.p27_particle_kernel --analyse'`) to add binned-mean R², the azimuth-shared component, the (rad − az) target and normal-approximation p-values for Holm.
- Script: `interp/p27_particle_kernel.py`.
- Results: `results/part/p27_particle_kernel.json`, `results/part/p27_particle_kernel_info.json`; per-edit logits in `results/part/p27_particle_kernel.npz` (17 MB, not for commit; also on the volume).
- Figure: `results/part/p27_particle_kernel.png`.
- Logs: `logs/c-p27-run2.log`, `logs/c-p27-ana.log`.

## Conclusion
We moved single particles of W and Z jets 0.01 rad away from the jet axis, one at a time, and measured how much the W-vs-Z score changed. On average the change grows with the particle's contribution to the axis moment M2 = Σ pT r² (r = distance from the axis): binned by that contribution, the mean responses of all four trained models follow it closely (R² about 0.9). But particle by particle, no simple formula predicts the change: M2, JEDI-style pT-rank windows, and pT-share or radius thresholds each explain only 2–8 % of the variance on held-out jets, and the rank-window and threshold forms do no better than M2. Much of the remaining per-particle response also appears when the same particle is moved around the axis at fixed r, which leaves M2 unchanged. What that component reads is still open; our guess is the particle's distances to its neighbours, but this run does not test that. Pair moves add up (additivity about 0.95), and the single-particle responses sum to the response for moving all particles together.

## Procedure
**Models.**
- ParT_full and ParT_kin (released JetClass weights).
- ParticleNet_full and ParticleNet_kin (released weights; network code `interp/pn_net.py`).
- Two controls:
  - an untrained ParT_full architecture (weaver initialisation, torch seed 2701);
  - ParT_kin with the output rows of every weight tensor permuted (one permutation per layer, seed 2701; P18 recipe, 48 tensors).

**Sample.**
- JetClass test_20M `WToQQ_106.root` and `ZToQQ_106.root`, 2,000 random rows each (numpy `default_rng(2701)`, sorted indices, not first-k). N = 4,000 jets; none was dropped (all had ≥ 2 eligible particles).
- Eligible particles: pT share z_i = pT_i / Σ pT > 0.5 %. That gives 97,370 particles, 24.3 per jet on average.

**Unedited accuracy** (10-class argmax, rebuilt base jets):
| model | this sample (W+Z) | W | Z | P1 sample (N = 10,000) |
|---|---|---|---|---|
| ParT_full | 75.1 % | 80.8 % | 69.3 % | 85.93 % (known 85.9 %) |
| ParT_kin | 62.5 % | 69.0 % | 56.1 % | 74.83 % (known ≈ 74.5 %) |
| PN_full | 73.0 % | 78.9 % | 67.0 % | 84.57 % |
| PN_kin | 59.2 % | 65.6 % | 52.8 % | 70.34 % |
| ParT untrained | 17.1 % | 23.7 % | 10.4 % | 7.46 % |
| ParT_kin permuted | 49.9 % | 99.8 % | 0.0 % | 10.03 % |

The permuted model calls almost every W/Z jet W.

**Build.** We used `part_core.build`. Each edit changes the stored (deta, dphi) of the moved particle(s) and keeps their pT. From those, build recomputes:
- the particle's η and φ (η = jet_eta + sign(jet_eta)·deta);
- px, py, pz and E;
- deltaR, deta and dphi;
- jet_pt and jet_energy, and hence logptrel and logerel;
- the pair features (from the four-vectors).

Because pT is kept, the pT ordering does not change.

**Edits** (dr = 0.01 rad; 335,841 forward passes per model):
- `rad_i`: particle i moved radially outward by dr.
- `radin_i`: particle i moved inward by dr. When r_i < dr it passes through the axis.
- `az_i` (control a): particle i rotated about the axis at fixed r_i. The chord length is dr and the sign is random per particle.
- `pair_ij`: six random eligible pairs per jet, both moved outward by dr.
- `spread` (control b): five random eligible particles each moved outward by dr/5. Two draws per jet.
- `all`: every eligible particle moved outward by dr together (calibration).
- `base2`: the unedited jet in another batch (numerical floor).

**Read-out.** d = change of (logit_Z − logit_W) relative to the rebuilt unedited jet. Positive means Z-ward.

**Per-particle targets.**
- d_i: the outward response.
- Odd part (d_out − d_in)/2, regressed on x_odd = (x_out − x_in)/2.
- Even part (d_out + d_in)/2.
- d_rad − d_az.

**Forms.** All forms use global coefficients with no intercept; k is the number of parameters. x is the exact ΔM2/M2 of the edit, about 2 pT_i r_i dr / M2.

| form | definition | k |
|---|---|---|
| M2 | x | 1 |
| M2_ctx | x·(1, s, s², s³), with s = base (logit_Z − logit_W)/5, clipped to ±2 | 4 |
| M2_rankwin | x per pT-rank window (1, 2, 3, 4–6, 7–10, 11–20, 21+) | 7 |
| rankwin_zr | (z, r, z·r) per window | 21 |
| hinge_z | (z − t)₊ and (z − t)₊·r, t ∈ {0.5, 1, 2, 5, 10, 20} % | 12 |
| hinge_r | z·1[r > t] and 1[r > t], t ∈ {0.05, 0.1, 0.2, 0.3}: the derivative of JEDI's Σ(r − t)₊ objects | 8 |
| pt_only | (z, z², √z) | 3 |
| r_only | (r, r², √r) | 3 |
| const | 1 | 1 |
| gbdt | gradient-boosted trees on (z, r, rank, x, M0, M2, m_jet, multiplicity, s); flexible benchmark | – |

**Metrics.**
- Held-out R²: 5-fold split over jets, with SST about the pooled mean.
- R² of binned means: 20 quantile bins of the held-out prediction.
- Per-jet-gain shape R²: gain fitted on even-rank particles and scored on odd-rank particles.
- BIC from the full-data fit.

**Statistics.**
- 95 % bootstrap over jets (1,000 resamples).
- H0 R²(form) = R²(M2): bootstrap normal-approximation p, Holm-corrected over 108 tests (4 trained models × 3 targets × 9 forms). The percentile-bootstrap p has a floor of 0.001, so its Holm minimum is 0.108.
- Azimuthal odd part: sign-flip permutation (2,000).

## Numbers
**Response sizes** (95 % CI). Columns: mean d_rad; mean odd part; mean even part; even share of per-particle variance; mean |d_az| / mean |d_rad|; mean signed s_i·d_az (perm p).

| model | d_rad | odd | even | even share | \|az\|/\|rad\| | s·d_az (perm p) |
|---|---|---|---|---|---|---|
| ParT_full | 0.027 [0.025, 0.029] | 0.025 [0.024, 0.026] | 0.002 [0.001, 0.003] | 0.47 [0.43, 0.52] | 0.70 [0.69, 0.72] | 0.000 [−0.001, 0.001] (0.99) |
| ParT_kin | 0.024 [0.022, 0.025] | 0.021 [0.020, 0.022] | 0.003 [0.002, 0.003] | 0.45 [0.42, 0.48] | 0.73 [0.71, 0.74] | 0.000 [−0.001, 0.001] (0.69) |
| PN_full | 0.022 [0.020, 0.024] | 0.024 [0.023, 0.025] | −0.002 [−0.003, −0.000] | 0.52 [0.49, 0.55] | 0.81 [0.80, 0.83] | 0.000 [−0.001, 0.001] (0.44) |
| PN_kin | 0.020 [0.019, 0.021] | 0.019 [0.018, 0.019] | 0.002 [0.001, 0.002] | 0.24 [0.23, 0.25] | 0.71 [0.70, 0.72] | −0.000 [−0.000, 0.000] (0.90) |
| ParT untrained | −0.001 | −0.001 | 0.000 | 0.08 | 0.29 | – (0.73) |
| ParT_kin perm | −0.002 [−0.003, −0.001] | −0.002 | −0.000 | 0.36 | 0.77 | – (0.25) |

- Mean |d_rad|: 0.067 (ParT_full), 0.054 (ParT_kin), 0.077 (PN_full), 0.041 (PN_kin).
- Numerical floor (|d_base2|) ≤ 1.1e-6.
- Correlation of |d_rad| with |d_az| for the same particle: 0.49, 0.57, 0.55, 0.44. Correlation of the even part with d_az: 0.52, 0.55, 0.56, 0.43.

**Held-out R²** (5-fold over jets) for the odd part, with Δ vs M2 [95 % CI] and Holm p.

| model | M2 | M2_ctx | M2_rankwin | rankwin_zr | hinge_z | hinge_r | pt_only | gbdt |
|---|---|---|---|---|---|---|---|---|
| ParT_full | 0.030 [0.024, 0.036] | 0.036 (+0.005 [0.002, 0.008], p 0.006) | 0.032 (+0.002, p 0.32) | 0.031 (+0.001, p 1) | 0.031 (+0.001, p 1) | 0.029 (−0.002, p 1) | 0.023 (−0.007, p 0.004) | 0.064 (+0.034 [0.027, 0.042]) |
| ParT_kin | 0.040 [0.033, 0.048] | 0.055 (+0.015, p 3e-10) | 0.043 (+0.003, p 0.03) | 0.046 (+0.006, p 0.62) | 0.045 (+0.005, p 1) | 0.037 (−0.003, p 1) | 0.033 (−0.007, p 0.17) | 0.110 (+0.070) |
| PN_full | 0.031 [0.025, 0.037] | 0.038 (+0.008, p 4e-6) | 0.033 (+0.002, p 0.03) | 0.031 (−0.000, p 1) | 0.031 (+0.000, p 1) | 0.026 (−0.005, p 0.09) | 0.023 (−0.008, p 3e-5) | 0.067 (+0.036) |
| PN_kin | 0.068 [0.058, 0.079] | 0.117 (+0.048 [0.039, 0.056], p 4e-25) | 0.080 (+0.012, p 5e-12) | 0.080 (+0.011, p 0.09) | 0.078 (+0.010, p 0.23) | 0.067 (−0.001, p 1) | 0.052 (−0.016, p 7e-4) | 0.223 (+0.155) |

- The outward response d_i gives the same ordering, with M2 R² of 0.019, 0.035, 0.018 and 0.078. The largest Δ for a rank-window or threshold form is +0.008 (PN_kin hinge_z).
- For d_rad − d_az, the M2 R² is 0.014, 0.020, 0.012 and 0.041.
- Controls: in ParT untrained, M2 gives −0.15 and r_only gives 0.11. In ParT_kin permuted, every form gives |R²| < 0.003.
- R² of binned means (outward response, M2 form): 0.93, 0.91, 0.95, 0.91 for the trained models. Forms that contain z all give 0.92–1.00 (pt_only 0.98–1.00). The untrained ParT gives −3.2.
- Per-jet-gain shape R² of the M2 form: −0.14, −0.21, −0.08, +0.04. A per-jet gain fitted on about 12 particles predicts the other 12 worse than the pooled mean.
- BIC orders the forms as follows:
  - ParT_full, ParT_kin and PN_full: M2_ctx < M2_rankwin ≈ M2 ≈ hinge_z < rankwin_zr < pt_only < hinge_r.
  - PN_kin: M2_ctx lowest, then M2_rankwin, rankwin_zr and hinge_z.

**Additivity and calibration** (95 % CI).

| model | additivity A | interaction share | Σd_pair / Σ(d_i+d_j) | Σ_i d_i / d_all | per-jet R² (Σd_i vs d_all) | spread / linear |
|---|---|---|---|---|---|---|
| ParT_full | 0.947 [0.900, 0.978] | 0.051 [0.021, 0.096] | 0.989 [0.974, 1.003] | 1.003 [0.951, 1.063] | −0.27 | 0.95 [0.88, 1.02] |
| ParT_kin | 0.968 [0.957, 0.977] | 0.030 [0.021, 0.040] | 0.992 [0.983, 1.002] | 1.062 [1.021, 1.103] | 0.32 | 0.85 [0.66, 1.02] |
| PN_full | 0.939 [0.917, 0.959] | 0.059 [0.040, 0.081] | 1.005 [0.983, 1.028] | 0.888 [0.825, 0.958] | −1.39 | 1.02 [0.93, 1.13] |
| PN_kin | 0.956 [0.951, 0.960] | 0.038 [0.035, 0.042] | 1.005 [0.997, 1.013] | 1.017 [0.989, 1.047] | 0.53 | 0.98 [0.95, 1.00] |
| ParT untrained | 1.000 | 0.000 | 1.000 | 0.989 | 1.00 | 1.04 |
| ParT_kin perm | 0.891 | 0.109 | 0.90 [−0.27, 2.11] | 1.45 [0.55, 2.51] | −1.46 | – |

The mean response to moving all eligible particles outward together (d_all) is 0.65, 0.54, 0.61 and 0.49 for the four trained models.

## Reading
**Pre-registered hypotheses.**
- **H0** (M2 held-out R² ≥ 0.6, within 0.05 of the best form, additivity ≥ 0.8):
  - The R² criterion is not met: 0.02–0.08 in all four trained models.
  - The within-0.05 criterion is met for every low-parameter form except M2_ctx in PN_kin (+0.048 on the odd part, at the limit).
  - Additivity is 0.94–0.97.
- **H1** (rank-window or threshold forms beat M2 by > 0.1): not observed. The largest Δ is +0.012 (PN_kin M2_rankwin, odd part), and hinge_r is at or below M2 in every model.
- **H2** (no low-parameter form reaches R² 0.3): observed in all four models.
  - The flexible tree model on the same per-particle and jet variables also stays at 0.04–0.22.
  - So most of the per-particle variance is not a function of (z, r, rank, M0, M2, m_jet, multiplicity, base score).

**Mean versus particle-level response.**
- The mean response tracks ΔM2/M2: binned-mean R² is about 0.9, and the outward response is mostly odd in the move direction.
- The scatter around that mean is as large as the mean (|d| about 2.5× mean d).
- About half of that scatter is even in the move direction (out and in give the same sign; even share 0.45–0.52 in ParT_full, ParT_kin and PN_full; 0.24 in PN_kin).
- The scatter correlates with the response to the azimuthal move of the same particle, which leaves r and M2 unchanged.
- The azimuthal move has zero signed mean in every model, so there is no direction-dependent per-particle term.
- The expected ParticleNet-only azimuthal signature (from P23's odd share) does not appear as a signed per-particle response. PN_full's |az|/|rad| (0.81) is larger than ParT_full's (0.70).

**Additivity and calibration.**
- Pairs add (A 0.94–0.97), and Σ_i d_i matches d_all on average: 1.00 for ParT_full, 0.89–1.06 across the four trained models.
- Jet by jet the match is poor (per-jet R² −1.4 to 0.5). The particle-specific components that do not depend on r add up in Σ_i d_i but are smaller in the whole-jet move.

**Controls.**
- The untrained ParT has an r-dependent but M2-independent kernel (r_only best; M2 negative) and near-perfect additivity, with responses about 50× smaller than the trained models.
- The permuted ParT_kin has responses of trained size (mean |d| 0.083) with no structure in any form (R² ≈ 0).

**Scope.**
- One step size (0.01 rad), W and Z jets only, one file pair (_106).
- The 'local neighbour' reading of the scatter is not tested here: no pair-distance-held edit was run.
- The per-jet-gain metric is limited by about 12 particles per half-jet.

## Explainer entry (for site/explain.py)
```python
P27_EXPLAINER = [
 ('Question', 'At the jet level ParT reads the axis moment $M_2=\\sum p_T r^2$ about the stored axis (claim C12). Does each particle enter with the weight $M_2$ predicts, $\\partial M_2/\\partial r_i = 2p_{T,i}r_i$, or with JEDI-linear-style weights (pT-rank windows, thresholds on pT share or radius)?', None),
 ('Background',
  'Moving particle $i$ outward by $\\delta r$ changes $M_2$ by $\\Delta M_2 = p_{T,i}[(r_i+\\delta r)^2-r_i^2]\\approx 2p_{T,i}r_i\\delta r$. A network that reads only $M_2$ responds to each particle in proportion to $x_i=\\Delta M_2/M_2$, with one gain per jet. JEDI-linear (Paper A) instead builds its mass from sums over pT-rank windows and hinges such as $\\sum (r-0.2)_+$, which would give rank- or threshold-shaped weights.', None),
 ('Method',
  '4,000 random W and Z jets (JetClass _106 files). Each particle with pT share above 0.5 % is moved by 0.01 rad, one at a time: outward, inward, and around the axis at fixed $r$ (which leaves $M_2$ unchanged). Pairs of particles and all particles together are also moved. Read-out: change $d$ of $\\ell_Z-\\ell_W$. '
  'Models: ParT_full/kin, ParticleNet_full/kin, an untrained ParT and a weight-permuted ParT_kin. Per-particle formulas (M2, rank windows, pT-share and radius hinges, pT-only, r-only, and a boosted-tree benchmark) are compared by $R^2$ on held-out jets (5-fold).', None),
 ('Result',
  'On average the response follows $x_i$: binned means give $R^2\\approx0.9$ in all four trained models (panel b). Particle by particle, no formula predicts it: held-out $R^2$ is 0.02–0.08 for M2 and the same within 0.012 for rank-window and threshold forms; a boosted tree on pT, $r$, rank and jet variables reaches only 0.04–0.22 (panel a). '
  'The moving-around-the-axis control gives 70–81 % of the radial response size with zero mean, and its size per particle correlates with the radial response (0.44–0.57).', 'results/part/p27_particle_kernel.png'),
 ('Checks',
  'Pair moves add (additivity 0.94–0.97) and the single-particle responses sum to the whole-jet move on average (ratio 0.89–1.06). The untrained ParT has a 50× smaller, $r$-only kernel; the weight-permuted ParT_kin has responses of the same size with no structure in any formula. Numerical floor below $10^{-6}$.', None),
 ('Why it matters',
  'The big models\' axis response is $M_2$-like in aggregate but is not carried by a per-particle weight of the JEDI type (no rank-window or threshold form beats $M_2$; claim A8 not supported at the per-particle level). About half of each particle\'s response does not depend on its radius and is not explained by the variables tested; whether it reads the particle\'s distances to its neighbours (pair features) is open.', None),
]
```
Conclusion (for site/conclusions.py): the text of the Conclusion section above.
