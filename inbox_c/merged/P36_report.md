# P36: on-manifold axis test: natural off-centre jets in ParT, ParticleNet and L-GATr (claims C3, D4; JEDI E23 analogue) (agent report, 2026-10-20)

Status: **done**. Exploratory node.

**Jobs**
- `anrunw-ai-c-p36-run` (Tesla V100, 480 s): `python -m interp.p36_natural_offset --stage nets`. Runs ParT_full, ParT_kin, PN_full, PN_kin and untrained ParT.
- `anrunw-ai-c-p36-lg2` (RTX 2080 Ti, 1,396 s, of which about 1,100 s were python imports from CephFS): `env-lgatr/bin/python -m interp.p36_natural_offset --stage lgatr`. Runs L-GATr.
- `anrunw-ai-c-p36-an` (job.sh, 1 CPU): `--stage analyze`. Statistics and figure.
- Smoke runs: `c-p36-smoke` (nets on 150 jets per class) and `c-p36-ansm` (analysis on the smoke output).
- `part_job.sh` was run through a `sed` copy that adds `k8s-haosu-09.sdsc.optiputer.net` to the excluded nodes. That node refused the GPU twice with UnexpectedAdmissionError, "GPU is lost". `part_job.sh` itself was not edited.

**Files**
- Script: `interp/p36_natural_offset.py`.
- Results: `results/part/p36_natural_offset.json` and `results/part/p36_nets_meta.json`. Per-jet logits and probes are in `/j-jepa-vol/aaron-interp/results/part/p36_nets.npz` and `/j-jepa-vol/aaron-interp/results/lgatr/p36_lgatr.npz` (volume only).
- Figure: `results/part/p36_natural_offset.png`.
- Logs: `logs/c-p36-{run,lg2,an,smoke,lg}.log`.

## Conclusion
In real jets the pT-weighted centre of the particles is close to the stored jet axis. Even in the 1 % most off-centre jets of each class it is only 0.03–0.10 rad away. We moved every particle of such a jet by the same small angle so that this centre lands on the axis. That leaves the jet's mass unchanged. ParT changes its W-vs-Z score by about as much as it does for a real mass change of the same size in the axis moment: the ratio is 1.3 (95 % interval 0.5–2.3) on the 5 % most off-centre W and Z jets. So the network has not learned to ignore offsets that occur naturally. Because the natural offsets are small, the effect corresponds to a mass shift of only about 1 %, and changes 2.5 % of ParT_full's calls on those jets. The Lorentz-equivariant L-GATr responds about 4 times less, and ParticleNet about as much as ParT. Still uncertain: the per-tail ratios have wide intervals (40–200 W/Z jets). A rotation of the jet about its axis was meant as the null control, but it changes more calls than recentring in every model, so it cannot serve as a null here.

## Question and hypotheses
All earlier C3 numbers (P1, P2, P23) come from artificial rigid shifts of the constituents. Here the question is whether the axis dependence matters for jets as they occur.

**The offset.** For each jet, $\vec c = \sum_i p_{T,i}\,(\Delta\eta_i, \Delta\phi_i) / \sum_i p_{T,i}$ is the pT centroid of the constituents relative to the stored jet axis.

**The moment.** The axis moment is $M_2 = \sum_i p_{T,i} r_i^2$, with $r_i$ the particle's distance from the stored axis. With $M_0 = \sum_i p_{T,i}$ and $M_{2,c}$ the same moment about the centroid, $M_2 = M_{2,c} + |\vec c|^2 M_0$.

**Prediction of the thesis.** If ParT's mass read-out is axis-relative, a jet with a large natural $|\vec c|$ carries an extra $|\vec c|^2 M_0$ in $M_2$. Recentring removes it, and the read-out should move as for a real mass change with the same change of $\tfrac12\log M_2$.

**The hypotheses.**
- H1, axis-relative thesis: the ratio $R_{nat}$ (defined under Procedure) is near the rigid-shift value for each model.
  - The P23 gains predict, to first order, $R_{rec} = 2g_{odd} - g_{even}$: 0.70 for ParT_full, −0.46 for ParT_kin, 0.77 for PN_full and 1.27 for PN_kin.
  - For L-GATr the prediction is about 0 (the phi part of the edit is an exact symmetry for it).
- H_alt, on-manifold invariance: the networks have learned to discount natural offsets. Then $R_{nat}$ is much smaller than the R of a random rigid shift of the same length on the same jets.
- H0 for each response: no change (mean read-out change = 0).

## Procedure
**Models** (all on the same jets):
- Released ParT_full and ParT_kin.
- Released ParticleNet_full and ParticleNet_kin (PN).
- Untrained ParT_full (weaver initialisation, torch seed 1), as the zero reference.
- L-GATr: our own run `anrunw-ai-c-p5-lgatr2`, latest checkpoint at job start: `runs/jc2m/jc2m_s0a/models/model_run0_it139999.pt` (70 % of the 200k-iteration schedule).
  - L-GATr reads lab four-momenta and computes its angular features about the particle-sum axis it recomputes.
  - So a rigid phi translation is an exact symmetry for it, and an eta translation at fixed pT is close to a longitudinal boost, which is not an exact symmetry.

**Sample.**
- JetClass test_20M `<class>_118.root`, 2,000 random rows per class (numpy `default_rng(3636)`, drawn class by class; no first-k), N = 20,000. No jet had fewer than 2 particles.
- Inputs are rebuilt with `part_core.build` ($\eta = \eta_{jet} + \mathrm{sign}(\eta_{jet})\,\Delta\eta$).
- L-GATr was run on a subset of 3,278 jets: all top-5 % jets, their matched low-offset partners, and 2,000 random jets.

**Selection.**
- Tails: per class, the top 1 / 2 / 5 % of $|\vec c|$ (N = 200 / 400 / 1,000).
- Matched low set: for each top-5 % jet, the nearest same-class jet with $|\vec c|$ below the class median. Distance is measured in standardised (log m, log pT_jet, N particles), weighted 10 : 3 : 1, with replacement (414 distinct jets).

**Edits.** Each edit acts on the stored $(\Delta\eta, \Delta\phi)$ of every real particle. pT and particle masses are kept, and four-vectors and per-particle features are rebuilt.

| edit | what changes in the jet | $M_2$ after | what it tests |
|---|---|---|---|
| rec (recentre) | every particle shifted by $-\vec c$ | $M_2 - \lvert\vec c\rvert^2 M_0$ | the natural offset removed |
| ref (reflect) | every particle shifted by $+\vec c$ (offset doubled) | $M_2 + 3\lvert\vec c\rvert^2 M_0$ | same offset direction, other sign |
| rs (random shift) | shift by $\lvert\vec c\rvert$ in a uniformly random direction | $M_2 + 2\lvert\vec c\rvert^2 M_0\cos\alpha + \lvert\vec c\rvert^2 M_0$ | same-size artificial shift (P1/P23 type) |
| rot (rotation) | $(\Delta\eta, \Delta\phi)$ rotated about the stored axis by $\theta \sim U(60°, 300°)$ | unchanged; $\lvert\vec c\rvert$ unchanged | moves particles, keeps $M_2$ and pair distances |
| dil_rec / dil_ref / dil_rs | all angles scaled by $k = \sqrt{M_2^{edit}/M_2}$ | same as rec / ref / rs | matched real mass change |

**Checks on the edits** (400 random jets for pair distances, all jets for mass):
- rec, ref, rs and rot leave every pairwise $\Delta R^2$ unchanged:
  - max change 9e-7, 9e-7, 1.1e-6 and 1.5e-6;
  - 0 of 725,767 pairs change by more than 1e-4.
- Invariant mass:
  - Relative change under rec: median 5e-6, 99th percentile 9e-5, max 1.6e-3. The η part at fixed pT is not exact for massive particles.
  - Under rot: median 0.5 %, max 30 %. Rotating in the (η, φ) plane keeps $\Delta R$ but not $\cosh\Delta\eta - \cos\Delta\phi$, so rot is not a fixed-mass edit.
- dil_rec changes the mass by $\Delta\log m$ = −0.0067 on tail5, against a predicted $\tfrac12\Delta\log M_2$ of −0.0068.

**Read-outs.**
1. W-vs-Z log-odds $\ell_W - \ell_Z$ (difference of the two logits), on the true W and Z jets of each set. This is the primary read-out, as in P23.
2. Class-representation mass probe, for ParT and PN.
   - It is a ridge regression of log m on the input of the final linear layer (ParT class token, 128 dimensions; PN pooled EdgeConv features, 256 dimensions).
   - It is fitted on unedited jets outside tail5 (14,250 fit / 4,750 validation).
   - Held-out R² of log m: ParT_full 0.89, ParT_kin 0.93, PN_full 0.91, PN_kin 0.94, untrained 0.74.

**Calibration.**
- $\kappa$ is the slope through 0 of the read-out change under the three matched dilations against their predicted $\tfrac12\Delta\log M_2$, pooled on the same jets (P23 method).
- $R(e) = \overline{d(e)} / (\kappa\,\overline{\tfrac12\Delta\log M_2(e)})$, where $d(e)$ is the read-out change under edit $e$.
- **R_nat = R(rec)** is the network's response to recentring divided by its response to a real mass change of the same predicted size. R = 1 means the same response as the real change; R = 0 means no response.
- The direct ratio $\overline{d(rec)}/\overline{d(dil\_rec)}$ is reported alongside. It is noisier because the dil_rec change is small.

**Statistics.**
- 95 % bootstrap over jets: 2,000 resamples (500 for the full 20,000-jet set), with numerator and κ resampled together.
- Exact McNemar test on per-jet call changes, rec vs rot.
- Sign-flip permutation tests (5,000), for H0 mean d(rec) = 0 and for H0 mean d(rec) = mean d(rot).
- Bootstrap p-values for R(rec) − R(rs) and for R(rec) − R(rec) of the untrained ParT. The latter uses the probe, since the untrained network's W/Z κ is about 0. It is paired on the same jets and resamples.
- Holm correction over the models × 3 tails within each family:
  - McNemar: m = 15;
  - permutation of rec vs rot: m = 15;
  - comparison with the untrained network: m = 12.

**Observational pairs.** For each tail W/Z jet and its matched low-offset partner, the unedited difference in $\ell_W - \ell_Z$ (tail − partner) is compared with the edit prediction −d(rec).

## Numbers
**Unedited accuracy.**
- P1 sample (rows 3,000–4,000 of each _100 file, N = 10,000): ParT_full 85.93 % (known 85.9), ParT_kin 74.47 % (known 74.5), PN_full 84.56 %, PN_kin 70.36 % (P4: 84.57 / 70.35), untrained 12.76 %.
- L-GATr it139999: 83.93 % on rows 3,000–3,300 (N = 3,000).
- This sample (N = 20,000): ParT_full 86.1 %, ParT_kin 74.1 %, PN_full 84.2 %, PN_kin 70.0 %, untrained 12.7 %, L-GATr 84.2 % (2,000 random jets).

**Size of the natural offsets.**

| set | N | median $\lvert\vec c\rvert$ [rad] (range) | median $\lvert\vec c\rvert^2 M_0/M_2$ | predicted $\tfrac12\Delta\log M_2$ under rec | median m [GeV] |
|---|---|---|---|---|---|
| all | 20,000 | 0.0077 | 0.0021 | −0.0017 | 115 |
| tail5 | 1,000 | 0.030 (0.014–0.10) | 0.012 | −0.0068 | 158 |
| tail2 | 400 | 0.039 (0.020–0.10) | 0.015 | −0.0086 | 196 |
| tail1 | 200 | 0.051 (0.028–0.10) | 0.019 | −0.0101 | 227 |
| low matched | 414 | 0.0041 | 0.0003 | −0.0002 | 139 |

For comparison, the P1 shift of 0.06 rad raises $\tfrac12\log M_2$ by about 0.03–0.1 on typical jets.

**R_nat on the W/Z log-odds** (true W and Z jets; [95 % CI]).

| model | set (W/Z jets) | κ | R(rec) = R_nat | R(rs) | R(ref) | P23 prediction for R(rec) | H0 d(rec)=0, perm p | R(rec) − R(rs), p |
|---|---|---|---|---|---|---|---|---|
| ParT_full | tail5 (200) | −6.6 [−9.2, −4.0] | **1.33 [0.54, 2.25]** | 1.69 [0.57, 3.09] | 1.10 [0.80, 1.52] | 0.70 | 0.0028 | 0.63 |
| ParT_full | tail2 (80) | −5.3 | 1.41 [−0.16, 3.72] | 1.67 | 1.03 [0.58, 1.83] | | 0.090 | 0.83 |
| ParT_full | tail1 (40) | −5.8 | 1.26 [−0.80, 3.52] | 1.55 | 1.00 [0.54, 1.96] | | 0.22 | 0.87 |
| ParT_full | all (4,000) | −9.7 [−11.0, −8.6] | 1.08 [0.83, 1.31] | 1.03 [0.77, 1.33] | 0.98 [0.87, 1.09] | | 0.002 | 0.82 |
| ParT_kin | tail5 (200) | −4.0 [−7.4, −0.9] | 0.39 [−1.79, 9.51] | 1.23 | −0.06 | −0.46 | 0.89 | 0.58 |
| ParT_kin | all (4,000) | −6.2 | 0.72 [−0.62, 2.75] | −0.45 | 0.20 | | 0.30 | 0.32 |
| PN_full | tail5 (200) | −4.7 | 1.01 [−0.74, 3.06] | −0.15 | 0.77 [0.22, 1.32] | 0.77 | 0.25 | 0.42 |
| PN_full | all (4,000) | −8.6 | 0.92 [0.60, 1.30] | 0.77 [0.46, 1.11] | 0.83 [0.72, 0.95] | | 0.002 | 0.56 |
| PN_kin | tail5 (200) | −3.9 | 0.48 [−0.79, 1.79] | 0.50 | 1.00 [0.53, 1.59] | 1.27 | 0.45 | 1.0 |
| PN_kin | all (4,000) | −7.1 | 0.73 [0.51, 0.95] | 0.90 [0.64, 1.17] | 0.85 [0.75, 0.96] | | 0.002 | 0.32 |
| L-GATr | tail5 (200) | −4.8 [−6.8, −2.9] | **0.31 [0.11, 0.65]** | −0.08 [−0.64, 0.24] | 0.10 [0.00, 0.26] | ≈ 0 | 0.0012 | 0.075 |
| L-GATr | all (407) | −9.4 | 0.22 [0.06, 0.44] | −0.18 | 0.04 | | 0.003 | 0.044 |
| untrained ParT | tail5 (200) | −0.12 [−0.24, +0.04] | (κ ≈ 0; undefined) | | | | | |

Mean change of $\ell_W - \ell_Z$ on tail5 W/Z jets, recentring vs matched real change:
- ParT_full: +0.038 [0.014, 0.063] vs +0.031.
- L-GATr: +0.0064 [0.0026, 0.0099] vs +0.027.

**Class-representation mass probe** (all classes; tail5, N = 1,000).

| model | probe κ | R(rec) | R(rs) | untrained R(rec) on the same jets | trained − untrained, p (Holm) |
|---|---|---|---|---|---|
| ParT_full | 0.23 [0.13, 0.33] | 3.16 [1.50, 6.45] | 3.56 [2.05, 6.16] | 4.77 [4.39, 5.17] | −1.61 [−3.41, 1.74], p 0.28 (1.0) |
| ParT_kin | 0.39 | −2.58 [−3.67, −1.66] | +3.68 [2.67, 4.99] | 4.77 | −7.35, p 0.0005 (0.0075) |
| PN_full | 0.65 | 1.44 [0.67, 2.20] | 0.32 [−0.43, 1.08] | 4.77 | −3.33, p 0.0005 (0.0075) |
| PN_kin | 1.07 | 0.48 [0.04, 0.92] | 0.82 [0.44, 1.22] | 4.77 | −4.30, p 0.0005 (0.0075) |

On all 20,000 jets: ParT_full 1.81 [1.55, 2.14], ParT_kin −1.72 [−1.99, −1.50], PN_full 0.84 [0.67, 1.02], PN_kin 0.10 [−0.04, 0.24], untrained 4.86 [4.72, 5.02]. In tail1 and tail2 the ParT probe κ is consistent with 0 (−0.02 to 0.14), so those probe ratios are undefined.

**10-class calls changed [%], tail5 (N = 1,000)** [95 % CI].

| model | acc. | rec | rs | dil_rec | ref | rot | McNemar rec vs rot (rec-only / rot-only), p_Holm |
|---|---|---|---|---|---|---|---|
| ParT_full | 78.7 | 2.5 [1.6, 3.6] | 2.6 [1.7, 3.7] | 1.1 [0.5, 1.8] | 2.9 | 10.2 [8.4, 12.1] | 13 / 90, 3e-14 |
| ParT_kin | 63.5 | 7.3 [5.8, 8.9] | 7.8 | 2.4 | 6.8 | 19.7 | 32 / 156, 1e-19 |
| PN_full | 76.9 | 3.3 [2.2, 4.5] | 3.3 | 1.0 | 4.3 | 15.7 | 21 / 145, 8e-23 |
| PN_kin | 57.5 | 5.6 [4.2, 7.0] | 7.2 | 1.1 | 7.6 | 10.8 | 32 / 84, 9e-6 |
| L-GATr | 74.5 | 0.3 [0.0, 0.7] | 0.5 | 1.1 [0.5, 1.8] | 0.7 | 20.1 [17.7, 22.7] | 2 / 200, 9e-56 |
| untrained ParT | 11.3 | 2.8 | 2.3 | 0.2 | 2.5 | 10.3 | 14 / 89, 2e-13 |

Other sets:
- tail1, ParT_full: rec 2.5 [0.5, 5.0], rot 15.5 %. ParT_kin: rec 11.0, rot 30.0 %.
- Low matched set (median $|\vec c|$ 0.004 rad): rec 0.5 % (ParT_full), 1.7 % (ParT_kin), 0.0 % (PN_full), 0.0 % (L-GATr).
- All jets: rec 0.5 % (ParT_full), 2.9 % (ParT_kin), 0.9 % (PN_full), 1.6 % (PN_kin), 0.1 % (L-GATr).

W→Z and Z→W flows on tail5 (100 W and 100 Z jets; before → after recentring):
- ParT_full: W→Z 14 → 13 %, Z→W 9 → 7 %.
- ParT_kin: W→Z 14 → 13 %, Z→W 5 → 5 %.
- PN_full: W→Z 13 → 11 %, Z→W 11 → 12 %.
- L-GATr: W→Z 12 → 12 %, Z→W 11 → 11 %.

**Recentring vs rotation, permutation of per-jet d(rec) − d(rot) on the W/Z log-odds.** No model and tail passes Holm (smallest p_Holm 0.23).

**Observational pairs** (unedited W/Z tail jet minus its matched low-offset partner, $\ell_W - \ell_Z$):

| model | tail5 (200 pairs) | edit prediction −d(rec) |
|---|---|---|
| ParT_full | +0.05 [−0.44, 0.61] | −0.04 [−0.06, −0.01] |
| ParT_kin | +0.03 [−0.23, 0.32] | −0.01 |
| PN_full | +0.58 [0.13, 1.03] | −0.02 |
| L-GATr | −0.08 [−0.51, 0.40] | −0.01 |

Despite the matching, the pairs still differ in mass: mean Δlog m = +0.024 (tail5) and +0.075 (tail1), and $\tfrac12\Delta\log M_2$ = 0.043. Their offsets are 0.023 vs 0.003 rad.

**Example jet** (figure, left and centre).
- A true Z→qq jet: m = 215.8 GeV, pT = 526 GeV, $|\vec c|$ = 0.069 rad (top 1 % of its class).
- Recentring changes $M_2/M_0$ from 0.1558 to 0.1511 at unchanged invariant mass and pair distances.
- ParT_full: P(Z) 0.62 → 0.48 and P(W) 0.12 → 0.26.

## Reading (scoped to these measurements)
- **ParT_full.** Natural offsets are not discounted. On the 5 % most off-centre W/Z jets, recentring moves the W/Z log-odds as a real mass change of the same $\tfrac12\Delta\log M_2$ does: R_nat 1.33 [0.54, 2.25]. It is not different from a random rigid shift of the same length on the same jets (R 1.69, difference p 0.63). On all 4,000 W/Z jets R_nat is 1.08 [0.83, 1.31] against 1.03 for the random shift.
  - H_alt (on-manifold invariance, R_nat ≪ R_rs) is not supported for ParT_full, PN_full or PN_kin.
  - H0 (no response) is excluded for ParT_full on tail5 (p 0.003) and on all jets.
  - R_nat is somewhat above the first-order P23 prediction (0.70). The P23 interval for $2g_o - g_e$ and this one overlap.
- **ParT_kin.** On the W/Z log-odds R_nat is undetermined (CI −1.8 to 9.5). On the probe, recentring raises the probed mass (R −2.6) while a random shift of the same length gives +3.7. This sign is what P23's ParT_kin gains predict: no odd (direction-dependent) term, so recentring acts only through the $+|\vec c|^2 M_0$ even part. Recentring changes 7.3 % of its calls on tail5, 3× the matched real change (2.4 %), so ParT_kin's calls also move through something other than the M2 read-out.
- **ParticleNet.**
  - PN_full: R_nat 0.92 [0.60, 1.30] on all W/Z jets, and 1.0 on tail5 with a wide interval.
  - PN_kin: R_nat 0.73 [0.51, 0.95] on all W/Z jets.
  - Their responses are of the same order as ParT_full's, not ½–⅔ of it as the plan expected.
- **L-GATr.** R_nat is 0.31 [0.11, 0.65] on tail5 and 0.22 [0.06, 0.44] on all W/Z jets, about ¼ of ParT_full. It changes 0.3 % of calls on tail5, below its 1.1 % under the matched real change.
  - The φ part of the recentring is an exact symmetry for L-GATr, so the non-zero response comes from the η part (not exactly a boost: its energy-based features and time reference change). This is consistent with P5 (Δη + 0.06 changed 0.5 % of calls).
  - L-GATr also responds in the opposite sense to the random shift (R(rs) −0.08), so its R_nat is not an axis-moment response.
- **Size of the effect.** The thesis prediction is satisfied in sign and size relative to the matched change, but the matched change itself is small.
  - Natural offsets in the top 5 % change $\tfrac12\log M_2$ by −0.007 on average, about a 0.7 % mass-equivalent shift.
  - So recentring changes only 2.5 % of ParT_full's calls there (W→Z 14 → 13 %), and 0.5 % over all jets.
  - The plan's expectation of "several % calls changed, well above rotation" is not met as stated: rotation changes more.
- **Rotation is not a null.** Rotating a jet about its stored axis by 60–300° changes 10–20 % of calls in every trained model, including L-GATr (20 %).
  - For L-GATr this is expected: the beam direction breaks the symmetry, and a rotation about the jet axis changes the jet's orientation relative to the beam.
  - Rotation also changes the invariant mass (median 0.5 %, up to 30 %).
  - The McNemar tests (rot ≫ rec) therefore measure the networks' sensitivity to jet orientation, not a floor for the axis effect.
  - The informative controls are the random shift of equal length (same $|\vec c|$, different direction) and the matched dilation.
- **Untrained ParT.** Its W/Z κ is about 0 (−0.12), so R is undefined on the log-odds. Its class-token probe still responds to recentring (R 4.8): a random-feature representation passes the axis-relative inputs (Δη, Δφ, ΔR) straight to a probe trained on mass. The trained networks' probe R_nat is below the untrained value (p_Holm 0.0075 for ParT_kin, PN_full and PN_kin; ParT_full p 0.28). The probe is therefore a weaker read-out here than the logits.
- **Observational pairs** do not resolve the edit-sized effect (±0.5 against a predicted −0.04). Matching on mass leaves a residual Δlog m of +0.02 to +0.08, and other jet-to-jet differences dominate. The one significant pair difference (PN_full +0.58) has the sign opposite to the axis prediction and is not interpreted.

## Issues
- **L-GATr on CPU.** A first L-GATr run on CPU (`c-p36-lg`, job.sh, deleted after the P1 check) gave chance accuracy, 10.0 % on 3,000 P1 jets.
  - On the GPU the dense ("materialised") attention-mask path of the authors' wrapper gives the same accuracy as xformers (83.93 %; max |Δlogit| 0.23).
  - So the CPU failure is specific to the CPU kernel path; not investigated further. L-GATr evaluation must run on GPU.
  - The GPU was idle for about 18 min during the CephFS imports (as in P5).
- **Checkpoint.** The L-GATr checkpoint is interim: it 139,999 of 200,000 (`anrunw-ai-c-p5-lgatr2` still running). Rerun the L-GATr stage on the final checkpoint in P6.
- **Small W/Z tails.** Per-class tails mean only 40 / 80 / 200 W/Z jets in tail1 / 2 / 5, so W/Z-log-odds ratios in tail1 and tail2 have intervals that include 0. A larger W/Z sample (e.g. 20,000 per class for W and Z only) would narrow them.
- **Rotation control.** The rotation control as specified (random angle) is not a null for any of these networks; see Reading.
- **Probe ratios.** Probe ratios in tail1 and tail2 for ParT are undefined: the probe κ is about 0 there (the probe shrinks toward the mean in the heavy tails).

## Explainer entry (for site/explain.py, Paper C)
**Question.** ParT's mass sense follows the axis moment $M_2$ rather than the invariant mass when the axis is moved artificially (C3). Does that matter for jets as they occur, whose particles are naturally off-centre?

**Background.**
- The pT centroid of a jet's particles, $\vec c = \sum_i p_{T,i}\vec r_i / \sum_i p_{T,i}$ (with $\vec r_i = (\Delta\eta_i, \Delta\phi_i)$ from the stored axis), is not exactly on the stored (four-momentum-sum) axis.
- About the stored axis, $M_2 = \sum_i p_{T,i} r_i^2 = M_{2,c} + |\vec c|^2 M_0$.
- A network that reads $M_2$ about the stored axis therefore sees an off-centre jet as slightly heavier than the same jet recentred, although the two have the same invariant mass $m^2 = (\sum_i p_i)^2$.
- "On-manifold" means jets of the kind that occur in data, as opposed to artificially shifted ones.

**Method.**
- From 20,000 JetClass test jets, take the 1 / 2 / 5 % with the largest $|\vec c|$ in each class.
- Recentre them: move every particle by $-\vec c$. All pair distances and the invariant mass stay the same, and $M_2$ falls by $|\vec c|^2 M_0$.
- Compare the change of the W-vs-Z log-odds with the change from a matched real mass change: all angles scaled by $k = \sqrt{M_2^{rec}/M_2}$, giving the same $\tfrac12\Delta\log M_2$.
- $R_{nat}$ is the ratio of the two responses (1 = the same response as a real mass change, 0 = no response).
- Controls: a shift of the same length in a random direction, doubling the offset, rotating the jet about its axis, an untrained ParT, and the Lorentz-equivariant L-GATr.

**Result.**
- Natural offsets are small: median 0.03 rad in the top 5 %, a −0.7 % change of $\sqrt{M_2}$.
- ParT_full responds to recentring as to a real mass change of that size: $R_{nat}$ = 1.3 [0.5, 2.3] on the top-5 % W/Z jets and 1.1 [0.8, 1.3] on all W/Z jets. A random shift of the same length gives the same.
- ParticleNet: 0.7–0.9. L-GATr: 0.2–0.3.
- Recentring changes 2.5 % of ParT_full's calls on these jets.
- Figure: a Z jet with $|\vec c| = 0.069$ rad, before and after recentring; ParT_full's P(Z) goes 0.62 → 0.48.

**Checks.**
- Pair distances unchanged to $10^{-6}$; invariant mass unchanged to $10^{-4}$ (99th percentile).
- Unedited accuracy reproduced: 85.93 % / 74.47 %.
- The response is zero for H0 at p = 0.003 (ParT_full, top 5 %).

**Why it matters.** The axis dependence of ParT's mass read-out is not an artefact of off-manifold edits: the network applies it to real jets. Because natural offsets are small, it shifts the effective mass by about 1 % and moves a few % of calls in the most off-centre jets. L-GATr's symmetry removes most of it.

**Glossary additions.**
- **pT centroid $\vec c$**: pT-weighted mean $(\Delta\eta, \Delta\phi)$ of a jet's particles relative to the stored jet axis.
- **on-manifold**: about jets as they occur in the data distribution (not artificially edited ones).
- **recentring**: rigid translation of all particles by $-\vec c$, so the pT centroid sits on the axis; invariant mass and pair distances unchanged.
- **matched dilation**: scaling all angles about the axis by $k$, chosen so that $\tfrac12\Delta\log M_2$ equals that of the edit being calibrated; a real mass change.
- **R_nat**: the network's read-out change under recentring divided by its change under the matched dilation, via the pooled calibration slope κ. 1 = responds as to a real mass change; 0 = ignores it.
- **κ (calibration slope)**: read-out change per unit $\tfrac12\Delta\log M_2$ under matched dilations.
- **R(rs)**: the same ratio for a rigid shift of length $|\vec c|$ in a random direction.
