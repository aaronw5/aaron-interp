# H50: prong momentum fraction z vs prong separation ΔR, D2 and reclustered prong masses (claim D18; proposed as H47)

Status: done (one run, not yet verified). Script `interp/h50_z_vs_dr.py`. Results `results/physics/h50_z_vs_dr.json` (also
`$AI_RESULTS/physics/h50_z_vs_dr.json`), metadata `results/physics/h50_{jc,hls}_meta.json`, model accuracies
`results/physics/h50_jc_models.json`, `h50_hls_models_{small,jedi}.json`. Figures: `results/physics/h50_edit.png` (one W jet under arms A, B and C),
`results/physics/h50_response.png` (W-jet response per arm, networks and references). Cluster jobs: d-h50-smoke, d-h50-smoke2,
d-h50-gsmoke, d-h50-ssmoke (smoke runs, 40 jets per class), d-h50-jc (CPU 4/8: JetClass edits, observables, references), d-h50-hls (CPU 4/8:
hls4ml edits, references, small nets, JEDI-linear), d-h50-g (GPU: JetClass networks), d-h50-s (statistics and figures). Per-jet arrays stay on the volume:
`/j-jepa-vol/aaron-interp/results/physics/h50_{jc,hls}_{inputs,obs,logits*}.npz`.

## Conclusion

H45 changed a jet's momentum sharing between its two prongs at fixed total four-momentum. That edit also changes the masses of the
prongs as the clustering algorithm re-finds them, by several GeV. When those prong masses are put back to their original values, every
tagger still responds by the same amount: about 1.0–1.1 times the original response for asymmetric W and Z jets, in both datasets. So
the response is not caused by the prong masses. A second edit changes only the momentum sharing z and keeps the prong separation and
both prong masses fixed. It moves the JetClass taggers in the direction mass-drop logic predicts: more balanced prongs make a jet look
more W/Z-like, and more asymmetric prongs less so. A reference classifier that never sees z but does see D2 and τ21 predicts the same
size. This is because changing z at fixed separation changes those shape variables too. The open question is therefore narrower: do
the networks read z itself or the energy-correlation shape (D2, τ21) that follows from it? Separation alone (z fixed) cannot explain
the response.

Terms: z = vector-pT fraction of the softer of the two prongs (prongs = the two exclusive-kT subjets of all constituents); ΔR =
rapidity–azimuth distance between the two prong four-vectors; reclustered prong masses m_hi / m_lo = masses of the heavier / lighter
prong after re-running the kT-2 clustering on the edited jet; D2 = e3/e2³ (energy correlation ratio, β = 1); τ21 = τ2/τ1
(N-subjettiness ratio); log-odds = log p(A)/p(B) from the class scores.

## Question and hypotheses

At fixed jet mass m and pT, m² ≈ m_a²/z + m_b²/(1−z) + z(1−z) pT² ΔR², so z and ΔR cannot vary independently at fixed four-momentum
(H45's edit D moves both). H45v also found that D moves the reclustered prong masses (−3.5 to +6.6 GeV). Three hypotheses:
- H_prongmass: the H45 response is caused by the reclustered prong-mass change. Prediction: response collapses (ratio ≈ 0) when
  the prong masses are restored (arm A).
- H0_dR: the networks read ΔR (and mass), not z. Prediction: arm C (z changed at fixed ΔR and prong masses) gives ≈ 0, as for the
  reference MLP0+dR on a z-only edit, and arm B (ΔR at fixed z) gives a response.
- H1_z: z is read. Prediction: arm C ≠ 0 with the sign of the z(1−z) balance (z up → W/Z score up on W/Z-vs-QCD scores), matching
  MLP0+z.

## Procedure

Samples (random rows, never first-k):
- JetClass test WToQQ_101, ZToQQ_101, ZJetsToNuNu_101, stored 500 < pT < 1000 GeV, |η| < 2, ≥ 3 particles, 3,000 random rows per class
  (rng 5001 / 5002 / 5003). Median jet pT 588 GeV.
- hls4ml `jedi_core.sample(per_class=2000, seed=5002)` (10,000 jets; W, Z, q, g analysed, tops only in the 5-class accuracy).

Every edit keeps each constituent's mass and its array slot (charge, PID and track features stay with the particle). It works in
(pT, rapidity y, azimuth φ) per constituent. A rigid translation of one prong in (y, φ) is a boost along the beam plus a rotation
about it, so it leaves that prong's mass and every constituent pT exactly unchanged. Prong labels = kT-2 subjets of the unedited jet.

Arm A: z and ΔR move together, prong masses fixed.
- D(Δc), Δc = ±0.3 (H45 edit): in the jet rest frame, rotate the decay axis so |cos θ*| changes by Δc. The jet four-momentum stays
  exactly fixed. Δc = +0.3 makes the prongs more asymmetric (z down, ΔR up).
- A(Δc): apply D, then recluster (kT-2). Restore each reclustered prong's mass to the mass of the original prong it shares most pT
  with: scale the particle offsets from the prong axis in (y, φ) by s_k (bisection), then translate the prong back onto its axis.
  Then move both prongs rigidly along their joining line (separation scale λ, bisection) so the jet mass is the original one.
  Recluster again and repeat (≤ 4 rounds) until the particles that change prong carry < 1 % of the jet pT. The code asserts that
  prong masses and jet mass equal their targets to 10⁻⁶ of the jet energy (float64).
- Achieved (JetClass W, Δc = +0.3): D gives Δz −0.088, ΔR +0.059, Δm_hi +5.5 GeV, Δm_lo −1.0 GeV, ΔD2 +1.53. A gives Δz −0.096,
  ΔR +0.067, Δm_hi +0.74 GeV, Δm_lo −0.27 GeV, ΔD2 +1.54, Δm 0.00, ΔpT 0.0. At Δc = −0.3, A gives Δm_hi −0.04 (D: −3.4) GeV.
  Accepted 7,342 / 5,560 of 9,000 JetClass jets (Δc = −0.3 / +0.3) and 6,669 / 5,945 of 8,000 hls4ml jets. The rest failed a
  bisection bracket or did not converge.
- Control cφ: rest-frame rotation by the same angle about the jet direction. This keeps z and ΔR. It is the H45 control, and it
  removes any generic response to rotating particles.
- Collapse ratio r_A = (A − cφ)/(D − cφ) on rows accepted for A, D and cφ.

Arm B: ΔR changes at fixed z and fixed prong masses (the jet mass changes).
- B(λ), λ = 0.85, 1.15: translate both prongs rigidly along their joining line about the pT-weighted centre, so ΔR → λΔR exactly.
  z and both prong masses stay exactly fixed. Achieved on JetClass W, λ = 1.15: Δm +9.6 GeV, ΔpT −0.8 GeV, ΔD2 −0.12.
- Control MBs (mass-only edit with the same Δm): scale every constituent pT by a common f (bisection to the jet mass of B) at fixed
  y, φ, m. z and ΔR stay fixed. Jet pT changes by +69 GeV (JetClass W) or +127 GeV (hls4ml W), and the prong masses by +3.0 / +1.1
  GeV. Why this control: at fixed z, ΔR and prong masses, the jet mass can only change through pT. MBs is therefore the only edit
  that changes m alone in the variables tested here. Its side effect is a pT change.
- Secondary control MBp: scale the offsets in both prongs by a common s (bisection to m_B). ΔR and z stay fixed, but the prong masses
  change by +12 / +3 GeV. This can only raise the mass, so it was built for λ = 1.15 only. It is accepted for about 30 % of jets.

Arm C: z changes at fixed ΔR and fixed prong masses (the jet mass changes).
- C(Δz), Δz = ±0.08: target z1 = clip(z + Δz, 0.03, 0.5). Scale the constituent pT of the softer prong by z1·W/w_s and of the harder
  prong by (1−z1)·W/w_h, where w = prong vector pT and W = w_s + w_h, so the jet scalar pT is kept. Restore each prong's mass to
  its original value by offset scaling (bisection), then translate each prong back onto its original axis. ΔR is exact; the prong
  masses are asserted to 10⁻⁶ of E. Achieved on JetClass W, Δz = +0.08: Δz +0.073, ΔR 0.000, Δm_hi −0.10 GeV, Δm +8.0 GeV,
  ΔD2 −0.54, Δτ21 −0.080.
- Control BC: the arm-B edit with λ solved so that the jet mass equals that of C. Same Δm, same z, ΔR changes instead (+0.037).
- Control MBsC / MBpC: as MBs / MBp, matched to the mass of C.

Acceptance (every variant except D and cφ): when the edited jet is reclustered, the particles that change prong carry < 1 % of the
jet scalar pT, and both prong centres lie within ΔR 0.75 of the jet axis. Requiring an identical partition would have kept only 30–50 %
of jets for B(0.85) and C(−0.08), because soft particles switch prong easily. Contrasts use the rows accepted for both members. Accepted
counts (JetClass, of 9,000): B(1.15) 8,835; B(0.85) 4,692; C(+0.08) 7,950; C(−0.08) 4,673; MBs ≥ 8,978; MBp(1.15) 3,384.

Models: JetClass ParT_full, ParT_kin, PN_full, PN_kin (released weights); nulls = 3 ParT_full copies with every weight tensor
permuted (torch seeds 25–27). hls4ml: JEDI-linear official n64 3-feature, dsets-d64-n64, jedis-d64-n64, tf-2b-d32-n64 (seed0); nulls
= 2 JEDI with redrawn weights (seeds 20, 21) and 2 dsets with permuted weights (seeds 27, 28). Every edited jet is rebuilt about its own
E-scheme axis.

References (trained only on unedited train jets, then scored on the edited jets): MLP(64,64) classifiers on the listed observables.
- MLP0: (m, pT, N).
- MLP0+dR, MLP0+pm (m_hi, m_lo), MLP0+D2, MLP0+z.
- MLPx: the H45 reference (m, pT, τ21, D2, N, ΔR); also MLPx+z and MLPx+pm.
- GBDT: HistGradientBoosting on all of these plus τ32, z_g and m_SD.
Two retrains per MLP and one GBDT. Train rows: JetClass train_100M *_002 files, 10,000 per class (rng 5016–5018); hls4ml train archive
without the dev split, 10,000 per class (rng 5019). Held-out AUC on JetClass W-vs-QCD: MLP0 0.89, MLP0+z 0.91, MLP0+dR 0.93, MLPx
0.946, GBDT 0.950.

Metric: Δs = s(variant) − s(unedited) per jet, with s = log-odds.
- JetClass scores: WQ = L_W − L_QCD, ZQ, WZ = L_W − L_Z.
- hls4ml scores: WQG = L_W − logsumexp(L_q, L_g), ZQG, WZ.
- Net contrast = mean of Δs(edit) − Δs(control), with a 95 % bootstrap CI over jets (2,000 resamples) and a one-sample t-test p.
- Holm correction over the 5 primary contrasts (A−cφ, A−D, B−MBs, C−MBsC, C−BC) × networks × (class, score) cells × 2 signs = 560
  tests.
- Ratio network / reference: bootstrap over jets jointly. It is reported only when |reference| ≥ 0.1.
- McNemar test on the calls (threshold = median unedited score of the signal class, signal efficiency 0.5), edit vs control.

## Unedited accuracy on these samples

| model | accuracy | known (H45v sample) |
|---|---|---|
| ParT_full / ParT_kin / PN_full / PN_kin (10-class, on W/Z/QCD jets) | 0.758 / 0.666 / 0.736 / 0.637 | 0.751 / 0.655 / 0.729 / 0.616 |
| JEDI-linear / dsets / jedis / tf (5-class) | 0.813 / 0.810 / 0.799 / 0.767 | 0.809 / 0.807 / 0.797 / 0.762 (test set) |
| permuted ParT ×3; redrawn JEDI ×2; permuted dsets ×2 | 0.008–0.115; 0.200; 0.121–0.208 | null |

## Numbers

Arm A collapse ratio r_A = (A − cφ)/(D − cφ), 95 % CI. A value of 1 means restoring the prong masses changes nothing; 0 means the
response disappears.

| cell | Δc | ParT_full | ParT_kin | PN_full | PN_kin | MLP0+pm (prong masses only) | MLPx |
|---|---|---|---|---|---|---|---|
| JetClass W, W-vs-QCD | +0.3 | 1.08 [1.05, 1.11] | 1.14 | 1.03 | 1.14 | 0.16 [0.12, 0.20] | 1.03 |
| JetClass Z, Z-vs-QCD | +0.3 | 1.06 [1.03, 1.09] | 1.07 | 1.01 | 0.99 | 0.18 | 1.00 |
| JetClass QCD, W-vs-QCD | −0.3 | 0.95 [0.90, 1.00] | 0.90 | 1.01 | 0.87 | 0.11 | 0.96 |
| JetClass W, W-vs-QCD | −0.3 | −0.44 [−0.80, −0.16] | 0.66 | 0.41 | 0.72 | 0.05 | 0.92 |

| cell | Δc | JEDI | dsets | jedis | tf | MLP0+pm | MLPx |
|---|---|---|---|---|---|---|---|
| hls W, W-vs-q/g | +0.3 | 1.00 [0.97, 1.03] | 1.00 | 0.99 | 0.96 | 0.25 | 1.14 |
| hls Z, Z-vs-q/g | +0.3 | 1.01 [0.99, 1.04] | 0.98 | 0.92 | 0.96 | – | 1.18 |
| hls W, W-vs-q/g | −0.3 | 0.77 [0.70, 0.85] | 0.46 | 0.68 | 0.84 | – | 0.77 |

The absolute values behind the first row (JetClass W, W-vs-QCD, Δc = +0.3, n = 2,156):
- ParT_full: D − cφ = −2.55, A − cφ = −2.76 [−2.87, −2.65]. A − D = −0.21 [−0.28, −0.13], Holm p < 10⁻¹⁰.
- References: MLPx −1.30, MLP0+dR −0.66, MLP0+z −0.47, MLP0+pm −0.05, GBDT −1.40. Null band −0.22 to −0.01.
- Calls at ε_S = 0.5: 54.3 % pass under cφ → 23.1 % under A (McNemar p = 2·10⁻¹⁶²).
- On the A − D contrast, Holm < 0.05 in 70 of 112 cells. Most are |A − D| ≤ 0.3. The largest are on JetClass QCD at Δc = +0.3:
  A − D = +0.40 to +0.54. There arm A undoes D's −0.16 to −0.33 response; the r_A CI includes 0 (ParT_full −0.14 [−0.46, 0.08]).

Arm B: ΔR × λ at fixed z and prong masses, minus the mass-matched pT scaling (B − MBs). Δ log-odds, 95 % CI.

| cell | λ | ParT_full | ParT_kin | PN_full | PN_kin | MLP0+dR | MLPx | GBDT | null |
|---|---|---|---|---|---|---|---|---|---|
| JetClass W, W-vs-QCD | 1.15 | −0.42 [−0.45, −0.40] | −0.24 | −0.39 | −0.19 | −0.11 | −0.12 | −0.11 | −0.05…−0.01 |
| JetClass W, W-vs-QCD | 0.85 | −0.69 [−0.72, −0.66] | −0.37 | −0.56 | −0.35 | +0.01 | −0.24 | −0.23 | 0.00…+0.03 |
| JetClass Z, Z-vs-QCD | 1.15 | −0.47 | −0.21 | −0.43 | −0.15 | −0.07 | −0.10 | −0.03 | −0.03…−0.01 |
| JetClass QCD, W-vs-QCD | 1.15 | +0.04 | +0.02 | +0.06 | +0.04 | +0.02 | −0.02 | +0.06 | −0.01…0.00 |

For ParT_full on JetClass W (λ = 1.15), B alone gives Δs = −1.04 and MBs alone −0.62; for MLPx the two are −0.41 and −0.32.

| cell | λ | JEDI | dsets | jedis | tf | MLPx |
|---|---|---|---|---|---|---|
| hls W, W-vs-q/g | 1.15 | +0.15 [+0.09, +0.20] | −1.66 | −3.56 | −5.05 | +0.06 |

For tf, B alone gives −4.82 and MBs alone +0.24. The small hls4ml nets respond differently to the two ways of adding about 10 GeV
of mass, while JEDI and the references do not. Because MBs also raises pT by about 12 %, the hls4ml B − MBs numbers combine a ΔR
effect with a pT effect.

Arm C: z changed at fixed ΔR and prong masses, minus a control with the same jet-mass change (C − MBsC, the pT-scaling control; C −
BC, the ΔR edit). Δ log-odds, 95 % CI.

| cell | Δz | contrast | ParT_full | ParT_kin | PN_full | PN_kin | MLP0+z | MLP0+dR | MLPx | GBDT |
|---|---|---|---|---|---|---|---|---|---|---|
| JetClass W, W-vs-QCD | +0.08 | C − MBsC | +0.23 [+0.19, +0.26] | +0.26 | +0.30 | +0.25 | +0.31 | +0.22 | +0.25 | +0.38 |
| JetClass W, W-vs-QCD | −0.08 | C − MBsC | −0.62 [−0.67, −0.57] | −0.97 | −0.65 | −0.93 | −0.37 | −0.21 | −0.80 | −0.82 |
| JetClass W, W-vs-QCD | +0.08 | C − BC | +0.50 [+0.46, +0.55] | +0.44 | +0.57 | +0.43 | +0.29 | +0.33 | +0.41 | +0.47 |
| JetClass QCD, W-vs-QCD | +0.08 | C − MBsC | +0.43 [+0.39, +0.48] | +0.43 | +0.46 | +0.45 | +0.26 | +0.26 | +0.32 | +0.44 |
| JetClass Z, Z-vs-QCD | −0.08 | C − MBsC | −0.61 | −0.98 | −0.54 | −0.96 | −0.27 | −0.21 | −0.65 | −0.75 |

The MLPx and MLP0+z columns show reference seed 0 for the ratios below and the seed mean in the table. Ratios, JetClass W
W-vs-QCD:
- Δz = +0.08: ParT_full / MLPx = 0.99 [0.86, 1.11]; ParT_full / MLP0+z = 0.82.
- Δz = −0.08: ParT_full / MLPx = 0.78 [0.73, 0.84]; ParT_full / MLP0+z = 1.96; ParT_kin / MLPx = 1.22.
- Calls at ε_S = 0.5 (W, Δz = −0.08, ParT_full): 49.8 % pass under MBsC → 43.2 % under C (McNemar p = 2·10⁻²³).

| cell | Δz | contrast | JEDI | dsets | jedis | tf | MLP0+z | MLP0+dR | MLPx | GBDT |
|---|---|---|---|---|---|---|---|---|---|---|
| hls W, W-vs-q/g | +0.08 | C − BC | +0.21 [+0.17, +0.26] | +0.13 | +0.14 | −0.10 | +0.48 | +0.59 | +0.45 | +0.10 |
| hls W, W-vs-q/g | −0.08 | C − BC | −0.11 [−0.16, −0.05] | −0.24 | −0.42 | −0.41 | −0.05 | +0.15 | +0.04 | −0.30 |
| hls q, W-vs-q/g | +0.08 | C − BC | +0.10 | +0.18 | +0.30 | +0.35 | +0.33 | +0.51 | +0.56 | +0.17 |
| hls W, W-vs-q/g | +0.08 | C − MBsC | +0.45 | −0.43 | −1.11 | −2.59 | +0.33 | +0.39 | +0.29 | +0.10 |

What the edits did to the observables. Arm C on JetClass W jets changes D2 by +0.93 (Δz = −0.08) and −0.54 (Δz = +0.08), and
τ21 by +0.095 / −0.080. The matched ΔR edit BC changes D2 by −0.08 and τ21 by −0.024. At fixed ΔR and prong masses, D2 and τ21
therefore move with z, because e2 ∝ z(1−z)ΔR to leading order. This is why MLPx, which never sees z, responds to arm C.

Holm (560 tests): adjusted p < 0.05 for A−cφ in 85/112 cells, A−D 70/112, B−MBs 96/112, C−MBsC 92/112, C−BC 90/112. Null-model
bands: |mean| ≤ 0.22 in the JetClass cells shown. In hls4ml W/Z cells the bands reach 1.2, because the redrawn JEDI outputs are
nearly constant and the permuted dsets output large values.

W-vs-Z pattern with prong masses restored (A − cφ, Δc = +0.3, asymmetric prongs, score log p(W)/p(Z)):

| | W jets | Z jets |
|---|---|---|
| JetClass ParT_full / PN_full | −1.62 / −1.33 | +0.27 / +0.11 |
| hls4ml JEDI / dsets | −1.60 / −2.14 | +1.89 / +1.63 |

The H45v dataset difference persists after the prong masses are restored. In hls4ml, asymmetric W jets move toward Z and asymmetric Z
jets toward W. In JetClass, asymmetric W jets move toward Z while Z jets move little. MLPx predicts W −0.24 / Z −0.02 (JetClass) and
W −1.61 / Z +0.47 (hls4ml).

## Reading (scoped to what was measured)

- H_prongmass (the H45 response is the reclustered prong-mass change): excluded for asymmetric edits on W/Z jets in both datasets
  (r_A = 0.92–1.14 for all 8 networks; MLP0+pm r_A = 0.16–0.25). For balanced edits (Δc = −0.3) on JetClass W jets, part of
  ParT_full's small D response (+0.27) is tied to the prong masses (A − cφ = −0.10).
- H0_dR (networks read ΔR, not z; arm C ≈ 0): excluded for the JetClass networks. Over the JetClass W, Z and QCD cells with W/Z-vs-QCD scores (4 networks × 4
  cells × 2 signs = 32), the contrast passes Holm and has the sign of the z(1−z) balance (z up → W/Z score up) in 29/32 cells for
  C − MBsC and 31/32 for C − BC. The exceptions are QCD and Z jets at Δz = −0.08 for ParT_full (|mean| ≤ 0.10). On hls4ml,
  C − BC is non-zero but smaller than every reference except GBDT (JEDI +0.21 vs MLP0+z +0.48 on W jets), and tf has the opposite
  sign at Δz = +0.08.
- H1_z (z read as z): not separated from "D2/τ21 read". The arm-C response equals the response of MLPx, which has D2 and τ21 but no
  z, within 0.8–1.2× on JetClass W (ParT_full 0.99 at +0.08, 0.78 at −0.08). It is larger than MLP0+z, which has z but no D2 (1.96×
  at −0.08). An edit that changes z at fixed D2 is not available at fixed ΔR and prong masses with this construction.
- Arm B: at fixed z, a separation change that adds about 10 GeV of mass lowers the JetClass W/Z scores by 0.2–0.7 log-odds more
  than a pT scaling that adds the same mass. The references show 0.0–0.24, so this network-specific part is 2–3× the references'.
  QCD jets: about 0.

## Caveats

- One sample per dataset, one seed; not yet verified. Acceptance (< 1 % pT migration on reclustering) keeps about 52 % of jets for
  B(0.85) and C(−0.08) on JetClass and fewer on hls4ml q/g (about 31–35 %). Contrasts at these settings are on a selected subset.
- MBs (the mass-only control) scales pT by +8 to +14 %. The hls4ml small nets respond strongly to it, so B − MBs and C − MBsC on
  hls4ml mix the tested effect with a pT effect. C − BC (fixed pT) is the cleaner hls4ml arm-C contrast.
- Arm A keeps the jet mass exactly; jet pT changes by < 0.1 GeV. Reclustered prong masses are restored to within 0.04–1.2 GeV on average
  (labels can still shift by < 1 % pT); D2 is not restored (it moves by the same amount as under D).
- MBp was rarely accepted (≈ 10–30 %) and only for mass increases; it is reported in the JSON only.

## Suggested next step

Separate z from D2: an edit that changes z at fixed D2, τ21, ΔR and prong masses. This needs non-rigid changes inside the prongs,
e.g. moving soft wide-angle radiation so that e3 compensates. Alternatively, use a reference trained on D2 with z decorrelated
(reweighting the training jets so z is independent of D2 at fixed m) and compare its arm-C response with the networks'.

## Explainer entry (for site/explain.py; merged by the ledger agent)

```python
H50_EXPLAINER = [
 ('Question', 'H45 showed that every tagger reacts when momentum is moved between the two prongs of a jet at fixed jet four-momentum. That edit also changes the prong separation $\\Delta R$, the shape variables $D_2$ and $\\tau_{21}$, and the prong masses found by reclustering. Which of these do the taggers actually read: the momentum fraction $z$, the separation $\\Delta R$, or the prong masses?', None),
 ('Physics background', 'For a two-prong jet, $m^2 \\approx m_a^2/z + m_b^2/(1-z) + z(1-z)\\,p_T^2\\,\\Delta R^2$, where $z$ is the softer prong\'s $p_T$ fraction and $m_a, m_b$ are the prong masses. At fixed $m$ and $p_T$, $z$ and $\\Delta R$ are tied, so no edit at fixed four-momentum can change one without the other. W/Z decays give balanced prongs (larger $z$); QCD splittings prefer small $z$. $D_2 = e_3/e_2^3$ with $e_2 \\propto z(1-z)\\Delta R$, so $D_2$ also moves with $z$.', None),
 ('Method', 'Three arms on real JetClass and hls4ml jets. Arm A: the H45 rest-frame edit $D(\\Delta c)$, then each reclustered prong\'s mass is restored to its original value by rescaling the particle offsets from the prong axis, and the jet mass is restored by moving the prongs rigidly (asserted to $10^{-6}$ of $E_{jet}$); control: rotation about the jet axis ($c\\varphi$). Arm B: both prongs moved rigidly apart in rapidity–azimuth so $\\Delta R \\to \\lambda\\Delta R$ at exactly fixed $z$ and prong masses; control: a common $p_T$ scaling of all particles giving the same mass change (keeps $z$ and $\\Delta R$). Arm C: particle $p_T$ re-shared between the prongs so $z \\to z \\pm 0.08$, prong masses restored and prongs put back on their axes ($\\Delta R$ exact); controls: the same mass change made by $p_T$ scaling or by a $\\Delta R$ change. Reference classifiers trained on unedited jets: $(m, p_T, N)$ plus $\\Delta R$, prong masses, $D_2$ or $z$; MLPx $(m, p_T, \\tau_{21}, D_2, N, \\Delta R)$; a GBDT on all.', 'results/physics/h50_edit.png'),
 ('Result', 'Arm A: restoring the prong masses leaves the response unchanged (ratio 0.92–1.14 for all eight taggers on asymmetric W/Z jets; e.g. ParT W jets −2.55 → −2.76 log-odds), while a classifier that only sees prong masses drops to 0.16. Arm C: changing only $z$ at fixed $\\Delta R$ and prong masses moves the JetClass taggers with the sign of the $z(1-z)$ balance (W jets −0.62 log-odds for $z-0.08$); MLPx, which sees $D_2$ and $\\tau_{21}$ but not $z$, predicts the same (ratio 0.8–1.0), because the $z$ edit also changes $D_2$ by up to 0.9. Arm B: a separation change at fixed $z$ lowers JetClass W/Z scores 2–3× more than the references do, relative to a $p_T$ scaling with the same mass change.', 'results/physics/h50_response.png'),
 ('Checks', 'Invariants asserted in float64 on every accepted jet; reclustering moves < 1 % of the jet $p_T$ between prongs; accuracies reproduce (ParT_full 0.758, JEDI-linear 0.813); permuted/redrawn networks ≈ 0 on JetClass; Holm over 560 tests; McNemar on calls.', None),
 ('Physical significance', 'The taggers\' response to the prong momentum sharing is not a prong-mass effect, and it is not explained by the prong separation alone: at fixed $\\Delta R$ and prong masses, more balanced prongs still look more like W/Z. Whether the networks read $z$ itself or the energy-correlation shape ($D_2$, $\\tau_{21}$) that $z$ drives at fixed $\\Delta R$ is still open.', None),
]
# glossary additions
('A / B / C arms (H50)', 'A: H45 edit D with the reclustered prong masses and the jet mass restored (z and ΔR move together); B: prongs moved rigidly apart, ΔR × λ at exactly fixed z and prong masses; C: z changed by re-sharing particle pT between the prongs at exactly fixed ΔR and prong masses'),
('MBs / MBp (mass-only controls, H50)', 'MBs: every particle pT scaled by one factor so the jet mass changes by the same amount as the tested edit; z and ΔR stay fixed, jet pT changes. MBp: offsets inside both prongs scaled so the jet mass matches; z and ΔR fixed, prong masses change'),
('collapse ratio r_A (H50)', '(response to arm A − control) / (response to the H45 edit D − control): 1 = restoring the prong masses changes nothing, 0 = the response came from the prong masses'),
('reclustered prong mass', 'mass of one of the two exclusive-kT subjets found by re-running the clustering on the (edited) jet; differs from the mass of a fixed set of particles when particles change subjet'),
```
