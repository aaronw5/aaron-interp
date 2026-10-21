# H50 verification (z vs ΔR, D2 and reclustered prong masses)

Verifier, loop D, 2026-10-20. Script `interp/h50_verify.py` (imports `interp/h50_z_vs_dr.py` unchanged and adds two controls).
Jobs: d-h50-verify (CPU 4/8, edits and references; a first attempt stopped on an assertion, see Code review), d-h50-verify-g (GPU,
JetClass networks), d-h50-verify-s (statistics). Results: `results/physics/h50_verify_z_vs_dr.json` (full statistics),
`results/physics/h50_verify_summary.json` (the cells quoted here), `h50_verify_edit.png`, `h50_verify_response.png`;
per-jet arrays on the volume as `h50_verify_jc_{inputs,obs,logits}.npz`.

## Conclusion

The two main measurements reproduce on new jets. (1) Putting the reclustered prong masses back after the H45 edit leaves the
taggers' response to more asymmetric prongs unchanged; ParT's ratio is 1.10. (2) Changing the momentum fraction z at fixed prong
separation and prong masses moves the scores in the direction of prong balance. But the claim that separation cannot explain the
response does not hold. A reference classifier that never sees z or D2, and reads only the jet mass, pT, multiplicity, the prong
separation ΔR and the two prong masses, reproduces the networks' z-edit response to within about 0.15 log-odds (log-odds =
log p(signal)/p(background)). These quantities fix z through the two-body mass formula, so this edit cannot tell whether a network
reads z or reads separation and mass together. The prong-mass result holds only for edits toward more asymmetric prongs. For edits
toward more balanced prongs, putting the prong masses back changes ParT's response; its ratio becomes negative.

## What was rerun

- Sample: JetClass test files WToQQ_102, ZToQQ_102 and ZJetsToNuNu_102 (H50 used the *_101 files). Selection 500 < pT < 1000 GeV,
  |η| < 2, at least 3 particles; 2,000 random rows per class (rng 6001–6003), so 6,000 jets. Edit rng 6010. The reference
  classifiers were retrained on train_100M *_003 (H50: *_002), 10,000 jets per class (rng 6016–6018).
- Everything else is the H50 code: arms A, B and C, the controls cφ, MBs, MBp, BC, MBsC and MBpC, the acceptance cut (< 1 % of jet
  pT changes prong when the edited jet is reclustered), bootstrap with B = 2,000, and Holm.
- Unedited 10-class accuracy on this sample: ParT_full 0.753, ParT_kin 0.649, PN_full 0.729, PN_kin 0.621 (H50: 0.758 / 0.666 /
  0.736 / 0.637). Permuted-weight ParT: 0.007, 0.116, 0.022.
- Holm over the 240 primary JetClass tests: 193 have adjusted p < 0.05.

## Added controls (why)

1. Reference MLP0+dR+pm: an MLP(64, 64) on (m, pT, N, ΔR, m_hi, m_lo). Here m = jet mass, N = constituent count, ΔR =
   rapidity–azimuth distance between the two exclusive-kT prongs, and m_hi / m_lo = heavier / lighter prong mass. It never sees z,
   D2 or τ21, but its inputs fix z through m² ≈ m_hi²/z + m_lo²/(1−z) + z(1−z)·pT²·ΔR² (approximately; the formula pairs each prong
   mass with that prong's momentum share). H50's own H0_dR is "the networks read ΔR and mass, not z". H50 tested it against
   MLP0+dR, which lacks the prong masses. Arm C holds the prong masses fixed while z changes, so a ΔR-and-mass reader needs the
   prong masses to register the change. MLP0+dR+pm is that reader. Also added: MLP0+z+dR.
2. Brot(λ): a rotation control for arm B. Both prongs are translated rigidly so that the line joining them rotates about the
   pT-weighted centre by θ = ±2·arcsin(|λ−1|/2) (θ = ±0.150 rad). Each prong moves the same distance as under B(λ). ΔR, z, the
   prong masses and every constituent pT stay exactly fixed. The jet mass changes by only ±0.03 GeV (self-test jet), against
   ±8 GeV under B. Brot minus the unedited jet measures any response to moving the prongs that does not come from the separation
   change.

## Numbers (JetClass W jets, score log p(W)/p(QCD) unless stated; mean Δ log-odds, edit − control, 95 % bootstrap CI over jets)

Arm A: collapse ratio r_A = (A − cφ)/(D − cφ). D = H45 decay-axis edit; A = D with the reclustered prong masses and the jet mass
restored; cφ = rotation about the jet axis. r_A = 1 means restoring the prong masses changes nothing; r_A = 0 means the response
disappears.

| cell | Δc | ParT_full | ParT_kin | PN_full | PN_kin | MLP0+pm | H50 ParT_full |
|---|---|---|---|---|---|---|---|
| W, W-vs-QCD | +0.3 (asymmetric) | 1.10 [1.07, 1.14] | 1.15 | 1.07 | 1.16 | 0.21 [0.15, 0.28] | 1.08 |
| Z, Z-vs-QCD | +0.3 | 1.10 [1.06, 1.14] | 1.10 | 1.05 | 1.03 | 0.23 | 1.06 |
| W, W-vs-QCD | −0.3 (balanced) | −0.42 [−0.86, −0.10] | 0.65 [0.57, 0.72] | 0.41 [0.28, 0.52] | 0.71 | 0.08 | −0.44 |
| Z, Z-vs-QCD | −0.3 | −0.68 [−1.30, −0.26] | 0.71 | 0.30 | 1.06 | 0.17 | −0.52 |
| Z, W-vs-Z | +0.3 | −2.87 [−14.6, −1.08] | −0.13 | −0.34 | – | – | −1.92 |

ParT_full, W jets, Δc = +0.3, n = 1,408: D − cφ = −2.57, A − cφ = −2.78 [−2.91, −2.65] (H50: −2.55 → −2.76).

Arm C: z ± 0.08 at fixed ΔR and fixed prong masses. MBsC = all constituent pT scaled to the same jet mass; BC = ΔR changed to the same
jet mass. The "− MLP0+dR+pm" column is the network minus the reference, jet by jet, with its CI. H0 for this column: the network
responds like a reader of (m, pT, N, ΔR, prong masses). H1: the network responds beyond that reader.

| contrast | Δz | ParT_full | − MLP0+dR+pm | PN_full | − MLP0+dR+pm | ParT_kin − ref | MLP0+dR | MLP0+dR+pm | MLPx |
|---|---|---|---|---|---|---|---|---|---|
| C − MBsC | +0.08 | +0.25 [+0.22, +0.29] | −0.16 [−0.19, −0.12] | +0.32 | −0.10 [−0.13, −0.06] | −0.15 | +0.20 | +0.41 | +0.25 |
| C − MBsC | −0.08 | −0.61 [−0.67, −0.56] | −0.01 [−0.06, +0.04] | −0.60 | −0.00 [−0.05, +0.04] | −0.33 | −0.16 | −0.60 | −0.76 |
| C − BC | +0.08 | +0.54 [+0.48, +0.60] | +0.07 [+0.03, +0.11] | +0.60 | +0.13 [+0.09, +0.16] | −0.04 | +0.28 | +0.48 | +0.45 |
| C − BC | −0.08 | −0.19 [−0.25, −0.13] | +0.10 [+0.04, +0.16] | −0.28 | +0.01 [−0.04, +0.07] | −0.41 | −0.06 | −0.29 | −0.50 |

H50 values for ParT_full: +0.23, −0.62, +0.50, −0.16. Z jets (Z-vs-QCD) give the same pattern; ParT_full − MLP0+dR+pm is
−0.19, −0.08, +0.05 and +0.26 in the same four rows. Against MLP0+dR alone (no prong masses), the original H50 cell C − MBsC
(+0.08) gave ParT_full − MLP0+dR = +0.01 [−0.02, +0.04]; this sample gives +0.06 [+0.02, +0.09].

Arm B: ΔR × λ at fixed z and prong masses.

| contrast | λ | ParT_full | ParT_kin | PN_full | PN_kin | MLP0+dR | MLP0+dR+pm | MLPx |
|---|---|---|---|---|---|---|---|---|
| B − MBs (pT-scaling control) | 1.15 (ΔR up, m +≈10 GeV) | −0.41 [−0.43, −0.38] | −0.22 | −0.38 | −0.17 | −0.06 | +0.00 | −0.15 |
| B − MBs | 0.85 (ΔR down, m −≈10 GeV) | −0.69 [−0.73, −0.66] | −0.35 | −0.53 | −0.32 | −0.14 | −0.41 | −0.26 |
| B − MBp (prong-width control, pT fixed) | 1.15 | +0.20 [+0.06, +0.36] | +0.10 | +0.27 | +0.18 | −0.15 | +0.21 | +0.16 |
| Brot − unedited | 1.15 | −0.018 [−0.026, −0.009] | −0.010 | −0.015 | −0.012 | 0.00 | −0.01 | 0.00 |
| Brot − unedited | 0.85 | −0.015 [−0.024, −0.006] | −0.010 | −0.011 | −0.008 | 0.00 | −0.01 | 0.00 |

QCD jets: B − MBs between −0.01 and +0.11; Brot between −0.002 and +0.012.

## Code review (interp/h50_z_vs_dr.py)

- Coordinates. Edits act on (pT, rapidity y, φ, m) for each constituent and are rebuilt with build(). Every network sees the
  edited jet rebuilt about its own E-scheme axis, using the shared H45/H23 scoring path. The observables come from the same arrays
  the networks scored (`P|v`). Rows are drawn at random by read_window, never as the first k. Padded slots are excluded throughout
  via nn. No convention error found.
- Conservation. The self-test reproduces every invariant: B, C and Brot keep z, ΔR and the prong masses to 10⁻⁹; MBs and MBp match
  the jet mass to 10⁻⁶; arm A matches its targets to 10⁻⁶·E.
- Crash on one edge case. restore_pm accepts a one-particle prong whose mass misses its target by less than 10⁻³ GeV. restore_job
  then asserts agreement to 10⁻⁶·E, so such a jet stops the whole run. This happened once on the new sample: target 8·10⁻⁴ GeV,
  restored mass 0. The original run did not meet the case, so H50's numbers are unaffected. h50_verify.py rejects the jet instead
  (3 jets in all).
- Arm C keeps the sum of the two prongs' vector pTs, not the jet scalar pT as the report states. The achieved jet-pT change is
  −0.6 / +0.5 GeV, which is negligible.
- Acceptance is asymmetric. B(0.85) is accepted for 51 % of jets (3,066 / 6,000 meet the migration cut) and B(1.15) for 99.8 %.
  C(−0.08) is accepted for about 53 %. The λ = 0.85 and Δz = −0.08 contrasts are therefore on a selected subset, as the report
  says.
- Working point. McNemar uses the median unedited signal score (ε_S = 0.5), as stated.

## Figure check

`results/physics/h50_edit.png`. It shows one W jet. Its titles agree with the arms: A keeps both prong masses (19.5 / 11.4 GeV) and
m = 79.0 GeV while z goes 0.34 → 0.19 and D2 1.07 → 1.74. B keeps z = 0.340 while ΔR goes 0.215 → 0.248. C keeps ΔR = 0.215 while
z goes 0.34 → 0.42. In arm A the soft prong is visibly spread out: its particles were pushed away from the prong axis to restore
its mass after it lost pT. So A changes the internal shape of the soft prong, which is why D2 does not return to its original value.

`h50_response.png` matches the JSON. In its JetClass panel, MLP0+dR (hatched brown) is close to the networks in "C: z+0.08 −
pT scaling".

## Corrected reading

- Prong masses (holds; scope narrowed). Restoring the reclustered prong masses leaves the W/Z-vs-QCD response to the asymmetric
  edit (Δc = +0.3) at 1.03–1.16 of its size for the four JetClass networks, on both samples. A classifier on prong masses alone
  drops to 0.16–0.23. For the balanced edit (Δc = −0.3) the ratio is −0.42 (ParT_full), 0.41 (PN_full) and 0.65–0.71 (kin
  models). There, the prong masses carry part of the (small) response. On the W-vs-Z score of Z jets the ratio also departs from 1.
- z vs separation (H0_dR not excluded). At fixed ΔR and prong masses, the z edit moves JetClass W/Z-vs-QCD scores with the sign of
  the prong balance; this is reproduced (+0.25 / −0.61 for ParT_full against the pT-scaling control). A reference with no z and
  no D2, fed (m, pT, N, ΔR, m_hi, m_lo), gives +0.41 / −0.60 on the same edit. The ParT_full and PN_full differences from it are
  −0.16 to +0.13 log-odds over the four W contrasts; 3 of these 8 CIs include 0. ParT_kin and PN_kin exceed it at Δz = −0.08 by
  0.3–0.4. Because those inputs fix z through the mass formula, arm C does not separate "z read" from "ΔR, m, pT and prong masses
  read". The sign-of-z(1−z) cell count does not exclude H0_dR, because every z-blind reference with ΔR and m has the same sign.
- Arm B (wording corrected). Relative to a pT scaling with the same mass change, a separation change lowers JetClass W/Z scores
  by 0.2–0.4 log-odds (ΔR × 1.15) and 0.3–0.7 (ΔR × 0.85). So the effect appears in both directions, not only for an increase.
  The pT-scaling control keeps m/pT and every angle fixed, so the contrast says that the networks penalise mass shifts that come
  with changed angles more than mass shifts at fixed angles. Against the prong-width control (mass changed at fixed pT and ΔR),
  the sign reverses: +0.10 to +0.27, close to MLP0+dR+pm (+0.21). Moving the prongs without changing ΔR (Brot) changes the scores
  by at most 0.02. So the arm-B response is not a generic response to translating particles.
- D2 / τ21: still not separated from z (unchanged from H50).
- hls4ml: not rerun.
