# H45v — skeptic check of H45 (prong momentum sharing at fixed jet four-momentum)

Script `interp/h45v_verify.py` (reuses the H45 functions; the edit is re-implemented and compared with `h45_prong_sharing.edit_one`
on every jet). Results `results/physics/h45_verify.json` (also `$AI_RESULTS/physics/h45_verify.json`), sample metadata
`results/physics/h45_verify_{jc,hls}_meta.json`. Cluster jobs: d-h45-vsmoke, d-h45-vsmoke-g, d-h45-vsmoke-s (smoke, 40 jets per class),
d-h45-verify (CPU: JetClass and hls4ml edits, observables, references, small nets, JEDI-linear), d-h45-verify-g (GPU: JetClass
networks), d-h45-verify-s (statistics). Per-jet arrays on the volume: `/j-jepa-vol/aaron-interp/results/physics/h45_verify*_{inputs,obs,logits*}.npz`
(delete together with the H45 ones).

## Conclusion

On fresh jets, all tested taggers again change their W/Z score when momentum is moved between the two prongs of a jet while the
jet's total four-momentum stays exactly fixed: more asymmetric prongs lower the W-vs-QCD score of W jets by about 2.5 log-odds for
ParT, and more balanced prongs raise the score of QCD jets. A new control that rotates the particles by the same angle but keeps both
prongs' momenta fixed moves the JetClass scores by at most 0.17 log-odds, so the response comes from where the two prongs point and
how they share momentum, not from rotating particles in general. Two parts of the H45 reading did not hold. In JetClass the W-vs-Z
score follows the edit in the same direction for W and Z jets: balanced prongs look more W-like and asymmetric prongs more Z-like.
Also, on hls4ml quark and gluon jets the small Deep Sets network responds just as much to the control as to the edit. Whether the
networks read the sharing z itself or the prong separation and shape variables that change with it remains open.

Terms: log-odds = log p(A)/p(B) from the network's class scores; prong = one of the two exclusive-kT subjets of all constituents;
z = pT fraction of the softer prong; |cos θ*| = angle of the prong axis to the jet direction in the jet rest frame (0 = balanced).

## What was checked

1. Code read (h45_prong_sharing.py). No bug found in: the edit (rotation sign, angle, rest-frame boost; `B(−β)·R·B(β)` applied to every
   real constituent, padded slots untouched); the controls; the samples (random rows via `read_window` / rng choice, never first-k);
   the network inputs (`jc_dict`/`part_core.build` re-sort particles by pT, carrying track/PID features and stored particle masses
   with their particles; axis = E-scheme sum of the edited jet, which D leaves unchanged; η = jet_eta + sign(jet_eta)·Δη);
   the hls4ml model inputs (pT, Δη, Δφ about each jet's own axis, pT-sorted, 64 slots); per-jet pairing of edited/unedited scores
   on the same rows; working point = median unedited score of the signal class (ε_S = 0.5).
   Remark (not a code bug): D fixes every p_i·p_j, hence the masses of the ORIGINAL prong assignments, but the observables and
   networks see the jet after reclustering. The reclustered heavier-prong mass changes under D by −3.5 GeV (W jets, Δc = −0.3)
   to +6.6 GeV (Z jets, Δc = +0.3), and D2 by −2.3 (QCD, Δc = −0.3) to +2.1 (hls W, Δc = +0.3). The report's statement "c-side changes
   prong masses and D2 (D does not)" holds only for the fixed assignment.
2. Fresh-sample rerun of D and cφ at Δc ∈ {−0.3, −0.15, +0.15, +0.3}, with MLPx/MLPz references retrained on fresh train rows.
3. New control cn (described below), at Δc = ±0.3.

## Procedure

Samples (no overlap with H45):
- JetClass test files WToQQ_101, ZToQQ_101, ZJetsToNuNu_101 (H45 used _100), stored 500 < pT < 1000 GeV, |η| < 2, ≥ 3 particles;
  1,200 random rows per class (rng 4601/4602/4603).
- hls4ml test jets, 1,200 random rows per class (rng 4614) drawn from the rows not in H45's sample (H45's rng-4504 draw reproduced and
  excluded; overlap 0). W, Z, q, g analysed; tops only in the 5-class accuracy.

Edit D(Δc): boost all constituents into the jet rest frame, rotate them by α = arccos c₀ − arccos c₁ about n₊ × β̂ (n₊ = rest-frame
direction of the forward prong, c₀ = |cos θ*|, c₁ = clip(c₀ + Δc, 0, 0.95)), boost back. Cross-check: identical to H45's edit to 0.0
on every jet; jet four-momentum changed by ≤ 1.2·10⁻¹² of the jet energy; p_i·p_j by ≤ 4·10⁻¹⁶ of E_jet² (3,000 jet-edits checked).

Controls (same jets):
- cφ (H45): rotation by the same |α| about β̂ in the rest frame = a rotation of the whole jet about its own axis. z, ΔR, prong energies
  fixed. This is close to a symmetry of jets, so it tests only orientation sensitivity.
- cn (NEW): rotation by the same |α| about n₊ (the prong axis itself) in the rest frame, random sign. Both prong four-momenta stay
  exactly fixed (checked to 1.3·10⁻¹²), so z, |cos θ*|, ΔR, the prong masses (original assignment) and the jet four-momentum are fixed.
  The particles inside each prong are turned about the prong axis. It is a rest-frame rotation of the same angle that is not a
  symmetry of the jet. Why this is the right comparison: it separates "moving the prong axis (changing the sharing)" from "applying a
  rest-frame rotation of size α to all particles". Achieved changes (JetClass W jets, Δc = +0.3): Δz −0.003, ΔR +0.004, Δτ21 +0.003,
  ΔD2 −0.04, median pT-weighted particle displacement 0.014 (D: 0.053, cφ: 0.042). On QCD and hls q/g jets cn lowers D2 by
  0.2–0.9, which MLPx also responds to.

Models: JetClass ParT_full, ParT_kin, PN_full, PN_kin; nulls = 3 ParT_full with every weight tensor permuted (torch seeds 15–17).
hls4ml JEDI-linear official n64 3-feature, dsets-d64-n64 seed0; nulls = 2 JEDI with redrawn weights (seeds 10, 11), 2 permuted dsets
(seeds 17, 18). (jedis and tf were not rerun.)
References: MLPx = MLP(64,64) on (m, pT, τ21, D2, N, ΔR); MLPz = MLPx + z_g; 3 retrains each, 10,000 fresh train jets per class
(JetClass train *_001 files, rng 4616–4618; hls4ml train archive without the dev split, rng 4619).

Metric: per jet Δs = s(variant) − s(unedited), with scores WQ = L_W − L_QCD, ZQ, WZ = L_W − L_Z (JetClass) and WQG = L_W −
logsumexp(L_q, L_g), ZQG, WZ (hls4ml). Contrasts D − cφ and cn − cφ on jets accepted for all variants (both prong centres within 0.75
of the axis). Mean with 95 % bootstrap CI over jets (2,000 resamples), sign-flip permutation p (2,000) and one-sample t-test p. Holm
over all 240 tests (160 D − cφ, 80 cn − cφ). H0: mean contrast = 0. H1: ≠ 0.

## Unedited accuracy on the fresh samples

| model | accuracy | H45 sample |
|---|---|---|
| ParT_full / ParT_kin / PN_full / PN_kin (10-class, W/Z/QCD jets) | 0.751 / 0.655 / 0.729 / 0.616 | 0.756 / 0.668 / 0.728 / 0.630 |
| JEDI-linear / dsets (5-class) | 0.809 / 0.803 | 0.809 / 0.807 |
| permuted ParT ×3; redrawn JEDI ×2; permuted dsets ×2 | 0.001–0.18; 0.20; 0.22–0.23 | null |

Reference held-out AUC: JetClass MLPx W-vs-QCD 0.940–0.945, W-vs-Z 0.738–0.741; hls4ml W-vs-q/g 0.972–0.974, W-vs-Z 0.974–0.983.

## Numbers (D − cφ, Δ log-odds, 95 % CI; H45 value in parentheses)

| cell | Δc | ParT_full | ParT_kin | PN_full | PN_kin | MLPx (3 retrains) | null |
|---|---|---|---|---|---|---|---|
| JetClass QCD, W-vs-QCD | −0.3 | +1.04 [0.95, 1.12] (+0.95) | +1.58 (+1.49) | +1.02 (+0.94) | +1.43 (+1.34) | +1.04…+1.22 | −0.16…−0.01 |
| JetClass W, W-vs-QCD | −0.3 | +0.28 [0.21, 0.34] (+0.23) | +0.65 (+0.56) | +0.48 (+0.44) | +0.68 (+0.60) | +0.48…+0.61 | −0.06…−0.01 |
| JetClass W, W-vs-QCD | +0.3 | −2.54 [−2.71, −2.38] (−2.61) | −1.88 (−1.94) | −2.39 (−2.48) | −1.83 (−1.84) | −1.33…−1.19 | +0.06…+0.12 |
| JetClass W, W-vs-Z | +0.3 | −1.59 [−1.69, −1.48] (−1.60) | −0.51 | −1.30 | −0.43 | −0.37…−0.23 | −0.05…+0.07 |
| JetClass Z, W-vs-Z | −0.3 | +0.43 [0.37, 0.49] (+0.44) | +0.18 | +0.37 | +0.18 | +0.06…+0.12 | −0.07…+0.05 |
| JetClass Z, W-vs-Z | +0.3 | −0.10 [−0.20, +0.01] (−0.11) | −0.13 | −0.32 | −0.12 | −0.21…−0.04 | −0.02…+0.20 |

| cell | Δc | JEDI-linear | dsets | MLPx | null |
|---|---|---|---|---|---|
| hls W, W-vs-q/g | +0.3 | −1.96 [−2.09, −1.83] (−1.76) | −2.13 (−2.08) | −1.83…−1.32 | 0.00…+0.53 |
| hls Z, W-vs-Z | +0.3 | +1.79 [1.64, 1.93] (+1.56) | +1.69 (+1.46) | +0.09…+0.71 | 0.00…+1.14 |
| hls q, W-vs-q/g | −0.3 | +0.04 [−0.02, 0.09] (+0.03) | +0.54 (+0.52) | −0.05…+0.03 | −0.11…0.00 |
| hls g, W-vs-q/g | +0.3 | −0.06 [−0.16, 0.03] (−0.09) | +0.53 (+0.59) | −0.78…−0.72 | 0.00…+0.75 |

All 160 network cells reproduce the H45 means within 0.24 log-odds (149 within 0.1). D − cφ ≠ 0 after Holm in 146/160 network
cells. Calls at ε_S = 0.5 (W jets, W-vs-QCD, Δc = +0.3): ParT_full passes 51.3 % under cφ → 27.4 % under D (McNemar p = 2·10⁻⁶¹);
JEDI-linear 49.8 % → 28.5 % (p = 6·10⁻⁵²). Dose (Spearman of D − cφ against Δc): ≤ −0.8 in 24/24 JetClass series (QCD series are
non-monotone between +0.15 and +0.3: −0.24 → −0.13 for ParT_full); in hls4ml 11/16, exceptions JEDI and dsets on Z jets W-vs-Z (+1.0,
+0.8) and dsets on g jets / q jets Z-vs-q/g. Network / MLPx ratio (cells with |MLPx| ≥ 0.2): median ParT_full 0.92, ParT_kin 1.41,
PN_full 0.93, PN_kin 1.31, JEDI 1.26, dsets 1.12; per-cell range for ParT_full 0.23–5.2. Per jet, ParT_full − MLPx on W jets at
Δc = +0.3 is −1.21 [−1.33, −1.09]: the full JetClass models respond about twice as strongly as MLPx to asymmetric W/Z prongs and
3–5× in the W-vs-Z cells, and less than MLPx (0.25–0.5×) to balanced W/Z prongs. MLPz − MLPx ≤ 0.2 in every cell.

JEDI-linear input moments under D (fresh hls4ml jets): 90th percentile |Δlog m_A| ≤ 0.0093, |ΔM0/M0| ≤ 5.4·10⁻⁴
(m_A = √(M0·M2), M0 = Σ pT, M2 = Σ pT (Δη² + Δφ²) about the jet axis). Reproduced.

New control cn (cn − cφ, Δ log-odds, 95 % CI), compared with D − cφ in the same cell:

| cell | Δc | network | cn − cφ | D − cφ | MLPx cn − cφ |
|---|---|---|---|---|---|
| JetClass W, W-vs-QCD | +0.3 | ParT_full | −0.17 [−0.21, −0.12] | −2.54 | +0.03 |
| JetClass Z, Z-vs-QCD | +0.3 | ParT_full | −0.15 [−0.20, −0.10] | −2.35 | +0.03 |
| JetClass QCD, W-vs-QCD | −0.3 | ParT_full / ParT_kin | +0.17 / +0.28 | +1.04 / +1.58 | +0.11…+0.17 |
| JetClass QCD, W-vs-QCD | +0.3 | ParT_full / ParT_kin | +0.25 / +0.41 | −0.13 / −0.17 | +0.21…+0.26 |
| hls W, W-vs-q/g | +0.3 | JEDI / dsets | −0.03 [−0.08, 0.03] / −0.34 | −1.96 / −2.13 | −0.02…−0.01 |
| hls q, W-vs-q/g | −0.3 | JEDI / dsets | −0.06 / +0.48 [0.41, 0.54] | +0.04 / +0.54 | 0.00 |
| hls g, Z-vs-q/g | +0.3 | JEDI / dsets | +0.18 / +0.76 [0.68, 0.84] | +0.01 / +1.29 | +0.03…+0.04 |

cn − cφ ≠ 0 after Holm in 55/80 cells (n ≈ 1,200 resolves 0.05 log-odds). On JetClass W/Z jets and on hls4ml W/Z jets for JEDI-linear,
cn moves scores by ≤ 0.17 while D moves them by up to 2.5 at the same rest-frame angle. On QCD jets cn gives +0.15…+0.41, of the size
MLPx predicts from cn's D2 decrease. For dsets on hls4ml q/g jets, cn reproduces the D response (D − cn between −0.09 and +0.14 on q
jets).

## Hypotheses and which are excluded

- "Taggers ignore the two-prong configuration at fixed jet four-momentum": excluded (D − cφ ≠ 0, Holm, in 146/160 cells; nulls ≈ 0).
- "The response is a generic reaction to rotating particles in the rest frame": excluded on JetClass W/Z jets and for JEDI-linear on
  hls4ml W/Z jets (cn ≤ 0.17 vs D up to 2.5). Not excluded for dsets on hls4ml q/g jets (cn ≈ D).
- "Uses z" vs "uses ΔR, τ21, D2, reclustered prong masses at fixed m": not separated (MLPz ≈ MLPx; at fixed m and pT, z and ΔR are tied
  one-to-one; D also changes the reclustered prong mass by several GeV and D2 by up to 2).
- W-vs-Z, "loss of W/Z identity in both directions" (H45 reading) vs "ordered in Δc (balanced → W, asymmetric → Z)": JetClass full
  models follow the ordered pattern for both W and Z jets (Z jets: +0.43 at Δc = −0.3, −0.10 to −0.32 at +0.3); hls4ml follows the
  loss pattern (W jets → Z, Z jets → W at Δc = +0.3).

## Corrections to the H45 reading

1. "Balanced prongs move scores toward W/Z and asymmetric prongs away, in order of Δc, on QCD, W and Z jets in both datasets": holds for
   all JetClass series and for hls4ml W/Z jets (W/Z-vs-q/g). On hls4ml q/g jets dsets raises the W/Z score for both signs of Δc, and
   the cn control does the same, so this is a generic response. JEDI-linear gives ≈ 0 on q jets for Δc < 0 and on g jets for Δc = +0.3
   (−0.06 vs MLPx −0.75).
2. "Asymmetric splitting reduces W-vs-Z separation in both directions": true only on hls4ml. In JetClass the W-vs-Z score is ordered in
   Δc for both classes: balanced → more W-like, asymmetric → more Z-like.
3. "JEDI-linear on q/g jets stays at the MLPx level": true on q jets; on g jets at Δc > 0 JEDI is near 0 where MLPx predicts −0.3 to −0.8.
4. "The c-side control changes prong masses and D2 (D does not)": D does not change the masses of the original particle assignment,
   but it changes the reclustered prong masses and D2 that the observables, and possibly the networks, see.
5. "MLPx predicts a similar size": this holds for the median (0.9–1.4). Per cell, the full JetClass models exceed MLPx about twofold on
   asymmetric W/Z prongs and 3–5× in W-vs-Z cells, and fall below it on balanced W/Z prongs. Net − MLPx CIs exclude 0 there.

Confirmed: all edit invariants; the size, sign and Δc ordering of the D − cφ responses on fresh jets (within 0.25 log-odds); cφ and nulls
≈ 0; MLPz ≈ MLPx; JEDI-linear's m_A and ΣpT unchanged (< 1 %) while its W/Z scores move by up to 2.

## Caveats

- One fresh seed; 1,200 jets per class (H45: 2,000); jedis and tf, the c-side control, C2ST and TN were not rerun.
- cn changes D2 by −0.2 to −0.9 on QCD and q/g jets, so it is not a zero-side-effect control there.
- Δc = −0.3 jets were off-manifold in H45 (C2ST up to 0.90); not re-measured.

## Glossary addition (for site/explain.GLOSSARY, to be merged by the ledger agent)

('cn (prong-axis rotation control)', 'control edit of H45v: in the jet rest frame, all particles are rotated by the same angle as the
tested edit, but about the prong axis itself, so both prong four-momenta (hence z, |cos θ*|, ΔR, prong masses) and the jet
four-momentum stay exactly fixed; only the arrangement of particles inside each prong changes')
