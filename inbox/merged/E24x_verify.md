# E24x verify: fresh-seed rerun and rigid-transform tests of the S58 / S31 names (2026-10-20)

## Conclusion
On a new sample, the two context-only summary units S58 and S31 again ignore the individual particles and depend only on the jet-level context, and their thresholds are again described by plain (not pT-weighted) particle-count sums: an off-centre count (dipole) for S31 and an elongation count (quadrupole) for S58, with held-out R² 0.93–0.95. Rotating or mirroring every particle in the (Δη, Δφ) plane changes both units, as a fixed-frame quantity predicts and a rotation-invariant one cannot. Two parts of the reading do not hold as stated. First, S58 is not a pure quadrupole in its response: turning the jet by 180° leaves every quadrupole sum unchanged, yet it still moves S58 by 40 % as much as a 90° turn does, and the quadrupole name does not predict this. Second, for S31 the network's per-slot rounding of the context input shapes the response to these edits so strongly that even the exact threshold predicts it poorly.

Terms: *threshold* θ = −(W3 · q(g) + b3), the jet-level part of the unit's pre-activation; *per-slot l3 quantizer* = the fixed-point rounding/clipping that the l3 layer applies separately to each of the 64 particle slots before the weights.

## Procedure
- Script `interp/e24x_verify.py` (imports the E24x probe, gate and multipole fit from `interp/e24x_context_jedi.py` unchanged). Job `e24x-verify` (1 CPU / 2 GiB, wall 808 s, peak 2,048 MiB incl. file cache); smoke `e24x-vsmoke` (N 1,500 / 996). Log `logs/e24x-verify.log`; results `results/jedi/e24x_verify.json`.
- Samples (fresh seeds): fit `sample(3000, 4101)`, N 15,000, accuracy 80.46 %; test `sample(1500, 4102)` minus 433 jets also in the fit sample, N 7,067, accuracy 80.94 %. Probe vs jedi_core: max |ΔS| = 0, max |Δg| = 0.
- (B) Arm 0 rerun: every particle of 2,000 test jets replaced by a random donor's, g held at the recipient's.
- (C) Arm 1 rerun: θ of S58, S31 fitted (on fit jets) by the E24x multipole chain (preg_k least squares on nested count-weighted sums m0, +m1, +m2, +m3, then ReLU, W3, b3), by MARS (≤ 8 hinges) and GBT on rotation-invariant jet observables; R² on test jets. Per-slot l3 check: slope of each window slot's pre-activation on the additive value ctx + b3.
- (D) g interchange, random donors, N 3,000: slope of actual ΔS on predicted ΔS (structural = exact θ; chain_exact = ReLU(preg) without add/l3 quantizers; named m0+m1 and m0+m1+m2), 1,000-jet bootstrap.
- (E) New causal test (missing in E24x): rigid edits applied to every real particle of the same 3,000 jets, full network re-run:
  - rot90 (Δη, Δφ) → (−Δφ, Δη): dipole rotated, both quadrupole sums change sign.
  - rot180 (Δη, Δφ) → (−Δη, −Δφ): dipole sums change sign, quadrupole sums $\sum r\cos2\alpha$, $\sum r\sin2\alpha$ unchanged (checked: change 1e-16 sd).
  - mirror_eta (Δη → −Δη), mirror_phi (Δφ → −Δφ).
  - pt_x2: every pT doubled, positions fixed (count sums unchanged; pT-weighted sums change).
  - Every rotation-invariant observable in E24x's set is unchanged by the four rigid edits (max relative change 0 for rot180 / mirrors; 0.016 for rot90 from float rounding of r).
  - Metrics: |ΔS|/sd(S) with 95 % bootstrap CI; slope, R² (no refit of slope) and correlation of actual on predicted ΔS; path calls (only S58, S31 set to the edited values).
  - H0 (competitor b of E24x: θ is a rotation-invariant scalar): ΔS = 0 under the rigid edits. H1 (multipole names): ΔS follows the named θ, slope 1. Pure-quadrupole test for S58: H0' |ΔS58|(rot180) = |ΔS58|(rot90) vs H1' |ΔS58|(rot180) ≈ 0 except for the m0+m1 terms of the fit (paired bootstrap). Holm over 6 slope tests (2 units × rot90, rot180, mirror_eta).
- (F) Untrained redraws (seeds 0, 1, 2): per-g preg R² by multipole block for the live unit among 58 / 31.

## Numbers (fresh seed; E24x values in brackets)
**Arm 0** (N 2,000): S58, S31 max |ΔS| = 0.0 [0.0]; other live S units changed in a median 98.2 % of jets [98 %].

**Arm 1, held-out θ R²** (test N 7,067)

| θ named by | S58 | S31 |
|---|---|---|
| multipole m0 / +m1 / +m2 | −1.16 / −1.15 / **0.933** [0.929] | −0.67 / **0.948** [0.951] / 0.966 |
| exact chain ReLU(preg) | 0.973 [0.972] | 0.988 [0.988] |
| MARS, rotation-invariant observables | 0.212 [0.217] | 0.233 [0.227] |
| GBT, rotation-invariant observables | 0.223 [0.221] | 0.267 [0.268] |

- S R²: multipole 0.790 [0.781, 0.799] (S58, m≤2) [0.784], 0.634 [0.615, 0.652] (S31, m≤1) [0.658]; exact-θ additive 0.853 / 0.645; chain 0.815 / 0.634; exact pre-activation 0.928 / 0.997.
- Per-g preg R²: S58 inputs m0+m1 0.04–0.23, +m2 0.984–0.991; S31 inputs m0 0.12–0.19, +m1 0.971–0.990. Largest standardised terms reproduce E24x's (g36 −Σ r sin2α, g44 −Σ r cos2α, g47 +Σ r sin2α, g63 +Σ r cos2α; g5 −ΣΔη, g29 +ΣΔη, g39 +ΣΔφ, g55 −ΣΔφ; each with −Σr).
- Per-slot l3 check (within the slot window): slope of slot pre-activation on ctx + b3 ranges 0.84–1.13 (median 1.04) for S58 and **0.18–1.08 (median 0.83) for S31**; within-jet sd of the pre-activation across slots 0.55 (S58) and 0.74 (S31) against sd(ctx + b3) 1.16 and 1.87.

**(D) g interchange, random donors** (N 3,000): S58 named m≤2 slope 0.996 [0.981, 1.010], R² 0.87; structural 1.010, R² 0.94. S31 named m≤1 slope 0.678 [0.669, 0.687], R² 0.71; structural 0.677, R² 0.74. (E24x random donor: 0.996 / 0.683.)

**(E) Rigid edits** (N 3,000; |ΔS|/sd with 95 % CI; slope [CI], R², corr of actual on predicted)

| edit | \|ΔS58\|/sd | S58 named m≤2 | S58 exact θ | \|ΔS31\|/sd | S31 named m≤1 | S31 exact θ | path calls |
|---|---|---|---|---|---|---|---|
| rot90 | 0.561 [0.544, 0.579] | 0.85 [0.82, 0.88], 0.54, 0.75 | 0.86, 0.75, 0.88 | 0.296 [0.285, 0.308] | 0.70 [0.65, 0.75], 0.20, 0.50 | 0.64, 0.44, 0.81 | 1.83 [1.37, 2.33] % |
| rot180 | 0.224 [0.212, 0.238] | 0.81 [0.70, 0.92], 0.10, 0.32 | 0.98, 0.72, 0.85 | 0.286 [0.275, 0.299] | 0.47 [0.42, 0.51], −0.05, 0.39 | 0.50, 0.02, 0.74 | 1.53 [1.10, 1.97] % |
| mirror_eta | 0.526 [0.510, 0.545] | 0.83 [0.80, 0.85], 0.54, 0.75 | 0.83, 0.75, 0.88 | 0.271 [0.259, 0.284] | 0.68 [0.62, 0.74], 0.25, 0.57 | 0.83, 0.80, 0.92 | 1.63 [1.20, 2.13] % |
| mirror_phi | 0.527 [0.510, 0.545] | 0.83 [0.81, 0.86], 0.53, 0.74 | 0.84, 0.75, 0.88 | 0.251 [0.240, 0.262] | 0.51 [0.45, 0.56], 0.01, 0.41 | 0.49, −0.05, 0.77 | 1.97 [1.50, 2.50] % |
| pt_x2 | 0.000 (all jets) | — | — | 0.022 [0.019, 0.025] (9.3 % of jets) | 0.45, −0.08, 0.38 | 0.57, 0.31, 0.87 | 0.17 % |

- Rotation-invariant θ (H0) predicts ΔS = 0 for the four rigid edits; actual |ΔS| is 0.22–0.56 sd (S58) and 0.25–0.30 sd (S31), all CIs exclude 0. Named-θ slope = 0 rejected in all 6 Holm tests (p 0.001 each, Holm 0.006).
- **Pure-quadrupole test (S58):** the named m≤2 θ predicts |ΔS58| 0.105 sd under rot180 (from its small m1 terms) against an actual 0.224 sd. Paired difference rot180 − rot90: −0.336 [−0.356, −0.316] sd, ratio 0.40, p 0.001. The exact θ predicts the rot180 change (slope 0.98, R² 0.72); the multipole name does not (R² 0.10). So 40 % of S58's 90°-rotation response survives a transform that leaves every quadrupole sum unchanged, and that part lies in the 7 % of θ variance outside the m≤2 fit (m3 adds 0.001 linearly).
- S31 contrast: rot180 / rot90 ratio 0.97 (p 0.13), as expected for dipole inputs.
- pt_x2: S58 unchanged in every jet (count-weighted, not pT-weighted, confirmed); S31 changes in 9.3 % of jets by 0.022 sd, via the soft-count (pT < 11 GeV ramp) term.

**(F) Untrained redraws** (θ R² m0 / +m1 / +m2; chain): seed 0 S58 0.12 / 0.30 / 0.32 (0.36); seed 1 S58 −0.60 / 0.19 / 0.28 (0.38); seed 2 S31 0.32 / 0.55 / 0.72 (0.73). Untrained preg m0 R² 0.05–0.72, m≤2 0.96–0.99. Reproduces E24x: low orders describe untrained inputs too, with mixed orders.

## Script review (interp/e24x_context_jedi.py, interp/e24x_plot.py)
- Stages, masks, quantizers: the probe reproduces jedi_core exactly (S and g) on both samples; W3 columns 58 and 31 have 4 non-zero rows, all on s channels that are 0 everywhere (max |own| = 0). Padding: since own ≡ 0, padded slots contribute the same u as real ones; the window (60 / 47 slots) is set by the l3 / qs1 quantizer masks, as stated. Fit/test split: overlap removed by hash. Arm-0 and arm-2 overrides use run_jetwise on the real network. No bug found that changes the numbers.
- Report arithmetic: "97–102 % of what the exact-θ additive form reaches (0.85, 0.66)" is 92 % for S58 against the additive form (0.784 / 0.851); 97 % is against the chain form (0.809). The node summary's "97–100 % of the exact-theta ceiling" has the same mix-up.
- Arm 3's g freeze holds all 64 g units (worker noted). Arm 3 tested single-particle edits only, with low per-jet R² (0.05–0.15); the rigid edits here are the stronger test of the names.
- Figure `results/jedi/e24x_context.png`: it reads at a glance and the numbers match the JSON. The dashed reference line is an R² of S (E24w half-plane fit), drawn against bars that are R² of θ, which is a different target. It is labelled "≤ 0.19", but this run's replica gives 0.201 for S31. The main competitor, θ from rotation-invariant observables (0.22–0.27), is not shown.

## Statistics summary
- H0 (rotation-invariant θ, ΔS = 0 under rigid edits): excluded, |ΔS| CIs exclude 0 in all 8 unit × edit cells; named-θ slope ≠ 0, Holm p 0.006 (6 tests).
- H0' (S58 responds to rot180 as to rot90) vs H1' (pure quadrupole, rot180 response ≈ named m1 part only): both excluded. Response ratio 0.40 [paired diff CI −0.356, −0.316], p 0.001; actual rot180 response 2.1 × the named prediction.
- Competing names: rotation-invariant scalars (excluded, above); multipole names (supported for θ variance on natural jets, R² 0.93 / 0.95; partial for edit responses, R² 0.53–0.54 for S58 under rot90 and mirrors, ≤ 0.25 for S31); exact θ additive (R² 0.72–0.75 for S58; −0.05 to 0.80 for S31).
- Calibration: the named θ was compared against the exact θ on the same edits (matched real change); slopes of the named form (0.81–0.85 S58) are within 0.17 of the exact form's (0.83–0.98).

## Corrected reading
S58 and S31 read no per-particle input (confirmed, fresh seed). Their thresholds are functions of g whose inputs are well fitted by count-weighted (not pT-weighted) fixed-frame sums: S31's by ± count dipoles (θ R² 0.95), S58's by ± count quadrupoles (θ R² 0.93); rotation-invariant observables reach ≤ 0.27, and rigid rotations / mirrors change both units, which excludes a rotation-invariant threshold. The quadrupole name is not complete for S58's causal response: a 180° rotation, which leaves all quadrupole sums unchanged, moves S58 by 0.22 sd (40 % of the 90° response), and the named θ explains only 10 % of that variance (exact θ 72 %). For S31 the dipole name describes θ, but the per-slot l3 quantizer (slot slopes 0.18–1.08) shapes S31's response so that even the exact θ predicts its rigid-edit changes with slope 0.49–0.83 and R² −0.05 to 0.80. The S-level shortfall of S58 (0.79 vs exact-θ additive 0.85) is partly naming and partly quantization, not only per-slot quantization.

## Explainer entry (for site/explain.py)
```python
E24X_VERIFY_EXPLAINER = [
 ('Question', 'Do the multipole names of S58 (count quadrupole) and S31 (count dipole) hold on a fresh sample, and do they predict what happens when the whole jet is rotated or mirrored?', None),
 ('Background', 'A count dipole $\\sum_i\\Delta\\eta_i$, $\\sum_i\\Delta\\phi_i$ changes sign under a $180^\\circ$ rotation $(\\Delta\\eta,\\Delta\\phi)\\to(-\\Delta\\eta,-\\Delta\\phi)$; a count quadrupole $\\sum_i r_i\\cos2\\alpha_i$, $\\sum_i r_i\\sin2\\alpha_i$ is unchanged by it and changes sign under a $90^\\circ$ rotation. Rotation-invariant quantities ($\\sum p_T$, $p_T^{\\rm lead}$, $M_2$) are unchanged by all rotations and mirrors.', None),
 ('Method', 'Fit sample(3000, 4101), test sample(1500, 4102) minus overlap ($N$ 7,067). Refit the E24x multipole chain; then rotate (90°, 180°) or mirror every particle of 3,000 test jets, rerun the network, and compare the actual $\\Delta S$ with the change predicted by the named threshold and by the exact threshold $\\theta=-(W_3\\cdot q(g)+b_3)$. Paired bootstrap for $|\\Delta S_{58}|$ at 180° vs 90°; Holm over 6 slope tests.', None),
 ('Result', 'Fresh-seed $\\theta$ $R^2$: 0.93 (S58, $m\\le2$), 0.95 (S31, $m\\le1$); rotation-invariant fits 0.21–0.27. Rotations and mirrors change S58 by 0.22–0.56 sd and S31 by 0.25–0.30 sd. A 180° turn still moves S58 by 40 % of its 90° response, which a pure quadrupole cannot do; the named $\\theta$ explains 10 % of that change, the exact $\\theta$ 72 %.', None),
 ('Checks', 'Probe identical to jedi_core; no per-particle input (particle swap with $g$ held: $\\Delta S=0$); doubling every $p_T$ leaves S58 unchanged in every jet. S31 per-slot quantizer slopes 0.18–1.08 limit how well any jet-level $\\theta$ predicts its edit response.', None),
 ('Why it matters', 'The names locate S58 and S31 among fixed-frame count shapes rather than rotation-invariant jet quantities, consistent with the azimuth-dependent context response (A13). The quadrupole name for S58 is a description of most of its threshold variance, not a complete causal account.', None),
]
```
