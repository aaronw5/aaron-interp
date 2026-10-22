# P56v: verification of P56 (invariant mass vs spread about L-GATr's own momentum axis)

Script `interp/p56v_verify.py`; job c-p56-verify (GPU, Tesla V100, 23 min, Complete); results `results/part/p56_verify.json`
(cluster: `/j-jepa-vol/aaron-interp/results/part/p56_verify.json`).

## Conclusion

On a fresh set of 4,000 W and Z jets, L-GATr's W-vs-Z score again moved less when the jet mass was raised by making particles heavier than when the same mass rise came from opening the angles (a quarter of the angle-made response here, a third in P56). A new control shows why this ratio cannot be read as "how much L-GATr follows the invariant mass": any change that pushes particles off their physical mass, in either direction and at fixed jet mass, moves the score towards W by about half of a real mass change, and this push works against the mass response in the heavier-particle edit. The part of P56 that said a larger spread at equal mass acts only through the per-particle angle inputs did not reproduce: on the new jets the four-vector route also carried a clear share. What still holds is that the angle inputs carry a fixed share of about 0.13 of a real mass change; how much L-GATr reads the invariant mass itself is not measured by this test.

## What was checked

**Code read** (`interp/p56_lgatr_mass_vs_spread.py`, `interp/part_core.py` build/kin, `interp/p44_lgatr_grades.py` LG, `interp/p5_lgatr_launch.py` to_lgatr_inputs):
- Rows: random `default_rng(5656).choice` per file, sorted; no first-k. W/Z arm mixes WToQQ and ZToQQ files, so no class-ordered subsampling.
- η build: `geom` and `mk` use η = jet_eta + sign(jet_eta)·Δη, the corrected convention; `mk` reorders the float64 energies with the same stable pT argsort as `build`, so energies stay attached to the right particles.
- Padded slots: `rm = pt > 0` masks every sum; L-GATr input multiplies four-vectors and features by the mask.
- Edits change what they claim: the stored geometry metadata show Δlog M2_own = 0 exactly for M, N, ZERO, PION; Δlog m matches the target to 2·10⁻⁴ in float32; N keeps Σ ΔE = 0 to 10⁻¹³.
- Routes: D = `lg.logits(edited, unedited)` (four-vectors from the edited jet, the 7 scalar features (log pT, log E, log pT/pT_J, log E/E_J, Δη, Δφ, ΔR about the own axis) from the unedited jet); F the reverse. The embedding asserts identical token structure.
- No coding error was found that changes the numbers.
- Figure: panel (c) labels the M arm "m +2–10 %" but the pooled arm covers +1 to +10 %; PION and ParT_kin points are clipped at the axis edges (stated in the report). Panels (a) and (b) show the edits on a real W jet as described.

**Rerun on fresh rows.** JetClass test_20M WToQQ_117 / ZToQQ_117, 2,000 rows per file drawn with `default_rng(5657)` from the rows P56 did not use (P56's draw regenerated and excluded; overlap 0). N = 4,000 jets, all with ≥ 3 particles. Model: L-GATr final (model_run0.pt). Edits: P56's own `edits()` generator, unchanged. Unedited accuracy on this sample 71.65 % (P56 sample 71.7 %).

**Controls added.**
- **NF (sign-flipped energy shuffle).** E_i(NF) = E0_i − (E_i(N) − E0_i): every particle that N made heavier is made lighter by the same amount and vice versa. Same |ΔE_i| as N (and as M), Σ ΔE = 0, Σ p unchanged, so the jet invariant mass m and every particle direction are fixed (max |Δlog m| 2·10⁻⁴ from float32). 46 % of particles end with E < |p| (as in N); 2 of about 160,000 particles get E ≤ 0 at +10 %.
  - Why this control: P56 read R_N ≠ 0 as "the four-vector route responds to individual particle masses". If the response were linear in the signed energy changes, NF would give the opposite response to N (antisymmetric). If it is a response to particles being pushed off their mass shell regardless of direction, NF gives the same response as N (symmetric). sym = ½(Δd_N + Δd_NF), anti = ½(Δd_N − Δd_NF).
- **Rerun noise.** The unedited jets run three times through L-GATr on GPU.

**Metric.** d = ℓ_W − ℓ_Z (difference of the W and Z logits). R = Σ_jets Δd(edit) / Σ_jets Δd(E4 with the same Δm), where E4 is the own-axis dilation: every particle's (Δη, Δφ) about the summed-momentum axis scaled by a per-jet factor until m reaches the same target. R = 1: the score moves as under a real, angle-made mass change of the same size; R = 0: no response. 2,000 bootstrap resamples over jets; bootstrap p-values.

## Numbers (P56 → fresh rows; 95 % bootstrap intervals)

| quantity | P56 (seed 5656) | P56v (seed 5657, disjoint rows) |
|---|---|---|
| R_M pooled (heavier particles, directions fixed), route A | 0.356 [0.283, 0.430] | 0.252 [0.174, 0.330] |
| R_M by dose +1 / +2 / +5 / +10 % | 0.17 / 0.43 / 0.51 / 0.28 | 0.02 / 0.27 / 0.39 / 0.20 |
| mean Δd, M +5 % vs M +10 % | −0.255 / −0.256 | −0.187 / −0.176 |
| R_M route D (four-vectors) / F (features) | 0.361 / −0.003 | 0.254 / −0.002 |
| R_N pooled (energy shuffle, m fixed) | −0.447 [−0.570, −0.330] | −0.561 [−0.690, −0.442] |
| R_NF pooled (sign-flipped shuffle, m fixed) | not run | −0.537 [−0.668, −0.408] |
| sym part ½(N + NF) | not run | −0.549 [−0.675, −0.427], p 0.0005 |
| anti part ½(N − NF) | not run | −0.012 [−0.042, 0.018], p 0.44 |
| X5 − M5 (spread up at equal m), route A | 0.177 [0.081, 0.267] | 0.348 [0.249, 0.444] |
| X5 − M5, route D | 0.048 [−0.046, 0.137] | 0.221 [0.123, 0.315], p(=0) 0.0005 |
| X5 − M5, route F | 0.131 [0.124, 0.137] | 0.133 [0.127, 0.139] |
| exact S pooled (spread up, other masses lowered, m fixed) | 0.240 [−0.19, 0.66], N 1,140 | −0.052 [−0.55, 0.44], N 1,123; p(R_S = 0) 0.83 |
| S route F | 0.141 | 0.124 [0.103, 0.143] |
| X pooled +1–10 %, A / D / F | (X+5 %: 0.683 / 0.558 / 0.128) | 0.687 / 0.588 / 0.134 |
| E4 route shares D / F | 0.78 / 0.23 | 0.78 / 0.23 |
| ZERO / PION | 1.55 / 4.98 | 1.67 / 6.86 |
| calls at M +5 %: changed, true-W → Z | 29.1 %, 11.3 → 15.8 % | 29.6 %, 12.5 → 17.3 % (lost / gained correct 761 / 138, McNemar p 5·10⁻¹⁰⁵) |
| calls at E4 +5 %: changed, true-W → Z | 16.6 %, 11.3 → 23.9 % | 16.0 %, 12.5 → 23.9 % |
| calls at NF +5 % | not run | 35.9 % changed, true-W → Z 12.5 → 17.4 % |

Rerun noise (identical input, GPU): per-jet Δd mean −5·10⁻⁵, sd 0.0057, max 0.156, 2.7 % of jets above 0.01, 0.05 % of calls changed. As an R contribution it is below 5·10⁻⁴ at every dose, so P56's "rerun floor 0.30" (a per-jet maximum) does not affect the pooled ratios.

## Reading against P56's statements

1. "L-GATr's W-vs-Z read-out moves under a heavier-particle mass rise at 0.36 of the angle-made move, entirely through the four-vectors (route D)." Reproduced in direction and route (D carries all of it, F ≈ 0). The size is 0.25 [0.17, 0.33] on disjoint rows; the two intervals overlap only in 0.28–0.33, so the between-sample spread is larger than either bootstrap interval suggests. The response does not grow from +5 % to +10 % on either sample (mean Δd −0.19 vs −0.18 here).
2. "The four-vector route also responds to individual particle masses at fixed jet mass (R_N −0.45)." Reproduced (−0.56). The NF control shows the response is symmetric in the sign of the energy changes: making a given set of particles heavier and the rest lighter, or the reverse, moves d by the same amount (anti part −0.01 [−0.04, 0.02]). The four-vector route therefore responds to particles being moved off their physical mass shell, in either direction, by pushing d towards W (positive Δd), not to the signed particle masses.
   - Consequence for item 1: the M edit also moves every particle off its mass shell (median added mass μ = 0.43 GeV at +5 %, above the pion mass in > 99.9 % of jets). If M carries a push of the same kind, its sign (towards W) is opposite to the mass response (towards Z), so R_M = 0.25–0.36 is the net of the two and is not an estimate of an invariant-mass share. The saturation from +5 % to +10 % fits a growing opposing term (|Δd_N| grows from 0.17 to 0.57 between the same doses), but the size of the push inside M is not measured here: M produces no spacelike particles, N and NF do.
3. "At equal jet mass, a larger spread about the own axis moves the read-out by 0.18 of a real mass change, through the angle features only (route F 0.13; route D 0.05, consistent with 0)." Not reproduced as stated. On disjoint rows X5 − M5 is 0.35 and route D carries 0.22 [0.12, 0.32] (p 0.0005). Route F 0.13 reproduces closely. X5 − M5 uses the off-shell M5 jet as its equal-mass reference, so its route-D part inherits the off-shell push of item 2; the exact S edit (feasible in 28 % of jets at +1 %) gives values consistent with 0 on both samples (0.24 and −0.05, intervals ±0.45).
4. "These particle-mass edits are distinguishable from real particles (C2ST 0.99)." Not rerun; geometry metadata and μ distribution reproduce (μ median 0.43 GeV at +5 %).
5. ParT_full / ParticleNet_full references were not rerun (L-GATr is the subject of the reading).

Competing hypotheses for the M result:
- H_inv-partial (L-GATr reads m with gain 0.25–0.36 through the four-vectors): predicts R_N ≈ R_NF ≈ 0. Excluded for the four-vector route as a whole (R_N, R_NF ≈ −0.55).
- H_signed-particle-mass (route D reads signed per-particle masses/energies): predicts R_NF ≈ −R_N. Excluded (anti part 0.0 ± 0.03; sym part −0.55).
- H_offshell (route D reads m plus a sign-symmetric response to off-shell particles): predicts R_N ≈ R_NF < 0 and an M response reduced and saturating with dose. Not excluded by any number here; the gain on m under this hypothesis is not identified without an on-shell way to change m at fixed directions.

## Issues

- The P56 rows and the verification rows come from the same two files (_117); the difference between samples (R_M 0.36 vs 0.25, X5 − M5 0.18 vs 0.35) exceeds what the bootstrap intervals over jets imply, so those intervals understate sample-to-sample variation for these ratios of sums.
- N and NF both make about 46 % of particles spacelike (E < |p|), M makes none, so the off-shell push measured by N/NF is not a quantitative correction for M.
- Figure panel (c) label "m +2–10 %" should read "+1–10 %".

## Corrected reading

L-GATr's W-vs-Z read-out moves under a heavier-particle mass rise at fixed directions by 0.25–0.36 of an angle-made rise of the same size (two disjoint samples), entirely through its four-vector inputs. The same four-vector route moves the read-out towards W by about 0.55 of a real mass change whenever particle energies are moved off their mass shell at fixed jet mass, symmetrically in the sign of the change (N and sign-flipped NF agree to 0.01), so the heavier-particle ratio mixes a mass response with an opposing off-shell response and does not measure an invariant-mass share. At equal jet mass, the per-particle angle features carry 0.13 of a real mass change from the larger spread on both samples; the four-vector route's share of the spread contrast is not stable (0.05 and 0.22), and the exact fixed-mass spread edit is consistent with 0.

## Conclusion (corrected for P56)

L-GATr's W-vs-Z score moved by a quarter to a third as much when the jet mass was raised through heavier particles as when the same rise came from wider angles, and all of that came through its four-vector inputs. Any change that pushes particles off their physical mass, at fixed jet mass and in either direction, moves the score the other way by about half of a real mass change, so this ratio mixes two opposing effects and does not tell how much L-GATr reads the invariant mass. A wider spread at equal mass reaches the score through the per-particle angle inputs with a stable share of about 0.13; whether the four-vector inputs also respond to the spread differed between two samples. An edit that changes the jet mass while keeping every particle at a physical mass is still needed.
