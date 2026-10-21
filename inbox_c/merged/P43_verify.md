# P43v: verification of P43 (spread about the stored axis vs centroid offset)

Script `interp/p43v_verify.py`. Job `c-p43-verify` (GPU, smoke + full, Complete). Results in `results/part/p43_verify.json`
and log in `logs/c-p43-verify.log`. The P43 pair rule and the edits E1–E5 are imported unchanged from
`interp/p43_spread_vs_centroid.py`.

## Conclusion

On a fresh sample, the first half of P43 reproduces. Moving two hard particles apart or together, so that the jet's angular
spread changes while its pT-weighted centre stays fixed, moves ParT_full's W/Z output by about as much as a real mass change of
the same size (ratio 0.97). The second half does not hold. P43's ratio of about 0.3 for a shift of the jet's centre off the
stored axis was measured on jets whose two hardest particles had already been moved by 0.03–0.13 rad. Applied to unedited jets,
the same rigid shift gives ParT_full about 0.75 of a real mass change (random direction), flat from 0.005 to 0.06 rad. So for
ParT the offset is down-weighted only mildly and does not weaken with size. ParticleNet_full does down-weight the offset
(0.2–0.7). Its weight depends on the direction of the shift and falls at larger shifts, so no single offset weight describes
that network.

## Definitions

- **Stored axis**: the jet (η, φ) stored in JetClass. The networks receive each particle's Δη, Δφ and ΔR measured from it.
- M0 = Σ pT. **c** = Σ pT (Δη, Δφ) / M0 is the pT-weighted centroid about the stored axis. **M2** = Σ pT r² is the spread
  about the stored axis. **M2c** = M2 − M0|c|² is the spread about the centroid. M0|c|² is the **offset term**.
- **d** = logit_Z − logit_W, the read-out used in P43 and P39v.
- **R(edit)** = Σ_jets Δd(edit) / Σ_jets Δd(E4 matched). It is the network's response to the edit divided by its response to a
  real angular dilation that changes log M2 by the same amount on the same jets. A dilation multiplies every particle's
  (Δη, Δφ) by one factor k, so it is a real change of jet mass.
- P43 edits, unchanged here:
  - E1: two similar-pT particles move apart or together along their joining line, with c fixed.
  - E2: the pair is translated so that c moves by δ, then an E1-type move restores M2c.
  - E3: E2 with the stored axis moved along with the centroid. The lab-frame particles are those of E2, and c and M2 are those
    of the unedited jet.
  - E5: the same displacement sizes as E1, in the direction that keeps c and M2 fixed.
  - **E2 − E3**: P43's offset estimate.
- **E6 (new, this verification): rigid offset shift on the unedited jet.** Every real particle's (Δη, Δφ) is increased by
  δ·e, and the stored axis moves by −δ·e. The lab-frame four-vectors, invariant mass and pairwise features (kT, z, ΔR, m²)
  therefore stay exactly unchanged. M2c is unchanged, and M2 changes by M0(|c + δe|² − |c|²), the offset term only.
  - E6e uses the P43 E2 direction e of each jet (a unit vector from the pair's pT-weighted barycentre toward the centroid). This
    is the same direction and the same ΔM2 as E2 − E3 (checked: max |Δlog M2 difference| = 0).
  - E6r uses a uniformly random direction.
  - δ ∈ {0.005, 0.01, 0.02, 0.04, 0.06} rad.
- **Holm**: step-down family-wise correction over the 4 trained models.

## Code review of `p43_spread_vs_centroid.py`

- Rows are drawn at random (`read_rand`, numpy default_rng, sorted random indices), not first-k. Padded slots have pT = 0 and
  carry no weight in the moments. The E4 dilation masks padded slots. `build` uses the η = jet_eta + sign(jet_eta)·Δη
  convention (no G192 mirror). Models are the released ParT and ParticleNet weights through the P39v runner. No hooks are
  involved.
- The edits do what they state. In the P43 JSON, E1 |Δc| = 0, E2 Δlog M2c = 0.0000, and E3 has c and M2 unchanged with lab
  m_inv equal to E2 to within 2.5·10⁻⁴.
- Minor: E3 shifts jet_eta using the original sign of jet_eta, while `build` takes the sign from the shifted jet_eta. For jets
  with |jet_eta| < δ, about 1 % or fewer, E3 is therefore a mirrored jet rather than E2's lab jet. This check cannot detect the
  mismatch, because m_inv is mirror-invariant. E6 in this verification avoids the sign change: such jets keep jet_eta and
  are shifted rigidly in lab η instead (35 of 4,502 kept jets at δ = 0.04; their max |Δlog m_inv| is 5·10⁻⁴).
- **Main problem (design):** E2 − E3 equals the response to a rigid shift by δ of all axis-relative coordinates. P43 measures
  it on the E3 jet, not on the original jet. The E3 jet already has its two hardest similar-pT particles moved by 0.03–0.13 rad.
  E3 alone changes 46 % of ParT_full's calls (P43 JSON), and its mean per-jet |Δd| is 2.7× that of the matched dilation. The
  denominator (E4) is measured on the unedited jet. The ratio therefore mixes the offset weight with how the network responds
  to a rigid shift on an already strongly deformed jet. P43 had no control for this, and E6 is that control.
- Figure (`p43_spread_vs_centroid.png`): the two jet panels show the edits correctly, with the centroid fixed under E1 and moved
  0.04 rad under E2 at fixed M2c. The bar panel shows raw E2, not the E2 − E3 contrast that the reading is based on. The headline
  offset number is therefore not in the figure.

## Procedure

- **Sample.** JetClass val_5M `WToQQ_121.root` and `ZToQQ_121.root` (P43 used `_120`), 10,000 random rows each, numpy
  default_rng(4344). The same admissible-pair rule as P43 keeps 4,502 of 20,000 jets (P43: 4,592). The 15,498 dropped jets get
  E6r only, because they have no admissible pair.
- **Models.** Released ParT_full, ParT_kin, ParticleNet_full (PN_full) and ParticleNet_kin (PN_kin).
- **Edits.** Kept jets get E1 (±2/5/10 %), E2 and E3 (δ = 0.01/0.02/0.04), E5±, E6e and E6r (5 doses), each with a per-jet
  matched E4 dilation.
- **Statistics.** 2,000 paired bootstrap resamples over jets, shared across edits. Two-sided bootstrap p values have a
  resolution of 0.0005. A sign-flip permutation test is used for E6e = 0. Holm correction is applied over the 4 models.
- **Hypotheses.**
  - H0: (E2 − E3) = E6e, i.e. P43's offset estimate equals the clean rigid-shift response with the same direction and ΔM2.
  - H1: they differ, i.e. the offset estimate depends on the prior two-particle deformation.
  - Second test: R(E1 raise) = R(E6e pooled over 0.01–0.04).
- **Predictions.**
  - "Reads M2 about the stored axis" (C18): E1 ≈ E6 ≈ 1.
  - "Reads M2c plus a down-weighted offset at about 0.3" (P43 reading): E6 ≈ 0.3.
  - "Reads M2c only": E6 ≈ 0.

## Numbers

Unedited 10-class accuracy:

| set | ParT_full | ParT_kin | PN_full | PN_kin |
|---|---|---|---|---|
| all 20,000 W/Z jets | 74.6 % | 63.4 % | 72.1 % | 60.6 % |
| 4,502 kept | 81.7 % | 70.8 % | 79.3 % | 69.2 % |
| 15,498 dropped | 72.6 % | 61.2 % | 70.1 % | 58.0 % |

These match P43 (kept 80.5 / 70.8 / 78.2 / 68.5 %).

P43 quantities on the fresh sample (R, 95 % CI):

| model | E1 raise | E1 lower | E2 − E3 (P43 offset) | E5 null |
|---|---|---|---|---|
| ParT_full | 0.97 [0.92, 1.02] | 0.86 [0.80, 0.91] | 0.34 [0.31, 0.37] | 0.10 [0.06, 0.14] |
| ParT_kin | 1.17 [1.13, 1.22] | 0.55 [0.50, 0.59] | 0.38 [0.33, 0.43] | 0.41 [0.37, 0.45] |
| PN_full | 0.92 [0.87, 0.96] | 0.95 [0.91, 1.00] | 0.32 [0.29, 0.36] | 0.00 [−0.04, 0.04] |
| PN_kin | 0.84 [0.81, 0.86] | 0.59 [0.57, 0.62] | −0.19 [−0.22, −0.16] | 0.12 [0.10, 0.14] |

All P43 values reproduce to within about 0.07, with two differences:
- ParT_full E1 lower is 0.86 here, against 0.92 in P43. The raise and lower intervals do not overlap, so the "same for raising
  and lowering" statement is marginal on this sample.
- ParT_full E5 is 0.10, against 0.03 in P43.

New control, E6 rigid offset shift on the unedited jet (R, 95 % CI), kept jets:

| δ (rad) | ParT_full E6r | ParT_full E6e | PN_full E6r | PN_full E6e | ParT_kin E6r | PN_kin E6r |
|---|---|---|---|---|---|---|
| 0.005 | 0.72 [0.51, 0.92] | 2.49 [2.16, 2.87] | 0.73 [0.46, 1.00] | −0.03 [−0.39, 0.31] | unresolved | 0.85 [0.66, 1.03] |
| 0.01 | 0.73 [0.63, 0.82] | 1.52 [1.40, 1.63] | 0.70 [0.58, 0.82] | 0.25 [0.10, 0.40] | 0.80 [0.47, 1.34] | 0.81 [0.73, 0.89] |
| 0.02 | 0.74 [0.70, 0.78] | 1.07 [1.02, 1.12] | 0.66 [0.60, 0.71] | 0.29 [0.22, 0.35] | 0.77 [0.68, 0.87] | 0.75 [0.71, 0.79] |
| 0.04 | 0.73 [0.71, 0.75] | 0.84 [0.81, 0.86] | 0.59 [0.56, 0.61] | 0.17 [0.14, 0.21] | 0.73 [0.70, 0.76] | 0.57 [0.54, 0.59] |
| 0.06 | 0.76 [0.75, 0.78] | 0.79 [0.77, 0.81] | 0.47 [0.45, 0.49] | 0.07 [0.04, 0.09] | 0.73 [0.72, 0.75] | 0.36 [0.34, 0.38] |
| pooled 0.01–0.04 | 0.73 [0.71, 0.76] | 0.91 [0.88, 0.94] | 0.61 [0.57, 0.64] | 0.20 [0.16, 0.24] | 0.74 [0.70, 0.78] | 0.61 [0.59, 0.64] |

On the dropped jets (no admissible pair), E6r pooled over 0.01–0.04 gives ParT_full 0.80 [0.78, 0.81], ParT_kin 0.79, PN_full
0.72 and PN_kin 0.66. ParT_full stays flat at 0.78–0.82 from 0.005 to 0.06 rad. PN_full falls from 0.82 to 0.56.

Geometry checks:
- E6: Δlog M2c = 0 (≤ 2·10⁻⁹); the realised Δlog M2 equals the planned value.
- Max |Δlog m_inv| ≤ 8·10⁻⁴; the only nonzero values come from the eta-sign jets, which are shifted in lab η.

Paired tests (bootstrap, Holm over 4 models):

| model | (E2 − E3) − E6e, pooled 0.01–0.04 | Holm p | E1 raise − E6e pooled | Holm p |
|---|---|---|---|---|
| ParT_full | −0.58 [−0.61, −0.54] | 0.002 | +0.06 [−0.00, 0.12] | 0.057 |
| ParT_kin | −0.61 [−0.66, −0.55] | 0.002 | +0.19 [0.12, 0.26] | 0.002 |
| PN_full | +0.13 [0.08, 0.17] | 0.002 | +0.72 [0.66, 0.78] | 0.002 |
| PN_kin | −0.41 [−0.45, −0.38] | 0.002 | +0.61 [0.56, 0.65] | 0.002 |

The ParT_full per-dose values of (E2 − E3) − E6e are −0.37, −0.62 and −0.58 at δ = 0.01, 0.02 and 0.04 (each p 0.0005).

W → Z rate on true W, ParT_full, kept jets (baseline 9.5 %):

| edit | W → Z rate | matched E4 |
|---|---|---|
| E6r δ = 0.04 | 16.3 % | 18.9 % |
| E6r δ = 0.06 | 28.7 % | 36.8 % |

E6r at δ = 0.06 changes 19.6 % of calls, consistent in size with P1's 0.06 rad φ shift (12 → 31 %).

## Reading

- E1, the spread edit at fixed centroid, reproduces. ParT_full and PN_full respond at 0.92–0.97 of a matched dilation when the
  two particles are pushed apart. Under lowering, ParT_full gives 0.86 [0.80, 0.91] on this sample (P43: 0.92).
- The E2 − E3 value of about 0.3 also reproduces numerically (0.34 for ParT_full, 0.32 for PN_full). However, the same rigid
  shift on unedited jets (E6e, same direction and same ΔM2) gives 0.91 for ParT_full. The difference is −0.58 [−0.61, −0.54]
  (Holm p 0.002). For ParT_full, then, P43's offset estimate is mostly set by the prior relocation of the two hardest particles
  (E3), not by the offset term.
- Clean offset response, ParT_full:
  - Random directions give 0.73 [0.71, 0.76] on kept jets and 0.80 on dropped jets, flat from 0.005 to 0.06 rad. This matches
    P39v's 0.71–0.84 for M2-raising rigid shifts.
  - In the E2 direction the ratio is higher at small δ (2.5 at 0.005, 1.5 at 0.01). Here the network's response has a part
    linear in δ, while ΔM2 is mostly quadratic.
  - E1 raise − E6e pooled is +0.06 [−0.00, 0.12], Holm p 0.057. At these doses the offset and spread terms are not separated in
    the E2 direction. In random directions the offset counts about 0.75 relative to the spread's 0.97.
- The P43 suggestion that the offset weight falls from about 0.75 at 0.006 rad to about 0.3 at 0.02–0.04 rad is excluded for
  ParT_full. The random-direction E6r is 0.72–0.76 at every dose.
- PN_full does down-weight the offset. Random directions give 0.61 [0.57, 0.64], with E1 raise − E6e = +0.72, Holm p 0.002.
  - The weight depends on direction: 0.20 in the E2 direction against 0.61 in random directions.
  - It falls with dose: 0.73 at 0.005 rad and 0.47 at 0.06 rad (kept jets); 0.82 and 0.56 on dropped jets.
  - For PN_full, P43's E2 − E3 value of 0.32 lies between the two direction-specific values.
- Kin models:
  - ParT_kin's E6r is about 0.74, flat in dose.
  - PN_kin's E6r falls from 0.85 to 0.36 with dose.
  - Both kin models also respond to E5 (0.12–0.41), as in P43.
- Competing hypotheses for ParT_full:
  - "Offset weight ≈ 0.3" is excluded by E6.
  - "Reads M2c only" is excluded (E6 = 0.73, p 0.0002 against 0).
  - "Reads M2 about the stored axis with one weight" is excluded for random directions (0.73 vs 0.97). In the E2 direction it
    is not excluded (0.91, Holm p 0.057).

## Corrected reading

For ParT_full and PN_full, changing the spread about a fixed centroid with two particles gives 0.92–0.97 of a matched real
dilation (reproduced). P43's offset estimate of 0.31 (E2 − E3) was measured on jets whose two hardest particles had already been
moved 0.03–0.13 rad. On unedited jets, a rigid offset shift with the same direction and ΔM2 gives ParT_full 0.91 (E2 direction)
and 0.73 (random directions, flat over 0.005–0.06 rad). For ParT_full the offset is therefore weighted at about 0.75–0.9 of the
spread, and no dose dependence is seen. PN_full weights the offset at 0.2–0.7, depending on direction, and lower at larger
offsets.

## Suggested explainer and glossary additions

- **E6 rigid offset shift**: all particles' axis-relative coordinates are shifted by δ·e while the lab-frame particles are
  unchanged (the stored axis moves by −δ·e). This changes only the offset term $M_0|\vec c|^2$ of
  $M_2 = M_{2c} + M_0|\vec c|^2$.
- **Explainer, Checks section**: the offset estimate has to be measured on the unedited jet. Measured on a jet already deformed
  by the compensating two-particle move (E3), ParT's response is 0.3 instead of 0.75–0.9.

## Issues / scope

- Job note: the first two pods of `c-p43-verify` failed at admission ("GPU is lost") on nodes k8s-haosu-12 and k8s-chase-ci-01.
  The job was deleted and resubmitted under the same name from a scratch copy of `part_job.sh` that also excludes k8s-haosu-12.
  The repo script was not changed.
- E6e at δ ≤ 0.01 has small denominators (Δlog M2 ≤ 0.005). Its values above 1 there reflect a response linear in δ against
  an M2 change that is mostly quadratic.
- Only W/Z jets and the read-out d were used. The W/Z-vs-QCD score was not re-run.
