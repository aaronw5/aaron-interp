# P36 verification: natural pT-centroid offset, recentring at fixed invariant mass (agent report, 2026-10-20)

Status: **done**. Job `anrunw-ai-c-p36-verify` (part_job.sh, GPU; smoke on 200 jets per class and the full run chained in one job, 457 s for the full run). Script `interp/p36v_verify.py`. Results `results/part/p36_verify.json` (per-jet logits in `/j-jepa-vol/aaron-interp/results/part/p36_verify_jets.npz`). Log `logs/c-p36-verify.log`.

## Conclusion
On a new, larger set of jets, the main P36 number holds. When every particle of a jet is shifted so that the pT-weighted centre of its particles sits on the stored jet axis, ParT's W-vs-Z score moves about as much as for a real mass change of the same size in the axis moment (ratio about 1.0 on the 5 % most off-centre W and Z jets). That means ParT measures the jet's spread about the stored axis rather than about the particles' own centre. The recentred jets are not natural jets, though. In real jets the stored axis is the direction of the total momentum to within 0.001 rad, and recentring moves that direction about 0.03 rad away from it. So this is a small artificial axis shift, of natural size and direction, and not a test on unedited jets. Two statements in the reading do not hold up. "Same as a random shift" holds on all W/Z jets but not on the most off-centre 5 %. The rotation control fails partly for another reason: in the full-input models, even an exact mirror image of the jet in φ changes 10–15 % of calls.

## What was checked
**Code review of `interp/p36_natural_offset.py`.**
- No bug found that changes the numbers.
- Rows are random (`default_rng`, sorted), never first-k.
- `build` uses $\eta = \eta_{jet} + \mathrm{sign}(\eta_{jet})\,\Delta\eta$, which is the corrected η convention.
- Padded slots are excluded everywhere: kin masks them, the edits multiply by $r = [p_T > 0]$, and the moments are pT-weighted.
- The read-out is the logits.
- The hook on `fc` reads the input of the final layer for both ParT and ParticleNet.
- $M_2^{rec} = M_2 - |\vec c|^2 M_0$ follows from $\vec c\cdot\vec M_1 = |\vec c|^2 M_0$.
- κ, R and Holm are implemented as described.
- The figure (`results/part/p36_natural_offset.png`) matches the text. The bar panel shows tail2, while the report tables show tail5; the numbers agree.

**Fresh rerun.**
- Sample: JetClass test_20M `<class>_119.root` (P36 used `_118`), numpy `default_rng(3690)`, random rows.
- 8,000 Wqq and 8,000 Zqq jets, plus 1,000 jets of each other class, for N = 24,000.
- So tail5 holds 800 true W/Z jets (P36: 200), tail2 320 and tail1 160.
- Models: ParT_full, ParT_kin, ParticleNet_full (PN_full), ParticleNet_kin (PN_kin).
- L-GATr and the untrained ParT were **not** rerun.
- Edits, tails, κ and $R$ are as in P36, with the same code. Symbols:
  - $\vec c$: the pT-weighted mean $(\Delta\eta, \Delta\phi)$ of the particles about the stored axis.
  - $M_0 = \sum p_T$ and $M_2 = \sum p_T r^2$.
  - κ: the slope of the read-out change against the predicted $\tfrac12\Delta\log M_2$, pooled over the matched dilations. A matched dilation scales all angles to give a real mass change of the same $\tfrac12\Delta\log M_2$.
  - $R(e) = \overline{d(e)}/(\kappa\,\overline{\tfrac12\Delta\log M_2(e)})$: the network's response to edit $e$ divided by its response to a real mass change of the same predicted size. R = 1 is the same response; R = 0 is no response.
- Statistics: bootstrap over jets (2,000), sign-flip permutation (5,000), exact McNemar on per-jet call changes, Holm over the 4 models.

**Unedited accuracy.**
- P1 sample (rows 3,000–4,000 of `_100`, N = 10,000): ParT_full 85.93 %, ParT_kin 74.46 %, PN_full 84.57 %, PN_kin 70.35 %. This reproduces the known values.
- This W/Z-enriched sample: 79.4 / 66.9 / 76.9 / 63.7 %. It is lower because two thirds of the jets are W/Z, the hardest pair of classes.

**New controls** (not in P36). Both keep pT, $M_0$, $M_2$, $|\vec c|$ and every pair ΔR.
- **mirp**: $\Delta\phi \to -\Delta\phi$, a mirror image of the jet in φ about its axis.
  - It keeps the invariant mass exactly: median relative change 5e-7, max 1.5e-4 (particle masses).
  - It keeps every particle energy.
  - It keeps the total-momentum direction on the stored axis.
  - The φ mirror is a symmetry of the collision physics, so a network that uses only kinematics should be nearly unchanged.
- **flip**: $(\Delta\eta, \Delta\phi) \to (-\Delta\eta, -\Delta\phi)$, a 180° rotation about the axis. The invariant mass is exact for massless particles (median 3e-4).
- **On-manifold check**: the offset of the constituents' total-momentum direction (the E-scheme axis, i.e. the direction of $\sum_i \vec p_i$) from the stored axis, before and after each edit.

## Numbers
**1. On-manifold check.**
- In unedited jets the stored axis is the total-momentum direction: median offset 0.00026 rad over all jets and 0.0007 rad in tail5.
- The pT centroid sits $|\vec c|$ = 0.0062 (all) and 0.025 rad (tail5) from it.
- After recentring, the total-momentum direction is 0.026 rad (tail5 median; 90th percentile 0.049) from the stored axis.
- After mirp it stays at 0.0007 rad. After a matched dilation it is 0.0005.
- So the natural "offset" $\vec c$ is the difference between the pT-weighted centroid and the total-momentum direction. That difference exists in every jet, mostly because η is not a linear coordinate for $p_z$. Recentring creates a jet whose stored axis is no longer its momentum direction, a configuration that does not occur in the data.

**2. R on the W-vs-Z log-odds** (true W/Z jets; [95 % CI]; P36 value in brackets).

| model | set (W/Z jets) | κ | R(rec) | R(rs) | R(rec) − R(rs), p | H0 d(rec)=0: perm p (Holm, m=4) |
|---|---|---|---|---|---|---|
| ParT_full | tail5 (800) | −7.4 [−9.0, −6.0] | **1.05 [0.74, 1.40]** (1.33 [0.54, 2.25]) | 0.35 [−0.06, 0.76] (1.69) | 0.70, p 0.007 | 2e-4 (8e-4) |
| ParT_full | tail2 (320) | −7.5 | 1.01 [0.57, 1.52] | 0.06 | | 2e-4 |
| ParT_full | tail1 (160) | −8.0 | 1.11 [0.52, 1.77] | −0.12 | | 0.001 |
| ParT_full | all (16,000) | −9.9 | **0.94 [0.83, 1.06]** (1.08) | 0.82 [0.70, 0.95] | p 0.17 | 2e-4 |
| ParT_kin | tail5 | −6.5 | −0.08 [−0.74, 0.68] (0.39) | 0.17 | p 0.67 | 0.81 |
| ParT_kin | all | −7.2 | 0.75 [0.21, 1.38] (0.72) | −0.18 [−0.78, 0.38] | p 0.07 | 0.006 |
| PN_full | tail5 | −4.8 | 0.87 [0.12, 1.63] (1.01) | 1.03 | p 0.79 | 0.023 (0.045) |
| PN_full | all | −8.3 | 0.79 [0.65, 0.95] (0.92) | 0.84 | p 0.73 | 2e-4 |
| PN_kin | tail5 | −3.7 | 1.55 [0.84, 2.33] (0.48) | 0.27 [−0.40, 0.91] | 0.017 | 2e-4 (8e-4) |
| PN_kin | all | −6.8 | 0.67 [0.55, 0.81] (0.73) | 0.79 | p 0.23 | 2e-4 |

- ParT_full on tail5: the mean change of $\ell_W - \ell_Z$ under rec is +0.037 [0.025, 0.051], and the direct ratio d(rec)/d(dil_rec) is 1.08.
- Prediction from the P23 gains for ParT_full R(rec): 0.70. For ParT_kin the P23 prediction ($R = 2g_{odd} - g_{even}$ with $g_{odd} \approx 0$) is −0.46. The log-odds give +0.75 [0.21, 1.38] on all W/Z jets (P36: +0.72), so that prediction is not met on the primary read-out. The P36 report cites only the probe for ParT_kin.

**3. 10-class calls changed [%]**, class-balanced, tail5 (N = 1,200) [95 % CI].

| model | rec | rs | dil_rec | **mirp** | flip | rot | McNemar rec vs mirp (rec-only / mirp-only), p_Holm |
|---|---|---|---|---|---|---|---|
| ParT_full | 2.4 [1.3, 3.6] | 2.4 | 0.6 | **9.6 [7.5, 11.8]** | 11.9 | 10.6 | 21 / 141, 2e-22 |
| ParT_kin | 9.3 [7.0, 11.7] | 7.9 | 3.9 | **2.9 [1.7, 4.3]** | 20.5 | 17.4 | 95 / 21, 4e-12 |
| PN_full | 3.2 [2.1, 4.4] | 4.0 | 0.8 | **15.1 [12.5, 17.8]** | 17.1 | 17.7 | 33 / 178, 1e-24 |
| PN_kin | 4.7 [3.2, 6.4] | 7.1 | 1.2 | **2.9 [1.6, 4.3]** | 10.2 | 12.9 | 62 / 18, 8e-7 |

- All jets: mirp changes 7.6 % (ParT_full), 1.6 % (ParT_kin), 11.8 % (PN_full) and 1.4 % (PN_kin) of calls.
- mirp lowers the accuracy of the full-input models from 79.4 to 75.3 % (ParT_full) and from 76.9 to 70.6 % (PN_full). The kin models are unchanged (66.9 → 67.1 %, 63.7 → 63.7 %).
- mirp does not shift the mean W/Z log-odds of ParT_full: +0.03 [−0.04, +0.11], p 0.39. The per-jet changes are large, though: mean |d| is 0.59 under mirp against 0.12 under rec.

## Reading (scoped to these measurements)
- **Reproduced.**
  - ParT_full: R(rec) ≈ 1 on the off-centre tails (1.05 [0.74, 1.40] on 800 W/Z jets) and on all W/Z jets (0.94).
  - Recentring changes 2.4 % of ParT_full calls on tail5 (P36: 2.5 %) and 0.6 % overall.
  - H0 (no response) is excluded for ParT_full.
  - PN_full and PN_kin on all W/Z jets give 0.79 and 0.67 (P36: 0.92 and 0.73).
  - H_alt (on-manifold invariance, R(rec) much smaller than R(rs)) is not supported in any model. Where the two differ (ParT_full and PN_kin tail5), R(rec) is the *larger* one.
- **Not reproduced: "the same as a random shift" on the off-centre tail.** On ParT_full tail5, R(rs) is 0.35 [−0.06, 0.76] against R(rec) 1.05 (p 0.007). It holds on all W/Z jets (0.82 vs 0.94, p 0.17). Under the P23 gain model, R(rec) = 2g_odd − g_even and R(rs) ≈ g_even (the direction term averages out). So the tail values imply a larger direction-dependent (odd) gain than direction-independent (even) gain, which matches the P23 prediction of 0.70 in sign and order.
- **Recentred jets are off-manifold.**
  - The P36 conclusion says that "the axis dependence also acts on natural jets, not only on artificial shifts". That is not what was measured. The edit is an artificial rigid axis shift of about 0.025 rad: smaller than P1's 0.06 rad, and in the direction of the jet's own pT centroid.
  - In unedited jets the stored axis coincides with the total-momentum direction to 0.0007 rad. After recentring it does not.
  - What the test does distinguish: a read-out based on the moment about the pT centroid predicts R = 0. Since $M_0 M_{2,c} \approx m^2$ in the small-angle limit, that is a read-out of the invariant mass. A read-out based on the moment about the stored axis predicts R = 1. ParT_full gives about 1.
  - It follows, by local linearity and without a test on unedited jets, that ParT's mass sense on real jets includes the term $|\vec c|^2 M_0$. That raises the effective $\sqrt{M_2}$ by about 0.6 % on the 5 % most off-centre jets.
- **Rotation and mirror controls.**
  - For ParT_full and PN_full, even the φ mirror changes 10–15 % of tail5 calls and lowers accuracy by 4–6 points, although it keeps the mass, $M_2$, every energy and the momentum axis.
  - For ParT_kin and PN_kin it changes about 3 %.
  - So the full-input models depend on the jet's orientation in φ through one or more of their extra inputs: impact parameters d0, dz and their errors, charge, particle-ID flags. Which input was not identified.
  - This explains part of why rotation is not a null for the full models, in addition to the mass change under rotation that P36 identified.
  - For the kin models the mirror is a near-null. Recentring changes more calls than the mirror: ParT_kin 9.3 vs 2.9 % (p_Holm 4e-12), PN_kin 4.7 vs 2.9 % (p_Holm 8e-7).
- **ParT_kin.**
  - The W/Z log-odds response on tail5 is not measurable: R −0.08 [−0.74, 0.68], perm p 0.81.
  - Recentring still changes 9.3 % of its calls, against 3.9 % for the matched real mass change and 2.9 % for the mirror. Its calls therefore move through something other than the W-vs-Z mass read-out.
  - The P23 "no odd term" prediction, R(rec) < 0 < R(rs), is reversed on the log-odds in both P36 and here. The P36 reading cites only the probe, which shows the predicted sign.
- **Not checked here:** the L-GATr numbers (R 0.2–0.3, "about four times less"), the untrained-ParT probe comparison, and the observational pairs.

## Corrected reading
Recentring naturally off-centre W/Z jets, at fixed invariant mass, moves ParT_full's W/Z log-odds by the amount a real mass change of equal $\tfrac12\Delta\log M_2$ produces. R(rec) is 1.05 [0.74, 1.40] on 800 top-5 % W/Z jets and 0.94 [0.83, 1.06] on all W/Z jets, so ParT's mass read-out uses the moment about the stored axis rather than about the pT centroid. The edit is still an artificial axis shift, about 0.025 rad between the momentum direction and the stored axis, of natural size and direction; unedited jets were not tested. On the most off-centre 5 %, R(rec) exceeds the random-shift R (0.35, p 0.007); on all W/Z jets the two agree. ParticleNet gives 0.7–0.8 on all W/Z jets, and the tail5 values are wider. Recentring changes 2.4 % of ParT_full's calls on tail5.

For the kin-input models the φ mirror is a near-null control (about 3 % of calls), and recentring exceeds it. For the full-input models, the φ mirror, the 180° flip and the rotation all change 10–18 % of calls, so none of these is a null there. L-GATr not re-checked.

## Glossary additions
- **mirp (φ mirror)**: $\Delta\phi_i \to -\Delta\phi_i$ for every particle. It keeps pT, energies, pair distances, the invariant mass, $M_2$ and the momentum axis.
- **flip**: $(\Delta\eta_i, \Delta\phi_i) \to (-\Delta\eta_i, -\Delta\phi_i)$, a 180° rotation about the stored axis.
- **E-scheme axis offset**: the angle between the direction of the constituents' summed three-momentum and the stored jet axis. It is about 0 for unedited JetClass jets.
