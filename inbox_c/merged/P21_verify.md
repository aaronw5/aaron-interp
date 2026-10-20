# P21 verify (P21v): prong-observable edits on fresh jets, with dilation, hard-mass and rotation-location controls (verifier report, 2026-10-20)

Job `anrunw-ai-c-p21-verify3` (complete, 1,335 s, GPU). The smoke stage ran inside `anrunw-ai-c-p21-verify2`, which was deleted before its full stage to fix a reporting line. A failed job named `anrunw-ai-c-p21-verify` (115 min old, no log) already existed and was left untouched.
- Script: `interp/p21v_verify.py`. It imports the P21 edit search, rebuild, observables and head patching unchanged.
- Results: `results/part/p21_verify.json`, `results/part/p21_verify_summary.json`.
- Log: `logs/c-p21-verify3.log`.

## Conclusion
On new jets, the network responses reported in P21 come back within their uncertainties. One sentence of the P21 conclusion needs correcting. Moving the soft particles toward the jet axis makes ParT_full about 0.3 more W-like than the unedited jet, not 0.7. The 0.7 is measured against a rotation control that itself makes W, Z and top jets about 0.4 less W-like. At fixed jet mass this edit raises the mass carried by the harder particles (those with at least 2 % of the jet pT) by about 16 %. A classifier that is also given that hard-particle mass predicts a shift of the same size, so this response is not unexplained. Restoring attention blocks or heads gives the same profile, with the same leading heads, for the prong edits and for rotations that barely change τ21, τ32 or D2. That test therefore cannot show whether any particular observable is carried by these blocks.

## Code check (`interp/p21_prong_part.py`)
No error found that changes P21's numbers:
- Rows are drawn at random. Padded particles are masked in every edit, observable and centroid.
- η is rebuilt as jet_eta + sign(jet_eta)·Δη, the same in edits and rebuild.
- `HBLK = [4..7]` addresses blocks 5–8. The head decomposition reproduces the attention output to 3e-5.
- Patching uses the base and edited jets with the same pT ordering, so particle slots line up.
- The 10-class accuracies on the P1 rows reproduce: ParT_full 85.93 %, ParT_kin 74.47 %, PN_full 84.56 %, PN_kin 70.36 %, init 13.58 %.

Points that affect the reading:
1. **The mass restore changes the hard prongs too.** It is a dilation of every particle about the axis.
   - For E3 at u = 0.6 the median dilation factor is k = 1.17 (10–90 %: 1.05–1.48). The hard prongs are spread by 17 % while the soft ones move in.
   - E3 is therefore not only a soft-particle move. At fixed m it moves mass from soft wide-angle particles to the hard ones: m_hard(z ≥ 0.02) rises by a median 15.8 %, mean +9.9 GeV.
2. **E1 contains an outward soft move.** Holding D2 needs a median u = 1.32, which is the E3 direction at u > 1. P21 states this.
3. **Figure panel (a) shows edit − rotation only.** The rotation arm's own response is not visible, and the GBDT cross is the 4-variable reference, while the text quotes the (m, τ21) reference.
4. **Rotation arms are not mass-filtered.** |dm/m| reaches 3 % (log of c-p21-run2). This is minor.

## Procedure (rerun)
**Sample.**
- JetClass test_20M `WToQQ_105`, `ZToQQ_105`, `TTBar_105`, `ZJetsToNuNu_105`. These files are not used by P21.
- 2,500 random rows per class (`default_rng(2121)`). N = 9,997 jets with ≥ 4 particles.
- ParT_full accuracy on this 4-class sample (10-class argmax): 80.6 %.

**Edits and controls.** E1, E2, E3 and the rot and raw arms are as in P21, at doses 0.6 and 1.25.

**Added arms:**
- `nodil`: the edit without the mass-restoring dilation, so the mass changes.
- `dil`: the unedited jet dilated by the per-jet factor k that the E restore applied.
- Additivity residual: E − (nodil + dil), each relative to the base jet.

**Added observables:**
- m_hard(z): the invariant mass of particles with pT fraction ≥ z, for z = 0.02 and 0.05.
- ΔR12: the distance between the two k-means subjet axes.

**Added references.** HistGradientBoosting on (log m, log m_hard, τ21, τ32, log D2), 2-fold cross-fitted on base jets. Held-out W/Z-vs-QCD AUC:
- 0.908 (z = 0.02) and 0.906 (z = 0.05);
- against 0.903 for (m, τ21, τ32, D2) and 0.877 for (m, τ21).

**Location** (1,000 jets accepted for all three edits at 0.6):
- Block and head patching in blocks 5–8 for E1, E2, E3, scrF and the controls E1rot, E3rot and E3dil, on ParT_full, ParT_kin and init.
- NB = 2,000 bootstrap resamples, so a Holm correction over 64 heads can reach 0.05.
- 32-head restoration vectors compared by Pearson correlation.
- 1-d DAS (P18 method) on ParT_full and init for E1, E3, scrF, E1rot and E3rot. Difference-in-differences (b7 − emb), trained − init.

**Statistics.**
- 95 % bootstrap CI over jets.
- Sign-flip permutation p (5,000 permutations) for edit − rotation.
- Holm over the 24 primary cells (4 networks × 3 edits × 2 doses).

**Hypotheses.**
- H0 (P21): edit response = rotation response.
- Competing explanations for E3:
  - (i) D2 itself.
  - (ii) The hard-particle mass at fixed m, a groomed-mass-like quantity.
  - (iii) A response specific to the rotation control.
- For location:
  - (L1) The patching profile is specific to prong edits.
  - (L0) It is generic to small moves of particles inside subjets.

## Numbers
**Behaviour.** Edit − rotation, mean Δlogit [95 % CI], fresh sample (P21 value in brackets):

| cell | ParT_full | ParT_kin | PN_full | PN_kin |
|---|---|---|---|---|
| E1 0.6, d_WQ | −0.112 [−0.154, −0.070] (−0.144) | −0.113 [−0.150, −0.076] (−0.110) | +0.012 [−0.027, 0.048], p 0.52 (+0.006) | −0.006 [−0.039, 0.025], p 0.74 (−0.014) |
| E1 1.25, d_WQ | +0.069 [0.036, 0.099] (+0.047) | +0.093 | +0.120 | +0.077 |
| E2 0.6, d_TQ | −0.271 [−0.310, −0.231] (−0.246) | +0.153 [0.121, 0.184] (+0.170) | +0.077 (+0.114) | +0.364 (+0.366) |
| E2 1.25, d_TQ | −0.330 [−0.357, −0.302] (−0.334) | −0.330 | −0.217 | −0.384 |
| E3 0.6, d_WQ | +0.728 [0.695, 0.760] (+0.729) | +0.713 | +0.693 | +0.676 |
| E3 1.25, d_WQ | −0.223 [−0.245, −0.202] (−0.215) | −0.206 | −0.137 | −0.209 |

- Holm p = 0.005 in 22 of 24 primary cells.
- PN E1 at 0.6 is not excluded (Holm 1.0), as in P21.

**E3 at u = 0.6 against the unedited jet** (not against rotation):

| model | edit vs base | rotation vs base |
|---|---|---|
| ParT_full | +0.310 [0.279, 0.340] | −0.417 [−0.445, −0.390] |
| ParT_kin | +0.355 [0.330, 0.381] | −0.358 |
| PN_full | +0.448 [0.420, 0.477] | −0.244 |
| PN_kin | +0.432 [0.411, 0.454] | −0.244 |

- ParT_full rotation response by class: W −0.62, Z −0.56, top −0.52, QCD +0.03.
- The rotation leaves τ21, τ32 and D2 within 3.7 % on average and m_hard unchanged (median +0.2 %).

**References for E3 at 0.6, edit vs base** (W/Z vs QCD):

| reference | predicted | ParT_full by class (W / Z / top / QCD) |
|---|---|---|
| (m, τ21) | +0.006 | |
| (m, τ21, τ32, D2) | +0.089 | |
| (m, m_hard02, τ21, τ32, D2) | +0.289 | +0.46 / +0.40 / +0.02 / +0.23 |
| (m, m_hard05, …) | +0.154 | |

- ParT_full by class: W +0.135, Z +0.572, top +0.096, QCD +0.424.
- The mean matches the m_hard02 reference. The per-class pattern does not: the reference moves W most, ParT moves Z and QCD most.

**Decomposition of E3 at 0.6, ParT_full d_WQ, vs base:**
- nodil (soft in, m −15 %, −16.5 GeV): +0.160.
- dil (base dilated by k, m +21 %): −1.045.
- E: +0.310.
- Additivity residual +1.20, so the two pieces do not add. A dilation that changes m is not a usable proxy for the dilation inside E.
- At 1.25: nodil −0.856, dil −0.077, E −0.413.

**E1 at 0.6 by class, edit vs base, ParT_full** (reference m_all in brackets):
- W −0.747 (+0.315); Z −0.278 (+0.31); top −0.465 (−0.119); QCD +0.260 (+0.271).
- QCD jets move as the reference predicts in sign and size. The opposite sign of the pooled response comes from W, Z and top jets.
- The same holds in ParT_kin (QCD +0.278; W −0.522), PN_full (QCD +0.307; W −0.558) and PN_kin (QCD +0.315; W −0.342).
- References for E1 0.6 edit − rotation:
  - (m, τ21) +0.276; m_all +0.199; m_hard02 +0.152.
  - All have the opposite sign to ParT_full's −0.112.

**Head and block patching, ParT_full.** Block restoration b5 / b6 / b7 / b8 [CI ≈ ±0.02]:

| arm | b5 / b6 / b7 / b8 | mean squared logit change |
|---|---|---|
| E1 | 0.314 / 0.336 / 0.271 / 0.210 | 19.6 |
| E2 | 0.325 / 0.351 / 0.284 / 0.217 | 11.9 |
| E3 | 0.306 / 0.329 / 0.288 / 0.218 | 14.5 |
| E3rot (soft-particle rotation) | 0.315 / 0.332 / 0.307 / 0.226 | 7.6 |
| E1rot (2-subjet rotation) | 0.278 / 0.313 / 0.275 / 0.262 | 2.3 |
| scrF | 0.132 / 0.153 / 0.179 / 0.219 | 69.3 |
| E3dil | 0.147 / 0.143 / 0.119 / 0.115 | 21.2 |

- Leading single heads:
  - E1: b6h5 0.082, b7h5 0.078, b5h5 0.069.
  - E3: b7h5 0.097, b5h5 0.091, b6h5 0.089.
  - E3rot: b7h5 0.095, b6h5 0.080, b5h5 0.076.
- Correlation of the 32-head restoration vectors: E1–E3rot 0.91, E2–E3rot 0.88, E3–E3rot 0.87, E1–E3 0.94, E1–scrF 0.52.
- In the untrained net, whose logit response is about 0.03 in mean squared change, the E1–E2–E3 head vectors also correlate at 0.94–0.98.
- With NB = 2,000, 29–32 of 32 heads per arm pass Holm 0.05 over 64 heads, E3rot included (32 / 32). Each head carries a small, nonzero share for every arm.
- ParT_kin: the E1, E2 and E3 blocks are within 0.03 of each other. E3rot is 0.20 / 0.26 / 0.24 / 0.23 against E3's 0.21 / 0.23 / 0.26 / 0.20.

**DAS 1-d IIA, ParT_full** (emb / b5 / b7 / c1):
- E1: 0.01 / 0.17 / 0.30 / 0.38.
- E3: 0.04 / 0.17 / 0.28 / 0.36.
- scrF: 0.02 / 0.67 / 0.70 / 0.58.
- E1rot: 0.03 / 0.10 / 0.14 / 0.27.
- E3rot: 0.02 / 0.13 / 0.22 / 0.34.

**Difference-in-differences, trained − init (b7 − emb):**
- E1 +0.160 [0.122, 0.195]; E3 +0.092 [0.058, 0.128]; scrF +0.503 [0.477, 0.528]. P21: +0.195, +0.083, +0.509.
- E1rot −0.024 [−0.061, 0.014], p 0.23; E3rot +0.036 [0.005, 0.067], p 0.025.

## Reading (scoped to these measurements)
**Reproduced on fresh jets:**
- The sign and size of every primary behaviour cell.
- The opposite sign of E1 against all three τ21-containing references.
- The non-monotone τ32 response of ParT_full.
- The null for ParticleNet E1.
- The DAS depth profiles and the difference-in-differences.
- The block and head patching pattern, with the same leading heads.

**Corrections:**
1. **The E3 size.**
   - The "+0.7 more W-like in all four networks" is edit − rotation. Against the unedited jet, E3 at 0.6 is +0.31 to +0.45.
   - The rotation control itself contributes −0.24 to −0.42, concentrated in W, Z and top jets. That makes the rotation an active manipulation for these networks, not a neutral baseline.
   - P21's "about 13 times what D2 predicts" compares edit − rotation with a reference that sees no change under rotation.
2. **What E3 changes.**
   - Because the mass restore dilates the hard prongs, E3 raises the hard-particle mass by a median 16 % at fixed m.
   - A reference that includes m_hard(z ≥ 0.02) predicts +0.29 against ParT_full's +0.31 for the mean, so explanation (ii) is not excluded. The class pattern differs from that reference, so it is not confirmed either.
   - Explanation (i), D2 alone, predicts +0.09.
3. **E1.**
   - The opposite sign holds for W, Z and top jets. QCD jets follow the references: +0.26 against +0.27.
   - E1 includes an outward soft move of about 32 % to hold D2, and E3 at u = 1.25 alone moves ParT_full by −0.41. The E1 sign may therefore come in part from the D2 compensation.
   - P21's raw arm (no compensation, d_WQ −0.21 against base, reference +0.96) argues against that being the whole effect. That arm also lowers D2 by 18 %.
4. **Location.**
   - Block and head patching gives the same profile and the same leading heads (head 5 of blocks 5–7) for the soft-particle rotation E3rot as for the prong edits (head-vector correlation 0.87–0.91).
   - In the untrained net the three edits also share one head pattern (0.94–0.98).
   - L0 (generic to small within-subjet moves) is therefore not excluded. The P21 statement "the three edits are carried by the same blocks and heads, so the late stage is not tied to any one observable" is not supported by patching, which does not separate prong edits from observable-neutral moves.
   - DAS does separate them. The training-specific rise is +0.16 (E1) and +0.09 (E3), against −0.02 and +0.04 for their rotations and +0.50 for the full scramble.
   - So the single-observable edits use a small part of the late-block direction that the full scramble uses.

**Not run / open:**
- An edit that holds m_hard (or a soft-drop mass) as well as m, τ21, τ32 and D2. This is the direct test of explanation (ii).
- `nodil` at dose 1.25 gave NaN logits for some jets in ParT (E1, E2), and those cells are missing.
- k-means subjets were kept, as in P21.

## Corrected reading (for the ledger)
At fixed m, τ32 and D2, lowering τ21 by about 11 % moves ParT_full's W-vs-QCD logit by −0.11 [−0.15, −0.07] relative to a matched within-subjet rotation (P21: −0.14). Gradient-boosted references on (m, τ21), (m, τ21, τ32, D2) or (m, m_hard, τ21, τ32, D2) predict +0.15 to +0.28. QCD jets follow the references (+0.26 vs +0.27) and W/Z/top jets move the other way (W −0.75). The radial soft-particle edit (D2 −4 %) moves ParT_full by +0.31 [0.28, 0.34] against the unedited jet, and by +0.73 against its rotation control, which itself moves −0.42. Through the mass restore, this edit raises the hard-particle mass by about 16 % at fixed m, and a reference including that mass predicts +0.29. Head and block patching in blocks 5–8 gives the same profile and leading heads for the prong edits and for observable-neutral rotations. DAS shows a training-specific late rise for E1 and E3 (+0.16, +0.09), above their rotations (−0.02, +0.04) and below the full scramble (+0.50).

## Explainer entry (addition to P21's)
- **Question.** Do P21's single-variable prong edits reproduce on new jets, and what else do they change?
- **Background.** Restoring the jet mass after an edit multiplies every angle by a factor $k$. Since $m^2 \approx \sum_{i<j} p_{T,i}p_{T,j}\,\Delta R_{ij}^2$ for a narrow jet, pulling soft particles in ($u<1$) lowers $m$, so $k>1$ also spreads the hard prongs. The hard-particle mass is $m_{\rm hard}(z_0)=\sqrt{\big(\sum_{z_i\ge z_0}p^\mu_i\big)^2}$ (four-vectors) with $z_i=p_{T,i}/\sum_j p_{T,j}$.
- **Method.**
  - Fresh rows (files `_105`, seed 2121, N = 9,997) and the same edits.
  - Added arms: the edit without mass restore, and the dilation alone.
  - Added references: gradient-boosted trees with $m_{\rm hard}$.
  - Patching of the rotation controls in blocks 5–8; correlation of the 32-head restoration vectors.
- **Result.**
  - $\Delta(\ell_W-\ell_{QCD})$ for E3: $+0.31\,[0.28,0.34]$ vs the unedited jet and $+0.73$ vs rotation. $m_{\rm hard}(0.02)$ rises by 16 %, and the reference with $m_{\rm hard}$ predicts $+0.29$.
  - The block restoration for the soft-particle rotation is 0.32 / 0.33 / 0.31 / 0.23 against 0.31 / 0.33 / 0.29 / 0.22 for E3 (blocks 5–8).
- **Checks.**
  - Unedited accuracy 85.93 % / 74.47 %.
  - Every primary behaviour cell reproduces within its CI or within 0.04.
  - Head decomposition error 3e-5.
- **Why it matters.**
  - The large soft-particle response has a candidate physical reading: at fixed jet mass, mass carried by hard prongs counts as more W-like than mass carried by wide-angle soft radiation.
  - Head patching does not distinguish prong edits from rotations, so it cannot locate an observable-specific stage.
