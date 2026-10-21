# H68 verify: particle-type channel of the full-input JetClass taggers (loop D)

Verifier script `interp/h68_verify.py` (imports the edit functions of `interp/h68_particle_type_channel.py` unchanged).
Job `anrunw-ai-d-h68-verify` (GPU, GTX 1080 Ti, 882 s; smoke `anrunw-ai-d-h68-verify-smoke`, 30 jets per class).
Results `results/physics/h68_verify.json`; log `logs/d-h68-verify.log`.

## Conclusion

The verifier repeated H68's relabel edits on a new JetClass file and got the same numbers within about 0.1 log-odds. It then
added a control H68 lacked: some charged hadrons were relabelled as neutral, and an equal amount of energy in real neutral
particles was relabelled as charged, so the jet's charged-energy fraction (f_ch, the share of jet energy carried by charged
particles) stayed nearly fixed. This swap lowered the signal scores about twice as much as H68's one-way relabel (ParT,
H→cc: −1.26 log-odds against −0.80). So the drop does not follow the charged-energy fraction. Any rearrangement of which
particles carry which label lowers the scores, in both directions and also among charged hadrons and photons alone. Whether
this comes from type–kinematics correlations the taggers learned or from label patterns that never occur in real jets is
still open.

## What was checked

**Code read.** `relabel`, `frac_set`, `csame`, `ckin`, `with_ghosts`, `scores`, the bootstrap and the Holm calls.
- No axis or coordinate error. Relabels touch only the type, charge and track fields, and leave px, py, pz, E unchanged. ParT_kin and PN_kin respond exactly 0 again.
- Padded slots are excluded from every sum (`mask`).
- Rows are random per class (`default_rng`), not first-k.
- Observables and scores are computed on the same jets.
- Score s_c = logit_c − logit_QCD on true class-c jets. For QCD jets, s = logit_QCD − logsumexp(the other logits).

**Design weaknesses found** (none is a code bug):
1. **c_kin.** The jitter ε ~ N(0, 0.005) has zero mean, so the mean Δs vanishes at first order whatever the network's kinematic sensitivity. The per-jet RMS of Δs under c_kin is ≤ 0.01 (ParT_full), against 0.5–1.6 under fr10. So this control cannot bound anything.
2. **Permuted-model null.** The comparison uses raw |Δs|. The permuted models' unedited scores spread by s.d. 0.29–0.96 log-odds, against 2.89 for ParT_full, so raw magnitudes are not comparable (see below).
3. **The H1 test has a gap.** Every H68 edit that moves f_ch also moves labels in one direction (charged → neutral). No edit moved labels while keeping f_ch fixed, so "scales with Δf_ch" could not be separated from "scales with the number or energy of relabelled particles".
4. **Job note.** k8s-chase-ci-01 is already excluded in the committed `part_job.sh` (commit ebb92ea, loop C). No change is needed.

**Figure** (`h68_particle_type.png`, read). The bars and intervals match `h68_jc.json`. The left panel shows the fr10 set on a real H→bb jet. The grey bars are labelled "control" but are as large as the edits, which the report states.

## Procedure

- **Sample:** JetClass `test_20M/<sample>_101.root`. H68 used `_100`, so these are fresh jets. 1,000 random rows per class (`default_rng(6811)`), N = 10,000.
- **References:** trained on `train_100M/<sample>_001.root` (H68 used `_000`), 2,000 random rows per class (`default_rng(6812)`).
- **Models:** ParT_full, PN_full, ParT_kin, PN_kin, and three weight-permuted ParT_full copies (torch seeds 7, 8, 9; H68 used 1–6).
- **Repeated edits** (H68 definitions): fr05/fr10/fr20, frg10, ph2nh, nh2ph, ch2nh2, ch2g2, csame_fr10, ckin_fr10, and the ρ = 10 ghost conditions.

**New controls**
- **swap10** (label swap at fixed composition):
  - the fr10 charged-hadron set (4.45 particles per jet, 10 % of jet energy) is relabelled as neutral hadrons;
  - in the same jet, real neutral hadrons are taken in random order until their summed energy is closest to the moved set's energy (2.68 per jet), and they are relabelled as charged hadrons;
  - the new charged hadrons get charge ±1 at random, and d0, dz and their errors copied from a real charged hadron of the sample in the same pT decile.
  - Four-momenta are unchanged. Δf_ch per class is −0.013 to −0.039, against −0.088 to −0.099 for fr10. The shortfall is in jets with less neutral-hadron energy than the moved set; 9 % of jets have no neutral hadron.
- **swapg10:** the same with photons. The fr10 set becomes photons, and an energy-matched set of real photons (7.2 per jet) becomes charged hadrons. Δf_ch is −0.003 to −0.016, and f_nh is unchanged.
- **fr10m:** fr10 with the relabelled particles made massless (E set to |p|, as for real Delphes neutral hadrons). This tests whether the leftover pion mass of a relabelled track matters.
- **Logit-scale normalisation:** Δs divided by the s.d. over jets of that model's unedited score, per class.
- **Stacked ghosts with a single label:** the ρ = 10 composition ghosts placed on the axis (0.002 rad) with all-neutral-hadron (gstacknh10) or all-photon (gstackph10) labels. Stacked share = Δs(on-axis) / Δs(uniform in the R = 0.8 disc), for each label.

**Statistics**
- Δs = mean over a class's 1,000 jets of (edited − unedited score), with a 95 % bootstrap interval (B = 1,000).
- p from the bootstrap s.e. Holm over ParT_full and PN_full × 21 conditions × 10 classes (420 tests).
- McNemar counts on 10-class calls.
- H1 (composition reader): Δs(swap) = slope × Δf_ch, using the per-class slope dΔs/dΔf_ch fitted on fr05/10/20 in this run.
- H0 for the swap: Δs(swap) − slope·Δf_ch = 0.

## Numbers

**Unedited 10-class accuracy (this sample).**
- ParT_full 0.8564 (H68 0.8606), PN_full 0.8382 (0.8415), ParT_kin 0.7463 (0.7395), PN_kin 0.6994 (0.6976).
- Permuted models 0.100–0.106.

**Reproduction.** ParT_full Δs, classes Hbb / Hcc / Hgg / H4q / Hqql / Zqq / Wqq / Tbqq / Tbl, with H68 values in brackets:

| edit | verify | H68 |
|---|---|---|
| fr10 | −0.44 [−0.49, −0.39], −0.80, −0.19, −0.64, −1.04, −0.53, −0.63, −0.89, −0.56 | (−0.46, −0.70, −0.16, −0.67, −1.00, −0.50, −0.61, −0.78, −0.55) |
| fr20 | −0.91, −1.39, −0.26, −1.15, −1.71, −0.88, −1.07, −1.56, −1.22 | (Hbb −0.99) |
| ph2nh | −1.92 to −2.59, Tbl −1.01, QCD +0.22 | (−2.02 to −2.55, Tbl −0.85, QCD +0.23) |
| frg10 | −0.11 to −0.71 | (−0.10 to −0.67) |
| csame_fr10 | −0.06 to −0.65; Holm passes 9/10 (ParT_full), 8/10 (PN_full) | (−0.05 to −0.61) |
| ch2g2 | \|Δs\| ≤ 0.05 | |
| ch2nh2 | −0.06 to −0.21 | |

- Slope dΔs/dΔf_ch (log-odds per unit f_ch): ParT_full 4.44 (Hbb) to 9.0 (Hqql), Hgg 1.39, QCD 0.62. H68: 4.7–9.3, Hgg 1.4.
- Sign agreement with GBDT_frac (the fraction-aware boosted classifier) under fr10: 4/9 for ParT_full and for PN_full (H68: 5/9). With LR(f_ch), the likelihood ratio of the charged-energy fraction alone: 9/9.
- Kinematics-only models: 0.00 on every relabel. Under ckin, per-jet RMS ≤ 0.04.

**New control: label swap at nearly fixed composition** (ParT_full Δs, Hbb … Tbl):

| edit | Δs | prediction of H1 (slope · Δf_ch) | observed − predicted | QCD jets |
|---|---|---|---|---|
| swap10 | −0.93 [−1.02, −0.84], −1.26, −0.45, −0.85, −1.59, −0.96, −1.03, −1.40, −0.94 | −0.02 to −0.31 | −0.43 to −1.29, all p < 1e−25 | −0.51 |
| swapg10 | −0.95, −1.03, −0.27, −0.93, −1.65, −0.84, −0.83, −1.32, −1.33 | −0.005 to −0.14 | −0.27 to −1.51 | −0.28 |

- PN_full gives the same pattern: swap10 −0.25 to −0.94, swapg10 −0.16 to −1.14.
- Holm: swap10 and swapg10 pass in 10/10 classes for both models.
- Calls (ParT_full, right→wrong / wrong→right over 10,000 jets): swap10 839 / 237 and swapg10 761 / 219, against fr10 392 / 163.
- GBDT_frac under swap10: −0.29 to +0.07 (Tbl −0.19, Hqql −0.29, the others within ±0.08).
- The swaps lower signal scores about 1.5–2× more than fr10, while moving f_ch by a quarter (swap10) or a tenth (swapg10) as much. swapg10 involves no neutral-hadron label at all, only charged ↔ photon.

**Leftover pion mass (fr10m − fr10).** ParT_full −0.013 to +0.034; PN_full ≤ 0.001 in magnitude. The massless version gives the same response.

**Permuted null, normalised by logit scale.** Mean over 10 classes:

| edit | measure | ParT_full | PN_full | perm7 / perm8 / perm9 |
|---|---|---|---|---|
| fr10 | raw \|Δs\| | 0.57 | 0.37 | 0.03 / 0.07 / 0.22 |
| fr10 | \|Δs\| / s.d.(s) | 0.19 | 0.13 | 0.09 / 0.09 / 0.23 |
| ph2nh | \|Δs\| / s.d.(s) | 0.63 | 0.51 | 0.39 / 0.19 / 0.33 |
| swap10 | \|Δs\| / s.d.(s) | 0.34 | 0.23 | 0.05 / 0.05 / 0.10 |
| csame_fr10 | \|Δs\| / s.d.(s) | 0.10 | 0.04 | 0.03 / 0.03 / 0.02 |

s.d.(s) is the spread over jets of the unedited score: 2.89 for ParT_full and 0.29 / 0.70 / 0.96 for the permuted copies. After this scaling, the fr10 response of ParT_full is 0.8–2.1× the permuted one, not 5×. For swap10, ph2nh and csame the ratio stays 1.6–5×.

**Ghost arm, stacked share by label** (ParT_full; Wqq with 95 % CI):

| label | Wqq | per class |
|---|---|---|
| measured composition | 0.41 [0.39, 0.43] | |
| all neutral hadron | 1.10 [1.07, 1.13] | 0.2–1.6 |
| all photon | 0.19 [0.18, 0.21] | |

- PN_full Wqq: 0.39 / 1.09 / 0.32.
- ParT_kin: the same with every label (0.20 Wqq). PN_kin stack −10 to −22 and clusters −0.3 to −2.3, as in H68.
- This tests H68's claim that H65v's share above 1 comes from the neutral-hadron label, with the same ghosts and only the label changed. The claim holds (H68 had inferred it with a different ghost spectrum).

## Reading

- **Reproduced.** All of H68's relabel numbers, the linear scaling over fr05/10/20, the exact zero of the kinematics-only models, the c_same response, the sign agreement (4–5/9 with GBDT_frac, 9/9 with LR(f_ch)) and the ghost-arm label modulation repeat on a fresh file.
- **Interpretation not supported.** H68 links the drop to the charged-energy fraction moved: "grows with the share of jet energy moved from charged to neutral hadrons, about 0.5 log-odds per 10 %". At nearly fixed f_ch, the label swap gives a larger drop, 3–20× the slope × Δf_ch prediction (p < 1e−25 on every class). So the fr-series scaling measures how many and how energetic the relabelled particles are, not a response to f_ch.
  - Both directions of relabelling lower signal scores: charged → neutral, neutral → charged, photon → neutral hadron (ph2nh) and neutral hadron → photon (nh2ph).
  - A charged ↔ photon swap without any neutral-hadron label also lowers them.
  - So the data do not support "the neutral-hadron label is evidence against a heavy-particle decay" (H68 report Conclusion).
- **"About 5× the permuted null"** holds for raw log-odds. After scaling by each model's score spread, it is 0.8–2.1× for fr10.
- **fr10m** excludes the leftover pion mass of relabelled tracks as the cause.

H1 (the network reads the composition fractions) predicts swap Δs ≈ 0; it is excluded. A rearrangement or out-of-distribution reading (any change to which particles carry tracks or which neutral label they carry lowers signal confidence) predicts a drop in both directions; it is consistent with the data. A third reading, in which the networks use real type–position or type–energy correlations, also predicts drops under every swap. This test does not separate it from an out-of-distribution response. A C2ST (a classifier two-sample test: a classifier trained to tell edited jets from real jets) or a GBDT with per-particle type × kinematics features would be needed.

## Corrected reading

With four-momenta fixed, ParT_full and PN_full lower the own-class-vs-QCD score of all nine signal classes under every particle-type relabel tested: charged → neutral hadron, charged → photon, photon ↔ neutral hadron, and swaps that keep the charged-energy fraction nearly fixed (Δf_ch −0.003 to −0.039). The swaps give 1.5–2× the drop of the one-way 10 % relabel. The response therefore tracks the amount of label rearrangement, not f_ch. It is linear in the relabelled energy along the fr series and 0 for the kinematics-only models. In raw log-odds it is about 5× the weight-permuted null, but only 0.8–2.1× after scaling by each model's score spread (fr10). A charge-sign swap at fixed type also moves the scores. On stacked ghosts, the share above 1 comes from the neutral-hadron label (Wqq 1.10 with all-neutral-hadron ghosts, against 0.41 with the measured composition).

## Corrected Conclusion

Changing particle-type labels while keeping every particle's momentum fixed lowers the signal scores of the type-aware JetClass taggers, and the verifier reproduced this on a new file. The drop does not follow the jet's charged-energy fraction. Swapping charged and neutral labels between particles of equal total energy, which leaves that fraction almost unchanged, lowers ParT's scores about twice as much as relabelling 10 % of the energy in one direction. So the taggers respond to which particles carry which label, in any direction, rather than to the jet's overall composition. Whether this reflects learned correlations between particle type and position or energy, or a reaction to label patterns that never occur in real jets, is not settled.
