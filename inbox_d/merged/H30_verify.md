# H30 verification: Lund-plane response map of quark–gluon taggers

Scripts: `interp/h30_verify.py` (GPU job `d-h30-verify2`) and `interp/h30_verify_auc.py` (GPU job `d-h30-verify-auc`).
- The job name `d-h30-verify` was already taken by an earlier attempt of this verification. That attempt failed with "CUDA error: invalid device ordinal" on a GTX 1080 Ti node before it produced output. It was left untouched, and this run used `d-h30-verify2`.

Results: `results/physics/h30_verify.json`, `h30_verify_checks.json` and `h30_verify_auc.json`. The per-jet files `h30_verify_logits_{py,hw}.npz` are on the volume only.

## Conclusion
H30's maps reproduce on a fresh sample. However, the main edit does two things at once: it moves the particle to a new angle, and it takes up to 50 GeV from the hardest particle and gives it to the moved particle. The rotation control does not separate the two.

Two added controls do separate them:
- **pT-only:** the same pT transfer, with the particle left at its own angle.
- **Angle-only:** the angular move with the particle's pT kept.

The result for Pythia-trained ParT on Pythia jets:
- **Small angle, kt 1.3 GeV.** The rise toward small angle comes from the pT transfer: the pT-only edit gives +0.24 on the angular slope. Moving the particle at its new pT has the opposite slope (−0.14).
- **Low kt.** The flat response is the sum of two opposite parts.
- **kt 7.5 GeV.** The move toward "quark" is entirely the wide-angle placement of a 25–40 GeV particle.

So the network map is not a single function of where the emission sits in the Lund plane. Still open: whether the generator difference is concentrated at kt above 1 GeV. In absolute size it is, but relative to the size of the response it is not, on Herwig jets.

## Code review of `interp/h30_qg_lund_map.py`
Checks on the fresh sample (`h30_verify_checks.json`):
- The hardest particle is at index 0 in every jet (Pythia rows from `data_top.sample`, pT-sorted; Herwig rows sorted explicitly).
- The main edit conserves ΣpT to within 2.3e-13 GeV and N exactly.
- Edits are applied to the unpadded lists. `from_list(Lpad=180)` pads afterwards, and the observables are computed on the same edited lists the models see.
- The axis is the E-scheme axis of the edited jet, iterated.
- Rows are random per class (`rng.choice`), not first-k.
- The working point is the logit sign (ℓ = 0) and AUC, as stated.

No bug that changes the numbers was found. Design points:
1. **The rotation control c1 does not control the second particle the main edit changes.** The main edit sets the hardest particle's pT to pT₁ − (k_t/ΔR − pT_i). For example, ΔR 0.026, k_t 1.29 GeV gives pT_i = 49.6 GeV, which takes about 49 GeV from the leading particle. c1 keeps every pT fixed. "main − c1" therefore mixes the angular placement with a change of the leading-particle fraction. That change alone shifts pT^D by −0.044 and girth by +0.011 in that cell.
2. **The selected source particle is soft, but the destination is not** (z up to 0.10). H30 already notes this (issue 1).
3. **The 3-vector sum is not conserved.** Only ΣpT and N are fixed: the jet direction and mass move. The axis is recomputed, so the features stay consistent.
4. **BH counts carry little information at this N.** The permuted-weight null reaches BH q < 0.05 in 22–27 of 27 cells with |effect| ≤ 0.03. Effect sizes, not cell counts, carry the result.
5. **Figure** (`h30_lund_map.png`, read):
   - The ΔR axis is inverted (0.3 on the left).
   - Skipped cells are white, while the title says grey.
   - The title says "soft particle (0.5–2 %)", which describes the source, not the destination.
   - The numbers in the panels match `h30_run.json`.

## Procedure
**Sample.** QG test jets, 1000 q + 1000 g per generator, random rows, seed 3002 (`h30_qg_lund_map.SEED` set to 3002, which also re-seeds the selection and control streams):
- Pythia: `data_qg` test split. 30 jets overlap with the H30 sample.
- Herwig: files 18–19. 39 jets overlap with the H30 sample.

The selected particle is found in 1998 (Pythia) and 1994 (Herwig) jets. The grid and cell rules are those of H30; 27 cells pass for each generator.

**Conditions per cell (r, k).**
- **m:** H30 main edit.
- **c:** H30 c1 (rotation at ΔR0).
- **p (pT-only, new):** pT_i → k_t/ΔR_r with the same transfer from the hardest particle. The particle is re-placed at its own ΔR0 along ψ0 about the edited axis.
- **x (angle-only, new):** pT_i kept, no transfer, particle placed at ΔR_r along ψ0.

Decomposition: (m − base) = (p − base) + (m − p). The term m − p is the angular move at the new pT.

**Models.** part_kin, pnet_kin (Pythia-trained); hw_part_kin, hw_pnet_kin (Herwig-trained); efn_t, pfn_t (H5 torch); part_kin_perm (null).

**Statistics.**
- Per term and cell: mean over jets with a 95 % bootstrap CI (1000 resamples over jets), sign-flip p (2000), BH over the 27 cells.
- k_t-row slope at k_t 1.29 GeV: cell ΔR 0.026 minus cell ΔR 0.312, per jet, on the jets valid in both cells (n = 1088 Pythia, 1120 Herwig). Bootstrap CI and sign-flip p.
- Generator difference: |Hw-trained − Py-trained| map per cell. Absolute mean and median relative value (|diff| / mean |map|) for k_t < 1 GeV vs > 1 GeV. The same quantities were recomputed from H30's `h30_run.json`.

## Checks: unedited models (AUC / acc)
| model | Pythia jets (seed 3002) | Herwig jets (seed 3002) |
|---|---|---|
| ParT_kin (Py) | 0.878 / 0.805 | 0.818 / 0.728 |
| PN_kin (Py) | 0.882 / 0.809 | 0.819 / 0.724 |
| ParT_kin (Hw) | 0.858 / 0.774 | 0.841 / 0.762 |
| PN_kin (Hw) | 0.854 / 0.763 | 0.840 / 0.761 |
| EFN (H5) / PFN (H5) | 0.843 / 0.864 | 0.797 / 0.811 |
| ParT_kin permuted | 0.527 | 0.533 |

The Pythia AUCs are 0.02–0.03 below H30's. `d-h30-verify-auc` measured ParT_kin over ten 1000 + 1000 Pythia samples (seeds 3001–3010):
- Range 0.878–0.908, mean 0.898. Seed 3002 is the lowest.
- Seed 3001 with 2000 + 2000 jets: 0.904, which reproduces H30.

The lower value is sample-to-sample spread, not a code difference.

## Numbers (Pythia jets unless stated; Δℓ = gluon log-odds change, positive = toward gluon)
**Reproduction of H30 (main − c1).**

| cell | Py ParT_kin, H30 | Py ParT_kin, fresh sample |
|---|---|---|
| ΔR 0.026, k_t 1.29 GeV | +0.285 | +0.296 [0.264, 0.328] |
| ΔR 0.312, k_t 1.29 GeV | +0.158 | +0.167 |
| ΔR 0.189, k_t 7.47 GeV | −0.276 | −0.277 [−0.324, −0.231] |
| ΔR 0.312, k_t 7.47 GeV | −0.120 | −0.118 [−0.149, −0.087] |

The Herwig-trained ParT values also reproduce: +0.391, −0.489 and −0.389 against H30's +0.373, −0.469 and −0.366.

**Decomposition, Py ParT_kin.** Columns: main − c1 | pT-only (p − base) | angle at new pT (m − p) | angle-only at old pT (x − base).

| cell | main − c1 | pT-only | angle at new pT | angle-only |
|---|---|---|---|---|
| ΔR 0.026, k_t 0.40 | +0.033 | +0.070 [0.061, 0.082] | −0.016 | −0.030 |
| ΔR 0.312, k_t 0.40 | +0.057 | −0.054 [−0.061, −0.048] | +0.149 [0.141, 0.157] | +0.213 |
| ΔR 0.026, k_t 1.29 | +0.296 | +0.226 [0.182, 0.273] | +0.098 [0.058, 0.138] | −0.035 |
| ΔR 0.312, k_t 1.29 | +0.167 | −0.011 | +0.224 [0.208, 0.240] | +0.213 |
| ΔR 0.189, k_t 7.47 | −0.277 | +0.146 [0.114, 0.178] | −0.398 [−0.435, −0.363] (q −0.26, g −0.59) | +0.177 |
| ΔR 0.312, k_t 7.47 | −0.118 | +0.097 [0.080, 0.114] | −0.168 [−0.198, −0.142] (q +0.08, g −0.43) | +0.215 |

The ΔR slope at k_t 1.29 GeV (ΔR 0.026 minus ΔR 0.312) was tested with H0: slope = 0 for each term.

| scorer | main − c1 | pT-only | angle at new pT |
|---|---|---|---|
| Py ParT_kin, Pythia jets | +0.123 [0.079, 0.164], p 5e-4 | +0.239 [0.194, 0.288], p 5e-4 | −0.142 [−0.185, −0.095], p 5e-4 |
| Py PN_kin, Pythia jets | +0.066 [0.023, 0.105] | +0.241 | −0.206 |
| Hw ParT_kin, Pythia jets | +0.282 [0.259, 0.305] | +0.139 | +0.125 [0.081, 0.169] |
| Hw PN_kin, Pythia jets | +0.221 | +0.007 [−0.046, 0.061], p 0.81 | +0.194 |
| Py ParT_kin, Herwig jets | +0.037 [−0.009, 0.083], p 0.10 | +0.253 | −0.241 |
| Py PN_kin, Herwig jets | −0.020 [−0.063, 0.022], p 0.33 | +0.260 | −0.302 |

H30's own Herwig-jet cells agree with the last two rows:
- Py ParT: +0.235 (ΔR 0.026) vs +0.216 (ΔR 0.312).
- Py PN: +0.176 vs +0.221.

Correlation across the 27 cells of main − c1 with each component:

| scorer | with pT-only | with angle at new pT |
|---|---|---|
| Py ParT | −0.01 | 0.84 |
| Py PN | −0.01 | 0.80 |
| Hw ParT | 0.04 | 0.94 |
| EFN | 0.59 | 0.73 |
| PFN | 0.85 | −0.03 |

The R² of the Py ParT main − c1 map on ΔpT^D (main − c1) across cells is 0.002.

**The k_t 7.47 GeV cells on Herwig jets (H30 data).** Py ParT is −0.138 (ΔR 0.189) and +0.019 [−0.003, 0.040] (ΔR 0.312). Py PN is −0.038 and +0.002. The quark-ward response of the Pythia-trained nets at ΔR 0.312 is absent on Herwig jets.

**Herwig-trained − Pythia-trained, |diff| per cell, low k_t (< 1 GeV) vs high.**

| pair | jets | run | absolute, low | absolute, high | relative (median), low | relative (median), high |
|---|---|---|---|---|---|---|
| ParT | Pythia | fresh | 0.030 | 0.090 | 0.22 | 0.55 |
| ParT | Pythia | H30 | 0.030 | 0.081 | 0.26 | 0.49 |
| ParT | Herwig | fresh | 0.050 | 0.098 | 0.72 | 0.49 |
| ParT | Herwig | H30 | 0.050 | 0.099 | 0.73 | 0.51 |
| PN | Herwig | fresh | 0.055 | 0.191 | 1.00 | 0.61 |
| PN | Herwig | H30 | 0.056 | 0.185 | 1.10 | 0.58 |

The mean map size is 0.06 at low k_t and 0.18 at high k_t. No training-seed replicas exist (`models/qg`, `models/qg_herwig` hold one run each), so the generator difference has no seed-spread reference.

## Competing hypotheses for the k_t 1.29 GeV ΔR shape (Py ParT, Pythia jets)
- **H_L: position in the Lund plane (angle at fixed emission pT) sets the response.** Predicts that the angle-at-new-pT term carries the rise toward small ΔR. Measured slope −0.142 [−0.185, −0.095]: excluded for this model and jet sample.
- **H_P: the pT transfer (leading-particle fraction, pT^D) sets the shape.** Predicts that the pT-only term carries the rise. Measured +0.239 [0.194, 0.288]: consistent.
- **H_G: girth alone.** Predicts a flat main − c1 row. Measured +0.123 on Pythia jets: excluded, as H30 states. On Herwig jets the row slope is +0.037 [−0.009, 0.083], which does not exclude H_G.
- **Herwig-trained ParT has both terms positive** (+0.139, +0.125), so its rise is not from the pT transfer alone.

## Corrected reading
- **Map values.** The H30 cell values reproduce on a fresh sample (2000 jets per generator, seed 3002): +0.296 vs +0.285 and −0.277 vs −0.276 for the two cells checked with CIs.
- **Slope at k_t 1.29 GeV.** For Pythia-trained ParT_kin and PN_kin on Pythia jets, the rise toward small ΔR comes from the transfer of up to about 49 GeV from the hardest particle to the moved particle. The pT-only edit at the source angle gives a slope of +0.24, and the angular move at fixed new pT gives −0.14 to −0.21. It is therefore not a response to the emission's angle in the Lund plane.
- **Herwig jets.** The main − c1 slope is not different from zero for the Pythia-trained nets (p 0.10 and 0.33).
- **Low k_t (0.4 GeV).** The flat ΔR profile is the sum of a pT-only term (+0.07 → −0.05 across ΔR) and an angular term (−0.02 → +0.15).
- **k_t 7.47 GeV.** The quark-ward response is the angular placement of a 25–40 GeV particle at ΔR 0.19–0.31 (−0.40 and −0.17, carried by gluon jets). The pT transfer alone is gluon-ward (+0.15, +0.10). In H30's data, PN_kin (Py) on Herwig jets does not show it (−0.04, +0.00).
- **Generator difference.**
  - In absolute size, Herwig-trained − Pythia-trained differences are 2–3× larger at k_t > 1 GeV, where the maps are 3× larger.
  - Relative to the map size, the difference is larger at k_t < 1 GeV on Herwig jets: median 0.72 vs 0.49 for ParT, 1.00 vs 0.61 for PN. On Pythia jets it is the reverse: 0.22 vs 0.55.
  - The statement "the H1 expectation (differences mainly at low k_t) is not met" therefore depends on the normalisation and on the jet sample.
  - There are no seed replicas.
