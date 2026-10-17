# X7 verify: axis-convention response, fresh jets, plus an M2-preserving translation control

Script: `interp/x7_verify.py` (imports X7's `matched_rotation`, `scale_jets`, `call_metrics`, `readout_metrics`, `holm`; X4's `axis_moment`, `to_inputs`, `scale_about_axis`; E17's `reaxis`).
Jobs:
- `x7-verify` (jax env, 1 CPU / 2 GiB). The first pod hung for 12 min in D state on ucm-fiona01 while reading the JEDI `.keras` file. I deleted it and resubmitted. The rerun took 579 s with a pod peak of 2047 MiB, which is the limit; the peak includes page cache.
- `x7-verify2` (conda env, 1 CPU / 2 GiB). 152 s, peak 1276 MiB.

Outputs: `results/jedi/x7_verify_jedi.json`, `results/jedi/x7_verify_small.json`, `logs/x7-verify*.log`.

## Code review of `interp/x7_axis_reverse.py`
- **Axis edit.** `reaxis` moves only real particles and wraps dphi; pT and padding are untouched. The hardest particle is slot 0 for every JEDI jet. Small-model inputs are built through `pt_sort` and torch `recentre_hardest`, as in training.
- **Prediction.** `pred = AM[ed] - AM[tax]` is ½Δlog(M0M2) about the origin of each model's inputs, with padded pT = 0. It is correct for both directions.
- **Calibration edit.** It scales about the training axis: about slot 0 for axhard models, about (0, 0) otherwise. Error ≤ 1.6e-6 here.
- **Rotation.** It rotates about the training axis, with displacement ratio 1.000 and ΔM2 ≤ 4e-8.
- **Read-out.** The ridge is fitted on one half of the natural jets and evaluated on the other half. No leakage was found.
- **Control B.** Its fit set excludes the evaluation rows.
- **Small-model rows.** They come from the val archive, which `best.pt` may have used for model selection. This affects only the accuracy check, not the edit responses.
- No bug was found that changes the reported numbers.

## Rerun on fresh jets
Samples:
- JEDI-linear: `jedi_core.sample(3000, seed=211)`, 15,000 jets (X7 used seed 107).
- Small models: val-archive rows drawn with rng 911, 2,000 per class, 10,000 jets, with zero overlap with X4/X7's rng-71 rows.

Edits as in X7, hardest ↔ stored only (no WTA).

Two controls are new:
- **m2t.** The axis edit, followed by scaling every real offset about the new origin by $e^{-\Delta}$. Particles move by the same translation as in the edit, and $\tfrac12\log(M_0M_2)$ about the model's own axis is restored to its natural value (error ≤ 4e-8).
  - Forward (stored-trained models), this lowers the stored-axis spread by $e^{-0.14}$.
  - Reverse (axhard models), it raises the stored-axis spread by $e^{+0.14}$.
- **rnd** (JEDI only). A rigid translation of the same length $|t|$ as the hardest-particle edit, in a random direction (rng 212). It is reported with its own Δ, its own calibration, and an M2-restored version, rnd_m2.

Columns: "changed" is % of calls, with a 95 % bootstrap CI over jets (1,000 resamples); "W→Z/Z→W" counts the switched calls; "read-out" is the ridge response divided by the calibration response, with a CI from 2,000 resamples.

| model (edit) | acc | edit changed | W→Z/Z→W | cal changed | rot changed | m2t changed | m2t W→Z/Z→W | m2t acc after | read-out edit | read-out m2t |
|---|---|---|---|---|---|---|---|---|---|---|
| JEDI-linear stored→hardest | 81.05 → 58.17 | 31.66 [30.89, 32.35] | 1598 / 5 | 33.44 [32.64, 34.21] | 3.33 [3.06, 3.61] | 11.40 [10.90, 11.94] | 95 / 329 | 76.28 | 0.81 [0.80, 0.83] | −0.18 [−0.19, −0.17] |
| Deep Sets s0 stored→hardest | 80.79 → 47.87 | 43.66 [42.66, 44.58] | 1031 / 3 | 35.51 [34.56, 36.45] | 3.21 [2.87, 3.53] | 14.23 [13.50, 14.91] | 44 / 82 | 74.18 | 0.63 [0.61, 0.64] | −0.38 [−0.39, −0.36] |
| Deep Sets s1 stored→hardest | 80.87 → 49.09 | 42.44 [41.44, 43.41] | 1068 / 2 | 34.90 [33.90, 35.76] | 2.63 [2.31, 2.96] | 18.00 [17.26, 18.75] | 100 / 154 | 71.43 | 0.44 [0.42, 0.46] | −0.63 [−0.65, −0.61] |
| Linformer s0 stored→hardest | 81.23 → 56.15 | 33.08 [32.14, 33.95] | 865 / 28 | 34.54 [33.59, 35.39] | 0.88 [0.71, 1.07] | 9.83 [9.26, 10.42] | 156 / 128 | 76.95 | 1.19 [1.11, 1.28] | −0.17 [−0.25, −0.11] |
| Linformer s1 stored→hardest | 80.84 → 54.72 | 35.23 [34.30, 36.16] | 941 / 4 | 35.44 [34.52, 36.35] | 1.20 [0.99, 1.42] | 14.74 [14.04, 15.42] | 181 / 166 | 73.65 | 0.62 [0.59, 0.66] | −0.59 [−0.66, −0.53] |
| Deep Sets s0 axhard→stored | 80.18 → 74.82 | 11.14 [10.56, 11.81] | 119 / 367 | 29.33 [28.44, 30.20] | 2.51 [2.20, 2.80] | 31.12 [30.25, 31.99] | 1207 / 1 | 57.61 | 0.16 [0.14, 0.17] | −0.47 [−0.49, −0.46] |
| Deep Sets s1 axhard→stored | 80.66 → 75.83 | 10.55 [9.90, 11.10] | 49 / 337 | 30.83 [29.98, 31.76] | 2.87 [2.52, 3.19] | 31.27 [30.34, 32.15] | 1063 / 3 | 58.17 | 0.38 [0.37, 0.40] | −0.11 [−0.13, −0.10] |
| Linformer s0 axhard→stored | 80.85 → 65.98 | 21.58 [20.81, 22.40] | 1 / 1033 | 32.72 [31.82, 33.64] | 0.99 [0.80, 1.20] | 23.17 [22.35, 23.97] | 1056 / 28 | 64.28 | 0.58 [0.56, 0.60] | −0.25 [−0.27, −0.23] |
| Linformer s1 axhard→stored | 80.83 → 68.24 | 18.43 [17.62, 19.18] | 0 / 912 | 33.50 [32.54, 34.38] | 1.01 [0.82, 1.21] | 23.23 [22.36, 24.08] | 886 / 68 | 64.37 | 0.59 [0.58, 0.60] | −0.11 [−0.13, −0.10] |

Comparison with X7:
- **Calls changed by the edit.** Within 0.4–1.0 point of X7 in every row. X7: JEDI 30.98; Deep Sets 42.80 / 41.77 forward and 11.09 / 10.12 reverse; Linformer 32.40 / 34.62 forward and 20.54 / 17.59 reverse.
- **Rotation.** Within 0.7 point.
- **Read-out ratio.** Within 0.1. Rerun reverse: 0.16 / 0.38 / 0.58 / 0.59; X7: 0.16 / 0.39 / 0.57 / 0.62.
- **Direction of switched calls.** The sign pattern replicates in all 9 rows.
- **Edit / calibration ratio of calls changed.** Forward 0.95–1.23; reverse 0.34–0.66 (X7: 0.33–0.63).

## New control m2t: a translation with ½log(M0M2) about the training axis held fixed
H0 (H_formula, strict): the call change is set by ½Δlog(M0M2) about the model's training axis, so m2t (Δ = 0) changes about as many calls as the rotation and has no W/Z direction.

### Stored-trained models
- **Calls changed.** m2t changes 9.8–18.0 % of calls (JEDI 11.4 %), against 0.9–3.3 % for the rotation, i.e. 3.4–12.3 times the rotation. This is 0.30–0.42 of the edit's calls changed.
- **Edit vs m2t (McNemar on changed calls).** p ≤ 1e-300 in every row.
- **Direction.**
  - Small models: weak or none (44/82, 100/154, 156/128, 181/166; Holm p 0.025, 0.025, 1, 1).
  - JEDI: 95 W→Z against 329 Z→W (Holm p 2e-30).
  - In every row, the edit's flow (865–1598 W→Z against ≤ 28 Z→W) is absent under m2t.
- **Reading of these numbers.** About 2/3 of the edit's call changes, and all of its W→Z flow, require a change of Δ about the training axis. A further 10–18 % of calls change under a matched translation with Δ = 0.
- **Comparison with the rotation.** The rotation underestimates this non-Δ part 3–12 fold. The JEDI and small-model readings in the thesis show radial (azimuth-independent) objects, so a rotation about the axis lies close to these models' invariant directions.

### Hardest-trained (axhard) models
- **Calls changed.** m2t changes 23–31 % of calls, 1.1–3.0 times the axis edit itself.
- **Direction.** 886–1207 W→Z against 1–68 Z→W. True W called Z rises 2.5–2.8 → 42.6–56.5 %.
- **What changes under m2t.** Δ about the training axis is 0. The stored-axis spread, close to a translation-invariant mass because M1 ≈ 0 about the stored axis, rises by $e^{0.143}$.
- **Comparison with a pure invariant-mass reader.** For one, m2t would match the calibration edit in size with the opposite direction: about 30 % of calls, W→Z.
  - Deep Sets axhard: 31.1 and 31.3 % against calibration 29.3 and 30.8 %.
  - Linformer axhard: 23.2 % against 32.7–33.5 %.
- **Reading.** H0 (strict) is excluded for these models: a Δ = 0 edit produces a W/Z flow of the same size as the Δ-driven flow. The W/Z direction of axhard models is set by ½Δlog(M0M2) about the hardest particle together with a stored-axis (translation-invariant) spread. The reverse edit's Z→W response (10.6–21.6 %) is the net of the two.
- **Agreement with X4.** This matches X4's partial-recentring read-out (0.13–0.41) and X7's own read-out ratios (0.16–0.62). Deep Sets axhard responds to the invariant spread at about the full calibration size; Linformer at about 0.7 of it.

## New control rnd (JEDI): a translation of the same length in a random direction
- **Values.** rnd has Δ = +0.141 (the edit: +0.141).
  - rnd: 35.41 % [34.70, 36.15] of calls changed, W→Z 1594 against 9.
  - Its own calibration: 33.37 %, 1632 / 1.
  - rnd_m2: 10.01 %, 135 / 197.
- **Paired comparison with the edit.** rnd changes 1061 calls the edit does not, and the edit changes 498 calls rnd does not (McNemar p 6e-47).
- **Reading.** For JEDI-linear, translating toward the hardest particle has no special effect beyond the size of Δ, which agrees with the formula. The 3.7-point excess of rnd over the edit is of the order of the m2t residual.

## Per-jet correlation of the Z−W margin change with Δ, conditioned on translation length
- **Without conditioning.** The rerun reproduces X7's per-class Spearman correlations: forward W +0.45 to +0.66 and Z −0.60 to −0.76; reverse W −0.27 to −0.66 and Z +0.25 to +0.73.
- **Partial Spearman given $|t|$.**
  - Values: −0.23 to +0.17.
  - Wrong-sign values: JEDI Z +0.13 [0.09, 0.17], Deep Sets stored Z +0.17 and +0.14.
  - Right-sign values: Linformer axhard W −0.23 and −0.15.
- **Partial Spearman given $|t|/\bar r$.**
  - Forward W: −0.07 to −0.14, the opposite sign to the unconditioned value.
  - Reverse Z: +0.14 to +0.20, the same sign.
- **Reading.** The per-jet association with Δ comes almost entirely from the translation length. Per jet it does not separate ½Δlog(M0M2) from displacement size, so it is not evidence for the formula over a size-of-shift response.

## Holm
- JEDI: 23 tests.
- Small models: 120 tests.
- The m2t direction tests survive Holm for JEDI and all four axhard models (p_Holm ≤ 2e-30). They do not survive for the Linformer stored models.

## Points in the X7 reading that need restating
1. **Rotation ratio.** "A rotation ... changes 3.6–38 times fewer calls" is reproduced. However, a translation with the same displacement and Δ = 0 about the training axis changes 0.30–0.42 of the edit's calls (stored-trained) and 1.1–3.0 times the edit's calls (axhard). The rotation is not an adequate covariate-shift control for these near-radial models.
   - The H_cov prediction "matched shift changes a similar share" is excluded against the rotation, but not against a matched translation.
   - The H_cov prediction on direction is excluded for stored-trained models: m2t has no W→Z flow.
2. **"Follow the sign of Δ about each model's own training axis."** For the axis edit itself this holds in all rows. For axhard models, an edit with Δ = 0 about the training axis gives a W→Z flow of 886–1207 against 1–68. The axhard models' calls follow Δ about the hardest particle plus a stored-axis spread term, and the reverse response is their net.
3. **Range in the thesis note.** "0.25–3.3 % against 6–43 %" leaves out the Deep Sets axhard→WTA rows (1.57 / 1.80 % of calls, 1.6–2.0 times the rotation). It should read 1.6–43 %.
4. **Invariant controls A and B.** These change 0 calls because their inputs are identical bit for bit by construction. This is a code check, not a measurement on a trained tagger, and it does not bear on H_cov.
5. **Per-jet margin correlation.** It is not specific to Δ: it vanishes or changes sign given $|t|$.

## Figure
`results/jedi/x7_axis_reverse.png` is readable. The bars and CIs match the summary JSON, and the right panel's counts match the flow tables.
- Each invariant-control row shows only a rotation bar; the edit and calibration bars are zero-width, with no "0" label.
- The legend's "rotation ... (M0, M2 unchanged)" is accurate.
- The figure has no M2-preserving-translation bar. Adding m2t as a fourth bar would show the non-Δ part of the response.
