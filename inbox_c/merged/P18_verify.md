# P18v: verification of P18 (ParT stage pipeline, C4), 2026-10-20

## Conclusion
In ParT, jet multiplicity and pT can be read from the input embedding and the jet mass from the first block on, but an untrained network of the same shape gives nearly the same read-outs (mass R² lower by 0.04–0.13 depending on its random initialisation). So these read-outs do not show that the variables were learned, and ParT has no "mass last" stage. Prong-shape variables (τ21, τ32) are read only by the trained network. When a jet's prong structure is destroyed by randomising every particle's azimuth about the jet axis, one direction in the jet-averaged representation reproduces a share of the network's response that rises from about 0.02 at the embedding to about 0.7 by blocks 6–7. An untrained network gives a flat share of about 0.5 from block 1 on, so the rise with depth depends on training, and the size of the share does not. Still uncertain: the scramble changes τ21, τ32 and D2 together, so which prong observable the network uses is not resolved.

## Verdict fields
- Reproduced: yes. Readability, calibration gains, t90 indices and all DAS depth profiles reproduce on fresh rows within 0.03, with a different seed and different files.
- Refuted: no for the ordering result (no mass-last stage) and for the readability and gain statements. Three DAS statements need correction (below): the scramble DAS covers only the per-particle part of the edit; IIA levels are not training-specific (untrained nets reach 0.45–0.95); and the soft-particle rise in ParT_full is matched by an untrained net.
- Confidence that the P18 reading is correct as stated: 0.6.

## Code review (interp/p18_stage_pipeline_part.py)
- Sites. `SKEY` maps emb → `model.embed`, b1…b8 → `model.blocks[0..7]`, c1/c2 → `model.cls_blocks[0..1]`. Forward hooks read the module outputs: (P, N, C) for particle modules and (1, N, C) for class blocks, where `out[0]` is the class token. The indices and modules are correct.
- Padded particles. `jetdistill.part.network.inputs` fills padded slots with copies of real particles (mask 0). The pooling weight `w = hypot(px, py) * mask` is 0 on those slots, so the copies do not enter the pooled read-out. N is computed from the mask, and `kin` and `jmass` are masked. The DAS shift is also added to padded slots. These slots are key-padding-masked in attention and zeroed after the embedding, so the shift has no effect there. No bug.
- Rows. `load_rows` takes `rng.choice` random rows of each class file (no first-k). The train/test split is a seeded permutation. The P1 rows (3,000–3,999 of each class) are used only for the reference accuracy, as in P1. Accuracy: 85.93 % / 74.47 % (full / kin).
- Build. `part_core.build` uses η = jet_eta + sign(jet_eta)·Δη, the fixed version.
- Edits match their descriptions. Mean true changes on the P18 sample:
  - φ+0.06: Δlog m_inv = 0.0000, Δlog m_A = +0.074.
  - Dilation: Δlog m_inv = +0.073, Δlog m_A = +0.074.
  - Scramble: Δτ21 = +0.29, Δτ32 = +0.095, Δlog D2 = +0.28, Δlog m_inv = −0.034; N, pT and m_A are fixed. On the fresh rows, the per-jet correlation of Δτ21 with Δlog D2 is 0.62 and with Δτ32 is −0.51.
  - soft5: ΔN = +5.0, Δlog pT = +0.026, Δlog m = +0.053.
  - pT×1.1: all masses also scale by 1.1.
- DAS counterfactual. For every edit except soft5, the counterfactual keeps the base 4-vectors (`das_data`: `(xe, vb, mb)`), so the pair features stay those of the base jet. This affects the scramble most:
  - The per-particle-only scramble changes 5.9–6.4 % of ParT_full calls (DAS evaluation jets). The full scramble changes 46.2–46.7 %.
  - P18 lists this choice under Issues. Its Conclusion nevertheless calls the per-particle edit "a prong-destroying edit", and as an input to ParT_full that per-particle edit leaves most of the prong information in place.
- Controls.
  - The `perm` net row-permutes each layer's outputs without permuting the next layer's inputs. This destroys the function (accuracy 10–12 %), and its logit responses are near 0, so it is a weak null for IIA.
  - The `rot` control does not test anything for the ridge probe, which is close to rotation-invariant: R² matches ParT_full to 0.01.
  - P18 ran DAS on `perm` only, not on the untrained net. This control was missing and was added here.
- Statistics. The Holm floor (1/1,001 × 55 = 0.055) and the degenerate t90 CIs are reported correctly in P18.
- Figure `results/part/p18_stage_pipeline.png` matches the JSON. The dotted lines in the top row are the maximum over perm and init. The DAS panels show only the perm net; adding an untrained-net line would change how the panels read (see below).

## Rerun 1: fresh rows, untrained-net DAS added
Job anrunw-ai-c-p18-verify (Complete, 102 min, GTX 1080 Ti); script interp/p18v_verify_part.py, which calls the P18 functions unchanged.
- Sample: `test_20M/<sample>_101.root` (P18 used `_100`), 2,000 random rows per class with `default_rng(1018)`, N = 20,000.
- All P18 seeds were replaced by seeds based on 1018: perm permutation, untrained init (torch seed 1018), splits and DAS starts.
- Accuracy on this sample: 86.41 % (full) and 74.26 % (kin); perm 10.4 %, init 10.4 %.
- φ+0.06 moves true-W → Z from 10.8 to 31.4 % (full).

Readability (ridge R², test half), fresh compared with P18:
- Every trained-model value agrees within 0.03 at every site.
- t90 sites:
  - Identical for N, pT, ⟨ΔR²⟩, τ21, m_inv and m_A.
  - τ32: b3 [b3, b4] (full) and b2 [b2, b3] (kin). P18 had b4 and b3.
- Order tests:
  - mass − prong t90 = −0.38 sites (full) and −0.42 (kin). One-sided p = 1.0 in H1's direction, as in P18.
  - prong − low = +1.04 (full) and +1.09 (kin), p = 0.001.
- Untrained net (seed 1018), log m_A at b1 / b4 / b8: 0.851 / 0.811 / 0.799. Trained ParT_full: 0.936 / 0.940 / 0.924.
  - With this seed the margin is 0.085–0.13. With P18's seed 18 it was 0.04–0.07.
  - So "within 0.04–0.07" depends on the initialisation seed.
- Calibration gains reproduce within 0.06.
  - τ21 under the scramble: 0.80 at b1 and 0.81 at b8 (full). Both untrained seeds: 0.00–0.05.
  - m_A under dilation: 0.97 at b1 (full). Untrained: 0.93.

DAS 1-d IIA, trained nets: the fresh values agree with P18 within 0.04 at every site, edit and model.

Added control 1: DAS refitted on the untrained ParT_full (seed 1018), with the same fit and evaluation. The table gives IIA at emb / b1 / b4 / b8.

| edit (DAS counterfactual as P18) | ParT_full | untrained | ParT_kin |
|---|---|---|---|
| φ+0.06 | 0.62 / 0.57 / 0.49 / 0.43 | 0.61 / 0.66 / 0.64 / 0.62 | 0.89 / 0.77 / 0.63 / 0.73 |
| dilation | 0.87 / 0.86 / 0.80 / 0.70 | 0.72 / 0.75 / 0.73 / 0.70 | 0.89 / 0.87 / 0.83 / 0.79 |
| pT×1.1 | 0.95 / 0.94 / 0.87 / 0.74 | 0.95 / 0.95 / 0.94 / 0.92 | 0.96 / 0.94 / 0.89 / 0.83 |
| soft5 | 0.31 / 0.31 / 0.32 / 0.43 | 0.51 / 0.64 / 0.63 / 0.62 | 0.31 / 0.31 / 0.34 / 0.52 |
| scramble (per-particle only) | 0.01 / 0.06 / 0.21 / 0.38 | 0.33 / 0.54 / 0.53 / 0.52 | 0.22 / 0.39 / 0.66 / 0.66 |

- Test of the depth change. H0: trained (b8 − emb) = untrained (b8 − emb), i.e. the depth profile is set by the architecture. H1: training changes the depth profile.
- Difference-in-differences with 1,000 paired jet bootstraps (1,000 evaluation jets), ParT_full:
  - φ: −0.20 [−0.23, −0.18]
  - dilation: −0.15 [−0.17, −0.12]
  - pT: −0.18 [−0.20, −0.16]
  - scramble: +0.18 [+0.14, +0.22]
  - soft5: +0.007 [−0.025, +0.040], p = 0.66
- ParT_kin: φ −0.18, dilation −0.07, pT −0.11, scramble +0.26 [+0.21, +0.30], soft5 +0.11 [+0.09, +0.13].
- All p = 0.001 except full soft5. Holm over 10 tests gives ≤ 0.01 for the nine cells other than full soft5.
- H0 is excluded for φ, dilation, pT and scramble in both models, and for soft5 in ParT_kin. It is not excluded for soft5 in ParT_full.
- IIA levels: the untrained net gives a higher IIA than ParT_full in 44 of 55 cells (CI < 0). This includes every scramble and soft5 cell, and φ from b1 on.
  - The P18 statement "trained − permuted +0.03 to +0.86" therefore depends on the permuted net, whose IIA is near 0.
  - IIA levels alone do not separate a trained from an untrained ParT.

## Rerun 2: full-jet scramble counterfactual, two untrained seeds, logit-change spectra
Job anrunw-ai-c-p18-verify2 (Complete, 41 min, GTX 1080 Ti, smoke and full run in one job); script interp/p18v2_fullcf_part.py.
- Sample: `test_20M/<sample>_102.root`, 500 random rows per class with `default_rng(2018)`, N = 5,000. Of these, 4,000 jets are for DAS training and 1,000 for evaluation, split by random permutation.
- DAS settings as P18. 1,000 jet bootstraps; 1,000 linearised random directions.
- Edits compared: scr, the per-particle inputs only (as P18), and scrF, the fully scrambled jet with pair features scrambled too.
- Models: ParT_full, ParT_kin, and untrained ParT_full at torch seed 18 (P18's init) and seed 1018 (P18v's init).

1-d IIA at emb / b2 / b4 / b6 / b7 / b8 / c1 (calls changed by the counterfactual on the 1,000 jets in brackets):

| model | scr (per-particle) | scrF (full jet) |
|---|---|---|
| ParT_full | 0.01 / 0.14 / 0.19 / 0.29 / 0.34 / 0.34 / 0.40 (5.9 %) | 0.02 / 0.37 / 0.59 / 0.68 / 0.68 / 0.65 / 0.55 (46.2 %) |
| ParT_kin | 0.19 / 0.46 / 0.66 / 0.71 / 0.70 / 0.68 / 0.60 (32.1 %) | 0.11 / 0.25 / 0.58 / 0.70 / 0.72 / 0.71 / 0.64 (63.4 %) |
| untrained, seed 18 | 0.28 / 0.45 / 0.45 / 0.45 / 0.45 / 0.45 / 0.42 (9.3 %) | 0.27 / 0.44 / 0.45 / 0.45 / 0.45 / 0.44 / 0.42 (9.4 %) |
| untrained, seed 1018 | 0.34 / 0.54 / 0.53 / 0.53 / 0.53 / 0.52 / 0.51 (5.1 %) | same as scr within 0.01 (5.1 %) |

- Random-direction null (linearised, 1,000 draws): maximum IIA ≤ 0.11 (full) and ≤ 0.16 (kin) over all sites. DAS exceeds this maximum at every site from b1 on.
- Full-jet scramble, ParT_full:
  - b8 − emb = +0.63 [+0.61, +0.64].
  - Difference-in-differences against the untrained nets: +0.45 [+0.43, +0.47] (seed 18) and +0.44 [+0.42, +0.47] (seed 1018), p = 0.001 each.
  - Trained − untrained per site: −0.25 to −0.36 at emb–b1, about 0 at b3–b4, and +0.13 to +0.24 at b5–b8 (seed 18). With seed 1018 the values are +0.06 to +0.16.
- ParT_kin full-jet scramble: difference-in-differences +0.43 [+0.40, +0.45] and +0.42 [+0.39, +0.45].
- Per-particle scramble, ParT_full: lower than both untrained nets at every site (by −0.02 to −0.49). Difference-in-differences +0.16 [+0.11, +0.21] and +0.15 [+0.10, +0.19].
- Call IIA (the share of call-changed jets that the interchange moves to the counterfactual call) for the full-jet scramble in ParT_full is at most 0.26 (b5–b6). So one direction recovers about 0.68 of the logit change but a minority of the call flips.
- In the untrained nets, scr and scrF give the same responses, because the untrained pair embedding has almost no effect on the logits.
- Logit-change spectra (uncentred second moment of L_cf − L_base over the 1,000 jets; top-1 eigenvalue share / participation ratio):
  - Trained ParT_full: 0.35–0.55 / 2.7–4.6 across edits.
  - Untrained: 0.40–0.90 / 1.2–4.4. The largest top-1 share is for pT×1.1 (0.84–0.90).
  - The untrained responses are somewhat more one-dimensional, but not enough to account for the IIA difference alone. This was not tested further.

## Assessment of the P18 reading
1. "N and pT readable from the embedding, mass from block 1 with no later increase; untrained ParT reaches the same for N/pT and comes within 0.04–0.07 for mass."
   - Reproduced.
   - The mass margin is 0.04–0.13 depending on the untrained seed.
2. "Prong variables are where the trained network exceeds both control networks; scramble gain 0 at emb and in controls, 0.3–0.8 from b1–b4 on."
   - Reproduced on fresh rows, with gains 0.80 (τ21) and 0.17 → 0.39 (τ32) in full, and 0.00–0.05 in both untrained seeds.
3. "Probe directions do not carry the edit responses; the fitted DAS directions do."
   - The DAS directions beat random directions (all cells from b1 on).
   - An untrained ParT also gives 1-d IIA of 0.45–0.95 for every edit, which is higher than ParT_full for scramble, soft5 and φ (after emb).
   - The IIA level is therefore not evidence of a learned 1-d variable. Only the depth profile differs from the untrained net.
4. "pT, dilation, phi: share highest at the embedding and falls with depth."
   - Reproduced, and the fall is training-specific: the untrained net is flat, and the difference-in-differences is −0.15 to −0.20 (full).
5. "Scramble and soft-particle: share rises with depth (b2–b8)."
   - Scramble: reproduced and training-specific (difference-in-differences +0.15 to +0.18 per-particle, +0.44 full-jet).
   - The P18 numbers (0.05 → 0.37) are for the per-particle-only scramble, which changes about 6 % of ParT_full calls. With the full-jet scramble (46 % of calls), the share rises from 0.02 to 0.68 and exceeds the untrained nets from b5 on.
   - soft5 in ParT_full: the rise (+0.12) is matched by the untrained net (difference-in-differences +0.007 [−0.025, 0.040]), so it is not training-specific. In ParT_kin it is (+0.11 [+0.09, +0.13]).
6. Hypotheses: H0 (flat) excluded; H1's mass-last order not observed; H_axis-alt consistent with m_A readability.
   - Reproduced (mass − prong t90 < 0 on fresh rows, p = 1.0 in H1's direction).
7. Conclusion sentence "A single direction's share of the response to a prong-destroying edit rises from about 0.05 at the embedding to about 0.4 by block 7".
   - The numbers are those of the per-particle-only scramble, which as an input to ParT_full mostly does not change prong structure (about 6 % of calls change).
   - It also omits that an untrained net gives about 0.5 at every block.
   - Corrected in the Conclusion above.

## Corrected reading
- **Readability.**
  - N and log pT are readable from the embedding (R² 0.99), and the masses (m_inv, m_A, ⟨ΔR²⟩) from b1 (R² 0.94), with no later increase.
  - An untrained ParT gives R² 0.99 for N and pT and 0.80–0.89 for the masses, a margin of 0.04–0.13 depending on its initialisation.
  - t90 order: (N, pT) at emb; then masses and τ21 at b1; then τ32 at b2–b4.
  - H1's mass-last order is not observed in either sample.
- **Prong variables.** τ21 and τ32 are the only targets whose readability and calibrated scramble response depend on training:
  - gain 0.77–0.80 (τ21) in the trained net against 0.00–0.05 in untrained nets of two seeds;
  - R² 0.80 against 0.22–0.38.
- **One-dimensional interchange (DAS, pooled particle tokens, base pair features kept except where stated).**
  - DAS directions exceed 1,000 random directions at every site from b1 on.
  - Untrained ParTs also reach 1-d IIA of 0.45–0.95, so IIA levels do not indicate learned variables.
  - Depth profiles that differ from untrained nets:
    - The fall with depth for pT×1.1, dilation and φ+0.06 (difference-in-differences −0.15 to −0.20, ParT_full).
    - The rise with depth for the azimuth scramble: per-particle-only edit 0.01 → 0.38; full-jet edit 0.02 → 0.68 at b6–b7, against a flat 0.45–0.54 in untrained nets; difference-in-differences +0.44 [+0.42, +0.47].
  - The soft-particle rise is training-specific in ParT_kin only.
- **A8.** No mass-last stage like JEDI-linear's is observed. The mass-type read-out is present from b1 at the level of an untrained net. The scramble-sensitive direction forms over b2–b7.

## Issues
- The DAS tests use one pooled direction (pT-weighted mean of the particle tokens), so per-particle or multi-dimensional encodings are not tested.
- Holm floor: with 1,000 bootstraps, p = 0.001 per test.
- Rerun 2 used 1,000 evaluation jets and 4,000 DAS-training jets, as P18.
- The untrained-net comparison uses two seeds only.

## Explainer additions (for site/explain.py)
- Check (fresh rows, `_101`, seed 1018): every ridge $R^2$ and every DAS IIA of the trained nets agree with P18 within 0.03–0.04. The mass-before-prong order is reproduced.
- Check (untrained network): define the depth change as $\Delta = \mathrm{IIA}_{b8}-\mathrm{IIA}_{emb}$, and the training effect as $\Delta_{\rm trained}-\Delta_{\rm untrained}$ with a paired jet bootstrap. For the full azimuth scramble this is $+0.44\,[0.42,0.47]$. For 5 soft particles in ParT_full it is $+0.007\,[-0.025,0.040]$.
- Check (edit scope): the per-particle-only scramble changes 6 % of ParT_full's calls and the full-jet scramble 46 %. The 1-d interchange recovers $0.34$ vs $0.68$ of the logit change at block 7, where $\mathrm{IIA}=1-\sum_j\lVert L^{\rm int}_j-L^{\rm cf}_j\rVert^2/\sum_j\lVert L^{\rm base}_j-L^{\rm cf}_j\rVert^2$.
- Why it matters: being readable, or being carried by a single direction, is not by itself evidence that a network learned a variable, because an untrained network of the same shape reads N, pT and mass and gives 1-d IIA of about 0.5. The evidence for learned stages comes from the differences to the untrained network: the prong read-outs, and the depth profile of the interchange.

## Files
- /Users/anrunw/Documents/aaron-interp/interp/p18v_verify_part.py (rerun 1)
- /Users/anrunw/Documents/aaron-interp/interp/p18v2_fullcf_part.py (rerun 2)
- /Users/anrunw/Documents/aaron-interp/results/part/p18_verify.json, p18_verify.png
- /Users/anrunw/Documents/aaron-interp/results/part/p18_verify2.json (and p18_verify2_smoke.json)
- Logs: /Users/anrunw/Documents/aaron-interp/logs/c-p18-verify.log, c-p18-verify2.log
