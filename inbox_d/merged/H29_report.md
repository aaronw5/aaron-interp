# H29: JEDI-linear W-pair-mass null retested with each edited jet rebuilt about its own axis (claim D1)

Script `interp/h29_jedi_wpair_rebuild.py`. Job `anrunw-ai-d-h29-run` (AI_PY=conda, 2 CPU / 4 GiB; edit → small → jedi (jax env) → stats; wall 1,009 s, peak 2.5 GiB, about 1 core on average). The first launch was evicted when its node went NotReady and was relaunched under the same name with no change. Smoke run: `d-h29-smoke` (300 per class). Figure job: `d-h29-fig`. Results are in `results/physics/`: `h29.json` (all statistics, 468 tests with Holm), `h29_edit.json` (sample, edit counts, axis shifts, m_A changes, references), `h29_scores_{jedi,small}.npz` and `h29_refs.npz` (per-jet logits and reference scores), and `h29_wpair_rebuild.png`. The per-jet inputs `h29_inputs.npz` are on the volume only. Log: `logs/d-h29-run.log`. The explainer entry `H29_EXPLAINER` is in `site/explain.py`.

## Conclusion

When each edited hls4ml top jet is rebuilt about its own axis, JEDI-linear lowers its top score under a W-pair-mass edit by about as much as a reference classifier that never sees the pair mass. At 60 GeV the two are equal (1.01× the reference), and the drop is less than half that of a classifier given the pair mass. So JEDI-linear's "no W-mass constraint" (D1) survives the rebuild that overturned the ParT null. The earlier fixed-axis build gave the opposite sign. That came from the edit's control moving the jet axis. Through JEDI-linear's axis-relative mass √(M0·M2) (the pT-weighted angular spread about the given axis), the fixed axis turned that move into an apparent mass change. Still uncertain: the pair-mass reference separates from the side-effect reference only at 60 and 100 GeV on hls4ml jets, the 100 GeV set has 144 jets, and the dose test on jets common to all four targets has only 29 jets.

## Procedure

- **Sample.** hls4ml test archive (`data/n64/full_test.npz`). 2,000 jets per class were drawn with `default_rng(2901)` from rows not used by H3 (rng 501, 4,000/class), H3v (rng 601, 2,000/class) or H23 (rng 9523, 1,000/class); overlap 0. That is N = 10,000 jets, random rows per class. Top-vs-QCD pair: t (2,000) vs q + g (4,000). W/Z jets are used only for the 5-class accuracy. Each jet is its 64 stored particles, taken as massless four-vectors at (η, φ) = (Δη, Δφ). The identity rebuild reproduces the stored inputs to 1.2e-4.
- **Builds.**
  - **st:** inputs (pT, Δη, Δφ) about the stored (0, 0) axis, as in H3/H3v.
  - **rc:** inputs about each variant's own E-scheme axis (the four-momentum sum of the 64 particles).
  - The unedited E-scheme axis lies 0.0015 rad (median; 99th percentile 0.013) from (0, 0).
- **H4-type edit** (`h4_verify.edit_set`, as in H23/H23v):
  - Three exclusive-kT subjets. The W pair is the pair closest to 80.4 GeV.
  - Tops with 105 < mSD < 210 GeV and 65 < mW < 95 GeV (1,197 of 2,000) are moved to 60, 70, 90 or 100 GeV. QCD jets in the mSD window with mW 50–70 or 90–115 GeV (279) are moved to 80.4 GeV (Q80).
  - Acceptance: |Δm| < 0.5 GeV, |Δτ32| < 0.02, and |mW − t| < 5 GeV (W) or |mW − mW0| < 2 GeV (N).
  - Analysis sets (W ∩ N ∩ R accepted): T60 264, T70 528, T90 444, T100 144, Q80 32.
  - Controls:
    - N: the non-W pair is given the same relative mass change, with the W pair fixed.
    - R: a rotation about the axis with the W edit's pT-weighted displacement.
    - TN / TW: the unedited jet is translated rigidly by the axis shift that N / W induce.
- **H3v k-means edit** (`h3_verify.wshift_all`, the edit behind the D1 null):
  - The W pair is moved to 60 / 100 GeV at fixed jet mass. τ32 is not held.
  - Control B holds the W pair at its mass and moves the non-W pair by the same Δm. Control A is H3's control, with the W pair free.
  - Sets: H3v_60 878 jets, H3v_100 717 jets (1,533 tops have a W candidate within 15 GeV of 80.4).
- **Models.**
  - JEDI-linear official n64 3-feature (`jedi_core.Jedi`), and the same network with redrawn weights (seed 0) as the null.
  - From `models/small`, 5-class: dsets-d64-n64, jedis-d64-n64 and tf-2b-d32-n64, seeds 0 and 1.
  - From H16, binary hls4ml t vs q/g: PFN with pT-fraction input (hlsA-frac-pfn) and with absolute-pT input (hlsA-abs-pfn), seeds 0 and 1.
- **Score.** Top-vs-QCD log-odds: 5-class s = L_t − logsumexp(L_q, L_g); binary s = L_t − L_bkg.
- **References.** Trained on 25,000 t + 25,000 q/g jets from the hls4ml training archive (rng 2902, dev split excluded, first 64 particles) with `h19_wpair_cross.fit_refs`:
  - MLPx: mSD, τ1–3, τ32, τ21, multiplicity, sorted subjet pT fractions and masses; no pair mass.
  - MLPxW: MLPx plus mW.
  - LR2: a histogram likelihood ratio in (mSD, τ32).
  - The reference features are axis-independent.
- **Metric.**
  - Per variant v, Δ_v = s_v − s_id on the same build. The primary quantity is W − N (H3v: E − B).
  - Each quantity gets a 95 % bootstrap CI over jets (B = 2,000) and a ratio to the MLPx and MLPxW W − N, with CI.
  - Tests:
    - one-sided t-test, and sign-flip permutation (10,000), of (W − N)_model − (W − N)_MLPx in the mW direction;
    - two-sided t-test against MLPxW;
    - McNemar W vs N on calls at the eS50 threshold (median score of unedited tops, same build), and on the top argmax call;
    - st − rc (two-sided);
    - translation residual (ΔN_st − ΔN_rc) − ΔTN_st;
    - dose test, paired on the tops accepted at all four targets.
  - Holm is applied over 468 tests (models × sets × builds: the vs-MLPx test and McNemar; st − rc per model × set; dose).
- **Hypotheses.**
  - H0 (D1 null): the rc W − N equals MLPx's, with no ordering in |mW − 80|.
  - H1 (W-mass use): the rc W − N is at least 2× MLPx and approaches MLPxW, and grows monotonically with |mW − 80|.
  - H2 (A4 axis-relative mass): st and rc differ mainly through the translation term TN.

## Unedited performance on this sample (reference value in parentheses)

| model | 5-class acc st / rc | top eff (argmax t) | AUC t vs q/g |
|---|---|---|---|
| JEDI-linear official | 80.49 / 80.45 % (80.92 % on 260k) | 82.4 % | 0.962 |
| JEDI-linear redrawn | 20.0 % (calls every jet t) | 100 % | 0.595 |
| Deep Sets s0 / s1 | 80.37 / 80.49 % (80.64 / 80.81) | 82.2 / 82.5 % (83.1 / 84.0) | 0.958 / 0.957 |
| JEDI-style s0 / s1 | 79.35 / 79.18 % (79.68 / 79.28) | 81.8 / 82.5 % | 0.957 |
| transformer s0 / s1 | 75.65 / 73.16 % (76.19 / 73.42) | 80.8 / 80.3 % | 0.953 / 0.944 |
| PFN frac s0 / s1 (binary) | acc t vs q/g 88.27 / 88.25 % (88.92 / 88.83, balanced test) | 90.7 / 90.5 % | 0.953 / 0.954 (0.956) |
| PFN abs s0 / s1 (binary) | 89.65 / 89.82 % (90.17 / 90.16) | 91.4 / 91.2 % | 0.963 (0.964) |
| MLPx / MLPxW / LR2 | binary acc 88.1 / 88.0 / 82.7 % | – | 0.951 / 0.951 / 0.920 |

Every unedited value is within 0.7 points of its reference, at most 1.6 binomial standard errors (0.4 points at N = 10,000; the PFN references are on a balanced test set). Transformer seed 1 is a weaker checkpoint (73.4 % on its own test set). Building the unedited jets about their own axis changes accuracy by at most 0.06 points.

## Numbers

### Axis shift and m_A (median)

| set | axis shift W (rad) | axis shift N (rad) | Δlog m_A about (0,0), N | Δlog m_A about (0,0), TN | Δlog m_A about (0,0), W | Δlog m_A, rc builds |
|---|---|---|---|---|---|---|
| T60 | 0.010 | 0.076 | +0.090 | +0.094 | +0.002 | ≤ 0.0004 |
| T70 | 0.005 | 0.039 | +0.028 | – | – | – |
| T90 | 0.006 | 0.028 | +0.012 | – | – | – |
| T100 | 0.012 | 0.043 | +0.030 | – | – | – |
| H3v_60 | 0.0008 (E) | 0.0054 (B) | – | – | – | – |
| H3v_100 | 0.0010 (E) | 0.0028 (B) | – | – | – | – |

### JEDI-linear, W − N in top log-odds

Each value is the mean [95 % CI]. "×MLPx" is the ratio to MLPx's W − N.

| set (n) | MLPx | MLPxW | st | rc | rc ×MLPx | rc ×MLPxW | rc − MLPx, one-sided p (Holm) |
|---|---|---|---|---|---|---|---|
| T60 (264) | −0.32 [−0.40, −0.23] | −0.77 [−0.85, −0.67] | +0.92 [+0.72, +1.12] | −0.32 [−0.47, −0.18] | 1.01 [0.53, 1.65] | 0.42 [0.23, 0.61] | −0.004 [−0.17, +0.17], p 0.48 (1) |
| T70 (528) | −0.08 [−0.12, −0.03] | −0.13 | +0.40 [+0.30, +0.50] | −0.15 [−0.23, −0.08] | 2.04 [0.85, 5.55] | 1.21 [0.55, 2.18] | −0.08, p 0.049 (1) |
| T90 (444) | −0.12 [−0.16, −0.07] | −0.24 | +0.33 [+0.22, +0.46] | +0.15 [+0.07, +0.24] | −1.29 | −0.61 | +0.26, p 1 (1) |
| T100 (144) | −0.20 [−0.32, −0.07] | −0.85 [−1.02, −0.69] | +0.11 [−0.20, +0.44] | −0.34 [−0.56, −0.10] | 1.69 [0.64, 3.88] | 0.40 [0.13, 0.63] | −0.14, p 0.098 (1) |
| Q80 (32) | +0.06 [−0.20, +0.33] | +0.38 | −0.03 [−0.29, +0.20] | +0.02 [−0.33, +0.33] | – | – | p 0.59 (1) |
| H3v_60, E − B (878) | −0.70 [−0.77, −0.63] | −0.86 | −0.37 [−0.45, −0.30] | −0.47 [−0.54, −0.39] | 0.66 [0.56, 0.78] | 0.54 [0.46, 0.63] | +0.24, p 1 (1) |
| H3v_100, E − B (717) | +0.71 [+0.62, +0.79] | +0.50 | +0.16 | +0.16 [+0.08, +0.24] | 0.23 | – | not informative: both references move up |

### JEDI-linear, axis term

At T60, ΔN is −1.55 [−1.74, −1.35] in the st build and −0.38 [−0.50, −0.24] in rc. The st − rc difference of W − N is +1.24 [+1.03, +1.44] (Holm 1e-22). The translation-only control gives ΔTN_st = −1.10 [−1.28, −0.91], and the residual (ΔN_st − ΔN_rc) − ΔTN_st is −0.07 [−0.25, +0.09].

| set | st − rc | ΔTN_st | residual |
|---|---|---|---|
| T60 | +1.24 | −1.10 | −0.07 [−0.25, +0.09] |
| T70 | +0.56 | −0.38 | −0.15 [−0.25, −0.05] |
| T90 | +0.18 | −0.20 | −0.01 [−0.09, +0.07] |
| T100 | +0.44 | −0.40 | −0.10 [−0.26, +0.08] |
| H3v_60 | +0.09 | −0.06 | −0.03 |

### JEDI-linear, calls at the eS50 threshold (% of jets whose call changes, W / N)

| set | st | rc |
|---|---|---|
| T60 | 14.0 / 44.3 % (McNemar 13/89, p 4e-15, Holm 1.8e-12) | 15.5 / 15.9 % (28/23, p 0.58) |
| T70 | 7.6 / 18.4 % (Holm 0.0045) | 7.0 / 9.3 % (p 0.52) |
| H3v_60, E / B | 13.3 / 15.0 % | 12.9 / 10.9 % (118/33, p 2e-12; this edit does not hold τ32, and MLPx also responds) |

### Null and other models

The redrawn-weight JEDI-linear gives |W − N| ≤ 0.003 in every cell.

The table gives rc W − N, with the ratio to MLPx in parentheses. ¹ marks cells with Holm < 0.05 on the vs-MLPx test.

| model | T60 | T70 | T90 | T100 |
|---|---|---|---|---|
| Deep Sets s0 / s1 | −0.33 (1.04×) / −0.21 (0.65×) | −0.22 (2.85×)¹ / −0.13 (1.66×) | +0.16 / +0.16 | −0.23 (1.16×) / −0.11 (0.57×) |
| JEDI-style s0 / s1 | +0.07 / −0.22 (0.70×) | +0.03 / −0.11 (1.43×) | +0.14 / +0.12 | +0.10 / −0.14 |
| transformer s0 / s1 | −0.38 (1.19×) / −0.22 (0.70×) | −0.25 (3.29×)¹ / −0.12 (1.64×) | +0.11 / +0.15 | −0.33 (1.65×) / −0.11 |
| PFN frac s0 / s1 | −0.58 (1.83×)¹ / −0.55 (1.76×) | −0.37 (4.85×)¹ / −0.35 (4.58×)¹ | +0.06 / +0.10 | −0.64 (3.20×)¹ / −0.62 (3.11×)¹ |
| PFN abs s0 / s1 | −0.58 (1.84×)¹ / −0.53 (1.69×) | −0.42 (5.62×)¹ / −0.36 (4.69×)¹ | +0.17 / +0.14 | −0.43 (2.14×) / −0.44 (2.20×) |

- PFN at T60: 0.70–0.76× MLPxW.
- PFN at T70: 2.7–3.3× MLPxW. The PFN T70 response is also present in the st build (4.0–6.7× MLPx; Holm ≤ 0.003).
- The st builds of every small model at T60 are positive (+0.13 to +1.37), as for JEDI-linear.
- The non-translation residual is larger in some small models than in JEDI-linear: transformer s0 −0.57 [−0.75, −0.39], PFN −0.50 to −0.61, Deep Sets −0.25 to −0.38 at T60.
- H3v_60, rc E − B: every small model lies at −0.01 to 1.01× MLPx.

### Dose tests

- **Paired, tops accepted at all four targets (n = 29).** No model has W − N more negative at 60 than at 70 GeV, or at 100 than at 90 GeV, after Holm (smallest p 0.024 for transformer s1 rc, Holm 1). JEDI-linear rc means: T60 −0.19, T70 −0.06, T90 +0.19, T100 +0.60.
- **Exploratory, not in the Holm family: Welch test of the rc excess over MLPx between the full sets.**
  - JEDI-linear: 60 − 70 is +0.075 (SE 0.10, one-sided p 0.77); 100 − 90 is −0.40 (SE 0.12, p 3e-4).
  - The PFNs, Deep Sets s0 and transformer s0 show the same pattern: no difference between 60 and 70 GeV, a negative 100 − 90 difference (p ≤ 0.001).
  - The 100 − 90 difference comes from T90, where all trained nets go up by +0.06 to +0.17 while both references go down.

## Reading

- **H0 vs H1 for JEDI-linear (rc).**
  - At 60 GeV the response equals MLPx (1.01× [0.53, 1.65]), and the CI excludes MLPxW (0.42× [0.23, 0.61]; two-sided p 4e-7).
  - At 100 GeV the response lies between the references (1.69× MLPx [0.64, 3.88], 0.40× MLPxW [0.13, 0.63]).
  - At 70 GeV it is 2.04× MLPx, but the CI is [0.85, 5.55] and Holm is 1. At 90 GeV its sign is opposite to both references.
  - No vs-MLPx test passes Holm. The paired dose test shows no ordering.
  - So the H1 prediction (≥ 2× MLPx, approaching MLPxW, monotone in |mW − 80|) is not met in any set where the references separate (60, 100 GeV). The plan's D1-null prediction (within the MLPx CI, no ordering) is met at 60 GeV and in the dose test.
- **H2 (st vs rc through translation).** For JEDI-linear, the st − rc difference matches the translation control within 0.1 at 60, 90 and 100 GeV, and within 0.15 at 70 GeV.
  - The fixed-axis build therefore measured mainly JEDI-linear's response to a shift of the constituents relative to the axis. In that build the non-W control moved the axis by 0.076 rad and raised m_A about (0, 0) by 9 %.
  - This is the axis-relative mass of A4. It gave a W − N of the opposite sign (+0.92 at 60 GeV).
- **The H3v edit behind the original D1 null** moves the axis by ≤ 0.006 rad, so st and rc agree (+0.09 difference). JEDI-linear responds at 0.66× MLPx. That edit does not hold τ32, and MLPx and MLPxW are close there (−0.70 vs −0.86), so it separates the readings less well than the H4-type edit.
- **Cross-model.** The hls4ml-trained PFNs respond beyond the side-effect reference at 60, 70 and 100 GeV. Their 60 GeV response stays below MLPxW, and their excess over MLPx is no larger at 60 than at 70 GeV. The other small nets lie at the MLPx level except one seed each of Deep Sets and the transformer at 70 GeV.
- **Not shown.**
  - MLPx and MLPxW have the same AUC (0.951) on hls4ml t vs q/g, so the pair mass adds little to a substructure classifier on this dataset. The positive control separates only at 60 and 100 GeV.
  - Q80 has 32 jets.
  - The 60 GeV W edit lies off the natural-jet manifold (H4: C2ST 0.98 on Top Landscape; not measured on hls4ml).

## Effect on claims

- **D1 (JEDI-linear part):** the null holds under the rebuild rule. Suggested wording: "rebuilt about each edited jet's own axis, JEDI-linear's response to a W-pair-mass edit at fixed m and τ32 equals that of a side-effect classifier without the pair mass at 60 GeV (1.01×) and is 0.40–0.42× a classifier with it at 60 / 100 GeV". The fixed-axis numbers from H3 cannot be cited for the H4-type edit. Confidence is limited by the weak hls4ml positive control.
- **D4 / A4:** a further instance. With the axis held fixed, the non-W control's axis shift is read as an m_A change: −1.10 log-odds from translation alone at 60 GeV.
- **D6 / D11 cross-model:** the hls4ml nets are now scored on the rebuilt basis. The PFNs respond at 1.7–1.8× MLPx at 60 GeV, against 2.9–4.5× for the JetClass ParT/ParticleNet in H23.
