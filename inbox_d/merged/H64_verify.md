# H64 verification: boost along the jet axis at fixed invariant mass (skeptic pass)

Loop D. Verifies `inbox_d/H64_report.md`. Script `interp/h64_verify.py` (imports the H64 edits and statistics from `interp/h64_axis_boost_panel.py` / `interp/h12_boost_panel.py`). Jobs: d-h64-verify (GPU, JetClass + Top + QG; 354 + 262 + 153 s), d-h64-verify-hls (CPU, 310 s), d-h64-verify-b (GPU: JetClass with constituent masses held fixed; Top and QG at small λ). All exit 0. The first three GPU submissions were rejected at admission ("GPU is lost" on k8s-chase-ci-01 and k8s-haosu-21). The accepted runs excluded those two nodes; part_job.sh itself was not changed. Results: `results/physics/h64_verify_{hls,jetclass,jetclass_mfix,top,qg,top_smalllam,qg_smalllam}.json`.

## Conclusion

The main measurement reproduces on new jets. When a jet is boosted along its own axis, its mass stays fixed while it gets narrower and its momentum rises. The JetClass- and Top-trained ParT, ParticleNet and LorentzNet change their output by about 0.1–0.45 of what an equal narrowing at fixed momentum does. A tagger that reads only the jet's angular size would show a ratio of 1, so these networks do not behave like one. The data do not show that they read the invariant mass, however. A small classifier fed only the jet's angular width and its pT, with no mass input, gives the same ratio on JetClass (about 0.3). The difference between datasets (hls4ml high, JetClass low) appears in such reference classifiers too. Two statements in H64 do not hold up:
- The ratio being "about 0 when pT is raised" comes from cancellation between classes in the summary statistic.
- The quark–gluon ratios (0.6–0.85) are not only an extrapolation outside the training pT range: they hold for jets that stay inside it.

## What was checked and how

**Edits** (same code as H64; λ = factor on the jet pT):
- **B**: exact Lorentz boost of every constituent along the jet three-momentum, so that |P_jet| → λ|P_jet|.
- **A**: every (Δη, Δφ) about the stored axis divided by λ, at fixed constituent pT.
- **P**: every four-vector × λ.
- **R**: rotation about the axis with the same pT-weighted displacement as B.
- **C (new)**: the composition P∘A. Angles are divided by λ, then every four-vector is multiplied by λ. To first order C changes (log m, log pT, log angle) by (0, +log λ, −log λ), the same as B, but it is not a Lorentz transformation.
- **P_mfix, C_mfix (new)**: the same as P and C, but only the constituent pT (and therefore E and |p|) is scaled. Each constituent's mass is kept, so a pion stays at 0.1396 GeV.

**Metrics**:
- Δ_X,k: the mean change of the one-vs-rest log-odds l_k = log p_k − log(1 − p_k) on true-class-k jets.
- R_B^w (H64's summary): Σ_k s_k Δ_B,k / Σ_k |Δ_A,k|, with s_k = sign(Δ_A,k).
- |R|_B (new, robust summary): Σ_k |Δ_B,k| / Σ_k |Δ_A,k|. This is the size of the boost response relative to the size of the narrowing response, summed over classes. It has no sign weights.
- "In-range": jets whose unedited and B-edited jet pT both lie inside the dataset's window (JetClass 500–1000, Top 550–650, QG 500–550 GeV). B, A and C are evaluated on the same subset of jets.

**New references** (sklearn MLP 32-32, trained on disjoint real jets of the same split):
- MLP(w, pT) on [log w, log pT], where w = √(M2/M0) is the pT-weighted RMS angle of the constituents about the stored axis (M0 = ΣpT, M2 = ΣpT·ΔR²). It sees the angular width and the jet pT but not the invariant mass.
- MLP(m) on [log m]: it reads the invariant mass only, so Δ_B = 0 by construction (positive control).

**Hypotheses**:
- H0_C: Δ_B = Δ_C (class-signed, paired, normal approximation). H1: the exact boost differs from the composition.
- H0_inv: Δ_B = 0.
- Holm correction within each part over model × λ.

**Samples** (new seeds, random rows; never first-k):
- hls4ml: draw seed 7461, 2,000 per class (10,000 jets), plus 3,000 per class reference.
- JetClass: jc_sample(3000, seed 7462), 1,500 per class evaluated (15,000), plus 1,500 per class reference (rng 7472).
- Top test: 2,000 + 2,000 (rng 7463), plus 10,000 + 10,000 reference.
- QG test: 2,000 + 2,000 (rng 7464), plus 10,000 + 10,000 reference.
- λ ∈ {0.8, 0.9, 1.1, 1.25}. Small-λ runs: Top {0.95, 0.97, 1.03, 1.05}; QG {0.97, 0.99, 1.01, 1.03}.

## Unedited accuracy on the new samples (H12/H64 value in brackets)

- JEDI-linear n64: 80.88 % (80.68).
- JetClass: ParT_full 85.64 (85.75), ParT_kin 74.11 (73.88), PN_kin 69.69 (70.04).
- Top: ParT 93.73 (93.98), PN 93.73 (94.02), LorentzNet 93.93 (94.15).
- QG: ParT 83.13 (82.63), PN 82.47 (82.69). The H64 sample's +0.8–1.5 pt excess on QG does not recur. This is consistent with a sample fluctuation.

## Code review

- **Boost**:
  - h12.boost was compared with an independent explicit 4×4 Lorentz matrix. That matrix takes its rapidity from atanh(|P|/E), where h12.boost uses asinh(|P|/m).
  - On jets with m > 1 GeV the two agree to ≤ 3.3e-12 relative on every hls4ml, JetClass, Top and QG sample at every λ.
- **Mass and pT in the model's own input arrays**:
  - The jet mass and pT were recomputed from the arrays the network is actually fed after `prep`: rebuilt px, py, pz, E for JetClass; P4 for Top; (pT, Δη, Δφ) for hls4ml; (pT, η, φ) for QG.
  - Under B the mass changes by at most 5e-8 (hls4ml), 2.5e-4 (Top), 5e-12 (QG) and 2.4e-3 (JetClass, float32 rebuild; the unedited rebuild already differs from the table by 1.3e-3).
  - The jet pT ratio equals λ to 1e-6.
- **Sampling**: JetClass rows are drawn at random per class, never first-k. The observables are computed on the same jets as the network input.
- **Padding**: padded slots (pT = 0) are masked in four(), rel() and the boost.
- **Working point**: 50 % per-class threshold on unedited own-class log-odds, as described.
- **Issue 1: the signed summary R_B^w is unstable.**
  - s_k = sign(Δ_A,k) is noise when a class has Δ_A,k ≈ 0 but a large Δ_B,k.
  - JEDI λ 1.25: the hls4ml gluon class has Δ_A = +0.01 (H64) vs −0.004 (new), while Δ_B = +1.27 vs +1.23. That one sign flip moves R_B^w from 0.38 to 0.18. The per-class values agree to within 0.07 log-odds.
  - JetClass ParT_full gives R_B^w 0.24 [0.23, 0.25] in H64 and 0.18 [0.17, 0.19] here at λ 0.9. The magnitude summary |R|_B is 0.27 and 0.26.
  - The bootstrap holds s_k fixed, so its interval does not include this instability.
  - The near-zero and negative R_B^w at λ > 1 for JetClass ParT_kin and PN_kin and for Top ParT are partly cancellation: per-class Δ_B has the opposite sign to Δ_A in several classes.
- **Issue 2: P scales the constituent masses as well.**
  - P multiplies every four-vector, so each constituent's mass is multiplied by λ. JetClass constituents are massive (pions etc.).
  - JetClass ParT_kin is very sensitive to this when λ < 1. At λ 0.9, Δ_P on QCD is −0.45 with masses scaled and +0.17 with masses kept; on Wqq it is −0.40 vs +0.25.
  - ParT_full and PN_kin change by ≤ 0.04.
  - With masses scaled, C differs from B for ParT_kin by −3.1 (λ 0.9) and −19.4 (λ 0.8) class-signed log-odds. With masses kept, the difference is −0.10 and −0.14.
  - H64's JetClass ParT_kin R_m^w and its B − (P+A) values at λ < 1 therefore contain a response to unphysical constituent masses. R_B (B over A) is unaffected, because neither B nor A changes constituent masses.

## Results

### 1. Reproduction of R_B^w (H64 signed summary; new sample, 95 % bootstrap)

| model | λ 0.8 | 0.9 | 1.1 | 1.25 | H64 |
|---|---|---|---|---|---|
| JEDI-linear | 1.04 | 0.83 [0.81, 0.85] | 0.35 [0.34, 0.36] | 0.18 [0.17, 0.19] | 1.05 / 0.86 / 0.35 / 0.38 |
| JetClass ParT_full | 0.25 | 0.18 [0.17, 0.19] | 0.10 [0.09, 0.11] | 0.07 | 0.24 / 0.24 / 0.16 / 0.07 |
| JetClass ParT_kin | 0.12 | 0.23 [0.21, 0.25] | −0.02 [−0.05, 0.01] | 0.01 | 0.09 / 0.25 / −0.05 / −0.01 |
| JetClass PN_kin | 0.13 | 0.25 | −0.04 | 0.01 | 0.14 / 0.25 / −0.03 / 0.01 |
| Top ParT | 0.20 | 0.20 [0.19, 0.22] | −0.08 [−0.10, −0.05] | −0.04 | 0.21 / 0.20 / −0.05 / −0.01 |
| Top PN | 0.44 | 0.43 | 0.18 | 0.15 | 0.45 / 0.45 / 0.20 / 0.17 |
| Top LorentzNet | 0.24 | 0.30 | 0.05 | 0.09 | 0.25 / 0.29 / 0.06 / 0.09 |
| QG ParT | 0.72 | 0.75 [0.74, 0.77] | 0.84 | 0.86 | 0.70 / 0.74 / 0.83 / 0.86 |
| QG PN | 0.59 | 0.61 | 0.64 | 0.65 | 0.58 / 0.61 / 0.64 / 0.64 |

All values are reproduced within about 0.06, except JEDI at λ 1.25 and ParT_full at λ 0.9 / 1.1. Those are the sign-weighting instability described under Issue 1.

### 2. Robust summary |R|_B = Σ_k|Δ_B,k| / Σ_k|Δ_A,k| (λ 0.8 / 0.9 / 1.1 / 1.25)

| model | H64 sample | new sample |
|---|---|---|
| JEDI-linear | 1.05 / 0.96 / 0.48 / 0.56 | 1.04 / 0.91 / 0.48 / 0.55 |
| hls4ml Deep Sets / transformer (H64 only) | 1.08 / 0.74 / 0.49 / 0.70; 1.19 / 0.97 / 0.50 / 0.55 | – |
| JetClass ParT_full | 0.26 / 0.27 / 0.18 / 0.12 | 0.25 / 0.26 / 0.18 / 0.12 |
| JetClass ParT_kin | 0.21 / 0.30 / 0.24 / 0.15 | 0.20 / 0.27 / 0.23 / 0.15 |
| JetClass PN_kin | 0.20 / 0.28 / 0.27 / 0.16 | 0.20 / 0.28 / 0.28 / 0.16 |
| Top ParT | 0.21 / 0.20 / 0.17 / 0.05 | 0.20 / 0.20 / 0.19 / 0.07 |
| Top PN | 0.45 / 0.45 / 0.38 / 0.17 | 0.44 / 0.43 / 0.34 / 0.15 |
| Top LorentzNet | 0.25 / 0.29 / 0.14 / 0.13 | 0.24 / 0.30 / 0.11 / 0.13 |
| ref MLP(m, pT): hls4ml / JetClass / Top | 1.67 / 1.42 / 0.91 / 1.08; 0.33 / 0.38 / 0.23 / 0.15; 0.75 / 0.39 / 0.38 / 0.24 | 1.49 / 1.15 / 0.81 / 0.94; 0.31 / 0.36 / 0.22 / 0.15; 1.14 / 0.62 / 0.25 / 0.23 |
| ref MLP(w, pT) (new): hls4ml / JetClass / Top | – | 1.74 / 1.44 / 0.93 / 1.12; 0.31 / 0.35 / 0.20 / 0.14; 1.55 / 0.89 / 0.52 / 0.31 |
| ref MLP(m/pT) | 1.00 | 1.00 |
| ref MLP(m) (new) | – | 0 (|R_B^w| < 1e-14) |

The network values reproduce to within 0.05 on every model and λ.

The magnitude ratio shows no "≈ 0 at λ > 1" for the JetClass and Top networks. At λ 1.1 it is 0.11–0.38.

On JetClass, the angle-width-plus-pT reference MLP(w, pT) and the mass-plus-pT reference MLP(m, pT) both give 0.14–0.36. This is the same range as ParT and PN (0.12–0.30).

### 3. Training-range control: R_B^w on in-range jets only

- **JetClass λ 0.9** (64 % of jets in range): ParT_full 0.22 [0.21, 0.23], ParT_kin 0.22 [0.19, 0.24], PN_kin 0.23 [0.21, 0.24]. All jets: 0.18, 0.23, 0.25.
- **Top λ 0.9** (34 % in range): ParT 0.16 [0.13, 0.19] vs 0.20 for all jets. PN 0.34 vs 0.43. LorentzNet 0.27 vs 0.30.
- **Top, small λ** (67–84 % in range): R_B^w in-range for λ 0.95 / 0.97 / 1.03 / 1.05:
  - ParT 0.26 / 0.33 / −0.13 / −0.09 (|R|_B 0.27–0.36);
  - PN 0.46 / 0.54 / 0.96 / 0.21 (|R|_B 0.51–0.92);
  - LorentzNet 0.43 / 0.56 / −0.15 / −0.07 (|R|_B 0.04–0.56).
- **QG, small λ** (63–91 % in range): R_B^w in-range for λ 0.97 / 0.99 / 1.01 / 1.03:
  - ParT 0.80 [0.77, 0.83] / 0.72 / 0.83 / 0.83;
  - PN 0.61 / 0.63 / 0.63 / 0.63.
  - These equal the out-of-range values at λ 0.8–1.25 (ParT 0.72–0.86, PN 0.59–0.65).
- **Top LorentzNet rotation control**: at small λ, the rotation control's class-signed response is 0.12–0.14 at λ < 1. That is about half the B response (0.22–0.27). At λ 1.05 it is −0.04, against a B response of −0.01. LorentzNet's B response at small λ is therefore not separable from its response to an equal-size rotation. This is consistent with H64's note about beam particles in its inputs.

### 4. Boost vs composition (C = P∘A at fixed constituent masses where relevant), class-signed Δ_B − Δ_C

- **Massless-constituent datasets** (hls4ml, Top, QG): |Δ_B − Δ_C| ≤ 0.07 log-odds against class-signed Δ_B of 0.03–14.6.
  - Holm: H0_C is rejected in 0/12 QG, 0/12 Top-small-λ and 0/8 QG-small-λ tests. It is rejected in 9/12 Top tests (λ ≥ 0.8; differences 0.015–0.057) and 3/4 hls4ml tests (differences 0.007–0.072).
- **JetClass with masses kept**: |Δ_B − Δ_C| ≤ 0.14, except ParT_full at λ 0.8 (−0.40).
- **Reading**: the boost response equals, to within a few percent, the response to "pT up by λ and angles down by 1/λ at fixed m". No part of it is specific to the exact Lorentz structure of the edit.

### 5. Tests (Holm within each part)

- **H0_inv (Δ_B = 0)** is rejected after Holm in 51 of 56 model × λ cells (hls4ml 4/4, JetClass 9/12, Top 12/12, QG 8/8, Top small-λ 10/12, QG small-λ 8/8). The class-signed 95 % interval of Δ_B includes 0 in four cells:
  - JetClass ParT_kin λ 1.1 [−0.16, 0.02] and λ 1.25 [−0.03, 0.20];
  - JetClass PN_kin λ 1.25 [−0.02, 0.13];
  - Top LorentzNet λ 1.05 [−0.04, 0.01].
- **Excluded by these measurements**:
  - The angular-size / m/pT-only prediction (|R|_B = 1) for JetClass ParT and PN (0.12–0.30) and for Top ParT (0.05–0.36).
  - The invariant-mass-only prediction (Δ_B = 0) for all networks except those cells.
- **Not excluded**: a reader of (m, pT) and a reader of (angular width, pT). Both reproduce the JetClass values, and both meet or exceed the hls4ml values.

## Figures

- `h64_wjet_boost.png` was checked. B at λ 1.25 keeps m = 79.4 GeV with pT 1173 → 1466 GeV. A keeps pT and gives m = 63.5 GeV. Both bring the prong separation from 0.163 to 0.130. The labels and numbers are consistent with the edits.
- `h64_RB.png` plots the signed R_B^w, so its λ 1.1 points near 0 for JetClass and Top carry the cancellation described above.

## Corrected reading

1. Under an exact boost along the jet axis at fixed invariant mass, JetClass ParT and ParticleNet and Top ParT change their per-class log-odds by 0.12–0.36 of the change under an equal angle-only narrowing at fixed pT. This is the class-summed magnitude ratio, for λ 0.8–1.25 and Top small λ. It holds in both directions of λ and on in-range jets.
2. Top PN gives 0.15–0.92. Top LorentzNet gives 0.04–0.56, and its boost response is comparable to its rotation-control response.
3. hls4ml JEDI-linear gives 0.48–1.05, and QG ParT/PN give 0.6–0.86. The QG values also hold on jets inside the 500–550 GeV training range.
4. The pure angular-size (m/pT) reader, with ratio 1, is excluded for JetClass/Top ParT and PN. The measurement does not separate an invariant-mass reader from a reader of angular width and pT: MLP(w, pT), which has no mass input, matches JetClass ParT within about 0.1.
5. The dataset contrast is also present in the references. MLP(m, pT) and MLP(w, pT) give 0.8–1.7 on hls4ml and 0.14–0.36 on JetClass.
6. H64's statements that need correcting:
   - The signed R_B^w is unstable when a class has Δ_A ≈ 0.
   - "≈ 0 for λ > 1" is a cancellation artifact.
   - JEDI λ 1.25 = 0.38 was not reproduced (0.18).
   - JetClass ParT_kin's P responses at λ < 1 include a response to rescaled constituent masses.
