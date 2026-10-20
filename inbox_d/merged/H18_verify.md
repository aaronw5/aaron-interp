# H18 verify — soft-particle count response: edit difference or model difference

Script `interp/h18_verify.py` (imports the H18 edit code). Job `d-h18-verify` (GPU, 21 min, exit 0). Results in `results/physics/h18_verify_{qgpy,jc,top}.json`, `h18_verify.json` (336 tests with Holm-adjusted p) and `h18_verify_summary.json`.

## Conclusion

The fresh-seed rerun reproduces the H18 numbers. A new edit splits soft particles into two collinear halves, which adds particles while keeping every IRC-safe quantity fixed (IRC-safe means unchanged by soft or collinear emissions). Under this split, quark–gluon networks change by −0.006 to −0.020 per added particle and the JetClass W-vs-QCD networks by −0.055 to −0.10. The count-only likelihood ratio, which uses particle number alone, gives −0.068 and −0.044 per particle on the two datasets. So the result that QG networks barely use particle count, while JetClass networks do, holds on an edit that is identical across datasets and exactly IRC-safe.

H18's one-particle edit e1 does not support that result. On QG it adds a particle of about 0.1 GeV, on JetClass about 1 GeV. With the added particle fixed at 1 GeV, the QG networks shift by as much as the JetClass networks, and about half of the QG shift is also seen in the IRC-safe EFN.

Still uncertain: in JetClass the split halves (≈0.8 GeV) may fall below the softest particles the networks saw in training. The top taggers also move against the count-only reference under the split, and the permuted-weight null PFN also responds (+0.012 per particle).

## Code review of `interp/h18_softcount_cross.py`

- **Axis and η convention.** JetClass goes through `jc_table` / `part_core.build` with η = jet_eta + sign(jet_eta)·Δη, and `jet_pt` / `jet_energy` are recomputed from the particles. Edits are placed about the stored axis, and the inputs are built about the stored axis (QG recomputes it, as stated). No problem found.
- **Padding.** `nreal`, `eaxis`, `jet4`, `mass_all` and `finalize` all mask pT > 0. N_seen is capped at 128 (64 for hls). No problem found.
- **Rows.** JetClass, QG, Top and hls samples use random indices; there is no first-k selection. The model scores, reference scores and N_seen come from the same jets file (`h18_<ds>_jets.npz`). No problem found.
- **Issue 1: e1 is not the same edit across datasets.** `e_onman` gives the added particle the jet's softest pT. The mean added pT from the H18 logs is 0.13 GeV (QG), 0.33 GeV (Top) and 0.99 GeV (JetClass). The fresh-sample medians are 0.09, 0.32 and 0.94 GeV. The cross-dataset e1 contrast therefore varies the added pT as well as the dataset. This is the basis of "H1 (edit difference) not supported".
- **Issue 2: c4 is not at fixed IRC-safe content.** c4_20 adds the full disc ΣpT at r ≤ R/20: +40 GeV for JetClass and Top, +10 GeV for QG. The figure's bottom-row label says "IRC-safe content ~fixed", supported only by the EFP-GBDT's small response. c5 matches the mass of pu, not of c4, so H18 has no control with the same four-momentum as c4.
- **Issue 3: ratios to LR(N) under e1.** For JetClass W vs QCD, LR(N) under e1 is +0.023 while the networks are negative. "1–4 × LR(N)" therefore holds only for c4, not for e1.
- **Minor.** The QG panel's bottom-row LR(N) bars run to the axis limit. The hls e1 / c5 ΔN_seen is 0.77 (slot saturation), as reported.

## Procedure

**Sample (seed 9311, random rows):**
- QG Pythia test: 4,000 q + 4,000 g (4.9 % overlap with H18's rows).
- JetClass first test file: 2,500 each of QCD, Wqq and Tbqq, with H18's seed-4181 rows excluded, plus 300 of each other class for the 10-class accuracy.
- Top Landscape test: 3,000 + 3,000 (2.6 % overlap).

**Unedited performance on this sample:**

| Dataset | Model performance |
|---|---|
| QG (AUC) | ParT 0.917, ParT_kin 0.906, PN 0.914, PN_kin 0.906, LorentzNet 0.921, PFN 0.886, EFN 0.892 / 0.876; permuted ParT_kin 0.540 |
| JetClass (10-class balanced accuracy) | ParT_full 0.866, ParT_kin 0.736, PN_full 0.850, PN_kin 0.700, redrawn ParT_full 0.100 |
| JetClass W-vs-QCD (AUC) | ParT_full 0.980, ParT_kin 0.972, PN_full 0.976, PN_kin 0.966 |
| Top (AUC) | ParT 0.985, PN 0.985, LorentzNet 0.986, PFN 0.979, EFN 0.974; permuted PFN 0.704 |

**Edits.** The H18 edits were rerun: e1, k1, k20 and c4_20. Three new controls were added:
- **e1p⟨x⟩:** e1 with the added pT fixed at x = 0.13, 0.5, 1 or 2 GeV. It uses the same random host particle and direction as e1, so the comparison is paired.
- **sp⟨k⟩ (collinear split):** k = 1, 5 or 20 distinct random particles of the softer half are each replaced by two pT/2 halves at ±δ/2 about the parent, with δ = 0.01 (sp5d002: δ = 0.002).
  - Count +k (sp20 gives +16.6 to +18.3, limited by the size of the soft half).
  - ΣpT and the pT-weighted centroid are unchanged; the mass changes by ≤ 0.006 GeV.
  - The EFNs respond by |Δs| ≤ 0.003 at sp20.
- **c4m_20:** the c4_20 particles merged into one particle carrying their summed four-momentum. It adds the same ΔΣpT and Δm as c4_20 with count +1. The contrast c4_20 − c4m_20 is the count effect at fixed added four-momentum.

**Reference.** LR(N) is rebuilt from H18's training dumps.

**Metrics.**
- Mean Δs, the signal log-odds change, with a 95 % bootstrap interval over jets (B = 1,000).
- Split slope: Δs against ΔN_seen over sp1/sp5/sp20, with a bootstrap interval.
- Sign-flip p in its normal limit, with Holm correction over 336 tests (networks × edits and paired contrasts; nulls excluded). 291 tests are below 0.05.

## Numbers

**Reproduction.** H18 → verify:

| Dataset / edit | Model | H18 | Verify |
|---|---|---|---|
| QG e1 | ParT | +0.015 | +0.016 |
| QG e1 | PFN | −0.056 | −0.056 |
| QG k20 | ParT | −1.411 | −1.424 |
| JetClass W e1 | ParT_full | −0.113 | −0.122 |
| JetClass W k20 | ParT_full | −2.431 | −2.144 (lower ΔΣpT) |
| JetClass W c4_20 | PN_kin | −4.297 | −4.178 |
| Top k20 | ParT | −0.704 | −0.732 |

**Added-pT scan (e1p), mean Δs for one added particle:**

| Added pT | QG networks | QG EFN | JetClass W networks | Top networks |
|---|---|---|---|---|
| softest pT (e1) | +0.004 to +0.025 (PFN −0.056) | −0.005 | −0.054 to −0.122 | −0.025 to +0.021 |
| 0.13 GeV | −0.008 to +0.013 (PFN −0.056) | −0.005 | −0.07 to −0.16 (redrawn null −0.135) | −0.108 to −0.023 |
| 1 GeV | −0.063 to −0.096 [CIs ±0.003] | −0.039 / −0.035 | −0.050 to −0.125 | −0.028 to +0.032 |
| 2 GeV | −0.078 to −0.120 | −0.071 / −0.065 | −0.045 to −0.113 | −0.025 to +0.050 |

LR(N) for one particle: QG −0.094, JetClass W +0.023, Top +0.076.

- QG at 0.13 GeV vs 1 GeV: in QG the networks' response grows with the added pT (0.13 → 1 GeV: ParT +0.006 → −0.096).
- JetClass at 0.13 GeV: here the redrawn-weight null also responds strongly (−0.135), which is consistent with a particle below the JetClass soft-particle range.

**Collinear split, slope per added particle [95 % CI]:**

| Task | Model | Slope |
|---|---|---|
| QG | ParT | −0.0165 [−0.0172, −0.0159] |
| QG | ParT_kin | −0.0082 [−0.0086, −0.0079] |
| QG | PN | −0.0199 [−0.0206, −0.0192] |
| QG | PN_kin | −0.0060 [−0.0064, −0.0055] |
| QG | LorentzNet | −0.0145 [−0.0152, −0.0138] |
| QG | PFN | −0.0473 [−0.0479, −0.0467] |
| QG | EFN | −0.0002 |
| QG | LR(N) | −0.0682 [−0.0689, −0.0675] |
| QG | permuted ParT_kin | +0.0037 |
| JetClass W | ParT_full | −0.0970 [−0.1002, −0.0941] |
| JetClass W | ParT_kin | −0.0547 [−0.0564, −0.0531] |
| JetClass W | PN_full | −0.0939 [−0.0969, −0.0910] |
| JetClass W | PN_kin | −0.1020 [−0.1042, −0.1001] |
| JetClass W | LR(N) | −0.0444 [−0.0464, −0.0426] |
| JetClass W | redrawn null | −0.0160 |
| JetClass top | ParT_full | −0.029 |
| JetClass top | PN_full | −0.009 |
| JetClass top | ParT_kin | +0.001 [−0.001, +0.003] |
| JetClass top | PN_kin | +0.003 |
| JetClass top | LR(N) | +0.035 |
| Top | ParT | −0.024 |
| Top | PN | −0.019 |
| Top | PFN | −0.028 |
| Top | LorentzNet | −0.018 |
| Top | EFN | 0.000 |
| Top | LR(N) | +0.023 |
| Top | permuted PFN | +0.012 |

- Slope relative to LR(N): QG networks 0.09–0.29 × LR(N) (PFN 0.69); JetClass W networks 1.2–2.3 × LR(N).
- QG ParT_kin vs JetClass ParT_kin: the slope difference is −0.0465 with s.e. 0.0009 (independent bootstraps), p < 10⁻¹⁰⁰.
- sp5 vs sp5d002 (δ = 0.01 vs 0.002), Holm-significant differences:
  - JetClass ParT_kin: W −0.277 vs −0.227; top also significant.
  - JetClass PN_kin: W −0.403 vs −0.413.
  - QG ParT: −0.125 vs −0.134.
  - The EFNs, at the 10⁻³ level.
  - Every other network: no significant difference.
  - So most of the split response does not depend on δ.

**c4_20 − c4m_20 per extra particle (count at fixed added four-momentum):**

| Task | Networks | LR(N) | c4m_20 alone (total Δs) |
|---|---|---|---|
| JetClass W | ParT_full −0.132, PN_full −0.111, PN_kin −0.185, ParT_kin −0.023 | −0.047 | −0.06 to −0.13 |
| QG | −0.022 to −0.005 (PN +0.026) | −0.078 | – |
| Top | ParT −0.023, PFN −0.041, LorentzNet −0.094, PN +0.006 | – | LorentzNet −3.42, others −0.11 to +0.08 |

- For JetClass W, the c4 response is mostly count: the added four-momentum alone (c4m_20) moves the networks by only −0.06 to −0.13 in total.
- For Top LorentzNet, about 40 % of the c4 response comes from the added core four-momentum.

## Hypotheses

- **H0, model/dataset difference** (count at fixed IRC-safe content is used by the JetClass W networks but not by the QG networks): supported by the exact collinear split and by c4 − c4m. Both are identical across datasets.
- **H1, edit difference:** it is excluded for the split and the c4 − c4m contrasts. It is **not excluded** for H18's e1 contrast. That contrast compares a 0.1 GeV addition (QG) with a 1 GeV addition (JetClass). At 1 GeV both datasets' networks give −0.05 to −0.12, and in QG 40–60 % of that is reproduced by the IRC-safe EFN. H18's statement "H1 not supported: JetClass W networks respond to e1" rests on this unmatched edit.

## Corrected reading

H18's energy-flow part is reproduced. The k-slope of the QG networks is −0.071 to −0.076 per particle, against −0.054 / −0.062 for the EFN.

The count part, with IRC-safe content fixed exactly (collinear split) or with the added four-momentum fixed (c4 − c4m), is reproduced as a model/dataset difference:
- **QG networks:** 0.1–0.3 × LR(N) per particle (PFN 0.7).
- **JetClass W-vs-QCD networks:** 1.2–2.3 × LR(N) under splits, and −0.02 to −0.19 per particle under c4 − c4m.

The e1 contrast between datasets is mainly a difference in the added particle's pT (median 0.09 GeV QG vs 0.94 GeV JetClass) and should not be cited for H0 vs H1.

Top LorentzNet's c4 response is partly the added core four-momentum (c4m alone −3.4). Its split slope is −0.018 per particle, so "Top LorentzNet counts" applies only to the c4 geometry.

Under exact splits, the Top Landscape networks and JetClass ParT_full move toward QCD while LR(N) moves toward top. The permuted PFN null moves toward top (+0.012 per particle).

## Corrected Conclusion

When soft particles add energy flow, quark–gluon networks move toward gluon about as much as an IRC-safe network that cannot count particles. When particles are added by splitting soft particles into collinear halves, which changes the count and nothing IRC-safe, quark–gluon networks move by 0.1–0.3 of the count-only likelihood ratio. JetClass W-vs-QCD networks move by 1.2–2.3 times it. So whether the network uses the particle count depends on the model and training data, not on the edit. H18's single-soft-particle comparison does not show this, because it added about 0.1 GeV in quark–gluon jets and about 1 GeV in JetClass jets. Still open: whether the JetClass split halves (≈0.8 GeV) lie outside the soft-particle range JetClass networks were trained on, and why the top taggers move against the count-only reference.

## Confidence

0.55 that H18's reading is correct as stated. The main D7/D10 scoping holds on cleaner controls. The e1-based exclusion of H1, the "IRC-safe content ~fixed" description of c4 and the LorentzNet count attribution need the corrections above.
