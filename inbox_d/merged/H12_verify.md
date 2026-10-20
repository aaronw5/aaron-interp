# H12 verify — mass-preserving boost vs pT-only and angle-only scalings

## Conclusion
A Lorentz boost along the jet axis that changes the jet pT by 10 % while keeping the invariant mass fixed changes about a third of the calls of the hls4ml-trained JEDI-linear tagger. It changes 1–4 % of the calls of the JetClass and Top Landscape taggers, which is less than either pT-only or angle-only scaling changes. This reproduces on a fresh sample. For JEDI-linear, most of the calls that change (62 % at pT × 0.9) are W or Z jets that become top. The same flow appears when only pT is lowered, so this part of the response follows the known ΣpT shortcut (D1) rather than a mass reading. The boost also leaves the axis-moment mass m_A = √(M0·M2) fixed, so H12 cannot tell m from m_A. Why hls4ml and Top models differ is not established: the Top taggers were also trained in a narrow pT window and see the same particle information.

## Code review (interp/h12_boost_panel.py, interp/h12_jnorm_train.py)
- **Boost B.** Checked against an independent implementation (`interp/h12_verify.py: boost_indep`). It uses relativistic velocity addition β = (v1 − v0)/(1 − v0·v1), with v = |P|/E, and the standard boost formula, not h12's arsinh rapidity.
  - hls4ml and JetClass, λ 0.9 and 1.1: particle pT agrees to 3e-12 (relative), η and φ to 2e-12, the axis to 7e-16. The invariant m is fixed to 6e-12. Jet pT ratio = λ to 1e-11.
  - Network calls under the two implementations agree on 100 % of jets (JEDI n64 and n32, ParT_full, PN_full) and on 99.99 % for Top ParT_kin.
  - One Top jet in this sample has m ≈ 0 (one massless constituent). On it, the velocity-addition formula gives 0/0. h12's code clamps m_J ≥ 1e-6 and handles this jet without error.
- **Padding.** Padded slots stay at pt = 0 under every edit (np.where(real, …)), and jet_obs / rel / displacement are masked. JetClass `build` recomputes jet_pt and jet_energy from the edited particles, so ParT's logptrel / logerel stay consistent. The round-trip difference is 6e-5.
- **Sampling.** JetClass uses random rows per class file (seed 301), not first-k. Top and QG use random rows per label, and LorentzNet uses [::2] of a label-mixed sorted index. Observables and network inputs come from the same table U per edit.
- **Controls.** R matches B's pT-weighted mean displacement per jet: 2·r̄·sin(θ/2) = D, which is exact.
- **Report statement "m_A changes by at most 0.12 %" under B.** The json gives |Δm_A/m_A| max = 0.71 % (hls4ml), 13 % (JetClass), 1.1 % (Top) and 0.27 % (QG). The mean is ≤ 1e-4. In this rerun the hls4ml 99th percentile is 0.12 % and the JetClass 99th percentile is 1.9 %. The 0.12 % figure is a hls4ml 99th percentile, not a maximum.
- **Retrains (h12_jnorm_train.py).** No defect found.
  - ptnorm divides pT by the ΣpT of the 64 visible particles.
  - ptflat uses per-class flat-ΣpT weights within the hls4ml spectrum. The weights do not add jets at other pT, so this retrain does not test whether training over a wider pT range removes the boost response.

## Rerun on a fresh sample (seed 917)
- Jobs:
  - d-h12-verify (CPU, jax): hls4ml, 3,000 jets per class, N = 15,000.
  - d-h12-verify-g (GPU): JetClass, 1,500 random rows per class, N = 15,000; Top Landscape test, 5,000 + 5,000 random rows.
- Outputs: `results/physics/h12_verify_hls.json` and `h12_verify_gpu.json`.

Unedited accuracy on these samples:

| model | accuracy | 95 % CI | H12 value |
|---|---|---|---|
| JEDI-linear n64 | 81.13 % | [80.5, 81.7] | 81.00 |
| JEDI-linear n32 | 77.64 % | | 77.92 |
| ParT_full | 86.24 % | | 85.75 |
| PN_full | 84.29 % | | 84.09 |
| Top ParT_kin | 94.34 % | | 93.98 |

% calls changed [95 % bootstrap]; H12 values in parentheses:

| model | B 0.9 | P 0.9 | A 0.9 | R 0.9 | T 0.9 | B 1.1 | P 1.1 | A 1.1 |
|---|---|---|---|---|---|---|---|---|
| JEDI n64 | 32.9 [32.2, 33.7] (32.5) | 34.3 (33.8) | 28.1 (28.3) | 2.4 [2.2, 2.6] (2.6) | 2.4 | 14.4 (14.4) | 30.7 (31.3) | 23.2 (24.0) |
| JEDI n32 | 37.8 (37.7) | 37.7 (37.5) | 31.7 (31.6) | 3.0 (3.1) | 3.3 | 17.4 (18.1) | 29.3 (29.6) | 22.7 (22.8) |
| ParT_full | 4.36 [4.03, 4.69] (4.5) | 6.89 (6.9) | 14.5 (14.4) | 0.40 (0.5) | 0.91 | 2.87 (2.8) | 8.57 (8.7) | 9.70 (9.8) |
| PN_full | 4.33 (4.3) | 8.56 (8.1) | 14.2 (14.5) | 0.94 (1.1) | 1.29 | 3.19 (3.2) | 9.11 (9.3) | 10.8 (10.5) |
| Top ParT_kin | 0.94 [0.76, 1.14] (1.1) | 3.34 (3.6) | 4.05 (4.1) | 0.38 (0.4) | 0.19 | 1.08 | 3.67 | 3.51 |

Exact McNemar tests, Holm-corrected across the 40 tests of this rerun:
- H0: B and the control change the same fraction of calls. H1: they differ.
- B vs R: p_Holm ≤ 2e-7 for every model and λ.
  - JEDI n64 at λ 0.9: +30.5 pp.
  - ParT_full at λ 0.9: +4.0 pp.
  - Top ParT at λ 0.9: +0.56 pp.
- B vs P, JEDI n64: −1.4 pp at λ 0.9 (p_Holm 2e-4) and −16.3 pp at λ 1.1.
- B vs P, JEDI n32 at λ 0.9: +0.1 pp (p 0.78).
- All numbers reproduce H12 within the bootstrap intervals.

## Checks of the reading
1. **"B is 3–4 times smaller than P or A" (JetClass).** The ratio holds for the kin models: P/B = 3.4 (ParT_kin) and 3.4 (PN_kin). For the full models, P/B is 1.5 (ParT_full: 6.9 vs 4.5) and 1.9 (PN_full: 8.1 vs 4.3); A/B is 3.2–3.4. The correct range is 1.5–4×. The B/R ratio is 2.5–9×, not 2–8×.
2. **What the JEDI-linear B response consists of.** Flow matrix from h12_hls_jedi.json, n64, λ 0.9:
   - B0.9: 6,490 calls change. Of these, 4,036 (62 %) are W→t (2,406) or Z→t (1,630). W→Z is 263 and Z→W is 14.
   - P0.9 (pT lowered, angles fixed, m → 0.9 m):
     - W/Z→t: 3,570.
     - Z→W: 681; W→Z: 1. This is the direction a mass reading predicts at the W/Z boundary.
   - A0.9: W→Z 3,372.
   - So the "net heavier-class flow" under P is driven by W/Z → t, the top-vs-rest ΣpT dependence of D1. Inside W vs Z, the pT-only edit moves calls toward the lighter class.
   - Mean Δlog-odds, rerun, n64, λ 0.9:

     | task | B | P | A | R |
     |---|---|---|---|---|
     | t vs QCD | +0.94 [0.91, 0.96] | +0.60 | +0.13 | 0.00 |
     | Z vs W | +1.45 | −1.36 | +2.81 | |

     On Z vs W, B = P + A to 0.01. On t vs QCD, B follows the pT channel.
   - The reading "lowering pT at fixed angles moves calls to the heavier class, the opposite of a mass reading" holds for the W/Z→t flow only. At the W/Z boundary the pT-only edit acts like a mass edit.
3. **B also fixes m_A.** m_A changes by ≤ 0.12 % in 99 % of hls4ml jets, and the m_A-window efficiencies are unchanged in the json. A B response therefore excludes a pure m_A reading as much as a pure invariant-m reading. The thesis note "Supports D4m/A4: models read axis-relative angles, not the invariant m" does not follow from H12: B cannot tell m from m_A. H12 shows that the JEDI-linear call depends on something beyond m_A, through pT/ΣpT and the angular scale separately.
4. **ptnorm retrain and EFN: B = A by construction.**
   - Their inputs are the momentum fractions z_i and the angles. Under B, z_i is unchanged to O(θ²), so B acts on these inputs as A does.
   - A network that cannot see the overall pT scale cannot compute m (m² ≈ pT²·Σ z_i z_j ΔR_ij²). It can only compute m/pT, which B changes.
   - So "removing the ΣpT dependence does not by itself give a boost-invariant mass reading" is a property of the input representation, not a measured property of the training. Only the ptflat retrain bears on weakness 7: B 21.1 / 25.0 % (seeds 0 / 1) against 35.4 % for the raw retrain.
5. **"Narrow training pT → angle reading" against the Top models.**
   - Top Landscape jets span 550–650 GeV; the sample IQR is 570–619. The hls4ml sample IQR is 993–1061 GeV, with 0.5–99.5 % at 707–1595.
   - Top constituents are massless four-vectors, which carry the same information as JEDI-linear's (pT, Δη, Δφ).
   - Yet the Top ParT, PN and LorentzNet B/A ratio is 0.21–0.31 at λ 0.9, against 0.77–1.30 for the hls4ml nets with raw pT input.
   - So neither the width of the training pT range nor four-vector input explains the hls4ml/Top difference. The difference is measured; its cause is not tested.
   - Absolute call-change rates are not comparable across datasets either. The same pT-blind EFN changes 36.8 % of calls on hls4ml (W/Z boundary, ~12 % mass separation) and 4.4 % on Top under the same B0.9.
6. **Controls.**
   - R: on-manifold symmetry control (C2ST 0.50).
   - T: off-manifold (C2ST 0.72 on hls4ml against 0.88 for B). It changes 2.4 % of JEDI n64 calls (2.4 % in the rerun). The B response is not generic sensitivity to any off-manifold input change of the same displacement.
   - The weight-permuted null is constant, as reported. No further control was needed for the B ≠ symmetry statement.
7. **QG.** Every boosted QG jet is outside the 500–550 GeV window at λ 0.9 (C2ST 0.99999). Only the λ 0.95/1.05 subsets are on-window, as the report states.

## Corrected reading
- **hls4ml.** All 11 hls4ml-trained nets change 21–38 % of 5-class calls under B at pT × 0.9, against 0.3–4.2 % under the matched rotation. Their outputs are not invariant under a boost that fixes both m and m_A.
  - For JEDI-linear n64, 62 % of the changed calls are W/Z→t. This flow also appears under the pT-only edit and matches the D1 ΣpT shortcut toward t.
  - At the W/Z boundary, JEDI-linear responds as Δ = Δ_P + Δ_A, with P and A in opposite mass directions and A larger.
  - For the pT-blind nets (EFN, ptnorm), B = A follows from their inputs.
- **JetClass.** ParT and ParticleNet change 1.5–4× fewer calls under B than under P or A (full models: 1.5–1.9× vs P, 3.2–3.4× vs A), and 2.5–9× more than under R.
- **Top Landscape.** ParT, ParticleNet and LorentzNet change 3–4× fewer calls under B than under P or A, although they were trained in a narrow pT window.
- **Causes not tested.** Narrow training pT is not the cause of the hls4ml behaviour; the Top taggers contradict it. H12 does not separate m from m_A.

## Corrected conclusion
We boosted each jet along its own axis, which changes its pT by 10 % but keeps its invariant mass (and its axis-moment mass m_A) fixed. Every hls4ml-trained tagger changed a quarter to a third of its calls (JEDI-linear: 33 %, against 2.4 % for a matched rotation). The JetClass and Top Landscape taggers changed several times fewer calls under the boost than under pT-only or angle-only scaling. For JEDI-linear, most of the changed calls are W or Z jets that become top, which also happens when only pT is lowered: this is the known ΣpT shortcut rather than a mass measurement. Why the hls4ml and Top taggers differ is not known: both were trained in narrow pT windows on equivalent particle inputs.

## Files
- `interp/h12_verify.py`
- `results/physics/h12_verify_hls.json`
- `results/physics/h12_verify_gpu.json`
- Logs: `/j-jepa-vol/aaron-interp/logs/d-h12-verify{,-g}.log`
