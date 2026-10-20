# H19 verification: W-pair mass, JetClass vs Top Landscape models

Pass 1 (JetClass sample): script `interp/h19_verify.py`. GPU job `d-h19-verify` (7 min, exit 0), after the smoke run `d-h19-verify-smoke`.
Results are in `results/physics/h19_verify.json` and `h19_verify_perjet.npz`; the log is `logs/d-h19-verify.log`.

Pass 2 (Top Landscape sample, 2026-10-20): script `interp/h19_verify_top.py`. GPU job `d-h19-vtop` (5 min, exit 0), after the smoke run `d-h19-vtop-smoke`.
Results are in `results/physics/h19_verify_top.json` and `h19_verify_top_perjet.npz`; the log is `logs/d-h19-vtop.log`. Copies of both passes are on the volume.

## Conclusion
H19 built every edited jet for the JetClass networks about the axis of the unedited jet, which does not move when the jet is edited. The non-W control edit (N) moves the true jet axis by a median of 0.13–0.15 in (η, φ); the W edit moves it by about 0.02. JetClass networks respond to that axis offset by itself; the Top-trained networks recompute the axis and do not.

When each edited jet is rebuilt about its own constituent-sum axis, the JetClass ParT and ParticleNet respond to the W-pair mass at about 3–4 times the side-effect reference (MLPx) when the W pair is moved down to 60 GeV, on JetClass jets and on fresh Top Landscape jets. This is the same level as the Top-trained networks. H19's low-side null for the JetClass networks, and its finding that the JetClass-vs-Top contrast depends on the edit direction, both come from the unmoved axis.

Still open: after recentring, the JetClass full models keep a larger non-W response on JetClass jets (Δ ≈ −1.1 to −1.4 log-odds at 60 GeV, against about −0.3 to −0.5 for the other networks); its source is not identified.

## What was checked in the script
- **Sampling.** Rows are random, never first-k (rng draws among eligible rows). The jc QCD draw used rng 4192, as reported.
- **Particle order.** It is preserved through the edit, `jc_dict` and `build` (pT is unchanged, so the pT sort is stable). Constituent masses come from `kin(J)` in the same order.
- **Working points, observables, metric.** The ε_S thresholds come from the sample's unedited tops. The acceptance and observables are computed on the same P4 the networks see. Score = L(Tbqq) − L(QCD) with Tbqq = 8, QCD = 0. The figure agrees with the JSON.
- **Axis convention (the defect).** `jc_dict` (JetClass sample) and `top_base` (Top sample) fix the axis at the unedited jet's axis. `h4_top_wmass.jet_inputs`, used by every Top-trained network, recomputes the axis from the edited constituents.
  - The H4 edits are rigid subjet moves at fixed jet mass, and they do not conserve the jet direction.
  - So the JetClass networks receive axis-relative inputs (Δη, Δφ, ΔR) translated by the axis shift of each edit. The Top networks do not.
  - D4 and D4m already show that ParT and PN respond to translated inputs.

## Procedure
**Sample.** Fresh JetClass test rows from `TTBar_100.root` and `ZJetsToNuNu_100.root`, in the H19 window (550 < jet_pT < 650 GeV, |η| < 2, ≥ 3 particles). The 5,000 + 5,000 H19 rows are excluded (overlap 0). Drawn: 5,000 Tbqq (rng 4391) and 3,000 QCD (rng 4392).

**Edit and acceptance.** As in H19 (`h4_verify.edit_set`), for the sets T60, T100 and T80. Analysis jets: 703 (T60), 483 (T100), 472 (T80).

**References.** MLPx_jc and MLPxW_jc, retrained exactly as in H19 (TTBar_101 / ZJetsToNuNu_101, 25,000 per class).

**Paths for the JetClass networks:**
- **st** (H19): inputs about the stored axis.
- **rc**: inputs about the constituent-sum axis of the jet being scored. The id, W, N and R variants are each recentred.

**New control T (translation only).** The unedited jet is translated rigidly in (η, φ) by the axis shift of the W edit (TW) or of the N edit (TN), then scored on the st path.

**Statistics.**
- Mean Δ log-odds with a 95 % bootstrap CI over jets (B = 2,000).
- Sign-flip permutation p.
- Holm correction over 39 tests (3 sets × 13 network-path pairs; the JSON `holm_family` field): (W − N) vs MLPx_jc, one-sided in the mW-window direction.
- McNemar on W vs N calls at ε_S = 0.5.

## Checks
**Unedited models on this sample** (8,000 jets; AUC / binary accuracy / 10-class accuracy):

| model | AUC | binary acc. | 10-class acc. |
|---|---|---|---|
| ParT_full | 0.9986 | 98.0 % | 88.0 % |
| ParT_kin | 0.9932 | 95.8 % | 83.0 % |
| PN_full | 0.9983 | 97.9 % | 87.4 % |
| PN_kin | 0.9901 | 95.2 % | 81.4 % |
| Top ParT | 0.9868 | 94.6 % | – |
| Top PN | 0.9858 | 94.3 % | – |
| Top LorentzNet | 0.9878 | 94.7 % | – |

- The rc path gives the same unedited numbers to within 0.0005 in accuracy.
- The stored axis and the constituent-sum axis agree on unedited jets: median ΔR 4×10⁻⁴, 99th percentile 0.012.

**Edit-induced axis shift** (median / 90th percentile, ΔR in (η, φ)):

| set | W | N | R |
|---|---|---|---|
| T60 | 0.021 / 0.065 | 0.130 / 0.225 | 0.004 |
| T100 | 0.021 / 0.066 | 0.085 / 0.195 | 0.005 |
| T80 | 0.011 / 0.038 | 0.065 / 0.132 | 0.002 |

## Numbers
### H19 replication (st path, fresh jets)
| set | model | W − N [95 % CI] | × MLPx | H19 |
|---|---|---|---|---|
| T60 | ParT_full | −0.54 [−0.73, −0.34] | 1.22 | −0.52, 1.18× |
| T60 | ParT_kin | −0.21 | 0.48 | 0.43× |
| T60 | PN_full | −0.56 | 1.26 | 1.34× |
| T60 | PN_kin | −0.33 | 0.74 | 0.98× |
| T100 | ParT_full | −1.81 | 3.92 | 4.16× |
| T100 | other JetClass nets | – | 2.7–3.7 | – |
| T80 | JetClass nets | +0.19 to +0.38 (Holm 0.0195) | – | +0.31 to +0.54 |

Top ParT, PN and LorentzNet: 3.03, 2.91 and 5.94 × MLPx at T60; 3.35, 4.14 and 6.01 × at T100.

H19's numbers reproduce on fresh jets.

### Translation-only control (st path, JetClass networks)
**T60.**
- dTN (the unedited jet moved by the N edit's axis shift) is −1.61 [−1.75, −1.46] for ParT_full, −1.69 for ParT_kin, −2.12 for PN_full and −1.88 for PN_kin.
- The N edit itself gives dN −2.66, −2.18, −2.17 and −1.75.
- dTW is −0.13 to −0.28.

**T100.** dTN is −0.89 to −1.42.

**T80.** dTN is −0.52 to −0.80.

**Top networks.** |dTW| and |dTN| ≤ 0.04 in every set; they recentre their inputs.

### Recentred path (rc), JetClass networks
| set | model | dW | dN | W − N [95 % CI] | × MLPx [CI] | Holm |
|---|---|---|---|---|---|---|
| T60 | ParT_full | −3.21 | −1.41 | −1.80 [−2.03, −1.59] | 4.06 [3.48, 4.76] | 0.0195 |
| T60 | ParT_kin | −2.53 | −0.66 | −1.87 [−2.05, −1.69] | 4.21 [3.69, 4.84] | 0.0195 |
| T60 | PN_full | −2.41 | −1.07 | −1.34 [−1.54, −1.13] | 3.01 [2.55, 3.56] | 0.0195 |
| T60 | PN_kin | −1.89 | −0.45 | −1.45 [−1.61, −1.28] | 3.25 [2.81, 3.76] | 0.0195 |
| T100 | ParT_full | −3.25 | −0.64 | −2.61 | 5.66 [4.66, 7.21] | 0.0195 |
| T100 | ParT_kin | −2.77 | −0.34 | −2.43 | 5.26 | 0.0195 |
| T100 | PN_full | −2.40 | −0.44 | −1.96 | 4.26 | 0.0195 |
| T100 | PN_kin | −1.99 | −0.08 | −1.90 | 4.13 | 0.0195 |
| T80 | ParT_full | – | – | −0.06 [−0.24, +0.14] | – | 1 |
| T80 | ParT_kin | – | – | +0.01 | – | 0.77 |
| T80 | PN_full | – | – | +0.03 | – | 0.68 |
| T80 | PN_kin | – | – | −0.02 | – | 0.93 |

- **References.** MLPx_jc gives W − N of −0.44 (T60), −0.46 (T100) and −0.08 (T80). MLPxW_jc gives −1.61 (3.6×), −1.40 (3.0×) and +0.07 [0.00, 0.15].
- **Paired change of W − N, st minus rc** (ParT_full): +1.26 [1.12, 1.40] at T60, +0.80 [0.66, 0.95] at T100, +0.32 [0.24, 0.40] at T80. For all four networks in all three sets, H0 (dN on the st path = dN on the rc path) has p = 0.001 (two-sided).
- **Calls at ε_S = 0.5, T60, rc path** (pass rate id → W / N):
  - ParT_full: 65.0 → 20.9 / 48.1 %; McNemar 226 / 35, p 2×10⁻³⁵. On the st path the N pass rate is 31.9 %.
  - ParT_kin: 68.4 → 25.0 / 59.0 %; p 3×10⁻⁴⁵.

### Hypotheses
- **H0_axis** (the H19 JetClass T60 null and the T80 sign come from the fixed axis): predicts dTN ≈ dN on the st path, and W − N approaching MLPxW on the rc path. Both are observed.
- **H19 reading** (JetClass networks treat any subjet-pair rearrangement alike, so the response depends on the edit direction): predicts that rc dN stays close to dW at T60. This is excluded: rc dN/dW is 0.26–0.47, and the rc W − N lies 3–4× MLPx with all CIs above 2.5.
- **H1 of H19** (JetClass networks at 2–4× MLPx on JetClass jets): met at both T60 and T100 on the rc path.

## Pass 2: Top Landscape sample (job d-h19-vtop)
**Sample.**
- 5,000 fresh Top Landscape test tops (rng 4591), excluding the H4, H4v and H19 rows (overlap 0; 187,085 fresh tops available).
- No fresh QCD rows remain after H19. Unedited accuracy therefore uses H19's 4,360 QCD rows (reused, not fresh).

**Edit, sets, acceptance.** As in H19. Analysis jets: 584 (T60), 425 (T100), 490 (T80).

**Paths for ParT_kin and PN_kin:**
- **st**: H19's `top_base` of the unedited jet, for every variant.
- **rc**: `top_base` of the jet being scored.

**Translation control.** TW / TN as in pass 1. On the rc path TN is within ±0.01 for both networks, as a recentred input must be.

**References.** MLPx_jc and MLPxW_jc, retrained as in H19. Top-trained networks are shown against MLPx_jc for comparison only. H19 compared them with MLPx_top, which was not refitted here.

**Holm.** Over 21 tests.

**Unedited models** (9,360 jets; AUC / binary accuracy; the st and rc paths are identical on unedited jets):

| model | AUC | binary acc. |
|---|---|---|
| ParT_kin (JetClass) | 0.9813 | 92.5 % |
| PN_kin (JetClass) | 0.9838 | 93.1 % |
| Top ParT | 0.9864 | 94.0 % |
| Top PN | 0.9860 | 94.1 % |
| Top LorentzNet | 0.9869 | 94.3 % |

**Axis shift** (median ΔR): W 0.023, N 0.151 (T60); W 0.019, N 0.082 (T100); W 0.011, N 0.069 (T80).

**References.**
- MLPx_jc W − N: −0.49 [−0.54, −0.43] (T60), −0.25 [−0.35, −0.16] (T100), −0.17 [−0.21, −0.12] (T80).
- MLPxW_jc W − N: −1.70 (T60), −0.99 (T100), 0.00 [−0.05, +0.05] (T80).
- LR2_jc: |W − N| ≤ 0.02.

| set | model | path | dN [95 % CI] | dTN | W − N [95 % CI] | × MLPx_jc [CI] | Holm |
|---|---|---|---|---|---|---|---|
| T60 | ParT_kin | st | −2.09 [−2.23, −1.95] | −1.73 | +0.02 [−0.13, +0.18] | −0.05 [−0.38, 0.25] | 1 |
| T60 | ParT_kin | rc | −0.44 [−0.54, −0.35] | +0.01 | −1.82 [−1.99, −1.65] | 3.76 [3.35, 4.25] | 0.011 |
| T60 | PN_kin | st | −1.43 [−1.53, −1.34] | −1.62 | −0.50 [−0.62, −0.37] | 1.02 [0.76, 1.30] | 0.87 |
| T60 | PN_kin | rc | −0.33 [−0.43, −0.23] | +0.01 | −1.51 [−1.66, −1.35] | 3.11 [2.73, 3.55] | 0.011 |
| T60 | Top ParT / PN / LNet | st | −0.33 / −0.24 / −0.57 | ≤ 0.05 | −1.71 / −1.92 / −2.68 | 3.5 / 4.0 / 5.5 | 0.011 |
| T100 | ParT_kin | st → rc | −0.83 → −0.17 | −0.75 → 0.00 | −0.75 → −1.52 | 2.95 → 6.00 [4.51, 8.93] | 0.011 |
| T100 | PN_kin | st → rc | −0.61 → −0.11 | −0.91 → 0.00 | −0.81 → −1.20 | 3.18 → 4.72 [3.56, 7.08] | 0.011 |
| T80 | ParT_kin | st → rc | −0.45 → −0.14 | −0.46 → 0.00 | +0.38 [0.25, 0.50] → +0.03 [−0.08, +0.15] | – | – |
| T80 | PN_kin | st → rc | −0.44 → −0.11 | −0.56 → +0.01 | +0.27 [0.18, 0.37] → −0.01 [−0.10, +0.07] | – | – |

- **H19's Top-sample numbers reproduce on the st path.** H19 had ParT_kin 0.13× and PN_kin 1.22× at T60, and 3.59× and 2.99× at T100.
- **Paired st − rc change of dN** (T60): ParT_kin −1.64 [−1.76, −1.52]; PN_kin −1.11 [−1.20, −1.01]. Both have p = 0.001 (two-sided sign-flip).
- **Calls at ε_S = 0.5, T60** (pass rate id → W / N):
  - ParT_kin st: 73.6 → 28.9 / 26.0 % (McNemar W vs N p 0.18).
  - ParT_kin rc: 73.6 → 24.8 / 62.5 % (p 4×10⁻⁴³).
  - PN_kin rc: 72.9 → 31.0 / 65.8 % (p 4×10⁻⁴²).
  - Top ParT: 70.9 → 36.3 / 66.3 % (p 1×10⁻³²).
- **T80 Holm values.** At T80 the Holm values of the rc rows (0.011) test W − N against MLPx_jc (−0.17), not against 0. The rc W − N CIs contain 0, matching MLPxW_jc (0.00).
- **Recentred non-W response.** On Top jets the recentred dN of the JetClass kin networks at T60 (−0.44, −0.33) is at the level of the Top networks (−0.24 to −0.57). The larger residual non-W response is seen only for the JetClass full models on JetClass jets (pass 1).

## Corrected reading
- **Top-trained networks.** Top ParT, PN and LorentzNet respond to the W-pair mass at 2.9–6.0 × MLPx on fresh JetClass jets, at T60 and at T100. This replicates H19's sample-independence result.
- **JetClass-trained networks.** ParT and PN (full and kin) respond at 3.0–4.2 × MLPx (T60) and 4.1–5.7 × MLPx (T100) when the inputs are computed about each edited jet's own axis. This is at or above the MLPxW level (3.6× and 3.0×).
- **T80 on the rc path.** W − N ≈ 0 for all networks, consistent with MLPxW (+0.07).
- **What H19's T60 null and T80 positive measured.** The JetClass networks' response to the 0.13 (median) axis offset that the N control introduces when the axis is held fixed. They do not measure a direction-dependent use of the W pair.
- **Remaining differences.** On the rc path the JetClass full models still have dN of −1.1 to −1.4 at T60, against −0.26 to −0.54 for the Top networks. The source is not identified.
- **Top Landscape sample (pass 2).** The same defect applies to the H19 Top-sample numbers for the JetClass kin models (`top_base` of the unedited jet). On fresh Top tops, with recentring, ParT_kin and PN_kin give 3.8× and 3.1× MLPx_jc at T60, against −0.05× and 1.0× on H19's fixed-axis path (details below).
- **Figure.** `results/physics/h19_wpair.png` agrees with `h19_wpair_cross.json`. The JetClass-network points in it are the fixed-axis values, so its T60 panel shows the artefact. It needs a redraw with the recentred values before it is used.
- **H3v.** H3v's JetClass ParT null should be checked for the same axis convention before it is used against D6.

## Explainer note (for site/explain.py, H19 entry, checks section)
A rigid move of subjets at fixed $m_J$ changes the jet direction: the N control moves it by a median $\Delta R\approx0.13$. JetClass networks take $\Delta\eta,\Delta\phi$ relative to the jet axis, and they respond to a translated input ($\Delta s\approx-1.6$ to $-2.1$ for a pure translation by the N shift). Each edited jet must therefore be rebuilt about its own axis $\hat n=\sum_i p_i/|\sum_i p_i|$. With that rebuild, $\Delta_{W-N}/\Delta_{W-N}^{\rm MLPx}$ is 3.0–4.2 at 60 GeV and 4.1–5.7 at 100 GeV for the JetClass networks.
