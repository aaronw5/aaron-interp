# P2 — ParT: pair route vs per-particle route for a real mass change (agent report, 2026-10-17)

Jobs `anrunw-ai-p2-quick` (50 jets/class, smoke, 71 s) and `anrunw-ai-p2` (TITAN Xp, 147 s), `python -m interp.p2_routes_part`.
Script `interp/p2_routes_part.py`; results `results/part/p2_routes.json` (smoke: `p2_routes_50.json`); table
`results/site/T12_part_routes.png` (rendered on the helper pod with `python interp/p2_routes_part.py --table`); log `logs/p2_cluster.log`
(copies of /j-jepa-vol/aaron-interp/{results/part,results/site,logs/p2.log}). No training.

## Procedure
- Sample: P1's 10,000 jets (rows 3,000–4,000 of each class's first test file, 1,000/class); baseline = `build` of the unedited particles
  (δ = 0), fed through `network.inputs` + the forward on precomputed inputs (`p1v_verify_part.fwd`).
- Edits (`part_core.build`): angles × k (every real particle's Δη, Δφ × k; pT and particle mass kept; px, py, pz, E, jet_pt, jet_energy
  rebuilt), k ∈ {0.9, 1.1, 1.25}; pT × c (every particle's pT and mass × c, i.e. every 4-vector × c), c ∈ {0.8, 1.25}.
  True Δlog m (jmass): −0.104, +0.094, +0.221; −0.223, +0.223.
- Variants: (a) full edit = (x_edit, v_edit); (b) pair route only = (x_base, v_edit): 4-vectors (the only source of the pair features;
  weaver's forward uses v nowhere else) from the edited jet, every per-particle input of the model's yaml (pt_log, e_log, logptrel,
  logerel, deltaR, deta, dphi; for ParT_full also charge, 5 PID flags, d0, d0err, dz, dzerr) from the unedited jet; (c) per-particle
  route only = (x_edit, v_base). (b) at angles × 1.1 is the reverse of P1's shift.
- What each edit changes (mean over 17.5 M real–real pairs, i ≠ j; fed per-particle features on real slots):
  angles × 1.1: ΔlnkT = ΔlnΔR = +0.095, Δlnm² +0.171, Δlnz 0; fed deltaR +0.092 (raw +0.023), e_log +0.002, logptrel +0.002,
  logerel +0.001, deta/dphi mean 0 (mean |Δ| 0.015), pt_log 0. angles × 1.25: +0.223, +0.402; deltaR fed +0.228. angles × 0.9: −0.105,
  −0.186. pT × c: ΔlnkT = ln c, Δlnm² = 2 ln c (±0.446), ΔlnΔR, Δlnz 0; fed pt_log and e_log 0.7 ln c (±0.156), every other
  per-particle feature 0.
- Metrics as P1/P1v: calls changed vs baseline, W→Z, Z→W, Δ(logit_W − logit_Z) on W, index 97 / s.d. on W and Z, Δlog m̂ = P1's ridge
  read-out of log m (fitted on the unedited z, seed-1 train half; R² 0.881 full, 0.920 kin, as P1) on the test half. Share = (b)/(a),
  (c)/(a). Additivity: (b) + (c) vs (a), and per jet L0 + ΔL_b + ΔL_c vs L_a.
- Controls: angles × 1.0 and pT × 1.0 through the same code, all three variants: 0.00 % calls changed, max |ΔL| 5.7e-6 (full) / 0 (kin);
  network.run(baseline) vs forward on precomputed inputs: 0 calls changed. Every edit: mask identical, padded slots zero, non-kinematic
  branches bit-identical. P1 reproduced: angles × 1.1 full 12.6 % changed, W→Z .371, Δlog m̂ +.076; × 1.25 37.2 %, .461; kin 18.7 %, .455.

## Responses
ParT_full (unedited: acc 85.9 %, W→Z 0.117, Z→W 0.160)
| edit (true Δlog m) | variant | acc | calls changed | W→Z | Z→W | Δ(lW−lZ) on W | idx 97 / s.d. W, Z | Δlog m̂ |
|---|---|---|---|---|---|---|---|---|
| angles × 0.9 (−0.104) | (a) full | 81.8 | 11.0 % | 0.034 | 0.435 | +0.34 | +0.171, +0.457 | −0.0907 |
|  | (b) pair only | 86.0 | 2.5 % | 0.082 | 0.189 | +0.16 | +0.047, +0.062 | −0.0010 |
|  | (c) per-particle only | 82.5 | 9.4 % | 0.040 | 0.390 | +0.34 | +0.161, +0.398 | −0.0884 |
| angles × 1.1 (+0.094) | (a) full | 80.9 | 12.6 % | 0.371 | 0.074 | −1.46 | −0.428, −0.194 | +0.0761 |
|  | (b) pair only | 85.6 | 2.5 % | 0.143 | 0.140 | −0.19 | −0.056, −0.042 | +0.0012 |
|  | (c) per-particle only | 82.2 | 10.7 % | 0.326 | 0.079 | −1.18 | −0.355, −0.184 | +0.0754 |
| angles × 1.25 (+0.221) | (a) full | 60.9 | 37.2 % | 0.461 | 0.031 | −2.39 | −0.747, −0.225 | +0.1707 |
|  | (b) pair only | 84.5 | 5.8 % | 0.190 | 0.120 | −0.51 | −0.145, −0.081 | +0.0036 |
|  | (c) per-particle only | 65.9 | 31.4 % | 0.494 | 0.040 | −2.32 | −0.709, −0.201 | +0.1718 |
| pT × 0.8 (−0.223) | (a) full | 79.0 | 14.5 % | 0.027 | 0.502 | +0.19 | +0.162, +0.560 | −0.1135 |
|  | (b) pair only | 85.8 | 3.2 % | 0.082 | 0.194 | +0.13 | +0.053, +0.086 | −0.0057 |
|  | (c) per-particle only | 79.9 | 12.8 % | 0.034 | 0.462 | +0.22 | +0.148, +0.507 | −0.1075 |
| pT × 1.25 (+0.223) | (a) full | 67.3 | 29.5 % | 0.505 | 0.043 | −2.35 | −0.726, −0.245 | +0.1477 |
|  | (b) pair only | 85.3 | 3.6 % | 0.155 | 0.135 | −0.28 | −0.085, −0.059 | +0.0075 |
|  | (c) per-particle only | 70.0 | 26.2 % | 0.492 | 0.052 | −2.23 | −0.677, −0.224 | +0.1386 |

| edit | share pair / per-particle: Δlog m̂ | ΔW→Z | ΔZ→W | Δ(lW−lZ) on W | idx 97 on W | (b)+(c) vs (a): Δlog m̂ | ΔW→Z | Δ(lW−lZ) on W | per jet: calls agree | R² ΔL |
|---|---|---|---|---|---|---|---|---|---|---|
| angles × 0.9 | 0.01 / 0.98 | 0.42 / 0.93 | 0.11 / 0.84 | 0.46 / 1.00 | 0.27 / 0.94 | −0.089 vs −0.091 | −0.112 vs −0.083 | +0.50 vs +0.34 | 0.987 | 0.98 |
| angles × 1.1 | 0.02 / 0.99 | 0.10 / 0.82 | 0.23 / 0.94 | 0.13 / 0.81 | 0.13 / 0.83 | +0.077 vs +0.076 | +0.235 vs +0.254 | −1.38 vs −1.46 | 0.987 | 0.98 |
| angles × 1.25 | 0.02 / 1.01 | 0.21 / 1.10 | 0.31 / 0.93 | 0.21 / 0.97 | 0.19 / 0.95 | +0.175 vs +0.171 | +0.450 vs +0.344 | −2.82 vs −2.39 | 0.909 | 0.95 |
| pT × 0.8 | 0.05 / 0.95 | 0.39 / 0.92 | 0.10 / 0.88 | 0.67 / 1.13 | 0.33 / 0.91 | −0.113 vs −0.114 | −0.118 vs −0.090 | +0.35 vs +0.19 | 0.982 | 0.97 |
| pT × 1.25 | 0.05 / 0.94 | 0.10 / 0.97 | 0.21 / 0.92 | 0.12 / 0.95 | 0.12 / 0.93 | +0.146 vs +0.148 | +0.413 vs +0.388 | −2.51 vs −2.35 | 0.955 | 0.97 |
"calls agree" = fraction of jets whose argmax of L0 + ΔL_b + ΔL_c equals the argmax of L_a; R² ΔL = R² of ΔL_b + ΔL_c for ΔL_a over all
10 logits. Per-jet R² of Δlog m̂_b + Δlog m̂_c for Δlog m̂_a: 0.92, 0.94, 0.81, 0.92, 0.91.

ParT_kin (unedited: acc 74.5 %, W→Z 0.162, Z→W 0.224; index 97 is not a W/Z neuron in ParT_kin)
| edit (true Δlog m) | variant | acc | calls changed | W→Z | Z→W | Δ(lW−lZ) on W | idx 97 / s.d. W, Z | Δlog m̂ |
|---|---|---|---|---|---|---|---|---|
| angles × 0.9 (−0.104) | (a) full | 69.0 | 18.0 % | 0.054 | 0.556 | +0.40 | −0.115, +0.031 | −0.0919 |
|  | (b) pair only | 74.2 | 5.1 % | 0.134 | 0.261 | +0.12 | −0.025, −0.011 | −0.0048 |
|  | (c) per-particle only | 70.0 | 14.8 % | 0.067 | 0.511 | +0.39 | −0.076, +0.053 | −0.0858 |
| angles × 1.1 (+0.094) | (a) full | 68.4 | 18.7 % | 0.455 | 0.087 | −1.07 | −0.006, −0.061 | +0.0793 |
|  | (b) pair only | 74.0 | 4.7 % | 0.192 | 0.195 | −0.11 | +0.035, +0.016 | +0.0044 |
|  | (c) per-particle only | 69.9 | 15.6 % | 0.423 | 0.103 | −0.92 | −0.032, −0.082 | +0.0765 |
| angles × 1.25 (+0.221) | (a) full | 46.4 | 49.6 % | 0.493 | 0.043 | −2.00 | −0.085, −0.076 | +0.1707 |
|  | (b) pair only | 72.9 | 9.5 % | 0.230 | 0.160 | −0.29 | +0.071, +0.040 | +0.0098 |
|  | (c) per-particle only | 50.7 | 44.1 % | 0.532 | 0.042 | −1.92 | −0.157, −0.147 | +0.1696 |
| pT × 0.8 (−0.223) | (a) full | 38.5 | 60.4 % | 0.634 | 0.040 | −3.17 | +1.163, +1.261 | −0.0288 |
|  | (b) pair only | 37.7 | 61.2 % | 0.286 | 0.023 | −2.77 | +1.254, +1.157 | +0.1535 |
|  | (c) per-particle only | 62.3 | 27.6 % | 0.046 | 0.657 | +0.29 | −0.171, +0.052 | −0.1663 |
| pT × 1.25 (+0.223) | (a) full | 44.5 | 51.3 % | 0.504 | 0.042 | −2.06 | −0.097, −0.090 | +0.1856 |
|  | (b) pair only | 69.7 | 13.4 % | 0.233 | 0.179 | −0.30 | +0.042, +0.009 | +0.0097 |
|  | (c) per-particle only | 49.8 | 44.9 % | 0.548 | 0.043 | −1.97 | −0.134, −0.126 | +0.1801 |

| edit | share pair / per-particle: Δlog m̂ | ΔW→Z | ΔZ→W | Δ(lW−lZ) on W | (b)+(c) vs (a): Δlog m̂ | ΔW→Z | Δ(lW−lZ) on W | per jet: calls agree | R² ΔL |
|---|---|---|---|---|---|---|---|---|---|
| angles × 0.9 | 0.05 / 0.93 | 0.26 / 0.88 | 0.11 / 0.86 | 0.31 / 0.98 | −0.091 vs −0.092 | −0.123 vs −0.108 | +0.51 vs +0.40 | 0.986 | 0.99 |
| angles × 1.1 | 0.06 / 0.96 | 0.10 / 0.89 | 0.21 / 0.88 | 0.11 / 0.86 | +0.081 vs +0.079 | +0.291 vs +0.293 | −1.03 vs −1.07 | 0.983 | 0.99 |
| angles × 1.25 | 0.06 / 0.99 | 0.21 / 1.12 | 0.35 / 1.01 | 0.14 / 0.96 | +0.179 vs +0.171 | +0.438 vs +0.331 | −2.21 vs −2.00 | 0.912 | 0.97 |
| pT × 0.8 | −5.34 / 5.78 | 0.26 / −0.25 | 1.09 / −2.35 | 0.87 / −0.09 | −0.013 vs −0.029 | +0.008 vs +0.472 | −2.47 vs −3.17 | 0.731 | 0.86 |
| pT × 1.25 | 0.05 / 0.97 | 0.21 / 1.13 | 0.25 / 0.99 | 0.14 / 0.96 | +0.190 vs +0.186 | +0.457 vs +0.342 | −2.26 vs −2.06 | 0.889 | 0.94 |
ParT_kin pT × 0.8: (a) Δlog m̂ is −0.029 (true −0.223), so its shares are ratios to a near-zero denominator. Variant (b) moves 0.54 of true W
jets to H→bb (unedited 0.004); own-class rates under (b): H→bb 0.96, H→cc 0.08, H→gg 0.10, H→4q 0.03, W 0.08 (unedited 0.68).
(a) sends 0.63 of W jets to Z and 0.19 to H→bb; (c) leaves W at 0.73.

## Reverse of P1's shift (variant (b) at angles × 1.1: pair features of a jet with true mass +9.4 %, every per-particle input unedited)
| | W→Z | W→W | Δ(lW−lZ) on W | idx 97 on W | Δlog m̂ | calls changed |
|---|---|---|---|---|---|---|
| ParT_full | 0.117 → 0.143 | 0.779 → 0.755 | −0.19 | −0.056 s.d. | +0.0012 | 2.5 % |
| ParT_kin | 0.162 → 0.192 | 0.684 → 0.649 | −0.11 | (+0.035) | +0.0044 | 4.7 % |
Same model, full edit angles × 1.1: W→Z 0.371 / 0.455, Δ(lW−lZ) −1.46 / −1.07. P1's φ + 0.06 shift (pair features fixed, deltaR edited):
W→Z 0.310 / 0.373. Float32 noise of the pair route alone (P1v variant (d), rigid rotation): 0.01–0.03 % (full) / 0.5–0.6 % (kin) of calls.

## Reading (scoped to these edits, this read-out, these jets)
- Read-out of log m: for every edit except ParT_kin pT × 0.8, the per-particle route carries 0.93–1.01 of Δlog m̂ and the pair route
  0.01–0.02 (ParT_full, angles), 0.05 (ParT_full, pT), 0.05–0.06 (ParT_kin); (b) + (c) equals (a) to ≤ 0.008.
- Calls: the pair route alone changes 2.5–5.8 % (full) / 4.7–13.4 % (kin, without pT × 0.8) of calls and moves W/Z calls in the direction of the mass
  change: share of ΔW→Z 0.10–0.21 at mass increases, 0.39–0.42 (full) / 0.26 (kin) at decreases; share of Δ(lW−lZ) on W 0.12–0.21
  (full, increases) and 0.46–0.67 (full, decreases, where (a) is +0.19 to +0.34). The per-particle route carries 0.81–1.13 of the same
  quantities. The pair-route response appears in the logits and index 97 (0.12–0.33 of the full shift on W, ParT_full) with a Δlog m̂ of
  ≤ 0.008 (ParT_full; ≤ 0.010 ParT_kin), i.e. it lies mostly outside the read-out's direction.
- Additivity: per-jet logits add (R² 0.94–0.99; calls agree on 0.89–0.99 of jets) except ParT_kin pT × 0.8 (R² 0.86, 0.73). Call-level
  quantities are sub-additive at the large edits ((b)+(c) exceeds (a) in ΔW→Z by 0.03–0.12 at × 1.25 and pT × 1.25).
- Reverse of P1: with every axis-relative input fixed, a +9.4 % mass change in the pair features moves W→Z by +0.026 (full) / +0.030
  (kin), 0.10 of the full-edit change in both models.
- The pair-route variant is not a ln m²-only edit: ln kT and ln ΔR change with ln m² under the angle scaling, ln kT with it under the pT
  scaling; the shares are for the pair features as a whole. The read-out was fitted on unedited jets, where the two routes co-vary.

## Notes
1. Control keys in the JSON are `ang1` and `pt1` (factor 1.0, formatted with :g).
2. ParT_kin's index-97 shares (e.g. −6.19 / 5.69 at angles × 1.1) divide by a full-edit shift of −0.006 s.d.; not meaningful.
3. Smoke run (50/class) gave the same signs; its read-out R² was 0.72 / 0.84 (250 training jets), so its Δlog m̂ values differ.
