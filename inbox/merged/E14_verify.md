# E14 verification (skeptic agent, 2026-10-17)

Script interp/e14_verify_jedi.py. It imports E14's Program and helpers unchanged. Modes:
- main: fresh-sample rerun plus V1–V3;
- `--head`: why the counterfactual depends on the head;
- `--ci`: 95 % bootstrap intervals;
- `--range`: d4 input range.

Cluster jobs:
| job | mode | wall time | exit |
|---|---|---|---|
| e14-vsmoke | 200 jets | 208 s | 0 |
| e14-verify | main | 657 s | 0 |
| e14-verify2 | `--head` | 744 s | 0 |
| e14-verify3 | `--ci` | 350 s | 0 |
| e14-verify4 | `--range` | 256 s | 0 |

All jobs ran at 1 CPU / 2 GiB. Results: results/jedi/e14_verify.json, e14_verify_head.json, e14_verify_ci.json, e14_verify_range.json. Logs: logs/e14-verify*.log.

Sample: `sample(per_class=2000, seed=31)` = 10,000 test jets. It shares 394 rows with E14 (seed 8) and 392 with E6v (seed 17). Fixed-point accuracy is 81.40 %. The program (float + masks) scores 80.73 %; it changes 4.19 % of calls and differs from make_masked by at most 7.6e-5 in u. With g at its mean the program scores 73.53 %. Intervals are 95 % bootstrap intervals over jets (1000 resamples, paired across directions), on the 9,999 jets with no input wrap in any edit.

## 1. Code review (interp/e14_halfplanes_jedi.py)
- **Stages and masks.** The program reproduces e4_e5_jedi.make_masked. The Part-1 decomposition is exact: the largest error is 3.1e-5 on the fresh sample. Edits (shift, rotate, mirror) move real particles only. Part 1 and the Part-4 sector statistics use real particles in each unit's pooled slots (the qs1 bits). The overrides (a_j → 0, g frozen) act on the intended stage.
- **Padded slots, a56.** The a56 ablation also zeroes a56 in padded slots, where a56 is on (pT = 0 < 31 GeV). Its 53 % calls changed therefore includes padded-slot terms. Its sector statistics do not include them, because they use real particles only.
- **Half-plane counts.** The text "nine-direction polygon" does not match the weights. The number of half-planes with nonzero Part-1 weight is:

| units | half-planes with nonzero weight |
|---|---|
| u51, u63 | 3 |
| u17, u20, u30, u52, u62 | 4 |
| u1, u29, u50, u56 | 5 |
| u60 | 6 |
| u13, u43 | 7 |
| u40 | 9 |

- **"One weighted sum".** Four of the 15 units take half-plane input through more than one s unit, each with its own ReLU:
  - u13 and u43 through s32, s53 and s60;
  - u40 through s32, s52 and s53;
  - u60 through s43 and s53.
- **Permutation control.** With 3–5 nonzero weights out of nine, many permutations produce no closed contour. Those draws are dropped. E14 kept 25–163 complete draws of 200 per contour, so its p resolution is 0.006–0.04, not "p < 0.005 for each". On seed 31, one permuted draw reproduces the u63 contour at 3 GeV exactly (p = 1/45 = 0.022).
- **Hybrid counterfactual.** E14 computes Δlog m̂ for a symmetrized map as head(S_own + ½[ΔS(x′) + ΔS(−x′)]) − head(S_own). Part of that ΔS comes from the reflected jet. The resulting S vector has negative entries in 6–24 % of jets (Section 3, V1). The fixed-point network cannot produce those values. The float program head accepts them.
- **m = 0 power fraction.** This metric (0.975–1.000) is dominated by the mean of f over all responding cells, including cells where f does not depend on θ. The contour anisotropy is measured by max/min (1.14–1.47), not by this number.
- **Statistics (WORKFLOW rule of 2026-10-17).** E14 reports no 95 % intervals and states no H0/H1. The 51 contour-control tests (17 contours × 3 controls) have no multiple-comparison correction. With 200 draws the smallest attainable p is 0.005, which is above the Holm threshold of 0.05/51 = 0.001.

## 2. Fresh-sample rerun (seed 31; E14 seed-8 values in brackets)
**Part 1.** Shares and paths reproduce within 4 points:
- half-plane share of the u variance:

| unit | share, % [seed 8] |
|---|---|
| u50 | 175 [175] |
| u1 | 124 [124] |
| u40 | 96 [97] |
| u29 | 83 [83] |
| u56 | 77 [77] |
| u60 | 73 [73] |
| u52 | 59 [59] |
| u13 | 37 [38] |
| u43 | 22 [22] |
| u30 | 20 [20] |
| u62 | 19 [19] |
| u63 | 18 [20] |
| u51 | 6 [7] |

- HP via g: u30 32.3 % [31.8]; u31 and u58 100 % [100]; 0 for every other unit.
- The s paths are identical.

**Part 2** (program, g at this sample's mean):
- 78 complete contours, max/min 1.142–1.465 [1.14–1.47].
- Polygon with Part-1 weights: rms relative error median 5.5e-8 [6e-8], ≤ 0.005 in 60/78 [60/78], max 0.064 [0.064].
- Controls (slot 5; 3 and 10 GeV; 17 contours; 200 draws each), median rms:
  - random directions with fitted weights 0.046–0.083 [0.046–0.089];
  - rotated directions 0.022–0.059 [0.022–0.059];
  - permuted Part-1 weights 0.22–0.81 [0.21–0.72].

**Part 3**, Δlog m̂ (h5 ridge, held-out R² 0.975), δ = 0.03:
- Measured, g free, in the order +η/−η/+φ/−φ/+diag/−diag: .086/.120/.089/.113/.080/.122 [.091/.125/.094/.117/.085/.126]. W→Z: .26/.59/.30/.52/.23/.61 [.26/.60/.29/.53/.21/.62].
- Odd parts and ratios (η, φ, diag) with 95 % intervals:

| quantity | δ = 0.03 | δ = 0.06 |
|---|---|---|
| odd/even, measured, g free | −0.164 [−0.171, −0.158], −0.118 [−0.124, −0.111], −0.208 [−0.215, −0.200] (seed 8: −0.155, −0.109, −0.195) | −0.104, −0.057, −0.113 |
| odd/even, maps | −0.145, −0.097, −0.176 (seed 8: −0.143, −0.092, −0.173) | −0.079, −0.049, −0.090 |
| odd(maps) / odd(measured, g frozen) | 0.99 [0.97, 1.01], 0.96 [0.93, 0.99], 0.98 [0.96, 0.99] | 1.00 [0.99, 1.02], 1.05 [1.03, 1.08], 1.03 [1.02, 1.05] |
| odd(g frozen) / odd(g free) | 0.85 [0.83, 0.87], 0.84 [0.81, 0.87], 0.84 [0.82, 0.86] | 0.76 [0.74, 0.78], 0.84 [0.80, 0.88], 0.78 [0.76, 0.80] |
| odd(maps) / odd(g free) | 0.84, 0.80, 0.82 (seed 8: 0.88, 0.81, 0.85) | 0.76, 0.88, 0.81 (seed 8: 0.80, 0.91, 0.84) |

  The context share of the odd part is 1 − frozen/free: 15–24 % on seed 31, against 9–21 % on seed 8 from the same ratio.
- Program counterfactuals (E14 construction):
  - point-symmetrized maps, odd part: +0.0004 [−0.0002, +0.0009], +0.0004 [−0.0001, +0.0008], +0.0005 [−0.0000, +0.0011] at δ 0.03; |odd| ≤ 0.0005 at δ 0.06, every interval covering 0;
  - θ-averaged maps: odd ≤ 0.0006;
  - even in η: η-odd +0.0004, φ-odd −0.0095;
  - even in φ: φ-odd +0.0004, η-odd −0.0142.
- Even parts:
  - δ 0.03: measured 0.101–0.103, maps 0.098–0.099, θ-averaged maps 0.099;
  - δ 0.06: measured 0.260–0.267, maps 0.268–0.269.
- W→Z with θ-averaged maps: .34–.36 [.33–.35].
- Per-unit ΔS, maps vs measured (g frozen), δ = 0.06:

| unit | slope | per-jet corr | Spearman |
|---|---|---|---|
| S29 | 1.06 | 0.93–0.95 | 1.00 |
| S1 | 1.08 | 0.94–0.96 | 1.00 |
| S50 | 1.07 | 0.93–0.95 | 1.00 |
| S43 | 1.00 | 0.995 | 1.00 |
| S30 | 1.01 | 0.86–0.89 | 0.83 |

  At δ 0.03 the per-jet correlations are lower: 0.83–0.88, and S30 0.71–0.78.
- The S30 over-prediction recurs: η odd/even, maps vs measured g frozen, is −0.306 vs −0.144 at δ 0.03 and −0.159 vs −0.088 at δ 0.06.

**Part 4** (one half-plane → 0, all particles):
- Calls changed: 8.7–30.5 % with g free [8.8–31.2], 6.9–28.6 % with g frozen [7.0–29.0].
- Own-half fraction: g frozen ≥ 0.991 [0.991]; g free 0.854–0.945 [0.856–0.944].
- Own/opposite sector, g free: 9.4–19.9 [9.5–20.2].
- With g frozen the opposite sector changes by exactly 0 for all nine.
- Median sector correlation, program vs measured: ≥ 0.9998.
- Controls: a0 changes 0 % of calls and 0 of S. a56 has sector CV 0.03 and changes 53.2 % of calls with g free.

## 3. Controls E14 did not run
**V2. Map vs sample, on the fixed-point network, no counterfactual arithmetic.**
- Procedure: the same six translations applied to the point-reflected sample −x (real particles negated). The odd part splits as odd(x) = A + B and odd(−x) = A − B. A is carried by the network's per-particle response; B comes from the sample's asymmetry.
- H0: B = 0. H1: B ≠ 0.
- δ = 0.03, g free, A (η, φ, diag): −0.0166 [−0.0168, −0.0163], −0.0115, −0.0205.
- δ = 0.03, g free, B: −0.0004 [−0.0011, +0.0002] (p 0.22), −0.0004 (p 0.22), −0.0005 (p 0.20).
- g frozen, B: −0.0000 to −0.0002 (p 0.39–0.89).
- δ = 0.06: B −0.0005 to +0.0003 (p 0.20–0.95).
- H0 is not rejected for any axis or δ. A equals the measured odd part within its interval.

**V1 and V4. Point-symmetrized maps on the fixed-point network.** With g frozen at g(x), S is a sum of per-particle terms, so ½[S(x′) + S(−x′)] is the S of the point-symmetrized map.
- E14's hybrid construction (S_own + ½[ΔS_x + ΔS_−x]) through the fixed-point head gives odd/even of +0.078/+0.052/+0.093 at δ 0.03 and +0.052/+0.025/+0.051 at δ 0.06. That is the opposite sign, at 50–60 % of the measured size.
- The cause, measured in `--range`:
  - all 19 live d4 inputs are unsigned;
  - the hybrid S has a negative entry in 6–24 % of jets (largest for −η, −φ, −diag);
  - real edits and the unedited sample have none;
  - with S clipped to the d4 range before the head, the hybrid gives +0.008/+0.004/+0.007 (δ 0.03) and +0.001/+0.001/+0.000 (δ 0.06).
- Transplanting the reflected jet's ΔS onto the original base:

| head | odd/even (η, φ, diag), δ 0.03 |
|---|---|
| fixed-point | 0.376, 0.271, 0.459 |
| fixed-point, S clipped | 0.159, 0.107, 0.192 |
| float program | 0.155, 0.106, 0.189 |

- A construction with no negative S: symmetrize both the base and the edited S, i.e. the head on S_sym(x + δd) minus the head on S_sym(x). It gives an odd part of +0.0000 [−0.0005, +0.0005], −0.0002, +0.0000 (δ 0.03) through the fixed-point head, and the same within 0.0001 through the program head. At δ 0.06: |odd| ≤ 0.0002, every interval covering 0. Its even part is 0.101 (δ 0.03) and 0.276 (δ 0.06).
- For real edits the fixed-point and program heads give the same Δlog m̂ within 0.0003. The S-override identity reproduces the g-frozen run exactly (0.0).

**V3. Baselines for the contour controls** (rms relative error of r_c(θ), slot 5, 17 contours):

| fit | rms relative error |
|---|---|
| best circle about the axis | 0.032–0.086 |
| Fourier m ≤ 1 | 0.027–0.084 |
| Fourier m ≤ 2 | 0.026–0.062 |
| Fourier m ≤ 4 (9 parameters) | 0.019–0.037 |
| nine random directions, 9 fitted weights (median) | 0.046–0.083; below the circle in 6/17 contours |
| real directions, Part-1 weights | 0.0000–0.0258; 0 for single-path units |

**Figure check.** results/jedi/e14_halfplanes.png panels (a)–(d) and e14_structure_table.png match the JSON. In panel (a) for S50 at 10 GeV, the fixed-point contour (dotted) lies about 0.08 rad inside the program contour, as the E14 caveat states.

## 4. Reading, corrected where the measurements differ
- **Part 1/2.** 15 particle-level u units take half-plane input through five s units (s43, s60, s53, s32, s52).
  - 11 of them use one s unit, so their on-contour is exactly the level set of one weighted sum of 3–5 half-planes.
  - u13, u43, u40 and u60 combine 2–3 s units. For them the averaged-weight polygon fits to rms ≤ 0.064 (u60, slot 20).
  - The contours are polygons in the 3–9 half-plane directions with nonzero weight; only u40 uses all nine. Max/min is 1.14–1.47; rms deviation from the best circle is 3–9 %.
  - Random or rotated direction sets with fitted weights reach 2–8 % rms, about the level of a circle (3–9 %). Permuted weights reach 22–81 % rms, except one draw for u63 that reproduces the contour exactly. The real directions give 0–2.6 %.
  - The untrained network has no on/off contour within r ≤ 0.4, so it gives no shape comparison.
- **Part 1, context.** g is per jet and adds no per-particle angular term. The half-plane P units reach the second round through g for u30 (32 % of its variance) and for u31 and u58 (100 %, jet-level units).
- **Part 3.** With g frozen, the per-particle maps reproduce the odd part of the h5 mass read-out under translation: 0.96–1.05 of the measured odd part across η, φ, diag and δ 0.03 / 0.06 (95 % intervals within [0.93, 1.08]); seed 8 gave 0.93–1.03.
  - The odd part is carried by the network's per-particle response, not by the sample: on the reflected sample −x the sample term B is −0.0005 to +0.0003 (p ≥ 0.20).
  - Point-symmetrizing or θ-averaging the maps removes the odd part (|odd| ≤ 0.0006, intervals covering 0). This holds in the program and, with base and edited S both symmetrized, on the fixed-point head. The even part changes by ≤ 1 % relative to the unsymmetrized response at the same g.
  - With g free, 15–24 % (seed 31) and 9–21 % (seed 8) of the odd part passes through g.
- **Part 4.** As E14 states: with g frozen the effect stays on its own half by construction; with g free 5–15 % spreads off the own half.
- **Scope.** The counterfactual Δlog m̂ values should come from the symmetrized-network construction. E14's hybrid construction sends S vectors with negative entries into the head (6–24 % of jets). It gives the same answer through the float head, and a sign-flipped one through the fixed-point head unless S is clipped.

Verifier fields: reproduced yes; refuted no; confidence 0.7 that the E14 reading holds as written. The wording items are "one weighted sum", "nine-direction polygon", the control description, the 9–20 % context range, and u31/u58 left out of the thesis note. The A4 part (odd term from the per-particle maps) holds on the fresh sample, with the V2 control and on the fixed-point network.
