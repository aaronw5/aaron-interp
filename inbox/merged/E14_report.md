# E14 — from nine half-planes to a radius: the second-round maps and the direction term of the mass read-out (worker report, 2026-10-16)

Claims: A3 ("the radial second-round objects are built in round two from round one's directional half-planes") and A4 (the 12–20 %
direction-dependent odd term of the mass response under translation "consistent with" the nine asymmetric half-plane cuts).
Script interp/e14_halfplanes_jedi.py (`python -m interp.e14_halfplanes_jedi`; `--smoke` 200 jets; `--plot` redraws from the JSON).
Cluster job e14-run (1 CPU / 2 GiB, 486 s, exit 0; pod peak memory 2047 MiB reported = request, cgroup value including page cache,
as in e5v/e15). Smoke run e14-smoke2 (200 jets) before it. Results: results/jedi/e14_halfplanes.json, e14_maps.npz,
e14_halfplanes.png (figure), e14_structure_table.png (u unit × source table); log logs/e14-run.log.

## Setup
- Sample: `sample(per_class=2000, seed=8)` = 10,000 test jets. Fixed-point network (jedi_core.Jedi): accuracy 80.77 %.
- Program: float + zero-bit masks (E3m/E5 arithmetic, written in numpy so the pre-activations decompose; matches
  e4_e5_jedi.make_masked to 8.8e-5 in u): 80.30 %, 4.10 % calls differ from fixed point. With g frozen at its sample mean (the E5 f_c
  construction, "the maps"): 73.56 %. Untrained control Jedi(untrained=True, seed 0): 20.0 %, live S units 13, 58.
- The nine half-plane statements a_j = ReLU(n_j·(Δη, Δφ) + β_j), n_j in rad⁻¹ (normal angle; cut offset from the axis):
  a29 (72, 0) 0° 0.0069 rad; a6 (36, 45) 51° 0.0082; a40 (−36, 42) 131° 0.0034; a57 (−39, 28) 144° 0.0001; a3 (−50, 17) 161° 0.0001;
  a43 (−44, −20) 204° 0.0052; a30 (−22, −56) 248° 0.0073; a36 (0, −78) 270° 0.0024; a46 (17, −45) 290° 0.0000.

## Part 1 — structure (program, exact decomposition; max error 3.1e-5)
pre_u_c(i) = Σ_j G_jc(i) a_j(i) + s-bias term + context term + b3; variance shares cov(term, pre)/var(pre) over real particles in
each unit's pooled slots. "HP via g" = context term minus the context term with g recomputed from P with the nine half-plane P units
at their sample mean.
- Angular input reaches u through five s units, each a weighted sum of a subset of the half-planes (W2 weights):
  s43 = ReLU(1.75 a29 + 1.50 a6 + 1.88 a36 + 2.75 a40 + 2.00 a43 − 1.50 a56 − 1.12 a4 + b) → u1 (W3 +0.47), u29 (+0.50), u56 (+0.18), u60 (−0.50);
  s60 = ReLU(−2.50 a6 − 1.00 a43 − 2.75 a46 − 2.00 a3 + b) → u52 (−4.00), u62 (−0.50), u43 (−0.62), u13 (−0.50), u30 (+0.16);
  s53 = ReLU(−2.00 a29 − 2.50 a40 − 2.25 a30 + 1.00 a15 + b) → u51 (+1.00), u63 (−1.00), u60 (+0.62), u13, u43;
  s32 = ReLU(−4 a6 − 2 a43 − 4 a46 − 3 a3 + 1.06 a15 + b) → u17 (+1.75), u20 (+0.62), u13 (−0.84), u43 (−0.44);
  s52 (a57 0.69, a36 0.41, a6 0.38, a29 0.31, a43 0.19, a32 −8.0, …) → u50 (+3.00), u40 (+1.12).
  Net effective weights ∂u/∂a_j on active particles have one sign per unit (e.g. u29: a29 +0.88, a6 +0.75, a36 +0.94, a40 +1.38, a43 +1.00;
  u17: a6 −4.0, a43 −2.0, a46 −4.0, a3 −3.0).
- Half-plane share of the pre-activation variance (%): u50 175, u1 124, u40 97, u29 83, u56 77, u60 73, u52 59, u13 38, u43 22, u63 20,
  u30 20, u62 19, u51 7, u20 2, u0 0, u17 −3, u19 0 (pT-only units: u0, u17, u19, u20). HP via g: 0 for every particle unit except
  u30 (31.8 %); u31, u58 100 % (g-only units).
- HP via s / (HP via s + |HP via g|) = 1.00 for the 13 units with a positive half-plane share other than u30 (u17, share −3 %, also
  receives half-plane input only through s); u30 0.38; u31, u58 0.

## Part 2 — polar maps and on-contours
Maps f_c(pT, r, θ) of one particle at slot 5 and 20 (pT 1, 3, 10, 30, 100 GeV; r 0–0.4, 41 steps; θ 72 × 5°); on-contour r_c(θ) =
first change of the on/off state along r on a fine grid (0.0005 rad) of the program maps; 78 complete contours (unit × slot × pT).
- θ-power fraction in m = 0 (all cells where the unit responds): 0.975–1.000 for every live unit at a pooled slot (u56 0.502 at slot 5,
  which u56 does not pool; 0.962 at slot 20). Reference: one half-plane ReLU(cos θ) gives 0.405.
- Contour anisotropy max/min 1.14–1.47. At 3 GeV, slot 5: u1 1.164, u29 1.189, u30 1.303, u43 1.302, u52 1.290, u62 1.304,
  u60 1.378, u63 1.427 (u50 at 10 GeV 1.148). θ of the minimum: 230–245° (−η/−φ quadrant) for u1, u29, u30, u43, u52, u62 at
  every pT; 125–130° for u60, u63 at 1–3 GeV. Direction of the minimum of the m = 1 component of r_c: 147–172° for the s43 and s60
  units (relative amplitude 0.03–0.12), 98–124° for u60, u63.
- Polygon model Σ_j w_j ReLU(r n_j·x̂ + β_j) = L (one fitted level L; shape predicted): with the Part-1 weights, rms relative error
  of r_c(θ) median 6e-8, ≤ 0.005 for 60 of 78 contours, max 0.064 (u60, slot 20); predicted max/min equals the measured one within
  0.003 for the single-path units (u1, u29, u30, u50, u52, u62, u63). Local-gradient weights ≤ 0.023 (62 contours); nine fitted
  weights ≤ 0.009. Equal weights on all nine: 0.19–0.28 (max/min 1.82–1.94).
  Controls (slot 5, 3 and 10 GeV, 17 contours, 200 draws each): Part-1 weights permuted among the nine half-planes, median rms
  0.21–0.72; nine random directions with fitted weights, median 0.046–0.089; the nine directions rotated by a random angle with
  fitted weights, median 0.022–0.059. No draw reached the real-direction error in any of the 17 contours (p < 0.005 each).
- Fixed-point network maps (g at its mean; r grid 0.01): contours within 0.016 rad of the program at slot 5 (u50 at 10 GeV:
  0.08 rad), within 0.043 rad at slot 20; max/min 1.16–1.70.
- Untrained control: no unit map turns on or off within r ≤ 0.4 (all pT, both slots); no contour exists.

## Part 3 — translation: maps vs measured, mass read-out, symmetrized-map counterfactuals
Rigid translation δ = 0.03, 0.06 rad along ±η, ±φ, ±diag (real particles; padded slots untouched; 0 jets with input wrap) and a 45°
rotation. Measured: fixed-point network, g free and g frozen at the unedited jet's g. Predicted ("maps"): program with g at its
sample mean = Σ over the jet's particles of f_c. Mass read-out: ridge of log m (invariant mass from the particles) on the
fixed-point h5, fit on 5,000 jets, held-out R² 0.971; predicted Δlog m̂ = read-out of the program head on S_own + ΔS_maps.
Counterfactual maps (program, g at mean): point-symmetrized ½[f(x) + f(−x)]; even in η ½[f(η, φ) + f(−η, φ)]; even in φ;
θ-averaged over 8 rotations ("radial").
- ΔS_c per direction, maps vs measured (g frozen): slope 1.06–1.08 (S29, S1, S50), 1.00 (S43), 0.98–1.01 (S30); Spearman of the
  six-direction ordering 0.94–1.00 (S29, S1, S50, S43), 0.77 (S30). Per-jet corr at δ = 0.06: S29 0.95, S1 0.95, S50 0.93–0.95,
  S43 0.995, S30 0.86–0.90. S-level odd/even (η, φ, diag), δ = 0.06, measured vs maps: S29 −0.153/−0.096/−0.177 vs
  −0.153/−0.104/−0.184; S1 −0.278/−0.144/−0.278 vs −0.260/−0.172/−0.297; S43 −0.164/−0.080/−0.152 vs −0.165/−0.078/−0.150.
  Point-symmetrized maps: |odd/even| ≤ 0.049 at δ 0.03 and ≤ 0.026 at δ 0.06 (largest for S1).
- Δlog m̂ (h5), δ = 0.03, measured g free: +η .091, −η .125, +φ .094, −φ .117, +diag .085, −diag .126 (per-direction s.e. 0.001;
  E6v, seed 17: .085/.117/.087/.110/.078/.118; Spearman 1.00). W→Z .26/.60/.29/.53/.21/.62 (E6v .28/.62/.32/.54/.24/.64).

| δ = 0.03, odd/even along η, φ, diag | η | φ | diag | odd part η, φ, diag |
|---|---|---|---|---|
| measured, g free | −0.155 | −0.109 | −0.195 | −0.0167, −0.0115, −0.0205 |
| measured, g frozen | −0.144 | −0.096 | −0.174 | −0.0152, −0.0100, −0.0180 |
| maps | −0.143 | −0.092 | −0.173 | −0.0147, −0.0093, −0.0174 |
| maps point-symmetrized | +0.003 | +0.008 | +0.007 | +0.0003, +0.0008, +0.0007 |
| maps even in η | +0.002 | −0.094 | −0.060 | |
| maps even in φ | −0.143 | +0.007 | −0.104 | |
| maps θ-averaged (radial) | +0.002 | +0.006 | +0.007 | |
Even part: measured 0.105–0.108, maps 0.101–0.102, radial maps 0.102. W→Z with radial maps .34/.35/.34/.33/.34/.35 (maps .21/.53/.24/.44/.18/.55).
δ = 0.06: odd/even measured g free −0.093/−0.053/−0.103, g frozen −0.072/−0.045/−0.082, maps −0.076/−0.048/−0.087, point-symmetrized
+0.002/−0.001/+0.001, radial +0.002/0.000/+0.001, even in η +0.002/−0.048/−0.036, even in φ −0.077/−0.001/−0.050.
- Odd part from the maps as a fraction of the measured odd part: of g frozen 97/93/97 % (δ 0.03) and 101/102/103 % (δ 0.06); of g
  free 88/81/85 % and 80/91/84 %. The remainder (9–20 % of the g-free odd part) is the context path (g frozen removes it).
- 45° rotation: measured ΔS/sd ≤ 0.004 (S29, S1, S50, S43, S30), maps ≤ 0.003; Δlog m̂ +0.0006 measured, +0.0003 maps; calls
  changed 3.8 % (fixed point, g free), 2.3 % (maps through the program head).

## Part 4 — one half-plane statement set to 0 for every particle (override of stage a, fixed-point network)
Per azimuth sector of the particle (12 × 30°): sector share of ΔS_c/sd(S_c), summed in |·| over the 19 live units. Own half =
sectors within 90° of the normal. With g frozen a particle off the half-plane cannot change (a_j = 0 there); with g free it can change
through P_j → g.

| ablated | normal | calls changed g free / frozen | Σ|share| frozen | own-half fraction frozen / free | own / opposite sector, g free | ΔS29/sd, ΔS1/sd |
|---|---|---|---|---|---|---|
| a29 | 0° | 31.0 / 27.4 % | 2.78 | 1.000 / 0.887 | 11.6 | −0.351, −0.368 |
| a6 | 51° | 31.2 / 29.0 % | 3.42 | 0.997 / 0.872 | 9.9 | −0.274, −0.327 |
| a40 | 131° | 29.4 / 25.1 % | 2.77 | 0.991 / 0.857 | 9.5 | −0.406, −0.410 |
| a57 | 144° | 9.7 / 8.4 % | 0.53 | 0.995 / 0.944 | 20.2 | 0, 0 |
| a3 | 161° | 12.2 / 11.1 % | 1.65 | 0.991 / 0.876 | 11.7 | 0, 0 |
| a43 | 204° | 24.4 / 22.4 % | 2.23 | 0.998 / 0.856 | 10.1 | −0.328, −0.410 |
| a30 | 248° | 8.8 / 7.0 % | 1.13 | 0.999 / 0.901 | 13.9 | 0, 0 |
| a36 | 270° | 26.4 / 25.2 % | 1.85 | 1.000 / 0.869 | 9.9 | −0.415, −0.418 |
| a46 | 290° | 15.7 / 13.8 % | 2.10 | 0.995 / 0.879 | 12.0 | 0, 0 |
- With g frozen the opposite sector changes by exactly 0 for all nine; the largest sector is the normal's sector or its neighbour for
  all nine. The program predicts the measured sector profiles (median per-unit corr ≥ 0.9998). S29 and S1 respond only to the five
  half-planes of s43.
- Controls: dead l1 unit a0: 0 change in every unit and 0 calls. pT ramp a56: sector CV 0.03 (all units; S29 0.030, S1 0.039),
  ΔS29/sd +0.670, ΔS1/sd +0.978, 53.0 % calls changed (g free), 30.6 % (g frozen).

## Reading (scoped to these measurements)
- The angular dependence of the 15 particle-level second-round units with half-plane input is, per particle, a level set of a
  weighted sum of the nine half-plane statements routed through five s units (s43, s60, s53, s32, s52). For units fed by one of
  these s units the on-contour is that s unit's level set exactly. The contours are nine-direction polygons with max/min 1.14–1.47,
  not circles. Random, rotated or permuted directions do not reproduce them, and the untrained network has no contour.
- The context g adds per-jet offsets. It adds no per-particle angular structure. For u30 it carries 32 % of the variance from the
  half-plane P units (jet level); for u31 and u58 it carries everything.
- Under rigid translation, Σ over particles of the per-particle maps reproduces the measured direction pattern of ΔS and of the h5
  mass read-out with g frozen (odd part 93–103 %). Point-symmetrizing or θ-averaging the maps sets the odd part to ≤ 0.0008 (from
  0.009–0.024) and leaves the even part unchanged (0.101–0.102). Symmetrizing in η alone removes only the η-odd part, and in φ alone
  only the φ-odd part. With g free, 9–20 % of the odd part comes through the context path.
- Against the plan's expectations:
  - Units with half-plane input draw 100 % of their half-plane input through s (13 units with a positive share, plus u17). The exceptions are u30 (38 %) and u31/u58 (0 %).
  - m = 0 carries 0.975–1.000 of the angular power.
  - The polygon reproduces r_c(θ) to ≤ 0.064 rms (60 of 78 contours ≤ 0.005).
  - The minimum of r_c lies at 230–245° (−η/−φ) for six units, but the m = 1 component of r_c points to 147–172° (−η side).
  - The ordering of the six directions is reproduced (Spearman 0.94–1.00), and the odd/even ratio of the mass read-out is 14/9/17 % (maps) vs 16/11/20 % (measured).
  - Off-half-plane change with g frozen is 0 by construction; own / opposite sector with g free is 9.5–20.
  - The dead unit gives 0 change; a56 gives a sector-uniform change (CV 0.03).

## Caveats
- For single-path units the polygon agreement is exact by construction once the path is identified. The content of the test is
  the path identification and the controls on the directions.
- Part 4 with g frozen is structural; its information is the size and shape per sector, the g-free spread (6–14 % to the off
  half) and the call changes.
- Maps, contours and counterfactuals are computed in the float + mask program with g at its sample mean. The fixed-point contours
  differ by ≤ 0.016 rad at slot 5 (u50 at 10 GeV: 0.08 rad) and ≤ 0.043 rad at slot 20. The counterfactual maps are interventions on
  the program, not on the fixed-point network.
- The direction of the odd term does not follow from the m = 1 component of r_c at one slot and pT; it is reproduced by the full
  per-particle sum (Part 3).
- The S30 maps over-predict the S-level odd/even along η (−0.157 vs −0.078 at δ 0.06) and order S30's directions with Spearman 0.77.
  S30's g-free response is twice its g-frozen response (context path, E5v).

## Suggested wording changes (for the ledger agent)
- A3: "the radial symmetry is built in round two from the half-planes" → the second-round on-contours are nine-direction polygons
  (max/min 1.14–1.47) formed as level sets of half-plane sums in five s units (s43: a29, a6, a36, a40, a43 → u1, u29, u56, u60;
  s60: a6, a43, a46, a3 → u52, u62, u43, u13, u30; s53: a29, a40, a30 → u51, u63; s32 → u17, u20; s52 → u50, u40). The context
  carries no per-particle angular input; for u30 it carries 32 % of the variance from the half-plane P units.
- A4: the odd term is reproduced by the per-particle maps (93–103 % of the g-frozen odd part; odd/even 14/9/17 % vs measured
  16/11/20 % at δ 0.03). It vanishes (≤ 0.0008) when the maps are point-symmetrized or θ-averaged, and 9–20 % of the g-free odd
  part passes through g.
