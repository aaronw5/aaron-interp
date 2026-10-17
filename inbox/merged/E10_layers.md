# E10 — readability of jet observables by layer (agent report, 2026-10-16; verification pending)

Sample 4,000 test jets/class (20,000), fixed 50/50 split, rng 0; 45 observables with a depth label (8 slot-order quantities *);
targets clipped to 0.5–99.5 %. Readouts on live units at P, g, S, h4, h5, z: RidgeCV (standardised) and an MLP (64 hidden, seed 0).
Controls: untrained network (Jedi(untrained=True), seed 0; bit widths and masks kept; acc 0.200; live units P/g/S/h4/h5/z = 9/6/1/7/17/15
vs trained 12/11/19/55/32/16), ridge on raw flattened input, class means only, within-class R². Trained acc 0.806.

Depth table — first stage at R² ≥ 0.9 / ≥ 0.7 (– never); linear R² at P/S/h5; u = untrained R² at the 0.7 stage:
d0: sum_pt P/P .98/.96/.87 u.01; n_particles S/P .89/.99/.94 u.68; n_pt_above_1 P/P .93/.99/.94 u.64; n_pt_above_5 P/P .97/.94/.87 u.46;
    n_pt_above_50 –/– .31/.56/.33; n_dr_0_0p05 S/S .34/.90/.80 u0; n_dr_0p1_0p2 –/S .49/.86/.75 u0; n_dr_0p4_up –/– .05/.26/.11;
    ΣpTΔR S/S .64/.99/.97 u0; ΣpTΔR² S/P .74/.99/.98 u.72; ΣΔR P/P 1.00/.99/.95 u.97; sum_pt_top5* P/P .96/.96/.79 u.32; pt_0* S/S .65/.90/.57 u0; pt_1* –/– .49/.57/.40
d1: girth S/S .66/.99/.97 u0; girth2 S/P .75/.98/.98 u.73; mass S/P .74/.97/.99 u.68; m² S/P .72/.98/.98 u.68; e2 S/S .49/.95/.96 u0;
    pt_dispersion S/P .78/.95/.70 u.28; LHA S/S .57/.98/.97 u0; pt_entropy P/P .93/.98/.84 u.44; psi_0p3 –/S .33/.82/.63 u0; z_dr_0p2_0p4 –/S .58/.89/.84 u0;
    z_0* –/S .63/.88/.57 u0; z_top5* P/P .95/.95/.80 u.37; pt1/pt0* –/– .14/.35/.15; dr01* –/– .12/.40/.36
d2: e3 –/S .54/.82/.87 u0; C2 –/S .48/.71/.76 u0; D2 –/h4 .21/.63/.79 u0; N2 –/S .19/.76/.78 u0; C3 –/– .08/.43/.37; planar_flow –/h4 .08/.67/.67 u0;
    eccentricity –/– .05/.56/.56; m/ΣpT S/P .75/.97/.98 u.71
d3: tau1 S/S .66/.99/.97 u0; tau2 –/S .66/.78/.81 u0; tau21 –/h4 .09/.68/.74 u0; tau32 –/– .24/.36/.44; sd_mass h4/S .38/.88/.88 u0; sd_zg –/– .06/.28/.28;
    sd_rg –/S .44/.81/.83 u0; sj2_zsoft –/– .16/.55/.54; sj2_dr –/h4 .41/.57/.70 u0

Spearman depth vs first stage (never = 6): at 0.9 ρ 0.51 (p 4e-4; 0.56 without slot quantities); at 0.7 ρ 0.38 (p 0.009; 0.49).
Median first stage at 0.9: S for d0, d1; never for d2 (7/8), d3 (7/9). At 0.7: d0 P–S, d1 S, d2 S–h4, d3 h4. Linear R² peaks at h4 for 8/8 d2
and 8/9 d3; d0 peaks at S (6), P (4), h4 (4). Median gain S → h4: d0 −0.01, d1 +0.01, d2 +0.07, d3 +0.07. MLP at S minus best linear at any
stage: +0.01…+0.04. Peak → z loses 0.09–0.22 (median). g is a function of P only, yet raises linear R² for planar_flow 0.08 → 0.47,
eccentricity 0.05 → 0.38 (MLP on P: 0.70, 0.60). Within-class R² (median d0/d1/d2/d3): P 0.52/0.09/−0.24/0.04; S 0.93/0.91/0.48/0.54.

Causal (2,000 held-out jets; control = whole jet rotated 90°):
- n_particles (d0): 5 softest particles split collinearly (Δn = +5; ΣpT, mass, girth unchanged): probe at S +2.60 ± 0.94 (slope 0.52;
  P 0.39, h4 0.58); control −0.004 (mean |Δ| 0.34); 5.1 % calls changed.
- girth (d1): particles with pT < 20 GeV moved to ΔR × 1.5 (cap 0.4), ΣpT and n fixed: Δgirth +0.0101 ± 0.0076; probe at S +0.0099 ± 0.0064
  (slope 0.91, corr 0.95; P 1.65, h4 0.80, z 1.17); control mean |Δ| 0.0018; side effects: mass +18 GeV, 2.8 particles/jet at ΔR ≥ 0.4; 37 % calls changed.
- D2 at fixed mass (d2): best of 9 moves of slot 1–3 about the other particles' centroid; mass change < 1e-6 GeV; ΔD2 +0.14 ± 0.76 (data sd 3.08),
  τ21 +0.14; probe at h4 (R² 0.82) −0.04 ± 0.19 (slope 0.057, corr 0.28); control mean |Δ| 0.30; 5.1 % calls changed. Fixed-mass D2 probe at h4:
  R² 0.64 in data (mass bins alone 0.44), edit slope 0.08.
Figure results/jedi/e10_layers.png. Caveats: untrained control nearly empty from S on (S has 1 live unit); untrained P has no pT weight
(ΣpT R² 0.01, ΣΔR 0.97); pt_1 1.00 from raw input but ≤ 0.57 in the network; 99.9 % of particles at ΔR < 0.44 so outward edits leave the data range.
Files: interp/e10_layers_jedi.py (modes read / causal / plot), results/jedi/e10_layers.json, e10_causal.json, logs/e10_read.log, e10_causal.log.
