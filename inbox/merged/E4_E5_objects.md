# E4 / E5 — the context round and the second-round objects (agent report, 2026-10-16; verification pending)

Scripts: interp/e4_e5_jedi.py, interp/e5_causal_jedi.py. Results: results/jedi/e4_e5.json, e5_causal.json. Logs: logs/e4_e5.log, e5_causal.log.

## E4 (claim A2): 20,000 test jets (4,000/class, seed 0), acc 80.58 %. u pre-activation = own + ctx + quantizer residual + b3 (exact, error 0.0).
- Context reaches 9 of 19 live u units; between-jet SD(ctx)/SD(own): u30 0.56, u17 0.29, u20 0.18, u0 0.14, rest ≤ 0.08; u31, u58 have no own part.
- On/off flips vs mean context (real particles, pooled slots): u31 40.7 %, u58 17.1 %, u30 9.0 %, u0 5.3 %, u17 2.3 %, rest ≤ 0.7 %.
- g → mean: 80.58 → 74.02 % (14.90 % calls changed); g → 0: 43.15 % (51.4 %); g → another jet's g: 63.64 % (28.0 %).
- One g unit at a time: g49 10.38 %, g4 4.17 %, g55 2.20 %, g39 1.98 %, g5 1.57 %, g63 1.34 %, g32 1.09 %, rest < 1 %.
- Path: S0 alone at mean-context value 10.38 %, S30 4.17 %, S31 3.55 %, S58 1.92 %; the four together 14.44 %.
- What drives g: g49 → u0 only (weight −3.25); 72 % of its variance from P15 (ΣpT), rest from soft counts P32/P56; best Q log ΣpT R² 0.69, 3 terms 0.944.
  g4 (→ u30): three annulus counts, R² 0.963. g55, g39, g5 (→ u31): from the half-plane P units; best 3-term Q R² ≤ 0.14. g63 (→ u58): 0.46.

## E5 (claim A3): 19 S units change ≥ 0.5 % of calls when mean-ablated. f_c from the masked-float forward (zero-bit masks, no rounding/wrap;
80.14 % on this sample, 4.18 % calls differ; reproduces 80.68 % / 4.49 % on the E1 sample), g at its mean, over real particles in the unit's slots.
Azimuth adds ≤ 0.017 R² for every unit: f_c = f(pT, r), r = ΔR to the axis. Naming: greedy linear fits over 380 Q + ~290 hand-built quantities,
fit on half the jets, R² on the other half. Causal: 10,000 new jets (seed 1), measured with g recomputed and frozen, vs each formula's prediction.

| S | cost | slots | f_c (GeV, r) | best naming, held-out R² | causal |
|---|---|---|---|---|---|
| 29 | 40.9 % | all | ReLU(4.44 ln pT + 93.8 r − 21.2); on beyond r = 0.23 / 0.12 / 0.01 at pT 1 / 10 / 100 | ΣpT·r², Σr², N(pT<2, r>0.1): 0.976; hinge sum 0.938; Q m²/ΣpT² 0.877 | r crossing +0.482 (pred +0.469), control +0.008 (0); random r: corr 0.97, slope 1.03; pT add +0.045 (pred +0.090) |
| 1 | 20.2 % | 1–63 | ReLU(89.3 r − 22.9 + 0.55 ln pT); r > 0.24–0.26, nearly pT-flat | Σ_{slots≥1}(r − 0.2)₊: 0.947; 3 terms 0.965 | r crossing +0.457 (+0.446), control +0.004; random r corr 0.94, slope 1.04; 1 GeV particle at r = 0.32: +0.414 (+0.357) |
| 0 | 12.0 % | hardest 14 | 0.32 pT + 1.40, no r term | ΣpT 0.934; log ΣpT 0.929; hinge sum 0.388 (context carries 61 % of its variance) | r moves 0.000; +10 % of leading pT on a slot 2–13 particle: +1.30 (g frozen +0.48, hinge +0.48, ΣpT fit +1.06); all pT × 1.1: +5.43 (frozen +1.71); 1 GeV added: +0.059 (frozen 0) |
| 50 | 11.9 % | 0–38 | ReLU(7.07 ln pT + 85.3 r − 34.7), R² 0.71 | ΣpT·r² over slots 0–38: 0.938 | — |
| 19 | 11.1 % | 13–63 | ReLU(2.86 − 1.39 pT): pT < 2.06; 2.86 per empty slot | N(pT > 1): 0.991; 3 terms 0.999 | — |
| 30 | 8.0 % | all | ReLU(3.47 − 21.0 r): r < 0.165; 3.38 per empty slot | Σr: 0.957; hinge sum 0.675 (ctx share 0.24) | — |
| 43 | 7.8 % | hardest 14 | ReLU(0.105 pT + 74.3 r − 11.3) | Q 3 terms (m_top15, …): 0.932 | — |
| 60 | 7.0 % | 0–25 | ReLU(5.14 ln pT − 142 r − 6.8); r < 0.04 at 10 GeV, < 0.12 at 100 GeV | 3 terms 0.969 | — |
Other 11: S13, S56, S17, S40, S51, S52, S63, S62, S20 each a hinge in (pT or ln pT, r) with a slot window (S20: slots 0–2 only), R² 0.88–0.996;
S31, S58 functions of g only (linear in g, R² 0.98, 0.85; best Q 3-term 0.21, 0.53).

A2: g → mean changes 14.90 % of calls; 14.44 % from S0, S30, S31, S58 alone; in the other 15 objects SD(ctx) ≤ 0.29 SD(own).
A3: 17 of 19 objects = a two-variable hinge in (pT or ln pT, r) + fixed slot window, held-out R² 0.67–0.996 (S0 0.39, context-dominated);
radial edits on S29, S1 match the hinge prediction (slope 1.03–1.04); S0's measured change is 2.5–3.2× the hinge prediction, through g49.

Issues: (1) e1_e3_jedi.py's E3 identity summed all 64 slots; with the qs zero-bit mask applied max error 4.63 → 0.63 (rel 0.057): the remainder is
per-slot rounding. (2) l3 input differs from mask·(s + g) in 5.9 % of entries by up to 32 (WRAP overflow active; u52 residual SD 9.8 vs own SD 42.6).
(3) S29's hinge over-predicts the pT dependence for hard particles. (4) Δη is rounded to steps up to 0.18 rad in slots ≥ 59.
