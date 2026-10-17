# E16 — neuron atlas of JEDI-linear 3-feature n64 (agent report, 2026-10-17; claim A11)

Scripts: interp/e16_atlas.py (modes run / cards / summary), interp/e16_plot.py. Results: results/jedi/e16/ (atlas_<stage>.json, images_<stage>.npz,
card_<unit>.png ×157, atlas_summary.json/.png). Cluster jobs e16-a (a,P,g), e16-b (S,h4), e16-c (h5,z), e16-final, e16-cards; 1 CPU / 2 Gi, 5–12 min each.

## Procedure
20,000 test jets (4,000/class, seed 21), exact fixed-point network, accuracy 80.97 %. Live = mean ablation changes ≥ 0.1 % of calls:
a 13, P 12, g 11, S 19, h4 54, h5 32, z 16 (144 units, a as reference). Per unit: per-class histogram; rank bins over on-jets (top 1 %, 90–99 %,
50–90 %; z also lowest 1 %) with mean pT and count images (32×32, ±0.4 rad; network frame and principal-axis-aligned), class mix, 10 observable
medians; description = best 1-D response curve over 539 candidates (Q library + E11 rank-window and half-plane sums; 20 quantile bins, interpolated,
fit/test halves) and best pair; activation-weighted k-means (k 2–4 by silhouette, k = 1 if < 0.25) on the 10 observables of the on-jets; upstream =
|weight × spread| of quantized inputs (S: per-particle u table in (pT, r) + rank window; P: the l1 cut); downstream = next-stage weights and
Δlogit for unit at p90 vs p10 (2,000 jets); steering = unit set to p99 on ≤ 4,000 off jets (z also p1 on on-jets), control = random live unit, same
stage, same procedure. Checks: S→logits tail and per-particle pass reproduce the network exactly (max |Δ| 0).

## Tally (thresholds: unexplained = best ≤ 2-observable held-out R² < 0.5; one main job = one cluster ≥ 80 % of activation mass; related = every
cluster ≥ 10 % of mass has its own best observable at |Spearman ρ| ≥ 0.7 with the unit's; else unrelated)
| stage | one | related | unrelated | unexplained |
|---|---|---|---|---|
| P | 0 | 12 | 0 | 0 |
| g | 3 | 5 | 3 | 0 |
| S | 1 | 17 | 0 | 1 |
| h4 | 13 | 30 | 7 | 4 |
| h5 | 10 | 12 | 8 | 2 |
| z | 4 | 6 | 6 | 0 |
| all 144 | 31 (21.5 %) | 82 (56.9 %) | 24 (16.7 %) | 7 (4.9 %) |
Cost-weighted: 13.6 / 72.5 / 12.5 / 1.4 %. k ≥ 2 forced: 14 / 90 / 33 / 7. k = 2 was chosen for 114 units; the second cluster is mostly the
higher-mass / higher-girth jets, over which the unit tracks the same observable. Description R² (1 / 2 obs) median 0.81 / 0.86; ≥ 0.7 for 110 units.
Steering: median calls changed 9.9 % vs control 9.8 % (unit > control in 69/144); share of changed calls going to the top-1 % bin's majority
class: median 0.15 overall, 0.61 for z.

## Ten costliest units (pattern; R² 1/2; category; offshoot; steering p99 → % calls changed, main destination, control)
- S29 40.3 %: ΣpT·r² (ranks 8–63) > 8.8 GeV·rad²; 0.94/0.96; related, 47 % cluster higher mass/girth; 98.6 % → t 97 % (S40 68.3 %).
- z1 21.3 %: ΣpT > 1,119 GeV, higher multiplicity; 0.49/0.76; top 1 % 99 % g; unrelated (46 % cluster tracks z(50 hardest)); negative pole W 60 %,
  pT0/ΣpT 0.37, τ21 0.24; 31.5 % → g 100 % (z10 24.1 %); p1 on on-jets 24.9 % → q 89 %.
- S1 20.2 %: Σ(r−0.2)₊ (ranks 1–63) > 1.50 rad; 0.95; related; 76.9 % → q 88 % (S51 12.5 %).
- h4:49 19.0 %: √(M0·M2) < 64.2 GeV; 0.97; top 1 % 81 % q; related; 18.8 % → g 45 % (h4:3 13.0 %).
- h5:26 15.4 %: ΣpT·r² > 10.0 GeV·rad²; 0.97/0.98; related; 32.3 % → t 71 % (h5:20 14.2 %).
- z0 14.9 %: ΣpT·r² < 7.18 GeV·rad², peak 6.23 (√(M0·M2) ≈ 80 GeV at ΣpT 1,026); 0.93/0.94; top 1 % 97 % W, mass 79 GeV; related; negative
  pole t 82 %, ΔR01 0.28 (side alone: ΣpT·r² > 10, R² 0.95); 17.2 % → W 68 % (z7 61.2 %); p1: 26.4 % → Z 99 %.
- S0 12.0 %: ΣpT > 1,087 GeV; 0.97; related (k 4); 69.2 % → g 96 % (S63 9.7 %).
- S50 11.6 %: ΣpT·r² (ranks 1–63) > 18.5 GeV·rad²; 0.94/0.96; related; 86.1 % → W 82 % (S52 11.3 %).
- S19 11.0 %: N(pT > 1 GeV, ranks 13–63) < 26.5; 0.99; related; 54.8 % → t 73 % (S51 12.5 %).
- g49 10.4 %: ΣpT < 1,087 GeV, lower n(pT > 5); 0.73/0.97; related; read only by S0 (w −3.25); 38.6 % → t 58 % (g63 5.0 %).

## Against earlier claims
- A1/E2 count: the 13 live cuts are 4 pT ramps (a32 < 11, a56 < 31, a21 < 93 GeV, a15 always on) + 9 half-planes; E2's "6 pT ramps" counted
  a4, a50 (constant on data).
- A2 reproduced (g49 → S0 only). A3: no S unit's best observable is directional; S31 (R² 0.23, unexplained) and S58 (0.51) are the g-only objects.
- A4: z0 (peak ≈ 80 GeV), z7 (√(M0·M2) 86.5–102, peak 91.6), z14 (≈ 86–101, peak 92) agree with E6 within 5 GeV; mass terms first at h4.
  z10 differs from E6's "top window, peak 135–148 GeV": on in 74 % of jets, best ΣpT·r (ranks 1–63) > 38.9 GeV·rad (R² 0.70; jet mass 0.64),
  top 1 % 95 % t at median 160 GeV. z9 differs from E11's top term (√(M0·M2) > 143): on for √(M0·M2) < 77.9 or 120–163 GeV, top 1 % g/q.

Representative cards: results/jedi/e16/card_S29.png, card_z0.png, card_z10.png, card_P29.png; summary results/jedi/e16/atlas_summary.png.
