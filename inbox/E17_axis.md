# E17: JEDI-linear and the jet-axis convention (2026-10-17)

- **Job:** `anrunw-ai-e17`, 647 s.
- **Files:** `interp/e17_axis_jedi.py`; `results/jedi/e17_axis.json`, `e17_axis.png`, `e17_axis_bins.png`.
- **Sample:** 20,000 jets (seed 41). Stored axis: accuracy 80.58 %, W→Z .021, Z→W .066, |M1|²/(M0M2) .00035.
- **Edit:** only the real particles' Δη, Δφ change (checked); max |Δlog m| 7e-12.
- **WTA axis:** P1v's code. It equals the hardest particle in 65.8 % of jets (ParT: 58 %).
- **Read-out:** ridge from z to log m, fitted on the train half; test-half R² .941.

| axis | shift median (p90) [rad] | \|M1\|²/(M0M2) | acc % | changed %, all (g/q/W/Z/t) | W→Z | Z→W | Δlog m̂ / ½Δlog M0M2 |
|---|---|---|---|---|---|---|---|
| centroid | .0015 (.0059) | 0 | 80.7 | 1.4 (1.4/1.4/0.8/2.0/1.7) | .021 | .065 | −.0004 / −.0004 |
| WTA | .040 (.098) | .176 | 59.3 | 30.3 (15/11/68/44/14) | .563 | .025 | .094 / .121 |
| hardest | .041 (.111) | .185 | 58.4 | 31.5 (16/12/69/46/15) | .490 | .025 | .115 / .140 |

**Δlog m̂ / prediction** (WTA, hardest):
- JEDI: 0.77, 0.82.
- ParT_full: 0.96, 1.06. ParT_kin: 0.91, 0.93.

**Class flow, stored-axis call → new-axis call** (each row counts into g q W Z t):
- WTA: g 3422 230 62 91 417; q 42 3726 128 23 70; W 9 529 1092 2485 26; Z 53 1544 7 1994 80; t 58 165 2 44 3701.
- Hardest: largest off-diagonal cells are W→Z 2172, Z→q 1511, W→q 800, g→t 439, g→q 257, Z→t 248.
- Centroid: every off-diagonal cell ≤ 43.

**WTA axis, compared with ParT** (shift; ½Δlog M0M2; calls changed):
- JEDI: .040 rad; .121; 30.3 %.
- ParT_full: .095 rad; .138; 11.7 %.
- ParT_kin: .095 rad; .138; 34.1 %.

**Calls changed by shift bin, WTA axis** (W/Z jets vs g/q/t jets):
- 0.02–0.04 rad: 31 % vs 17 %.
- 0.04–0.07 rad: 71 % vs 34 %.
- 0.07–0.10 rad: 96 % vs 22 %.

**Δlog m̂ against the prediction:**
- Shifts under 0.07 rad: below the prediction.
- Shifts above 0.07 rad: at or above it.

**Input range:**
- Quantizer-range overflow: 0.09 % of jets (WTA), 0.01 % (hardest).
- Particles past the training 99 % angle: 2.0 % stored, 4.1 % WTA, 4.5 % hardest.
