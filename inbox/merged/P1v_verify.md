# P1v — verification of P1 (ParT translation test), 2026-10-17

Jobs `anrunw-ai-p1v-quick` (50 jets/class, smoke) and `anrunw-ai-p1v-run` (GTX 1080 Ti, 176 s), `python -m interp.p1v_verify_part`.
Script `interp/p1v_verify_part.py`, figure script `interp/p1v_plot.py` (helper pod). Results `results/part/p1v_verify.json`,
`results/part/p1v_verify.png`, log `logs/p1v_cluster.log` (copies of /j-jepa-vol/aaron-interp/{results/part,logs/p1v-run.log}).
Sample: P1's rows 3,000–4,000 (10,000 jets), rebuilt δ = 0 baseline; read-out = P1's ridge (same seed, split, alphas; R² 0.881 / 0.920 reproduced).

## Check 1 — edit hygiene (φ + 0.06, inputs from `network.inputs`, pairs from weaver's `pairwise_lv_fts`)
- Per-particle, max |Δ| on real slots: dphi +0.0600 on every particle; deltaR raw 0.060 (mean +0.0080; fed ×4: 0.24); pt_log,
  logptrel ≤ 9.5e-7 (float32 rounding of pT); e_log, logerel, deta, charge, 5 PID flags, d0, d0err, dz, dzerr: 0 exactly.
  Mask and trimmed length identical; pz, E bit-identical; jet_pt rel. 3e-7.
- Pair features, 17.9 M real-real pairs: lnz max |Δ| 4.8e-7; lnkT, lnΔR 2.3e-3 (3e-7 of pairs > 1e-3); lnm2 max 17.7, 2.1 % of
  pairs > 1e-3, 0.43 % > 1e-2. Float32-vs-float64 of the *unedited* pair features: lnkT/lnΔR 3.1e-3, lnm2 max 17.4, 4.0 % > 1e-3.
  The pair features change by the float32 noise of their own computation (lnm2: E² − p² cancellation), not to float32 epsilon.

## Check 2 — which input carries the response (Δlog m̂ on the test half; W→Z; calls changed)
| | ParT_full δ 0.03 / 0.06 / 0.09 / −0.06 | ParT_kin δ 0.03 / 0.06 / 0.09 / −0.06 |
|---|---|---|
| (a) full edit | .014 / .052 / .110 / .049; W→Z .149/.310/.490/.309; 2.8/8.1/17.3/8.4 % | .024 / .095 / .193 / .102; W→Z .210/.373/.442/.355; 5.8/19.6/39.9/20.0 % |
| (b) deltaR unedited (only dphi edited) | .001 / .003 / .005 / −.001; W→Z .118/.119/.124/.116; 0.8/1.3/1.8/1.3 % | .015 / .062 / .127 / .066; W→Z .183/.230/.250/.230; 6.5/17.5/32.6/17.4 % |
| (c) dphi unedited (only deltaR edited) | .014 / .052 / .111 / .052; W→Z .151/.312/.488/.320; 2.4/7.9/17.1/8.6 % | .008 / .033 / .075 / .035; W→Z .197/.301/.437/.315; 4.9/11.8/22.3/12.4 % |
| (d) both unedited (pair route only) | .0000 at all δ; W→Z .117; 0.01–0.03 %; max |ΔL| ≤ 1.9 | ≤ .0002; W→Z .161–.163; 0.53–0.62 %; max |ΔL| ≤ 10.3 |
| (f) 4-vectors unedited | = (a) to 3 digits | = (a) to 3 digits |
| prediction ½Δlog M0M2 | .021 / .073 / .143 / .073 | same |
Baselines: W→Z .117 (full), .162 (kin). (a) reproduces P1 exactly (ratio Δlog m̂ / prediction: full 0.66, 0.71, 0.77; kin 1.10, 1.30,
1.35). Index 97 on W jets, δ 0.06, full: (a) −0.353, (b) −0.014, (c) −0.354 s.d. Kin: (b) + (c) = .095 = (a) in Δlog m̂.
−0.06 gives the same sign and size as +0.06 in every variant.

## Check 3 — fresh rows 6,000–7,000 (10,000 jets), φ + 0.06
ParT_full: W→Z .122 → .335, calls changed 8.3 %, Δlog m̂ .054 (refit read-out, R² 0.890) / .051 (P1's read-out, R² 0.892) vs
prediction .074 (ratio 0.73 / 0.69); index 97 on W −0.345. ParT_kin: W→Z .174 → .373, 19.7 %, Δlog m̂ .092 / .097 (ratio 1.25 / 1.29).

## Check 4 — axis convention (4-vectors unchanged; only deta, dphi, deltaR recomputed)
| axis | displacement median (W, QCD), p90 [rad] | \|M1\|²/(M0M2) median | pred | full: acc, changed, W→Z, Δlog m̂ | kin: acc, changed, W→Z, Δlog m̂ |
|---|---|---|---|---|---|
| stored, through the same code | 0 | 0.0020 | 0 | 85.9, 0 %, .117, 0 | 74.5, 0 %, .162, 0 |
| E-scheme from the stored particles | 3e-4 (2e-4, 3e-4), 1.6e-3 | 0.0024 | .0002 | 85.9, 0.08 %, .117, .0002 | 74.5, 0.18 %, .163, .0001 |
| hardest particle | 0.102 (0.080, 0.034), 0.226 | 0.244 | .176 | 78.9, 14.0 %, .333, .186 | 53.0, 38.0 %, .351, .163 |
| WTA (C/A, WTA pT recombination) | 0.095 (0.075, 0.030), 0.187 | 0.219 | .138 | 80.9, 11.7 %, .340, .133 | 56.0, 34.1 %, .377, .125 |
WTA axis = hardest particle in 58 % of jets; deta sign convention (sign of axis η) flips for 255 jets (2.6 %) with the WTA axis.

## Check 5 — script review
Padding unchanged (mask, padded deta/dphi = 0, all other branches bit-identical); ridge fitted once on unedited z (tr half), applied
to the te half of edited and unedited z, same 128 features; baseline = `net.run(build(δ = 0))` for every row; `network.run` equals the
forward on precomputed inputs (max |ΔL| 5.7e-6). No bug affecting P1's numbers. Notes: ParT_kin's alpha (1e-4) is at the grid's
lower edge; the pair route alone changes 0.5–0.6 % of ParT_kin calls (float32 noise), which is part of P1's small-δ kin rows.

## Verdict
Reproduced: yes. Refuted: no. Confidence 0.9.
Corrected reading: the numbers stand (full ratio 0.66–0.77, not 0.7–0.8). The response enters through the per-particle axis-relative
inputs only: in ParT_full entirely through deltaR (dphi fed unedited changes nothing; deltaR fed unedited removes 92–100 % of Δlog m̂
and 98–100 % of the W→Z change); in ParT_kin dphi carries 0.65 and deltaR 0.35 of Δlog m̂, additively. The pair features are invariant up to
float32 noise (lnm2), which changes 0.01 % (full) / 0.6 % (kin) of calls. The edit is equivalent to displacing the axis: using the
WTA axis instead of the stored axis (median 0.095 rad) moves W→Z to .340 (full) / .377 (kin) and changes 12 % / 34 % of calls.
Figure: `results/part/p1v_verify.png` (Δlog m̂ and W→Z vs δ per variant).
