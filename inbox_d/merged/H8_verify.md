# H8 verify — EFP/ADO guided search vs causal edits (JEDI-linear, ParT): fresh-sample rerun and an axis-moment positive control

Script `interp/h8_verify.py` (wraps `interp/h8_efp_ado.py` unchanged; adds the control). Jobs: `anrunw-ai-d-h8-verifyg` (GPU, ParT
forward passes, 4 min) and `anrunw-ai-d-h8-verify` (1 CPU / 2 GiB; JEDI forward passes, observables, searches, control; 56 min, peak
1,551 MiB). Results: `results/physics/h8_verify.json` (control and rerun summary), `results/physics/h8_verify_work/h8_efp.json`,
`h8_jedi_net.json`, `h8_jc_net.json` (large npz on the volume only). Logs: `logs/d-h8-verify.log`, `logs/d-h8-verifyg.log`.

## Conclusion
On new jets, the H8 numbers come back at the same size: shape-based stand-ins rank W/QCD and top/QCD jets like each network in
92–97 % of signal–background pairs and change no calls when the particle coordinates are recomputed about another axis, while the
networks change up to 38 %. The stand-in's zero response is a property of the observable list, not of the networks. Replacing jet
mass by the same mass written as √(ΣpT · ΣpT ΔR²) about the axis the network is given (the axis-moment mass from A4, equal to the
jet mass on unedited jets) leaves the ranking agreement unchanged and makes JEDI-linear's Z/W stand-in change 37.7 % of calls under
the winner-take-all re-axis against the network's 37.6 %. So most of the "gap" for JEDI-linear is one known variable, the
un-recentred axis-moment mass; for ParT the same substitution over-predicts the response, and the response to an exact collinear
split (0.1–3.4 % of calls) is not explained by either stand-in.

## Code check (`interp/h8_efp_ado.py`)
- Rows: hls4ml via `jedi_core.sample` indices (random per class); JetClass via `h1_irc_panel.jc_sample` (random rows per class,
  sorted). No first-k subsampling. Build η = jet_η + sign(jet_η)·Δη (`part_core.build`); unedited and edited jets use the same build.
- Observables are computed from the arrays the network was run on (`x_<cond>` for JEDI; `kin` of the rebuilt jets for ParT), real
  particles only (pT > 0). Padded slots are excluded from EFPs, τ_N, m_SD and from the control's moments.
- Edits: WTA re-axis keeps four-vectors (invariant mass unchanged; checked by the diag); translation shifts Δφ of real particles only;
  the boost keeps m and τ ratios at small angles and changes ΣpT; the split is exact except in the 22.4 % of JEDI jets that fill 64
  slots (log: max |Δp3|/p3 = 5.2e-3, from dropping the softest particle). No conservation error found.
- Working point: binary call = (log Σ p_sig − log Σ p_bkg) > 0. For JEDI W/QCD and top/QCD this is not a fixed signal efficiency
  (31–33 % of the task jets are called signal); % calls changed depends on this threshold. Not an error, but not a published working point.
- Statistics: when the surrogate changes 0 % of calls, the exact McNemar test reduces to a test that the network changes > 0 calls.
  The 33/36 BH-significant tests therefore restate D4/A4 (the networks are not axis- or translation-invariant); they carry no
  information about the surrogate. ADO intervals resample jets but not the search; the chosen sets beyond the first observable are
  not stable across samples (below).
- No bug found that changes a number.

## Fresh-sample rerun (H8 pipeline unchanged)
Sample: hls4ml `jedi_core.sample(per_class=1500, seed 811)`, N 7,500; JetClass 1,000 random rows per class (seed 812), N 10,000
(4,000 QCD/W/Z/top for the surrogates). Test half: about 2,250 (JEDI W or top vs QCD), 1,500 (JEDI Z/W), 1,000 (JetClass) jets per task.

Unedited accuracy on these samples: JEDI-linear 81.32 % (H8 80.86); 5-class calls changed by WTA 30.04 % (D4 30.3). ParT_full
86.37 % (H8 86.05); ParT_kin 74.80 % (H8 73.71). Redrawn: JEDI 20.0 % (constant call), ParT_full 10.03 % (one class; 0.05 % on
the 4-class rows).

| model | task | chosen (fresh) | ADO fresh [95 % CI] | ADO H8 | WTA net / sur % | translate | boost | split |
|---|---|---|---|---|---|---|---|---|
| JEDI | W/QCD | τ21, β2#15, β2#93 | 0.944 [0.934, 0.954] | 0.933 | 17.1 / 0.0 | 41.4 / 0.0 | 8.8 / 8.2 | 1.2 / 0.1 |
| JEDI | top/QCD | β0.5#58, β0.5#16, m_SD | 0.961 [0.956, 0.967] | 0.956 | 6.9 / 0.0 | 10.3 / 0.0 | 9.0 / 6.0 | 1.0 / 0.4 |
| JEDI | Z/W | m, β0.5#48 | 0.977 [0.971, 0.982] | 0.977 | 37.6 / 0.0 | 54.0 / 0.0 | 18.0 / 1.9 | 1.7 / 0.0 |
| ParT_full | W/QCD | τ21, β1#4, τ32, β1#14, β2#14, β1#34, m | 0.945 [0.933, 0.957] | 0.896 | 3.4 / 0.0 | 3.5 / 0.0 | 1.7 / 4.1 | 0.5 / 0.0 |
| ParT_full | top/QCD | β1#94 (e2²·e3), τ32 | 0.972 [0.963, 0.980] | 0.966 | 2.0 / 0.0 | 0.7 / 0.0 | 0.6 / 2.9 | 0.1 / 0.0 |
| ParT_full | Z/W | m, β2#15 | 0.709 [0.683, 0.731] | 0.663 | 17.1 / 0.0 | 16.3 / 0.0 | 6.7 / 13.3 | 3.4 / 0.0 |
| ParT_kin | W/QCD | τ21, β1#4, τ32, β2#12, β2#14, β1#14, β2#35 | 0.923 [0.907, 0.936] | 0.915 | 4.9 / 0.0 | 6.0 / 0.0 | 2.1 / 9.6 | 0.6 / 0.0 |
| ParT_kin | top/QCD | β1#94, τ32 | 0.973 [0.965, 0.981] | 0.969 | 5.1 / 0.0 | 3.2 / 0.0 | 0.5 / 2.4 | 0.4 / 0.0 |
| ParT_kin | Z/W | m, β2#14, β0.5#1, τ21 | 0.764 [0.743, 0.784] | 0.754 | 27.2 / 0.0 | 20.3 / 0.0 | 4.5 / 8.2 | 2.2 / 0.0 |

Reproduced: ADO 0.92–0.98 for W/QCD and top/QCD and JEDI Z/W; ParT Z/W 0.71/0.76; τ21 first for W/QCD, m first for Z/W, τ32 in
every ParT top set; surrogates 0.0 % under WTA and translation; boost not a zero control; redrawn JEDI constant call.
Not reproduced: JEDI W/QCD stops at 3 observables (H8: 8, at the cap); ParT top/QCD picks EFP #94 (the product e2·e2·e3) instead
of the triangle e3 alone (the triangle enters as a factor); ParT_full W/QCD ADO 0.945 vs 0.896 (intervals do not overlap).

## Missing control: is the surrogate's 0 % response fixed by the pool? (positive control)
The H8 pool is translation-invariant and axis-free by construction, so its 0 % change under WTA and translation is known before
the run. A positive control asks whether an equally good surrogate that contains an axis-relative variable would reproduce the
networks' responses.

- Variable: m_A = √(M0·M2), M0 = Σ pT_i, M2 = Σ pT_i (Δη_i² + Δφ_i²) about the axis the network is given (A4's mass variable for
  JEDI-linear, un-recentred). For massless particles at small angle m² ≈ M0·M2 − |M1|², M1 = Σ pT_i (Δη_i, Δφ_i). On unedited jets
  |M1|²/(M0·M2) has median 0.0004 (hls4ml) / 0.0017 (JetClass), and corr(log m, log m_A) = 0.99999 / 0.9991. WTA raises the median
  of |M1|²/(M0·M2) to 0.17 / 0.18, translation to 0.36 / 0.13 (median ½Δlog M0M2: WTA 0.095 / 0.099, translation 0.22 / 0.068).
- 'swap' surrogate: the fresh-sample H8 set with m replaced by m_A (m_A appended when m was not chosen); same HGB, same fit/test halves.
- 'axis' search: the guided search rerun on irc + m_A + ΣpT.
- H0 (per model × task × edit): % calls changed, network = swap surrogate (exact McNemar on discordant jets, BH over 36 tests).
  Competing predictions: if the network's axis response goes through m_A, the swap surrogate changes a similar share of calls and the
  per-jet score changes correlate (r → 1); if it goes through another axis-relative quantity, the swap surrogate stays near 0 or r ≈ 0.

| model | task | swap ADO (irc ADO) | WTA net / swap %, diff [95 % CI], r(Δscore), BH q | translate net / swap %, diff CI, r, q | split net / swap % |
|---|---|---|---|---|---|
| JEDI | W/QCD | 0.968 (0.944) | 17.1 / 22.5, [−6.6, −4.0], 0.90, 2e-15 | 41.4 / 33.2, [6.7, 9.7], 0.87, 4e-25 | 1.2 / 0.1 |
| JEDI | top/QCD | 0.964 (0.961) | 6.9 / 3.2, [2.5, 4.9], 0.43, 2e-9 | 10.3 / 2.7, [6.4, 9.1], 0.08, 1e-25 | 1.0 / 0.4 |
| JEDI | Z/W | 0.977 (0.977) | 37.6 / 37.7, [−1.7, 1.5], 0.88, 0.96 | 54.0 / 51.8, [0.9, 3.6], 0.91, 0.003 | 1.7 / 0.0 |
| ParT_full | W/QCD | 0.946 (0.945) | 3.4 / 5.3, [−3.7, −0.2], 0.62, 0.06 | 3.5 / 5.7, [−3.9, −0.5], 0.53, 0.03 | 0.5 / 0.0 |
| ParT_full | top/QCD | 0.983 (0.972) | 2.0 / 4.8, [−4.4, −1.4], 0.51, 6e-4 | 0.7 / 1.5, [−1.6, 0.2], 0.32, 0.23 | 0.1 / 0.0 |
| ParT_full | Z/W | 0.695 (0.709) | 17.1 / 28.3, [−14.2, −7.8], 0.46, 3e-11 | 16.3 / 28.1, [−15.4, −8.3], 0.31, 9e-11 | 3.4 / 0.0 |
| ParT_kin | W/QCD | 0.950 (0.923) | 4.9 / 9.2, [−6.3, −2.3], 0.63, 3e-4 | 6.0 / 7.3, [−3.2, 0.7], 0.52, 0.29 | 0.6 / 0.0 |
| ParT_kin | top/QCD | 0.982 (0.973) | 5.1 / 4.3, [−0.8, 2.6], 0.54, 0.45 | 3.2 / 1.4, [0.6, 3.1], 0.37, 0.007 | 0.4 / 0.0 |
| ParT_kin | Z/W | 0.758 (0.764) | 27.2 / 31.2, [−7.3, −0.7], 0.59, 0.03 | 20.3 / 26.3, [−9.3, −2.7], 0.44, 9e-4 | 2.2 / 0.0 |

- With m_A in place of m, the natural-jet ADO is unchanged or higher in all 9 rows, and the WTA/translation response goes from 0 % to
  the network's order of magnitude in all 9 rows, with per-jet Δscore correlations 0.31–0.91. JEDI Z/W WTA: H0 not rejected (q 0.96),
  r 0.88. ParT: the swap surrogate over-predicts in 6 of 9 WTA/translation cells (ParT only partly follows M2, as in P1: 0.7–0.8 of
  the M2 prediction for ParT_full).
- The 'axis' search picks m_A only for ParT_full Z/W (and ΣpT once); elsewhere it keeps m, because on natural jets m and m_A rank
  pairs identically (corr 0.99999). ADO on natural jets therefore cannot tell an axis-free mass from an axis-relative one; which of
  two equally good surrogates is selected, and hence the surrogate's edit response, is set by the pool and the search order, not by the network.
- Split: m_A is collinear-safe, so the swap surrogate also changes 0.0 % under the split; the networks' 0.1–3.4 % split response is
  not explained by this control.
- Boost: neither swap nor 'axis' reproduces JEDI's boost response (Z/W 18.0 vs 2.3 %); ΣpT (D1) was offered but not selected by ADO.

## Null (redrawn weights), normalised
Mean |Δscore| / sd(unedited score) on the test jets (WTA, translate, boost, split):
- Trained ParT_full: W/QCD 0.16 / 0.17 / 0.07 / 0.04; Z/W 0.31 / 0.28 / 0.13 / 0.08. Redrawn ParT_full: 0.75 / 0.89 / 0.64 / 0.52 (W/QCD).
- Trained JEDI Z/W 0.93 / 1.22 / 0.44 / 0.07. Redrawn JEDI W/QCD 0.58 / 1.09 / 0.37 / 0.26: the redrawn JEDI score moves as much as
  the trained one in sd units; its binary call is constant only because its score never crosses 0.
- The redrawn ParT_full predicts one class for 99.95 % of the 4-class jets; its 10-class calls change by 0.27–0.47 % under the edits
  (trained ParT_full 2.7–17.3 %). Its large binary-call changes (8–36 %) come from a near-zero binary score.
So "an untrained ParT also responds" holds for the binary log-ratio but not for 10-class calls, and neither null tests learned use.

## Figure (`results/physics/h8_fig.png`)
Four panels (WTA, translate, boost, split), network vs HGB surrogate % calls changed with 95 % bootstrap bars. Values match
`h8_efp.json`. The title states the surrogate pool and the ADO range; no verdict wording. It shows only the axis-free surrogate, so
the 0 % orange bars under WTA and translation are fixed by construction.

## Corrected reading
- Reproduced on fresh jets: ADO 0.92–0.98 (W/QCD, top/QCD, JEDI Z/W), 0.71–0.76 for ParT Z/W; axis-free surrogates 0.0 % under WTA and
  translation against 0.7–54 % for the networks; exact collinear split 0.1–3.4 % vs 0.0 %. Chosen sets beyond the first observable vary
  with the sample (JEDI W/QCD 3 vs 8; ParT top/QCD e2²·e3 vs e3).
- The 0 % surrogate response is fixed by the axis-free pool. An ADO-equivalent surrogate using the axis-moment mass √(M0·M2) about the
  given axis (= m on natural jets) reproduces JEDI-linear's Z/W WTA response (37.6 vs 37.7 %, r 0.88) and most of its W/QCD response,
  and gives ParT responses of the same order (over-predicting). The WTA/translation gap is therefore largely the known A4 variable,
  not something outside standard jet-mass physics; natural-jet ADO cannot distinguish the two mass definitions.
- Not explained by m_A: the collinear-split response of all networks and JEDI's boost (ΣpT) response; ParT Z/W natural-jet ranking.
- The McNemar/BH tests in H8 compare against a constant-zero surrogate and test only that the networks change calls (D4/A4).
- Redrawn nulls: binary-score sensitivity only; the redrawn ParT is near-constant in 10-class calls and the redrawn JEDI score moves without crossing 0.
