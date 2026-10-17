# E20: causal test of the pT-rank windows (claims A1/A3; reviewer objection "objects = sum pooling / EFN filters")

Agent report, 2026-10-17. Script `interp/e20_rank_windows_jedi.py` (part 1 default; part 2 `--programs`), figure `interp/e20_plot.py`.
Results `results/jedi/e20_rank_windows.json`, `results/jedi/e20_programs.json`, `results/jedi/e20_rank_windows.png`.
Logs `logs/e20-run3.log`, `logs/e20-prog.log`. Cluster jobs: e20-run3 and e20-prog, each 1 CPU / 2 GiB. Both runs reported a pod memory peak
of 2047 MiB, which is the request (the cgroup figure includes page cache), and used 0.2 cores on average. Two earlier submissions (e20-run,
e20-run2) were deleted: e20-run was stuck in CephFS D-state on chi-dgx-node01, and e20-run2 was stopped so that edit A12 could be added.

## Procedure
- **Sample.** `sample(per_class=2000, seed=41)` gives N = 10,000 jets. pT is non-increasing along the slots in 100 % of them. The fixed-point
  `Jedi(64)` scores 81.15 % on this sample, which is inside the 80.6–81.3 % check.
- **Windows.** Read from the zero-bit masks (`iq.quantizer.b > 0`) of qs1 for S and of qs for P. The 19 live S units have these windows:
  - 0–13: u0, u43, u52, u63
  - 0–12 and 14: u51
  - 13–63: u19
  - 1–63: u1
  - 0–10: u13; 0–9: u17; 0–2: u20; 5–63: u62; 8–63: u56
  - all 64 slots: u29, u30
  - partial windows: u50, u60, u40, u31, u58

  The lower edge of u19 lies between slots 12 and 13, not between 13 and 14 as the plan assumed. Edit A12 was added to cover it.
  The live P windows have no boundary at 12|13, 13|14 or 0|1. P32 has a boundary at 5|6.
- **Edits.** pT order is kept, and jets that lack the particles an edit needs are skipped.

  | Edit | Change | N |
  |---|---|---|
  | A13 | swap (Δη, Δφ) between ranks 13 and 14 | 9,951 |
  | A12 | swap (Δη, Δφ) between ranks 12 and 13 | 9,965 |
  | A0 | swap (Δη, Δφ) between ranks 0 and 1 | 10,000 |
  | I5 | swap (Δη, Δφ) between ranks 5 and 6 | 10,000 |
  | I30 | swap (Δη, Δφ) between ranks 30 and 31 | 7,913 |
  | N13 | rank-13 pT set just below rank 14's, then re-sorted (mean ΔpT −1.74 GeV) | 9,912 |
  | N5 | the same nudge at ranks 5/6 (mean ΔpT −7.74 GeV) | 9,977 |
  | N12 | the same nudge at ranks 12/13 (part 2 only) | 9,930 |

  The new pT is $p_{T,k+1} - \min(0.01\,p_{T,k+1}, (p_{T,k+1}-p_{T,k+2})/2)$.
- **Metrics on the network.** For each edit and live S unit:
  - $\Delta S$, compared with the window identity $\mathrm{pred_{win}} = \tfrac1{16}\sum_i m_{ic}\,\Delta u_{ic}$, where $u$ is the network's own float u before the qs1 quantizer;
  - the no-window sum $\mathrm{pred_{all}} = \tfrac1{16}\sum_i \Delta u_{ic}$;
  - the fraction of jets with $|\Delta S|$ above 1 LSB (the coarsest qs1 step over the kept slots), and the fraction with $\Delta S \neq 0$;
  - the % of calls changed, and accuracy before → after with an exact McNemar test.

  Confidence intervals come from 1,000 bootstrap resamples over jets.
- **Contrasts.** Paired comparisons A13–I5, A13–I30, A12–I5, A0–I5 and N13–N5. Each unit uses a Wilcoxon test on $|\Delta S_A| - |\Delta S_I|$, with Holm
  correction over 95 tests (19 units × 5). Calls use an exact McNemar test.
- **Mask controls.** These use the float + masks program (E3m). Each control is run on qs1 alone and on all four slot-mask layers (qs, l2, l3, qs1):
  - permuted along the slot axis (10 draws);
  - shifted by ±3 slots (edge value replicated);
  - all ones.

  Part 2 adds, per jet:
  - $\Delta S$ in the network and in the programs with each single layer's slot mask set to one;
  - $\Delta S$ with all four slot masks set to one, and with all four shifted by ±3.

  H0 for part 2: removing the slot masks leaves $|\Delta S|$ unchanged. It is tested with a Wilcoxon test, Holm-corrected over 152 tests (8 edits × 19 units).
- **Untrained controls.**
  - `Jedi(untrained=True, seed 0)` keeps the trained number formats and masks.
  - A He-normal float network (kernels $N(0, 2/\mathrm{fan_{in}})$, biases 0) runs with the trained masks.

## Numbers

**Identity (network).** $\Delta S$ against $\mathrm{pred_{win}}$ gives R² ≥ 0.98 for most live units under the boundary edits:
- A13: S43 1.000, S63 0.999, S52 1.000, S1 0.988 [0.987, 0.989], S29 0.998, S60 0.984, S31 0.985.
- N13: S0 0.987 [0.986, 0.987].
- Lower values: S30 0.91, from the fixed-point context path. S62 0.48.

$\mathrm{pred_{all}} = \mathrm{pred_{win}}$ for every unit except S13 and S17. For all other units $u$ is already 0 outside the qs1 window. For S13 and S17 the qs1 mask
removes nonzero $u$:
- S13 under N13: R²_win 0.982, R²_all −51.8.
- S17 under N13: R²_win 0.963, R²_all −1.66.

**Window-bounded angular units (0–13: S43, S52, S63), edit A13, float + masks program, mean |ΔS| [95 % CI].**

| Program | S43 | S52 | S63 |
|---|---|---|---|
| Trained masks | 0.0856 [0.0825, 0.0888] | 0.0567 [0.0539, 0.0599] | 0.0833 [0.0812, 0.0854] |
| All four slot masks = 1 | 0.0039 [0.0037, 0.0041] | 0.0013 [0.0012, 0.0014] | 0.0025 [0.0024, 0.0026] |
| Shift −3 (window 0–10) | 0.0000 | 0.0000 | 0.0000 |
| Shift +3 (window 0–16) | 0.0039 | 0.0013 | 0.0025 |

- Holm p for "mask = ones" ≤ 4e-220 for all three units.
- Network vs program, per-jet R²: 0.96, 0.89, 0.95.
- Matched control I5 (inside), trained masks vs all ones:
  - S43: 0.0176 vs 0.0176
  - S52: 0.0036 vs 0.0036
  - S63: 0.0071 vs 0.0071
  - Holm p = 1 for all three.
- I30 (outside the window): 0.0000 with the trained masks.
- N13 gives the same pattern: S43 0.084 → 0.006, S52 0.056 → 0.001, S63 0.083 → 0.003.
- Setting any single layer's slot mask to one (qs, l2, l3 or qs1) leaves the A13 response of S43, S52 and S63 unchanged (0.0856, 0.0567, 0.0833).
  The 0–13 window is therefore set in more than one layer.

**Other units.**
- **S1 (1–63).**
  - A13 gives 0.0149 with the masks and 0.0011 with ones; ones on l3 alone gives 0.0011.
  - In the network, A13 |ΔS| is 0.0209 against 0.0017 for I5 (Holm p 3e-225).
  - A0 crosses the S1 boundary but gives only 0.0008 in the network, with 0.1 % of jets nonzero. The leading particle rarely has r > 0.2, so the S1 hinge is off for it.
- **S13 (0–10) and S17 (0–9).**
  - The window is set by qs1 alone. With ones on qs1, A13 gives S13 0.0095 and S17 0.081, against 0 with the masks.
  - Shifting the window by +3 moves the S17 response to A12: 0.089, against 0 with the masks.
- **S29 (all 64).**
  - Masks vs ones under A13: 0.0121 vs 0.0121.
  - Network, A13 vs I30: 0.0269 vs 0.0266, difference CI [−0.0010, 0.0016].
  - Under A0, ones on l2 removes the response (0.0132 → 0), so slot 0 has its own per-particle function through the l2 mask.
- **S30 (all 64).** In the network, A13 gives 0.046, with 26 % of jets above 1 LSB. In every float program it gives 0.000 (network vs program R² 0.00). The S30
  response to angle swaps comes from the fixed-point path.
- **S0 and S19 (pT-only units).** ΔS = 0 exactly under every angle swap (A13, A12, A0, I5, I30), in the network and in all programs.
  - Under N13, S0 changes by 0.099 (45.5 % of jets above 1 LSB). Under N5 it changes by 0.410 (87.5 %). The nudge sizes differ (mean ΔpT −1.7 vs −7.7 GeV).
  - S19 counts particles with pT < 2 GeV. It does not respond to N12, because the 1 % nudges do not cross 2 GeV. The S19 window (12|13) is not tested by these edits.

**"Inside" edits on the network.** ΔS is not 0 for ≥ 99 % of jets. Under I5, the fraction of jets above 1 LSB is:
- S43 27.2 %, S17 26.2 %, S13 11.9 %, S30 5.6 %, S29 1.3 %.
- Fraction with ΔS ≠ 0: S43 37.6 %, S60 51.8 %.

In the program these I5 responses do not change when the masks are removed (see above).

**Calls (network), % changed [CI], with McNemar tests.**

| Edit | Calls changed | Accuracy before → after | McNemar on correctness |
|---|---|---|---|
| A13 | 1.20 [0.98, 1.41] | 81.14 → 81.17 | — |
| I5 | 1.03 [0.83, 1.22] | 81.15 → 81.01 | — |
| I30 | 0.32 | — | — |
| A0 | 3.03 [2.72, 3.36] | 81.15 → 79.60 | p 5e-22 |
| A12 | 0.71 | — | — |
| N13 | 1.23 | — | — |
| N5 | 1.96 | — | — |

| Paired comparison | Calls changed | McNemar p |
|---|---|---|
| A13 vs I5 | 1.20 vs 1.04 | 0.24 |
| A13 vs I30 | 1.35 vs 0.32 | 7e-16 |
| A0 vs I5 | 3.03 vs 1.03 | 4e-28 |
| A12 vs I5 | 0.71 vs 1.03 | 0.009 |
| N13 vs N5 | 1.23 vs 1.97 | 1.4e-6 |

Program, A13 with masks vs ones_all4: 0.49 % vs 0.20 % (McNemar p 2e-4).

**Mask controls: accuracy (program with trained masks 80.66 %; 3.93 % of calls differ from the network).**

| Program | qs1 only | All four slot layers |
|---|---|---|
| Permuted, 10 draws | 34.85 % (29.68–41.47); 61.5 % of calls changed | 31.63 % (29.64–37.84); 64.7 % |
| Shift +3 | 80.48 % | 74.75 % |
| Shift −3 | 79.76 % | 78.39 % |
| All ones | 80.33 % | 44.47 % |

Ones on a single layer: qs 80.70 %, l2 79.61 %, l3 80.37 %, qs1 80.33 %.

**Untrained controls.**
- `Jedi(untrained=True)`: accuracy 20.0 %. 17 of the 19 S units are constant (sd 0), so ΔS = 0 for every edit. This control is uninformative.
- He-normal float network with the trained masks: accuracy 20.0 %. Mean |ΔS|/sd(S), A13 vs I5 vs I30:
  - S0: 0.114 vs 0.015 vs 0.0001
  - S43: 0.078 vs 0.020 vs 0.0002
  - S52: 0.140 vs 0.012 vs 0.0001
  - S63: 0.142 vs 0.011 vs 0.0001

## Reading (scoped to these measurements)
- **Window-bounded angular units.** For S43, S52 and S63 (window 0–13), an angle swap across 13|14 changes S by 0.057–0.086 per jet. Setting all slot masks to
  one reduces this by 95–98 %. Moving the window to 0–10 gives 0, and moving it to 0–16 gives the no-window value. A swap at 5|6, inside the window, gives the same
  response with or without masks. So these objects respond to rank-window membership, as the window identity predicts, and a permutation-invariant sum with the
  same weights does not.
- **Same pattern elsewhere.**
  - S1 shows it through the l3 slot mask.
  - S13 and S17 show it through qs1: a shifted window moves the response to the new boundary.
  - The He-normal network with the trained masks shows the same across > inside pattern. The pattern follows from the masks, not from trained weights.
- **H1 as stated is not met in four respects.**
  1. "Inside" swaps change S for up to 27 % of jets above 1 LSB (S43, S17). These responses are unchanged by mask removal: they come from swapping angles between
     particles of different pT in a hinge that is not separable in (pT, r).
  2. S0 and S19 have no angular dependence, so angle swaps cannot test their windows. For S0 the pT nudge follows the identity (R² 0.987). The S19 window was not
     probed.
  3. S29 and S30 do respond to A13. S29 responds equally at the rank-matched I30. S30 responds only through the fixed-point path.
  4. Calls change more for ACROSS than INSIDE only for A0 vs I5 (p 4e-28) and A13 vs I30 (p 7e-16). They do not for A13 vs I5 (p 0.24), and the order is reversed
     for A12 vs I5 and N13 vs N5.
- **Alternative hypothesis.** The alternative (windows are quantisation noise and units are effectively permutation-invariant) predicts equal ACROSS and INSIDE
  responses and permuted-mask accuracy within 2 points of 80.68 %. Both predictions are excluded:
  - ACROSS vs INSIDE for S43, S52, S63: Holm p ≤ 4e-220;
  - permuted masks: 31.6–34.9 %.
- **Single-layer redundancy.** Setting any single layer's slot masks to one costs ≤ 1.1 points, and setting qs1 alone to ones costs 0.3 points. Setting all four
  to ones costs 36 points. The windows are set redundantly across qs, l2, l3 and qs1, so the qs1 mask is not the sole place that defines an object's window.

## Issues
- The plan placed u19's boundary at 13|14. The masks put it at 12|13, so A12 and N12 were added. The u19 window remains untested, because u19 only changes when a
  pT crosses 2 GeV.
- The coarse-LSB criterion is lenient for units with steps up to 1.0 (S1, S50, S51). The fraction with ΔS ≠ 0 and the fraction above the fine LSB are also in the json.
- The "inside" and "across" edits sit at different ranks, so their baselines differ: for S29, A13 vs I5 gives 3.3× with no window. The mask-removal comparison at
  fixed rank, in the program, is the matched control used for the window claim.
- The untrained Jedi control is degenerate (S constant).

## Explainer entry (for site/explain.py, Paper A; proposed `E20_EXPLAINER`)
- **Question.** Are the pT-rank windows of the second-round objects causal, or would a plain permutation-invariant sum over particles behave the same?
- **Background.** Each object is $S_c = \tfrac1{16}\sum_i m_{ic}\,u_c(x_i)$. The mask $m_{ic}\in\{0,1\}$ comes from HGQ's zero-bit entries and is fixed per slot
  $i$, with slots ordered by $p_T$. For example, $S_{43}$, $S_{52}$ and $S_{63}$ use the 14 hardest particles. A sum-pooling (EFN-type) network has $m_{ic}=1$
  everywhere.
- **Method.** On 10,000 jets, swap only the angles $(\Delta\eta,\Delta\phi)$ of the particles at ranks 13 and 14, which crosses the 0–13 boundary. Do the same for
  ranks 5 and 6, which lie inside the window. pT order is unchanged, so the jets stay valid inputs. Compare $\Delta S$:
  - in the network;
  - in the float program with the trained masks;
  - in the same program with all slot masks set to 1;
  - with the masks shifted by ±3 slots.

  Figure: results/jedi/e20_rank_windows.png.
- **Result.** The swap across 13|14 changes $S_{43}$, $S_{52}$ and $S_{63}$ by 0.086, 0.057 and 0.083 per jet. With the masks set to 1 the change is
  0.004, 0.001 and 0.003. Shifting the window to 0–10 gives 0. The swap at 5|6 gives 0.018, 0.004 and 0.007, the same with or without masks. Permuting the
  masks across slots lowers accuracy from 80.7 % to 32–35 %, and setting all slot masks to 1 lowers it to 44.5 %.
- **Checks.**
  - The network's $\Delta S$ equals the window sum of its own per-particle $u$ (R² ≥ 0.98 for most units).
  - A random-weight network with the trained masks shows the same across/inside difference.
  - Swaps inside a window still change hinge objects, because the hinge is not separable in $(p_T, r)$.
  - Angle swaps do not move the $p_T$-only objects $S_0$ and $S_{19}$.
  - Removing one layer's masks costs ≤ 1.1 points: the windows are set in several layers at once.
- **Why it matters.** It separates the program's objects from sum pooling with a causal edit and a matched control. Whether a particle counts depends on its pT
  rank as well as on its cut, and a network without the rank masks responds differently to the same edit.
