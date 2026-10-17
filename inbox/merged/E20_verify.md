# E20v: skeptic rerun of E20 (pT-rank windows, claims A1/A3)

Verifier report, 2026-10-17.
- Scripts: `interp/e20_verify.py` (rerun, edge scan, near-rank DiD, calls, accuracy controls) and `interp/e20_verify_add.py` (add-layer masks). Figure script: `interp/e20_verify_plot.py`.
- Results: `results/jedi/e20_verify.json`, `results/jedi/e20_verify_add.json`. Figure: `results/jedi/e20_verify_profiles.png`.
- Logs: `logs/e20-verify.log`, `logs/e20-verify-add.log`.
- Jobs: e20-verify and e20-verify-add, each 1 CPU / 2 GiB. Both exited 0. The pod memory peak was 2047 MiB in both (the cgroup figure includes page cache).

## Code review of `interp/e20_rank_windows_jedi.py` / `e20_plot.py`
- **Edits.** The swaps exchange only (Δη, Δφ) between two real particles, so pT order is kept and padded slots are never touched.
  - Jets that lack the needed particles are skipped. The nudge edit changes the moved particle's pT (−1.7 GeV mean at 13/14, −7.7 GeV at 5/6), and E20 reports this.
  - Both windows are read from `iq.quantizer.b > 0`, as in E3m.
  - `shift_mask` behaves as stated: +3 turns 0–13 into 0–16, and −3 turns it into 0–10.
  - No split leakage: nothing is fitted.
- **Omitted mask layer.** The float + masks program (`prog_fwd`, copied from E3m) leaves out the zero-bit masks of the `add` layer. That layer quantizes s and g separately before l3.
  - Mask on s: shape (1, 64, 64), 90.7 % zero bits, slot-dependent. Kept slots: s14 and s32 0–13, s53 0–16, s52 0–38/42–44/47, s43 and s60 0–63.
  - Mask on g: shape (1, 1, 64), 82.8 % zero bits; g channels kept: 4, 5, 29, 32, 36, 39, 44, 47, 49, 55, 63.
  - E20's "all four slot masks = 1" therefore means qs, l2, l3 and qs1 set to 1, with no add-layer mask at all.
  - The effect is measured below: it changes the redundancy statement, not the window effect.
- **Mask structure on l1.** The l1 input mask is not slot-dependent: pT, Δη and Δφ are kept at all 64 slots.
- **LSB criterion.** The 1-LSB threshold uses the coarsest qs1 step over kept slots, which is 1.0 for S1, S50 and S51. E20 states this.
- **"Rank-matched" wording.** The reading calls I5 "a rank-matched swap inside the window". I5 is at rank 5, not matched to 13. The matching in E20 is only across mask conditions at a fixed rank. The adjacent-rank inside swap W12 is added below.
- **Figure.** `e20_rank_windows.png` was checked.
  - Panel (c) is the float program, and it is labelled as such. "All slot masks = 1" in its legend excludes the add-layer mask.
  - The "A13: 13|14" and "I5: 5|6" labels in panel (b) are printed over the S19 and S13 bars.
  - Otherwise it reads correctly.

## Procedure (verification)
- **Sample.** `sample(per_class=2000, seed=77)` gives N = 10,000 fresh jets. pT is non-increasing in 100 % of them. Fixed-point `Jedi(64)` scores 80.64 %.
  - The edge scan uses the 9,861 jets with more than 17 particles, so that every W_k for k ≤ 16 is defined.
  - W30 is defined on 7,968 jets.
- **Edits.** W_k swaps (Δη, Δφ) between ranks k and k+1, for k = 0…16 and 30. W13 = A13, W12 = A12, W5 = I5, W0 = A0, W30 = I30.
- **Programs.** E20's own `prog_run`, `shift_mask` and `permute_mask` (imported) are run with:
  - the trained masks;
  - qs, l2, l3 and qs1 all set to 1 (ones_all4);
  - all four shifted by +3 or −3;
  - each single layer set to 1;
  - the pairs l2+l3, l3+qs1 and l2+qs1 set to 1.

  e20-verify-add runs the same program with the add-layer masks included, and also with the four slot masks set to 1 while the add masks are kept.
- **Metrics and tests.**
  - Mean |ΔS| per unit with 95 % bootstrap CIs (1,000 resamples over jets).
  - Edge test: for each live S unit with a qs1 window edge e ≤ 15, H0 is that the swap across the edge gives no larger |ΔS| than the adjacent swaps W_{e−1} and W_{e+1} (those that are not edges themselves). The test is a paired Wilcoxon on per-jet |ΔS_e| − |ΔS_nb|, in the network and in the program, Holm over 46 tests.
  - Shifted-edge tests: W10 vs W9 and W11, and W16 vs W15, Holm over 18 tests.
  - Near-rank difference in differences: $\mathrm{DiD} = (|\Delta S_{W13}|_{\rm mask} - |\Delta S_{W13}|_{\rm ones}) - (|\Delta S_{W12}|_{\rm mask} - |\Delta S_{W12}|_{\rm ones})$.
  - Calls changed in the network, with exact McNemar tests.
  - Accuracy with permuted, shifted or all-ones masks.
  - sd(S) per program, as a scale check.

## Numbers

### 1. Reproduction of E20's headline (seed 77; E20 seed 41 in brackets)
Mean |ΔS| under W13 (A13), float + masks program:

| unit | trained masks | ones_all4 | reduction [95 % CI] | shift −3 | shift +3 | network | R² net vs program |
|---|---|---|---|---|---|---|---|
| S43 | 0.0841 [0.0813, 0.0871] (0.0856) | 0.0039 (0.0039) | 95.4 % [95.1, 95.6] | 0.0000 | 0.0039 | 0.0820 | 0.96 |
| S52 | 0.0560 [0.0529, 0.0589] (0.0567) | 0.0013 (0.0013) | 97.6 % [97.5, 97.8] | 0.0000 | 0.0013 | 0.0599 | 0.88 |
| S63 | 0.0838 [0.0817, 0.0858] (0.0833) | 0.0026 (0.0025) | 96.9 % [96.8, 97.1] | 0.0000 | 0.0026 | 0.0860 | 0.95 |
| S1 | 0.0144 (0.0149) | 0.0011 (0.0011) | 92.2 % [91.5, 92.8] | | | 0.0200 | 0.87 |

- **I5 (W5).** Trained masks vs ones_all4: S43 0.0178 vs 0.0178, S52 0.0036 vs 0.0036, S63 0.0069 vs 0.0069. E20 had 0.0176, 0.0036, 0.0071.
- **S1.** With l3 set to 1, W13 gives 0.0011 (E20: 0.0011).
- **S17.** With qs1 set to 1, W13 gives 0.085 (E20: 0.081). With the masks shifted +3, W12 gives 0.089 (E20: 0.089).
- **Scale check.** sd(S43) is 1.96 with the trained masks and 4.00 with ones_all4. The 20× drop in the W13 response is therefore not a loss of scale.
- **Accuracy (program with trained masks 80.05 %).**
  - Permuted qs1 masks: 36.0 % (32.9–40.0, 5 draws).
  - Permuted masks on all four layers: 32.6 % (31.5–33.9, 3 draws).
  - Masks shifted +3 / −3: qs1 alone 79.97 / 79.25 %; all four 74.72 / 77.98 %.
  - All ones: qs1 alone 79.97 %; all four 43.35 %.
  - E20 values: 34.85, 31.63, 80.48 / 79.76, 74.75 / 78.39, 80.33 and 44.47 %.

### 2. Edge scan
Mean |ΔS| per swap W_k (figure: `results/jedi/e20_verify_profiles.png`).

- **S43, S52, S63 (window 0–13).**
  - With the trained masks, the profile falls from W0 (0.040 / 0.035 / 0.022) to W12 (0.0043 / 0.0014 / 0.0028). It then jumps at W13 (0.084 / 0.056 / 0.084) and is 0 from W14 on.
  - Edge vs neighbours:

    | unit | program, W13 − W12 | program, W13 − W14 | network, W13 − W12 | network, W13 − W14 |
    |---|---|---|---|---|
    | S43 | +0.080 [0.077, 0.083] | +0.084 | +0.064 [0.061, 0.067] | +0.082 |
    | S52 | +0.055 | +0.056 | +0.049 | +0.060 |
    | S63 | +0.081 | +0.084 | +0.073 | +0.086 |

    All of these have Holm p ≤ 4e-162.
  - With the masks shifted −3, the peak moves to W10: S43 0.088 vs 0.0075 (W9) and 0 (W11), Holm p ≈ 0. With +3 it moves to W16: S43 0.088 vs 0.0031 (W15).
  - S52 and S63 show the same: W10 under −3 gives 0.052 and 0.086; W16 under +3 gives 0.065 and 0.083. All Holm p ≤ 3e-173.
  - With the trained masks, W10 and W16 give no peak: S43 W10 0.0061 vs W9 0.0075; W16 = 0.
  - With ones_all4 the profile decreases monotonically and has no peak (argmax at W0).
- **S17 (0–9) and S13 (0–10).**
  - S17: the edge at W9 gives 0.108 against 0.008 at W8 and 0 at W10. In the network it is 0.111 against 0.034 and 0. Holm p ≈ 0.
  - S13: the edge at W10 gives 0.016 against 0.004 (W9) and 0 (W11) in the program, and 0.013 against 0.0085 and 0 in the network. Holm p ≤ 4e-11.
  - The peak moves under the shifts: S17 to W6 (0.129) under −3 and to W12 (0.089) under +3.
  - With qs1 alone set to 1, the S13 and S17 peaks move to W13 (0.010 and 0.085). The l3 / add masks impose a 0–13 edge on these units as well.
- **S62 (5–63).** The edge at W4 gives 0.0076 against 0 at W3 and W5 (program), and 0.0124 against 0 and 0.0007 (network). Holm p ≤ 4e-91.
- **Edges with no larger response at the edge.**
  - S1 (1–63), edge W0: the network gives 0.0011 against 0.0044 at W1. The S1 hinge is rarely on for the leading particle, as E20 noted.
  - S20 (0–2), edge W2: 0.0003 against 0.0024 at W1.
  - S56 (8–63) and S51 (0–12, 14): |ΔS| = 0 at the edge and at the neighbours.
  - S0 and S19 do not respond to angles.

  Of the 11 units with a qs1 edge at rank ≤ 15, six show an edge peak in both the program and the network (S43, S52, S63, S13, S17, S62). S1 shows one at W13 but not at its own qs1 edge W0, and four (S20, S56, S51, plus S0 / S19, which have no angular term) show none.

### 3. Near-rank difference in differences (inside swap at the adjacent rank)
- **Inside swap W12.** The response is unchanged by mask removal: S43 0.00432 vs 0.00432; S52 0.00145 vs 0.00145; S63 0.0028 vs 0.0028.
- **DiD (W13 vs W12).**
  - S43: 0.0802 [0.0772, 0.0833].
  - S52: 0.0546 [0.0515, 0.0576].
  - S63: 0.0812 [0.0792, 0.0832].
  - Wilcoxon p ≈ 0 for all three.
- **Outside swap W14** (trained masks): 0.0000 for all three units.

### 4. Calls (network, % changed)

| comparison | % changed | McNemar p |
|---|---|---|
| W13 vs W12 | 1.03 vs 0.84 | 0.12 |
| W13 vs W14 | 1.03 vs 0.84 | 0.093 |
| W13 vs W5 | 1.03 vs 1.04 | 1.0 |
| W0 vs W5 | 3.27 vs 1.04 | 1e-30 |
| W13 vs W30 | 1.10 vs 0.43 | 3e-8 |

This reproduces E20's point (4): the call-level difference is not significant for A13 vs I5, and is significant for A0 vs I5 and A13 vs I30.

### 5. Which layers set the 0–13 window (pair removals and add-layer masks)
In the program without add masks:

| masks set to 1 | accuracy | W13 peak of S43 / S52 / S63 |
|---|---|---|
| any single layer | 79.0–80.2 % | kept |
| l2 + l3 | 78.30 % | kept |
| l2 + qs1 | 78.56 % | kept |
| l3 + qs1 | 53.44 % | gone; profile identical to ones_all4 |
| all four | 43.35 % | gone |

- Adding the add-layer masks to the trained program changes nothing: accuracy 80.05 %, and calls vs the network 4.32 % in both versions.
- With the add masks kept and qs, l2, l3, qs1 set to 1:
  - S43 W13 0.0839 (trained 0.0841), S52 0.0560, S63 0.0838. Accuracy 52.21 %.
  - S17 W13 0.0847.
- With the add mask on s also set to 1: S43 W13 0.0039 and accuracy 43.35 %, the E20 ones_all4 values.
- So in the network the 0–13 window of S43, S52 and S63 is set by three masks: the add-layer mask on s (s14, s32 kept at slots 0–13), the l3 input mask, and the qs1 mask.
  - Any one of them alone keeps the W13 peak.
  - The qs and l2 masks do not contribute to these units.
- S30's network-only angle response is not reproduced with the add masks either: program |ΔS| is 0 and R² is 0.0. It comes from the fixed-point path, as E20 states.

### 6. Inside-swap responses: float program vs network
The network's inside response is larger than the program's:

| swap | unit | network | program |
|---|---|---|---|
| W5 | S63 | 0.0169 | 0.0069 (R² 0.30) |
| W5 | S17 | 0.0326 | 0.0143 (R² 0.46) |
| W5 | S43 | 0.0228 | 0.0178 (R² 0.91) |
| W12 | S43 | 0.0183 | 0.0043 (R² 0.17) |
| W12 | S52 | 0.0113 | 0.0014 (R² −0.01) |

The non-separable-hinge part (the program) accounts for 12–78 % of the network's inside response, depending on unit and rank. The rest is not in the float program, and is consistent with fixed-point rounding (not separately tested).

## Hypotheses and which are excluded
- **H0, no window.** The objects are permutation-invariant sums, and the masks act as noise. It predicts:
  - equal |ΔS| for the edge swap and the adjacent swaps;
  - no movement of the peak when the masks are shifted;
  - permuted masks within 2 points of 80 %.

  All three predictions are excluded for S43, S52, S63, S13, S17 and S62: edge vs adjacent Holm p ≤ 4e-11; the peak moves by exactly ±3; permuted masks give 32–36 %.
- **H1, window identity.** Under H1, |ΔS| at the edge swap exceeds |ΔS| at the adjacent inside swap. The adjacent swap outside the window gives 0. The peak follows the mask position, and the inside swap's response does not depend on the masks. All four predictions hold for those six units.
- **H1' (E20's wording), windows set "in qs, l2, l3 and qs1".** Not supported as stated. For the 0–13 units the window is set by the add (s) mask, l3 and qs1, and qs and l2 have no effect on them.

## Corrections to the E20 reading
1. "A rank-matched swap inside the window gives the same response with or without masks": I5 is at rank 5, so it is not rank-matched to A13.
   - At the adjacent inside rank (W12) the response is also unchanged by mask removal (0.0043 vs 0.0043 for S43).
   - The DiD W13 vs W12 is 0.080 [0.077, 0.083] (S43), 0.055 (S52) and 0.081 (S63).
2. "The windows are set redundantly across qs, l2, l3 and qs1; removing all four costs 36 points": E20's program omits the add-layer masks.
   - For S43, S52 and S63 the window is set in three places: the add-layer mask on s (s14 and s32 kept at slots 0–13), the l3 input mask, and qs1.
   - qs and l2 have no effect on these units' windows.
   - Removing l3 + qs1 (add masks absent) costs 26.6 points. Removing qs, l2, l3 and qs1 with the add masks kept costs 27.8 points, and the 0–13 window is still present.
3. Inside-swap responses, "from the (pT, r) hinge not being separable": this holds for the float program. In the network, 12–78 % of the inside response (unit- and rank-dependent; W12: 23 % for S43, 12 % for S52) comes from the hinge, and the rest is not in the float program.
4. The edge test generalises to S13, S17 and S62, each at its own qs1 edge with a matched adjacent control. It does not show an edge peak for S1 at its qs1 edge 0|1, for S20, or for S56 and S51 (no angular response).

## Explainer entry (proposed addition to the E20 explainer, Paper A)
- **Question.** Does the response follow the window edge wherever the edge is placed?
- **Method.**
  - On 9,861 fresh jets (seed 77), the angles of ranks $k$ and $k+1$ are swapped for $k = 0,\dots,16$.
  - $\overline{|\Delta S_c|}$ is measured in the float + masks program with the trained masks, with the masks shifted by $\pm 3$ slots, and with the masks removed.
  - A window object $S_c = \tfrac1{16}\sum_i m_{ic}\,u_c(x_i)$ changes under a swap at $k$ by $\tfrac1{16}[m_{kc}(u_c(x_k') - u_c(x_k)) + m_{k+1,c}(u_c(x_{k+1}') - u_c(x_{k+1}))]$. Only at an edge ($m_{kc} \neq m_{k+1,c}$) does the full single-particle change survive without cancellation.
- **Result.** For $S_{43}$, the peak sits at $k = 13$ (0.084 against 0.004 at $k = 12$ and 0 at $k = 14$). It moves to $k = 10$ and $k = 16$ with the shifted masks (0.088 each) and disappears with the masks removed. The same holds for $S_{52}$, $S_{63}$, $S_{13}$, $S_{17}$ and $S_{62}$.
- **Check.** The 0–13 window is written three times in the network: in the add-layer mask on s, in the l3 input mask, and in the qs1 mask. Any one of them alone reproduces the peak.
