# B6 second verification (2026-10-20): another fresh jet sample, and Deep Sets / transformer against the tuned baseline grid

## Conclusion
On a third independent set of jets, the 30-term rank-window program again follows the networks' call changes under a 0.06 rad shift of the jet about as often as a boosted-tree model of the same size whose settings were chosen on held-out jets: the gap was +1.7 points for JEDI-linear and −0.3 for PFN, against −0.1 and −0.8 in the first verification. Deep Sets (+0.2) and the transformer (+1.6) give the same picture when refitted against the same baseline grid on B6's jets. Under the shift with the jet mass proxy M2 (Σ pT·r², r = angular distance from the jet axis) held fixed, the program's lead in JEDI-linear replicates (+1.7 points), but under that edit all surrogates sit within about 3 points of a predictor that never changes its call. The statement "within ±1 point, intervals include 0" is sample-dependent: across three jet samples and four networks the gap at equal size ranges from −1.3 to +1.7 points.

## Procedure
- Code review of `interp/b6_pareto.py` and `interp/b6_verify.py`: no defect found in stages, masks, splits or edits beyond what the first verification lists. Checked: `indices()` matches `sample()`; edits multiply by the real-particle mask (pT > 0), so padded slots are not moved; m2h rescales both angles about the stored axis by √(M2/M2′) (M0 and M2 held, padding stays 0); `hgb_bits` counts all 5 per-class trees per iteration of the GBDT classifier; the MARS intercept is charged 32 bits per logit; bootstrap resamples are shared between program and baseline (paired).
- New script `interp/b6_verify2.py` (imports the observables, MARS fit, bit costs, edits and network forwards of `b6_pareto` / `b6_verify`). Baseline set at ≤ prog30's bits, as in b6_verify: CART d2–10 (call / logits); GBDT leaves {2, 3, 4, 8} × lr {0.1, 0.2, 0.5, 1.0} × iterations up to the budget (call / logits); MLP h2, h4. Selection rules: devsc (highest dev same call), devedit (highest dev agreement under the same edit; fresh mode only), oracle (highest test agreement; upper bound for the baseline), b6gbdt (B6's GBDT L4 T10 lr 0.2).
- Job `b6-verify2` (`python -m interp.b6_verify2 fresh`), 1 CPU / 2 GiB, wall 323 s, peak 2,047 MiB (at the request limit; no OOM), 0.64 cores, exit 0.
  - Models JEDI-linear n64 (jedi_core) and PFN 100-100-128 s0.
  - F = `sample(4000, 4701)`: 13,333 fit / 6,667 dev (perm rng 0). T = `sample(2000, 4702)` minus every row of F, `sample(6000, 0)`, `sample(4000, 2601)`, `sample(4000, 3701)`, `sample(2000, 2602)`, `sample(2000, 3702)`: N = 6,509, disjoint from all earlier B6 fit and test sets. Edit seeds 4703–4717.
  - Output `$AI_RESULTS/jedi/b6_verify2.json`; log `logs/b6-verify2.log`.
- Job `b6-verify3` (`python -m interp.b6_verify2 stored`), 1 CPU / 2 GiB, wall 815 s, peak 1,151 MiB, 0.56 cores, exit 0.
  - All four networks on B6's stored jets (F = `sample(4000, 2601)`, T N = 8,152) and stored logits / program calls (`b6_logits_<m>.npz`, `b6_pred_<m>.npz`: prog16 and prog30 as fitted in B6). Baseline grid refitted on B6's F-fit rows. Dev edits were not stored, so no devedit rule. Fills the gap of b6_verify, which did not rerun Deep Sets and the transformer.
  - Output `$AI_RESULTS/jedi/b6_verify3.json`; log `logs/b6-verify3.log`.
- Metric: per-jet edit agreement (surrogate changes its call iff the network does, to the same call). Paired jet bootstrap B = 1000; exact McNemar; Holm over the trans / m2h / pt09 rows (42 in verify2, 60 in verify3). H0: equal per-jet agreement of program and matched baseline; H1: different. Specificity: paired bootstrap of [lead on edit] − [lead on rdt] (rdt = random-direction 0.06 rad translation).

## Numbers
Network on the new T (N 6,509): JEDI 81.09 %, PFN 81.96 %. Calls changed by trans 56.6 / 57.3 %, m2h 17.6 / 16.6 %, pt09 34.2 / 31.9 %, rdt 60.3 / 62.3 %, rpd 53.1 / 55.3 %.

**prog30 minus matched baseline, agreement in points [95 % CI], Holm p. Fresh sample 2 (seeds 4701/4702).**

| model | edit | b6gbdt (L4 T10 lr 0.2) | devsc | devedit |
|---|---|---|---|---|
| JEDI | trans | +8.11 [7.14, 9.08], 4e-54 | +1.67 [0.57, 2.83], 0.076 (GBDT L4 T10 lr 0.5) | +3.12 [1.58, 4.69], 0.004 (GBDT-reg L8 T2 lr 0.1) |
| JEDI | m2h | +2.52 [1.80, 3.24], 2e-10 | +1.74 [1.03, 2.46], 6e-5 | +1.58 [0.89, 2.27], 2e-4 |
| JEDI | pt09 | +7.79 [6.67, 8.82] | +4.22 [3.24, 5.21], 8e-15 | +4.22 |
| JEDI | rdt (control) | +6.87 | −0.46 [−1.55, 0.61] | −0.46 |
| PFN | trans | +3.89 [2.93, 4.82], 9e-15 | −0.34 [−1.23, 0.66], 1 | −1.12 [−2.06, −0.23], 0.27 |
| PFN | m2h | +1.94 [1.28, 2.61], 1e-6 | +0.83 [0.17, 1.51], 0.30 | +0.83 |
| PFN | pt09 | +4.64 | +1.24 [0.18, 2.21], 0.30 | +1.24 |
| PFN | rdt (control) | +2.26 | −1.49 [−2.40, −0.58] | −1.92 |

Raw agreement on trans, JEDI: prog30 59.96 %, GBDT lr 0.5 58.29 %, GBDT lr 0.2 51.85 %, never-flip 43.37 %. On m2h: prog30 84.85 %, GBDT lr 0.5 83.12 %, never-flip 82.41 %.

**B6's stored jets (N 8,152), extended grid, devsc / oracle.**

| model | trans devsc | trans oracle | m2h devsc | pt09 devsc |
|---|---|---|---|---|
| JEDI | −1.34 [−2.32, −0.43], Holm 0.10 | −2.26 [−3.19, −1.28] (GBDT L2 T5 lr 1.0, 2.6k bits) | +2.01 [1.35, 2.64], 9e-8 | +5.00 |
| Deep Sets | +0.23 [−0.81, 1.18], 1 | −2.36 [−3.34, −1.41] (GBDT-reg L8 T5 lr 1.0) | +0.43 [−0.21, 1.07], 1 | −1.24 |
| PFN | +0.81 [−0.15, 1.72], 0.93 | +0.06 [−0.94, 1.09] | +0.10 [−0.49, 0.69], 1 | +1.23 |
| transformer | +1.59 [0.78, 2.47], 0.008 | −1.72 [−2.69, −0.74] (MLP h4) | +1.69 [0.96, 2.36], 3e-5 | −2.29 |

b6gbdt rule on stored jets reproduces B6's matched numbers exactly (JEDI +2.80, Deep Sets +4.04, PFN +4.47, transformer +4.96 on trans).

**prog30 trans across the three jet samples, devsc rule.** JEDI −1.34 (B6 jets), −0.12 (verify 1), +1.67 (verify 2); PFN +0.81, −0.79, −0.34. Against B6's lr 0.2 GBDT: JEDI +2.80, +5.02, +8.11; PFN +4.47, +4.30, +3.89.

**Specificity, prog30, devsc: [lead on edit] − [lead on rdt].**
- Fresh 2: JEDI trans +2.14 [0.89, 3.33], m2h +2.20 [0.98, 3.47]; PFN trans +1.15 [0.06, 2.20], m2h +2.32 [1.31, 3.44].
- Stored: JEDI trans +1.02, m2h +4.37 [3.30, 5.48]; Deep Sets +1.37 / +1.57; PFN 0.00 / −0.71; transformer +1.40 / +1.50.

**m2h relative to the never-flip reference (B6 stored, b6.json).** Never-flip agreement under m2h is 82.5 % (JEDI), 84.6 (Deep Sets), 83.3 (PFN), 75.3 (transformer); prog30 is 85.3, 85.7, 84.0, 76.4. The flipped-set recall of prog30 under m2h is 0.35 (JEDI) vs 0.15 for GBDT L4 T10 lr 0.2 and 0.35 for GBDT L16 T300 (1.6 Mbit); Jaccard 0.34 vs 0.16 vs 0.34. Recall / Jaccard were not stored for the tuned (lr 0.5) GBDT.

**Metric dependence on trans (b6.json).** Jaccard of the flipped-jet sets (ignores the new call): JEDI prog30 0.63, GBDT L4 T10 lr 0.2 0.72–0.73, CART d7 0.71. The lead of prog30 over B6's lr 0.2 GBDT in agreement (same new call required) does not appear in Jaccard; there the GBDT is higher.

**Larger surrogates on trans (b6.json), best above prog30's bits.** JEDI: GBDT L16 T30 (12×) 69.5 vs 58.0 (+11.5); Deep Sets: GBDT-reg L16 T300 (122×) +5.9, MLP h16 (4×) +5.3; PFN: GBDT-reg L16 T300 (124×) +6.9; transformer: MLP h32 (7.7×) +4.9.

## Competing hypotheses
- H_prog (program exceeds every equal-DL baseline on trans): predicts positive devsc and oracle leads. Oracle leads on trans are ≤ 0 in JEDI (B6 jets), Deep Sets, PFN and the transformer; devsc leads range −1.3 to +1.7. Excluded as a general statement.
- H_equal (program ≈ tuned baseline at equal DL, within a few points, sign depending on jets and selection): predicts small leads of either sign. Observed.
- H_tuning (B6's lead came from the under-fitted lr 0.2 GBDT): predicts large lead against lr 0.2 and small lead against tuned GBDT. Observed in all four networks and three jet samples.
- For m2h in JEDI: lead over tuned GBDT positive in all three samples (+2.0, +2.0, +1.7), Holm ≤ 6e-5; also the transformer on B6 jets (+1.7); not in PFN (+0.7, +0.8, +0.1) or Deep Sets (+0.4).

## Figure check
`results/jedi/b6_trans.png` and `b6_m2h.png` re-read. trans: the CART (green) and MLP (orange) points at log2 bits ≈ 13.6–13.7 reach prog30's height in JEDI, Deep Sets and PFN; in the transformer the MLP h4 point is above prog30. m2h: every family lies within a few points of the never-flip line at ≤ 14 bits; prog30 is the highest point at ≤ 13.8k bits in all panels. The never-flip labels overlap the line (cosmetic).

## Issues
- The JEDI devsc trans gap differs by 3 points between jet samples (−1.3 to +1.7); the size of the program's fitted terms changes between samples (prog30 12.8k–13.4k bits), which moves which baselines are eligible. Single-sample statements of "±1 point" are not stable.
- Stored-jet mode reuses B6's F and T, so Deep Sets and the transformer are tested against the tuned grid but not on fresh jets.
- Under m2h the program's lead is of the same size as its margin over the never-flip reference (≈ 1–3 points); recall / Jaccard of the tuned GBDT under m2h were not computed.
- Agreement and Jaccard rank prog30 and the lr 0.2 GBDT in opposite order on trans; B6's comparisons use agreement only.
- `b6-verify2` peaked at 2,047 MiB of a 2 GiB request; a rerun with more models should request 3 GiB.
- `pt09` in the transformer: prog30 (85.4 %) is below never-flip (94.9 %).

## Corrected reading
At about 13 kbit the blind rank-window MARS program agrees with the networks' call changes under the 0.06 rad translation about as often as a tree / GBDT / MLP of at most the same bits selected on dev jets: differences of −1.3 to +1.7 points across three jet samples in JEDI-linear and PFN, +0.2 (Deep Sets) and +1.6 (transformer) on B6's jets; with test-edit (oracle) selection the baseline is 0 to 2.4 points higher in all four networks. Against B6's single GBDT setting (lr 0.2) the program is 2.8 to 8.1 points higher in all four networks and all three samples, but that GBDT has a higher flipped-set Jaccard than the program, so the lead depends on requiring the same new call. Under the M2-held translation the program keeps a lead over the dev-selected baseline in JEDI-linear in all three samples (+1.7 to +2.0 points; Holm ≤ 6e-5) and in the transformer on B6's jets (+1.7), not in PFN or Deep Sets; under this edit all surrogates at ≤ 14k bits are within about 3 points of the never-flip reference. Larger GBDTs and MLPs (4–124× the bits) reach 5–12 points higher translation agreement.

---

# B6 verification: matched-description-length edit fidelity, fresh jets, tuned GBDT and selection rules (2026-10-19)

## Conclusion
On fresh jets, the B6 result reproduces when the program is compared with the same boosted-tree setting that B6 used (4 leaves, 10 rounds, learning rate 0.2): about +5 points on the 0.06 rad shift for JEDI-linear and +4 for PFN. That setting is under-fitted for its size. A boosted-tree model with the same number of bits but a learning rate of 0.5, selected on held-out jets, agrees with the network's call changes under the shift as often as the 30-term program (JEDI-linear −0.1 points, PFN −0.8 points, neither significant). The program's lead under the M2-held shift is smaller but still present in JEDI-linear (+2.0 points); in PFN it is +0.7 and not significant after correction. "The program agrees with the networks' call changes more often than the best tree or boosted-tree model of equal size" is therefore not supported for the shift edit. Deep Sets and the transformer were not rerun.

## Procedure
- Code review of `interp/b6_pareto.py` (all four modes):
  - `indices()` reproduces `jedi_core.sample()`'s row selection (same rng calls), so the removal of F and `sample(6000, 0)` rows from T is correct. There is no fit/test overlap.
  - Edits are multiplied by the real-particle mask (pT > 0), so padding is never moved. pt09 scales only real pT (padding is 0). A13 swaps only (Δη, Δφ) of ranks 13 and 14 in jets with ≥ 15 particles. m2h rescales the angles by √(M2/M2′), so M0 and M2 are held.
  - The bootstrap resamples are shared across the test set and the edits of the same length, so the paired CIs are paired.
  - The MARS logit offset c cancels in the argmax (the refitted intercept absorbs it), so it costs no bits.
  - No defect in stages, masks or splits was found. The defect is in the baseline: the matched-DL comparison uses one GBDT hyper-parameter setting (lr 0.2) at the overlap size.
- Re-analysis of the stored B6 results (`results/jedi/b6.json`, no new fits). Among *all* tree, GBDT and MLP surrogates with bits ≤ prog30's, ranked by test-edit agreement (oracle selection), on trans:
  - JEDI: CART on logits d6 (12.7k bits) 58.73 % vs prog30 57.99 %.
  - Deep Sets: CART d6 62.97 vs 63.19.
  - PFN: CART d6 56.84 vs 57.74.
  - transformer: MLP h4 (13.5k bits) 52.92 vs 51.20.
  - B6's matched rule (highest dev same call) instead picked GBDT L4 T10 or CART d7–d9, which are lower on the edit.
- New job `b6-verify` (`interp/b6_verify.py`): 1 CPU / 2 GiB, wall 414 s, peak 1,222 MiB, 0.91 cores, exit 0. Output: `$AI_RESULTS/jedi/b6_verify.json`, copied to `results/jedi/b6_verify.json`; log `logs/b6-verify.log`.
  - Models: JEDI-linear n64 (jedi_core) and PFN 100-100-128 s0. Deep Sets and the transformer need the torch env and were not rerun.
  - Jets: F = `sample(4000, 3701)`: 13,333 fit / 6,667 dev (perm rng 0). T = `sample(2000, 3702)` minus every row of F, `sample(6000, 0)` and B6's `sample(4000, 2601)`: N = 7,444.
  - Same 98 observables, MARS program (K 16, 30), bit costs and edit definitions as B6 (imported from `b6_pareto`). The edits are also applied to the dev jets.
  - Baselines with bits ≤ the program's:
    - CART d2–10, on the call and on the logits;
    - MLP h2, h4;
    - a GBDT grid on the call and on the logits: leaves {2, 3, 4, 8} × learning rate {0.1, 0.2, 0.5, 1.0} × rounds up to the bit budget ({5, 10, 20, 26}, {5, 10, 15}, {5, 10}, {2, 5}). The grid includes B6's L4 T10 lr 0.2.
  - Matched baseline, chosen by one of four rules:
    - **devsc**: highest dev same call (B6's rule) over the extended grid;
    - **devedit**: highest dev agreement under the same edit (no test jets used);
    - **oracle**: highest test agreement (an upper bound for the baseline);
    - **b6gbdt**: B6's GBDT L4 T10 lr 0.2, the replication.
  - Statistics: paired jet bootstrap (B = 1000) of the per-jet agreement difference; exact McNemar; Holm over 42 tests (2 models × 2 programs × trans/m2h/pt09 × rules with a distinct test). H0: equal per-jet agreement; H1: different.
  - Specificity: paired bootstrap of [lead on trans] − [lead on rdt].

## Numbers
Network accuracy on the new T: JEDI 80.84 %, PFN 81.03 %. Calls changed: trans 56.3 / 55.8 %, m2h 16.7 / 16.7 %, pt09 33.8 / 31.1 %, rdt 59.5 / 61.4 %, rpd 52.8 / 55.8 % (JEDI / PFN). These match B6.

**prog30 (12.9k / 13.1k bits) minus the matched baseline, agreement in points [95 % CI], Holm p**

| model | edit | b6gbdt (L4 T10 lr 0.2) | devsc (extended grid) | devedit |
|---|---|---|---|---|
| JEDI | trans | +5.02 [4.08, 5.92], 3e-24 | −0.12 [−1.09, 0.77], 1 (GBDT L4 T10 lr 0.5) | −0.12 (same) |
| JEDI | m2h | +2.90 [2.30, 3.55], 4e-17 | +2.03 [1.38, 2.69], 6e-8 | +1.89 [1.32, 2.51] (L2 T20 lr 0.5) |
| JEDI | pt09 | +4.66 [3.63, 5.60], 2e-17 | +1.02 [0.05, 1.96], 0.81 | +1.02 |
| JEDI | rdt (control) | +4.15 [3.30, 5.01] | −1.25 [−2.27, −0.26] | −1.25 |
| JEDI | rpd (control) | +0.32 [−0.55, 1.18] | −2.74 [−3.68, −1.84] | −2.74 |
| PFN | trans | +4.30 [3.37, 5.20], 2e-17 | −0.79 [−1.71, 0.07], 0.95 (GBDT L3 T15 lr 0.5) | −0.79 |
| PFN | m2h | +1.64 [0.99, 2.27], 3e-5 | +0.71 [0.00, 1.38], 0.81 | +0.71 |
| PFN | pt09 | +3.12 [2.12, 4.14], 6e-8 | +1.75 [0.78, 2.71], 0.011 | +1.61 [0.64, 2.53] |
| PFN | rdt (control) | +3.98 [3.06, 4.86] | +0.05 [−0.89, 0.91] | +0.05 |
| PFN | rpd (control) | +0.12 [−0.82, 1.02] | −1.73 [−2.58, −0.90] | −1.73 |

The oracle rule gives the same baseline or one within 0.1 points, except PFN m2h (+0.60 [−0.07, 1.25]).

**Raw agreement on trans, JEDI.**
- prog30: 59.07 %.
- GBDT L4 T10, lr 0.2 vs lr 0.5 (both 12,192 bits): 54.04 % vs 59.19 %.
- Same call on unedited T: prog30 90.99 %; GBDT lr 0.2 89.31 %; GBDT lr 0.5 90.44 %.

**prog16 (6.6k bits) vs the matched baseline on trans.**
- devsc (GBDT L4 T5 lr 0.5): JEDI +2.66 [1.61, 3.61], Holm 1.5e-6; PFN +3.20 [2.23, 4.04], Holm 2e-9.
- devedit: JEDI −0.55 [−1.59, 0.54] (GBDT L2 T10 lr 1.0); PFN +1.16 [−0.01, 2.26] (CART d7).
- m2h, devedit: JEDI +1.36 [0.67, 2.06]; PFN +0.97 [0.32, 1.67].

**Specificity, prog30, [lead on trans] − [lead on rdt].**
- JEDI: +1.13 [0.01, 2.20], p 0.04 (devsc).
- PFN: −0.85 [−1.84, 0.11], p 0.09.
- b6gbdt: JEDI +0.87 [−0.08, 1.83]; PFN +0.32 [−0.66, 1.26].

**Competing hypotheses.**
- H_prog ("the rank-window program follows the shift better than any baseline of equal DL") predicts a positive lead under every selection rule. It is excluded for trans at 13k bits under devsc and devedit in both models: the CIs include 0 and lie below +1.0.
- H_tuning ("B6's lead comes from an under-fitted GBDT setting") predicts that the lead vanishes with a tuned learning rate at the same bits. This is observed for trans and rdt, and partly for pt09 and m2h.

## Figure check
`results/jedi/b6_trans.png`:
- Four panels, log2 bits on the x axis, agreement on the y axis, never-flip lines; the legend is readable.
- The figure shows the issue directly. In the JEDI, Deep Sets and PFN panels, the CART frontier (green) reaches the same height as prog30 at 12.7k bits (CART on logits, d6). In the transformer panel, the MLP point at about 13.5k bits is above prog30.
- The "never-flip" labels overlap the line in two panels (cosmetic).

## Issues
- Deep Sets and the transformer were not rerun: their logits need the torch env and a second job. The re-analysis of stored B6 results already shows a baseline at ≤ the program's bits matching or exceeding prog30 on trans in JEDI and the transformer.
- The GBDT grid is small (4 learning rates). A finer search would favour the baselines further, not the program.
- Bits are B6's 32-bit convention. Quantised GBDT thresholds would halve GBDT bits, which also favours the baseline.

## Corrected reading
At about 13 kilobits, the blind rank-window MARS program agrees with the network's call changes under the 0.06 rad translation:
- more often than B6's single GBDT setting (lr 0.2): replicated on fresh jets, +5.0 JEDI, +4.3 PFN;
- not more often than a GBDT of equal bits with learning rate 0.5 selected on dev jets: −0.1 [−1.1, 0.8] JEDI, −0.8 [−1.7, 0.1] PFN.

Under the M2-held translation, the program keeps a lead over the tuned GBDT in JEDI-linear (+2.0 [1.4, 2.7]). In PFN the lead is +0.7 [0.0, 1.4], Holm 0.81.

The transfer statement holds only in the form "the program reaches tree/GBDT-level edit fidelity at equal DL in all four networks". It does not hold as "exceeds".
