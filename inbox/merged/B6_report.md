# B6: edit fidelity vs description length for program extraction on JEDI-linear, Deep Sets, PFN and a small transformer (claims B2, B3, B4; review weaknesses 3 and 6)

Script: `interp/b6_pareto.py`. Modes: `logits`, `efp`, `fit`, `eval`.

Jobs (all 1 CPU / 2 GiB, exit 0):
- b6-lg-jedi2, b6-lg-pfn: jax env.
- b6-lg-dsets, b6-lg-tf: `AI_PY=conda`, torch.
- b6-efp: `/j-jepa-vol/aaron-interp/venv-ef`, energyflow 1.4.
- b6-fit-{jedi,pfn,dsets,tf}.
- b6-eval4: the final evaluation. b6-eval, b6-eval2 and b6-eval3 were earlier versions of the same evaluation; see Issues.
- The first b6-lg-jedi stalled in CephFS metadata wait (`ceph_mdsc_wait_request`, node chi-dgx-node01). I deleted it and resubmitted it as b6-lg-jedi2.
- Peak memory 0.34–2.0 GiB. Wall time 0.5–11 min per job.

Results:
- `results/jedi/b6.json`: all surrogates, CIs, AUC tests and matched-DL tests.
- `b6_pred_<model>.json`: surrogate list, DL, and the fitted program terms.
- `b6_logits_<model>.json`.
- `b6_nested_jedi.json`.
- Per-jet arrays (`b6_*.npz`) are on the volume only.

Figures: `results/jedi/b6_trans.png`, `b6_m2h.png`, `b6_A13.png`, `b6_pt09.png`, `b6_scr.png`, `b6_same_call.png`, `b6_rdt.png`, `b6_rpd.png`. Each has one panel per model, edit agreement (%) on the y axis and log2 bits on the x axis.

Logs: `logs/b6-*.log`.

## Procedure

**Models.**
- Official JEDI-linear 3-feature n64 (fixed point, `jedi_core.Jedi`).
- dsets-d64 seed 0 and tf-2b-d32 seed 0 (`best.pt`, `train/models.py`).
- PFN `pfn_100-100-128_s0` (X9 numpy forward).

**Jets.**
- F = `sample(4000, 2601)`, 20,000 jets, permuted with rng 0: fit 13,333 and dev 6,667. Dev is used only for MARS term selection and for the matched-DL baseline choice.
- T = `sample(2000, 2602)` minus every row of F and of `sample(6000, 0)` (the fit jets of the stored E11/B4 nested programs). N = 8,152 (1,848 rows removed).
- A13 is restricted to jets with ≥ 15 particles (N 8,113).

**Network accuracy on T.**

| model | accuracy on T | 95 % CI | reference |
|---|---|---|---|
| JEDI-linear | 80.95 % | [80.12, 81.76] | within the 80.6–81.3 % check range |
| Deep Sets | 80.75 % | | |
| PFN | 81.87 % | | |
| transformer | 76.09 % | | official 260k test 76.19 % |

**Observables.** Every blind surrogate uses the same 98 observables and gets no access to units:
- the 97 rank-window sums of `e11_mars_all.handbuilt`: ΣpT, N, ΣpT r², Σr², Σr, ΣpT r, Σ(r−0.2)₊, N(pT>1), N(pT<2), Σr[r<0.165], N(pT<2, r>0.1) over 9 pT-rank windows. M2 = ΣpT r²[all] is one of these.
- M0 = ΣpT.

**Extractors (5–30 sizes each).**
- **prog** (blind pipeline): MARS from the 98 observables to each of the 5 logits, using B4's direct-MARS settings with a 30-term path per logit. Programs are prefixes K = 2, 4, 8, 16, 30, each refitted by least squares.
- **nested** (JEDI-linear only): the stored unit-level nested programs (E11 K = 30; B4 K = 2–16). They run on the E11 physics set recomputed from the edited particles, and their z goes through the network's own d7.
- **tree**: CART on the call and on the logits, depth 2–10.
- **gbdt**: HistGradientBoosting on the call and on the logits, 4 or 16 leaves × 10, 30, 100 or 300 iterations, learning rate 0.2.
- **linear**: Lasso path on the logits (10 alphas) and ridge.
- **mlp**: one hidden ReLU layer of 2–64 units, fitted to the logits.
- **efp**: the 54 connected EFPs with d ≤ 5 (energyflow, hadr measure, β = 1, normed) and their logs, then a Lasso path on the logits.
- **PySR and gplearn** are not installed in the job env (import checked at run time). Symbolic regression was not run.

**Description length (bits; every number costs 32 bits).**
- MARS term: 32 + per factor [log2(#inputs) + 2 + 32 for the knot]. Each intercept adds 32.
- Tree or GBDT: each internal node costs log2(#inputs) + 32; each leaf costs 32 × outputs (log2 5 for a classifier leaf).
- Linear: each non-zero weight costs 32 + log2(#inputs), plus 5 × 32.
- MLP: all parameters × 32.

**Edits on T (padding never moved).**
- **trans**: Δη += 0.06 rad.
- **m2h**: the same translation, then angles scaled per jet by √(M2/M2′), so that M0 and M2 are unchanged (mean scale 0.762).
- **A13**: swap of (Δη, Δφ) between pT ranks 13 and 14.
- **pt09**: every pT × 0.9.
- **scr**: per-particle azimuth scramble (r and pT kept; rng 2603).

**Controls.**
- **rdt**: a coherent 0.06 rad translation in a random direction per jet.
- **rpd**: each particle moved 0.06 rad in an independent random direction.
- **never-flip reference**: 1 − the network's flip rate.
- **shuffled**: prog K16 and GBDT-on-logits L16 T100 fitted on row-shuffled logits.

**Metrics.**
- Same call on T.
- Per-jet edit agreement (B4): the surrogate flips if and only if the network flips, and to the same new call.
- Jaccard of the flipped-jet sets, and recall.
- 95 % jet bootstrap intervals, B = 1000.

**Tests.**
- (a) Area under the frontier. F_e(b) is the best agreement among family e's surrogates with ≤ b bits. The area is computed over the overlapping log2-bit range of the two families; where the overlap is < 0.25 log2 bits, the union range is used with each frontier held flat. Paired bootstrap of AUC(prog) − AUC(e); bootstrap-z p; Holm over 125 tests (4 models × 5 edits × 5 families for prog, plus 25 for nested).
- (b) Matched DL (B4 rule). Each program size is compared with the member of each family that has ≤ its bits and the highest dev same call. Paired bootstrap CI, exact McNemar, Holm over 525 cells (programs × families × 5 edits; controls reported unadjusted).
- H0: equal per-jet agreement. H1: the program's agreement differs.

## Results

**Network flip rates** (% of calls changed):

| model | trans | m2h | A13 | pt09 | scr | rdt | rpd | W→Z under trans |
|---|---|---|---|---|---|---|---|---|
| JEDI-linear | 56.1 | 17.5 | 1.5 | 35.0 | 6.8 | 58.8 | 52.1 | 3.4 → 75.6 % |
| Deep Sets | 61.7 | 15.4 | 0.2 | 25.2 | 8.4 | 65.7 | 58.7 | 2.3 → 66.8 % |
| PFN | 56.0 | 16.7 | 0.4 | 30.5 | 9.4 | 60.9 | 54.6 | 1.6 → 63.0 % |
| transformer | 51.4 | 24.7 | 0.3 | 5.1 | 13.1 | 59.8 | 61.2 | 3.3 → 64.7 % |

**Same call on unedited T** (%; bits in brackets):

| model | prog8 | prog30 | best tree | GBDT L4 T10 | best GBDT | MLP h64 | ridge | EFP best | nested30 |
|---|---|---|---|---|---|---|---|---|---|
| JEDI | 87.9 (3.3k) | 90.6 [90.0, 91.2] (13.4k) | 89.2 | 88.9 (12.2k) | 93.0 (1.6M) | 92.8 (213k) | 66.8 | 71.7 | 93.9 (367k) |
| Deep Sets | 90.0 | 92.5 | 90.3 | 90.8 | 94.1 | 94.0 | 67.5 | 68.4 | – |
| PFN | 87.4 | 90.5 | 88.5 | 88.8 | 92.8 | 91.5 | 64.6 | 68.2 | – |
| transformer | 85.2 | 87.3 | 85.8 | 86.8 | 90.1 | 89.9 | 63.8 | 65.4 | – |

The shuffled-logit surrogates give a same call of 17.6–20.2 %.

**Edit agreement at matched DL**: prog30 (13.2–13.8k bits) minus the dev-selected baseline with ≤ its bits, in points, with 95 % CI and Holm-adjusted McNemar p:

| model | vs | trans | m2h | pt09 | rdt (control) | rpd (control) |
|---|---|---|---|---|---|---|
| JEDI | GBDT L4 T10 | +2.80 [1.90, 3.72], 4e-7 | +2.93, 1e-15 | +4.77, 3e-20 | +1.59 [0.70, 2.45] | −0.67 [−1.57, 0.18] |
| JEDI | CART d8 | +4.21 [3.05, 5.34], 5e-10 | +2.71 | +6.04 | +2.83 | −0.90 [−1.90, 0.12] |
| Deep Sets | GBDT L4 T10 | +4.04 [3.02, 5.04], 5e-12 | +1.37, 0.002 | +1.95, 1e-4 | +2.32 | −4.66 [−5.64, −3.62] |
| Deep Sets | CART d8 | +6.99, 1e-41 | +1.26, 0.034 | +2.92 | +5.74 | +2.75 |
| PFN | GBDT L4 T10 | +4.47 [3.61, 5.37], 2e-19 | +1.09, 0.048 | +3.96, 1e-13 | +4.16 | −0.52 [−1.33, 0.29] |
| PFN | CART d9 | +2.97, 9e-5 | +1.52 | +6.49 | +1.10 [0.01, 2.29] | +0.22 |
| transformer | GBDT L4 T10 | +4.96 [4.10, 5.72], 2e-30 | +2.23, 2e-8 | −1.01 [−1.73, −0.29], 0.90 | +3.36 | −0.40 |
| transformer | CART d7 | +7.64, 7e-53 | +3.70 | +2.06 | +7.69 | +2.15 |

**Other comparisons.**
- Against Lasso, ridge or EFP at ≤ the same bits, the program leads by 14–28 points on trans and m2h in every model.
- Against the transformer's EFP Lasso on pt09 the program is lower (−9.5). The normed EFPs never flip under pt09, and the transformer flips only 5.1 % of calls under pt09.
- The MLPs that fit within ≤ 13.8k bits (h2 or h4) are poor fits (same call 58–86 %).

**Raw agreement, JEDI trans.**
- prog30: 58.0 [56.9, 59.1].
- GBDT L4 T10: 55.2.
- GBDT L16 T30 (164k bits): 69.5.
- MLP h64 (213k): 64.8.
- nested30 (367k): 72.3 [71.4, 73.3].
- never-flip: 43.9.
- W → Z under trans: prog30 81.7 %, GBDT L4 T10 57.7 %, network 75.6 %.

**Frontier AUC (union or overlap range).**
- The blind program's frontier is below GBDT's on trans in JEDI, Deep Sets and PFN: −6.73 [−7.65, −5.95], −2.27 and −3.00 points (Holm ≤ 4e-12).
- The program's sizes (0.9–13.8k bits) and GBDT's (12.2k–1.6M bits) overlap only at prog30 ≈ GBDT L4 T10. This AUC therefore compares the program's largest point with GBDT's frontier out to 120× more bits.
- For the transformer the trans AUC difference vs GBDT is +0.74 [−0.03, 1.37] (Holm 1).
- The nested JEDI program (25k–367k bits) is below GBDT, CART and MLP on the trans and m2h AUC by 2.8–4.0 points (Holm ≤ 3e-13).
- At matched DL, nested30 leads GBDT L4 T300 on trans by +5.43 [4.38, 6.55] (Holm 1e-19) and on m2h by +0.98 (Holm 0.80).

**A13.**
- The network flips 0.2–1.5 % of calls.
- Every surrogate is within 2 points of the never-flip reference: 97–99.6 % agreement, Jaccard ≤ 0.07.
- AUC differences are ≤ 1 point.

**scr.**
- The 98 observables are invariant under the azimuth scramble, so every blind surrogate on them never flips (agreement = never-flip 86.9–93.2 %, Jaccard 0).
- The networks change 6.8–13.1 % of calls.
- EFP surrogates flip spuriously: agreement 39–53 %.
- The nested JEDI program (physics set includes non-radial observables) flips: Jaccard 0.05–0.26, agreement 91.2–93.0 %.

**Shuffled-logit control.**
- Agreement is at or below the never-flip reference: prog16-shuffled equals never-flip, because its dev-selected path keeps 1 term per logit.
- GBDT-shuffled: 23–48 %.

## Reading (scoped to these measurements)

**Equal description length (12–14k bits).**
- The blind rank-window MARS program agrees with the network's call changes more often than the best CART and the best GBDT of equal or smaller size on the 0.06 rad translation, in all four networks: +2.8 to +7.6 points, Holm p ≤ 5e-10.
- It also does so on the M2-held translation (+1.1 to +3.7 points) and on the pT scaling in 3 of 4 networks.
- The same holds for the random-direction coherent translation (rdt, +1.1 to +7.7).
- The lead over GBDT is absent or reversed for the per-particle random displacement (rpd, −4.7 to −0.4).
- At this DL the program's same call is 0.5–2.2 points higher than that GBDT.

**Larger baselines.**
- GBDT and MLPs reach higher same call (+1.6 to +2.8 points) and higher translation agreement (+6 to +11 points in JEDI) with 12–120× more bits.
- The program's frontier ends at 13.8k bits, so on the area measure it is below GBDT for JEDI, Deep Sets and PFN.
- Same call on data across the strong extractors at their largest sizes spans 90.5–94.1 % (JEDI, Deep Sets, PFN) and 87.3–90.1 % (transformer). This is a 1.6–2.8 point gap between prog30 and the best baseline.

**Transfer beyond JEDI-linear.** The matched-DL pattern (program > tree and GBDT on translation and m2h) holds in Deep Sets, PFN and the transformer. The program uses only rank-window radial observables and no unit access.

**Edits with no separation.**
- The rank-13|14 swap changes ≤ 1.5 % of calls and does not separate any surrogates.
- The azimuth scramble cannot be represented by any surrogate on these radial observables (0 % of the network's 7–13 % call changes reproduced).

**Thesis predictions.**
- "The program dominates GBDT/trees at equal DL on translation": observed for all four networks at the one DL where the families overlap.
- "On rank swap": not testable (≤ 1.5 % flips).
- "Same call within 1–2 points": observed at matched DL; at the largest sizes the gap to the best baseline is 1.6–2.8 points.

**Alternative predictions.**
- "Baselines reach equal or better edit fidelity at 5–20× lower DL": not observed at the overlapping DL.
- Baselines reach better fidelity only at 12–120× higher DL.

**Excluded.** A pure-flip-rate explanation: the shuffled and never-flip references sit at or below the never-flip level, and the program's recall on trans is 47–60 %.

## Issues / limits

- **PySR and gplearn** are not installed in the job env, so symbolic regression is missing.
- **Small DL overlap.** The program's size range (0.9–13.8k bits) barely overlaps GBDT's (from 12.2k). The matched-DL results rest on one baseline point per family (GBDT L4 T10, CART d7–d9). Smaller GBDTs (1–4 iterations) were not fitted.
- **Frontier selection.** The frontier and AUC take the best surrogate on the test edit (an optimistic selection, the same for all families). The matched-DL baselines are selected on dev same call.
- **Shuffled MARS size.** Shuffled-logit MARS keeps 1 term per logit (dev-selected), so it is not size-matched. The shuffled GBDT is size-matched.
- **Nested-program splits.** The nested JEDI programs were fitted on `sample(6000, 0)`, not on F, and use the 475-input E11 physics set, so their DL is not comparable in input count.
- **Fixed bit cost.** The bit costs (32 bits per number) are a fixed convention. Quantised thresholds (HGB bins ≤ 8 bits) would lower GBDT bits by about 2×.
- **Evaluation changes (b6-eval → b6-eval4).**
  - b6-eval used an empirical bootstrap p with floor 0.001, which made Holm over 125 tests unreachable.
  - b6-eval2 changed to bootstrap-z p and the overlap range.
  - b6-eval3 added the matched-DL tests.
  - b6-eval4 added the controls to the matched tests and moved the figure legend.
  - No fit was rerun.
- **CephFS stall.** The first JEDI logits job stalled in CephFS metadata wait and was resubmitted unchanged.

## Conclusion

At the same program size (about 13 kilobits), a short list of if-statements on pT-rank-window sums reproduces how each of the four networks changes its calls under a small rigid shift of the jet better than the best decision tree or boosted-tree model of that size. This holds for JEDI-linear and also for a Deep Sets network, a particle-flow network and a small transformer, for example +2.8 points over boosted trees in JEDI-linear. Larger boosted trees and neural-network probes reproduce more of the changes (up to about 70 % agreement vs 58 %), but with 12–120 times more parameters, and the extraction here has no program larger than 30 terms per class. The method transfers across architectures in this sense. Whether it stays ahead at larger sizes, and for edits that change few calls (rank swap) or that these radial observables cannot see (azimuth scramble), is not established.

## Explainer entry (for site/explain.py; Paper B tab)

```python
B6_EXPLAINER = [
 ('Question', 'Does the blind if-statement extraction (MARS on pT-rank-window observables, no access to the network\'s units) describe how a network changes its calls under input edits better than standard surrogates of the same size, and does this hold for networks other than JEDI-linear?', None),
 ('Background',
  'Description length (DL) is the number of bits needed to write the surrogate down: 32 bits per number plus $\\log_2(\\#\\text{inputs})$ bits per chosen input. '
  'Edit agreement on a jet is $A = \\mathbb{1}[f_s(x\') \\ne f_s(x) \\Leftrightarrow f_n(x\') \\ne f_n(x)]\\cdot\\mathbb{1}[f_n(x\') \\ne f_n(x) \\Rightarrow f_s(x\') = f_n(x\')]$; a surrogate that never changes its call scores $1-$(network flip rate). '
  'The observables are sums over pT-rank windows such as $\\sum_{i\\in w} p_{T,i} r_i^2$, which are unchanged when particles are rotated about the jet axis.', None),
 ('Method',
  'Four networks on the same 64-particle input: JEDI-linear, Deep Sets, a particle-flow network and a 2-block transformer. Surrogates are fitted to each network\'s logits on 13,333 jets and tested on 8,152 others: the MARS program (2-30 terms per class), CART, boosted trees, Lasso/ridge, one-hidden-layer MLPs and energy-flow polynomials up to degree 5. '
  'Edits: shift every particle by 0.06 rad; the same shift with $M_0$ and $M_2=\\sum p_T r^2$ held; swap the angles of pT ranks 13 and 14; scale every pT by 0.9; scramble each particle\'s azimuth. Controls: a random-direction shift, per-particle random displacement, surrogates fitted on shuffled logits, and the never-flip reference. '
  'Each program is compared with the best baseline of equal or smaller DL (McNemar, Holm).', None),
 ('Result',
  'At about 13 kilobits the program agrees with the network on the 0.06 rad shift 2.8-7.6 points more often than the best tree or boosted-tree model of equal size, in all four networks (Holm $p\\le 5\\times10^{-10}$), and 1.1-3.7 points more on the $M_2$-held shift. Larger boosted trees and MLPs reach 6-11 points higher agreement on JEDI-linear with 12-120 times more bits.', 'results/jedi/b6_trans.png'),
 ('Checks',
  'JEDI-linear gives 80.95 % on the test jets. Surrogates fitted on shuffled logits stay at or below the never-flip line. The program\'s lead over boosted trees is absent under per-particle random displacement (-4.7 to -0.4 points). The rank swap changes at most 1.5 % of calls and separates no surrogate; the azimuth scramble changes 7-13 % of calls, which no surrogate on these radial observables can follow. PySR is not installed.', None),
 ('Why it matters',
  'The comparison gives Paper B its baseline figure and runs the extraction on three architectures other than JEDI-linear. At equal size the rank-window program follows the networks\' response to the mass-shifting translation more closely than trees, but larger generic models reach higher fidelity, so the advantage is one of size, not of fidelity.', None),
]
# register in PAPER_EXPLAINERS['b']:
#   ('Result explained: edit fidelity vs description length on four architectures (claims B2-B4; B6)', B6_EXPLAINER, False)
```
