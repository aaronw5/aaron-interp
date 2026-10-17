# B4 verification: fresh-sample rerun of the translation and same-call comparisons, with a same-baseline control test

Script: `interp/b4_verify.py` (`python -m interp.b4_verify 23`). Job: b4-verify, 1 CPU / 2 GiB, exit 0, wall 532 s, pod peak memory 2048 MiB (page cache included; no OOM).
Results: `results/jedi/b4_verify.json`. Log: `logs/b4-verify.log`.

## Procedure

- **Fit split.** Identical to B4/E11: `sample(6000, seed 0)`, permutation rng 0, first 2/3 (20,000 jets). Nothing is fitted or selected on the fresh jets.
- **Models.**
  - Nested programs mars2–mars30 and zphys30 are read from the stored formula files.
  - Direct MARS uses the stored term lists (`b4_direct_terms.json`), with the least-squares coefficients refitted on the fit split as B4 did.
  - The tree and GBDT baselines are refitted with B4's settings and random_state 0. They are the dev-selected and B4-test-selected ("oracle") baselines B4 named, plus the models in B4's reach table.
- **Check 0 (reproduction).** On the B4 test split (seed 11, N 8,848), every reloaded or refitted model gives exactly B4's stored same-call. Examples: mars16 92.92, mars30 94.02, gbdt_L16_T16 92.88, gbdtreg_L32_T16 92.48, direct100 92.91.
- **Fresh test.** `sample(2000, seed 23)` minus every jet of seeds 0, 11 and 91, leaving N = 8,218 (1,782 removed).
  - Physics recomputed on the unedited jets matches the stored values to a maximum relative difference of 3.2e-9. No non-finite inputs.
- **Edits on the fresh jets.**
  - `trans`: Δη += 0.06 for every real particle (B4's edit).
  - `transphi`: Δφ += 0.06 for every real particle. This is a second coherent shift, run as a replicate.
  - `trans_c`: each real particle moved 0.06 rad in an independent random direction (B4's control, rng 101).
- **Statistics.**
  - Agreement and recall are as in B4. 95 % bootstrap CIs over jets (B = 1000).
  - McNemar tests on per-jet agreement, BH-corrected over 15 cells.
  - New test, the difference of leads for a fixed baseline: per jet, $[\,a_n-a_b\,]_{\rm trans}-[\,a_n-a_b\,]_{\rm trans\_c}$, with a paired bootstrap CI and a two-sided bootstrap p.
    - H0: the nested program's lead over that baseline is the same on the coherent shift and on the random-direction control.
    - H1: the leads differ.

## Script audit (b4_baselines_jedi.py)

- **Edits.**
  - `edit_shift` and `randdisp` move only real particles (pT > 0).
  - `rotate` acts on all slots, but padded slots have zero angles and stay zero.
  - The A13 swap moves only (Δη, Δφ) of slots 13 and 14.
  - `ramp` keeps the pT rank order (previous slot ≥ hi, next slot ≤ lo).
- **Physics.** Recomputed on the edited particles, including the hand-built H sums, which use the un-recentred angles.
- **Nested logits.** Come from the network's own d7, including its input quantizer (`e11_verify.Last`).
- **Splits.** I found no leakage. Fit and dev come from seed 0, and test, confirmation and fresh are disjoint from it.
- **Paired tests.** The same bootstrap index matrix is used for every model within a set.
- **Baseline asymmetry in the control comparisons.** The baseline differs between comparisons:
  - The trans_c "oracle" is chosen per edit. For mars16 it is gbdt_L16_T16 on trans_c, while on trans it is gbdtreg_L32_T8.
  - So B4 compares the translation lead and the control lead against different models.
  - No code bug was found; the issue is in the inference drawn (see below).

## Results (fresh N 8,218)

### Network

- Accuracy: 81.27 % [80.40, 82.08].
- Flip rates:
  - trans: 55.51 % [54.49, 56.51]
  - transphi: 56.11 % [55.06, 57.07]
  - trans_c: 52.15 % [51.05, 53.21]
- True-W jets called Z: 77.9 % (trans), 78.1 % (transphi), 49.2 % (trans_c).

### Same call

| model | DL | same call, fresh | B4 test |
|---|---|---|---|
| mars16 | 2,219 | 93.14 [92.59, 93.67] | 92.92 |
| mars30 | 3,824 | 94.06 [93.53, 94.56] | 94.02 |
| gbdt_L32_T128 | 20,454 | 93.68 [93.14, 94.21] | 93.86 |

Paired same-call differences:

- mars16 − gbdt_L16_T16: +0.22 [−0.30, 0.73]. B4: +0.05.
- mars30 − gbdt_L4_T128: +0.86 [0.37, 1.40]. B4: +0.90.
- gbdt_L32_T128 − mars30: −0.38 [−0.88, 0.11].

Reach (baseline − nested same call, fresh):

| baseline (DL) | nested (DL) | difference |
|---|---|---|
| tree_d4 (15) | mars2 (286) | +2.46 [1.59, 3.25] |
| direct8 (40) | mars4 (570) | +0.21 [−0.50, 0.91] |
| direct16 (80) | mars8 (1,123) | +0.35 [−0.28, 1.00] |
| gbdt_L8_T32 (1,280) | mars16 (2,219) | +0.04 [−0.50, 0.54] |
| direct100 (485) | mars16 (2,219) | −0.28 [−0.84, 0.26] |

The DL ratio at equal same-call:

- 19× for mars2 (tree_d4).
- 14× for mars4 and mars8 (direct MARS).
- 4.6× for mars16 (direct100, within CI) or 1.7× (GBDT).

### Translation agreement and recall, fresh

| model | trans agree | trans recall | transphi agree | trans_c recall | W → Z (trans) |
|---|---|---|---|---|---|
| mars16 | 70.42 [69.4, 71.4] | 61.3 [59.9, 62.8] | 69.85 | 61.9 | 86.4 % |
| mars30 | 72.15 [71.2, 73.1] | 68.7 [67.4, 70.0] | 72.78 | 71.1 | 77.6 % |
| gbdtreg_L32_T8 | 67.62 | 50.5 [49.1, 52.0] | 67.86 | 56.9 | 69.1 % |
| gbdtreg_L32_T16 | 68.86 | 51.8 [50.3, 53.2] | 69.06 | 59.3 | 74.5 % |
| gbdt_L32_T128 (DL 20,454) | 63.34 | 37.9 | 61.90 | 60.2 | 72.9 % |
| direct100 | 38.79 | 23.4 [22.2, 24.6] | 38.32 | 56.1 [54.7, 57.5] | 0.0 % |
| zphys30 | 50.11 | 18.1 [17.1, 19.2] | 48.89 | 57.1 [55.8, 58.6] | 0.1 % |
| tree_d6 (on call) | 53.14 | 51.3 | 54.42 | 58.0 | 66.7 % |

### Paired comparisons, nested − baseline, agreement points

Baselines are those B4 named. BH q is over 15 cells.

| nested vs baseline | trans | transphi | trans_c | lead difference trans − trans_c | transphi − trans_c |
|---|---|---|---|---|---|
| mars16 vs gbdt_L16_T16 (B4 dev-selected; B4 trans_c oracle) | +15.67 [14.46, 16.90], q 5e-130 | +16.09, q 4e-132 | +0.46 [−0.44, 1.50], q 0.36 | +15.21 [13.74, 16.67], p < 0.001 | +15.62, p < 0.001 |
| mars16 vs gbdtreg_L32_T8 (B4 trans oracle) | +2.80 [1.69, 3.91], q 4e-7 | +1.98 [0.95, 2.93], q 3e-4 | **+2.66 [1.73, 3.65], q 7e-8** | **+0.13 [−1.29, 1.52], p 0.86** | −0.68 [−1.96, 0.62], p 0.30 |
| mars30 vs gbdt_L4_T128 (B4 dev-selected) | +15.37, q 2e-115 | +16.84, q 2e-142 | +4.59 [3.60, 5.62], q 6e-18 | +10.78 [9.24, 12.27], p < 0.001 | +12.25, p < 0.001 |
| mars30 vs gbdtreg_L32_T16 (B4 trans oracle) | +3.29 [2.10, 4.45], q 5e-8 | +3.72 [2.66, 4.78], q 2e-10 | +5.23 [4.17, 6.27], q 4e-23 | −1.95 [−3.41, −0.41], p 0.014 | −1.51 [−2.94, 0.01], p 0.052 |
| mars30 vs gbdt_L16_T32 (B4 trans_c oracle) | +15.48, q 8e-110 | +17.43, q 1e-142 | +4.27 [3.25, 5.32], q 5e-16 | +11.21, p < 0.001 | +13.15, p < 0.001 |

On B4's own test jets, mars16 vs gbdtreg_L32_T8 on trans_c is 72.31 vs 69.70, a difference of +2.6, computed from the stored b4.json per-model agreements.

## Comparison with B4's reading

1. **Reproduced on fresh jets.**
   - Same-call parity at K = 16, 30 against the dev-selected baseline.
   - No baseline in the grid above mars30.
   - The translation lead of mars16 and mars30 over the B4 trans oracle: +2.80 and +3.29, against B4's +2.40 and +3.46.
   - The same lead under a Δφ shift: +1.98 and +3.72.
   - Translation recall: 61 % and 69 %, against ≤ 52 % for every evaluated baseline of equal or smaller DL.
   - Direct MARS and zphys30 produce no W → Z changes under the shift.
2. **Not supported: "For mars16 the translation lead holds and the control lead does not".**
   - B4 took the translation lead against gbdtreg_L32_T8 and the control lead against gbdt_L16_T16.
   - Against the same baseline (gbdtreg_L32_T8), mars16's control lead is +2.66 [1.73, 3.65] on fresh jets (+2.6 on B4's jets). The difference of leads is +0.13 [−1.29, 1.52], p 0.86.
   - Against gbdt_L16_T16, the difference is +15.2, but that model reproduces only 23 % of the shift's call changes.
   - So at K = 16, as at K = 30, the lead over the strongest translation baseline does not depend on whether the displacement is coherent. For mars30 vs gbdtreg_L32_T16, the control lead is larger than the translation lead (−1.95 [−3.41, −0.41], p 0.014).
3. **What trans_c measures.** The random-direction displacement is not a null for the mass mechanism.
   - Its expected change of $\sum_i p_{T,i} r_i^2$ is $|d|^2 \sum_i p_{T,i}$, the same as the coherent shift when $\sum_i p_{T,i}\vec r_i \approx 0$.
   - The network sends 49 % of W jets to Z under it.
   - The physics-to-logit MARS surrogates (direct100, zphys30) reproduce 56–57 % of the network's trans_c call changes but 18–23 % of the trans changes. So trans_c separates the direct MARS surrogates from the nested program only through its contrast with trans.
   - trans_c does not separate the nested program from GBDT on logits by mechanism.
4. **Wording.** "Surrogates fitted directly from physics to z or to the logits produce no W → Z changes" holds only for the MARS fits (direct MARS, zphys30) and for the linear models.
   - GBDT on logits sends 74.5 % of W jets to Z (gbdtreg_L32_T16, fresh).
   - Single trees send 67 % (tree_d6).
5. **DL ratios.** "4–14 times smaller DL for K = 2–16" should read:
   - 19× for K = 2 (tree_d4);
   - 14× for K = 4, 8 (direct MARS);
   - 4.6× for K = 16 (direct100, same call −0.28 [−0.84, 0.26] on fresh jets; 0.01 short on B4's split) or 1.7× (GBDT).
6. **Not rerun.** Rotation, A13 and a32x were not rerun. On B4's split all surrogates sit at the never-flip floor for these edits, and no claim of a difference rests on them.

## Figure (b4_fidelity_dl.png)

- The panels match b4.json.
- The translation panel shows GBDT on logits only at leaves 32. The leaves 8/16 points that exceed mars2–mars8 on translation agreement (gbdtreg_L8_T4 63.1, gbdtreg_L16_T4 64.3) are not drawn, although they are the comparison B4 uses for K = 2–8.
- The control panel averages four controls with different flip rates (52 %, 40 %, 8 %, 0.5 %), so it does not show the trans_c comparison discussed in the reading.

## Corrected reading

- **Same call.**
  - At K = 16 and 30 the nested program is within 1 point of the dev-selected baseline of equal or smaller DL. This replicates on 8,218 fresh jets: +0.22 [−0.30, 0.73] and +0.86 [0.37, 1.40].
  - For K = 2–16, baselines reach the same call at 19× (K = 2), 14× (K = 4, 8) and 4.6× (K = 16, direct MARS within CI) lower DL.
  - mars30's 94.06 % is not exceeded: gbdt_L32_T128 is −0.38 [−0.88, 0.11].
- **Coherent 0.06 rad shift (Δη or Δφ).**
  - mars16 and mars30 agree with the network's call changes 2.0–3.7 points more often than the strongest baseline of equal or smaller DL (gbdtreg_L32_T8 / L32_T16; q ≤ 3e-4).
  - They reproduce 61 % and 69 % of the network's call changes, against 50–52 %.
- **Matched random-direction displacement (trans_c).**
  - The nested program leads that same baseline by a similar or larger amount: K = 16 +2.66 [1.73, 3.65]; K = 30 +5.23 [4.17, 6.27].
  - Lead difference trans − trans_c: K = 16 +0.13 [−1.29, 1.52], p 0.86; K = 30 −1.95 [−3.41, −0.41], p 0.014.
  - The measured lead is therefore not specific to the coherent shift at either K. It is a lead on both mass-raising displacements.
- **Specific to the coherent shift.** The failure of the direct physics-to-logit MARS fits (direct100, zphys30) is specific to the coherent shift: they reproduce 18–23 % of the shift's call changes and 0 % W → Z, against 56–57 % of the random-displacement changes.
- **Rank swap, rotation, a32 crossing.** They do not separate the surrogates; the network changes 1–4 % of calls and every surrogate misses ≥ 88 % of them.

## Explainer entry (for site/explain.py, Paper B tab, after B4)

```python
B4V_EXPLAINER = [
 ('Question', 'Do the B4 comparisons between the nested if-statement program and the baseline surrogates hold on new jets, and is the program\'s lead under the 0.06 rad shift specific to the coherent shift?', None),
 ('Background',
  'B4 compared the program with gradient-boosted trees, decision trees and MARS fitted straight from physics to the logits. Edit agreement on a jet is 1 when the surrogate changes its call exactly when the network does, to the same class. '
  'The control edit moves each particle 0.06 rad in its own random direction. Its expected change of $\\sum_i p_{T,i} r_i^2$ is $|d|^2\\sum_i p_{T,i}$, the same as the coherent shift, so it also raises the network\'s mass variable (49 % of W jets become Z).', None),
 ('Method',
  'Same fit split as B4; every model is the one B4 named, re-loaded or refitted deterministically (B4\'s same-call values reproduced exactly). Fresh test: 8,218 jets (seed 23, no overlap with seeds 0, 11, 91). '
  'Edits: $\\Delta\\eta \\mathrel{+}= 0.06$, $\\Delta\\phi \\mathrel{+}= 0.06$, random-direction 0.06 rad. For a fixed baseline $b$, the lead difference per jet is $[a_n-a_b]_{\\rm shift}-[a_n-a_b]_{\\rm random}$, bootstrapped over jets.', None),
 ('Result',
  'Same call replicates (16 terms: $+0.22$ [$-0.30$, $0.73$] vs boosted trees; 30 terms: $+0.86$ [$0.37$, $1.40$]). Under the shift the 16- and 30-term programs agree with the network 2.8 and 3.3 points more often than the strongest boosted-tree model (recall 61 % and 69 % vs 50-52 %). '
  'Under the random displacement they lead the same model by 2.7 and 5.2 points; the lead difference is $+0.13$ [$-1.29$, $1.52$] (16 terms) and $-1.95$ [$-3.41$, $-0.41$] (30 terms). MARS fitted straight to the logits reproduces 23 % of the shift\'s call changes and 56 % of the random displacement\'s.', None),
 ('Checks', 'Recomputed physics matches stored values to $3\\times10^{-9}$; the $\\Delta\\phi$ shift gives the same leads as the $\\Delta\\eta$ shift (+2.0, +3.7).', None),
 ('Why it matters',
  'The program\'s advantage over boosted trees is an advantage on mass-raising displacements in general, not on the coherent shift only. The edit that separates surrogates by mechanism is the coherent shift against the direct physics-to-logit MARS fits.', None),
]
```
