# B6w verify: symbolic-regression baselines, fresh samples and an observable-pool control

Node B6w-verify (skeptic for B6w; Paper B, claims B4 / B6t). Script `interp/b6w_verify.py` (modes prep, fit).
Jobs: `b6w-verify-pfn` (1 CPU / 2 GiB, 2,484 s, peak 1,339 MiB); `b6w-verify` (JEDI; its pod was evicted when the node went
NotReady, no result) rerun as `b6w-verify-j` (1 CPU / 2 GiB, 2,339 s, peak 2,047 MiB, at the limit).
Results: `results/jedi/b6w_verify_jedi.json`, `results/jedi/b6w_verify_pfn.json`. Logs: `/j-jepa-vol/aaron-interp/logs/b6w-verify-j.log`, `b6w-verify-pfn.log`.

## Conclusion
On fresh jet samples, symbolic-regression formulas built from standard jet observables again failed to follow the networks
when the jet was shifted with its mass and $\sum p_T r^2$ held fixed, and again matched the networks' calls on ordinary jets.
Under the plain 0.06 rad shift, the program's lead over them was about 5–8 points rather than 12–17, and some PySR formulas
of the same size did better than the program. The same lack of response to edits appears when MARS (the method that builds
the program) or boosted trees are fitted to the same standard observables. The gap therefore comes from the observables
given to the fitter, not from symbolic regression as a method. Against trees and boosted trees on the rank-window
observables, the program's lead under the shift varied from −6 to +8 points between samples and was not specific to the
shift direction.

## Code check (interp/b6w_symreg.py)
- Splits: `indices()` reproduces `jedi_core.sample()` index selection (same rng calls, sorted), so the `keep` mask lines up
  with the rows; the fresh test set had 0 rows in F (checked).
- Edits (checked on the fresh sample in `build()`): padded slots unchanged for every edit (max |Δ| 0); trans moves only Δη of
  real particles, by 0.0600 (other features Δ 0); m2h holds $\sum p_T r^2$ (max relative change 8e-8) and $\sum p_T$ (Δ 0);
  A13 changes only slots 13 and 14, both real in every selected jet; pT is sorted descending in 100 % of jets, so slots 13/14
  are pT ranks 14/15 (1-based). rcn: mean offset 0.0023 rad (B6w: 0.0024).
- Agreement metric, McNemar, Holm and bootstrap code: no defect found. HAND observables: mass, soft-drop mass, $\tau_{21}$,
  $\tau_{32}$, $D_2$, N, $\sum p_T$ are invariant under a common $\Delta\eta$ shift (a longitudinal boost); only
  $\sum p_T r^n$ (n = 1, 2, 3) about the stored axis respond to trans. The SR surrogates flip 12–50 % of calls under trans,
  through $\sum p_T r$ and $\sum p_T r^2$ (visible in the PySR equations).
- Selection-dependent statement in B6w: the matched tables use dev-same-call selection, but the figure's step lines are the
  running maximum of the test-set metric (an oracle frontier). In the JEDI translation panel the PySR line reaches 54 % at
  1.5 kbit (PySR c12: 54.0 %), level with program K8 (54.2 %), which contradicts the "12–17 points higher" text. PySR
  members' translation agreement is non-monotonic in complexity (JEDI, B6w sample: 40.7–54.0 %).
- The figure has 12 panels (three metrics × four networks) and more than one message; never-flip labels overlap lines in
  the m2h row.

## Procedure (rerun)
- Jets: F = `sample(4000, 5301)` split by rng 0 into fit 13,333 / dev 6,667; T = `sample(2000, 5401)` minus F rows:
  N 9,296 (704 removed); A13 N 9,253. Networks: JEDI-linear n64 (jedi_core) and PFN 100-100-128 s0 (numpy forward).
- Edits as B6w (new seeds for random controls): trans, m2h, A13, pt09, rcn; controls rdt (random-direction 0.06 rad), rcr
  (random-direction shift by |centroid|), never-flip, GBDT on row-shuffled logits.
- Surrogates on F-fit: program = MARS on the 98 rank-window observables, K 4/8/16/30; CART d2–10; HistGBDT on rank windows,
  leaves 2–16 × lr 0.2/0.5 × iterations 5/10/25, 32-, 8- and 6-bit cuts; gplearn on the 10 standard observables
  (population 2,000, 20 generations, parsimony 3e-4/1e-3/3e-3, 4,000 rows, seed 5410; B6w used 3,000/30); PySR on the
  standard observables (JEDI only; B6w settings, complexity caps 3–40).
- New control (observable pool vs fitting method): MARS on the 10 standard observables (`prog_hand`, K 4/8/16/30) and
  HistGBDT on the standard observables (`gbdt_hand`, 32- and 8-bit cuts). H0: at matched bits, a surrogate on the standard
  observables has the same translation / m2h agreement as the program. Competing predictions: (a) the SR deficit is a
  property of SR → MARS/GBDT on the standard observables track the edits like the program; (b) it is a property of the
  observable pool → MARS/GBDT on the standard observables show the same deficit.
- Metric: edit agreement per jet (surrogate flips iff the network flips, to the same call); never-flip = 1 − network flip
  rate. Matched test: dev-same-call-selected member of each family with ≤ program bits; paired jet bootstrap (B 1000),
  exact McNemar, Holm over all (program size × group × edit) tests per network (JEDI 105, PFN 90). Specificity:
  [trans lead] − [rdt lead], bootstrap p.

## Results
Networks on T: JEDI-linear accuracy 80.55 % (B6w 80.83 %); flips trans 55.6 % [54.6, 56.5], m2h 16.3 %, A13 1.25 %,
rcn 1.30 % vs rcr 1.27 %. PFN accuracy 81.22 % (B6w 81.66 %); trans 56.1 %, m2h 16.8 %, A13 0.35 %, rcn 0.56 % vs rcr 0.58 %.

Program K8 (3.2 kbit) − symbolic regression (dev-selected, ≤ program bits), points [95 % CI]; all Holm < 1e-14:

| network | baseline | same call | trans | m2h | B6w trans |
|---|---|---|---|---|---|
| JEDI-linear | PySR c40 (3.1 kbit) | +0.15 [−0.48, 0.71] | +5.51 [4.17, 6.68] | +10.92 [10.04, 11.78] | +11.9 (c23) |
| JEDI-linear | gplearn hand (2.9 kbit) | +6.52 | +7.58 [6.54, 8.58] | +14.32 | +17.9 |
| PFN | gplearn hand (2.3 kbit) | +6.15 | +6.12 [4.86, 7.44] | +21.94 | +14.4 |

- Translation agreement, JEDI: never-flip 44.4 %; program K8 54.2 %; PySR c40 48.7 %; gplearn 46.6 %. PySR c17 (2.0 kbit)
  57.8 % and c23 (2.3 kbit) 57.1 % exceed program K8 (54.2 %); at K4 (1.65 kbit) the dev-selected PySR c12 equals the
  program on trans (−0.04 [−1.16, 1.03]) and on same call (+0.46).
- m2h: every symbolic surrogate is below never-flip (JEDI 57–81 % vs 83.7 %; PFN 57–72 % vs 83.2 %); the program is at or
  above never-flip from K8 (JEDI 84.7 %, PFN 81.6–84.7 %).
- Observable-pool control (program K8 / K16 / K30 − same-method surrogate on the standard observables):

| network | baseline family | trans | m2h |
|---|---|---|---|
| JEDI | MARS on standard obs. | +13.2 / +13.0 / +12.5 | +15.2 / +11.1 / +13.8 |
| JEDI | GBDT on standard obs. | +11.9 / +10.9 / +11.9 | +30.1 / +18.7 / +22.8 |
| PFN | MARS on standard obs. | +12.2 / +14.3 / +19.5 | +19.6 / +25.6 / +24.6 |
| PFN | GBDT on standard obs. | +6.9 / +6.3 / +6.8 | +29.2 / +33.2 / +32.2 |

  MARS on the standard observables reaches about the same call as the program (JEDI 88.6 % at 3.5 kbit, 90.4 % at 7.3 kbit; program K8 89.1 %, K16 90.1 %) with
  translation agreement 41–46 % (never-flip 44.4 %). On these observables, MARS and GBDT are as far below the program as PySR
  and gplearn are, or further; PySR/gplearn are 3–8 points above MARS-on-standard on trans. Prediction (b) holds; (a) is
  excluded (Holm < 1e-50 for every MARS-on-standard comparison on trans and m2h).
- Program vs pooled rank-window trees / GBDT (dev-selected), trans: JEDI K8 +12.0 (tree d6), K16 +8.5 [7.4, 9.5] (tree d7),
  K30 −5.7 [−6.7, −4.8] (GBDT 6-bit L4 T10); PFN K8 +2.4, K16 +1.4 [0.5, 2.2], K30 −2.8 [−3.7, −2.0]. B6w on its sample:
  JEDI +1.0 / −0.3, PFN +3.6 / +2.1 (K16 / K30). The selected baseline changes between CART and GBDT with the sample.
- Specificity [trans lead] − [rdt lead] vs pooled rank-window baselines: JEDI K16 +0.16 [−0.82, 1.05], K30 +0.18
  [−0.87, 1.18]; PFN K16 −0.47 [−1.31, 0.42], K30 −0.40 [−1.29, 0.54]. Against symbolic regression and standard-observable
  surrogates, the program's lead is larger on rdt than on trans (−1.1 to −6.3 points, p ≤ 0.04).
- A13: program − pooled rank-window baseline −0.65 to −0.11 (PFN), −0.16 to −0.01 (JEDI); rcn: −0.10 to +0.19 (Holm ≥ 0.11 in JEDI). Same conclusion as B6w.
- Shuffled-logit GBDT: same call 18.2 % (JEDI), 19.9 % (PFN); chance 20 %.

## Assessment of the B6w reading
- Reproduced: SR same call within 0.2–6.5 points of the program; SR below never-flip under m2h; A13 and rcn flip ≤ 1.3 % of
  calls, no more than rcr; the program's lead over trees/GBDT is not specific to the stored translation direction (rdt
  gives the same lead).
- Not reproduced: the size of the translation lead over SR (fresh samples +5.5 to +7.6, not +12 to +18); "SR at the
  never-flip level" holds for the dev-selected members only (4–5 points above never-flip here), and PySR members of
  2.0–2.3 kbit exceed the program; "level to +3.6 at 6.6–13.5 kbit" vs trees/GBDT (fresh range −5.7 to +8.5).
- Missing control, now run: fitting MARS or GBDT to the same standard observables produces the same deficit, so the B6w
  thesis note "the program's edit fidelity per bit was higher than the symbolic-regression baselines" compares observable
  pools, not fitting methods. gplearn on the rank-window observables (B6w) fitted poorly (same call 45–78 %), so a
  method comparison on the same pool is not available at matched quality.
- rdt is itself a translation off the stored axis; it controls for the η direction, not for stored-axis use. The
  "stored-axis direction" wording in the B6w thesis note should read "the η direction".

## Corrected reading
At 1.6–3.3 kbit, symbolic regression (PySR, gplearn) on ten standard jet observables matched the networks' calls on
unedited jets to within 0.2–6.6 points of the rank-window program on two samples. Under the translation with $M_0$ and
$M_2$ held it changed calls the networks kept (3–28 points below never-flip). Under the plain 0.06 rad translation the
dev-selected symbolic surrogates were within −12 to +9 points of never-flip and 5.5–18 points below the program at K8 (sample-
dependent; one exception in the transformer, and individual PySR members at 2.0–2.3 kbit above the program in JEDI on the
fresh sample). MARS and GBDT fitted to the same standard observables show the same deficit (translation agreement at
never-flip, m2h below never-flip), so the difference reflects the observable pool: stored-axis rank-window sums vs standard
observables that are mostly invariant under translation. Against trees and GBDT on the rank-window observables, the
program's translation lead at 6.6–13.5 kbit ranged from −5.7 to +9.1 points over samples and networks and was the same on
the random-direction translation. A13 and recentring changed ≤ 1.4 % of calls, no more than their controls.

## Explainer entry (site/explain.py, Paper B; skeptic check)
- Question: is the program's advantage over symbolic regression due to the fitting method or to the observables it is given?
- Background: a surrogate's edit agreement $A=\frac1N\sum_j \mathbb 1[\text{surrogate flips}_j=\text{network flips}_j
  \wedge \text{same new call}]$; never-flip agreement $=1-$ network flip rate. Standard observables (mass, $\tau_{21}$,
  $D_2$, …) are unchanged by a shift $\Delta\eta\to\Delta\eta+0.06$, because that shift is a longitudinal boost.
- Method: on fresh jets (fit `sample(4000, 5301)`, test `sample(2000, 5401)`, N 9,296), fit the program (MARS on rank-window
  sums), PySR / gplearn, and MARS and GBDT on the standard observables, to JEDI-linear and PFN logits; compare at equal bits.
- Result: all surrogates on standard observables, whatever the fitter, sit near never-flip under translation and below it
  under the $M_2$-held translation; the program is 5–20 points higher.
- Checks: edits verified (padding fixed, $\sum p_T r^2$ held to 8e-8); network flip rates reproduce B6w; shuffled-logit
  surrogate at chance; random-direction translation gives the same leads.
- Why it matters: a reviewer's "use symbolic regression" baseline fails here for a reason that is about inputs, so the
  comparison supports "rank-window observables carry the networks' translation response" more than "the program is a
  better extraction method".
