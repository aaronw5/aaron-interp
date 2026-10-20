# B6w: Paper-B baselines with symbolic regression. Fidelity and edit agreement vs description length

Node B6w (stage 2; Paper B, claims B4 / B6t; parent B6v). Script `interp/b6w_symreg.py` (modes logits, obs, fit, pysr, gpl, eval).
Results: `results/jedi/b6w.json` (all surrogates, matched tests, specificity), `b6w_pred_<m>`, `b6w_pysr_<m>`, `b6w_gpl_<m>[_<part>]` (.json/.npz),
figure `results/jedi/b6w_frontier.png`. Logs: `/j-jepa-vol/aaron-interp/logs/b6w-*.log` (eval: `logs/b6w-eval1.log`).

## Conclusion
Formulas found by symbolic regression (gplearn, PySR) on standard jet observables reproduce the networks' calls on ordinary jets
almost as well as the rank-window program of the same size, but they do not follow how the networks' calls change when the jet is
shifted by 0.06 rad: they score at about the level of a surrogate that never changes its call, about 12–17 points below the program.
Against gradient-boosted trees with quantised cuts the program is level or a few points ahead at 6–13 kbit, and most of that lead
also appears for a shift in a random direction, so it is not specific to the stored-axis translation. The two edits meant to
separate mechanisms (swapping the angles of pT ranks 13 and 14, and pT-weighted recentring) move fewer than 1.5 % of the networks'
calls, no more than their controls, so they do not decide between the program and the baselines.

## Procedure
- Jets: `jedi_core.sample(4000, 3701)` = 20,000 jets split (rng 0) into fit 13,333 / dev 6,667; test = `sample(2000, 4701)` minus every
  jet also in the fit sample: N 9,224 (776 removed). A13 uses the 9,182 test jets with ≥ 15 real particles.
- Networks (targets = logits): JEDI-linear n64 (jedi_core), Deep Sets d64 s0, PFN 100-100-128 s0, transformer 2b-d32 s0.
  Unedited test accuracy: JEDI-linear 80.83 % [80.06, 81.57] (inside the 80.6–81.3 % check), Deep Sets 80.78 %, PFN 81.66 %, transformer 76.58 %.
- Edits on the test jets (observables recomputed from the edited particles): trans ($\Delta\eta \mathrel{+}= 0.06$), m2h (trans then angles
  rescaled so $M_0$, $M_2=\sum p_T r^2$ about the stored axis are held), A13 (swap $(\Delta\eta,\Delta\phi)$ of pT ranks 13, 14), pt09 (all pT × 0.9),
  scr (per-particle azimuth scramble, $r$, $p_T$ kept), rcn (pT-weighted recentring: every particle moved by $-c$, $c=\sum p_T\vec r/\sum p_T$;
  $\sum p_T r^2$ about the centroid unchanged to 8e-7 relative, about the stored axis it drops by $\sum p_T|c|^2$).
  Controls: rdt (0.06 rad translation, random direction per jet), rpd (per-particle random 0.06 rad), rcr (shift by $|c|$ in a random
  direction; matched control for rcn), never-flip surrogate (agreement = 1 − network flip rate), fits to row-shuffled logits.
- Network call changes (flip rate): trans 52–62 %, m2h 15–25 %, pt09 5–35 %, scr 7–13 %, A13 0.3–1.2 %, rcn 0.5–1.4 % (rcr 0.5–1.3 %).
  The recentring offset is small: mean $|c|$ = 0.0024 rad (median 0.0015, p90 0.0058).
- Surrogates (fitted to logits on the fit split; call = argmax): program = MARS on the 98 rank-window observables, prefixes K 2–100
  (0.9–47 kbit; K 45–100 extend B6's ≤ 13 kbit program); CART (call / logits, depth 2–10); HistGBDT with 32-bit cuts and with inputs
  quantised to 4/6/8 bits (cut cost = q bits + 64 bits range per observable used); gplearn (population 3000, 30 generations, parsimony
  1e-4…0.03; functions add sub mul div sqrt log abs neg max min) on standardised rank-window observables (gpl_rw) and on 10 standard
  observables (gpl_hand: mass, soft-drop mass, $\tau_{21}$, $\tau_{32}$, $D_2$, N, $\sum p_T$, $\sum p_T r^n$, n = 1, 2, 3); PySR 2.x on the
  standard observables (200 iterations, 15 populations, maxsize 40; one surrogate per complexity cap 3–40). Symbolic fits use 4,000 fit rows.
- Description length: B6 rule (32 bits per number + log2(#inputs) per structure element); symbolic expressions: log2(#functions) per
  operator, log2(#inputs) per variable, 32 per constant, + 64 per logit scale and per standardised observable used.
- Metrics: same call on unedited test jets; edit agreement per jet (surrogate flips iff the network flips, and to the same new call).
  95 % jet bootstrap (B 1000).
- Test: H0 at equal DL the program's edit agreement equals the best baseline's (baseline of ≤ program bits, selected by dev same call);
  H1 differs. Exact McNemar on paired per-jet agreement; Holm over 4 networks × 6 edits × 6 program sizes (K 4, 8, 16, 30, 60, 100;
  144 tests); per-family comparisons Holm over 816 tests. Specificity: [lead on trans] − [lead on rdt], [lead on rcn] − [lead on rcr], bootstrap p.

## Setup and debugging
- venv `/j-jepa-vol/aaron-interp/envs/symreg` (jax-env site-packages via .pth; gplearn 0.4.2 with a scikit-learn ≥ 1.6 `_validate_data` shim;
  PySR with Julia under `/j-jepa-vol/aaron-interp/envs/julia`). Both packages run (smoke test and the fits); the `STATUS` file written by the
  first setup attempt still reads FAILED and is stale.
- Debug 1: the large-budget gplearn jobs for JEDI, PFN, transformer were OOM-killed at 2 GiB (gplearn keeps every generation's population).
  Fix: drop `_programs` after each fit (predict uses only `_program`), and split each network into rw / hand / shuffled jobs (1 CPU / 2 GiB;
  peak 0.4–0.6 GiB, 0.5–0.94 cores). Same seeds as the single-job version.

## Results
Matched comparisons (program − dev-selected baseline of ≤ program bits, edit agreement in points; * Holm < 0.05).

Program vs symbolic regression at K 8 (program 3.2–3.4 kbit; baselines 1.8–3.3 kbit):

| network | baseline | same call | trans | m2h | pt09 |
|---|---|---|---|---|---|
| JEDI-linear | PySR c23 (2.2 kbit) | +1.5 | +11.9* | +22.9* | +4.1* |
| JEDI-linear | gplearn hand (2.6 kbit) | +2.2 | +17.9* | +22.0* | +0.7 |
| Deep Sets | PySR c30 | +1.4 | +16.6* | +21.3* | +2.5* |
| PFN | PySR c40 | +0.6 | +12.6* | +25.5* | +6.0* |
| transformer | PySR c40 | +2.7 | +17.4* | +3.4* | +1.9* |
| transformer | gplearn hand (1.9 kbit) | +6.6 | −3.1* | +6.5* | +3.5* |

- The symbolic surrogates reach 82–89 % same call at ≤ 3.3 kbit (program K8: 85–90 %), but their translation agreement is at the never-flip
  level (JEDI-linear: never-flip 43.5 %, PySR c40 41.8 %, best gplearn-hand 44.8 %; program K8 54.2 %), and on m2h they are 6–28 points
  below never-flip (they change calls the network keeps). gplearn on the rank-window observables is lower still (same call 45–78 %).
  One exception: transformer, gplearn-hand at 1.9 kbit has 56.2 % translation agreement vs program K8 53.1 %.
- Program vs pooled baselines (trees, GBDT 32-bit and quantised, symbolic): translation lead at K16 / K30 (6.6–13.5 kbit):
  JEDI +1.0 / −0.3, Deep Sets +3.6* / +3.1*, PFN +3.6* / +2.1*, transformer +9.1* / +2.8*; at K 4–8 (vs CART depth 5–6) +3.2 to +13.8*.
  Extended program K60 / K100 (27–47 kbit) vs GBDT of ≤ equal bits: translation −2.8* to +2.3*, m2h +0.5 to +2.0*.
  Same call: −2.5 to +1.5 points at all sizes.
- Specificity: [trans lead] − [rdt lead] at K30: JEDI +0.94 [−0.07, +1.95] (p 0.067), Deep Sets +2.30 [+1.34, +3.31] (p 0.001),
  PFN +0.90 [−0.05, +1.88] (p 0.073), transformer +0.61 [−0.25, +1.49] (p 0.18). Most of the translation lead also appears on the
  random-direction translation control.
- A13: program − pooled baseline −0.5 to +0.6 points (Holm < 0.05 in 3 of 24: JEDI K8 +0.59, JEDI K30 −0.47, Deep Sets K16 +0.38).
  rcn: the network flip rate equals that of its matched control rcr (e.g. JEDI 1.40 % vs 1.28 %); [rcn lead] − [rcr lead] within ±0.17 points
  for all 24 (network, K) pairs. The prediction "program leads on A13 / recentring" cannot be evaluated: these edits change ≤ 1.4 % of calls.
- scr: rank-window observables are radial, so every rank-window surrogate (program, CART, GBDT, gplearn-rw) never flips under scr and scores
  exactly the never-flip agreement (87.0–93.5 %); surrogates on the standard observables flip under scr and score 52–91 %. The program's scr
  "lead" over symbolic regression is this structural zero, not tracking.
- Primary Holm family (144): 46 significant with program ahead, 9 with the baseline ahead, 89 not significant.
- Controls: shuffled-logit fits (program K16, GBDT, gplearn) same call 17.5–20.4 % (chance 20 %).

## Reading (scoped)
- H0 (equal agreement at equal DL) is rejected against symbolic regression for trans and m2h in all four networks at K 8–30 (except the
  transformer gplearn-hand case above). Symbolic regression on standard observables fits the unedited calls but not the call changes under
  translation, consistent with the networks' use of un-recentred, stored-axis quantities that the standard observables (mostly
  translation-insensitive) do not carry, even though $\sum p_T r^n$ about the stored axis was in the pool.
- Against tuned / quantised GBDT the B6v picture holds: level to +3.6 points at 6–13 kbit outside the transformer, and the lead is mostly
  shared with the random-direction translation control.
- The competing predictions (statement-object form → lead on A13 / recentring; otherwise SR or quantised GBDT matching the program) are
  not separated: A13 and rcn do not move the networks beyond their controls; SR is below the program; quantised GBDT is level.

## Figure
`results/jedi/b6w_frontier.png`: rows same call / translation agreement / m2h agreement vs description length (kbit, log), columns the four
networks; lines = best member of each family at or below each size; dashed = never-flip.

## Explainer entry (for site/explain.py, Paper B)
- Question: does a compact program written over pT-rank-window observables describe what the networks do better than the standard ways of
  getting a formula from a network (symbolic regression) or a small tree ensemble, at the same size?
- Background: a surrogate is a small model fitted to reproduce a network's outputs. "Description length" counts the bits needed to write it
  (32 bits per number). Matching calls on ordinary jets is easy; the stricter check is whether the surrogate changes its call on edited jets
  exactly when the network does.
- Method: fit each surrogate to the logits on 13,333 jets; on 9,224 test jets apply edits (0.06 rad translation $\Delta\eta\to\Delta\eta+0.06$;
  translation with $M_0$ and $M_2=\sum_i p_{T,i} r_i^2$ held; pT × 0.9; azimuth scramble; rank-13/14 swap; recentring $\vec r_i\to\vec r_i-\vec c$,
  $\vec c=\sum p_T\vec r/\sum p_T$) and score agreement $A=\frac1N\sum_j \mathbb 1[\text{surrogate flips}_j=\text{network flips}_j \wedge \text{same new call}]$.
- Result: figure above; symbolic regression sits at the never-flip line under translation while the program is 12–17 points higher at
  ~3 kbit; quantised GBDT is level with the program.
- Checks: network accuracy reproduced (JEDI 80.83 %); shuffled-logit fits at chance; random-direction and random-recentring controls;
  McNemar with Holm over 144 tests.
- Why it matters: symbolic regression is the standard baseline reviewers ask for; it shows that a formula on standard observables can match
  calls without following the network's response to edits, and that the program's edit fidelity is matched by trees, not exceeded.
