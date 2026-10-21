# E29 — Automated circuit discovery on JEDI-linear: does an algorithm recover the hand-named program?

Node E29, round 14, stage 3, claims A1, A2, A3, A5, A6 (exploratory method; Paper B baseline). Status: done (no verifier yet).
Script `interp/e29_auto_circuit_jedi.py` (modes: default run, `--followup`, `--plot`, `--smoke`). Jobs `e29-run` (1 CPU / 2 GiB,
8,320 s wall, 0.87 cores average, pod peak memory 2,047 MiB = limit incl. page cache, not killed) and `e29-followup` (1 CPU / 2 GiB,
1,519 s). Results `results/jedi/e29_auto_circuit.json`, `results/jedi/e29_followup.json`, figure `results/jedi/e29_auto_circuit.png`,
logs `logs/e29-run.log`, `logs/e29-followup.log`.

## Conclusion

An automated search that only looks at which internal connections change the network's output keeps every one of the 13 first-layer
particle cuts and 14–16 of the 17 hand-named second-round jet quantities, and with them recovers 95 % of the network's accuracy above
chance using about 119 of the 186 active units. The units the search leaves out are mostly ones the hand program also kept: 9 of the 11
jet-context units and about 30 of the 55 units in the first dense layer; with all 16 output-layer units the hand program reaches 98 %
with 156 units. So the hand-named particle cuts and jet quantities are what an algorithm finds as well, but because the named set
covers almost every active unit at those two layers, this agreement cannot be told apart from "any method keeps the active units";
the test does not single out the hand names over other descriptions of the same units.

## Question

The program elements of JEDI-linear (13 first-layer cuts, 17 second-round (pT, r) hinge objects, the context set S0/S30/S31/S58,
the E11 hinge formulas for h4/h5/z) were all named by hand-designed tests. Does a standard automated circuit-discovery method, given
only the network's weights and activations, select the same units; and how does the hand program compare with the algorithm's
circuits on accuracy recovered per unit kept?

## Background (definitions)

- **Unit**: one channel of one layer of JEDI-linear 3-feature n64 (official checkpoint, 80.9 % on the full test set). Layers, in
  forward order: `a` (64 first-layer cuts per particle, ReLU(w · (pT, Δη, Δφ) + b)), `P` (pooled cuts, the per-channel sum over
  pT-rank slots kept by the zero-bit masks), `g` (64 jet-context units, dense on P), `s` (64 per-particle units, dense on a),
  `u` (per-particle, dense on the quantized sum s + g), `S` (64 second-round objects, pooled u), `h4` (64), `h5` (32), `z` (16
  pre-ReLU neurons), 5 logits (g, q, W, Z, t). A per-particle layer (a, s, u) counts as one unit per channel (all particles together).
- **Live unit**: a unit whose value is not constant over the 5,000 fit jets (all slots). 186 units are live: a 13, P 12, g 11, s 9,
  u 19, S 19, h4 55, h5 32, z 16.
- **Edge**: a nonzero fused (quantized, batch-norm-folded) kernel entry between two live units of consecutive layers, plus the
  identity edges a_c → P_c and u_c → S_c of the two pooling steps; 1,216 edges (a→P 13, a→s 32, P→g 52, s→u 35, g→u 15, u→S 19,
  S→h4 312, h4→h5 484, h5→z 206, z→logit 48).
- **Corruption** (what a removed unit or edge is replaced by): *mean* — every unit at its average over the 5,000 fit jets (for a, s, u
  the average per pT-rank slot), as in E5/X8; *resample* — every unit at the value it takes on a donor jet of the partner class with
  nearly the same jet pT (W ↔ Z, q ↔ g, t → W or Z; donor drawn among the 5 nearest in ΣpT of the partner class; rng 2903 fit, 2904
  test), as in X8. Resample asks "what does the unit add beyond what a jet of the confusable class with the same pT would give".
- **Literal circuit (node level)**: a set of units; every unit in the set computes normally from its inputs, every other unit (all
  layers, dead units included) is replaced by its corrupted value. This is the definition used for every F below.
- **Edge-level circuit**: a set of edges; the receiver of a kept edge reads the source's real value, the receiver of a removed edge
  reads the source's corrupted value.
- **F (faithfulness)**: $F = (\mathrm{acc}_{\rm circ} - \mathrm{acc}_0)/(\mathrm{acc}_{\rm full} - \mathrm{acc}_0)$, where acc_0 is the
  accuracy with every unit corrupted (mean: 19.95 %, a constant call; resample: 14.61 %) and acc_full = 80.49 %. F = 1: the circuit
  gives the full network's accuracy; F = 0: no better than everything corrupted.
- **KL**: $\mathrm{KL}(p_{\rm full}\,\|\,p_{\rm circ}) = \frac1N\sum_j \sum_c p_{\rm full}(c|j)\log\frac{p_{\rm full}(c|j)}{p_{\rm circ}(c|j)}$,
  the average information lost when the circuit's class probabilities replace the network's.
- **EAP (edge attribution patching)**: estimates, for every edge at once, how much the metric $m$ = mean log-probability of the
  network's own call would drop if that edge alone were switched to its corrupted input, from one gradient: with an edge mask
  $M_e \in [0, 1]$ (1 = clean), $A_e = \int_0^1 \partial m / \partial M_e \,|_{M = 1 - \alpha G}\, d\alpha$ (G = all graph edges);
  integrated over 5 steps α = 0.1 … 0.9 (the EAP-IG variant; the plain α = 0 gradient is also reported). Circuits: the k edges with
  largest |A_e|, k = 10 … 800, pruned to edges on a path to the logits; units = endpoints. A unit ranking (sum of A over a unit's
  outgoing edges) gives "top-n units" circuits.
- **ACDC-style pruning**: start from all edges; visit receivers from the logits backward (logits, z, h5, h4, S, u, g, s, P); for each,
  try removing its incoming edges one at a time (smallest |A_e| first) and keep the removal if KL(full ‖ circuit) rises by less than
  τ; τ = 1e-5 … 3e-2 (8 values, ascending, each continuing from the previous circuit, so circuits are nested). KL measured on 1,000
  fit jets.
- **Hand program** (what the hand-designed tests named; node-level circuits): *handA* = the 13 live l1 cuts + all live P (12), g (11),
  s (9) as needed intermediates + the 17 S hinge objects (live S minus S31, S58; same u channels) + h4/h5 units with E11 held-out
  R² ≥ 0.9 (40 h4, 21 h5) + z units with E11 R² ≥ 0.9 or a named mass window (z0, z1, z2, z7, z8, z9, z10, z12, z14; 9 of 16) —
  149 units. Follow-up variants: *handA_allz* (all 16 z, 156 units), *handA_allhead* (all live h4, h5, z, 182 units),
  *handA_allhead_noctxS* (without S0, S30), *handA_allhead_g49only* (g49 the only g unit), *handB_allhead* (+ S31, S58 = all 186 live
  units).
- **Jaccard overlap** of two sets: |A ∩ B| / |A ∪ B|.
- **McNemar (exact)**: paired test on per-jet correctness of two circuits; reported as (jets only circuit 1 gets right, jets only
  circuit 2 gets right, two-sided p).
- **Holm**: step-down multiple-comparison correction of p-values.
- **W→Z flow**: share of true W jets the circuit calls Z minus the same share for the full network; q→g likewise.

## Procedure

1. Samples: fit / discovery = `jedi_core.sample(1000, 2901)` (5,000 jets, 1,000 per class); ACDC uses its rows [::5] (1,000 jets);
   test = `sample(1600, 2902)` minus 151 jets byte-identical to a fit jet = 7,849 jets; random-circuit nulls on test[::8] (982 jets).
2. Edge-masked forward pass written in JAX with the network's own quantizers and fused kernels (l3's quantizer acts on s + g, so the
   four on/off combinations of the s→u and g→u edges into one channel are quantized separately; exact). Checks on the test jets:
   all edges on vs `Jedi.run`: max |Δlogit| 0.0, 0 calls differ (N 7,849); node-level ablation through masks vs
   `Jedi.run_jetwise` with per-slot mean overrides, for handA and for a random 149-unit set: max |Δlogit| 0.0, 0 / 982 calls differ.
   Same checks pass on all 5 control networks.
3. Unedited network on the test jets: 80.49 % [79.63, 81.32] (95 % bootstrap over jets, 1,000 replicates). The task's reference band
   is 80.6–81.3 %; the point estimate is 0.1 pt below it, the interval covers it (E11 reported 80.42 % on another sample).
4. EAP-IG and plain EAP on the 5,000 fit jets (batches of 500), per corruption. Completeness check (sum of attributions vs the actual
   metric drop with everything corrupted) and a direct check: for 40 edges (20 largest |A|, 20 random) the exact drop in m from removing
   that edge alone, on 1,000 fit jets.
5. Circuits per corruption: EAP top-k edges (k = 10, 20, 40, 80, 150, 250, 400, 800), EAP top-n units (n = 5 … 185 plus the hand
   sizes), ACDC at 8 τ. Each scored on the 7,849 test jets at node level (and edge level for edge circuits): accuracy, F with 95 %
   bootstrap interval, F_call (same with "same call as the full network"), KL-based F_KL = 1 − KL(circuit)/KL(all corrupted),
   same-call %, 5 × 5 confusion, W→Z and q→g flows.
6. Controls: (a) for each EAP-edge and ACDC circuit, 200 random circuits with the same number of units per layer drawn from live units
   (and 100 drawn from all 64 / 32 / 16 channel indices, mean corruption), scored the same way on 982 test jets; permutation p =
   (1 + #null ≥ observed) / 201. Jaccard nulls: 500 size-matched random draws from live units and from all channels per circuit.
   (b) hand circuits scored with the same F, McNemar against the found circuits nearest in size. (c) the whole pipeline on three
   untrained redraws (`Jedi(64, untrained=True, seed=0,1,2)`), JEDI-linear n32 and the permutation-invariant n64 (mean and resample
   EAP, mean ACDC, no random-circuit F nulls).
7. Holm across the 8 EAP edge sizes × 2 corruptions within each test family (S Jaccard vs live null, l1 Jaccard vs live null,
   F vs live null).

## Results

### EAP attributions agree with exact edge removals to first order

| corruption | Σ A_e | m(clean) − m(all corrupted) | Spearman(plain, IG) | exact single-edge check (40 edges): Pearson / Spearman |
|---|---|---|---|---|
| mean | 1.438 | 1.465 | 0.63 | 0.64 / 0.72 |
| resample | 3.086 | 2.941 | 0.65 | 0.84 / 0.87 |

Gradients pass through the quantizers (straight-through), and the attributions sum to within 2 % (mean) and 5 % (resample) of the total
metric drop. The plain gradient ranks edges differently from the integrated version (Spearman 0.63), so only EAP-IG is used.

### Faithfulness vs size (node-level literal circuits, 7,849 test jets; F with 95 % interval)

| circuit | edges | units | F mean corr. | F resample corr. | random live units, same count per layer: mean / 95th pct / max (p) |
|---|---|---|---|---|---|
| EAP top-150 edges | 111 / 118 | 73 / 69 | 0.547 [0.526, 0.566] | 0.668 [0.652, 0.683] | mean: 0.022 / 0.131 / 0.235 (0.005) |
| EAP top-250 edges | 211 / 213 | 107 / 98 | 0.797 [0.782, 0.811] | 0.759 [0.745, 0.774] | 0.074 / 0.269 / 0.385 (0.005) |
| EAP top-400 edges | 382 / 377 | 145 / 135 | 0.955 [0.946, 0.965] | 0.904 [0.894, 0.916] | 0.357 / 0.659 / 0.815 (0.005) |
| ACDC τ 0.03 | 37 / 103 | 31 / 65 | 0.311 [0.288, 0.334] | 0.669 [0.653, 0.684] | 0.005 / 0.024 / 0.168 (0.005) |
| ACDC τ 0.01 | 138 / 187 | 85 / 97 | 0.808 [0.793, 0.821] | 0.888 [0.876, 0.900] | 0.031 / 0.180 / 0.281 (0.005) |
| **ACDC τ 0.003** | 249 / 272 | **119 / 115** | **0.955 [0.945, 0.965]** | **0.942 [0.932, 0.952]** | 0.159 / 0.392 / 0.576 (0.005) |
| ACDC τ 0.001 | 376 / 432 | 148 / 158 | 0.977 [0.969, 0.986] | 0.977 [0.969, 0.985] | 0.461 / 0.738 / 0.882 (0.005) |
| handA (hand program) | — | 149 | 0.871 [0.859, 0.885] | 0.904 [0.894, 0.916] | 0.339 / 0.600 (0.005) |
| handA_allz | — | 156 | 0.976 [0.968, 0.986] | 0.930 [0.920, 0.940] | 0.519 / 0.797 (0.005) |
| handA_allhead | — | 182 | 0.992 [0.986, 0.999] | 0.984 [0.978, 0.991] | 0.767 / 0.956 (0.020) |
| handA_allhead without S0, S30 | — | 178 | 0.864 [0.850, 0.878] | 0.869 [0.858, 0.880] | 0.569 / 0.866 (0.040) |
| handA_allhead, g49 the only g unit | — | 172 | 0.974 [0.965, 0.983] | 0.945 [0.936, 0.954] | 0.676 / 0.884 (0.005) |

(Two numbers in a cell = mean / resample circuit. Random-null columns are for the mean-corruption circuit; resample nulls are lower,
e.g. ACDC τ 0.003: 0.089 / 0.262 / 0.472.) Random circuits drawn from all channels instead of live ones: F ≤ 0.024 at every size
(dead-channel dilution, as in E26v). The p-values sit at the floor of 200 draws (0.005); with Holm over 16 tests the adjusted floor is
0.08, so the significance statement rests on "observed F above the maximum of 200 draws" (true at 150–400 edges in both corruptions).

- H0 (F): a found circuit is no more faithful than random live units with the same count per layer. Rejected at every size from
  150 to 400 edges / 69–145 units (observed above all 200 draws); not rejected at ≤ 40 edges (F ≈ 0 for both) or at 800 edges
  (F 0.985 vs random 0.954, p 0.23), where nearly all live units are kept.
- Smallest circuit with F ≥ 0.9: ACDC τ 0.003, 119 units (mean) / 115 units (resample); EAP top-400 edges, 145 / 135 units. The two
  other trained networks give the same size: ACDC τ 0.003 on JEDI-linear n32 116 units F 0.912 (acc 77.08 % full), on the
  permutation-invariant n64 122 units F 0.946 (acc 80.95 % full).
- EAP edge circuits are less faithful than ACDC at the same unit count (107 units F 0.797 vs ACDC 119 units 0.955; edge-level F at
  400 edges 0.659 vs node-level 0.955), because EAP's linear estimate misses interactions (Pearson 0.64 with exact removals).

### Hand program vs found circuits at matched size (McNemar on per-jet correctness)

| comparison (mean corruption) | F | jets only found right / only hand right | p |
|---|---|---|---|
| ACDC τ 0.001 (148 units) vs handA (149) | 0.977 vs 0.871 | 765 / 260 | 3e-58 |
| ACDC τ 0.001 (148) vs handA_allz (156) | 0.977 vs 0.976 | 177 / 173 | 0.87 |
| ACDC τ 0.003 (119) vs handA_allz (156) | 0.955 vs 0.976 | 176 / 278 | 2e-6 |
| EAP top-149 units vs handA (149) | 0.823 vs 0.871 | 567 / 794 | 8e-10 |
| EAP top-400 edges (145) vs handA_allz (156) | 0.955 vs 0.976 | 229 / 329 | 3e-5 |
| resample: ACDC τ 0.003 (115) vs handA_allz (156) | 0.942 vs 0.930 | 347 / 286 | 0.017 |
| resample: ACDC τ 0.001 (158) vs handA_allhead (182) | 0.977 vs 0.984 | 216 / 254 | 0.088 |

The plan's hand circuit handA lies below the ACDC curve (F 0.871 at 149 units vs 0.977 at 148). Its deficit comes from the 7 z units
that the E11 R² ≥ 0.9 / mass-window rule leaves out (z3, z4, z5, z6, z11, z13, z15): restoring them (handA_allz) gives F 0.976, equal
to ACDC at 148 units (McNemar p 0.87). ACDC reaches F 0.955 with 37 fewer units than handA_allz; handA_allz is 0.021 higher in F
(p 2e-6, mean corruption) and 0.012 lower under resample corruption (p 0.017).

### Which units the search keeps (ACDC τ 0.003; hand program = handA_allz)

| layer | live | hand program | ACDC mean (119) | ACDC resample (115) | units the hand program keeps and ACDC drops |
|---|---|---|---|---|---|
| l1 cuts a | 13 | 13 | 13 | 13 | none |
| P | 12 | 12 | 8 | 5 | mean: P3, P6, P43, P46 |
| g (context) | 11 | 11 | 2 (g4, g49) | 2 (g4, g55) | 9 of 11 |
| s | 9 | 9 | 7 | 7 | s6, s18 |
| S | 19 | 17 | 16, all named (drops S52; no S31/S58) | 15 = 14 named + S31 (drops S0, S52, S63) | S52 (mean); S0, S52, S63 (resample) |
| h4 | 55 | 40 | 23 | 24 | 19 / 20 |
| h5 | 32 | 21 | 21 | 19 | 2 / 3 |
| z | 16 | 16 | 13 | 15 | z2, z6, z15 / z2 |

- The 13 l1 cuts (the pT ramps and the nine angular half-planes) are all kept by every circuit with F ≥ 0.9 (ACDC and EAP); the
  65–97-unit ACDC circuits keep 12 of 13 (mean τ 0.01 drops a21). EAP's unit ranking puts S29 (ReLU of 4.44 ln pT + 93.8 r − 21.2 over all slots), S50 (ΣpT·r² over ranks 0–38), S0 (ΣpT
  of the 14 hardest), S19 (soft multiplicity) and S1 (Σ_{ranks ≥ 1}(r − 0.2)₊) first among the S objects (mean corruption) — the same
  five as the largest E1 mean-ablation costs (S29 40.8 %, S1 20.6 %, S0 12.3 %, S50 12.0 %, S19 11.5 % of calls), a consistency
  between two ablation-based rankings, not an independent check.
- Context (claim A2): the search keeps 2 of the 11 g units. Keeping g49 alone and all downstream units gives F 0.974 (mean) / 0.945
  (resample) vs 0.992 / 0.984 with all 11 g units, so the other 10 g units carry 0.018 / 0.039 of F. Of the four context objects,
  S0 and S30 are kept by both methods and removing them from handA_allhead lowers F by 0.13 / 0.12 (0.992 → 0.864; 0.984 → 0.869);
  S31 and S58 are left out by ACDC at mean corruption and add 0.008 / 0.016 to handA_allhead (to F 1.000, the full live set).
- Confusion flows: handA_allz W→Z +0.023, q→g −0.011 (mean); ACDC τ 0.003 W→Z +0.001, q→g +0.050; smaller circuits move W jets to Z
  (EAP top-150 edges W→Z +0.50; ACDC τ 0.01 +0.16).

### Overlap with the named sets (the planned H0 test)

- H0: the found l1 / S sets overlap the named sets no more than random live sets of the same size. Not rejected at any size: Jaccard
  with the 17 named S objects reaches 0.89–0.94 at ≥ 119 units, but random draws from the 19 live S units reach the same values
  (permutation p 0.03–1.0; Holm 1.0 for all 16 tests); for the l1 cuts the named set *is* the live set (13 of 13), so p = 1 by
  construction. Against random draws from all 64 channels p = 0.002 at every size with ≥ 2 S units — this only says the search avoids
  constant (dead) channels.
- The planned test therefore has no power here: the hand naming covers 13 of 13 live cuts and 17 of 19 live S units, so agreement
  with the names cannot be separated from agreement with "the live units". The informative comparisons are the per-layer drops above
  (g, P, h4) and the F-vs-size curve.

### Controls (c)

- Untrained redraws (seeds 0–2): every call is the same class (accuracy 20.09 / 19.95 / 20.03 %), so F is undefined and ACDC removes
  every edge without raising KL above 1e-4 at τ ≥ 3e-5 (seed 2: at τ ≥ 1e-3). Live S units: 1–2 (S58; S58, S60; S30, S31); EAP's S
  sets have Jaccard 0–0.06 with the 17 named S objects; live l1 cuts 8–10, all among the trained network's 13 (the redraw keeps the
  trained number formats and masks). The S-level overlap is not reproduced by the architecture alone.
- JEDI-linear n32 and permutation-invariant n64: channel indices are not shared between independently trained networks, and the overlap
  with the n64 index sets is at the random level (n32 S Jaccard ≤ 0.22, p ≥ 0.28 vs live; perminv ≤ 0.19). Their F-vs-size curves
  follow the n64 curve (F ≥ 0.9 at 116 and 122 units).

## Competing hypotheses

- Thesis ("the network is the named program"): predicted found S / l1 sets far above random live overlap (Jaccard ≥ 0.6, p < 0.01),
  hand circuit on or above the algorithm's curve, untrained overlap at random level. Measured: Jaccard ≥ 0.6 at ≥ 98 units, but not
  above the live-random null (untestable, named ≈ live); the hand circuit with all z units is on the ACDC curve at 148–156 units
  (McNemar p 0.87); the plan's handA (9 of 16 z) is below it; untrained overlap is at 0–0.06.
- Alternative 1 ("same F with different S units"): excluded at the S layer — every circuit with F ≥ 0.9 contains 14–17 of the 17
  named S objects and all 13 cuts; no circuit with F ≥ 0.9 replaces them by other units (there are only 2 other live S units).
- Alternative 2 ("F 0.9 at a size well below the named program"): partly supported — ACDC reaches F 0.955 with 119 units vs 156 for
  the hand program with all z; the difference is in context units (9 of 11 g dropped), P (4) and h4 (17), not in the cuts or S
  objects.

## Issues and limits

- Random-circuit F nulls use 200 draws (p floor 0.005, Holm-adjusted floor 0.08); the planned 500 were cut to keep the job under
  2.5 h on 1 CPU. The observed F is above every draw at 150–400 edges.
- The hand program's h4/h5/z content is my construction from E11 R² and the named mass windows; the z rule drives handA's lower F.
- ACDC discovery used 1,000 jets and a KL threshold; KL estimates at τ ≤ 1e-4 are near the sampling noise of 1,000 jets.
- Untrained networks are constant-call; F and KL-based measures are degenerate there, so control (c) only constrains the overlap.
- Main run and follow-up give handA (resample) accuracy 0.7420 vs 0.7418 (2 of 7,849 jets) on different pods; attributed to
  floating-point differences between CPUs at quantizer boundaries; all other follow-up numbers match the main run.
- Pod peak memory reached the 2 GiB limit (cgroup peak includes page cache; no OOM). A rerun could request 1 CPU / 3 GiB.
- Paired McNemar here compares correctness, not call changes of a single edit.

## Explainer entry (site/explain.py, Paper B baseline / Paper A tab)

**Question.** If an algorithm looks for the smallest part of JEDI-linear that still classifies jets, does it pick the same particle
cuts and jet quantities that the hand-built program names?

**Background.** Interpretability work usually names parts of a network by designing tests. Automated circuit discovery instead
removes connections one by one and keeps only those whose removal changes the output. Removing a unit means replacing it by a
"corrupted" value: its average over jets (mean) or its value on a jet of the confusable class with the same pT (resample).
JEDI-linear has 186 active units and 1,216 connections between them.

**Method.** EAP estimates the effect of removing each connection $e$ from a gradient,
$A_e = \int_0^1 \frac{\partial m}{\partial M_e}\big|_{M = 1-\alpha G} d\alpha$ with $m = \langle \log p(\text{network's call}) \rangle$;
ACDC removes connections from the output backward whenever $\Delta \mathrm{KL}(p_{\rm full}\|p_{\rm circ}) < \tau$. A circuit is
scored by keeping only its units and corrupting all others:
$F = \frac{\mathrm{acc}_{\rm circ} - \mathrm{acc}_0}{\mathrm{acc}_{\rm full} - \mathrm{acc}_0}$.

**Result (figure `e29_auto_circuit.png`).** ACDC reaches $F = 0.955$ with 119 units; random sets of active units with the same count
per layer reach at most 0.58. Every circuit with $F \ge 0.9$ keeps all 13 cuts and 14–17 of the 17 named S objects. The hand program
with all output-layer units needs 156 units for $F = 0.976$; the extra units are 9 jet-context units, 4 pooled cuts and 17 h4 units.

**Checks.** Masked forward pass equals the network exactly (max $|\Delta\mathrm{logit}| = 0$); EAP attributions sum to the total
effect within 2–5 %; single-edge estimates correlate 0.64–0.84 with exact removals; untrained redraws give constant calls and no
circuit; two other trained JEDI networks give the same F-vs-size curve.

**Why it matters.** The hand-named cuts and S objects are also what an output-driven search keeps, but since the names cover almost
all active units at those layers this is consistency, not a test that singles the names out. The search shows that most of the
context round (9 of 11 units) and about 30 h4 units can be removed at a cost of about 2 % of accuracy above chance.

## Glossary entries (explain.GLOSSARY)

- **EAP (edge attribution patching)**: a first-order estimate of how much the output metric changes when one connection is switched
  from its real input to a corrupted input, computed for all connections at once from gradients; EAP-IG averages the gradient over
  several interpolation steps between clean and corrupted.
- **ACDC (automatic circuit discovery)**: greedy pruning that removes a connection whenever doing so raises the KL divergence between
  the full network's and the circuit's class probabilities by less than a threshold τ, working backward from the output.
- **Faithfulness F**: the share of the network's accuracy above the "everything corrupted" level that a circuit recovers:
  F = (acc_circuit − acc_all corrupted) / (acc_full − acc_all corrupted).
- **KL divergence**: $\sum_c p(c)\log(p(c)/q(c))$, the average information lost when class probabilities q replace p; zero when equal.
- **Literal circuit**: only the circuit's units compute from their inputs; every other unit is set to its corrupted value.
- **Mean / resample corruption**: replacing a unit by its average over jets, or by its value on a pT-matched jet of the confusable class.
- **Jaccard overlap**: |A ∩ B| / |A ∪ B| for two sets of units.

## Numbers for the structured result

Unedited 80.49 % [79.63, 81.32] (N 7,849). Smallest F ≥ 0.9: ACDC τ 0.003 119 units F 0.955 [0.945, 0.965] (mean), 115 units 0.942
[0.932, 0.952] (resample); random live max 0.576 / 0.472. Hand: handA 149 units 0.871 [0.859, 0.885] / 0.904; handA_allz 156 units
0.976 [0.968, 0.986] / 0.930; ACDC 148 units vs handA_allz McNemar 177/173 p 0.87. Overlap vs live-random Holm p = 1 everywhere.
