# E29 verify — automated circuit discovery on JEDI-linear, rerun on a fresh sample with two added controls

Verifier for node E29. Script `interp/e29_verify.py` (imports the E29 classes from `interp/e29_auto_circuit_jedi.py` unchanged).
Jobs: `e29-verify` (mean corruption; 1 CPU / 2 GiB; 1,236 s; peak 1,795 MiB), `e29-verify-rs` (resample corruption; 1 CPU / 2 GiB;
1,673 s; peak 1,985 MiB), smoke `e29-verify-smoke`. Results `results/jedi/e29_verify_mean.json`, `results/jedi/e29_verify_resample.json`;
logs `logs/e29-verify.log`, `logs/e29-verify-rs.log`.

## Conclusion

On a new set of jets, the automated search again keeps all 13 first-layer particle cuts and 14–17 of the 17 hand-named jet
quantities, and again recovers about 95 % of the accuracy above chance with about 115 of 186 active units, far above random unit sets
of the same size. The added test shows that which jet quantities are kept matters: replacing the found set by another set of the same
size from the active units lowers the recovered accuracy, and sets with more of the hand-named quantities do better. One statement of
E29 holds only for one of its two ways of switching units off: the hand program with all output-layer units matches the search at
equal size when removed units are set to their average (mean corruption), but when they are set to values from a jet of the
confusable class (resample corruption) the search's circuit of the same size recovers 0.05 more of the accuracy (F 0.98 vs 0.93).

## What was checked

1. **Code review** of `interp/e29_auto_circuit_jedi.py` (masked forward, graph, EAP-IG, ACDC, literal node circuits, scorer, nulls,
   hand circuits, follow-up). Findings:
   - Exactness: the masked forward equals `Jedi.run` (max |Δlogit| 0.0, 0 calls differ; reproduced on the new sample, N 7,844). The
     l3 quantizer acting on s + g is handled by the four on/off combinations; correct.
   - Literal node circuit (`masks_from_nodes`): an edge reads the clean value iff its source unit is in the set; receivers outside
     the set only matter through their outgoing edges, which read the corrupted value. Correct. Edges outside the graph (dead
     sources, zero kernel entries) are always clean; dead units are constant. On the new test jets no unit outside the fit-sample
     live set varies (checked for every stage: 0 units), so nothing outside the graph is silently left clean.
   - Padded slots: the per-slot means and the donor activations include padded slots, as the network computes them; no particle
     count is taken from them. No split leakage: fit and test samples are disjoint by byte hash (151 shared jets removed in E29,
     156 here); ACDC is fitted on fit[::5] (every fifth row, so all five classes are represented in equal shares even if rows are class-ordered)
     and every F is on test jets.
   - Random-circuit null (E29 control a): `random_nodes` draws the u and S channels independently, while every found circuit has
     its u and S sets on the same channels (u_c → S_c is an identity pooling edge, so a u unit whose S unit is corrupted carries
     nothing, and an S unit whose u unit is corrupted is constant). The E29 null is therefore built from structurally weaker sets.
     A paired null was added (below); it changes the null mean by ≤ 0.04 F.
   - Random-circuit p-values in E29 sit at the 200-draw floor (0.005, Holm floor 0.08). Rerun with 500 draws.
   - McNemar (exact, two-sided) and Holm implementations: correct.
   - No bug that changes a reported number was found.
2. **Rerun on a fresh sample** (procedure identical to E29): fit / discovery `sample(1000, 3901)` (5,000 jets; ACDC on rows [::5],
   1,000 jets), test `sample(1600, 3902)` minus 156 jets shared with fit = 7,844 jets; donors rng 3903 / 3904; nulls on test[::8]
   (981 jets). Same graph (186 live units, 1,216 edges; live sets identical to E29), EAP-IG 5 steps, ACDC τ ladder 1e-5 … 3e-2.
3. **Added controls** for the ACDC circuits at τ 0.003 and 0.001 (500 draws each, F on test[::8]):
   - *paired random*: same number of units per layer drawn from live units, with the u and S units on the same channels and each
     drawn P channel's a channel kept (the comparison set with the same structure as the found circuits);
   - *S-swap*: the found circuit with only its u/S channel set replaced by a random set of the same size from the 19 live S
     channels (u and S on the same channels); everything else (cuts, P, g, s, h4, h5, z) fixed. This asks whether the particular
     second-round objects matter given the rest of the circuit (E29's thesis note says "same F with different S units excluded",
     which E29 did not test directly);
   - *g-swap* (τ 0.003): the found circuit with its g (jet-context) units replaced by every other g set of the same size (exhaustive).

Definitions (as in E29): **F** (faithfulness) $= (\mathrm{acc}_{\rm circ} - \mathrm{acc}_0)/(\mathrm{acc}_{\rm full} - \mathrm{acc}_0)$,
acc_0 = accuracy with every live unit corrupted, acc_full = unedited network; 1 = full accuracy, 0 = no better than everything
corrupted. **Literal (node-level) circuit**: the listed units compute normally, every other unit is replaced by its corrupted value.
**Mean corruption**: unit at its fit-sample average (per pT-rank slot for per-particle layers). **Resample corruption**: unit at its
value on a donor jet of the confusable class (W↔Z, q↔g, t→W/Z) with nearly the same jet pT. **ACDC-style pruning**: remove edges
from the logits backward while KL(full ‖ circuit) rises by less than τ. **EAP-IG**: integrated-gradient edge attribution.
**McNemar**: paired test on per-jet correctness (jets only circuit 1 right / only circuit 2 right). **Holm**: step-down
multiple-comparison correction. **ΔF interval**: 95 % paired bootstrap over test jets (1,000 replicates).

## Results (fresh sample; E29 values in brackets)

Unedited accuracy 81.49 % [80.62, 82.31] (E29 sample: 80.49 %). acc_0: mean 19.98 %, resample 13.53 %.
EAP-IG completeness: Σ A 1.454 vs metric drop 1.453 (mean); 3.078 vs 2.926 (resample).

| circuit | corruption | units | F [95 % CI] | l1 cuts | named S (of 17) | g units | P |
|---|---|---|---|---|---|---|---|
| ACDC τ 0.003 | mean | 116 [119] | 0.960 [0.951, 0.970] [0.955] | 13 | 16 (drops S52) [16] | g4, g49 [g4, g49] | 6 [8] |
| ACDC τ 0.001 | mean | 149 [148] | 0.981 [0.974, 0.989] [0.977] | 13 | 17 [16] | 5 [8] | 7 [9] |
| ACDC τ 0.003 | resample | 114 [115] | 0.943 [0.933, 0.953] [0.942] | 13 | 14 + S31 (drops S0, S52, S63) [same] | g4 [g4, g55] | 4 [5] |
| ACDC τ 0.001 | resample | 156 [158] | 0.977 [0.969, 0.985] [0.977] | 13 | 16 [17] | 8 [8] | 9 [9] |
| ACDC τ 0.01 | mean / resample | 83 / 93 | 0.851 / 0.846 [0.808 / 0.888] | 12 / 13 | 12 / 14 | | |
| handA (E11 h4/h5/z, 9 of 16 z) | mean / resample | 149 | 0.864 / 0.904 [0.871 / 0.904] | | | | |
| handA_allz (all 16 z) | mean / resample | 156 | 0.975 / 0.925 [0.976 / 0.930] | | | | |
| handA_allhead (all live h4/h5/z) | mean / resample | 182 | 0.990 / 0.980 [0.992 / 0.984] | | | | |
| handA_allhead without S0, S30 | mean / resample | 178 | 0.872 / 0.870 [0.864 / 0.869] | | | | |
| handA_allhead, g49 the only g unit | mean / resample | 172 | 0.973 / 0.938 [0.974 / 0.945] | | | | |

Every circuit with F ≥ 0.9 in either sample keeps all 13 l1 cuts and 14–17 of the 17 named S objects (E29's reading says 14–16;
the τ 0.001 resample circuit in E29 and the τ 0.001 mean circuit here keep all 17).

### Found circuits vs hand program at matched size (McNemar; ΔF = found − hand with 95 % paired bootstrap)

| comparison | ΔF [95 % CI] | only found right / only hand right | p |
|---|---|---|---|
| mean: ACDC τ 0.001 (149) vs handA_allz (156) | +0.006 [−0.001, 0.014] | 211 / 181 | 0.14 [E29: 177 / 173, p 0.87] |
| mean: ACDC τ 0.003 (116) vs handA_allz (156) | −0.015 [−0.023, −0.007] | 155 / 228 | 2e-4 [E29: p 2e-6] |
| mean: ACDC τ 0.001 (149) vs handA (149) | +0.118 [0.105, 0.130] | 814 / 247 | 4e-71 [E29: 3e-58] |
| **resample: ACDC τ 0.001 (156) vs handA_allz (156)** | **+0.053 [0.043, 0.062]** | **513 / 233** | **5e-25** [E29 follow-up JSON, 158 units: 440 / 199, p 7e-22; not in the E29 reading] |
| resample: ACDC τ 0.003 (114) vs handA_allz (156) | +0.018 [0.008, 0.027] | 383 / 289 | 3e-4 [E29: 347 / 286, p 0.017] |
| resample: ACDC τ 0.001 (156) vs handA_allhead (182) | −0.003 [−0.011, 0.005] | 223 / 238 | 0.51 [E29 158 units: p 0.088] |

H0: the found circuit and the hand program get the same jets right. Under mean corruption H0 is not rejected for ACDC at ~149
units vs handA_allz (p 0.14 here, 0.87 in E29). Under resample corruption it is rejected in both samples: at equal size (156 units)
the ACDC circuit recovers 0.053 more F, and the 114-unit ACDC circuit already exceeds handA_allz. The hand program with all z units
reaches the resample ACDC curve only when all live h4/h5/z units are added (182 units, p 0.51).

### Random-circuit nulls (F on 981 test jets; 500 draws; observed = found circuit on the same jets)

| circuit | corruption | observed F | null | null mean [95 % CI of mean] | 95th pct | max | p | Holm (5 tests per corruption) |
|---|---|---|---|---|---|---|---|---|
| ACDC τ 0.003 | mean | 0.938 | unpaired (E29 null) | 0.115 [0.104, 0.126] | 0.370 | 0.667 | 0.002 | 0.010 |
| ACDC τ 0.003 | mean | 0.938 | paired u/S, a/P | 0.158 [0.148, 0.172] | 0.423 | 0.677 | 0.002 | 0.010 |
| ACDC τ 0.003 | mean | 0.938 | S-swap | 0.793 [0.776, 0.805] | 0.926 | 0.938 | 0.008 | 0.016 |
| ACDC τ 0.001 | mean | 0.970 | paired | 0.482 | 0.771 | 0.900 | 0.002 | 0.010 |
| ACDC τ 0.001 | mean | 0.970 | S-swap | 0.919 | 0.972 | 0.973 | 0.25 | 0.25 |
| ACDC τ 0.003 | resample | 0.939 | unpaired | 0.117 [0.109, 0.124] | 0.284 | 0.485 | 0.002 | 0.010 |
| ACDC τ 0.003 | resample | 0.939 | paired | 0.133 [0.126, 0.142] | 0.324 | 0.462 | 0.002 | 0.010 |
| ACDC τ 0.003 | resample | 0.939 | S-swap | 0.701 [0.692, 0.711] | 0.870 | 0.933 | 0.002 | 0.010 |
| ACDC τ 0.001 | resample | 0.972 | paired | 0.419 | 0.639 | 0.809 | 0.002 | 0.010 |
| ACDC τ 0.001 | resample | 0.972 | S-swap | 0.915 | 0.972 | 0.988 | 0.10 | 0.10 |

- H0 (F): the found circuit is no more faithful than random live units with the same count per layer. Rejected at τ 0.003 and
  0.001 in both corruptions with the paired null (p 0.002, Holm 0.01; observed F above all 500 draws). Pairing u with S raises the
  null mean by 0.02–0.04 F; E29's unpaired null did not bias the conclusion.
- S-swap, H0: with everything else fixed, the specific u/S channel set gives no higher F than a random same-size set of live S
  channels. Rejected at τ 0.003 (16 of 19 / 15 of 19 S kept; p 0.008 mean, 0.002 resample). Mean F of the swapped circuits by the
  number of named S objects in the draw: mean corruption 14 → 0.769, 15 → 0.845, 16 → 0.919; resample 13 → 0.687, 14 → 0.721,
  15 → 0.793. A same-size draw with more named objects means one with fewer of S31/S58, so this measures that S31/S58 contribute
  less than the named 17 at this size. Not rejected at τ 0.001 (18–19 of 19 S kept, almost no room to swap; p 0.25 / 0.10).
- g-swap (τ 0.003, all 55 pairs for mean / all 11 singles for resample): the found g set has the highest F (0.938 / 0.939); other
  sets average 0.862 / 0.902; every top pair contains g49 (second best 0.926, a 12-jet difference out of 981). The single g unit
  carrying most of the context effect is g49 (mean) / g4 (resample); E29's "g4 + g49 or g55" is sample-dependent in its second unit.

## Competing hypotheses

| hypothesis | prediction | result |
|---|---|---|
| H_struct: any live units of the right per-layer count give the same F | found F ≈ random (paired) | excluded (0.94 vs null max 0.68, p 0.002) |
| H_Sany: given the rest of the circuit, any same-size S set works | found F ≈ S-swap mean | excluded at 15–16 of 19 S units (0.94 vs 0.79 / 0.70); untestable at 18–19 |
| H_hand=found (E29 reading): hand program with all z sits on the ACDC curve | ΔF ≈ 0 at matched size in both corruptions | holds for mean (ΔF +0.006, p 0.14); excluded for resample (ΔF +0.053, p 5e-25) |
| overlap test (planned): named sets overlap found sets above random | Jaccard above live-random null | untestable (named ≈ live), as E29 states |

## Figure check

`results/jedi/e29_auto_circuit.png` read. Readable; axes labelled; one message per panel. Two notes: (1) the random-null band is the
unpaired null at mean corruption only; (2) the legend does not say that filled hand-program markers are mean corruption and open ones
resample (the resample hand points sit below the resample ACDC curve, which is visible on close reading: open diamond at 156 units,
F 0.93, vs resample ACDC 0.977 at 158). Adding "filled = mean, open = resample" to the legend would fix (2).

## Corrected reading

Within JEDI-linear n64 (two disjoint samples, 7,849 and 7,844 test jets), every ACDC or EAP circuit with F ≥ 0.9 keeps all 13 live l1
cuts and 14–17 of the 17 hand-named S objects; the smallest such ACDC circuits have 114–119 units (F 0.94–0.96), and random live-unit
circuits with the same per-layer counts and paired u/S channels reach at most 0.68 (p 0.002 over 500 draws, Holm 0.01). With the cuts,
P, g, s, h4, h5 and z of the found circuit fixed, replacing its S set by another same-size set of live S units lowers F from 0.94 to
0.79 (mean) / 0.70 (resample) on average (p ≤ 0.008), so the selection among S units matters at that size; it favours the 17 named
objects over S31/S58. The named-set overlap test has no power because the names cover 13/13 live cuts and 17/19 live S units. The
hand program with E11-selected h4/h5/z (149 units) lies below the ACDC curve in both corruptions; with all 16 z units it matches
ACDC at equal size under mean corruption (McNemar p 0.87 / 0.14) but lies 0.05 F below it under resample corruption (156 units:
0.925–0.930 vs 0.977, p 5e-25 / 7e-22), and needs all live h4/h5/z units (182) to match. Keeping only g49 (mean) or g4
(resample) among the 11 g units costs 0.02 / 0.04 F; removing S0 and S30 costs 0.11–0.13 F; untrained redraws give constant calls.

## Issues for the ledger

- E29 reading: "14–16 of the 17 hand-named S objects" → 14–17.
- E29 reading and Conclusion: "with all z units it matches ACDC at 148 units (McNemar p 0.87)" and "The hand-built program with all
  output-layer units does as well at about 156 units" hold for mean corruption only; under resample corruption the hand program with
  all z units is 0.05 F below ACDC at equal size (p 7e-22 in E29's own follow-up JSON; p 5e-25 here).
- E29 thesis note "alternative 'same F with different S units' excluded" was not tested in E29; the S-swap control here supports it
  at 15–16 of 19 S units only.
- E29 random-circuit p-values (200 draws, Holm floor 0.08) are superseded by the 500-draw paired nulls here (Holm 0.01).
- New glossary terms used here: *paired random null* (random units with u and S on the same channels), *S-swap null* (found circuit
  with only its second-round object set re-drawn), *g-swap*.

## Explainer entry (for site/explain.py, Paper A, after E29)

- **Question.** Does the agreement between the automated circuit and the hand-named program survive a new sample, and does the
  choice of second-round objects matter once the rest of the circuit is fixed?
- **Background.** E29 found that ACDC keeps all named cuts and nearly all named S objects, but the names cover almost every active
  unit, so overlap alone cannot tell "hand names" from "active units".
- **Method.** Same pipeline on `sample(1000, 3901)` / `sample(1600, 3902)`. New controls: (i) random circuits with
  $u_c$ and $S_c$ on the same channels (since $S_c = \tfrac1{16}\sum_i u_{i,c}$); (ii) the found circuit with its S set
  replaced by a random same-size subset of the live S units; (iii) all same-size g sets.
- **Result.** $F = 0.96$ (mean) / $0.94$ (resample) at 114–116 units; paired random null max 0.68; S-swap mean 0.79 / 0.70,
  rising with the number of named S objects in the draw. Hand program with all z: equal to ACDC under mean corruption, 0.05 lower
  under resample corruption.
- **Checks.** Masked forward = network (0 calls differ); no unit varies on test that is constant on fit; Holm over 5 tests per
  corruption.
- **Why it matters.** The algorithm's choice of second-round objects is not arbitrary at this size, and it ranks the 17 named
  objects above S31/S58; the hand program's output-layer description is complete only under one of the two corruption choices.
