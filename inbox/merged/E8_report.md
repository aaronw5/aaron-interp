# E8 — classes as conjunctions of unit windows, imposed on JEDI's own z and S (claim A6)

Script: `interp/e8_conjunctions_jedi.py` (figure and table: `interp/e8_fig.py`). Cluster job `anrunw-ai-e8-rerun` (1 CPU / 2 GiB;
wall 319 s, peak 1904 MiB, 0.29 cores average). Results: `results/jedi/e8_conjunctions.json`; log `logs/e8-rerun.log`;
figure `results/jedi/e8_conjunctions.png`; rule table `results/jedi/e8_rules_table.png` (for the user).

## Procedure
- Network: JEDI-linear 3-feature n64, fixed-point forward pass (`jedi_core.Jedi(64)`); call = argmax of the logits.
- Sample: `sample(per_class=6000, seed=4)` = 30,000 test jets; random split (rng 0): 20,000 fit / 10,000 evaluation.
- Checks: accuracy 81.29 % on the seed-0 10,000-jet sample (E1: 81.29 %); 80.84 % on the 30,000 jets, 80.95 % on the
  evaluation jets (calls g/q/W/Z/t = 2110/1990/2055/1821/2024). A per-jet override of z with the network's own z, and of S
  with its own S (`run_jetwise`), changes 0 of 10,000 calls. The interventions run through a copy of the network's tail
  (d4→d5→d6→relu→d7 from S; relu→d7 from z; same layers, quantizers, fused weights); on 29 runs checked against
  `run_jetwise` it gave 0 mismatched calls.
- Step 1 (description), per class c and level (z: 16 pre-ReLU neurons; S: the 19 live S units of E1): a 3-deep decision
  tree (sklearn, min_samples_leaf 100) on the fit jets predicting call == c. Thresholds are restricted to a grid per unit:
  the 5–95 % quantiles (19 knots) plus the on/off threshold (unit > 0). For z the quantiles are taken over the values > 0,
  because d7 reads relu(z) and all values ≤ 0 are the same input to d7. Rule = the leaf with the highest F1 on the fit jets.
  Precision (share of jets inside the rule that the network calls c) and coverage (share of the network's c calls inside
  the rule) are measured on the evaluation jets.
- Step 2 (z) and step 3 (S), intervention: on evaluation jets called otherwise, the rule's units are set to the medians of
  fit jets called c inside the rule; other units are untouched. Metric: share now called c. Further measurements:
  - dose: the rule units moved a fraction f = 0.25 / 0.5 / 0.75 / 1 of the way to those medians;
  - AND test: each single rule unit and each pair imposed alone (same values);
  - necessity: evaluation jets called c inside the rule get the rule units set to the medians of fit jets called otherwise.
- Controls:
  - the same number of random units set to class-c medians: 10 draws from all units, plus 10 draws from the units outside
    the rule;
  - the rule imposed on `Jedi(untrained=True, seed=0)`, once with the trained values and once rank-matched (each unit set
    to its own quantile at the rank that the trained value has in the trained distribution);
  - all units (16 z / 19 live S) set to the class-c medians.

## Numbers (evaluation jets; ~7,900–8,200 jets called otherwise per class)

| class | level | rule | precision / coverage | switched: rule | best single / pair | random all (no rule unit drawn) / outside rule | all units | still c after rule → other medians |
|---|---|---|---|---|---|---|---|---|
| g | z | z1 ≥ 4.33, z4 < 0.435, z8 < 3.87 | 96.5 / 74.8 % | 54.4 % | z1 36.0 / z1&z4 47.1 % | 10.2 (2.7) / 3.8 % | 100 % | 0.0 % |
| q | z | z1 < 3.89, z3 > 0, z12 ≥ 1.38 | 98.2 / 72.8 % | 31.2 % | z1 23.0 / z12&z1 32.4 % | 12.4 (2.1) / 3.9 % | 100 % | 95.3 % |
| W | z | z0 ≥ 1.35, z1 < 4.33, z4 > 0 | 98.4 / 82.8 % | 42.0 % | z0 10.7 / z0&z4 35.9 % | 11.1 (1.4) / 7.3 % | 100 % | 22.9 % |
| Z | z | z0 < 1.35, z5 ≥ 0.44, z7 ≥ 1.04 | 96.7 / 76.6 % | 62.0 % | z7 27.8 / z7&z0 53.2 % | 11.7 (4.1) / 4.5 % | 100 % | 57.9 % |
| t | z | z1 < 4.87, z4 > 0, z8 ≥ 2.59 | 95.8 / 69.5 % | 25.9 % | z8 17.8 / z8&z1 23.8 % | 11.0 (1.9) / 5.1 % | 100 % | 91.0 % |
| g | S | S0 ≥ 22.4, S19 < 5.92, S60 ≥ 7.38 | 94.4 / 29.8 % | 47.8 % | S0 34.5 / S0&S19 45.2 % | 4.5 (2.7) / 4.6 % | 100 % | 0.3 % |
| q | S | S0 < 25.2, S19 ≥ 5.92, S50 < 0.656 | 95.2 / 69.4 % | 20.5 % | S19 9.8 / S19&S0 15.1 % | 10.6 (11.5) / 7.9 % | 100 % | 81.0 % |
| W | S | 0.656 ≤ S50 < 2.04, S62 < 2 | 70.4 / 55.8 % | 5.2 % | S50 4.4 % | 6.9 (6.6) / 4.3 % | 100 % | 97.4 % |
| Z | S | S29 < 12.2, S50 ≥ 2.04 | 63.7 / 60.8 % | 14.7 % | S29 8.3 % | 2.8 (1.8) / 3.2 % | 100 % | 67.4 % |
| t | S | S0 < 22.4, S13 ≥ 0.508, S50 ≥ 3.7 | 92.2 / 68.6 % | 33.1 % | S50 6.7 / S50&S13 23.6 % | 9.3 (5.6) / 12.3 % | 100 % | 71.5 % |

- Random draws from all units: at z, 5–9 of the 10 draws contain a rule unit. The draws that contain one switch up to 46 %
  (g: draws with z1 switch 46.0 and 34.6 %). The draws without one average 1.4–4.1 %.
- Sum of the single-unit shares vs the conjunction:
  - z: g 40.2 vs 54.4; q 29.9 vs 31.2; W 21.1 vs 42.0; Z 47.4 vs 62.0; t 24.6 vs 25.9 %.
  - S: g 47.9 vs 47.8; q 21.2 vs 20.5; W 5.3 vs 5.2; Z 13.5 vs 14.7; t 16.3 vs 33.1 %.
- Dose, share called c at f = 0.25 / 0.5 / 0.75 / 1:

  | class | z | S |
  |---|---|---|
  | g | 6.3 / 17.0 / 35.1 / 54.4 | 8.5 / 21.9 / 35.9 / 47.8 |
  | q | 4.0 / 12.5 / 27.1 / 31.2 | 3.8 / 8.0 / 14.9 / 20.5 |
  | W | 2.1 / 5.9 / 14.1 / 42.0 | 0.9 / 2.0 / 3.5 / 5.2 |
  | Z | 4.0 / 17.3 / 38.7 / 62.0 | 1.2 / 4.7 / 10.2 / 14.7 |
  | t | 2.6 / 6.7 / 13.4 / 26.0 | 3.7 / 8.1 / 15.1 / 33.1 |

- Switched share at z by the jet's original call:

  | rule | jets originally called g | q | W | Z | t |
  |---|---|---|---|---|---|
  | g | – | 99.9 | 35.7 | 18.5 | 61.0 |
  | q | 95.6 | – | 3.8 | 0.05 | 19.9 |
  | W | 44.3 | 84.3 | – | 37.2 | 2.4 |
  | Z | 37.9 | 89.5 | 100 | – | 21.7 |
  | t | 31.5 | 6.9 | 45.1 | 18.8 | – |

- Other positive leaves imposed at z: t leaf [z8 < 2.59, z10 ≥ 2.55, z13 < 1.23] (precision 93 %, coverage 4 %) switches
  56.4 %; g leaf [z1 ≥ 4.33, z4 < 0.435, z8 ≥ 3.87] (63 %, 5 %) switches 65.7 %.
- Untrained network (seed 0):
  - It calls every jet t (accuracy 20.0 %), so the t row has no other-called jets. 15 of its z units vary across jets; 1 of
    its 19 live-S units varies.
  - With the trained values, the g rule switches 100 % of its jets into g; the q, W and Z rules switch 0 %. Setting all
    16 z to the trained g medians also gives 100 % g, and the trained q/W/Z medians give 0 %. Its d7 maps the trained
    z1 ≈ 6.4 to g.
  - Rank-matched values: 0 % for g, q, W and Z at z and at S.
- The trees fitted on the untrained network's own z and S give no rule (one class called).

## Windows in GeV and negative poles (all 30,000 jets; mass = jet mass, IQR of the jets inside the window)

**z units**

| unit | rule window: mass IQR (GeV) | on side (> 0) | off side (≤ 0) |
|---|---|---|---|
| z0 (E6 W window, peak 75 GeV) | z0 ≥ 1.35: 78–81 | 16,846 jets: 35 % true W, 29 % q; median mass 74 GeV, multiplicity 39 | 13,154 jets: 41 % t, 28 % Z; 112 GeV, multiplicity 58 |
| z7 (E6 Z window 94–98) | z7 ≥ 1.04: 89–92 | 8,603 jets, 59 % Z, median 90 GeV | – |
| z8 | z8 ≥ 2.59: 124–170 | 40 % t, 30 % g; median 106 GeV, multiplicity 61 | W/q/Z; 78 GeV, multiplicity 37 |
| z1 | g rule z1 ≥ 4.33: 60–128, multiplicity median 64 | – | 2,913 jets: 47 % W, 34 % Z, 0.2 % g; 80 GeV, multiplicity 34 |
| z4 | – | W/Z/t 29 % each; 90 GeV | 11,141 jets: 39 % g, 45 % q; 51 GeV |
| z12 | z12 ≥ 1.38: 27–46 | 36 % q, 29 % W; 59 GeV | – |
| z3 | – | q 31 %, W 34 %; multiplicity 34 | t 35 %, g 29 %; multiplicity 59 |
| z5 | – | – | 6,959 jets: 58 % t, 32 % g; 154 GeV, multiplicity 64 |

The median ΣpT is 990–1040 GeV on both sides of every rule unit.

**S units (rule windows)**

| class | window: mass IQR (GeV) |
|---|---|
| W | 0.656 ≤ S50 < 2.04: 79–89 |
| Z | S50 ≥ 2.04: 91–163 |
| t | S50 ≥ 3.7: 143–173 |
| q | S50 < 0.656: 31–59 |

S50 is ΣpT·r² over ranks 0–38 (E5).

## Reading (scoped to these measurements)
- Description at z: for all five classes, a 3-window conjunction on z describes the network's calls with precision
  95.8–98.4 % and coverage 69.5–82.8 %. At S, the g, q and t rules have precision ≥ 92 % (coverage of g 29.8 %). The W
  and Z rules at S reach 70.4 / 55.8 % and 63.7 / 60.8 %: they share the S50 axis and split it at 2.04.
- Imposing the z rule on jets called otherwise:
  - The rule switches 26–62 % of those jets (g 54, Z 62; W 42, q 31, t 26).
  - Random units with no rule unit in the draw switch 1.4–4.1 %; draws outside the rule, 3.8–7.3 %.
  - All 16 neurons switch 100 %, so 38–74 points of the switch need units outside the rule.
  - The switches are concentrated between neighbouring classes: g rule on q calls 99.9 %, q rule on g calls 95.6 %, Z rule
    on W calls 100 %, W rule on q calls 84 %. From distant classes the shares are lower (Z rule on t 21.7 %, q rule on Z
    0.05 %).
- AND test at z: for W and Z the conjunction exceeds the sum of its single units (W 42.0 vs 21.1 %; z0&z4 35.9 % vs
  10.7 + 10.1 %). For q and t the conjunction is about the sum of the singles, and one unit carries most of it (z1, z8).
  At z the logits are one dense layer (d7) on relu(z), so each unit's contribution to the logit margin is additive. The
  super-additive switched shares are the argmax threshold applied to that sum.
- Necessity: setting the rule units to the medians of jets called otherwise removes the g call (0 % still g at z, 0.3 %
  at S). For q and t, 91–95 % of the calls remain at z: other neurons carry these calls without the rule units.
- At S: g 47.8 % and t 33.1 % (random 4.5 / 9.3 %); q 20.5 %; Z 14.7 %; W 5.2 % (random 6.9 %).
- Against the pre-registered expectations:
  - thesis ("≥ 50 % at z for every class, random ≤ 5 %, untrained ~0"): met for g and Z;
  - alternative ("switches at the J33b level 5–36 % with the all-unit ceiling far above"): met for q and t at z, and for
    q, W and Z at S;
  - W at z (42 %) is between the two.
- The untrained control with trained values is not ~0 for g (100 %). Its random d7 sends z1 ≈ 6.4 to g. With rank-matched
  values it is 0 % for every class.

## Possible wording for A6 (for the ledger)
"The network's calls for each class are described by a conjunction of 2–3 neuron windows on z: precision ≥ 96 %,
coverage 70–83 %. The windows include the mass windows of E6: z0 78–81 GeV for W, z7 89–92 GeV for Z, z8 124–170 GeV for t.
Imposing the conjunction on the network's own neurons moves 26–62 % of the jets called otherwise into the class; random
units of the same number move 1–7 %. The moved jets come mostly from neighbouring classes (q ↔ g, W ↔ Z). The remaining
neurons carry the rest of the logit margin: all 16 neurons move 100 %."

## Run notes
- First launch (`e8-run`): the Python process sat in D state (`ceph_mdsc_wait_request`) for 10 min at 3 mCPU on node
  `k8s-haosu-07.sdsc.optiputer.net`. I deleted the job and relaunched it as `e8-rerun`, which ran on another node. That node
  is a candidate for the exclusion list in job.sh; I did not edit job.sh.
- `./sync.sh` reported tar errors on `.git/objects/*` ("Cannot unlink: Directory not empty"). The md5 sums of the script
  and `jedi_core.py` on the volume match the local files.
- A previous worker's run (`e8-run3`, same script without the AND-test, rank-matched and draw-unit fields) gave the same
  numbers on every shared field.
