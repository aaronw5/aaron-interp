# E8 verification: classes as conjunctions of neuron windows on JEDI's own z and S (claim A6)

Script `interp/e8_verify.py`. Cluster job `anrunw-ai-e8-verify`: 1 CPU / 2 GiB, wall 615 s, peak 1404 MiB, 0.27 cores on average.
A smoke run, `anrunw-ai-e8-verify-smoke` (800 jets per class), came first.
Results: `results/jedi/e8_verify.json` (also at `$AI_RESULTS/jedi/e8_verify.json`). Log: `logs/e8-verify.log`.

## Code review of `interp/e8_conjunctions_jedi.py` and `interp/e8_fig.py`
- **Stages.** z is the d6 output before the ReLU; S is the qs1 pool output. The tail copy computes d4→d5→d6→relu→d7 from S and relu→d7 from z, with the layers' input quantizers and fused weights.
  - On the fresh sample, a per-jet override with the network's own z or S changes 0 calls.
  - The tail matches `run_jetwise` on 12 runs with 0 mismatched calls.
  - Running the tail on a subset of rows changes 0 calls (the dense layers act per jet).
- **Tree thresholds.** The mapping from tree split to window in `leaves_of` is correct: bin ≤ b ⇔ unit < G[b], and bin > b ⇔ unit ≥ G[b]. `inside` uses the same inequalities.
- **Split.** The grid, trees and medians come from the fit jets. Precision, coverage and interventions are measured on the evaluation jets. No leakage.
- **No particle edits.** Padded slots play no part. Mass, ΣpT and multiplicity are the stored observables.
- **The random-unit control is not value-matched.**
  - The rule units are set to the in-rule medians of jets called c; the random units are set to the class-c medians.
  - With class-c medians, the rule switches less than with in-rule medians:

    | level | g | q | W | Z | t |
    |---|---|---|---|---|---|
    | z, class-c medians (%) | 53.0 | 30.0 | 37.0 | 54.5 | 23.0 |
    | z, in-rule medians (%) | 54.4 | 31.2 | 42.0 | 62.0 | 25.9 |
    | S, class-c medians (%) | 29.8 | | | | |
    | S, in-rule medians (%) | 47.8 | | | | |
- **Random bars in the figure.** Panel a draws the random bar at the mean over all 10 draws (10.2–12.4 % at z). Five to nine of those draws contain a rule unit.
  - The reading's 1.4–4.1 % is the mean over the draws that contain no rule unit.
  - Its 3.8–7.3 % is the mean over the draws taken from outside the rule.
- **Untrained control.** The untrained network calls every jet t.
  - For t it has no jets called otherwise. For g, q, W and Z it has no jets called c.
  - Its result (trained values 100 % for g, 0 % for q, W, Z) is set by its random d7. It does not compare the rule with chance.
- **Figure and table.** I opened both and they match the JSON. Panel a shows the untrained bar at 100 % for g; the rank-matched value (0 %) is not drawn.
- **Log location.** The worker's script writes its log to `$PWD/logs`, which on the cluster is `/j-jepa-vol/aaron-interp/code/logs`. This has no effect on the numbers.

## Sample
- I drew `sample(per_class=7800, seed=23)` = 39,000 jets.
- I dropped the 4,545 jets that are also in E8's `sample(6000, seed=4)`, matched by the bytes of the (64, 3) particle array.
- I kept 6,000 jets per class at random (rng 23): 30,000 jets, 0 overlap with E8.
- Split: random 2/3 fit (20,000) / 1/3 evaluation (10,000), rng 1.
- Network accuracy: 80.72 % on all 30,000 jets, 81.56 % on the evaluation jets.
- Calls on the evaluation jets, g/q/W/Z/t: 2146 / 1847 / 2082 / 1898 / 2027.

## A. E8's rules and values (read from E8's JSON) applied to the fresh evaluation jets
- The rule units are set to E8's in-rule medians on the jets called otherwise (7,854–8,153 jets per class).
- "fresh med." = the same edit with the in-rule medians recomputed on the fresh fit jets.
- Necessity = in-rule jets called c, with the rule units set to E8's medians of jets called otherwise; share still called c.

| level | class | rule | precision / coverage, E8 → fresh | switched, E8 → fresh (fresh med.) | single units (fresh) | dose f = .25/.5/.75/1 (fresh) | necessity, E8 → fresh |
|---|---|---|---|---|---|---|---|
| z | g | z1 ≥ 4.33, z4 < 0.435, z8 < 3.87 | 96.5/74.8 → 98.1/76.4 | 54.4 → 54.0 (55.4) | z1 35.4, z4 2.0, z8 1.9 | 5.8/17.0/34.5/54.0 | 0.0 → 0.0 |
| z | q | z1 < 3.89, z3 > 0, z12 ≥ 1.38 | 98.2/72.8 → 98.4/71.5 | 31.2 → 31.8 (31.9) | z1 23.6, z12 4.7, z3 2.3 | 3.9/13.1/27.9/31.8 | 95.3 → 96.1 |
| z | W | z0 ≥ 1.35, z1 < 4.33, z4 > 0 | 98.4/82.8 → 98.3/82.0 | 42.0 → 40.6 (39.3) | z0 10.6, z4 9.3, z1 0.3 | 2.1/6.2/14.1/40.6 | 22.9 → 22.3 |
| z | Z | z0 < 1.35, z5 ≥ 0.44, z7 ≥ 1.04 | 96.7/76.6 → 97.4/77.3 | 62.0 → 61.4 (62.2) | z7 27.3, z0 18.8, z5 1.2 | 4.2/17.8/39.5/61.4 | 57.9 → 57.4 |
| z | t | z1 < 4.87, z4 > 0, z8 ≥ 2.59 | 95.8/69.5 → 96.2/69.7 | 26.0 → 26.2 (25.6) | z8 18.2, z1 3.9, z4 2.1 | 2.3/6.4/13.5/26.2 | 91.0 → 89.7 |
| S | g | S0 ≥ 22.4, S19 < 5.92, S60 ≥ 7.38 | 94.4/29.8 → 93.3/29.3 | 47.8 → 47.8 (48.6) | S0 36.4, S19 8.9, S60 3.5 | 8.9/22.0/35.6/47.8 | 0.3 → 0.2 |
| S | q | S0 < 25.2, S19 ≥ 5.92, S50 < 0.656 | 95.2/69.4 → 95.9/68.2 | 20.5 → 21.4 (21.3) | S19 9.9, S0 6.6, S50 5.0 | 3.9/8.4/15.7/21.4 | 81.0 → 80.1 |
| S | W | 0.656 ≤ S50 < 2.04, S62 < 2 | 70.4/55.8 → 70.7/55.3 | 5.2 → 5.5 (5.6) | S50 4.8, S62 0.9 | 1.0/2.2/3.8/5.5 | 97.4 → 97.7 |
| S | Z | S29 < 12.2, S50 ≥ 2.04 | 63.7/60.8 → 65.1/60.5 | 14.7 → 15.5 (15.5) | S29 9.1, S50 5.3 | 1.4/5.3/10.9/15.5 | 67.4 → 64.4 |
| S | t | S0 < 22.4, S13 ≥ 0.508, S50 ≥ 3.7 | 92.2/68.6 → 93.0/70.1 | 33.1 → 33.7 (32.8) | S50 6.8, S13 6.4, S0 3.1 | 3.5/8.2/14.9/33.7 | 71.5 → 70.8 |

- **Switched share by the jet's original call (fresh, z).**
  - g rule: q 99.95 %, t 62.0 %.
  - Z rule: W 100 %, q 89.7 %.
  - W rule: q 83.5 %.
  - q rule: g 95.5 %, Z 0.11 %.
  - E8's values are within 1–4 points.
- **Additivity at z.** For every jet and all five logits, the logit change from setting the 3 rule units equals the sum of the three single-unit logit changes: maximum |difference| 0.0. The mean |Δlogit| is 0.98–2.04.
- **Mass of the jets inside each window (25 / 50 / 75 %, all 30,000 fresh jets):**
  - z0 ≥ 1.35: 77.7 / 79.6 / 80.8 GeV
  - z7 ≥ 1.04: 89.6 / 90.9 / 92.3 GeV
  - z8 ≥ 2.59: 124.0 / 153.2 / 170.0 GeV
  - z12 ≥ 1.38: 27.5 / 36.4 / 46.1 GeV
  - S50 in [0.656, 2.04): 78.7 / 81.0 / 88.7 GeV
- **Which windows are E6 neurons.** z0 and z7 are E6's W and Z window neurons. z8 is not an E6 window: E6's top window is z10 (135–148 GeV).

## B. E8's tree procedure refit on the fresh fit jets
- **S level.** All five rules use the same units as E8, with thresholds within 0.2 of E8's:

  | class | precision (%) | coverage (%) | switched (%) |
  |---|---|---|---|
  | g | 92.6 | 29.6 | 49.8 |
  | q | 94.3 | 69.3 | 21.5 |
  | W | 65.7 | 66.8 | 5.4 |
  | Z | 64.8 | 60.8 | 15.2 |
  | t | 93.8 | 67.7 | 33.5 |
- **z level, g and t.** Same units as E8:
  - g: z1 ≥ 4.32, z4 < 0.424, z8 < 3.54 → precision 98.4 %, coverage 75.0 %, switched 55.1 %.
  - t: z1 < 5.57, z4 ≥ 0.152, z8 ≥ 2.62 → 96.7 %, 66.6 %, 26.8 %.
- **z level, q, W and Z.** One unit differs from E8's rule:
  - q: z1 < 3.88, z9 ≥ 1.74, z12 ≥ 1.38 (z9 replaces z3) → precision 98.0 %, coverage 73.2 %, switched 50.2 %.
  - W: z0 ≥ 1.36, z4 > 0, z9 < 1.44 (z9 replaces z1) → 98.2 %, 82.9 %, 41.5 %.
  - Z: z0 < 1.18, z7 ≥ 0.865, z8 < 0.965 (z8 replaces z5) → 97.6 %, 79.6 %, 75.7 %.

## C1. Every K-unit set, against the rule's own set
- **Procedure.**
  - Edit: K units, where K is the size of E8's rule (3, or 2 for W and Z at S), set to one of two value sources.
  - Value sources: (i) the class-c medians of fit jets called c; (ii) the medians of fit jets called c inside E8's rule, which is the rule's own value source.
  - Sample: the evaluation jets called otherwise.
  - Sets tested: all 560 triples of the 16 z neurons; at S, all 969 triples or 171 pairs of the 19 live units.
- **Results with value source (ii):**

| level | class | rule set: switched (rank / sets) | highest sets (switched) | sets with no rule unit: mean (max) | all sets: median |
|---|---|---|---|---|---|
| z | g | 55.4 % (6 / 560) | z1 z4 z9 61.6; z1 z4 z10 61.6 | 3.8 (12.8) % | 4.6 % |
| z | q | 31.9 % (39 / 560) | z1 z4 z9 52.5; z1 z9 z10 52.2 | 5.6 (31.6) % | 6.9 % |
| z | W | 39.3 % (10 / 560) | z0 z4 z7 48.3; z0 z4 z8 46.4 | 5.8 (26.4) % | 7.0 % |
| z | Z | 62.2 % (6 / 560) | z0 z7 z8 75.7; z6 z7 z8 70.4 | 4.4 (28.7) % | 5.8 % |
| z | t | 25.6 % (78 / 560) | z8 z10 z13 77.3; z8 z9 z10 66.5 | 7.7 (58.1) % | 7.5 % |
| S | g | 48.6 % (8 / 969) | S0 S19 S63 52.2; S0 S19 S29 51.2 | 4.5 (18.4) % | 6.0 % |
| S | q | 21.3 % (130 / 969) | S29 S30 S50 65.4; S17 S29 S50 55.8 | 8.8 (43.5) % | 9.5 % |
| S | W | 5.6 % (59 / 171) | S1 S29 17.3; S29 S50 17.0 | 4.7 (17.3) % | 3.9 % |
| S | Z | 15.5 % (1 / 171) | S29 S50 15.5; S29 S43 13.6 | 3.1 (8.8) % | 3.4 % |
| S | t | 32.8 % (146 / 969) | S13 S29 S50 95.2; S0 S29 S50 94.7 | 14.2 (80.6) % | 8.4 % |

- **With value source (i)**, the z ranks are 3, 38, 10, 8 and 79.
- **Percentile of the rule's set** among all sets: 86–99 % at z; 66–99 % at S.

## C2. The same tree and imposition in rotated bases
- **Procedure.**
  - Units are standardized on the fit jets: z (16 units); S (19 live units).
  - The rotations are 5 random orthogonal rotations (QR of a Gaussian matrix, rng 101–105), plus the identity.
  - Trees: E8's 3-deep tree on the rotated coordinates, with a plain 19-knot grid (no on/off knot).
  - Imposition: the rule's 2–3 rotated coordinates are set to their in-rule medians and rotated back. This edits a 2–3-dimensional subspace, as the neuron rule does.
  - A rotation round trip without an edit changes 0 calls.

| level | class | neuron basis, E8 rule (fresh): precision / coverage / switched | 5 rotations: precision | coverage | switched (mean) |
|---|---|---|---|---|---|
| z | g | 98.1 / 76.4 / 54.0 | 88.7–97.2 | 46.4–67.4 | 6.8–56.1 (26.3) |
| z | q | 98.4 / 71.5 / 31.8 | 96.2–99.0 | 65.9–76.8 | 4.0–36.6 (18.0) |
| z | W | 98.3 / 82.0 / 40.6 | 96.3–99.1 | 51.4–76.4 | 3.7–60.8 (28.6) |
| z | Z | 97.4 / 77.3 / 61.4 | 87.4–97.6 | 63.7–79.9 | 13.6–57.6 (38.8) |
| z | t | 96.2 / 69.7 / 26.2 | 92.2–95.8 | 59.6–82.1 | 2.3–69.4 (34.8) |
| S | g | 93.3 / 29.3 / 47.8 | 88.4–95.7 | 26.2–46.2 | 20.7–57.3 (37.1) |
| S | q | 95.9 / 68.2 / 21.4 | 89.0–94.2 | 61.3–68.5 | 12.6–28.4 (20.2) |
| S | W | 70.7 / 55.3 / 5.5 | 51.9–68.9 | 37.7–69.7 | 4.1–9.7 (6.4) |
| S | Z | 65.1 / 60.5 / 15.5 | 51.5–69.7 | 29.5–67.8 | 11.0–25.8 (15.6) |
| S | t | 93.0 / 70.1 / 33.7 | 89.6–91.8 | 64.9–75.5 | 13.7–78.3 (49.4) |

- **Comparison at z.**
  - The neuron basis has a higher precision than every rotation for g and t, and a precision within the rotation range for q, W and Z.
  - It has a higher coverage than every rotation for g and W. For q, Z and t, at least one rotation reaches a higher coverage.
- **Identity basis with the plain grid (no on/off knot).** The z rules give precision 96.5–99.0 % and coverage 68.9–81.3 %. For t, the rule changes to [z0, z1, z4], which switches 8.2 %.

## C3. Entering the windows instead of moving to the medians
- **Procedure.**
  - Jets: the evaluation jets called otherwise.
  - Edit: each rule unit that lies outside its window is moved just inside it. Units already inside are left as they are, and every edited jet satisfies the rule afterwards.
  - Target value, variant "inner": the 5 % quantile (lower bound) or the 95 % quantile (upper bound) of the fit jets called c inside the rule.
  - Target value, variant "edge": the window edge itself.
  - "donor": the rule units are copied together from a random fit jet called c inside the rule.
  - Reference: natural precision = the share of unedited in-rule jets that the network calls c.

| level | class | natural precision | inner | edge | donor | in-rule medians |
|---|---|---|---|---|---|---|
| z | g | 98.1 | 17.0 | 12.8 | 57.2 | 55.4 |
| z | q | 98.4 | 25.1 | 4.7 | 31.0 | 31.9 |
| z | W | 98.3 | 7.6 | 3.4 | 35.9 | 39.3 |
| z | Z | 97.4 | 21.5 | 9.2 | 57.8 | 62.2 |
| z | t | 96.2 | 7.2 | 5.7 | 24.6 | 25.6 |
| S | g | 93.3 | 22.4 | 19.1 | 55.3 | 48.6 |
| S | q | 95.9 | 9.7 | 5.9 | 22.0 | 21.3 |
| S | W | 70.7 | 2.4 | 1.7 | 6.5 | 5.6 |
| S | Z | 65.1 | 2.9 | 2.4 | 16.6 | 15.5 |
| S | t | 93.0 | 4.7 | 3.2 | 43.3 | 32.8 |

All values are % of the jets called c. Mean |change| of the rule units at z: inner 1.26–1.69; donor 2.34–3.04.

## Reading as measured here (proposed correction of E8's reading)
- **Description at z.** E8's fixed rules reproduce on 10,000 fresh, disjoint evaluation jets: precision 96.2–98.4 %, coverage 69.7–82.0 %.
  - Refitting on fresh jets keeps the units for g and t and changes one unit for q, W and Z.
  - 2–3-coordinate rules in random rotations of z reach 87–99 % precision and 46–82 % coverage.
  - So the description is not specific to the neuron basis. The neuron basis gives the highest coverage for g and W.
- **Imposition at z** (rule units set to the in-rule medians). Switched: g 54.0, q 31.8, W 40.6, Z 61.4, t 26.2 %.
  - Among all 560 three-neuron sets with the same value source, the rule's set ranks 6, 39, 10, 6 and 78.
  - The highest sets switch 61.6, 52.5, 48.3, 75.7 and 77.3 %. The t set {z8, z10, z13} contains E6's top window neuron z10, which is not in the t rule.
  - Sets with no rule unit switch 3.8–7.7 % on average, and up to 12.8–58.1 %.
- **Windows versus margin.** Moving the rule units just inside their windows switches 7.2–25.1 % (edge: 3.4–12.8 %), while 96–98 % of unedited in-rule jets are called the class.
  - Moving them to typical in-rule values (donor or median) switches 24.6–62.2 %.
  - The rule's logit change is the exact sum of the single-unit changes.
  - So the measured switch follows how far the units move along the additive logit margin through d7, not whether the jet satisfies the window conjunction.
- **Window identities.** z0 and z7 are E6's W and Z window neurons (jets inside: 78–81 and 90–92 GeV). The t rule's z8 window holds jets of 124–170 GeV and is not E6's top window (z10).
- **S level.** Reproduced within 1–3 points; refit gives the same units. Among all sets of the same size:
  - the Z rule's pair {S29, S50} ranks 1 of 171 (15.5 %);
  - the W rule's pair switches 5.6 %, against a mean of 4.7 % for pairs without a rule unit;
  - the t rule's set ranks 146 of 969 ({S13, S29, S50}: 95.2 %).
- **E8's untrained control.** It does not test the rule: the untrained network calls every jet t.
