# E30: weak activations and negative poles of S29, S1, S0, z0, z7, z10 (claims A3, A5, A11; PAPERS.md "What to look for")

Scripts: `interp/e30_weak_poles_jedi.py` (cluster jobs e30-quick [smoke run, 400/320 per class], e30-run [1 CPU / 3 GiB; the
2 GiB attempt was killed while loading data, the 3 GiB run peaked at 1.5–3.0 GiB], e30-fig [example jet]), `interp/e30_plot.py`
(figure, rendered locally from the JSON). Results: `results/jedi/e30_weak_poles.json`, `results/jedi/e30_fig_jet.npz`, figure
`results/jedi/e30_weak_poles.png`. Logs: `logs/e30-run.log`, `logs/e30-quick.log`, `logs/e30-fig.log`.

## Conclusion
The three second-round objects tested do the same thing in their weakest jets as in their strongest. When one particle is moved
outward or inward by 0.01–0.03 rad, or one hard particle's pT is nudged by 2 %, the network's response per unit of formula
prediction is the same in every tenth of the activation range. S1 and S0 have no exception. S29 (a sum over particles that
switches on for wide-angle, harder particles) responds at 0.79 of the prediction in its weakest tenth, against 1.02 in the
strongest. For the mass-window neurons z0 (about 75 GeV) and z7 (about 94 GeV), the off (negative) side is not a second
signal. It continues the same mass read away from the window: z0 is most negative for heavy top jets and z7 for light quark jets.
The ReLU discards this side, and passing it to the output layer costs 10–15 points of accuracy. Not settled: the weak-range
formula test for the z neurons gave no usable reference, because the best available z formula does not reproduce the network's
response to a change in one S unit even in the strongest jets.

## Procedure
- Network: JEDI-linear 3-feature n64, fixed point (`jedi_core.Jedi(64)`). Untrained control: `Jedi(64, untrained=True, seed=0)`
  (weights redrawn from the trained mean and sd).
- Jets: fit sample `sample(2000, 3001)` (10,000 jets); test sample `sample(1600, 3002)` minus the 308 jets also in the fit
  sample (N = 7,692). Fixed-point accuracy: fit 80.99 %, test 81.62 %. The test value is 0.3 points above the 80.6–81.3 %
  reference range, which is about 0.7 of one binomial SE (0.44 points at N = 7,692).
- Checks: the network with S overridden by its own S gives max |Δz| = 0. Recomputing d7 in float on ReLU(z) gives
  max |Δlogit| 0.107 and 0.22 % of calls differ (the ReLU bypass of Part 3 adds a float term to the fixed-point logits).
- Units, with what they compute (E5/E5v/E6/E7):
  - S29: Σ over all 64 rank slots of ReLU(4.44 ln pT + 93.8 r − 21.2)/16. This is a "hinge": a particle contributes once it is
    beyond an r that falls with pT (0.23 at 1 GeV, 0.12 at 10 GeV). It is correlated with ΣpT·r² + Σr² + soft count.
  - S1: Σ over ranks ≥ 1 of ReLU(0.55 ln pT + 89.3 r − 22.9)/16. This counts particles beyond r ≈ 0.24 almost regardless of pT
    (≈ Σ_{ranks ≥ 1}(r − 0.2)₊).
  - S0: (0.32 pT + 1.40)/16 summed over the 14 hardest particles (the direct term), plus a jet-context term through g49, which
    reads ΣpT of all particles.
  - z0, z7, z10: pre-ReLU d6 neurons whose positive side is a jet-mass window near 75, 94–98 and 135–148 GeV.
- Decile: one tenth of the jets ordered by the unedited unit value. Edges are taken from the fit sample. S units use all jets;
  z units use jets with z > 0. Ties at zero are broken at random so that deciles stay equal in size (S1 is exactly 0 on 10.3 %
  of jets, so its decile 1 is all zeros).

### Part 1: does the formula fitted on the strongest 30 % of jets describe the weaker 70 %?
- Main formula, fitted on the top 3 fit deciles only:
  - S29: a + b·(ΣpT·r²) + c·Σr² + d·N(pT < 2, r > 0.1), and separately a + b·(E5 hinge sum).
  - S1: a + b·Σ_{ranks ≥ 1}(r − 0.2)₊.
  - S0: a + b·(ΣpT of the 14 hardest).
  - z units: E7's 32-term hinge formula on the 19 live S units (`e7_S.json`, fitted on 20,000 other jets), applied unchanged.
    The same 32 terms were also refit on the top 3 deciles.
- In each test decile, three predictions were compared:
  - "top-fit": the formula fitted on the top 3 deciles;
  - "same-form": the same terms refit on that decile;
  - "free": a greedy 3-term (S) or 8-term (z) search refit on that decile. For S units the search runs over ~200 energy-flow
    quantities (counts, ΣpT, Σ(r − R)₊, ΣpT·r^β, in all slots or in the unit's rank window) plus the hinge sum. For z units it
    runs over S_c, (S_c − t)₊ and (t − S_c)₊ at 9 quantile knots.
- Metrics: RMSE (root-mean-square error) and mean residual (bias), both in units of the unit's sd over all fit jets. Held-out
  R² within a decile is also stored, but it is dominated by the narrow range of values inside one decile (strongly negative for
  any formula with a small offset), so the RMSE comparison is the one used.
- Test: H0: RMSE(top-fit) = RMSE(free) in the decile; 300-resample bootstrap p; BH (Benjamini–Hochberg false-discovery
  correction) over 60 cells. A "second job" would show as a low top-fit fit plus a free refit that needs other quantities.

### Part 2: causal dose, i.e. does a fixed-size edit move the unit by the predicted amount in every decile?
- Edit slope = Σ_jets measured ΔS / Σ_jets predicted ΔS within a decile, with signs aligned so that ± edits add. A slope of 1
  means the network moves exactly as the formula predicts. The measurement is the network's own S after the edit, with g
  recomputed.
- S29 and S1: one particle in the unit's rank window, whose hinge is on before or after the edit, is moved radially by
  Δr ∈ ±{0.01, 0.02, 0.03} rad. Its pT, its azimuth and every other particle stay fixed, so the rank order is unchanged.
  Prediction: Δ(hinge sum) with E5's coefficients, not refit. This is the check E5v made at slope 0.99, now split by decile.
  Matched control: the same Δr applied to a particle the formula says cannot matter. For S1 this is the rank-0 particle, which is
  outside its window. For S29, whose window is all 64 slots, it is a particle whose hinge is off both before and after the move.
- S0: the pT of one of the 14 hardest particles is multiplied by 1 ± 0.02 and the ranks are re-sorted. Prediction: the top-fit
  coefficient (0.0200 per GeV) × Δ(ΣpT of the 14 hardest). This equals the direct per-particle term 0.32/16 = 0.020 per GeV,
  so the prediction leaves out the context route through g. Control: the same nudge on a particle of rank ≥ 14.
- z units: an S-level intervention. The S unit with the largest E7-predicted effect is set to S ± 0.5 sd (central difference),
  and h4, h5 and z are recomputed by the network. Prediction: the E7 formula. Control: the live S unit absent from the formula,
  shifted by the same 0.5 sd.
- Untrained control: the same edits on the untrained network (the unit's measured change divided by the trained prediction).
- Statistics: 1,000 bootstrap resamples over jets for each slope. H0: slope(decile d) = slope(top 3 deciles); H1: different.
  Tested with a bootstrap difference and a 2,000-shuffle permutation of decile membership, with BH over 42 tests (6 units × 7
  lower deciles).

### Part 3: negative poles of z0, z7, z10 (pre-ReLU z < 0)
- Descriptive: class mix of the most negative 1 % and 10 % of test jets; median jet mass; mean z per jet-mass decile; AUC of
  −z per class within z < 0 (AUC = probability that a random jet of that class has a more negative z than a random other jet);
  Spearman correlation with each live S unit; a greedy 8-term hinge fit of the negative side on S (fit on fit jets with
  z < 0, R² on test jets with z < 0).
- Causal role. In the trained net, d7 sees only ReLU(z), so the negative side has no effect on the logits ("clamp").
  ReLU bypass: logits' = logits + α·min(z_k, 0)·W7[k, :], where W7[k, :] is the d7 row of neuron k. With α = 1 this is the
  slope the output layer applies to the positive side (the linear read); α = 0.5 was also run. Metrics: calls changed, change
  in accuracy, McNemar test on paired correctness (exact binomial on jets that become wrong vs right).
- Control: 20 random unit directions v in h5. Each gives z_r = a·(v·h5) + b, scaled so that the same fraction of jets is
  negative as for z_k and the sd equals z_k's, and is passed through the same W7 row.

## Results (test jets, 95 % bootstrap intervals)

### Part 2: edit slope by decile (figure, left panel)
| unit (what it computes) | N | overall slope | deciles 1 … 10 | top 3 deciles | control / predicted | untrained |
|---|---|---|---|---|---|---|
| S29 (Σ wide-angle hinge, all ranks) | 7,680 | 0.999 [0.977, 1.021] | 0.79, 0.98, 1.03, 1.02, 1.06, 1.04, 1.02, 1.06, 1.07, 0.94 | 1.020 [0.976, 1.065] | 0.04 (d1) – 0.18 (d10); all 0.120 [0.112, 0.129] | 0.000 (unit does not move) |
| S1 (Σ (r − 0.24)₊, ranks ≥ 1) | 5,717 | 0.960 [0.929, 0.998] | 0.97, 0.99, 0.88, 0.95, 0.97, 0.94, 1.05, 0.98, 0.97, 0.94 | 0.964 [0.906, 1.021] | 0.000 exactly (rank-0 particle) | 0.000 |
| S0 (ΣpT of 14 hardest + context) | 7,634 | 2.65 [2.58, 2.72] | 2.46, 2.80, 2.59, 2.75, 2.76, 2.75, 2.71, 2.90, 2.70, 2.07 | 2.55 [2.44, 2.67] | 0.19 [0.17, 0.22] (rank ≥ 14 particle) | 0.000 |
- S29, decile 1: 0.79 [0.73, 0.85] vs top 1.02. Difference −0.23; permutation p 0.0005; BH q 0.0014. It is the only S-unit
  decile that differs from the top after BH. S1: smallest q 0.22. S0: smallest q 0.09 (decile 2, 2.80, which is higher, not
  lower). S0's decile 10 (2.07 [1.88, 2.25]) belongs to the top group and was not tested.
- S0's slope of ≈ 2.6 relative to the direct per-particle term matches E5's 2.5–3.2× (the jet-context route through g49) and is
  the same in deciles 1–9. The rank ≥ 14 control moves S0 by 0.19 of the prediction; this goes through g, which reads ΣpT of all
  particles.
- S29 control: a particle with its hinge off moves S29 by 0.12 of a matched on-hinge move. The network's u29 has a small
  response below the fitted hinge onset, which the hinge formula sets to zero.
- Calls changed by the single-particle edits: S29 1.94 %, S1 0.89 %, S0 0.65 %.
- z units (S-level ± 0.5 sd on the unit with the largest predicted effect: z0 → S50, z7 → S1, z10 → S50):
  - Per-jet correlation of measured with predicted Δz: 0.08, −0.34, −0.54.
  - Top-decile slopes: −1.13 [−1.26, −1.01], −0.88 [−0.95, −0.81], −3.54 [−4.89, −2.83]. The network moves opposite to the
    formula's prediction already in the strongest jets.
  - Controls (formula predicts 0): 0.10, 0.29, −0.33 of the predicted size. Untrained: −0.62, +0.22, +0.74.
  - The E7 formula's partial derivative with respect to a single S unit does not match the network. The formula was fitted on
    S values that move together on real jets, and a one-unit shift leaves that joint distribution. There is therefore no valid
    strong-range reference for z. The BH flags on z cells (19 of 21) measure this formula failure and are not read as a
    weak-range effect.

### Part 1: top-3-decile formula in the lower deciles
- S29 (3-term named form):
  - Top-fit RMSE: 0.12–0.16 sd in deciles 1–6 and 0.15–0.29 in 7–10.
  - Same-form refit: 0.03–0.04 sd in deciles 1–6. Free 3-term refit: 0.03–0.04 sd in deciles 1–6.
  - Top-fit bias: +0.05 to +0.14 sd in deciles 1–6.
  - The free search picks the hinge sum in 10 of 10 deciles. The other two terms vary (hard-particle counts, Σ ln pT beyond
    r 0.025–0.1).
- S1:
  - Top-fit RMSE: 0.13–0.25 sd in deciles 2–7, bias +0.10 to +0.12 sd.
  - Same-form refit: 0.04–0.08 sd. Free: 0.03–0.08 sd.
  - The free search picks the hinge sum or a Σ(r − 0.2…0.3)₊ sum in 9 of 9 non-zero deciles.
- S0:
  - The ΣpT(14 hardest) line fitted on the top 3 deciles does not extend downward: R² −0.22 on all test jets, bias +0.85 to
    +1.6 sd in deciles 1–7.
  - Same-form refit: 0.03–0.10 sd in deciles 2–8.
  - The free search picks ΣpT of all particles in 9 of 10 deciles (the context route), and ΣpT(pT > 15) in decile 10. The
    top-decile line is too steep for the weak range: S0 grows more slowly than linearly in ΣpT (E5: log ΣpT R² 0.93).
- RMSE(top-fit) − RMSE(free) is positive in all 30 S cells after BH: +0.07 to +0.18 sd for S29 and S1, +0.5 to +1.5 sd for S0.
  The same terms, refit, close nearly all of the gap (same-form ≈ free). The weak range is therefore described by the same
  quantity with a different offset or scale, not by different quantities.
- z units:
  - E7 formula unchanged: R² on positive test jets 0.66 (z0), 0.68 (z7), 0.82 (z10). Per-decile RMSE 0.34–0.84 sd. Bias
    +0.2 sd in decile 1 and −0.4 to −0.7 sd in deciles 8–10: the formula compresses the range.
  - E7 terms refit on the top 3 deciles: R² 0.30, −0.18, 0.35, with RMSE 1.4–1.7 sd in decile 1.
  - Free 8-term refits within a decile: held-out R² −0.31 to +0.33, and the S units selected change from decile to decile.
    No stable within-decile formula exists at this term budget, so Part 1 does not separate "one job" from "two jobs" for z.

### Part 3: negative poles
| neuron (window) | z < 0 | most negative 1 % (g q W Z t, %) | most negative 10 % | median mass: most-neg 10 % / z > 0 | AUC(−z) within z < 0 | S fit of the off side (R², leading S) |
|---|---|---|---|---|---|---|
| z0 (≈ 75 GeV) | 43.7 % | 6.6 3.9 0 0 89.5 | 10.4 6.0 0.1 0.1 83.4 | 167 / 74 GeV | t 0.88, Z 0.10 | 0.886; S50 (ρ −0.81), S29 (−0.79), S30 (+0.77) |
| z7 (94–98 GeV) | 70.8 % | 18.4 10.5 0 0 71.1 | 14.8 53.2 3.5 3.8 24.7 | 36 / 90 GeV | q 0.78, W 0.21 | 0.695; S50 (+0.41), S17 (−0.36), S29 (+0.34) |
| z10 (135–148 GeV) | 25.2 % | 22.4 47.4 7.9 10.5 11.8 | 8.5 66.4 11.7 10.8 2.6 | 31 / 90 GeV | q 0.69, g 0.32 | 0.774; S19 (ρ −0.63), S51 (−0.55) |
- Mean z per jet-mass decile (edges 8, 36, 54, 74, 80, 83, 90, 93, 118, 163, 342 GeV):
  - z0: 0.6, 0.6, 0.9, 1.9, 1.8, 0.3, −0.5, −2.7, −5.3, −5.9.
  - z7: −4.6, −4.1, −2.8, −1.0, −0.3, 0.8, 2.5, 0.9, −1.7, −3.6.
  - z10: −1.5, −0.1, 0.6, 0.7, 0.9, 0.9, 1.1, 1.7, 3.0, 2.1.
  - Each is one curve over mass. For z0 it falls on the heavy side; for z7 it peaks at 90–93 GeV and falls on both sides.
- ReLU bypass, α = 1:
  - z0: 19.9 % [19.0, 20.8] of calls changed, accuracy −15.2 points [−16.0, −14.4]. McNemar: 1,198 jets become wrong,
    30 become right, p < 1e−300. Main flows: t → Z 1,120, g → Z 306.
  - z7: 16.5 % [15.7, 17.3] changed, −9.6 points [−10.5, −8.8]. McNemar 890 / 151, p 5e−128. Main flow: q → W 1,019.
  - z10: 0.27 % [0.17, 0.39] changed, −0.09 points [−0.21, +0.01], McNemar 13 / 6, p 0.17.
- ReLU bypass, α = 0.5: z0 3.5 % changed, −1.5 points; z7 3.1 %, −0.6 points; z10 0.18 %, −0.03 points.
- Matched random h5 directions (20 draws), for comparison with the α = 1 bypass:
  - z0 row: 12.4 % changed (range 4.7–20.6 %), −8.1 points (range −13.6 to −2.9). z0's own negative side lowers accuracy more
    than all 20 random directions (rank p 0.048); its call-change rate is within their range (p 0.095).
  - z7 row: 11.8 % changed (5.9–21.0 %), −6.6 points (−15.8 to −2.2). z7 lies within the random range on both metrics
    (p 0.19 for calls, 0.24 for accuracy).
  - z10 row: 0.9 % changed (0–3.9 %).

## Reading (scoped to these measurements)
- S29, S1, S0, single-particle edits:
  - In deciles 2–10, the network's response per unit of formula prediction stays within 0.88–1.07 for S29 and S1 and within
    2.46–2.90 for S0 (constant ×2.6 through the context route). No lower decile of S1 or S0 differs from the top after BH.
  - S29's weakest decile responds at 0.79 [0.73, 0.85].
  - In the weak range, the free decile refits select the same quantity as the main formula (hinge sum / Σ(r − 0.2)₊ /
    ΣpT); refitting those terms alone matches the free refit's RMSE. The top-decile linear map has an offset in the weak range,
    which is a calibration of the same quantity.
  - These measurements support "weak activations are neighbours of the main pattern". The alternative of a second role (weak
    edit slope away from 1, or other quantities driving the weak range) is excluded for S1 and S0. For S29 it is excluded in
    deciles 2–10; the weakest decile shows a 21 % lower response.
- The S29 control (a particle below the fitted onset) moves S29 by 0.12 of an on-hinge move. The unit's onset is softer than the
  fitted hinge, which may also explain the weaker response in decile 1, where the edited particles are closer to the onset
  (not separated here).
- z neurons, positive side: weak versus strong range is not determined. The available formula (E7, 32 terms on S) does not
  reproduce single-S-unit interventions in any decile (per-jet correlation 0.08 to −0.54).
- z0 and z7, negative side: the pre-ReLU value continues the same mass read past the window. z0 becomes negative as mass rises
  above ≈ 90 GeV (top jets; AUC 0.88 for top). z7 is negative both below and above its 90–93 GeV peak, most strongly for light
  q/g jets (AUC 0.78 for q). It is well described by the mass-carrying S objects (S50, S29; R² 0.89 and 0.70).
  - Passing this side through the positive-side output weights changes 16–20 % of calls and lowers accuracy 9.6–15.2 points.
    The direction of the flow is set by the d7 row (t → Z for z0, q → W for z7).
  - The size is comparable to passing a random h5 direction of the same scale (z0 more harmful than all 20, z7 within the
    range). So the trained output layer does not use, and could not use as wired, a "not-this-mass" signal.
- z10's negative side (25 % of jets, light q jets near 31 GeV) has almost no effect if passed on (0.27 % of calls).
- Under the competing hypotheses:
  - "Negative side carries a structured second read that would change call outcomes": the read is structured (class mix, mass,
    R² 0.7–0.9), but it is the same mass variable on the other side of the window. Its call changes are of the size a random
    signal through the same row produces, and they lower accuracy.
  - "Superposition in the weak range": not observed for S29 (deciles 2–10), S1 or S0. For the z positive side it is not tested
    with an adequate reference.

## Issues
- The R² thresholds of the plan (drop ≤ 0.1) cannot be read inside one decile, because of the restricted range. RMSE was used
  instead; the R² values are in the JSON.
- The z part of Part 2 has no valid reference (formula derivative mismatch in the strongest deciles too). A z dose test would
  need either an input edit with a known mass change per decile (e.g. the E6 angle scale, measured Δz vs Δm in each decile) or a
  formula fitted with single-unit interventions.
- The bypass adds a float term to fixed-point logits; the d7 float recompute differs on 0.22 % of calls.
- The untrained network's S29, S1 and S0 do not move under these edits (exactly 0). The untrained control rules out an
  edit-artifact response, but it does not provide a non-zero null slope.
- Test-sample accuracy 81.62 % (N = 7,692) is 0.3 points above the reference range; fit sample 80.99 %.

## Thesis effect
refines. For A3/A11 ("one job per unit"): S1 and S0 keep one input-output relation across their full range, and S29 across 90 %
of it, with a 21 % weaker response in its weakest tenth. For A5: the negative pole of z0 and z7 is the same mass read past the
window and is discarded by the ReLU. The weak-range question for the z neurons stays open.

## Suggested TREE.md row
| E30 | E5, E5v, E6, E7 | 3 | A3, A5, A11 | weak deciles and negative poles of S29, S1, S0, z0, z7, z10: top-3-decile formula fits; one-particle radial / pT edits, slope per decile; z S-level ± 0.5 sd; ReLU bypass of z < 0 vs 20 random h5 directions | interp/e30_weak_poles_jedi.py | done | edit slope per decile, RMSE per decile, calls changed / Δacc under bypass | results/jedi/e30_weak_poles.png | S29 0.94–1.07 in d2–10, 0.79 in d1; S1 0.88–1.05; S0 2.46–2.90 (×2.6 context); z0/z7 off side = same mass read, bypass −15.2 / −9.6 points (random h5 −8.1 / −6.6); z dose test without valid reference |

## Explainer entry (for site/explain.py, Paper A tab, claims A3/A11; cross-reference from A5)

**Conclusion.** The three second-round objects tested do the same thing in their weakest jets as in their strongest: a
one-particle edit moves each by the same multiple of the formula's prediction in every tenth of its range. The only exception is
the weakest tenth of S29, which responds at about 0.8. For two mass-window neurons, the negative side of the neuron continues the
same mass measurement away from the window. The network discards it, and passing it on would cost 10–15 points of accuracy.
Whether the mass neurons' weak range follows the same formula is not settled.

**Question.** A unit can have one main job and still do something else when it fires weakly, or carry information in its negative
(pre-ReLU) side. Cammarata's checklist asks for both checks. If the weak range held a second job, describing a unit by one
formula would be incomplete.

**Background.**
- S objects are sums over particles of a per-particle function, pooled over a fixed window of pT ranks:
  - $S_{29} = \tfrac1{16}\sum_i \mathrm{ReLU}(4.44\ln p_{T,i} + 93.8\, r_i - 21.2)$, all ranks;
  - $S_1 = \tfrac1{16}\sum_{i\ge 1}\mathrm{ReLU}(0.55\ln p_{T,i} + 89.3\, r_i - 22.9)$;
  - $S_0 = \tfrac1{16}\sum_{i<14}(0.32\,p_{T,i} + 1.40)$ plus a jet-context term through $g$.
  Here $r$ is the angular distance to the jet axis. z0, z7 and z10 are the last hidden neurons before the ReLU. Their positive
  side is a jet-mass window near 75, 94–98 and 135–148 GeV.
- A decile is one tenth of the jets ordered by the unit's value.
- The edit slope is $\sum \Delta S_{\rm measured} / \sum \Delta S_{\rm predicted}$ over the jets of a decile: 1 means the network
  moves exactly as the formula predicts.

**Method.**
1. Jets: 10,000 fit jets (seed 3001) and 7,692 independent test jets (seed 3002).
2. Edits:
   - S29 and S1: move one particle radially by 0.01–0.03 rad at fixed $p_T$ and azimuth. The rank order is unchanged.
   - S0: change one of the 14 hardest particles' $p_T$ by ±2 %.
   - Predicted $\Delta S$ comes from the hinge formula, with no refit.
3. Controls:
   - the same edit on a particle the formula ignores (outside the rank window, or below the hinge onset);
   - the same edit on an untrained copy of the network.
4. For z0, z7, z10: describe the jets with the most negative $z$. Then add $\alpha\,\min(z,0)\,W_7[k,:]$ to the logits (a ReLU
   bypass) and compare with 20 random directions in the previous layer, scaled to the same fraction negative and the same sd.
5. Statistics: 95 % bootstrap intervals over jets; permutation test of each decile against the top three; Benjamini–Hochberg
   correction over 42 tests; McNemar test for paired call changes.

**Result** (figure `e30_weak_poles.png`; left: edit slope vs decile with controls; right: one Z jet with the moved 11.7 GeV
particle, predicted $\Delta S_{29}$ +0.176, measured +0.188).
- Edit slopes:
  - S29: 0.94–1.07 in deciles 2–10 and 0.79 [0.73, 0.85] in decile 1 (BH q 0.0014).
  - S1: 0.88–1.05 in every decile (no decile differs).
  - S0: 2.46–2.90 in deciles 1–9. The factor ≈ 2.6 is the jet-context route through $g$ and is constant.
  - Controls: 0–0.19 of the prediction. Untrained network: 0.
- Negative poles:
  - z0 < 0 picks top jets (90 % of the most negative 1 %, median mass 167 GeV).
  - z7 < 0 picks light jets (median 36 GeV) and top jets (71 % of the most negative 1 %).
  - The bypass changes 19.9 % (z0) and 16.5 % (z7) of calls and lowers accuracy 15.2 and 9.6 points. The random directions
    give −8.1 and −6.6 points on average, with ranges that include z7's value.

**Checks.**
- The S override with the network's own S is exact; the float d7 recompute matches the fixed-point calls on 99.8 % of jets.
- Refitting the main formula's own terms on each decile fits as well as a free search over ~200 quantities, and the free search
  picks the same quantity in every decile.
- The z dose test was not interpretable: the 32-term z formula does not reproduce single-S-unit interventions even in the
  strongest jets.

**Why it matters.** For a physicist, these sums behave like single observables over their whole range: a particle further out
in angle adds the same amount whether the jet is narrow or wide. This supports describing each of these units by one formula.
The mass neurons compute "how far from this mass window" on both sides, but the network keeps only the inside. The information
that a jet is far from 75 GeV, or far from 94 GeV, is present in those neurons but not used by the output layer.

**Glossary additions:**
- edit slope: measured over predicted change of a unit under an input edit.
- activation decile: one tenth of the jets ordered by the unit's value.
- ReLU bypass: passing a neuron's negative pre-activation through the output weights that normally see only its positive part.
- negative pole: the pre-ReLU side of a neuron below zero.
