# E21: S-level patches along iso-product and iso-sum curves of the M0 and M2 carriers (stage 4; A4/A9 open item after E15v)

Script: `interp/e21_isoproduct_jedi.py` (figure: `interp/e21_plot.py`). Results: `results/jedi/e21_isoproduct.json` (final, run 3).
Earlier runs: `results/jedi/e21_run1_cdesign.json` and `results/jedi/e21_run2.json`. Logs: `logs/e21-run1-cdesign.log`, `logs/e21-run2.log`
and `logs/e21-run3.log`. Figure: `results/jedi/e21_isoproduct.png`.

The jobs were e21-run, e21-run2 and e21-run3, each with 1 CPU / 2 GiB and exit 0. Run 3: wall time 151 s, peak memory 1338 MiB.
Runs 1 and 2 peaked at 2047 MiB while loading the sample.

## Sample and checks
- Sample: `sample(per_class=2000, seed=47)`, N = 10,000, exact `Jedi(64)` (3-feature n64).
- Accuracy 0.8058 (binomial sd 0.0040).
- `run_jetwise` with S0, S50 and S29 set to their own values: max |Δ| = 0 at h4, h5, z and the logits.
- The S → logits tail used for the patches reproduces `run_jetwise` exactly: max |Δ| = 0 on the base S and on a test patch.
- On its on-region, h4 is affine in S. Measured $\Delta h_4$ vs $K_{d4}^\top \Delta S$: slope 0.9989, correlation 0.9992 (134,845 unit-jet pairs). So a single h4 unit cannot form a product of S0 and S50; a product can only come from ReLU gating across units.
- Carriers (mean, sd, fraction zero):
  - S0: 18.39, 5.51, 0.03 %
  - S50: 2.60, 3.19, 15.5 %
  - S29: 9.75, 8.90, 0.5 %
- Affine maps from S to the true moments:
  - $M_0 = \sum p_T \approx 21.27\,S_0 + 653$ GeV (r = 0.959)
  - $M_2 = \sum p_T r^2 \approx 2.64\,S_{50} + 2.80$ GeV rad² (r = 0.962)
  - So S0 carries M0 with an offset: around the mean, $d\log S_0 / d\log M_0 \approx 2.7$.

## Procedure (final design, run 3)
These are unit interventions at S only; no input is edited. Each jet gets a signed step $\varepsilon \in \{\pm0.15, \pm0.3, \pm0.5\}$ of standardised length in $(z_{S0}, z_{S_k})$ space. Every other S unit keeps its value.

**Iso-product edit.** It moves along $m_0 m_k = \text{const}$, where $m_u = a_u S_u + b_u$:
- $m_0 \to m_0 e^{\delta}$ and $m_k \to m_k e^{-\delta}$;
- $\delta = \varepsilon / \lVert (m_u / (a_u\,\mathrm{sd}_u))_u \rVert$.

**Iso-sum edit.** It moves along $z_0 + \overline{z_k} = \text{const}$:
- $z_0 \mathrel{+}= \varepsilon/\sqrt{1+K}$ and $z_k \mathrel{-}= \varepsilon/\sqrt{1+K}$.

**Arms:**
- A: $k \in \{S50\}$, with $a = 1$, $b = 0$ (so $S_0 S_{50}$ is held fixed).
- B: $k \in \{S50, S29\}$, with $a = 1$, $b = 0$.
- C: $k \in \{S50\}$, with the affine moments above (so $M_0 M_2$ is held fixed).

**Jets left out.** A jet is dropped if any patched S < 0 or any $m_u \le 0$ in any step. That leaves:
- A: 6,355 jets (36 % dropped, mainly S50 = 0)
- B: 6,574 jets
- C: 6,351 jets

**Metric.** For each live unit U (h4: 55, h5: 32, z: 16) and for the W − Z logit margin (logit 2 − logit 3):
$$R_U = \frac{\sum_{j,\varepsilon} |\Delta U|_{\text{iso-prod}}}{\sum_{j,\varepsilon} |\Delta U|_{\text{iso-sum}}}$$
- Product rule: R = 0. Equal-weight standardised sum: R = ∞.
- 95 % CI from 300 Poisson bootstraps over jets.
- H0 (R = 1) is tested with a paired z-test on $D_j = |\Delta U|_{\text{prod}} - |\Delta U|_{\text{sum}}$, with Holm correction over all responsive units of the arm.
- "Responsive" means mean |ΔU| > 0.01 unit sd.
- Calls: % changed per step; McNemar exact test of iso-product vs iso-sum.

**Rules computed on the same patched S** (3 seeds):
- $V_{sum}$ and $V_{prod}$;
- 64 ReLU windows on each;
- 64 ReLU units on random directions in $(z_0, \overline{z_k})$ (separate carriers);
- control (b): 64 random ReLU units on the whole trained S (rand1), then 32 more on top (rand2).

**Controls:**
- (a) 20 random pairs of live S units given the same per-jet standardised displacements as arm A.
- (c) the untrained network (failed, see Issues).

**Angle test (added in run 3).** For each jet, the iso-product step has a direction $\theta = \arctan(|\Delta z_k|/|\Delta z_0|)$. Arm A's quartile edges are 9.1°, 12.5° and 27.7°. Whatever its direction, a product-of-carriers unit gives R = 0. I compared measured R per θ bin with a fixed-weight additive rule, $\Delta U = w_0\Delta z_0 + \sum_k w_k\Delta z_k$, fitted on even jets and evaluated on odd jets.

**Calibration (real input edits).** Two edits, with $c \in \{0.7, \dots, 1.4\}$:
- pT × c with angles × 1/c, which keeps $M_0 M_2$ fixed. The plan's version (angles × $c^{-1/2}$) keeps $M_2$ fixed, not $\sqrt{M_0 M_2}$, so I ran both.
- pT × c with angles × $c^{-1/2}$.

## Numbers
### Read-out units and margin, all classes (R with 95 % CI; paired p for every entry < 1e-7 unless noted)
| target | arm A (S0·S50) | arm B (S0, S50 + S29) | arm C (affine M0·M2) | random pairs, arm A steps (5 / 50 / 95 %) |
|---|---|---|---|---|
| z0 (W window) | 0.733 [0.722, 0.744] | 0.358 [0.354, 0.363] | 0.789 [0.776, 0.803] | 0.41 / 0.77 / 1.38 |
| z7 (Z window) | 0.404 [0.398, 0.410] | 0.286 [0.281, 0.292] | 0.514 [0.507, 0.520] | 0.43 / 0.98 / 1.37 |
| z14 (Z) | 0.584 [0.579, 0.590] | 0.382 [0.377, 0.388] | 0.595 [0.589, 0.602] | 0.42 / 0.96 / 1.25 |
| z10 (top) | 0.964 [0.951, 0.975] (p 3e-8) | 0.562 [0.553, 0.572] | 1.115 [1.096, 1.130] | 0.58 / 0.95 / 1.33 |
| W − Z margin | 0.443 [0.435, 0.450] | 0.279 [0.274, 0.283] | 0.461 [0.451, 0.470] | 0.38 / 0.91 / 1.32 |

Network vs random-pair draws in arm A: rank p (two-sided, 20 draws) is 0.19 for z7, 0.29 for the margin, 0.57 for z14, 0.95 for z10 and 1.0 for z0.

### By true class (arm A / arm B)
| class | N | z0 | z7 | W − Z margin |
|---|---|---|---|---|
| W | 1,591 / 1,660 | 0.868 / 0.323 | 0.304 / 0.157 | 0.446 / 0.196 |
| Z | 1,769 / 1,794 | 0.596 / 0.269 | 0.332 / 0.227 | 0.364 / 0.303 |
| top | 1,900 / 1,922 | 0.844 / 0.735 | 0.743 / 0.719 | 0.749 / 0.579 |

### Per-unit R by stage (no-clip jets; median $\log_{10}R$ [IQR]; Holm-significant R < 1 / R > 1 out of the responsive units)
| stage | arm A | arm B | arm C | random pairs, A steps: draw medians, 5 / 50 / 95 % |
|---|---|---|---|---|
| h4 | +0.08 [−0.24, +0.12]; 8 / 16 of 24 | −0.20 [−0.43, +0.03]; 23 / 10 of 34 | +0.14 [−0.27, +0.15]; 7 / 17 of 24 | −0.55 / −0.01 / +0.14 |
| h5 | −0.02 [−0.17, +0.08]; 16 / 14 of 32 | −0.22 [−0.41, −0.03]; 26 / 4 of 32 | −0.00 [−0.21, +0.13]; 16 / 16 of 32 | −0.41 / −0.02 / +0.11 |
| z | −0.09 [−0.17, +0.04]; 10 / 6 of 16 | −0.31 [−0.45, −0.07]; 12 / 4 of 16 | −0.04 [−0.12, +0.07]; 9 / 7 of 16 | −0.36 / −0.02 / +0.08 |
| W − Z margin | −0.35 | −0.56 | −0.34 | −0.42 / −0.04 / +0.12 |

Rules and the random-ReLU control on the same patched S (arm A; arms B and C are similar):

| rule | median $\log_{10}R$ | Holm-significant R < 1 / R > 1 | pooled R |
|---|---|---|---|
| $V_{prod}$ | −3.4 | – | – |
| windows on $V_{prod}$ | −3.4 | – | – |
| $V_{sum}$ and windows on it | > +6 (R ≈ ∞) | – | – |
| ReLU on separate carriers | −0.02 to −0.05 | – | – |
| rand1 on S | −0.12 to +0.05 | 41 / 21 of 62 (seed 0) | 0.84–1.08 |
| rand2 on S | −0.05 to +0.00 | 18 / 11 of 32 (seed 0) | 0.89–0.96 |

### Angle test, arm A (test jets; R by θ quartile, quartile medians ≈ 7°, 11°, 16°, 43°)
| target | measured R by bin (95 % CI ±0.01–0.04) | fixed-weight additive fit | held-out R² of the additive fit (iso-prod / iso-sum) |
|---|---|---|---|
| z0 | 0.86, 0.66, 0.60, 1.00 | 0.45, 0.30, 0.17, 0.97 | 0.11 / 0.42 |
| z7 | 0.37, 0.29, 0.47, 1.00 | 0.68, 0.52, 0.27, 0.96 | −0.23 / 0.29 |
| W − Z margin | 0.47, 0.39, 0.44, 1.03 | 0.12, 0.05, 0.22, 0.97 | −0.18 / 0.17 |

Arm B, margin: 0.16, 0.31, 0.46, 1.03 against additive 0.45, 0.48, 0.46, 0.83.

In the top quartile (θ ≈ 43°) the two edits point in nearly the same direction, so R → 1 there by construction.

### Calls (% changed, mean over the 6 steps; McNemar of iso-prod vs iso-sum)
| arm | iso-product | iso-sum | McNemar |
|---|---|---|---|
| A | 5.5 % | 6.9 % | at ε = ±0.5: 132 vs 290 and 195 vs 356 discordant; p 1e-14, 7e-12 |
| B | 7.2 % | 23.9 % | at ε = +0.5: 75 vs 2,437 discordant; p < 1e-300 |
| C | 5.8 % | 6.9 % | – |
| random pairs (arm A steps), 5 / 50 / 95 % | 0.8 / 2.2 / 4.4 % | 1.2 / 2.4 / 7.4 % | – |

By class:
- Arm A on W jets: 3.5 % vs 6.7 %. On top jets: 5.9 % vs 5.3 %.
- Arm B on W jets: 6.4 % vs 39.4 %. On Z jets: 6.5 % vs 33.1 %.

### Calibration (real input edits; medians over jets with S0, S50, S29 > 0)
pT × 1.1 with angles / 1.1 keeps $M_0 M_2$ fixed. Its effects:
- S0 × 1.33 (ΔS0 = 0.99 sd) and S50 × 0.88 (0.09 sd); S0·S50 × 1.17.
- Other S units move by 0.16 sd on average (max 0.38).
- 14.6 % of calls change.

At c = 0.9: S0 × 0.68 and S50 × 1.14; 32.7 % of calls change.

pT × 1.1 with angles × $1.1^{-1/2}$ ($M_2$ fixed): S0 × 1.33, S50 × 0.99, 18.2 % of calls change, W → Z 18 %. At c = 1.25: W → Z 84 %.

The S-level steps here (0.15–0.5 sd along the iso-curves) are therefore 0.15–0.5 of the S0 displacement of a real ±10 % iso-product input edit. That real edit also moves the other S units by 0.16 sd on average, which the S-patch does not.

### Run 1 (plan's c-parametrisation; superseded)
The edits were S0 × c with S50 / c or S50 + (1 − c) S0 sd50/sd0, for c ∈ {0.7, …, 1.4}. The iso-sum edit made S50 < 0 for 81 % of jets: 1,938 jets remained, 6 of them W jets.

On those jets, R for z0 / z7 / margin was 0.95 / 0.92 / 0.95. On these jets the two edits move S50 by similar amounts, so R is close to 1 by construction. This design was replaced by steps matched in standardised length.

## Reading (scoped to these patches)
**H0 (R = 1).** It is rejected for most read-out targets in every arm (R 0.28–0.79), but not for z10 in arm A.

**Arm A against its geometric control.** With S0 and S50 patched, the read-out R values lie inside the 5–95 % range of random S-unit pairs given the same per-jet displacements. The exception is z7 (0.404 against a 5th percentile of 0.431; rank p = 0.19). So arm A's R < 1 at z0, z14 and the margin is not specific to the M0/M2 carriers.

**Angle test.** R ≈ 0 at every step direction is not observed: margin R = 0.39–0.47 in the three lower θ bins and → 1 in the top bin. This excludes an exact function of $S_0 S_{50}$ (or of the affine $M_0 M_2$, arm C) at the read-out.

**Fixed-weight additive rule.** It does not describe the responses either:
- held-out R² ≤ 0.11 for iso-product steps;
- its per-bin R differs from the measured R in both directions (z7 measured below it; z0 and margin measured above it).

**Stage transition.** None is seen between S and h5:
- In arms A and C, the per-unit $\log_{10}R$ medians at h4, h5 and z are −0.09 to +0.14. These lie inside the random-pair draw range and inside the rand1/rand2-on-S range.
- rand1 on S also gives R significantly < 1 for 41 of 62 units. This is the plan's alternative outcome: random ReLU layers show R < 1 too, so this test does not locate a product stage.

**Arm B (S29 added as a second M2 carrier).** It gives lower R everywhere:
- per-unit medians: h4 −0.20, h5 −0.22, z −0.31;
- margin 0.28, z7 0.29;
- iso-sum changes 23.9 % of calls against 7.2 % for iso-product (W jets: 39 % vs 6 %).

There is no random-pair control with B's three-unit geometry, so B's lower R is not separated from a larger weight of S29 in the read-outs.

**Thesis predictions:**
| prediction | result |
|---|---|
| R < 0.3 at z0 / z7 / margin | Met only in arm B for z7 (0.286) and the margin (0.279); z0 is 0.358 in B and 0.40–0.79 in A and C. |
| Transition between S and h5 | Not observed. |
| Iso-product changes < 3 % of calls | Not met (5.5–7.2 %). |
| Iso-sum changes > 10 % of calls | Met in arm B only (23.9 %). |

## Issues
- **Untrained control (c) gave no data.** In the redrawn network, S0 or S58 is 0 in every jet, so all 10,000 jets were dropped. Both debug attempts went to the design fix (run 2) and the angle test (run 3), so I did not rerun it. A fix would pick an untrained unit pair that is non-zero on most jets.
- **Random-pair control covers arm A only.** Arms B and C have no pair control with their own displacements.
- **Change from the plan's c-design.** As specified, the iso-sum edit leaves S ≥ 0 for only 19 % of jets. The final design uses steps of matched standardised length. The calibration uses angles × 1/c for the $M_0 M_2$-preserving real edit, because the plan's angles × $c^{-1/2}$ keeps $M_2$ fixed instead.
- **No-clip selection.** It removes jets with S50 = 0 (15.5 %) and is not class-balanced: arm A keeps 1,591 W, 1,769 Z and 1,900 top jets.

## Explainer entry (for site/explain.py, Paper A, A4/A9)
- **Question.** Does any stage of JEDI-linear combine the M0 carrier (S0) and the M2 carriers (S50, S29) as a product $M_0 M_2$, and if so at which stage (h4, h5, z)?
- **Background.**
  - E15 measured an interaction with a read-out probe. E15v showed that the same probe gives similar values for ReLU windows on a sum and for random ReLU layers, so "multiplied in h4/h5" was withdrawn.
  - A read-out-free test moves S0 and S50 directly. Moving along a curve on which the product is constant gives no change for a unit that depends only on the product. Moving along a line on which a standardised sum is constant gives no change for a unit that depends only on that sum.
- **Method.**
  - For 10,000 jets (seed 47), set S0 and S50 (and S29) to values on the curve $m_0 m_k = $ const, and separately on the line $z_0 + \overline{z_k} = $ const, with steps of 0.15–0.5 sd. Record $R = \sum|\Delta U|_{\text{prod}} / \sum|\Delta U|_{\text{sum}}$ for every h4, h5 and z unit and for the W − Z logit margin.
  - Controls: the same displacements applied to random pairs of S units; random ReLU layers on the trained S.
  - Bin jets by the direction of their iso-product step: a product unit gives R = 0 at every direction.
- **Result.** Figure `e21_isoproduct.png`.
  - With S0 and S50, R at the W − Z margin is 0.44 [0.44, 0.45] and at z7 0.40. Random S-unit pairs given the same displacements give 0.38–1.32 and 0.43–1.37 (5–95 %).
  - R rises from 0.39–0.47 to 1.03 as the step direction approaches the diagonal.
  - Per-unit R at h4, h5 and z matches random ReLU layers on S.
  - Adding S29 as a second M2 carrier lowers the margin R to 0.28. The iso-sum edit then changes 24 % of calls and the iso-product edit 7 %.
- **Checks.**
  - Exact patch identity (max |Δ| = 0).
  - Accuracy 0.8058.
  - h4 is affine in S on its on-region (slope 0.999).
  - A real pT × 1.1, angles / 1.1 edit moves S0 by 0.99 sd and S50 by 0.09 sd.
  - The untrained control is missing.
- **Why it matters.** Step 5 of the narrative ("the dense layers multiply them") and the mass example of A9 still have no supporting test:
  - The read-outs respond less along the iso-product curve than along the iso-sum line.
  - For S0 with S50, random S-unit pairs do the same.
  - Neither an exact product nor a fixed-weight sum of the carriers describes the responses.
  - The S29 arm is the open lead; it needs its own pair control.
