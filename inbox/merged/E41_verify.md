# E41 verification: per-stage read-outs of the true mass vs the axis mass, with a mass-preserving control

Script `interp/e41_verify.py` (new; imports the E41 functions unchanged). Cluster job `e41-verify2` at 1 CPU / 2 GiB: wall 932 s, peak 2,047 MiB, exit 0. The name `e41-verify` was already taken by the worker's replication job, which is complete; it was left alone so its log is not overwritten. Smoke run: `e41-vsmoke`. Results: `results/jedi/e41_verify.json` (smoke: `e41_verify_smoke.json`).

## Conclusion
On new jets, the network's decision barely responds when the true jet mass is lowered by 10 % with the axis-based mass held fixed (5 % of calls change, split evenly between W and Z), and the E41 read-out pattern repeats. A new control moves the same particles by the same amounts but keeps the true mass, and it reproduces most of what the network does under the mass-lowering edit. The learned DAS direction (a direction in the activations found by distributed alignment search, defined below) undoes the control's call changes about as well as the mass edit's (0.52 vs 0.65 of the changed calls at S). So that direction tracks how the hardest particles are arranged around the axis, not the mass. Two statements in E41 need correcting: the trained read-outs are not "worse than untrained networks" by a measure that ignores the subset's spread, and the context vector g does move with the edit, mostly for reasons other than the mass.

## Definitions
- $m$: invariant mass of the summed massless constituent four-vectors. $M_0=\sum p_T$, $\vec M_1=\sum p_T(\Delta\eta,\Delta\phi)$, $M_2=\sum p_T r^2$, all about the stored axis. $m_A=\sqrt{M_0M_2}$ is the axis-moment mass, and $\Delta=\log m-\log m_A$.
- turn10 (E41 / E27w arm B): the 14 hardest particles turned in azimuth about the axis toward $\vec M_1$ until $m$ falls 10 %. Each particle keeps its $p_T$ and $r$, so $M_0$, $M_2$ and $m_A$ stay fixed, while $|\vec M_1|$ grows (median $|\vec M_1|/M_0$ goes from 0.002 to 0.038).
- ctrl10 (new here; E27w's M1-held control): the same 14 particles turned by the same azimuth amounts $|\delta\theta_i|$ with random signs. A minimum-norm azimuth correction on the same particles then restores the vector $\vec M_1$, and a common rescale keeps the displacement $D=\sqrt{\sum_i|\delta\vec x_i|^2}$ equal to turn10's. $p_T$, $r$, $M_0$, $M_2$, $m_A$ and $\vec M_1$ are held fixed, so $m$ is held too: median $\Delta\log m=5\times10^{-6}$, maximum 0.007, maximum $|\Delta\log m_A|=3\times10^{-8}$. It is the right comparison for turn10 because it moves the same particles by the same amount in the same coordinate, but leaves the mass unchanged.
- dil10: all angles scaled per jet so that $m$ matches turn10's $m$. This is a real 10 % mass change, and $m_A$ moves with it.
- $R_m$: mean change of a ridge $\log m$ read-out under an edit ÷ its mean change under dil10 on the same jets. The read-out is fitted on natural, shifted and dilated jets. 1 means the read-out moves as for a real mass change, 0 means it does not move.
- slope: mean change of the pool $\Delta$ read-out under turn10 ÷ the true mean change of $\Delta$.
- DAS (distributed alignment search): learn a unit direction $v$ at stage S, h4 or h5 such that copying the unedited jet's coordinate along $v$ into the edited jet's activation restores the unedited jet's call. IIA (interchange-intervention accuracy) on informative pairs is the share of pairs whose call the edit changed and the swap restored. $v_{turn}$ is trained on turn10 pairs, and $v_{ctrl}$ on ctrl10 pairs.
- Band: the read-out statistic on untrained (u) or row-permuted (p) copies of the network.

## Procedure
- **Samples.** Fit: `sample(500, 4121)`, 2,500 jets. Test: `sample(500, 4122)` minus 23 overlapping jets, leaving 2,477. Both seeds are new. Test accuracy on natural jets is 0.818.
- **Pool.** The E41 `build_pool` without changes: 21,271 fit rows and 21,064 test rows. turn10 and ctrl10 were both kept for 1,549 test jets and 1,578 fit jets. turn10 was recomputed, and it matches the pool rows exactly (max diff 0).
- **Read-outs.** For each of the 10 stages: ridge and MLP (64 hidden units, 12,000 rows, 80 iterations) of $\Delta$, fitted on all fit-pool rows as in the E41 "pool" fit. They are scored on the test turned and shifted subsets and also applied to the ctrl10 rows. The same was run on a mini band of 4 control networks (untrained u0 and u1, permuted p0 and p1).
- **Interchange.** At S, h4 and h5 with the exact fixed-point tail (`x10_das_jedi.JNet`; tail check max |Δlogit| = 0). DAS used Adam for 400 steps. Each of $v_{turn}$ and $v_{ctrl}$ was scored on both test pair sets. The null is 100 random unit directions.
- **Statistics.** 95 % CIs from 1,000 Poisson jet-bootstrap resamples. McNemar on paired call-change indicators.

## Results

**1. Calls (1,549 matched test jets).**

| edit | calls changed [95 % CI] | W→Z | Z→W |
|---|---|---|---|
| turn10 ($m$ −10 %, $m_A$ fixed) | 5.1 % [4.1, 6.3] | 3.0 % | 2.8 % |
| ctrl10 ($m$, $m_A$, $\vec M_1$ fixed) | 2.8 % [2.1, 3.7] | 2.3 % | 1.6 % |
| dil10 (real 10 % mass change) | 34.0 % [31.6, 36.5] | 0 | 91.9 % |

- turn10 vs ctrl10: 49 jets change only under turn10 and 14 only under ctrl10; McNemar p = 1.1e-5.
- turn10 vs dil10: McNemar p = 5e-101.
- H0 (turn10 changes calls at the ctrl10 rate) is rejected. The 2.3-point excess is not directed toward W, whereas a real mass decrease sends 92 % of Z calls to W. E41's 4.5 % with symmetric flow is reproduced.

**2. Pool $\Delta$ read-outs on turned test jets (trained network).**

| stage | MLP $R^2$ | MLP corr | ridge corr | slope (MLP) | ctrl10 / turn10 read-out change |
|---|---|---|---|---|---|
| a | −0.11 | 0.73 | 0.09 | 1.09 | −0.08 |
| P | −1.95 | 0.04 | 0.02 | 0.16 | 0.46 |
| g | −1.93 | 0.01 | 0.05 | 0.16 | 0.54 |
| s | −1.56 | 0.11 | 0.13 | −0.02 | 1.09 |
| u | −2.02 | 0.15 | 0.13 | 0.11 | 0.40 |
| S | −1.94 [−2.27, −1.67] | 0.14 | 0.21 | 0.11 [0.10, 0.13] | 0.39 [0.32, 0.47] |
| h4 | −1.63 [−1.92, −1.37] | 0.13 | 0.11 | 0.10 [0.09, 0.12] | 0.48 [0.39, 0.56] |
| h5 | −2.05 | 0.12 | 0.10 | 0.08 | 0.44 |
| z | −2.14 | 0.11 | 0.07 | 0.06 | 0.43 |
| logits | −1.92 | 0.02 | −0.08 | 0.01 | 0.27 |

- Shifted test jets: MLP $R^2$ 0.65 at S and 0.71 at h4 (E41: 0.68 and 0.74).
- Mini band, turned subset: MLP $R^2$ −0.33 to −1.63, so the trained values are lower. Correlation, which does not depend on the subset's spread: band maximum ridge corr at S −0.005 (trained 0.21), at u 0.04 (trained 0.13), at h4 0.04 (trained 0.11). Band maximum MLP corr at h4 / h5 / z: 0.04 / −0.004 / 0.009 (trained 0.13 / 0.12 / 0.11).
- E41's own band, re-read from `e41_ctrl_*.json`: trained ridge corr on turned jets at S is 0.22 vs a band q95 of 0.07, and at u 0.16 vs 0.06.
- So the trained read-outs follow the turn's $\Delta$ more than untrained or permuted ones do. They pick up about 0.1 of the true change (slope), and 40–50 % of that response also appears under the mass-preserving ctrl10.
- The negative $R^2$ comes from a constant offset. The pool fit is dominated by shifted rows, so its prediction on turned rows sits far from their mean. This offset makes $R^2$ fall below the band's even where the read-out carries more of the turn's $\Delta$ than the band's does.

**3. $R_m$ of the $\log m$ read-out (fit nat+shift+dil), turn10 vs ctrl10.**

| stage | g | u | S | h4 | h5 | logits |
|---|---|---|---|---|---|---|
| turn10 | 0.875 [0.84, 0.91] | 0.11 | 0.19 [0.17, 0.22] | 0.058 | 0.038 | 0.047 |
| ctrl10 | 0.733 [0.70, 0.77] | 0.08 | 0.16 [0.15, 0.18] | 0.016 | 0.004 | −0.022 |
| turn − ctrl | 0.14 [0.12, 0.17] | 0.03 [0.02, 0.04] | 0.03 [0.005, 0.05] | 0.04 [0.03, 0.06] | 0.03 [0.02, 0.05] | 0.07 [0.05, 0.09] |

- E41 reported 0.95 at g. The fresh sample gives 0.875.
- 84 % of g's response also appears under the mass-preserving edit. g is the context vector built from the first-layer half-plane objects, and its read-out responds to how the hardest particles are arranged in azimuth.
- The part specific to the mass is 0.03–0.14 of a real change.

**4. Interchange (1,549 test pairs; 79 informative turn pairs and 44 informative ctrl pairs). IIA on informative pairs [95 % CI]; random median in parentheses.**

| stage | $v_{turn}$ on turn pairs | $v_{turn}$ on ctrl pairs | $v_{ctrl}$ on ctrl pairs | $v_{ctrl}$ on turn pairs | cos($v_{turn}$, $v_{ctrl}$) |
|---|---|---|---|---|---|
| S | 0.65 [0.54, 0.75] (0.20) | 0.52 [0.37, 0.68] (0.23) | 0.32 [0.18, 0.46] (0.23) | 0.39 (0.20) | 0.04 |
| h4 | 0.44 [0.34, 0.55] (0.09) | 0.41 [0.27, 0.57] (0.09) | 0.18 [0.08, 0.31] (0.09) | 0.18 (0.09) | 0.35 |
| h5 | 0.37 [0.26, 0.48] (0.10) | 0.34 [0.20, 0.49] (0.11) | 0.25 [0.12, 0.39] (0.11) | 0.30 (0.10) | 0.01 |

- $v_{turn}$ reproduces E41's numbers (0.60 / 0.44 / 0.34). The flow it restores is symmetric: W→Z and Z→W at 4.9 / 4.9 % at S.
- $v_{turn}$ restores the calls changed by the mass-preserving ctrl10 at nearly the same rate: excess over random 0.30 / 0.32 / 0.23 on ctrl pairs, vs 0.44 / 0.35 / 0.27 on turn pairs, with overlapping CIs. H1 ("$v_{turn}$ is specific to the true-mass change") predicts restoration near random on ctrl pairs. That prediction is not met at any of the three stages (rank p 0.01 vs random on ctrl pairs).
- The planned contrast (excess of $v_{turn}$ on turn pairs minus excess of $v_{ctrl}$ on ctrl pairs) is 0.35 [0.16, 0.53] at S, 0.26 [0.09, 0.42] at h4 and 0.13 [−0.05, 0.30] at h5. This contrast is confounded: $v_{ctrl}$ had 47 informative training pairs against 73 for $v_{turn}$, and $v_{ctrl}$ scores no better on its own pairs than on turn pairs. The cross-application in the previous bullet is the cleaner comparison.

## Code review of `e41_true_mass_layers_jedi.py` / `e41_plot.py`
- **Checked with no problem found.**
  - Edits touch only real particles, with padding kept by `to_x` and the `* real` masks.
  - pT, r, ranks and slots are unchanged by the turn; $|\Delta\log m_A|<3\times10^{-8}$.
  - Stage means at a, s and u are over real particles only.
  - Fit and test jets are disjoint (hash de-duplication), and all versions of a jet stay in one sample.
  - Ridge alpha folds are grouped by jet. The bootstrap resamples whole jets.
  - The exact tail is checked before swaps (`jn.check` sets the shape).
  - The DAS training labels are the source's calls.
- **Control defect.** `permuted_jedi` permutes the rows of each kernel. The first-layer kernel has 3 input rows (pT, Δη, Δφ), so only 6 permutations exist.
  - In `e41_ctrl_p.json`, copy p6 draws the identity permutation. Its stage a and P are exactly the trained network's: ridge pool $R^2(\Delta)$ 0.11966 and $R^2(\log m)$ 0.80697, identical to trained.
  - p3 and p8 only swap Δη and Δφ. p0, p1, p2 and p4 share one permutation, and so do p5 and p7.
  - The band at a and P therefore contains the trained first layer itself, and the trained value at a on turned jets is in the band partly because the band contains it.
  - Re-tested against the 10 untrained redraws only: a pool 0.867 is inside (0.859–0.916), a turned 0.309 is inside (0.195–0.537), and P is inside. The E41 statements about a and P survive. At g, trained pool 0.148 is above all 10 untrained (max 0.135); E41's Holm-corrected test there was not significant.
- **Metric choice.** "Below the band" on turned jets is stated from $R^2$, which is dominated by the constant offset described in Result 2. By the band's 5th percentile, the trained MLP values are below the band at u, h4, h5, z and logits, and inside it at P, g, s and S (E41 summary). No test for "below" was run.
- **Budget.** The main-arm stage-a turned $R^2$ (0.605, MLP 25k rows) exceeds the band maximum (0.558, 15k rows). Only the matched-budget value (0.309) sits inside.
- **Figure** (`e41_true_mass.png`). The left panels show the edits on a real W jet correctly. In the right panel, `fill_between(step='mid')` on x positions offset by ±0.13 makes stage a's red band (−1.0 to 0.51) spread toward P, so P appears to have a band reaching 0.5. All trained turned-jet points at P..logits are clipped at −1.2. The panel shows only $R^2$, which hides that trained correlations on turned jets exceed the band's.

## Corrected reading
- **Readable Δ.** Above untrained and permuted networks, a quantity correlated with $\Delta$ is readable from s through the logits, mainly where $\Delta$ moves through an axis translation.
- **Turned jets, read-outs.** Where only the true mass moves at fixed $(p_T, r)$ (hardest-14 turn), the read-outs follow about 0.1 of the true change. They do follow it more than untrained or permuted networks do (correlation 0.11–0.21 at u..h5 vs band ≤ 0.07). 40–50 % of that response also appears under a mass-preserving rearrangement of the same particles, which leaves 0.03–0.06 of a real mass change at S..h5 that depends on the mass.
- **Turned jets, decision.** The turn changes 5 % of calls, more than the mass-preserving control (2.8 %, McNemar p 1e-5). The changes are symmetric between W and Z, whereas a real 10 % mass change gives 34 %, almost all Z→W.
- **Context vector g.** The $\log m$ read-out at g moves 0.88–0.95 of a real change under the turn and 0.73 under the control, so g responds mainly to the azimuthal arrangement of the hardest particles.
- **DAS direction.** The 1-dim DAS direction restores mass-preserving rearrangement call changes at nearly the same rate as the turn's. It encodes the arrangement of the hardest particles about the axis, not the true mass. Alt 2 (true mass computed and used weakly) is not supported by the interchange.
- **Statements to drop.** Remove "worse than untrained networks" (an $R^2$ offset effect) and "no stage after the first layer tracks the change" (g does, mostly for non-mass reasons).
