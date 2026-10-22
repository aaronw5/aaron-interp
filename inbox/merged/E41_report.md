# E41: does any JEDI-linear stage carry the true invariant mass separately from the axis-moment mass?

Node E41 (stage 3; claims A4, A4o, A9; user question, round 20). Script `interp/e41_true_mass_layers_jedi.py` (arms main / ctrl / swap / combine), figure `interp/e41_plot.py`.
Results: `results/jedi/e41_main.json`, `e41_ctrl_u.json`, `e41_ctrl_p.json`, `e41_ctrl_rt.json`, `e41_swap.json`, `e41_summary.json`, `e41_example.npz`, figure `e41_true_mass.png`. Replication on seeds 4111/4112: `e41_main_s4111.json`, `e41_swap_s4111.json`.

## Conclusion
JEDI-linear's layers carry a quantity that separates the true jet mass from the axis-based mass $m_A=\sqrt{M_0M_2}$ only when every particle is translated off the stored axis. When the true mass is lowered by up to 10 % with $m_A$ held fixed (the hardest particles turned about the axis), no read-out from any stage after the first layer tracks the change: it does worse than a constant guess, and worse than the same read-out on untrained networks. A one-dimensional direction found by distributed alignment search (DAS: a learned direction whose value is copied from one jet's activations into another's) does undo some of the turn's few call changes, but those changes go to W and to Z about equally, as they would if the turn changed something other than the mass. The network shows no sign of building the true invariant mass; what it encodes beyond $m_A$ is a response to particles displaced from the axis, and whether that DAS direction encodes anything mass-specific has not been tested against a mass-preserving control.

## Definitions (also proposed for `explain.GLOSSARY`)
- $m$: invariant mass of the summed constituent four-vectors (massless constituents, exact; GeV).
- $M_0=\sum_i p_{T,i}$, $\vec M_1=\sum_i p_{T,i}(\Delta\eta_i,\Delta\phi_i)$, $M_2=\sum_i p_{T,i} r_i^2$, all about the stored jet axis; $r_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$.
- $m_A=\sqrt{M_0M_2}$ (axis-moment mass). For small angles $m^2\approx M_0M_2-|\vec M_1|^2$, so $m=m_A$ up to the centroid term $|\vec M_1|^2/(M_0M_2)$. Natural jets (fit sample, 5,000): median $m/m_A$ = 0.9997, 1st percentile 0.994; correlation of $\log m$ with $\log m_A$ 0.999994.
- $\Delta=\log m-\log m_A$: the part of the true mass not contained in $m_A$ (sd 0.0019 on natural jets).
- Held-out $R^2$: $1-\mathrm{SSE}/\mathrm{SST}$ of a read-out on jets not used for fitting. It is negative when the read-out does worse than predicting the test mean.
- Band: the 5–95 % range of the same statistic over 20 control networks (10 untrained redraws and 10 row-permuted copies).
- DAS (distributed alignment search): learn a unit direction $v$ in a stage's activations such that replacing the base jet's coordinate along $v$ by the source jet's coordinate, $h'=h_b+((h_s-h_b)\cdot v)v$, makes the network output the counterfactual call. IIA (interchange-intervention accuracy) is the share of pairs where it does.
- Informative pairs: pairs whose counterfactual call differs from the base call. IIA on these is not inflated by pairs whose call never changes.
- C2ST (classifier two-sample test): out-of-fold AUC of a classifier separating edited from natural jets. 0.5 means the two sets cannot be told apart.
- Steering R: change of the W–Z logit margin when the activation is moved along a read-out's gradient by the amount that shifts the read-out by $\log 0.9$, divided by the margin change from a real 10 % mass reduction (all angles × 0.9-matched) on the same jets.
- Holm: step-down family-wise correction over the 10 stages. McNemar: exact binomial test on paired call-change indicators.

## Procedure
**Network.** Official JEDI-linear 3-feature n64, exact fixed-point forward pass (`interp.jedi_core.Jedi`). Accuracy on natural jets: 0.803 on the fit sample, 0.808 on the test sample (expected 0.806–0.813). The fit sample is 0.3 points under the range; the replication samples give 0.809 / 0.814.

**Samples.** Fit = `sample(per_class=1000, seed=4101)`, 5,000 jets. Test = `sample(1000, 4102)` minus 93 jets also in fit, leaving 4,907 jets. All five classes.

**Edited pool.** Each jet contributes the versions below where the edit is solvable: 42,480 fit rows and 41,737 test rows. Changes are test-sample medians.

| version | what is changed | held fixed | $\Delta\log m$ | $\Delta\log m_A$ | kept (test) |
|---|---|---|---|---|---|
| nat | nothing | – | 0 | 0 | 4,907 |
| shift3 / shift6 | every real particle translated by 0.03 / 0.06 rad in one of ±Δη, ±Δφ (random per jet) | every pairwise angle; $m$ exact | 0 | +0.065 / +0.221 | 4,907 / 4,907 |
| turn_hard14 2/5/10 | the 14 hardest particles turned in azimuth about the axis toward $\vec M_1$ (E27w arm B), so $m$ falls 2 / 5 / 10 % | each particle's $p_T$ and $r$, hence $M_0$, $M_2$, $m_A$, ranks, slots | −0.020 / −0.051 / −0.105 | 0 (< 1e-9) | 4,598 / 3,916 / 3,157 |
| turn_soft 2/5/10 | the softer half turned the same way | same | −0.020 / −0.051 / −0.105 | 0 | 621 / 64 / 4 |
| turnup | the 14 hardest turned away from $\vec M_1$ to maximise $m$ | $p_T$, $r$ | +0.00016 (90th pct +0.0008) | 0 | 2,985 |
| dil 2/5/10 | all angles × a, with a matched per jet to the turn_hard14 mass | $p_T$; $\Delta$ fixed | −0.020 / −0.051 / −0.105 | the same | 4,598 / 3,916 / 3,157 |

The soft-half turn rarely reaches the target, and the opposite turn can raise $m$ only by the small natural centroid term, so the turn family consists almost entirely of hardest-14 turns. Figure, left panels: a real W jet ($m$ 82.1 GeV). The turn takes $m$ to 73.9 GeV with $m_A$ at 82.2 GeV. The shift leaves $m$ at 82.1 GeV and takes $m_A$ to 105.9 GeV.

**C2ST** (HistGradientBoosting, 5-fold; natural half vs edited versions of the other half; fit sample). AUC on axis-free / axis-relative features:

| family | shift3 | shift6 | turn_hard14 5 | turn_hard14 10 | turnup | dil5 | dil10 | natural vs natural |
|---|---|---|---|---|---|---|---|---|
| axis-free | 0.49 | 0.48 | 0.75 | 0.83 | 0.58 | 0.77 | 0.84 | 0.48 |
| axis-relative | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | 0.78 | 0.49 |

Families with axis-free AUC > 0.6 (flagged): turn_hard14 5 and 10, dil5, dil10. A real mass change of the same size (dil) is separated by the same amount (0.77–0.84), so on axis-free observables the turned jets differ from natural jets as much as a real mass change of the same size does. Shifted jets cannot be told from real ones without the axis (0.48–0.49), as in E31.

**Read-outs.** For each stage and each target ($\log m$, $\log m_A$, $\Delta$) two read-outs were fitted: ridge regression (alpha by 3-fold CV grouped by jet) and an MLP (one hidden layer of 64 ReLU units, early stopping). The stages are a (mean and $p_T$-weighted mean of the per-particle first-layer outputs, 128 dims), P, g, s and u (means over real particles), S, h4, h5, z and logits. Fits:
- pool: fit on all fit-sample rows, scored on held-out test rows, overall and on the turn and shift subsets.
- turnT: fit on nat + shift + dil, scored on the turn rows.
- revT: fit on the turn rows, scored on nat + shift + dil.

Raw-input baselines use the same read-outs on the 64×3 input and on $(\log M_0, \log|\vec M_1|, \log M_2)$.

**Controls.** 10 untrained redraws (`Jedi(untrained=True, seed=k)`) and 10 copies with the rows of every kernel permuted (opt-in helper `permuted_jedi`, one random permutation per weight matrix, biases and batch-norm kept) form the band. The 7 non-degenerate networks of E39 EDIT_NETS (official n64 / n32 / perminv, hgq150 s0/s2, float s0/s1) give the retrained comparison. In the band tests, "trained" is the official_n64 entry of the control run, which uses the same MLP budget (15,000 rows, 80 iterations) as the band networks. The main arm used 25,000 rows and 120 iterations, and its values are 0.01–0.07 higher.

**Statistics.** 95 % CIs from 1,000 Poisson bootstrap resamples over test jets, with all versions of a jet resampled together. H0: trained $R^2(\Delta)\le$ the band's 95th percentile; H1: above it. $p_z$ is a one-sided normal test of trained vs band mean, with variance $\mathrm{sd}_{band}^2+\mathrm{se}_{boot}^2$, Holm-corrected over the 10 stages. $p_{rank}=(1+\#\{\text{band}\ge\text{trained}\})/21$ has a floor of 0.048 and cannot pass Holm alone.

**Interchange.** Run at S, h4 and h5, the only stages where the exact fixed-point tail with per-unit clipping and a straight-through gradient exists (`x10_das_jedi.JNet`). These three are also the swap-eligible stages from the read-out tests. Turn pairs: base = turned jet (turn_hard14 10, plus 4 soft pairs), source = its unturned original, which has the same $m_A$ and a 10 % larger $m$; counterfactual = the original's call. The 1-dim DAS direction $v_\Delta$ was trained on fit-sample pairs (Adam, 400 steps). The ridge $\Delta$ probe direction was also swapped. Positive control: $v_{m_A}$ from DAS on (original, shift6) pairs toward the shifted call, plus the $\log m_A$ probe direction. Null: 200 random unit directions per stage, giving a rank p and McNemar of $v$ vs each draw (median p reported). Test pairs: 3,161 turn pairs, of which 143 are informative (the turn changed the call), and 4,907 shift pairs, of which 2,933 are informative.

## Results

### 1. Read-outs of $\Delta$ per stage (held-out $R^2$, trained vs band)
MLP read-out; trained value [95 % CI]; band 95th percentile in parentheses.

| stage | all pool rows | shifted jets only | turned jets only |
|---|---|---|---|
| a | 0.867 [0.856, 0.877] (0.915) | 0.833 (0.893) | 0.309 [0.241, 0.361] (0.513) |
| P | 0.160 (0.198) | −0.06 (0.01) | −1.86 (−1.25) |
| g | 0.148 (0.120) | −0.03 (−0.10) | −2.47 (−0.90) |
| s | 0.495 (0.245) | 0.48 (0.08) | −1.46 (−0.52) |
| u | 0.597 (0.144) | 0.64 (−0.11) | −1.86 (−0.77) |
| S | 0.652 [0.633, 0.670] (0.115) | 0.68 (−0.13) | −1.48 [−1.63, −1.33] (−0.41) |
| h4 | 0.700 [0.680, 0.718] (0.095) | 0.74 (−0.20) | −1.35 [−1.52, −1.20] (−0.34) |
| h5 | 0.629 (0.089) | 0.66 (−0.21) | −1.55 (−0.34) |
| z | 0.529 (0.071) | 0.53 (−0.27) | −1.62 (−0.34) |
| logits | 0.232 (0.053) | 0.10 (−0.29) | −1.84 (−0.35) |
| input $(M_0,\lvert M_1\rvert,M_2)$ | 0.989 | – | 0.977 |
| input 64×3 (flattened) | −0.02 | – | – |

- **Pool.** Above the band (Holm $p_z\le 3\times10^{-5}$) at s, u, S, h4, h5, z and logits. Inside the band at a, P and g (Holm $p_z$ 0.78). Ridge read-outs give the same stage pattern at lower values: trained 0.04–0.35, with h4 0.352 [0.336, 0.366] vs a band 95th percentile of 0.064.
- **Shifted jets.** Above the band at s through logits (Holm $p_z\le 1.3\times10^{-4}$).
- **Turned jets.** At every stage after a, $R^2$ is −1.3 to −2.5, below the band (rank p 0.67–1.0). Correlation of the read-out with the true $\Delta$ on turned jets: 0.00–0.16 at P..logits, vs 0.78–0.88 on shifted jets at s..z. At stage a, $R^2$ 0.31 lies inside the band (rank p 0.38).
- **Retrained networks.** Same pattern (pool MLP $R^2$ 0.51–0.86 at S/h4; turned subset −0.9 to −2.5 at h4). The float retrains carry $\Delta$ more (0.84–0.92 at s..S).
- **turnT** (fit where $\Delta$ moves through $m_A$, test where it moves through $m$): $R^2$ −1.9 to −4.0 at every stage. Even the $(M_0,|M_1|,M_2)$ input baseline fails this test (MLP −7.9, ridge −11.3) while it scores 0.977 when the turn is in the fit set. The transfer test therefore cannot distinguish the hypotheses and is reported for completeness only.
- **revT:** −0.06 to 0.00 at every stage.

### 2. Calibrated read-out response to the turn
A ridge $\log m$ read-out fitted on nat + shift + dil was applied to the turn_hard14 10 jets and to their dil10 copies (same $\Delta\log m=-0.105$; test jets). R_m = mean read-out change under the turn ÷ mean change under the real mass change, where 1 means the read-out follows the true mass:

| stage | a | P | g | s | u | S | h4 | h5 | z | logits |
|---|---|---|---|---|---|---|---|---|---|---|
| R_m | 0.038 | 0.002 | 0.95 | 0.001 | 0.15 | 0.25 [0.23, 0.26] | 0.087 [0.076, 0.099] | 0.048 | 0.034 | 0.042 |

The g read-out ($R^2$ for $\log m$ only 0.63) moves with the turn at 0.95 of a real change. g is the context vector built from the first-layer half-plane objects, which respond to azimuthal moves. At h4 the response is 0.087, vs −0.014..+0.001 for E27w's read-out fitted on natural jets only. The difference comes from the shift rows in the fit set: they teach the read-out to discount the centroid offset, so it now responds to the turn's growing $|\vec M_1|$. The pool $\Delta$ read-out's slope on the turn (predicted ÷ true $\Delta$ change) is 0.13 at S, 0.094 at h4 and 0.060 at h5.

### 3. Call changes from the edits themselves (test sample)
- turn_hard14 10 changes 4.5 % of calls: W→Z 2.2 %, Z→W 3.9 %.
- dil10, the real 10 % mass change, changes 33.8 %: Z→W 91.5 %, W→Z 0.
- shift6 changes 59.8 %: W→Z 64.6 %.
- turnup changes 1.2 %.

### 4. Interchange at S, h4, h5 (test pairs)
**Turn pairs** (true-m counterfactual). IIA on the 143 informative pairs, and share of all 3,161 calls changed by the swap:

| stage | $v_\Delta$ DAS IIA (inf.) | random median | DAS calls changed | random median / 95th pct | McNemar vs random (median p) | flow W→Z / Z→W | $\Delta$-probe swap calls changed (rank p) |
|---|---|---|---|---|---|---|---|
| S | 0.60 [0.52, 0.68] | 0.15 | 5.2 % [4.4, 5.9] | 1.9 % / 4.0 % | 1.8e-16 | 5.4 % / 4.5 % | 4.1 % (0.055) |
| h4 | 0.44 [0.36, 0.52] | 0.056 | 3.2 % [2.6, 3.8] | 0.63 % / 0.92 % | 2.2e-16 | 4.2 % / 3.2 % | 0.57 % (0.71) |
| h5 | 0.34 [0.26, 0.41] | 0.077 | 2.0 % [1.5, 2.5] | 0.66 % / 0.95 % | 1.7e-7 | 3.1 % / 2.1 % | 0.66 % (0.52) |

Without any swap, 95.5 % of turned jets already have the original's call. The cosine between $v_\Delta$ (DAS) and the $\Delta$ read-out direction is 0.06–0.09.

**Shift pairs** (positive control, $v_{m_A}$). IIA on 2,933 informative pairs: 0.62 / 0.65 / 0.44 at S / h4 / h5, vs random 0.018 / 0.009 / 0.012. Calls changed 51.5 / 50.5 / 36.8 % (random 95th pct 17.4 / 3.7 / 6.5 %). W→Z 78 / 68 / 46 %, matching the shift's own W→Z (65 %).

**Steering calibration.** Moving the activation along the gradient of a ridge read-out, by the amount that shifts that read-out by a 10 % mass change, moves the W–Z margin by R = (steered response) ÷ (response to dil10, which is −3.50 log-odds):
- $\Delta$ read-out: −0.023 / −0.033 / −0.058 at S / h4 / h5. At S the step is large and changes 84 % of calls in both directions, so the mean margin cancels.
- $\log m_A$ read-out (positive control): 0.11 [0.09, 0.14] / 0.001 / −0.001.
- $\log m$ read-out: 0.041 / −0.022 / −0.011.

Because the $m_A$ read-out direction itself does not move the decision at h4/h5, read-out directions are not causal handles here, as found in E35/X10. The steering numbers therefore do not test the hypotheses.

### 5. Replication (seeds 4111/4112; main and swap arms; band from 4101/4102)
- Accuracy 0.809 / 0.814.
- Pool MLP $R^2(\Delta)$: S 0.685, h4 0.710. Shifted subset: 0.74 / 0.77. Turned subset: −1.4 to −2.4 at P..logits. Stage a turned subset: 0.49. $(M_0,|M_1|,M_2)$ baseline on turned jets: 0.978.
- R_m: S 0.23, h4 0.077, g 0.85.
- C2ST axis-free: shift 0.49–0.50, turn10 0.84, dil10 0.84.
- DAS on turn pairs (145 informative): IIA 0.55 / 0.50 / 0.42 vs random 0.19 / 0.10 / 0.10. Calls changed 4.9 / 2.9 / 2.5 % vs random median 2.1 / 0.83 / 0.80 %. W→Z 5.8 / 3.8 / 4.0 % and Z→W 4.0 / 1.8 / 1.6 %.
- $\Delta$-probe swaps at h4/h5 inside the random band (rank p 0.49, 0.79).
- Cosine between DAS and probe directions 0.10.

## Reading (scoped to these measurements)
- **Where $\Delta$ is readable.** A quantity correlated with $\Delta$ is linearly and nonlinearly readable above untrained and permuted networks from the second-round statements (s, u) through the logits. On held-out jets this readability comes from the translated jets, where $\Delta$ moves because $m_A$ rises with the centroid offset $|\vec M_1|$. On jets where only the true mass moves (hardest-14 turn, $m_A$ fixed to 1e-9), no stage after the first layer predicts $\Delta$ better than a constant, while the same read-out on the three moments $(M_0,|\vec M_1|,M_2)$ reaches 0.98. The network therefore encodes the effect of an axis displacement but not the mass deficit that an azimuthal rearrangement produces at fixed $(p_T,r)$.
- **First layer.** The first-layer statements (a) carry $\Delta$ on turned jets only at the level of untrained and permuted networks (0.31 vs band up to 0.56). That is input information that any random first layer retains, not something trained.
- **Interchange.** A 1-dim direction at S/h4/h5 transfers 34–60 % of the turn's call changes back to the original's call, well above random directions. Those call changes go both ways (W→Z 3–5 %, Z→W 2–5 %), they are 1/7 of a real 10 % mass change's call changes, and the direction is nearly orthogonal to the $\Delta$ read-out. E27w/E27wv found that the turn's call changes are matched by an M1-restoring random-sign control, so they are not mass-ordered. Swapping the $\Delta$ read-out direction itself changes no more calls than random directions at h4/h5. The DAS direction has not been tested against DAS trained on the E27w control pairs (same particles, same turn sizes, $m$ held). That is the comparison needed to say whether it carries anything mass-specific.

## Predictions and which are excluded
| hypothesis | prediction | outcome |
|---|---|---|
| A4: M2-like axis mass, true $m$ not built | $R^2(\Delta)$ inside the band at every stage; the turn moves no read-out | Pool and shifted-jet $R^2$ are above the band at s..logits, so A4's literal prediction fails for the pool. On turned jets every stage is at or below the band, and read-outs move 0.03–0.25 of a real change (g: 0.95). Consistent with A4 for the true-mass component. |
| Alt 1: true $m$ computed, not used | $R^2(\Delta)$ above the band somewhere; $v_\Delta$ swap no better than random | Above the band only through the shifted jets (an $m_A$/centroid effect, not true mass). The $\Delta$-probe swap is random-level at h4/h5. "True $m$ computed" is not supported for the turn. |
| Alt 2: true $m$ computed and used weakly | swap moves calls above random with flow toward the true-$m$ counterfactual | DAS swaps exceed random, but flow is symmetric (not toward the heavier class), at 1/7 of a real change's call changes. Not excluded and not supported; the decisive control (DAS on M1-matched control pairs) was not run. |
| Alt 3: $\Delta$ readable for architectural reasons | untrained/permuted reach trained $R^2$ | Holds at a, P and g (inside the band). Excluded at s..logits for the pool and shifted jets (Holm $p_z\le 3\times10^{-5}$). |

## Issues and limits
- Turn family = hardest-14 turns. The soft-half turn was solvable for 621 / 64 / 4 test jets at 2 / 5 / 10 %, and the opposite turn changes $m$ by only +0.016 % (median), so turns that raise $m$ are absent.
- The turn changes $m$ by at most 10 %, while shift6 changes $m_A$ by 22 %. The turned subset's $\Delta$ variance is smaller, which makes negative $R^2$ easier. The correlation figures (0.00–0.16 turned vs 0.78–0.88 shifted) do not depend on that variance and show the same split.
- turnT/revT transfer fails even for the exact-moment baseline, so it does not discriminate between the hypotheses.
- The MLP budget differs between the main arm (25k rows / 120 iterations) and the controls (15k / 80). Band tests use the matched official_n64 control-run values.
- Interchange was run only at S, h4 and h5, where the exact tail exists. P, g, s and u were not swapped; a, P and g were inside the band anyway.
- Steering calibration is uninformative: the positive control ($m_A$ read-out direction) gives R ≈ 0 at h4/h5.
- Pod peak memory reached the 2 GiB limit in the smoke run, so the full arms ran at 1 CPU / 3 GiB (peak 2.4–3.0 GiB).
- Debug log:
  1. Smoke run: a version with zero kept jets crashed the edit-check summary; guarded. Not counted as a debug attempt.
  2. Debug 1 of 2: the untrained-redraw control crashed at seed 7, whose h4 has no live unit (all 64 units constant), leaving the MLP with zero features. Dead stages now give a constant read-out ($R^2\approx0$); the arm was rerun.

## Suggested follow-ups
- DAS on the E27w M1-restoring control pairs (same particles and turn sizes, $m$ held) at S/h4/h5. If that direction transfers the same share of call changes, the turn-DAS result is not about mass.
- Swap at g, where the read-out response to the turn is 0.95.

## Explainer entry (for `site/explain.py`, Paper A tab)
**Question.** Does any layer of JEDI-linear compute the true invariant mass $m$, separately from the axis-moment mass $m_A=\sqrt{M_0M_2}$ that earlier work found it uses?

**Background.** For small angles $m^2\approx M_0M_2-|\vec M_1|^2$. On real jets the centroid term is tiny ($m/m_A$ = 0.9997 median), so every earlier read-out, all fitted on natural jets, could not tell the two masses apart. The difference is $\Delta=\log m-\log m_A$.

**Method.** Build jets where the two disagree:
- shifting every particle off the axis raises $m_A$ at fixed $m$;
- turning the 14 hardest particles in azimuth about the axis lowers $m$ by 2–10 % with every particle's $p_T$ and $r$ fixed, so $m_A$ is fixed;
- scaling all angles changes both together.

Fit linear and small-MLP read-outs of $\Delta$ at each of the 10 stages, score them on held-out jets, and compare with 20 untrained or weight-permuted networks. Then copy a learned direction from unturned into turned jets' activations (DAS) and count how many calls follow the true-mass counterfactual, against 200 random directions.

**Result (figure `e41_true_mass.png`).**
- On shifted jets $\Delta$ is readable from the second-round statements onward (MLP $R^2$ 0.68 at S, 0.74 at h4; untrained band < 0).
- On turned jets no stage after the first layer predicts $\Delta$ better than a constant ($R^2$ −1.3 to −2.5), while the same read-out on $(M_0,|\vec M_1|,M_2)$ reaches 0.98.
- A DAS direction undoes 34–60 % of the turn's few call changes (above random), but those changes go both ways and total 1/7 of a real 10 % mass change's.

**Checks.**
- Replication on samples 4111/4112.
- 7 retrained networks show the same pattern.
- C2ST: turned jets differ from real jets on axis-free observables by the same AUC as a real mass change of equal size (0.83 vs 0.84).

**Why it matters.** The network's internal "mass" responds to particle displacement from the stored axis, not to the four-vector mass. A change of the true mass that leaves $(p_T, r)$ about the axis fixed is not represented.

## Conclusion (site/conclusions.py entry, EXPLAINER and CLAIM A4)
JEDI-linear's layers carry a quantity that separates the true jet mass from the axis-based mass $m_A=\sqrt{M_0M_2}$ only when particles are translated off the stored axis. When the true mass is lowered by up to 10 % at fixed $m_A$, no stage after the first layer tracks the change, doing worse than untrained networks. A learned direction undoes a few of the edit's call changes, but they go to W and Z about equally. There is no sign that the network builds the invariant mass; whether that direction is mass-specific has not been checked against a mass-preserving control.
