# E43 verify: azimuthal-arrangement route in JEDI-linear

Node E43, verifier pass. The worker's reading under test: "No measurement. The script has not executed past import on the cluster, so its runtime correctness and timing are untested."

Jobs run:
- `e43-vsmoke`: unpatched module, smoke size. Crashed at step 3.
- `e43-verify`: unpatched module, full size. Crashed at step 3, at the same line.
- `e43-vsmoke2` and `e43-verify2`: patched in memory, as described under Bug 1. Both completed.
- `e43-vextra`: extra controls. Completed.

Every job ran at 1 CPU / 2 GiB. Peak memory was 1.0–1.9 GiB. `e43-verify2` used 0.94 cores on average over 29 min.

Outputs:
- `$AI_RESULTS/jedi/e43_verify_main_s2243.json`
- `$AI_RESULTS/jedi/e43_verify_checks_s2243.json`
- `$AI_RESULTS/jedi/e43_verify_extra.json`
- `$AI_RESULTS/jedi/e43_verify_main_smoke2.json`
- `$AI_RESULTS/jedi/e43_verify_example_s2243.npz`
- figure `results/jedi/e43_verify_example.png`

Copies of all of these are in `results/jedi/`, and the logs are in `logs/e43-*.log`.

Verifier code:
- `interp/e43v_verify.py` runs the worker's module unchanged, except for the one-line patch applied in memory, and adds independent edit checks.
- `interp/e43v_extra.py` adds a W/Z-only test and a selection-matched C2ST.

The worker's file `interp/e43_azimuth_route_jedi.py` was not modified.

## Conclusion
The worker's statement is accurate: their script had never run. Running it now shows it could not have finished. A coding error stops it halfway, so the "untested" status hid a real bug.

With that one line fixed, the planned tests ran on a fresh sample:
- Changing how elongated the hardest particles' pattern is (its quadrupole $Q_2$), with the jet mass held fixed, moves the network's W-versus-Z score no more than a scrambled rearrangement of the same particles. The quadrupole-specific part is at most about 1 % of the effect of a real 10 % mass change.
- No single arrangement quantity predicts the score change.
- The effect of rearranging the hardest particles reaches the output mainly through the per-particle branch of the network. It does not run mainly through the two jet-level context units S31 and S58.

Still uncertain: the quadrupole edit can be solved for only a few percent of W and Z jets. So the W/Z test rests on 85–250 jets for the pT-weighted quadrupole, and on 368–1,266 for the particle-count-weighted one.

## Is the worker's reading correct?
Yes, as a statement of fact. No E43 job completed before this pass.

The reading is incomplete in one respect. It says the script's "runtime correctness ... is untested". The script has a crash bug that stops every run at step 3, so the full run (`e43-run`) would have failed as written.

## Bugs found in `interp/e43_azimuth_route_jedi.py`
1. **Crash in step 3 (line 497).**
   ```python
   WbP = np.random.default_rng(4303).poisson(1.0, (NB, len(rows)))
   ```
   Here `rows` is a boolean mask over all N jets. The weights therefore have N columns where they need `rows.sum()`. The result is:
   `ValueError: operands could not be broadcast together with shapes (3750,6) (7500,1)`.
   This was seen in `e43-vsmoke` and `e43-verify`. The fix is `int(np.sum(rows))`, which the verifier applies in memory.
2. **The C2ST compares edited jets with a differently selected natural set (lines 553–558).**
   - C2ST is a two-sample classifier test: AUC 0.5 means edited and natural jets are indistinguishable.
   - The script compares all natural jets of one half with the edited jets the solver kept in the other half.
   - The Q2 solver keeps mainly low-$Q_2$, low-multiplicity, quark-like jets. So the AUC also measures this selection.
   - On true W/Z jets (`e43-vextra`), for the Q2+ edit:

     | Comparison | Axis-free AUC |
     |---|---|
     | As the script does it | 0.85 |
     | Same kept jets, unedited (selection only) | 0.78 |
     | Selection-matched | 0.54 [0.46, 0.61] |

   The script's Q2 C2ST numbers are therefore not interpretable as "edited jets look unnatural".
3. **The random-hold null cannot be formed for three of the six named holds.**
   - The null is a rank against 200 random holds of the same size, drawn from the other live units of the same layer.
   - For `P_HP9`, `s_poly5` and `u_poly15` there are too few other live units: 3, 4 and 4 against hold sizes 9, 5 and 15.
   - The JSON records only a note. So the s/u branch holds have no random-unit control. The null exists only for the S-unit holds.
4. **The naming p-values cannot pass Holm by construction.**
   - The margin, S31 and S58 targets use 200 shuffled pairings. The minimum p is then 1/201.
   - About 33 candidate observables enter the Holm correction, so the smallest Holm p is 0.159 for every target. The MARS null (20 draws) has a minimum p of 0.048.
   - Effect sizes far above the null (for example $R^2$ 0.39 against a null q95 of 0.001) are therefore reported as Holm p 0.159. The number of draws should be at least 33/0.05, about 700.
5. **The control-network comparison (step 5) runs on 41 jets.**
   - It uses only jets kept by ctrl10, Q2cp, Q2cm and dil10 at once, and only test jets.
   - On those 41 jets the trained net's accuracy is 0.83. Most untrained and l1-permuted nets have accuracy 0.07–0.45, and three have margin sd 0. Two untrained redraws and l1perm_12 are degenerate.
   - The ranks p 0.33–0.67 therefore have little power.
6. **Docstring inaccuracies.**
   - It says ctrl10 holds m to about 1e-5. The median |Δlog m| is 1.7e-5, but the maximum is 0.0082.
   - It says Fwd reproduces `Jedi.run` "to 0". This is true: max |Δlogit| = max |ΔS| = 0.
7. **Structural tautologies in the hold table.** These are true by construction, not results:
   - `a_HP9` removal is 1.000 for every edit. Only the nine half-plane l1 units have angular weight, and pT is unchanged.
   - `g_all` equals `P_HP9` exactly, because g sees the angle edit only through P of those nine units.
   - `P_HP9+s_poly5` = 1.000.

No split leakage was found:
- The fit/test split is a random permutation by jet.
- An edited jet and its natural partner fall on the same side.
- Partners for the pair slope come from the fit jets.

Padded slots are never counted as particles. My independent check confirms the following:
- No padded entry changes.
- No pT changes.
- No particle outside the 14 hardest moves for turn10, ctrl10, Q2± or Q2c±.
- The radius r changes by at most 5e-8.

## Procedure (e43-verify2, patched; fresh seed)
**Sample.** `interp.jedi_core.sample(per_class=1500, seed=2243)`: 7,500 jets of W, Z, top, q and g. These are split into fit and test halves by a random permutation (seed 2244).

**Network.** JEDI-linear 3-feature n64. Accuracy is 0.8056 on the sample (fit 0.812, test 0.799). The replica matches exactly (Δ = 0). The l1 units with angular weight are {3, 6, 29, 30, 36, 40, 43, 46, 57}, the same nine half-planes as E14.

**Edits.** Every edit touches only the 14 hardest particles, and each particle keeps its pT and its distance r to the stored axis. So
$M_0=\sum p_T$, $M_2=\sum p_T r^2$ and $m_A=\sqrt{M_0M_2}$ are fixed.
- **Q2± (pT quadrupole):** scales $C_2=\sum p_T r^2 e^{2i\theta}$ by $1\pm0.2$ at fixed orientation, holding vector $\vec M_1=\sum p_T(\Delta\eta,\Delta\phi)$ and the true mass m. Here $Q_2=|C_2|/M_2$ measures how elongated the pT-weighted pattern is: 0 for round, 1 for a line.
- **Q2c± (control):** the same particles, each moved by the same $|\delta\theta_i|$ with random sign, then projected so that $\vec M_1$, m and complex $C_2$ are unchanged. It is rescaled to the same total displacement $D=\sqrt{\sum_i|\delta\vec x_i|^2}$. The control changes the arrangement by the same amount but leaves the quadrupole fixed, so edit minus control isolates the quadrupole.
- **Xq2± and Xq2c±:** the same pair for the count-weighted quadrupole $q_2=|\sum r e^{2i\theta}|/\sum r$. The script chose it automatically as the top harmonic candidate in naming.
- **turn10 / ctrl10:** the E41 turn, which lowers m by 10 % by turning the hardest 14 toward $\phi(\vec M_1)$, and its random-sign, $\vec M_1$-holding control.
- **dil10:** all angles scaled so that m falls 10 %. This is a real mass change, used as the reference.
- **rot45:** the whole jet rotated by 45°.

**Metric.** The W–Z margin is $\mathrm{logit}_W-\mathrm{logit}_Z$. We report mean $|\Delta\text{margin}|$ for each edit, the edit-minus-control difference, calls changed and McNemar tests. The bootstrap uses 1,000 Poisson draws over jets.

## Results

### 1. Edit checks (independent recomputation, `e43_verify_checks_s2243.json`)

| edit | kept / 7,500 | max \|Δm\|/m | max \|Δ$\vec M_1$\|/M0 | median ΔQ2/Q2 | max rel. \|ΔC2\| |
|---|---|---|---|---|---|
| Q2p | 925 | 6.7e-6 | 1.0e-6 | +0.200 | 0.200 |
| Q2cp | 657 | 4.4e-7 | 9.9e-8 | 0.000 | 2.7e-5 |
| Q2m | 1,769 | 9.6e-6 | 1.0e-6 | −0.200 | 0.201 |
| Q2cm | 881 | 1.5e-7 | 3.4e-7 | 0.000 | 8.1e-5 |
| turn10 | 4,744 | 0.100 | 0.089 | −0.46 | — |
| ctrl10 | 4,704 | 0.0082 | 3.2e-9 | −0.36 | — |

**Selection.** The fraction of jets kept by class is:

| edit | W | Z | g | q | top |
|---|---|---|---|---|---|
| Q2p | 0.043 | 0.043 | 0.105 | 0.297 | 0.129 |
| Q2m | 0.18 | 0.18 | — | 0.40 | — |

Median natural $Q_2$ is 0.32 for kept jets and 0.77 for dropped ones. The likely cause is that raising $|C_2|$ by 20 % is impossible when $Q_2>0.83$, because $Q_2\le1$. So in the main run the Q2 test is mostly a test on quark, gluon and top jets.

**Side effects.** These are axis-free shape observables, edit vs matched control, in units of their natural sd, with a paired bootstrap:
- Q2p and its control change $e_2^{(\beta)}$ ($\beta$ = 0.5, 1) and the hardest-pair $\Delta R_{01}$ by the same amount (p 0.08–0.78).
- Q2m changes $e_2^{(1)}$ by 0.032 against 0.016 for its control, and $\Delta R_{01}$ by 0.089 against 0.060 (p < 0.001).
- So Q2m is not perfectly matched on prong-separation observables. These differences are small (< 0.1 sd).

The figure `results/jedi/e43_verify_example.png` shows one W jet (m 80.2 GeV, Q2 0.427 → 0.513). The pT-quadrupole edit moves mainly the softer members of the hardest 14 that sit at large r, by up to about 0.1 in $(\Delta\eta,\Delta\phi)$. The two leading prongs barely move. On this jet the margin goes from +5.43 to +5.20 under the edit and to +5.31 under the control.

### 2. Responses (test jets; dil10 reference |Δmargin| 3.08)
Mean |Δmargin| for each edit: ctrl10 0.367, turn10 0.585, rot45 0.392, Q2p 0.083, Q2m 0.172, Q2cp 0.073, Q2cm 0.085.

S58 is the S unit that moves most, relative to its sd, under every arrangement edit:

| edit | S58 |ΔS|/sd |
|---|---|
| ctrl10 | 0.58 |
| turn10 | 0.62 |
| rot45 | 0.52 |
| Q2p | 0.15 |
| Q2m | 0.20 |

Under dil10, S56 and S1 move most instead.

### 3. Naming (ctrl10 + turn10 pooled; held-out $R^2$ on test jets)
- **Margin change:** best candidate pt_all9 (pT inside each of the nine half-planes) with $R^2$ 0.040 [0.025, 0.053]. All candidates jointly reach 0.045. $Q_2$ gives 0.001 and the E24x count quadrupole −0.000. The best MARS hinge fit gives 0.046. No candidate predicts the margin change.
- **S58 change:** count quadrupole $q_2$ gives $R^2$ 0.24 [0.22, 0.26] linear and 0.28 [0.25, 0.30] MARS. $Q_2$ gives 0.21. The shuffled-pairing null q95 is 0.001.
- **S31 change:** count dipole $q_1=|\sum r e^{i\theta}|/\sum r$ gives $R^2$ 0.39 [0.36, 0.41] linear and 0.47 [0.44, 0.50] MARS. Null q95 is 0.001.
- Holm p is 0.159 for every target. This is the floor set by Bug 4, not a weak effect.

### 4. Causal test of the quadrupole
**Hypotheses.**
- H0: the margin response to a ±20 % quadrupole change equals that of the matched control.
- H1: it is larger.

**Main run** (all classes, test jets):

| edit | n | \|Δmargin\| edit | \|Δmargin\| control | difference [CI] | p | calls changed (edit vs ctrl) | McNemar p |
|---|---|---|---|---|---|---|---|
| Q2p | 333 | 0.076 | 0.072 | [−0.009, 0.016] | 0.28 | 1.2 % vs 0.9 % | 1.0 |
| Q2m | 442 | 0.104 | 0.083 | [0.007, 0.037] | 0.001 | 1.4 % vs 1.1 % | 1.0 |

- Antisymmetric part, (Q2p − Q2m)/2: −0.0076 [−0.018, 0.0006]. The control gives +0.0008.
- R_Q is the response divided by the slope b of margin on $Q_2$ over real nearest-neighbour jet pairs. Here b = 0.13 [−0.06, 0.35], so R_Q is undefined in practice: R_Q 0.45 [−1.7, 2.7].

**True W and Z only** (`e43-vextra`; fresh sample seed 2343, 6,000 W/Z jets). "Specific" is |edit| − |control|.

| edit | n | specific [CI] | specific ÷ dil10 [CI] | p | calls changed: edit / ctrl / dil10 |
|---|---|---|---|---|---|
| Q2p | 85 | −0.010 [−0.070, 0.049] | −0.003 [−0.018, 0.013] | — | 3.5 % / 4.7 % / 53 % |
| Q2m | 250 | +0.049 [−0.004, 0.105] | 0.013 [−0.001, 0.028] | 0.075 | 2.4 % / 2.0 % / 46 % |
| Xq2p | 368 | 0.003 [−0.035, 0.041] | 0.001 [−0.009, 0.010] | — | — |
| Xq2m | 1,266 | 0.040 [0.013, 0.066] | 0.009 [0.003, 0.015] | 0.001 | 1.3 % / 1.3 % (McNemar 10:10) |

**Competing hypotheses.**
- "The network reads the pT quadrupole $Q_2$ like a mass" predicts specific ÷ dil10 of order ΔQ2-sized effects, i.e. calls changed well above the control. This is excluded at the measured CI upper bound of 0.028 of a real mass change, on the jets where the edit is solvable.
- "Reads the count quadrupole" gives a nonzero but tiny effect: 0.009 of a real mass change, with no change in calls.
- "Reads only the amount of rearrangement" is consistent with all four rows.

### 5. Route (holds; removal = 1 − Σ|Δmargin held| / Σ|Δmargin free|)

| hold | ctrl10 (n 2,328) | turn10 | Q2p | Q2m |
|---|---|---|---|---|
| s_poly5 (5 s units fed by the half-planes) | 0.60 [0.58, 0.63] | 0.70 | 0.61 | 0.61 |
| u_poly15 | 0.64 | 0.72 | 0.68 | 0.67 |
| P_HP9 = g_all (context branch) | 0.10 [0.08, 0.12] | 0.06 | 0.09 | 0.06 |
| S31+S58 | 0.086 [0.065, 0.106] | 0.041 | 0.047 | 0.054 |

Against 200 random S-unit holds of the same size (Holm over three S holds per edit), S31+S58 has rank p 0.37 (ctrl10), 0.36 (Q2p) and 0.18 (Q2m), Holm 0.90–1.0. S58 alone under Q2m has p 0.005, Holm 0.03, removal 0.072.

The single S units that remove most are:
- ctrl10: S13 0.128, S40 0.067, S58 0.063.
- turn10: S29 0.135, S13 0.111, S43 0.103.

The s/u holds have no random-unit null (Bug 3).

So most of the rearrangement effect passes through the structural (per-particle) s → u branch, about 0.6–0.7. The context branch P → g carries about 0.06–0.10. S31 and S58 together carry 0.04–0.09, which is not above random S units.

### 6. Control networks (41 jets; low power, Bug 5)
On these jets, the trained net's ctrl10 response divided by its dil10 response is 0.115, and its Q2-specific response is 0.011 sd [−0.002, 0.026].

| network family | Q2-specific (sd) | rank p | ctrl10/dil10 | rank p |
|---|---|---|---|---|
| 20 l1-permuted | median 0.013 | 0.62 | 0.30–1.8 | — |
| 5 untrained | — | 0.33 | — | 0.67 |

The trained net does not differ from either control family on these statistics.

## Corrected reading
E43 (run by the verifier with one bug fixed; seeds 2243 and 2343):
- On the jets where the edit is solvable, JEDI-linear does not respond specifically to the hardest particles' pT-weighted quadrupole at fixed m, M0, M2 and $\vec M_1$. The edit-minus-control margin change is at most 0.028 of a real 10 % mass change on true W/Z jets, with no change in calls.
- The count-weighted quadrupole has a small specific effect: 0.009 [0.003, 0.015] of a real mass change.
- Under the E41 rearrangements no single arrangement observable predicts the margin change (held-out $R^2$ ≤ 0.05). The changes of S58 and S31 follow the count-weighted quadrupole and dipole ($R^2$ 0.24–0.28 and 0.39–0.47).
- About 0.6–0.7 of the margin change passes through the five s units fed by the half-plane cuts, 0.06–0.10 through the context branch, and ≤ 0.09 through S31+S58, which is not above random S-unit holds.
- The route "half-planes → s/u → S31/S58" is supported for its first part. It is not supported as an S31/S58 route.

## Explainer entry (for site/explain.py)
**Question.** When the 14 hardest particles of a jet are rearranged around the jet axis without changing the jet mass, which property of the arrangement does JEDI-linear respond to, and through which internal units?

**Background.**
- Earlier nodes established that the network's mass-like variable is $M_2=\sum p_T r^2$ about the stored axis (A4). They also showed that turning the hardest particles in azimuth at fixed $M_2$ changes about 4 % of calls (A4m). That response is not mass-specific: a scrambled turn does 60–70 % of it.
- The candidate quantity tested here is the pT-weighted quadrupole
  $$Q_2=\frac{\left|\sum_i p_{T,i} r_i^2 e^{2i\theta_i}\right|}{\sum_i p_{T,i} r_i^2},$$
  which is 1 for particles on a line through the axis and 0 for a round pattern. A two-prong W or Z jet has $Q_2\approx0.8$.

**Method.**
1. Change $Q_2$ by ±20 % by moving only azimuths $\theta_i$ of the hardest 14, while holding the jet mass and $\vec M_1$ fixed. This needs a constrained Gauss–Newton solve.
2. Compare with a control that moves the same particles by the same amounts with random signs and keeps $C_2$ fixed.
3. Calibrate against a real 10 % mass change (dil10).
4. For the route, freeze groups of units at their unedited values and measure how much of the margin change disappears.

**Result.** The figure is `e43_verify_example.png`.
- Specific quadrupole response on W/Z jets: ≤ 0.028 of a real mass change.
- Count quadrupole: 0.009.
- Route: s units 0.6–0.7, context 0.06–0.10, S31+S58 ≤ 0.09, not above random.

**Checks.**
- The fresh-seed rerun and the extra-sample run (2343) agree in sign and size.
- Edit invariants hold to below 1e-5.
- The C2ST was corrected for selection.
- Not covered: two-prong jets with $Q_2>0.83$, where +20 % is infeasible, and a random-unit null for the s/u holds.

**Why it matters.** The network's sensitivity to how the hardest particles are arranged around the axis is not a sensitivity to their elongation. Most of it travels particle by particle, before pooling, rather than through the jet-level context units.

## Glossary additions
- **$Q_n$:** $|\sum p_T r^2 e^{in\theta}|/\sum p_T r^2$, the pT r²-weighted azimuthal harmonic n about the stored axis.
- **$q_n$:** $|\sum r e^{in\theta}|/\sum r$, the same harmonic weighted by particle count (each particle weighted by r).
- **Removal share:** $1-\sum_j|\Delta\text{margin}_{\text{held}}|/\sum_j|\Delta\text{margin}_{\text{free}}|$, the fraction of an edit's margin change that disappears when a unit group is frozen at its unedited value.
- **Selection-matched C2ST:** a classifier test comparing edited jets with the same subset of jets unedited, from the other half of the sample.
