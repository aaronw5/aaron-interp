# E27: which radial observable is JEDI-linear's mass variable (angularity family, matched radial edits; claims A4 / A3; exploratory; agent report 2026-10-21)

## Conclusion
We moved the particles of real jets toward or away from the jet axis, without changing their pT or direction around the
axis. The moves were chosen so that $M_2=\sum p_T r^2$ (the pT-weighted squared spread) changed while one other angularity
(another power of $r$, or another pT weighting) stayed fixed, or the reverse. Raising $M_2$ with any of the 14 others held
moves the network's mass reading as a real mass change does and sends about 8 % of W-called jets to Z. Raising any other
member with $M_2$ held sends no more W jets to Z than a random move of the same size. Within this family, then, the mass
response follows $M_2$, not particle-count-weighted or pT²-weighted spreads and not the girth-like $\sum p_T r$.
Still uncertain: the planned test (move one member while holding all 14 others) cannot be done, because the members are almost
linearly dependent. Among the pT-weighted neighbours ($\beta$ = 1.5, 3), holding $\sum p_T r^3$ changes the size of the $M_2$
response (h4 read-out 0.67, h5 read-out 1.39, W → Z 13 % vs 10 %), so the exact power of $r$ near 2 is not pinned down.

## Terms used
- **Angularity** $\lambda(\kappa,\beta)=\sum_i p_{T,i}^{\kappa}\, r_i^{\beta}$. Here $r_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$
  is particle $i$'s distance to the stored jet axis, and the sum runs over the jet's stored particles (up to 64).
  - $\kappa$ sets the pT weighting: 0 counts particles, 1 weights by pT, 2 weights by pT².
  - $\beta$ is the power of $r$: low $\beta$ weights particles near the axis more, high $\beta$ weights the outskirts.
  - $\lambda(1,2)=M_2$, the pT-weighted spread that enters the jet mass, $m^2\approx M_0M_2-|\vec M_1|^2$.
  - $\lambda(1,1)=\sum p_T r$ is girth up to normalisation. $\lambda(1,0)=M_0=\sum p_T$, and $\lambda(0,0)$ is the particle count.
- $\vec M_1=\sum_i p_{T,i}(\Delta\eta_i,\Delta\phi_i)$: the pT-weighted offset of the particles from the axis.
- **Radial edit:** each particle's $(\Delta\eta,\Delta\phi)$ is multiplied by $e^{\delta_i}$.
  - Its pT, its direction around the axis, and therefore its pT rank and input slot do not change.
  - $M_0$ and the particle count stay exactly fixed.
- **D:** angular displacement size, $D=\sqrt{\sum_i(\Delta r_i)^2}$ in rad.
- **Mass read-out $\hat m$:** a ridge (linear) fit of $\log m$ on the 64 values of layer h4, or the 32 values of layer h5.
  - Fitted on unedited jets (half A of the sample); responses are evaluated on half B.
  - Held-out $R^2$ = 0.970 for both h4 and h5.
- **R:** the network's response to an edit divided by its response to a real mass change of the same predicted size,
  $R=\overline{\Delta\log\hat m}(\text{edit})/\overline{\Delta\log\hat m}(\text{dilation})$, computed on the same jets.
  - The dilation is $r\to\sqrt{1.05}\,r$ for every particle: $M_2$ rises 5 % and $M_0$ is fixed.
  - If the edited member were the network's mass partner of $M_0$, a 5 % change of it would give $\Delta\log m=\tfrac12\log1.05$, the
    same as the dilation, so R = 1. A member the network does not use for mass gives R = 0.
- **W → Z flow:** the share of jets the network called W before the edit that it calls Z after. It uses the calls only, so
  it does not depend on the read-out.
- **McNemar test:** a paired test of whether the edit and its null change different numbers of the same jets' calls.
- **Holm correction:** adjusts the p-values for the 14 members tested per edit type.

## Files
- Script: `interp/e27_angularity_jedi.py`. Figure: `interp/e27_plot.py`.
- Job: `./job.sh e27-run 1 3 'python -m interp.e27_angularity_jedi'`
  - Wall 2,880 s; pod peak 2,770 MiB; 0.88 CPU.
  - 3 GiB was requested because the 500-jet smoke run (`e27-smoke`, 1 CPU / 2 GiB) peaked at 2,047 MiB, the 2 GiB cap.
- Results: `results/jedi/e27_angularity.json`, `results/jedi/e27_angularity_perjet.npz` (per-jet read-out deltas, calls and
  keep masks for every edit; the example jet), `results/jedi/e27_angularity.png`.
- Logs: `logs/e27-run.log`, `logs/e27-smoke.log`.

## Procedure
- **Sample:** `interp.jedi_core.sample(per_class=2000, seed=1201)`, 10,000 jets, 2,000 per class, all 5 classes.
- **Network:** official 3-feature n64 JEDI-linear. Unedited accuracy on this sample is 0.8076 (expected 0.806–0.813).
- **Edit solver:** per jet, $\delta_i$ is the minimum-norm Gauss–Newton solution (batched pseudo-inverse, 12 iterations) of a
  set of constraints on $\log\lambda_k$ and on $\vec M_1$.
  - $\vec M_1$ is always held. Two rows scaled by $|\vec M_1|$ keep its length and direction.
  - $\delta$ is solved per particle, not as a smooth function $f(r)$. A function of $r$ alone treats two particles at the same $r$
    the same way, so it cannot separate $\kappa$ = 0, 1, 2 members at one $\beta$.
  - A jet is kept for an edit if:
    - each constrained member is within 0.5 % of its target (0.2 % at the 2 % dose);
    - $|\vec M_1|$ changes by ≤ 0.5 %;
    - $\max|\delta_i|\le\log 2$, so no particle more than doubles or halves its distance to the axis.
- **Arm 0, the planned design (one member +5 %, the other 14 and $\vec M_1$ held):** infeasible.
  - Kept jets out of 10,000: 0 for 13 members, 3 for (1, 3), 166 for (2, 3).
  - The median of $\max|\delta|$ hits the solver bound of 3.
  - Debug attempt 1 (local check on 100 real W jets, before any cluster run): the minimum-norm exact solution needs
    $\max|\delta|$ of 5 to 1,700. With 25–50 particles per jet the power sums $\sum p_T^\kappa r^\beta$ are nearly linearly dependent
    (smallest singular value of the constraint Jacobian about 1e-5 of the largest).
  - The measurement therefore uses arm 1.
- **Arm 1: pairwise decoupling edits.** For each member $a \neq (1,2)$:
  - **A_a:** $\lambda_a\times1.05$ with $M_2$ and $\vec M_1$ held. Does the response follow member $a$ once $M_2$ is fixed?
  - **B_a:** $M_2\times1.05$ with $\lambda_a$ and $\vec M_1$ held. Does the response to $M_2$ survive with member $a$ fixed?
  - **Null nA_a / nB_a:** a random direction in the null space of the same constraints ($\lambda_a$, $M_2$, $\vec M_1$ all held),
    scaled so that D matches A_a or B_a on the same jet (within 5 %).
    - The null changes the jet's radial profile by the same amount but leaves the tested quantities fixed. That makes it the
      right comparison for "any radial reshuffle of this size".
  - **M2h:** $M_2\times1.05$ with only $\vec M_1$ held (minimum-norm $M_2$ edit), and its matched null.
  - **ref:** the uniform dilation $r\to\sqrt{1.05}\,r$, the positive reference (c). It changes every $\beta>0$ member: by +1.2 %,
    +2.5 %, +3.7 %, +5 % and +7.6 % for $\beta$ = 0.5, 1, 1.5, 2, 3.
  - Jets used for member $a$: those kept in all four of A_a, B_a, nA_a and nB_a.
  - **Doses:** 5 % for all members. The four $\kappa$ = 1 members were also run at 2 %, because A_a at 5 % meets the $\log 2$ cap on
    only 31–61 % of jets and those kept jets are mostly q/g. At 5 % only 6–9 % of the jets kept for $\kappa$ = 1, $\beta$ ≤ 1.5 are
    true W, against 20 % in the full sample. The 2 % dose keeps 5,863–8,974 jets with a near-balanced class mix.
- **Untrained control (b):** `Jedi(untrained=True, seed=0)` on the same edited jets.
  - Calls are argmax(logits − mean logits on the unedited sample).
  - Its h4/h5 log-m read-outs have held-out $R^2$ = 0.00 / 0.05. Its R is therefore undefined, and calls and W → Z flow are
    the comparison.
- **Statistics:**
  - 95 % CIs: 2,000 Poisson bootstrap resamples over jets. Ratios are recomputed in each resample.
  - H0 for each member and edit: the response equals the matched-null response.
  - Read-out test: paired sign-flip permutation (20,000) on the per-jet difference (edit − null).
  - Calls and W → Z test: exact McNemar, edit vs null.
  - Holm correction over the 14 members, per edit type and metric.

## Results (trained network)

References (5 % dose; all 10,000 jets, of which 2,035 are W-called):

| edit | $R_{h4}$ | $R_{h5}$ | W → Z | calls changed |
|---|---|---|---|---|
| ref (dilation, M2 +5 %) | 1 (Δlog m̂ = +0.0246) | 1 | 8.3 % [7.1, 9.6] | 4.2 % |
| M2h (M2 +5 %, only $\vec M_1$ held) | 0.971 [0.96, 0.98] | 0.952 [0.94, 0.97] | 7.7 % [6.6, 8.9] | 4.1 % |
| null for M2h | −0.002 [−0.01, 0.01] | 0.002 | 0.54 % | 1.6 % |

M2h vs its null: W → Z discordant pairs 147 vs 1 (McNemar p 8e-43); read-out permutation p 5e-5.

Per member, 5 % dose (h4 R; h5 R; W → Z flow with the matched null in parentheses; n = kept jets):

| member $(\kappa,\beta)$ | n | A: $R_{h4}$ [CI] | A: $R_{h5}$ | A: W → Z (null) | B: $R_{h4}$ [CI] | B: $R_{h5}$ | B: W → Z (null) |
|---|---|---|---|---|---|---|---|
| (0, 0.5) | 8,464 | +0.04 [−0.03, 0.09] | −0.08 | 1.5 % (1.7) | 0.96 [0.94, 0.97] | 0.95 | 8.4 % (0.8) |
| (0, 1) | 9,856 | +0.07 [0.04, 0.10] | −0.12 | 1.1 % (1.4) | 0.96 | 0.98 | 8.1 % (0.8) |
| (0, 1.5) | 9,972 | +0.04 | −0.10 | 1.0 % (1.3) | 0.96 | 1.00 | 8.3 % (1.1) |
| (0, 2) | 9,990 | +0.02 | −0.08 | 0.8 % (0.9) | 0.97 | 1.01 | 8.0 % (0.7) |
| (0, 3) | 9,995 | +0.00 | −0.06 | 0.3 % (1.0) | 0.98 | 1.03 | 7.7 % (0.8) |
| (1, 0.5) | 3,302 | −0.15 [−0.24, −0.07] | +0.59 | 0.7 % (2.2) | 0.93 [0.89, 0.97] | 0.80 | 7.5 % (1.5) |
| (1, 1) girth-like | 4,002 | −0.32 [−0.39, −0.24] | +0.47 | 2.7 % (3.0) | 0.88 [0.84, 0.92] | 0.68 | 9.9 % (2.7) |
| (1, 1.5) | 3,143 | −0.55 [−0.66, −0.45] | +0.73 | 0.8 % (3.2) | 0.65 [0.57, 0.72] | 0.11 | 9.6 % (3.2) |
| (1, 3) | 6,145 | +0.02 [−0.01, 0.06] | −0.28 | 0.7 % (1.8) | 0.67 [0.62, 0.72] | 1.39 | 13.0 % (2.1) |
| (2, 0.5) | 8,950 | −0.32 [−0.36, −0.28] | −0.02 | 1.8 % (1.0) | 1.01 | 0.95 | 8.3 % (0.9) |
| (2, 1) | 9,967 | −0.14 | +0.00 | 1.7 % (1.0) | 1.02 | 0.95 | 8.0 % (0.7) |
| (2, 1.5) | 9,992 | −0.09 | −0.00 | 1.0 % (1.0) | 1.03 | 0.94 | 7.7 % (1.1) |
| (2, 2) | 9,991 | −0.07 | +0.01 | 1.0 % (1.0) | 1.02 | 0.93 | 8.1 % (0.7) |
| (2, 3) | 9,926 | −0.03 | +0.02 | 0.7 % (0.8) | 1.02 | 0.90 | 8.0 % (0.8) |

- Null-edit R: |R| ≤ 0.10 at h4 for every member and both edits; ≤ 0.17 at h5.
- W → Z flow, Holm-corrected McNemar vs null:
  - B edits: p ≤ 0.016 for all 14 members (≤ 1e-27 for the 10 members with ≥ 6,000 jets).
  - A edits: p ≥ 0.09 for all 14 members.
  - In 11 of the 14 A edits the W → Z flow is at or below its null.
  - No A edit's W → Z flow reaches a third of the dilation's flow on the same jets (6.7–9.9 %); the largest ratio is 0.28 (A (1, 1)).
- Calls changed (all classes), A edits vs null:
  - Above the null for 8 members (Holm p ≤ 0.005); below it for (0, 3) (1.4 % vs 1.8 %, p 0.036).
  - The added changes are not W → Z. For A (1, 1), the top flows are g → q 57, g → Z 30, g → W 26 (4,002 jets).
    For A (0, 0.5) they are t → g 76, q → g 68, W → g 62.
  - Reshaping the radial profile at fixed $M_2$ therefore moves quark/gluon and top calls, not W/Z calls.

$\kappa$ = 1 members at the 2 % dose (ref Δlog m̂ = +0.0100; dilation W → Z on the same jets 3.1–4.0 %):

| member | n | A: $R_{h4}$ | A: $R_{h5}$ | A: W → Z (null), Holm p | B: $R_{h4}$ | B: $R_{h5}$ | B: W → Z (null), Holm p |
|---|---|---|---|---|---|---|---|
| (1, 0.5) | 6,107 | −0.50 [−0.61, −0.38] | +0.65 | 3.1 % (1.9), 0.17 | 1.02 [0.97, 1.07] | 0.80 | 3.5 % (1.0), 1e-4 |
| (1, 1) | 6,957 | −0.46 [−0.56, −0.36] | +0.52 | 3.3 % (1.8), 0.055 | 1.02 [0.96, 1.08] | 0.69 | 3.0 % (1.6), 0.003 |
| (1, 1.5) | 5,863 | −0.69 [−0.82, −0.56] | +0.79 | 4.2 % (3.0), 0.20 | 1.01 [0.92, 1.10] | 0.39 | 3.6 % (1.7), 0.014 |
| (1, 3) | 8,974 | +0.11 [0.07, 0.16] | −0.22 | 0.6 % (1.1), 0.12 | 0.70 [0.64, 0.78] | 1.29 | 4.1 % (1.5), 8e-10 |

At 2 %, the A edits for $\beta$ ≤ 1.5 give W → Z flows close to the dilation's on the same jets, but none is separated from its
null after Holm (p 0.055–0.20). The h4 and h5 read-outs, which agree on natural jets ($R^2$ 0.97 each), move in opposite
directions under these edits.

## Untrained control (h4/h5 read-outs carry no mass information, $R^2$ ≈ 0; calls only)
- Calls on the unedited sample: g 996, W 5,051, Z 3,899, t 54 (q 0).
- M2h vs its null: W → Z 2.5 % vs 2.7 % (McNemar p 0.66); calls changed 3.3 % vs 3.7 %.
- B edits vs their nulls: W → Z 2.8–7.4 % vs 2.9–8.5 %. No consistent excess: the Holm p-values that fall below 0.05 go in both
  directions.
- The trained network's selective W → Z response to $M_2$ (B 7.5–13 % vs null 0.7–3.2 %) is absent in the untrained one.
- Count-weighted A edits do move the untrained calls (A (0, 0.5): W → Z 21.8 % vs 11.6 %; calls 24.7 % vs 13.9 %). Random
  networks respond to the particle-count-weighted profile, which the trained network does not use for W/Z.

## Hypotheses and which are excluded (5 % dose unless stated)
- **Thesis ($M_2$-like):** predicts A_a R ≈ 0 and W → Z at the null; B_a R ≈ 1 and W → Z at the dilation level.
  - Matches for all 5 $\kappa$ = 0 and all 5 $\kappa$ = 2 members:
    - A $R_{h4}$ −0.32 to +0.07 and $R_{h5}$ −0.12 to +0.02;
    - B $R_{h4}$ 0.96–1.03 and $R_{h5}$ 0.90–1.03;
    - W → Z: A 0.3–1.8 % vs null 0.7–1.7 %; B 7.7–8.4 % vs dilation 8.3 %.
  - For $\kappa$ = 1:
    - B W → Z is at or above the dilation level (7.5–13.0 %), so the call response survives holding each neighbour.
    - A W → Z is not above the null.
    - The read-out R departs from 0 / 1. A $R_{h4}$ is −0.15 to −0.55 and B $R_{h4}$ is 0.65–0.93, and h5 gives other values.
- **Alternative 1 (girth-like $\beta$ = 1 carries the response):** predicts A (1, 1) W → Z at the dilation level and B (1, 1)
  at about 0.
  - Observed: A (1, 1) gives 2.7 % vs null 3.0 % (Holm p 1); B (1, 1) gives 9.9 % vs null 2.7 % (Holm p 0.001).
  - Excluded for the call response on the 4,002 kept jets. At 2 %: A 3.3 % vs null 1.8 % (p 0.055); B 3.0 % vs 1.6 % (p 0.003).
- **Alternative 2 (a count-weighted member carries it):** predicts A (0, β) at the dilation level.
  - Observed: A gives 0.3–1.5 % W → Z and $R_{h4}$ ≤ 0.07; B gives 7.7–8.4 %.
  - Excluded for all 5 $\beta$.
- **Alternative 3 (response spread over $\beta$, no single member):** predicts intermediate A and B for the members next to $M_2$.
  - Excluded for $\kappa$ = 0 and $\kappa$ = 2.
  - Not excluded among pT-weighted members near $\beta$ = 2. Holding $\sum p_T r^3$ changes the $M_2$ response: $R_{h4}$ 0.67,
    $R_{h5}$ 1.39, W → Z 13.0 % vs 9.9 % for the dilation on the same jets. Holding $\sum p_T r^{1.5}$ lowers $R_{h4}$ to 0.65
    (h5 0.11) at 5 %, but not at 2 % ($R_{h4}$ 1.01).
  - A variable such as $\sum p_T r^{\beta}$ with $\beta$ somewhat above 2, or a mix of $\beta$ = 2 and 3, is not ruled out.

## Reading (scoped to these measurements)
- With $M_2$, $M_0$ and $\vec M_1$ fixed, radial reshuffles that change any other member of the 15-member family by 5 % do not
  move W-called jets to Z beyond a matched random reshuffle. Raising $M_2$ by 5 % with any one other member fixed moves 7.5–13 %
  of them, as the uniform dilation does (8.3 %).
- The particle-count weighting ($\kappa$ = 0) and the pT² weighting ($\kappa$ = 2) are excluded as the network's mass spread
  variable; the pT weighting ($\kappa$ = 1) is selected.
- Within $\kappa$ = 1, the exponent is located between about 1.5 and 3, with the closest test at $\beta$ = 3 (holding it changes
  the response size).
- The linear read-outs at h4 and h5 disagree under the $\kappa$ = 1 decoupling edits, by up to 1.3 in R. These edits move the
  hidden state off the natural-jet manifold where the read-outs were fitted. For $\kappa$ = 1 the call-based W → Z flow is the
  read-out-free metric.

## Issues and limits
1. The planned all-members-held design is infeasible (arm 0); the result rests on pairwise decoupling, which holds two family
   members at a time, not fourteen. Other members move under each edit: the median changes are in `edit_member_changes` in the
   json. For example, A (1, 1) changes $\sum p_T r^3$ by −10 % and pT² members by up to +15 %.
2. Selection: for $\kappa$ = 1, $\beta$ ≤ 1.5 at 5 %, only 31–40 % of jets are kept (A and B and both nulls feasible), mostly
   q/g. W-called jets number 125–263 there, so the W → Z CIs are wide (e.g. B (1, 1.5): 9.6 % [4.8, 14.9]). The 2 % dose
   keeps 59–90 %.
3. The edits are per-particle minimum-norm solutions. They can move single particles by up to a factor 2 in $r$ (smaller at
   2 %), unlike the smooth profile of a physical change. The matched-D null controls for the size of the reshuffle, not for
   its shape.
4. R depends on the read-out layer for $\kappa$ = 1 edits (issue noted in the Reading).
5. One seed (1201) and one untrained redraw (seed 0). A verifier should rerun on a fresh seed and add $\beta$ = 2.5 to probe
   the exponent between 2 and 3.
6. Deviation from the plan: per-particle $\delta$ instead of the smooth $f(r)$ basis (see Procedure); pairwise edits instead of
   holding all 14 (arm 0); an added 2 % dose for $\kappa$ = 1. Held-out read-out uses half B of the jets (≈ 5,000; fewer per
   member after the keep masks).

## Explainer entry (site/explain.py, Paper A tab)
- **Question:** JEDI-linear's mass variable behaves like $\sqrt{M_0M_2}$ with $M_2=\sum_i p_{T,i}r_i^2$. Earlier tests compared
  $M_2$ with only three alternatives. Is it $M_2$ rather than another angular spread
  $\lambda(\kappa,\beta)=\sum_i p_{T,i}^\kappa r_i^\beta$ with a different pT weighting $\kappa$ or power of angle $\beta$?
- **Background:** for massless particles,
  $m^2 \approx \sum_{i<j}p_{T,i}p_{T,j}\Delta R_{ij}^2 = M_0M_2-|\vec M_1|^2$, so $M_2$ is the angularity that enters the mass.
  Girth ($\beta$ = 1) and multiplicity-type ($\kappa$ = 0) angularities are standard quark/gluon variables. On ordinary jets
  all members are strongly correlated, so only edits that change one while holding another can tell them apart.
- **Method:** multiply each particle's distance to the axis by $e^{\delta_i}$. pT, direction around the axis, pT rank,
  $M_0$ and $\vec M_1$ stay fixed.
  - Edit A: one member +5 % with $M_2$ fixed. Edit B: $M_2$ +5 % with that member fixed.
  - Measure $R=\overline{\Delta\log\hat m}/\overline{\Delta\log\hat m}_{\text{dilation}}$ and the share of W calls that become Z.
  - Compare each edit with a random radial reshuffle of the same size that holds the same quantities.
- **Result** (figure `results/jedi/e27_angularity.png`):
  - B edits send 7.5–13 % of W-called jets to Z, against 8.3 % for a true 5 % dilation and ≤ 3.2 % for the nulls.
  - A edits send 0.3–2.7 %, not above their nulls.
  - For count-weighted and pT²-weighted members, R ≈ 0 for A and ≈ 1 for B.
- **Checks:**
  - Matched-size null reshuffles (|R| ≤ 0.1 at h4).
  - Untrained network: no M2-specific W → Z, 2.5 % vs 2.7 %.
  - 2 % dose for the pT-weighted neighbours.
  - Holm over 14 members.
  - Holding all 14 members at once is impossible (needs factor $e^{5}$–$e^{1700}$ changes in $r$).
- **Why it matters:** the network's mass input is the pT-weighted, squared-angle spread that appears in the invariant-mass
  formula, not a count-weighted or girth-like width. Which power of $r$ near 2 it uses (2 vs 3) is not resolved.

## Glossary entries (explain.GLOSSARY)
- ('angularity λ(κ, β)', '$\\lambda(\\kappa,\\beta)=\\sum_i p_{T,i}^{\\kappa}\\, r_i^{\\beta}$, a family of jet-shape sums;
  $r_i$ is the particle\'s distance to the jet axis. $\\lambda(1,2)=M_2$, $\\lambda(1,1)$ is girth (×M0), $\\lambda(0,0)$ the
  particle count')
- ('κ (angularity)', 'the power of pT in an angularity: 0 counts particles equally, 1 weights by pT, 2 by pT²; higher κ
  emphasises the hardest particles')
- ('β (angularity)', 'the power of the distance r to the axis in an angularity: low β weights particles near the core,
  high β the outskirts; β = 2 is the power that enters the invariant mass')
- ('radial edit', 'an input edit that multiplies each particle\'s distance to the jet axis by a factor and keeps its pT and its
  direction around the axis (so pT rank, M0 and particle count are unchanged)')
- ('decoupling edit (E27)', 'a radial edit chosen per jet so that one quantity changes by a set amount while named others are
  held fixed, e.g. M2 +5 % with Σ pT r fixed')
