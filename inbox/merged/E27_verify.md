# E27 verification: angularity-family decoupling edits on JEDI-linear, fresh sample, with a logit-margin metric, the true-mass change and a new member (κ 1, β 2.5) (verifier report 2026-10-21)

## Conclusion
On a new set of 10,000 jets, moving particles toward or away from the jet axis so that a particle-count-weighted or
pT²-weighted spread changes while $M_2=\sum p_T r^2$ stays fixed again sends no more W-called jets to Z than a random move of the
same size, and raising $M_2$ with one of those spreads held again sends about as many as a real mass change (8–9 %). This repeats
E27's result that the network's mass response is pT-weighted. Among the pT-weighted spreads the network does not respond to $M_2$
alone: at fixed $M_2$, changing $\sum p_T r^{\beta}$ for β between 0.5 and 2.5 moves the network's W-versus-Z output about as much
as a real mass change does (in either direction depending on β), and at a 2 % dose the low-β edits send as many W jets to Z as a real
mass change. Which pT-weighted radial combination the network uses, and whether the girth-like $\sum p_T r$ takes part, remains open.

## Files
- Script: `interp/e27_verify.py` (imports E27's solver, null edits and kept-jet rules from `interp/e27_angularity_jedi.py`;
  adds the member (κ 1, β 2.5) to the constraint list).
- Job: `./job.sh e27-verify 'python -m interp.e27_verify --seed 2207'` (1 CPU / 2 GiB; wall 1,180 s; pod peak 1,895 MiB;
  0.80 CPU on average).
- Results: `$AI_RESULTS/jedi/e27_verify.json` (local copy `results/jedi/e27_verify.json`); log `logs/e27-verify.log`.

## Code review of `interp/e27_angularity_jedi.py` and `interp/e27_plot.py`
- Stage, masks, quantiser: the script uses `Jedi.run` unchanged (stages h4, h5, logits). Padded slots are excluded from the
  edit and from all sums by `real = pT > 0`; edits multiply $(\Delta\eta,\Delta\phi)$ of real particles only, so pT, pT rank,
  input slot, $M_0$ and the particle count are unchanged.
- Read-out split: the ridge read-out of $\log m$ is fitted (standardisation, α selection and weights) on half A of the jets and
  evaluated on half B only. No leakage found.
- The edit does what is stated: in this run the true jet-mass change of every A edit is ≤ 0.2 % of the dilation's
  (`R_true` −0.002 to +0.002) and every B edit's is 0.998–1.002 of it.
- Scope point (not a bug): all edits hold $M_0$ and $\vec M_1$, so $m^2\approx M_0M_2-|\vec M_1|^2$ moves exactly with $M_2$
  (R_true above). These edits therefore cannot tell "$M_2$" from "the jet mass $m$" apart; they compare $M_2$ (equivalently $m$)
  with the other 14 angularities.
- The matched null is a per-particle random (uncorrelated) radial jitter in the null space of the held rows, matched in
  displacement size D. It controls the size of a reshuffle, not its shape; the A edits are smooth in r. No coherent-shape null
  was run.
- The W → Z flow counts only moves toward Z. An edit that moves the W/Z output toward W (lower mass) scores as "no response".
  E27 had no signed, read-out-free metric; this verification adds one (Δ margin below).
- Figure `results/jedi/e27_angularity.png`: values match the json. The h4 heat map for edit A shows (κ 2, β 0.5) at −0.32
  [−0.36, −0.28], which the reading summarises as "R about 0". The suptitle states a result in words, not a measurement.

## Procedure
- **Sample:** `interp.jedi_core.sample(per_class=2000, seed=2207)`, N = 10,000 jets, 2,000 per class (E27 used seed 1201).
  Unedited accuracy 0.8039. Read-out held-out $R^2$ of $\log m$: h4 0.973, h5 0.972.
- **Network:** official 3-feature n64 JEDI-linear; control: the untrained redraw `Jedi(untrained=True, seed=0)`, calls taken
  as argmax(logits − mean unedited logits).
- **Edits (E27's own code):** each real particle's $(\Delta\eta,\Delta\phi)$ about the stored axis is multiplied by $e^{\delta_i}$;
  δ is the per-particle minimum-norm Gauss–Newton solution of constraints on $\log\lambda(\kappa,\beta)$ with
  $\lambda(\kappa,\beta)=\sum_i p_{T,i}^{\kappa}r_i^{\beta}$, and $\vec M_1=\sum_i p_{T,i}(\Delta\eta_i,\Delta\phi_i)$ held.
  - A_a: member a × (1 + dose), $M_2$ and $\vec M_1$ held. B_a: $M_2$ × (1 + dose), member a and $\vec M_1$ held.
  - Null nA_a / nB_a: random radial jitter holding member a, $M_2$ and $\vec M_1$, with the same displacement size
    $D=\sqrt{\sum_i(\Delta r_i)^2}$ as A_a / B_a on the same jet (within 5 %).
  - Dilation (the reference for a real mass change): $r\to\sqrt{1+\text{dose}}\,r$ for every particle ($M_2$ × (1 + dose),
    $M_0$ fixed).
  - Jets kept for member a: A_a, B_a and both nulls all meet the tolerance (0.5 % at the 5 % dose, 0.2 % at 2 %) and
    max |δ| ≤ log 2.
- **Members:** 5 % dose: (0, 1), (0, 2), (1, 0.5), (1, 1), (1, 1.5), (1, 2.5) [new], (1, 3), (2, 0.5), (2, 2).
  2 % dose: the five κ = 1 members.
- **Metrics** (on member a's kept jets; W-called = called W by the network before the edit):
  - W → Z flow: share of W-called jets called Z after the edit.
  - **F** = (flow(edit) − flow(null)) / (flow(dilation) − flow(null)) on the same jets: 0 = like a random reshuffle,
    1 = like a real mass change of the same size.
  - **R_marg** = mean Δ($\text{logit}_Z-\text{logit}_W$) under the edit / the same under the dilation, on the same W-called jets.
    A signed, read-out-free measure of how far the W-versus-Z output moves, in units of a real mass change. 1 = moves toward Z
    as the dilation does; −1 = moves toward W by the same amount.
  - R_h4, R_h5: E27's read-out ratio (mean Δ log m̂ edit / dilation) on held-out jets.
  - **R_true** = mean Δ log m (true jet mass from the edited four-vectors) / the same for the dilation.
- **Statistics:** 95 % CIs from 1,000 Poisson bootstrap resamples over jets (ratios recomputed per resample); McNemar exact test
  for W → Z of edit vs its null and of edit vs the dilation (paired, same jets); sign-flip permutation (10,000) for the per-jet
  margin difference edit − null; Holm over the members of each dose (9 at 5 %, 5 at 2 %) per edit type and test.
- **Hypotheses.** H0 (per edit): the response equals the matched null's. Thesis "$M_2$ is the radial input": A gives F ≈ 0 and
  R_marg ≈ 0; B gives F ≈ 1 and R_marg ≈ 1. Alternative "member a carries it": A F ≈ 1, B F ≈ 0. Alternative "a mix of
  pT-weighted members": intermediate or sign-changed A and B for the κ = 1 neighbours.

## Results (trained network)

References: dilation 5 %: W → Z 8.0 % [6.9, 9.2] of 2,032 W-called jets. $M_2$-only edit (M2h, 5 %): W → Z 8.6 % vs null 1.5 %
(McNemar 146 vs 1, p 2e-42), R_marg 0.98 [0.95, 1.01], R_true 1.001. At 2 %: dilation 3.3 %, M2h 3.0 % vs null 1.1 % (p 7e-11),
R_marg 0.99 [0.94, 1.05].

κ = 0 and κ = 2 members, 5 % dose (n 9,013–9,995; 1,642–2,032 W-called):

| member | A: W → Z (null; dilation) | A: F [CI] | A vs dilation, Holm p | A: R_marg [CI] (null) | B: W → Z (null) | B: F [CI] | B: R_marg [CI] |
|---|---|---|---|---|---|---|---|
| (0, 1) | 1.4 % (1.8; 8.0) | −0.06 [−0.16, 0.04] | 1e-36 | +0.29 [0.25, 0.33] (0.10) | 8.9 % (0.9) | 1.13 [1.02, 1.27] | 0.96 [0.92, 0.99] |
| (0, 2) | 1.0 % (1.5; 8.0) | −0.08 [−0.16, 0.00] | 1e-41 | +0.10 [0.08, 0.12] (0.02) | 8.6 % (1.1) | 1.09 [0.97, 1.23] | 0.98 [0.95, 1.01] |
| (2, 0.5) | 1.9 % (1.8; 8.4) | 0.03 [−0.07, 0.13] | 4e-27 | −0.23 [−0.29, −0.18] (0.05) | 8.8 % (1.2) | 1.05 [0.93, 1.19] | 1.08 [1.04, 1.11] |
| (2, 2) | 1.3 % (1.3; 8.0) | 0.00 [−0.08, 0.06] | 4e-40 | −0.05 [−0.08, −0.03] (0.02) | 8.6 % (1.2) | 1.09 [0.97, 1.22] | 1.13 [1.09, 1.17] |

- A vs null (W → Z), Holm p ≥ 0.58; B vs null Holm p < 1e-20 for all four.
- R_h4 for A: 0.03, 0.01, −0.36 [−0.40, −0.32], −0.09 (E27: 0.07, 0.02, −0.32, −0.07).

κ = 1 members (W-called jets per member in brackets):

| dose, member | A: W → Z (null; dilation) | A: F [CI] | A vs dilation, Holm p | A: R_marg [CI] (null) | B: W → Z (null) | B: F [CI] | B: R_marg [CI] (null) | A: R_h4 / R_h5 |
|---|---|---|---|---|---|---|---|---|
| 5 %, (1, 0.5) [133] | 3.0 % (2.3; 6.0) | 0.20 [−1.3, 1.5] | 0.34 | −0.98 [−1.52, −0.61] (0.28) | 9.0 % (2.3) | 1.8 [0.6, 8.0] | 1.11 [0.93, 1.35] (0.13) | −0.18 / +0.65 |
| 5 %, (1, 1) [268] | 2.2 % (3.4; 8.6) | −0.21 [−1.4, 0.17] | 0.001 | −0.97 [−1.38, −0.67] (0.11) | 10.8 % (2.2) | 1.35 [0.87, 2.27] | 1.32 [1.16, 1.52] (0.10) | −0.26 / +0.53 |
| 5 %, (1, 1.5) [139] | 1.4 % (1.4; 7.2) | 0.00 [−0.50, 0.56] | 0.077 | −1.58 [−2.27, −1.11] (0.09) | 12.2 % (2.9) | 2.2 [1.0, 6.7] | 1.78 [1.46, 2.26] (0.29) | −0.47 / +0.80 |
| 5 %, (1, 2.5) [243] | 3.3 % (2.1; 11.1) | 0.14 [−0.13, 0.43] | 0.001 | **+0.96 [0.75, 1.22]** (0.13) | 16.0 % (3.7) | 1.67 [1.13, 2.80] | **0.27 [−0.18, 0.67]** (0.03) | −0.25 / −0.89 |
| 5 %, (1, 3) [899] | 1.9 % (1.7; 9.1) | 0.03 [−0.13, 0.15] | 5e-18 | +0.35 [0.29, 0.41] (0.04) | 14.3 % (2.9) | 1.84 [1.43, 2.41] | 0.88 [0.75, 0.99] (0.08) | −0.02 / −0.32 |
| 2 %, (1, 0.5) [785] | 4.6 % (3.3; 3.9) | 2.0 [−6, 10] | 1.0 | −1.27 [−1.85, −0.86] (0.33) | 4.6 % (1.8) | 1.29 [0.83, 2.25] | 1.39 [1.24, 1.60] (0.10) | −0.32 / +0.64 |
| 2 %, (1, 1) [1,048] | 4.4 % (3.3; 4.4) | 1.0 [−0.15, 4.0] | 1.0 | −0.94 [−1.29, −0.64] (0.10) | 4.7 % (2.6) | 1.16 [0.67, 2.00] | 1.45 [1.29, 1.65] (0.11) | −0.50 / +0.46 |
| 2 %, (1, 1.5) [756] | 5.2 % (3.0; 4.4) | 1.6 [0.57, 6.0] | 1.0 | −1.60 [−2.22, −1.13] (0.12) | 4.9 % (2.6) | 1.31 [0.65, 3.15] | 1.98 [1.68, 2.37] (0.35) | −0.63 / +0.73 |
| 2 %, (1, 2.5) [1,140] | 2.0 % (2.2; 3.8) | −0.11 [−1.1, 0.41] | 0.002 | **+1.19 [0.97, 1.43]** (0.24) | 7.1 % (2.4) | 3.4 [2.1, 7.2] | **0.16 [−0.17, 0.44]** (0.23) | −0.01 / −0.65 |
| 2 %, (1, 3) [1,752] | 1.0 % (1.3; 3.6) | −0.12 [−0.40, 0.07] | 4e-11 | +0.45 [0.36, 0.56] (0.09) | 5.1 % (1.6) | 1.74 [1.32, 2.43] | 0.73 [0.59, 0.86] (0.16) | +0.09 / −0.23 |

- A vs null (W → Z), Holm p: 5 %: 1.0 for all five; 2 %: 0.40, 0.40, 0.012, 0.85, 0.77.
- Margin edit − null, sign-flip permutation, Holm p ≤ 0.001 for every κ = 1 A edit at both doses (B (1, 2.5): 0.28 at 5 %,
  0.66 at 2 %).
- Kept jets at 5 % for κ = 1, β ≤ 2.5: 3,141–4,014 (W-called 133–268), as in E27 (3,143–4,002); at 2 %: 5,917–7,004.

Untrained control: M2h W → Z 2.7 % vs null 3.3 % (McNemar p 0.04, toward the null); B edits vs nulls 2.5–8.1 % vs 3.2–9.7 %;
13 of 14 at or below their null, one above it ((2, 2) at 5 %: 6.1 vs 5.1 %, raw McNemar p 0.01). The selective W → Z response to $M_2$ is absent in the untrained network, as in E27. Its margin ratios
(M2h 0.47 vs null 0.63) show that the untrained margin moves under any edit, so R_marg is interpreted only for the trained net.

## Comparison with E27's reading (what reproduces, what does not)
1. κ = 0 and κ = 2: reproduced. A edits give F −0.08 to 0.03 and differ from the dilation (Holm p ≤ 4e-27); B edits give
   F 1.05–1.13. The margin agrees (A R_marg −0.23 to +0.29; B 0.96–1.13). The h4 read-out for A (2, 0.5) is −0.36, so
   "R about 0 for edit A" holds for 3 of the 4 tested members, not (2, 0.5).
2. κ = 1, call level at 5 %: reproduced on the q/g-dominated kept jets: A flows at the null; A (1, 1) differs from the dilation
   (Holm p 0.001), A (1, 0.5) and A (1, 1.5) do not (p 0.34, 0.077; 133–139 W-called jets).
3. κ = 1, call level at 2 % (the dose with 59–90 % of jets kept): A edits for β ≤ 1.5 give W → Z flows equal to or above the
   dilation's on the same jets (F 1.0–2.0; A vs dilation Holm p 1.0; A (1, 1.5) above its null, Holm p 0.012). E27's own 2 % run
   shows the same (3.1–4.2 % vs dilation 3.5–4.0 %). The statement "changing any other spread with M2 held does not [reproduce the
   response]" and the exclusion of the girth-like (1, 1) member do not hold at this dose.
4. κ = 1, signed margin (not measured in E27): every κ = 1 A edit moves the W − Z logit margin by 0.35–1.6 dilation units
   (Holm p ≤ 0.001 vs null), toward W for β ≤ 1.5 and toward Z for β = 2.5 and 3. For the new member (1, 2.5), raising it with
   $M_2$ held moves the margin as much as a real mass change (R_marg 0.96 and 1.19), while raising $M_2$ with it held moves the
   margin by 0.27 and 0.16 (CIs include 0), though the W → Z flow then exceeds the dilation's (F 1.7, 3.4). The margin and the
   call flow follow different combinations of the pT-weighted moments; h4 and h5 read-outs disagree with each other for all
   κ = 1 edits, as in E27.
5. The true-mass ratio R_true is 0.00 for A and 1.00 for B: under these edits $M_2$ and the jet mass $m$ move together, so the
   result is a comparison of $m$ (or $M_2$) against the other angularities, not of $M_2$ against $m$.

## Corrected reading (scoped to these measurements)
With $M_0$, $\vec M_1$ and $M_2$ held, radial edits that change particle-count-weighted or pT²-weighted angularities by 5 % move
W-called jets to Z at the rate of a matched random reshuffle (F −0.08 to 0.03 on seed 2207; E27 seed 1201 agrees), and raising
$M_2$ with one of them held moves them at the dilation's rate (F 1.05–1.13). The network's mass response is therefore carried by
a pT-weighted (κ = 1) radial quantity. Within κ = 1 the response is not a function of $M_2$ alone: with $M_2$ held, changing
$\sum p_T r^{\beta}$ by 2–5 % moves the W − Z logit margin by 0.35–1.6 times a real mass change (toward W for β ≤ 1.5, toward
Z for β ≥ 2.5), and at the 2 % dose the β ≤ 1.5 edits produce W → Z flows equal to the dilation's. The pT-weighted radial
combination the network uses, and whether $\sum p_T r$ enters it, are not determined by this design.

## Issues
1. One verification seed (2207) and one untrained redraw; 1,000 bootstrap resamples and 10,000 permutations (E27: 2,000 / 20,000).
2. At 5 % the κ = 1, β ≤ 2.5 kept sets are 31–40 % of jets, mostly q/g (133–268 W-called), so F CIs there are wide.
3. R_marg is a mean over W-called jets, dominated by jets far from the W/Z boundary; W → Z flow counts only jets near the
   boundary. The two disagree for (1, 2.5) and are reported separately; no per-mass-bin breakdown was run.
4. No coherent-shape null (a smooth radial profile change at fixed $M_2$ drawn at random) was run; the null is per-particle jitter.
5. Outputs written only to `$AI_RESULTS/jedi/e27_verify.json` and this report; nothing committed; site files not edited.

## Glossary additions (for explain.GLOSSARY)
- **F (flow fraction):** (W → Z flow of an edit − flow of its matched null) / (flow of the uniform dilation − flow of the null),
  on the same jets. 0 = the edit acts like a random reshuffle of the same size; 1 = it acts like a real mass change.
- **R_marg:** mean change of $\text{logit}_Z-\text{logit}_W$ on W-called jets under an edit, divided by the same under the uniform
  dilation $r\to\sqrt{1+\text{dose}}\,r$. A signed, read-out-free response in units of a real mass change.
- **R_true:** mean change of the true jet log-mass (from the edited particle four-vectors) under an edit, divided by the dilation's.
- **Angularity λ(κ, β):** $\sum_i p_{T,i}^{\kappa} r_i^{\beta}$ over the jet's particles, $r_i$ the distance to the stored axis.

## Explainer entry (for site/explain.py)
- **Question.** E27 concluded that JEDI-linear's mass input is $M_2=\sum p_T r^2$ rather than any other angularity
  $\lambda(\kappa,\beta)=\sum p_T^{\kappa}r^{\beta}$. Does that hold on new jets, with a signed output metric and one more neighbour
  (β = 2.5)?
- **Background.** With $M_0=\sum p_T$ and $\vec M_1$ fixed, $m^2\approx M_0M_2-|\vec M_1|^2$, so changing $M_2$ changes the jet
  mass. The angularities are almost linearly dependent, so they can only be separated two at a time.
- **Method.** Radial edits $r_i\to e^{\delta_i}r_i$ (pT and azimuth fixed) that raise one angularity by 2 % or 5 % while holding
  $M_2$ (edit A), or raise $M_2$ while holding it (edit B); random reshuffles of equal size as nulls; the uniform dilation
  $r\to\sqrt{1.05}\,r$ as the real mass change. Metrics: $F=\frac{f_\text{edit}-f_\text{null}}{f_\text{dil}-f_\text{null}}$ for the
  W → Z flow f, and $R_\text{marg}=\overline{\Delta(\ell_Z-\ell_W)}_\text{edit}/\overline{\Delta(\ell_Z-\ell_W)}_\text{dil}$.
- **Result.** κ = 0, 2: A F ≈ 0, B F ≈ 1 (pT weighting selected). κ = 1: with $M_2$ held, $|R_\text{marg}|$ 0.35–1.6 (sign
  changes between β = 1.5 and 2.5); at 2 % the β ≤ 1.5 A edits give F 1.0–2.0. Figure to make: R_marg and F per member for
  A and B (data in `results/jedi/e27_verify.json`).
- **Checks.** True mass moves only under B (R_true 1.00 vs 0.00); untrained network shows no selective W → Z response.
- **Why it matters.** It fixes the pT weighting of the mass variable and shows that among pT-weighted radial moments the network's
  W/Z decision is not a function of $M_2$ alone.
