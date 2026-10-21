# E26: where JEDI-linear's response to an exact boost runs (agent report, 2026-10-20; claims A1, A3, A4, A9; exploratory, links to D9 / H12)

## Conclusion
When a jet is boosted along its own axis, its mass stays the same but its pT changes and its constituents spread out or
close in. At $\lambda = 0.9$ (pT lowered by 10 %) about a third of JEDI-linear's calls change. Holding the network's
summed-pT object S0 at its pre-boost value removes about half of those changes. The network's mass read-out at h4 does not
move under the boost. So the response comes from ΣpT (M0) and the angular spread (M2) entering the decision separately,
not from a shifted mass value. The predicted route through the soft-particle pT ramps and the 11 GeV threshold is excluded.
Still open: at $\lambda > 1$ no single second-round object carries the response (only a few h4/h5 units do), and why the
classes depend on ΣpT at fixed mass is not tested here.

## Files
- Script `interp/e26_boost_route.py`. Main run: `./job.sh e26-run 1 3 'python -m interp.e26_boost_route --seed 2601'`
  (wall 960 s, peak 1719 MiB, 0.99 CPU). Supplementary null run: `./job.sh e26-null1 1 2 '... --null1 8000'` (622 s).
- Figure script `interp/e26_plot.py`.
- Results: `results/jedi/e26_boost_route.json`, `e26_boost_route_null1.json`, `e26_boost_route_perjet.npz`, `e26_boost_route.png`.
- Logs: `logs/e26-run.log`, `logs/e26-null1.log`.
- The 2 GiB request was raised to 3 GiB because the smoke run peaked at 2047 MiB on a 2 GiB request.
- `job.sh`: node gpu-09.nrp.mghpcc.org was added to the excluded-node list. The first smoke pod sat in D state
  (`ceph_mdsc_wait_request`) for 10 min while opening the model file.

## Procedure
- **Sample:** `interp.jedi_core.sample(per_class=2000, seed=2601)`, 10,000 jets. Seed 2602 is kept for the verifier.
- **Network:** official 3-feature n64, unedited accuracy 0.8068 (expected 0.806–0.813). The table round trip
  (`h1_irc_panel.hls_table` → `h12.sortT` → `hls_prep`) reproduces the raw-input logits exactly (max |Δlogit| 0). The exact
  fixed-point tail from S/h4/h5 (`x10_das_jedi.JNet.exact`) reproduces `run_jetwise` with max |Δlogit| 0.
- **Edits** (imported from `interp.h12_boost_panel`, not re-implemented), with $\lambda \in \{0.8, 0.9, 1.1, 1.25\}$:
  - B: exact boost along the jet axis; |m change| ≤ 1.1e-11 relative; ΣpT × 0.995λ; $m_A=\sqrt{M_0M_2}$ changes by ≤ 4e-5.
  - P: pT-only scaling.
  - A: angle-only scaling.
  - Control R: rotation with the same pT-weighted displacement as B.
  - Control T: random-direction translation of the same length.
- **Edits added here:**
  - M0-only: pT × c, angles × $c^{-1/2}$.
  - M2-only: angles × $\sqrt{c}$.
  - Bc: B with every particle whose pT crosses the a32 edge clamped 0.01 % on its unboosted side. The edge computed from the
    fused l1 weights is 10.97 GeV; the empirical edge is 10.98 GeV. Bc changes m by ≤ 2.4 % (max over jets, λ 0.9).
  - D(k*): angle dilation, used as the matched calibration.
- **Holds:** the network sees the boosted jet, and one unit or set is reset to its value on the unboosted jet.
  - l1 (per particle) holds: the cut's value on the same particle before the boost, placed in that particle's boosted pT-rank
    slot. These run a full forward pass through `run_jetwise`.
  - g holds: full forward pass.
  - S, h4, h5 holds: the exact tail from that stage.
  - Live units: l1 13 (pT ramps a15, a21, a32, a56; angular a3, 6, 29, 30, 36, 40, 43, 46, 57); g 11; S 19; h4 55; h5 32.
  - Ramp edges: a15 is on for all pT (linear in pT); a21 is on below 93.2 GeV; a32 below 10.97 GeV; a56 below 31.0 GeV.
  - X10 DAS directions: M0 and M2 at S and h4 (k = 1, 2), refit here with X10's exact data, pairs and seeds.
    Validation balanced agreement: S M0 0.90, S M2 0.70–0.76, h4 0.80–0.89.
- **Metric:** fraction of B's call changes removed, $1 - N_\text{changed}(\text{hold})/N_\text{changed}(B)$.
  - Negative values mean the hold adds call changes.
  - Paired McNemar test, B vs hold.
  - Fraction of the mean W-jet shift removed, for $l_W - l_Z$ and for $\log p_W/(p_g+p_q)$.
  - 95 % intervals: Poisson bootstrap over jets (1,000 draws, the same weights for every hold).
- **Nulls:**
  - Random live-S subsets, size 1–5 (500 draws per size, every λ). Sizes 1–3 were also fully enumerated (19, 171 and
    969 subsets) at λ 0.9 and 1.1.
  - Random orthonormal S directions, dims 1–5 (500 draws in the main run). The supplement used 8,000 draws at dim 1 and
    2,000 at dims 2 and 4.
  - Random h4 directions, dims 2 and 4.
  - 20 random 4-unit l1 subsets. The same draws plus 2 random S units form the "mixed" null for the 6-object thesis set.
  - $p_\text{rand} = (1 + \#\{\text{null} \ge \text{obs}\})/(1 + \#\text{null})$. Holm correction over the 130 single-unit
    holds.
- **Step 2 read-outs:** standardised ridge regressions of $\log m$, one on h4 (E15) and one on z (X10).
  - Fit on half A of the jets (unedited); evaluated on half B. Held-out R²: h4 0.972, z 0.938.
  - Slopes through the origin over the 4 values of λ: $s_E = \sum_\lambda \log\lambda\,\overline{\Delta\log\hat m}_E / \sum_\lambda \log^2\lambda$.
- **Untrained control:** n64 redraws, seeds 0–2. Logits are centred on their unedited mean (as in X10).

## Results

### Boost and controls
B changes the calls; R and T change few. McNemar B vs R: p < 1e-270 at every λ.

| λ | B calls changed [95 % CI] | R | T | P | A | in window / out (n out) |
|---|---|---|---|---|---|---|
| 0.8 | 48.3 % [47.3, 49.2] | 3.2 % | 5.1 % | 49.7 % | 45.6 % | 49.6 % / 12.4 % (362) |
| 0.9 | 32.0 % [31.1, 32.9] | 2.7 % | 2.6 % | 33.7 % | 28.1 % | 32.3 % / 8.4 % (131) |
| 1.1 | 14.6 % [13.9, 15.3] | 2.4 % | 2.3 % | 30.8 % | 24.1 % | 14.8 % / 3.4 % (174) |
| 1.25 | 25.1 % [24.2, 25.9] | 2.9 % | 4.1 % | 60.8 % | 46.5 % | 26.0 % / 5.5 % (434) |

Class flow under B:
- λ 0.9: W → t 1,167 of 2,034 W calls; Z → t 788 of 1,874 Z calls.
- λ 1.1: q → g 476; t → g 359.
- λ 1.25: t → g 871; q → g 821.
- Mean shift on true-W jets at λ 0.9: $\Delta(l_W - l_Z) = -2.48$, $\Delta\log p_W/p_{QCD} = -2.80$.

### Step 1: holds (removed share of B's call changes, 95 % bootstrap CI)

| hold | λ 0.9 | λ 1.1 | λ 0.8 | λ 1.25 |
|---|---|---|---|---|
| all 13 live l1 cuts (sanity) | 99.0 % [98.6, 99.3] | 98.6 % | – | – |
| whole S / h4 / h5 stage | 100 % | 100 % | 100 % | 100 % |
| DAS M0+M2 at S, 2 dims (k1 each) | 83.2 % [81.9, 84.7] | 55.3 % [51.0, 58.8] | 79.1 % | 46.0 % |
| DAS M0+M2 at S, 4 dims (k2 each) | 84.8 % [83.5, 86.1] | 64.9 % [61.8, 68.0] | 82.2 % | 60.7 % |
| DAS M0 at S, 1 dim | 60.2 % [58.2, 62.2] | 3.6 % [−2.4, 9.4] | 29.3 % | −26.7 % |
| DAS M2 at S, 1 dim | 16.5 % [14.4, 18.4] | −43.2 % [−49.4, −37.4] | 10.7 % | −99.0 % |
| S0 (ΣpT of the 14 hardest) | 55.1 % [53.1, 57.2] | −0.3 % [−6.6, 5.5] | 22.3 % | −33.0 % |
| g49 (context unit fed by ΣpT) | 50.5 % [48.6, 52.4] | 7.8 % [3.1, 12.2] | – | – |
| g (all units) | 55.0 % [53.1, 57.0] | 12.7 % [8.5, 17.2] | – | – |
| a15 (linear-pT cut) | 39.9 % [37.8, 42.2] | −26.0 % [−33.1, −19.4] | – | – |
| 4 pT-ramp cuts | 12.5 % [9.9, 15.1] | −64.0 % [−73.4, −55.3] | – | – |
| ramps + S0 + S19 (thesis set) | 12.5 % [9.9, 15.1] | −64.0 % | – | – |
| ramps + S0 + S19 + g | 18.2 % [15.7, 20.6] | −56.7 % | – | – |
| 9 angular cuts | −5.2 % [−8.0, −2.5] | −109.8 % | – | – |
| a32 (pT < 11 GeV cut) | −2.0 % [−3.0, −1.0] | 0.1 % [−1.9, 2.1] | – | – |
| Bc: 11 GeV crossings clamped | −0.2 % [−0.6, 0.3], McNemar p 0.46 | 0.2 % [−0.3, 0.7], p 0.58 | 0.0 % | 0.6 % (p 0.049) |
| S19 (soft multiplicity) | 1.3 % [0.7, 1.9] | 2.7 % [1.5, 3.8] | −0.2 % | 4.5 % |
| S0 + S19 | 56.1 % [54.0, 58.2] | 0.7 % [−5.3, 6.5] | 23.3 % | −32.0 % |
| best S pair (enumerated): S0 + S29 | 78.2 % | 50.4 % | – | – |

How to read the table:
- Under B, 65.6 % of jets have at least one particle crossing 11 GeV (1.19 crossings per jet at λ 0.9).
- "ramps + S0 + S19" gives the same calls as "ramps" at both λ. Holding the four pT-ramp cuts therefore already leaves S0
  and S19 at values that do not change any call. This was inferred from identical counts and W-logit shifts; S was not
  compared directly.
- With the ramps held, the network sees only the angular part of the boost. That part changes calls on its own (A edit:
  28 % / 24 %), so this hold removes little at λ 0.9 and adds changes at λ 1.1.
- S0 hold, λ 0.9: removes 54 % [52, 56] of the W-vs-QCD log-odds shift and turns 828 W calls into Z. With ΣpT fixed and the
  angles widened, M2 rises.
- DAS M0+M2 at S (2 dims), λ 0.9: removes 97 % of the W − Z shift and 95 % [94, 96] of the W-vs-QCD shift.

Size-matched nulls:
- **Random 1-dim S directions** (8,000 draws), 97.5 % quantile / max of the removed share: λ 0.9 0.220 / 0.439; λ 1.1
  0.072 / 0.186.
- **Single units passing Holm (130 tested; Holm floor 0.016):**
  - λ 0.9: S0 (p 1.2e-4, Holm 0.016), g49 (0.016), h4 53 (47.0 %, 0.016), h5 24 (43.2 %, 0.032). a15 has p 1.5e-3, Holm 0.19.
  - λ 1.1: h5 25 (30.9 %), h4 39 (25.5 %), h4 3 (23.4 %), h5 1 (18.2 %), all with Holm ≤ 0.048. No l1, g or S unit passes.
- **Random S units:** S0 is the largest of the 19 single S units at λ 0.9 (enumerated p = 1/19 = 0.053; this null cannot go lower).
- **S0 + S19 vs random S pairs:** 6.4 % of the 171 pairs remove at least as much (all of them contain S0). Against random
  2-dim S directions: p 5e-4.
- **DAS M0+M2 vs random 2- / 4-dim S directions:** null max 0.43 / 0.48 at λ 0.9 and 0.20 / 0.19 at λ 1.1; p 5e-4 at both λ.
- **Thesis set and pT ramps vs random l1 holds:** pT ramps 12.5 % vs random 4-cut l1 holds (median 13 %, 97.5 % 73 %),
  p_rand 0.62. Thesis set vs the mixed 6-object null: p 0.71.
- **McNemar (hold vs B, removal ≠ 0), Holm:** 54 / 130 single units at λ 0.9; 152 / 472 over all λ.

### Step 2: mass direction
Model: $\log\hat m = a\log M_0 + b\log M_2$, so $s_{M0}=a$, $s_{M2}=b$, $s_P=a+b$, $s_A=-2b$ and $s_B=a-b$ to first order.
The scale-free weight difference is $\kappa=(a-b)/(a+b)$. Equal weights predict $\kappa = 0$ and $s_B = 0$.

| read-out | $s_B$ | $s_P$ | $s_A$ | $a$ ($s_{M0}$) | $b$ ($s_{M2}$) | κ from B/P | κ from M0/M2 | κ from P, A | $s_B-(s_P+s_A)$ |
|---|---|---|---|---|---|---|---|---|---|
| h4 | 0.003 [−0.007, 0.012] | 0.951 | −0.971 | 0.476 | 0.491 | 0.003 [−0.007, 0.012] | −0.016 [−0.027, −0.006] | −0.021 [−0.031, −0.011] | 0.023 [0.019, 0.026] |
| z | 0.077 [0.070, 0.085] | 0.882 | −0.864 | 0.488 | 0.434 | 0.088 [0.080, 0.096] | 0.059 [0.050, 0.066] | 0.020 [0.011, 0.028] | 0.060 [0.055, 0.065] |

Per-λ mean $\Delta\log\hat m$ under B:
- h4: −0.008, −0.004, −0.005, −0.006 (λ 0.8 → 1.25). The P response is −0.22 to +0.20.
- z: +0.024, +0.028, +0.033, +0.065. This rise has the same sign for λ < 1 and λ > 1, so the z slope is not a pure
  $(a-b)\log\lambda$ term.

Matched calibration: the angle dilation $D(k^*)$ with $\log k^* = \Delta\log\hat m_B$.
- h4 gain 0.99. Calibrated B response −0.0036 [−0.0052, −0.0021] (λ 0.9) and −0.0045 (λ 1.1), in units of $\log m$.
- z gain 0.87–0.88. Calibrated B response +0.032 [0.029, 0.035] (λ 0.9) and +0.038 (λ 1.1).

DAS patches (only B's component along the S direction placed on the unboosted state; z read-out), λ 0.9 / 1.1:
- M0: −0.006 / +0.075.
- M2: +0.094 / −0.062.
- Both: +0.088 / +0.014.
- Full B: +0.028 / +0.033.
- The two directions move the read-out in opposite senses, and their sum differs from full B. Directions outside the DAS
  pair also act.

### Untrained control (n64 redraws, seeds 0–2; 1–2 live S units each)
- Calls changed by B at λ 0.9 (centred calls): 22.2 %, 0.4 %, 11.7 %.
- The thesis-set hold removes 0 %, 21 % (of 39 changes) and 0 %.
- Holding all cuts removes 94–97 %.

## Hypotheses
- **H0:** holding any single named unit removes no more of B's call changes than a size-matched random hold.
  - Rejected at λ 0.9 for S0, g49, h4 53 and h5 24: removed 43–55 % vs random 1-dim S directions with max 44 % and
    97.5 % quantile 22 %; Holm ≤ 0.032 over 130.
  - At λ 1.1, rejected only for four h4/h5 units (18–31 %, Holm ≤ 0.048). No l1, g or S unit is rejected there.
- **Thesis prediction** (ramps + S0 + S19 remove ≥ 60 %; M0 and M2 weighted unequally in the mass read-out, with the sign
  that makes boosted W read as Z or QCD): excluded on both parts.
  - The set removes 12.5 % [9.9, 15.1] at λ 0.9 and −64 % at λ 1.1.
  - The h4 read-out has κ = 0.003 [−0.007, 0.012]; the read-out-free moment-edit estimate is −0.016.
  - The observed λ 0.9 flow is W → t (heavier), not Z / QCD.
- **Alternative 1 (distributed; no hold of ≤ 5 units beats random):** excluded at λ 0.9 (S0 alone, Holm 0.016; the S0 + S29
  pair 78 %). At λ 1.1 it is excluded only for h4/h5 units. At S level the best single unit (2.7 %) is inside the null
  (enumerated), while the S0 + S29 pair (50 %) exceeds every random 2-dim direction (max 20 %).
- **Alternative 2 (mass-moment route at S; l1 ramps remove little):** consistent with the data.
  - M0 + M2 DAS directions at S remove 83 % (λ 0.9) and 55 % (λ 1.1); p 5e-4 vs random. The pT-ramp cuts remove 12.5 % / −64 %.
  - The route is through M0 and M2 as separate directions, not through the network's $\log m$ read-out, which stays fixed at h4.
- **Alternative 3 (11 GeV soft-particle threshold):** excluded. Clamping the crossings removes −0.2 % [−0.6, 0.3]
  (McNemar p 0.46), and the a32 hold removes −2.0 %.

## Reading (scoped to these measurements)
- **λ 0.9:** the boost changes calls mainly through the ΣpT chain a15 → P → g49 → S0. Holding S0 removes 55 % of the
  changes; holding S0 and S29, an M2-like ΣpT·r² object, removes 78 %. Holding the M0 + M2 DAS directions at S removes 83 %.
- **Mass read-out:** the $\log m$ read-out at h4 is unchanged by B (|Δ| ≤ 0.008), and its M0 and M2 weights are equal
  within 2 %. The network's mass variable is boost-invariant to first order in this sense. The call change comes from M0
  (and M2) being used by the class decision in addition to their product.
- **λ > 1:** the responses of the M0 and M2 directions largely cancel. The remaining changes (mostly to gluon) pass through
  h4 3, h4 39, h5 1 and h5 25 rather than through a single S object.

## Issues and limits
1. The removed share is a net count: a hold can remove some changes and create others (for example, the S0 hold at λ 0.9
   creates W → Z). The McNemar discordant counts and the flow matrices are in the json.
2. l1 holds put each unboosted particle in its boosted rank slot. The per-slot input quantizer then differs from the
   unboosted run, which is why holding all live cuts removes 98.6–99.0 % rather than 100 %.
3. The DAS directions are refit (X10's subspaces were not stored). The data, pairs and seeds are X10's, but the fit was not
   compared numerically with X10's stored effective IIA.
4. The read-out κ estimates disagree slightly (h4: 0.003 vs −0.016 vs −0.021). B = P + A holds to 2 % of $s_P$ at h4 and 7 %
   at z.
5. The untrained nets call one or two classes for most jets, so their B change rates (0.4–22 %) are not comparable to the
   trained net's in size.
6. A hold of g or S0 is a hold at the unboosted value; it is not a test of which input feature drives S0 beyond ΣpT (E5: S0
   follows ΣpT, R² 0.93).

## Explainer entry (for site/explain.py, Paper A tab)
- **Question:** a boost along the jet axis keeps the jet mass fixed but changes about a third of JEDI-linear's calls (H12).
  Which objects inside the network carry that change?
- **Background:** for massless constituents with pT fractions about the axis,
  $M_0=\sum_i p_{T,i}$ and $M_2=\sum_i p_{T,i}\,r_i^2$, with $m^2 \approx M_0 M_2$. A boost with $p_T \to \lambda p_T$
  gives $M_0\to\lambda M_0$ and $M_2\to M_2/\lambda$ (angles $\propto 1/\lambda$), so $m$ is unchanged. A read-out
  $a\log M_0 + b\log M_2$ moves by $(a-b)\log\lambda$.
- **Method:** the network sees the boosted jet while one unit is reset to its value on the unboosted jet (a "hold"). We
  count how many of the boost's call changes disappear, compare with random holds of the same size, and fit $a$ and $b$
  from pT-only, angle-only, M0-only and M2-only edits.
- **Result** (figure `results/jedi/e26_boost_route.png`):
  - At λ = 0.9, holding the ΣpT object S0 removes 55 % of the changes. Holding the M0 and M2 directions at S removes 83 %.
  - The soft-particle 11 GeV threshold removes none.
  - The h4 mass read-out does not move ($a=b$ within 2 %).
- **Checks:** R and T controls (2–5 % changes), 8,000 random directions, enumerated S subsets, untrained redraws, exact
  identities (all cuts held: 99 %; whole stage held: 100 %), Holm over 130 units.
- **Why it matters:** the boost response does not come from a mis-measured mass. JEDI-linear reads ΣpT itself as a class
  feature next to its boost-invariant mass variable. This makes it pT-dependent at fixed mass, which bears on D9's
  question of why hls4ml-trained taggers are boost-sensitive.
