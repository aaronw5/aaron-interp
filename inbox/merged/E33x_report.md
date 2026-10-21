# E33x: recurrence of program units across 10 trained JEDI-linear networks, compared with random formulas and untrained networks

**Conclusion.** Some second-round unit tracks the momentum-weighted squared distance of particles from the jet axis (Σ pT r²) in 9 of the 10 trained JEDI-linear networks. Only 1 of 145 random formulas that fit the reference network equally well shows up that often. This is evidence (p = 0.014, not significant after correcting for the 8 formulas tested) that the unit is learned and not simply produced by feeding in pT-sorted particles. When one particle is moved along the radius, the response grows close to r² (fitted power 1.7–2.15) in the 8 quantized 3-feature networks. The soft-particle count recurs in 6 of 9 other networks, which random formulas also reach about 5 % of the time, so its recurrence is not distinguished from chance here. In untrained networks the units mostly do not survive the quantizers, so the "untrained" comparison is meaningful only for the float architecture. There, 3 of 6 random redraws already contain a unit fitting the squared-radius formula.

## Question
Do the named second-round objects of the official JEDI-linear 3-feature n64 network recur in retrained / differently configured networks more often than (i) random rank-window formulas that fit the official network equally well and (ii) untrained networks? E33/E33v found recurrence (r2 unit 9/10, soft 7/10) but with only 30 decoys the decoy test could not reach p < 0.01.

## Terms
- **S unit**: one of the 64 second-round pooled objects S = Σ_i u_i / 16 (sum over particles of the second-round per-particle statements). Live = s.d. > 1e-9 over the fit jets.
- **r2 formula** (pT-weighted squared-radius unit, reference S29): a + b Σ pT r² + c Σ r² + d N(pT < 2 GeV, r > 0.1), r = √(Δη² + Δφ²) of a particle from the jet axis.
- **soft formula** (soft / empty-slot count, reference S19): Σ over slots of rank ≥ k of (t − pT)₊ (empty slot counts pT = 0); t, k fitted.
- **hard**: Σ pT of the k hardest particles; **core**: Σ (t − r)₊.
- **Competitors**: total pT, Σ r, multiplicity N, Σ pT r.
- **Decoy**: random rank-window hinge formula Σ_{rank<k} ReLU(cos θ z_pT + sin θ z_r + b), θ ~ U(0, 2π), b ~ N(0, 1), k = 1..n fitted, z = standardised pT, r (rng 4433), 600 drawn, plus E33v's three fixed decoys (Σ_{rank<k} r, Σ_{rank<k} pT², Σ_{rank≥k}(r − t)₊).
- **Hungarian matching**: the 4 named formulas are assigned to distinct S units by maximising the summed fit R², so one unit serves only one formula. Each decoy or competitor enters the same matching as a 5th row.
- **Recurrence count**: the number of the 9 other trained networks in which the formula's assigned unit reaches held-out (test) R² ≥ 0.9. A formula **qualifies** if it reaches R² ≥ 0.9 in the reference.
- **Holm**: Holm–Bonferroni step-down correction of a family of p-values.
- **Clopper-Pearson (CP)**: exact binomial 95 % interval.

## Procedure
- **Samples.** Fit = `jedi_core.sample(2000, 4401)` and test = `sample(2000, 4402)`, 10,000 jets each, the same as E33v. Each network gets its own input copy: n32 = first 32 slots, n128 = 128-slot copy of the same rows, 16-feature = 16-column copy (copy check max |Δ| = 0).
- **Networks.** Official 3f n64 (reference), n32, n128, perm-inv n64, 16f n64; X12 retrained HGQ-r150 s0–s2 and float s0–s1. Untrained controls are redraws of each of these 10 networks with seeds 0, 1 and 2 (30 networks; `redraw_untrained`: each weight tensor redrawn from a Gaussian with its trained mean and s.d., number formats kept).
- **Fits.** Per network, every live S unit is fitted with each formula. Each unit takes its best family parameter by fit R², then the units are assigned by Hungarian matching. R² is reported on the test jets.
- **Recurrence test.** H0: a named formula recurs no more often than the qualifying decoys. p = (1 + #{qualifying decoys with count ≥ the formula's count}) / (1 + #qualifying). Holm is applied over the 8 formulas. Per network, p_fm = (1 + #{qualifying decoys with test R² in network m ≥ the formula's}) / (1 + #qualifying), with Holm over 8 formulas × 9 networks.
- **Untrained base rate.** The same count over the 30 untrained networks, compared with the trained count (out of 9) by a one-sided Fisher exact test.
- **Radial weight w(r).** Applied to each trained network's r2-assigned unit.
  - Edit: per jet, one real particle with r in a bin (bins 0.015–0.05, 0.05–0.1, 0.1–0.15, 0.15–0.2, 0.2–0.3, 0.3–0.4) is moved radially to r ± 0.01. Its (Δη, Δφ) is rescaled, so pT, pT rank and azimuth are held fixed; the network sees the same particle list with one particle slightly further from or closer to the axis.
  - Control: an azimuthal move of the same arc length (0.01) at fixed r and pT. A purely radial formula predicts zero response to it.
  - Response: D = (S(r + 0.01) − S(r − 0.01)) / 0.02 per particle. Per bin, OLS D = α_b + γ_b pT, so γ_b is the response per unit radius per GeV.
  - Fit: for a unit ∝ Σ pT r^β, γ_b = C β r_b^(β−1). β comes from a weighted grid fit over β ∈ [−1, 4]. The CI is a 1,000-draw within-bin bootstrap, and H0 β = 2 is tested with a two-sided bootstrap p.
- **Matched hard/soft pT-add.** Applied to each trained network's hard-assigned unit (the "pT-sum unit").
  - Soft edit: per jet, δ = min(1 GeV, 0.9 × gap) is added to the rank ≥ 14 particle with the largest pT gap to the next-harder particle. The gap cap keeps the pT ordering; jets with δ < 0.05 GeV are dropped. N = 9,961 jets, mean δ 0.98 GeV, mean rank 17.5.
  - Hard edit: the same δ added to the hardest particle.
  - Ratio = response per GeV for the soft edit / response per GeV for the hard edit. A total-pT unit predicts 1; a hard-window unit with k ≤ 14 predicts 0. 1,000-draw jet bootstrap.
- **z windows.** E33 step 5 on the test jets: all particles' (Δη, Δφ) are scaled by 1.05 and 0.95, which changes jet mass by ±5 %. The window centre is the mass where the binned dz/d ln m changes sign, with |mean/se| ≥ 3 on both sides. Statistic T = number of (reference window, other trained network) pairs with a centre within ±10 GeV. Positional null: each other network's centres are shifted by an independent U(−50, 50) GeV offset, 1,000 draws.
- **Jobs.** e33x-a (5 official networks, 1,155 s), e33x-b3 (5 X12 networks, 523 s, 1 CPU / 3 GiB, peak 981 MiB), e33x-c (30 untrained networks, 1,717 s) and e33x-comb, all 1 CPU / 2 GiB unless stated. Smoke run: e33x-smoke.

## Numbers
**Accuracy on test (N 10,000).** Reference 80.43 % [79.64, 81.20] (Wilson), identical to E33v on the same sample; its CI covers the 80.6–81.3 % range. Other networks: n32 77.50, n128 80.66, perminv 81.12, 16f 81.58, HGQ s0 76.26, s1 57.15, s2 77.73, float s0 81.74, s1 81.69 %. All 30 untrained networks: 18.3–20.9 % (chance).

**Decoys.** 145 of 603 qualify in the reference: share 0.240, CP [0.207, 0.277]. Their recurrence counts over the 9 other networks, histogram for counts 0–9: 2, 35, 38, 24, 32, 7, 5, 1, 1, 0. Mean 2.70 of 9.

**Recurrence of named formulas and competitors (Hungarian matching):**

| formula | ref test R² | recurrence (of 9) | share of decoys ≥ (CP 95 %) | p | Holm (8 formulas) | untrained (of 30) | Fisher p trained > untrained |
|---|---|---|---|---|---|---|---|
| r2 (Σ pT r² unit) | 0.977 | 8 | 0.007 [0.0002, 0.038] | 0.014 | 0.11 | 3 | 2.2e-5 |
| soft count | 0.999 | 6 | 0.048 [0.020, 0.097] | 0.055 | 0.38 | 0 | 2.6e-5 |
| hard (k hardest pT) | 0.931 | 6 | 0.048 | 0.055 | 0.38 | 2 | 6.1e-4 |
| core Σ(t − r)₊ | 0.966 | 2 | 0.745 | 0.75 | 1.0 | 2 | 0.22 |
| total pT | 0.925 | 3 | 0.483 | 0.49 | 1.0 | 0 | 0.009 |
| Σ r | 0.852 (does not qualify) | 2 | – | – | – | 1 | – |
| multiplicity N | 0.631 (does not qualify) | 5 | – | – | – | 0 | – |
| Σ pT r | 0.791 (does not qualify) | 1 | – | – | – | 0 | – |

- **Unassigned best unit (no one-unit-one-formula rule).** 209 decoys qualify, mean recurrence 3.22. r2 8/9, p = 0.0095 (1 of 209 decoys ≥ 8); soft 6/9, p = 0.076; hard 6/9, p = 0.076.
- **Per-network r2 test R².** n32 0.965, n128 0.919, perminv 0.959, 16f 0.936, HGQ s0 0.954, s1 0.937, s2 0.955, float s0 0.899, float s1 0.906 (the 9th network misses the 0.9 cut by 0.001).
- **Per-network p against decoy R² in the same network.** r2: 0.021–0.41; soft: 0.007 in 5 networks. After Holm over 72 tests, all ≥ 0.49.
- **"r2-like" decoys.** The decoys whose best column correlates with the reference r2 formula at |corr| ≥ 0.9 (8 of them) recur in a mean of 1.25 of 9 networks; the other 137 recur in 2.78. Radial-hinge decoys that resemble the r2 formula do not recur more often than other decoys.
- **Where the untrained r2 fits come from.** All 3 untrained networks with r2 R² ≥ 0.9 are float-architecture redraws (float s0 u1, s0 u2, s1 u2; 52–56 live S units). The 24 untrained quantized redraws have 0–7 live S units and no unit fits any named formula. "Above the untrained rate" therefore says that quantized untrained networks have almost no live S units. For the float architecture the untrained rate is 3/6 redraws, against 1/2 trained float networks (float s1 0.906; float s0 0.899).

**Radial weight β (unit ∝ Σ pT r^β; 95 % bootstrap CI; p for H0 β = 2; azimuthal/radial response ratio):**

| network | β | CI | p(β = 2) | azimuthal / radial |
|---|---|---|---|---|
| official n64 | 1.90 | [1.83, 1.96] | 0.004 | 0.54 [0.53, 0.56] |
| n32 | 1.72 | [1.67, 1.77] | 0.002 | 0.50 |
| n128 | 1.89 | [1.85, 1.94] | 0.002 | 0.38 |
| perminv | 2.01 | [1.96, 2.05] | 0.58 | 0.30 |
| 16f | 1.26 | [1.23, 1.30] | 0.002 | 0.004 |
| HGQ s0 | 2.15 | [2.08, 2.22] | 0.002 | 0.55 |
| HGQ s1 | 1.92 | [1.87, 1.98] | 0.014 | 0.44 |
| HGQ s2 | 2.06 | [2.00, 2.12] | 0.038 | 0.52 |
| float s0 | 1.87 | [1.84, 1.90] | 0.002 | 0.28 |
| float s1 | 1.50 | [1.47, 1.54] | 0.002 | 0.24 |

**Per-bin γ_b (response per unit r per GeV).**
- Reference: 0.018, 0.037, 0.061, 0.087 for r < 0.2, which is linear in r (as Σ pT r² predicts), then 0.074 at r 0.2–0.3 and 0.007 at r 0.3–0.4, so the weight flattens beyond r ≈ 0.2.
- n32 and n128 show the same drop in the 0.3–0.4 bin (−0.007, −0.126).
- HGQ s0–s2 keep rising to 0.18–0.22 at r 0.3–0.4.
- The azimuthal control produces 24–55 % of the radial response in all 3-feature networks and 0.4 % in the 16-feature network, whose input includes ΔR directly.

**Matched pT-add on the hard-assigned unit (response per GeV, soft/hard ratio, 95 % CI):**
- Close to a total-pT response: reference 0.82 [0.80, 0.85] (p < 0.002 for ratio = 1), n128 1.01 [1.00, 1.03], perminv 1.07 [1.04, 1.10], HGQ s0 0.96 [0.90, 1.03] (p 0.26).
- No response outside the hard window: n32 0.00, 16f 0.00, HGQ s1 0.02 [0.00, 0.04], HGQ s2 0.00.
- Intermediate: float s0 0.24 [0.22, 0.27], float s1 0.51 [0.49, 0.53].

**z windows.**
- The reference has 13 window centres on these jets: 59, 68, 76, 87, 90, 94, 100, 101, 112, 115, 125, 145 and 147 GeV.
- T = 93 of 117 pairs, against 83.9 [72, 94] for the shifted null (p = 0.049). With 13 centres spread over 59–147 GeV, nearly any shifted set of 9–15 centres falls within ±10 GeV of one of them, so this test has little resolving power.
- Untrained networks: 0–3 centres each, T = 27 over 30 networks (8.1 when scaled to 9).

## Reading (scoped to what was measured)
- **Squared-radius unit.** It recurs in 8 of 9 other trained networks (9 of 10 counting the reference). Among 145 random rank-window hinge formulas that fit the reference unit as well, 1 recurs that often (p = 0.014; Holm over 8 formulas 0.11; unassigned version p = 0.0095). Decoys that resemble the r2 formula recur less, not more (1.25 vs 2.78 of 9). This is against the alternative "any radial rank-window hinge recurs because the inputs are pT-sorted and radial". The pre-registered threshold p < 0.01 is reached only in the unassigned version, and not after Holm.
- **Radial weight of the same unit.** Measured by the network's response to ±0.01 radial moves of one particle, it follows pT r^β with β 1.72–2.15 in the 8 quantized 3-feature networks, 1.50–1.87 in the float networks and 1.26 in the 16-feature network. In the reference the weight is r² up to r ≈ 0.2 and flat beyond. The E33v edit slopes of 0.3–0.7 are therefore not explained by β ≈ 1. They are compatible with the flattening at large r, combined with the large azimuthal response (24–55 % of the radial response), which shows these units are not purely radial at the one-particle level.
- **Soft count, hard window and core.** The soft count and the hard window each recur in 6 of 9 networks, which 4.8 % of decoys also reach (p = 0.055); this is not distinguished from the decoy rate. The core sum recurs at the decoy rate. The pT-add control reproduces E33v: the reference, n128, perminv and HGQ s0 hard units respond per GeV to soft-particle pT about as much as to hardest-particle pT (total-pT-like), while n32, 16f and HGQ s1/s2 have a genuine hard window.
- **Untrained comparison.** Untrained quantized networks have almost no live S units, so "above the untrained rate" (Fisher p ≈ 2e-5) mainly reflects that. In the float architecture, random redraws produce an r2-fitting unit in 3 of 6 cases.
- **Mass windows.** Recurrence in z is not resolved by the positional null (p = 0.049, with a nearly saturated statistic).

## Competing hypotheses
- **Thesis** (r2 and soft p < 0.01 and above the untrained rate; β ≈ 2 in most networks). r2: p = 0.014 with Hungarian matching and 0.0095 without, above the quantized-untrained rate; β 1.7–2.15 in 8/10 networks. Soft: p = 0.055, so not confirmed.
- **Alternative** (decoy base rate is high, so recurrence reflects the input format; β 0.3–0.7). The decoy mean is 2.7 of 9 and r2-like decoys recur less; β is not in 0.3–0.7 in any network. Both predictions are excluded for the r2 unit. For the soft count the input-format reading is not excluded.

## Checks and caveats
- **Job b stall and file overwrite.** Job e33x-b first stalled for about 15 min in a CephFS metadata wait (state D, `ceph_mdsc_wait_request`) on node chi-dgx-node01.csuchico.edu, then finished. Its replacement e33x-b2 ended in a failed state while writing and left e33x_trained_b.json empty. The X12 part was rerun as e33x-b3 (1 CPU / 3 GiB). The numbers above come from e33x-b3, whose per-network values match the e33x-b log line for line (e.g. HGQ s0 β 2.15, ratio 0.96). The node should be added to job.sh's exclusion list; I did not edit job.sh, because the task restricts writes to interp/, results/, logs/ and inbox/.
- **Memory.** Peak pod memory reads 2,047 MiB in the 2 GiB jobs, which includes page cache; the 3 GiB rerun peaked at 981 MiB.
- **Hungarian matching.** Decoys enter the matching as a 5th row, while the roles share it with only each other. This costs a decoy at most the units the roles hold; the unassigned version, which treats both alike, gives the same ordering.
- **Smaller recurrence counts than E33v.** Soft recurs 6/9 here against 7 in E33v, because the 0.9 cut is applied to the Hungarian-assigned unit (n128 0.756, HGQ s1 0.653, float s1 0.711).
- **Not rerun:** circuit95 (ACDC) on fresh jets.

## Files
- interp/e33x_recurrence_null.py, interp/e33x_plot.py
- results/jedi/e33x.json, e33x_trained_a.json, e33x_trained_b.json, e33x_untrained.json, e33x_recurrence_null.png
- logs/e33x-a.log, e33x-b.log, e33x-b2.log, e33x-b3.log, e33x-c.log, e33x-comb.log

## Explainer entry (for site/explain.py)
- **Question.** Is the squared-radius unit learned, or a by-product of feeding the network pT-sorted particle lists?
- **Background.** E33 found a unit fitting $a + b\sum_i p_{T,i} r_i^2 + c\sum_i r_i^2 + d\,N_{\rm soft}$ in 9 of 10 networks, but with only 30 random formulas it could not show this is rarer than chance.
- **Method.** 603 random formulas $\sum_{\mathrm{rank}<k}\mathrm{ReLU}(\cos\theta\, z_{p_T} + \sin\theta\, z_r + b)$; keep those fitting the reference (145); count in how many of the 9 other networks each reappears ($R^2 \ge 0.9$). One-particle radial moves $r \to r \pm 0.01$ measure $\gamma(r) = \partial S / (\partial r\, \partial p_T) = C\beta r^{\beta-1}$.
- **Result.** r2: 8/9 networks, against 1 of 145 random formulas ($p = 0.014$); $\beta = 1.72$–$2.15$ in quantized 3-feature networks (figure e33x_recurrence_null.png).
- **Checks.** Unassigned matching ($p = 0.0095$); r2-like decoys recur less; untrained quantized networks have no live fitting units; float redraws fit r2 in 3 of 6.
- **Why it matters.** It separates a learned jet observable from the input format.
- **Glossary additions.** Decoy formula, recurrence count, Clopper-Pearson interval, radial weight power β.
