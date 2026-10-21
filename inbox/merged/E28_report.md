# E28: collinear splits and soft additions inside JEDI-linear (claims A3 / A1; exploratory)

Script `interp/e28_irc_split_jedi.py` (job e28-run2, 1 CPU / 2 GiB, wall 1071 s, peak memory 2047 MiB), figure `interp/e28_plot.py`.
Results `results/jedi/e28_irc_split.json`, per-jet call changes `results/jedi/e28_irc_split_perjet.npz`, figure `results/jedi/e28_irc_split.png`, log `logs/e28-run.log`.

## Conclusion
JEDI-linear changes about 2-4 % of its calls when one particle is split into two particles at the same position, which leaves every pT-weighted angular quantity of the jet unchanged. For splits of particles below the hardest one, this is 2-3 times more than a control that moves a particle by one rank without adding one. The calls move in the direction of higher multiplicity (quark to gluon, W to Z), and adding a particle removes a fixed "empty slot" contribution from two named sums (S19, the soft-particle count over ranks 13-63, and S30, the sum over particles near the axis). The change is spread over many objects: holding S0 and S19 at their unsplit values removes only about 10 % of the changed calls, so most of the path is not yet attributed to named objects.

## Definitions
- **IRC safety.** An observable is infrared and collinear (IRC) safe if it does not change when a particle is split into two collinear particles with the same total momentum (collinear safety) or when a particle of vanishing energy is added (infrared or soft safety). Jet mass, $M_0$, $M_2$ and all angularities with exponent 1 in pT are IRC safe; particle multiplicity is not.
- **Collinear split (this node).** The particle of pT rank $k$ (rank 0 = hardest) with $(p_T, \Delta\eta, \Delta\phi)$ is replaced by two particles $(z\,p_T, \Delta\eta, \Delta\phi)$ and $((1-z)\,p_T, \Delta\eta, \Delta\phi)$; the jet is re-sorted by pT. Then $M_0 = \sum_i p_{T,i}$, $M_1 = \sum_i p_{T,i} r_i$, $M_2 = \sum_i p_{T,i} r_i^2$ (with $r_i = \sqrt{\Delta\eta_i^2 + \Delta\phi_i^2}$ to the jet axis) and the massless jet four-momentum are unchanged (measured max relative change $\le 4.4\times10^{-16}$ for $M_0, M_1, M_2$, $\le 3.4\times10^{-13}$ for the mass). Only the particle count $N$ (+1) and the pT ranks of the softer particles change.
- **Named objects** (second-round objects $S_u = \frac{1}{16}\sum_i u_{i,u}$, E5/E20): $S_0$ = sum of pT over the 14 hardest particles (rank window 0-13); $S_1 = \sum_{\text{ranks}\ge1}(r-0.2)_+$; $S_{19}$ = soft multiplicity, a hinge $\mathrm{ReLU}(2.86 - 1.39\,p_T)$ per slot over ranks 13-63, which also gives 2.86 for each empty slot; $S_{29} \approx \sum p_T r^2 + \sum r^2 +$ soft count (all ranks); $S_{30}$ = sum of a core hinge $\mathrm{ReLU}(3.47 - 21.0\,r)$ over all slots, which gives 3.38 per empty slot. "Rank-window objects" = the 17 live S units whose slot window is not 0-63 (all live S except $S_{29}$, $S_{30}$).
- **Calls changed** = fraction of jets whose arg-max class differs from the unedited jet. **sd units**: $\Delta S_u / \mathrm{sd}(S_u)$, sd over the unedited subset.
- **Hold / share removed.** The edited jet is run with chosen S units (or all 64 g units) replaced by the unedited jet's values (`run_jetwise` override). Share removed $= 1 - \#\text{changed(edit + hold)} / \#\text{changed(edit)}$; negative values mean the hold increases the number of changed calls.
- **Matched-rank control (swap).** The angles of the particles at ranks $k$ and $k+1$ are exchanged, so the particle that sat at rank $k$ moves one rank down and exchanges pT with its neighbour; $N$ is fixed and nothing is split. It changes $M_1$, $M_2$ and the mass (it is not moment-preserving), so it is a conservative comparison for the split.
- **McNemar**: exact binomial test on discordant pairs of paired binary outcomes. **Holm**: step-down family-wise correction.

## Procedure
- Sample: `jedi_core.sample(per_class=2000, seed=1202)` = 10,000 test jets. Unedited accuracy on all 10,000: **80.50 %** (binomial SE 0.40 pp; slightly below the 80.6-81.3 % check band; same code path that gave 81.15 % on E20's seed-41 sample). Edits on the **7,370 jets with $N \le 60$** (accuracy 80.88 %; class counts g 834, q 1801, W 1902, Z 1867, t 966; gluon and top jets are under-represented because many have more than 60 particles). Inputs are pT-sorted, padding is exactly (0, 0, 0).
- Edits: split at rank $k \in \{0, 5, 13, \text{soft}\}$ (soft = uniform random rank in 21..N-1, jets with $N \ge 22$; ranks 0/5/13 need $N > k+1$), $z \in \{0.5, 0.8\}$; split + removal of the softest particle (count fixed, "splitfix", z 0.5); soft addition of $n = 1..5$ particles of pT 0.5 GeV each, at $r$ copied from a random real particle of the same jet and a uniform random azimuth ($\Delta M_0 = 0.5 n$ GeV; mean relative $\Delta M_2$ +0.3 % (n 1) to +1.7 % (n 5)).
- Controls: (a) matched-rank swap at the same $k$; (b) a pT = 0 entry in the first empty slot at the rank-0 particle's angles ("pad0_at") and at (0, 0) ("pad0_origin"); (c) the untrained network (`Jedi(untrained=True, seed)`, seeds 0, 1; raw calls and calls from logits centred on their unedited mean); (d) scale: removal of the softest real particle, and matched data pairs (nearest neighbour of the same true class in standardised $\log M_0$, $\log M_2/M_0$, $M_1/M_0$, $\log m$ with $N' = N+1$, vs with $N' = N$).
- Statistics: 1000 multinomial bootstrap resamples over jets (same resamples for every quantity); exact McNemar on correct/incorrect per edit (Holm over 21 edits); split vs swap: exact McNemar and sign-flip permutation (10,000) on paired changed indicators, Holm over the 8 (rank, z) cells; per-unit holds: McNemar edit vs edit+hold, Holm over the 19 live S units per edit; directional flows: two-sided binomial sign test.

## Results

### 1. Calls changed (95 % bootstrap CI); H0: split changes as many calls as the matched-rank swap
| rank $k$ | split z 0.5 | split z 0.8 | swap (control) | split − swap (z 0.5) | p (perm, Holm 8) | split + drop softest |
|---|---|---|---|---|---|---|
| 0 | 4.31 % [3.87, 4.78] | 3.57 % [3.13, 4.02] | 3.76 % [3.32, 4.19] | +0.56 pp [−0.01, +1.15] | 0.14 | 4.22 % |
| 5 | 2.86 % [2.50, 3.26] | 2.48 % [2.14, 2.84] | 1.06 % [0.84, 1.29] | +1.80 pp [+1.41, +2.24] | 8e-4 | 3.27 % |
| 13 | 2.98 % [2.60, 3.35] | 2.43 % [2.09, 2.78] | 1.26 % [1.01, 1.53] | +1.72 pp [+1.34, +2.10] | 8e-4 | 3.09 % |
| > 20 | 2.43 % [2.09, 2.83] | 2.03 % [1.71, 2.38] | 0.79 % [0.59, 1.00] | +1.64 pp [+1.31, +2.03] | 8e-4 | 2.20 % |

z 0.8 vs swap: +1.4, +1.2, +1.2 pp at ranks 5, 13, > 20 (Holm p 8e-4 each); −0.19 pp [−0.75, +0.37] at rank 0 (p 0.55). McNemar exact on the same pairs gives Holm p from 1e-19 to 2e-12 for ranks 5/13/> 20.
Accuracy is unchanged by splits (McNemar correct→wrong vs wrong→correct, e.g. rank 0 z 0.5: 129 vs 128, p 1.0). The rank-0 swap lowers accuracy 80.88 → 78.72 % (202 vs 43, Holm p 2e-24); it is a large angular edit (the two hardest particles exchange positions).
Soft additions: 1.78 % [1.49, 2.09] (n 1), 2.33, 3.22, 4.08, 4.91 % [4.42, 5.39] (n 5). Removing the softest particle: 1.87 % [1.56, 2.17].

### 2. Direction of the call changes (rows = unedited call → edited call; binomial sign test)
| edit | q→g | g→q | p | W→Z | Z→W | p |
|---|---|---|---|---|---|---|
| split r0 z0.5 | 105 | 4 | 2e-26 | 56 | 27 | 0.002 |
| split r13 z0.5 | 47 | 15 | 6e-5 | 30 | 26 | — |
| split r>20 z0.5 | 35 | 8 | 4e-5 | 25 | 11 | — |
| splitfix r0 (N fixed) | 73 | 20 | 3e-8 | 48 | 36 | — |
| splitfix r5 / r13 / r>20 | 24 / 29 / 16 | 42 / 35 / 33 | 0.04 / 0.53 / 0.02 | — | — | — |
| soft add n 1 / n 5 | 39 / 130 | 7 / 1 | 2e-6 / 1e-37 | 17 / 87 | 1 / 2 | — / 1e-23 |
| drop softest (N − 1) | 2 | 52 | 2e-13 | 6 | 14 | — |
| swap r0 | 8 | 5 | — | 117 | 29 | 1e-13 |

Adding a particle (split or soft addition) moves calls from quark to gluon and from W to Z; removing one moves them from gluon to quark. When the count is restored after the split (splitfix), the total number of changed calls is the same (difference −0.1 to +0.4 pp) but the q↔g direction becomes balanced or reversed for ranks 5, 13, > 20.

### 3. Named objects (mean $\Delta S/\mathrm{sd}$, 95 % CI; mean $|\Delta S|/\mathrm{sd}$)
| edit | S0 | S19 | S29 | S30 | S1 |
|---|---|---|---|---|---|
| split r0 z0.5 | −0.006 (0.036) | −0.100 [−0.101, −0.099] (0.100) | +0.022 (0.048) | −0.015 (0.054) | +0.009 (0.081) |
| split r5 z0.5 | −0.016 (0.032) | −0.100 (0.100) | +0.023 (0.053) | −0.024 (0.059) | +0.009 (0.081) |
| split r13 z0.5 | +0.013 (0.020) | −0.096 (0.098) | +0.010 (0.048) | −0.038 (0.068) | +0.012 (0.088) |
| split r>20 z0.5 | +0.001 (0.006) | −0.038 (0.042) | +0.011 (0.039) | −0.071 (0.084) | +0.046 (0.100) |
| soft add n 1 | +0.009 | −0.036 | +0.008 | −0.047 | +0.022 |
| soft add n 5 | +0.042 | −0.181 | +0.044 | −0.236 | +0.120 |
| drop softest | −0.006 | +0.028 | −0.038 | +0.107 | −0.110 |
| swap r0 / r13 | 0.000 | 0.000 | 0.000 (0.003) | 0.000 (0.005 / 0.013) | 0.000 |

All CIs are narrower than ±0.003 sd. Every mean $|\Delta S|$ for a split is ≤ 0.10 sd. $S_{19}$ drops by 0.100 sd in 99 % of jets for any split of a particle harder than 2 GeV: in raw units $-0.18 \approx -2.86/16$, which is the loss of one empty slot in $S_{19}$'s window; splitting a particle already below 2 GeV (rank > 20) changes it less (−0.038 sd). $S_0$ moves by ≤ 0.036 sd (mean |.|): the summed pT of the 14 hardest particles drops by 15.5 GeV on average for rank-0/5 splits, and $S_0$ reads it mainly through the context g (E4/E5), which changes little. $S_{30}$ responds most to soft additions (−0.047 sd per added particle; empty-slot term 3.38/16).

### 4. Mediation: share of the edit's call changes removed by a hold (95 % CI; McNemar p edit vs edit+hold)
| edit | S19 | S0 | S0+S19 | 17 rank-window S | S29+S30 | g (all 64) |
|---|---|---|---|---|---|---|
| split r0 z0.5 | 0.07 [0.02, 0.12] p 0.009 | 0.02 [−0.03, 0.06] | 0.11 [0.05, 0.16] p 7e-4 | 0.50 [0.42, 0.58] p 1e-20 | −0.08 [−0.16, 0.00] | 0.01 [−0.06, 0.07] |
| split r5 z0.5 | 0.03 [−0.05, 0.10] | 0.05 [−0.01, 0.12] | 0.07 [−0.02, 0.15] | 0.22 [0.10, 0.34] p 0.003 | −0.17 [−0.32, −0.05] | 0.02 [−0.08, 0.11] |
| split r13 z0.5 | 0.06 [−0.02, 0.13] | 0.05 [−0.01, 0.09] | 0.10 [0.02, 0.17] p 0.02 | 0.18 [0.06, 0.30] p 0.008 | 0.01 [−0.11, 0.12] | 0.15 [0.06, 0.22] p 8e-4 |
| split r>20 z0.5 | 0.02 [−0.04, 0.08] | 0.02 [−0.02, 0.06] | 0.05 [−0.02, 0.11] | 0.04 [−0.14, 0.19] | −0.14 [−0.30, 0.01] | 0.18 [0.08, 0.27] p 0.002 |
| soft add n 1 | 0.02 | 0.02 | 0.07 | 0.24 [0.08, 0.38] p 0.008 | −0.02 | 0.31 [0.20, 0.42] p 3e-6 |
| soft add n 5 | 0.12 [0.07, 0.17] p 3e-5 | 0.02 | 0.18 [0.12, 0.23] p 3e-9 | 0.06 [−0.04, 0.15] | 0.08 [−0.02, 0.16] | 0.41 [0.35, 0.47] p 1e-27 |

Sanity: holding all 19 live S units removes 1.000 for every edit. Single-unit holds (Holm over 19 per edit): no single unit removes more than 0.09 of the hard-split changes; for soft splits and soft additions $S_{30}$ alone removes 0.13-0.41 (Holm p ≤ 0.03 for r>20 z0.8 and n 2-5). Holding $S_{56}$ (window 8-63) at its unedited value **increases** the changed calls in every edit (share −0.15 to −0.58, Holm p ≤ 9e-4): on its own, $S_{56}$'s response to the extra particle partly cancels the other objects' effect on the calls. Holds are not additive (single-unit shares do not sum to the set shares).

### 5. Controls
- (b) Padding: a pT = 0 entry at (0, 0) changes no call (0/7370, exact no-op). A pT = 0 entry at the hardest particle's angles changes 0.35 % [0.23, 0.50] of calls and moves $S_{30}$ by −0.016 sd: slots with pT = 0 but nonzero angles are not equivalent to padding (the angular l1 half-plane cuts have zero pT weight).
- (c) Untrained network: raw calls are constant (seed 0 all t, seed 1 all W), so 0 calls change under every edit. With logits centred on their unedited mean, seed 0 changes 13-14 % of calls under splits, 0.9-2.3 % under swaps, 7.3 % / 23.5 % under soft additions n 1 / n 5; seed 1 (centred calls 99.7 % W) changes ≤ 0.2 %. A random-weight network of this architecture is count-sensitive; the trained network's split response (2-4 %) is smaller.
- (d) Scale: removing the softest particle changes 1.87 % of calls. Matched data pairs: call disagreement 15.1 % [14.3, 16.0] for pairs with $N' = N + 1$ (n 7,747, median match distance 0.19) vs 15.6 % [14.9, 16.3] for $N' = N$ (n 9,990, 0.17); difference −0.5 pp [−1.6, +0.6]. At this matching resolution, the pair difference from other jet properties (15 %) is much larger than the one-particle effect, so the pair control does not resolve a count effect.

### 6. Example (figure panel c)
Gluon jet (sample index 6323, 39 particles), rank-13 particle (8.4 GeV) split z 0.5 into two 4.2 GeV particles at ranks 20, 21. Call g (p 0.49) → q (p 0.48); this jet sits near the g/q boundary. $\Delta S/\mathrm{sd}$: S0 0.00, S19 −0.09, S29 +0.03, S30 +0.02.

## Reading (scoped to this sample and these edits)
- Thesis prediction "hard splits change calls above the matched-rank control": holds for ranks 5 and 13 (+1.7-1.8 pp, Holm p 8e-4); at rank 0 the control (swap of the two hardest) itself changes 3.8 % of calls and the split is not separated from it (+0.56 pp, p 0.14). Splits of soft particles (rank > 20) also exceed the control (+1.6 pp).
- Thesis prediction "hard splits move S0": $S_0$ moves by ≤ 0.036 sd (mean |.|) and holding it removes ≤ 0.05 of the changes (CIs include 0). Not supported at this resolution.
- Thesis prediction "soft splits move S19": $S_{19}$ moves more for hard splits (−0.100 sd) than for soft splits (−0.038 sd), through the empty-slot term; holding $S_{19}$ removes 0.02-0.07 of split changes and 0.12-0.14 of the n 3-5 soft-addition changes.
- Alternative "calls follow only pT-weighted moments; splits change about as many calls as the control": excluded for ranks 5, 13, > 20 (both z). Its second part, "object changes below 0.1 sd", holds for every named object on average (max mean |ΔS| 0.100 sd for $S_{19}$).
- Where the unsafety enters: the 17 rank-window objects held together remove 0.50 of rank-0 split changes and 0.18-0.24 for ranks 5/13 and one soft addition; the context g removes 0.15-0.41 for rank 13, soft splits and soft additions. The remaining shares are not attributed to a single named object; $S_{56}$ acts against the change.

## Issues
1. Unedited accuracy 80.50 % on the full 10,000 jets, 0.1 pp below the requested check band (within 1 binomial SE of the 80.9 % test accuracy); same forward pass as E20 (81.15 %).
2. Peak memory 2047 MiB equals the 2 GiB request; a larger sample would need 3 GiB. First submission (e28-run) stalled at the import step on node ucm-fiona01.ucmerced.edu (CPU 7 millicores for 12 min) and was deleted; e28-run2 ran on rci-tide-gpu-04.sdsu.edu.
3. Gluon jets: only 834 of 2000 have $N \le 60$, so the q/g flows are on a multiplicity-biased subset.
4. Rank-0 matched-rank control is not moment-preserving and is a large edit; a moment-matched control at rank 0 is not available in this design.
5. The untrained control is degenerate with raw calls (constant output); the centred variant is reported for seeds 0, 1 only.
6. The matched-data-pair control (d) is limited by matching resolution (15 % disagreement at ΔN = 0).

## Explainer (for site/explain.py, Paper A tab)
**Question.** Is JEDI-linear's decision collinear-safe and soft-safe, and through which named objects does a split or an added soft particle change it?
**Background.** A jet observable is collinear safe if splitting one particle into two collinear particles with the same total momentum leaves it unchanged; soft safe if adding a particle of vanishing energy leaves it unchanged. Mass and pT-weighted moments $M_\kappa = \sum_i p_{T,i} r_i^\kappa$ are both; particle count is neither. JEDI-linear's second-round objects include pT-rank windows (e.g. $S_0 = $ pT sum of ranks 0-13) and per-slot hinges that also count empty slots ($S_{19}$, $S_{30}$), so they can respond to a split even though $M_0, M_1, M_2$ do not.
**Method.** On 7,370 test jets with at most 60 particles, replace the particle of rank $k$ by $(z p_T, \eta, \phi)$ and $((1-z)p_T, \eta, \phi)$ and re-sort; $M_0, M_1, M_2$ and the jet mass are unchanged to $10^{-13}$. Compare with exchanging the positions of ranks $k$ and $k+1$ (no new particle), adding 0.5 GeV particles, and removing the softest particle. Hold named objects at their unsplit values to measure how much of the call change goes through them: $\text{share} = 1 - \#\text{changed}_{\text{hold}}/\#\text{changed}_{\text{edit}}$.
**Result.** (figure `e28_irc_split.png`) Splits change 2.0-4.3 % of calls; for ranks 5, 13 and > 20 this is 1.6-1.8 pp above the control (Holm p 8e-4). Adding a particle shifts calls toward gluon (105 q→g vs 4 g→q for the rank-0 split) and toward Z; removing one shifts them toward quark. $S_{19}$ drops by one empty-slot unit (0.10 sd) for every split; $S_0$ moves by ≤ 0.04 sd. Holding $S_0$ and $S_{19}$ removes about 10 % of the changed calls; holding all 17 rank-window objects removes 50 % at rank 0 and about 20 % at ranks 5/13.
**Checks.** A pT = 0 entry identical to padding changes nothing; a pT = 0 entry at a real angle changes 0.35 % of calls. Holding all 19 live objects removes 100 %. Untrained network: constant raw calls; with centred logits one seed changes 13-14 % of calls under splits. Accuracy is unchanged by the splits (changes go both ways).
**Why it matters.** The trained JEDI-linear is not collinear or soft safe: part of its quark/gluon and W/Z decision follows particle count, which is read through the empty-slot terms of named objects, and the rest of the split response is spread across many rank-window objects rather than through $S_0$.

Glossary additions: IRC safety; collinear split; matched-rank control (swap); empty-slot term ($S_{19}$ 2.86/16, $S_{30}$ 3.38/16 per empty slot); share removed (hold).

Conclusion (table row): Splitting one particle into two collinear particles, which leaves the jet's mass and all pT-weighted moments unchanged, changes 2-4 % of JEDI-linear's calls, more than moving a particle by one rank, and pushes quark calls toward gluon. Part of this goes through the count of empty slots in two named sums (soft count S19 and core sum S30); holding the named S0 and S19 removes only about a tenth of the change, so most of the path is spread over other objects and is not yet named.
