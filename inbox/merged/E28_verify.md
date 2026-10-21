# E28 verification: collinear and soft splitting inside JEDI-linear

Script `interp/e28_verify.py` (imports the edit functions of `interp/e28_irc_split_jedi.py` unchanged), figure `interp/e28_verify_plot.py` -> `results/jedi/e28_verify.png`.
Job e28-verify (1 CPU / 2 GiB, node ucm-fiona01.ucmerced.edu, wall 726 s, peak memory 1159 MiB, 0.99 cores). Results `results/jedi/e28_verify.json` (also `$AI_RESULTS/jedi/e28_verify.json`), log `logs/e28-verify.log`.

## Conclusion
On a new set of jets, splitting one particle into two particles at the same position (which leaves the jet mass and every pT-weighted angular sum unchanged) again changes 2-4 % of JEDI-linear's calls and pushes quark calls toward gluon and W calls toward Z, as E28 reported. A new control that moves the same particle without adding one (rotating it about the jet axis) changes about 1 % of calls, so the split also exceeds its control for the hardest particle; E28's statement that the rank-0 split is not separated from its control holds only for E28's angle-swap control. E28's mediation reading does not survive a size-matched control: holding the 17 "rank-window" objects removes fewer changed calls than holding a random 17 of the 19 active objects, and holding S0 and S19 removes about as much as a random pair. Which internal objects carry the split response is therefore still open.

## Definitions
- **Collinear split**: the particle at pT rank $k$ (rank 0 = hardest), $(p_T, \Delta\eta, \Delta\phi)$, is replaced by $(z p_T, \Delta\eta, \Delta\phi)$ and $((1-z)p_T, \Delta\eta, \Delta\phi)$, then the jet is re-sorted by pT. $M_0=\sum p_T$, $M_1=\sum p_T r$, $M_2=\sum p_T r^2$ ($r=\sqrt{\Delta\eta^2+\Delta\phi^2}$) and the jet mass are unchanged (measured max relative change $4.6\times10^{-16}$ for $M_2$, $4\times10^{-13}$ for the mass). The count $N$ rises by one and every particle softer than the daughters moves down one rank.
- **swap_r$k$ (E28's control)**: the angles of ranks $k$ and $k+1$ are exchanged; $N$ and the pT list are fixed, but $M_1$, $M_2$ and the mass change (rank 0: mean relative mass change 1.0 %, max $M_2$ change 670 %).
- **rot_r$k$ (new control)**: the particle at rank $k$ is rotated about the jet axis by a uniform random azimuth, keeping its pT and $r$. $N$, every rank, $M_0$, $M_1$, $M_2$ are unchanged (max relative change $\le 1.1\times10^{-7}$, float32 rounding); only the jet mass moves (mean relative change 2.0 % at rank 0, 0.2 % at rank 5, $\le$ 0.05 % at ranks 13 and > 20). It is the same-rank, same-count edit of the same particle without the added particle, which is the comparison the split needs.
- **Hold / share removed**: the edited jet is run with chosen second-round objects $S_u$ (or all 64 context units $g$) replaced by their values on the unedited jet (`run_jetwise`). Share removed $=1-\#\text{changed(edit+hold)}/\#\text{changed(edit)}$; negative = the hold increases changes.
- **Random-set hold control (new)**: the same share for 20 random 2-unit subsets and 20 random 17-unit subsets of the 19 live $S$ units (the units that are not constant). Holding 17 of 19 is the same as letting only the other 2 respond. Rank p = (1 + #random sets at least as extreme) / 21.
- Named objects (E5/E20): $S_0$ = summed pT of the 14 hardest particles; $S_{19}$ = soft multiplicity over ranks 13-63 (a per-slot hinge that also gives 2.86 per empty slot); $S_{29}\approx\sum p_T r^2+\sum r^2+$ soft count (all ranks); $S_{30}$ = core sum $\sum \mathrm{ReLU}(3.47-21r)$ over all slots (3.38 per empty slot); $S_{56}$ = a live object over slots 8-63. "17 rank-window objects" = the 19 live $S$ minus $S_{29}$, $S_{30}$.
- **McNemar**: exact binomial test on discordant pairs. **Holm**: step-down family-wise correction. sd units: $\Delta S/\mathrm{sd}(S)$ over the unedited jets.

## Procedure
- Sample: `jedi_core.sample(per_class=2000, seed=2029)` = 10,000 test jets (E28 used seed 1202). Unedited accuracy 79.93 % (E28 sample 80.50 %; both below the 80.6-81.3 % band; balanced per-class sampling, binomial SE 0.40 pp). Edits on the 7,389 jets with $N\le60$ (accuracy 80.36 %; g 848, q 1784, W 1915, Z 1888, t 954).
- Edits (E28's code): split at rank 0, 5, 13, random rank > 20 with $z$ 0.5, 0.8; split + drop softest (N fixed); 1 and 5 added 0.5 GeV particles; drop softest; pT = 0 entry at the rank-0 angles (pad0_at) and at (0,0) (pad0_origin). Controls: swap_r$k$ and the new rot_r$k$ at the same slots.
- Holds on split z0.5 (four ranks), soft_add1, soft_add5: $S_{19}$, $S_0$, $S_0+S_{19}$, 17 window objects, $S_{29}+S_{30}$, $S_{29}$, $S_{30}$, $S_{56}$, all live $S$, all live except $S_{29}$, all live except $S_{30}$, $g$; 20 random 2-sets and 20 random 17-sets.
- Statistics: 1000 multinomial bootstrap resamples over jets; split vs control: sign-flip permutation (10,000) Holm over 8 (rank, z) cells per control, plus exact McNemar; flows: two-sided binomial sign test. H0 (call level): the split changes no more calls than the matched control. H1: it changes more.

## Results

### 1. Calls changed (95 % bootstrap CI) and split minus control
| rank | split z0.5 | split z0.8 | swap | rot (new) | split − swap (z0.5) | Holm p | split − rot (z0.5) | Holm p |
|---|---|---|---|---|---|---|---|---|
| 0 | 4.13 % [3.64, 4.55] | 3.28 | 3.55 [3.13, 4.02] | 1.26 [1.01, 1.52] | +0.58 pp [+0.01, +1.14] | 0.10 | +2.87 pp [+2.38, +3.36] | 8e-4 (McNemar 5e-34) |
| 5 | 2.83 [2.46, 3.22] | 2.37 | 1.29 | 0.96 | +1.54 [+1.10, +1.94] | 8e-4 | +1.87 [+1.48, +2.25] | 8e-4 |
| 13 | 2.69 [2.34, 3.07] | 2.30 | 1.18 | 0.86 | +1.51 [+1.16, +1.86] | 8e-4 | +1.84 [+1.48, +2.19] | 8e-4 |
| > 20 | 2.30 [1.95, 2.67] | 1.89 | 0.96 | 1.32 | +1.33 [+0.97, +1.68] | 8e-4 | +0.98 [+0.60, +1.35] | 8e-4 |

E28 (seed 1202): split 4.31 / 2.86 / 2.98 / 2.43 %, swap 3.76 / 1.06 / 1.26 / 0.79 %, split − swap +0.56 / +1.80 / +1.72 / +1.64 pp. The z0.8 split also exceeds rot at every rank (+2.02, +1.41, +1.44, +0.57 pp; Holm p ≤ 0.002) but not swap at rank 0 (−0.27 pp, p 0.35).
Accuracy: splits do not lower it (rank 0 z0.5 80.36 → 80.54 %); swap_r0 lowers it to 78.59 % (182 vs 51 discordant, p 2e-18); rot leaves it (rank 0: 42 vs 33, p 0.36). So the rank-0 swap is a large, accuracy-lowering angular edit; the rot control at rank 0 is not.
Other edits: soft_add1 1.87 % [1.54, 2.18], soft_add5 4.65 % [4.17, 5.14]; drop softest 2.08 %; pad0_at 0.34 % [0.22, 0.47]; pad0_origin 0/7389 (bit-identical to padding, so this is an identity check, not a control).
Split + drop softest vs split (z0.5): −0.12 pp [−0.45, +0.19] (rank 0), +0.42 pp [+0.11, +0.74] (rank 5), +0.31 [+0.01, +0.63] (rank 13), +0.06 [−0.30, +0.42] (> 20). E28 had +0.41 [+0.11, +0.73] at rank 5, so "the total number of changed calls stays the same" holds at ranks 0 and > 20 and is about 0.3-0.4 pp higher with the count restored at ranks 5 and 13, on both samples.

### 2. Direction (unedited call → edited call)
| edit | q→g | g→q | p | W→Z | Z→W | p |
|---|---|---|---|---|---|---|
| split r0 z0.5 | 94 | 6 | 2e-21 | 58 | 30 | 0.004 |
| split r13 z0.5 | 38 | 10 | | 38 | 23 | |
| split r>20 z0.5 | 37 | 12 | | 22 | 22 | |
| splitfix r5 / r13 / r>20 | 16 / 21 / 14 | 48 / 35 / 37 | | | | |
| soft add 5 | 131 | 3 | 4e-35 | 76 | 0 | 3e-23 |
| drop softest | 2 | 50 | 6e-13 | 9 | 26 | 0.006 |
| rot r0 / r13 | 4 / 6 | 5 / 6 | | 21 / 12 | 25 / 7 | |

Reproduced: adding a particle moves calls q→g and W→Z, removing one moves g→q; restoring the count after the split reverses the q/g direction at ranks 5, 13, > 20. The rot control has no direction.

### 3. Objects (mean ΔS/sd)
$S_{19}$: −0.099 (ranks 0, 5), −0.094 (13), −0.038 (> 20), −0.179 (5 soft additions); E28: −0.100, −0.100, −0.096, −0.038, −0.181. $S_0$ mean |ΔS| ≤ 0.035 sd for every split. $S_{30}$ −0.015 to −0.071. So "every split lowers $S_{19}$ by one empty-slot unit (0.10 sd)" holds for splits of particles above about 2 GeV; a split at rank > 20 lowers it by 0.038 sd. (The share of jets within ±0.01 sd of −0.100 is 35 %; the change is quantized by the pooling quantizer, so the per-jet value is spread around the mean; the mean and the sign are what reproduce.)

### 4. Holds vs random sets of the same size (share removed, 95 % CI)
| edit | S0+S19 | random 2-sets mean [min, max] | rank p | 17 window S | random 17-sets mean [min, max] | rank p (≤) | removed, all held but S29 | removed, all held but S30 | S29 alone | S56 alone | g |
|---|---|---|---|---|---|---|---|---|---|---|---|
| split r0 | 0.105 [0.05, 0.16] | 0.01 [−0.09, 0.12] | 0.10 | 0.50 [0.42, 0.58] | 0.63 [0.45, 0.86] | 0.19 | 0.60 | 0.77 | −0.06 | −0.12 | 0.09 |
| split r5 | 0.03 [−0.05, 0.10] | −0.02 [−0.26, 0.08] | 0.33 | 0.23 [0.09, 0.34] | 0.57 [0.22, 0.86] | 0.10 | 0.44 | 0.63 | −0.16 | −0.23 | −0.05 |
| split r13 | 0.12 [0.04, 0.20] | 0.00 [−0.23, 0.12] | 0.10 | 0.17 [0.02, 0.30] | 0.58 [0.20, 0.90] | 0.05 | 0.35 | 0.60 | −0.11 | −0.21 | 0.15 |
| split r>20 | 0.08 [0.00, 0.15] | −0.02 [−0.51, 0.16] | 0.29 | 0.00 [−0.20, 0.15] | 0.52 [−0.12, 1.00] | 0.19 | 0.27 | 0.42 | −0.23 | −0.51 | 0.18 |
| soft +1 | 0.17 [0.09, 0.25] | 0.01 [−0.40, 0.33] | 0.19 | 0.30 [0.14, 0.43] | 0.63 [0.20, 1.00] | 0.19 | 0.51 | 0.58 | −0.19 | −0.40 | 0.39 |
| soft +5 | 0.20 [0.13, 0.26] | 0.05 [−0.41, 0.42] | 0.19 | 0.15 [0.05, 0.23] | 0.63 [0.30, 1.00] | 0.05 | 0.57 | 0.42 | −0.09 | −0.41 | 0.37 |

("all held but S29" = all 19 live S held except S29, so only S29 responds; 1 minus this share is the fraction of changes S29 alone reproduces: 0.40, 0.56, 0.65, 0.73 for the four splits.) $S_{30}$ alone: 0.05, 0.02, 0.06, 0.09 for the splits; 0.20 [0.10, 0.28] and 0.36 [0.29, 0.42] for 1 and 5 soft additions (E28: 0.15, 0.39).
- The E28 numbers reproduce (S0+S19 0.03-0.12; window 0.50 / 0.23 / 0.17 / 0.00; g 0.09-0.18 for splits, 0.37-0.39 for soft additions; S56 negative).
- H0 (hold control): the named set removes no more than a random set of the same size. For S0+S19, rank p 0.10-0.33 against 20 random pairs: not excluded. For the 17 window objects, the named set removes less than the random 17-sets in every edit (rank p for "≤" 0.05-0.19): letting only $S_{29}$ and $S_{30}$ respond reproduces 50-100 % of the changes, at least as much as letting a random pair respond. Letting only $S_{29}$ respond reproduces 40-73 % of the split changes, while holding $S_{29}$ alone increases them (−0.06 to −0.23); $S_{56}$ behaves the same way. The objects act through redundant and opposing routes, so a hold share is not a share of the path.
- With 20 random sets per size the smallest attainable rank p is 0.048; no Holm correction was applied across the 6 edits.

## Code check (interp/e28_irc_split_jedi.py, interp/e28_plot.py)
- Split: daughters at the parent's float32 angles; remainder pT computed in float64 from the float32 first daughter, so $M_0$ is conserved to 1 ulp; the new particle is written into slot N (first empty) and the jet is stable-re-sorted. No leakage: moments unchanged to $10^{-16}$, mass to $10^{-13}$. Only $N$ and ranks change, as stated.
- Padding: `nreal` counts pT > 0; padding is exactly (0, 0, 0) in the sample (max |angle| in padding 0.0). The network sees padding as the standardised value $-\text{shift}/\text{scale}$, which is why empty slots contribute to $S_{19}$, $S_{30}$; this is the model's input convention, not a bug.
- Holds override S after the pooling quantizer (qs1) with the exactly computed unedited S, and g after dd; consistent with jedi_core. "all live S = 1.00" is guaranteed by construction (the other 45 S units are constant), so it checks only that the override works.
- Swap control: not moment-preserving (changes $M_1$, $M_2$, mass; rank 0 lowers accuracy 2 points); addressed here by rot_r$k$.
- pad0_origin is bit-identical to padding, so its 0/7370 is an identity, not a null control.
- Figure e28_irc_split.png: panel (b) labels pad0_at as "null: pT = 0 entry in padding"; pad0_at is a pT = 0 entry at the hardest particle's angles (not equivalent to padding). Otherwise panels match the JSON.
- The thesis_note's p 1e-20 for the window hold is McNemar against no hold, not against a size-matched hold.
- Issue (3) of E28 (ucm-fiona01 stalls at import): this verification ran on ucm-fiona01 without a stall (726 s), so one stall does not establish a bad node.
- Memory: without the matched-pair and untrained parts the job peaks at 1159 MiB; E28's 2047 MiB comes from those parts.

## Corrected reading
On two samples (seeds 1202, 2029; 7,370 and 7,389 jets with N ≤ 60), an exact collinear split, which leaves M0, M1, M2 and the jet mass unchanged, changes 2.0-4.3 % of JEDI-linear's calls. Against a same-rank control that rotates the same particle about the axis (N, ranks, M0, M1, M2 fixed), the excess is +2.9 pp at rank 0, +1.9 pp at ranks 5 and 13 and +1.0 pp at rank > 20 (Holm p 8e-4, seed 2029); against E28's angle-swap control the excess is +1.3 to +1.8 pp at ranks 5, 13, > 20 and +0.6 pp (Holm p 0.10-0.14) at rank 0, where the swap itself is a large edit. Adding a particle moves calls q→g and W→Z; removing one moves g→q; restoring the count after the split keeps the number of changed calls within 0.4 pp and reverses the q/g direction at ranks 5, 13, > 20. Splits of particles above about 2 GeV lower S19 by 0.10 sd (one empty slot), splits at rank > 20 by 0.04 sd; S0 moves ≤ 0.035 sd. Holding S0 + S19 removes 0.03-0.12 of the split changes and the 17 rank-window objects 0.00-0.50; neither exceeds holds of random live-S sets of the same size (random 17-sets remove 0.52-0.63 on average), and letting only S29 respond reproduces 40-73 % of the split changes while holding S29 alone increases them. Holding g removes 0.15-0.18 of rank-13 and soft-split changes and 0.37-0.41 of soft-addition changes; S30 alone 0.15-0.39 of soft-addition changes (not compared with random single units). Conclusion: splitting a particle into two collinear ones changes 2-4 % of JEDI-linear's calls and pushes quark calls toward gluon, more than moving the same particle without splitting it, so the network is neither collinear- nor soft-safe. The internal route is not identified by these holds: neither S0 + S19 nor the rank-window objects remove more than random sets of the same size, and the objects respond through redundant and opposing paths.

## Explainer addition (site/explain.py, E28 checks)
**Check (verification, seed 2029).** A second matched control rotates the split particle about the jet axis, which keeps $N$, every rank and $M_0, M_1, M_2$ fixed: it changes 0.9-1.3 % of calls, against 2.3-4.1 % for the split, so the split exceeds a same-rank, same-count edit at every rank including the hardest particle. Holding a named set of objects was compared with holding random sets of the same size: share removed $1-\#\text{changed}_\text{hold}/\#\text{changed}_\text{edit}$ is 0.00-0.50 for the 17 rank-window objects versus 0.52-0.63 on average for random 17-sets, so the hold shares do not single out the rank-window objects.
Glossary additions: rotation control (rot_r$k$); random-set hold control; rank p.
