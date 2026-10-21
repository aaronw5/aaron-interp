# E28w: which program element carries JEDI-linear's collinear-split response (claim A15; stage 4)

Script `interp/e28w_split_route_jedi.py` (job e28w-run, 1 CPU / 3 GiB, node moff.sdstate.edu, wall 1,313 s, peak memory 1,271 MiB = 41 % of request, 0.98 cores). Smoke jobs e28w-smoke (stalled at import on ucm-fiona01.ucmerced.edu, deleted), e28w-smoke2, e28w-smoke3. Figure `interp/e28w_plot.py` -> `results/jedi/e28w_split_route.png`. Results `results/jedi/e28w_split_route.json`, per-jet call changes `results/jedi/e28w_split_route_perjet{,_gbal,_hiN}.npz`, log `logs/e28w-run.log`.

## Conclusion
When one particle of a jet is split into two particles at the same position, JEDI-linear changes 2-4 % of its calls, mostly from quark to gluon, about three times as often as when the same particle is only rotated about the jet axis. This run tested four candidate routes. Two are excluded. Clamping the soft-pT cut (a per-particle switch that turns on below 11 GeV) does not remove the change; it adds to it. The particle pushed out of the 64 input slots does not explain the change either. Returning the softer particles to their unsplit slots removes about a fifth to a third of the changes for splits below the hardest particle, but it adds changes for splits of the hardest particle. Holding the two sums that count empty slots (S19, S30) removes 10-20 %, which is more than most random pairs of objects but not significant after correction. Most of the response therefore comes from the two halves themselves, spread over many objects, and its route is still not identified.

## Definitions
- **Collinear split.** The particle at pT rank $k$ (rank 0 = hardest; $k \in \{0, 5, 13, 21\}$, 21 = the first rank above 20), with $(p_T, \Delta\eta, \Delta\phi)$, is replaced by two particles $(z\,p_T, \Delta\eta, \Delta\phi)$ and $((1-z)\,p_T, \Delta\eta, \Delta\phi)$, $z \in \{0.5, 0.7, 0.9\}$, and the jet is re-sorted by pT. $M_0=\sum p_T$, $M_1=\sum p_T r$, $M_2=\sum p_T r^2$ ($r=\sqrt{\Delta\eta^2+\Delta\phi^2}$) and the jet mass are unchanged (max relative change $\le 10^{-12}$ on jets with $N<64$). The particle count $N$ goes up by one, and every particle softer than the halves moves down one input slot. For jets with $N = 64$ the input keeps the 64 hardest of the 65 particles, so the softest particle leaves the input ("overflow"). A split is applied when $N \ge k+2$.
- **Rotation control (rot).** The same particle is rotated about the jet axis by a uniform random azimuth. $N$, every rank, and $M_0$, $M_1$, $M_2$ stay fixed. This is the same-rank edit without the added particle.
- **Calls changed**: the fraction of jets whose arg-max class differs from the unedited jet. **q→g flow**: the number of jets whose call moves from quark to gluon, compared with gluon to quark by a binomial sign test.
- **Share removed** $= 1 - \#\text{changed(split + intervention)} / \#\text{changed(split)}$ on the same jets. In words, it is the fraction of the split's call changes that disappear when one element of the network is kept in its unsplit state. A negative value means the intervention adds call changes; the element's response to the split then works against the other elements' effect on the call.
- **a32**: an l1 per-particle cut $\mathrm{ReLU}(3.86 - 0.352\,p_T)$ (pT in GeV, zero angle weight). It is on for $p_T < 10.97$ GeV. Other pT ramps: a15 ($0.088\,p_T + 3.94$, always on), a21 (on below 93 GeV), a56 (on below 31 GeV). The other 9 live l1 cuts are angular half-planes with zero pT weight.
- **Per-slot terms.** The two pooled sums are exact sums of per-slot quantized terms: $P = s_0 \sum_i q_i(a_i)$ (feeds the context $g$) and $S = s_1 \sum_i q_i(u_i)$. The quantizer $q_i$ depends on the slot $i$, and its zero-bit entries define each object's pT-rank window.
- **S19** = soft multiplicity: a per-slot hinge $\mathrm{ReLU}(2.86 - 1.39\,p_T)$ over ranks 13-63, which also gives 2.86 for every empty slot. **S30** = core sum $\sum \mathrm{ReLU}(3.47 - 21\,r)$ over all slots, which gives 3.38 per empty slot. **S0** = summed pT of the 14 hardest particles. **window17** = the 17 live S objects with a rank window. **g** = the 64 jet-context units (11 live). **h4** = the first jet-level layer after S (55 live units).
- **McNemar**: exact binomial test on discordant pairs. **Holm**: step-down family-wise correction. **rank p** = (1 + number of null sets with share ≥ the named set's share) / (1 + number of null sets).

## Procedure
- Network: official JEDI-linear 3-feature n64, exact fixed-point forward (`jedi_core`). Before use, the tail S→logits used for S and h4 patches was checked to reproduce the network: calls equal 1.00000, max |Δlogit| 0. The per-slot term construction reproduces the split jet's S exactly (max |ΔS| 0 on every edit).
- Samples: **main** = `sample(per_class=1500, seed=2801)`, 7,500 jets of all multiplicities. Unedited accuracy **81.03 %**, inside the 80.6-81.3 % check band. N ≤ 60: 5,546; 61-63: 251; 64: 1,703. Two strata come from `sample(per_class=3000, seed=2802)`. **gbal** is the jets with N ≤ 60, 1,000 per class (5,000 jets, acc 79.34 %), which restores the gluon jets that E28's N ≤ 60 cut removed. **hiN** is 3,867 jets with N > 60 (g 1,692, q 321, W 127, Z 173, t 1,554; 3,336 of them with N = 64; acc 79.16 %).
- Hypotheses and the intervention for each:
  - **H_soft (a32).** In the split jet, a32 of both halves is set to the parent's a32 value (from the unedited jet at the parent's slot); every other particle keeps its own a32. The strata are 'cross' (parent a32 = 0 and at least one half has a32 > 0, so the split turns the soft cut on) and 'nocross'. The null is the same clamp on each other live l1 unit (12 units).
  - **H_rank (rank windows).** Rank-restore path patch. Each particle other than the halves contributes its per-slot term computed in its unsplit slot, and the two halves contribute their terms in their split slots. The parent's term and one empty-slot term (slot N; for N = 64 the pushed-out softest particle) are removed, so the counts of particles and empty slots equal the split jet's. 'rank_PS' restores both P (hence g) and S; 'rank_S' restores only S. Identity check: on jets where the re-sorted split moves no other particle ('noshift'), the patched S equals the split S (max |ΔS| = 0 on all such jets). The input-level variant 'noresort' (half 2 placed in slot N without re-sorting) is listed only as a reference: it puts a hard particle into a soft slot, which never happens in data, and it changes 34 % of calls at rank 0.
  - **H_slot (overflow), N = 64 jets.** (i) drop_softest: remove the softest particle of the unsplit jet, which is the particle the split pushes out. (ii) split_nooverflow: the same split applied to the jet without its softest particle, compared with that jet.
  - **H_S.** Each live S unit, the named sets S19, S0, S30, S29, S0+S19, S19+S30, S19+S29+S30 and window17, each live h4 unit, and all of g are held at the unsplit jet's values. The null is every live-S set of the same size, enumerated: 19, 171, 969 and 171 sets for sizes 1, 2, 3 and 17. For h4 singles the null is all 55 live units. Holm is applied over the 8 named sets for rank p, and over units for McNemar (edit vs edit + patch).
- Controls: rot at the same rank. Matched data pair: the closest pair of real particles in each jet (if dR < 0.01) is merged into one particle (pT summed, pT-weighted angles) and the jet is re-sorted; this is a real near-collinear pair undone, the reverse of the split. Untrained network: `Jedi(untrained=True, seed 0/1)`, calls from logits minus their unedited mean.
- Statistics: 1,000 multinomial bootstrap resamples over jets for every rate and share; exact McNemar; binomial sign test for flows. H_S scans and the l1 nulls were run at every z in main and at z 0.5 in the strata.

## Results

### 1. Split response (main, z 0.5; 95 % CI). H0: split changes no more calls than rot.
| rank k | median parent pT | split | rot | split − rot | McNemar p | q→g / g→q |
|---|---|---|---|---|---|---|
| 0 | 200 GeV | 4.35 % [3.89, 4.81] | 1.28 % | +3.07 pp | 8e-41 | 82 / 4 |
| 5 | 47 GeV | 3.11 % [2.69, 3.52] | 1.07 % | +2.04 pp | 3e-24 | 37 / 13 |
| 13 | 18 GeV | 3.00 % [2.61, 3.39] | 1.12 % | +1.87 pp | 4e-21 | 43 / 12 |
| 21 | 8.7 GeV | 2.84 % [2.45, 3.24] | 0.88 % | +1.95 pp | 4e-23 | 38 / 10 |

z 0.7 / 0.9: 3.75 / 3.05 % (rank 0) down to 2.43 / 1.63 % (rank 21); every split exceeds rot (p ≤ 5e-6). gbal (gluon-balanced): 4.44, 2.98, 2.77, 2.83 % at z 0.5, with q→g 82/3, 36/11, 38/9, 42/9. hiN: 4.60, 3.85, 3.78, 3.15 %, with balanced q/g flows (few quark jets). Accuracy changes by ≤ 0.7 pp.

### 2. H_soft: a32 clamp (share removed, main)
| rank | z 0.5 | z 0.7 | z 0.9 | fraction 'cross' (z 0.5 / 0.9) |
|---|---|---|---|---|
| 0 | 0.000 | 0.000 | −0.009 [−0.02, 0.00] | 0 / 0.11 |
| 5 | 0.000 | −0.004 | −0.061 [−0.14, 0.01] | 0.03 / 0.99 |
| 13 | −0.080 [−0.16, −0.01] | −0.176 [−0.28, −0.08] | −0.266 [−0.41, −0.14] (p 5e-6) | 0.54 / 0.75 |
| 21 | −0.104 [−0.20, −0.01] | −0.295 [−0.44, −0.18] | −0.578 [−0.82, −0.38] (p 5e-10) | 0.33 / 0.26 |

In the 'cross' stratum (a half turns a32 on) the share is −0.29 to +0.03 (e.g. rank 13 z 0.5: −0.054 [−0.14, 0.03], n 4,021); 'nocross' −0.89 to 0. gbal: rank 13 −0.22 to −0.37, rank 21 −0.16 to −0.45. Null: the other 12 live l1 units, clamped the same way, remove ≤ 0.023 (angular cuts give the parent's value to both halves already, so their clamp is nearly a no-op). The pT ramp a15 gives −0.1 to −11.7, because it rises with pT and its clamp gives both halves the parent's full value. H_soft predicted "most changes removed where a half crosses 11 GeV": **excluded**. a32's response to the halves acts against the call change.

### 3. H_rank: rank-restore path patch (share removed, 95 % CI)
| rank | main z 0.5 | main z 0.7 | main z 0.9 | gbal z 0.5 | hiN z 0.5 |
|---|---|---|---|---|---|
| 0 | −0.18 [−0.30, −0.07] | −0.15 [−0.29, −0.04] | +0.15 [0.04, 0.27] | −0.18 | −0.21 |
| 5 | +0.08 [−0.07, 0.19] | +0.33 [0.23, 0.42] (p 4e-7) | +0.31 [0.18, 0.41] | +0.15 | +0.14 |
| 13 | +0.19 [0.07, 0.30] (p 0.007) | +0.07 [−0.07, 0.19] | +0.09 [−0.04, 0.22] | +0.23 | +0.25 |
| 21 | +0.34 [0.23, 0.43] (p 3e-8) | +0.28 [0.16, 0.39] | +0.20 [0.05, 0.32] | +0.36 | +0.20 |

rank_S (S sum only restored, g from the split jet) gives similar values, 0.0-0.1 lower: −0.21, +0.07, +0.14, +0.29 at z 0.5. At rank 0, restoring the slots moves even more calls q→g (140 q→g vs 2 g→q after the patch, vs 82/4 for the split): the halves' own terms carry the q→g push, and at rank 0 the shift of the softer particles partly cancels it. 'noshift' jets (the re-sorted split moves no other particle; mostly z 0.9 at rank 21, n 757): split changes 0.5 %, rot 1.1 %. H_rank predicted "most changes removed for rank ≤ 13": **not observed**. The share is at most 0.38 in any cell and is negative at rank 0 for z 0.5 and 0.7.

### 4. H_slot: overflow at N = 64
| rank (z 0.5) | split changed (N = 64) | drop_softest changed | split changes also changed by drop | split_nooverflow changed | share removed |
|---|---|---|---|---|---|
| 0 | 4.46 % | 7.11 % [5.9, 8.4] | 0.18 | 9.98 % | −1.24 [−1.79, −0.88] |
| 5 | 4.05 % | 7.11 % | 0.32 | 8.69 % | −1.15 |
| 13 | 3.46 % | 7.11 % | 0.34 | 8.28 % | −1.39 |
| 21 | 3.35 % | 7.11 % | 0.32 | 8.40 % | −1.51 |

hiN (3,336 jets with N = 64) gives the same pattern: split 3.1-4.2 %, drop 6.8 %, split_nooverflow 8.1-9.3 %. The split's rate at N = 64 (3.4-4.5 %) is close to its rate at N ≤ 60 (2.6-4.2 %). The overflow does not add a separate effect; 18-34 % of the changed jets also change when the softest particle is dropped. With one empty slot available (the jet minus its softest particle), the same split changes 2-3 times as many calls. This is consistent with the empty-slot terms (S19, S30), which change only when the split fills an empty slot. Removing the softest particle moves calls only g→q (0 q→g vs 9 g→q, main). H_slot predicted "the overflow accounts for the effect at N > 60": **excluded**.

### 5. H_S: path patches vs size-matched random sets (main, z 0.5)
| rank | S19 | S0+S19 | S19+S30 | null q95 (pairs) | rank p (Holm 8) | window17 / null mean | g | best h4 unit (Holm p) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.095 [0.04, 0.15] | 0.110 | 0.138 [0.08, 0.19] | 0.120 | 0.035 (0.28) | 0.49 / 0.61 | 0.095 [0.02, 0.16] | h4_41 0.144 (4e-7) |
| 5 | 0.004 | 0.047 | 0.090 | 0.056 | 0.017 (0.14) | 0.28 / 0.58 | 0.090 | h4_6 0.077 (0.35) |
| 13 | 0.071 | 0.121 | 0.129 [0.05, 0.21] | 0.098 | 0.012 (0.09) | 0.17 / 0.61 | 0.152 [0.08, 0.23] | h4_49 0.089 (0.16) |
| 21 | 0.114 [0.03, 0.19] | 0.099 | 0.178 [0.07, 0.27] | 0.114 | 0.012 (0.09) | 0.24 / 0.66 | 0.188 [0.10, 0.26] | h4_6 0.074 (0.9) |

Over all 12 main (rank, z) cells, S19+S30 has uncorrected rank p 0.012-0.15 (≤ 0.035 in 9 of 12). After Holm over the 8 named sets it is 0.09-1.0, so it is not significant. Single S units cannot reach rank p < 0.05 against 19 alternatives. window17 removes less than random 17-sets in every cell (rank p 0.72-0.98), as in E28v. Holding S1 (Σ over ranks ≥ 1 of (r − 0.2)₊) or S56 increases the changes in every cell (−0.04 to −0.48). Holding g removes 0.06-0.25 (McNemar p ≤ 0.01 in 9 of 12 main cells). h4_41 removes 0.12-0.15 of the rank-0 changes in main and gbal (Holm p ≤ 0.002, above the h4 q95 of 0.06-0.08); this is not seen at other ranks. H0 for H_S ("no named element removes more than random sets"): **not excluded after Holm**. The candidate is S19+S30, the two empty-slot sums.

### 6. Controls
- **Matched data pair**: 99.97 % of jets already contain two real particles within dR < 0.01 (median closest-pair distance 0.0016). Merging that pair changes 4.59 % [4.13, 5.11] of calls, with g→q 65 vs q→g 11 (p 2e-10), the mirror image of the split's q→g (max relative mass change 0.6 %).
- **Untrained network** (centred calls). Seed 0 (calls spread over g/W/Z): splits change 14.4-14.8 % of calls, rot 1.1-6.0 %. Seed 1 (99.6 % of calls W): splits ≤ 0.4 %, rot ≤ 0.2 %. A random-weight network of this architecture is count-sensitive; the trained network's response (2.8-4.4 %) is smaller.
- **Identity checks**: tail reproduction 1.00000; split S from per-slot terms exact (0); rank-restore patch on noshift jets exact (0).

### 7. Example (figure panel a)
Gluon jet (main index 1149, 33 particles). Its rank-13 particle (pT ≈ 18.6 GeV) is split at z 0.5 into two 9.3 GeV halves at slots 17 and 18, and the particles that were at slots 14-32 move down one slot. The call goes from q (p_q 0.44, p_g 0.37) to g (p_g 0.45). With the rank-restore patch the call is q again (p_q 0.41).

## Reading (scoped to these samples and edits)
- The split response reproduces on a third sample (seed 2801; E28 seed 1202, E28v seed 2029), on a gluon-balanced sample and on jets with N > 60. Split − rot is +1.9 to +3.1 pp at z 0.5, and the q→g direction holds wherever quark jets are present.
- H_soft and H_slot are excluded. The a32 clamp and the overflow controls do not remove the changes; they increase them.
- H_rank is partly supported for splits below the hardest particle. Keeping the softer particles in their unsplit slots removes 0.08-0.38 of the changes at ranks 5-21 (CI excludes 0 in 9 of 12 main/strata cells at those ranks). At rank 0 the share is −0.15 to −0.21 for z 0.5 / 0.7. The remaining 0.6-0.9 of the response comes from the two halves' own terms at their slots: the extra particle count, the empty slot they fill, and the halved pT.
- H_S: S19+S30 (the two sums with a constant term per empty slot) is the highest-ranked named set but not significant after Holm. Each other named set is at the level of random sets.
- Competing predictions: "a32 switching on carries the response" is excluded (share ≤ 0). "Rank shift carries most of it" is excluded (share ≤ 0.38). "Overflow at N > 60 carries it" is excluded (rate at N = 64 equals rate at N ≤ 60; drop_softest is the opposite direction). "Empty-slot count via S19/S30 carries part" is consistent but not significant against random pairs (Holm p ≥ 0.09).

## Issues
1. The first smoke job stalled at import on ucm-fiona01.ucmerced.edu (14 millicores for 10 min), the second time this node did so for this study (E28). The job was deleted and resubmitted; job.sh was not changed.
2. The input-level 'noresort' and 'shiftonly' controls from the plan put a hard particle, or padding, into slots they never occupy in data (noresort changes 34 % of calls at rank 0). They were replaced by the exact per-slot path patch; noresort is kept in the JSON as a reference only.
3. The l1 clamp null is weak: the 9 angular cuts are nearly no-ops for collinear halves, and the pT ramp a15 clamp duplicates a hard particle's value. The comparison for a32 rests on its own CI and sign.
4. For size-1 S sets the smallest attainable rank p is 0.05 (19 alternatives), so single named units cannot pass Holm.
5. H_slot's split_nooverflow uses a different base jet (the jet minus its softest particle); its larger rate also contains that base jet's own position near the decision boundary.
6. Requested 1 CPU / 3 GiB (not the exempt 1/2) because E28 peaked at 2,047 MiB on 2 GiB; measured 1,271 MiB (41 %) and 0.98 cores, inside the 20-150 % / 20-200 % bands.

## Explainer (site/explain.py, Paper A, claim A15)
**Question.** Which part of JEDI-linear makes its calls change when a particle is split into two collinear particles?
**Background.** Splitting a particle $(p_T, \eta, \phi)$ into $(z p_T, \eta, \phi)$ and $((1-z) p_T, \eta, \phi)$ leaves the jet mass and every $M_\kappa = \sum_i p_{T,i} r_i^\kappa$ unchanged. JEDI-linear still changes 2-4 % of its calls (E28, E28v). The network sees particles in pT-ordered slots. A split (i) creates two softer particles, which can switch on the soft cut $a_{32} = \mathrm{ReLU}(3.86 - 0.352\,p_T)$ (on below 11 GeV); (ii) moves every softer particle down one slot, across the rank windows of the second-round objects; (iii) at $N = 64$ pushes the softest particle out of the input; (iv) fills one empty slot, which lowers the empty-slot terms of $S_{19}$ and $S_{30}$.
**Method.** On 7,500 test jets (plus 5,000 gluon-balanced and 3,867 high-multiplicity jets), split the particle of rank 0, 5, 13 or 21 at $z$ = 0.5, 0.7, 0.9. Then undo one element at a time inside the network: set $a_{32}$ of the halves to the parent's value; rebuild the pooled sums $P = s_0\sum_i q_i(a_i)$, $S = s_1\sum_i q_i(u_i)$ with every other particle's term taken from its unsplit slot; compare with dropping the softest particle; hold S objects, h4 units or $g$ at unsplit values, against all size-matched sets of live S objects. Share removed $= 1 - \#\text{changed}_\text{intervention}/\#\text{changed}_\text{split}$.
**Result.** (figure `e28w_split_route.png`) Splits change 2.8-4.4 % of calls against 0.9-1.3 % for rotating the same particle. Clamping $a_{32}$ removes none (share 0 to −0.58). Restoring the slots of the softer particles removes 0.08-0.34 at ranks 5-21 and −0.18 at rank 0. The overflow at $N=64$ explains none (rate at $N=64$ equals the rate at $N\le60$). Holding $S_{19}+S_{30}$ removes 0.09-0.18, above 95 % of random pairs in some cells but not after Holm correction.
**Checks.** The tail and per-slot reconstructions reproduce the network exactly. On jets where the split moves no other particle, the slot-restore patch is an identity. Merging a real pair of particles closer than dR 0.01 (present in almost every jet) changes 4.6 % of calls in the opposite direction (gluon → quark).
**Why it matters.** The network's lack of collinear safety is not one switch: neither the soft cut, nor the slot overflow, nor a single named object carries it. About a fifth to a third comes from the rank shift of the softer particles. The rest comes from the two halves themselves through many objects, with the empty-slot sums as the strongest single candidate.

Glossary additions: rank-restore path patch (per-slot terms); overflow (N = 64); share removed (negative = intervention adds changes); rank p against enumerated size-matched sets; matched data pair (merge of a real pair at dR < 0.01); a32 (soft-pT cut, on below 11 GeV).

Conclusion (table row): Splitting one particle into two collinear ones changes about 3 % of JEDI-linear's calls, mostly from quark toward gluon. Neither the 11 GeV soft cut nor the particle pushed out of the 64 slots carries this. Keeping the softer particles in their original slots removes about a fifth to a third of the changes for particles below the hardest. The rest comes from the two halves themselves, spread over many objects, with the two empty-slot sums as the leading but not significant candidate.
