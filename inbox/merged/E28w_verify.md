# E28w verification: route of JEDI-linear's collinear-split response

Script `interp/e28w_verify.py`, a new script that imports the worker's edit and patch functions from `interp/e28w_split_route_jedi.py`. Job e28w-verify: 1 CPU / 2 GiB, node moff.sdstate.edu, wall 288 s, peak 1,390 MiB, 0.40 cores. Results are in `results/jedi/e28w_verify.json` and the log is in `logs/e28w-verify.log`. The log shows a CUDA-plugin traceback at import; it is the usual non-fatal message under JAX_PLATFORMS=cpu, and the job exited 0.

## Conclusion
On a new set of 6,000 jets, splitting one particle into two particles at the same position again changes about 3-5 % of JEDI-linear's calls, against about 1 % for rotating that particle, and the calls again move from quark toward gluon. The main route findings also hold. The 11 GeV soft cut does not carry the effect. Putting the softer particles back in their original input slots removes about 0.3 of the changes for splits of the 14th and 22nd hardest particles. Two statements need correcting. First, the two sums that count empty slots (S19, S30) were the pre-named candidate, and on this new sample they remove more changes than all but 1-2 % of random pairs at every split rank, so the earlier "not significant" verdict comes from the first sample only. Second, part of what the worker attributes to "the halves' own terms" is an effect of the particle count alone: adding one 0.3 GeV particle, with nothing else moved, changes 1.25 % of calls, against 0.17 % for adding the same pT to the hardest particle. It is still unknown which objects carry the rest, since no pair of objects reproduces more than 0.45 of the changes.

## Code review of `e28w_split_route_jedi.py`
- `split_ext` adds half 2 in an appended 65th slot, re-sorts with a stable sort and keeps 64 slots. The halves' slots `d1`/`d2` come from the inverse permutation. This is correct. At z 0.5 the two halves have equal pT and the stable sort keeps half 1 first.
- `rank_patch`: the weight `W` removes the parent's unsplit term and one empty-slot term (slot N, or slot 63 when N = 64 and half 2 survives). When half 2 is pushed out (`e = -1`), only the parent's term is removed. The term count is 64 in every case. The quantizers are applied per slot (`L['qs'].iq`, `L['qs1'].iq`) before summing, so the patch uses the same per-slot zero-bit masks as the network. The scale and the `dd` input quantizer are applied as in `jedi_core`. The worker's checks (split S rebuilt from its terms, and the identity on noshift jets) are exact, and both are exact again here (`check` 0.0 in every cell).
- The l1 clamp overrides `a` after l1, so it acts on pooling and on l2, as intended. The clamp "both halves := parent's a32" does not restore the pooled a32 sum, because it doubles the parent's value. I added a sum-preserving variant (half 1 := parent's a32, half 2 := 0).
- `sh_S` / tail: holding all live S at unsplit values removes 1.000 of the changes in every cell, and so does holding all 64 S units. The tail and hold machinery is therefore exact.
- `drop_softest` is applied to every jet, but it is used only on N = 64 rows. This is correct.
- I found no bug in stages, masks, quantizers or padding. Padding slots are counted as empty, not as particles, because `nreal` uses pT > 0. Padding enters the network as (0 − shift)/scale, which gives the nonzero empty-slot terms.
- Report error: section 4 of `inbox/E28w_report.md` says "Removing the softest particle moves calls only g→q (0 q→g vs 9 g→q)". The largest flow of drop_softest is g→t: 76 in main (g→W 20, g→q 9) and 48 in this sample (g→q 5).

## Replication: `sample(per_class=1200, seed=3907)`, 6,000 jets
Accuracy is 81.13 %. The calls from the worker's `Net.fwd` and from `jedi_core.Jedi.run` agree on 100 % of jets, both unedited and split. N ≤ 60: 4,445 jets; 61-63: 204; 64: 1,351. Edit: exact collinear split at z 0.5, plus z 0.9 at rank 21. Control: rotation of the same particle about the jet axis (rot). The 95 % intervals are from 1,000 bootstraps over jets.

| rank | split changed | rot | split − rot (95 % CI) | McNemar p | q→g / g→q | rank_PS share removed | a32 clamp (both = parent) | a32 clamp (sum-preserving) | S19+S30 held (rank p vs 171 pairs) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.78 % [4.28, 5.35] | 1.12 % | +3.7 pp [3.2, 4.2] | 1e-41 | 68 / 7 | −0.07 [−0.18, 0.03] (p 0.21) | 0.000 | 0.000 | 0.153 (0.023) |
| 5 | 3.25 % [2.82, 3.70] | 0.93 % | +2.3 pp [1.9, 2.7] | 1e-25 | 29 / 13 | +0.11 [−0.03, 0.25] (p 0.13) | 0.005 | 0.005 | 0.113 (0.012) |
| 13 | 3.10 % [2.68, 3.52] | 0.90 % | +2.2 pp [1.8, 2.6] | 5e-24 | 32 / 11 | +0.28 [0.15, 0.41] (p 4e-4) | −0.06 [−0.14, 0.01] | −0.16 [−0.26, −0.07] | 0.146 (0.012) |
| 21 | 2.77 % [2.38, 3.23] | 1.04 % | +1.7 pp [1.3, 2.2] | 5e-15 | 31 / 7 | +0.29 [0.17, 0.41] (p 6e-5) | −0.09 [−0.21, 0.02] | −0.39 [−0.57, −0.24] | 0.191 (0.018) |
| 21, z 0.9 | 1.55 % | 1.04 % | +0.5 pp [0.1, 0.9] | 0.009 | 20 / 6 | +0.05 [−0.15, 0.21] | −0.65 [−0.96, −0.42] | −1.10 [−1.54, −0.78] | 0.080 (0.11) |

- **Split vs rot (H0: split changes no more calls than rot).** H0 is rejected at every rank, and the q→g direction holds (sign test p ≤ 0.02). The rates match E28w main (4.35 / 3.11 / 3.00 / 2.84 %).
- **H_rank.** The rank-restore patch rebuilds P and S with every other particle's term from its unsplit slot. It removes 0.28-0.29 of the changes at ranks 13 and 21, against 0.19 / 0.34 in E28w. At rank 5 the share is +0.11 and its CI includes 0. At rank 0 the share is −0.07 [−0.18, 0.03] (McNemar p 0.21). The negative rank-0 share in E28w (−0.18 [−0.30, −0.07]) is therefore not reproduced at significance on this sample, although its sign agrees.
- **H_soft.** Neither clamp removes changes at any rank. The sum-preserving clamp sets the a32 column of P exactly to its unsplit value, and it adds changes: −0.16 and −0.39 at ranks 13 and 21 (z 0.5), and −1.10 at rank 21 (z 0.9). The worker's statement that the clamp "adds 8-58 %" holds at z 0.9 (−0.65, p 4e-10). At z 0.5 the CIs of the both-parent clamp include 0 here (−0.06, −0.09). Exclusion of H_soft: reproduced.
- **H_slot, rank 13, N = 64 (n 1,351).** Split 3.77 %, drop softest 5.18 %, split of the jet without its softest particle 7.18 %. This reproduces the E28w pattern (3.46 / 7.11 / 8.28 %). The split rate by N is 4.4-7.4 % at N 61-63, against 2.4-4.6 % at N ≤ 60, which again points to the last empty slots.
- **S19+S30 held (H0: this pair removes no more than a random pair of live S units).** S19+S30 was named before E28w ran and is the only pair tested here. On the fresh sample its rank p against all 171 live-S pairs is 0.023 / 0.012 / 0.012 / 0.018 at ranks 0 / 5 / 13 / 21 (z 0.5). With Holm over the 4 ranks, each is 0.047. This confirmation differs from the E28w reading "no named set passes Holm"; that reading used Holm over 8 named sets on the discovery sample. The effect is small: 0.11-0.19 of the changes.
- **Free-pair sufficiency (new).** All live S units are held except one pair, and the metric is the fraction of the split's changes reproduced on the same jet with the same new call. S29+S30 reproduces 0.24-0.35 of the changes (rank p 0.04-0.25 among 171 pairs). The best pairs reach 0.41-0.45 and always include S1, S19, S29, S30 or S43. No pair carries a majority of the changes, which supports "spread over objects".
- **Merge of the closest real pair** (100 % of jets have a pair at dR < 0.01, median 0.0015). Merging it changes 3.82 % [3.37, 4.32] of calls, g→q 45 vs q→g 6 (p 2e-8), against 4.59 % in E28w. The result is reproduced.

## Missing control (new): particle count without any rank shift, on N ≤ 63 jets (n 4,649)
The rank-restore patch keeps the split's particle count and empty-slot count, so its residual mixes "the halves' terms" with "one more particle and one fewer empty slot". I ran four edits that move no other particle:
- **split_soft**: an exact collinear split of the softest particle at z 0.5. The halves land in slots N−1 and N, and nothing else moves (checked on all jets).
- **rot_soft**: the softest particle rotated about the jet axis.
- **add_soft**: one extra particle with pT = 0.999 × the softest pT (median 0.34 GeV, which raises $M_0 = \sum p_T$ by 0.03 %), placed at the (Δη, Δφ) of a random particle of the jet. It lands in slot N.
- **bump_hard**: the same pT added to the hardest particle. This matches the change in $M_0$ while keeping N and every rank fixed.

| edit | calls changed (95 % CI) | q→g / g→q | S19+S30 held: share removed (rank p) |
|---|---|---|---|
| split_soft | 2.06 % [1.68, 2.51] | 23 / 4 (p 3e-4) | 0.125 (0.047) |
| rot_soft | 1.55 % [1.23, 1.92] | 16 / 8 | n/a |
| add_soft | 1.25 % [0.95, 1.58] | 13 / 6 (p 0.17) | 0.397 (0.006) |
| bump_hard | 0.17 % [0.06, 0.30] | 1 / 0 | n/a |

- split_soft − rot_soft = +0.5 pp [0.13, 0.95] (McNemar p 0.02). A collinear split with no rank shift changes 2.1 % of calls. This is close to the rank-13/21 split rate after the rank-restore patch (3.1 % × (1 − 0.28) ≈ 2.2 %).
- add_soft − bump_hard = +1.1 pp [0.8, 1.4] (McNemar 53 vs 3, p 8e-13). One added soft particle moves calls about seven times as often as the same pT added to the hardest particle. The two S sums with an empty-slot term, S19 and S30, carry 0.40 of those changes, which is above every other pair but one (rank p 0.006).
- Of the jets whose call changes under the rank-k split, 15-24 % also change under add_soft and 20-30 % under rot_soft, but only 2-3 % under bump_hard. Many of the changed jets are near a decision boundary and change under any small edit of the soft particles.

H0 for this control is "adding a soft particle in an empty slot changes no more calls than the M0-matched hard bump". It is rejected. The particle count (empty-slot filling) is therefore a measured component of the residual that the worker attributes to "the halves' own per-slot terms". add_soft's own q→g direction is not significant (13/6).

## Figure
`results/jedi/e28w_split_route.png`. Panel (a) shows the split on a real gluon jet. Panel (b) shows shares with CIs and null q95 ticks, and its values match the JSON. Panel (c) shows split, rot and merge rates. The axes are in physical units, and the figure is readable.

## Corrected reading
An exact collinear split changes 2.8-4.8 % of JEDI-linear's calls, against 0.9-1.3 % for rotating the same particle, on two independent samples (seeds 2801, 3907). The calls move quark → gluon. Merging a real dR < 0.01 pair, present in about every jet, changes 3.8-4.6 % of calls gluon → quark. Clamping the soft cut a32 removes no changes, including a variant that restores a32's pooled sum exactly, so a32 is not the route. The overflow at N = 64 is not the route either. Rebuilding the pooled sums with the softer particles in their unsplit slots removes 0.19-0.34 of the changes at ranks 13 and 21. At rank 5 the share is not distinguishable from 0 on the fresh sample. At rank 0 it is −0.18 on seed 2801 and −0.07 (CI includes 0) on seed 3907. A split of the softest particle, which shifts nothing, still changes 2.1 % of calls (rot 1.5 %). Adding one 0.3 GeV particle changes 1.25 % against 0.17 % for a matched hard-pT bump. Part of the non-shift residual is therefore the particle count itself. Holding the two sums with an empty-slot term, S19+S30, removes 0.11-0.19 of the split's changes. As a pre-named pair on the fresh sample this is above 98 % of random pairs at all four ranks (Holm over ranks 0.047), and it removes 0.40 of the add-particle changes. The rest of the response is spread over objects: no pair of S objects reproduces more than 0.45 of the changes.

Conclusion: (see top).
