# B10: if-statement programs for non-JEDI hls4ml taggers (method generality)

Node B10 (stage 3; claims B2n, B6t, B; exploratory). Module `interp/b10_programs_small_models.py`. Results:
`results/jedi/b10.json` (aggregate), `b10_net_<net>.json/.npz` (per network: surrogate metadata, calls on dev / test / edited
jets), `b10_prep.json` (sample, edits, half-plane lines), `b10_torch.json`, figure `results/jedi/b10.png`. Logs: `logs/b10-*.log`.

## Conclusion

Short if-statement programs, about 7,000 bits (16 hinge terms per class), give the same call as the network on 87–92 % of
test jets. This holds for Deep Sets, Linformer, transformer and Particle Flow Network taggers, as it does for JEDI-linear, and
at matched size these programs agree more often with the network than gradient-boosted trees or MARS on standard jet
observables. However, a program fitted to the truth labels alone already agrees with every trained network on 85–90 % of jets.
On the shift edit, a network's own programs reproduce more of its call changes than other networks' programs in only 2 of 9
networks once the variation between repeated fits is taken into account. Adding azimuthal inputs lets programs respond to the
mass-lowering turn, but only weakly, and they respond about equally to a control rearrangement that keeps the mass fixed. What
is uncertain: how much of the program agreement describes a given network rather than the task all these networks solve.

## Question

Does the B9 recipe (MARS if-statement programs over a fixed jet-feature pool) describe small taggers other than JEDI-linear
(attention models, permutation-invariant sums) as well as it describes JEDI-linear? Is the program for a network specific to
that network, so that its own program beats programs fitted to other networks and to the truth labels? With azimuthal inputs
added, can a program reproduce the network's response to an azimuthal turn? B9 could not express that edit.

## Procedure

**Networks** (existing checkpoints only, no training). Every network sees the same 64-slot (pT, Δη, Δφ) jets.
| id | network | accuracy on the 20,000-jet sample |
|---|---|---|
| jedi_off | official JEDI-linear 3-feature n64 (reference row; `jedi_core.Jedi`) | 81.23 % (check range 80.6–81.3 %: inside) |
| ds_s0 / ds_s1 | Deep Sets `dsets-d64-n64` seed 0 / 1 (train/models.py, torch) | 80.97 / 81.11 % |
| linf_s0 / linf_s1 | Linformer `linf-2b-d32-k8-n64` seed 0 / 1 | 81.29 / 80.94 % |
| tf_s0 / tf_s1 | transformer `tf-2b-d32-n64` seed 0; `tf-2b-d32-n64-p30` seed 1 | 76.72 / 76.72 % |
| pfn_s0 / pfn_s1 | X9 PFN `pfn_64-64-32` seed 0 / 1 (exact numpy forward) | 81.38 / 81.23 % |
| *_u0 | one untrained redraw per architecture (JEDI: redraw_untrained seed 0; torch: default init, torch.manual_seed(5000); PFN: he_uniform, zero biases, rng 317000) | 19.6–20.6 % |

The torch-model test accuracies on the STATUS.md archive are 80.64 / 80.81 (ds), 81.07 / 80.76 (linf) and 76.19 / 76.24 (tf).
The 6,000 test jets here give 80.82 / 80.67, 80.95 / 80.63 and 75.97 / 76.18.

**Sample.** `interp.jedi_core.sample(per_class=4000, seed=3170)` gives N = 20,000 jets. A random permutation (rng 3170) splits
them into fit0 / fit1 / fit2 (4,000 jets each, disjoint, for three repeated program fits), dev (2,000, for model selection) and
test (6,000; every reported number comes from these jets).

**Feature pool (140 inputs, the same for every network).**
- B9 pool (98 inputs): sums over pT-rank windows of pT, counts, pT·r², r², r, pT·r and (r−0.2)₊, plus M0 = ΣpT. Here r is the
  angular distance of a particle from the stored jet axis.
- Azimuthal pool (42 inputs, new):
  - Complex moments C_n = Σ pT_i z_i^n / Σ pT_i, with z_i = Δη_i + iΔφ_i and n = 1..4, taken over the 14 hardest particles and
    over all particles. Each gives |C_n|, Re C_n and Im C_n (24 inputs). |C_1| is the pT-weighted centroid offset |M1|/M0;
    C_2..C_4 measure the quadrupole and higher azimuthal structure about the axis.
  - The JEDI-linear l1 half-plane cuts: the nine units a29, a6, a40, a57, a3, a43, a30, a36 and a46, each the line
    α_η Δη + α_φ Δφ + β = 0 taken from the official l1 kernel in physical units, with zero pT weight and offset ≤ 0.008 rad.
    For each cut, the pT sum and the particle count on its positive side (18 inputs).
- Standard observables (10, used only by the baselines): m, m_SD (soft-drop mass), τ21 and τ32 (N-subjettiness ratios), D2
  (energy-correlation ratio), N, ΣpT, ΣpT rⁿ for n = 1, 2, 3.

**Program (if-statement program).** This is the B9 recipe: MARS (multivariate adaptive regression splines, a greedy sum of
hinge terms max(0, x − t) or max(0, t − x) and their pairwise products) fitted to each of the network's 5 logits, with a forward
path of 32 terms. Nested programs with K = 4, 8, 16 and 32 terms per class are refitted by least squares, and the call is the
argmax. Each network gets three fits, on fit0, fit1 and fit2.

Description length DL (bits), the program size: per class, 32 for the intercept; per term, 32 for the coefficient; per factor,
log2(140) + 2 + 32 for the knot (no knot term for a linear factor). K16 programs are 6,230–7,212 bits.

**Baselines (fit0), compared at matched bits.** At matched bits, the baseline is the family member with DL ≤ the program's DL
and the highest dev same-call.
- MARS on the standard observables (K = 2..32).
- HistGradientBoosting classifier (GBDT) on the standard observables, and GBDT on the full pool (inputs quantised to 2^q
  levels; DL = B9 `hgb_bits_q`).
- CART (one decision tree) on the pool, depth 2–12.
- Lasso per logit on the standardised pool.
- B9-pool program: the same MARS recipe on the 98 B9 inputs only.

**Controls (K16, fit0).**
- Truth program: MARS fitted to log(0.95·onehot(label) + 0.01), never to a network.
- Shuffled-logit program: fit rows' logits permuted (rng 3176).
- Random-feature program: MARS on 6 pool inputs drawn at random (rng 3177).

**Edits on the test jets.** Padding is never moved.
- **shift**: every real particle's Δη + 0.06 rad. This is a rigid translation off the stored axis: the invariant mass stays
  the same and M2 = ΣpT r² about the axis rises. N = 6,000 jets.
- **turn10**: the 14 hardest particles (pT ranks 0–13) are turned in azimuth about the stored axis, toward the direction of
  the pT-weighted centroid. All 14 move by a common fraction of their angle gap, solved per jet so that the jet mass m falls by
  10 % (median Δlog m = −0.1054). Each particle keeps its pT and r, so M0, M2 and every B9-pool input are unchanged (max
  relative ΔM2 0.0). This is the E27w arm-B / E41 turn.
- **ctrl10**: control for turn10 (E41v). The same 14 particles are turned by the same |Δθ_i|, each in a random direction
  (rng 3171). A minimum-norm azimuth correction then restores the vector M1, and a common rescale keeps the total displacement
  equal to the turn's. pT, r, M0, M1, M2 and m are held (max |Δlog m| 0.001), so only the azimuthal arrangement changes.
  turn10 and ctrl10 use the same 2,814 jets: those where both converge (turn alone: 3,786).
- **soft**: one 0.5 GeV particle is added at r = 0.3 with a random azimuth (rng 3175). The other particles get one common scale
  and boost so that the jet four-momentum is exact. N = 4,669 jets (those with N ≤ 63).

**Metrics.**
- Same call: the share of test jets where the program's argmax equals the network's.
- κ (Cohen kappa): agreement beyond chance. For 5-class calls; for edits, between the network-flip indicator and the
  program-flip indicator. κ = 0 means the program's flips are unrelated to the network's; κ = 1 means identical flip sets.
- Recall on an edit: the share of jets whose call the network changes where the program also changes its call to the same new
  class.
- Fit-to-fit range: max − min over the three repeated fits.
- Transfer: network A's fit0 K16 program scored against network B's calls.

**Statistics.** 1,000-draw jet bootstrap over test rows, with edited sets resampled through their base rows. Bootstrap p has a
floor of 0.001, so Holm p has a floor of 0.009 for 9 networks. Paired comparisons use exact McNemar (program vs matched-bits
baseline; own fit0 vs the best donor of another architecture). Holm correction runs across networks and baselines per (K,
metric) for matched comparisons, and across networks for transfer.

**Jobs** (1 CPU / 2 GiB each).
| job | wall time | notes |
|---|---|---|
| b10-prep | 372 s | |
| b10-torch (conda) | 93 s | |
| b10-jedi, b10-pfn, b10-ds, b10-linf, b10-tf | 580–1,315 s each | peak 274–450 MiB |
| b10-agg | 190 s | |

A smoke chain (b10s-*) ran first. Its first prep pod stalled in `ceph_mdsc_wait_request` (D state) on
k8s-haosu-10.sdsc.optiputer.net, so that node was added to the job.sh exclusion list. Python imports on the volume took about
13 minutes in the smoke rerun because CephFS metadata operations were slow (9 s for 862 files). No code bug was found. The
figure's x tick labels were changed and the figure re-rendered locally from b10.json (`--plot` mode). No number changed.

## Results

### 1. Program agreement and size

Same call on the 6,000 test jets. The first three columns are the mean over the 3 fits, with the K16 fit-to-fit range in
brackets.
| network | K4 (~1.6 kbit) | K8 (~3.3 kbit) | K16 (~6.8 kbit) [range] | K32 (~13.7 kbit) | best member at DL ≤ 7 kbit, per fit |
|---|---|---|---|---|---|
| JEDI-linear | 79.5 | 88.0 | 89.8 [0.1] | 90.8 | 89.9 / 89.8 / 89.9 |
| Deep Sets s0 | 87.5 | 90.2 | 91.8 [0.8] | 92.4 | 92.1 / 91.3 / 92.1 |
| Deep Sets s1 | 86.1 | 90.6 | 92.2 [0.6] | 92.9 | 92.4 / 91.8 / 92.4 |
| Linformer s0 | 81.7 | 88.4 | 90.0 [0.1] | 91.2 | 88.6 / 90.0 / 90.0 |
| Linformer s1 | 83.6 | 88.7 | 90.3 [0.1] | 91.0 | 90.3 / 90.2 / 90.3 |
| transformer s0 | 83.2 | 85.6 | 87.2 [0.5] | 88.0 | 86.1 / 87.4 / 85.4 |
| transformer s1 | 79.8 | 85.6 | 86.9 [0.4] | 87.9 | 85.7 / 85.8 / 87.0 |
| PFN s0 | 82.4 | 88.1 | 89.6 [0.9] | 90.7 | 89.6 / 89.2 / 90.1 |
| PFN s1 | 76.8 | 88.6 | 89.8 [0.8] | 91.2 | 90.3 / 89.7 / 88.7 |

The K16 bootstrap CIs are ±0.7–0.8 pp, for example JEDI-linear 89.85 [89.1, 90.6] and transformer s0 86.92 [86.1, 87.8]. K16
κ (5-class) is 0.84–0.91.

Controls at K16 (fit0), across trained networks:
| program | same call | κ |
|---|---|---|
| truth program | 85.3–90.3 % | 0.82–0.88 |
| random-feature program | 69.2–72.7 % | 0.62–0.66 |
| shuffled-logit program | 19.1–24.2 % | ≤ 0.05 |

Untrained redraws give a near-constant call (one class on 91–100 % of jets), so any program matches them on 95.6–100 % of jets
(κ 0–0.69). They are left out of every summary below.

### 2. Matched bits

Program fit0 K16 minus the best baseline with DL ≤ the program's, in pp of same call. A positive number means the program
agrees more often.
| baseline | range of differences over the 9 trained networks | significant (Holm p < 0.05) |
|---|---|---|
| MARS on standard observables | +1.6 to +3.1 | 9 / 9 |
| GBDT on standard observables | +0.7 to +2.7 | 6 / 9 (not significant: Linformer s0, Linformer s1, PFN s0) |
| GBDT on the pool | +0.7 to +4.6 | 7 / 9 (not significant: both transformers) |
| CART | +2.3 to +4.4 | 9 / 9 |
| Lasso | +25 to +28 | 9 / 9 |
| B9 pool alone | −0.5 to +1.7 | program ahead in 3 / 9 (Linformer s1, transformer s1, PFN s0, where the B9-pool member chosen was K8) |

At K32 the program is ahead of GBDT on standard observables in 8 / 9 networks (+0.6 to +2.9) and of GBDT on the pool in 7 / 9.

Shift recall at matched bits (K16): the program is ahead of MARS and GBDT on standard observables in 9 / 9 networks
(+8 to +63 pp), and of GBDT on the pool in 7 / 9 (Linformer s0: −7.0 pp).

### 3. Edits: recall of the network's call changes (K16; fits 0 / 1 / 2)

| network | network flips: shift / turn10 / ctrl10 / soft | shift recall | turn10 recall (κ) | ctrl10 recall | soft recall (fit0) |
|---|---|---|---|---|---|
| JEDI-linear | 3390 / 130 / 81 / 115 | 59.0 / 59.6 / 57.9 | 2.3 / 8.5 / 3.1 (0.05–0.09) | 2.5 / 2.5 / 2.5 | 2.6 |
| Deep Sets s0 | 3781 / 139 / 72 / 129 | 55.4 / 63.9 / 57.4 | 6.5 / 7.9 / 10.1 (0.10–0.15) | 11.1 / 5.6 / 6.9 | 8.5 |
| Deep Sets s1 | 3747 / 108 / 47 / 102 | 66.1 / 66.1 / 60.8 | 7.4 / 2.8 / 4.6 (0.07–0.11) | 0.0 / 6.4 / 0.0 | 8.8 |
| Linformer s0 | 3330 / 203 / 41 / 100 | 30.9 / 35.4 / 26.3 | 5.9 / 1.5 / 4.4 (0.12–0.13) | 4.9 / 0 / 0 | 5.0 |
| Linformer s1 | 3622 / 204 / 40 / 114 | 59.4 / 55.9 / 57.8 | 3.4 / 10.8 / 5.4 (0.07–0.13) | 2.5 / 2.5 / 0 | 4.4 |
| transformer s0 | 3117 / 288 / 108 / 131 | 52.6 / 54.3 / 51.9 | 7.6 / 6.6 / 4.5 (0.05–0.11) | 3.7 / 3.7 / 2.8 | 8.4 |
| transformer s1 | 3609 / 254 / 68 / 135 | 49.1 / 43.4 / 44.6 | 5.9 / 6.3 / 5.5 (0.09–0.10) | 8.8 / 4.4 / 1.5 | 3.7 |
| PFN s0 | 3925 / 163 / 81 / 68 | 53.7 / 46.8 / 48.2 | 5.5 / 2.5 / 3.1 (0.04–0.19) | 3.7 / 6.2 / 7.4 | 4.4 |
| PFN s1 | 3784 / 198 / 86 / 77 | 59.0 / 56.1 / 60.7 | 15.7 / 10.6 / 11.6 (0.18–0.24) | 11.6 / 5.8 / 5.8 | 7.8 |

- **Network flip rates.** The networks change 52–65 % of calls under the shift. They change 4.6–10.2 % under turn10 and
  1.4–3.8 % under ctrl10. The turn changes more calls than the mass-preserving control in every network, as E41v found for
  JEDI-linear (5.1 vs 2.8 %).
- **Turn edit.** The B9-pool program cannot respond to turn10 or ctrl10, because none of its inputs changes; its recall is 0
  by construction. With the azimuthal pool, turn10 κ is 0.04–0.24 (recall 1.5–15.7 %). The controls give: random-feature
  κ ≤ 0.04, shuffled κ ≤ 0.016 (shuffled recall 8–37 % is a by-product of frequent random flips, with κ ≈ 0), truth program
  κ 0.02–0.12. Programs give similar recall on ctrl10 (0–11.6 %) as on turn10, so what they reproduce is the response to a
  change in azimuthal arrangement, not a mass-specific response.
- **Fit-to-fit spread.** The spread in shift recall is 1.7–9.1 pp (Linformer s0 26.3–35.4, Deep Sets s0 55.4–63.9), which
  reproduces the B9v observation (up to 10 pp). On natural jets the spread in same call is ≤ 0.9 pp.

### 4. Transfer: own programs vs other networks' programs vs the truth program (K16)

In the table below, "own" is the mean over the 3 own fits and "transfer" is the mean over the 8 other trained networks' fit0
programs.
| network B | natural: own / transfer / truth | own − transfer (Holm p) | shift recall: own [fit range] / transfer / truth | own − transfer, 95 % CI (Holm p) | exceeds fit range? |
|---|---|---|---|---|---|
| JEDI-linear | 89.8 / 88.2 / 88.2 | +1.6 (0.009) | 58.8 [1.7] / 55.8 / 49.4 | +3.0 [+2.1, +3.9] (0.009) | yes |
| Deep Sets s0 | 91.8 / 90.1 / 90.2 | +1.7 (0.009) | 58.9 [8.5] / 62.2 / 51.8 | −3.3 [−4.0, −2.7] (0.009) | no (own lower) |
| Deep Sets s1 | 92.2 / 90.3 / 90.3 | +1.9 (0.009) | 64.3 [5.3] / 61.9 / 53.0 | +2.5 [+1.7, +3.3] (0.009) | no |
| Linformer s0 | 90.0 / 88.8 / 88.6 | +1.2 (0.009) | 30.8 [9.1] / 36.0 / 25.4 | −5.2 [−6.1, −4.3] (0.009) | no (own lower) |
| Linformer s1 | 90.3 / 88.8 / 88.6 | +1.5 (0.009) | 57.7 [3.5] / 57.6 / 49.8 | +0.1 [−0.8, +1.0] (0.80) | no |
| transformer s0 | 87.2 / 84.9 / 85.3 | +2.3 (0.009) | 52.9 [2.5] / 48.4 / 42.9 | +4.6 [+3.5, +5.5] (0.009) | yes |
| transformer s1 | 86.9 / 85.1 / 85.3 | +1.8 (0.009) | 45.7 [5.7] / 51.6 / 49.5 | −5.9 [−6.9, −5.0] (0.009) | no (own lower) |
| PFN s0 | 89.6 / 89.4 / 89.2 | +0.3 (0.20) | 49.6 [6.9] / 53.6 / 43.3 | −4.0 [−4.9, −3.1] (0.009) | no (own lower) |
| PFN s1 | 89.8 / 88.6 / 88.6 | +1.2 (0.009) | 58.6 [4.6] / 59.4 / 50.1 | −0.9 [−1.6, −0.2] (0.046) | no |

Transfer minus truth on shift recall is +2.0 to +10.6 pp in 9 / 9 networks (Holm p 0.009). On natural jets transfer and truth
are equal (−0.4 to +0.2 pp, Holm p 1).

The jet-bootstrap CIs do not contain the fit-to-fit variation. On shift recall, the variation between repeated fits of one
network's program is often larger than the gap between own and transfer.

McNemar on shift hits, own fit0 vs the best donor of another architecture: the donor reproduces more of the network's shift
flips in 5 / 9 networks (Holm p ≤ 0.002: Deep Sets s0, Linformer s0, transformer s1, PFN s0, PFN s1). Own fit0 reproduces
more in 1 / 9 (transformer s0, p 4e-4), and JEDI-linear's best donor (Linformer s0) reproduces more (268:374, p 2e-4).

### 5. Predictions vs results

| prediction | thesis (short programs describe small set taggers generally) | alternative (the recipe works because of JEDI's rank windows and pure cuts) | measured |
|---|---|---|---|
| same call at ≤ ~7 kbit | ≥ 85 % for Deep Sets, PFN, Linformer | Linformer / transformer well below 85 % | 88.6–92.4 % for Deep Sets, PFN, Linformer; 85.4–87.4 % for transformers. The alternative's prediction is excluded |
| program vs GBDT on standard observables, matched bits | program ahead | no gain | ahead by +0.7 to +2.7 pp, significant in 6 / 9. The alternative's "no gain" is excluded for 6 networks; for Linformer and PFN s0 the gain is not separable |
| shift recall | own > transfer > truth, beyond fit spread | own = transfer within fit spread | transfer > truth in 9 / 9. Own > transfer beyond the fit range only for JEDI-linear and transformer s0; own < transfer in 5 / 9. The thesis's "own > transfer" holds for 2 / 9; the alternative's prediction matches 7 / 9 |
| turn recall | > 0 with the azimuthal pool | – | recall 1.5–15.7 %, κ 0.04–0.24 vs ≤ 0.04 for random-feature and shuffled controls. Recall on the mass-preserving control is of the same size |

## Reading (scoped to what was measured)

- **Same call.** At about 7 kbit, MARS programs over a 140-input pool give the same call as each of the eight non-JEDI taggers
  on 85–92 % of test jets. The fidelity-vs-size curve has the same shape as JEDI-linear's, lower for the two transformers. The
  three repeated fits differ by ≤ 0.9 pp on natural jets.
- **The agreement is mostly shared.** The truth-label program, fitted without any network, agrees with each network on
  85–90 % of jets. Each network's own program adds 0.3–2.3 pp over it (Holm p 0.009 in 8 / 9). Programs fitted to other
  networks are no closer to a network than the truth program on natural jets. The own-program margin is a small part of the
  agreement.
- **Shift response.** Programs fitted to any network reproduce 2–11 pp more shift-induced call changes than the truth program.
  A network's own programs do not, in general, reproduce its shift response better than other networks' programs (2 / 9
  beyond the fit range, 5 / 9 lower). Linformer s0 is an exception: its shift recall is 26–35 % against 50–66 % for the other
  trained networks.
- **Azimuthal inputs.** They make the turn and its control expressible. The programs' turn responses are above the control
  programs' in κ, but small (κ ≤ 0.24), and of the same size for the mass-preserving control. Programs therefore pick up some
  sensitivity to the hardest particles' azimuthal arrangement, as E41v found for JEDI-linear's internals.
- **The azimuthal pool on natural jets.** Adding it changes natural same call by −0.5 to +1.7 pp at matched bits relative to
  the B9 pool (3 / 9 significant).

## Caveats

- The truth program reaches 85–90 % same call, so "same call" alone does not separate a description of the network from a
  description of the task. The network-specific part is measured by the own-minus-truth and own-minus-transfer columns.
- Transfer uses donors' fit0 programs only, while own uses the mean of 3 fits.
- With only 3 fits per network, the fit-to-fit range is a coarse estimate of that variance.
- The turn10 and ctrl10 flip counts are small (40–288 jets), so recall on them has wide intervals (see `b10.json` recall_ci).
- tf_s1 is the patience-30 rerun of seed 1 (the plan's choice). Its accuracy equals tf_s0's (76.7 %).
- The 7-kbit criterion selects the K16 member in most fits; for Linformer s0 fit0, transformer s0 fits 0 and 2, transformer s1
  fits 0 and 1, and PFN s1 fit2 it selects K8, because the K16 DL there is 7,025–7,212 bits, just above 7,000.
- job.sh change: k8s-haosu-10.sdsc.optiputer.net was added to the node exclusion list (CephFS stall, 2026-10-22).

## Explainer entry (for site/explain.py)

**Question.** Can a short list of if-statements describe what several different small jet taggers do, not only JEDI-linear?

**Background.** B9 extracted, from each of ten JEDI-linear networks, a program of the form
$$\ell_c(x) = a_c + \sum_{k=1}^{K} b_{ck}\, h_{k}(x), \qquad h_k(x) = \max(0, \pm(f_{j}(x) - t)) \;\text{or a product of two such hinges},$$
where $f_j$ are jet features (pT-rank-window sums), and the program's call is $\arg\max_c \ell_c$. Every B9 network had JEDI's
architecture, so the success could come from JEDI's built-in rank windows and cuts.

**Method.**
1. Take eight non-JEDI taggers: Deep Sets, Linformer, transformer and PFN, two seeds each.
2. On 4,000 jets, fit the program to each network's five outputs. Repeat on two more disjoint sets of 4,000 jets.
3. Measure, on 6,000 other jets, how often the program's call equals the network's. Measure the size as description length,
   $\mathrm{DL} = \sum_c \big[32 + \sum_k (32 + \sum_{\text{factors}} (\log_2 140 + 2 + 32))\big]$ bits.
4. Compare at equal DL with boosted trees on standard jet observables, and with a program fitted to the truth labels alone.
5. Edit jets and ask whether the program changes its call when the network does:
   - shift all particles by $\Delta\eta = 0.06$ rad;
   - turn the 14 hardest particles in azimuth so that $m$ falls 10 % at fixed $p_T$ and $r$;
   - the same turn with the centroid $M_1$ and $m$ held (control);
   - add one soft particle.

   Recall is $\#\{\text{network flips reproduced}\}/\#\{\text{network flips}\}$, and $\kappa$ is Cohen's kappa of the two flip
   indicators.

**Result (figure `b10.png`).** At about 7 kbit, programs match 87–92 % of calls for every trained tagger. They are ahead of
boosted trees on standard observables by 0.7–2.7 pp. A truth-label program already matches 85–90 %. On the shift, programs
fitted to any tagger reproduce more call changes than the truth program, but a tagger's own program is not reliably better than
another tagger's.

**Checks.**
- Three repeated fits per network.
- Untrained redraws (constant calls).
- Shuffled-logit and random-feature programs.
- A mass-preserving control for the turn edit.
- Holm-corrected McNemar and bootstrap tests.

**Why it matters.** The if-statement description is not specific to JEDI-linear's architecture. But most of what it captures is
shared by all taggers trained on this task, so a program's agreement with a network is not evidence that it describes that
particular network.

**Glossary additions.**
- **Description length (DL)**: the number of bits needed to write the program (coefficients, knots, input names).
- **Fit-to-fit range**: the spread of a metric over programs fitted to disjoint jet sets of the same network.
- **Transfer**: a program fitted to network A, scored against network B.
- **Truth program**: the same recipe fitted to the truth labels instead of a network.
- **Complex moment $C_n$**: $\sum_i p_{T,i} (\Delta\eta_i + i\Delta\phi_i)^n / \sum_i p_{T,i}$.
- **Half-plane count**: the number (or pT sum) of particles on one side of a fixed line through the jet axis, the line taken
  from one of JEDI-linear's first-layer cuts.
- **Linformer**: a transformer whose attention is computed through a fixed-size projection of the particle axis.
- **PFN (Particle Flow Network)**: sum over particles of a learned per-particle map, followed by a classifier.

## Files
- /Users/anrunw/Documents/aaron-interp/interp/b10_programs_small_models.py
- /Users/anrunw/Documents/aaron-interp/results/jedi/b10.json, b10.png, b10_prep.json, b10_torch.json, b10_net_*.json/.npz
- /Users/anrunw/Documents/aaron-interp/logs/b10-{prep,torch,jedi,pfn,ds,linf,tf,agg}.log
