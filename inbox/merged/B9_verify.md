# B9 verification: fresh jets, a same-network refit control, and the truth-program comparison

Verifier node for B9 (`interp/b9_programs_across_nets.py`, report `inbox/B9_report.md`). Script `interp/b9_verify.py`, cluster Job `b9-verify` (1 CPU / 2 GiB, wall 1,426 s, exit 0; pod peak memory 2,047 MiB including page cache, no OOM). Results `results/jedi/b9_verify.json` (per-network inputs `b9_verify_net_<network>.json` on the volume). The log shows a JAX CUDA-plugin warning at start-up; the run is CPU-only (`JAX_PLATFORMS=cpu`) and the warning has no effect.

## Conclusion
On new jets, the B9 recipe again gives programs that match each network's call on 87–91 % of jets, and a network's own program again does better than another network's program. A second program fitted to the same network on different jets does exactly as well as the first. The gap to another network's program is therefore a real difference between networks, not fitting noise. The B9 statement that a program fitted only to the truth labels "does as well as a transferred program on both" natural jets and the shift edit does not reproduce. On the shift edit, other networks' programs reproduce about 7 percentage points more of a network's call changes than the truth-label program does. On natural jets the tie holds only when the 32-slot network is included in the pairs. Still uncertain: shift-edit recall for a single network varies by up to 10 points between programs fitted on different jets, and this variation is not in B9's jet-bootstrap intervals.

## Code review of `b9_programs_across_nets.py`

| Check | Finding |
|---|---|
| Stage / forward pass | Calls are the argmax of `logits` from `jedi_core.Jedi` (official networks) or `e33.Model` (X12 retrains), on the same raw inputs as the reproduction check. Official n64 test accuracy 80.55 % in B9; 81.61 % on the fresh test set here. |
| Padded slots | Every pool sum in `e11_mars_all.handbuilt` is masked by `real = pT > 0`. Padding is not counted as particles. For n32, slots ≥ 32 are zeroed, and the network receives `x[:, :32]` of the same edited jet. Network and program see the same particles. |
| Split leakage | fit / dev / test row sets are disjoint (dev minus fit, test minus fit and dev). The truth program uses labels on fit and dev only. No leakage found. |
| Edits | shift: +0.06 rad on Δη of real particles only. turn: pT and r kept; measured max $\lvert\Delta M_2/M_2\rvert = 7\times10^{-8}$. split: collinear 50/50, re-sorted. soft: four-momentum compensation by one common scale and boost, so every particle moves slightly (median of the per-jet maximum relative pT change 0.09 %). That is the E40 design, not an error. |
| Matched DL | The baseline member is chosen by dev natural same-call with DL ≤ the program's DL, then compared on test. Correct. |
| Recurrence null | Key-relabel null: each term's input redrawn uniformly from the 98 pool inputs. This is a weak null. MARS does not choose inputs uniformly, and any jet-classification target selects $M_2=\sum p_T r^2$ and $M_0=\sum p_T$ first. The truth program, untrained floor and a same-network refit (added here) are the informative references. The untrained floor gives one constant call, so it is uninformative except for term counts. |
| Transfer and n32 | Transfer pairs that involve n32 compare programs on different inputs: n32's program uses 32-slot features, and the other networks' programs use 64-slot features. Of B9's 72 natural pairs, 16 involve n32. These pairs drive the B9 natural "transfer − truth ≈ 0" result (next section). |
| Interval scope | All B9 intervals on per-network program numbers are jet bootstraps of one fitted program. They do not include program-fit variability (a different fit sample gives a different MARS path). The refit control below measures that variability. |

No coding error was found that changes a reported number. The issues are interpretive: the n32 view mismatch, the weak null, and missing fit variability.

### B9 transfer re-aggregated without n32 (from `b9.json`, no new run)
Mean over pairs, 9 networks (72 pairs) and the 8 networks without n32 (56 pairs):

| Quantity | All 9 | Without n32 |
|---|---|---|
| natural same-call, own − transfer | +1.25 pp | +0.69 pp |
| natural same-call, transfer − truth program | +0.47 pp | +0.88 pp |
| shift recall, own − transfer | +2.72 pp | +1.52 pp |

## Rerun on fresh jets (job b9-verify)

**Procedure.**
- Samples (`jedi_core.sample` rows):
  - fit = sample(1000, 5901): 5,000 jets;
  - fit2 = sample(1000, 5904) minus fit rows: 4,923 jets;
  - dev = sample(1000, 5902) minus fit / fit2 rows: 4,818 jets;
  - test = sample(1000, 5903) minus the rows of the other three sets: 4,710 jets.

  None of these seeds was used in B9 (B9 used 4901–4903).
- Networks: the 7 edit-capable B9 networks: official n64, n32, perm-inv n64, HGQ-r150 s0 / s2, float s0 / s1.
- Surrogates, built with the B9 functions (same 98-input pool, MARS settings, DL definition and n32 view):
  - **own**: K16 program (16 terms per logit) fitted to the network's 5 logits on fit;
  - **refit (new control)**: the same recipe fitted to the same network's logits on fit2, an independent set of jets. It measures how much a program changes when only the fitting jets change;
  - **truth**: K16 program fitted to the labels on fit, with target $\log(0.95\,\mathrm{onehot}+0.01)$;
  - **MARS-STD**: MARS on the 10 standard observables, K = 2 to 16. Matched-DL member: DL ≤ own K16, best dev same-call. In practice this is K8 on every network.
- Edit: shift, a rigid translation that adds 0.06 rad to Δη of every real particle (the W→Z flow edit of E6), applied to all 4,710 test jets. Turn, split and soft were not rerun. Turn is invisible to the pool by construction, and split and soft have few network call changes.
- Metrics:
  - **same call**: share of test jets where the surrogate's call equals the network's;
  - **shift recall**: share of the network's call changes under shift that the surrogate reproduces with the same new call;
  - **shift agreement**: per jet, neither changes its call, or both change to the same call.
- Statistics:
  - per-network differences: 1,000-draw jet bootstrap and exact McNemar test (an exact binomial test on the jets where only one of the two paired indicators is 1);
  - transfer means over ordered network pairs: network-level bootstrap with 2,000 draws;
  - Holm correction over the 7 networks for the matched-DL test.

**Reproduction.**

| Network | Test acc. | own K16 same call | refit K16 | truth K16 | own shift recall [95 % CI] | B9 own shift recall | refit | truth (B9 truth) |
|---|---|---|---|---|---|---|---|---|
| official n64 | 81.6 % | 90.2 % | 90.5 % | 89.2 % | 55.5 [53.8, 57.4] | 57.3 | 54.6 | 40.2 (52.3) |
| official n32 | 78.2 % | 86.8 % | 86.8 % | 84.8 % | 52.8 [51.1, 54.6] | 53.5 | 51.3 | 49.3 (47.9) |
| perminv n64 | 82.6 % | 90.9 % | 90.8 % | 89.4 % | 53.3 [51.5, 55.0] | 48.3 | 48.3 | 44.7 (49.9) |
| hgq150 s0 | 77.7 % | 88.8 % | 88.2 % | 86.0 % | 64.9 [63.2, 66.6] | 64.0 | 65.6 | 56.4 (64.4) |
| hgq150 s2 | 78.4 % | 90.0 % | 90.4 % | 87.7 % | 71.1 [69.4, 72.8] | 66.0 | 71.0 | 61.1 (67.4) |
| float s0 | 83.1 % | 88.9 % | 88.9 % | 88.5 % | 44.0 [42.3, 45.7] | 34.2 | 34.4 | 26.2 (29.0) |
| float s1 | 82.3 % | 89.0 % | 89.6 % | 88.1 % | 38.3 [36.5, 40.1] | 40.7 | 40.4 | 30.1 (34.0) |

- Same-call fidelity reproduces: 86.8–90.9 % here vs 86.9–90.3 % in B9.
- Shift recall of the own program reproduces within about 5 pp on 6 of 7 networks. Float s0 differs by 9.8 pp: 44.0 % here vs 34.2 % in B9, while the refit on the new jets gives 34.4 %. Per-network shift recall therefore depends on which jets the program was fitted on, by up to about 10 pp. That is larger than the jet-bootstrap interval of ±1.8 pp.

**Matched-DL program vs MARS-STD (K8 member chosen in every case).**
- Natural same-call: the program is higher on 7/7 networks, by +1.0 to +5.1 pp; Holm p ≤ 0.016 on all 7. Reproduces B9.
- Shift agreement: the program is higher on 6/7 networks, by +3.5 to +17.5 pp (Holm p < 1e-6). On float s1 the difference is +1.15 pp, CI [−0.49, +2.93], Holm p = 0.21.

  In B9 the exception was float s0 instead, where MARS-STD was higher by 3.2 pp. Here float s0 has the program higher by +5.3 pp [3.8, 6.9]. The count "6 of 7" reproduces, but the network that falls short is not the same.

**Same-network refit control and transfer.**
- H0: transfer loss (own − transfer) equals the loss from refitting the same network on other jets (own − refit).
- H1: transfer loses more, meaning the networks differ.

| Quantity, mean over pairs (network bootstrap CI, p) | 7 networks (42 pairs) | without n32 (30 pairs) |
|---|---|---|
| natural, own − refit | −0.08 pp [−0.32, +0.16], p 0.49 | −0.10 pp [−0.37, +0.16], p 0.49 |
| natural, refit − transfer | +1.43 pp [+0.51, +2.46], p 0.0005 | +0.81 pp [+0.41, +1.24], p 0.0005 |
| natural, own − transfer | +1.34 pp [+0.42, +2.40], p 0.002 | +0.71 pp [+0.24, +1.11], p 0.003 |
| natural, transfer − truth | +0.21 pp [−0.70, +0.96], p 0.62 | **+0.78 pp [+0.53, +1.06], p 0.0005** |
| shift recall, own − refit | +2.03 pp [−0.38, +4.56], p 0.13 | +2.12 pp [−0.69, +4.85], p 0.15 |
| shift recall, refit − transfer | +1.67 pp [−1.85, +4.79], p 0.29 | +0.75 pp [−3.10, +3.84], p 0.58 |
| shift recall, own − transfer | +3.70 pp [+1.72, +5.56], p 0.002 | +2.87 pp [+1.15, +4.98], p 0.03 |
| shift recall, transfer − truth | **+6.57 pp [+2.74, +10.30], p 0.0005** | **+8.53 pp [+6.42, +11.38], p 0.0005** |
| shift agreement, own − transfer | +3.30 pp [+1.58, +5.13] | +2.43 pp [+0.98, +4.10] |
| shift agreement, transfer − truth | +3.76 pp [+0.93, +6.29], p 0.008 | +5.36 pp [+4.17, +6.66], p 0.0005 |

Per network, the own program has higher shift recall than the truth program on all 7 networks: +3.5 to +17.8 pp, McNemar on the network's flipped jets p < 1e-6 in every case. It also has higher natural same-call on 6 of 7: +0.4 to +2.8 pp, with float s0 at p = 0.29.

What each hypothesis predicted, and what the run shows:
- **"Own − transfer is fitting noise"** predicts that natural refit − transfer ≈ 0. The run gives +1.43 pp [+0.51, +2.46], so this is excluded on natural jets. On shift recall, refit − transfer is +1.67 pp [−1.85, +4.79]. The shift part of the own − transfer gap is not separable from fit-to-fit variation.
- **"Programs describe the task only (truth ≈ transfer)"** predicts transfer − truth ≈ 0 on both metrics. The run excludes this on shift (+6.6 pp, p 0.0005) and, without the n32 pairs, on natural jets (+0.78 pp, p 0.0005).
- **"Networks share a mechanism beyond the label function"** predicts transfer > truth and refit ≈ own > transfer. This is consistent with natural jets (without n32) and with shift (transfer > truth).

**Recurrence with a same-network ceiling.** Matching follows the B9 rule: same input, same hinge type, knot within 0.05 in quantile units, any logit. The table gives the share of a network's 80 K16 terms that have a match in each reference program.

| Network | refit of the same network | other networks' programs (mean of 6) | truth program |
|---|---|---|---|
| official n64 | 0.56 | 0.33 | 0.26 |
| official n32 | 0.49 | 0.10 | 0.23 |
| perminv n64 | 0.45 | 0.32 | 0.28 |
| hgq150 s0 | 0.43 | 0.31 | 0.35 |
| hgq150 s2 | 0.43 | 0.32 | 0.25 |
| float s0 | 0.53 | 0.35 | 0.29 |
| float s1 | 0.59 | 0.38 | 0.35 |

- A program refitted to the same network on other jets shares only 43–59 % of its terms with the original. The B9 matching rule is therefore noisy even when the network is the same.
- Cross-network recurrence (0.31–0.38) is about 60–75 % of this ceiling, and slightly above the share matched in the truth program (0.25–0.35).
- The mean count over the 6 other networks is 1.8–2.3, a rate of 0.31–0.38 per network. This matches B9's 2.7–3.2 of 8, a rate of 0.34–0.40.
- In B9 (`b9.json`, K2 families), the hard-pT-sum family ($\sum p_T$ over a top-k window, or $M_0$) is present in the first two terms per logit of the truth program as well, for every network (9/9). Its presence in all trained networks and absence in the untrained ones therefore does not separate network from task.

## Figure check (`results/jedi/b9.png`)
- **Left panel**: same-call vs DL for 10 trained networks, with log-scaled bits and percentage axes. Readable; one message.
- **Right panel**: shift recall bars (own / other networks' mean / truth program) with no intervals. The truth bars rest on one fitted truth program per network. On fresh jets the truth bars move by 2–12 pp, to 40.2 % instead of 52.3 % on official n64, while the other bars move less. The panel's visual message ("truth ≈ transfer") does not reproduce.

  Suggested change: add the refit bar and fit-variability intervals, or show a mean over several fit samples.

## Corrected reading
- **Same recipe, all networks.** Applied to 9 trained JEDI-linear networks of normal accuracy, the recipe gives K16 programs (about 6,600 bits) that match the network's call on 87–90 % of test jets. This reproduces on fresh jets: 86.8–90.9 %.
- **Matched DL.**
  - On natural jets, the program has higher same-call than every baseline family on all 9 networks (MARS-STD rechecked here: 7/7).
  - On shift agreement, it is higher than MARS-STD on 6 of 7 networks in both runs, but the network where it is not higher changes between runs (float s0 in B9, float s1 here).
- **Turn edit.** The (pT, r) pool cannot represent the turn edit (recall 0).
- **Recurrence.** About 30 of 80 terms per network recur above the key-relabel null. That null is weak: a refit of the same network on other jets shares 43–59 % of terms, other networks share 31–38 %, and the truth program shares 25–35 %. The hard-pT-sum and $M_2$ hinge terms are also chosen by the truth-label fit.
- **Transfer.** Network A's program on network B loses 1.3 pp of natural same-call (0.7 pp without the 32-slot network) relative to B's own program. A refit of B's own program on other jets loses nothing (−0.1 pp), so the natural-jet gap is a network difference.
- **Truth program.** It does not match a transferred program:
  - on the shift edit, transferred programs reproduce 6.6 pp more of the network's call changes (fresh jets);
  - on natural jets, transferred programs are 0.8 pp higher when the n32 pairs, whose inputs differ, are excluded.

  The B9 result "truth ≈ transfer on shift recall" (−0.01 pp) depends on one fit of the truth program. On fresh jets the truth program's shift recall falls by 2–12 pp on 6 of 7 networks.
- **Fit variability.** Shift recall of a single program varies by up to 10 pp between fit samples. B9's per-network shift numbers and the shift part of own − transfer are within fit-to-fit variation.

## New glossary terms
- **Refit control**: the same program recipe fitted to the same network on a second, disjoint set of jets. It measures how much a program changes when only the fitting jets change.
- **Self-recurrence ceiling**: the share of a program's terms that are matched in the refit of the same network. Cross-network recurrence is read relative to it.
