# B5 verification (job b5-verify, 1 CPU / 2 GiB, exit 0, wall 679 s)

- Script: `interp/b5_verify.py`. Results: `results/jedi/b5_verify.json`, `results/jedi/b5_verify_calls.npz`. Log: `logs/b5-verify.log`.
- No refit. The stored programs from `b5_programs.json` (axis20, axis50, inv50, all50) and mars30 (E11) are evaluated on a fresh sample.

## Code review of `interp/b5_compact_jedi.py` and `interp/b5_plot.py`

- **Splits.** Test is seed 11 minus seed 0, and confirmation is seed 91 minus seeds 0 and 11 (`E.indices`, the same rule as `jedi_core.sample`). Fit and dev come from seed 0. The program size T is not selected on test. No leakage found.
- **Padding.**
  - `edit_shift`, `randdisp` and `reaxis` move real particles only.
  - `rotate` acts on all slots, but padded slots sit at (0, 0) and stay there.
  - `handbuilt`, `extras` and the EFPs give padded slots weight 0 (window mask `real`, or $p_T = 0$).
  - The jump edit re-sorts by pT; the selected particle becomes slot 0 in 100 % of selected jets.
- **Physics recompute.** On the unedited jets, the recomputed physics matches the stored values to 5.7e-8, and the network calls are identical (B5 log).
- **Network stage.** The network is run through `Jedi.run` with the default quantizers. mars30 goes through `e11_verify.chain` and `Last`, the same path as E11.
- **Pools.**
  - P_axis contains `X:M0·M2` and `X:|M1|²`. Their difference is close to the pairwise (invariant) quantity $\sum_{i<j} p_{T,i}p_{T,j}\Delta R_{ij}^2$, so P_axis is not free of invariant information. It can build that quantity only with two linear terms.
  - Whether a program uses it can be checked in `b5_programs.json`; this does not change the pool contrast.
- **No code bug found** that changes the reported numbers.

## Fresh-sample rerun

- **Sample.** `sample(2000, seed 23)` minus every jet of seeds 0, 11 and 91: N 8,218 (1,645 true W).
- **Network accuracy.** 81.27 % [80.40, 82.08].
- **Statistics.** Bootstrap B = 1000 over jets; McNemar on paired call-change indicators; Holm over 50 cells (5 models × 10 edits).

### Replication of the B5 numbers

Rows marked "new" are controls that B5 did not run (definitions in the next section).

| edit | network % calls changed [95 % CI] | network W→Z % | axis20 | axis50 | inv50 | all50 | mars30 |
|---|---|---|---|---|---|---|---|
| trans, Δη +0.06 | 55.5 [54.5, 56.5] | 77.9 | 50.9 | 68.1 | 0.0 | 44.3 | 59.5 |
| trans_neg, Δη −0.06 (new) | 62.1 [61.1, 63.1] | 43.3 | 51.0 | 68.2 | 0.0 | 44.6 | 59.0 |
| trans_rd, rigid 0.06 in a random direction (new) | 59.1 [58.0, 60.1] | 62.7 | 50.8 | 68.3 | 0.0 | 44.5 | 59.1 |
| trans_c, per-particle random 0.06 | 52.2 [51.0, 53.2] | 49.2 | 51.4 | 63.8 | 56.5 | 62.9 | 54.4 |
| rot 45° | 3.7 [3.3, 4.1] | 2.5 | 0.0 | 0.0 | 0.3 | 0.2 | 1.8 |
| wta | 29.8 [28.8, 30.8] | 58.4 | 26.8 | 32.6 | 0.0 | 16.4 | 29.4 |
| wta_c | 33.2 [32.2, 34.3] | 53.6 | 27.7 | 33.7 | 0.0 | 16.4 | 31.7 |
| wta_m, M2-matched (new) | 33.6 [32.6, 34.6] | 55.3 | 27.5 | 33.8 | 0.0 | 16.1 | 31.1 |
| hardest | 30.8 [29.8, 31.7] | 50.8 | 28.4 | 33.8 | 0.0 | 17.0 | 31.0 |
| hardest_c | 34.3 [33.2, 35.2] | 49.8 | 29.2 | 34.8 | 0.0 | 16.8 | 33.0 |

- **Same call.** axis20 82.5 %, axis50 89.1 % [88.4, 89.7], inv50 88.5 % [87.8, 89.2], all50 87.7 %, mars30 94.1 %.
- **trans and wta.** The B5 values replicate within about 1 point.
- **Paired differences vs the network, trans:**
  - axis50 +12.6 [+11.6, +13.6] points;
  - axis20 −4.6 [−5.9, −3.2];
  - mars30 +4.0 [+3.1, +5.0], Holm p 1.2e-17.
- **Reproduction criterion.** No ≤ 50-term program meets it on trans, as in B5.
- **Pool contrast.** It replicates: under trans, wta and hardest, the axis programs change 27–68 % of calls and the invariant program 0 %, while same call on unedited jets differs by 0.6 points (axis50 vs inv50).

### Control 1: M2-matched re-axis (wta_m), testing the reading's WTA-control sentence

- **Construction.** For each jet, the axis is moved in a random direction (rng 202), with the length solved so that $M_2=\sum_i p_{T,i} r_i^2$ about the new axis changes by exactly the same amount as under the WTA re-axis. Here $\Delta M_2 = M_0|a|^2 - 2\,\vec a\cdot\vec M_1$. A solution was found for 100 % of jets; median length 0.0396 rad (WTA: 0.0396).
- **Diagnostics.**
  - The pT centroid lies 0.0015 rad from the stored axis (median), against a re-axis length of 0.040 rad.
  - The median cosine between the WTA direction and $\vec M_1$ is 0.03.
  - Median ΔM2: wta 1.645, wta_c 1.655.
  - $|\Delta M_2|$ is smaller under wta than under wta_c in 50.3 % of jets.
  - Median relative change of $\sqrt{M_0M_2}$: wta +10.10 %, wta_c +10.21 %.
- **Hypotheses.**
  - H0: the gap between the wta and wta_c call-change rates is set by ΔM2. Prediction: wta_m = wta.
  - H1: the gap depends on the direction of the re-axis relative to the particles, at equal ΔM2. Prediction: wta_m = wta_c > wta.
- **Network results.**
  - wta − wta_m: −3.8 [−4.5, −3.1] points, McNemar 266 vs 580 discordant jets, p 1.3e-27.
  - wta_m − wta_c: +0.4 [−0.2, +1.1] points, p 0.25.
  - wta − wta_c: −3.4 [−4.1, −2.8] points, p 4.5e-23 (B5: −3.6 points).
- **Outcome.** H0 is excluded, and the H1 prediction is observed.
- **Programs.**
  - axis50: wta 32.6, wta_c 33.7, wta_m 33.8 %. The gap is 1.1 points, against the network's 3.4–3.8.
  - mars30: 29.4 / 31.7 / 31.1 %.

### Control 2: sign and direction of the rigid shift (trans_neg, trans_rd)

- **Prediction of an $M_2$-about-the-axis description.** $M_2$ changes by $M_0 d^2 - 2\vec d\cdot\vec M_1$. Because $|\vec M_1|/M_0 \approx 0.0015 \ll 0.06$, this predicts the same response for +Δη, −Δη and any direction.
- **Network.**
  - trans vs trans_neg: 55.5 vs 62.1 % calls changed, difference −6.6 [−7.5, −5.7] points, McNemar p 5.4e-54.
  - W→Z: 77.9 % under +Δη, 43.3 % under −Δη, 62.7 % under a random direction.
  - trans vs trans_rd: −3.6 [−4.3, −2.9] points, p 8.1e-23.
- **Programs: no change with the sign or direction of the shift** (axis50, all50, mars30 differ by ≤ 0.5 points in calls changed and ≤ 2.5 points in W→Z).
  - W→Z under −Δη: mars30 73.1 % (+29.8 [+26.9, +32.5] points vs the network) and all50 77.4 % (+34.1 [+31.2, +36.9]).
  - W→Z under a random direction: mars30 +14.5 [+11.9, +17.1] points.
- **rot 45°.** The network changes 3.7 % of calls; the axis programs change 0.0 % (Holm p 3.9e-90).
- **Hypotheses.**
  - H0: the network's response to a rigid 0.06 rad shift is the same for every direction. It is excluded (p 5e-54).
  - The prediction that the mars30 and P_all T 50 programs follow this H0 is observed.
- **Relation to earlier facts.** This is consistent with the odd 12–20 % direction term noted in E6.

### Per-jet agreement

Edit agreement minus the never-flip reference:
- axis20: trans −3.3 points, wta −3.6 points.
- axis50: trans +6.8 points.

The matching of call-change rates between the axis programs and the network does not extend to the same jets.

## Figure check

- `b5_summary.png`: readable, axes labelled.
  - The jump panel is dominated by the network's wide CI band (N 212). It carries no separation, and the reading says so.
  - The suptitle states the invariant-pool result.
- `b5.png`: consistent with `b5.json`.
- No error found in either figure.

## Corrected reading

- **Pool contrast.** It replicates on seed 23 (N 8,218).
  - At equal budgets, the axis-relative programs change calls under the +0.06 Δη shift and the WTA / hardest re-axis at rates that bracket the network's: T 20 51 / 27 / 28 %, T 50 68 / 33 / 34 %, against the network's 56 / 30 / 31 %.
  - The invariant-pool programs change 0 % (by construction).
  - Same call differs by 0.6 points.
  - Per-jet agreement with the network's changes is at or below the never-flip reference at T 20, and +6.8 points at T 50.
- **Strict criterion.** No ≤ 50-term program meets it on the shift; mars30 also fails it (+4.0 points).
- **Direction.** The W→Z match of mars30 and P_all T 50 (within 3 points) holds for the +Δη shift only. The network's W→Z rate depends on the shift's sign (77.9 % for +Δη, 43.3 % for −Δη, 62.7 % for a random direction). The programs give the same rate for every direction (mars30 73–78 %, P_all T 50 77–78 %).
- **WTA control.** Replaces B5's sentence. The network changes more calls under a random-direction re-axis than under the WTA re-axis (33.2 vs 29.8 %). The same holds when the random re-axis is matched per jet to the WTA re-axis in ΔM2 (33.6 %, p 1.3e-27). So the gap is not accounted for by $M_2$ about the axis. It depends on the direction of the re-axis, as do the sign dependence of the shift response and the 3.7 % rotation response, which the axis programs (radial terms only) do not reproduce.
- **Axis jump.** Uninformative for JEDI-linear (4 vs 7 of 212 jets), unchanged.

## Explainer entry (for site/explain.py, Paper B; site/ is outside this worker's write scope)

- **Question.** Does the network respond to a shift or re-axis only through how far the particles move from the axis (an $M_2$-type quantity), or also through the direction of the move?
- **Background.** About an axis $\vec a$, $M_2(\vec a) = M_2 - 2\,\vec a\cdot\vec M_1 + M_0|\vec a|^2$. With the stored axis within 0.0015 rad of the pT centroid, a move of length $d$ changes $M_2$ by $\approx M_0 d^2$ in any direction.
- **Method.** On 8,218 held-out jets:
  - shift all particles by 0.06 rad in +Δη, −Δη and a random direction;
  - re-axis on the WTA axis, in a random direction of equal length, and in a random direction whose length gives the same $\Delta M_2$ as WTA for that jet.
  - Count the network's call changes and compare them by paired McNemar tests.
- **Result.**
  - Calls changed: +Δη 55.5 %, −Δη 62.1 %, random direction 59.1 %; W→Z 78 / 43 / 63 %.
  - WTA re-axis 29.8 % against the $M_2$-matched random re-axis 33.6 % (p 1.3e-27).
  - The 3,824-term program and the 50-term programs give the same response for every direction.
- **Checks.**
  - Same split rules as B5, with seeds 0, 11 and 91 excluded.
  - The B5 numbers replicate within about 1 point.
  - The $\Delta M_2$ match is exact for every jet (by construction).
- **Why it matters.** The size of the shift response is matched by programs built from axis-relative radial quantities. Its direction and sign dependence are not, so the network also uses a direction-dependent term that these programs leave out.
