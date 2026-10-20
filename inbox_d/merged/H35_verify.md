# H35 verify: b-prong localisation of displaced tracks (JetClass ParT_full, ParticleNet_full)

## Conclusion

With the jet's total number of displaced tracks held fixed, ParT and ParticleNet lower their top score when the displaced tracks are moved from one subjet to another. With a properly matched control, the drop is about 1.7–2.0 log-odds units for ParT and 1.1 for ParticleNet; H35 reported 2.9 and 2.1. The taggers therefore use where displaced tracks sit, not only how many there are. They do not use the b position implied by the three-prong mass structure. Moving the displaced tracks from a W-decay subjet that holds most of them onto the mass-assigned b-subjet also lowers ParT's score, even in jets whose W and top masses are well reconstructed. What the tests support is that the taggers penalise moving displacement off whichever subjet already carries most of it. Still uncertain: in those jets the displaced W-subjet may be the true b, or a charm hadron from W → cs, and JetClass has no per-particle truth to tell these apart.

(Log-odds = log p(Tbqq)/(1 − p(Tbqq)) from the 10-class softmax. A displaced track is a charged track with S = |d0|/σ(d0) > 2.)

## Code check of interp/h35_top_bprong.py

- Rows: random indices within the pT/η window (`jc_read`, sorted random choice). No first-k subsampling.
- Build: every variant, the identity included, goes through `part_core.build` (η = jet_η + sign(jet_η)·Δη). The edits change only (d0val, d0err, dzval, dzerr). The kinematic-only models give |Δ| = 0 for every edit, which confirms this.
- Padding: `sig` masks with `mask > 0 & charge != 0`, and `kt3_batch` labels only the first n = Σmask slots (−1 elsewhere). Padded slots are not counted.
- Same jets: the subjet labels are computed on the stored particle order. The edits are applied in that order before `build` re-sorts by pT, which permutes all arrays together, so the observables and the network inputs refer to the same particles.
- Invariants: the jet displaced-track count and the S multiset are asserted for M, C1 and C2, and the assertions pass on the fresh sample for all new edits too.
- No coding bug found.
- **Design issue: C1 is not a matched control for M.**
  - Size: C1 swaps k1 = min(k, n_prompt(c)) tracks against M's k. On the fresh sample the means over all tops are 2.36 vs 2.98.
  - pT: the receiving tracks differ. M moves the displacement onto the hardest prompt tracks of the W prong, C1 onto the hardest prompt tracks of c. The mean |Δ ln pT| between M's and C1's receiving tracks is 1.20.
  - Effect: because each (d0, σd0) tuple moves to a track of different pT, the contrast M − C1 mixes location with track-pT and size effects.
- The worker's equal-size subset uses normal-approximation intervals, not a bootstrap; the worker reported this.
- **Figure** (h35_bprong.png): the bars, intervals and labels match the JSON. The title states the measured ordering. It shows only the unmatched C1 control.

## Procedure (verify)

- **Sample:** JetClass test TTBar_101 and ZJetsToNuNu_101; H35 used the _100 files, so the jets are disjoint. Same window: 550 < pT < 650 GeV, |η| < 2, ≥ 3 particles. 2,000 random rows per class (rng 3551 / 3552).
- **Roles and edits:** subjet roles and the M / C1 / C2 edits are imported unchanged from `interp/h35_top_bprong.py`.
- **New edits on tops:** only the impact-parameter tuples are exchanged. The jet displaced count and the S multiset are asserted unchanged.
  - **Mk:** count-matched M. C1's k1 sources (the hardest displaced tracks of c) go to the k1 hardest prompt tracks of the W prong w, so Mk and C1 have equal size jet by jet.
  - **Mpt:** count- and pT-matched M. For each C1 receiving track, the prompt track of w with the nearest ln pT receives the same source track (greedy, without replacement). This lowers the mean |Δ ln pT| to C1's receiving tracks from 1.20 to 0.58.
  - **RV / RVpt:** the reverse move. wd is the W prong with more displaced tracks, and kr = min(n_disp(wd), n_prompt(c), n_prompt(wd)). The kr hardest displaced tracks of wd go onto prompt tracks of c: the hardest ones (RV), or the ones pT-matched to CW's receiving tracks (RVpt).
  - **CW:** the same kr source tracks swapped with the kr hardest prompt tracks inside wd.
- **Models:** ParT_full, PN_full, ParT_perm (weights permuted per tensor, seed 5) and ParT_kin (zero check).
- **Statistics:**
  - Per-jet change in top log-odds, with a 95 % bootstrap CI over jets (B = 2,000).
  - Two-sided sign-flip permutation test (5,000 permutations).
  - Holm correction over 48 tests: {ParT_full, PN_full, ParT_perm} × all contrast/set cells.
  - McNemar test on the top calls (argmax = Tbqq).
- **Analysis sets:**
  - agree: the mass-assigned b-prong c has more displaced tracks than either W prong. This holds for 52.4 % of tops (H35: 50.8 %).
  - disagree: the rest of the tops.
  - low: tops below the median unedited ParT_full top log-odds, to check for saturation.
- **Run:** `./part_job.sh d-h35-verify 'python -m interp.h35_verify'` (79 s, RTX 2080 Ti). `d-h35-verify2` reran the same sample with the RVpt edit and the low-score sets added (H35V_TAG=h35_verify2); it is a superset of the first run, and its numbers are quoted below.

## Unedited models on the fresh sample (4,000 jets)

| model | 10-class acc | binary acc | H35 sample |
|---|---|---|---|
| ParT_full | 86.8 % | 98.4 % | 86.8 / 98.35 % |
| PN_full | 85.6 % | 98.15 % | 85.7 / 98.1 % |
| ParT_kin | 81.1 % | 95.2 % | 81.1 / 95.6 % |
| ParT_perm | 14.1 % | 49.5 % | 14.7 / 49.8 % |

ParT_kin's maximum |Δ| is 0.000 for every edit.

## Numbers (change in top log-odds, mean [95 % CI], Holm p over 48 tests)

**Reproduction of H35:**

| set (N) | contrast | ParT_full | PN_full | ParT_perm |
|---|---|---|---|---|
| agree (1,048) | M | −5.72 [−5.93, −5.51] | −4.28 [−4.45, −4.09] | |
| agree (1,048) | C1 | −2.64 [−2.77, −2.50] | −2.22 [−2.35, −2.10] | |
| agree (1,048) | M − C1 | −3.08 [−3.27, −2.87], p 0.0096 | −2.05 [−2.21, −1.90], p 0.0096 | +0.02 [−0.02, +0.06], p 1 |
| all tops (1,701) | M − C1 | −2.19 [−2.36, −2.04] | −1.42 [−1.54, −1.31] | +0.02 |
| QCD (1,258) | M − C1 | +0.02 [−0.10, +0.15], p 1 | +0.19 [+0.07, +0.32], p 0.062 | +0.00 |

- **Top calls** (ParT_full, agree set): unedited 98.0 %, after C1 95.2 %, after M 71.7 %.
- **McNemar, M vs C1:** ParT_full 267 vs 20; PN_full 141 vs 33; ParT_perm 164 vs 167 (p = 0.91).

**Matched controls (agree set with k1 ≥ 1, N = 1,014):**

| contrast | ParT_full | PN_full | ParT_perm |
|---|---|---|---|
| Mk − C1 (equal size) | −2.26 [−2.46, −2.06] | −1.20 [−1.34, −1.05] | +0.04 [−0.00, +0.08] |
| Mpt − C1 (equal size and pT-matched) | −1.72 [−1.90, −1.53], p 0.0096 | −1.06 [−1.19, −0.92], p 0.0096 | +0.03 [−0.02, +0.07] |
| Mpt − C1, low-score half (N = 413) | −1.28 [−1.52, −1.05], p 0.0096 | −0.72 [−0.91, −0.53], p 0.0096 | +0.03 |
| Mpt − C1, all tops (N = 1,658) | −1.21 [−1.35, −1.08] | −0.76 [−0.86, −0.67] | +0.02 |

- **Top calls** (ParT_full): unedited 97.9 %, after C1 95.1 %, after Mpt 83.9 %. McNemar Mpt vs C1: 134 vs 21 (p = 2e−21). PN_full: 69 vs 29 (p = 7e−5).

**Reverse move, W prong to b-prong (RVpt − CW):**

| set (N) | ParT_full | PN_full | ParT_perm |
|---|---|---|---|
| agree (773) | +0.09 [−0.06, +0.25], p 1 | −0.01 [−0.11, +0.08], p 1 | +0.00 |
| disagree (913) | −0.62 [−0.79, −0.44], p 0.0096 | −0.27 [−0.41, −0.14], p 0.0096 | −0.03 [−0.07, +0.01] |
| disagree, low-score (546) | −0.38 [−0.59, −0.18], p 0.012 | −0.11 [−0.27, +0.05], p 1 | −0.05 |
| all tops (1,686) | −0.29 [−0.42, −0.17] | −0.15 [−0.24, −0.07] | −0.02 |

- **Without pT matching (RV − CW):** −0.65 [−0.84, −0.46] on the disagree set and +0.07 on the agree set (ParT_full).
- **Top calls** (ParT_full, disagree set): after CW 89.6 %, after RVpt 85.8 %. McNemar RVpt vs CW: 76 vs 41 (p = 0.0016).

## Part 3: mass-assignment quality and residual pT mismatch (d-h35-verify3c)

- **Purpose:**
  - Test whether the reverse-move drop in disagree jets comes from wrong kT-3 mass assignments.
  - Test whether the residual pT mismatch of Mpt (|Δ ln pT| = 0.58) drives Mpt − C1.
- **Sample:** the same 2,000 top jets as above (TTBar_101, rng 3551).
- **Edits:** the edits are unchanged (`interp/h35_verify3.py` imports `h35_top_bprong.make_edits` and `h35_verify.new_edits`).
- **Good mass assignment:** |m_Wpair − 80.4| < 10 GeV and 150 < m_jet < 200 GeV. This holds for 57.5 % of tops: 63.8 % of agree jets and 50.5 % of disagree jets.
- **Zero-mismatch extrapolation:**
  - Per-jet least-squares line of (M′ − C1) against mean |Δ ln pT| between M′'s and C1's receiving tracks.
  - The Mk and Mpt rows are pooled, giving two mismatch levels per jet.
  - The intercept is the contrast at zero mismatch, with a bootstrap over jets (B = 2,000).
- **Statistics:** sign-flip p (5,000); Holm over 18 cells.
- **Run:**
  - The GPU queue was full: the first pod hit an image-pull failure on gpn-fiona-mizzou-9, and the relaunch stayed unscheduled for 20 min.
  - The run went as a CPU job instead: `AI_PY=conda ./job.sh d-h35-verify3c 4 8 '<env-part on PATH, JETDISTILL_DEVICE=cpu> python -m interp.h35_verify3'`. It took 464 s, at 3.7 cores and 2.4 GiB peak.
  - The unedited top-call rates match the GPU run exactly (ParT_full 0.959, PN_full 0.9475, ParT_perm 0.281).

| set (N) | contrast | ParT_full | PN_full | ParT_perm |
|---|---|---|---|---|
| disagree, good mass (465) | RVpt − CW | −0.55 [−0.79, −0.30], Holm 0.006 | −0.19 [−0.38, +0.00], Holm 0.39 | −0.03 [−0.09, +0.03] |
| disagree, bad mass (448) | RVpt − CW | −0.69 [−0.93, −0.46], Holm 0.004 | −0.36 [−0.54, −0.17], Holm 0.004 | −0.03 |
| agree, good mass (490) | RVpt − CW | +0.26 [+0.06, +0.47], raw p 0.010, Holm 0.094 | +0.05 [−0.07, +0.17] | −0.00 |
| agree, good mass (649) | Mpt − C1 | −1.93 [−2.15, −1.71], Holm 0.004 | −1.18 [−1.34, −1.02], Holm 0.004 | +0.04 [−0.01, +0.09] |
| agree, bad mass (365) | Mpt − C1 | −1.33 [−1.65, −1.03], Holm 0.004 | −0.85 [−1.08, −0.63], Holm 0.004 | +0.00 |
| agree (1,014) | intercept at zero pT mismatch | −1.97 [−2.19, −1.72], slope −0.02 [−0.21, +0.16] | −1.15 [−1.32, −0.98], slope +0.03 [−0.12, +0.16] | +0.03 [−0.02, +0.08] |

- **Reverse move:** ParT's drop for the move from a displacement-heavy W prong onto the mass-assigned b-prong persists in good-mass jets (−0.55, CI fully negative). PN's drop in good-mass jets is −0.19, and its CI reaches 0.
- **Agree set:** moving minority W-prong displacement onto the mass-assigned b-prong in good-mass jets gives ParT +0.26. This is the H1 direction, but it does not pass Holm (0.094).
- **Pt mismatch:** the residual pT mismatch does not predict the per-jet contrast (slope consistent with 0). The pT-matched contrast at zero mismatch is −1.97 (ParT) and −1.15 (PN), so the Mpt − C1 values above are not inflated by the leftover mismatch.
- **By mass quality:** Mpt − C1 is larger in good-mass jets (ParT −1.93 vs −1.33).

## Hypotheses and which are excluded

| hypothesis | Mpt − C1 (agree) | RVpt − CW (disagree) | RVpt − CW (agree) | measured |
|---|---|---|---|---|
| H0: count only, or the effect comes from the pT/size mismatch of the tuples | 0 | 0 | 0 | excluded: Mpt − C1 ≠ 0 |
| H1: displacement is expected on the mass-assigned b-prong (the H35 reading) | < 0 | > 0 | ≥ 0 | excluded: RVpt − CW (disagree) < 0 with the CI fully negative, also in the low-score half for ParT |
| H2: any move between subjets lowers the score | < 0 | < 0 | < 0 | not supported: agree RVpt − CW CI contains 0 (ParT +0.09 [−0.06, +0.25]); QCD M − C1 ≈ 0 |
| H3: displacement is expected on the subjet that already holds most displaced tracks | < 0 | < 0 | ≈ 0 | all three match |

- H3 and "the network locates the true b by other cues, and the mass assignment is wrong in the disagree jets" make the same predictions here. JetClass stores no per-particle truth, so these two cannot be separated with this sample.
- Part 3 narrows the mass-assignment explanation for ParT:
  - In disagree jets with good W and top masses, RVpt − CW is still −0.55.
  - A wrong-assignment account would predict this contrast to shrink toward 0 or turn positive. It changes by only +0.14 relative to the bad-mass jets.
  - A good mass window does not guarantee that the b is assigned correctly. A W → cs decay also puts displaced charm tracks on a W prong. So H1 is not excluded in every jet; it does not describe the average response.

## Corrected reading

- **Size of the effect:**
  - Held fixed: the jet-level displaced-track count, the S multiset and the kinematics.
  - Edit: move displacement from the subjet that holds most displaced tracks (the mass-assigned b-prong in the agree set) onto a W prong.
  - Result: the Tbqq log-odds fall by 1.7 (ParT_full) and 1.1 (PN_full) more than for the same-size, pT-matched exchange inside that subjet. Extrapolated to zero pT mismatch: −1.97 [−2.19, −1.72] and −1.15 [−1.32, −0.98].
  - The unmatched H35 contrast was −2.9 / −2.1 (fresh sample −3.1 / −2.05). About 35–45 % of it comes from the control swapping fewer and softer tracks.
- **Reverse direction:** moving displacement from a W prong that holds most displaced tracks onto the mass-assigned b-prong also lowers the score (ParT −0.62, PN −0.27, pT-matched). This is the opposite sign from what a b-location expectation set by the subjet mass assignment predicts. For ParT the drop holds in jets with |m_Wpair − 80.4| < 10 GeV and 150 < m_jet < 200 GeV (−0.55 [−0.79, −0.30]).
- **Agree set:** moving minority displacement onto the mass-assigned b-prong in the agree set changes nothing (CI contains 0).
- **Ratio to the subjet-count classifier:** the "3× / 2× the subjet-count classifier" statement uses the unmatched contrast. The Sref response to the matched edits was not computed.
- **QCD jets:** M − C1 is +0.02 (ParT) and +0.19 (PN, Holm p = 0.062).

## Issues

1. The H35 control C1 is mismatched in size and in the pT of the receiving tracks; the matched contrast is about 55 % of the reported one.
2. The interpretation "relative to the three-prong mass structure, as a classical subjet b-tag does" is contradicted by the reverse-move test. What is supported is that displacement is tied to the subjet carrying most of it.
3. The within-b-prong control C1 is still −2.6. The networks respond to which tracks inside a prong carry the displacement (pT rank, σd0–pT consistency or angle). Mpt reduces but does not remove the pT mismatch (|Δ ln pT| = 0.58). Part 3 shows that this leftover mismatch does not predict the contrast (slope ≈ 0).
4. The Sref ratio was not recomputed for the matched edits.
5. Single test file per class per run; two disjoint samples (H35 and this verify) give consistent unmatched numbers.
6. Part 3 ran on CPU (GPU queue full). The identical unedited top-call rates show the CPU and GPU forward passes agree on this sample.

## Explainer addendum (Checks step, for site/explain.py)

```python
H35_VERIFY_CHECKS = ('Checks (verify)', 'Fresh JetClass files (TTBar_101, ZJetsToNuNu_101; 2,000 jets each). The H35 contrast reproduces: $M-C1=-3.08\\,[-3.27,-2.87]$ (ParT). Control matched in swap size and receiving-track $p_T$: $M_{p_T}-C1=-1.72\\,[-1.90,-1.53]$ (ParT), $-1.06\\,[-1.19,-0.92]$ (ParticleNet); permuted ParT $+0.03$. Reverse move, from a W prong that holds most displaced tracks onto the mass-assigned $b$-prong: $\\Delta\\,\\mathrm{logit}=-0.62\\,[-0.79,-0.44]$ relative to the within-prong swap, the opposite sign from a mass-assigned $b$ location; in jets with $|m_{W\\,\\mathrm{pair}}-80.4|<10$ GeV and $150<m_J<200$ GeV it is still $-0.55\\,[-0.79,-0.30]$. Extrapolated to zero receiving-track $p_T$ mismatch, the matched contrast is $-1.97$ (ParT) and $-1.15$ (ParticleNet). The response follows the subjet that already carries most displaced tracks, not the mass assignment.', None)
```

## Files

- Scripts: /Users/anrunw/Documents/aaron-interp/interp/h35_verify.py, /Users/anrunw/Documents/aaron-interp/interp/h35_verify3.py
- Results: /Users/anrunw/Documents/aaron-interp/results/physics/h35_verify2.json (superset), h35_verify.json, h35_verify3.json, h35_verify2_perjet.npz, h35_verify_perjet.npz
- Logs: /j-jepa-vol/aaron-interp/logs/d-h35-verify.log, d-h35-verify2.log, d-h35-verify3c.log (d-h35-verify3 and d-h35-verify3b were deleted unrun: image-pull failure, then no free GPU)
