# E39x: how many dimensions do retrained JEDI-linear networks share at S and h4? (loop A round 19, stage 3, exploratory method)

**Conclusion.** We joined two separately trained JEDI-linear networks with a linear map ("stitching") that passes only r numbers
from the first network to the second, and counted how often the joined network still gives the second network's decision. When
the map is trained on the second network's own classification loss, 4 numbers are enough for 87 % agreement at the
second-round object layer S. That is close to the 90 % at which the two unedited networks agree. A map built only to copy the
layer needs about 8 numbers. Four numbers taken from the layer carry more than the first network's 5 class scores do at S, but
not at the next layer h4. The four named program elements (Σ pT r², soft count, hard window, core) carry somewhat more than
random units of the same number. But on their own, with everything else held at its mean, they reproduce only about 30 % of
decisions. So the shared content is low-dimensional, but it is not shown to be these named units. Whether a joined network
responds to an edit like its first or its second network is still not resolved with 7 networks. The one exception is the Z-jet
prong-sharing response at S, which follows the receiving network.

## Terms (defined here; added to `site/explain.py` GLOSSARY)
- **S**: the second-round object layer (64 sums over particles, $S=\sum_i u_i/16$). **h4**: the first dense layer after S (64 units).
- **Stitching**: run network A up to layer L (S or h4), apply a map from A's layer into B's layer, clip every unit to B's own
  [min, max] on the fit jets (B's HGQ input quantizer wraps out-of-range values), then run B's exact fixed-point tail.
- **Same-call %**: the share of test jets on which the stitched network's argmax class equals B's own unedited argmax class.
  **Native agreement**: the share of jets on which unedited A and unedited B give the same call.
- **Ridge map** (E39): $\hat L_B=(L_A-\bar L_A)W+\bar L_B$, ridge strength chosen on a 20 % hold-out of the fit jets.
- **Reduced-rank regression (RRR), rank r**: $W_r = W V_r V_r^\top$. Here $V_r$ holds the top-r right singular vectors of the
  fitted values $(L_A-\bar L_A)W$ on the fit jets. This is the rank-r map closest to the ridge prediction in squared error. The
  map passes r numbers from A to B. r = 64 is the full map.
- **Task-loss map**: a rank-r map ($W=UV$, $U$ 64×r, $V$ r×64, plus a bias) started at the RRR map. It is trained with Adam (300
  steps, learning rate 3e-3, batch 2048 fit jets) to minimise the cross-entropy of B's frozen tail on the true labels. Gradients go
  through a float copy of B's tail (fused kernels, input quantizers omitted). That copy's call agrees with the exact tail on
  96.4–100 % of test jets (HGQ s0–s2 96–97 %, official 98.9–99.4 %, float 100 %). Evaluation always uses B's exact tail.
- **Logit-only donor** (E39v): a ridge map from A's 5 output logits into B's layer. It can only pass A's class scores (rank ≤ 5).
- **Subset stitch**: pass only the k S units of A that E33x named for a role, mapped onto B's units for the same roles. Each
  role must have a held-out formula $R^2\ge 0.9$ in both networks. Every other S unit of B is held at its mean over the fit jets.
  - The four roles are: r2, a unit fitting $a+b\sum p_T r^2+c\sum r^2+d\,N_{\rm soft}$; soft, a soft-particle count; hard, a
    pT sum over a hard rank window; core, a sum of r over the core.
  - Control: 200 random sets of k live S units (fit s.d. > 1e-6) in A and in B, with the same ridge map, mean fill and tail.
- **Inheritance index**: $I=(r_{\rm stitch}-r_B)/(r_A-r_B)$, where $r_X$ is network X's mean response to an edit. I = 1 means the
  stitched network responds like donor A; I = 0 means it responds like recipient B.
  - Pooled over the ordered pairs with $|r_A-r_B|>2$ jet-bootstrap SE as $\sum(r_s-r_B)(r_A-r_B)/\sum(r_A-r_B)^2$.
  - Because that selection is symmetric in A and B, a stitched response equal to one constant in every pair gives exactly 0.5.
    This value was recomputed: 0.500 in every cell. **0.5 is the no-information null.**
- **Network-level bootstrap**: resample the networks with replacement (2,000 draws). Each draw weights every ordered pair of
  distinct networks by the product of the networks' counts and recomputes the pooled number. The interval therefore reflects
  which networks were trained, not only which jets were drawn.
- **McNemar (exact)**: a binomial test on the jets where exactly one of two stitches gives B's call. **Holm**: step-down
  multiple-comparison correction.

## Procedure
- **Networks**: the 10 networks of E33x/E39, loaded through `e33_program_recurrence_jedi.Model`: official 3f n64 (reference),
  3f n32, 3f n128, perm-inv n64, 16f n64, X12 HGQ-r150 s0/s1/s2, and X12 float s0/s1.
  - HGQ s1 (56.5 %) is computed but left out of every pooled number. That leaves 9 networks and 72 ordered pairs for same-call,
    and 7 networks and 42 pairs for edits.
  - Untrained donors: `redraw_untrained` seed 0 of each network. Their accuracy is 18–21 %, i.e. chance.
- **Samples**: fit = `sample(2000, 4941)`, N = 10,000, used for the maps, the task-loss training and the clip ranges. Test =
  `sample(2000, 4942)` minus the 398 rows shared with fit, N = 9,602. n32 reads the first 32 slots. n128 and 16f read the same
  rows from their own files (copy check max |Δx| = 0).
- **Exactness checks**:
  - B's tail run from its own S or h4 reproduces its logits with max |Δ| = 0 for every network.
  - **Official n64 accuracy is 80.81 % [79.97, 81.61], inside the expected 80.6–81.3 %.**
- **Arm 1 (rank)**: RRR maps at r = 1, 2, 4, 8, 16, 64 at S and h4, for all 90 ordered pairs. Controls at each r:
  - a random map with the same singular values as $W_r$ (Haar-random orthogonal factors);
  - an untrained donor with its own RRR map.
  - Reference: the logit-only donor.
- **Arm 2 (task loss)**: task-loss maps at r = 1, 2, 4, 8, 64, compared with RRR at the same r on same-call, accuracy and test
  cross-entropy.
- **Arm 3 (subset)**: subset stitch at S for the 68 non-degenerate pairs that share at least one named role (k = 1–4).
  - Per-pair p = (1 + #random ≥ named)/201, with Holm over the 68 pairs.
  - Across pairs: network bootstrap of the mean of (named − random mean).
  - Reference without stitching: B's own role units kept and the rest held at the mean, against B's own random live sets.
- **Arm 4 (edits)** use the edit builder of `interp/e39_stitch_retrains_jedi.py` on the 8 networks with a 3-feature 64-slot
  input; n32 sees the first 32 slots of the edited jet.
  - **T**: every real particle's Δη + 0.06 rad, inputs kept relative to the stored E-scheme axis; true W jets, N = 1,928;
    response = 1[called Z after] − 1[called Z before], i.e. the W→Z flow.
  - **D**: the E35 prong-sharing edit with dc = +0.3. The two exclusive-kT prongs' |cos θ*| is raised by 0.3 in the jet rest
    frame, at fixed jet four-momentum (max relative ΔP4 2e-13). Response = Δ(ℓ_W − ℓ_Z), on W jets (DW) and Z jets
    (DZ) of the 3,866 accepted W+Z jets.
  - **N**: one 1 GeV particle added at r = 0.3, random azimuth, inserted at its pT rank; q and g jets with ≤ 63 particles,
    N = 2,675; response = Δ(ℓ_g − ℓ_q).
  - Stitch variants: full-rank ridge, RRR rank 4, untrained donor (full), and logit-only donor, each at S and h4.
- **Job**: `e39x-run`, 1 CPU / 2 GiB, about 1,155 s of script time. A smoke job `e39x-smoke` ran first.
  - The script logged "wrote …/e39x.json", but the pod then ended without the wrapper's exit line and the Job is marked failed.
  - The smoke run peaked at 2,047 MiB of 2 GiB, so the pod was most likely killed for memory after the final save.
  - The JSON is complete (all sections present). Request 3 GiB for a rerun.

## Results
### Arm 1: same-call % with B against map rank
Means are over 72 pairs; brackets are network-bootstrap 95 % intervals.

| layer | r = 1 | 2 | 4 | 8 | 16 | 64 (full) | logit-only | native | untrained (r 64) | random map (r 64) |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 49.9 [48.2, 51.7] | 59.7 [57.1, 62.6] | **75.3 [72.3, 77.9]** | 83.9 [80.6, 87.1] | 86.5 [83.3, 89.8] | 86.6 [83.4, 90.0] | 66.0 [63.5, 68.6] | 89.5 | 35.4 [27.1, 44.7] | 22.3 |
| h4 | 58.3 [53.5, 63.2] | 65.3 [62.0, 68.4] | 77.8 [75.0, 81.2] | 86.5 [84.3, 88.5] | 89.1 [87.0, 91.3] | 89.7 [87.4, 92.2] | 78.6 [75.8, 81.4] | 89.5 | 34.1 [26.2, 42.6] | 23.4 |

- **Fitted-value variance kept**: share of the ridge prediction's variance in the top r directions at S, mean over pairs: r 1
  0.63, r 2 0.81, r 4 0.94, r 8 0.991, r 16 0.999.
- **Untrained donors** stay at 31–35 % for every r; random maps stay at 21–23 %.
- **Rank r − logit-only, S**:
  - r 2: −6.3 points [−9.3, −3.4];
  - **r 4: +9.2 points [+6.8, +11.4], network-bootstrap p 5e-4;**
  - r 8: +17.8 [+15.5, +20.2];
  - r 64: +20.6 [+18.9, +22.3].
  - Per pair, exact McNemar with Holm over 72 pairs: rank 4 above logit-only in 67/72 pairs, below in 0. Rank 2 above in 6,
    below in 56. Rank 1 below in 72/72.
- **Rank r − logit-only, h4**:
  - **r 4: −0.7 points [−2.6, +1.2], p 0.50**; per pair, rank 4 above in 21 and below in 29 of 72.
  - r 8: +8.0 [+6.2, +9.6]; r 64: +11.1 [+9.4, +12.7].
- **Rank 4 − untrained donor rank 4**: S +40.4 points [+31.6, +48.7]; h4 +44.4 [+37.8, +50.9].
- **Jet bootstrap of the pooled pair-mean indicator** (all 90 pairs): rank 4 at S [74.9, 75.8] against logit-only [65.0, 66.3].
  Jet-level intervals are about 10× narrower than the network-level ones.
- **Clipped share** (S): 10.9 % of unit values at rank 4, 3.1 % at full rank.

### Arm 2: task-loss map against reconstruction (RRR) at the same rank
Same-call % with B, with network-bootstrap intervals.

| layer | task r 1 | r 2 | **r 4** | r 8 | r 64 | task − RRR at r 4 (same-call) | task − RRR at r 4 (accuracy) | task − RRR at r 4 (test CE) |
|---|---|---|---|---|---|---|---|---|
| S | 70.2 [68.2, 72.0] | 81.3 [79.2, 83.2] | **87.3 [85.2, 89.2]** | 89.0 | 89.1 [87.0, 91.3] | +12.0 [+10.0, +14.2] | +10.8 [+9.4, +12.5] | −0.36 [−0.42, −0.31] |
| h4 | 73.8 [72.6, 75.0] | 85.4 [83.4, 87.4] | **88.6 [86.8, 90.6]** | 89.3 | 89.1 [87.3, 91.1] | +10.8 [+9.6, +12.0] | +9.1 [+7.2, +11.1] | −0.23 [−0.28, −0.18] |

- **Accuracy of the stitched network** at S: task-loss r 4 77.8 %, r 64 79.2 %; RRR r 4 67.0 %, r 64 75.8 %. At h4: task-loss r 4
  79.2 %; RRR r 4 70.0 %.
- **Training check**: the mean float-tail fit cross-entropy at S, r 4, fell from 1.06 to 0.65 over training.
- **Full rank**: task − RRR is +2.5 points [+1.0, +3.9] at S and −0.6 [−1.3, +0.1] at h4.
- **The task map was trained on the true labels, not on B's calls.** Its same-call gain could therefore partly come from
  accuracy. Its accuracy at r 4 is still 1.3–3 points below B's own.

### Arm 3: named-role units against random units of the same number (S)
- **Named sets**: E33x roles with R² ≥ 0.9 in that network. Official n64 S0 / S29 / S19 / S30, n32 S41 / S5 / S34 / S10, n128
  only hard S41 and r2 S54, perm-inv S41 / S13 / S27 / S48, 16f S20 / S44 / S43, HGQ s0 S22 / S52 / S4, HGQ s2 S32 / S6 / S48,
  float s0 soft S4 only, float s1 r2 S5 only. This gives 68 pairs with k ≥ 1.
- **All of B's S held at its mean** gives B's call on 18.8–21.5 % of jets.
- **Subset stitch, same-call with B**: named 29.7 % against random 23.1 % (mean over 68 pairs).
  - **Named − random: +6.6 points [+2.1, +11.9], network-bootstrap p 0.0045.**
  - By k: k 1 (26 pairs) 19.9 against 19.8; k 2 (12) 31.2 against 22.0; k 3 (24) 36.6 against 26.6; k 4 (6) 41.6 against 26.1.
  - Per pair, 24/68 have p < 0.05 uncorrected.
  - Holm: 0/68. The smallest attainable p with 200 random sets is 1/201 = 0.005, so Holm over 68 pairs cannot reach 0.05. This
    is a design limit of this run; 14,000 random sets per pair would be needed.
- **k = 1 pairs into float s1** (its r2 unit only) give 8–9 % same-call, below the all-at-mean value.
- **No-stitch reference** (B's own role units, rest at mean, against B's own random live sets): official n64 57.8 % against
  31.3 % (p 0.015). The other quantized and 16f networks give 32–52 % against 23–40 % (p 0.07–0.25). Float s0 / s1 (k 1) give
  19.0 / 8.4 % against 19.3 / 19.6 %.

### Arm 4: inheritance index with network-level 95 % intervals
42 ordered pairs of 7 networks; null 0.5. Each cell gives I [95 % interval], p against 0.5.

| layer, response (eligible pairs) | full-rank stitch | rank-4 stitch | untrained donor | logit-only donor | full − untrained |
|---|---|---|---|---|---|
| S, T (42) | 0.36 [0.03, 0.89], p 0.54 | 0.85 [0.63, 1.22], p 0.021 | 0.28 [−0.15, 0.88] | 0.66 [0.05, 1.00] | +0.08 [−0.24, 0.48] |
| S, DW (38) | 0.95 [0.50, 1.56], p 0.053 | 0.56 [0.19, 1.44] | 0.76 [0.45, 1.41] | 0.66 [−0.09, 0.87] | +0.20 [−0.07, 0.30] |
| S, DZ (40) | **0.04 [−0.36, 0.22], p 0.011** | −0.08 [−0.57, 0.35], p 0.024 | 0.34 [0.14, 0.64] | 0.21 [−0.14, 0.54] | −0.30 [−0.96, +0.05], p 0.066 |
| S, N (38) | 0.68 [0.29, 0.82], p 0.32 | 0.24 [−0.27, 0.66] | 0.86 [0.50, 1.46] | 0.62 [0.12, 0.77] | −0.19 [−0.79, 0.26] |
| h4, T (42) | 0.42 [−0.03, 0.96] | 0.63 [0.30, 1.18] | 0.33 [−0.11, 0.91] | 0.91 [0.50, 1.26] | +0.09 [−0.17, 0.39] |
| h4, DW (38) | 0.51 [0.20, 0.76], p 0.94 | 0.42 [−0.06, 0.75] | 0.90 [0.55, 1.41] | 0.57 [0.12, 0.85] | −0.39 [−0.98, −0.26], p 0.006 |
| h4, DZ (40) | 0.29 [−0.06, 0.44], p 0.030 | 0.19 [−0.17, 0.29], p 0.001 | 0.23 [0.07, 0.42] | 0.24 [−0.28, 0.35] | +0.06 [−0.25, 0.16] |
| h4, N (38) | 0.53 [0.09, 0.66] | 0.01 [−0.40, 0.12], p 0.001 | 0.79 [0.47, 1.47] | 0.72 [0.15, 0.93] | −0.26 [−1.21, 0.13] |

- **Per-network mean responses**: T 0.20 (float s0) to 0.75 (n64); DW −0.82 to −2.36; DZ 0.90 to 2.60; N 0.10 to 0.23. All have
  jet-bootstrap intervals; see `nets_info` in the JSON.
- These p-values are not corrected over the 32 cells.

### Housekeeping: E39 figure, panel (d), redrawn
- `interp/e39_plot.py` now computes panel (d) from the E39 run-3 per-pair rows with the network-level bootstrap above (7
  networks, 2,000 draws, seed 4949) and draws the 0.5 no-information line. The axis is −0.8 to 2.5; intervals are clipped at the
  axis.
- E39 numbers at network level, as trained / untrained / random map:
  - T S 0.35 [0.07, 0.85] / 0.47 / 0.55;
  - DW S 1.00 [0.45, 1.75] / 0.70 [0.54, 1.19] / 0.60;
  - DZ S 0.04 [−0.34, 0.21] / 0.44 [0.33, 0.60] / 0.53;
  - N S 0.68 [0.34, 0.80] / 0.96;
  - Kabs S 1.09 [0.81, 1.36].
- These agree with the E39v jackknife.

## Hypotheses
- **H0** (no low-dimensional shared content): same-call is flat above the logit-only donor only at full rank. **H1**: rank ≤ 4
  already reaches ≥ 85 % same-call, and named subsets beat random subsets.
- **Measured at S**:
  - Reconstruction rank 4 gives 75.3 % [72.3, 77.9], below 85 %.
  - The task-loss rank-4 map gives 87.3 % [85.2, 89.2], above 85 %.
  - Rank 4 exceeds logit-only at S with non-overlapping network-level intervals (75.3 [72.3, 77.9] against 66.0 [63.5, 68.6]).
  - H0 as stated (gain over logit-only only at full rank) is excluded at S: rank 4 and rank 8 already exceed it.
- **Measured at h4**: rank 4 does not exceed logit-only (77.8 against 78.6, p 0.50). Rank 8 does (+8.0 points).
- **Competing hypothesis** "the shared content is A's class scores": it predicts rank 4 ≈ logit-only. It is excluded at S
  (+9.2 points, p 5e-4, 67/72 pairs Holm) and not excluded at h4.
- **Named > random subsets**: +6.6 points [+2.1, +11.9] across pairs (p 0.0045). Per pair it is not testable at Holm 0.05 with
  200 random sets. The named units alone reach 30 % (all-at-mean 19 %; full map 87 %), so they hold a small part of what the
  full map passes.

## Reading (scoped to these measurements)
- **Dimensions B's tail needs**: about 4 dimensions of A's S suffice for B's tail to reproduce B's call on 87 % of jets, when the
  map is chosen for B's loss. A map chosen to reconstruct B's whole layer needs about 8.
  - So the directions B's tail uses are fewer than the directions that carry most of B's S variance.
  - The 4 S dimensions carry more about B's decision than A's 5 logits.
  - At h4 they carry about as much as the logits, which is consistent with h4 being closer to the decision.
- **Named units**: the four named S units are mapped across networks better than random units of equal number. Passing only them
  reproduces 30–42 % of calls (k 2–4), so the low-dimensional shared content is not located in those units alone. It could be
  spread over linear combinations of several units. This was not tested here; the RRR directions could be compared with the
  named units.
- **Edit inheritance at network level**: most cells include 0.5 within their 95 % interval, or are not separated from the
  untrained or logit-only donors.
  - The S DZ response is recipient-like: full 0.04 (p 0.011 against 0.5), rank 4 −0.08. The untrained donor gives 0.34
    [0.14, 0.64]; full − untrained is −0.30 [−0.96, +0.05], p 0.066.
  - The S T response through the rank-4 map is donor-like: 0.85, p 0.021. The full map gives 0.36, and the two are not
    reconciled.
  - At h4, the untrained donor is more donor-like than the trained one for DW: full − untrained −0.39 [−0.98, −0.26]. An
    untrained first half that is near chance accuracy still passes edit-driven changes.
  - With 7 networks, the network-level intervals span about 0.5–1.0 units of I.

## Issues
- **Job status**: the job is marked failed after the final save, most likely killed for memory at exit (2 GiB). The results file
  is complete; request 3 GiB for any rerun.
- **Subset arm**: per-pair Holm cannot reach 0.05 with 200 random sets. Float nets contribute only k = 1 pairs, because their
  named roles mostly have R² < 0.9.
- **Task-loss map**: it was trained on the true labels, not on B's calls. Its gradients come from a float copy of B's tail
  (96–100 % call agreement with the exact tail).
- **Statistics**: network-bootstrap intervals with 7 or 9 networks are percentile intervals from few units; their coverage is
  approximate. Untrained donors use one redraw (seed 0).

## Files
- Code: `interp/e39x_stitch_rank_jedi.py` and `interp/e39x_plot.py` (new); `interp/e39_plot.py` (panel d redrawn).
- Results: `results/jedi/e39x.json`, `results/jedi/e39x_rank.png`, `results/jedi/e39_stitch.png` (redrawn).
- Logs: `logs/e39x-run.log`. The smoke run is in `results/jedi/e39x_smoke.json`.
- Glossary: four entries added to `site/explain.py` GLOSSARY (reduced-rank regression, task-loss stitch, subset stitch,
  network-level bootstrap).

## Explainer entry (for `site/explain.py`, Paper A tab)
- **Question**: two separately trained JEDI-linear networks can be joined at S by a linear map (E39). How many numbers does the
  map have to pass, and are they the named program elements?
- **Background**: stitching runs network A up to its object layer S, maps it linearly into network B's S, and lets B finish. If
  B then decides as it would on its own, the information B's later layers use is present in A's S in linearly recoverable form.
- **Method**: restrict the map to rank r ($W_r=WV_rV_r^\top$, reduced-rank regression), or train a rank-r map on B's
  classification loss. Count how often the joined network gives B's call (same-call %).
  - Controls: an untrained A, a random map with the same singular values, and a map from A's 5 class scores alone.
  - Separately, pass only the units for the four named roles and hold the rest of B's S at its mean, against random units of the
    same number.
  - Edit responses (translation, prong sharing, soft particle) go through the joined networks, with the inheritance index $I$
    and network-level intervals.
- **Result**: figure `results/jedi/e39x_rank.png`.
  - Rank 4 trained on B's loss gives 87 % same-call at S, against 90 % native agreement and 66 % from class scores.
  - Rank 4 fitted to reproduce B's layer gives 75 %.
  - Named units alone give 30 % against 23 % for random units.
  - Edit inheritance is mostly at the 0.5 no-information value, except the Z-jet prong-sharing response at S, which follows
    the recipient.
- **Checks**: the official n64 reaches 80.81 % on this sample, and the tails are exact. The float copy of the tail used for
  gradients agrees with the exact tail on 96–100 % of calls. Untrained donors give 31–35 % and random maps 21–23 % at every rank.
- **Why it matters**: retrained networks share a small number of decision-relevant directions at the object layer, more than
  their outputs. The named units are a part of that shared content but do not account for it, so the claim that the
  named roles are the shared program (A1s/A1x) needs the shared directions themselves to be identified.
