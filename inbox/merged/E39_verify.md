# E39 verification: model stitching across the 10 trained JEDI-linear networks (skeptic rerun)

**Conclusion.** On a fresh set of jets, joining two independently trained JEDI-linear networks with one fitted linear map at
the S layer (the second-round sums over particles) or at h4 (the first dense layer after it) again reproduces the receiving
network's call on about 86 % (S) to 89 % (h4) of jets. A new control maps only the donor's five class scores into the receiver,
and it reaches 66 % (S) and 78 % (h4). So the layer carries more shared, linearly readable information than the donor's final
decision alone. The edit-inheritance statements do not hold up once the uncertainty from having only 7 networks is counted.
The Z-jet prong-sharing response follows the receiving network. The W-jet response "following the donor" also appears when
the donor is an untrained redraw or only passes its class scores. The translation-edit index is consistent with "no
information". "No information" corresponds to I = 0.5, not I = 0 as E39 assumed.

## Terms
- **Stitch**: run donor network A up to layer L (S or h4) and apply an affine map (linear map plus offset) fitted by ridge
  regression. Clip each unit to recipient B's range on the fit jets, then run B's exact fixed-point later layers.
- **Same-call with B**: the share of test jets where the stitched argmax class equals B's own unedited argmax.
- **Native agreement**: the share of jets where unedited A and unedited B make the same call.
- **Output-only donor (logit donor, new control)**: the same ridge + clip + B's tail, but the map's input is A's 5 output logits
  (class scores) instead of A's S or h4. It can carry A's decision and its confidence, and nothing else from A's internal layer.
- **Inheritance index**: $I = (r_{\rm stitch}-r_B)/(r_A-r_B)$. Here $r_X$ is the mean per-jet change of network X's output under
  an input edit. I = 1 means the stitched network responds like donor A; I = 0 means it responds like recipient B.
  - Pooled index: $\sum (r_s-r_B)(r_A-r_B)/\sum (r_A-r_B)^2$ over "eligible" ordered pairs, those with $|r_A-r_B| > 2$
    bootstrap SE.
- **Network jackknife**: recompute a pooled quantity 7 times. Each time, leave out one of the 7 non-degenerate edit networks
  and every pair that contains it. $SE = \sqrt{(k-1)/k\,\sum(\theta_i-\bar\theta)^2}$, interval = estimate ± 1.96 SE. This counts the
  variation between networks, which a jet bootstrap does not.
- **Regression b, c**: least-squares fit $r_{\rm stitch} = a + b\,r_A + c\,r_B$ over the 42 non-degenerate ordered pairs.
  b measures dependence on the donor's own response, and c dependence on the recipient's. SEs come from the network jackknife.
- **McNemar, Holm**: exact binomial test on jets where exactly one of (stitched, B) is correct; Holm step-down correction over
  the 90 pairs.

## Code check (interp/e39_stitch_retrains_jedi.py, interp/e39_plot.py)
- **Stages**: S is the `qs1` pool output and h4 is the `d4` output, as in jedi_core. `Tail` applies B's own input quantizer of
  d4 (from S) or d5 (from h4).
  - E39: tail check max |Δlogit| = 0 for quantized networks and ≤ 2.4e-6 for the float networks.
  - Fresh run: same values (max |Δlogit| 0 / ≤ 2.4e-6).
- **Split**: test rows that are also fit rows are removed (E39 401, fresh 375). The ridge hold-out uses fit jets only. No
  leakage was found.
- **Padding**: real particles are those with pT > 0.
  - T shifts only real particles.
  - N writes into the first padded slot, then re-sorts by pT (sorted check true).
  - K requires ≥ 15 real particles.
  - No padded slot is counted as a particle.
- **D reference**: the unedited reference for D is the rebuilt jet (`model_x`), not the stored one. The rebuild differs from
  the stored inputs by up to 0.32 (E39) / 0.28 (fresh). This is most likely float16 rounding of stored pT for hard particles;
  it was not checked further. Edited and reference jets go through the same rebuild, so the response is not affected. The
  stitching map, however, was fitted on stored jets.
- **Bug found in the verifier's own first job** (e39-verify): McNemar was not computed for untrained pairs, which raised a
  KeyError after the stitching finished. Its log is kept in logs/e39-verify.log. The fixed rerun is e39-verify2, and all
  numbers below come from it.
- **Analysis issue in E39 (not a code bug)**: the pooled I CIs come from a jet bootstrap only. The between-network variation,
  which dominates with 7 networks, is not included. For DW at S the network-jackknife interval is about 6× wider (below).
- **Analysis issue in E39**: eligibility $|r_A-r_B| > 2$ SE is symmetric in A and B, so the eligible set holds both (A, B) and
  (B, A). If the stitched response is any constant c in every pair, the pooled I is then exactly 0.5:
  $(c-r_B)(r_A-r_B)+(c-r_A)(r_B-r_A) = (r_A-r_B)^2$.
  - Checked numerically: 0.500 in every cell.
  - E39 describes the null as "$I = -r_B/(r_A-r_B)$, non-zero". The pooled no-information value is 0.5, so "nearer the
    recipient" requires I clearly below 0.5, not just below 1.

## Procedure (fresh run)
- **Script**: `interp/e39_verify.py`. It reuses `ridge`, `apply`, `Tail`, `build_edits`, `resp` and `redrawn` from E39
  unchanged.
- **Jobs**: e39-verify (crashed at the summary step), then e39-verify2. Each ran on 1 CPU / 2 GiB, wall time 1,086 s, with peak
  memory at the 2 GiB limit (including page cache).
- **Samples**:
  - Fit: `sample(2000, 4939)`, N = 10,000.
  - Test: `sample(2000, 4940)` minus the 375 rows shared with fit, N = 9,625.
  - Overlap with E39: 760 of the test rows and 726 of the fit rows were also in E39's fit or test rows.
- **Networks**: the same 10 (E33 `Model`).
  - Official n64 accuracy on the fresh test jets: 80.81 % [79.96, 81.64], inside the 80.6–81.3 % band.
- **Edits**: the same definitions as E39.
  - T (W jets, N = 1,921): every real particle's Δη + 0.06 rad.
  - D, dc = +0.3: N = 3,858 W+Z jets, all accepted, max relative ΔP4 = 1.4e-13.
  - N (q+g jets, N = 2,675): one 1 GeV particle added at r = 0.3.
  - K (N = 9,583): angles of the pT-rank 13 and 14 particles swapped.
- **Controls**:
  - Untrained redraw of A: seed 0 only (E39 averaged seeds 0–2).
  - New: output-only (logit) donor.
  - Random-map and shuffled-row controls were not rerun (E39: 20–23 % same-call).
- **Statistics**:
  - 1,000-draw jet bootstrap.
  - Per pair: sign-flip permutation (2,000 draws) of the per-jet same-call difference, with Holm over 90 pairs.
  - Across pairs: sign-flip permutation (10,000 draws).
  - Network jackknife for every pooled I and for the regression coefficients.
- **Re-analysis**: the same jackknife and regression applied to E39's own run-3 rows (`results/jedi/e39.json`).
- **Outputs**: `results/jedi/e39_verify.json`, figure `results/jedi/e39_verify.png`, logs `logs/e39-verify.log` and
  `logs/e39-verify2.log`.

## Results

### Same-call (90 ordered pairs; "nd" = without the degenerate HGQ s1 network, 72 pairs)
| | E39 S | fresh S | E39 h4 | fresh h4 |
|---|---|---|---|---|
| trained-donor same-call with B, mean (nd) | 86.2 % (87.2 %) | 86.3 % (87.2 %) | 89.1 % (90.2 %) | 89.1 % (90.2 %) |
| untrained-donor same-call | 34.3 % | 34.2 % | 32.7 % | 33.4 % |
| **output-only (logit) donor same-call (new)** | – | 66.1 % | – | 78.0 % |
| native A–B agreement (nd) | 89.9 % | 90.1 % | 89.9 % | 90.1 % |
| stitched accuracy − B accuracy (nd) | −4.2 pts | −4.2 pts | −1.6 pts | −1.5 pts |
| map test R² (nd), trained donor / logit donor | 0.88 / – | 0.89 / 0.55 | 0.88 / – | 0.88 / 0.53 |
| McNemar Holm < 0.05 (of 90) | 66 | 65 | 62 | 59 |

- **Trained vs untrained donor**
  - H0: trained-donor same-call ≤ untrained-donor same-call. H1: trained-donor same-call is greater.
  - Result: Holm p < 0.05 in 90/90 pairs at both layers; across pairs p = 1e-4.
  - Mean difference (nd), with jet-bootstrap interval: +51.8 pts [51.4, 52.2] at S and +56.1 pts [55.7, 56.5] at h4.
- **Trained S/h4 donor vs output-only donor (new)**
  - H0: a map from A's S (or h4) gives no higher same-call than a map from A's 5 logits. H1: it gives higher same-call.
  - Mean difference (nd):
    - At S: +20.6 pts [20.1, 21.2], per-pair range +4.6 to +30.2.
    - At h4: +10.6 pts [10.2, 11.0], per-pair range +1.8 to +16.7.
  - Holm p < 0.05 in 90/90 pairs at each layer; across pairs p = 1e-4.
  - Caveat: the affine map from 5 logits into B's 64 units has much less capacity (R² 0.55). The control rules out "the
    stitch only transmits A's output", but it does not by itself prove that the internal coordinates match.
- **Jets where A and B disagree.** Share of jets where the stitched call equals A's / B's (nd mean); sign-flip p is for
  follows-A − follows-B over pairs:

  | layer | E39 (A / B, p) | fresh (A / B, p) |
  |---|---|---|
  | S | 47.5 / 44.6 %, p = 0.04 | 47.4 / 44.9 %, p = 0.14 |
  | h4 | 55.1 / 39.2 %, p = 1e-4 | 55.1 / 39.3 %, p = 1e-4 |

  - The h4 lean toward the donor reproduces. At S, the small difference between A and B is not significant on fresh jets;
    "splits about evenly" is the better description.

### Edit inheritance (7 non-degenerate edit networks, 42 ordered pairs)
Each cell gives the pooled I with its network-jackknife 95 % interval. The second line of the trained-donor cell gives E39's
jet-bootstrap interval for the same quantity. The constant-response null is 0.500 in every cell.

| response | layer | E39 rows: trained | fresh: trained | fresh: logit donor | fresh: untrained (s0) | fresh: trained − untrained | fresh: trained − logit |
|---|---|---|---|---|---|---|---|
| T (W→Z flow) | S | 0.35 [−0.14, 0.84]<br>(jet: [0.33, 0.38]) | 0.38 [−0.07, 0.83] | 0.68 [0.37, 1.00] | 0.27 [−0.14, 0.67] | +0.11 [−0.12, 0.34] | −0.30 [−0.98, 0.38] |
| T | h4 | 0.39 [−0.09, 0.87] | 0.45 [0.02, 0.87] | 0.92 [0.66, 1.19] | 0.32 [−0.03, 0.66] | +0.13 [−0.22, 0.48] | −0.47 [−0.93, −0.02] |
| DW (Δ(ℓ_W−ℓ_Z), W jets) | S | 1.00 [0.62, 1.39]<br>(jet: [0.95, 1.06]) | 0.97 [0.62, 1.32] | 0.67 [0.38, 0.97] | 0.72 [0.44, 1.01] | +0.25 [−0.16, 0.65] | +0.30 [−0.30, 0.89] |
| DW | h4 | 0.56 [0.27, 0.86] | 0.54 [0.24, 0.84] | 0.63 [0.15, 1.10] | 0.92 [0.61, 1.23] | −0.38 [−0.59, −0.17] | −0.09 [−0.85, 0.67] |
| DZ (Δ(ℓ_W−ℓ_Z), Z jets) | S | 0.04 [−0.26, 0.34]<br>(jet: [−0.04, 0.10]) | 0.12 [−0.08, 0.33] | 0.24 [0.06, 0.42] | 0.38 [0.16, 0.59] | −0.26 [−0.62, 0.11] | −0.12 [−0.30, 0.07] |
| DZ | h4 | 0.26 [−0.05, 0.56] | 0.30 [0.05, 0.56] | 0.26 [−0.08, 0.60] | 0.23 [0.09, 0.36] | +0.08 [−0.10, 0.25] | +0.04 [−0.20, 0.29] |
| N (Δ(ℓ_g−ℓ_q)) | S | 0.68 [0.35, 1.00] | 0.71 [0.52, 0.90] | 0.65 [0.23, 1.06] | 0.90 [0.42, 1.38] | −0.19 [−0.64, 0.25] | +0.06 [−0.26, 0.38] |
| N | h4 | 0.51 [0.08, 0.93] | 0.52 [0.16, 0.88] | 0.72 [0.13, 1.30] | 0.80 [0.38, 1.21] | −0.28 [−0.88, 0.33] | −0.20 [−0.43, 0.04] |
| Kabs (\|Δ log p(true)\|) | S | 1.09 [0.79, 1.40] | 1.11 [0.84, 1.37] | 1.70 [1.19, 2.22] | 0.18 [−0.14, 0.50] | +0.92 [0.67, 1.18] | −0.60 [−0.89, −0.30] |
| Kabs | h4 | 0.97 [0.86, 1.08] | 0.98 [0.88, 1.07] | 1.35 [1.21, 1.50] | 0.33 [0.12, 0.53] | +0.65 [0.51, 0.79] | −0.37 [−0.56, −0.19] |

- **Point estimates.** All trained-donor values reproduce within 0.08 of E39.
- **Accuracy-matched subset (fresh)**: stitches whose accuracy is at least B's − 2 points.
  - DW at S: 0.33 (16 pairs; E39 0.42).
  - DZ at S: −0.02.
  - T: 0.27 at S and 0.20 at h4.
  - N: 0.14 at S and 0.08 at h4.
- **Regression $r_{\rm stitch} = a + b\,r_A + c\,r_B$, at S (network-jackknife SE):**

  | response | trained donor | logit donor | untrained donor |
  |---|---|---|---|
  | DZ | b = 0.14 ± 0.15, c = 0.90 ± 0.09 (E39 rows: 0.00 ± 0.20, 0.93 ± 0.12) | c = 0.34 ± 0.23 | c = 0.04 ± 0.13 |
  | DW | b = 0.73 ± 0.21, c = −0.20 ± 0.30 (E39 rows: 0.78 ± 0.24, −0.22 ± 0.29) | b = 0.56 ± 0.17 | b = 0.50 ± 0.30 (E39 3 seeds: 0.44 ± 0.17) |
  | T | b = 0.16 ± 0.36, c = 0.40 ± 0.15 | | |

- **Kabs**: the untrained-donor value moves from 1.16 (E39, mean of 3 seeds) to 0.18 (seed 0 only), and the logit donor
  exceeds the trained donor. Kabs measures response size, not direction. Its I is not used for donor vs recipient.

### Competing readings and which are excluded

| reading | prediction | outcome |
|---|---|---|
| (a) The stitch only passes A's final decision | Same-call of the S/h4 stitch ≈ same-call of the logit donor | Excluded: +20.6 pts (S) / +10.6 pts (h4), 90/90 pairs |
| (b) The stitch passes no trained structure | Same-call ≈ untrained-donor same-call | Excluded: +52 to +56 pts |
| (c) The DW response is carried by A's trained S representation | I(DW, S) above both the untrained-donor and logit-donor controls | Not shown: differences +0.25 [−0.16, 0.65] and +0.30 [−0.30, 0.89]. The donor dependence b is also present with untrained donors (0.50) and logit donors (0.56). |
| (d) The DZ response is set by B's tail | c ≈ 1 for the trained stitch, while it is low for controls whose S input lacks trained structure | Supported: c = 0.90 ± 0.09 vs 0.04 (untrained) and 0.34 (logit). I(DZ, S) is below 0.5 with interval [−0.08, 0.33]. Its difference from the untrained control is significant on E39's rows (−0.40 [−0.77, −0.04]) but not on fresh jets (−0.26 [−0.62, 0.11]). |
| (e) T follows the recipient | I(T) clearly below 0.5 | Not shown: 0.38 [−0.07, 0.83] (S) and 0.45 [0.02, 0.87] (h4) both include 0.5 |

## Figures
- `results/jedi/e39_stitch.png` (E39), panel (d):
  - The bars carry jet-bootstrap intervals only. These are 5–10× narrower than the between-network uncertainty.
  - There is no line at I = 0.5, the pooled no-information value. Only I = 0 and I = 1 are marked.
  - Suggested fix: add a dashed line at 0.5 and network-jackknife intervals. Panels (a)–(c) read correctly.
- `results/jedi/e39_verify.png` (new): pooled I on fresh jets for the trained donor, the logit donor and the untrained donor,
  with network-jackknife 95 % intervals and the 0.5 null line.

## Corrected reading
- **Same-call reproduces on fresh jets.** One ridge affine map lets B's later layers reproduce B's call on 86.3 % (S) and
  89.1 % (h4) of jets. Untrained donors give 34 %, and a map from A's 5 class scores alone gives 66 % (S) and 78 % (h4). The
  native A–B agreement is 90.1 %.
- **Disagreement jets.** At h4, the stitched call follows A more often than B (55 vs 39 %, p = 1e-4). At S the split is even
  (47 vs 45 %, p = 0.14). The E39 p = 0.04 at S does not reproduce.
- **Edit inheritance with network-level uncertainty.** With 7 networks counted and I = 0.5 as the no-information value:
  - The Z-jet prong-sharing response at S depends on the recipient (c = 0.90 ± 0.09; I = 0.12 [−0.08, 0.33]).
  - The W-jet prong-sharing response at S depends on the donor (b = 0.73 ± 0.21; I = 0.97 [0.62, 1.32]). It is not
    separated from untrained-donor or logit-donor stitches, which show the same donor dependence. It is therefore not shown to
    be carried by A's trained S representation.
  - T (I 0.38–0.45) and N are consistent with 0.5 or with the controls.
  - The E39 jet-bootstrap intervals understate the uncertainty of the pooled I.

## Files
- Script: `interp/e39_verify.py`
- Results: `results/jedi/e39_verify.json` (also on the volume under `$AI_RESULTS/jedi/`)
- Figure: `results/jedi/e39_verify.png`
- Logs: `logs/e39-verify.log` (first attempt, crashed at the summary step), `logs/e39-verify2.log`
- Plot: `interp/e39_verify_plot.py <json> <png>`, rendered locally with matplotlib (a few seconds).
