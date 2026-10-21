# E39: model stitching across the 10 trained JEDI-linear networks (loop A round 18, stage 3, exploratory)

**Conclusion.** We took each trained JEDI-linear network's internal layer (S, the second-round sums over particles, or h4, the
first dense layer after it). We mapped it with one fitted linear-plus-offset (affine) map into a different trained network's
coordinates and let the second network finish the classification. The joined ("stitched") network gives the second network's
call on about 86 % (S) to 89 % (h4) of jets. That is about as often as the two networks agree with each other unedited (90 %).
The same procedure with an untrained network as the first half gives about 34 %. So the information that later layers use is
recoverable by a linear map between retrains. But on jets where the two networks disagree, the stitched call follows neither
network clearly at S, and follows the donor more often at h4. "The same computation up to a change of basis" is therefore not
separated from "different computations whose outputs a linear map can translate". Edit responses do not give one answer: the
W-jet response to more lopsided prong sharing follows the donor, the Z-jet response follows the recipient, and the other edits
fall in between or are matched by the untrained-donor control.

## Terms
- **Stitching**: run network A up to layer L (S or h4) and apply an affine map W. Then clip each unit to recipient B's own range
  and run B's remaining layers. B's exact fixed-point tail is used, with B's own input quantizer of the next layer.
- **Affine map**: $\hat L_B = (L_A - \bar L_A)\,W + \bar L_B$. It is fitted by ridge regression on the fit jets in float64. The
  ridge strength $\lambda = \alpha\,\mathrm{tr}(X_c^\top X_c)/64$ uses $\alpha \in [10^{-7}, 10^{-1}]$, chosen on a 20 % hold-out.
- **Clip**: B's HGQ input quantizer wraps values that fall outside its range, so every stitched unit is clipped to B's [min, max]
  on the fit jets. Clipped share of unit values: S 3.1 %, h4 17.7 %. Without the clip, stitched accuracy is 15 (S) and 24 (h4)
  points lower.
- **Same-call % with B**: the share of test jets where the stitched argmax class equals B's own unedited argmax.
- **Native agreement**: the share of jets where unedited A and unedited B give the same call.
- **Inheritance index** $I_e = (r_{\rm stitch} - r_B)/(r_A - r_B)$. Here $r_X$ is network X's mean response to edit $e$. I = 1
  means the stitched network responds like donor A; I = 0 means it responds like recipient B. I is computed only on pairs with
  $|r_A - r_B| > 2$ bootstrap SE. The pooled value is $\sum (r_s - r_B)(r_A - r_B) / \sum (r_A - r_B)^2$ over eligible pairs.
- **McNemar**: an exact binomial test on the jets where exactly one of (stitched, B) is correct. **Holm**: step-down correction
  for multiple tests (here over 90 pairs).

## Procedure
- **Networks**: official 3f n64 (reference), 3f n32, 3f n128, perm-inv n64, 16f n64, X12 HGQ-r150 s0/s1/s2, X12 float s0/s1.
  All are loaded with `e33_program_recurrence_jedi.Model`. HGQ s1 (57 %) is in every table but left out of pooled numbers.
- **Groups**: Q3 = quantized 3-feature networks (n64, n32, n128, HGQ s0, s2); F3 = float s0, s1.
- **Fit jets**: `sample(2000, 3939)`, 10,000 jets.
- **Test jets**: `sample(2000, 3940)` minus the 401 rows shared with fit, N = 9,599. n128 and 16f read the same rows from
  their own files (copy check max |Δx| = 0).
- **Exactness**: B's tail run from its own S or h4 reproduces its logits (max |Δlogit| 0, float nets 2e-6).
- **Reference accuracy**: official n64 is 80.43 % [79.57, 81.20] on this sample. That is 0.2 points under the 80.6–81.3 %
  band, but the interval covers it. E33x reports the same 80.43 % on its own sample with the same loader.
- **Stitches**: 90 ordered pairs × 2 layers, plus 10 self-stitches.
- **Controls**:
  - (i) A random affine map with the fitted map's singular values (Haar-random orthogonal factors).
  - (ii) Untrained donors: redraws of A (`redraw_untrained`, seeds 0–2, trained number formats kept), each with its own ridge
    map into B.
  - (iii) A map fitted on shuffled rows (jet i of A paired with a random jet of B). This keeps the marginals and removes jet
    identity.
  - (iv) Self-stitch A = B through the ridge map, which gives the fitting-loss floor.
- **Statistics**:
  - 1,000-draw jet bootstrap.
  - Per pair: a one-sided paired sign-flip permutation (5,000 draws) of the per-jet same-call indicator, trained donor vs the
    mean of 3 untrained donors. Holm over 90 pairs per layer.
  - Across pairs: a sign-flip permutation (10,000) on the 90 paired differences.
  - McNemar for stitched vs B correctness.
- **Edits**: real input edits on the 8 networks with a 3-feature 64-slot input. n32 sees the first 32 slots of the edited jet.
  n128 and 16f are excluded because the edits do not define their extra slots and columns.
  - **T**: every real particle's Δη + 0.06 rad, inputs kept relative to the stored E-scheme axis (E6). True W jets (N = 1,931).
    Response = 1[called Z after] − 1[called Z before], the W→Z flow.
  - **D**: E35/H45 prong-sharing edit `edit_one` with dc = +0.3. The jet is boosted to its rest frame and the |cos θ*| of the two
    exclusive-kT prongs is raised by 0.3. Jet four-momentum, prong masses and N stay fixed (max relative ΔP4 0). Inputs are
    rebuilt about the E-scheme axis, and the reference is the rebuilt unedited jet. Response = Δ(ℓ_W − ℓ_Z), on W jets (DW)
    and Z jets (DZ) separately (3,838 W+Z jets).
  - **N**: one particle added with pT 1 GeV at r = 0.3 from the axis and random azimuth, inserted at its pT rank. q+g jets
    with ≤ 63 particles (N = 2,670). Response = Δ(ℓ_g − ℓ_q).
  - **K**: (Δη, Δφ) of pT ranks 13 and 14 swapped (E20 A13), on jets with ≥ 15 particles (N = 9,558). Response = Δ log
    p(true class), signed (K) and absolute (Kabs).
- **Secondary**: Spearman correlation between same-call at S and the number of E33x named roles (r2, soft, hard, core;
  R² ≥ 0.9) present in both networks.
- **Script and outputs**:
  - `interp/e39_stitch_retrains_jedi.py`
  - Jobs: e39-smoke, e39-run (run 1), e39-run3 (final), each 1 CPU / 2 GiB, about 810 s.
  - Peak memory was at the 2 GiB limit, including CephFS page cache.
  - Results: `results/jedi/e39.json` (final), `results/jedi/e39_run1.json`
  - Figure: `results/jedi/e39_stitch.png` (`interp/e39_plot.py`)
  - Logs: `logs/e39-run3.log`

**Run history.** Run 1 had two problems, and both were changed in run 3 (same seeds):
- It pooled W and Z jets into one D response, but their responses have opposite sign (E37v), so the mean cancelled.
- Its signed K was near 0 in every network.

Run 3 splits D into DW and DZ, adds Kabs, and adds the disagreement-jet metric. All other numbers are identical between runs.

## Results

### Same-call (90 pairs per layer)

| | S | h4 |
|---|---|---|
| trained-donor same-call with B, mean (non-degenerate 72 pairs) | 86.2 % (87.2 %) | 89.1 % (90.2 %) |
| range (non-degenerate) | 74.9–94.2 % | 83.0–94.2 % |
| untrained-donor same-call with B | 34.3 % | 32.7 % |
| random map, singular values matched | 23.3 % | 23.4 % |
| shuffled-row map | 20.0 % | 21.6 % |
| self-stitch (A = B) | 99.3–100 % | 99.3–100 % |
| native A–B agreement, mean (non-degenerate) | 89.9 % | 89.9 % |
| stitched same-call with A (non-degenerate) | 87.5 % | 91.8 % |
| map test R², trained / untrained donor | 0.88 / 0.27 | 0.88 / 0.19 |
| stitched accuracy − B accuracy (non-degenerate) | −4.2 pts | −1.6 pts |

- **H0** (trained-donor same-call ≤ untrained-donor same-call) is rejected in 90/90 pairs at both layers.
  - Per-pair Holm p ≤ 0.018.
  - Smallest per-pair difference: +22.6 pts (S), +33.6 pts (h4).
  - Across pairs: p = 1e-4.
- **McNemar**: stitched correctness differs from B in 66/90 (S) and 62/90 (h4) pairs (Holm < 0.05). In 78 (S) and 61 (h4) pairs
  B is the more often correct.
- **By group**, trained same-call / native agreement / accuracy change:
  - Q3 → Q3: 86.1 % / 88.0 % / −4.1 pts at S; 88.9 % / 88.0 % / −1.7 pts at h4.
  - Q3 → F3: 79.3 % at S, 87.4 % at h4.
  - F3 → Q3: 90.7 % at S, 91.0 % at h4.
  - Cross-family donors at S: perm-inv → Q3 90.1 %, 16f → Q3 89.6 %.
  - The n32 donor is lowest (76–87 %).
  - Untrained float donors reach 55–62 %; untrained quantized donors reach 19–40 %.
- **Jets where A and B disagree** (mean 971 per pair, 72 pairs). Share where the stitched call equals A's / B's:
  - S: 47.5 % / 44.6 %. A > B in 53 % of pairs; sign-flip p = 0.04.
  - h4: 55.1 % / 39.2 %. A > B in 92 % of pairs; p = 1e-4.
- **Secondary**: same-call does not track the E33x named-role overlap. Spearman ρ = −0.10 (p 0.40) at S and −0.19 (p 0.10) at h4.

**Thesis predictions vs observed:**
- Same-call ≥ 90 % at S for Q3: observed 86.1 %, with native agreement 88.0 %.
- Untrained ≤ 40 %: observed 27 % for Q3 donors and 34 % overall.

### Edit inheritance (56 pairs of the 8 edit networks; pooled over eligible pairs without HGQ s1; 95 % jet bootstrap)

| response | layer | eligible | I trained donor | I acc-matched (stitched acc ≥ B − 2 pts) | I untrained donor | I random map |
|---|---|---|---|---|---|---|
| T (W→Z flow) | S | 40 | 0.35 [0.33, 0.38] | 0.26 (18) | 0.47 | 0.55 |
| T | h4 | 40 | 0.39 [0.37, 0.41] | 0.14 (25) | 0.34 | 0.45 |
| DW (Δ(ℓ_W−ℓ_Z), W jets) | S | 38 | 1.00 [0.95, 1.06] | 0.42 (17) | 0.70 | 0.60 |
| DW | h4 | 38 | 0.56 [0.53, 0.60] | 0.38 (23) | 0.86 | 0.64 |
| DZ (Δ(ℓ_W−ℓ_Z), Z jets) | S | 38 | 0.04 [−0.04, 0.10] | −0.05 (18) | 0.44 | 0.53 |
| DZ | h4 | 38 | 0.26 [0.22, 0.30] | 0.07 (23) | 0.26 | 0.43 |
| N (Δ(ℓ_g−ℓ_q)) | S | 40 | 0.68 [0.65, 0.70] | 0.16 (18) | 0.96 | 0.70 |
| N | h4 | 40 | 0.51 [0.48, 0.53] | 0.07 (25) | 0.91 | 0.29 |
| Kabs (\|Δ log p(true)\|) | S | 40 | 1.09 [1.05, 1.13] | 0.82 (17) | 1.16 | 1.88 |
| Kabs | h4 | 40 | 0.97 [0.94, 1.00] | 0.92 (24) | 0.92 | 1.47 |
| K signed | both | 0 | no pair passes 2 SE (all \|r\| ≤ 0.0015) | | | |

- **Unedited-network responses (8 edit nets, non-degenerate):**
  - T: 0.19 (float s0) to 0.74 (official n64).
  - DW: −0.76 to −2.27.
  - DZ: +0.93 to +2.66.
  - N: 0.10 to 0.23.
- **Untrained-donor and random-map stitched responses** are near 0 for T (−0.02 / −0.07) and much smaller for D. A non-zero I
  in those columns therefore comes from the index's arithmetic, with $r_{\rm stitch}\approx 0$ giving $I = -r_B/(r_A-r_B)$. It
  is not a donor response. The trained-donor I is informative only where it departs from that column.
- **Per-pair regression** across 42 pairs, $r_{\rm stitch} = a + b\,r_A + c\,r_B$:
  - DW at S: b = 0.78, c = −0.22.
  - DZ at S: b = 0.00, c = 0.93.
  - T at S: b = 0.11, c = 0.41.
  - N at S: b = 0.74, c = 0.39. The untrained donor gives b = 1.04 here, so the donor dependence of N is also present with
    redrawn weights and kept number formats.

## Reading (scoped to these measurements)

1. **Layer-level stitching.** An affine map recovers enough of each retrain's S or h4 for another retrain's tail to give B's
   call on 86–89 % of jets. Untrained donors (34 %), a random map (23 %) and a shuffled-row map (20 %) all fall far below. The
   stitched same-call stays within the native agreement of the two networks (90 %), and the stitched network costs 1.6–4.2
   points of accuracy.
2. **Whose decisions the stitched network carries.** On the jets where A and B disagree, the stitched call splits about evenly
   at S (47 % A, 45 % B) and leans to A at h4 (55 % A, 39 % B). A high same-call % alone cannot distinguish "shared basis" from
   "A's computation translated into B's coordinates". The h4 result shows that the map carries part of A's own call
   boundaries into B's tail.
3. **Edit responses.**
   - The W-jet prong-sharing response at S follows the donor (I = 1.00 [0.95, 1.06]), and the Z-jet response follows the
     recipient (I = 0.04 [−0.04, 0.10]). On W jets, where E37 recurrence is 9/10, the response that differs between networks
     is carried in the donor's S representation. On Z jets it is set by the recipient's tail.
   - Restricted to accuracy-matched stitches, DW at S falls to 0.42, so the donor reading weakens on the stitches closest to B
     in accuracy.
   - T stays between donor and recipient, nearer B (0.35–0.39), and nearer still in accuracy-matched stitches (0.14–0.26).
   - N and Kabs are matched or exceeded by the untrained-donor or random-map control, so I does not identify their location.
4. **Retrain stability vs named units.** Stitch quality does not follow the number of E33x named roles shared by a pair. The
   linear recoverability of the layer is not explained by unit-level recurrence of the four named formulas.

## Issues / limits
- The thesis criteria are only partly met. Same-call at S for Q3 is 86 % (prediction ≥ 90 %); native agreement is the relevant
  ceiling and is itself 88 %.
- I is unstable when $|r_A - r_B|$ is small, and the untrained-donor column shows its non-zero null. Statements rest on the
  contrasts with that column and on the accuracy-matched subset.
- One map family (ridge affine) is used. A map trained on B's loss, as in P45, may raise same-call.
- The clip is needed because HGQ quantizers wrap out-of-range values. h4 needs it for 18 % of values.
- n128 and 16f are excluded from the edit arm.
- HGQ s1 is reported but left out of pooled numbers.
- Peak memory reached the 2 GiB request (cgroup peak includes page cache). The run completed.
- Not yet verified by a skeptic rerun on fresh samples.

## Explainer entry (for site/explain.py, Paper A tab)
- **Question**: do retrained JEDI-linear networks build the same second-round layer up to a change of coordinates?
- **Background**: E33x tested named units one at a time. Stitching tests the whole layer, and uses the recipient's own later
  layers as the judge of function.
- **Method**: $\hat S_B = (S_A - \bar S_A) W + \bar S_B$ (ridge), clip to B's range, then B's fixed-point tail. Reported:
  same-call %, accuracy, and the inheritance index $I = (r_{\rm stitch} - r_B)/(r_A - r_B)$ under translation, prong-sharing,
  soft-particle and rank-swap edits.
- **Result with figure**: `results/jedi/e39_stitch.png` (panels a–d).
- **Checks**: untrained donors, random-spectrum and shuffled-row maps, self-stitch, accuracy-matched subset, tail exactness.
- **Why it matters**: it shows which parts of the program are a property of the architecture and training, and which belong
  to one checkpoint.

**Glossary additions**: stitching, affine map, same-call %, native agreement, inheritance index I, clip (HGQ wrap).
