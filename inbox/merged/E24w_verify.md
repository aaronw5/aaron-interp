# E24w verify: prong-preserving vs prong-destroying edits in JEDI-linear (2026-10-20)

## Conclusion
On two fresh samples of two-prong test jets, JEDI-linear again changes the same share of calls whether the softer prong is rotated intact or scattered around the jet axis (difference −0.3 and −0.5 points, both intervals containing 0). A classifier that reads τ21 (N-subjettiness ratio, small for two-prong jets) at the same held radial quantities changes about 4 points more calls under scattering. Per jet, JEDI's calls after the two edits often differ; this is about as large as the difference after moving the intact prong to its mirror position, so it reflects where the prong sits and not whether it is intact. Not settled: a classifier that reads both true mass and τ21 can hide a τ21 use in this design (one such control is noisy from draw to draw), and Deep Sets shows a small positive prong response in all three samples.

## Verdict on the E24w reading
- The main reading (D − P ≈ 0 for JEDI-linear; the held-moment τ21 tree value excluded) reproduces on two fresh samples.
- Additions:
  - "Same share of calls" does not mean "same calls". P and D change mostly different jets. In E24w's own data, calls differ between P and D on 7.83 % of jets, while each edit changes 7.3–7.6 %.
  - A location-matched null (mirror rotation P2) accounts for that per-jet disagreement within its interval.
- Corrections:
  - Thmt's D − P ≈ 0 is draw-dominated: on the fresh sample Thmt gives +1.30 [0.32, 2.29], with a D2 − D null of +1.81 [0.27, 3.44].
  - Deep Sets' D − P is positive in all three samples.
  - The g share of the P − R response is 80–100 %.
- No code bug found that changes the main contrast.

## Code review (interp/e24w_prong_jedi.py, e24w_plot.py)
- Stage and arithmetic: the l1 + pooling replica PF uses the layer's own input quantizer, kernel and `qs` quantizer. It reproduces P exactly (max |ΔP| 0, in the worker run and in both verify runs). Network calls use `Jedi.run` and `run_jetwise` (g reshaped to (J, 1, 64), as in jedi_core).
- Padding:
  - `kt2` labels padding −1, so the soft mask never contains padding (`soft_label_on_padding` = 0 in verify).
  - `rot` and `scramble` touch only masked slots; padding changes are 0 in every edit.
  - Slot order (pT rank) is unchanged, so the rank-window masks see the same ranks.
- Edits:
  - pT is exact. r is held to ≤ 5e-8; M2 and Σ pT r to ≤ 7e-8.
  - Mass is computed with (Δη, Δφ) as (η, φ). Mass is invariant to a translation in η and φ, so this is correct. A rotation in the (Δη, Δφ) plane is not a Lorentz transformation; R changes m by a median 0.11 %.
- Split leakage: the tree pool is `sample(4000, 2802)` minus the main jets, by test-archive index with an assert. No overlap.
- Selection effects:
  - a is a linear combination of quantised pooled values, so a(β) is a step function. Whether |a − a0| ≤ 0.01 sd(a) can be reached depends on the sample's fitted a.
  - Acceptance varies between samples: P succeeded on 6,376 / 6,860 jets (worker), 3,457 / 6,892 (verify 1) and 4,466 / 4,880 (verify 2). R succeeded on 80 % (worker) and 30 % (verify 1).
  - The analysis set is therefore sample-dependent. The main contrast reproduces on all three sets.
- Figure (`results/jedi/e24w_prong.png`): the points and intervals match the JSON. Its title ("destroying ≈ rotating") is a summary phrase.
- Report numbers checked against `logs/e24w-run2.log` and the JSON: they agree.

## Procedure
- **Job e24w-verify** (`interp/e24w_verify.py`):
  - Ran 2026-10-20 06:25–07:09 UTC, 1 CPU / 2 GiB, wall 2,660 s, 0.52 cores. Peak 2,047 MiB, including page cache.
  - This script was in the repo from an earlier attempt of this verify node. I reviewed it and used its outputs; I did not rerun it.
  - Sample `sample(3000, 2901)`, N 15,000; JEDI accuracy 80.68 %. Same selection: 8,017 two-prong jets.
  - Same within-jet a. Edits R, P, D, plus **D2**, an independent second scramble, as the draw-to-draw null.
  - Analysis set (P and D both succeeded): N 3,457. D2 subset 3,311; R subset 1,110.
  - Controls fit on `sample(4000, 2902)` minus the main jets:
    - Trees Thm, Thmt, Tht.
    - Smooth MLPs (64-64) Mhm, Mhmt, Mht on the same features.
  - Also computed: a signed two-prong score, log P(W or Z) − log P(q or g).
- **Job e24w-verify2** (`interp/e24w_verify_p2.py`, new):
  - 1 CPU / 2 GiB, wall 630 s, 0.62 cores, peak 2,047 MiB.
  - Sample `sample(2000, 3001)`, N 10,000; JEDI 80.99 %; 5,333 two-prong jets.
  - Edits: D, D2, P (α, with m_P = m_D) and **P2**, the soft prong rotated by −α. P2 is a mirror position at the same angle to the hard prong, so it has the same m to O(R²). Required: |m_P2/m_D − 1| ≤ 1 % (median 0.39 %). Every edit ends with the a-holding β.
  - Analysis set: all four edits succeeded, N 2,251 (g 227, q 267, W 693, Z 663, t 401).
  - Read-outs:
    - Per-jet call disagreement between pairs of edited versions.
    - E_loc = 1[call(P) ≠ call(D)] − 1[call(P) ≠ call(P2)]: the disagreement beyond moving an intact prong to another place.
    - E_draw = 1[P ≠ D] − 1[D ≠ D2].
  - H0 (no integrity use): E_loc = 0. H1: E_loc > 0.
  - Statistics: 1,000-jet bootstrap and paired bootstrap p.
  - Models: JEDI official, with g held, and with S58/S31 held; Deep Sets s0; tree Tht (fit on `sample(4000, 3002)` minus the main jets).
- **Local work:** a sub-second numpy read of the saved npz files for the per-jet disagreement in verify 1 and in E24w.
- **Outputs:**
  - `results/jedi/e24w_verify.json`, `e24w_verify_calls.npz` (logits).
  - `results/jedi/e24w_verify_p2.json`, `e24w_verify_p2_calls.npz`.
  - `logs/e24w-verify.log`, `logs/e24w-verify2.log`.

## Numbers
**Replication of the main contrast (% calls changed; D − P paired, 95 % jet bootstrap; exact McNemar)**

| sample | N | JEDI P / D | JEDI D − P | g held D − P | S58/S31 held D − P | Tht D − P | Mht D − P | Deep Sets D − P |
|---|---|---|---|---|---|---|---|---|
| worker (2801) | 6,376 | 7.59 / 7.31 | −0.28 [−0.96, 0.36], p 0.44 | −0.25 | −0.30 | +4.25 [3.72, 4.80] | – | +0.86 [0.20, 1.51] |
| verify 1 (2901) | 3,457 | 7.20 / 6.91 | −0.29 [−1.19, 0.64], 119/129, p 0.57 | +0.38 [−0.29, 1.10] | +0.12 [−0.58, 0.81] | +3.53 [2.89, 4.19] | +4.17 [3.53, 4.92] | +1.07 [0.26, 1.91], p 0.020 |
| verify 2 (3001) | 2,251 | 7.55 / 7.02 | −0.53 [−1.78, 0.71], 91/103, p 0.43 | −0.58 [−1.42, 0.27] | −0.76 [−1.64, 0.18] | +3.91 [3.06, 4.75] | – | +1.51 [0.31, 2.62], p 0.018 |

- Paired JEDI − tree, verify 1:
  - vs Tht: −3.82 [−4.92, −2.72], p 0.001.
  - vs Mht: −4.45 [−5.61, −3.36], p 0.001.
- Draw-to-draw null D2 − D, verify 1:
  - JEDI +0.27 [−0.39, 1.00]; Tht +0.03; Mht +0.15.
  - Thmt +1.81 [0.27, 3.44].
- Mass + τ21 controls, verify 1:
  - Thmt D − P +1.30 [0.32, 2.29], p 0.013. Its own D2 − D null is larger, so this value is draw-dominated.
  - Mhmt D − P −0.43 [−1.36, 0.46].
  - Thm D − P −0.12; Mhm 0.00.
- Response to moving the prong, verify 1 (R subset, N 1,110):
  - JEDI P − R +2.34 [0.90, 3.87]; D − R +2.07 [0.63, 3.51].
  - g held: −0.09 / −0.09. S58/S31 held: 0.00 / −0.27.
  - Worker values: +3.24 / +3.25; with g held +0.70 / +0.54.
- Edit checks:
  - Verify 1: Δm/m P −10.51 %, D −10.51 %. Δτ21 P +0.030, D +0.374.
  - Verify 2: P2 Δm/m −8.87 %, Δτ21 +0.027. D2 Δm/m −10.2 %; D2 is not mass-matched (median |m_D2/m_D − 1| 4.8 %).

**Per-jet disagreement (verify 2, N 2,251; % of jets whose calls differ between the two edited versions)**

| model | P≠D | P≠P2 | D≠D2 | E_loc = (P≠D) − (P≠P2) | E_draw = (P≠D) − (D≠D2) | D − P (rates) |
|---|---|---|---|---|---|---|
| JEDI official | 9.06 | 8.09 | 5.51 | +0.98 [−0.31, 2.31], p 0.135 | +3.55 [2.22, 4.75] | −0.53 |
| JEDI, g held | 4.22 | 4.80 | 4.18 | −0.58 [−1.42, 0.27], p 0.19 | +0.04 [−0.84, 0.93] | −0.58 |
| JEDI, S58/S31 held | 4.93 | 5.11 | 4.40 | −0.18 [−1.11, 0.71], p 0.70 | +0.53 [−0.36, 1.38] | −0.76 |
| Deep Sets s0 | 8.97 | 6.93 | 4.98 | +2.04 [0.89, 3.11], p 0.001 | +4.00 [2.89, 5.11] | +1.51 |
| tree Tht (positive control) | 4.89 | 1.42 | 2.00 | +3.47 [2.67, 4.31], p 0.001 | +2.89 [2.00, 3.82] | +3.91 |

- Paired JEDI − Tht E_loc: −2.49 [−3.91, −1.07], p 0.001. Paired JEDI − Deep Sets E_loc: −1.07 [−2.58, 0.53], p 0.17.
- Holm over the three JEDI arms and Deep Sets (E_loc): JEDI 0.41, g held 0.41, S58/S31 held 0.70, Deep Sets 0.004.
- Verify 1 (no P2): JEDI P≠D 7.70 vs D≠D2 4.77, E_draw +2.93 [1.99, 3.90]. E24w's data: P≠D 7.83 % while P and D change 7.59 % and 7.31 %.
- Signed two-prong score (verify 1, D − P in sd units). The τ21 controls give opposite signs: Tht −0.017, Mht +0.017, Mhmt +0.038. The score is therefore not calibrated as a directional prong read-out here. JEDI's value (+0.0015 [−0.005, 0.008]) is not interpreted.

## Competing hypotheses
- **H_int (JEDI reads prong integrity as a τ21 classifier does at held moments).** Predicts D − P ≈ +3.5 to +4.3 and E_loc ≈ +3.5 (Tht, Mht).
  - JEDI D − P: −0.28, −0.29, −0.53. E_loc: +0.98, paired −2.49 vs Tht.
  - Excluded at the τ21-tree size.
- **H_loc (JEDI responds to where the soft prong's momentum sits, not to its integrity).** Predicts D − P ≈ 0 and P≠D ≈ P≠P2 > D≠D2.
  - Observed: 9.06 ≈ 8.09 > 5.51.
  - The response disappears with g held (E_draw +0.04) and with S58/S31 held (+0.53).
  - Not excluded; consistent with the data.
- **H_mask (a τ21 use hidden by a response to true mass).** Predicts D − P ≈ 0 with large P/D call changes, as in Thmt/Mhmt (17–25 % of calls).
  - JEDI changes 7–7.6 % under P and D, and about 0 points above rotation with g held. This is much smaller than any mass-reading control.
  - Not excluded by this design. Its precondition, a strong true-mass response, is not seen in JEDI.
- **Small integrity use (≤ about 2 points per jet).** The E_loc interval upper end is +2.31, so a use of that size is not excluded.

## Corrected reading
On two-prong jets (exclusive-kt 2 subjets, z_s ≥ 0.1, τ21 < 0.5), with each particle's pT and r held and the jet mass matched, JEDI-linear changes the same share of calls under a rigid rotation of the softer prong (P) and an azimuthal scatter of its particles (D). D − P is −0.28 [−0.96, 0.36] (N 6,376), and on two fresh samples −0.29 [−1.19, 0.64] (N 3,457) and −0.53 [−1.78, 0.71] (N 2,251). The held-moment τ21 tree gives +3.5 to +4.3, and a held-moment τ21 MLP +4.2. P and D change largely different jets: per jet, calls differ on 7.8–9.1 % of jets. That disagreement equals the disagreement between P and its mirror rotation P2 within the interval: E_loc +0.98 [−0.31, 2.31], against +3.47 for the τ21 tree. With g held, the disagreement drops to the draw-to-draw level. The response to moving the soft prong (P − R +2.3 to +3.2 points) passes 80–100 % through g, and through S58/S31. Not excluded:
- a τ21 use masked by a true-mass response; Thmt is draw-noisy here (D2 − D +1.81), and Mhmt gives −0.43 [−1.36, 0.46];
- an integrity response of up to about 2 points per jet.

Deep Sets s0 gives D − P > 0 in all three samples: +0.86, +1.07, +1.51; E_loc +2.04 [0.89, 3.11].

## Explainer entry (for site/explain.py; E24W_VERIFY_EXPLAINER)
- **Question:** Is E24w's "JEDI-linear does not distinguish an intact from a scattered soft prong" stable on new jets? Does the equal share of changed calls hide per-jet differences?
- **Background:** E24w compared two edits that hold $p_{T,i}$ and $r_i$ for every particle (so $M_0=\sum p_{T,i}$, $M_2=\sum p_{T,i} r_i^2$, $\sum p_{T,i} r_i$) and the jet mass $m$.
  - P rotates the softer prong rigidly: $\theta_i\to\theta_i+\alpha$ for $i\in$ soft.
  - D scatters it: $\theta_i\sim U(0,2\pi)$.
  - Equal call-change rates can arise even when the two edits flip different jets.
- **Method:**
  - Two fresh samples: sample(3000, 2901) and sample(2000, 3001).
  - Added edits:
    - D2, a second scatter, as the draw-to-draw null.
    - P2, the mirror rotation $\theta_i\to\theta_i-\alpha$. Since $m^2\approx M_0M_2-|\vec M_1|^2$ and $|\vec M_1|^2=|\vec H|^2+|\vec S|^2+2\vec H\cdot\vec S$ is unchanged by $\alpha\to-\alpha$, P2 has the same mass.
  - Per-jet disagreement $\Pr[c(P)\ne c(D)]$, compared with $\Pr[c(P)\ne c(P_2)]$: $E_{loc}=\overline{1[c_P\ne c_D]-1[c_P\ne c_{P_2}]}$.
- **Result:** D − P is −0.29 [−1.19, 0.64] and −0.53 [−1.78, 0.71]; the τ21 tree gives +3.5 and +3.9. Calls differ between P and D on 9.1 % of jets, between P and P2 on 8.1 % and between D and D2 on 5.5 %. $E_{loc}=+0.98\,[-0.31,2.31]$, against +3.47 for the τ21 tree. With g held, all disagreements fall to 4.2–5.2 %.
- **Checks:**
  - $M_0$ exact; $M_2$ and $r$ held to $7\times10^{-8}$.
  - P2 mass matched to D's: median 0.39 %.
  - Padding never edited.
  - l1 + pooling replica exact.
  - Tree pool disjoint from the analysis jets.
- **Why it matters:** The E24w null replicates on two new samples. JEDI-linear's per-jet sensitivity to the soft prong is about its location, read through the jet context g, and not about its integrity. A τ21 use hidden by a mass response remains untested by this design. Deep Sets shows a small integrity response.
