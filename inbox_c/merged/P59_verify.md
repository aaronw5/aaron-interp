# P59v — skeptic check of P59 (DAS for the jet mass inside L-GATr)

Job c-p59-verify (GPU, 28 min, exit 0). Script interp/p59v_verify.py (imports P59's functions unchanged). Results results/part/p59_verify.json, log logs/c-p59-verify.log.

## Conclusion

The verification reproduces P59's main finding on new files. One direction among L-GATr's 32 auxiliary scalar channels carries about 40 % of the network's response to a real jet-mass change. The direction found in P59 transfers to the new jets unchanged, and holding it fixed removes 30–50 % of that response. A new control changes the jet mass or the angular size m/pT while leaving the particle masses alone. Under this control, both the direction and the network output follow the invariant mass and not the angular spread. P59's "not the invariant mass alone" holds only for the edits that give particles unphysical masses. Still uncertain: the remaining ~60 % of the mass response is spread over other channels, and a small pT term (6–15 % at the direction) is present.

## Script review (interp/p59_lgatr_das_mass.py)

- Rows: random (default_rng), never first-k. Padded particles: masked in kin/jmass/own_spread. The embedding asserts that the token count equals particles + 3 and that the global token comes first. Batch-to-batch source mapping uses the same slice for base and source.
- Build: part_core.build uses η = jet_eta + sign(jet_eta)·Δη (the corrected convention); dilate() uses the same sign for |p|. L-GATr reads only (E, px, py, pz) and the particle-ID features, so the stored axis enters only through the rebuilt four-momenta.
- Hooks: forward hooks on model.net.blocks[k]; the tail re-runs blocks k+1..11 and linear_out with the base attention mask. The logged tail check max |Δlogit| is 0. The read-out is the global token's scalar parts (TO_CL class map).
- Interchange: U Uᵀ acts on the channel index within one grade (and on t, x, y, z together for grade 1), so it is Lorentz-equivariant. Unaligned tokens are zeroed by the token mask before the clamp(−1→0) index is used. No bug found.
- Edits: the counterfactual changes only angles (pT and particle masses fixed). M and S change particle masses (stated by P59).
- Reporting and statistics issues (not code bugs):
  1. The Holm-corrected "Gaussian p ≤ 3e-26" extrapolates a normal tail from 100 null draws. The empirical p from those draws cannot go below 0.0099, and Holm over 144 tests on empirical p would reject nothing. The effect is large, but that p-value is not measured. The bootstrap interval of the DAS value against the maximum of the null draws is a defensible statement; it is used below.
  2. All-grades d = 4: P59 says its β_cf is "near the shuffled control". In P59's own JSON (ctl) that holds only at blocks 0–2 (0.44/0.35, 0.40/0.31, 0.48/0.30). At blocks 3–9 it is 0.47–0.53 against shuffled 0.09–0.23. It contains the aux d = 4 slots (0.41–0.45), so it carries at least the aux share. The figure panel (b) shows this; the text does not.
  3. The 40-step DAS fit is fragile. On the new fit pairs at block 3 it reached β_cf 0.156 [0.142, 0.171], against 0.399 with 200 steps (below). P59's own 40-step values were not inflated: 200 steps add 0.01–0.03.
  4. P59's aux d = 4 mass subspace moves the output by F 0.20–0.29 on the pT decoy, where pT changes at fixed mass. The reading does not mention this. The d = 1 direction's response to a pT-only change is measured directly below (edit P) and is small.

## Procedure

- Model: L-GATr final checkpoint model_run0.pt (the same as P59). Unedited accuracy on the P1 sample: 83.73 % (N 3,000; expected 83.73 %). 10-class accuracy on the 3,000 new evaluation base jets: 73.8 %.
- Sample: JetClass test_20M WToQQ / ZToQQ / ZJetsToNuNu files _115 (fit) and _116 (evaluation), not used by P59 or any earlier node. 5,000 random rows per class per file (default_rng 7171 / 7172); pairs and edits default_rng 7173.
- Pairs (P59 rule): base b and source s with pT within 5 % and 0.05 ≤ |ln m_s − ln m_b| ≤ 0.3; 1,000 bases per class; 3,000 fit + 3,000 evaluation pairs (median |Δln m| 0.162). Counterfactual (cf) = b with every particle's (Δη, Δφ) about the stored axis scaled by λ until m = m_s (pT and particle masses fixed). It flips 35.6 % of 10-class calls.
- Directions tested: aux d = 1 (one direction in the 32 auxiliary scalars) at blocks 3, 4, 6, 9, 11:
  - orig_aligned / orig_global: P59's saved directions (p59_U_*.npz), not refitted, applied to the new pairs.
  - refit40 / refit200: DAS refitted on the new fit pairs with 40 or 200 Adam steps.
  - shuffled40: DAS fitted on the cf output of a random other pair.
- Metrics (3,000 evaluation pairs, 95 % bootstrap over pairs, 1,000 draws). d = l_W − l_Z (W minus Z logit).
  - IIA (interchange intervention accuracy): the fraction of pairs whose patched call equals the cf call. IIA_ch: the same on pairs the cf flips.
  - F: least-squares slope of the patched change of d on the cf change of d.
  - β_cf, β_src: joint fit Δd_patch = β_cf Δd_cf + β_src Δd_source. β_cf is the share of the patched change that follows the real mass change; β_src is the share that follows the source jet's own output.
- Null: 200 random aux directions (Haar) on the first 1,000 evaluation pairs (F and β_cf); empirical p = (1 + #null ≥ observed)/201.
- Necessity (new): on the cf jet, set the direction to the base jet's value (all tokens, or the global token only) and run the rest of the network. "Remaining" = slope of (d_nec − d_b) on (d_cf − d_b): 1 = nothing removed, 0 = the whole mass response removed. Null: 30 random aux directions.
- Edit triple at fixed particle masses (new control; 1,000 random true W + 1,000 random true Z of file _116; each a 5 % change):
  - D: angles about the stored axis × λ (median 1.051) so that m × 1.05; pT fixed. Δln m 0.049, Δln pT −0.0005, own-axis spread (|p|-weighted mean θ² about the jet momentum) +9.9 %.
  - B: every particle four-momentum × 1.05. m × 1.05 and pT × 1.05, angles fixed, own-axis spread 0. Particle masses × 1.05 (a pion 0.140 → 0.147 GeV, inside the training range, unlike P59's M edit with masses of several GeV).
  - P: pT × 1.05 with angles × λ (median 0.952) so that m is unchanged (Δln m 0, max error 1e-4). Own-axis spread −9.8 %; particle masses fixed.
  - Ratios: ratio_B = ΣΔX(B)/ΣΔX(D), ratio_P = ΣΔX(P)/ΣΔX(D), with X = network d or the direction's coordinate at the global token.
  - Predictions. Invariant-mass reader: ratio_B 1, ratio_P 0. Angular-size reader (m/pT, or the own-axis spread): ratio_B 0, ratio_P −1. A pure pT reader: ΔX(D) = 0. If X ≈ a ln m + b ln pT, then ratio_B = 1 + b/a and ratio_P = b/a.

## Numbers

Replication (aligned tokens unless stated; block: value [95 % CI]):

| block | orig_aligned β_cf | orig_global β_cf | refit200 β_cf | refit40 β_cf | shuffled40 β_cf | null β_cf 97.5 % / max | orig_aligned F | IIA / IIA_ch (orig_aligned) |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.396 [0.373, 0.423] | 0.416 | 0.399 | 0.156 | −0.129 | 0.042 / 0.052 | 0.420 [0.397, 0.444] | 0.740 / 0.325 |
| 4 | 0.401 [0.375, 0.427] | 0.444 | 0.413 | 0.374 | −0.021 | 0.050 / 0.083 | 0.450 [0.425, 0.474] | 0.741 / 0.344 |
| 6 | 0.390 [0.365, 0.415] | 0.430 | 0.422 | 0.404 | 0.026 | 0.053 / 0.075 | 0.456 [0.432, 0.481] | 0.743 / 0.356 |
| 9 | 0.399 [0.375, 0.424] | 0.435 | 0.430 | 0.415 | −0.060 | 0.057 / 0.118 | 0.454 [0.429, 0.479] | 0.740 / 0.357 |
| 11 | 0.198 [0.182, 0.213] | 0.214 | 0.214 | 0.204 | 0.048 | 0.049 / 0.061 | 0.305 | 0.677 / 0.181 |

- P59 on files _112/_104: β_cf 0.36–0.43 at blocks 3–9 and 0.19 at block 11. No patch on the new pairs gives IIA 0.644 (P59: 0.635). β_src for the DAS directions is 0.01–0.08 at blocks 3–9.
- Empirical p (F and β_cf, orig_aligned and refit40) is 0.005 at every block, the minimum for 200 draws. Every DAS β_cf CI lies above the largest of the 200 random draws. Holm over these 20 tests at a floor of 0.005 cannot go below 0.1, so the significance statement rests on that separation from the null, not on a p-value.
- Necessity (orig_aligned, all tokens; "remaining" fraction of the cf response): blocks 3 / 4 / 6 / 9 / 11 = 0.63 [0.62, 0.65] / 0.60 [0.58, 0.62] / 0.56 [0.54, 0.58] / 0.51 [0.50, 0.53] / 0.67 [0.67, 0.68].
  - Global token only: 0.60 / 0.60 / 0.52 / 0.49 / 0.67.
  - Random aux directions: 0.92–1.04 (range of 30 draws).
  - Fraction of cf-flipped calls restored to the base call: 0.33 / 0.36 / 0.38 / 0.42 / 0.22.
- Edit triple, network output d (N 2,000):
  - Mean Δd: D −0.497, B −0.419, P +0.085. Linearity check: D + P = −0.412 against B −0.419.
  - ratio_B 0.84 [0.83, 0.86], ratio_P −0.17 [−0.18, −0.16].
  - Calls changed: D 15.9 %, B 12.9 %, P 3.6 %.
- Edit triple, orig_aligned direction at the global token:

  | block | ratio_B | ratio_P |
  |---|---|---|
  | 3 | 0.93 [0.92, 0.93] | −0.05 [−0.06, −0.05] |
  | 4 | 0.94 [0.93, 0.94] | −0.06 [−0.06, −0.05] |
  | 6 | 0.88 [0.87, 0.88] | −0.10 [−0.11, −0.10] |
  | 9 | 0.84 [0.83, 0.85] | −0.14 [−0.15, −0.14] |
  | 11 | 0.78 | −0.19 |

  - Block 4, in units of the coordinate's s.d. over the 2,000 jets: D +0.285, B +0.267, P −0.016.
  - Correlation of the coordinate with ln m / ln pT / ln(m/pT) at block 4: 0.83 / 0.16 / 0.61.
  - refit and global-token directions give the same ratios within 0.03.

## Reading (scoped to what was measured)

- Reproduced: on new files, P59's saved aux direction carries β_cf 0.39–0.44 of the real-mass response at blocks 3–9 (0.20 at block 11). It is readable at the global token alone. It sits above 200 random directions (the null maximum is ≤ 0.12) and above the shuffled-pair DAS (≤ 0.03). It is also necessary for part of the response: fixing it removes 37–49 % of the cf change of d, against ≤ 8 % for random directions.
- Not reproduced as stated, interpretation:
  - Edits that keep particle masses physical give ratio_B 0.84–0.94 and ratio_P −0.05 to −0.15 for the direction, and 0.84 / −0.17 for the network output. Invariant-mass reader predicts (1, 0); angular-size / own-axis-spread reader predicts (0, −1).
  - The spread reader is excluded. The invariant mass with a small pT term fits: the coordinate ≈ ln m − b·ln pT with b ≈ 0.06 at blocks 3–4 and 0.14 at block 9, from ratio_P = −b. The network output gives b ≈ 0.17. A pT dependence of the W/Z decision is expected physically.
  - So P59's "the stored variable is not the invariant mass alone" holds only for edit M, which gives particles masses of several GeV at fixed momenta, far outside the training data. Under on-manifold edits the direction follows the invariant mass.
  - P59's "whether it tracks the angular spread remains open" is answered for this contrast: it does not.
- Grade-1 / all-grades: P59's "all-grade β_cf near its shuffled control" is wrong at blocks 3–9 (0.47–0.53 against 0.09–0.23). The grade-1 statement stands (0.13–0.21 against 0.12–0.15).
- Not checked here: grade-0, grade-1 and all-grades variants on the new files, and d = 2, 4. The untrained-model control was not rerun.

## Corrected reading

In L-GATr, one direction among the 32 auxiliary scalars carries 0.36–0.44 of the W-vs-Z response to a real mass change from block 3 to block 9. This replicates on new files with the original direction unchanged. The direction is readable at the global token. Holding it at the base jet's value removes 37–49 % of the response, against ≤ 8 % for random directions. The grade-0 scalar parts carry ≤ 0.09. The all-grades d = 4 subspace carries 0.47–0.53, above its shuffled control (0.09–0.23) from block 3 on. No d ≤ 4 subspace reaches IIA 0.8.

Under edits at physical particle masses, the direction and the output follow the invariant mass. Changing m at fixed angles (four-momenta × 1.05) gives 0.84–0.94 of the response to the same m change made with angles. Changing pT and angular size at fixed m gives −0.05 to −0.17. The own-axis spread or angular-size reader is excluded. The opposite-sign response to edit M (particle masses raised to several GeV at fixed momenta) is a response to unphysical per-particle masses.
