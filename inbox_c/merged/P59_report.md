# P59 — Distributed alignment search for the jet mass inside L-GATr (claims C22, C25, C6 v2; ParT comparison C7)

Status: done (exploratory). Jobs c-p59-b01, c-p59-b23, c-p59-b456, c-p59-b7to11 (main, GPU), c-p59-ctl-a/b/c (follow-up controls, GPU), c-p59-merge3 (CPU); smoke c-p59-smoke.
Script interp/p59_lgatr_das_mass.py. Results results/lgatr/p59_lgatr_das_mass.json (merged; per-job p59_b*.json, p59_ctl_*.json; subspaces p59_U_*.npz), figure results/lgatr/p59_lgatr_das_mass.png. Explainer P59_EXPLAINER and six glossary entries added to site/explain.py.

## Conclusion

L-GATr is the Lorentz-equivariant tagger. Each of its tokens also carries 32 "auxiliary" scalar channels, which do not change under Lorentz transformations. One direction in these channels carries about 40 % of the network's response to a real change of the jet mass. Copying that direction from a jet of a different mass moves the W-versus-Z output most of the way a real mass change does. Random directions and the same search run on mismatched jet pairs do not do this. The direction sits at the jet's summary token from the fourth block on. The scalar parts of the multivectors carry little of the response, and no small set of directions reproduces the network's calls under a mass change, so the rest of the mass information is spread out. Giving particles mass at fixed momenta moves the output and this direction the opposite way to a real mass change, so the stored variable is not the invariant mass alone. Whether it is the angular spread is still open, because the spread edit used here also changed the particle masses.

(Mass-specific share β_cf: the part of the patched output change that follows the counterfactual mass change, fitted jointly with the part that follows the source jet's own output. 1 = the full response to the mass change, 0 = none. Measured values: 0.36–0.43 at blocks 3–9, against at most 0.04 for the shuffled-pair control and at most 0.10 for random directions.)

## Procedure

- Model: L-GATr final checkpoint model_run0.pt (run anrunw-ai-c-p5-lgatr2, authors' jctagging config, 12 blocks; residual stream per token = 16 multivector channels × 16 components + 32 auxiliary scalars). Untrained control: same config, torch seed 5959, no weights loaded.
- Unedited accuracy: P1 sample (rows 3,000–3,300 of each _100 test file, N 3,000) 83.73 / 83.77 / 83.73 / 83.77 % in the four main jobs (expected 83.73 %); 10-class accuracy on the 3,000 evaluation base jets (W, Z, QCD) 73.7 %.
- Sample: JetClass test_20M WToQQ / ZToQQ / ZJetsToNuNu, files _112 (fit) and _104 (evaluation). The planned _126 / _127 do not exist (test_20M holds _100–_119); _112 and _104 were used only by ParT nodes. 5,000 random rows per class per file (default_rng(5959) fit rows, default_rng(5960) evaluation rows, default_rng(5961) pairs and edits; never first-k). All jets rebuilt from per-particle (pT, Δη, Δφ, mass) so base and edited jets differ only by the edit.
- Pairs: base b and source s from the same split with |pT_s/pT_b − 1| ≤ 0.05, 0.05 ≤ |ln m_s − ln m_b| ≤ 0.3, m ≥ 10 GeV; one random eligible source per base; 1,000 bases per class → 3,000 fit and 3,000 evaluation pairs. Evaluation pairs: median |Δln m| 0.167, median |Δln pT| 0.024.
- Counterfactual target: b with every particle's (Δη, Δφ) about the stored axis multiplied by λ (pT and particle mass fixed), λ by bisection so that m(b_λ) = m_s (relative error ≤ 8e-4 on evaluation pairs; one fit pair hit the bracket, 2 %). λ range 0.56–1.45. The counterfactual flips the 10-class call on 36.6 % of evaluation pairs.
- Interchange: after block k (k = 0..11), h_b ← h_b + U Uᵀ (h_s − h_b), then blocks k+1..11 and the read-out run on the edited state. The read-out takes the scalar parts of the output channels at the global token (mean_aggregation is off), so the global token after block 11 is the read-out aggregation site. Tokens: aligned = global, the two reference tokens and particles matched by pT rank up to min(n_b, n_s) (83.9 % of base tokens, 82.5 % of particle tokens); or the global token only.
- Variants (U grade-preserving, i.e. Lorentz-equivariant): g0 = 16 scalar parts; g1 = 16 four-vectors, one U for t, x, y, z; aux = 32 auxiliary scalars; all = one d-dim U in each of grades 0–4 and aux (6d slots). d = 1, 2, 4 (global-token mode d = 1, 4).
- DAS fit: U = QR(A), A trained with Adam (lr 0.08, 40 steps of 100 fit pairs) to minimise KL(softmax(l_cf) ‖ softmax(l_patched)).
- Metrics on the 3,000 held-out pairs, 95 % bootstrap over pairs (1,000 draws): IIA = fraction with patched call = counterfactual call; IIA_ch = the same on the 1,097 pairs whose call the counterfactual flips; F = Σ(Δd_p Δd_cf)/ΣΔd_cf², d = l_W − l_Z; joint fit Δd_p = β_cf Δd_cf + β_src Δd_src (Δd_src = source jet's own d minus base d; corr(Δd_cf, Δd_src) = 0.39).
- Controls: (i) 100 random grade-preserving subspaces of the same d per setting on the first 200 evaluation pairs (30 for global-token mode); empirical p and a Gaussian-tail p from the null mean and s.d.; Holm over 144 (12 blocks × 4 variants × 3 d, aligned) on the Gaussian p. (ii) DAS on untrained L-GATr (d = 4). (iii) pT decoy: donors with 0.05 ≤ |Δln pT| ≤ 0.3 at mass within 5 %, target = b with pT × pT_s/pT_b and angles rescaled so m = m_b; DAS on it, cross-evaluated on the mass task and vice versa; subspace overlap ‖U_mᵀU_d‖²/d. (iv) shuffled pairs: DAS trained on the counterfactual output of a random other pair (all variants, d = 1, 4; follow-up jobs, same seeds reproduce the main fits).
- Edits (P56's plan; P56 had not run): 1,000 true W + 1,000 true Z evaluation jets. M: every particle mass m_i' = √(m_i² + (κ|p_i|)²) at fixed three-momentum, κ (median 0.033) so that m rises 4.9 % (own-axis spread change 0); DM: angles × λ (median 1.051), m +4.9 %, own-axis spread (r²) +9.9 %; R0: particle masses raised (κ0 median 0.048) so that m = 1.10 m_b; S: angles × 1.05 and particle masses lowered (κ1 median 0.034 < κ0 for 100 % of jets) so that m = m(R0) (Δln m 2e-7, spread +9.8 %); DS: R0 with angles × 1.05 at κ0 (m +4.0 %, spread +9.8 %). Read-out: the d = 1 coordinate at the global token (scalar for g0 / aux; Minkowski square for g1); ratio_M = Σ Δproj(M) / Σ Δproj(DM), ratio_S = Σ Δproj(S − R0) / Σ Δproj(DS − R0); network R from d = l_W − l_Z.

## Numbers

H0: no subspace beats random (IIA − IIA_random interval includes 0). H1: some d ≤ 4 subspace reaches IIA ≥ 0.8 by some block.

| setting (aligned tokens) | blocks | IIA | IIA_ch | F | β_cf (DAS) | β_cf (shuffled) | random F (2.5–97.5 %) |
|---|---|---|---|---|---|---|---|
| no patch | all | 0.635 | 0 | 0 | – | – | – |
| aux, d 1 | 3–9 | 0.720–0.744 | 0.32–0.36 | 0.42–0.48 | 0.36–0.43 | −0.13 to 0.04 | ≤ 0.10 |
| aux, d 1 | 0 / 10 / 11 | 0.690 / 0.677 / 0.675 | 0.21 / 0.19 / 0.18 | 0.22 / 0.33 / 0.29 | 0.21 / 0.17 / 0.19 | 0.02 / −0.03 / 0.02 | ≤ 0.07 |
| aux, d 4 | 3–9 | 0.737–0.745 | 0.37–0.42 | 0.47–0.51 | 0.41–0.45 | −0.07 to 0.07 | ≤ 0.20 |
| all grades, d 4 | 0–9 | 0.58–0.74 | 0.41–0.51 | 0.59–0.71 | 0.40–0.53 | 0.09–0.35 | 0.07–0.87 |
| g0, d 4 | 0–11 | 0.638–0.662 | 0.09–0.13 | 0.09–0.16 | 0.04–0.09 | −0.06 to 0.03 | ≤ 0.09 |
| g1, d 4 | 0–10 | 0.53–0.64 | 0.06–0.30 | 0.04–0.56 | 0.01–0.29 | 0.04–0.26 | up to 0.84 |
| whole state swapped | 0–11 | 0.36–0.46 | 0.26–0.40 | 0.66–0.92 | −0.07 to 0.13 (β_src 0.77–1.00) | – | – |

- Best IIA: 0.745 [0.730, 0.760] (aux, d 2, block 6); best IIA_ch: 0.511 [0.481, 0.542] (all, d 4, block 2). H1 (IIA ≥ 0.8) not reached in any of 144 settings, nor IIA_ch ≥ 0.8.
- Example (block 4, aux d 1): IIA 0.735 [0.719, 0.751], IIA_ch 0.349 [0.321, 0.378] vs random median 0.046 [0, 0.092], F 0.454 [0.427, 0.478] vs random [−0.017, 0.071], β_cf 0.408 [0.381, 0.434], β_src 0.052; shuffled β_cf −0.015 [−0.039, 0.009]. Empirical p 0.0099 (the minimum with 100 draws) at every block for aux and g0; Holm-adjusted Gaussian p for aux d 1 F ≤ 3e-26 at all 12 blocks; for g0 d 4 F 5e-6 to 0.024 at blocks 0–9. all-grades d 4: Holm p for F 1 / 0.085 / 0.014 at blocks 0–2 and ≤ 7e-4 from block 3; g1 d 4: Holm p ≥ 0.13 at every block, because random four-vector subspaces already move the output (they carry source momenta).
- Global token only (aux d 1): F 0.43–0.50 at blocks 3–9 (block 4: 0.447 [0.421, 0.469]) — the same as aligned-token patching.
- d dependence (aux): d 1 → d 4 raises β_cf by ≤ 0.07 at blocks 3–9.
- Untrained L-GATr: the counterfactual changes no call (IIA 1 trivially); F −0.24 to 0.30 (aux d 4: −0.19 to 0.12) with wide intervals.
- pT decoy (target changes 8.7 % of calls): aux decoy subspace F on its own task 0.02–0.11 and on the mass task 0.02–0.05; aux mass subspace on the decoy task 0.12–0.30; aux mass vs decoy subspace overlap 0.08–0.15 (random 4-dim subspaces in 32 dimensions: 0.125). For g1 / all the decoy subspaces transfer 0.12–0.51 of the mass response (unspecific transfer, as for the shuffled control).
- Edits, network output (N 2,000): Δd(DM) −0.49, Δd(M) +1.03, R_M = −2.11 [−2.44, −1.82]; Δd(S) −0.15, Δd(DS) −0.025, R_S = 6.1 [4.4, 9.0]; calls changed M 62.1 %, DM 15.5 %, S 37.9 %, DS 7.7 %.
- Edits, aux d 1 coordinate at the global token (global-token-trained; correlation with ln m 0.65–0.92 at blocks 0–9): ratio_M −2.9 to −5.2 (blocks 1–11; block 0 −0.66), ratio_S +2.3 to +4.8 (block 0 0.94); in s.d. of the coordinate (block 4): DM +0.28, M −1.46, S +0.47, DS +0.13. Grade 0 d 1: ratio_M −1.4 to −6.7, ratio_S unstable (DS changes ≤ 0.05 s.d.).
- ParT comparison (P8, P8v): one 1-dim jet-mean direction in ParT_full carried 0.58–0.69 of the φ-shift response at every depth from the embedding; L-GATr's best single direction carries F 0.42–0.48 (β_cf 0.36–0.43) of a real mass change, from block 3 (0.21 at block 0). The edits differ (ParT: φ shift / dilation; here: dilation to a donor's mass), so the comparison is approximate.

## Reading (scoped to what was measured)

- H0 is rejected for the auxiliary-scalar and grade-0 variants at every block (against random subspaces and against the shuffled-pair DAS). For grade 1 and all-grades, the trained subspace is not separable from the shuffled control at most blocks; their F mostly measures transfer of source-jet information (β_src 0.1–0.33), which the joint fit separates.
- H1 is not supported: no d ≤ 4 subspace in any grade reaches IIA 0.8; the largest mass-specific share is 0.53 (all grades, 24 slots, block 6) and 0.45 for the auxiliary scalars.
- Thesis "L-GATr stores the invariant mass in Lorentz-scalar channels from early blocks": the location part holds for the auxiliary scalars (not grade 0): one auxiliary direction carries 0.21 at block 0, 0.29 at block 1, 0.39–0.43 from block 3 to block 9, falling to 0.17–0.19 at blocks 10–11. The predicted grade-0 subspace (β_cf ≤ 0.09) is excluded. The prediction "projection ratio_M ≈ 1, ratio_S ≈ 0" is excluded (ratio_M −2.9 to −5.2).
- Alternative A (own-axis spread in the auxiliary scalars; ratio_M ≈ 0, ratio_S ≈ 1): the auxiliary variant is the best, but the ratios (M strongly negative, S +2.3 to +4.8) match neither reader. The network output gives the same pattern (R_M −2.1, R_S 6.1). Edit M gives particles masses of a few GeV at fixed momenta (median κ|p| ≈ 3 GeV for a 100 GeV particle), which L-GATr can read directly from each particle's p_i·p_i, so M changes a per-particle invariant that is absent from the training data. Edit S lowers the particle masses relative to R0 (κ 0.048 → 0.034), and a response linear in κ fitted to M (−1.46 s.d. per Δκ 0.033 at block 4) predicts about +0.6 s.d. for S from the particle masses alone (observed +0.47). So the S response can be explained by the particle-mass change without any spread effect (an inference, not tested separately). The edit pair does not separate the invariant mass from the spread for this network.
- Alternative B (no low-dimensional subspace, mass use distributed): partly. A single direction carries about 40 % of the mass-specific response, well above every control. The rest is not captured at d ≤ 4 per grade, and calls are reproduced on at most half of the flipped pairs.
- Grade 1 at block 11 cannot affect the output (F exactly 0): the read-out takes only scalar parts, as expected for an equivariant linear map.

## Issues / caveats

- Files: _126/_127 replaced by _112/_104 (see Procedure). The null has 100 random draws (not 200) per setting on 200 pairs (30 draws for global-token mode) to fit the GPU budget. Empirical p therefore cannot go below 0.0099, and Holm is applied to a Gaussian-tail p from the null (stated in the JSON as pz_*). Controls (ii), (iii) were run at d = 4 only; global-token mode at d = 1, 4 only.
- DAS used 40 optimisation steps; losses fell (e.g. aux block 4: KL 0.48 → 0.35), but larger budgets could raise the shares somewhat.
- The first shuffled control (main run) covered only g0 and all-grades; the follow-up jobs ran it for all variants with the joint β_cf/β_src fit. The table uses the follow-up numbers (same seeds; the DAS fits reproduce, e.g. aux block 3 F 0.415 in both).
- Edits M and S follow P56's plan, which had not run. They move per-particle masses far outside the training range, so they do not isolate the invariant mass from the spread for L-GATr. A spread edit at fixed particle masses and fixed invariant mass is needed (for example, at large angles where m² and M0·M2 about the momentum axis differ, as P56 also proposed).
- The fraction-of-tokens-patched is 84 % (aligned mode); the global-token result shows the variable is available at the jet token, so the alignment rule matters little for aux.

## Proposed follow-up

- P59v (skeptic): fresh files; 1,000 random draws for the auxiliary d = 1 setting at blocks 3–9; longer DAS fits; necessity test (hold the found direction at its unedited value on the counterfactual jet and measure the response that disappears); and the aux direction's response to a spread-only edit at fixed particle masses (large-angle or top jets, P56 design) to settle invariant mass vs spread.
