# E42: does JEDI-linear's top call use the W-pair mass inside the jet? (loop A, round 20, exploratory; candidate claim A20t; relates to A3, A5n, A6)

**Conclusion.** We moved the two subjets that form the W candidate inside real top jets closer together or further apart, keeping the total jet mass, the three-prong shape (tau32) and the jet axis fixed. JEDI-linear's top score fell slightly more under this edit than under a control edit that changes a different pair of subjets. The extra drop was about 0.2–0.4 in log-odds. A classifier that reads the W-pair mass directly, scaled to this network's sensitivity to the total mass, would drop by about 3.5–4. So the network uses the internal W-pair mass only weakly, and almost no top call changed. Two things are less certain: on the held-out test sample alone the 60 GeV effect does not pass the Holm correction, and two of the seven related networks respond in the opposite direction.

## Question
Does the top call of JEDI-linear (official 3-feature n64, hls4ml 5-class, class t) use the mass of the W-candidate subjet pair (mW) of a three-prong jet beyond the total jet mass? If it does, which second-round objects S and which z neurons carry that use?

## Procedure
- **Code.** Script `interp/e42_top_wpair_jedi.py`. Cluster job `anrunw-ai-e42-run` at 1 CPU / 2 GiB, wall time 1,303 s. The edit part ran in the conda env (numba); the network part ran in the jax env. Smoke runs: e42-smoke, -smoke2, -smoke4.
- **Samples.**
  - `jedi_core.sample(2000, 4201)` is the **fit** sample. It is used for the unit scan, the naming fits and the REFfit reference.
  - `sample(2000, 4202)` is the **test** sample, used for all headline numbers. Each sample has 10,000 jets.
  - **Top arm:** t jets with soft-drop mass 105 < mSD < 210 GeV and 65 < mW < 95 GeV. There are 1,172 such jets in the fit sample and 1,153 in the test sample.
  - **QCD arm (Q80):** q/g jets in the same mSD window with mW in 50–70 or 90–115 GeV (249 / 283 jets).
- **Subjets.** We split each jet into three exclusive-kT subjets of its 64 constituents (taken as massless), using the E-scheme and rapidity–azimuth distances. This is the loop-D H4/H29 code (`h4_top_wmass`, `h4_verify.edit_set`). The W pair is the subjet pair whose mass is closest to 80.4 GeV; c is the third subjet.
- **Edits.** Every edit moves whole subjets rigidly in (η, φ). The pT of every particle and the subjet masses do not change.
  - **W⟨t⟩:** the opening of the W pair is rescaled about its pT-weighted centre until mW = t (t = 60 or 100 GeV). Subjet c is then moved along the line from the W centre so that the jet mass m is restored. tau32 (the 3-prong vs 2-prong N-subjettiness ratio) is restored with a common subjet-size factor.
  - **N⟨t⟩ (control):** the non-W pair (the harder W daughter plus c) gets the same relative mass change; the W pair is not moved. This control is the right comparison because it changes the internal angles by a similar amount and keeps m and tau32, but leaves mW alone. W − N therefore isolates the part of the response that is specific to the W pair.
  - **R⟨t⟩ (control):** the whole jet is rotated about its axis by the angle whose pT-weighted mean particle displacement equals that of the W edit.
  - **Q80:** the same W, N and R edits on q/g jets, setting mW to 80.4 GeV.
  - **DIL (real-mass reference):** every particle's (Δη, Δφ) is scaled by λ = 0.85, 0.92, 0.96, 1.04, 1.08 or 1.15. This changes the total mass and all pair masses by the factor λ while keeping tau32 (median |Δtau32| ≈ 1e-8).
- **Acceptance (after reclustering).**
  - W edit: |Δm| < 0.5 GeV, |Δtau32| < 0.02 and |mW − t| < 5 GeV.
  - N edit: |Δm| < 0.5 GeV, |Δtau32| < 0.02 and |ΔmW| < 2 GeV.
  - Analysis set = jets where both W and N pass. Test-sample sizes: T60 256, T100 145, Q80 23. Fit-sample sizes: 232, 152, 24.
- **Network input.** Every variant, including the unedited one, is rebuilt about its own E-scheme axis (the H23/H29 "rc" build). This way no edit shifts the axis the network sees.
- **Score.** Top log-odds s = L_t − logsumexp(L_g, L_q, L_W, L_Z). The top call means argmax = t. The per-jet response is Δ_v = s_v − s_unedited.
- **Reference ("mW reader").** An MLP(m, tau32, mW) classifier (sklearn, 32-32), trained on t vs q/g using 25k + 25k hls4ml train-archive jets (dev split excluded, rng 4205). Its AUC on the test sample is 0.911. A second copy, REFfit, is trained on the fit sample's 6,000 natural jets (AUC 0.902).
- **Calibration.**
  - k_DIL is the network's response to a real total-mass change, per unit of the reference's response to that change. It is measured at linear order from the antisymmetric DIL pairs: k = Σ[Δnet(λ) − Δnet(1/λ)] / Σ[Δref(λ) − Δref(1/λ)].
  - R_WN = mean(Δnet_W − Δnet_N) / (k_DIL · mean(Δref_W − Δref_N)).
  - In words, R_WN is the network's W-pair-specific response divided by what a classifier reading mW would show once scaled to this network's own mass sensitivity. R_WN = 1 means the network reads mW like the reference; R_WN = 0 means it does not read it at all.
  - The calibration was changed after the smoke run. The network's DIL response is even in λ: both shrinking and stretching the jet lower its top score. A through-origin slope would mix that even part into k, so it is kept only as `k_fit_through_origin`.
- **Statistics.**
  - 95 % intervals: 2,000-draw jet bootstrap.
  - H0: W − N = 0. Tested with a two-sided paired sign-flip permutation (10,000 draws).
  - McNemar (exact binomial on the discordant jets) compares the top calls after the W edit and after the N edit.
  - Holm correction over the 18 main tests: (W − N, W − R, McNemar) × 3 sets × 2 samples.
- **C2ST (classifier two-sample test).** A gradient-boosted classifier tries to tell edited test jets from natural fit-sample jets of the same class and window, with 2-fold cross-validation. An AUC of 0.5 means the two are indistinguishable. Feature set A (axis-free) is m, mSD, tau1, tau21, tau32 and the particle count. Feature set B adds mW, the smallest pair mass and the three-subjet mass.
- **Routing (holds).**
  - **What a hold is:** for each live S unit, we keep the unit at its unedited value while the network runs on the W60 and N60 jets (run_jetwise override).
  - **Share removed** = 1 − Σ Δ_hold / Σ Δ. It is computed for the W60 response Δ = s_W − s_id and for W − N.
  - **Null:** 500 random one-direction holds and 500 random three-direction holds in the standardised live-S space. For each, the edited S is projected back to the unedited S along random orthonormal directions. p = (1 + #null ≥ observed)/501, Holm over the 19 live S units.
  - **Scan and confirmation:** the scan is done on the fit sample's T60 set. The top 8 units are confirmed on the test sample's T60 and T100 sets. The same procedure is applied to the 16 z units.
- **Naming.** The top 3 S units are fitted with the E33 rank-window hinge library: orthogonal matching pursuit, at most 8 terms, fitted on the fit sample, R² measured on the test sample. They are also probed with one-particle radial edits: one particle's (Δη, Δφ) is multiplied by 1.1 on 1,153 test tops, and the mean dS/dr is recorded per pT-rank window and per r bin.
- **z-window read.** We measure Δz under W and N for z0 (mass window near 75 GeV), z7 and z14 (94–98 GeV) and z10 (135–148 GeV), with Holm over 4 units × 3 sets.
- **Controls.**
  - 10 untrained redraws: `Jedi(untrained=True, seed 0..9)`.
  - 7 related networks from the E39 EDIT_NETS set, run through `e33.Model`: official n32, perm-inv n64, HGQ-r150 s0–s2 and float s0–s1.

## Checks
- Identity rebuild vs stored inputs: max |Δ| = 6.1e-5.
- Unedited 5-class accuracy, stored inputs: 81.67 % on the fit sample and 80.49 % on the test sample. The recentred inputs give 81.64 % and 80.49 %.
  - The test value is 0.1 point below the expected 80.6–81.3 % band. The sampling standard error at N = 10,000 is about 0.4 point, and H29 found the same 80.49 % on another 10,000-jet draw.
- An identity hold (every live unit set to its own value) changes the score by exactly 0.
- Untrained redraws: accuracy 20.0 %, |W − N| ≤ 0.021 in every set.
- Edit medians on the T60 test set:
  - mW: 79.7 (unedited), 60.0 (W), 79.6 (N), 79.7 (R) GeV.
  - Jet mass: 169.9 GeV for every variant.
- One example jet: W-pair mass 79.7 → 60.0 GeV with jet mass 174.94 → 174.94 GeV. Under N, the W-pair mass stays at 80.1 GeV.

## Results (test sample 4202 unless noted)

| set | Δ_W | Δ_N | Δ_R | W − N [95 % CI] | perm p (Holm) | mW reader W − N | R_WN [CI] | top call rate unedited/W/N, McNemar p |
|---|---|---|---|---|---|---|---|---|
| T60 (n 256) | −0.667 | −0.489 | +0.022 | **−0.178 [−0.320, −0.026]** | 0.019 (0.21) | −1.55 | 0.047 [0.007, 0.083] | 0.980/0.973/0.973, 1.0 |
| T100 (n 145) | −0.261 | +0.032 | +0.038 | **−0.293 [−0.505, −0.071]** | 0.0073 (0.088) | −1.43 | 0.084 [0.021, 0.143] | 0.966/0.972/0.966, 1.0 |
| Q80 (n 23) | −0.009 | +0.075 | −0.035 | −0.084 [−0.343, +0.154] | 0.54 (1) | +0.53 | −0.06 [−0.30, 0.11] | 0.43/0.48/0.48, 1.0 |
| fit T60 (n 232) | −0.644 | −0.258 | +0.070 | −0.386 [−0.539, −0.230] | 1e-4 (0.0018) | −1.48 | 0.094 [0.057, 0.132] | 0.974/0.948/0.961, 0.51 |
| fit T100 (n 152) | −0.229 | +0.096 | +0.126 | −0.325 [−0.545, −0.113] | 0.0068 (0.088) | −1.41 | 0.083 [0.029, 0.141] | 0.993/0.987/0.987, 1.0 |

- **Calibration values.**
  - k_DIL = 2.46 [2.03, 3.15] on the test sample and 2.77 [2.24, 3.61] on the fit sample.
  - Network vs reference slope per unit ln m (linear order): 2.96 vs 1.21 (test).
  - The network's DIL responses are −2.04 at λ = 0.85 and −1.05 at λ = 1.15. The reference's are −1.37 and −1.00.
  - So the network is about 2.5 times as sensitive to a real total-mass change as the mW reader. An mW reader scaled to this sensitivity would give W − N = −3.8 at T60 and −3.5 at T100.
- **Raw ratio without calibration.** Network W − N divided by reference W − N is 0.11 (T60) and 0.21 (T100). Against REFfit (W − N −0.41 and −0.56) the ratios are 0.44 and 0.52.
- **Rotation control.** Δ_R is within ±0.13 of 0 in every set. W − R is −0.69 (T60) and −0.30 (T100), Holm ≤ 0.003.
- **C2ST (AUC with feature set A / with feature set B).**
  - Unedited analysis jets: 0.63 / 0.70. The AUC is above 0.5 because the acceptance cuts select a subset of the natural jets.
  - N: 0.61 / 0.66. R: 0.64 / 0.70.
  - W60: 0.79 / 1.00 (mW = 60 GeV does not occur in natural tops). W100: 0.67 / 1.00.
  - DIL λ = 0.96 / 1.04: 0.69 / 0.66 (set A).
- **z-window read (W − N, T60).**
  - z0 (~75 GeV window): −0.187 [−0.283, −0.094], Holm 0.0012.
  - z14 (94–98 GeV): +0.107 [0.013, 0.194], Holm 0.24.
  - z7 (94–98 GeV): −0.12 [−0.39, 0.16].
  - z10 (135–148 GeV): −0.029 [−0.122, 0.055].
  - At T100: z0 −0.173 (Holm 0.24), z7 +0.35 (Holm 0.37), z10 −0.036.
  - Δz10 is about −0.19 under both W60 and N60, so it carries the common part of the response, not the W-specific part.
- **S holds (W60 response, fit sample → test sample).**
  - Holding S58 removes 0.84 [0.74, 0.95] → 0.83 [0.74, 0.92] of the response. p = 0.002 vs the one-direction null, Holm 0.038 (fit) and 0.016 (test).
  - S60: 0.18 → 0.29. S13: 0.07 → 0.03. Every other unit: ≤ 0.07.
  - Holding the top 3 (S58, S60, S13) together removes 1.02 → 1.06, p(three-direction null) = 0.002. At T100 (test) it removes 0.88 [0.54, 1.42].
  - The share of W − N removed by holding S58 is 1.9 (fit) and 4.0 (test), i.e. more than all of it. S58 carries the response to both W and N edits; it does not separate W from N.
  - Caveat on the null: random-direction holds have a negative mean share (−0.94). Projecting S along random directions makes the response larger, so the null is lenient and the p-values are optimistic.
- **z holds.**
  - z4 removes 0.42 (fit) → 0.47 (test); z10 removes 0.34 → 0.27. Neither passes Holm against the one-direction null (fit p 0.052 / 0.078).
  - Holding the top 3 (z4, z10, z12) together removes 0.87 → 0.84, p(three-direction null) = 0.014 / 0.032.
- **Naming.**
  - **S60:** test R² 0.988. It is a core hinge: (0.08 − r)₊ over the pT ranks 0–13 plus (pT − 2)₊ over the ranks 0–19. One-particle edit: dS60/dr = −10.1 for ranks 0–3 at r < 0.1, and 0 for r > 0.2.
  - **S13:** test R² 0.95. It is a hard-particle radial sum: ΣpT r² and Σr over the 9 hardest particles, plus (r − 0.1)₊ and (r − 0.25)₊ terms. dS13/dr = +7.3 for ranks 0–3 at 0.1 < r < 0.2, and +1.4 for ranks 4–13. This is a subjet-scale radial hinge.
  - **S58:** not named (test R² 0.23; natural spread 3.7). The one-particle edits show mixed signs, so it is not a single (pT, r) hinge. It is one of the context-sensitive units from the E24x list (S0/S30/S31/S58).
- **Untrained redraws (10).** |W − N| ≤ 0.021 in every set.
- **Related networks (test sample, W − N at T60 / T100).**
  - official n32 (77.5 %): −0.37 / −0.26.
  - perm-inv n64 (81.4 %): −0.34 / −0.38.
  - float s0 (82.0 %): −0.35 / −0.64.
  - float s1 (81.8 %): −0.37 / −0.71.
  - hgq150 s0 (76.3 %): **+0.20 / +0.11**.
  - hgq150 s2 (77.5 %): **+0.11 / +0.08**.
  - hgq150 s1 (56 %, degenerate): +0.04 / +0.01.
  - R_WN for the four networks with negative W − N is 0.08–0.19.

## Reading (scoped to these measurements)
- **Hypothesis H_M2: the network reads only a total-mass-like M2 variable plus radial hinges.** It predicts W − N ≈ 0.
  - Measured W − N is −0.18 and −0.29 (test) and −0.39 and −0.33 (fit). These are nonzero on both samples, with the same sign at both targets, and far outside the untrained range (≤ 0.021).
  - Holm over 18 tests: the fit-sample T60 effect is 0.0018, test T60 is 0.21, and both T100 effects are 0.088.
  - The size is 5–9 % of an mW reader calibrated to this network's mass sensitivity (R_WN 0.05–0.09). H_M2 is therefore excluded only as an exact statement.
- **Hypothesis H_prong: the network uses internal prong structure.** It predicts W − N < 0 for t both below and above 80 GeV. This sign is observed at 60 and 100 GeV, but the response is small, and no top call changes measurably (McNemar p ≥ 0.5).
- **z10.** Its W − N is ≈ 0, so the W-specific part does not pass through the 135–148 GeV window unit. The z0 (~75 GeV) unit responds to W − N on both targets (−0.19 / −0.17).
- **Route.** The whole W60 response, the shared part included, is carried by S58 (about 83 %) with S60 and S13. Holding S58 removes the W and N responses together, so this run does not isolate one carrier of the W-pair-specific part. S13 is a hinge at the subjet scale (r 0.1–0.25, hardest 9 particles), as H_prong predicted, but it carries only 3–7 %.
- **Related networks.** The sign recurs in 4 of 7 networks, at a similar size. It is reversed in the two non-degenerate HGQ-r150 retrains. The W-pair response is therefore not a fixed property of the architecture and training recipe.
- **Effect on the thesis:** refines. The "one mass variable plus radial hinges" description holds for the top class at the level of calls and of about 90 % of the log-odds response. A small, sign-consistent W-pair term remains in the official network.

## Limitations
- The calibration reference is a single MLP fit; H29v found ±50 % spread across reference retrains.
- k_DIL comes from a response that is even in λ, so only the antisymmetric (linear-order) part is used.
- The random-direction hold null is lenient: its mean share is negative.
- The Q80 arm has only 23–24 jets.
- The W edit is off-manifold in feature set B (C2ST AUC 1.0). This is inherent to setting mW to 60 or 100 GeV.
- The test-sample accuracy is 80.49 %, 0.1 point below the stated band.

## Files
- `interp/e42_top_wpair_jedi.py`
- `results/jedi/e42.json`, `results/jedi/e42_edit.json`, `results/jedi/e42_inputs.npz`
- `results/jedi/e42_wpair.png`, redrawn locally from the json/npz:
  - (a) an edited example jet;
  - (b) W − N for the official network, the related networks and the untrained redraws;
  - (c) S-hold shares on the fit and test samples.
- `logs/e42-run.log`

## Explainer entry (for site/explain.py; not edited here)
- **Question.** Does JEDI-linear's top score depend on the W-pair mass inside a top jet once the total mass is fixed?
- **Background.** A top quark decays to b W, giving three prongs, and two of them form a W with mass $m_W \approx 80$ GeV. Earlier work found that JEDI-linear's mass read-out is a single $M_2$-like variable, $\sum p_T r^2$.
- **Method.** Edit $m_{W}\to t$ at fixed $m$, $\tau_{32}$ and axis by rigid subjet moves. Compare with a control that moves the non-W pair, and report $\Delta s_W-\Delta s_N$.
  - Calibration: $R_{WN}=\overline{\Delta s_W-\Delta s_N}\,/\,(k_{DIL}\,\overline{\Delta f_W-\Delta f_N})$, where $f$ is the MLP$(m,\tau_{32},m_W)$ log-odds and $k_{DIL}=\sum[\Delta s(\lambda)-\Delta s(1/\lambda)]/\sum[\Delta f(\lambda)-\Delta f(1/\lambda)]$ over real dilations.
- **Result.** $R_{WN}\approx0.05$–$0.09$ (figure e42_wpair.png).
- **Checks.** Untrained redraws, the rotation control, C2ST and the related networks.
- **Why it matters.** It says whether a compact tagger reads prong-level structure or only a global mass.

**Glossary additions:**
- **W − N:** the change of top log-odds under the W-pair edit minus that under the matched non-W-pair edit.
- **R_WN:** as defined above.
- **DIL:** a uniform angular dilation of the jet; a real total-mass change at fixed tau32.
- **share removed:** the fraction of an edit's score change that disappears when one unit is held at its unedited value.
