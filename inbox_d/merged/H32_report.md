# H32: W decay angle inside top at fixed m_top and m_W. Full run of H31 plus a side-effect-matched control (claim D6 / D11 / D13 extension)

**Status: done.** Script `interp/h32_top_dalitz_full.py` (imports the H31 edit, `interp/h31_top_dalitz.py`), follow-up regression `interp/h32_adjust.py`, figure `interp/h32_plot.py`.

Results in `/j-jepa-vol/aaron-interp/results/physics/`, copied to `results/physics/`:
- `h32_top_dalitz_full.json`: all statistics per sample × class × Δc × model.
- `h32_adjust.json`: residual side-effect regression (b_cos).
- `h32_perjet.npz`: per-jet scores, observables, acceptance flags, labelled-subjet fractions.
- `h32_dalitz.png`: figure. Smoke: `h32_smoke_top_dalitz_full.json`, `h32_smoke_adjust.json`, `h32_smoke_dalitz.png`.

Jobs (logs `/j-jepa-vol/aaron-interp/logs/<name>.log`; `logs/d-h32-run2.log` copied):
- `d-h32-smoke`: stopped by a self-test assertion (mirror candidates, see Procedure); fixed. `d-h32-smoke2`: 150 + 150 jets per sample, exit 0, 5 min.
- `d-h32-run`: full run; the pod was evicted after 20 min when its node (`k8s-haosu-05.sdsc.optiputer.net`) went NotReady (TaintManagerEviction). No code fault. Resubmitted unchanged except for a per-sample checkpoint of the per-jet file.
- `d-h32-run2`: full run, exit 0, 31 min, GPU.
- `d-h32-adjs` (smoke), `d-h32-post` (regression + figure), `d-h32-plots` / `d-h32-plot2` (figure): CPU, 1 CPU / 2 GiB.

## Conclusion
We rotated the W decay inside top jets so that the W decay angle |cos θ*| changed while the top mass and the W mass stayed fixed. Each tagger's response was compared with a control edit that copied the edit's change in how energy is shared between the prongs, at a fixed decay angle. Increasing |cos θ*| (one W daughter made softer and wider) lowered the top score of all nine taggers more than the control did. But a small classifier that sees neither the decay angle nor the pair masses dropped by a similar amount at the smaller step, because the control reproduced only about half of the edit's change in τ32 (the N-subjettiness ratio that measures three-prong-ness).

After a per-jet correction for this leftover τ32 and energy-sharing mismatch, PFN and EFN keep a dependence on the decay angle of about −1.4 to −1.8 log-odds per unit |cos θ*| on both samples, well outside what the angle-blind classifier leaks. ParT, ParticleNet and LorentzNet keep a smaller dependence, about a third of PFN's; for most of them the interval overlaps the leak on at least one of the two samples. Still uncertain: the correction is a linear regression, the control jets at the largest step are further from natural jets than the edited ones, and the JetClass full-input models respond strongly to the control edit itself.

## Hypotheses and outcome
- **H1 (angle used):** D − c-side nonzero, ordered in Δ|cos θ*|, with the sign of MLP5c's, and larger than the angle-blind references.
- **H0 (side effects only):** D − c-side inside the c-r null, and as large as for MLPx / MLP3, which lack |cos θ*| and the pair masses.
- **Measured:** D − c-side is outside the c-r null and ordered for every network (Spearman over Δc −0.8 to −1.0 on top jets). It is also nonzero for MLPx and MLP3 (MLPx −0.41 to −0.51 at +0.3 on Top Landscape), because c-side reproduces about half of D's Δτ32. So the raw D − c-side does not exclude H0 at Δc = +0.3. The networks exceed MLPx by 1.2–2.3× at +0.6 (except JetClass ParT_full / PN_full: 0.7–1.0×).
- **After the per-jet adjustment (b_cos):** the 95 % CI lies outside the MLPx leak range (−0.09 to −0.32 Top, −0.08 to −0.24 JetClass) for PFN and EFN on both samples, for JetClass PN_kin on Top Landscape jets and for Top LorentzNet on JetClass jets. Top ParT, JetClass ParT_kin, Top ParticleNet and JetClass ParT_full / PN_full overlap the MLPx leak range on every sample where they were run; Top LorentzNet overlaps it on Top Landscape jets (upper bound −0.28) and JetClass PN_kin on JetClass jets.
- **MLP5c as reference:** its b_cos changes sign across the 5 retrains (−0.16 to +0.65 on Top Landscape, −0.50 to +0.21 on JetClass). A small MLP given |cos θ*| and the pair masses together does not settle on one use of the angle, so "the sign of MLP5c's" is not a stable prediction. MLP5 (pair masses, no |cos θ*|) is stable: −0.17 to −0.49 and −0.22 to −0.85.

## Procedure
**Samples.** Random rows, never first-k; seed 3201, which is disjoint from the H31 smoke seed 3101.
- **Top Landscape:** test split (converted parquet), `h4_top_wmass._sample(2000, 3201)`: 2,000 top + 2,000 QCD. Jets are rebuilt through (pT, η, φ, m) and sorted by pT.
- **JetClass:** test `TTBar_100` (Tbqq) and `ZJetsToNuNu_100` (QCD), stored 550 < pT < 650 GeV, |η| < 2, ≥ 3 particles, 2,000 per class (`h19_wpair_cross.jc_read`, rng 3201 / 3202; η = jet_eta + sign(jet_eta)·deta).

**Dalitz variables (H31).** Three exclusive-kT subjets (E-scheme). The W pair (a, b) is the pair with mass closest to 80.4 GeV; c is the third subjet. m_bq,hi / m_bq,lo = max / min(m_ac, m_bc). |cos θ*| = |n·u|, with n the direction of a and u the direction of c, both in the W rest frame.

**Edit D(Δc), Δc ∈ {−0.6, −0.3, +0.3, +0.6} (H31, unchanged).** All constituents of a and b are boosted to the W rest frame, rotated rigidly about n₊ × u so that |cos θ*| → clip(c₀ + Δc, 0, 0.995), and boosted back. The W four-momentum, m_W, m_top, every intra-subjet invariant and N are unchanged.

**Controls** (same jets):
- **cφ:** rotation of the W system about u by the same angle |α|. |cos θ*| and m_ac, m_bc are unchanged.
- **cr:** rotation about a random axis by |α|. The c-r null is cr − cφ.
- **TN:** rigid (η, φ) translation by D's pT-weighted mean displacement.
- **c-side (new).** A per-jet Lorentz transformation chosen among 48 × 33 = 1,584 candidates:
  - W constituents get Λ_T(Q)·Λ_W(R); b-subjet constituents get Λ_T(Q).
  - Λ_W(R) = B(−β_W) R B(β_W). R is a rotation about u (24 azimuths), or the mirror θ* → π − θ* followed by a rotation about u (24 azimuths).
  - Λ_T(Q) = B(−β_T) Q B(β_T). Q is the identity or a rotation of the whole top decay in the top rest frame, tilting the W direction by ρ ∈ {0.05, 0.1, 0.2, 0.35} rad toward 8 azimuths.
  - Every candidate keeps the jet four-momentum, m_top, m_W, every intra-subjet invariant and |cos θ*| exactly. m_ac and m_bc are exact for non-mirrored candidates; the mirror swaps them only up to a term from the W-frame energy difference of the two W subjets (≈ 1 GeV on the synthetic test jet; measured mean |Δm_bq| of c-side ≤ 2.2 GeV, against 10–38 GeV for D).
  - **Step 1:** keep the 10 candidates closest to D in the three labelled-subjet pT fractions (W prongs sorted, b subjet), with all subjets within ΔR 0.75 of the jet axis.
  - **Step 2:** among them, pick the one minimising Σ(Δf − Δf_D)²/0.02² + (Δτ32 − Δτ32_D)²/0.01² (τ32 recomputed on each candidate jet, R = 0.8).
  - **Deviation from the plan:** "within-prong rescaling" was not used. Rescaling constituent momenta inside a prong changes the prong mass, and therefore m_W and m_top. The top-frame tilt Q supplies the extra freedom instead.
  - **Self-test** (synthetic three-prong jet, five candidates): worst change in m_top, m_W, subjet masses, jet four-momentum, |cos θ*|·100 (and pair masses for non-mirrored candidates) = 8·10⁻¹¹.
  - The first smoke run asserted pair-mass invariance for the mirror candidates too and stopped; the assertion was restricted to non-mirrored candidates.

**Acceptance (all variants):** after re-clustering, |Δm| < 0.5 GeV and |Δm_W| < 2 GeV. Analysis set per (Δc, class): jets accepted for D, cφ and c-side. Every edited jet is rebuilt about its own constituent-sum axis for every network (H25 path).

**Models.**
- Top Landscape ParT, ParticleNet, LorentzNet (released), PFN and EFN (H2); JetClass ParT_kin and PN_kin, all on both samples.
- JetClass ParT_full and PN_full on the JetClass sample only.
- JetClass score: L(Tbqq) − L(QCD).

**References.** MLP(64, 64), 25k + 25k disjoint jets per dataset (Top val rng 4194; JetClass `*_101` in-window rng 4193 / 4194), 5 retrains each (random_state 0–4: initialisation and early-stopping split).
- MLP3: (m, m_W, τ32).
- MLP5: MLP3 inputs + (m_bq,hi, m_bq,lo).
- MLP5c: MLP5 inputs + |cos θ*|.
- MLPx: the side effects without pair masses or angle (m, m_W, m_SD, τ1–3, τ21, τ32, N, sorted subjet pT fractions, sorted subjet masses).

**Metrics.**
- **Effect:** D − c-side and D − cφ in top log-odds, with a 95 % bootstrap CI over jets (B = 2000).
- **Tests:** two-sided sign-flip permutation p (5,000). Holm over 128 tests per contrast (networks × Δc × samples × classes); floor 128/5001 = 0.026.
- **Null:** c-r null = cr − cφ, with its CI.
- **Calls:** McNemar D vs c-side at ε_S = 0.5 and 0.3.
- **Ratios:** to each MLP retrain (ranges over 5).
- **Dose:** Spearman of the D − c-side means against Δc.
- **C2ST:** edited vs natural pool jets of the same class, matched in standardized (m, τ32, m_W, |cos θ*|); HGB on 20 substructure features, 2-fold AUC.

**Residual adjustment (`h32_adjust.py`).** Per (sample, model), on the top jets pooled over the four Δc cells (Top n = 2,231; JetClass n = 2,696):

  y_j = s_D − s_c-side = b₀ + b_cos·Δ|cos θ*| + b_τ·Δτ32 + b_hi·Δf_hi + b_lo·Δf_lo

Δ = D minus c-side, re-clustered observables per jet. b_cos is the score change per unit |cos θ*| (with the pair masses that move with it) at zero residual τ32 / fraction mismatch. Bootstrap CI over jets (B = 2000), two-sided bootstrap p, Holm over 16 network × sample tests.
- **Built-in control:** MLPx and MLP3 have no |cos θ*| or pair masses, so their b_cos measures leak from side effects that the regressors do not capture.
- Correlations among the regressors: Δ|cos θ*| vs Δτ32 r = 0.51; Δf_hi vs Δf_lo r = −0.75 (Top) / −0.67 (JetClass).

## Checks: unedited models on these samples (4,000 jets each)
| model | Top sample AUC / acc. (%) | JetClass sample AUC / acc. (%) | known (H25 / H19) |
|---|---|---|---|
| Top ParT | 0.9862 / 93.7 | 0.9849 / 94.0 | 0.9867 / 94.3 |
| Top ParticleNet | 0.9845 / 93.8 | 0.9836 / 94.0 | 0.9868 / 94.0 |
| Top LorentzNet | 0.9867 / 94.3 | 0.9860 / 94.2 | 0.9879 / 94.5 |
| Top PFN | 0.9782 / 92.7 | 0.9792 / 92.0 | 0.9816 / 93.0 |
| Top EFN | 0.9728 / 91.5 | 0.9773 / 92.7 | 0.9783 / 92.6 |
| JetClass ParT_kin | 0.9815 / 92.4 | 0.9920 / 95.8 | 0.9930 / 95.7 (JetClass) |
| JetClass PN_kin | 0.9821 / 92.3 | 0.9885 / 94.9 | 0.9897 / 94.9 |
| JetClass ParT_full | – | 0.9987 / 98.6 | 0.9987 / 98.4 (H19) |
| JetClass PN_full | – | 0.9979 / 98.2 | 0.9980 / 98.1 (H19) |
| MLP3 (5 retrains) | 0.9659–0.9666 / 90.2–90.6 | 0.9710–0.9724 / 90.6–91.1 | – |
| MLP5 (5 retrains) | 0.9689–0.9701 / 91.0–91.2 | 0.9740–0.9761 / 91.5–92.1 | – |
| MLP5c (5 retrains) | 0.9696–0.9709 / 90.8–91.4 | 0.9761–0.9767 / 92.0–92.4 | – |
| MLPx (5 retrains) | 0.9755–0.9763 / 92.0–92.5 | 0.9790–0.9797 / 92.6–93.0 | – |

- Every network is within 0.6 AUC points and 1.1 accuracy points of the earlier value on the matching dataset (JetClass models compared on the JetClass sample) (accuracy uncertainty on 4,000 jets ≈ ±0.4 points). The largest gap is Top EFN on the Top sample: 91.5 % vs 92.6 %.
- **TN translation control:** |Δ score| ≤ 0.08 log-odds for every network at Δc = +0.3 on both samples.
- **Acceptance:** of 4,000 jets per sample, D is accepted for 2,401 / 2,532 / 1,232 / 758 (Top, Δc −0.6 … +0.6) and 2,565 / 2,635 / 1,427 / 954 (JetClass). c-side is accepted for 1,984–2,132 (Top) and 2,312–2,396 (JetClass). At +Δc one W daughter becomes soft and wide, so re-clustering often moves particles between subjets.

## What the edits do (top jets in the analysis set; mean changes; labelled-subjet fractions f use the original kT-3 labels)

| sample, Δc | n | Δ\|cos θ*\| D / c-side / cφ | Δτ32 D / c-side / cφ | Δf_W,hi D / c-side | Δf_b D / c-side | Δm_bq,lo (GeV) D / c-side | Δm_SD (GeV) D / c-side | C2ST AUC id / D / cφ / c-side |
|---|---|---|---|---|---|---|---|---|
| Top, -0.6 | 809 | -0.255 / -0.000 / -0.001 | -0.0219 / -0.0115 / +0.0051 | -0.0081 / -0.0073 | +0.0000 / +0.0003 | +12.8 / -0.2 | -1.43 / -0.36 | 0.504 / 0.634 / 0.517 / 0.528 |
| Top, -0.3 | 831 | -0.192 / -0.000 / -0.001 | -0.0180 / -0.0103 / +0.0032 | -0.0065 / -0.0060 | +0.0000 / +0.0003 | +9.5 / -0.0 | -0.92 / -0.37 | 0.505 / 0.552 / 0.532 / 0.495 |
| Top, +0.3 | 409 | +0.271 / +0.010 / +0.001 | +0.0517 / +0.0254 / +0.0059 | +0.0146 / +0.0082 | +0.0001 / +0.0020 | -16.4 / -0.8 | -1.55 / -2.39 | 0.503 / 0.555 / 0.524 / 0.564 |
| Top, +0.6 | 182 | +0.539 / +0.008 / +0.009 | +0.1264 / +0.0529 / +0.0232 | +0.0483 / +0.0294 | +0.0002 / +0.0049 | -36.6 / -2.2 | -4.41 / -3.48 | 0.492 / 0.529 / 0.562 / 0.646 |
| JetClass, -0.6 | 938 | -0.285 / -0.000 / -0.001 | -0.0263 / -0.0134 / +0.0025 | -0.0107 / -0.0101 | -0.0000 / +0.0004 | +14.9 / +0.4 | -1.72 / -0.55 | 0.524 / 0.644 / 0.543 / 0.556 |
| JetClass, -0.3 | 948 | -0.201 / -0.002 / -0.001 | -0.0236 / -0.0114 / +0.0033 | -0.0070 / -0.0068 | -0.0000 / +0.0000 | +10.7 / +0.3 | -0.92 / -0.14 | 0.481 / 0.556 / 0.523 / 0.520 |
| JetClass, +0.3 | 534 | +0.285 / +0.001 / +0.002 | +0.0693 / +0.0365 / +0.0086 | +0.0226 / +0.0169 | +0.0001 / +0.0017 | -18.2 / -0.7 | -0.56 / -1.20 | 0.514 / 0.543 / 0.523 / 0.616 |
| JetClass, +0.6 | 276 | +0.544 / +0.001 / +0.001 | +0.1638 / +0.0683 / +0.0306 | +0.0630 / +0.0489 | +0.0003 / +0.0053 | -38.3 / -1.2 | -0.44 / -6.08 | 0.461 / 0.572 / 0.535 / 0.649 |

- c-side reproduces 56–97 % of D's change in the leading W-prong fraction (56 % at Top +0.3, 61 % at Top +0.6, 75–97 % elsewhere). It keeps |cos θ*| within 0.01 and the pair masses within 2.2 GeV.
- c-side reproduces 42–57 % of D's Δτ32 (Top +0.6: +0.053 vs +0.126; JetClass +0.6: +0.068 vs +0.164).
- c-side moves the b-subjet fraction by up to 0.005 (D: 0.000) and m_SD by up to −6.1 GeV (JetClass +0.6; D: −0.4 GeV).
- **C2ST:** at +0.6 the c-side jets are further from matched natural jets than D's (0.646 / 0.649 vs 0.529 / 0.572). At −0.6 the order is reversed (D 0.634 / 0.644, c-side 0.528 / 0.556).

## Numbers: D − c-side (top log-odds), top jets


**Top Landscape sample, top jets.** Cell: D − c-side [95 % CI]; Holm-adjusted p (floor 0.026); below: D − cφ.

| model | Δc −0.6 | −0.3 | +0.3 | +0.6 |
|---|---|---|---|---|
| Top ParT | -0.01 [-0.10, +0.08] p 1<br>cφ: +0.07 | +0.05 [-0.02, +0.13] p 1<br>cφ: +0.03 | -0.62 [-0.80, -0.45] p 0.026<br>cφ: -0.82 | -1.90 [-2.29, -1.54] p 0.026<br>cφ: -1.70 |
| Top ParticleNet | +0.19 [+0.04, +0.34] p 0.86<br>cφ: +0.18 | +0.42 [+0.28, +0.56] p 0.026<br>cφ: +0.03 | -0.43 [-0.68, -0.17] p 0.13<br>cφ: -0.98 | -2.15 [-2.81, -1.53] p 0.026<br>cφ: -1.60 |
| Top LorentzNet | +0.08 [-0.01, +0.18] p 1<br>cφ: +0.08 | +0.15 [+0.07, +0.23] p 0.03<br>cφ: +0.01 | -0.63 [-0.80, -0.47] p 0.026<br>cφ: -0.94 | -1.95 [-2.34, -1.58] p 0.026<br>cφ: -2.00 |
| Top PFN | +0.66 [+0.57, +0.76] p 0.026<br>cφ: +0.49 | +0.67 [+0.58, +0.76] p 0.026<br>cφ: +0.35 | -0.53 [-0.71, -0.36] p 0.026<br>cφ: -0.98 | -2.41 [-2.93, -1.95] p 0.026<br>cφ: -1.97 |
| Top EFN | +0.28 [+0.21, +0.36] p 0.026<br>cφ: +0.39 | +0.21 [+0.15, +0.28] p 0.026<br>cφ: +0.31 | -0.81 [-0.94, -0.69] p 0.026<br>cφ: -0.97 | -2.04 [-2.31, -1.77] p 0.026<br>cφ: -2.22 |
| JetClass ParT_kin | +0.05 [-0.02, +0.11] p 1<br>cφ: +0.13 | +0.10 [+0.03, +0.16] p 0.29<br>cφ: +0.07 | -0.53 [-0.67, -0.39] p 0.026<br>cφ: -0.79 | -1.58 [-1.88, -1.29] p 0.026<br>cφ: -1.49 |
| JetClass PN_kin | +0.05 [-0.02, +0.12] p 1<br>cφ: +0.21 | +0.07 [+0.01, +0.14] p 0.99<br>cφ: +0.13 | -0.69 [-0.83, -0.55] p 0.026<br>cφ: -0.83 | -1.68 [-1.97, -1.40] p 0.026<br>cφ: -1.50 |
| MLP3 (range, 5 retrains) | +0.06 to +0.07<br>cφ: +0.17 to +0.20 | +0.05 to +0.06<br>cφ: +0.14 to +0.16 | -0.23 to -0.20<br>cφ: -0.38 to -0.33 | -0.65 to -0.56<br>cφ: -0.88 to -0.73 |
| MLP5 (range, 5 retrains) | +0.07 to +0.15<br>cφ: +0.21 to +0.27 | +0.06 to +0.12<br>cφ: +0.17 to +0.22 | -0.46 to -0.38<br>cφ: -0.62 to -0.50 | -1.20 to -0.99<br>cφ: -1.41 to -1.18 |
| MLP5c (range, 5 retrains) | -0.31 to +0.00<br>cφ: -0.21 to +0.12 | -0.19 to +0.01<br>cφ: -0.12 to +0.11 | -0.41 to -0.21<br>cφ: -0.55 to -0.31 | -1.21 to -0.76<br>cφ: -1.44 to -0.89 |
| MLPx (range, 5 retrains) | +0.11 to +0.15<br>cφ: +0.29 to +0.32 | +0.08 to +0.11<br>cφ: +0.22 to +0.27 | -0.51 to -0.41<br>cφ: -0.71 to -0.57 | -1.31 to -1.07<br>cφ: -1.70 to -1.43 |

c-r null (cr − cφ), range over networks of the mean: Δc -0.6: -0.10 to +0.04; Δc -0.3: -0.06 to +0.01; Δc +0.3: -0.05 to +0.07; Δc +0.6: -0.37 to +0.38

Ratio to the angle-blind MLPx (D − c-side, range over 5 retrains) at +0.3 / +0.6: Top ParT 1.23–1.53 / 1.45–1.78; Top ParticleNet 0.84–1.04 / 1.64–2.01; Top LorentzNet 1.25–1.55 / 1.49–1.82; Top PFN 1.05–1.30 / 1.84–2.26; Top EFN 1.61–1.99 / 1.55–1.91; JetClass ParT_kin 1.04–1.29 / 1.20–1.48; JetClass PN_kin 1.36–1.68 / 1.28–1.57

McNemar, calls at ε_S = 0.5, D vs c-side (pass→fail / fail→pass, p) at +0.6: Top ParT 52/8, p 5e-09; Top ParticleNet 50/8, p 2e-08; Top LorentzNet 52/10, p 6e-08; Top PFN 71/9, p 4e-13; Top EFN 73/2, p 2e-19; JetClass ParT_kin 53/7, p 8e-10; JetClass PN_kin 51/5, p 1e-10

**JetClass sample, top jets.** Cell: D − c-side [95 % CI]; Holm-adjusted p (floor 0.026); below: D − cφ.

| model | Δc −0.6 | −0.3 | +0.3 | +0.6 |
|---|---|---|---|---|
| Top ParT | +0.02 [-0.07, +0.13] p 1<br>cφ: +0.14 | +0.14 [+0.04, +0.24] p 0.26<br>cφ: +0.24 | -0.98 [-1.16, -0.80] p 0.026<br>cφ: -1.15 | -2.05 [-2.37, -1.73] p 0.026<br>cφ: -1.94 |
| Top ParticleNet | +0.04 [-0.15, +0.23] p 1<br>cφ: -0.00 | +0.25 [+0.08, +0.41] p 0.15<br>cφ: +0.17 | -0.82 [-1.12, -0.50] p 0.026<br>cφ: -1.37 | -2.44 [-3.00, -1.95] p 0.026<br>cφ: -1.57 |
| Top LorentzNet | +0.07 [-0.06, +0.22] p 1<br>cφ: +0.12 | +0.23 [+0.10, +0.36] p 0.043<br>cφ: +0.29 | -1.25 [-1.47, -1.02] p 0.026<br>cφ: -1.58 | -2.73 [-3.09, -2.34] p 0.026<br>cφ: -2.66 |
| Top PFN | +0.55 [+0.47, +0.64] p 0.026<br>cφ: +0.49 | +0.52 [+0.44, +0.61] p 0.026<br>cφ: +0.42 | -0.97 [-1.14, -0.82] p 0.026<br>cφ: -1.25 | -2.55 [-2.87, -2.26] p 0.026<br>cφ: -2.23 |
| Top EFN | +0.36 [+0.29, +0.45] p 0.026<br>cφ: +0.49 | +0.37 [+0.30, +0.44] p 0.026<br>cφ: +0.47 | -1.05 [-1.18, -0.91] p 0.026<br>cφ: -1.24 | -2.47 [-2.69, -2.24] p 0.026<br>cφ: -2.65 |
| JetClass ParT_kin | +0.04 [-0.06, +0.14] p 1<br>cφ: -0.06 | +0.19 [+0.09, +0.28] p 0.026<br>cφ: +0.07 | -0.86 [-1.03, -0.68] p 0.026<br>cφ: -1.30 | -1.97 [-2.25, -1.67] p 0.026<br>cφ: -2.00 |
| JetClass PN_kin | +0.03 [-0.06, +0.13] p 1<br>cφ: +0.07 | +0.19 [+0.10, +0.27] p 0.026<br>cφ: +0.19 | -0.91 [-1.07, -0.74] p 0.026<br>cφ: -1.23 | -2.13 [-2.41, -1.86] p 0.026<br>cφ: -2.14 |
| JetClass ParT_full | +0.42 [+0.29, +0.58] p 0.026<br>cφ: -0.41 | +0.73 [+0.59, +0.88] p 0.026<br>cφ: -0.10 | -0.45 [-0.70, -0.20] p 0.08<br>cφ: -1.47 | -1.26 [-1.64, -0.88] p 0.026<br>cφ: -1.99 |
| JetClass PN_full | +0.61 [+0.48, +0.75] p 0.026<br>cφ: -0.06 | +0.81 [+0.68, +0.93] p 0.026<br>cφ: +0.13 | -0.23 [-0.45, -0.01] p 1<br>cφ: -1.19 | -1.15 [-1.47, -0.82] p 0.026<br>cφ: -1.77 |
| MLP3 (range, 5 retrains) | +0.09 to +0.10<br>cφ: +0.21 to +0.24 | +0.09 to +0.10<br>cφ: +0.20 to +0.23 | -0.30 to -0.25<br>cφ: -0.54 to -0.45 | -0.91 to -0.74<br>cφ: -1.21 to -1.01 |
| MLP5 (range, 5 retrains) | +0.08 to +0.27<br>cφ: +0.20 to +0.37 | +0.11 to +0.24<br>cφ: +0.24 to +0.34 | -0.62 to -0.53<br>cφ: -0.90 to -0.74 | -1.58 to -1.33<br>cφ: -1.97 to -1.66 |
| MLP5c (range, 5 retrains) | -0.11 to +0.09<br>cφ: -0.00 to +0.18 | +0.02 to +0.14<br>cφ: +0.14 to +0.24 | -0.62 to -0.50<br>cφ: -0.80 to -0.72 | -1.48 to -1.35<br>cφ: -1.75 to -1.62 |
| MLPx (range, 5 retrains) | +0.14 to +0.20<br>cφ: +0.33 to +0.44 | +0.14 to +0.20<br>cφ: +0.28 to +0.37 | -0.56 to -0.44<br>cφ: -0.88 to -0.69 | -1.55 to -1.23<br>cφ: -1.99 to -1.51 |

c-r null (cr − cφ), range over networks of the mean: Δc -0.6: -0.28 to -0.04; Δc -0.3: -0.04 to +0.06; Δc +0.3: -0.27 to -0.13; Δc +0.6: -0.44 to +0.73

Ratio to the angle-blind MLPx (D − c-side, range over 5 retrains) at +0.3 / +0.6: Top ParT 1.76–2.25 / 1.32–1.66; Top ParticleNet 1.48–1.89 / 1.58–1.99; Top LorentzNet 2.24–2.86 / 1.76–2.22; Top PFN 1.75–2.24 / 1.65–2.07; Top EFN 1.89–2.41 / 1.60–2.01; JetClass ParT_kin 1.54–1.97 / 1.27–1.60; JetClass PN_kin 1.63–2.08 / 1.38–1.73; JetClass ParT_full 0.81–1.03 / 0.81–1.03; JetClass PN_full 0.42–0.54 / 0.74–0.93

McNemar, calls at ε_S = 0.5, D vs c-side (pass→fail / fail→pass, p) at +0.6: Top ParT 79/12, p 3e-13; Top ParticleNet 75/12, p 3e-12; Top LorentzNet 85/9, p 1e-16; Top PFN 110/3, p 5e-29; Top EFN 134/4, p 9e-35; JetClass ParT_kin 69/10, p 6e-12; JetClass PN_kin 77/6, p 8e-17; JetClass ParT_full 50/24, p 0.003; JetClass PN_full 44/21, p 0.006

**QCD jets** (same edit applied to the pair closest to 80.4 GeV). On the Top sample the achieved Δ|cos θ*| is +0.21 to +0.23 at +Δc, because many QCD jets already sit near |cos θ*| = 1. Call changes at ε_S = 0.5 involve at most 2.9 % of QCD jets.
- Top Landscape sample: D − c-side at +0.3 is −0.17 to −0.37 for the seven networks run there; at −Δc it is −0.31 to +0.08.
- JetClass ParT_full and PN_full: c-side − cφ = +0.96 to +1.04 on QCD jets at −Δc, and −0.63 to −1.02 on top jets at all Δc. These two models respond to the c-side transformation itself much more than the kinematics-only models do (|c-side − cφ| ≤ 0.88 for the kin and Top models on top jets).

## Residual-adjusted angle dependence b_cos (log-odds per unit |cos θ*|), top jets pooled over Δc

| model | Top: b_cos [95 % CI] (Holm p) | Top: b_τ32 | JetClass: b_cos [95 % CI] (Holm p) | JetClass: b_τ32 |
|---|---|---|---|---|
| Top ParT | -0.43 [-0.71, -0.15] (0.042) | -11.6 | -0.23 [-0.48, +0.04] (0.32) | -15.9 |
| Top ParticleNet | -0.68 [-1.19, -0.16] (0.06) | -12.3 | +0.03 [-0.40, +0.48] (0.87) | -19.4 |
| Top LorentzNet | -0.59 [-0.89, -0.28] (0.008) | -11.6 | -0.67 [-1.01, -0.30] (0.008) | -18.1 |
| Top PFN | -1.82 [-2.18, -1.46] (0.008) | -11.8 | -1.63 [-1.88, -1.38] (0.008) | -13.3 |
| Top EFN | -1.39 [-1.59, -1.19] (0.008) | -11.1 | -1.60 [-1.78, -1.42] (0.008) | -12.9 |
| JetClass ParT_kin | -0.41 [-0.63, -0.17] (0.008) | -10.2 | -0.14 [-0.38, +0.12] (0.61) | -14.9 |
| JetClass PN_kin | -0.64 [-0.87, -0.42] (0.008) | -9.7 | -0.37 [-0.61, -0.12] (0.016) | -14.7 |
| JetClass ParT_full | – | – | -0.32 [-0.71, +0.06] (0.34) | -13.0 |
| JetClass PN_full | – | – | -0.46 [-0.78, -0.10] (0.054) | -12.2 |
| MLP3 | +0.04 to +0.06 (5 retrains) | -8.7 to -7.6 | +0.03 to +0.07 (5 retrains) | -9.5 to -7.8 |
| MLP5 | -0.49 to -0.17 (5 retrains) | -12.6 to -9.8 | -0.85 to -0.22 (5 retrains) | -13.0 to -9.5 |
| MLP5c | -0.16 to +0.65 (5 retrains) | -13.1 to -10.7 | -0.50 to +0.21 (5 retrains) | -13.1 to -10.9 |
| MLPx | -0.32 to -0.09 (5 retrains) | -13.5 to -11.2 | -0.24 to -0.08 (5 retrains) | -13.5 to -11.6 |

- **Decomposition** at Δc = +0.6 (b_cos·mean Δ|cos θ*| vs the summed τ32 + fraction terms), Top / JetClass:
  - PFN: −0.97 / −0.89 vs −0.86 / −1.26.
  - EFN: −0.74 / −0.87 vs −0.82 / −1.23.
  - LorentzNet: −0.31 / −0.36 vs −0.84 / −1.75.
  - ParT_top: −0.23 / −0.13 vs −0.84 / −1.52.
  - MLPx: −0.13 / −0.08 vs −0.99 / −1.22.
- b_τ32 is −10 to −19 log-odds per unit τ32 for every network, against −8 to −14 for the MLPs.

## Reading
- **Raw contrast.** On top jets the decay-angle edit D lowers every network's top score more than the matched control c-side when |cos θ*| is raised: −0.2 to −1.3 log-odds at +0.3 and −1.2 to −2.7 at +0.6 (Holm p ≤ 0.026 in 29 of 32 network cells at +Δc). Both are outside the c-r null.
- **Angle-blind references.** The raw contrast is also large for the MLPs without |cos θ*| or pair masses: MLPx −0.41 to −0.56 at +0.3 and −1.07 to −1.55 at +0.6. c-side copies the W-prong pT fractions but only 42–57 % of D's Δτ32.
  - At +0.3 the networks lie at 0.8–2.9× MLPx (JetClass ParT_full / PN_full: 0.4–1.0×).
  - At +0.6 they lie at 1.2–2.3× MLPx, except JetClass ParT_full / PN_full at 0.7–1.0×.
  - The excess over MLPx is present but not large for most networks.
- **At −Δc** (|cos θ*| lowered, more symmetric W decay), the networks' D − c-side are small (−0.01 to +0.81). The MLPs give +0.05 to +0.27, except MLP5c (−0.31 to +0.14). The response is asymmetric: lowering |cos θ*| from c₀ ≈ 0.28–0.32 changes the score less than raising it by the same step.
- **Adjusted contrast.** After the per-jet adjustment for the residual τ32 and fraction mismatch:
  - PFN and EFN keep b_cos ≈ −1.4 to −1.8 per unit |cos θ*| on both samples, 4–20× the MLPx leak (−0.08 to −0.32), with CIs that exclude it. For them, 40–53 % of the adjusted +0.6 response is attributed to the angle term.
  - LorentzNet (JetClass sample) and JetClass PN_kin (Top sample) have CIs just outside the leak range.
  - ParT_top, ParT_kin, Top ParticleNet and the JetClass full-input models have b_cos between +0.03 and −0.68, with CIs overlapping the leak range on at least one sample. For these models the measured response is mostly accounted for by the τ32 term (b_τ32 ≈ −10 to −19 per unit τ32).
- **Full-input models.** JetClass ParT_full and PN_full respond to c-side itself (c-side − cφ ≈ −0.6 to −1.0 on top jets, ≈ +1.0 on QCD jets). For them c-side is not a neutral control. A plausible source is the track-displacement and particle-ID inputs, which the Lorentz transformation leaves unchanged while the particle directions move; this was not tested here.
- **Off-manifold distance.** At +0.6 the c-side jets have C2ST AUC 0.65 against matched natural jets, against 0.53–0.57 for D. Part of the D − c-side contrast at +0.6 may therefore come from c-side being atypical rather than from D.

## Issues and limits
- **Partial matching.** c-side matches the prong pT fractions closely but reproduces only about half of D's Δτ32. The angle-specific statement rests on the linear adjustment, which assumes additive, linear dependence on (Δ|cos θ*|, Δτ32, Δf_hi, Δf_lo) within the edit range. The MLPx leak (b_cos −0.08 to −0.32) shows that the regressors do not absorb all side effects.
- **Search space.** A larger candidate set (finer Q tilts, or edits inside a prong at fixed prong mass) could match τ32 better. Matching τ32 exactly at fixed pair masses and |cos θ*| may need non-rigid changes inside the prongs.
- **MLP5c reference.** Its angle dependence is not stable across retrains (b_cos changes sign). MLP5 and MLPx are the stable references here.
- **Acceptance bias.** At +0.6 only 182 (Top) / 276 (JetClass) top jets are in the analysis set; they have lower c₀ (0.18 / 0.21) than at −0.6 (0.28 / 0.32).
- **Not done.** No explicit edit inside the prongs; no LorentzNet invariance control (not applicable, since every edit here is a physical change). The `site/explain.py` entry is given below as text, because writes are limited to `interp/`, `results/`, `logs/`, `inbox_d/` and `running/D.md`; the ledger agent can paste it.

## Explainer entry (for `site/explain.py`, Paper D list, `H32_EXPLAINER`)
**Question.** When a tagger sees a top-quark jet, does it use the angle at which the W boson decays (in the W rest frame), or only the energy sharing between prongs that comes with that angle?

**Physics background.**
- In t → bW, W → qq̄′, the W polarisation fixes the decay-angle distribution: 1/N dN/d cos θ* = (3/4) F₀ (1 − cos²θ*) + (3/8) F_L (1 − cos θ*)² + (3/8) F_R (1 + cos θ*)², with F₀ ≈ 0.7.
- At fixed m_t and m_W, cos θ* is equivalent to the split of the two b–quark pair masses: m²_bq₁ − m²_bq₂ ∝ cos θ*.
- In the lab, a larger |cos θ*| makes one W daughter softer and wider, which also raises τ32 and moves energy between prongs.

**Method.**
- **Edit D:** rotate the W decay in its rest frame so that |cos θ*| → |cos θ*| + Δc, with Λ_D = B(−β_W) R_k(α) B(β_W) applied to the W constituents; m_t, m_W and every intra-subjet invariant are unchanged.
- **Control c-side:** a Lorentz transformation Λ = Λ_T(Q)·Λ_W(R), with R a rotation about the b direction (or the mirror θ* → π − θ*) and Q a tilt of the whole decay in the top rest frame. It keeps |cos θ*| and every pair mass fixed and is chosen per jet to copy D's change in the subjet pT fractions and τ32.
- **Contrast:** Δs = s(D) − s(c-side), in top log-odds, compared with MLPs that do or do not see |cos θ*|.
- **Adjustment:** Δs = b₀ + b_cos Δ|cos θ*| + b_τ Δτ32 + b_hi Δf_hi + b_lo Δf_lo, fitted per jet.

**Result (figure `h32_dalitz.png`).**
- Raising |cos θ*| by 0.6 lowers every tagger's top log-odds by 1.2–2.7 more than c-side does.
- An MLP without the angle drops by 1.1–1.6, because c-side reproduces only about half of the τ32 change.
- After the adjustment, PFN and EFN keep b_cos ≈ −1.4 to −1.8, while ParT, ParticleNet and LorentzNet keep +0.03 to −0.68, which is close to the angle-blind leak of −0.1 to −0.3.

**Checks.**
- Self-test: exact invariants (to 10⁻¹⁰ GeV).
- Unedited accuracies reproduce the earlier values.
- Translation control |Δs| ≤ 0.08.
- c-r null (rotation about a random axis): −0.44 to +0.73.
- 5 MLP retrains.
- C2ST of edited vs natural jets 0.50–0.65.

**Physical significance.** Top taggers follow the W decay angle inside the top mainly through its τ32 / energy-sharing consequences. The Deep Sets models PFN and EFN show an additional angle-specific dependence; for the transformer and graph taggers the angle-specific part is small and not separable from the side effects at this statistics.
