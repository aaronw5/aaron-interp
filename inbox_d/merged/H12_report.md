# H12 — Mass-preserving boost along the jet axis vs pT-only and angle-only scalings, 20 models on 4 datasets (claims D1, D8, D4m; new sub-claim D9)

## Conclusion
We lowered or raised each jet's transverse momentum (pT) by an exact Lorentz boost along the jet direction. This keeps the invariant jet mass exactly fixed and changes the opening angles between particles. Every model trained on the hls4ml sample changed a quarter to over a third of its calls under this boost, as much as under scalings that do change the mass (JEDI-linear n64: 32.5 % at pT × 0.9, against 2.6 % for a rotation of equal size). So no hls4ml-trained model reads a boost-invariant mass: their calls follow pT and opening angles at the fixed ~1 TeV pT of their training jets. The JetClass, Top Landscape and quark–gluon taggers changed 3–4 times fewer calls under the boost than under either scaling (ParT_full: 4.5 % vs 6.9 % and 14.4 %), but still more than the rotation control. The pT-normalised JEDI-style retrain is blind to the pT-only scaling by construction, yet it still changes as many calls under the boost as the EFN. This means that removing the ΣpT dependence (weakness 7) does not by itself give a boost-invariant mass reading. Open: the Top and QG training pT windows (550–650 and 500–550 GeV) are narrow, so most boosted jets lie outside them, and the network response cannot be fully separated from the off-window shift (C2ST AUC 0.9 at pT × 0.9 on Top).

## Question and hypotheses
Do taggers read jet mass as the boost-invariant m, or as a combination of pT and opening angles fixed by their training pT range?
- Edits, each applied to every particle four-vector, with λ the factor applied to the jet pT:
  - **B**: exact boost along the jet three-momentum with rapidity ζ chosen so |P_jet| → λ|P_jet|. Invariant m and the particle masses are exactly unchanged (max relative change 2e-11), m_A changes by < 0.12 %, and opening angles scale by about 1/λ.
  - **P**: pT-only, all four-vectors × λ at fixed η, φ, so m → λm.
  - **A**: angle-only, (Δη, Δφ) about the stored axis × 1/λ at fixed pT, so m → m/λ.
  - To first order B = P followed by A.
- H0 / H1 (network reads the invariant mass or other boost-invariant shapes): B changes calls at the rotation-control level; P and A move calls in opposite mass directions.
- H1′ (follows pT; D1 ΣpT shortcut): B behaves like P.
- H1″ (reads opening angles at fixed pT, m_A-like at the training pT): B behaves like A.
- Paired test per model × λ: exact McNemar on the jets whose calls change, B vs R, B vs P, B vs A (also P vs R, A vs R, B vs T). Holm correction per family within each dataset, and globally across all datasets (h12_summary.json `holm_global`).
- Correction to the plan: the "boost ×0.9" edit of D1/H3 was pT × k with angles / k, which is a first-order version of B and not a pT-only edit. Edit B reproduces it: JEDI-linear W call 87.3 → 26.8 % here vs 87.5 → 27.0 % in H3.

## Procedure
- Script: `interp/h12_boost_panel.py` (`--part hls_jedi | hls_small | jetclass | top | qg | summary`).
- Jobs: d-h12-jedi (CPU, jax), d-h12-small2 (CPU, conda; it includes the retrains), d-h12-jc (GPU), d-h12-tq (GPU). The Top part of d-h12-tq finished; the pod ended before the QG part wrote anything, so QG was rerun as d-h12-qg (GPU, exit 0). Then d-h12-sum (CPU) ran the summary and the figure.
- Smoke runs: d-h12-sm*.
- Samples (seed 301):
  - hls4ml: `jedi_core.sample(per_class=4000, seed=301)`, N = 20,000, 64 slots, massless particles.
  - JetClass: 4,000 random rows per class from each class's first test file, N = 40,000. Build uses η = jet_η + sign(jet_η)·Δη (round-trip max diff in the json).
  - Top Landscape test: 20,000 top + 20,000 QCD random rows. LorentzNet uses rows [::2].
  - QG test: 20,000 q + 20,000 g random rows.
- λ values: {0.8, 0.9, 1.1, 1.25}. Top and QG also use {0.95, 1.05}, because their training windows are narrower.
- Controls:
  - **R**: rotation about the stored axis, with the pT-weighted mean displacement matched per jet to B's.
  - **T**: translation in a random direction, with the same displacement.
  - Weight-permuted JEDI-linear n64. This null gives one constant call (accuracy 20 %), so it changes 0 % under every edit and is a weak null.
  - EFN, which uses pT fractions z_i and so is exactly invariant under P.
- References per jet: m, m_A = √(M0·M2) about the stored axis, ΣpT, and N, on all jets. On 1,000 jets (rows [::20] for hls4ml, [::40] for the others): τ21, τ32, D2 and m_SD.
- Classical working points: m window and m_A window, each at signal quartiles (ε_S = 0.5); τ32 cut at ε_S 0.3 / 0.5 for top.
- Metrics: % 5/10/2-class calls changed, with a 95 % bootstrap over jets (B = 1,000); class flow matrices; mean Δ log-odds per binary task; signal and background efficiency at the unedited ε_S 0.3 / 0.5 network thresholds; in-window subsets; C2ST (HGB on log ΣpT, m, m_A, N, mean ΔR; 2-fold AUC).
- Decomposition: Δlog-softmax(B) = a·Δ(P) + b·Δ(A), pooled over λ and classes, with a bootstrap over jets.
- Arm 2 (weakness 7): JEDI-style float (`jedis-d64`) retrains, 2 seeds each, via `interp/h12_jnorm_train.py` (job d-h12-jnorm, plus d-h12-jraw for the control):
  - ptnorm: input pT_i/ΣpT.
  - ptflat: per-class flat-ΣpT sample weights.
  - raw: same pipeline, no change.
  - All six trained and wrote test.json. The d-h12-jnorm job shows Failed only because its log wrapper's `tail` call failed after training (exit 1). The boost battery was run on them in this round (hls_small part).

## Unedited accuracy / AUC on these samples [reference]
- JEDI-linear 3-feature: n64 81.00 % [80.92 %], n32 77.92 [78.00], n16 71.92 [71.95]. W vs QCD AUC 0.977 (n64). Permuted n64: 20.0 %.
- Small hls4ml models (seed 0): dsets 80.75 [80.64], linf 80.86 [81.07], tf_d64 81.47 [81.44], jedis 79.33 [79.68], EFN 72.08 [72.06].
- Retrains: ptnorm 76.92 / 76.99 [76.93 / 76.96], ptflat 77.15 / 76.22 [77.22 / 76.16], raw 79.03 / 78.85 [79.29 / 79.12].
- JetClass (10-class): ParT_full 85.75 [85.9], ParT_kin 73.88 [74.5 / 73.7], PN_full 84.09 [84.4 published], PN_kin 70.04.
- Top: ParT_kin 93.98 % (AUC 0.9862) [94.0], PN_kin 94.02 (0.9860) [94.0], LorentzNet 94.15 (0.9865) [94.25], PFN 92.85 (0.9804) [92.83], EFN 92.21 (0.9767) [92.18].
- QG: ParT_kin 82.63 (AUC 0.9020) [82.7], PN_kin 82.69 (0.9021) [82.7], EFN 79.20 (0.8663).

## Numbers
% calls changed [95 % bootstrap]:

| model | B 0.9 | P 0.9 | A 0.9 | R 0.9 | B 1.1 | P 1.1 | A 1.1 | R 1.1 |
|---|---|---|---|---|---|---|---|---|
| JEDI-linear n64 | 32.5 [31.8, 33.1] | 33.8 [33.2, 34.5] | 28.3 [27.6, 28.9] | 2.6 [2.4, 2.8] | 14.4 [14.0, 14.9] | 31.3 [30.6, 31.9] | 24.0 [23.4, 24.6] | 2.4 |
| JEDI-linear n32 | 37.7 | 37.5 | 31.6 | 3.1 | 18.1 | 29.6 | 22.8 | 3.1 |
| JEDI-linear n16 | 38.4 | 22.7 | 29.5 | 4.2 | 21.0 | 22.3 | 21.5 | 3.8 |
| Deep Sets d64 | 25.9 | 25.6 | 31.2 | 1.2 | 11.9 | 26.4 | 22.3 | 1.2 |
| linf-2b-d32-k8 | 36.6 | 35.5 | 29.0 | 0.3 | 15.0 | 30.6 | 22.1 | 0.3 |
| transformer tf-2b-d64 | 36.9 | 27.1 | 31.2 | 0.5 | 15.3 | 25.8 | 22.6 | 0.4 |
| JEDI-style float (jedis) | 34.2 | 9.0 | 30.4 | 0.9 | 17.9 | 12.2 | 20.7 | 0.9 |
| retrain raw s0 | 35.4 | 9.6 | 31.6 | 1.0 | 19.8 | 10.9 | 20.6 | 0.8 |
| retrain ptnorm s0 | 31.2 | 0.0 | 31.3 | 1.1 | 20.3 | 0.0 | 20.5 | 0.8 |
| retrain ptflat s0 | 21.1 | 5.8 | 27.5 | 1.3 | 11.7 | 8.1 | 22.3 | 1.0 |
| EFN (hls4ml) | 36.8 | 0.0 | 37.0 | 0.9 | 24.0 | 0.0 | 24.0 | 0.8 |
| ParT_full (JetClass) | 4.5 [4.3, 4.7] | 6.9 [6.7, 7.2] | 14.4 [14.1, 14.7] | 0.5 | 2.8 | 8.7 | 9.8 | 0.5 |
| ParT_kin | 5.8 | 20.0 | 20.8 | 2.3 | 4.5 | 17.1 | 16.0 | 2.2 |
| PN_full | 4.3 | 8.1 | 14.5 | 1.1 | 3.2 | 9.3 | 10.5 | 0.8 |
| PN_kin | 5.2 | 17.5 | 22.4 | 1.4 | 3.8 | 18.3 | 17.5 | 1.2 |

Top / QG at λ 0.95 (B / P / A / R) and 0.9:

| model | λ 0.95 | λ 0.9 |
|---|---|---|
| Top ParT_kin | 0.6 [0.5, 0.7] / 1.8 / 1.9 / 0.3 | 1.1 / 3.6 / 4.1 / 0.4 |
| Top PN_kin | 0.6 / 1.5 / 1.8 / 0.2 | 1.3 / 3.3 / 4.2 / 0.4 |
| Top LorentzNet | 0.6 / 1.6 / 1.8 / 0.4 | 0.9 / 3.3 / 4.2 / 0.5 |
| Top PFN | 1.8 / 0.0 / 1.9 / 0.4 | 3.8 / 0.0 / 3.9 / 0.8 |
| Top EFN | 2.0 / 0.0 / 2.1 / 0.6 | 4.4 / 0.0 / 4.6 / 1.1 |
| QG ParT_kin | 1.1 [1.0, 1.2] / 0.7 / 1.5 / 0.4 | 2.1 / 1.3 / 3.2 / 0.5 |
| QG PN_kin | 1.0 / 0.7 / 1.5 / 0.2 | 2.0 / 1.3 / 2.8 / 0.4 |
| QG EFN | 2.8 / 0.0 / 2.8 / 0.8 | 5.6 / 0.0 / 5.6 / 1.7 |

### Paired tests
- B vs R: every model × λ has Holm-corrected global p ≤ 3e-3. The largest is Top LorentzNet at λ 0.95 (B − R = +0.2 pp); every other is ≤ 1e-9.
- B vs P, JEDI-linear n64:
  - λ 0.9: B − P = −1.4 pp [CI in json], p 4e-6 (Holm 3e-5).
  - λ 1.1: −16.8 pp, p < 1e-300.
  - n32 at λ 0.9: +0.2 pp, p 0.52.
  - Deep Sets at λ 0.9: +0.3 pp, p 0.31.
- B vs A:
  - EFN, PFN and the ptnorm retrain: |B − A| ≤ 0.25 pp. QG EFN: p 0.10–0.49, not significant after Holm. hls4ml / Top EFN–PFN: p 1e-3 to 1e-13 at |Δ| ≤ 0.25 pp.
  - JetClass and Top transformers / GNNs: B − A = −2.3 to −17.2 pp, all p < 1e-100.

### Class flow (net % of jets moved to a heavier-mass class, λ 0.9 / 1.1)
Mass order: g = q < W < Z < H < t.

| model | B | P | A |
|---|---|---|---|
| JEDI-linear n64 | +22.5 / −7.7 | +11.7 / +4.6 | +16.2 / −20.8 |
| ParT_full | +1.5 / −1.1 | −4.7 / +4.7 | +5.4 / −6.6 |
| PN_kin | −0.1 / −0.1 | −11.2 / +10.3 | +10.6 / −10.8 |

- Under B0.9 JEDI-linear sends W → t (2,406 jets) and Z → t (1,630). Under P0.9 it sends W/Z → t as well (1,312 and 2,258), so for JEDI-linear lowering pT at fixed angles moves calls to the heavier class. This is the opposite of what a mass reading predicts.
- For JetClass, P and A move calls in opposite mass directions, as a mass reading predicts, and B nearly cancels.

### Mean Δ log-odds
- W vs QCD (JEDI-linear n64), mean [95 %]:
  - λ 0.9: B −1.67 [−1.71, −1.64], P −1.77, A −1.01, R −0.00 [−0.01, +0.01].
  - λ 1.1: B −0.42, P −1.06, A −0.72.
- ParT_full W vs QCD:
  - λ 0.9: B −0.12, P −0.26, A −0.40.
  - λ 1.1: B +0.02, P −0.16, A −0.31.
- Top ParT_kin top vs QCD, λ 0.9: B −0.04, P −0.45, A −0.22, R −0.00.
- QG ParT_kin q vs g:
  - λ 0.95: B −0.067, P +0.018, A −0.089.
  - λ 1.05: B +0.071, P −0.019, A +0.088.

### Decomposition B = a·P + b·A
Fitted on Δlog-softmax, pooled over λ; [95 %] bootstrap over jets.
- JEDI-linear n64: a 0.56 [0.555, 0.565], b 0.18 [0.179, 0.186], R² 0.61.
- n32: a 0.60, b 0.25, R² 0.65.
- n16: a 0.62, b 0.65, R² 0.71.
- Deep Sets: a 0.44, b 0.31, R² 0.59.
- transformer: a 0.53, b 0.43, R² 0.66.
- jedis: a 0.48, b 0.84, R² 0.76.
- retrain raw: a 0.62, b 0.82, R² 0.81.
- ptflat: a 0.24, b 0.77, R² 0.77.
- ptnorm / EFN: b 0.998, R² 1.000 (P has no response).
- JetClass ParT_full: a −0.11, b 0.28, R² 0.49; PN_kin: a −0.02, b 0.13, R² 0.16.
- Top ParT_kin: a 0.01, b 0.11, R² 0.12.
- QG ParT_kin: a 0.58, b 0.84, R² 0.87.
- rms(B) / rms(P + A): JEDI n64 2.12 / 4.04; ParT_full 0.79 / 2.71; Top ParT 0.34 / 1.57; QG ParT 0.17 / 0.26 (rms R 0.04).

### References and working points
- Under B: m changes by < 1e-10 (relative), m_A by ≤ 0.12 %.
- τ21, τ32 and D2 change by ≤ 0.5 % (median |rel|), and m_SD by ≤ 0.3 GeV on average.
- m- and m_A-window signal efficiencies stay at 0.50 (all datasets).
- τ32 cut efficiencies (top): sig 0.30 → 0.30–0.303, bkg 0.025 → 0.023–0.025.
- P and A move the m window: JEDI hls4ml W window sig 0.50 → 0.007 under P0.9 and 0.50 → 0.019 under A0.9.
- τ ratios are invariant under P and A as well (scale-free), so they do not separate the hypotheses.

### On-manifold checks
- In-window fraction under B0.9: hls4ml 0.986 (window = 0.5–99.5 % of sample jet pT), JetClass 0.640 (500–1000 GeV), Top 0.323, QG 0 (0.95: 0.41).
- B calls changed on in-window jets:
  - JEDI n64 32.8 %.
  - ParT_full 4.2 %.
  - Top ParT 1.2 %.
  - QG ParT at λ 0.95: 1.15 % [1.00, 1.34].
- Central-95 % pT window (post hoc, h12_summary.json): JEDI n64 B0.9 35.8 %, ParT_full 4.2 %, Top ParT 1.2 %.
- C2ST AUC at λ 0.9: B 0.88 (hls4ml), 0.69 (JetClass), 0.93 (Top); A 0.74 / 0.62 / 0.67; R ≈ 0.50.

## Reading
- **hls4ml-trained models (11 nets, 4 architectures, 5 training recipes).**
  - B changes 21–38 % of calls at λ 0.9 against 0.3–4.2 % for R, so H1 is excluded for every one.
  - JEDI-linear n64/n32, linf and Deep Sets follow P at λ 0.9 (B − P within ±1.4 pp), but not at λ 1.1 (B 12–18 % vs P 26–31 %). The pT dependence is therefore asymmetric: lowering pT at fixed m sends W/Z calls to t.
  - The float JEDI-style net and its raw retrain respond weakly to P (9–10 %) and follow A (b ≈ 0.82–0.84).
  - The pT-normalised retrain and the EFN cannot see the overall pT, and B = A exactly within 0.25 pp.
- **Weakness 7 (ΣpT shortcut).**
  - The flat-ΣpT-weighted retrain lowers the P response from 9.6 % (raw retrain) to 5.8 %, and the B response from 35.4 % to 21.1 %.
  - The ptnorm retrain removes the P response by construction, but its B response stays at 31 %.
  - Across these retrains, the size of the pT-only response does not set the boost response. At the single training pT scale, the networks read opening angles.
- **JetClass ParT / ParticleNet.**
  - B is 3–4 times smaller than P or A, and P and A move calls in opposite mass directions (−4.7 vs +5.4 net mass-ordered flow for ParT_full). This matches a mainly mass-like reading.
  - The remaining B response (4.3–5.8 %, 2–8 times R) is excluded from H1 and is not explained by a linear P/A mix (R² 0.13–0.49).
- **Top Landscape.**
  - ParT, ParticleNet and LorentzNet: B 0.6 % vs P/A 1.5–1.9 % vs R 0.2–0.4 % at λ 0.95, and the same pattern at 0.9.
  - PFN and EFN see only pT fractions, so B = A (1.8–2.0 %).
- **QG.** ParT and ParticleNet B responses lie between P and A, with b ≈ 0.84 (A-dominated). For quark/gluon jets the angular width at fixed pT is the physical discriminant, and a boost changes it at fixed m. A non-zero B response here is not evidence against a mass reading.
- **Not measured.** Whether the JetClass / Top B residual comes from the pT dependence the networks learned from their training pT spectra (which would be physical), or from extrapolation off-window. The in-window subsets keep the B ≪ P, A ordering but do not separate these two.

## Explainer entry (for site/explain.py, H12_EXPLAINER)
1. **Question.** Does a tagger read jet mass as the Lorentz-invariant $m$, or as a pT–angle combination tied to the pT range it was trained on?
2. **Physics background.**
   - For a boosted jet, $m^2\approx p_T^2\langle\Delta R^2\rangle$-type sums: $m^2\simeq\sum_{i<j}p_{T,i}p_{T,j}\Delta R_{ij}^2$. Lowering pT by a boost along the jet axis at fixed $m$ widens the opening angles by $\approx1/\lambda$.
   - An invariant-mass reader is unchanged by such a boost. A reader of $p_T$ (or $\sum p_T$) responds like a pure pT scaling $P_\lambda$ ($m\to\lambda m$). A reader of angles at fixed pT responds like a pure angle scaling $A_\lambda$ ($\Delta R\to\Delta R/\lambda$, $m\to m/\lambda$).
   - To first order the boost $B_\lambda=A_\lambda\circ P_\lambda$.
3. **Method.**
   - Exact boost with rapidity $\zeta=\operatorname{arsinh}(\lambda|\vec P|/m)-\operatorname{arsinh}(|\vec P|/m)$ along $\hat P$, applied to every four-vector.
   - Controls: a rotation about the axis and a random translation, each with the same pT-weighted displacement. Also a weight-permuted JEDI-linear, and EFN / pT-normalised models that are exactly invariant under $P$.
   - Samples: 20,000 hls4ml, 40,000 JetClass, 40,000 Top, 40,000 QG jets (seed 301).
   - Metrics: % calls changed (bootstrap CI), exact McNemar against the rotation, Holm correction, class flow, Δlog-odds, and the fit $\Delta_B=a\Delta_P+b\Delta_A$.
4. **Result (figure `results/physics/h12_boost.png`).**
   - At pT × 0.9, hls4ml-trained models change 21–38 % of calls under B (JEDI-linear 32.5 %, rotation 2.6 %), about as many as under $P$ or $A$.
   - JetClass ParT / ParticleNet: 4.3–5.8 % under B vs 7–22 % under $P$ / $A$.
   - Top taggers at pT × 0.95: 0.6 % vs 1.5–1.9 %.
   - QG taggers: B between $P$ and $A$.
5. **Checks.**
   - Unedited accuracies reproduce the references within 0.6 points.
   - $m$ fixed to $10^{-10}$ and $m_A$ to 0.12 % under B; τ21, τ32 and D2 unchanged within 0.5 %.
   - Edit B reproduces the D1/H3 JEDI W-call drop (87 → 27 %).
   - In-window subsets give the same ordering.
   - C2ST: boosted jets are separable from natural ones (AUC 0.69–0.93).
   - pT-normalised and flat-ΣpT retrains of the JEDI-style net still respond to B at 21–31 %.
6. **Physical significance.**
   - Taggers trained on one narrow pT bin (hls4ml, ~1 TeV) learn mass through opening angles at that pT, so the same jet seen at another pT is called differently even though its mass is unchanged.
   - Taggers trained over wide pT ranges (JetClass) or with explicit four-vectors (Top ParT, LorentzNet) respond 3–4 times less to the boost than to either single scaling. This matches a mostly invariant-mass reading with a residual pT dependence.

## Files
- Script: `interp/h12_boost_panel.py`; `interp/h12_jnorm_train.py` (retrains, in `/j-jepa-vol/aaron-interp/models/small/jedis-d64-n64-h12{ptnorm,ptflat,raw}/seed{0,1}`).
- Results (local copies in `results/physics/`): `h12_hls_jedi.json`, `h12_hls_small.json`, `h12_jetclass.json`, `h12_top.json`, `h12_qg.json` (each with a `_perjet.npz`), `h12_summary.json` (global Holm, central-95 % window), and the figure `h12_boost.png`.
- Logs: `/j-jepa-vol/aaron-interp/logs/d-h12-{jedi,small2,jc,tq,qg,sum,jnorm,jraw}.log`.
