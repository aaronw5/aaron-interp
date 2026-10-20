# H25v — skeptic pass on H25 (top taggers and τ32 at fixed jet mass and W-pair mass)

Script `interp/h25_verify.py`. Job `d-h25-verify` (GPU, 12 min, exit 0); smoke `d-h25-verify-smoke` (crashed with exit 134 on overflowing coordinates in the first version of the new edit; fixed by the max-offset normalisation and the ΔR guard below). Results: `results/physics/h25_verify.json`. Log: `logs/d-h25-verify.log`.

## Conclusion
We repeated H25 on fresh jets and got the same result. When the three subjets are made narrower or wider at fixed jet mass and fixed W-pair mass, every tagger's score moves the way a τ32-based classifier predicts, by a similar amount (Top ParT −1.15 log-odds on top jets at s = 1.4, against −1.14 in H25). We then added a test that H25 lacked: we changed τ32 while keeping each subjet's mass fixed, by reshaping the radial profile of the particles inside each subjet. Here the networks did not follow τ32 consistently. Making the subjet cores sharper (τ32 down) pushed four of seven networks toward "top". Making the profile flatter (τ32 up) left the Top-Landscape networks near zero or pushed them the wrong way, and PFN moved against τ32 in every case. So the taggers respond to the shape of the radiation inside the prongs beyond m and mW, but the H25 response cannot be read as "the networks use τ32". It goes mostly with the subjet width and subjet mass that H25's edit changes together with τ32, plus a dependence on the core profile that differs from network to network. Both edits produce jets that a classifier can tell apart from natural jets, so the networks are partly extrapolating.

## Code review of `interp/h25_top_tau32.py`
I found no bug that changes the numbers. What I checked:
- **Rows.** Random rows on both datasets: `data_top.sample` rng and `jc_read` rng among eligible rows; no first-k selection.
- **Padding.** Padded particles are excluded everywhere: `_jet` uses `P4[j, :nn[j]]`, the axis uses the `real` mask, and `jet_inputs` / `build` mask by `nn` or pT > 0.
- **Axis.** Each edited jet is rebuilt about its own constituent-sum axis for every model:
  - Top nets: `jet_inputs` recomputes the axis.
  - JetClass nets: `jet_eta` / `jet_phi` are replaced; `build` recomputes `jet_pt` and `jet_energy` and uses η = jet_eta + sign(jet_eta)·Δη.
  - The c3 translation control gives |Δ| ≤ 0.12, which confirms this.
- **Same jets.** Observables and model scores are computed on the same edited arrays. For Top, `sort_pt` is applied before both; pT is unchanged by the edits, so the order is the same.
- **Conservation.** The edit does not claim four-momentum conservation for the subjets; the report states that subjet masses change. m and mW are restored and checked after re-clustering. The median |Δm| is < 0.004 GeV here as well.
- **Working point.** ε_S = 0.3 / 0.5 thresholds are taken from the sample's unedited tops. The report says rejection is unresolved.

Points the report should state more plainly:
1. **LR2 is a construction check.** LR(m, mW) = 0 is guaranteed by the restore of m and mW. It shows that the restore worked; it is not an informative null.
2. **c1 is not a null on QCD.** It moves QCD scores by up to +0.43 (LorentzNet, JetClass PN_kin), and LR3 / MLP3 by +0.17 at s = 0.5. On QCD, part of S − c1 comes from c1. The sign of dS alone (without subtracting c1) still agrees with H1 in all cells except PFN, Top QCD, s = 2.0.
3. **The analysis sets at s > 1 are selected.** S is accepted for 19–34 % of jets at s = 1.4 and 5–10 % at s = 2.0. c1 is accepted for 54–65 %.
4. **"Same direction as a τ32 classifier" does not single out τ32.** MLPx, which has no τ input, predicts the same sign in every S cell.

## Procedure (rerun and new control)
**Samples.** Fresh rows, with the H25 rows excluded by replaying H25's rng calls.
- **Top Landscape test:** 2,000 top + 2,000 QCD, rng 52025.
- **JetClass `TTBar_100` / `ZJetsToNuNu_100`:** 550–650 GeV, |η| < 2, ≥ 3 particles; 2,000 per class, rng 52025 / 52026.
- **References:** refit exactly as in H25 (same rows and seeds; Top val and JetClass `_101`, 25k per class).

**Unedited accuracy (AUC / acc, 4,000 jets).**

| sample | model | AUC | acc. |
|---|---|---|---|
| Top | Top ParT | 0.9856 | 93.6 % |
| Top | Top PN | 0.9845 | 93.6 % |
| Top | LorentzNet | 0.9857 | 93.6 % |
| Top | PFN | 0.9796 | 92.8 % |
| Top | EFN | 0.9750 | 91.9 % |
| Top | JetClass ParT_kin | 0.9811 | 92.4 % |
| Top | JetClass PN_kin | 0.9821 | 92.6 % |
| JetClass | JetClass ParT_kin | 0.9925 | – |
| JetClass | JetClass PN_kin | 0.9892 | – |
| JetClass | Top ParT | 0.9872 | – |

**Edits.**
- **S(s), c1(s), s ∈ {0.7, 1.4}:** H25's own `edit_jet`, imported unchanged.
- **New P(k), k ∈ {0.5, 1.6}: profile edit at fixed subjet mass.**
  - Inside each exclusive-kT subjet, each constituent keeps the direction of its offset from the subjet centroid. The offset length becomes ρ → λ·ρmax·(ρ/ρmax)^k.
  - λ is found by bisection so that the subjet invariant mass equals its unedited value (|Δm_k| < 0.05 GeV).
  - At fixed second moment, k < 1 flattens the profile, which raises the pT-weighted mean distance and so raises τ3 and τ32. k > 1 sharpens the core and lowers τ32.
  - Then the H25 rigid restore brings back mW and m.
  - Jets with an edited constituent beyond ΔR = 1.0 from the axis are rejected (1.2 after the restore). This removes 47 % of jets at k = 1.6, so the P1.6 set is a selected subset.
- **cP(k):** the H25 c1 rotation with P's pT-weighted displacement on the same jet, then the same restore.
- **Acceptance:** |Δm| < 0.5 GeV and |ΔmW| < 2 GeV.

**Metric and tests.**
- Metric: mean over jets of (edit − control) in top log-odds, with a 95 % bootstrap CI (B = 2,000).
- H0 (edit − control = 0): two-sided sign-flip permutation (5,000).
- H0_side (network response = MLPx response, per jet): sign-flip permutation.
- Holm correction over 112 tests (7 networks × 4 edits × 2 samples × 2 classes); the Holm floor is 0.0224.

## What the P edit does (top jets; Top sample / JetClass sample)
| | P0.5 | P1.6 |
|---|---|---|
| Δτ32 (P − cP) | +0.063 [+0.060, +0.066] / +0.072 | −0.086 [−0.089, −0.082] / −0.087 |
| median subjet-mass-sum change | 0.00 GeV (median abs 0.04) / 0.00 | −0.41 GeV (median abs 0.89) / −0.09 |
| ΔmSD | +2.5 / +1.4 GeV | −2.1 / −1.5 GeV |
| Δτ21 | +0.032 / +0.033 | −0.035 / −0.037 |
| MLPx (side effects, no τ) | −0.04 / −0.02 | +0.06 / +0.04 |
| MLP3 (m, τ32, mW) | −0.48 [−0.51, −0.46] / −0.54 | +0.60 [+0.57, +0.62] / +0.57 |
| C2ST AUC, P / cP / identity | 0.78 / 0.52 / 0.50 (Top); 0.86 / 0.54 / 0.54 (JetClass) | 0.71 / 0.50 / 0.50; 0.79 / 0.55 / 0.56 |

Compared with S, P changes τ32 by a similar amount (S at s = 0.7: −0.110) while the subjet masses stay fixed, where S changes their sum by about 16 GeV. MLPx does not respond to P (|Δ| ≤ 0.06), so the side-effect reference predicts no response, while MLP3 predicts about ±0.5.

## Numbers
### S replication (top jets)
Edit − control, log-odds [95 % CI]; H25's value in brackets.

| model | Top sample, s = 0.7 | Top sample, s = 1.4 | JetClass sample, s = 1.4 |
|---|---|---|---|
| Top ParT | +0.67 [+0.60, +0.73] (+0.62) | −1.15 [−1.26, −1.04] (−1.14) | −1.16 (−1.14) |
| Top PN | +1.00 (+0.98) | −1.39 (−1.48) | −1.83 (−1.83) |
| LorentzNet | +0.80 (+0.79) | −1.19 (−1.23) | −2.03 (−1.85) |
| PFN | +0.17 (+0.17) | −0.67 (−0.67) | −0.43 (−0.49) |
| EFN | +1.03 (+1.04) | −1.47 (−1.43) | −1.17 (−1.20) |
| JetClass ParT_kin | +0.56 (+0.57) | −1.11 (−1.16) | −1.39 (−1.38) |
| JetClass PN_kin | +0.74 (+0.78) | −0.95 (−1.04) | −1.42 (−1.43) |
| MLP3 | +0.81 (+0.79) | −0.85 (−0.76) | −0.73 (−0.75) |
| MLPx | +0.32 (+0.32) | −0.65 (−0.62) | −0.42 (−0.43) |

- 55 of 56 S tests pass Holm p < 0.05, all with the sign MLP3 predicts. 47 of 56 exceed MLPx at Holm p < 0.05.
- The replication agrees with H25 within about 0.2 log-odds in every cell.

### P (τ32 changed at fixed subjet mass)
P − cP, log-odds; * = Holm p < 0.05. "Same sign as MLP3" means the network moves the way τ32 predicts.

| model | Top tops P0.5 | Top tops P1.6 | JetClass tops P0.5 | JetClass tops P1.6 | Top QCD P0.5 / P1.6 | JetClass QCD P0.5 / P1.6 |
|---|---|---|---|---|---|---|
| MLP3 (τ32 prediction) | −0.48 | +0.60 | −0.54 | +0.57 | −0.35 / +0.55 | −0.27 / +0.47 |
| Top ParT | +0.09 [+0.03, +0.15] | +0.72* | −0.02 | +0.68* | +0.31* / +0.34* | +0.09* / +0.30* |
| Top PN | +0.15* | +0.98* | −0.13* | +1.57* | +0.26* / +0.85* | −0.04 / +1.25* |
| LorentzNet | +0.04 | +1.05* | −0.23* | +2.09* | +0.15 / +0.82* | −0.09 / +0.96* |
| PFN | +0.28* | −0.37* | +0.17* | −0.14* | +0.30* / −0.17* | +0.16* / −0.11* |
| EFN | −0.11* | +0.17* | −0.12* | +0.18* | +0.12* / −0.03 | +0.05* / −0.03 |
| JetClass ParT_kin | −0.10* | −0.32* | −0.45* | +0.17 | +0.04 / −1.35* | +0.29* / +0.40* |
| JetClass PN_kin | +0.11* | +0.83* | −0.41* | +1.22* | +0.18* / +0.59* | +0.17* / +0.52* |

- **Holm.** 46 of 56 P tests pass Holm p < 0.05: 26 with the τ32 (MLP3) sign and 20 with the opposite sign. 44 of 56 differ from MLPx at Holm p < 0.05.
- **Ratio to MLP3.**
  - P1.6 on tops: Top ParT 1.2, Top PN 1.6–2.8, LorentzNet 1.8–3.7, JetClass PN_kin 1.4–2.1, EFN 0.3, JetClass ParT_kin −0.5 to +0.3, PFN −0.6 to −0.25.
  - P0.5 on Top tops: −0.6 to +0.2 for all seven networks.
  - P0.5 on JetClass tops: JetClass ParT_kin 0.84, JetClass PN_kin 0.77, the Top nets −0.3 to +0.4.

## Hypotheses (P edit)
| hypothesis | prediction for P − cP | observed | status |
|---|---|---|---|
| H_tau: the network reads τ32 (first-moment width) as MLP3 does | sign of MLP3 in all cells, ratio about as under S (0.7–2.8) | the MLP3 sign in 26 / 46 significant cells; P0.5 on Top tops −0.6 to +0.2 × MLP3 for all networks; PFN opposite in 8 / 8 cells | excluded as a general description; holds for P1.6 for Top ParT, PN, LorentzNet and JetClass PN_kin |
| H_mass: the network reads subjet masses / second moments only (MLPx-like) | about 0 (MLPx ≤ 0.06) | 44 / 56 differ from MLPx; P1.6 responses up to +2.1 | excluded |
| H_profile: the network reads the radial profile inside the prongs, in a way that differs between networks | an asymmetric, network-dependent response | sharper cores move toward top for four networks; a flatter profile gives about 0 for the Top nets; PFN reverses | consistent; not tested directly |

## Corrected reading
- **S replication.** Under H25's subjet-width edit at fixed m and mW, all seven networks move their top log-odds against the rotation control. The response has the sign predicted by both the (m, τ32, mW) and the side-effect references, and it reproduces on fresh rows (Top ParT on Top tops at s = 1.4: −1.15 vs −1.14).
- **S does not isolate τ32.** S changes τ32, subjet masses (median sum ±12 to 17 GeV), mSD and τ21 together. MLPx without τ predicts the same sign in all S cells.
- **P edit.** When τ32 is changed at fixed subjet masses, the networks do not follow τ32 consistently:
  - Sharper cores (τ32 −0.09) move Top ParT, Top PN, LorentzNet and JetClass PN_kin toward top, by 1.2–3.7 × MLP3.
  - A flatter profile (τ32 +0.06) moves no Top-trained network toward QCD on Top tops.
  - PFN moves against τ32 in every cell.
  - JetClass ParT_kin is mixed, and is −1.35 on Top QCD at P1.6.
- **Overall.** The networks use the radiation pattern inside the prongs beyond m and mW. The data do not support reading the H25 response as τ32 use; it goes with subjet width and mass, plus a network-dependent dependence on the core profile.
- **Off-manifold.** Both S and P produce off-manifold jets (C2ST 0.71–0.86 for P, 0.74–1.00 for S).

## Issues
- P1.6 rejects 47 % of jets for constituents beyond ΔR = 1.0, so it is a selected subset. P / cP acceptance is 55–62 %.
- P also changes mSD by ±1.4 to 2.5 GeV and τ21 by ±0.03. MLPx, which includes mSD, does not respond (|Δ| ≤ 0.06).
- c1 / cP are not nulls on QCD (up to +0.3).
- The P edit was designed in this verification and is not pre-registered. Only two k values were run.

## Explainer entry (for site/explain.py; to be merged by the ledger)
```python
H25V_EXPLAINER = [
 ('Question', 'H25 found that top taggers react to making subjets narrower or wider at fixed jet mass $m$ and W-pair mass $m_W$. That edit changes $\\tau_{32}$ and the subjet masses together. Do the taggers follow $\\tau_{32}$ itself when the subjet masses are held fixed?', None),
 ('Physics background', 'For massless constituents a subjet mass is $m_k^2\\approx p_{T,k}^2\\,\\mathrm{Var}_{p_T}(\\Delta\\vec R)$, a second moment of the angular offsets, while $\\tau_3\\propto\\sum_i p_{T,i}\\,\\Delta R_{i,\\mathrm{axis}}$ is a first moment. At fixed second moment, a flatter radial profile raises the first moment ($\\langle\\rho\\rangle\\le\\sqrt{\\langle\\rho^2\\rangle}$ with equality for a flat profile), so $\\tau_{32}$ can be moved at fixed $m_k$.', None),
 ('Method', 'Fresh rows (H25 rows excluded): Top Landscape and JetClass (550–650 GeV), 2,000 top + 2,000 QCD each. Replication of H25 $S(s)$, $s=0.7, 1.4$, with its rotation control. New edit $P(k)$: inside each exclusive-$k_T$ subjet, $\\rho_i\\to\\lambda\\rho_{\\max}(\\rho_i/\\rho_{\\max})^k$ with $\\lambda$ solved so that $m_k$ is unchanged ($|\\Delta m_k|<0.05$ GeV), $k=0.5$ (flatter, $\\tau_{32}$ up) and $1.6$ (sharper, $\\tau_{32}$ down); then $m$ and $m_W$ restored by rigid subjet moves. Control $cP$: rotation inside each subjet with the same displacement. Metric: $P-cP$ in top log-odds, bootstrap CI, sign-flip permutation, Holm over 112 tests; references MLP$(m,\\tau_{32},m_W)$ and MLPx (side effects, no $\\tau$).', None),
 ('Result', 'The $S$ responses reproduce H25 within about 0.2 log-odds (Top ParT $-1.15$ vs $-1.14$ at $s=1.4$). $P$ moves $\\tau_{32}$ by $+0.06$ / $-0.09$ with the subjet masses fixed; MLP$(m,\\tau_{32},m_W)$ predicts $\\mp0.5$ and MLPx predicts 0. Sharper cores move Top ParT, PN, LorentzNet and JetClass PN_kin toward top (1.2–3.7 × the $\\tau_{32}$ classifier). A flatter profile leaves the Top-Landscape networks at $-0.6$ to $+0.2$ × the prediction. PFN moves against $\\tau_{32}$ in all 8 cells. Only 26 of the 46 significant cells have the $\\tau_{32}$ sign.', None),
 ('Checks', 'Unedited AUCs 0.975–0.993. Subjet mass changes under $P$: median $\\le0.9$ GeV versus 12–17 GeV under $S$. The $P(1.6)$ set excludes 47 % of jets (constituents pushed beyond $\\Delta R=1$). C2ST of edited vs natural tops: 0.71–0.86 for $P$ (identity 0.50–0.56). The $P$ edit was designed in this verification and is not pre-registered.', None),
 ('Physical significance', 'Top taggers use the radiation pattern inside the three prongs beyond $m$ and $m_W$, but not as the single number $\\tau_{32}$: their response follows the subjet width and mass and depends, differently for each network, on how sharp the subjet cores are.', None),
]
```
Entry line for the `'d'` list: `('Verification: do top taggers follow τ32 when the subjet masses are held fixed? (claim D6; H25v)', H25V_EXPLAINER, False)`.
