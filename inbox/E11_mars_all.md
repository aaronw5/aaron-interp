# E11 — the whole network as hinge formulas (agent report, 2026-10-16)

Script: interp/e11_mars_all.py (modes fit a <stage> / fit b / control untrained|shuffle / eval). Results: results/jedi/e11_fit_a_<stage>.json,
e11_fit_b_<stage>.json (formula.py term format, plus per-term spread, weight share and importance), e11_control_untrained_*.json,
e11_control_shuffle_S.json, e11_mars_all.json (R², substitution, controls, per-unit lookup, family shares), e11_evolution.png.
Logs: logs/e11-{b,a-g,a-s-stage,a-h4,a-h5,a-z,untr-1,untr-2,shuf-s,eval,eval2,smoke}.log (cluster jobs; logs/e11.log belongs to the E11-windows run).
All fits ran on the cluster (1 CPU / 2 Gi each, 8–20 min per job); supplement with half-plane sums: prefix e11hp (section 7).

## 1. Procedure
- Network: JEDI-linear 3-feature n64, exact fixed-point forward (interp.jedi_core.Jedi). Stages P, g, S, h4, h5, z; live units = mean-ablation
  cost ≥ 0.1 % of calls (E1 table, results/jedi/e1_e3.json): P 12, g 11, S 19, h4 54, h5 32, z 16 → 144 units.
- Jets: everything from the TEST file (the earlier MARS work, J12, fitted on the TRAIN archive). sample(per_class=6000, seed=0) = 30,000 jets, fixed
  permutation (rng 0) → 20,000 fit / 10,000 validation. Held-out test: sample(per_class=2000, seed=11) = 10,000 jets minus the 1,152 also in the
  fitting sample → 8,848 jets (1,718–1,777 per class). Network accuracy on them 80.42 % (full test file 80.9 %).
- Target per unit: post-ReLU value for P, g, S, h4, h5; pre-ReLU value for z, clipped below at −0.2·spread as mars.py does. R² throughout is
  mars.py's: of max(0, formula) against max(0, unit).
- Fitter: jetdistill.mars.fit_neuron unchanged (forward selection over Q, max(0, Q − t), max(0, t − Q), products of a chosen threshold term with a
  threshold term of one of the 10 observables most correlated with the residual); knots = 5–95 % quantiles (19 per input); cap 30 terms per unit;
  stop after 5 steps without validation gain; selection step on 6,000 of the fit jets (mars.py default 12,000; reduced for the 2 Gi jobs),
  coefficients refit on all 20,000.
- Inputs (a) physics: the test file's Q library (383) + 95 hand-built sums over pT-rank windows (all, top 14, ranks 1–63, 13–63, top 39, 26, 3, 11,
  ranks 8–63): N, ΣpT, ΣpT·r², Σr², Σr, ΣpT·r, Σ(r − 0.2)₊, N(pT > 1 GeV), N(pT < 2 GeV), Σr[r < 0.165], N(pT < 2, r > 0.1), plus |M1| and
  √(M0·M2); 475 inputs after removing constant ones (pT in GeV, r = ΔR in rad).
- Inputs (b) compositional, following the network's graph: P ← physics; g ← the 12 fitted P values; S ← the 11 fitted g values + physics (S reads
  the particles directly and g); h4 ← the 19 fitted S; h5 ← the 54 fitted h4; z ← the 32 fitted h5. Each stage is fitted on the CHAIN values of the
  previous stage (formulas evaluated recursively from physics on the same jets), so (b) is one nested program particles → z. (b) at P = (a) at P.
- Evaluation on the 8,848 test jets: per-unit R²; stage substitution with Jedi.run_jetwise (stage k's live units replaced per jet, non-live units
  and everything downstream computed by the network): single (a) = f_k^(a)(physics); single (b) = f_k^(b) applied to the network's TRUE values of
  stage k−1; cumulative (a) = stages P..k all replaced by their physics formulas; cumulative (b) = P..k replaced by the chain values. Same call = %
  of jets with the network's own call; accuracy vs the labels. Reference row: all live units of the stage set to their mean.
- Controls: untrained network (Jedi(untrained=True, seed 0): weights redrawn from the trained mean/SD), same pipeline (a) with the same number of
  units per stage, chosen by activation spread; shuffled targets (stage S permuted across jets in every split, rng 2).

## 2. Per-unit held-out R² (test jets)
| stage | live units | (a) median / min / cost-weighted | n(R² ≥ 0.9) of (a) | (b) chain median / min / cost-w | (b) single median / min / cost-w | terms (a) / (b) | untrained, (a) median / min / max |
|---|---|---|---|---|---|---|---|
| P | 12 | 0.817 / 0.785 / 0.894 | 3 | 0.817 / 0.785 / 0.894 | 0.817 / 0.785 / 0.894 | 309 / 309 | 0.803 / 0.188 / 0.839 (9 units with spread > 0) |
| g | 11 | 0.752 / 0.368 / 0.805 | 3 | 0.572 / 0.390 / 0.791 | 0.735 / 0.541 / 0.876 | 304 / 252 | 0.773 / 0.665 / 0.851 (6 units) |
| S | 19 | 0.981 / 0.535 / 0.968 | 16 | 0.982 / 0.555 / 0.969 | 0.982 / 0.691 / 0.974 | 496 / 504 | 0.084 / 0.000 / 0.169 (2 units) |
| h4 | 54 | 0.950 / 0.547 / 0.954 | 40 | 0.958 / 0.610 / 0.961 | 0.985 / 0.654 / 0.981 | 1558 / 1348 | UNTR_H4 |
| h5 | 32 | 0.923 / 0.566 / 0.937 | 21 | 0.946 / 0.653 / 0.956 | 0.979 / 0.752 / 0.979 | 928 / 948 | UNTR_H5 |
| z | 16 | 0.897 / 0.720 / 0.912 | 6 | 0.941 / 0.825 / 0.948 | 0.983 / 0.947 / 0.987 | 477 / 463 | 0.140 / 0.091 / 1.000 (15 units; the 1.000 is a unit equal to n_real_top10) |
Cost-weighted = weighted by the unit's E1 ablation cost. Shuffled-target control, stage S: R² median −0.000, range −0.001 … 0.000 (33 terms kept over
19 units). Validation − test R² differences: −0.005 … +0.002 per stage.
- P: P15 = ΣpT (1 term, R² 1.000); P56 = ΣpT over ranks 8–63 (0.999); P32 = soft count (0.998). The nine angular units (E2's half-planes) stop at
  0.785–0.843 with Σr plus per-particle signed angles and the orientation angle: no library observable is a one-sided angular sum.
- g: g49 (cost 10.4 %) = hinge in ΣpT (sum_pt < 1263 GeV, R² 0.998; on P̂: P15 < 5.68, 0.998); g4 = Σr (0.994); g32 = P56 (0.999, 1 term);
  g55, g39, g5, g29 (half-plane P inputs) 0.37–0.46 on physics and 0.39–0.46 on P̂.
- S: 16 of 19 at ≥ 0.9; S31 0.535 and S58 0.668 are the two g-only objects (E5), S56 0.896. S0 on the chain uses g:49 (0.998).
- (b) single ≥ (b) chain ≥ (a) from S onward; the chain loses most at g (median 0.572 vs 0.752).

## 3. Stage substitution (same call % / accuracy %; network 80.42 % on these jets)
| stage replaced | single (a) | single (b) on true inputs | cumulative P..k (a) | cumulative P..k (b), the nested program | all live units at their mean |
|---|---|---|---|---|---|
| P | 95.7 / 80.22 | 95.7 / 80.22 | 95.7 / 80.22 | 95.7 / 80.22 | 81.6 / 71.91 |
| g | 96.7 / 80.57 | 96.5 / 80.53 | 96.7 / 80.57 | 96.2 / 80.72 | 85.3 / 74.28 |
| S | 94.3 / 80.85 | 94.8 / 81.09 | 94.3 / 80.85 | 94.2 / 80.99 | 18.7 / 19.91 |
| h4 | 94.7 / 81.06 | 95.7 / 79.59 | 94.7 / 81.04 | 94.6 / 80.83 | 18.7 / 19.91 |
| h5 | 93.7 / 80.94 | 96.2 / 80.09 | 93.7 / 80.94 | 94.2 / 80.63 | 19.9 / 20.14 |
| z | 92.1 / 80.11 | 97.1 / 80.31 | 92.1 / 80.11 | **94.0 / 80.72** | 19.9 / 20.14 |
- The all-if-statement model (last row of cumulative (b), then the network's own last layer with its rounding): 94.0 % same call, 80.72 % accuracy
  vs 80.42 % network on the same jets. Size: 144 units, 3,824 terms, 6,064 if-statements (2,344 of the terms are products), 394 distinct inputs
  (211 physics quantities enter through the P and S formulas). Per-class accuracy g/q/W/Z/t: program 78.0 / 70.7 / 87.1 / 85.0 / 82.8; network
  NET_PER_CLASS.
- z fitted directly on physics (the 16-neuron MARS formula redone on test-file jets, 30-term cap): 92.1 % same call, 80.11 % (477 terms, 160
  inputs). Reference J12 (train archive, 100-term cap, 40,000 fit jets): formula 81.31 % vs network 80.94 %, same call ≈ 90 %.
- Cumulative (a) equals single (a) to within 0.02 points at every stage: when stage k is replaced from physics, the earlier replacements only reach
  the output through the non-live units.
- Accuracy stays within 80.1–81.1 % under every single and cumulative substitution while the same-call rate is 92–97 %: the formulas change 3–8 % of
  the calls without changing the error rate. Mean-ablation of a whole stage from S on gives 19–20 % (chance).

## 4. What each unit looks at (top-3 terms by |coef × spread| with their share; thresholds in physical units: GeV, rad, counts; (b) terms in the previous stage's own units)
Full table for all 144 units: e11_mars_all.json → lookup. Units ordered by E1 cost.

P (physics only):
| unit | cost % | R² | top-3 terms |
|---|---|---|---|
| P15 | 8.1 | 1.000 | sum_pt (1.00) |
| P32 | 3.5 | 0.998 | n_pt_above_5 (0.11); z_top20_slots < 0.936 (0.11); n_pt_above_5 < 46 (0.10) |
| P40 | 3.1 | 0.795 | Σr[all] (0.22); eta_0 < 0.030 (0.09); orientation_deg < 36.5 (0.08) |
| P29 | 3.1 | 0.820 | Σr[ranks 1–63] (0.18); eta_0 < 0.021 (0.10); eta_2 < 0.0096 (0.09) |
| P56 | 3.0 | 0.999 | ΣpT[ranks 8–63] (0.82); pt_8 > 21.8 (0.04); n_particles < 64 (0.03) |
| P6 | 2.3 | 0.785 | Σr[1–63] (0.21); orientation_deg > −26.6 (0.08); eta_1 > −0.023 (0.06) |
| P43 | 2.2 | 0.811 | Σr[all] (0.22); orientation_deg < 18.0 (0.07); phi_0 > −0.022 (0.06) |
| P36 | 2.0 | 0.843 | phi_1 < 0.023 (0.21); phi_1 > −0.032 (0.13); absphi_1 > 0.027 (0.12) |
| P46 | 1.9 | 0.813 | Σr[1–63] (0.22); orientation_deg < 18.0 (0.11); phi_1 > −0.059 (0.07) |
| P30 | 1.6 | 0.796 | Σr[1–63] (0.16); phi_0 > −0.030 (0.12); phi_0 > −0.030 × eta_0 < 0.0047 (0.08) |
| P3 | 1.2 | 0.827 | Σr[all] (0.20); eta_0 > −0.022 × z_10 < 0.029 (0.07); orientation_deg > 9.3 (0.06) |
| P57 | 0.8 | 0.795 | Σr[all] (0.21); orientation_deg > −17.9 (0.09); eta_0 > −0.0093 (0.08) |

g (top 6 of 11):
| unit | cost % | R² (a) | top-3 (a) | R² (b) | top-2 (b) |
|---|---|---|---|---|---|
| g49 | 10.4 | 0.998 | sum_pt < 1263 (0.44); ΣpT[13–63] < 444 (0.09); ΣpT[top39] > 1226 (0.08) | 0.998 | P15 < 5.68 (0.37); P32 > 1.19 (0.15) |
| g4 | 4.3 | 0.994 | Σr[all] (0.62); orientation_deg > −35.8 (0.06); orientation_deg > −8.8 × n_pt_above_5 < 46 (0.03) | 0.990 | P29 (0.19); P46 (0.15) |
| g55 | 2.4 | 0.464 | phi_0 < 0.041 (0.13); phi_0 < −0.014 (0.09); sj2_mass1 < 64.2 (0.07) | 0.464 | P46 > 0.638 × P36 > 0.380 (0.28); P46 > 0.638 (0.19) |
| g39 | 2.1 | 0.368 | phi_0 > −0.041 (0.12); phi_9 > 0.088 (0.08); orientation_deg < −8.8 × sj2_mass1 < 64.2 (0.08) | 0.390 | P43 < 3.00 (0.12); P43 < 3.00 × P6 < 1.96 (0.12) |
| g5 | 1.8 | 0.394 | girth2_top10 < 0.0070 (0.10); girth2_top10 < 0.0070 × eta_3 < 0.034 (0.09); eta_0 < 0.054 (0.08) | 0.446 | P43 > 0.792 × P3 > 0.220 (0.22); P43 > 0.792 × P56 > 5.74 (0.13) |
| g63 | 1.4 | 0.796 | orientation_deg < 27.5 (0.12); orientation_deg < −35.8 (0.09); orientation_deg < 0.18 (0.09) | 0.659 | P36 < 4.07 × P29 < 4.40 (0.16); P36 < 4.07 (0.15) |

S (all 19):
| unit | cost % | R² (a) | top-3 (a) | R² (b) | top-2 (b) |
|---|---|---|---|---|---|
| S29 | 40.8 | 0.992 | ΣpT·r²[13–63] (0.18); Σr[top14] > 0.467 (0.10); Σ(r−0.2)₊[13–63] (0.10) | 0.992 | ΣpT·r²[13–63] (0.14); Σr[top14] > 0.467 (0.12) |
| S1 | 20.6 | 0.970 | Σ(r−0.2)₊[1–63] (0.64); n_dr_0.2–0.4 (0.14); ΣpT·r²[13–63] > 1.26 (0.04) | 0.971 | Σ(r−0.2)₊[1–63] (0.62); n_dr_0.2–0.4 (0.14) |
| S0 | 12.2 | 0.994 | sum_pt < 1263 (0.53); ΣpT[top39] > 1226 (0.11); log_sum_pt (0.08) | 0.998 | g49 > 6.10 × pt_14 < 26.2 (0.17); log_sum_pt (0.12) |
| S50 | 12.0 | 0.981 | ΣpT·r²[top26] > 0.49 (0.31); ΣpT·r²[top39] > 2.27 (0.14); ΣpT·r²[top3] > 1.55 (0.11) | 0.982 | ΣpT·r²[top26] > 0.49 (0.34); ΣpT·r²[top3] > 1.55 (0.10) |
| S19 | 11.5 | 0.999 | N[13–63] (0.60); N(pT<2)[8–63] (0.10); n_pt_above_1 < 64 (0.09) | 0.999 | N[13–63] (0.60); N(pT<2)[8–63] (0.10) |
| S30 | 8.4 | 0.996 | Σr[all] < 10.3 (0.45); Σr[all] < 8.30 (0.11); Σr[all] > 10.3 (0.06) | 0.996 | Σr[all] < 10.3 (0.45); g4 (0.14) |
| S43 | 8.1 | 0.985 | sum_pt_top2 > 318 (0.17); ΣpT·r[top14] > 21.3 (0.14); Σr[top14] > 0.271 (0.12) | 0.985 | sum_pt_top2 > 318 (0.18); ΣpT·r[top14] > 21.3 (0.16) |
| S60 | 7.2 | 0.988 | Σr[top14] (0.22); Σr²[top14] < 0.298 (0.15); Σr[top14] < 1.71 (0.08) | 0.989 | Σr[top14] (0.24); Σr²[top14] (0.19) |
| S13 | 7.2 | 0.971 | ΣpT·r²[top11] > 2.71 (0.28); Σr[top11] > 0.645 (0.11); ΣpT·r²[top3] < 5.10 (0.08) | 0.971 | ΣpT·r²[top11] > 2.71 (0.31); Σr[top11] > 0.645 (0.11) |
| S17 | 5.5 | 0.991 | ΣpT·r[top11] > 10.4 (0.16); girth2_top2 < 0.0024 (0.11); sum_pt_top10 > 557 (0.06) | 0.991 | ΣpT·r[top11] > 10.4 (0.17); girth2_top2 < 0.0024 (0.10) |
| S56 | 5.1 | 0.896 | Σ(r−0.2)₊[13–63] (0.26); ΣpT·r²[13–63] > 2.81 (0.13); soft1_dr < 0.391 (0.06) | 0.900 | Σ(r−0.2)₊[13–63] (0.22); max_dr > 0.318 × soft1_pt < 2.24 (0.11) |
| S40 | 3.8 | 0.952 | ΣpT·r²[top14] > 2.64 (0.19); ΣpT·r²[top14] > 5.72 (0.11); ΣpT·r²[top26] > 3.44 (0.08) | 0.952 | same two (0.19, 0.11) |
| S31 | 3.5 | 0.535 | ΣpT·r[top3] > 12.3 (0.17); ΣpT·r[top3] < 14.5 (0.08); Σr[top3] > 0.087 (0.06) | 0.555 | e2 > 0.0105 (0.08); ΣpT·r²[13–63] < 4.39 (0.08) |
| S51 | 2.8 | 0.992 | sum_pt_top2 > 191 (0.16); sum_pt_top3 (0.13); ΣpT·r[top3] > 19.1 (0.07) | 0.990 | sum_pt_top2 > 191 (0.22); ΣpT·r[top3] > 19.1 (0.08) |
| S63 | 2.3 | 0.966 | Σr[top14] > 0.271 (0.14); psi_0.1 > 0.084 (0.10); Σ(r−0.2)₊[top14] (0.08) | 0.966 | Σr²[top14] > 0.0044 (0.18); Σ(r−0.2)₊[top14] (0.12) |
| S52 | 2.3 | 0.910 | Σ(r−0.2)₊[top14] (0.35); Σr²[top11] > 0.161 (0.07); z_dr_0.2–0.4 > 0.051 × mass_top10 > 24.1 (0.06) | 0.913 | Σ(r−0.2)₊[top14] (0.34); Σr²[top11] > 0.127 (0.06) |
| S62 | 2.2 | 0.956 | Σr²[8–63] (0.30); n_dr_0.2–0.4 (0.23); n_particles > 54 (0.07) | 0.956 | same (0.30, 0.23) |
| S58 | 2.0 | 0.668 | planar_flow > 0.18 (0.24); planar_flow > 0.18 × mean_phi2 < 0.012 (0.12); orientation_deg > −45 (0.10) | 0.675 | planar_flow > 0.18 (0.22); N2 < 0.336 (0.10) |
| S20 | 1.5 | 0.993 | pt_0 > 253 (0.25); pt_0 > 175 × girth2_top2 < 0.0095 (0.19); pt_0 > 253 × pt1_dr01 < 22.8 (0.09) | 0.992 | pt_0 > 232 × Σr²[top3] < 0.0018 (0.17); pt_0 > 175 × Σr²[top3] < 0.0018 (0.17) |

h4 (top 6 of 54) and h5 (top 6 of 32):
| unit | cost % | R² (a) | top-3 (a) | R² (b) | top-2 (b) |
|---|---|---|---|---|---|
| h4:49 | 19.7 | 0.991 | √(M0·M2) < 100.6 GeV (0.30); ΣpT·r²[all] < 9.85 (0.11); Σr[1–63] < 5.55 (0.10) | 0.989 | S29 < 12.4 × S13 < 2.61 (0.10); S30 > 5.80 (0.08) |
| h4:41 | 7.0 | 0.979 | n_pt_above_1 > 32 (0.12); n_particles > 41 (0.11); Σr[all] > 7.37 (0.10) | 0.989 | S19 < 3.71 (0.11); S19 < 7.08 × S60 > 0.880 (0.10) |
| h4:39 | 4.6 | 0.986 | sum_pt < 1052 (0.36); log_sum_pt < 6.958 (0.24); log_sum_pt < 6.942 (0.09) | 0.994 | S0 < 18.1 (0.18); S0 < 18.1 × S43 > 2.26 (0.14) |
| h4:47 | 4.4 | 0.983 | ΣpT·r²[all] < 9.85 (0.22); girth < 0.086 (0.13); ΣpT·r²[top26] < 8.15 (0.11) | 0.983 | S50 < 2.78 × S0 > 11.2 (0.21); S50 < 2.46 × S0 > 11.2 (0.17) |
| h4:51 | 4.3 | 0.975 | ΣpT·r²[top11] < 9.35 (0.14); girth2_top3 < 0.0080 (0.10); girth2_top5 < 0.011 (0.09) | 0.980 | S13 < 0.795 (0.19); S60 > 1.82 (0.19) |
| h4:6 | 4.2 | 0.986 | ΣpT·r²[all] (0.29); Σ(r−0.2)₊[8–63] (0.17); Σr[8–63] > 1.96 (0.12) | 0.989 | S1 (0.35); S30 < 16.7 (0.19) |
| h5:26 | 16.1 | 0.988 | ΣpT·r²[all] < 9.85 (0.19); mass < 100.6 (0.14); √(M0·M2) < 78.4 (0.11) | 0.988 | h4:49 < 2.19 (0.18); h4:49 < 2.57 × h4:39 > 0.039 (0.16) |
| h5:8 | 6.4 | 0.935 | Σr[top26] < 4.41 × Σr[13–63] > 1.20 (0.15); n_dr_0–0.05 > 10 × n_real_top30 > 29 (0.12); Σr[top14] < 1.99 (0.07) | 0.961 | h4:41 > 0.067 (0.30); h4:41 (0.16) |
| h5:11 | 6.3 | 0.990 | mass < 83.1 (0.31); mass_top50 < 86.0 × Σ(r−0.2)₊[13–63] > 0 (0.10); mass < 83.1 × Σ(r−0.2)₊[13–63] > 0.046 (0.09) | 0.989 | h4:49 > 1.72 (0.13); h4:60 > 0.449 (0.12) |
| h5:25 | 5.9 | 0.976 | log_sum_pt > 6.894 (0.20); log_sum_pt > 6.958 (0.16); log_sum_pt > 6.909 (0.11) | 0.992 | h4:3 < 4.18 (0.16); h4:3 < 4.18 × h4:39 < 2.23 (0.13) |
| h5:3 | 5.3 | 0.957 | √(M0·M2) > 54.1 (0.22); mass > 92.9 (0.18); ΣpT·r²[all] < 7.35 (0.08) | 0.974 | h4:49 > 3.52 (0.17); h4:49 > 0.363 (0.16) |
| h5:20 | 5.0 | 0.973 | mass < 92.9 (0.23); mass_over_sum_pt < 0.091 (0.21); girth2 < 0.0082 (0.12) | 0.974 | h4:47 > 0.466 (0.22); h4:49 > 0.96 × h4:1 < 0.337 (0.13) |

z (all 16):
| unit | cost % | R² (a) | top-3 (a) | R² (b) | top-2 (b) |
|---|---|---|---|---|---|
| z1 | 21.6 | 0.938 | log_sum_pt > 6.909 (0.32); sum_pt_top50 > 961 (0.13); log_sum_pt > 6.97 (0.11) | 0.970 | h5:8 > 0.315 (0.17); h5:25 > 0.199 (0.17) |
| z0 | 15.1 | 0.925 | mass < 79.7 (0.18); mass < 91.1 (0.13); ΣpT·r²[all] < 7.35 (0.11) | 0.947 | h5:26 < 1.68 (0.24); h5:11 > 0.294 (0.12) |
| z10 | 6.9 | 0.868 | √(M0·M2) < 80.9 (0.09); mass > 163 (0.09); mass_top50 < 86.0 (0.09) | 0.931 | h5:0 > 0.243 (0.14); h5:26 < 6.72 (0.12) |
| z9 | 4.8 | 0.947 | √(M0·M2) > 143 (0.13); ΣpT·r²[all] > 9.85 (0.09); ΣpT·r²[top39] < 6.17 (0.08) | 0.964 | h5:11 < 4.83 (0.09); h5:2 > 0.442 × h5:0 < 3.04 (0.08) |
| z8 | 4.3 | 0.973 | ΣpT·r²[all] > 1.24 (0.17); ΣpT·r²[all] < 9.85 (0.15); √(M0·M2) > 35.8 (0.11) | 0.987 | h5:29 > 0.454 (0.20); h5:24 > 0.049 (0.09) |
| z13 | 4.1 | 0.898 | mass > 143 (0.20); sum_pt < 1012 (0.08); mass_top50 > 138 (0.07) | 0.948 | h5:24 < 5.15 × h5:0 < 3.04 (0.29); h5:14 (0.09) |
| z7 | 3.4 | 0.892 | mass < 91.1 (0.16); (m/ΣpT)² < 0.0079 (0.10); mass_top50 < 92.2 (0.10) | 0.924 | h5:17 > 1.72 × h5:13 < 1.26 (0.14); h5:3 > 1.97 × h5:17 < 2.75 (0.10) |
| z4 | 3.0 | 0.891 | mass < 83.1 (0.29); mass < 74.7 (0.11); √(M0·M2) < 100.6 (0.09) | 0.936 | h5:10 > 0.210 (0.16); h5:30 < 0.619 (0.09) |
| z5 | 2.8 | 0.898 | log_sum_pt > 6.909 (0.20); sum_pt_top50 > 936 (0.16); log_sum_pt > 6.812 (0.07) | 0.956 | h5:25 > 0.071 × h5:0 > 1.63 (0.16); h5:25 > 0.071 × h5:0 > 3.04 (0.14) |
| z11 | 2.5 | 0.886 | √(M0·M2) < 93.0 (0.19); √(M0·M2) < 80.9 (0.17); ΣpT·r[top39] < 74.5 (0.08) | 0.928 | h5:26 < 2.98 × h5:19 > 0 (0.13); h5:17 > 0.630 (0.09) |
| z6 | 2.4 | 0.832 | ΣpT·r²[all] < 9.85 (0.18); ΣpT·r²[all] < 8.35 (0.17); √(M0·M2) < 120 (0.14) | 0.896 | h5:11 > 1.63 (0.28); h5:15 > 0.577 (0.07) |
| z12 | 2.4 | 0.945 | mass < 87.5 (0.25); mass < 87.5 × psi_0.3 > 0.931 (0.22); mass < 83.1 (0.20) | 0.958 | h5:20 > 1.70 (0.22); h5:3 < 1.97 × h5:19 > 0 (0.10) |
| z15 | 2.0 | 0.720 | ΣpT·r[all] < 79.3 (0.11); girth < 0.062 (0.10); sum_pt < 1002 (0.08) | 0.825 | h5:1 > 0.166 × h5:17 < 3.37 (0.18); h5:7 > 0.408 (0.13) |
| z3 | 1.1 | 0.897 | Σr[all] < 5.18 (0.16); Σr²[8–63] < 0.539 × z_top50_slots > 0.971 (0.13); Σr[all] < 5.18 × Σr[top26] < 1.81 (0.09) | 0.923 | h5:31 > 0.793 (0.14); h5:17 > 3.37 (0.12) |
| z14 | 0.7 | 0.808 | mass < 91.1 (0.21); √(M0·M2) < 80.9 (0.15); (m/ΣpT)² < 0.0082 (0.11) | 0.878 | h5:3 > 1.75 × h5:20 < 1.70 (0.15); h5:3 > 1.75 (0.14) |
| z2 | 0.4 | 0.906 | ΣpT·r²[top39] > 5.44 (0.13); mass > 92.9 (0.12); ΣpT·r²[all] > 5.89 (0.11) | 0.963 | h5:25 > 1.13 × h5:13 < 1.26 (0.21); h5:25 > 3.68 × h5:13 < 1.26 (0.13) |
Mass thresholds that recur across h5 and z (GeV): 74.7, 78.4, 79.7, 80.9, 83.1, 87.5, 91.1, 92.9, 93.0, 100.6, 143, 163 (E6's windows: W 75, Z 94–98, t 135–148).

## 5. Feature evolution — results/jedi/e11_evolution.png
Cell = % of the stage's formula weight (Σ over units of the unit's |coef × spread| shares, units weighted by E1 cost; products split half/half).
(a) each stage's own physics fit; (b) the nested program with the physics content propagated through the chain (a term on S29 counts as S29's
own family shares, and so on).
| family | P (a) / (b) | g | S | h4 | h5 | z |
|---|---|---|---|---|---|---|
| ΣpT / pT scale | 34 / 34 | 39 / 38 | 12 / 11 | 14 / 15 | 17 / 17 | 25 / 18 |
| multiplicity | 6 / 6 | 3 / 6 | 12 / 12 | 9 / 13 | 5 / 14 | 4 / 14 |
| radial moments (r, r², ΣpT·r², angular sums) | 47 / 47 | 37 / 45 | 70 / 70 | 59 / 65 | 48 / 63 | 38 / 61 |
| mass family | 0 / 0 | 3 / 0 | 3 / 3 | 13 / 2 | 21 / 2 | 28 / 2 |
| N-subjettiness, ECFs | 0 / 0 | 0 / 0 | 0 / 1 | 2 / 1 | 4 / 1 | 3 / 1 |
| subjet / prong | 0 / 0 | 1 / 0 | 0 / 0 | 1 / 0 | 1 / 0 | 0 / 0 |
| pT shares, other | 12 / 12 | 17 / 12 | 3 / 3 | 3 / 4 | 4 / 4 | 2 / 4 |
Equal-weight versions in e11_mars_all.json → evolution.a_equal / b_equal. Family assignment: interp/e11_mars_all.py family(); per-particle signed
angles (eta_i, phi_i, orientation) are counted under the radial/angular family.
- In (a) the mass family's share rises 0 → 3 → 3 → 13 → 21 → 28 % from P to z, and the first mass term appears at h4 (h4:49: √(M0·M2) < 100.6 GeV).
  In (b) the same units are written on S objects (S29, S13, S30, S0, …), so their physics content stays at the S-stage composition (radial 61–65 %,
  ΣpT 15–18 %, multiplicity 13–14 %, mass family 2 %).
- N-subjettiness / ECF / subjet quantities carry ≤ 4 % of the weight at any stage in either reading.

## 6. Controls
- Untrained network (same fitter, (a) inputs, units by activation spread): P median 0.803 (trained 0.817), g 0.773 (0.752); S 0.084 over the only
  2 units with any spread (trained 0.981), z 0.140 (trained 0.897). h4 / h5: UNTR_LINE
  The untrained P and g units are pooled sums of random cuts and are named as well as the trained ones (top terms Σr, Σr over rank windows).
- Shuffled targets (stage S): R² −0.001 … 0.000 for all 19 units; the fitter keeps 0–4 terms per unit (33 in total) at R² ≈ 0.

## 7. Supplement: half-plane sums added to the physics inputs (prefix e11hp)
HP_SECTION

## 8. Issues and deviations
- Everything is fitted and tested on the TEST file (fit / validation / test disjoint; 1,152 overlapping jets removed from the test sample). The
  260,000 test jets suffice for these sample sizes.
- Selection step on 6,000 fit jets instead of mars.py's 12,000 (2 Gi job limit; the coefficient refit uses all 20,000). Term cap 30 (mars.py 100).
- logs/e11.log was already in use by the E11-windows experiment (interp/e11_windows_jedi.py); this run's logs are logs/e11-*.log.
- The eval log contains a JAX "CUDA plugin configuration error" traceback at start-up: the cluster env probes CUDA before falling back to CPU
  (JAX_PLATFORMS=cpu); it does not affect the run.
- Pod peak memory reached the 2 Gi request in every job (no OOM; exit 0).
- (b) single substitution at h4 (formulas on the true S values) has 95.7 % same call but 79.59 % accuracy, 0.8 points below the network; cumulative
  (b) at h4 is 80.83 %. Recorded as measured; not investigated.
- The angular P units (and the g units fed by them) are the weak link of the chain: no library observable is a one-sided angular sum (section 7
  tests this directly).
