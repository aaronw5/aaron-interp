# H30: Lund-plane response map of quark–gluon taggers (claim D7, exploratory)

Script `interp/h30_qg_lund_map.py`. It has five steps: `eval` (GPU), `ref --part lund|efp` (CPU, venv-ef), `edchk`, `stats` and `fig`.

Results are in `/j-jepa-vol/aaron-interp/results/physics/`. The small files are copied to `results/physics/`:
- `h30_run.json`: all statistics.
- `h30_lund.npz` / `h30_lund.json`: Lund density tables and LR(N) fits.
- `h30_models.json`: checkpoints used.
- Figures `h30_lund_map.png` and `h30_lund_fit.png`.
- Per-jet files, on the volume only: `h30_logits_{py,hw}.npz`, `h30_edits_{py,hw}.npz`, `h30_ref_{py,hw}.npz`, `h30_efp_{py,hw}.npz`.

Jobs (logs in `logs/d-h30-*.log`):
- Smoke runs, 100 + 100 jets: `d-h30-smk-ev`, `d-h30-smk-ref` and `d-h30-smk-st2`.
- `d-h30-run2`: GPU evaluation, 20 min. The first submission, `d-h30-run`, failed at pod start and left no log; it was resubmitted unchanged.
- `d-h30-ref`: Lund references, 4 CPU, 18 min.
- `d-h30-efp`: GBDT(EFP+N), 4 CPU, 14 min.
- `d-h30-edchk`: checks that the CPU environment rebuilds the GPU step's edits. Maximum parameter difference 1.6e-13.
- `d-h30-stat`: statistics.
- `d-h30-fig`: figures.

## Conclusion
We took one soft particle in each quark or gluon jet and moved it to different angles and transverse momenta relative to the jet axis. We then measured how far each tagger's output moved toward "gluon" compared with a rotation of the same particle around the axis.

ParT and ParticleNet move toward gluon when the particle gets a larger transverse momentum relative to the axis (k_t) or a smaller angle. This follows the gluon/quark emission-density ratio, and also pT^D, better than girth, which depends only on k_t here. For the hardest wide-angle emissions (k_t ≈ 7.5 GeV), however, the networks move toward quark, while every classical reference moves toward gluon. No single one of the three quantities describes the networks' maps.

The Herwig-trained networks differ from the Pythia-trained ones mostly at k_t above 1 GeV, not in the non-perturbative region below 1 GeV. The prediction that generator differences would sit at low k_t is therefore not met.

Still open:
- Whether the hard-emission cells, which take up to half of the leading particle's pT, are realistic jets. No C2ST was run.
- Whether "Lund density" and pT^D can be told apart with this grid, since the two regressors are correlated (r = −0.64).

## Procedure
**Sample.** Quark–gluon test jets, pT 500–550 GeV, |y| < 1.7, R = 0.4:
- Pythia 8.226: `interp.data_qg` test split.
- Herwig 7.1: Zenodo 3066475 files 18–19.

Each generator contributes 2000 quark + 2000 gluon jets: random rows, seed 3001, via `interp.h14_qg_generator.load_sample`.

**Models.** The logit is ℓ = log p_g − log p_q. Checkpoints are the same as in H24:
- Pythia-trained: ParT_kin (epoch 14) and PN_kin (epoch 17).
- PFN ×2 (H2 energyflow port, H5 torch).
- EFN ×2 (IRC-safe).
- ParT_kin with permuted weights (null, c3).
- Herwig-trained: ParT_kin (epoch 12) and PN_kin (epoch 19).

**Selected particle.** In each jet, one particle with z = pT_i/ΣpT in [0.005, 0.02] is chosen at random, never the hardest. Such a particle exists in 3991 of 4000 jets per generator. Mean z0 = 0.011. On Pythia the median source ΔR0 = 0.078 and the median kt0 = pT_i·ΔR0 = 0.40 GeV.

**Main edit m(r,k).** The particle is moved to a grid cell:
- ΔR centres 0.026, 0.042, 0.070, 0.115, 0.189, 0.312: the centres of 6 log bins in [0.02, 0.4].
- k_t centres 0.40, 0.72, 1.29, 2.32, 4.17, 7.47 GeV: the centres of 6 log bins in [0.3, 10].

How the edit is built:
- The particle's pT is set to k_t/ΔR. The hardest particle gives or takes the difference, so ΣpT and N are fixed.
- The particle is placed at ΔR from the edited jet's own E-scheme axis, at its original azimuth. The axis is iterated 6 times; the achieved ΔR equals the cell centre to 4 digits.
- A cell is skipped for a jet if the new z > 0.10 or the hardest particle would lose more than 50 % of its pT.
- This leaves 27 cells with ≥ 200 jets. Nine small-ΔR, high-k_t cells are empty. In 3 of the 27 cells only 2250–3270 jets pass.

**Controls.**
- c1: the same particle, pT unchanged, rotated around the axis at its own ΔR0. The rotation angle is chosen so that the chord equals the main edit's displacement (π if the displacement is larger than 2ΔR0). Radial observables stay fixed: Δgirth (c1 − base) is 0 to −4e-5.
- c2: the same particle, pT unchanged, moved the same distance in a random direction, staying within ΔR < 0.4.
- c3: the permuted-weight ParT.
- c4: the EFN.

Every edited jet is rebuilt about its own axis (`h5_qg_counting.from_list` → `kinematics`).

**References.** Each reference is fitted per generator on that generator's training split:
- LR(Lund): C/A clustering of all constituents in numpy, followed by primary declustering along the harder branch, on 20k + 20k training jets. ln ρ_g/ρ_q is tabulated in 15 × 17 bins of (ln 1/ΔR, ln k_t). The reference is LR = Σ over primary emissions of ln ρ_g/ρ_q(bin). This is the Poisson-process likelihood ratio without its constant term, so its accuracy at threshold 0 is not meaningful; its AUC is.
- LR(girth) and LR(pT^D): 40-bin histograms. LR(N): negative binomial; it is 0 for every edit, since N is fixed.
- GBDT(EFP d ≤ 5 + N): 15k + 15k training jets, evaluated on 500 + 500 sample jets for base, main and c1.

**Metrics.**
- Per cell: Δℓ(main) − Δℓ(c1), averaged over jets. 95 % bootstrap CI from 1000 resamples of jets. Sign-flip permutation p with 2000 sign vectors. BH correction over the 27 cells of each map.
- Per-model fit: the 27 cell means of the map are fitted to the cell means of three regressors, each standardised to 1 SD across cells, plus an intercept:
  - a: Δ ln ρ_g/ρ_q = table(destination) − table(source), using the table of the generator the model was trained on.
  - b: Δgirth.
  - c: ΔpT^D (main − c1).
  - CIs come from refitting the 1000 jet-bootstrap cell means. Each coefficient's p-value comes from permuting that regressor across cells (5000 permutations). Holm correction is applied over models within each (generator, coefficient) family.
- Herwig- minus Pythia-trained, paired on the same jets, per cell, with BH:
  - The contrast mean|Δ| over low-k_t cells (k_t centres 0.40 and 0.72 GeV) minus mean|Δ| over the other cells, with a bootstrap CI and a permutation over cell labels (one-sided, low > high).
  - The correlation across cells with the generators' own Δ ln ρ_g/ρ_q difference.

**Predictions on this grid.**
- Girth: about Δk_t/pT_J, so it depends on k_t only and is flat in ΔR at fixed k_t.
- pT^D: depends on z = k_t/(ΔR·pT_J), so it rises toward small ΔR at fixed k_t.
- Lund ratio: the measured ln ρ_g/ρ_q is about 0 at k_t ≈ 0.4 GeV and 0.45–0.83 at k_t > 2 GeV in Pythia (ln C_A/C_F = 0.81). Herwig is lower, 0.28–0.67. So the Lund ratio predicts a response that rises in k_t up to about 2 GeV and then flattens, with mild ΔR dependence.

## Checks: unedited models on these samples (AUC / accuracy)
| model | Pythia jets | Herwig jets | known (H24, 8000 jets) |
|---|---|---|---|
| ParT_kin, Pythia-trained | 0.904 / 0.829 | 0.817 / 0.724 | 0.904 / 0.833; 0.815 / 0.721 |
| PN_kin, Pythia-trained | 0.903 / 0.830 | 0.814 / 0.718 | 0.905 / 0.832; 0.817 / 0.720 |
| ParT_kin, Herwig-trained | 0.886 / 0.790 | 0.836 / 0.762 | 0.888 / 0.803; 0.836 / 0.760 |
| PN_kin, Herwig-trained | 0.881 / 0.788 | 0.837 / 0.760 | 0.884 / 0.796; 0.838 / 0.762 |
| PFN (H2) / PFN (H5) | 0.894 / 0.887 | 0.816 / 0.811 | 0.896; 0.818 |
| EFN (H2) / EFN (H5) | 0.883 / 0.866 | 0.804 / 0.798 | 0.886; 0.809 |
| ParT_kin, permuted | 0.548 / 0.499 | 0.520 / 0.499 | 0.539; 0.510 |
| LR(Lund), own generator | 0.825 | 0.776 | – |
| LR(N) / LR(pT^D) / LR(girth), Pythia | 0.839 / 0.804 / 0.730 | 0.793 / 0.751 / 0.720 | – |
| GBDT(EFP+N), own generator | 0.860 / 0.770 | 0.797 / 0.728 | – |

Mean number of primary C/A emissions: Pythia quark 12.7, gluon 14.5; Herwig quark 12.3, gluon 13.7.

## Numbers
**Maps, Pythia jets, main − c1 in Δℓ** (rows k_t 0.40 → 7.47 GeV; columns ΔR 0.026 → 0.312; "–" = cell skipped):

| k_t \ ΔR | 0.026 | 0.042 | 0.070 | 0.115 | 0.189 | 0.312 |
|---|---|---|---|---|---|---|
| ParT_kin (Py-trained), 0.40 | +0.036 | +0.027 | +0.026 | +0.045 | +0.062 | +0.046 |
| 0.72 | +0.109 | +0.086 | +0.085 | +0.094 | +0.122 | +0.116 |
| 1.29 | +0.285 | +0.150 | +0.114 | +0.116 | +0.160 | +0.158 |
| 2.32 | – | – | +0.182 | +0.118 | +0.141 | +0.172 |
| 4.17 | – | – | – | +0.119 | +0.021 | +0.136 |
| 7.47 | – | – | – | – | −0.276 | −0.120 |
| ParT_kin (Hw-trained), 0.40 / 1.29 / 7.47 | +0.029 / +0.373 | +0.020 / +0.246 | +0.022 / +0.193 | +0.021 / +0.158 | +0.012 / +0.135 / −0.469 | −0.018 / +0.090 / −0.366 |
| LR(Lund, Py), 0.40 / 1.29 / 7.47 | +0.081 / +0.488 | +0.040 / +0.381 | +0.019 / +0.293 | +0.016 / +0.244 | −0.003 / +0.224 / +0.452 | +0.019 / +0.250 / +0.402 |
| LR(girth, Py), 1.29 / 7.47 | +0.150 | +0.124 | +0.125 | +0.127 | +0.131 / +0.936 | +0.137 / +0.785 |
| LR(pT^D, Py), 1.29 | +0.553 | +0.332 | +0.187 | +0.088 | +0.022 | −0.023 |

Selected Pythia-jet cells, with CIs and the class split:

| cell | ParT_kin (Py-trained) [95 % CI] | quark jets | gluon jets |
|---|---|---|---|
| ΔR 0.026, k_t 1.29 GeV | +0.285 [0.263, 0.307] | +0.342 | +0.181 |
| ΔR 0.189, k_t 7.47 GeV | −0.276 [−0.312, −0.243] | +0.056 | −0.734 |
| ΔR 0.312, k_t 7.47 GeV | −0.120 [−0.145, −0.097] | +0.240 | −0.488 |

- EFN: all 27 cells positive, +0.02 to +0.47, and rising with k_t up to 7.47 GeV (+0.42).
- PFN (H2): +0.02 to +0.32.

Significance:
- Cells at BH q < 0.05: ParT_kin 27/27 on Pythia and 24/27 on Herwig; Herwig-trained ParT_kin 26/27 and 27/27.

Controls:
- c1 alone (rotation, Δℓ vs base): ParT_kin +0.006 to +0.043; EFN +0.008 to +0.062.
- c2 (random relocation of equal size): ParT_kin +0.016 to +0.244; EFN up to +0.325.
- Permuted ParT: |map| ≤ 0.025.

**Fits on Pythia jets.** Standardised coefficients a (Δ ln ρ_g/ρ_q), b (Δgirth) and c (ΔpT^D), with 95 % CI and Holm-adjusted p:

| scorer | a | b | c | R² | single-regressor R² (Lund, girth, pT^D) |
|---|---|---|---|---|---|
| ParT_kin (Py) | +0.079 [0.074, 0.083], Holm 0.004 | −0.105 [−0.113, −0.097], 0.004 | +0.008 [−0.001, 0.016], 1.0 | 0.75 | 0.02, 0.39, 0.00 |
| PN_kin (Py) | +0.070, 0.004 | −0.080, 0.004 | +0.011, 1.0 | 0.70 | 0.04, 0.29, 0.00 |
| ParT_kin (Hw) | +0.115, 0.004 | −0.177, 0.004 | −0.005, 1.0 | 0.86 | 0.02, 0.49, 0.00 |
| PN_kin (Hw) | +0.159, 0.004 | −0.275, 0.004 | +0.045, 0.75 | 0.83 | 0.00, 0.64, 0.04 |
| PFN (H2) | +0.025, 0.004 | −0.015, 0.16 | −0.046, 0.004 | 0.82 | 0.50, 0.09, 0.74 |
| PFN (H5) | +0.010, 0.65 | +0.024, 0.004 | −0.050, 0.004 | 0.84 | 0.51, 0.44, 0.73 |
| EFN (H2) | +0.080, 0.004 | +0.048, 0.004 | −0.004, 1.0 | 0.92 | 0.80, 0.68, 0.34 |
| LR(Lund, Py) | +0.102 | +0.013 | −0.052 | 0.98 | 0.90, 0.39, 0.67 |
| LR(girth, Py) | +0.043 | +0.219 | −0.018 | 0.97 | 0.49, 0.93, 0.29 |
| LR(pT^D, Py) | +0.002 | +0.001 | −0.164 | 1.00 | 0.41, 0.19, 1.00 |
| GBDT(EFP+N, Py) | +0.035 | −0.011 | −0.004 | 0.60 | 0.55, 0.07, 0.26 |
| permuted ParT | 0.000 | −0.001 | +0.005 | 0.71 | – |

- Correlations between the regressors across cells: Lund–girth +0.58, Lund–pT^D −0.64, girth–pT^D −0.43.
- On Herwig jets the coefficients have the same signs. For ParT_kin (Py-trained): a +0.093, b −0.058, c +0.041.
- Using the other generator's density table changes R² by −0.14 to +0.03.

**Herwig-trained − Pythia-trained, paired on the same jets.**

| pair | jets | cells at BH < 0.05 | mean abs. difference, k_t < 1 GeV | mean abs. difference, k_t > 1 GeV | contrast (low − high) [95 % CI] | permutation p (low > high) |
|---|---|---|---|---|---|---|
| ParT | Pythia | 24/27 | 0.030 | 0.081 | −0.051 [−0.056, −0.046] | 0.998 |
| ParT | Herwig | 26/27 | 0.050 | 0.099 | −0.049 | 0.988 |
| PN | Pythia | 24/27 | 0.032 | 0.173 | −0.140 | 1.0 |
| PN | Herwig | 26/27 | 0.056 | 0.185 | −0.130 | 0.996 |

- The largest single differences are in the ΔR 0.312, k_t 7.47 GeV cell: −0.245 (ParT, Pythia jets) and −0.286 (ParT, Herwig jets).
- Correlation across cells with the generators' Δ ln ρ_g/ρ_q difference: +0.16 to +0.36 (permutation p 0.06–0.43).

## Reading
**Low k_t (0.4–0.7 GeV).**
- ParT and PN maps are nearly flat in ΔR (spread ≤ 0.04) and positive: +0.03 to +0.12.
- In these cells LR(girth) is about 0 and LR(pT^D) is negative at large ΔR. The Lund-table regressor is also negative here (−0.13 to −0.16), because the source particle sits at k_t ≈ 0.4 GeV.

**k_t = 1.29 GeV, varying ΔR.**
- The nets rise toward small ΔR. Pythia-trained ParT goes from +0.16 to +0.29, Herwig-trained ParT from +0.09 to +0.37.
- LR(Lund) and LR(pT^D) have the same shape.
- LR(girth) is flat (+0.12 to +0.15).
- This excludes girth as the sole variable (H0) for these cells.

**Highest k_t (4.2–7.5 GeV, z = 0.02–0.10).**
- ParT, PN and both Herwig-trained nets fall and turn negative: −0.12 to −0.47. The drop comes from gluon jets (Pythia-trained ParT on gluon jets: −0.73).
- EFN, PFN, LR(Lund), LR(girth) and LR(pT^D) all stay positive.
- This region gives the negative girth coefficient in the fits. With it, the three-regressor fit reaches R² 0.70–0.86, while Lund alone gives 0.02 and girth alone gives 0.29–0.64.
- The fitted coefficient pattern (a > 0, b < 0) describes cell means. It does not identify a single variable, and the regressors are correlated.

**Other models.**
- EFN follows Lund and girth (a and b both positive, R² 0.92–0.95).
- PFN follows pT^D (c < 0, single-regressor R² 0.73–0.74).
- The permuted ParT gives |map| ≤ 0.025.

**Generator dependence.**
- The H1 expectation that Herwig- and Pythia-trained maps differ mainly at k_t < 1 GeV is not met. The differences are 2–5 × larger at k_t > 1 GeV, mostly in the wide-angle 7.5 GeV cells.
- The prediction of the girth alternative (identical maps across generators) is also excluded: 24–26 of 27 cells differ at BH < 0.05.

## Issues and limits
- **Hard emissions.** The k_t ≥ 4 GeV cells give the moved particle up to 10 % of the jet pT and take up to half of the hardest particle's pT. They are not single soft emissions, and no C2ST was run on the edited jets.
- **k_t definition.** k_t is defined relative to the jet axis, not to the harder C/A branch. The Lund-table regressor looks up the moved particle's axis coordinates, while LR(Lund) reclusters the whole jet. The two agree in shape (LR(Lund)'s fit gives a single-regressor Lund R² of 0.90).
- **Not separated.** Lund density and pT^D are correlated across cells (r = −0.64) and are not separated by this grid. A grid at fixed z (changing ΔR and k_t together) would separate them.
- **Plan wording.** The plan stated H0 as "flat in k_t at fixed z·ΔR"; on this grid, girth is a function of k_t alone. The predictions above are stated for the edit as run.
- **Seeds.** There is one training run per generator for ParT and PN (as in H24), so the Herwig − Pythia map difference is not compared with a seed spread.
- **Explainer.** An explainer entry was added in `site/explain.py` (`H30_EXPLAINER`, Paper D list).
