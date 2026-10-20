# P26 — Sophon (ParT trained on JetClass-II, 188 classes): is the axis-relative mass response still there? (claims C3s, P20)

Status: **done** (exploratory). Jobs `anrunw-ai-c-p26-run2` (final; identical numbers to `c-p26-run`, which only lacked the ONNX check), `c-p26-smoke`, `c-p26-plot`. Script `interp/p26_sophon_axis.py`; network code `interp/sophon_part.py`; weight download `interp/p26_fetch.sh`; figure `interp/p26_plot.py`. Results `results/part/p26_sophon_axis_run.json`, figure `results/part/p26_sophon_axis.png`.

## Conclusion

Conclusion: Sophon is the released Particle Transformer trained by its authors on the much larger JetClass-II dataset (188 classes). Its W-versus-Z output barely moves when the jet mass is really increased by 12 %: on W jets the change is 1.6 % of the gap between typical Z and W scores, against 28 % for the JetClass ParT. So Sophon's output has no mass response for the axis test to measure. A linear mass read-out of its internal class token responds to the axis shift (invariant mass unchanged) only 0.13–0.25 as much as to a real mass change, against 0.78–0.97 for ParT_full and 0.56–0.84 for an untrained network. The pre-registered thesis is not supported for Sophon's output, and the alternative's condition that the dilation response stays is not met. Sophon also differs from ParT in its mass-decorrelated training, its normalised four-vector inputs and its flavour-based classes, so this test cannot tell whether data scale caused the difference.

## Model and released files (checked online first)

- Released files: `jet-universe/sophon` on Hugging Face and GitHub contain `models/JetClassII_Sophon/model.pt` (a PyTorch state dict, 237 tensors), `model.onnx`, the network file `networks/example_ParticleTransformer_sophon.py`, the data config `data/JetClassII/JetClassII_full.yaml` and `train_sophon.sh`. Both weight files were downloaded inside the job into `/j-jepa-vol/aaron-interp/models/sophon` (sha256 model.pt `cc7c33b5…`, model.onnx `f54e9fd5…`). Nothing was trained.
- Architecture: the README says Sophon was trained with weaver-core branch `dev/custom_train_eval`. Its `ParticleTransformer.py` was copied verbatim into `interp/sophon_part.py`; it differs from the 0.4.17 version used for the JetClass ParT. The configuration comes from the network file and `train_sophon.sh`: 8 particle blocks and 2 class blocks (not 6 + 2 as the task said), embed (128, 512, 128), pair embed (64, 64, 64), pre-activation pair embedding, fc 128 → 512 → 188. That makes **2.305 M parameters**.
- Inputs follow the yaml: 17 per-particle features. They are log(pT·500/jet_pT), log(E·500/jet_pT), log pT_rel, log E_rel, ΔR, charge, 5 PID flags, tanh d0, d0err, tanh dz, dzerr, Δη and Δφ. The four-vectors are scaled by 500/jet_pT, padding is wrap padding, and the model runs in fp32. JetClass-II stores Δη with the same sign(η_jet) convention as JetClass, checked on 2,000 jets (max deviation 2·10⁻⁷), so `part_core.build` serves both datasets.
- **Implementation check**: the released ONNX export (batch size fixed to 1; output is softmax(188) plus the 128-d token) was run on 200 sample-A jets with the same inputs. The maximum |Δ softmax| against our PyTorch run was **6.0·10⁻⁶**, and the top-1 agreement was 100 %.
- Class map: the Z group is X→bb, cc, ss (labels 0–2). The W group is X→bc, cs, bq, cq, sq (labels 4–8). X→qq (u/d, label 3) is produced by both W→ud and Z→uu/dd, so it is left out of both groups. The QCD group is labels 161–187. Group score = log Σ softmax. Read-out d = log p(Z group) − log p(W group); for ParT, d = logit_Z − logit_W. The mapped call is the argmax over the (QCD, Z, W) group scores.

## Samples

- **A** (JetClass): test_20M/{ZJetsToNuNu, ZToQQ, WToQQ}_112.root, 5,000 random rows per class (default_rng(2601)), N = 15,000, rebuilt unedited. 75.7 % of these jets pass Sophon's training selection (200 < pT < 2500, 20 < m_SD < 500 GeV). Median m_inv by class: QCD 69.6, Z 92.4, W 83.7 GeV.
- **B** (JetClass-II test shards, already on the volume at `/j-jepa-vol/JetClass2/Pythia/data`, read in place): Res2P_0250–0253 and QCD_0350. Selection: 500 < pT < 1000 and 20 < m_SD < 500 GeV. Z-like truth is X→bb/cc/ss and W-like truth is X→cs/cq/sq, both with generator resonance mass 70 < m_X < 100 GeV. QCD is any label 161–187. Up to 2,000 jets per group were taken (default_rng(2602)). The four shards held only 1,496 W-like jets, so N = 5,496 (QCD 2,000 / Z 2,000 / W 1,496).
- Reference reproduction: on the P1 sample, ParT_full 10-class accuracy is **0.8593** (P1: 0.859) and ParT_kin **0.7447** (P1: 0.745).

## Edits and controls (as P1 / P1v / P22)

| edit | what changes | Δlog m_inv | Δlog m_A (A) |
|---|---|---|---|
| φ + 0.06 | every particle's Δφ + 0.06 rad about the stored axis | 0 (−5·10⁻⁸) | +0.126 |
| matched dilation | (Δη, Δφ)·k_j, with k_j giving the same Δlog m_A as φ | +0.121 | +0.125 |
| rot (null) | whole jet and axis rotated 0.06 rad in φ | 4·10⁻⁷ | 0 |
| φ, ΔR fed unedited | route through Δφ only | 0 | – |
| φ, Δφ fed unedited | route through ΔR only | 0 | – |
| WTA | Δη, Δφ, ΔR recomputed about the WTA axis (P1v code); four-vectors unchanged | 0 | +0.112 |
| translation control (X8v) | axis moved by the WTA offset length in a random direction | 0 | +0.113 |
| ΔR-keeping rotation control | every particle rotated about the stored axis by a random angle per jet (max ΔR change 2·10⁻⁷) | −0.0009 | 0 |

The median WTA offset is 0.064 rad (A) and 0.062 rad (B). The WTA axis coincides with the hardest particle in 65 % (A) and 62 % (B) of jets.

Statistics:
- R = per-jet least-squares slope through 0 of Δd(φ) on Δd(dilation), with a 2,000-sample bootstrap over jets.
- R_probe = mean Δprobe(φ) / mean Δprobe(dilation) on the test half. The probe is a ridge regression of log m_inv on the 128-d class token (P1v split, seed 1), with a 1,000-sample bootstrap. It is calibrated against the matched real change by construction.
- W → Z uses McNemar on the paired Z-call indicator. WTA versus control uses McNemar on the paired "call changed" indicator.
- Permutation tests (2,000) compare R between models on the same jets. Holm correction is applied over 21 tests.

## Numbers

Accuracy (Wilson 95 %):

| model | sample | mapped 3-class acc | W/Z only (sign of d) | resonance vs QCD |
|---|---|---|---|---|
| Sophon | A (JetClass) | 0.626 [0.618, 0.634]; in Sophon selection 0.597 | 0.656 | 0.831 |
| Sophon | B (JetClass-II) | 0.752 [0.741, 0.763] | 0.778 | 0.867 |
| ParT_full | A | 0.824 [0.818, 0.830] | 0.831 | 0.923 |
| ParT_full | B | 0.710 [0.698, 0.722] | 0.718 | 0.852 |
| ParT_kin | A | 0.766 | 0.757 | 0.912 |
| ParT_kin | B | 0.578 | 0.537 | 0.811 |

- Sophon on B: 188-class top-1 accuracy is 0.4995 (QCD 0.49, Z-like 0.67, W-like 0.28). Top-1 falls in the true major group (two-prong resonance or QCD) for 0.758.
- On A, the mean p(X→qq) is 0.19 on Z jets and 0.22 on W jets. 26 % of A's top-1 calls are X→qq (label 3), a class that is in neither group.
- The resonance-vs-QCD accuracy gap A − B is −3.6 pt, so the pre-set domain-shift flag (gap > 10 pt) is not raised. The mapped W/Z accuracy drops 12.6 pt from B to A, about the size of ParT's drop from A to B (−11.4 pt). Part of Sophon's drop comes from the qq decays that its labels do not split into W or Z.

Mass response of the output and R:

| model, sample | Z−W separation of d | Δd on W, dilation | Δd on W, φ | dil / separation | R [95 % CI] | rot null | R_res | R_probe [95 % CI] (probe R²) |
|---|---|---|---|---|---|---|---|---|
| Sophon, A | 1.94 [1.86, 2.03] | +0.032 [0.027, 0.037] | −0.097 [−0.107, −0.087] | 0.016 | −0.256 [−0.308, −0.207] | +0.004 | −0.21 | **0.125 [0.109, 0.140]** (0.76) |
| Sophon, B | 3.55 [3.41, 3.71] | −0.014 [−0.028, 0.000] | −0.069 [−0.084, −0.055] | −0.004 | −0.006 [−0.051, +0.033] | +0.003 | −0.11 | **0.252 [0.230, 0.273]** (0.78) |
| ParT_full, A | 4.81 | +1.34 [1.30, 1.38] | +1.08 [1.05, 1.12] | 0.278 | **0.801 [0.792, 0.810]** | −0.000 | 0.64 | 0.776 [0.756, 0.796] (0.89) |
| ParT_full, B | 4.15 | +0.51 | +0.41 | 0.123 | 0.759 [0.736, 0.780] | +0.000 | 0.71 | 0.971 [0.943, 0.996] (0.85) |
| ParT_kin, A | 1.83 | +0.98 | +0.76 | 0.535 | **0.734 [0.723, 0.745]** | +0.007 | 0.61 | 1.507 [1.469, 1.545] (0.92) |
| ParT_kin, B | 0.49 | +0.43 | +0.16 | 0.884 | 0.626 [0.596, 0.657] | +0.016 | 0.68 | 0.786 [0.753, 0.819] (0.89) |
| untrained Sophon arch., seed 100, A | −0.01 | −0.009 | −0.050 | – | 0.545 [0.513, 0.579] | −0.000 | 0.13 | 0.844 [0.821, 0.869] (0.26) |
| untrained, seed 101, A | −0.01 | −0.005 | +0.015 | – | 0.150 [0.120, 0.179] | −0.000 | −0.06 | 0.563 [0.542, 0.585] (0.28) |

The ParT_full and ParT_kin rows on sample A reproduce earlier values: R 0.79/0.72 (P11, P16, P22 smoke) and probe-free W → Z 11–12 → 31–33 %.

W → Z (true-W jets with mapped call Z), before → after φ + 0.06, with McNemar and Holm:
- Sophon A: 24.3 → 23.5 %, Δ −0.9 pt [−1.5, −0.3] (p 0.003, Holm 0.012).
- Sophon B: 20.9 → 19.7 %, Δ −1.2 pt [−2.2, −0.2] (p 0.020, Holm 0.040).
- ParT_full A: 11.5 → 32.9 %, Δ +21.4 pt [20.2, 22.5] (Holm 10⁻²⁶²).
- ParT_kin A: 17.9 → 41.7 %, Δ +23.8 pt.
- ParT_full B: 19.7 → 26.6 %; ParT_kin B: 27.8 → 34.9 %.
- Under the matched dilation: Sophon A 24.3 → 24.9 %; ParT_full A 11.5 → 37.1 %.

Routes for the φ response (slope of the partial-edit Δd on the full-edit Δd):
- ParT_full A: ΔR held 0.045, Δφ held 0.984. The response goes through the per-particle ΔR input, as in P1v.
- ParT_kin A: 0.53 / 0.44.
- Sophon A: 0.55 / 0.40. Sophon's φ response is itself small and of the opposite sign to its dilation response, so these shares describe a different, small response.

Axis convention. Top-1 calls changed (188 classes for Sophon, 10 for ParT); mapped-3 in parentheses; McNemar of WTA against each control:

| model, sample | WTA | translation, same length | ΔR-keeping rotation | WTA vs translation p (Holm) | WTA vs rotation p (Holm) |
|---|---|---|---|---|---|
| Sophon, A | 7.3 % (5.4 %) | 6.6 % (5.0 %) | 16.0 % (10.1 %) | 0.004 (0.012) | 8·10⁻¹⁵⁹ |
| Sophon, B | 6.8 % (3.0 %) | 6.1 % (2.9 %) | 18.3 % (8.8 %) | 0.036 (0.040) | 6·10⁻⁹⁰ |
| ParT_full, A | 21.8 % (15.6 %) | 28.2 % (19.4 %) | 8.4 % (5.5 %) | 2·10⁻¹²⁶ | 10⁻²⁸¹ |
| ParT_full, B | 18.2 % (11.7 %) | 24.7 % (15.5 %) | 9.7 % (6.3 %) | 4·10⁻³⁹ | 7·10⁻⁴⁷ |
| ParT_kin, A | 34.4 % (20.1 %) | 40.7 % (24.9 %) | 9.5 % (5.7 %) | 3·10⁻⁹⁶ | < 10⁻³⁰⁰ |
| untrained, seeds 100 / 101, A | 20.4 / 18.1 % | 19.2 / 17.4 % | 21.9 / 21.5 % | – | – |

The untrained top-1 changes come from near-uniform 188-way outputs; their mapped-3 changes are 0. ParT's WTA numbers reproduce D4 (P1v: 11.7 % mapped / 21.8 % top-1 for ParT_full here).

Permutation tests of R (sample A): Sophon − ParT_full = −1.06, p 5·10⁻⁴ (the floor at 2,000 permutations), Holm 0.0035. Sophon − ParT_kin = −0.99, Holm 0.0035. Sophon − untrained seed 100 = −0.80, Holm 0.0035.

## Hypotheses (pre-registered in the task)

- **H0** (invariant-mass reading): R = 0. **H1** (axis reading): R > 0.
- **Thesis** (a stored-axis transformer reads an axis moment at any training scale): R 0.5–1.0 and WTA ≥ ~10 % of calls changed.
  - ParT_full and ParT_kin on both samples meet it.
  - Sophon does not: R −0.26 (A) and −0.01 (B); WTA top-1 7.3 % / 6.8 %, mapped 5.4 % / 3.0 %.
- **Alternative** (scale and diversity remove it): R < 0.3 with the dilation response kept, and WTA ≈ rotation control.
  - For Sophon the dilation response is not kept: Δd(dil) on W jets is 1.6 % (A) and −0.4 % (B) of the Z−W separation, against 12–88 % for ParT.
  - Sophon's WTA change is 0.5 of the rotation control (16 %), not equal to it, and 1.1× the translation control.
  - The alternative as written is therefore not met either. R on Sophon's output is a ratio of two small numbers and is not used as evidence.
- **What the measurement does separate:**
  - Sophon's output scores have almost no response to a real 12 % mass change. This is consistent with its mass-decorrelated training: samples were reweighted to flat pT and m_SD in each major class.
  - Its class-token mass probe (R² 0.76–0.78) moves under the φ shift at 0.13 [0.11, 0.14] (A) and 0.25 [0.23, 0.27] (B) of its dilation movement. This lies below ParT_full (0.78 / 0.97), ParT_kin (1.51 / 0.79) and the untrained architecture (0.56 / 0.84); all bootstrap intervals are disjoint. No formal test between probe ratios was run.
  - Untrained-network probe ratios of 0.56–0.84 show that the architecture with these inputs starts from a largely axis-relative mass representation. Sophon's training moved it towards the invariant mass. ParT's training kept it axis-relative or raised it.
- **Excluded for Sophon:** the thesis (R and WTA criteria both fail on both samples). **Not decided:** whether the cause is data scale and diversity, mass decorrelation, or the 500/jet_pT-normalised four-vectors (the pair features ln kT and ln m² become pT-relative). A training run that changes one of these at a time would be needed (P20).

## Issues and caveats

- **Read-out definition.** No Sophon class is a W or a Z. The flavour-group read-out separates Z from W mainly through bb/cc/ss versus cs/cq/sq. W→ud and Z→uu/dd fall in X→qq, which is left out (26 % of A's top-1 calls). The output test therefore measures a flavour-defined W/Z score that its training made mass-independent.
- **Domain.** Sophon on JetClass reaches 62.6 % mapped (83.1 % resonance vs QCD), against 75.2 % / 86.7 % on JetClass-II jets of matched pT and m_X. The resonance-vs-QCD gap is small (3.6 pt). Sample B is flavour-selected while A is not, so the mapped accuracies are not like-for-like.
- **Sample B size.** The W-like group has 1,496 jets because only 4 shards were read.
- **Dropped criterion.** The "6 + 2 blocks" in the task was wrong: the released config has 8 + 2.
- **Debugging.** The ONNX check failed in run 1 for two reasons: the export takes batch size 1, and its output includes the 128-d embedding. Both were fixed in run 2 (one debug attempt). No other numbers changed.

## Explainer entry (for site/explain.py)

P26_EXPLAINER:
- **Question**: ParT reads a jet's mass from the angular spread about the stored jet axis, $m_A=\sqrt{M_0M_2}$ with $M_0=\sum p_T$, $M_2=\sum p_T\Delta R^2$, not from the invariant mass (claim C3). Is this still true for Sophon, the same architecture trained by its authors on JetClass-II (188 classes, ~100× more jets, wider pT and mass ranges)?
- **Background**: shifting every constituent by $\delta\phi=0.06$ rad about a fixed axis leaves $m_{\rm inv}$ unchanged but raises $M_2$ by $\delta\phi^2M_0$ (+12 % in $m_A$). A per-jet dilation of the angles by $k_j$ gives the same $\Delta\log m_A$ and a real mass change. The ratio $R=\Delta d(\phi)/\Delta d(\text{dil})$ is 0 for an invariant-mass reader and 1 for an $m_A$ reader. Sophon was trained mass-decorrelated: every class was reweighted to flat $p_T$ and $m_{SD}$.
- **Method**: released weights and code, checked against the released ONNX export (max $|\Delta p|=6\cdot10^{-6}$). There were 15,000 JetClass W/Z/QCD jets and 5,496 JetClass-II jets with $70<m_X<100$ GeV. Read-out $d=\log p(Z\text{-like})-\log p(W\text{-like})$ uses Sophon's flavour classes (bb, cc, ss vs cs, cq, sq). A ridge probe of $\log m_{\rm inv}$ was fitted on the 128-d class token. Inputs were also recomputed about the WTA axis, with a random-direction axis move of the same length and a $\Delta R$-keeping rotation as controls. The references were ParT_full/kin and the untrained architecture.
- **Result** (figure `results/part/p26_sophon_axis.png`): Sophon's output barely responds to a real 12 % mass increase (1.6 % of the Z−W gap, vs 28 % for ParT_full). The axis-shift test therefore has nothing to divide by at the output, and W → Z under the φ shift is 24.3 → 23.5 % (ParT_full 11.5 → 32.9 %). Inside the network, the mass probe responds to the φ shift at 0.13–0.25 of its response to the dilation, against 0.78–0.97 (ParT_full) and 0.56–0.84 (untrained). WTA inputs change 7 % of Sophon's top-1 calls, close to the random axis move (6.6 %) and below the rotation control (16 %).
- **Checks**: ParT reproduces its known accuracies (85.9 / 74.5 %) and R (0.80 / 0.73). The rotation null is ≈ 0 for all models. Sophon's accuracy on JetClass jets is 83 % resonance-vs-QCD, against 87 % on JetClass-II jets.
- **Why it matters**: the axis-relative mass reading is not a fixed property of the ParT architecture; a model trained differently (more and more varied data, mass decorrelation, pT-normalised four-vectors) has a mostly invariant mass representation and no mass-driven output. Which of these three differences matters is open (P20).
