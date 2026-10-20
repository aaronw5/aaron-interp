# H36: axis convention on Top Landscape and quark–gluon taggers, predicted from each model's translation response (claims D4, D4m)

- **Status:** done.
- **Job:** `anrunw-ai-d-h36-run`: GPU, 59 min, exit 0. Smoke runs: `d-h36-smoke`, `d-h36-smoke2`.
- **Script:** `interp/h36_axis_top_qg.py`.
- **Results (in `results/physics/`):**
  - `h36_axis_top_qg.json`
  - `h36_prereg.json`
  - `h36_perjet.npz`
  - `h36_axis.png`
- **Log:** `logs/d-h36-run.log`.

## Conclusion
We recomputed each tagger's inputs about the winner-take-all axis or the hardest-particle axis instead of the usual momentum-sum axis. This changes 3–25 % of the calls of the Top and quark–gluon taggers; LorentzNet, which reads only four-vectors, changes none. A prediction made in advance from each model's response to plain shifts of its inputs got the size right within a factor of about 0.4–1.3. It was within its interval for 14 of 24 quark–gluon network cells but only 1 of 8 top-tagging cells, where the networks changed fewer calls than predicted.

The prediction misses a random-direction shift of the same size by a similar factor, so the miss comes mostly from the prediction method, not from the axis itself. Shifting toward the hardest particle changed at most about 4 percentage points more calls than a random-direction shift. A weight-permuted (untrained) network shows a similar 1–2 point excess on quark–gluon jets, so this excess does not show that the trained networks read the axis in a learned way. One open question remains: why the top-tagging ParT changes fewer calls under the new axis than under a random shift.

## Procedure
**Samples** (random rows, never first-k):
- **Top Landscape test:** `h4_top_wmass._sample(6000, 3601)`, the same rows as `data_top.sample`.
- **QG Pythia test:** `h14_qg_generator.load_sample('py', 6000, 3602)`.
- **QG Herwig 7.1 test (files 18–19):** `load_sample('hw', 6000, 3602)`.
- **Split:** per class, a random 1,000 jets go to calibration and 5,000 to the main sample. Split rng: 36011 (Top), 36021 (Py), 36022 (Hw).

**Axes per jet:**
- **E:** the stored axis, i.e. the constituent four-momentum sum.
- **WTA:** C/A clustering of all constituents to one jet with winner-take-all $p_T$ recombination. The axis is the direction of the particle reached by following the harder branch.
- **HP:** the highest-$p_T$ constituent.
- WTA and HP coincide in 61 % of Top jets, 68 % of Pythia jets and 67 % of Herwig jets.

**Models.** Scores are log-odds: Top scores are top-positive; QG scores are gluon-positive, as in H24.
- **Top:** ParT and ParticleNet (the H4 snapshots of the H2 kin trainings); LorentzNet (released); PFN and EFN (H2); ParT with permuted weights (null).
- **QG:** every model is run on both generator samples:
  - Pythia-trained ParT_kin, PN_kin, PFN (`pfn_k`) and EFN (`efn_k`);
  - Herwig-trained ParT_kin and PN_kin (H14/H24). No Herwig-trained PFN or EFN exists.
  - ParT_kin with permuted weights (null);
  - released LorentzNet (four-vectors plus PID).
- **Classical, QG:**
  - LR(girth): a histogram LR of girth about the axis in use.
  - LR(N): multiplicity only.
  - Both are trained on 20k jets per class from the same generator's train split (seed 36031).
- **Classical, Top:** LR($m_{SD}$, $\tau_{32}$). Soft drop uses C/A and $\tau_N$ uses exclusive-$k_T$ axes, so no jet axis enters and its calls change by 0 % by construction (not recomputed).

**Why TN equals the axis recompute.** In all of these models, the axis enters only through $(\Delta\eta_i,\Delta\phi_i)$ and $\Delta R_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$. Recomputing about a new axis is therefore an exact rigid translation of the stored $(\Delta\eta,\Delta\phi)$ by $-(\mathrm{sign}(\eta_J)\,\delta\eta,\ \delta\phi)$. TN, the translation by the same per-jet offset, runs through a separate code path as an implementation check: $\max|l_{\rm axis}-l_{\rm TN}|$ is 0 to $5\times10^{-7}$ for every model. The test of H1 against H0 therefore uses two comparisons:
- observed calls changed against the translation-based prediction;
- the new axis against R, a random-direction translation of the same per-jet size.

**Step 1: calibration** (2,000 jets per sample):
- Rigid translations $(\Delta\eta,\Delta\phi)\mathrel{+}=r(\cos\theta_k,\sin\theta_k)$ with $\theta_k=k\pi/4$, $k=0..7$.
- $r\in\{2,5,10,20,50,100,200,400\}$ mrad. The planned 2–20 mrad grid was extended because Top WTA offsets have a median of 93 mrad.
- Recorded per model, class and $r$: the distribution of $\Delta l$.

**Step 2: pre-registered prediction.** For main jet $j$ with unedited score $l_{0j}$, class $y_j$ and offset size $r_j$:
- $p_j=P(\mathrm{sign}(l_{0j}+\Delta l)\neq\mathrm{sign}(l_{0j}))$, from the calibration $\Delta l$ of class $y_j$ at $r_j$ (empirical, interpolated in $\log r$).
- Predicted % of calls changed $=\langle p_j\rangle$.
- A secondary prediction, pred2 (added after the smoke run, before the full run), also conditions on the calibration quartile of $|l_0|$.
- Predictions were computed and logged before step 3 was scored for each dataset (log lines at 470 s, 1817 s and 2997 s; step 3 at 602 s, 1937 s and 3113 s). The JSON file was written after each dataset finished.

**Step 3: measurement** on 10,000 main jets per sample. Conditions: E, WTA, HP, TN_WTA, TN_HP, R_WTA, R_HP (R direction from rng 36041+).

**Metrics:**
- **Calls changed:** % of calls changed (call = sign $l$) with a 95 % bootstrap interval ($B=2000$).
- **Axis vs R:** the same for R. McNemar on the paired changed-indicators gives axis − R, with its interval from the discordant counts.
- **Performance:** accuracy and AUC before → after.
- **Observed / predicted:** with a 95 % interval from a joint bootstrap of main and calibration jets (500), and a two-sided bootstrap p for H1 (obs = pred).
- **Holm correction** over 52 cells (models × samples × axes) for each family.
- **Repeat floor:** an E2 repeat of the unedited scoring gives the GPU non-determinism floor.

## Checks: unedited models on the main samples (10,000 jets each; AUC / accuracy)
| model | this sample | known |
|---|---|---|
| Top ParT | 0.9842 / 93.5 % | 0.9858 published; 0.9867 / 94.3 % (H25, 6,000 jets) |
| Top PN | 0.9843 / 93.7 % | 0.9858 published; 0.9868 / 94.0 % (H25) |
| Top LorentzNet | 0.9852 / 93.7 % | 0.9868 published; 0.9879 (H25) |
| Top PFN / EFN | 0.9782 / 92.3 %; 0.9754 / 92.2 % | 0.9816 / 0.9783 (H25) |
| Top ParT permuted | 0.717 / 66.6 % | – |
| QG Py ParT_kin (Py jets / Hw jets) | 0.898 / 82.7 %; 0.809 / 71.8 % | 0.904 / 83.3 %; 0.815 / 72.1 % (H24) |
| QG Py PN_kin | 0.898 / 82.5 %; 0.810 / 71.6 % | 0.905; 0.817 (H24) |
| QG PFN / EFN (Py) | 0.889 / 82.0 %; 0.881 / 80.8 % (Py jets) | 0.896; 0.886 (H24) |
| QG Hw ParT_kin / PN_kin (Hw jets) | 0.831 / 75.5 %; 0.832 / 75.6 % | 0.836 / 76.0 %; 0.838 / 76.2 % (H24) |
| QG LorentzNet (Py / Hw jets) | 0.913 / 84.1 %; 0.799 / 69.7 % | – |
| QG ParT permuted | 0.545 / 49.9 % | 0.539 (H24) |
| LR(girth), LR(N), own generator (Py; Hw) | 0.744, 0.839; 0.713, 0.787 | LR(N) 0.846 / 0.790 (H24) |

All values are within about 0.006 AUC of the earlier samples. Implementation checks:
- **Determinism floor:** repeat scoring changes 0 calls in every model. Maximum logit jitter: LorentzNet $5.4\times10^{-3}$ (Top) and $6\times10^{-4}$ (QG); 0 for the other models.
- **TN identity:** axis − TN max $|\Delta l|\le5\times10^{-7}$.

## Axis offsets (main jets; median, p90 in mrad)
| sample | WTA − E | HP − E | WTA median by class |
|---|---|---|---|
| Top | 93, 239 | 101, 275 | top 157, QCD 40 |
| QG Pythia | 17, 72 | 17, 79 | gluon 24, quark 11 |
| QG Herwig | 20, 84 | 21, 90 | gluon 28, quark 13 |

Offsets outside the 2–400 mrad grid: 0.4–1.7 % of Top jets and 3–4 % of QG jets (below 2 mrad).

## Numbers: % calls changed [95 % CI]

**Columns:**
- obs: the new axis.
- pred: the pre-registered prediction [calibration interval].
- o/p: obs / pred [interval], with Holm-corrected H1 p.
- R: random-direction shift of the same size.
- axis − R: in percentage points [CI], with Holm-corrected McNemar p.
- acc: accuracy before → after.

### Top Landscape (10,000 jets)
| model | axis | obs | pred | o/p (Holm p) | R | axis − R (Holm p) | acc % |
|---|---|---|---|---|---|---|---|
| ParT | WTA | 3.86 [3.49, 4.23] | 8.87 [8.42, 9.33] | 0.44 [0.39, 0.48] (<1e-3) | 5.04 [4.60, 5.47] | −1.18 [−1.60, −0.76] (1e-6) | 93.5 → 92.6 |
| ParT | HP | 4.68 [4.27, 5.10] | 9.68 | 0.48 [0.45, 0.53] (<1e-3) | 5.75 | −1.07 [−1.50, −0.64] (7e-5) | 93.5 → 91.7 |
| PN | WTA | 9.31 [8.76, 9.89] | 16.04 | 0.58 [0.54, 0.61] (<1e-3) | 9.07 | +0.24 [−0.27, +0.75] (1) | 93.7 → 87.6 |
| PN | HP | 10.30 | 17.44 | 0.59 [0.55, 0.62] (<1e-3) | 10.19 | +0.11 [−0.43, +0.65] (1) | 93.7 → 86.9 |
| LorentzNet | WTA / HP | 0 [0, 0] | 0 | – | 0 | 0 | 93.7 → 93.7 |
| PFN | WTA | 8.65 [8.10, 9.20] | 13.81 | 0.63 [0.59, 0.66] (<1e-3) | 9.94 | −1.29 [−1.82, −0.76] (9e-5) | 92.3 → 88.1 |
| PFN | HP | 12.65 | 15.91 | 0.80 [0.76, 0.83] (<1e-3) | 12.09 | +0.56 [−0.05, +1.17] (1) | 92.3 → 84.5 |
| EFN | WTA | 14.72 [14.05, 15.41] | 16.43 | 0.90 [0.86, 0.93] (<1e-3) | 12.79 | +1.93 [+1.33, +2.53] (2e-8) | 92.2 → 83.0 |
| EFN | HP | 19.68 [18.92, 20.45] | 19.71 | 1.00 [0.96, 1.03] (1) | 15.69 | +3.99 [+3.34, +4.64] (1e-31) | 92.2 → 78.6 |
| ParT permuted | WTA | 11.85 | 10.09 | 1.17 [1.11, 1.23] (<1e-3) | 12.15 | −0.30 [−1.06, +0.46] (1) | 66.6 → 64.2 |
| ParT permuted | HP | 13.21 | 10.98 | 1.20 [1.15, 1.26] (<1e-3) | 13.18 | +0.03 [−0.74, +0.80] (1) | 66.6 → 63.6 |

Margin-conditioned pred2, o/p2 [interval]:

| model | WTA | HP |
|---|---|---|
| ParT | 0.67 [0.61, 0.74] | 0.75 |
| PN | 0.81 | 0.77 |
| PFN | 0.83 | 1.00 [0.95, 1.05] |
| EFN | 1.06 | 1.11 |

R / pred (the prediction applied to the random-direction shift): ParT 0.57, PN 0.57, PFN 0.72, EFN 0.78, permuted 1.20.

AUC before → after, WTA:

| model | AUC before → after |
|---|---|
| ParT | 0.9842 → 0.9780 |
| PN | 0.9843 → 0.9637 |
| PFN | 0.9782 → 0.9440 |
| EFN | 0.9754 → 0.8970 |

### QG Pythia jets (10,000)
| model | axis | obs | pred | o/p (Holm p) | R | axis − R (Holm p) | acc % |
|---|---|---|---|---|---|---|---|
| ParT_kin (Py) | WTA | 3.00 [2.67, 3.35] | 2.73 | 1.10 [0.99, 1.22] (1) | 2.51 | +0.49 [+0.15, +0.83] (0.17) | 82.7 → 82.5 |
| ParT_kin (Py) | HP | 3.22 | 3.02 | 1.07 [0.96, 1.17] (1) | 2.91 | +0.31 [−0.04, +0.66] (1) | 82.7 → 82.5 |
| PN_kin (Py) | WTA | 4.31 [3.91, 4.75] | 5.67 | 0.76 [0.70, 0.84] (<1e-3) | 4.49 | −0.18 [−0.60, +0.24] (1) | 82.5 → 82.1 |
| PN_kin (Py) | HP | 4.79 | 6.25 | 0.77 [0.71, 0.83] (<1e-3) | 5.42 | −0.63 [−1.07, −0.19] (0.17) | 82.5 → 82.0 |
| PFN (Py) | WTA | 11.35 [10.72, 11.94] | 12.04 | 0.94 [0.89, 0.99] (0.62) | 9.49 | +1.86 [+1.23, +2.49] (5e-7) | 82.0 → 78.4 |
| PFN (Py) | HP | 12.58 | 12.51 | 1.01 [0.96, 1.06] (1) | 10.78 | +1.80 [+1.15, +2.45] (3e-6) | 82.0 → 77.8 |
| EFN (Py) | WTA | 22.50 [21.67, 23.37] | 19.46 | 1.16 [1.12, 1.20] (<1e-3) | 21.01 | +1.49 [+0.81, +2.17] (8e-4) | 80.8 → 71.1 |
| EFN (Py) | HP | 24.00 | 20.19 | 1.19 [1.15, 1.22] (<1e-3) | 22.95 | +1.05 [+0.37, +1.73] (0.10) | 80.8 → 70.5 |
| ParT_kin (Hw) | WTA | 2.86 [2.54, 3.17] | 2.83 | 1.01 [0.91, 1.13] (1) | 2.50 | +0.36 [+0.02, +0.70] (1) | 79.5 → 80.3 |
| ParT_kin (Hw) | HP | 3.01 | 3.11 | 0.97 [0.87, 1.07] (1) | 2.83 | +0.18 (1) | 79.5 → 80.3 |
| PN_kin (Hw) | WTA | 2.92 | 3.32 | 0.88 [0.79, 0.97] (<1e-3) | 2.61 | +0.31 (1) | 79.3 → 79.7 |
| PN_kin (Hw) | HP | 3.13 | 3.55 | 0.88 [0.80, 0.97] (0.10) | 2.69 | +0.44 [+0.12, +0.76] (0.27) | 79.3 → 79.7 |
| ParT permuted | WTA | 3.87 [3.49, 4.26] | 2.40 | 1.61 [1.47, 1.76] (<1e-3) | 2.47 | +1.40 [+0.99, +1.81] (8e-10) | 49.9 → 50.7 |
| ParT permuted | HP | 4.29 | 2.68 | 1.60 (<1e-3) | 3.05 | +1.24 [+0.81, +1.67] (1e-6) | 49.9 → 50.8 |
| LorentzNet | WTA / HP | 0 | 0 | – | 0 | 0 | 84.1 → 84.1 |
| LR(girth) | WTA | 5.06 [4.64, 5.48] | 3.62 | 1.40 [1.24, 1.54] (<1e-3) | 4.44 | +0.62 [+0.09, +1.15] (0.68) | 69.3 → 71.5 |
| LR(girth) | HP | 5.29 | 3.66 | 1.44 (<1e-3) | 4.52 | +0.77 (0.17) | 69.3 → 71.4 |
| LR(N) | WTA / HP | 0 | 0 | – | 0 | 0 | 76.8 → 76.8 |

### QG Herwig jets (10,000)
| model | axis | obs | pred | o/p (Holm p) | R | axis − R (Holm p) | acc % |
|---|---|---|---|---|---|---|---|
| ParT_kin (Py) | WTA | 4.15 [3.78, 4.53] | 3.76 | 1.10 [1.01, 1.19] (0.76) | 4.08 | +0.07 [−0.35, +0.49] (1) | 71.8 → 71.2 |
| ParT_kin (Py) | HP | 4.40 | 4.01 | 1.10 (0.76) | 4.07 | +0.33 (1) | 71.8 → 70.9 |
| PN_kin (Py) | WTA | 6.15 | 7.21 | 0.85 [0.80, 0.92] (<1e-3) | 6.42 | −0.27 (1) | 71.6 → 70.8 |
| PN_kin (Py) | HP | 6.70 | 7.61 | 0.88 [0.82, 0.94] (0.10) | 7.24 | −0.54 [−1.06, −0.02] (1) | 71.6 → 70.8 |
| PFN (Py) | WTA | 14.55 [13.90, 15.26] | 12.46 | 1.17 [1.11, 1.23] (<1e-3) | 13.02 | +1.53 [+0.83, +2.23] (8e-4) | 72.4 → 67.6 |
| PFN (Py) | HP | 15.65 | 12.79 | 1.22 (<1e-3) | 14.09 | +1.56 [+0.84, +2.28] (9e-4) | 72.4 → 66.7 |
| EFN (Py) | WTA | 24.44 [23.59, 25.28] | 19.33 | 1.26 [1.22, 1.30] (<1e-3) | 23.49 | +0.95 [+0.29, +1.61] (0.17) | 71.1 → 61.0 |
| EFN (Py) | HP | 25.56 | 19.87 | 1.29 (<1e-3) | 24.10 | +1.46 [+0.80, +2.12] (8e-4) | 71.1 → 60.3 |
| ParT_kin (Hw) | WTA | 4.30 [3.91, 4.73] | 3.92 | 1.10 [1.01, 1.19] (0.55) | 3.56 | +0.74 [+0.33, +1.15] (0.017) | 75.5 → 75.1 |
| ParT_kin (Hw) | HP | 4.44 | 4.20 | 1.06 (1) | 3.71 | +0.73 [+0.33, +1.13] (0.014) | 75.5 → 75.1 |
| PN_kin (Hw) | WTA | 4.10 [3.71, 4.51] | 4.47 | 0.92 [0.84, 1.00] (1) | 3.57 | +0.53 [+0.15, +0.91] (0.22) | 75.6 → 75.5 |
| PN_kin (Hw) | HP | 4.44 | 4.72 | 0.94 (1) | 3.97 | +0.47 (0.6) | 75.6 → 75.4 |
| ParT permuted | WTA | 4.56 | 2.66 | 1.72 [1.58, 1.86] (<1e-3) | 3.14 | +1.42 [+0.96, +1.88] (6e-8) | 49.9 → 51.1 |
| ParT permuted | HP | 5.26 | 2.97 | 1.77 (<1e-3) | 3.23 | +2.03 [+1.54, +2.52] (2e-14) | 49.9 → 51.3 |
| LorentzNet | WTA / HP | 0 | 0 | – | 0 | 0 | 69.7 → 69.7 |
| LR(girth) | WTA | 5.27 [4.86, 5.73] | 3.79 | 1.39 [1.25, 1.54] (<1e-3) | 4.65 | +0.62 [+0.07, +1.17] (0.75) | 66.3 → 67.9 |
| LR(girth) | HP | 5.36 | 3.87 | 1.39 (<1e-3) | 5.02 | +0.34 (1) | 66.3 → 68.0 |
| LR(N) | WTA / HP | 0 | 0 | – | 0 | 0 | 71.7 → 71.7 |

**By class** (WTA, % calls changed), Top and QG Pythia:

| model | class 1 | class 0 |
|---|---|---|
| Top ParT | top 4.1 | QCD 3.6 |
| Top PN | top 14.5 | QCD 4.1 |
| QG EFN (Py jets) | quark 12.9 | gluon 32.1 |

All other per-class values are in the JSON.

**Mean |Δl|** (WTA, observed / predicted / R):

| model | observed | predicted | R |
|---|---|---|---|
| Top ParT | 0.82 | 1.07 | 0.96 |
| Top EFN | 1.77 | 2.30 | 1.74 |
| QG Py ParT_kin (Py jets) | 0.156 | 0.169 | 0.146 |

## Hypotheses

**H1 (pre-registered): the translation response predicts the axis response.** Predictions: o/p ≈ 1 for ParT, PN, PFN and EFN; LorentzNet exactly 0; permuted nets ≈ R.
- **LorentzNet:** 0 calls changed on all three samples and both axes. Logit jitter equals the repeat floor.
- **o/p ≈ 1 (Holm p > 0.05):**
  - QG trained networks: 14 of 24 cells, o/p 0.94–1.10 in those cells; the full QG range is 0.76–1.29.
  - Top: 1 of 8 cells (EFN, HP). The others lie at 0.44–0.90.
- **Prediction-method bias:** the same prediction applied to R gives R/pred 0.57–0.80 on Top networks and 0.76–1.22 on QG networks. Most of the departure of o/p from 1 is therefore shared with the random-direction shift.
- **Permuted nets ≈ R:** holds on Top (axis − R −0.30 and +0.03 pp, Holm p = 1). It is excluded on QG: +1.2 to +2.0 pp, Holm p 1e-6 to 2e-14.

**H0: a response to the axis beyond the translation.** Predictions: measured ≫ predicted, and axis − TN outside the R interval.
- **axis − TN:** identically 0 for these inputs (see Procedure), so this part of H0 cannot be tested.
- **Measured ≫ predicted:** not observed. The largest o/p for a trained network is 1.29 (QG EFN on Herwig jets). Top networks lie at or below 1.
- **axis − R:**
  - Trained networks range from −1.3 to +4.0 pp. The largest is Top EFN HP at +4.0 [+3.3, +4.6].
  - The QG permuted null alone gives +1.2 to +2.0 pp. On QG jets, a shift toward the hardest particle changes more calls than a random-direction shift even for an untrained function.
  - On QG, the trained-network excesses (PFN +1.5 to +1.9, EFN +1.0 to +1.5, ParT/PN −0.6 to +0.7) are not larger than the permuted null's.

**Dataset prediction (Top nets change fewer calls than JEDI on hls4ml):**
- Top WTA: 3.9 % (ParT) to 14.7 % (EFN), against JEDI-linear 30.3 % on hls4ml (E17).
- This holds even though Top WTA offsets are larger: median 93 mrad against 40 mrad on hls4ml.
- Top ParT (3.9 %) also changes fewer calls than JetClass ParT_full (11.7 %) and ParT_kin (34.1 %) did in D4.

**QG prediction (small changes because WTA ≈ E-scheme for one-prong jets):**
- Offsets are 11 mrad (quark) and 24 mrad (gluon).
- ParT- and PN-type networks change 2.9–6.7 % of calls.
- PFN and EFN change 11–26 %, as their translation response predicted (o/p 0.94–1.29).

## Reading (scoped to what was measured)
- For every model tested, the axis convention enters only as a rigid translation of $(\Delta\eta,\Delta\phi)$. The per-model size of the change follows the measured translation response within a factor of 0.4–1.3, and the model ranking is preserved:
  - QG: ParT ≈ PN < PFN < EFN.
  - Top: ParT < PFN ≈ PN < EFN.
- The four-vector-only LorentzNet is unaffected.
- On Top jets, ParT and PFN change fewer calls under the WTA axis than under a random-direction shift of the same size:
  - ParT: −1.2 pp; PFN: −1.3 pp.
  - Top EFN changes more under the HP axis: +4.0 pp.
  - The R − pred bias says that the pooled, class-level calibration overstates flips for these jets. It does not say why the WTA direction differs from a random direction.
- LR(girth): girth about the WTA or HP axis raises its accuracy (69.3 → 71.5 % Pythia; 66.3 → 67.9 % Herwig). Its calls change in the gluon → quark direction: mean Δl −0.13, against +0.11 under R. This is a known function whose response depends on the shift direction.

## Issues
- **Grid deviation:** the calibration grid is 2–400 mrad, not the planned 2–20 mrad, because Top offsets are 93 mrad median. 0.4–1.7 % of Top offsets lie above the grid (value at 400 mrad used).
- **The prediction model is crude:** the empirical $\Delta l$ is pooled per class (pred2: per class × $|l_0|$ quartile) and averaged over directions. Its own bias, measured with R, is a factor of 0.57–1.30, so the o/p tests mostly test this method.
- **TN is an identity:** for these input sets, TN is identical to the axis recompute, so the axis-beyond-translation part of H0 is tested only through axis vs R. R is not a clean null on QG: the permuted network shows axis > R.
- **Missing models:** no Herwig-trained PFN or EFN exists, so the QG Herwig-trained set is ParT_kin and PN_kin only. No second training seed exists for any network.
- **Pre-registration timing:** prediction values were computed and logged before step 3 for each dataset, but `h36_prereg.json` was written after that dataset's step 3 finished. The prediction code does not read step-3 outputs.
- **LorentzNet jitter:** LorentzNet logits jitter by up to $5\times10^{-3}$ between identical runs (GPU scatter non-determinism). No call changes.

## Explainer entry (for site/explain.py; to be merged by the ledger)
```python
H36_EXPLAINER = [
 ('Question', 'Recomputing the inputs about the winner-take-all (WTA) axis instead of the stored E-scheme axis changed 12–34 % of hls4ml / JetClass tagger calls (D4). Does the same happen for Top Landscape and quark–gluon taggers, and can the size be predicted beforehand from how each model responds to a plain shift of its inputs (D4m)?', None),
 ('Physics background',
  'Constituent inputs are angles relative to a jet axis: $\\Delta\\eta_i=\\mathrm{sign}(\\eta_J)(\\eta_i-\\eta_a)$, $\\Delta\\phi_i=\\phi_i-\\phi_a$. The E-scheme axis is the four-momentum sum; the WTA axis (C/A clustering with winner-take-all recombination) follows the harder branch and lands on one particle, typically in the hardest prong; the hardest-particle (HP) axis is the leading constituent. '
  'Changing the axis is a rigid translation of all $(\\Delta\\eta_i,\\Delta\\phi_i)$ by the axis offset $\\delta$. Lorentz-invariant networks built on four-vectors (LorentzNet) cannot see it; IRC-safe classical observables with no axis (LR($m_{SD}$, $\\tau_{32}$), LR(N)) are unchanged; girth $\\sum_i z_i\\Delta R_i$ is axis dependent.', None),
 ('Method',
  'Top Landscape (seed 3601) and quark–gluon Pythia / Herwig (seed 3602) test jets: 1,000 per class for calibration, 5,000 per class measured. Step 1: rigid shifts $r\\in[2,400]$ mrad in 8 directions give each model\'s $\\Delta l$ distribution per class. Step 2 (before measuring): per jet $p_j=P(\\mathrm{sign}(l_{0j}+\\Delta l)\\ne\\mathrm{sign}(l_{0j}))$ at its own offset $|\\delta_j|$; predicted % calls changed $=\\langle p_j\\rangle$. Step 3: inputs recomputed about WTA and HP; controls TN (the same shift through a separate code path) and R (same size, random direction). '
  'Models: Top ParT, PN, LorentzNet, PFN, EFN, permuted ParT; QG Pythia- and Herwig-trained ParT_kin / PN_kin, PFN, EFN, LorentzNet, permuted ParT, LR(girth), LR(N). Bootstrap intervals, McNemar axis vs R, Holm over 52 cells.', None),
 ('Result',
  'Median WTA offset 93 mrad on Top (top jets 157, QCD 40), 17 / 20 mrad on QG Pythia / Herwig (quark 11, gluon 24). Calls changed (WTA): Top ParT 3.9 %, PN 9.3 %, PFN 8.7 %, EFN 14.7 %, LorentzNet 0; QG ParT_kin 2.9–4.3 %, PN_kin 2.9–6.2 %, PFN 11–15 %, EFN 22–24 %, LorentzNet 0, LR(N) 0. Observed / predicted 0.94–1.10 in 14 of 24 QG network cells (QG range 0.76–1.29), 0.44–1.00 on Top; the prediction misses the random-direction shift by a similar factor (R / pred 0.57–1.22). Shifts toward the hardest particle change at most +4.0 pp more calls than random shifts (Top EFN, HP); the permuted QG network shows +1.2 to +2.0 pp.', 'results/physics/h36_axis.png'),
 ('Checks',
  'Unedited AUCs reproduce earlier samples (Top ParT 0.9842, QG Py ParT_kin 0.898, Hw ParT_kin 0.831). TN and the axis recompute agree to $5\\times10^{-7}$ in every model (the axis enters only through $\\Delta\\eta,\\Delta\\phi$). LorentzNet: 0 calls changed; logit jitter $\\le5\\times10^{-3}$ equals the repeat-scoring floor. The calibration grid was extended from 2–20 to 2–400 mrad because Top offsets are larger; 0.4–1.7 % of Top offsets lie above it.', None),
 ('Physical significance',
  'On Top and quark–gluon jets the axis convention acts on these taggers as a translation of the constituent image: the fraction of calls that change follows each model\'s translation sensitivity (EFN and PFN most, ParT least) and the offset size (large in tops, small in one-prong quark jets), and a Lorentz-invariant tagger is immune. There is no sign of a large axis-specific response beyond translation; a 1–4 point excess of hardest-particle-directed over random shifts also appears in an untrained network.', None),
]
```
Entry line for the `'d'` list: `('Result explained: axis convention (WTA, hardest particle) on Top and quark–gluon taggers, predicted from translation response (claims D4, D4m; H36)', H36_EXPLAINER, False)`.
