# E19: axis convention by class, and top jets (2026-10-17)

- **Question (user):** "Wouldn't other heavy multi-prong jets change too, like top?"
- **Hypotheses:**
  - H1: for top jets the networks' mass reading rises as the per-jet $\tfrac12\Delta\log(M_0M_2)$ about the new axis predicts, and the top call is kept because the top decision responds to mass one-sidedly.
  - H0: the top reading does not rise.
- **Jobs:**
  - `anrunw-ai-e19-jedi`: 1 CPU, 74 s.
  - `anrunw-ai-e19-part`: GPU, 91 s.
  - `anrunw-ai-e19-plot`.
  - Smoke runs: `e19-jedi-smoke`, `e19-part-smoke`.
- **Scripts:**
  - `interp/e19_common.py`: kT 3-subjet clustering and statistics.
  - `interp/e19_axis_jedi.py`
  - `interp/e19_axis_part.py`
  - `interp/e19_plot.py`
- **Results:**
  - `results/jedi/e19_axis_by_class.{json,npz}`
  - `results/part/e19_axis_by_class.{json,npz}`
  - `results/site/T16_axis_by_class.png`, `results/site/F9_top_axis.png`
  - Logs: `logs/e19-{jedi,part,plot}.log`

## Procedure

### Samples
- **JEDI-linear 3f n64:** 20,000 hls4ml test jets (4,000 per class, seed 81).
- **ParT_full and ParT_kin:** 10,000 JetClass test jets (rows 8,000–9,000 of each class's first test file).

### Edits
Axis edits follow E17 and P1v check 4. The particles' $(\Delta\eta, \Delta\phi)$, and for ParT also $\Delta R$, are recomputed about a new axis; momenta stay unchanged.
- Identity: the stored axis.
- Null controls: the pT centroid (JEDI) and the recomputed E-scheme axis (ParT).
- Edits: the WTA axis and the hardest particle.

### Read-out
- A ridge regression from the last hidden layer to $\log m$: JEDI's 16 pre-ReLU z neurons, ParT's 128-d class token.
- It is fitted on unedited jets of the train half (50/50 split) and evaluated on the test half.
- Test-half $R^2$:
  - JEDI 0.940; per class g/q/W/Z/t .93/.89/.80/.86/.89.
  - ParT_full 0.894; Tbqq only 0.37.
  - ParT_kin 0.924; Tbqq 0.72.

### Prediction and calibration
- **Prediction:** the per-jet $\tfrac12\Delta\log(M_0M_2)$ about the new axis.
- **Matched calibration edit:** each jet's angles are scaled about the stored axis by $e^{\text{pred}_j}$. This is a real mass change of the predicted size (true $\Delta\log m$ matches the prediction to 0.001).
- "cal ratio" = Δlog m̂(axis) / Δlog m̂(scale).

### Statistics
- Bootstrap 95 % CIs: 2,000 resamples over jets.
- Wilson intervals for call fractions.
- Exact McNemar test for own-class accuracy.
- One-sample t-test of H0 ($\Delta\log\hat m = 0$) per class. Holm correction over 2 axes × classes: the largest adjusted p is 1e-49 (JEDI), 2e-35 (ParT_full) and 5e-5 (ParT_kin).
- The null control axes move the read-out by ≤ 0.0014 (JEDI) or ≤ 0.001 (ParT) in every class.

## 1. Per class, WTA axis (T16; hardest-particle axis in T16's last column)

| model | class | Δlog m̂ [CI] | pred | ratio [CI] | cal ratio [CI] | calls changed % [CI] | own acc % | p(H0) |
|---|---|---|---|---|---|---|---|---|
| JEDI | g | .067 [.061,.072] | .080 | 0.83 [.78,.89] | 0.93 [.88,.98] | 15.2 [14.1,16.4] | 78.8→69.0 | 1e-117 |
| JEDI | q | .029 [.026,.032] | .055 | 0.53 [.48,.57] | 0.75 [.69,.81] | 10.3 [9.4,11.3] | 71.7→72.4 | 1e-56 |
| JEDI | W | .084 [.079,.088] | .149 | 0.56 [.54,.58] | 0.48 [.46,.50] | 68.2 [66.7,69.6] | 86.2→22.5 | 1e-200 |
| JEDI | Z | .092 [.087,.098] | .156 | 0.59 [.57,.62] | 0.56 [.54,.58] | 44.7 [43.2,46.2] | 83.5→48.6 | 1e-197 |
| JEDI | t | .149 [.143,.156] | .164 | 0.91 [.87,.95] | 1.02 [.98,1.06] | 12.2 [11.2,13.2] | 84.0→83.2 | 1e-296 |
| ParT_full | Wqq | .138 [.130,.147] | .141 | 0.98 [.94,1.03] | 1.06 [1.02,1.11] | 31.5 [28.7,34.4] | 78.9→52.0 | 1e-124 |
| ParT_full | Zqq | .132 [.122,.141] | .136 | 0.97 [.92,1.02] | 1.05 [1.01,1.10] | 27.4 [24.7,30.2] | 70.9→61.8 | 1e-107 |
| ParT_full | Tbqq | .251 [.235,.269] | .146 | 1.72 [1.64,1.79] | 1.72 [1.60,1.86] | 5.3 [4.1,6.9] | 95.5→92.1 | 1e-105 |
| ParT_full | Hgg | .150 [.139,.161] | .148 | 1.01 [.96,1.06] | 1.17 [1.10,1.26] | 15.8 [13.7,18.2] | 76.0→66.4 | 1e-94 |
| ParT_kin | Wqq | .156 [.147,.165] | .141 | 1.11 [1.06,1.16] | 1.40 [1.34,1.48] | 42.6 [39.6,45.7] | 70.0→37.3 | 1e-131 |
| ParT_kin | Zqq | .128 [.120,.136] | .136 | 0.94 [.89,.99] | 1.10 [1.03,1.17] | 38.3 [35.3,41.4] | 55.7→47.6 | 1e-117 |
| ParT_kin | Tbqq | .065 [.055,.075] | .146 | 0.45 [.38,.52] | 0.52 [.44,.60] | 42.1 [39.1,45.2] | 89.6→55.5 | 1e-32 |
| ParT_kin | Hgg | .077 [.067,.087] | .148 | 0.52 [.45,.59] | 0.63 [.54,.72] | 61.7 [58.6,64.7] | 67.5→21.7 | 1e-43 |

- The other ParT classes are in T16 and the JSON. ParT_kin Tbl is the only class with a negative read-out change: −.019 [−.028, −.010].

### Median axis displacement (WTA)
- JEDI g/q/W/Z/t: .022 / .009 / .044 / .050 / .083 rad.
- ParT Tbqq .149, Wqq .076, QCD .028 rad.

### Own-class McNemar, WTA
- JEDI t: 155 lost vs 126 gained, p = 0.095. Hardest axis: p = 0.65.
- JEDI W: p < 1e-300. JEDI Z: p = 1e-290.
- ParT_full Tbqq: 40 vs 6, p = 3e-7.
- ParT_kin Tbqq: 355 vs 14, p = 1e-86.

### Flows for true-t jets (JEDI, WTA)
- Calls after the edit: t 83.3 %, q 6.6 %, Z 4.6 %, g 4.5 %, W 1.1 %.
- Stored-axis t calls that stayed t: 3,204 of 3,328.

### Flows for true-W and true-Z jets (JEDI, WTA)
- Stored W calls (2,850 jets): 2,178 now Z, 408 now q.
- Stored Z calls (3,270 jets): 1,450 now q.

### Per-jet slope of Δlog m̂ on pred, WTA (slope, R²)
- JEDI W: 0.85, .56.
- JEDI Z: 0.99, .63.
- JEDI t: 0.59, .15.
- ParT_full Tbqq: 1.61, .63.
- ParT_kin Tbqq: 0.17, .02.

## 2. Top jets

### Read-out mass, stored → WTA (F9)
- **JEDI t:** quantiles 10/50/90 % go 92/146/189 → 95/175/233 GeV.
  - Fraction above 143 GeV: 0.529 → 0.690 [0.670, 0.710].
  - McNemar on crossing 143 GeV: 330 jets up vs 6 down, p = 3e-89.
- **ParT_full Tbqq:** median 171 → 218 GeV (90 %: 200 → 302).
- **ParT_kin Tbqq:** median 170 → 182 GeV.

### JEDI t-logit neurons
The three neurons with the largest $|K_7[k,t]\cdot \mathrm{sd}(\mathrm{ReLU}\,z_k)|$ (this replaces the z10 assumption, as requested):
- **z10:** contribution +1.16; weights to g/q/W/Z/t −.02/−.02/0/0/+.98.
- **z13:** contribution −0.79; weight to t −0.91, so this neuron counts against t.
- **z8:** contribution +0.37; weights to W/Z −0.88/−0.94, to t +0.21.

Response to the read-out mass on unedited test-half jets (mean value; fraction on):

| neuron | 100–110 | 130–143 | 170–190 | 190–220 | 220–260 | 260–320 GeV |
|---|---|---|---|---|---|---|
| z10 | 1.44; .92 | 3.02; .99 | 2.72; .92 | 1.94; .81 | −1.8; .36 (n = 33) | −4.6; 0 (n = 12) |
| z13 | 1.79; .91 | 1.25; .77 | −1.77; .21 | −3.68; .05 | −9.8; 0 | −14.3; 0 |
| z8 | 1.76; .88 | 3.82; 1.0 | 3.81; 1.0 | 3.86; 1.0 | 4.1; 1.0 | 3.4; 1.0 |

- z13 turns off above about 143–170 GeV and stays off, which removes its negative t contribution.
- z8 is on from about 110 GeV and stays on.
- z10 is a window: on at 100–220 GeV and off above about 220 GeV.
- t-logit margin (t minus the best other class) by read-out mass:
  - −2.3 at 100–110 GeV, +1.3 at 130–143, +3.9 at 170–190, +3.8 at 190–220.
  - +1.6 at 220–260 and +0.5 at 260–320 (n = 33 and 12).
- On WTA-edited jets, z10 drops at lower read-out mass than on unedited jets: mean 0.70 at 170–190 GeV vs 2.72 unedited. So z10 is not a function of the read-out mass alone.

Top jets, stored → WTA:

| neuron | fraction on [CI] | McNemar | mean value |
|---|---|---|---|
| z10 | .967 → .662 [.647, .676] | 1,224 off vs 5 on | 2.62 → 0.48 |
| z13 | .520 → .275 [.261, .289] | 997 off vs 14 on, p = 1e-273 | −0.25 → −5.31 |
| z8 | .946 → .977 | 128 on vs 4 off, p = 5e-33 | 3.52 → 3.31 |

- t-logit change: −1.21 [−1.26, −1.16].
- Mean t margin: 2.33 → 1.79.
- Hardest axis: the same pattern. z10 .967 → .603, z13 .520 → .261.

### ParT Tbqq-logit neurons (W[:, Tbqq] · sd)
- **ParT_full:** h16 +1.90, h104 −1.86, h68 +1.79.
  - Unedited response: h16 rises above 143 GeV and stays positive to 260; h104 falls with mass; h68 peaks at 170–190 GeV (1.82) and falls to 0.64 at 220–260.
  - Tbqq jets under WTA: h16 1.35 → 1.21, h104 −1.28 → −1.06, h68 2.56 → 2.19. Tbqq logit −1.43 [−1.55, −1.31].
- **ParT_kin:** h103 +2.00, h51 −1.26, h121 −1.14.
  - Unedited response: h121 falls from about +0.5 (100–110 GeV) to −0.98 (170–190).
  - Under WTA it moves from −1.40 to −0.20, toward its low-mass value, although the read-out mass rises. Tbqq logit −2.05.

### Three-prong prediction (true-t test-half jets; exclusive kT, 3 subjets; resolved = each subjet pT fraction ≥ 0.10 and pairwise ΔR ≥ 0.10)
- **JEDI:** 1,067 of 2,006 jets resolved. The WTA axis lies in the hardest subjet in 89 % of them (hardest particle: 70 %).
  - Median $\sqrt{M_0M_2}$ about that subjet divided by m: 1.18 (point-prong formula 1.20; symmetric case $\sqrt2$).
  - The subjet prediction reproduces the actual-axis prediction: slope 0.99, R² .90.
  - Read-out change on the subjet prediction: slope 0.54, R² .12 (through the origin 0.91).
  - Hardest axis: slope 0.82, R² .27 (through the origin 0.99).
  - Mean read-out change when the axis is in the hardest subjet vs another subjet (WTA): .164 vs .246.
- **ParT_full:** 393 of 510 jets resolved. Slope 1.53, R² .54 (WTA); 1.52, R² .64 (hardest).
- **ParT_kin:** slope 0.26, R² .05 (WTA); 0.30, R² .10 (hardest).

## 3. ParT_kin flows and ParT_full vs ParT_kin

### Tbqq under WTA
- **ParT_kin:** own 89.6 → 55.5 %. New calls: Tbl 33.2 %, Hqql 5.0 %.
- **ParT_full:** 95.5 → 92.1 %. Losses are spread over QCD, Hbb, Hcc, H4q and Zqq, each 1.2–1.7 %.

### Hgg under WTA
- **ParT_kin:** 67.5 → 21.7 %. New calls: Tbl 46.4 %, H4q 10.7 %, Hcc 6.1 %, Hbb 5.5 %.
- **ParT_full:** 76.0 → 66.4 %. New calls: QCD 11.5 %, H4q 10.9 %.

### Tbl calls on other classes under WTA (ParT_kin)
- QCD jets called Tbl: < 1 % → 15.1 %.
- Wqq jets: 3.9 %.
- Zqq jets: 5.4 %.

### Read-out change, ParT_full vs ParT_kin
- Tbqq: 0.251 vs 0.065.
- Hgg: 0.150 vs 0.077.
- W/Z: about the same (0.13–0.16).

### Input-group intervention (ParT_full; WTA and stored axis both with the group fed as 0)
- Displacement inputs (d0, d0err, dz, dzerr) fed as 0: accuracy 24.5 % on the stored axis.
- Charge and PID fed as 0: accuracy 13.2 % on the stored axis.
- Both variants are outside the training distribution and do not separate the input groups. No ParT_kinpid model is available in `jetdistill/part/network.py` (FEATURES has only kin and full).

## Reading (scoped to these measurements)
- **H0 for top:** rejected in all three networks.
  - JEDI t: Δlog m̂ = +0.149, p ≈ 1e-296.
  - ParT_full Tbqq: +0.251, p ≈ 1e-105.
  - ParT_kin Tbqq: +0.065, p ≈ 1e-32.
  - The paired difference against the null control axis is also significant in every class.
- **JEDI's top reading rises as predicted:** cal ratio 1.02 [0.98, 1.06] for t, against 0.48 for W and 0.56 for Z.
- **JEDI top calls:** own-class accuracy is unchanged (McNemar p = 0.095). The read-out above 143 GeV rises from 53 % to 69 % of top jets.
  - Of the three t-logit neurons, z13 (anti-t) and z8 (t, anti-W/Z) respond one-sidedly to mass and both move toward t.
  - z10 is a window that switches off for 31 % of top jets and lowers the t logit by 1.2 on average. The margin stays positive on average.
  - H1 holds for the read-out and for z13/z8. The "one-sided threshold" form does not hold for z10.
- **ParT_full:** reads Tbqq jets 1.7× the prediction and keeps 92 % of the top calls.
- **ParT_kin:** reads them at 0.45× the prediction, with per-jet R² .02. Its Tbqq and Hgg losses go mainly to Tbl, not to a heavier class. The ParT_kin top change is not described by the mass reading.

## Issues
- ParT_full's read-out has per-class R² 0.37 on Tbqq, so its Tbqq ratio of 1.72 rests on a weaker probe. The matched calibration edit gives the same 1.72.
- The "resolved" cut (0.10 / 0.10) was chosen before looking at the results and not varied.
- The input-group variant is uninformative (see section 3).
- Nothing was committed.
