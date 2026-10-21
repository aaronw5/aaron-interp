# H58: classical-tagger fingerprint rerun with an m_SD-only tagger, matched working points and 55 permuted copies per architecture (claim D23)

## Conclusion
Nine physics edits were applied to fresh jets from hls4ml, JetClass and the Top Tagging Landscape. For each network we asked which simple classical tagger changes its yes/no calls in the most similar pattern, with every tagger and network set to keep 50 % of signal jets. On top-quark tasks, the networks (JetClass and Top Landscape, except the EFN) follow a window on the soft-drop groomed mass more closely than a window on the ungroomed mass; on JetClass this difference is about +0.1 to +0.18 in cosine similarity. Adding the N-subjettiness τ32 cut to the groomed-mass window does not make the match closer for these networks. A τ21 cut adds to the match only for JetClass Z versus QCD (all four networks) and τ32 only for the Top Landscape EFN. The hls4ml networks do not form one group. Three of five (Deep Sets, JEDI-linear, JEDI-style float) are closer to the ungroomed mass window than to the groomed one on W and Z. The Transformer goes the other way, the Linformer shows no clear difference after Holm, and on top four of five are closer to the groomed window. So D23 as stated (JetClass/Top follow groomed mass + shape, hls4ml follow a plain mass window) is not supported. Two narrower statements are supported: top taggers follow groomed rather than ungroomed mass; JetClass Z taggers also follow τ21. When the classical taggers sit at their published cuts instead of 50 % efficiency, every network looks closer to the shape taggers. Part of H55's "shape" preference was therefore caused by the working point.

## What was done
Script `interp/h58_fingerprint_rerun.py` (imports the H55 edit and observable code unchanged). Cluster jobs:
- `d-h58-edit`: edits and observables;
- `d-h58-jedi`, `d-h58-{dsets,linf,tf,jedis}`: hls4ml networks, CPU;
- `d-h58-gjcp`, `d-h58-gjcn`, `d-h58-gtop`, `d-h58-gqg`: GPU. These were launched from a scratchpad copy of part_job.sh that also excludes node k8s-chase-ci-01, which rejected pods with "GPU is lost", as in H55v;
- `d-h58-stats`.

Smoke runs: `d-h58-{esmoke,esmoke2,jsmoke,ssmoke,gsmoke,tsmoke}`.

Results:
- `results/physics/h58_fingerprint.json`
- `results/physics/h58_contrasts.png`
- per-dataset accuracy files in `results/physics/h58_work/`
- tables and log-odds on the volume only, under `/j-jepa-vol/aaron-interp/results/physics/h58_work/`

### Samples (fresh rows, random indices only)
| dataset | rows | disjoint from |
|---|---|---|
| hls4ml | 3,000 per class (g, q, W, Z, t) = 15,000 jets, `default_rng(5801)` over full_test.npz rows | the rows of H55 (seed 5501) and H55v (seed 5591), excluded explicitly |
| JetClass | Hbb, Wqq, Zqq, Tbqq, QCD: 3,000 random rows each from `test_20M/*_102.root` (`default_rng 5802`); η = jet_eta + sign(jet_eta)·Δη | H55 / H55v used `*_100.root` |
| Top Landscape | test split, 3,000 top + 3,000 QCD (`default_rng 5803`) | H55 rows (5503) excluded |
| QG (Pythia) | test split, 3,000 quark + 3,000 gluon (`default_rng 5804`) | H55 rows (5504) excluded |

Each binary task uses the signal class plus its background classes. hls4ml W/Z/top use q+g as background (9,000 jets per task). JetClass and Top use QCD (6,000 jets per task).

### Edits
The nine H55 edits, unchanged. Each produces a new particle list that is passed to both the networks and the classical observables.
- **split:** the hardest particle is replaced by two collinear halves. The four-momentum and every IRC-safe observable are unchanged exactly; the particle count N becomes N+1.
- **ksplit3:** three random particles of the softer half are each split the same way (N+3).
- **soft:** one neutral particle with 0.1 % of the jet pT is added at R/2 < ΔR < R from the axis. Its three-momentum is taken from the hardest particle.
- **wta:** the inputs are measured about the winner-take-all axis instead of the stored axis.
- **ctrl:** the axis is moved by the same distance as in wta, in a random direction.
- **gin / gout:** three soft particles (2 % of jet pT) are placed where soft drop keeps them (gin) or removes them (gout). The ungroomed mass change is matched between the two.
- **decay:** a jet rest-frame rotation raises |cos θ*| of the two-prong axis by 0.3. The jet four-momentum, the jet mass, N and all pairwise masses are fixed.
- **zboost:** a boost along the beam with rapidity shift 0.5. m, pT, Δy and Δφ are fixed.

**Edits that apply to only part of the sample:**
- hls4ml split, ksplit3 and soft do not apply to the 23 %, 25 % and 23 % of jets whose softest particle would leave the 64-slot window.
- gin / gout matched sets cover 59 % (hls4ml), 74 % (JetClass) and 62 % (Top) of jets.

### Classical taggers
Observables:
- m: invariant mass.
- m_SD: soft-drop mass (C/A clustering, β = 0, z_cut = 0.1).
- τ21, τ32: N-subjettiness ratios with exclusive-kT axes, β = 1.
- D2 = e3/e2³, β = 1.

There are two versions of every window and shape tagger.

1. **Published cuts (secondary analysis).** All windows are 65–105 GeV (W/Z) or 105–210 GeV (top).

   | tagger | definition |
   |---|---|
   | V_τ21 | m_SD window and τ21 < 0.45 |
   | V_DDT | m_SD window and τ21 + 0.063 ln(m_SD²/pT) < 0.55 |
   | V_D2 | m_SD window and D2 below its in-window signal median |
   | top_τ32 | m_SD window and τ32 < 0.54 |
   | **msd_V / msd_top (new)** | m_SD window only |
   | **mass_V / mass_top** | ungroomed m window only |

2. **Matched 50 % working point ("@50", main analysis).** Each tagger's cut is reset so that it keeps exactly 50 % of the unedited signal jets of the task. The networks are cut at the same 50 %.
   - **msd@50 / mass@50:** the shortest m_SD (or m) interval holding 50 % of the signal. For example, JetClass W: m_SD 70.4–85.2 GeV and m 74.3–88.8 GeV; JetClass top: m_SD 152.9–182.5 GeV.
   - **shape@50:** the published m_SD window plus the shape variable cut at the value giving 50 % total signal efficiency. For example, JetClass W: τ21 < 0.398; JetClass top: τ32 < 0.526. The published window held at least 55 % of the signal in every task, so the fallback (a wider window) was never used.

**Numerical-precision snap (added after a pre-run).** Observable changes below float precision are set to exactly 0: |Δm|, |Δm_SD|, |ΔpT| < 0.05 GeV and |Δτ|, |ΔD2| < 1e-4.
- In the pre-run, zboost changed the msd@50 / mass@50 call of 1–2 hls4ml jets sitting on the narrow window edges (chg 0.0001–0.0002), through rounding of a quantity that the edit holds fixed exactly.
- Every network has chg = 0 under zboost on hls4ml, so this column had a near-zero scale. Dividing by that scale inflated it and made the hls4ml bootstrap intervals two-peaked (e.g. msd_V@50 cosine 0.24 [0.22, 0.48]).
- After the snap, those intervals are 0.47 [0.46, 0.49]. Snap counts per edit and observable are stored in `snapped_jets_all_rows`.

### Networks
Each network's call is signal when its task log-odds exceeds the median over unedited signal jets.
- **hls4ml:** JEDI-linear (official n64, 3-feature), Deep Sets d64, Linformer, Transformer, JEDI-style float.
- **JetClass:** ParT_full, ParT_kin, PN_full, PN_kin.
- **Top:** ParT_kin, PN_kin, PFN, EFN, LorentzNet.
- **QG:** part_kin, pnet_kin, pfn_k, efn_k, lorentznet. Descriptive only; there is no shape question for q/g.

### Null
Weight-permuted copies: every weight tensor's entries are shuffled, with seeds from 5800 upward.
- Copies whose unedited log-odds were constant over jets (std < 1e-6) were rejected and further seeds drawn, until 55 copies per architecture were kept.
- Every architecture has its own copies. LorentzNet copies and ParT_kin / PN_kin copies are new compared with H55.
- JEDI-linear needed 223 tries: 168 copies gave constant output.
- In the statistics, a copy whose calls never change under any edit on a task is also dropped for that task. This removed 19–25 of the JEDI copies (33–36 left on W/Z/top, 30 on q/g) and no copies of any other architecture.

### Fingerprint and tests
**Fingerprint.**
- For each row (network, permuted copy or classical tagger) and edit:
  - chg = the fraction of the task's jets (where the edit applies) whose call flips;
  - flow = (calls gained − calls lost) / jets.
- The fingerprint is the 18-component vector (chg_e, flow_e) over the nine edits (14 for QG).
- Each component is divided by its RMS over the trained networks and the @50 classical rows of the task.
- cos = the cosine of the angle between two fingerprints: 1 = same pattern, 0 = unrelated.

**Main contrasts** (pre-specified; network × task, W/Z/top tasks only). The shape tagger is V_τ21@50 for W/Z and top_τ32@50 for top.
- d_shape = cos(net, shape@50) − cos(net, msd@50): does the τ cut add similarity beyond the groomed-mass window?
- d_groom = cos(net, msd@50) − cos(net, mass@50): groomed or ungroomed mass window?
- d_sm = cos(net, shape@50) − cos(net, mass@50) = d_shape + d_groom: the D23 headline contrast.

**Statistics.**
- Paired jet bootstrap, B = 50,000 multinomial resamples, with the column scaling recomputed in every draw.
  - The 95 % CI comes from the same draws.
  - H0: d = 0, two-sided bootstrap p (floor 4e-5).
  - The plan specified B = 1,000. With that B the p floor (2e-3), multiplied by 96 tests, cannot reach Holm < 0.01, so B was raised.
- Holm over all 96 contrast tests (32 network × task cells × 3 contrasts). BH is reported as a secondary correction.
- Null for each contrast: the same contrast computed for every permuted copy of the network's architecture (median, and the fraction of copies with a contrast ≥ the network's).

**Secondary analyses.**
- Cosine to each tagger against the permuted copies: exact rank p = (1 + #{copies ≥ net}) / (n + 1); Holm and BH over 162 cells.
- Best-of-shape contrast: max over V_τ21@50, V_DDT@50, V_D2@50, minus msd@50, with the selection redone in each draw.
- Published-cut analysis (B = 2,000).
- Leave-one-edit-out: one edit's two columns are dropped and the nearest tagger and contrast signs recomputed.
- Unscaled cosine.
- Exact McNemar test on gained versus lost calls, per network × task × edit.

## Unedited models on these samples (reproduction check)
**hls4ml (5-class accuracy):**

| network | accuracy | known |
|---|---|---|
| JEDI-linear | 0.8064 | 0.8092 |
| Deep Sets | 0.8009 | 0.806 |
| Linformer | 0.8057 | 0.811 |
| Transformer | 0.8122 | H55: 0.816 |
| JEDI-style float | 0.7936 | H55: 0.801 |

**JetClass (10-class argmax, per class: QCD / Hbb / Zqq / Wqq / Tbqq):**

| network | QCD | Hbb | Zqq | Wqq | Tbqq |
|---|---|---|---|---|---|
| ParT_full | 0.778 | 0.926 | 0.697 | 0.798 | 0.956 |
| ParT_kin | 0.723 | 0.631 | 0.565 | 0.702 | 0.904 |
| PN_full | 0.763 | 0.911 | 0.663 | 0.776 | 0.950 |
| PN_kin | 0.698 | 0.508 | 0.532 | 0.669 | 0.901 |

All are within 0.02 of the H55 / H18 values (ParT_full known: QCD 0.783, Zqq 0.689, Wqq 0.805, Tbqq 0.956).

**Top Landscape (accuracy, AUC in brackets):**

| network | this sample | known |
|---|---|---|
| ParT_kin | 0.942 (0.986) | 0.940 |
| PN_kin | 0.945 (0.987) | 0.940 |
| PFN | 0.935 (0.981) | 0.928 |
| EFN | 0.923 (0.978) | 0.922 |
| LorentzNet | 0.945 (0.987) | H55: 0.942 |

**QG (accuracy, AUC in brackets):**

| network | this sample | H55 |
|---|---|---|
| part_kin | 0.827 (0.901) | 0.827 |
| pnet_kin | 0.826 | 0.833 |
| pfn_k | 0.815 | 0.818 |
| efn_k | 0.807 | 0.808 |
| lorentznet | 0.841 | 0.851 |

**Permuted copies (accuracy):** hls4ml 0.10–0.32 (JEDI 0.200); JetClass 0.00–0.24; Top 0.18–0.78; QG 0.25–0.77.

## Numbers
### Main contrasts at matched 50 % working points
Each cell gives d [95 % CI], with the Holm p in brackets where it is below 0.05. "≈0" means the CI includes 0.

| task | network | d_shape (τ cut beyond m_SD) | d_groom (m_SD vs m) | d_sm (shape vs m) |
|---|---|---|---|---|
| JetClass top | ParT_full | +0.019 [−0.027, +0.067] | **+0.183 [+0.136, +0.225]** (0.004) | **+0.202** [+0.162, +0.240] (0.004) |
| JetClass top | ParT_kin | +0.034 [−0.029, +0.097] | **+0.103** [+0.043, +0.159] (0.028) | **+0.137** (0.004) |
| JetClass top | PN_full | +0.016 [−0.031, +0.066] | **+0.151** [+0.104, +0.196] (0.004) | **+0.167** (0.004) |
| JetClass top | PN_kin | +0.028 [−0.036, +0.092] | **+0.107** [+0.045, +0.165] (0.038) | **+0.135** (0.004) |
| Top Landscape | ParT_kin | +0.013 [−0.041, +0.073] | +0.107 [+0.048, +0.156] (Holm 0.063) | **+0.120** (0.013) |
| Top Landscape | PN_kin | +0.037 [+0.009, +0.070] (Holm 0.46) | **+0.083** [+0.050, +0.107] (0.013) | **+0.120** (0.004) |
| Top Landscape | LorentzNet | +0.034 [−0.033, +0.105] | +0.097 [+0.028, +0.157] (Holm 0.33) | **+0.131** (0.012) |
| Top Landscape | PFN | +0.137 [+0.033, +0.241] (Holm 0.34) | +0.140 [+0.036, +0.231] (Holm 0.33) | **+0.277** (0.004) |
| Top Landscape | EFN | **+0.274 [+0.210, +0.333]** (0.004) | **−0.142** [−0.201, −0.083] (0.008) | **+0.132** (0.015) |
| JetClass Z | ParT_full | **+0.053 [+0.027, +0.080]** (0.0096) | +0.002 ≈0 | **+0.055** (0.036) |
| JetClass Z | ParT_kin | **+0.056** [+0.028, +0.084] (0.012) | **−0.064** [−0.093, −0.033] (0.008) | −0.008 ≈0 |
| JetClass Z | PN_full | **+0.074** [+0.048, +0.099] (0.004) | −0.023 ≈0 | +0.051 (Holm 0.077) |
| JetClass Z | PN_kin | **+0.061** [+0.028, +0.093] (0.015) | **−0.087** [−0.120, −0.051] (0.004) | −0.026 ≈0 |
| JetClass W | ParT_full | −0.017 [−0.042, +0.008] | +0.003 ≈0 | −0.014 ≈0 |
| JetClass W | ParT_kin | −0.010 ≈0 | −0.060 [−0.094, −0.022] (Holm 0.11) | −0.070 (Holm 0.063) |
| JetClass W | PN_full | −0.004 ≈0 | −0.011 ≈0 | −0.015 ≈0 |
| JetClass W | PN_kin | −0.030 ≈0 | **−0.076** [−0.113, −0.034] (0.020) | **−0.106** (0.004) |
| hls4ml W | Deep Sets | −0.003 ≈0 | **−0.131** [−0.144, −0.118] (0.004) | **−0.135** (0.004) |
| hls4ml W | JEDI-linear | −0.018 [−0.032, −0.004] (Holm 0.45) | **−0.048** [−0.061, −0.035] (0.004) | **−0.066** (0.004) |
| hls4ml W | JEDI-style float | −0.012 ≈0 | **−0.075** (0.004) | **−0.088** (0.004) |
| hls4ml W | Linformer | −0.012 (Holm 1) | +0.016 [+0.005, +0.028] (Holm 0.22) | +0.004 ≈0 |
| hls4ml W | Transformer | **−0.021** [−0.030, −0.012] (0.004) | **+0.048** [+0.039, +0.057] (0.004) | **+0.027** (0.008) |
| hls4ml Z | Deep Sets | +0.009 ≈0 | **−0.149** (0.004) | **−0.140** (0.004) |
| hls4ml Z | JEDI-linear | +0.004 ≈0 | **−0.056** [−0.069, −0.043] (0.004) | **−0.052** [−0.070, −0.034] (0.004) |
| hls4ml Z | JEDI-style float | +0.014 ≈0 | **−0.055** (0.004) | **−0.040** (0.012) |
| hls4ml Z | Linformer | −0.001 ≈0 | +0.018 (Holm 0.19) | +0.017 ≈0 |
| hls4ml Z | Transformer | −0.001 ≈0 | **+0.033** (0.004) | **+0.033** (0.004) |
| hls4ml top | Deep Sets | +0.048 [+0.007, +0.088] (Holm 0.72) | **+0.099** (0.004) | **+0.146** (0.004) |
| hls4ml top | JEDI-linear | −0.002 ≈0 | **+0.071** [+0.039, +0.102] (0.008) | **+0.069** (0.028) |
| hls4ml top | JEDI-style float | +0.052 (Holm 0.95) | **−0.051** (0.017) | +0.001 ≈0 |
| hls4ml top | Linformer | **+0.035** [+0.019, +0.050] (0.005) | **+0.100** (0.004) | **+0.135** (0.004) |
| hls4ml top | Transformer | +0.027 (Holm 0.50) | +0.020 ≈0 | +0.047 (Holm 0.20) |

Counts: 37 of 96 contrast tests have Holm p < 0.01 and 50 have Holm p < 0.05.

### Permuted-copy null for the contrasts (median over copies)
- **JetClass W/Z, d_groom:** −0.07 to −0.20. Untrained-weight networks with the same inputs follow the ungroomed window more than the groomed one. The full-input networks (d_groom ≈ 0) lie above every copy (fraction of copies ≥ net = 0.00). The kin networks lie within the copy distribution (PN_kin W: fraction 0.29; PN_kin Z: 0.60).
- **JetClass top, d_groom:** copy medians +0.016 to +0.023; every trained network lies above all copies (fraction 0.00–0.09).
- **d_shape:** copy medians are −0.03 to +0.05 everywhere.
- **hls4ml d_groom:** copy medians are +0.00 to +0.06. Deep Sets, JEDI-linear and JEDI-style float lie below 93–100 % of their copies on W and Z.

### Nearest tagger at 50 % (fraction of bootstrap draws)
- **Top tasks.** top_τ32@50 is nearest for every JetClass and Top network (0.62–1.00), but msd_top@50 is within 0.01–0.04 for the weaver networks (see d_shape). JEDI-linear on hls4ml top is nearest msd_top@50 (0.58).
- **W/Z tasks.** V_D2@50 is nearest for most networks:
  - JetClass ParT_full and PN_full on W; all four JetClass networks on Z;
  - all hls4ml networks on W except the Transformer (V_DDT@50, 0.70);
  - all hls4ml networks on Z except Deep Sets (mass_V@50, 0.75).
  - JetClass ParT_kin and PN_kin on W are nearest mass_V@50 (0.58, 0.91).
- **Best-of-shape contrast** (V_D2@50 selected in nearly all draws): positive with p < 0.01 for every hls4ml and JetClass W/Z network (+0.014 to +0.144), except the hls4ml W Transformer (+0.003).
  - V_D2 was not the pre-specified shape tagger, and its D2 cut is not a published value. The τ21 result above is the pre-registered test.
- **QG:** part_kin, pnet_kin, pfn_k and lorentznet are nearest LR(N, pT^D) (0.91–1.00); efn_k is nearest LR(girth) (1.00), as in H55.

### Published cuts (secondary)
- With the classical taggers at their published cuts (signal efficiency 0.35–0.93), d_shape is positive for every network and task (+0.04 to +0.46, CIs above 0).
- The nearest tagger is V_D2 or top_τ32 for every network except hls4ml Z, where Deep Sets, JEDI-linear, JEDI-style float, Linformer and Transformer are all nearest mass_V (bootstrap 0.52–0.64), as in H55.
- The τ preference that H55 reported is therefore largely absent once the working point is matched.

### Leave-one-edit-out
- **Top tasks:** dropping any one of the nine edits leaves the nearest tagger unchanged in 0–2 of 9 cases. The sign of d_sm does not flip for any JetClass or Top network (0 of 9), and d_groom flips at most once.
- **JetClass W:** the nearest tagger changes in 3 of 9 drops, and d_sm flips in 1–3.
- **hls4ml W/Z:** the nearest tagger changes in 2–3 of 9 drops. The sign of d_groom is stable for Deep Sets (0 flips) and JEDI-linear / JEDI-style float (0–1).

### Other families
- **Cosine above the permuted-copy null (rank p):** the floor is 1/56 = 0.018 (JEDI 1/34 – 1/37). No cell passes Holm over 162 cells (minimum Holm 1.0); 125 of 162 have BH q < 0.05. Passing Holm at this number of cells would need about 3,000 copies per architecture. The main contrasts do not depend on this test.
- **McNemar on call flow:** 289 of 368 network × task × edit tests have Holm p < 0.05.

## Reading
- **Competing hypotheses for each network × task.** The three contrasts sort the possibilities as follows:
  - **H_shape** (the network responds like an m_SD + τ tagger) predicts d_shape > 0.
  - **H_groom** (it responds like a groomed-mass window alone) predicts d_shape ≈ 0 and d_groom > 0.
  - **H_mass** (it responds like an ungroomed-mass window) predicts d_groom < 0 and d_sm < 0.
- **What each hypothesis matches:**

  | hypothesis | cells it matches |
  |---|---|
  | H_groom | JetClass top (4/4), Top Landscape ParT_kin / PN_kin / LorentzNet, hls4ml top Deep Sets / JEDI-linear / Linformer. H_shape is not excluded there; d_shape CIs reach +0.06 to +0.10. |
  | H_shape | JetClass Z (4/4), Top EFN, hls4ml top Linformer |
  | H_mass | hls4ml W and Z for Deep Sets, JEDI-linear and JEDI-style float; JetClass W PN_kin; JetClass Z ParT_kin and PN_kin (d_groom < 0, but d_shape > 0 there) |

- **Exclusions after Holm** (Holm < 0.01, unless a cell gives other values):
  - H_mass is excluded for JetClass top and the Top weaver networks; on the Top Landscape, ParT_kin reaches Holm 0.013 and LorentzNet 0.012.
  - H_groom over H_mass is excluded for the three hls4ml W/Z networks listed above.
  - For JetClass W, none of the three hypotheses is separated for ParT_full and PN_full.
- **D23 predictions written before the run:**
  - "JetClass/Top: d_shape > 0, Holm < 0.01, stable under leave-one-out" holds for JetClass Z ParT_full and PN_full, and for Top EFN. For ParT_kin and PN_kin on JetClass Z, d_shape > 0 has Holm 0.012 and 0.015.
  - It does not hold for JetClass W, JetClass top or the Top ParT / PN / LorentzNet / PFN networks.
  - For top tasks the outcome is the alternative written before the run: "the m_SD-only tagger matches as well (difference CI includes 0) and D23 reduces to groomed vs ungroomed mass".
  - "hls4ml: mass window ≥ m_SD + shape" holds for Deep Sets, JEDI-linear and JEDI-style float on W and Z (d_sm < 0, Holm ≤ 0.012). It fails for the Transformer (d_sm > 0) and on hls4ml top (d_sm > 0 for Deep Sets, JEDI-linear and Linformer).
- **Why the working point matters.**
  - At published cuts, the mass windows keep 70–93 % of the signal, against 35–70 % for the shape taggers. A wide window has few jets near its edges and so changes fewer calls under edits.
  - Once all rows sit at 50 %, the narrow mass windows respond to soft (mass_V@50 chg 0.21–0.23 on hls4ml) and to gin/gout.
  - The network fingerprints' similarity to them rises accordingly.
- **Edit-level picture behind d_groom.**
  - Under gout (soft particles groomed away), the networks change 15–27 % (W/Z) or 2–4 % (top) of their calls. msd@50 changes ≈ 0 % and mass@50 changes 20–30 %.
  - Under gin, all three change calls.
  - On JetClass and Top top tasks the networks' gout/gin ratio is 0.4–0.6, nearer the groomed window (0) than the ungroomed one (1).
  - On hls4ml W/Z the ratio is 1.0: these networks respond equally to particles soft drop keeps and removes. This is H49's ungroomed-mass result.

## Issues and limitations
- **Snap.** The numerical-precision snap was added after looking at the first stats output (the hls4ml two-peaked bootstrap). It changes only calls caused by sub-0.05 GeV rounding; the effect is stated above. Before the snap, the JetClass contrasts were within 0.003 of the values reported. The pre-snap run had no LorentzNet row on Top, so Top values shifted by up to 0.017 (EFN d_shape 0.257 → 0.274) through the changed column scaling.
- **Bootstrap size.** B was raised from the planned 1,000 to 50,000 so that the bootstrap p can pass Holm over 96 tests; the CIs are equivalent.
- **Choice of shape tagger.** The pre-specified shape tagger is the CMS τ21 (τ32 for top). V_D2@50 is often nearer, but its cut is a sample median, not a published value, and choosing it is post hoc (the best-of-shape p includes that selection).
- **hls4ml windows.** The hls4ml m_SD@50 / m@50 windows are only about 3.5 GeV wide (77.8–81.2 GeV for W), because hls4ml jet masses peak sharply. This makes these taggers respond strongly to small mass changes (soft, gin).
- **JEDI-linear null.** 75 % of JEDI-linear permuted copies give constant output, so its null pool is 30–36 copies per task (rank p floor about 0.03).
- **The fingerprint metric.**
  - Standardisation uses the trained and @50 classical rows. Unscaled cosines give the same signs for d_sm on all JetClass and Top top cells, but the opposite sign for hls4ml W/Z d_sm (raw −0.09 to −0.20 for all five networks, including Linformer and Transformer).
  - The metric is therefore sensitive to column weighting where the contrasts are small.
- **Edits that apply to only part of the sample.** The hls4ml count edits exclude 23–25 % of jets; gin/gout are not defined for QG.
- **Hbb.** Hbb vs QCD has no classical tagger; its scores are stored but not analysed.

## Explainer entry (site/explain.py, H58_EXPLAINER)
1. **Question.** When a network's calls change under the same physics edits as a classical tagger's, which tagger is it most like: a groomed-mass window, the same window plus an N-subjettiness cut, or an ungroomed mass window? And is the answer the same once every tagger keeps the same fraction of signal?
2. **Physics background.**
   - Soft drop (β = 0, z_cut = 0.1) removes soft wide-angle branches. The groomed mass $m_{SD}$ is therefore insensitive to soft radiation that the ungroomed mass $m$ picks up.
   - $\tau_{21}=\tau_2/\tau_1$ and $\tau_{32}=\tau_3/\tau_2$ measure how two-prong or three-prong a jet is.
   - A tagger's working point is the cut that defines its yes/no call. A wide window keeps most signal and rarely changes its call under a small edit, so comparisons between taggers must use the same signal efficiency.
3. **Method.**
   - Nine edits (collinear splits, a soft particle, an axis change and its random-direction control, soft particles kept or removed by grooming, a decay-angle rotation at fixed four-momentum, a beam boost) are applied to 15,000 hls4ml, 15,000 JetClass, 6,000 Top and 6,000 QG jets.
   - Fingerprint $f=(\mathrm{chg}_e,\mathrm{flow}_e)_e$, with each component divided by its RMS. Similarity $\cos(f_a,f_b)=f_a\cdot f_b/(|f_a||f_b|)$.
   - Contrasts:
     - $d_{shape}=\cos(f_{net},f_{m_{SD}+\tau})-\cos(f_{net},f_{m_{SD}})$
     - $d_{groom}=\cos(f_{net},f_{m_{SD}})-\cos(f_{net},f_{m})$
   - Every tagger is set to 50 % signal efficiency. Paired jet bootstrap, $B=5\times10^4$; Holm over 96 tests.
   - Null: 55 weight-permuted copies per architecture.
4. **Result (figure `h58_contrasts.png`).**
   - Top taggers: $d_{groom}=+0.10$ to $+0.18$ on JetClass. $d_{shape}$ is consistent with 0, except for the Top EFN at $+0.27$.
   - JetClass Z: $d_{shape}=+0.05$ to $+0.07$.
   - hls4ml W/Z: $d_{groom}<0$ for Deep Sets, JEDI-linear and JEDI-style float, $>0$ for the Transformer.
   - At published cuts every network looks closer to the shape taggers.
5. **Checks.**
   - Unedited accuracies reproduce the known values to within 0.02.
   - IRC-safe taggers change 0 calls under splits; the mass window changes 0 under the decay rotation.
   - Leave-one-edit-out leaves the top-task signs unchanged.
   - Rows are disjoint from H55 / H55v.
6. **Physical significance.**
   - Trained top taggers respond to edits like a groomed-mass window: they ignore soft particles that soft drop removes. Beyond that window, the τ32 cut adds no measurable similarity.
   - Several hls4ml W/Z networks respond like an ungroomed mass window instead.

**Glossary additions:**
- **Matched working point (@50):** every tagger's cut is set so that it keeps 50 % of signal jets.
- **d_shape, d_groom, d_sm:** differences of cosine similarities of call-change fingerprints (defined above).
- **Weight-permuted copy:** a network with each weight tensor's entries shuffled, used as a null.
- **Rank p:** (1 + number of null copies at least as extreme) / (copies + 1).

## Proposed claim update
D23 (0.3 → narrowed): at matched 50 % signal efficiency, under the nine-edit library:
- JetClass top taggers (4/4) are closer to a soft-drop-mass window than to an ungroomed-mass window, by +0.10 to +0.18 in cosine (Holm < 0.04). Top Landscape PN, ParT and LorentzNet are closer by +0.08 to +0.11; PN passes Holm (0.013), ParT_kin and LorentzNet do not (Holm 0.063 and 0.33), and their d_sm (τ32 tagger vs ungroomed window) has Holm 0.013 and 0.012.
- The τ32 cut adds no similarity beyond the m_SD window, except for the EFN.
- JetClass Z taggers (4/4) are closer to m_SD + τ21 than to m_SD alone (+0.05 to +0.07; Holm < 0.02).
- hls4ml W/Z networks split by architecture: Deep Sets, JEDI-linear and JEDI-style float are closer to the ungroomed mass window; the Transformer is closer to the groomed one.

Suggested confidence: 0.55 for the top-tagger groomed-mass statement; the "hls4ml as a group follows a plain mass window" part is not supported.
