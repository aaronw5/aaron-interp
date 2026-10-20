# P22: What in training produces ParT's axis response (claims C10, C3s, C3)

Status: **done** (round C6). Analysis job `anrunw-ai-c-p22-run` (log `logs/c-p22-run.log`, 1,321 s on one GPU). Training jobs `anrunw-ai-c-p22-t/-t2/-u/-a2` are Complete.
Results: `results/part/p22_planted_true_run.json` and figure `results/part/p22_planted_true_run.png`.
Script: `interp/p22_planted_true_part.py`, run with `--per 10000 --tag run`. Training: `interp/p22_planted_true_train.py`, described in the round C5 version of this report.

## Conclusion

Conclusion: we trained Particle Transformers on 1.5 million W, Z and QCD jets with the true labels. They respond to a rigid shift of the jet about its stored axis only about 0.2–0.25 times as strongly as to a real mass change of the same axis-moment size. This is the same level as networks trained on a mass-window label, and far below the 0.72–0.79 of the released JetClass models. So at this training scale, the true labels on stored-axis inputs do not reproduce the released response; the pre-registered thesis range (0.6–0.8) is excluded. Training the same network with random axis offsets lowers the ratio to 0.01 with no loss in accuracy. This shows that the smaller response which does appear is learned from the stored-axis inputs and is not imposed by the architecture. These runs cannot tell which of the released training's differences sets the larger response: 100M jets, 10 classes, or the longer schedule.

## Pre-registered predictions and outcome

R is the per-jet least-squares slope (through 0) of Δd(φ + 0.06) on Δd(matched dilation). d is the W-vs-Z read-out: ℓ_W − ℓ_Z for 3- and 10-output models, and ℓ_in − ℓ_out for the 2-output window arms. The interval is a 95 % bootstrap over jets (2,000 resamples).

| hypothesis | prediction | observed | status |
|---|---|---|---|
| Thesis: true labels on stored-axis inputs are enough | R_t, R_t2 in 0.6–0.8; R_u < 0.3 with the dilation response kept and accuracy within 1 pt | R_t 0.219, R_t2 0.254; R_u 0.009; dilation Δd u/t = 0.99; accuracy u − t −0.07 pt | R_t part excluded; R_u part holds |
| Alt 1: the released R needs 10 classes or data scale | R_t ≈ 0.2, like the window arms | R_t 0.219 / 0.254 vs R_a 0.185 / R_a2 0.269 | consistent |
| Alt 2: architecture or inputs force the response | R_u ≈ R_t | R_u − R_t = −0.210, permutation p_Holm 0.003 | excluded |

## Procedure

- **Sample:** fresh rows of JetClass test_20M/{ZJetsToNuNu, ZToQQ, WToQQ}_104.root, 10,000 random rows per file (numpy default_rng(2204 + class index), sorted), N = 30,000. The sample is rebuilt unedited through `part_core.build`, the same path as every edit.
- **Models:**
  - t: true labels, seed 14.
  - t2: true labels, seed 15.
  - u: true labels, with each training jet given a random rigid (Δη, Δφ) offset, |c| ~ U(0, 0.08) rad.
  - a2: 1[|m_inv − 85| < 10 GeV] label, seed 15.
  - P14 arm a: the same window label, seed 14.
  - Released ParT_kin and ParT_full (10 outputs).
  - Four untrained 3-output ParT_kin initialisations (torch seeds 100–103) as the floor.
  - All planted arms: ParT_kin inputs, 1.5M jets, 10 epochs, the P14 code path.
- **Edits:**
  - φ: every real particle's Δφ + 0.06 rad about the stored axis. Δlog m_inv = 0.0000 and Δlog m_A = +0.126 (sample mean), where m_A = √(M0·M2) is the axis moment.
  - Matched dilation: (Δη, Δφ) × k_j per jet, with k_j giving the same Δlog m_A. Δlog m_inv = +0.122 and Δlog m_A = +0.126.
  - ×1.1: all angles scaled by 1.1 (the P2 reference real change).
  - Rotation null: the whole jet, axis included, rotated 0.06 rad in φ. The inputs are identical up to float rounding.
- **Metrics:**
  - R with bootstrap CI, its rotation null, and R on true-W jets only.
  - W → Z: the fraction of true-W jets called Z (argmax over QCD/Z/W) before and after the edit, with a bootstrap CI and McNemar on the paired Z-call indicator (Holm over all models × edits).
  - Route shares (P2 method): the response with only the per-particle inputs edited (pair four-vectors kept), and with only the pair four-vectors edited. Each is reported as a slope on the full-edit response.
  - 1-d DAS (P8 method) at the output of particle blocks 2 and 8 on W/Z jets. The call is argmax(ℓ_Z, ℓ_W). The targets are 1[m_inv < 85.8 GeV] and 1[m_A < 85.15 GeV] (matched rate). The control is a random direction.
- **Tests:**
  - Permutation tests that swap the two models' per-jet (Δd_φ, Δd_dil) pairs (2,000 permutations), H0: R_x = R_y. Pairs: (t, a), (u, t), (t, t2), (a, a2), (t, released_kin), (u, init0), with Holm correction over the six.
  - McNemar on the 3-class correctness of t vs u.

## Numbers

**Accuracy on this sample**

| model | 3-class acc (W/Z/QCD) [Wilson 95 %] | other |
|---|---|---|
| t | 0.7487 [0.7438, 0.7536] | – |
| t2 | 0.7485 [0.7436, 0.7534] | – |
| u | 0.7480 [0.7431, 0.7529] | u − t −0.0007 [−0.0031, +0.0016], McNemar p 0.58 (669 vs 648 discordant) |
| a / a2 | own window label 0.9830 / 0.9869 | – |
| released ParT_kin | 0.7640 (argmax over 3 of 10 logits) | 10-class 0.661 here; P1-sample 10-class **0.7447** (P1: 0.745) |
| released ParT_full | 0.8277 | 10-class 0.760 here; P1-sample 10-class **0.8593** (P1: 0.859) |
| untrained init0–3 | 0.307–0.351 | – |

**Axis response (N = 30,000)**

| model | R [95 % CI] | R on W jets | rotation null R | W → Z before → after φ (change [CI]) | W → Z after dilation | mean Δd dilation |
|---|---|---|---|---|---|---|
| t | 0.219 [0.211, 0.226] | 0.196 | −0.0002 | 18.9 → 25.4 % (+6.5 [5.9, 7.0]) | 49.0 % | −0.467 |
| t2 | 0.254 [0.246, 0.262] | 0.230 | −0.0003 | 20.5 → 28.1 % (+7.6 [6.9, 8.2]) | 51.1 % | −0.453 |
| u | 0.009 [0.007, 0.011] | 0.006 | +0.0004 | 18.6 → 18.9 % (+0.4 [0.1, 0.6]), McNemar p_Holm 0.14 | 48.7 % | −0.463 |
| a (window, s14) | 0.185 [0.181, 0.188] | – | +0.0001 | in-window call 41.5 → 41.0 % | – | −1.157 |
| a2 (window, s15) | 0.269 [0.265, 0.273] | – | −0.0001 | in-window call 41.3 → 40.6 % | – | −1.336 |
| released ParT_kin | 0.722 [0.713, 0.730] | 0.719 | +0.006 [0.003, 0.009] | 18.6 → 41.4 % (+22.8 [21.9, 23.7]) | 48.7 % | −0.495 |
| released ParT_full | 0.791 [0.784, 0.799] | 0.793 | 0.0000 | 11.0 → 32.3 % (+21.3 [20.5, 22.2]) | 37.1 % | −0.627 |
| untrained init0–3 | −0.57 to +0.53 | – | ≈ 0 | – | – | |Δd| 0.04–0.07 |

The untrained networks' mean |Δd| under dilation is 0.04–0.07, against 0.63 for t, so their ratio R is a ratio of small numbers and is not a calibrated floor.

**Permutation tests on R** (2,000 permutations; the minimum attainable p is 0.0005)

| H0 | R_x − R_y | p | p_Holm (6 tests) |
|---|---|---|---|
| R_t = R_a | +0.034 | 0.0005 | 0.003 |
| R_u = R_t | −0.210 | 0.0005 | 0.003 |
| R_t = R_t2 | −0.035 | 0.0005 | 0.003 |
| R_a = R_a2 | −0.085 | 0.0005 | 0.003 |
| R_t = R_released_kin | −0.503 | 0.0005 | 0.003 |
| R_u = R_init0 | +0.049 | 0.0005 | 0.003 |

The seed-to-seed differences (t vs t2: 0.035; a vs a2: 0.085) are resolved by the test. They are 7–17 % of the t vs released_kin gap.

**Route shares of a real mass change** (slope on the full-edit response, CI width ≤ 0.01)

| model | dilation: per-particle only | dilation: pair only | ×1.1: per-particle only | ×1.1: pair only | additivity R² |
|---|---|---|---|---|---|
| t | 0.299 | 0.686 | 0.331 | 0.673 | 0.91 |
| t2 | 0.314 | 0.672 | 0.350 | 0.654 | 0.90 |
| u | 0.099 | 0.888 | 0.120 | 0.875 | 0.97–0.98 |
| a | 0.174 | 0.826 | 0.207 | 0.809 | 0.94–0.95 |
| a2 | 0.217 | 0.788 | 0.251 | 0.773 | 0.92–0.93 |
| released ParT_kin | 0.828 | 0.172 | 0.845 | 0.164 | 0.95–0.97 |
| released ParT_full | 0.839 | 0.151 | 0.850 | 0.145 | 0.93–0.96 |
| untrained | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |

**DAS** (1-d, W/Z test jets; IIA against the m_inv target; random-direction control in brackets)

| model | block 2 | block 8 |
|---|---|---|
| t | 0.611 (0.496) | 0.711 (0.493) |
| t2 | 0.608 (0.502) | 0.708 (0.495) |
| u | 0.572 (0.501) | 0.681 (0.497) |
| a | 0.566 (0.513) | 0.774 (0.518) |
| a2 | 0.580 (0.522) | 0.752 (0.515) |
| released ParT_kin | 0.651 (0.498) | 0.738 (0.499) |

The m_inv and m_A targets disagree on only 1.9 % of W/Z jets. The directions trained for the two targets have |cos| = 0.99–1.00, so DAS here locates a W/Z mass direction but does not separate m_inv from m_A. The IIA on the disagreeing pairs (0.44–0.65, about 170 pairs) is at chance for every model.

## Reading

- With the same data (1.5M jets), schedule and architecture, changing the label from a mass window to the true W/Z/QCD class changes R by +0.03 (t vs a). The two seeds of each label differ by 0.035–0.085. Training with random axis offsets changes R by −0.21 (u vs t) at unchanged accuracy, and leaves the response to a real dilation at 99 % of t's.
- The shift response in t and t2 is measurable in the calls: W → Z rises by 6.5–7.6 points, against 22–23 points for the released models.
- Route shares follow R. The planted arms take 66–89 % of a real mass change through the pair features, and the released models 15–17 %. Arm u has the lowest per-particle share (0.10). Within the pair route, the pair features (ln kT, ln z, ln ΔR, ln m²) do not depend on the stored axis. This matches R_u ≈ 0: a network that reads mass through the pair route is not moved by a rigid shift of the axis.
- What is not tested here: which difference between the planted runs and the released training (100M vs 1.5M jets, 10 vs 3 classes, a much larger number of optimisation steps) raises the per-particle share and R to the released level. The P26 Sophon run (JetClass-II, 188 classes) and a 10-class or larger-data planted arm would separate these.
- Released ParT_kin's rotation null is +0.006 [0.003, 0.009], which is 0.8 % of its R. This is consistent with float32 rounding in the rotated px, py.

## Explainer entry (for site/explain.py)
- **Question.** The released Particle Transformers change their W/Z call when the whole jet is shifted rigidly about its stored axis, which leaves the invariant mass unchanged. Is this learned from the true class labels on inputs measured relative to that axis, or is it built into the architecture?
- **Background.**
  - ParT's per-particle inputs include $\Delta\eta_i$, $\Delta\phi_i$ and $\Delta R_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$ about the stored jet axis. Its pair inputs ($\ln k_T$, $\ln z$, $\ln\Delta R_{ij}$, $\ln m_{ij}^2$) do not depend on the axis.
  - The axis moment is $m_A=\sqrt{M_0M_2}$ with $M_0=\sum_i p_{T,i}$ and $M_2=\sum_i p_{T,i}\,\Delta R_i^2$. A rigid shift $\Delta\phi_i\to\Delta\phi_i+0.06$ raises $m_A$ by about 13 % and leaves $m_{inv}$ unchanged.
- **Method.**
  - Train ParT_kin on 1.5M W/Z/QCD jets with the true labels (two seeds), with the true labels plus a random rigid offset of each training jet, and with a mass-window label (two seeds).
  - Measure $R=\dfrac{\sum_j \Delta d_j^{\phi}\,\Delta d_j^{dil}}{\sum_j (\Delta d_j^{dil})^2}$, where $d=\ell_W-\ell_Z$. $\Delta d^{dil}$ comes from a per-jet dilation $(\Delta\eta,\Delta\phi)\to k_j(\Delta\eta,\Delta\phi)$ with the same $\Delta\log m_A$ as the shift.
  - Statistics: bootstrap over jets for the intervals, and permutation tests (per-jet swap of the two models' responses) for differences between arms.
  - Route shares: edit only the per-particle inputs, or only the pair four-vectors, and regress each response on the full response.
- **Result (figure `results/part/p22_planted_true_run.png`).** True-label arms give $R=0.22$ and $0.25$, the window arms $0.18$ and $0.27$, and the offset-trained arm $0.009$. The released models give $0.72$ (kin) and $0.79$ (full). The offset arm's accuracy matches the true-label arm (74.8 % vs 74.9 %).
- **Checks.**
  - The released models reproduce 74.47 % / 85.93 % 10-class accuracy on the P1 sample.
  - A whole-jet rotation, axis included, gives $R\le0.006$.
  - The offset arm keeps 99 % of the true-label arm's response to a real dilation.
  - Untrained networks respond about 15× less to the dilation, so their ratio is not used as a floor.
- **Why it matters.** At this training scale, the label is not what makes ParT's mass read-out depend on the axis as strongly as in the released models. The axis dependence that does appear is removed by offset augmentation at no cost in accuracy. The released models' larger response therefore comes from some other part of their training: more data, more classes or a longer schedule.
