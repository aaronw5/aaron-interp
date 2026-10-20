# P22 verification: planted ParT_kin arms, axis response R

Verifier job `anrunw-ai-c-p22-verify` (Complete, exit 0; smoke run on 900 jets followed by the full run, 418 s on a GTX 1080 Ti). Script `interp/p22_verify.py`. Results `results/part/p22_verify.json` (smoke run: `p22_verify_smoke.json`). Figure `results/part/p22_verify.png`. Log `logs/c-p22-verify.log`.

## Conclusion

Conclusion: on a fresh set of jets, the P22 numbers reproduce. Networks trained on 1.5 million W, Z and QCD jets with the true labels respond to a +0.06 rad φ shift about the stored jet axis about 0.22–0.25 times as strongly as to a matched real mass change. The same networks respond about twice as strongly (0.47–0.49) to the same shift in the −φ direction; the released models respond equally in both directions. So the planted value depends on the shift direction, but in every direction tested it stays below the released 0.73–0.80 and below the pre-registered 0.6–0.8. Training with random axis offsets brings it to about 0.01 in all directions. Over the 10 training epochs the planted response does not move toward the released level. Which difference in the released training sets the larger response is still untested.

## What was checked in the code (interp/p22_planted_true_part.py and its imports)

- **Row selection:** random rows per class file (`default_rng(2204 + class)`), with no first-k selection. The P1v `split` used for the probe is a random permutation.
- **Edits:**
  - Only real particles are shifted (`dp + 0.06 * real`). The padded slots are re-zeroed by `part_core.build`.
  - The η convention is `jet_eta + sign(jet_eta)·Δη` in both `build` and the training augmentation `offset`, so there is no η-mirror bug.
  - The φ edit leaves m_inv unchanged (Δlog m_inv 6e-9) and raises m_A by Δlog 0.126. The matched dilation gives Δlog m_A 0.126.
  - The rotation null leaves the inputs equal up to rounding.
- **Inputs:** training and analysis compute features with the same `jetdistill.part.network.inputs` ('kin'). The analysis feeds rebuilt four-vectors, which `build` reproduces to 1.2e-7.
- **Read-out:** ℓ_W − ℓ_Z uses columns [1, 2] of the 3-output arms (class order QCD 0, Z 1, W 2, as in `CLS3`) and [iZ, iW] of the released 10-output models.
- **DAS sites:** `sites=(1, 7)` are particle blocks 2 and 8 (0-based). The hook is on `blocks[site]` with the token mask applied.
- **Training check:** arm u's augmentation log reports a maximum |Δm_inv|/m_inv of 9e-4 under the offsets (η-shift versus rapidity), with a mean Δlog m_A of +0.118. Arms t and t2 were still improving at epoch 9 (val accuracy 0.741–0.743, train loss 0.59 and falling).
- No indexing, hook or row-selection bug was found.

## Re-run on fresh rows, with added shift directions and training epochs

- **Sample:** test_20M/{ZJetsToNuNu, ZToQQ, WToQQ}_105.root (P22 used _104). 5,000 random rows per file (`default_rng(3105 + class)`), N = 15,000, rebuilt unedited through `part_core.build`.
- **Models:**
  - P22 arms t, t2, u and a2, and P14 arm a.
  - Released ParT_kin and ParT_full.
  - Arms t and t2 at epochs 0, 2, 4, 6 and 8 (`epoch-<e>.pt`); `final.pt` is epoch 9.
- **Edits:**
  - φ + 0.06 rad, as in P22.
  - φ − 0.06 rad (added).
  - η + 0.06 (added): every real particle's Δη + 0.06, which is a rigid shift in η.
  - Each shift has its own per-jet matched dilation, k_j = √(M0M2(shift)/M0M2(base)).
  - Rotation null, as in P22.
- **Δlog m_A / Δlog m_inv:** φ± 0.127 / 0.000; η+ 0.115 / 0.000. The matched dilations give Δlog m_A 0.126 (φ±) and 0.113 (η).
- **Metrics:** two forms of R, the route shares of the φ+ matched dilation (P2 method), and W → Z on true-W jets under φ + 0.06.
  - R as in P22: the per-jet slope through 0 of Δd(shift) on Δd(matched dilation).
  - R as a ratio of means.
- **Statistics:**
  - 95 % bootstrap over jets (2,000 resamples).
  - Permutation tests (per-jet swap, 2,000) for (u, t), (t, a), (t, released_kin), (t epoch 0, t final) and (t2 epoch 0, t2 final), with Holm over the five.
  - McNemar on the paired Z-call indicator, with Holm over the 15 models and checkpoints.

**Accuracy on this sample (unedited):**
- 3-class: t 0.7469, t2 0.7471, u 0.7479, released_kin 0.7637, released_full 0.8229.
- Own window label: a 0.9836, a2 0.9871.
- Released 10-class: kin 0.6615, full 0.7581 (W/Z/QCD-only sample). The P1-sample check, 0.7447 / 0.8593, was done in P22 and not repeated.

**R [95 % CI] by shift direction (N = 15,000):**

| model | φ + 0.06 | φ − 0.06 | η + 0.06 | ratio of means, φ+ / φ− | rotation null |
|---|---|---|---|---|---|
| t | 0.221 [0.212, 0.230] | 0.470 [0.457, 0.484] | 0.206 [0.196, 0.217] | 0.265 / 0.473 | +0.0003 |
| t2 | 0.251 [0.240, 0.261] | 0.491 [0.477, 0.505] | 0.216 [0.205, 0.227] | 0.277 / 0.484 | −0.0002 |
| u | 0.010 [0.008, 0.012] | 0.004 [0.002, 0.006] | 0.019 [0.016, 0.022] | 0.001 / 0.001 | +0.0003 |
| a (window, s14) | 0.180 [0.175, 0.185] | 0.414 [0.407, 0.420] | 0.213 [0.207, 0.218] | 0.101 / 0.336 | +0.0001 |
| a2 (window, s15) | 0.263 [0.257, 0.269] | 0.437 [0.430, 0.444] | 0.263 [0.257, 0.270] | 0.198 / 0.427 | −0.0002 |
| released_kin | 0.729 [0.720, 0.740] | 0.741 [0.729, 0.752] | 0.860 [0.848, 0.873] | 0.677 / 0.706 | +0.0076 |
| released_full | 0.796 [0.785, 0.805] | 0.770 [0.760, 0.780] | 0.795 [0.784, 0.807] | 0.805 / 0.779 | +0.0001 |

The φ+ column reproduces P22 within 0.005 for every model (P22: t 0.219, t2 0.254, u 0.009, a 0.185, a2 0.269, released 0.722 / 0.791).

The φ− versus φ+ difference was not permutation-tested. For t it is +0.249, against 95 % half-widths of 0.009 and 0.014. Under a normal approximation that treats the two as independent, this is z ≈ 30 (p < 1e-10). For the released models the difference is +0.012 (kin) and −0.026 (full).

**Other quantities reproduced from P22:**
- W → Z on true-W jets under φ + 0.06:
  - t 19.0 → 24.9 % (+6.0 [5.2, 6.8], p_Holm 2e-49).
  - u 18.4 → 18.7 % (+0.3 [−0.0, +0.7], p_Holm 0.10).
  - released_kin 17.9 → 42.0 % (+24.2 [22.9, 25.5]).
  - released_full 11.2 → 32.5 % (+21.3 [20.1, 22.5]).
- Mean |Δd| under the φ+ dilation: t 0.635, u 0.628 (u keeps 99 % of t's response).
- Per-particle (pair) route share of the φ+ dilation:
  - t 0.296 (0.680), t2 0.309 (0.671), u 0.096 (0.886).
  - a 0.173 (0.826), a2 0.216 (0.787).
  - released_kin 0.826 (0.170), released_full 0.842 (0.149).
- Permutation tests on R(φ+), Holm over five, all p 0.0005, p_Holm 0.0025:
  - u − t −0.211.
  - t − a +0.041.
  - t − released_kin −0.508.

**R across training (arms t / t2):**

| epoch | R φ+ (t / t2) | R φ− (t / t2) | R η+ (t / t2) | 3-class acc (t / t2) | mean |Δd| dilation (t / t2) |
|---|---|---|---|---|---|
| 0 | 0.139 / 0.067 | 0.734 / 0.933 | −0.190 / −0.013 | 0.693 / 0.687 | 0.44 / 0.43 |
| 2 | 0.108 / 0.197 | 0.692 / 0.679 | 0.141 / 0.039 | 0.733 / 0.734 | 0.53 / 0.51 |
| 4 | 0.231 / 0.172 | 0.510 / 0.587 | 0.167 / 0.168 | 0.735 / 0.735 | 0.56 / 0.55 |
| 6 | 0.209 / 0.195 | 0.536 / 0.583 | 0.222 / 0.282 | 0.738 / 0.740 | 0.55 / 0.58 |
| 8 | 0.223 / 0.237 | 0.457 / 0.500 | 0.207 / 0.216 | 0.747 / 0.746 | 0.62 / 0.63 |
| 9 (final) | 0.221 / 0.251 | 0.470 / 0.491 | 0.206 / 0.216 | 0.747 / 0.747 | 0.64 / 0.63 |

- R(φ+) rises from epoch 0 to 9: +0.082 for t and +0.184 for t2 (permutation p_Holm 0.0025 for each).
- R(φ−) falls: from 0.73 to 0.47 for t, and from 0.93 to 0.49 for t2.
- The mean of the two φ directions goes from 0.44 to 0.35 (t) and from 0.50 to 0.37 (t2).
- In no direction does R move toward the released 0.73–0.80 over the last six epochs.

## Assessment of the P22 reading

1. **Direction dependence (not in P22).** The planted arms' R depends on the sign of the φ shift: 0.18–0.26 for φ+ and 0.41–0.49 for φ−. The released models are within 0.03 of each other in the two directions.
   - The P22 statements "R_t 0.22 / 0.25" and "Alt 1 prediction (R_t ≈ 0.2, like the window arms) matches" hold for the pre-registered +φ edit only.
   - The conclusion's "only about 0.2–0.25 times as strongly" should give the range over directions: 0.2–0.5.
   - What survives in every direction:
     - t ≈ a (differences +0.04, +0.06 and −0.01 for φ+, φ− and η+).
     - u ≈ 0 (≤ 0.02).
     - All planted arms below the pre-registered 0.6–0.8 and below the released models.
2. **"Route shares track R."** This holds for the split between planted and released models and for u. Within the planted arms the order does not match: a2 has a higher R(φ+) than t (0.263 vs 0.221) but a lower per-particle share (0.22 vs 0.30). A scoped statement: models with a per-particle share ≥ 0.8 have R ≥ 0.7; models with a share ≤ 0.31 have R ≤ 0.49 in every direction.
3. **The "many more optimisation steps" alternative.** Within 10 epochs at fixed data, R does not approach the released value. R(φ+) rises by 0.08–0.18 and R(φ−) falls by 0.26–0.44. This does not exclude an effect of the ~55× more steps in the released training, but it gives no trend toward it.
4. **Controls.**
   - The rotation null only checks numerics (its inputs are identical). The real control is arm u, which is adequate: same data, seed and schedule, same accuracy, 99 % of the dilation response.
   - The untrained-init floor is correctly set aside in P22.
   - The bootstrap intervals cover jet sampling only. Seed-to-seed spread (0.02–0.08) is larger than the interval widths, so arm comparisons should be read against the two-seed spread. P22 states this.
5. **Unreported probe read-out.** The P22 JSON contains R_probe (the log m_inv ridge probe on the class token): t 0.04, t2 0.11, released_kin 1.43, released_full 0.79. It disagrees with the logit R for t and released_kin, so it is not a calibrated read-out of the axis response. The P22 reading does not use it, which is correct.
6. **Figure (`p22_planted_true_run.png`).** The bars match the JSON. Two labelling issues:
   - The title says "ParT_kin by training label" but includes released ParT_full.
   - The dotted line at R = 1 is not explained.
   - The figure shows only the +φ direction, so it should say that R is direction-dependent for the planted arms (see `p22_verify.png`).
7. **DAS.** Not re-run. P22's caveat (m_inv and m_A targets agree on 98 % of W/Z jets; |cos| 0.99) is correct, so the DAS rows support only "a W/Z mass direction exists at block 8".

## Corrected reading

On the same 1.5M jets, schedule and architecture:
- **Label (t vs a):** training on the true W/Z/QCD class instead of a mass-window label changes R by −0.01 to +0.06 depending on the shift direction.
- **Direction:** the planted arms' R depends on the shift direction: 0.18–0.26 for φ + 0.06, 0.41–0.49 for φ − 0.06, and 0.21–0.26 for η + 0.06.
- **Released models:** they give 0.73–0.86 in all three directions.
- **Pre-registered thesis:** the range for R_t (0.6–0.8) is excluded in every direction tested.
- **Augmentation:** random axis-offset training lowers R to ≤ 0.02 in all directions, at unchanged accuracy (−0.07 pt in P22, +0.1 pt here) and with 99 % of the dilation response kept, so Alt 2 is excluded.
- **Training length:** over epochs 0–9, R(φ+) rises by 0.08–0.18 and R(φ−) falls by 0.26–0.44, with no approach to the released level.
- **Route shares:** the per-particle route carries 0.10–0.31 of a real dilation in the planted arms and 0.83–0.84 in the released models.

## Explainer entry (for site/explain.py)

- **Question.** Does the P22 result hold on new jets and for other shift directions, and does the planted networks' axis response grow with training?
- **Background.** $R=\sum_j \Delta d_j^{s}\Delta d_j^{dil}/\sum_j(\Delta d_j^{dil})^2$, where $d=\ell_W-\ell_Z$. Here $s$ is a rigid shift of every particle about the stored axis, and $dil$ is a per-jet dilation $(\Delta\eta,\Delta\phi)\to k_j(\Delta\eta,\Delta\phi)$ with the same $\Delta\log m_A$, where $m_A=\sqrt{M_0M_2}$.
- **Method.** 15,000 new jets. Shifts $\Delta\phi\pm0.06$ and $\Delta\eta+0.06$, each with its own matched dilation. Checkpoints after epochs 0–9 of the two true-label runs.
- **Result (figure `results/part/p22_verify.png`).**
  - The +φ values reproduce P22: $R=0.22$ and $0.25$ (true labels), $0.01$ (offset training), $0.73$ and $0.80$ (released).
  - The −φ shift gives $R=0.47$ and $0.49$ for the true-label runs.
  - The released models are symmetric in direction.
- **Checks.**
  - Rotation null $|R|\le0.008$.
  - Offset-trained arm $R\le0.02$ in all directions.
  - Accuracy matches P22.
- **Why it matters.** The planted networks' response is direction-dependent, but in no direction does it reach the released models' level. The conclusion that 1.5M-jet, 3-class training does not reproduce the released axis response holds. Its size for the planted arms is 0.2–0.5, not 0.2–0.25.
