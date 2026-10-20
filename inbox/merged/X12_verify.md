# X12 verify: fresh-sample rerun of the X12 translation and rank-window tests, a matched real-change calibration of the mass response, network-level tests

Status: done. Confidence that the X12 reading is correct as stated: 0.55 (corrections below; the mass-size statement and the
radial-hinge significance change).
Code: `interp/x12_verify.py` (imports `Net`, `translate`, `scale_angles`, `moments`, `swap`, `window_test`, `mcnemar`, `holm` from
`interp/x12_retrained_jedi.py` unchanged). Results: `results/jedi/x12_verify_s2_<model>.json`, `x12_verify_s2_summary.json`
(seed 2, this verify), `x12_verify_<model>.json`, `x12_verify_summary.json` (seed 1, first verify job). Logs:
`logs/x12-verify.log` (seed 1: per-model steps finished, its `--combine` failed with `KeyError: 'alpha_all'`, the official
X12 JSON predates that field; fixed), `logs/x12-verify-b.log` (seed 2 run + both combines).
Cluster jobs: x12-verify (1 CPU / 2 GiB, 2,273 s, peak 2,047 MiB, failed in combine only), x12-verify-b (1 CPU / 2 GiB, 1,434 s,
peak 1,556 MiB, 0.79 cores).

## Conclusion
On two new sets of jets, the retrained quantized networks again send most true W jets to the Z class when every particle is
shifted by 0.06 rad, and the official network's large particle-swap effect at the "14 hardest particles" boundary again appears
in no retrained network. The new control changes one part of the reading: when each network is compared with its own response
to a genuine widening of the jet that gives the same pT-weighted spread $\sum p_T r^2$ as the shift, the networks trained
without quantization respond to the shift as much as the quantized ones do (shift response / matched widening 0.69–1.30 vs
1.02–1.38). The smaller W→Z fraction in the float networks therefore reflects where they place the Z class boundary for the
too-wide mass (about 105 GeV) that the shift imitates, not a weaker reliance on the uncentred spread. The excess of
centre-and-spread ("radial hinge") objects in quantized networks is significant at the network level only when all five usable
quantized networks are pooled (p = 0.048, the smallest value attainable).

## Code review
- **Stages and forward.** `Net.forward` follows the jedi_core stage order (l1 → qs → dd = g; l2; add; l3 → qs1 = S; d4–d7) with
  each layer's own input quantizer and fused kernel; it equals `Jedi.run` (official, max |Δlogit| 0) and the Keras model on CPU
  (7 of 8 retrained models, max |Δlogit| ≤ 3e-6). The stored GPU `test_probs` agree in argmax on 99.6–99.97 % of the sample
  rows (float16 inputs, tf32), which also confirms that `full_test.npz` rows are the 260,000-jet val archive in order.
- **Split.** Training rows = train archive minus `splits.npz['dev']`; validation = dev; test = val archive. The battery and both
  verify samples come from the val archive only. No leakage found.
- **Padding.** `translate`, `scale_angles`, `rotate_matched` and `moments` act on particles with pT > 0 only; `swap` uses jets
  with more than k + 1 real particles. No padded slot is counted as a particle.
- **Edits.** T changes only (Δη) of real particles (invariant mass unchanged: max |Δlog m| 1e-7). The M2-held edit holds $M_0$ and
  $M_2=\sum p_T r^2$ about the stored axis exactly (max |Δlog M2| 8e-8), but lowers the invariant mass of W jets by 23 %
  (mean Δlog m = −0.27); the report does not state this. It therefore does not by itself separate "network reads $M_2$" from
  "network reads invariant mass"; the T edit (invariant mass fixed, W→Z 0.6–0.8) does.
- **Mask-keeping null.** The permutation acts on l3 output columns, d4 input rows and d4 input-quantizer tables together; qs1
  masks and l1/l2/dd are unchanged; tables are restored after the draws (restore check max |Δlogit| recorded). Correct as
  described.
- **Window test design.** The X12 "own edge" for retrained networks is at ranks 24–45 while the inside pair is chosen anywhere
  (ranks 4–37). Across and inside pairs then differ in particle pT, so the ratio mixes the mask effect with the pT dependence of
  the objects. The official test (13|14 vs 11|12) does not have this problem. Addressed below with the fixed 13|14 edge and a
  rank scan.
- **Radial-hinge test.** The Fisher test pools units across networks as if independent and compares 5–11 mapped units per
  quantized network with 20 per float network. Networks are the replication unit; addressed below.
- **Figure `results/jedi/x12_final.png` (read).** (c) has no unedited bar, so HGQ-150 s0's M2-held bar (0.147) cannot be read as
  equal to its own unedited rate (0.147); (b) does not show which rank edge each model uses (13|14 official vs 24–45 retrained);
  x labels of (a) overlap ("HGQ-150s0"); the suptitle states a reading ("recurs ... does not"). No number in the figure is wrong.
- No bug found that changes a reported number.

## Procedure (verify)
- Samples: `jedi_core.sample(4000, seed=1)` and `sample(4000, seed=2)`, each minus the jets in the X12 seed-0 sample:
  N = 18,024 and 18,025 jets (N_W = 3,610 and 3,599). All 9 X12 models (official + 8 retrained), last-epoch checkpoints.
- Translation (true W, W→Z = fraction called Z; flow = edited − unedited, 1,000-draw jet bootstrap):
  T (+0.06 rad Δη), Tr (0.06 rad random direction, rng 2201), M2-held (Tr, then angles scaled to restore $M_2$), M1-held (Tr,
  then angles scaled to restore $M_1=\sum p_T r$), shrink (angles scaled by the M2-held factor, no translation), and two
  calibration edits added here (seed 2 only):
  - **dilM2**: angles scaled per jet by $s_T=\sqrt{M_2(T)/M_2}$ (mean 1.32), so $M_0$ and $M_2$ about the stored axis equal those
    of the translated jet exactly, while the invariant mass is multiplied by $\approx s_T$ (W: about 80 → 105 GeV).
  - **dilZ**: angles × 91.19/80.38 (a real W → Z mass shift).
  - $H_{M2}$ (network reads $M_0, M_2$ about the axis): flow(T) = flow(dilM2). $H_m$ (network reads a recentred / invariant
    mass): flow(T) ≈ 0 while flow(dilM2) > 0. Statistic: ratio flow(T)/flow(dilM2), paired jet bootstrap (1,000); McNemar T vs
    dilM2 on paired Z calls.
- Rank windows: (a) the model's own X12 edge and inside pair; (b) the official edge 13|14 vs 11|12 (and vs 5|6) with the live S
  units whose qs1 mask changes at 13|14, or the top-10 live S units by X12 cost when none does; (c) 13|14 with all live S units;
  (d) scan of adjacent swaps k|k+1 (ranks held by ≥ 50 % of jets), $m_k$ = mean |ΔS|/sd(S) over live S, peak index
  $m_k/\mathrm{median}_j m_j$ (x[::3] seed 1, x[::4] seed 2).
- Network-level tests (combine): exact one-sided permutation over network labels, HGQ vs float, for the radial-hinge share
  (from the X12 naming), flow(T), flow(dilZ) and flow(T)/flow(dilM2). Minimum attainable p: 1/6 (2 vs 2), 1/21 (5 vs 2).
- Holm over the new 13|14 window tests and T-vs-M2-held tests of the 8 retrained models (16 tests).

## Results

### Reproduction (seed 1 / seed 2 vs X12 seed 0)
| model | acc s1 / s2 | W→Z unedited | T | Tr | M2-held | own-edge ratio s1 / s2 (X12) |
|---|---|---|---|---|---|---|
| official | 81.2 / 80.7 | 0.028 / 0.026 | 0.769 / 0.767 | 0.602 / 0.619 | 0.044 / 0.044 | 5.35 / 5.18 (5.28) |
| HGQ-150 s0 | 77.5 / 77.2 | 0.143 / 0.143 | 0.695 / 0.700 | 0.752 / 0.747 | 0.139 / 0.137 | 1.11 / 1.25 (1.08) |
| HGQ-150 s1 * | 57.0 / 57.0 | 0.491 / 0.492 | 0.156 / 0.159 | 0.179 / 0.177 | 0.561 / 0.563 | 0.00 / 0.00 (0.00) |
| HGQ-150 s2 | 78.3 / 77.9 | 0.064 / 0.061 | 0.726 / 0.723 | 0.745 / 0.733 | 0.085 / 0.083 | 0.55 / 0.56 (0.59) |
| HGQ-50 s0 | 78.6 / 78.3 | 0.074 / 0.074 | 0.671 / 0.676 | 0.672 / 0.682 | 0.124 / 0.124 | 1.42 / 1.36 (1.36) |
| HGQ-50 s1 | 79.0 / 78.6 | 0.052 / 0.055 | 0.615 / 0.614 | 0.625 / 0.619 | 0.179 / 0.186 | 1.33 / 1.35 (1.31) |
| HGQ-50 s2 | 80.7 / 80.3 | 0.038 / 0.038 | 0.787 / 0.789 | 0.639 / 0.645 | 0.122 / 0.110 | 1.48 / 1.42 (1.48) |
| float s0 | 82.5 / 82.4 | 0.018 / 0.017 | 0.226 / 0.221 | 0.243 / 0.243 | 0.036 / 0.033 | 0.34 / 0.34 (0.34) |
| float s1 | 82.6 / 82.2 | 0.016 / 0.015 | 0.349 / 0.351 | 0.311 / 0.320 | 0.034 / 0.031 | 0.50 / 0.49 (0.49) |

\* degenerate network. Every X12 number in this table reproduces within its interval. The mass element calls reproduce
(HGQ-150 s0 M2-held 0.139 / 0.137 vs threshold 0.15, equal to its unedited rate: McNemar M2-held vs unedited p 0.53 / 0.40;
HGQ-50 s1 0.179 / 0.186 > 0.15). T vs M2-held McNemar p ≤ 1e-127 in every usable model (Holm, 16 tests). +Δη excess T > Tr
again only in the official network and HGQ-50 s2.

### Competing hypothesis M1 (seed 1)
M1-held leaves W→Z 0.19–0.24 in every network vs M2-held 0.03–0.19; McNemar M1-held vs M2-held p ≤ 1e-11 in every network.
$H_{M1}$ ($\sum p_T r$ as the read variable) is excluded as the sole variable; $M_2$ fits better in all 9.

### Matched real-change calibration (seed 2, new)
| model | flow T [CI] | flow dilM2 [CI] | T / dilM2 [CI] | McNemar T vs dilM2 (T only / dilM2 only, p) | W→Z dilZ |
|---|---|---|---|---|---|
| official | 0.741 [0.727, 0.757] | 0.606 [0.590, 0.624] | 1.22 [1.20, 1.25] | 628 / 143, 3e-73 | 0.835 |
| HGQ-150 s0 | 0.557 [0.537, 0.577] | 0.493 [0.472, 0.513] | 1.13 [1.10, 1.16] | 443 / 213, 2e-19 | 0.833 |
| HGQ-150 s2 | 0.662 [0.645, 0.681] | 0.615 [0.598, 0.633] | 1.08 [1.05, 1.10] | 351 / 184, 5e-13 | 0.814 |
| HGQ-50 s0 | 0.602 [0.582, 0.621] | 0.543 [0.525, 0.563] | 1.11 [1.08, 1.14] | 439 / 226, 1e-16 | 0.807 |
| HGQ-50 s1 | 0.558 [0.539, 0.575] | 0.546 [0.527, 0.564] | 1.02 [0.99, 1.06] | 517 / 473, 0.17 | 0.795 |
| HGQ-50 s2 | 0.751 [0.735, 0.766] | 0.545 [0.528, 0.564] | 1.38 [1.35, 1.41] | 799 / 57, 2e-168 | 0.844 |
| float s0 | 0.205 [0.190, 0.218] | 0.297 [0.282, 0.314] | 0.69 [0.64, 0.74] | 283 / 616, 4e-29 | 0.855 |
| float s1 | 0.336 [0.320, 0.353] | 0.258 [0.243, 0.273] | 1.30 [1.23, 1.38] | 637 / 359, 1e-18 | 0.854 |

- H0: flow(T) = flow(dilM2) (network reads $M_0, M_2$ about the axis); H1: they differ. The ratio is 0.69–1.38 in all 8 usable
  networks; $H_m$ (ratio ≈ 0) is excluded in every network, float included. H0 is rejected in 7 of 8 at small effect, in both
  directions, so the networks also read something beyond $(M_0, M_2)$ (as in X9v / E6).
- Network level: T/dilM2, HGQ (5 usable) vs float: permutation p = 0.24; dilZ W→Z, HGQ vs float: p = 1.0 (float 0.854–0.855 is at
  the top of the HGQ range 0.795–0.844); flow(T), HGQ vs float: p = 0.048 (the minimum).
- Reading: both arms respond to the translation about as they respond to a real widening with the same $M_0, M_2$; both
  respond strongly to a real W → Z mass shift. The float networks' lower W→Z under T (and under dilM2) goes with their lower
  W→Z at ~105 GeV equivalent mass (dilM2: 0.26–0.30 vs 0.49–0.62 HGQ), i.e. a narrower Z class in mass, not a lower share of the
  uncentred spread in their mass variable.

### Rank windows (seed 2; seed 1 in parentheses)
- Official 13|14 vs 11|12: 5.18 [4.97, 5.41] (5.35); with all live S units 2.74 (2.77).
- Retrained networks at 13|14 vs 11|12: HGQ-150 s0 0.78 [0.13, 3.67] (only S4, mask edge at 13|14, nearly constant), s2 0.20
  [0.13, 0.30], HGQ-50 0.60–0.90, float 0.76–0.83; one-sided sign-flip p ≥ 0.60 for every retrained model (Holm 1.0). Only
  HGQ-150 s0, s1, s2 have any live S unit with a qs1 edge at 13|14 (1, 2, 1 units); the official has 5.
- Scan: peak index at 13 = 2.86 official (2.83) vs 0.43–0.92 retrained HGQ and 1.30–1.53 float. HGQ-50 s0 and s1 have their
  largest peak at ranks 40 and 42 (index 3.52, 4.44; seed 1: 3.50, 4.38), at their own mask edges (43|44, 42|43). Every other
  network, official and float included, peaks at 0|1 (float 7.4–9.7). These high-rank peaks use only jets with ≥ 42–44 particles,
  so they are not comparable in N to the official 13|14 value. A swap response localized at a mask edge thus recurs in 2 of 5
  usable quantized networks, at a different rank and on a minority of jets; the X12 statement "none reproduces the official
  across-edge response" holds for the official edge and for the ratio statistic.

### Radial-hinge objects at the network level
Per network (from X12): official 11/16; HGQ-150 4/8, 5/5 (degenerate), 7/7; HGQ-50 6/11, 5/8, 2/9; float 3/20, 4/20 (top-7:
1/7, 2/7). Exact permutation, one-sided: primary HGQ-150 usable (2) vs float (2) p = 0.17; all 5 usable HGQ vs float p = 0.048;
pooled Fisher (X12, units as independent) p 7.8e-6 / 0.0013. The direction holds in every network; the significance at the
replication unit (network) is at the floor of what 2–5 networks per arm can give.

### l1 structure (weights)
Units carrying both a pT and an angle weight: HGQ-150 0, 0, 0; HGQ-50 3, 2, 3 (all still "pure" by the 90 % spread rule);
float 64, 64. Consistent with the X12 reading that purity follows from sparsity.

## Corrected reading
- Mass response: the translation response that is removed by holding $\sum p_T r^2$ fixed recurs in 4 of the 5 usable retrained
  HGQ networks by the preregistered thresholds (HGQ-150 s0 meets "M2-held < 0.15" only because its unedited rate is 0.14; HGQ-50
  s1 misses it at 0.18–0.19), reproduced on two fresh samples. The float networks show 2–3× lower W→Z fractions under T, but
  relative to a real widening with the same $M_0$ and $M_2$ their response is 0.69–1.30 vs 1.02–1.38 for HGQ (network-level
  p = 0.24), and their response to a real W → Z mass shift is as large (0.85 vs 0.80–0.84). Strike "the un-recentred response is
  weaker without quantization" and "its size depends on training"; replace with "both arms read the uncentred spread to a
  similar degree; the float networks call fewer ~105 GeV-equivalent W jets Z".
- Rank windows: as stated (none reproduces the official 13|14 response: 0.20–0.90 vs 5.18 at the same edge, sign-flip p ≥ 0.60),
  with the addition that HGQ-50 s0 and s1 show a localized swap peak at their own mask edges near rank 40–42.
- Pure cuts: as stated.
- Radial hinge objects: "more frequent with quantization" in every network; network-level permutation p = 0.048 (5 usable HGQ vs
  2 float) or 0.17 (primary arm), not 7.8e-6.

## Explainer entry (for site/explain.py, Paper A tab, appended to the X12 explainer under "Checks")
**Question.** Do the networks trained without quantization respond less to an uncentred shift of the jet, or do they read the
same uncentred spread but draw the W/Z boundary differently?

**Background.** A rigid shift by $d=0.06$ rad leaves the invariant mass unchanged but raises
$M_2=\sum_i p_{T,i} r_i^2$ about the stored axis by about $M_0 d^2$. A real widening of the jet by a factor $s_T$ (all angles
$\times s_T$) raises $M_2$ by $s_T^2$ and the invariant mass by $s_T$. With $s_T=\sqrt{M_2(T)/M_2}$ (mean 1.32), both edits give
the same $(M_0, M_2)$.

**Method.** For each network, compare the W→Z flow under the shift with the flow under the matched widening:
$R=\frac{\Delta_{W\to Z}(T)}{\Delta_{W\to Z}(\text{dil}_{M_2})}$, jet bootstrap CI; a network that reads only $(M_0,M_2)$ gives
$R=1$, one that reads a recentred mass gives $R\approx 0$. Also a real W → Z widening by $m_Z/m_W$.

**Result.** $R$ = 1.02–1.38 (quantized), 0.69–1.30 (float), 1.22 (official); W→Z under the real W → Z widening 0.80–0.84
(quantized) and 0.85 (float).

**Checks.** Seed-2 sample (18,025 jets, seed-0 jets removed); all X12 translation numbers reproduce on seeds 1 and 2.

**Why it matters.** The uncentred mass reading is shared by quantized and float training of this architecture; the difference
in W→Z fraction comes from where the Z class ends in mass.

**Conclusion.** Networks trained with and without quantization respond to the 0.06 rad shift about as much as to a real
widening of the jet that gives the same pT-weighted spread, so both read the uncentred spread. The float networks send fewer
shifted W jets to Z because they call fewer jets of about 105 GeV mass Z, not because they ignore the spread. Uncertain: two
float seeds only, and the ratio varies between seeds (0.69 and 1.30).
