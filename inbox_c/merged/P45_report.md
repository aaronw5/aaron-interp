# P45 — Model stitching ParticleNet_full → ParT_full (loop C, round C10, exploratory)

Claims: A8 (do the big models share stages), C5 (ParticleNet's axis response is about 2/3 of ParT's), C3. Proposed new claim id
**C21** (the plan said "C20", but C20 is already used by P39 in PAPERS.md).

## Conclusion

The per-particle features after ParticleNet's first or second graph layer can replace ParT's own particle tokens at ParT's
first three attention blocks. The only addition is one learned affine (linear plus offset) map. Accuracy drops by 0.9–2.1
percentage points, against 8.7–13.5 points when the same map is fitted to an untrained ParticleNet. Features after the third
graph layer lose 5–9 points. How strongly the stitched network responds to a mass-preserving shift of the jet axis depends
on the network that does the remaining computation: it matches ParT when six or more ParT blocks follow the stitch, and
ParticleNet when only ParT's last block follows. Untrained features fed into ParT's early blocks also give a ParT-sized
response, so this test does not show that ParticleNet's trained features contain ParT's axis response. It shows that ParT's
later blocks set the size of the response from any features that carry the particle angles.

## Question

Previous comparisons of ParT and ParticleNet compared their outputs, for example the W → Z rate under an axis shift. P45
asks whether their internal features can be swapped. It also asks where the axis response is set: in early features that
both networks share, or in ParT-specific later blocks.

## Procedure

Models: ParT_full (released weights, weaver-core 0.4.17) and ParticleNet_full (released weights, interp/pn_net.py). Both are
frozen; only the affine maps are trained. Script `interp/p45_stitch_pn_part.py`; job `anrunw-ai-c-p45-run` (TITAN Xp,
48 min, smoke then full run); results `results/part/p45_stitch.json`, figure `results/part/p45_stitch.png`.

Stitch diagram:

```
jet ─► ParticleNet: BN ─► EdgeConv-1 (64/particle) ─► EdgeConv-2 (128) ─► EdgeConv-3 (256) ─► mean over particles (256) ─► fc ─► PN logits
                               │ ec1                 │ ec2                │ ec3                     │ pooled
                               ▼  affine map t_i = W x_i + b per particle (padded slots set to 0)   ▼ affine 256 → 128
jet ─► ParT: embed ─► block 1 ─► … ─► block k−1 ─╳─► [mapped PN tokens] ─► block k ─► … ─► block 8 ─► 2 class blocks ─► LayerNorm ─╳─► [mapped pooled] ─► head
             pair bias U(k_T, z, ΔR, m²) of every block still computed from the real jet; padding mask kept
```

- S1 (12 points): source is the EdgeConv-1, -2 or -3 output per particle (64, 128, 256 numbers). Target is the input of ParT
  particle block k, k = 1, 3, 5, 8; the input of block 1 is the embedding output. Blocks 1 … k−1 of ParT are skipped.
- S2 (1 point): source is ParticleNet's pooled 256-vector (the input of its fc layers). Target is ParT's final class-token
  vector after the LayerNorm (the input of ParT's last linear layer).
- What the shift reaches: a rigid φ + 0.06 rad shift of every particle leaves every pairwise quantity (k_T, z, ΔR, m²)
  unchanged. In a stitched network the shift therefore reaches ParT only through the donor's features, because ParT's pair
  bias does not change. The matched dilation does change the pair features, so it reaches ParT through both routes; P2
  found that the pair route carries 1–6 % of ParT's response to real mass changes.
- Training: each map starts at the least-squares (ridge 1e-4 × mean diagonal) fit from the source to ParT's own activation
  at the target, fitted on 20,224 jets. It is then trained for one pass to minimise KL(p_ParT ‖ p_stitched).
  KL(p ‖ q) = Σ_c p_c log(p_c / q_c) over the 10 class probabilities, where p_ParT is ParT's own output on the same jet.
  Training data: 300,000 JetClass training jets from train_100M, 2 random files per class with 15,000 random rows each,
  default_rng(4545), never first-k. Batches of 256 are grouped by particle multiplicity in random order; Adam, lr 5e-4,
  3 % warm-up, cosine decay. ParT's accuracy on these training jets is 86.11 %.
- Controls: (a) untrained donor: the same 13 stitches from a ParticleNet with PyTorch default random weights (torch seed
  4545, BatchNorm at default statistics). This measures what an affine map plus ParT's remaining blocks can do with a
  generic feature map of the jet. (b) raw-input floor: an affine map from ParT's own 17 scaled input features per particle
  into block k. (c) identity ceiling: an affine map from ParT's own block-k input, started at the identity; this is also a
  check of the code. (d) a second training seed (`_s1`) for the 13 trained-donor points, with different batch composition,
  batch order and least-squares jets.
- Evaluation samples: P1 (rows 3,000–4,000 of each class's test_20M *_100 file, N 10,000, rebuilt with part_core.build) and
  a fresh sample (val_5M *_121, 1,000 random rows per class, default_rng(4546), N 10,000, rebuilt).
- Edits on the evaluation jets: φ + 0.06 rad added to every real particle. This is a rigid rotation about the beam: the
  invariant mass changes by at most 0.005 in log m and the masks are unchanged. Matched dilation: every particle's
  (Δη, Δφ) is multiplied by k_j = √(M0M2(shifted) / M0M2(unedited)) for each jet, so that the axis-moment mass √(M0·M2)
  rises as much as the shift raises it. Here M0 = Σ pT and M2 = Σ pT r² about the stored axis. The mean k_j is 1.08 and the
  mean true Δlog m is +0.072.
- Metrics:
  - Stitching gap = acc(ParT alone) − acc(stitched) on the same jets, in percentage points, with a paired 95 % bootstrap
    over jets (1,000 resamples). An exact McNemar test on per-jet correctness compares ParT with the stitched network and
    the trained-donor stitch with the untrained-donor stitch.
  - 10-class confusion and per-class recall (in the JSON).
  - W → Z: the fraction of true-W jets called Z before and after the shift, with a McNemar test.
  - R: the per-jet slope through 0 of the change of d = logit_W − logit_Z under the shift against its change under the
    matched dilation, over true W, Z and QCD jets (as in P22/P22v). R = 1 means the shift is treated like a real mass
    change of the same axis-moment size; R = 0 means only the invariant mass is read.
  - Bootstrap p for R(stitched) − R(PN alone) and R(stitched) − R(ParT alone).
  - Holm correction over the 13 trained-donor stitch points for each family. With 1,000 resamples the bootstrap p cannot be
    smaller than 0.001, so the smallest Holm value it can reach is 0.013.

Hypotheses as pre-registered in the plan:
- H1 (shared stages): ec2/ec3 → blocks 1–3 recover to within 2 points of ParT, far above the untrained donor; when stitched
  late, the stitched R equals ParticleNet's R.
- H0 (no shared stages): gaps above 5 points at every point except the identity; the untrained donor as good as the trained
  one; or R follows the recipient.

## Numbers

Unedited models on the same jets: ParT_full 85.93 % (P1; reference 85.9 %) and 86.29 % (_121). ParticleNet_full 84.56 / 84.61 %
(reference 84.57 % on P1). ParT R 0.798 [0.776, 0.820] / 0.785; ParticleNet R 0.532 [0.499, 0.564] / 0.513, which is 0.67 / 0.65
of ParT's (C5 said about 2/3). W → Z under the shift: ParT 11.7 → 31.0 % (P1) and 12.5 → 30.3 % (_121); ParticleNet
12.3 → 28.5 % and 11.8 → 24.7 %. Code checks: our ParT forward pass matches weaver's to |Δlogit| ≤ 4e-6 and our ParticleNet
pass matches exactly (0); identity stitches give a gap of −0.01 to +0.07 points and agree with ParT on 99.9 % of calls.

Full table (gap > 0 means the stitched network is less accurate than ParT; "Holm p gap" tests gap = 0; "vs untrained" tests
equal accuracy of the trained-donor and untrained-donor stitch at the same point):

| stitch | P1: acc % / gap [95 % CI] | _121: acc % / gap [95 % CI] | Holm p gap (P1 / _121) | Holm p vs untrained (P1 / _121) | R P1 | R _121 | Holm p R vs PN (P1/_121) | Holm p R vs ParT (P1/_121) | W→Z % _121 before → after |
|---|---|---|---|---|---|---|---|---|---|
| ParT | 85.93 | 86.29 | – / – | – / – | 0.798 [0.776, 0.820] | 0.785 [0.762, 0.808] | – / – | – / – | 12.5 → 30.3 |
| PN | 84.56 | 84.61 | – / – | – / – | 0.532 [0.499, 0.564] | 0.513 [0.486, 0.542] | – / – | – / – | 11.8 → 24.7 |
| pn1>b1 | 85.08 / +0.85 [+0.44, +1.25] | 84.95 / +1.34 [+0.93, +1.73] | 9.1e-05 / 1.1e-10 | 2.9e-115 / 1.4e-110 | 0.785 [0.766, 0.804] | 0.776 [0.756, 0.795] | 0.013 / 0.013 | 0.45 / 0.72 | 12.7 → 27.8 |
| pn1>b3 | 84.18 / +1.75 [+1.27, +2.17] | 84.21 / +2.08 [+1.61, +2.56] | 2.2e-12 / 3.6e-17 | 1.9e-165 / 1.7e-180 | 0.753 [0.729, 0.776] | 0.753 [0.728, 0.777] | 0.013 / 0.013 | 0.03 / 0.16 | 14.3 → 31.3 |
| pn1>b5 | 82.06 / +3.87 [+3.32, +4.44] | 82.48 / +3.81 [+3.22, +4.41] | 5.2e-38 / 2e-37 | 7.2e-253 / 3.7e-281 | 0.685 [0.657, 0.712] | 0.696 [0.670, 0.720] | 0.013 / 0.013 | 0.013 / 0.013 | 13.5 → 31.9 |
| pn1>b8 | 76.69 / +9.24 [+8.56, +9.96] | 77.15 / +9.14 [+8.41, +9.88] | 1.1e-132 / 1.9e-132 | 0 / 0 | 0.719 [0.688, 0.749] | 0.738 [0.710, 0.767] | 0.013 / 0.013 | 0.013 / 0.16 | 15.0 → 33.2 |
| pn2>b1 | 85.03 / +0.90 [+0.47, +1.32] | 84.59 / +1.70 [+1.24, +2.16] | 9.1e-05 / 8.5e-14 | 2.2e-103 / 1.6e-82 | 0.826 [0.803, 0.849] | 0.809 [0.786, 0.832] | 0.013 / 0.013 | 0.04 / 0.16 | 12.1 → 31.7 |
| pn2>b3 | 84.58 / +1.35 [+0.90, +1.80] | 84.49 / +1.80 [+1.34, +2.25] | 6e-08 / 6.6e-15 | 1.4e-148 / 1.3e-153 | 0.835 [0.815, 0.856] | 0.822 [0.801, 0.845] | 0.013 / 0.013 | 0.013 / 0.056 | 12.7 → 32.4 |
| pn2>b5 | 83.90 / +2.03 [+1.56, +2.53] | 84.20 / +2.09 [+1.57, +2.61] | 5.4e-14 / 1.9e-15 | 8.8e-275 / 3.4e-315 | 0.671 [0.649, 0.691] | 0.672 [0.651, 0.693] | 0.013 / 0.013 | 0.013 / 0.013 | 11.5 → 27.9 |
| pn2>b8 | 81.35 / +4.58 [+3.98, +5.20] | 81.87 / +4.42 [+3.85, +5.06] | 2.9e-48 / 1.7e-47 | 0 / 0 | 0.507 [0.489, 0.524] | 0.487 [0.469, 0.505] | 0.14 / 0.14 | 0.013 / 0.013 | 12.2 → 22.4 |
| pn3>b1 | 77.28 / +8.65 [+7.90, +9.30] | 77.77 / +8.52 [+7.82, +9.24] | 1.5e-121 / 2.3e-125 | 0.32 / 0.18 | 0.739 [0.693, 0.784] | 0.771 [0.725, 0.816] | 0.013 / 0.013 | 0.04 / 0.72 | 14.3 → 25.8 |
| pn3>b3 | 79.73 / +6.20 [+5.55, +6.85] | 80.26 / +6.03 [+5.38, +6.73] | 2.2e-72 / 1.3e-73 | 1.7e-31 / 3.4e-36 | 0.773 [0.734, 0.810] | 0.731 [0.684, 0.778] | 0.013 / 0.013 | 0.45 / 0.16 | 16.3 → 28.1 |
| pn3>b5 | 80.97 / +4.96 [+4.31, +5.54] | 81.04 / +5.25 [+4.61, +5.93] | 2.4e-51 / 4.8e-58 | 5.7e-180 / 4e-185 | 0.687 [0.651, 0.724] | 0.678 [0.635, 0.720] | 0.013 / 0.013 | 0.013 / 0.013 | 16.0 → 25.3 |
| pn3>b8 | 80.98 / +4.95 [+4.28, +5.55] | 80.61 / +5.68 [+5.04, +6.43] | 1.6e-48 / 2.4e-65 | 0 / 0 | 0.490 [0.454, 0.526] | 0.482 [0.439, 0.521] | 0.084 / 0.14 | 0.013 / 0.013 | 14.8 → 21.9 |
| pnpool>cls | 82.89 / +3.04 [+2.46, +3.59] | 83.01 / +3.28 [+2.70, +3.81] | 7.2e-25 / 1.9e-30 | 0 / 0 | 0.345 [0.313, 0.377] | 0.353 [0.323, 0.382] | 0.013 / 0.013 | 0.013 / 0.013 | 14.3 → 21.6 |
| u1>b1 | 76.96 / +8.97 [+8.30, +9.65] | 76.93 / +9.36 [+8.61, +10.14] | – / – | – / – | 0.616 [0.590, 0.642] | 0.620 [0.595, 0.645] | – / – | – / – | 18.8 → 30.2 |
| u1>b3 | 73.22 / +12.71 [+11.92, +13.50] | 72.80 / +13.49 [+12.69, +14.30] | – / – | – / – | 0.618 [0.590, 0.645] | 0.619 [0.587, 0.648] | – / – | – / – | 25.4 → 41.2 |
| u1>b5 | 66.27 / +19.66 [+18.81, +20.51] | 65.86 / +20.43 [+19.55, +21.39] | – / – | – / – | 0.724 [0.697, 0.752] | 0.746 [0.713, 0.775] | – / – | – / – | 33.5 → 47.0 |
| u1>b8 | 43.86 / +42.07 [+41.05, +43.13] | 43.41 / +42.88 [+41.81, +43.96] | – / – | – / – | 0.888 [0.852, 0.921] | 0.891 [0.855, 0.923] | – / – | – / – | 32.6 → 47.2 |
| u2>b1 | 77.05 / +8.88 [+8.20, +9.57] | 77.55 / +8.74 [+8.07, +9.52] | – / – | – / – | 0.684 [0.659, 0.706] | 0.678 [0.656, 0.699] | – / – | – / – | 18.8 → 29.1 |
| u2>b3 | 74.28 / +11.65 [+10.88, +12.40] | 74.09 / +12.20 [+11.38, +13.02] | – / – | – / – | 0.713 [0.686, 0.741] | 0.705 [0.678, 0.732] | – / – | – / – | 25.5 → 41.9 |
| u2>b5 | 67.63 / +18.30 [+17.44, +19.17] | 66.80 / +19.49 [+18.61, +20.46] | – / – | – / – | 0.800 [0.773, 0.827] | 0.812 [0.786, 0.838] | – / – | – / – | 31.9 → 50.1 |
| u2>b8 | 45.96 / +39.97 [+38.96, +41.00] | 45.73 / +40.56 [+39.49, +41.64] | – / – | – / – | 0.958 [0.926, 0.991] | 0.971 [0.936, 1.006] | – / – | – / – | 34.7 → 53.3 |
| u3>b1 | 77.71 / +8.22 [+7.55, +8.90] | 78.33 / +7.96 [+7.33, +8.65] | – / – | – / – | 0.741 [0.716, 0.766] | 0.722 [0.695, 0.745] | – / – | – / – | 15.9 → 30.2 |
| u3>b3 | 74.77 / +11.16 [+10.41, +11.87] | 74.88 / +11.41 [+10.64, +12.20] | – / – | – / – | 0.784 [0.762, 0.807] | 0.767 [0.740, 0.793] | – / – | – / – | 23.3 → 43.0 |
| u3>b5 | 67.62 / +18.31 [+17.46, +19.16] | 67.19 / +19.10 [+18.25, +20.02] | – / – | – / – | 0.863 [0.840, 0.886] | 0.869 [0.843, 0.893] | – / – | – / – | 29.7 → 50.8 |
| u3>b8 | 47.62 / +38.31 [+37.32, +39.42] | 47.31 / +38.98 [+37.91, +40.09] | – / – | – / – | 1.028 [0.996, 1.059] | 1.025 [0.990, 1.058] | – / – | – / – | 35.3 → 54.7 |
| upool>cls | 43.87 / +42.06 [+41.03, +43.11] | 43.35 / +42.94 [+41.89, +44.12] | – / – | – / – | 0.305 [0.242, 0.370] | 0.268 [0.210, 0.336] | – / – | – / – | 25.5 → 24.4 |
| raw>b1 | 72.69 / +13.24 [+12.42, +14.05] | 72.93 / +13.36 [+12.60, +14.23] | – / – | – / – | 0.483 [0.452, 0.512] | 0.486 [0.457, 0.514] | – / – | – / – | 24.6 → 32.5 |
| raw>b3 | 62.66 / +23.27 [+22.30, +24.20] | 62.36 / +23.93 [+23.00, +24.85] | – / – | – / – | 0.488 [0.452, 0.525] | 0.479 [0.444, 0.512] | – / – | – / – | 25.3 → 34.9 |
| raw>b5 | 48.20 / +37.73 [+36.69, +38.72] | 47.90 / +38.39 [+37.32, +39.52] | – / – | – / – | 0.590 [0.554, 0.626] | 0.596 [0.560, 0.631] | – / – | – / – | 28.5 → 36.4 |
| raw>b8 | 29.86 / +56.07 [+54.99, +57.10] | 30.11 / +56.18 [+55.14, +57.25] | – / – | – / – | 0.721 [0.685, 0.755] | 0.738 [0.703, 0.772] | – / – | – / – | 31.0 → 39.8 |
| id>b1 | 85.86 / +0.07 [+0.00, +0.14] | 86.30 / -0.01 [-0.06, +0.03] | – / – | – / – | 0.797 [0.775, 0.819] | 0.784 [0.760, 0.807] | – / – | – / – | 12.4 → 30.2 |
| id>b3 | 85.93 / +0.00 [-0.03, +0.03] | 86.30 / -0.01 [-0.05, +0.03] | – / – | – / – | 0.798 [0.776, 0.820] | 0.785 [0.761, 0.808] | – / – | – / – | 12.5 → 30.3 |
| id>b5 | 85.92 / +0.01 [-0.03, +0.05] | 86.30 / -0.01 [-0.08, +0.06] | – / – | – / – | 0.799 [0.777, 0.821] | 0.786 [0.762, 0.809] | – / – | – / – | 12.4 → 30.4 |
| id>b8 | 85.92 / +0.01 [-0.04, +0.05] | 86.30 / -0.01 [-0.06, +0.03] | – / – | – / – | 0.798 [0.776, 0.820] | 0.785 [0.761, 0.808] | – / – | – / – | 12.5 → 30.2 |
| pn1>b1_s1 | 85.06 / +0.87 [+0.48, +1.28] | 85.07 / +1.22 [+0.82, +1.60] | – / – | – / – | 0.791 [0.771, 0.810] | 0.782 [0.762, 0.802] | – / – | – / – | 12.7 → 28.3 |
| pn1>b3_s1 | 84.13 / +1.80 [+1.33, +2.25] | 84.20 / +2.09 [+1.59, +2.57] | – / – | – / – | 0.770 [0.746, 0.792] | 0.769 [0.745, 0.793] | – / – | – / – | 14.2 → 31.8 |
| pn1>b5_s1 | 82.13 / +3.80 [+3.22, +4.39] | 82.34 / +3.95 [+3.38, +4.55] | – / – | – / – | 0.714 [0.686, 0.740] | 0.723 [0.697, 0.747] | – / – | – / – | 14.3 → 32.4 |
| pn1>b8_s1 | 77.05 / +8.88 [+8.19, +9.60] | 77.54 / +8.75 [+7.99, +9.48] | – / – | – / – | 0.784 [0.752, 0.813] | 0.804 [0.775, 0.834] | – / – | – / – | 14.6 → 35.1 |
| pn2>b1_s1 | 85.03 / +0.90 [+0.46, +1.34] | 84.63 / +1.66 [+1.21, +2.11] | – / – | – / – | 0.822 [0.799, 0.845] | 0.806 [0.782, 0.829] | – / – | – / – | 12.5 → 31.4 |
| pn2>b3_s1 | 84.52 / +1.41 [+0.96, +1.88] | 84.53 / +1.76 [+1.31, +2.21] | – / – | – / – | 0.850 [0.830, 0.871] | 0.836 [0.814, 0.858] | – / – | – / – | 12.5 → 33.0 |
| pn2>b5_s1 | 83.86 / +2.07 [+1.57, +2.57] | 84.12 / +2.17 [+1.66, +2.69] | – / – | – / – | 0.709 [0.688, 0.729] | 0.710 [0.689, 0.731] | – / – | – / – | 11.4 → 28.9 |
| pn2>b8_s1 | 81.53 / +4.40 [+3.80, +5.02] | 82.04 / +4.25 [+3.70, +4.84] | – / – | – / – | 0.551 [0.533, 0.569] | 0.531 [0.513, 0.549] | – / – | – / – | 11.8 → 23.7 |
| pn3>b1_s1 | 77.54 / +8.39 [+7.69, +9.05] | 77.99 / +8.30 [+7.61, +9.02] | – / – | – / – | 0.719 [0.674, 0.762] | 0.750 [0.705, 0.794] | – / – | – / – | 13.9 → 24.4 |
| pn3>b3_s1 | 79.99 / +5.94 [+5.27, +6.63] | 80.08 / +6.21 [+5.58, +6.93] | – / – | – / – | 0.757 [0.718, 0.793] | 0.712 [0.666, 0.758] | – / – | – / – | 15.0 → 27.0 |
| pn3>b5_s1 | 81.12 / +4.81 [+4.16, +5.40] | 81.10 / +5.19 [+4.56, +5.85] | – / – | – / – | 0.690 [0.654, 0.727] | 0.677 [0.635, 0.718] | – / – | – / – | 15.9 → 25.8 |
| pn3>b8_s1 | 81.06 / +4.87 [+4.21, +5.49] | 80.46 / +5.83 [+5.17, +6.52] | – / – | – / – | 0.489 [0.453, 0.525] | 0.478 [0.435, 0.518] | – / – | – / – | 16.3 → 23.1 |
| pnpool>cls_s1 | 82.89 / +3.04 [+2.46, +3.61] | 83.01 / +3.28 [+2.73, +3.83] | – / – | – / – | 0.303 [0.271, 0.335] | 0.315 [0.284, 0.345] | – / – | – / – | 14.5 → 19.7 |

Per-class recall (_121; QCD, Hbb, Hcc, Hgg, H4q, Hlνqq, Zqq, Wqq, tbqq, tblν), in %: ParT 77.1 93.6 87.1 79.5 86.3 97.7 69.6 78.2 95.2 98.6;
ec1 → b1 75.2 91.4 84.8 78.0 86.1 97.1 67.7 77.1 94.1 98.0; ec3 → b1 71.7 86.8 62.3 82.6 77.3 94.7 47.9 67.6 90.2 96.6 (the losses are
mainly in H→cc and Z→qq); untrained ec2 → b1 72.4 71.3 73.5 73.9 82.0 93.4 56.8 69.1 88.7 94.4 (the loss is mainly in H→bb). Final training KL per stitch: trained ec1/ec2 → b1–b3 0.04–0.08, ec3 0.18–0.29, untrained
0.29–1.27, raw 0.44–1.72, identity < 2e-5.

## Reading

- Accuracy: the EdgeConv-1 and EdgeConv-2 outputs mapped into blocks 1–3 give gaps of 0.85–2.09 points across both samples
  and both seeds. Every gap is different from 0 (Holm p ≤ 1e-4). The untrained donor at the same points gives 8.7–13.5 points
  (McNemar Holm p ≤ 1e-82). The ec1/ec2 stitches lie within about 0.5 points of ParticleNet alone. The gap grows with k for
  ec1 (0.85 → 9.2) and ec2 (0.9 → 4.6). EdgeConv-3 is the exception: 8.5–8.7 points at block 1, which is the same as the
  untrained donor (Holm p 0.32 / 0.18), and 4.8–6.2 points at blocks 3–8. The pooled vector mapped into ParT's class token
  gives 3.0–3.3 points (untrained 42). The raw-input floor gives 13.2 (block 1) to 56 (block 8) points.
- H1's accuracy prediction (within 2 points at ec2/ec3 → blocks 1–3, far above the untrained donor) holds for ec2, and for
  ec1, which H1 did not name. It does not hold for ec3, which is 6–8.7 points from ParT. H0's "gaps above 5 points
  everywhere" and "untrained as good as trained" are excluded except at ec3 → block 1.
- Axis response: stitches that enter early (ec1/ec2 → blocks 1–3) give R 0.75–0.85. This differs from ParticleNet's
  0.51–0.53 (Holm p ≤ 0.013 at all four points on both samples) and is within 0.05 of ParT's 0.79–0.80 (Holm p vs ParT
  0.013–0.72). Stitches that enter at block 8 from ec2/ec3 give R 0.48–0.55, which cannot be told apart from ParticleNet
  (Holm p 0.08–0.14) and differs from ParT (Holm p 0.013). Block 5 gives intermediate values (0.67–0.72). ec1 → block 8 gives
  0.72–0.80, but that stitch loses 9 points of accuracy. The pooled stitch gives R 0.30–0.35, below both ParticleNet (0.51–0.53)
  and ParT. W → Z under the shift follows the same pattern: ec2 → b1 12.1 → 31.7 % and ec2 → b8 12.2 → 22.4 % (_121).
- Untrained-donor control for R: stitches from the untrained ParticleNet give R 0.62–0.74 at block 1 and 0.89–1.03 at block 8,
  so R rises the later the stitch enters. The raw-input stitch gives 0.48–0.49 at block 1 and 0.72–0.74 at block 8. A
  ParT-sized R after an early stitch therefore does not require the trained donor. ParT's remaining blocks produce a
  response of that size from generic features of the particle angles. The φ shift reaches the stitched network only
  through the donor features, so in every stitched network the response is carried into ParT by the donor features and
  then sized by the ParT blocks that follow.
- H1's R prediction ("stitched R equals PN's R when stitched late") holds for ec2/ec3 → block 8 and not for ec1 → block 8.
  H0's "R follows the recipient" holds for early stitches. The two statements are compatible: the response size is set by
  whichever network does most of the remaining computation. The pooled stitch (R 0.30–0.35 < 0.51) shows that a linear map
  of ParticleNet's pooled vector into ParT's class token keeps less of the response than ParticleNet's own nonlinear fc
  head does.

## Limits

- Exploratory node, no verifier yet.
- One training length (one pass over 300k jets) and least-squares initialisation. Gaps could shrink with longer training;
  whether the training KL had levelled off was not checked.
- The maps are trained to imitate ParT's outputs on unedited jets. A ParT-like R in early stitches could partly reflect that
  imitation target, although the shift itself is never seen in training.
- R uses the d = logit_W − logit_Z read-out over W, Z and QCD jets only.
- The untrained donor uses default BatchNorm statistics, not ones fitted to the data.
- ParticleNet_kin / ParT_kin were not stitched.

## Proposed claim (for the ledger agent)

C21 (new, exploratory): with one affine map, ParticleNet_full's EdgeConv-1/2 per-particle features replace ParT_full's
tokens at blocks 1–3 within 0.9–2.1 points of 10-class accuracy (untrained ParticleNet 8.7–13.5 points; EdgeConv-3 5–9
points). The size of the mass-fixed φ-shift response (R) of the stitched network follows ParT (0.75–0.85 vs 0.79) when the
stitch is at blocks 1–3 and ParticleNet (0.48–0.55 vs 0.51–0.53) when it is at block 8. Untrained features at early blocks
also give R 0.62–0.78. Suggested confidence 0.55. Bearing on A8: the early per-particle stages are affinely compatible
between the two architectures up to the second EdgeConv layer. Bearing on C5: ParticleNet's smaller R comes from its own
later layers (EdgeConv-3, pooling and fc), not from smaller axis information in its early features.
