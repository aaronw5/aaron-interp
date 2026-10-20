# H16: boost sensitivity, training dataset vs architecture/inputs (open question a; claim D9)

Status: done. Jobs: d-h16-train-s0 and d-h16-train-s1 (CPU, 15 configs × 1 seed each, finished 03:29 / 03:46 UTC); battery d-h16-run (CPU 2 / 6 Gi, exit 0, 52 min; one debug fix: an invalid keyword name in a log dict).
Scripts: `interp/h16_train.py`, `interp/h16_boost_dataset.py`. Results: `results/physics/h16_{hls_all,hls_win,top,summary}.json`, figure `results/physics/h16_boost_dataset.png`, log `logs/d-h16-run.log`.

## Conclusion
We trained the same three small networks (a JEDI-style net, a small transformer and a Particle Flow Network) on three samples: hls4ml jets, hls4ml jets in a narrow pT window, and Top Landscape jets. For the top-vs-quark/gluon decision, a mass-preserving boost that lowers the jet pT by 10 % changes 4–7 % of the calls of every one of these networks, against about 0.5 % for a rotation of the same size. Networks trained on Top Landscape respond about as much as those trained on hls4ml, and narrowing the hls4ml pT window changes little. So in these small networks the boost sensitivity comes from the network and its inputs, not from the training sample. A network that sees only pT fractions and angles cannot tell a boost from an angle-only rescaling, and these networks do not separate the two. Not resolved: the 21–38 % of H12 were five-class calls dominated by W/Z ↔ top changes, which this binary task does not contain, and the larger Top models (ParT, LorentzNet; 1.1 % in H12) were not retrained on hls4ml.

## Question and hypotheses
- **Question.** H12 found that hls4ml-trained nets change 21–38 % of their five-class calls under an exact boost along the jet axis at pT × 0.9. Top Landscape and JetClass nets change 1–4 %. Is this because of the training sample or because of the architecture and inputs?
- **H1 (dataset).** The hls4ml training sample causes the sensitivity, through its pT spectrum, mass–pT correlation or particle content.
  - Prediction: Top-trained small nets are nearly boost-invariant (B/P ≤ 0.5, B ≈ 1–5 %), while hlsA and hlsB nets change 20–35 %.
  - If hlsB drops to the Top level, the width of the pT window is the cause.
- **H0 (architecture/inputs).** The small hls4ml-style nets are boost-sensitive whatever sample they are trained on.
  - Prediction: Top-trained small nets respond like the hls4ml-trained ones.

## Procedure
- **Training** (`interp/h16_train.py`). The networks are binary taggers, class 1 = top.
  - jedis = `JediStyleTagger(d=64)`, about 19.5 k parameters.
  - tf = `TransformerTagger(d=32, 2 blocks, 2 heads)`.
  - pfn = PFN with Φ 3-100-100-128 and F 100-100-100.
  - Each sees the 64 hardest particles as (f, Δη, Δφ), with f = pT_i / Σ₆₄ pT ("frac") or f = pT_i in GeV ("abs").
  - Each feature is standardised over the real particles.
  - Optimiser: AdamW, lr 1e-3 with cosine decay, 30 epochs, early stop on validation loss (patience 5).
  - Seeds 0 and 1.
- **Datasets.** Every dataset has n_train = 89,473 per class, the number of in-window hls4ml t jets.
  - hlsA: hls4ml t vs (q ∪ g), full spectrum. The training pT_J 0.5–99.5 % range is 676–1660 GeV.
  - hlsB: the same jets restricted to ±8 % of the hls4ml median pT_J (1017 GeV), i.e. 936–1099 GeV. This window has the same relative width as Top Landscape's 550–650 GeV.
  - top: Top Landscape t vs QCD.
  - pT_J = |Σ over the 150 stored particles of the pT vector|.
  - hls4ml validation uses the `splits.npz` dev rows (t and q/g only; the W/Z inclusion was fixed before training). hls4ml test uses the val archive. Top uses its val and test splits. All rows are random, never first-k.
  - Arms: frac on hlsA, hlsB and top; abs on hlsA and top. That gives 15 configs × 2 seeds = 30 nets.
- **Battery samples** (`interp/h16_boost_dataset.py`). Seed 4161; random rows; 5,000 t + 5,000 bkg each.
  - hls_all: hls4ml val archive. Particles are massless at (η, φ) = (Δη, Δφ), with the axis at (0, 0).
  - hls_win: the same, restricted to the hlsB window.
  - top: Top Landscape test split, using all constituent four-vectors and the E-scheme axis.
  - Native sample of each net: hlsA → hls_all, hlsB → hls_win, top → top.
  - The hlsA nets were also scored on hls_win.
- **Edits.** λ ∈ {0.9, 0.95, 1.05, 1.1} is the factor on the jet pT. Inputs are re-measured about the edited stored axis.
  - B: exact Lorentz boost along the jet momentum, |P| → λ|P|. It keeps m and m_A = √(M0·M2) fixed; angles scale by about 1/λ.
  - P: pT-only scaling, all four-vectors × λ. This is a no-op for frac inputs.
  - A: angle-only scaling, (Δη, Δφ) × 1/λ.
  - R (control): rotation about the axis, with the pT-weighted mean displacement matched to B per jet.
  - M (m held, m_A moved): every particle is translated by t·u, with u a random direction and the axis kept. t is solved per jet so that m_A → k·m_A, with k = 1/0.95 or 1/0.9 (the m_A change of A at λ 0.95 / 0.9). Invariant m and jet pT stay exactly fixed.
  - RM (control for M): rotation with pT-weighted mean displacement t.
- **Metrics.**
  - % of binary calls (argmax) changed, with a 95 % bootstrap over jets.
  - The same on the in-window subset: jets whose pT_J before and after the edit lie in the net's training window.
  - Signal efficiency and 1/ε_B at the threshold that gives ε_S = 0.5 on the unedited jets.
  - Exact McNemar tests B vs R, B vs P, B vs A and M vs RM per net × λ, Holm-corrected within each family (192 / 192 / 192 / 96 tests).
  - Dataset contrasts of the seed-mean B % changed (hlsA − top, hlsB − top, hlsA − hlsB), normal approximation, Holm over 48 contrasts (36 frac, 12 abs).
  - Seed spread.

## Checks
- **Unedited accuracy.** Every net reproduces its test.json accuracy within 0.42 pp on its battery sample (table below).
  - Top-trained nets: 91.3–92.9 %. hlsA nets: 88.1–90.1 %. hlsB nets: 89.5–90.2 %.
  - For reference, the H12 Top PFN gave 92.85 %.
- **Edit fidelity.**
  - B0.9 keeps invariant m to 6e-13 (99th percentile of |Δm/m|).
  - B0.9 keeps m_A to 0.13 % (hls4ml, p99) and 0.55 % (Top, p99).
  - B0.9 changes jet pT by exactly −10.0 %.
  - M(1/0.9) gives m_A +11.11 % with |Δm/m| ≤ 4e-9 and |ΔpT_J| ≤ 2e-16 (p99).
  - R changes m by ≤ 3e-6 on average.
- **Round trip.** Converting the table back to inputs reproduces the stored 64-slot hls4ml inputs exactly (max diff 0).
  - On Top, the per-slot differences come from pT-tie slot swaps in the stored conversion.
  - Slot-order-free check on Top: the centroids agree to ≤ 8e-5 and ⟨r²⟩ to ≤ 6e-5.
- **Seed spread.** For B0.9 the largest seed-to-seed difference is 0.85 pp (top-abs-jedis, 4.70 vs 5.55 %). Most configs differ by ≤ 0.5 pp.
- **Cross-check with H12.** The H12 Top PFN (released Top PFN, different sample) changed 3.8 % at B0.9 and 3.9 % at A0.9. The Top-trained PFNs here change 3.8–5.0 % and 4.8–5.3 %.

## Numbers
### Unedited accuracy on the native battery sample (s0 / s1) vs test.json; AUC; 1/eps_B at eps_S 0.5
| config | sample | accuracy % | test.json % | AUC | 1/eps_B |
|---|---|---|---|---|---|
| hlsA-frac-jedis | hls_all | 88.30 / 88.29 | 88.58 / 88.43 | 0.9515 / 0.9506 | 70 / 62 |
| hlsB-frac-jedis | hls_win | 89.66 / 89.72 | 89.65 / 89.84 | 0.9591 / 0.9605 | 65 / 68 |
| top-frac-jedis | top | 91.27 / 91.61 | 90.98 / 91.23 | 0.9689 / 0.9718 | 88 / 96 |
| hlsA-frac-tf | hls_all | 88.16 / 88.34 | 88.47 / 88.45 | 0.9515 / 0.9523 | 72 / 74 |
| hlsB-frac-tf | hls_win | 89.48 / 89.74 | 89.89 / 89.84 | 0.9603 / 0.9602 | 89 / 79 |
| top-frac-tf | top | 92.47 / 92.50 | 92.18 / 92.25 | 0.9761 / 0.9766 | 132 / 128 |
| hlsA-frac-pfn | hls_all | 88.90 / 88.72 | 88.92 / 88.83 | 0.9555 / 0.9552 | 100 / 94 |
| hlsB-frac-pfn | hls_win | 90.24 / 90.02 | 90.17 / 90.16 | 0.9649 / 0.9646 | 139 / 122 |
| top-frac-pfn | top | 92.63 / 92.52 | 92.35 / 92.37 | 0.9780 / 0.9773 | 143 / 156 |
| hlsA-abs-jedis | hls_all | 89.16 / 89.34 | 89.41 / 89.61 | 0.9575 / 0.9588 | 79 / 88 |
| top-abs-jedis | top | 91.74 / 91.44 | 91.47 / 91.02 | 0.9714 / 0.9673 | 93 / 77 |
| hlsA-abs-tf | hls_all | 88.20 / 88.09 | 88.46 / 88.32 | 0.9512 / 0.9512 | 64 / 82 |
| top-abs-tf | top | 92.36 / 92.56 | 92.21 / 92.36 | 0.9762 / 0.9766 | 119 / 119 |
| hlsA-abs-pfn | hls_all | 90.07 / 89.95 | 90.17 / 90.16 | 0.9627 / 0.9633 | 114 / 119 |
| top-abs-pfn | top | 92.72 / 92.88 | 92.63 / 92.55 | 0.9781 / 0.9779 | 128 / 132 |

### % binary calls changed at lambda 0.9, native sample, s0 / s1 (95 % bootstrap CI of s0)
| config | B | CI(s0) | P | A | R | B in-window (n) | B/A | B/P |
|---|---|---|---|---|---|---|---|---|
| hlsA-frac-jedis | 4.80 / 5.13 | [4.36, 5.23] | 0.00 / 0.00 | 4.85 / 5.18 | 0.52 / 0.60 | 4.81 / 5.13 (9867) | 0.99 | — (P no-op) |
| hlsB-frac-jedis | 4.35 / 4.91 | [3.97, 4.76] | 0.00 / 0.00 | 4.36 / 4.93 | 0.49 / 0.40 | 4.23 / 5.18 (2200) | 1.00 | — (P no-op) |
| top-frac-jedis | 5.17 / 5.08 | [4.73, 5.60] | 0.00 / 0.00 | 5.36 / 5.31 | 0.47 / 0.78 | 5.73 / 5.45 (3174) | 0.96 | — (P no-op) |
| hlsA-frac-tf | 4.86 / 5.00 | [4.46, 5.28] | 0.00 / 0.00 | 4.93 / 5.01 | 0.53 / 0.32 | 4.80 / 4.97 (9867) | 0.99 | — (P no-op) |
| hlsB-frac-tf | 4.76 / 4.31 | [4.33, 5.18] | 0.00 / 0.00 | 4.81 / 4.38 | 0.43 / 0.35 | 4.14 / 4.32 (2200) | 0.99 | — (P no-op) |
| top-frac-tf | 3.96 / 4.14 | [3.55, 4.38] | 0.00 / 0.00 | 4.08 / 4.30 | 0.51 / 0.42 | 4.41 / 4.51 (3174) | 0.97 | — (P no-op) |
| hlsA-frac-pfn | 5.54 / 5.11 | [5.07, 6.01] | 0.00 / 0.00 | 5.62 / 5.19 | 0.65 / 0.72 | 5.52 / 5.08 (9867) | 0.99 | — (P no-op) |
| hlsB-frac-pfn | 5.26 / 5.26 | [4.81, 5.72] | 0.00 / 0.00 | 5.34 / 5.33 | 0.64 / 0.66 | 4.27 / 4.50 (2200) | 0.99 | — (P no-op) |
| top-frac-pfn | 4.98 / 4.76 | [4.57, 5.38] | 0.00 / 0.00 | 5.02 / 4.84 | 0.84 / 0.74 | 4.57 / 4.41 (3174) | 0.99 | — (P no-op) |
| hlsA-abs-jedis | 5.75 / 6.29 | [5.29, 6.19] | 3.12 / 3.84 | 4.02 / 4.35 | 0.45 / 0.56 | 5.65 / 6.17 (9867) | 1.44 | 1.73 |
| top-abs-jedis | 4.70 / 5.55 | [4.27, 5.07] | 4.64 / 4.41 | 5.29 / 5.68 | 0.63 / 0.64 | 3.97 / 4.57 (3174) | 0.93 | 1.13 |
| hlsA-abs-tf | 4.72 / 4.47 | [4.32, 5.15] | 1.43 / 1.40 | 4.19 / 4.42 | 0.69 / 0.60 | 4.67 / 4.41 (9867) | 1.07 | 3.25 |
| top-abs-tf | 4.09 / 4.43 | [3.70, 4.50] | 3.97 / 3.62 | 4.48 / 4.45 | 0.55 / 0.44 | 4.06 / 3.88 (3174) | 0.95 | 1.12 |
| hlsA-abs-pfn | 6.94 / 6.94 | [6.42, 7.42] | 4.76 / 5.36 | 4.56 / 4.19 | 0.60 / 0.54 | 6.95 / 6.96 (9867) | 1.59 | 1.37 |
| top-abs-pfn | 4.19 / 3.76 | [3.81, 4.61] | 3.24 / 3.26 | 5.27 / 4.90 | 0.76 / 0.66 | 3.75 / 3.31 (3174) | 0.78 | 1.22 |

### % binary calls changed at lambda 1.1, native sample, s0 / s1 (95 % bootstrap CI of s0)
| config | B | CI(s0) | P | A | R | B in-window (n) | B/A | B/P |
|---|---|---|---|---|---|---|---|---|
| hlsA-frac-jedis | 4.27 / 4.58 | [3.91, 4.67] | 0.00 / 0.00 | 4.31 / 4.61 | 0.44 / 0.56 | 4.19 / 4.51 (9857) | 0.99 | — (P no-op) |
| hlsB-frac-jedis | 3.96 / 3.84 | [3.57, 4.33] | 0.00 / 0.00 | 3.99 / 3.86 | 0.42 / 0.29 | 3.95 / 4.07 (4173) | 0.99 | — (P no-op) |
| top-frac-jedis | 5.26 / 4.89 | [4.85, 5.69] | 0.00 / 0.00 | 5.51 / 5.12 | 0.44 / 0.56 | 3.96 / 3.86 (4794) | 0.95 | — (P no-op) |
| hlsA-frac-tf | 4.53 / 4.58 | [4.10, 4.92] | 0.00 / 0.00 | 4.55 / 4.60 | 0.41 / 0.26 | 4.46 / 4.53 (9857) | 1.00 | — (P no-op) |
| hlsB-frac-tf | 3.81 / 3.87 | [3.43, 4.18] | 0.00 / 0.00 | 3.84 / 3.89 | 0.42 / 0.30 | 4.22 / 4.24 (4173) | 0.99 | — (P no-op) |
| top-frac-tf | 4.20 / 4.47 | [3.82, 4.59] | 0.00 / 0.00 | 4.30 / 4.75 | 0.50 / 0.32 | 3.71 / 3.57 (4794) | 0.96 | — (P no-op) |
| hlsA-frac-pfn | 5.15 / 5.25 | [4.73, 5.60] | 0.00 / 0.00 | 5.18 / 5.31 | 0.52 / 0.58 | 5.09 / 5.21 (9857) | 0.99 | — (P no-op) |
| hlsB-frac-pfn | 4.74 / 4.65 | [4.28, 5.19] | 0.00 / 0.00 | 4.73 / 4.69 | 0.54 / 0.57 | 4.94 / 4.60 (4173) | 1.00 | — (P no-op) |
| top-frac-pfn | 4.86 / 4.70 | [4.43, 5.28] | 0.00 / 0.00 | 5.09 / 5.02 | 0.68 / 0.62 | 3.67 / 4.05 (4794) | 0.95 | — (P no-op) |
| hlsA-abs-jedis | 7.02 / 7.58 | [6.55, 7.58] | 3.77 / 3.98 | 3.98 / 4.15 | 0.42 / 0.39 | 7.06 / 7.58 (9857) | 1.80 | 1.88 |
| top-abs-jedis | 3.23 / 3.13 | [2.87, 3.57] | 2.70 / 2.45 | 5.52 / 5.37 | 0.63 / 0.58 | 2.86 / 2.77 (4794) | 0.58 | 1.23 |
| hlsA-abs-tf | 4.27 / 4.72 | [3.88, 4.67] | 1.15 / 1.34 | 3.77 / 4.21 | 0.58 / 0.48 | 4.25 / 4.73 (9857) | 1.13 | 3.61 |
| top-abs-tf | 3.01 / 3.03 | [2.67, 3.34] | 2.50 / 2.63 | 4.56 / 4.76 | 0.35 / 0.46 | 2.92 / 2.86 (4794) | 0.65 | 1.18 |
| hlsA-abs-pfn | 8.57 / 9.00 | [8.04, 9.14] | 5.64 / 6.16 | 4.10 / 4.17 | 0.42 / 0.48 | 8.66 / 9.08 (9857) | 2.12 | 1.49 |
| top-abs-pfn | 3.03 / 2.73 | [2.71, 3.37] | 2.38 / 2.42 | 4.91 / 4.67 | 0.56 / 0.51 | 2.86 / 2.84 (4794) | 0.60 | 1.20 |

### m held, m_A moved (M, k = 1/0.9 and 1/0.95) vs matched rotation RM; % calls changed, s0 / s1
| config | M 1/0.95 | RM | M 1/0.9 | CI(s0) | RM |
|---|---|---|---|---|---|
| hlsA-frac-jedis | 4.84 / 4.83 | 1.40 / 1.32 | 7.77 / 8.01 | [7.23, 8.29] | 1.54 / 1.83 |
| hlsB-frac-jedis | 3.83 / 4.41 | 1.21 / 1.21 | 6.64 / 7.43 | [6.20, 7.16] | 1.47 / 1.39 |
| top-frac-jedis | 4.71 / 5.49 | 1.85 / 2.02 | 8.76 / 10.82 | [8.19, 9.31] | 2.26 / 2.51 |
| hlsA-frac-tf | 4.52 / 4.10 | 1.83 / 1.61 | 6.64 / 6.63 | [6.14, 7.12] | 2.38 / 2.04 |
| hlsB-frac-tf | 5.66 / 4.62 | 1.78 / 1.25 | 8.47 / 7.87 | [7.92, 8.98] | 2.38 / 1.80 |
| top-frac-tf | 4.86 / 6.09 | 1.81 / 1.47 | 7.68 / 10.36 | [7.20, 8.20] | 2.43 / 2.05 |
| hlsA-frac-pfn | 4.88 / 4.85 | 1.43 / 1.76 | 8.91 / 9.01 | [8.36, 9.52] | 1.84 / 2.21 |
| hlsB-frac-pfn | 4.59 / 4.54 | 1.65 / 1.56 | 8.15 / 8.49 | [7.60, 8.66] | 1.93 / 1.69 |
| top-frac-pfn | 6.71 / 5.89 | 2.02 / 1.88 | 12.02 / 11.08 | [11.36, 12.62] | 2.38 / 2.15 |
| hlsA-abs-jedis | 4.73 / 4.45 | 1.36 / 1.63 | 7.56 / 7.16 | [7.06, 8.05] | 1.86 / 1.77 |
| top-abs-jedis | 5.20 / 5.01 | 1.91 / 1.82 | 9.82 / 10.88 | [9.28, 10.41] | 2.44 / 2.09 |
| hlsA-abs-tf | 5.67 / 4.96 | 2.16 / 2.12 | 8.15 / 7.74 | [7.62, 8.72] | 2.92 / 2.73 |
| top-abs-tf | 4.56 / 5.55 | 1.75 / 1.48 | 7.56 / 9.75 | [7.07, 8.09] | 2.53 / 1.93 |
| hlsA-abs-pfn | 4.05 / 4.63 | 1.45 / 1.80 | 7.21 / 8.19 | [6.72, 7.70] | 1.95 / 1.97 |
| top-abs-pfn | 5.74 / 6.10 | 1.87 / 1.87 | 10.76 / 11.00 | [10.19, 11.33] | 2.11 / 2.22 |

### Working point (threshold fixed at unedited eps_S = 0.5): signal efficiency and 1/eps_B, s0 / s1
| config | 1/eps_B unedited | B0.9 eps_S | B0.9 1/eps_B | B1.1 eps_S | B1.1 1/eps_B | R0.9 eps_S |
|---|---|---|---|---|---|---|
| hlsA-frac-jedis | 70 / 62 | 0.530 / 0.523 | 40 / 41 | 0.377 / 0.388 | 106 / 82 | 0.494 / 0.501 |
| hlsB-frac-jedis | 65 / 68 | 0.491 / 0.452 | 38 / 45 | 0.337 / 0.292 | 116 / 125 | 0.497 / 0.504 |
| top-frac-jedis | 88 / 96 | 0.447 / 0.452 | 79 / 106 | 0.402 / 0.395 | 100 / 102 | 0.499 / 0.500 |
| hlsA-frac-tf | 72 / 74 | 0.556 / 0.540 | 41 / 46 | 0.395 / 0.393 | 98 / 96 | 0.501 / 0.500 |
| hlsB-frac-tf | 89 / 79 | 0.515 / 0.513 | 54 / 48 | 0.385 / 0.354 | 147 / 128 | 0.501 / 0.500 |
| top-frac-tf | 132 / 128 | 0.514 / 0.506 | 104 / 109 | 0.400 / 0.386 | 161 / 147 | 0.504 / 0.495 |
| hlsA-frac-pfn | 100 / 94 | 0.522 / 0.526 | 51 / 50 | 0.386 / 0.390 | 139 / 152 | 0.505 / 0.497 |
| hlsB-frac-pfn | 139 / 122 | 0.442 / 0.461 | 64 / 62 | 0.338 / 0.328 | 172 / 167 | 0.499 / 0.500 |
| top-frac-pfn | 143 / 156 | 0.484 / 0.481 | 139 / 135 | 0.396 / 0.387 | 172 / 161 | 0.501 / 0.499 |
| hlsA-abs-jedis | 79 / 88 | 0.576 / 0.578 | 42 / 41 | 0.303 / 0.272 | 192 / 238 | 0.494 / 0.501 |
| top-abs-jedis | 93 / 77 | 0.277 / 0.261 | 250 / 200 | 0.555 / 0.574 | 46 / 35 | 0.498 / 0.501 |
| hlsA-abs-tf | 64 / 82 | 0.556 / 0.558 | 42 / 47 | 0.389 / 0.366 | 102 / 152 | 0.504 / 0.502 |
| top-abs-tf | 119 / 119 | 0.391 / 0.385 | 161 / 167 | 0.508 / 0.503 | 91 / 78 | 0.499 / 0.503 |
| hlsA-abs-pfn | 114 / 119 | 0.535 / 0.543 | 58 / 51 | 0.333 / 0.321 | 294 / 357 | 0.503 / 0.502 |
| top-abs-pfn | 128 / 132 | 0.328 / 0.323 | 250 / 238 | 0.537 / 0.556 | 96 / 75 | 0.501 / 0.498 |

### Dataset contrasts of seed-mean B % changed (native samples), normal approx., Holm over 48 contrasts
| contrast | a | b | a − b [95 % CI] | p | Holm |
|---|---|---|---|---|---|
| frac-jedis|B0.9|hlsA-top | 4.96 | 5.12 | -0.16 [-0.59, +0.27] | 0.46 | 1 |
| frac-jedis|B0.9|hlsB-top | 4.63 | 5.12 | -0.50 [-0.92, -0.07] | 0.022 | 0.56 |
| frac-jedis|B0.9|hlsA-hlsB | 4.96 | 4.63 | +0.34 [-0.08, +0.75] | 0.12 | 1 |
| frac-jedis|B1.1|hlsA-top | 4.42 | 5.08 | -0.65 [-1.07, -0.23] | 0.0022 | 0.078 |
| frac-jedis|B1.1|hlsB-top | 3.90 | 5.08 | -1.18 [-1.58, -0.77] | 1.4e-08 | 5.5e-07 |
| frac-jedis|B1.1|hlsA-hlsB | 4.42 | 3.90 | +0.52 [+0.13, +0.92] | 0.0086 | 0.27 |
| frac-jedis|B1.05|hlsA-top | 2.27 | 2.35 | -0.09 [-0.38, +0.20] | 0.55 | 1 |
| frac-jedis|B1.05|hlsB-top | 1.92 | 2.35 | -0.43 [-0.72, -0.15] | 0.0026 | 0.089 |
| frac-jedis|B1.05|hlsA-hlsB | 2.27 | 1.92 | +0.34 [+0.06, +0.63] | 0.016 | 0.46 |
| frac-tf|B0.9|hlsA-top | 4.93 | 4.05 | +0.88 [+0.47, +1.29] | 2.1e-05 | 0.00081 |
| frac-tf|B0.9|hlsB-top | 4.54 | 4.05 | +0.48 [+0.09, +0.88] | 0.017 | 0.47 |
| frac-tf|B0.9|hlsA-hlsB | 4.93 | 4.54 | +0.39 [-0.02, +0.81] | 0.063 | 1 |
| frac-tf|B1.1|hlsA-top | 4.55 | 4.33 | +0.22 [-0.18, +0.62] | 0.29 | 1 |
| frac-tf|B1.1|hlsB-top | 3.84 | 4.33 | -0.49 [-0.88, -0.11] | 0.012 | 0.37 |
| frac-tf|B1.1|hlsA-hlsB | 4.55 | 3.84 | +0.71 [+0.32, +1.11] | 0.00036 | 0.013 |
| frac-tf|B1.05|hlsA-top | 2.22 | 2.14 | +0.08 [-0.20, +0.37] | 0.56 | 1 |
| frac-tf|B1.05|hlsB-top | 1.97 | 2.14 | -0.17 [-0.44, +0.11] | 0.24 | 1 |
| frac-tf|B1.05|hlsA-hlsB | 2.22 | 1.97 | +0.25 [-0.03, +0.53] | 0.081 | 1 |
| frac-pfn|B0.9|hlsA-top | 5.33 | 4.87 | +0.45 [+0.02, +0.89] | 0.039 | 0.96 |
| frac-pfn|B0.9|hlsB-top | 5.26 | 4.87 | +0.39 [-0.04, +0.82] | 0.075 | 1 |
| frac-pfn|B0.9|hlsA-hlsB | 5.33 | 5.26 | +0.06 [-0.37, +0.50] | 0.77 | 1 |
| frac-pfn|B1.1|hlsA-top | 5.20 | 4.78 | +0.42 [-0.01, +0.85] | 0.054 | 1 |
| frac-pfn|B1.1|hlsB-top | 4.70 | 4.78 | -0.08 [-0.50, +0.33] | 0.69 | 1 |
| frac-pfn|B1.1|hlsA-hlsB | 5.20 | 4.70 | +0.50 [+0.08, +0.93] | 0.02 | 0.54 |
| frac-pfn|B1.05|hlsA-top | 2.67 | 2.28 | +0.39 [+0.09, +0.69] | 0.012 | 0.37 |
| frac-pfn|B1.05|hlsB-top | 2.46 | 2.28 | +0.18 [-0.12, +0.47] | 0.25 | 1 |
| frac-pfn|B1.05|hlsA-hlsB | 2.67 | 2.46 | +0.21 [-0.09, +0.52] | 0.17 | 1 |
| abs-jedis|B0.9|hlsA-top | 6.02 | 5.12 | +0.90 [+0.45, +1.34] | 9.5e-05 | 0.0035 |
| abs-jedis|B1.1|hlsA-top | 7.30 | 3.18 | +4.12 [+3.69, +4.55] | 5.7e-77 | 2.7e-75 |
| abs-jedis|B1.05|hlsA-top | 3.65 | 1.53 | +2.12 [+1.81, +2.43] | 9.7e-41 | 4.4e-39 |
| abs-tf|B0.9|hlsA-top | 4.59 | 4.26 | +0.33 [-0.07, +0.74] | 0.1 | 1 |
| abs-tf|B1.1|hlsA-top | 4.50 | 3.02 | +1.48 [+1.10, +1.85] | 8.3e-15 | 3.5e-13 |
| abs-tf|B1.05|hlsA-top | 2.27 | 1.65 | +0.62 [+0.35, +0.90] | 6.4e-06 | 0.00025 |
| abs-pfn|B0.9|hlsA-top | 6.94 | 3.98 | +2.97 [+2.52, +3.41] | 4.2e-39 | 1.9e-37 |
| abs-pfn|B1.1|hlsA-top | 8.79 | 2.88 | +5.91 [+5.45, +6.36] | 2.5e-142 | 1.2e-140 |
| abs-pfn|B1.05|hlsA-top | 4.30 | 1.45 | +2.85 [+2.53, +3.18] | 7.2e-66 | 3.3e-64 |

max seed spread B0.9: 0.0085


## Statistics summary
- **B vs R.** In all 192 net × λ × sample tests, B changes more calls than the matched rotation. The largest Holm p is 3e-18.
  - On the native samples, B/R = 5.6–12.2 at λ 0.9 and 5.3–19.5 at λ 1.1.
- **B vs A, frac inputs.** 95 of the 96 frac tests on the hls samples have Holm p > 0.05, and so do 18 of 24 on Top. B and A differ by ≤ 0.32 pp.
  - Reason: a longitudinal boost multiplies every constituent pT by close to the same factor, so pT fractions are unchanged. The net then sees only the angular rescaling.
- **B vs P and B vs A, abs inputs.**
  - hlsA-abs nets: B > P (+1.6 to +3.6 pp, Holm ≤ 4e-16) and B ≥ A (+0.05 to +4.8 pp; 11 of 12 tests Holm < 0.01).
  - top-abs nets: B ≈ P at λ 0.9 (+0.06 to +1.14 pp; 3 of 6 tests Holm < 0.01) and B < A at λ 1.1 (−1.6 to −2.3 pp, Holm ≤ 2e-17).
- **Dataset contrasts, frac inputs, B0.9.** hlsA − top lies between −0.16 and +0.88 pp; hlsB − top between −0.50 and +0.48 pp; hlsA − hlsB between +0.06 and +0.39 pp.
  - 3 of the 36 frac contrasts (λ 0.9 / 0.95 / 1.05 / 1.1) are Holm-significant: frac-tf B0.9 hlsA − top +0.88 pp (Holm 8e-4); frac-tf B1.1 hlsA − hlsB +0.71 pp (Holm 0.013); frac-jedis B1.1 hlsB − top −1.18 pp (Holm 6e-7).
  - The signs of the significant contrasts are mixed.
- **Dataset contrasts, abs inputs.** hlsA-abs − top-abs is +0.33 to +2.97 pp at B0.9 and +1.48 to +5.91 pp at B1.1. 10 of the 12 abs contrasts are Holm < 4e-3, all positive (hls4ml-trained nets respond more).
- **M vs RM.** In all 96 tests, the m-held, m_A-moved translation changes more calls than the matched rotation (Holm ≤ 1e-29).
  - At k = 1/0.9: M 6.6–12.0 %, RM 1.4–2.9 %.

## Hypotheses: predictions vs measurement
| hypothesis | prediction | measured (B0.9, native samples, seed means) | status |
|---|---|---|---|
| H1, dataset | Top small nets B 1–5 % with B/P ≤ 0.5; hlsA/hlsB 20–35 % | Top small nets 4.0–5.1 %, B/P 1.12–1.22 (abs), B/A 0.78–0.99; hlsA 4.6–6.9 %, hlsB 4.5–5.3 % | excluded: B/P ≤ 0.5 is not met for any Top net, and the hls nets are not at 20–35 % on this binary task |
| H1(b), pT-window width | hlsB drops to the Top level while hlsA does not | hlsA − hlsB frac +0.06 to +0.39 pp at B0.9, none Holm-significant; hlsB ≈ top | no distinguishing effect: hlsA, hlsB and top are within 1 pp |
| H0, architecture/inputs | Top small nets respond like hls ones | frac: |hls − top| ≤ 0.9 pp at λ 0.9 (≤ 1.2 pp at λ 1.1), B/A ≈ 1 for all 18 frac nets; abs: top nets 0.3–3.0 pp lower than hlsA | consistent (frac); the abs arm adds a dataset-dependent part of 0.3–3 pp |
| — | absolute level 20–35 % (both H1 and H0 copied H12's five-class numbers) | 4–7 % of binary t vs q/g calls | the level does not transfer from the five-class task to the binary one |

## Reading
- **Frac inputs.** All 18 frac-input nets respond to the boost exactly as to an angle-only rescaling, whatever the training sample (B/A = 0.95–1.00).
  - Trained on hlsA, hlsB or Top, they change 4.0–5.5 % of binary calls at λ 0.9 and 1.9–2.8 % at λ 0.95 / 1.05.
  - The rotation control gives 0.3–0.8 %.
  - These inputs carry no jet-pT scale. The network therefore sees m/pT_J (an angular size), not m, and a boost along the axis is indistinguishable from an angular rescaling. This is a property of the input representation. Training on Top Landscape's narrow window does not change it.
  - The same mechanism explains H12's pT-normalised hls4ml retrain (21–31 % five-class).
- **Abs inputs.** The training sample changes how the absolute pT enters.
  - hlsA-abs nets respond to B more than to either P or A (B/P 1.4–3.3, B/A 1.1–1.6 at λ 0.9). Lowering pT and widening angles push the score in the same direction.
  - At B0.9 they raise the signal efficiency at fixed threshold from 0.50 to 0.54–0.58, and 1/ε_B falls from 64–119 to 41–58.
  - top-abs nets respond to B about as much as to P, and less than to A at λ 1.1 (B/A 0.58–0.65).
  - At B0.9 they lower the signal efficiency from 0.50 to 0.26–0.39, and 1/ε_B rises to 160–250.
  - The Top-trained abs nets partly offset pT against angle, in the direction of a mass reading. They do not reach the near-invariance of the H12 Top ParT_kin (B0.9 1.1 % vs A0.9 4.1 %).
- **In-window subsets.** For hlsB and top, only 22 % (hlsB) and 32 % (top) of jets stay in the training window under B0.9.
  - For hlsB the in-window values are close to the full-sample ones: frac nets 4.1–5.2 % in window vs 4.3–5.3 %.
  - For top they are close too: frac nets 4.4–5.7 % in window vs 4.0–5.2 %; abs 3.3–4.6 % vs 3.8–5.6 %.
  - So leaving the training window does not account for the response.
- **M edit.** It holds the invariant mass and jet pT fixed and moves m_A by +11 %.
  - It changes 6.6–12.0 % of calls in every net, against 1.4–2.9 % for a rotation of the same pT-weighted displacement.
  - It changes more than A0.9, which moves both m and m_A by +11 % (4.0–5.7 %).
  - So none of these nets reads the invariant mass. All of them respond to the moments of the particles about the given axis.
  - Caveat (as in D4m): M also moves the pT centroid off the axis by t, while RM does not. M vs RM therefore measures "m_A plus centroid offset" against a centroid-preserving control, not m_A alone.
- **Relation to D9.** The hls4ml-vs-Top contrast in H12 compared different architectures (JEDI-linear and small nets vs ParT, LorentzNet, ParticleNet) on different tasks (five-class with W/Z vs binary). With architecture and task matched, the small nets give about the same boost response on both datasets. The 21–38 % vs 1–4 % contrast in D9 is therefore not attributable to the training sample by this measurement.
  - What remains open is how much of the five-class 21–38 % comes from the W/Z ↔ top boundary. H12v found 62 % of JEDI-linear's changed calls are W/Z → top.

## Open / next
- Score H12's five-class hls4ml models (JEDI-linear, small nets) on the binary t vs q/g decision under B0.9. This separates the task effect from the architecture effect on the same checkpoints.
- Train a ParT_kin-type or LorentzNet on hls4ml t vs q/g (abs four-vectors). If it comes out near-invariant (B ≪ A), the architecture/input claim extends to large models.
- Retrain the frac nets with the jet pT_J added as one global input, to test whether a pT scale alone lets them learn m.

## Explainer entry (for site/explain.py, H16_EXPLAINER)
- **Question.** Are small jet taggers sensitive to a boost along the jet axis because of the sample they were trained on (hls4ml) or because of what they are and what they see?
- **Physics background.**
  - The invariant jet mass $m^2 = E^2-|\vec P|^2$ is unchanged by a Lorentz boost along $\vec P$.
  - For a collinear jet, $m \approx p_{T,J}\,\theta_{\rm jet}$. Lowering $p_{T,J}$ by $\lambda$ at fixed $m$ widens all angles by $\approx 1/\lambda$.
  - A tagger that reads $m$ should not change its call. A tagger that reads angles alone, or pT alone, should.
  - With pT-fraction inputs $z_i = p_{T,i}/\sum p_T$, the boost leaves $z_i$ fixed. The network sees only $(\Delta\eta,\Delta\phi)\to(\Delta\eta,\Delta\phi)/\lambda$, which is the same as an angle-only rescaling.
- **Method.**
  - Train the same three small networks (JEDI-style, 2-block transformer, Particle Flow Network) on three samples (hls4ml full pT, hls4ml in $\pm8\%$ of the median pT, Top Landscape 550–650 GeV), with inputs $z_i$ or $p_{T,i}$ and two seeds each.
  - Apply to 10,000 test jets per sample (seed 4161):
    - B, the exact boost: $|\vec P|\to\lambda|\vec P|$, $m$ fixed.
    - P: $p_{T,i}\to\lambda p_{T,i}$.
    - A: angles $\to$ angles$/\lambda$.
    - R: a rotation with matched displacement.
    - M: a translation that keeps $m$ and $p_{T,J}$ and sets $m_A=\sqrt{M_0M_2}\to k\,m_A$.
  - Metric: % of top-vs-QCD calls changed, with McNemar tests and Holm correction.
- **Result** (figure `h16_boost_dataset.png`).
  - At $\lambda=0.9$, every net changes 4–7 % of calls (rotation 0.3–0.8 %).
  - Top-trained and hls4ml-trained nets differ by under 1 percentage point with $z_i$ inputs, and by 0.3–3 points with absolute pT.
  - With $z_i$ inputs, B and A give the same calls.
  - M (fixed $m$) changes 7–12 %, more than A.
- **Checks.**
  - Unedited accuracies reproduce test values within 0.4 pp.
  - B keeps $m$ to $10^{-12}$ and $m_A$ to 0.5 % (p99).
  - The in-window subsets give the same picture.
  - Seed differences are ≤ 0.85 pp.
  - The Top PFN response agrees with the released-model value from H12.
- **Physical significance.**
  - The boost response of small taggers is set by their inputs and architecture, not by the training sample.
  - A network fed only pT fractions and angles cannot read the invariant mass. It reads $m/p_T$ and moments about the jet axis.
  - The near boost-invariance of large Top-tagging models (H12) is a property of those models, not of the Top Landscape sample.

