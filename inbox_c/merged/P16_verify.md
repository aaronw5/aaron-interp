# P16v: verification of P16 (planted-label ParT arms), 2026-10-20

## Conclusion
Particle Transformers trained from scratch to flag a jet-mass window respond to a rotation that leaves the invariant mass unchanged at about 0.2 of their response to an equal real mass change, while the released ParT models respond at about 0.75; this gap holds on fresh jets and when the released models are read out as "W or Z vs QCD" instead of "W vs Z". So the ParT architecture with these inputs can learn a mostly rotation-invariant mass, and the released models' axis-relative response is not forced by the architecture and inputs. The label built on the axis-relative mass equals the invariant-mass label on almost every real jet, so arm b does not show whether the training target decides which mass the network reads. Which difference from the released training (number of classes, data size, training length, random seed) produces the larger response is not identified; the scratch networks' value also shifts during their own training.

## Verdict fields
- Reproduced: yes (all Test 1 and Test 2 numbers within 0.01 on fresh rows).
- Refuted: no for the measured numbers and the "architecture does not force it" reading; the attribution sentences need narrowing (below).
- Confidence that the P16 reading is correct as stated: 0.55.

## Code review (interp/p14_planted_part.py, interp/p14_eap_part.py not rerun, interp/p14_planted_train.py)
- Sample: random indices over the three test files (`default_rng(14).choice`), not first-k. OK.
- Build: `part_core.build` uses η = jet_eta + sign(jet_eta)·Δη (the fixed version). OK.
- Padded particles: the phi edit is `dp + 0.06 * real` with real = pT > 0; moments and masses are masked. OK.
- Edits: phi changes mean Δlog m_inv by −4×10⁻⁸ and Δlog m_A by +0.123 on the fresh rows; dil matches Δlog m_A per jet (Δlog m_inv +0.119); rot null gives R ≈ 10⁻⁴. The edits do what is stated.
- Arm c feed zeroes e_log, logerel, deltaR, deta, dphi; its phi response equals the rot null (calls changed 0.025 % in both). OK.
- No layer/hook bug found in Tests 1–2 (they use only logits and the `fc` input hook).
- Training script: arms a, b, c and s all use `torch.manual_seed(14)`, the same validation split and the same batch order (`seed=1000+ep`). Arms a and b therefore share initialisation and data order and differ only in 2.3 % of labels. The permutation test R_a ≠ R_b (p = 0.0005) resamples jets only; it does not include training-seed variance, which is not measured.
- Figure `results/part/p14_planted_p16.png`: matches the JSON. Arm s is plotted with an R bar of −0.13 although its responses are numerical noise; the tick label says "(constant)".

## Rerun (job anrunw-ai-c-p16-verify, Complete, 2 min 50 s; script interp/p16v_verify.py)
- Sample: N = 20,000 random rows (`default_rng(16)`) of `test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_101.root` (6,610 / 6,672 / 6,718). Labels a and b agree on 97.7 %; corr(m_inv, m_A) = 0.9989.
- Edits, read-outs, R and route split as P16; bootstrap 2,000 resamples over jets; paired permutation test (2,000).
- Results: `results/part/p16_verify.json`, figure `results/part/p16_verify.png` (its left legend repeats "P16 read-out" once; cosmetic).

### Test 1 and route split, fresh rows (P16 values in parentheses)
| model | own-label / 10-class acc | R [95 % CI] | rot-null R | calls changed under phi | dil route per-particle / pair |
|---|---|---|---|---|---|
| a | 0.981 [0.979, 0.983] | 0.184 [0.180, 0.189] (0.185) | ≈ 0 | 5.9 % [5.5, 6.2] (5.85) | 0.177 / 0.827 (0.172 / 0.830) |
| b | 0.980 | 0.215 [0.210, 0.220] (0.215) | ≈ 0 | 6.5 % (6.25) | 0.215 / 0.791 (0.211 / 0.796) |
| c | 0.977 | 0.000 (0.000) | ≈ 0 | 0.025 % | −0.005 / 1.005 |
| released ParT_full | 10-class 0.755 on these W/Z/QCD jets | 0.793 [0.784, 0.802] (0.790) | 0.0001 | 19.2 % (18.5) | 0.839 / 0.149 (0.834 / 0.153) |
| released ParT_kin | 0.662 | 0.719 [0.709, 0.729] (0.723) | 0.009 | 31.4 % (31.0) | 0.817 / 0.181 (0.832 / 0.168) |

Permutation tests (H0: equal R, paired per jet): R_a − R_b = −0.031, R_a − R_released_kin = −0.535, R_a − R_released_full = −0.609, R_a − R_c = +0.184; all p = 0.0005 (floor).

### Added control 1: read-out of the released models
H0: the planted–released gap comes from comparing a window read-out (planted) with a W-vs-Z read-out (released). Prediction under H0: a window-like read-out of the released models gives R near 0.2.
- Read-out logsumexp(logit_W, logit_Z) − logit_QCD: R = 0.654 [0.642, 0.666] (full), 0.628 [0.614, 0.642] (kin); rot null 0.0003 / 0.0075.
- By true class (W-vs-Z read-out): W 0.79 / 0.71, Z 0.83 / 0.75, QCD 0.66 / 0.65 (full / kin).
- H0 is excluded: the gap to arm a (0.18) stays at 0.44–0.47 with the alternative read-out. The released R depends on the read-out by 0.09–0.14.

### Added control 2: training length (per-epoch checkpoints of the same runs)
H0: R of the planted arms is constant over training. Competing reading: R grows with training toward the released value (training-length confound).

| checkpoint | own-label acc | R [95 % CI] |
|---|---|---|
| a epoch 0 | 0.857 | 0.241 [0.234, 0.249] |
| a epoch 2 | 0.970 | 0.118 [0.112, 0.124] |
| a epoch 5 | 0.978 | 0.120 [0.116, 0.125] |
| a epoch 7 | 0.983 | 0.161 [0.157, 0.165] |
| a epoch 9 (final) | 0.981 | 0.184 [0.180, 0.189] |
| b epoch 2 | 0.969 | 0.134 [0.127, 0.140] |
| b epoch 5 | 0.975 | 0.138 [0.133, 0.143] |
| b epoch 9 (final) | 0.980 | 0.215 [0.210, 0.220] |

- R_a(ep 5) − R_a(ep 9) = −0.064, p = 0.0005; R_a(ep 0) − R_a(ep 9) = +0.057, p = 0.0005. H0 is excluded.
- R rises from 0.12 to 0.18–0.22 during the learning-rate decay epochs (7–9) at constant accuracy. It stays far below 0.72 within these 10 epochs (≈ 29k steps), but the trend does not exclude a rise with the much longer released training.

### Added check 3: window-edge accuracy (requested in P16 issues)
Arm a, by |m_inv − 85 GeV|: accuracy 1.000 for 0–8 GeV and ≥ 12 GeV; 0.898 (8–10 GeV, n = 1,402) and 0.809 (10–12 GeV, n = 1,221). Under phi, 31 % and 41 % of calls change in these two edge bins and 0–0.2 % at > 15 GeV or < 5 GeV. Arm b: same pattern (31 % / 44 %). Arm c: 0.2 % in the edge bins. The residual phi response of arms a and b is concentrated on jets within 2 GeV of the window edges, where their errors are.

## Points in the P16 reading that the evidence does not support
1. "H_data in its strict form is also excluded, because an m_A-window label gives 0.215, not about 1." On real jets m_A and m_inv correlate at 0.999 and labels a and b agree on 97.7 %; arm b also shares initialisation and batch order with arm a. The label b gives almost no training signal that separates the two masses, and none on rotated (off-axis) jets, which do not occur in training. Arm b therefore does not test whether the training target sets the mass that is read. A decorrelated label (e.g. a window on m_A / m_inv, or training on jets with randomised axis offsets) is needed.
2. "So it comes from the released models' training task or data." This is attribution by elimination over an incomplete list: training length (≈ 29k vs ~10⁶ steps), learning-rate schedule and training seed also differ and are not separated; R of the planted arms changes by 0.06–0.12 across their own epochs (control 2). Corrected: "comes from some difference of the released training setup (classes, data size, training length, seed), not identified here".
3. "DAS and EAP-IG did not separate the m_inv arm from the m_A arm" as method validation (C10). Arms a and b compute nearly the same function (R 0.18 vs 0.22, labels 97.7 % equal, same init and data order), so there was little planted difference for the tools to find. This result neither validates nor fails the tools. Also, in arm a block 8, the m_inv direction beat the m_A direction on discriminating pairs (p_Holm 0.009), so "did not separate" holds for arms b and c only.

Supported as stated: planted R ≈ 0.2 vs released 0.72–0.79; arm c R = 0; route reversal (planted 80–83 % through pair features, released 82–84 % through per-particle angles); the released response is not forced by architecture + kin inputs, at least for the 10-epoch, 3-class training here.

## Corrected reading
Planted ParT arms (kin inputs, 2 outputs, 10 epochs on 1.5M W/Z/QCD jets) give R = 0.18 (m_inv label) and 0.22 (m_A label) on fresh rows, against 0.79 / 0.72 for the released models with the W−Z read-out and 0.65 / 0.63 with a W-or-Z vs QCD read-out. The planted arms take 79–83 % of a real mass change through the pair features; the released models take 82–84 % through the per-particle angles. ParT with these inputs can therefore learn a mostly rotation-invariant mass. Because labels a and b coincide on 97.7 % of jets and the arms share initialisation and data order, arm b does not test whether the label sets the mass read. The source of the released models' larger R (classes, data size, training length, seed) is not identified; the planted R itself moves between 0.12 and 0.24 across epochs.

## Explainer additions
- Check: on fresh rows (test files _101, seed 16) all R values and route shares reproduce within 0.01.
- Check: with the read-out $\log(e^{\ell_W}+e^{\ell_Z})-\ell_{QCD}$ the released models give $R=0.65$ / $0.63$, so the gap to the planted arms is not a read-out artefact.
- Check: $R$ of arm a over epochs 0, 2, 5, 7, 9 is 0.24, 0.12, 0.12, 0.16, 0.18.

## Files
- /Users/anrunw/Documents/aaron-interp/interp/p16v_verify.py
- /Users/anrunw/Documents/aaron-interp/results/part/p16_verify.json
- /Users/anrunw/Documents/aaron-interp/results/part/p16_verify.png
- log /j-jepa-vol/aaron-interp/logs/c-p16-verify.log

## Addendum (second verifier pass, 2026-10-20)
- Re-read interp/p14_planted_part.py, interp/p16v_verify.py and the job log of anrunw-ai-c-p16-verify (git afa8311, RTX 2080 Ti, exit 0, 158 s). The JSON `results/part/p16_verify.json` matches the log line by line; no new job was launched (c-p16-verify already exists and is Complete; a further fresh-row rerun would repeat it).
- Checked again: random-index sampling (no first-k), fixed η build via `part_core.build`, phi edit applied only to real particles (`dp + 0.06 * real`), dilation factor k_j = sqrt(M0M2(phi)/M0M2(base)) so Δlog m_A is matched per jet (0.1230 vs 0.1230), rot null R ≤ 0.009 for every model, read-out hook on `fc` input only. No layer, head or padding bug found in Tests 1–2.
- Figure `results/part/p14_planted_p16.png`, right panel: arm s is drawn with route shares 0.63 / 0.37. Arm s outputs a constant, so these shares are ratios of numerical noise and carry no information; the same holds for its R bar of −0.13. Both bars should be removed or greyed.
- Missing control (not cheap: one training run takes 1.5–2.7 h on one GPU): a planted ParT_kin trained on the true W/Z/QCD class labels with the same 1.5M jets, 10 epochs and seed. Its R would separate "label type" (mass window vs class) from "data size / training length / 10 classes" as the source of the released models' R ≈ 0.72–0.79. Together with a decorrelated m_A label (window on m_A/m_inv, or training jets with random axis offsets) and a second training seed per arm, this is proposed for a follow-up node.
- Confidence in the P16 reading as stated stays at 0.55: the numbers and the "architecture with kin inputs does not force the axis-relative response" sentence hold; the "m_A label did not produce it" and "so it comes from the training task or data" sentences are not supported as worded (points 1–2 above).
