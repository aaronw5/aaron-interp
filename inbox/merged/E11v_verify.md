# E11 verification (skeptic agent, 2026-10-17)

Script: interp/e11_verify.py (modes core / fresh / edit / nonlive). Results: results/jedi/e11_verify_{core,fresh,edit,nonlive}.json.
Cluster logs: /j-jepa-vol/aaron-interp/logs/e11v-{core,fresh,edit,nonlive}.log (smoke runs e11v-smoke-*). All jobs ran at 1 CPU / 2 GiB.
Nothing was refitted: every number comes from the stored formulas e11_fit_*.json and e11hp_fit_*.json.
Jet sets: E11 test = sample(2000, seed 11) minus seed 0 (8,848 jets); fresh = sample(4000, seed 31) minus seeds 0 and 11 (17,011 jets; 2,989 removed).
Paired accuracy differences are given as (program right, network wrong) − (program wrong, network right) over N, ± the paired se.

## 1. Leakage and exactness (E11 test jets)
- Indices: 10,000 seed-11 jets, 1,152 overlap removed → 8,848. Test ∩ fit = 0, test ∩ dev = 0, fit ∩ dev = 0 (20,000 / 10,000 jets).
  Content hashes of the particle arrays: 0 test jets identical to any fit/dev jet.
- Code path of cumulative (b): chain values come only from the physics dict (stored Q + handbuilt(x)); no network activation is an input.
  Stored Q = compute(x) on the same 64 particles: recomputed on 10,000 fresh jets, max relative difference 0.0 over 383 keys; calls 100 % equal.
- All 16 z units are live, so the override replaces z completely. Logits of E11's all-stage override = z-only override = z-only override with
  every particle set to 0 = d7 applied to the program's z alone: max |Δlogit| 0 in each comparison.
- Last layer d7: ReLU, unsigned fixed-point input quantizer (RND, WRAP; i = 2–4 bits, f = 3–5 bits), then K·h + b. The network's own logits
  from its own z through this standalone d7: max |Δ| 0; float64 round/wrap path (jetdistill.network.logits_from_h): same call 100 %, max |Δ| 0.
  Wrap events: 0 jets for the program, 0 for the network.
- Non-live units (P 52, g 53, S 45, h4 10, h5 0, z 0) set to their mean over the 20,000 fit jets: 94.02 % / 80.72 % (published 94.0 / 80.72);
  0 calls differ from E11's version (hp: 94.64 / 80.75, 0 differ). The end-to-end number does not depend on non-live network values.
  The intermediate cumulative row P..h4 changes by 2 calls (hp 5); every other row by 0.
- Size: 144 units, 3,824 terms, 6,064 hinge factors (2,344 products), 394 distinct inputs, 268 of them physics quantities.
  E11's "211 physics quantities" is inconsistent with its own total (at most 128 inputs can be chain values); 268 is the count.
  hp: 3,711 terms, 5,862 hinge factors.
- Other code observations: input keys are filtered on test-split finiteness (no target or label information); the half-plane inputs were
  added after test R² had been seen (addressed by the fresh sample).

## 2. Fresh sample (17,011 jets; network 80.74 %, per class g/q/W/Z/t 79.3 / 71.9 / 87.0 / 83.2 / 82.1)
| program | same call | accuracy | Δacc vs network | per class g/q/W/Z/t |
|---|---|---|---|---|
| (b) chain | 94.23 % | 80.44 % | −0.31 ± 0.17 | 78.2 / 71.0 / 86.9 / 83.7 / 82.3 |
| (b) chain, half-plane inputs | 94.84 % | 80.52 % | −0.22 ± 0.16 | 78.6 / 71.2 / 86.9 / 83.3 / 82.5 |
| (a) z on physics | 92.50 % | 80.25 % | −0.49 ± 0.19 | 76.3 / 71.4 / 87.8 / 84.3 / 81.5 |
| E11 test jets, (b) chain | 94.02 % | 80.72 % | +0.29 ± 0.23 | |

Rejection at 50 % signal efficiency, program / network (ratio, bootstrap 95 % interval, 300 resamples):
| task | (b) | (b) half-plane | (a) z on physics |
|---|---|---|---|
| g vs rest (P_g) | 71.6 / 68.4 = 1.05 [0.94, 1.16] | 0.99 [0.90, 1.08] | 0.95 [0.85, 1.10] |
| q vs rest | 38.5 / 37.4 = 1.03 [0.95, 1.07] | 0.99 [0.92, 1.04] | 1.00 [0.93, 1.08] |
| W vs rest | 316 / 453 = 0.70 [0.50, 0.95] | 0.75 [0.52, 0.95] | 0.77 [0.50, 1.00] |
| Z vs rest | 974 / 1049 = 0.93 [0.50, 1.57] | 0.87 [0.49, 1.31] | 0.77 [0.38, 1.26] |
| t vs rest | 209 / 234 = 0.89 [0.74, 1.14] | 0.95 [0.82, 1.21] | 0.95 [0.73, 1.16] |
| t vs QCD (P_t/(P_t+P_g+P_q)) | 136 / 133 = 1.02 [0.73, 1.25] | 0.98 [0.78, 1.23] | 0.91 [0.67, 1.15] |
| g vs q / q vs g | 0.99 [0.84, 1.13] / 0.93 [0.83, 1.10] | 0.98 / 0.94 | 0.91 / 0.86 |
W and Z vs QCD: 1–6 of 6,791 QCD jets pass the 50 % cut in either model; not resolved at this sample size.

## 3. Disagreements (E11 test jets: 529 = 5.98 %; fresh: 982 = 5.77 %)
- By true class (test): g 5.8, q 6.9, W 5.4, Z 6.0, t 5.8 % of the class. Network → program flows: W→Z 78, Z→W 50, g→t 54, t→g 47,
  g→q 47, q→g 46. Network right 200, program right 226, neither 103 (fresh: 421 / 369 / 192).
- The network's top-2 softmax margin, median: 0.135 on disagreeing jets, 0.815 on agreeing jets.
- Per stage, cost-weighted RMS error / unit spread, agreeing → disagreeing (ratio): P 0.248 → 0.300 (1.21), g 0.316 → 0.410 (1.30),
  S 0.135 → 0.150 (1.11), h4 0.167 → 0.191 (1.14), h5 0.184 → 0.230 (1.25), z 0.212 → 0.309 (1.46; z:0 0.21 → 0.44).
  Cost-weighted R² agree / disagree: P .896 / .867, g .789 / .801, S .969 / .963, h4 .962 / .940, h5 .957 / .927, z .953 / .819. Fresh: same pattern.
- First stage at which cumulative (b) P..k leaves the network's call, of the 529: P 197, g 45, S 143, h4 53, h5 50, z 41.
  Half-plane run (474): P 58, g 47, S 204, h4 70, h5 49, z 46.
- Chain restarted from the network's true values at stage k (formulas downstream only), fraction of the 529 that return to the network's call:
  P 12 %, g 15 %, S 54 %, h4 62 %, h5 71 % (fresh: 13, 17, 55, 60, 71 %). With only the z formula (restart at h5) 2.95 % of calls still differ.

## 4. Single (b) at h4
- Reproduced: 95.71 % same call, 79.59 % accuracy, Δ −0.84 ± 0.20. Fresh: 95.70 %, 79.97 %, Δ −0.78 ± 0.15.
- The same h4 formulas given the chain values Ŝ (the inputs they were fitted on) instead of the true S, substituted alone at h4:
  94.56 %, 80.83 % (Δ +0.41 ± 0.22); fresh 94.82 %, 80.73 % (Δ −0.01 ± 0.16); equal to cumulative (b) at h4.
  With true S the cost-weighted h4 R² is higher (0.981 vs 0.961) and accuracy 1.2 points lower.
- Per class, true-S substitution vs network (test): g −0.3, q +0.7, W −1.0, Z −0.8, t −2.8 points; flows t→g 68 vs g→t 18. Fresh: t −2.9,
  W −1.5; t→g 146 vs g→t 37.
- Cost-weighted |mean bias| of the h4 outputs (spread units): 0.018–0.032 per class with true S, 0.008–0.015 with Ŝ.
- One unit at a time (true-S formula for that unit only): the sum of the 54 single-unit accuracy changes is −0.31 points against the −0.84 total.
  Largest single: h4:44 −0.24 (R² 0.79 on true S vs 0.954 on Ŝ; bias 0.117 vs 0.010 sd). Leave-one-out recovery: h4:49 +0.30, h4:44 +0.27, h4:47 +0.23.
- S inputs: replacing one Ŝ unit by its true value (others Ŝ) lowers accuracy by 0.44 (S:50), 0.28 (S:19), 0.24 (S:13) points; the reverse
  swaps raise it by 0.29, 0.28, 0.32.
- Half-plane run: −0.46 ± 0.18 (test), −0.53 ± 0.13 (fresh), chain-input version −0.07 to +0.42.

## 5. Edits (10,000 fresh jets, 2,000 per class; physics inputs recomputed from the edited particles)
Translation +0.06 rad in η, padding fixed (mass change 0.0; √(M0·M2) +41 %, ΣpT·r²[all] +126 %); pT × 0.8 (mass, √(M0·M2), ΣpT·r² −20 %).
| flow (fraction of true class) | network | (b) chain | (b) half-plane | (a) z on physics |
|---|---|---|---|---|
| unedited W→Z | 2.9 % | 3.4 % | 3.7 % | 2.1 % |
| translation: W→Z | 74.5 % | 77.6 % | 81.5 % | 0.1 % (W→W 62.9 %, W→t 17.4 %) |
| translation: Z→q | 67.0 % | 54.1 % | 57.6 % | 29.0 % (Z→t 51.0 %) |
| translation: q→W | 17.9 % | 43.5 % | 35.8 % | 8.4 % |
| translation: same call with network | — | 73.9 % | 81.9 % | 51.8 % |
| pT × 0.8: Z→t | 62.5 % | 71.2 % | 63.8 % | 61.3 % |
| pT × 0.8: W→q / W→t | 54.4 / 34.4 % | 47.7 / 41.9 % | 50.2 / 38.6 % | 60.0 / 27.1 % |
| pT × 0.8: same call with network | — | 88.1 % | 90.3 % | 83.9 % |
| unedited: same call with network | — | 94.2 % | 94.9 % | 92.6 % |
- Mean unit shifts under the translation, chain vs network (slope over units; median per-jet correlation): S 1.03 (0.89), h4 0.99 (0.86),
  h5 0.98 (0.83); P 0.56 (0.13) on library inputs, 1.02 (0.94) with half-plane inputs. z is compared before its ReLU, where the formulas
  were fitted only down to −0.2·spread (slope 0.44); this number does not measure the call-relevant response.
- The (b) program contains mass-family inputs (about 3 % of the S-stage weight) but carries the translation response through the
  radial sums (ΣpT·r², Σr, √(M0·M2)) at P and S. It moves W to Z under a mass-preserving edit as the network does. The (a) formula does not:
  0.1 % W→Z.

## Verdict
- Reproduced: yes. 94.02 % / 80.72 % (hp 94.64 / 80.75) exactly. No leakage: the end-to-end program takes only physics inputs computed
  from the particles, and the network's last layer is applied with its own rounding. The result is unchanged with non-live units at their
  fit means and with the particles zeroed.
- Refuted: partly.
  (i) "80.72 % vs the network's 80.42 %": +0.29 ± 0.23 on the test jets, −0.31 ± 0.17 on the fresh sample. Accuracy is equal to within about 0.5 points, not higher.
  (ii) W-vs-rest rejection at 50 % efficiency is 0.70× the network's [0.50, 0.95] (hp 0.75×). Other classes are within the bootstrap intervals.
  (iii) The physics-quantity count is 268, not 211.
- Confidence 0.85 for "the nested program reproduces the network's calls on 94 % of held-out jets and follows the network under the
  translation and pT edits"; 0.3 for any accuracy advantage.
- Corrected reading: the stage-by-stage hinge program (3,824 terms, 6,064 if-statements, 268 physics inputs) agrees with the network on
  94.0–94.2 % of held-out jets (94.6–94.8 % with half-plane inputs). Its accuracy matches the network's within ±0.3 points; its W
  rejection at 50 % efficiency is 30 % lower. Under the translation and pT edits it reproduces the network's main flows (W→Z 78 % vs 75 %;
  Z→t 71 % vs 63 %), with 74 % and 88 % call agreement on the edited jets. The direct z-on-physics formula does not reproduce the
  translation response (W→Z 0.1 %).
  Disagreements are low-margin jets, spread over all classes. They first appear at P (angular units, library inputs) or at S (half-plane run).
  The single-(b) h4 drop is an input mismatch: the formulas were fitted on the chain's Ŝ and lose 1.2 points of accuracy (t → g) when given the true S.

Note: interp/e11_verify.py was committed to the repository by another process during this run (commits ac3ea35 onward); this agent made no commits.
