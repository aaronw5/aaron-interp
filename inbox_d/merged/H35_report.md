# H35: where the b sits inside a top jet (JetClass ParT_full, ParticleNet_full)

## Conclusion

The test kept the number of displaced tracks in each top jet fixed and moved the displacement from the b-quark subjet (the "b-prong") onto tracks of a W-decay subjet. ParT's top score then fell by about 3 units of log-odds more than it did when the same values were swapped among tracks inside the b-prong. ParticleNet fell by about 2 units more. Both full-input taggers use which subjet the displaced tracks sit in, not only how many there are in the jet: a b-hadron decay is expected on the subjet that, together with the W pair, makes the top mass. Two things are still uncertain. Even the control swap inside the b-prong lowered the score by more than 2 units, so the networks also respond to which tracks within a prong carry the displacement. The b-prong label here comes from the subjet masses and agrees with the displaced-track count in only 51 % of top jets.

(Log-odds = log p(top)/(1 − p(top)) from the 10-class softmax. A displaced track is a charged track with transverse impact-parameter significance S = |d0|/σ(d0) > 2.)

## Procedure

- **Sample:** JetClass test files TTBar_100 (Tbqq) and ZJetsToNuNu_100 (QCD). The jets have stored 550 < pT < 650 GeV, |η| < 2 and ≥ 3 particles. There are 2,000 jets per class, drawn as random rows (rng 3501 / 3502). The H19 window was used so that the accuracies can be compared. Every variant, including the unedited one, is rebuilt with `part_core.build` (η = jet_η + sign(jet_η)·Δη). The four-momenta are never changed; only (d0val, d0err, dzval, dzerr) are exchanged or zeroed.
- **Prongs:** three exclusive-kT subjets (E-scheme). The W pair (a, b) is the pair with mass closest to 80.4 GeV, and c (the b-prong) is the third subjet. Cross-check: the assignment "agrees" if n_disp(c) > max(n_disp(a), n_disp(b)). This holds for 50.8 % of tops, or 59.8 % if ties count. 83.3 % of tops have ≥ 1 displaced track in c. Mean displaced tracks per top jet: 3.10 in c, 3.27 in a + b, 6.37 in the jet.
- **Edits:**
  - **M:** the target is the W prong w with more prompt tracks. The k = min(n_disp(c), n_prompt(w)) highest-pT displaced tracks of c swap their (d0, σd0, dz, σdz) with the k highest-pT prompt tracks of w, rank by rank. This leaves unchanged the jet's displaced-track count, the multiset of S values and all kinematics; both are asserted in the code. After M, n_disp(c) goes from 3.10 to 0.19 on average.
  - **C1:** the same exchange inside c. Displaced tracks of c are swapped with k1 = min(k, n_prompt(c)) prompt tracks of c.
  - **C2:** inside w. M's k target tracks are swapped with the next k2 charged tracks of w.
  - **R:** every displaced track of c is made prompt (d0, dz × 0, errors kept; G194 convention).
  - **Rk / RWk:** an equal number kk = min(n_disp(c), n_disp(a + b)) of displaced tracks is made prompt, either in c (Rk) or in the W prongs (RWk).
  - **ALL:** every displaced track in the jet is made prompt (the G194 anchor).
  - **QCD:** the same edits between the two hardest subjets. The source is the subjet with more displaced tracks.
- **Models and controls:**
  - ParT_full and PN_full (inputs include d0 and dz).
  - ParT_kin and PN_kin (no track inputs; zero response by construction).
  - ParT_perm: ParT_full with the entries of each weight tensor permuted (torch seed 5).
- **References:** trained on JetClass training rows (train_100M TTBar_000 / ZJetsToNuNu_000, same window, 20,000 per class, rng 3503 / 3504). Each was retrained 5 times on 80 % subsamples.
  - Jref: (m, τ32, m_Wpair, n_disp,jet).
  - Sref: Jref plus (n_disp(c), max / min n_disp(a, b)), i.e. a classical subjet b-tag count.
  - Each comes as LR and as MLP(64, 64). The reported response is the mean over the 5 retrains.
- **Statistics:**
  - Per-jet change in top log-odds, with a 95 % bootstrap CI over jets (B = 2,000).
  - Two-sided sign-flip permutation test (5,000 permutations) of H0: mean = 0.
  - Holm correction over 35 tests: models × {M − C1, M − C2} × {top_agree, top_all, qcd}, plus Rk − RWk on top_kk.
  - McNemar test on the top calls (argmax = Tbqq) for M vs C1.
- **Analysis sets:**
  - top_agree: tops where the assignment agrees and k ≥ 1; N = 1,016. This is the primary set.
  - top_all: tops with k ≥ 1; N = 1,665.
  - qcd: QCD jets with k ≥ 1; N = 1,282.
  - top_kk: tops with kk ≥ 1; N = 1,421.
- **Run:** `./part_job.sh d-h35-run 'python -m interp.h35_top_bprong'`, 190 s on an RTX 2080 Ti. The smoke run was d-h35-smoke. The figure was redrawn by d-h35-fig.

## Unedited models on this sample (4,000 jets)

| model | 10-class acc | binary acc (L_T > L_QCD) | AUC | H19 window, 10-class / binary |
|---|---|---|---|---|
| ParT_full | 86.8 % | 98.35 % | 0.9986 | 86.5 / 98.4 % |
| PN_full | 85.7 % | 98.1 % | 0.9978 | 85.5 / 98.1 % |
| ParT_kin | 81.1 % | 95.6 % | 0.9916 | 81.2 / 95.8 % |
| PN_kin | 79.9 % | 94.7 % | 0.9888 | 79.8 / 95.1 % |
| ParT_perm | 14.7 % | 49.8 % | 0.452 | none |

- **References (binary acc on the sample):** Jref LR 91.8 %, Jref MLP 93.7 %, Sref LR 92.1 %, Sref MLP 93.6 %.
- **Input-path check:** the rebuilt identity jet vs the raw read_root jet differs by at most 1.05 in ParT_full top log-odds on 1 of 4,000 jets (float32 rebuild). Every comparison below is rebuilt vs rebuilt.
- **Kin sanity:** for ParT_kin and PN_kin the maximum |Δ| over all edits is 0.000.

## Numbers (top_agree, N = 1,016; change in top log-odds, mean [95 % CI])

| edit | ParT_full | PN_full | ParT_perm (null) | Sref MLP | Jref MLP |
|---|---|---|---|---|---|
| M | −5.59 [−5.79, −5.40] | −4.24 [−4.41, −4.07] | +0.00 [−0.04, +0.04] | −0.99 | 0 |
| C1 | −2.65 [−2.79, −2.51] | −2.15 [−2.27, −2.03] | +0.02 [−0.02, +0.05] | 0 | 0 |
| C2 | −0.68 [−0.77, −0.60] | −0.25 [−0.31, −0.20] | +0.00 | 0 | 0 |
| **M − C1** | **−2.94 [−3.13, −2.75]**, Holm p = 0.007 | **−2.09 [−2.25, −1.94]**, Holm p = 0.007 | −0.015 [−0.055, +0.024], p = 0.46 | −0.99 [−1.03, −0.95] | 0.00 |
| M − C2 | −4.91 [−5.11, −4.70], Holm 0.007 | −3.99 [−4.17, −3.82], Holm 0.007 | −0.00 | −0.99 | 0 |
| R (b-prong prompt) | −5.25 [−5.43, −5.07] | −4.94 [−5.11, −4.77] | +0.01 | −3.23 | −2.21 |
| ALL (G194 anchor) | −5.65 [−5.82, −5.47] | −5.34 [−5.49, −5.17] | +0.01 | −4.15 | −3.50 |

- **Top-call rate** (ParT_full; id → M / C1 / C2 / R / ALL): 98.3 → 74.1 / 94.8 / 97.1 / 73.1 / 67.8 %. PN_full: 98.0 → 84.7 / 95.1 / 97.3 / 68.9 / 63.9 %.
- **McNemar test on top calls, M vs C1:** ParT_full 231 pass→fail vs 21 fail→pass, PN_full 126 vs 21, both p < 1e−6. ParT_perm 159 vs 163, p = 0.87.
- **Ratio of M − C1 to Sref:** ParT_full 2.97 [2.78, 3.19] × Sref-MLP (2.89 × Sref-LR). PN_full 2.12 [1.96, 2.27] × Sref-MLP. The per-jet correlation of M − C1 with Sref-MLP is 0.29 for ParT_full and 0.31 for PN_full.
- **Sref retrain spread:** the M − C1 values of the 5 MLP retrains range from −0.70 to −1.21. Jref is 0 by construction in every retrain.
- **Equal-size subset:** C1 swaps fewer tracks than M on average (3.29 vs 4.48; k1 = k in 58 % of jets). On the 591 jets with k1 = k, M − C1 is −2.20 ± 0.24 for ParT_full, −1.47 ± 0.17 for PN_full and −0.03 ± 0.05 for ParT_perm (normal-approximation 95 % intervals; no bootstrap).
- **top_all** (N = 1,665, assignment not required to agree): M − C1 is −2.18 [−2.34, −2.02] for ParT_full and −1.50 [−1.61, −1.38] for PN_full, both Holm p = 0.007. Sref-MLP gives −0.68.
- **Equal-count removal (top_kk, N = 1,421; kk = 1.53 tracks per jet on average):**

| | Rk per track | RWk per track | Rk − RWk (jet) | Holm p |
|---|---|---|---|---|
| ParT_full | −0.79 [−0.85, −0.73] | −0.31 [−0.36, −0.26] | −1.23 [−1.41, −1.07] | 0.007 |
| PN_full | −0.74 [−0.79, −0.68] | −0.25 [−0.30, −0.20] | −1.21 [−1.37, −1.04] | 0.007 |
| Sref MLP | −0.47 | −0.12 | −0.77 | none |
| Jref MLP | −0.34 | −0.34 | 0.00 | none |

- **QCD (N = 1,282; displacement moved between the two hardest subjets):**
  - M − C1: ParT_full +0.12 [−0.01, +0.24] (Holm 1.0); PN_full +0.14 [+0.02, +0.25] (Holm 0.67).
  - M and C1 each raise the top log-odds by +1.0 to +1.4. The top-call rate changes from 1.6 % to 5.1 % (M) and 5.3 % (C1) for ParT_full.
  - Sref-MLP M − C1 is +0.11.

## Hypotheses

- **H1 (b-prong localisation):** M lowers the top log-odds; M − C1 lies outside the permuted-null interval; the response tracks the subjet-b-tag reference, not the jet-level count.
  - Measured: M − C1 = −2.94 (ParT_full) and −2.09 (PN_full), against a null of [−0.055, +0.024].
  - Sref gives −0.99 and Jref 0.
  - The sign and the significance match H1. The magnitude is 2–3 × the Sref classifier.
- **H0 (count only):** M − C1 ≈ 0, as for Jref, and equal drops per track for R and R-W.
  - Excluded for both networks: M − C1 is ≠ 0 at Holm p = 0.007, and the per-track drop is 0.79 (b-prong) vs 0.31 (W prong) for ParT_full.

## Reading (scoped to what was measured)

- **Location response:** with the jet-level displaced-track count and the set of S values held fixed, ParT_full and PN_full lower the Tbqq log-odds when the displacement is moved from the mass-assigned b-prong to a W prong. The drop is larger than for the same exchange inside the b-prong (C1) or inside the W prong (C2). In 23 % of agreeing tops the top call is lost. The weight-permuted ParT and the jet-level-count classifier show no response.
- **Per-track drops:** making displaced tracks prompt lowers the score 2.5–3 × more per track when they sit on the b-prong than on a W prong.
- **QCD jets:** moving the displacement between the two hardest subjets gives the same response as moving it within one subjet, so in QCD jets the location effect is not resolved.
- **The C1 control is not neutral:** C1 lowers the top log-odds by 2.1–2.6 while keeping per-prong counts. The networks also respond to which tracks within the b-prong carry the displacement, possibly the pT or angular position of the displaced tracks relative to the prong axis. This is not separated here. Part of M's −5.6 is this within-prong effect; M − C1 removes it under the assumption that the two add.
- **Edit size mismatch:** C1 averages fewer swaps than M. On the subset with equal sizes, M − C1 is −2.2 (ParT) and −1.5 (PN), smaller but of the same sign.
- **Log-odds saturation:** the unedited median top log-odds of agreeing tops is 7.1 (ParT) and 5.9 (PN). Log-odds changes of this size near saturation correspond to the call changes listed above.

## Explainer entry (for site/explain.py; not written there, since site/ is outside this loop's write scope)

```python
H35_EXPLAINER = [
 ('Question', 'In $t\\to bW\\to bq\\bar q\'$ the $b$ hadron flies a few mm before decaying, so its charged tracks have large impact parameters. G194 showed that JetClass ParT loses top jets when all displaced tracks are made prompt. Does the tagger only count displaced tracks anywhere in the jet, or does it check that they sit on the prong that, with the W pair, makes $m_t$?', None),
 ('Physics background', 'A top jet has three prongs: two from $W\\to q\\bar q\'$ with $m_{q\\bar q\'}\\approx 80.4$ GeV and the $b$. Classical subjet $b$-tagging asks for displaced tracks (significance $S=|d_0|/\\sigma_{d_0}>2$) in the subjet not in the W pair. A jet-level count ignores where they are.', None),
 ('Method', 'JetClass test jets, $550<p_T<650$ GeV, 2,000 $t\\to bqq$ and 2,000 QCD (random rows, seed 3501). Three exclusive-$k_T$ subjets; W pair = pair with $m$ closest to 80.4 GeV; $b$-prong $c$ = the third. Edit M: the $k$ highest-$p_T$ displaced tracks of $c$ exchange $(d_0,\\sigma_{d_0},d_z,\\sigma_{d_z})$ with the $k$ highest-$p_T$ prompt tracks of a W prong, so $N_{\\rm disp}^{\\rm jet}$, the multiset of $S$ and all four-momenta are unchanged. Control C1: the same exchange inside $c$. Metric: $\\Delta\\,\\mathrm{logit}\\,p(t)$, $M-C1$ per jet, bootstrap CI, sign-flip permutation, Holm over 35 tests. Nulls: weight-permuted ParT; kinematic-only ParT/PN (zero by construction). References: MLP on $(m,\\tau_{32},m_W,N_{\\rm disp}^{\\rm jet})$ and the same plus per-subjet counts.', None),
 ('Result', 'On 1,016 tops where the mass-assigned $b$-prong also has the most displaced tracks: $M-C1=-2.94\\,[-3.13,-2.75]$ (ParT_full), $-2.09\\,[-2.25,-1.94]$ (ParticleNet_full); permuted null $-0.02\\,[-0.06,+0.02]$; per-subjet-count classifier $-0.99$, jet-count classifier $0$. ParT top calls 98.3 % → 74.1 % (M) vs 94.8 % (C1). Equal numbers of tracks made prompt: $-0.79$ per track on the $b$-prong vs $-0.31$ on a W prong.', 'h35_bprong.png'),
 ('Checks', 'Unedited accuracies reproduce H19 (ParT_full 86.8 % 10-class, 98.4 % binary). Kinematic-only models: $|\\Delta|=0$. Count and $S$ multiset asserted unchanged. Equal-size subset ($k_{C1}=k_M$, 591 jets): $-2.2$ (ParT), $-1.5$ (PN). QCD jets: location change between the two hardest subjets gives $M-C1\\approx+0.1$ (not significant after Holm). C1 itself lowers the score by $\\approx 2.6$: within-prong placement also matters.', None),
 ('Physical significance', 'The full-input JetClass taggers use where displaced tracks sit relative to the three-prong mass structure, as a subjet $b$-tag does, and respond 2–3 times more strongly than a classifier given the per-subjet displaced-track counts.', None),
]
```

## Files

- Script: /Users/anrunw/Documents/aaron-interp/interp/h35_top_bprong.py
- Results: /Users/anrunw/Documents/aaron-interp/results/physics/h35_top_bprong.json, h35_perjet.npz
- Figure: /Users/anrunw/Documents/aaron-interp/results/physics/h35_bprong.png
- Logs: /j-jepa-vol/aaron-interp/logs/d-h35-run.log (d-h35-smoke, d-h35-fig)
