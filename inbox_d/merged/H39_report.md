# H39: axis direction against architecture-matched nulls (claim D4m; loop D, round D8)

## Conclusion

Moving each tagger's particle coordinates to the winner-take-all or hardest-particle axis is a shift of the whole jet image. The part of the call change that comes from the shift pointing at the hardest particle (the "direction part") was measured for trained ParT, ParticleNet, PFN and EFN taggers on top and quark–gluon jets. It was compared with six untrained networks of each architecture on the same jets. In all 42 testable cells the trained networks lie inside the spread of their untrained copies (largest deviation 2.3 standard deviations of that spread, Holm-corrected p = 1). The sign of the direction part of a trained network also depends on where the decision threshold is placed (Top EFN: +3.5 points at the sign of the log-odds, −1.0 points at 50 % signal efficiency). Untrained EFNs show the same kind of direction effect (quark–gluon mean +2.4 points), so an energy-weighted architecture produces it before any training. Still uncertain: with six untrained networks per architecture, the test can only detect direction effects several times larger than the untrained spread. The second-seed and shuffled-label trainings, which give the seed-to-seed sign test, are still running (jobs listed below).

## Procedure

- **Script:** `interp/h39_axis_null.py`, GPU Job `d-h39-run` (smoke 300 jets, then the full run; 44 min, exit 0). Training arm: `interp/h39_train_ef.py`, `interp/h39_ef_jobs.sh`, `interp/h39_weaver_train.sh`, `interp/h39_seeded_net.py`.
- **Samples:** random rows per class, never first-k.
  - Top Landscape test: `data_top.sample(per_class=5000, seed=3901)` (read through `h4_top_wmass._sample`, same rows).
  - Quark–gluon (QG) Pythia test: `data_qg.sample(per_class=5000, seed=3911)`.
  - 10,000 jets each.
  - Median axis offset $|\delta|$ for WTA: Top 94 mrad (top 157, QCD 41); QG 17 mrad (quark 11, gluon 25). WTA = hardest particle in 60 % of Top jets.
- **Edits (inputs as in H36):**
  - E: unedited, scored twice (E2 = GPU repeat floor).
  - WTA, HP: all $(\Delta\eta_i,\Delta\phi_i)$ and $\Delta R_i$ recomputed about the axis.
  - TN: the same per-jet shift applied to the stored arrays through a separate code path (implementation check).
  - $R_k$, $k=1..20$: a translation of the same per-jet size $|\delta_j|$ in a uniformly random direction, drawn independently per $k$ and jet (rng 39021 / 39022).
- **Calls:**
  - sign(log-odds), as H36;
  - published working points $\varepsilon_S=0.3$ and $0.5$. Signal = top (Top), quark (QG). The threshold is the $(1-\varepsilon_S)$ quantile of the unedited signal-oriented score on this sample's signal jets, held fixed under edits.
- **Metric per network × axis × call:**
  - $d_j = \mathbf 1[\text{call changes about the axis}] - \frac{1}{20}\sum_k \mathbf 1[\text{call changes under } R_k]$;
  - direction part $D = 100\,\langle d_j\rangle$ (percentage points, pp), with a 95 % bootstrap interval over jets (B = 2000) and $p_{\rm jet}$ for $D\neq0$ (normal approximation).
- **Networks:**
  - **trained, seed 0:** the H36 models (Top: H4 snapshots of the H2 weaver ParT_kin / PN_kin; H2 energyflow PFN / EFN. QG: H2 Pythia ParT_kin, PN_kin, PFN, EFN).
  - **init (null), 6 per architecture:**
    - ParT and ParticleNet built with the authors' constructor arguments after `torch.manual_seed(1000+k)`;
    - EFN and PFN built by energyflow with seed $k$ (he_uniform kernels, zero biases) and saved untrained (`models/h39_init/`, job `d-h39-ef-init`).
    - The same initial networks are used on both datasets. The plan asked for 3; the cost was small, so 6 were used.
  - **shuf (null) and s1 (seed spread):** training, not yet included (see Training status). The script includes them automatically once finished.
  - **LorentzNet (Top, QG):** four-vector inputs, so the axis does not enter. D = 0 by construction; it was not re-scored under edits, and only its unedited accuracy is reported.
- **Null and tests per dataset × architecture × axis × call:**
  - The null is the set of D values of the init networks that are non-degenerate: unedited score SD > $10^{-6}$ and ≥ 0.5 % of calls changed under R.
  - $z = (D_{\rm trained} - \bar D_{\rm null})/s_{\rm null}$.
  - $p_{\rm pred}$: two-sided $t$ prediction-interval p, $t = (D-\bar D)/(s\sqrt{1+1/n})$, df $n-1$.
  - $p_{\rm emp} = (1+\#\{|D_{\rm null}-\bar D|\ge|D-\bar D|\})/(n+1)$. Its minimum is 1/7 = 0.143 for $n=6$.
  - Holm correction over the 42 cells with $n\ge2$: 8 network-datasets × 2 axes × 3 calls = 48, minus 6 cells whose null had < 2 non-degenerate members.
- **H0:** $D_{\rm trained}$ is drawn from the distribution of D over same-architecture networks not trained on the labels.
- **H1 (D4m, formula-driven):** $D_{\rm trained}$ lies outside that distribution, with the same sign in seeds s0 and s1.
- **Competing reading (generic translation sensitivity):** $D_{\rm trained}$ lies inside it, with the sign changing between seeds.

## Implementation checks

- **TN path:** axis recompute and TN shift give identical logits (max |Δl| = 0 on both samples, all trained networks).
- **EF port:** the torch EF port reproduces the numpy energyflow forward pass used in H36 (Top PFN / EFN: max |Δl| = 3.8e-6 / 3.1e-6).
- **Repeat floor:** max |Δl| between E and E2 is 2.6e-6, and 0.00 % of calls change under any call definition.
- **Smoke run:** 300 jets, same code; the full run reproduced its checks.

## Unedited accuracy on these samples (sign of log-odds; AUC)

| sample | network | acc | AUC | background rejection at $\varepsilon_S$ 0.3 / 0.5 | % calls changed WTA / HP (sign) |
|---|---|---|---|---|---|
| Top | ParT_kin (s0) | 94.18 % | 0.9867 | 2500 / 357 | 3.44 / 4.36 |
| Top | ParticleNet_kin (s0) | 94.05 % | 0.9868 | 2500 / 556 | 9.04 / 9.64 |
| Top | PFN (s0) | 93.13 % | 0.9812 | 833 / 357 | 8.68 / 12.48 |
| Top | EFN (s0) | 92.43 % | 0.9776 | 1000 / 156 | 14.80 / 19.22 |
| Top | LorentzNet | 94.30 % | 0.9879 | – | 0 / 0 (by construction) |
| QG Py | ParT_kin (s0) | 82.45 % | 0.8989 | 79 / 30.5 | 2.57 / 2.91 |
| QG Py | ParticleNet_kin (s0) | 82.36 % | 0.8985 | 76 / 29.4 | 4.07 / 4.33 |
| QG Py | PFN (s0) | 81.70 % | 0.8886 | 72 / 25.6 | 11.36 / 13.06 |
| QG Py | EFN (s0) | 80.74 % | 0.8808 | 68 / 27.8 | 23.35 / 25.28 |
| QG Py | LorentzNet | 84.15 % | 0.9151 | – | 0 / 0 (by construction) |

- **Agreement with earlier numbers:** the accuracies agree with the H2 full-test-set values within the binomial error on 10,000 jets (±0.3–0.4 pp): Top ParT 93.996 %, PN 94.019 %, PFN 92.83 %, EFN 92.18 %; QG ParT 82.70 %, PN 82.73 %, PFN 81.68 %, EFN 80.92 %.
- **Reproduction of H36 / H36v:**
  - % calls changed (WTA, sign) reproduces: Top ParT 3.44 vs 3.34–3.86; EFN 14.80 vs 14.72–15.48; QG PFN 11.36 vs 11.35–12.05; QG EFN 23.35 vs 21.6–23.5.
  - D at the sign call reproduces H36v: Top ParT WTA −1.52 vs −1.71; Top EFN +1.96 / +3.51 vs +2.56 / +3.98; QG PFN +1.52 / +2.05 vs +2.37 / +2.43.
- **Init networks (unedited):** accuracy 34.6–57.5 %, AUC 0.10–0.90.
  - The AUC is far from 0.5 for some untrained EFN / PFN, because their output is a fixed function of the energy flow that correlates with the class (Top EFN init AUC 0.81–0.90 or 0.10–0.12, i.e. the same ranking with either orientation).
  - At the sign call most init networks give the same call for every jet (degenerate: 0 calls move). Every init network is non-degenerate at the working points.

## Results: direction part D (pp) against the same-architecture init null

### Calls at $\varepsilon_S = 0.5$

| sample | network | axis | D trained [95 % CI] | % axis / % R | null mean ± SD [min, max] (n = 6) | z | $p_{\rm pred}$ | $p_{\rm emp}$ | Holm |
|---|---|---|---|---|---|---|---|---|---|
| Top | ParT | WTA | −2.12 [−2.57, −1.69] | 10.80 / 12.93 | −0.06 ± 1.34 [−1.82, +1.86] | −1.54 | 0.214 | 0.143 | 1 |
| Top | ParT | HP | −1.38 [−1.81, −0.97] | 12.30 / 13.68 | −0.02 ± 1.41 [−1.68, +1.69] | −0.96 | 0.413 | 0.571 | 1 |
| Top | ParticleNet | WTA | −0.58 [−0.88, −0.29] | 18.29 / 18.87 | +0.34 ± 1.13 [−0.86, +1.96] | −0.82 | 0.482 | 0.714 | 1 |
| Top | ParticleNet | HP | −0.40 [−0.68, −0.10] | 18.78 / 19.18 | +0.95 ± 1.49 [−0.70, +3.20] | −0.90 | 0.441 | 0.571 | 1 |
| Top | PFN | WTA | −1.12 [−1.58, −0.65] | 13.86 / 14.98 | −1.50 ± 1.46 [−3.27, −0.07] | +0.26 | 0.823 | 1.000 | 1 |
| Top | PFN | HP | −1.13 [−1.64, −0.65] | 14.90 / 16.03 | −1.45 ± 1.77 [−3.83, +0.28] | +0.18 | 0.875 | 1.000 | 1 |
| Top | EFN | WTA | −1.13 [−1.73, −0.56] | 19.10 / 20.23 | −1.21 ± 1.50 [−3.02, +0.49] | +0.05 | 0.963 | 1.000 | 1 |
| Top | EFN | HP | −1.04 [−1.62, −0.49] | 20.20 / 21.24 | −1.87 ± 1.37 [−3.64, −0.61] | +0.61 | 0.599 | 0.857 | 1 |
| QG Py | ParT | WTA | +0.17 [−0.02, +0.36] | 1.64 / 1.47 | −0.05 ± 0.56 [−1.17, +0.37] | +0.39 | 0.734 | 0.571 | 1 |
| QG Py | ParT | HP | +0.16 [−0.03, +0.36] | 1.68 / 1.52 | −0.04 ± 0.60 [−1.23, +0.41] | +0.32 | 0.778 | 0.714 | 1 |
| QG Py | ParticleNet | WTA | +0.35 [+0.10, +0.61] | 2.42 / 2.07 | +0.27 ± 0.40 [−0.21, +0.92] | +0.19 | 0.870 | 1.000 | 1 |
| QG Py | ParticleNet | HP | +0.30 [+0.03, +0.55] | 2.50 / 2.20 | +0.40 ± 0.41 [+0.05, +1.17] | −0.24 | 0.831 | 1.000 | 1 |
| QG Py | PFN | WTA | +0.68 [+0.30, +1.05] | 5.05 / 4.37 | −0.29 ± 0.47 [−1.03, +0.23] | +2.07 | 0.114 | 0.143 | 1 |
| QG Py | PFN | HP | +1.16 [+0.79, +1.53] | 5.83 / 4.67 | −0.26 ± 0.62 [−1.42, +0.24] | +2.26 | 0.090 | 0.143 | 1 |
| QG Py | EFN | WTA | +3.88 [+3.34, +4.44] | 17.19 / 13.31 | +2.47 ± 2.46 [−0.31, +6.37] | +0.57 | 0.619 | 0.714 | 1 |
| QG Py | EFN | HP | +3.94 [+3.37, +4.52] | 18.95 / 15.01 | +2.43 ± 2.50 [−0.27, +6.71] | +0.60 | 0.600 | 0.571 | 1 |

### Calls at $\varepsilon_S = 0.3$ (selected; full table in the JSON)

- **Top (all four trained networks):** D from −1.88 to +0.04 pp. Null means are −2.39 to +1.21, and every $|z|\le0.71$.
- **QG ParticleNet:** WTA +0.42 [+0.22, +0.63] against a null of +0.07 ± 0.17 [−0.10, +0.30] (z +2.09, $p_{\rm pred}$ 0.111).
- **QG EFN:** +2.36 / +2.61 against a null of +1.67 ± 1.98 / +1.52 ± 2.00.

### Calls = sign(log-odds), as in H36

- **Top ParT:** WTA −1.52 [−1.82, −1.22] against a null of −0.09 ± 0.63 (n = 4, z −2.27, $p_{\rm pred}$ 0.136).
- **Top PFN:** WTA −1.47 against a null of +0.38 ± 0.15 (n = 2, z −12.7, $p_{\rm pred}$ 0.061).
- **Top EFN:** HP +3.51 [+2.99, +4.02] against a null of +2.19 ± 0.88 (n = 3, $p_{\rm pred}$ 0.33). WTA +1.96 equals the null mean (+1.96).
- **QG EFN:** +1.53 / +1.89 against a null of +0.79 / +1.09 (n = 2).
- **ParticleNet (both samples) and QG PFN:** no non-degenerate init null at this call (all or all but one init networks give one call for every jet).

### Summary across the 42 testable cells

- Smallest $p_{\rm pred}$ = 0.061 (Top PFN, sign, WTA, null n = 2). Every Holm-corrected p = 1.
- **Outside the null range:** 10 of 42 cells, but none of them significant. For a draw from the null distribution, the chance of landing outside the range of $n$ null values is $2/(n+1)$ (0.29 for $n=6$), so about 11 such cells are expected by chance.
- **Detectable size:** at Holm level over 42 cells ($p<0.0012$, $t_5$), a cell would need $|z|\gtrsim7$, i.e. $|D-\bar D_{\rm null}|$ of about 3–18 pp given null SDs of 0.4–2.5 pp at $\varepsilon_S=0.5$. Without correction it would need $|z|\gtrsim2.8$.

## Reading (scoped to what was measured)

- **H0 vs H1 for seed 0:** H0 is not rejected in any cell. The one-seed part of the D4m (formula-driven) prediction, "D outside the architecture null", is not observed at any working point or for the sign call.
- **Closest cells to the edge:** QG PFN at $\varepsilon_S=0.5$ (+0.68 / +1.16 pp; z +2.1 / +2.3; above all six init values; $p_{\rm emp}$ = 0.143, the smallest attainable) and QG ParticleNet at $\varepsilon_S=0.3$ (z +2.1). Their sign matches the sign-call D of the same network (QG PFN +1.5 / +2.1).
- **Dependence on the call definition:** the sign of D for a given trained network depends on where the decision threshold is placed:
  - Top EFN +1.96 / +3.51 pp (sign) vs −1.13 / −1.04 (0.5) and −1.62 / −1.88 (0.3);
  - Top PFN HP +0.56 (sign) vs −1.13 (0.5).
  - At the working points all four Top networks have D < 0 (−0.40 to −2.12 pp at 0.5). The init null means of PFN and EFN are also negative there (−1.2 to −1.9).
- **Architecture effect without training:** the direction part exists in untrained networks.
  - Untrained QG EFNs have mean D +2.4 pp at 0.5 (5 of 6 positive on WTA, up to +6.4 / +6.7).
  - Untrained Top EFNs at the sign call have +1.4 to +2.8 pp.
  - So an energy-weighted sum $\sum_i z_i\Phi(\Delta\eta_i,\Delta\phi_i)$ with random $\Phi$ already responds differently to a shift that puts the hardest particle at the origin than to a shift in a random direction. The trained QG EFN (+3.9 pp) and Top EFN (sign, +2.0 / +3.5) lie within this spread.
- **Predictions of the competing readings:**
  - Generic translation sensitivity predicts D inside the null. This is observed for seed 0.
  - D4m additionally predicts the same sign across seeds and obs/pred ≈ 1 from a mass-formula calibration.
    - The seed part is not measured yet: s1 trainings are running.
    - No mass-formula calibration exists for the Top / QG models. The H36 translation-calibration obs/pred (0.44–1.29) was not re-measured here.
- **LorentzNet:** zero by construction, not a measured invariance control (as H36v noted).

## Training status (launched this round; status running)

| job | what | state at report time |
|---|---|---|
| `d-h39-top-ef` (12 CPU, 36 Gi) | Top EFN and PFN, s1 and shuf (four processes) | running; epoch 5 of ≤ 40 (s1) / 21 and 16 (shuf); ~12 min per epoch |
| `d-h39-qg-ef` (12 CPU, 36 Gi) | QG EFN and PFN, s1 and shuf | running; epoch 4; ~14 min per epoch; peak memory 29.7 GiB |
| `d-h39-top-pn-s1` (GPU, 4 CPU, 32 Gi) | Top ParticleNet_kin, torch seed 1 | running since 21:49 UTC (Quadro RTX 6000; GPU 99 %); epoch 0: train loss 0.1924, val acc 0.9314 (s0 epoch 0: 0.1921, 0.9312); ~18 min per epoch, 20 epochs |
| `d-h39-top-pn-shuf` | Top ParticleNet_kin, labels = hash(jet pT, η, φ) mod 2, 20 epochs | running since 21:53 UTC; label check on the test file: mean label 0.501, agreement with the true class 0.500 (class 1: 0.501, class 0: 0.501); first-batch loss 0.706 |
| `d-h39-qg-pn-s1` | QG ParticleNet_kin, seed 1 | pending |
| `d-h39-qg-pn-shuf` | QG ParticleNet_kin, hash labels | pending |
| `d-h39-qg-part-s1` | QG ParT_kin, seed 1 (optional arm) | pending |
| `d-h39-ef-init` | 6 untrained EFN + 6 PFN | done |

- **Shuffled EF smoke check:** train/val loss 0.6932 and accuracy 0.500 through epoch 4–5, as expected for permuted labels. Whether these networks end non-degenerate (logit SD > 0) is written to their `info.json` (`test_logit_std`). The s1 networks reach val accuracy 0.917 (Top EFN, epoch 5) and 0.789 (QG PFN, epoch 4).
- **ParticleNet hash labels:** these are not a permutation. Each jet's label is a deterministic pseudo-random bit computed by weaver from (jet $p_T$, η, φ). The job prints its balance and its agreement with the true label on the test file before training.
- **First ParticleNet attempt:** it was killed for running out of memory at 16 Gi while loading data. All GPU jobs were relaunched at 32 Gi.
- **Next round:** rerunning `./part_job.sh d-h39-run2 'python -m interp.h39_axis_null'` once the jobs finish adds s1 (seed-sign test) and shuf (larger null) automatically. Change TAG or copy the outputs to keep this round's files.

## Issues

- **Size of the null:** n = 6 init networks per architecture. The empirical p cannot go below 0.143, so Holm over 42 cells can never be passed with it. The t-based $p_{\rm pred}$ assumes the null D values are normally distributed.
- **Sign-call nulls:** these are thin. ParticleNet and QG PFN have no non-degenerate init network at the sign call: an untrained network gives the same sign for nearly every jet.
- **ParticleNet init outputs:** the score SD is only ~0.003 (BatchNorm at its default running statistics). At fixed working points such networks still change 1–24 % of calls under a shift, so they count as non-degenerate by the rule above, but the metric is the same as for trained networks.
- **Seeds:** the init networks are shared between datasets (same seeds); within a dataset the null members are independent.
- **Sample:** QG Herwig was not part of this plan (Pythia only).

## Explainer entry (for site/explain.py, Paper D)

```
H39_EXPLAINER = [
 ('Question', 'H36v found that the part of the axis-convention response that depends on the shift pointing at the hardest particle lies inside a band of weight-permuted networks, but permuted EFN / PFN / ParticleNet were constant and there was one training seed. Is the direction part of trained taggers outside the range of untrained networks of the same architecture (claim D4m)?', None),
 ('Physics background', 'For these taggers the axis enters only through $(\\Delta\\eta_i,\\Delta\\phi_i)$, so recomputing about the WTA or hardest-particle axis is a rigid translation by the axis offset $\\delta_j$. The direction part is $D=\\langle \\mathbf 1[\\text{call changes about the axis}]-\\tfrac1{20}\\sum_k\\mathbf 1[\\text{call changes under a shift of size }|\\delta_j|\\text{ in direction }k]\\rangle$. A learned use of the axis (D4m) predicts $D$ outside what untrained functions of the same form give.', None),
 ('Method', 'Top Landscape (seed 3901) and quark-gluon Pythia (seed 3911) test jets, 5,000 per class. Trained ParT, ParticleNet, PFN, EFN against 6 epoch-0 networks per architecture; 20 random directions per jet; calls at the sign of the log-odds and at signal efficiency 0.3 and 0.5. $z=(D-\\bar D_{\\rm null})/s_{\\rm null}$, $t$ prediction-interval and empirical p, Holm over 42 cells. Second-seed and shuffled-label trainings launched.', None),
 ('Result', 'All 42 cells inside the untrained spread ($|z|\\le2.3$, Holm $p=1$). At $\\varepsilon_S=0.5$: Top $D$ from $-2.1$ to $-0.4$ pp (untrained means $-1.9$ to $+1.0$); QG EFN $+3.9$ pp vs untrained $+2.4\\pm2.5$; QG PFN $+0.7$ / $+1.2$ pp vs $-0.3\\pm0.5$ (closest to the edge). The sign of $D$ changes with the threshold (Top EFN $+3.5$ at the sign call, $-1.0$ at $\\varepsilon_S=0.5$).', 'results/physics/h39_axis_null.png'),
 ('Checks', 'Axis recompute equals the shift of the stored arrays exactly; the torch EF port matches the numpy forward pass to $4\\times10^{-6}$; repeat scoring changes no call. Unedited accuracies agree with the H2 test-set values within 0.4 pp; H36 / H36v call changes and sign-call $D$ reproduce.', None),
 ('Physical significance', 'With one training seed, the extra response to shifts toward the hardest particle is what an untrained network of the same architecture also gives; energy-weighted architectures (EFN) respond to that direction before training. No learned, formula-driven use of the axis direction is measured on Top or quark-gluon jets; the seed-to-seed sign test is pending.', None),
]
```
Conclusion for site/conclusions.py (`H39_EXPLAINER`): "Compared with six untrained copies of the same architecture, the trained taggers' extra response to shifting the jet image toward its hardest particle is never outside the untrained range, and its sign changes with where the decision threshold is placed. Untrained energy-weighted networks (EFN) already show such a direction effect of a few points, so it does not require training. A second training seed and shuffled-label trainings, which would test whether the sign is stable across trainings, are still running."

## Files

- `interp/h39_axis_null.py`, `interp/h39_train_ef.py`, `interp/h39_ef_jobs.sh`, `interp/h39_weaver_train.sh`, `interp/h39_seeded_net.py`
- `results/physics/h39_axis_null.json`, `h39_perjet.npz`, `h39_axis_null.png` (smoke: `h39_smoke_*`)
- logs: `/j-jepa-vol/aaron-interp/logs/d-h39-run.log` (copy in `logs/`), `d-h39-*-ef.log`, `h39-ef-<ds>-<arch>-<var>.log`, `d-h39-<ds>-pn-<var>.log`, `d-h39-qg-part-s1.log`
- models (when finished): `/j-jepa-vol/aaron-interp/models/{top,qg}/{efn,pfn}_{s1,shuf}/`, `{top,qg}/pnet_kin_{s1,shuf}/`, `qg/part_kin_s1/`, `models/h39_init/`
