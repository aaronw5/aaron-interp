# X5 verify: Deep Sets float / L1 / HGQ, elements (a) pure cuts and (d) shift response, fresh sample

## Conclusion (plain terms)
- **Found:** On a new sample (seed 4242), both main results repeat in all 12 models.
  - (d) The 0.06 rad translation turns 50–88 % of W jets into Z calls. The equal-displacement rotation gives 2–8 %, about the unedited rate.
  - (a) Pure first-layer cuts exceed the permutation null in HGQ seeds 0 and 2 of both arms, and in no float or L1 model.
- **Means:** The reading holds as measured, with three limits.
  1. The "HGQ" models differ from the float models in more than quantization: Adam without weight decay, batch norm folded, 10 epochs, a different lr and batch size, and keras initialisation. The β = 1e-8 arm has almost no bit penalty and shows the same purity. The (a) effect therefore belongs to the HGQ training pipeline. Quantization is not separated from the other differences.
  2. At the same seed, the two HGQ arms have nearly the same first layer: the per-unit pT-share correlation is 0.976–0.987, and 89–98 % of units have the same type. "2 of 3 seeds per arm" is therefore one result (seeds 0 and 2), not two independent ones.
  3. The fresh-sample calibrated ratio range is 0.71–1.29, against 0.72–1.26 reported.
- **Uncertain:**
  - In the β = 1e-8 arm, presence of (a) depends on the purity threshold: 2/3 seeds at 0.9, 3/3 at 0.8, 1/3 at 0.95.
  - In HGQ, the matched-scale calibration edit (angles × 1.35) causes a quantizer overflow in 19–28 % of jets, against 0.3–0.5 % unedited. The HGQ ratio denominator therefore contains wrap-mode effects.
  - Elements (b) and (c) were not rerun.

## Code check (interp/x5_dsets_sparsity.py, train/train_hgq_dsets.py, interp/x5_plot.py)
- **Forward and inputs.** The forward uses the fused (batch-norm-folded) quantized kernels and each layer's input quantizer. Padded rows (raw pT = 0) are zeroed after normalisation and masked after phi2. Step 0 reproduces the stored test probabilities, so the forward is correct.
- **Edits.** The (d) edits move real particles only, and padded rows stay 0. The translation leaves the invariant mass unchanged (max |Δlog m| = 2e-7). The rotation keeps r, so it keeps M0 and M2.
- **Splits.** HGQ training uses train minus splits.npz dev, and testing uses the val archive (the full_test.npz the sample is drawn from). There is no split leakage.
- **(a) ignores the phi1 input quantizer.** The bug check was run: the iq-aware spread sd(iq(x)) equals the raw sd to 4 digits (47.59 vs 47.59–47.61 GeV; 0.1112 and 0.1093 rad). The phi1 iq has zero zero-bit (slot, feature) entries and 0 overflowing entries. The iq-aware p equals the raw p (table below).
- **(a) has no jet-level spread.** The jet bootstrap of P is degenerate (P_ci = [P, P]), so the only spread is across seeds.
- **The regime is not isolated in training.** HGQ training differs from train_small.py in optimizer, batch norm, epochs, lr, batch size and initialisation (module docstring). There is no keras float model built by the same script with quantizers off. A model with HGQ's training settings but without quantizers would separate the two, but that needs a training run, which the rules exclude here.
- **Figure** (results/jedi/x5_dsets_sparsity.png): it matches the JSON values. The "$\sqrt{M_0M_2}$ prediction" label in panel (d) is clipped at the left axis.

## Procedure (verification)
- **Script:** `interp/x5_verify.py`. It imports X5's network classes, edits and ridge read-out unchanged.
- **Jobs:** `x5-verify` (float + HGQ β 3e-4) and `x5-verify-b` (L1 + HGQ β 1e-8). Each was 1 CPU / 2 GiB, exit 0, 800 s wall, peak 2047 MiB.
- **Sample:** `sample(per_class=2000, seed=4242)`, 10,000 test jets (X5 used seed 311), with 2,000 true-W jets.
- **(a):**
  - Live phi1 units are recomputed on the new sample. P is computed on the live units, and P_nz on the units with any nonzero weight.
  - Null: X6v column permutation with a new RNG and 2,000 draws (X5 used 200, which gave a p floor of 0.005). Thresholds 0.8 / 0.9 / 0.95.
  - For HGQ, an iq-aware version replaces sd(x) with sd(iq(x)).
  - H0: P_nz ≤ null. H1: P_nz > null. One-sided p.
- **(d):**
  - Edits: translate +0.06 rad in Δη; rotation with equal mean displacement; random-direction translation; matched angular scale k = exp(0.302) = 1.35.
  - Added control: **translate −0.06 rad in Δη**. The $\sqrt{M_0M_2}$ hypothesis predicts the same increase as for +0.06 rad, because M2 grows by about $\delta^2 M_0$ in either direction.
  - Added read-out control: the **rotation read-out ratio**, which is the rotation's read-out shift divided by the matched-scale shift.
  - Added check for HGQ: the **quantizer overflow fraction**, with $|{\rm iq}(v)-v|>2^{-f}$, at all 7 quantized layers per edit. W→Z is also reported on the W jets with no overflow in the unedited or translated pass.
  - Ridge read-out of log m on h3: fitted on half the sample, ratios measured on the other half, 1,000 bootstraps.
  - McNemar on the paired Z calls of the 2,000 W jets. H0: the translation and the control give equal W→Z. H1: they differ.

## Results

**(a) Pure share, threshold 0.9.** Columns: P on live units; null 95 % range; one-sided p from 2,000 draws; p at thresholds 0.8 and 0.95; iq-aware p. X5 values are in brackets.

| model | P | null 95 % | p | p @0.8 / @0.95 | iq-aware p |
|---|---|---|---|---|---|
| float s0 / s1 / s2 | .381 / .312 / .391 | [.36,.50] [.28,.41] [.31,.45] | .96 / .84 / .45 [.96/.88/.47] | ≥ .39 / ≥ .43 | – |
| L1 s0 / s1 / s2 | 1.0 / 1.0 / 1.0 (14, 12, 12 live) | [.89,1] [.87,1] [.88,1] | .72 / .58 / .56 [.74/.55/.56] | ≥ .60 / ≥ .51 | – |
| HGQ β 3e-4 s0 / s1 / s2 | .500 / .297 / .422 | [.28,.42] [.22,.31] [.25,.36] | **.0005** / .175 / **.0015** [.005/.21/.005] | .0015,.0075,.0015 / .0025,.0015,.0005 | .0005 / .175 / .0010 |
| HGQ β 1e-8 s0 / s1 / s2 | .484 / .312 / .422 | [.27,.41] [.22,.31] [.25,.34] | **.0010** / .062 / **.0005** [.005/.090/.010] | .0005,.0005,.0015 / .15,.25,.008 | .0005 / .059 / .0005 |

- Presence uses Holm over the 4 elements, which here means p × 4 < 0.05 on the smallest p:
  - HGQ β 3e-4: 2/3 at threshold 0.9; 3/3 at 0.8 and at 0.95.
  - HGQ β 1e-8: 2/3 at 0.9; 3/3 at 0.8; 1/3 at 0.95.
  - Float and L1: 0/3 at every threshold.
- **Same-seed similarity of the two HGQ arms:** the per-unit pT-share Pearson r is 0.987 / 0.979 / 0.976, and the same type holds in 0.89 / 0.98 / 0.94 of 64 units. The seed that misses the null (seed 1) is the same in both arms.

**(d) Shift response.** W→Z is the fraction of 2,000 true-W jets called Z. The calibrated ratio has a 95 % bootstrap interval.

| model | W→Z unedited → +Δη / −Δη / random direction / rotation | calibrated ratio [95 %] | rotation read-out ratio [95 %] | McNemar +Δη vs rotation |
|---|---|---|---|---|
| float s0 | .018 → .672 / .428 / .570 / .032 | .869 [.850,.887] | .004 [.000,.008] | p < 1e-240 |
| float s1 | .022 → .722 / .557 / .602 / .033 | .706 [.686,.725] | .000 [−.005,.004] | p < 1e-240 |
| float s2 | .022 → .532 / .697 / .598 / .034 | .911 [.892,.932] | .005 [.000,.010] | 2e-242 |
| L1 s0 | .031 → .829 / .710 / .777 / .038 | .981 [.960,1.003] | .000 [−.004,.004] | p < 1e-240 |
| L1 s1 | .023 → .802 / .710 / .776 / .032 | .857 [.833,.882] | .000 [−.003,.003] | p < 1e-240 |
| L1 s2 | .025 → .797 / .620 / .696 / .039 | .818 [.798,.837] | .007 [.003,.011] | p < 1e-240 |
| HGQ 3e-4 s0 | .059 → .876 / .878 / .874 / .071 | 1.037 [1.009,1.065] | .002 [−.005,.009] | p < 1e-240 |
| HGQ 3e-4 s1 | .076 → .661 / .887 / .819 / .075 | .973 [.947,1.000] | −.004 [−.011,.003] | 2e-268 |
| HGQ 3e-4 s2 | .061 → .815 / .881 / .826 / .065 | .977 [.952,1.003] | −.005 [−.011,.002] | p < 1e-240 |
| HGQ 1e-8 s0 | .045 → .712 / .779 / .697 / .049 | 1.286 [1.248,1.327] | .004 [−.002,.011] | p < 1e-240 |
| HGQ 1e-8 s1 | .018 → .499 / .783 / .701 / .022 | 1.006 [.979,1.036] | −.001 [−.010,.008] | 5e-245 |
| HGQ 1e-8 s2 | .036 → .741 / .735 / .805 / .043 | 1.071 [1.045,1.096] | −.001 [−.006,.005] | p < 1e-240 |

- **Prediction vs measurement.** The mean $\tfrac12\Delta\log(M_0M_2)$ on the test half is 0.302 (X5: 0.301).
  - Invariant-mass hypothesis: predicts a read-out shift of 0. Excluded in all 12 models (every ratio CI is above 0.68).
  - Out-of-distribution displacement without a mass dependence: predicts that the equal-displacement rotation responds as well. Rotation read-out ratios are −0.005 to 0.007, and rotation W→Z is within 0.015 of the unedited rate.
  - $\sqrt{M_0M_2}$ hypothesis: predicts a response to −Δη as well as to +Δη. The −Δη translation gives W→Z 0.43–0.89 in all 12 models. The +Δη / −Δη difference is significant in 10 of 12 (McNemar p 1e-10 to 1e-102); its sign varies by model (−Δη larger in 6 of 12). This is consistent with a √(M0 M2)-type response plus a model-specific direction term (the odd 12–20 % term found for JEDI-linear).
- **HGQ overflow.**
  - Fraction of jets with any quantizer overflow: unedited 0.33–0.47 %; +Δη 0.7–3.9 %; −Δη 0.5–12.7 %; rotation 1.3–1.8 %; matched scale **19.4–28.2 %**. The phi1 input quantizer never overflows. Overflows occur in phi2, pool and h1–h3, at ≤ 3e-4 of entries.
  - W→Z on W jets without overflow (n = 1,938–1,988) changes by ≤ 0.005 against all W jets.
  - The matched-scale edit itself gives W→Z 0.26–0.59 in HGQ, against 0.11–0.26 in float/L1. Overflow is therefore present in the calibration denominator of the HGQ ratios. It was not removed.

## Reading (corrected, scoped to these 12 dsets-d64 models)
- **(d)** A 0.06 rad translation in either Δη direction changes W calls to Z in float, L1 and HGQ-trained Deep Sets: 43–89 % of W jets, against 2–8 % under an equal-displacement rotation. The ridge read-out of log m moves 0.71–1.29 times the shift produced by a matched real angular rescale of size ½Δlog(M0 M2). Under the rotation, the read-out moves −0.005 to 0.007 times that shift.
- **(a)** First-layer units with ≥ 90 % of their input spread from pT alone or from the angles alone exceed the column-permutation null only in the models trained with train_hgq_dsets.py (seeds 0 and 2; 30–50 % of 64 units). That script differs from the float training in quantization and also in optimizer, batch norm folding, epochs, lr, batch size and initialisation. The β = 1e-8 arm (≤ 3.1 % zero weights) shows the same, and its first layer is near-identical to the β = 3e-4 arm at the same seed. In the β = 1e-8 arm the result depends on the threshold (1/3 seeds at 0.95).
- **L1:** 100 % pure on 12–14 live units, equal to its permutation null (p 0.56–0.72).
- **(b), (c):** these follow the X5 numbers and were not rerun.

## Issues
- **No control for the training pipeline.** No float model was trained with the HGQ pipeline, and training is excluded under the current rules. Until one exists, X5's "only under HGQ" for (a) cannot be attributed to quantization.
- **HGQ calibrated ratio.** The HGQ ratio uses a calibration edit with 19–28 % overflowing jets. The HGQ-vs-float ratio difference (X5 regime test, Holm p 0.095–0.14) should not be read further.
- **Null-test p floor.** X5's 200-draw (a) null had a p floor of 0.005. With Holm × 4 the threshold is 0.0125, which seed 2 of the β 1e-8 arm (p 0.010) only just passed. With 2,000 draws its p is 0.0005.
- **Not rerun.** (b), (c) and the regime permutation test.

## Files
- `interp/x5_verify.py`
- `results/jedi/x5_verify_{float,l1,hgq,hgq_acc}_s{0,1,2}.json`, `results/jedi/x5_verify.json`
- logs `/j-jepa-vol/aaron-interp/logs/x5-verify.log`, `x5-verify-b.log`

## Explainer entry (for site/explain.py; not inserted)
```python
X5V_EXPLAINER = [
 ('Question', 'Do the X5 results (the shift response in every training regime; pure first-layer cuts above the permutation null only with HGQ) repeat on new jets, and are they produced by the quantizers?', None),
 ('Background', 'X5 trained the same Deep Sets three ways and measured four JEDI-linear elements. A verifier repeats the measurement on a different sample and adds controls that could break the reading.', None),
 ('Method', 'New sample of 10,000 test jets (seed 4242). (a) Share of first-layer units with $\\ge 90\\,\\%$ of input spread from $p_T$ or from the angles, against 2,000 column permutations, at thresholds 0.8 / 0.9 / 0.95, with and without the input quantizer. (d) W$\\to$Z and the log-mass read-out under translations by $\\pm0.06$ rad in $\\Delta\\eta$, a rotation of equal displacement and an angular rescale by $e^{0.302}$; the read-out shift is divided by the rescale\'s shift. Quantizer overflow ($|{\\rm iq}(v)-v|>2^{-f}$) is counted at every HGQ layer.', None),
 ('Result', 'Translation: 43–89 % of W jets become Z in all 12 models (rotation 2–8 %), read-out ratio 0.71–1.29 (rotation $-0.005$ to $0.007$). Pure cuts above the null: HGQ seeds 0 and 2 only (p $\\le 0.0015$); the two HGQ arms have near-identical first layers (pT-share correlation $\\ge 0.976$).', 'results/jedi/x5_dsets_sparsity.png'),
 ('Checks', 'The phi1 input quantizer leaves the input spread unchanged (sd equal to 4 digits) and never overflows. The rescale edit overflows a later quantizer in 19–28 % of HGQ jets. The HGQ models also differ from float in optimizer, batch norm, epochs and initialisation.', None),
 ('Why it matters', 'The shift response is reproducible and specific to displacement that raises $M_2$. The purity effect is tied to the HGQ training script, not shown to be tied to quantization.', None),
]
```
