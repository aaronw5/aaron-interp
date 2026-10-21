# H44v — skeptic check of H44 (b vs c: impact-parameter size at fixed displaced-track counts; displaced-system mass)

Node H44v (loop D). Script `interp/h44_verify.py` (imports the H44 edit, model and statistics code from `interp/h44_bc_lifetime.py`). Jobs: `d-h44-vsmoke` (CPU 4 / 8 GiB, 311 s, smoke, 200 jets per class), `d-h44-verify` (CPU 8 / 12 GiB, 1,148 s, peak memory 4.4 GiB, 7.1 cores on average). Results: `results/physics/h44_verify.json`, `results/physics/h44_verify_perjet.npz` (smoke: `h44_verify_smoke.json`); logs `logs/d-h44-verify.log`, `logs/d-h44-vsmoke.log`.

## Conclusion

The fresh sample reproduces H44's main measurement. Shrinking displaced-track impact parameters while keeping every count above each significance threshold fixed moves H→bb jets toward H→cc in both ParT and ParticleNet, so the taggers use more than the number of displaced tracks. But an edit that shrinks or stretches each track at random by the same amount moves H→bb jets toward cc by as much (−0.23 log-odds for ParT). The same shrink applied only to prompt (non-displaced) tracks also moves every class toward cc. In JetClass data at fixed counts, b-jet displaced tracks have smaller, not larger, significances than c-jet ones. So "the taggers read the B-hadron lifetime" is not established: about half of the 15–18 % effect is a response to any disturbance of the impact-parameter pattern, and the remaining directional part is not the count-conditional lifetime difference seen in the data. The vertex-mass question is still open.

## What was checked

1. **Code review** of `interp/h44_bc_lifetime.py`: bin map, count assertions, padding mask, sampling, scores, statistics, figures.
2. **Fresh-sample rerun** of L(0.4), Lc(0.4), Lc(2.0), Vm/C1m/Vp/C1p/Cr on different files (`_101` instead of `_100`) with new seeds.
3. **Four missing controls** (defined below): a finer count-preserving map (LcF), a random-direction edit of equal size (Lr), the same map on prompt tracks (LcP), and an S-preserving rescaling (Ls).
4. **Data check (no network):** at fixed displaced-track counts, do b-jet displaced tracks have larger significance than c-jet ones in JetClass?
5. **Pre-registered pT-mismatch adjustment** of the V − C1 contrasts. In H44 this adjustment was post hoc.

## Definitions (as in H44, repeated so this entry reads alone)

- **S** = |d0val| / d0err: the transverse impact-parameter significance of a charged track. d0val is the distance of closest approach to the beam line in the transverse plane; d0err is its uncertainty. The networks see tanh(d0val), d0err, tanh(dzval) and dzerr, not S.
- **Displaced track**: charged, S > 2. **Prompt track**: charged, S ≤ 2. **n(S>s)**: number of charged tracks with S > s.
- **s_bc** = log(p_bb/p_cc) = L_Hbb − L_Hcc, from the 10-class logits. **s_bq** = log(p_bb/p_QCD).
- **Response**: per-jet change of the score (edited minus unedited, both rebuilt with `part_core.build`), averaged over jets. Negative s_bc response means "toward H→cc".
- **L(λ)**: d0val and dzval of displaced tracks × λ. Counts change.
- **Lc(λ)** (H44's edit): inside each S bin (2,3], (3,5], (5,10], an in-bin power map moves S toward the lower edge (λ < 1) or the upper edge (λ > 1); above 10, S' = 10 + λ(S − 10). n(S > 1, 2, 3, 5, 10) are all unchanged.
- **LcF(λ)** (new): the same map on 18 fine bins with edges 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8.5, 10, 12, 15, 20, 30, 50, 100, 200, 500, ∞ (tail: S' = 500 + λ(S − 500)). The track count above each of these 18 thresholds stays fixed (asserted). A network that only counted tracks above thresholds at any of these edges would give 0. A smooth dependence on S would give a smaller response of the same sign.
- **Lr** (new; random-direction control of equal size): each displaced track independently gets the coarse Lc map with λ = 0.4 or λ = 2.5 (probability ½ each, rng 4460). Counts are fixed. The per-jet sum of |Δ ln S| is 3.52 on bb jets, against 3.18 for Lc(0.4); the signed sum is +0.31 (Lc(0.4): −3.18). So Lr disturbs the in-bin S values by the same amount as Lc(0.4), with no net shrinkage. If the Lc(0.4) response came from the direction "shorter displacement", Lr would give about 0 or a small positive value. If it came from disturbing the S pattern at all, Lr would give a response of about the same size.
- **LcP(0.4)** (new; prompt-track control): the in-bin shrink applied only to prompt tracks. For S in (0,1], S' = S^2.5; for S in (1,2], the power map with edges 1 and 2. Displaced tracks are untouched and counts are fixed. Prompt-track impact parameters are dominated by resolution, so a lifetime reading predicts a much smaller response than for Lc(0.4).
- **Ls(0.4)** (new): d0val, d0err, dzval and dzerr of displaced tracks all × 0.4. S is unchanged (every S and every count fixed, asserted), but the absolute displacement and its error shrink. A network that reads only S would give 0.
- **Vm / Vp, C1m / C1p, Cr**: H44's displaced ↔ prompt IP-tuple exchanges inside the subjet with the larger displaced-system mass m_SV. Vm lowers m_SV by ≥ 1 GeV and Vp raises it by 1–4 GeV. C1 is the same kind of exchange with |Δm_SV| ≤ 5 %, and Cr is a random exchange. The **pT mismatch** of an exchange is |ln(ΣpT in / ΣpT out)| of the tracks whose IP tuples are swapped. The **mismatch-adjusted contrast** is the intercept of a per-jet linear regression of (V − C1) on (mismatch_V − mismatch_C1), i.e. the V − C1 contrast at equal mismatch. It was pre-registered in the script header.
- **Permuted band**: [min, max] of the mean response of 4 weight-permuted ParT_full copies (seeds 5–8).
- **Holm**: step-down family-wise correction over the 44 tested cells (ParT_full, PN_full × 11 contrasts × {bb, cc} jets, s_bc). The floor is 0.0088.
- **H0 / H1 tests**: two-sided sign-flip permutation test of H0 "mean response = 0" (5,000 flips); 95 % bootstrap intervals over jets (2,000 resamples).

## Code review of `interp/h44_bc_lifetime.py`

- **Sampling**: random rows via `default_rng(seed).choice`, never first-k. Labels are asserted per class.
- **Padded particles**: `sig()` uses `mask > 0` and `charge != 0`, so padded entries are never counted.
- **Momenta**: four-momenta are never changed and every variant is rebuilt with the corrected η. The network sees exactly the jets whose observables are computed.
- **Bin map** `lc_factor`: correct, with the upper edge included in its bin, consistent with n(S>s). The float32 reverts are counted. The count assertions are real (`np.array_equal` per jet).
- **V edits**: the S-multiset and per-subjet count assertions are correct.
- **Statistics**: the McNemar, Holm and bootstrap code is correct.
- No error that changes a reported number was found.
- **Figure** `h44_jet_edit.png`: the edit is shown as described. Minor issue: in the "unedited" panel the legend covers a neutral particle at (Δη, Δφ) ≈ (−0.36, −0.70) that is visible in the other two panels. `h44_response.png` matches the JSON. It does not show the GBDT reference, which the report quotes.
- **Interpretive gaps** (not code bugs):
  - (i) The only null for Lc is the weight-permuted network. That network has no class structure, so it cannot separate a directional size effect from a general "pattern disturbed → less confident" effect in the trained network. Lr below is that control.
  - (ii) Lc keeps counts fixed at five S thresholds. It does not keep fixed a count in absolute |d0| or a count at any other S threshold.
  - (iii) The "direction expected from the B lifetime" was not compared with the data at fixed counts.

## Sample and unedited accuracy

JetClass test_20M `HToBB_101`, `HToCC_101`, `ZJetsToNuNu_101`: 1,500 random rows each (rng 4451 / 4452 / 4453), N = 4,500. These are different files from H44's `_100`.

| model | 10-class acc | bb-vs-cc acc | bb-vs-cc AUC |
|---|---|---|---|
| ParT_full | 85.4 % | 98.3 % | 0.998 |
| PN_full | 82.6 % | 97.4 % | 0.996 |
| ParT_perm1–4 | 0.0–31.9 % | 48.8–50.1 % | 0.45–0.53 |

Float32 reverts (tracks left unedited because rounding would move them across a bin edge): Lc(0.4) 16, LcF(0.4) 31, Lr 7, LcP 24, Ls 0.

## Results: Δ s_bc, mean [95 % bootstrap CI], Holm p; permuted band in brackets

### H→bb jets (N = 1,500)

| edit | per-jet Σ Δ ln S | ParT_full | PN_full | permuted band |
|---|---|---|---|---|
| L(0.4) | −8.21 | −1.42 [−1.51, −1.32] (0.009) | −1.78 [−1.90, −1.66] (0.009) | [−0.05, +0.04] |
| Lc(0.4) | −3.18 | **−0.202 [−0.227, −0.175]** (0.009) | **−0.294 [−0.347, −0.244]** (0.009) | [−0.012, +0.021] |
| Lc(2.0) | +2.88 | +0.010 [−0.022, +0.039] (1) | +0.052 [−0.004, +0.108] (0.75) | [−0.006, +0.027] |
| LcF(0.4) | −0.89 | −0.034 [−0.042, −0.027] (0.009) | −0.053 [−0.072, −0.034] (0.009) | [−0.010, +0.007] |
| LcF(2.0) | +0.76 | +0.014 [+0.007, +0.021] (0.009) | +0.037 [+0.021, +0.052] (0.009) | [−0.004, +0.003] |
| **Lr (random direction)** | +0.31 (Σ\|Δ\| 3.52) | **−0.232 [−0.271, −0.193]** (0.009) | **−0.263 [−0.338, −0.190]** (0.009) | [−0.013, +0.006] |
| LcP(0.4) (prompt tracks) | −13.6 | −0.192 [−0.232, −0.157] (0.009) | −0.319 [−0.369, −0.271] (0.009) | [+0.001, +0.026] |
| Ls(0.4) (S fixed) | 0 | −3.76 [−3.92, −3.59] (0.009) | +1.57 [+1.43, +1.71] (0.009) | [−0.08, +0.19] |

Lc(0.4)/L(0.4) ratio: ParT 0.14 [0.13, 0.16], PN 0.17 [0.14, 0.19]. H44 found 0.15 / 0.18, so this is reproduced.

### H→cc jets (N = 1,500)

| edit | ParT_full | PN_full | permuted band |
|---|---|---|---|
| Lc(0.4) | −0.014 [−0.035, +0.007] (1) | −0.016 [−0.040, +0.007] (1) | [−0.015, +0.015] |
| Lc(2.0) | **+0.158 [+0.136, +0.179]** (0.009) | **+0.167 [+0.140, +0.194]** (0.009) | [−0.009, 0.000] |
| LcF(0.4) | −0.030 [−0.036, −0.024] (0.009) | −0.032 [−0.040, −0.025] (0.009) | [−0.007, +0.004] |
| LcF(2.0) | +0.034 [+0.029, +0.038] (0.009) | +0.030 [+0.025, +0.036] (0.009) | [−0.003, 0.000] |
| **Lr** | **+0.286 [+0.238, +0.340]** (0.009) | **+0.231 [+0.191, +0.275]** (0.009) | [−0.008, +0.006] |
| LcP(0.4) | −0.307 [−0.330, −0.286] (0.009) | −0.318 [−0.344, −0.292] (0.009) | [−0.012, +0.008] |
| Ls(0.4) | +1.69 (0.009) | +1.10 (0.009) | [−0.04, +0.07] |

### QCD jets (N = 1,500; not in the Holm family)

| edit | Δ s_bc ParT / PN | Δ s_bq ParT / PN |
|---|---|---|
| Lc(0.4) | −0.012 / +0.001 | −0.029 / −0.009 |
| Lc(2.0) | +0.038 / +0.041 | +0.088 [+0.075, +0.101] / +0.076 |
| LcF(0.4) | −0.012 / −0.013 | −0.031 / −0.028 |
| Lr | +0.041 / +0.028 | +0.083 / +0.057 |
| LcP(0.4) | −0.122 / −0.125 | −0.306 [−0.326, −0.289] / −0.284 |
| L(0.4) | −0.137 / −0.106 | −0.236 / −0.196 |

### bb-vs-QCD score on H→bb jets

| edit | Δ s_bq ParT / PN |
|---|---|
| Lc(0.4) | −0.033 [−0.045, −0.023] / −0.042 |
| L(0.4) | −0.40 / −0.41 |
| LcP(0.4) | −0.088 / −0.113 |

H44's −0.034 is reproduced.

### Displaced-system mass (V edits; bb jets: Vm set 1,009, Vp set 1,088; cc jets: 715 / 881)

| contrast | bb ParT | bb PN | cc ParT | cc PN |
|---|---|---|---|---|
| Vm − C1m (raw) | **+0.22 [+0.09, +0.36]** (0.018) | +0.41 [+0.24, +0.59] (0.009) | −0.13 [−0.34, +0.06] (1) | −0.10 (1) |
| Vp − C1p (raw) | −0.47 [−0.60, −0.34] (0.009) | −0.04 [−0.19, +0.11] (1) | −0.21 [−0.40, −0.03] (0.38) | −0.13 (0.75) |
| Vm − C1m at equal pT mismatch (pre-registered) | +0.44 [+0.30, +0.59] | +0.60 [+0.40, +0.79] | −0.19 [−0.39, +0.02] | −0.12 [−0.28, +0.03] |
| Vp − C1p at equal pT mismatch (pre-registered) | −0.44 [−0.58, −0.33] | −0.01 [−0.16, +0.13] | −0.26 [−0.45, −0.06] | −0.17 [−0.32, −0.02] |
| C1m alone | −1.26 [−1.37, −1.14] | −1.87 [−2.03, −1.71] | +1.25 | +0.68 |
| Cr − C1m | −0.36 [−0.50, −0.22] | −0.12 [−0.29, +0.05] | +0.16 | +0.13 |

The raw ParT Vm − C1m on bb jets changed from H44's −0.05 [−0.17, +0.07] to +0.22 [+0.09, +0.36]. The other raw V cells keep their H44 sign. At equal pT mismatch, on bb jets, lowering m_SV moves both networks toward bb and raising it moves ParT toward cc. That is the sign of the data-fitted LR_vm (at fixed n(S>2), higher m_SV is more cc-like), the opposite of the physical vertex-mass expectation. On cc jets both directions move toward cc, so no monotone function of m_SV fits both classes. The C1 exchange alone (same mass) is 2–4 times larger than any mass contrast.

### Data check: in-bin significance at fixed counts (no network)

For each jet with displaced tracks, three quantities were computed: u_bin, the mean in-bin position ln(S/lo)/ln(hi/lo) over displaced tracks in the finite bins (2,3], (3,5], (5,10]; the mean ln(S/10) over tracks with S > 10; and the median and maximum ln S. Each is compared as bb − cc within strata of identical (n(S>2), n(S>3), n(S>5), n(S>10)), using strata with ≥ 5 jets of each class and weighting by stratum size. Intervals are 95 % bootstrap over jets (1,000 resamples). Sample: train_100M `HToBB_001` / `HToCC_001`, 20,000 random rows each (rng 4461 / 4462); the test sample with truth labels is shown alongside.

| quantity | bb − cc at fixed counts, train (≈ 25–29k jets in strata) | test sample (≈ 400 jets in strata) |
|---|---|---|
| u_bin (finite bins) | +0.017 [+0.009, +0.026] | +0.009 [−0.059, +0.060] |
| mean ln(S/10), S > 10 | −0.94 [−0.98, −0.89] | −0.84 [−1.10, −0.58] |
| median ln S | −0.42 [−0.46, −0.40] | −0.48 [−0.73, −0.26] |
| max ln S | −0.99 [−1.06, −0.96] | −1.25 [−1.57, −0.80] |

At fixed counts, the displaced tracks of H→bb jets have smaller significances than those of H→cc jets, mainly in the S > 10 tail. Inside the finite bins the difference is about 0.02 of a bin width. Lc(0.4) moves the tail most (H44: max ln S −0.72 on bb jets). So the networks' direction (shorter → cc) is opposite to the count-conditional relation in the data. This agrees with H44's LR_life and GBDT references, which predicted +0.17 for Lc(0.4). The networks' direction can still come from relations that also involve other inputs (track pT, angle, d0err, dz). Ls shows that the networks depend strongly on d0err and absolute d0 at fixed S, and with opposite signs for ParT and PN. These data do not show which of these relations the networks use.

## Hypotheses and which are excluded

- **H0a (count only at the thresholds S = 1, 2, 3, 5, 10)**: excluded. Lc ≠ 0 outside the band, reproduced on new files (Holm 0.009).
- **H0b (count only at any threshold among 18 finer edges)**: excluded. LcF is ≠ 0 in both directions in both Higgs classes and on QCD, and its sign follows λ. Per unit Σ Δ ln S, LcF(0.4) gives 0.039 (ParT, bb) against 0.063 for Lc(0.4), consistent with a dependence on the impact-parameter values that is smooth rather than step-like.
- **H1a as stated by H44 ("the Lc(0.4) response is the directional lifetime effect, 15–18 % of full scaling")**: not supported in that form. The random-direction edit Lr, of equal per-track size and with net shrinkage +0.31 (slightly toward longer), gives −0.23 / −0.26 on bb jets and +0.29 / +0.23 on cc jets, i.e. toward the other Higgs class as large as Lc(0.4). A purely directional effect predicts about 0 or a small positive value on bb jets; it is excluded as the full explanation. A directional component remains. Lc(2.0) and Lc(0.4) differ in the direction H1a predicts in both classes: ½[Lc(2.0) − Lc(0.4)] = +0.11 (ParT) / +0.17 (PN) on bb jets and +0.09 / +0.09 on cc jets. The small LcF edits are antisymmetric in λ in all three classes. The symmetric part, ½[Lc(2.0) + Lc(0.4)], is −0.10 / −0.12 on bb jets (toward cc) and +0.07 / +0.08 on cc jets (toward bb). This part is a move toward the other class for any disturbance of the pattern. It is consistent with the large Lr response, and not with lifetime.
- **H1a specificity (displaced tracks only)**: excluded. LcP(0.4), which leaves displaced tracks unchanged, moves s_bc by −0.19 / −0.32 (bb), −0.31 / −0.32 (cc) and −0.12 / −0.13 (QCD).
- **H44's "D2 holds for bb vs QCD"**: supported only on H→bb jets for the displaced-track edit (Δ s_bq −0.033). On QCD jets the bb-vs-QCD score responds at fixed counts to Lc(2.0) (+0.088, 37 % of |L(0.4)|) and to the prompt-track shrink LcP (−0.31, larger than L(0.4)'s −0.24).
- **H1b (physical vertex-mass direction, lower m_SV → more cc-like)**: excluded on bb jets for both networks at equal pT mismatch (Vm +0.44 / +0.60, opposite sign). Not resolved on cc jets. Same status as H44: open.

## Corrected reading

1. With every count above S = 1, 2, 3, 5, 10 fixed, and also with counts fixed at 18 finer thresholds, ParT_full and PN_full respond to the impact-parameter values of tracks. H0 (count only) is excluded for bb vs cc. This is reproduced on new files.
2. About half of the Lc(0.4) response on H→bb jets (−0.20 / −0.29) does not depend on the direction of the edit: a random shrink-or-stretch edit of the same per-track size gives −0.23 / −0.26.
3. The part that does depend on direction (shorter → cc-like, longer → bb-like) is about +0.1 log-odds between Lc(0.4) and Lc(2.0) on bb and cc jets.
4. The same kind of edit on prompt tracks gives responses as large, in all classes.
5. In the data at fixed counts, b-jet displaced tracks have smaller significances than c-jet ones. So the response is not shown to be a use of the B-hadron lifetime. It is a dependence on the impact-parameter values whose physical content is not identified.
6. On QCD jets the bb-vs-QCD score also depends on impact-parameter values at fixed counts (prompt-track edit −0.31), so the scoping "D2 holds for bb vs QCD" applies only to the displaced-track edit on H→bb jets.
7. Vertex mass: at equal pT mismatch, both networks on bb jets move opposite to the physical expectation, and cc jets do not follow one direction. The question is unresolved.

## Thesis note for the ledger

- D2 should not be re-scoped to "lifetime used for b vs c".
- A supported statement is narrower: "at fixed displaced-track counts, ParT and ParticleNet respond to impact-parameter values; about half of the response on H→bb jets is direction-independent; the direction-dependent part has the sign of a lifetime reading but is opposite to the count-conditional data relation; prompt-track IPs matter as much" (exploratory, about 0.5).
- Next test: a matched edit that changes S jointly with track pT and angle along the decay relation d0 ≈ γβcτ·sin θ (the lifetime relation), against an edit of equal size that breaks it. This would separate a lifetime reading from a general IP-pattern dependence.

## Explainer entry (for site/explain.py; to merge into H44_EXPLAINER under 'Checks')

```python
('Skeptic check (H44v)', 'Fresh JetClass files (1,500 jets per class). Lc(0.4) reproduced ($-0.20$ ParT, $-0.29$ ParticleNet on $H\\to b\\bar b$). Three new controls: (1) LcF, the same map on 18 finer significance bins, gives a smaller response of the same sign, so the networks do not only count tracks above thresholds; (2) Lr, which shrinks or stretches each displaced track at random by the same amount ($\\sum|\\Delta\\ln S|$ 3.5 vs 3.2), moves $b\\bar b$ jets toward $c\\bar c$ by $-0.23$ and $c\\bar c$ jets toward $b\\bar b$ by $+0.29$, so about half of the Lc response does not depend on the direction of the edit; (3) LcP, the same shrink on prompt tracks ($S\\le2$), moves every class toward $c\\bar c$ by $0.12$–$0.32$. In JetClass at fixed counts, $b\\bar b$ displaced tracks have smaller $S$ than $c\\bar c$ ones (max $\\ln S$ $-0.99$), so the direction-dependent part is not the count-conditional lifetime difference in the data.', None),
```

Glossary additions:
- `('Lr (random-direction control)', 'an edit that shrinks or stretches each displaced track\'s impact parameter inside its significance bin, choosing shrink or stretch at random per track, so that the edit has the same size as Lc but no net direction')`
- `('LcF (fine count-preserving map)', 'the Lc map on 18 narrow significance bins, so the number of tracks above each of 18 thresholds stays fixed')`
- `('LcP (prompt-track control)', 'the Lc shrink applied only to prompt tracks (S ≤ 2), whose impact parameters are dominated by measurement resolution')`
