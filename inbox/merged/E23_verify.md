# E23 verification: recentring off-centre jets, fresh sample and a displacement-matched control (2026-10-17)

**Node:** E23 (A4 / D4 read-out claim, review weakness 1). Verifier rerun of arm (B) and arm (C) on a new sample, plus four controls that E23 did not run.

**Files**
- Script: `interp/e23_verify.py`. It imports the E23 helpers (`moments`, `edit_shift`, `edit_scale`, `edit_rot`, `Boot`, `Ridge`) from `interp/e23_tail_jedi.py` unchanged.
- Results: `results/jedi/e23_verify.json`.
- Logs: `logs/e23-verify.log`. The smoke run (`e23-vsmoke`, 500 per class) wrote to the same JSON path and was overwritten.
- Jobs: `e23-vsmoke`, then `e23-verify`. Both ran at 1 CPU / 2 GiB. Wall time 907 s, CPU 118 s, pod peak memory 2048 MiB. `sample()` loads the full test array, so the peak sits at the limit, as it did in E17 and E23.
- No figure was made. The E23 figure was opened and checked (see the end of this report).

## Code review of `interp/e23_tail_jedi.py`

**Edits**
- Only real particles ($p_T>0$) move. $p_T$, slot order and padded slots stay the same; the run asserts this.
- Recentring and the reflected edit shift $\Delta\eta$ (a longitudinal boost) and $\Delta\phi$ (an azimuthal rotation), so $m$ is invariant up to float32 rounding ($|\Delta\log m|\le4.4\times10^{-8}$).
- The rotation preserves $M_2$ and $|\vec M_1|$.
- The P freeze overrides the nine units with zero $p_T$ weight and leaves the rest alone. On unedited jets it reproduces the logits exactly.

**Read-out**
- The read-out sample comes from a different seed and excludes jets that are also in the main sample. Its mean and sd come from the fit half only, so the read-out does not leak into the test.

**Statistics**
- Ratios use paired bootstrap draws (the same indices for numerator and denominator).
- McNemar is exact and two-sided. Holm runs over 37 set × edit tests.

**Defects found**
- The even/odd split has no interval (E23 issue 3).
- The rotation is a weak null for "generic displacement" (see C1 below).

No bug was found that changes a reported number.

## Procedure
- **Sample:** `sample(20000, 211)`, N = 100,000. Unedited accuracy 80.86 %.
- **Read-out:** ridge of $\log m$ on $z$, E17 procedure. It is fitted on `sample(4000, 213)` with the 7,656 shared jets removed: 12,344 jets, 6,172 used for the fit, $\alpha=10^{-2.5}$. Held-out $R^2$ is 0.944 (0.941 on the main sample).
- **Sets:** as in E23.

| set | n | g / q / W / Z / t | $q$ p50 | $d$ p50 [mrad] |
|---|---|---|---|---|
| Tq1 | 1,000 | 311 / 72 / 1 / 4 / 612 | 0.0068 | 14.3 |
| Tq5 | 5,000 | 1053 / 341 / 15 / 14 / 3577 | 0.0045 | 11.0 |
| Td5 | 5,000 | 754 / 276 / 11 / 6 / 3953 | 0.0044 | 11.0 |
| Tq5c | 5,000 | 1000 each | 0.0028 | 6.3 |

- **Edits rerun:** recentre ($\vec c=-\vec M_1/M_0$), reflect ($+\vec M_1/M_0$), rotation (c3), matched real change (offsets $\times e^{\rm pred}$), and recentre with the angle P units frozen (c4).
- **New controls:**
  - **C1, perp.** A rigid shift by $d=|\vec M_1|/M_0$ perpendicular to $\vec M_1$, run with both signs and averaged per jet.
    - Every particle moves by exactly the recentring displacement (checked to $6\times10^{-8}$).
    - Because $\vec c\cdot\vec M_1=0$, $M_2\to M_2+M_0d^2$, so the prediction is $\tfrac12\log(1+q)>0$. $m$ is unchanged ($|\Delta\log m|\le4\times10^{-8}$).
    - Predictions:
      - H1 (read-out follows $\sqrt{M_0M_2}$ about the stored axis): Δread = gain × pred > 0.
      - H0 (invariant mass): 0.
      - H2 (generic displacement lowers the read-out): Δread < 0, near the recentring value.
    - The matched real change for this edit is "cal_perp".
  - **C2.** Matched real change at 4× the size (offsets $\times e^{4\,\rm pred}$), response divided by 4. This checks the calibration edit for linearity and input-quantizer effects.
  - **C3.** Matched real change with the same nine angle-only P units frozen. E23 compared the P-frozen recentring with the unfrozen matched change.
  - **C4.** Per-jet even and odd parts, with bootstrap intervals:
    - even $=\tfrac12(\Delta_{\rm rec}+\Delta_{\rm ref})$;
    - odd $=\tfrac12(\Delta_{\rm ref}-\Delta_{\rm rec})$;
    - even − perp, tested with a sign-flip permutation.
- **Statistics:** 1,000-draw jet bootstrap, paired across edits; sign-flip permutation with 4,000 draws; Holm over 8 normal-approximation tests (recentre and perp × 4 sets).

## Results (read-out change ×10⁻³, 95 % bootstrap CI)

| set | recentre: pred / Δread | real Δread (1×; 4×/4) | **obs/real** (E23) | reflect obs/real (E23) | **perp: pred / Δread** | perp obs/real | rotation Δread |
|---|---|---|---|---|---|---|---|
| Tq1 | −3.76 / −10.9 [−13.3, −8.6] | −4.37; −4.26 | **2.50 [1.91, 3.27]** (1.86) | 1.26 [1.08, 1.45] (1.16) | +3.73 / +2.18 [+0.76, +3.54] | 0.46 [0.17, 0.76] | −0.93 [−2.85, +1.03] |
| Tq5 | −2.47 / −5.03 [−5.91, −4.15] | −2.42; −2.38 | **2.08 [1.67, 2.49]** (1.56) | 1.12 [1.00, 1.23] (1.06) | +2.46 / +1.98 [+1.43, +2.54] | 0.80 [0.58, 1.04] | +0.20 [−0.49, +0.87] |
| Td5 | −2.38 / −4.86 [−5.74, −3.90] | −2.40; −2.35 | **2.03 [1.63, 2.46]** (1.50) | 1.13 [1.00, 1.25] (1.08) | +2.36 / +1.96 [+1.37, +2.50] | 0.78 [0.56, 1.01] | +0.07 [−0.63, +0.82] |
| Tq5c | −1.86 / −4.40 [−5.19, −3.61] | −1.95; −1.95 | **2.26 [1.80, 2.74]** (1.57) | 1.16 [1.01, 1.29] (1.19) | +1.85 / +1.46 [+0.96, +1.93] | 0.73 [0.52, 1.00] | +0.07 [−0.58, +0.72] |

**Recentre, H0 (Δread = 0)**
- Holm-adjusted normal p from 10^−18.2 to 10^−27.1.
- Permutation p is at its floor of 1/4001.

**Perp, H0**
- Holm-adjusted normal p is 10^−2.5 (Tq1) and 10^−7.8 to 10^−10.4 (the three 5 % sets).
- Permutation p: 0.006 (Tq1) and 1/4001 for the rest.
- Perp raises the read-out, with the opposite sign to recentring at the same per-particle displacement. This excludes H2.

**Matched real change**
- Response/prediction: 0.98–1.05 for the three 5 % sets, 1.16 for Tq1 (E23: 1.11–1.53).
- The 4× edit divided by 4 agrees with the 1× edit to within 0.11×10⁻³ in every set, so the calibration edit is linear and not set by the input quantizer.

**Even/odd parts (per jet, with intervals; E23 gave point values only)**

| set | even obs/pred (E23) | odd obs/pred (E23) | even − perp |
|---|---|---|---|
| Tq1 | 0.70 [0.30, 1.08] (0.94) | 1.82 [1.56, 2.07] (1.66) | +0.39 [−1.02, +1.82], p 0.58 |
| Tq5 | 0.64 [0.42, 0.88] (0.75) | 1.34 [1.21, 1.48] (1.24) | −0.42 [−1.01, +0.12], p 0.12 |
| Td5 | 0.71 [0.44, 0.96] (0.80) | 1.38 [1.23, 1.52] (1.28) | −0.32 [−0.84, +0.26], p 0.28 |
| Tq5c | 0.62 [0.34, 0.92] (1.04) | 1.50 [1.34, 1.66] (1.43) | −0.33 [−0.83, +0.17], p 0.18 |

- The even part of recentre/reflect equals the perp response within the intervals. The response is therefore consistent with a term quadratic in the shift plus a term linear in $\vec c\cdot\vec M_1$.
- On the 5 % sets, the quadratic term runs at 0.6–0.8 of its $\sqrt{M_0M_2}$ prediction. The bulk random shifts in (C) give the same, 0.56–0.57.
- The linear (cross) term runs at 1.3–1.8 of its prediction, and every interval excludes 1.
- So the read-out is not a function of $\sqrt{M_0M_2}$ about the stored axis alone: the two terms carry different gains.

**P freeze (C3)**
- Freezing the angle-only P units leaves the matched real change almost unchanged: Tq5 −2.38 vs −2.42 unfrozen.
- It reduces the recentring response to 65–80 % of its unfrozen value (E23: 57–80 %).
- The frozen recentring over the frozen real change is 1.52–1.62, and every interval excludes 1 (Tq5 1.60 [1.30, 1.90]). 20–35 % of the response passes through the P → g route. The excess over the matched real change remains with that route closed.

**Calls**
- Calls changed by recentring: Tq5 1.80 %, Tq5c 2.84 %.
- By perp: Tq5 1.96 / 1.86 % for the two signs; Tq5c 3.00 / 2.86 %.
- Recentre − perp: Tq5 −0.11 pp [−0.49, +0.24]; Tq5c −0.09 pp [−0.51, +0.35]; Tq1 +0.2 pp [−0.75, +1.25].
- Recentre − rotation: Tq5c +0.74 pp [0.26, 1.20], as in E23.
- Matched real change: 0.3–0.7 %.
- Z→W vs W→Z for Tq5c recentring: 8 vs 6 (McNemar p 0.79). E23 had 10 vs 5. Perp: 4 vs 9 (p 0.27).
- Tq5c accuracy: 72.48 → 73.04 %.

**(C) Random shift on 20,000 bulk jets**

| size | ratio | calls changed | Z→W vs W→Z (p) |
|---|---|---|---|
| p50, 1.47 mrad | 0.78 [0.47, 1.10] | 1.36 % | 35 vs 32 (0.81) |
| p90, 5.84 mrad | 0.56 [0.53, 0.60] | 2.43 % | 37 vs 76 (3.1e-4) |
| p99, 13.1 mrad | 0.57 [0.56, 0.58] | 4.06 % | 22 vs 175 (9e-31) |

These reproduce E23.

## Hypotheses: predictions and outcome (fresh sample)
- **H0 ($m$):** predicts recentre = perp = 0. Excluded: recentre Holm p ≤ 10^−18, perp ≤ 10^−2.5.
- **H2 (generic displacement lowers the read-out):** predicts perp ≈ recentre < 0. Excluded: perp is +1.5 to +2.2×10⁻³ against recentre −4.4 to −10.9×10⁻³, at the same per-particle displacement.
- **H1 (function of $\sqrt{M_0M_2}$ about the stored axis, one gain):** predicts recentre, reflect and perp all at about 1× their matched real change. Measured:
  - recentre 2.0–2.5;
  - reflect 1.12–1.26;
  - perp 0.46–0.80.
  - The cross term runs at 1.3–1.8, and its interval excludes 1. H1 with one gain is excluded. What holds is the sign of ½Δlog(M0M2) about the stored axis for all three edits.

## Controls: adequacy
- **Rotation (c3):** a weak null for H2. A rotation about the axis preserves every radial quantity. Seventeen of the 19 second-round objects are radial (A3), so a rotation leaves them unchanged by construction. Its zero read-out response does not test generic sensitivity to translation. The perp shift does, and it gives the H1 sign.
- **Bulk recentring (c1):** its displacement is 0.65 mrad against 8–15 mrad on the tails, so it does not control for displacement size.
- **Matched real change:** linear (4× check), and close to 1× of the prediction on this sample.

## Corrections to the E23 reading
1. **Magnitude.** E23 gives "1.5–2.0× the matched real change". On the fresh sample the ratio is 2.03–2.50, and the intervals reach 1.63–3.27. Over both samples the four common sets give 1.50–2.50.
2. **Even part.** E23 gives "even part 0.75–1.04 of its prediction". On the fresh sample it is 0.62–0.71 (intervals 0.30–1.08), equal to the perp response. The odd part, 1.34–1.82 here against 1.24–1.66 in E23, has intervals that exclude 1.
3. **H2 exclusion.** E23 rests it on the rotation and the sign reversal. The rotation is not displacement-generic. The displacement-matched perpendicular translation now excludes H2 for the read-out: it raises the read-out at 0.46–0.80 of its matched real change.
4. **Calls.** At natural offsets, recentring changes calls at the rate of a perpendicular translation of the same size (difference −0.1 pp, interval ±0.4 pp). Both lower and raise the predicted mass, so the call changes at 6–15 mrad follow the translation size, not the sign of the mass change. The mass-directional W→Z excess appears at 5.8–13 mrad random shifts on bulk jets (reproduced).
5. **Thesis note.** The note says the edit "moves the read-out with $\sqrt{M_0M_2}$ about the stored axis". As measured, the read-out follows the sign of $\tfrac12\Delta\log(M_0M_2)$ for recentre, reflect and perp. The gain is 0.5–2.5 of a matched real change and depends on the edit: the cross term carries 1.3–1.8 of its prediction and the quadratic term 0.6–0.8. The phrase "answers review weakness 1" is editorial; it would read as a statement of what was measured.

## Figure (`results/jedi/e23_tail.png`)
- Axes are in physical units (×10⁻³ in log), with 95 % bars and a legend.
- In E23 all nine rotation points sit slightly below 0; on the fresh sample they are ≈ 0.
- The title "Recentring (m fixed) lowers the mass read-out; rotation does not" is a declarative statement, and the left panel holds 9 sets × 5 edits.
- The figure needs no redo for correctness.
