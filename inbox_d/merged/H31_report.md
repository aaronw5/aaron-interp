# H31: Dalitz position inside top. Do top taggers use the W decay angle |cos θ*| at fixed m_top and m_W?

**Status: smoke scale only. The full run did not execute.**
- The smoke run (`d-h31-smoke`, 150 jets per class per dataset) completed, exit 0.
- The full run (2,000 per class) was launched three times (`d-h31-run`, `d-h31-run2`, `d-h31-run3`). Each time the scheduler placed the pod on `fiona8.ucsc.edu`, which failed admission ("UnexpectedAdmissionError … GPU is lost"), so the job never started.
- Keeping the job off that node needs a change to the launcher's node-exclusion list. The permission system refused that change (and the job deletes). The failed job objects `anrunw-ai-d-h31-run2` and `anrunw-ai-d-h31-run3` may still exist; they expire through the 24 h TTL.
- To finish: add `fiona8.ucsc.edu` to the excluded hostnames in `part_job.sh` (or wait until the node is repaired), then run `./sync.sh && ./part_job.sh d-h31-run4 'python -m interp.h31_top_dalitz'`. The smoke run took about 5 min, so the full run should take about 30–45 min.
- The figure code was fixed after the smoke run (panel titles overlapped; the legend lacked the full-input models). The fixed version has been checked only for syntax. It has not been synced to the cluster.

Script `interp/h31_top_dalitz.py`. Smoke results: `results/physics/h31_smoke_top_dalitz.json` and `results/physics/h31_smoke_dalitz.png` (copied from the volume). Log: `/j-jepa-vol/aaron-interp/logs/d-h31-smoke.log`.

## Conclusion
We rotated the W decay inside each jet, in the W rest frame. This changes the W decay angle |cos θ*| and the split between the two b–quark pair masses. The jet mass and the W-pair mass stay fixed.

On JetClass top jets, moving |cos θ*| up by about 0.3 lowered the top score of all nine taggers by 0.8 to 1.5 in log-odds (n = 50 jets). The control, a rotation of equal size that leaves |cos θ*| fixed, was subtracted. A small classifier that sees the pair masses moved by −0.64. On Top Landscape tops the same direction appears, but most intervals include zero.

This is a smoke-scale result: 20–95 jets per cell, and no test passes the Holm correction (its floor is 0.26 with 500 permutations). The edit also changes τ32 and the subjet energy sharing. A classifier given those side effects but no pair masses responds about as much. So these numbers do not yet separate use of the decay angle from use of the energy sharing between prongs.

## Procedure
**Samples.** Random rows, never first-k.
- **Top Landscape:** test split, `h4_top_wmass._sample(150, 3101)` in the smoke run (2,000 per class in the full run). Jets are rebuilt through (pT, η, φ, m) and sorted by pT.
- **JetClass:** `TTBar_100` (Tbqq) and `ZJetsToNuNu_100` (QCD), stored 550 < pT < 650 GeV, |η| < 2, ≥ 3 particles. Rows drawn with rng 3101 / 3102 among eligible rows, with η = jet_eta + sign(jet_eta)·deta.

**Dalitz variables.**
- Three exclusive-kT subjets (E-scheme).
- The W pair (a, b) is the pair with mass closest to 80.4 GeV, giving m_W. c is the third subjet.
- m_bq_hi and m_bq_lo are the max and min of (m_ac, m_bc).
- |cos θ*| = |n·u|, where n is the direction of subjet a and u the direction of subjet c, both in the W rest frame. Quark flavour is not known, so only |cos θ*| is defined.

**Edit D(Δc), Δc ∈ {−0.6, −0.3, +0.3, +0.6}.**
- All constituents of a and b are boosted to the W rest frame.
- n₊ is the daughter direction with n₊·u ≥ 0, and c₀ = n₊·u. The target is c₁ = clip(c₀ + Δc, 0, 0.995).
- One rigid rotation about n₊ × u, by α = arccos c₀ − arccos c₁, is applied to all constituents of a and b, which are then boosted back.
- Unchanged: the W four-momentum, m_W, m_top, all invariant masses inside each subjet, and N.
- Self-test on a synthetic jet: |Δm|, |Δm_W| and |Δm_subjet| < 2·10⁻¹² GeV, and the c₁ target is reached.
- **Acceptance**, after re-clustering the edited jet: |Δm| < 0.5 GeV and |Δm_W| < 2 GeV. The analysis set is the jets where both D and cφ are accepted.
- QCD jets get the same edit on their pair closest to 80.4 GeV.

**Controls.** Same jets, same angle |α|.
- **cφ:** rotation of the W system about u (the W flight direction in the top frame). |cos θ*|, m_ac and m_bc are unchanged exactly at subjet level; constituent positions in the lab frame move.
- **cr:** rotation about a random axis.
- **TN:** rigid (η, φ) translation of the whole jet by D's pT-weighted mean displacement, in a random direction.
- Every edited jet is rebuilt about its own constituent-sum axis for every network (H25 path).

**Models.**
- Top Landscape ParT, PN, LorentzNet, PFN and EFN; JetClass ParT_kin and PN_kin, all on both samples.
- JetClass ParT_full and PN_full on the JetClass sample only.
- JetClass score: L(Tbqq) − L(QCD).

**References.** MLP(64, 64), retrained per dataset on disjoint jets: Top val with rng 4194; JetClass `*_101` in-window jets. Smoke: 2,000 per class; full: 25,000.
- **MLP3:** (m, m_W, τ32).
- **MLP5:** MLP3 inputs + (m_bq_hi, m_bq_lo).
- **MLP5c:** MLP5 inputs + |cos θ*|.
- **MLPx:** the edit's side effects without pair masses: m, m_W, m_SD, τ1–3, τ21, τ32, N, sorted subjet pT fractions, sorted subjet masses.
- Deviation from plan: the references were fitted inside the GPU job, not in a separate CPU job `d-h31-ref`.

**Metrics.**
- **Effect:** D − cφ in top log-odds, with a 95 % bootstrap CI over jets.
- **Tests:**
  - Two-sided sign-flip permutation p for H0: D − cφ = 0, and per jet against MLP3 and MLP5.
  - Holm correction over 128 tests: 9 or 7 networks × 4 Δc × 2 samples × 2 classes.
- **Ratios** to MLP5, MLP5c, MLP3 and MLPx, with bootstrap CIs.
- **Calls:** % changed at ε_S = 0.5 and 0.3, and McNemar D vs cφ.
- **Dose:** Spearman correlation of the means against Δc.
- **C2ST:** edited vs natural pool jets, matched in standardized (m, τ32, m_W, |cos θ*|); HGB on 20 substructure features.

## Checks: unedited models (smoke sample, 300 jets per dataset)
| model | Top sample AUC / acc. | JetClass sample AUC / acc. | known (H25 / H19) |
|---|---|---|---|
| Top ParT | 0.9915 / 95.0 % | 0.9913 / 94.0 % | 0.9867 / 94.3 % |
| Top PN | 0.9900 / 95.3 % | 0.9902 / 95.7 % | 0.9868 / 94.0 % |
| Top LorentzNet | 0.9918 / 95.3 % | 0.9926 / 96.0 % | 0.9879 / 94.5 % |
| Top PFN | 0.9844 / 92.3 % | 0.9834 / 92.0 % | 0.9816 / 93.0 % |
| Top EFN | 0.9805 / 93.0 % | 0.9823 / 94.0 % | 0.9783 / 92.6 % |
| JetClass ParT_kin | 0.9899 / 94.7 % | 0.9952 / 96.3 % | 0.9930 / 95.7 % (JetClass) |
| JetClass PN_kin | 0.9913 / 94.7 % | 0.9950 / 96.0 % | 0.9897 / 94.9 % |
| JetClass ParT_full | – | 0.9980 / 97.0 % | 0.9987 / 98.4 % (H19) |
| JetClass PN_full | – | 0.9972 / 98.0 % | 0.9980 / 98.1 % (H19) |
| MLP3 / MLP5 / MLP5c / MLPx | 0.973 / 0.968 / 0.972 / 0.982 | 0.982 / 0.981 / 0.982 / 0.982 | – |

- All networks are within about 1 point of the earlier values. With 300 jets the uncertainty on accuracy is about ±1.3 points.
- **Translation control TN:** |ΔTN| ≤ 0.19 log-odds for every network and cell, mostly ≤ 0.05.
- **Unedited |cos θ*|.**
  - Top jets fall from 1.2–1.4 (|c| < 0.5) to 0.1–0.5 (0.9–1.0), close to the parton-level shape 1.275 − 0.825 c² for a longitudinal W fraction F₀ = 0.7.
  - QCD jets rise toward |c| = 1 (JetClass: 0.5 at |c| < 0.1, 1.4 at 0.9–1.0).
  - Fitted on the reference jets, adding |cos θ*| to MLP5 raises the internal validation accuracy from 91.0 to 92.5 % (Top) and from 94.0 to 94.25 % (JetClass).

## What the edit does (smoke, top jets; mean changes, D vs cφ)
| sample, Δc | n | Δ\|cos θ*\| D / cφ | Δm_bq_lo D (GeV) | Δτ32 D / cφ | Δm_W D / cφ (GeV) |
|---|---|---|---|---|---|
| Top, −0.6 | 90 | −0.27 / +0.00 | +12.1 | −0.012 / +0.006 | −0.25 / −0.01 |
| Top, −0.3 | 95 | −0.21 / −0.01 | +9.3 | −0.015 / +0.003 | −0.21 / +0.04 |
| Top, +0.3 | 48 | +0.27 / +0.00 | −15.9 | +0.035 / +0.019 | −0.22 / +0.05 |
| Top, +0.6 | 20 | +0.49 / +0.01 | −32.0 | +0.097 / +0.025 | +0.03 / −0.14 |
| JetClass, −0.6 | 77 | −0.30 / −0.01 | +14.3 | −0.022 / +0.010 | −0.11 / +0.04 |
| JetClass, −0.3 | 76 | −0.21 / −0.00 | +9.7 | −0.025 / +0.002 | −0.09 / +0.03 |
| JetClass, +0.3 | 50 | +0.31 / −0.00 | −18.4 | +0.061 / +0.008 | −0.04 / +0.04 |
| JetClass, +0.6 | 30 | +0.50 / −0.00 | −34.0 | +0.107 / +0.056 | −0.05 / −0.18 |

- **Acceptance**, of 300 jets, for D / cφ: 62 / 63 % at Δc = −0.6, 35 / 68 % at +0.3, and 22–25 / 53–66 % at +0.6.
  - At +Δc one daughter becomes soft and wide, so re-clustering often reassigns particles or the W-pair label.
  - The accepted +Δc jets have lower c₀ (Top: 0.23 at +0.6, against 0.32 at −0.6).
- **QCD jets:** the achieved Δ|cos θ*| is smaller (+0.02 to +0.17 at +Δc), because many QCD jets already have |cos θ*| near 1.
- **C2ST AUC** (smoke, ≤ 95 jets): D 0.43–0.60, against 0.35–0.55 for the identity. The smoke sample is too small for this check.

## Numbers: D − cφ (top log-odds) [95 % CI], smoke
**JetClass sample, top jets:**

| model | Δc −0.6 | −0.3 | +0.3 | +0.6 |
|---|---|---|---|---|
| Top ParT | +0.34 [−0.08, +0.81] | +0.32 [+0.03, +0.61] | −0.86 [−1.29, −0.33] | −0.70 [−1.84, +0.47] |
| Top PN | +0.12 [−0.60, +0.72] | +0.47 [+0.05, +0.97] | −1.47 [−2.22, −0.83] | −1.39 [−3.38, +0.47] |
| Top LorentzNet | +0.36 [−0.36, +1.11] | +0.37 [−0.07, +0.83] | −1.13 [−1.65, −0.55] | −0.94 [−2.51, +0.81] |
| Top PFN | +0.48 [+0.13, +0.80] | +0.46 [+0.24, +0.66] | −1.15 [−1.52, −0.77] | −1.16 [−2.12, +0.07] |
| Top EFN | +0.52 [+0.16, +0.90] | +0.48 [+0.19, +0.78] | −1.21 [−1.59, −0.84] | −1.78 [−2.60, −0.89] |
| JetClass ParT_kin | +0.07 [−0.51, +0.61] | −0.12 [−0.45, +0.20] | −0.93 [−1.52, −0.40] | −0.57 [−2.06, +1.20] |
| JetClass PN_kin | +0.29 [−0.19, +0.89] | +0.11 [−0.20, +0.44] | −1.04 [−1.62, −0.48] | −0.82 [−2.35, +0.74] |
| JetClass ParT_full | −0.03 [−0.76, +0.61] | −0.19 [−0.60, +0.21] | −0.96 [−1.57, −0.37] | −0.81 [−2.35, +1.10] |
| JetClass PN_full | +0.20 [−0.39, +0.79] | +0.01 [−0.28, +0.34] | −0.83 [−1.43, −0.28] | −0.61 [−2.09, +1.16] |
| MLP3 (m, m_W, τ32) | +0.23 [+0.08, +0.38] | +0.21 [+0.09, +0.33] | −0.44 [−0.63, −0.26] | −0.38 [−0.79, +0.01] |
| MLP5 (+ pair masses) | +0.33 [+0.18, +0.49] | +0.29 [+0.19, +0.43] | −0.64 [−0.82, −0.46] | −0.76 [−1.17, −0.34] |
| MLP5c (+ \|cos θ*\|) | +0.15 [−0.03, +0.35] | +0.22 [+0.08, +0.39] | −0.49 [−0.74, −0.26] | −0.81 [−1.28, −0.36] |
| MLPx (side effects) | +0.26 [+0.08, +0.44] | +0.20 [+0.08, +0.35] | −0.68 [−0.91, −0.39] | −1.05 [−1.56, −0.51] |

**Top Landscape sample, top jets:**
- **Δc = +0.3:** networks −0.07 to −0.73. ParT −0.58 [−1.18, −0.02]; EFN −0.73 [−1.05, −0.46]; MLP5 −0.25 [−0.46, −0.06].
- **Δc = +0.6:** networks −0.66 to −2.05 (n = 20). Only the PFN and EFN intervals exclude 0.
- **Δc = −0.6 and −0.3:** ParT, PN, LorentzNet, ParT_kin and PN_kin are −0.56 to +0.00, all intervals including 0. PFN and EFN are +0.16 to +0.38. MLP5 is +0.17 / +0.18.

**QCD jets.**
- The networks are mostly within ±0.5 with intervals including 0 on the Top sample.
- On the JetClass sample at +Δc, ParT_kin, PN_kin, ParT_full and PN_full move +0.37 to +0.77 (top-ward). MLP5 moves −0.10 / −0.11 there.

**Holm.** No test passes (minimum Holm p = 0.26, the floor for 500 permutations over 128 tests).

## Reading (smoke scale, scoped to what was measured)
- **JetClass tops, Δc = +0.3:**
  - All nine networks lower the top score beyond the cφ control, with CIs excluding 0.
  - Point ratios are 1.3–2.3 × MLP5 and 1.2–2.2 × MLPx.
  - The sign agrees with H1: moving |cos θ*| toward 1 lowers the top score of top jets.
- **Negative Δc (JetClass tops):** the networks move by −0.19 to +0.52, mostly top-ward, the same sign as MLP5. The pT-normalised Top PFN and EFN show the most consistent response.
- **Not separated from side effects:** MLPx has no pair masses, and its response is the same size as MLP5's in every top cell.
  - D also moves τ32 (+0.06 to +0.11 at +Δc) and the subjet pT fractions.
  - At smoke scale, the data do not separate a response to the Dalitz position from a response to prong-energy sharing and τ32.
- **Competing hypotheses:**
  - H0 (response = MLP3 ≈ 0) is not excluded at the Holm level.
  - H1 (response follows MLP5 in sign and dose) is consistent with the JetClass top cells. The dose is not monotone beyond +0.3, where n drops to 20–30.
  - The third outcome ("opposite sign, or only in full-input models") is not seen on top jets: kin and full models respond alike.
  - On JetClass QCD at +Δc, the JetClass-trained nets move top-ward while MLP5 does not. With 32–45 jets this is not resolved.

## Explainer entry (for site/explain.py; to be added by the ledger/site agent after the full run)
```python
H31_EXPLAINER = [
 ('Question', 'Top taggers use the W-pair mass and the shape of the prongs. Do they also use where the three prongs sit on the top Dalitz plane, i.e. the W decay angle $\\cos\\theta^*$ at fixed $m_t$ and $m_W$?', None),
 ('Physics background', 'In $t\\to bW\\to bq\\bar q\'$ the W is about 70 % longitudinally polarised ($F_0\\approx0.7$). For unlabelled quarks the distribution of $|\\cos\\theta^*|$ in the W rest frame is $\\tfrac32F_0(1-c^2)+\\tfrac34F_L(1+c^2)\\approx1.275-0.825c^2$. At fixed $m_t$, $m_W$, $m_{bq_1}^2+m_{bq_2}^2$ is fixed and $m_{bq_1}^2-m_{bq_2}^2\\propto\\cos\\theta^*$. QCD three-prong jets pile up at $|\\cos\\theta^*|\\to1$ (one soft prong).', None),
 ('Method', 'Three exclusive-$k_T$ subjets; W pair = pair closest to 80.4 GeV. Boost the W pair to its rest frame, rotate both daughters rigidly so that $|\\cos\\theta^*|\\to\\mathrm{clip}(c_0+\\Delta c,0,0.995)$, boost back: $m_t$, $m_W$, the W four-momentum and every subjet\'s internal masses are unchanged. Control $c\\phi$: the same angle about the W flight direction ($|\\cos\\theta^*|$, $m_{bq}$ fixed). Metric: $D-c\\phi$ in top log-odds vs MLPs on $(m,m_W,\\tau_{32})$, $+m_{bq}$, $+|\\cos\\theta^*|$ and side effects.', None),
 ('Result', 'Smoke scale (20-95 jets per cell): on JetClass tops, $\\Delta c=+0.3$ lowers all nine taggers by 0.8-1.5 log-odds beyond $c\\phi$ (MLP with pair masses $-0.64$); no test passes Holm. Full run pending.', 'results/physics/h31_smoke_dalitz.png'),
 ('Checks', 'Unedited AUCs within 1 point of H25/H19; translation control $\\le0.19$; top $|\\cos\\theta^*|$ follows the $F_0=0.7$ shape.', None),
 ('Physical significance', 'If confirmed at full statistics and separated from the prong-energy-sharing side effect, top taggers use the W helicity structure of the decay, not only $m_t$, $m_W$ and prong shapes.', None),
]
```
