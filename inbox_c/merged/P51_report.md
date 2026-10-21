# P51: does L-GATr ignore the jet axis at fixed invariant mass? (pre-registered C6 test, P46 relaunched on the final checkpoint)

## Conclusion
We boosted every particle along the beam, which leaves the jet mass unchanged. The final L-GATr checkpoint did not move its W-vs-Z score: R = −0.001, where R is the score change divided by the change under a real mass change of the same predicted size. ParT_full and ParT_kin gave R = 0.81 and 0.75. Two edits rearranged particles at exactly fixed jet mass: E1m (moves two particle pairs) and E5 (a null rearrangement). Under both, L-GATr's R stayed within a 95 % interval around 0 that reaches at most about 0.3. So L-GATr's W-vs-Z read-out follows the invariant mass, not a jet axis.

Two results limit what this says for claim C6:
- ParT_full also gives R ≈ 0 on E1m and E5. Only the boost (plus the rotation and stored-axis checks, which L-GATr ignores by construction) separates L-GATr from ParT_full.
- The rearrangements still change about 11 % of L-GATr's 10-class calls, but the changes do not point toward W or Z.

## Procedure
- **Model.** L-GATr is our Lorentz-equivariant geometric-algebra transformer, trained in run anrunw-ai-c-p5-lgatr2 with the authors' jctagging config on 2M JetClass jets. Final checkpoint: `model_run0.pt`. The script now maps tag `final` to this file (`CKFILE` in `interp/p46_lgatr_c6.py`).
- **References:**
  - L-GATr checkpoint it174999.
  - Untrained L-GATr: same config, torch seed 4646.
  - Released ParT_full and ParT_kin.
  - The P10 retrains ParT_full_s0 and ParT_kin_s0. These are in the json only.
- **Sample.** Unchanged from P46: JetClass val_5M WToQQ_123 and ZToQQ_123, 2,000 random rows per class (numpy default_rng(4646), not first-k), N = 4,000 W/Z jets. Accuracy sample: 1,000 random rows per class for all 10 classes (N = 10,000). It uses the _123 files, except _124 for QCD, Hcc and TTBar, whose _123 file is L-GATr's validation file.
- **Job.** `./sync.sh && ./part_job.sh c-p51-run '… --stage lgatr --ckpts final,it174999,untrained && … --stage part && … --stage analyze'`.
  - The job ran on k8s-haosu-23 (RTX 2080 Ti), not on k8s-chase-ci-01, and took 46 min.
  - A smoke job, c-p51-smoke (150 jets per class), ran first.
  - The analysis was rerun as a CPU job (c-p51-ana3, `./job.sh`) after adding the pooled E1m and call-flow metrics described below. Model outputs were not recomputed.
- **Read-out.** d = logit(W) − logit(Z).
- **R.** R = Σ_jets Δd(edit) / Σ_jets Δd(matched dilation E4). In words, R is the network's response to the edit divided by its response to a real mass change of the same predicted size. R = 0 means no mean response; R = 1 means a response like a real mass change.
- **E4 (the control).** E4 is a matched angular dilation. Every particle's (Δη, Δφ) about the stored axis is multiplied by k, with k² = M2(edited)/M2 per jet. M2 = Σ pT·r² is the pT-weighted second angular moment about the stored axis, and r is the particle's angular distance from that axis. E4 really changes the jet mass and every pair mass, by the amount that an axis-moment mass estimate √(M0·M2), with M0 = Σ pT, predicts for the edit. It is the right denominator because it is the response a mass reader should give if it reads the edit as a mass change.

## Edits
In every edit, particle pT, particle masses, PID and track inputs are held fixed.

**BOOST ±0.06.** Every particle's rapidity y is shifted by ±0.06 at fixed pT and mass. This is an exact longitudinal Lorentz boost: the jet invariant mass changes by at most 5·10⁻⁴ in log, which is float32 rebuilding. The stored jet axis is not moved. For ParT, the particles therefore sit about 0.06 off the axis in η, and M2 grows by 17 % (mean Δlog M2 = 0.17).

L-GATr's input under a boost is not exactly invariant, because of two pieces:
- the per-particle scalar features log E and log E/E_jet (and, slightly, Δη and ΔR about the E-scheme axis);
- the fixed time-reference "spurion" (1, 0, 0, 0), which does not boost with the jet.

The route split is:
- **A.** The full boost, as the network sees a real boosted jet. The time reference is held fixed.
- **D** (the plan's route b). Four-vectors boosted, the 7 per-particle scalar features held at their unedited values. Of the 7, only log E, log E/E_jet, Δη and ΔR change under a boost.
- **F.** Only the scalar features taken from the boosted jet; four-vectors unedited.
- **C.** Four-vectors boosted, features held, and the time reference boosted with the jet. This is an exact symmetry of L-GATr, so C measures the numerical floor.
- **Route sizes.** Feature route = A − D; time-reference route = D − C. Route (c) of the plan, "time reference held fixed", is A itself.

**E1m_f, f ∈ {±2, ±5, ±10 %}.** First the E1 step (from P43): two similar-pT particles among the 8 hardest are moved apart (or together) along their connecting line about their pT barycentre. This changes the pT-weighted spread M2 about the fixed pT centroid by the fraction f.

Then a compensation step: a second pair (the highest-pT-sum pair among the 8 hardest that shares no particle with the first) is moved together (or apart) along its own line until the jet invariant mass equals the unedited value. The match is solved by float64 bisection: the residual is |Δlog m| ≤ 2·10⁻¹³ in float64 and ≤ 1.7·10⁻⁴ after float32 rebuilding. The pT centroid stays fixed.

N per dose is 307–577 jets: the jets with an admissible pair (920 of 4,000) and a valid compensation. Each dose is normalised by the E4 matched to its uncompensated E1 step (the same pair move).

Pooled forms:
- **Linear part** ("E1m pooled, raise − lower" = antisym) = (Σ raise − Σ lower) / (E4 raise − E4 lower).
- **Even part** (sym) = (Σ raise + Σ lower) / (E4 raise − E4 lower).

These two pooled forms were added after the smoke run showed opposite-sign R for raise and lower. They are not pre-registered.

**E5±.** P43's null rearrangement: the same pair and per-particle displacement size as E1 ±10 %, in the direction that keeps M2 and the centroid unchanged (|Δlog m| ≤ 2·10⁻³). It is pooled as (Δd(E5+) + Δd(E5−)) / (Δd(E4|E1+10 %) − Δd(E4|E1−10 %)).

**PHI ±0.06 and E6 (symmetry and plumbing checks).**
- PHI ±0.06: every particle's φ is shifted by 0.06 rad. This is a rotation about the beam.
- E6: the stored axis is moved by 0.005–0.06 rad, with the lab four-vectors fixed.

Neither can change L-GATr's input. They are reported only as checks.

## Statistics
- **Bootstrap.** 2,000 bootstrap resamples over jets give 95 % intervals. The resamples are paired across edits and models.
- **H0: mean Δd = 0.** Tested with a sign-flip permutation (5,000 flips) and Holm-corrected over the 27 edit arms per model.
- **McNemar test.** Exact McNemar on per-jet correctness before vs after an edit, for paired call changes.
- **W→Z flow.** The share of true-W jets whose 10-class argmax is Z, before → after.
- **Model differences.** Paired-bootstrap differences between models, Holm-corrected over 10 families.

**Hypotheses (pre-registered):**
- **H_thesis.** R(E1m) and R(E5) are about 0 (interval includes 0, below 0.1). BOOST changes at most a few tenths of a percent of calls, and any change is carried by the log E route.
- **H_alt.** L-GATr uses frame-dependent information. R(E1m) is 0.2 or more, and BOOST changes calls through route b or c.

## Numbers

### Accuracy (unedited)
10-class accuracy, N = 10,000:

| model | 10-class accuracy |
|---|---|
| L-GATr final | 84.57 % |
| L-GATr it174999 | 84.55 % |
| untrained L-GATr | 10.0 % (one class for every jet) |
| ParT_full | 86.20 % |
| ParT_kin | 74.26 % |

- The ParT values match the expected 85.9 % and 74.5 % on the P1 sample.
- The quoted 83.73 % for L-GATr final was measured on the P1 sample, not this one; this run did not evaluate the P1 sample.
- L-GATr per-class recall: W 78.3 %, Z 67.7 %.
- Feeding the stored inputs instead of the rebuilt ones gives 84.55 %.

**Rerun floor (L-GATr trained).** The same input run twice gives max |Δd| = 0.25 over 4,000 jets. This is GPU non-determinism in the trained model; the untrained model gives 1·10⁻⁵. The mean |Δd| in the exact-symmetry arm C is 0.010, and PHI changes 0.10–0.15 % of calls. That is the noise level for calls changed.

### BOOST ±0.06 (8,000 jet-edits)

| model | R [95 % CI] | calls changed | W→Z (true W called Z) | McNemar lost/gained, p |
|---|---|---|---|---|
| L-GATr final | −0.0008 [−0.0019, 0.0005] | 0.51 % | 10.80 → 10.78 % | 16/16, 1.0 |
| L-GATr it174999 | −0.000 [−0.001, 0.001] | 0.58 % | 11.25 → 11.25 % | 18/19, 1.0 |
| ParT_full | 0.805 [0.793, 0.819] | 23.4 % | 10.9 → 29.4 % | 1296/430, 1e-100 |
| ParT_kin | 0.753 [0.736, 0.769] | 33.7 % | 16.4 → 36.0 % | 1685/562, 2e-129 |

- **Matched dilation E4 for comparison.** L-GATr final mean Δd = −0.84 / −0.86, calls changed 28–29 %, W→Z 10.8 → 36 %.
- **Model differences.** L-GATr final − ParT_full: −0.806 [−0.819, −0.793], Holm p = 0.005. L-GATr final − ParT_kin: −0.753 [−0.770, −0.736], Holm p = 0.005.
- **Per-arm tests (L-GATr final).** BOOST+0.06: sign-flip p = 0.008, Holm p = 0.13 (mean Δd = +0.0026). BOOST−0.06: Holm p = 1.0.

**Route split, L-GATr final, BOOST+0.06 (BOOST−0.06 alike):**

| route | mean \|Δd\| | calls changed |
|---|---|---|
| A, full boost | 0.035 | 0.35 % (0.68 % for −0.06) |
| D, features held | 0.125 | 2.6 % (2.8 %) |
| F, features only | – | 2.8 % (2.7 %) |
| C, exact symmetry | 0.010 | 0.15 % (0.10 %) |

- The feature route |A − D| and the time-reference route |D − C| are of equal size: the feature route's share of the summed absolute route effects is 0.50. They have opposite sign and cancel in A.
- Mean signed values (+0.06): feature route +0.0001, time-reference route +0.0026.

Untrained L-GATr: R = 0.051 [0.049, 0.054], relative to a tiny E4 response (mean Δd = 0.004); 0 calls changed, because it predicts one class for every jet. 95 % of its route effect is in the feature route.

### Reference E1 (real mass change by the pair spread, no compensation)

| model | R, raise | R, lower |
|---|---|---|
| L-GATr final | 1.08 [0.95, 1.23] | 0.90 [0.77, 1.03] |
| ParT_full | 1.09 [0.97, 1.20] | 0.91 [0.78, 1.04] |

So L-GATr's mean response to a real mass change matches the dilation calibration.

### E1m (jet mass restored exactly)

| model | raise +2–10 % | lower −2–10 % | pooled linear | pooled even |
|---|---|---|---|---|
| L-GATr final | −0.12 [−0.53, 0.26] | 0.00 [−0.29, 0.30] | −0.05 [−0.23, 0.13] | −0.05 [−0.35, 0.23] |
| L-GATr it174999 | −0.11 [−0.51, 0.28] | 0.01 [−0.28, 0.31] | −0.04 [−0.21, 0.13] | −0.05 [−0.35, 0.23] |
| ParT_full | 0.13 [−0.29, 0.54] | −0.25 [−0.57, 0.05] | −0.10 [−0.28, 0.08] | 0.20 [−0.08, 0.51] |
| ParT_kin | 0.56 [0.28, 0.88] | −0.81 [−1.09, −0.56] | −0.25 [−0.39, −0.09] | 0.71 [0.50, 0.95] |
| untrained L-GATr | −0.63 | −0.08 | −0.29 | −0.18 |

The untrained row is a ratio of near-zero responses and has 0 calls changed.

- **Sign-flip tests.** L-GATr final, all six E1m doses: sign-flip p = 0.12–0.89, Holm p = 1.0.
- **Call flow, L-GATr final.** Calls changed: 11.6 % (raise, N = 1,009 jet-edits), 11.4 % (lower, N = 1,504).
  - McNemar lost/gained: 71/32 (p = 1.5·10⁻⁴) and 98/52 (p = 2.2·10⁻⁴).
  - W→Z: 16.2 → 16.5 % (raise) and 13.1 → 14.0 % (lower). The matched E4 at +10 % gives 29.4 %.
  - Per-jet mean |Δd| under E1m is 1.3–2.5 times that under the matched E4, while the signed mean is close to 0.
  - ParT_full: calls changed 11.8 / 12.0 %, W→Z 13.3 → 14.5 % and 11.3 → 12.4 %.
- **Model differences (paired, Holm).**
  - L-GATr final − ParT_full, linear part: 0.05 [−0.09, 0.20], p = 1.0. Even part: −0.26 [−0.43, −0.08], p = 0.046.
  - L-GATr final − ParT_kin, even part: −0.76 [−1.09, −0.45], p = 0.005.

### E5 null rearrangement (920 jets per sign)

| model | R [95 % CI] |
|---|---|
| L-GATr final | 0.074 [−0.033, 0.185] |
| L-GATr it174999 | 0.068 [−0.038, 0.174] |
| ParT_full | 0.063 [−0.053, 0.173] |
| ParT_kin | 0.473 [0.380, 0.581] |

- **L-GATr final.** Calls changed 8.6 %; McNemar 87/43 (p = 1.4·10⁻⁴); W→Z 11.7 → 11.9 % (E5+) and 11.7 → 13.8 % (E5−). Holm p for the mean Δd = 1.0.
- **Model differences.** L-GATr final − ParT_full: 0.011 [−0.079, 0.102], p = 1.0. L-GATr final − ParT_kin: −0.40 [−0.54, −0.27], p = 0.005.

### Symmetry and plumbing checks (L-GATr final)

| check | R [95 % CI] | calls changed |
|---|---|---|
| PHI ±0.06 | 0.000 [−0.001, 0.001] | 0.10 % |
| E6 0.01–0.06 rad | −0.002 [−0.003, 0.000] | 0.18 % at 0.06 rad |

Both are at the rerun floor, as expected by construction. ParT_full gives R = 0.79 on both; ParT_kin gives 0.75 / 0.76.

### Checkpoint stability
L-GATr final − it174999: every family differs by less than 0.03 (Holm p ≥ 0.27).

## Reading (scoped to what was measured)
1. **Boost.** Under an exact longitudinal boost at fixed invariant mass, the trained L-GATr's W-vs-Z read-out moves by less than 0.2 % of a matched real mass change. Calls changed (0.5 %) are a few tenths of a percent above the rerun / exact-symmetry floor (0.10–0.15 %), with McNemar p = 1.0.
   - The two frame-dependent inputs each move the read-out when changed alone. Updating only the boosted log E-type features, or only the four-vectors with the time reference held, changes about 2.7 % of calls and gives mean |Δd| 0.125.
   - Their effects have opposite sign and cancel in a real boost, leaving mean |Δd| 0.035 against a floor of 0.010.
   - The pre-registered expectation that "any change is carried by the log E route" is not what was measured: the residual is split evenly between the feature route and the time-reference route.
   - ParT reads the boost as an axis offset (R 0.75–0.81, W→Z 11 → 29 % for ParT_full).
2. **Rearrangements at fixed mass.**
   - E1m and E5 leave L-GATr's mean W-vs-Z read-out within intervals around 0. The pooled linear R is −0.05 [−0.23, 0.13], and E5 is 0.07 [−0.03, 0.19]. This excludes R ≥ 0.2 for the pooled E1m and for E5, but not for the separate raise and lower arms, whose upper bounds are 0.26 and 0.30.
   - The pre-registered "interval below 0.1" holds for the point estimates but not for the interval upper bounds (0.13–0.30).
   - The edits still change about 11 % (E1m) and 9 % (E5) of 10-class calls, with more correct calls lost than gained. Per-jet |Δd| is as large as under a real mass change, but it has no common sign toward W or Z.
   - L-GATr reads more of the pair arrangement than the invariant mass alone, but not as a mass shift.
3. **Comparison with ParT.** On E1m and E5, ParT_full's mean R is equally close to 0 (difference 0.01–0.05, Holm p = 1.0). ParT_kin responds (E5 R 0.47, E1m even part 0.71). The C6 sentence "ParT_full and ParT_kin give R ≈ 0.7–0.9" holds for edits that move particles relative to the stored axis (BOOST, PHI, E6), not for the centroid-fixed rearrangements E1m and E5.

## Competing hypotheses
| hypothesis | prediction | measured | status |
|---|---|---|---|
| H_thesis (L-GATr reads the invariant mass, not an axis) | BOOST R ≈ 0, calls ≤ few tenths %; E1m / E5 R ≈ 0 | BOOST R −0.001, calls 0.51 %; E1m pooled −0.05 [−0.23, 0.13], E5 0.07 [−0.03, 0.19] | consistent, with wide E1m intervals; route attribution differs (both routes, cancelling) |
| H_alt (frame dependence via log E / time spurion) | E1m R ≥ 0.2; BOOST changes calls via b or c | BOOST A changes 0.51 % of calls; b and c each change about 2.7 % when applied alone but cancel; E1m pooled upper bound 0.13 | excluded for the full boost and the pooled E1m; not excluded for the separate E1m arms (upper bounds 0.26 and 0.30) |
| H_axis (L-GATr reads an axis like ParT) | BOOST R 0.7–0.9 | −0.001, difference from ParT_full −0.81 [−0.82, −0.79] | excluded |

## Uncertain / open
- The E1m intervals are wide: N = 307–577 jets per dose, and the small doses have small E4 denominators.
- The E1m / E5 arms do not separate L-GATr from ParT_full. The separation from ParT rests on BOOST, plus PHI and E6, which hold by construction.
- What L-GATr reads in the 11 % of calls that E1m changes was not identified. Candidates are pair masses of the moved particles and subjet structure.
- The trained L-GATr rerun floor (max |Δd| 0.25) has not been traced. It sits below every effect reported above except the BOOST residual, whose size it limits.

## Files
- Script: `interp/p46_lgatr_c6.py`. The edits for P51 are: tag `final` → `model_run0.pt` (CKFILE), output prefix p51 (`P46_NODE`), `LGATr_final` as main model, route split / per-route calls changed, pooled E1m linear / even parts, per-family call flow, and the new figure.
- Results:
  - `results/lgatr/p51_lgatr_c6.json` (analysis)
  - `results/lgatr/p51_lg_final_it174999_untrained.json` (L-GATr accuracy, rerun floor)
  - `results/part/p51_part.json` (ParT accuracy)
  - `results/lgatr/p51_lgatr_c6_smoke.json` (smoke)
  - The npz files with per-jet logits are on the volume only: `/j-jepa-vol/aaron-interp/results/lgatr/p51_lg_final_it174999_untrained.npz` and `results/part/p51_part.npz`.
  - Sample cache: `/j-jepa-vol/aaron-interp/results/lgatr/p46_sample.npz`.
- Figure: `results/lgatr/p51_lgatr_c6.png`.
  - Left panel: one W jet (pT 520 GeV) with the E1m +10 % edit drawn (arrows at 5× the displacement). The invariant mass is 75.3 GeV before and after.
  - Right panel: R with 95 % intervals per edit family for L-GATr final, L-GATr it174999, ParT_kin and ParT_full.
- Logs: `logs/c-p51-run.log`, `logs/c-p51-ana.log`.

## Explainer entry (for site/explain.py; to be merged by the ledger agent)
**Question.** Does the Lorentz-equivariant tagger L-GATr read the jet's invariant mass, or, like ParT, the spread of particles about a jet axis?

**Background.** ParT's W-vs-Z score moves when every particle is shifted off the stored jet axis even though the invariant mass is unchanged (C3). L-GATr takes the particles' four-momenta $p_i=(E,p_x,p_y,p_z)$ and two fixed reference vectors: a beam plane and a time direction $t=(1,0,0,0)$. It also takes per-particle scalars such as $\log E_i$ and $\Delta\eta_i,\Delta\phi_i$, computed about the axis of $\sum_i p_i$. A rotation about the beam leaves all of these unchanged by construction. A boost along the beam does not: $\log E_i$ changes and $t$ is not boosted.

**Method.**
- Read-out: $d=\ell_W-\ell_Z$.
- Ratio: $R=\sum_{\rm jets}\Delta d_{\rm edit}/\sum_{\rm jets}\Delta d_{\rm E4}$, where E4 scales every particle's angle about the axis by $k=\sqrt{M_2'/M_2}$, $M_2=\sum_i p_{T,i}\,r_i^2$. This is a real mass change of the same predicted size.
- Edits:
  - BOOST: $y_i\to y_i\pm0.06$ for every particle, so $m_{\rm jet}$ is unchanged.
  - E1m: one pair is spread apart, and a second pair is moved together until $m_{\rm jet}$ equals its old value to $10^{-13}$.
  - E5: a pair move at fixed $M_2$.
- Route split for BOOST: hold the scalar features; hold or boost $t$. Boosting $t$ as well is an exact symmetry, which gives the numerical floor.
- Sample: 4,000 W/Z jets (val_5M _123, default_rng(4646)). Statistics: bootstrap intervals, McNemar, Holm.

**Result** (figure `results/lgatr/p51_lgatr_c6.png`).
- BOOST: L-GATr $R=-0.001$ [−0.002, 0.001] and 0.5 % of calls changed. ParT_full: $R=0.81$, with true-W jets called Z rising from 11 to 29 %.
- E1m pooled: L-GATr $R=-0.05$ [−0.23, 0.13]. E5: $R=0.07$ [−0.03, 0.19]. ParT_full is also near 0 on these; ParT_kin is not (E5 0.47).

**Checks.**
- Rotation about the beam and a stored-axis move change 0.1–0.2 % of L-GATr calls, which is the rerun floor.
- Holding $\log E$ alone, or the time reference alone, changes about 2.7 % of calls. The two effects cancel in a real boost.
- The rearrangements change about 11 % of calls without a W/Z direction.

**Why it matters.** The axis-relative mass response found in ParT is absent in a network whose inputs are Lorentz four-vectors. In ParT_full, that response comes from moving particles relative to the stored axis, not from rearranging them at fixed mass.

**Glossary additions:**
- **BOOST (P51).** Every particle's rapidity is shifted by the same amount at fixed pT and mass. This is a Lorentz boost along the beam, so all masses are unchanged.
- **E1m (P51).** A two-pair rearrangement that changes the pT-weighted spread of one pair and then moves a second pair until the jet invariant mass is restored exactly.
- **Time-reference spurion.** L-GATr's fixed input vector (1, 0, 0, 0) that marks the lab frame's time direction. It is not boosted with the jet, so it lets the network depend on the frame.
- **Route split (P51).** The full edit compared against the edit with the per-particle scalar features held (D), only the features changed (F), or the time reference boosted too (C, an exact symmetry).
