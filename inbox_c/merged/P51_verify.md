# P51v: verification of P51 (L-GATr final checkpoint, C6 test: is L-GATr's W-vs-Z score moved by edits that keep the jet invariant mass fixed?)

## Conclusion
On a fresh set of 3,000 W and Z jets, boosting every particle along the beam (jet mass unchanged) again leaves the final L-GATr's W-vs-Z score unchanged (R = −0.0001, against 0.82 for ParT_full), so the boost result of P51 reproduces. The rearrangement edits E1m and E5 keep both the invariant mass and the pT-weighted angular spread M2 fixed (to 10⁻⁵ in log), so a network reading the invariant mass and a network reading the spread about its own momentum axis both predict R = 0 there; ParT_full, which reads the spread about the stored axis, also gives R ≈ 0. These two edits therefore cannot show that L-GATr "follows the invariant mass, not a jet axis". A new control shows that 15–22 % of L-GATr's response to a real mass change passes through its per-particle angle features, which are measured about the jet's four-momentum axis. What is open is an edit that separates the invariant mass from the spread about L-GATr's own four-momentum axis; at small angles the two are equal to leading order (m² ≈ M0·M2).

## What was checked in the code (interp/p46_lgatr_c6.py)
- **Sample.** Random rows from `default_rng(4646)` with `np.sort(rng.choice(...))`, so there is no first-k selection of the class-ordered rows. Jets with fewer than 2 particles are dropped (none were).
- **η build.** `fourvec` and `part_core.build` use η = jet_eta + sign(jet_eta)·Δη, which is the corrected build without the G192 mirror. The BOOST edit uses the same sign factor when it converts the shifted rapidity back to Δη.
- **Padded slots.** All edits multiply by the real-particle mask (`rm`, `r_`). `M2of` sums pT·r², which is 0 in padded slots. `compensate` takes pairs only among real particles (pT > 0).
- **BOOST.** It changes only each particle's rapidity, at fixed pT, mass and φ. Rebuilt four-vectors keep the jet invariant mass to |Δlog m| ≤ 4.5·10⁻⁴ (float32). Route C boosts the time reference with t = (cosh y, 0, 0, sinh y) in the multivector slots e0 and e3. This has the same sign convention as the particle boost, and C changes 0.20–0.27 % of calls (the floor), which confirms the convention.
- **Route plumbing.** D and F swap only the scalar-feature block `e['s']`. The jet structure is asserted equal, and pT ordering is unchanged under BOOST, E1m and E5 because particle pT is fixed.
- **E1m normalisation.** Each E1m dose is divided by the E4 dilation matched to the *uncompensated* E1 step. This is as documented. The edit's own ΔM2 is about 0, so R(E1m) measures the response relative to a real mass change of the size of the first pair move.
- **No bug found** that changes the reported numbers. One limitation of the design is not a bug: E1m and E5 hold M2 as well as m (measured mean |Δlog M2| ≤ 8·10⁻⁵).

## Procedure of the rerun
- **Script:** `interp/p51v_verify.py`. It imports P51's own edit generator (`p46_lgatr_c6.edits`) and changes only its seed and dose list.
- **Job:** `c-p51-verify` (part_job.sh, k8s-haosu-23, RTX 2080 Ti, Complete, about 35 min including a 20-min CephFS stall at import).
- **Sample:**
  - JetClass val_5M WToQQ_124 and ZToQQ_124, which P51 did not use (P51 used _123; L-GATr's W/Z validation file is _121).
  - 1,500 random rows per class from `default_rng(5199)`, N = 3,000.
  - E1 pair plans and E6 directions use seeds 5199 + 1/2/7 instead of 4646 + 1/2/7.
  - Doses ±5 % and ±10 %; E6 was not run.
  - E1 pairs were admissible in 667 jets. E1m N per dose: 222–364.
- **Models:** L-GATr final (`model_run0.pt`), released ParT_full and ParT_kin.
- **Read-out:** d = logit(W) − logit(Z).
- **R** = Σ_jets Δd(edit) / Σ_jets Δd(E4). E4 is the matched dilation: every particle's (Δη, Δφ) about the stored axis is multiplied by k, with k² = M2(edit)/M2. M2 = Σ pT·r² is the pT-weighted second angular moment about the stored axis. E4 is a real change of the jet and pair masses, of the size that the axis-moment mass √(M0·M2), with M0 = Σ pT, predicts for the edit. In words, R is the network's response to the edit divided by its response to a real mass change of the same predicted size.
- **Statistics:**
  - 2,000 bootstrap resamples over jets, one shared count matrix, paired across models and edits.
  - Sign-flip permutation test on the per-jet Δd (5,000 permutations), Holm-corrected over the 6 families per model.
  - Exact McNemar on per-jet correctness.
  - Paired-bootstrap model differences, Holm-corrected over 7 families.

### Added controls (not in P51)
1. **Route split of a real mass change for L-GATr.** For the E4 dilations and for E1 ±10 %, two partial inputs were run:
   - D: four-vectors edited, the 7 per-particle scalar features held at their unedited values.
   - F: four-vectors unedited, only the scalar features taken from the edited jet. The features are log pT, log E, log pT/pT_jet, log E/E_jet, and Δη, Δφ, ΔR about the jet's four-momentum-sum ("E-scheme") axis.

   R_D = ΣΔd_D / ΣΔd_A and R_F = ΣΔd_F / ΣΔd_A, where A is the full edit. R_F is the share of L-GATr's response to a real mass change that passes through its axis-relative scalar features.
2. **BOOST ±0.3** (5× the rapidity), routes A, C, D and F. This tests whether the frame-dependent residual grows with the boost.
3. **Per-jet correlation** of the feature route (A − D) with the time-reference route (D − C) under BOOST. This tests whether the two routes cancel jet by jet.
4. **Rerun** of the unedited input, to measure the GPU nondeterminism floor.

## Results (fresh sample; P51 values in brackets)

Unedited top-1 accuracy on these W/Z jets (10-class argmax, W/Z jets only):

| model | accuracy |
|---|---|
| L-GATr | 71.8 % |
| ParT_full | 73.8 % |
| ParT_kin | 64.1 % |

Rerun floor of L-GATr on an identical input: max |Δd| 0.27, mean |Δd| 0.0017, calls changed 0.03 %.

**BOOST ±0.06** (exact longitudinal boost at fixed invariant mass; stored axis fixed; 6,000 jet-edits):

| model | R [95 % CI] | calls changed | W→Z (true W called Z) | McNemar lost/gained, p |
|---|---|---|---|---|
| L-GATr final | −0.0001 [−0.0018, 0.0013] (P51 −0.0008) | 0.70 % (0.51 %) | 11.7 → 11.5 % | 17/18, p 1.0 |
| ParT_full | 0.820 [0.804, 0.836] (0.805) | 23.5 % | 10.5 → 30.1 % | 963/343, p 2e-68 |
| ParT_kin | 0.745 [0.726, 0.765] (0.753) | 34.3 % | 14.9 → 34.8 % | 1270/412 |

- L-GATr − ParT_full: −0.821 [−0.836, −0.805], Holm p 0.0035.
- Sign-flip p for L-GATr's mean Δd: 0.93 (+0.06) and 0.78 (−0.06).
- H0: L-GATr's mean Δd under BOOST is 0. H1: R ≥ 0.2, a mass-like response. H0 is not rejected, and H1 is excluded by the interval.

**BOOST route split, L-GATr** (+0.06 / −0.06):

| route | mean \|Δd\| | calls changed |
|---|---|---|
| A, full boost | 0.035 / 0.036 | 0.40 / 1.00 % |
| D, features held | 0.126 / 0.126 | 2.3 / 2.9 % |
| F, features only | 0.126 / 0.126 | 2.6 / 2.4 % |
| C, exact symmetry | 0.010 / 0.010 | 0.27 / 0.23 % |

- Per-jet correlation of the feature route (A − D) with the time-reference route (D − C): −0.96 / −0.96.
- The two routes therefore cancel jet by jet, not only on average. This reproduces P51's "equal size, opposite sign".

**BOOST ±0.3** (added control):
- L-GATr mean Δd: +0.015 [0.007, 0.024] and +0.017 [0.009, 0.025], sign-flip p 4e-4 and 2e-4. This is 1.8–1.9 % of the mean Δd under the matched E4 for ±0.06.
- Calls changed 2.7–3.0 %; McNemar p 0.70 and 0.31.
- D and F each change 12–14 % of calls; they cancel again (correlation −0.97).
- At 5× the boost a small, non-zero mean residual appears. It stays below 2 % of a real mass change one fifth its size.

**Route split of a real mass change, L-GATr** (added control 1):

| change | R_D (four-vectors) [CI] | R_F (axis-relative features) [CI] | D + F additivity |
|---|---|---|---|
| E4 dilation, BOOST-matched (±) | 0.80 [0.79, 0.82] | 0.21–0.21 [0.20, 0.23] | 1.01–1.02 |
| E4 dilation, E1-matched (±10 %) | 0.79–0.80 | 0.22 [0.20, 0.23] | 1.01–1.02 |
| E1 ±10 % (pair moved, centroid fixed) | 0.88 [0.85, 0.90] | 0.14–0.15 [0.12, 0.18] | 1.02–1.03 |

- In P51's own data the E1 +10 % values are R_D 0.97 and R_F 0.12, consistent with this.
- Reading: 79–88 % of L-GATr's response to a real mass change enters through the four-vectors. 14–22 % enters through the scalar features Δη, Δφ and ΔR about the four-momentum-sum axis (log E changes only through the rebuilt η).
- So L-GATr does use axis-relative angles. The axis is its own, recomputed from the particles, not the stored axis.

**E1m: pair spread changed, then a second pair moved until the jet invariant mass is restored.**
- Geometry: |Δlog m| ≤ 4·10⁻⁴ (float32) and mean Δlog M2 ≤ 5·10⁻⁵.

| model | raise R [CI] | lower R [CI] | pooled linear (raise − lower) [CI] | even part [CI] |
|---|---|---|---|---|
| L-GATr | 0.05 [−0.40, 0.49] | 0.06 [−0.32, 0.43] | 0.055 [−0.160, 0.292] (P51 −0.047 [−0.226, 0.129]) | −0.01 [−0.36, 0.32] |
| ParT_full | −0.18 | −0.02 | −0.082 [−0.291, 0.122] | −0.06 |
| ParT_kin | 0.56 | −0.66 | −0.197 [−0.341, −0.047] | 0.62 [0.41, 0.86] |

- The pooled linear upper bound is 0.29 here. P51's "R ≥ 0.2 excluded for pooled E1m" therefore does not reproduce on the fresh sample.
- Per-jet |Δd| under E1m (0.83–0.96) is larger than under the matched E4 (0.54–0.55) for L-GATr and ParT_full alike.
- L-GATr calls changed 14.0 % (raise) and 17.0 % (lower). McNemar 41/18 (p 0.004) and 75/23 (p 1e-7): accuracy drops, with no W/Z direction.
- ParT_full: 15.7 % and 15.0 % of calls changed, McNemar p 2e-6 and 8e-9.
- The large undirected response is common to both networks, not specific to L-GATr.

**E5: rearrangement at fixed M2 and centroid** (pooled (Δd(E5+) + Δd(E5−)) / (Δd(E4|E1+10 %) − Δd(E4|E1−10 %))):

| model | R [CI] | calls changed |
|---|---|---|
| L-GATr | −0.026 [−0.156, 0.096] (P51 0.074 [−0.033, 0.185]) | 9.4 % |
| ParT_full | 0.016 [−0.100, 0.139] | 6.9 % |
| ParT_kin | 0.404 [0.297, 0.520] | 15.4 % |

- L-GATr − ParT_full: −0.042 [−0.128, 0.042], Holm p 1.0.

**PHI ±0.06** (rotation about the beam; plumbing check): L-GATr R 0.0003, 0.23 % of calls; ParT_full 0.800.

## Competing hypotheses and which are excluded
- **H_inv: L-GATr reads the invariant mass.** Predicts R ≈ 0 for BOOST, PHI, E1m and E5, and R ≈ 1 for a real dilation.
- **H_stored: L-GATr reads the spread about the stored axis**, as ParT does. Predicts R ≈ 0.8 for BOOST and PHI. Excluded (R ≤ 0.0013), and also excluded by construction, because L-GATr's inputs do not contain the stored axis.
- **H_own: L-GATr reads the spread about its own four-momentum-sum axis.** Predicts R ≈ 0 for BOOST and PHI (the particles do not move relative to that axis) and R ≈ 0 for E1m and E5 (M2 about the centroid held, m held). It predicts the same as H_inv for every edit run in P51 and P51v, so it is **not excluded**. Control 1 shows that 14–22 % of the response passes through features defined about this axis, which is compatible with H_own contributing.
- **H_frame: L-GATr's frame-dependent channels (log E features, time reference) carry a mass-like signal.** Predicts R ≠ 0 under BOOST. Excluded at ±0.06 (|R| < 0.002). At ±0.3 a residual of 1.8 % of a real mass change appears.

## Check of P51's figure (results/lgatr/p51_lgatr_c6.png)
- The right panel matches the json values.
- The labels "E1m … (mass restored)" and "E5 … (mass ~fixed)" omit that M2 is also held fixed in both. A reader would take these rows as discriminating invariant mass from axis spread, which they do not.
- The panel title "L-GATr: R near 0 for every mass-fixed edit" is accurate for L-GATr. The panel also shows ParT_full near 0 on the same E1m/E5 rows, and the report should say what that implies.
- In the left panel, only one of the two orange (first-pair) arrows is distinguishable, because the second sits on top of a large-pT particle.

## Corrected reading
- **BOOST** (exact longitudinal boost, invariant mass fixed, stored axis fixed): the final L-GATr's W-vs-Z read-out moves by |R| < 0.002 of a matched real mass change on two independent samples. ParT_full gives 0.81–0.82 and ParT_kin 0.75. The log-E-type feature route and the time-reference route each change 2.3–2.9 % of calls on their own and cancel jet by jet (correlation −0.96). At a 5× larger boost a residual of 1.8 % of a real mass change appears.
- **E1m and E5** hold both the invariant mass and M2 about the pT centroid. They give R ≈ 0 for L-GATr and for ParT_full alike, and so do not separate an invariant-mass reader from an axis-spread reader. P51's exclusion of R ≥ 0.2 for pooled E1m does not reproduce (fresh upper bound 0.29); for E5 it does (upper bound 0.10).
- **Real mass change**: 14–22 % of L-GATr's response enters through its per-particle angle features about its own four-momentum-sum axis. The supported statement is therefore "L-GATr does not read the stored axis and its frame-dependent inputs cancel". "L-GATr reads the invariant mass rather than any axis" is not supported.

## Explainer entry (for site/explain.py; the ledger agent adds it)
- **Question.** Does a Lorentz-equivariant tagger (L-GATr) change its W-vs-Z decision when the jet is boosted along the beam, which leaves the jet mass unchanged? ParT does (C3).
- **Background.** ParT's mass evidence follows the spread of particles about the stored jet axis, $M_2=\sum_i p_{T,i}\,r_i^2$. L-GATr receives the lab four-momenta, two fixed reference vectors (the beam plane and the time direction $t=(1,0,0,0)$), and 7 per-particle scalars computed about the jet's four-momentum axis. It never receives the stored axis.
- **Method.** BOOST shifts every rapidity, $y_i\to y_i+0.06$, at fixed $p_T$ and mass. This is an exact boost, so $m_{jj}$ is unchanged, while $M_2$ about the stored axis grows by about 17 %. The response ratio is $R=\sum_j\Delta d_j(\text{edit})/\sum_j\Delta d_j(\text{E4})$ with $d=\ell_W-\ell_Z$. E4 multiplies all angles by $k$, $k^2=M_2^{\text{edit}}/M_2$, which is a real mass change of the same predicted size. The route split holds either the scalar features or the time reference fixed.
- **Result.** $R_{\text{L-GATr}}=-0.0001\,[-0.0018,0.0013]$ against $R_{\text{ParT\_full}}=0.82$. The two frame-dependent routes cancel jet by jet (correlation $-0.96$). For a real mass change, $R_F\approx0.15$–$0.22$ of the response goes through the axis-relative scalar features. Figure: P51's p51_lgatr_c6.png (right panel).
- **Checks.** Exact-symmetry route C (0.2 % of calls); rerun floor; fresh sample; boost at 5× (residual 1.8 %).
- **Why it matters.** The axis-relative mass reading in ParT does not arise in a network that is not given the stored axis. Whether L-GATr's mass reading is the invariant mass, or the spread about its own momentum axis, cannot be told apart with edits that keep $m^2\approx M_0M_2$, since the two are equal at small angles.

## Glossary additions
- **R_F / R_D (route shares):** fraction of the response to a real mass change that remains when only the per-particle scalar features (R_F), or only the four-vectors (R_D), are taken from the edited jet.
- **E-scheme (four-momentum-sum) axis:** direction of the summed particle four-momenta. L-GATr's Δη, Δφ and ΔR features are measured about it.
- **Time reference (spurion):** the fixed four-vector (1, 0, 0, 0) that L-GATr receives as an extra token. It breaks exact Lorentz symmetry, so a boost along the beam can change the output through it.

## Files
- Script: /Users/anrunw/Documents/aaron-interp/interp/p51v_verify.py
- Results: /Users/anrunw/Documents/aaron-interp/results/part/p51_verify.json, p51_verify_lg_meta.json (cluster: /j-jepa-vol/aaron-interp/results/part/p51_verify_*.npz)
- Log: /Users/anrunw/Documents/aaron-interp/logs/c-p51-verify.log
