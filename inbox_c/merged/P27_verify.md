# P27 verify: per-particle kernel of the axis response (ParT_full/kin, ParticleNet_full/kin) (verifier, 2026-10-20)

## Conclusion
On fresh jets, all of P27's main numbers came out the same. A single particle's effect on the W-vs-Z score still cannot be predicted from its position (held-out R² 0.02–0.08). The new controls change two parts of the reading. First, the average response does not follow the axis moment M2 = Σ pT r² as closely as P27 says. At fixed pT share, the response rises with the particle's distance r from the axis, but only about 0.6–0.8 times as steeply as proportionality to M2 would require. A pT-share-only formula fits the binned averages as well as M2 does. Second, in ParT the large "around-the-axis" component is carried by the pair inputs (the particle-pair features built from four-vectors). Freezing those inputs removes 96–98 % of it, and what remains is a smooth response that is mostly odd under out vs in. The component is also not smooth in the step size. ParticleNet carries the same component in its per-particle features, not in its kNN coordinates, so this test does not separate neighbour effects from per-particle effects there.

## What was checked in the script (`interp/p27_particle_kernel.py`)
- Rows come from `rng.choice` over the whole file and are not first-k. Padded slots are never edited, because z > 0.5 % needs pT > 0.
- Build is `part_core.build`, with η = jet_eta + sign(jet_eta)·deta. This is the corrected build without the η mirror.
- pT is kept, so the pT order is unchanged. `read_root` sorts by pT, so `rank` = array index = pT rank. The verify run confirms that the stored pT rows are non-increasing.
- The read-out is computed against the rebuilt unedited jet in the same chunk. Odd/even and azimuth look-ups are keyed by (jet, particle).
- The Holm family has 108 tests (4 models × 3 targets × 9 forms).
- No layer or head indexing is involved, and no hooks are used.
- No code bug was found.
- One reading issue: in P27's own JSON, `pt_only` reaches binned-mean R² 0.96–1.00 against 0.91–0.95 for M2, and its held-out R² is within 0.01 of M2. The binned R² of about 0.9 therefore does not separate "∝ dM2/M2" from "∝ pT share".

## Procedure (verify run)
- Script `interp/p27v_particle_kernel_verify.py`.
- Jobs:
  - `anrunw-ai-c-p27-verify` (GPU GTX 1080 Ti, smoke stage then full run, 938 s).
  - `anrunw-ai-c-p27-verify-ana2` (CPU, 31 s, re-analysis that adds route-specific fits). `c-p27-verify-ana` was launched with an unedited script and deleted unused.
- Results: `results/part/p27_verify.json`, `p27_verify_info.json`, `p27_verify.png`. The npz is on the volume only.
- Sample: test_20M `WToQQ_107` and `ZToQQ_107` (P27 used `_106`), 1,000 random rows each (`default_rng(2777)`). N = 2,000 jets and 48,628 eligible particles (24.3 per jet).
- Unedited 10-class accuracy on this W+Z sample:

| model | accuracy |
|---|---|
| ParT_full | 74.75 % |
| ParT_kin | 62.50 % |
| PN_full | 72.60 % |
| PN_kin | 57.85 % |

  P27 found 75.1 / 62.5 / 73.0 / 59.2 %. P27 also reproduced 85.93 % for ParT_full on the P1 sample.
- Edits per eligible particle:
  - outward and inward radial moves of 0.01 rad;
  - a fixed-r rotation with chord 0.01 and random sign;
  - outward and inward radial moves of 0.003 rad.
- Read-out: d = Δ(logit_Z − logit_W). Bootstrap uses 1,000 resamples over jets. p-values use the normal approximation to the bootstrap.
- **Control 1, route split.** Every edit was run three times:
  - full: edited per-particle features and edited pair inputs;
  - part: edited per-particle features with unedited pair inputs. For ParT the pair inputs are the four-vectors that build the pair features; for PN they are the first-EdgeConv kNN coordinates;
  - pair: unedited per-particle features with edited pair inputs.

  Unedited rows agree across the three runs to 0.0 (max |Δ|).
- **Control 2, step size.** For a smooth response the even/odd variance ratio scales by (0.003/0.01)² = 0.09.
- **Control 3, pT vs M2.** d was regressed on 10 z-decile indicators plus x = dM2/M2. The ratio β_within-z / β_M2-only is 0 under a pT-only reading and 1 under proportionality to M2.

## Results
**Replication** (fresh rows; P27 values in parentheses; order ParT_full / ParT_kin / PN_full / PN_kin):
- mean d_rad: 0.029 [0.027, 0.032] (0.027) / 0.024 (0.024) / 0.025 (0.022) / 0.021 (0.020).
- even share of the per-particle variance: 0.46 [0.40, 0.51] (0.47) / 0.44 (0.45) / 0.49 (0.52) / 0.24 (0.24).
- |az|/|rad|: 0.70 (0.70) / 0.72 (0.73) / 0.81 (0.81) / 0.72 (0.71).
- corr(|d_rad|, |d_az|): 0.57 / 0.61 / 0.57 / 0.45.
- held-out R² of M2, outward: 0.028 [0.021, 0.036] / 0.039 / 0.022 / 0.082 (0.019 / 0.035 / 0.018 / 0.078).
- held-out R² of M2, odd part: 0.041 / 0.047 / 0.038 / 0.071 (0.030 / 0.040 / 0.031 / 0.068).
- signed azimuthal mean: 0.000 in all four (p 0.58–0.76).

**Control 3, pT vs M2.** pt_only (3 parameters) minus M2 held-out R²:
- outward target: −0.002 [−0.008, 0.004] / −0.001 / −0.005 / −0.006;
- odd target: −0.009 to −0.018.

Binned-mean R² is 0.98–1.00 for pt_only and 0.93–0.95 for M2.

The within-z-decile slope on x, relative to the M2-only slope:

| target | ParT_full | ParT_kin | PN_full | PN_kin |
|---|---|---|---|---|
| outward | 0.65 [0.50, 0.80] | 0.60 [0.47, 0.72] | 0.78 [0.62, 0.93] | 0.63 [0.55, 0.70] |
| odd part | 0.72 [0.61, 0.82] | 0.69 [0.59, 0.78] | 0.74 [0.65, 0.82] | 0.74 [0.66, 0.81] |

- H0a (pT only, ratio 0): p < 1e-17 in all eight tests.
- H0b (∝ M2, ratio 1): p = 2.6e-6, 7.8e-10, 4.6e-3 and 7.3e-22 for the outward target, and 1.2e-7 to 1.0e-12 for the odd target.

Both nulls are rejected after Holm correction over the 8 tests per null. Both pure readings are excluded. The response depends on r at fixed pT share, with 0.6–0.8 of the M2 slope.

**Control 1, routes** (outward 0.01 move):
- Route additivity R² (full ≈ part + pair): 0.985 / 0.995 / 0.984 / 0.984.
- ParT_full / ParT_kin:
  - The part route carries 0.75 [0.71, 0.80] / 0.80 of the mean response, but only 0.13 / 0.13 of the mean square.
  - The pair route carries 0.23 / 0.20 of the mean response and 0.80 / 0.78 of the mean square.
  - Share of the even-part mean square: part 0.012 / 0.002, pair 0.96 / 0.98.
  - Even share within each route: part 0.04 / 0.01, pair 0.56 / 0.55.
  - |az|/|rad| within each route: part 0.22 / 0.15, pair 0.93 / 0.94.
  - Held-out R² of M2 on the part route alone: 0.089 [0.068, 0.110] / 0.137. pt_only gives 0.069 / 0.114.
- PN_full / PN_kin: the part route (features with the kNN coordinates unedited) carries 0.99 of the mean, 0.99 / 0.98 of the mean square and 0.99 / 0.97 of the even part. The kNN-coordinate route carries 0.01–0.04.

**Control 2, step size.** The even/odd variance ratio at 0.003 relative to 0.01 is 0.375 [0.25, 0.50] / 1.72 [1.10, 2.46] / 0.43 / 0.69. The smooth prediction is 0.09, with p = 2e-5, 3e-6, 7e-15 and 5e-69.
- The ParT part route alone gives 0.12 / 0.14, close to 0.09. Its odd slope is 0.304 / 0.301, against 0.3 for a linear response.
- The ParT pair route gives 0.32 / 1.36.
- The mean |d| scales by 0.30–0.37 when the step goes from 0.01 to 0.003. The even component is therefore close to first order in |step|, which is a kink-like response.

## Figure
- `results/part/p27_particle_kernel.png` (P27): the panels match the JSON. Panel (b) has a log x axis, so a linear d(x) looks curved. Panel (a)'s y range up to 1 hides the differences between forms, all of which are below 0.12 for trained models.
- `results/part/p27_verify.png` (this run) has three panels:
  - (a) even share at the two step sizes, with the smooth prediction;
  - (b) route shares;
  - (c) per-particle forms on the fresh rows.

## Issues with P27 as stated
1. The statement "Respond … in proportion to that particle's contribution to M2" is not supported. A pT-only form fits the binned means as well. The within-z test excludes both pure pT dependence and proportionality to dM2/M2.
2. The open component in P27's conclusion is now resolved for ParT. It is carried by the pair inputs (control 1), and it is not a smooth second-order term (control 2). For ParticleNet it sits in the per-particle feature path. Those EdgeConv features include neighbour differences, so the neighbour hypothesis is not separated there.
3. The comparison of rank windows and thresholds with M2 (ΔR² ≤ 0.012) is made where every form explains under 10 % of the variance. For ParT, most of that variance is the non-smooth pair-route term. This comparison is weak evidence on A8. A sharper test would fit the forms on the ParT part route, where M2 reaches 0.09–0.14. This run did not fit rank windows there.
4. Pair additivity, calibration and the untrained/permuted controls were not re-run.

## Explainer entry (for site/explain.py)
- **Question.** When one particle of a W/Z jet is moved slightly away from the jet axis, what sets how much ParT's or ParticleNet's W-vs-Z score changes?
- **Background.** P27 found that the binned mean response follows $x_i=\Delta M_2/M_2$ with $M_2=\sum_j p_{T,j} r_j^2$, but that per particle $R^2<0.1$. About half of the variance was even under $r\to r\pm\delta$ and also appeared under a fixed-$r$ rotation.
- **Method.**
  - Fresh rows: 2,000 jets.
  - Moves of $\delta=\pm0.01$ and $\pm0.003$ rad, plus a rotation.
  - Each edit was run with only the per-particle features edited, or only the pair inputs edited.
  - $d=\Delta(\text{logit}_Z-\text{logit}_W)$, with even part $(d_++d_-)/2$ and odd part $(d_+-d_-)/2$.
  - For a smooth $d(\delta)=a\delta+b\delta^2$, the ratio $\mathrm{Var}_{\rm even}/\mathrm{Var}_{\rm odd}$ scales by $(0.003/0.01)^2=0.09$.
  - The regression $d=\sum_q c_q\,1[z\in q]+\beta_w x$ tests whether $r$ matters at fixed pT share $z$.
- **Result.**
  - In ParT, editing only the pair inputs reproduces 96–98 % of the even mean square and nearly all of the rotation response. Editing only the per-particle features gives 75–80 % of the mean response with an even share of 0.01–0.04.
  - The even/odd ratio changes by 0.38–1.72 instead of 0.09.
  - $\beta_w/\beta_{M_2}=0.6$–$0.8$.
  - Figure: `p27_verify.png`.
- **Checks.**
  - Unedited rows are identical across the three route runs.
  - Route additivity is $R^2\ge0.984$.
  - The ParT per-particle route alone scales as a smooth linear response (odd slope 0.30 for a 0.3 step ratio).
- **Why it matters.** The mean "axis" response of these taggers flows through the per-particle angular features and depends on both $p_T$ share and $r$, but it is not exactly $\propto\Delta M_2$. The large particle-to-particle scatter in ParT comes from a non-smooth pair-feature term. This separates the per-particle axis reading (C3/C12) from the pair attention.
