# E33 verify: do the program's objects recur across JEDI-linear checkpoints?

Verifier for node E33. Script `interp/e33_verify.py` (it imports E33's own functions from `interp/e33_program_recurrence_jedi.py`).
Jobs: `e33-vsmoke` (1,000 jets, 4 models) and `e33-verify` (1 CPU / 2 GiB, wall 2,170 s, exit 0). Results:
`results/jedi/e33_verify.json` (per model: `e33_verify_<model>.json`), figure `results/jedi/e33_verify.png`, log `logs/e33-verify.log`.

## Conclusion

Three of the four reference quantities do show up again in most of the other quantized networks:

- a pT-weighted squared-radius sum;
- a soft-particle / empty-slot count;
- a pT sum.

The fresh-sample rerun gives the same counts as E33. The "hardest particles" label does not hold for the reference network, though. Its
pT-sum unit responds just as strongly to pT added to its softest particles as to pT added to its hardest ones. It therefore measures the
total jet pT, not a sum over the k hardest particles. E33's control missed this because it compared changes of very different size.
Only 3 networks (official n32, 16-feature, retrained HGQ seed 2) have a unit that truly ignores particles outside a hard-particle
window, and the window size differs between them. A second open point is how often any reference-fitted formula would recur by chance.
Random formulas that fit a reference unit also recur in about 3 of 9 other networks. The roles recur in 4–8 of 9, which is not clearly
above that baseline at this decoy count.

## What was checked

1. **Code review** of `e33_program_recurrence_jedi.py`. The script was checked for the wrong stage, the wrong mask or quantizer, padded
   slots counted as particles, leakage between fit and test jets, and edits that change more than intended.
2. **Rerun of E33 steps 3–4 on fresh jets.** E33 used fit `sample(2000, 3301)` and test `sample(2000, 3302)`. This rerun uses fit
   `sample(2000, 4401)` and test `sample(2000, 4402)`, N = 10,000 each. The steps are:
   - the four role fits;
   - Hungarian assignment of roles to units (one role per unit, chosen to maximise the summed fit $R^2$);
   - the 250-rotation null;
   - the one-particle edits with E33's controls (1,000-draw bootstrap; sign-flip p from 10,000 draws).

   The rerun covers all 10 trained models. The pruning (ACDC) and z-window steps were not rerun.
3. **Missing control 1: matched-size control for the "hard" role** (new).
4. **Missing control 2: decoy-formula null for "recurrence"** (new).
5. **Competing formulas** for the assigned units:
   - total pT for the hard unit;
   - a linear-radius alternative for the r2 unit;
   - plain multiplicity for the soft and core units.
6. **Data checks.** In 2,000 test jets per model, particles are sorted by descending pT in 100 % of jets, and zero-padded slots always
   follow the real particles. Rank windows and the "empty slot" terms therefore mean what the script assumes.

### Definitions

- **S unit**: one of the 64 second-round jet quantities, $S_c = \tfrac1{16}\sum_i u_c(\text{particle } i)$.
- **Rank**: slot index after sorting by pT (0 = hardest). **r**: $\sqrt{\Delta\eta^2+\Delta\phi^2}$ to the jet axis.
- **Roles** (as in E33):
  - hard $=a+b\sum_{\text{rank}<k}p_T$;
  - r2 $=a+b\sum p_T r^2+c\sum r^2+d\,N(p_T<2, r>0.1)$;
  - soft $=a+b\sum_{\text{slots}\ge k}(t-p_T)_+$;
  - core $=a+b\sum(t-r)_+$.
- **found_dir**: the role's unit has held-out $R^2\ge0.9$ and an edit slope CI above 0. In addition, the in-window $|\Delta S|$ must
  exceed every control's $|\Delta S|$ (sign-flip p < 0.05).
- **Per-GeV response**: $\beta=\sum \Delta S\,\Delta p_T/\sum\Delta p_T^2$, the change of the unit per GeV of pT added to one particle.
- **Decoy family**: a randomly drawn formula of the same form as the hard role, $\sum_{\text{rank}<k}\mathrm{ReLU}(\cos\theta\,z_{p_T}+\sin\theta\,z_r+b)$:
  - $\theta\sim U(0,2\pi)$, $b\sim N(0,1)$, k = 1..n fitted;
  - $z$ = pT and r standardised over real particles;
  - 30 families with seed 4407, plus three fixed decoys ($\sum_{\text{rank}<k} r$, $\sum_{\text{rank}<k} p_T^2$, $\sum_{\text{rank}\ge k}(r-t)_+$).

## Code review

No bug was found in the forward pass, the masks or the fit/test split. Specifically:

- the forward pass matches `Jedi.run` / `Net.run`;
- the role fits choose parameters on the fit jets and report $R^2$ on the test jets;
- padded slots enter only the soft and core features, where E33 intends them to;
- one-particle edits change one particle, rank kept.

Design issues:

- **Hard-role control is not matched in size** (main issue). The in-window edit multiplies a rank < k particle's pT by 1.1. The control
  multiplies a rank ≥ k particle's pT by 1.1. When k is large (reference k = 55–56, HGQ s0 k = 57–58), control particles are the softest
  ones. The mean added pT is 0.07 GeV for the control against 1.26 GeV in-window. A unit that sums all pT therefore passes this control
  (control ratio 0.06). The sign-flip test compares $|\Delta S|$ without normalising for the size of the edit.
- **No control is possible when k = n.** This applies to official n128 (k = 128) and perm-inv (k = 64). The n128 unit passes the plan
  criterion on the fresh sample (slope 0.996 [0.941, 1.032]) with no control, and it is a total-pT unit ($R^2$ vs total pT 0.967, same
  as best k).
- **The rotation null does not test whether the specific formulas recur.** It tests whether some formula lines up with a single unit
  rather than a random mixture of units. Units after a ReLU-and-sum are expected to be more axis-aligned than random rotations. So
  "far above rotations" does not show that these particular formulas, rather than any formula fitted on the reference, recur. See the
  decoy null below.
- **Role assignment is not stable for some parameters.** Between the E33 and fresh samples:
  - 16-feature hard unit: S56, k = 1 → S20, k = 3;
  - n128 soft: k = 9 → 0;
  - HGQ s0 soft: k = 9 → 0;
  - float s0 hard: S31, k = 1 → S16, k = 3.

## Rerun on fresh jets (E33 steps 3–4)

Reference accuracy is 80.43 % (E33: 80.82 %).

| role | found_dir models (fresh) | E33 | plan criterion (slope CI ∋ 1) | rotation-null expectation | Poisson-binomial p |
|---|---|---|---|---|---|
| hard | n64, n32, 16f, HGQ s0, HGQ s2 (5) | same 5 | n128, HGQ s0, HGQ s2 (3) | 0.09 | 4.0e-9 |
| r2 | all trained except float s0 (9; float s0 $R^2$ 0.899, float s1 0.906) | 9 (float s0 in, s1 out) | 0 | 1.05 | 3.8e-9 |
| soft | n64, n32, perm-inv, 16f, HGQ s0, HGQ s2, float s0 (7) | same 7 | 0 | 0.05 | 4.5e-15 |
| core | 0 (outer-particle control 2.1–8.1× the in-window prediction) | 0 | 0 | 0.10 | 1 |

- Slopes reproduce within their intervals. For example, reference r2 slope 0.715 [0.678, 0.757] (E33 0.695), soft 1.087 [1.065, 1.110]
  (E33 1.066), core 0.877 (E33 0.879).
- The r2 slopes are 0.30–0.73 in every model.
- The E33 numbers in the reading reproduce on fresh jets: counts, slopes, the dense float layer (64 live S units; 0 and 1 of 33 decoys
  fitted) and core failing its control.

## Missing control 1: matched-size control for the hard role (per GeV)

**Edit.** In-window edit as in E33: one random particle of rank < k has its pT × 1.1, capped at the pT of the next harder particle so
that ranks are kept. The matched-step control adds the same ΔpT (in GeV) to a random particle of rank ≥ k, again capped at the next
harder particle's pT. Angles are unchanged in both. We also re-read E33's ×1.1 control per GeV.

**Metric.** Ratio = $\beta_{\rm ctrl}/\beta_{\rm in}$, the unit's response per GeV added outside the window divided by its response per
GeV added inside, on the same jets.

- H0 (no rank window; the unit sums all pT): ratio = 1.
- H1 (window, as the fitted formula states): ratio < 1. The formula predicts 0.
- Paired jet bootstrap (1,000); p = P(ratio_boot ≥ 1).

| model | k | $R^2$ (best k) | $R^2$ (total pT) | ratio, matched step [95 % CI] | p (H0) | ratio, E33's ×1.1 control [CI] |
|---|---|---|---|---|---|---|
| official n64 (ref) | 56 | 0.931 | 0.925 | **1.03 [0.76, 1.34]** | 0.59 | 0.87 [0.62, 1.17] |
| official n32 | 10 | 0.999 | 0.583 | 0.00 [0.00, 0.00] | 0.001 | 0.00 |
| 16-feature n64 | 3 | 0.920 | 0.011 | 0.20 [0.17, 0.24] | 0.001 | 0.30 [0.24, 0.36] |
| retrained HGQ s0 | 57 | 0.985 | 0.983 | **1.16 [0.73, 1.63]** | 0.75 | 1.11 [0.64, 1.68] |
| retrained HGQ s1 | 7 | 0.809 | 0.144 | 0.21 [0.06, 0.40] | 0.001 | 0.25 |
| retrained HGQ s2 | 3 | 0.979 | 0.079 | 0.14 [0.10, 0.17] | 0.001 | 0.25 |
| retrained float s0 | 3 | 0.649 | 0.197 | 0.38 [0.35, 0.41] | 0.001 | 0.47 |
| retrained float s1 | 8 | 0.682 | 0.117 | 0.64 [0.55, 0.74] | 0.001 | 0.73 |
| official n128 / perm-inv | 128 / 64 | 0.967 / 0.935 | 0.967 / 0.935 | no control possible (k = n) | – | – |

**Result.**

- In the reference and HGQ s0, a GeV added to a soft particle outside the fitted window moves the "hard" unit as much as a GeV added
  inside it. H0 (no window) is not rejected, and the total-pT fit is within 0.006 of the best-k fit. Including n128 and perm-inv, these
  4 units are fitted by total jet pT ($\sum_{\rm all} p_T$) as well as by any window.
- Within this role family, a unit with a hard-particle window (ratio < 1, $R^2\ge0.9$) appears in official n32 (k = 10), 16-feature
  (k = 3; k = 1 in E33) and HGQ s2 (k = 3). That is 3 of 10, expected 0.09 under the rotation null (p 5e-5). The window size differs
  between these models.
- The reference S0 is described in E29 as "ΣpT of the 14 hardest via g". On these jets its jet-level fit does not separate a 14- or
  56-particle window from the total.

**Physics meaning.** Jets in this sample cover a narrow range of pT. A unit tracking total jet pT carries jet-energy-scale information,
not information about how pT is shared among the hardest constituents. Only the second would be a substructure variable.

## Missing control 2: decoy-formula null for recurrence

**Question.** E33 compares role recurrence with random rotations of the S space. Here the question is different: for a formula that fits
some reference unit at $R^2\ge0.9$, how often does some unit in another trained network also fit it at $R^2\ge0.9$?

**Procedure.** Same fit/test jets. Per model and family: best live S unit and parameter by fit $R^2$; test $R^2$ reported. This is
$R^2$ only, with no edits. Of the 33 decoys, 12 fit a reference unit at test $R^2\ge0.9$. For each of those 12, we count how many of the
9 other trained networks also reach 0.9.

**Result.**

- Decoys recur in 1–6 of 9 networks, mean 3.4 (share 0.38).
- Roles, best unit, $R^2$ only: hard 6, r2 8, soft 6, core 4.
- Empirical p = (1 + #decoys with a count ≥ the role's) / 13: hard 0.15, r2 0.077 (the smallest value 12 decoys allow), soft 0.15,
  core 0.54.
- Per model, the number of decoys fitted at $R^2\ge0.9$ is 5–20 of 33 in the quantized networks and 0–1 in the float copies.

**Meaning.** In quantized JEDI-linear networks, many second-round units are close to single rank-window sums of simple per-particle
functions. A smooth formula that fits the reference therefore often fits a unit elsewhere. Against this baseline:

- r2 recurs most often (8 of 9), at the edge of what 12 decoys can resolve;
- hard and soft recur as often as the most-recurring decoy;
- core recurs at the decoy median.

The rotation-null expectation (0.05–1.2 networks) understates this baseline. The causal edits add specificity that the decoys, being
$R^2$-only, do not have:

- r2: a radial move of a particle changes the unit 2–240× more than an azimuthal move of the same size;
- soft: a pT step across the fitted edge changes the unit, the same step far above the edge changes it little.

## Competing formulas (assigned units, fresh test jets)

- **r2 role.** The $p_T r^2$ form fits better than the linear-radius alternative $a+b\sum p_T r+c\sum r+d\,N$ in every model:
  0.90–0.98 against 0.82–0.92; for the reference 0.977 against 0.919. Multiplicity alone gives 0.05–0.36. The edits test radial vs
  azimuthal dependence, not $r^2$ vs $r$. The single-particle slopes of 0.30–0.73 mean the per-particle function differs from $p_T r^2$.
- **Soft role.** Multiplicity N alone fits the soft unit at $R^2$ 0.92–0.965 in n64, perm-inv, 16-feature, HGQ s0 and HGQ s2. In
  float s0 it fits at 0.935, the same as the soft formula. In n32 it gives 0.54. These units are mostly a particle / empty-slot count.
  The pT dependence at the edge is causal, but its size varies: slope 1.09 (ref), 1.28 (perm-inv), 0.82 (HGQ s2), 0.64 (16f), 0.45
  (n32), 0.26 (HGQ s0), 0.09 (float s0). The float s0 unit responds to an edge crossing at 9 % of the formula.
- **Hard role.** See the table above: total pT fits as well as any window in 4 of 7 models with $R^2\ge0.9$.

## Figure check

**`e33_recurrence.png`** (E33). The three panels are readable and the axes are in physical units. Issues:

- The left panel marks the reference and HGQ s0 hard cells with "*" and "✓". Per the control above, those units are total-pT units.
- The fitted k (1 to 128) is not shown, so "sum pT of k hardest" hides that the window differs from model to model.
- The grey band in the middle panel (expected 0.5–0.65) has no label.
- The right panel has no null for crossing positions. The report already says this.

**`e33_verify.png`** (this verifier).

- Left: per-GeV ratio of the hard unit with 95 % CI. Reference and HGQ s0 sit at 1; n32, 16f and HGQ s2 sit near 0–0.2.
- Right: recurrence counts of the 12 reference-matched decoys against the four roles.

## Corrected reading

In quantized JEDI-linear networks (official n32, n64, n128, perm-inv, 16-feature; retrained HGQ s0 and s2), two formulas from the
reference recur:

- **Squared-radius quantity.** A single second-round unit fits the pT-weighted squared-radius quantity at held-out $R^2\ge0.9$ (9 of 10
  trained networks). It responds to a radial move of one particle more than to an azimuthal move of the same size.
- **Soft / empty-slot count.** A unit fits the soft / empty-slot count (7 of 10). It is largely multiplicity, with a causal pT edge of
  model-dependent size (slope 0.09–1.28).

Both reproduce on fresh jets (fit seed 4401, test seed 4402). The "hardest-particle pT sum" does not recur as a window:

- In the reference (k = 56), HGQ s0, n128 and perm-inv, the matched unit responds equally per GeV to particles outside the window
  (reference ratio 1.03 [0.76, 1.34]). It is a total-jet-pT unit. E33's control passed it only because the control edits were about
  17× smaller in pT.
- A genuine hard-particle window appears in 3 networks (n32 k = 10, 16-feature k = 1–3, HGQ s2 k = 3), with different k.

Further points:

- The core-sum role fails its control everywhere.
- The float copies have a dense, mostly unnamed second-round layer.
- Against a null of random rank-window formulas fitted on the reference, which recur in 3.4 of 9 other networks on average, the roles'
  $R^2$ recurrence (4–8 of 9) is not significantly higher (p ≥ 0.077 with 12 decoys).
- The jet-level formulas do not predict single-particle edit sizes (r2 slopes 0.30–0.73).
- Mass-window recurrence was not tested by a valid null.

## Issues / limits of this verification

- Pruning (ACDC) and z-window steps were not rerun. The reading's circuit-size and z statements rest on E33 alone. The z statement
  is already a "not established" statement.
- The decoy null is $R^2$-only, and 12 reference-matched decoys limit p to ≥ 1/13. More decoy families (e.g. 200) would resolve
  r2 vs decoys.
- The matched-step control is capped at the next harder particle's pT (rank kept). For k = 56–57 the mean step is 0.09 GeV against
  1.27 GeV in-window, so the per-GeV ratios have wide CIs. A larger matched step that allows re-sorting would sharpen them, at the cost
  of changing rank masks.
- Untrained control: one live S unit, so it does not test whether the roles appear without training. Shuffled-label or early-epoch
  controls (GPU training, already approved in train/) would be the stronger null.
- The `e33-verify` pod's first 15 minutes were slow CephFS reads (process in D state, `ceph_mdsc_wait_request`, node
  k8s-chase-ci-01.calit2.optiputer.net). It recovered, and the node was not added to the exclusion list.

New glossary terms: **per-GeV response (β)**, **matched-step control**, **decoy-formula null**.
