# E21 verification (skeptic, 2026-10-17)

Node: E21. S-level patches along the iso-product and iso-sum curves of the M0 and M2 carriers.
Script reviewed: `interp/e21_isoproduct_jedi.py`. Figure script: `interp/e21_plot.py`.
Verification script: `interp/e21_verify.py`. Outputs: `results/jedi/e21_verify_geometry.json` and this report.

## Status of the rerun
- The job `e21-verify` (1 CPU / 2 GiB; `python -m interp.e21_verify`, `sample(per_class=2000, seed=91)`) was submitted after `./sync.sh`.
- After `sample N 10000` (26 s), the Python process sat in kernel D state on node `rci-tide-gpu-06.sdsu.edu`:
  - first in `ceph_mdsc_wait_request`, then in `folio_wait_bit_common`;
  - pod CPU was 5 m-cores after about 9 minutes.
- This is the CephFS stall listed in `job.sh`'s node exclusions. That node is not on the list yet.
- I tried to delete the job and resubmit it with this node excluded. The session's permission classifier denied the delete, the resubmit, and then all `kubectl` calls, including read-only `get` and log `tail`.
- **No numbers from a fresh seed are in this report.** The job may still be running or stalled. Either way it needs a human to delete or rerun it. To do that:
  1. Add `rci-tide-gpu-06.sdsu.edu` to the node exclusion list in `job.sh`.
  2. Run `./job.sh e21-verify 'python -m interp.e21_verify'`.
- Everything below comes from code review and from the E21 run 3 numbers, plus hand computation. No new network evaluation was done.

## Code review of e21_isoproduct_jedi.py
- **Stage, quantizer and masks.** The S patch uses `run_jetwise` on `S`, which is after the `qs1` pooling quantizer. The tail (d4 → d7) reuses each layer's `iq` and fused kernels. The identity check gives max |Δ| = 0 at h4, h5, z and the logits, both on the base S and on a test patch. Padded slots are not involved, because only S columns are edited.
- **Edits.** Only S0, S50 and S29 are changed in the arms. Every other S unit keeps its value, as stated.
- **The iso-product edit holds the product exactly**, using $m_0 e^{\delta}$, $m_k e^{-\delta}$. The iso-sum edit is exact.
- **Not checked: the d4 input quantizer (`d4.iq`) re-quantizes the patched S.** If its step is coarse relative to the patch (ΔS0 ≈ 0.8 at ε = 0.15), the iso-product patch is not iso-product after quantization. That would add a nonzero R even for a unit that depends only on the product. `e21_verify.py` logs the `d4.iq` bits for S0, S50 and S29, but that output is pending.
- **The random-pair control (a) is compared on different jets.**
  - Each draw is scored on its own no-clip subset: 341 to 5,560 jets (listed in the JSON).
  - The network's arm A value is scored on 6,355 jets.
  - The comparison "network inside the 5–95 % range of draws" therefore mixes jet populations. `e21_verify.py` also scores the network on each draw's jets.
- **Untrained control (c).** The fallback took the two S units with the largest sd. It returned (58, 0) with sd (0.77, 0.0), so `clip_frac` = 1.0. The second-largest sd being 0 means that at most one S unit varies across these jets in the redrawn network (seed 0). An S-level pair control on this untrained network is therefore not possible. Two alternatives: other redraw seeds, or a pair at an earlier stage.
- **Holm correction and the paired z-test** are as described. With N ≈ 6,400, almost every unit rejects R = 1, including 41 of 62 random ReLU units. H0: R = 1 is therefore not an informative null here. The report says so.
- **Statement in the reading.** "R is 0.28–0.79, and z10 in arm A is 0.96" leaves out z10 in arm C: R = 1.115 [1.096, 1.130], which is significantly above 1.

## Figure (e21_isoproduct.png)
- **Left panel.** The axes, the random-pair band, the arms A and B, and the R < 0.3 line are readable.
- **Right panel.** R by θ quartile, with dashed lines for the additive fit. The markers in the legend separate measured values from the fit.
- The figure matches the JSON. The title states the arm A result scoped to the control. No change is needed, except to note that the band is computed on different jets (see above).

## Geometry of the arm A test (hand computation; results/jedi/e21_verify_geometry.json)
Notation:
- $t$ is the per-jet tangent angle, $\tan t = (S_{50}/\mathrm{sd}_{50}) / (S_0/\mathrm{sd}_0)$.
- The iso-product step in $(z_{S0}, z_{S50})$ is $(\cos t, -\sin t)$.
- The iso-sum step is $(1, -1)/\sqrt2$.

Both edits raise S0 and lower S50 for ε > 0. They are $45° - t$ apart, which is about 32° at the median $t \approx 12.5°$.

For a unit that is linear in the patch, with $\rho = (\partial U/\partial z_0)/(\partial U/\partial z_{50})$:
$$R = \sqrt2\,\frac{|\rho\cos t - \sin t|}{|\rho - 1|}$$
- **Product rule** ($\rho = \tan t$): $R = 0$.
- **Unit reading S50 only**: $R = \sqrt2 \sin t$. This is 0.31 at $t = 12.5°$, and 0.17, 0.27, 0.39, 0.96 at the bin medians 7°, 11°, 16°, 43°.
- **Unit reading S0 only**: $R = \sqrt2 \cos t$, which is 1.38 at $t = 12.5°$.

Random-pair 5–95 % ranges from E21:
- margin: 0.38–1.32
- z0: 0.41–1.38
- z7: 0.43–1.37

These match the single-carrier bounds [0.31, 1.38]. So the random-pair band mostly reflects where a unit's sensitivity falls between its two carriers. It is a weak test of whether a unit is specific to the M0/M2 carriers. The margin's R of 0.44 sits near the S50-dominated end.

Effective weight ratios implied by the measured margin R per θ bin (0.47, 0.39, 0.44, 1.03), treating each bin as one linear unit:

| θ bin median | product rule $\rho = \tan t$ | implied $\rho$, same-sign root | implied $\rho$, opposite-sign root |
|---|---|---|---|
| 7° | 0.12 | 0.34 | −0.32 |
| 11° | 0.19 | 0.37 | −0.12 |
| 16° | 0.29 | 0.46 | −0.06 |
| 43° | 0.93 | 0.97 | – |

On the same-sign root, the implied ratio changes less across the three lower bins (0.34–0.46) than the product rule predicts (0.12–0.29). Per-jet gating makes these effective values only. The direct per-jet test is the gradient-ratio test in `e21_verify.py` (pending).

## Assessment of the reading, claim by claim
1. **"H0 (R = 1) rejected for most read-out targets."** Reproduced from the JSON. Add z10 in arm C: R = 1.115, which is above 1.
2. **"Arm A's R < 1 is not specific to the M0/M2 carriers."** Supported as a statement about the control. However, the control is scored on different jets, and its range is close to the geometric bounds for single-carrier units. So the arm A test cannot separate a product from additive dependence that is dominated by S50. The wording "not specific" stands. The test's power is low in both directions.
3. **"The angle test excludes an exact function of S0·S50 (or of the affine M0·M2) at the read-out."** Supported by the run 3 numbers, with the margin's R at 0.39–0.47 in the three lower bins and tight confidence intervals.
   - **Open caveat:** rounding of the patched S by the `d4` input quantizer is not quantified.
   - **Scope:** it excludes an exact function only. A product-like dependence with jet-to-jet scatter in gradient direction also gives R > 0, because R sums absolute values. That case is not excluded.
4. **"Per-unit R does not change between S and h5 and matches random ReLU layers on S."** Reproduced from the JSON: arm A medians at h4, h5, z are +0.08, −0.02, −0.09, against random-pair draw medians of −0.55 to +0.14.
5. **"The test does not locate a product stage."** Supported as stated.
6. **"Arm B is the remaining lead; it needs a three-unit control."** Correct. The controls are coded but not run:
   - (S0, S50, random k), which tests whether S29 is specific;
   - random triples with arm B displacements.

## Corrected reading (proposed)
The S-level patch test does not locate a stage where a product of S0 and S50 forms.

**Arm A (S0, S50).**
- The iso-product and iso-sum steps are about 32° apart, and the iso-product step lies within about 12° of the S0 axis.
- So R mainly measures how much a read-out depends on S50 relative to S0. For units that read one carrier, R lies between $\sqrt2\sin t \approx 0.31$ and $\sqrt2\cos t \approx 1.38$.
- The network's read-out R values (margin 0.44, z7 0.40) and the random-pair band (0.38–1.38) both fall in this range.

**Angle test.** It excludes a read-out that is an exact function of S0·S50 or of the affine M0·M2. This holds pending a check of the d4 input quantizer's rounding of the patch. It does not exclude product-like dependence with per-jet scatter.

**Additive rule.** A single fixed-weight additive rule fits the iso-product responses with held-out R² ≤ 0.11.

**Arm B (S0, S50 + S29).** It gives lower R (margin 0.28) and a larger call difference (24 % vs 7 %), but it has no control with its own geometry.

**Untrained control.** It produced no data: the redrawn network has at most one varying S unit.

## Issues
1. No fresh-seed numbers. The cluster job stalled on a CephFS node, and cluster access was then denied in this session.
2. The random-pair control is scored on different jet subsets from the network (341–5,560 jets against 6,355).
3. Untrained control: in the redrawn network (seed 0) at most one S unit varies (sd of the second-ranked unit is 0), so no S-level pair exists. `e21_verify.py`'s untrained section will hit the same limit unless other redraw seeds are used.
4. z10 in arm C (R = 1.115 > 1) is missing from the reading.
5. The arm A design has low power. Recommended primary test: the per-jet gradient-ratio slope. Regress $\log(g_{50}/g_0)$ on $\log(S_0/S_{50})$:
   - product: slope 1;
   - fixed additive rule: slope 0;
   - controls: random ReLU layers on S, random S-unit pairs, the untrained network.
   This is implemented in `interp/e21_verify.py`.
6. The rounding of the patched S by the `d4` input quantizer is unquantified.
