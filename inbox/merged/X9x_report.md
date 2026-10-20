# X9x report: augmentation-range scan and centroid-offset read-out (claims A10r, A10m)

**Conclusion.** We trained PFN, EFN and Deep Sets on jets that were randomly shifted by up to A radians. Each of these networks ignores a rigid shift of the whole jet up to about A. Beyond A, it changes its calls about as much as it does for a real increase in the jet's mass spread. This holds for every range tested (0.04–0.30 rad), with a fitted slope of 0.91–1.00 for the break point against A. So the point where a network starts reading a shifted jet as heavier is set by the shifts it saw in training, not by the architecture. Within that range, none of our read-out tests found a variable that tracks the signed offset of the jet's centre and is used in the calls. The best single direction we found follows an even function of the shift, like the mass spread M2. We have not ruled out an explicit recentring step that uses two directions together.

## Procedure
- **Networks.** We used stored-axis inputs with no recentring, seeds 0 and 1, in three architectures: PFN (Φ 64-64-32), EFN (Φ 100-100-128) and Deep Sets d64 N 64.
  - A = 0 and A = 0.08 reuse the X9 / small-model / X9w checkpoints.
  - The 18 new trainings use A ∈ {0.04, 0.16, 0.30} rad. Tags are -aug04 / -aug16 / -aug30.
  - The augmentation translates every particle of a training jet by |c| ~ U(0, A) in a random direction, redrawn every epoch.
  - PFN / EFN were trained with interp/x9_train.py through train/x9w_cpu_batch.sh, using the same flags as X9w (`--epochs 60 --max_minutes 55`, patience 5). They ran as CPU Jobs x9x-pfn (10 CPU / 22 Gi; measured 8.0 cores, 18.4 GiB) and x9x-efn (16 / 22; measured 12.3 cores, 18.1 GiB).
  - These run on CPU, not GPU, because the energyflow venv is TF-CPU, as in X9w.
  - Deep Sets were trained on the GPU through train/make_jobs.py, new set `x9x` (job train-x9x, 6 runs, 30 epochs, 257 s), then exported with `AI_PY=conda … --export_dsets`.
- **Untrained controls.** Two redraws per architecture: energyflow he_uniform kernels for PFN / EFN, torch default init for Deep Sets.
- **Sample.** E = sample(per_class=2000, seed=3801), N 10,000 (2,000 true W). Hold-out H = sample(2000, 3802) minus 1,990 shared rows, N 8,010.
- **Forward-pass checks.** The numpy forward of every network was checked against the training framework: PFN / EFN max |Δp| ≤ 4.5e-6, Deep Sets max |Δlogit| ≤ 2.1e-5.
- **Official JEDI-linear on E.** 80.66 % [79.87, 81.42].
- **(1) Edit scan.**
  - Edit: every real particle moved by T ∈ {0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.30, 0.40} rad, at 8 azimuths (averaged per jet).
  - Null: a rotation about the axis with mean displacement T (rng 3810+i; it saturates near 0.18 rad for T ≥ 0.24).
  - Matched mass edit dilM2: angles scaled by s = √(1 + T²M0/M2), which gives the same azimuth-mean Δlog M2 as the translation.
  - Metrics: W→Z flow; % calls changed; excess dc(T) = calls(T) − calls(rot) (paired per jet); calibrated Rc = dc / calls(dilM2). All with 1,000-draw jet-bootstrap CIs. We also ran McNemar on W→Z at azimuth 0, T vs rot, with Holm over 90 (network, T) tests per architecture.
- **Deviation from the plan.** W→Z flow is not monotone in T. W jets pass through Z towards top at large offsets (for example, PFN A = 0 flow is 0.82 at T = 0.04 and −0.02 at 0.12), and flow(dilM2) ≤ 0 there. So the calibrated break was moved to calls changed. The pre-registered T*_2sd (first T with W→Z excess > 2 sd) is still reported. We added two break offsets after the smoke run: T*_calls (dc > 2 sd) and T*_25 (Rc reaches 0.25, linearly interpolated).
- **(2) Latent read-out** on the pooled latent (PFN / EFN L, Deep Sets S).
  - Ridge probe of the centroid offset c (2-vector), fitted on E jets translated randomly up to 0.4 rad and tested on H.
  - probe_nat: ridge on natural offsets only.
  - Calibration: probe slope on natural offsets vs on +Δη translations.
  - 1-dim DAS on a signed Δη grid (0, ±0.04 … ±0.40). We used 4,000 fit pairs on E and 4,000 test pairs on H. The intervention is L' = L_b + u u·(L_s − L_b), and the target is the network's call on the base jet at the source offset.
  - Effective IIA has pair-bootstrap CIs. Baselines: 200 random u, the probe c_η direction, and untrained networks. We also report the R² of u·L against t (signed), t², log M2 and log m.
- **Jobs.** x9x-{pfn,efn,ds}-{a,b,c}, x9x-off and x9x-comb, all at 1 CPU / 2 GiB. Each took 8–22 min and exited 0. Peak memory reached 2,047 MiB (the limit) in 6 of 9 parts, most likely file cache from the mmap'd archive; no job was killed.
- **Code.** interp/x9x_augrange.py, interp/x9x_plot.py, train/make_jobs.py (set x9x).
- **Results.** results/jedi/x9x_{pfn,efn,dsets}_{a,b,c}.json, x9x_official.json, x9x.json (combined), x9x_augrange.png.

## Numbers
**Test accuracy (260k), seeds 0 / 1, by A = 0 / 0.04 / 0.08 / 0.16 / 0.30.** The EFN aug runs stopped at 22–38 epochs (patience or the 55-min limit).

| architecture | A = 0 | A = 0.04 | A = 0.08 | A = 0.16 | A = 0.30 |
|---|---|---|---|---|---|
| PFN | 81.2/81.1 | 81.1/81.1 | 80.9/81.3 | 80.6/81.3 | 79.8/80.2 |
| EFN | 74.0/74.4 | 73.5/74.1 | 73.4/73.4 | 72.2/71.6 | 70.3/71.0 |
| Deep Sets | 80.6/80.8 | 80.5/80.6 | 80.0/79.7 | 79.5/79.5 | 78.6/78.6 |

**Calibrated excess Rc (calls changed beyond rotation, in units of the matched dilM2 edit):**
- **A = 0:** 0.85–1.24 at every T in all 6 networks.
- **Last grid point below A** (95 % CI):
  - A = 0.08 at T = 0.06: 0.014–0.043
  - A = 0.16 at T = 0.12: 0.029–0.076
  - A = 0.30 at T = 0.24: 0.083–0.170
  - CIs exclude 0 for A ≥ 0.08, so there is a small response inside the trained range that grows towards its edge.
- **First grid point above A:** A = 0.08 at T = 0.12: 0.50–0.77; A = 0.16 at T = 0.24: 0.85–0.97; A = 0.30 at T = 0.40: 0.70–1.00.

**Break offset T*_25** (Rc reaches 0.25), seeds 0 / 1:

| architecture | A = 0.04 | A = 0.08 | A = 0.16 | A = 0.30 |
|---|---|---|---|---|
| PFN | 0.056/0.059 | 0.095/0.098 | 0.174/0.178 | 0.277/0.307 |
| EFN | 0.051/0.056 | 0.094/0.092 | 0.176/0.172 | 0.295/0.308 |
| Deep Sets | 0.050/0.050 | 0.091/0.092 | 0.175/0.175 | 0.314/0.319 |

- A = 0 networks give 0.020, the first grid point (floor).
- Slope of T* on A (intercept in brackets): PFN 0.91 (0.023), EFN 0.95 (0.018), Deep Sets 1.00 (0.014).
- Pearson r 0.997 / 0.999 / 0.999. Permutation of A over the 10 networks, one-sided p = 2e-5 per architecture (floor of 50,000 draws); pooled, stratified mean r 0.998, p = 5e-5 (floor).

**Pre-registered T*_2sd** (W→Z excess > 2 sd) does not track A:
- r = 0.34 / 0.10 / −0.02, p = 0.19 / 0.41 / 0.48; pooled p = 0.24.
- It fires at 0.02–0.12 in all aug networks because of the small inside-range response, and W→Z is not monotone.
- **T*_calls:** r = 0.51 / 0.93 / 0.80 (p = 0.07 / 8e-4 / 4e-3), slopes 0.10–0.17.

**McNemar T vs rot (Holm < 0.05):**
- inside the range (T ≤ A): 20/40, 23/40, 26/40
- outside the range: 47/50 in each architecture

**Growth beyond A:** the grid is too coarse to fit (T − A)^k. Rc goes from < 0.17 to ≥ 0.5 within one grid step; a fit was possible in only 4 of 24 networks (k = 0.14–1.42).

**Latent read-out:**
- **probe_tr R² (hold-out):**
  - trained PFN 0.94–0.98, untrained PFN 0.95
  - EFN 1.00 trained and untrained
  - Deep Sets 0.97–0.98, untrained 0.97
- **probe_nat R²:**
  - PFN: A > 0 0.55–0.79, A = 0 0.20, untrained 0.09
  - Deep Sets: A > 0 0.55–0.74, A = 0 0.16–0.36, untrained 0.17–0.31
  - EFN: 0.86–1.00 everywhere
- **Calibration** (probe slope on translations / on natural offsets): 0.98–1.19.
- **DAS effective IIA:**
  - trained networks 0.28–0.52, vs random-direction 95th percentile 0.06–0.31
  - p = 0.005 (floor 1/201) in 27/30, the exceptions being 0.010, 0.020 and 0.114, all A = 0.30
  - untrained redraws 0.26–0.93, p = 0.005 in 5/6
  - inside-range pairs: IIA 0.19–0.40 on 2–571 pairs
- **What the DAS direction u·L follows (R²):** in 28/30 trained networks, t² 0.65–0.98 and log M2 0.34–0.78, with signed t ≤ 0.27. The exceptions are Deep Sets A = 0.04 s0 (t 0.85) and A = 0 s1 (t 0.42 = t² 0.45). Invariant mass log m ≤ 0.03 everywhere.
- **Probe c_η direction, IIA vs random 95th percentile:** above it in PFN 2/8, EFN 6/8 and Deep Sets 0/8 aug networks (values 0.00–0.31, against DAS 0.28–0.52).

## Reading (scoped to what was measured)
- **H0 vs H1 for the break point.**
  - The pre-registered T*_2sd does not separate H0 (T* independent of A) from H1, because it fires on the small inside-range response.
  - The calibrated break T*_25 follows A with slope 0.91–1.00 and intercept ≤ 0.023 rad in all three architectures (p at the permutation floor).
  - So a break at the same T for every A (an input-side limit) is excluded.
  - Below A the response is not exactly zero: 1–17 % of the matched mass response at the last grid point below A, growing with T.
- **H_recentre vs H_tolerant.**
  - **Signed offset readable but not used.** The signed offset is linearly readable from the pooled latent of every network, untrained ones included. So readability alone says nothing about use. Swapping along the probe's offset direction moves calls no more than random directions in Deep Sets and in most PFNs.
  - **Best direction is even.** The best 1-dim direction transfers 0.28–0.52 of the call changes, a level untrained networks also reach. It follows t² / log M2 rather than signed t in 28/30 trained networks.
  - **Fit with the hypotheses.** These results do not support an explicit signed-offset variable that is subtracted (H_recentre as stated). They fit a response that is even in the offset and suppressed up to A (H_tolerant).
- **Still open.**
  - Recentring through a 2-dim offset subspace combined non-linearly was not tested.
  - The (T − A)^k shape cannot be resolved on this grid.
- **Other observation.** Aug-trained PFN and Deep Sets represent natural centroid offsets more linearly than A = 0 or untrained ones (probe_nat 0.55–0.79 vs 0.09–0.36). This is a correlation, not a causal result.
- **Accuracy cost** at A = 0.30: about 1.2 points (PFN), 2.1 points (Deep Sets) and 3.6 points (EFN, partly from early stopping).

## Issues
- The plan's GPU route for PFN / EFN was replaced by CPU Jobs, the same route X9w used.
- The EFN aug runs are less trained than A = 0 (22–38 epochs).
- JEDI-linear was not in the scan, as planned.
- Peak memory reached the 2 GiB limit in 6 parts, with no failures.
- The pre-registered metric was replaced after the smoke run (reasons above). Both are reported.

## Explainer entry (for site/explain.py; Paper A tab)
- **Question.** If a set network is trained on jets randomly shifted by up to A radians, up to what shift does it ignore shifts, and does it do so by measuring and subtracting the jet's centre?
- **Background.** A rigid shift $T$ of every particle leaves the invariant mass unchanged. It does raise the stored-axis spread $M_2=\sum_i p_{T,i} r_i^2$ to $M_2 + T^2 M_0$. Networks trained on unshifted jets read this as a heavier jet: W jets become Z. In X9w, training with shifts up to 0.08 rad removed the response at 0.06 rad but not at 0.12 rad.
- **Method.**
  - Train PFN, EFN and Deep Sets with $A\in\{0, 0.04, 0.08, 0.16, 0.30\}$ rad (2 seeds each).
  - Shift test jets by $T$ at 8 azimuths, and count calls changed beyond a rotation of equal displacement. Express this in units of a matched dilation of the angles by $s=\sqrt{1+T^2M_0/M_2}$, which changes $M_2$ by the same amount without moving the centre: $R_c(T)=\frac{\Delta_{\rm calls}(T)-\Delta_{\rm calls}(\mathrm{rot})}{\Delta_{\rm calls}(\mathrm{dil}M_2)}$.
  - Define the break offset $T^*$ as where $R_c$ reaches 0.25.
  - Read out the centroid $\vec c=\sum p_T\,(\Delta\eta,\Delta\phi)/\sum p_T$ with linear probes. Use 1-dim distributed alignment search: swap $u\,u^\top(L_s-L_b)$ between jets.
- **Result** (figure x9x_augrange.png). $T^*\approx A$ for all three architectures (slope 0.91–1.00, intercept ≤ 0.023 rad, permutation $p=2\times10^{-5}$). Below $A$, $R_c$ is 0.01–0.17. Above $A$, it rises to 0.5–1.0 within one grid step.
- **Checks.**
  - Rotation null; matched dilation; untrained networks; Holm-corrected McNemar.
  - The pre-registered W→Z-based break does not track $A$; this is explained by the non-monotone W→Z flow and the small response inside the range.
  - The offset is linearly readable in untrained networks too. The swapped direction that moves calls follows $T^2$ / $\log M_2$, not the signed offset.
- **Why it matters.** The size of shift that set networks tolerate is learned from the training data, not fixed by the architecture. Inside the tolerated range, the measured variables are even in the offset (like $M_2$), not a signed offset estimate that is subtracted.
- **Conclusion.** The same plain-terms paragraph as at the top of this report.
