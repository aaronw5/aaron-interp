# X9x verification: augmentation-range scan, break offset and offset read-out

**Conclusion.** On a new jet sample, set networks (PFN, EFN, Deep Sets) trained on jets shifted by up to A radians still change few calls for whole-jet shifts below A, and start changing many calls once the shift passes A. The break point moves with A with slope 0.9–1.0 in all three architectures. Past A, a shift moves calls towards heavier classes in every network. However, the individual new calls agree with those of the matched mass-spread edit (angles stretched until the jet's second angular moment about the axis grows by the same amount) only for A ≤ 0.08 rad; for A ≥ 0.16 the agreement is at the level expected by chance. So "past A, the network responds as to a larger mass spread" holds for the number of calls changed and for the heavier direction, not call by call. Within the trained range, removing the two latent directions that read out the jet's centre offset undoes most of the network's tolerance in Deep Sets and EFN (more than norm-matched random directions do), but not in PFN. A recentring-like read-out is therefore used in two of the three architectures; the earlier one-direction (DAS) test could not have detected it.

## What was checked
1. Code review of interp/x9x_augrange.py, interp/x9x_plot.py, the X9x set in train/make_jobs.py, and the training augmentation (interp/x9_train.py `augment_np`, train/train_small.py `augment_translate`).
2. Fine-grid replication of Rc and T*_25 on a new sample (job x9x-verify, interp/x9x_verify.py part 1).
3. Missing control for H_recentre: removing the 2-dim linear offset read-out from the latent (x9x-verify, part 2; x9x-verify2, norm-matched null).
4. Missing control for "beyond A responds as to a real M2 increase": direction and per-jet identity of the call changes (new job x9x-verify3, interp/x9x_verify3.py).

Parts 2–3 were run earlier under this verification node (jobs x9x-verify and x9x-verify2, logs on the volume, results/jedi/x9x_verify.json and x9x_verify2.json). This report re-reads and summarises them, and adds part 4.

## 1. Code review
- Training augmentation (PFN/EFN numpy, Deep Sets torch): a rigid translation of the real particles only, with |c| ~ U(0, A) and a uniform direction, redrawn per jet and epoch. This matches the edit used in the scan (`translate`: real particles only). No padding slot is moved.
- Pooled latent (`SetNet.latent`): φ is evaluated on real particles only and summed with weight 1 (PFN mask, Deep Sets) or z = pT/ΣpT (EFN). PFN padding masks match energyflow (rows equal to 0 after standardisation). Forward checks against TF/torch: max |Δp| ≤ 4.5e-6 and max |Δlogit| ≤ 2.1e-5 (worker log).
- Split: hold-out H = sample(2000, 3802) minus the 1,990 rows shared with E (`sample_rows`). The probes are fitted on E and tested on H. The DAS is fitted on E pairs and tested on H pairs. No leakage was found.
- dilM2 scales the angles of real particles only. s = √(1 + T²M0/M2) equals the azimuth-mean ΔM2 of a translation, because the cross term 2t·ΣpT r averages to 0 over azimuth. Correct.
- Rotation null: `rotate_matched` rotates the real particles about the stored axis. Its mean displacement saturates at the half turn for large T (stated by the worker).
- Issues found (none changes the headline numbers):
  - (a) The A = 0 networks have T*_25 = 0.02, the first grid point (left-censored). This adds two floor points to each slope fit. The fit over augmented networks only gives slopes of 0.89 (PFN), 0.89 (EFN) and 1.04 (Deep Sets) (verify part 1).
  - (b) Grid placement. A sits exactly on a grid point and the next point is far away (0.30 → 0.40). Linear interpolation of a step-like Rc then puts T*_25 just above A by construction. The fine grid (part 2 below) removes this.
  - (c) The 1-dim DAS is fitted on all signed pairs, and its effective pairs are those whose call changes. Inside the trained range almost no pair changes its call: inside effective pairs are 0–571 per network, and 0–8 for A = 0.04. The fitted direction is therefore set by the beyond-A response, which is even in the offset. The DAS cannot detect a variable whose job is to cancel the shift (a recentring read-out), because a cancelled shift produces no call change to transfer. The removal test in part 3 is the matched test for that hypothesis.
  - (d) In verify part 2, `probe_R2_deta_on_V` is strongly negative (−21 to −88) for PFN and Deep Sets. This is an artefact of computing R² on jets that all have the same +T_in shift (small target variance). The probe response slope on +Δη translations is 0.89–1.07 for all 18 networks (verify2), so the probe reads translations correctly.
  - (e) Figure results/jedi/x9x_augrange.png: the legend prints r = 1.00 (rounded; the values are 0.997–0.999). The A = 0 points in the right panel sit at the grid floor. The curves are seed means with bands. No mismatch with the JSON was found.

## 2. Fine-grid replication (x9x-verify)
- Sample: sample(1600, 4801), N 8,000 (no overlap of seeds with X9x).
- All 30 trained networks.
- Offsets T ∈ {A/2, A, A+0.01, A+0.02, A+0.04, A+0.08}; for A = 0, {0.02, 0.04}.
- Edits: translation at 4 azimuths, rotation null, dilM2.
- Metric: Rc with a 1,000-draw jet bootstrap.

| | PFN | EFN | Deep Sets |
|---|---|---|---|
| Rc at T = A/2 (A > 0) | −0.003 – 0.066 | 0.018 – 0.098 | 0.006 – 0.125 |
| Rc at T = A | 0.030 – 0.296 | 0.073 – 0.299 | 0.051 – 0.152 |
| Rc at T = A + 0.04 | 0.27 – 0.71 | 0.38 – 0.80 | 0.40 – 0.56 |
| Rc at T = A + 0.08 | 0.85 – 0.98 | 0.82 – 1.02 | 0.60 – 1.07 |
| T*_25 − A (8 aug networks) | −0.030 … +0.025 | −0.035 … +0.022 | +0.011 … +0.022 |
| Pearson r (10 networks), permutation p | 0.993, 1e-4 (floor) | 0.993, 1e-4 | 0.9995, 1e-4 |
| slope (intercept), all 10 | 0.91 (0.025) | 0.90 (0.023) | 1.02 (0.015) |
| slope, aug only | 0.89 | 0.89 | 1.04 |

The two negative offsets T*_25 − A are PFN and EFN at A = 0.30, seed 0. Their Rc at T = A is already 0.30 [0.28, 0.31], so T* ≈ A − 0.03 there.

A = 0 networks: Rc is 0.86–1.33 at T = 0.02 and 0.90–1.15 at T = 0.04.

The original numbers replicate on the fresh sample, with one exception. The stated "T* = A + 0.01–0.02 rad" holds for A ≤ 0.16. For A = 0.30, T* − A ranges from −0.035 to +0.022.

## 3. 2-dim offset read-out removal (x9x-verify, x9x-verify2): H_recentre control
- Networks: A ∈ {0, 0.16, 0.30}, seeds 0 and 1, all three architectures (18 networks).
- Split of sample(1600, 4801): fit F (4,000) and test V (4,000), rng 4820.
- Ridge probe from the latent to the pT-weighted centroid c, fitted on F jets translated by |t| ~ U(0, 0.4).
- Q = the orthonormalised 2-dim span of the raw-coordinate read-out vectors for c_η and c_φ.
- Edit on V: translation by T_in = 0.75 A (0.12 for A = 0) at 4 azimuths. dL = L(translated) − L(unedited).
- Interventions on the pooled latent:
  - full: L0 + dL
  - remove: L0 + dL − QQᵀdL
  - only: L0 + QQᵀdL
- Null: 200 random 2-dim subspaces, with the removed component rescaled per jet to the norm of the Q component (norm-matched; rng 4831).
- Metric: fraction of jets whose call differs from the unedited call.
- H0 (no linear 2-dim offset read-out used for tolerance): remove − full equals the random-subspace value.
- H1 (recentring through that read-out): remove − full is above the random 95th percentile.

| network (A, seed) | calls: full → remove | remove − full [95 % CI] | norm-matched random mean (q95) | p (200 draws) | heavier / lighter after removal |
|---|---|---|---|---|---|
| Deep Sets 0.16 s0 | 0.069 → 0.617 | +0.548 [0.534, 0.561] | 0.273 (0.421) | 0.005 | 0.53 / 0.08 |
| Deep Sets 0.16 s1 | 0.071 → 0.462 | +0.392 [0.381, 0.404] | 0.222 (0.333) | 0.015 | 0.42 / 0.03 |
| Deep Sets 0.30 s0 | 0.115 → 0.586 | +0.471 [0.459, 0.484] | 0.291 (0.424) | 0.020 | 0.54 / 0.04 |
| Deep Sets 0.30 s1 | 0.112 → 0.677 | +0.566 [0.551, 0.579] | 0.350 (0.473) | 0.005 | 0.65 / 0.03 |
| EFN 0.16 s0 | 0.110 → 0.762 | +0.652 [0.638, 0.666] | 0.447 (0.538) | 0.005 | 0.56 / 0.18 |
| EFN 0.16 s1 | 0.119 → 0.765 | +0.646 [0.632, 0.657] | 0.404 (0.521) | 0.005 | 0.59 / 0.15 |
| EFN 0.30 s0 | 0.198 → 0.821 | +0.623 [0.610, 0.636] | 0.492 (0.570) | 0.010 | 0.35 / 0.37 |
| EFN 0.30 s1 | 0.152 → 0.670 | +0.519 [0.503, 0.532] | 0.550 (0.616) | 0.76 | 0.03 / 0.53 |
| PFN 0.16 s0 | 0.091 → 0.635 | +0.545 [0.534, 0.556] | 0.440 (0.551) | 0.070 | 0.38 / 0.20 |
| PFN 0.16 s1 | 0.075 → 0.668 | +0.593 [0.580, 0.605] | 0.457 (0.566) | 0.020 | 0.53 / 0.11 |
| PFN 0.30 s0 | 0.133 → 0.665 | +0.532 [0.521, 0.542] | 0.526 (0.601) | 0.43 | 0.38 / 0.26 |
| PFN 0.30 s1 | 0.114 → 0.619 | +0.505 [0.491, 0.517] | 0.536 (0.622) | 0.72 | 0.08 / 0.47 |

- A = 0 networks (the translation already changes 70–90 % of calls): remove − full is −0.017 to +0.028, inside the random range (p 0.12–0.80).
- The removal p-values are at the permutation floor (1/201). With Holm over the 12 augmented networks, no cell is below 0.05 (the smallest is 12 × 0.005 = 0.06). With Benjamini–Hochberg at q 0.05, 8 of 12 cells pass: Deep Sets 4/4, EFN 3/4, PFN 1/4.
- "Only" (moving only the Q component) is not above random in any network (p 0.06–0.99).
- Norm-matched random removals also raise calls to 22–55 %. So the cancellation is spread over many latent directions; the offset read-out is the largest single 2-dim contributor only in Deep Sets and EFN.

Interpretation, limited to what was measured: within the trained range, the latent change caused by a shift contains a component along the linear centroid read-out. In Deep Sets (4/4) and EFN (3/4), that component cancels more of the shift's effect on the calls than a norm-matched random 2-dim component does. When it is removed, Deep Sets calls move towards heavier classes (0.42–0.65 heavier vs 0.03–0.08 lighter). This is the direction expected if the read-out offsets the growth of the axis moment, which by the parallel-axis relation is M2(axis) = M2(centroid) + M0|c|². PFN shows no such effect (1/4). X9x's statement "a recentring step that uses two directions together is still untested" is superseded for Deep Sets and EFN.

## 4. Direction and identity of the beyond-A response (new: x9x-verify3)
- Procedure: interp/x9x_verify3.py, job x9x-verify3 (1 CPU / 2 GiB, wall 531 s, peak 958 MiB, exit 0).
- Sample: sample(1600, 4901), N 8,000.
- Networks: all 30 trained networks.
- Offsets: T = A/2 (inside) and T = A + 0.04, A + 0.08 (beyond; for A = 0, 0.04 and 0.08). That is 60 beyond cells and 24 inside cells.
- Edits:
  - translation by T at 4 azimuths
  - dilM2 (angles × s)
  - matched contraction (angles × 1/s; same |Δlog M2|, opposite sign)
  - rotation null
- Mass rank: g = q (0) < W (1) < Z (2) < t (3).
- Metrics:
  - Δrank = mean rank change over all jets.
  - Rrank = Δrank(translation) / Δrank(dilM2).
  - Call agreement: among jets changed by both the translation and dilM2, the fraction with the same new call. Its chance level is the agreement expected if the two new calls were independent given the unedited call.
- Statistics: 1,000-draw jet bootstrap for Δrank and Rrank, and 300 draws for agreement − chance at azimuth 0. Two-sided p for Δrank(translation) = 0, Holm over the 60 beyond cells.
- Hypotheses:
  - H0 (generic out-of-range perturbation): Δrank = 0 and agreement = chance.
  - H1 (acts as an M2 increase): Δrank > 0, Rrank near 1, and agreement above chance.

| | PFN | EFN | Deep Sets |
|---|---|---|---|
| inside: calls changed (translation / rotation) | 0.030–0.103 / 0.033–0.063 | 0.058–0.168 / 0.053–0.099 | 0.032–0.101 / 0.026–0.054 |
| inside: Δrank(translation) | −0.009 … +0.019 | −0.005 … +0.028 | −0.012 … +0.014 |
| beyond: Δrank(translation), 20 cells each | +0.22 … +0.88 | +0.29 … +1.36 | +0.26 … +1.03 |
| beyond: Δrank(dilM2) | −0.86 … +0.76 | −1.01 … +1.01 | −0.27 … +1.61 |
| beyond: Δrank(contraction) | −1.13 … −0.21 | −1.10 … −0.17 | −1.14 … −0.21 |
| beyond: Rrank (A = 0 / 0.04) | 0.67–1.30 | 1.17–5.68 | 0.56–1.54 |
| beyond: Rrank (A ≥ 0.16) | −2.10 … −0.44 (8/8 negative) | −8.1 … +7.0 | −7.2 … +2.8 |
| beyond: agreement − chance (A ≤ 0.04) | +0.05 … +0.11 | +0.05 … +0.11 | +0.07 … +0.14 |
| beyond: agreement − chance (A ≥ 0.16) | 0.00 … +0.07 | −0.01 … +0.02 | +0.00 … +0.05 |

- Δrank(translation) > 0 in all 60 beyond cells. The smallest bootstrap lower bound is +0.19, so the smallest z ≈ 15. The bootstrap p is at its 1/1,000 floor, so the Holm-adjusted p for all 60 cells is 0.06, at the floor set by 1,000 draws. H0 (direction-neutral) is excluded by the CIs; the Holm figure is limited by the number of draws.
- In 8 PFN cells (A = 0.16, 0.30), 2 EFN cells (A = 0.16 at T = 0.24) and 2 EFN cells (A = 0.30 s0), plus 3 Deep Sets cells, the matched dilM2 itself moves calls towards lighter classes. There, s = √(1 + T²M0/M2) ≈ 2–3, so the particles are pushed far from the axis. In those cells Rrank is negative, and the per-jet agreement with dilM2 is at chance (EFN A ≥ 0.16: −0.01 to +0.02).

Interpretation, limited to what was measured:
- Beyond A, a shift moves calls towards heavier classes in every network, as a mass-spread increase does at small T.
- The matched dilation used as the calibration stops being a "heavier" edit at the large T needed for A ≥ 0.16.
- For A ≥ 0.16, the jets and the new calls affected by the shift are not those affected by the matched dilation (agreement at chance).
- So Rc ≈ 0.5–1 beyond A is a match in the number of calls changed. For A ≥ 0.16 it is not a match of the response itself.

## Assessment of the X9x reading
- **Replicated:**
  - Inside-range Rc is small (fresh: −0.07 to 0.16 at A/2).
  - Rc rises past A, and T*_25 tracks A with slope 0.89–1.04 in all three architectures (r ≥ 0.993).
  - The pre-registered W→Z T* does not track A.
- **Needs narrowing:**
  - "T* = A + 0.01–0.02 rad" holds for A ≤ 0.16. At A = 0.30, T* − A runs from −0.035 to +0.022.
  - "Beyond A they respond about as they do to a real increase in mass spread" (thesis note) holds for the number of calls changed and for the heavier direction. Per jet, the new calls agree with the matched dilation only for A ≤ 0.04–0.08.
- **Superseded:**
  - "Within the trained range, no signed centre-offset variable that the network uses was found" / "a recentring step that uses two directions … untested". The 2-dim removal test finds that the linear offset read-out carries part of the cancellation in Deep Sets (4/4) and EFN (3/4), above norm-matched random subspaces (BH q 0.05; Holm limited by the permutation floor). PFN shows this in 1/4.
  - The 1-dim DAS result (even in t) does not bear on this question; see code-review item (c).
- **Unchanged caveats:** JEDI-linear is not in the scan. EFN augmented runs are under-trained (22–38 epochs).

## Explainer entry (for site/explain.py)
- **Question.** If a set network is trained on jets that are randomly shifted by up to A radians, how does it respond to a shift of the whole jet? Does it undo the shift (recentre)?
- **Background.** These networks see particle angles relative to a stored jet axis. Shifting every particle by a vector $\mathbf t$ moves the pT-weighted centre $\mathbf c$ to $\mathbf c+\mathbf t$. It also increases the axis moment $M_2=\sum_i p_{T,i}\,|\mathbf r_i|^2$, by the parallel-axis relation $M_2^{\text{axis}} = M_2^{\text{centroid}} + M_0|\mathbf c|^2$ with $M_0=\sum_i p_{T,i}$. A network that uses $M_2$ as a mass proxy reads a shifted jet as wider, and so heavier, unless it subtracts $M_0|\mathbf c|^2$.
- **Method.**
  - Shift by $T$ at 4 azimuths. Compare with a rotation of equal mean displacement, and with a dilation $\mathbf r\to s\mathbf r$, $s=\sqrt{1+T^2M_0/M_2}$, that changes $M_2$ by the same azimuth-averaged amount.
  - $R_c=(\text{calls}_T-\text{calls}_{\text{rot}})/\text{calls}_{\text{dil}}$, and $T^*$ is where $R_c=0.25$.
  - To test recentring: fit a linear read-out $Q$ (2-dim) of $\mathbf c$ from the pooled latent $L$. Then apply $L_0+\Delta L-QQ^\top\Delta L$ for a within-range shift, and compare with norm-matched random 2-dim subspaces.
  - Direction: mean change of mass rank $g=q<W<Z<t$, and per-jet agreement of the new call with the dilation's new call.
- **Result (figure: results/jedi/x9x_augrange.png).**
  - $T^*$ follows $A$ with slope 0.9–1.0 (PFN, EFN, Deep Sets).
  - Within range, removing $QQ^\top\Delta L$ raises call changes from 7–20 % to 46–82 % in Deep Sets and EFN, above random subspaces.
  - Beyond $A$, shifts push calls heavier. The new calls match the dilation's only for $A\le0.08$.
- **Checks.**
  - Fresh samples (seeds 4801, 4901).
  - Fine $T$ grid around $A$.
  - Norm-matched random-subspace null.
  - Matched contraction (direction control).
  - A = 0 networks (no tolerance to remove).
- **Why it matters.** The range a network tolerates is set by its training data. In two of three architectures, part of that tolerance comes from a linear read-out of the jet centre that offsets the moment change. This is the mechanism claim A10r/A10m asks about.

## Files
- interp/x9x_verify.py (parts 2, 3), interp/x9x_verify3.py (part 4)
- results/jedi/x9x_verify.json, x9x_verify2.json, x9x_verify3.json
- logs: /j-jepa-vol/aaron-interp/logs/x9x-verify.log, x9x-verify2.log, x9x-verify3.log (local copy logs/x9x-verify3.log)
