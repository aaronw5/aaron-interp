# E32: JEDI-linear on realistic jet-axis errors (L1-tower and seed axes)

Node E32, stage 4. Claims A4, A4o, D4. Loop A, round 15. This is the on-manifold axis test that the round-14 review asked for.

## Conclusion

When the jet axis is moved by the error a Level-1 trigger would make, JEDI-linear's W/Z decision changes almost as it does for a real jet-mass change of the same predicted size. Here the axis is recomputed from calorimeter towers of 0.087 × 0.087, which typically moves it by 18 mrad. Measured as a ratio, the network responds at 0.9 of a real mass change, and the W→Z flow is the same size as for that mass change. So the reading "the network's mass variable is the pT-weighted spread $M_2$ about the axis it is given" holds for realistic axis errors, and not only for the large artificial shifts used before. Two points are still open:

- **The re-axised jets are not fully realistic.** Seen from the stored axis, they are not ordinary jets: a classifier separates them through quantities measured about the axis (girth and the pT-centroid offset). They are indistinguishable only on axis-free quantities.
- **Small offsets do not follow the prediction.** For offsets of a few mrad, the response falls to about half of the predicted one. At about 1–2 mrad it can no longer be told apart from the network's response to a plain rotation of the jet.

## Question

The round-14 C2ST (E31/E31v) showed that the rigid 0.06 rad shift moves jets off the real-jet manifold, because real jets never put their pT centroid that far from the axis. This experiment asks what JEDI-linear does when the axis error is a realistic one, such as a trigger would make.

## Background and definitions

- **Stored axis.** The jet axis used to write each particle's $(\Delta\eta, \Delta\phi)$ in the dataset. The network was trained on these coordinates. This is the "exact axis" in what follows.
- **$M_0, M_1, M_2$.** Sums over the real particles of a jet, with $r_i$ the particle's distance from the stored axis:
  - $M_0 = \sum_i p_{T,i}$;
  - $\vec M_1 = \sum_i p_{T,i}(\Delta\eta_i, \Delta\phi_i)$, the pT-weighted centroid times $M_0$;
  - $M_2 = \sum_i p_{T,i} r_i^2$, the "un-recentred" spread, taken about the axis and not about the centroid.
- **Jet mass.** For small angles, $m^2 \approx M_0 M_2 - |\vec M_1|^2$. Earlier work (E6, E27w) found that JEDI-linear's mass variable behaves like $\sqrt{M_0 M_2}$ about whatever axis it is given.
- **Axis offset $\vec c$.** Moving the axis by $\vec c$ re-expresses each particle as $(\Delta\eta_i - c_\eta,\ \Delta\phi_i - c_\phi)$. This changes $M_2$ to $M_2' = M_2 - 2\vec c\cdot\vec M_1 + M_0|\vec c|^2$. It changes neither $M_0$ nor the true jet mass, since the particles are the same.
- **Predicted change of the mass variable.** $\Delta\log(M_0M_2) = \log(M_2'/M_2)$.
- **L1 tower.** A Level-1 calorimeter trigger tower, the cell of $0.0873 \times 0.0873$ in $(\eta, \phi)$ in which the trigger sums energy.
- **E-scheme axis.** The direction of the summed four-momentum of the particles. Here these are massless vectors, and the axis is the rapidity and azimuth of their sum.
- **Seed axis.** In L1 jet algorithms, the jet is first placed on its hardest deposit (the "seed"), then optionally recentred.
- **C2ST (classifier two-sample test).** A classifier is trained to tell edited jets from unedited ones. AUC = 0.5 means it cannot tell them apart; 1 means it separates them completely.
- **kNN support share.** The fraction of edited jets that lie outside the region holding 99 % of real jets. The distance used is the mean distance to the 10 nearest real jets in standardised log-feature space, so 1 % of real jets fall outside by construction.
- **$R$.** The network's response to the axis change divided by its response to a matched dilation, which is a real-looking mass change of the same predicted $\Delta\log(M_0M_2)$. Formally, $R=\sum_j e_j d_j/\sum_j d_j^2$, where:
  - $e_j$ is the change of the W−Z logit margin under the axis change;
  - $d_j$ is its change under the dilation;
  - the sum runs over jets the unedited network calls W or Z.

  $R = 1$ means the network treats the axis change exactly as a real mass change of the predicted size.
- **W→Z flow.** The share of W-called jets that are called Z after the edit; Z→W is defined the same way.
- **McNemar test.** An exact test on paired per-jet "call changed" indicators: it counts jets that change under one edit only against jets that change under the other only.
- **Holm.** A step-down correction of p-values for multiple tests.

## Procedure

**Network.** The official JEDI-linear 3-feature n64 network (inputs pT, Δη, Δφ of up to 64 particles), run with the exact fixed-point forward pass `interp.jedi_core`.

**Samples.**
- Main: `sample(5000, 3201)`, N = 25,000 (5 classes × 5,000). The unedited network has accuracy **81.08 %**, inside the 80.6–81.3 % check range.
- Replicate: `sample(5000, 3202)` with the 2,437 jets also present in sample 3201 removed (byte-identical particle arrays), N = 22,563. Accuracy **81.03 %**.

**Edits.** Each edit re-expresses the same physical particles about a new axis. Every particle keeps its pT and slot, and the jet four-momentum and true mass are unchanged; only $(\Delta\eta, \Delta\phi)$ change.
- **T0:** the E-scheme axis of the 64 stored particles. This is a reference for how far the stored axis already sits from the 64 kept particles.
- **T1:** the E-scheme axis computed after moving every particle to the centre of its 0.0873 × 0.0873 tower. Each moved particle is summed as a massless vector of the same pT (ET-scheme deposit).
  - The absolute jet η, φ is not stored, so the tower-grid phase is drawn uniformly per jet.
  - Jets cover 14 towers (median).
  - There is no tower threshold and no pileup.
- **T2:** as T1, with towers of 0.0436 (22 towers per jet, median).
- **T3:** the axis placed on the hardest particle (L1 seed).
- **T4:** the T3 seed followed by one E-scheme recentring step, using the particles within $R = 0.4$ of the seed. In the median jet these carry all of the jet's pT, so T4 ≈ T0.

**Controls (same jets).**
- **C1, matched dilation.** All angles are multiplied by $\sqrt{M_2'/M_2}$ per jet. The predicted change matches within 1e-7. This is a real-looking mass change of the same predicted size, and it forms the denominator of $R$.
- **C2, rigid translation.** The jet is translated by the same $|\vec c|$ in a uniformly random azimuth. This tests whether the direction of the seeded offset matters.
- **C3, rotation.** The jet is rotated about the stored axis by a uniform random angle. This leaves $M_0$, $M_2$ and the mass unchanged, so it is the null for call changes. JEDI-linear is not rotation-invariant (its first layer has nine angular half-plane cuts), so C3 itself changes about 4 % of calls.
- **C4, untrained JEDI-linear.** Redraw seeds 1 and 2.

**Metrics.**
- Calls changed, with a 95 % jet-bootstrap CI (500 Poisson resamples), and McNemar tests against C3, C1 and C2.
- W→Z and Z→W counts, and the share of flows whose sign matches $\Delta M_2$.
- $R$ on the W−Z margin. As a second read-out, $R_{h4}$ uses the same projection on a ridge log-m read-out of layer h4 (held-out $R^2$ 0.973), taken over all jets.
- $R$ by terciles and quintiles of $|\vec c|$, plus the bootstrap p-value for H0: $R = 1$.

**Freezes under T1.** These ask where the call change passes. Units are held at their unedited per-jet values while the input is the T1 jet:
- g: all 64 context units, of which 11 are live;
- S0 + S29: S0 is ΣpT of the 14 hardest particles via g; S29 ≈ ΣpT·r² + Σr² + soft count;
- h45_top: the 4 h4 and 4 h5 units with the largest standardised response to a 5 % $M_2$ dilation, ranked on half the sample. These are h4 47, 6, 50, 59 and h5 20, 26, 7, 31; the replicate picks h5 0 in place of 31;
- h45_E6: E6's mass units, h4 18, h5 0 and h5 28.

The metric is the share of T1 call changes removed. It is evaluated on all T1-changed jets plus random jets: 4,912 jets including 2,172 changed in the main sample, and 3,749 including 1,915 in the replicate. The comparison is a set of random holds with the same number of live units: 200 in the main sample, 100 in the replicate. The random units are drawn as follows:
- for S0 + S29, from the live S units;
- for the h4/h5 sets, from the live h4 ∪ h5 units;
- for g, which has no size-matched set inside its own layer, from the live S ∪ h4 ∪ h5 units.

The p-value is a rank p, Holm-corrected over the 4 holds.

**On-manifold check.** This runs on 10,000 random jets of sample 3201. A HistGBDT C2ST compares each edited sample with the unedited jets (`e31_onmanifold_jedi.c2st`: two swapped halves × 5-fold out-of-fold scores, 300 bootstrap resamples, 300 swap permutations). Four feature sets are used:
- **inv**, axis-free: m, jet pT, N, τ21, τ32, D2, pT^D;
- **obs8**: inv + girth about the axis;
- **obs**: obs8 + the centroid term $|\vec M_1|^2/(M_0M_2)$;
- **plist**: the 30 hardest particles × (pT, Δη, Δφ).

The kNN support share (as in E31v) is computed on inv, obs8 and obs.

**Jobs and files.**
- `e32-t12` (main: forwards, freezes, C2ST for T0/T1/T2): 1,506 s, peak 1,717 MiB.
- `e32-t34` (C2ST for T3/T4, untrained): 1,100 s, peak 2,047 MiB.
- `e32-rep`: 1,240 s, peak 2,047 MiB.
- All three requested 1 CPU / 2 GiB. The peaks of two jobs reached the request (cgroup peak, which includes page cache); both completed.
- Script: `interp/e32_seeded_axis_jedi.py`. Figure: `interp/e32_plot.py`.
- Results: `results/jedi/e32_main.json`, `e32_c2st34.json`, `e32_rep.json`, `e32_seeded_axis.png`. Logs: `logs/e32-*.log`.

## Results

### 1. Size of the realistic axis errors (main sample; replicate within 0.1 mrad)

| axis | $\lvert\vec c\rvert$ median (q10–q90), mrad | share < 7 mrad | median $\Delta\log(M_0M_2)$ (q90) | share with $\Delta M_2 > 0$ |
|---|---|---|---|---|
| T0 E-scheme, 64 particles | 1.5 (0.2–5.9) | 92 % | −0.0004 (0.000) | 1.3 % |
| T1 L1 towers 0.087 | 18.3 (7.3–31.3) | 9 % | +0.046 (0.40) | 96 % |
| T2 towers 0.044 | 7.9 (3.0–14.2) | 42 % | +0.007 (0.08) | 85 % |
| T3 hardest particle | 41 (5.5–112) | 13 % | +0.21 (0.65) | 99.9 % |
| T4 seed + 1 recentring | 1.6 (0.2–8.2) | 88 % | −0.0002 (0.000) | 14 % |

The plan expected 3–15 mrad offsets. The L1-tower axis (T1) is larger, at 18 mrad median. Only the 0.044 towers (T2) fall in the 3–15 mrad range. T0 and T4 move the axis onto the pT centroid of the 64 kept particles, which slightly lowers $M_2$.

### 2. Call changes and flows (main / replicate)

| edit | calls changed % [95 % CI] | C1 dilation | C2 random direction | C3 rotation | McNemar T vs C3 | T vs C1 | T vs C2 |
|---|---|---|---|---|---|---|---|
| T0 | 1.35 [1.21, 1.49] / 1.30 | 0.10 | 1.52 | 4.12 | 170:862, p 6e-112 | 327:15, p 1e-77 | 164:208, p 0.026 |
| T1 | **8.69 [8.34, 9.02] / 8.49** | 6.08 / 6.25 | 8.82 / 8.70 | 4.20 / 3.93 | 1761:638, p 1e-120 | 1132:480, p 9e-61 | 752:785, p 0.41 |
| T2 | 2.88 [2.66, 3.09] / 2.90 | 1.18 | 2.98 | 4.13 | 436:748, p 1e-19 | 556:132, p 1e-62 | 374:398, p 0.41 |
| T3 | 31.2 [30.7, 31.8] / 31.8 | 33.1 | 35.0 | 4.03 | 7266:470, p ≈ 0 | 1221:1691, p 3e-18 | 824:1776, p 2e-79 |
| T4 | 1.79 [1.61, 1.97] / 1.67 | 0.21 | 1.69 | 4.18 | 265:863, p 3e-74 | 420:25, p 1e-93 | 244:219, p 0.26 |

W→Z and Z→W flows (main; replicate in brackets):

- **T1.** W→Z 921 jets (18.1 %) against 858 (16.9 %) under the matched dilation. The McNemar test gives 283:220, p 0.006. In the replicate, 823 (17.5 %) against 852 (18.1 %), p 0.18. All 921 W→Z flows have $\Delta M_2 > 0$. Z→W is 25 against 3, and none of those 25 has $\Delta M_2 < 0$. The 2.6-point excess of calls changed over C1 comes from flows other than W→Z. It equals what a random-direction translation of the same size produces (C2, p 0.41) and is the size of the rotation null.
- **T2.** W→Z 144 against 107 (p 0.002); 99 % of these have $\Delta M_2 > 0$. Z→W 49 against 2. T2 changes fewer calls than a rotation does.
- **T3.** W→Z 2,664 (52.3 %) against 2,695 (52.9 %) (p 0.29).
- **T0 / T4.** W→Z 33 / 38 and Z→W 46 / 48, against 0–4 under C1. All T0 W→Z flows have $\Delta M_2 < 0$, the opposite sign to the $M_2$ prediction. These rates equal C2's (p 0.03–0.6) and are a third to a half of the rotation null's.

### 3. R = margin response to the axis change ÷ response to the matched dilation

Main sample, with the replicate value after the slash. H0 is $R = 1$. All CIs are 95 % jet-bootstrap intervals; the p-value floor is 0.004 (500 resamples). The C3 column is the rotation null for $R$. The $R_{h4}$ column is the h4 log-m read-out version.

| edit | $R$ [95 % CI] | p (R = 1) | C2 $R$ | C3 $R$ | $R_{h4}$ |
|---|---|---|---|---|---|
| T0 | 0.26 [0.13, 0.40] / 0.51 [0.34, 0.67] | ≤ 0.004 | 0.44 | 0.30 [−0.12, 0.68] | 0.57 / 0.61 |
| T1 | **0.91 [0.89, 0.93] / 0.89 [0.87, 0.91]** | ≤ 0.004 | 0.905 / 0.90 | 0.062 [0.049, 0.074] | 1.31 / 1.34 |
| T2 | 0.76 [0.71, 0.80] / 0.74 [0.68, 0.79] | ≤ 0.004 | 0.74 / 0.81 | 0.26 | 1.07 / 1.04 |
| T3 | 0.88 [0.87, 0.88] / 0.89 | ≤ 0.004 | 0.95 / 0.96 | 0.017 | 0.77 / 0.76 |
| T4 | 0.27 [0.13, 0.42] / 0.50 [0.33, 0.66] | ≤ 0.004 | 0.24 | 0.27 [−0.18, 0.63] | 0.94 [0.82, 1.06] (p 0.44) |

$R$ by tercile of $|\vec c|$ (main; replicate in brackets):

| edit | lowest tercile | middle tercile | highest tercile |
|---|---|---|---|
| T1 | 0.82 [0.74, 0.89] at 0.1–14 mrad (0.84) | 0.86 [0.82, 0.90] at 14–23 mrad (0.85) | 0.92 [0.90, 0.94] at 23–54 mrad (0.90) |
| T2 | 0.51 (0.63) | 0.63 (0.69) | 0.82 (0.76) |
| T3 | 0.65 (0.68) | 0.82 (0.83) | 0.97 (0.97) |

Over the same T1 terciles, the rotation-null $R$ is 0.30 → 0.10 → 0.05. By true class, the T1 $R$ is 0.91 for W jets and 0.91 for Z jets.

On the h4 read-out, $R_{h4}$ is 1.31 for T1 but 0.91 on the margin. This read-out over-responds to translations, as in E27w arm A (1.19). The two read-outs order T1 and T3 differently, so the read-out choice matters at the 30 % level.

### 4. Where the T1 call change passes (freezes, main / replicate)

The p-values are the main-sample rank p against the size-matched random holds, followed by the Holm-corrected value (Holm over 4).

| hold (held at unedited values) | calls removed | margin response removed | random holds: median / q97.5 (calls) | p_rank / Holm |
|---|---|---|---|---|
| g, all 64 context units | 16.5 % / 16.9 % | 2.8 % | 33 % / 59 % | 0.93 / 1.0 |
| S0 + S29 | 24.9 % / 25.6 % | 29.7 % | 16 % / 29 % | 0.10 / 0.31 (replicate 0.12 / 0.36) |
| h45_top, 8 most M2-responsive h4/h5 units | **49.5 % / 48.5 %** | **83.1 % / 79.0 %** (null q97.5 55 %) | 17 % / 46 % | 0.025 / 0.10 (replicate 0.050 / 0.20) |
| h45_E6, h4 18, h5 0, h5 28 | 2.1 % / 2.0 % | 0.0 % | 8 % / 36 % | 0.97 / 1.0 |

The T1 margin response passes mainly through the h4/h5 units that respond most to a real $M_2$ dilation: holding them removes 79–83 % of it, above the random-hold 97.5 % quantile of 55 %. For calls, no hold passes Holm at 0.05. The context g carries almost none of the margin response (3 %). This matches C2 = T1: the direction-dependent context route (S31/S58 via g) plays no measurable role at this offset size.

### 5. On-manifold check (10,000 jets of sample 3201)

C2ST AUC [95 % CI]:

| feature set | T0 | T1 | T2 | T3 | T4 | C1 (T1) | C1 (T3) | C3 |
|---|---|---|---|---|---|---|---|---|
| inv (axis-free) | 0.500 | 0.503 | 0.504 | 0.500 | 0.503 | 0.645 | 0.780 | 0.494 |
| obs8 (+ girth about axis) | 0.504 | 0.733 [0.726, 0.740] | 0.616 | 0.829 | 0.501 | 0.651 | 0.780 | 0.493 |
| obs (+ centroid term) | 0.984 | 0.987 | 0.950 | 0.999 | 0.925 | 0.654 | 0.788 | 0.495 |
| plist (30 hardest particles) | 0.501 | 0.736 | 0.581 | 1.000 | 0.499 | 0.545 | 0.601 | 0.501 |

Share of jets outside the 99 % real-jet region (unedited: 1.0 %):

| feature set | T0 | T1 | T2 | T3 | T4 | C1 |
|---|---|---|---|---|---|---|
| inv | 1.0 % | 1.0 % | 1.0 % | 1.0 % | 1.0 % | 1.0 % |
| obs8 | 1.0 % | 3.6 % | 1.2 % | 1.1 % | 1.0 % | 1.0–1.1 % |
| obs | 14.3 % | 48.2 % | 22.7 % | 87.3 % | 11.7 % | 1.0–1.2 % |

The centroid term $|\vec M_1|^2/(M_0M_2)$ explains the obs column:
- Unedited jets: median $3.5\times10^{-4}$, 99th percentile $5.8\times10^{-3}$, maximum 0.024.
- T1: median 0.047, and 65 % of jets lie above the unedited maximum.
- T2: median 0.0079, 24 % above the unedited maximum.
- T3: median 0.18, 88 % above.
- T0 and T4: medians $7\times10^{-7}$ and $2\times10^{-6}$. Their axis sits on the centroid, which stored-axis jets never do exactly.

The axis-free features cannot separate any re-axised sample, because the particles are the same; this holds by construction. A real mass change (C1) moves the axis-free features (AUC 0.65–0.78) but stays inside the real-jet support.

### 6. Untrained network (C4)

Both redraws give one constant call on all jets: the margin standard deviation over jets is 0.007 for seed 1 and 0.041 for seed 2, and the calls changed are 0 for every edit. Their $R$ values are 0.11–0.91 (seed 1) and 0.37–0.56 (seed 2), with wide CIs. There is therefore no usable untrained comparison for calls. The untrained $R$ is not 1 and does not rise with $|\vec c|$ in a consistent way.

## Hypotheses

- **H0, $R = 1$** (the network follows the predicted un-recentred mass change). Rejected for every edit (bootstrap p ≤ 0.004), although T1 (0.89–0.91) and T3 (0.88–0.89) lie within the 0.8–1.2 band seen for real off-centre jets (E23wv IV 0.87 at z).
- **H1a, $R < 1$ because realistic offsets are small.** Holds in the sense that $R$ rises with $|\vec c|$ for every edit:
  - T1: 0.82 → 0.92;
  - T2: 0.51 → 0.82;
  - T3: 0.65 → 0.97.

  The premise that realistic offsets lie mostly below 7 mrad holds only for T0/T4. Only 9 % of L1-tower (T1) offsets are below 7 mrad.
- **H1b, a direction term or flows not explained by $\Delta M_2$.**
  - For the tower axes T1/T2: random-direction translations of the same $|\vec c|$ give the same calls (p 0.41) and the same $R$ (0.905 vs 0.91). W→Z flows are 99–100 % sign-matched to $\Delta M_2$, and holding g removes 3 % of the margin response. A direction term is not seen at this size.
  - For T3, the offset toward the hardest particle changes fewer calls than a random direction (824:1776) and gives a lower $R$ (0.88 vs 0.95). C2's $R$ is computed against T3's dilation, so part of this difference may come from C2's different cross term (see Limitations).
  - At 1–2 mrad (T0/T4), the flows have the opposite sign to $\Delta M_2$ and are not explained by it, but they occur at the rate of a random translation (C2) and below the rotation null.
- **Thesis expectations from the plan.** Calls changed of 1–5 % at 3–15 mrad: T2 at 8 mrad gives 2.9 %, and T1 at 18 mrad gives 8.7 %. $R$ of 0.8–1.2: met for T1 and T3; T2 is 0.74–0.76. Flows sign-matched: met for T1, T2 and T3. C2ST AUC near 0.5 for T1/T2: met only on axis-free features. With girth the AUCs are 0.73 and 0.62, and with the centroid term 0.99 and 0.95.
- **Alternative: the network recentres or reads an invariant.** This predicts $R$ well below 1 at small $|\vec c|$ and calls indistinguishable from C3. That pattern appears only at 1–2 mrad. There, the matched dilation predicts almost no change (C1 changes 0.1–0.2 % of calls), so the test has little power. At 8–54 mrad the alternative is excluded: T1's $R$ is 0.91 against a rotation null of 0.06, and T1 changes more calls than C3 (1761:638, p 1e-120).

## Reading (scoped to what was measured)

1. **Tower axis (T1).** On the 0.087 L1-tower axis, JEDI-linear's W−Z margin moves by 0.89–0.91 of the response to a real-looking mass change of the same predicted $\Delta\log(M_0M_2)$, on two disjoint samples. The W→Z flow matches the matched dilation within 7 %. This response passes through the h4/h5 units most responsive to $M_2$: holding them removes 79–83 % of the margin response, against a random-hold 97.5 % quantile of 55 %.
2. **Offset size.** The ratio $R$ rises with the offset size. At the scale of a natural axis disagreement (1–2 mrad), the response is a few tenths of the prediction and cannot be told apart from the rotation null. That scale is the gap between the stored axis and the E-scheme axis of the 64 kept particles.
3. **Off the stored-axis manifold.** The realistic seeded axes still produce inputs that differ from stored-axis jets in axis-relative quantities: girth about the axis and the centroid offset. For T1, 48 % of jets lie outside the 99 % real-jet support on obs, and 3.6 % on obs8. Even T0 and T4, which recompute the axis offline from the same particles, are separable (AUC 0.92–0.98), because they place the axis exactly on the pT centroid.

So "on-manifold" for an axis error holds only with respect to axis-free quantities. A network trained on stored-axis jets sees a tower-seeded axis as a centroid offset larger than any in its training sample, for 65 % of jets.

## Limitations

- **C2 dilation mismatch.** C2's $R$ uses T's matched dilation as its denominator. C2's own predicted $\Delta M_2$ differs through the cross term $-2\vec c\cdot\vec M_1$:
  - for T1–T3 the $M_0|\vec c|^2$ term dominates, so the comparison is approximately matched;
  - for T0/T4 it is not (C2's predicted change has the opposite sign), so C2's $R$ there is not interpretable.

  The call and flow comparisons with C2 are unaffected.
- **Flow sign-match columns.** These use T's $\Delta M_2$, so they are meaningful only for T and C1.
- **Tower model.** Particles are moved to the tower centre with their pT. There is no energy threshold, no pileup and no detector response, and the grid phase is random because absolute η, φ are not stored. The seed axes T3/T4 use particles, not towers.
- **h45_top selection.** These units were ranked on half A of the same sample with a different edit (a 5 % dilation). The evaluation uses all jets.
- **E6 units.** E6's h4 18, h5 0 and h5 28 remove nothing. E6's unit numbering may not refer to the same units; this was not checked.
- **Untrained redraws.** Both are constant-call networks, so the untrained comparison for calls is empty.
- **Bootstrap floor.** The bootstrap p floor is 0.004 (500 resamples).
- **Read-out choice.** The margin and h4 read-out ratios differ by up to 0.4 for T1, so a single number for "the" $R$ depends on the read-out.

## Glossary entries (for explain.GLOSSARY)

- **L1 tower:** a Level-1 calorimeter trigger tower, the 0.0873 × 0.0873 cell in (η, φ) in which the hardware trigger sums energy. A jet axis built from towers is quantised to this grid.
- **Seed axis:** the starting jet direction of a trigger jet algorithm, placed on the hardest deposit (here the hardest particle). It may be followed by recentring steps.
- **E-scheme:** recombination by adding four-momenta; the E-scheme axis is the direction of the summed four-momentum of the jet's particles.
- **C2ST (classifier two-sample test):** a classifier is trained to separate two samples (e.g. edited vs real jets) and scored out of fold. AUC 0.5 means the samples cannot be told apart on the given features, and 1 means they separate completely.
- **R (E32):** the network's W−Z margin response to an axis change, divided by its response to a real-looking mass change (an angular dilation) of the same predicted $\Delta\log(M_0M_2)$, projected jet by jet. R = 1 means the network treats the axis error exactly as that mass change.
- **Centroid term:** $|\vec M_1|^2/(M_0 M_2)$, the share of the un-recentred spread $M_2$ that comes from the pT centroid sitting off the axis. Its median is $3.5\times10^{-4}$ for stored-axis jets.

## Explainer entry (for site/explain.py, Paper A tab)

1. **Question.** Does JEDI-linear's mass reading respond to the kind of axis error a trigger makes, as it does to the large artificial shifts used before?
2. **Background.** The network reads mass as $\sqrt{M_0M_2}$ with $M_2=\sum p_T r^2$ about the stored axis. Moving the axis by $\vec c$ changes $M_2$ to $M_2-2\vec c\cdot\vec M_1+M_0|\vec c|^2$ while the true jet mass stays fixed. E31 showed the 60 mrad shift used so far produces jets unlike any real jet.
3. **Method.** The axis is recomputed four ways:
   - L1 towers of 0.087 (median offset 18 mrad);
   - towers of 0.044 (8 mrad);
   - the hardest particle (41 mrad);
   - the hardest particle plus one recentring step (1.6 mrad).

   Each jet's particles are re-expressed about the new axis. The network's W−Z margin response is compared with its response to a dilation giving the same predicted $\Delta\log(M_0M_2)$, as the ratio $R$. Controls are a random-direction translation of the same size, a random rotation and untrained networks. Units in g, S and h4/h5 are frozen to find where the response passes. A C2ST and a kNN support test check whether the re-axised jets look like real jets.
4. **Result.** See the figure `results/jedi/e32_seeded_axis.png`. For the L1-tower axis, $R$ = 0.89–0.91 on two samples, and the W→Z flow equals the dilation's (18 % of W-called jets). $R$ rises with offset size (0.82 → 0.92 for T1), a random direction gives the same response, and the response passes through the most mass-responsive h4/h5 units (79–83 % removed by holding them).
5. **Checks.**
   - Rotation null: $R$ = 0.06.
   - Replicate on disjoint jets: within 0.02.
   - The C2ST separates the re-axised jets only through axis-relative quantities (girth, centroid offset).
6. **Why it matters.** The $M_2$-about-the-axis reading is not limited to unphysical inputs. A realistic trigger-axis error changes JEDI-linear's W/Z decision by close to the amount a real mass change of the same predicted size would. The inputs produced by such an axis lie outside the stored-axis training distribution in their centroid offset.
