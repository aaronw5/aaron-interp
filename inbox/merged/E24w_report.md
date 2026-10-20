# E24w: prong use inside JEDI-linear at fixed M0, M2 and radial moments (prong-preserving vs prong-destroying edits) (2026-10-20)

## Conclusion
In two-prong test jets, we compared two edits of the softer prong. One rotates the prong as a whole around the jet axis. The other scatters its particles around the axis. Both keep every particle's momentum and distance from the axis, and both give the same jet mass. JEDI-linear changes the same share of calls under the two edits (difference −0.3 points, 95 % interval [−1.0, +0.4]). A boosted-tree classifier that sees the held radial quantities plus the two-prong variable τ21 changes 4.3 points more calls when the prong is scattered. So on these jets JEDI-linear's calls show no use of whether the soft prong stays intact beyond where its momentum sits. It does respond to moving the prong (3.2 points above a rigid rotation), almost entirely through its jet-level context vector.

Still uncertain:
- A classifier that reads true jet mass together with τ21 also shows no difference here, because the 11 % mass drop that both edits cause dominates its calls. A τ21 use that is masked by a mass response is therefore not excluded.
- The planned "anisotropy" variable could not be named. Linear and hinge fits on the nine pooled half-plane values explain at most 18 % of what the two context units S58 and S31 do. Holding the anisotropy at its fitted value therefore did not hold S58 and S31. The exact holds of g and of S58/S31 are the controls that work here.

## Procedure
- **Script and jobs:**
  - Script `interp/e24w_prong_jedi.py`; figure `interp/e24w_plot.py` (rendered locally).
  - Job `e24w-run2`: 1 CPU / 2 GiB, wall 1,019 s, peak 2,046 MiB (the cgroup peak includes file cache from reading the test archive), 0.74 cores on average.
  - Output `results/jedi/e24w_prong_jedi.json`, `e24w_calls.npz` (per-jet calls, τ21, edit success masks), `e24w_prong.png`; log `logs/e24w-run2.log`.
  - First run `e24w-run` (v1, see "Debug history"): `e24w_prong_jedi_v1.json`, `e24w_calls_v1.npz`, `logs/e24w-run-v1.log`.
- **Sample:** `jedi_core.sample(per_class=3000, seed=2801)`, N = 15,000 test jets.
- **Accuracy check:**
  - JEDI-linear official on the full sample: 81.60 %. This is above the expected 80.6–81.3 % band; binomial 95 % ±0.62 points.
  - The l1 + pooling replica used for the searches reproduces P exactly (max |ΔP| = 0).
  - τ21 recomputation vs the stored observable: max difference 5.8e-8.
- **Two-prong selection:**
  - Exclusive-kt reclustering of the real particles to exactly 2 subjets. Pairwise distance min(pT_i, pT_j)² ΔR_ij², no beam distance, pT-weighted (Δη, Δφ) recombination.
  - Softer subjet: pT fraction z_s ≥ 0.1 and ≥ 2 particles. τ21 < 0.5.
  - 8,093 jets pass: g 902, q 716, W 2,311, Z 2,327, t 1,837. Median z_s is 0.31; median softer-subjet multiplicity is 18.
- **Step 1, naming the anisotropy a:**
  - S58 and S31 (exact network) were fitted on the nine pooled half-plane values P_HP = P[29, 6, 40, 57, 3, 43, 30, 36, 46].
  - (i) Across jets: least squares, with R² on a held-out half.
  - (ii) Within jets (used for a in v2): ΔS58 and ΔS31 regressed on ΔP_HP. The changes come from 4 random rigid rotations and 2 softer-subjet scrambles of 1,500 selected jets; fit on half the jets, tested on the other half.
  - (iii) A MARS-style forward selection of ≤ 8 hinge terms, for both (i) and (ii).
  - a is the first principal direction of the two standardised fitted changes. The tolerance is tol = 0.01 sd(a), with sd taken over the 15,000 jets.
- **Step 2, edits:**
  - Every edit keeps each particle's pT and r about the stored axis. Measured: M0 exact; M2 and Σ pT r to ≤ 7e-8 (relative); r to ≤ 5e-8; padding untouched.
  - Every edit ends with a rigid rotation of the whole jet by an angle β. β is found by a 2° grid plus bisection so that |a − a0| ≤ tol, using the exact l1 + pooling arithmetic.
  - **R** (negative control): the rigid rotation only, with |β| ≥ 30°. Median |β| is 129°.
  - **D** (prong-destroying): each softer-subjet particle gets θ_i ~ U(0, 2π), with up to 12 redraws until a matching β exists (mean 1.7 draws).
  - **P** (prong-preserving): the softer subjet's particles are rotated rigidly about the jet axis by α, with |α| ≥ 10° (median |α| 64°). α is chosen so that the jet mass equals D's mass for the same jet (median |m_P/m_D − 1| 0.08 %, max 1.2 %); then β is applied.
    - Rotating one prong about the axis changes m. Matching m to D's makes P and D differ in prong shape and not in m.
  - **D′**: every particle scrambled, then β (mean 2.0 draws).
- **Analysis sets (v2):**
  - Main set, jets where both P and D reached tol: N = 6,376 (g 661, q 571, W 1,861, Z 1,846, t 1,437).
  - R contrasts use the 5,533 of these where R also reached tol; D′ contrasts use 5,680; R and D′ together, 4,987.
- **Networks:** official JEDI-linear n64; 8 X12 retrained copies (hgq-r150_s1 is the known 57 % copy); Deep Sets d64 s0; PFN 64-64-32 s0.
- **Extra JEDI arms (exact unit holds, via `run_jetwise`):**
  - g held at its unedited value. This holds every jet-level input to the second round.
  - S58 and S31 held at their unedited values.
- **Controls:**
  - sklearn HistGradientBoosting trees (300 iterations), fit on `sample(4000, 2802)` minus the main jets (18,876 jets). Features are recomputed on the edited jets.
  - **Th**: log M0, log(M2/M0), Σ pT r / M0, N. All held, so 0 change by construction.
  - **Thm**: Th + log m.
  - **Thmt**: Thm + τ21.
  - **Tht**: Th + τ21. This is the positive control at held moments.
  - **Tt**: τ21 only.
- **Statistics:**
  - Paired per-jet call-change indicators; 1,000-jet bootstrap 95 % intervals; exact McNemar (two-sided and one-sided upper); Holm over the 11 networks.
  - H0: P(call changed | D) = P(call changed | P). H1: D > P (prong use).
  - JEDI vs tree: paired bootstrap on the per-jet (D − P) difference.

## Numbers
**Edits (main set)**

| edit | median \|Δm\|/m | mean Δm/m | Δτ21 mean [CI] | W τ21 median | JEDI \|ΔS58\|/sd | \|ΔS31\|/sd | \|ΔS29\|, \|ΔS1\|, \|ΔS30\| /sd; S0, S19 |
|---|---|---|---|---|---|---|---|
| none | – | – | – | 0.222 | – | – | – |
| R | 0.11 % | 0.0 % | 0.000 | 0.218 | 0.44 | 0.30 | 0.06, 0.12, 0.07; 0, 0 |
| P | 9.2 % | −10.9 % | +0.032 [0.031, 0.033] | 0.251 | 0.89 | 1.23 | 0.06, 0.12, 0.07; 0, 0 |
| D | 9.2 % | −10.9 % | +0.370 [0.366, 0.373] | 0.631 | 1.12 | 1.17 | 0.06, 0.12, 0.07; 0, 0 |
| D′ | 2.3 % | −3.5 % | +0.427 [0.423, 0.430] | 0.697 | 1.21 | 0.73 | 0.05, 0.11, 0.06; 0, 0 |

- τ21 D − P: +0.338 [0.334, 0.341].
- a is held to ≤ 0.01 sd(a) in every edit. The second component b moves 0.24–0.34 of its sd.

**Calls changed (%), main set, with 95 % CI**

| model | R | P | D | D′ | D − P [CI] | McNemar n10/n01, p (one-sided; Holm 11) | D − R | P − R |
|---|---|---|---|---|---|---|---|---|
| JEDI official | 3.92 | 7.59 | 7.31 | 5.21 | −0.28 [−0.96, 0.36] | 230/248, p 0.44 (0.81; 1.0) | +3.25 [2.55, 3.98] | +3.24 [2.55, 3.90] |
| JEDI, g held | 3.09 | 3.89 | 3.64 | 3.24 | −0.25 [−0.75, 0.28] | 121/137, p 0.35 | +0.54 [0.04, 1.03] | +0.70 [0.18, 1.21] |
| JEDI, S58/S31 held | 3.36 | 4.27 | 3.97 | 3.29 | −0.30 [−0.80, 0.25] | 124/143, p 0.27 | +0.60 | +0.85 |
| X12 hgq-r150_s0 / s1 / s2 | 7.95 / 13.25 / 6.16 | | | | −0.08 / −0.33 / −0.44 | p 0.87 / 0.53 / 0.25 | | |
| X12 hgq_s0 / s1 / s2 | 6.14 / 7.48 / 4.01 | 9.24 / 14.26 / 7.56 | 8.78 / 9.38 / 6.82 | | −0.45 / −4.88 [−5.85, −3.94] / −0.74 [−1.44, −0.06] | p 0.28 / 2e-22 / 0.039 (Holm 1 / 2e-21 / 0.35) | | |
| X12 float_s0 / s1 | 4.01 / 4.10 | 9.94 / 9.63 | 9.66 / 9.28 | | −0.28 / −0.35 | p 0.45 / 0.34 | +5.37 / +4.92 | +5.49 / +4.97 |
| Deep Sets s0 | 4.74 | 6.60 | 7.47 | 7.27 | +0.86 [0.20, 1.51] | 265/210, p 0.013 (0.007; Holm 0.13) | +2.62 | +1.72 |
| PFN s0 | 5.95 | 9.38 | 9.97 | 7.66 | +0.60 [−0.25, 1.36] | p 0.16 (0.08; 1.0) | +3.89 | +3.40 |
| tree Th (held features) | 0 | 0 | 0 | 0 | 0 | – | 0 | 0 |
| tree Thm (+ m) | 0.13 | 23.65 | 23.65 | 5.77 | 0.00 [−0.13, 0.13] | 9/9, p 1 | +23.6 | +23.6 |
| tree Thmt (+ m + τ21) | 0.22 | 21.74 | 21.68 | 8.82 | −0.06 [−0.72, 0.61] | 266/270, p 0.90 | +21.2 | +21.5 |
| tree Tht (+ τ21) | 0 | 1.73 | 5.98 | 6.27 | +4.25 [3.72, 4.80] | 299/28, p 2e-58 | +5.75 | +1.61 |
| tree Tt (τ21 only, 35 % acc.) | 0 | 46.1 | 94.5 | 99.0 | +48.3 | p ≈ 0 | | |

- **JEDI minus tree, paired per-jet (D − P):**
  - vs Tht: −4.53 [−5.41, −3.67] points, p 0.001.
  - vs Thmt: −0.22 [−1.19, 0.71], p 0.66.
  - vs Thm: −0.28 [−0.97, 0.38].
- **JEDI W → Z flow** (true W jets): R 0.54 %, P 1.40 %, D 0.64 %, D′ 0.59 %. D − P: −0.75 [−1.40, −0.21] points.
- **JEDI W → q flow:** ≤ 0.18 % in every edit.
- **JEDI class-centred |Δlogit|:** R 0.71, P 1.17, D 1.18, D′ 0.99. D − P: +0.005 [−0.016, 0.025], p 0.58.
- **D′ − R** (the analogue of the E24 excess, on this two-prong subset with a matched): JEDI +0.98 [0.34, 1.58]; g held −0.06 [−0.58, 0.44].
- **Full-sample accuracies:**
  - JEDI 81.60 %.
  - X12 copies 57.5–82.8 %.
  - Deep Sets 81.2 %, PFN 81.6 %.
  - Trees Th 79.59 %, Thm 79.64 %, Thmt 79.87 %, Tht 79.69 %. τ21 adds 0.1–0.2 points of accuracy.

**Naming (Step 1)**
- **S58 fits on P_HP:**
  - Across jets: linear R² 0.087 (all 64 P values: 0.094), MARS (8 hinges) 0.167.
  - Within jets: linear 0.074, MARS 0.182.
- **S31 fits on P_HP:**
  - Across jets: linear 0.046 (all P: 0.112), MARS 0.186.
  - Within jets: linear 0.036, MARS 0.042.
- **The chosen a** (v2, within-jet PC1, 67 % of the fitted variance) explains 7.3 % of the within-jet ΔS58 variance and 0.5 % of the ΔS31 variance.
- **Direct edit, all 8,093 selected jets rotated by 45°:**
  - |Δa| median 0.28 sd(a).
  - |ΔS58| 0.56 sd, |ΔS31| 0.29 sd.
  - Radial objects: S29 0.06, S1 0.13, S30 0.08 sd; S0 and S19 exactly 0.
- **Holding a at 1 % of its sd** leaves S58 moving 0.44 (R) to 1.21 (D′) sd and S31 0.30–1.23 sd.

## Reading (scoped to what was measured)
- **H0 vs H1:** for the official network, D − P = −0.28 [−0.96, 0.36] points and the one-sided p is 0.81, so H0 is not rejected. The same holds with g held and with S58 and S31 held.
- **"Prong use as large as the held-moment τ21 tree":**
  - That prediction (+4.25 [3.72, 4.80]) lies outside the JEDI interval; paired difference −4.53 [−5.41, −3.67].
  - The "no prong use" prediction (D = P, as Th and Thm) is inside it.
  - A classifier that reads true mass and τ21 (Thmt) gives D − P ≈ 0 on these edits, because P and D both lower m by 11 % and that moves 21–24 % of its calls. JEDI's data do not separate "no τ21 use" from a τ21 use that is swamped by a mass response of the Thmt type. JEDI, however, changes 7.6 % of calls under P/D, against 23.6 % for the mass tree.
- **Response to moving the soft prong:**
  - P − R +3.24 and D − R +3.25 points, at fixed M0, M2, Σ pT r, N and fixed a.
  - With g held: +0.70 / +0.54. With S58 and S31 held: +0.85 / +0.60.
  - About 80 % of that response passes through the jet-level context g, mostly through S58 and S31. This matches the E24v decomposition (2.99 → 0.54 points).
- **Moving vs scattering the prong:** JEDI's W → Z flow is larger when the prong is moved intact (P 1.40 %) than when it is scattered (D 0.64 %).
- **Other networks:**
  - Six of the eight retrained JEDI copies give D − P intervals that contain 0. hgq_s2 gives −0.74 [−1.44, −0.06]; hgq_s1 gives −4.88 (P changes more calls than D).
  - Deep Sets gives +0.86 [0.20, 1.51] (one-sided p 0.007, Holm 0.13). PFN gives +0.60 [−0.25, 1.36].
  - No network reaches the Tht value.
- **The anisotropy a:**
  - The plan's a is not a usable name for what S58 and S31 compute. Linear and ≤ 8-hinge functions of the nine pooled half-plane values explain ≤ 19 % of their variance across and within jets.
  - "Matched a" therefore does not hold S58 and S31. The plan's test "does the matched-a constraint remove the E24 excess" is answered by the exact g / S58-S31 holds, not by a.
  - On this subset, the whole-jet scramble D′ excess over rotation is +0.98 [0.34, 1.58] at matched a and −0.06 [−0.58, 0.44] with g held.

## Checks and debug history
- **v1** (`e24w-run`, N = 1,510): a from the across-jet fit; analysis set required all four edits (R reached tol for only 36 % of jets, which limited N). Results:
  - JEDI D − P +0.26 [−1.13, 1.66], p 0.78.
  - Tht +3.38 [2.38, 4.57].
  - g held +0.66 [−0.33, 1.66].
  - S58/S31 under matched a moved 0.48–1.15 sd, which led to the within-jet a in v2.
- **v2:** within-jet a; main analysis set = P ∧ D; R and D′ contrasts on their own subsets. Same reading as v1, with intervals about 2× narrower.
- **Exactness and edit checks:**
  - P replica exact (max |ΔP| 0).
  - Padding never edited.
  - M0 exact; M2, Σ pT r and r held to ≤ 7e-8.
  - R leaves m (median 0.11 %) and τ21 (|Δ| < 1e-9) unchanged.
  - Th tree changes 0 calls (held-feature sanity check).
- **Accuracy:** JEDI 81.60 % on this sample is 0.3 points above the requested 80.6–81.3 % band. The network and forward pass are the same as in E24/E24v (80.89 %, 81.10 % on 2,000-per-class samples); the binomial 95 % interval here is ±0.62 points.
- **Limits:**
  - One D draw and one P solution per jet; draw-to-draw spread not measured (E24v: about ±0.5 points for rotation draws).
  - The kt reclustering uses a pT-weighted recombination, not full four-vectors.
  - Only Deep Sets s0 and PFN s0 were run.
  - Trees are fit on unedited jets, so edited jets are off their training distribution, as in E24v.

## Explainer entry (for site/explain.py; E24W_EXPLAINER)
- **Question:** At fixed M0 = Σ pT_i, M2 = Σ pT_i r_i², Σ pT_i r_i, multiplicity and jet mass, does JEDI-linear's call depend on whether the softer of the two prongs is intact? E24/E24v showed that a full azimuthal scramble changes about 7 % of calls, the same share as a τ21 classifier, so that test could not separate prong use from mass and radial shape.
- **Background:** N-subjettiness ratio $\tau_{21}=\tau_2/\tau_1$, with $\tau_N=\sum_i z_i\min_k\Delta R_{ik}/R_0$; two-prong W/Z jets have small $\tau_{21}$. Rotating one prong rigidly about the jet axis keeps each particle's $(p_T, r)$, and so $M_0$, $M_2$, $\sum p_T r$, and keeps the prong intact ($\Delta\tau_{21}=+0.03$). Scattering its particles in azimuth keeps the same quantities and destroys the prong ($\Delta\tau_{21}=+0.37$). Because jet mass satisfies $m^2\approx M_0M_2-|\vec M_1|^2$ (small angles), both edits change $m$; choosing the rotation angle so that $m$ equals the scattered jet's $m$ leaves prong shape as the only difference.
- **Method:**
  - 15,000 test jets (sample(3000, 2801)).
  - Two-prong jets: exclusive-kt to 2 subjets, $z_s\ge0.1$, $\tau_{21}<0.5$ (8,093 jets).
  - Edits: P (soft prong rotated by α, mass matched to D), D (soft-prong particles $\theta_i\sim U(0,2\pi)$), D′ (all particles scattered), R (rigid rotation). Each ends with a whole-jet rotation chosen so that a fitted half-plane combination $a=\vec w\cdot P_{HP}$ stays within $0.01\,\mathrm{sd}(a)$.
  - Read-out: % calls changed, paired D − P with 95 % bootstrap and McNemar.
  - Exact holds of the jet context g and of S58, S31 inside JEDI.
  - Positive control: boosted trees on the held features plus $\tau_{21}$.
- **Result (figure `results/jedi/e24w_prong.png`):** JEDI-linear changes 7.6 % (P) and 7.3 % (D) of calls; D − P = −0.28 [−0.96, 0.36] points (McNemar p 0.44, N = 6,376). The tree on held features plus $\tau_{21}$ gives +4.25 [3.72, 4.80]; paired difference −4.53 points (p 0.001). Moving the prong at all changes 3.2 points more calls than a rigid rotation; with g held this falls to 0.5–0.7.
- **Checks:**
  - $M_0$ exact; $M_2$, $\sum p_Tr$, $r$ held to $7\times10^{-8}$.
  - $m_P/m_D$ matched to 0.08 % (median).
  - Held-feature tree changes 0 calls.
  - v1 (N = 1,510) gives the same sign and reading.
  - Retrained JEDI copies: 6/8 contain 0. Deep Sets +0.86 [0.20, 1.51]; PFN +0.60.
- **Why it matters:** This is the first test in the project that holds JEDI-linear's mass and radial inputs fixed while toggling prong structure alone. It shows no call-level use of prong integrity (claim A13: the azimuthal dependence is response to where the soft prong's momentum sits, routed through g, not τ21-type shape). Two caveats:
  - A classifier that combines true mass with $\tau_{21}$ would also show D = P here, so a mass-masked $\tau_{21}$ use is not excluded.
  - The "anisotropy" that S58 and S31 read is not captured by linear or 8-hinge functions of the pooled half-planes (R² ≤ 0.19).
