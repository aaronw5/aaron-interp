# H69 — width, prong sharing, D2 and τ21 separated at fixed jet mass and jet pT (claim D25 v2)

Script `interp/h69_width_sharing_d2.py`.

Jobs (all finished):
- smoke runs, 30 jets per class: `d-h69-sm-{top,hls,jc,jedi,gpu,sum,sum2,sum3,fig}`;
- edits, CPU: `d-h69-ed-hls`, `d-h69-ed-jc`, `d-h69-ed-top`;
- network evaluation: `d-h69-ev-jedi` (CPU) and `d-h69-ev-gpu` (JetClass and Top, GPU);
- references, statistics and figures: `d-h69-sum3`, then `d-h69-figr` (figure redraw).

Two debug fixes were needed:
- the summary read a pickle inside a list comprehension (EOFError);
- φ divided by zero for a permuted network with a constant output.

Results:
- in the repo: `results/physics/h69_summary.json`, `h69_edits_{hls,jetclass,top}.json`, `h69_jet.png`, `h69_response.png`;
- per-jet tables and logits on the volume: `/j-jepa-vol/aaron-interp/results/physics/h69_work/`.

## Conclusion

At fixed jet mass and jet pT, the JetClass and Top Landscape taggers change their scores when either of two things changes:
- the prong pT sharing (how evenly the jet's pT is split between its two or three subjets), or
- D2 or τ21 (radiation patterns inside and between the prongs).

For the 10 % D2 edit, the response is 3–6 times that of a control edit that moves particles as far without changing these quantities. It is 1.2–1.4 times for the same network with permuted weights. No single-variable reference classifier reproduces the networks' pattern:
- a classifier that reads only mass, pT and prong sharing responds almost only to the sharing edit;
- one that reads only mass, pT and D2 responds mostly to the D2 edit;
- the networks sit in between.

So the networks read several shape variables, not one. The pT-weighted width cannot be tested on signal jets. On two-prong jets it is almost fixed by mass, pT and sharing (w ≈ √(2s)·m/pT). A width change of ±10 % with mass, pT and sharing held could be built for at most 0.7 % of W/Z/H jets and for no top jets.

Two things are still uncertain:
- The control does not reweight particle pT, while the edits do. The permuted-weight networks also respond to the edits; for the Top LorentzNet null the D2-edit response even exceeds the trained network's.
- The D2 edit also changes τ21 by about the same fraction. The two are separated only by the τ21-at-fixed-D2 edit, which can be solved on 12–20 % of two-prong signal jets and on < 1 % of top jets.

## Question and hypotheses

H67 and H67v showed the following. JetClass and Top taggers respond to an edit at fixed mass and fixed pT that changes the width w and the prong sharing s equally. The second variable these taggers read (besides mass) was therefore left unidentified.

Competing hypotheses and what each predicts:

| hypothesis | prediction |
|---|---|
| H_s: the networks follow prong sharing | E_s moves the score like MLP(m,s,pT); E_w and E_D2 at fixed s stay within the crot_dm band (R ≈ 1) |
| H_D2 | E_D2 dominates and matches MLP(m,D2,pT) |
| H_w | E_w dominates |
| several variables | every edit exceeds crot_dm and no single-variable reference matches |

## Definitions

All quantities are measured about the E-scheme axis, the direction of the summed constituent three-momentum of the unedited jet.

- **w (width)**: w = Σ_i pT_i ΔR_i / Σ_i pT_i, where ΔR_i is the (η, φ) distance of particle i to the axis.
- **Prongs**: exclusive-kT subjets of the unedited jet: 3 for top-class jets, 2 otherwise. The labels are kept fixed through every edit.
- **s (prong sharing)**: s = 1 − Σ_k (P_k/P)², where P_k is the pT of prong k and P = Σ pT. For two prongs s = 2z(1 − z), with z the softer prong's pT share.
- **D2**: e3/e2³ with β = 1, the ratio of the three-point to the cubed two-point energy correlator, on all constituents. Low D2 means two-prong-like.
- **τ21**: τ2/τ1 (β = 1, exclusive-kT winner-take-all axes), the N-subjettiness ratio. Low τ21 means two-prong-like.
- **Δs**: the change in the network's own-class one-vs-rest log-odds, log[p_k/(1 − p_k)]. It is computed per jet: edited minus unedited.
- **R(E)**: mean|Δs| under edit E divided by mean|Δs| under the control crot_dm on the same jets. R = 1 means the edit moves the score no more than the control.
- **φ_sD2**: the share of the response that goes to the sharing edit, M_s/(M_s + M_D2). M_X is the mean |Δs| over the + and − edits of X.
- **Holm**: step-down family-wise correction of p-values over the 403 model × class × edit tests.
- **McNemar**: paired test on calls that flip at the class's ε_S = 0.5 working point. The threshold is the median unedited own-class log-odds of the class's jets; the comparison is the edit vs crot_dm on the same jets.

## Procedure

### Edits

Every edit is solved exactly per jet with a damped Newton iteration: finite-difference Jacobian, least-squares step, 80 iterations. Jet pT, η and φ are restored exactly. The particle count and order are unchanged, and each particle keeps its direction.

The four levers:

- **c (width lever, all prongs).** pT_i is multiplied by (1 + c·h_i). Inside each prong, h_i is the particle's distance r_i to the jet axis made pT-weighted-orthogonal to {1, Δη, Δφ, r²}, scaled to max|h| = 1.
  - For every c, exactly: the prong pT, the prong pT-weighted centroid and the prong's Σ pT r² are unchanged. The prong four-momentum and the jet mass are therefore unchanged to first order, while Σ pT r changes.
  - The pT moves between the core and the edge of each prong.
  - Deviation from the plan: the plan specified a two-set core/edge split at 0.05 from the prong axis. That split cannot hold P_k, the centroid and Σ pT r² at once; H67's lever diagnostic showed it moves m and w together.
- **cs (D2 lever, softest prong only).** Core particles (within 0.05 of the prong centroid; the pT-median split for 10–20 % of prongs with < 2 % or > 98 % of pT inside 0.05) are scaled by (1 − cs·a_E). Edge particles are scaled by (1 + cs·a_C), where a_C and a_E are the core and edge pT shares. The prong pT is kept and its mass changes.
- **f (sharing lever).** Moves pT between prongs (H67 lever).
- **u (opening lever).** Scales the prong centroids about the axis.

The edits, with the quantity held fixed and the levers solved:

| edit | changed | held fixed | levers |
|---|---|---|---|
| E_w ±10 %, ±5 % | w | m, pT, s | f, u, c |
| E_s ±10 %, ±5 % | s | m, pT, w | f, u, c |
| E_D2 ±10 %, ±5 % | D2 | m, pT, s (w free) | f, u, cs |
| E_τ21 ±10 % | τ21 | m, pT, s, D2 | f, u, c, cs |

A jet counts only if its residual is < 1e-5 in log for m, w and s, and < 1e-4 for D2 and τ21. The ±5 % edits were added because the ±10 % width edit rarely converges.

### Controls

- **crot_dm.** Each prong is rotated about its centroid by one angle per jet. The angle is chosen so that the pT-weighted particle displacement equals the mean displacement of that jet's converged 10 % edits. Then m, w and s are restored exactly.
  - Achieved median displacement: 0.0019 rad (hls4ml), 0.0042 rad (JetClass), 0.0078 rad (Top), equal to the target.
  - D2 and τ21 change by ≤ 0.1 % (median).
- **crand.** Every particle is displaced by the same distance in a random direction, then m and s are restored.
- **Permuted nulls.** One weight-permuted network per architecture: JEDI-linear, ParT_full, PN_full, and Top ParT, PN, LorentzNet, PFN, EFN.

### References

Reference classifiers trained on train rows of each dataset:
- MLP(32, 32) on log of each feature set: MLP(m,w,pT), MLP(m,s,pT) (s from re-clustered 2- and 3-subjet sharing), MLP(m,D2,pT), MLP(m,τ21,pT);
- GBDT on 32 energy-flow polynomials.

Each reference is evaluated on the same edited jets as the networks.

### Statistics

- 95 % bootstrap intervals over jets (1,000 resamples).
- Paired sign-flip test of |Δs(E)| against |Δs(crot_dm)|. With 2,000 flips the p-value floors at 5e-4, so Holm uses the large-sample form of the same test.
- McNemar on call flips at ε_S = 0.5.
- Holm over 403 trained-model × class × edit tests.

### Samples

| dataset | evaluated jets | reference rows |
|---|---|---|
| hls4ml | `jedi_core.sample(per_class=1000, seed=6901)`, 5,000 jets; edits on all, W/Z/t reported | 3,000 per class from `train_ptetaphi_f16` (rng 6911), 64 hardest particles |
| JetClass | val_5M `<sample>_122.root`, 1,000 random rows each of Wqq, Zqq, Tbqq, Hbb, QCD (rng 6902; file not used by H67/H67v) | train_100M `<sample>_005.root`, 3,000 per class (rng 6912) |
| Top Landscape | test, 2,000 + 2,000 random rows (rng 6903) | train split, 3,000 + 3,000 (rng 6913) |

## Unedited accuracy on these jets (all reproduce)

| model | here | known |
|---|---|---|
| JEDI-linear (5 classes) | 0.813 [0.802, 0.824] | 0.807 |
| ParT_full W / Z / t / QCD | 0.795 / 0.705 / 0.955 / 0.781 (Hbb 0.919) | 0.79 / 0.69 / 0.96 / 0.77 |
| ParT_kin | 0.722 / 0.558 / 0.916 / 0.731 | 0.69 / 0.56 / 0.91 / 0.71 |
| PN_full | 0.774 / 0.659 / 0.928 / 0.767 | 0.77 / 0.64 / 0.95 / 0.75 |
| PN_kin | 0.695 / 0.532 / 0.904 / 0.695 | 0.67 / 0.53 / 0.91 / 0.69 |
| Top ParT / PN / LorentzNet / PFN / EFN | 0.948 / 0.946 / 0.950 / 0.932 / 0.921 | 0.937 / 0.940 / 0.943 / 0.929 / 0.923 |

Permuted nulls are at chance or below:
- JEDI_perm0 gives a constant output, so all its Δs = 0 and it cannot serve as a response floor;
- ParT_full_perm0 0.05, PN_full_perm0 0.14;
- Top nulls 0.34–0.57.

The reference accuracies are in `h69_summary.json` (JetClass 5-class: 0.59–0.66; Top: 0.885–0.890).

## Numbers

### 1. Edits are exact; side effects

Over converged jets:
- max |Δlog m| ≤ 1e-8;
- max |Δlog pT| ≤ 7e-16;
- axis shift ≤ 7e-8 rad.

Convergence fraction (W / Z / top, or W / Z / H / t for JetClass) and median side effects:

| edit | hls4ml W / Z / t | JetClass Wqq / Zqq / Hbb / Tbqq | Top top | side effects (median) |
|---|---|---|---|---|
| E_w ±10 % | ≤ 0.007 / ≤ 0.001 / 0 | ≤ 0.007 / ≤ 0.007 / ≤ 0.002 / 0 | 0 | Δlog D2 −1.5 to −2.7 × Δlog w |
| E_w ±5 % | 0.02–0.03 | 0.04–0.05 / 0.03–0.05 / 0.02 / 0 | 0 | same ratio |
| E_s ±10 % | 0.05–0.08 | 0.12–0.16 / 0.13–0.15 / 0.07–0.10 / 0.05–0.08 | 0.06–0.10 | ΔD2 ≤ 1 %; Δτ21 2–4 % (two-prong), ±15–17 % (top) |
| E_D2 ±10 % | 0.60–0.76 / 0.13–0.24 (t) | 0.59–0.66 / 0.57–0.65 / 0.54–0.68 / 0.06–0.08 | 0.06–0.09 | Δlog τ21 ≈ +1.2 × Δlog D2 (two-prong); Δlog w ≈ −0.1 × Δlog D2 |
| E_τ21 ±10 % | 0.05–0.08 / ≤ 0.01 (t) | 0.14–0.20 / 0.12–0.16 / 0.09–0.12 / ≤ 0.008 | ≤ 0.006 | Δw ≤ 0.7 % |

For QCD jets E_w converges more often:
- ±5 %: 0.15–0.32;
- ±10 %: 0.01–0.13.

Why E_w rarely converges on signal jets: for two-prong jets the width is close to √(2s)·m/pT.
- median w/(√(2s)·m/pT) = 0.97 (hls4ml), 0.95 (JetClass), 0.83 (Top QCD);
- correlation of the logs = 0.92–0.93.

In the small-angle limit m²/pT² ≈ ⟨r²⟩ and w = ⟨r⟩. With the prong pTs fixed, w can then move only through second-order changes of each prong's radial profile.

### 2. Response relative to crot_dm (R, 95 % CI), signal classes

Columns: E_w ±5 %, E_s ±10 %, E_D2 ±10 % and E_τ21 ±10 % (E_s / E_D2 / E_τ21 also in `h69_summary.json` at ±5 %).

| model / class | R(E_w ±5) | R(E_s ±10) | R(E_D2 ±10) | R(E_τ21 ±10) | φ_sD2 (10 %) |
|---|---|---|---|---|---|
| ParT_full Wqq | 2.6 [2.0, 3.4] | 3.3 [2.9, 3.9] | 5.1 [4.5, 5.8] | 4.1 [3.4, 5.0] | 0.40 [0.36, 0.43] |
| ParT_full Zqq | 2.7 [1.9, 3.6] | 3.5 [2.9, 4.2] | 4.9 [4.3, 5.4] | 3.6 [3.1, 4.3] | 0.42 |
| ParT_full Hbb | 3.3 [2.1, 4.6] | 3.6 [2.8, 4.4] | 5.1 [4.5, 5.8] | 4.0 [3.3, 4.7] | 0.41 |
| ParT_full Tbqq | – | 6.8 [5.5, 8.5] | 7.9 [6.5, 9.5] | – | 0.46 |
| ParT_full_perm0 Wqq / Zqq / Hbb | 1.4–1.7 | 1.2–1.3 | 1.3 | 1.3–1.5 | 0.50–0.51 |
| ParT_kin Wqq | 1.7 [1.3, 2.3] | 2.2 [1.9, 2.5] | 1.3 [1.2, 1.5] | 1.9 [1.6, 2.2] | 0.62 [0.59, 0.65] |
| PN_full Wqq | 1.8 [1.5, 2.2] | 2.6 [2.3, 3.1] | 3.8 [3.4, 4.2] | 2.6 [2.2, 2.9] | 0.41 |
| PN_full_perm0 Wqq | 2.3 [1.8, 2.8] | 1.7 [1.5, 2.0] | 2.0 [1.8, 2.2] | 2.1 [1.9, 2.4] | 0.46 |
| PN_kin Wqq | 2.6 [2.1, 3.3] | 3.2 [2.8, 3.8] | 4.2 [3.8, 4.7] | 3.2 [2.8, 3.7] | 0.43 |
| JEDI-linear W | 0.58 [0.42, 0.78] | 1.10 [0.85, 1.38] | 1.43 [1.33, 1.55] | 1.39 [1.19, 1.61] | 0.44 |
| JEDI-linear Z | 0.43 [0.30, 0.58] | 1.02 [0.76, 1.34] | 1.61 [1.49, 1.73] | 1.13 [0.91, 1.36] | 0.39 |
| JEDI-linear t | – | 2.3 [1.9, 2.8] | 1.9 [1.6, 2.2] | – | 0.55 |
| Top ParT top | – | 6.5 [5.8, 7.2] | 5.9 [5.3, 6.5] | – | 0.52 |
| Top ParT_perm0 top | – | 2.6 | 4.1 | – | 0.39 |
| Top PN / LorentzNet / PFN / EFN top | – | 4.8 / 5.0 / 10.9 / 9.1 | 4.4 / 4.2 / 6.1 / 9.9 | – | 0.52 / 0.54 / 0.64 / 0.48 |
| Top PN / LorentzNet / PFN / EFN perm0 | – | 2.7 / 4.1 / 4.0 / 6.4 | 4.3 / 11.4 / 1.6 / 3.0 | – | 0.39 / 0.26 / 0.72 / 0.68 |

Reference φ_sD2 at 10 %, over W/Z/H/t (and Top top):

| reference | φ_sD2 |
|---|---|
| MLP(m,s,pT) | 0.71–0.89 (Top 0.90) |
| MLP(m,D2,pT) | 0.12–0.39 (Top 0.37) |
| MLP(m,τ21,pT) | 0.33–0.77 (Top 0.67) |
| GBDT(EFP) | 0.38–0.54 (Top 0.45) |
| MLP(m,w,pT) | 0 (its inputs are fixed by both edits) |

Two sanity checks on the references:
- MLP(m,D2,pT) has R(E_τ21) = 0.00;
- MLP(m,w,pT) has |Δs| ≈ 1e-5 of the networks' under E_s.

Both confirm that the edits hold what they claim to hold.

### 3. Tests

- **Sign-flip test**, |Δs(E)| vs |Δs(crot_dm)|: 323 of 403 trained model × class × edit tests have Holm p < 0.05.
  - Every ParT_full, PN_full and PN_kin E_D2 ±10 % test on Wqq, Zqq and Hbb passes (Holm p ≤ 1e-25), and so does every E_s ±10 % test on Wqq and Zqq (Holm p ≤ 1e-3). PN_full Hbb E_s (Holm p ≈ 0.1) fails.
  - Most tests that fail are E_w ±5 % (n = 22–49) and Tbqq E_s (n = 30).
  - The 2,000-flip permutation p (floor 5e-4) passes no test after Holm. That reflects the floor, not the data.
- **McNemar** at ε_S = 0.5: 92 of 403 tests have Holm p < 0.05.
  - ParT_full Wqq E_D2−10: 14.7 % of calls flip vs 2.1 % under crot_dm.
  - PN_full Zqq E_D2+10: 12.3 % vs 3.9 %.
- **crand** (random displacement, equal size, m and s restored): R = 1.0–2.1 for the JetClass and Top networks. Exception: ParT_kin, PFN and EFN have R < 1 (0.3–0.7).
- **Per-jet correlation** of signed Δs between each network and each reference, pooled over the six 10 % edits:
  - JetClass: ≤ 0.29 for every reference;
  - Top taggers: highest for MLP(m,τ21,pT), 0.35–0.53.

## Reading

### Hypotheses

**H_s** predicts R(E_D2) ≈ 1 and R(E_τ21) ≈ 1 at fixed s. It is excluded for ParT_full, PN_full, PN_kin and all Top taggers:
- R(E_D2) = 3.3–9.9 (group R, both signs) with CIs above the permuted-null values for ParT_full (1.3), PN_full (2.0) and Top ParT (4.1 at 10 %);
- R(E_τ21) = 2.6–4.1 on two-prong signal jets.

**H_D2** predicts φ_sD2 near MLP(m,D2,pT) (0.12–0.39) and R(E_s) ≈ 1 at fixed D2. It is excluded:
- E_s changes D2 by ≤ 1 % but moves the scores by R = 1.8–10.9;
- network φ_sD2 = 0.38–0.64 is above the MLP(m,D2,pT) range in every JetClass cell;
- in the Top cells the networks (0.48–0.64) are above MLP(m,D2,pT) (0.37).

**H_w** cannot be decided on signal jets, because E_w is not reachable there (section 1). Where E_w ±5 % is reachable (22–49 jets per class), the JetClass networks give R = 1.6–3.3. This is below or equal to their E_D2 response.

**Several variables.** The JetClass/Top networks respond beyond crot_dm to all reachable edits. Their φ_sD2 lies between the sharing-only and D2-only references, near GBDT(EFP) and MLP(m,τ21,pT). This is the outcome the plan lists as "nets read several shape variables". The per-jet correlations with any single-variable reference are low (≤ 0.29 on JetClass). No single-variable reference predicts which jets move.

### JEDI-linear (hls4ml)

The responses are small:
- R(E_D2) = 1.4–1.9 (Holm p ≤ 2e-2);
- R(E_s) = 1.0–1.4 on W/Z, with CIs that include 1 at 10 %;
- R(E_w ±5) = 0.4–0.6, below the control.

This fits A4/D1 (a mass-window reader with little shape input) but is not a confirmation of it, because its permuted null is degenerate.

### ParT_kin

ParT_kin has R(E_D2) of 1.3–1.7 and R(E_s) of 1.8–2.6; φ_sD2 = 0.52–0.62. It leans further toward sharing than the other JetClass networks.

### Caveat on the floor

crot_dm moves particles but does not reweight their pT, while every shape edit does. The permuted networks respond to the edits at R = 1.1–2.3 (JetClass) and up to 11 (Top LorentzNet_perm0, E_D2). For the Top taggers, the trained/null contrast is therefore weak: for LorentzNet the null's R(E_D2) exceeds the trained one. Only the JetClass ParT_full/PN contrasts with their nulls are clear.

Also:
- E_D2 moves τ21 by about 1.2 times the D2 change, so E_D2 alone does not separate D2 from τ21;
- E_τ21 at fixed D2 does, but only for two-prong jets.

## Physics in one paragraph

For a jet made mostly of two hard prongs, the mass, the pT and the prong pT sharing already fix the pT-weighted width to within a few per cent. So "width" is not a separate handle on signal jets. What remains free at fixed mass and pT is:
- how the pT is shared between prongs, and
- how the radiation inside and around the prongs is distributed, which D2 and τ21 measure.

The JetClass and Top taggers use both, in proportions that no one-variable cut-based tagger matches.

## Figures

- `results/physics/h69_jet.png`: one Top Landscape top jet (pT 568 GeV) before and after E_s (−5 %), E_D2 (−5 %), E_τ21 (−10 %) and crot_dm.
  - Each panel title gives m, w, s, D2 and τ21.
  - Prong core particles are filled and edge particles open; the core radius 0.05 is shown as a dashed circle.
  - The E_w panel reads "no exact solution on this jet": no top jet admits E_w.
- `results/physics/h69_response.png`: R for E_w ±5 %, E_s ±5 %, E_D2 ±5 % and E_τ21 ±10 % for every trained network and its permuted null.

## Explainer entry (for site/explain.py; not written there, since site/ is outside this loop's write scope)

**Question.** Beside the jet mass, which shape variable do jet taggers read: the width w, the prong pT sharing s, D2 or τ21?

**Background.**
- $w=\sum_i p_{T,i}\Delta R_i/\sum_i p_{T,i}$ is the pT-weighted width.
- $s=1-\sum_k (P_k/P)^2$ is the prong sharing; for two prongs $s=2z(1-z)$.
- $D_2=e_3/e_2^3$ is the ratio of the three-point to the cubed two-point energy correlator.
- $\tau_{21}=\tau_2/\tau_1$ is the N-subjettiness ratio.

In the small-angle limit $m^2/p_T^2\simeq\langle r^2\rangle$ and $w=\langle r\rangle$. For two point-like prongs $w=\sqrt{2s}\,m/p_T$, so at fixed $m$, $p_T$ and $s$ the width can only move through the radial profile inside each prong.

**Method.**
- Per jet, solve the four levers (sharing f, opening u, a within-prong radial reweighting c that keeps every prong's pT, centroid and $\sum p_T r^2$, and a core/edge reweighting cs of the softest prong) so that one variable moves by ±5 % or ±10 % and the others stay exact.
- Compare the change in the tagger's own-class log-odds with that under a prong rotation of equal particle displacement that keeps m, pT, w and s: $R=\langle|\Delta s_E|\rangle/\langle|\Delta s_{\rm crot}|\rangle$.
- Compare with MLP(m,s,pT), MLP(m,D2,pT), MLP(m,τ21,pT), MLP(m,w,pT) and GBDT(EFP) on the same edited jets.

**Result** (figure `h69_response.png`, jet `h69_jet.png`):
- ParT_full W: $R(E_s)=3.3$, $R(E_{D2})=5.1$, $R(E_{\tau21})=4.1$; its permuted copy gives $R\approx1.3$.
- The share $\phi_{sD2}=M_s/(M_s+M_{D2})=0.40$, between MLP(m,s,pT) (0.77) and MLP(m,D2,pT) (0.18).

**Checks.**
- m, pT and the axis are exact to $10^{-8}$.
- MLP(m,D2,pT) does not respond to the τ21 edit, and MLP(m,w,pT) does not respond to the sharing edit.
- Unedited accuracies reproduce.
- Holm-corrected sign-flip and McNemar tests.

**Physical significance.** At fixed mass and pT, the taggers read both how the pT is split between the prongs and how radiation is spread within them, not a single classical shape variable. The jet width is not an independent variable for two-prong jets.

Glossary additions:
- **prong sharing s**: 1 − Σ(P_k/P)² over the jet's kT subjets; 0 for one dominant prong, 0.5 for two equal prongs.
- **R (H69)**: mean |Δ log-odds| under an edit divided by that under a displacement-matched prong rotation.
- **φ_sD2**: share of the response to a sharing edit among (sharing edit + D2 edit).
- **crot_dm**: the displacement-matched prong rotation control.
- **width lever c**: within-prong pT reweighting along the distance to the jet axis, orthogonal to {1, Δη, Δφ, r²}.
