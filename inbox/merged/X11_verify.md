# X11 verify — two verifier passes (fresh samples, 250 fresh raw rotations of JEDI S, generic library, untrained architecture control)

Status: done. Confidence that the X11 reading is correct as stated: 0.85 (corrections below). Refuted: no.
Pass 2 (this section, 2026-10-19 evening): code `interp/x11_verify2.py` (imports `interp.x11_sae_atlas` and
`interp.x11_verify.compare` unchanged); results `results/jedi/x11_verify2_trained.json`, `x11_verify2_untrained.json` (also on
the volume); logs `logs/x11-verify2.log`, `logs/x11-verify2-u.log`. Jobs x11-verify2 (1 CPU / 2 GiB, 1,004 s, peak 1.61 GiB) and
x11-verify2-u (1 CPU / 2 GiB, 1,701 s, peak 2.05 GiB, at the limit; it finished). The job name x11-verify was already used by
pass 1 (job still present), so pass 2 used x11-verify2. The first x11-verify2 attempt exited after 21 s reading an empty
`x11_prereg.json` (read during the sync); the file on the volume matches the local one (md5 bff78bb5…) and the rerun ran normally.
Pass 1 (earlier the same day) is kept unchanged below.

## Conclusion
On a third, fresh set of jets JEDI-linear's pooled object units are again written as short pT/angle formulas for 79 % of units,
and none of 50 new random mixings of the same units reaches that rate (over both verifier passes, none of 250). The same network
architecture with random (untrained) weights gives units that almost never fit such formulas, so the formula-readable units come
from training; this control cannot say whether an untrained network would also prefer its own unit basis, because nothing in it
is readable in either basis. Still open: whether the basis preference is learned or a property of the per-particle ReLU-sum
architecture, and how much the SAE result depends on the additive hinge scorer.

(Terms: "named" = held-out R² ≥ 0.9 for a ≤ 8-term hinge formula of the 97 rank-window observables; "rotation" / "mixing" =
random orthogonal combination of the live units; "untrained" = every weight redrawn from a Gaussian with the trained layer's
mean and spread.)

## Code review of `interp/x11_sae_atlas.py` (pass 2, independent of pass 1)
- Stage: S is `pool('qs1', u)` from `jedi_core` (after the pooling quantizer), h4 = `dense('d4', S)`; the forward pass is the
  unmodified `Jedi(64)`; accuracy 80.91 % on the pass-2 fit sample (15,000 jets).
- Padded slots: library observables, edits (`m &= real`), the azimuthal rotation and `in_range` act on pT > 0 only. The
  network itself sees padded slots as in training (correct: no change to the model's input).
- Split: fit-train / fit-dev from `sample(5000, 0)`, test from `sample(5000, 29)` with the 2,449 overlapping rows removed via
  `indices`; fitter selection jets drawn from fit-train only. No leakage found.
- Edits: pT × 1.1 inside the window then re-sort (library recomputed, so rank crossings enter the prediction); angle × 1.1
  scales (Δη, Δφ) of real particles in the window. Only the intended particles change.
- Statistics: the jet bootstrap holds formulas fixed, so the frac-named CI is degenerate for S ([0.789, 0.789]); refit
  variance is not in any interval (stated in X11). SAE/cross-model p = 0.002 is the floor for 500 resamples. Holm over 14 tests
  is applied correctly (`holm`). The causal-fraction comparison with rotations counts passes over all directions, so it inherits
  the naming excess (pass 1: conditional on being named, 0.47 native vs 0.50 rotations).
- Figure `results/jedi/x11_atlas.png` (read): matches the summary numbers; the PFN native marker is hidden under the PCA/SAE
  markers in both panels. No number changes.
- No bug found that changes a reported number.

## Procedure (pass 2)
- Jets: fit = `sample(3000, 401)` (15,000; permutation rng(0): 10,000 fit-train / 5,000 fit-dev); test = `sample(3000, 409)`
  minus 859 jets also in the fit sample → 14,141. Library, fitter (4,000 selection jets), naming rule, live-unit rule as X11.
  Bootstrap B = 300 test-jet resamples (rng 11), formulas fixed.
- Part trained: JEDI-linear S (quantized network) native vs 50 raw rotations (rng 9000+i) and 10 z-scored rotations
  (rng 9100+i); Deep Sets dsets-d64 seed 0 S native vs 10 raw rotations (rng 9200+i).
- Part untrained (missing control; pass 1's quantized untrained network had 1 live unit): JEDI-linear's architecture in float64
  (same einsum equations, pool scales, input scaling; no quantizers), each fused kernel and bias of l1, dd, l2, l3 redrawn
  N(trained mean, trained s.d.), seeds 0 and 1; Deep Sets with φ₀/φ₂ redrawn the same way (seed 0). S native vs 20 raw
  rotations (+ 5 z-scored for JEDI).
- H0: the native fraction named is exchangeable with the rotation fractions; H1: native above.
  p = (1 + #{rot ≥ native}) / (n + 1).

## Results (pass 2)

| network, layer | live d | native named | rotations n, mean [min, max] | rot ≥ native | perm. p | native − rot mean [95 % CI] |
|---|---|---|---|---|---|---|
| JEDI-linear S, trained (X11: 0.79; pass 1: 0.79) | 19 | 0.789 (15/19), median R² 0.957 | 50 raw, 0.397 [0.263, 0.526] (s.d. 0.069) | 0/50 | 0.020 | +0.393 [0.381, 0.399] |
| JEDI-linear S, trained, z-scored rot. | 19 | 0.789 | 10, 0.279 [0.211, 0.368] | 0/10 | 0.091 | +0.511 [0.505, 0.516] |
| Deep Sets S, trained (X11: 0.29) | 59 | 0.288 (17/59), median R² 0.658 | 10 raw, 0.086 [0.051, 0.119] | 0/10 | 0.091 | +0.202 [0.198, 0.207] |
| JEDI arch., untrained float, seed 0 | 56 | 0.000 (0/56), median R² 0.285 | 20 raw, 0.004 [0, 0.018] | 20/20 | 1.0 | −0.004 |
| JEDI arch., untrained float, seed 1 | 57 | 0.018 (1/57), median R² 0.346 | 20 raw, 0.014 [0, 0.035] | 13/20 | 0.67 | +0.004 |
| Deep Sets, untrained, seed 0 | 46 | 0.000 (0/46), median R² 0.370 | 20 raw, 0.010 [0, 0.065] | 20/20 | 1.0 | −0.010 |

- Accuracy (pass-2 jets): JEDI-linear 80.91 % fit / 80.84 % test; Deep Sets 80.63 % (reference 80.64 %).
- Named JEDI-linear S units: S0, S1, S13, S17, S19, S20, S29, S30, S40, S43, S50, S51, S60, S62, S63, the same 15 as X11 and pass 1.
- Across the three independent jet samples the native fraction is 0.789 each time; rotations reaching it: 0/20 (X11),
  0/200 (pass 1), 0/50 (pass 2). JEDI S native is 5.7 rotation-s.d. above the rotation mean in pass 2.
- Untrained controls: trained JEDI S named 0.79 vs untrained architecture 0.00 / 0.02 (median R² 0.96 vs 0.29 / 0.35).

## Predictions of competing hypotheses
- "X11's native excess is a 20-rotation fluctuation": predicts native inside the rotation spread on new jets. Excluded
  (0/50 here, 0/200 in pass 1).
- "Units of this architecture are readable by this library without training": predicts untrained native named fraction near
  the trained one. Excluded (0/56, 1/57 vs 15/19).
- "The basis preference is a property of the per-particle ReLU-sum architecture, not learned": predicts untrained native >
  untrained rotations. Not testable here: both are at 0–0.02 (floor), so this hypothesis is neither excluded nor supported.
- "Native units follow their formulas under edits more than mixed directions do": not supported (pass 1: conditional pass
  rate 0.47 vs 0.50).

## Corrected reading (both passes)
X11's reading holds as stated, with these refinements:
1. The 20-rotation floor (p 0.048, Holm 0.29) does not limit the result: on two fresh samples 0/200 and 0/50 rotations reach the
   native JEDI-linear S fraction (p 0.005 and 0.020); h4 0/40 (p 0.024); Deep Sets S 0/20 and 0/10.
2. The causal-pass excess over rotations (0.37 vs 0.22) follows from the naming excess; conditional on being named, units and
   rotated directions pass the edit test at the same rate.
3. With a generic 380-observable library the JEDI-linear S excess remains (0.74 vs ≤ 0.58; pass 1).
4. Untrained architecture control: the hand-built library names essentially no untrained unit or rotation (≤ 0.065), so the
   readability of the native units requires training; whether the preference for the native basis over rotations is learned
   or architectural is not determined by this control.
Unchanged limits: formula-fixed intervals; SAE comparison specific to additive hinge naming (no top-k or larger SAE).

## Explainer entry (pass 2; Paper A tab, site/explain.py)
**Question.** Does the X11 result (JEDI-linear's pooled units are more often readable as short formulas than random mixings of
them) hold on a third set of jets, and does a network with the same architecture but random weights show the same thing?

**Background.** For a layer with live units $h_1,\dots,h_d$, a rotation gives directions $v = Qh$ with $Q$ Haar-random
orthogonal. A direction is "named" if a formula $\hat v = c_0 + \sum_{t\le 8} c_t B_t(\mathcal O)$, with hinge terms
$B_t = \max(0, \pm(\mathcal O_k - \theta))$ of rank-window observables such as $\sum_{i\in w} p_{T,i} r_i^2$, reaches held-out
$R^2 \ge 0.9$. The untrained control redraws each layer's weights from $\mathcal N(\mu_\text{layer}, \sigma_\text{layer})$.

**Method.** Fit on 10,000 jets, select the number of terms on 5,000, score on 14,141 new jets. Compare the native fraction
$f_\text{nat}$ with 50 rotations; $p = (1 + \#\{f_\text{rot} \ge f_\text{nat}\})/(n+1)$. Repeat on untrained JEDI-architecture
(2 seeds) and Deep Sets (1 seed) networks.

**Result.** Trained JEDI-linear S: $f_\text{nat} = 0.79$, rotations $0.40 \pm 0.07$ (max 0.53), $p = 0.020$. Deep Sets: 0.29 vs
$\le 0.12$. Untrained networks: $f_\text{nat} = 0.00, 0.02, 0.00$, rotations $\le 0.065$. Figure: `results/jedi/x11_atlas.png`
(X11; pass-2 numbers are in the table).

**Checks.** Same 15 named units on all three samples; accuracy 80.91 %; formulas held fixed in the bootstrap.

**Why it matters.** The readable-unit basis of JEDI-linear's pooled layer is reproducible and appears only after training; the
untrained control is at floor, so it does not separate a learned basis preference from one built into the architecture.

## Conclusion (table row)
On fresh jets JEDI-linear's pooled units are again readable as short formulas far more often than any random mixing of them,
and the same architecture with random weights has almost no readable units. Whether the preference for the unit basis itself
is learned is still open.

---

# Pass 1 (earlier 2026-10-19; kept as written)

Status: done. Confidence that the X11 reading is correct as stated: 0.85 (corrections below).
Code: `interp/x11_verify.py` (imports the X11 functions unchanged: `Fitter`, `library`, `edit`, `in_range`, `train_sae`, `boot_r2`).
Results: `results/jedi/x11_verify.json` (summary, Holm), `x11_verify_main.json`, `x11_verify_untrained.json`, `x11_verify_qlib.json`
(also on the volume under `/j-jepa-vol/aaron-interp/results/jedi/`). Logs: `logs/x11-verify.log`, `logs/x11-verify-ut.log`, `logs/x11-verify-q.log`.
Cluster jobs: x11-verify (1 CPU / 2 GiB, 2,398 s, peak 1.67 GiB), x11-verify-ut (323 s, 1.51 GiB), x11-verify-q (1,399 s, 1.92 GiB).

## Conclusion
On a new set of jets, JEDI-linear's pooled object units can be written as short pT/angle formulas for 79 % of units, again
above every one of 200 new random mixings of the same units, and a sparse autoencoder again gives fewer such directions. The same
ordering holds with a generic library of 380 standard jet observables that was not built from JEDI-linear's mechanism, so the
result does not come from the tailored library. The edit test does not separate the bases: among directions that are named, the
share that also follows its formula under input edits is the same for units and for random mixings (about one half). Still open:
an untrained-network control could not be run in a useful form (the untrained quantized network has one live pooled unit).

(Terms: "named" = held-out R² ≥ 0.9 for a ≤ 8-term hinge formula of the observable library; "rotation" = random orthogonal
mixing of the live units; SAE = sparse autoencoder.)

## Code review of `interp/x11_sae_atlas.py`
- Stages: S = `pool('qs1', u)` and h4 = `dense('d4', S)` from `jedi_core.Jedi.run` (after the pooling quantizer, as in the
  network). No mask or quantizer change; the forward pass is the unmodified `Jedi(64)`. Accuracy reproduced (80.87 % fit sample;
  81.25 % on the new fit sample).
- Padded slots: the library (`handbuilt`), the edits (`m &= real`), the rotation edit and `in_range` all restrict to pT > 0;
  Deep Sets masks padded particles after φ₂ (`u * m`). No padded slot is counted as a particle.
- Split: formulas are selected on 4,000 fit-train jets, refit on fit-train, K chosen on fit-dev, scored on test jets with the
  fit-sample overlap removed (2,449 jets). No leakage found.
- Edits: pT × 1.1 inside the window, then re-sorted (window membership after the edit is recomputed in the library, so the
  predicted change includes rank crossings); angle × 1.1 changes only (Δη, Δφ) of particles in the window. The edit changes
  only what is intended. The l1 range filter keeps 100 % of jets (not binding).
- Statistics: the frac-named bootstrap holds the formulas fixed (stated in the report); its CI is degenerate ([0.789, 0.789]) for
  JEDI S because no unit's R² is near 0.9. The permutation floor (0.048) is the binding limit of the original design.
- Figure `results/jedi/x11_atlas.png` (read): readable; the PFN native marker is hidden under the PCA / SAE markers at 0.10 (left)
  and under PCA at 0.03 (right); the title says "N = 25,000 minus overlap" where the test N is 22,551. Neither changes a number.
- No bug found that changes a reported number.

## Procedure (verify)
- Jets: fit = `sample(4000, 101)` (20,000; permutation rng(0): 13,333 fit-train / 6,667 fit-dev); test = `sample(4000, 103)` minus
  1,570 jets also in the fit sample → 18,430; causal = 3,000 test jets (rng 5) in JEDI-linear's l1 range under all 18 edits.
  Bootstrap B = 500 test-jet resamples (rng 11). New rotation seeds (raw 5000+i, z-scored 6000+i, h4 7000+i); SAE seeds 12 / 14.
- Part main: JEDI-linear S native vs 200 raw + 20 z-scored rotations; SAE 2× / 4× top-d (λ rule as X11); causal re-test of the
  named native S units with the X11 rule and the mismatched-formula control; h4 native vs 40 raw rotations. Library: the 97
  hand-built rank-window observables (as X11).
- Part qlib (missing control): the hand-built library's windows and thresholds (top14, r13-63, 0.165 rad, pT < 2 GeV, …) were
  chosen from JEDI-linear's own units (E3m/E5), so the "blind" score can favour JEDI-linear's basis. Rerun with the file's generic
  Q library (380 finite, non-constant observables: masses, N-subjettiness, ECFs, subjet and pT-share variables, top-k sums; no
  edits possible, naming only) for JEDI-linear S (native vs 50 raw rotations) and Deep Sets S (native vs 20 raw rotations).
- Part untrained (missing control): `Jedi(64, untrained=True, seed=0)`, S native vs 50 raw + 10 z-scored rotations.
- H0 (per row): the native fraction named is exchangeable with the rotation fractions; H1: native above. p = (1 + #{rot ≥ native}) / (n + 1).
  Holm over the 7 verify tests.

## Results (verify)

| layer, library | native named | rotations: n, mean [min, max] | rot ≥ native | perm. p | Holm (7) | native − rot mean [95 % CI] |
|---|---|---|---|---|---|---|
| JEDI S, hand-built (X11: 0.79; 20 rot 0.32–0.63) | 0.789 (15/19) | 200 raw, 0.400 [0.211, 0.632] | 0/200 | 0.005 | 0.025 | +0.389 [0.384, 0.396] |
| JEDI S, hand-built, z-scored rot. | 0.789 | 20, 0.250 [0.158, 0.316] | 0/20 | 0.048 | 0.095 | +0.539 [0.534, 0.547] |
| JEDI h4, hand-built (X11: 0.55) | 0.564 (31/55; CI 0.545–0.582) | 40 raw, 0.282 [0.200, 0.364] | 0/40 | 0.024 | 0.078 | +0.282 [0.257, 0.301] |
| JEDI S, generic Q library | 0.737 (14/19) | 50 raw, 0.408 [0.211, 0.579] | 0/50 | 0.020 | 0.078 | +0.328 [0.272, 0.336] |
| Deep Sets S, generic Q library (X11 hand-built: 0.29) | 0.271 (16/59) | 20 raw, 0.103 [0.051, 0.153] | 0/20 | 0.048 | 0.095 | +0.169 [0.165, 0.172] |

- Rotation distribution for JEDI S (200 raw rotations): s.d. 0.083; native is 4.7 s.d. above the mean (normal approximation
  p = 1.5e-6; the permutation p 0.005 is the floor for 200 rotations).
- SAE, JEDI S (fresh fit): 2× λ 0.03, dev FVU 0.044, top-19 named 0.158; 4× λ 0.03, FVU 0.041, top-19 named 0.158.
  SAE − native = −0.63 [−0.63, −0.63] and −0.63 [−0.63, −0.58], bootstrap p 0.002, Holm 0.014 (X11: −0.63 / −0.58).
- Named native S units: the same 15 units as X11 (S0, S1, S13, S17, S19, S20, S29, S30, S40, S43, S50, S51, S60, S62, S63);
  with the Q library 14 named (S1 drops to R² 0.889; S31, S52, S56, S58 not named in either library).
- Causal re-test (3,000 jets, formulas refit on the new sample): 8/15 pass (X11: 7/15). Pass: S0 0.88 [0.85, 0.92], S1 1.16
  [1.13, 1.19], S13 0.93 [0.90, 0.96] (X11 fail at 0.78 with the top-11 window), S29 0.91, S40 1.08, S43 1.08, S50 0.97, S51 0.82
  [0.81, 0.84]. Fail: S17 0.37, S19 (soft multiplicity) 0.13 [0.11, 0.15], S20 0.78 [0.76, 0.80] (X11: 0.28 under pT × 1.1 of ranks 1–63),
  S30 (Σr over all particles) 0.48, S60 0.49, S62 0.63, S63 0.45. Mismatched-formula control 0/15.
  The top term (and so the edit window) changed for S13, S19, S29, S43, S50 after refitting; per-unit pass/fail is the same as
  X11 for 14/15 units; S13 moves from fail (0.78) to pass (0.93) with a different top term.
- Causal pass among named directions (X11 data, H0: same rate in both bases): JEDI S native 7/15 = 0.47 vs raw rotations
  84/167 = 0.50 (12 of 20 rotations ≥ native, p 0.62); h4 18/30 = 0.60 vs 0.55 (p 0.48); Deep Sets 12/17 = 0.71 vs 0.68 (p 0.55);
  PFN 1/3 vs 0.85. The X11 excess of the causal fraction over rotations (JEDI S 0.37 vs 0.22) equals the naming excess times the
  same pass rate; it is not a separate basis effect.
- Untrained control: the untrained quantized JEDI-linear has 1 live S unit (S58; test accuracy 19.9 %), R² 0.018; native and
  every rotation are named 0/1. This control is degenerate and says nothing about basis preference. A float untrained network or
  X8's weight-permuted copies would be the usable versions.

## Predictions and which are excluded
- "Native basis not preferred; X11's excess is a 20-rotation fluctuation": predicts native within the rotation spread. Excluded:
  0/200 fresh rotations reach the native fraction (p 0.005, Holm 0.025 over the 7 verify tests).
- "The excess comes from a library tailored to JEDI-linear's units": predicts the excess vanishes with a generic library and the
  JEDI–Deep Sets gap shrinks. Not observed: generic library JEDI S 0.74 vs rotations ≤ 0.58 (p 0.020); Deep Sets 0.27 (hand-built 0.29),
  so the JEDI − Deep Sets gap is 0.47 (hand-built X11: 0.50).
- "Native units follow their formulas under edits more than mixed directions do" (implicit in X11's causal-fraction comparison):
  not supported; the conditional pass rates are equal within 0.05 for JEDI S, h4 and Deep Sets.

## Corrected reading
X11's reading holds on fresh jets with three changes:
1. The native-vs-rotation result is not limited by the 20-rotation floor: with 200 fresh rotations, 0 reach the JEDI-linear S
   native fraction (p 0.005, Holm 0.025 in this verifier's 7-test family; 4.7 s.d. above the rotation mean).
2. The causal-pass comparison with rotations (0.37 vs 0.22) is a consequence of more native units being named; conditional on
   being named, the edit pass rate is the same for native units and rotated directions (0.47 vs 0.50 for JEDI S, p 0.62). The edit
   test does not add evidence for a preferred basis.
3. The basis preference for JEDI-linear S and Deep Sets S, and the JEDI − Deep Sets gap, persist with a generic 380-observable
   library not derived from JEDI-linear (0.74 vs ≤ 0.58; 0.27 vs ≤ 0.15), so they are not an artefact of the hand-built windows.
   The SAE comparison was rechecked only on the hand-built library.
Unchanged limits: formula-fixed bootstrap intervals; SAE result specific to additive hinge naming; no usable untrained control.

## Explainer entry (Paper A tab; site/explain.py)
**Question.** Does X11's finding (JEDI-linear's units are more nameable than random mixings of them) hold on new jets, with more
random mixings, and with an observable library that was not built from JEDI-linear?

**Background.** X11 compared a layer's units $h_k$ with rotated directions $v = Qh$ ($Q$ Haar-random orthogonal). With 20
rotations the smallest attainable permutation p-value is $1/21 = 0.048$. X11's library of rank-window sums
($\sum_{i\in w} p_{T,i} r_i^2$ etc.) was designed from JEDI-linear's own units, which could favour its basis.

**Method.** New fit jets (20,000) and test jets (18,430). Fit $\hat v_j = c_0 + \sum_{t\le 8} c_t B_t(\mathcal O)$ by forward
selection; named if held-out $R^2 \ge 0.9$. Compare native units with 200 rotations; repeat with the generic library of 380
standard jet observables (masses, $\tau_N$, ECFs, subjet variables). Re-run the edit test: scale $p_T$ or $(\Delta\eta,\Delta\phi)$
by 1.1 in the top term's window and require the slope of $\Delta v_j$ on $\Delta\hat v_j$ to have its 95 % CI in $[0.8, 1.2]$.
$p = (1 + \#\{\text{rot} \ge \text{native}\})/(n+1)$.

**Result.** JEDI S: native 0.79, rotations $0.40 \pm 0.08$ (max 0.63), $p = 0.005$. Generic library: 0.74 vs $\le 0.58$
($p = 0.02$); Deep Sets 0.27 vs $\le 0.15$. SAE top-19: 0.16. Edit test: 8/15 named units pass; conditional on being named,
rotated directions pass at the same rate (0.50 vs 0.47).

**Checks.** Accuracy 81.25 % (fit sample); mismatched-formula control 0/15; the same 15 units named as in X11; untrained network
not usable (1 live unit).

**Why it matters.** The preferred-basis result for JEDI-linear's pooled objects is not a small-sample or library-design effect;
the edit test checks formula fidelity of named directions but does not distinguish units from mixtures.

## Conclusion (table row)
On fresh jets JEDI-linear's pooled units are named by short formulas far more often than any of 200 random mixings, also with a
generic observable library, and a sparse autoencoder does not raise the rate. Named units and named mixings follow their
formulas under edits equally often, so the edit test does not add evidence for the preferred basis; an untrained control is still missing.
