# X11 — Preregistered blind atlas: JEDI-linear units vs SAE features vs random rotations vs Deep Sets / PFN units (claim A11; review weaknesses 4 and 2)

Status: done.
Code: `interp/x11_sae_atlas.py` (per-model run and `--combine`). Preregistration: `interp/x11_prereg.json`, copied to
`results/jedi/x11_prereg.json` before any fit; every run checked the volume copy against the repo copy ("preregistration unchanged" in each log).
Results: `results/jedi/x11_jedi.json`, `x11_dsets.json`, `x11_pfn.json` (every direction: formula terms, held-out R², edit, slope + CI),
`x11_summary.json` (table, headline tests, Holm), figure `results/jedi/x11_atlas.png`; smoke run `x11_jedi_smoke.json`.
Logs: `logs/x11-smoke.log`, `logs/x11-run-{jedi,dsets,pfn}.log`, `logs/x11-combine.log`.
Cluster: jobs x11-run-jedi / -dsets / -pfn (1 CPU, 3 GiB; peak 1.8 / 2.3 / 1.5 GiB; 797 / 1284 / 313 s), x11-combine (1 CPU, 2 GiB, 119 s).
sklearn was not needed (MARS = the repo's `jetdistill.mars` candidates/knots; SAE in jax on CPU).

## Conclusion
JEDI-linear's pooled object units can each be written as a short formula of pT and angular-distance sums over pT-rank
windows for 15 of 19 units (79 %), more often than for any of 20 random rotations of the same activations (best rotation 63 %).
Sparse-autoencoder features of the same activations are named far less often (16–21 %), so on this observable library an
SAE does not make the representation more nameable; the same holds for Deep Sets and the PFN. Deep Sets units are also more
nameable than their rotations (29 % vs at most 12 %), so a privileged basis is not unique to JEDI-linear, while the PFN latent
is not. Open: about half of the named JEDI-linear units (7 of 15) also pass the edit test, and the rotation p-value (0.048) is the
smallest possible with 20 rotations and does not survive Holm correction over the 14 headline tests.

(Terms: "named" = the unit's value on held-out jets is reproduced with R² ≥ 0.9 by a ≤ 8-term hinge formula of the fixed observable
library; "rotation" = a random orthogonal mixing of the live units, which keeps all information but destroys the unit basis;
SAE = sparse autoencoder, a learned overcomplete sparse dictionary of directions.)

## Procedure
- Models and layers. JEDI-linear official 3-feature n64 (`interp.jedi_core.Jedi(64)`): S (pooled second-round objects, 64) and
  h4 (first MLP layer, 64). Deep Sets dsets-d64 float seed 0: S = Σ over particles of φ₂ (64). PFN (64, 64, 32) seed 0 from X9:
  latent L (32). Live units: s.d. > 1e-6 and > 1e-6 on ≥ 0.5 % of fit-train jets → JEDI S 19, h4 55, Deep Sets S 59, PFN L 30.
- Jets. Fit = `sample(5000, 0)` (25,000; fixed permutation: 16,666 fit-train, 8,334 fit-dev for term-count selection).
  Test = `sample(5000, 29)` minus the 2,449 jets also in the fit sample → 22,551. Causal = 5,000 test jets (rng 5), all inside
  JEDI-linear's l1 input range under all 18 edits (100 % of candidates were in range); the same jets for all three models.
- Library. The E11 hand-built rank-window observables (`interp.e11_mars_all.handbuilt`: per window N, ΣpT, ΣpT·r², Σr², Σr, ΣpT·r,
  Σ(r−0.2)₊, N(pT>1), N(pT<2), Σr[r<0.165], N(pT<2, r>0.1) over windows all, top3, top11, top14, top26, top39, r1-63, r8-63, r13-63;
  |M1|, √(M0M2)) + ΣpT[all], N[all]: 97 observables, 3,103 candidate columns (linear + hinges at 19 knots each). pT in GeV, r in rad.
- Bases per layer. (a) native live units; (b) 20 Haar-random rotations of the raw live activations (rng 100+i; primary null);
  (b') 10 rotations of the z-scored activations (secondary null); (c) PCA directions; (d) ReLU SAE, dictionary 2d and 4d, input
  centred and divided by the rms s.d., L1 penalty λ = largest of {0.3, 0.1, 0.03, 0.01, 0.003} with fit-dev FVU ≤ 0.10
  (Adam 1e-3, batch 512, 3,000 steps, 1 CPU). SAE scored as the top d features by mean f² (primary) and all features alive on
  ≥ 0.5 % of jets (secondary). Chosen λ / dev FVU / mean active features: JEDI S 0.03 / 0.04 / 10–13; JEDI h4 0.03 / 0.09 / 11–13;
  Deep Sets 0.01 / 0.035 / 27–30; PFN 0.03 / 0.07 / 11–15 (all chosen by the rule).
- Naming score (blind, code only). Forward selection as `jetdistill.mars.fit_neuron` (linear terms, hinges, products of a chosen
  hinge with hinges of the 10 observables most correlated with the residual) on 4,000 fit-train jets, least-squares refit on all
  fit-train jets, ≤ 8 terms, term count K = smallest with fit-dev R² within 0.005 of the best. Score = held-out R² on the 22,551
  test jets; named iff R² ≥ 0.90. Description length = K and atoms (1 per linear term, 2 per hinge).
- Causal score. For each named direction: the term with the largest |coef|·s.d.; its rank window w; edits on the causal jets:
  pT × 1.1 of the real particles in w (re-sorted) or (Δη, Δφ) × 1.1 of the particles in w — the one with the larger predicted
  change. Predicted change = formula(edited) − formula(original); actual change = the network's direction value on the edited
  jet − original. OLS slope of actual on predicted, 95 % CI from 500 jet resamples (rng 13); pass iff the CI lies inside [0.8, 1.2].
- Controls. Random rotations (above); mismatched formula (another named direction's formula, derangement rng 17, same edit,
  same pass rule); azimuthal rotation of every particle about the axis (all library observables unchanged, max change 2.2e-6 s.d.).
- Statistics. Fraction-named CI: 500 jet resamples of the test jets (rng 11, shared by all jobs, so model contrasts are paired);
  this interval covers test-jet sampling with the fitted formulas held fixed, not refit variance. native − rotation: permutation
  p = (1 + #{rotation ≥ native}) / 21 (floor 0.048). SAE/PCA − native: paired bootstrap, two-sided p (floor 0.002). Holm over
  the 14 headline tests. Per-direction BH (named; slope equivalence) reported as secondary counts.
- Accuracy check (unedited networks on the fit sample): JEDI-linear 80.87 % (reference 80.9 %; required 80.6–81.3 %), Deep Sets
  80.57 % (ref. 80.64 %), PFN 81.30 % (ref. 81.23 %). Test sample: 80.33 / 80.22 / 80.68 %.

## Results (test jets N = 22,551; causal jets N = 5,000)

Fraction of directions named (held-out R² ≥ 0.9), with the 20 raw rotations as range; causal = named and slope CI in [0.8, 1.2]
(fraction of all directions).

| layer (d) | native | rotations raw: mean [min, max] | rot. z-scored mean | PCA | SAE 2× top-d | SAE 4× top-d | native causal | rotations causal mean [max] |
|---|---|---|---|---|---|---|---|---|
| JEDI S (19) | **0.79** (15/19) | 0.44 [0.32, 0.63] | 0.28 | 0.11 | 0.16 | 0.21 | 0.37 (7) | 0.22 [0.32] |
| JEDI h4 (55) | **0.55** (30/55; CI 0.53–0.56) | 0.30 [0.20, 0.36] | 0.05 | 0.07 | 0.09 | 0.11 | 0.33 (18) | 0.16 [0.25] |
| Deep Sets S (59) | **0.29** (17/59) | 0.08 [0.00, 0.12] | 0.01 | 0.03 | 0.07 | 0.03 | 0.20 (12) | 0.05 [0.10] |
| PFN L (30) | 0.10 (3/30) | 0.12 [0.07, 0.20] | 0.01 | 0.10 | 0.10 | 0.10 | 0.03 (1) | 0.10 [0.17] |

All-alive SAE features (secondary): JEDI S 3/38 and 4/73; h4 5/109 and 6/141; Deep Sets 4/118 and 3/221; PFN 4/60 and 3/111.
Median held-out R² per direction: native JEDI S 0.955, h4 0.906, Deep Sets 0.663, PFN 0.694; raw rotations (median of medians)
0.885 / 0.858 / 0.697 / 0.709; SAE 4× top-d 0.786 / 0.691 / 0.416 / 0.506.
Description length of named directions (mean K terms / atoms): JEDI S native 5.8 / 12.9, rotations 6.5 / 12.6; h4 native 6.1 / 16.4,
rotations 7.2 / 15.3; Deep Sets native 5.7 / 11.6; SAE 4× 6.5–7.7 / 18–22.

Headline tests (difference in fraction named; 95 % jet-bootstrap CI; p; Holm over 14):

| test | diff | CI | p | Holm |
|---|---|---|---|---|
| JEDI S native − raw rotations | +0.35 | [0.34, 0.36] | 0.048 (perm., 0/20 rotations ≥ native) | 0.29 |
| JEDI h4 native − raw rotations | +0.25 | [0.23, 0.27] | 0.048 (0/20) | 0.29 |
| Deep Sets S native − raw rotations | +0.21 | [0.21, 0.21] | 0.048 (0/20) | 0.29 |
| PFN L native − raw rotations | −0.02 | [−0.02, −0.01] | 0.71 (14/20 ≥ native) | 1.0 |
| JEDI S SAE 2× − native | −0.63 | [−0.63, −0.63] | 0.002 | 0.028 |
| JEDI S SAE 4× − native | −0.58 | [−0.58, −0.53] | 0.002 | 0.028 |
| JEDI h4 SAE 2× / 4× − native | −0.45 / −0.44 | [−0.49, −0.42] / [−0.45, −0.42] | 0.002 | 0.028 |
| Deep Sets SAE 2× / 4× − native | −0.22 / −0.25 | point | 0.002 | 0.028 |
| PFN SAE 2× / 4× − native | 0.00 / 0.00 | [0.00, 0.03] | 1.0 | 1.0 |
| JEDI S native − Deep Sets S native | +0.50 | point | 0.002 | 0.028 |
| JEDI S native − PFN L native | +0.69 | point | 0.002 | 0.028 |

Causal detail, JEDI S native (unit: top term → edit, slope [95 % CI]):
pass — S0 (ΣpT[all] with hinges; R² 0.990) pT×1.1 all 0.85 [0.83, 0.88]; S1 (Σ(r−0.2)₊ over ranks 8–63; 0.955) r×1.1 ranks 1–63
1.16 [1.14, 1.18]; S29 (ΣpT·r² ranks 8–63 + Σr; 0.985) 0.91 [0.90, 0.92]; S40 (ΣpT·r² top 14; 0.928) 1.05 [1.02, 1.08];
S43 (ΣpT top 3; 0.963) 1.06 [1.04, 1.09]; S50 (ΣpT·r² top 39; 0.970) 0.94 [0.92, 0.96]; S51 (ΣpT top 3 hinge; 0.971) 0.84 [0.83, 0.85].
fail (named, slope outside) — S13 0.78, S17 0.32, S19 (soft multiplicity, R² 0.995) 0.13, S20 0.28, S30 (Σr all, R² 0.989) 0.48,
S60 0.54, S62 0.70, S63 0.60. Not named: S31 (R² 0.25), S58 (0.25), S52 (0.85), S56 (0.83).
Mismatched-formula control: native passes 0/15 (JEDI S), 1/30 (h4), 4/17 (Deep Sets), 0/3 (PFN); raw rotations 11/167, 11/324,
3/89, 8/71. Azimuthal rotation of the jet (library unchanged): median |change| / |change under the chosen edit| = 0.43 (JEDI S),
0.38 (h4), 0.49 (Deep Sets), 1.59 (PFN) — the directions respond to an edit the library cannot see.

## Predictions and which are excluded
- Thesis (JEDI native ≥ 0.6 and above every rotation; SAE adds ≤ 0.1; Deep Sets / PFN native below their SAE and within rotation range):
  JEDI part observed (0.79; 20/20 rotations below; SAE − native = −0.58 to −0.63). Deep Sets / PFN part not observed: Deep Sets native
  (0.29) is above its SAE (0.03–0.07) and above all 20 rotations; PFN native (0.10) equals its SAE (0.10) and lies inside the rotation
  range.
- Alternative "sum pooling makes any basis equally nameable" (native within rotation range): excluded at the permutation floor for
  JEDI S, JEDI h4 and Deep Sets S (unadjusted p 0.048; Holm 0.29); not excluded for PFN L.
- Alternative "Deep Sets / PFN native equal to JEDI native": excluded (diff +0.50 and +0.69, Holm 0.028).
- SAE-dictionary hypothesis (an SAE recovers more nameable directions than the native units): excluded for JEDI S, h4 and Deep Sets
  (Holm 0.028); for PFN no difference.

## Reading (scoped to these measurements)
- On this library, JEDI-linear's S and h4 native units are named more often than rotations of the same activations, and named
  directions have a similar description length in both bases (S: 5.8 vs 6.5 terms, 12.9 vs 12.6 atoms). The excess over rotations persists at h4 (0.55 vs ≤ 0.36).
- The raw-rotation null is partly nameable (0.32–0.63 for S) because raw rotations are dominated by the few high-variance units
  (S0 = ΣpT); the z-scored rotations, which weight all units equally, are lower (0.16–0.37 for S, ≤ 0.07 for h4).
- The SAE features score lower than the native units for every model. The SAE outputs are non-negative and zero on a large share of
  jets, which a ≤ 8-term additive hinge formula fits less well; the test therefore measures nameability by this library, not
  monosemanticity in general. A different SAE (top-k activation, larger dictionary, longer training) was not tried.
- Naming does not imply the edit test: 7 of 15 named JEDI S units pass, 8 have slope CIs outside [0.8, 1.2] (e.g. soft multiplicity
  S19 at 0.13 under pT ×1.1 of ranks 8–63, where re-sorting and the soft-pT ramp a32 make the effect of a 10 % pT change on counts
  nonlinear). The causal fraction is above the raw-rotation mean for JEDI S (0.37 vs 0.22, p 0.048) and h4 (0.33 vs 0.16, p 0.048).
- The jet-bootstrap CIs are narrow because the formulas are held fixed; they do not include variability from refitting formulas or
  from the choice of rotation seeds beyond the 20 drawn. The minimum attainable permutation p with 20 rotations is 0.048.
- For the "already clean units" framing (weakness 4): JEDI native units are nameable at 0.79 (S) / 0.55 (h4) with an SAE not adding
  named directions. For weakness 2: Deep Sets units are also above their rotations (at a lower rate, 0.29), so a basis preference is
  not specific to JEDI-linear's architecture; the PFN latent shows none.

## Checks
- Preregistration written before the fits and checked unchanged in all four jobs.
- Unedited JEDI-linear accuracy on the fit sample 80.87 % (inside 80.6–81.3 %); Deep Sets and PFN match their references within 0.1 pt.
- Fit/test disjointness: 2,449 overlapping test jets removed. Fit sample pT-sorted (fraction 1.000).
- Azimuthal-rotation edit changes every library observable by ≤ 2.2e-6 s.d. (the library is rotation invariant as intended).
- Smoke run (600/400 jets per class) completed before the full runs.
- Deviation from the computed plan: SAE trained with plain jax Adam (not sklearn); 10 z-scored rotations added as a secondary null;
  the jobs requested 3 GiB (peak 1.5–2.3 GiB) instead of 2 GiB.

## Explainer entry (Paper A tab; site/explain.py)
**Question.** Are JEDI-linear's hidden units individually interpretable as physics objects, or would any mixing of the same
activations (a random rotation) or a sparse-autoencoder dictionary be equally or more interpretable?

**Background.** A layer with $d$ live units $h\in\mathbb R^d$ can be read in any basis $v = Q h$ with $Q$ orthogonal; a random $Q$
keeps all information. If units are privileged, single units should be describable by short formulas more often than rotated
directions. A sparse autoencoder learns $f=\mathrm{ReLU}(W_e(x-b_d)+b_e)$, $\hat x = W_d f + b_d$, minimising
$\lVert x-\hat x\rVert^2/d+\lambda\sum_k f_k$; its features are the usual alternative basis in mechanistic interpretability.

**Method.** For every direction $v_j$, fit $\hat v_j=c_0+\sum_{t\le 8} c_t B_t(\mathcal O)$ with $B_t$ linear terms or hinges
$\max(0,\pm(\mathcal O_k-\tau))$ (and their products) of 97 rank-window observables $\mathcal O$ such as
$\sum_{i\in w}p_{T,i}r_i^2$ and $\sum_{i\in w}(r_i-0.2)_+$, on 16,666 jets; call $v_j$ named if
$R^2=1-\mathbb E[(v_j-\hat v_j)^2]/\mathrm{Var}(v_j)\ge 0.9$ on 22,551 held-out jets. For named directions, scale $p_T$ or
$(\Delta\eta,\Delta\phi)$ by 1.1 inside the top term's window and regress the network's change $\Delta v_j$ on the formula's
change $\Delta\hat v_j$: pass if the 95 % CI of the slope lies in $[0.8,1.2]$. Compare native units, 20 random rotations,
PCA and SAE features (2× and 4× dictionaries) for JEDI-linear S and h4, a Deep Sets pooled layer and a PFN latent.

**Result.** Figure `results/jedi/x11_atlas.png`. Fraction named: JEDI S 0.79 (rotations 0.32–0.63, SAE 0.16–0.21), JEDI h4 0.55
(rotations 0.20–0.36, SAE 0.09–0.11), Deep Sets 0.29 (rotations 0.00–0.12, SAE 0.03–0.07), PFN 0.10 (rotations 0.07–0.20,
SAE 0.10). Edit test passed by 7 of 19 JEDI S units (rotations 0.05–0.32 of directions).

**Checks.** Preregistered rule and threshold; accuracy 80.87 %; disjoint fit/test jets; mismatched-formula control passes 0/15
for JEDI S; permutation test over rotations (p = 0.048, the minimum for 20 rotations; Holm 0.29); paired bootstrap for SAE −
native (Holm 0.028).

**Why it matters.** On this observable library JEDI-linear's units are a preferred basis (more nameable than rotations) and an
SAE does not add nameable directions; the Deep Sets pooled layer also has a preferred basis at a lower rate, while the PFN
latent does not. Half of the named JEDI units also follow their formula under input edits.

## Conclusion (table row)
JEDI-linear's pooled object units are named by short pT/angle formulas for 79 % of units, more than any random rotation of the
same activations, and a sparse autoencoder does not raise that rate; Deep Sets units are also above their rotations (29 %), the
PFN latent is not. About half of the named JEDI units also follow their formula under input edits, and the rotation test sits at
the minimum p-value attainable with 20 rotations.
