# E26w verify: is total pT (S0) a specific feature under the mass-fixed boost? (skeptic pass; claims A14 / A3)

Script `interp/e26w_verify.py` (imports E26w's `Tilt`, `iso`, `capt`, `StratAUC`, `jet_feats` and the E26 / h12 helpers unchanged).
Job `e26w-verify` (1 CPU / 2 GiB, wall 1,061 s, exit 0, 0.30 cores on average); smoke `e26w-vsmoke` passed.
Results `results/jedi/e26w_verify.json`, log `logs/e26w-verify.log`.

## Conclusion
On a fresh set of 10,000 jets, every E26w number replicates. When jet pT drops by 10 % at fixed mass, holding the network's
total-pT quantity (S0) at its old value undoes 57 % of the changed decisions. The new check changes the interpretation. Inside the
network's pooled layer, the boost moves S0 far more than anything else: S0 carries about two-thirds of the layer's total boost
displacement. As a result, no direction without S0 moves as far, and E26w's "random directions moved equally far" comparison
always contained a large S0 component. That comparison therefore cannot show that the effect is not specific to total pT.
Against random directions of equal data spread that exclude S0, holding S0 removes far more (57 % vs at most 37 %). E26w's
statement that the effect "tracks the size of the boost displacement rather than anything unique to total pT" is not supported.
The supported reading is that the boost reaches this layer mainly through S0. Still uncertain: at the opposite boost (pT up 10 %),
holding S0 undoes nothing.

## What was checked
- **Network:** JEDI-linear 3-feature n64 (official checkpoint). Layers as in `interp/jedi_core.py`:
  - S: the 64 pooled second-round objects, 19 of them live (non-constant);
  - g: context units;
  - h4, h5: dense layers after S.
- **Units by what they compute:**
  - S0 = sum of pT of the 14 hardest particles;
  - S29 = pT-weighted Σr² plus soft terms (an angular-spread object);
  - g49 = context unit fed by summed pT;
  - a15 = first-layer cut linear in particle pT.
- **Fresh sample:** `sample(per_class=2000, seed=2611)`, N = 10,000.
  - The null geometry (S covariance Σ, boost-displacement second moment C_D, principal directions) is fit on `sample(2000, seed 2612)`.
  - This sample shares 385 jets with E26w's evaluation sample and 369 with its own fit sample (both about 4 %).
  - Unedited accuracy 0.8105, and the exact fixed-point tail reproduces the network (max |Δlogit| 0).
- **Edit B:** the exact Lorentz boost along the jet momentum, imported from `interp.h12_boost_panel`.
  - It scales |P_jet| by λ and keeps the jet invariant mass fixed, so particle angles to the axis scale by about 1/λ.
  - λ 0.9: jet ΣpT −10 %, and 33.1 % [32.2, 34.0] of calls change.
  - λ 1.1: 14.7 % [14.1, 15.4] of calls change.
- **Hold:** the network sees the boosted jet, and the component of S along a subspace Q is reset to its unboosted value:
  S_hold = S_B + (S_0 − S_B) QQᵀ, clipped to the representable range.
- **Metric:** removed share = 1 − (calls still changed with the hold) / (calls changed by B alone). It is a net share and can be
  negative.
- **Definitions:**
  - C_D = E[D Dᵀ] with D = S(boosted) − S(unboosted), so C_D[u,u] is the mean squared boost displacement of unit u.
  - Shift-matched null: random live directions q tilted toward C_D so that qᵀC_D q equals S0's value. "Moved as far by the boost as S0."
  - Variance-matched null: the same construction with Σ. "Same data spread as S0."
- **Statistics:** 1,000 Poisson bootstraps over jets, 500 null draws per family, rank p = (1 + #null ≥ obs) / (1 + #null).

### Code review of `interp/e26w_boost_specificity.py`
- No bugs that change the numbers:
  - the stage is right (S after pooling, exact quantized tail);
  - live coordinates are taken from unedited and boosted jets;
  - the hold changes only the subspace;
  - padded slots carry pT 0 and are excluded from the boost;
  - Δm/m ≤ 1e-11;
  - the fit and evaluation samples are separate draws (4 % overlap, stated).
- The shift-matched null is built correctly. Its interpretation is the problem (below).
- The figure is consistent with the JSON. Its title ("not more than random directions that move as much under the boost")
  carries the unsupported reading.

## 1. Replication on the fresh sample (E26w → verify)
| hold | λ 0.9 E26w | λ 0.9 verify [95 % CI] | λ 1.1 E26w | λ 1.1 verify |
|---|---|---|---|---|
| S0 | 55.9 % | 56.8 % [55.0, 58.6] | 0.1 % | −0.6 % [−6.8, 4.7] |
| PC1 (top-variance direction) | 2.6 % | 6.2 % [3.9, 8.6] | −129.5 % | −123 % |
| S0+S29 | 77.7 % | 78.2 % [76.7, 79.7] | 51.0 % | 53.1 % |
| DAS M0, 1-dim | 61.5 % | 61.8 % [60.0, 63.5] | 3.9 % | 3.7 % |
| DAS M0+M2, 2-dim | 84.3 % | 84.0 % [82.5, 85.3] | 54.8 % | 55.4 % |

- DAS (distributed alignment search) is a learned subspace of S that transfers an M0-only or M2-only edit, where M0 = ΣpT and
  M2 = ΣpT r². The refit DAS M0 direction has cos² 0.99 with the S0 axis, so on this network DAS M0 and S0 are the same direction.
- S0 − PC1 at λ 0.9: +50.6 pp [47.6, 53.6]. McNemar on the remaining changes: 749 changes remain only with S0 and 2,425 only
  with PC1, p 3e-204.
- Nulls for S0 at λ 0.9:

  | null | median | 97.5 % | max | p |
  |---|---|---|---|---|
  | iso | | 33.1 % | | 0.002 |
  | var | | 50.0 % | 74.4 % | 0.022 (E26w 0.018) |
  | shift | 61.5 % | 79.6 % | | 0.68 (E26w 0.72) |

- λ 1.1: S0 sits inside the iso and var nulls (p 0.18 / 0.17).

## 2. New control: can a boost-shift-matched direction avoid S0?
The question is whether E26w's shift-matched null tests the alternative "any direction that the boost moves this far removes as
much", independently of S0.

C_D geometry (fit sample):

| | λ 0.9 | λ 1.1 |
|---|---|---|
| C_D[S0,S0] | 29.2 | 30.2 |
| next units | S29 6.05, S1 3.44, S58 1.73 | S29 3.67, S58 1.84, S1 1.76 |
| S0's share of tr C_D | 66 % | 74 % |
| cos² of C_D's top eigenvector with S0 | 0.77 | 0.86 |
| largest qᵀC_D q over live directions orthogonal to S0 | 10.8 | 6.6 |

- **S0-free shift-matched null** (the same target, drawn on the live coordinates without S0): 0 of 1,500 attempts reached the
  target at either λ. No live S direction without S0 moves under the boost by as much as S0, and the maximum reachable is 37 %
  (λ 0.9) or 22 % (λ 1.1) of S0's value.
- **S0 content of E26w's shift-matched draws:** cos² with the S0 axis has 5th percentile 0.33 and median 0.61 at λ 0.9, and
  5th percentile 0.56 at λ 1.1. No draw has cos² < 0.25. The draws with cos² ≥ 0.5 remove a median 66.2 %; those with cos² 0.25–0.5
  remove 58.4 %.
- **S0-free variance-matched null** (data variance equal to Var(S0) = 29.6, S0 excluded):
  - λ 0.9: 97.5 % 25.6 %, max 37.0 %. S0's 56.8 % is above all 500 draws (p 0.002, the floor).
  - λ 1.1: 97.5 % 6.6 %; S0 −0.6 % (p 0.19).
- **Every live S unit held alone (λ 0.9):**
  - S0 56.8 %;
  - all 18 other live units −9.5 % to 7.5 % (S30 7.5 %; S29 4.7 % with C_D 6.05; S1 4.8 % with C_D 3.44).
  - At λ 1.1: S0 −0.6 %, and the others −18 % to 4.7 %.

Hypotheses at λ 0.9:
- H0 (E26w's alternative): any S direction moved this far by the boost removes as much as S0. It cannot be tested at S, because
  no S0-free direction is moved this far. It is neither supported nor excluded.
- H0′: any S0-free direction of S0's data variance removes as much. Excluded (p 0.002).
- H1: the boost's effect at S runs mainly through S0. Consistent with all of the following:
  - S0 holds 66 % of the displacement energy;
  - S0 alone removes 57 %;
  - no other unit removes more than 7.5 %;
  - adding S29's correlated movement (S0+S29) brings it to 78 %.

The shift-matched null's p 0.68 means that S0 mixed with its correlated co-movers (S29, S1) removes as much as S0 or more.
It does not mean that S0 is interchangeable with other directions.

## 3. Path patch a15 → g49 → S0 (replicated)
| λ 0.9 | E26w | verify |
|---|---|---|
| ΔS0 (mean) | −5.30 | −5.31 |
| share via g49 | 68 % | 68.1 % (−3.61), direct −1.65 |
| S0 with the g49 route reverted | 50.0 % | 51.5 % [49.6, 53.4] |
| S0 with the direct route reverted | 20.6 % | 21.8 % [20.2, 23.5] |
| a15 held | 41.1 % | 41.1 % [39.1, 43.2] |
| a15 not via g | 8.4 % | 8.8 % [7.0, 10.3] |
| ΔS0 with a15 held | −0.31 | −0.32 |

New checks:
- Holding g49 changes no other S unit: max |ΔS_u| over u ≠ 0 is exactly 0.
- With S0 then set back to its boosted value, 0 of 10,000 calls differ from B.
- So g49 reaches the calls only through S0. λ 1.1 gives the same 69 / 31 % split.

## 4. Single h4 / h5 units vs live isotropic nulls (2,000 draws, Holm over the layer's live units)
- **λ 0.9:** h4 53 45.7 %, h4 39 36.2 % (Holm 0.027 each), h5 24 42.6 % (Holm 0.016). This replicates E26w.
- **λ 1.1:**
  - h4 39 28.2 % and h4 3 24.8 %, both Holm 0.055 (null max 30.1 %). They do not pass at 0.05 on this sample, so E26w's λ 1.1 h4
    result sits at the edge of its null (E26w null max 26.5 % vs 26.2 % / 23.7 %).
  - h5 25 34.8 % (Holm 0.032) replicates.
- What these units compute is not named.

## 5. Data check: does the W vs Z number survive narrower mass bins? (n64 test set, 260k jets, 100 bootstraps)
The stratified AUC is the probability that a jet of the first class has the larger value than a jet of the second class in the
same jet-mass bin.

| pair | ΣpT 14 hardest, 5 GeV | ΣpT 14 hardest, 1 GeV | ΣpT all, 1 GeV | in-bin mass AUC 5 → 1 GeV |
|---|---|---|---|---|
| W vs Z | 0.584 [0.574, 0.592] | 0.566 [0.556, 0.579] | 0.637 [0.627, 0.647] | 0.322 → 0.475 |
| W vs t | 0.731 | 0.725 [0.716, 0.733] | 0.843 | 0.430 → 0.476 |
| Z vs t | 0.735 | 0.738 [0.729, 0.748] | 0.836 | 0.507 → 0.501 |
| q vs g | 0.555 | 0.555 [0.550, 0.559] | 0.324 | 0.475 → 0.497 |

- With 1 GeV bins the residual mass separation nearly disappears (W vs Z 0.475), and the W vs Z ΣpT AUC changes only from 0.584 to
  0.566. E26w's caveat 3 (residual mass in 5 GeV bins) therefore accounts for little of the W vs Z number.
- This is a data correlation, not a test of the network.
- E26w's thesis note says the check "excludes" a training artefact. The data check shows only that ΣpT carries class information
  at fixed mass in this dataset, which is consistent with the network using it.

## Corrected reading
- At λ 0.9, holding S0 removes 57 % of the boost's call changes, more than any S0-free direction of equal data variance (max 37 %,
  p 0.002) and more than any other single live S unit (≤ 7.5 %).
- The boost's displacement at S is concentrated in S0 (66–74 % of tr C_D), so a displacement-matched null cannot be drawn without
  S0. "Specific to S0" and "follows the boost displacement" cannot be separated at S.
- The supported statement is that the boost's effect at S runs mainly through S0, and S0's change arrives 68 % via g49
  (a15 → g49 → S0; g49 acts only through S0).
- At λ 1.1, holding S0 removes nothing, so the S0 route accounts for the pT-lowering direction only.
- The λ 1.1 single-h4 Holm passes are at the edge of the null (fresh Holm 0.055).

## Explainer entry (for site/explain.py; ledger agent to place)
**Question.** E26w compared the S0 hold with random directions that the boost moves as far as S0. Does that comparison show the
effect is not specific to total pT?

**Background.** The boost-displacement second moment is $C_D = \mathbb{E}[DD^\top]$, with $D = S_B - S_0$. A direction $q$ is
displaced as much as S0 when $q^\top C_D q = (C_D)_{00}$.

**Method.**
- Draw random unit directions on the live S coordinates with S0 excluded ($q_0 = 0$) and try to meet that target.
- Separately, draw S0-free directions with $q^\top\Sigma q = \Sigma_{00}$ and hold each:
  $S_{\rm hold} = S_B + (S_0 - S_B)qq^\top$.

**Result.**
- $\max_{q\perp e_0} q^\top C_D q = 10.8$ against $(C_D)_{00} = 29.2$, so no S0-free direction is displaced as far as S0.
- S0 holds $66\%$ of $\mathrm{tr}\,C_D$.
- S0 removes 57 % of the call changes, while S0-free equal-variance directions remove at most 37 %.

**Why it matters.** The boost acts on the pooled layer almost entirely through total pT. A null matched on displacement therefore
re-samples S0 itself and cannot test whether S0 is specific.

**Glossary additions:**
- *C_D (boost-displacement second moment)*: E[(S_B − S_0)(S_B − S_0)ᵀ] over jets.
- *Shift-matched null*: random directions with the same C_D projection as the named one.
- *S0-free null*: random directions with zero S0 component.
