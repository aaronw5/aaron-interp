# H1 verify: IRC-safety panel (collinear split, soft emission)

## Conclusion
On new jet samples, every trained tagger again changes 1–5 % of its calls when its hardest particle is replaced by two exactly collinear pieces, while the IRC-safe energy-flow network changes none of the calls on jets that keep all their particles. In seven of the eight taggers the split changes more calls than moving the same momentum to a neighbour. The soft-emission part does not show particle counting in the plain sense. The added particles are softer than any particle in the training data (the smallest is 0.25 GeV in hls4ml and 0.12 GeV in JetClass), and the response depends strongly on where the particle is placed. For ParT_kin and ParticleNet_kin the response grows as the particle gets softer (ParT_kin: 3.1 % of calls at 10⁻⁴ of the jet pT, 17 % at 10⁻⁶). ParT_full's soft response stops falling at about 0.55 % of calls. Two things remain open. The evidence that the taggers use multiplicity rests on the collinear splits and on removing the softest particle, not on soft emission. Why ParT_kin changes about 1.2 % of calls whenever the hardest particle moves at all has not been tested.

## What was checked
Code review of `interp/h1_irc_panel.py`, a rerun on fresh samples, and new controls in `interp/h1_verify.py` (which imports the H1 edit functions unchanged) and `interp/h1_verify_ptmin.py`.

Jobs: `d-h1-verify` (GPU: ParT_full, ParT_kin, PN_full, PN_kin), `d-h1-verify-s` (conda: Deep Sets, Linformer, tf-2b-d64-e100 and the H1-trained EFN), `d-h1-verify-j` (jax: JEDI-linear n64) and `d-h1-verify-p` (particle-pT floors). Results are in `results/physics/h1_verify_{jedi,small,jetclass}.json`, `h1_verify_ptmin_jetclass.json` and `h1_verify.json` (summary; Holm over all 90 edit-vs-control tests).

Samples:
- hls4ml: `hls_sample(2000, seed 411)`, 10,000 jets, 64 slots; 22.65 % of the jets fill all 64 slots.
- JetClass: `jc_sample(1000, seed 412)`, random rows of each class's file, 10,000 jets.

H1 used seeds 401 and 402.

### Code review
- **Edits.** At ΔR = 0 the split conserves the four-momentum exactly. It keeps the daughter mass ratio (z m, (1−z) m), so E₁ = z E. The finalize step re-sorts the particles, cuts them to the slot limit and zeroes the padding. Recoil, reshuffle and merge conserve the 3-momentum. I found no padded-particle leaks: real particles are pT > 0, the `mask` is rebuilt from pT, and ParT `build` re-sorts and recomputes jet_pt and jet_energy.
- **Rank and recoil.** `rank_slot` and `soft()` assume that slot 0 holds the hardest particle. Measured on the new samples, slot 0 is the hardest in 100 % of jets and pT is non-increasing in 100 % of jets, in both datasets. The assumption holds.
- **Row selection.**
  - JetClass rows are random per class (`rng.choice`), not first-k.
  - The observables use rows `[::10]` of the class-concatenated arrays, an even stride over all classes. They are computed on the same edited tables the models see; the JetClass second pass repeats the edit with the same seed.
- **Charge.** In JetClass both daughters keep the parent's charge and impact parameters, so a split of a charged particle adds one track and one unit of charge. H1's charged-to-neutral relabel control covers this for ParT_full and PN_full (5.26 against 5.63 %, p 0.32).
- **Bootstrap.** `boot_binary` uses a parametric bootstrap (binomial draws at the observed rate). For a single proportion it is equivalent to resampling jets.
- **Figure** (`h1_panel.png`). The panels match the JSON. In the right panel the EFN (dashed) and ParticleNet_kin (solid) are drawn in the same grey.

### Null floors (new)
| model | same input twice | finalize only (re-sort + cut) | batch 97 instead of 256 |
|---|---|---|---|
| all 9 trained models | 0.00 % | 0.00 % | 0.00 % (JetClass models) |

Unedited accuracy on the new samples [95 % CI]:
- JEDI 81.16 [80.42, 81.94]
- Deep Sets 81.14
- Linformer 81.25
- transformer 81.71
- EFN 72.05
- ParT_full 86.25 [85.55, 86.92]
- ParT_kin 74.19
- PN_full 84.63
- PN_kin 70.35

## Rerun of the key H1 numbers (fresh samples)
% calls changed [95 % CI]. The control is C1 reshuffle (the same momentum fraction moved to the nearest neighbour). Holm is over 90 tests.

| model | split r0 z0.05 ΔR0 (H1) | verify | control | edit − control [CI] | Holm p |
|---|---|---|---|---|---|
| JEDI | 2.60 | 2.61 [2.32, 2.94] | 1.18 | +1.43 [+1.13, +1.73] | < 1e-16 |
| Deep Sets | 1.06 | 1.11 | 0.27 | +0.84 [+0.64, +1.03] | < 1e-14 |
| Linformer | 1.91 | 1.92 | 0.72 | +1.20 [+0.94, +1.47] | < 1e-16 |
| transformer | 2.53 | 2.63 | 2.06 | +0.57 [+0.32, +0.80] | 1e-4 |
| EFN | 0.25 | 0.16 (0.00 without overflow) | 0.00 | +0.16 | 5e-4 |
| ParT_full | 2.11 | 2.14 [1.86, 2.43] | 0.77 | +1.37 [+1.07, +1.69] | < 1e-16 |
| ParT_kin | 5.02 | 4.74 | 5.00 | −0.26 [−0.74, +0.26] | 1.0 |
| PN_full | 3.12 | 3.10 | 0.74 | +2.36 [+2.03, +2.72] | < 1e-16 |
| PN_kin | 5.68 | 5.21 | 2.47 | +2.74 [+2.27, +3.19] | < 1e-16 |

The other split and multi-split numbers also reproduce:
- **z 0.5, control larger.** ParT_full 3.40 against 5.80; ParT_kin 9.06 against 12.56; PN_full 3.18 against 4.58.
- **k = 8 splits.** JEDI 10.73 (H1 11.62); ParT_full 17.45 (17.85); PN_full 21.09 (21.55).
- **Net flow at k = 8.** JEDI +741 into g; ParT_full Wqq −424, QCD +264.

## Soft emission: new controls
**pT floor of the data** (`h1_verify_ptmin*`):

| | hls4ml | JetClass |
|---|---|---|
| median jet pT | 1020 GeV | 588 GeV |
| smallest particle pT in the sample | 0.25 GeV | 0.12 GeV |
| 0.1 % quantile of particle pT | 0.26 GeV | 0.47 GeV |
| soft particle at ε = 1e-4 (median) | 0.10 GeV (below every particle in 100 % of jets) | 0.059 GeV (100 %) |
| soft particle at ε = 1e-3 | 1.0 GeV (0 %) | 0.59 GeV (0 %) |
| particles at r > 0.4 from the axis | 0.27 % | 15.4 % |
| particles at r < 0.05 from the axis | 27.2 % | 8.3 % |

At ε ≤ 1e-4 the added particle is softer than any particle in either dataset. In hls4ml the uniform disc r < 0.8 also puts it at r > 0.4 in 75 % of jets, where 0.27 % of real particles lie.

**ε sweep (uniform disc r < 0.8), % calls changed; recoil-only control in brackets:**

| model | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|
| JEDI | 2.56 (0.00) | 2.47 (0.00) | 3.02 (0.12) | 3.80 (0.76) |
| Deep Sets | 6.92 (0.00) | 6.83 (0.01) | 7.19 (0.02) | 8.85 (0.35) |
| Linformer | 1.94 (0.00) | 2.02 (0.00) | 2.03 (0.07) | 2.66 (0.42) |
| transformer | 2.10 (0.00) | 2.30 (0.01) | 2.23 (0.06) | 3.00 (0.49) |
| EFN | 0.00 (0.00) | 0.10 (0.00) | 0.62 (0.02) | 7.13 (0.58) |
| ParT_full | 0.56 (0.02) | 0.54 (0.01) | 0.80 (0.08) | 1.39 (0.48) |
| ParT_kin | 17.44 (1.22) | 12.41 (1.10) | 3.06 (1.26) | 3.19 (2.21) |
| PN_full | 1.08 (0.00) | 0.96 (0.00) | 0.56 (0.07) | 1.29 (0.53) |
| PN_kin | 4.60 (0.00) | 3.26 (0.01) | 2.07 (0.12) | 2.87 (1.26) |

**Position of the soft particle**: core (r < 0.05) against edge (0.4 < r < 0.8), ε = 1e-6, % calls changed (Holm p of core against edge):

| model | core | edge | Holm p |
|---|---|---|---|
| JEDI | 0.22 | 2.97 | < 1e-16 |
| Deep Sets | 0.50 | 9.39 | < 1e-16 |
| Linformer | 0.87 | 2.08 | < 1e-16 |
| transformer | 0.86 | 2.50 | < 1e-16 |
| EFN | 0.00 | 0.00 | 1 |
| ParT_full | 0.88 | 0.51 | 1e-4 |
| ParT_kin | 13.72 | 18.30 | < 1e-16 |
| PN_full | 1.57 | 0.91 | 1e-6 |
| PN_kin | 30.20 | 3.18 | < 1e-16 |

**Count restore.** The softest real particle is removed, then a particle with 10⁻⁶ of the jet pT is added (r < 0.8). Among the jets whose call the removal changed, the table gives the fraction whose call returns to the unedited call [95 % CI], with Fisher exact p against the EFN (Holm over 16):

| model | n | restored | Holm p vs EFN |
|---|---|---|---|
| EFN | 75 | 0.013 [0.00, 0.04] | n/a |
| JEDI | 311 | 0.38 [0.32, 0.43] | 2e-11 |
| Deep Sets | 173 | 0.54 [0.46, 0.61] | 3e-17 |
| Linformer | 325 | 0.67 [0.62, 0.72] | 2e-27 |
| transformer | 151 | 0.60 [0.52, 0.68] | 2e-19 |
| ParT_full | 118 | 0.14 [0.08, 0.19] | 0.019 |
| ParT_kin | 209 | 0.55 [0.48, 0.61] | 2e-18 |
| PN_full | 120 | 0.19 [0.13, 0.27] | 7e-4 |
| PN_kin | 254 | 0.19 [0.14, 0.24] | 5e-4 |

At ε = 1e-4 the EFN itself restores 0.55, so only the 1e-6 row has a clean IRC-safe baseline. ParT_kin's soft edit alone changes 17 % of its calls, so its restored fraction includes chance returns.

**Removing the softest particle** (an in-distribution change of N by −1), % calls changed and the flow:
- JEDI 3.11 % (g −201); Linformer 3.25 % (g −250); Deep Sets 1.73 % (g −94); EFN 0.75 %.
- ParT_full 1.18 % (Wqq +17); PN_full 1.20 % (Wqq +33).
- Each flow runs opposite to the k = 8 split.

### Hypotheses for the soft response
| | prediction | observed | excluded? |
|---|---|---|---|
| H_s: IRC-safe shape | rate → 0 as ε → 0; no position dependence at ε → 0 | EFN: 0.00 at 1e-6 | excluded for all 8 trained taggers (rates of 0.56–17 % at 1e-6, recoil ≤ 1.2 %) |
| H_c: pure count | rate flat in ε and in position; drop + add restores most drop-changed calls; flow toward high-N classes | hls4ml models: flat in ε, restore 0.38–0.67, but core and edge differ 2.4- to 19-fold; ParT_full: flat (0.56/0.54) and core ≈ edge (0.88/0.51), restore 0.14 | excluded as the only mechanism for every model; closest for ParT_full and PN_full |
| H_o: response to an input outside the training range | rate depends on ε below the data's pT floor and on position, flows not aligned with N | ParT_kin 3.1 → 17.4 % and PN_kin 2.1 → 4.6 % (core 30.2 %) as ε falls 1e-4 → 1e-6 | not excluded; fits the kin models best |

For the hls4ml models the soft response is consistent with an unweighted sum over particles of a function of angle that is largest at wide angle, which counts particles weighted by position rather than by pT. This is a reading of the position contrast, not a tested mechanism.

### ParT_kin sensitivity floor (new)
A recoil of 10⁻⁶ of the jet pT taken from the hardest particle, with nothing added, changes 1.22 % [1.02, 1.44] of ParT_kin's calls (1.10 % at 1e-5, 1.26 % at 1e-4). The rate does not scale with the size of the change, and the rerun and rebatch nulls are 0.00 %. The other models give ≤ 0.02 % at 1e-6. Part of every ParT_kin rate under an edit of the hardest particle (split, reshuffle, soft) is therefore this floor. The source was not tested; a candidate is the float32 pair mass ln m² of near-collinear pairs, which is dominated by rounding.

## Corrections to the H1 reading
1. **Collinear split.** The H1 reading on collinear splits reproduces on fresh samples: rates within the H1 intervals, the same signs against control, and the EFN at 0 on jets without overflow. The split exceeds its count-fixed control at z 0.05 for 7 of 8 taggers. ParT_kin is the exception (−0.26 [−0.74, +0.26]).
2. **"Rates that barely fall as the particle gets softer, so they count particles."**
   - At ε ≤ 1e-4 the soft particle is softer than every particle in both datasets, and in hls4ml it mostly sits at angles where 0.27 % of real particles lie.
   - The response depends on position: in hls4ml it is 2.4- to 19-fold larger at the edge than in the core.
   - For ParT_kin and PN_kin it rises as ε falls.
   - The soft-emission test therefore shows IR unsafety (a non-vanishing response as ε → 0). It does not on its own show that the response is a particle count.
   - Support for multiplicity use comes from the k-split growth, the removal of the softest particle (flows opposite to the splits), and the partial count restore (0.38–0.67 in hls4ml models, 0.14–0.19 in the full JetClass models, against 0.013 for the EFN).
3. **"ParT_full's soft response: unknown whether it levels off."** Resolved: 0.56 % at 1e-6 and 0.54 % at 1e-5, against 0.02 and 0.01 % for the recoil control.
4. **"Rate falls ≤ 2-fold for the taggers."** This holds between 1e-4 and 1e-3. Below 1e-4 the hls4ml models and ParT_full stay flat, and ParT_kin and PN_kin rise.
5. **ParT_kin.** A size-independent floor of about 1.2 % applies under any change to the hardest particle (see above).
6. **Soft-response flows.** These are not toward high-multiplicity classes in general. Deep Sets' soft response at 1e-6 moves W → Z (W −277, Z +167); JEDI's is mixed (g +74, W −55, Z −37).

## Explainer entry (addition for H1; for site/explain.py)
- **Question.** Is the taggers' response to a very soft added particle a count of particles, or a response to an input unlike anything in training?
- **Background.** IR safety requires $\lim_{\epsilon\to0}O(p_1,\dots,p_n,\epsilon q)=O(p_1,\dots,p_n)$ for every direction $q$. A pure multiplicity term $\sum_i 1$ violates it with a response independent of $\epsilon$ and of the direction of $q$. A term $\sum_i g(\Delta\eta_i,\Delta\phi_i)$ with no $p_T$ weight violates it with a response that depends on direction.
- **Method.**
  - Add one particle with $p_T=\epsilon\,p_{T,\rm jet}$ for $\epsilon\in\{10^{-6},10^{-5},10^{-4},10^{-3}\}$, either uniform in $r<0.8$, in the core $r<0.05$, or at the edge $0.4<r<0.8$. Its 3-momentum is taken from the hardest particle; the matched control is that recoil alone.
  - Count restore: remove the softest particle, add the $\epsilon=10^{-6}$ particle, and measure the fraction of removal-changed calls that return to the original call.
- **Result.**
  - The EFN's rate goes to 0 at $\epsilon=10^{-6}$. The hls4ml taggers stay at 2–7 %, mostly from edge placements. ParT_full levels off at 0.55 %. ParT_kin rises to 17 %.
  - Removal followed by the $10^{-6}$ addition returns 38–67 % of changed calls in the hls4ml taggers, against 1 % for the EFN.
- **Checks.**
  - Re-evaluation, re-sorting and rebatching change 0 calls.
  - Slot 0 holds the hardest particle in all jets.
  - The added particle is below the smallest particle pT in the data (0.25 GeV hls4ml, 0.12 GeV JetClass).
- **Physical significance.** All trained taggers are IR unsafe. The form of the unsafety differs by model:
  - hls4ml taggers: position-weighted counting;
  - ParT_full: a small constant rate;
  - kinematic-only JetClass models: a growing response to sub-threshold inputs.

  A real detector never delivers particles below its threshold, so the physically relevant multiplicity response is the one measured with collinear splits and the removal of real particles.
