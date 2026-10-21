# H76 verification (skeptic, job d-h76-verify, seed 7621): per-sign responses, a size-matched random control, and extra angularity references

## Conclusion

On a fresh sample the H76 numbers reproduce: every trained quark/gluon network responds mainly to the LHA and width edits, and its five-number response pattern matches the IRC-safe energy-flow network's (cosine 0.98–1.00). The response is lopsided, though. Raising LHA or lowering width at fixed other angularities moves ParT_kin toward "quark" about four to six times more than the opposite edits move it toward "gluon". Both of those edits push the jet's energy toward a single common distance from the axis. A random rearrangement of the same size that keeps all five observables fixed barely moves the networks, so the response is specific to the change in radial shape. The pattern is not specific to the EFN: a boosted tree trained only on seven angularities (β = 0.1 to 3) plus pT^D gives the same pattern (cosine 0.95–0.98). Single-variable classifiers, for example on the leading particle's distance from the axis, lie on the same line (|cosine| 0.94–0.99). So the five-number vector shows that the networks read the radial energy profile and not the particle count. It cannot show which radial variable they read, and the claim that they read "a combination no single angularity measures" is not supported.

Terms used below:
- λ^κ_β = Σ_i z_i^κ (ΔR_i/R)^β is a generalized angularity. z_i = pT_i/ΣpT, ΔR_i is the distance of particle i from the E-scheme jet axis, and R = 0.4.
- LHA = λ(1, ½) and width = λ(1, 1).
- D = λ(1,1)/λ(1,½)² is the radial dispersion. By the Cauchy–Schwarz inequality D ≥ 1, with equality when all the energy sits at one common radius.
- r_1 is the radius of the leading particle, in units of R.
- s = log p(gluon) − log p(quark) is the network's gluon log-odds.
- crand is H76's random edit. It is drawn in the null space of the constraints, so all five observables, ΣpT and the axis stay fixed. Its size is the mean of the + and − edit sizes.
- crand_big is new here. It is like crand, but has the full size and per-particle profile of the larger edit (LHA+ or width−).
- C2ST is a classifier two-sample test: the AUC of a classifier that tries to separate edited from real jets.
- McNemar is the paired test on how many network calls (s > 0) change.
- Holm is the step-down correction for multiple tests.

## Code review of interp/h76_angularity_qg.py

I found no bug that changes the numbers:
- Rows are random (`rng.choice` per class), not the first k.
- Padded slots are masked in the solver (pt = exp(a)·mask, angularity sums × mask).
- The tolerance check recomputes every kept quantity in numpy on the edited list.
- The references are trained on the train split.
- Re-padding the unedited jets to 25 extra slots changes the ParT_kin logits by at most 1e-6.

The edit axis is the jet rapidity y_J, while the networks use the jet pseudorapidity η_J. Both are held within 0.5 % of R, so this does not matter.

Three design points matter for the reading:
1. **The slope is an odd difference.** The reported slope is (s(t+) − s(t−))/2 scaled ×10/3. It hides a large even part (s(t+) + s(t−))/2, and the + and − edits differ in size by 2.5–3.4× (median ‖δ‖ LHA+ 0.355 vs LHA− 0.104; width− 0.255 vs width+ 0.111). crand is matched only to the mean size.
2. **The edits move angularities outside the five.** On accepted jets (median over jets):
   - LHA+3 % raises λ(1,3) by 12.8 % and λ(1,¼) by 2.5 %, and lowers D by 5.7 %.
   - width−3 % raises λ(1,3) by 11.1 % and lowers D by 3.0 %.

   So "partial derivative at fixed others" holds only for the five chosen observables. The edit direction is the minimum-norm one, and a non-angularity model can respond to anything else that edit changes.
3. **The five-number vector has little resolving power.** Lowering LHA and raising width at fixed λ(1,2) and pT^D are both moves away from the bound D = 1, so any model of the radial profile gets a vector close to (0, −a, +b, ~0, ~0) (see the references below).

## Procedure

Job d-h76-verify (GPU) runs `python -m interp.h76_verify --gens py,hw --per_class 2000 --seed 7621`. The script is interp/h76_verify.py, which imports H76's sampler, solver, edits, tolerance and references unchanged.

**Sample.** For each generator (KMT Pythia test split; Herwig 7.1 files 18–19): 2,000 quark + 2,000 gluon jets, random rows, seed 7621, N ≥ 8. Jets accepted in every condition: 83.8 % (Pythia) and 80.1 % (Herwig); all statistics use these jets.

**Edits.** These are H76's own edits:
- LHA, width, pT^D and λ(1,2) each moved ±3 % at fixed other three, ΣpT and axis;
- N± (exact splits / merges of the softest 10 %, then restore);
- crand±.

**New control: crand_big.** For LHA+ and width−, it is a null-space random edit with the same per-particle size profile and the full norm of that edit. It is applied with both signs and then restored to all five observables, ΣpT and the axis.

**New references.** Gradient-boosted trees (GBDT), trained on 20k + 20k Pythia train jets:
- single-variable GBDTs on λ(1,0.1), λ(1,0.25), D and r_1;
- "g_ext": N, pT^D and λ(1,β) for β ∈ {0.1, 0.25, 0.5, 1, 1.5, 2, 3};
- "g_ext_irc": the same without N;
- H76's GBDT on the five ("g5").

**Metrics** (per jet; bootstrap over jets, 1,000 resamples):
- per-sign excess e(t±) = Δs(t±) − ½[Δs(crand+) + Δs(crand−)];
- excess over crand_big;
- the even part ½[Δs(t+) + Δs(t−)] − crand;
- call flips at s = 0, with a McNemar test against crand_big;
- H76's odd 5-vector and its cosine with each reference.

**Tests.** H0: excess = 0. H1: excess ≠ 0. The p-value is a sign-flip test (10,000 sign vectors), Holm-corrected over 70 tests per generator.

## Unedited models on this sample (AUC / accuracy at s = 0)

| model | Pythia jets | Herwig jets |
|---|---|---|
| ParT_kin (Py-trained) | 0.905 / 0.829 | 0.806 / 0.714 |
| PN_kin | 0.905 / 0.826 | 0.806 / 0.713 |
| LorentzNet | 0.925 / 0.856 | 0.798 / 0.697 |
| PFN | 0.898 / 0.824 | 0.807 / 0.722 |
| EFN | 0.889 / 0.813 | 0.803 / 0.716 |
| ParT_kin (Hw-trained) | 0.888 / 0.800 | 0.834 / 0.760 |
| ParT_kin weight-permuted | 0.554 / 0.498 | 0.527 / 0.500 |
| GBDT g5 / g_ext / g_ext_irc | 0.866 / 0.866 / 0.846 | 0.797 / 0.798 / 0.782 |
| GBDT on D / r_1 / λ(1,0.1) | 0.638 / 0.666 / 0.715 | 0.651 / 0.681 / 0.711 |

## Results

### 1. Reproduction

The ParT_kin 5-vector on Pythia jets (N, LHA, width, pT^D, λ(1,2)) is (−0.010, −0.935, +0.703, −0.063, −0.223) per +10 %. H76 had (−0.013, −0.93, +0.71, −0.060, −0.22). The cosine with the EFN is:
- Pythia jets: 0.996 (ParT), 0.997 (PN), 0.984 (LorentzNet), 0.983 (PFN), 0.985 (Hw-trained ParT);
- Herwig jets: 0.95–0.99.

### 2. Per-sign responses (Pythia jets, gluon log-odds at ±3 %, excess over crand [95 % CI])

| model | LHA+ | LHA− | width+ | width− | even part LHA / width |
|---|---|---|---|---|---|
| ParT_kin | −0.464 [−0.494, −0.435] | +0.106 [0.086, 0.125] | +0.056 [0.036, 0.074] | −0.360 [−0.383, −0.336] | −0.179 / −0.152 |
| PN_kin | −0.476 | +0.108 | +0.050 | −0.390 | −0.184 / −0.170 |
| LorentzNet | −0.235 | +0.056 | +0.009 [−0.011, 0.030] | −0.158 | −0.089 / −0.075 |
| EFN | −0.672 | +0.284 | +0.210 | −0.579 | −0.194 / −0.184 |
| ParT_kin (Hw-trained) | −0.308 | +0.108 | +0.101 | −0.296 | −0.100 / −0.097 |
| GBDT g_ext_irc | −0.164 | +0.233 | +0.152 | −0.121 | +0.035 / +0.015 |

On Herwig jets ParT_kin gives −0.314, +0.132, +0.064 and −0.257.

The sign pattern "LHA down / width up → gluon" holds on both sides; every excess is non-zero after Holm except LorentzNet width+. For the trained networks, however, the toward-quark side (LHA+, width−) is 4–8× larger than the toward-gluon side for ParT/PN (2.4–2.9× for the EFN and the Herwig-trained ParT), and the even part is −0.15 to −0.19 for ParT/PN. The angularity-only GBDT has no such asymmetry: its even part is +0.02 to +0.04.

### 3. Size-matched random control (crand_big)

Shift Δs under crand_big (Pythia):

| model | crand_big for LHA+ | crand_big for width− |
|---|---|---|
| ParT_kin | +0.072 [0.057, 0.087] | +0.065 |
| PN_kin | +0.046 | +0.034 |
| EFN | +0.136 | +0.096 |

C2ST of crand_big against real jets: 0.55 (Pythia) and 0.59–0.61 (Herwig). For the edits themselves: 0.87 (LHA+) and 0.86 (width−).

Excess of the edit over crand_big, ParT_kin: LHA+ −0.479 [−0.509, −0.452], width− −0.345 [−0.368, −0.324] (p < 1e-4, Holm < 0.05).

Calls changed at s = 0, ParT_kin, Pythia:

| edit | edit flips | crand_big flips | McNemar b / c | p |
|---|---|---|---|---|
| LHA+ | 13.9 % | 5.8 % | 383 / 111 | 5e-36 |
| width− | 10.3 % | 4.5 % | 275 / 81 | 8e-26 |

**Competing hypotheses**
- "Generic response to any edit of this size": excluded, because a size-matched null-space edit moves ParT by +0.07, in the opposite direction.
- "Response to off-manifold structure", as opposed to the radial shape: not separated. crand_big is closer to the data than the edit (C2ST 0.55 vs 0.87), so this control does not match the edit's distance from real jets.

### 4. Count and pT^D edits (Pythia, Δs per edit; the even part is ½[Δs(N+) + Δs(N−)])

| model | N+ (split) | N− (merge) | even part | odd part per +10 % |
|---|---|---|---|---|
| ParT_kin | −0.061 [−0.065, −0.058] | −0.042 [−0.048, −0.036] | −0.052 [−0.055, −0.049] | −0.010 |
| PN_kin | −0.100 | −0.022 | −0.061 | −0.039 |
| LorentzNet | −0.058 | −0.073 | −0.065 | +0.007 |
| ParT_kin weight-permuted | −0.052 | +0.050 | −0.001 | −0.051 |
| PFN | +0.092 | −0.106 | −0.007 | +0.099 |
| ParT_kin (Hw-trained) | −0.143 | +0.044 | −0.049 | −0.094 |

Adding particles (exact collinear halves; C2ST 1.00) and removing particles (merges; C2ST 0.95) both move the trained ParT, PN and LorentzNet toward quark, by about 0.05 log-odds per 10 % of N. The odd ("counting") part is small, as H76 reports. The trained networks do respond to these edits, though, in a way that does not depend on the sign of the count change. That points to a response to the edit itself (coincident particle pairs, merged particles), not to the count. "Smaller than the permuted network" holds only for the odd part.

pT^D: ParT_kin moves −0.084 (+3 %) and −0.049 (−3 %) relative to crand, an even response of −0.067. The odd part, −0.063 per +10 %, is small, as H76 reports.

### 5. Is the 5-vector specific to the EFN? (cosine with ParT_kin's 5-vector, Pythia; Herwig in brackets)

| reference | cosine |
|---|---|
| EFN | 0.996 (0.990) |
| GBDT on 7 λ(1,β) + pT^D, no N (g_ext_irc) | 0.973 (0.961) |
| GBDT g_ext (same + N) | 0.722 (0.699) |
| GBDT g5 (H76) | 0.498 (0.470); 4 IRC-safe components only: 0.853 |
| single GBDT on r_1 | −0.990 (−0.982) |
| single GBDT on D | −0.968 |
| single GBDT on λ(1,0.1) | −0.967 |
| single GBDT on λ(1,0.25) | −0.949 |

The other networks show the same pattern (g_ext_irc 0.93–0.98; r_1 −0.95 to −0.99).

**Competing hypotheses**
- "The networks follow an EFN-specific function rather than an angularity classifier": not supported. A GBDT on angularities alone matches to 0.95–0.98, and the low cosine with g5 comes from g5's N component.
- "A combination no single angularity measures": not supported. Single radial variables lie on the same line (|cos| 0.94–0.99).

The negative signs mean the networks move opposite to what a data-trained classifier on r_1, D or λ(1,0.1) does under these edits. This rules out "the networks read r_1 (or D) the way the data-trained classifier does". The 5-vector cannot distinguish among radial variables.

### Statistics

Holm over 70 per-sign and crand_big tests per generator: 66 / 70 have p_Holm < 0.05 (Pythia), and 66 / 70 (Herwig). The ones above 0.05 are:
- LorentzNet width+;
- PFN pT^D−;
- the weight-permuted ParT on LHA+ and on LHA+ vs crand_big.

Results file: /Users/anrunw/Documents/aaron-interp/results/physics/h76_verify.json; log: logs/d-h76-verify.log.

## Figure

results/physics/h76_angularity.png shows a Pythia gluon jet under width +10 % and pT^D +10 %, with the unit-vector heat map. The values match the stats file. The example uses the ±10 % side arm, which width− passes for only 53 % of jets; the main ±3 % edit is not shown. The heat map gives only the odd part, so the per-sign asymmetry in section 2 cannot be seen in it.

## Corrected reading

On jets accepted by the solver, the IRC-safe edits moved one of LHA, width, pT^D and λ(1,2) by 3 % at fixed other three, ΣpT and axis. They also moved unconstrained angularities: λ(1,3) by up to +13 % and the dispersion D by up to −6 %.

Trained QG networks respond mostly to the LHA/width edits, in the direction "LHA down / width up → gluon". Most of the response comes from the two edits that push energy toward a single radius (LHA+ and width−; quark-ward by 0.3–0.5 log-odds at ±3 % for ParT/PN). The other two edits (LHA−, width+) give +0.05 to +0.11, and the even part is −0.15 to −0.19. A size-matched random edit at fixed observables moves ParT by +0.07, so size alone does not explain the response.

The five-component response vector matches the EFN's (0.98–1.00). It also matches a GBDT on angularities alone (0.95–0.98), and single radial variables lie on the same line (|cos| 0.94–0.99). The vector therefore separates "radial energy profile" from "particle count" but does not identify the variable.

The count edits move ParT/PN/LorentzNet toward quark whether particles are added or removed (≈ −0.05 log-odds per 10 % of N). The odd (counting) part is ≤ 0.04, and PFN and the Herwig-trained ParT/PN keep a non-zero odd part.

## Issues

1. The odd-slope metric with a mean-size crand hides a one-sided response. The ±3 % → ±10 % linear scaling is not supported (H76's own ±10 % arm also shows sub-linearity).
2. The LHA/width edits are off the jet distribution (C2ST 0.85–0.87). No control matches that distance while leaving the radial shape alone, so "shape" and "off-manifold" responses are not separated.
3. The minimum-norm edit direction moves λ(1,3), λ(1,¼), D and r_1. "At fixed others" holds only for the five chosen observables.
4. The count edits produce exactly coincident or merged particles (C2ST 0.95–1.00). The trained networks' even response to them is not explained.
5. One training seed per network. Jets near the moment bound (16–20 %) are excluded from every statistic.

---

# Appendix: worker self-rerun (seed 7611), as submitted with H76


## Conclusion

The rerun used a new random sample (seed 7611, 2,000 quark + 2,000 gluon jets per generator) and the same code. It reproduces H76. Every trained network's response to one-at-a-time angularity edits points along "width up / LHA down", like the IRC-safe EFN (cosine 0.975–0.996). The count and pT^D components are small, and the vectors differ from H76 by at most 0.04 per component. The edits hold the other angularities, jet pT and axis fixed to a median 10⁻⁶–10⁻⁴ relative. The solver rejects 3–21 % of jets for the LHA and width edits. This rerun was done by the same worker with the same code, so it checks sampling noise and tolerances, not the edit design: the LHA/width edits remain off-manifold.

## Procedure

Job d-h76-vrun (GPU) runs `python -m interp.h76_angularity_qg --step run --seed 7611 --per_class 2000 --tag _v7611 --c2st_n 4000`. Job d-h76-vstats (CPU) runs the stats and fig steps on the same tag. The edits, controls, models, references and metrics are as in H76_report.md. The jet set is new: a different random draw from the same test files, overlapping H76's by chance only.

## Unedited models (4,000 jets per generator; AUC / accuracy, Pythia ; Herwig)

| model | Pythia jets | Herwig jets |
|---|---|---|
| ParT_kin | 0.907 / 0.832 | 0.808 / 0.720 |
| PN_kin | 0.907 / 0.832 | 0.809 / 0.719 |
| LorentzNet | 0.919 / 0.848 | 0.798 / 0.697 |
| PFN | 0.895 / 0.818 | 0.810 / 0.728 |
| EFN | 0.889 / 0.815 | 0.800 / 0.712 |
| ParT_kin (Hw) | 0.888 / 0.800 | 0.831 / 0.760 |
| PN_kin (Hw) | 0.885 / 0.791 | 0.831 / 0.762 |
| permuted ParT_kin | 0.562 | 0.520 |

All are within 0.005 AUC of H76 and of H24.

## Constraint tolerance and solver failures

Kept quantities are checked by an independent numpy recomputation: the other three IRC-safe angularities, Σ pT, the vector jet pT and N within 0.5 %, and the jet η and φ within 0.5 % of R. In accepted jets they move by a median of 10⁻⁶ (pT^D edits) to 10⁻⁴ (λ(1,2) edits) relative; the 99th percentile is at most 0.11 % (λ(1,2) edits and the crand controls).

Acceptance by edit (Pythia ; Herwig):

| edit | acceptance |
|---|---|
| pT^D ±3 % | 100 % |
| λ(1,2) ±3 % | ≥ 99.7 % |
| LHA+ / LHA− | 84.6 / 97.2 % ; 78.8 / 95.8 % |
| width+ / width− | 95.8 / 93.2 % ; 94.2 / 90.3 % |
| N± | 100 % |

Jets accepted in every main condition: 84.4 % (Pythia) and 78.8 % (Herwig), against 84.6 % and 79.8 % in H76.

## Reproduction: unit response vectors (N, LHA, width, (pT^D)², λ(1,2)), Pythia jets

| model | H76 (seed 7601, 8,000 jets) | H76v (seed 7611, 4,000 jets) |
|---|---|---|
| ParT_kin | (−0.01, −0.78, 0.60, −0.05, −0.18), \|g\| 1.19 | (−0.01, −0.80, 0.57, −0.05, −0.20), \|g\| 1.16 |
| PN_kin | (−0.03, −0.77, 0.61, −0.03, −0.19) | (−0.04, −0.78, 0.60, −0.04, −0.20) |
| LorentzNet | (0.02, −0.81, 0.55, −0.05, −0.20) | (0.01, −0.83, 0.51, −0.07, −0.21) |
| PFN | (0.11, −0.82, 0.55, −0.05, −0.07) | (0.12, −0.84, 0.53, −0.05, −0.08) |
| EFN | (0.00, −0.76, 0.64, −0.02, −0.12) | (0.00, −0.76, 0.64, −0.01, −0.13) |
| ParT_kin (Hw) | (−0.10, −0.69, 0.67, −0.04, −0.23) | (−0.10, −0.69, 0.67, −0.04, −0.24) |
| GBDT all five | (0.82, −0.52, 0.16, −0.19, 0.05) | (0.84, −0.48, 0.16, −0.20, 0.05) |

ParT_kin N effect per +10 %:

| | Pythia jets | Herwig jets |
|---|---|---|
| H76v | −0.014 [−0.018, −0.010] | −0.012 |
| H76 | −0.013 | −0.010 |

Permuted ParT gives −0.052 and LR(N) +0.402.

- **Cosine with the EFN:** 0.984–0.996 for ParT, PN, LorentzNet and PFN on Pythia jets, 0.975–0.988 on Herwig jets.
- **Cosine with the all-five GBDT:** ParT 0.47 (Pythia jets).
- **H0** (vector = GBDT vector): rejected for every network (p 5 × 10⁻⁵).
- **Holm:** 135 / 152 tests significant at 0.05. The added non-significant ones are EFN pT^D, LorentzNet N and EFN N, as expected for an IRC-safe network.
- **Generator dependence:** the Herwig-minus-Pythia-trained N slope reproduces (ParT −0.085, PN −0.13).

## C2ST (Herwig and Pythia, 4,000 jets)

The pattern is the same as in H76:
- unedited-vs-unedited: 0.527 / 0.507;
- pT^D ±3 %: 0.51–0.54;
- crand and crot: 0.50–0.57;
- λ(1,2) ±3 %: 0.56–0.67;
- LHA and width ±3 %: 0.84–0.90;
- ±10 % LHA / width: 0.95–0.98;
- count edits: 0.95–1.0.

## Issues

- This is a self-rerun (same worker, same code), not an independent verifier pass.
- No second training seed of the networks was used.
- The ±10 % linearity arm and the off-manifold LHA/width edits carry over unchanged from H76.
