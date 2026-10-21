# H61 verify: fresh-event rerun of the Delphes, MPI-off top hadronisation test, plus three missing controls

Verifier for node H61 (claim D24). Script `interp/h61_verify.py`; jobs `d-h61-verify-gv0`, `d-h61-verify-ghzp` (generation, 1 CPU / 4 GiB, 19 and 11 min) and `d-h61-verify` (GPU scoring and statistics, 6 min). Results: `$AI_RESULTS/physics/h61_verify.json` (local copy `results/physics/h61_verify.json`), scores `$AI_RESULTS/physics/h61_verify_scores.npz`, data `/j-jepa-vol/aaron-interp/data/top/gen_h61v/`.

## Conclusion

On a fresh set of simulated events the result reproduces. Changing only Pythia's hadronisation settings adds about 8 % more reconstructed particles to top jets at the same jet mass and three-prong shape (τ32). This moves the Top Landscape ParT top score by −0.29 log-odds toward QCD; H61 measured −0.31. The IRC-safe EFN moves by +0.03. Two new controls show that the networks respond to the number of objects and not to where the soft objects sit:
- moving the same soft objects by the same small angle without adding any leaves every network unchanged;
- splitting randomly chosen constituents, hard ones included, gives a shift as large as splitting the softest ones, or larger.

Three points remain open:
- Our generated jets still differ from the Top Landscape training jets.
- The split does not change only the count, because it also halves the momenta of the split objects.
- The merge test overshoots, because the merged jets end below the original count.

## What was checked

### 1. Code review of `interp/h61_top_delphes.py` and `interp/h61_install_delphes.sh`

| item | finding |
|---|---|
| axis / coordinates | Model inputs use the constituent-sum axis with Δη·sign(η_jet) (`h4_top_wmass.jet_inputs`, the Top Landscape conversion), the same path as H60 and the Top Landscape test rows. Correct. |
| padded particles | `massless_P4` zeroes rows with pT = 0; `pt_feats`, `_obs` and the scorers use `n`. Padding is not counted. Correct. |
| pairing | Every event is reseeded with seed(w, cls, i) before `p.next()`, so the hard process and shower are the same in V0 and Hz+. Pairs are matched on the key (worker, class, event, slot). The Delphes seed is per chunk, so smearing is not paired. V0r measures this at ≤ 0.024. Correct. |
| `det_jets` index offset | `Event.reset()` re-adds the system entry. `off = ev.size()` = 1 is used when mapping constituents. Correct (this is the fixed bug, issue 2). |
| split | Each of the k softest objects becomes two halves of pT/2 at ±0.005 in (η, φ). The jet four-momentum is kept to O(δ²): measured Δm = 0.000 and EFN Δs = −0.000. As described. The edit also halves the pT of the k split objects. On the softest objects the change in pT^D is below 0.1 %. |
| merge | Three-momenta added, the result made massless, so the pair's energy is not conserved. The report does not claim it is. Merged jets end 2.3 objects below V0, because k = max(ΔN, 0). This is stated. |
| observables vs model inputs | Computed from the same (P, n) arrays that are scored. Correct. |
| working point | Cut at the median of the paired V0 top scores, so ε_S(V0) = 0.50 by construction. McNemar on paired calls. Correct. |
| row selection | Top Landscape rows are drawn with `rng.choice` per class, not first-k. Correct. |
| truth matching | b, q and q′ are taken from `p.process`, the hard-process record before the transverse recoil from ISR. In `h61_jet.png` all three markers sit about 0.15 away from their particle clusters, in the same direction. This changes which jets pass the 0.8 truth match (the same for every setting), not the pairing. Minor. |
| memory | `d-h61-g-hzp2` peaked at 3,071 MiB on a 3 GiB limit, and the verify jobs at 4,095 MiB on 4 GiB. Both are at the limit, probably page cache. Operational note only. |

No bug was found that changes a reported number.

### 2. Figures

- **`h61_shifts.png`** matches the JSON.
  - Top jets: ParT, PN and PFN sit at about −0.3 under Hz+ and the split. EFN and GBDT(EFP) sit at 0.
  - QCD jets: the signs differ between architectures.
  - LorentzNet batch 1 is off-scale on QCD jets. The legend's N and m entries are cut off at the colon, but the values are in the report.
- **`h61_jet.png`** shows one paired top jet: 41 → 60 objects, m 179 → 176 GeV. The b, q and q′ markers are offset, as noted above.

### 3. Fresh-event rerun (job d-h61-verify)

**Procedure.** Unchanged H61 chain (`H61.gen_worker`: Pythia 8.317, 14 TeV, Monash, MPI off, Delphes 3.5.1 ATLAS card, anti-kT 0.8 on EFlow objects, 550–650 GeV, |η| < 2, truth-matched tops).
- Fresh worker seeds 6201–6210; H61 used 6101–6120, so the event seeds do not overlap.
- 5,000 top + 5,000 QCD jets each for V0 (Monash) and Hz+ (aLund 0.79, bLund 0.88, σ 0.247 GeV).
- 7,456 V0/Hz+ pairs: 3,461 top, 3,995 QCD.
- Split and merge rebuilt with `H61.build_edits` (k = max(N_Hz+ − N_V0, 0); mean k = 7.0 for top jets).

**Symbols.**
- **s:** top log-odds.
- **Δs:** paired mean change of s, variant minus V0, with a 95 % bootstrap CI over jet pairs (B = 2,000).
- **net − EFN:** paired difference between the network's shift and the EFN's shift. Its p-value is bootstrap-z, Holm-corrected over 72 tests: 6 networks × 6 edits × 2 classes. Holm is a step-down correction that controls the chance of any false positive in the family.
- **share:** Δs(edit) / Δs(Hz+) on the same V0 keys.
- **per object:** share ÷ (ΔN_edit / ΔN_Hz+), i.e. the shift per added object relative to Hz+.
- **ε_S:** top efficiency at the V0 50 % cut.

**Unedited models on fresh V0** (AUC / accuracy at s > 0, 10,000 jets):

| model | AUC / accuracy |
|---|---|
| ParT | 0.9882 / 94.5 % |
| PN | 0.9876 / 94.8 % |
| LorentzNet | 0.9886 / 94.8 % |
| PFN | 0.9823 / 92.2 % |
| EFN | 0.9809 / 93.3 % |
| ParT_kin | 0.9873 / 93.4 % |
| PN_kin | 0.9887 / 94.9 % |
| permuted ParT | 0.773 / 71.5 % |

H61's V0 values are within 0.001–0.007 of these.

**Hz+ on top jets.** ΔN = +4.76 [4.41, 5.11] objects (+7.7 %), Δm = −0.01 [−0.40, 0.37] GeV, Δτ32 = −0.0001, ΔmW = +0.32 [−0.06, 0.70] GeV, ΔpT^D −2.8 %.

| model | H61 Δs | fresh Δs [95 % CI] | fresh net − EFN [CI], Holm p | ε_S 0.50 → (McNemar p) |
|---|---|---|---|---|
| ParT | −0.313 | −0.288 [−0.348, −0.229] | −0.319 [−0.377, −0.262], 5e−25 | 0.454 (5e−10) |
| PN | −0.308 | −0.222 [−0.324, −0.084] | −0.253 [−0.353, −0.113], 8e−4 | 0.473 (3e−4) |
| LorentzNet b32 | −0.415 | −0.449 [−0.528, −0.371] | | 0.452 (1e−9) |
| PFN | −0.334 | −0.383 [−0.423, −0.341] | −0.414 [−0.455, −0.372], 4e−81 | 0.424 (5e−27) |
| ParT_kin | −0.073 | −0.075 [−0.123, −0.029] | −0.106 [−0.153, −0.058], 3e−4 | 0.483 (0.02) |
| PN_kin | −0.201 | −0.205 [−0.260, −0.147] | | 0.468 (2e−5) |
| EFN | +0.014 | +0.031 [+0.001, +0.059] | – | 0.510 (0.10) |
| permuted ParT | +0.030 | +0.037 [+0.030, +0.044] | | |

Every H61 value lies inside the fresh 95 % CI except two:
- PFN: −0.334 lies just above the fresh upper limit of −0.341.
- Permuted ParT: +0.030 is at the fresh lower limit.

QCD jets reproduce the sign pattern:
- ParT −0.125 (H61 −0.160), PFN −0.204 (−0.210).
- PN +0.203 (+0.170), LorentzNet +0.379 (+0.257).
- EFN −0.017.

**Split and merge, top jets** (fresh; H61 in brackets):

| model | split Δs | split / Hz+ [CI] | per object | merge: share removed [CI] |
|---|---|---|---|---|
| ParT | −0.302 (−0.278) | 1.05 [0.88, 1.30] (0.89) | 0.71 (0.57) | 1.60 [1.34, 1.98] (1.39) |
| PN | −0.371 (−0.342) | 1.67 [1.15, 3.99] (1.11) | 1.14 (0.71) | 2.16 [1.48, 5.22] (1.46) |
| PFN | −0.373 (−0.350) | 0.97 [0.90, 1.06] (1.05) | 0.66 (0.67) | 1.18 [1.09, 1.29] (1.26) |
| ParT_kin | −0.318 (−0.296) | 4.3 [2.7, 11.0] (4.1) | 2.9 | 2.0 [1.3, 5.2] |

Merged Hz+ jets keep the Hz+ pT^D change (−2.6 %) and end 2.24 objects below V0. ParT still moves +0.171 above V0. Read together, the softer pT spectrum carried over from Hz+ is not what lowers ParT's top score here. The count deficit is a confound for the size of the overshoot.

### 4. Missing controls (run in d-h61-verify)

For every control: the same 3,461 / 3,995 pairs and the same k as the split.

| control | what is changed | ΔN | ParT | PN | PFN | ParT_kin | EFN |
|---|---|---|---|---|---|---|---|
| jitter | the k softest objects each moved 0.005 in (η, φ) in a random direction; pT and N unchanged (the same particles and displacement as one split half) | 0 | +0.000 | −0.001 | +0.000 | +0.001 | −0.000 |
| split_rand | k constituents drawn at random from the whole jet (any pT) split into halves 0.01 apart | +7.00 | −0.387 [−0.408, −0.366] | −0.327 | −0.553 | −0.184 | −0.001 |
| split_d02 | the softest-k split with the halves 0.02 apart instead of 0.01 | +7.00 | −0.303 | −0.375 | −0.373 | −0.318 | −0.000 |

These are top-jet Δs values.
- Under jitter, every top-jet network − EFN test has Holm p = 1.0. On QCD jets LorentzNet moves +0.041 (Holm 0.17).
- split_rand per added object relative to Hz+: ParT 0.91, PN 1.00, PFN 0.98, ParT_kin 1.68.

**Hypotheses and which are excluded.**
- **H0 (no count response; net − EFN = 0 under Hz+).** Excluded for ParT, PN, PFN and ParT_kin on top jets (Holm ≤ 8e−4).
- **A1 (the split response comes from moving soft objects by 0.005, which is off-distribution geometry, not the count).** It predicts jitter ≈ split. Measured: jitter ≈ 0, split ≈ −0.30. Excluded for all networks on top jets.
- **A2 (the response depends on the arbitrary 0.01 separation).** It predicts split_d02 ≠ split. Measured: equal to within 0.005. Excluded.
- **A3 (the networks respond specifically to extra *soft* objects).** It predicts split_rand < split. Measured: split_rand ≥ split for ParT and PFN, about equal for PN, and smaller only for ParT_kin and PN_kin. Not supported for the Top Landscape networks: they respond to the object count at fixed IRC-safe content, whichever objects are split.
- **A4 (particle-level / MPI artefact).** It predicts a detector-level shift within the EFN band. Excluded on the fresh sample, as in H61.

## Corrections to the H61 reading

1. **"0.55–0.72× the particle-level shift for the Top Landscape networks."** From the report's own table the ratios are ParT 0.72, PN 0.74, LorentzNet 0.51 and PFN 0.58. The range is 0.51–0.74.
2. **"The exact collinear split ..., which changes only the count."** The split also halves the pT of the k split objects. It keeps the IRC-safe content (EFPs, m, τ32) fixed, not only "the count". The jitter control (new) shows that moving those objects alone does nothing. That supports reading the split as a count response.
3. **"Treat extra soft particles ... as a sign of QCD."** The split_rand control shows that the Top Landscape ParT, PN and PFN respond as much or more when random, including hard, constituents are split. The supported statement is "more objects at fixed IRC-safe substructure". The softness of the objects added by hadronisation is not what drives it.
4. **Merge.** On the fresh sample, merge removes more than all of the shift: ParT 1.60, PN 2.16, PFN 1.18. The merged jets end 2.2 objects below V0, so these shares are upper bounds.
5. **Sizes.** On fresh events PN's Hz+ top shift is −0.22 [−0.32, −0.08], PFN's −0.38, ParT's −0.29. The reading's range "0.20–0.42" becomes about 0.20–0.45 when both samples are included.

Not rechecked: the C2ST of 0.74 between our Delphes jets and Top Landscape. That caveat stands, and the absolute sizes on true Top Landscape conditions remain uncertain.

## Corrected reading

On Delphes ATLAS-card jets generated with MPI off, the hadronisation-only Hz+ change raises the top-jet object count by 6.7–7.7 % at fixed m, τ32 and mW, in two independent event samples.
- Under it, the Top Landscape ParT, PN, LorentzNet and PFN and the JetClass PN_kin lower the top log-odds by 0.20–0.45, while the EFN moves +0.01 to +0.03.
- For the Top Landscape networks that is 0.51–0.74 of the particle-level shift.

A collinear split that adds objects at fixed IRC-safe content reproduces 0.9–1.7 of the shift for ParT, PN and PFN, or 0.57–1.14 per added object.
- Moving the same soft objects by the same angle without adding any changes no network score (|Δs| ≤ 0.009 on tops).
- Splitting random constituents instead of the softest gives an equal or larger shift.
- So the response is to the object count at fixed IRC-safe content, not to the softness or the position of the added objects.

Merging the extra objects back removes 1.2–2.2 of the shift. These shares are upper bounds, because merged jets end about 2 objects below V0.

JetClass ParT_kin moves only −0.07 under Hz+ but −0.30 to −0.32 under the split.

On QCD jets the sign depends on architecture: ParT and PFN shift QCD-ward, PN and LorentzNet top-ward.

The generated jets remain separable from Top Landscape jets (C2ST 0.74).

## Corrected conclusion (for the H61 entry)

At detector level with MPI off, as in the Top Landscape training sample, Pythia's hadronisation can add about 7 % more reconstructed particles to top jets at the same mass and three-prong shape. This lowers the Top Landscape ParT top score by about 0.3 log-odds in two independent event samples, while an IRC-safe network does not move. Adding objects by splitting particles reproduces this, while moving the same particles without adding any does nothing. The taggers therefore read the number of objects at fixed IRC-safe substructure as a sign of QCD, and this holds on their own type of input, not only at particle level. Two points remain uncertain. Our Delphes jets are still separable from Top Landscape jets (classifier AUC 0.74). JetClass ParT_kin shows only a small shift under the hadronisation change, although it responds to the count edit.
