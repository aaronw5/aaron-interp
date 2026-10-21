# H48 verification — lepton-decay endpoints (t→bℓν m_bl endpoint, H→ℓνqq′ visible-mass edge)

Verifier node for H48. Script `interp/h48_verify.py` (imports the edit, rebuild, observable and model code of `interp/h48_lepton_endpoint.py` and adds a second edit). Jobs: `d-h48v-smoke2` (300 jets per class, 172 s) and `d-h48-verify` (CPU 6 / 12 GiB, 969 s, peak 5.1 GiB, 5.6 cores on average). Results: `results/physics/h48_verify.json`, `h48_verify_perjet.npz`, figure `results/physics/h48_verify_scan.png`, log `logs/d-h48-verify.log`. Nothing committed.

## Conclusion

On a new sample, the H48 effect comes back at the same size. When the lepton in an H→ℓνqq′ jet is rotated away from the other particles so that the jet mass rises 20 GeV across 125 GeV, all four networks lower their Higgs score about twice as much as a mass-only classifier predicts (ParT_full −3.98 vs −2.08 log-odds for the across-minus-below step). A new control raises the same jet mass by the same 20 GeV across 125 GeV by rotating the hardest hadron instead of the lepton. With that edit the networks follow the mass-only classifier (excess −0.12 [−0.34, +0.12] log-odds). So crossing 125 GeV in visible mass is not by itself what the networks penalise: the extra drop needs the lepton to move away from the quark particles, and it is largest near the edge. Still open: whether this is a lepton–hadron pair-mass endpoint or some other piece of lepton–quark geometry (the hadron edit also changes the mass of the quark system). For leptonic top jets, H48's statement that full-input ParT shows no extra drop at 152.6 GeV holds. The four-momentum-only models' drop at that edge is also lepton-specific, and it keeps growing above the edge instead of being a step at it.

## What was checked

### Code review of `interp/h48_lepton_endpoint.py`

- **Rows.** Rows are drawn at random from each file (`default_rng(4801+k).choice`, sorted, read in chunks), never first-k. Labels are asserted per class.
- **Axis and convention.** Every variant, the unedited jet included, is rebuilt from four-vectors about its own E-scheme axis through `part_core.build`, which uses η = jet_eta + sign(jet_eta)·Δη. Particle-level arrays (PID, impact parameters) are re-sorted with the same pT order as the kinematics. The H48 log shows that rebuilding changes 2 of 12,000 ParT_full calls.
- **Padding.** `p4` multiplies by the mask, so padded slots enter neither m_vis, the lepton index nor B.
- **Edit conservation.** E and C keep the moved particle's energy and |p| exactly and leave every other particle unchanged. My rerun asserts this: (|p|, E) of the moved particle are equal, and all other slots are bit-identical. E hits the mass target to 10⁻³·m, and C keeps the mass.
- **Observables and inputs.** Observables are computed on the same rebuilt jets the networks see.
- **Working point.** The 50 % signal-efficiency threshold is the median score over unedited signal jets of the same sample, which is correct.
- **Statistics.** The Holm family and the permutation floors are as stated.

Points that matter:

1. **Reference fit on raw jets.** The references are fitted on stored (not rebuilt) jets. `obs(Jr)` therefore takes f_ℓ = pT_ℓ / stored jet_pt for training but pT_ℓ / recomputed jet_pt on the edited jets. The effect is small, and my rerun fits the references on rebuilt jets.
2. **Reference stability.** The MLP(m) reference is not stable at the 0.4 log-odds level for t→bℓν. Its across − below step is −0.56 in H48 (train file `_000`, random_state 0) and −0.95 here (train file `_001`, rebuilt inputs, random_state 1). For H→ℓνqq′ it is stable (−2.13 vs −2.08). The "beyond MLP(m)" numbers for top jets therefore carry an unquantified reference uncertainty of about this size. That is larger than H48's quadrature interval (±0.15).
3. **Jet-edit figure.** The "x" marker (lepton position before the edit) in `h48_jet_edit.png` is not in the legend. Otherwise the figure matches the text: the lepton moves 3.6° and the mass goes from 144.4 to 164.4 GeV under E and stays at 144.4 GeV under C.
4. **Data curve windows.** The data-curve slope test uses ±40 GeV windows. In `h48_response.png` all four networks' mean s_Tbl starts to fall at about 130–140 GeV, before 152.6 GeV, so a ±40 GeV slope difference does not localise a change at the endpoint.

No bugs were found that change H48's numbers.

### Rerun on a fresh sample, plus two controls H48 lacks

- **Sample.** JetClass test_20M `*_101` files (H48 used `*_100`). Random rows: TTBarLep 6,000, HToWW2Q1L 8,000, ZJetsToNuNu 2,000, TTBar 2,000 (`default_rng(4851+k)`). References: LR(m), MLP(m) and MLP(m, lep) as in H48, but trained on train_100M `*_001`, 15,000 random rows per class (rng 4861..4864), on rebuilt jets.
- **Unedited 10-class accuracy per class** (t→bℓν / H→ℓνqq′ / QCD / t→bqq′), which reproduces H48:
  - ParT_full 0.987 / 0.981 / 0.777 / 0.950
  - PN_full 0.982 / 0.977 / 0.757 / 0.933
  - ParT_kin 0.943 / 0.871 / 0.726 / 0.901
  - PN_kin 0.921 / 0.826 / 0.694 / 0.895
  - ParT_perm1 and ParT_perm2 (all weight tensors permuted) 0.35 and 0.05 overall
- **Edits.** In every edit one particle moves, keeping its energy and |p|; the rest of the jet is unchanged.
  - **E_lep / C_lep** (H48's edit). The lepton (hardest isElectron/isMuon particle) is rotated in the plane of (p_rest, p_ℓ) to the opening angle that changes the jet mass by exactly Δm. The control rotates it about the p_rest direction by the same 3-D angle, at fixed opening angle and mass.
  - **E_had / C_had** (new). The same construction applied to the hardest particle not flagged as a lepton, which is a hadron or photon of the b or qq′ system. The lepton does not move. Its isolation changes little (mean ΔI_ℓ at the crossing window: −0.11 for H, compared with −0.65 under E_lep).
  - **Why this control.** Under "the network uses the visible-mass edge m_vis ≤ m_H", any edit that pushes m_vis across 125 GeV should give the extra drop, whichever particle moves. Under "the network responds to the lepton's angle to the quark system", only E_lep should.
- **Mass scan.** The unedited jet mass is binned into 10 GeV windows [X+L, X+L+10] with L = −63, −53, …, +27 and Δm = +20 GeV, with at most 600 jets per window (random subset). X = 125 GeV for H→ℓνqq′, scored with s_Hqql = log p(H→ℓνqq′)/(p(QCD)+p(t→bqq′)). X = 152.6 GeV for t→bℓν, scored with s_Tbl.
  - **below** = L ∈ {−43, −33} (H48's below arm)
  - **cross** = L = −13 (H48's across arm)
  - **above** = L ∈ {+7, +17} (new: the jet starts and ends above the edge)
- **Metric.** D = (E − C)_network − (E − C)_MLP(m), per jet: the network's response beyond what a classifier given only the jet mass predicts. Intervals are 95 % bootstrap over jets (B = 1,000). The p-values for single windows are 5,000 sign flips. Contrasts between windows use 2,000 label permutations. Holm is applied over the 48-test family (4 networks × 2 classes × 2 edits × {cross, cross − below, cross − above}). The permutation floors are p_H 0.0096 and 0.018.

## Results

**H→ℓνqq′, lepton edit (reproduction).** Across − below of E − C (log-odds):

| model | this rerun | H48 |
|---|---|---|
| ParT_full | −3.98 [−4.33, −3.65] | −4.14 |
| PN_full | −3.87 | −3.90 |
| ParT_kin | −3.69 | −3.91 |
| PN_kin | −3.61 | −3.80 |
| MLP(m) | −2.08 [−2.11, −2.05] | −2.13 |
| MLP(m, lep) | −1.76 | −2.19 |
| permuted | +0.06, +0.05 | – |

D by window, ParT_full:
- below −1.25 [−1.39, −1.11]
- cross −3.15 [−3.47, −2.83] (p_H 0.0096)
- above −1.71 [−2.30, −1.16] (n = 173)

Contrasts:
- cross − below: −1.90 [−2.23, −1.57] (p_H 0.018)
- cross − above: −1.44 [−2.06, −0.80] (p_H 0.018)
- PN_full, ParT_kin and PN_kin have the same pattern: cross −3.12, −2.36, −2.12, with every cross − below and cross − above p_H 0.018.
- Against MLP(m, lep), cross − below is −2.22 for ParT_full.

The excess is largest where the shift crosses 125 GeV, and it is also present below and above. The isolation change does not explain the peak: ΔI_ℓ under E_lep is −0.95 to −1.10 below and −0.65 at the crossing.

**H→ℓνqq′, hadron edit (new control).** The mass shift is the same (+20 GeV, crossing 125 GeV in the cross window), and MLP(m) responds identically (cross −2.23 under both edits).

| | ParT_full | PN_full | ParT_kin | PN_kin |
|---|---|---|---|---|
| D at cross | −0.12 [−0.34, +0.12], p_H 1 | −0.04 [−0.29, +0.22], p_H 1 | +0.09 [−0.14, +0.33], p_H 1 | +0.74 [+0.50, +0.95] |
| cross − below of D | +0.93 [+0.66, +1.20] | +0.87 | +0.71 | +0.48 |
| across − below of E − C | −1.14 [−1.40, −0.88] | −1.19 | −1.35 | −1.58 |

MLP(m)'s across − below of E − C is −2.07, so with the hadron edit the networks' step at 125 GeV is smaller than the mass-only prediction. The permuted networks give −0.07 and +0.02.

**t→bℓν, lepton edit (reproduction).**

Across − below of E − C:

| model | this rerun | H48 |
|---|---|---|
| ParT_full | −0.39 [−0.50, −0.28] | −0.50 |
| PN_full | −0.58 | −0.72 |
| ParT_kin | −1.67 | −1.85 |
| PN_kin | −1.36 | −1.44 |
| MLP(m) | −0.95 | −0.56 |

D by window (below / cross / above):
- ParT_full +0.59 / +1.16 / +0.14
- ParT_kin +0.77 / +0.05 / −0.99
- PN_kin +0.75 / +0.35 / −0.80
- PN_full +2.31 / +2.69 / +0.83

The kin models' cross − below D is −0.71 [−0.92, −0.51] and −0.40 [−0.54, −0.26], smaller than H48's −1.29 and −0.89 because this MLP(m) is steeper. Their cross − above D is +1.04 and +1.15 (p_H 0.018): the kin models' extra drop is larger above the edge than at it.

Reverse arm (Δm = −20 GeV from [X+7, X+27], n = 233), E − C:
- ParT_full +0.04 [−0.14, +0.21]
- PN_full −1.25
- ParT_kin +0.32
- PN_kin +0.29
- MLP(m) +1.50

**t→bℓν, hadron edit (new control).** Across − below of E − C:
- ParT_full +0.11 [−0.09, +0.33]
- PN_full −0.40
- ParT_kin −0.06 [−0.22, +0.11]
- PN_kin −0.24
- MLP(m) −0.94

D at cross is +2.54, +2.67, +1.41 and +1.69: no network drops at 152.6 GeV when a hadron carries the mass shift. PN_full raises s_Tbl under both edits (+0.9 to +1.9 in every window), so its positive response is not specific to the lepton.

## Hypotheses

- **H0 (smooth mass dependence: response = MLP(m)).** Excluded for H→ℓνqq′ under the lepton edit (D at cross −3.15, p_H 0.0096). Not excluded for H→ℓνqq′ under the hadron edit (D −0.12 [−0.34, +0.12]).
- **H1a (visible-mass edge: any crossing of 125 GeV gives an extra drop).** Predicts D < 0 at cross under both edits. Excluded by the hadron edit for ParT_full, PN_full and ParT_kin (D ≥ −0.34 at the 95 % bound). PN_kin's D is +0.74.
- **H1b (response to the lepton's angle to the quark system, independent of mass).** Predicts a flat D across windows under E_lep. Excluded: D at cross is more negative than both below and above (p_H 0.018 for all four networks).
- **H1c (lepton-angle response that grows near the edge)** is what remains. The extra response needs the lepton to move and is largest when the move pushes m_vis across 125 GeV. One version is an endpoint in the lepton–hadron-system pair term m_vis² − m_B² = 2 p_ℓ·p_B, which the hadron edit leaves almost unchanged. It is not separated here from other lepton–qq′ geometry. The hadron edit also changes m_B (the qq′ system mass), which the networks may score separately.
- **t→bℓν endpoint.** For ParT_full, no extra drop beyond MLP(m) at 152.6 GeV in either sample (D cross − below ≥ 0). For the kin models, the lepton-edit drop is not a step at the edge: it grows above it (cross − above +1.04, +1.15), is absent under the hadron edit, and is not reversed (reverse +0.3 vs MLP(m) +1.50).

## Issues with H48 as reported

1. **The reading overstates "visible-mass shift".** The result holds only for a shift made by moving the lepton. The hadron control shows that the same visible-mass shift across 125 GeV gives no excess. "Consistent with use of the H→ℓνqq′ visible-mass edge" should be replaced by a lepton–quark-angle response that is largest near the edge.
2. **The t→bℓν "beyond MLP(m)" numbers depend on the reference fit.** MLP(m)'s step differs by 0.4 log-odds between the two trainings. ParT_full "equal to MLP(m)" becomes "no larger than MLP(m)". The kin excess shrinks from −1.29 / −0.89 to −0.71 / −0.40.
3. **The kin-model top drop is not edge-localised.** It is larger above the edge than at it.
4. **Earlier H48 caveats still apply.** One Δm (20 GeV) was used, there is no C2ST of the edited jets, and the H reverse arm is small (40 jets here, 23 in H48).

## Corrected reading

**H→ℓνqq′.** Rotating the lepton away from the other particles so that m_vis rises 20 GeV across 125 GeV lowers all four networks' Higgs score by 3.6–4.0 log-odds more than the same edit below the edge. A mass-only reference predicts 2.1. The excess beyond MLP(m) peaks where the shift crosses 125 GeV, and it reproduces on a second sample. Raising m_vis by the same 20 GeV across 125 GeV by rotating the hardest hadron gives no excess beyond MLP(m) (ParT_full −0.12 [−0.34, +0.12]). The networks therefore do not use the visible mass crossing 125 GeV as such. They respond to the lepton's angle to the quark system, most strongly when that pushes m_vis across the edge.

**t→bℓν.** ParT_full shows no drop beyond MLP(m) at 152.6 GeV. The kin models' drop under the lepton edit grows above the edge, is absent under the hadron edit and does not reverse. PN_full raises s_Tbl whenever a particle is rotated outward.

## Proposed status

D20 for H→ℓνqq′: change from "visible-mass edge used" to "lepton-angle response peaked near the H visible-mass edge; mass crossing alone not used", confidence about 0.35. The t→bℓν endpoint is not used by any of the four networks (about 0.6).
