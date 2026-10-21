# H69 verification: width, prong sharing, D2 and τ21 at fixed jet mass and pT

Verifier script: `interp/h69v_verify.py`. It reuses H69's edit solver (`interp/h69_width_sharing_d2.py`: `solve3`, `apply3`, `geometry69`) and H69's network evaluation (`part_eval`).

Jobs:
- `d-h69-verify` (edits, CPU);
- `d-h69-verify-gpj` and `d-h69-verify-gpt` (network evaluation, GPU);
- `d-h69-verify-s2` (summary);
- smoke runs `d-h69-verify-sm` (failed on a missing env variable) and `d-h69-verify-sm2` (10 jets per class).

Outputs:
- `results/physics/h69_verify.json`;
- `results/physics/h69_verify_edits_{jetclass,top}.json`;
- per-jet tables and logits on the volume in `/j-jepa-vol/aaron-interp/results/physics/h69_verify_work/`.

## Conclusion

At fixed jet mass and jet pT, ParticleNet and the full-input ParT on JetClass change their W/Z/H scores more under edits of D2 (and, less clearly, of the prong pT sharing and τ21) than under a random reweighting of particle pT of the same size. The new control removes about half of the response H69 reported against its rotation control: for ParT_full on W jets, the D2-edit response is 2.4–3.5 times that of the matched random reweighting, against 3.4–5.0 times the rotation control. The kinematics-only ParT on JetClass shows no response beyond the random reweighting. On the Top Landscape taggers, the trained networks and their weight-permuted copies respond by similar factors to several of the edits, so a learned dependence on these shape variables is not established there. The pT-weighted width is not unreachable on signal jets; it can be changed at fixed mass, pT and sharing only together with a change of D2 by about 20–45 %, so width and D2 (prong mass) cannot be separated on these jets.

## What was checked

### 1. Code review of `interp/h69_width_sharing_d2.py`

Checked and found consistent:
- **Axis:** all edits restore the E-scheme axis exactly (axis shift ≤ 7e-8 rad), and ParT is fed coordinates relative to the stored jet axis (`H.rel`). The edit tables keep particle order and `ext` features, and `sortT` re-sorts everything consistently.
- **Padding:** every lever is masked by `real`.
- **Sampling:** JetClass rows are random within per-class files, and Top rows are random (`H._rows`); neither is first-k.
- **Same jets:** network inputs are built from the saved edited tables, and the observables are computed on the same tables.
- **Mass and pT:** exact on converged jets.

Three issues were found.

**(a) The headline R values are unpaired.**
- The R values quoted in the report table and Reading come from `class_stats` → `phi[sz]['R']`. That is the mean |Δs| over the jets where each edit converged, divided by the mean |Δs(crot_dm)| over all jets where crot_dm converged. Δs is the change in the own-class log-odds.
- For the top-quark classes the edits converge on only 6–10 % of jets, and those jets respond more strongly to the control. Example: Top ParT, mean |Δs(crot_dm)| = 0.44–0.45 on the E_s-converged jets against 0.19 on all jets.
- The paired R (the same jets for edit and control, in `edits[e]['R']`) on H69's own outputs:

| model and class | edit | reported group R | paired R |
|---|---|---|---|
| ParT_full Tbqq | E_s | 6.8 | 1.38 / 1.79 |
| ParT_full Tbqq | E_D2 | 7.9 | 3.95 / 6.04 |
| PN_full Tbqq | E_s | 4.2 | 0.93 / 1.13 |
| Top ParT top | E_s | 6.5 | 2.17 / 3.26 |
| Top ParT top | E_D2 | 5.9 | 4.05 / 5.11 |
| Top PFN top | E_s | 10.9 | 4.30 / 4.85 |

- On W/Z/H jets the paired and group values agree within about 30 %.
- The ranges "R = 1.8–10.9" (E_s) and "3.3–9.9" (E_D2) in the Reading are therefore inflated at the top end.

**(b) The control crot_dm matches particle displacement but not pT reweighting.**
- Median moved pT share is the fraction of jet pT moved between particles, Σ|ΔpT_i| / (2 Σ pT_i).
- In H69's JetClass outputs: E_s 4.7–5.6 %, E_D2 3.4–3.6 %, E_τ21 5.3–6.1 %, E_w 3.8–5.7 %, against crot_dm 1.1 % and crand 0.0006 %.
- The shape edits are mainly pT reweightings: their particle displacement is 0.002–0.005 rad.

**(c) H_w was untested because of the lever set, not the physics.**
- E_w used the levers f (sharing), u (opening) and c. Lever c holds each prong's pT, centroid and Σ pT r², so it holds each prong's mass to first order. With these levers, w can move only through higher radial moments.
- The prong-mass lever cs was left out of E_w. At fixed m, pT and s, the width can also change by trading prong opening against prong mass (m² ≈ pT² [s Δ²/2 + prong-mass terms], w ≈ s Δ for two prongs, Δ = prong separation).

### 2. Rerun on a fresh sample with new controls

**Samples:**
- JetClass val_5M `<sample>_121.root` (H69 used `_122`): 500 random rows each of Wqq, Zqq, Tbqq, Hbb, QCD (rng 6990).
- Top Landscape test: 1,000 top + 1,000 QCD random rows, excluding H69's 4,000 rows (rng 6991).

**Unedited accuracy on these jets:**
- ParT_full Wqq / Zqq / Hbb / Tbqq / QCD: 0.804 / 0.702 / 0.908 / 0.942 / 0.748.
- PN_full: 0.760 / 0.650 / 0.876 / 0.946 / 0.748.
- ParT_kin 9-class accuracy over these five classes: 0.710; PN_kin: 0.664.
- Top ParT / PN / LorentzNet / PFN / EFN: 0.943 / 0.944 / 0.941 / 0.933 / 0.923.
- Permuted nulls: 0.04–0.57.
- All agree with H69 and with H64.

**Edits:** E_s ±10 %, E_D2 ±10 %, E_τ21 ±10 % and crot_dm, exactly as in H69. Convergence matches H69, e.g. E_D2 −10 %: 0.65–0.73 of W/Z/H/QCD jets and 0.10 of top jets.

**New controls.** Each is matched per jet to the mean moved pT share of that jet's converged shape edits. Mass is exact to ≤ 1e-5 in log.
- **crw:** a random reweighting inside each prong, pT_i × (1 + a·h_i).
  - h = an independent normal number per particle, made pT-weighted-orthogonal inside each prong to {1, Δη, Δφ, r²}, the same construction as H69's width lever.
  - This holds the prong pT, centroid and Σ pT r².
  - Then m, w, s and D2 are restored exactly with the levers f, u, c, cs.
  - Achieved moved share: 5.1 % (JetClass) and 4.8 % (Top), against matched targets of 5.1 % and 4.8 %.
  - τ21 is free: 5–95 % range −7 % to +5 %.
- **crw_sm:** the same with a smooth random profile (random 2nd/3rd-order polynomial of the particle's offset from its prong centroid). Moved share 3.6 % / 2.5 %, because the scale was capped for 70–90 % of jets.
- **crw_ms:** the iid reweighting with only m and s restored. ΔD2 ≤ 0.1 % median; τ21 within ±5 %.
- **Ew±10cs:** H69's width edit with the prong-mass lever cs added (levers f, u, c, cs; targets m, w ±10 %, s).

**Metric.** R_C(E) = mean |Δs(E)| / mean |Δs(C)|, paired over jets where both E and C converged; Δs is the own-class one-vs-rest log-odds change. Uncertainties are 95 % bootstrap intervals over jets (1,000 resamples).

**Test.**
- H0: mean |Δs(E)| = mean |Δs(crw)| on the same jets.
- Large-sample sign-flip test.
- Holm correction over 162 trained-model × class × edit tests: 110 have p < 0.05.

### 3. Results against the pT-reweighting control (fresh sample)

**JetClass, paired R per edit (+10 % / −10 %):**

| model | class | R_crot_dm(E_D2) | R_crw(E_D2) [95 % CI] | R_crw(E_s) | R_crw(E_τ21) |
|---|---|---|---|---|---|
| ParT_full | Wqq | 3.4 / 5.0 | 2.38 [1.99, 2.86] / 3.48 [2.90, 4.24] | 1.72 [1.18, 2.54] / 2.05 [1.48, 3.01] | 1.35 [0.97, 1.85] / 1.87 [1.37, 2.62] |
| ParT_full | Zqq | 4.4 / 6.0 | 2.79 / 4.09 | 1.50 [0.94, 2.57] / 3.07 | 2.00 / 2.04 |
| ParT_full | Hbb | 5.5 / 7.0 | 3.14 / 3.92 | 1.47 [0.90, 2.55] / 1.97 | 1.87 / 1.79 |
| ParT_full_perm0 | W/Z/H | 1.2–1.4 | 1.05–1.31 | 0.93–1.26 | 0.81–1.51 |
| PN_full | Wqq | 2.7 / 4.6 | 2.59 / 3.90 | 2.20 / 2.72 | 1.22 [0.89, 1.70] / 2.07 |
| PN_full_perm0 | W/Z/H | 1.6–2.7 | 1.24–1.58 | 0.93–1.59 | 1.12–1.36 |
| PN_kin | W/Z/H | 3.5–4.4 | 2.04–4.05 | 0.92–2.45 | 1.34–2.99 |
| ParT_kin | Wqq | 1.6 / 1.1 | 0.77 [0.64, 0.90] / 0.48 [0.41, 0.57] | 1.02 / 1.43 | 1.09 / 0.99 |
| ParT_kin | Z/H | 1.5–1.8 | 0.75–0.96 | 0.75–1.26 | 1.07–1.23 |

Network φ_sD2 = M_s / (M_s + M_D2), the share of the response that goes to the sharing edit, reproduces: ParT_full 0.41 / 0.44 / 0.39 (W / Z / H); H69 reported 0.40 / 0.42 / 0.41.

**Top Landscape, top class, paired R_crw (+10 % / −10 %):**

| model | E_s trained | E_s null | E_D2 trained | E_D2 null |
|---|---|---|---|---|
| ParT | 2.15 / 4.97 | 2.16 / 1.50 | 4.22 / 2.82 | 1.43 / 2.96 |
| PN | 2.38 / 5.44 | 1.63 / 1.44 | 2.96 / 3.98 | 1.39 / 3.50 |
| LorentzNet | 2.69 / 4.44 | 1.15 / 1.23 | 2.47 / 4.07 | 1.77 / 4.77 |
| PFN | 4.57 / 6.07 | 12.7 / 25.3 | 4.16 / 3.71 | 12.1 / 9.2 |
| EFN | 8.5 / 11.1 | 9.3 / 9.4 | 16.1 / 11.3 | 9.6 / 5.0 |

- **PFN and EFN:** their permuted nulls hardly respond to crw (mean |Δs| 0.001–0.01), so R is large for trained and null alike. For these two energy-weighted models, crw is not an informative floor.
- **QCD jets, Top dataset:** trained R_crw(E_D2) = 1.8–7.1; the nulls give 1.5–11.

### 4. Width reachability (Ew±10cs) and the w ≈ √(2s)·m/pT identity

**Ew±10cs convergence** (fresh JetClass sample):
- Wqq 0.11 / 0.22, Zqq 0.10 / 0.15, Hbb 0.05 / 0.11, Tbqq ≤ 0.012, QCD 0.42–0.47 (+10 % / −10 %);
- Top dataset: top 0.001, QCD 0.40–0.44.

H69's E_w (without cs) reached ≤ 0.007 of W/Z/H jets.

**Side effect on the converged W/Z/H jets:** median Δlog D2 = −0.21 to −0.30 for +10 % w and +0.32 to +0.39 for −10 % w. τ21 changes with a 5–95 % range up to ±0.8 in log.

**Identity on unedited two-prong signal jets** (w divided by √(2s)·m/pT):
- median 0.96;
- interquartile range 0.90–0.99;
- 5–95 % range 0.72–1.13 (Wqq 0.75–1.13);
- QCD median 0.85, 5–95 % range 0.37–1.21.

## Competing readings and status

- **"The responses are generic pT-reweighting sensitivity"** predicts R_crw ≈ 1. This is excluded for E_D2 on JetClass ParT_full, PN_full and PN_kin, where every CI lies above 1 and above the corresponding null. It is not excluded for ParT_kin (R_crw(E_D2) = 0.48–0.96). It is not excluded for E_s +10 % on ParT_full Zqq/Hbb or for E_τ21 +10 % on ParT_full and PN_full Wqq, where the CIs include 1.
- **H_s (sharing only)** predicts R_crw(E_D2) ≈ 1 at fixed s. It is excluded for ParT_full, PN_full and PN_kin on W/Z/H.
- **H_D2 (D2 only)** predicts R_crw(E_s) ≈ 1 at fixed D2. It is excluded on ParT_full and PN_full for Wqq and for Zqq/Hbb at −10 %; for +10 % on Zqq and Hbb the CIs include 1.
- **Top taggers:** "learned shape dependence" against "architecture-level sensitivity" is not resolved. The trained/null R_crw ratio for ParT, PN and LorentzNet ranges from 0.85 to 3.8 depending on edit sign and model (below 1 for PFN), and for LorentzNet and ParT at E_D2 −10 % the null exceeds the trained network.
- **H_w** is not excluded and not tested. Width can be edited, but only together with a 2–3× larger D2 change, so a width response and a D2 response are confounded on signal jets.

## Corrected reading

**JetClass (ParT_full, PN_full, PN_kin), W/Z/H jets:**
- At fixed m and pT, the E_D2 edit at fixed sharing moves the own-class score 2.0–4.6 times as much as a random within-prong pT reweighting of the same moved pT share that keeps m, w, s and D2. The permuted nulls give 1.0–1.6.
- The E_s edit at fixed D2 and w gives 1.1–3.1 (PN_full and ParT_full; some +10 % CIs include 1).
- The E_τ21 edit at fixed D2 and s gives 1.2–3.1.
- So these networks read D2/prong-mass information and, more weakly, the prong pT sharing.
- The R values against crot_dm reported by H69 overstate the effect by about 1.3–2× on JetClass. On top-quark jets the unpaired group R is inflated further (e.g. ParT_full Tbqq E_s 6.8, against a paired value of 1.4–1.8).

**ParT_kin:** shows no response beyond pT reweighting.

**Top taggers:** trained and permuted networks respond comparably to several edits, so the Top result does not support a learned shape dependence.

**Width:** on two-prong signal jets, w ≈ √(2s)·m/pT holds within ±10 % for the middle half of jets, not within a few per cent for all. A ±10 % width edit at fixed m, pT and s exists on 5–22 % of W/Z/H jets once prong mass may change, and it moves log D2 by 0.2–0.4 (about 20–45 %). So width is tied to D2 rather than fixed by m, pT and s. The statement "H_w cannot be tested" should read "H_w cannot be separated from H_D2 on signal jets".

## Figures

`results/physics/h69_jet.png` is as described: a top jet at pT 568 GeV, with E_s −5 %, E_D2 −5 %, E_τ21 −10 % and crot_dm, and an empty E_w panel.

`results/physics/h69_response.png` has three problems:
- It plots R only against crot_dm.
- It uses the unpaired group R.
- Its title "Trained taggers respond to …" is not supported for the Top rows, where several null rows lie at R = 2–8.

## Glossary additions

- **crw:** random within-prong pT reweighting, matched to the edits' moved pT share, with m, w, s and D2 restored.
- **moved pT share:** Σ|ΔpT_i| / (2 Σ pT_i), the fraction of jet pT that an edit moves between particles.
- **paired R:** edit and control |Δs| averaged over the same jets.
