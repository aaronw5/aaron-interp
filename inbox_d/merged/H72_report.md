# H72 report: angular reach, i.e. which annulus of the jet each tagger reads (loop D, round D15)

Node H72, new D claim (links D1/D25 mass, D24 count, D10 pileup). Script `interp/h72_angular_reach.py`.

- Results: `results/physics/h72_summary.json` (all numbers), plus `h72_{hls,jc,top,qg}_{nets,efp}.json`. The per-jet npz files stay on the volume under `/j-jepa-vol/aaron-interp/results/physics/`.
- Figure: `results/physics/h72_reach.png`.
- Status: done (exploratory).

## Conclusion

We removed the real particles that lie beyond a radius r_cut from the jet axis, or pulled them in to r_cut, and measured how far out each tagger's calls start to change.

**Top taggers (JetClass and Top Landscape ParT, ParticleNet, LorentzNet)**
- They respond from the same radius as a plain-mass window, a soft-drop-mass window or a particle-count likelihood ratio: half of their response is reached at ΔR ≈ 0.46–0.48 on R = 0.8 jets.
- When the original jet mass and pT are restored after the removal, these networks still lose most of their response (Top ParT: 27 of 35 lost percentage points of top efficiency at r_cut = 0.4).
- So they do not follow plain mass alone. The removal also lowers the particle count, and this is consistent with the count channel of D24.
- The Top PFN behaves differently. Its response disappears once mass and pT are restored, as a mass reader's would.

**Quark–gluon networks**
- Removing particles beyond r_cut moves them strongly towards "quark".
- Once mass and pT are restored (count still lowered), the response falls to about zero. A count-only reference keeps its full response under the same edit.
- Here the networks follow the jet's energy-weighted angular spread more than the particle count.

**W/Z networks**
- They respond only to the innermost cut (ΔR ≈ 0.25–0.27), as all the references do.

**Still uncertain**
- The two location controls cannot be read cleanly at small r_cut. Random inside removal is possible only for jets with more particles inside than outside (17 % of Top jets at r_cut = 0.3). Random azimuthal rotation breaks the second prong of W/Z/top jets.
- The reach measured on signal efficiency saturates for top jets, because all top calls are lost by r_cut = 0.3.

## Question and why

From how far out of the jet axis does each tagger take its information? Peripheral radiation weights the plain jet mass m (m² ≈ Σ_ij pT_i pT_j ΔR_ij²) and the girth g = Σ pT_i ΔR_i / Σ pT_i. Grooming removes it from the soft-drop mass m_SD. It also carries part of the particle count N. H49/H65 added soft ghost particles inside and outside the soft-drop core but never removed or moved real peripheral radiation.

## Procedure

**Geometry**
- ΔR_i = √(Δη_i² + Δφ_i²) is the distance of particle i from the E-scheme axis. The E-scheme axis is the direction of the summed 3-momenta of the particles the model sees, after its slot cut.
- r_cut = f·R with f ∈ {0.75, 0.5, 0.375, 0.25}:
  - r_cut = 0.6 / 0.4 / 0.3 / 0.2 for the R = 0.8 jets (hls4ml, JetClass, Top Landscape);
  - r_cut = 0.3 / 0.2 / 0.15 / 0.1 for the R = 0.4 quark–gluon (QG) jets.
- "Outer" particles are the real particles with ΔR > r_cut.
- Model inputs are always computed about the stored, unedited jet axis, as in H65.

**Arms (edits of the real jet)**
- **Rm:** remove every outer particle. N, pT, m, m_SD and girth all fall.
- **Pi:** move every outer particle radially along its own direction from the axis to ΔR = r_cut, keeping its pT (and particle mass). N and summed pT stay fixed; only angles change, so m and girth fall less than under Rm.
- **Rm+m:** Rm, then restore the original jet m and pT.
  - Every remaining momentum is scaled by one factor a, which restores pT.
  - Every angular offset from the remaining jet's own E-scheme axis is scaled by one factor b, which restores m. The two steps are iterated 8 times.
  - A jet counts as valid if |m/m0 − 1| < 2 %, |pT/pT0 − 1| < 1 % and b ≤ 5.
  - What changes: N is lowered and the angular pattern is rescaled. m, pT and (approximately) girth are held fixed.

**Controls and why**
- **Cr (for Rm): random inside removal, same N and pT lost, different location.**
  - For each outer particle (in random order), remove the unused inside particle (ΔR ≤ r_cut) nearest to it in log pT.
  - Valid if the removed pT matches within 20 % and at least one inside particle remains.
  - Rm − Cr isolates "where the removed particles were" at matched count and pT loss.
- **Rot (for Pi): random azimuthal rotation of the outer particles.** Each outer particle gets an independent random azimuth about the axis at its own ΔR. The annulus, the radii, pT and N are unchanged; only the angles are scrambled. Pi − Rot compares "moved inward" with "moved within the same annulus".
- **Cr+m (for Rm+m):** Cr followed by the same m, pT restoration.
- **Nulls:** JEDI with redrawn weights; ParT_full with permuted weights (seed 7) and untrained; Top PFN permuted; QG part_kin permuted.

**Exclusions.** Jets left with no particle, or given a non-finite score by any network, are excluded from that condition for every tagger. This affects only r_cut = 0.2 (QG 0.1): 4 / 3 / 17 / 2 / 3 jets in JetClass W / Z / t, Top and QG. In these jets the E-scheme axis lies between prongs with no particle within r_cut.

**Samples.** Random rows; seeds 7201–7204; 2,000 jets per class.
- hls4ml: `jedi_core.sample(2000, 7201)`, 10,000 jets in 5 classes.
- JetClass: first test file per class, QCD / Zqq / Wqq / Tbqq, 8,000 jets.
- Top Landscape test: 4,000 jets.
- QG Pythia test: 4,000 jets.

**Models.** As in H65.
- hls4ml: JEDI-linear n64 3-feature.
- JetClass: ParT_full, ParT_kin, PN_full, PN_kin.
- Top Landscape: ParT, PN, LorentzNet, PFN, EFN.
- QG: part_kin, pnet_kin, pfn_k, efn_k, lorentznet.

**References, evaluated on the same edited jets**
- win_m: plain-mass window. win_msd: m_SD window (C/A, β = 0, z_cut = 0.1). Each is the shortest interval holding 50 % of unedited signal; for QG, m below the quark median.
- LR_girth and LR_N: histogram likelihood ratios of girth and of N.
- MLP_mptN: an MLP on (m, log pT, N).
- GBDT_EFP: gradient-boosted trees on the 54 log energy-flow polynomials (d ≤ 5, β = 1, energyflow 1.3 in venv-ef).
- The LR, MLP and GBDT references are 2-fold cross-fit on the unedited sample, so each jet is scored by the half not trained on it.

**Metrics.** Computed on the signal jets of each task (signal class vs QCD; QG: quark vs gluon).
- **Δε_S:** change of the fraction of signal jets still called signal, at the threshold that keeps 50 % of unedited signal (ε_S = 0.5).
- **Δs:** mean change of the network's log-odds score.
- **Arm − control:** per-jet paired difference, with a 95 % bootstrap CI over jets (B = 1000).
  - H0: arm and control move the score equally. Tested with a sign-flip permutation test (10⁴ flips, so the smallest p is 1e-4) and an exact McNemar test on the calls.
  - Holm correction over 300 tests (trained nets × r_cut × arms, all datasets) in each family.
- **r_50:** the r_cut at which |Δε_S| of an arm first reaches half its value at f = 0.25. It is found by walking inward from (R, 0) with linear interpolation, with a bootstrap CI. A larger r_50 means the tagger reacts to particles further out.

**Runs (cluster).**
- d-h72-hls (jax env).
- d-h72-gpu (JetClass, Top and QG nets).
- d-h72-obs (conda; observables and particle lists).
- d-h72-efp (venv-ef, 4 CPU, 45 min).
- d-h72-sum3 (summary). Two summary bugs were fixed before it: d-h72-sum2 added the empty-jet exclusion, and d-h72-sum3 set the excluded non-finite scores to zero so the means are finite. Neither fix changed a d_effS value outside r_cut = 0.2.

## Unedited accuracies (sample used; reproduce known values)

| Dataset | Model | This sample | Known value |
|---|---|---|---|
| hls4ml | JEDI-linear | 5-class 0.8102 | 0.8092 |
| hls4ml | JEDI redrawn | 0.200 | chance |
| JetClass | ParT_full | QCD / Zqq / Wqq / Tbqq 0.764 / 0.696 / 0.801 / 0.953 | 0.783 / 0.689 / 0.805 / 0.956 |
| JetClass | ParT_kin | 0.709 / 0.560 / 0.696 / 0.896 | 0.718 / 0.550 / 0.702 / 0.912 |
| JetClass | PN_full | 0.756 / 0.652 / 0.786 / 0.947 | 0.764 / 0.652 / 0.784 / 0.949 |
| JetClass | PN_kin | 0.682 / 0.542 / 0.673 / 0.906 | 0.693 / 0.528 / 0.680 / 0.913 |
| JetClass | ParT_full permuted / untrained | about 0 per class | null |
| Top | ParT / PN / LNet / PFN / EFN | AUC 0.986 / 0.985 / 0.987 / 0.979 / 0.975 | 0.986 / 0.986 / 0.987 / 0.982 / 0.975–0.979 |
| Top | PFN permuted | AUC 0.695 | null |
| QG | part_kin / pnet_kin / pfn_k / efn_k / lorentznet | AUC 0.902 / 0.903 / 0.893 / 0.884 / 0.921 | 0.901 / 0.902 / 0.88–0.90 / 0.884 / 0.917 |
| QG | part_kin permuted | 0.555 | null |

References (cross-fit AUC):

| Reference | hls4ml W / Z / t | JetClass W / Z / t | Top | QG |
|---|---|---|---|---|
| GBDT_EFP | 0.954 / 0.947 / 0.927 | 0.893 / 0.893 / 0.948 | 0.943 | 0.802 |
| LR_N | – | – | 0.70 | 0.84 |
| LR_girth | – | – | 0.93 | 0.73 |

## What each edit does to the jet (signal jets, mean)

| Dataset | Outer particles at r_cut 0.6 / 0.4 / 0.3 / 0.2 (QG 0.3 / 0.2 / 0.15 / 0.1) | Share of jet pT |
|---|---|---|
| hls4ml | 0.0 / 0.1 / 3.3 / 9.1 | – / – / 1.3 / 5.7 % |
| JetClass | 2.4 / 6.7 / 10.7 / 17.1 | 1.5 / 5.9 / 11.4 / 23.2 % |
| Top | 6.7 / 16.8 / 24.0 / 32.8 | 2.8 / 10.8 / 20.1 / 36.0 % |
| QG | 5.0 / 11.6 / 16.0 / 21.7 | 1.6 / 5.3 / 8.8 / 15.7 % |

Effect of each edit on JetClass top jets at r_cut = 0.4:

| Edit | Δm (GeV) | Δm_SD (GeV) | ΔN |
|---|---|---|---|
| Rm | −68 | −60 | −13.6 |
| Pi | −21 | −10 | 0 |
| Rm+m | 0 | −1.3 | −13.4 |
| Cr | −18.5 | −15.8 | −11.7 |
| Rot | −4.4 | −21 | 0 |

Rot shifts m_SD by −21 GeV because it scatters a prong; it is not a substructure-neutral control.

Valid fraction of Cr at the four r_cut values:

| Dataset | Cr valid |
|---|---|
| hls4ml | 0.99 / 0.97 / 0.85 / 0.64 |
| JetClass | 0.90 / 0.79 / 0.64 / 0.39 |
| Top | 0.90 / 0.50 / 0.17 / 0.02 |
| QG | 0.76 / 0.40 / 0.18 / 0.04 |

Rm+m is valid on ≥ 0.81 of jets everywhere.

## Results

### 1. Reach r_50 under removal (Rm)

r_50 is an absolute ΔR with its 95 % CI. All values below use Δε_S.

**Top jets, R = 0.8**

| Tagger | JetClass t | Top Landscape |
|---|---|---|
| ParT_full / ParT | 0.480 [0.472, 0.489] | 0.469 [0.46, 0.48] |
| PN_full / PN | 0.471 [0.46, 0.48] | 0.46 [0.45, 0.47] |
| ParT_kin | 0.463 [0.45, 0.47] | – |
| LNet | – | 0.459 [0.45, 0.47] |
| PFN | – | 0.297 [0.29, 0.31] |
| EFN | – | 0.32 [0.31, 0.33] |
| win_m | 0.497 | 0.544 |
| win_msd | 0.470 | 0.516 |
| LR_N | 0.458 | 0.472 |
| LR_girth | 0.422 | 0.453 |
| GBDT_EFP | 0.443 | 0.471 |
| MLP_mptN | 0.556 | 0.568 |

**JetClass W, R = 0.8**
- ParT_full 0.251 [0.250, 0.257], ParT_kin 0.262 [0.254, 0.441].
- References: win_m 0.273, win_msd 0.267, LR_N 0.271, LR_girth 0.278, MLP_mptN 0.322.
- JetClass Z is similar: nets 0.25–0.27, references 0.29–0.30.

**hls4ml**
- The jets are compact: the median particle ΔR is 0.093, and only 5.7 % of the pT lies beyond 0.2.
- JEDI t: 0.267 [0.263, 0.271]. References 0.26–0.29.
- JEDI W/Z: the Rm response is not monotonic. The W efficiency rises by +0.09 at r_cut = 0.3 and then falls back to +0.06 at 0.2, while Δs reaches −0.59 at 0.2. r_50 is not read for W/Z.

**QG, R = 0.4**

| Tagger | r_50 |
|---|---|
| part_kin | 0.226 [0.22, 0.23] |
| pnet_kin | 0.219 |
| lorentznet | 0.211 |
| pfn_k | 0.198 |
| efn_k | 0.194 |
| win_m | 0.237 |
| LR_N | 0.230 |
| GBDT_EFP | 0.241 |
| LR_girth | 0.198 |
| win_msd | 0.167 |

**Nearest reference** (L2 distance between the Δε_S(r_cut) curves; bootstrap frequency):
- JetClass W/Z nets: LR_N (0.53–0.99).
- JetClass t: win_msd / win_m.
- Top ParT / PN / LNet: LR_N (0.58) or GBDT_EFP (0.4).
- Top PFN / EFN: LR_girth (0.80).
- QG part_kin / pnet_kin / pfn_k: LR_N. QG efn_k / lorentznet: GBDT_EFP.

### 2. Mass and pT restored (Rm+m) against plain removal (Rm)

Values are Δε_S on signal jets. On Rm+m, win_m is 0 by construction.

**JetClass t at r_cut 0.6 / 0.4 / 0.3**
- ParT_full: Rm −0.09 / −0.36 / −0.47; Rm+m −0.11 / −0.35 / −0.45.
- PN_full: Rm+m −0.08 / −0.32 / −0.44.
- References on Rm+m: win_msd −0.02 / −0.02 / −0.07; LR_N −0.11 / −0.30 / −0.45; MLP_mptN +0.02 / +0.01 / −0.09.

**Top at r_cut 0.4**
- Rm → Rm+m: ParT −0.35 → −0.27; PN −0.34 → −0.26; LNet −0.34 → −0.22; EFN −0.10 → −0.10; PFN −0.08 → +0.03.
- LR_N −0.35 → −0.35.

**JetClass W at r_cut 0.3 / 0.2**
- ParT_full Δs: Rm −1.11 / −4.00; Rm+m −0.78 / −1.35.
- ParT_full Δε_S: Rm 0.00 / −0.19; Rm+m −0.06 / −0.15.

**hls4ml t at r_cut 0.2**
- JEDI Δε_S: Rm −0.41; Rm+m −0.08. Restoring m and pT removes about 80 % of the response.
- MLP_mptN: −0.42 → −0.05.

**QG at r_cut 0.2 / 0.15 / 0.1** (part_kin Δs: Rm +1.10 / +1.43 / +1.68; Rm+m +0.15 / +0.32 / +0.70)
- part_kin: Rm +0.28 / +0.38 / +0.47; Rm+m −0.00 / +0.05 / +0.16.
- lorentznet: Rm+m +0.05 / +0.11 / +0.20.
- efn_k: Rm+m −0.12 / −0.13 / −0.12.
- LR_N: Rm+m +0.27 / +0.36 / +0.44, the same as its Rm.

### 3. Pull-in at fixed N (Pi) against removal (Rm)

Values are Δε_S.
- **JetClass t, ParT_full:** Pi −0.11 / −0.28 at r_cut 0.4 / 0.3, against Rm −0.36 / −0.47. Pi lowers m by 21 / 39 GeV; Rm lowers it by 68 / 107 GeV. win_m: Pi −0.18 / −0.37.
- **Top ParT:** Pi −0.03 / −0.20 against Rm −0.35 / −0.49.
- **QG part_kin:** Pi +0.04 / +0.11 / +0.27 at 0.2 / 0.15 / 0.1, against Rm +0.28 / +0.38 / +0.47. win_m: Pi +0.12 / +0.19 / +0.28.

### 4. Arm minus matched control

Values are on signal jets where both arm and control are valid: Δε_S difference [95 % CI], Δs difference [CI], McNemar counts (arm only / control only), Holm-corrected McNemar p.

| Tagger, r_cut | Rm − Cr | Pi − Rot | Rm+m − Cr+m |
|---|---|---|---|
| JetClass t ParT_full, 0.4 (n = 1617 / 2000 / 1610) | −0.19 [−0.21, −0.17]; Δs −5.10 [−5.35, −4.87]; 35/346; 5e-63 | +0.24 [0.22, 0.26]; 498/17; 1e-121 | −0.25 [−0.27, −0.22]; Δs −4.29; 49/448; 3e-79 |
| Top ParT, 0.4 (n = 1108) | −0.35 [−0.38, −0.32]; 20/408; 9e-93 | +0.29; 601/20 | −0.33 [−0.36, −0.29]; 47/408; 2e-70 |
| Top PFN, 0.4 | −0.23 | −0.06 | −0.11 [−0.14, −0.08], Holm 3e-9; at 0.3 −0.04 [−0.10, 0.02], Holm 1.0 |
| JetClass W ParT_full, 0.4 | +0.02 [0.00, 0.03]; Holm 1.0 (Δs −0.27 [−0.37, −0.16]) | – | −0.06 [−0.07, −0.04]; Holm 5e-8 |
| QG part_kin, 0.2 (n = 665) | +0.15 [0.12, 0.18]; Holm 1e-19 | – | −0.07 [−0.10, −0.04]; Holm 9e-3 |
| QG lorentznet, 0.2 | – | – | −0.00 [−0.03, 0.02]; Holm 1.0 |
| QG lorentznet, 0.15 | – | – | +0.00; Holm 1.0 |

- Family totals: 252 of 300 permutation tests and 221 of 300 McNemar tests have p_Holm < 0.05.
- The permutation p floor is 1e-4, which gives a Holm floor of 0.03.
- In Pi − Rot, the positive sign for the JetClass, Top and QG networks comes mostly from Rot lowering the score. Rot alone gives JetClass t ParT_full −0.35 at 0.4 and JetClass W −0.11 at 0.3, because the scrambled azimuths break the prong structure (Δm_SD −21 GeV).

### 5. Nulls

- ParT_full permuted / untrained: |Δs| ≤ 0.15 (JetClass) under every edit.
- JEDI redrawn: Δs ≈ 0.
- QG part_kin permuted: |Δs| ≤ 0.13.
- Top PFN permuted: Rm Δε_S −0.46 at 0.4, with Δs −0.42. Sum pooling counts particles by construction, as in H65.

## Reading (scoped to what was measured)

**Competing hypotheses and what the data exclude**

- **(a) The network reads plain mass.** Predictions: r_50 near win_m, and Rm+m ≈ 0.
  - Holds for the Top PFN: Rm+m −0.04 [−0.10, 0.02] relative to Cr+m at 0.3, and +0.03 raw at 0.4.
  - Holds largely for hls4ml JEDI t: 80 % of the response is removed by Rm+m. This is consistent with D1.
  - Excluded for the JetClass and Top ParT / PN / LNet top tags. Rm+m keeps 75–100 % of the Rm loss at r_cut 0.4–0.6, while win_m's Rm+m is 0 by construction and win_msd's is −0.02.
- **(b) The network reads a groomed core (m_SD).** Prediction: Δ small down to r_cut ≈ 0.3.
  - Excluded for the top taggers. At r_cut 0.4, Δε_S is −0.34 to −0.36 for the JetClass nets and −0.34 to −0.35 for the Top nets. win_msd itself loses −0.36 / −0.44 there, because the top's third prong lies near ΔR ≈ 0.4 (the figure shows such a jet).
  - At fixed m and pT (Rm+m), win_msd is near 0 but the nets are not, so the nets do not follow m_SD under this edit.
- **(c) The network counts particles (D24 channel).** Predictions: Rm > Pi, Pi ≈ 0 at fixed N, and Rm+m ≈ Rm.
  - JetClass / Top top taggers: Rm > Pi at r_cut 0.4 (−0.36 vs −0.11 for JetClass ParT_full; −0.35 vs −0.03 for Top ParT) and Rm+m ≈ Rm, as LR_N does. Pi is not 0 at 0.3 (−0.20 to −0.28), and Pi also lowers m by 39–42 GeV there. A count reading plus a mass-like reading is not excluded; a count-only reading is not established.
  - QG: Rm+m ≈ 0 for part_kin and lorentznet, while LR_N keeps +0.27. A count-only reading is excluded for these models under this edit. Restoring m and pT (and with them the girth) removes the response.
- **(d) "QG shows the largest outer reach".**
  - In units of R, QG r_50 / R = 0.49–0.57, about the same as top (0.57–0.60) and larger than W/Z (0.31–0.34).
  - In absolute ΔR, QG (0.19–0.23) is smaller than top (0.46–0.48).
  - The expectation is not supported as stated.

**Limits**
- Δε_S at ε_S = 0.5 saturates at −0.5 for top jets by r_cut = 0.3, which caps r_50 from above. The score-based reach (Δs) is not saturated, but it is dominated by the innermost cut (f_ds ≈ 0.33–0.47 of R everywhere).
- The Cr control exists only for jets with more particles inside r_cut than outside, so Rm − Cr at r_cut ≤ 0.3 (Top, QG) is measured on a compact-jet subset (n = 336 of 2,000 Top top jets at 0.3).
- Rot changes m_SD by −14 to −28 GeV on two- and three-prong jets, so Pi − Rot is not a clean "radius only" comparison.
- Rm+m rescales all angles of the remaining jet by b (median about 1–2). It restores m, pT and approximately girth, but it is a non-physical rescaling of the jet's angular pattern.
- JetClass "full" models: removed charged tracks take their displacement (d0, dz) inputs with them. Pi and Rot keep each particle's type and track inputs.

## Explainer entry (for site/explain.py, to be integrated by the ledger agent)

**H72_EXPLAINER**

**Question.** From how far out of the jet axis do jet taggers take their information? Do they follow quantities that weight the periphery (plain mass, girth), the groomed core (soft-drop mass), or the number of particles?

**Physics background**
- For a boosted jet, $m^2 \approx \sum_{i<j} p_{T,i}\,p_{T,j}\,\Delta R_{ij}^2$, so particles far from the axis add mass in proportion to their angle squared.
- Girth is $g = \sum_i p_{T,i}\,\Delta R_i / \sum_i p_{T,i}$.
- Soft drop (β = 0, $z_{\rm cut} = 0.1$) removes soft wide-angle branches.
- Gluon jets radiate more widely than quark jets.
- A top jet's three prongs are spread over $\Delta R \sim 2m_t/p_T \approx 0.6$ at $p_T \approx 600$ GeV.

**Method**
- With $\Delta R_i$ measured from the E-scheme axis and $r_{\rm cut} = fR$, $f \in \{0.75, 0.5, 0.375, 0.25\}$, three arms are applied:
  - Rm: remove all particles with $\Delta R_i > r_{\rm cut}$.
  - Pi: move them radially in to $r_{\rm cut}$ at fixed $p_T$.
  - Rm+m: Rm followed by $p \to a\,p$ and $\Delta\vec{R} \to b\,\Delta\vec{R}$ that restore $m$ and $p_T$.
- Controls: random inside removal with the same $N$ and $p_T$ loss; a random azimuthal rotation of the outer particles; the same restoration after the inside removal.
- References: mass and $m_{SD}$ windows, LR(girth), LR(N), MLP(m, pT, N), GBDT(EFPs).
- Metric: change of signal efficiency at fixed $\varepsilon_S = 0.5$. $r_{50}$ is the $r_{\rm cut}$ at which half the response at $f = 0.25$ is reached.

**Result**
- Top taggers: $r_{50} \approx 0.46$–$0.48$ (R = 0.8), close to the mass, $m_{SD}$ and LR(N) references.
- With $m$ and $p_T$ restored, ParT keeps 27 of its 35 lost percentage points of top efficiency at $r_{\rm cut} = 0.4$ (Top Landscape). The PFN keeps none.
- QG networks: $r_{50} \approx 0.19$–$0.23$ (R = 0.4). Their response vanishes once $m$ and $p_T$ are restored, while LR(N) keeps +0.27.
- W/Z: $r_{50} \approx 0.25$–$0.27$, together with every reference.
- Figure: `h72_reach.png`.

**Checks**
- Unedited accuracies reproduce.
- Permuted and untrained nets have $|\Delta s| \le 0.15$, except the permuted PFN (sum pooling counts).
- Jets emptied by the cut are excluded (≤ 17 per sample).
- Holm over 300 tests.
- The inside-removal control exists only for compact jets at small $r_{\rm cut}$. The rotation control breaks prongs.

**Physical significance**
- The large top taggers lose their top calls when the outer third of the jet is removed even when mass and $p_T$ are held fixed, which plain or groomed mass alone does not do. This is consistent with the particle-count channel (D24).
- Quark–gluon taggers follow the angular spread that the mass restoration puts back, not the particle count.

## Glossary additions (explain.GLOSSARY)

- **r_cut, outer particles (H72):** the radius from the E-scheme jet axis beyond which particles are edited. Outer particles are those with ΔR > r_cut.
- **Rm / Pi / Rm+m (H72):**
  - Rm: remove the outer particles.
  - Pi: pull them radially in to r_cut at fixed pT (count and summed pT fixed).
  - Rm+m: Rm followed by one momentum scale a and one angular scale b about the remaining jet's axis that restore the original jet m and pT.
- **Cr / Rot / Cr+m (H72):**
  - Cr: remove a pT-matched set of random particles from inside r_cut (same count and pT lost, different place).
  - Rot: give each outer particle a random azimuth at its own ΔR (same annulus, angles scrambled).
  - Cr+m: Cr followed by the same restoration as Rm+m.
- **r_50 (angular reach, H72):** the r_cut at which a tagger's change in signal efficiency reaches half its value at r_cut = R/4, interpolated walking inward from r_cut = R. Larger means the tagger reacts to particles further out.
- **girth g:** Σ pT_i ΔR_i / Σ pT_i about the jet axis, the pT-weighted mean distance of the particles from the axis.
