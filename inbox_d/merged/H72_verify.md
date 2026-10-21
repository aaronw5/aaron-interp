# H72 verify: angular reach (loop D, round D15, skeptic)

Script `interp/h72_verify.py`. Job `d-h72-verify` (GPU, exit 0, 18 min). Results: `results/physics/h72_verify.json` and `h72_verify_{jc,top,qg}.json`. Log: `logs/d-h72-verify.log`.

## Conclusion

The rerun on new jets gives the same numbers as H72. However, H72's explanations of why the networks respond are not supported.

**Top taggers.** Removing the outer particles and then restoring the jet mass and pT still costs ParT, ParticleNet and LorentzNet most of their top calls. Putting the removed number of particles back changes almost nothing, and in some models it goes the wrong way. Particles are put back by splitting existing ones into exactly collinear halves, which leaves mass and every other energy-flow observable unchanged. So the remaining loss is not a particle-count effect. It comes with a loss of three-prong structure: τ32 rises from 0.52 to 0.64.

**Top PFN and quark–gluon networks.** These look unaffected after mass restoration, but that is a cancellation. A count term and an angular-shape term of similar size and opposite sign cancel. Neither "PFN reads mass" nor "QG networks read angular spread rather than count" holds.

**Still open.** The mass restoration stretches the remaining jet's angles by a median factor of 1.4–1.9. It therefore changes the shape of the prongs as well as removing particles.

## Code check (`interp/h72_angular_reach.py`)

**No bugs found in these points:**
- ΔR is measured from the E-scheme axis of the particles the model sees. Padded slots are excluded by the `pt > 0` mask, and model inputs are taken about the stored axis, as in H65.
- Rows are random: `H65.load` uses an rng choice per class, and JetClass uses `jc_sample4`, not the first k rows.
- Observables are computed on the same finalized (slot-cut) jets the networks score.
- The working point keeps 50 % of unedited signal, using the median score of unedited signal jets.
- Rm+m (remove, then restore m and pT) restores m and pT as claimed. On the fresh rows: Δm = −0.005 GeV and ΔpT ≈ 0 for JetClass top jets; −0.002 GeV for Top.

**Design points that affect the reading:**
1. **Rm+m is a large angular stretch.** After removal, all angular offsets of the remaining jet are multiplied by b to restore m.
   - b: median 1.41 / 1.43 (JetClass / Top, r_cut 0.4), 90th percentile 2.3 / 2.2.
   - At r_cut 0.3: median 1.83 / 1.91, 90th percentile 3.7 / 3.6.
   - So Rm+m changes the prong separations and prong shapes, not only the count.
2. **The count reading has the wrong sign.** D24 v2 says these top networks lower the top score when the count rises at fixed IRC-safe content, and so raise it when the count falls. Rm+m lowers N by 13–20 and lowers the top score. That is opposite to the D24 count channel.
3. **The IRC-safe references in H72's own numbers already lose under Rm+m.** IRC-safe means the observable does not change when a particle is split collinearly or a very soft particle is added.
   - GBDT on energy-flow polynomials (EFPs): −0.18 of −0.31 (JetClass t, r_cut 0.4) and −0.17 of −0.34 (Top), in Δε_S.
   - Top EFN: Δs −0.46.
   - So a tagger with no access to the count also loses top calls under this edit.
4. **The headline r_50 is computed on a saturating metric.** Δε_S at the ε_S = 0.5 threshold reaches −0.5 (all top calls lost) by r_cut 0.3 for every top tagger, so r_50 is fixed by the 0.6 and 0.4 points alone.
   - The score-based reach (f_ds in h72_summary.json) separates the models: Top ParT 0.37, PN 0.29, LNet 0.29 in absolute ΔR, against LR_N 0.35 and GBDT_EFP 0.42.
   - So "same radius as a mass cut or particle count" holds for Δε_S only.
5. **W/Z "react only at the innermost cut" holds for Δε_S only.** In log-odds, JetClass ParT_full already moves at r_cut 0.3: W −1.11, Z −2.25.
6. **Different valid subsets in the per-condition Δε_S.** The Cr control (random inside removal) is valid only on compact jets. Their unedited efficiency is not 0.5, which is why Top LNet's Cr Δε_S can reach −0.67. H72's paired arm − control statistics use the common subset and are correct.
7. **Figure `h72_reach.png`.** The figure is correct.
   - The example top jet has a whole prong at ΔR ≈ 0.5, which Rm at 0.4 removes. This illustrates point 3.
   - The legend label "removed at r_cut ≤ 0.2" means removed only at the smallest cut. The wording can be misread.
   - The control curve (blue dash-dot) is on Cr-valid jets only.

## Rerun and new control

**Sample.** Fresh random rows, 2,000 jets per class:
- JetClass: QCD, Zqq, Wqq and Tbqq from the first test file per class, seed 7262. Tested task: top vs QCD.
- Top Landscape test: seed 7263.
- QG Pythia test: seed 7264.

**Models.** JetClass ParT_full, ParT_kin, PN_full, PN_kin; Top ParT, PN, LorentzNet, PFN, EFN; QG part_kin, pnet_kin, pfn_k, efn_k, lorentznet. Nulls: ParT_full with permuted weights, ParT_full untrained, Top PFN permuted, QG part_kin permuted.

**Unedited AUC on these rows:**
- JetClass t vs QCD: ParT_full 0.9986, PN_full 0.9979.
- Top: ParT 0.9873, PN 0.9862, LNet 0.9881, PFN 0.9819, EFN 0.9752.
- QG: part_kin 0.9004, lorentznet 0.9177.
- These match H72 and the known values.

**Edits.** The r_cut index is 1 or 2: 0.4 / 0.3 for R = 0.8 jets, 0.2 / 0.15 for QG.
- **Rm, Rm+m, Cr+m:** as in H72, built with H72's own `conditions()`.
- **Rm+m+split (new):** Rm+m, then the particle count is put back to the unedited N. k = N0 − N random remaining particles are each split into two pT/2 halves 0.01 apart (H1 `split`, z = 0.5, dR = 0.01).
  - This holds the jet four-momentum and every IRC-safe observable fixed and changes only the count.
  - Check on 500 signal jets: Δτ32, Δτ21, ΔD2, Δm_SD and Δgirth between Rm+m and Rm+m+split are all ≤ 0.002 or ≤ 0.02 GeV. ΔN = +14.0 (JetClass) and +19.5 (Top).
  - Rm+m+split − Rm+m is the effect of the count alone.
- **Split (new):** the same number of collinear splits as there are outer particles, applied to the unedited jet. This raises the count at fixed IRC-safe content and measures each network's count slope (the D24 direction) on these jets.

**Metric.**
- Δs: mean change of the log-odds score (signal vs QCD; quark vs gluon) on signal jets.
- Δε_S: change of signal efficiency at the threshold that keeps 50 % of unedited signal.
- Paired tests: paired bootstrap 95 % CI over jets (B = 1000), sign-flip permutation p (10⁴ flips) and exact McNemar test on the calls.
- Holm correction over 168 tests (trained nets). 160 permutation and 148 McNemar tests have p_Holm < 0.05. The permutation floor gives a Holm floor of 0.02.

**Hypotheses for the residual Rm+m loss of the top taggers:**
- **H_count** (H72's reading): the network counts particles.
  - Prediction: Rm+m+split returns the score towards the unedited value, i.e. Rm+m+split − Rm+m is about −(Rm+m − base) and positive.
  - D24 v2 predicts the opposite sign: count up at fixed content moves top jets towards QCD.
- **H_shape**: the network reads prong structure that removal plus stretching destroys.
  - Prediction: Rm+m+split ≈ Rm+m, and τ32 rises under Rm+m.

## Results

**Reproduction of H72 (signal jets, r_cut index 1):**

| Model | Rm Δs (H72) | Rm+m Δs (H72) | Rm+m Δε_S (H72) |
|---|---|---|---|
| Top ParT | −5.45 (−5.42) | −2.32 (−2.35) | −0.25 (−0.27) |
| Top PN | −5.46 (−5.36) | −2.54 (−2.53) | – |
| Top LNet | −6.71 (−6.58) | −1.90 (−1.86) | – |
| Top PFN | −0.94 (−1.05) | +0.16 (+0.09) | +0.06 (+0.03) |
| JetClass ParT_full t | −7.43 (−7.32) | −5.87 (−5.81) | −0.35 (−0.35) |
| QG part_kin | +1.08 (+1.10) | +0.13 (+0.15) | −0.02 (−0.00) |

Rm+m − Cr+m for Top ParT: −2.31 [−2.50, −2.13], McNemar 45/425 (arm only / control only), p_Holm 1e-76. H72 reported −0.33 in Δε_S; the rerun gives −0.34.

**Count restoration at fixed IRC-safe content.** Columns: Rm+m+split − Rm+m Δs [95 % CI] (McNemar p_Holm); split − base Δs (count slope); Rm+m − base Δs.

| Model | Rm+m+split − Rm+m | Split − base | Rm+m − base |
|---|---|---|---|
| Top ParT | +0.14 [+0.08, +0.21] (0.6) | −0.86 | −2.32 |
| Top PN | +0.12 [+0.03, +0.23] (1e-3) | −0.72 | −2.54 |
| Top LNet | −0.54 [−0.63, −0.45] | −1.53 | −1.90 |
| Top PFN | −1.24 [−1.30, −1.18] (2e-111) | −1.37 | +0.16 |
| Top EFN | −0.00 | −0.00 | −0.46 |
| JetClass ParT_full | −1.09 [−1.20, −0.99] | −2.05 | −5.87 |
| JetClass PN_full | −0.89 [−0.99, −0.80] | −1.70 | −4.81 |
| JetClass ParT_kin | +0.59 [+0.53, +0.65] | −0.38 | −3.69 |
| JetClass PN_kin | +0.17 [+0.11, +0.23] | −0.39 | −3.45 |
| QG part_kin | −0.43 [−0.45, −0.41] (8e-44) | −0.31 | +0.13 |
| QG lorentznet | −0.80 [−0.84, −0.76] | −0.58 | +0.38 |
| QG pfn_k | −0.36 | −0.38 | +0.14 |
| QG efn_k | −0.09 | −0.09 | −0.37 |
| Nulls (permuted / untrained ParT, permuted part_kin) | \|Δs\| ≤ 0.05 | | |

r_cut index 2 gives the same pattern:
- Top ParT: +0.94 against an Rm+m loss of −3.86.
- JetClass ParT_full: −1.07 against −8.24.
- QG part_kin: −0.59 against +0.27.

**Observables on 500 random signal jets (base → Rm+m), r_cut 0.4:**

| Observable | JetClass t | Top |
|---|---|---|
| m (GeV) | 171.7 → 171.7 | 175.4 → 175.4 |
| τ32 | 0.49 → 0.63 | 0.52 → 0.64 |
| τ21 | 0.52 → 0.40 | 0.52 → 0.43 |
| m_SD (GeV) | 155.0 → 154.2 | 165.6 → 159.4 |
| girth | 0.237 → 0.241 | 0.25 → 0.25 |
| N | −14.0 | −19.5 |

τ32 = τ3/τ2 is the N-subjettiness ratio with exclusive-kT axes; it is lower for three-prong jets. Rm+m leaves m, girth and (nearly) m_SD in place but moves τ32 from a top-like towards a QCD-like value. Rm+m+split leaves every one of these observables at its Rm+m value.

**What the hypotheses predicted:**

**Top ParT, PN and LorentzNet; JetClass ParT and PN.**
- Restoring the count recovers at most 0.14 / 2.32 = 6 % (Top ParT), 5 % (Top PN) and 16 % (JetClass ParT_kin) of the Rm+m loss.
- For Top LNet, JetClass ParT_full and PN_full it deepens the loss (−0.54, −1.09, −0.89), in the D24 direction.
- H_count is excluded as the explanation of the residual Rm+m loss.
- H_shape is consistent with the result: τ32 rises by 0.12–0.14 at fixed m, and the IRC-safe EFN and GBDT_EFP also lose under Rm+m.

**Top PFN.**
- Rm+m ≈ 0 (+0.16), but restoring the count moves the score by −1.24 [−1.30, −1.18].
- So under Rm+m, the count drop pushes about +1.2 log-odds towards top, and the rest of the edit pushes about −1.1 towards QCD.
- The near-zero response is a cancellation, not a mass reading.

**QG networks.**
- Rm+m ≈ +0.13 (part_kin), but restoring the count moves the score by −0.43, and a count increase on the unedited jet gives −0.31.
- These networks therefore do respond to the particle count at fixed IRC-safe content: part_kin about −0.036 log-odds per particle, lorentznet −0.067. LR_N, the count-only likelihood ratio, gives about +0.09 per particle in H72.
- Under Rm+m, a count term (towards quark) and an angular-stretch term (towards gluon, b median 1.49) cancel.
- "Count-only" stays excluded, because LR_N keeps its full response and the nets do not.
- "Follow angular spread rather than the count" is not supported.

## Corrected reading

**Reach.** Under removal of the particles beyond r_cut, top taggers lose half their top calls at the 50 % working point between r_cut 0.6 and 0.4 (r_50 ≈ 0.46–0.48), close to mass, m_SD and count references. This metric saturates by r_cut 0.3. In log-odds the reach differs between models: about 0.29 for Top PN and LNet, 0.37 for Top ParT.

**Top taggers.** After m and pT are restored by an angular stretch, the top taggers keep most of the loss. This excludes a plain-mass or m_SD-only reading under this edit.

**What drives the remaining loss.** The loss is not driven by the lowered count: restoring the count with exact collinear splits recovers ≤ 16 %, or deepens the loss in the direction of D24. It coincides with a loss of three-prong structure (τ32 0.52 → 0.64 at fixed m), which the IRC-safe EFN and GBDT_EFP also respond to.

**Cancellations.** The near-zero Rm+m response of the Top PFN and of the QG networks is a cancellation between a count term (fixed-content split slope −0.3 to −1.4 log-odds for the removed count) and a shape term of opposite sign.

**W/Z.** The "innermost cut only" statement holds for Δε_S at ε_S = 0.5 only.

**Links to other claims.** H72's links to D24 ("consistent with the count channel") and D1 ("PFN and JEDI follow mass") should be removed. JEDI-linear was not rerun here, but the PFN case shows that Rm+m ≈ 0 is not sufficient for a mass reading.

**Suggested confidence for the new D claim as worded by H72:** 0.2. A narrowed claim at about 0.5 would cover only:
- the reach values at the 50 % working point;
- the exclusion of m- and m_SD-only readings for the top taggers.
