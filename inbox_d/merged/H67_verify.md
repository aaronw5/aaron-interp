# H67 verification — mass reader or width + pT reader

Verifier script `interp/h67_verify.py`. It imports `interp/h67_mass_vs_width.py` unchanged and changes only the seeds and the output directory. It adds one control (crot_dm) and five reference classifiers.
Jobs (all finished):
- edits: `d-h67-verify` (JetClass), `d-h67-verify-hls`, `d-h67-verify-top`;
- evaluation: `d-h67-verify-jedi2` (`d-h67-verify-jedi` stalled in CephFS D-state and was deleted), `d-h67-verify-gjc`, `d-h67-verify-gtop` (GPU);
- summary and check: `d-h67-verify-sum`.

Outputs:
- `results/physics/h67_verify.json`: constraint check, prong-sharing check, crot_dm statistics, φ against every reference, reproduction;
- `results/physics/h67_verify/`: `h67_summary.json`, `h67_edits_*.json`, `h67_phi.png`, `h67_jet.png`;
- per-jet tables and logits on the volume: `/j-jepa-vol/aaron-interp/results/physics/h67_verify/work/`.

## Conclusion

The fresh-sample rerun reproduces every number. JetClass and Top Landscape taggers respond to both edits, so they read neither the jet mass alone nor the width plus pT alone. The share φ cannot say more than that.

The reason is how the edits are built. The "width at fixed mass" edit and the "mass at fixed width" edit both make the prong pT sharing more unequal by the same amount. They differ mainly in how far the prongs are pulled apart. As a result:
- a classifier that sees only the prong sharing gives φ ≈ 0.52;
- randomly permuted networks give φ ≈ 0.49;
- classifiers that read mass plus any one substructure variable (prong sharing, D2, τ21 or w) give φ between 0.2 and 0.6.

The networks' values (0.33–0.47) therefore do not single out any of these variables.

The hls4ml networks on W and Z jets (JEDI-linear φ = 0.12 on W) cannot be called mass readers on this evidence. Their width-edit response is significantly above a displacement-matched control. Reference classifiers that read mass together with prong sharing or τ21 give equally low φ on these classes.

## What was checked

### 1. Code review of `interp/h67_mass_vs_width.py`

| check | result |
|---|---|
| axis convention | Edits are built about the jet's E-scheme axis (summed constituent three-momentum). JetClass inputs are rebuilt about the stored jet_eta/jet_phi with η = jet_eta + sign(jet_eta)·Δη (`part_core.build`), as required. |
| padded particles | Every map and observable masks by pT > 0. Padded slots are zero in all edited tables (0 non-zero padded entries in any edit, any dataset). `geometry()` asserts that real particles come first. |
| particle alignment | Track/PID features (`ext`) are permuted together with pT in `sortT` and passed to `build` as `extra`, so per-particle features stay with their particles after re-sorting. |
| conservation claims | Recomputed independently from the saved tables with `H.jetp4`, on converged jets. Max \|Δm/m − target\| ≤ 4e-8, \|ΔpT/pT\| ≤ 9e-8, \|Δw/w − target\| ≤ 6e-8. Exceptions are four JetClass edits and the hls crot_dm, at ≤ 8e-6. Jet-axis shift ≤ 5e-5 rad, particle count unchanged. The worker's "1e-14" applies to the float64 arrays. The networks saw float32 pT, which is still exact at 1e-7. |
| sampling | JetClass rows are random per class (`rng.choice`, then a random 2,000/3,000 split), not first-k. hls4ml uses `jedi_core.sample`, Top uses random rows per class. |
| same jets | Observables, references and network logits use the same saved tables. Diverged solves (NaN; 1–7 JetClass jets per edit) are replaced by the unedited jet before evaluation and are excluded from statistics because their residual is not finite. |
| working point | McNemar uses the median of the unedited own-class log-odds per class (the 50 % signal-efficiency point of that class). This is a reasonable choice for a call-flip count. |

No code bug changes a number.

### 2. Rerun on fresh samples

Seeds 6761 (hls4ml), 6762 (JetClass), 6763 (Top). Same sizes as H67: 2,000 evaluated jets per class, plus disjoint reference jets.

Unedited accuracy on this sample:

| model | accuracy |
|---|---|
| JEDI-linear | 0.804 |
| ParT_full, per class W / Z / top / QCD | 0.80 / 0.69 / 0.95 / 0.77 |
| ParT_kin | 0.714 |
| PN_full | 0.783 |
| PN_kin | 0.695 |
| Top ParT / PN / LorentzNet / PFN / EFN | 0.937 / 0.938 / 0.940 / 0.928 / 0.919 |

Every φ on pair "uneq" reproduces:
- φ = Σ|Δs(E_w−10)| / (Σ|Δs(E_w−10)| + Σ|Δs(E_m+10)|), where Δs is the change in own-class one-vs-rest log-odds.
- The new and original 95 % CIs overlap in all 46 network × class cells.
- The largest shift is 0.016 (ParT_kin QCD: 0.390 → 0.406).

| model / class | φ original | φ fresh sample [95 % CI] |
|---|---|---|
| ParT_full Wqq | 0.350 | 0.351 [0.343, 0.359] |
| ParT_full Tbqq | 0.416 | 0.415 [0.408, 0.422] |
| JEDI-linear W | 0.116 | 0.118 [0.113, 0.123] |
| Top ParT top | 0.434 | 0.428 [0.422, 0.433] |

### 3. What the two edits change in the jet

Definitions:
- s_K = 1 − Σ_k (P_k/P)²: prong pT sharing from exclusive-kT subjets. P_k is the pT of subjet k and P is the jet's ΣpT. For two prongs s_2 = 2z(1−z), where z is the softer prong's pT share.
- f: prong-sharing parameter of the edit map. u: opening-angle scale of the edit map.

Medians on converged jets (JetClass):

| edit | target | f | u | Δlog s (Wqq / Zqq / Tbqq / QCD) |
|---|---|---|---|---|
| E_w−10 | w −10 %, m fixed | −0.260 | 1.165 | −0.27 / −0.27 / −0.24 / −0.27 |
| E_m+10 | m +10 %, w fixed | −0.254 | 1.291 | −0.26 / −0.25 / −0.22 / −0.29 |

hls4ml and Top give the same picture (hls W: −0.24 vs −0.22; Top top: −0.24 vs −0.21). On a real W jet (`h67_verify/h67_jet.png`, and also the worker's `h67_jet.png`), both edited panels show the softer prong's share moving from 0.21 to 0.16, or from 0.12 to 0.09. The two edits differ only in the prong opening: about +16 % for E_w and +29 % for E_m.

Consequence: φ does not measure "mass reading versus width reading". It measures how a network's response to (sharing change + small opening) compares with its response to (the same sharing change + larger opening). This is why the following references all fall in 0.2–0.6 (fresh-sample references, pair "uneq"):

| reference (MLP(32,32) on reference jets) | JetClass W / Z / top / QCD | Top top | hls W / Z |
|---|---|---|---|
| MLP(s, pT): prong sharing only, no mass | 0.52 / 0.52 / 0.53 / 0.53 | 0.52 | 0.54 / 0.53 |
| MLP(D2, pT) | 0.59 / 0.59 / 0.55 / 0.59 | 0.56 | 0.61 / 0.60 |
| MLP(m, s, pT) | 0.26 / 0.23 / 0.46 / 0.39 | 0.53 | 0.05 / 0.08 |
| MLP(m, D2, pT) | 0.28 / 0.26 / 0.57 / 0.42 | 0.60 | 0.09 / 0.13 |
| MLP(m, τ21, pT) | 0.27 / 0.21 / 0.26 / 0.37 | 0.28 | 0.05 / 0.08 |
| MLP(m, w, pT) | 0.29 / 0.24 / 0.41 / 0.49 | 0.35 | 0.08 / 0.10 |
| GBDT(EFP) | 0.36 / 0.37 / 0.49 / 0.38 | 0.44 | 0.11 / 0.13 |
| permuted ParT_full (null) | 0.48 / 0.50 / 0.50 / 0.49 | 0.46 (perm. ParT) | – |
| MLP(m, pT) / MLP(m_A, pT) / MLP(w, pT) | 0 / ≤ 0.05 / 1 | 0 / 0.04 / 1 | 0 / 0.002 / 1 |

Network φ values against these references:
- Inside a 95 % CI: GBDT(EFP) for ParT_full Wqq and QCD, ParT_kin Zqq and PN_full QCD; MLP(m, w, pT) for ParT_full and PN_full Tbqq; MLP(m, s, pT) for ParT_full QCD, PN_full QCD and PN_kin Zqq; MLP(m, τ21, pT) for PN_kin QCD; MLP(m, D2, pT) for Top EFN QCD.
- No network CI contains MLP(m, pT), MLP(m_A, pT) or MLP(w, pT).

The exclusions in the H67 reading hold: pure mass, pure m_A and pure (w, pT) are excluded. Which second variable the networks use alongside mass is not resolved, and the w-versus-mass framing of φ ("share of the response that goes to width") is not what φ measures.

### 4. Missing control: displacement-matched prong rotation (crot_dm)

- **Edit.** Each prong is rotated about its pT-weighted centroid by one angle θ per jet, chosen so that the pT-weighted particle displacement equals E_w−10's on the same jet: 2 sin(θ/2) times the mean distance to the prong centroid, sign random. (f, u) are then re-solved so that Δm = Δw = 0.
- **Achieved displacement.** Median JetClass 0.027 rad, vs E_w−10 0.026 rad and crot 0.048 rad. hls 0.0137 vs 0.0134; Top 0.039 vs 0.038.
- **Held fixed.** N, m, w, jet pT and axis are held fixed. D2 and τ21 change by ≤ 0.003 and ≤ 0.0012 (medians).
- **Why it is the right comparison.** It moves the particles as far as the width edit does, but changes none of the tested quantities.

Hypotheses and test:
- H0: mean |Δs(E_w−10)| = mean |Δs(crot_dm)| on the same jets.
- H1: they differ.
- Test: paired bootstrap over jets (1,000 resamples), Holm-corrected over 31 network × class cells.

Result: all 31 cells reject H0 (Holm p ≤ 0.031; the bootstrap resolution floor is 1/1000 before correction).

| model / class | n | \|Δs\| E_w−10 | \|Δs\| E_m+10 | \|Δs\| crot | \|Δs\| crot_dm | E_w − crot_dm [95 % CI] | φ_excess(crot) → φ_excess(crot_dm) [CI] |
|---|---|---|---|---|---|---|---|
| ParT_full Wqq | 1491 | 1.59 | 2.96 | 1.04 | 0.68 | +0.90 [+0.82, +0.98] | 0.22 → 0.28 [0.27, 0.30] |
| ParT_full Tbqq | 596 | 3.73 | 5.27 | 1.76 | 1.72 | +2.01 [+1.77, +2.25] | 0.36 → 0.36 |
| ParT_full QCD | 1109 | 0.97 | 1.52 | 0.88 | 0.42 | +0.55 [+0.49, +0.62] | 0.13 → 0.33 [0.32, 0.35] |
| PN_kin Wqq | 1491 | 0.52 | 1.81 | 0.60 | 0.42 | +0.10 [+0.07, +0.14] | 0.00 → 0.07 [0.05, 0.09] |
| Top ParT QCD | 2599 | 0.61 | 0.86 | 0.69 | 0.38 | +0.22 [+0.20, +0.25] | 0.00 → 0.32 [0.30, 0.33] |
| JEDI-linear W | 1629 | 0.69 | 5.19 | 0.60 | 0.52 | +0.17 [+0.13, +0.20] | 0.02 → 0.035 [0.027, 0.042] |
| JEDI-linear Z | 1588 | 0.94 | 3.11 | 0.83 | 0.74 | +0.20 [+0.15, +0.25] | 0.05 → 0.08 [0.06, 0.10] |

Here φ_excess is φ computed after subtracting the control's |Δs| sum from both edits' sums.

What this changes:
- The H67 statement "φ_excess 0 on QCD" came from the oversized crot. With the matched control, QCD-class jets respond to E_w above the floor in every JetClass and Top network (φ_excess 0.16–0.33).
- The H67 statement that JEDI-linear's E_w response on W and Z is "within 0–4 % of the crot floor" is incorrect as worded. It is 15 % (W) and 13 % (Z) above crot, and 33 % and 27 % above crot_dm, which is significant. What is small is φ_excess (≤ 0.08).

### 5. Figures

- `h67_phi.png`: the data match the JSON. The title is cut at the right edge ("95 % boots…"). The x-axis label describes φ as "0: responds only to mass edit; 1: only to width edit". This is true for those two extreme readers, but the figure has no marker for the prong-sharing-only reader at about 0.5.
- `h67_jet.png`: correct. It shows the same change in prong share in both edited panels, which is the point of section 3. The H67 text does not draw that conclusion from it.

## Issues with the H67 reading

1. **φ is not a mass/width mixing share.** Both edits change the prong sharing by the same amount (Δlog s −0.27 vs −0.26 on JetClass W). A sharing-only reader gives φ = 0.52, a D2-only reader 0.55–0.60, and permuted networks 0.49. φ between 0 and 1 therefore excludes only the two extreme readers. It does not place the networks "between" mass and width.
2. **Reference assignment is incomplete.** In the original set only GBDT(EFP) and MLP(m, w, pT) were candidates. MLP(m, s, pT), MLP(m, τ21, pT) and MLP(m, D2, pT) fall inside network CIs in several cells. None of the mass-plus-one-variable references is preferred overall.
3. **hls4ml "mass readers" (D9).** JEDI-linear W φ = 0.118 [0.113, 0.123] excludes MLP(m_A, pT) (0.002) and MLP(m, pT) (0). It is no lower than MLP(m, s, pT) (0.047), MLP(m, τ21, pT) (0.053) and MLP(m, w, pT) (0.084) on the same jets, and the E_w response is above the matched floor. On hls W and Z, reading mass plus a prong-sharing variable produces the same low φ, so a low φ does not identify a mass-only reader.
4. **"Within 0–4 % of the crot floor"** should read "φ_excess ≤ 0.05". The response itself is 13–15 % above crot.
5. **"φ_excess 0 on QCD"** is an artefact of crot moving particles about twice as far as E_w. With crot_dm, φ_excess on QCD is 0.16–0.33.
6. **Selected subset.** Pair "uneq" uses the 70 % of jets (JetClass) and about 50 % (Top) where both solves converged. Jets whose prongs already have very unequal sharing are under-represented. The accuracies are for the full sample.
7. **A4 sentence.** "The A4 prediction that both edits move m_A does not hold" describes the edit, not the network. E_w keeps m_A fixed because, for these jets, m_A ≈ m about the E-scheme axis. The m-versus-m_A question is untested here, as the worker's thesis note says.

## Corrected reading

The edits are exact: jet m, pT, η, φ fixed to ≤ 1e-7, w changed by ±10 %, padding untouched. The φ values reproduce on a fresh sample within 0.016.

- On JetClass and Top, every tagger responds to the width edit at fixed mass well above a displacement-matched prong rotation (all 31 cells, Holm p < 0.05), and to the mass edit at fixed width. Pure mass, pure m_A and pure (w, pT) readers are excluded.
- Both edits change the prong pT sharing by the same amount and differ mainly in prong opening angle. Reference classifiers reading mass plus prong sharing, D2, τ21 or w all give φ between 0.2 and 0.6, a sharing-only reader gives 0.52, and permuted networks give 0.49.
- The networks' φ = 0.33–0.47 therefore does not identify the second variable. It is compatible with several mass-plus-substructure readers and with GBDT(EFP).
- On hls4ml W and Z, JEDI-linear and the small networks respond weakly to the width edit: |Δs| is about 13 % of their mass-edit response on W and about 30 % on Z. Their response is still significantly above the displacement-matched floor, and their φ (0.07–0.23) matches mass-plus-sharing references, so the data do not make them pure mass readers.

## Corrected Conclusion

At fixed jet mass and pT, JetClass and Top Landscape taggers change their output when the jet is edited so that its pT is shared less evenly between its prongs. The response is 1.2–4 times larger than the response to rotating each prong by the same distance (2.3 times for ParT on W jets). So these taggers read more than mass. They also do not read width and pT alone, because they respond to a mass change at fixed width.

The measurement cannot say which second quantity they use: the width, the prong pT sharing, D2 or τ21. Both edits change the prong sharing equally, and classifiers built on mass plus any one of these give similar responses.

The hls4ml networks respond only weakly to the width edit on W and Z jets, about 13 % of their mass-edit response on W. That is also what a classifier using mass plus prong sharing does on those jets, so it does not show that they use mass alone.

## Suggested next step (not run)

Use an edit at fixed m, w and pT that changes s through the third degree of freedom: move pT between the core and the edge of a prong while u and f compensate. Alternatively, use a 3-target solve (m, w, s) with a third lever. Then E_w would be "w at fixed m and s", and the sharing-only and width-only readers would predict different φ.

## Glossary additions

- **s_K**: prong pT sharing, 1 − Σ_k (P_k/P)² over K exclusive-kT subjets; for two prongs s_2 = 2z(1−z).
- **crot_dm**: displacement-matched prong rotation. Each prong is rotated about its centroid by the angle that moves particles as far as E_w−10 does, then m and w are restored exactly.
- **φ_excess(crot_dm)**: φ computed after subtracting crot_dm's Σ|Δs| from both edits' sums.
- **MLP(s, pT), MLP(m, s, pT), MLP(m, D2, pT), MLP(m, τ21, pT), MLP(D2, pT)**: reference MLP(32, 32) classifiers trained on the disjoint reference jets, using those features in logarithm.
