# P55v: verification of P55 (L-GATr literal attention-head circuits, W vs Z and t vs QCD)

## Conclusion
On a fresh sample, P55's numbers reproduce under its own protocol. Still, part of what P55 reads as "the early blocks act as a group" comes from how a head is switched off. P55 replaces a head's output with the output from another jet. That jet is matched only in transverse momentum, so it usually points in a different direction (median 1.6 rad away in azimuth and 0.9 in pseudorapidity). This verify run instead takes the replacement jet at the same pT and η and rotates it about the beam axis onto the target jet, which is an exact symmetry of L-GATr. With these direction-matched replacements, switching off one head costs on average 2–3 points instead of about 10. Only a handful of heads in blocks 0–2 still matter one at a time, and the 32 top-ranked heads recover 0.72 of the top-vs-QCD accuracy (P55: below 0.6). The claim that a 16-head circuit recovers at least 80 % of the decision is still excluded under every ablation tested. P55's random-set comparison could never pass its multiple-test correction, so it does not show that particular heads are interchangeable. Still uncertain: how many heads a faithful circuit needs, and whether the E4 mass-response numbers change with direction-matched replacements (not rerun).

## Terms
- **Head output O_bh**: the attention output of head h in block b (softmax(q·k/√72)·V; 72 numbers per token: 4 multivector channels × 16 components + 8 scalars), before the head scale and the output projection. The script hooks `blk.attention.attention` in every block, i.e. this quantity.
- **Ablating a head**: overwriting its O_bh. The three variants:
  - **pt donor (P55's main protocol)**: the donor is a random choice among the 5 jets nearest in jet pT, of any class.
  - **aligned donor (new control)**: the donor is a random choice among the 5 jets nearest in (ln pT, η_jet), using z-scored Euclidean distance and any class. All of the donor's particles are then rotated about the beam axis by φ_target − φ_donor, with φ and η taken from the summed lab-frame four-momentum. In both resample variants the donor's head outputs are computed with the unedited network and pasted in, particle k ← donor particle k mod n_donor.
  - **mean**: each token gets the mean O_bh over a reference set, with separate means for the global token, the beam spurion, the time spurion and the particles.
- **Why the aligned control is the right comparison.** L-GATr reads lab-frame four-momenta (E, px, py, pz). The grade-1 part of a head output (a four-vector) and its grade-2–4 parts (bivector, axial vector, pseudoscalar) point along the jet. A pT-only donor sits at a different azimuth φ and pseudorapidity η, so its pasted outputs point away from the target jet. Rotation about the beam axis leaves both of L-GATr's reference ("spurion") tokens unchanged, so it is an exact symmetry of the network. The rotated donor is therefore the same physical donor jet, presented in the target jet's direction. Matching η is the closest available substitute for a longitudinal boost, which is not a symmetry because of the time reference.
- **Circuit of size k**: the top k heads of P55's frozen greedy ranking (`results/lgatr/p55_ranking.json`) compute normally. The other 96 − k are ablated together, and all MLPs are kept.
- **F (faithfulness)** = (acc_circuit − acc_floor) / (acc_unedited − acc_floor). Here acc is binary accuracy (the sign of l_W − l_Z, or of l_t − l_QCD, on the task's true jets), and floor means all 96 heads ablated with the same variant.
- **Block-matched random set**: k random heads with the same number of heads per block as the greedy set.
- **Rank p** = (1 + #random sets with accuracy ≥ circuit) / (1 + n_random).
- **Holm**: step-down family-wise correction, here over the 5 values of k per task and arm.
- **McNemar**: exact binomial test on the jets whose call flips between two configurations.

## Script review (interp/p55_lgatr_head_circuit.py)
- **Hooks and indices.** The attention is re-implemented inside `blk.attention.attention` for each of the 12 blocks, and head index = block·8 + head throughout.
  - w = 1 reproduces the unedited logits exactly (P55 check).
  - The re-implementation agrees with the model's own xformers forward to max |Δlogit| 0.15 with the same call (P49/P55).
- **Padding.** Padded tokens are masked with −inf in the softmax and are never written back (`out[:, G[V]] = O[:, V]`). The particle-token count is asserted against the jet mask.
- **Token order.** Token order (global, two spurions, particles) was checked against the authors' `embed_tagging_data_into_ga`, which prepends spurions and then the global token. This matches the donor map (tokens 0–2 to themselves).
- **Row sampling.** Rows are random permutations of each class file (`rng.permutation`, `rng.choice`), never first-k.
- **Jet build.** The η-mirror bug is absent: `part_core.build` uses η = jet_eta + sign(jet_eta)·Δη. Lab-frame px, py, pz come from this build.
- **No code bug found.** The issues below are about design.
- **Issue 1: the donor ignores jet direction.** pT-only donors in an equivariant network give off-direction donor outputs. This was tested below and is the main finding.
- **Issue 2: the Holm comparison cannot reject by construction.** With 100 random sets the smallest attainable rank p is 1/101 = 0.0099, and Holm over 10 tests multiplies it by 10, giving 0.099. The statement "no ranked set exceeds the block-matched random sets after Holm (best 0.099)" is therefore the floor of the test, not evidence. At W/Z k = 4, 24 and 32 the greedy set beat all 100 random sets (raw p 0.0099).
- **Issue 3: the block-matched null overlaps heavily with the circuit.** At k = 16 the greedy W/Z set holds 7, 7 and 2 heads of blocks 0, 1 and 2 (8 heads each). A block-matched random set therefore shares 12.75 of 16 heads with it on average (t/QCD 11.25 of 16). "Random heads from the same blocks do about as well" says that the last one or two heads chosen in each block matter little. It does not show that the early heads are interchangeable.
- **Minor reporting points.**
  - Removed-alone Holm-significant heads are 72 of 96 for W/Z and 63 for t/QCD, including blocks 6–10, not only "blocks 0–5".
  - The panel (c) title of the figure is cut off at the right edge. Otherwise the figure matches the JSON numbers.

## Procedure (job anrunw-ai-c-p55-verify, RTX 2080 Ti, 2,922 s, exit 0; smoke c-p55-verify-smoke)
- **Script**: `interp/p55v_verify.py` (imports P55's model wrapper, batching and statistics unchanged).
- **Sample**: JetClass test_20M `<class>_114.root` (used by neither P55 nor L-GATr training/validation), `default_rng(5599)`, 500 random rows per class.
  - N = 5,000 jets; 1,000 task jets per task.
  - Mean-ablation reference set: 100 other random rows per class.
- **Rankings**: P55's frozen greedy order (not re-ranked).
- **Per arm (pt, aligned, mean) and task**:
  - unedited, floor, greedy k = 4, 8, 16, 24, 32;
  - 20 block-matched and 10 unrestricted random sets per k;
  - every head removed alone (pt and aligned arms);
  - 10-class accuracy of every circuit on all 5,000 jets.
- **Paired test, aligned vs pt (same jets)**: ΔF with a 1,000-resample bootstrap over jets, a two-sided bootstrap p with Holm over the 10 (task, k) cases, and McNemar on the circuit's calls.
- **Symmetry check**: all 512 jets rotated by random azimuths. Same call 100 %, max |Δlogit| 0.79 (the batch composition differs; P51 found rotation changes 0.05–0.15 % of calls).
- **Donor match**:
  - pt donors: median |Δln pT| 1.2·10⁻⁴, |Δη| 0.89, |Δφ| 1.56 rad.
  - aligned donors: |Δln pT| 4·10⁻³, |Δη| 0.023, |Δφ| ≤ 7·10⁻⁷ after rotation.
  - Donor of the same class: 10 % in both.

## Results (binary accuracy on 1,000 task jets per task; [95 % bootstrap over jets])
- **Unedited L-GATr on this sample**: 10-class 85.16 % [84.16, 86.10]; W/Z 82.0 %; t/QCD 98.0 %.
- **Floors**: 48–51 % binary and 10.0–10.3 % 10-class in all arms.

### Replication of P55 (pt arm), greedy F at k = 4 / 8 / 16 / 24 / 32
| task | this run | P55 |
|---|---|---|
| W/Z | 0.083 / 0.067 / 0.058 [−0.03, 0.14] / 0.153 / 0.248 [0.16, 0.33] | 0.063 / 0.034 / −0.016 / 0.146 / 0.186 |
| t/QCD | 0.060 / 0.134 / 0.243 [0.20, 0.29] / 0.591 [0.54, 0.63] / 0.441 | 0.071 / 0.133 / 0.193 / 0.568 / 0.418 |

- Every value is within the bootstrap intervals, and the k = 24 > k = 32 non-monotonicity for t/QCD reproduces.
- Mean ablation reproduces P55's secondary numbers: t/QCD k = 16 F 0.640 (P55 0.650); W/Z k = 24 F 0.516 (P55 0.516).
- With mean ablation, F = 0 at k ≤ 8 for both tasks, and for W/Z also at k = 16.
- 10-class accuracy with the circuit kept: 10.7–24.7 % (pt), 11.1–28.1 % (aligned), 10.0–39.2 % (mean; W/Z k = 32 gives 39.2 %).

### Single heads removed alone (all other heads computing)
| | pt donors | aligned donors | paired difference, aligned − pt [95 %] |
|---|---|---|---|
| W/Z: mean drop over 96 heads | −9.6 pts [−11.0, −8.1] | −1.8 pts [−2.3, −1.2] | +7.8 pts [+6.7, +8.9] |
| W/Z: heads Holm-significant (McNemar, 96 tests) | 73 | 12 | |
| W/Z: heads with a drop > 10 pts | 35 | 4 | |
| W/Z: mean drop in blocks 3–11 | −5.1 pts | −0.5 pts | |
| t/QCD: mean drop over 96 heads | −9.5 pts [−10.3, −8.6] | −2.8 pts [−3.1, −2.4] | +6.7 pts [+6.1, +7.3] |
| t/QCD: heads Holm-significant | 40 | 11 | |
| t/QCD: heads with a drop > 10 pts | 28 | 6 | |
| t/QCD: mean drop in blocks 3–11 | −2.3 pts | +0.1 pts | |

- **Heads still necessary with aligned donors.**
  - W/Z: b0h5 −23.1, b0h2 −20.0, b1h1 −13.8, b0h6 −12.4.
  - t/QCD: b1h2 −45.4, b0h5 −43.7, b0h2 −42.9, b2h3 −40.9, b2h5 −37.6, b1h1 −37.0.
  - All are in blocks 0–2. Under pt donors these heads cost the same or more, and dozens more heads in blocks 3–10 also cost points.

### Circuits with aligned donors, greedy F at k = 4 / 8 / 16 / 24 / 32
- **W/Z**: 0.094 / 0.088 / 0.174 [0.10, 0.25] / 0.254 / 0.310 [0.23, 0.39].
- **t/QCD**: 0.091 / 0.293 / 0.376 [0.32, 0.42] / 0.597 / 0.724 [0.68, 0.77].
- **ΔF, aligned − pt (paired, Holm over 10)**:
  - t/QCD k = 8: +0.16 [+0.10, +0.22], McNemar 109 lost / 180 gained, p 3.5·10⁻⁵;
  - t/QCD k = 16: +0.13 [+0.07, +0.19], McNemar p 3.7·10⁻⁴;
  - t/QCD k = 32: +0.28 [+0.22, +0.35], McNemar 64 / 195, p 1.4·10⁻¹⁶;
  - Holm p < 0.001 for these three; the other seven cases have Holm p ≥ 0.29.
- **Block-matched random sets (aligned), mean F [95 %]**:
  - t/QCD k = 32: 0.39 [0.08, 0.71]; the greedy set beats all 20 (raw p 0.048, Holm 0.24 = the minimum attainable with 20 sets and 5 tests).
  - W/Z k = 24 and 32: the greedy set beats all 20 random sets.
- **Unrestricted random sets** (any block): mean F ≤ 0.10 in every arm.

### Hypotheses
- **H0 (ablation choice irrelevant)**: aligned and pt donors give the same F and single-head effects.
  - Excluded for single-head removal: the paired difference is +7.8 and +6.7 points, both intervals far from 0.
  - Excluded for t/QCD circuits at k = 8, 16, 32 (Holm p < 0.001).
  - Not excluded for W/Z circuits (Holm p ≥ 0.29).
- **H1 of P55 (sparse circuit, F ≥ 0.8 at k ≤ 16)**: excluded in all three arms. The largest F at k ≤ 16 is 0.38 [0.32, 0.42] (aligned) and 0.64 (mean, t/QCD).
- **H2 of P55 (distributed: any early heads equivalent)**: the test is uninformative. The block-matched null overlaps 70–80 % with the circuit, and the Holm floor sits at 0.099 (P55) or 0.24 (here). Where a comparison is possible, the greedy set beats every random set at W/Z k = 4, 24, 32 (P55) and at t/QCD k = 32 (aligned).
- **H_offdir (pt-only donors push ablated heads off the data manifold)**: supported. Single-head costs in blocks 3–11 vanish when the donor is direction-matched (W/Z −5.1 → −0.5 points), and t/QCD circuits gain 0.13–0.28 in F.

## Corrected reading
These statements hold, as measured:
- With only the k greedy-ranked heads computing, no circuit of k ≤ 16 heads reaches F ≥ 0.8 for either task under any of the three ablations tested. The largest F at k ≤ 16 is 0.38 (aligned) and 0.64 (mean ablation, t/QCD).
- The ranked heads sit in blocks 0–3.

These numbers depend on the ablation:
- With direction-matched donors, t/QCD F is 0.29 / 0.38 / 0.60 / 0.72 at k = 8 / 16 / 24 / 32, and W/Z F at most 0.31 (k = 32).
- With P55's pT-only donors, the same numbers are lower: t/QCD 0.13 / 0.24 / 0.59 / 0.44, W/Z at most 0.25.

These statements from P55 need correcting:
- **Single-head costs.** The large single-head costs in blocks 3–11 under pT-only donors (72 of 96 heads Holm-significant for W/Z) largely vanish with direction-matched donors (12 of 96). The individually necessary heads are a handful in blocks 0–2: b0h2, b0h5, b1h1, b1h2, b2h3, b2h5, which cost 37–45 points for t/QCD.
- **Random-set comparison.** The block-matched random comparison has a Holm floor of 0.099 and shares 70–80 % of its heads with the circuit, so it does not decide between "specific heads" and "any early heads".
- **E4 mass response.** The E4 R_c values were computed with pT-only donors and were not rerun with aligned donors.

## Files
- Script: `interp/p55v_verify.py`.
- Results: `results/part/p55_verify.json` (smoke: `p55_verify_smoke.json`).
- Logs: `logs/c-p55-verify.log`, `logs/c-p55-verify-smoke.log`.

## Explainer additions (glossary)
- **Direction-matched (aligned) resample ablation**: a head's output is replaced by its output on another jet of similar pT and η, rotated about the beam axis onto the target jet's azimuth. For a Lorentz-equivariant network this keeps the pasted four-vectors pointing along the target jet:
  $$O_{bh}(x)\leftarrow O_{bh}\big(R_z(\phi_x-\phi_{x'})\,x'\big),\qquad x'\ \text{matched in}\ (\ln p_T,\eta).$$
- **Holm floor of a rank test**: with n random sets and m tests, the smallest Holm-adjusted rank p is m/(n+1).
