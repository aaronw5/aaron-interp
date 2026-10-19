# P4 verify: ParticleNet axis-relative mass response (verifier report, 2026-10-19)

## Conclusion

Moving every particle 0.06 rad away from the jet axis leaves the jet mass unchanged. On 10,000 jets not used by P4, this edit raises the share of true W jets that ParticleNet_full calls Z from 12 % to 26 %. ParT_full on the same jets goes from 12 % to 33 %, so ParticleNet shows about two-thirds of ParT's increase (about half for the kinematics-only models). As in P4, the change reaches the network through each particle's input distance to the axis (ΔR). Particle-by-particle terms of the first layer pass it on; the neighbour graphs do not. Edits that keep every particle's ΔR and the jet mass unchanged (rotations about the axis, a mirror in φ) do not reproduce the effect: the kinematics-only models do not respond, and the full models respond weakly, mostly because the mirror breaks the sign pattern of charge and impact parameter. That ParT and ParticleNet share this behaviour is measured. The claim that the JetClass input format causes it has not been tested: no model without axis-relative inputs has been run yet.

## What was checked

Code (read in full): `interp/p4_particlenet.py`, `interp/pn_net.py` (weaver-core 0.4.17 ParticleNet plus the official get_model settings), `interp/pn_data/JetClass_{kin,full}.yaml`, `interp/part_core.py`, `jetdistill/part/network.py` (inputs, raw_features).

- Network: `pn_batch` repeats weaver's `ParticleNet.forward` step by step: mask, bn_fts, coord shift 1e9, `knn` (top-k+1 minus self), `get_graph_feature_v1` = [x_i, x_j − x_i], conv/bn/act, mean over k, shortcut 1×1 conv + bn, mean pool with counts, fc[0] (Linear+ReLU+Dropout) → fc[1]. The weights load with strict=True, and the split forward matches the native forward exactly (max |ΔL| = 0 on all three samples). The read-out layer `h` is the post-ReLU 256-d input of the last Linear, as stated.
- Padding: `inputs` wrap-fills padded slots up to the sample's largest multiplicity, and `pn_inputs` zero-pads to 128. Both are multiplied by the mask before BN and kNN, the same as weaver with `pad_mode: wrap`. kNN overlaps are averaged over real particles only.
- Sampling: each block of rows (3,000–4,000, 6,000–7,000) comes from one class's own test file, so the sample is not a first-k slice of class-ordered rows. The bootstrap is class-stratified.
- Build: `pc.build` uses the sign-corrected η (η = jet_eta + sign(jet_eta)·Δη). Raw and rebuilt accuracies agree (PN_full 84.57 / 84.57 %).
- Edits: φ + 0.06 changes only the Δφ and ΔR features. pT, E, jet_pt (|Σ p⃗_T|) and jet_energy are invariant under a common φ rotation, and the masks are equal. The true Δlog m is at most 0.0046.
- Route variants: for the φ shift, `dR_only` ≡ `dedp_removed` and `dedp_only` ≡ `dR_removed` give identical numbers. This is expected, because only ΔR and Δφ change, and it confirms that the feature swap is indexed correctly.
- The layer-1 kNN graph is built from Euclidean distances in (Δη, Δφ). A common shift or a uniform scaling cannot change the neighbour order, so "layer-1 graph unchanged, share 1.000" and "points-only share 0" hold by construction. They are not measurements. The measured graph result is the one for layers 2–3, which are built in feature space: their overlap is 0.94 / 0.91, and fixing all three graphs keeps 0.96–1.01 of the response.
- Noise: changing the batch size to 97 moves logits by up to 0.78 (PN_full, rows 6,000) because of near-ties in the feature-space kNN. It changes ≤ 0.04 % of calls, which is small compared with every effect reported.

No indexing, hook, padding, sampling or build bug was found.

## Re-run on fresh rows (job anrunw-ai-c-p4-verify, A10, 215 s)

Procedure: `interp/p4v_verify.py` uses P4's own functions. Sample: rows 8,000–9,000 of each class's first test file (N = 10,000, 1,000 true W). Bootstrap: 2,000 class-stratified resamples (seed 2). Calibration: angles × k with k = 1.0770 (target and achieved mean Δlog m = 0.0732).

| model | acc. unedited | W→Z unedited → φ+0.06 [95 % CI] | ΔW→Z [CI] | McNemar n01/n10, p | calls changed |
|---|---|---|---|---|---|
| ParticleNet_full | 83.81 % [83.10, 84.46] | 12.2 → 26.3 % [23.5, 28.9] | +14.1 [11.7, 16.6] | 166/25, 1e-26 | 8.8 % |
| ParticleNet_kin | 69.01 % [68.12, 69.79] | 18.2 → 26.8 % [24.0, 29.5] | +8.6 [6.1, 11.1] | 128/42, 3e-11 | 12.4 % |
| ParT_full | 85.41 % | 11.6 → 32.7 % [29.9, 35.7] | +21.1 [18.6, 24.0] | 224/13, 8e-51 | 8.5 % |
| ParT_kin | 73.08 % | 16.4 → 35.7 % [32.7, 38.8] | +19.3 [16.3, 22.4] | 233/40, 3e-34 | 19.5 % |

Read-out of log m (ridge on h; test-half R² 0.867 full / 0.903 kin) under φ+0.06, with ½Δlog(M0·M2) = 0.0732 predicted:

| model | Δlog m̂ [CI] | gain c | raw ratio | calibrated ratio [CI] | p(H0inv: 0) | p(H0M2: ratio 1) |
|---|---|---|---|---|---|---|
| PN_full | +0.0372 [0.0352, 0.0392] | 0.746 | 0.509 | 0.682 [0.641, 0.721] | < 1e-3 (normal 5e-295) | < 1e-3 (normal 5e-42) |
| PN_kin | +0.0371 [0.0352, 0.0391] | 0.742 | 0.507 | 0.684 [0.645, 0.723] | < 1e-3 | < 1e-3 (normal 4e-42) |

P4 reported 0.637–0.707 on its two samples. Across all of P4's ratio tests the range is 0.623–0.707 (rows 6,000, rand0.06 = 0.623), so the reading's "0.64–0.71" should read 0.62–0.71.

Route shares of Δ(logit_W − logit_Z) on true W under φ+0.06 (fresh rows; P4's values in brackets):

| variant | PN_full | PN_kin |
|---|---|---|
| all 3 kNN graphs fixed | 0.968 [0.953, 0.983] (P4 0.963–0.975) | 0.997 [0.981, 1.013] (0.991–0.992) |
| only ΔR edited | 1.067 [1.035, 1.097] (1.08–1.10) | 1.323 [1.271, 1.380] (1.37–1.40) |
| ΔR fed unedited | 0.007 [−0.007, 0.021] | −0.023 [−0.044, −0.004] |
| only (Δη, Δφ) edited | 0.007 | −0.023 |
| only shortcut W_sc x_i edited | 1.004 [0.969, 1.041] | 1.004 [0.961, 1.048] |
| shortcut fed unedited | 0.188 [0.167, 0.211] | 0.327 [0.298, 0.358] |
| only centre x_i edited | 0.183 [0.161, 0.204] | 0.291 [0.263, 0.320] |
| only edge x_j − x_i edited | 0.037 [0.026, 0.048] | 0.091 [0.076, 0.108] |

Every P4 route number reproduces within its interval or within 0.05.

## Added controls: edits that keep ΔR and the jet mass

In P4 the random-direction shift was a second edit of the same kind, not a null. The only null was a re-run of the same inputs. The control added here tests whether the network responds to any change of particle positions of this size (H2), or only to changes of the distance to the axis (H1: ΔR / axis moment).

Each edit keeps every particle's ΔR, so ½Δlog(M0·M2) = 0 (measured |mean| < 3e-8):
- refl: Δφ → −Δφ. The invariant mass is exactly unchanged (max |Δlog m| 4e-4). The pT-weighted displacement is 0.20 (0.15 on W).
- rot0.3: rotation of (Δη, Δφ) about the axis by 0.3 rad. Displacement 0.046 (0.034 on W). Mean Δlog m −2e-4.
- rot90: rotation by 90°. Displacement 0.22 (0.16 on W). Mean Δlog m −0.0014, but up to 0.38 on single jets, because a 90° rotation in (η, φ) is not a symmetry for wide jets.

Predictions. H1: ΔW→Z ≈ 0 for all three edits. H2: ΔW→Z of the order of the shift's effect scaled by displacement, i.e. about +8 points for PN_full under rot0.3 and more than +14 points under refl and rot90.

| model | ΔW→Z refl [CI] | ΔW→Z rot0.3 | ΔW→Z rot90 | calls changed refl / rot0.3 / rot90 | Holm p (McNemar W→Z) refl / rot0.3 / rot90 |
|---|---|---|---|---|---|
| PN_full | +4.7 [2.8, 6.6] | +0.6 [−0.2, 1.4] | +3.8 [2.0, 5.6] | 11.9 / 2.5 / 11.9 % | 5e-6 / 1 / 3e-4 |
| PN_kin | +0.1 [−0.5, 0.7] | −0.3 [−1.2, 0.6] | +0.7 [−0.6, 2.0] | 1.5 / 2.9 / 7.4 % | 1 / 1 / 1 |
| ParT_full | +3.3 [1.8, 4.8] | −0.5 [−1.2, 0.1] | +3.2 [1.7, 4.8] | 7.3 / 1.3 / 6.7 % | 1e-4 / 1 / 4e-4 |
| ParT_kin | 0.0 [−0.7, 0.7] | +0.8 [−0.3, 1.9] | +2.3 [0.7, 3.9] | 1.7 / 4.2 / 9.0 % | 1 / 1 / 0.047 |

Holm correction is over 16 McNemar tests: 4 models × (shift + 3 controls). Δlog m̂ for PN_kin is −0.0002 (refl) and +0.0017 (rot0.3), compared with +0.0371 for the shift. For PN_full it is +0.0266 (refl) and +0.0050 (rot0.3).

For PN_full and ParT_full the φ mirror was followed up, because their responses there (accuracy 83.8 → 77.9 % for PN_full) sit far above those of the kin models. Job anrunw-ai-c-p4-verify-q, same rows.

| edit (full models) | PN_full ΔW→Z [CI] | PN_full calls changed | ParT_full ΔW→Z [CI] | ParT_full calls changed |
|---|---|---|---|---|
| Δφ → −Δφ | +4.7 [2.8, 6.6] | 11.9 % | +3.3 [1.8, 4.8] | 7.3 % |
| + charge → −charge | +4.0 [2.2, 5.9] | 12.2 % | +3.8 [1.8, 5.8] | 10.7 % |
| + charge and d0 → − | +1.7 [0.4, 3.1] (p 0.019 unadj.) | 3.1 % | +1.9 [0.4, 3.5] (p 0.023) | 2.9 % |
| charge → −charge alone | +4.5 [2.7, 6.4] | 6.2 % | +5.4 [3.4, 7.5] | 10.0 % |
| φ + 0.06 | +14.1 [11.7, 16.6] | 8.8 % | +21.0 [18.5, 23.9] | 8.5 % |

What the controls show:
- H2 is excluded. rot0.3 moves no model's W→Z (all CIs include 0, Holm p = 1), even though H2 predicts about +8 points for PN_full. For the kin models, the mirror and rot90 also leave W→Z unchanged.
- In the full models, the response to the mirror mostly comes from the joint sign pattern of Δφ, charge and d0: flipping all three restores accuracy (PN_full 83.5 % vs 83.8 %) and cuts calls changed from 11.9 % to 3.1 %. One possible cause is track bending in the magnetic field together with the impact-parameter sign convention. That cause was not isolated.
- A residual of +1.7 to +1.9 points W→Z remains in the full models after the sign-consistent mirror. That is 8–12 % of the shift's effect, and it is not significant after Holm correction.
- H1 is supported for the W/Z response, with that small residual for the full models.

## Assessment of the P4 reading and conclusion

Every number in the reading reproduces on fresh rows. Corrections:
1. "almost as much as ParT does" holds for the full models on rows 3,000 (+16 vs +19 points), but not on fresh rows (+14 vs +21) or for kin (+11 vs +21; fresh +9 vs +19). Better: "about two-thirds of ParT's increase (full), about half (kin)".
2. The calibrated ratio range is 0.62–0.71, not 0.64–0.71. The fresh rows give 0.68 for both models.
3. "Under any rigid shift or uniform scaling, the layer-1 kNN graph is identical": this holds by construction (kNN on Euclidean (Δη, Δφ)). It should be stated as a property of the architecture, not as a finding. The finding about graphs is the one for layers 2–3: fixing all graphs keeps 0.96–1.01.
4. The conclusion sentence "the axis-relative mass response comes from the JetClass per-particle input format, not from ParT's attention" goes beyond what was measured. P4 shows that two architectures trained on the same inputs both route the response through the per-particle ΔR input. That is consistent with the input-format hypothesis, but a shared learned solution would predict the same. The test is a model without axis-relative inputs (L-GATr, P5) or a retrained model without ΔR. That ParT's response does not run through attention was already shown in P2/P2v (per-particle route 93–101 %). P4 does not add evidence for that part.
5. The full models are not mirror-symmetric in φ, because of how charge and d0 are signed. Future mass-invariant controls on the full models should flip charge and d0 together with Δφ.
6. Figure p4_wz_shift.png: the numbers match the JSON. The title "turns W calls into Z" overstates the effect, since 28 % of W jets are called Z after the edit. Suggested title: "A 0.06 rad shift off the axis raises W→Z calls in ParticleNet and ParT". In p4_routes.png the legend covers the ends of the bottom bars in the ParticleNet_full panel. The values can be read in the JSON.

Corrected reading: on the same jets, a rigid 0.06 rad shift that leaves the invariant mass unchanged raises the W→Z rate by 14–16 points in ParticleNet_full (ParT_full 19–21) and by 9–11 points in ParticleNet_kin (ParT_kin 19–21), over three samples of 10,000 jets. Once calibrated against a matched real mass change, the read-out of log m moves by 0.62–0.71 of the √(M0·M2) axis-moment prediction. Both 0 and 1 are excluded. Fixing the feature-space kNN graphs of layers 2–3 (layer 1 is unchanged by construction) keeps 0.96–1.01 of the response. Fed alone, the ΔR feature gives 0.94–1.41 of it; fed unedited, it leaves |share| ≤ 0.06. Edited alone, the layer-1 shortcut term gives 0.85–1.05 and the centre term 0.15–0.30, and the edge term gives ≤ 0.10. Rotations about the axis that keep ΔR do not move W→Z in any model (rot0.3). The φ mirror moves W→Z by 0 points in the kin models, and by ≤ 1.9 points in the full models when charge and d0 are flipped with it.

## Explainer entry (for site/explain.py; ledger agent)

- Question: does ParticleNet, a graph network with no attention, change its W/Z decision when the particles are moved relative to the jet axis while the jet mass stays fixed?
- Background: JetClass gives each particle its offset from the jet axis, $(\Delta\eta, \Delta\phi)$, and its distance $\Delta R=\sqrt{\Delta\eta^2+\Delta\phi^2}$. A common shift $\Delta\phi_i\to\Delta\phi_i+\delta$ leaves the invariant mass $m^2=(\sum_i p_i)^2$ unchanged, but it changes the axis moment $M_2=\sum_i p_{T,i}\Delta R_i^2$ by $\delta^2 M_0 - 2\delta M_{1\phi}$.
- Method: apply $\delta=0.06$ to 10,000 test jets, run ParticleNet and ParT, and count true W jets called Z. Then feed the edited value of one input group at a time. Controls: rotations $(\Delta\eta,\Delta\phi)\to R_\alpha(\Delta\eta,\Delta\phi)$ and the mirror $\Delta\phi\to-\Delta\phi$, which keep every $\Delta R_i$ and $m$.
- Result: W→Z rises from 12 % to 26 % in ParticleNet and from 12 % to 33 % in ParT. Feeding only the edited $\Delta R$ reproduces the change. Rotations that keep $\Delta R$ do not (figure: p4_wz_shift.png; routes: p4_routes.png).
- Checks: fresh rows; McNemar tests; class-stratified bootstrap; the split forward equals the library forward exactly. The mirror control needs charge and $d_0$ flipped too in the full models.
- Why it matters: the non-invariant, axis-relative mass response is not specific to attention. Whether it comes from the input format is tested next with L-GATr, which has no axis-relative inputs.

## Files

- Script: /Users/anrunw/Documents/aaron-interp/interp/p4v_verify.py (`main`: fresh-row re-run + controls; `charge_check`: mirror follow-up)
- Results: /Users/anrunw/Documents/aaron-interp/results/part/p4_verify.json, /Users/anrunw/Documents/aaron-interp/results/part/p4_verify_charge.json (cluster: /j-jepa-vol/aaron-interp/results/part/)
- Logs: /Users/anrunw/Documents/aaron-interp/logs/c-p4-verify.log, /Users/anrunw/Documents/aaron-interp/logs/c-p4-verify-q.log
- Jobs: anrunw-ai-c-p4-verify. The first attempt failed at admission because the node's GPU was lost (k8s-haosu-06, "GPU is lost"). It was deleted and relaunched unchanged and completed in 215 s. anrunw-ai-c-p4-verify-q completed in 171 s.
- Nothing committed.
