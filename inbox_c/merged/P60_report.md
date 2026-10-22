# P60 — L-GATr trained with ParT's stored-axis inputs (loop C, round C13; claims C6 v2, C3s, C10 v5; proposes C27)

Status: **running** (training job `anrunw-ai-c-p60-lgsa`, run `jc2m_sa_s0`; started 2026-10-21 23:16 UTC on k8s-haosu-15, RTX 2080 Ti, GPU 86–91 % between validations; the authors' code estimates 34.4 h of training, P5 took 43 h with validations, so ETA ≈ 2026-10-23 12:00–18:00 UTC). Interim battery on the first checkpoint (iteration 4999) done. Final battery in a later round.

## Conclusion

When L-GATr (a Lorentz-equivariant transformer that by default measures each particle's angle about the summed four-momentum of the jet) is retrained identically except that those per-particle angles are measured about ParT's stored jet axis, it starts responding to moving particles off that axis at fixed invariant mass: after 5,000 training iterations its W-vs-Z score moves by 0.38 of its response to a real mass change of the same predicted size, against 0.000 for the unmodified L-GATr at the same iteration. The whole response passes through those angle inputs; the four-vectors alone contribute nothing. So the axis response is set at least in part by the input representation, not by the architecture; the value is still about half of the ParT retrains (0.77–0.85), and whether it grows toward them with training waits for later checkpoints.

## Question and design

ParT and ParticleNet move their W/Z decision when every particle is moved off the stored jet axis at fixed invariant mass (C3, C5); our L-GATr does not (P51, C6 v2: R ≈ 0). Two differences coexist: (1) L-GATr is Lorentz-equivariant and gets the four-vectors directly; (2) its per-particle Δη, Δφ, ΔR are computed about the axis of the summed particle four-momenta (which moves with the particles), ParT's about the stored axis (which does not). P60 changes only (2).

- H0 (architecture sets it): stored-axis L-GATr R(a), R(b) inside [−0.1, 0.1], like P5.
- H1 (input representation sets it): R(a), R(b) ≥ 0.3, toward the P10 ParT retrains (0.6–0.8 in P11), growing with iteration.

## Procedure

**Weights check (2026-10-21, user rule).** No released L-GATr JetClass weights: heidelberg-hepml/lorentz-gatr README and Releases (no entries; the README only mentions reloading one's own runs), heidelberg-hepml/lgatr README (no weights; links a separate quantised-tagger repo without stated checkpoints), HuggingFace model search "gatr" (7 unrelated models), Zenodo search "L-GATr JetClass" (JetClass datasets and OmniJet-alpha weights only), web search. Same as P5 (2026-10-19).

**Model and training.** Authors' code ext/lorentz-gatr @ 35aca2a, config_paper/jctagging unchanged (12 blocks, 16 multivector / 32 scalar hidden channels, 8 heads, Lion lr 3e-4, cosine schedule, batch 128, 200,000 iterations, validation every 5,000, best-validation checkpoints), same 2M-jet subset ($V/data/jetclass2m: 2 random training files per class, seed 2026), same file ranges (train [0,2], val [120,121], test [100,101]), seed 0, the P5 two-line loader patch. One change, `interp/p60_lgatr_storedaxis.patch` with `data.features=storedaxis`:

- `miniweaver/storedaxis.yaml` = the authors' default.yaml plus two raw inputs `part_deta`, `part_dphi` (JetClass branches, no transformation), so the loader delivers 12 per-particle rows.
- `embedding.py`: when `stored_axis_features` is true, the last two scalar columns are taken out (so the network still sees 10 PID/track scalars) and replace the authors' recomputed angles. With $\eta^{\rm st},\phi^{\rm st}$ the stored jet axis and $s=\mathrm{sign}(\eta^{\rm st})$:
  - authors (P5): $\Delta\phi_i=\mathrm{wrap}(\phi_i-\phi_J)$, $\Delta\eta_i=-(\eta_i-\eta_J)$, $\Delta R_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$, with $(\eta_J,\phi_J)$ the direction of $\sum_i p_i$ (four-momentum sum of the particles);
  - P60: $\Delta\phi_i=\phi_i-\phi^{\rm st}$, $\Delta\eta_i=s\,(\eta_i-\eta^{\rm st})$ (the stored `part_dphi`, `part_deta`, exactly ParT's inputs), $\Delta R_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$;
  - unchanged: $\log p_{T,i}$, $\log E_i$, $\log(p_{T,i}/p_{T,J})$, $\log(E_i/E_J)$ (about the four-momentum sum, as the authors), the four-vector multivector inputs $p_i/20\,$GeV, the beam (x-y plane) and time $(1,0,0,0)$ reference multivectors, and the preprocessing $(x-\mu)\cdot f$ with $(\mu,f)=(0,1)$ for Δφ, Δη and $(0.2,4)$ for ΔR.
  - `jetclassexperiment.py`: new option `storedaxis` (in_s_channels 10) and the flag; for every other option the flag is false and the code path is the authors'.
- The job runs from a private clone ext/lorentz-gatr-p60 (35aca2a + P5 patch + P60 patch); the shared ext/lorentz-gatr is not modified.

**Unit test before launch** (`python -m interp.p60_axis_battery --stage unittest`, run inside the training job on the GPU before `run.py`; also as CPU jobs c-p60-ut, c-p60-ut2; `results/lgatr/p60_unittest.json`), 200 jets (rows 3000–3019 of each class's test file 100), 7,848 particles:
- T1: patched Δφ, Δη columns equal ParT's `part_dphi`, `part_deta` inputs (jetdistill `inputs(J, 'full')`) to 0.0 (bit-identical); ΔR equals $\sqrt{\Delta\eta^2+\Delta\phi^2}$ of the stored values to 1.2e-7. 6 of 7,848 particles differ from ParT's ΔR input because ParT clips the scaled value at 5 (ΔR > 1.45 rad; stored ΔR 1.77–2.27); L-GATr's preprocessing does not clip (neither does the authors' own-axis ΔR). The 14 other scalar columns equal those of the default path exactly.
- T2: with `features=default` the patched embedding equals the pristine 35aca2a embedding bit for bit (scalars, multivectors, token layout): P5's inputs are reproduced.
- T3: the miniweaver loader with storedaxis.yaml (val WToQQ_122, 200 jets) yields 12 feature rows; per jet, $\phi(p_x,p_y)-\Delta\phi$ is constant to 8e-8 and $\eta(p)\mp\Delta\eta$ to 1.7e-7 (sign + in 102 jets, − in 98, as $s$): the stored branches belong to the same particles in the same order.
- T4: random-init forward finite on the GPU for both configs. (On CPU without xformers the random-init forward is non-finite for the authors' default config too, c-p60-ut2; the GPU path is the one used.)

**Smoke check of the training** (mlflow, same database as P5): train loss stored-axis vs P5 at iteration 1 / 1000 / 2000 / 3000 / 4000 / 5000–6000: 3.499 / 1.052 / 0.964 / 0.479 / 0.535 / 0.641 vs 3.497 / 0.968 / 0.887 / 0.476 / 0.555 / 0.600. First validation (iteration 4999, 1M validation jets): accuracy 0.7894 vs P5 0.7917, loss 0.590 vs 0.585. GPU 86–91 % during training steps, 0 % during the ~15 min validations (P5 the same).

**Battery** (`interp/p60_axis_battery.py`, jobs c-p60-part (ParT models, GPU), c-p60-run2 (L-GATr, GPU L4; c-p60-run failed at the first batch on an A10 node with "CUDA error: an illegal memory access", relaunched through `interp/p60_gpu_job.sh` = part_job.sh without A10 and with batch 128), c-p60-ana (analysis, CPU)).

- Samples: (1) P1 sample, rows 3000–3999 of each class's test file 100 (N 10,000): 10-class accuracy and confusion; its W and Z jets (N 2,000) also get the edits. (2) fresh val_5M WToQQ_122 / ZToQQ_122, 2,000 random rows per class, numpy default_rng(6060), sorted, ≥ 2 particles (N 4,000). The planned "_128" does not exist (val_5M holds files 120–124; 121 is L-GATr's validation file, 123/124 were used in P46/P51/P51v).
- Edits (particle pT, masses, PID, track inputs fixed): (a) PHI ± 0.06: every particle's Δφ + 0.06 rad with the stored axis kept; the lab four-vectors rotate about the beam (invariant mass unchanged: mean Δlog m 5e-8); for P60 L-GATr only the stored-axis Δφ/ΔR inputs change. (b) E6 ± 0.06: only the stored jet_phi moved by 0.06 rad, particles fixed (four-vectors identical, max |Δp| 0); every Δφ input shifts by ∓0.06. (c) BOOST ± 0.06: every particle's rapidity + 0.06 at fixed pT and mass (exact longitudinal boost, P51; Δlog m 4e-7). (d) E4|edit: matched dilation, every particle's (Δη, Δφ) × k with $k^2=M_2'/M_2$, $M_2=\sum_i p_{T,i}r_i^2$ about the stored axis: a real mass change (mean Δlog m +0.085) of the size the edit predicts if the network read $M_2$. All edits raise $M_2$ by Δlog M2 ≈ +0.172.
- Read-out $d=\ell_W-\ell_Z$ (W logit minus Z logit). $R=\sum_{\rm jets}\Delta d_{\rm edit}/\sum_{\rm jets}\Delta d_{\rm E4|edit}$: 0 = no response, 1 = the same response as to a real mass change of the same predicted size. Pooled over the two signs. W → Z = share of true-W jets called Z. Routes (L-GATr): A full edit; D = edited four-vectors with the scalar inputs held at unedited values; F = only the scalar inputs (including the stored-axis Δη/Δφ/ΔR) edited.
- Models: P60 L-GATr it4999 (LGsa); P5 L-GATr it4999 and final (own-axis inputs); untrained stored-axis L-GATr (torch seed 6060); released ParT_full, ParT_kin; P10 retrains ParT_full_s0, ParT_kin_s0 (same 2M jets).
- Statistics: 2,000 bootstrap resamples over jets (paired across models), sign-flip permutation (5,000) of the per-jet Δd, exact McNemar on per-jet correctness, Holm within sample (24 sign-flip tests; 21 paired differences; the paired-difference Holm floor at 2,000 resamples is 0.0105).

## Numbers

Unedited accuracy, P1 sample, N 10,000 (10 classes): ParT_full 85.93 %, ParT_kin 74.47 %, ParT_full_s0 83.71 %, ParT_kin_s0 72.57 % (all as P11); P5 L-GATr it4999 79.24 % (P5: 79.23 %), P5 final 84.10 %; **P60 L-GATr it4999 78.92 %** (Δ vs P5 at the same iteration −0.32 points; per-jet calls on the P1 sample were not stored, so only an unpaired 95 % bound ±1.1 points is available; 1M-jet validation 78.94 vs 79.17 %); untrained 10.0 % (one class for every jet). Confusion matrices in results/lgatr/p60_axis_battery_it4999.json. Rerun floor (same input twice) max |Δd| 0.003–0.19 (xformers batch nondeterminism).

R, 95 % bootstrap interval (val sample N 4,000; P1 W/Z sample N 2,000 in brackets):

| model | (a) PHI ± 0.06 | (b) E6 ± 0.06 | (c) BOOST ± 0.06 |
|---|---|---|---|
| P60 L-GATr stored-axis, it4999 | **0.378 [0.370, 0.385]** (0.370 [0.359, 0.380]) | **0.378 [0.370, 0.385]** (0.370) | **0.376 [0.368, 0.384]** (0.370) |
| P5 L-GATr own-axis, it4999 | 0.000 [−0.000, 0.000] (−0.000) | −0.000 (−0.000) | −0.001 [−0.001, −0.001] (−0.001) |
| P5 L-GATr own-axis, final | 0.000 [−0.001, 0.001] | 0.000 | −0.001 [−0.002, 0.001] |
| untrained stored-axis L-GATr | 0.88 (Δd −0.007 vs E4 −0.008) | 0.88 | 0.98 |
| ParT_full_s0 (2M retrain) | 0.849 [0.833, 0.866] (0.851) | 0.850 (0.851) | 0.856 (0.847) |
| ParT_kin_s0 (2M retrain) | 0.765 [0.747, 0.783] (0.771) | 0.771 (0.770) | 0.730 (0.730) |
| ParT_full (released) | 0.795 [0.781, 0.810] (0.805) | 0.795 | 0.806 |
| ParT_kin (released) | 0.741 (0.743) | 0.747 | 0.734 |

P60 L-GATr it4999, val sample: mean Δd −0.280 (edit) vs −0.740 (E4) per jet-edit; PHI: calls changed 12.7 % (E4 29.5 %), W → Z 14.3 → 23.3 % (E4 39.7 %), McNemar lost/gained correct 601/301 (p 9e-24); BOOST: 13.1 %, W → Z 14.3 → 23.8 %, 632/296 (p 1e-28). Sign-flip p 2e-4 for every family (Holm 0.005). P1 W/Z sample: PHI W → Z 16.5 → 25.9 %, 285/131 (p 3e-14). Per sign: PHI + / − 0.394 / 0.362 (val), 0.386 / 0.355 (P1).
Routes (P60 L-GATr, val): PHI and E6 route D (four-vectors only) 0.000 [−0.000, 0.000], route F (scalar inputs only) 0.378 [0.370, 0.385]; BOOST D −0.004 [−0.005, −0.004], F 0.381 [0.373, 0.389].
P5 L-GATr it4999 (val): PHI calls changed 0.04 %, W → Z 14.9 → 14.9 %, McNemar 2/1; BOOST 0.75 %, sign-flip Holm p 0.005 (mean Δd +0.0008, 0.1 % of E4).
Paired differences of R (val): P60 − P5 it4999: PHI +0.378 [0.370, 0.385], E6 +0.378, BOOST +0.377 (Holm p 0.0105, the floor); P60 − ParT_full_s0: −0.471 [−0.487, −0.456] (PHI), −0.480 (BOOST); P60 − ParT_kin_s0: −0.387 [−0.404, −0.371] (PHI), −0.354 (BOOST); P60 − untrained: −0.50 / −0.61.
Untrained stored-axis L-GATr: calls changed 0.0 % under every edit and under E4; its R ≈ 0.9 is a ratio of responses about 1 % the size of the trained models' (|Δd| ≤ 0.008), so it is not a training reference (as P11v found for untrained ParT).

Sensitivity to a real mass change (E4, mean Δd per jet-edit, val): P60 −0.740, P5 it4999 −0.706: the two L-GATr models respond equally to a real widening of the jet at this iteration.

## Reading

- The pre-registered H0 band [−0.1, 0.1] is excluded for (a) and (b) at iteration 4999 on both samples (lower bound 0.359); the H1 threshold 0.3 is passed (lower bound 0.359). The P5 model at the same iteration and at the end of training stays at |R| ≤ 0.001.
- For this model (a) and (b) give the same input change (the four-vector part of (a) is an exact rotation symmetry; route D = 0.000), so they are one measurement; (c) BOOST changes the four-vectors and log E as well, and still gives the same R (0.376) with the four-vector route at −0.004: the response follows the stored-axis Δη/Δφ/ΔR inputs.
- R = 0.38 is about half of the ParT retrains on the same 2M jets (0.77–0.85) and of the released ParT (0.74–0.81). Whether it grows toward them with training (the second half of H1) is not yet measured: only one checkpoint exists; P10 found the ParT_full retrain already at R 0.99 after its first epoch (1M jets ≈ 8,000 iterations of 128), so L-GATr at 640k training jets reaching 0.38 is not directly comparable in training time.
- Accuracy at iteration 4999: −0.3 points vs P5 (validation −0.2 points); the stored-axis features neither help nor hurt measurably yet.
- Not separated here: whether the network reads ΔR about the stored axis as a proxy of the jet mass (axis moment $M_2$) or uses the offset of the stored axis from the four-momentum axis in another way; the matched-dilation calibration assumes the former.

## Files

- interp/p60_lgatr_storedaxis.patch, interp/p60_lgatr_storedaxis_job.sh (training; node exclusions updated after the first launch sat 20 min pulling the image on k8s-haosu-09), interp/p60_axis_battery.py (unittest / lgatr / part / analyze), interp/p60_gpu_job.sh (part_job.sh copy without A10 GPUs).
- results/lgatr/p60_unittest.json, p60_part.json/.npz, p60_lg_it4999.json/.npz, p60_axis_battery_it4999.json, p60_axis_battery_it4999.png.
- Training: $V/models/lgatr/runs/jc2m/jc2m_sa_s0/ (config.yaml, models/model_run0_it*.pt), log $V/logs/c-p60-lgsa.log, mlflow run cc6d89f7354042558564561e8b63b5c5.
- Next round: `./interp/p60_gpu_job.sh c-p60-run3 'LGATR_SRC=/j-jepa-vol/aaron-interp/ext/lorentz-gatr-p60 /j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p60_axis_battery --stage lgatr --its it4999,it24999,it99999,it199999 && python -m interp.p60_axis_battery --stage analyze --its it4999,it24999,it99999,it199999'` (best-validation checkpoints only exist at iterations where validation loss improved; final `model_run0.pt` to be added as a tag). Store P1 per-jet calls for a paired accuracy McNemar.

## Explainer entry (for site/explain.py; P60_EXPLAINER)

- **Question.** Is ParT's response to particles moved off the stored jet axis at fixed invariant mass (C3) caused by its architecture or by the inputs it is given? Test: give the Lorentz-equivariant L-GATr ParT's stored-axis angle inputs and nothing else new.
- **Background.** L-GATr receives each particle's four-momentum $p_i=(E,p_x,p_y,p_z)$, a beam and a time reference, and 7 scalar features. By default (P5) three of them are angles about the jet's own four-momentum axis $(\eta_J,\phi_J)$ = direction of $\sum_i p_i$: $\Delta\phi_i=\phi_i-\phi_J$, $\Delta\eta_i=-(\eta_i-\eta_J)$, $\Delta R_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$. That axis moves with the particles, so shifting all particles leaves them unchanged, and P5/P51 found $R\approx0$. ParT's angles are about the stored axis $(\eta^{\rm st},\phi^{\rm st})$, which does not move.
- **Method.** Retrain L-GATr with the authors' settings on the same 2M jets, with $\Delta\phi_i=\phi_i-\phi^{\rm st}$, $\Delta\eta_i=\mathrm{sign}(\eta^{\rm st})(\eta_i-\eta^{\rm st})$ (the stored JetClass values ParT reads), $\Delta R_i=\sqrt{\Delta\eta_i^2+\Delta\phi_i^2}$; all else unchanged (unit-tested bit-identical to ParT's inputs). Edits at fixed invariant mass: (a) all particles $\phi_i\to\phi_i\pm0.06$, (b) stored axis $\phi^{\rm st}\to\phi^{\rm st}\pm0.06$, (c) boost along the beam, rapidity ±0.06. Read-out $d=\ell_W-\ell_Z$; $R=\sum\Delta d_{\rm edit}/\sum\Delta d_{\rm E4}$, E4 = all angles scaled by $k=\sqrt{M_2'/M_2}$, $M_2=\sum_i p_{T,i}r_i^2$ (a real mass change of the same predicted size).
- **Result (figure results/lgatr/p60_axis_battery_it4999.png).** After 5,000 iterations: $R=0.378\,[0.370,0.385]$ for (a) and (b), 0.376 for (c), on 4,000 fresh W/Z jets; true-W jets called Z rise 14 → 23 %. Own-axis L-GATr at the same iteration: $R=0.000$. ParT retrains on the same jets: 0.77–0.85.
- **Checks.** Same R on a second sample (P1 W/Z, 0.370); the response passes entirely through the angle inputs (four-vectors alone: 0.000; boost four-vector route −0.004); accuracy 78.9 vs 79.2 % (no cost); untrained network barely responds (|Δd| 1 % of trained, no calls changed).
- **Why it matters.** The axis-relative mass response is not specific to non-equivariant architectures: a Lorentz-equivariant network acquires it when its angle features are measured about the stored axis. The response is about half of ParT's after 5,000 iterations; later checkpoints show whether it grows.
- Glossary additions: "stored-axis features (P60)": per-particle Δη, Δφ, ΔR measured about the jet axis stored in the JetClass file (jet_eta, jet_phi), as ParT reads them, instead of about the direction of the summed particle four-momenta. "E6 (P60)": only the stored jet_phi moved by ±0.06 rad, particles fixed; the four-vectors are unchanged and every Δφ input shifts by ∓0.06.
