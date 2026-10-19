# P5 — L-GATr: released-weights check, training launch, smoke check, symmetry predictions (claims C3, A8), 2026-10-19

Status: **running** (training job `anrunw-ai-c-p5-lgatr2`, ETA about 2026-10-21 05:00 UTC). Scripts `interp/p5_lgatr_setup.sh`
(code, env, data subset), `interp/p5_lgatr_job.sh` (training job), `interp/p5_lgatr.patch` (2-line loader fix), `interp/p5_lgatr_launch.py`
(loader + symmetry battery on the P1 rows, for C2/C3). Results `results/lgatr/p5_randinit.json`, `results/lgatr/p5_it4999.json`;
logs on the volume `/j-jepa-vol/aaron-interp/logs/c-p5-{setup..setup5,lgatr,lgatr2,run,ck5k}.log`. No figure (interim run only).

## Conclusion
No trained L-GATr (a transformer built to respect the symmetries of special relativity) is published for JetClass, so we are training one with
the authors' code and settings on 2 million JetClass jets; the run is healthy and should finish in about 34 hours. An early checkpoint already
ignores a rigid rotation of all particles about the beam (1 changed call in 10,000), the edit that moves ParT's W/Z calls strongly. This holds
by construction, because L-GATr measures particle positions from an axis it recomputes itself, so it says nothing yet about how L-GATr estimates
mass; the test that can, edits that keep the invariant mass but change the spread of particles around the axis, is pre-registered and runs on
the final checkpoint in the next rounds.

## Procedure
1. **Released weights** (checked 2026-10-19): heidelberg-hepml/lorentz-gatr README and releases, heidelberg-hepml/lgatr, tagging-guide,
   lloca-experiments, tagger-quantization, arXiv 2405.14806 / 2411.00446 (SciPost Phys. 19, 108), Zenodo / HuggingFace, web search. The README
   mentions only datasets and reloading one's own runs; no JetClass or top-tagging checkpoint is published. → train.
2. **Code**: `git clone heidelberg-hepml/lorentz-gatr` → `/j-jepa-vol/aaron-interp/ext/lorentz-gatr` @ `35aca2ac4921b8092b5a14f40deed11e7c015e53`.
   Env `/j-jepa-vol/aaron-interp/env-lgatr` (python 3.10, torch 2.4.0+cu121, xformers 0.0.27.post2, clifford 1.4.0, numba 0.58.1, hydra 1.3.2,
   mlflow 2.4.1; `pip freeze` in `env-lgatr/freeze.txt`). Patch (`interp/p5_lgatr.patch`): the JetClass DataLoaders drop
   `sampler=DistributedSampler` (torch rejects a sampler on an IterableDataset; first launch `c-p5-lgatr` died with that error) and
   `cycle()` calls `set_epoch` only if present. No model, loss or optimiser change.
3. **Config**: authors' `config_paper/jctagging.yaml` unchanged (12 blocks, 16 multivector / 32 scalar hidden channels, 8 heads, 1,082,068
   parameters; Lion lr 3e-4, cosine; batch 128; 200,000 iterations = 12.8 epochs of 2M; validation every 5,000; beam reference `xyplane`,
   two beams, beam token, time reference, `add_scalar_features` on). Overrides (data / bookkeeping only): `data.data_dir`, file ranges
   train [0,2], val [120,121], test [100,101], run name `jc2m_s0a`, seed 0, base_dir / mlflow paths under `/j-jepa-vol/aaron-interp/models/lgatr`.
4. **Data subset** (`/j-jepa-vol/aaron-interp/data/jetclass2m/selection.json`, symlinks into read-only `/j-jepa-vol/JetClass/Pythia`): per class
   2 of the 100 training files drawn at random (numpy seed 2026; e.g. HToBB 36, 46; WToQQ 44, 64; ZToQQ 15, 27) = 2,000,000 jets; 1 random
   validation file per class (120–124) = 1M jets; test file 100 (holds the P1 rows). Whole files, so no first-k row selection.
5. **Training job** `anrunw-ai-c-p5-lgatr2` (`interp/p5_lgatr_job.sh c-p5-lgatr2 jc2m_s0a`; 4 CPU, 16 Gi, 1 GPU, deadline 72 h), started
   2026-10-19 15:38 UTC on an RTX 2080 Ti (11 GB). Checkpoints `/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models/`.
6. **Eval stub** `interp/p5_lgatr_launch.py`: builds the authors' model from the run's `config.yaml`, loads `['model']`, embeds jets with the
   authors' `dense_to_sparse_jet` + `embed_tagging_data_into_ga` (lab four-momenta /20 GeV + 10 PID/track scalars, scalar features added by
   the embedding), maps class order to `part_core.CL`. Sample: `part_core.sample(1000, 3000)` = P1 rows 3,000–4,000 of each class's test
   file 100, N = 10,000, rebuilt with `part_core.build`. Edits (all against the rebuilt jets): batch-order permutation (seed 1; noise floor),
   exact lab rotation of (px, py) by 0.06 rad, P1 build edit Δφ + 0.06 (stored axis kept), P1 build edit Δη + 0.06, exact longitudinal boost
   of rapidity 0.06, control angles × 1.1 about the stored axis (real mass change, mean Δlog m = +0.0941). Metrics: % calls changed (95 %
   bootstrap over jets, 2,000 resamples), McNemar exact test on correct→wrong vs wrong→correct, W→Z and Z→W rates, Δ(ℓ_W − ℓ_Z) on true W.
   Run twice: random init (`c-p5-run`, GTX 1080 Ti) and the iteration-4,999 checkpoint (copied to `models/lgatr/interim/jc2m_s0a/`,
   job `c-p5-ck5k`, GTX 1080 Ti, 35 min, of which 29 min python imports from CephFS).

## Numbers
Smoke check of the training (mlflow db, `jc2m_s0a`):

| quantity | value |
|---|---|
| train loss (it 0 → 1k → 3k → 9k → 14k) | 3.50 → 0.97 → 0.48 → 0.44 → 0.56 (batch-128 values; noisy) |
| val accuracy / AUC-ovo / loss, it 4,999 | 0.7917 / 0.9758 / 0.5854 |
| val accuracy / loss, it 9,999 | 0.8086 / 0.5356 |
| speed | 0.59 s / iteration (it 13k→14k); 186k left ≈ 30.5 h + ~37 validations → ETA ≈ 2026-10-21 05:00 UTC |
| GPU utilisation (10-min samples) | 76–92 % (9 % once, during validation at 16:58) |
| pod use vs request (`kubectl top`) | memory 7.4 GiB / 16 Gi (46 %), CPU 1.1 / 4 cores (28 %), GPU memory 9.3 / 11 GB |

Unedited accuracy on the P1 sample (N = 10,000): random init 9.88 % (chance 10 %); it-4,999 checkpoint 79.23 %, against 79.17 % on its
own validation set at the same iteration. Stored vs rebuilt jets: 0 calls changed (both models).

Edits, it-4,999 checkpoint (N = 10,000; [95 % bootstrap CI]; McNemar b = correct→wrong, c = wrong→correct):

| edit | true Δlog m | calls changed % | McNemar b/c, p | W→Z % before → after | Δ(ℓ_W−ℓ_Z) on true W | median abs Δlogit |
|---|---|---|---|---|---|---|
| batch-order permutation (noise floor) | 0 | 0.00 [0.00, 0.00] | 0/0, 1 | 17.9 → 17.9 | 0.0000 | 7e-5 |
| lab φ rotation 0.06 rad | 0 | 0.01 [0.00, 0.03] | 0/1, 1 | 17.9 → 17.9 | −0.0000 [−0.0001, 0.0001] | 3.7e-4 |
| P1 build Δφ + 0.06 | 0 | 0.01 [0.00, 0.03] | 0/1, 1 | 17.9 → 17.9 | −0.0000 [−0.0001, 0.0001] | 3.7e-4 |
| P1 build Δη + 0.06 | +0.00001 | 0.51 [0.37, 0.66] | 14/26, 0.081 | 17.9 → 17.6 | +0.019 [+0.017, +0.020] | 0.017 |
| longitudinal boost y = 0.06 | 0.0000 | 0.52 [0.38, 0.67] | 17/24, 0.35 | 17.9 → 17.4 | −0.0006 [−0.0026, +0.0015] | 0.016 |
| control: angles × 1.1 | +0.0941 | 13.61 [12.95, 14.28] | 793/329, < 1e-6 | 17.9 → 42.6 | −0.945 [−0.993, −0.895] | 0.32 |

Random init, same edits: rotation and Δφ build 0.00 % (max |Δlogit| ≤ 1.3e-4, floor 3.7e-5), Δη build 0.20 % [0.12, 0.29], boost
0.21 % [0.12, 0.30], angles × 1.1 1.81 % [1.54, 2.08].
ParT reference (P1, same rows, Δφ + 0.06): W→Z 12 → 31 % (ParT_full), 16 → 37 % (ParT_kin).

Hypothesis tests (pre-registered in the `interp/p5_lgatr_launch.py` docstring before any trained checkpoint existed):
- H_sym (rotation about the beam is an exact symmetry, so response = float noise, < 0.1 % calls) vs H1 (response above floor): 0.01 %
  [0.00, 0.03], McNemar p = 1; H_sym not excluded, threshold met. ParT's +19-point W→Z response is excluded for L-GATr (CI upper bound
  0.03 % calls changed, W→Z change 0.0 points).
- Boost / Δη shift (not exact symmetries: time reference and log E features change): 0.5 % of calls change, W→Z moves −0.3 to −0.5 points,
  Δ(ℓ_W−ℓ_Z) ≤ 0.02 in magnitude, against −0.945 for the real 9.9 % mass change. No accuracy change (McNemar p 0.08, 0.35).
- Not yet testable (needs final checkpoint, C2/C3): H_inv (mass enters only through invariants: read-out ratio ≈ 0 on iso-invariant-mass
  edits that change the axis moment $M_2$, E23 / R1 family) vs H_axis (Δη/Δφ/ΔR scalar features about the recomputed axis, or the reference-
  vector symmetry breaking, give a non-zero ratio, as ParT's 0.7–1.4).

## Reading (scoped to these measurements)
- For L-GATr the P1 edit (Δφ + 0.06 with the stored axis kept) is the same input as a rigid rotation about the beam: the network reads
  lab four-momenta and computes its Δη, Δφ, ΔR scalar features about the particle-sum axis, so the stored jet axis never enters. Both
  edits give 1 changed call in 10,000 and median |Δlogit| 3.7e-4, about 5× the batch-order floor (7e-5) and 1.5× the stored-vs-rebuilt
  float difference (2.4e-4). The zero response is therefore a property of the input pipeline plus the rotation equivariance, not evidence
  about the mass mechanism.
- The model at 2.5 % of its schedule already separates W from Z on mass: a real 9.9 % mass increase moves true-W → Z from 17.9 to 42.6 %.
- A longitudinal boost of 0.06 changes 0.5 % of calls; the network is close to but not exactly boost-invariant (the time reference and
  energy-based scalar features break it), at 2.5 % of training.
- All checkpoint numbers are interim (it 4,999 of 200,000); they are a smoke test of the eval path, not results for C3.

## Issues
- First training launch `c-p5-lgatr` failed at DataLoader construction (DistributedSampler on an IterableDataset, authors' code with
  torch 2.4); fixed by the 2-line patch, relaunched as `c-p5-lgatr2`. Old run dir `runs/jc2m/jc2m_s0` (empty) left on the volume.
- Python imports from the CephFS venv take 15–29 min (GPU idle in eval jobs `c-p5-run`, `c-p5-ck5k`). For C2/C3 evals: copy env-lgatr
  to the pod's local disk first, or run the eval on CPU via job.sh.
- Training uses 2M jets (2 % of the authors' 100M) and 200k iterations of batch 128 (authors' paper settings at full JetClass scale); the
  final accuracy will be below the published 86.6 % (100M). User default for the training scale still pending.
- `force_xformers` uses block-diagonal attention: batch composition changes logits by up to 0.13 (max) but median 7e-5; calls unchanged.
- No figure produced (interim only).

## Explainer entry (for site/explain.py, Paper C)
**Question.** ParT's mass sense moves when all particles are shifted relative to the jet axis at fixed invariant mass (C3). Does a network
built to respect the symmetries of special relativity (L-GATr) do the same, and can it serve as the contrast model?

**Background.** L-GATr represents each particle's four-momentum $p_i = (E, p_x, p_y, p_z)$ as a multivector and uses only Lorentz-equivariant
layers; extra "reference" tokens (the beam directions and the time direction $(1,0,0,0)$) break the symmetry to rotations about the beam.
A rigid rotation $\phi_i \to \phi_i + \delta$ of all particles is then an exact symmetry, $f(R_\delta p) = f(p)$, and leaves $m^2 = (\sum_i p_i)^2$
unchanged. It does change ParT's inputs if the stored jet axis is held fixed, which raises the axis moment $M_2 = \sum_i p_{T,i}\Delta R_i^2$.

**Method.** No published weights, so we train L-GATr with the authors' code and settings on 2M random-file JetClass jets. On 10,000 test
jets (1,000 per class) we compare logits before and after: a φ rotation of 0.06 rad, an η shift and a longitudinal boost of 0.06, and a
real mass change (angles × 1.1, $\Delta\log m = +0.094$); noise floor = reordering jets within batches.

**Result.** Early checkpoint (79 % accuracy): rotation changes 0.01 % of calls (ParT: true-W → Z 12 → 31 %); boost 0.5 %; real mass change
moves W → Z from 17.9 to 42.6 %.

**Checks.** Random-init network: rotation 0.00 %; stored vs rebuilt jets: 0 calls; accuracy on this sample matches the run's own
validation accuracy (79.2 % vs 79.2 %).

**Why it matters.** It confirms the trained network is rotation-equivariant end to end and that its mass response is live; the decisive
test, iso-mass edits that change $M_2$ without being a symmetry, runs on the final checkpoint.
