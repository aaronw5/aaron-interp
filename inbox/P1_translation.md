# P0 / P1 — ParT setup and the translation test (agent report, 2026-10-16; P1 numbers pending: cluster job queued)

## Status
- P0 (setup + smoke test): done. Local smoke test ran before the user's "no local compute" change; the cluster run of the
  same test is queued (GPU job `anrunw-ai-p1`, pod Pending: insufficient GPUs at launch time).
- P1: script written and verified to import/run on the cluster (50 jets, CPU, helper pod); the full run is the second half
  of the same job (`python -m interp.p0_smoke_part && python -m interp.p1_translation_part`). No P1 numbers yet.
  When the job finishes: log `/j-jepa-vol/aaron-interp/logs/p1.log`, results `/j-jepa-vol/aaron-interp/results/part/
  {p0_smoke,p1_translation}.json`; then on the helper pod `python -m interp.p1_plot` (matplotlib 3.7.2 is in the image; none
  of the local pythons has it) and copy `results/part/p1_translation.png` + the two JSONs back with `kubectl exec … cat`.

## What was copied / integrated (all inside aaron-interp)
- `jetdistill/part/` ← `jld-features/jetdistill/part/` (whole package). `network`, `data`, `clsfit`, `attnfit`, `clseval`,
  `clspage`, `fetch` import; `part/jets.py` does not (needs `observables.library.is_block` / `compute.block_values`, which the
  JEDI-only observables lack) — not needed for P0/P1.
- `jetdistill/config.py`: `JETDISTILL_TAGGER` switch (unset = jedi: DATA/MODELS/RESULTS defaults, CLASSES, N_* unchanged;
  checked by import), `part` branch (CLASSES 10, NETWORKS ('kin','full'), RELU False, N_*), plus NC, net_dir, parse_net,
  n_particles. ParT SETUPS not added (pipeline not used).
- `interp/part_core.py`: PRE/files/fof/PK, kin, build, jmass, sub, cat (G192 header), moments (E6 definition), `Attention`
  hooks (G201: forward(Jb, capture, M), pre-hook setting need_weights / adding a mask), N98/ACTIVE constants, docstring with
  the working command and the env vars each module reads. `interp/_lund.py`, `interp/_wintop.py` copied verbatim with the
  authorship note.
- `run_part.sh` (local wrapper, jld-features venv, MPS; now unused per the user's rule), `part_job.sh` (cluster GPU job:
  4 CPU, 16 Gi, 1 GPU, weaver image, PATH → `/j-jepa-vol/aaron-interp/env-part`, JETDISTILL_TAGGER=part, JETDISTILL_DEVICE=cuda,
  JETDISTILL_DATA=/j-jepa-vol/aaron-interp/code/cluster/data_part, JETDISTILL_MODELS=/j-jepa-vol/aaron-interp/models/part),
  `cluster/data_part/test/files.json` (the 20 JetClass test files, absolute paths into /j-jepa-vol/JetClass/Pythia/test_20M).
- Cluster: weights copied to `/j-jepa-vol/aaron-interp/models/part/ParT_{full,kin,kinpid}.pt` (md5 identical to local:
  719fb783…, 47795640…, 06a63e24…); venv `/j-jepa-vol/aaron-interp/env-part` (--system-site-packages over /opt/conda:
  torch 1.13.1 + CUDA 11.7, uproot 5.0.10, awkward 2.3.1, numpy 1.24.4) with weaver-core 0.4.17 — the image's weaver-core 0.3.5
  has no ParticleTransformer at all.
- Scripts: `interp/p0_smoke_part.py`, `interp/p1_translation_part.py`, `interp/p1_plot.py`.

Local working command (recorded in part_core.py; not to be run now):
`JETDISTILL_TAGGER=part KERAS_BACKEND=jax JETDISTILL_DEVICE=mps JETDISTILL_DATA=…/jet-latent-distillation/data_part
JETDISTILL_MODELS=…/jet-latent-distillation/models_part AI_RESULTS=results PYTHONPATH=…/aaron-interp
…/jld-features/.venv/bin/python interp/<script>.py` (= `./run_part.sh interp/<script>.py`).

## Smoke test (local, MPS, 400 jets/class, rows 3,000–3,400; `results/part/p0_smoke.json`, `logs/p0_smoke.log`)
- ParT_full 86.2 % (G192 86.2 %), ParT_kin 74.4 % (74.4 %). Own-class rates identical to G192's "before" columns
  (QCD 77, Hbb 92, Hcc 83, Hgg 81, H4q 84, Hqql 100, Zqq 72, Wqq 78, Tbqq 96, Tbl 100).
- Hooks on (capture, no mask): max logit difference 8.6e-6 (full), 9.5e-6 (kin).
- Moments on JetClass jets: corr(√(M0M2 − |M1|²), m) 0.99905, median ratio 0.9929; |M1|²/(M0M2) median 2.0e-3
  (JEDI/hls4ml 3.5e-4); particle mass median 0.008 GeV, p95 0.49 GeV.
- Cluster check (helper pod, CPU, 5 jets/class): ParT_full runs under weaver 0.4.17 / torch 1.13.1; 47/50 correct;
  rebuild identity max |ΔE| 6e-5 GeV, |Δpz| 8e-6 GeV (after the fix below).

## Bugs and surprises
1. **JetClass `part_deta` is sign-flipped**: η_particle = jet_eta + sign(jet_eta)·deta (max deviation 1.2e-7 on 1,000
   jets); φ = jet_phi + dphi exactly. G192's `build` used `jet_eta + deta`, which mirrors the particles of the 50 % of jets
   with jet_eta < 0 in η: pz/E off by up to 284/265 GeV, jet_energy by up to 16 %, and the *unedited* rebuilt jets changed
   0.90 % of ParT_full's calls (4.47 % of ParT_kin's). Fixed in `interp/part_core.build` (rebuild now exact to float32).
   G192–G206's edited jets carried this mirror for negative-η jets (effect on G192's unedited baseline: 0.9 % of calls).
2. **G-series neuron numbers are 1-based**: index 98 of z has s.d. 0.0000; the 13 active neurons here are indices
   16, 97, 68, 18, 24, 104, 115/120, 78, 52, 81, 83, 124 = G5b's 17, 98, 69, … − 1, in G5b's strength order. "n98" = index 97
   (s.d. 1.77 on 50 jets); P1 reports it as such and logs its last-layer weights × s.d. against G5b (+2.06 W, −0.67 Z).
3. `files.json` paths are relative to the jld-features working directory; `fof` resolves them against DATA.parent, or
   takes absolute paths (cluster file).
4. No local python has matplotlib (jld venv, homebrew 3.11, system 3.14); the cluster image has 3.7.2.
5. The attention hooks use `register_forward_pre_hook(with_kwargs=True)` (torch ≥ 2.0); on the cluster's torch 1.13 the
   smoke test reports them unavailable and continues; P1 does not use them.

## P1 design (as run on the cluster)
1,000 jets/class, rows 3,000–4,000; ParT_full and ParT_kin. Baseline = the same jets rebuilt with δ = 0. Edits: Δη or Δφ of
every real particle + δ, δ ∈ {0.01, 0.02, 0.03, 0.045, 0.06, 0.09} (pT and particle masses kept; padded slots checked);
controls angles × 1.1 and × 1.25. Metrics per edit: accuracy, calls changed, W→Z / Z→W (fraction of true W / Z jets), mean
Δ(logit_W − logit_Z) on W jets, n98 (index 97) mean shift / s.d. (all, W, Z) and every active neuron's, ridge read-out of
log m on the unedited z (50/50 split, held-out R², linear and + PCA products) and its mean shift on the test half vs
½Δlog(M0·M2) from the edited particles and vs 0, with the across-jet slope on the per-jet prediction. JEDI control values
for the figure (E6, results/jedi/e6_mass.json): W→Z 0.020 → 0.262/0.293 (η/φ, δ 0.03) → 0.765/0.763 (0.06); Δlog m̂ at z
0.067/0.070 → 0.253/0.256; √(M0M2) prediction 0.113 → 0.306; angles × 1.1: W→Z 0.708, Δlog m̂ 0.085.
Expected prediction size on JetClass jets: M2/M0 ≈ (m/pT)² is larger than for hls4ml jets, so ½Δlog M2 at 0.06 rad is
of order 0.1 rather than 0.3; the δ grid goes to 0.09 for that reason.
