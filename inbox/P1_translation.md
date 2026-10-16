# P0 / P1 — ParT setup and the translation test (agent report, 2026-10-16)

Run on the cluster: GPU job `anrunw-ai-p1b` (NVIDIA A10; log `logs/p1b_cluster.log`, copy of `/j-jepa-vol/aaron-interp/logs/p1b.log`),
`python -m interp.p0_smoke_part && python -m interp.p1_translation_part`; results `results/part/p0_smoke.json`,
`results/part/p1_translation.json`, figure `results/part/p1_translation.png` (rendered on the helper pod with `interp/p1_plot.py`).
The first job `anrunw-ai-p1` (TITAN Xp, after a 60-min queue) failed on a formatting bug in my smoke script (fixed, relaunched).

## P0 — what was copied / integrated (paths under aaron-interp)
- `jetdistill/part/` ← `jld-features/jetdistill/part/` (whole package). `network`, `data`, `clsfit`, `attnfit`, `clseval`, `clspage`,
  `fetch` import; `part/jets.py` does not (needs `observables.library.is_block` / `compute.block_values`, absent in the JEDI-only
  observables) — not needed for P0/P1.
- `jetdistill/config.py`: `JETDISTILL_TAGGER` switch; unset → every JEDI value unchanged (checked by import: DATA/MODELS/RESULTS
  defaults, CLASSES, N_*, SETUPS); `part` → CLASSES (10), NETWORKS ('kin', 'full'), RELU False, N_*; plus NC, net_dir, parse_net,
  n_particles. ParT SETUPS not added (pipeline not used).
- `interp/part_core.py`: PRE/files/fof/PK, kin, build, jmass, sub, cat (G192 header), moments (E6 definitions), `Attention` hooks
  (G201: forward(Jb, capture, M), pre-hook setting need_weights / adding a mask), N98 / ACTIVE constants, docstring with the working
  command and the env vars each module reads. `interp/_lund.py`, `interp/_wintop.py` copied verbatim with the authorship note.
- `run_part.sh` (local wrapper: jld-features venv, MPS — not to be used while the user's computer is busy), `part_job.sh` (cluster GPU
  job: 4 CPU, 16 Gi, 1 GPU, weaver image, PATH → `/j-jepa-vol/aaron-interp/env-part`, JETDISTILL_TAGGER=part, JETDISTILL_DEVICE=cuda,
  JETDISTILL_DATA=/j-jepa-vol/aaron-interp/code/cluster/data_part, JETDISTILL_MODELS=/j-jepa-vol/aaron-interp/models/part; creates
  env-part if missing), `cluster/data_part/test/files.json` (the 20 JetClass test files, absolute paths into
  /j-jepa-vol/JetClass/Pythia/test_20M; travels with sync.sh).
- Cluster: weights `/j-jepa-vol/aaron-interp/models/part/ParT_{full,kin,kinpid}.pt` (md5 identical to the local files: 719fb783…,
  47795640…, 06a63e24…); venv `/j-jepa-vol/aaron-interp/env-part` = /opt/conda (python 3.10, torch 1.13.1 + CUDA 11.7, uproot 5.0.10,
  awkward 2.3.1, numpy 1.24.4) + weaver-core 0.4.17 (the image's weaver-core 0.3.5 has no ParticleTransformer).
- Scripts: `interp/p0_smoke_part.py`, `interp/p1_translation_part.py`, `interp/p1_plot.py`.
- Local command (recorded in part_core.py): `JETDISTILL_TAGGER=part KERAS_BACKEND=jax JETDISTILL_DEVICE=mps
  JETDISTILL_DATA=…/jet-latent-distillation/data_part JETDISTILL_MODELS=…/jet-latent-distillation/models_part AI_RESULTS=results
  PYTHONPATH=…/aaron-interp …/jld-features/.venv/bin/python interp/<script>.py` (= `./run_part.sh interp/<script>.py`).
  Env vars read: config.py — JETDISTILL_TAGGER/DATA/MODELS/RESULTS/SMOKE; part/network.py — JETDISTILL_DEVICE ('cpu' default);
  part/data.py — none. KERAS_BACKEND is unused by the ParT path.

## Smoke test (400 jets/class, rows 3,000–3,400 of each class's first test file)
| | ParT_full | ParT_kin |
|---|---|---|
| cluster (A10), accuracy | 86.2 % (G192 86.2 %) | 74.4 % (G192 74.4 %) |
| local (MPS, before the no-local-compute rule) | 86.2 % | 74.4 % |
| own-class rates | identical to G192 (QCD 77, Hbb 92, Hcc 83, Hgg 81, H4q 84, Hqql 100, Zqq 72, Wqq 78, Tbqq 96, Tbl 100) | QCD 70, Hbb 63, Hcc 59, Hgg 74, H4q 78, Hqql 88, Zqq 59, Wqq 69, Tbqq 90, Tbl 94 |
| rebuild of the unedited jets through `build`: calls changed | 0.05 % (max logit diff 1.9) | 1.05 % (8.7) |
| inference | 0.4 s / 1,000 jets | 0.1 s / 1,000 jets |
Rebuild identity: max |Δpx,py| 6e-5, |Δpz| 3e-5, |ΔE| 1.2e-4 GeV, mass 3.4e-4 relative; the stored jet_energy differs from ΣE of the
128 stored particles by up to 2.5 % (median ratio 0.99984) and `build` recomputes it (→ logerel input), hence the rebuilt δ = 0
baseline in P1. Moments on JetClass jets: corr(√(M0M2 − |M1|²), m) 0.99905, median ratio 0.9929; |M1|²/(M0M2) median 2.0e-3
(hls4ml 3.5e-4); M2/M0 median 0.0336; particle mass median 0.008 GeV, p95 0.49 GeV. Attention hooks: max logit diff 9e-6 locally
(torch 2.14); unavailable on the cluster's torch 1.13 (`with_kwargs`), not used by P1.

## P1 — procedure
Sample: 1,000 test jets per class (rows 3,000–4,000), 10,000 jets, mass median 114.7 GeV; ParT_full and ParT_kin (accuracy 85.9 % /
74.5 % on the rebuilt baseline; raw 85.9 / 74.8 %). Edit: every real particle's (Δη, Δφ) + δ along η or along φ, δ ∈ {0.01, 0.02,
0.03, 0.045, 0.06, 0.09}; pT and particle masses kept, px/py/pz/E rebuilt (η = jet_eta + sign(jet_eta)·Δη); padded slots unchanged
(checked). Invariant mass: φ shift mean Δlog m 0.00000 (max |.| 0.007, float32); η shift mean +0.00001 (max 0.009). Σ pT unchanged
(|Δlog M0| ≤ 1e-7). ½Δlog(M0·M2) from the edited particles (mean): η 0.0002 / 0.0051 / 0.0139 / 0.0335 / 0.0593 / 0.1245;
φ 0.0028 / 0.0101 / 0.0213 / 0.0441 / 0.0729 / 0.1427 (η < φ: |M1| is not exactly 0 and the η shift points against the mean M1η).
Controls: angles × 1.1 (Δlog m +0.094, ½Δlog M0M2 +0.095) and × 1.25 (+0.221 / +0.223); JEDI E6 (inbox/E6_mass.md,
results/jedi/e6_mass.json). Read-out: ridge on the unedited class token z (128-d), 50/50 split, alpha on a validation part of the
train half; held-out R² of log m: ParT_full 0.881 linear (13 active neurons only 0.616; + PCA products 0.869), ParT_kin 0.920
(0.495; 0.929). Index 97 (= G-series "n98"; last layer × s.d.: W +2.06, Z −0.67 exactly as G5b; corr with log m −0.30 on
ParT_full); for ParT_kin index 97 is not a W/Z neuron (last layer × s.d. W +0.10) and its row is reported only for completeness.
Slope: across-jet regression of Δlog m̂ on the per-jet ½Δlog(M0·M2) (undefined for the scalings, where the prediction is the same
for every jet).

## P1 — table (φ shift = exact rotation; η shift in parentheses where it differs by more than the last digit)
ParT_full (unedited: W→Z 0.117, Z→W 0.160, W→W 0.779, Z→Z 0.720)
| δ [rad] | acc | calls changed | W→Z | Z→W | Δ(logit_W − logit_Z) on W | idx 97 shift / s.d.: all, W, Z | Δlog m̂ (lin) | pred ½Δlog M0M2 | slope, r |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 86.0 | 0.7 % (0.7) | 0.114 (0.114) | 0.159 (0.158) | −0.02 | −0.002, −0.008, −0.008 | +0.002 (−0.001) | 0.003 (0.000) | 0.01, 0.01 |
| 0.02 | 85.9 | 1.5 % (1.4) | 0.125 (0.122) | 0.145 (0.153) | −0.10 | −0.006, −0.034, −0.029 | +0.007 (+0.001) | 0.010 (0.005) | 0.02, 0.05 |
| 0.03 | 85.7 (86.0) | 2.8 % (2.3) | 0.149 (0.133) | 0.128 (0.149) | −0.23 (−0.14) | −0.012, −0.079, −0.062 | +0.014 (+0.006) | 0.021 (0.014) | 0.04, 0.09 |
| 0.045 | 85.1 (85.6) | 5.1 % (4.2) | 0.216 (0.192) | 0.112 (0.121) | −0.58 (−0.43) | −0.022, −0.191, −0.124 | +0.030 (+0.019) | 0.044 (0.034) | 0.08, 0.16 |
| 0.06 | 83.7 (84.5) | 8.1 % (7.2) | 0.310 (0.283) | 0.079 (0.090) | −1.11 (−0.91) | −0.029, −0.353, −0.180 | +0.052 (+0.037) | 0.073 (0.059) | 0.14, 0.25 |
| 0.09 | 77.7 (79.4) | 17.3 % (15.2) | 0.490 (0.467) | 0.047 (0.046) | −2.13 (−1.98) | −0.001, −0.686, −0.227 | +0.110 (+0.090) | 0.143 (0.125) | 0.31, 0.45 |
| angles × 1.1 | 80.9 | 12.6 % | 0.371 | 0.074 | −1.46 | +0.035, −0.428, −0.194 | +0.076 | 0.095 (true Δlog m 0.094) | – |
| angles × 1.25 | 60.9 | 37.2 % | 0.461 | 0.031 | −2.39 | +0.242, −0.747, −0.225 | +0.171 | 0.223 (0.221) | – |
Largest mean neuron shifts at φ 0.06 (s.d. units): idx 83 −0.09, 104 −0.07, 115 −0.05, 78 +0.04, 52 +0.04 (at × 1.1: 83 −0.18, 78 +0.12, 81 +0.10).

ParT_kin (unedited: W→Z 0.162, Z→W 0.224, W→W 0.684, Z→Z 0.600)
| δ [rad] | acc | calls changed | W→Z | Z→W | Δ(logit_W − logit_Z) on W | idx 97 shift / s.d.: all, W, Z | Δlog m̂ (lin) | pred ½Δlog M0M2 | slope, r |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 74.5 (74.8) | 1.9 % (2.6) | 0.169 (0.162) | 0.225 (0.215) | −0.02 | +0.001, +0.005, +0.004 | +0.002 (+0.003) | 0.003 (0.000) | 0.02, 0.04 |
| 0.02 | 74.4 (74.7) | 3.6 % (4.3) | 0.182 (0.177) | 0.206 (0.201) | −0.09 (−0.05) | +0.005, +0.011, +0.006 | +0.010 (+0.011) | 0.010 (0.005) | 0.07, 0.11 |
| 0.03 | 73.8 (74.3) | 5.8 % (6.6) | 0.210 (0.198) | 0.185 (0.176) | −0.20 (−0.16) | +0.013, +0.027, +0.011 | +0.024 (+0.025) | 0.021 (0.014) | 0.14, 0.20 |
| 0.045 | 71.5 | 11.3 % (12.8) | 0.286 (0.280) | 0.150 (0.147) | −0.44 (−0.41) | +0.035, +0.056, +0.026 | +0.054 (+0.057) | 0.044 (0.034) | 0.27, 0.33 |
| 0.06 | 67.2 (66.3) | 19.6 % (22.1) | 0.373 (0.358) | 0.102 (0.099) | −0.79 (−0.78) | +0.070, +0.090, +0.053 | +0.095 (+0.101) | 0.073 (0.059) | 0.43, 0.47 |
| 0.09 | 54.0 (51.9) | 39.9 % (42.6) | 0.442 (0.422) | 0.042 (0.039) | −1.35 (−1.40) | +0.184, +0.168, +0.157 | +0.193 (+0.202) | 0.143 (0.125) | 0.72, 0.66 |
| angles × 1.1 | 68.4 | 18.7 % | 0.455 | 0.087 | −1.07 | −0.081, −0.006, −0.061 | +0.079 | 0.095 (0.094) | – |
| angles × 1.25 | 46.4 | 49.6 % | 0.493 | 0.043 | −2.00 | −0.250, −0.085, −0.076 | +0.171 | 0.223 (0.221) | – |
Largest mean neuron shifts at φ 0.06: idx 104 −0.23, 124 −0.22, 18 −0.17, 24 −0.14, 115 +0.14 (at × 1.1: 16 +0.28, 120 −0.14, 52 −0.12).

JEDI-linear (E6, 3,000 jets/class hls4ml, same metrics at the z stage): δ 0.03: W→Z 0.262 (η) / 0.293 (φ), Δlog m̂ 0.067 / 0.070,
prediction 0.113; δ 0.06: W→Z 0.765 / 0.763, Δlog m̂ 0.253 / 0.256, prediction 0.306; unedited W→Z 0.020; angles × 1.1: W→Z 0.708,
Δlog m̂ 0.085 (true 0.095).

## Reading (scoped to the measurements)
- Both ParT networks respond to the translation: at δ = 0.06 rad ParT_full changes 8.1 % of its calls and moves 31 % of true W jets
  to Z (from 12 %); ParT_kin 20 % and 37 % (from 16 %); JEDI 76 % (from 2 %). On the same ParT jets, angles × 1.1 (a real +9.4 %
  mass change) gives W→Z 37 % (full) and 46 % (kin), i.e. the 0.06-rad translation produces 0.8 (full) / 0.8 (kin) of the W→Z effect
  of a +10 % mass change, and the 0.09-rad translation exceeds it (49 % / 44 %).
- The read-out mass follows the √(M0·M2) direction with a smaller slope for ParT_full (Δlog m̂ 0.052 vs predicted 0.073 at 0.06;
  0.110 vs 0.143 at 0.09; ratio 0.7–0.8 at every δ ≥ 0.03), and with a larger one for ParT_kin (0.095 vs 0.073; 0.193 vs 0.143;
  ratio 1.1–1.4). The invariant-mass expectation is 0 at every δ. The across-jet slope on the per-jet prediction rises with δ
  (full 0.14 → 0.31, r 0.25 → 0.45; kin 0.43 → 0.72, r 0.47 → 0.66): jets with larger δ²M0/M2 (narrower jets) move more, as the
  √(M0M2) term predicts, but the slope is below 1 and part of the shift is not explained by the per-jet prediction.
- Under the translation the W/Z neuron (index 97) moves on W jets in the direction it moves under a real mass increase (−0.35 s.d.
  at 0.06 vs −0.43 at × 1.1 on ParT_full), while its all-jet mean hardly moves (−0.03 vs +0.04); on ParT_kin index 97 is not a W/Z
  neuron. The η shift gives 10–30 % smaller effects than the φ shift on ParT_full at the same δ, consistent with its smaller
  ½Δlog M0M2 (0.059 vs 0.073 at 0.06).
- For the PAPERS.md cross-model mass row: the expectation "pair features are translation invariant, so ParT's mass should be the
  invariant mass" is not what the measurement shows; the per-particle inputs (Δη, Δφ, ΔR, and e_log/logerel for η) carry a
  translation response of the same sign as JEDI's, smaller per rad on ParT_full (W→Z +19 pt vs +74 pt at 0.06; Δlog m̂ 0.05 vs 0.25,
  predictions 0.07 vs 0.31) and intermediate on ParT_kin.

## Bugs and surprises
1. JetClass `part_deta` is sign-flipped: η = jet_eta + sign(jet_eta)·deta (max deviation 1.2e-7 on 1,000 jets); φ = jet_phi + dphi
   exactly. G192's `build` used jet_eta + deta, which mirrors the particles of the 50 % of jets with jet_eta < 0 in η: pz/E off by up
   to 284/265 GeV, jet_energy by up to 16 %, and the *unedited* rebuilt jets changed 0.90 % of ParT_full's calls (4.47 % ParT_kin,
   local check). Fixed in `interp/part_core.build`. G192–G206's edited jets carried this mirror for negative-η jets.
2. G-series neuron numbers are 1-based: z index 98 has s.d. 0.0000; the active indices are G5b's list minus one, so "n98" = index 97
   (s.d. 1.76; last layer × s.d. W +2.06, Z −0.67 = G5b's numbers). The task text's "index 98" is index 97.
3. JetClass's stored jet_energy ≠ ΣE of the 128 stored particles (up to 2.5 %); `build` recomputes it.
4. files.json paths are relative to jld-features' cwd; `fof` resolves against DATA.parent or takes absolute paths.
5. No local python has matplotlib; the cluster image has 3.7.2 (figure rendered on the helper pod).
6. Attention hooks need torch ≥ 2.0; on torch 1.13 the smoke test reports them unavailable (my first version crashed while
   formatting that message: job `anrunw-ai-p1`); P1 does not use them.
7. The slope column is meaningless for the scalings (identical prediction for every jet); printed as ±1e4 in the log, omitted here.
