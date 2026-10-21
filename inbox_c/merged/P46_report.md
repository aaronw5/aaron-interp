# P46: pre-registered C6 test on L-GATr (round C11). Not run

## Conclusion
There are no results yet. The script is written and compiles, but the cluster job never started. Both submissions were placed on a node whose GPU had failed, and the auto-mode permission classifier then blocked all further kubectl commands, including relaunching the job on another node and reading job status. Claim C6 is still untested, so its confidence stays at 0.0.

## State
- Script: `interp/p46_lgatr_c6.py`. It passed `py_compile` and has been synced to the volume. It has three stages: `lgatr` (env-lgatr), `part` (env-part) and `analyze`.
- Job `anrunw-ai-c-p46-run` (deleted) and job `anrunw-ai-c-p46-run2` were both scheduled on k8s-chase-ci-01.calit2.optiputer.net. Both pods ended in `UnexpectedAdmissionError` with "GPU is lost" from the device plugin, so no code ran. c-p46-run2 should be in state Failed (backoffLimit 0). I could not confirm this because kubectl was blocked.
- L-GATr checkpoints: at 09:00 UTC the newest was it174999. it199999 was expected around 10:40 UTC.
- To relaunch: add k8s-chase-ci-01.calit2.optiputer.net to the NotIn node list in part_job.sh, then run:
  `./sync.sh && ./part_job.sh c-p46-run3 '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p46_lgatr_c6 --stage lgatr --ckpts it174999,it154999,untrained && python -m interp.p46_lgatr_c6 --stage part && python -m interp.p46_lgatr_c6 --stage analyze'`.
  Once it199999 exists, add it with `--stage lgatr --ckpts it199999` followed by `--stage analyze`. The analysis picks it as the main L-GATr model automatically. Reading the sample takes about 15–60 min on CephFS the first time; the sample is then cached in `results/lgatr/p46_sample.npz`.

## Design as implemented (pre-registered)
- **Sample.** val_5M WToQQ_123 and ZToQQ_123, 2,000 random rows each, default_rng(4646). Jets with fewer than 2 particles are dropped. The 10-class accuracy sample has 1,000 rows per class: _123, or _124 for QCD, Hcc and TTBar, whose _123 file is L-GATr's validation file.
- **Models.** L-GATr it174999 and it154999 (later it199999), and an untrained L-GATr (same config, torch seed 4646). ParT references: released ParT_full and ParT_kin, plus the P10 retrains ParT_kin_s0 and ParT_full_s0, which were trained on the same 2M jets as L-GATr.
- **Edits.** pT, particle masses and PID/track inputs are held fixed in every edit.
  - PHI ±0.06: every particle's Δφ shifted by ±0.06 rad with the stored axis fixed. This is an exact rotation about the beam.
  - BOOST ±0.06: every particle's rapidity shifted by ±0.06. This is an exact longitudinal boost, so the jet and pair masses are unchanged.
  - E6 0.005–0.06 rad: the constituents are shifted rigidly and the stored axis moves the opposite way, so the lab four-vectors are unchanged.
  - E1 ±2/5/10 %: P43's two-particle spread edit. It changes the invariant mass.
  - E1m: E1 followed by a second-pair compensation that restores the invariant mass exactly (float64 bisection).
  - E5±: P43's null edit, with M2 and the centroid unchanged.
  - Each edit gets a matched dilation E4. E4 scales all angles about the stored axis so that M2 = Σ pT r² changes by the same amount the edit predicts.
- **Metric.** d = ℓ_W − ℓ_Z. R = Σ Δd(edit) / Σ Δd(matched dilation).
- **L-GATr route split.** A is the full edit. D holds the 7 scalar features (log pT, log E, the log ratios, dφ, dη, dR, all about the four-momentum-sum axis) at their unedited values. F edits only the features. C applies only to BOOST: it co-boosts the time spurion with features held, which is an exact-symmetry check.
- **Statistics.** 2,000 bootstraps over jets, shared across edits and models; sign-flip permutation tests; McNemar tests; Holm correction over the edit × dose grid. L-GATr minus ParT R differences use paired bootstraps.

## Expected by construction (checked against the embedding code, `experiments/tagging/embedding.py`)
L-GATr computes its scalar features from the four-momenta, about the axis of their sum. It never reads jet_eta/jet_phi or the stored Δη/Δφ. Two consequences:
- PHI should give R ≈ 0 to float precision. A rotation about the beam leaves both spurions and all 7 features unchanged.
- E6 gives L-GATr the same input up to float32 rebuilding.

Both are symmetry or plumbing checks, not learned properties (PAPERS.md, reviewer lens, round C10). The informative edits are:
- BOOST: the frame-dependent log E route and the time spurion.
- E1m and E5: rearrangements at fixed mass.

At fixed invariant mass and fixed pT centroid, M2 about the momentum axis is itself fixed to leading order in small angles (m² ≈ M0·M2c). Any iso-mass edit that changes M2 therefore does so about an axis L-GATr does not read.
