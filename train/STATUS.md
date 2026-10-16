# Small-tagger training: status (2026-10-16)

Set: 7 models (`tf-1b-d16`, `tf-2b-d16`, `tf-1b-d32`, `tf-2b-d32`, `linf-1b-d32-k4`, `linf-2b-d32-k8`, `dsets-d64`)
× N ∈ {32, 64} × seeds {0, 1} = 28 runs. Defaults of `train_small.py`: AdamW lr 1e-3 (200 warm-up steps, cosine to 0 over
30 epochs), batch 1024, patience 5 on val loss, train = train archive minus `splits.npz` dev (595,000), val = dev (25,000),
test = val archive (260,000). Code snapshot per job under `/j-jepa-vol/aaron-interp/models/small/_snapshots/`.

## Jobs (namespace cms-ml, labels owner=anrunw, study=aaron-interp, set=small-taggers; 1 GPU each, limits = requests)
| job | runs (all 7 models, concurrently on one GPU) | request | state |
|---|---|---|---|
| `anrunw-ai-train-smoke` | CPU smoke test: tf-1b-d16, linf-1b-d32-k4, dsets-d64 (N=32), tf-1b-d16 `--cls_token` (N=64); 1 epoch, 20k jets → `models/small-smoke/` | 1 CPU, 2 Gi (exempt size) | complete, exit 0, 127 s |
| `anrunw-ai-train-n32-s0` | N=32, seed 0 | 6 CPU, 16 Gi | complete: 7/7 runs, 5 min on an NVIDIA L40 (rci-tide-gpu-05.sdsu.edu); monitor at 3 min: GPU 100 %, 6.2 GB GPU memory, container memory 15.0 GiB, 7 processes |
| `anrunw-ai-train-n32-s1` | N=32, seed 1 | 6 CPU, 20 Gi | launched 2026-10-16 (after n32-s0's memory reading) |
| `anrunw-ai-train-n64-s0` | N=64, seed 0 | 6 CPU, 20 Gi | launched 2026-10-16 |
| `anrunw-ai-train-n64-s1` | N=64, seed 1 | 6 CPU, 20 Gi | launched 2026-10-16 |

Memory was raised from 16 to 20 Gi for the remaining jobs because n32-s0 used 15.0 GiB of 16 (7 CUDA processes) and the
N=64 runs hold twice the data per process. CPU use per job is printed by the monitor lines of the later jobs (`cpu … cores`)
and at the end of each job log (`average cpu`); n32-s0 ran before that line was added.

## First results: N = 32, seed 0 (test archive, 260,000 jets; best = lowest val loss; none of the runs early-stopped)
| model | params | best epoch | val acc | test acc | test AUC g/q/W/Z/t | rej50 g/q/W/Z/t |
|---|---|---|---|---|---|---|
| tf-1b-d16 | 2,405 | 30 | 0.6836 | 0.6808 | 0.866/0.883/0.914/0.904/0.940 | 17/22/29/37/52 |
| tf-2b-d16 | 4,629 | 30 | 0.7516 | 0.7524 | 0.910/0.909/0.954/0.946/0.956 | 30/29/80/202/119 |
| tf-1b-d32 | 8,901 | 30 | 0.7189 | 0.7183 | 0.894/0.898/0.933/0.921/0.947 | 24/27/39/75/96 |
| tf-2b-d32 | 17,445 | 30 | 0.7737 | 0.7724 | 0.922/0.917/0.963/0.955/0.960 | 37/33/114/319/149 |
| linf-1b-d32-k4 | 9,029 | 29 | 0.7439 | 0.7416 | 0.906/0.903/0.948/0.940/0.952 | 28/31/83/153/81 |
| linf-2b-d32-k8 | 17,957 | 29 | 0.7853 | 0.7841 | 0.930/0.919/0.968/0.963/0.962 | 41/37/202/493/133 |
| dsets-d64 | 11,269 | 30 | 0.7794 | 0.7778 | 0.923/0.916/0.966/0.959/0.958 | 36/35/163/364/115 |

Reference on the same test set: JEDI-linear 3-feature n32 ≈ 78 %, n64 80.94 %.

Validation curve, tf-1b-d16-n32 seed 0 (val loss / val acc): init 1.678/0.201; ep1 1.057/0.593; ep2 1.020/0.612; ep3 0.993/0.625;
ep4 0.970/0.634; ep6 0.949/0.643; ep10 0.922/0.655; ep15 0.904/0.663; ep20 0.883/0.676; ep25 0.873/0.681; ep30 0.871/0.684.
dsets-d64-n32 seed 0: init 1.618/0.200; ep1 0.996/0.611; ep3 0.778/0.719; ep8 0.708/0.751; ep14 0.662/0.773; ep22 0.642/0.778;
ep30 0.639/0.779 (val loss spikes at ep9 1.046, ep11 0.788, ep13 0.769, then recovers).
Dead units (fraction with pre-activation max ≤ 0 on 10k held-out jets): transformers 0 in every GELU layer at every epoch;
dsets-d64 phi1/phi2/rho1/rho2/rho3 = 0.016/0.094/0.219/0.156/0.188 from epoch 1 on (init 0.016/0.062/0.172/0.188/0.125).

## Where results land
`/j-jepa-vol/aaron-interp/models/small/<model>-n<N>/seed<k>/`: `arch.json`, `log.csv` (per epoch incl. `dead_<layer>`),
`ckpt/epoch_0.pt` … `epoch_30.pt`, `best.pt`, `test.json`, `test_probs.npy`, `train.log`.
Job logs: `/j-jepa-vol/aaron-interp/logs/train-<job>.log` (start/end per run, monitor lines every 3 min).

## Check progress
```
kubectl -n cms-ml get jobs -l set=small-taggers
POD=$(kubectl -n cms-ml get pods -l owner=anrunw -o name | sed 's|pod/||' | grep '^anrunw-helper-' | head -1)
kubectl -n cms-ml exec $POD -- tail -5 /j-jepa-vol/aaron-interp/logs/train-n64-s0.log
kubectl -n cms-ml exec $POD -- bash -c 'cd /j-jepa-vol/aaron-interp/models/small; for f in */seed*/test.json; do printf "%-28s %s\n" $f "$(grep -m1 \"accuracy\" $f)"; done'
kubectl -n cms-ml exec $POD -- cut -d, -f1,4-7 /j-jepa-vol/aaron-interp/models/small/tf-1b-d16-n64/seed0/log.csv
```
A failed or missing run is rerun by resubmitting its job with a suffix (`python train/make_jobs.py --only n64-s0 --suffix -b
--mem 20`, then `kubectl apply -f train/jobs/n64-s0-b.yaml`); runs with `test.json` are skipped.
