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
| `anrunw-ai-train-n32-s1` | N=32, seed 1 | 6 CPU, 20 Gi | **failed**: Job started 20:04:46 UTC, pod failed 20:36:27 UTC, `BackoffLimitExceeded` (backoffLimit 0). The container never ran: no job log, no `_snapshots/n32-s1-*`, no `*-n32/seed1` directories (gpu_job.sh creates the first two in its first lines). The pod object and its events no longer existed when checked (~5 h later), so the reason (scheduling/eviction, image pull, GPU admission) is not recorded. |
| `anrunw-ai-train-n32-s1-b` | N=32, seed 1 (resubmit of the above) | 6 CPU, 20 Gi | launched 2026-10-17 ~01:07 UTC; `backoffLimit: 1` (one retry; gpu_job.sh skips runs with test.json); running: all 7 runs started 01:14:44 UTC on an RTX 2080 Ti (k8s-haosu-12.sdsc.optiputer.net) |
| `anrunw-ai-train-n64-s0` | N=64, seed 0 | 6 CPU, 20 Gi | complete: 7/7 runs, 21 min on an RTX 2080 Ti (k8s-haosu-16.sdsc.optiputer.net); monitor: GPU 100 %, 6.8 GB GPU memory, container memory 12.8 GiB peak, CPU 5.0–6.0 cores (job average 5.13) |
| `anrunw-ai-train-n64-s1` | N=64, seed 1 | 6 CPU, 20 Gi | complete: 7/7 runs, 16 min on an RTX 2080 Ti (k8s-haosu-07.sdsc.optiputer.net); GPU 98–100 %, memory 12.7 GiB peak, CPU job average 5.01 cores |

Memory was raised from 16 to 20 Gi after n32-s0 used 15.0 GiB of 16 (7 CUDA processes); the N=64 jobs then peaked at
12.8 GiB (64 % of 20 Gi). CPU: N=64 jobs averaged 5.0–5.1 of 6 cores (83–86 %). GPU: 100 % while ≥ 3 runs were active,
98 % with 1 run left.

## Results so far (test archive, 260,000 jets; best = lowest val loss)

### N = 32, seed 0 (no run early-stopped)
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

### N = 64, seeds 0 and 1 (test acc seed0 / seed1; best epoch / epochs run per seed)
| model | params | best ep (s0; s1) | test acc s0 / s1 | test AUC g/q/W/Z/t (s0) | rej50 g/q/W/Z/t (s0) |
|---|---|---|---|---|---|
| tf-1b-d16 | 2,405 | 30/30; 30/30 | 0.6931 / 0.6921 | 0.858/0.889/0.933/0.917/0.940 | 13/24/46/72/49 |
| tf-2b-d16 | 4,629 | 30/30; **9/14** | 0.7384 / 0.7239 | 0.895/0.902/0.954/0.936/0.950 | 21/26/70/208/89 |
| tf-1b-d32 | 8,901 | 30/30; 30/30 | 0.6938 / 0.6915 | 0.868/0.890/0.930/0.911/0.942 | 15/24/42/64/61 |
| tf-2b-d32 | 17,445 | 30/30; **7/12** | 0.7619 / 0.7342 | 0.917/0.913/0.960/0.944/0.956 | 30/29/85/313/140 |
| linf-1b-d32-k4 | 9,157 | 30/30; 30/30 | 0.7669 / 0.7778 | 0.936/0.910/0.953/0.948/0.956 | 54/34/120/188/77 |
| linf-2b-d32-k8 | 18,469 | 29/30; 29/30 | 0.8106 / 0.8076 | 0.950/0.929/0.976/0.972/0.968 | 71/38/367/1236/170 |
| dsets-d64 | 11,269 | 29/30; 28/30 | 0.8064 / 0.8081 | 0.945/0.926/0.973/0.968/0.964 | 64/37/343/980/153 |

Early stops (patience 5 on val loss): tf-2b-d16-n64 seed 1 stopped after epoch 14 (best epoch 9, val loss 0.763, lr 7.9e-4 at
the best epoch); tf-2b-d32-n64 seed 1 stopped after epoch 12 (best epoch 7, val loss 0.734, lr 8.7e-4). Their checkpoints
cover epochs 0–14 and 0–12. For comparison, tf-2b-d32-n64 seed 0 had val-loss rises at epochs 5, 10, 12–13, 15–16 and
reached val loss 0.671 at epoch 30. A rerun of these two without early stopping would be `--patience 30` (not done).

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
kubectl -n cms-ml exec $POD -- tail -5 /j-jepa-vol/aaron-interp/logs/train-n32-s1-b.log
kubectl -n cms-ml exec $POD -- bash -c 'cd /j-jepa-vol/aaron-interp/models/small; for f in */seed*/test.json; do printf "%-28s %s\n" $f "$(grep -m1 \"accuracy\" $f)"; done'
kubectl -n cms-ml exec $POD -- cut -d, -f1,4-7 /j-jepa-vol/aaron-interp/models/small/tf-1b-d16-n64/seed0/log.csv
```
A failed or missing run is rerun by resubmitting its job with a suffix (`python train/make_jobs.py --only n64-s0 --suffix=-b
--mem 20` — `=` needed, `-b` alone parses as a flag — then `kubectl apply -f train/jobs/n64-s0-b.yaml`); runs with `test.json` are skipped.
