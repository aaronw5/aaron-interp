# Small taggers on the hls4ml jet dataset (training set for the interpretability study)

PyTorch, from scratch, trained as GPU Jobs on the cluster (weaver image, its own torch 1.13.1 / CUDA 11.7). All models see
exactly JEDI-linear's inputs: the first N particles (pT order), features (pT, Δη, Δφ) normalised per feature with
`data/norm_stats.json[N]` (shift/scale), padded rows (raw pT == 0) set to 0 and passed as a mask; 5-class softmax head.

## Files
- `models.py` — the architectures (`tf-{B}b-d{D}`, `linf-{B}b-d{D}-k{K}`, `dsets-d{D}`), attention written out explicitly.
- `train_small.py` — data, training loop, evaluation; one run per invocation.
- `gpu_job.sh` — runs inside the GPU Job: several runs concurrently on one GPU, nvidia-smi / memory / CPU monitor every
  3 min, skip-if-done. Run spec `model:N:seed[:tag[:flags]]`: `tag` is appended to the run directory (`-p30`), `flags` are
  comma-separated `train_small.py` options written with `=` (`--epochs=100,--patience=100`).
- `make_jobs.py` — writes the Job YAMLs to `jobs/` for a named set (`--set main`: one job per (N, seed), the 7 models
  concurrently; `--set extra1`: the patience-30 reruns and the 100-epoch tf-2b-d32 / tf-2b-d64 runs).
- `summarize_small.py` — `collect` (on the cluster: run records → `summary_small.json`) and `render` (Markdown tables and
  the PNG `results/site/T7_small_models.png`, mean ± spread over seeds).
- `STATUS.md` — what was launched and how to check it.

## Models
| name | architecture |
|---|---|
| `tf-{B}b-d{D}` | Linear(3→D) → B pre-LN blocks [2 heads, MLP ×2, GELU] → final LN → masked mean pool → Linear(D→5). No positional encoding. `--cls_token`: read out a learned CLS token instead (off by default). |
| `linf-{B}b-d{D}-k{K}` | as `tf`, but K and V are projected from N tokens to K with one learned (N×K) matrix E, shared across heads and between K and V; padded tokens are zeroed before the projection. |
| `dsets-d{D}` | per-particle MLP 3→D→D (ReLU) → masked sum → MLP D→D→32→16 (ReLU) → Linear(16→5). Float JEDI-like Deep Sets without the context round. |

Parameter counts: tf-2b-d64 67,653; tf-1b-d16 2,405; tf-2b-d16 4,629; tf-1b-d32 8,901; tf-2b-d32 17,445; linf adds N×K per block
(linf-1b-d32-k4 9,029 at N = 32, 9,157 at N = 64; linf-2b-d32-k8 17,957 / 18,469); dsets-d64 11,269 (independent of N).

## Training (defaults of `train_small.py`)
AdamW (wd 0.01), lr 1e-3, 200 warm-up steps then cosine decay to 0 over `--epochs 30`, batch 1024, label smoothing 0,
early stopping on the held-out split (patience 5, on val loss), best = lowest val loss. Split: training = train archive
minus `splits.npz['dev']` (595,000 jets); held-out = `dev` (25,000 balanced); test = the val archive (260,000 jets),
never used for training or selection. The dataset is held on the GPU as float16 and normalised per batch.

Per run (`/j-jepa-vol/aaron-interp/models/small/<model>-n<N>/seed<k>/`):
- `arch.json` — architecture args, normalisation (shift/scale, N), split, training args, parameter count, best epoch
- `log.csv` — per epoch: lr, train/val loss and accuracy, `best` flag, `dead_<layer>` = fraction of units in each
  ReLU/GELU layer whose pre-activation never exceeds 0 on 10,000 fixed held-out jets (valid tokens only); row 0 = init
- `ckpt/epoch_<k>.pt` — state_dict after epoch k; `epoch_0.pt` = initialisation (untrained control)
- `best.pt` — state_dict at the lowest val loss
- `test.json` — test accuracy, per-class accuracy, AUC (one vs rest, score = class probability), background rejection
  1/FPR at 50 % signal efficiency per class, confusion matrix
- `test_probs.npy` — softmax outputs on the test archive, (260000, 5) float32
- `train.log` — stdout of the run

## Run
One run by hand (inside the weaver image; `LD_PRELOAD=/opt/conda/lib/libstdc++.so.6` is needed there):
```
/opt/conda/bin/python3 train/train_small.py --model tf-1b-d16 --n 32 --seed 0 \
    --data_dir /j-jepa-vol/aaron-interp/data --out /j-jepa-vol/aaron-interp/models/small/tf-1b-d16-n32/seed0
```
Smoke test (CPU, 1 epoch, 20k jets): add `--epochs 1 --max_train 20000 --max_test 20000 --device cpu`.

The set, as GPU Jobs (from the repo root):
```
./sync.sh                                   # mirror the repo to /j-jepa-vol/aaron-interp/code
python train/make_jobs.py --set main        # -> train/jobs/n{32,64}-s{0,1}.yaml (run on the helper pod if local compute is off)
kubectl apply -f train/jobs/n32-s0.yaml     # one job first; check utilisation in logs/train-n32-s0.log, then the rest
```
Each job: `gpu_job.sh <job> <concurrency> <model:N:seed>...`; a run whose directory already has `test.json` is skipped,
so a resubmitted job (`make_jobs.py --suffix=-b`) only redoes what is missing. Job names `anrunw-ai-train-<job>`,
labels `owner=anrunw, study=aaron-interp, set=small-taggers`, 1 GPU, limits = requests.

## The exact set launched
- `main`: 7 models × N ∈ {32, 64} × seeds {0, 1} = 28 runs, 4 jobs (`n32-s0`, `n32-s1` → resubmitted as `n32-s1-b`,
  `n64-s0`, `n64-s1`), each the 7 models of one (N, seed) concurrently on one GPU (6 CPU, 16–20 Gi).
- `extra1`: `n64-p30` = tf-2b-d16 and tf-2b-d32 at N = 64, seed 1, `--patience=30` (reruns of the two main runs that
  early-stopped; directories `<model>-n64-p30/seed1`; 2 CPU, 6 Gi); `n64-e100` = tf-2b-d32 and tf-2b-d64 at N = 64, seeds
  0 and 1, `--epochs=100 --patience=100` (directories `<model>-n64-e100/seed<k>`; 4 CPU, 10 Gi).
See `STATUS.md` for job states, measured resource use, results and progress commands.

## Check progress
```
kubectl -n cms-ml get jobs -l set=small-taggers
kubectl -n cms-ml exec <helper-pod> -- tail -5 /j-jepa-vol/aaron-interp/logs/train-n32-s0.log
kubectl -n cms-ml exec <helper-pod> -- cat /j-jepa-vol/aaron-interp/models/small/tf-1b-d16-n32/seed0/log.csv
kubectl -n cms-ml exec <helper-pod> -- bash -c 'for f in /j-jepa-vol/aaron-interp/models/small/*/seed*/test.json; do echo $f; grep -m1 accuracy $f; done'
```
