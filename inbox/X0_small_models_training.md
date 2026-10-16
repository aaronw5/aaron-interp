# X0 — small taggers trained on hls4ml (agent report, 2026-10-16)

Code: train/models.py (native transformer tf-{1,2}b-d{16,32}: per-particle linear embedding → pre-LN blocks, 2 heads, MLP ×2, GELU → masked
mean pooling → linear head; Linformer linf-1b-d32-k4, linf-2b-d32-k8; Deep Sets dsets-d64: 3→64→64 ReLU → sum → 64→64→32→16 ReLU → 5),
train/train_small.py, train/gpu_job.sh (several runs per GPU; nvidia-smi + cgroup monitor every 3 min), train/make_jobs.py, README.md, STATUS.md.
Inputs exactly JEDI-linear's: first N pT-ranked particles, (pT, Δη, Δφ) normalised with norm_stats.json[N], padded rows zero, mask passed.
Split: train = train archive minus the dev jets of splits.npz (595,000); val = dev (25,000); test = val archive (260,000). AdamW, lr 1e-3 cosine,
batch 1024, 30 epochs; per-epoch checkpoints epoch_0 (init) … epoch_30, best.pt, log.csv with dead-unit fractions per layer, test.json.
Jobs: anrunw-ai-train-n32-s0 (done), -n32-s1, -n64-s0, -n64-s1 (7 models each on one GPU; 6 CPU / 16–20 Gi). n32-s0 on an L40: GPU 100 %,
container memory 15.0 GiB, 5 min of training.

Test accuracy, N = 32, seed 0 (260,000 jets): tf-1b-d16 68.08 %, tf-1b-d32 71.83, linf-1b-d32-k4 74.16, tf-2b-d16 75.24, tf-2b-d32 77.24,
dsets-d64 77.78, linf-2b-d32-k8 78.41 (JEDI-linear n32: 78.0 official). No run early-stopped; best epochs 29–30 (still improving at the end of
the cosine schedule → a longer schedule is needed for final models). Dead units: 0 in the GELU layers; dsets ReLU layers 1.6 / 9.4 / 21.9 / 15.6 /
18.8 % from epoch 1 on. tf-1b-d16 val loss/acc: init 1.678/0.201 → ep1 1.057/0.593 → ep10 0.922/0.655 → ep30 0.871/0.684.
Outputs: /j-jepa-vol/aaron-interp/models/small/<model>-n<N>/seed<k>/; logs /j-jepa-vol/aaron-interp/logs/train-<job>.log.
Issue: sync.sh failed once with tar "Cannot open: File exists" on a read-only .git object (fixed: --unlink-first).
