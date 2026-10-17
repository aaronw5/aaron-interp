# X0b — small taggers, N = 64 results and job status (agent report, 2026-10-17)
n64-s0 and n64-s1 complete (7/7 each) on RTX 2080 Ti: 21 / 16 min, GPU 98–100 %, memory 12.7–12.8 of 20 GiB, CPU 5.0–5.1 of 6 cores.
n32-s1 failed before its container started (BackoffLimitExceeded, backoffLimit 0; cause not recorded, pod gone); relaunched as
anrunw-ai-train-n32-s1-b with backoffLimit 1 (runs with test.json are skipped on retry).
Test accuracy on 260,000 jets, N = 64, seed 0 / seed 1: tf-1b-d16 .6931/.6921; tf-2b-d16 .7384/.7239 (s1 early-stopped at epoch 14, best 9);
tf-1b-d32 .6938/.6915; tf-2b-d32 .7619/.7342 (s1 early-stopped at 12, best 7); linf-1b-d32-k4 .7669/.7778; linf-2b-d32-k8 .8106/.8076;
dsets-d64 .8064/.8081. JEDI-linear 3-feature n64: 80.94 %. All other runs' best epoch 28–30 (still improving at the end of the schedule).
Requested next: early-stopped runs rerun with patience 30; tf-2b-d32 and tf-2b-d64 at N = 64 for 100 epochs (2 seeds); full table PNG.
