# Cluster rules (NRP Nautilus, https://nrp.ai — user 2026-10-16: "follow these rules")

Source: nrp.ai/documentation/userdocs/start/policies and running/jobs (read 2026-10-16). Namespace cms-ml, PVC j-jepa-vol.

- Batch work runs as Jobs with a command that ends by itself. No `sleep infinity`, no script ending in sleep (bannable).
- Interactive pods without a controller are destroyed after 6 h (limits 2 GPU, 32 GB, 16 cores). Deployments are removed after 2 weeks
  unless on the exceptions list; deployments cannot request a GPU.
- Requests vs limits: CPU, memory and ephemeral-storage limits within 20 % of requests; with > ~100 pods/jobs, limit = request. We set
  limit = request always.
- Utilisation thresholds (a pod violates if): GPU utilisation ≤ 40 % of requested GPUs; CPU use outside 20–200 % of request; memory use
  outside 20–150 % of request. At most 4 violating pods per user. Pods requesting exactly 1 CPU and 2 GB memory are exempt.
  → Sizing policy here: light CPU experiments (JAX JEDI runs, probes) request exactly 1 CPU / 2Gi (exempt); anything larger is sized to
  measured use (memory request ≈ 0.8 × peak; CPU request ≈ threads used). GPU jobs must keep the GPU above 40 % busy: tiny models are
  trained several at a time in one job (parallel processes on the one GPU), not one tiny model per GPU.
- Requested GPUs are unavailable to others while the pod runs: request a GPU only when the job starts using it at once. > 50 GPUs needs
  a plan via support. A100 quota is zero by default.
- No fair queue: do not submit large batches at once; keep the number of concurrent jobs small (≤ ~10) unless using a work queue.
- Storage: not archival; purge data no longer used; volumes untouched for 6 months may be purged. Keep /j-jepa-vol/aaron-interp tidy
  (results JSON/PNG only; no duplicate datasets — read other datasets in place).
- Check the portal's Violations page when in doubt.
