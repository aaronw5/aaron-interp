#!/bin/bash
# Run an interp script against ParT (Particle Transformer, JetClass) with jld-features' venv (read-only) on the Apple GPU.
#   ./run_part.sh interp/p0_smoke_part.py [args]            (log: write it yourself, e.g. > logs/p0_smoke.log 2>&1)
# One heavy MPS process at a time: another session may use the GPU. If the log shows "command buffer" errors or ParT_full's
# accuracy on unedited jets is not ≈ 86 % on its sample (G192: 86.2 % on 400 jets/class, rows 3,000–3,400), rerun alone.
set -euo pipefail
cd "$(dirname "$0")"
export JETDISTILL_TAGGER=part KERAS_BACKEND=jax JETDISTILL_DEVICE="${JETDISTILL_DEVICE:-mps}"
export JETDISTILL_DATA=/Users/anrunw/Documents/distill-part/jet-latent-distillation/data_part
export JETDISTILL_MODELS=/Users/anrunw/Documents/distill-part/jet-latent-distillation/models_part
export AI_RESULTS=results PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
exec /Users/anrunw/Documents/distill-part/jld-features/.venv/bin/python "$@"
