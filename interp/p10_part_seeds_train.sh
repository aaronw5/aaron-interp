#!/bin/bash
# P10 payload (runs inside the GPU job of interp/p10_part_seeds_job.sh): ParT retrained from scratch with the authors' script
# (particle_transformer @ ext/, train_JetClass.sh, weaver-core 0.4.17 from env-part) on the JetClass 2M-train subset of P5
# (/j-jepa-vol/aaron-interp/data/jetclass2m, drawn by interp/p5_lgatr_setup.sh, selection.json: 2 random training files per
# class = 2M jets; 1 random validation file per class = 1M; test file 100 of each class = 1M). Launched one seed per GPU job (2026-10-19:
# three seeds on one 22 GB A10 ran out of GPU memory at the first steps, 48 GB GPUs stayed unavailable for 20 min, and a single
# seed alone kept an A10 85-100 % busy at 2.1 steps/s): interp/p10_part_seeds_job.sh c-p10-<feat>-s<k> 'bash interp/p10_part_seeds_train.sh <feat> <k>'
# with P10_CPU=2 P10_MEM=12 and P10_GPUS = the >= 22 GB products. Several seeds per call still run concurrently on one GPU.
# Settings: train_JetClass.sh unchanged (ParT: batch 512, start lr 1e-3, ranger, AMP, 50 epochs, flat+decay schedule, 2 loader
# workers, fetch step 0.01, streaming from the ROOT files) except (i) data = the 2M subset (the script reads
# ./datasets/JetClass/Pythia/<split>; symlinked here to the subset), (ii) samples per epoch 1,024,000 (train) and 128,000 (val),
# 1/10 of the authors' 10,240,000 / 1,280,000 (authors' total = 512M samples = 256 passes over 2M jets; here 51.2M = 25.6
# passes; the 50-epoch schedule shape is kept), (iii) --model-prefix / --log / --tensorboard redirected, (iv) seeds: weaver
# 0.4.17 has no seed option and torch's default generator seed is a fixed constant, so unseeded runs would start from identical
# weights; a per-seed `weaver` wrapper first on PATH sets random / numpy / torch / cuda seeds to <seed> and then calls
# weaver.train.main (the same entry point as env-part/bin/weaver).
# Outputs: /j-jepa-vol/aaron-interp/models/part_seeds/<feat>_s<seed>/ (net_epoch-<e>_state.pt per epoch, net_best_epoch_state.pt,
# train.log, stdout.log, predict_output/pred.root).
#   interp/p10_part_seeds_train.sh <full|kin> <seed> [<seed> ...]
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 is set (allocator setting only; no effect on the computation).
set -uo pipefail
# The body is one function, parsed completely before it runs: bash reads a plain script incrementally, so a ./sync.sh that
# rewrote this file while a 9 h training was waiting (2026-10-19, header comment lengthened) made the four jobs started before
# that sync resume at a stale byte offset after the final test step ("line 54: port: command not found", "line 56: syntax error
# near unexpected token `)'", exit 2; P11 diagnosis). Training and the final test had completed.
main() {
FEAT=$1; shift
V=/j-jepa-vol/aaron-interp; PT=$V/ext/particle_transformer
mkdir -p $PT/datasets/JetClass/Pythia
for s in train_100M val_5M test_20M; do ln -sfn $V/data/jetclass2m/$s $PT/datasets/JetClass/Pythia/$s; done
echo "[$(date -u +%FT%TZ)] p10 train ParT $FEAT seeds $*; particle_transformer $(git -C $PT rev-parse --short HEAD); weaver $(python -c 'import weaver; print(weaver.__version__)' 2>/dev/null)"
echo "data: $(ls $PT/datasets/JetClass/Pythia/train_100M/ | wc -l) train files -> $(readlink -f $PT/datasets/JetClass/Pythia/train_100M/HToBB_000.root)"
nvidia-smi -L
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128   # limits allocator fragmentation: three trainings share one GPU
( while true; do sleep 300; echo "[$(date -u +%FT%TZ)] monitor: gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader); mem $(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes) / 1048576 )) MiB"; done ) &
MON=$!
cd $PT
PIDS=()
for S in "$@"; do
  OUT=$V/models/part_seeds/${FEAT}_s$S; mkdir -p $OUT/bin
  cat > $OUT/bin/weaver <<'EOW'
#!/bin/bash
exec python -c 'import os, sys, random, numpy as np, torch
s = int(os.environ["P10_SEED"]); random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
print("[p10] seed", s, "torch initial_seed", torch.initial_seed(), flush=True)
sys.argv[0] = "weaver"
from weaver.train import main
sys.exit(main())' "$@"
EOW
  chmod +x $OUT/bin/weaver
  echo "[$(date -u +%FT%TZ)] start ${FEAT}_s$S -> $OUT"
  ( export PATH=$OUT/bin:$PATH P10_SEED=$S
    ./train_JetClass.sh ParT $FEAT --samples-per-epoch 1024000 --samples-per-epoch-val 128000 \
      --model-prefix $OUT/net --log $OUT/train.log --tensorboard p10_${FEAT}_s$S ) > $OUT/stdout.log 2>&1 &
  PIDS+=($!)
  sleep 20
done
rc=0
for p in "${PIDS[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null
# weaver prints one "Test metric" (accuracy) per test file group = per class file (100k jets each): print all ten and their mean
for S in "$@"; do echo "== ${FEAT}_s$S"; grep -E "Running on test file group|Test metric" $V/models/part_seeds/${FEAT}_s$S/train.log \
  | sed -E 's/.*file group ([A-Za-z0-9]+).*/\1/; s/.*Test metric ([0-9.]+).*/\1/' | paste - - \
  | awk '{print; s += $2; n++} END {if (n) printf "mean over %d class files: %.5f\n", n, s / n}'; done
echo "[$(date -u +%FT%TZ)] p10 train done rc=$rc"
return $rc
}
main "$@"; exit $?
