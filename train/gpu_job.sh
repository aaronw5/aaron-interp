#!/bin/bash
# Runs inside a GPU Job (weaver image): trains several small taggers concurrently on the one GPU.
#   gpu_job.sh <job_name> <concurrency> <model:N:seed[:tag[:flags]]> [...]
# Each run: python train_small.py --model <model> --n <N> --seed <seed> [flags]
#   -> $V/models/small/<model>-n<N><tag>/seed<seed>/   (skipped when that directory already holds test.json)
# tag: optional directory suffix (e.g. -p30) so variants do not overwrite the base runs; flags: optional train_small.py
# options, comma-separated, each written with '=' (e.g. --epochs=100,--patience=100). At most <concurrency> runs train at once (NRP: the GPU must
# stay > 40 % busy; one tiny model does not do that). A monitor prints nvidia-smi utilisation and the container's
# memory use every 3 min into the job log. Extra train_small.py flags can be passed in $TRAIN_EXTRA.
# Exit status 1 if any run ended without test.json.
set -uo pipefail
JOB=$1; CONC=$2; shift 2
V=/j-jepa-vol/aaron-interp
PY=/opt/conda/bin/python3
export LD_PRELOAD=/opt/conda/lib/libstdc++.so.6 HDF5_USE_FILE_LOCKING=FALSE PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TRAIN_EXTRA=${TRAIN_EXTRA:-}

# snapshot the code this job ran (and keep running it from the snapshot, so later edits in code/train do not matter)
SNAP=$V/models/small/_snapshots/$JOB-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$SNAP" && cp $V/code/train/*.py $V/code/train/gpu_job.sh "$SNAP"/
echo "[$(date -u +%FT%TZ)] job $JOB on $(hostname); code $(git -C $V/code rev-parse --short HEAD 2>/dev/null || echo none) -> $SNAP"
echo "runs ($CONC concurrent): $*"
nvidia-smi -L
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1

# monitor: GPU utilisation / memory, cgroup memory and average CPU cores of this container since the last line
mem_now() { for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.usage_in_bytes; do
              [ -r $f ] && { echo "$(( $(cat $f) / 1048576 )) MiB"; return; }; done; echo "n/a"; }
cpu_usec() { if [ -r /sys/fs/cgroup/cpu.stat ]; then awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat;
             elif [ -r /sys/fs/cgroup/cpu/cpuacct.usage ]; then echo $(( $(cat /sys/fs/cgroup/cpu/cpuacct.usage) / 1000 )); else echo 0; fi; }
T0=$(date +%s); C0=$(cpu_usec)
( t=$T0; c=$C0; while true; do sleep 180; t1=$(date +%s); c1=$(cpu_usec)
    echo "[$(date -u +%FT%TZ)] monitor: gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) ; mem $(mem_now) ; cpu $(awk -v a=$c -v b=$c1 -v s=$t -v e=$t1 'BEGIN{printf "%.2f", (b-a)/1e6/(e-s)}') cores ; running $(pgrep -fc train_small.py)"
    t=$t1; c=$c1; done ) &
MON=$!

run_one() {
  local spec=$1 model n seed tag flags name out
  IFS=: read -r model n seed tag flags <<< "$spec"
  name=$model-n$n$tag; out=$V/models/small/$name/seed$seed
  if [ -f "$out/test.json" ]; then echo "[$(date -u +%FT%TZ)] skip $name/seed$seed (test.json exists)"; return 0; fi
  mkdir -p "$out"
  echo "[$(date -u +%FT%TZ)] start $name/seed$seed ${flags//,/ }"
  $PY "$SNAP/train_small.py" --model "$model" --n "$n" --seed "$seed" --data_dir $V/data --out "$out" ${flags//,/ } $TRAIN_EXTRA > "$out/train.log" 2>&1
  local rc=$?
  echo "[$(date -u +%FT%TZ)] end $name/seed$seed rc=$rc: $(grep -E '^\[.*\] test ' "$out/train.log" | tail -1)"
  return $rc
}

PIDS=()
for spec in "$@"; do
  while [ "$(jobs -rp | grep -vc "^$MON$")" -ge "$CONC" ]; do wait -n; done
  run_one "$spec" &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
kill $MON 2>/dev/null

missing=0
for spec in "$@"; do
  IFS=: read -r model n seed tag flags <<< "$spec"
  [ -f "$V/models/small/$model-n$n$tag/seed$seed/test.json" ] || { echo "missing result: $model-n$n$tag/seed$seed"; missing=1; }
done
T1=$(date +%s); C1=$(cpu_usec)
echo "[$(date -u +%FT%TZ)] job $JOB finished, missing=$missing; wall $((T1 - T0))s; average cpu $(awk -v a=$C0 -v b=$C1 -v s=$T0 -v e=$T1 'BEGIN{printf "%.2f", (b-a)/1e6/(e-s)}') cores; mem now $(mem_now)"
exit $missing
