#!/bin/bash
# X12: runs inside a GPU Job (weaver image + the read-only keras/jax/hgq2 env /j-jepa-vol/phat-jet-aaron/env with jax CUDA 12):
# trains several JEDI-linear n64 models (train/train_hgq_jedi.py) concurrently on the one GPU.
#   gpu_job_x12.sh <job_name> <concurrency> <arm:seed[:tag[:flags]]> [...]
# Each run: python train_hgq_jedi.py --arm <arm> --seed <seed> [flags]  -> $V/models/small/jedi-n64-x12-<arm><tag>/seed<seed>/
# (skipped when that directory already holds test.json). flags: comma-separated, each written with '=' (e.g. --epochs=200).
# Monitor line (GPU utilisation / memory, container memory, CPU cores) every 3 min. Exit 1 if any run lacks test.json.
set -uo pipefail
JOB=$1; CONC=$2; shift 2
V=/j-jepa-vol/aaron-interp
PY=/j-jepa-vol/phat-jet-aaron/env/bin/python
export LD_PRELOAD=/opt/conda/lib/libstdc++.so.6 KERAS_BACKEND=jax JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export HDF5_USE_FILE_LOCKING=FALSE PYTHONDONTWRITEBYTECODE=1 KERAS_HOME=/tmp/keras MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JETDISTILL_DATA=$V/data
SNAP=$V/models/small/_snapshots/$JOB-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$SNAP" && cp $V/code/train/*.py $V/code/train/*.sh "$SNAP"/
echo "[$(date -u +%FT%TZ)] job $JOB on $(hostname); code $(git -C $V/code rev-parse --short HEAD 2>/dev/null || echo none) -> $SNAP"
echo "runs ($CONC concurrent): $*"
nvidia-smi -L
$PY -c "import jax; print('jax devices', jax.devices())" || exit 1
mem_now() { for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory/memory.usage_in_bytes; do
              [ -r $f ] && { echo "$(( $(cat $f) / 1048576 )) MiB"; return; }; done; echo "n/a"; }
cpu_usec() { if [ -r /sys/fs/cgroup/cpu.stat ]; then awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat;
             elif [ -r /sys/fs/cgroup/cpu/cpuacct.usage ]; then echo $(( $(cat /sys/fs/cgroup/cpu/cpuacct.usage) / 1000 )); else echo 0; fi; }
T0=$(date +%s); C0=$(cpu_usec)
( t=$T0; c=$C0; while true; do sleep 180; t1=$(date +%s); c1=$(cpu_usec)
    echo "[$(date -u +%FT%TZ)] monitor: gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) ; mem $(mem_now) ; cpu $(awk -v a=$c -v b=$c1 -v s=$t -v e=$t1 'BEGIN{printf "%.2f", (b-a)/1e6/(e-s)}') cores ; running $(pgrep -fc train_hgq_jedi.py)"
    t=$t1; c=$c1; done ) &
MON=$!
run_one() {
  local spec=$1 arm seed tag flags name out
  IFS=: read -r arm seed tag flags <<< "$spec"
  name=jedi-n64-x12-$arm$tag; out=$V/models/small/$name/seed$seed
  if [ -f "$out/test.json" ]; then echo "[$(date -u +%FT%TZ)] skip $name/seed$seed (test.json exists)"; return 0; fi
  mkdir -p "$out"
  echo "[$(date -u +%FT%TZ)] start $name/seed$seed ${flags//,/ }"
  $PY "$SNAP/train_hgq_jedi.py" --arm "$arm" --seed "$seed" --data_dir $V/data --out "$out" ${flags//,/ } > "$out/train.log" 2>&1
  local rc=$?
  echo "[$(date -u +%FT%TZ)] end $name/seed$seed rc=$rc: $(grep -E '\] test ' "$out/train.log" | tail -1)"
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
  IFS=: read -r arm seed tag flags <<< "$spec"
  [ -f "$V/models/small/jedi-n64-x12-$arm$tag/seed$seed/test.json" ] || { echo "missing result: jedi-n64-x12-$arm$tag/seed$seed"; missing=1; }
done
T1=$(date +%s); C1=$(cpu_usec)
echo "[$(date -u +%FT%TZ)] job $JOB finished, missing=$missing; wall $((T1 - T0))s; average cpu $(awk -v a=$C0 -v b=$C1 -v s=$T0 -v e=$T1 'BEGIN{printf "%.2f", (b-a)/1e6/(e-s)}') cores; mem now $(mem_now)"
exit $missing
