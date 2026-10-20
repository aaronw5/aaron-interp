#!/bin/bash
# H14: train ParT / ParticleNet on the Herwig 7.1 quark-gluon set (Zenodo 3066475, 20 x 100k jets) with the authors' script
# (particle_transformer train_QuarkGluon.sh, same weaver settings as H2: 20 epochs, batch 512, ranger, in-memory,
# --train-val-split 0.8889, 1.6M samples per epoch). Data: parquet written by interp.h14_qg_generator --step hwdata in
# /j-jepa-vol/aaron-interp/data/qg/herwig/parquet (train_file_0..17 = herwig files 0..17 -> 1.6M train + 200k val;
# test_file_0..1 = herwig files 18..19, 200k test), passed by overriding --data-train / --data-test (env.sh blanks
# DATADIR_QuarkGluon). Outputs /j-jepa-vol/aaron-interp/models/qg_herwig/<part|pnet>_<feat>/.
#   interp/h14_hw_train.sh <ParT|PN> <feat> [<feat> ...]       run inside interp/h2_gpu_job.sh
set -uo pipefail
MODEL=$1; shift 1
V=/j-jepa-vol/aaron-interp; PT=$V/ext/particle_transformer; HWP=$V/data/qg/herwig/parquet
[ -f $HWP/done ] || { echo "no Herwig parquet ($HWP/done missing)"; exit 1; }
echo "[$(date -u +%FT%TZ)] h14_hw_train $MODEL $*; particle_transformer $(git -C $PT rev-parse --short HEAD); weaver $(python -c 'import weaver; print(weaver.__version__)' 2>/dev/null)"
nvidia-smi -L
( while true; do sleep 300; echo "[$(date -u +%FT%TZ)] monitor: gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader); mem $(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes) / 1048576 )) MiB"; done ) &
MON=$!
cd $PT
PIDS=()
for FEAT in "$@"; do
  name=$( [ "$MODEL" = PN ] && echo pnet || echo part )_$FEAT
  OUT=$V/models/qg_herwig/$name; mkdir -p $OUT
  echo "[$(date -u +%FT%TZ)] start $name -> $OUT"
  ./train_QuarkGluon.sh $MODEL $FEAT --data-train "$HWP/train_file_*.parquet" --data-test "$HWP/test_file_*.parquet" \
    --model-prefix $OUT/net --log $OUT/train.log --tensorboard h14_hw_${name} > $OUT/stdout.log 2>&1 &
  PIDS+=($!)
done
rc=0
for p in "${PIDS[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null
for FEAT in "$@"; do name=$( [ "$MODEL" = PN ] && echo pnet || echo part )_$FEAT
  echo "== $name"; grep -E "Test metric|Best epoch" $V/models/qg_herwig/$name/train.log | tail -3; done
echo "[$(date -u +%FT%TZ)] h14_hw_train done rc=$rc"
exit $rc
