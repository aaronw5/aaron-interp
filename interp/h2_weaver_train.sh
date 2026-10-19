#!/bin/bash
# H2: train ParT / ParticleNet from scratch with the authors' scripts (particle_transformer repo, train_TopLandscape.sh /
# train_QuarkGluon.sh, weaver-core 0.4.17 from env-part) inside a GPU job (interp/h2_gpu_job.sh). Several feature types run
# concurrently on the one GPU. Outputs: /j-jepa-vol/aaron-interp/models/<top|qg>/<model>_<feat>/ (net_best_epoch_state.pt,
# net_epoch-*_state.pt, train.log, predict_output/pred.root). The authors' settings are kept (20 epochs, batch 512, ranger,
# in-memory); only --model-prefix, --log, --tensorboard are redirected.
#   interp/h2_weaver_train.sh <top|qg> <ParT|PN> <feat> [<feat> ...]      (feat: kin for top; kin, kinpid, kinpidplus for qg)
set -uo pipefail
DS=$1; MODEL=$2; shift 2
V=/j-jepa-vol/aaron-interp; PT=$V/ext/particle_transformer
mkdir -p $PT/datasets/TopLandscape $PT/datasets/QuarkGluon
for s in train val test; do ln -sf /j-jepa-vol/ParT_top/TopLandscape/${s}_file.parquet $PT/datasets/TopLandscape/${s}_file.parquet; done
for f in /j-jepa-vol/QuarkGluon/train/train_file_*.parquet /j-jepa-vol/QuarkGluon/test/test_file_*.parquet; do
  ln -sf $f $PT/datasets/QuarkGluon/$(basename $f); done
echo "[$(date -u +%FT%TZ)] h2_weaver_train $DS $MODEL $*; particle_transformer $(git -C $PT rev-parse --short HEAD); weaver $(python -c 'import weaver; print(weaver.__version__)' 2>/dev/null) $(which weaver)"
nvidia-smi -L
( while true; do sleep 300; echo "[$(date -u +%FT%TZ)] monitor: gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader); mem $(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes) / 1048576 )) MiB"; done ) &
MON=$!
cd $PT
PIDS=()
for FEAT in "$@"; do
  name=$( [ "$MODEL" = PN ] && echo pnet || echo part )_$FEAT
  OUT=$V/models/$DS/$name; mkdir -p $OUT
  if [ "$DS" = top ]; then S=./train_TopLandscape.sh; else S=./train_QuarkGluon.sh; fi
  echo "[$(date -u +%FT%TZ)] start $name -> $OUT"
  $S $MODEL $FEAT --model-prefix $OUT/net --log $OUT/train.log --tensorboard h2_${DS}_${name} > $OUT/stdout.log 2>&1 &
  PIDS+=($!)
done
rc=0
for p in "${PIDS[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null
for FEAT in "$@"; do name=$( [ "$MODEL" = PN ] && echo pnet || echo part )_$FEAT
  echo "== $name"; grep -E "Test metric|test.*acc|roc_auc|Best epoch" $V/models/$DS/$name/train.log | tail -5; done
echo "[$(date -u +%FT%TZ)] h2_weaver_train done rc=$rc"
exit $rc
