#!/bin/bash
# H39: EFN / PFN variants for one dataset, four runs concurrently in one CPU job (3 threads each):
# efn s1, efn shuf, pfn s1, pfn shuf (interp/h39_train_ef.py). Per-run logs: /j-jepa-vol/aaron-interp/logs/h39-ef-<ds>-<arch>-<var>.log
#   ./job.sh d-h39-<ds>-ef 12 36 'bash interp/h39_ef_jobs.sh <ds>'
#   bash interp/h39_ef_jobs.sh init      (epoch-0 EFN / PFN, seeds 0-5, no data; 1 CPU)
set -uo pipefail
DS=$1; V=/j-jepa-vol/aaron-interp; PY=$V/venv-ef/bin/python
if [ "$DS" = init ]; then
  for arch in efn pfn; do for s in 0 1 2 3 4 5; do OMP_NUM_THREADS=1 $PY -m interp.h39_train_ef --arch $arch --variant init --seed $s || exit 1; done; done
  exit 0
fi
PIDS=()
for arch in efn pfn; do for var in s1 shuf; do
  L=$V/logs/h39-ef-$DS-$arch-$var.log
  echo "[$(date -u +%FT%TZ)] start $DS $arch $var -> $L"
  OMP_NUM_THREADS=3 $PY -m interp.h39_train_ef --arch $arch --ds $DS --variant $var --seed 1 > $L 2>&1 &
  PIDS+=($!)
done; done
( while true; do sleep 600; echo "[$(date -u +%FT%TZ)] monitor: mem $(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0) / 1048576 )) MiB; $(for f in $V/logs/h39-ef-$DS-*.log; do echo -n "$(basename $f .log): $(grep -c epoch $f) ep; "; done)"; done ) &
MON=$!
rc=0; for p in "${PIDS[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null
for f in $V/logs/h39-ef-$DS-*.log; do echo "== $f"; tail -2 $f; done
echo "[$(date -u +%FT%TZ)] h39_ef_jobs $DS done rc=$rc"; exit $rc
