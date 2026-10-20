#!/bin/bash
# X9w: several interp.x9_train runs (energyflow PFN / EFN, isolated venv) concurrently in one CPU Job (run through ./job.sh).
#   bash train/x9w_cpu_batch.sh <threads_per_run> <concurrency> "<x9_train args>" ["<x9_train args>" ...]
# Each run's log: /j-jepa-vol/aaron-interp/logs/x9w-train-<tag>.log (tag from the --arch/--seed/--tag args). Exit 1 if any run failed.
set -uo pipefail
TH=$1; CONC=$2; shift 2
V=/j-jepa-vol/aaron-interp
[ -x /tmp/venv-ef/bin/python ] || tar xf $V/venv-ef.tar -C /tmp || exit 1
export OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH MKL_NUM_THREADS=$TH
rc=0; P=()
for args in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n || rc=1; done
  name=$(echo "$args" | sed -E 's/.*--arch ([a-z]+).*--seed ([0-9]+).*--tag[= ]([^ ]+).*/\1-s\2\3/')
  echo "[$(date -u +%FT%TZ)] start $name: $args"
  ( /tmp/venv-ef/bin/python -m interp.x9_train $args > $V/logs/x9w-train-$name.log 2>&1; r=$?
    echo "[$(date -u +%FT%TZ)] end $name rc=$r: $(grep 'test acc' $V/logs/x9w-train-$name.log | tail -1 | cut -c1-160)"; exit $r ) &
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || rc=1; done
echo "[$(date -u +%FT%TZ)] all runs finished rc=$rc"; exit $rc
