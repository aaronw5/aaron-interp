#!/bin/bash
# X9w: one GPU Job that runs the Deep Sets runs (train/gpu_job.sh, torch) and the JEDI-linear runs (train/gpu_job_x12.sh, keras/jax)
# side by side on the one GPU, so both sets share the GPU instead of two GPU Jobs.
#   gpu_job_x9w.sh <job_name> <concurrency> <spec> [...]
# spec starting with "dsets-" -> gpu_job.sh run spec model:N:seed[:tag[:flags]]; otherwise -> gpu_job_x12.sh spec arm:seed[:tag[:flags]].
# <concurrency> applies to each of the two sets. Logs: $V/logs/train-<job>-dsets.log and train-<job>-jedi.log (plus this one).
set -uo pipefail
JOB=$1; CONC=$2; shift 2
V=/j-jepa-vol/aaron-interp; D=(); J=()
for s in "$@"; do case "$s" in dsets-*) D+=("$s");; *) J+=("$s");; esac; done
echo "[$(date -u +%FT%TZ)] $JOB: dsets runs ${D[*]:-none}; jedi runs ${J[*]:-none}"
P=()
if [ ${#D[@]} -gt 0 ]; then bash $V/code/train/gpu_job.sh "$JOB-dsets" "$CONC" "${D[@]}" > $V/logs/train-$JOB-dsets.log 2>&1 & P+=($!); fi
if [ ${#J[@]} -gt 0 ]; then bash $V/code/train/gpu_job_x12.sh "$JOB-jedi" "$CONC" "${J[@]}" > $V/logs/train-$JOB-jedi.log 2>&1 & P+=($!); fi
rc=0; for p in "${P[@]}"; do wait "$p" || rc=1; done
echo "[$(date -u +%FT%TZ)] $JOB finished rc=$rc"; exit $rc
