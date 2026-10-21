#!/bin/bash
# H57 copy of job.sh with hcc-nrp-shor-c6021.unl.edu excluded (Pythia ran ~8x slower there, 2026-10-21)
# Run a CPU-only command on the cluster as a Job. The command runs in /j-jepa-vol/aaron-interp/code (run ./sync.sh
# first) with the read-only python env /j-jepa-vol/phat-jet-aaron/env first on PATH (keras 3.12, jax 0.6.2 cpu,
# hgq2 0.1.9, scikit-learn 1.7, matplotlib 3.10, h5py, pandas; no bytecode written) and the data / model / result paths
# set. AI_PY=conda ./job.sh ... uses the image's /opt/conda python instead (torch 1.13, uproot, awkward, weaver-core).
#   ./job.sh <name> '<command>'                    1 CPU, 2 GiB (the default; NRP-exempt size, use it for light runs)
#   ./job.sh <name> <cpu> <mem_gi> '<command>'     a larger request: only when measured use needs it (see below)
# e.g. ./job.sh e1e3-3f-n64 'python -m interp.e1_e3_sweep --family 3-feature --n 64'
# Job anrunw-ai-<name> (labels owner=anrunw, study=aaron-interp); log /j-jepa-vol/aaron-interp/logs/<name>.log
# NRP rules (nrp.ai policies): requests == limits; CPU use must stay within 20-200% of the request and memory within
# 20-150%; pods requesting exactly 1 CPU and 2 GB are exempt; at most 4 violating pods per user. Size any larger request
# to the measured use (kubectl top pod while it runs, or prefix the command with /usr/bin/time -v for peak memory).
# Measured 2026-10-16, interp.e1_e3_sweep 3-feature n64 (20k-jet sample + 260k-jet accuracy): ~1.6 CPU, ~1.2 GiB.
set -euo pipefail
if [ $# -eq 2 ]; then NAME=$1; CPU=1; MEM=2; CMD=$2
elif [ $# -eq 4 ]; then NAME=$1; CPU=$2; MEM=$3; CMD=$4
else sed -n 2,17p "$0"; exit 1; fi
PYENV=${AI_PY:-jax}   # AI_PY=conda: the image's /opt/conda python (torch 1.13, uproot, awkward, weaver) instead of the jax env
V=/j-jepa-vol/aaron-interp
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: anrunw-ai-$NAME
  namespace: cms-ml
  labels: {owner: anrunw, study: aaron-interp}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: {owner: anrunw, study: aaron-interp}
    spec:
      restartPolicy: Never
      activeDeadlineSeconds: 86400
      affinity:  # nodes where CephFS metadata ops stalled (D state in ceph_mdsc_wait_request), 2026-10-08; gpu-09.nrp.mghpcc.org 2026-10-20 (E26); k8s-gpu-2.ucsc.edu 2026-10-21 (E26w); ucm-fiona01.ucmerced.edu, k8s-haosu-12.sdsc.optiputer.net, rci-tide-gpu-04.sdsu.edu 2026-10-21 (E31)
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [hcc-nrp-shor-c6021.unl.edu, k8s-haosu-24.sdsc.optiputer.net, ry-gpu-01.sdsc.optiputer.net, nautilus-it-gpu05.fullerton.edu, fiona-prg1.cesnet.cz, k8s-chase-ci-07.calit2.optiputer.net, dtn-gpu2.kreonet.net, gpu-09.nrp.mghpcc.org, k8s-gpu-2.ucsc.edu, ucm-fiona01.ucmerced.edu, k8s-haosu-12.sdsc.optiputer.net, rci-tide-gpu-04.sdsu.edu]
      containers:
      - name: cpu
        image: gitlab-registry.nrp-nautilus.io/jmduarte/weaver:latest
        command: ["/bin/bash", "-c"]
        args:
        - |
          export LD_PRELOAD=/opt/conda/lib/libstdc++.so.6 KERAS_BACKEND=jax JAX_PLATFORMS=cpu HDF5_USE_FILE_LOCKING=FALSE
          export PYTHONDONTWRITEBYTECODE=1 KERAS_HOME=/tmp/keras MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/cache
          export OMP_NUM_THREADS=$CPU OPENBLAS_NUM_THREADS=$CPU MKL_NUM_THREADS=$CPU
          [ "$PYENV" = conda ] || export PATH=/j-jepa-vol/phat-jet-aaron/env/bin:\$PATH
          export JETDISTILL_DATA=$V/data JETDISTILL_MODELS=$V/models/official_models JETDISTILL_RESULTS=$V/data AI_RESULTS=$V/results
          cd $V/code
          mkdir -p $V/logs
          t0=\$(date +%s)
          ( echo "[\$(date -u +%FT%TZ)] \$(hostname) git \$(git rev-parse --short HEAD 2>/dev/null || echo none); request $CPU CPU, ${MEM}Gi"
            $CMD; rc=\$?; w=\$(( \$(date +%s) - t0 ))
            mem=\$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null || echo 0)
            cpu=\$(awk '/^usage_usec/ {print \$2/1e6}' /sys/fs/cgroup/cpu.stat 2>/dev/null || awk '{print \$1/1e9}' /sys/fs/cgroup/cpuacct/cpuacct.usage 2>/dev/null || echo 0)
            echo "[\$(date -u +%FT%TZ)] exit \$rc; wall \${w}s; pod peak memory \$(( mem / 1048576 )) MiB (request ${MEM}Gi); CPU \${cpu}s = \$(awk -v c=\$cpu -v w=\$w 'BEGIN{printf "%.2f", c/(w>0?w:1)}') cores on average (request $CPU)"
            exit \$rc ) > $V/logs/$NAME.log 2>&1
        resources:
          limits: {cpu: "$CPU", memory: ${MEM}Gi}
          requests: {cpu: "$CPU", memory: ${MEM}Gi}
        volumeMounts:
        - {name: vol, mountPath: /j-jepa-vol}
      volumes:
      - name: vol
        persistentVolumeClaim: {claimName: j-jepa-vol}
EOF
