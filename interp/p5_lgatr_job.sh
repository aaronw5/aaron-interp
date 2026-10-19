#!/bin/bash
# P5: L-GATr training on the JetClass 2M-train subset as a GPU Job (NRP: requests == limits; GPU must stay > 40 % busy).
# Authors' code and paper config (heidelberg-hepml/lorentz-gatr @ 35aca2a, config_paper/jctagging.yaml + default_tagging.yaml +
# model/gatr_tagging.yaml: 12 blocks, 16 mv / 32 s hidden channels, 8 heads, Lion lr 3e-4, cosine schedule, batch 128,
# 200,000 iterations, validation every 5,000, beam reference xyplane (two beams, beam token) + time reference, scalar features
# on). Overrides (all data / bookkeeping): data.data_dir -> $V/data/jetclass2m (see interp/p5_lgatr_setup.sh), file ranges
# train [0,2] (2 random files/class = 2M jets), val [120,121] (1 random file/class), test [100,101]; base_dir / mlflow paths ->
# $V/models/lgatr; run_name, seed 0; save_source false. Code patch interp/p5_lgatr.patch (applied by the job): the JetClass
# loaders drop sampler=DistributedSampler (torch's DataLoader rejects a sampler for an IterableDataset; the authors' later
# lloca-experiments repo builds them the same way) and cycle() calls sampler.set_epoch only if it exists. Prerequisite: interp/p5_lgatr_setup.sh (env-lgatr, ext/, data).
#   interp/p5_lgatr_job.sh <name> <run_name> [extra hydra overrides]
# e.g. interp/p5_lgatr_job.sh c-p5-lgatr jc2m_s0
# Job anrunw-ai-<name>; log $V/logs/<name>.log; checkpoints $V/models/lgatr/runs/jc2m/<run_name>/models/
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,12p "$0"; exit 1; }
NAME=$1; RUN=$2; shift 2; EXTRA="$*"; CPU=4; MEM=16
V=/j-jepa-vol/aaron-interp
OVR="data.data_dir=$V/data/jetclass2m data.train_files_range=[0,2] data.val_files_range=[120,121] data.test_files_range=[100,101] \
exp_name=jc2m run_name=$RUN seed=0 base_dir=$V/models/lgatr save_source=false \
mlflow.db=$V/models/lgatr/runs/jc2m/mlflow/mlflow.db mlflow.artifacts=$V/models/lgatr/runs/jc2m/mlflow/artifacts $EXTRA"
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
      activeDeadlineSeconds: 259200
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [k8s-haosu-24.sdsc.optiputer.net, ry-gpu-01.sdsc.optiputer.net, nautilus-it-gpu05.fullerton.edu, fiona-prg1.cesnet.cz, k8s-chase-ci-07.calit2.optiputer.net, dtn-gpu2.kreonet.net]
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-A10, NVIDIA-L40, NVIDIA-A40, NVIDIA-RTX-A6000, NVIDIA-GeForce-RTX-3090, NVIDIA-L4, NVIDIA-RTX-A5000, NVIDIA-GeForce-RTX-4090, NVIDIA-L40S, NVIDIA-GeForce-RTX-2080-Ti, Tesla-V100-SXM2-32GB, NVIDIA-TITAN-RTX, Quadro-RTX-6000]
      containers:
      - name: gpu
        image: gitlab-registry.nrp-nautilus.io/jmduarte/weaver:latest
        command: ["/bin/bash", "-c"]
        args:
        - |
          export HDF5_USE_FILE_LOCKING=FALSE PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/cache
          export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
          mkdir -p $V/logs $V/models/lgatr/runs/jc2m/mlflow/artifacts
          cd $V/ext/lorentz-gatr && git checkout -q -- . && git apply $V/code/interp/p5_lgatr.patch
          t0=\$(date +%s)
          ( echo "[\$(date -u +%FT%TZ)] \$(hostname) lorentz-gatr \$(git rev-parse --short HEAD) python $V/env-lgatr/bin/python \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
            echo "overrides: $OVR"; echo "patch: \$(git diff --stat | tail -1)"
            ( while sleep 600; do echo "[gpu \$(date -u +%FT%TZ)] \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
            $V/env-lgatr/bin/python run.py -cp config_paper -cn jctagging $OVR; rc=\$?
            mem=\$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null || echo 0)
            echo "[\$(date -u +%FT%TZ)] exit \$rc; wall \$(( \$(date +%s) - t0 ))s; pod peak memory \$(( mem / 1048576 )) MiB (request ${MEM}Gi)"
            kill %1 2>/dev/null; exit \$rc ) > $V/logs/$NAME.log 2>&1
        resources:
          limits: {cpu: "$CPU", memory: ${MEM}Gi, nvidia.com/gpu: 1}
          requests: {cpu: "$CPU", memory: ${MEM}Gi, nvidia.com/gpu: 1}
        volumeMounts:
        - {name: vol, mountPath: /j-jepa-vol}
      volumes:
      - name: vol
        persistentVolumeClaim: {claimName: j-jepa-vol}
EOF
