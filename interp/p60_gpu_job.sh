#!/bin/bash
# P60: copy of ./part_job.sh (same env-part venv, paths, 4 CPU / 16 Gi / 1 GPU) with two scheduling changes after job c-p60-run
# failed on hcc-nrp-shor-c6017.unl.edu (NVIDIA A10) with "CUDA error: an illegal memory access" in the first L-GATr (xformers)
# batch: that node is excluded and the GPU product is restricted to the list the L-GATr training jobs use, without NVIDIA-A10.
#   interp/p60_gpu_job.sh <name> '<command>'
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,6p "$0"; exit 1; }
NAME=$1; CMD=$2; CPU=4; MEM=16
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
      activeDeadlineSeconds: 43200
      affinity:  # nodes where CephFS metadata ops stalled (D state in ceph_mdsc_wait_request), 2026-10-08
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [k8s-haosu-24.sdsc.optiputer.net, ry-gpu-01.sdsc.optiputer.net, nautilus-it-gpu05.fullerton.edu, fiona-prg1.cesnet.cz, k8s-chase-ci-07.calit2.optiputer.net, dtn-gpu2.kreonet.net, k8s-haosu-06.sdsc.optiputer.net, fiona8.ucsc.edu, k8s-haosu-22.sdsc.optiputer.net, k8s-haosu-09.sdsc.optiputer.net, k8s-chase-ci-01.calit2.optiputer.net, hcc-nrp-shor-c6017.unl.edu]
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-L40, NVIDIA-A40, NVIDIA-RTX-A6000, NVIDIA-GeForce-RTX-3090, NVIDIA-L4, NVIDIA-RTX-A5000, NVIDIA-GeForce-RTX-4090, NVIDIA-L40S, NVIDIA-GeForce-RTX-2080-Ti, Tesla-V100-SXM2-32GB, NVIDIA-TITAN-RTX, Quadro-RTX-6000]
      containers:
      - name: gpu
        image: gitlab-registry.nrp-nautilus.io/jmduarte/weaver:latest
        command: ["/bin/bash", "-c"]
        args:
        - |
          export LD_PRELOAD=/opt/conda/lib/libstdc++.so.6 HDF5_USE_FILE_LOCKING=FALSE
          export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/cache
          export OMP_NUM_THREADS=$CPU OPENBLAS_NUM_THREADS=$CPU MKL_NUM_THREADS=$CPU
          [ -x $V/env-part/bin/python ] || { /opt/conda/bin/python3 -m venv --system-site-packages $V/env-part && $V/env-part/bin/pip install --no-cache-dir weaver-core==0.4.17; }
          export PATH=$V/env-part/bin:\$PATH
          export JETDISTILL_TAGGER=part JETDISTILL_DEVICE=cuda
          export JETDISTILL_DATA=$V/code/cluster/data_part JETDISTILL_MODELS=$V/models/part JETDISTILL_RESULTS=$V/results/part_pipeline AI_RESULTS=$V/results
          cd $V/code
          mkdir -p $V/logs
          ( echo "[\$(date -u +%FT%TZ)] \$(hostname) git \$(git rev-parse --short HEAD 2>/dev/null || echo none) python \$(which python) \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; $CMD; rc=\$?; echo "[\$(date -u +%FT%TZ)] exit \$rc"; exit \$rc ) > $V/logs/$NAME.log 2>&1
        resources:
          limits: {cpu: "$CPU", memory: ${MEM}Gi, nvidia.com/gpu: 1}
          requests: {cpu: "$CPU", memory: ${MEM}Gi, nvidia.com/gpu: 1}
        volumeMounts:
        - {name: vol, mountPath: /j-jepa-vol}
      volumes:
      - name: vol
        persistentVolumeClaim: {claimName: j-jepa-vol}
EOF
