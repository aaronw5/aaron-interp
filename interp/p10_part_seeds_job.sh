#!/bin/bash
# P10 copy of interp/h2_gpu_job.sh (deadline 72 h; CPU / memory from P10_CPU / P10_MEM, default 8 / 32; GPUs from P10_GPUS, default >= 44 GB; the P10 launch used 2 CPU / 12 Gi and the >= 22 GB list, one seed per job). Run a command on the cluster as a GPU Job (NRP: requests == limits; CPU use 20-200 %
# of request, memory 20-150 %). The command runs in /j-jepa-vol/aaron-interp/code (run ./sync.sh first) with the venv
# /j-jepa-vol/aaron-interp/env-part first on PATH: --system-site-packages over the image's /opt/conda python (torch 1.13.1 +
# CUDA 11.7, uproot 5.0, awkward 2.3) plus weaver-core 0.4.17 (the version the released ParT weights are run with; the image's
# weaver-core 0.3.5 has no ParticleTransformer). The job creates the venv if it is missing.
# Paths: JETDISTILL_TAGGER=part, JETDISTILL_DEVICE=cuda (read by jetdistill/part/network.py), JETDISTILL_DATA=
# /j-jepa-vol/aaron-interp/code/cluster/data_part (its test/files.json, kept in this repo, points at the read-only JetClass
# test files /j-jepa-vol/JetClass/Pythia/test_20M/<sample>_10[01].root — the same 20 files as the local data_part/raw/test),
# JETDISTILL_MODELS=/j-jepa-vol/aaron-interp/models/part (ParT_{full,kin,kinpid}.pt, md5-identical to the local copies).
#   interp/p10_part_seeds_job.sh <name> '<command>'
# e.g. interp/p10_part_seeds_job.sh c-p10-full 'bash interp/p10_part_seeds_train.sh full 0 1 2'
# Job anrunw-ai-<name> (labels owner=anrunw, study=aaron-interp); log /j-jepa-vol/aaron-interp/logs/<name>.log
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,11p "$0"; exit 1; }
NAME=$1; CMD=$2; CPU=${P10_CPU:-8}; MEM=${P10_MEM:-32}
G24='NVIDIA-GeForce-RTX-3090, NVIDIA-A10, NVIDIA-GeForce-RTX-4090, NVIDIA-TITAN-RTX, Quadro-RTX-6000, NVIDIA-RTX-A5000'
G48='NVIDIA-L40, NVIDIA-L40S, NVIDIA-RTX-A6000, NVIDIA-A40, Quadro-RTX-8000'
GPUS=${P10_GPUS:-$G48}
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
      activeDeadlineSeconds: 259200
      affinity:  # nodes where CephFS metadata ops stalled (D state in ceph_mdsc_wait_request), 2026-10-08
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [k8s-haosu-24.sdsc.optiputer.net, ry-gpu-01.sdsc.optiputer.net, nautilus-it-gpu05.fullerton.edu, fiona-prg1.cesnet.cz, k8s-chase-ci-07.calit2.optiputer.net, dtn-gpu2.kreonet.net]
              - key: nvidia.com/gpu.product   # P10_GPUS: GPU products allowed (one ParT at batch 512 used 15 GB in H2; three run here)
                operator: In
                values: [$GPUS]
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
