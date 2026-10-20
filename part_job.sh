#!/bin/bash
# Run a ParT (Particle Transformer, JetClass) command on the cluster as a GPU Job (NRP: requests == limits; CPU use 20-200 %
# of request, memory 20-150 %). The command runs in /j-jepa-vol/aaron-interp/code (run ./sync.sh first) with the venv
# /j-jepa-vol/aaron-interp/env-part first on PATH: --system-site-packages over the image's /opt/conda python (torch 1.13.1 +
# CUDA 11.7, uproot 5.0, awkward 2.3) plus weaver-core 0.4.17 (the version the released ParT weights are run with; the image's
# weaver-core 0.3.5 has no ParticleTransformer). The job creates the venv if it is missing.
# Paths: JETDISTILL_TAGGER=part, JETDISTILL_DEVICE=cuda (read by jetdistill/part/network.py), JETDISTILL_DATA=
# /j-jepa-vol/aaron-interp/code/cluster/data_part (its test/files.json, kept in this repo, points at the read-only JetClass
# test files /j-jepa-vol/JetClass/Pythia/test_20M/<sample>_10[01].root — the same 20 files as the local data_part/raw/test),
# JETDISTILL_MODELS=/j-jepa-vol/aaron-interp/models/part (ParT_{full,kin,kinpid}.pt, md5-identical to the local copies).
#   ./part_job.sh <name> '<command>'                      (4 CPU, 16 Gi, 1 GPU)
# e.g. ./part_job.sh p1 'python -m interp.p0_smoke_part && python -m interp.p1_translation_part'
# Job anrunw-ai-<name> (labels owner=anrunw, study=aaron-interp); log /j-jepa-vol/aaron-interp/logs/<name>.log
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,11p "$0"; exit 1; }
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
                values: [k8s-haosu-24.sdsc.optiputer.net, ry-gpu-01.sdsc.optiputer.net, nautilus-it-gpu05.fullerton.edu, fiona-prg1.cesnet.cz, k8s-chase-ci-07.calit2.optiputer.net, dtn-gpu2.kreonet.net, k8s-haosu-06.sdsc.optiputer.net, fiona8.ucsc.edu, k8s-haosu-22.sdsc.optiputer.net]
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
