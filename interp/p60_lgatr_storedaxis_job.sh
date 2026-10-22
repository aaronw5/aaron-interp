#!/bin/bash
# P60: L-GATr trained with ParT's stored-axis inputs, as a GPU Job (NRP: requests == limits; GPU must stay > 40 % busy).
# Same as interp/p5_lgatr_job.sh (authors' code heidelberg-hepml/lorentz-gatr @ 35aca2a, config_paper/jctagging: 12 blocks,
# 16 mv / 32 s channels, 8 heads, Lion lr 3e-4, cosine, batch 128, 200,000 iterations, validation every 5,000; same 2M-jet
# subset $V/data/jetclass2m and file ranges train [0,2], val [120,121], test [100,101]; seed 0) with ONE change:
# data.features=storedaxis, provided by interp/p60_lgatr_storedaxis.patch: the per-particle scalar features dphi, deta, dR of
# add_scalar_features are the stored JetClass part_dphi / part_deta (about jet_eta / jet_phi, ParT's inputs) instead of being
# recomputed about the particle four-momentum sum; log pT, log E, log pT/pT_jet, log E/E_jet, four-vectors, spurions unchanged.
# The code runs from a private clone $V/ext/lorentz-gatr-p60 (35aca2a + p5 loader patch + p60 patch), so the shared
# ext/lorentz-gatr used by other jobs is never modified. Before training, the job runs the unit test
# (python -m interp.p60_axis_battery --stage unittest: stored features == ParT inputs to 1e-6, default path == pristine code,
# loader check, forward check); training starts only if it passes.
#   interp/p60_lgatr_storedaxis_job.sh <name> <run_name> [extra hydra overrides]
# e.g. interp/p60_lgatr_storedaxis_job.sh c-p60-lgsa jc2m_sa_s0
# Job anrunw-ai-<name>; log $V/logs/<name>.log; checkpoints $V/models/lgatr/runs/jc2m/<run_name>/models/
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,16p "$0"; exit 1; }
NAME=$1; RUN=$2; shift 2; EXTRA="$*"; CPU=4; MEM=16
V=/j-jepa-vol/aaron-interp
OVR="data.data_dir=$V/data/jetclass2m data.train_files_range=[0,2] data.val_files_range=[120,121] data.test_files_range=[100,101] \
exp_name=jc2m run_name=$RUN seed=0 base_dir=$V/models/lgatr save_source=false \
mlflow.db=$V/models/lgatr/runs/jc2m/mlflow/mlflow.db mlflow.artifacts=$V/models/lgatr/runs/jc2m/mlflow/artifacts data.features=storedaxis $EXTRA"
SRC=$V/ext/lorentz-gatr-p60
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
                values: [k8s-haosu-24.sdsc.optiputer.net, ry-gpu-01.sdsc.optiputer.net, nautilus-it-gpu05.fullerton.edu, fiona-prg1.cesnet.cz, k8s-chase-ci-07.calit2.optiputer.net, dtn-gpu2.kreonet.net, k8s-haosu-06.sdsc.optiputer.net, fiona8.ucsc.edu, k8s-haosu-22.sdsc.optiputer.net, k8s-haosu-09.sdsc.optiputer.net, k8s-chase-ci-01.calit2.optiputer.net]
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
          git config --global --add safe.directory '*'
          [ -d $SRC/.git ] || git clone -q $V/ext/lorentz-gatr $SRC
          cd $SRC && git checkout -q 35aca2ac4921b8092b5a14f40deed11e7c015e53 && git checkout -q -- . && rm -f experiments/tagging/miniweaver/storedaxis.yaml
          git apply $V/code/interp/p5_lgatr.patch && git apply $V/code/interp/p60_lgatr_storedaxis.patch
          t0=\$(date +%s)
          ( echo "[\$(date -u +%FT%TZ)] \$(hostname) lorentz-gatr-p60 \$(git rev-parse --short HEAD) python $V/env-lgatr/bin/python \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
            echo "overrides: $OVR"; echo "patch: \$(git diff --stat | tail -1)"
            ( while sleep 600; do echo "[gpu \$(date -u +%FT%TZ)] \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
            ( cd $V/code && LGATR_SRC=$SRC JETDISTILL_TAGGER=part JETDISTILL_DATA=$V/code/cluster/data_part AI_RESULTS=$V/results PYTHONPATH=$V/code \
              $V/env-lgatr/bin/python -m interp.p60_axis_battery --stage unittest ); rc=\$?
            echo "[\$(date -u +%FT%TZ)] unit test exit \$rc"
            [ \$rc -eq 0 ] && { $V/env-lgatr/bin/python run.py -cp config_paper -cn jctagging $OVR; rc=\$?; }
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
