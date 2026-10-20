#!/bin/bash
# H39: ParticleNet / ParT (kin features) with the authors' scripts (train_TopLandscape.sh / train_QuarkGluon.sh, weaver-core
# 0.4.17, authors' settings: 20 epochs, batch 512, ranger), variants for the architecture-matched null of the axis-direction test.
#   s1    torch seeded with <seed> before the model is built (interp/h39_seeded_net.py); labels as the authors'
#   shuf  labels independent of the class: h39_rl = hash((jet_pt, jet_eta, jet_phi)) mod 2 (weaver's _hash = Python tuple hash,
#         deterministic), jet_is<signal> = h39_rl, jet_is<background> = 1 - h39_rl; same 20 epochs (same number of steps)
# Output: /j-jepa-vol/aaron-interp/models/<ds>/<pnet|part>_kin_<var>/ (net_best_epoch_state.pt, net_epoch-*_state.pt, train.log,
# data.yaml, predict_output/)
#   interp/h2_gpu_job.sh d-h39-<ds>-<arch>-<var> 'bash interp/h39_weaver_train.sh <top|qg> <PN|ParT> <s1|shuf> <seed>'
set -uo pipefail
DS=$1; MODEL=$2; VAR=$3; SEED=$4
V=/j-jepa-vol/aaron-interp; PT=$V/ext/particle_transformer; C=$V/code
name=$( [ "$MODEL" = PN ] && echo pnet || echo part )_kin_$VAR
NETF=$PT/networks/$( [ "$MODEL" = PN ] && echo example_ParticleNet.py || echo example_ParticleTransformer.py )
OUT=$V/models/$DS/$name; mkdir -p $OUT
if [ "$DS" = top ]; then S=./train_TopLandscape.sh; Y=$PT/data/TopLandscape/top_kin.yaml; TF=/j-jepa-vol/ParT_top/TopLandscape/test_file.parquet
else S=./train_QuarkGluon.sh; Y=$PT/data/QuarkGluon/qg_kin.yaml; TF=/j-jepa-vol/QuarkGluon/test/test_file_0.parquet; fi
python - "$Y" "$OUT/data.yaml" "$VAR" "$TF" <<'PYEOF'
import re, sys
src, dst, var, tf = sys.argv[1:5]
t = open(src).read()
if var == 'shuf':
    t, n1 = re.subn(r'\n(\s+)(jet_is\w+): label\n', r'\n\1h39_rl: _hash(jet_pt, jet_eta, jet_phi) % 2\n\1\2: h39_rl\n', t)
    t, n2 = re.subn(r': 1-label\n', r': 1-h39_rl\n', t)
    assert n1 == 1 and n2 == 1, (n1, n2)
    import numpy as np, pyarrow.parquet as pq
    T = pq.read_table(tf, columns=['label', 'jet_pt', 'jet_eta', 'jet_phi'])
    a = [T.column(c).to_numpy() for c in ('jet_pt', 'jet_eta', 'jet_phi')]; y = T.column('label').to_numpy()
    rl = np.array([x.__hash__() for x in zip(*a)]) % 2
    print(f'shuffled-label check on {tf}: {len(y)} jets, mean h39_rl {rl.mean():.4f}, agreement with label {np.mean(rl == y):.4f}, '
          f'mean h39_rl in class 1 {rl[y == 1].mean():.4f}, class 0 {rl[y == 0].mean():.4f}', flush=True)
open(dst, 'w').write(t)
PYEOF
[ $? -eq 0 ] || { echo "data.yaml failed"; exit 1; }
grep -n "h39_rl\|jet_is" $OUT/data.yaml
for s in train val test; do ln -sf /j-jepa-vol/ParT_top/TopLandscape/${s}_file.parquet $PT/datasets/TopLandscape/${s}_file.parquet; done
echo "[$(date -u +%FT%TZ)] h39_weaver_train $DS $MODEL $VAR seed $SEED -> $OUT; particle_transformer $(git -C $PT rev-parse --short HEAD)"
nvidia-smi -L
( while true; do sleep 300; echo "[$(date -u +%FT%TZ)] monitor: gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader); mem $(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes) / 1048576 )) MiB; $(grep -c 'Current validation metric' $OUT/train.log 2>/dev/null) epochs validated"; done ) &
MON=$!
cd $PT
H39_BASE_NET=$NETF H39_SEED=$SEED $S $MODEL kin --data-config $OUT/data.yaml --network-config $C/interp/h39_seeded_net.py \
  --model-prefix $OUT/net --log $OUT/train.log --tensorboard h39_${DS}_${name} > $OUT/stdout.log 2>&1
rc=$?
kill $MON 2>/dev/null
grep -E "Current validation metric|Test metric" $OUT/train.log | tail -4
echo "[$(date -u +%FT%TZ)] h39_weaver_train done rc=$rc"; exit $rc
