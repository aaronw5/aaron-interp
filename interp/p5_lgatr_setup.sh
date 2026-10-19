#!/bin/bash
# P5 setup (CPU job, run once): L-GATr code, python env and the JetClass 2M-train subset on the volume.
#   AI_PY=conda ./job.sh c-p5-setup 1 4 'bash interp/p5_lgatr_setup.sh'
# 1. clone heidelberg-hepml/lorentz-gatr into $V/ext/lorentz-gatr at the pinned commit (authors' code, unmodified)
# 2. venv $V/env-lgatr (image python 3.10, no system site packages): torch 2.4.0 + xformers 0.0.27.post2 (cu121 wheels; 0.0.28 no longer exports fmha.BlockDiagonalMask, which experiments/tagging/wrappers.py imports)
#    and the repo's requirements.txt packages (pytest / black / torchdiffeq left out: tests and event generation only; clifford 1.4.0 from PyPI with numba 0.58.1 (clifford 1.4 calls numba.generated_jit, removed in numba 0.59): gatr.primitives.linear imports it)
# 3. $V/data/jetclass2m/{train_100M,val_5M,test_20M}: symlinks into the read-only /j-jepa-vol/JetClass/Pythia, named so that the
#    authors' file-range logic (<class>_<NNN>.root over range(*files_range)) picks them up:
#      train_100M/<class>_000.root, _001.root  <- 2 of the 100 training files per class, drawn at random (seed 2026) = 2M jets
#      val_5M/<class>_120.root                <- 1 of the 5 validation files (120-124) per class, at random (same rng) = 1M jets
#      test_20M/<class>_100.root              <- test file 100 of each class (holds the P1 rows 3,000-4,000) = 1M jets
#    the draw is written to $V/data/jetclass2m/selection.json
set -euo pipefail
V=/j-jepa-vol/aaron-interp
COMMIT=35aca2ac4921b8092b5a14f40deed11e7c015e53
SRC=/j-jepa-vol/JetClass/Pythia
echo "[setup] $(date -u +%FT%TZ) clone"
mkdir -p $V/ext
if [ ! -d $V/ext/lorentz-gatr/.git ]; then git clone -q https://github.com/heidelberg-hepml/lorentz-gatr.git $V/ext/lorentz-gatr; fi
git -C $V/ext/lorentz-gatr checkout -q $COMMIT
echo "[setup] lorentz-gatr at $(git -C $V/ext/lorentz-gatr rev-parse HEAD); status: $(git -C $V/ext/lorentz-gatr status --porcelain | wc -l) changed files"

echo "[setup] $(date -u +%FT%TZ) venv"
[ -x $V/env-lgatr/bin/python ] || /opt/conda/bin/python3 -m venv $V/env-lgatr
P=$V/env-lgatr/bin/python
$P -m pip install -q --no-cache-dir --upgrade pip
$P -m pip install -q --no-cache-dir torch==2.4.0 xformers==0.0.27.post2
$P -m pip install -q --no-cache-dir 'numpy<2' scipy scikit-learn matplotlib h5py hdf5plugin tqdm wget einops opt_einsum \
   torch_geometric==2.5.3 torch-ema 'pytorch_optimizer==3.1.2' hydra-core==1.3.2 mlflow==2.4.1 \
   'PyYAML>=5.4' 'awkward0>=0.15.5' 'uproot>=4.2.0,<5.2.0' 'awkward>=1.8.0,<2.7' lz4 xxhash vector tables 'clifford==1.4.0' 'numba==0.58.1'
$P -m pip freeze > $V/env-lgatr/freeze.txt
echo "[setup] $(wc -l < $V/env-lgatr/freeze.txt) packages; torch $($P -c 'import torch; print(torch.__version__)')"

echo "[setup] $(date -u +%FT%TZ) import check"
( cd $V/ext/lorentz-gatr && $P -c 'import logging.handlers, hydra, experiments.base_experiment, experiments.tagging.jetclassexperiment, experiments.tagging.wrappers, gatr, xformers, mlflow, pytorch_optimizer; print("imports ok")' )

echo "[setup] $(date -u +%FT%TZ) JetClass 2M subset"
$P - <<'EOF'
import json, os, numpy as np
V = '/j-jepa-vol/aaron-interp'; SRC = '/j-jepa-vol/JetClass/Pythia'; D = f'{V}/data/jetclass2m'
classes = ['ZJetsToNuNu', 'HToBB', 'HToCC', 'HToGG', 'HToWW4Q', 'HToWW2Q1L', 'TTBar', 'TTBarLep', 'WToQQ', 'ZToQQ']
rng = np.random.default_rng(2026); sel = dict(seed=2026, train={}, val={}, test={})
for s in ('train_100M', 'val_5M', 'test_20M'): os.makedirs(f'{D}/{s}', exist_ok=True)
def link(src, dst):
    assert os.path.exists(src), src
    if os.path.islink(dst): os.remove(dst)
    os.symlink(src, dst)
for c in classes:
    tr = sorted(int(i) for i in rng.choice(100, 2, replace=False)); va = int(rng.choice(np.arange(120, 125)))
    for k, i in enumerate(tr): link(f'{SRC}/train_100M/{c}_{i:03d}.root', f'{D}/train_100M/{c}_{k:03d}.root')
    link(f'{SRC}/val_5M/{c}_{va}.root', f'{D}/val_5M/{c}_120.root')
    link(f'{SRC}/test_20M/{c}_100.root', f'{D}/test_20M/{c}_100.root')
    sel['train'][c] = tr; sel['val'][c] = va; sel['test'][c] = 100
json.dump(sel, open(f'{D}/selection.json', 'w'), indent=1); print('[setup] selection', json.dumps(sel))
import uproot
t = uproot.open(f'{D}/train_100M/HToBB_000.root')['tree']; print('[setup] HToBB_000 ->', os.readlink(f'{D}/train_100M/HToBB_000.root'), t.num_entries, 'jets')
EOF
echo "[setup] $(date -u +%FT%TZ) done"
