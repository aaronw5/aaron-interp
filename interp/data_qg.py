"""Quark-gluon loader (Komiske, Metodiev, Thaler, JHEP 01 (2019) 121; Pythia 8.226, pp -> Z(nu nu) + (q or g), anti-kT R = 0.4,
500 < pT < 550 GeV, |y| < 1.7, all final-state particles with PDG id; parquet conversion of the particle_transformer repo,
utils/convert_qg_datasets.py). Read-only source: /j-jepa-vol/QuarkGluon/{train/train_file_0..17, test/test_file_0..1}.parquet
(1.8M / 200k jets; label 1 = quark, 0 = gluon). The QuarkGluon.tar listing holds the same 20 Pythia parquet files only (no
Herwig sample; checked 2026-10-19 with tar t, nothing extracted).

Converted arrays in $AI_DATA_QG (default /j-jepa-vol/aaron-interp/data/qg)/{train,test}/, same layout as interp.data_top
(label, jet, kin3, full, pid; train also is_val = last 11.11 % of each file, the val split of weaver --train-val-split 0.8889;
train has no full.npy). sample(per_class, seed, split='test') as interp.data_top.sample (random rows per class).
"""
import os
from pathlib import Path
from interp import data_top as _t

SRC = Path('/j-jepa-vol/QuarkGluon')
DST = Path(os.environ.get('AI_DATA_QG', '/j-jepa-vol/aaron-interp/data/qg'))
log = _t.log


def convert():
    info = {}
    for split, n in (('test', 2), ('train', 18)):
        if (DST / split / 'done').exists():
            log(f'{split}: exists'); continue
        files = [SRC / split / f'{split}_file_{i}.parquet' for i in range(n)]
        info[split] = _t.convert_files(files, DST / split, full=split == 'test', pid=True,
                                       val_frac=(1 - 0.8889) if split == 'train' else None)
        (DST / split / 'done').write_text('ok')
    return info


def sample(per_class=5000, seed=0, split='test'):
    return _t.sample(per_class, seed, split, root=DST)


def load_split(split, keys=('label', 'jet', 'full', 'pid')):
    return _t.load_split(split, root=DST, keys=keys)


if __name__ == '__main__':
    print(convert())
