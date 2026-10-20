"""H30 verification: spread of the unedited ParT_kin / EFN AUC over random 1000 q + 1000 g Pythia test samples (seeds 3001-3010)
and the H30 seed with 2000 + 2000. -> $AI_RESULTS/physics/h30_verify_auc.json"""
import json, os
from pathlib import Path
import numpy as np
import torch
from interp import h5_qg_counting as H
from interp import h14_qg_generator as H14
V = Path('/j-jepa-vol/aaron-interp'); RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
Mo = H.Models('cuda' if torch.cuda.is_available() else 'cpu'); Mo.add_weaver('part_kin', 'part', False)
net = H.make_ef('efn'); net.load_state_dict(torch.load(V / 'models' / 'qg' / 'h5_efn_torch.pt', map_location='cpu'))
Mo.add_torch('efn_t', net, {})
out = {}
for seed, pc in [(s, 1000) for s in range(3001, 3011)] + [(3001, 2000)]:
    L0, y, info = H14.load_sample('py', pc, seed); Jt = H.from_list(L0, Lpad=180)
    out[f'{seed}_{pc}'] = {nm: H.roc_metrics(Mo.logit(nm, Jt).astype(float), y) for nm in ('part_kin', 'efn_t')}
    print(seed, pc, {k: round(v['AUC'], 4) for k, v in out[f'{seed}_{pc}'].items()}, flush=True)
(RES / 'h30_verify_auc.json').write_text(json.dumps(out, indent=1))
