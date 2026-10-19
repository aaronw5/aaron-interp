"""X5: write the weights of trained dsets-d64 runs as numpy archives, for analyses in the keras/jax env (no torch there).

    AI_PY=conda ./job.sh x5-export 'python train/export_npz.py <run_dir> [<run_dir> ...]'

Per run dir: weights.npz from best.pt and, when present, weights_zeroed.npz from best_zeroed.pt (X5 L1 runs). Keys: the
state_dict names with '.' -> '_' (phi_0_weight (64, 3), phi_0_bias, phi_2_*, rho_0_*, rho_2_*, rho_4_*, head_*), plus shift and
scale (arch.json inputs). Also writes epoch0.npz from ckpt/epoch_0.pt (initialisation).
"""
import json, os, sys
import numpy as np
import torch

for d in sys.argv[1:]:
    arch = json.load(open(os.path.join(d, 'arch.json')))
    extra = dict(shift=np.asarray(arch['inputs']['shift'], np.float64), scale=np.asarray(arch['inputs']['scale'], np.float64))
    for src, dst in (('best.pt', 'weights.npz'), ('best_zeroed.pt', 'weights_zeroed.npz'), ('ckpt/epoch_0.pt', 'epoch0.npz')):
        p = os.path.join(d, src)
        if not os.path.exists(p): continue
        sd = torch.load(p, map_location='cpu')
        np.savez(os.path.join(d, dst), **{k.replace('.', '_'): v.numpy() for k, v in sd.items()}, **extra)
        print(d, src, '->', dst, {k: tuple(v.shape) for k, v in sd.items()}, flush=True)
