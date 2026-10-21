"""P44v analysis (CPU): redundancy of grade 0 and the auxiliary scalars s on P44's own per-jet logits (p44_main.npz, N 10,000).
Per block k (mean ablation): drop_s, drop_g0, drop_inv (inv = g0 + s) in accuracy points vs unedited, and the interaction
I = drop_inv - drop_s - drop_g0 (0 if the two groups act independently and additively; > 0 if each can stand in for the other),
plus the cost of removing g0 once s is already removed (acc_s - acc_inv) with a paired 95 % bootstrap over jets and McNemar.
  ./job.sh c-p44-verify-an 1 2 'python -m interp.p44v_analyze'   -> $AI_RESULTS/part/p44_verify_an.json
"""
import json, os
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

R = Path(os.environ.get('AI_RESULTS', 'results'))
Z = np.load(R / 'lgatr' / 'p44_main.npz'); y = Z['y']; c0 = Z['L0'].argmax(1) == y; Lg = Z['L_grades']
GN = ['s', 'g0', 'g1', 'g2', 'g3', 'g4', 'inv', 'cov']; rng = np.random.default_rng(4545); n = len(y); B = 2000
idx = rng.integers(0, n, (B, n)); out = {}
for k in range(Lg.shape[0]):
    c = {g: Lg[k, GN.index(g), 0].astype(np.float32).argmax(1) == y for g in ('s', 'g0', 'inv')}
    for mode, mi in (('mean', 0), ('resample', 1)):
        c = {g: Lg[k, GN.index(g), mi].astype(np.float32).argmax(1) == y for g in ('s', 'g0', 'inv')}
        dI = (c0.astype(float) - c['inv']) - (c0.astype(float) - c['s']) - (c0.astype(float) - c['g0'])   # per-jet interaction
        dg = c['s'].astype(float) - c['inv']                                                           # g0 cost given s removed
        bI, bg = dI[idx].mean(1), dg[idx].mean(1)
        b_, c_ = int((c['s'] & ~c['inv']).sum()), int((~c['s'] & c['inv']).sum())
        out[f'b{k}_{mode}'] = dict(acc0=float(c0.mean()), acc_s=float(c['s'].mean()), acc_g0=float(c['g0'].mean()), acc_inv=float(c['inv'].mean()),
                                  interaction=[float(dI.mean()), float(np.percentile(bI, 2.5)), float(np.percentile(bI, 97.5))],
                                  g0_cost_given_s_removed=[float(dg.mean()), float(np.percentile(bg, 2.5)), float(np.percentile(bg, 97.5))],
                                  mcnemar_s_vs_inv=dict(lost=b_, gained=c_, p=float(binomtest(b_, b_ + c_).pvalue) if b_ + c_ else 1.0))
        print(k, mode, {kk: (np.round(v, 3).tolist() if isinstance(v, list) else v) for kk, v in out[f'b{k}_{mode}'].items()}, flush=True)
(R / 'part').mkdir(parents=True, exist_ok=True)
(R / 'part' / 'p44_verify_an.json').write_text(json.dumps(out, indent=1))
print('wrote')
