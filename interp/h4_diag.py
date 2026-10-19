"""H4 smoke diagnostic: acceptance breakdown of the edits (reads physics/h4_smoke/h4_edits.npz)"""
import numpy as np, os, sys
from pathlib import Path
d = Path(os.environ['AI_RESULTS']) / 'physics' / (sys.argv[1] if len(sys.argv) > 1 else 'h4_smoke')
E = dict(np.load(d / 'h4_edits.npz'))
for name, tags in (('T', ['60', '100']), ('Q', ['80'])):
    O0 = {k.split('_obs_')[1]: E[k] for k in E if k.startswith(f'{name}_id_obs_')}
    for tag in tags:
        t = float(tag) if name == 'T' else 80.4
        for k in 'WNR':
            rr = E[f'{name}_{k}{tag}_rows']; O = {kk.split('_obs_')[1]: E[kk] for kk in E if kk.startswith(f'{name}_{k}{tag}_obs_')}
            dm = O['m'] - O0['m'][rr]; dt = O['tau32'] - O0['tau32'][rr]; dW = O['mW'] - (t if k == 'W' else O0['mW'][rr]); dsd = O['mSD'] - O0['mSD'][rr]
            q = lambda x: np.round(np.percentile(x, [10, 50, 90]), 3).tolist()
            print(name, tag, k, 'n', len(rr), '| dm', q(dm), 'frac<0.5', np.mean(np.abs(dm) < 0.5).round(3), '| dtau32', q(dt), 'frac<0.02', np.mean(np.abs(dt) < 0.02).round(3),
                  'frac<0.05', np.mean(np.abs(dt) < 0.05).round(3), '| dmW', q(dW), 'frac<5', np.mean(np.abs(dW) < 5).round(3), 'frac<2', np.mean(np.abs(dW) < 2).round(3), '| dmSD', q(dsd), '| D', q(E[f'{name}_{k}{tag}_D']))
