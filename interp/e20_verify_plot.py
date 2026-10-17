"""E20v figure: |dS| of S43 / S63 / S17 per angle swap W_k (ranks k <-> k+1), float + masks program with the trained, shifted and
removed slot masks, and the fixed-point network. Reads $AI_RESULTS/jedi/e20_verify.json; writes results/jedi/e20_verify_profiles.png."""
import json, os
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.load(open(RES / 'e20_verify.json'))
ST = [('net', 'fixed-point network', '#1f2328', '-', 'o'), ('mask', 'program, trained masks', '#2a6fdb', '-', 's'),
      ('shift-3_all4', 'masks shifted -3', '#d9822b', '--', 'v'), ('shift+3_all4', 'masks shifted +3', '#8b5cf6', '--', '^'),
      ('ones_l3_qs1', 'l3 + qs1 masks = 1', '#a0a7b0', ':', 'x')]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True)
for a, (c, win) in zip(ax, (('43', '0-13'), ('63', '0-13'), ('17', '0-9'))):
    for p, lab, col, ls, mk in ST:
        a.plot(range(17), R['profiles'][c][p]['profile'], ls=ls, marker=mk, ms=4, color=col, label=lab)
    a.set_title(f'$S_{{{c}}}$ (qs1 window {win}), N = {R["N_scan"]:,} jets, seed {R["seed"]}', fontsize=10, loc='left')
    a.set_xlabel('swap W$_k$: angles of ranks k and k+1 exchanged'); a.set_xticks(range(0, 17, 2))
    for s in ('top', 'right'): a.spines[s].set_visible(False)
ax[0].set_ylabel(r'mean $|\Delta S|$ per jet'); ax[0].legend(fontsize=8, frameon=False)
fig.suptitle('|dS| peaks at the swap across the window edge and the peak moves with the shifted masks (13 -> 10 / 16); '
             'with the l3 and qs1 masks set to 1 (program without add-layer masks) there is no peak', fontsize=11, x=.01, ha='left')
fig.tight_layout(); out = ROOT / 'results' / 'jedi' / 'e20_verify_profiles.png'; fig.savefig(out, dpi=120); print(out)
