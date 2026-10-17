"""E22v figure from $AI_RESULTS/jedi/e22_verify.json -> e22_verify.png (python -m interp.e22_verify_plot)"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'
d = json.loads((OUT / 'e22_verify.json').read_text()); POST = ('S', 'h4', 'h5', 'z')
arms = [('prog', 'program (E22 arm)', '#1c5cab'), ('prog_base_agree', 'program, bases where program call = network call', '#7aa6d9'),
        ('ru_prog', "swap-free: program's runner-up class on the base jet", '#e39b3b'),
        ('ru_net', "swap-free: network's runner-up class on the base jet", '#b23a48'),
        ('src', "swap-free: network's call on the source jet", '#999999'), ('perm', 'permuted alignment (E22 null)', '#cccccc'),
        ('fnet', 'stage-s formulas, network layers downstream', '#5aa469')]
fig, ax = plt.subplots(figsize=(11, 4.6)); w = 0.115
for k, s in enumerate(POST):
    st = d['per_unit'][s]['stage']
    for m, (key, lab, c) in enumerate(arms):
        v = st[key]['mean']; ci = st[key].get('ci', st[key].get('interval'))
        ax.bar(k + (m - 3) * w, v, w, color=c, label=lab if k == 0 else None)
        ax.plot([k + (m - 3) * w] * 2, ci, color='#333333', lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(POST); ax.set_ylim(0, 1.3); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel('mean effective IIA over live units\n(pairs where the one-unit swap changes the network call)')
ax.legend(fontsize=7.5, frameon=False, loc='upper left', ncol=2, bbox_to_anchor=(0, 1.0))
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
m = d['meta']
ax.set_title(f"E22v, fresh sample (seed {m['seed']}, {m['jets']} jets, {m['pairs']} pairs): one-unit swaps vs swap-free predictors "
             '(bars: 95 % bootstrap over pairs)', fontsize=9.5, loc='left')
fig.tight_layout(); fig.savefig(OUT / 'e22_verify.png', dpi=150); print('wrote', OUT / 'e22_verify.png')
