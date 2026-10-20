"""X10 figure: DAS effective IIA vs random orthonormal subspaces, per stage, variable and k (JEDI-linear S/h4/h5, PFN L).
Local, seconds:  python interp/x10_plot.py  (reads results/jedi/x10_trained.json, x10_pfn.json; writes results/jedi/x10_das.png)"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parents[1] / 'results' / 'jedi'
VARS = ('M0', 'M2', 'D', 'WZ'); LAB = {'M0': r'$M_0=\sum p_T$', 'M2': r'$M_2=\sum p_T r^2$', 'D': r'$\frac{1}{2}\log M_0M_2$', 'WZ': 'W$-$Z logit'}
KS = ('1', '2', '4'); COL = {'1': '#2a6fb0', '2': '#e08a2c', '4': '#3a9a5b'}
T = json.loads((D / 'x10_trained.json').read_text()); P = json.loads((D / 'x10_pfn.json').read_text())
panels = [('JEDI-linear S (64)', T['stages']['S']), ('JEDI-linear h4 (64)', T['stages']['h4']), ('JEDI-linear h5 (32)', T['stages']['h5']),
          ('PFN latent L (32), X9 model', P['stages']['L'])]
fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
for ax, (title, res) in zip(axs, panels):
    for j, V in enumerate(VARS):
        for i, k in enumerate(KS):
            x = j + (i - 1) * 0.26; e = res['k'][k]['das'][V]['rand']; q = res['k'][k]['random']['rand'][V]['q']
            if q: ax.add_patch(plt.Rectangle((x - 0.11, q[0]), 0.22, q[2] - q[0], color='0.82', lw=0))
            if q: ax.plot([x - 0.11, x + 0.11], [q[1]] * 2, color='0.45', lw=1)
            if e['eff'] is not None:
                ax.errorbar(x, e['eff'], yerr=[[e['eff'] - e['eff_ci'][0]], [e['eff_ci'][1] - e['eff']]], fmt='o', color=COL[k], ms=5,
                            label=f'DAS, k = {k}' if j == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels([LAB[v] for v in VARS], fontsize=9); ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1); ax.grid(axis='y', alpha=.3)
axs[0].set_ylabel('effective IIA (pairs where the counterfactual call changes)')
axs[0].add_patch(plt.Rectangle((0, 0), 0, 0, color='0.82', label='random subspaces, 95 % band'))
axs[0].legend(fontsize=8, loc='upper right')
fig.suptitle('Effective IIA of learned k-dim subspaces (dots, 95 % CI) vs 1,000 random k-dim subspaces (grey, 95 % band); '
             '5,000 test pairs, stored-axis variables', fontsize=10)
fig.tight_layout(); fig.savefig(D / 'x10_das.png', dpi=130); print('wrote', D / 'x10_das.png')
