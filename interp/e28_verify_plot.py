"""E28 verification figure: (a) calls changed by the collinear split (z 0.5) vs the two matched-rank controls (angle swap with rank k+1;
rotation of the same particle about the jet axis, which keeps N, ranks, M0, M1, M2); (b) share of split-changed calls removed by holding
named S sets vs random sets of the same size (20 random 2-unit and 20 random 17-unit subsets of the 19 live S units).
python -m interp.e28_verify_plot  (reads results/jedi/e28_verify.json, writes results/jedi/e28_verify.png)"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.loads((RES / 'e28_verify.json').read_text()); E = R['edits']
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
a = ax[0]; ranks = ['0', '5', '13', 'soft']; lab = ['rank 0', 'rank 5', 'rank 13', 'rank > 20']; w = 0.27
for i, (nm, pat, col) in enumerate([('split z=0.5', 'split_r{}_z0.5', '#2b6cb0'), ('control: swap angles with rank k+1', 'swap_r{}', '#999999'),
                                    ('control: rotate particle k about the axis\n(N, ranks, M0, M1, M2 fixed)', 'rot_r{}', '#38a169')]):
    v = np.array([E[pat.format(r)]['changed'] for r in ranks]) * 100; ci = np.array([E[pat.format(r)]['changed_ci'] for r in ranks]).T * 100
    xs = np.arange(4) + (i - 1) * w
    a.bar(xs, v, w, color=col, label=nm); a.errorbar(xs, v, yerr=[v - ci[0], ci[1] - v], fmt='none', ecolor='k', capsize=2, lw=0.8)
a.set_xticks(range(4)); a.set_xticklabels(lab); a.set_ylabel('calls changed (%)'); a.legend(fontsize=8)
a.set_title(f"(a) split vs matched-rank controls, {R['n_sub']} jets (seed {R['seed']})", fontsize=10)
b = ax[1]; eds = ['split_r0_z0.5', 'split_r5_z0.5', 'split_r13_z0.5', 'split_rsoft_z0.5', 'soft_add1', 'soft_add5']
elab = ['split r0', 'split r5', 'split r13', 'split r>20', 'soft +1', 'soft +5']
for j, e in enumerate(eds):
    H = E[e]['holds']
    r2 = [H[f'rand2_{i}']['removed'] for i in range(20)]; r17 = [H[f'rand17_{i}']['removed'] for i in range(20)]
    b.scatter(np.full(20, j - 0.15) + np.random.default_rng(j).uniform(-0.04, 0.04, 20), r2, s=8, c='#bbbbbb')
    b.scatter(np.full(20, j + 0.15) + np.random.default_rng(j + 9).uniform(-0.04, 0.04, 20), r17, s=8, c='#f6ad55')
    b.scatter([j - 0.15], [H['S0+S19']['removed']], marker='D', c='#2b6cb0', s=30, zorder=3)
    b.scatter([j + 0.15], [H['window17']['removed']], marker='D', c='#c05621', s=30, zorder=3)
b.scatter([], [], s=8, c='#bbbbbb', label='random 2 of 19 live S held'); b.scatter([], [], marker='D', c='#2b6cb0', label='S0 + S19 held')
b.scatter([], [], s=8, c='#f6ad55', label='random 17 of 19 held'); b.scatter([], [], marker='D', c='#c05621', label='17 rank-window S held (only S29, S30 free)')
b.axhline(0, color='k', lw=0.6); b.set_xticks(range(len(eds))); b.set_xticklabels(elab); b.set_ylabel('share of changed calls removed by the hold')
b.legend(fontsize=7.5, loc='upper left'); b.set_title('(b) hold shares vs random S sets of the same size', fontsize=10)
fig.tight_layout(); fig.savefig(RES / 'e28_verify.png', dpi=130); print('wrote', RES / 'e28_verify.png')
