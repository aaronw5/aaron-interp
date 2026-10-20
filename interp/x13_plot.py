"""X13 figure from $AI_RESULTS/jedi/x13_summary.json -> x13_axis_size.png.
Left: per trained network, read-out response to the 0.06 rad translation (R) vs to the radial-only edit (R_rad), both divided by
the response to the matched dilation; points on the diagonal = translation response equals the radial-only response.
Right: R per architecture class, one dot per trained network (seed), grey crosses = untrained redraws (read-out refit)."""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')) / 'jedi'
S = json.loads((OUT / 'x13_summary.json').read_text())
COL = {'jedi-linear': '#1f77b4', 'jedi-linear-float': '#17becf', 'pfn': '#9467bd', 'efn': '#8c564b', 'dsets': '#2ca02c',
       'jedis': '#98df8a', 'linf': '#ff7f0e', 'tf': '#d62728'}
LAB = {'jedi-linear': 'JEDI-linear (HGQ)', 'jedi-linear-float': 'JEDI-linear float', 'pfn': 'PFN', 'efn': 'EFN', 'dsets': 'Deep Sets',
       'jedis': 'JEDI-style Deep Sets', 'linf': 'Linformer', 'tf': 'transformer'}
tr = [n for n, v in S['nets'].items() if v['meta']['trained']]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for c in COL:
    ns = [n for n in tr if S['nets'][n]['meta']['cls'] == c]
    x = np.array([S['nets'][n]['last']['R_rad'] for n in ns]); y = np.array([S['nets'][n]['last']['R'] for n in ns])
    ax[0].scatter(x, y, s=28, color=COL[c], label=f'{LAB[c]} ({len(ns)})', alpha=.85, edgecolor='none')
ax[0].plot([0, 2.2], [0, 2.2], color='gray', lw=.8, ls='--'); ax[0].axhline(0, color='gray', lw=.5)
ax[0].set_xlim(0.5, 2.2); ax[0].set_ylim(-1.3, 2.2)
ax[0].set_xlabel('radial-only edit: read-out $\\Delta\\log m$ / matched dilation')
ax[0].set_ylabel('0.06 rad translation: read-out $\\Delta\\log m$ / matched dilation  (R)')
A = S['across']['R_vs_Rrad']
ax[0].set_title(f'Response to translation vs to radial-only edit, 56 trained networks\n(dashed: equal; Spearman {A["rho"]:.2f}, n = {A["n"]})', fontsize=10)
ax[0].legend(fontsize=7.5, loc='lower right')
order = list(COL)
for i, c in enumerate(order):
    ns = [n for n in tr if S['nets'][n]['meta']['cls'] == c]
    y = [S['nets'][n]['last']['R'] for n in ns]
    ax[1].scatter(i + np.linspace(-.18, .18, len(y)), y, s=26, color=COL[c], zorder=3)
    u = [S['nets'][n]['last']['R'] for n, v in S['nets'].items() if not v['meta']['trained'] and v['meta']['cls'] == c
         and v['last']['R'] is not None]
    if u: ax[1].scatter(i + .32 + np.zeros(len(u)), np.clip(u, -1.3, 3.0), marker='x', s=14, color='gray', lw=.8)
ax[1].axhline(1, color='gray', lw=.6, ls=':'); ax[1].axhline(0, color='gray', lw=.5)
ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([LAB[c] for c in order], rotation=30, ha='right', fontsize=8)
ax[1].set_ylim(-1.3, 3.0); ax[1].set_ylabel('R (translation / matched dilation)')
V = S['variance']['logR_readout']
ax[1].set_title(f'R per architecture, one dot per trained seed / size\n(between-family share of var log R {V["between_share"]:.2f}, '
                f'permutation p {V["perm_p"]:.2f}); x = untrained', fontsize=10)
fig.text(0.01, 0.01, 'N = 2,500 test jets per network (sample per_class=1000, seed=1313); ridge log m read-out on the last hidden layer.',
         fontsize=7.5, color='gray')
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / 'x13_axis_size.png', dpi=130); print('wrote', OUT / 'x13_axis_size.png')
