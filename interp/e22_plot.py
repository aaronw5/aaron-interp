"""E22 figure from $AI_RESULTS/jedi/e22_iia.json -> e22_iia.png (python -m interp.e22_plot)"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'
d = json.loads((OUT / 'e22_iia.json').read_text()); POST = ('S', 'h4', 'h5', 'z')
col = dict(S='#1c5cab', h4='#8fb3de', h5='#e39b3b', z='#b23a48')
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.45, 1]))
lab = {('S', 0): (4, 6), ('S', 50): (4, -11), ('S', 19): (-24, 6), ('S', 1): (5, -3), ('S', 29): (5, -3), ('z', 0): (5, 0), ('z', 7): (5, 4),
       ('z', 1): (5, 0), ('h4', 49): (5, 0), ('h5', 26): (5, -3)}
for s in POST:
    rr = [r for r in d['per_unit'][s]['units'] if r['n_eff'] >= 20]
    x = np.array([r['cost'] for r in rr]); y = np.array([r['iia_eff'] for r in rr])
    lo = y - np.array([r['iia_eff_ci'][0] for r in rr]); hi = np.array([r['iia_eff_ci'][1] for r in rr]) - y
    ax[0].errorbar(x, y, yerr=[lo, hi], fmt='o', ms=4.5, color=col[s], ecolor=col[s], elinewidth=0.7, alpha=0.9, label=f'{s} units ({len(rr)})')
    for r in rr:
        if (s, r['unit']) in lab:
            ax[0].annotate(f'{s}{r["unit"]}', (r['cost'], r['iia_eff']), fontsize=8, xytext=lab[(s, r['unit'])], textcoords='offset points')
lo_, hi_ = min(d['per_unit'][s]['stage_perm_null_interval'][0] for s in POST), max(d['per_unit'][s]['stage_perm_null_interval'][1] for s in POST)
ax[0].axhspan(lo_, hi_, color='#cccccc', alpha=0.5, lw=0)
ax[0].text(0.33, hi_ + 0.012, 'permuted alignment, stage means (95 % of draws)', fontsize=8, color='#444444')
ax[0].set_xscale('log'); ax[0].set_xlabel('E1 cost of the unit (% of calls changed when the unit is set to its mean)')
ax[0].set_ylabel('effective IIA: program call = network call,\non pairs where the swap changes the network call')
ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8, loc='upper left', frameon=False)
ax[0].set_title('One unit swapped from a source jet into a base jet (5000 pairs)', fontsize=10, loc='left')
names = ['program (unit-aligned)', 'permuted alignment', 'random direction', 'probe from 380 observables', 'formula, network downstream']
cols = ['#1c5cab', '#bbbbbb', '#7f7f7f', '#e39b3b', '#5aa469']; w = 0.16
for k, s in enumerate(POST):
    v = [d['per_unit'][s]['stage_mean_iia_eff'], d['per_unit'][s]['stage_perm_null_mean'], d['rotation'][s]['iia_eff_mean'],
         float(np.nanmean([r['iia_eff'] for r in d['probe'][s]['units'] if r['n_eff'] >= 20])),
         float(np.nanmean([r['iia_eff'] for r in d['controls']['trained_a'][s] if r['n_eff'] >= 20]))]
    for m, vv in enumerate(v):
        ax[1].bar(k + (m - 2) * w, vv, w, color=cols[m], label=names[m] if k == 0 else None)
    ri = d['rotation'][s]['iia_eff_interval']; ax[1].plot([k, k], ri, color='#333333', lw=0.8)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([f'{s}' for s in POST]); ax[1].set_ylim(0, 1)
ax[1].set_ylabel('mean effective IIA over the stage\'s live units')
ax[1].legend(fontsize=8, frameon=False, loc='upper left', ncol=2)
ax[1].set_title('Alignments and controls (bar line: 95 % of random directions)', fontsize=10, loc='left')
for a_ in ax:
    for sp in ('top', 'right'): a_.spines[sp].set_visible(False)
fig.suptitle('E22: after a one-unit swap the program gives the network\'s new call in 19–87 % of pairs (permuted alignment 18–41 %)', fontsize=10, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / 'e22_iia.png', dpi=150); print('wrote', OUT / 'e22_iia.png')
