"""X9w figure: W -> Z flow under a 0.06 rad rigid translation (+deta) per training cell and architecture, with the rotation and
M2-held controls, and test accuracy per cell.  Reads results/jedi/x9w_repr_2x2.json, writes results/jedi/x9w_repr_2x2.png.
    python -m interp.x9w_plot          (local matplotlib is fine: seconds)
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
CELLS = [('stored', 'stored axis'), ('aug', 'stored + aug'), ('rc', 'recentred'), ('rcaug', 'recentred + aug')]
ARCH = [('pfn', 'PFN'), ('efn', 'EFN'), ('dsets', 'Deep Sets'), ('jedi', 'JEDI-linear\n(HGQ-50)')]
COL = ['#2b6cb0', '#63b3ed', '#c05621', '#f6ad55']


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((RES / 'x9w_repr_2x2.json').read_text())['models']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
    w = 0.19
    for j, (a, al) in enumerate(ARCH):
        for i, (c, cl) in enumerate(CELLS):
            ms = [v for k, v in R.items() if v.get('arch') == a and v.get('cell') == c and 'edits' in v]
            if not ms: continue
            xp = j + (i - 1.5) * w
            fT = [m['edits']['T']['flow'] for m in ms]; fR = [m['edits']['rot']['flow'] for m in ms]; fM = [m['edits']['M2held']['flow'] for m in ms]
            ax[0].bar(xp, np.mean(fT), w * 0.92, color=COL[i], label=cl if j == 0 else None)
            for m in ms:
                lo, hi = m['edits']['T']['flow_ci']; ax[0].plot([xp, xp], [lo, hi], color='k', lw=0.8)
            ax[0].scatter([xp] * len(fT), fT, s=10, color='k', zorder=3)
            ax[0].scatter([xp] * len(fR), fR, marker='_', s=90, color='0.45', zorder=3, label='rotation (null)' if (j, i) == (0, 0) else None)
            ax[0].scatter([xp] * len(fM), fM, marker='x', s=22, color='#9b2c2c', zorder=3, label='$M_2$-held translation' if (j, i) == (0, 0) else None)
            acc = [100 * m['info']['acc_test'] for m in ms]
            ax[1].bar(xp, np.mean(acc), w * 0.92, color=COL[i]); ax[1].scatter([xp] * len(acc), acc, s=10, color='k', zorder=3)
    for k in (0, 1):
        ax[k].set_xticks(range(len(ARCH))); ax[k].set_xticklabels([al for _, al in ARCH], fontsize=9)
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set_ylabel('W→Z flow (fraction of true W jets, edited − unedited)')
    ax[0].set_title('(a) 0.06 rad translation of all particles: W jets newly called Z', fontsize=10)
    ax[0].legend(fontsize=7.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax[1].set_ylabel('test accuracy (260,000 jets) [%]'); ax[1].set_ylim(60, 83)
    ax[1].set_title('(b) accuracy per training cell', fontsize=10)
    fig.suptitle('X9w: translation response by input representation and training (sample(2000, 909), 2,000 W jets; dots = seeds, bars = 95 % CI)', fontsize=9.5)
    fig.tight_layout(); out = RES / 'x9w_repr_2x2.png'; fig.savefig(out, dpi=150); print('wrote', out)


if __name__ == '__main__':
    main()
