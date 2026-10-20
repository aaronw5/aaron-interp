"""X9w-verify figure: translation response at 0.06 rad (inside the augmentation range |c| <= 0.08) and 0.12 rad (outside it),
stored-axis vs augmented networks, on sample(2000, 2610).  Reads results/jedi/x9w_verify.json, writes results/jedi/x9w_verify.png.
    python -m interp.x9w_verify_plot          (local matplotlib is fine: seconds)
"""
import json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
ARCH = [('pfn', 'PFN'), ('efn', 'EFN'), ('dsets', 'Deep Sets'), ('jedi', 'JEDI-linear\nHGQ-50')]
SER = [('stored', 'T', 'stored axis, 0.06 rad', '#2b6cb0'), ('stored', 'T12', 'stored axis, 0.12 rad', '#90cdf4'),
       ('aug', 'T', 'aug |c|≤0.08, 0.06 rad', '#c05621'), ('aug', 'T12', 'aug |c|≤0.08, 0.12 rad', '#f6ad55')]


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((RES / 'x9w_verify.json').read_text())['models']
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6)); w = 0.2
    for k, (metric, lab) in enumerate([('flow', 'W→Z flow (edited − unedited, true W jets)'), ('calls_changed', 'fraction of all jets whose call changes')]):
        for j, (a, al) in enumerate(ARCH):
            for i, (c, ed, sl, col) in enumerate(SER):
                ms = [v for v in R.values() if v.get('arch') == a and v.get('cell') == c and 'edits' in v]
                vals = [m['edits'][ed][metric] for m in ms]; xp = j + (i - 1.5) * w
                ax[k].bar(xp, np.mean(vals), w * 0.92, color=col, label=sl if j == 0 else None)
                ax[k].scatter([xp] * len(vals), vals, s=10, color='k', zorder=3)
                if metric == 'flow':
                    for m in ms:
                        lo, hi = m['edits'][ed]['flow_ci']; ax[k].plot([xp, xp], [lo, hi], color='k', lw=0.8)
                else:
                    rot = [m['edits']['rot']['calls_changed'] for m in ms]
                    ax[k].scatter([xp] * len(rot), rot, marker='_', s=80, color='0.4', zorder=3, label='rotation null' if (j, i) == (0, 0) else None)
        ax[k].set_xticks(range(len(ARCH))); ax[k].set_xticklabels([al for _, al in ARCH], fontsize=9)
        ax[k].axhline(0, color='k', lw=0.6); ax[k].set_ylabel(lab, fontsize=9)
    ax[0].set_title('(a) W jets newly called Z under a rigid +Δη translation', fontsize=10)
    ax[1].set_title('(b) calls changed (all 10,000 jets)', fontsize=10)
    ax[0].legend(fontsize=7.5, loc='upper left', frameon=False); ax[1].legend(fontsize=7.5, loc='upper left', frameon=False)
    fig.suptitle('sample(2000, 2610), N = 10,000; dots = networks (seeds), whiskers = 95 % jet-bootstrap CI', fontsize=9)
    fig.tight_layout(); out = RES / 'x9w_verify.png'; fig.savefig(out, dpi=150); print('wrote', out)


if __name__ == '__main__':
    main()
