"""E25 figure: W -> Z fraction (true W jets) under each input edit, three official JEDI-linear n64 models, main sample.
python interp/e25_plot.py  (reads results/jedi/e25_main.json, writes results/jedi/e25_edits.png)"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.loads((RES / 'e25_main.json').read_text())['S4']
ARMS = [('R', 'rotation about\naxis (control)'), ('T', 'translate 0.06 rad\n(all columns)'), ('T_dRonly', 'only ΔR\ncolumn edited'),
        ('T_dR', 'all but ΔR\nedited'), ('T_rel', 'only Δη, Δφ\nedited'), ('Tabs', 'absolute η\n+0.06'), ('Tphi', 'absolute φ\n+1 rad'),
        ('dilM2', 'dilation with\nM2 of T')]
MOD = [('16f', '16-feature', '#2a78d6'), ('16f_perminv', '16-feature perm.-inv.', '#1baf7a'), ('3f', '3-feature (pT, Δη, Δφ)', '#9a9a94')]
fig, ax = plt.subplots(figsize=(10, 4.2)); w = 0.26; x = np.arange(len(ARMS))
for i, (m, lab, col) in enumerate(MOD):
    v = np.array([R[m]['arms'][a]['W_to_Z'] for a, _ in ARMS]); ci = np.array([R[m]['arms'][a]['boot'] for a, _ in ARMS])
    ax.bar(x + (i - 1) * w, 100 * v, w, color=col, label=lab, yerr=100 * np.abs(ci.T - v), capsize=2, error_kw=dict(lw=0.8))
ax.axhline(100 * R['16f']['W_to_Z_unedited'], color='k', lw=0.6, ls=':')
ax.set_xticks(x); ax.set_xticklabels([l for _, l in ARMS], fontsize=8.5)
ax.set_ylabel('true W jets called Z (%)'); ax.set_ylim(0, 85)
ax.set_title('16-feature JEDI-linear: the response to moving particles off the jet axis goes through the ΔR column only', fontsize=10.5)
ax.legend(fontsize=8.5, frameon=False, loc='upper right'); ax.spines[['top', 'right']].set_visible(False)
fig.text(0.01, 0.01, 'N = 3,000 true W jets per bar (sample(3000, 2501)); bars: 95 % bootstrap intervals; dotted line: unedited 16-feature (1.5 %)', fontsize=7.5)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(RES / 'e25_edits.png', dpi=150); print('wrote', RES / 'e25_edits.png')
