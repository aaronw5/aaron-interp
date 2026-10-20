"""E23x figure: share of the odd (direction-dependent) response to a 30-60 mrad rigid shift removed by freezing S31+S58,
all of g, compared with the band of 136 random pairs of other live S units.  Reads $AI_RESULTS/jedi/e23x_centre.json.
  python -m interp.e23x_plot  -> results/jedi/e23x_centre.png"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.loads((RES / 'e23x_centre.json').read_text())
st = ['h4', 'h5', 'z', 'margin']; lab = ['h4', 'h5', 'z', 'W−Z margin']
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
x = np.arange(len(st)); w = 0.26
band = R['null_band']['pairs_all']
blo = np.array([band[s]['lo'] for s in st]) * 100; bhi = np.array([band[s]['hi'] for s in st]) * 100
ax.bar(x, bhi - blo, 0.9, bottom=blo, color='#d9d9d9', zorder=1, label='95 % band, 136 random pairs of other live S units')
arms = [('S31_S58', 'S31 + S58 frozen', '#2a6fdb', -w / 2), ('g_all', 'all 64 g units frozen', '#e07b39', w / 2)]
for nm, l, c, dx in arms:
    v = np.array([R['freeze'][nm]['pooled'][s]['odd_removed']['share'] for s in st]) * 100
    lo = np.array([R['freeze'][nm]['pooled'][s]['odd_removed']['ci'][0] for s in st]) * 100; hi = np.array([R['freeze'][nm]['pooled'][s]['odd_removed']['ci'][1] for s in st]) * 100
    ax.bar(x + dx, v, w, color=c, label=l, zorder=2); ax.errorbar(x + dx, v, yerr=[v - lo, hi - v], fmt='none', ecolor='k', lw=1, capsize=2, zorder=3)
ax.axhline(50, color='k', ls='--', lw=1); ax.text(3.45, 51.5, 'H1: > 50 %', ha='right', fontsize=8)
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel('odd shift response removed (%)'); ax.set_ylim(-40, 60)
ax.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0.0, 0.06), frameon=False)
ax.set_title('Freezing S31 + S58 removes about a quarter of the odd response at h4–z, none at the margin', fontsize=9)
ax.text(0.01, 0.02, f'JEDI-linear, N {R["N_test"]} jets (seed 2311); rigid shifts 30, 60 mrad × 8 azimuths; bars: 95 % jet bootstrap', transform=ax.transAxes, fontsize=7, color='#555')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(RES / 'e23x_centre.png'); print(RES / 'e23x_centre.png')
