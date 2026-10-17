"""E20 figure: rank windows on one real jet + the across/inside swap response with and without the slot masks.
Reads $AI_RESULTS/jedi/e20_rank_windows.json and e20_programs.json; writes results/jedi/e20_rank_windows.png.
python -m interp.e20_plot  (matplotlib only)"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.load(open(RES / 'e20_rank_windows.json')); P = json.load(open(RES / 'e20_programs.json'))
BLUE, GRAY, INK, MUTED = '#2a6fdb', '#a0a7b0', '#1f2328', '#6a737d'

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15, 1.25]))
# (a) the jet
d = R['diagram_jet']; xp = np.array(d['particles']); n = len(xp)
a = ax[0]; rk = np.arange(n); inw = rk <= 13
a.scatter(xp[~inw, 1], xp[~inw, 2], s=8 + 4 * np.sqrt(xp[~inw, 0]) * 6, facecolors='none', edgecolors=GRAY, lw=1, label='ranks 14+')
a.scatter(xp[inw, 1], xp[inw, 2], s=8 + 4 * np.sqrt(xp[inw, 0]) * 6, color=BLUE, alpha=.8, label='ranks 0-13 (window of $S_{0}, S_{43}, S_{52}, S_{63}$)')
for r_ in (13, 14):
    a.annotate(str(r_), (xp[r_, 1], xp[r_, 2]), xytext=(5, 5), textcoords='offset points', fontsize=10, color=INK)
a.annotate('', xy=(xp[14, 1], xp[14, 2]), xytext=(xp[13, 1], xp[13, 2]),
           arrowprops=dict(arrowstyle='<->', color=INK, lw=1.2, shrinkA=6, shrinkB=6))
a.set_xlabel(r'$\Delta\eta$'); a.set_ylabel(r'$\Delta\phi$ (rad)')
a.set_title(f'(a) one W jet, {n} particles (marker area ~ $p_T$)\nedit A13: swap angles of ranks 13 and 14', fontsize=10, loc='left')
a.legend(fontsize=8, loc='lower left', frameon=False); a.set_aspect('equal', 'datalim')
# (b) windows
b = ax[1]
rows = [('S0', '0', 'pT only'), ('S43', '43', ''), ('S52', '52', ''), ('S63', '63', ''), ('S19', '19', 'pT only'),
        ('S1', '1', ''), ('S13', '13', ''), ('S17', '17', ''), ('S29', '29', ''), ('S30', '30', '')]
for i, (lab, c, note) in enumerate(rows):
    w = R['windows_S'][c]
    for seg in w.split(','):
        lo, hi = (int(v) for v in seg.split('-')) if '-' in seg else (int(seg), int(seg))
        b.barh(i, hi - lo + 1, left=lo - .5, height=.6, color=BLUE if c in ('0', '43', '52', '63') else GRAY)
    b.text(64.5, i, f'{w}' + (f'  ({note})' if note else ''), va='center', fontsize=8, color=MUTED)
for s_, ls in ((13.5, '-'), (5.5, ':')):
    b.axvline(s_, color=INK, lw=1, ls=ls)
b.text(14.2, 4.0, 'A13: 13|14', fontsize=8, color=INK, va='center'); b.text(6.0, 6.0, 'I5: 5|6', fontsize=8, color=INK, va='center')
b.set_yticks(range(len(rows))); b.set_yticklabels([r[0] for r in rows]); b.invert_yaxis()
b.set_xlim(-1, 84); b.set_xticks([0, 13, 32, 63]); b.set_xlabel('pT rank (slot)')
b.set_title('(b) rank windows (qs1 zero-bit masks); blue = edge at 13|14', fontsize=10, loc='left')
for s in ('top', 'right'): b.spines[s].set_visible(False)
# (c) response
c_ = ax[2]; units = ['43', '52', '63', '1']; W = .2; xs = np.arange(len(units))
bars = [('A13', 'mask', BLUE, 'A13 (across), trained masks'), ('A13', 'ones_all4', GRAY, 'A13, all slot masks = 1'),
        ('I5', 'mask', BLUE, 'I5 (inside), trained masks'), ('I5', 'ones_all4', GRAY, 'I5, all slot masks = 1')]
for k, (e, p, col, lab) in enumerate(bars):
    v = [P['edits'][e]['units'][u][p]['mean_abs'] for u in units]
    ci = np.array([P['edits'][e]['units'][u][p]['ci'] for u in units]).T
    c_.bar(xs + (k - 1.5) * W, v, W * .9, color=col, alpha=1 if e == 'A13' else .45, hatch=None if e == 'A13' else '//',
           edgecolor='white', label=lab, yerr=[np.array(v) - ci[0], ci[1] - np.array(v)], ecolor=INK, capsize=2)
c_.set_xticks(xs); c_.set_xticklabels([f'$S_{{{u}}}$' for u in units])
c_.set_ylabel(r'mean $|\Delta S|$ per jet (float + masks program)')
c_.set_title('(c) angle swap response, N = 9,951-10,000 jets, 95 % CI', fontsize=10, loc='left')
c_.legend(fontsize=8, frameon=False)
for s in ('top', 'right'): c_.spines[s].set_visible(False)
fig.suptitle('A swap across rank 13|14 moves the 0-13-window objects only while the slot masks are in place; a swap at 5|6 does not depend on them',
             fontsize=11, x=.01, ha='left')
fig.tight_layout()
out = ROOT / 'results' / 'jedi' / 'e20_rank_windows.png'; fig.savefig(out, dpi=130); print(out)
