"""E42x figure (re-render from results/jedi/e42x.json and e42x_scores.npz; seconds, local).
a: one real top jet, unedited vs W-rigid (W pair -> 60 GeV) vs N-matched (non-W pair, same displacement).
b: W - N per network and for the pair-blind / pair-aware GBDT references, three edit sets (primary Wr - ND).
Run: python interp/e42x_plot.py [results_dir]"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'results') / 'jedi'
R = json.loads((D / 'e42x.json').read_text()); Z = np.load(D / 'e42x_scores.npz'); ex = R['edit']['example']
NAMES = {'official_n64': 'JEDI-linear official n64', 'official_n32': 'official n32', 'perminv_n64': 'perm-inv n64', 'hgq150_s0': 'HGQ-r150 s0',
         'hgq150_s1': 'HGQ-r150 s1', 'hgq150_s2': 'HGQ-r150 s2', 'float_s0': 'float retrain s0', 'float_s1': 'float retrain s1', 'untrained_s0': 'untrained redraw'}
fig, ax = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(width_ratios=[1, 1.35]))
a = ax[0]; lab = Z['EX__lab']; cols = np.array(['tab:blue', 'tab:orange', 'tab:green'])
for v, mk, nm in (('id', 'o', 'unedited'), ('Wr', '^', 'W-rigid: W pair -> 60 GeV'), ('ND', 's', 'N-matched: non-W pair, same displacement')):
    X = Z[f'EX__{v}']; real = X[:, 0] > 0; n = int(real.sum()); c = cols[np.clip(lab[:n], 0, 2)]
    a.scatter(X[real, 1], X[real, 2], s=10 + 8 * X[real, 0] ** 0.7, c=c if v == 'id' else 'none', edgecolors=c, marker=mk, alpha=0.55 if v == 'id' else 0.9, lw=0.9,
              label=f'{nm}\n   W-pair mass {ex[v + "_mWc"]:.0f} GeV, jet mass {ex[v + "_m"]:.0f} GeV, tau32 {ex[v + "_tau32"]:.2f}')
a.set_xlabel(r'$\Delta\eta$ (about the jet axis)'); a.set_ylabel(r'$\Delta\phi$ (rad)')
a.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
a.set_title('a  one top jet: whole subjets translated, none resized\n(colour = subjet, marker area ~ pT)', fontsize=9.5, loc='left')
b = ax[1]; names = [n for n in NAMES if n in R['nets'] and 'error' not in R['nets'][n]]
keys = [('T60|Wr-ND', 'W pair 65-95 -> 60 GeV', 'tab:red'), ('T100|Wr-ND', 'W pair 65-95 -> 100 GeV', 'tab:purple'), ('T80|Wr-ND', 'off-peak W pair -> 80 GeV', 'tab:brown')]
rows = [NAMES[n] for n in names] + ['GBDT without pair masses', 'GBDT with pair masses']; yy = np.arange(len(rows))[::-1].astype(float)
for i, (k, lb, col) in enumerate(keys):
    vals = [(R['nets'][n]['sets'][k]['W_minus_N'], R['nets'][n]['sets'][k]['W_minus_N_ci']) for n in names]
    M0 = R['nets']['official_n64']['sets'][k]; vals += [(M0['refs'][r]['W_minus_N'], M0['refs'][r]['W_minus_N_ci']) for r in ('BLIND', 'AWARE')]
    v = np.array([q[0] for q in vals]); lo = np.array([q[1][0] for q in vals]); hi = np.array([q[1][1] for q in vals])
    b.errorbar(v, yy + (1 - i) * 0.25, xerr=[v - lo, hi - v], fmt='o', ms=4, color=col, label=lb, capsize=2)
b.axvline(0, color='k', lw=0.7); b.axhline(yy[len(names) - 1] - 0.5, color='grey', lw=0.5, ls=':')
b.set_yticks(yy); b.set_yticklabels(rows, fontsize=8.5)
b.set_xlabel('W - N (top log-odds): score change under the W-pair edit\nminus score change under the matched non-W edit')
b.set_title('b  an m_W reader gives W - N < 0 for 60 / 100 GeV and > 0 for the move onto 80 GeV\n(95 % bootstrap CI over jets)', fontsize=9.5, loc='left')
b.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=3, frameon=False)
fig.tight_layout(); p = D / 'e42x_wpair_rigid.png'; fig.savefig(p, dpi=130); print(p)
