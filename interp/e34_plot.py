"""E34 figure: (a) a real gluon jet before / after each edit; (b) change of JEDI-linear's g-q margin per edit (edit - null) next to
the MLP references and the untrained network.  python -m interp.e34_plot  (reads $AI_RESULTS/jedi/e34_quark_gluon.json)"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
tag = sys.argv[1] if len(sys.argv) > 1 else 'e34_quark_gluon'
R = json.loads((RES / f'{tag}.json').read_text())
EDN = {'Ecount': 'count +5 %\n(soft splits)', 'Egirth': 'girth +5 %\n(M2 fixed)', 'EptD': 'pT^D +5 %\n(pT 2nd -> 1st)',
       'Emass': 'mass +5 %\n(girth fixed)'}
REF = {'Ecount': 'MLP_N', 'Egirth': 'MLP_girth', 'EptD': 'MLP_ptD', 'Emass': 'MLP_m'}

fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], hspace=0.45, wspace=0.3)
ex = R.get('example')
if ex:
    J0 = ex['jets']['orig']
    for i, e in enumerate(('orig', 'Ecount', 'Egirth', 'EptD', 'Emass')):
        ax = fig.add_subplot(gs[0, i]); J = ex['jets'][e]
        ax.scatter(J0['deta'], J0['dphi'], s=np.array(J0['pt']) * 1.2 + 4, facecolors='none', edgecolors='0.7', lw=0.8)
        ax.scatter(J['deta'], J['dphi'], s=np.array(J['pt']) * 1.2 + 4, c='C0' if e == 'orig' else 'C3', alpha=0.6)
        lim = max(0.45, 1.05 * max(np.abs(J0['deta']).max(), np.abs(J0['dphi']).max()))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal'); ax.tick_params(labelsize=7)
        ax.set_xlabel(r'$\Delta\eta$', fontsize=8)
        if i == 0: ax.set_ylabel(r'$\Delta\phi$ (rad)', fontsize=8)
        t = 'unedited gluon jet' if e == 'orig' else EDN[e].replace('\n', ' ')
        ax.set_title(f'{t}\nN {J["N"]:.0f}, girth {J["girth"]:.4f}\npT^D {J["pTD"]:.3f}, m {J["mass"]:.1f} GeV\n'
                     f'margin g-q {J["margin"]:+.2f}, call {J["call"]}', fontsize=7.5)
    fig.text(0.01, 0.97, '(a) one real gluon jet (marker area ~ particle pT; grey = unedited)', fontsize=10)

ax = fig.add_subplot(gs[1, :]); M = R['main']['edits']
eds = list(EDN); x = np.arange(len(eds)); w = 0.2
series = [('JEDI-linear', lambda e: (M[e]['edit_minus_null']['mean'], M[e]['edit_minus_null']['ci']), 'C0'),
          ('MLP on that variable alone', lambda e: (M[e]['mlp'][REF[e]]['edit_minus_null'], M[e]['mlp'][REF[e]]['ci']), 'C1'),
          ('MLP on all four', lambda e: (M[e]['mlp']['MLP_all']['edit_minus_null'], M[e]['mlp']['MLP_all']['ci']), 'C2'),
          ('untrained JEDI-linear', lambda e: (M[e].get('untrained', {}).get('edit_minus_null', 0), M[e].get('untrained', {}).get('ci', [0, 0])), '0.5')]
for k, (lab, f, c) in enumerate(series):
    v = [f(e) for e in eds]; m = np.array([a for a, _ in v]); ci = np.array([b for _, b in v])
    ax.bar(x + (k - 1.5) * w, m, w, color=c, label=lab, yerr=[m - ci[:, 0], ci[:, 1] - m], capsize=3, error_kw=dict(lw=0.8))
ax.axhline(0, color='k', lw=0.6); ax.set_xticks(x); ax.set_xticklabels([f'{EDN[e]}\n(n = {M[e]["n"]} jets)' for e in eds], fontsize=9)
ax.set_ylabel('change of g-q log-odds\n(edit minus matched null)', fontsize=9); ax.legend(fontsize=8, ncol=4, loc='upper right')
ax.set_title('(b) each edit moves one jet property by +5 % with the other three held within 1 %; > 0 = toward gluon '
             '(10,000 g/q test jets, seed 3401; 95 % bootstrap CI)', fontsize=9.5)
out = RES / f'{tag}.png'; fig.savefig(out, dpi=130, bbox_inches='tight'); print('wrote', out)
