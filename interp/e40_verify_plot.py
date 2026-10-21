"""E40 verify figure: results/jedi/e40_verify.png from results/jedi/e40_verify.json (local matplotlib, seconds).
Margin change from adding 3 soft particles at fixed jet four-momentum, vs the radius r at which they are added."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.loads((RES / 'e40_verify.json').read_text()); E = R['edits']; H = R['holds']

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
a = ax[0]
for pt, col in (('0.5', '#7b4fd0'), ('0.05', '#b49be6')):
    rs = sorted(float(k.split('_r')[1]) for k in E if k.startswith(f'emit_k3_pt{pt}_r'))
    v = np.array([E[f'emit_k3_pt{pt}_r{r:g}']['dmargin'] for r in rs]); ci = np.array([E[f'emit_k3_pt{pt}_r{r:g}']['ci'] for r in rs])
    a.errorbar(rs, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt='o-', color=col, capsize=3, label=f'3 particles of {pt} GeV')
for r in (0.02, 0.3):
    c = E[f'comp_k3_r{r:g}']; a.errorbar([r], [c['dmargin']], yerr=[[c['dmargin'] - c['ci'][0]], [c['ci'][1] - c['dmargin']]],
                                         fmt='s', color='#9a9a9a', capsize=3, label='compensation only (no particles added)' if r == 0.02 else None)
a.axhline(0, color='k', lw=0.6); a.axvline(0.165, color='#3fa34d', lw=0.8, ls=':'); a.axvline(0.2, color='#3fa34d', lw=0.8, ls='--')
a.text(0.168, 0.12, 'S30 term\nzero beyond', fontsize=7.5, color='#3fa34d'); a.text(0.205, 0.05, 'S1 range\nr > 0.2', fontsize=7.5, color='#3fa34d')
a.set_xlabel('distance r of the added particles from the jet axis'); a.set_ylabel(r'mean change of $\log p(g)/p(q)$ (q and g jets)')
a.set_title(f'(a) soft emissions at fixed jet four-momentum; fresh sample seed 4502, N = {R["n_qg"]} q/g jets', fontsize=9.5)
a.legend(fontsize=8, loc='lower right')

b = ax[1]
es = [('emit_k3_pt0.5_r0.02', '0.5 GeV\nr 0.02'), ('emit_k3_pt0.05_r0.02', '0.05 GeV\nr 0.02'), ('emit_k3_pt0.5_r0.1', '0.5 GeV\nr 0.1'),
      ('emit_k3_pt0.5_r0.3', '0.5 GeV\nr 0.3'), ('emit_k3_pt0.05_r0.3', '0.05 GeV\nr 0.3'), ('split_soft', 'soft split'), ('merge2', 'merge 2\nsoftest')]
x = np.arange(len(es)); w = 0.27
for j, (u, lab, col) in enumerate([('S19', 'S19 (soft-pT hinge, ranks 13-63)', '#2a6fdb'), ('S1', r'S1 $\Sigma(r-0.2)_+$', '#3fa34d'),
                                   ('S30', r'S30 $\Sigma$ReLU(3.47-21r), empty slots', '#e08a1e')]):
    v = np.array([H[e]['singles'][u] for e, _ in es]); ci = np.array([H[e][f'{u}_ci'] for e, _ in es])
    b.bar(x + (j - 1) * w, np.clip(v, -0.5, 1.6), w, color=col, label=lab,
          yerr=[np.clip(v, -0.5, 1.6) - np.clip(ci[:, 0], -0.5, 1.6), np.clip(ci[:, 1], -0.5, 1.6) - np.clip(v, -0.5, 1.6)], capsize=2)
b.axhline(0, color='k', lw=0.6); b.set_xticks(x); b.set_xticklabels([e[1] for e in es], fontsize=8); b.set_ylim(-0.5, 1.7)
b.set_ylabel('share of margin change removed by the hold'); b.legend(fontsize=7.5, loc='upper left')
b.set_title('(b) hold one S object at its unedited value (clipped at 1.6)', fontsize=9.5)
fig.tight_layout(); fig.savefig(RES / 'e40_verify.png', dpi=130); print('wrote', RES / 'e40_verify.png')
