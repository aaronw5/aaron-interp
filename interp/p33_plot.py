"""P33 summary figure: smoothness ratio of the even pair-route response vs nearest-neighbour distance, network vs input ln dR.
  ./job.sh c-p33-plot 'python -m interp.p33_plot'   -> $AI_RESULTS/part/p33_smoothness.png
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

D = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
R = json.loads((D / 'p33b_input_geometry.json').read_text())
fig, a = plt.subplots(figsize=(7.2, 4.6))
labs = list(R['ParT_full']['cond']['pair64']['bins']); x = np.array([0.0017, 0.0055, 0.017, 0.039, 0.08])
off = dict(inp=0.90, full=0.97, kin64=1.03, kin32=1.10)
def pts(key, getter, **kw):
    v = np.array([getter(b)['v'] for b in labs]); lo = np.array([getter(b)['ci'][0] for b in labs]); hi = np.array([getter(b)['ci'][1] for b in labs])
    a.errorbar(x * off[key], v, yerr=[v - lo, hi - v], capsize=3, **kw)
pts('inp', lambda b: R['ParT_full']['cond']['pair64']['bins'][b]['inp'], color='k', marker='s', ls='-', label=r'input $\ln\Delta R$ alone (no network)')
pts('full', lambda b: R['ParT_full']['cond']['pair64']['bins'][b]['net'], color='C0', marker='o', ls='-', label='ParT_full, pair route')
pts('kin64', lambda b: R['ParT_kin']['cond']['pair64']['bins'][b]['net'], color='C1', marker='o', ls='-', label='ParT_kin, pair route (pair inputs in float64)')
pts('kin32', lambda b: R['ParT_kin']['cond']['pair']['bins'][b]['net'], color='C1', marker='o', mfc='none', ls=':', label='ParT_kin, pair route (float32, as deployed)')
a.axhline(0.09, color='grey', ls='--', lw=1, label='smooth response: $(0.003/0.01)^2=0.09$')
a.axvline(0.01, color='grey', ls=':', lw=0.8); a.text(0.0105, 14, 'step 0.01 rad', fontsize=8, color='grey')
a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(0.05, 30)
a.set_xticks(x); a.set_xticklabels(['<0.003', '0.003-0.01', '0.01-0.03', '0.03-0.05', '>0.05'])
a.set_xlabel(r'distance of the moved particle to its nearest neighbour $\Delta R_{\rm nn}$ [rad]')
a.set_ylabel('even/odd variance ratio, 0.003 rad step over 0.01 rad step')
a.set_title('The pair-route response is non-smooth only where the input ln ΔR is:\nfor particles with a neighbour closer than the step', fontsize=10)
a.legend(fontsize=7.5, loc='upper left'); fig.tight_layout(); fig.savefig(D / 'p33_smoothness.png', dpi=130); print('wrote', D / 'p33_smoothness.png')
