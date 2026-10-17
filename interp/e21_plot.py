"""E21 figure: R = |response| along the iso-product curve / along the iso-sum line (S-level patches of S0 and S50 [+ S29]),
for the read-out units z0, z7, z14, z10 and the W - Z logit margin, network arms A, B against the random-pair control
(5-95 % of 20 draws) and against the fixed-weight additive prediction per tangent-angle bin.
Reads results/jedi/e21_isoproduct.json, writes results/jedi/e21_isoproduct.png.   Run: python -m interp.e21_plot
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
r = json.loads((D / 'e21_isoproduct.json').read_text())
T = ['z0', 'z7', 'z14', 'z10', 'W-Z margin']
LAB = ['z0\n(W, 75 GeV)', 'z7\n(Z, 94-98)', 'z14\n(Z)', 'z10\n(top)', 'W - Z\nlogit margin']
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
xs = np.arange(len(T))
ctrl = r['control_random_pairs']
cv = np.array([[d['named'].get(t, np.nan) for t in T] for d in ctrl])
lo, hi = np.nanpercentile(cv, 5, 0), np.nanpercentile(cv, 95, 0)
a = ax[0]
a.bar(xs, hi - lo, bottom=lo, width=0.6, color='0.85', label='random S-unit pairs given arm A steps (5-95 %, 20 draws)')
for arm, col, dx, nm in (('A', 'C0', -0.12, 'S0 with S50'), ('B', 'C3', 0.12, 'S0 with S50 + S29 (no pair control)')):
    nd = r['arms'][arm]['named']['all']
    R = [nd[t]['R'] for t in T]; ci = np.array([nd[t]['R_ci'] for t in T])
    a.errorbar(xs + dx, R, yerr=[np.array(R) - ci[:, 0], ci[:, 1] - np.array(R)], fmt='o', color=col, ms=7, capsize=3, label=f'network, {nm}')
a.axhline(1, color='k', lw=0.8, ls=':'); a.axhline(0.3, color='C2', lw=0.8, ls='--', label='prediction R < 0.3')
a.set_xticks(xs); a.set_xticklabels(LAB); a.set_ylim(0, 1.8)
a.set_ylabel(r'$R = \sum|\Delta U|_{\rm iso\text{-}product} \;/\; \sum|\Delta U|_{\rm iso\text{-}sum}$')
a.set_title('Arm A (S0, S50): inside the random-pair 5-95 % range except z7 (just below)\n(product rule: R = 0; equal-weight sum: R = inf)', fontsize=10)
a.legend(fontsize=7.5, loc='upper left')
b = ax[1]
ang = r['arms']['A'].get('angle')
if ang:
    th = ang['theta_deg_median_by_bin']
    for t, col in zip(['z7', 'W-Z margin', 'z0'], ['C1', 'C4', 'C0']):
        o = ang[t]['R_obs_by_theta_bin']; ci = np.array(ang[t]['R_obs_ci']); p = ang[t]['R_additive_pred_by_theta_bin']
        b.errorbar(th, o, yerr=[np.array(o) - ci[:, 0], ci[:, 1] - np.array(o)], fmt='o-', color=col, capsize=3, label=f'{t} measured')
        b.plot(th, p, 's--', color=col, alpha=0.6, mfc='none', label=f'{t} fixed-weight additive fit')
    b.axhline(0, color='k', lw=0.8); b.set_ylim(0, 1.3)
    b.set_xlabel(r'tangent angle of the iso-product step in $(z_{S0}, z_{S50})$ (deg, quartile medians)')
    b.set_ylabel('R')
    b.set_title('Arm A: R by direction of the iso-product step\n(product rule: R = 0 at every angle)', fontsize=10)
    b.legend(fontsize=7.5)
fig.suptitle(f'E21: S-level patches, {r["N"]} jets (seed {r["seed"]}), steps of 0.15-0.5 sd', fontsize=10)
fig.tight_layout()
fig.savefig(D / 'e21_isoproduct.png', dpi=130)
print('wrote', D / 'e21_isoproduct.png')
