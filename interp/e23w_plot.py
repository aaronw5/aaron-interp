"""E23w figure: (left) odd/even gain ratio a1/a2 per stage, two samples, untrained redraws; (right) product slope per z unit
and the W-Z margin vs the random S-pair band.  python interp/e23w_plot.py (local, reads results/jedi/e23w_gains*.json)"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')) / 'jedi'
A = json.load(open(D / 'e23w_gains.json')); B = json.load(open(D / 'e23w_gains_verify.json'))
ST = ['S', 'h4', 'h5', 'z', 'margin']; LB = ['S\n(2nd-round sums)', 'h4', 'h5', 'z', 'W$-$Z\nlogit margin']
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1, 1.5]))
a = ax[0]; x = np.arange(len(ST))
for R, dx, c, nm in ((A, -0.12, '#1f5fa8', 'seed 2311 (N tail 1,250)'), (B, 0.12, '#d0661c', 'seed 2312 replicate (N tail 1,133)')):
    r = [R['gains']['tail'][s]['all']['ratio'] for s in ST]; lo = [R['gains']['tail'][s]['all']['ratio_ci'][0] for s in ST]; hi = [R['gains']['tail'][s]['all']['ratio_ci'][1] for s in ST]
    a.errorbar(x + dx, r, yerr=[np.subtract(r, lo), np.subtract(hi, r)], fmt='o', color=c, capsize=3, label=nm)
    for u in R['untrained']:
        v = [u['gains'][s]['ratio'] if u['gains'][s]['ratio'] is not None else np.nan for s in ST]
        a.plot(x + dx, v, 'x', color='0.6', ms=5)
a.plot([], [], 'x', color='0.6', label='untrained redraws (3 per sample)')
a.axhline(1, color='k', lw=1, ls='--'); a.text(4.45, 1.03, 'single function of $M_0M_2$: 1', ha='right', fontsize=8)
a.axhline(0, color='k', lw=0.6)
a.set_xticks(x); a.set_xticklabels(LB, fontsize=8); a.set_ylim(-0.9, 1.6)
a.set_ylabel('$a_1/a_2$  (centroid-linear gain / quadratic gain)')
a.set_title('Rigid translations 10-60 mrad of off-centre jets', fontsize=10); a.legend(fontsize=7, loc='lower right')
b = ax[1]; un = [f'z{i}' for i in range(16)] + ['margin']; x = np.arange(len(un))
for R, dx, c in ((A, -0.15, '#1f5fa8'), (B, 0.15, '#d0661c')):
    U = R['product']['units']; s = np.array([U[u]['slope'] for u in un]); lo = np.array([U[u]['ci'][0] for u in un]); hi = np.array([U[u]['ci'][1] for u in un])
    nl = np.array([U[u]['null_q025_q50_q975'][0] for u in un]); nh = np.array([U[u]['null_q025_q50_q975'][2] for u in un])
    b.vlines(x + dx, nl, nh, color=c, alpha=0.25, lw=7)
    b.errorbar(x + dx, s, yerr=[s - lo, hi - s], fmt='o', color=c, capsize=2, ms=4)
b.axhline(1, color='k', ls='--', lw=1); b.axhline(0, color='k', lw=0.6)
b.text(-0.6, 1.6, 'product $M_0M_2$: 1', ha='left', fontsize=8); b.text(-0.6, -2.4, 'additive: 0', ha='left', fontsize=8)
b.set_ylim(-25, 25); b.set_xticks(x); b.set_xticklabels([u.replace('margin', 'W$-$Z') for u in un], fontsize=8)
b.set_ylabel(r'mixed slope $d\log(\partial U/\partial\log M_2)/d\log M_0$')
b.set_title('Product test: network (points, 95 % CI) vs 200 random S-unit pairs (bands, 95 %)', fontsize=10)
fig.suptitle('JEDI-linear: the centroid-linear term enters below the quadratic term from S onward; no z unit departs from the random-pair band', fontsize=10.5)
fig.tight_layout(); fig.savefig(D / 'e23w_gains.png', dpi=130); print(D / 'e23w_gains.png')
