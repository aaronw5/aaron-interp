"""E24x figure: held-out R^2 of the context thresholds of S58 and S31 (and of S58, S31 themselves) when the g inputs are
named by count-weighted angular multipole sums of the particles.  python -m interp.e24x_plot [results_dir]"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')) / 'jedi'
R = json.loads((D / 'e24x_context.json').read_text())
lv = ['m0', 'm0+m1', 'm0+m1+m2']
lab = ['radial + pT sums\n(N, Σr, Σr², Σ ln pT)', '+ dipole sums\n(Σ Δη, Σ Δφ, ...)', '+ quadrupole sums\n(Σ r cos 2α, Σ r sin 2α, ...)']
fig, ax = plt.subplots(figsize=(8.2, 4.6)); xx = np.arange(len(lv))
for k, (u, c, nm) in enumerate(((58, '#3b6fb6', 'S58 threshold θ'), (31, '#d0743c', 'S31 threshold θ'))):
    y = [R['arm1'][f'S{u}']['theta_multipole']['levels'][l]['theta_r2'] for l in lv]
    b = ax.bar(xx + (k - 0.5) * 0.36, np.maximum(y, 0), 0.34, color=c, label=nm)
    for xi, yi in zip(xx + (k - 0.5) * 0.36, y):
        ax.text(xi, max(yi, 0) + 0.02, f'{yi:.2f}' if yi > 0 else '< 0', ha='center', fontsize=9, color='0.2')
ax.axhline(0.19, color='0.45', ls='--', lw=1)
ax.text(-0.45, 0.215, 'E24w: best fit of S58, S31 on pooled half-planes (R² ≤ 0.19)', fontsize=8.5, color='0.3')
ax.set_xticks(xx); ax.set_xticklabels(lab, fontsize=9); ax.set_ylim(0, 1.08)
ax.set_ylabel('held-out R² (N = 7,034 test jets)')
ax.set_title('S31 needs count-dipole sums, S58 count-quadrupole sums', fontsize=11)
ax.legend(frameon=False, loc='upper left'); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(D / 'e24x_context.png', dpi=130)
print('saved', D / 'e24x_context.png')
