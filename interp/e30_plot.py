"""E30 figure (local rendering from the job's JSON/npz): edit slope vs activation decile for S29, S1, S0 with the matched
controls, and one real test jet with the S29 edit marked.  python interp/e30_plot.py [results_dir]"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = Path(sys.argv[1] if len(sys.argv) > 1 else 'results') / 'jedi'
d = json.loads((R / 'e30_weak_poles.json').read_text())
fd = np.load(R / 'e30_fig_jet.npz')
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw=dict(width_ratios=[1.45, 1]))
lab = {'S29': 'S29: Σ hinge in (ln pT, r), all ranks',
       'S1': 'S1: Σ over ranks ≥ 1 of (r − r_on)₊, r_on ≈ 0.24',
       'S0': 'S0: ΣpT of the 14 hardest (direct term; context g adds the rest)'}
for (u, c), off in zip((('S29', 'C0'), ('S1', 'C1'), ('S0', 'C2')), (-0.12, 0, 0.12)):
    P = d['part2'][u]; D = [r for r in P['deciles'] if 'slope' in r]
    x = np.array([r['decile'] for r in D]) + off
    y = np.array([r['slope'] for r in D]); lo = np.array([r['ci'][0] for r in D]); hi = np.array([r['ci'][1] for r in D])
    ax[0].errorbar(x, y, yerr=[y - lo, hi - y], color=c, marker='o', ms=4.5, lw=1.6, capsize=2.5, label=lab[u])
    yc = np.array([r['control_over_pred'] for r in D])
    ax[0].plot(x, yc, color=c, ls=':', lw=1.2, marker='x', ms=4)
ax[0].plot([], [], color='0.4', ls=':', marker='x', label='matched control edit (formula predicts 0)')
ax[0].axhspan(0.9, 1.1, color='0.88', zorder=0)
ax[0].fill_between([], [], color='0.88', label='0.9–1.1: measured ≈ predicted')
ax[0].axhline(0, color='0.6', lw=0.8)
ax[0].set_xlabel('activation decile of the unedited unit (1 = weakest 10 % of jets, 10 = strongest)')
ax[0].set_ylabel('edit slope = measured ΔS / formula-predicted ΔS')
ax[0].set_xticks(range(1, 11)); ax[0].set_ylim(-0.15, 3.3); ax[0].set_xlim(0.5, 10.6)
ax[0].legend(fontsize=8, loc='center', bbox_to_anchor=(0.5, 0.6), frameon=False)
ax[0].set_title('Response to a one-particle edit, by activation decile (7,692 test jets)', loc='left', fontsize=10.5)

x = fd['jet_x']; real = x[:, 0] > 0; j = int(fd['jet_j']); a, b, c0 = fd['hinge']
A = ax[1]
A.scatter(x[real, 1], x[real, 2], s=6 + 9 * np.sqrt(x[real, 0]), c='0.55', alpha=0.55, edgecolor='none')
r0 = np.hypot(x[j, 1], x[j, 2]); th = np.arctan2(x[j, 2], x[j, 1]); r1 = r0 + float(fd['jet_dr'])
A.annotate('', xy=(r1 * np.cos(th), r1 * np.sin(th)), xytext=(x[j, 1], x[j, 2]), arrowprops=dict(arrowstyle='->', color='C3', lw=2))
A.scatter([x[j, 1]], [x[j, 2]], s=70, facecolor='none', edgecolor='C3', lw=1.6)
rb = -(c0 + a * np.log(max(x[j, 0], 1e-3))) / b
t = np.linspace(0, 2 * np.pi, 300); A.plot(rb * np.cos(t), rb * np.sin(t), 'C3--', lw=0.9)
A.text(0.02, 0.02, f'dashed circle: r where S29 switches on\nfor a {x[j, 0]:.1f} GeV particle (r = {rb:.3f})\nmarker area ∝ √pT',
       transform=A.transAxes, fontsize=8, color='0.3')
lim = max(0.42, r1 + 0.06); A.set_xlim(-lim, lim); A.set_ylim(-lim, lim); A.set_aspect('equal')
A.set_xlabel('Δη to jet axis'); A.set_ylabel('Δφ to jet axis')
A.set_title(f"One {'gqWZt'[int(fd['jet_label'])]} jet, S29 decile {int(fd['jet_decile'])}: one {x[j, 0]:.1f} GeV particle\n"
            f"moved r {r0:.3f} → {r1:.3f} rad at fixed pT and azimuth\n"
            f"ΔS29 predicted {float(fd['jet_pred']):+.3f}, measured {float(fd['jet_meas']):+.3f}", loc='left', fontsize=9.5)
fig.tight_layout(); fig.savefig(R / 'e30_weak_poles.png', dpi=130); plt.close(fig)
print('wrote', R / 'e30_weak_poles.png')
