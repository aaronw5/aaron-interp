"""E36 figure: one W jet with the found smallest call-flipping edit (arrows), and where the edit size goes (found vs random).
python interp/e36_plot.py  (reads results/jedi/e36.json, e36_examples.npz; writes results/jedi/e36_edit.png)"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/jedi')
R = json.loads((D / 'e36.json').read_text()); E = np.load(D / 'e36_examples.npz')
k = int(np.argsort(E['eps'])[len(E['eps']) // 4])          # lower-quartile-size example among W->Z flips that transferred
x0, xe, v = E['x0'][k].astype(float), E['xe'][k].astype(float), E['v'][k].astype(float)
real = x0[:, 0] > 0; SC = 0.02
d_eta, d_phi = SC * v[real, 1], SC * v[real, 2]; dpt = np.expm1(v[real, 0])
fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw=dict(width_ratios=[1.05, 1]))
a = ax[0]; P = x0[real]; mv = np.hypot(d_eta, d_phi).max(); amp = max([a_ for a_ in (1, 2, 5, 10, 20) if a_ * mv <= 0.15] or [1])
sc = a.scatter(P[:, 1], P[:, 2], s=6 + 300 * P[:, 0] / P[:, 0].max(), c=100 * dpt, cmap='coolwarm', vmin=-30, vmax=30, edgecolor='#333', lw=0.4, zorder=2)
a.quiver(P[:, 1], P[:, 2], amp * d_eta, amp * d_phi, angles='xy', scale_units='xy', scale=1, width=0.004, color='#1c5cab', zorder=3)
for rr in (0.1, 0.2): a.add_patch(plt.Circle((0, 0), rr, fill=False, ls=':', color='#888'))
a.plot(0, 0, '+', color='k', ms=10)
lim = max(0.25, np.abs(P[:, 1:]).max() * 1.1); a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
a.set_xlabel('Δη to jet axis'); a.set_ylabel('Δφ to jet axis (rad)')
a.set_title(f'One W-called jet: smallest found edit to a Z call (ε = {E["eps"][k]:.1f})\njet pT, mass and axis unchanged; arrows = particle moves × {amp};\ncolour = pT change, marker area ∝ particle pT', fontsize=10, loc='left')
cb = fig.colorbar(sc, ax=a, shrink=0.8); cb.set_label('pT change of the particle (%)')
b = ax[1]; S = R['main']['norm_shares']
keys = ['coord: pT', 'coord: radial', 'coord: azimuthal', 'ranks 0-13', 'ranks 14-63', 'r < 0.1', '0.1 <= r < 0.2', 'r >= 0.2']
lab = ['pT', 'radial move', 'azimuthal move', 'pT ranks 0–13', 'pT ranks 14–63', 'r < 0.1', '0.1 ≤ r < 0.2', 'r ≥ 0.2']
yy = np.arange(len(keys))[::-1]
for off, src, col, nm in ((0.18, 'found', '#1c5cab', 'found flipping edit'), (-0.18, 'random', '#aaaaaa', 'random edit, same size')):
    m = np.array([S[kk][src]['mean'] for kk in keys]); ci = np.array([S[kk][src]['ci'] for kk in keys])
    b.barh(yy + off, m, height=0.34, color=col, label=nm, xerr=[m - ci[:, 0], ci[:, 1] - m], ecolor='#333', capsize=2)
b.set_yticks(yy); b.set_yticklabels(lab); b.set_xlabel('share of the edit size ε² (mean over W/Z/q/g jets)')
for yv in (5.5, 2.5): b.axhline(yv, color='#ccc', lw=0.8)
b.legend(loc='lower right', fontsize=9, frameon=False); b.set_xlim(0, 1)
b.set_title('Where the edit goes: coordinate, pT rank, radius', fontsize=10, loc='left')
for s in ('top', 'right'): b.spines[s].set_visible(False)
fig.suptitle('JEDI-linear: the smallest call-flipping edits at fixed jet pT, mass and axis', fontsize=12, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(D / 'e36_edit.png', dpi=140); print('wrote', D / 'e36_edit.png')
