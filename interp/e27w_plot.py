"""E27w figure: one real W jet before / after the arm-B azimuth-only edit (soft half, m -5 %), and the Z -> W flow for
arm B (hardest 14), its M1-held control and the matched mass dilation.
  python -m interp.e27w_plot   (reads $AI_RESULTS/jedi/e27w_mass_vs_m2.json, e27w_example.npz; writes e27w_mass_vs_m2.png)"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.loads((OUT / 'e27w_mass_vs_m2.json').read_text()); X = np.load(OUT / 'e27w_example.npz')
CL = ['g', 'q', 'W', 'Z', 't']
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
x, xb, mv = X['x'], X['xB'], X['moved'].astype(bool); re = x[:, 0] > 0
s = 8 + 400 * x[:, 0] / x[re, 0].max()
a = ax[0]
a.scatter(x[re & ~mv, 1], x[re & ~mv, 2], s=s[re & ~mv], c='0.55', alpha=.6, label='not moved')
a.scatter(x[mv, 1], x[mv, 2], s=s[mv], facecolors='none', edgecolors='tab:blue', label='moved particle, before')
a.scatter(xb[mv, 1], xb[mv, 2], s=s[mv], c='tab:orange', alpha=.8, label='moved particle, after')
for i in np.flatnonzero(mv):
    a.annotate('', xy=(xb[i, 1], xb[i, 2]), xytext=(x[i, 1], x[i, 2]), arrowprops=dict(arrowstyle='->', color='tab:orange', lw=.7, alpha=.7))
lim = 1.1 * np.abs(np.r_[x[re, 1:].ravel(), xb[re, 1:].ravel()]).max(); a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
a.axhline(0, c='0.85', lw=.5); a.axvline(0, c='0.85', lw=.5)
a.set_xlabel(r'$\Delta\eta$ to stored axis'); a.set_ylabel(r'$\Delta\phi$ to stored axis (rad)')
e = R['example']
a.set_title(f'True W jet: softer half turned about the axis (pT, r fixed)\n'
            f'm {e["m0"]:.0f} → {e["mB"]:.0f} GeV, $M_2$ unchanged, |M1|/M0 {e["M1_0"]:.4f} → {e["M1_B"]:.4f}; call {CL[e["call0"]]} → {CL[e["callB"]]}', fontsize=9.5)
a.legend(fontsize=8, loc='lower left')
b = ax[1]; V = [f'hard14_{d}' for d in (2, 5, 10)]
keys = [('B', 'arm B: hardest 14 turned toward M1 (m down, M2 fixed)', 'tab:orange'), ('ctrl', 'control: same particles turned, M1 fixed (m fixed)', 'tab:blue'),
        ('dil_m', 'real mass change of the same size (all angles scaled)', '0.3')]
w = 0.26
for j, (k, lab, col) in enumerate(keys):
    v = 100 * np.array([R['armB'][n][k]['ZW'] for n in V]); ci = 100 * np.array([R['armB'][n][k]['ZW_ci'] for n in V], dtype=float)
    pos = np.arange(len(V)) + (j - 1) * w
    b.bar(pos, v, w, color=col, label=lab); b.errorbar(pos, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt='none', c='k', lw=.8, capsize=2)
    for p_, v_ in zip(pos, v): b.text(p_, v_ + 1.5, f'{v_:.1f}', ha='center', fontsize=7.5)
b.set_xticks(np.arange(len(V))); b.set_xticklabels([f'jet mass −{n.split("_")[1]} %\n(N {R["armB"][n]["kept"]} jets, {R["armB"][n]["B"]["n_Z"]} Z-called)' for n in V], fontsize=8.5)
b.set_ylabel('Z-called jets called W after the edit (%)'); b.set_ylim(0, 100)
b.set_title('Z → W flow: azimuth-only edit vs control vs real mass change (95 % bootstrap CI)', fontsize=10)
b.legend(fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig(OUT / 'e27w_mass_vs_m2.png', dpi=130); print('wrote', OUT / 'e27w_mass_vs_m2.png')
