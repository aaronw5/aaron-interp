"""E41 figure: (left) a real W jet before / after the hardest-14 azimuth turn (m -10 %, m_A fixed) and the 0.06 rad shift
(m fixed, m_A up); (right) held-out R^2 of Delta = log m - log m_A per stage: trained JEDI-linear vs the untrained / permuted
band and the retrained networks.  python interp/e41_plot.py [results_dir]"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'results' / 'jedi')
S = json.loads((D / 'e41_summary.json').read_text()); M = json.loads((D / 'e41_main.json').read_text())
EX = np.load(D / 'e41_example.npz')
STG = ['a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z', 'logits']

fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
x, xt, xs, mv = EX['x'], EX['x_turn'], EX['x_shift'], EX['moved'].astype(bool)
real = x[:, 0] > 0; sz = lambda p: 8 + 140 * p / p.max()  # noqa: E731
for a_, xe, ttl, k in ((ax[0], xt, 'turn: 14 hardest rotated about the axis', 'turn'), (ax[1], xs, 'shift: every particle moved 0.06 rad (random direction)', 'shift')):
    a_.scatter(x[real, 1], x[real, 2], s=sz(x[real, 0]), facecolors='none', edgecolors='0.55', lw=0.8, label='original')
    sel = real & (mv if k == 'turn' else real)
    a_.scatter(xe[sel, 1], xe[sel, 2], s=sz(x[sel, 0]), c='C3', alpha=0.7, label='edited')
    if k == 'turn':
        for i in np.flatnonzero(sel): a_.plot([x[i, 1], xe[i, 1]], [x[i, 2], xe[i, 2]], c='C3', lw=0.6, alpha=0.6)
    a_.plot(0, 0, '+', c='k', ms=12, mew=1.5); a_.set_aspect('equal')
    lim = 1.15 * max(np.abs(x[real, 1:]).max(), np.abs(xe[real, 1:]).max()); a_.set_xlim(-lim, lim); a_.set_ylim(-lim, lim)
    a_.set_xlabel(r'$\Delta\eta$ to stored axis'); a_.set_ylabel(r'$\Delta\phi$ to stored axis (rad)')
    a_.set_title(f'{ttl}\n' + fr'$m$ {EX["m_nat"]:.1f}$\to${EX[f"m_{k}"]:.1f} GeV,  $m_A$ {EX["mA_nat"]:.1f}$\to${EX[f"mA_{k}"]:.1f}', fontsize=10)
    a_.legend(fontsize=8, loc='lower right')

a_ = ax[2]; xs_ = np.arange(len(STG)); YL = -1.2
for key, col, lab, off in (('Delta_mlp_on_shift', 'C0', 'shifted jets (Δ moved by m_A)', -0.13), ('Delta_mlp_on_turn', 'C3', 'turned jets (Δ moved by m)', 0.13)):
    T = S['tests'][key]
    lo = np.array([T[s]['band_q05'] for s in STG]); hi = np.array([T[s]['band_q95'] for s in STG])
    a_.fill_between(xs_ + off, np.maximum(lo, YL), np.maximum(hi, YL), color=col, alpha=0.15, step='mid')
    tr = np.array([T[s]['trained'] for s in STG]); ci = np.array([T[s]['trained_ci'] for s in STG], float)
    a_.errorbar(xs_ + off, np.maximum(tr, YL), yerr=[np.maximum(tr, YL) - np.maximum(ci[:, 0], YL), np.maximum(ci[:, 1], YL) - np.maximum(tr, YL)],
                fmt='o', c=col, ms=5, capsize=2, label=f'trained, {lab}')
a_.fill_between([], [], [], color='0.5', alpha=0.2, label='untrained / permuted nets, 5-95 %')
b = S['baselines']['M0_M1_M2']
a_.axhline(M['baselines']['M0_M1_M2']['Delta']['mlp']['pool_on_turn']['r2'], ls='--', c='C3', lw=1, label=r'same read-out on input $(M_0,|M_1|,M_2)$, turned jets')
a_.axhline(0, c='k', lw=0.5); a_.axhline(YL, c='0.7', lw=0.5, ls=':')
a_.text(len(STG) - 0.5, YL + 0.03, 'values below -1.2 drawn at -1.2', ha='right', va='bottom', fontsize=7, color='0.4')
a_.set_xticks(xs_); a_.set_xticklabels(STG); a_.set_xlabel('network stage (input → output)')
a_.set_ylabel(r'held-out $R^2$ of $\Delta = \log m - \log m_A$ (MLP read-out)')
a_.set_title('Δ is readable when the axis is shifted, not when only the true mass changes', fontsize=10)
a_.legend(fontsize=7.5, loc='lower left', bbox_to_anchor=(0, 0.08)); a_.set_ylim(YL - 0.05, 1.05)
fig.tight_layout(); fig.savefig(D / 'e41_true_mass.png', dpi=130); print('wrote', D / 'e41_true_mass.png')
