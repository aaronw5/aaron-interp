"""E40 figure: results/jedi/e40_qg_count.png from results/jedi/e40_qg_count.json (local matplotlib, seconds)."""
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
tag = sys.argv[1] if len(sys.argv) > 1 else 'e40_qg_count'
R = json.loads((RES / f'{tag}.json').read_text())
E = R['edits']
C_COL, C_NC, C_ROT, C_RP = '#2a6fdb', '#e08a1e', '#9a9a9a', '#3fa34d'

fig, ax = plt.subplots(2, 2, figsize=(12, 9.4))
fig.suptitle('E40: JEDI-linear gluon-vs-quark margin responds to the particle count at fixed jet four-momentum', fontsize=12.5)

# (a) example gluon jet before / after a soft collinear split
a = ax[0, 0]; ex = R['example']
x0 = np.array(ex['x0']); xs = np.array(ex['x_split'])
a.scatter(x0[:, 1], x0[:, 2], s=6 + 4 * x0[:, 0], facecolors='none', edgecolors='k', lw=0.8, label=f'unedited, N = {ex["N"]}')
k = ex['slot']; par = x0[k]
same = np.all(np.isclose(xs[:, 1:], par[1:], atol=1e-6), axis=1)
a.scatter(xs[same, 1], xs[same, 2], s=6 + 4 * xs[same, 0], color=C_COL, alpha=0.6,
          label=f'split particle: {par[0]:.2f} GeV -> 2 x {par[0] / 2:.2f} GeV, same position')
a.annotate('split here', (par[1], par[2]), xytext=(par[1] + 0.12, par[2] + 0.12), arrowprops=dict(arrowstyle='->', color=C_COL), color=C_COL)
a.set_xlabel(r'$\Delta\eta$ to jet axis'); a.set_ylabel(r'$\Delta\phi$ to jet axis [rad]')
a.set_title(f'(a) real gluon jet (m = {ex["jet_mass"]:.0f} GeV), soft particle at rank {k} split in two\n'
            f'margin log p(g)/p(q) on this jet: {ex["m0"]:+.2f} -> {ex["m_split"]:+.2f}; mean over q/g jets '
            f'{R["edits"]["split_soft_f0.5"]["qg"]["dmargin"]:+.2f}', fontsize=9.5)
lim = max(0.45, np.abs(x0[:, 1:]).max() * 1.05); a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
a.legend(fontsize=8, loc='lower left')

# (b) margin change per edit, collinear vs matched controls
b = ax[0, 1]
groups = [('hardest\n(rank 0)', 'r0'), ('rank 13\n(13|14 edge)', 'r13'), ('soft\n(rank >= 30)', 'soft')]
xs_ = np.arange(len(groups)); w = 0.2
for j, (lab, key, col) in enumerate([('collinear split', 'split_{}_f0.5', C_COL), ('split, halves 0.05 apart', 'nc_{}_f0.5', C_NC),
                                     ('split, no rank shift (b)', 'rankpres_{}_f0.5', C_RP), ('rotation (N fixed)', 'rot_{}', C_ROT)]):
    v = [E[key.format(g)]['qg']['dmargin'] for _, g in groups]; ci = [E[key.format(g)]['qg']['dmargin_ci'] for _, g in groups]
    b.bar(xs_ + (j - 1.5) * w, v, w, color=col, label=lab, yerr=np.array([[vv - c[0], c[1] - vv] for vv, c in zip(v, ci)]).T, capsize=2)
for j, (lab, key, col) in enumerate([('+1 particle 0.5 GeV', 'emit_k1', '#7b4fd0'), ('+3', 'emit_k3', '#7b4fd0'),
                                     ('+3 at 0.05 GeV', 'emit_k3_pt0.05', '#b49be6'), ('merge 2 softest', 'merge2', '#c0392b'),
                                     ('move 2 softest together', 'move2', '#e8a39a')]):
    v = E[key]['qg']['dmargin']; c = E[key]['qg']['dmargin_ci']
    b.bar(3.1 + 0.62 * j, v, 0.5, color=col, yerr=[[v - c[0]], [c[1] - v]], capsize=2)
b.axhline(0, color='k', lw=0.6)
xt = list(xs_) + [3.1 + 0.62 * j for j in range(5)]
b.set_xticks(xt); b.set_xticklabels([g[0] for g in groups] + ['+1 part.\n0.5 GeV', '+3 part.\n0.5 GeV', '+3 part.\n0.05 GeV',
                                                             'merge 2\nsoftest', 'move 2\ntogether'], fontsize=7.5)
b.set_ylabel(r'mean change of $\log p(g)/p(q)$ (q and g jets)')
b.set_title('(b) splits (f = 0.5), emissions and merges; 95 % bootstrap CI', fontsize=10)
b.legend(fontsize=8, loc='lower left')

# (c) natural relation margin vs N on unedited jets
c = ax[1, 0]; nat = R['natural']; bins = nat['bins']
mid = np.array([(bb['lo'] + bb['hi']) / 2 for bb in bins])
for key, lab, col in (('mean_margin_g', 'gluon jets', '#c0392b'), ('mean_margin_q', 'quark jets', '#2a6fdb'), ('mean_margin', 'both', 'k')):
    v = np.array([bb[key] if bb[key] is not None else np.nan for bb in bins]); c.plot(mid, v, 'o-', color=col, label=lab, ms=4)
c.set_xlabel('number of particles N (unedited jets, fit sample)'); c.set_ylabel(r'mean $\log p(g)/p(q)$')
c.set_title(f'(c) natural slope: {nat["slope_pooled"]:.3f} per particle (both classes), '
            f'{nat["slope_within_class"]:.3f} within class', fontsize=10)
c.legend(fontsize=8)

# (d) share of the margin change removed by holding S objects at unedited values
d = ax[1, 1]; H = R['holds']
he = [('split_r0_f0.5', 'split\nrank 0'), ('split_r13_f0.5', 'split\nrank 13'), ('split_soft_f0.5', 'split\nsoft'),
      ('emit_k3', '+3 soft\nparticles'), ('merge2', 'merge 2\nsoftest')]
names = [('S19', 'S19 soft multiplicity', '#2a6fdb'), ('S0', 'S0 pT of 14 hardest', '#e08a1e'), ('S29', r'S29 $\Sigma p_T r^2$-like', '#9a9a9a'),
         ('S1', r'S1 $\Sigma (r-0.2)_+$ (not named in plan)', '#3fa34d')]
xx = np.arange(len(he)); w = 0.2
for j, (nm, lab, col) in enumerate(names):
    if nm == 'S1':
        v = [H[e]['singles']['S1'] for e, _ in he]; d.bar(xx + (j - 1.5) * w, v, w, color=col, label=lab, hatch='//', alpha=0.7)
    else:
        v = [H[e][nm]['share'] for e, _ in he]; ci = [H[e][nm]['ci'] for e, _ in he]
        d.bar(xx + (j - 1.5) * w, v, w, color=col, label=lab, yerr=np.array([[vv - cc[0], cc[1] - vv] for vv, cc in zip(v, ci)]).T, capsize=2)
q95 = [H[e]['single_null_q95'] for e, _ in he]
d.scatter(xx, q95, marker='_', s=500, color='k', zorder=5, label='95th pct of all 19 single live S objects')
d.axhline(0, color='k', lw=0.6); d.set_xticks(xx); d.set_xticklabels([h[1] for h in he], fontsize=9)
d.set_ylabel('share of margin change removed'); d.set_title('(d) hold one S object at its unedited value', fontsize=10)
d.legend(fontsize=7.5, loc='upper left')

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(RES / f'{tag}.png', dpi=130); print('wrote', RES / f'{tag}.png')
