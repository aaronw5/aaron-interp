"""E37x figure (redrawn in E37y): (a, b) one real W jet before / after Ez(+0.15) (harder-prong pT fraction z_h +0.15 at fixed
prong separation and prong masses), constituents coloured by kT prong, marker area ~ pT, prong pT shares printed; (c) mean change
of the h4 sharing coordinate u by jet-mass bin under Ez, Edr (same jet-mass change, Delta R12 moved instead of z), Em (dilation,
same mass change), Erot (rotation), from E37x; (d) direct steering of u inside the network (E37y arm 1): mean W-Z margin change
per unit u by 5-GeV jet-mass bin for W and Z jets, 95 % jet-bootstrap intervals.
Reads results/jedi/e37x.json and results/jedi/e37y.json; writes results/jedi/e37x.png.
Run locally: python interp/e37x_plot.py [e37x.json] [e37y.json]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'jedi' / 'e37x.json'
PY = Path(sys.argv[2]) if len(sys.argv) > 2 else P.parent / 'e37y.json'
R = json.loads(P.read_text()); RY = json.loads(PY.read_text())
BINS = ['65-75', '75-85', '85-95', '95-105']; DZ = '+0.15'
COL = dict(Ez='#c2410c', Edr='#2563eb', Em='#6b7280', Erot='#a3a3a3')
LAB = dict(Ez=r'$E_z$: $z_h$ +0.15 ($\Delta R_{12}$, prong masses fixed)', Edr=r'$E_{\Delta R}$: $\Delta R_{12}$ moved, same $\Delta m$',
           Em=r'$E_m$: all angles dilated, same $\Delta m$', Erot=r'$E_{rot}$: rotation (control)')
fig = plt.figure(figsize=(13.5, 8)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.42, wspace=0.32)
ex = RY.get('example')
lim = None
if ex:
    allx = np.r_[ex['unedited']['deta'], ex['Ez+0.15']['deta']]; ally = np.r_[ex['unedited']['dphi'], ex['Ez+0.15']['dphi']]
    lim = 1.15 * max(np.abs(allx).max(), np.abs(ally).max())
    ptmax = max(max(ex['unedited']['pt']), max(ex['Ez+0.15']['pt']))
for i, nm in enumerate(('unedited', 'Ez+0.15')):
    ax = fig.add_subplot(gs[0, i])
    if ex:
        d = ex[nm]; pt = np.array(d['pt']); lab = np.array(ex['prong']); sh = np.array(d['prong_pt']) / sum(d['prong_pt'])
        for k, c in ((0, '#7c3aed'), (1, '#059669')):
            s = lab == k
            ax.scatter(np.array(d['deta'])[s], np.array(d['dphi'])[s], s=4 + 700 * pt[s] / ptmax, c=c, alpha=0.75, edgecolors='k', linewidths=0.3,
                       label=f'prong {k + 1}: {100 * sh[k]:.0f} % of prong pT')
        o = d['obs']
        ax.set_title(f'{"unedited W jet" if i == 0 else "after $E_z$(+0.15)"}: $z_h$ = {o["zh"]:.2f}\n$\\Delta R_{{12}}$ = {o["dR"]:.3f}, '
                     f'$m$ = {o["m"]:.1f} GeV, call {d["call"]}\nW−Z margin {d["margin"]:+.2f}, $u$ = {d["u"]:+.2f}', fontsize=9.5)
        ax.legend(fontsize=8, loc='lower left')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r'$\Delta\eta$ from jet axis'); ax.set_ylabel(r'$\Delta\phi$ from jet axis' if i == 0 else '')
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=0.2)
tab = R['table']; xb = np.arange(len(BINS))
ax = fig.add_subplot(gs[1, 0:2])
for j, kd in enumerate(('Ez', 'Edr', 'Em', 'Erot')):
    m, lo, hi = [], [], []
    for b in BINS:
        ws = [tab.get(f'{kd}{DZ}|{b}|{c}', {}) for c in ('W', 'Z')]; n = [w.get('n', 0) for w in ws]
        vals = [w.get('du_das', {}) for w in ws]
        if sum(n) == 0 or any('mean' not in v for v, nn in zip(vals, n) if nn >= 5): m.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        wt = np.array([nn if 'mean' in v else 0 for v, nn in zip(vals, n)], float); wt /= wt.sum()
        m.append(sum(w_ * v['mean'] for w_, v in zip(wt, vals) if 'mean' in v)); lo.append(sum(w_ * v['ci'][0] for w_, v in zip(wt, vals) if 'mean' in v)); hi.append(sum(w_ * v['ci'][1] for w_, v in zip(wt, vals) if 'mean' in v))
    m, lo, hi = map(np.array, (m, lo, hi)); off = (j - 1.5) * 0.09
    ax.errorbar(xb + off, m, yerr=[m - lo, hi - m], fmt='o-', color=COL[kd], label=LAB[kd], capsize=3, lw=1.6)
ax.axhline(0, color='k', lw=0.6); ax.set_xticks(xb); ax.set_xticklabels([f'{b} GeV' for b in BINS])
ax.set_xlabel('unedited jet mass'); ax.set_ylabel(r'mean change of the h4 sharing coordinate $\Delta u$')
ax.set_title('(c) E37x: W+Z jets, edit size +0.15 (95 % jet-bootstrap CI)', fontsize=10); ax.legend(fontsize=8.5, loc='best')
ax = fig.add_subplot(gs[:, 2])
pb = RY['arm1_steering']['per_bin_5GeV']; fb = [f'{lo}-{lo + 5}' for lo in range(65, 105, 5)]; xc = np.array([lo + 2.5 for lo in range(65, 105, 5)])
for c, col, mk in (('W', '#0f766e', 'o'), ('Z', '#b45309', 's')):
    ok = [f'{b}|{c}' in pb for b in fb]
    m = np.array([pb[f'{b}|{c}']['slope'] if o_ else np.nan for b, o_ in zip(fb, ok)])
    ci = np.array([pb[f'{b}|{c}']['ci'] if o_ else [np.nan, np.nan] for b, o_ in zip(fb, ok)], float)
    nn = [pb[f'{b}|{c}']['n'] if o_ else 0 for b, o_ in zip(fb, ok)]
    ax.errorbar(xc + (0.4 if c == 'Z' else -0.4), m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt=mk + '-', color=col, capsize=3, lw=1.6, label=f'{c} jets')
    for x_, y_, n_ in zip(xc, m, nn):
        if n_: ax.annotate(str(n_), (x_ + (0.4 if c == 'Z' else -0.4), y_), textcoords='offset points', xytext=(0, 7 if c == 'Z' else -12), fontsize=6.5, color=col, ha='center')
ax.axvspan(80, 100, color='#e5e7eb', alpha=0.5, lw=0, label='mass-matched window (E37y)')
ax.axhline(0, color='k', lw=0.6); ax.set_xlabel('unedited jet mass [GeV]')
ax.set_ylabel(r'W−Z logit-margin change per unit of $u$ pushed into h4')
ax.set_title('(d) E37y: $u$ pushed directly inside the network\n(> 0: toward W; < 0: toward Z; numbers = jets per bin)', fontsize=10); ax.legend(fontsize=8.5, loc='lower right')
A = RY['arm1_steering']['matched']
fig.suptitle(f'Sharing coordinate $u$ in h4: follows the prong pT split; pushing it moves W and Z jets of the same mass differently '
             f'(80–100 GeV: W {A["W_matched"]:+.2f}, Z {A["Z_matched"]:+.2f} per unit $u$)', fontsize=11.5)
out = P.parent / 'e37x.png'; fig.savefig(out, dpi=130, bbox_inches='tight'); print('wrote', out)
