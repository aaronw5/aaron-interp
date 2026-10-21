"""E37x figure: (a, b) one real W jet before / after Ez(+0.15) (harder-prong pT fraction z +0.15 at fixed prong separation and
prong masses), constituents coloured by kT prong, marker area ~ pT; (c) mean change of the h4 sharing coordinate u by jet-mass
bin under Ez, Edr (same jet-mass change, Delta R12 moved instead of z), Em (dilation, same mass change), Erot (rotation);
(d) mean W-Z logit-margin change by bin for W and Z jets.  Reads results/jedi/e37x.json; writes results/jedi/e37x.png.
Run locally: python interp/e37x_plot.py [path/to/e37x.json]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'jedi' / 'e37x.json'
R = json.loads(P.read_text())
BINS = ['65-75', '75-85', '85-95', '95-105']; DZ = '+0.15'
COL = dict(Ez='#c2410c', Edr='#2563eb', Em='#6b7280', Erot='#a3a3a3')
LAB = dict(Ez=r'$E_z$: $z$ +0.15 ($\Delta R_{12}$, prong masses fixed)', Edr=r'$E_{\Delta R}$: $\Delta R_{12}$ moved, same $\Delta m$',
           Em=r'$E_m$: all angles dilated, same $\Delta m$', Erot=r'$E_{rot}$: rotation (control)')
fig = plt.figure(figsize=(13, 7.6)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.38, wspace=0.32)
ex = R.get('example')
for i, nm in enumerate(('unedited', 'Ez+0.15')):
    ax = fig.add_subplot(gs[0, i])
    if ex:
        d = ex[nm]; pt = np.array(d['pt']); lab = np.array(ex['prong'])
        for k, c in ((0, '#7c3aed'), (1, '#059669')):
            s = lab == k; ax.scatter(np.array(d['deta'])[s], np.array(d['dphi'])[s], s=4 + 600 * pt[s] / pt.max(), c=c, alpha=0.75, edgecolors='k', linewidths=0.3,
                                     label=f'prong {k + 1}')
        o = d['obs']
        ax.set_title(f'{"unedited W jet" if i == 0 else "after $E_z$(+0.15)"}\n$z$ = {o["z"]:.2f}, $\\Delta R_{{12}}$ = {o["dR"]:.3f}, '
                     f'$m$ = {o["m"]:.1f} GeV\nW−Z margin {ex["margin"][nm]:+.2f}, $u$ = {ex["u"][nm]:+.2f}', fontsize=9.5)
        if i == 0: ax.legend(fontsize=8, loc='lower left')
    ax.set_xlabel(r'$\Delta\eta$ from jet axis'); ax.set_ylabel(r'$\Delta\phi$ from jet axis' if i == 0 else '')
    if ex:
        allx = np.r_[ex['unedited']['deta'], ex['Ez+0.15']['deta']]; ally = np.r_[ex['unedited']['dphi'], ex['Ez+0.15']['dphi']]
        h = 1.15 * max(np.abs(allx).max(), np.abs(ally).max()); ax.set_xlim(-h, h); ax.set_ylim(-h, h)
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
ax.set_title('W+Z jets, edit size +0.15 (95 % jet-bootstrap CI; approx. W/Z-weighted)', fontsize=10); ax.legend(fontsize=8.5, loc='best')
ax = fig.add_subplot(gs[:, 2])
for c, col, mk in (('W', '#0f766e', 'o'), ('Z', '#b45309', 's')):
    pb = R['interaction'][c]['per_bin']; m = np.array([pb.get(b, {}).get('a_du', np.nan) for b in BINS])
    ci = np.array([pb.get(b, {}).get('a_ci', [np.nan, np.nan]) for b in BINS], float)
    ax.errorbar(xb + (0.05 if c == 'Z' else -0.05), m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt=mk + '-', color=col, capsize=3, lw=1.6, label=f'{c} jets')
ax.axhline(0, color='k', lw=0.6); ax.set_xticks(xb); ax.set_xticklabels(BINS); ax.set_xlabel('unedited jet mass [GeV]')
ax.set_ylabel(r'W−Z margin change per unit $\Delta u$ (slope $a$, $\Delta\log m$ held in fit)')
ax.set_title('effect of the coordinate on the W/Z call\n(> 0: toward W; < 0: toward Z; all edits pooled)', fontsize=10); ax.legend(fontsize=9)
pz = R['readout']['das']['calibrated'].get('all|WZ', {}).get('pooled', {})
fig.suptitle(f'E37x: the h4 sharing coordinate $u$ follows the prong momentum fraction $z$ at fixed $\\Delta R_{{12}}$ '
             f'($R_z$ = {pz.get("R_z", float("nan")):.2f}, $\\Delta R_{{12}}$ edit {pz.get("R_Edr", float("nan")):.2f}); its effect on the call changes sign near 85 GeV', fontsize=11.5)
out = P.parent / 'e37x.png'; fig.savefig(out, dpi=130, bbox_inches='tight'); print('wrote', out)
