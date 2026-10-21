"""E35 figures from results/jedi/e35_prong_sharing_route[_smoke].json (local matplotlib, seconds).
  e35_edit.png   one real W jet (main sample) before / after D(+0.3) and the cn control: (deta, dphi) about the jet axis, marker area
                 proportional to pT, colour = kT prong; title per panel: call, W-Z margin, prong pT, jet mass.
  e35_route.png  share of D's W-Z margin response removed by each hold (main, 95 % jet-bootstrap CI) with the 95th percentile of
                 its size-matched random-hold null; response per dc for W and Z jets (network D, cn, cphi, R); per-cell shares of the holds.
Usage: python -m interp.e35_plot [--smoke]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results' / 'jedi'
sm = '--smoke' in sys.argv
R = json.loads((D / f'e35_prong_sharing_route{"_smoke" if sm else ""}.json').read_text())
M = R['main']; ex = M['example']

# ---------------------------------------------------------------- edit figure
fig, axs = plt.subplots(1, 3, figsize=(13, 5.3), sharex=True, sharey=True)
cols = ['#1f77b4', '#d95f02']
allp = np.concatenate([np.r_[ex['particles'][k]['deta'], ex['particles'][k]['dphi']] for k in ex['particles']])
lim = min(0.8, 1.08 * np.abs(allp).max())
titles = {'unedited': 'unedited W jet', 'D+0.3': r'D: $|\cos\theta^*|$ +0.3 (rest-frame turn)', 'cn+0.3': 'cn control: same turn about the prong axis'}
for ax, k in zip(axs, ('unedited', 'D+0.3', 'cn+0.3')):
    P = ex['particles'][k]; pt = np.array(P['pt']); pr = np.array(P['prong'])
    for q in (0, 1):
        s = pr == q
        ax.scatter(np.array(P['deta'])[s], np.array(P['dphi'])[s], s=4 + 900 * pt[s] / pt.max(), c=cols[q], alpha=0.65, edgecolors='k', linewidths=0.3,
                   label=f'prong {q + 1}: $p_T$ {ex[k]["prong_pt"][q]:.0f} GeV')
    ax.axhline(0, c='0.8', lw=0.6); ax.axvline(0, c='0.8', lw=0.6)
    ax.set_title(f'{titles[k]}\ncall {ex[k]["call"]}, $\\ell_W-\\ell_Z$ = {ex[k]["margin_WZ"]:+.2f}, jet m = {ex[k]["jet_m"]:.1f} GeV', fontsize=10)
    lg = ax.legend(loc='lower left', fontsize=8, frameon=False)
    for h in lg.legend_handles: h.set_sizes([40])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta\eta$ to jet axis')
axs[0].set_ylabel(r'$\Delta\phi$ to jet axis [rad]')
fig.suptitle(f'One W jet, same jet four-momentum (pT, η, φ, m) in all panels. D moves momentum from the softer to the harder prong '
             f'(|cos θ*| {ex["c0"]:.2f} → {ex["c1"]:.2f}); marker area ∝ particle pT', fontsize=10)
fig.tight_layout(rect=[0, 0.02, 1, 0.93]); fig.savefig(D / 'e35_edit.png', dpi=130); plt.close(fig)

# ---------------------------------------------------------------- route figure
H = M['holds']['results']; E = M['edits']; dcs = [float(d) for d in R['dcs']]; U = R.get('untrained', {})
SG = None
p_s = D / f'e35_singles{"_smoke" if sm else ""}.json'
if p_s.exists(): SG = json.loads(p_s.read_text())
fig, (a2, a1) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
for c, mk in (('W', 'o'), ('Z', 's')):
    for kind, col in (('D', 'k'), ('cn', '#7570b3'), ('cphi', '#66a61e'), ('R', '#e7298a')):
        v = [E[str(d)][kind][c]['dmargin'] for d in dcs]
        a2.plot(dcs, v, marker=mk, c=col, ls='-' if c == 'W' else '--', label=f'{kind}, {c} jets')
a2.axhline(0, c='0.7', lw=0.8); a2.set_xlabel(r'requested change of $|\cos\theta^*|$ (dc); dc > 0: more asymmetric prongs')
a2.set_ylabel(r'mean change of $\ell_W-\ell_Z$ (log-odds)')
a2.set_title('(a) W−Z margin response at fixed jet four-momentum (N = 2000 W, 2000 Z)', fontsize=10); a2.legend(fontsize=7.5, ncol=2, frameon=False, loc='lower left')
order = ['S29', 'h4dir', 'h5dir', 'mass', 'S1', 'S30', 'S19', 'radial', 'S0', 'S31', 'S58', 'named7', 'g']
lab = {'S29': 'S29 (pT-weighted r², mass-like)', 'h4dir': 'h4 mass direction', 'h5dir': 'h5 mass direction', 'mass': 'S29 + h4 mass direction',
       'S1': 'S1 (Σ (r−0.2)₊, ranks ≥ 1)', 'S30': 'S30 (Σ r, core)', 'S19': 'S19 (soft count)', 'radial': 'S1 + S30 + S19',
       'S0': 'S0 (pT of 14 hardest)', 'S31': 'S31 (dipole count)', 'S58': 'S58 (quadrupole count)', 'named7': 'all 7 named S', 'g': 'g (all 64 context units)'}
if SG:
    for cc in ('W', 'Z'):
        best = max(SG['singles'], key=lambda u: SG['singles'][u][f'0.3|{cc}']['share'])
        if best not in order: order.append(best); lab[best] = f'{best} (largest single hold, dc +0.3, {cc})'
cells = [('-0.3', 'W', '#1b9e77', 'dc −0.3, W jets'), ('0.3', 'W', '#d95f02', 'dc +0.3, W jets'), ('0.3', 'Z', '#7570b3', 'dc +0.3, Z jets')]
yv = np.arange(len(order))[::-1]
for yy, nm in zip(yv, order):
    for j, (dc, c, col, l) in enumerate(cells):
        if nm in H: v = H[nm]['per_dc'][dc][c]
        else: v = SG['singles'][nm][f'{dc}|{c}']['share']
        a1.plot([np.clip(v, -1.5, 1.5)], [yy + 0.22 * (1 - j)], marker='o' if abs(v) <= 1.5 else '>', c=col, ms=6, ls='none', label=l if yy == yv[0] else None)
    if nm in H:
        r = H[nm]; a1.errorbar(r['share'], yy - 0.33, xerr=[[r['share'] - r['ci'][0]], [r['ci'][1] - r['share']]], fmt='D', c='k', ms=4, capsize=2,
                               label='pooled over all (dc, class), 95 % CI' if yy == yv[0] else None)
a1.set_yticks(yv); a1.set_yticklabels([lab[n] for n in order], fontsize=9); a1.axvline(0, c='0.7', lw=0.8); a1.axvline(1, c='0.7', lw=0.8, ls=':')
a1.set_xlim(-1.6, 1.6); a1.set_xlabel('share of the D response removed by the hold (1 = all; clipped at ±1.5)')
a1.set_title('(b) Share of the D margin response removed by holding one element at its unedited value', fontsize=10); a1.legend(fontsize=7.5, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(D / 'e35_route.png', dpi=130); plt.close(fig)
print('wrote', D / 'e35_edit.png', D / 'e35_route.png')
