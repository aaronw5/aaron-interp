"""E28 figure: (a) % of JEDI-linear calls changed by a collinear split of the particle of pT rank k (z 0.5, 0.8) vs the matched-rank
control (angles of ranks k, k+1 exchanged), soft additions, and scale references; (b) a real gluon jet before / after one split.
python -m interp.e28_plot  (reads results/jedi/e28_irc_split.json, writes results/jedi/e28_irc_split.png)"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.loads((RES / 'e28_irc_split.json').read_text()); E = R['edits']
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.35, 0.9, 1.1]))

a = ax[0]; ranks = ['0', '5', '13', 'soft']; lab = ['rank 0', 'rank 5', 'rank 13', 'rank > 20']
series = [('split z=0.5', 'split_r{}_z0.5', '#2b6cb0'), ('split z=0.8', 'split_r{}_z0.8', '#63a4e0'),
          ('matched-rank control (swap with rank k+1)', 'swap_r{}', '#999999')]
w = 0.26
for i, (nm, pat, col) in enumerate(series):
    v = np.array([E[pat.format(r)]['changed'] for r in ranks]) * 100
    ci = np.array([E[pat.format(r)]['changed_ci'] for r in ranks]).T * 100
    xs = np.arange(4) + (i - 1) * w
    a.bar(xs, v, w, color=col, label=nm); a.errorbar(xs, v, yerr=[v - ci[0], ci[1] - v], fmt='none', ecolor='k', capsize=2, lw=0.8)
ds = E['drop_softest']['changed'] * 100
a.axhline(ds, color='#c05621', ls='--', lw=1, label='scale: remove the softest particle (N - 1)')
a.set_xticks(range(4)); a.set_xticklabels(lab); a.set_ylabel('calls changed (%)')
a.set_title('(a) collinear split of one particle\n(all pT-weighted moments fixed)', fontsize=10); a.legend(fontsize=8, loc='upper right')

b = ax[1]; ns = [1, 2, 3, 4, 5]
v = np.array([E[f'soft_add{n}']['changed'] for n in ns]) * 100; ci = np.array([E[f'soft_add{n}']['changed_ci'] for n in ns]).T * 100
b.bar(ns, v, 0.6, color='#38a169'); b.errorbar(ns, v, yerr=[v - ci[0], ci[1] - v], fmt='none', ecolor='k', capsize=2, lw=0.8)
p0 = E['pad0_at']['changed'] * 100; b.axhline(p0, color='#666666', ls=':', lw=1, label='null: pT = 0 entry in padding')
b.legend(fontsize=8, loc='upper left')
b.set_xlabel('number of added 0.5 GeV particles'); b.set_ylabel('calls changed (%)'); b.set_title('(b) soft particles added', fontsize=10)
b.set_ylim(0, max(v.max(), (E['split_r0_z0.5']['changed'] * 100)) * 1.25)

c = ax[2]; ex = R.get('example')
if ex:
    B0 = np.array(ex['before']); A1 = np.array(ex['after'])
    for X, mk, col, lw, nm in ((B0, 'o', 'none', 1.0, 'before'), (A1, 'o', None, 0, 'after')):
        pass
    sz = lambda p: 8 + 400 * p / B0[:, 0].max()
    c.scatter(B0[:, 2], B0[:, 1], s=sz(B0[:, 0]), facecolors='none', edgecolors='#555555', lw=0.8, label='before split')
    rk = np.arange(len(A1)); inw = rk < 14
    c.scatter(A1[inw, 2], A1[inw, 1], s=sz(A1[inw, 0]), c='#2b6cb0', alpha=0.45, label='after: ranks 0-13 (S0 window)')
    c.scatter(A1[~inw, 2], A1[~inw, 1], s=sz(A1[~inw, 0]), c='#dd6b20', alpha=0.45, label='after: ranks 14+')
    k = ex['slot']; same = np.flatnonzero((A1[:, 1] == B0[k, 1]) & (A1[:, 2] == B0[k, 2]))
    c.scatter([B0[k, 2]], [B0[k, 1]], marker='x', c='k', s=40, label=f'split particle (rank {k}, {B0[k, 0]:.1f} GeV)')
    c.annotate(f'rank {k}, {B0[k, 0]:.1f} GeV -> two {A1[same[0], 0]:.1f} GeV\ndaughters at ranks {same[0]}, {same[-1]} (outside 0-13)',
               (B0[k, 2], B0[k, 1]), xytext=(12, -32), textcoords='offset points', fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.6))
    S0, S1 = ex['S_before'], ex['S_after']; sd = ex['sd_S']
    txt = (f"call {ex['call_before']} (p {max(ex['p_before']):.2f}) -> {ex['call_after']} (p {max(ex['p_after']):.2f})\n"
           + 'ΔS/sd: ' + ', '.join(f"S{u} {(S1[u] - S0[u]) / sd[u]:+.2f}" for u in ('0', '19', '29', '30')))
    c.text(0.02, 0.02, txt, transform=c.transAxes, fontsize=7.5, va='bottom')
    c.set_xlabel('Δφ to jet axis (rad)'); c.set_ylabel('Δη to jet axis (rad)'); c.legend(fontsize=7, loc='upper right')
    c.set_title(f"(c) real gluon jet, {ex['edit'].replace('_', ' ')}", fontsize=10)
fig.suptitle(f"JEDI-linear (n64, 3 features): calls changed by exact collinear splits and soft additions, "
             f"{R['n_sub']} test jets with ≤ 60 particles (seed 1202); bars 95 % bootstrap CI", fontsize=10)
fig.tight_layout(); fig.savefig(RES / 'e28_irc_split.png', dpi=130); print('wrote', RES / 'e28_irc_split.png')
