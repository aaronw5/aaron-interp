"""E24 figure (local, seconds): python -m interp.e24_plot -> results/jedi/e24_azimuth.png"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1] / 'results' / 'jedi'
S = json.loads((R / 'e24_summary.json').read_text()); D = json.loads((R / 'e24_main.json').read_text())
lab = {'jedi': 'JEDI-linear\nofficial', 'x12:hgq-r150_s0': 'JEDI HGQ\nr150 s0', 'x12:hgq-r150_s1': 'JEDI HGQ\nr150 s1*', 'x12:hgq-r150_s2': 'JEDI HGQ\nr150 s2',
       'x12:hgq_s0': 'JEDI HGQ\ns0', 'x12:hgq_s1': 'JEDI HGQ\ns1', 'x12:hgq_s2': 'JEDI HGQ\ns2', 'x12:float_s0': 'JEDI float\ns0', 'x12:float_s1': 'JEDI float\ns1',
       'dsets_s0': 'Deep Sets\ns0', 'dsets_s1': 'Deep Sets\ns1', 'pfn_s0': 'PFN\ns0', 'pfn_s1': 'PFN\ns1', 'tf_s0': 'Transf.\ns0', 'tf_s1': 'Transf.\ns1'}
ms = [m for m in lab if m in S['models']]
fig, ax = plt.subplots(2, 1, figsize=(13, 9.5), gridspec_kw=dict(height_ratios=[1.1, 1]))
ser = [('rot', 'rigid rotation (control)', '#8a8985'), ("A'", 'azimuth scramble, M0, M1, M2 held', '#2a78d6'),
       ('calib', 'real jets, Δτ21 ≈ scramble, matched m, M0, M2, pT\n(minus τ21-matched pairs)', '#eb6834')]
for k, (key, nm, col) in enumerate(ser):
    xs = np.arange(len(ms)) + (k - 1) * 0.26
    v = [S['models'][m][key] for m in ms]; mu = np.array([100 * a['mean'] for a in v]); lo = np.array([100 * a['ci'][0] for a in v]); hi = np.array([100 * a['ci'][1] for a in v])
    ax[0].bar(xs, mu, 0.24, color=col, label=nm); ax[0].errorbar(xs, mu, [mu - lo, hi - mu], fmt='none', ecolor='#0b0b0b', lw=1, capsize=2)
ax[0].set_xticks(range(len(ms))); ax[0].set_xticklabels([lab[m] for m in ms], fontsize=8.5)
ax[0].set_ylabel('calls changed [% of 10,000 jets]'); ax[0].set_ylim(0, 16.5); ax[0].legend(fontsize=9, frameon=False, loc='upper left')
ax[0].set_title('Scrambling every particle\'s azimuth (W: τ21 0.26 → 0.72) changes 6–9 % of calls in JEDI-linear, Deep Sets and PFN, 14 % in the transformer', fontsize=10.5)
ax[0].spines[['top', 'right']].set_visible(False); ax[0].grid(axis='y', color='#e4e3df', lw=0.6); ax[0].set_axisbelow(True)
ax[0].text(0.995, 0.97, '* collapsed seed (57 % accuracy); bars: 95 % jet-bootstrap intervals', transform=ax[0].transAxes, ha='right', va='top', fontsize=8, color='#52514e')
I = D['models']['jedi']['internals']['per_edit']; units = sorted(I["A'"]['S'], key=lambda u: -I["A'"]['S'][u]['dS_over_sd']['mean'])
radial = {'S29', 'S1', 'S0', 'S19', 'S30'}
for k, (e, nm, col) in enumerate([('rot', 'rigid rotation', '#8a8985'), ('Br', 'within-sector scramble (rotated cells)', '#1baf7a'), ("A'", 'azimuth scramble, M0, M1, M2 held', '#2a78d6')]):
    xs = np.arange(len(units)) + (k - 1) * 0.26; a = [I[e]['S'][u]['dS_over_sd'] for u in units]
    mu = np.array([x['mean'] for x in a]); lo = np.array([x['ci'][0] for x in a]); hi = np.array([x['ci'][1] for x in a])
    ax[1].bar(xs, mu, 0.24, color=col, label=nm); ax[1].errorbar(xs, mu, [mu - lo, hi - mu], fmt='none', ecolor='#0b0b0b', lw=1, capsize=2)
ax[1].set_xticks(range(len(units))); ax[1].set_xticklabels([u + ('\nradial' if u in radial else '') for u in units], fontsize=8.5)
ax[1].set_ylabel('mean |ΔS| / sd(S) per jet'); ax[1].legend(fontsize=9, frameon=False, loc='upper right')
ax[1].set_title('JEDI-linear second-round objects: the scramble moves the context-only units S58, S31; the radial objects change ≤ 0.2 sd (S0, S19: 0)', fontsize=10.5)
ax[1].spines[['top', 'right']].set_visible(False); ax[1].grid(axis='y', color='#e4e3df', lw=0.6); ax[1].set_axisbelow(True)
fig.suptitle('E24: azimuth scrambling at fixed pT, r and pT rank; sample(2000, 2401), N = 10,000 test jets', fontsize=11)
fig.tight_layout(); fig.savefig(R / 'e24_azimuth.png', dpi=130); print(R / 'e24_azimuth.png')
