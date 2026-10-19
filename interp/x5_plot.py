"""X5 figure: the four element statistics per training regime (float / L1 / HGQ Deep Sets, 3 seeds each).
    python -m interp.x5_plot            (reads results/jedi/x5_<regime>_s<seed>.json, writes results/jedi/x5_dsets_sparsity.png)"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
REG = [('float', 'float'), ('l1', 'L1\nsparsity\nmatched'), ('hgq', 'HGQ 3e-4\nmax\nsparsity'),
       ('hgq_acc', 'HGQ 1e-8\naccuracy\nmatched')]
COL = {'float': '#4C72B0', 'l1': '#DD8452', 'hgq': '#55A868', 'hgq_acc': '#2E7D4F'}
D = {(r, s): json.loads((RES / f'x5_{r}_s{s}.json').read_text()) for r, _ in REG for s in (0, 1, 2) if (RES / f'x5_{r}_s{s}.json').exists()}


def get(el, d):
    if el == 'a': return d['a']['P'], d['a']['P_ci'], d['a']['null_q']
    if el == 'b': return d['b']['max_contrast'], d['b']['max_contrast_ci'], None
    if el == 'c': return d['c']['frac'], d['c']['frac_ci'], None
    return d['d']['calibrated_ratio'], d['d']['calibrated_ratio_ci'], None


TIT = {'a': '(a) pure pT-only / angle-only cuts\namong live first-layer units',
       'b': '(b) largest edge contrast of the\nangle-swap response over pT ranks 1-15',
       'c': '(c) count-weighted radial objects\n(fraction of live pooled units)',
       'd': '(d) response to a 0.06 rad shift\n÷ response to a matched mass change'}
YL = {'a': 'fraction of live units', 'b': 'contrast $c_k$ (max over k)', 'c': 'fraction of live pooled units', 'd': 'calibrated read-out ratio'}
fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
for i, el in enumerate('abcd'):
    A = ax[i]
    for j, (r, lab) in enumerate(REG):
        for s in (0, 1, 2):
            if (r, s) not in D: continue
            v, ci, nq = get(el, D[(r, s)]); x = j + (s - 1) * 0.18
            if v is None: continue
            lo, hi = (ci if ci and ci[0] is not None else (v, v))
            A.errorbar(x, v, yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt='o', color=COL[r], ms=6, capsize=3,
                       label=None)
            if nq is not None:
                A.plot([x + 0.07, x + 0.07], [nq[0], nq[2]], color='0.6', lw=4, alpha=0.6, solid_capstyle='butt')
    A.set_xticks(range(len(REG))); A.set_xticklabels([l for _, l in REG], fontsize=7.5); A.set_title(TIT[el], fontsize=10)
    A.set_ylabel(YL[el], fontsize=9); A.grid(axis='y', alpha=0.3)
    if el == 'a':
        A.axhline(1.0, color='k', ls=':', lw=1); A.text(-0.45, 1.02, 'JEDI-linear 13/13', fontsize=7)
        A.plot([], [], color='0.6', lw=4, alpha=0.6, label='null 95 % range\n(weights permuted)'); A.legend(fontsize=7, loc='lower right')
        A.set_ylim(0, 1.08)
    if el == 'b':
        A.axhline(1.5, color='k', ls='--', lw=1); A.text(-0.4, 1.52, 'window threshold 1.5', fontsize=7)
        A.set_ylim(0.8, max(1.8, A.get_ylim()[1]))
    if el == 'c': A.set_ylim(-0.02, max(0.3, A.get_ylim()[1]))
    if el == 'd':
        A.axhline(1.0, color='k', ls=':', lw=1); A.axhline(0.0, color='k', lw=0.8)
        A.text(-0.4, 1.02, r'$\sqrt{M_0M_2}$ prediction', fontsize=7); A.text(-0.4, 0.02, 'invariant mass', fontsize=7)
        A.set_ylim(-0.1, 1.3)
fig.suptitle('Deep Sets dsets-d64, 3 seeds per regime, 10,000 test jets (sample seed 311); bars: 95 % jet-bootstrap intervals',
             fontsize=10)
fig.tight_layout()
out = RES / 'x5_dsets_sparsity.png'; fig.savefig(out, dpi=130); print(out)
