"""E28w figure: (a) the split on a real jet (pT per input slot before / after: the softer particles move down one slot);
(b) share of the split's call changes removed by each intervention (main sample, z 0.5, four split ranks) with the
size-matched null q95; (c) calls changed: split vs rotation control vs merging a real near-collinear pair.
Run: python -m interp.e28w_plot   (reads results/jedi/e28w_split_route.json; writes results/jedi/e28w_split_route.png)"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.loads((RES / 'e28w_split_route.json').read_text())
M = R['main']['edits']; RANKS = R['ranks']

fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1.5, 0.9]))
# (a)
ex = R.get('example')
if ex:
    n = ex['N'] + 1; s = np.arange(n)
    pb = np.array(ex['pt_before'][:n]); ps = np.array(ex['pt_split'][:n])
    ax[0].step(s, np.where(pb > 0, pb, np.nan), where='mid', color='0.5', lw=2, label=f'unsplit (call {ex["calls"]["unsplit"]})')
    ax[0].step(s, ps, where='mid', color='C3', lw=1.2, label=f'rank-13 particle split z = 0.5 (call {ex["calls"]["split"]})')
    for d in (ex['d1'], ex['d2']): ax[0].plot(d, ps[d], 'o', color='C3', ms=6)
    ax[0].set_yscale('log'); ax[0].set_xlabel('input slot (pT rank)'); ax[0].set_ylabel('particle pT [GeV]')
    ax[0].set_title(f'(a) {ex["label"]} jet, {ex["N"]} particles: halves (dots) at slots {ex["d1"]}, {ex["d2"]};\n'
                    f'softer particles move down one slot. Rank-restore patch: call {ex["calls"]["rank_PS"]}', fontsize=9)
    ax[0].legend(fontsize=8)
# (b)
items = [('a32 clamp (soft-pT cut)', lambda e: e['H_soft']['a32'], lambda e: e['H_soft']['null_other_l1']['q95']),
         ('rank restore (other particles in unsplit slots)', lambda e: e['H_rank']['rank_PS'], None),
         ('S19+S30 held (empty-slot sums)', lambda e: e['H_S']['S_named']['S19+S30'], lambda e: e['H_S']['S_named']['S19+S30']['null_q95']),
         ('g held (jet context)', lambda e: e['H_S']['g_all'], None)]
w = 0.8 / len(items)
for j, (nm, f, fq) in enumerate(items):
    xs = np.arange(len(RANKS)) + (j - len(items) / 2 + 0.5) * w
    d = [f(M[f'split_r{k}_z0.5']) for k in RANKS]
    v = np.array([q['removed'] for q in d]); ci = np.array([q['removed_ci'] for q in d])
    ax[1].bar(xs, v, w, color=f'C{j}', label=nm, yerr=[v - ci[:, 0], ci[:, 1] - v], capsize=2, ecolor='0.3')
    if fq:
        ax[1].scatter(xs, [fq(M[f'split_r{k}_z0.5']) for k in RANKS], marker='_', s=120, color='k', zorder=5)
ax[1].axhline(0, color='k', lw=0.6)
ax[1].set_xticks(np.arange(len(RANKS))); ax[1].set_xticklabels([f'rank {k}' for k in RANKS])
ax[1].set_ylabel('share of split call changes removed')
ax[1].set_title('(b) share of split call changes removed (z = 0.5, 7,500 jets, 95 % CI);\nblack tick = q95 of the size-matched null (other l1 units / all 171 pairs of live S)', fontsize=9)
ax[1].legend(fontsize=7, loc='lower right')
# (c)
sp = [M[f'split_r{k}_z0.5']['changed'] * 100 for k in RANKS]; ro = [M[f'rot_r{k}']['changed'] * 100 for k in RANKS]
xs = np.arange(len(RANKS))
ax[2].bar(xs - 0.2, sp, 0.4, color='C3', label='split z = 0.5'); ax[2].bar(xs + 0.2, ro, 0.4, color='0.6', label='rotation (same particle)')
dp = R['data_pair_merge']
ax[2].axhline(dp['changed'] * 100, color='C0', ls='--', label=f'merge closest real pair (dR < 0.01)')
ax[2].set_xticks(xs); ax[2].set_xticklabels([f'rank {k}' for k in RANKS]); ax[2].set_ylabel('calls changed [%]')
ax[2].set_title('(c) calls changed', fontsize=9); ax[2].legend(fontsize=7)
fig.suptitle('E28w: restoring the softer particles\' slots removes 8-34 % of the split\'s call changes at ranks 5-21 and adds changes at rank 0; clamping the 11 GeV cut a32 removes none', fontsize=10)
fig.tight_layout(); fig.savefig(RES / 'e28w_split_route.png', dpi=130)
print('wrote', RES / 'e28w_split_route.png')
