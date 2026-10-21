"""X14 figure from $AI_RESULTS/jedi/x14_summary.json (local matplotlib):
(a) R vs epoch for runs that share initialisation and batch order but differ in schedule length (30 vs 100 epochs);
(b) R vs epoch, all trained Deep Sets / Linformer / transformer runs (fraction of training);
(c) grid: final R per (init seed, data-order seed) for tf-2b-d32 and linf-2b-d32-k8 with 95 % jet-bootstrap CIs.
    python interp/x14_plot.py [summary.json] [out.png]"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
f = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi' / 'x14_summary.json'
out = Path(sys.argv[2]) if len(sys.argv) > 2 else f.parent / 'x14_axis_training.png'
S = json.loads(f.read_text()); R = S['runs']
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
# (a)
pairs = [('tf-2b-d32-n64__seed0', 'tf-2b-d32-n64-e100__seed0', '#1f77b4', 'init/order seed 0'),
         ('tf-2b-d32-n64-p30__seed1', 'tf-2b-d32-n64-e100__seed1', '#d62728', 'init/order seed 1')]
for a, b, c, lab in pairs:
    for k, ls, l2 in ((a, '-', '30-epoch schedule'), (b, '--', '100-epoch schedule')):
        r = R[k]; e = np.array(r['epochs']); y = np.array(r['R']); lo = np.array([q[0] for q in r['R_ci']]); hi = np.array([q[1] for q in r['R_ci']])
        ax[0].plot(e, y, ls, color=c, label=f'tf-2b-d32, {lab}, {l2}'); ax[0].fill_between(e, lo, hi, color=c, alpha=.2)
ax[0].axhline(0, color='gray', lw=.6, ls=':'); ax[0].axhline(1, color='gray', lw=.6, ls=':')
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('R (read-out shift response / matched dilation)')
ax[0].set_title('(a) same initialisation and batch order,\nonly the learning-rate schedule length differs', fontsize=10)
ax[0].legend(fontsize=7)
# (b)
col = {'tf': '#d62728', 'linf': '#ff7f0e', 'dsets': '#2ca02c'}; seen = set()
for k, r in R.items():
    if '-x14d' in k: continue
    c = k.split('-')[0]; e = np.array(r['epochs'], float); ax[1].plot(e / e[-1], r['R'], color=col[c], lw=.9, alpha=.75,
                                                                    label={'tf': 'transformer', 'linf': 'Linformer', 'dsets': 'Deep Sets'}[c] if c not in seen else None); seen.add(c)
ax[1].axhline(0, color='gray', lw=.6, ls=':'); ax[1].axhline(1, color='gray', lw=.6, ls=':')
ax[1].set_xlabel('fraction of training (epoch / last epoch)'); ax[1].set_ylabel('R')
ax[1].set_title(f'(b) {sum(1 for k in R if "-x14d" not in k)} existing runs (N 32 / 64); oracle: true mass R = 0, axis moment R = 1', fontsize=10)
ax[1].legend(fontsize=8)
# (c)
G = S.get('grid', {})
for j, (m, c) in enumerate((('tf-2b-d32', '#d62728'), ('linf-2b-d32-k8', '#ff7f0e'))):
    if m not in G: continue
    g = G[m]; M = np.array(g['R']); C = np.array(g['R_ci'])
    for i in range(3):
        for d in range(3):
            x = j * 4 + i + (d - 1) * 0.22
            ax[2].errorbar(x, M[i, d], yerr=[[M[i, d] - C[i, d, 0]], [C[i, d, 1] - M[i, d]]], fmt=['o', 's', '^'][d], color=c, ms=5,
                           label=f'data-order seed {d}' if (j == 0 and i == 0) else None)
    ax[2].text(j * 4 + 1, 1.62, f'{m}\nshare init {g["share_init"]:.2f} (p {g["p_init_within_data"]:.2f})\n'
               f'order {g["share_data"]:.2f} (p {g["p_data_within_init"]:.2f}), rest {g["share_resid"]:.2f}', ha='center', va='bottom', fontsize=7.5)
ax[2].set_xticks([0, 1, 2, 4, 5, 6]); ax[2].set_xticklabels(['init 0', 'init 1', 'init 2'] * 2)
ax[2].set_ylabel('final R (epoch 30)'); ax[2].set_ylim(0.3, 1.85); ax[2].axhline(1, color='gray', lw=.6, ls=':')
ax[2].set_title('(c) 3 init seeds x 3 batch-order seeds, N = 64', fontsize=10)
ax[2].legend(fontsize=7, loc='lower center')
fig.tight_layout(); fig.savefig(out, dpi=130); print('wrote', out)
