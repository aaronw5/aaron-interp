"""E25x figure: effective IIA (donor design, k = 2) of M0 and M2 swaps at the per-particle stages a, u and the pooled stages g, S,
3-feature and 16-feature JEDI-linear n64, with the 95th percentile of the non-trained baselines and the random-subspace 97.5 % line.
   python interp/e25x_plot.py [results_dir]   -> results/jedi/e25x.png
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'results' / 'jedi'
ST = ['a', 'g', 'u', 'S']
LBL = ['a (per particle,\npools to P)', 'g (jet, from P)', 'u (per particle,\npools to S)', 'S (jet)']
COL = dict(M0='#1f77b4', M2='#d62728')
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, m, name in zip(axs, ('3f', '16f'), ('3-feature JEDI-linear n64', '16-feature JEDI-linear n64')):
    f = D / f'e25x_{m}.json'
    if not f.exists(): ax.set_title(name + ' (missing)'); continue
    C = json.loads(f.read_text())['cells']
    for j, V in enumerate(('M0', 'M2')):
        xs = [i + (j - 0.5) * 0.18 for i in range(len(ST))]
        for k, mk, ls in (('2', 'o', '-'), ('1', 's', ':')):
            ys = [C[f'{s}|don|{V}|k{k}']['eff'] for s in ST]
            if k == '2':
                lo = [ys[i] - C[f'{s}|don|{V}|k{k}']['eff_ci'][0] for i, s in enumerate(ST)]
                hi = [C[f'{s}|don|{V}|k{k}']['eff_ci'][1] - ys[i] for i, s in enumerate(ST)]
                ax.errorbar(xs, ys, yerr=[lo, hi], color=COL[V], marker=mk, ls=ls, capsize=3,
                            label=f'{V} = {"Σ pT" if V == "M0" else "Σ pT ΔR²"} swap, 2 dims')
            else:
                ax.plot(xs, ys, color=COL[V], marker=mk, ls=ls, alpha=0.6, ms=4, label=f'{V}, 1 dim')
            bq = [C[f'{s}|don|{V}|k{k}'].get('base_q95') for s in ST]
            if k == '2':
                ax.scatter(xs, [b if b is not None else float('nan') for b in bq], marker='_', s=250, color='k', zorder=5,
                           label='non-trained nets, 95th pct (2 dims)' if V == 'M0' else None)
    ax.set_xticks(range(len(ST))); ax.set_xticklabels(LBL, fontsize=8); ax.set_title(name, fontsize=10)
    ax.set_ylim(0, 1); ax.grid(alpha=0.3)
axs[0].set_ylabel('interchange accuracy on call-changing pairs')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, loc='lower center', ncol=5, frameon=False)
fig.suptitle('Σ pT (M0) and Σ pT ΔR² (M2) can be swapped through 1–2 per-particle directions from the first layer (a) on', fontsize=10.5)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(D / 'e25x.png', dpi=130); print('wrote', D / 'e25x.png')
