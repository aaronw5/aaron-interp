"""P26 figure from results/part/p26_sophon_axis_run.json (numpy + matplotlib only).
  ./job.sh c-p26-plot 'python -m interp.p26_plot'     -> results/part/p26_sophon_axis.png
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'


def main():
    r = json.load(open(OUT / 'p26_sophon_axis_run.json')); R = r['rows']
    keys = ['sophon__A', 'sophon__B', 'part_full__A', 'part_full__B', 'part_kin__A', 'part_kin__B', 'sophon_init0__A', 'sophon_init1__A']
    names = ['Sophon\nJetClass', 'Sophon\nJetClass-II', 'ParT_full\nJetClass', 'ParT_full\nJetClass-II', 'ParT_kin\nJetClass',
             'ParT_kin\nJetClass-II', 'untrained\nseed 100', 'untrained\nseed 101']
    x = np.arange(len(keys)); fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # (a) W-jet read-out change per Z-W separation, under the matched dilation and the phi shift
    k3 = keys[:6]; x3 = np.arange(6)
    for j, (e, c, lab) in enumerate((('dil', 'C1', 'matched dilation (real mass +12 %)'), ('phi', 'C0', 'φ + 0.06 rad (invariant mass unchanged)'))):
        v = np.array([[R[k]['edits'][e]['dd_mean_W']] + R[k]['edits'][e]['dd_mean_W_ci'] for k in k3]); s = np.array([R[k]['sep_ZminusW'][0] for k in k3])[:, None]
        v = v / s; ax[0].bar(x3 + (j - 0.5) * 0.38, v[:, 0], 0.38, color=c, label=lab)
        ax[0].errorbar(x3 + (j - 0.5) * 0.38, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt='none', ecolor='k', capsize=2)
    ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xticks(x3); ax[0].set_xticklabels(names[:6], fontsize=8)
    ax[0].set_ylabel('Δ[log p(Z) − log p(W)] on true-W jets\n/ (mean on Z jets − mean on W jets)')
    ax[0].set_title('(a) Output: W/Z score change per Z–W separation'); ax[0].legend(fontsize=7.5, loc='upper right')
    # (b) internal mass probe: phi response per dilation response
    v = np.array([R[k]['R_probe'] for k in keys])
    ax[1].bar(x, v[:, 0], color=['C0', 'C0', 'C1', 'C1', 'C1', 'C1', 'C7', 'C7'])
    ax[1].errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt='none', ecolor='k', capsize=2)
    ax[1].axhline(1, color='k', ls=':', lw=0.8); ax[1].text(7.4, 1.02, 'reads the axis moment', ha='right', fontsize=7)
    ax[1].axhline(0, color='k', lw=0.6); ax[1].text(7.4, 0.03, 'reads the invariant mass', ha='right', fontsize=7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7); ax[1].set_ylabel('Δ probe(φ shift) / Δ probe(matched dilation)')
    ax[1].set_title('(b) Class token: linear log-mass probe')
    # (c) WTA axis vs controls, top-1 calls changed
    W = np.array([[R[k]['edits'][e]['changed_top1'][0] for e in ('wta', 'trc', 'rotc')] for k in keys]) * 100
    for j, (lab, c) in enumerate(zip(('WTA axis', 'axis moved by the same length, random direction', 'particles rotated about the axis (ΔR kept)'), ('C0', 'C8', 'C7'))):
        ax[2].bar(x + (j - 1) * 0.27, W[:, j], 0.27, label=lab, color=c)
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, fontsize=7); ax[2].set_ylabel('top-1 calls changed [%]')
    ax[2].set_title('(c) Inputs recomputed about another axis'); ax[2].legend(fontsize=7, loc='upper left')
    fig.suptitle('Sophon (JetClass-II, 188 classes): no W/Z response to a real mass change, and a probe that follows the invariant mass more than ParT\'s', fontsize=11)
    fig.tight_layout(); fn = OUT / 'p26_sophon_axis.png'; fig.savefig(fn, dpi=130); print('wrote', fn)


if __name__ == '__main__':
    main()
