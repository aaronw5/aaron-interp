"""P1v figure: read-out mass shift and W→Z under the φ translation, per input variant (check 2), ParT_full and ParT_kin.
  AI_RESULTS=/j-jepa-vol/aaron-interp/results python interp/p1v_plot.py   (helper pod, env-part: matplotlib)
  -> results/part/p1v_verify.png"""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
R = json.load(open(OUT / 'p1v_verify.json'))
DS = ['0.03', '0.06', '0.09']
V = [('a_full_edit', 'full edit', '#2a78d6', 'o'), ('b_deltaR_unedited', 'ΔR fed unedited (only Δφ edited)', '#eb6834', 's'),
     ('c_dphi_unedited', 'Δφ fed unedited (only ΔR edited)', '#1baf7a', '^'), ('d_both_unedited', 'ΔR and Δφ fed unedited', '#eda100', 'D'),
     ('f_vectors_unedited', '4-vectors unedited (no pair route)', '#e87ba4', 'v')]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
fig, ax = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True)
for j, n in enumerate(('full', 'kin')):
    M = R['models'][n]; c2 = M['check2']; x = [0.0] + [float(d) for d in DS]
    pred = [0.0] + [c2[d]['a_full_edit']['pred'] for d in DS]
    ax[0, j].plot(x, pred, color='#8a8984', lw=2, ls='--', label='prediction ½Δlog(M0·M2)')
    for k, lab, col, mk in V:
        ax[0, j].plot(x, [0.0] + [c2[d][k]['dlogm_hat'] for d in DS], color=col, lw=2, marker=mk, ms=7, label=lab)
        ax[1, j].plot(x, [M['baseline']['W_to_Z']] + [c2[d][k]['W_to_Z'] for d in DS], color=col, lw=2, marker=mk, ms=7, label=lab)
    ax[0, j].set_title(f'ParT_{n}', loc='left', fontsize=11, color='#0b0b0b')
    ax[0, j].set_ylabel('Δ log m̂ (class-token read-out)'); ax[1, j].set_ylabel('fraction of true W called Z')
    ax[1, j].set_xlabel('rigid φ shift δ of every particle [rad] (jet axis fixed)')
    for a in ax[:, j]: a.grid(axis='y', color='#e4e3df', lw=0.8); a.set_axisbelow(True)
ax[0, 0].legend(frameon=False, fontsize=8.5, loc='upper left')
fig.suptitle('Response to a φ translation, by the input fed edited (10,000 JetClass test jets)', x=0.01, ha='left', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / 'p1v_verify.png', dpi=130); print('wrote', OUT / 'p1v_verify.png')
