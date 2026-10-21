"""E33x figure from results/jedi/e33x.json -> results/jedi/e33x_recurrence_null.png
(left) how many of the 9 other trained networks have a unit fitting each formula (test R^2 >= 0.9): grey histogram = random
rank-window hinge formulas that fit the reference equally well; markers = named formulas and competitors.
(right) radial weight power beta (unit ~ sum pT r^beta) of each network's squared-radius unit, 95 % bootstrap CI."""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
S = json.loads((D / 'e33x.json').read_text()); H = S['hung']
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
h = np.array(H['decoy_recurrence_hist'], float); x = np.arange(len(h))
ax[0].bar(x, h / h.sum(), color='0.75', label=f'random hinge formulas fitting the reference (N = {H["n_qualifying"]})')
lab = dict(r2='Σ pT r² unit', soft='soft count', hard='hardest-k pT', core='core Σ(t−r)₊', total_pT='total pT', sum_r='Σ r', mult_N='multiplicity', sum_pT_r='Σ pT r')
col = dict(r2='C3', soft='C0', hard='C2', core='C1', total_pT='C4', sum_r='C5', mult_N='C6', sum_pT_r='C7')
ymax = (h / h.sum()).max(); seen = {}
for f, e in H['formulas'].items():
    if not e['qualifies']: continue
    k = e['recurrence']; seen[k] = seen.get(k, 0) + 1; y = ymax * (1.05 + 0.09 * seen[k])
    ax[0].plot(k, y, 'v', color=col[f], ms=9, label=f'{lab[f]}: {k}/9, p = {e["p"]:.3f}')
ax[0].set_xlabel('number of the 9 other trained networks with a matching unit (test R² ≥ 0.9)')
ax[0].set_ylabel('share of random formulas'); ax[0].set_xticks(x); ax[0].set_ylim(0, ymax * 2.0)
ax[0].legend(fontsize=7, loc='upper left', frameon=False)
ax[0].set_title('Recurrence across retrained JEDI-linear networks', fontsize=10)
RW = S['radial_weight']; ms = [m for m in RW if RW[m].get('beta') is not None]
for i, m in enumerate(ms):
    b, (lo, hi) = RW[m]['beta'], RW[m]['beta_ci']
    ax[1].errorbar(b, i, xerr=[[b - lo], [hi - b]], fmt='o', color='C3', capsize=3)
ax[1].axvline(2, color='k', ls='--', lw=1); ax[1].text(2.02, -0.45, 'pT r²', fontsize=8)
ax[1].axvline(1, color='0.5', ls=':', lw=1); ax[1].text(1.02, -0.45, 'pT r', fontsize=8, color='0.4')
ax[1].set_yticks(range(len(ms))); ax[1].set_yticklabels(ms, fontsize=8)
ax[1].set_xlabel('β in per-particle weight pT·r^β (from ±0.01 radial moves)')
ax[1].set_title('Radial power of the squared-radius unit', fontsize=10)
fig.tight_layout(); out = D / 'e33x_recurrence_null.png'; fig.savefig(out, dpi=130); print(out)
