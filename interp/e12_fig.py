"""E12 figure: % of calls changed when all 16 neurons z are replaced by K-term formulas, read by the network's own d7.
Reads results/jedi/e12_z_on_S.json; writes results/jedi/e12_z_on_S.png (run with any Python that has matplotlib)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.loads(Path('results/jedi/e12_z_on_S.json').read_text())
KS = R['KS']; res = R['results']
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', violet='#6250d6', red='#e34948')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def curve(b, fam):
    d = res[b][fam]
    return [d[str(K)]['changed'] if 'changed' in d[str(K)] else d[str(K)]['changed_mean'] for K in KS]


plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True, facecolor='#fcfcfb')
panels = [
    ('a  Hinge formulas: network objects S vs hand observables',
     [('S', 'hinge', f'S objects (19 live units)', C['blue'], '-'),
      ('hand', 'hinge', f'hand observables ({len(R["hand_observables"])})', C['orange'], '-'),
      ('hand_all', 'hinge', f'all Q_ observables ({R["n_hand_all"]})', C['orange'], ':'),
      ('h4', 'hinge', 'h4 = ReLU(W4·S + b) (diagnostic)', C['green'], '--')]),
    ('b  Controls (hinge formulas)',
     [('S', 'hinge', 'S objects, selected terms', C['blue'], '-'),
      ('randS', 'hinge', 'S objects, random terms (mean of 10)', C['violet'], '--'),
      ('initS', 'hinge', 'random-init network S', C['magenta'], '-'),
      ('raw', 'hinge', 'raw inputs (pT, Δη, Δφ × 64)', C['yellow'], '-'),
      ('P', 'hinge', 'first-round objects P (11)', C['aqua'], '-')]),
    ('c  Formula families on S (Paper B baselines)',
     [('S', 'hinge', 'hinge (MARS-style)', C['blue'], '-'),
      ('S', 'tree', 'regression tree (K leaves)', C['red'], '-'),
      ('S', 'poly', 'degree-2 polynomial', C['violet'], '-'),
      ('S', 'ridge', 'linear (ridge)', C['aqua'], '-')]),
]
fm = R.get('front_to_back', {}).get('float_mask_changed')
for ax, (title, series) in zip(axs, panels):
    for b, fam, lab, col, ls in series:
        if b not in res or fam not in res[b]: continue
        ax.plot(KS, curve(b, fam), ls, color=col, lw=2, marker='o', ms=5, label=lab)
    ax.axhline(3, color=INK2, lw=1, ls=(0, (2, 2)))
    ax.text(KS[-1], 3.15, '3 %', color=INK2, ha='right', va='bottom', fontsize=9)
    if fm is not None:
        ax.axhline(fm, color=GRID, lw=1.5)
        ax.text(KS[0], fm * 1.05, f'float + mask program {fm:.1f} %', color=INK2, ha='left', va='bottom', fontsize=8)
    ax.set_xscale('log', base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
    ax.set_yscale('log'); ax.set_yticks([2, 3, 5, 10, 20, 40, 80]); ax.set_yticklabels(['2', '3', '5', '10', '20', '40', '80'])
    ax.set_ylim(1.5, 90); ax.grid(axis='y', color=GRID, lw=0.8); ax.set_facecolor('#fcfcfb')
    ax.set_xlabel('terms per neuron K (tree: leaves)'); ax.set_title(title, fontsize=10, loc='left', color=INK)
    ax.legend(frameon=False, fontsize=8, loc='upper right', labelcolor=INK2)
axs[0].set_ylabel('% of calls changed vs network\n(all 16 neurons z substituted, network d7)')
fig.suptitle(f'JEDI-linear 64p: neurons z as K-term formulas; {R["n_eval"]:,} evaluation jets (fit on {R["n_fit"]:,}; seed {R["seed"]})',
             x=0.01, ha='left', fontsize=11, color=INK)
fig.tight_layout()
out = Path('results/jedi/e12_z_on_S.png'); fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); print(out)
