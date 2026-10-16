"""E13 figure: agreement of the edit response between the fixed-point network and the float+mask program, stage by stage
(reads results/jedi/e13_edit_fidelity.json)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parents[1] / 'results' / 'jedi'
R = json.loads((OUT / 'e13_edit_fidelity.json').read_text())
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150)
for (g, e), col in zip(R['figure_values'].items(), COL):
    ax.plot([0, 1, 2], e['median'], '-o', color=col, lw=2, ms=8, label=g)
ax.set_xticks([0, 1, 2], ['first-round objects P\n(median over 12 live units)', 'second-round objects S\n(median over 19 live units)',
                          'logits\n(mean over 5 classes)'])
ax.set_ylim(0, 1.02); ax.set_ylabel('Pearson r of the per-jet edit response\n(network vs float+mask program)')
ax.set_title(f'Edit responses: network vs float+mask program, by stage\n({R["N"]:,} test jets, seed {R["seed"]}; '
             'one particle edited per jet; median over the edits in each group)', fontsize=9.5)
ax.grid(axis='y', color='#dddddd', lw=0.6); ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig(OUT / 'e13_edit_fidelity.png'); print('saved', OUT / 'e13_edit_fidelity.png')
