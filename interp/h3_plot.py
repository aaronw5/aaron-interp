"""H3 figure: change of the key call rate (percentage points) under each edit (filled) and its matched control (open), 95 %
bootstrap intervals, for the edits that carry claims D1 / D2.  Reads $AI_RESULTS/physics/h3_jedi.json, h3_part.json.
  ./job.sh d-h3-plot 'python -m interp.h3_plot'     -> $AI_RESULTS/physics/h3_effects.png"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
J = json.loads((R / 'h3_jedi.json').read_text()); P = json.loads((R / 'h3_part.json').read_text())
JW = json.loads((R / 'h3_jedi_wshift.json').read_text()); PW = json.loads((R / 'h3_part_wshift.json').read_text())   # matched-shift W rows
ROWS = [('JEDI', 'boost x0.9', 'W', 'boost ×0.9, W jets (W call)'),
        ('JEDI', 'boost x0.9', 'Z', 'boost ×0.9, Z jets (Z call)'),
        ('JEDI', 'boost x1.25', 't', 'boost ×1.25, top jets (top call)'),
        ('JEDI', 'W pair -> 60 GeV', 't', 'W pair → 60 GeV in top (top call)'),
        ('JEDI', 'W pair -> 100 GeV', 't', 'W pair → 100 GeV in top (top call)'),
        ('JEDI', 'W pair -> 120 GeV', 't', 'W pair → 120 GeV in top (top call)'),
        ('JEDI', 'remove b (recentred, boosted back)', 't', 'remove b from top (W or Z call)'),
        ('ParT_full', 'boost x0.9', 'Wqq', 'boost ×0.9, W jets (W call)'),
        ('ParT_full', 'decay collapse', 'Wqq', 'collapse decay splitting, W (W call)'),
        ('ParT_full', 'W pair -> 60 GeV', 'Tbqq', 'W pair → 60 GeV in top (top call)'),
        ('ParT_full', 'W pair -> 100 GeV', 'Tbqq', 'W pair → 100 GeV in top (top call)'),
        ('ParT_full', 'W pair -> 120 GeV', 'Tbqq', 'W pair → 120 GeV in top (top call)'),
        ('ParT_full', 'remove b (recentred, boosted back)', 'Tbqq', 'remove b from top (W or Z call)'),
        ('ParT_full', 'drop displaced tracks', 'Hbb', 'drop displaced tracks, H→bb (control: drop prompt tracks)'),
        ('ParT_full', 'keep 2 most displaced', 'Hbb', 'keep 2 displaced tracks, H→bb (control: lifetime ×0.35)'),
        ('ParT_full', 'lepton de-isolated (4 nearest to dR 0.03)', 'Tbl', 'de-isolate lepton, t→bℓν'),
        ('ParT_full', 'lepton charge flipped (and d0 sign)', 'Tbl', 'flip lepton charge, t→bℓν'),
        ('ParT_full', 'pi0 merge (photon pairs 85-185 MeV merged)', 'Hbb', 'merge π⁰-like photon pairs, H→bb'),
        ('ParT_full', 'dead cone filled (6 x 1 GeV photons within dR 0.015 of each b/c)', 'Hbb', 'fill b dead cone, H→bb')]


def find(model, edit, cls):
    j, p = (JW, PW) if edit.startswith('W pair') else (J, P)
    comps = j['comparisons'] if model == 'JEDI' else p['models'][model.split('_')[1]]['comparisons']
    for r in comps:
        if r['edit'] == edit and r['cls'] == cls and 'key_before' in r: return r


fig, ax = plt.subplots(figsize=(9, 8.5)); y = np.arange(len(ROWS))[::-1]; labels = []
for yy, (m, e, c, lab) in zip(y, ROWS):
    r = find(m, e, c)
    if r is None: labels.append(f'{m}: {lab} (missing)'); continue
    labels.append(f'{m}: {lab}  (n={r["n"]})')
    ci = np.array(r['key_change_ci']) * 100; ax.errorbar(100 * r['key_change'], yy + 0.15, xerr=[[100 * r['key_change'] - ci[0]], [ci[1] - 100 * r['key_change']]],
                                                          fmt='o', color='#1f5fa8', ms=6, capsize=2)
    if 'ctrl_key_change' in r:
        cc = np.array(r['ctrl_key_change_ci']) * 100
        ax.errorbar(100 * r['ctrl_key_change'], yy - 0.15, xerr=[[100 * r['ctrl_key_change'] - cc[0]], [cc[1] - 100 * r['ctrl_key_change']]],
                    fmt='o', mfc='white', color='#c0392b', ms=6, capsize=2)
ax.axvline(0, color='0.5', lw=0.8); ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('change in rate of the named call [percentage points], 95 % CI')
ax.plot([], [], 'o', color='#1f5fa8', label='edit'); ax.plot([], [], 'o', mfc='white', color='#c0392b', label='matched control')
ax.legend(loc='lower left', fontsize=8); ax.set_title('Each physics edit vs its matched control', fontsize=11)
fig.tight_layout(); fig.savefig(R / 'h3_effects.png', dpi=150); print('wrote', R / 'h3_effects.png')
