"""E31 figure: one real W jet before / after each loop-A edit (particles in the (deta, dphi) plane about the stored axis, marker area
proportional to pT), and the C2ST AUC of every edit next to its matched real change (dilation) and its null edit, for both feature
sets.  Reads $AI_RESULTS/jedi/e31_onmanifold.json and e31_examples.npz (or the paths given).  Writes results/jedi/e31_onmanifold.png.
  python -m interp.e31_plot [--smoke]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
CL = ['g', 'q', 'W', 'Z', 't']
PANELS = [('shift', 'shift: all particles\n$\\Delta\\eta$ + 0.06 rad'), ('turn10', 'turn: 14 hardest turned\nin azimuth, $m$ − 10 %'),
          ('split0', 'split: hardest particle\ninto two halves (same place)'), ('soft', 'soft: one 0.5 GeV\nparticle added'),
          ('boost', 'boost: |P| × 0.9\nat fixed mass'), ('dil_M2', 'reference: dilation\n(angles × const., same $\\Delta M_2$ as shift)')]
ORDER = [('shift', 'dil_M2', 'rot_shift'), ('turn5', 'dil_m5', 'rot_turn5'), ('turn10', 'dil_m10', 'rot_turn10'),
         ('split0', 'dil5', 'rot_split0'), ('split13', 'dil5', 'rot_split13'), ('split21', 'dil5', 'rot_split21'),
         ('soft', 'dil5', 'soft_ctrl'), ('boost', 'dil_B', 'rot_boost')]


def main(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sfx = '_smoke' if a.smoke else ''
    R = json.loads((RES / f'e31_onmanifold{sfx}.json').read_text()); X = np.load(RES / f'e31_examples{sfx}.npz')
    fig = plt.figure(figsize=(15, 8.2)); gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.12, 2.3], hspace=0.38, wspace=0.28)
    x0 = X['x0']; r0 = x0[:, 0] > 0; s = lambda p: 6 + 600 * p / x0[r0, 0].max()
    for i, (k, ttl) in enumerate(PANELS):
        ax = fig.add_subplot(gs[i // 3, i % 3]); xe = X[f'x_{k}']; re = xe[:, 0] > 0
        ax.scatter(x0[r0, 2], x0[r0, 1], s=s(x0[r0, 0]), facecolors='none', edgecolors='0.55', lw=0.8, label='unedited')
        ax.scatter(xe[re, 2], xe[re, 1], s=s(xe[re, 0]), c='C3', alpha=0.55, lw=0, label='edited')
        if k == 'soft':
            new = [j for j in np.flatnonzero(re) if not np.any(np.all(np.isclose(x0[r0], xe[j]), 1))]
            ax.scatter(xe[new, 2], xe[new, 1], s=60, marker='x', c='k', label='added particle')
        ax.set_xlim(-0.45, 0.45); ax.set_ylim(-0.45, 0.45); ax.set_aspect('equal'); ax.axhline(0, c='0.85', lw=0.5, zorder=0); ax.axvline(0, c='0.85', lw=0.5, zorder=0)
        ax.set_title(f'{ttl}\ncall {CL[int(X["call0"])]} → {CL[int(X[f"call_{k}"])]}', fontsize=9)
        ax.set_xlabel('$\\Delta\\phi$ [rad]', fontsize=8); ax.set_ylabel('$\\Delta\\eta$ [rad]', fontsize=8); ax.tick_params(labelsize=7)
        if i == 0: ax.legend(fontsize=7, loc='lower left')
    ax = fig.add_subplot(gs[:, 4]); C = R['c2st']; FS = (('obs', 'obs'), ('obs8', 'obs w/o |M1|'), ('plist', '30-particle list'))
    ticks, labs = [], []; base = 0.0
    for o in ORDER[::-1]:
        for j, (fs, lab) in enumerate(FS[::-1]):
            yy = base + j * 0.28
            for idx, col, mk, nm in ((2, '0.55', '^', 'null edit (same displacement)'), (1, 'C0', 's', 'matched real change (dilation)'), (0, 'C3', 'o', 'edit')):
                v = C[fs][o[idx]]; ax.errorbar(v['auc'], yy, xerr=[[v['auc'] - v['ci'][0]], [v['ci'][1] - v['auc']]], fmt=mk, color=col, ms=4.5, lw=1,
                                                label=nm if (o == ORDER[0] and j == 0) else None)
            ax.text(0.392, yy, lab, fontsize=6, va='center', color='0.35')
        ticks.append(base + 0.28); labs.append(o[0]); base += 1.0
        ax.axhline(base - 0.18, c='0.9', lw=0.6)
    ax.axvline(0.5, c='k', lw=0.8, ls=':'); ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=9); ax.set_xlim(0.385, 1.01); ax.set_ylim(-0.3, base - 0.2)
    ax.tick_params(axis='y', pad=2)
    ax.set_xlabel('C2ST AUC  (0.5 = edited jets look like real jets; 1 = always told apart)', fontsize=8)
    ax.set_title('Can a classifier tell edited jets from real jets?\nfeature sets: jet observables (obs), obs without the centroid offset, particle list', fontsize=9)
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3); ax.grid(axis='x', alpha=0.3)
    fig.suptitle(f'E31: loop-A edits on one real W jet (N = {int((x0[:, 0] > 0).sum())}, m = {R["example"]["m0"]:.0f} GeV), and their C2ST AUC on {R["N"]} jets', fontsize=11)
    out = ROOT / 'results' / 'jedi' / f'e31_onmanifold{sfx}.png'; out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=130, bbox_inches='tight'); print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); main(ap.parse_args())
