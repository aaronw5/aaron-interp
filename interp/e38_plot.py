"""E38 figure: (a) the T(0->1, 0.2) edit on a real W jet, (b) W-Z margin change map at f = 0.2 (W and Z jets), (c) E36 flips
undone by holding S20+S51+S43 vs the radial triple and matched random triples.   Run: python -m interp.e38_plot"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'


def main(sfx=''):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((OUT / f'e38{sfx}.json').read_text()); M = R['main']['map']; cells = M['cells']
    ranks = ['0', '1', '2', '3', '4', '5', 'rest']; lab = ['0', '1', '2', '3', '4', '5', '6–63']
    def grid(vals, f='0.20'):
        G = np.full((7, 7), np.nan)
        for c, v in zip(cells, vals):
            kj, ff = c.split('@'); k, j = kj.split('->')
            if ff == f: G[ranks.index(k), ranks.index(j)] = np.nan if v is None else v
        return G
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1.05, 0.9]))
    # (a) jet
    d = np.load(OUT / f'e38_fig_jet{sfx}.npz'); x0, xe = d['x0'], d['xe']; r0 = x0[:, 0] > 0
    a = ax[0]
    a.scatter(x0[r0, 1], x0[r0, 2], s=x0[r0, 0] * 0.8, facecolors='none', edgecolors='0.5', lw=1, label='unedited')
    re = xe[:, 0] > 0
    a.scatter(xe[re, 1], xe[re, 2], s=xe[re, 0] * 0.8, c='C3', alpha=0.45, label='edited (re-sorted)')
    for i in (0, 1):
        a.annotate(f'rank {i}: {x0[i, 0]:.0f} → {x0[i, 0] * (0.8 if i == 0 else 1) + (0.2 * x0[0, 0] if i == 1 else 0):.0f} GeV',
                   (x0[i, 1], x0[i, 2]), xytext=(8, 8 - 18 * i), textcoords='offset points', fontsize=8)
    a.set_xlabel('Δη to jet axis'); a.set_ylabel('Δφ to jet axis (rad)'); a.set_title('(a) edit T(0→1, 0.2) on a W jet\n'
                                                                                       '20 % of the hardest particle\'s pT to the second; angles re-solved', fontsize=9)
    a.legend(fontsize=7, loc='lower right'); a.set_aspect('equal', 'datalim')
    # (b) map
    G = grid(M['wz_on_WZ']['mean']); Gn = grid(M['null']['wz_on_WZ']['mean']); P = grid(M['T_minus_null']['wz_p_holm'])
    v = np.nanmax(np.abs(G)); b = ax[1]; im = b.imshow(G, cmap='RdBu_r', vmin=-v, vmax=v)
    for i in range(7):
        for j in range(7):
            if np.isfinite(G[i, j]):
                b.text(j, i, f'{G[i, j]:+.2f}', ha='center', va='center', fontsize=7)
    b.set_xticks(range(7)); b.set_xticklabels(lab); b.set_yticks(range(7)); b.set_yticklabels(lab)
    b.set_xlabel('destination pT rank j'); b.set_ylabel('source pT rank k')
    b.set_title('(b) mean change of logit(W) − logit(Z) on W and Z jets, f = 0.2\n'
                f'soft-sector reshuffle of the same GeV: {np.nanmin(Gn):+.2f} to {np.nanmax(Gn):+.2f}', fontsize=9)
    plt.colorbar(im, ax=b, fraction=0.046, label='Δ W−Z margin (logit)')
    # (c) holds
    H = R.get('e36_hold', {}); c = ax[2]
    if 'sets' in H:
        s = H['sets']; rt = H['random_triples']; a_ = s['S20+S51+S43 (leading-particle)']; b_ = s['S1+S29+S30 (radial)']
        names = ['S20+S51+S43\n(leading-particle\nobjects)', 'S1+S29+S30\n(radial\nobjects)', f'{rt["n"]} random\ntriples:\nmedian', 'random\ntriples:\n95th pct']
        vals = [a_['margin_removed']['value'], b_['margin_removed']['value'], rt['margin_removed_q50_q95_max'][0], rt['margin_removed_q50_q95_max'][1]]
        err = [[vals[0] - a_['margin_removed']['ci'][0], vals[1] - b_['margin_removed']['ci'][0], 0, 0], [a_['margin_removed']['ci'][1] - vals[0], b_['margin_removed']['ci'][1] - vals[1], 0, 0]]
        c.bar(range(4), vals, yerr=err, color=['C3', 'C0', '0.6', '0.8'], capsize=3)
        for i, vv in enumerate(vals): c.text(i, vv + 0.03, f'{vv:.2f}', ha='center', fontsize=8)
        c.set_xticks(range(4)); c.set_xticklabels(names, fontsize=8); c.set_ylabel('share of the flip\'s margin change removed')
        c.set_title(f'(c) E36 minimal call flips (N {H["n_flips"]}): hold 3 S objects\nat their unedited values; rank p '
                    f'{a_["rank_p_margin_removed"]:.2f} (leading), {b_["rank_p_margin_removed"]:.2f} (radial)', fontsize=9)
        c.set_ylim(0, max(vals) * 1.3 + 0.02)
    fig.tight_layout(); fig.savefig(OUT / f'e38_hardness{sfx}.png', dpi=130); print('wrote', OUT / f'e38_hardness{sfx}.png')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '')
