import json, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = json.load(open(sys.argv[1])); out = sys.argv[2]
ED = [('T', 'T: +0.06 rad shift\n(W jets)'), ('DW', 'D: prong sharing\n(W jets)'), ('DZ', 'D: prong sharing\n(Z jets)'), ('N', 'N: +1 GeV particle\n(q+g jets)')]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
cols = dict(stitch='#2f6fd6', logit='#c77d1a', untrained='#8a8f98')
lab = dict(stitch='trained donor, map from S / h4', logit='trained donor, map from its 5 class scores only', untrained='untrained redraw of donor (seed 0)')
for ax, L in zip(axs, ('S', 'h4')):
    for i, (rn, nm) in enumerate(ED):
        o = R['inheritance'][L][rn]
        for j, k in enumerate(('stitch', 'logit', 'untrained')):
            p = o[k]['pooled']; x = i + (j - 1) * 0.25
            ax.bar(x, p['est'], 0.23, color=cols[k], label=lab[k] if i == 0 else None)
            if p.get('ci'): ax.plot([x, x], p['ci'], color='k', lw=1.2)
    ax.axhline(0.5, color='crimson', ls='--', lw=1.2); ax.axhline(0, color='k', lw=0.6); ax.axhline(1, color='k', lw=0.6, ls=':')
    ax.text(3.45, 1.47, 'dashed line I = 0.5: value when the stitched response\nis the same constant in every pair (no information)', color='crimson', fontsize=8, ha='right', va='top')
    ax.set_xticks(range(len(ED))); ax.set_xticklabels([n for _, n in ED], fontsize=9)
    ax.set_title(f'stitch at {L}', fontsize=11)
axs[0].set_ylabel('pooled inheritance index I\n(1 = like donor A, 0 = like recipient B)')
axs[0].set_ylim(-0.3, 1.5); axs[0].legend(fontsize=8, loc='upper left', frameon=False)
fig.suptitle('E39 verification (fresh jets, N 9,625): pooled I with 95 % intervals from leaving out one network at a time (7 networks, 42 ordered pairs)', fontsize=10.5)
fig.tight_layout(); fig.savefig(out, dpi=130)
