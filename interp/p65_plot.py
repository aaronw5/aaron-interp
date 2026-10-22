"""P65 figure: (a) one H->bb jet, charged tracks coloured by transverse impact-parameter significance S = |d0|/sigma(d0), the 3
largest-S tracks ringed; (b) the same jet coloured by the attention of the top-ranked head (frozen ranking = resample ablation on
b-vs-c); (c) b-vs-c accuracy of the literal circuit (top-k heads + top class-MLP neurons kept, everything else resample-ablated) vs
k, against block-matched random head sets, the full network, ParT_kin (no displacement inputs) and the all-ablated floor.
  python -m interp.p65_plot [--smoke]  -> $AI_RESULTS/part/p65_flavour_circuit.png"""
import json, os, sys
from pathlib import Path
import numpy as np

OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TAG = '_smoke' if '--smoke' in sys.argv else ''


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    R = json.loads((OUTD / f'p65_flavour_circuit{TAG}.json').read_text()); F = R['models']['ParT_full']; j = F['figure']['jet']
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    pt, de, dp, S, q = (np.asarray(j[k], float) for k in ('pt', 'deta', 'dphi', 'S', 'charge'))
    A = np.asarray(j['attention'], float); ch = q != 0; sz = 12 + 260 * np.sqrt(pt / pt.max())
    top3 = np.argsort(-np.where(ch, S, -1))[:3]
    for a in ax[:2]:
        a.scatter(de[~ch], dp[~ch], s=sz[~ch], c='0.8', edgecolors='0.6', lw=0.5, label='neutral particles')
        a.set_xlabel('Δη to jet axis'); a.set_ylabel('Δφ to jet axis [rad]'); a.set_aspect('equal', 'datalim')
    sc = ax[0].scatter(de[ch], dp[ch], s=sz[ch], c=np.clip(S[ch], 0.1, None), cmap='viridis', norm=LogNorm(0.1, max(S.max(), 10)), edgecolors='k', lw=0.4)
    ax[0].scatter(de[top3], dp[top3], s=sz[top3] * 1.8, facecolors='none', edgecolors='#d1495b', lw=1.8, label='3 largest-S tracks')
    plt.colorbar(sc, ax=ax[0], fraction=0.046, label='track |d0| / σ(d0)')
    ax[0].set_title(f'One H→bb jet (p_T {j["jet_pt"]:.0f} GeV): tracks by displacement\n(marker area ∝ √p_T)', fontsize=10)
    ax[0].legend(fontsize=8, loc='lower left')
    sa = ax[1].scatter(de, dp, s=sz, c=A, cmap='magma_r', edgecolors='k', lw=0.4, vmin=0)
    ax[1].scatter(de[top3], dp[top3], s=sz[top3] * 1.8, facecolors='none', edgecolors='#d1495b', lw=1.8)
    plt.colorbar(sa, ax=ax[1], fraction=0.046, label='attention weight')
    sh = F['figure']['shares']['unedited']
    kind = 'class-token query' if j['head'].startswith('b') else 'mean over particle queries'
    ax[1].set_title(f'Same jet: attention of head {j["head"]} ({kind})\nall H→bb jets: {sh["top3_S"]:.2f} on 3 largest-S tracks, {sh["bottom3_S"]:.2f} on 3 smallest',
                    fontsize=10)
    C = F['circuit']; KS = R['KS']; xs = np.arange(len(KS)); blue, grey = '#1f6fb4', '0.55'
    for i, k in enumerate(KS):
        r = C['random'][f'k{k}']['acc_bc']['random']
        ax[2].vlines(i, r['q05'], r['q95'], color=grey, lw=6, alpha=0.5)
        ax[2].plot(i, r['mean'], '_', color='k', ms=12)
    ax[2].plot(xs, [C['random'][f'k{k}']['acc_bc']['selected'] for k in KS], 'D', color=blue, ms=8, label='top-k heads + top class-MLP neurons (literal circuit)')
    ax[2].plot([], [], color=grey, lw=6, alpha=0.5, label='100 random head sets, same blocks (5–95 %)')
    full = F['eval']['acc_bc']; kin = R['models']['ParT_kin']['eval']['acc_bc']; floor = C['random']['floor_sub']['acc_bc']
    for v, lab, ls in ((full, 'full ParT_full', ':'), (kin, 'ParT_kin (no displacement inputs)', '--'), (floor, 'everything ablated', '-.')):
        ax[2].axhline(v, color='k', ls=ls, lw=1); ax[2].text(len(KS) - 0.6, v + 0.006, lab, fontsize=8, ha='right')
    ax[2].set_xticks(xs); ax[2].set_xticklabels([f'k = {k}' for k in KS]); ax[2].set_ylim(0.4, 1.02)
    ax[2].set_xlabel('top-k heads kept (all else resample-ablated)')
    ax[2].set_ylabel('H→bb vs H→cc accuracy (sign of log p_bb/p_cc)')
    ax[2].set_title(f'Literal circuit alone: b-vs-c accuracy\n({C["n_neurons"]} class-MLP neurons kept in every set)', fontsize=10)
    ax[2].legend(fontsize=7.5, loc='lower left')
    fig.suptitle(f'ParT_full, JetClass test jets ({R["files"]}, seed {R["seed"]})', fontsize=10, y=1.0)
    fig.tight_layout(); fo = OUTD / f'p65_flavour_circuit{TAG}.png'; fig.savefig(fo, dpi=140, bbox_inches='tight'); print('wrote', fo)


if __name__ == '__main__':
    main()
