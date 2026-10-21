"""Figure for the E33x verification: recurrence of the r2 formula against three decoy nulls (fresh jets), and the per-bin radial
response gamma_b / r_b (constant for a pT r^2 weight) in each trained network.
  python -m interp.e33x_verify_plot   -> $AI_RESULTS/jedi/e33x_verify.png"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'


def main():
    V = json.loads((OUT / 'e33x_verify.json').read_text()); N = V['nets']; ref = N['official_n64']
    others = [m for m in N if m != 'official_n64']; v = lambda x: -1 if x is None else x  # noqa: E731
    names = V['d3l_triples']
    sets = [('1-regressor rank-window hinges (E33x decoys)', ref['decoys_hung'], lambda R: R['decoys_hung'], None),
            ('3-regressor random hinge sums', [t[0] for t in ref['d3h']], lambda R: [t[0] for t in R['d3h']], None),
            ('3-regressor library sums without Σ pT r²', [t[0] for t in ref['d3l']], lambda R: [t[0] for t in R['d3l']],
             lambda i: 'pTr2[all]' not in names[i])]
    k = sum(v(N[m]['r2_hung']) >= 0.9 for m in others)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    w = 0.27; cols = ['#9a9a9a', '#4c78a8', '#59a14f']
    for s, (lab, r0, get, keep) in enumerate(sets):
        q = [i for i in range(len(r0)) if v(r0[i]) >= 0.9 and (keep is None or keep(i))]
        rec = np.array([sum(v(get(N[m])[i]) >= 0.9 for m in others) for i in q]); h = np.bincount(rec, minlength=10) / len(q)
        ge = int((rec >= k).sum())
        ax[0].bar(np.arange(10) + (s - 1) * w, h, w, color=cols[s], label=f'{lab}: {ge} of {len(q)} reach ≥ {k}, p = {(1 + ge) / (1 + len(q)):.3f}')
    ax[0].axvline(k, color='#d62728', ls='--'); ax[0].text(k - 2.6, 0.15, f'Σ pT r² formula: {k}/9', color='#d62728')
    ax[0].set_xticks(range(10)); ax[0].set_xlabel('number of the 9 other trained networks with a matching unit (test R² ≥ 0.9)')
    ax[0].set_ylabel('share of formulas that fit the reference'); ax[0].legend(fontsize=7.5, loc='upper right')
    ax[0].set_title('Fresh jets (5501/5502), one unit per formula')
    for m in N:
        rw = N[m]['radial_weight']; b = [x for x in rw['bins'] if 'gamma' in x]
        r = np.array([x['r_mean'] for x in b]); g = np.array([x['gamma'] for x in b])
        ax[1].plot(r, g / r / (g[:4] / r[:4]).mean(), 'o-', ms=3, lw=1.2 if m != 'official_n64' else 2.5,
                   label=f'{m}: β {rw["beta"]:.2f}, χ²/dof {rw["chi2"] / (rw["n_bins"] - 2):.0f}')
    ax[1].axhline(1, color='k', lw=0.8, ls=':'); ax[1].set_ylim(-1, 3)
    ax[1].set_xlabel('distance of the moved particle from the jet axis r'); ax[1].set_ylabel('γ_b / r_b, scaled to its mean for r < 0.2')
    ax[1].set_title('Radial response per GeV / r (flat = weight pT r²)'); ax[1].legend(fontsize=6.5, ncol=2, loc='upper left')
    fig.tight_layout(); fig.savefig(OUT / 'e33x_verify.png', dpi=130); print('wrote', OUT / 'e33x_verify.png')


if __name__ == '__main__':
    main()
