"""H32 figure: D - cside (top log-odds) against the requested change of |cos theta*|, top jets, both samples; networks as lines,
MLP5c / MLPx references as the range over 5 retrains, c-r null (cr - cphi) as the grey band (envelope of the networks' 95 % CIs).
Run: python -m interp.h32_plot   (reads $AI_RESULTS/physics/h32_top_dalitz_full.json, writes h32_dalitz.png)"""
import json
import os
from pathlib import Path
import numpy as np

RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
LAB = {'ParT_top': 'Top ParT', 'PN_top': 'Top ParticleNet', 'LNet_top': 'Top LorentzNet', 'PFN_top': 'Top PFN', 'EFN_top': 'Top EFN',
       'ParT_kin': 'JetClass ParT_kin', 'PN_kin': 'JetClass PN_kin', 'ParT_full': 'JetClass ParT_full', 'PN_full': 'JetClass PN_full'}
COL = {'ParT_top': '#c05621', 'PN_top': '#dd6b20', 'LNet_top': '#975a16', 'PFN_top': '#b7791f', 'EFN_top': '#d69e2e',
       'ParT_kin': '#2b6cb0', 'PN_kin': '#4299e1', 'ParT_full': '#553c9a', 'PN_full': '#9f7aea'}


def main(tag='h32'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = json.loads((RES / f'{tag}_top_dalitz_full.json').read_text()); dcs = R['dc_values']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    for ax, (ds, dsn) in zip(axes, (('top', 'Top Landscape top jets'), ('jc', 'JetClass top jets (550-650 GeV)'))):
        S = R['sets'][ds]['dc']; dv = [dc for dc in dcs if 'models' in S[f'{dc}|top']]
        lo_, hi_ = [], []
        for dc in dv:
            Mm = S[f'{dc}|top']['models']; nets = [m for m in LAB if m in Mm]
            lo_.append(float(np.median([Mm[m]['cr_minus_cphi_ci95'][0] for m in nets]))); hi_.append(float(np.median([Mm[m]['cr_minus_cphi_ci95'][1] for m in nets])))
        ax.fill_between(dv, lo_, hi_, color='0.85', label='random-rotation null (cr − cφ), median of the networks\' 95 % CIs')
        for k, m in enumerate(LAB):
            pts = [(dc, S[f'{dc}|top']['models'][m]) for dc in dv if m in S[f'{dc}|top']['models']]
            if not pts: continue
            x = np.array([p[0] for p in pts]) + 0.012 * (k - 4); y = np.array([p[1]['D_minus_cside'] for p in pts])
            ci = np.array([p[1]['D_minus_cside_ci95'] for p in pts])
            ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], marker='o', ms=4, lw=1.2, capsize=2, color=COL[m], label=LAB[m])
        for ref, c_, nm in (('MLP5c', 'k', 'MLP(m, m_W, τ32, pair masses, |cos θ*|)'), ('MLPx', '#e53e3e', 'MLP(side effects, no pair masses or angle)')):
            v = [[S[f'{dc}|top']['models'][r]['D_minus_cside'] for r in S[f'{dc}|top']['models'] if r.split('@')[0] == ref] for dc in dv]
            ax.fill_between(dv, [min(a) for a in v], [max(a) for a in v], color=c_, alpha=0.18, lw=0)
            ax.plot(dv, [np.median(a) for a in v], ls='--', color=c_, lw=1.6, label=f'{nm}, 5 retrains')
        n_ = [S[f'{dc}|top']['n'] for dc in dv]
        ax.axhline(0, color='0.5', lw=0.8); ax.set_xticks(list(dcs))
        ax.set_title(f'{dsn}; {min(n_)}-{max(n_)} jets per point', fontsize=10, loc='left')
        ax.set_xlabel('requested change of |cos θ*| (W rest frame), m_top and m_W fixed')
    axes[0].set_ylabel('Δ top log-odds: decay-angle edit minus matched control\n(control: prong pT shares matched, τ32 about half matched,\n|cos θ*| and pair masses fixed)')
    h, l = axes[1].get_legend_handles_labels(); fig.legend(h, l, fontsize=7.5, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Top-score change from rotating the W decay angle, minus a control that copies the prong energy sharing at fixed angle;\n'
                 'the MLP without the angle (red) responds about as much as the networks at Δ|cos θ*| = +0.3', fontsize=10.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); fig.savefig(RES / f'{tag}_dalitz.png', dpi=120); print('wrote', RES / f'{tag}_dalitz.png')


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'h32')
