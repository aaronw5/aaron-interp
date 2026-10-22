"""H81 figure: (a) heat map of the response vector r (models x edits, seed replicates as separate rows) on each jet set,
(b) pooled variance shares (dataset, architecture, interaction, seed) with 95 % bootstrap intervals, (c) one Top Landscape top
jet under E1 (collinear split) and E6 (W decay-angle rotation) in (Delta eta, Delta phi) about its own axis.
Reads $AI_RESULTS/physics/h81_stats.json and h81_example.npz; writes h81_dataset_vs_arch.png.
  python -m interp.h81_plot [--smoke]"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'physics'
LAB = dict(JC_ParT_rel='JetClass ParT (released)', JC_ParT_s0='JetClass ParT seed 0', JC_ParT_s1='JetClass ParT seed 1',
           JC_ParT_s2='JetClass ParT seed 2', JC_PN_rel='JetClass ParticleNet (released)', Top_ParT_s0='Top ParT seed 0',
           Top_ParT_s1='Top ParT seed 1', Top_PN_s0='Top ParticleNet seed 0', Top_PN_s1='Top ParticleNet seed 1',
           Top_LNet='Top LorentzNet', Top_PFN='Top PFN', Top_EFN='Top EFN', CMS='m_SD + tau32 cut (CMS WP)', LR_N='LR(N)')
ESHORT = dict(E1='E1 split', E2='E2 soft add', E3='E3 E-scale', E4='E4 boost', E5='E5 WTA axis', E6='E6 W angle', E7='E7 periph.')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s_ = '_smoke' if '--smoke' in sys.argv else ''
    S = json.loads((OUT / f'h81_stats{s_}.json').read_text()); ex = np.load(OUT / f'h81_example{s_}.npz')
    E = S['edits']; names = S['models']
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 1], hspace=0.42, wspace=0.9)
    vmax = max(np.nanmax(np.abs(np.array([S['per_set'][ds]['r'][m] for m in names]))) for ds in ('jc', 'top'))
    vmax = min(vmax, 3.0)
    for k, (ds, title) in enumerate((('jc', 'JetClass jets (TbqqP vs ZJets, 550-650 GeV)'), ('top', 'Top Landscape jets'))):
        ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
        R = np.array([S['per_set'][ds]['r'][m] for m in names])
        im = ax.imshow(R, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                ax.text(j, i, f'{R[i, j]:+.2f}', ha='center', va='center', fontsize=7, color='k' if abs(R[i, j]) < 0.6 * vmax else 'w')
        ax.set_xticks(range(len(E))); ax.set_xticklabels([ESHORT[e] for e in E], rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([LAB.get(m, m) for m in names] if k == 0 else [], fontsize=8)
        for b in (4.5, 8.5) if 'Top_ParT_s1' in names else (4.5, 7.5):
            ax.axhline(b, color='k', lw=0.8)
        ax.set_title(f'{title}\nr = mean[s(edit) - s(control)] / SD(s)', fontsize=9)
        if k == 1: fig.colorbar(im, ax=ax, fraction=0.04, label='r (SD units of the tagger score)')
    # variance shares
    ax = fig.add_subplot(gs[1, 0:2]); D = S['decomp'].get('primary', {})
    facs = ['dataset', 'architecture', 'interaction', 'seed']; cols = ['#3b6fb6', '#d1793b', '#8a8a8a', '#5aa469']
    w = 0.38
    for k, ds in enumerate(('jc', 'top')):
        if ds not in D: continue
        P = D[ds]['pooled']; v = [P[f'share_{f}'] for f in facs]; lo = [P[f'share_{f}_ci'][0] for f in facs]; hi = [P[f'share_{f}_ci'][1] for f in facs]
        x = np.arange(len(facs)) + (k - 0.5) * w
        ax.bar(x, v, w, color=cols, alpha=1.0 if k == 0 else 0.55, edgecolor='k', lw=0.5,
               yerr=[np.subtract(v, lo), np.subtract(hi, v)], capsize=3, label='JetClass jets' if k == 0 else 'Top Landscape jets')
    ax.set_xticks(range(len(facs))); ax.set_xticklabels(['dataset', 'architecture', 'interaction', 'seed'], fontsize=8)
    ax.set_ylabel('share of response variance (7 edits pooled)'); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title('Variance shares of r, 7 edits pooled\n(dark: JetClass jets, light: Top jets; 95 % CI)', fontsize=9)
    # example jet
    for k, (cond, ttl) in enumerate((('E1', 'E1: 10 softest particles split in two (same position)'), ('E6', 'E6: W decay angle rotated (+0.3 in |cos θ*|)'))):
        ax = fig.add_subplot(gs[1, 2 + 2 * k:4 + 2 * k])
        b = lambda c, q: ex[f'{c}|{q}']
        ax.scatter(b('base', 'deta'), b('base', 'dphi'), s=4 + 300 * b('base', 'pt') / b('base', 'pt').max(), facecolors='none',
                   edgecolors='0.4', lw=0.7, label='unedited')
        if cond == 'E1':
            n0 = len(b('base', 'pt')); pe = b('E1', 'pt'); new = np.ones(len(pe), bool)
            sm = np.argsort(b('base', 'pt'))[:10]
            ax.scatter(b('base', 'deta')[sm], b('base', 'dphi')[sm], s=40, marker='x', color='#c0392b', label='split (1 -> 2, half pT each)')
            ax.set_title(f'{ttl}\nN {n0} -> {len(pe)}', fontsize=8)
        else:
            ax.scatter(b('E6', 'deta'), b('E6', 'dphi'), s=4 + 300 * b('E6', 'pt') / b('base', 'pt').max(), color='#c0392b', alpha=0.6, lw=0,
                       label='after E6')
            ax.set_title(ttl, fontsize=8)
        ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_aspect('equal')
        ax.set_xlabel('Δη from jet axis'); ax.set_ylabel('Δφ from jet axis'); ax.legend(fontsize=6, loc='lower left')
    fig.suptitle('H81: response of top taggers to seven physics edits (each minus its control), by training dataset and architecture', fontsize=11)
    p = OUT / f'h81_dataset_vs_arch{s_}.png'; fig.savefig(p, dpi=130, bbox_inches='tight'); print('wrote', p)


if __name__ == '__main__':
    main()
