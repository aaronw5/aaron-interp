"""E26w figure: (a, b) share of the boost's call changes removed by holding one S-level direction at its unboosted value,
lam 0.9 / 1.1, with the variance-matched and isotropic live random-direction null bands; (c) route split a15 -> g49 -> S0;
(d) a real W jet before / after the boost; (e) sum pT class separation at fixed jet mass in the training data.
  python -m interp.e26w_plot   (reads $AI_RESULTS/jedi/e26w_boost_specificity.json; writes results/jedi/e26w_boost_specificity.png)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
ROWS = [('S0', 'S0: sum pT of 14 hardest', 1), ('DAS_M0_k1', 'DAS M0 direction', 1), ('S29', 'S29: pT-weighted r$^2$ sum', 1),
        ('DAS_M2_k1', 'DAS M2 direction', 1), ('PC1', 'PC1 (top variance)', 1),
        ('S0+S29', 'S0 + S29', 2), ('DAS_M0M2_k1', 'DAS M0 + M2', 2), ('PC1-2', 'PC1-2', 2),
        ('DAS_M0M2_k2', 'DAS M0 + M2 (2+2)', 4), ('PC1-4', 'PC1-4', 4)]


def main(tag=''):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = json.loads((OUT / f'e26w_boost_specificity{tag}.json').read_text())
    fig = plt.figure(figsize=(16, 10.5)); gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1], hspace=0.55, wspace=0.42)
    for j, lam in enumerate(('0.9', '1.1')):
        a = fig.add_subplot(gs[0, j]); E = d['lam'][lam]; H = E['holds']; NU = E['nulls_S']; yy = np.arange(len(ROWS))[::-1]
        for y_, (k, lab, dim) in zip(yy, ROWS):
            iso = NU[f'iso_k{dim}']['q']; a.plot([100 * iso[0], 100 * iso[3]], [y_ - 0.28] * 2, color='0.6', lw=5, solid_capstyle='butt')
            vm = NU.get(f'var_{k}')
            if vm and vm['q']: a.plot([100 * vm['q'][0], 100 * vm['q'][3]], [y_ + 0.24] * 2, color='#e69f00', lw=4, solid_capstyle='butt')
            sm_ = NU.get(f'shift_{k}')
            if sm_ and sm_['q']: a.plot([100 * sm_['q'][0], 100 * sm_['q'][3]], [y_ + 0.38] * 2, color='#cc79a7', lw=4, solid_capstyle='butt')
            r = H[k]; col = '#2e5eaa' if k.startswith(('S0', 'S29')) else ('#7a7a7a' if k.startswith('PC') else '#4c9f70')
            a.barh(y_, 100 * r['removed'], 0.34, color=col)
            a.errorbar(100 * r['removed'], y_, xerr=[[100 * (r['removed'] - r['removed_ci'][0])], [100 * (r['removed_ci'][1] - r['removed'])]], fmt='none', ecolor='k', lw=0.8)
        a.set_yticks(yy); a.set_yticklabels([f'{lab}  (k={dim})' for _, lab, dim in ROWS] if j == 0 else [], fontsize=8.5); a.axvline(0, color='k', lw=0.6)
        a.set_xlabel('share of boost call changes removed by the hold (%)')
        a.set_title(f'boost λ = {lam}: jet pT × {lam}, mass fixed; {100 * E["B"]["changed"]:.0f} % of calls change', fontsize=10)
        if j == 0:
            from matplotlib.lines import Line2D
            a.legend(handles=[Line2D([], [], color='#cc79a7', lw=5, label='random live directions, boost-shift-matched (95 % band)'),
                              Line2D([], [], color='#e69f00', lw=5, label='random live directions, variance-matched (95 % band)'),
                              Line2D([], [], color='0.6', lw=5, label='random live directions, isotropic (95 % band)')], fontsize=7.5, loc='upper center', bbox_to_anchor=(0.62, -0.13), ncol=1)
        lo_ = min(-60, 100 * min(H[k]['removed'] for k, _, _ in ROWS) - 5); a.set_xlim(max(lo_, -150), 100)
    # (c) route split
    a = fig.add_subplot(gs[0, 2]); P = d['lam']['0.9']['path']
    rows = [('S0_full', 'S0 held'), ('S0_context_route_reverted', 'S0: route via g49 reverted'), ('S0_direct_route_reverted', 'S0: direct route reverted'),
            ('g49_all_routes', 'g49 held'), ('g49_with_S0_kept_boosted', 'g49 held, S0 boosted'), ('a15_all_routes', 'a15 held'),
            ('a15_via_g_only', 'a15 via g only'), ('a15_not_via_g', 'a15 not via g')]
    yy = np.arange(len(rows))[::-1]
    for y_, (k, lab) in zip(yy, rows):
        r = P[k]; a.barh(y_, 100 * r['removed'], 0.6, color='#2e5eaa' if k.startswith('S0') else ('#b5651d' if k.startswith('g') else '#6a4c93'))
        a.errorbar(100 * r['removed'], y_, xerr=[[100 * (r['removed'] - r['removed_ci'][0])], [100 * (r['removed_ci'][1] - r['removed'])]], fmt='none', ecolor='k', lw=0.8)
    a.set_yticks(yy); a.set_yticklabels([lab for _, lab in rows], fontsize=8.5); a.axvline(0, color='k', lw=0.6)
    a.set_xlabel('share of boost call changes removed (%)'); a.set_title('route split, λ = 0.9', fontsize=10)
    # (d) example jet
    a = fig.add_subplot(gs[1, 0]); ex = d.get('example_jet')
    if ex:
        p0, pB = np.asarray(ex['p0']), np.asarray(ex['pB'])
        a.scatter(p0[:, 1], p0[:, 2], s=4 * np.sqrt(p0[:, 0]) ** 1.6, facecolors='none', edgecolors='#2e5eaa', lw=0.9, label=f'original: m {ex["m0"]:.1f} GeV, sum pT {ex["sumpt0"]:.0f} GeV, call {ex["call0"]}')
        a.scatter(pB[:, 1], pB[:, 2], s=4 * np.sqrt(pB[:, 0]) ** 1.6, color='#d1495b', alpha=0.55, lw=0, label=f'boosted λ {ex["lam"]}: m {ex["mB"]:.1f} GeV, sum pT {ex["sumptB"]:.0f} GeV, call {ex["callB"]}')
        a.set_xlabel('Δη to jet axis'); a.set_ylabel('Δφ to jet axis'); a.set_aspect('equal', 'datalim')
        a.set_title(f'a true W jet (marker area ∝ particle pT); with S0 held the call is {ex["callS0hold"]}', fontsize=10); a.legend(fontsize=7.5, loc='upper left')
    # (e) data check
    a = fig.add_subplot(gs[1, 1:]); D = d.get('data', {}).get('train')
    if D:
        pairs = list(D['auc']); xx = np.arange(len(pairs))
        for off, feat, col, lab in ((-0.27, 'sumpt', '#2e5eaa', 'sum pT (all particles)'), (0, 'sumpt14', '#5fa2dd', 'sum pT (14 hardest)'), (0.27, 'n', '0.6', 'particle multiplicity (reference)')):
            v = [D['auc'][p][feat]['auc'] for p in pairs]; ci = np.array([D['auc'][p][feat]['ci'] for p in pairs])
            a.bar(xx + off, np.asarray(v) - 0.5, 0.25, bottom=0.5, color=col, label=lab)
            a.errorbar(xx + off, v, yerr=[np.asarray(v) - ci[:, 0], ci[:, 1] - np.asarray(v)], fmt='none', ecolor='k', lw=0.8)
        a.axhline(0.5, color='k', lw=0.6); a.axhspan(0.45, 0.55, color='0.9', zorder=0, label='|AUC − 0.5| < 0.05')
        a.set_xticks(xx); a.set_xticklabels([p.replace('_vs_', ' vs ') for p in pairs]); a.set_ylabel('AUC within 5 GeV jet-mass bins')
        a.set_title(f'training data (N = {D["N"]:,}): does sum pT separate classes at fixed jet mass? AUC = P(first class has larger value)', fontsize=10)
        a.legend(fontsize=8, ncol=4, loc='lower left'); a.set_ylim(0, 1)
    fig.suptitle('E26w (JEDI-linear 3-feature n64): at λ 0.9 holding S0 (sum pT) removes more call changes than PC1 and than variance-matched random directions,\n'
                 'but not more than random directions that move as much under the boost; 68 % of the boost change of S0 arrives through context unit g49', fontsize=11.5)
    fn = ROOT / 'results' / 'jedi' / f'e26w_boost_specificity{tag}.png'; fig.savefig(fn, dpi=110, bbox_inches='tight'); print(fn)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '')
