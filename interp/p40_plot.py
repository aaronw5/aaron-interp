"""P40 figure from $AI_RESULTS/part/p40_neuron_atlas.json (no network runs).
(a) share of the 1,024 post-GELU class-MLP units that are 'clean' (one label above the untrained-net threshold) and
    'selective' (its label's family leads the next family by >= 0.1), for neurons, raw / standardised random rotations
    (10 seeds; range), untrained and weight-permuted nets; ParT_full and ParT_kin.
(b) among steered units whose +2 s.d. step moves the target-class log-odds by >= 0.05, the share moving it in the sign the
    label predicts: neurons vs same-index rotated directions vs matched-s.d. other-label neurons (post hoc subset).
(c) joint patch of the mass-labelled units to their values on 5 %-dilated jets: share R of the W log-odds response to the
    real dilation carried by them, vs the same number of random rotated directions (10 draws, range).
  ./job.sh c-p40-plot 'python -m interp.p40_plot'  -> results/part/p40_neuron_atlas.png
"""
import json, os
from pathlib import Path
import numpy as np

D = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((D / 'p40_neuron_atlas.json').read_text()); M = R['models']
    f, ax = plt.subplots(1, 3, figsize=(16, 4.9))
    a = ax[0]; w = 0.38
    for k, feat in enumerate(('full', 'kin')):
        at = M[f'ParT_{feat}']['atlas']['post']['thr_untrained']
        groups = [('neurons', at['neuron']['frac'], at['neuron']['frac_selective'], None, None),
                  ('rotated\n(raw)', at['rot_raw']['mean'], at['rot_raw']['frac_selective_mean'], (at['rot_raw']['min'], at['rot_raw']['max']), at['rot_raw']['frac_selective_all']),
                  ('rotated\n(std.)', at['rot_std']['mean'], at['rot_std']['frac_selective_mean'], (at['rot_std']['min'], at['rot_std']['max']), at['rot_std']['frac_selective_all']),
                  ('untrained', M[f'ParT_{feat}_untrained']['atlas']['post']['thr_untrained']['frac'], M[f'ParT_{feat}_untrained']['atlas']['post']['thr_untrained']['frac_selective'], None, None),
                  ('permuted', M[f'ParT_{feat}_permuted']['atlas']['post']['thr_untrained']['frac'], M[f'ParT_{feat}_permuted']['atlas']['post']['thr_untrained']['frac_selective'], None, None)]
        x0 = k * 6
        for j, (nm, c, s, cr, sa) in enumerate(groups):
            a.bar(x0 + j - w / 2, c, w, color='#9bb8d3', label='clean (one label above threshold)' if k == 0 and j == 0 else None)
            a.bar(x0 + j + w / 2, s, w, color='#1f4e79', label='selective (one label family leads by >= 0.1)' if k == 0 and j == 0 else None)
            if sa: a.plot([x0 + j + w / 2] * 2, [min(sa), max(sa)], 'k-', lw=1)
        a.text(x0 + 2, 1.06, f'ParT_{feat}', ha='center', fontsize=9)
    a.set_xticks([x for k in range(2) for x in range(k * 6, k * 6 + 5)]); a.set_xticklabels([g for _ in range(2) for g in ('neurons', 'rot.\nraw', 'rot.\nstd', 'untr.', 'perm.')], fontsize=7)
    a.set_ylim(0, 1.12); a.set_ylabel('share of the 1,024 class-MLP units'); a.legend(fontsize=7, loc='center right')
    a.set_title('(a) neurons are not more readable than rotated directions', fontsize=9)
    a = ax[1]; xs, labs = [], []
    for k, feat in enumerate(('full', 'kin')):
        rows = M[f'ParT_{feat}']['write_units']
        for j, (nm, g) in enumerate((('neurons', lambda r: r['steer']), ('rotated', lambda r: r['rot']['steer']), ('matched', lambda r: r['matched']['steer']))):
            dl = np.array([g(r)['2']['dl'] for r in rows]); s = np.array([r['s'] for r in rows]); big = np.abs(dl) >= 0.05
            fr = float((s * dl > 0)[big].mean()); n = int(big.sum()); x = k * 4 + j
            a.bar(x, fr, 0.7, color=['#1f4e79', '#c55a11', '#7f7f7f'][j]); a.text(x, fr + 0.02, f'{int(round(fr * n))}/{n}', ha='center', fontsize=7)
            xs.append(x); labs.append(f'{feat}\n{nm}')
    a.axhline(0.5, color='k', lw=0.7, ls='--'); a.set_xticks(xs); a.set_xticklabels(labs, fontsize=7); a.set_ylim(0, 1.05)
    a.set_ylabel('share moving the target class the predicted way')
    a.set_title('(b) units with a visible effect (|change of log-odds| >= 0.05 at +2 s.d.)\n(300 random clean units per model; dashed = chance)', fontsize=9)
    a = ax[2]
    for k, feat in enumerate(('full', 'kin')):
        C = M[f'ParT_{feat}']['calibration']; j = C['joint']['Wqq']; rd = C['joint_random_dirs']['Wqq']
        a.bar(k * 3, j['R']['v'], 0.8, color='#1f4e79', label=f'{C["n_mass_units"]} mass-labelled units' if k == 0 else None)
        a.errorbar(k * 3, j['R']['v'], yerr=[[j['R']['v'] - j['R']['ci'][0]], [j['R']['ci'][1] - j['R']['v']]], color='k', capsize=3)
        a.bar(k * 3 + 1, rd['mean'], 0.8, color='#7f7f7f', label='same number of random directions' if k == 0 else None)
        a.plot([k * 3 + 1] * 2, [rd['min'], rd['max']], 'k-', lw=1)
    a.axhline(0, color='k', lw=0.6); a.set_xticks([0.5, 3.5]); a.set_xticklabels(['ParT_full', 'ParT_kin'])
    a.set_ylabel('share R of the W log-odds change\nunder a real 5 % mass increase'); a.legend(fontsize=7)
    a.set_title('(c) mass-labelled units carry 17-30 % of the\nresponse to a real mass change (angles x 1.05)', fontsize=9)
    f.tight_layout(); f.savefig(D / 'p40_neuron_atlas.png', dpi=130); print('wrote', D / 'p40_neuron_atlas.png')


if __name__ == '__main__':
    main()
