"""E26 figure: (left) share of the boost's call changes removed when one object is held at its unboosted value, lam 0.9 and 1.1,
with the random-S-subspace null; (right) change of the h4 / z read-out of log m under boost B, pT-only P and angle-only A.
  python -m interp.e26_plot   (reads $AI_RESULTS/jedi/e26_boost_route.json; writes results/jedi/e26_boost_route.png)"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = json.loads((OUT / 'e26_boost_route.json').read_text())
    rows = [('a_all_live', 'all 13 particle cuts (sanity)', 'holds'), ('S_M0M2_k1', 'M0 + M2 DAS directions at S (2 dims)', 'das_holds'),
            ('S_M0_k1', 'M0 DAS direction at S (1 dim)', 'das_holds'), ('S0', 'S0: ΣpT of the 14 hardest particles', 'holds'),
            ('g49', 'g49: context unit fed by ΣpT', 'holds'), ('a15', 'cut a15: linear in particle pT', 'holds'),
            ('a_ramps', '4 pT-ramp cuts (a15, a21, a32, a56)', 'holds'), ('a32', 'cut a32: on for pT < 11 GeV', 'holds'),
            ('clamp', 'boost with 11 GeV crossings clamped', 'clamp'), ('S19', 'S19: soft multiplicity', 'holds'),
            ('S_M2_k1', 'M2 DAS direction at S (1 dim)', 'das_holds')]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]; yy = np.arange(len(rows))[::-1]
    for j, (lam, col, off) in enumerate((('0.9', '#2e5eaa', 0.18), ('1.1', '#d1495b', -0.18))):
        E = d['lam'][lam]; v, lo, hi = [], [], []
        for k, _, src in rows:
            r = E['clamp'] if src == 'clamp' else E[src][k]; v.append(100 * r['removed']); lo.append(100 * r['removed_ci'][0]); hi.append(100 * r['removed_ci'][1])
        a.barh(yy + off, v, 0.34, color=col, label=f'boost λ = {lam} (pT × {lam}, mass fixed): {100 * E["B"]["changed"]:.0f} % of calls change')
        a.errorbar(v, yy + off, xerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt='none', ecolor='k', lw=0.7)
        q = E['null_subspace']['S']['1']['q']
        a.axvspan(100 * q[0], 100 * q[2], color=col, alpha=0.08)
    a.axvline(0, color='k', lw=0.6); a.set_yticks(yy); a.set_yticklabels([r[1] for r in rows], fontsize=8.5)
    a.set_xlabel('boost call changes removed by holding the object at its unboosted value (%)')
    a.set_xlim(-80, 105); a.legend(fontsize=8.5, frameon=False, loc='upper center', bbox_to_anchor=(0.4, -0.11), ncol=1)
    a.set_title('Which held object removes the boost response\n(shaded: 95 % band of random 1-dim S directions; bars: 95 % bootstrap)', fontsize=9.5)
    b = ax[1]; lams = np.array(d['lams'])
    for st, ls in (('h4', '-'), ('z', '--')):
        P = d['mass_direction'][st]['per_lam']
        for E_, col, lab in (('B', '#2e5eaa', 'boost (m fixed)'), ('P', '#d1495b', 'pT × λ (m × λ)'), ('A', '#edae49', 'angles / λ (m / λ)')):
            b.plot(lams, P[E_], ls, marker='o', color=col, label=f'{lab}, read-out at {st}')
    b.plot(lams, np.log(lams), ':', color='grey', label='true Δ log m for pT × λ')
    b.axhline(0, color='k', lw=0.6); b.set_xlabel('jet pT factor λ'); b.set_ylabel('mean change of the read-out of log m (jets of half B)')
    b.legend(fontsize=7.5, frameon=False); b.set_title('Mass read-out: h4 does not move under the boost;\nz moves up by ~0.03 for every λ', fontsize=9.5)
    fig.suptitle('E26 JEDI-linear n64, 10,000 jets: the boost response at λ 0.9 runs through ΣpT (S0, g49, cut a15), not through the mass read-out',
                 fontsize=10.5)
    fig.tight_layout(); p = OUT / 'e26_boost_route.png'; fig.savefig(p, dpi=130); print('wrote', p)


if __name__ == '__main__':
    main()
