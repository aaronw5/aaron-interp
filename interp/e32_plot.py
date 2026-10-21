"""E32 figure: (left) one real W jet drawn about the stored (exact) axis with the L1-tower (T1) and hardest-particle (T3) axes and the
offset arrow; (right) R_margin (network W-Z margin response to the axis change / response to a matched dilation) vs the axis offset
|c| (quintiles of |c| per edit), main sample 3201 and replicate 3202, with the rotation null (C3) and the R = 1 line.
  python -m interp.e32_plot    -> results/jedi/e32_seeded_axis.png   (reads results/jedi/e32_main.json, e32_rep.json)
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    M = json.loads((RES / 'e32_main.json').read_text()); Rp = json.loads((RES / 'e32_rep.json').read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    e = M['example']; P = np.array(e['particles']); pt, de, dp = P[:, 0], P[:, 1], P[:, 2]
    a1.scatter(de, dp, s=8 + 400 * pt / pt.max(), c='#4a6fa5', alpha=.55, edgecolors='none', label='particles (area $\\propto p_T$)')
    a1.plot(0, 0, 'k+', ms=14, mew=2, label='stored (exact) axis')
    t1, t3 = e['T1_axis'], e['T3_axis']
    a1.plot(*t1, 'o', mfc='none', mec='#c0392b', ms=10, mew=2, label=f'L1-tower axis T1 (|c| = {1e3 * np.hypot(*t1):.0f} mrad)')
    a1.plot(*t3, 's', mfc='none', mec='#e67e22', ms=9, mew=2, label=f'hardest-particle axis T3 (|c| = {1e3 * np.hypot(*t3):.0f} mrad)')
    a1.annotate('', xy=t1, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.6))
    a1.set_xlim(-0.45, 0.45); a1.set_ylim(-0.45, 0.45); a1.set_aspect('equal')
    a1.set_xlabel('$\\Delta\\eta$ about the stored axis'); a1.set_ylabel('$\\Delta\\phi$ about the stored axis (rad)')
    a1.set_title(f'one W jet (m = {e["mass"]:.0f} GeV): the axis error a trigger makes', fontsize=10.5)
    ins = a1.inset_axes([0.63, 0.06, 0.34, 0.34]); ins.scatter(de, dp, s=8 + 400 * pt / pt.max(), c='#4a6fa5', alpha=.55, edgecolors='none')
    ins.plot(0, 0, 'k+', ms=12, mew=2); ins.plot(*t1, 'o', mfc='none', mec='#c0392b', ms=9, mew=2); ins.plot(*t3, 's', mfc='none', mec='#e67e22', ms=8, mew=2)
    ins.annotate('', xy=t1, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.4))
    ins.set_xlim(-0.03, 0.03); ins.set_ylim(-0.03, 0.03); ins.set_xticks([-0.02, 0, 0.02]); ins.set_yticks([-0.02, 0, 0.02]); ins.tick_params(labelsize=6.5, labelbottom=False, labeltop=True)
    ins.set_xlabel('zoom: ±30 mrad', fontsize=7.5)
    a1.legend(loc='upper left', fontsize=7.5, frameon=False)

    a2.axhspan(0.8, 1.2, color='#dddddd', alpha=.6, lw=0, label='0.8–1.2: real off-centre jets (E23wv)')
    a2.axhline(1, color='k', lw=1, ls='--', label='R = 1: responds as to a real mass change')
    col = dict(T1='#c0392b', T2='#8e44ad', T3='#e67e22')
    lab = dict(T1='T1 L1 towers 0.087', T2='T2 towers 0.044', T3='T3 hardest particle')
    for t in ('T1', 'T2', 'T3'):
        for R, mk, off, ls in ((M, 'o', 1.0, '-'), (Rp, 'D', 1.04, ':')):
            Q = R['edits'][t]['by_c_quintile']; x = np.array([q['c_median_mrad'] for q in Q]) * off
            y = np.array([q['R_margin']['R'] for q in Q]); lo = np.array([q['R_margin']['ci'][0] for q in Q]); hi = np.array([q['R_margin']['ci'][1] for q in Q])
            a2.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk + ls, color=col[t], ms=5, capsize=2, lw=1.2,
                        label=lab[t] + (' (sample 3201)' if R is M else ' (replicate 3202)'), mfc=col[t] if R is M else 'white')
        Q = M['edits'][t]['by_c_quintile']; x = np.array([q['c_median_mrad'] for q in Q])
        a2.plot(x, [q['R_margin_C3']['R'] for q in Q], 'x', color='grey', ms=6, label='rotation null C3' if t == 'T1' else None)
    a2.set_xscale('log'); a2.set_xlabel('axis offset |c| (mrad), median of each quintile'); a2.set_ylim(-0.2, 1.3)
    a2.set_ylabel('R = margin response to new axis / to matched dilation')
    a2.set_title('R rises with the axis offset (grey: rotation null, no mass change)', fontsize=10.5)
    a2.legend(fontsize=7, loc='lower right', frameon=False, ncol=1)
    fig.tight_layout(); out = RES / 'e32_seeded_axis.png'; fig.savefig(out, dpi=130); print('wrote', out)


if __name__ == '__main__':
    main()
