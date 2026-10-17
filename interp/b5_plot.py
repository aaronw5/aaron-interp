"""B5 summary figure from results/jedi/b5.json: same call and calls changed under the shift, WTA re-axis and axis jump, for the
direct shared-budget programs on P_axis / P_inv / P_all vs program size; network (orange band) and nested programs (lines).
    python -m interp.b5_plot            -> results/jedi/b5_summary.png"""
import json, os, sys
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')) / 'jedi'


def main(prefix='b5'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = json.loads((OUT / f'{prefix}.json').read_text()); rows = R['models']; net = R['network']['edits']
    col = dict(axis='#1c5cab', inv='#c0392b', all='#2a9d3c')
    lab = dict(axis='P_axis: observables about the jet axis + pT', inv='P_inv: m, EFPs + pT (translation invariant)', all='P_axis + P_inv')
    panels = [('same call with the network (%)', None), ('0.06 rad shift in Δη: calls changed (%)', 'trans'),
              ('re-axis on WTA axis: calls changed (%)', 'wta'), (f'axis jump: calls changed (%), N {net["jump"]["N"]} jets', 'jump')]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, (title, e) in zip(axs, panels):
        if e is not None:
            n = net[e]['flip']; ax.axhspan(n[1] * 100, n[2] * 100, color='#f2b134', alpha=0.45, lw=0)
            ax.axhline(n[0] * 100, color='#b07800', lw=1.8, label='network (95 % CI band)')
        for nm, ls in (('mars30', '--'), ('mars16', ':')):
            r = next(r for r in rows if r['name'] == nm); v = (r['same_call_test'] if e is None else r['edits'][e]['flip'])[0] * 100
            ax.axhline(v, color='#444444', lw=1.2, ls=ls, label=f'nested program {nm} ({r["dl"]:,} terms)')
        for pool in ('axis', 'inv', 'all'):
            rr = sorted([r for r in rows if r['pool'] == pool], key=lambda r: r['dl'])
            x = [r['dl'] for r in rr]; v = np.array([r['same_call_test'] if e is None else r['edits'][e]['flip'] for r in rr]) * 100
            ax.plot(x, v[:, 0], marker='o', ms=5, lw=1.8, color=col[pool], label=lab[pool]); ax.fill_between(x, v[:, 1], v[:, 2], color=col[pool], alpha=0.18, lw=0)
        ax.set_title(title, fontsize=10.5, loc='left'); ax.set_xscale('log'); ax.set_xticks([5, 10, 20, 30, 50]); ax.set_xticklabels(['5', '10', '20', '30', '50'])
        ax.minorticks_off(); ax.set_xlabel('hinge terms in the program (all 5 logits)', fontsize=9); ax.grid(alpha=0.3, lw=0.5); ax.tick_params(labelsize=9)
    axs[0].plot([], [], color='#b07800', lw=1.8, label='network, 95 % CI band (edit panels)')
    axs[0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'JEDI-linear, test jets N {R["splits"]["test"]:,}: programs on invariant observables (red) change no call under the shift or the re-axis; '
                 'the network changes 56 % and 30 %', fontsize=11.5, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93)); p = OUT / f'{prefix}_summary.png'; fig.savefig(p, dpi=130); print('wrote', p)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'b5')
