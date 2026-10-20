"""X12 figure: the three program elements per model (official, retrained HGQ seeds, pipeline-matched float seeds).
    python -m interp.x12_plot [--ckpt final]      reads $AI_RESULTS/jedi/x12_summary_<ckpt>.json -> x12_<ckpt>.png
Panels: (a) pure-cut fraction among nonzero l1 units vs the column-permutation null (median, 97.5 %);
(b) rank-window swap ratio (across a qs1 mask edge / inside a window, 95 % jet-bootstrap CI) vs the mask-keeping weight-permuted
null p90 (HGQ) - float models use ranks 13|14 vs 5|6; (c) W -> Z fraction under a 0.06 rad translation along +deta, a random-
direction translation of the same length and the M2-held translation."""
import argparse, json, os
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')) / 'jedi'
LAB = {'official': 'official\nHGQ', 'hgq-r150_s0': 'HGQ-150\ns0', 'hgq-r150_s1': 'HGQ-150\ns1',
       'hgq-r150_s2': 'HGQ-150\ns2', 'hgq_s0': 'HGQ-50\ns0', 'hgq_s1': 'HGQ-50\ns1', 'hgq_s2': 'HGQ-50\ns2',
       'float_s0': 'float\ns0', 'float_s1': 'float\ns1'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='final'); a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    S = json.loads((OUT / f'x12_summary_{a.ckpt}.json').read_text()); R = S['rows']; ms = [m for m in LAB if m in R]
    xs = np.arange(len(ms)); fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), dpi=130)
    # (a)
    P = [R[m]['P_nz'] for m in ms]; nq = [R[m]['P_null_q'] for m in ms]
    ax[0].bar(xs, P, color=['#444' if m == 'official' else ('#2a6fbb' if m.startswith('hgq') else '#d0812a') for m in ms], width=.55)
    ax[0].errorbar(xs, [q[1] for q in nq], yerr=[[0] * len(ms), [q[3] - q[1] for q in nq]], fmt='_', color='k', ms=22, capsize=4,
                   label='permutation null (median, 97.5 %)')
    for i, m in enumerate(ms): ax[0].text(i, P[i] + .02, f'p={R[m]["P_p"]:.2g}', ha='center', fontsize=8)
    ax[0].set_ylim(0, 1.12); ax[0].set_ylabel('fraction of nonzero first-layer units\nthat are pT-only or angle-only')
    ax[0].set_title('(a) pure cuts'); ax[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.27))
    # (b)
    for i, m in enumerate(ms):
        w = R[m]['window']; ci = w.get('ci') or [None, None]
        if w['stat'] is None: continue
        if w['stat'] <= 0.1:   # zero response (no change across the edge): drawn at the axis floor
            ax[1].plot(i, 0.12, 'v', color='k'); ax[1].text(i, 0.15, '0', ha='center', fontsize=8)
        else:
            ax[1].errorbar(i, w['stat'], yerr=[[w['stat'] - ci[0]], [ci[1] - w['stat']]] if ci[0] is not None else None, fmt='o', color='k', capsize=4)
        if w.get('null_p90') is not None: ax[1].plot([i - .25, i + .25], [w['null_p90']] * 2, color='r', lw=2)
        ax[1].text(i, 13, f'{R[m]["qs1_windows"]}', ha='center', fontsize=8)
    ax[1].text(-0.6, 16.5, 'window channels:', fontsize=8)
    ax[1].axhline(1, color='grey', ls=':'); ax[1].set_yscale('log'); ax[1].set_ylim(0.1, 22)
    ax[1].set_ylabel('response to angle swap:\nacross mask edge / inside window'); ax[1].set_title('(b) rank windows (red bar: weight-permuted null, 90th pct.)', fontsize=10)
    # (c)
    for j, (k, c) in enumerate((('WZ_T', '#2a6fbb'), ('WZ_Tr', '#7fb0e0'), ('WZ_M2held', '#d0812a'))):
        ax[2].bar(xs + (j - 1) * .26, [R[m][k] for m in ms], width=.26, color=c,
                  label={'WZ_T': 'translate 0.06 rad (+Δη)', 'WZ_Tr': 'translate 0.06 rad (random dir.)', 'WZ_M2held': 'translate, Σ pT r² held'}[k])
    ax[2].set_ylim(0, 1); ax[2].set_ylabel('fraction of true W jets called Z'); ax[2].set_title('(c) un-recentred mass response')
    ax[2].legend(fontsize=8, loc='upper right')
    for x_ in ax:
        x_.set_xticks(xs); x_.set_xticklabels([LAB[m] + f'\n{100 * R[m]["acc_test"]:.1f} %' + ('*' if R[m]['acc_test'] < 0.7 else '') for m in ms], fontsize=8)
    fig.suptitle('Retrained JEDI-linear n64: the mass response recurs in HGQ copies, the official rank-window effect does not '
                 '(test accuracy under each label; * degenerate model)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); p = OUT / f'x12_{a.ckpt}.png'; fig.savefig(p); print('wrote', p)


if __name__ == '__main__':
    main()
