"""E39x figure from results/jedi/e39x.json -> results/jedi/e39x_rank.png (local matplotlib, seconds)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / 'results' / 'jedi' / (sys.argv[1] if len(sys.argv) > 1 else 'e39x.json')
d = json.loads(J.read_text())
ink, ink2 = '#1f2328', '#57606a'
RANKS = d['ranks']; TL = d['task_ranks']
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.3)


def ci(o): return o.get('ci') or [o['est'], o['est']]


for i, L in enumerate(('S', 'h4')):
    ax = fig.add_subplot(gs[0, i]); o = d['rank_summary'][L]
    lg = o['logit']; ax.axhspan(*[100 * v for v in ci(lg)], color='#d1495b', alpha=.15)
    ax.axhline(100 * lg['est'], color='#d1495b', lw=1.5, label=f'logit-only donor (A\'s 5 class scores) {100 * lg["est"]:.0f} %')
    nat = o['native']; ax.axhline(100 * nat['est'], color=ink, ls='--', lw=1, label=f'A and B agree unedited {100 * nat["est"]:.0f} %')
    for key, rr, c, nm, mk in (('rank', RANKS, '#2f6fde', 'rank-r map fitted to reproduce B\'s layer', 'o'),
                               ('task', TL, '#14866d', 'rank-r map trained on B\'s classification loss', 's'),
                               ('untrained', RANKS, '#9a6700', 'untrained redraw of A as donor', '^'),
                               ('random', RANKS, '#8c959f', 'random map, same singular values', 'v')):
        est = [100 * o[f'{key}_{r}']['est'] for r in rr]; lohi = np.array([[100 * v for v in ci(o[f'{key}_{r}'])] for r in rr])
        ax.errorbar(rr, est, yerr=[np.array(est) - lohi[:, 0], lohi[:, 1] - np.array(est)], color=c, marker=mk, ms=5, lw=1.5, capsize=3, label=nm)
    ax.set_xscale('log', base=2); ax.set_xticks(RANKS); ax.set_xticklabels([str(r) if r < 64 else '64\n(full)' for r in RANKS])
    ax.set_ylim(0, 100); ax.set_xlabel('rank r of the map (number of dimensions passed from A to B)', fontsize=9)
    ax.set_ylabel('same call as recipient B [% of test jets]', fontsize=9)
    ax.set_title(f'({"ab"[i]}) stitch at {L}: B\'s call vs map rank\nmean over 72 network pairs; bars: 95 % interval\nfrom resampling the 9 networks',
                 fontsize=9.5, loc='left', color=ink)
    if i == 0: ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0, -0.22), ncol=3, frameon=False)

# (c) subset stitch
ax = fig.add_subplot(gs[0, 2]); sub = d.get('subset', {})
if 'rows' in sub:
    rows = sub['rows']; k = np.array([r['k'] for r in rows]); nm_ = np.array([100 * r['named'] for r in rows]); rd = np.array([100 * r['random_mean'] for r in rows])
    q5 = np.array([100 * r['random_q05'] for r in rows]); q95 = np.array([100 * r['random_q95'] for r in rows])
    o = np.argsort(nm_ - rd); xs = np.arange(len(rows))
    ax.vlines(xs, q5[o], q95[o], color='#8c959f', lw=2, alpha=.6, label='random unit sets of the same size (5–95 %)')
    ax.scatter(xs, rd[o], color='#57606a', s=10, label='random sets, mean')
    ax.scatter(xs, nm_[o], color='#2f6fde', s=14, label='named role units (r2, soft count, hard window, core)')
    am = np.mean(list(sub['all_at_mean_same_B'].values())) * 100
    ax.axhline(am, color=ink, ls=':', lw=1, label=f'all of B\'s S held at its mean {am:.0f} %')
    nb = sub['named_minus_random_netboot']
    ax.set_title(f'(c) stitch through k = 1–4 S units only (rest of B at its mean)\nnamed − random {100 * nb["est"]:+.1f} points '
                 f'[{100 * ci(nb)[0]:+.1f}, {100 * ci(nb)[1]:+.1f}] (network resampling)\n{sub["n_pairs"]} pairs; per-pair p ≥ 1/201, so Holm over pairs cannot pass 0.05',
                 fontsize=9.5, loc='left', color=ink)
    ax.set_xlabel('network pair (sorted by named − random)', fontsize=9); ax.set_ylabel('same call as recipient B [% of test jets]', fontsize=9)
    ax.set_ylim(0, 100); ax.legend(fontsize=7.5, loc='upper left', frameon=False)

# (d) inheritance
ax = fig.add_subplot(gs[1, :])
NAME = dict(T='T: 0.06 rad translation\n(W→Z flow, W jets)', DW='D: prong sharing +0.3\n(Δ(ℓW−ℓZ), W jets)', DZ='D: prong sharing +0.3\n(Δ(ℓW−ℓZ), Z jets)',
            N='N: +1 GeV particle at r = 0.3\n(Δ(ℓg−ℓq), q+g jets)')
V = (('full', '#2f6fde', 'full-rank map from trained A'), ('rank4', '#14866d', 'rank-4 map from trained A'),
     ('untrained', '#9a6700', 'untrained redraw of A'), ('logit', '#d1495b', 'A\'s 5 class scores only'))
x0 = 0; ticks = []; tl = []
for r in ('T', 'DW', 'DZ', 'N'):
    for j, L in enumerate(('S', 'h4')):
        I = d['inheritance'][L][r]
        for q, (key, c, nm) in enumerate(V):
            xp = x0 + j * 5 + q; o = I[key]
            if o['est'] is None: continue
            ax.bar(xp, o['est'], color=c, width=.85, label=nm if (r == 'T' and j == 0) else None)
            lo, hi = ci(o); ax.plot([xp, xp], np.clip([lo, hi], -0.95, 2.45), color=ink, lw=1.2)
        ax.text(x0 + j * 5 + 1.5, -1.25, f'{L}\nn={I["n_eligible"]}', ha='center', fontsize=7.5, color=ink2)
    ticks.append(x0 + 4); tl.append(NAME[r]); x0 += 12
ax.axhline(1, color='#d1495b', ls='--', lw=1); ax.axhline(0, color=ink, lw=.8)
ax.axhline(0.5, color='#6e40c9', ls=':', lw=1.5, label='0.5: value for a response unrelated to A and B')
ax.text(-0.8, 1.05, 'I = 1: responds like donor A', color='#d1495b', fontsize=8); ax.text(-0.8, -0.15, 'I = 0: responds like recipient B', color=ink, fontsize=8)
ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=8.5); ax.set_ylim(-1.35, 2.5); ax.set_ylabel('pooled inheritance index I', fontsize=9)
ax.legend(fontsize=8, ncol=5, loc='upper left', frameon=False)
ax.set_title('(d) edit responses through stitched networks: I = (r_stitch − r_B) / (r_A − r_B) over pairs with |r_A − r_B| > 2 SE; '
             'lines: 95 % interval from resampling the 7 networks (clipped at the axis)', fontsize=9.5, loc='left', color=ink)
o = d['rank_summary']['S']
fig.suptitle(f'E39x: how many dimensions does one trained JEDI-linear network need to pass to another? At S, rank 4 gives B\'s call on '
             f'{100 * o["rank_4"]["est"]:.0f} % of jets, full rank {100 * o["rank_64"]["est"]:.0f} %, class scores alone {100 * o["logit"]["est"]:.0f} % '
             f'({d["sample"]["N_test"]:,} test jets)', fontsize=11.5, x=0.01, ha='left', color=ink)
out = J.with_name('e39x_rank.png' if 'smoke' not in J.name else 'e39x_rank_smoke.png'); fig.savefig(out, dpi=130, bbox_inches='tight'); print(out)
