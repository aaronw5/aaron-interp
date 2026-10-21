"""E39 figure from results/jedi/e39.json -> results/jedi/e39_stitch.png (local matplotlib, seconds)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / 'results' / 'jedi' / (sys.argv[1] if len(sys.argv) > 1 else 'e39.json')
d = json.loads(J.read_text())
nets = d['nets']; DEG = 'hgq150_s1'
LAB = dict(official_n64='3f n64 (ref)', official_n32='3f n32', official_n128='3f n128', perminv_n64='perm-inv n64', f16_n64='16f n64',
           hgq150_s0='HGQ s0', hgq150_s1='HGQ s1 (57 %)', hgq150_s2='HGQ s2', float_s0='float s0', float_s1='float s1')
ink, ink2 = '#1f2328', '#57606a'
fig = plt.figure(figsize=(15.5, 10.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.45)

for i, L in enumerate(('S', 'h4')):
    ax = fig.add_subplot(gs[0, i]); P = d['pairs'][L]
    M = np.array([[P[f'{A}|{B}']['same_B'] * 100 for B in nets] for A in nets])
    im = ax.imshow(M, vmin=70, vmax=100, cmap='viridis')
    for a in range(len(nets)):
        for b in range(len(nets)):
            ax.text(b, a, f'{M[a, b]:.0f}', ha='center', va='center', fontsize=7.5, color='white' if M[a, b] < 88 else 'black')
    ax.set_xticks(range(len(nets))); ax.set_xticklabels([LAB[n] for n in nets], rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(nets))); ax.set_yticklabels([LAB[n] for n in nets], fontsize=8)
    ax.set_xlabel('recipient B (its later layers)', fontsize=9, color=ink2)
    if i == 0: ax.set_ylabel('donor A (its layers up to the stitch)', fontsize=9, color=ink2)
    t = d['tests'][L]
    ax.set_title(f'({"ab"[i]}) stitch at {L}: % of test jets where the stitched call = B\'s call\n'
                 f'trained donors mean {100 * t["trained_same_B_mean"]:.1f} %, untrained donors {100 * t["untrained_same_B_mean"]:.1f} %',
                 fontsize=9.5, loc='left', color=ink)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('same call as B [%]', fontsize=8)

# (c) disagreement jets: follows A vs B
ax = fig.add_subplot(gs[0, 2])
xs = []; labels = []
for j, L in enumerate(('S', 'h4')):
    P = d['pairs'][L]; fa, fb, nat, sb = [], [], [], []
    for k, v in P.items():
        A, B = k.split('|')
        if A == B or DEG in (A, B) or 'disagree' not in v or v['disagree']['follows_B'] is None: continue
        fa.append(v['disagree']['follows_A'] * 100); fb.append(v['disagree']['follows_B'] * 100)
    for q, (vals, c, nm) in enumerate(((fb, '#2f6fde', 'same call as recipient B'), (fa, '#d1495b', 'same call as donor A'))):
        xpos = j * 3 + q
        ax.scatter(np.full(len(vals), xpos) + np.random.default_rng(q + 2 * j).uniform(-.25, .25, len(vals)), vals, s=9, color=c, alpha=.6,
                   label=nm if j == 0 else None)
        ax.plot([xpos - .35, xpos + .35], [np.mean(vals)] * 2, color=ink, lw=2)
        ax.text(xpos, 103, f'{np.mean(vals):.0f} %', ha='center', fontsize=8.5, color=ink)
ax.set_xticks([0.5, 3.5]); ax.set_xticklabels(['stitch at S', 'stitch at h4'], fontsize=9)
ax.set_ylim(0, 110); ax.set_ylabel('% of the jets on which A and B disagree', fontsize=9)
ax.legend(fontsize=8, loc='lower center', frameon=False)
ax.set_title('(c) jets where the two trained networks call differently:\nwhose call does the stitched network give? (72 pairs, HGQ s1 left out)',
             fontsize=9.5, loc='left', color=ink)

# (d) inheritance index with network-level 95 % intervals (E39x housekeeping: bootstrap over networks, not jets; 0.5 null)
NETS7 = [n for n in d['edit_nets'] if n != DEG]


def netboot_I(rows, key, nb=2000, seed=4949):
    """pooled I over eligible ordered pairs of the 7 non-degenerate edit networks; networks resampled with replacement and
    pairs of distinct networks weighted by the product of their counts -> (estimate, [2.5 %, 97.5 %])"""
    k = len(NETS7); ix = {n: i for i, n in enumerate(NETS7)}; NUM = np.zeros((k, k)); DEN = np.zeros((k, k))
    for rw in rows:
        A, B = rw['pair'].split('|')
        if A not in ix or B not in ix or not rw['eligible']: continue
        rs = rw[key]; dA = rw['r_A'] - rw['r_B']
        NUM[ix[A], ix[B]] = (rs - rw['r_B']) * dA; DEN[ix[A], ix[B]] = dA ** 2
    if DEN.sum() == 0: return None, None
    rng = np.random.default_rng(seed); C = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(nb)]).astype(float)
    n = np.einsum('bi,ij,bj->b', C, NUM, C); dd = np.einsum('bi,ij,bj->b', C, DEN, C); v = n[dd > 0] / dd[dd > 0]
    return NUM.sum() / DEN.sum(), [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


ax = fig.add_subplot(gs[1, :])
resp = [r for r in ('T', 'DW', 'DZ', 'N', 'Kabs') if r in d['inheritance']['S']]
NAME = dict(T='T: 0.06 rad translation\n(W→Z flow, W jets)', DW='D: prong sharing +0.3\n(Δ(ℓW−ℓZ), W jets)', DZ='D: prong sharing +0.3\n(Δ(ℓW−ℓZ), Z jets)',
            D='D: prong sharing +0.3\n(Δ(ℓW−ℓZ), W+Z)', N='N: +1 GeV particle at r = 0.3\n(Δ(ℓg−ℓq), q+g jets)',
            Kabs='K: swap ranks 13/14\n(|Δ log p(true)|)', K='K: swap ranks 13/14\n(Δ log p(true))')
x0 = 0; ticks = []; tl = []
for r in resp:
    for j, L in enumerate(('S', 'h4')):
        I = d['inheritance'][L][r]
        for q, (key, c, nm) in enumerate((('r_stitch', '#2f6fde', 'trained donor A'), ('r_untrained_stitch', '#9a6700', 'untrained redraw of A as donor'),
                                          ('r_random_stitch', '#8c959f', 'random map, same singular values'))):
            v, ci = netboot_I(I['rows'], key)
            xp = x0 + j * 4 + q
            if v is None: continue
            ax.bar(xp, v, color=c, width=.85, label=nm if (r == resp[0] and j == 0) else None)
            if ci: ax.plot([xp, xp], np.clip(ci, -0.55, 2.48), color=ink, lw=1.2)
        ne = sum(rw['eligible'] for rw in I['rows'] if DEG not in rw['pair'])
        ax.text(x0 + j * 4 + 1, -0.72, f'{L}\nn={ne}', ha='center', fontsize=7.5, color=ink2)
    ticks.append(x0 + 3); tl.append(NAME[r]); x0 += 10
ax.axhline(1, color='#d1495b', ls='--', lw=1); ax.axhline(0, color=ink, lw=.8)
ax.axhline(0.5, color='#6e40c9', ls=':', lw=1.5, label='0.5: value for a response unrelated to A and B (no information)')
ax.text(-0.8, 1.06, 'I = 1: responds like donor A', color='#d1495b', fontsize=8); ax.text(-0.8, -0.16, 'I = 0: responds like recipient B', color=ink, fontsize=8)
ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=8.5); ax.set_ylim(-0.8, 2.5)
ax.set_ylabel('pooled inheritance index I', fontsize=9)
ax.legend(fontsize=8, ncol=4, loc='upper left', frameon=False)
ax.set_title('(d) edit responses of stitched networks, I = (r_stitch − r_B) / (r_A − r_B), pooled over pairs with |r_A − r_B| > 2 SE; '
             'lines: 95 % interval from resampling the 7 networks (not jets; clipped at the axis)', fontsize=9.5, loc='left', color=ink)
fig.suptitle('E39: an affine map from one trained JEDI-linear network\'s S or h4 layer into another\'s later layers gives the recipient\'s call '
             'about as often as two trained networks agree with each other (9,599 test jets)', fontsize=11.5, x=0.01, ha='left', color=ink)
out = J.with_name('e39_stitch.png'); fig.savefig(out, dpi=130, bbox_inches='tight'); print(out)
