"""E39x verification figure from results/jedi/e39x_verify.json -> results/jedi/e39x_verify.png (local matplotlib, seconds)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / 'results' / 'jedi' / 'e39x_verify.json'
d = json.loads(J.read_text()); ink, ink2 = '#1f2328', '#57606a'
fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.25))

ax = axs[0]
bars = (('ridge_4', 'A\'s S, rank 4,\nfitted to copy B\'s layer', '#2f6fde'), ('ridge_logit', 'A\'s 5 logits,\nfitted to copy B\'s layer', '#d1495b'),
        ('task_S4', 'A\'s S, rank 4,\ntrained on B\'s loss', '#14866d'), ('task_logit5', 'A\'s 5 logits,\ntrained on B\'s loss', '#e07a5f'),
        ('task_S64', 'A\'s S, full rank,\ntrained on B\'s loss', '#0b5d4b'), ('task_untr4', 'untrained A\'s S, rank 4,\ntrained on B\'s loss', '#9a6700'))
w = 0.13
for j, L in enumerate(('S', 'h4')):
    o = d['summary'][L]
    for q, (k, nm, c) in enumerate(bars):
        x = j + (q - 2.5) * w; v = o[k]; ax.bar(x, 100 * v['est'], w * 0.9, color=c, label=nm if j == 0 else None)
        ax.plot([x, x], [100 * v['ci'][0], 100 * v['ci'][1]], color=ink, lw=1.2)
        ax.text(x, 100 * v['ci'][1] + 1, f'{100 * v["est"]:.0f}', ha='center', fontsize=7.5, color=ink)
ax.axhline(100 * d['summary']['S']['native']['est'], color=ink, ls='--', lw=1, label='A and B agree unedited (89 %)')
ax.set_xticks([0, 1]); ax.set_xticklabels(['stitch at S (second-round objects)', 'stitch at h4 (first dense layer)'], fontsize=9)
ax.set_ylim(0, 105); ax.set_ylabel('same call as recipient B [% of 9,643 test jets]', fontsize=9)
ax.legend(fontsize=7.5, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.36), frameon=False)
ax.set_title('(a) what A passes to B: under B\'s own loss, A\'s 5 class scores give B\'s call as often as 4 directions of A\'s S\n'
             '56 ordered pairs of 8 networks; bars: 95 % interval from resampling the networks', fontsize=9.5, loc='left', color=ink)

ax = axs[1]; s = d['subset']; ks = [4, 3]
cond = (('named', 'A\'s named units -> B\'s named units', '#2f6fde'), ('randA_namedB', 'random A units -> B\'s named units', '#8fb3f0'),
        ('namedA_randB', 'A\'s named units -> random B units', '#c9a227'), ('randAB', 'random A units -> random B units (E39x control)', '#8c959f'),
        ('own_named_B', 'B\'s own named units, no stitch', '#1f2328'))
for j, k in enumerate(ks):
    for q, (key, nm, c) in enumerate(cond):
        x = j + (q - 2) * 0.16; v = 100 * s['by_k'][str(k)][key]
        ax.bar(x, v, 0.144, color=c, label=nm if j == 0 else None); ax.text(x, v + 1, f'{v:.0f}', ha='center', fontsize=7.5, color=ink)
ax.axhline(19.5, color=ink2, ls=':', lw=1, label='all of B\'s S at its mean (~20 %)')
ax.set_xticks(range(len(ks))); ax.set_xticklabels([f'k = {k} shared roles\n({s["by_k"][str(k)]["n"]} pairs)' for k in ks], fontsize=9)
ax.set_ylim(0, 60); ax.set_ylabel('same call as recipient B [% of test jets]', fontsize=9)
ax.legend(fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, -0.42), frameon=False, ncol=2)
nm_ = s['named_minus_randA_namedB']; ch = s['randA_namedB_minus_randAB']
ax.set_title(f'(b) only k S units passed, rest of B\'s S at its mean. Of the +{100 * s["named_minus_randAB"]["est"]:.1f}-point named gap,\n'
             f'+{100 * ch["est"]:.1f} comes from which B units are kept, +{100 * nm_["est"]:.1f} [{100 * nm_["ci"][0]:.1f}, {100 * nm_["ci"][1]:.1f}] from A\'s named units',
             fontsize=9.5, loc='left', color=ink)
fig.suptitle('E39x verification (fresh jets, seeds 6941/6942): the rank-4 result is matched by A\'s class scores under B\'s loss', fontsize=11.5,
             x=0.01, ha='left', color=ink)
out = J.with_name('e39x_verify.png'); fig.savefig(out, dpi=130, bbox_inches='tight'); print(out)
