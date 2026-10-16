"""E11 figures from results/jedi/e11_windows.json (needs matplotlib; run with any python that has it)
  e11_swap.png      % calls changed when pT ranks k and k+1 are swapped, three programs, two samples
  e11_controls.png  accuracy of the float program with the trained slot masks vs replaced masks
"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('results/jedi/e11_windows.json'))
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COL = {'fixed': '#eb6834', 'mask': '#2a78d6', 'float': '#1baf7a'}
NAME = {'fixed': 'fixed-point network', 'mask': 'float + trained masks', 'float': 'float, no masks'}
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})

# ---- swap figure: top = float programs (masked, unmasked), bottom = fixed point
fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw=dict(height_ratios=(1.15, 1)))
for axx, progs in ((ax, ('mask', 'float')), (bx, ('fixed',))):
    for p in progs:
        for s in ('seed1', 'seed0'):
            e = R[s]['e']['k']; k = np.array(sorted(int(v) for v in e))
            axx.plot(k, [e[str(i)][p]['chg'] for i in k], color=COL[p], alpha=1.0 if s == 'seed0' else 0.4,
                     lw=2 if s == 'seed0' else 1.2, marker='o' if s == 'seed0' else None, ms=3.2,
                     label=NAME[p] if s == 'seed0' else None)
    axx.grid(axis='y', color=GRID, lw=0.8)
e0 = R['seed0']['e']['k']
ANN = {0: ('u1 (1–63),\nl2 on a4, a21', (2, 0.62)), 9: ('u17 (0–9)', (5.0, 0.42)), 10: ('u13 (0–10)', (11.2, 0.30)),
       13: ('u0, u43, u52, u63 (0–13);\nl3 on t14, t32, t49 (0–13)', (19, 0.70)),
       16: ('l3 on t39, t53 (0–16)', (19, 0.44)), 61: ('l2 on a4 (1–61)', (52.5, 0.30))}
for k, (txt, xy) in ANN.items():
    ax.annotate(txt, (k, e0[str(k)]['mask']['chg']), xytext=xy, fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7, shrinkA=1, shrinkB=3))
ax.set_ylim(-0.04, 0.9); bx.set_ylim(0, 1.6); bx.set_xlim(-1, 63)
ax.set_ylabel('% calls changed'); bx.set_ylabel('% calls changed')
ax.legend(frameon=False, fontsize=9, loc='upper right'); bx.legend(frameon=False, fontsize=9, loc='upper left')
ax.text(0.995, 0.70, 'labels: masks with an edge between slots k and k+1\n(kept slots in brackets)', transform=ax.transAxes,
        ha='right', fontsize=8.5, color=MUTED)
bx.set_xlabel('swapped pT ranks k and k+1 (0 = hardest particle); same particles, only their slot order changes')
fig.suptitle('Swapping ranks k and k+1: the masked float program changes calls at mask edges (k = 0, 9, 10, 13, 16);\n'
             'the unmasked float program at no k; the fixed-point network at every k (0.3–1.4 %)',
             x=0.01, ha='left', fontsize=11.5, color=INK)
fig.text(0.01, 0.01, 'N = 10,000 test jets per sample (2,000 per class), jets with both slots real (10,000 at k = 0, 2,236 at k = 62). '
         'Thick: sample 0 (seed 0); thin: sample 1 (seed 1).', fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig('results/jedi/e11_swap.png', dpi=150); plt.close(fig)

# ---- controls figure
rows = []
for s in ('seed0', 'seed1'):
    b, d = R[s]['base'], R[s]['d']
    rows.append([('fixed-point network', 100 * b['fixed']['acc'], None),
                 ('float + trained masks', 100 * b['mask']['acc'], None),
                 ('masks rolled by −1 slot (all 4 layers)', 100 * d['roll']['-1']['acc'], None),
                 ('masks rolled by +1 slot', 100 * d['roll']['1']['acc'], None),
                 ('float, no slot masks', 100 * R[s]['b']['layer']['all_slot_masks']['acc'], None),
                 ('masks permuted per channel (10 draws)', 100 * d['joint']['perm_acc_mean'], 100 * d['joint']['perm_acc_sd']),
                 ('masks replaced by kept fraction (scale only)', 100 * d['joint']['scale_only']['acc'], None),
                 ('masks reversed in slot order', 100 * d['joint']['reversed']['acc'], None)])
fig, ax = plt.subplots(figsize=(9, 4.4))
lab = [r[0] for r in rows[0]][::-1]; y = np.arange(len(lab))
v0 = [r[1] for r in rows[0]][::-1]; e0_ = [r[2] or 0 for r in rows[0]][::-1]; v1 = [r[1] for r in rows[1]][::-1]
cols = [COL['mask'] if 'trained' in t or 'fixed' in t else '#9a9893' for t in lab]
ax.barh(y, v0, color=cols, height=0.62, xerr=e0_, error_kw=dict(ecolor=INK, lw=1, capsize=3))
ax.scatter(v1, y, marker='|', s=160, color=INK, zorder=3, label='sample 1 (seed 1)')
for yi, v, er in zip(y, v0, e0_): ax.text(v + er + 1.2, yi, f'{v:.1f} %', va='center', fontsize=9, color=INK)
ax.axvline(20, color=MUTED, lw=0.8, ls=':'); ax.text(20.5, -0.75, 'chance 20 %', fontsize=8.5, color=MUTED)
ax.set_yticks(y, lab, fontsize=9.5); ax.set_xlim(0, 95); ax.set_ylim(-0.9, len(lab) - 0.5); ax.set_xlabel('accuracy on 10,000 test jets (%)')
fig.suptitle('Accuracy with the trained slot masks vs masks with the same kept count per channel (qs, l2, l3, qs1)',
             x=0.01, ha='left', fontsize=11.5, color=INK)
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.grid(axis='x', color=GRID, lw=0.8)
fig.tight_layout(); fig.savefig('results/jedi/e11_controls.png', dpi=150); plt.close(fig)
print('saved results/jedi/e11_swap.png, results/jedi/e11_controls.png')
