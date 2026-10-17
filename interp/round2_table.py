import textwrap, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cols = ['node', 'question', 'key numbers (95 % CI)', 'verifier / confidence', 'thesis effect']
W = [7, 30, 62, 30, 26]
WW = [7, 38, 84, 38, 34]
rows = [
 ['E20\nE20v', 'Do the pT-rank windows act causally? Angle swap of adjacent ranks across vs inside a window edge; masks removed, shifted, permuted',
  'Swap 13|14: S43 0.084 [0.081, 0.087], S52 0.056, S63 0.084 per jet; adjacent inside swap 12|13: 0.001-0.004; 14|15: 0. '
  'Masks removed: 95-98 % smaller. Masks shifted -3/+3: peak moves to 10|11 / 16|17 (0.05-0.09). '
  'Same edge peak for S13, S17, S62; none for S1, S20, S51, S56. Permuted masks: 32-36 % accuracy (trained 80.7 %).',
  'reproduced, seed 77; 0.75. Window written in 3 layers (add mask, l3, qs1); E20 omitted the add-layer mask',
  'refines A1/A3: object = cut AND rank window, causal for 6 angular objects; S0/S19 untested'],
 ['E7\nE7v', 'Are the 16 z neurons formulas on the 19 S objects? Same for 87 hand observables; edits without refit',
  'K = 32: calls changed S 9.2-10.4 %, hand 7.4-7.5 %, random S terms 22-23 %. '
  'Translation 0.06 rad, dz slope: S 0.57 [0.57, 0.58], hand 0.12-0.16, hand without mass 0.37; W->Z network 77 %, S 88 %, hand 2-4 %, hand without mass 76 %, random S terms 72-86 %. '
  'Rotation 45 deg slope: S 0.42, hand <= 0.011.',
  'reproduced, seeds 7/71; 0.55. Hand failure needs mass in the basis; W->Z after shift not specific to the fit',
  'refines A5 (approximate, 0.4-0.6 of edit size); B3 basis-dependent (0.75 -> 0.6)'],
 ['E21\nE21v', 'Do the read-outs combine S0 (M0) and S50 (M2) as a product, and at which stage? S-level iso-product vs iso-sum patches',
  'R = iso-prod / iso-sum response: margin 0.44 (seed 47), 0.43 (seed 91); random S pairs 0.36-1.38; per-unit R = random ReLU layers. '
  'Gradient-ratio slope (product 1, sum 0): margin 0.64 [0.59, 0.68], z7 0.94, z0 0.80; random pairs -0.59 to 0.66; random ReLU -0.41 to 0.01. '
  'With S29: margin R 0.28, below all 20 triple controls; calls iso-sum 23.5 % vs iso-prod 7.0 %.',
  'verifier: no rerun (job stalled), 0.6; job later completed: arm A within 0.02; not reviewed by the verifier',
  'weakens "multiplied in h4/h5"; product-like dependence at the read-out, stage open (A4 product 0.35)'],
]
def wrap(s, w): return '\n'.join('\n'.join(textwrap.wrap(p, w)) for p in s.split('\n'))
cells = [[wrap(c, w) for c, w in zip(r, WW)] for r in rows]
nl = [max(c.count('\n') + 1 for c in r) for r in cells]
LH = 0.155
fig_h = 0.9 + sum(LH * n + 0.2 for n in nl) + 0.1
fig = plt.figure(figsize=(17, fig_h), dpi=150)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(fig_h, 0)
tot = sum(W); xs = [0.01]
for w in W: xs.append(xs[-1] + 0.98 * w / tot)
y = 0.15
ax.text(0.01, y, 'Round 2 results: JEDI-linear 3-feature n64 (fixed point, 80.9 %)', fontsize=13, weight='bold', va='top')
y += 0.4
for i, c in enumerate(cols): ax.text(xs[i] + 0.004, y, c, fontsize=10.5, weight='bold', va='top')
y += 0.3; ax.plot([0.01, 0.99], [y, y], color='#333', lw=1)
for r, n in zip(cells, nl):
    y += 0.08
    for i, c in enumerate(r): ax.text(xs[i] + 0.004, y, c, fontsize=9.3, va='top', weight='bold' if i == 0 else 'normal', linespacing=1.3)
    y += LH * n + 0.12; ax.plot([0.01, 0.99], [y, y], color='#bbb', lw=0.7)
fig.savefig('/Users/anrunw/Documents/aaron-interp/results/jedi/round2_table.png', facecolor='white')
