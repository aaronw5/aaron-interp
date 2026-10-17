"""Diagram of where JEDI-linear forms its mass variable (numbers from E5, E5v, E6, E6v, E11). python site/explain_mass_fig.py"""
import pathlib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parents[1] / 'results' / 'site'; OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(15, 6.2), dpi=150); ax.axis('off'); ax.set_xlim(0, 15); ax.set_ylim(0, 6.2)
C = dict(stage='#eef2f6', m0='#f6e7da', m2='#e2edf6', prod='#e8f1e7', win='#f3eed9')


def box(x, y, w, h, text, fc, fs=9.5, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12', fc=fc, ec='#7a8796', lw=1))
    lines = text.split('\n')
    ax.text(x + 0.12, y + h - 0.16, lines[0], va='top', ha='left', fontsize=fs + 0.5, weight='bold' if bold_first else 'normal')
    if len(lines) > 1: ax.text(x + 0.12, y + h - 0.48, '\n'.join(lines[1:]), va='top', ha='left', fontsize=fs, linespacing=1.35)


def arrow(x0, y0, x1, y1, text=None):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', lw=1.2, color='#55606e'))
    if text: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08, text, ha='center', fontsize=8.5, color='#55606e')


# stage row
stages = [('particles', 'pT, Δη, Δφ\nof 64 particles'), ('P', 'first-round\nobjects'), ('g', 'jet context'), ('S', 'second-round\nobjects'),
          ('h4, h5', 'dense layers'), ('z', '16 class\nneurons'), ('scores', 'g, q, W, Z, t')]
xs = [0.2, 2.25, 4.3, 6.35, 8.4, 10.6, 12.8]
for (n, t), x in zip(stages, xs): box(x, 5.0, 1.75, 1.0, f'{n}\n{t}', C['stage'])
for a, b in zip(xs[:-1], xs[1:]): arrow(a + 1.8, 5.5, b - 0.05, 5.5)

# M0 path
box(2.25, 3.2, 1.75, 1.35, 'P15\n0.088·pT + 3.94\nper particle → Σ pT', C['m0'])
box(4.3, 3.2, 1.75, 1.35, 'g49\n72 % from P15\n(+ soft counts)', C['m0'])
box(6.35, 3.2, 1.75, 1.35, 'S0  → M0\nΣ pT of the 14\nhardest particles', C['m0'])
arrow(4.0, 3.87, 4.3, 3.87); arrow(6.05, 3.87, 6.35, 3.87)
# M2 path
box(6.35, 1.35, 1.75, 1.55, 'S29, S50  → M2\nΣ hinge(ln pT, r)\n≈ Σ pT·r² over\nrank windows', C['m2'])
ax.text(4.3, 2.1, 'nine angular half-plane\ncuts (P) combined into\nfunctions of r in round two', fontsize=8.5, color='#55606e', va='center')
arrow(5.95, 2.1, 6.35, 2.1)
# product
box(8.4, 2.25, 1.95, 2.1, 'h4, h5  → √(M0·M2)\n9 units in h4, 8 in h5\nrespond to angle and\npT edits as a mass does;\nfreezing S0 halves their\npT response', C['prod'])
arrow(8.1, 3.87, 8.4, 3.6); arrow(8.1, 2.1, 8.4, 2.6)
# windows
box(10.6, 2.0, 2.0, 2.6, 'z  → mass windows\nz0 (W): peak 75–77 GeV\nz7, z14 (Z): 94–98 GeV\nz10 (top gate): 135–148\nW → Z calls switch at\n80–90 GeV of edited mass', C['win'])
arrow(10.35, 3.3, 10.6, 3.3)
arrow(12.6, 3.3, 12.85, 5.0)
# read-out strip
ax.text(0.2, 0.75, 'Linear read-out of the jet mass from each stage (held-out R²):   P 0.73   ·   g 0.76   ·   S 0.977   ·   h4 0.991   ·   h5 0.985   ·   z 0.967',
        fontsize=9.5)
ax.text(0.2, 0.3, 'Sources: E5/E5v (objects), E6/E6v (mass tests), E11 (h4:49 fitted as √(M0·M2) < 100.6 GeV, R² 0.991). Numbers on JEDI-linear 3-feature, 64 particles.',
        fontsize=8.5, color='#666')
ax.set_title('Where JEDI-linear forms its mass variable: M0 and M2 are summed separately, multiplied in h4/h5, and read by the class neurons as windows',
             loc='left', fontsize=11.5)
fig.tight_layout(); fig.savefig(OUT / 'X3_mass_path.png'); print('ok')
