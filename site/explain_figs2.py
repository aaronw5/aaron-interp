"""Explanatory figures: X0 JEDI-linear architecture, X4 the first-layer cuts, X5 the second-round objects as functions of (pT, r).
Numbers from results/jedi/e1_e3.json (E1/E2) and inbox/E4_E5_objects.md (E5). python site/explain_figs2.py"""
import json, pathlib
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / 'results' / 'site'; OUT.mkdir(parents=True, exist_ok=True)

# ---------------- X0 architecture
fig, ax = plt.subplots(figsize=(15, 7.4), dpi=150); ax.axis('off'); ax.set_xlim(0, 15); ax.set_ylim(0, 7.4)
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.1', fc=fc, ec='#7a8796', lw=1))
    ax.text(x + 0.1, y + h - 0.12, title, va='top', fontsize=10.5, weight='bold')
    ax.text(x + 0.1, y + h - 0.5, body, va='top', fontsize=9, linespacing=1.35)
def arr(x0, y0, x1, y1, t=None):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', lw=1.2, color='#55606e'))
    if t: ax.text((x0 + x1) / 2 + 0.05, (y0 + y1) / 2, t, fontsize=8.5, color='#55606e')
PP, JJ, DD = '#e2edf6', '#f6e7da', '#e8f1e7'
ax.text(0.1, 7.15, 'per particle (same weights for all 64 particles)', fontsize=10, color='#1f5f8b', weight='bold')
ax.text(0.1, 3.05, 'per jet', fontsize=10, color='#b4562b', weight='bold')
box(0.1, 5.2, 2.3, 1.75, 'input x_i', '(pT, Δη, Δφ) of particle i\ni = 0 … 63, sorted by pT\nempty slots: pT = 0', PP)
box(2.9, 5.2, 2.5, 1.75, 'a  first-layer cuts', 'a_i = ReLU(W1·x_i + b1)\n64 units per particle\n13 change ≥ 1 % of calls', PP)
box(6.0, 5.2, 2.6, 1.75, 's  per-particle layer', 's_i = ReLU(W2·a_i + b2)\n64 units per particle\n9 change ≥ 1 % of calls', PP)
box(9.2, 5.2, 2.9, 1.75, 'u  second-round cuts', 'u_i = ReLU(W3·(s_i + g) + b3)\n64 units per particle\ncuts can shift with the jet (g)', PP)
box(2.9, 2.3, 2.5, 1.75, 'P  first-round objects', 'P = (1/64) Σ_i a_i\n64 numbers per jet\n(count + sums of the particles\npassing each cut)', JJ)
box(6.0, 2.3, 2.6, 1.75, 'g  jet context', 'g = ReLU(Wg·P + bg)\n64 numbers per jet\nadded to every particle\'s\nsecond-round input', JJ)
box(9.2, 2.3, 2.9, 1.75, 'S  second-round objects', 'S = (1/16) Σ_i u_i\n64 numbers per jet\n19 change ≥ 1 % of calls', JJ)
box(9.2, 0.1, 2.9, 1.7, 'h4, h5  dense layers', 'h4 = ReLU(W4·S + b4)  (64)\nh5 = ReLU(W5·h4 + b5)  (32)\nno particles any more', DD)
box(12.5, 0.1, 2.4, 1.7, 'z  class neurons', 'z = W6·h5 + b6  (16)\nscores = W7·ReLU(z) + b7\nfor g, q, W, Z, t', DD)
arr(2.4, 6.07, 2.9, 6.07); arr(5.4, 6.07, 6.0, 6.07); arr(8.6, 6.07, 9.2, 6.07)
arr(4.15, 5.2, 4.15, 4.05, 'sum over particles'); arr(5.4, 3.17, 6.0, 3.17); arr(7.3, 4.05, 9.6, 5.2, '+ g to every particle')
arr(10.65, 5.2, 10.65, 4.05, 'sum over particles'); arr(10.65, 2.3, 10.65, 1.8); arr(12.1, 0.95, 12.5, 0.95)
ax.text(12.5, 6.9, 'Names: the letters are the layer names in our code.\nP = pooled (summed) first round, g = global context,\nS = summed second round, h4/h5 = hidden layers 4 and 5,\nz = the 16 neurons before their ReLU.\nFixed point: entries with 0 bits are always zero, so\neach object only sums over fixed pT ranks.', fontsize=8.8, va='top', color='#333')
ax.set_title('JEDI-linear (3 features, 64 particles; 80.9 % on the hls4ml test set): what each letter is', loc='left', fontsize=12)
fig.tight_layout(); fig.savefig(OUT / 'X0_architecture.png'); plt.close(fig)

# ---------------- X4 first-layer cuts
d = json.load(open(ROOT / 'results' / 'jedi' / 'e1_e3.json')); cost = d['E1']['a']['changed']
live = [e for e in d['E2'] if cost[e['c']] > 0.001]
ang = [e for e in live if e['main'] != 'pT']; ptc = [e for e in live if e['main'] == 'pT']
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150)
a = axs[0]; t = np.linspace(-0.4, 0.4, 2)
for e in ang:
    ae, ap, b = e['a_eta'], e['a_phi'], e['beta']; n = np.hypot(ae, ap); ue, up = ae / n, ap / n
    off = -b / n  # signed distance of the line from the axis
    px, py = ue * off, up * off; tx, ty = -up, ue
    a.plot([px - tx * 0.45, px + tx * 0.45], [py - ty * 0.45, py + ty * 0.45], c='#1f5f8b', lw=1 + 6 * cost[e['c']])
    a.annotate('', (px + ue * 0.12, py + up * 0.12), (px, py), arrowprops=dict(arrowstyle='->', color='#b4562b', lw=1.2))
    a.text(px + ue * 0.15, py + up * 0.15, f"a{e['c']}", fontsize=8, ha='center', va='center')
a.plot(0, 0, '+', c='k', ms=12, mew=2)
a.set_xlim(-0.35, 0.35); a.set_ylim(-0.35, 0.35); a.set_aspect('equal'); a.set_xlabel('Δη [rad]'); a.set_ylabel('Δφ [rad]')
a.set_title('Nine angular cuts: lines through the jet axis (+);\narrow = side where the unit is on; line width = decision cost', fontsize=10, loc='left')
a = axs[1]; pt = np.linspace(0, 60, 400)
for e in ptc:
    v = np.maximum(e['a_pt'] * pt + e['beta'], 0)
    a.plot(pt, v, lw=2, label=f"a{e['c']}: {'on for pT < %.1f GeV' % (-e['beta'] / e['a_pt']) if e['a_pt'] < 0 else 'always on, linear in pT → sums to ΣpT'}  (cost {cost[e['c']] * 100:.0f} %)")
a.set_xlabel('particle pT [GeV]'); a.set_ylabel('unit value'); a.legend(fontsize=8.5); a.set_xlim(0, 60)
a.set_title('Four pT cuts: "how soft is this particle" ramps and one ΣpT unit', fontsize=10, loc='left')
fig.suptitle('JEDI-linear\'s first layer: 13 cuts on single particles (each unit is zero on one side of a line in (pT, Δη, Δφ))', x=0.01, ha='left', fontsize=12)
fig.text(0.01, 0.01, 'No unit mixes pT and angle. The angular cuts have no pT weight, so they count particles on one side of the axis regardless of their momentum. Source: E2 (results/jedi/e1_e3.json).', fontsize=8.5, color='#444')
fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig(OUT / 'X4_first_layer_cuts.png'); plt.close(fig)

# ---------------- X5 second-round objects
pt = np.logspace(np.log10(0.5), np.log10(300), 200); r = np.linspace(0, 0.45, 200); PT, R = np.meshgrid(pt, r)
objs = [('S29  (cost 40.9 %, all ranks)\nReLU(4.44 ln pT + 93.8 r − 21.2)', np.maximum(4.44 * np.log(PT) + 93.8 * R - 21.2, 0), '≈ ΣpT·r² + Σr²: wide-angle particles count more'),
        ('S1  (20.2 %, ranks 1–63)\nReLU(89.3 r − 22.9 + 0.55 ln pT)', np.maximum(89.3 * R - 22.9 + 0.55 * np.log(PT), 0), 'Σ (r − 0.2)₊: particles beyond r ≈ 0.25'),
        ('S0  (12.0 %, ranks 0–13)\n0.32 pT + 1.40', 0.32 * PT + 1.40, 'ΣpT of the 14 hardest (via g: total pT)'),
        ('S19  (11.1 %, ranks 13–63)\nReLU(2.86 − 1.39 pT)', np.maximum(2.86 - 1.39 * PT, 0), 'soft multiplicity: particles below 2 GeV'),
        ('S30  (8.0 %, all ranks)\nReLU(3.47 − 21.0 r)', np.maximum(3.47 - 21.0 * R, 0), 'core count: particles within r < 0.165')]
fig, axs = plt.subplots(1, 5, figsize=(18, 4.6), dpi=150, sharey=True)
for a, (t, Z, name) in zip(axs, objs):
    im = a.pcolormesh(PT, R, Z, shading='auto', cmap='Blues'); a.set_xscale('log'); a.set_title(t, fontsize=9, loc='left'); a.set_xlabel('particle pT [GeV]')
    a.text(0.03, 0.03, name, transform=a.transAxes, fontsize=8.5, bbox=dict(fc='white', ec='#ccc'))
axs[0].set_ylabel('r = ΔR to the jet axis [rad]')
fig.suptitle('Second-round objects: each is a cut on (pT, r) applied to every particle in a pT-rank window and summed (colour = one particle\'s contribution)', x=0.01, ha='left', fontsize=12)
fig.text(0.01, 0.01, 'Fitted per-particle responses with the jet context at its mean (E5). Ten of 17 such objects change ≤ 0.1 SD when the whole jet is rotated (E5v), so they depend on r, not on direction.', fontsize=8.5, color='#444')
fig.tight_layout(rect=(0, 0.05, 1, 0.9)); fig.savefig(OUT / 'X5_second_round_objects.png'); plt.close(fig)
print('ok')
