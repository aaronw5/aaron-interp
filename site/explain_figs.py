"""Explanatory figures for the results page: the mass formula and the jet-axis convention on one real W jet.
Reads one W jet from the hls4ml test file (JETDISTILL_RESULTS/_jets/n64/full_test.npz). A few seconds of compute.
python site/explain_figs.py  ->  results/site/X1_mass_shift.png, results/site/X2_axis.png"""
import os, pathlib
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'site'; OUT.mkdir(parents=True, exist_ok=True)
D = np.load(pathlib.Path(os.environ.get('JETDISTILL_RESULTS', '/Users/anrunw/Documents/jet-latent-distillation/results')) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')
y = np.asarray(D['y'][:20000]); mass = np.asarray(D['Q_mass'][:20000])
# a W jet near the W mass with two clear prongs
cand = np.flatnonzero((y == 2) & (np.abs(mass - 80.4) < 2))
x = None


def masses(de, dp):
    M0 = pt.sum(); M1 = np.array([(pt * de).sum(), (pt * dp).sum()]); M2 = (pt * (de ** 2 + dp ** 2)).sum()
    return np.sqrt(max(M0 * M2 - M1 @ M1, 0)), np.sqrt(M0 * M2), M1 / M0


def wta_axis(pt, de, dp):
    """C/A clustering in (Δη, Δφ) with winner-take-all recombination: the merged object takes the harder input's direction"""
    P = [(p, a, b) for p, a, b in zip(pt, de, dp)]
    while len(P) > 1:
        best = None
        for i in range(len(P)):
            for k in range(i + 1, len(P)):
                d = (P[i][1] - P[k][1]) ** 2 + (P[i][2] - P[k][2]) ** 2
                if best is None or d < best[0]: best = (d, i, k)
        _, i, k = best; a, b = P[i], P[k]; h = a if a[0] >= b[0] else b
        P = [P[n] for n in range(len(P)) if n not in (i, k)] + [(a[0] + b[0], h[1], h[2])]
    return np.array([P[0][1], P[0][2]])


for j in cand[:300]:
    xx = np.asarray(D['x'][j], np.float64); r_ = xx[:, 0] > 0
    if r_.sum() < 25 or np.asarray(D['Q_dr01'][j]) < 0.15: continue
    pt, de, dp = xx[r_, 0], xx[r_, 1], xx[r_, 2]
    w = wta_axis(pt, de, dp)
    if 0.065 < np.hypot(*w) < 0.09: J = j; break
m0, s0, c0 = masses(de, dp)
import json
RO = json.load(open(ROOT / 'results' / 'jedi' / 'x_readout_example.json')) if (ROOT / 'results' / 'jedi' / 'x_readout_example.json').exists() else None
fs = 10
# ---------------- X1: the shift
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
for k, (d, title) in enumerate([(0.0, 'A W jet as the networks see it'), (0.06, 'The same jet shifted by δ = 0.06 rad in η')]):
    a = ax[k]; a.scatter(de + d, dp, s=8 + 400 * pt / pt.max(), c='#1f5f8b', alpha=.6, edgecolors='none')
    a.plot(0, 0, '+', c='k', ms=14, mew=2); a.annotate('jet axis (fixed)', (0, 0), (0.05, -0.27), fontsize=fs, arrowprops=dict(arrowstyle='->', lw=.8))
    cx, cy = c0[0] + d, c0[1]; a.plot(cx, cy, 'x', c='#b4562b', ms=10, mew=2); a.annotate('pT-weighted centre', (cx, cy), (cx + 0.04, cy + 0.2), fontsize=fs, color='#b4562b', arrowprops=dict(arrowstyle='->', lw=.8, color='#b4562b'))
    m, s, _ = masses(de + d, dp)
    a.set_title(title, fontsize=fs + 1, loc='left'); a.set_xlim(-0.4, 0.4); a.set_ylim(-0.4, 0.4); a.set_aspect('equal')
    a.set_xlabel('Δη [rad]'); a.set_ylabel('Δφ [rad]')
    a.text(0.02, 0.97, f'invariant mass m = {m:.1f} GeV\n√(M0·M2) about the axis = {s:.1f} GeV', transform=a.transAxes, va='top', fontsize=fs,
           bbox=dict(fc='white', ec='#ccc'))
ds = np.linspace(0, 0.09, 31); mm = np.array([masses(de + d, dp)[0] for d in ds]); ss = np.array([masses(de + d, dp)[1] for d in ds])
a = ax[2]
a.axhspan(74.7, 76.6, color='#2d7a4f', alpha=.18); a.text(0.0915, 75.6, 'W-neuron peak', va='center', fontsize=fs - 1, color='#2d7a4f')
a.axhspan(94.0, 98.3, color='#9a6b00', alpha=.18); a.text(0.0915, 96.1, 'Z-neuron peaks', va='center', fontsize=fs - 1, color='#9a6b00')
a.plot(ds, mm, c='k', lw=2, label='invariant mass m (what physics uses)')
a.plot(ds, ss, c='#b4562b', lw=2, label='√(M0·M2) about the axis (formula, computed from the particles)')
if RO:
    dd = np.array([e['delta'] for e in RO['shift']]); rr = np.array([e['network_readout'] for e in RO['shift']]); cc = [e['call'] for e in RO['shift']]
    a.plot(dd, rr, c='#1f5f8b', lw=2, ls='--', label=f"JEDI-linear's own mass read-out (linear map from h4, R² {RO['readout_r2']:.2f})")
    prev = None
    for d_, r_, c_ in zip(dd, rr, cc):
        if c_ != prev: a.annotate(f'called {c_}', (d_, r_), (d_, r_ + 4), fontsize=8.5, color='#1f5f8b', arrowprops=dict(arrowstyle='-', lw=.6, color='#1f5f8b')); prev = c_
a.set_xlabel('shift δ of every particle [rad]'); a.set_ylabel('GeV'); a.set_xlim(0, 0.09); a.legend(fontsize=fs - 2, loc='upper left')
a.set_title('Mass as δ grows', fontsize=fs + 1, loc='left')
fig.suptitle('Why a shifted jet is called Z: the true mass does not change, the networks\' mass variable does (one W jet from the hls4ml test set)', fontsize=fs + 2, x=0.01, ha='left')
fig.text(0.01, 0.005, 'm² ≈ M0·M2 − |M1|² with M0 = Σ pT, M1 = Σ pT·r⃗, M2 = Σ pT·r². On real jets the axis is the pT-weighted centre, so M1 ≈ 0 and both curves start at the same value.\n'
         'Measured on 10,000 jets: JEDI-linear calls 77 % of W jets Z at δ = 0.06; ParT_full 31 % (12 % unshifted). The read-out is a linear map fitted on ordinary jets; on this jet it reads 90 GeV unshifted (true 81).', fontsize=fs - 1, color='#444')
fig.tight_layout(rect=(0, 0.07, 1, 0.94)); fig.savefig(OUT / 'X1_mass_shift.png'); plt.close(fig)

# ---------------- X2: the axis
w = wta_axis(pt, de, dp); h = np.array([de[np.argmax(pt)], dp[np.argmax(pt)]])
fig, ax = plt.subplots(1, 2, figsize=(12, 6.2), dpi=150)
a = ax[0]; a.scatter(de, dp, s=8 + 400 * pt / pt.max(), c='#1f5f8b', alpha=.6, edgecolors='none')
a.plot(0, 0, '+', c='k', ms=14, mew=2, label='stored axis (E-scheme: summed momentum; = pT-weighted centre)')
a.plot(*w, 's', mfc='none', mec='#b4562b', ms=11, mew=2, label=f'winner-take-all axis ({np.hypot(*w):.3f} rad away)')
a.plot(*h, '^', mfc='none', mec='#2d7a4f', ms=11, mew=2, label=f'hardest particle ({np.hypot(*h):.3f} rad away)')
a.set_xlim(-0.4, 0.4); a.set_ylim(-0.4, 0.4); a.set_aspect('equal'); a.set_xlabel('Δη [rad]'); a.set_ylabel('Δφ [rad]')
a.legend(fontsize=fs - 1, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False); a.set_title('Three ways to define the axis of the same W jet', fontsize=fs + 1, loc='left')
a = ax[1]; rows = []
for name, c in [('stored (E-scheme)', np.zeros(2)), ('winner-take-all', w), ('hardest particle', h)]:
    m, s, _ = masses(de - c[0], dp - c[1]); rows.append((name, m, s))
names = [r[0] for r in rows]; xs = np.arange(3); W_ = 0.26
a.bar(xs - W_, [r[1] for r in rows], W_, color='k', label='invariant mass m (formula)')
a.bar(xs, [r[2] for r in rows], W_, color='#b4562b', label='√(M0·M2) about that axis (formula)')
for i, r in enumerate(rows): a.text(i, r[2] + 1, f'{r[2]:.0f}', ha='center', fontsize=fs - 1); a.text(i - W_, r[1] + 1, f'{r[1]:.0f}', ha='center', fontsize=fs - 1)
if RO:
    ro = [e['network_readout'] for e in RO['axes']]; calls = [e['call'] for e in RO['axes']]
    a.bar(xs + W_, ro, W_, color='#1f5f8b', label='JEDI-linear\'s own mass read-out (h4)')
    for i, (v, c_) in enumerate(zip(ro, calls)): a.text(i + W_, v + 1, f'{v:.0f}\ncalled {c_}', ha='center', fontsize=fs - 1, color='#1f5f8b')
a.axhspan(94.0, 98.3, color='#9a6b00', alpha=.18); a.axhspan(74.7, 76.6, color='#2d7a4f', alpha=.18)
a.set_xticks(xs); a.set_xticklabels(names); a.set_ylabel('GeV'); a.legend(fontsize=fs - 1, loc='upper left'); a.set_ylim(0, max(r[2] for r in rows) * 1.25)
a.set_title('Mass variable about each axis', fontsize=fs + 1, loc='left')
fig.suptitle('The jet axis is a convention; the invariant mass does not depend on it, the networks\' mass variable does', fontsize=fs + 2, x=0.01, ha='left')
fig.text(0.01, 0.005, 'Measured on 10,000 JetClass jets with inputs about the WTA axis: ParT_full changes 11.7 % of its calls (accuracy 85.9 → 80.9 %), ParT_kin 34.1 % (74.5 → 56.0 %).\n'
         'The WTA axis follows the branch with the larger summed pT, which need not contain the single hardest particle. Shaded: W- and Z-neuron peaks of JEDI-linear.', fontsize=fs - 1, color='#444')
fig.tight_layout(rect=(0, 0.07, 1, 0.94)); fig.savefig(OUT / 'X2_axis.png'); plt.close(fig)
print('jet', J, 'm', round(m0, 1), 'WTA', w.round(3), 'hardest', h.round(3), [tuple(round(v, 1) if isinstance(v, float) else v for v in r) for r in rows])
