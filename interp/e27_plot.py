"""E27 figure: per-member calibrated read-out response R on the (beta, kappa) grid for the two decoupling edits, W->Z flow vs the
matched null, and one real W jet before / after the M2 +5 % edit (M1 held).
  python -m interp.e27_plot [--json results/jedi/e27_angularity.json] [--title '...']
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
KAPS = (0, 1, 2); BETS = (0.5, 1.0, 1.5, 2.0, 3.0); CL = ['g', 'q', 'W', 'Z', 't']


def mname(k, b): return f'k{k}_b{b:g}'


def main(a):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = json.loads(Path(a.json).read_text()); T = R['results']['trained']['5']; T2 = R['results']['trained'].get('2')
    npz = np.load(Path(a.json).with_name(Path(a.json).stem + '_perjet.npz'))
    fig = plt.figure(figsize=(16, 16)); gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.95], hspace=0.36, wspace=0.75)
    st = a.stage
    for col, (kind, ttl) in enumerate((('A', 'Edit A: member +5 %, M2 held'), ('B', 'Edit B: M2 +5 %, member held'))):
        ax = fig.add_subplot(gs[0, 2 * col:2 * col + 2]); M = np.full((3, 5), np.nan); txt = {}
        for ki, k in enumerate(KAPS):
            for bi, b in enumerate(BETS):
                if (k, b) == (1, 2.0):
                    e = T['M2h']; M[ki, bi] = e[f'R_{st}']; txt[(ki, bi)] = (e[f'R_{st}'], e[f'R_{st}_ci'], 'M2 +5 %\n(M1 held)'); continue
                e = T['members'][mname(k, b)]; M[ki, bi] = e[kind][f'R_{st}']
                txt[(ki, bi)] = (e[kind][f'R_{st}'], e[kind][f'R_{st}_ci'], f'null {e["n" + kind][f"R_{st}"]:+.2f}\nn {e[kind]["n"]}')
        im = ax.imshow(M, cmap='RdBu_r', vmin=-1.2, vmax=1.2, aspect='auto')
        for (ki, bi), (v, ci, s2) in txt.items():
            ax.text(bi, ki - 0.22, f'{v:+.2f}', ha='center', va='center', fontsize=12, weight='bold')
            ax.text(bi, ki + 0.2, f'[{ci[0]:+.2f}, {ci[1]:+.2f}]\n{s2}', ha='center', va='center', fontsize=8)
        ax.set_xticks(range(5)); ax.set_xticklabels([f'{b:g}' for b in BETS]); ax.set_yticks(range(3))
        ax.set_yticklabels(['κ 0 (count)', 'κ 1 (pT)', 'κ 2 (pT²)']); ax.set_xlabel('β (power of r)')
        ax.set_title(f'{ttl}\nR = Δ log m̂ ({st}) / Δ log m̂ (dilation, M2 +5 %)', fontsize=10)
        if col == 1: cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label('R')
    # W->Z flow, edit vs null, per member (A and B)
    ax = fig.add_subplot(gs[1, 0:2]); names = [mname(k, b) for k in KAPS for b in BETS if (k, b) != (1, 2.0)]
    yy = np.arange(len(names))
    for j, (kind, c, mk) in enumerate((('A', '#c0392b', 'o'), ('B', '#2471a3', 's'))):
        v = np.array([T['members'][n][kind]['wz'] for n in names]); ci = np.array([T['members'][n][kind]['wz_ci'] for n in names])
        vn = np.array([T['members'][n]['n' + kind]['wz'] for n in names])
        off = -0.17 if kind == 'A' else 0.17
        ax.errorbar(100 * v, yy + off, xerr=100 * np.abs(ci.T - v), fmt=mk, color=c, ms=5, label=f'edit {kind}', capsize=2)
        ax.plot(100 * vn, yy + off, '|', color=c, ms=10, mew=2, alpha=.5, label=f'null for {kind}')
    rf = T['ref']['wz']; ax.axvline(100 * rf, color='k', ls='--', lw=1, label=f'dilation M2 +5 % ({100 * rf:.1f} %)')
    ax.set_yticks(yy); ax.set_yticklabels([n.replace('k', 'κ ').replace('_b', ', β ') for n in names], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel('W → Z flow (% of W-called jets called Z after the edit)'); ax.legend(fontsize=7, loc='lower right'); ax.set_title('W → Z flow, edit vs matched null', fontsize=10)
    # example jet
    ax = fig.add_subplot(gs[2, 0]); x0, x1 = npz['x_ex'], npz['x_ex_M2h']; rl = x0[:, 0] > 0; ex = R['example']
    s = 6 + 300 * x0[rl, 0] / x0[rl, 0].max()
    ax.scatter(x0[rl, 1], x0[rl, 2], s=s, facecolors='none', edgecolors='0.45', lw=1, label='before')
    ax.scatter(x1[rl, 1], x1[rl, 2], s=s, color='#c0392b', alpha=.55, lw=0, label='after (M2 +5 %, M1 held)')
    for i in np.flatnonzero(rl):
        ax.annotate('', xy=(x1[i, 1], x1[i, 2]), xytext=(x0[i, 1], x0[i, 2]), arrowprops=dict(arrowstyle='->', color='#c0392b', lw=.7))
    ax.plot(0, 0, 'k+', ms=12); lim = 1.1 * np.abs(np.r_[x0[rl, 1:].ravel(), x1[rl, 1:].ravel()]).max()
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal'); ax.set_xlabel('Δη to stored axis'); ax.set_ylabel('Δφ to stored axis (rad)')
    ax.legend(fontsize=7, loc='upper right')
    lc = ex['lambda_change']
    ax.set_title(f'real W jet, {ex["n_real"]} particles, m = {ex["mass_GeV"]:.0f} GeV\nnetwork call {CL[ex["call_before"]]} → {CL[ex["call_after"]]} (marker area ∝ pT)', fontsize=9)
    ax = fig.add_subplot(gs[2, 1]); r0 = np.hypot(x0[rl, 1], x0[rl, 2]); r1 = np.hypot(x1[rl, 1], x1[rl, 2])
    ax.scatter(r0, 1000 * (r1 - r0), s=s, color='#c0392b', alpha=.6, lw=0); ax.axhline(0, color='0.6', lw=.8)
    ax.set_xlabel('distance r to axis before (rad)'); ax.set_ylabel('Δr (mrad)')
    ax.set_title(f'per-particle radial shift (same jet)\nM2 {100 * lc["k1_b2"]:+.1f} %, Σ pT r {100 * lc["k1_b1"]:+.1f} %, Σ r² {100 * lc["k0_b2"]:+.1f} %, M0 0 %', fontsize=9)
    # member changes induced by edits A / B for three members (what else moves)
    ax = fig.add_subplot(gs[2, 2:4]); EC = R['edit_member_changes']; allm = [mname(k, b) for k in KAPS for b in BETS]
    for nm_, c in (('A_k1_b1', '#c0392b'), ('A_k0_b2', '#e67e22'), ('B_k1_b1', '#2471a3'), ('ref', 'k')):
        if nm_ not in EC: continue
        v = [100 * EC[nm_]['median_change'][m] for m in allm]
        ax.plot(range(15), v, 'o-', color=c, ms=4, lw=1, label={'ref': 'dilation (M2 +5 %)'}.get(nm_, nm_.replace('_k', ': κ ').replace('_b', ', β ')))
    ax.axhline(0, color='0.6', lw=.8); ax.set_xticks(range(15)); ax.set_xticklabels([m.replace('k', '').replace('_b', ', ') for m in allm], rotation=60, fontsize=8)
    ax.set_xlabel('member (κ, β)'); ax.set_ylabel('median change of member (%)'); ax.legend(fontsize=7); ax.set_title('What else each edit changes (median over kept jets)', fontsize=10)
    # 2 % dose for kappa 1 + untrained
    ax = fig.add_subplot(gs[1, 2:4]); rows = []
    if T2:
        for b in (0.5, 1.0, 1.5, 3.0):
            e = T2['members'][mname(1, b)]; rows.append((f'κ 1, β {b:g} (2 %)', e['A'][f'R_{st}'], e['A'][f'R_{st}_ci'], e['B'][f'R_{st}'], e['B'][f'R_{st}_ci'], e['A']['n']))
    for b in (0.5, 1.0, 1.5, 3.0):
        e = T['members'][mname(1, b)]; rows.append((f'κ 1, β {b:g} (5 %)', e['A'][f'R_{st}'], e['A'][f'R_{st}_ci'], e['B'][f'R_{st}'], e['B'][f'R_{st}_ci'], e['A']['n']))
    yy = np.arange(len(rows))
    for j, (lab, c) in enumerate((('A (member +, M2 held)', '#c0392b'), ('B (M2 +, member held)', '#2471a3'))):
        v = np.array([r[1 + 2 * j] for r in rows]); ci = np.array([r[2 + 2 * j] for r in rows])
        ax.errorbar(v, yy + (-.15 if j == 0 else .15), xerr=np.abs(ci.T - v), fmt='o', color=c, capsize=2, label=lab)
    ax.axvline(0, color='0.6', lw=.8); ax.axvline(1, color='0.6', lw=.8, ls='--')
    ax.set_yticks(yy); ax.set_yticklabels([f'{r[0]}  n {r[5]}' for r in rows], fontsize=8); ax.invert_yaxis(); ax.set_xlabel(f'R ({st} read-out)')
    ax.legend(fontsize=7); ax.set_title('pT-weighted neighbours of M2 at two doses', fontsize=10)
    fig.suptitle(a.title or 'E27: JEDI-linear mass read-out under radial edits that separate M2 from other angularities', fontsize=12)
    fig.subplots_adjust(top=0.94)
    fig.text(0.5, 0.005, 'Read-out: ridge fit of log m on h4 (held-out R² 0.97). The h5 read-out (R² 0.97) gives the same R within 0.3 for κ 0 and κ 2 members; '
             'for κ 1 members it differs (e.g. κ 1, β 1.5: edit A h4 −0.55, h5 +0.73). W → Z flow uses the network calls only. n = jets kept for that member.',
             ha='center', fontsize=9, color='0.25')
    out = OUT / 'e27_angularity.png'; fig.savefig(out, dpi=130, bbox_inches='tight'); print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--json', default=str(OUT / 'e27_angularity.json')); ap.add_argument('--title', default='')
    ap.add_argument('--stage', default='h4')
    main(ap.parse_args())
