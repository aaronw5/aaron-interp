"""E19 figures: results/site/T16_axis_by_class.png (per-class table, WTA axis, JEDI + ParT_full + ParT_kin) and
results/site/F9_top_axis.png (read-out mass before / after the WTA axis for W, Z, t; JEDI thresholds marked).
Reads results/jedi/e19_axis_by_class.{json,npz} and results/part/e19_axis_by_class.{json,npz}.
  ./job.sh e19-plot 'python -m interp.e19_plot'
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); SITE = RES / 'site'
SURF, INK, INK2, GRID = '#fcfcfb', '#1f1f1e', '#5f5e59', '#e4e3df'
C0, C1 = '#8f8d86', '#eb6834'
SFX = sys.argv[1] if len(sys.argv) > 1 else ''; SFXP = sys.argv[2] if len(sys.argv) > 2 else SFX


def ci(v, c, f='{:+.3f}'):
    return f'{f.format(v)} [{f.format(c[0])}, {f.format(c[1])}]' if c[0] is not None else f.format(v)


def lp(h):
    t = h['H0_readout_zero']; hp = h.get('H0_holm_p')
    if t['log10p'] is None: return '–'
    return f'{t["p"]:.2g}' if t['log10p'] > -4 else f'1e{int(np.floor(t["log10p"]))}'


def hm(A, cls):
    return max(A[a]['classes'][k]['H0_holm_p'] for a in ('wta', 'hardest') for k in cls)


def table(J, P):
    rows = []
    for mdl, cls, A in [('JEDI-linear', J['classes'], J['axes'])] + [(f'ParT_{n}', P['classes'], P['models'][n]['axes']) for n in ('full', 'kin')]:
        for k in cls:
            w = A['wta']['classes'][k]; h = A['hardest']['classes'][k]
            rows.append([mdl, k, str(w['n_te']), ci(w['dlogm_hat'], w['dlogm_hat_ci']), f'{w["pred"]:+.3f}',
                         ci(w['ratio'], w['ratio_ci'], '{:.2f}') if w['ratio'] is not None else '–',
                         ci(w['ratio_cal'], w['ratio_cal_ci'], '{:.2f}') if w['ratio_cal'] is not None else '–',
                         ci(100 * w['changed'], [100 * x for x in w['changed_ci']], '{:.1f}'),
                         f'{100 * w["own0"]:.1f} → {100 * w["own1"]:.1f}', lp(w),
                         f'{h["ratio"]:.2f} / {100 * h["changed"]:.1f}' if h['ratio'] is not None else '–'])
    head = ['model', 'true\nclass', 'N\n(test)', 'Δlog m̂ [95 % CI]', '½Δlog(M0·M2)\npredicted', 'Δlog m̂ / pred\n[95 % CI]',
            'Δlog m̂ / Δlog m̂(scale)\n[95 % CI]', 'calls changed %\n[Wilson 95 %]', 'own-class acc %\nstored → WTA', 'p (H0:\nΔlog m̂ = 0)',
            'hardest axis:\nratio / changed %']
    xs = [0.005, 0.085, 0.13, 0.17, 0.31, 0.385, 0.50, 0.625, 0.745, 0.835, 0.905]
    al = ['left', 'left'] + ['center'] * 9
    nr = len(rows); H = 1.6 + 0.3 * nr
    fig = plt.figure(figsize=(17, H), facecolor=SURF); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, H)
    ax.text(0.005, H - 0.15, 'Mass read-out change and calls under the WTA axis, per true class', fontsize=14, color=INK, va='top')
    ax.text(0.005, H - 0.48, f'JEDI-linear: {J["N"]:,} hls4ml test jets (seed {J["seed"]}); ParT: {P["N"]:,} JetClass test jets (rows {P["rows"][0]:,}–{P["rows"][1]:,} per class). '
            'Δlog m̂: ridge read-out of log m (JEDI: 16 z neurons; ParT: 128-d class token) fitted on unedited train-half jets; test-half means.\n'
            'Scale: each jet\'s angles × exp(pred) about the stored axis (a real mass change of the predicted size). Calls on all jets. '
            f'p: one-sample t-test per class; largest Holm-adjusted p over 2 axes × classes: JEDI {hm(J["axes"], J["classes"]):.0e}, '
            f'ParT_full {hm(P["models"]["full"]["axes"], P["classes"]):.0e}, ParT_kin {hm(P["models"]["kin"]["axes"], P["classes"]):.0e}.', fontsize=9.5, color=INK2, va='top')
    top = H - 1.2
    for x, h, a in zip(xs, head, al): ax.text(x + (0.06 if a == 'center' and x == 0.17 else 0), top, h, fontsize=9.5, color=INK2, ha=a, va='center', fontweight='bold')
    ax.plot([0.005, 0.995], [top - 0.25, top - 0.25], color=INK2, lw=0.8)
    for i, r in enumerate(rows):
        yy = top - 0.25 - 0.3 * (i + 0.55)
        if i and r[0] != rows[i - 1][0]: ax.plot([0.005, 0.995], [yy + 0.15, yy + 0.15], color=GRID, lw=1.2)
        hl = r[1] in ('W', 'Z', 't', 'Wqq', 'Zqq', 'Tbqq', 'Hgg')
        for j, (x, v, a) in enumerate(zip(xs, r, al)):
            if j == 0 and i and r[0] == rows[i - 1][0]: continue
            ax.text(x + (0.06 if j == 3 else 0), yy, v, fontsize=9.5, color=INK, ha=a, va='center', fontweight='bold' if (j == 0 or (j == 1 and hl)) else 'normal')
    SITE.mkdir(parents=True, exist_ok=True); fig.savefig(SITE / f'T16_axis_by_class{SFX}.png', dpi=130, facecolor=SURF); plt.close(fig)


def figure(J, P, NJ, NP):
    fig, axs = plt.subplots(3, 3, figsize=(14, 10), facecolor=SURF)
    bins = np.arange(0, 352.5, 5)
    rowsdef = [('JEDI-linear', NJ, NJ['mhat0'], NJ['wta_mhat'], NJ['y'], NJ['te'], [2, 3, 4], ['W', 'Z', 't'])]
    for n in ('full', 'kin'):
        rowsdef.append((f'ParT_{n}', NP, NP[f'{n}_mhat0'], NP[f'{n}_wta_mhat'], NP['y'], NP['te'], [P['classes'].index(c) for c in ('Wqq', 'Zqq', 'Tbqq')], ['Wqq', 'Zqq', 'Tbqq']))
    for r, (mdl, N, m0, m1, y, te, ks, names) in enumerate(rowsdef):
        for c, (k, nm) in enumerate(zip(ks, names)):
            a = axs[r, c]; a.set_facecolor(SURF); s = te & (y == k)
            for sp in ('top', 'right'): a.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'): a.spines[sp].set_color(GRID)
            a.tick_params(colors=INK2, labelsize=9)
            if r == 0:
                a.axvspan(75, 80, color='#2a78d6', alpha=0.18, lw=0, label='W window 75–80 GeV'); a.axvspan(91, 100, color='#1baf7a', alpha=0.2, lw=0, label='Z window 91–100 GeV')
                a.axvline(143, color=INK, lw=1.2, ls='--', label='top threshold 143 GeV')
            a.hist(np.clip(m0[s], 0, 350), bins=bins, color=C0, alpha=0.55, label='stored axis')
            a.hist(np.clip(m1[s], 0, 350), bins=bins, histtype='step', color=C1, lw=2, label='WTA axis')
            md0, md1 = np.median(m0[s]), np.median(m1[s])
            a.set_title(f'{mdl}, true {nm}: median {md0:.0f} → {md1:.0f} GeV', loc='left', fontsize=10.5, color=INK)
            if r == 2: a.set_xlabel('read-out mass exp(log m̂) [GeV]', fontsize=10, color=INK2)
            if c == 0: a.set_ylabel('test-half jets per 5 GeV', fontsize=10, color=INK2)
            if r == 0 and c == 2: a.legend(frameon=False, fontsize=8.5, loc='upper right', labelcolor=INK)
            a.set_xlim(0, 350)
    fig.text(0.01, 0.005, 'Top row: JEDI-linear class-neuron mass thresholds from E6 / E11 (W 75–80, Z 91–100, top > 143 GeV); none are marked for ParT. '
             'Values above 350 GeV are put in the last bin.', fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(SITE / f'F9_top_axis{SFX}.png', dpi=130, facecolor=SURF); plt.close(fig)


if __name__ == '__main__':
    J = json.loads((RES / 'jedi' / f'e19_axis_by_class{SFX}.json').read_text()); P = json.loads((RES / 'part' / f'e19_axis_by_class{SFXP}.json').read_text())
    NJ = dict(np.load(RES / 'jedi' / f'e19_axis_by_class{SFX}.npz')); NP = dict(np.load(RES / 'part' / f'e19_axis_by_class{SFXP}.npz'))
    table(J, P); figure(J, P, NJ, NP); print('wrote', SITE / f'T16_axis_by_class{SFX}.png', SITE / f'F9_top_axis{SFX}.png')
