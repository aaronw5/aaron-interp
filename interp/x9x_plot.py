"""X9x figure: calibrated call-change excess Rc(T) vs translation T for each augmentation range A (mean of seeds 0, 1; whiskers =
seed range), and the break offset T*_25 vs A.  Reads results/jedi/x9x_{pfn,efn,dsets}_{a,b,c}.json and x9x.json.
  python -m interp.x9x_plot   -> results/jedi/x9x_augrange.png
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
ARCH = {'pfn': 'PFN', 'efn': 'EFN', 'dsets': 'Deep Sets'}
AS = [0.0, 0.04, 0.08, 0.16, 0.30]
COL = {0.0: '#222222', 0.04: '#1f77b4', 0.08: '#2ca02c', 0.16: '#ff7f0e', 0.30: '#d62728'}


def nets(arch):
    D = {}
    for p in ('a', 'b', 'c'):
        f = RES / f'x9x_{arch}_{p}.json'
        if f.exists(): D.update(json.loads(f.read_text())['nets'])
    return D


def main():
    C = json.loads((RES / 'x9x.json').read_text())
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05]))
    for k, arch in enumerate(ARCH):
        D = nets(arch); a = ax[k]
        for A in AS:
            rows = [v for v in D.values() if v.get('A') == A and 'per_T' in v]
            if not rows: continue
            Ts = [float(t) for t in rows[0]['per_T']]
            R = np.array([[v['per_T'][str(t) if str(t) in v['per_T'] else t]['Rc'] or 0 for t in rows[0]['per_T']] for v in rows])
            m = R.mean(0); a.plot(Ts, m, 'o-', color=COL[A], lw=1.6, ms=4, label=f'A = {A:.2f} rad')
            a.fill_between(Ts, R.min(0), R.max(0), color=COL[A], alpha=0.18, lw=0)
            if A > 0: a.axvline(A, color=COL[A], ls=':', lw=1)
        a.axhline(0, color='0.6', lw=0.8); a.axhline(0.25, color='0.6', lw=0.8, ls='--')
        a.set_title(ARCH[arch]); a.set_xlabel('rigid translation T of all particles [rad]')
        if k == 0: a.set_ylabel('extra calls changed vs rotation,\nin units of the matched M2 dilation')
        a.set_ylim(-0.2, 1.5); a.set_xlim(0, 0.42)
    ax[0].legend(fontsize=8, title='training shift range', title_fontsize=8, loc='upper left')
    a = ax[3]; mk = {'pfn': 'o', 'efn': 's', 'dsets': '^'}
    for arch in ARCH:
        e = C['per_arch'].get(arch)
        if not e: continue
        A = np.array([n['A'] for n in e['networks']]); T = np.array([n['T_star_25'] for n in e['networks']])
        cen = np.array([n['T_star_25_censored'] for n in e['networks']])
        jit = {'pfn': -0.006, 'efn': 0.0, 'dsets': 0.006}[arch]
        a.scatter(A[~cen] + jit, T[~cen], marker=mk[arch], s=36, label=f'{ARCH[arch]} (r = {e["T_star_25"]["pearson_r"]:.2f})', zorder=3)
        if cen.any(): a.scatter(A[cen] + jit, np.full(cen.sum(), 0.44), marker=mk[arch], s=36, facecolors='none', edgecolors='k', zorder=3)
    a.plot([0, 0.4], [0, 0.4], color='0.5', ls='--', lw=1, label='T* = A')
    a.set_xlabel('training shift range A [rad]'); a.set_ylabel('break offset T* (Rc reaches 0.25) [rad]')
    a.set_xlim(-0.02, 0.33); a.set_ylim(0, 0.46); a.legend(fontsize=8, loc='upper left'); a.set_title('break offset vs training range')
    fig.suptitle('X9x: calls changed by a rigid translation (beyond a matched rotation), per training shift range A; sample(2000, 3801), N 10,000, 2 seeds per A', fontsize=11)
    fig.tight_layout(); out = RES / 'x9x_augrange.png'; fig.savefig(out, dpi=130); print('wrote', out)


if __name__ == '__main__':
    main()
