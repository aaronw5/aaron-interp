"""E25w figure from results/jedi/e25w_{form,dose,base}.json -> results/jedi/e25w.png (run locally; matplotlib only)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / 'results' / 'jedi'
form = sys.argv[1] if len(sys.argv) > 1 else 'e25w_form.json'
F = json.loads((R / form).read_text()); D = json.loads((R / 'e25w_dose.json').read_text()); Bs = json.loads((R / 'e25w_base.json').read_text())
col = dict(M0='#1f77b4', M2='#d62728', D='#2ca02c')
lab = dict(M0='M0 = Σ pT', M2='M2 = Σ pT ΔR²', D='½ log(M0·M2)')
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

# A: DAS effective IIA per stage (16f n64, k = 2)
a = ax[0]; M = F['models']['16f']; st = ['S', 'h4', 'h5', 'z', 'logits']; x = np.arange(len(st))
qual = [n for n in Bs['test']['qualifying']]
for V in ('M0', 'M2', 'D'):
    e = [M['das'][s][V]['2']['eff'] for s in st]; ci = np.array([M['das'][s][V]['2']['eff_ci'] for s in st])
    a.errorbar(x, e, yerr=[np.array(e) - ci[:, 0], ci[:, 1] - np.array(e)], color=col[V], marker='o', capsize=3, label=f'DAS {lab[V]}')
    a.axhline(M['das']['S'][V]['2']['full_swap'], color=col[V], ls='--', lw=1)
    bm = [np.median([Bs['nets'][n]['das'][s][V]['2']['eff'] for n in qual]) for s in st[:4]]
    a.plot(x[:4], bm, color=col[V], ls=':', marker='x', lw=1)
rq = [max(M['das'][s][V]['2']['random']['q'][2] for V in ('M0', 'M2', 'D')) for s in st]
a.fill_between(x, 0, rq, color='0.85', label='random 2-dim subspaces (97.5 %)')
a.plot([], [], 'k--', lw=1, label='full-stage swap (source call)'); a.plot([], [], 'k:', marker='x', lw=1, label='same DAS, random-body baselines (median)')
a.set_xticks(x, st); a.set_ylim(0, 1); a.set_xlabel('stage (S = second pooled sum)'); a.set_ylabel('effective interchange accuracy')
a.set_title('A. Mass variables sit in a 2-dim subspace already at S', fontsize=10); a.legend(fontsize=7, loc='upper right')

# B: where true-W calls go under translation (16f n64)
b = ax[1]; dm = D['models']['16f_n64']['doses']; ds = sorted(dm, key=float); xd = [float(d) for d in ds]
for cls, c, nm in (('Z', '#d62728', 'Z'), ('t', '#9467bd', 'top'), ('qg', '#8c564b', 'q or g'), ('W', '#7f7f7f', 'W (kept)')):
    f = lambda arm: [dm[d][arm]['W_new_call']['q']['f'] + dm[d][arm]['W_new_call']['g']['f'] if cls == 'qg' else dm[d][arm]['W_new_call'][cls]['f'] for d in ds]  # noqa: E731
    b.plot(xd, f('T'), color=c, marker='o', label=f'{nm}: translation')
    b.plot(xd, f('dilR'), color=c, ls='--', lw=1)
b.plot(xd, [dm[d]['R']['W_new_call']['Z']['f'] for d in ds], color='#d62728', ls=':', marker='s', ms=3, label='Z: rotation (same displacement)')
b.plot([], [], 'k--', lw=1, label='matched ΔR-only dilation')
ax2 = b.twiny(); ax2.set_xlim(b.get_xlim() if False else (0.008, 0.127)); b.set_xlim(0.008, 0.127)
mf = [np.exp(D['dose_meta'][d]['half_dlogM2_mean']) * 80.4 for d in ds]
ax2.set_xticks(xd, [f'{m:.0f}' for m in mf], fontsize=7); ax2.set_xlabel('W mass × mean √(M2 change) (GeV)', fontsize=8)
b.set_xlabel('translation of all particles off the stored axis, d (rad)'); b.set_ylabel('fraction of true W jets')
b.set_title('B. Translation moves W calls through Z to top and q/g', fontsize=10, pad=28); b.legend(fontsize=7, loc='center right')

# C: W -> Z flow, translation vs matched dilation, trained vs baselines (0.06 rad, test half)
c = ax[2]
for n, m in Bs['nets'].items():
    fx, fy = m['arms']['dilR']['flow'], m['arms']['T']['flow']
    if n == 'trained': c.plot(fx, fy, 'r*', ms=14, label='trained network')
    elif n == 'trained_refit': c.plot(fx, fy, 'o', color='orange', ms=8, label='trained body, refit read-out')
    else:
        kind = n.rstrip('0123456789'); mk = dict(init='s', gauss='^', perm='D')[kind]
        c.plot(fx, fy, mk, color='0.3' if m['qualifies'] else '0.75', mfc='0.3' if m['qualifies'] else 'none', ms=6)
c.plot([], [], 's', color='0.3', label='random body, init scale'); c.plot([], [], '^', color='0.3', label='random body, trained-moment scale')
c.plot([], [], 'D', color='0.3', label='weights permuted (masks kept)'); c.plot([], [], 'o', color='0.75', mfc='none', label='open: accuracy ≤ 40 % (not used)')
c.plot([-0.1, 0.6], [-0.1, 0.6], 'k:', lw=1); c.set_xlim(-0.06, 0.6); c.set_ylim(-0.06, 0.6)
c.set_xlabel('W→Z flow under ΔR-only dilation'); c.set_ylabel('W→Z flow under 0.06 rad translation')
c.set_title('C. Same ratio, smaller size in non-trained bodies', fontsize=10); c.legend(fontsize=7, loc='upper left')
fig.suptitle('16-feature JEDI-linear n64 (N 6,694 jets): where the mass variable is held, and what the translation does', fontsize=11)
fig.tight_layout(); fig.savefig(R / 'e25w.png', dpi=130); print('saved', R / 'e25w.png')
