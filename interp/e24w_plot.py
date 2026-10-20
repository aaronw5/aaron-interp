"""E24w figure: D - P (prong destroyed minus prong kept, % of calls changed, paired, 95 % jet bootstrap) per model.
  python -m interp.e24w_plot  (reads results/jedi/e24w_prong_jedi.json, writes results/jedi/e24w_prong.png)"""
import json, os
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
D = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
R = json.loads((D / 'e24w_prong_jedi.json').read_text())
rows = [('JEDI-linear (official)', R['models']['jedi'], '#1f4e8c'),
        ('  same, jet context g held', R['models']['jedi|g_frozen'], '#1f4e8c'),
        ('  same, S58 and S31 held', R['models']['jedi|S58_S31_frozen'], '#1f4e8c')]
rows += [(f'retrained JEDI {k[4:]}' + (' (57 % acc.)' if k.endswith('r150_s1') else ''), v, '#7f8fa6') for k, v in R['models'].items() if k.startswith('x12:')]
rows += [('Deep Sets', R['models']['dsets_s0'], '#7f8fa6'), ('PFN', R['models']['pfn_s0'], '#7f8fa6'),
         ('tree: M0, M2, Σ pT r, N + mass', R['trees']['Thm'], '#9a9a9a'),
         ('tree: + mass + τ21', R['trees']['Thmt'], '#9a9a9a'),
         ('tree: M0, M2, Σ pT r, N + τ21', R['trees']['Tht'], '#c0612b')]
fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
for i, (lab, M, c) in enumerate(rows):
    d = M['D_minus_P']['calls']; y = len(rows) - 1 - i
    ax.errorbar(d['mean'] * 100, y, xerr=[[100 * (d['mean'] - d['ci'][0])], [100 * (d['ci'][1] - d['mean'])]],
                fmt='o', color=c, ms=6, lw=2, capsize=0)
ax.axvline(0, color='#444', lw=1)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
ax.set_xlabel('D − P: extra % of calls changed when the softer prong is destroyed\nrather than rotated (same jets, same mass, 95 % CI)', fontsize=9)
ax.set_title(f"JEDI-linear: destroying the softer prong ≈ rotating it (D − P ≈ 0)\n"
             f"(N = {R['acceptance']['analysis']} two-prong test jets; τ21 tree: +{R['trees']['Tht']['D_minus_P']['calls']['mean']*100:.1f} points)", fontsize=10)
ax.grid(axis='x', color='#ddd', lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(D / 'e24w_prong.png'); print('wrote', D / 'e24w_prong.png')
