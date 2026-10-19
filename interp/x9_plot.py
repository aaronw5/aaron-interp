"""X9 figure (local, seconds): JEDI-linear vs PFN / EFN on the same 10,000 test jets (sample(2000, 313)).
  /Users/anrunw/Documents/jedi-distill/.venv/bin/python interp/x9_plot.py   -> results/jedi/x9_compare.png
Panels: (a) W → Z fraction of true-W jets under the frozen edits; (b) rank-window swap (ranks 13|14 vs 11|12): mean |ΔL|/sd
across / inside, with the pT-share (z) gap ratio of the swapped pairs; (c) mean effective IIA with the permuted-alignment mean."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'results' / 'jedi'
MODELS = [('pfn_64-64-32_s0', 'PFN s0'), ('pfn_64-64-32_s1', 'PFN s1'), ('efn_100-100-128_s0', 'EFN s0'), ('efn_100-100-128_s1', 'EFN s1')]
D = {lab: json.loads((OUT / f'x9_{m}.json').read_text()) for m, lab in MODELS if (OUT / f'x9_{m}.json').exists()}
D = {k: v for k, v in D.items() if 'tests_vs_jedi' in v}          # finished runs only
if not D: sys.exit('no x9 results')
ref = next(iter(D.values()))
EDITS = [('rotation_matched', 'rotation\n(control)'), ('translation_random_dir', 'transl.\nrandom dir'), ('translation_eta', 'transl.\n+Δη'),
         ('translation_phi', 'transl.\n+Δφ'), ('wta_axis', 'WTA\naxis')]
cols = {'JEDI-linear': '#222222', 'PFN s0': '#1f77b4', 'PFN s1': '#6baed6', 'EFN s0': '#d62728', 'EFN s1': '#fc9272'}
rows = [('JEDI-linear', None)] + [(lab, D[lab]) for lab in D]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
w = 0.8 / len(rows)
for i, (lab, d) in enumerate(rows):
    src = ref if d is None else d; who = 'jedi' if d is None else 'model'
    v = [src['step3']['edits'][e][who]['W_to_Z'] * 100 for e, _ in EDITS]
    lo = [src['step3']['edits'][e][who]['W_to_Z_ci'][0] * 100 for e, _ in EDITS]; hi = [src['step3']['edits'][e][who]['W_to_Z_ci'][1] * 100 for e, _ in EDITS]
    x = np.arange(len(EDITS)) + (i - (len(rows) - 1) / 2) * w
    ax[0].bar(x, v, w, color=cols[lab], label=f'{lab} ({(src["step0"]["acc_jedi_T"] if d is None else src["step0"]["acc_model_T"]) * 100:.1f} %)',
              yerr=[np.subtract(v, lo), np.subtract(hi, v)], capsize=1.5, error_kw=dict(lw=0.6))
ax[0].set_xticks(np.arange(len(EDITS))); ax[0].set_xticklabels([l for _, l in EDITS]); ax[0].set_ylabel('true-W jets called Z (%)')
ax[0].set_title('(a) 0.06 rad edits (m unchanged): W → Z', fontsize=10); ax[0].set_ylim(0, 105); ax[0].legend(fontsize=8, title='model (accuracy on these jets)', title_fontsize=8, ncol=5, loc='upper center', columnspacing=0.8, handlelength=1.2)
# (b) rank swap
labs = [lab for lab, _ in rows]; vals, cis = [], []
for lab, d in rows:
    rec = (ref if d is None else d)['step3']['rank_swap'][0]; r = rec['jedi' if d is None else 'model']; vals.append(r['ratio']); cis.append(r['ratio_ci'])
rec0 = ref['step3']['rank_swap'][0]
ax[1].bar(range(len(labs)), vals, color=[cols[l] for l in labs], yerr=[np.subtract(vals, [c[0] for c in cis]), np.subtract([c[1] for c in cis], vals)], capsize=2)
ax[1].axhline(1, color='grey', lw=0.8, ls=':'); ax[1].axhline(rec0['z_gap_ratio'], color='k', lw=0.8, ls='--')
ax[1].text(0.6, 1.6, f'dashed: pT-share gap ratio\nof the swapped pairs ({rec0["z_gap_ratio"]:.2f});\ndotted: 1', ha='left', va='bottom', fontsize=7)
ax[1].set_xticks(range(len(labs))); ax[1].set_xticklabels(labs, rotation=30, fontsize=8)
ax[1].set_ylabel('mean |ΔL|/sd: swap 13|14 ÷ swap 11|12')
ax[1].set_title(f'(b) angle swap across JEDI\'s rank-14 edge\n(N = {rec0["N"]} jets)', fontsize=10)
# (c) IIA
iia = [(ref if d is None else d)['step4']['jedi' if d is None else 'model'] for _, d in rows]
m = [r['mean_eff_iia'] for r in iia]; ci = [r['mean_eff_iia_ci'] for r in iia]; pm = [r['permuted_mean'] for r in iia]
ax[2].bar(range(len(labs)), m, color=[cols[l] for l in labs], yerr=[np.subtract(m, [c[0] for c in ci]), np.subtract([c[1] for c in ci], m)], capsize=2,
          label='named formulas')
ax[2].scatter(range(len(labs)), pm, marker='_', s=300, color='k', zorder=3, label='permuted alignment (200)')
for i, r in enumerate(iia): ax[2].text(i, 0.02, f'{len(r["units"])} u', ha='center', fontsize=7, color='w')
ax[2].set_xticks(range(len(labs))); ax[2].set_xticklabels(labs, rotation=30, fontsize=8); ax[2].set_ylim(0, 1)
ax[2].set_ylabel('mean effective IIA (latent / S units)'); ax[2].set_title('(c) interchange interventions\n(5,000 jet pairs)', fontsize=10)
ax[2].legend(fontsize=7, loc='upper right')
fig.suptitle('Same 10,000 test jets: PFN and EFN move W jets to Z under a 0.06 rad translation like JEDI-linear; only JEDI-linear responds to the rank-14 edge', fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / 'x9_compare.png', dpi=130); print('wrote', OUT / 'x9_compare.png')
