"""P1 figure: results/part/p1_translation.png from results/part/p1_translation.json (needs matplotlib; any python that has it;
run from the repo root, AI_RESULTS as for the experiment).
Left: fraction of true W jets called Z against the translation δ [rad], ParT_full / ParT_kin (this run) and JEDI-linear
(E6: results/jedi/e6_mass.json, the run of inbox/E6_mass.md; 3,000 test jets per class, hls4ml). Right: the mean shift of the
read-out mass Δlog m̂ (ridge on the unedited last-stage neurons: ParT's class token z, JEDI's z), with each sample's
√(M0·M2) prediction ½Δlog(M0·M2) and the invariant mass (0). Filled markers / lines: shift along φ (an exact rotation);
hollow markers: along η."""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
R = json.load(open(OUT / 'p1_translation.json'))
# JEDI-linear, E6 (results/jedi/e6_mass.json: calls.free[eta/phi δ].W.Z; transfer.free.z; transfer.truth.dlog_sqrtM0M2; calls.none)
JEDI = dict(d=[0.0, 0.03, 0.06], WZ_phi=[0.020, 0.293, 0.763], WZ_eta=[0.020, 0.262, 0.765], dm_phi=[0.0, 0.070, 0.256], dm_eta=[0.0, 0.067, 0.253],
            pred=[0.0, 0.113, 0.306], k11_WZ=0.708, k11_dm=0.085)
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COL = {'full': '#2a78d6', 'kin': '#eb6834', 'jedi': '#1baf7a'}
NAME = {'full': 'ParT_full', 'kin': 'ParT_kin', 'jedi': 'JEDI-linear (E6)'}
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 12.5})
D = [0.0] + R['deltas']
series = {}
for n in ('full', 'kin'):
    E = R['models'][n]['edits']; u = R['models'][n]['unedited']
    series[n] = dict(d=D, WZ_phi=[u['W_to_Z']] + [E[f'phi{d}']['W_to_Z'] for d in R['deltas']], WZ_eta=[u['W_to_Z']] + [E[f'eta{d}']['W_to_Z'] for d in R['deltas']],
                     dm_phi=[0.0] + [E[f'phi{d}']['dlogm_hat_lin'] for d in R['deltas']], dm_eta=[0.0] + [E[f'eta{d}']['dlogm_hat_lin'] for d in R['deltas']],
                     pred=[0.0] + [E[f'phi{d}']['pred_sqrtM0M2'] for d in R['deltas']], k11_WZ=E['k1.1']['W_to_Z'], k11_dm=E['k1.1']['dlogm_hat_lin'])
series['jedi'] = JEDI
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
for ax in (a1, a2):
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.set_xlabel('rigid shift of all particles  δ  [rad]')
    ax.set_xlim(-0.003, 0.095); ax.set_xticks([0, 0.03, 0.06, 0.09])
for n in ('jedi', 'full', 'kin'):
    s = series[n]; c = COL[n]
    a1.plot(s['d'], 100 * np.array(s['WZ_phi']), '-o', color=c, lw=2, ms=7, label=NAME[n] + ', shift along φ')
    a1.plot(s['d'], 100 * np.array(s['WZ_eta']), 'o', color=c, mfc='white', mew=1.6, ms=7, label=NAME[n] + ', along η')
    a2.plot(s['d'], s['dm_phi'], '-o', color=c, lw=2, ms=7)
    a2.plot(s['d'], s['dm_eta'], 'o', color=c, mfc='white', mew=1.6, ms=7)
a2.plot(JEDI['d'], JEDI['pred'], '-', color=COL['jedi'], lw=1.1, alpha=0.55)
a2.plot(D, series['full']['pred'], '-', color=COL['full'], lw=1.1, alpha=0.55)
a2.axhline(0, color=MUTED, lw=1.1)
a2.text(0.091, JEDI['pred'][-1] * 0.06 / 0.09 * 1.0 if False else JEDI['pred'][-1], '√(M0·M2) prediction,\nhls4ml jets', color=COL['jedi'], fontsize=9.5, va='bottom', ha='right', alpha=0.9)
a2.text(0.091, series['full']['pred'][-1], '√(M0·M2) prediction,\nJetClass jets', color=COL['full'], fontsize=9.5, va='bottom', ha='right', alpha=0.9)
a2.text(0.091, 0.004, 'invariant mass: 0', color=MUTED, fontsize=9.5, va='bottom', ha='right')
a1.set_ylabel('true W jets called Z  [%]'); a1.set_ylim(0, 100)
a2.set_ylabel('shift of the read-out mass  Δ log m̂   (0.1 ≈ +10 %)')
a1.legend(frameon=False, fontsize=9.5, loc='upper left', ncol=1)
f, k, j = series['full'], series['kin'], JEDI
i06 = D.index(0.06)
a1.set_title(f'W → Z at δ = 0.06 rad (m unchanged): JEDI {100 * j["WZ_phi"][2]:.0f} %, ParT_full {100 * f["WZ_phi"][i06]:.0f} %, ParT_kin {100 * k["WZ_phi"][i06]:.0f} %\n'
             f'control, angles × 1.1 (m +10 %): JEDI {100 * j["k11_WZ"]:.0f} %, ParT_full {100 * f["k11_WZ"]:.0f} %, ParT_kin {100 * k["k11_WZ"]:.0f} %', loc='left')
a2.set_title(f'read-out mass at δ = 0.06: JEDI {j["dm_phi"][2]:+.2f} (predicted {j["pred"][2]:.2f}), ParT_full {f["dm_phi"][i06]:+.3f} (predicted {f["pred"][i06]:.2f}), '
             f'ParT_kin {k["dm_phi"][i06]:+.3f}\ncontrol, angles × 1.1: JEDI {j["k11_dm"]:+.2f}, ParT_full {f["k11_dm"]:+.2f}, ParT_kin {k["k11_dm"]:+.2f}  (true Δlog m +0.095)', loc='left')
fig.tight_layout(); fig.savefig(OUT / 'p1_translation.png'); print('wrote', OUT / 'p1_translation.png')
