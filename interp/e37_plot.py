"""E37 figure: (a) one test W jet before / after D(+0.3) (constituents in the jet frame, colour = kT prong, area = pT);
(b) share of D's W-Z margin change reproduced by the learned subspace (DAS) vs its dimension k at S / h4 / h5, with the
random-subspace 95th percentile and the share of a real mass change (dilation) moved by the same subspace;
(c) D minus cn margin change (dc +0.3) per network for W and Z jets, in units of each network's margin s.d.
Usage: python -m interp.e37_plot [--smoke]   (reads results/jedi/e37_das*.json, e37_recur*.json; writes results/jedi/e37_das.png)"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / 'results' / 'jedi'
sm = '_smoke' if '--smoke' in sys.argv else ''
D = json.loads((R / f'e37_das{sm}.json').read_text())['trained']; C = json.loads((R / f'e37_recur{sm}.json').read_text())
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1.15]))
ex = D['example']; lab = np.array(ex['prong'])
for nm, mk, a_ in (('unedited', 'o', 0.35), ('D+0.3', 'o', 1.0)):
    e = ex[nm]; pt = np.array(e['pt'])
    for k, col in ((0, 'tab:blue'), (1, 'tab:orange')):
        s = lab == k
        ax[0].scatter(np.array(e['deta'])[s], np.array(e['dphi'])[s], s=8 + 250 * pt[s] / pt.max(), c=col, alpha=a_, edgecolors='k' if nm == 'D+0.3' else 'none', linewidths=0.4,
                      label=f'{"edited" if nm != "unedited" else "unedited"}, prong {k + 1}')
o0, o1 = ex['unedited']['obs'], ex['D+0.3']['obs']
ax[0].set_title(f'(a) W jet, D(+0.3): z_min {o0["z_min"]:.2f} -> {o1["z_min"]:.2f}, m {o0["m"]:.1f} -> {o1["m"]:.1f} GeV\n'
                f'W-Z margin {ex["margin"]["unedited"]:+.2f} -> {ex["margin"]["D"]:+.2f} (log-odds)', fontsize=9)
ax[0].set_xlabel(r'$\Delta\eta$ to jet axis'); ax[0].set_ylabel(r'$\Delta\phi$ to jet axis (rad)'); ax[0].legend(fontsize=7, loc='lower left', markerscale=0.4); ax[0].set_aspect('equal', 'datalim')
cols = dict(S='tab:grey', h4='tab:red', h5='tab:purple')
for st in ('S', 'h4', 'h5'):
    ks = sorted(int(k) for k in D['das'][st]['k']); f = [D['das'][st]['k'][str(k)]['test']['self']['frac'] for k in ks]
    lo = [D['das'][st]['k'][str(k)]['test']['self']['frac_ci'][0] for k in ks]; hi = [D['das'][st]['k'][str(k)]['test']['self']['frac_ci'][1] for k in ks]
    rq = [D['das'][st]['k'][str(k)]['random']['self']['frac_q'][2] for k in ks]; dl = [D['das'][st]['k'][str(k)]['test']['dil']['frac'] for k in ks]
    ax[1].errorbar(ks, f, yerr=[np.array(f) - lo, np.array(hi) - f], color=cols[st], marker='o', capsize=2, label=f'{st}: learned subspace, D response')
    ax[1].plot(ks, rq, color=cols[st], ls=':', lw=1, label=f'{st}: random subspaces, 95th pct')
    ax[1].plot(ks, dl, color=cols[st], ls='--', marker='x', lw=1, label=f'{st}: same subspace, dilation response')
ax[1].axhline(0, color='k', lw=0.5); ax[1].set_ylim(-0.1, 1.65); ax[1].set_xlabel('subspace dimension k'); ax[1].set_ylabel('share of the margin change reproduced by the patch')
ax[1].set_title('(b) interchange of k coordinates, held-out jets', fontsize=9); ax[1].legend(fontsize=6, ncol=1)
nets = list(C['nets']); y = np.arange(len(nets))
for c, off, col in (('W', -0.15, 'tab:blue'), ('Z', 0.15, 'tab:orange')):
    v = [C['nets'][n]['per_dc']['+0.30'][c]['diff_sd_units'] or 0 for n in nets]
    ci = [C['nets'][n]['per_dc']['+0.30'][c]['diff_sd_ci'] or [0, 0] for n in nets]
    ax[2].errorbar(v, y + off, xerr=[np.array(v) - [a[0] for a in ci], [a[1] for a in ci] - np.array(v)], fmt='o', color=col, ms=4, capsize=2, label=f'{c} jets')
ax[2].axvline(0, color='k', lw=0.5); ax[2].set_yticks(y); ax[2].set_yticklabels(nets, fontsize=7); ax[2].invert_yaxis()
ax[2].set_xlabel('D(+0.3) minus cn: W-Z margin change (network margin s.d.)\n(untrained s.d. 0-0.12 log-odds; redraws 7, 8 constant)'); ax[2].legend(fontsize=7)
s_ = C['summary']['+0.30']; ax[2].set_title(f'(c) recurrence: W<0 and Z>0 in {s_["pattern_trained"]} trained, {s_["pattern_untrained"]} untrained', fontsize=9)
fig.tight_layout(); out = R / f'e37_das{sm}.png'; fig.savefig(out, dpi=130); print('wrote', out)
