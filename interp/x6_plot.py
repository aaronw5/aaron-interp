"""X6 summary + figure from $AI_RESULTS/jedi/x6_<family>_n<n>.json (no network runs; seconds).

Adds, per qs / qs1 channel, the exact probability that k kept slots placed at random among n ranks form one contiguous run,
(n - k + 1) / C(n, k) (the Monte Carlo null of the frozen plan has a floor of 1/1001, so Holm over ~19 channels cannot reach
0.01); Holm over the channels with that exact p for fully contiguous masks and the Monte Carlo p otherwise.
Radial maps: for each S unit also the angular-dependence check at pT = 10 GeV (variance of f over r and θ; 0 = the unit does not
depend on the particle's position, a pT-only object, for which the radial share is 1 by construction).
Writes results/jedi/x6_summary.json and results/jedi/x6_models.png.     python -m interp.x6_plot
"""
import json, os, sys
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.e19_common import holm

D = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
MODELS = [('3-feature', 64, 'n64 (reference)'), ('3-feature', 32, 'n32'), ('3-feature', 128, 'n128'),
          ('3-feature-perminv', 64, 'perminv n64')]


def summarise(R):
    n = R['n']; out = dict(model=R.get('model'), acc_M=R.get('acc_M'), acc_C=R.get('acc_C'), official=R.get('official_acc'))
    out['live'] = {s: v['n_live'] for s, v in R['E1']['stages'].items()}
    out['live_holm'] = {s: v['n_live_holm'] for s, v in R['E1']['stages'].items()}
    out['live_untrained'] = {s: v['n_live'] for s, v in R['E1_untrained']['stages'].items()}
    out['untrained_call_frac'] = R['E1_untrained']['call_frac']
    out['E2'] = dict(R['E2']['counts'], n_live=R['E2']['n_live'], pt_or_angle_only=R['E2']['fractions']['pt_or_angle_only'],
                     fisher_p=R['E2']['fisher_vs_n64']['p'],
                     offsets=[(u['c'], round(u['line_dist_rad'], 4)) for u in R['E2']['units'] if u['cat'] == 'angle_offset'])
    out['E3'] = {}
    for l in ('qs', 'qs1'):
        E = R['E3'][l]; ps, keys = [], []
        for c, e in E['channels'].items():
            k = e['kept']
            if 0 < k < n and e['contiguity'] == 1.0:
                e['p_exact_single_run'] = (n - k + 1) / comb(n, k); p = e['p_exact_single_run']
            else:
                p = e['p']
            ps.append(p); keys.append(c)
        hp = holm(ps)
        for c, h in zip(keys, hp): E['channels'][c]['holm_p_exact'] = h
        part = [c for c, e in E['channels'].items() if 0 < e['kept'] < n]
        out['E3'][l] = dict(varies_along_slots=E['varies_along_slots'], bits_shape=E['bits_shape'], n_channels=E['n_channels'],
                            n_partial=len(part), n_fully_contiguous=sum(E['channels'][c]['contiguity'] == 1.0 for c in part),
                            n_holm_exact_lt_001=sum(E['channels'][c]['holm_p_exact'] < 0.01 for c in part),
                            n_holm_exact_gt_005=sum(E['channels'][c]['holm_p_exact'] > 0.05 for c in part),
                            median_contiguity=float(np.median([E['channels'][c]['contiguity'] for c in part])) if part else None,
                            windows={c: E['channels'][c]['runs'] for c in part})
    E4 = R['E4']; out['E4'] = []
    keys = [('net_M', 'network'), ('prog_mask_M', 'float + masks'), ('prog_ones_M', 'masks removed'), ('net_C', 'network (C)')]
    for i, p in enumerate(E4['plan']):
        row = dict(edge=f'{p["edge"]}|{p["edge"] + 1}', inside=f'{p["inside"]}|{p["inside"] + 1}', units=p['units'], note=p.get('note'))
        for k, lab in keys:
            r = E4[k][i]; row[lab] = dict(ratio=r['ratio'], ci=r['ratio_ci'], p=r['perm_p'], N=r['N'], across=r['across'], inside=r['inside_mean'])
        out['E4'].append(row)
    if 'post_hoc' in E4:
        out['E4_post_hoc'] = []
        for i, p in enumerate(E4['post_hoc']['plan']):
            row = dict(edge=f'{p["edge"]}|{p["edge"] + 1}', inside=f'{p["inside"]}|{p["inside"] + 1}', units=p['units'])
            for k, lab in keys:
                r = E4['post_hoc'][k][i]; row[lab] = dict(ratio=r['ratio'], ci=r['ratio_ci'], p=r['perm_p'], N=r['N'])
            out['E4_post_hoc'].append(row)
    out['E4_program'] = E4['program']
    out['E5'] = {}
    for c, e in R['E5']['S'].items():
        f = np.asarray(e['map_pt10'], np.float64)
        out['E5'][c] = dict(slot=e['slot'], radial_share=e['radial_share'], var_pt10=float(f.var()),
                            r_dependent=bool(f.var() > 1e-12), cost=R['E1']['stages']['S']['changed'][int(c)])
    a = [e['radial_share'] for e in R['E5']['a_controls'].values() if e['radial_share'] is not None]
    hp_ = [e['radial_share'] for c, e in R['E5']['a_controls'].items() if e['radial_share'] is not None and e['radial_share'] < 0.9]
    out['E5_l1_halfplane_radial_share'] = [float(min(hp_)), float(max(hp_))] if hp_ else None
    out['E5_untrained'] = {c: e['radial_share'] for c, e in R['E5'].get('untrained_S', {}).items()}
    E6 = R['E6']; out['E6'] = {}
    for s in ('M', 'C', 'untrained_M'):
        e = E6[s]; d = e['directions']
        out['E6'][s] = dict(N_W=e['N_W'], unedited=e['W_to_Z_unedited'], plus_eta=d['0']['W_to_Z'], plus_eta_ci=d['0']['ci'],
                            dirs=[d[k]['W_to_Z'] for k in d], mean8=e['W_to_Z_mean_8dirs'], rotation=e['rotation']['W_to_Z'],
                            rotation_ci=e['rotation']['ci'], mcnemar_plus_eta_log10p=d['0']['mcnemar_vs_rotation']['log10p'],
                            mcnemar_plus_eta_n=(d['0']['mcnemar_vs_rotation']['n10'], d['0']['mcnemar_vs_rotation']['n01']))
    return out


def main():
    S = {}
    for fam, n, lab in MODELS:
        p = D / f'x6_{fam}_n{n}.json'
        if not p.exists(): print('missing', p); continue
        R = json.loads(p.read_text())
        if 'E6' not in R: print('incomplete', p); continue
        S[lab] = summarise(R)
    (D / 'x6_summary.json').write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=float)[:200000])
    if '--no-plot' not in sys.argv: plot(S)


def plot(S):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    labs = list(S); x = np.arange(len(labs))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw=dict(width_ratios=[1, 2, 1.1]))
    # (a) live l1 cut types
    cats = [('ramp', 'pT-only ramp', '#4c78a8'), ('halfplane', 'angle-only, line through axis', '#f58518'),
            ('angle_offset', 'angle-only, line 0.01–0.03 rad off axis', '#fdbf6f'), ('mixed', 'mixed pT + angle', '#d62728')]
    bot = np.zeros(len(labs))
    for k, name, col in cats:
        v = np.array([S[l]['E2'][k] for l in labs]); ax[0].bar(x, v, bottom=bot, color=col, label=name, width=0.6); bot += v
    ax[0].set_xticks(x, labs, fontsize=8.5); ax[0].set_ylabel('live l1 units (≥ 1 % of calls)')
    ax[0].set_title('(a) live first-layer units by weight pattern\n(p: shuffle null of the nonzero kernel entries, X6v)', fontsize=10)
    ax[0].legend(fontsize=7.5, loc='upper left'); ax[0].set_ylim(0, 25)
    vfile = {'n64 (reference)': '3-feature_n64', 'n32': '3-feature_n32', 'n128': '3-feature_n128', 'perminv n64': '3-feature-perminv_n64'}
    for i, l in enumerate(labs):
        fv = D / f'x6_verify_{vfile.get(l, "")}.json'
        if fv.exists():
            pv = json.loads(fv.read_text())['V2']['all_units']['p']
            ax[0].text(x[i], bot[i] + 0.4, f'p = {pv:.2f}', ha='center', fontsize=8)
    # (b) across / inside ratio at the first planned (frozen) edge; n128 also its post hoc edge 19|20
    w = 0.25; cols = {'network': '#222222', 'float + masks': '#4c78a8', 'masks removed': '#bbbbbb'}
    # every tested edge: frozen edges and post hoc edges (X6v: all edges shown)
    rowsb = []
    for l in labs:
        short = l.split(' ')[0]
        rowsb += [(l, r, f'{short}\n{r["edge"]} vs {r["inside"]}') for r in S[l]['E4']]
        rowsb += [(l, r, f'{short} post hoc\n{r["edge"]} vs {r["inside"]}') for r in S[l].get('E4_post_hoc', [])]
    xb = np.arange(len(rowsb))
    for j, k in enumerate(cols):
        r = np.array([(row.get(k) or {}).get('ratio') or np.nan for _, row, _ in rowsb], dtype=float)
        r = np.where(r > 0, r, np.nan)
        ci = np.array([(row.get(k) or {}).get('ci') or [np.nan, np.nan] for _, row, _ in rowsb], dtype=float)
        ax[1].bar(xb + (j - 1) * w, r, w, color=cols[k], label=k)
        ax[1].errorbar(xb + (j - 1) * w, r, yerr=[r - ci[:, 0], ci[:, 1] - r], fmt='none', ecolor='#d62728', lw=1)
    ax[1].axhline(1, color='k', lw=0.8, ls=':'); ax[1].set_yscale('log')
    ax[1].set_xticks(xb, [t.replace(" vs ", "\nvs ") for _, _, t in rowsb], fontsize=7)
    ax[1].set_ylabel('mean |ΔS|/sd(S): across-edge swap ÷ inside swap')
    ax[1].set_title('(b) (Δη, Δφ) swap of adjacent pT ranks: across ÷ inside a window', fontsize=10); ax[1].legend(fontsize=7.5)
    # (c) translation W -> Z
    t = np.array([S[l]['E6']['M']['plus_eta'] for l in labs]); tc = np.array([S[l]['E6']['M']['plus_eta_ci'] for l in labs])
    rr = np.array([S[l]['E6']['M']['rotation'] for l in labs]); rc = np.array([S[l]['E6']['M']['rotation_ci'] for l in labs])
    lo8 = np.array([min(S[l]['E6']['M']['dirs']) for l in labs]); hi8 = np.array([max(S[l]['E6']['M']['dirs']) for l in labs])
    ax[2].bar(x - 0.17, t, 0.34, color='#f58518', label='translation 0.06 rad along +Δη')
    ax[2].errorbar(x - 0.17, t, yerr=[t - tc[:, 0], tc[:, 1] - t], fmt='none', ecolor='k', lw=1)
    ax[2].vlines(x - 0.17, lo8, hi8, color='#7f3b08', lw=3, alpha=0.5, label='range over 8 directions')
    ax[2].bar(x + 0.17, rr, 0.34, color='#bbbbbb', label='rotation about the axis, matched displacement')
    ax[2].errorbar(x + 0.17, rr, yerr=[rr - rc[:, 0], rc[:, 1] - rr], fmt='none', ecolor='k', lw=1)
    ax[2].set_xticks(x, labs, fontsize=8.5); ax[2].set_ylabel('fraction of true-W jets called Z'); ax[2].set_ylim(0, 1)
    ax[2].set_title('(c) rigid shift of the jet (invariant mass unchanged)', fontsize=10); ax[2].legend(fontsize=7.5, loc='upper right')
    fig.suptitle('Four official JEDI-linear checkpoints, 20,000 test jets each (seed 0); bars 95 % intervals. '
                 'perminv: no slot masks, n64 ranks used in (b)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(D / 'x6_models.png', dpi=130); print('wrote', D / 'x6_models.png')


if __name__ == '__main__':
    main()
