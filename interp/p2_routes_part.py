"""P2 — ParT: how much of the response to a real mass change goes through the pair features and how much through the
per-particle inputs.

ParT's pair features (ln kT, ln z, ln ΔR_ij, ln m²_ij; built inside the model by weaver's pairwise_lv_fts from the particles'
(px, py, pz, E)) allow the invariant mass to be computed; its per-particle inputs are measured from the jet axis (deta, dphi,
deltaR) or relative to the jet (logptrel, logerel). P1 / P1v: a rigid shift (mass and pair features unchanged) changes the
calls, for ParT_full entirely through the per-particle deltaR. P2 splits the response to edits that change the true mass.

Sample: P1's 10,000 jets (rows 3,000–4,000 of each class's first test file, 1,000 per class); baseline Jb = build(J0) (δ = 0).
Edits (part_core.build from the stored particles' pT, Δη, Δφ, particle mass; padded slots stay padded):
  ang k, k ∈ {0.9, 1.1, 1.25}: every real particle's (Δη, Δφ) × k; pT and particle mass kept; px, py, pz, E, jet_pt,
          jet_energy rebuilt. True mass × ≈ k.
  pt c, c ∈ {0.8, 1.25}: every real particle's pT and mass × c; Δη, Δφ kept: every 4-vector × c, true mass × c.
  ang 1.0, pt 1.0: the same code with factor 1 (controls).
Variants, from the fed per-particle inputs x (network.inputs: every feature of the model's yaml, scaled and clipped) and the
4-vectors v (the only source of the pair features; the model uses v nowhere else):
  (a) full edit: x_edit, v_edit;   (b) pair route only: x_base, v_edit;   (c) per-particle route only: x_edit, v_base.
  (b) at ang 1.1 is the reverse of P1's shift: pair features from a jet whose particles were moved apart (true mass + 9.4 %),
  every per-particle (axis-relative) input from the unedited jet.
Metrics (as P1 / P1v; baseline = the forward on (x_base, v_base)): accuracy, calls changed; W→Z, Z→W, W→W, Z→Z; mean
Δ(logit_W − logit_Z) on W and on Z jets; index 97 shift / s.d. on all, W, Z jets; P1's ridge read-out of log m (fitted on the
unedited class token z, train half, seed 1, same alphas) mean change on the test half, against the true Δlog m (jmass of the
edited jet). Route shares: (b)/(a), (c)/(a) for Δlog m̂, ΔW→Z, ΔZ→W, Δ(logit_W − logit_Z) on W, index 97 on W. Additivity:
(b) + (c) vs (a) for the same quantities, and per jet: logits L0 + ΔL_b + ΔL_c (calls agreeing with (a), W→Z), R² of
Δlog m̂_b + Δlog m̂_c and of Δ(logit_W − logit_Z)_b + _c as predictions of (a).
Edit description: mean change of the 4 pair features on real–real pairs (i ≠ j) and of every fed per-particle feature on
real slots; padding and non-kinematic branches checked unchanged.
  ./part_job.sh p2 'python -m interp.p2_routes_part'        -> results/part/p2_routes.json, results/site/T12_part_routes.png
  ./part_job.sh p2-quick 'python -m interp.p2_routes_part 50'  (50 jets per class: p2_routes_50.json, no table)
  python interp/p2_routes_part.py --table                   (re-render the table from the json; needs matplotlib only)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
SITE = OUT.parent / 'site'
EDITS = [('ang', 0.9), ('ang', 1.1), ('ang', 1.25), ('pt', 0.8), ('pt', 1.25), ('ang', 1.0), ('pt', 1.0)]
VAR = ('a_full', 'b_pair_only', 'c_perparticle_only')
SHARE_KEYS = ('dlogm_hat', 'dW_to_Z', 'dZ_to_W', 'dlogit_WZ_on_W', 'dlogit_WZ_on_Z', 'n98_W', 'n98_Z')
PAIR_NAMES = ['lnkT', 'lnz', 'lnDeltaR', 'lnm2']
log = lambda *a: print(*a, flush=True)
ename = lambda kind, f: f'{kind}{f:g}'


def pair_delta(vb, ve, m, batch=48):
    """mean and mean |.| of the change of weaver's 4 pair features (pairwise_lv_fts as PairEmbed calls it) between the
    4-vectors vb and ve, over real–real pairs with i ≠ j"""
    import torch
    from functools import partial
    from weaver.nn.model.ParticleTransformer import pairwise_lv_fts
    f = partial(pairwise_lv_fts, num_outputs=4, eps=1e-8, for_onnx=False)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    S, SA, n = np.zeros(4), np.zeros(4), 0
    order = np.argsort(m[:, 0].sum(1), kind='stable')
    with torch.no_grad():
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]; P = int(m[idx, 0].sum(1).max())
            tb, tE = (torch.from_numpy(np.ascontiguousarray(a[idx, :, :P])).to(dev) for a in (vb, ve))
            mm = torch.from_numpy(np.ascontiguousarray(m[idx, 0, :P])).to(dev) > 0
            off = mm[:, :, None] & mm[:, None, :] & ~torch.eye(P, dtype=torch.bool, device=dev)[None]
            D = f(tE.unsqueeze(-1), tE.unsqueeze(-2)) - f(tb.unsqueeze(-1), tb.unsqueeze(-2))
            for k in range(4):
                d = D[:, k][off].double(); S[k] += float(d.sum()); SA[k] += float(d.abs().sum())
            n += int(off.sum())
    return dict(pairs=n, mean=dict(zip(PAIR_NAMES, (S / n).tolist())), mean_abs=dict(zip(PAIR_NAMES, (SA / n).tolist())))


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from interp.part_core import ParTNetwork, inputs, sample, kin, build, jmass, moments, CL, N98
    from interp.p1v_verify_part import Ridge, r2, split, fwd
    from jetdistill.part.network import FEATURES
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    PER = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    J0 = sample(PER, 3000); y = J0['label'].astype(int); N = len(y)
    pt, de, dp, ms = kin(J0); real = pt > 0
    Jb = build(J0, pt, de, dp, ms); m0 = jmass(Jb); logm0 = np.log(np.maximum(m0, 1.0)); M0m = moments(*kin(Jb)[:3])
    trf, trv, tr, te = split(N)
    log(f'{N} jets ({PER}/class, rows 3000-{3000 + PER}); mass median {np.median(m0):.1f} GeV ({time.time() - t0:.0f} s)')

    JE = {}
    for kind, f in EDITS:
        JE[ename(kind, f)] = build(J0, pt, de * f, dp * f, ms) if kind == 'ang' else build(J0, pt * f, de, dp, ms * f)
    res = dict(N=N, per_class=PER, rows=[3000, 3000 + PER], seed=1, edits=[ename(*e) for e in EDITS], variants=list(VAR), n98_index=N98,
               mass_median=float(np.median(m0)), truth={}, models={})
    OTHER = ('charge', 'isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon', 'd0val', 'd0err', 'dzval', 'dzerr', 'label')
    for name, Je in JE.items():
        me = jmass(Je); dlm = np.log(np.maximum(me, 1.0)) - logm0; Me = moments(*kin(Je)[:3]); pad = Je['mask'] == 0
        t = dict(dlogm_mean=float(dlm.mean()), dlogm_te=float(dlm[te].mean()), dlogm_W=float(dlm[y == iW].mean()), dlogm_Z=float(dlm[y == iZ].mean()),
                 dlogm_absdev_from_mean_max=float(np.abs(dlm - dlm.mean()).max()),
                 pred_sqrtM0M2_te=float(0.5 * np.log(Me['M0M2'] / M0m['M0M2'])[te].mean()),
                 mass_median=float(np.median(me)), mass_median_W=float(np.median(me[y == iW])), mass_median_Z=float(np.median(me[y == iZ])),
                 mask_equal=bool(np.array_equal(Je['mask'], Jb['mask'])),
                 padded_zero=bool(all(np.all(Je[k][pad] == 0) for k in ('px', 'py', 'pz', 'energy', 'deta', 'dphi'))),
                 other_branches_equal=bool(all(np.array_equal(Je[k], Jb[k]) for k in OTHER)),
                 jet_pt_ratio_mean=float((Je['jet_pt'] / Jb['jet_pt']).mean()), jet_energy_ratio_mean=float((Je['jet_energy'] / Jb['jet_energy']).mean()))
        res['truth'][name] = t
        log(f'edit {name:7s}: true Δlog m mean {t["dlogm_mean"]:+.4f} (test half {t["dlogm_te"]:+.4f}, W {t["dlogm_W"]:+.4f}); mass median W {t["mass_median_W"]:.1f} Z {t["mass_median_Z"]:.1f} GeV; '
            f'½Δlog M0M2 {t["pred_sqrtM0M2_te"]:+.4f}; mask equal {t["mask_equal"]}, padded zero {t["padded_zero"]}, other branches equal {t["other_branches_equal"]}; '
            f'jet_pt × {t["jet_pt_ratio_mean"]:.4f}, jet_energy × {t["jet_energy_ratio_mean"]:.4f}')

    pairs_done = False
    for n in ('full', 'kin'):
        net = ParTNetwork(n); t1 = time.time(); names = [f[0] for f in FEATURES[n]]
        xb, vb, mb = inputs(Jb, n); real_slot = mb[:, 0] > 0
        z0, L0 = fwd(net, xb, vb, mb); Rr = net.run(Jb); c0 = L0.argmax(1); sd0 = z0.std(0)
        ridge = Ridge(z0, logm0, trf, trv, tr); rz0 = ridge(z0[te])
        dWZ0 = L0[:, iW] - L0[:, iZ]
        base = dict(acc=float((c0 == y).mean()), W_to_Z=float((c0[y == iW] == iZ).mean()), Z_to_W=float((c0[y == iZ] == iW).mean()),
                    W_to_W=float((c0[y == iW] == iW).mean()), Z_to_Z=float((c0[y == iZ] == iZ).mean()),
                    ridge_r2=r2(rz0, logm0[te]), alpha=float(ridge.alpha), n98_sd=float(sd0[N98]),
                    run_vs_fwd_max_dlogit=float(np.abs(Rr['logits'] - L0).max()), run_vs_fwd_calls_changed=float((Rr['logits'].argmax(1) != c0).mean()),
                    calls_from_W=[float((c0[y == iW] == k).mean()) for k in range(10)])
        row = dict(features=names, baseline=base, edits={})
        log(f'ParT_{n}: baseline {base}')

        def metrics(z, L):
            c = L.argmax(1); d97 = (z[:, N98] - z0[:, N98]) / sd0[N98]; dl = ridge(z[te]) - rz0; dWZ = (L[:, iW] - L[:, iZ]) - dWZ0
            e = dict(acc=float((c == y).mean()), changed=float((c != c0).mean()), W_to_Z=float((c[y == iW] == iZ).mean()), Z_to_W=float((c[y == iZ] == iW).mean()),
                     W_to_W=float((c[y == iW] == iW).mean()), Z_to_Z=float((c[y == iZ] == iZ).mean()),
                     dlogit_WZ_on_W=float(dWZ[y == iW].mean()), dlogit_WZ_on_Z=float(dWZ[y == iZ].mean()),
                     n98_all=float(d97.mean()), n98_W=float(d97[y == iW].mean()), n98_Z=float(d97[y == iZ].mean()),
                     dlogm_hat=float(dl.mean()), dlogm_hat_W=float(dl[y[te] == iW].mean()), dlogm_hat_Z=float(dl[y[te] == iZ].mean()),
                     max_abs_dlogit=float(np.abs(L - L0).max()), calls_from_W=[float((c[y == iW] == k).mean()) for k in range(10)],
                     own_class=[float((c[y == k] == k).mean()) for k in range(10)])
            e['dW_to_Z'] = e['W_to_Z'] - base['W_to_Z']; e['dZ_to_W'] = e['Z_to_W'] - base['Z_to_W']
            return e, dict(L=L, dl=dl, dWZ=dWZ)

        for name, Je in JE.items():
            xe, ve, me = inputs(Je, n)
            assert np.array_equal(me, mb) and xe.shape == xb.shape and ve.shape == vb.shape
            feat = {k: dict(mean_real=float((xe[:, i] - xb[:, i])[real_slot].mean()), mean_abs_real=float(np.abs(xe[:, i] - xb[:, i])[real_slot].mean()),
                            max_abs_real=float(np.abs(xe[:, i] - xb[:, i])[real_slot].max())) for i, k in enumerate(names)}
            E = dict(fed_feature_change=feat, vec_max_abs_change=float(np.abs(ve - vb).max()), variants={})
            if not pairs_done:
                res['truth'][name]['pair_feature_change'] = pd = pair_delta(vb, ve, mb)
                log(f'   {name}: pair features Δ mean {pd["mean"]} (mean |Δ| {pd["mean_abs"]}) on {pd["pairs"]} pairs')
            log(f'   {name}: fed per-particle Δ mean (real slots): ' + ', '.join(f'{k} {v["mean_real"]:+.4f}' for k, v in feat.items() if v['max_abs_real'] > 0))
            arr = {}
            for v_, (x, v) in zip(VAR, ((xe, ve), (xb, ve), (xe, vb))):
                z, L = fwd(net, x, v, mb); e, a = metrics(z, L); E['variants'][v_] = e; arr[v_] = a
                log(f'   {name:7s} {v_:18s} | acc {100 * e["acc"]:5.1f} changed {100 * e["changed"]:5.2f} % | W→Z {e["W_to_Z"]:.3f} Z→W {e["Z_to_W"]:.3f} W→W {e["W_to_W"]:.3f} | '
                    f'Δ(lW−lZ) W {e["dlogit_WZ_on_W"]:+.3f} Z {e["dlogit_WZ_on_Z"]:+.3f} | idx97/sd W {e["n98_W"]:+.3f} Z {e["n98_Z"]:+.3f} | Δlog m̂ {e["dlogm_hat"]:+.4f} '
                    f'(true {res["truth"][name]["dlogm_te"]:+.4f}) | max|ΔL| {e["max_abs_dlogit"]:.2e}')
            ea, eb, ec = (E['variants'][v] for v in VAR)
            E['share'] = {k: dict(pair=(eb[k] / ea[k] if abs(ea[k]) > 1e-9 else None), perparticle=(ec[k] / ea[k] if abs(ea[k]) > 1e-9 else None)) for k in SHARE_KEYS}
            La, Lb, Lc = (arr[v]['L'] for v in VAR); Ladd = Lb + Lc - L0; cadd = Ladd.argmax(1); ca = La.argmax(1)
            dla, dlb, dlc = (arr[v]['dl'] for v in VAR); wa, wb, wc = (arr[v]['dWZ'] for v in VAR)
            ok = float(np.var(dla)) > 1e-12
            E['additivity'] = dict(sum_bc={k: eb[k] + ec[k] for k in SHARE_KEYS}, a={k: ea[k] for k in SHARE_KEYS},
                                   ratio={k: ((eb[k] + ec[k]) / ea[k] if abs(ea[k]) > 1e-9 else None) for k in SHARE_KEYS},
                                   perjet_calls_agree_with_a=float((cadd == ca).mean()), perjet_changed_add=float((cadd != c0).mean()),
                                   perjet_W_to_Z_add=float((cadd[y == iW] == iZ).mean()), perjet_Z_to_W_add=float((cadd[y == iZ] == iW).mean()),
                                   perjet_r2_dlogm_hat=(r2(dlb + dlc, dla) if ok else None), perjet_r2_dlogit_WZ=(r2(wb + wc, wa) if float(np.var(wa)) > 1e-12 else None),
                                   perjet_r2_dlogits=(r2((Lb + Lc - 2 * L0).ravel(), (La - L0).ravel()) if float(np.var(La - L0)) > 1e-12 else None),
                                   perjet_corr_dlogm_hat_b_c=(float(np.corrcoef(dlb, dlc)[0, 1]) if np.std(dlb) > 1e-9 and np.std(dlc) > 1e-9 else None))
            A = E['additivity']
            log(f'   {name:7s} shares pair / per-particle: ' + ', '.join(f'{k} {s["pair"] if s["pair"] is None else round(s["pair"], 3)} / {s["perparticle"] if s["perparticle"] is None else round(s["perparticle"], 3)}' for k, s in E['share'].items()))
            log(f'   {name:7s} additivity (b)+(c) / (a): ' + ', '.join(f'{k} {A["sum_bc"][k]:+.4f} vs {A["a"][k]:+.4f}' for k in SHARE_KEYS)
                + f' | per jet: calls agree {A["perjet_calls_agree_with_a"]:.4f}, changed {100 * A["perjet_changed_add"]:.2f} %, W→Z {A["perjet_W_to_Z_add"]:.3f}, '
                  f'R² Δlog m̂ {A["perjet_r2_dlogm_hat"]}, R² Δ(lW−lZ) {A["perjet_r2_dlogit_WZ"]}, R² ΔL {A["perjet_r2_dlogits"]}')
            row['edits'][name] = E
            del xe, ve, me, arr
        pairs_done = True
        row['reverse_of_P1'] = dict(edit='ang1.1', variant='b_pair_only', **{k: row['edits']['ang1.1']['variants']['b_pair_only'][k] for k in
                                    ('W_to_Z', 'W_to_W', 'Z_to_W', 'dW_to_Z', 'dlogit_WZ_on_W', 'n98_W', 'dlogm_hat', 'calls_from_W', 'changed')})
        res['models'][n] = row; log(f'ParT_{n} done ({time.time() - t1:.0f} s)')
        del net
    fn = OUT / ('p2_routes.json' if PER == 1000 else f'p2_routes_{PER}.json')
    json.dump(res, open(fn, 'w'), indent=1); log(f'wrote {fn}')
    if PER == 1000: table(res)
    log(f'P2 DONE ({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------------------------------------ table (site/tables.py style)
def table(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    if res is None: res = json.load(open(OUT / 'p2_routes.json'))
    lab = {'ang0.9': 'angles × 0.9', 'ang1.1': 'angles × 1.1', 'ang1.25': 'angles × 1.25', 'pt0.8': 'pT × 0.8', 'pt1.25': 'pT × 1.25'}
    tri = lambda E, k, fm: ' / '.join(fm(E['variants'][v][k]) for v in VAR)
    sh = lambda E, k: ' / '.join('–' if E['share'][k][r] is None else f'{E["share"][k][r]:.2f}' for r in ('pair', 'perparticle'))
    rows = []
    for n in ('full', 'kin'):
        M = res['models'][n]
        for name in lab:
            E = M['edits'][name]; A = E['additivity']
            rows.append([f'ParT_{n}', f'{lab[name]}  ({res["truth"][name]["dlogm_te"]:+.3f})',
                         tri(E, 'dlogm_hat', lambda x: f'{x:+.3f}'), f'{A["sum_bc"]["dlogm_hat"]:+.3f}',
                         tri(E, 'W_to_Z', lambda x: f'{x:.2f}'), tri(E, 'Z_to_W', lambda x: f'{x:.2f}'),
                         tri(E, 'changed', lambda x: f'{100 * x:.1f}'), tri(E, 'n98_W', lambda x: f'{x:+.2f}'),
                         sh(E, 'dlogm_hat'), sh(E, 'dlogit_WZ_on_W')])
    cols = ['model', 'edit  (true Δlog m)', 'Δlog m, read-out\n(a) / (b) / (c)', '(b) + (c)\nΔlog m', 'W→Z\n(a) / (b) / (c)', 'Z→W\n(a) / (b) / (c)',
            'calls changed [%]\n(a) / (b) / (c)', 'index 97 on W [s.d.]\n(a) / (b) / (c)', 'share of Δlog m\npair / per-particle',
            'share of Δ(logit W − Z), W jets\npair / per-particle']
    widths = [0.85, 1.8, 2.0, 0.8, 1.5, 1.5, 1.6, 1.8, 1.5, 2.3]
    bf, bk = res['models']['full']['baseline'], res['models']['kin']['baseline']
    rf, rk = res['models']['full']['reverse_of_P1'], res['models']['kin']['reverse_of_P1']
    ctrl = max(res['models'][n]['edits'][c]['variants'][v]['changed'] for n in ('full', 'kin') for c in ('ang1', 'pt1') for v in VAR)
    note = ('(a) full edit; (b) pair route only: 4-vectors (pair features) from the edited jet, every per-particle input from the unedited jet; '
            '(c) per-particle route only: per-particle inputs from the edited jet, 4-vectors from the unedited jet.\n'
            f'Unedited: ParT_full W→Z {bf["W_to_Z"]:.2f}, Z→W {bf["Z_to_W"]:.2f}; ParT_kin W→Z {bk["W_to_Z"]:.2f}, Z→W {bk["Z_to_W"]:.2f}. '
            'Shares = (b)/(a), (c)/(a). Index 97 is the W/Z neuron of ParT_full (not a W/Z neuron in ParT_kin). Read-out: P1 ridge on the class token, test half.\n'
            f'Reverse of P1\'s shift = (b) at angles × 1.1: W→Z {bf["W_to_Z"]:.2f} → {rf["W_to_Z"]:.2f} (full), {bk["W_to_Z"]:.2f} → {rk["W_to_Z"]:.2f} (kin). '
            f'Controls (factor 1.0 through the same code): calls changed ≤ {100 * ctrl:.2f} % in (a), (b), (c). {res["N"]:,} JetClass test jets.\n'
            f'ParT_kin, pT × 0.8, (b): {res["models"]["kin"]["edits"]["pt0.8"]["variants"]["b_pair_only"]["calls_from_W"][1]:.2f} of true W jets called H→bb '
            f'(unedited {res["models"]["kin"]["baseline"]["calls_from_W"][1]:.3f}).')
    SITE.mkdir(parents=True, exist_ok=True)
    Wd = sum(widths); Ht, Hti, Hn = 0.34 * (len(rows) + 1.6), 0.45, 0.9; H = Ht + Hti + Hn
    fig = plt.figure(figsize=(Wd, H), dpi=160); ax = fig.add_axes([0.003, Hn / H, 0.994, Ht / H]); ax.axis('off')
    t = ax.table(cellText=rows, colLabels=cols, colWidths=[w / Wd for w in widths], bbox=[0, 0, 1, 1], cellLoc='left', colLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(9)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#d8d4ca'); cell.PAD = 0.03
        if r == 0: cell.set_facecolor('#eeebe3'); cell.set_text_props(weight='bold'); cell.set_height(cell.get_height() * 1.6)
        elif r % 2 == 0: cell.set_facecolor('#f8f7f3')
    fig.text(0.003, 1 - 0.12 / H, 'ParT: response to a real mass change, by input route (P2)', va='top', fontsize=11, weight='bold')
    fig.text(0.003, (Hn - 0.1) / H, note, va='top', fontsize=8, color='#555')
    fig.savefig(SITE / 'T12_part_routes.png', bbox_inches='tight', pad_inches=0.1); log(f'wrote {SITE / "T12_part_routes.png"}')


if __name__ == '__main__':
    if sys.argv[1:] == ['--table']: table()
    else: main()
