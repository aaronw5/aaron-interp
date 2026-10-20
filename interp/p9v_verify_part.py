"""P9 verification (P9v): re-measure ParT_full's W-vs-Z faithfulness-vs-size curve on a fresh sample, with the P9 edge
ranking (results/part/p9_eap_scores.npz: EAP-IG scores, MLP-unit grouping) frozen, plus controls P9 did not run.

Sample: JetClass val_5M WToQQ_120.root / ZToQQ_120.root (files P9 did not use), NPER random rows each, numpy default_rng(9090).
Corruptions (donors, opposite class):
  B   : P9's rule (one of the 5 nearest in jet pT among the other class), default_rng(31)
  Bm  : opposite class, among the 30 nearest in jet pT the 3 nearest in multiplicity, one at random (default_rng(32)).
        Tests P9 issue 4 (particle slots are paired by pT rank between jets of different multiplicity).
Masks (graph and grouping identical to P9; checked):
  edge_top{k}      top-k edges by |EAP-IG| (dense grid 10 .. 15,000)
  eap_top{k}       top-k edges by |EAP| (alpha = 1) instead of EAP-IG (ranking control)
  rnd{k}_{r}       k uniformly random edges, 3 draws (size control for the curve)
  node_top{k}      literal node circuits, top-k nodes by summed outgoing EAP-IG (dense grid 8 .. 320)
  acdc{tau}_literal the three P9 ACDC literal circuits (node lists from p9_eap_wz.json)
  randlit89_{r}    20 random literal circuits matched to the 89-node ACDC circuit per node type+block (B only)
Metric: F = sum s (d_circ - d_corr) / sum s (d_full - d_corr), d = l_W - l_Z; W/Z accuracy. Bootstrap over jets (NB = 1000,
identical resampling weights for B and Bm so differences are paired). Slope of F per doubling of the edge count (least squares
on log2 k over 10 <= k <= 8,000) with its bootstrap interval; edges needed for F = 0.8 by log-linear interpolation.

  ./part_job.sh c-p9-verify 'python -m interp.p9v_verify_part'   -> $AI_RESULTS/part/p9_verify.json, p9_verify.png
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.p9_eap_wz_part import Graph, Net, multi_eval, Scorer, donors_B, load_sel, ci, mcnemar, clean, log, NB, OUT, VAL

NPER = int(os.environ.get('P9V_NPER', 3000))
FN = OUT / 'p9_verify.json'


def donors_Bm(pt, mult, y, iW, iZ, rng, npt=30, nm=3):
    don = np.zeros(len(y), int)
    for a, b in ((iW, iZ), (iZ, iW)):
        ia, ib = np.where(y == a)[0], np.where(y == b)[0]
        for i in ia:
            c = ib[np.argsort(np.abs(pt[ib] - pt[i]))[:npt]]
            c = c[np.argsort(np.abs(mult[c] - mult[i]), kind='stable')[:nm]]; don[i] = rng.choice(c)
    return don


def interp08(ks, fs):
    ks, fs = np.asarray(ks, float), np.asarray(fs, float)
    for i in range(1, len(ks)):
        if fs[i] >= 0.8 > fs[i - 1]:
            t = (0.8 - fs[i - 1]) / (fs[i] - fs[i - 1]); return float(2 ** (np.log2(ks[i - 1]) + t * (np.log2(ks[i]) - np.log2(ks[i - 1]))))
    return None


def main():
    import torch, uproot
    from interp.part_core import kin, build, jmass, cat, sub, CL, PRE
    OUT.mkdir(parents=True, exist_ok=True); iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    G = Graph(); sc = np.load(OUT / 'p9_eap_scores.npz'); ig, eap = sc['ig'], sc['eap']
    assert (sc['W'] == G.W).all() and (sc['R'] == G.R).all(), 'graph differs from P9'
    p9 = json.load(open(OUT / 'p9_eap_wz.json'))
    res = dict(NPER=NPER, NB=NB, sample_seed=9090, files={})
    parts = []; rng = np.random.default_rng(9090)
    for c in ('Wqq', 'Zqq'):
        p = VAL / f'{PRE[c]}_120.root'; N = uproot.open(p)['tree'].num_entries; r = np.sort(rng.choice(N, NPER, replace=False))
        (J,), _ = load_sel(p, [r]); parts.append(J); res['files'][c] = dict(file=p.name, entries=N, first_rows=r[:5].tolist())
    J0 = cat(parts); J = build(J0, *kin(J0)); y = J0['label'].astype(int); mass = jmass(J); mult = J['mask'].sum(1)
    log(f'sample {len(y)} jets, classes {np.bincount(y, minlength=10).tolist()}')
    donB = donors_B(J['jet_pt'], y, iW, iZ, np.random.default_rng(31))
    donM = donors_Bm(J['jet_pt'], mult, y, iW, iZ, np.random.default_rng(32))
    res['donors'] = {k: dict(rel_dpt_median=float(np.median(np.abs(J['jet_pt'][d] - J['jet_pt']) / J['jet_pt'])),
                             abs_dmult_median=float(np.median(np.abs(mult[d] - mult))), abs_dmult_mean=float(np.mean(np.abs(mult[d] - mult))),
                             dm_median=float(np.median(mass[d] - mass)), same_class=float((y[d] == y).mean()))
                     for k, d in (('B', donB), ('Bm', donM))}
    log(f'donors {res["donors"]}')

    net = Net(G); net.set_perm([np.asarray(p) for p in sc['perms']])
    tm = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(net.dev)
    rank = np.argsort(-np.abs(ig), kind='stable'); rank_eap = np.argsort(-np.abs(eap), kind='stable')
    ns = {nd: 0.0 for nd in G.nodes}
    for e in range(G.NE): ns[G.W[e]] += ig[e]
    nr = sorted(G.nodes, key=lambda nd: -abs(ns[nd]))
    assert nr[:64] == p9['circuits']['node_top64']['nodes'], 'node ranking differs from P9'
    EK = [10, 30, 100, 300, 1000, 1075, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000]
    NK = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320]
    masks = {}
    for k in EK:
        a = np.zeros(G.NE); a[rank[:k]] = 1; masks[f'edge_top{k}'] = tm(a)
    for k in (1075, 4000, 8000):
        a = np.zeros(G.NE); a[rank_eap[:k]] = 1; masks[f'eap_top{k}'] = tm(a)
        for r_ in range(3):
            a = np.zeros(G.NE); a[np.random.default_rng(100 + r_ + k).choice(G.NE, k, replace=False)] = 1; masks[f'rnd{k}_{r_}'] = tm(a)
    for k in NK: masks[f'node_top{k}'] = tm(G.node_mask(nr[:k]))
    acdc = {t: p9['circuits'][f'acdc{t}_edges']['nodes'] for t in p9['TAUS']}
    for t, nodes in acdc.items(): masks[f'acdc{t}_literal'] = tm(G.node_mask(nodes))
    masks['ref_lit_b2h8_b1h3_n98'] = tm(G.node_mask(['C1H7', 'C0H2', 'z97']))
    masks['all_ablated'] = tm(np.zeros(G.NE)); masks['full'] = tm(np.ones(G.NE))
    res['n_edges'] = {k: int(v.sum().item()) for k, v in masks.items()}
    rm = {}; ncats = {}
    for nd in G.nodes: ncats.setdefault(G.ncat(nd), []).append(nd)
    big = acdc[min(acdc)]; rr = np.random.default_rng(77)
    for r_ in range(20):
        nodes = []
        for c_ in sorted({G.ncat(nd) for nd in big}):
            nodes += list(rr.choice(ncats[c_], sum(G.ncat(nd) == c_ for nd in big), replace=False))
        rm[f'randlit89_{r_}'] = tm(G.node_mask(nodes))

    idx = np.arange(len(y)); out = {}
    for nm_, don, extra in (('B', donB, rm), ('Bm', donM, {})):
        t1 = time.time(); L = multi_eval(net, J, idx, don, {**masks, **extra}, check=True)
        S = Scorer(y, iW, iZ, L['clean'], L['corr'], np.random.default_rng(5))
        R = {k: S.summary(L[k]) for k in list(masks) + list(extra)}
        e = dict(exactness=L['_check'], unedited_wz_acc=float(S.ok0.mean()), corr_wz_acc=float((np.where(S.dco > 0, iW, iZ) == y).mean()),
                 seconds=time.time() - t1)
        # slope per doubling and F = 0.8 crossing, with bootstrap
        sel = [k for k in EK if 10 <= k <= 8000]; X = np.log2(sel)
        Fm = np.array([R[f'edge_top{k}']['F'] for k in sel]); Fb = np.stack([R[f'edge_top{k}']['_fb'] for k in sel], 1)
        slope = float(np.polyfit(X, Fm, 1)[0]); sb = np.polyfit(X, Fb.T, 1)[0]
        seg = {f'{a}-{b}': float((R[f'edge_top{b}']['F'] - R[f'edge_top{a}']['F']) / np.log2(b / a)) for a, b in ((10, 100), (100, 1000), (1000, 8000))}
        Fall = [R[f'edge_top{k}']['F'] for k in EK]; Fall_b = np.stack([R[f'edge_top{k}']['_fb'] for k in EK], 1)
        k08 = interp08(EK, Fall); k08b = [interp08(EK, row) for row in Fall_b]; k08b = [v for v in k08b if v is not None]
        Fn = [R[f'node_top{k}']['F'] for k in NK]
        e.update(slope_per_doubling_10_8000=slope, slope_ci=ci(sb), segment_slopes=seg, edges_for_F0p8=k08, edges_for_F0p8_ci=ci(k08b),
                 nodes_for_F0p8=interp08(NK, Fn))
        if extra:
            Fr = np.array([R[f'randlit89_{r_}']['F'] for r_ in range(20)]); Ar = np.array([R[f'randlit89_{r_}']['wz_acc'] for r_ in range(20)])
            e['randlit89'] = dict(F_mean=float(Fr.mean()), F_max=float(Fr.max()), acc_mean=float(Ar.mean()), acc_max=float(Ar.max()),
                                  p_perm_F=float((1 + (Fr >= R[f'acdc{min(acdc)}_literal']['F']).sum()) / 21))
        e['circuits'] = {k: dict(F=v['F'], F_ci=v['F_ci'], wz_acc=v['wz_acc'], wz_acc_ci=v['wz_acc_ci'], mcnemar_vs_full=v['mcnemar_vs_full'],
                                 n_edges=res['n_edges'].get(k)) for k, v in R.items() if not k.startswith('randlit')}
        out[nm_] = (e, R); res[nm_] = e
        log(f'{nm_} ({e["seconds"]:.0f} s): exactness {e["exactness"]}; unedited W/Z {e["unedited_wz_acc"]:.4f}, all resampled {e["corr_wz_acc"]:.4f}; '
            f'slope {slope:.3f} {np.round(ci(sb), 3).tolist()}; segments {seg}; F=0.8 at {k08} edges {np.round(ci(k08b), 0).tolist()}, {e["nodes_for_F0p8"]} nodes')
        for k in masks: log(f'  {k:24s} edges {res["n_edges"][k]:6d}  F {R[k]["F"]:+.3f} {np.round(R[k]["F_ci"], 3).tolist()}  W/Z {R[k]["wz_acc"]:.4f}')
    # paired B vs Bm differences (same bootstrap weights)
    RB, RM = out['B'][1], out['Bm'][1]; diff = {}
    for k in ('edge_top1075', 'edge_top4000', 'edge_top8000', 'node_top128', 'node_top256', f'acdc{min(acdc)}_literal'):
        d = RM[k]['_fb'] - RB[k]['_fb']
        diff[k] = dict(F_B=RB[k]['F'], F_Bm=RM[k]['F'], dF=RM[k]['F'] - RB[k]['F'], dF_ci=ci(d), p_two_sided=float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())) or 1 / NB))
    res['B_vs_Bm'] = diff; log(f'B vs Bm: {clean(diff)}')
    # P9 numbers for comparison
    P9R = p9['eval_B']['circuits']
    res['p9_reference'] = {k: dict(F=P9R[k]['F'], F_ci=P9R[k]['F_ci'], wz_acc=P9R[k]['wz_acc']) for k in P9R if k in masks}
    json.dump(clean(res), open(FN, 'w'), indent=1); log(f'wrote {FN}')
    plot(res, p9)
    log('P9v DONE')


def plot(res, p9):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), dpi=140)
    a = ax[0]
    for nm_, col in (('B', '#2c7fb8'), ('Bm', '#e67e22')):
        C = res[nm_]['circuits']; ks = sorted([k for k in C if k.startswith('edge_top')], key=lambda k: C[k]['n_edges'])
        x = [C[k]['n_edges'] for k in ks]; f = [C[k]['F'] for k in ks]
        a.plot(x, f, 'o-', ms=3, color=col, label=f'top-k EAP-IG edges, fresh sample, donors {nm_}')
        if nm_ == 'B':
            for k in ('eap_top1075', 'eap_top4000', 'eap_top8000'):
                a.scatter([C[k]['n_edges']], [C[k]['F']], marker='s', color='#27ae60', s=20, label='top-k by plain EAP' if k == 'eap_top1075' else None)
            for k in [k for k in C if k.startswith('rnd')]:
                a.scatter([C[k]['n_edges']], [C[k]['F']], marker='x', color='#999999', s=20, label='k random edges' if k.endswith('1075_0') else None)
    P = p9['eval_B']['circuits']; ks = sorted([k for k in P if k.startswith('edge_top')], key=lambda k: P[k]['n_edges'])
    a.plot([P[k]['n_edges'] for k in ks], [P[k]['F'] for k in ks], 'k--', lw=0.8, label='P9 (evaluation set, files _123)')
    a.axhline(0.8, color='k', ls=':', lw=0.8); a.set_xscale('log'); a.set_xlabel('edges kept (of 21,504)'); a.set_ylabel('faithfulness F')
    a.set_title('(a) F vs edges kept, rest resampled from a W<->Z donor', fontsize=10); a.legend(fontsize=7)
    a = ax[1]
    for nm_, col in (('B', '#2c7fb8'), ('Bm', '#e67e22')):
        C = res[nm_]['circuits']; ks = sorted([k for k in C if k.startswith('node_top')], key=lambda k: int(k[8:]))
        a.plot([int(k[8:]) for k in ks], [C[k]['wz_acc'] * 100 for k in ks], 'o-', ms=3, color=col, label=f'top-k nodes literal, donors {nm_}')
        a.axhline(res[nm_]['unedited_wz_acc'] * 100, color=col, ls='--', lw=0.7)
    a.set_xscale('log'); a.set_xlabel('nodes kept (of 433)'); a.set_ylabel('W vs Z accuracy [%]')
    a.set_title('(b) literal node circuits, fresh sample', fontsize=10); a.legend(fontsize=7)
    fig.suptitle('P9 verification: ParT_full W vs Z faithfulness on WToQQ/ZToQQ_120 (frozen P9 ranking)', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / 'p9_verify.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
