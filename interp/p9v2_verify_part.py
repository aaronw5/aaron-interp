"""P9 verification, part 2 (P9v2): controls P9 and P9v did not run, on fresh JetClass rows, with P9's graph and MLP-unit
grouping frozen (results/part/p9_eap_scores.npz).

Samples (JetClass val_5M, random rows, never first-k):
  T (search set): NPER random rows each of WToQQ_121.root / ZToQQ_121.root, numpy default_rng(6161)
  V (evaluation): NPER random rows each of WToQQ_122.root / ZToQQ_122.root, numpy default_rng(6262)
  donors B (P9's rule: opposite class, one of the 5 nearest in jet pT), default_rng(63) on T, default_rng(64) on V.
1. Attribution stability: EAP-IG (P9's 5 alpha steps) recomputed on T; Spearman of |IG_T| vs |IG_P9| over all edges, overlap of
   the top-1,075 sets; F on V of the top-k edges by IG_T.
2. Stronger circuit search at fixed size (tests "no small circuit"): a soft top-K mask
   m_e = sigmoid((theta_e - tau_K) / T), tau_K = K-th largest theta (detached), T annealed 0.5 -> 0.02, theta initialised at the
   standardised log|IG_P9|, Adam (lr 0.02), loss = -sum_j s_j (d_j(m) - d_j^corr) / sum_j s_j (d_j - d_j^corr) on 4 random
   batches of T per step, OPT_STEPS steps; then the binary top-K edges by theta. K = 1,075 (5 %) and 2,150 (10 %).
   Evaluated on V as edge circuits and as node-induced literal circuits.
3. Anchored random controls for the ACDC literal circuits (89 and 27 nodes, node lists from p9_eap_wz.json):
   anchor = E, every class-stage node (C*) and z coordinate of the circuit, kept as is; the particle-stage nodes (P*B, P*H, P*M)
   replaced by random nodes of the same type and block (NR draws) -> does the particle-stage selection matter beyond the
   read-out?  anchor_only = the anchor with no particle-stage node.  z97rand = E + z97 kept, all other nodes random per type
   and block (NR draws).
Metric F = sum s (d_circ - d_corr) / sum s (d_full - d_corr), d = l_W - l_Z; W/Z accuracy; bootstrap over jets (1,000).
  ./part_job.sh c-p9-verify2 'python -m interp.p9v2_verify_part'  -> $AI_RESULTS/part/p9_verify2.json, p9_verify2.png
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.p9_eap_wz_part import (Graph, Net, multi_eval, Scorer, donors_B, load_sel, batches, ci, clean, log, NB, OUT, VAL,
                                   IGB, ALPHAS)

NPER = int(os.environ.get('P9V_NPER', 2500))
OPT_STEPS = int(os.environ.get('P9V_OPT', 800))
NR = 20
FN = OUT / 'p9_verify2.json'


def main():
    import torch, uproot
    from interp.part_core import kin, build, cat, sub, CL, PRE
    OUT.mkdir(parents=True, exist_ok=True); iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    G = Graph(); sc = np.load(OUT / 'p9_eap_scores.npz'); ig9 = sc['ig']
    assert (sc['W'] == G.W).all() and (sc['R'] == G.R).all()
    p9 = json.load(open(OUT / 'p9_eap_wz.json'))
    res = dict(NPER=NPER, NB=NB, OPT_STEPS=OPT_STEPS, NR=NR, files={})

    def load(tag, seed):
        rng = np.random.default_rng(seed); parts = []
        for c in ('Wqq', 'Zqq'):
            p = VAL / f'{PRE[c]}_{tag}.root'; N = uproot.open(p)['tree'].num_entries; r = np.sort(rng.choice(N, NPER, replace=False))
            (J,), _ = load_sel(p, [r]); parts.append(J); res['files'][f'{c}_{tag}'] = dict(file=p.name, entries=N, first_rows=r[:5].tolist())
        J0 = cat(parts); return build(J0, *kin(J0)), J0['label'].astype(int)
    JT, yT = load(121, 6161); JV, yV = load(122, 6262)
    donT = donors_B(JT['jet_pt'], yT, iW, iZ, np.random.default_rng(63)); donV = donors_B(JV['jet_pt'], yV, iW, iZ, np.random.default_rng(64))
    log(f'T {np.bincount(yT, minlength=10).tolist()}, V {np.bincount(yV, minlength=10).tolist()}')
    net = Net(G); net.set_perm([np.asarray(p) for p in sc['perms']]); dev = net.dev
    tm = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(dev)
    sT = np.where(yT == iW, 1.0, -1.0)
    bT = batches(JT, np.arange(len(yT)), IGB)

    # ---- 1. EAP-IG on T
    t1 = time.time(); igT = np.zeros(G.NE); Dd = 0.0
    for bi in bT:
        (ic, idn), _ = net.prep(sub(JT, bi), sub(JT, donT[bi])); s = torch.from_numpy(sT[bi]).float().to(dev)
        with torch.no_grad():
            od = net.fwd(idn); cz = net.corr_of(od); oc = net.fwd(ic)
        Dd += float((s * ((oc['L'][:, iW] - oc['L'][:, iZ]) - (od['L'][:, iW] - od['L'][:, iZ]))).sum())
        for a in ALPHAS:
            mm = torch.full((G.NE,), a, device=dev, requires_grad=True)
            L = net.fwd(ic, mm, cz)['L']; g, = torch.autograd.grad((s * (L[:, iW] - L[:, iZ])).sum(), mm)
            igT += g.double().cpu().numpy() / len(ALPHAS)
    igT /= len(yT); Dd /= len(yT)
    from scipy.stats import spearmanr
    r9 = np.argsort(-np.abs(ig9), kind='stable'); rT = np.argsort(-np.abs(igT), kind='stable')
    res['ig_stability'] = dict(spearman_abs=float(spearmanr(np.abs(igT), np.abs(ig9)).correlation),
                               overlap_top={str(k): float(len(np.intersect1d(r9[:k], rT[:k])) / k) for k in (10, 100, 1075, 4000)},
                               completeness_T=float(igT.sum() / Dd), top10_T=[f'{G.W[e]}->{G.R[e]} {igT[e] / Dd:+.3f}' for e in rT[:10]],
                               seconds=time.time() - t1)
    log(f'IG on T ({time.time() - t1:.0f} s): {res["ig_stability"]}')

    # ---- 2. soft top-K mask optimisation on T
    l9 = np.log(np.abs(ig9) + 1e-12); th0 = (l9 - l9.mean()) / l9.std()
    opt = {}
    for K in (1075, 2150):
        t1 = time.time(); th = torch.tensor(th0, dtype=torch.float32, device=dev, requires_grad=True)
        ad = torch.optim.Adam([th], lr=0.02); rng = np.random.default_rng(K); hist = []
        for step in range(OPT_STEPS):
            T = 0.5 * (0.02 / 0.5) ** (step / max(1, OPT_STEPS - 1))
            ad.zero_grad(); den = 0.0; cache = []; tot = 0.0
            for b in rng.choice(len(bT), 4, replace=False):
                bi = bT[b]; (ic, idn), _ = net.prep(sub(JT, bi), sub(JT, donT[bi])); s = torch.from_numpy(sT[bi]).float().to(dev)
                with torch.no_grad():
                    od = net.fwd(idn); cz = net.corr_of(od); oc = net.fwd(ic)
                    dco = od['L'][:, iW] - od['L'][:, iZ]; den = den + float((s * (oc['L'][:, iW] - oc['L'][:, iZ] - dco)).sum())
                cache.append((ic, cz, s, dco))
            tau = torch.topk(th.detach(), K).values[-1]
            for ic, cz, s, dco in cache:
                m = torch.sigmoid((th - tau) / T); L = net.fwd(ic, m, cz)['L']
                lo = -(s * (L[:, iW] - L[:, iZ] - dco)).sum() / den; lo.backward(); tot -= float(lo)
            del cache; ad.step(); hist.append(tot)
            if step % 100 == 0: log(f'  opt K {K} step {step}: soft F {hist[-1]:.3f}, T {T:.3f}')
        sel = np.argsort(-th.detach().cpu().numpy(), kind='stable')[:K]
        opt[K] = sel; res[f'opt_K{K}'] = dict(soft_F_last100=float(np.mean(hist[-100:])), overlap_with_ig9_topK=float(len(np.intersect1d(sel, r9[:K])) / K),
                                              seconds=time.time() - t1, hist=hist[::10])
        log(f'opt K {K} ({time.time() - t1:.0f} s): soft F (last 100) {res[f"opt_K{K}"]["soft_F_last100"]:.3f}, overlap with IG top-K {res[f"opt_K{K}"]["overlap_with_ig9_topK"]:.2f}')

    # ---- masks
    masks = {}
    for k in (1075, 2150, 4000, 6300, 8000):
        a = np.zeros(G.NE); a[r9[:k]] = 1; masks[f'ig9_top{k}'] = tm(a)
    for k in (1075, 4000, 8000):
        a = np.zeros(G.NE); a[rT[:k]] = 1; masks[f'igT_top{k}'] = tm(a)
    for K, sel in opt.items():
        a = np.zeros(G.NE); a[sel] = 1; masks[f'opt{K}_edges'] = tm(a)
        nodes = sorted({G.W[e] for e in sel} | {G.R[e] for e in sel} - {'OUT', 'LOGIT'}, key=G.node_order)
        masks[f'opt{K}_literal'] = tm(G.node_mask(nodes)); res[f'opt_K{K}']['n_nodes'] = len(nodes)
        res[f'opt_K{K}']['literal_edges'] = int(G.node_mask(nodes).sum())
    ncats = {}
    for nd in G.nodes: ncats.setdefault(G.ncat(nd), []).append(nd)
    rr = np.random.default_rng(99); acdc = {t: p9['circuits'][f'acdc{t}_edges']['nodes'] for t in p9['TAUS']}
    for t in (0.001, 0.003):
        nodes = acdc[t]; n_ = len(nodes); masks[f'acdc{n_}_literal'] = tm(G.node_mask(nodes))
        anchor = [nd for nd in nodes if nd == 'E' or nd[0] in 'Cz']; partn = [nd for nd in nodes if nd[0] == 'P']
        masks[f'anchor{n_}_only'] = tm(G.node_mask(anchor))
        for r_ in range(NR):
            new = []
            for c_ in sorted({G.ncat(nd) for nd in partn}):
                new += list(rr.choice(ncats[c_], sum(G.ncat(nd) == c_ for nd in partn), replace=False))
            masks[f'anchrand{n_}_{r_}'] = tm(G.node_mask(anchor + new))
        if t == 0.001:
            rest = [nd for nd in nodes if nd not in ('E', 'z97')]
            for r_ in range(NR):
                new = []
                for c_ in sorted({G.ncat(nd) for nd in rest}):
                    pool = [x for x in ncats[c_] if x != 'z97']
                    new += list(rr.choice(pool, sum(G.ncat(nd) == c_ for nd in rest), replace=False))
                masks[f'z97rand{n_}_{r_}'] = tm(G.node_mask(['E', 'z97'] + new))
    masks['all_ablated'] = tm(np.zeros(G.NE)); masks['full'] = tm(np.ones(G.NE))
    res['n_edges'] = {k: int(v.sum().item()) for k, v in masks.items()}

    # ---- evaluation on V (and opt circuits on T)
    t1 = time.time(); LV = multi_eval(net, JV, np.arange(len(yV)), donV, masks, check=True)
    S = Scorer(yV, iW, iZ, LV['clean'], LV['corr'], np.random.default_rng(5)); R = {k: S.summary(LV[k]) for k in masks}
    res['V'] = dict(exactness=LV['_check'], unedited_wz_acc=float(S.ok0.mean()), corr_wz_acc=float((np.where(S.dco > 0, iW, iZ) == yV).mean()),
                    seconds=time.time() - t1,
                    circuits={k: dict(F=v['F'], F_ci=v['F_ci'], wz_acc=v['wz_acc'], wz_acc_ci=v['wz_acc_ci'], n_edges=res['n_edges'][k])
                              for k, v in R.items() if 'rand' not in k})
    for K in opt:
        d = R[f'opt{K}_edges']['_fb'] - R[f'ig9_top{K}']['_fb']
        res[f'opt_K{K}']['dF_vs_ig9_V'] = dict(dF=R[f'opt{K}_edges']['F'] - R[f'ig9_top{K}']['F'], ci=ci(d),
                                               p_two_sided=float(max(1 / NB, min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))),
                                               p_F_ge_0p8=float(max(1 / NB, (R[f'opt{K}_edges']['_fb'] >= 0.8).mean())))
    ctl = {}
    for pre, ref in (('anchrand89', 'acdc89_literal'), ('anchrand27', 'acdc27_literal'), ('z97rand89', 'acdc89_literal')):
        Fr = np.array([R[f'{pre}_{r_}']['F'] for r_ in range(NR)]); Ar = np.array([R[f'{pre}_{r_}']['wz_acc'] for r_ in range(NR)])
        ctl[pre] = dict(reference=ref, F_ref=R[ref]['F'], F_mean=float(Fr.mean()), F_ci_draws=ci(Fr), F_max=float(Fr.max()), acc_mean=float(Ar.mean()),
                        acc_max=float(Ar.max()), F_values=Fr, p_perm=float((1 + (Fr >= R[ref]['F']).sum()) / (1 + NR)))
    res['controls'] = ctl
    log(f'V ({time.time() - t1:.0f} s): exactness {LV["_check"]}; unedited W/Z {res["V"]["unedited_wz_acc"]:.4f}; all resampled {res["V"]["corr_wz_acc"]:.4f}')
    for k in masks:
        if 'rand' not in k: log(f'  {k:22s} edges {res["n_edges"][k]:6d}  F {R[k]["F"]:+.3f} {np.round(R[k]["F_ci"], 3).tolist()}  W/Z {R[k]["wz_acc"]:.4f}')
    log(f'controls: {clean({k: {kk: vv for kk, vv in v.items() if kk != "F_values"} for k, v in ctl.items()})}')
    log(f'opt: {clean({K: {kk: vv for kk, vv in res[f"opt_K{K}"].items() if kk != "hist"} for K in opt})}')
    om = {k: masks[k] for k in masks if k.startswith('opt') or k in ('ig9_top1075', 'ig9_top2150')}
    LT = multi_eval(net, JT, np.arange(0, len(yT), 2), donT, om)
    ST = Scorer(yT[::2], iW, iZ, LT['clean'], LT['corr'], np.random.default_rng(6))
    res['T_check'] = {k: ST.summary(LT[k])['F'] for k in om}; log(f'opt circuits on T[::2]: {res["T_check"]}')
    json.dump(clean(res), open(FN, 'w'), indent=1); log(f'wrote {FN}')
    plot(res, p9); log('P9v2 DONE')


def plot(res, p9):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), dpi=140); C = res['V']['circuits']
    a = ax[0]; P = p9['eval_B']['circuits']; ks = sorted([k for k in P if k.startswith('edge_top')], key=lambda k: P[k]['n_edges'])
    a.plot([P[k]['n_edges'] for k in ks], [P[k]['F'] for k in ks], 'k--', lw=0.8, label='P9 curve (files _123)')
    for pre, col, lb in (('ig9_top', '#2c7fb8', 'top-k by P9 EAP-IG, files _122'), ('igT_top', '#27ae60', 'top-k by EAP-IG recomputed on _121')):
        kk = sorted([k for k in C if k.startswith(pre)], key=lambda k: C[k]['n_edges'])
        a.plot([C[k]['n_edges'] for k in kk], [C[k]['F'] for k in kk], 'o', color=col, ms=5, label=lb)
    for K in (1075, 2150):
        k = f'opt{K}_edges'
        if k in C: a.scatter([K], [C[k]['F']], marker='*', s=90, color='#c0392b', zorder=5, label='optimised top-K mask (trained on _121)' if K == 1075 else None)
    a.axhline(0.8, color='k', ls=':', lw=0.8); a.axvline(1075, color='k', ls=':', lw=0.6); a.set_xscale('log')
    a.set_xlabel('edges kept (of 21,504)'); a.set_ylabel('faithfulness F'); a.set_title('(a) F vs edges kept, W<->Z donors', fontsize=10); a.legend(fontsize=7)
    a = ax[1]; ctl = res['controls']; labels = []
    for i, (pre, lb) in enumerate((('anchrand89', '89-node ACDC:\nparticle nodes random'), ('z97rand89', '89-node ACDC:\nall random but E, n98'),
                                   ('anchrand27', '27-node ACDC:\nparticle nodes random'))):
        v = np.array(ctl[pre]['F_values'], float); a.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, s=8, color='#999999')
        a.scatter([i], [ctl[pre]['F_ref']], marker='D', color='#c0392b', s=40, zorder=4); labels.append(lb)
        an = 'anchor89_only' if '89' in pre else 'anchor27_only'
        if pre != 'z97rand89': a.scatter([i], [C[an]['F']], marker='_', color='k', s=300, zorder=4)
    a.set_xticks(range(3)); a.set_xticklabels(labels, fontsize=8); a.set_ylabel('faithfulness F')
    a.set_title('(b) ACDC literal circuit (red) vs random replacements (grey);\nblack bar = read-out anchor alone', fontsize=10)
    fig.suptitle('P9 verification 2: ParT_full W vs Z circuits, files _121/_122', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / 'p9_verify2.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
