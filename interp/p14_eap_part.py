"""P14 test 4 — P9's whole-network EAP-IG on the planted-label ParTs (2 outputs, kin inputs) and the released ParT_kin.

Reuses interp/p9_eap_wz_part.py unchanged: Graph (433 writer nodes, 21,504 edges: embedding, pair-bias channels, particle heads,
particle-MLP groups of 32 units, class heads, class-MLP groups, class-token coordinates), Net (exact decomposition of weaver's
ParticleTransformer, resample ablation of edges), batches, donors_B. Net is built around a given model instead of ParT_full
(ParTNetwork is swapped for a shim during Net.__init__), and prep() feeds the model's own inputs (kin set; arm c position channels
zeroed).
Metric: d = logit_1 - logit_0 (planted arms; released_kin: logit_W - logit_Z), s = +1 for label 1 (W), -1 for label 0 (Z).
Donors: P9's corruption B: for each jet, one of the 5 jets nearest in jet pT among the jets with the other label.
F = sum s (d_circ - d_corr) / sum s (d_full - d_corr). Steps as P9: unit attribution -> MLP grouping; EAP-IG (alphas 0.1-0.9)
on the discovery jets; on the evaluation jets F and accuracy for the top-k edges and the node-induced literal circuits of the top-k
nodes (every edge between kept nodes kept, everything else resampled); size at F = 0.8 by log-linear interpolation; the smallest
literal node circuit with F >= 0.8: own-label accuracy, 2 x 2 confusion, call by true class (QCD / Z / W), McNemar vs the full
model, F and accuracy of NRAND random node circuits with the same node-category counts (permutation p).
"""
import math
import numpy as np


def run_eap(M, Jb, lab, y3, idx_disc, idx_eval, rng, nrand=20, log=print):
    import torch
    import interp.part_core as pc
    from interp import p9_eap_wz_part as p9
    from interp.part_core import sub, inputs, cat
    G = p9.Graph()

    class Shim:
        def __init__(s, n):
            s.model, s.device = M.m, M.dev

    class PNet(p9.Net):
        def __init__(self):
            orig = pc.ParTNetwork; pc.ParTNetwork = Shim
            try:
                super().__init__(G)
            finally:
                pc.ParTNetwork = orig

        def prep(self, Jc, Jd):
            t = self.t; x, v, mk = inputs(cat([Jc, Jd]), M.kind); x = M.x(x); n = len(Jc['mask'])
            x, v, mk = (t.from_numpy(np.ascontiguousarray(a)).to(self.dev) for a in (x, v, mk)); mb = mk.bool()
            with t.no_grad():
                Emb = self.m.embed(x).masked_fill(~mb.permute(2, 0, 1), 0).permute(1, 0, 2).contiguous()
                P = v.shape[-1]; U = self.m.pair_embed(v).view(-1, p9.H, P, P)
            pad = ~mb[:, 0]
            return (dict(E=Emb[:n], U=U[:n], pad=pad[:n]), dict(E=Emb[n:], U=U[n:], pad=pad[n:])), (x, v, mk)

    net = PNet(); i1, i0 = (1, 0) if M.arm else (pc.CL.index('Wqq'), pc.CL.index('Zqq'))
    lab = np.asarray(lab).astype(int); yy = np.where(lab == 1, 1, 0)
    res = dict(n_disc=len(idx_disc), n_eval=len(idx_eval), n_edges=G.NE, n_nodes=len(G.nodes), nrand=nrand)
    # donors (corruption B) within each set, by label
    def donors(idx):
        d = p9.donors_B(Jb['jet_pt'][idx], yy[idx], 1, 0, rng); return idx[d]
    donD = np.zeros(len(yy), int); donE = np.zeros(len(yy), int); donD[idx_disc] = donors(idx_disc); donE[idx_eval] = donors(idx_eval)
    s_all = np.where(yy == 1, 1.0, -1.0)
    dfun = lambda L: L[:, i1] - L[:, i0]
    # step 0: unit attribution -> MLP groups
    C = p9.C; sc_u = [np.zeros(4 * C) for _ in range(10)]
    uidx = np.sort(rng.choice(idx_disc, min(500, len(idx_disc)), replace=False))
    for bi in p9.batches(Jb, uidx, p9.IGB):
        (ic, idn), _ = net.prep(sub(Jb, bi), sub(Jb, donD[bi]))
        with torch.no_grad(): hd = net.fwd(idn, rec=True)['h']
        probe = [torch.zeros(len(bi), ic['E'].shape[1], 4 * C, device=net.dev, requires_grad=True) for _ in range(8)] + \
                [torch.zeros(len(bi), 4 * C, device=net.dev, requires_grad=True) for _ in range(2)]
        o = net.fwd(ic, probe=probe); s = torch.from_numpy(s_all[bi]).float().to(net.dev)
        gr = torch.autograd.grad((s * dfun(o['L'])).sum(), probe)
        for k in range(10):
            sc_u[k] += ((hd[k] - o['h'][k].detach()) * gr[k]).sum(tuple(range(gr[k].dim() - 1))).cpu().numpy()
    net.set_perm([np.argsort(-np.abs(s_), kind='stable') for s_ in sc_u])
    # exactness
    with torch.no_grad():
        bi = p9.batches(Jb, idx_eval[:200], 48)[0]; (ic, idn), raw = net.prep(sub(Jb, bi), sub(Jb, donE[bi]))
        Lm = net.m(*(a[:len(bi)] for a in raw)).float(); oc = net.fwd(ic); od = net.fwd(idn); cz = net.corr_of(od)
        one = torch.ones(G.NE, device=net.dev); zero = torch.zeros(G.NE, device=net.dev)
        res['exactness'] = dict(clean_vs_model=float((Lm - oc['L']).abs().max()), ones_vs_clean=float((net.fwd(ic, one, cz)['L'] - oc['L']).abs().max()),
                                zeros_vs_donor=float((net.fwd(ic, zero, cz)['L'] - od['L']).abs().max()))
    log(f'  EAP {M.name}: exactness {res["exactness"]}')
    # step 1: EAP-IG
    ig = np.zeros(G.NE); Dcl = Dco = 0.0
    for bi in p9.batches(Jb, idx_disc, p9.IGB):
        (ic, idn), _ = net.prep(sub(Jb, bi), sub(Jb, donD[bi])); s = torch.from_numpy(s_all[bi]).float().to(net.dev)
        with torch.no_grad():
            od = net.fwd(idn); cz = net.corr_of(od); oc = net.fwd(ic)
        Dcl += float((s * dfun(oc['L'])).sum()); Dco += float((s * dfun(od['L'])).sum())
        for a in p9.ALPHAS:
            mm = torch.full((G.NE,), a, device=net.dev, requires_grad=True)
            g, = torch.autograd.grad((s * dfun(net.fwd(ic, mm, cz)['L'])).sum(), mm); ig += g.double().cpu().numpy() / len(p9.ALPHAS)
    ig /= len(idx_disc); dtot = (Dcl - Dco) / len(idx_disc)
    res['eap'] = dict(D_per_jet=dtot, sum_ig=float(ig.sum()), completeness_ratio=float(ig.sum() / dtot) if dtot else None)
    rank = np.argsort(-np.abs(ig), kind='stable'); nscore = {nd: 0.0 for nd in G.nodes}
    for e in range(G.NE): nscore[G.W[e]] += ig[e]
    nodes_ranked = sorted(G.nodes, key=lambda nd: -abs(nscore[nd]))
    res['top_nodes'] = [dict(node=nd, share=nscore[nd] / dtot) for nd in nodes_ranked[:20]]
    cats = {}
    for nd in nodes_ranked[:40]: cats[G.ncat(nd)] = cats.get(G.ncat(nd), 0) + 1
    res['top40_node_categories'] = cats
    log(f'  EAP {M.name}: D per jet {dtot:.3f}, sum IG ratio {res["eap"]["completeness_ratio"]}; top nodes ' +
        ', '.join(f'{d["node"]} {d["share"]:+.3f}' for d in res['top_nodes'][:10]))
    # evaluation
    tm = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(net.dev)
    EK = sorted(set([10, 30, 100, 300, 1000, 2000, 4000, 8000, 15000, G.NE])); NK = [2, 4, 8, 16, 32, 64, 128, 256, len(G.nodes)]
    masks = {f'edge_top{k}': tm(np.isin(np.arange(G.NE), rank[:k])) for k in EK}
    masks.update({f'node_top{k}': tm(G.node_mask(nodes_ranked[:k])) for k in NK})

    def evaluate(masks, idx):
        pos = {int(i): r for r, i in enumerate(idx)}; out = {k: np.zeros((len(idx), 2)) for k in list(masks) + ['clean', 'corr']}
        with torch.no_grad():
            for bi in p9.batches(Jb, idx, p9.BATCH):
                (ic, idn), _ = net.prep(sub(Jb, bi), sub(Jb, donE[bi])); rows = [pos[int(i)] for i in bi]
                oc = net.fwd(ic); od = net.fwd(idn); cz = net.corr_of(od)
                pick = lambda L: L[:, [i0, i1]].double().cpu().numpy()
                out['clean'][rows] = pick(oc['L']); out['corr'][rows] = pick(od['L'])
                for k, mk in masks.items(): out[k][rows] = pick(net.fwd(ic, mk, cz)['L'])
        return out
    LE = evaluate(masks, idx_eval); s = s_all[idx_eval]; yl = yy[idx_eval]; Wb = p9.boot_W(len(idx_eval), 1000, rng)
    dcl = LE['clean'][:, 1] - LE['clean'][:, 0]; dco = LE['corr'][:, 1] - LE['corr'][:, 0]; den = s * (dcl - dco)
    ok0 = (dcl > 0).astype(int) == yl

    def summ(L):
        d = L[:, 1] - L[:, 0]; num = s * (d - dco); call = (d > 0).astype(int); ok = call == yl
        return dict(F=float(num.sum() / den.sum()), F_ci=p9.ci((Wb @ num) / (Wb @ den)), acc=float(ok.mean()), acc_ci=p9.ci((Wb @ ok) / Wb.sum(1)),
                    agree_full=float((call == (dcl > 0)).mean()), mcnemar_vs_full=p9.mcnemar(ok0, ok),
                    confusion_label_call=[[int(((yl == i) & (call == j)).sum()) for j in (0, 1)] for i in (0, 1)],
                    call1_by_class={c: float(call[y3[idx_eval] == ci_].mean()) for ci_, c in enumerate(('QCD', 'Z', 'W')) if (y3[idx_eval] == ci_).any()})
    R = {k: summ(LE[k]) for k in masks}
    res['unedited_acc'] = float(ok0.mean()); res['corr_acc'] = float(((dco > 0).astype(int) == yl).mean()); res['curves'] = R

    def interp_size(keys, sizes):
        F = [R[k]['F'] for k in keys]
        for j in range(1, len(F)):
            if F[j] >= 0.8 > F[j - 1]:
                a = (0.8 - F[j - 1]) / (F[j] - F[j - 1]); return float(math.exp(math.log(sizes[j - 1]) + a * (math.log(sizes[j]) - math.log(sizes[j - 1]))))
        return float(sizes[0]) if F and F[0] >= 0.8 else None
    res['edges_at_F0p8'] = interp_size([f'edge_top{k}' for k in EK], EK); res['edge_frac_at_F0p8'] = res['edges_at_F0p8'] / G.NE if res['edges_at_F0p8'] else None
    res['nodes_at_F0p8'] = interp_size([f'node_top{k}' for k in NK], NK)
    kmin = next((k for k in NK if R[f'node_top{k}']['F'] >= 0.8), None); res['literal_circuit'] = None
    if kmin is not None:
        nodes = nodes_ranked[:kmin]; lit = dict(n_nodes=kmin, n_edges=int(G.node_mask(nodes).sum()), nodes=nodes if kmin <= 64 else None, **R[f'node_top{kmin}'])
        ncats = {}
        for nd in G.nodes: ncats.setdefault(G.ncat(nd), []).append(nd)
        rm = {}
        for r_ in range(nrand):
            pick = []
            for c_ in sorted({G.ncat(nd) for nd in nodes}):
                pick += list(rng.choice(ncats[c_], sum(G.ncat(nd) == c_ for nd in nodes), replace=False))
            rm[f'rand{r_}'] = tm(G.node_mask(pick))
        LR = evaluate(rm, idx_eval); Fr = np.array([summ(LR[k])['F'] for k in rm]); Ar = np.array([summ(LR[k])['acc'] for k in rm])
        lit.update(random_F_mean=float(Fr.mean()), random_F_ci=p9.ci(Fr), random_acc_mean=float(Ar.mean()), random_acc_max=float(Ar.max()),
                   p_perm_F=float((1 + (Fr >= lit['F']).sum()) / (1 + nrand)), p_perm_acc=float((1 + (Ar >= lit['acc']).sum()) / (1 + nrand)))
        res['literal_circuit'] = lit
    log(f'  EAP {M.name}: unedited acc {res["unedited_acc"]:.4f}; edges at F=0.8 {res["edges_at_F0p8"]} ({res["edge_frac_at_F0p8"]}); nodes at F=0.8 {res["nodes_at_F0p8"]}; '
        + (f'literal {kmin} nodes: F {res["literal_circuit"]["F"]:.3f}, acc {res["literal_circuit"]["acc"]:.4f} (random F {res["literal_circuit"]["random_F_mean"]:.3f}, acc {res["literal_circuit"]["random_acc_mean"]:.4f})'
           if res['literal_circuit'] else 'no node circuit reaches F = 0.8'))
    del net; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return res
