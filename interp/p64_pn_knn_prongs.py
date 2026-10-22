"""P64 - ParticleNet's feature-space kNN graphs: do the EdgeConv-2/3 neighbours stay inside one prong, and does cutting the
cross-prong (or within-prong) edges change the call? (loop C, round C14, exploratory; claim A8 / C5 / new C24)

Models (frozen): ParticleNet_full (17 inputs) and ParticleNet_kin (7 inputs), official JetClass weights
(/j-jepa-vol/aaron-interp/models/particlenet), network code interp/pn_net.py (weaver-core 0.4.17 copy); own forward checked
against weaver's ParticleNet.forward (|dlogit| <= 1e-5). Controls (graph purity only): untrained ParticleNet_full, two inits
(PyTorch default init, torch seeds 6464, 6465; BatchNorm running statistics at their defaults 0 / 1), and channel-permuted
ParticleNet_full (every weight matrix of every 1x1 conv / linear layer has its input-channel axis permuted by an independent
random permutation, torch seed 6466; BatchNorm kept): same weight values, learned wiring between layers destroyed.

EdgeConv layer l (EC1, EC2, EC3; k = 16 each): EC1 links each particle to its 16 nearest particles in (deta, dphi); EC2 and
EC3 rebuild the graph as the 16 nearest particles in the previous layer's output features (64-d, 128-d).

Sample: val_5M files *_124 of QCD (ZJetsToNuNu), Z (ZToQQ), W (WToQQ), top (TTBar, all-hadronic): 3,000 random rows per class,
numpy default_rng(6464) (the plan named *_126, which does not exist in val_5M: files 120-124 only). N 12,000. Unedited 10-class
accuracy reported on this sample and on the P1 reference sample (test_20M *_100 rows 3,000-4,000 per class, N 10,000).

Prong labels: exclusive-kT clustering of the jet's real particles (E-scheme recombination of four-vectors, pairwise distance
d_ij = min(pT_i^2, pT_j^2) * ((y_i - y_j)^2 + dphi_ij^2), no beam distance, so every particle ends in one subjet), stopped at
N = 2 subjets (W, Z) or N = 3 (top, QCD) (primary); the other N is stored as a robustness row. Jets with fewer real
particles than N are excluded from purity (counted).

Observation: edge purity of particle i in layer l = (number of i's real neighbours, self excluded, in i's own subjet) / (number
of i's real neighbours). Jet purity = mean over real particles (unweighted) or pT-weighted mean. Compared with
  (a) random: the expected purity of a degree-matched uniform random graph, (n_s(i) - 1) / (n - 1) (n_s = size of i's subjet,
      n = real particles), computed exactly;
  (b) EC1 (the eta-phi graph of the same network);
  (c) the same layer's graph in untrained / channel-permuted ParticleNet_full.
95 % bootstrap over jets (2,000 resamples), paired bootstrap p for EC_l - EC1 and EC_l - random. Also: fraction of EC_l's
neighbours that are also EC1 neighbours (overlap).

Interventions on the real network (all weights fixed; graphs replaced one layer set at a time: EC2, EC3, both):
  R1 within-prong: every cross-prong neighbour replaced by the nearest same-prong particle (in the layer's own feature
     distance) not yet in the list (= the 16 nearest same-prong particles; original neighbours kept where the prong has too
     few particles);
  R2 cross-prong: every within-prong neighbour replaced by the nearest cross-prong particle (likewise);
  R3r1 / R3r2 control: for each particle, the same number of edges as R1 / R2 replaced, prong-blind: that many of its
     original neighbours (random choice) dropped and that many random other particles of the jet added;
  R3n1 / R3n2 control: as R3r but the added particles are the nearest non-neighbours in feature space (prong-blind and
     distance-matched to R1 / R2's "nearest" rule);
  R4: EC2 / EC3 graph replaced by the EC1 (eta-phi) graph of the same jet.
  Metrics vs the unedited network: 10-class accuracy and confusion (rows: the 4 true classes); calls changed: argmax over 10
  classes (all jets), W vs Z (sign of l_W - l_Z on true W and Z jets), top vs QCD (sign of l_t - l_QCD on true top and QCD
  jets), W vs QCD (sign of l_W - l_QCD on true W and QCD jets); 95 % bootstrap; McNemar on the per-jet "call changed"
  indicator of R1 vs R3r1 / R3n1 and R2 vs R3r2 / R3n2; Holm over arms x layer sets x call types x models.
Physics edit (true W jets with 2 subjets): each prong moved rigidly along the line through the jet's pT centroid so that the
  prong centroids' distance from the pT centroid is scaled by s in {0.6, 0.8, 1.25, 1.5} (pT centroid fixed); then every
  particle's pT and mass scaled by lambda = m0 / m' so that the jet's invariant mass is exactly unchanged (jet pT scales by
  lambda); plus a 90-degree rotation of all particles about the pT centroid (separation and mass fixed). Response of the
  EC1 / EC2 / EC3 purity (subjet labels of the unedited jet) and of the W-vs-Z score.
  ./sync.sh && ./part_job.sh c-p64-run 'python -m interp.p64_pn_knn_prongs --smoke && python -m interp.p64_pn_knn_prongs'
  -> $AI_RESULTS/particlenet/p64_pn_knn[_smoke].json, p64_pn_knn[_smoke].png;  --plot redraws the figure from the json
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SMOKE = '--smoke' in sys.argv
TAG = '_smoke' if SMOKE else ''
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'particlenet'
FN, FIG = RES / f'p64_pn_knn{TAG}.json', RES / f'p64_pn_knn{TAG}.png'
PNDIR = Path('/j-jepa-vol/aaron-interp/models/particlenet')
VAL = Path('/j-jepa-vol/JetClass/Pythia/val_5M')
CLS = (('QCD', 0, 3), ('Zqq', 6, 2), ('Wqq', 7, 2), ('Tbqq', 8, 3))      # name, label index, primary number of subjets
NPC = 60 if SMOKE else 3000
NP1 = 30 if SMOKE else 1000
NB = 200 if SMOKE else 2000
BS = 250
iQ, iZ, iW, iT = 0, 6, 7, 8
LAYERSETS = dict(ec2=(1,), ec3=(2,), both=(1, 2))
ARMS = ('R1', 'R2', 'R3r1', 'R3r2', 'R3n1', 'R3n2', 'R4')
SEPS = (0.6, 0.8, 1.0, 1.25, 1.5)
KEEP = ['px', 'py', 'pz', 'energy', 'deta', 'dphi', 'd0val', 'd0err', 'dzval', 'dzerr', 'charge', 'isChargedHadron', 'isNeutralHadron',
        'isPhoton', 'isElectron', 'isMuon', 'mask', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'label']
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ data
def load_sample():
    from interp.part_core import read_root, PRE, cat, build, kin
    import uproot
    rng = np.random.default_rng(6464); parts_all, info = [], []
    for c, lab, _ in CLS:
        path = VAL / f'{PRE[c]}_124.root'; N = uproot.open(path)['tree'].num_entries
        if SMOKE: N = min(N, 20000)
        rows = np.sort(rng.choice(N, NPC, replace=False)); parts = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r):
                J = read_root(path, start=s, stop=min(s + 25000, N)); parts.append({k: J[k][r - s] for k in KEEP})
        J = cat(parts); assert (J['label'] == lab).all(), (c, np.unique(J['label'])); parts_all.append(J)
        info.append(dict(file=path.name, rows_in_file=int(N), n=int(len(rows))))
        log(f'  {path.name}: {len(rows)} random rows of {N}')
    J = cat(parts_all); return build(J, *kin(J)), info


def excl_kt(J, dev):
    """exclusive-kT subjet labels (pairwise kT distance only, E-scheme) of every real particle: lab2, lab3 (n, 128) int,
    -1 for padded slots and for jets with fewer real particles than the number of subjets"""
    import torch
    n = len(J['label']); out2 = np.full((n, 128), -1, np.int64); out3 = np.full((n, 128), -1, np.int64)
    for s0 in range(0, n, 500):
        sl = slice(s0, min(n, s0 + 500)); m = torch.from_numpy(J['mask'][sl] > 0).to(dev)
        p = torch.stack([torch.from_numpy(J[k][sl].astype(np.float64)) for k in ('px', 'py', 'pz', 'energy')], -1).to(dev) * m[..., None]
        B, P, _ = p.shape; act = m.clone(); lab = torch.arange(P, device=dev).expand(B, P).clone(); nact = act.sum(1)
        eye = torch.eye(P, dtype=torch.bool, device=dev); lab3 = lab.clone(); ar = torch.arange(B, device=dev)
        while True:
            lab3 = torch.where((nact == 3)[:, None], lab, lab3)
            need = nact > 2
            if not bool(need.any()): break
            pt2 = p[..., 0] ** 2 + p[..., 1] ** 2; E, pz = p[..., 3], p[..., 2]
            y = 0.5 * torch.log((E + pz).clamp(min=1e-12) / (E - pz).clamp(min=1e-12)); ph = torch.atan2(p[..., 1], p[..., 0])
            dy = y[:, :, None] - y[:, None, :]; dp = (ph[:, :, None] - ph[:, None, :] + math.pi) % (2 * math.pi) - math.pi
            d = torch.minimum(pt2[:, :, None], pt2[:, None, :]) * (dy ** 2 + dp ** 2)
            ok = act[:, :, None] & act[:, None, :] & ~eye
            d = torch.where(ok, d, torch.full_like(d, float('inf'))).view(B, -1); a = d.argmin(1); i, j = a // P, a % P
            i = torch.where(need, i, torch.zeros_like(i)); j = torch.where(need, j, torch.zeros_like(j))
            upd = need[:, None]
            p[ar, i] = torch.where(upd, p[ar, i] + p[ar, j], p[ar, i]); p[ar, j] = torch.where(upd, torch.zeros_like(p[ar, j]), p[ar, j])
            act[ar, j] = torch.where(need, torch.zeros_like(need), act[ar, j])
            lab = torch.where(upd & (lab == j[:, None]), i[:, None], lab); nact = nact - need.long()
        mm = m.cpu().numpy(); L2, L3 = lab.cpu().numpy(), lab3.cpu().numpy(); nr = mm.sum(1)
        for b in range(B):
            r = mm[b]
            if nr[b] >= 2: out2[s0 + b, r] = np.unique(L2[b, r], return_inverse=True)[1]
            if nr[b] >= 3: out3[s0 + b, r] = np.unique(L3[b, r], return_inverse=True)[1]
    return out2, out3


# ------------------------------------------------------------------------------------------------ network
def tensors(J, n, dev, lab):
    from interp.p50_pn_edge_function import tensors as t50
    import torch
    B = t50(J, n, dev); B['lab'] = torch.from_numpy(lab).to(dev)
    pt = np.hypot(J['px'], J['py']) * (J['mask'] > 0); B['pt'] = torch.from_numpy(pt.astype(np.float32)).to(dev); return B


def rank_rows(v):
    import torch
    return torch.argsort(torch.argsort(v, -1), -1)


class Graph:
    """graph rewiring for one layer: pts (N,C,P) the layer's kNN coordinates (with the 1e9 padded shift), idx (N,P,k)"""

    def __init__(self, pts, idx, B, gen):
        import torch
        N, C, P = pts.shape; self.k = idx.shape[-1]; self.P = P; dev = pts.device; self.gen = gen
        x = pts.double(); xx = (x ** 2).sum(1); dist = xx[:, :, None] + xx[:, None, :] - 2 * torch.einsum('ncp,ncq->npq', x, x)
        self.rank = rank_rows(dist); real = B['m'][:, 0] > 0; self.eye = torch.eye(P, dtype=torch.bool, device=dev)[None]
        self.realj = real[:, None, :] & ~self.eye; self.padj = ~real[:, None, :] & ~self.eye
        self.inl = torch.zeros(N, P, P, dtype=torch.bool, device=dev).scatter_(-1, idx, True) & self.realj
        lab = B['lab']; valid = (lab >= 0)
        self.same = (lab[:, :, None] == lab[:, None, :]) & valid[:, :, None] & valid[:, None, :] & self.realj
        self.cross = ~(lab[:, :, None] == lab[:, None, :]) & valid[:, :, None] & valid[:, None, :] & self.realj

    def from_tier(self, tier):
        import torch
        tier = torch.where(self.eye, torch.full_like(tier, 3), tier)
        return (tier * (4 * self.P) + self.rank).topk(self.k, -1, largest=False).indices

    def nrep(self, new):
        import torch
        newin = torch.zeros_like(self.inl).scatter_(-1, new, True) & self.realj
        return (self.inl & ~newin).sum(-1)

    def allowed(self, A):
        import torch
        tier = torch.full_like(self.rank, 2); tier = torch.where(self.realj & ~A, torch.ones_like(tier), tier)
        tier = torch.where(A, torch.zeros_like(tier), tier); new = self.from_tier(tier); return new, self.nrep(new)

    def blind(self, m, nearest):
        import torch
        r1 = torch.rand(self.rank.shape, generator=self.gen, device=self.rank.device)
        o1 = rank_rows(torch.where(self.inl, r1, torch.full_like(r1, 2.0)))
        cand = self.realj & ~self.inl
        v2 = self.rank.float() if nearest else torch.rand(self.rank.shape, generator=self.gen, device=self.rank.device)
        o2 = rank_rows(torch.where(cand, v2, torch.full_like(v2, 1e9))); mm = torch.minimum(m, cand.sum(-1))[..., None]
        sel = (self.inl & (o1 >= mm)) | (cand & (o2 < mm))
        tier = torch.where(sel, torch.zeros_like(self.rank), torch.where(self.padj, torch.ones_like(self.rank), torch.full_like(self.rank, 2)))
        new = self.from_tier(tier); return new, self.nrep(new)


def edit_graph(arm, pts, idx, idx1, B, gen):
    if arm == 'R4':
        G = Graph(pts, idx, B, gen); return idx1, G.nrep(idx1)
    G = Graph(pts, idx, B, gen)
    if arm == 'R1': return G.allowed(G.same)
    if arm == 'R2': return G.allowed(G.cross)
    _, m = G.allowed(G.same if arm.endswith('1') else G.cross)
    return G.blind(m, nearest=arm.startswith('R3n'))


def run(net, B, arm=None, layers=(), rec=False, gen=None):
    """own forward of ParticleNet (use_fusion False). arm: graph rewiring applied in `layers` (1 = EC2, 2 = EC3).
    rec: also return the kNN index of every layer. Returns logits, [idx per layer], [replaced edges per edited layer]"""
    import torch
    from interp import pn_net
    md = net; mask = B['m']; shift = (mask == 0) * 1e9; counts = torch.clamp(mask.sum(-1), min=1.0)
    fts = md.bn_fts(B['x']) * mask; idxs, reps = [], []
    for li, conv in enumerate(md.edge_convs):
        pts = (B['p'] if li == 0 else fts) + shift; idx = pn_net.knn(pts, conv.k)
        if li == 0: idx1 = idx
        if arm is not None and li in layers:
            idx, nr = edit_graph(arm, pts, idx, idx1, B, gen); reps.append(nr)
        if rec: idxs.append(idx)
        msg = pn_net.get_graph_feature_v1(fts, conv.k, idx)
        for c2, b2, act in zip(conv.convs, conv.bns, conv.acts): msg = act(b2(c2(msg)))
        sc = conv.sc_bn(conv.sc(fts)) if conv.sc is not None else fts
        fts = conv.sc_act(sc + msg.mean(-1)) * mask
    return md.fc(fts.sum(-1) / counts), idxs, reps


def purity(idx, B, lab=None, idx1=None):
    """per-jet purity of graph idx (N,P,k): dict of (N,) arrays: u (unweighted), w (pT-weighted), ru / rw (random
    expectation), ov (fraction of neighbours also in idx1), valid; and per-particle purity (N,P)"""
    import torch
    lab = B['lab'] if lab is None else lab; real = B['m'][:, 0] > 0; N, P, k = idx.shape; fl = idx.reshape(N, -1)
    nl = lab.gather(1, fl).view(N, P, k); nr = real.gather(1, fl).view(N, P, k)
    nr = nr & (idx != torch.arange(P, device=idx.device)[None, :, None])
    same = (nl == lab[:, :, None]) & nr; nre = nr.sum(-1); pur = same.sum(-1).float() / nre.clamp(min=1).float()
    valid_j = (lab >= 0).any(1); ok = (real & (nre > 0) & valid_j[:, None]).float(); pt = B['pt'] * ok
    nreal = real.sum(1, keepdim=True).float(); ns = (lab[:, :, None] == lab[:, None, :]).float().mul(real[:, None, :].float()).sum(-1)
    rnd = (ns - 1) / (nreal - 1).clamp(min=1)
    sw = lambda v, w: (v * w).sum(1) / w.sum(1).clamp(min=1e-12)
    out = dict(u=sw(pur, ok), w=sw(pur, pt), ru=sw(rnd, ok), rw=sw(rnd, pt), valid=valid_j & (ok.sum(1) > 0))
    if idx1 is not None:
        in1 = torch.zeros(N, P, P, dtype=torch.bool, device=idx.device).scatter_(-1, idx1, True)
        ovl = in1.gather(-1, idx) & nr; out['ov'] = sw(ovl.sum(-1).float() / nre.clamp(min=1).float(), ok)
    return {a: b.cpu().numpy() for a, b in out.items()}, (pur * ok).cpu().numpy()


# ------------------------------------------------------------------------------------------------ statistics
class Boot:
    def __init__(self, n):
        rng = np.random.default_rng(6468); self.I = rng.integers(0, n, (NB, n)).astype(np.int32)

    def mean(self, v, sel=None):
        v = np.asarray(v, float); sel = np.ones(len(v), bool) if sel is None else sel
        b = np.array([np.mean(v[I][sel[I]]) for I in self.I])
        return [float(np.mean(v[sel])), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], b

    def ratio(self, a, b, sel):
        a, b = np.asarray(a, float), np.asarray(b, float)
        bs = np.array([np.mean(a[I][sel[I]]) / max(np.mean(b[I][sel[I]]), 1e-12) for I in self.I])
        return [float(a[sel].mean() / max(b[sel].mean(), 1e-12)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def pboot(b):
    """two-sided bootstrap p for H0: mean difference = 0 from the bootstrap distribution b of the difference"""
    b = np.asarray(b); return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1.0 / len(b)))


def mcnemar(a, b):
    from scipy import stats
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    return dict(n10=n10, n01=n01, p=1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5))))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); out = [None] * m; run_ = 0.0
    for r, i in enumerate(order): run_ = max(run_, min(1.0, (m - r) * ps[i])); out[i] = run_
    return out


PAIRS = dict(WZ=(iW, iZ), tQ=(iT, iQ), WQ=(iW, iQ))


def flips(L0, L, y):
    out = dict(argmax=(L0.argmax(1) != L.argmax(1)))
    for nm, (a, b) in PAIRS.items():
        out[nm] = np.sign(L0[:, a] - L0[:, b]) != np.sign(L[:, a] - L[:, b])
    return out


def pair_sel(y):
    return dict(argmax=np.ones(len(y), bool), WZ=np.isin(y, (iW, iZ)), tQ=np.isin(y, (iT, iQ)), WQ=np.isin(y, (iW, iQ)))


def confusion(L, y):
    return {str(c): np.bincount(L[y == lab].argmax(1), minlength=10).tolist() for c, lab, _ in CLS}


# ------------------------------------------------------------------------------------------------ edits
def sep_edit(J, lab2, s=None, rot=False):
    from interp.part_core import build, kin, jmass
    pt, de, dp, ms = kin(J); real = pt > 0; w = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12)
    ce, cp = (w * de).sum(1, keepdims=True), (w * dp).sum(1, keepdims=True); nde, ndp = de.copy(), dp.copy()
    if rot:
        nde = ce - (dp - cp); ndp = cp + (de - ce)
    else:
        for a in (0, 1):
            ia = (lab2 == a) & real; wa = np.where(ia, pt, 0); sa = np.maximum(wa.sum(1, keepdims=True), 1e-12)
            cae, cap = (wa * de).sum(1, keepdims=True) / sa, (wa * dp).sum(1, keepdims=True) / sa
            nde = np.where(ia, nde + (s - 1) * (cae - ce), nde); ndp = np.where(ia, ndp + (s - 1) * (cap - cp), ndp)
    nde, ndp = nde * real, ndp * real
    J1 = build(J, pt, nde, ndp, ms); lam = (jmass(J) / np.maximum(jmass(J1), 1e-9))[:, None]
    J2 = build(J, pt * lam, nde, ndp, ms * lam)
    p2 = kin(J2)[0]; order_ok = bool(np.allclose(p2, (pt * lam).astype(np.float32), rtol=1e-4, atol=1e-4))
    chk = dict(dlogm_absmax=float(np.abs(np.log(np.maximum(jmass(J2), 1e-9)) - np.log(np.maximum(jmass(J), 1e-9))).max()),
               lambda_mean=float(lam.mean()), lambda_p5_p95=[float(np.percentile(lam, 5)), float(np.percentile(lam, 95))],
               order_kept=order_ok, mask_equal=bool(np.array_equal(J2['mask'], J['mask'])))
    return J2, chk


# ------------------------------------------------------------------------------------------------ main pieces
def batches(J, n, dev, lab):
    from interp.part_core import sub
    nj = len(J['label'])
    return [tensors(sub(J, np.arange(s, min(nj, s + BS))), n, dev, lab[s:s + BS]) for s in range(0, nj, BS)]


def observe(net, TB, keep_jet=None, alt=None):
    """unedited logits and per-layer purity arrays (concatenated over batches); per-particle purity of jet keep_jet;
    alt: alternative label array (the other number of subjets) for the robustness row"""
    import torch
    Ls, P = [], [dict() for _ in range(3)]; pp = None; PA = [dict() for _ in range(3)]; off = 0
    with torch.no_grad():
        for bi, B in enumerate(TB):
            L, idxs, _ = run(net, B, rec=True); Ls.append(L.double().cpu().numpy())
            for li in range(3):
                d, per = purity(idxs[li], B, idx1=idxs[0])
                for a, v in d.items(): P[li].setdefault(a, []).append(v)
                if alt is not None:
                    da, _ = purity(idxs[li], B, lab=torch.from_numpy(alt[off:off + len(L)]).to(L.device))
                    for a, v in da.items(): PA[li].setdefault(a, []).append(v)
                if keep_jet is not None and off <= keep_jet < off + len(L):
                    pp = pp or {}; pp[f'ec{li + 1}'] = per[keep_jet - off].tolist()
            off += len(L)
    P = [{a: np.concatenate(v) for a, v in d.items()} for d in P]
    PA = [{a: np.concatenate(v) for a, v in d.items()} for d in PA] if alt is not None else None
    return np.concatenate(Ls), P, PA, pp


def purity_summary(P, y, bt, ref=None):
    """purity rows per layer and class group; paired bootstrap of EC_l - EC1, EC_l - random, and model - ref (same layer)"""
    groups = dict(WZ=np.isin(y, (iW, iZ)), W=y == iW, Z=y == iZ, top=y == iT, QCD=y == iQ, all=np.ones(len(y), bool))
    out = {}
    for li in range(3):
        d = P[li]; ok = d['valid']; o = {}
        for g, sel in groups.items():
            s = sel & ok; r = dict(n=int(s.sum()))
            for wt in ('u', 'w'):
                m, b = bt.mean(d[wt], s); mr, br = bt.mean(d['r' + wt], s); r[wt] = m; r['rand_' + wt] = mr
                dr = bt.mean(d[wt] - d['r' + wt], s); r['minus_rand_' + wt] = dr[0]; r['p_minus_rand_' + wt] = pboot(dr[1])
                if li > 0:
                    d1 = bt.mean(d[wt] - P[0][wt], s); r['minus_ec1_' + wt] = d1[0]; r['p_minus_ec1_' + wt] = pboot(d1[1])
                if ref is not None:
                    dd = bt.mean(d[wt] - ref[li][wt], s & ref[li]['valid']); r['minus_ref_' + wt] = dd[0]; r['p_minus_ref_' + wt] = pboot(dd[1])
            if 'ov' in d and li > 0: r['overlap_ec1'] = bt.mean(d['ov'], s)[0]
            o[g] = r
        out[f'ec{li + 1}'] = o
    return out


def interventions(net, TB, L0, y, bt, dev):
    import torch
    out = {}; FL = {}; sel = pair_sel(y)
    for ls, layers in LAYERSETS.items():
        for arm in ARMS:
            g = torch.Generator(device=dev).manual_seed(6467); Ls, R = [], [[] for _ in layers]
            with torch.no_grad():
                for B in TB:
                    L, _, reps = run(net, B, arm=arm, layers=layers, gen=g); Ls.append(L.double().cpu().numpy())
                    real = (B['m'][:, 0] > 0)
                    for q, nr in enumerate(reps): R[q].append((nr.float() * real).sum().item() / real.sum().item() / 16.0)
            L = np.concatenate(Ls); f = flips(L0, L, y); nm = f'{arm}-{ls}'; FL[nm] = f
            r = dict(acc=float((L.argmax(1) == y).mean()), confusion=confusion(L, y),
                     frac_edges_replaced={f'ec{li + 1}': float(np.mean(v)) for li, v in zip(layers, R)},
                     mcnemar_acc_vs_unedited=mcnemar(L.argmax(1) == y, L0.argmax(1) == y))
            for c, sl in sel.items(): r[f'changed_{c}'] = bt.mean(f[c], sl)[0]
            out[nm] = r
            log(f'    {nm:10s} acc {r["acc"]:.4f} edges replaced {r["frac_edges_replaced"]} changed argmax {r["changed_argmax"][0]:.4f} '
                f'WZ {r["changed_WZ"][0]:.4f} tQ {r["changed_tQ"][0]:.4f} WQ {r["changed_WQ"][0]:.4f}')
    tests = {}
    for ls in LAYERSETS:
        for a, cs in (('R1', ('R3r1', 'R3n1')), ('R2', ('R3r2', 'R3n2'))):
            for c in cs:
                for t, sl in sel.items():
                    k = f'{a}-{ls} vs {c}-{ls} [{t}]'; mc = mcnemar(FL[f'{a}-{ls}'][t][sl], FL[f'{c}-{ls}'][t][sl])
                    mc['ratio'] = bt.ratio(FL[f'{a}-{ls}'][t], FL[f'{c}-{ls}'][t], sl); tests[k] = mc
        for t, sl in sel.items():   # R2 vs R1 (cross-prong graph vs within-prong graph) as a descriptive comparison
            k = f'R2-{ls} vs R1-{ls} [{t}]'; mc = mcnemar(FL[f'R2-{ls}'][t][sl], FL[f'R1-{ls}'][t][sl])
            mc['ratio'] = bt.ratio(FL[f'R2-{ls}'][t], FL[f'R1-{ls}'][t], sl); tests[k] = mc
    return out, tests


def sep_scan(nets, J, y, lab2, n_of, dev, bt):
    """W jets with two subjets: separation scale / rotation; purity and W-vs-Z score per model"""
    from interp.part_core import sub
    import torch
    rows = np.where((y == iW) & (lab2 >= 0).any(1))[0]; JW = sub(J, rows); LW = lab2[rows]; out = dict(n=int(len(rows)), edits={})
    base = {}
    for e in [1.0] + [x for x in SEPS if x != 1.0] + ['rot90']:
        if e == 1.0: Je, chk = JW, dict(note='unedited')
        else: Je, chk = sep_edit(JW, LW, s=None if e == 'rot90' else e, rot=(e == 'rot90'))
        res = dict(check=chk)
        for mname, net in nets.items():
            TB = batches(Je, n_of[mname], dev, LW); Ls, P = [], [dict() for _ in range(3)]
            with torch.no_grad():
                for B in TB:
                    L, idxs, _ = run(net, B, rec=True); Ls.append(L.double().cpu().numpy())
                    for li in range(3):
                        d, _ = purity(idxs[li], B)
                        for a, v in d.items(): P[li].setdefault(a, []).append(v)
            L = np.concatenate(Ls); P = [{a: np.concatenate(v) for a, v in d.items()} for d in P]; dWZ = L[:, iW] - L[:, iZ]
            if e == 1.0: base[mname] = (P, dWZ)
            ex = np.exp(L - L.max(1, keepdims=True)); r = dict(dWZ=float(dWZ.mean()), pW=float((ex[:, iW] / ex.sum(1)).mean()))
            P0, d0 = base[mname]; sel = P[0]['valid'] & P0[0]['valid']; r['ddWZ'] = bt_mean_sub(d0, dWZ, sel)
            for li in range(3):
                for wt in ('u', 'w'):
                    r[f'ec{li + 1}_{wt}'] = float(P[li][wt][sel].mean())
                    r[f'ec{li + 1}_{wt}_minus_unedited'] = bt_mean_sub(P0[li][wt], P[li][wt], sel)
            # difference of differences: EC3 change minus EC1 change (does EC3 follow the eta-phi geometry?)
            for li in (1, 2):
                r[f'ec{li + 1}_u_change_minus_ec1_change'] = bt_mean_sub(P0[li]['u'] - P0[0]['u'], P[li]['u'] - P[0]['u'], sel)
            res[mname] = r
            log(f'    sep {e}: {mname:12s} purity u EC1 {r["ec1_u"]:.3f} EC2 {r["ec2_u"]:.3f} EC3 {r["ec3_u"]:.3f} dWZ {r["dWZ"]:+.3f} {chk}')
        out['edits'][str(e)] = res
    return out


def bt_mean_sub(a, b, sel):
    """mean of (b - a) over sel with a 95 % bootstrap interval and two-sided bootstrap p"""
    v = (np.asarray(b, float) - np.asarray(a, float))[sel]; rng = np.random.default_rng(6469); n = len(v)
    bs = np.array([v[rng.integers(0, n, n)].mean() for _ in range(NB)])
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), pboot(bs)]


def pick_top_jet(J, y, lab3):
    """first top jet in sample order with >= 40 real particles and every one of its 3 subjets holding >= 6 particles"""
    for i in np.where(y == iT)[0]:
        r = J['mask'][i] > 0
        if r.sum() >= 40 and all(((lab3[i] == a) & r).sum() >= 6 for a in range(3)): return int(i)
    return int(np.where(y == iT)[0][0])


def channel_permute(net, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in net.modules():
            if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                W = mod.weight; perm = torch.randperm(W.shape[1], generator=g).to(W.device); W.copy_(W[:, perm])
    return net


# ------------------------------------------------------------------------------------------------ main
def main():
    import torch
    from interp import pn_net
    from interp.part_core import sample, build, kin, sub, jmass
    from jetdistill.part.network import FEATURES
    RES.mkdir(parents=True, exist_ok=True); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device {dev} {torch.cuda.get_device_name(0) if dev == "cuda" else ""} torch {torch.__version__}')
    J, info = load_sample(); y = J['label'].astype(int); n = len(y)
    t1 = time.time(); lab2, lab3 = excl_kt(J, dev); log(f'exclusive kT done ({time.time() - t1:.0f} s)')
    prim = np.where(np.isin(y, (iW, iZ))[:, None], lab2, lab3); alt = np.where(np.isin(y, (iW, iZ))[:, None], lab3, lab2)
    real = J['mask'] > 0
    sub_info = {c: dict(n_jets=int((y == lab).sum()), n_valid=int(((prim >= 0).any(1) & (y == lab)).sum()),
                        mean_real=float(real[y == lab].sum(1).mean()),
                        mean_subjet_sizes=sorted([float(((prim[y == lab] == a) & real[y == lab]).sum(1).mean()) for a in range(ns)], reverse=True))
                for c, lab, ns in CLS}
    log('subjets', sub_info)
    kj = pick_top_jet(J, y, lab3); bt = Boot(n)
    J1 = sample(NP1, 3000); J1 = build(J1, *kin(J1))
    res = dict(node='P64', smoke=SMOKE, N=n, files=info, subjets=sub_info, NB=NB, figure_jet=dict(index=kj), models={}, controls={})
    fj = dict(deta=J['deta'][kj][real[kj]].tolist(), dphi=J['dphi'][kj][real[kj]].tolist(), pt=np.hypot(J['px'][kj], J['py'][kj])[real[kj]].tolist(),
              lab=lab3[kj][real[kj]].tolist(), jet_pt=float(J['jet_pt'][kj]), mass=float(jmass(sub(J, [kj]))[0]))
    res['figure_jet'].update(fj); nets = {}; n_of = {}; P_ref = None
    for mname, nn_ in (('PN_full', 'full'), ('PN_kin', 'kin')):
        nf = len(FEATURES[nn_]); net = pn_net.load_particlenet(PNDIR / f'ParticleNet_{nn_}.pt', nf, dev)
        for p in net.parameters(): p.requires_grad_(False)
        from weaver.nn.model.ParticleNet import ParticleNet as WPN
        w = WPN(input_dims=nf, **pn_net.PN_CFG); sd = torch.load(PNDIR / f'ParticleNet_{nn_}.pt', map_location='cpu')
        w.load_state_dict({k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}); w = w.to(dev).eval()
        TB = batches(J, nn_, dev, prim); B = TB[0]
        with torch.no_grad(): d = float((run(net, B)[0] - w(B['p'].clone(), B['x'].clone(), B['m'])).abs().max())
        R = dict(weaver_check_max_abs_dlogit=d); log(f'{mname}: own forward vs weaver max |dlogit| {d:.2e}'); assert d <= 1e-4, d
        lab1 = np.full((len(J1['label']), 128), -1, np.int64); TB1 = batches(J1, nn_, dev, lab1)
        with torch.no_grad(): L1 = np.concatenate([run(net, b)[0].double().cpu().numpy() for b in TB1])
        R['acc_P1'] = float((L1.argmax(1) == J1['label']).mean())
        L0, P, PA, pp = observe(net, TB, keep_jet=kj, alt=alt)
        R['acc_sample'] = float((L0.argmax(1) == y).mean()); R['confusion_unedited'] = confusion(L0, y)
        R['acc_sample_per_class'] = {c: float((L0[y == lab].argmax(1) == lab).mean()) for c, lab, _ in CLS}
        log(f'{mname}: acc P1 {R["acc_P1"]:.4f} (N {len(L1)}), acc this sample {R["acc_sample"]:.4f} {R["acc_sample_per_class"]}')
        R['purity'] = purity_summary(P, y, bt); R['purity_alt_nsub'] = purity_summary(PA, y, bt); R['figure_jet_purity'] = pp
        for li in range(3):
            q = R['purity'][f'ec{li + 1}']
            log(f'  {mname} EC{li + 1} purity u/w: WZ {q["WZ"]["u"][0]:.3f}/{q["WZ"]["w"][0]:.3f} (rand {q["WZ"]["rand_u"][0]:.3f}) '
                f'top {q["top"]["u"][0]:.3f}/{q["top"]["w"][0]:.3f} (rand {q["top"]["rand_u"][0]:.3f}) QCD {q["QCD"]["u"][0]:.3f} (rand {q["QCD"]["rand_u"][0]:.3f}) '
                f'overlap EC1 {q["all"].get("overlap_ec1", [1.0])[0]:.3f}')
        if nn_ == 'full': P_ref = P
        R['arms'], R['tests'] = interventions(net, TB, L0, y, bt, dev)
        res['models'][mname] = R; nets[mname] = net; n_of[mname] = nn_; json.dump(res, open(FN, 'w'), indent=1, default=float)
        del TB; torch.cuda.empty_cache()
    # Holm over all R vs R3 McNemar tests (arms x layer sets x call types x models)
    keys = [(m, k) for m in res['models'] for k in res['models'][m]['tests'] if ' vs R1-' not in k]
    for (m, k), h in zip(keys, holm([res['models'][m]['tests'][k]['p'] for m, k in keys])): res['models'][m]['tests'][k]['holm'] = h
    res['n_holm'] = len(keys)
    # controls on PN_full
    nf = len(FEATURES['full'])
    ctrl = {}
    for s in (6464, 6465):
        torch.manual_seed(s); ctrl[f'untrained_s{s}'] = pn_net.ParticleNet(input_dims=nf, **pn_net.PN_CFG).to(dev).eval()
    ctrl['chperm_s6466'] = channel_permute(pn_net.load_particlenet(PNDIR / 'ParticleNet_full.pt', nf, dev), 6466)
    TB = batches(J, 'full', dev, prim)
    for cn, cnet in ctrl.items():
        for p in cnet.parameters(): p.requires_grad_(False)
        Lc, Pc, _, ppc = observe(cnet, TB, keep_jet=kj)
        rc = dict(acc_sample=float((Lc.argmax(1) == y).mean()), purity=purity_summary(Pc, y, bt), figure_jet_purity=ppc)
        rc['trained_minus_control'] = purity_summary(P_ref, y, bt, ref=Pc)
        res['controls'][cn] = rc
        q = rc['purity']
        log(f'control {cn}: acc {rc["acc_sample"]:.4f}; purity u WZ EC1/2/3 {q["ec1"]["WZ"]["u"][0]:.3f}/{q["ec2"]["WZ"]["u"][0]:.3f}/{q["ec3"]["WZ"]["u"][0]:.3f} '
            f'top {q["ec1"]["top"]["u"][0]:.3f}/{q["ec2"]["top"]["u"][0]:.3f}/{q["ec3"]["top"]["u"][0]:.3f}')
        json.dump(res, open(FN, 'w'), indent=1, default=float)
    del TB; torch.cuda.empty_cache()
    # physics edit: prong separation at fixed invariant mass, rotation
    nets_s = dict(nets); nets_s['untrained_s6464'] = ctrl['untrained_s6464']; n_of['untrained_s6464'] = 'full'
    res['sep_scan'] = sep_scan(nets_s, J, y, lab2, n_of, dev, bt)
    json.dump(res, open(FN, 'w'), indent=1, default=float); plot(res); log('done', FN)


# ------------------------------------------------------------------------------------------------ figure
def plot(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(20, 11.5)); fj = res['figure_jet']
    de, dp, pt, lb = (np.array(fj[k]) for k in ('deta', 'dphi', 'pt', 'lab')); mk = ['o', 's', '^']; sz = 15 + 250 * np.sqrt(pt / pt.max())
    for q, l in enumerate((1, 3)):
        a = ax[0, q]; pur = np.array(res['models']['PN_full']['figure_jet_purity'][f'ec{l}'])[:len(de)]
        for g in range(3):
            s_ = lb == g
            sc = a.scatter(de[s_], dp[s_], s=sz[s_], c=pur[s_], cmap='viridis', vmin=0, vmax=1, marker=mk[g], edgecolors='k', linewidths=0.4,
                           label=f'subjet {g + 1} ({int(s_.sum())} particles)')
        plt.colorbar(sc, ax=a, label='per-particle edge purity\n(fraction of its 16 neighbours in its own subjet)')
        a.set_xlabel('Δη to jet axis'); a.set_ylabel('Δφ to jet axis [rad]'); a.legend(fontsize=7, loc='lower left')
        a.set_title(f'({"ab"[q]}) one top jet (pT {fj["jet_pt"]:.0f} GeV, m {fj["mass"]:.0f} GeV), PN_full\n'
                    f'{"EC1 graph: 16 nearest in (Δη, Δφ)" if l == 1 else "EC3 graph: 16 nearest in learned 128-d features"}; marker = exclusive-kT subjet, size ∝ √pT', fontsize=9)
    a = ax[0, 2]; groups = (('WZ', 'W, Z (2 subjets)'), ('top', 'top (3 subjets)'), ('QCD', 'QCD (3 subjets)'))
    series = [('PN_full', res['models']['PN_full']['purity'], '#ff7f0e'), ('PN_kin', res['models']['PN_kin']['purity'], '#9467bd'),
              ('untrained PN_full (s6464)', res['controls']['untrained_s6464']['purity'], 'grey'),
              ('channel-permuted PN_full', res['controls']['chperm_s6466']['purity'], '#8c564b')]
    xs = np.arange(9); w = 0.18
    for q, (nm, P, c) in enumerate(series):
        v = [P[f'ec{l}'][g]['u'] for g, _ in groups for l in (1, 2, 3)]
        a.bar(xs + (q - 1.5) * w, [x[0] for x in v], w, color=c, label=nm, yerr=np.array([[x[0] - x[1], x[2] - x[0]] for x in v]).T, capsize=2)
    rnd = [res['models']['PN_full']['purity'][f'ec{l}'][g]['rand_u'][0] for g, _ in groups for l in (1, 2, 3)]
    for x_, r in zip(xs, rnd): a.plot([x_ - 0.4, x_ + 0.4], [r, r], c='k', lw=1.5)
    a.plot([], [], c='k', lw=1.5, label='random graph, same degree')
    a.set_xticks(xs); a.set_xticklabels([f'EC{l}\n{lab}' if l == 2 else f'EC{l}' for _, lab in groups for l in (1, 2, 3)], fontsize=8)
    a.set_ylim(0, 1.05); a.set_ylabel('edge purity (mean over particles, then over jets)'); a.legend(fontsize=7, loc='lower right')
    a.set_title(f'(c) fraction of a particle\'s 16 neighbours in its own subjet, per layer\n(val_5M _124, N {res["N"]} jets)', fontsize=9)
    arms = ('R1', 'R2', 'R3r2', 'R3n2', 'R4'); lab_ = ['R1 within-prong\nonly', 'R2 cross-prong\nonly', 'R3 random\n(R2 count)',
                                                       'R3n nearest, prong-blind\n(R2 count)', 'R4 EC1 graph']
    for q, ls in enumerate(('ec3', 'both')):
        a = ax[1, q]; xs = np.arange(len(arms)); w = 0.2
        for z, (mn, t, c) in enumerate((('PN_full', 'WZ', '#ff7f0e'), ('PN_full', 'tQ', '#d62728'), ('PN_kin', 'WZ', '#9467bd'), ('PN_kin', 'tQ', '#17becf'))):
            v = [res['models'][mn]['arms'][f'{r}-{ls}'][f'changed_{t}'] for r in arms]
            a.bar(xs + (z - 1.5) * w, [x[0] for x in v], w, color=c, label=f'{mn}: {"W vs Z" if t == "WZ" else "top vs QCD"} calls',
                  yerr=np.array([[x[0] - x[1], x[2] - x[0]] for x in v]).T, capsize=2)
        a.set_xticks(xs); a.set_xticklabels(lab_, fontsize=8); a.set_ylabel('fraction of calls changed vs the unedited network'); a.legend(fontsize=7)
        a.set_title(f'({"de"[q]}) {"EC3 graph" if ls == "ec3" else "EC2 and EC3 graphs"} rewired, all weights fixed', fontsize=9)
    a = ax[1, 2]; S = res['sep_scan']['edits']; ss = list(SEPS)
    for mn, c in (('PN_full', '#ff7f0e'), ('PN_kin', '#9467bd'), ('untrained_s6464', 'grey')):
        for l, ls in ((1, ':'), (3, '-')):
            a.plot(ss, [S[str(s)][mn][f'ec{l}_u'] for s in ss], ls=ls, marker='o', c=c, label=f'{mn} EC{l}')
    a.axhline(np.mean([S['1.0'][mn]['ec1_u'] for mn in ('PN_full',)]) * 0 + res['models']['PN_full']['purity']['ec1']['W']['rand_u'][0], c='k', lw=1, label='random graph')
    a.set_xlabel('prong separation scale s (prongs moved about the pT centroid; jet mass held fixed)')
    a.set_ylabel('edge purity, true W jets (2 subjets)'); a.legend(fontsize=7, ncol=2)
    a.set_title(f'(f) EC1 (dotted) and EC3 (solid) purity vs prong separation (N {res["sep_scan"]["n"]} W jets)', fontsize=9)
    fig.suptitle('P64: do ParticleNet\'s learned neighbour graphs (EC2, EC3) group particles by prong?', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG, dpi=110); log('figure', FIG)


if __name__ == '__main__':
    if '--plot' in sys.argv: plot(json.load(open(FN)))
    else: main()
