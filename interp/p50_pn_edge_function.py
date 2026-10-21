"""P50 - what ParticleNet's EdgeConv-1/2 edge function computes, and whether its invariant or its oriented part sets
ParticleNet's smaller axis response (loop C, round C11, exploratory; takes up P47).

Models (frozen): ParticleNet_full (17 inputs) and ParticleNet_kin (7 inputs), official JetClass weights
(/j-jepa-vol/aaron-interp/models/particlenet), network code interp/pn_net.py (weaver-core 0.4.17 copy). Own message-level
forward checked against weaver's ParticleNet.forward (|dlogit| <= 1e-5 required). Controls: untrained ParticleNet_full
(PyTorch default init, torch seed 5050) and weight-permuted ParticleNet_full (every trained weight tensor's entries permuted,
torch seed 5053; BatchNorm running statistics kept).

EdgeConv layer l (l = 1, 2): for each particle i and each of its k = 16 nearest neighbours j (layer 1: nearest in (deta, dphi);
layer 2: nearest in the 64-d layer-1 output), the edge message is
    e_ij = h_l(x_i, x_j - x_i)   (3 x [1x1 conv, BatchNorm, ReLU]; 64 channels in layer 1, 128 in layer 2)
and the layer output is ReLU(shortcut(x_i) + mean_j e_ij). x = the BatchNorm'd 17 / 7 input features in layer 1, the 64-d
layer-1 output in layer 2.

Samples: val_5M *_123 of each class. Evaluation: 1,000 random rows per class (default_rng(5050)), N 10,000. Fit: 500 random rows
per class from the remaining rows (default_rng(5051)), N 5,000, split by jet 80 / 20 into fit-train / held-out (same rng);
35 % of the real particles of each fit jet (torch seed 5052) are kept as message centres with all 16 of their edges.
P1 reference sample (test_20M *_100 rows 3,000-4,000 per class) for the unedited accuracy only.
Step 1: record e_ij per channel (before the mean over the 16 neighbours) on the unedited fit jets.
Step 2: fit e_ij per channel from feature sets (pair invariants as in ParT's pair embedding: ln kT, ln z, ln dR, ln m2_ij from
  the two particles' four-vectors, dR in (rapidity, phi); a flag for padded neighbours, invariants set to 0 there):
    A  = centre input of the layer (x_i) + invariants
    B  = A + signed deta_ij, dphi_ij (orientation of the pair)
    C  = A + non-angular input differences of the neighbour ((x_j - x_i) of the BatchNorm'd inputs other than deta, dphi, deltaR)
    N  = null: x_i + invariants of a shuffled neighbour (the 16 neighbours of i randomly permuted, torch seed 5052)
    F  = ceiling: the layer's exact inputs (x_i, x_j - x_i)
  each as (hinge) an additive piecewise-linear model (every input column + 7 hinges ReLU(u - q) at its 1/8..7/8 quantiles,
  no interactions, ridge, output clamped at 0; the C1-style formula form) and (mlp) an MLP D -> 256 -> 256 -> C (ReLU; output
  clamped at 0; Adam 2e-3, cosine, 3,000 steps of 4,096 edges). Held-out R^2 per channel (edge level and after the mean over
  the 16 neighbours, the quantity the network uses).
Step 3 (formula circuits kept alone): replace all EdgeConv-1 messages (arm ec1), then EdgeConv-1 and -2 messages (arm ec12),
  by a fit (A, B, C, N, F; hinge or mlp), the rest of the network intact (kNN graphs recomputed from the substituted features).
  Accuracy, 10-class confusion, McNemar on per-jet correctness vs the unedited network. Edits on the evaluation jets (stored
  axis fixed, masks unchanged): phi + 0.06 rad on every real particle (rigid rotation about the beam: invariant mass and every
  pair invariant unchanged); matched dilation dil_phi: (deta, dphi) x k_j, k_j = sqrt(M0M2(shifted) / M0M2(unedited)); the same
  for eta + 0.06 (P43v direction check) with its own matched dilation. d = l_W - l_Z.
  R = slope through 0 of Delta d(shift) against Delta d(matched dilation), main over true W and Z jets (plan), also over W, Z and
  QCD (P45 definition). 95 % bootstrap over jets (2,000), paired permutation p for Delta R (arm - unedited, B - A, C - A; 2,000
  sign swaps of the per-jet pair), Holm over arms and layers. R of an arm is compared only with arms whose accuracy is within
  1 point; otherwise flagged not comparable.
  ./sync.sh && ./part_job.sh c-p50-run 'python -m interp.p50_pn_edge_function --smoke && python -m interp.p50_pn_edge_function'
  -> $AI_RESULTS/particlenet/p50_pn_edge[_smoke].json, p50_pn_edge[_smoke].png
  second run with longer MLP training: ./part_job.sh c-p50-long 'P50_STEPS=12000 python -m interp.p50_pn_edge_function' -> p50_pn_edge_s12000.json
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SMOKE = '--smoke' in sys.argv
TAG = ('_smoke' if SMOKE else '') + (f"_s{os.environ['P50_STEPS']}" if os.environ.get('P50_STEPS') else '')
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'particlenet'
FN, FIG = RES / f'p50_pn_edge{TAG}.json', RES / f'p50_pn_edge{TAG}.png'
PNDIR = Path('/j-jepa-vol/aaron-interp/models/particlenet')
VAL = Path('/j-jepa-vol/JetClass/Pythia/val_5M')
NEV = 60 if SMOKE else 1000
NFIT = 40 if SMOKE else 500
NP1 = 50 if SMOKE else 1000
KEEPC = 0.35
NB = 200 if SMOKE else 2000
NPERM = 200 if SMOKE else 2000
STEPS = 60 if SMOKE else int(os.environ.get('P50_STEPS', 3000))      # P50_STEPS: longer MLP training (second run)
SHIFT = 0.06
K = 16
iZ, iW, iQ = 6, 7, 0
KINDS = ('A', 'B', 'C', 'N', 'F')
FORMS = ('hinge', 'mlp')
KEEP = ['px', 'py', 'pz', 'energy', 'deta', 'dphi', 'd0val', 'd0err', 'dzval', 'dzerr', 'charge', 'isChargedHadron', 'isNeutralHadron',
        'isPhoton', 'isElectron', 'isMuon', 'mask', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'label']
CONDS = ('base', 'phi', 'dil_phi', 'eta', 'dil_eta')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ data
def load_samples():
    from interp.part_core import read_root, PRE, CL, cat, build, kin
    import uproot
    re, rf = np.random.default_rng(5050), np.random.default_rng(5051); E, F, info = [], [], []
    for c in CL:
        path = VAL / f'{PRE[c]}_123.root'; N = uproot.open(path)['tree'].num_entries
        if SMOKE: N = min(N, 20000)
        ev = np.sort(re.choice(N, NEV, replace=False)); rest = np.setdiff1d(np.arange(N), ev)
        ft = np.sort(rf.choice(rest, NFIT, replace=False)); rows = np.union1d(ev, ft); parts = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r):
                J = read_root(path, start=s, stop=min(s + 25000, N)); parts.append({k: J[k][r - s] for k in KEEP})
        J = cat(parts); ie = np.isin(rows, ev)
        E.append({k: v[ie] for k, v in J.items()}); F.append({k: v[~ie] for k, v in J.items()})
        info.append(dict(file=path.name, rows_in_file=int(N), eval=int(ie.sum()), fit=int((~ie).sum())))
        log(f'  {path.name}: {ie.sum()} eval + {(~ie).sum()} fit random rows of {N}')
    reb = lambda J: build(J, *kin(J))
    return reb(cat(E)), reb(cat(F)), info


def edits(J):
    from interp.part_core import build, kin, moments, jmass
    pt, de, dp, ms = kin(J); real = pt > 0; Mb = moments(pt, de, dp)['M0M2']; ok = Mb > 0; C = dict(base=J); chk = {}
    for nm, (a, b) in dict(phi=(de, dp + SHIFT * real), eta=(de + SHIFT * real, dp)).items():
        Js = build(J, pt, a, b, ms); Mp = moments(*kin(Js)[:3])['M0M2']
        kj = np.where(ok, np.sqrt(Mp / np.where(ok, Mb, 1.0)), 1.0)[:, None]
        Jd = build(J, pt, de * kj, dp * kj, ms); C[nm] = Js; C['dil_' + nm] = Jd
        m0, ms_, md = (np.log(np.maximum(jmass(x), 1.0)) for x in (J, Js, Jd))
        chk[nm] = dict(shift_dlogm_absmax=float(np.abs(ms_ - m0).max()), shift_dlogm_mean=float((ms_ - m0).mean()),
                       dil_dlogm_mean=float((md - m0).mean()), k_mean=float(kj.mean()),
                       mask_equal=bool(np.array_equal(Js['mask'], J['mask']) and np.array_equal(Jd['mask'], J['mask'])))
    return C, chk


# ------------------------------------------------------------------------------------------------ network pieces
def tensors(J, n, dev):
    """batch tensors as weaver feeds ParticleNet (length 128): x (N,F,128) masked, p (N,2,128) = (deta, dphi), m (N,1,128),
    fv (N,4,128) = (px, py, pz, E) of the real particles"""
    import torch
    from jetdistill.part.network import inputs, FEATURES
    x, _, m = inputs(J, n); P = x.shape[-1]
    xp = np.zeros((len(x), x.shape[1], 128), np.float32); xp[..., :P] = x; mp = np.zeros((len(x), 1, 128), np.float32); mp[..., :P] = m
    xp = xp * mp; names = [f[0] for f in FEATURES[n]]; ip = [names.index('deta'), names.index('dphi')]
    fv = np.stack([J[k][:, :128] * (J['mask'][:, :128] > 0) for k in ('px', 'py', 'pz', 'energy')], 1).astype(np.float32)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    return dict(x=T(xp), p=T(xp[:, ip]), m=T(mp), fv=T(fv))


def gather(x, idx):
    """x (N,F,P), idx (N,P,k) -> (N,F,P,k) with out[n,:,i,a] = x[n,:,idx[n,i,a]]"""
    import torch
    N, F, P = x.shape; k = idx.shape[-1]
    base = torch.arange(N, device=x.device).view(-1, 1, 1) * P
    return x.transpose(2, 1).reshape(-1, F)[(idx + base).view(-1)].view(N, P, k, F).permute(0, 3, 1, 2)


def pairinv(fv, j):
    """ParT's pair features (weaver pairwise_lv_fts): ln kT, ln z, ln dR, ln m2 for (i, j); fv (N,4,P), j (N,P,k) -> (N,4,P,k)"""
    import torch
    eps = 1e-8; fi = fv.unsqueeze(-1); fj = gather(fv, j)
    def ptrp(f):
        px, py, pz, E = f[:, 0], f[:, 1], f[:, 2], f[:, 3]; pt = torch.sqrt(px ** 2 + py ** 2)
        rap = 0.5 * torch.log(1 + 2 * pz / (E - pz).clamp(min=1e-20)); phi = torch.atan2(py, px); return pt, rap, phi
    pti, ri, phii = ptrp(fi); ptj, rj, phij = ptrp(fj)
    dphi = (phij - phii + math.pi) % (2 * math.pi) - math.pi; delta = torch.sqrt((rj - ri) ** 2 + dphi ** 2)
    ptmin = torch.minimum(pti, ptj); lndelta = torch.log(delta.clamp(min=eps)); lnkt = torch.log((ptmin * delta).clamp(min=eps))
    lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
    s = fi + fj; m2 = s[:, 3] ** 2 - s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2; lnm2 = torch.log(m2.clamp(min=eps))
    return torch.stack([lnkt, lnz, lndelta, lnm2], 1)


class Model:
    def __init__(self, name, n, net, dev):
        from jetdistill.part.network import FEATURES
        self.name, self.n, self.net, self.dev = name, n, net, dev
        names = [f[0] for f in FEATURES[n]]
        self.na = [i for i, f in enumerate(names) if f not in ('deta', 'dphi', 'deltaR')]
        self.ip = [names.index('deta'), names.index('dphi')]

    def run(self, B, sub=None, rec=False, nperm_gen=None):
        """own forward of ParticleNet (use_fusion False); sub: {layer index: Fit} replaces that layer's edge messages.
        rec: also return [(centre input, knn idx, messages, xb)] for layers 0 and 1"""
        import torch
        from interp import pn_net
        md = self.net; mask = B['m']; shift = (mask == 0) * 1e9; counts = torch.clamp(mask.sum(-1), min=1.0)
        xb = md.bn_fts(B['x']) * mask; fts = xb; recs = []
        for li, conv in enumerate(md.edge_convs):
            pts = (B['p'] if li == 0 else fts) + shift; idx = pn_net.knn(pts, conv.k)
            if sub is not None and li in sub: msg = sub[li].apply(self, fts, idx, B, xb)
            else:
                msg = pn_net.get_graph_feature_v1(fts, conv.k, idx)
                for c2, b2, act in zip(conv.convs, conv.bns, conv.acts): msg = act(b2(c2(msg)))
            if rec and li < 2: recs.append((fts, idx, msg, xb))
            sc = conv.sc_bn(conv.sc(fts)) if conv.sc is not None else fts
            fts = conv.sc_act(sc + msg.mean(-1)) * mask
        L = md.fc(fts.sum(-1) / counts)
        return (L, recs) if rec else L


def components(M, cin, idx, B, xb, gen=None):
    """per-edge feature blocks (N, d, P, k): cen = x_i, inv = 4 pair invariants + padded-neighbour flag, invN = the same for a
    shuffled neighbour, de = signed (deta_ij, dphi_ij), na = non-angular (x_j - x_i) of the BatchNorm'd inputs, nb = x_j - x_i
    of the layer's own input"""
    import torch
    N, Ci, P = cin.shape; k = idx.shape[-1]
    def inv(j):
        pj = 1 - gather(B['m'], j); return torch.cat([pairinv(B['fv'], j) * (1 - pj), pj], 1)
    out = dict(cen=cin.unsqueeze(-1).expand(-1, -1, -1, k), inv=inv(idx))
    if gen is not None:
        perm = torch.argsort(torch.rand(N, P, k, generator=gen, device='cpu'), -1).to(idx.device); out['invN'] = inv(idx.gather(-1, perm))
    pj = 1 - gather(B['m'], idx)
    out['de'] = (gather(B['p'], idx) - B['p'].unsqueeze(-1)) * (1 - pj)
    xa = xb[:, M.na]; out['na'] = gather(xa, idx) - xa.unsqueeze(-1)
    out['nb'] = gather(cin, idx) - cin.unsqueeze(-1)
    return out


COMPOSE = dict(A=('cen', 'inv'), B=('cen', 'inv', 'de'), C=('cen', 'inv', 'na'), N=('cen', 'invN'), F=('cen', 'nb'))


class Fit:
    """fitted edge function for one layer and feature set: predict (M, D) -> (M, C) clamped at 0"""

    def __init__(self, kind, form, mu, sd, knots=None, W=None, mlp=None):
        self.kind, self.form, self.mu, self.sd, self.knots, self.W, self.mlp = kind, form, mu, sd, knots, W, mlp

    def design(self, X):
        import torch
        Z = (X - self.mu) / self.sd
        if self.form == 'mlp': return Z
        H = torch.relu(Z.unsqueeze(-1) - self.knots.unsqueeze(0))            # (M, D, 7)
        return torch.cat([Z, H.reshape(len(Z), -1), torch.ones(len(Z), 1, device=Z.device)], 1)

    def predict(self, X):
        import torch
        out = []
        for s in range(0, len(X), 200000):
            D = self.design(X[s:s + 200000])
            out.append(torch.relu(self.mlp(D) if self.form == 'mlp' else D @ self.W))
        return torch.cat(out)

    def apply(self, M, cin, idx, B, xb):
        import torch
        # the null fit is applied to the true neighbours: the mean over the 16 neighbours of a shuffled-neighbour prediction
        # equals that over the true neighbours, since the shuffle only permutes the same 16 neighbours of i
        comp = components(M, cin, idx, B, xb, gen=None)
        X = torch.cat([comp[c if c != 'invN' else 'inv'] for c in COMPOSE[self.kind]], 1)
        N, D, P, k = X.shape; Y = self.predict(X.permute(0, 2, 3, 1).reshape(-1, D))
        return Y.view(N, P, k, -1).permute(0, 3, 1, 2)




# ------------------------------------------------------------------------------------------------ step 1: record the edges
def record(M, JF, train_jets):
    """for layers 0 and 1: kept centres' 16 edges -> dict of component tensors (Mc, d, 16) on the GPU, target (Mc, C, 16),
    split flag (Mc,) 1 = fit-train jet"""
    import torch
    from interp.part_core import sub
    gen = torch.Generator().manual_seed(5052); n = len(JF['label']); D = [dict() for _ in range(2)]
    with torch.no_grad():
        for s in range(0, n, 128):
            rows = np.arange(s, min(n, s + 128)); B = tensors(sub(JF, rows), M.n, M.dev); _, recs = M.run(B, rec=True)
            keep = (torch.rand(len(rows), 128, generator=gen) < KEEPC).to(M.dev) & (B['m'][:, 0] > 0)      # (N, P)
            tr = torch.from_numpy(np.isin(rows, train_jets)).to(M.dev).unsqueeze(1).expand(-1, 128)
            for li, (cin, idx, msg, xb) in enumerate(recs):
                comp = components(M, cin, idx, B, xb, gen=gen); comp['y'] = msg
                for kk, v in comp.items():
                    D[li].setdefault(kk, []).append(v.permute(0, 2, 1, 3)[keep].contiguous())      # (Mc, d, k)
                D[li].setdefault('train', []).append(tr[keep])
    return [{kk: torch.cat(v) for kk, v in d.items()} for d in D]


def flat(T):
    """(Mc, d, k) -> (Mc*k, d)"""
    return T.permute(0, 2, 1).reshape(-1, T.shape[1])


def r2(Y, P, k=K):
    """per-channel held-out R^2 at edge level and after the mean over the k neighbours; Y, P (Mc*k, C)"""
    import torch
    out = {}
    for lev, (y, p) in dict(edge=(Y, P), agg=(Y.view(-1, k, Y.shape[1]).mean(1), P.view(-1, k, P.shape[1]).mean(1))).items():
        y, p = y.double(), p.double(); sst = ((y - y.mean(0)) ** 2).sum(0); sse = ((y - p) ** 2).sum(0)
        live = sst / len(y) > 1e-6
        rc = (1 - sse / sst.clamp(min=1e-30))[live].cpu().numpy()
        out[lev] = dict(mean=float(rc.mean()) if len(rc) else float('nan'), median=float(np.median(rc)) if len(rc) else float('nan'),
                        frac_ge_0p7=float((rc >= 0.7).mean()) if len(rc) else float('nan'),
                        pooled=float(1 - sse[live].sum() / sst[live].sum()), n_live=int(live.sum()), n_ch=int(len(sst)),
                        per_channel=[round(float(v), 4) for v in rc])
    return out


def fit_one(Dl, kind, form, seed):
    import torch
    cols = COMPOSE[kind]; tr = Dl['train']
    X = torch.cat([Dl[c] for c in cols], 1); Y = Dl['y']
    Xtr, Ytr, Xte, Yte = flat(X[tr]), flat(Y[tr]), flat(X[~tr]), flat(Y[~tr])
    mu = Xtr.mean(0); sd = Xtr.std(0).clamp(min=1e-3)
    if form == 'hinge':
        Z = (Xtr[::7] - mu) / sd; q = torch.quantile(Z[:100000].float(), torch.linspace(0.125, 0.875, 7, device=Z.device), dim=0).T.contiguous()
        f = Fit(kind, form, mu, sd, knots=q); G = 0; C = 0
        for s in range(0, len(Xtr), 100000):
            Dd = f.design(Xtr[s:s + 100000]).double(); G = G + Dd.T @ Dd; C = C + Dd.T @ Ytr[s:s + 100000].double()
        lam = 1e-3 * G.diagonal().mean(); f.W = torch.linalg.solve(G + lam * torch.eye(len(G), dtype=G.dtype, device=G.device), C).float()
    else:
        torch.manual_seed(seed); d, c = X.shape[1], Y.shape[1]
        net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, c)).to(X.device)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        Ztr = (Xtr - mu) / sd; g = torch.Generator(device='cpu').manual_seed(seed)
        for st in range(STEPS):
            b = torch.randint(0, len(Ztr), (4096,), generator=g).to(X.device)
            loss = ((net(Ztr[b]) - Ytr[b]) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        net.eval()
        for p in net.parameters(): p.requires_grad_(False)
        f = Fit(kind, form, mu, sd, mlp=net)
    with torch.no_grad(): P = f.predict(Xte)
    return f, r2(Yte, P), dict(n_train_edges=int(len(Xtr)), n_test_edges=int(len(Xte)), D=int(X.shape[1]))


# ------------------------------------------------------------------------------------------------ evaluation
def batches_of(C, n, dev):
    from interp.part_core import sub
    return {c: [tensors(sub(J, np.arange(s, min(len(J['label']), s + 250))), n, dev) for s in range(0, len(J['label']), 250)] for c, J in C.items()}


def logits(M, TB, sub=None):
    import torch
    with torch.no_grad():
        return {c: torch.cat([M.run(B, sub=sub) for B in bl]).double().cpu().numpy() for c, bl in TB.items()}


def mcnemar(a, b):
    from scipy import stats
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    return dict(n10=n10, n01=n01, p=1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5))))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); out = [None] * m; run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


def ci(b): return [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]


class Stats:
    def __init__(self, y):
        self.y = y; n = len(y); rng = np.random.default_rng(50500)
        self.Wb = np.zeros((NB, n), np.float32)
        for b in range(NB): self.Wb[b] = np.bincount(rng.integers(0, n, n), minlength=n)
        self.sets = dict(WZ=(y == iW) | (y == iZ), WZQ=(y == iW) | (y == iZ) | (y == iQ))
        self.prng = np.random.default_rng(50501)

    def deltas(self, L, e):
        d = lambda X: X[:, iW] - X[:, iZ]
        return d(L[e]) - d(L['base']), d(L['dil_' + e]) - d(L['base'])

    def R(self, L, e, s):
        a, b = self.deltas(L, e); m = self.sets[s]; a, b = a[m], b[m]; W = self.Wb[:, m]
        return float((a * b).sum() / (b * b).sum()), (W @ (a * b)) / (W @ (b * b))

    def perm_dR(self, La, Lb, e, s):
        """paired permutation p for H0: R(a) = R(b): per jet the two arms' (Delta d shift, Delta d dilation) pairs are swapped
        with probability 1/2"""
        m = self.sets[s]; ap, ad = (v[m] for v in self.deltas(La, e)); bp, bd = (v[m] for v in self.deltas(Lb, e))
        obs = (ap * ad).sum() / (ad * ad).sum() - (bp * bd).sum() / (bd * bd).sum()
        S = self.prng.random((NPERM, m.sum())) < 0.5
        na = np.where(S, bp * bd, ap * ad).sum(1); da = np.where(S, bd * bd, ad * ad).sum(1)
        nb = np.where(S, ap * ad, bp * bd).sum(1); db = np.where(S, ad * ad, bd * bd).sum(1)
        null = na / da - nb / db
        return float(obs), float((1 + (np.abs(null) >= abs(obs)).sum()) / (NPERM + 1))

    def row(self, L, ref=None):
        y = self.y; ok = (L['base'].argmax(1) == y).astype(np.float64); ws = self.Wb.sum(1)
        cf = np.zeros((10, 10))
        for a, b in zip(y, L['base'].argmax(1)): cf[a, b] += 1
        cf = cf / np.maximum(cf.sum(1, keepdims=True), 1)
        out = dict(acc=float(ok.mean()), acc_ci=ci(self.Wb @ ok / ws), own_class=np.diag(cf).round(4).tolist(), confusion=cf.round(4).tolist())
        isW = y == iW; out['WtoZ_before'] = float((L['base'].argmax(1)[isW] == iZ).mean()); out['WtoZ_after_phi'] = float((L['phi'].argmax(1)[isW] == iZ).mean())
        rb = {}
        for e in ('phi', 'eta'):
            for s in ('WZ', 'WZQ'):
                r, b = self.R(L, e, s); out[f'R_{e}_{s}'] = r; out[f'R_{e}_{s}_ci'] = ci(b); rb[e, s] = b
            a, b = self.deltas(L, e); m = self.sets['WZ']
            out[f'dd_{e}_mean'] = float(a[m].mean()); out[f'dd_dil_{e}_mean'] = float(b[m].mean())
        out['R_phi_minus_eta_WZ'] = [out['R_phi_WZ'] - out['R_eta_WZ']] + ci(rb['phi', 'WZ'] - rb['eta', 'WZ'])
        if ref is not None:
            okr = (ref['base'].argmax(1) == y)
            out['dacc_vs_unedited'] = [float(ok.mean() - okr.mean())] + ci(self.Wb @ (ok - okr) / ws)
            out['mcnemar_vs_unedited'] = mcnemar(okr, ok > 0); out['calls_changed'] = float((L['base'].argmax(1) != ref['base'].argmax(1)).mean())
            out['comparable'] = bool(abs(ok.mean() - okr.mean()) <= 0.01)
            for e in ('phi', 'eta'):
                for s in ('WZ', 'WZQ'):
                    dR, p = self.perm_dR(L, ref, e, s); _, ba = self.R(L, e, s); _, bb = self.R(ref, e, s)
                    out[f'dR_{e}_{s}_vs_unedited'] = [dR] + ci(ba - bb); out[f'p_dR_{e}_{s}_vs_unedited'] = p
            _, a1 = self.R(L, 'phi', 'WZ'); _, a2 = self.R(L, 'eta', 'WZ'); _, b1 = self.R(ref, 'phi', 'WZ'); _, b2 = self.R(ref, 'eta', 'WZ')
            out['dir_dep_minus_unedited'] = [out['R_phi_minus_eta_WZ'][0] - (self.R(ref, 'phi', 'WZ')[0] - self.R(ref, 'eta', 'WZ')[0])] + ci((a1 - a2) - (b1 - b2))
        return out


# ------------------------------------------------------------------------------------------------ figure
def plot(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5)); lab = dict(A='A: x_i + pair invariants', B='B: A + signed Δη, Δφ', C='C: A + non-angular x_j − x_i',
                                                             N='null: shuffled neighbour', F='ceiling: exact inputs')
    cols = dict(A='#1f77b4', B='#d62728', C='#2ca02c', N='grey', F='k')
    for j, mname in enumerate(('PN_full', 'PN_kin')):
        a = ax[j]; FT = res['models'][mname]['fits']; xs = np.arange(4); w = 0.16
        for q, kd in enumerate(KINDS):
            v = [FT[f'ec{l}'][f'mlp-{kd}'][lev]['mean'] for l in (1, 2) for lev in ('edge', 'agg')]
            a.bar(xs + (q - 2) * w, np.maximum(v, -0.1), w, color=cols[kd], label=lab[kd])
        a.axhline(0.7, ls=':', c='k', lw=1); a.axhline(0, c='k', lw=0.6); a.set_xticks(xs)
        a.set_xticklabels(['EC1 per edge', 'EC1 mean\nover 16 neighbours', 'EC2 per edge', 'EC2 mean\nover 16 neighbours'], fontsize=8)
        a.set_ylim(-0.1, 1.25); a.set_ylabel('held-out R² of the edge message (mean over channels)')
        a.set_title(f'{mname}: how well each input set predicts the edge messages (MLP fit)', fontsize=9)
        a.legend(fontsize=7, loc='upper center', ncol=3)
    a = ax[2]; xs = np.arange(6); names = ['unedited'] + list(KINDS)
    for off, mname, c in ((-0.12, 'PN_full', '#ff7f0e'), (0.12, 'PN_kin', '#9467bd')):
        M = res['models'][mname]; rows = [M['unedited']] + [M['arms'][f'mlp-{kd}-ec1'] for kd in KINDS]
        y = [r['R_phi_WZ'] for r in rows]; e = np.array([[r['R_phi_WZ'] - r['R_phi_WZ_ci'][0], r['R_phi_WZ_ci'][1] - r['R_phi_WZ']] for r in rows]).T
        a.errorbar(xs + off, y, yerr=e, fmt='o', c=c, capsize=3, label=f'{mname} (unedited acc {M["unedited"]["acc"] * 100:.1f} %)')
        for x_, r in zip(xs[1:], rows[1:]):
            a.text(x_ + off, r['R_phi_WZ_ci'][0] - 0.03, f'{r["dacc_vs_unedited"][0] * 100:+.1f}', fontsize=7, ha='center', va='top', color=c)
        a.axhline(M['unedited']['R_phi_WZ'], c=c, lw=0.8, ls='-.')
    a.axhline(0.8, ls='--', c='k', lw=0.8); a.text(5.4, 0.81, 'ParT_full ≈ 0.8', fontsize=7, ha='right', va='bottom')
    a.set_xticks(xs); a.set_xticklabels(['unedited', 'A: invariants', 'B: + signed\nΔη, Δφ', 'C: + non-angular\ndifferences', 'null', 'ceiling'], fontsize=8)
    a.set_ylim(0, 0.95); a.set_ylabel('R (W, Z jets) = Δd(φ + 0.06) / Δd(matched dilation)')
    a.set_title('EdgeConv-1 messages replaced by the MLP fit, rest of ParticleNet intact\n(numbers: accuracy change, points)', fontsize=9)
    a.legend(fontsize=7, loc='upper left')
    fig.suptitle(f'P50: ParticleNet EdgeConv edge function (val_5M _123; fit N {res["N_fit"]} jets, evaluation N {res["N_eval"]} jets; MLP {res["STEPS"]} steps)', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG, dpi=130); log('figure', FIG)


# ------------------------------------------------------------------------------------------------ main
def main():
    import torch
    from interp import pn_net
    from interp.part_core import sample, build, kin, sub
    from jetdistill.part.network import FEATURES
    RES.mkdir(parents=True, exist_ok=True); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device {dev} {torch.cuda.get_device_name(0) if dev == "cuda" else ""} torch {torch.__version__}')
    JE, JF, info = load_samples(); C, chk = edits(JE); log('edit check', chk)
    rng = np.random.default_rng(5051); perm = rng.permutation(len(JF['label'])); train_jets = np.sort(perm[: int(0.8 * len(perm))])
    J1 = sample(NP1, 3000); J1 = build(J1, *kin(J1))
    res = dict(node='P50', smoke=SMOKE, N_eval=int(len(JE['label'])), N_fit=int(len(JF['label'])), files=info, edit_check=chk,
               KEEPC=KEEPC, NB=NB, NPERM=NPERM, STEPS=STEPS, models={}, controls={})
    st = Stats(JE['label'].astype(int))
    for mname, n in (('PN_full', 'full'), ('PN_kin', 'kin')):
        nf = len(FEATURES[n]); net = pn_net.load_particlenet(PNDIR / f'ParticleNet_{n}.pt', nf, dev)
        for p in net.parameters(): p.requires_grad_(False)
        M = Model(mname, n, net, dev); R = dict()
        # weaver check
        from weaver.nn.model.ParticleNet import ParticleNet as WPN
        w = WPN(input_dims=nf, **pn_net.PN_CFG); sd = torch.load(PNDIR / f'ParticleNet_{n}.pt', map_location='cpu')
        w.load_state_dict({k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}); w = w.to(dev).eval()
        B = tensors(sub(J1, np.arange(0, len(J1['label']), max(1, len(J1['label']) // 300))), n, dev)
        with torch.no_grad(): d = float((M.run(B) - w(B['p'].clone(), B['x'].clone(), B['m'])).abs().max())
        R['weaver_check_max_abs_dlogit'] = d; log(f'{mname}: own forward vs weaver max |dlogit| {d:.2e}'); assert d <= 1e-5, d
        TB1 = batches_of(dict(base=J1), n, dev); L1 = logits(M, TB1)['base']; R['acc_P1'] = float((L1.argmax(1) == J1['label']).mean())
        TB = batches_of(C, n, dev); L0 = logits(M, TB); R['unedited'] = st.row(L0)
        log(f'{mname}: acc P1 {R["acc_P1"]:.4f}, eval {R["unedited"]["acc"]:.4f}; R_phi(W/Z) {R["unedited"]["R_phi_WZ"]:.3f} '
            f'(W/Z/QCD {R["unedited"]["R_phi_WZQ"]:.3f}); R_eta(W/Z) {R["unedited"]["R_eta_WZ"]:.3f}')
        # step 1 and 2
        t1 = time.time(); D = record(M, JF, train_jets)
        log(f'{mname}: recorded edges ({time.time() - t1:.0f} s): EC1 {D[0]["y"].shape}, EC2 {D[1]["y"].shape}, train centres {int(D[0]["train"].sum())}')
        R['fits'] = {}; FITS = {}
        for li in (0, 1):
            R['fits'][f'ec{li + 1}'] = {}
            for form in FORMS:
                for kd in KINDS:
                    t1 = time.time(); f, rr, meta = fit_one(D[li], kd, form, 5054 + 10 * li + KINDS.index(kd)); FITS[li, form, kd] = f
                    rr.update(meta); R['fits'][f'ec{li + 1}'][f'{form}-{kd}'] = rr
                    log(f'  {mname} EC{li + 1} {form:5s} {kd}: R2 edge mean {rr["edge"]["mean"]:.3f} pooled {rr["edge"]["pooled"]:.3f} '
                        f'frac>=0.7 {rr["edge"]["frac_ge_0p7"]:.2f} | agg mean {rr["agg"]["mean"]:.3f} pooled {rr["agg"]["pooled"]:.3f} '
                        f'(live {rr["edge"]["n_live"]}/{rr["edge"]["n_ch"]}, D {meta["D"]}, {time.time() - t1:.0f} s)')
        del D; torch.cuda.empty_cache()
        # step 3
        R['arms'] = {}; ARML = {}
        for form in FORMS:
            for L in ('ec1', 'ec12'):
                for kd in KINDS:
                    nm = f'{form}-{kd}-{L}'; sub_ = {0: FITS[0, form, kd]} if L == 'ec1' else {0: FITS[0, form, kd], 1: FITS[1, form, kd]}
                    La = logits(M, TB, sub_); ARML[nm] = La; r = st.row(La, ref=L0); R['arms'][nm] = r
                    log(f'  {mname} {nm:14s} acc {r["acc"]:.4f} ({r["dacc_vs_unedited"][0] * 100:+.2f} pts, McNemar p {r["mcnemar_vs_unedited"]["p"]:.2g}) '
                        f'R_phi {r["R_phi_WZ"]:.3f} [{r["R_phi_WZ_ci"][0]:.3f}, {r["R_phi_WZ_ci"][1]:.3f}] dR {r["dR_phi_WZ_vs_unedited"][0]:+.3f} p {r["p_dR_phi_WZ_vs_unedited"]:.3g} '
                        f'R_eta {r["R_eta_WZ"]:.3f} comparable {r["comparable"]}')
        names = list(R['arms'])
        R['holm_mcnemar'] = dict(zip(names, holm([R['arms'][k]['mcnemar_vs_unedited']['p'] for k in names])))
        cmp_ = [k for k in names if R['arms'][k]['comparable']]
        R['holm_dR_phi_WZ'] = dict(zip(cmp_, holm([R['arms'][k]['p_dR_phi_WZ_vs_unedited'] for k in cmp_]))) if cmp_ else {}
        # between-arm comparisons B - A, C - A, N - A (same form and layer set)
        R['between'] = {}
        for form in FORMS:
            for L in ('ec1', 'ec12'):
                a = f'{form}-A-{L}'
                for kd in ('B', 'C', 'N'):
                    b = f'{form}-{kd}-{L}'; ok = abs(R['arms'][a]['acc'] - R['arms'][b]['acc']) <= 0.01
                    dR, p = st.perm_dR(ARML[b], ARML[a], 'phi', 'WZ'); _, ba = st.R(ARML[b], 'phi', 'WZ'); _, bb = st.R(ARML[a], 'phi', 'WZ')
                    R['between'][f'{b} - {a}'] = dict(dR_phi_WZ=[dR] + ci(ba - bb), p=p, comparable=bool(ok),
                                                      dacc=float(R['arms'][b]['acc'] - R['arms'][a]['acc']))
        bk = [k for k, v in R['between'].items() if v['comparable']]
        for k, h in zip(bk, holm([R['between'][k]['p'] for k in bk])): R['between'][k]['holm'] = h
        res['models'][mname] = R; json.dump(res, open(FN, 'w'), indent=1, default=float)
        # controls on PN_full: untrained and weight-permuted networks
        if n == 'full':
            torch.manual_seed(5050); un = pn_net.ParticleNet(input_dims=nf, **pn_net.PN_CFG).to(dev).eval()
            pm = pn_net.load_particlenet(PNDIR / 'ParticleNet_full.pt', nf, dev); g = torch.Generator().manual_seed(5053)
            with torch.no_grad():
                for p in pm.parameters():
                    if p.numel() > 1: p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g).to(dev)].view_as(p))
            for cn, cnet in (('untrained', un), ('permuted', pm)):
                for p in cnet.parameters(): p.requires_grad_(False)
                Mc = Model(cn, n, cnet, dev); Lc = logits(Mc, TB); rc = st.row(Lc, ref=L0); D = record(Mc, JF, train_jets); rc['fits'] = {}
                for form in FORMS:
                    for kd in KINDS:
                        _, rr, meta = fit_one(D[0], kd, form, 5154 + KINDS.index(kd)); rr.update(meta); rc['fits'][f'ec1-{form}-{kd}'] = rr
                        log(f'  control {cn} EC1 {form:5s} {kd}: R2 edge mean {rr["edge"]["mean"]:.3f} agg mean {rr["agg"]["mean"]:.3f} (live {rr["edge"]["n_live"]})')
                del D; torch.cuda.empty_cache()
                log(f'control {cn}: acc {rc["acc"]:.4f} R_phi(W/Z) {rc["R_phi_WZ"]:.3f} R_eta {rc["R_eta_WZ"]:.3f}')
                res['controls'][cn] = rc; json.dump(res, open(FN, 'w'), indent=1, default=float)
        del TB, net, M; torch.cuda.empty_cache()
    json.dump(res, open(FN, 'w'), indent=1, default=float); plot(res); log('done', FN)


if __name__ == '__main__':
    if '--plot' in sys.argv: plot(json.load(open(FN)))
    else: main()
