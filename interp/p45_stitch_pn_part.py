"""P45 - model stitching ParticleNet_full -> ParT_full (loop C, round C10, exploratory).

Question: does ParT_full accept ParticleNet_full's internal per-particle features (through a learned affine map) in place of its
own tokens, and does the response of the stitched network to a mass-fixed rigid phi shift follow the donor (ParticleNet) or the
recipient (ParT)?

Models (frozen): ParT_full (released, JETDISTILL_MODELS/ParT_full.pt, weaver-core 0.4.17 ParticleTransformer) and ParticleNet_full
(released, /j-jepa-vol/aaron-interp/models/particlenet/ParticleNet_full.pt, interp/pn_net.py). Only the affine maps are trained.

Stitch (diagram):
    jet --> PN: bn -> EdgeConv1 (64/particle) -> EdgeConv2 (128) -> EdgeConv3 (256) -> mean pool (256) -> fc -> PN logits
                          |  ec1            |  ec2              |  ec3                |  pool
                          v  A: affine per particle (d -> 128), padded slots set to 0     v  A: 256 -> 128
    jet --> ParT: embed -> block 1 -> ... -> block k-1 -X-> [A(PN features)] -> block k -> ... -> block 8 -> 2 class blocks -> norm -> head
                 (pair bias U(kT, z, dR, m^2) of every block computed from the real jet; padding mask kept)      [A(pool)] -^ (S2)
  S1: source ec1 / ec2 / ec3 per-particle output, target = input of ParT particle block k (k = 1, 3, 5, 8; block 1 input = the
      embedding output). S2: source PN's pooled 256-d vector (input of its fc), target = ParT's final class-token vector (the norm
      output, the input of ParT's last Linear). 12 S1 + 1 S2 = 13 stitch points.
Training: A initialised by least squares (ridge 1e-4 x mean diagonal) from the source to ParT's own target activations on the
  first NLS jets of the training order, then trained to minimise KL(p_ParT || p_stitched) (ParT's own softmax on the same jet as
  the target; KL = sum_c p_ParT,c log(p_ParT,c / p_stitched,c)) for one pass over 300,000 JetClass training jets (train_100M,
  2 random files per class, 15,000 random rows each, default_rng(4545); never first-k), batches of 256 grouped by multiplicity,
  Adam lr 5e-4 with 3 % warm-up and cosine decay.
Controls: (a) untrained ParticleNet (same architecture, PyTorch default init, torch seed 4545, BatchNorm at default statistics) at
  the same 13 points; (b) raw input: affine from ParT's own 17 scaled input features per particle into block k (floor);
  (c) identity: affine from ParT's own block-k input (initialised at identity) into block k (ceiling and code check);
  (d) a second training seed for the 13 trained-PN points (other batch composition / order and other least-squares jets).
Evaluation samples: P1 (rows 3,000-4,000 of each class's test_20M *_100 file, N 10,000, part_core.sample, rebuilt) and a fresh
  val_5M *_121 sample (1,000 random rows per class, default_rng(4546), N 10,000, rebuilt with part_core.build).
Edits on the evaluation jets: phi + 0.06 rad on every real particle (rigid rotation about the beam: invariant mass and every
  pairwise feature unchanged); matched dilation: all (deta, dphi) x k_j with k_j = sqrt(M0M2(shifted)/M0M2(unedited)) per jet
  (a real mass change of the size the axis moment predicts for the shift; M0 = sum pT, M2 = sum pT r^2 about the stored axis).
Metrics: accuracy, 10-class confusion; stitching gap = acc(ParT) - acc(stitched) with paired 95 % bootstrap over jets (1,000
  resamples) and exact McNemar on per-jet correctness (ParT vs stitched; trained vs untrained PN); Holm over the 13 points.
  W -> Z: fraction of true-W jets called Z before / after phi + 0.06 (McNemar). R = slope through 0 over true W, Z and QCD jets of
  the per-jet change of d = l_W - l_Z under the shift against its change under the matched dilation (as P22/P22v); bootstrap
  p for R(stitched) - R(PN alone) and - R(ParT alone), Holm over the 13 points.
  ./sync.sh && ./part_job.sh c-p45-run 'python -m interp.p45_stitch_pn_part --smoke && python -m interp.p45_stitch_pn_part'
  -> $AI_RESULTS/part/p45_stitch[_smoke].json, p45_stitch[_smoke].png
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SMOKE = '--smoke' in sys.argv
TAG = '_smoke' if SMOKE else ''
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN, FIG = RES / f'p45_stitch{TAG}.json', RES / f'p45_stitch{TAG}.png'
PNW = Path('/j-jepa-vol/aaron-interp/models/particlenet/ParticleNet_full.pt')
TRAIN = Path('/j-jepa-vol/JetClass/Pythia/train_100M')
VAL = Path('/j-jepa-vol/JetClass/Pythia/val_5M')
NTR_FILE = 300 if SMOKE else 15000          # per file, 2 files per class
NEV = 100 if SMOKE else 1000                # per class, _121 sample
NP1 = 100 if SMOKE else 1000                # per class, P1 sample
NLS = 1000 if SMOKE else 20000              # jets for the least-squares initialisation
NB = 200 if SMOKE else 1000
BS = 256
LR = 5e-4
KS = (1, 3, 5, 8)
SHIFT = 0.06
iZ, iW, iQ = 6, 7, 0
KEEP = ['px', 'py', 'pz', 'energy', 'deta', 'dphi', 'd0val', 'd0err', 'dzval', 'dzerr', 'charge', 'isChargedHadron', 'isNeutralHadron',
        'isPhoton', 'isElectron', 'isMuon', 'mask', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'label']
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ data
def load_rows(path, n, rng, chunk=25000):
    from interp.part_core import read_root, sub, cat
    import uproot
    N = uproot.open(path)['tree'].num_entries
    if SMOKE: N = min(N, 20000)
    rows = np.sort(rng.choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r):
            J = read_root(path, start=s, stop=min(s + chunk, N)); parts.append({k: J[k][r - s] for k in KEEP})
    return cat(parts), dict(file=str(path), rows_in_file=int(N), drawn=int(n))


def rebuild(J):
    from interp.part_core import build, kin
    return build(J, *kin(J))


def train_jets():
    from interp.part_core import PRE, CL, cat
    rng = np.random.default_rng(4545); parts, info = [], []
    for c in CL:
        for fi in rng.choice(100, 2, replace=False):
            J, inf = load_rows(TRAIN / f'{PRE[c]}_{fi:03d}.root', NTR_FILE, rng); parts.append(J); info.append(inf)
            log(f'  train {Path(inf["file"]).name}: {inf["drawn"]} random rows of {inf["rows_in_file"]}')
    return cat(parts), info


def eval_samples():
    from interp.part_core import PRE, CL, cat, sample, sub, kin, build, moments
    S = {}
    J1 = sample(NP1, 3000); S['P1'] = dict(J=rebuild({k: J1[k] for k in KEEP}), info='test_20M *_100 rows 3000-%d per class (P1 sample)' % (3000 + NP1))
    rng = np.random.default_rng(4546); parts = []
    for c in CL:
        J, _ = load_rows(VAL / f'{PRE[c]}_121.root', NEV, rng); parts.append(J)
    S['v121'] = dict(J=rebuild(cat(parts)), info=f'val_5M *_121, {NEV} random rows per class, default_rng(4546)')
    for nm, s in S.items():
        J = s['J']; pt, de, dp, ms = kin(J); real = pt > 0
        Jp = build(J, pt, de, dp + SHIFT * real, ms)
        Mb = moments(*kin(J)[:3])['M0M2']; Mp = moments(*kin(Jp)[:3])['M0M2']; ok = Mb > 0
        kj = np.where(ok, np.sqrt(Mp / np.where(ok, Mb, 1.0)), 1.0)[:, None]
        Jd = build(J, pt, de * kj, dp * kj, ms)
        s['cond'] = dict(base=J, phi=Jp, dil=Jd)
        from interp.part_core import jmass
        m0, mp, md = (np.log(np.maximum(jmass(x), 1.0)) for x in (J, Jp, Jd))
        s['edit_check'] = dict(phi_dlogm_absmax=float(np.abs(mp - m0).max()), dil_dlogm_mean=float((md - m0).mean()),
                               k_mean=float(kj.mean()), mask_equal=bool(np.array_equal(Jp['mask'], J['mask']) and np.array_equal(Jd['mask'], J['mask'])))
        log(f'sample {nm}: N {len(J["label"])}; edit check {s["edit_check"]}')
    return S


# ------------------------------------------------------------------------------------------------ networks
class Nets:
    def __init__(self):
        import torch
        from jetdistill.part.network import ParTNetwork, FEATURES
        from interp import pn_net
        self.t = torch; self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.environ.setdefault('JETDISTILL_DEVICE', self.dev)
        self.part = ParTNetwork('full').model.to(self.dev).eval()
        nf = len(FEATURES['full']); names = [f[0] for f in FEATURES['full']]; self.ipts = [names.index('deta'), names.index('dphi')]
        self.pn = pn_net.load_particlenet(PNW, nf, self.dev).eval()
        torch.manual_seed(4545); self.upn = pn_net.ParticleNet(input_dims=nf, **pn_net.PN_CFG).to(self.dev).eval()
        for m in (self.part, self.pn, self.upn):
            for p in m.parameters(): p.requires_grad_(False)

    def tensors(self, Jb):
        from interp.part_core import inputs
        t = self.t; x, v, m = inputs(Jb, 'full'); P = x.shape[-1]
        xp = np.zeros((len(x), x.shape[1], 128), np.float32); xp[..., :P] = x; mp = np.zeros((len(x), 1, 128), np.float32); mp[..., :P] = m
        T = lambda a: t.from_numpy(np.ascontiguousarray(a)).to(self.dev)
        return dict(x=T(x), v=T(v), m=T(m), xp=T(xp), pp=T(xp[:, self.ipts]), mp=T(mp), P=P)

    def pn_feats(self, model, B):
        """ParticleNet forward (use_fusion False) keeping each EdgeConv output (N, C, 128) and the pooled vector"""
        mask = B['mp']; points = B['pp'] * mask; feats = B['xp'] * mask; shift = (mask == 0) * 1e9
        counts = self.t.clamp(mask.sum(-1), min=1.0); fts = model.bn_fts(feats) * mask; outs = []
        for i, conv in enumerate(model.edge_convs):
            fts = conv((points if i == 0 else fts) + shift, fts) * mask; outs.append(fts)
        pooled = fts.sum(-1) / counts
        return outs, pooled, model.fc(pooled)

    def part_pre(self, B):
        md = self.part; mask = B['m'].bool(); pad = ~mask.squeeze(1); P = B['P']
        h = md.embed(B['x']).masked_fill(~mask.permute(2, 0, 1), 0)
        am = md.pair_embed(B['v'], None).view(-1, P, P)
        return h, am, pad

    def part_from(self, h, am, pad, k, collect=False):
        """ParT from the input of particle block k (1..8; k = 9: input of the class blocks) to the logits"""
        md = self.part; hs = []
        for i in range(k - 1, 8):
            if collect: hs.append(h)
            h = md.blocks[i](h, x_cls=None, padding_mask=pad, attn_mask=am)
        if collect: hs.append(h)
        cls = md.cls_token.expand(1, h.size(1), -1)
        for b in md.cls_blocks: cls = b(h, x_cls=cls, padding_mask=pad)
        xc = md.norm(cls).squeeze(0)
        return md.fc(xc), xc, hs


def stitch_list():
    L = []
    for seed in (0, 1):
        for src in ('pn', 'u') if seed == 0 else ('pn',):
            for e in (1, 2, 3):
                for k in KS: L.append(dict(name=f'{src}{e}>b{k}' + ('' if seed == 0 else '_s1'), src=f'{src}{e}', k=k, seed=seed))
            L.append(dict(name=f'{src}pool>cls' + ('' if seed == 0 else '_s1'), src=f'{src}pool', k='cls', seed=seed))
        if seed == 0:
            for k in KS: L.append(dict(name=f'raw>b{k}', src='raw', k=k, seed=0))
            for k in KS: L.append(dict(name=f'id>b{k}', src='id', k=k, seed=0))
    return L


DIMS = dict(pn1=64, pn2=128, pn3=256, u1=64, u2=128, u3=256, pnpool=256, upool=256, raw=17, id=128)


class Ctx:
    """per-batch no-grad quantities shared by all stitches"""

    def __init__(self, N, B, need):
        t = N.t
        with t.no_grad():
            self.B = B; self.P = B['P']; self.h0, self.am, self.pad = N.part_pre(B)
            self.src = {}
            if any(s.startswith('pn') for s in need):
                o, pool, self.pnL = N.pn_feats(N.pn, B); self.src.update(pn1=o[0], pn2=o[1], pn3=o[2], pnpool=pool)
            if any(s.startswith('u') for s in need):
                o, pool, _ = N.pn_feats(N.upn, B); self.src.update(u1=o[0], u2=o[1], u3=o[2], upool=pool)
            self.src['raw'] = B['x']
            if 'id' in need or 'target' in need:
                self.L, self.xc, hs = N.part_from(self.h0, self.am, self.pad, 1, collect=True); self.hs = hs   # hs[k-1] = input of block k

    def source(self, s, k):
        """(P, N, d) per-particle source or (N, d) pooled"""
        if s.endswith('pool'): return self.src[s]
        if s == 'id': return self.hs[k - 1]
        a = self.src[s][..., :self.P] if s != 'raw' else self.src[s]
        return a.permute(2, 0, 1)

    def run(self, N, st, A):
        if st['k'] == 'cls':
            return N.part.fc(A(self.source(st['src'], None)))
        tok = A(self.source(st['src'], st['k'])).masked_fill(self.pad.t().unsqueeze(-1), 0)
        return N.part_from(tok, self.am, self.pad, st['k'])[0]


def batches(mult, seed, bs=BS):
    rng = np.random.default_rng(45450 + seed); key = rng.random(len(mult))
    o = np.lexsort((key, mult)); bl = [o[i:i + bs] for i in range(0, len(o), bs)]
    return [bl[i] for i in rng.permutation(len(bl))]


def fit_init(N, JT, sts, order):
    """least-squares (ridge) affine maps from each source to ParT's own activation at the target, on the first NLS jets of `order`"""
    from interp.part_core import sub
    t = N.t; G, C = {}, {}; n = 0; need = {st['src'] for st in sts} | {'target'}
    with t.no_grad():
        for idx in order:
            if n >= NLS: break
            cx = Ctx(N, N.tensors(sub(JT, idx)), need); n += len(idx); real = ~cx.pad.t(); done = set()          # real: (P, N)
            for st in sts:
                if st['src'] == 'id': continue
                k = st['k']; X = cx.source(st['src'], k)
                if k == 'cls': Xr, Y = X, cx.xc
                else: Xr, Y = X[real], cx.hs[k - 1][real]
                Xr = t.cat([Xr.double(), t.ones(len(Xr), 1, dtype=t.float64, device=Xr.device)], 1)
                key = (st['src'], k)
                if (st['src'], k == 'cls') not in done:
                    G[st['src'], k == 'cls'] = G.get((st['src'], k == 'cls'), 0) + Xr.T @ Xr; done.add((st['src'], k == 'cls'))
                C[key] = C.get(key, 0) + Xr.T @ Y.double()
    W = {}
    for st in sts:
        if st['src'] == 'id': continue
        g = G[st['src'], st['k'] == 'cls']; lam = 1e-4 * g.diagonal().mean()
        W[st['name']] = t.linalg.solve(g + lam * t.eye(len(g), dtype=g.dtype, device=g.device), C[st['src'], st['k']])
    return W, n


def make_affine(N, st, Winit):
    t = N.t; d = DIMS[st['src']]; A = t.nn.Linear(d, 128).to(N.dev)
    with t.no_grad():
        if st['src'] == 'id': A.weight.copy_(t.eye(128)); A.bias.zero_()
        else:
            w = Winit[st['name']]; A.weight.copy_(w[:-1].T.float()); A.bias.copy_(w[-1].float())
    return A


def train(N, JT, TL, sts, seed):
    """one pass over the training jets for the stitches of one seed (shared batches); returns the trained affine maps"""
    from interp.part_core import sub
    t = N.t; F = t.nn.functional; mult = JT['mask'].sum(1).astype(int); order = batches(mult, seed)
    t1 = time.time(); Winit, nls = fit_init(N, JT, sts, order); log(f'seed {seed}: least-squares init on {nls} jets ({time.time() - t1:.0f} s)')
    As = {st['name']: make_affine(N, st, Winit) for st in sts}
    opts = {k: t.optim.Adam(A.parameters(), lr=LR) for k, A in As.items()}
    nst = len(order); warm = max(1, int(0.03 * nst)); lrf = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, nst - warm)))
    scheds = {k: t.optim.lr_scheduler.LambdaLR(o, lrf) for k, o in opts.items()}
    need = {st['src'] for st in sts}; run = {k: [] for k in As}; hist = {k: [] for k in As}; t1 = time.time()
    for s, idx in enumerate(order):
        cx = Ctx(N, N.tensors(sub(JT, idx)), need); tl = F.log_softmax(t.from_numpy(TL[idx]).to(N.dev), 1)
        for st in sts:
            k = st['name']; L = cx.run(N, st, As[k]); loss = F.kl_div(F.log_softmax(L, 1), tl, log_target=True, reduction='batchmean')
            opts[k].zero_grad(set_to_none=True); loss.backward(); opts[k].step(); scheds[k].step(); run[k].append(float(loss))
        if (s + 1) % max(1, nst // 10) == 0 or s == nst - 1:
            for k in As: hist[k].append([s + 1, float(np.mean(run[k]))]); run[k] = []
            el = time.time() - t1
            log(f'seed {seed}: step {s + 1}/{nst} ({el:.0f} s, eta {el / (s + 1) * (nst - s - 1):.0f} s); KL ' +
                ', '.join(f'{k} {hist[k][-1][1]:.3f}' for k in list(As)[:: max(1, len(As) // 8)]))
    for A in As.values(): A.eval()
    return As, hist


def teacher(N, JT):
    from interp.part_core import sub
    t = N.t; mult = JT['mask'].sum(1).astype(int); o = np.argsort(mult, kind='stable'); TL = np.zeros((len(o), 10), np.float32)
    with t.no_grad():
        for i in range(0, len(o), 512):
            idx = o[i:i + 512]; B = N.tensors(sub(JT, idx)); h, am, pad = N.part_pre(B)
            TL[idx] = N.part_from(h, am, pad, 1)[0].float().cpu().numpy()
    return TL


def evaluate(N, S, sts, As):
    """logits of ParT, PN and every stitch on each condition of the sample (N, 10)"""
    from interp.part_core import sub
    t = N.t; Jb = S['cond']['base']; n = len(Jb['label']); mult = Jb['mask'].sum(1).astype(int); o = np.argsort(mult, kind='stable')
    names = ['ParT', 'PN'] + [st['name'] for st in sts]
    out = {c: {k: np.zeros((n, 10)) for k in names} for c in S['cond']}; need = {st['src'] for st in sts} | {'id', 'pn'}
    with t.no_grad():
        for i in range(0, n, 256):
            idx = o[i:i + 256]
            for c, J in S['cond'].items():
                cx = Ctx(N, N.tensors(sub(J, idx)), need)
                out[c]['ParT'][idx] = cx.L.double().cpu().numpy(); out[c]['PN'][idx] = cx.pnL.double().cpu().numpy()
                for st in sts: out[c][st['name']][idx] = cx.run(N, st, As[st['name']]).double().cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------------ statistics
def mcnemar(a, b):
    from scipy import stats
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    return dict(n10=n10, n01=n01, p=1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5))))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); out = [None] * m; run = 0.0
    for r, i in enumerate(order): run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


def ci(b): return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def bp(d):
    """two-sided bootstrap p for H0: statistic = 0 from bootstrap replicates d"""
    d = np.asarray(d); return float(max(1.0 / len(d), min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def metrics(S, out, sts):
    y = S['cond']['base']['label'].astype(int); n = len(y); rng = np.random.default_rng(45460)
    Wb = np.zeros((NB, n), np.float32)
    for b in range(NB): Wb[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    isW = y == iW; wz = (y == iW) | (y == iZ) | (y == iQ); wsum = Wb.sum(1)
    sl = lambda yv, xv: (float((yv * xv).sum() / (xv * xv).sum()), (Wb[:, wz] @ (yv * xv)) / (Wb[:, wz] @ (xv * xv)))
    d = lambda L: L[:, iW] - L[:, iZ]
    base = {}; R = {}
    for k in ['ParT', 'PN'] + [st['name'] for st in sts]:
        L0, Lp, Ld = (out[c][k] for c in ('base', 'phi', 'dil'))
        ok = (L0.argmax(1) == y).astype(np.float64); dp, dd = (d(Lp) - d(L0))[wz], (d(Ld) - d(L0))[wz]
        r, rb = sl(dp, dd); c0, c1 = L0.argmax(1)[isW] == iZ, Lp.argmax(1)[isW] == iZ
        cf = np.zeros((10, 10))
        for a, b in zip(y, L0.argmax(1)): cf[a, b] += 1
        cf = cf / np.maximum(cf.sum(1, keepdims=True), 1)
        row = dict(acc=float(ok.mean()), acc_ci=ci(Wb @ ok / wsum), R=r, R_ci=ci(rb), dd_phi=float(dp.mean()), dd_dil=float(dd.mean()),
                   WtoZ_before=float(c0.mean()), WtoZ_after=float(c1.mean()), WtoZ_mcnemar=mcnemar(c0, c1),
                   own_class=np.diag(cf).tolist(), confusion=cf.round(4).tolist())
        base[k] = dict(ok=ok, L0=L0, rb=rb); R[k] = row
    P = base['ParT']; PN = base['PN']; cP = P['L0'].argmax(1)
    for k in [st['name'] for st in sts]:
        b, row = base[k], R[k]; g = (Wb @ (P['ok'] - b['ok'])) / wsum
        row.update(gap=float(P['ok'].mean() - b['ok'].mean()), gap_ci=ci(g), mcnemar_vs_ParT=mcnemar(P['ok'] > 0, b['ok'] > 0),
                   agree_ParT=float((b['L0'].argmax(1) == cP).mean()),
                   KL_ParT=float(np.mean(np.sum(_sm(P['L0']) * (_ls(P['L0']) - _ls(b['L0'])), 1))),
                   R_minus_PN=[float(row['R'] - R['PN']['R'])] + ci(b['rb'] - PN['rb']), p_R_vs_PN=bp(b['rb'] - PN['rb']),
                   R_minus_ParT=[float(row['R'] - R['ParT']['R'])] + ci(b['rb'] - P['rb']), p_R_vs_ParT=bp(b['rb'] - P['rb']))
        u = k.replace('pn', 'u', 1) if k.startswith('pn') and not k.endswith('_s1') else None
        if u in base:
            row['mcnemar_vs_untrained'] = mcnemar(base[u]['ok'] > 0, b['ok'] > 0)
            row['acc_minus_untrained'] = [float(b['ok'].mean() - base[u]['ok'].mean())] + ci((Wb @ (b['ok'] - base[u]['ok'])) / wsum)
    main = [st['name'] for st in sts if st['src'].startswith('pn') and st['seed'] == 0]
    for key, get in (('holm_gap', lambda r: r['mcnemar_vs_ParT']['p']), ('holm_vs_untrained', lambda r: r['mcnemar_vs_untrained']['p']),
                     ('holm_R_vs_PN', lambda r: r['p_R_vs_PN']), ('holm_R_vs_ParT', lambda r: r['p_R_vs_ParT'])):
        for k, p in zip(main, holm([get(R[k]) for k in main])): R[k][key] = p
    for k in [st['name'] for st in sts] + ['ParT', 'PN']:
        R[k]['WtoZ_mcnemar_p'] = R[k]['WtoZ_mcnemar']['p']
    return R


def _ls(L):
    L = L - L.max(1, keepdims=True); return L - np.log(np.exp(L).sum(1, keepdims=True))


def _sm(L): return np.exp(_ls(L))


# ------------------------------------------------------------------------------------------------ figure
def plot(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    M = res['samples']['v121']['models']; fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    xs = list(KS) + [10]; lab = [f'block {k}' for k in KS] + ['class token\n(pooled)']
    cols = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
    for j, (key, yl) in enumerate((('acc', 'accuracy (10 classes) [%]'), ('R', 'R = d-response to phi+0.06 / to matched dilation'))):
        a = ax[j]; sc = 100 if key == 'acc' else 1
        for e in (1, 2, 3):
            a.plot(KS, [M[f'pn{e}>b{k}'][key] * sc for k in KS], 'o-', c=cols[e], label=f'ParticleNet EdgeConv-{e} -> ParT')
            a.plot(KS, [M[f'u{e}>b{k}'][key] * sc for k in KS], 'x--', c=cols[e], alpha=0.6, label=f'untrained PN EdgeConv-{e}')
            a.plot(KS, [M[f'pn{e}>b{k}_s1'][key] * sc for k in KS], '.', c=cols[e], ms=4)
        a.plot([10], [M['pnpool>cls'][key] * sc], 's', c='k', label='PN pooled -> ParT class token')
        a.plot([10], [M['upool>cls'][key] * sc], 'x', c='grey', label='untrained PN pooled')
        a.plot(KS, [M[f'raw>b{k}'][key] * sc for k in KS], 'v:', c='grey', label='raw 17 inputs -> ParT (floor)')
        a.plot(KS, [M[f'id>b{k}'][key] * sc for k in KS], '^-', c='purple', alpha=0.6, label='ParT own tokens (ceiling)')
        a.axhline(M['ParT'][key] * sc, c='purple', lw=1, ls='-.', label='ParT_full alone'); a.axhline(M['PN'][key] * sc, c='brown', lw=1, ls='-.', label='ParticleNet_full alone')
        a.set_xticks(xs); a.set_xticklabels(lab, fontsize=8); a.set_ylabel(yl); a.set_xlabel('where ParticleNet features enter ParT')
        if key == 'acc': a.set_ylim(max(0, min(M[f'raw>b{k}']['acc'] for k in KS) * 100 - 5), 90)
    ax[0].legend(fontsize=6.5, loc='lower left', ncol=2)
    fig.suptitle(f'P45: ParticleNet_full features through an affine map into ParT_full (val_5M _121, N {res["samples"]["v121"]["N"]})', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG, dpi=130); log('figure', FIG)


# ------------------------------------------------------------------------------------------------ main
def main():
    import torch
    from interp.part_core import sub
    RES.mkdir(parents=True, exist_ok=True); torch.manual_seed(4545)
    N = Nets(); log(f'device {N.dev}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')
    res = dict(node='P45', smoke=SMOKE, NB=NB, BS=BS, LR=LR, KS=list(KS), SHIFT=SHIFT, NLS=NLS)
    S = eval_samples()
    # identity checks: own ParT forward vs weaver forward; own PN forward vs weaver forward
    J = sub(S['P1']['cond']['base'], np.arange(0, len(S['P1']['cond']['base']['label']), max(1, len(S['P1']['cond']['base']['label']) // 300)))
    B = N.tensors(J)
    with torch.no_grad():
        h, am, pad = N.part_pre(B); L1 = N.part_from(h, am, pad, 1)[0]; L2 = N.part(B['x'], B['v'], B['m'])
        _, _, P1 = N.pn_feats(N.pn, B); P2 = N.pn(B['pp'].clone(), B['xp'].clone(), B['mp'])
    res['identity_check'] = dict(part_max_abs_dlogit=float((L1 - L2).abs().max()), pn_max_abs_dlogit=float((P1 - P2).abs().max()), n=len(J['label']))
    log('identity check', res['identity_check'])
    t1 = time.time(); JT, info = train_jets(); res['train'] = dict(files=info, N=int(len(JT['label'])), per_class=np.bincount(JT['label'].astype(int), minlength=10).tolist())
    log(f'training jets {len(JT["label"])} loaded ({time.time() - t1:.0f} s)')
    t1 = time.time(); TL = teacher(N, JT); res['train']['ParT_acc_train'] = float((TL.argmax(1) == JT['label']).mean())
    log(f'teacher logits ({time.time() - t1:.0f} s); ParT accuracy on the training jets {res["train"]["ParT_acc_train"]:.4f}')
    sts = stitch_list(); As = {}; res['train']['history'] = {}
    for seed in (0, 1):
        ss = [st for st in sts if st['seed'] == seed]; A, hist = train(N, JT, TL, ss, seed); As.update(A); res['train']['history'].update(hist)
        res['stitches'] = sts; json.dump(res, open(FN, 'w'), indent=1, default=float)
    del JT
    res['samples'] = {}
    for nm, s in S.items():
        t1 = time.time(); out = evaluate(N, s, sts, As); M = metrics(s, out, sts)
        res['samples'][nm] = dict(info=s['info'], N=int(len(s['cond']['base']['label'])), edit_check=s['edit_check'], models=M)
        log(f'== sample {nm} ({time.time() - t1:.0f} s): ParT acc {M["ParT"]["acc"]:.4f} R {M["ParT"]["R"]:.3f} W->Z {M["ParT"]["WtoZ_before"]:.3f}->{M["ParT"]["WtoZ_after"]:.3f}; '
            f'PN acc {M["PN"]["acc"]:.4f} R {M["PN"]["R"]:.3f} W->Z {M["PN"]["WtoZ_before"]:.3f}->{M["PN"]["WtoZ_after"]:.3f}')
        for st in sts:
            r = M[st['name']]
            log(f'  {st["name"]:14s} acc {r["acc"]:.4f} gap {r["gap"] * 100:+.2f} [{r["gap_ci"][0] * 100:+.2f}, {r["gap_ci"][1] * 100:+.2f}] '
                f'agree {r["agree_ParT"]:.3f} KL {r["KL_ParT"]:.3f} | R {r["R"]:.3f} [{r["R_ci"][0]:.3f}, {r["R_ci"][1]:.3f}] W->Z {r["WtoZ_before"]:.3f}->{r["WtoZ_after"]:.3f}'
                + (f' | Holm gap {r["holm_gap"]:.2g} R-PN {r["holm_R_vs_PN"]:.2g} R-ParT {r["holm_R_vs_ParT"]:.2g} vs-untr {r["holm_vs_untrained"]:.2g}' if 'holm_gap' in r else ''))
        json.dump(res, open(FN, 'w'), indent=1, default=float)
    plot(res); log('done', FN)


if __name__ == '__main__':
    main()
