"""P33: what ParT's pair features read in the axis response (loop C, round C7; takes up the proposed P30).

P27v found that about half of the per-particle response of ParT to a small radial move is even under out vs in, that 96-98 % of
this even part goes through the pair inputs U_ij = PairEmbed(ln kT, ln z, ln dR, ln m2), and that it does not shrink as step^2.
This script splits that pair route by channel, by head and by the pair-embedding hinges, and tests a neighbour-holding move.

Models: ParT_full, ParT_kin (released weights); controls ParT_untr (ParT_full architecture, weaver initialisation, torch seed
3316) and ParT_kin_perm (output rows of every weight tensor permuted, P27 recipe, seed 3316). Controls run on NCTRL jets per class
and without the head knock-outs.
Sample: JetClass test_20M WToQQ_116 + ZToQQ_116, PER random rows each (numpy default_rng(3316), never first-k). Build:
part_core.build (eta = jet_eta + sign(jet_eta) deta), pT kept, so the pT order and every particle index are unchanged.
Edits per jet (rows): base; phi (all particles dphi + 0.06); dil (deta, dphi scaled by s = sqrt(M2'/M2) with M2' the axis moment
after the phi shift: the matched dilation); for each of the 10 highest-pT particles: s+ / s- (radial move out / in by 0.01 rad),
s3+ / s3- (0.003 rad), c+ / c- (neighbour-holding: the particle and every particle within dR < 0.05 of it moved by the same
vector, 0.01 rad along the particle's radial direction, so the distances inside that cluster stay fixed).
Conditions per row (the model is run with its own forward, the pair bias U passed explicitly):
  full  edited per-particle features x_e, U from the edited pair features
  part  x_e, U from the unedited pair features
  pair  unedited x_b, U from the edited pair features (all four channels)
  kT, z, dR, m2  x_b, U from the pair-feature tensor with only that channel taken from the edited jet
  KO0..KO7 (rows s+/s- only, trained models only) as 'pair' but head h's pair bias U_h set to its unedited value
Per row also: mean |U_c - U_b| per head for c in (pair, kT, z, dR, m2); the number of (pair, unit) sign changes of the
pre-activations (after BatchNorm, before GELU) of the four pair-embedding layers under 'pair', separately for near-collinear pairs
(unedited dR < 0.05) in layer 1; sum |delta pre-activation| of layer 1; mean |delta F_c| of each channel over changed pairs.
Read-out d = change of logit_Z - logit_W versus the unedited jet (P27 convention). Even part (d+ + d-)/2, odd part (d+ - d-)/2.
  ./sync.sh && ./part_job.sh c-p33-smoke 'python -m interp.p33_pair_channels --smoke --models ParT_full,ParT_untr'
  ./part_job.sh c-p33-full 'python -m interp.p33_pair_channels --models ParT_full'   (and ParT_kin, and ParT_untr,ParT_kin_perm)
  AI_PY=... ./job.sh c-p33-ana 1 2 'python -m interp.p33_pair_channels --analyse'      (CPU; numpy/scipy/matplotlib only)
  -> $AI_RESULTS/part/p33_pair_channels[_smoke]_<model>.npz, p33_pair_channels[_smoke].json, p33_pair_channels[_smoke].png
"""
import argparse, itertools, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--analyse', action='store_true')
ap.add_argument('--models', default='ParT_full,ParT_kin,ParT_untr,ParT_kin_perm')
ARGS = ap.parse_args(); SMOKE = ARGS.smoke
SEED = 3316; PER = 8 if SMOKE else 1500; NCTRL = 4 if SMOKE else 300; NB = 100 if SMOKE else 1000
DR, DR3, DRN, DPHI, RCL, NTOP, BS, JCH = 0.01, 0.003, 1e-6, 0.06, 0.05, 10, 96, 16
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
FILES = dict(Wqq='WToQQ_116.root', Zqq='ZToQQ_116.root')
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'; TAG = '_smoke' if SMOKE else ''
FN, PNG = OUTD / f'p33_pair_channels{TAG}.json', OUTD / f'p33_pair_channels{TAG}.png'
NPZ = lambda m: OUTD / f'p33_pair_channels{TAG}_{m}.npz'
ETYPES = ['base', 'phi', 'dil', 's+', 's-', 's3+', 's3-', 'c+', 'c-', 'n+', 'n-']
CH = ['kT', 'z', 'dR', 'm2']
NOCH = [f'no_{c}' for c in CH]
BCOND = ['full', 'part', 'pair'] + CH + ['m2diag', 'm2off', 'pair64'] + NOCH
COND = BCOND + [f'KO{h}' for h in range(8)]; NB_ = len(BCOND)
UCOND = ['pair'] + CH + ['m2diag', 'm2off'] + NOCH          # conditions whose pair bias differs from the unedited one
TRAINED = ['ParT_full', 'ParT_kin']; CTRL = ['ParT_untr', 'ParT_kin_perm']
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------- collect
def load_rows(path, n, rng, chunk=25000):
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries
    if SMOKE: N = min(N, 25000)
    rows = np.sort(rng.choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r): parts.append(sub(read_root(path, s, min(s + chunk, N)), r - s))
    return cat(parts), rows


def load_model(name):
    import torch
    from interp.part_core import ParTNetwork
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import CFG, FEATURES
    if name == 'ParT_full': mod, feat = ParTNetwork('full').model, 'full'
    elif name == 'ParT_kin': mod, feat = ParTNetwork('kin').model, 'kin'
    elif name == 'ParT_untr':
        torch.manual_seed(SEED); mod, feat = ParticleTransformer(input_dim=len(FEATURES['full']), **CFG), 'full'
    elif name == 'ParT_kin_perm':
        pk = ParTNetwork('kin'); g = torch.Generator().manual_seed(SEED); sd = pk.model.state_dict(); new = {}; perms = {}
        for k, v in sd.items():
            if k.endswith('weight') and v.ndim >= 2 and v.is_floating_point():
                p = torch.randperm(v.shape[0], generator=g); perms[k[:-len('weight')]] = p; new[k] = v[p].clone()
        for k, v in sd.items():
            if k in new: continue
            pre = k[:-len('bias')] if k.endswith('bias') else None
            if pre is not None and pre in perms and v.ndim == 1 and v.shape[0] == len(perms[pre]): new[k] = v[perms[pre]].clone()
            elif k.endswith('in_proj_bias') and (k[:-len('in_proj_bias')] + 'in_proj_') in perms:
                new[k] = v[perms[k[:-len('in_proj_bias')] + 'in_proj_']].clone()
            else: new[k] = v
        pk.model.load_state_dict(new); mod, feat = pk.model, 'kin'
    else: raise ValueError(name)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; mod.to(dev).eval()
    for p in mod.parameters(): p.requires_grad_(False)
    return mod, feat, dev


def inputs_fixed(J, n, P):
    """jetdistill.part.network.inputs with a fixed number of slots P (>= the largest multiplicity), weaver 'wrap' padding"""
    from jetdistill.part.network import FEATURES, raw_features
    vec = np.stack([J[k][:, :P] for k in ('px', 'py', 'pz', 'energy')], 1)
    J = {k: (v.astype(np.float64) if isinstance(v, np.ndarray) and v.dtype == np.float32 else v) for k, v in J.items()}   # float64 maths: no SIMD-position rounding
    m = J['mask'][:, :P] > 0; cnt = m.sum(1)
    wrap = np.arange(P)[None, :] % np.maximum(cnt, 1)[:, None]
    fill = lambda a: np.where(m, a[:, :P], np.take_along_axis(a[:, :P], wrap, 1))
    feats = []
    for k, c, s, cl in FEATURES[n]:
        x = raw_features(J, k)
        if c is not None: x = np.clip((x - np.float32(c)) * np.float32(s), *cl)
        feats.append(fill(np.where(J['mask'] > 0, x, 0)))
    vec = np.stack([fill(vec[:, i]) for i in range(4)], 1)
    return np.stack(feats, 1).astype(np.float32), vec.astype(np.float32), m[:, None, :].astype(np.float32)


def v64(J, pt, deta, dphi, mass, P):
    """(B, 4, P) float64 four-vectors from (pT, deta, dphi, mass), same order and 'wrap' padding as inputs_fixed (no float32 rounding)"""
    o = np.argsort(-pt, 1, kind='stable'); tk = lambda a: np.take_along_axis(np.asarray(a, np.float64), o, 1)[:, :P]
    pt, deta, dphi, mass = tk(pt), tk(deta), tk(dphi), tk(mass); m = pt > 0; cnt = m.sum(1)
    sgn = np.where(J['jet_eta'] < 0, -1.0, 1.0)[:, None]; eta = J['jet_eta'].astype(np.float64)[:, None] + sgn * deta
    phi = J['jet_phi'].astype(np.float64)[:, None] + dphi
    V = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + mass ** 2)], 1) * m[:, None]
    wrap = np.arange(P)[None, :] % np.maximum(cnt, 1)[:, None]
    return np.where(m[:, None], V, np.take_along_axis(V, np.repeat(wrap[:, None], 4, 1), 2))


def radial_unit(de, dp):
    r = np.hypot(de, dp)
    return (de / r, dp / r) if r > 1e-6 else (1.0, 0.0)


def plan(pt, de, dp):
    """edited (deta, dphi) for one chunk of jets; meta = (jet, etype, particle, cluster size, nn dR, r, z, x_odd_single, x_odd_cluster)"""
    meta, DE, DP = [], [], []
    for j in range(len(pt)):
        m = pt[j] > 0; n = int(m.sum()); e0, p0 = de[j].astype(np.float64), dp[j].astype(np.float64); z = pt[j] / pt[j].sum()
        M2 = (pt[j] * (e0 ** 2 + p0 ** 2)).sum(); M2s = (pt[j] * (e0 ** 2 + (p0 + DPHI * m) ** 2)).sum(); s = math.sqrt(M2s / M2)
        def add(t, e, p, i=-1, cs=0, nn=np.nan, r=np.nan, zz=np.nan, xs=np.nan, xc=np.nan):
            meta.append((j, ETYPES.index(t), i, cs, nn, r, zz, xs, xc)); DE.append(e); DP.append(p)
        add('base', e0.copy(), p0.copy()); add('phi', e0.copy(), p0 + DPHI * m); add('dil', e0 * s, p0 * s)
        for i in range(min(NTOP, n)):
            ue, up = radial_unit(e0[i], p0[i]); dist = np.hypot(e0 - e0[i], p0 - p0[i]); dist[~m] = np.inf
            cl = np.where(dist < RCL)[0]; d2 = dist.copy(); d2[i] = np.inf; nn = float(d2.min())
            r = float(np.hypot(e0[i], p0[i]))
            xs = 2 * pt[j, i] * r / M2                                                     # d(M2)/M2 per unit outward step
            xc = 2 * (pt[j, cl] * (e0[cl] * ue + p0[cl] * up)).sum() / M2                   # same for the rigid cluster move
            info = dict(i=i, cs=len(cl), nn=nn, r=r, zz=float(z[i]), xs=xs, xc=xc)
            for t, d in (('s+', DR), ('s-', -DR), ('s3+', DR3), ('s3-', -DR3), ('n+', DRN), ('n-', -DRN)):
                e, p = e0.copy(), p0.copy(); e[i] += d * ue; p[i] += d * up; add(t, e, p, **info)
            for t, d in (('c+', DR), ('c-', -DR)):
                e, p = e0.copy(), p0.copy(); e[cl] += d * ue; p[cl] += d * up; add(t, e, p, **info)
    return np.array(meta, np.float64), np.stack(DE), np.stack(DP)


class Runner:
    """ParT forward with the pair bias passed explicitly, the pair-feature tensor built by weaver's own pairwise_lv_fts"""

    def __init__(self, mod, dev):
        import torch
        from weaver.nn.model.ParticleTransformer import pairwise_lv_fts
        self.torch, self.mod, self.dev, self.plf = torch, mod, dev, pairwise_lv_fts

    def T(self, a): return self.torch.from_numpy(np.ascontiguousarray(a)).to(self.dev)

    def feats(self, v):                         # (B, 4, P) -> (B, 4, P, P): ln kT, ln z, ln dR, ln m2
        return self.plf(v.unsqueeze(-1), v.unsqueeze(-2), num_outputs=4, eps=1e-8)

    def embed(self, F, cap=False):
        torch = self.torch; B, C, P, _ = F.shape; z = F.reshape(B, C, P * P); pres = []
        for k, layer in enumerate(self.mod.pair_embed.embed):
            z = layer(z)
            if cap and k > 0 and isinstance(layer, torch.nn.BatchNorm1d): pres.append(z)
        return z.view(B, -1, P, P), pres

    def fwd(self, x, U, m):
        mod = self.mod; P = x.shape[-1]; mask = m.bool(); pad = ~mask.squeeze(1)
        h = mod.embed(x).masked_fill(~mask.permute(2, 0, 1), 0); am = U.reshape(-1, P, P)
        for b in mod.blocks: h = b(h, x_cls=None, padding_mask=pad, attn_mask=am)
        c = mod.cls_token.expand(1, h.size(1), -1)
        for b in mod.cls_blocks: c = b(h, x_cls=c, padding_mask=pad)
        return mod.fc(mod.norm(c).squeeze(0))


def collect(name, J, iW, iZ):
    from interp.part_core import kin, build, sub
    torch = __import__('torch'); mod, feat, dev = load_model(name); R = Runner(mod, dev)
    N = len(J['label']); pt, de, dp, ms = kin(J); f8 = lambda a: np.asarray(a, np.float64)
    J0 = build(J, f8(pt), f8(de), f8(dp), f8(ms))         # float64 maths inside build, float32 output: edited and unedited rows round alike
    ko = name in TRAINED
    META, LG, DU, CR, DF, L10, CHK = [], [], [], [], [], np.zeros((N, 10), np.float32), []
    tstart = time.time(); eye = None
    with torch.no_grad():
        for c0 in range(0, N, JCH):
            jj = np.arange(c0, min(c0 + JCH, N)); P = int(J0['mask'][jj].sum(1).max())
            eye = torch.eye(P, dtype=torch.bool, device=dev)[None]
            meta, DE, DP = plan(pt[jj], de[jj], dp[jj]); jl = meta[:, 0].astype(int)
            xb, vb, mb = (R.T(a) for a in inputs_fixed(sub(J0, jj), feat, P))
            Lb = R.fwd(xb, R.embed(R.feats(vb))[0], mb); L10[jj] = Lb.float().cpu().numpy()
            if c0 < 3 * JCH:   # own forward vs the weaver forward on the unedited jets
                Lw = mod(xb, vb, mb); CHK.append(float((Lw - Lb).abs().max()))
            Jr = sub(J, jj[jl])
            lg = np.full((len(jl), len(COND), 2), np.nan, np.float32); du = np.zeros((len(jl), len(UCOND), 8), np.float32)
            cr = np.zeros((len(jl), 6), np.float32); dfc = np.zeros((len(jl), 4), np.float32)
            for b0 in range(0, len(jl), BS):
                sl = np.arange(len(jl))[b0:b0 + BS]; Jb = sub(Jr, sl); p_, _, _, m_ = kin(Jb)
                E = build(Jb, f8(p_), DE[sl], DP[sl], f8(m_))
                assert np.array_equal(E['mask'], J0['mask'][jj[jl[sl]]]) and np.allclose(kin(E)[0], pt[jj[jl[sl]]], rtol=1e-5, atol=1e-5)
                xe, ve, _ = (R.T(a) for a in inputs_fixed(E, feat, P)); g = R.T(jl[sl]).long()
                xbb, mbb, vbb = xb[g], mb[g], vb[g]
                Fbb = R.feats(vbb); Ubb, preb = R.embed(Fbb, cap=True)          # unedited pair inputs, same batch shape as the edited ones
                rpb = (mbb[:, 0, :, None] * mbb[:, 0, None, :]) > 0
                ncb = rpb & (Fbb[:, 2] < math.log(RCL)) & ~eye
                Fe = R.feats(ve); Ue, pree = R.embed(Fe, cap=True)
                # hinge crossings of the four pair-embedding layers (pairs counted once: /2); cr columns: L1 all pairs, L1 near-collinear, L2, L3, L4, sum |delta pre1|
                for l in range(4):
                    s = ((pree[l] > 0) != (preb[l] > 0)).view(len(sl), -1, P, P) & rpb[:, None]
                    cr[sl, 1 + l if l else 0] = (s.sum((1, 2, 3)) / 2).float().cpu().numpy()
                    if l == 0:
                        cr[sl, 1] = ((s & ncb[:, None]).sum((1, 2, 3)) / 2).float().cpu().numpy()
                        a = (pree[0] - preb[0]).abs().view(len(sl), -1, P, P) * rpb[:, None]
                        cr[sl, 5] = (a.sum((1, 2, 3)) / 2).float().cpu().numpy()
                del pree, preb, s, a
                dFe = (Fe - Fbb).abs() * rpb[:, None]; chg = ((dFe[:, 2] > 1e-6) & rpb).sum((1, 2)).clamp(min=1)
                dfc[sl] = (dFe.sum((2, 3)) / chg[:, None]).float().cpu().numpy(); del dFe
                Us = {'pair': Ue}
                for k, c in enumerate(CH):
                    Fc = Fbb.clone(); Fc[:, k] = Fe[:, k]; Us[c] = R.embed(Fc)[0]
                for k, c in enumerate(CH):                                     # all four edited except channel c
                    Fc = Fe.clone(); Fc[:, k] = Fbb[:, k]; Us[f'no_{c}'] = R.embed(Fc)[0]
                for c, w in (('m2diag', eye), ('m2off', ~eye)):
                    Fc = Fbb.clone(); Fc[:, 3] = torch.where(w, Fe[:, 3], Fbb[:, 3]); Us[c] = R.embed(Fc)[0]
                npair = rpb.sum((1, 2)).float()
                for k, c in enumerate(UCOND):
                    du[sl, k] = (((Us[c] - Ubb).abs() * rpb[:, None]).sum((2, 3)) / npair[:, None]).float().cpu().numpy()
                F64 = R.feats(R.T(v64(Jb, f8(p_), DE[sl], DP[sl], f8(m_), P))).float()      # pair features without float32 rounding
                out = {'full': R.fwd(xe, Ue, mbb), 'part': R.fwd(xe, Ubb, mbb), 'pair': R.fwd(xbb, Ue, mbb),
                       'pair64': R.fwd(xbb, R.embed(F64)[0], mbb)}
                for c in CH + ['m2diag', 'm2off'] + NOCH: out[c] = R.fwd(xbb, Us[c], mbb)
                for k, c in enumerate(BCOND): lg[sl, k] = out[c][:, [iW, iZ]].float().cpu().numpy()
                if ko:
                    et = meta[sl, 1].astype(int); sel = np.where((et == 3) | (et == 4))[0]
                    if len(sel):
                        t = R.T(sel).long()
                        for h in range(8):
                            Uk = Ue[t].clone(); Uk[:, h] = Ubb[t, h]
                            lg[sl[sel], NB_ + h] = R.fwd(xbb[t], Uk, mbb[t])[:, [iW, iZ]].float().cpu().numpy()
                del Us, Ue, Fe, Fbb, Ubb, F64
            meta[:, 0] = jj[jl]; META.append(meta); LG.append(lg); DU.append(du); CR.append(cr); DF.append(dfc)
            if (c0 // JCH) % 10 == 0:
                el = time.time() - tstart; log(f'{name}: jets {c0 + len(jj)} / {N}, {el:.0f} s, ETA {el / (c0 + len(jj)) * (N - c0 - len(jj)):.0f} s')
    META = np.concatenate(META); y = J['label'].astype(int)
    from jetdistill.config import CLASSES as CL
    acc = dict(all=float((L10.argmax(1) == y).mean()), **{c: float((L10.argmax(1)[y == CL.index(c)] == CL.index(c)).mean()) for c in FILES},
               N=int(N), own_vs_weaver_forward_maxabs=max(CHK) if CHK else None)
    np.savez_compressed(NPZ(name), meta=META, lg=np.concatenate(LG), du=np.concatenate(DU), cr=np.concatenate(CR), df=np.concatenate(DF),
                        L10=L10, y=y, acc=json.dumps(acc), cond=np.array(COND), pt=pt, de=de, dp=dp)
    log(f'{name}: accuracy {acc}; wrote {NPZ(name)} ({len(META)} rows)')
    del mod; torch.cuda.empty_cache()


def collect_all(names):
    from interp.part_core import kin, sub, cat
    from jetdistill.config import CLASSES as CL
    OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); Js = []
    for c, fn in FILES.items():
        J, rows = load_rows(TEST / fn, PER, rng); Js.append(J); log(f'loaded {fn}: {len(rows)} random rows, first {rows[:3].tolist()}')
    J = cat(Js); pt = kin(J)[0]; keep = np.where((pt > 0).sum(1) >= 2)[0]; J = sub(J, keep)
    y = J['label'].astype(int); crng = np.random.default_rng(SEED + 1)
    ctrl = np.sort(np.concatenate([crng.choice(np.where(y == CL.index(c))[0], NCTRL, replace=False) for c in FILES]))
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    for nm in names: collect(nm, J if nm in TRAINED else sub(J, ctrl), iW, iZ)


# ----------------------------------------------------------------------------------------------------------------- analyse
def boot_W(n, nb, rng):
    W = np.zeros((nb, n)); [W.__setitem__(b, np.bincount(rng.integers(0, n, n), minlength=n)) for b in range(nb)]; return W


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]; return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def st(v, bs, v0=None):
    o = dict(v=float(v), ci=ci(bs))
    if v0 is not None:
        b = np.asarray(bs, float); b = b[np.isfinite(b)]; o['H0'] = v0
        o['p_boot'] = float(max(1 / len(b), min(1, 2 * min((b <= v0).mean(), (b >= v0).mean())))) if len(b) else None
    return o


def signflip_p(D, nperm=4000, seed=0):
    """paired sign-flip permutation over jets, H0: E[D_j] = 0 vs H1: > 0 (one-sided)"""
    D = np.asarray(D, float); rng = np.random.default_rng(seed); obs = D.mean()
    per = (rng.choice([-1.0, 1.0], (nperm, len(D))) * D).mean(1); return float((1 + (per >= obs).sum()) / (nperm + 1))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj.tolist()


def analyse_model(mn):
    Z = np.load(NPZ(mn)); meta, lg, du, cr, dfc = Z['meta'], Z['lg'].astype(np.float64), Z['du'], Z['cr'], Z['df']
    acc = json.loads(str(Z['acc'])); jet, et = meta[:, 0].astype(int), meta[:, 1].astype(int); N = len(Z['y'])
    sc = lg[:, :, 1] - lg[:, :, 0]; ib = np.where(et == 0)[0]; brow = np.full(N, -1); brow[jet[ib]] = ib
    C = {c: k for k, c in enumerate(COND)}; ref = np.array([C['pair'] if c.startswith('KO') else C[c] for c in COND])
    d = sc - sc[brow[jet]][:, ref]                                               # (rows, conds) vs the unedited jet in the same condition
    nC = sc.shape[1]
    Wb = boot_W(N, NB, np.random.default_rng(SEED + 5))
    idx = {t: np.where(et == ETYPES.index(t))[0] for t in ETYPES[3:]}
    for t in ETYPES[3:]: assert np.array_equal(jet[idx[t]], jet[idx['s+']]) and np.array_equal(meta[idx[t], 2], meta[idx['s+'], 2])
    I = idx['s+']; jr = jet[I]; S = lambda v: np.bincount(jr, v, minlength=N); cj = S(np.ones(len(I)))
    cs, nn, rr, zz, xs, xc = (meta[I, k] for k in (3, 4, 5, 6, 7, 8))
    R = dict(acc=acc, n_particles=int(len(I)), n_jets=int(N))
    R['floor_base_rows_maxabs_vs_full'] = {c: float(np.abs(sc[ib, C[c]] - sc[ib, 0]).max()) for c in BCOND if c != 'pair64'}
    R['pair64_unedited_shift'] = dict(mean_abs=float(np.abs(sc[ib, C['pair64']] - sc[ib, 0]).mean()), max_abs=float(np.abs(sc[ib, C['pair64']] - sc[ib, 0]).max()),
                                      note='score change when the unedited jet\'s pair features are computed in float64 instead of float32')
    ev, od = {}, {}
    for tag, a, b in (('s', 's+', 's-'), ('s3', 's3+', 's3-'), ('c', 'c+', 'c-'), ('n', 'n+', 'n-')):
        ev[tag] = (d[idx[a]] + d[idx[b]]) / 2; od[tag] = (d[idx[a]] - d[idx[b]]) / 2
    e, o = ev['s'], od['s']; ep = e[:, C['pair']]; Spp = S(ep ** 2)
    # 1) route and channel shares of the even part (single 0.01 moves)
    out = {}
    for c in BCOND:
        k = C[c]; se = S(e[:, k] ** 2); sx = S(e[:, k] * ep); so = S(o[:, k] ** 2)
        out[c] = dict(even_ms=st(se.sum() / cj.sum(), (Wb @ se) / (Wb @ cj)), odd_ms=st(so.sum() / cj.sum(), (Wb @ so) / (Wb @ cj)),
                      even_ms_over_pair=st(se.sum() / Spp.sum(), (Wb @ se) / (Wb @ Spp)),
                      even_proj_share_of_pair=st(sx.sum() / Spp.sum(), (Wb @ sx) / (Wb @ Spp)),
                      odd_mean=st(o[:, k].mean(), (Wb @ S(o[:, k])) / (Wb @ cj)),
                      even_share_of_variance=st(se.sum() / (se.sum() + so.sum()), (Wb @ se) / (Wb @ (se + so))))
    for c in CH:   # necessity: share of the even pair response removed when channel c alone is left unedited
        k = C[f'no_{c}']; sx = S(e[:, k] * ep); out[c]['even_removed_share_when_left_unedited'] = st(1 - sx.sum() / Spp.sum(), 1 - (Wb @ sx) / (Wb @ Spp))
    sf = S(e[:, C['full']] ** 2); out['pair_even_ms_over_full'] = st(Spp.sum() / sf.sum(), (Wb @ Spp) / (Wb @ sf))
    resid = ep - e[:, [C[c] for c in CH]].sum(1); s1, s2 = S(resid ** 2), S((ep - ep.mean()) ** 2)
    out['channel_additivity_R2_even'] = st(1 - s1.sum() / s2.sum(), 1 - (Wb @ s1) / (Wb @ s2))
    resid = e[:, C['full']] - e[:, C['part']] - ep; s1, s2 = S(resid ** 2), S((e[:, C['full']] - e[:, C['full']].mean()) ** 2)
    out['route_additivity_R2_even'] = st(1 - s1.sum() / s2.sum(), 1 - (Wb @ s1) / (Wb @ s2))
    # permutation: channel vs none (floor = the unedited row run through the same condition)
    fl = d[ib][np.argsort(jet[ib])]
    pc = []
    for c in CH:
        D = S(e[:, C[c]] ** 2) / np.maximum(cj, 1) - fl[:, C[c]] ** 2; out[c]['perm_p_vs_none'] = signflip_p(D, seed=SEED); pc.append(out[c]['perm_p_vs_none'])
    for c, p in zip(CH, holm(pc)): out[c]['perm_p_vs_none_holm4'] = p
    ps = {c: (Wb @ S(e[:, C[c]] * ep)) / (Wb @ Spp) for c in CH}
    top = max(CH, key=lambda c: out[c]['even_proj_share_of_pair']['v']); out['top_channel'] = top
    b = ps[top]; out['top_channel_share_vs_0.70'] = dict(v=out[top]['even_proj_share_of_pair']['v'], ci=ci(b), p_share_below_0_70=float((b < 0.70).mean()))
    for a_, b_ in itertools.combinations(CH, 2):
        out[f'share_diff_{a_}_minus_{b_}'] = st(out[a_]['even_proj_share_of_pair']['v'] - out[b_]['even_proj_share_of_pair']['v'], ps[a_] - ps[b_], 0.0)
    out['mean_abs_dF_per_changed_pair_s+'] = {c: float(dfc[I, k].mean()) for k, c in enumerate(CH)}
    out['mean_abs_dU_per_head'] = {c: {u: du[np.concatenate([idx['s+'], idx['s-']]), k].mean(0).round(6).tolist()} for k, c in enumerate(UCOND) for u in ['s+-']}
    R['single'] = out
    # 2) smoothness: even/odd variance ratio at 0.003 over 0.01, per condition
    sm = {}
    for c in BCOND:
        k = C[c]; e1, o1, e3, o3 = S(ev['s'][:, k] ** 2), S(od['s'][:, k] ** 2), S(ev['s3'][:, k] ** 2), S(od['s3'][:, k] ** 2)
        v = (e3.sum() / o3.sum()) / (e1.sum() / o1.sum()); bb = ((Wb @ e3) / (Wb @ o3)) / ((Wb @ e1) / (Wb @ o1))
        sm[c] = st(v, bb, (DR3 / DR) ** 2)
        sm[c]['even_ms_ratio_0.003_over_0.01'] = float(e3.sum() / e1.sum()); sm[c]['odd_ms_ratio_0.003_over_0.01'] = float(o3.sum() / o1.sum())
    R['smoothness_even_odd_ratio_0.003_over_0.01'] = sm
    tiny = {}
    for c in BCOND:
        k = C[c]; e1, o1, en, on = S(ev['s'][:, k] ** 2), S(od['s'][:, k] ** 2), S(ev['n'][:, k] ** 2), S(od['n'][:, k] ** 2)
        tiny[c] = {'even_ms_1e-6_over_0.01': st(en.sum() / e1.sum(), (Wb @ en) / (Wb @ e1), (DRN / DR) ** 2),
                   'odd_ms_1e-6_over_0.01': float(on.sum() / o1.sum()), 'even_ms_1e-6': float(en.sum() / cj.sum()),
                   'even_share_of_variance_1e-6': float(en.sum() / (en.sum() + on.sum()))}
    R['tiny_move_1e-6_vs_0.01'] = tiny
    # 3) neighbour-holding (cluster) move vs single move
    cl = {}
    for grp, msk in (('cluster_size_ge2', cs >= 2), ('cluster_size_1', cs == 1)):
        g = {}; Sm = lambda v: np.bincount(jr[msk], v, minlength=N)
        for c in BCOND:
            k = C[c]; a, b2 = Sm(ev['c'][msk, k] ** 2), Sm(ev['s'][msk, k] ** 2)
            g[c] = dict(even_ms_cluster_over_single=st(a.sum() / b2.sum(), (Wb @ a) / (Wb @ b2), 1.0))
            oc, os_ = od['c'][msk, k], od['s'][msk, k]
            g[c]['odd_slope_on_x_cluster'] = float((oc * xc[msk]).sum() / (xc[msk] ** 2).sum())
            g[c]['odd_slope_on_x_single'] = float((os_ * xs[msk]).sum() / (xs[msk] ** 2).sum())
            g[c]['odd_mean_cluster_over_single'] = float(oc.mean() / os_.mean())
        g['n_particles'] = int(msk.sum()); g['mean_cluster_size'] = float(cs[msk].mean()); g['mean_xc_over_xs'] = float(np.mean(xc[msk]) / np.mean(xs[msk]))
        cl[grp] = g
    R['cluster_vs_single'] = cl
    # 4) heads
    if nC > NB_ and np.isfinite(d[I, C['KO0']]).all():
        hd = {}; rng = np.random.default_rng(SEED + 9); half = rng.permutation(N) % 2; contr = []
        for h in range(8):
            k = C[f'KO{h}']; ch_ = ep - e[:, k]; contr.append(ch_)
            sx, sr = S(ep * ch_), S(e[:, k] ** 2)
            hd[h] = dict(proj_share=st(sx.sum() / Spp.sum(), (Wb @ sx) / (Wb @ Spp), 1 / 8),
                         remaining_even_ms_after_KO=st(sr.sum() / Spp.sum(), (Wb @ sr) / (Wb @ Spp), 1.0),
                         odd_proj_share=float((o[:, C['pair']] * (o[:, C['pair']] - o[:, k])).sum() / (o[:, C['pair']] ** 2).sum()))
            bb = (Wb @ sx) / (Wb @ Spp); hd[h]['p_one_sided_share_le_1_8'] = float(max(1 / NB, (bb <= 1 / 8).mean()))
        for h, p in zip(range(8), holm([hd[h]['p_one_sided_share_le_1_8'] for h in range(8)])): hd[h]['p_holm8'] = p
        contr = np.stack(contr, 1); r_ = ep - contr.sum(1)
        hd['head_additivity_R2_even'] = float(1 - (r_ ** 2).sum() / ((ep - ep.mean()) ** 2).sum())
        hd['sum_single_shares'] = float(sum(hd[h]['proj_share']['v'] for h in range(8)))
        # top-3 heads chosen on half A, evaluated on half B; null: every 3-head set (56) on half B
        def shares(mask_j):
            m = mask_j[jr]; den = (ep[m] ** 2).sum(); return np.array([(ep[m] * contr[m, h]).sum() / den for h in range(8)])
        sA, sB = shares(half == 0), shares(half == 1); top3 = np.argsort(-sA)[:3].tolist()
        combos = np.array([sB[list(c)].sum() for c in itertools.combinations(range(8), 3)])
        hd['top3_heads_halfA'] = top3; hd['top3_share_halfB'] = float(sB[top3].sum())
        hd['random3_share_halfB'] = dict(mean=float(combos.mean()), q2_5=float(np.percentile(combos, 2.5)), q97_5=float(np.percentile(combos, 97.5)),
                                         p_frac_ge_top3=float((combos >= sB[top3].sum()).mean()))
        mjB = half == 1
        bb = np.array([((Wb[:, mjB] @ S(ep * contr[:, top3].sum(1))[mjB]) / (Wb[:, mjB] @ Spp[mjB]))]).ravel()
        hd['top3_share_halfB_ci'] = ci(bb)
        hd['dU_per_head_by_channel_s+-'] = {c: du[np.concatenate([idx['s+'], idx['s-']]), k].mean(0).round(6).tolist() for k, c in enumerate(UCOND)}
        R['heads'] = {str(k): v for k, v in hd.items()}
    # 5) hinge map: L1 sign changes summed over out and in moves, near-collinear subset, nearest-neighbour dR
    c1 = cr[idx['s+'], 0] + cr[idx['s-'], 0]; cn = cr[idx['s+'], 1] + cr[idx['s-'], 1]; call = cr[idx['s+']][:, [0, 2, 3, 4]].sum(1) + cr[idx['s-']][:, [0, 2, 3, 4]].sum(1)
    hm = dict(mean_L1_crossings=float(c1.mean()), frac_particles_L1_cross=float((c1 > 0).mean()), frac_particles_nc_cross=float((cn > 0).mean()),
              mean_L1_crossings_nc=float(cn.mean()))
    from scipy.stats import spearmanr
    ae = np.abs(ep)
    hm['spearman_abs_even_pair_vs_L1_cross'] = float(spearmanr(ae, c1).correlation)
    hm['spearman_abs_even_pair_vs_nc_cross'] = float(spearmanr(ae, cn).correlation)
    hm['spearman_abs_even_pair_vs_all_layer_cross'] = float(spearmanr(ae, call).correlation)
    hm['spearman_abs_even_pair_vs_nn_dR'] = float(spearmanr(ae, nn).correlation)
    m = cn > 0; a1 = S(ep ** 2 * m); fr = S(m.astype(float))
    hm['even_ms_share_of_nc_crossing_particles'] = st(a1.sum() / Spp.sum(), (Wb @ a1) / (Wb @ Spp))
    hm['particle_share_of_nc_crossing_particles'] = st(fr.sum() / cj.sum(), (Wb @ fr) / (Wb @ cj))
    hm['excess_share'] = st(a1.sum() / Spp.sum() - fr.sum() / cj.sum(), (Wb @ a1) / (Wb @ Spp) - (Wb @ fr) / (Wb @ cj), 0.0)
    bins = {}
    for lab, mk in (('L1_cross_0', c1 == 0), ('L1_cross_1_10', (c1 > 0) & (c1 <= 10)), ('L1_cross_11_100', (c1 > 10) & (c1 <= 100)), ('L1_cross_gt100', c1 > 100)):
        bins[lab] = dict(n=int(mk.sum()), mean_even_pair_sq=float((ep[mk] ** 2).mean()) if mk.any() else None,
                         even_ms_share=float((ep[mk] ** 2).sum() / (ep ** 2).sum()))
    hm['by_L1_crossings'] = bins
    nb = {}; edges = [0, 0.003, 0.01, 0.03, 0.05, 0.1, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (nn >= lo) & (nn < hi); lab = f'nn_dR_{lo}_{hi}'
        if mk.sum() < 5: nb[lab] = dict(n=int(mk.sum())); continue
        k = C['pair']; e1, o1, e3, o3 = (v[mk, k] ** 2 for v in (ev['s'], od['s'], ev['s3'], od['s3']))
        nb[lab] = dict(n=int(mk.sum()), mean_even_pair_sq=float(e1.mean()), even_ms_share=float(e1.sum() / (ep ** 2).sum()),
                       even_share_of_variance_pair=float(e1.sum() / (e1.sum() + o1.sum())),
                       smooth_ratio_pair=float((e3.sum() / o3.sum()) / (e1.sum() / o1.sum())),
                       mean_L1_crossings=float(c1[mk].mean()), dR_share=float((ev['s'][mk, C['dR']] * ev['s'][mk, k]).sum() / e1.sum()))
    hm['by_nearest_neighbour_dR'] = nb
    R['hinge'] = hm
    # 6) whole-jet edits
    wj = {}
    for t in ('phi', 'dil'):
        it = np.where(et == ETYPES.index(t))[0]; jt = jet[it]; Sj = lambda v: np.bincount(jt, v, minlength=N)
        wj[t] = {c: st(d[it, C[c]].mean(), (Wb @ Sj(d[it, C[c]])) / (Wb @ Sj(np.ones(len(it))))) for c in BCOND}
        for c in COND[1:7]: wj[t][c]['share_of_full_mean'] = float(d[it, C[c]].mean() / d[it, C['full']].mean())
    R['whole_jet'] = wj
    R['sample_profile'] = dict(mean_cluster_size=float(cs.mean()), frac_cluster_ge2=float((cs >= 2).mean()), median_nn_dR=float(np.median(nn)),
                               frac_nn_lt_0_01=float((nn < 0.01).mean()), frac_nn_lt_0_003=float((nn < 0.003).mean()))
    log(f"{mn}: acc {acc['all']:.4f}; pair/full even {out['pair_even_ms_over_full']['v']:.3f}; channel proj shares "
        + ', '.join(f"{c} {out[c]['even_proj_share_of_pair']['v']:.3f}" for c in CH) + f"; add R2 {out['channel_additivity_R2_even']['v']:.3f}; "
        f"cluster/single even (pair, cs>=2) {cl['cluster_size_ge2']['pair']['even_ms_cluster_over_single']['v']:.3f}; smooth pair {sm['pair']['v']:.2f}")
    return R


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return clean(o.item())
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def analyse():
    res = dict(settings=dict(seed=SEED, per_class=PER, nctrl_per_class=NCTRL, dr=DR, dr3=DR3, dphi=DPHI, cluster_dR=RCL, ntop=NTOP, NB=NB,
                             files=FILES, readout='d = change of logit_Z - logit_W vs the unedited jet', conditions=COND), models={})
    for mn in TRAINED + CTRL:
        if NPZ(mn).exists(): res['models'][mn] = analyse_model(mn)
    FN.write_text(json.dumps(clean(res), indent=1)); log(f'wrote {FN}'); return res


def plot(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    M = [m for m in TRAINED if m in res['models']]; R = res['models']
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6)); cols = {'ParT_full': 'C0', 'ParT_kin': 'C1'}
    a = ax[0]; xs = np.arange(len(CH))
    for k, m in enumerate(M):
        for q, (key, hatch, lab) in enumerate((('even_proj_share_of_pair', None, 'only this channel edited'), ('even_removed_share_when_left_unedited', '//', 'lost when only this channel is not edited'))):
            v = np.array([R[m]['single'][c][key]['v'] for c in CH]); lo = np.array([R[m]['single'][c][key]['ci'][0] for c in CH]); hi = np.array([R[m]['single'][c][key]['ci'][1] for c in CH])
            a.bar(xs + (2 * k + q - 1.5) * 0.2, v, 0.2, yerr=[v - lo, hi - v], capsize=2, color=cols[m], hatch=hatch, alpha=0.9 if q == 0 else 0.5, label=f'{m}: {lab}')
    a.axhline(0.7, color='k', ls='--', lw=0.8, label='thesis threshold 0.70'); a.axhline(0, color='k', lw=0.5)
    a.set_xticks(xs); a.set_xticklabels([r'$\ln k_T$', r'$\ln z$', r'$\ln\Delta R$', r'$\ln m^2$'])
    a.set_ylabel('share of the even pair-route response'); a.legend(fontsize=6); a.set_title('(a) pair channels: one edited / one held back\n(single particle moved 0.01 rad)', fontsize=9)
    a = ax[1]; cc = ['full', 'pair', 'part', 'dR', 'kT', 'm2']; xs = np.arange(len(cc))
    for k, m in enumerate(M):
        g = R[m]['cluster_vs_single']['cluster_size_ge2']; v = np.array([g[c]['even_ms_cluster_over_single']['v'] for c in cc])
        lo = np.array([g[c]['even_ms_cluster_over_single']['ci'][0] for c in cc]); hi = np.array([g[c]['even_ms_cluster_over_single']['ci'][1] for c in cc])
        a.bar(xs + (k - 0.5) * 0.38, v, 0.38, yerr=[v - lo, hi - v], capsize=2, color=cols[m], label=m)
    a.axhline(1, color='k', lw=0.6, ls=':'); a.set_xticks(xs); a.set_xticklabels(cc); a.set_yscale('log')
    a.set_ylabel('even mean square: cluster move / single move'); a.legend(fontsize=7)
    a.set_title(r'(b) moving the particle with its neighbours ($\Delta R<0.05$)' + '\nparticles with >= 1 neighbour', fontsize=9)
    a = ax[2]; xs = np.arange(8)
    for k, m in enumerate(M):
        if 'heads' not in R[m]: continue
        H = R[m]['heads']; v = np.array([H[str(h)]['proj_share']['v'] for h in range(8)]); lo = np.array([H[str(h)]['proj_share']['ci'][0] for h in range(8)]); hi = np.array([H[str(h)]['proj_share']['ci'][1] for h in range(8)])
        a.bar(xs + (k - 0.5) * 0.38, v, 0.38, yerr=[v - lo, hi - v], capsize=2, color=cols[m], label=m)
    a.axhline(1 / 8, color='k', ls='--', lw=0.8, label='equal split 1/8'); a.set_xticks(xs); a.set_xticklabels([f'head {h}' for h in range(8)], fontsize=7)
    a.set_ylabel('share of even pair-route response'); a.legend(fontsize=7); a.set_title('(c) pair bias of one head held at its unedited value', fontsize=9)
    a = ax[3]
    for m in M:
        nb = R[m]['hinge']['by_nearest_neighbour_dR']; labs = [k for k in nb if nb[k].get('mean_even_pair_sq') is not None]
        mid = [0.5 * (float(k.split('_')[2]) + min(float(k.split('_')[3]), 0.2)) for k in labs]
        a.plot(mid, [nb[k]['mean_even_pair_sq'] for k in labs], 'o-', color=cols[m], label=m)
    a.axvline(DR, color='k', ls=':', lw=0.8, label='step 0.01 rad'); a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel(r'distance to the nearest other particle $\Delta R_{\rm nn}$ [rad]'); a.set_ylabel('mean (even part of pair-route response)$^2$')
    a.legend(fontsize=7); a.set_title('(d) even response vs nearest-neighbour distance', fontsize=9)
    fig.tight_layout(); fig.savefig(PNG, dpi=120); log(f'wrote {PNG}')


if __name__ == '__main__':
    if not ARGS.analyse:
        collect_all([m for m in ARGS.models.split(',') if m])
    else:
        plot(analyse())
