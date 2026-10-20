"""P21 -- which prong observable ParT uses (claim C4, prong stage). P18 / P18v found one training-specific late-block response in
ParT: the share of the full-jet azimuth-scramble response carried by one pooled direction rises to ~0.7 by blocks 6-7 (untrained
flat ~0.5). The scramble moves tau21, tau32 and D2 together. Here each observable is moved with the other two held.

Models: released ParT_full, ParT_kin; ParticleNet_full / _kin (released, interp/pn_net.py; behaviour only, cross-architecture row);
  controls with the ParT_full architecture (P18 make_models): 'init' = weaver default init, torch seed 18, untrained; 'perm' = the
  released ParT_full with the output rows of every weight tensor permuted (one permutation per layer, seed 18).
Sample: JetClass test_20M WToQQ_103, ZToQQ_103, TTBar_103, ZJetsToNuNu_103 (classes Wqq, Zqq, Tbqq, QCD); PER random rows per class
  (numpy default_rng(2103), never first-k); jets with >= 4 particles. Unedited 10-class accuracy reported on this sample and on the
  P1 rows (rows 3,000-4,000 of each class's _100 file, part_core.sample; reference 85.9 % full, 74.5 % kin).
Frame: every jet (base and edits) is rebuilt about its own E-scheme axis (4-vector sum of its particles; part_core.build,
  eta = jet_eta + sign(jet_eta) deta) and, for edits, its angles are dilated about that axis until the invariant mass equals the
  base mass (|dm/m| < 0.5 % required). Particle pT, masses, PID / impact-parameter fields and N are unchanged in every edit.
Observables (as P18.substructure, torch float64): tau_k (beta = 1, k-means axes, R0 = 0.8), tau21, tau32, D2 (beta = 1, 24 hardest).
Edit components (in the (deta, dphi) plane; clusters = k-means 2- or 3-subjet assignment of the base jet, the tau axes):
  A(s2): every particle's offset from its 2-subjet pT centroid x s2;  B(s3): the same about the 3-subjet centroids;
  C(u): soft particles (pT fraction z < 0.02) moved radially about the jet pT centroid by u (wide-angle soft redistribution).
  Applied in the order A, B, C, then the rebuild + mass restore (a uniform dilation leaves tau21, tau32, D2 unchanged).
Edits, nominal dose s in {0.6, 0.8, 1.25}:
  E1 'tau21 edit' = A(s) with (s3, u) chosen per jet (grid 11 x 11 on [0.5, 2] + two 5 x 5 refinements) to hold tau32 and D2;
  E2 'tau32 edit' = B(s) with (s2, u) chosen to hold tau21 and D2;  E3 'D2 edit' = C(s) with (s2, s3) chosen to hold tau21, tau32.
  Accepted jets: both held observables within 2 % and m within 0.5 % after the rebuild (achieved deltas recorded).
Controls per edit and dose (same jets): rot = rotation of each cluster's particles about its centroid (E1: 2-subjet clusters; E2:
  3-subjet clusters; E3: only the soft particles, about their own 3-subjet cluster's centroid) by the angle that gives the same pT-weighted mean displacement as the edit (random sign per cluster), then the same
  rebuild + mass restore (prong structure kept; achieved observable changes recorded); rc = the primary component with the same dose
  about clusters built around 2 (E1) / 3 (E2) random constituents (drawn with probability ~ pT; nearest assignment) or, for E3,
  about a random constituent instead of the jet centroid (no compensation). raw = the primary component alone at the same dose with no compensation
  (A(s), B(s) or C(s); then the rebuild + mass restore), to separate the primary component from the compensating ones. Particle pT
  are unchanged in every edit and control, so
  |dpT_i| is matched trivially (= 0). scrF = the P18 full-jet azimuth scramble about the base jet's axis (positive reference).
Metrics: d_WQ = logit(Wqq) - logit(QCD), d_TQ = logit(Tbqq) - logit(QCD); change per jet vs the base jet; % 10-class calls changed;
  edit - rot paired difference (95 % bootstrap CI over jets, NB resamples; sign-flip permutation p; McNemar on call changes);
  Holm over nets x edits x doses for the primary score (E1, E3 -> d_WQ; E2 -> d_TQ). Reference classifiers: HistGradientBoosting on
  (log m, tau21) and (log m, tau32, log D2) and (log m, tau21, tau32, log D2), W/Z-vs-QCD and top-vs-QCD, 2-fold cross-fitted on the
  base jets of this sample; predicted change = f(edited observables) - f(base). Slope of d on the primary observable's change
  (through 0) for the net and for the reference; ratio net / reference.
Location (ParT_full, ParT_kin, init; DAS dose 0.6 for E1-E3, plus scrF): (i) per-site 1-d DAS (P18 method: pooled pT-weighted
  particle mean, class token at c1; counterfactual = the full edited jet; q fitted by Adam STEPS x BS on N_DAS_TR jets, IIA on N_EVAL
  jets, 1,000 linearised random directions); trained - init per site and difference-in-differences (b7 - emb), paired bootstrap.
  (ii) head patching in blocks 5-8: in the edited jet's forward pass, the output of one head (or all 8 heads of a block) is replaced
  by its value on the unedited jet; restoration R = 1 - sum ||L_patch - L_base||^2 / sum ||L_edit - L_base||^2 and the primary
  score share S = sum (d_edit - d_patch) / sum (d_edit - d_base); one-sided bootstrap p (H0: R <= 0), Holm over the 64 heads of
  ParT_full + ParT_kin per edit; same on init and perm.
H0: response to an edit = response to its matched rotation control. H1: the network responds to the edited observable.
  Thesis: E1 moves d_WQ with a slope comparable to GBDT(m, tau21); E2 moves d_TQ; response located in blocks 6-7 above init / perm.
  Alternatives: equal response to E1-E3 (joint prong shape); response = rotation control; untrained net shows the same late share.

  ./sync.sh && ./part_job.sh c-p21-smoke 'python -m interp.p21_prong_part --smoke'
  ./sync.sh && ./part_job.sh c-p21-run 'python -m interp.p21_prong_part'
  -> $AI_RESULTS/part/p21_prong_part.json, p21_prong_part.png
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SMOKE = '--smoke' in sys.argv
PLOT = '--plot' in sys.argv
_argv = sys.argv; sys.argv = [_argv[0]] + (['--smoke'] if SMOKE else [])
import interp.p18_stage_pipeline_part as P  # noqa: E402
sys.argv = _argv

SEED = 2103
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TAG = 'p21_prong_part_smoke' if SMOKE else 'p21_prong_part'
FN, FIG = OUT / f'{TAG}.json', OUT / f'{TAG}.png'
PER = 150 if SMOKE else 4000
CLS = ['Wqq', 'Zqq', 'Tbqq', 'QCD']
SV = (0.6, 0.8, 1.25)
DAS_S = 0.6
KINDS = ('E1', 'E2', 'E3')
HOLD = dict(E1=('tau32', 'D2'), E2=('tau21', 'D2'), E3=('tau21', 'tau32'))
PRIM = dict(E1='tau21', E2='tau32', E3='D2')
PSCORE = dict(E1='dWQ', E2='dTQ', E3='dWQ', scrF='dWQ')
TOL, MTOL, ZSOFT = 0.02, 0.005, 0.02
NB = 200 if SMOKE else 1000
NPERM = 500 if SMOKE else 5000
N_DAS_TR = 200 if SMOKE else 3000
N_EVAL = 100 if SMOKE else 1000
P.STEPS = 10 if SMOKE else 150
P.BS = 32 if SMOKE else 128
P.NB = NB
SITES = ['emb', 'b2', 'b4', 'b5', 'b6', 'b7', 'b8', 'c1']
HBLK = [4, 5, 6, 7]
iQ, iZ, iW, iT = 0, 6, 7, 8
GCHUNK = 300 if SMOKE else 4000
log = P.log
ci, holm, strip = P.ci, P.holm, P.strip


# ================================================================================================ observables (torch)
def kmeans_t(torch, pt, x, y, k, iters=6):
    """P18.kmeans in torch: seeds = hardest particle, then the pT-weighted farthest; 6 Lloyd steps (pT weights)"""
    J = pt.shape[0]; ar = torch.arange(J, device=pt.device); pts = torch.stack([x, y], -1); ax = pts.new_zeros(J, k, 2)
    ax[:, 0] = pts[ar, pt.argmax(1)]
    for j in range(1, k):
        d = (pts[:, :, None] - ax[:, None, :j]).norm(dim=-1).min(-1).values * pt
        ax[:, j] = pts[ar, d.argmax(1)]
    for _ in range(iters):
        a = (pts[:, :, None] - ax[:, None]).norm(dim=-1).argmin(-1)
        for j in range(k):
            w = pt * (a == j); s = w.sum(1, keepdim=True)
            ax[:, j] = torch.where(s > 0, (w[..., None] * pts).sum(1) / s.clamp_min(1e-12), ax[:, j])
    return ax, (pts[:, :, None] - ax[:, None]).norm(dim=-1)


def obs_t(torch, pt, x, y):
    """tau1-3, tau21, tau32 (beta 1, k-means axes, R0 0.8) and D2 (beta 1, 24 hardest; jets are pT-sorted) as P18.substructure"""
    z = pt / pt.sum(1, keepdim=True).clamp_min(1e-12); o = {}
    for k in (1, 2, 3): o[f'tau{k}'] = (z * kmeans_t(torch, pt, x, y, k)[1].min(-1).values).sum(1) / P.R0
    o['tau21'] = o['tau2'] / o['tau1'].clamp_min(1e-12); o['tau32'] = o['tau3'] / o['tau2'].clamp_min(1e-12)
    zt, et, ft = z[:, :24], x[:, :24], y[:, :24]
    R = ((et[:, :, None] - et[:, None]) ** 2 + (ft[:, :, None] - ft[:, None]) ** 2).sqrt()
    W = zt[:, :, None] * zt[:, None, :] * R; e2 = W.sum((1, 2)) / 2
    e3 = (W * torch.einsum('jik,jk,jkl->jil', R, zt, R)).sum((1, 2)) / 6
    o['D2'] = e3 / (e2 ** 3).clamp_min(1e-12)
    return o


def centroids(torch, pt, x, y, sel):
    w = pt * sel; W = w.sum(1, keepdim=True).clamp_min(1e-12)
    return (w * x).sum(1, keepdim=True) / W, (w * y).sum(1, keepdim=True) / W


def cl_scale(torch, pt, x, y, lab, k, s):
    for c in range(k):
        sel = (lab == c) & (pt > 0); cx, cy = centroids(torch, pt, x, y, sel)
        x, y = torch.where(sel, cx + s[:, None] * (x - cx), x), torch.where(sel, cy + s[:, None] * (y - cy), y)
    return x, y


def cl_rot(torch, pt, x, y, lab, k, alpha, sgn, move=None):
    """rotation about each cluster's pT centroid (centroid from the whole cluster; only `move` particles rotated if given)"""
    for c in range(k):
        sel = (lab == c) & (pt > 0); cx, cy = centroids(torch, pt, x, y, sel); a = (alpha * sgn[:, c])[:, None]
        if move is not None: sel = sel & move
        dx, dy = x - cx, y - cy; nx, ny = cx + a.cos() * dx - a.sin() * dy, cy + a.sin() * dx + a.cos() * dy
        x, y = torch.where(sel, nx, x), torch.where(sel, ny, y)
    return x, y


def soft_scale(torch, pt, x, y, soft, u, cx=None, cy=None):
    if cx is None: cx, cy = centroids(torch, pt, x, y, pt > 0)
    return torch.where(soft, cx + u[:, None] * (x - cx), x), torch.where(soft, cy + u[:, None] * (y - cy), y)


def own_rho(torch, pt, x, y, lab, k):
    rho = torch.zeros_like(x)
    for c in range(k):
        sel = (lab == c) & (pt > 0); cx, cy = centroids(torch, pt, x, y, sel)
        rho = torch.where(sel, ((x - cx) ** 2 + (y - cy) ** 2).sqrt(), rho)
    return rho


def edit_positions(torch, dev, pt_np, x_np, y_np, seed):
    """per chunk: (x, y) of every edit / control variant in the base (deta, dphi) frame + search diagnostics"""
    T = lambda a: torch.from_numpy(np.asarray(a, np.float64)).to(dev)
    pt, x0, y0 = T(pt_np), T(x_np), T(y_np); real = pt > 0; J = pt.shape[0]
    lab2 = kmeans_t(torch, pt, x0, y0, 2)[1].argmin(-1).masked_fill(~real, -1)
    lab3 = kmeans_t(torch, pt, x0, y0, 3)[1].argmin(-1).masked_fill(~real, -1)
    soft = (pt / pt.sum(1, keepdim=True) < ZSOFT) & real
    base = obs_t(torch, pt, x0, y0); g = torch.Generator(device=dev).manual_seed(seed)
    out, info = {}, {}
    lg = torch.linspace(math.log(0.5), math.log(2.0), 11, dtype=torch.float64, device=dev); st0 = float(lg[1] - lg[0])
    one = torch.ones(J, dtype=torch.float64, device=dev)

    def apply(s2, s3, u):
        x, y = cl_scale(torch, pt, x0, y0, lab2, 2, s2); x, y = cl_scale(torch, pt, x, y, lab3, 3, s3)
        return soft_scale(torch, pt, x, y, soft, u)
    for kind in KINDS:
        for s in SV:
            sv = one * s
            par = (lambda p, q: (sv, p, q)) if kind == 'E1' else (lambda p, q: (p, sv, q)) if kind == 'E2' else (lambda p, q: (p, q, sv))

            def devf(p, q):
                o = obs_t(torch, pt, *apply(*par(p, q)))
                return torch.stack([(o[h] / base[h].clamp_min(1e-12) - 1).abs() for h in HOLD[kind]], 0).max(0).values
            bd = torch.full((J,), float('inf'), dtype=torch.float64, device=dev); bp, bq = one.clone(), one.clone()
            for a in lg:
                for b in lg:
                    p, q = one * a.exp(), one * b.exp(); d = devf(p, q); u = d < bd
                    bd, bp, bq = torch.where(u, d, bd), torch.where(u, p, bp), torch.where(u, q, bq)
            for st in (st0 / 2, st0 / 4):
                p0, q0 = bp.log(), bq.log()
                for da in (-1, -0.5, 0, 0.5, 1):
                    for db in (-1, -0.5, 0, 0.5, 1):
                        if da == 0 and db == 0: continue
                        p, q = (p0 + da * st).exp(), (q0 + db * st).exp(); d = devf(p, q); u = d < bd
                        bd, bp, bq = torch.where(u, d, bd), torch.where(u, p, bp), torch.where(u, q, bq)
            xe, ye = apply(*par(bp, bq)); out[(kind, s, 'E')] = (xe, ye); out[(kind, s, 'raw')] = apply(*par(one, one))
            info[(kind, s)] = dict(dev=bd, p=bp, q=bq)
            # matched rotation control
            D = (pt * ((xe - x0) ** 2 + (ye - y0) ** 2).sqrt()).sum(1) / pt.sum(1)
            lab, k = (lab2, 2) if kind == 'E1' else (lab3, 3); rho = own_rho(torch, pt, x0, y0, lab, k)
            if kind == 'E3': rho = torch.where(soft, rho, torch.zeros_like(rho))
            mrho = (pt * rho).sum(1) / pt.sum(1); alpha = 2 * torch.asin((D / (2 * mrho.clamp_min(1e-9))).clamp(max=1.0))
            sgn = (torch.randint(0, 2, (J, k), generator=g, device=dev) * 2 - 1).double()
            out[(kind, s, 'rot')] = cl_rot(torch, pt, x0, y0, lab, k, alpha, sgn, move=(soft if kind == 'E3' else None))
            # random-cluster control
            if kind in ('E1', 'E2'):
                k = 2 if kind == 'E1' else 3
                c = torch.multinomial(pt.float().clamp_min(0), k, replacement=False, generator=g)
                cxs, cys = torch.gather(x0, 1, c), torch.gather(y0, 1, c)
                labr = ((x0[:, :, None] - cxs[:, None]) ** 2 + (y0[:, :, None] - cys[:, None]) ** 2).argmin(-1).masked_fill(~real, -1)
                out[(kind, s, 'rc')] = cl_scale(torch, pt, x0, y0, labr, k, sv)
            else:
                c = torch.multinomial(pt.float().clamp_min(0), 1, generator=g)
                out[(kind, s, 'rc')] = soft_scale(torch, pt, x0, y0, soft, sv, torch.gather(x0, 1, c), torch.gather(y0, 1, c))
    return {k: (v[0].cpu().numpy(), v[1].cpu().numpy()) for k, v in out.items()}, \
        {k: {a: b.cpu().numpy() for a, b in v.items()} for k, v in info.items()}, {k: v.cpu().numpy() for k, v in base.items()}


# ================================================================================================ rebuild about own axis
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def finalize(Jb, pt, x, y, ms, m0, iters=6):
    """jet rebuilt about its own E-scheme axis; angles dilated about that axis until the invariant mass equals m0 (iterated)"""
    from interp.part_core import build
    real = pt > 0; pt = pt.astype(np.float64); ms = ms.astype(np.float64)
    sg = np.where(Jb['jet_eta'] < 0, -1.0, 1.0)[:, None]
    eta = Jb['jet_eta'][:, None].astype(np.float64) + sg * x; phi = Jb['jet_phi'][:, None].astype(np.float64) + y

    def axis(eta, phi):
        px, py, pz = pt * np.cos(phi) * real, pt * np.sin(phi) * real, pt * np.sinh(eta) * real
        E = np.sqrt((pt * np.cosh(eta)) ** 2 + ms ** 2) * real; S = [a.sum(1) for a in (px, py, pz, E)]
        return np.arcsinh(S[2] / np.maximum(np.hypot(S[0], S[1]), 1e-9)), np.arctan2(S[1], S[0]), \
            np.sqrt(np.maximum(S[3] ** 2 - S[0] ** 2 - S[1] ** 2 - S[2] ** 2, 0))
    for it in range(iters):
        eJ, pJ, m = axis(eta, phi)
        k = np.clip(np.where(m > 0, m0 / np.maximum(m, 1e-9), 1.0), 0.5, 2.0)[:, None]
        eta = eJ[:, None] + k * (eta - eJ[:, None]); phi = pJ[:, None] + k * wrap(phi - pJ[:, None])
    eJ, pJ, m = axis(eta, phi); sgn = np.where(eJ < 0, -1.0, 1.0)[:, None]
    Jt = dict(Jb); Jt['jet_eta'] = eJ.astype(np.float32); Jt['jet_phi'] = wrap(pJ).astype(np.float32)
    de = np.where(real, sgn * (eta - eJ[:, None]), 0); dp = np.where(real, wrap(phi - pJ[:, None]), 0)
    return build(Jt, pt.astype(np.float32), de.astype(np.float32), dp.astype(np.float32), ms.astype(np.float32))


def obs_np(torch, dev, J, chunk=4000):
    from interp.part_core import kin, jmass
    pt, de, dp, _ = kin(J); o = {}
    for i in range(0, len(pt), chunk):
        T = lambda a: torch.from_numpy(a[i:i + chunk].astype(np.float64)).to(dev)
        r = obs_t(torch, T(pt), T(de), T(dp))
        for k, v in r.items(): o.setdefault(k, []).append(v.cpu().numpy())
    o = {k: np.concatenate(v) for k, v in o.items()}; o['m'] = jmass(J).astype(np.float64); return o


# ================================================================================================ networks
def pn_logits(torch, model, J, n, batch=500):
    from jetdistill.part.network import inputs, FEATURES
    names = [f[0] for f in FEATURES[n]]; ie, ip = names.index('deta'), names.index('dphi'); N = len(J['mask'])
    order = np.argsort(J['mask'].sum(1), kind='stable'); L = np.zeros((N, 10)); dev = next(model.parameters()).device
    with torch.no_grad():
        for i in range(0, N, batch):
            idx = order[i:i + batch]; x, _, m = inputs({k: (a[idx] if np.ndim(a) else a) for k, a in J.items()}, n); Pp = x.shape[-1]
            if Pp < 128:
                x = np.concatenate([x, np.zeros(x.shape[:2] + (128 - Pp,), np.float32)], -1); m = np.concatenate([m, np.zeros(m.shape[:2] + (128 - Pp,), np.float32)], -1)
            p = np.ascontiguousarray(x[:, [ie, ip]]); T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
            L[idx] = model(T(p), T(x), T(m)).double().cpu().numpy()
    return L


class HeadPatch:
    """per-head output of the attention in blocks HBLK: record the per-head contributions C_h = (A_h V_h) W_o,h^T on one pass;
    on a later pass replace the selected heads' contributions in one block by the recorded ones (resample from the unedited jet)"""

    def __init__(self, torch, model, blocks=HBLK):
        self.t, self.model, self.mode, self.C, self.target, self.chk = torch, model, None, {}, None, 0.0
        for bi in blocks:
            a = model.blocks[bi].attn; a.forward = self._wrap(bi, a, a.forward)

    def _wrap(self, bi, attn, orig):
        F = self.t.nn.functional

        def f(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, **kw):
            if self.mode is None or (self.mode == 'patch' and self.target[0] != bi):
                return orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, **kw)
            out, A = orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
            E, H = attn.embed_dim, attn.num_heads; d = E // H
            V = F.linear(value, attn.in_proj_weight[2 * E:], attn.in_proj_bias[2 * E:])
            Vh = V.view(V.shape[0], V.shape[1], H, d).permute(1, 2, 0, 3)
            Ch = self.t.einsum('bhpd,ehd->bhpe', A @ Vh, attn.out_proj.weight.view(E, H, d))
            if self.mode == 'record':
                self.C[bi] = Ch.detach()
                self.chk = max(self.chk, float((out - (Ch.sum(1).permute(1, 0, 2) + attn.out_proj.bias)).abs().max()))
            else:
                hs = self.target[1]; out = out + (self.C[bi][:, hs] - Ch[:, hs]).sum(1).permute(1, 0, 2)
            return out, None
        return f


# ================================================================================================ statistics
def bmean(boot, x, sel):
    x = np.asarray(x, np.float64); s = sel.astype(np.float64); a, b = boot.dot(x * s, s); bs = a / np.maximum(b, 1e-30)
    return float((x * s).sum() / max(s.sum(), 1e-30)), ci(bs), bs


def perm_p(torch, dev, d, n=NPERM, seed=0):
    d = torch.from_numpy(np.asarray(d, np.float64)).to(dev); obs = float(d.mean().abs()); g = torch.Generator(device=dev).manual_seed(seed); cnt = 0
    for i in range(0, n, 500):
        k = min(500, n - i); sg = torch.randint(0, 2, (k, len(d)), generator=g, device=dev).double() * 2 - 1
        cnt += int(((sg @ d).abs() / len(d) >= obs - 1e-15).sum())
    return float((1 + cnt) / (1 + n))


def mcnemar(a, b):
    from scipy.stats import binom
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum()); n = n01 + n10
    return dict(edit_only=n01, control_only=n10, p=float(min(1.0, 2 * binom.cdf(min(n01, n10), n, 0.5))) if n else 1.0)


def fit_refs(OB, y, seed):
    """HistGradientBoosting references, 2-fold cross-fitted; returns predictor(obs_dict) -> {name: logit per jet}"""
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    feats = dict(m_t21=lambda o: np.stack([np.log(o['m'] + 1), o['tau21']], 1),
                 m_t32_D2=lambda o: np.stack([np.log(o['m'] + 1), o['tau32'], np.log(np.clip(o['D2'], 1e-3, 1e4))], 1),
                 m_all=lambda o: np.stack([np.log(o['m'] + 1), o['tau21'], o['tau32'], np.log(np.clip(o['D2'], 1e-3, 1e4))], 1))
    tasks = dict(WZ=(np.isin(y, [iW, iZ, iQ]), np.isin(y, [iW, iZ])), TQ=(np.isin(y, [iT, iQ]), y == iT))
    fold = np.random.default_rng(seed).integers(0, 2, len(y)); M = {}; auc = {}
    from sklearn.metrics import roc_auc_score
    for tn, (sel, lab) in tasks.items():
        for fn, f in feats.items():
            X = f(OB); ms = [HGB(max_iter=200, learning_rate=0.1, random_state=seed).fit(X[sel & (fold == k)], lab[sel & (fold == k)]) for k in (0, 1)]
            M[(tn, fn)] = (f, ms)
            pr = np.where(fold == 0, ms[1].decision_function(X), ms[0].decision_function(X)); auc[f'{tn}|{fn}'] = float(roc_auc_score(lab[sel], pr[sel]))

    def pred(o):
        return {f'{tn}|{fn}': np.where(fold == 0, ms[1].decision_function(f(o)), ms[0].decision_function(f(o))) for (tn, fn), (f, ms) in M.items()}
    return pred, auc


# ================================================================================================ main
def main():
    import torch
    from interp.part_core import CL, PRE, cat, kin, sub, jmass, sample, inputs, build
    from interp import pn_net
    from jetdistill.part.network import FEATURES
    assert (CL.index('QCD'), CL.index('Zqq'), CL.index('Wqq'), CL.index('Tbqq')) == (iQ, iZ, iW, iT)
    OUT.mkdir(parents=True, exist_ok=True)
    R = dict(node='P21', script='interp/p21_prong_part.py', smoke=SMOKE, seed=SEED, files='_103.root', classes=CLS, per_class=PER,
             doses=SV, das_dose=DAS_S, tol_hold=TOL, tol_mass=MTOL, z_soft=ZSOFT, NB=NB, NPERM=NPERM, DAS=dict(sites=SITES, steps=P.STEPS,
             bs=P.BS, n_train=N_DAS_TR, n_eval=N_EVAL), head_blocks=[b + 1 for b in HBLK])
    save = lambda: FN.write_text(json.dumps(strip(R), indent=1))
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    # ---- sample
    Js = []
    for c in CLS:
        Js.append(P.load_rows(P.TEST / f'{PRE[c]}_103.root', PER, rng)); log(f'loaded {c}: {PER} random rows of {PRE[c]}_103.root')
    J0 = cat(Js); del Js
    keep = np.flatnonzero((J0['mask'] > 0).sum(1) >= 4); R['n_drawn'] = int(len(J0['label'])); J0 = sub(J0, keep)
    N = len(J0['label']); y = J0['label'].astype(int); R['N'] = N; log(f'N = {N} jets (>= 4 particles) of {R["n_drawn"]}')
    pt, de, dp, ms = kin(J0); Jb = build(J0, pt, de, dp, ms); pt, de, dp, ms = kin(Jb); m0 = jmass(Jb).astype(np.float64)
    Jbase = finalize(Jb, pt, de.astype(np.float64), dp.astype(np.float64), ms, m0)
    assert np.array_equal(Jbase['mask'], Jb['mask'])
    # ---- models
    M = P.make_models(['full', 'kin']); dev = M['full'].device
    PNDIR = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    PN = {n: pn_net.load_particlenet(PNDIR / f'ParticleNet_{n}.pt', len(FEATURES[n]), dev) for n in ('full', 'kin')}
    NAMES = dict(full='ParT_full', kin='ParT_kin', init='ParT_init', perm='ParT_perm', pnfull='PN_full', pnkin='PN_kin')

    def logits(mn, J):
        if mn.startswith('pn'): return pn_logits(torch, PN[mn[2:]], J, mn[2:])
        return M[mn].run(J)['logits']
    # ---- unedited accuracy
    J1 = sample(per_class=1000, start=3000); pt1, de1, dp1, ms1 = kin(J1); J1b = build(J1, pt1, de1, dp1, ms1); acc = {}
    for mn in NAMES: acc[NAMES[mn]] = float((logits(mn, J1b).argmax(1) == J1['label']).mean())
    R['accuracy_P1_rows'] = acc; log('accuracy on the P1 rows (10 classes x 1,000, built):', acc); del J1, J1b
    # ---- observables check against P18 (numpy)
    ob_base = obs_np(torch, dev, Jbase); ob_jb = obs_np(torch, dev, Jb)
    pk, ek, fk_, _ = kin(Jb); nchk = min(300, N); t21, t32, d2 = P.substructure(pk[:nchk], ek[:nchk], fk_[:nchk])
    R['obs_check_vs_p18_numpy'] = dict(tau21=float(np.abs(t21 - ob_jb['tau21'][:nchk]).max()), tau32=float(np.abs(t32 - ob_jb['tau32'][:nchk]).max()),
                                       D2_rel=float(np.abs(d2 / ob_jb['D2'][:nchk] - 1).max()))
    R['recentre_check'] = {k: dict(mean_abs_rel=float(np.mean(np.abs(ob_base[k] / np.maximum(ob_jb[k], 1e-12) - 1)))) for k in ('tau21', 'tau32', 'D2', 'm')}
    log('obs check vs P18 numpy:', R['obs_check_vs_p18_numpy'], '; recentring of the base jet:', R['recentre_check'])
    # ---- edit positions (GPU search)
    t1 = time.time(); POS = {}; INFO = {}; BASEO = {}
    for i in range(0, N, GCHUNK):
        sl = slice(i, min(N, i + GCHUNK))
        pos, info, bo = edit_positions(torch, dev, pt[sl], de[sl], dp[sl], SEED + i)
        for k, v in pos.items(): POS.setdefault(k, []).append(v)
        for k, v in info.items():
            for a, b in v.items(): INFO.setdefault(k, {}).setdefault(a, []).append(b)
        for k, v in bo.items(): BASEO.setdefault(k, []).append(v)
        log(f'  edit search: jets {sl.stop} / {N} ({time.time() - t1:.0f} s)')
    POS = {k: (np.concatenate([a[0] for a in v]), np.concatenate([a[1] for a in v])) for k, v in POS.items()}
    INFO = {k: {a: np.concatenate(b) for a, b in v.items()} for k, v in INFO.items()}
    # ---- rebuild all variants, observables, logits
    pos_base_nets = ['full', 'kin', 'pnfull', 'pnkin', 'init', 'perm']
    LG = {mn: {} for mn in NAMES}; OB = {'base': ob_base}; JDAS = {}
    for mn in pos_base_nets: LG[mn]['base'] = logits(mn, Jbase)
    R['accuracy_sample'] = {NAMES[mn]: float((LG[mn]['base'].argmax(1) == y).mean()) for mn in NAMES}
    R['accuracy_sample_by_class'] = {NAMES[mn]: {c: float((LG[mn]['base'].argmax(1)[y == CL.index(c)] == CL.index(c)).mean()) for c in CLS} for mn in NAMES}
    log('accuracy on this sample (10-class argmax, rebuilt base jets):', R['accuracy_sample'])
    das_pool = None
    variants = [(k, s, v) for k in KINDS for s in SV for v in ('E', 'rot', 'rc', 'raw')] + [('scrF', None, 'E')]
    for (kind, s, v) in variants:
        name = 'scrF' if kind == 'scrF' else f'{kind}_{s}_{v}'
        if kind == 'scrF': Je = P.make_edit(Jbase, 'scr', 1.0, np.random.default_rng(SEED + 5))
        else: Je = finalize(Jb, pt, *POS[(kind, s, v)], ms, m0)
        assert np.array_equal(Je['mask'], Jbase['mask']), name
        OB[name] = obs_np(torch, dev, Je)
        nets_here = pos_base_nets if (kind == 'scrF' or (s == DAS_S and v != 'raw')) else ['full', 'kin', 'pnfull', 'pnkin']
        for mn in nets_here: LG[mn][name] = logits(mn, Je)
        if v == 'E' and (kind == 'scrF' or s == DAS_S): JDAS[kind] = Je
        log(f'  variant {name}: nets {nets_here}; mean dtau21 {np.mean(OB[name]["tau21"] - ob_base["tau21"]):+.4f} dtau32 '
            f'{np.mean(OB[name]["tau32"] - ob_base["tau32"]):+.4f} dlogD2 {np.mean(np.log(OB[name]["D2"] / ob_base["D2"])):+.4f} '
            f'|dm/m| max {np.max(np.abs(OB[name]["m"] / np.maximum(ob_base["m"], 1e-9) - 1)):.2e}')
    # ---- acceptance
    rel = lambda name, k: OB[name][k] / np.maximum(ob_base[k], 1e-12) - 1
    ACC = {}; R['edits'] = {}
    for kind in KINDS:
        for s in SV:
            nm = f'{kind}_{s}_E'
            a = (np.abs(rel(nm, HOLD[kind][0])) <= TOL) & (np.abs(rel(nm, HOLD[kind][1])) <= TOL) & (np.abs(rel(nm, 'm')) <= MTOL)
            ACC[(kind, s)] = a; e = dict(accepted=int(a.sum()), accepted_frac=float(a.mean()),
                                         accepted_by_class={c: float(a[y == CL.index(c)].mean()) for c in CLS},
                                         search_dev_median=float(np.median(INFO[(kind, s)]['dev'])), comp_params_median=[float(np.median(INFO[(kind, s)]['p'])), float(np.median(INFO[(kind, s)]['q']))])
            for v in ('E', 'rot', 'rc', 'raw'):
                n2 = f'{kind}_{s}_{v}'
                e[v] = {k: dict(mean_rel=float(np.mean(rel(n2, k)[a])), sd_rel=float(np.std(rel(n2, k)[a])), mean_abs_rel=float(np.mean(np.abs(rel(n2, k)[a]))),
                                frac_abs_rel_lt_1pct=float(np.mean(np.abs(rel(n2, k)[a]) < 0.01)), mean_abs=float(np.mean((OB[n2][k] - ob_base[k])[a])))
                        for k in ('tau21', 'tau32', 'D2', 'm', 'tau1', 'tau2', 'tau3')}
            R['edits'][f'{kind}_{s}'] = e
            log(f'  {kind} s={s}: accepted {a.sum()} / {N}; rel change (E) prim {PRIM[kind]} {e["E"][PRIM[kind]]["mean_rel"]:+.3f}, '
                f'rot tau21/tau32/D2 {e["rot"]["tau21"]["mean_rel"]:+.3f}/{e["rot"]["tau32"]["mean_rel"]:+.3f}/{e["rot"]["D2"]["mean_rel"]:+.3f}')
    R['edits']['scrF'] = {k: dict(mean_rel=float(np.mean(rel('scrF', k))), mean_abs=float(np.mean(OB['scrF'][k] - ob_base[k]))) for k in ('tau21', 'tau32', 'D2', 'm')}
    save()
    # ---- behaviour
    boot = P.Boot(torch, dev, N, NB, SEED + 21)
    sc = lambda L: dict(dWQ=L[:, iW] - L[:, iQ], dTQ=L[:, iT] - L[:, iQ])
    R['behaviour'] = {}; R['edit_vs_control'] = {}; ptests = []
    for mn in NAMES:
        B0 = sc(LG[mn]['base']); c0 = LG[mn]['base'].argmax(1); bh = {}; evc = {}
        for name, L in LG[mn].items():
            if name == 'base': continue
            if name == 'scrF': sel = np.ones(N, bool)
            else: kind, s = name.split('_')[0], float(name.split('_')[1]); sel = ACC[(kind, s)]
            S = sc(L); ch = L.argmax(1) != c0; o = dict(n=int(sel.sum()), calls_changed=float(ch[sel].mean()))
            for k in ('dWQ', 'dTQ'):
                est, cc, _ = bmean(boot, S[k] - B0[k], sel); o[k] = est; o[k + '_ci'] = cc
                o[k + '_by_class'] = {c: float((S[k] - B0[k])[sel & (y == CL.index(c))].mean()) for c in CLS}
            bh[name] = o
        for kind in KINDS:
            for s in SV:
                if f'{kind}_{s}_E' not in LG[mn]: continue
                sel = ACC[(kind, s)]; LE = LG[mn][f'{kind}_{s}_E']; o = {}
                for ctl in ('rot', 'rc', 'raw'):
                    if f'{kind}_{s}_{ctl}' not in LG[mn]: continue
                    LC = LG[mn][f'{kind}_{s}_{ctl}']; oc = {}
                    for k in ('dWQ', 'dTQ'):
                        d = (sc(LE)[k] - sc(LC)[k]); est, cc, _ = bmean(boot, d, sel)
                        oc[k] = dict(diff=est, ci=cc, p_perm=perm_p(torch, dev, d[sel], seed=SEED + 7))
                    oc['mcnemar'] = mcnemar((LE.argmax(1) != c0)[sel], (LC.argmax(1) != c0)[sel]); o[ctl] = oc
                    if ctl == 'rot' and mn in ('full', 'kin', 'pnfull', 'pnkin'): ptests.append((mn, f'{kind}_{s}', PSCORE[kind], oc[PSCORE[kind]]['p_perm']))
                evc[f'{kind}_{s}'] = o
        R['behaviour'][NAMES[mn]] = bh; R['edit_vs_control'][NAMES[mn]] = evc
        log(f'  behaviour {NAMES[mn]}: ' + '; '.join(f'{n} dWQ {v["dWQ"]:+.3f} dTQ {v["dTQ"]:+.3f} ch {v["calls_changed"]:.3f}' for n, v in bh.items() if n.endswith('_E') or n == 'scrF'))
    for (mn, ks, k, p), h in zip(ptests, holm([t[3] for t in ptests])):
        R['edit_vs_control'][NAMES[mn]][ks]['rot'][k]['p_holm'] = h
    save()
    # ---- reference classifiers and slopes
    pred, auc = fit_refs(ob_base, y, SEED + 3); R['ref_auc'] = auc; P0 = pred(ob_base); R['reference'] = {}; R['slopes'] = {}
    for kind in KINDS:
        for s in SV:
            sel = ACC[(kind, s)]; o = {}
            for v in ('E', 'rot', 'raw'):
                Pv = pred(OB[f'{kind}_{s}_{v}']); o[v] = {nm: bmean(boot, Pv[nm] - P0[nm], sel)[:2] for nm in Pv}
            R['reference'][f'{kind}_{s}'] = o
    for kind in KINDS:
        dprim = lambda nm: (OB[nm][PRIM[kind]] - ob_base[PRIM[kind]]) if kind != 'E3' else np.log(OB[nm]['D2'] / ob_base['D2'])
        for s in SV:
            sel = ACC[(kind, s)]; nm = f'{kind}_{s}_E'; dx = dprim(nm) * sel; Pv = pred(OB[nm]); o = {}
            ref_nm = dict(E1='WZ|m_t21', E2='TQ|m_t32_D2', E3='WZ|m_all')[kind]; rk = ref_nm.split('|')[0]
            rd = (Pv[ref_nm] - P0[ref_nm]) * sel; a, b = boot.dot(rd * dx, dx * dx); rbs = a / np.maximum(b, 1e-30)
            o['reference'] = dict(name=ref_nm, slope=float((rd * dx).sum() / max((dx * dx).sum(), 1e-30)), ci=ci(rbs))
            for mn in ('full', 'kin', 'pnfull', 'pnkin'):
                k = 'dWQ' if rk == 'WZ' else 'dTQ'; d = (sc(LG[mn][nm])[k] - sc(LG[mn]['base'])[k]) * sel
                a2, _ = boot.dot(d * dx, dx * dx); nbs = a2 / np.maximum(b, 1e-30)
                o[NAMES[mn]] = dict(score=k, slope=float((d * dx).sum() / max((dx * dx).sum(), 1e-30)), ci=ci(nbs),
                                    ratio_to_ref=float((d * dx).sum() / (rd * dx).sum()) if abs((rd * dx).sum()) > 1e-12 else None,
                                    ratio_ci=ci(nbs / np.where(np.abs(rbs) > 1e-12, rbs, np.nan)))
            R['slopes'][f'{kind}_{s}'] = o
    log('reference AUCs:', auc); log('slopes:', json.dumps(strip({k: {a: (b['slope'], b.get('ratio_to_ref')) for a, b in v.items()} for k, v in R['slopes'].items()})))
    save(); del LG
    # ---- DAS per site
    pool = np.flatnonzero(ACC[('E1', DAS_S)] & ACC[('E2', DAS_S)] & ACC[('E3', DAS_S)])
    R['das_pool'] = int(len(pool)); n_ev = min(N_EVAL, len(pool) // 4); n_tr = min(N_DAS_TR, len(pool) - n_ev)
    pr = np.random.default_rng(SEED + 9).permutation(pool); didx = np.sort(pr[:n_tr + n_ev]); R['das_n'] = dict(train=int(n_tr), eval=int(n_ev))
    loc = np.random.default_rng(SEED + 10).permutation(len(didx)); tr_i, ev_i = np.sort(loc[:n_tr]), np.sort(loc[n_tr:])
    D = {}
    for fk in ('full', 'kin'):
        D[fk] = {'base': inputs(sub(Jbase, didx), fk)}
        for kind in list(KINDS) + ['scrF']: D[fk][kind] = inputs(sub(JDAS[kind], didx), fk)
    cnt = (Jbase['mask'][didx] > 0).sum(1)
    P.EDITS = list(KINDS) + ['scrF']; P.SITES = SITES; P.VARS = {k: k for k in P.EDITS}
    nets = {mn: P.Net(M[mn]) for mn in ('full', 'kin', 'init')}; boot_ev = P.Boot(torch, dev, len(ev_i), NB, SEED + 22); O = {}; R['das'] = {}
    for mn, net in nets.items():
        t1 = time.time(); O[mn], _, fits = P.das_model(net, D[P.FEAT_OF(mn)], cnt, tr_i, ev_i, np.random.default_rng(SEED + 32), None, NAMES[mn], boot_ev)
        R['das'][NAMES[mn]] = dict(variables=O[mn], fits=fits)
        for ed in P.EDITS:
            v = O[mn][ed]; log(f'  DAS {NAMES[mn]} {ed}: IIA ' + ' '.join(f'{s}={v["sites"][s]["das"]["iia"]:.2f}' for s in SITES) + f' (calls changed {v["n_call_changed"]}/{v["N_eval"]})')
        log(f'  DAS {NAMES[mn]} done ({time.time() - t1:.0f} s)'); save()
    R['das_vs_init'] = {}
    for mn in ('full', 'kin'):
        o = {}
        for ed in P.EDITS:
            A, B = O[mn][ed]['sites'], O['init'][ed]['sites']
            dd = (A['b7']['das']['_bs'] - A['emb']['das']['_bs']) - (B['b7']['das']['_bs'] - B['emb']['das']['_bs'])
            o[ed] = dict(per_site={s: dict(diff=float(A[s]['das']['iia'] - B[s]['das']['iia']), ci=ci(A[s]['das']['_bs'] - B[s]['das']['_bs'])) for s in SITES},
                         did_b7_emb=float(np.mean(dd)), did_ci=ci(dd), did_p=P.p_two(dd),
                         b7_minus_emb=float(A['b7']['das']['iia'] - A['emb']['das']['iia']), b7_minus_emb_ci=ci(A['b7']['das']['_bs'] - A['emb']['das']['_bs']))
        R['das_vs_init'][NAMES[mn]] = o
        log(f'  DAS {NAMES[mn]} vs init, DiD (b7 - emb):', {ed: (round(v['did_b7_emb'], 3), [round(x, 3) for x in v['did_ci']]) for ed, v in o.items()})
    save()
    for net in nets.values():
        for h in net.hooks: h.remove()
    # ---- head patching, blocks 5-8
    R['heads'] = {}; ev_g = didx[ev_i]; order = ev_g[np.argsort(cnt[ev_i], kind='stable')]; nE = len(order)
    Bh = P.Boot(torch, dev, nE, NB, SEED + 23); hp_p = {ed: [] for ed in P.EDITS}
    for mn in ('full', 'kin', 'init', 'perm'):
        fk = P.FEAT_OF(mn); HP = HeadPatch(torch, M[mn].model); model = M[mn].model
        Lb = np.zeros((nE, 10)); Le = {ed: np.zeros((nE, 10)) for ed in P.EDITS}; Lp = {}
        sets = [(bi, [h]) for bi in HBLK for h in range(8)] + [(bi, list(range(8))) for bi in HBLK]
        t1 = time.time()
        for i in range(0, nE, 128):
            rows = np.arange(i, min(nE, i + 128)); idx = order[rows]
            T = lambda trip: tuple(torch.from_numpy(np.ascontiguousarray(a)).to(dev) for a in trip)
            with torch.no_grad():
                xb = T(P.slc(inputs(sub(Jbase, idx), fk), np.arange(len(idx)))); HP.mode = 'record'; Lb[rows] = model(*xb).double().cpu().numpy()
                for ed in P.EDITS:
                    HP.mode = None; xe = T(P.slc(inputs(sub(JDAS[ed], idx), fk), np.arange(len(idx)))); Le[ed][rows] = model(*xe).double().cpu().numpy()
                    for (bi, hs) in sets:
                        HP.mode = 'patch'; HP.target = (bi, hs); Lp.setdefault((ed, bi, len(hs) == 8, hs[0]), np.zeros((nE, 10)))[rows] = model(*xe).double().cpu().numpy()
                HP.mode = None
        log(f'  head patching {NAMES[mn]}: {len(sets)} head sets x {len(P.EDITS)} edits on {nE} jets ({time.time() - t1:.0f} s); '
            f'per-head decomposition check max |out - sum_h C_h - b| = {HP.chk:.2e}')
        o = dict(decomposition_check=HP.chk)
        for ed in P.EDITS:
            k = PSCORE[ed]; s_ = (lambda L: L[:, iW] - L[:, iQ]) if k == 'dWQ' else (lambda L: L[:, iT] - L[:, iQ])
            e0 = ((Le[ed] - Lb) ** 2).sum(1); de0 = s_(Le[ed]) - s_(Lb); oe = dict(score=k, mean_sq_logit_change=float(e0.mean()), mean_score_change=float(de0.mean()), heads={}, blocks={})
            for (ed2, bi, allh, h), L in Lp.items():
                if ed2 != ed: continue
                e = ((L - Lb) ** 2).sum(1); a, b = Bh.dot(e, e0); rb = 1 - a / np.maximum(b, 1e-30)
                ds = s_(Le[ed]) - s_(L); a2, b2 = Bh.dot(ds, de0); sb = a2 / np.where(np.abs(b2) > 1e-12, b2, np.nan)
                cell = dict(restoration=float(1 - e.sum() / max(e0.sum(), 1e-30)), restoration_ci=ci(rb), p_one_sided=float(max(1 / NB, (rb <= 0).mean())),
                            score_share=float(ds.sum() / de0.sum()) if abs(de0.sum()) > 1e-12 else None, score_share_ci=ci(sb), _rb=rb)
                if allh: oe['blocks'][f'b{bi + 1}'] = cell
                else:
                    oe['heads'][f'b{bi + 1}h{h + 1}'] = cell
                    if mn in ('full', 'kin'): hp_p[ed].append((mn, f'b{bi + 1}h{h + 1}', cell['p_one_sided']))
            o[ed] = oe
        R['heads'][NAMES[mn]] = o
        for ed in P.EDITS:
            log(f'  heads {NAMES[mn]} {ed}: block restoration ' + ' '.join(f'{b}={v["restoration"]:.3f}' for b, v in o[ed]['blocks'].items()) +
                '; top heads ' + ', '.join(f'{h} {v["restoration"]:.3f}' for h, v in sorted(o[ed]['heads'].items(), key=lambda t: -t[1]['restoration'])[:4]))
        for bi in HBLK: del model.blocks[bi].attn.forward
    for ed, lst in hp_p.items():
        for (mn, h, p), ph in zip(lst, holm([t[2] for t in lst])): R['heads'][NAMES[mn]][ed]['heads'][h]['p_holm_64'] = ph
    R['heads_vs_init'] = {}
    for mn in ('full', 'kin'):
        o = {}
        for ed in P.EDITS:
            o[ed] = {}
            for b in [f'b{bi + 1}' for bi in HBLK]:
                for ctl in ('init', 'perm'):
                    A = R['heads'][NAMES[mn]][ed]['blocks'][b]; Bc = R['heads'][NAMES[ctl]][ed]['blocks'][b]; d = A['_rb'] - Bc['_rb']
                    o[ed][f'{b}_vs_{ctl}'] = dict(diff=float(A['restoration'] - Bc['restoration']), ci=ci(d), p=P.p_two(d))
        R['heads_vs_init'][NAMES[mn]] = o
    R['runtime_s'] = time.time() - P.T0; save(); log('wrote', FN)
    try: plot(json.loads(FN.read_text()))
    except Exception as e: log('plot failed:', repr(e))


# ================================================================================================ figure
def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = res or json.loads(FN.read_text()); s0 = res['das_dose']
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    a = ax[0]; nets = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin']; ed = ['E1', 'E2', 'E3']; w = 0.18
    lab = {'E1': r'$\tau_{21}$ edit', 'E2': r'$\tau_{32}$ edit', 'E3': r'$D_2$ edit'}
    for j, k in enumerate(('dWQ', 'dTQ')):
        for i, n in enumerate(nets):
            v = [res['edit_vs_control'][n][f'{e}_{s0}']['rot'][k]['diff'] for e in ed]
            lo = [v[t] - res['edit_vs_control'][n][f'{e}_{s0}']['rot'][k]['ci'][0] for t, e in enumerate(ed)]
            hi = [res['edit_vs_control'][n][f'{e}_{s0}']['rot'][k]['ci'][1] - v[t] for t, e in enumerate(ed)]
            xs = np.arange(3) + 4 * j + (i - 1.5) * w
            a.bar(xs, v, w, yerr=[lo, hi], color=['#1f4e79', '#5b9bd5', '#c55a11', '#f4b183'][i], label=n if j == 0 else None)
        ref = [res['reference'][f'{e}_{s0}']['E'][('WZ' if k == 'dWQ' else 'TQ') + '|m_all'][0] - res['reference'][f'{e}_{s0}']['rot'][('WZ' if k == 'dWQ' else 'TQ') + '|m_all'][0] for e in ed]
        a.plot(np.arange(3) + 4 * j, ref, 'kx', ms=9, mew=2, label='GBDT(m, τ21, τ32, D2)' if j == 0 else None)
    a.axhline(0, color='k', lw=0.6); a.set_xticks(list(range(3)) + list(range(4, 7)))
    a.set_xticklabels([lab[e] for e in ed] * 2, fontsize=8, rotation=20)
    a.set_ylabel('edit − rotation control, mean Δlogit'); a.set_title(f'(a) logit change at dose {s0}: W vs QCD (left), top vs QCD (right)', fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=0.3, axis='y')
    a = ax[1]; S = res['DAS']['sites']; x = np.arange(len(S)); col = dict(E1='#1f4e79', E2='#c55a11', E3='#548235', scrF='#7f7f7f')
    for e in ['E1', 'E2', 'E3', 'scrF']:
        for n, ls in (('ParT_full', '-o'), ('ParT_init', ':')):
            d = res['das'][n]['variables'][e]['sites']; a.plot(x, [d[s]['das']['iia'] for s in S], ls, ms=3, color=col[e],
                                                                 label=(lab.get(e, 'full scramble') if n == 'ParT_full' else None))
    a.set_xticks(x); a.set_xticklabels(S); a.set_ylim(-0.1, 1); a.set_ylabel('1-d interchange: logit recovery (IIA)')
    a.set_title('(b) ParT_full by site (solid) vs untrained ParT (dotted)', fontsize=9); a.legend(fontsize=7); a.grid(alpha=0.3)
    a = ax[2]; bl = [f'b{b + 1}' for b in HBLK]
    for e in ['E1', 'E2', 'E3', 'scrF']:
        for n, ls in (('ParT_full', '-o'), ('ParT_init', ':'), ('ParT_perm', '--')):
            a.plot(range(4), [res['heads'][n][e]['blocks'][b]['restoration'] for b in bl], ls, ms=4, color=col[e], label=(lab.get(e, 'full scramble') if n == 'ParT_full' else None))
    a.set_xticks(range(4)); a.set_xticklabels(['block 5', 'block 6', 'block 7', 'block 8']); a.axhline(0, color='k', lw=0.6)
    a.set_ylabel('share of logit change removed by restoring the block\'s 8 heads'); a.set_title('(c) ParT_full attention blocks (dotted: untrained, dashed: permuted)', fontsize=9)
    a.legend(fontsize=7); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG, dpi=110); print('wrote', FIG)


if __name__ == '__main__':
    plot() if PLOT else main()
