"""P18 -- ParT stage pipeline (claim C4; A8): at which block do mass, prong and count variables become readable, and at which
block does a one-dimensional interchange of the variable reproduce the network's response to a real change of it?

Models: released ParT_full and ParT_kin (JetClass, weaver-core 0.4.17). Controls (ParT_full architecture, full inputs):
  perm : the released ParT_full with, in every Linear / Conv1d / attention in-projection, the output rows of the weight (and the
         matching bias entries) permuted by an independent random permutation per layer (seed 18); norms, cls_token untouched.
  init : the ParT_full architecture at weaver's default initialisation (torch.manual_seed(18)), no training.
  rot  : the ParT_full features multiplied by a random orthogonal 128 x 128 matrix (seed 18) before the probe (basis control).
Sample: JetClass test_20M, <sample>_100.root of each of the 10 classes, PER random rows per class (numpy default_rng(18), never
first-k), N = 10 x PER (50,000 in the full run). Train / test halves from a permutation (seed 18); ridge alpha and MLP early stop
on the last quarter of the train half. Unedited accuracy also on the P1 rows (3,000-4,000 of each class's _100 file).
Build: part_core.build (eta = jet_eta + sign(jet_eta) * deta), all px, py, pz, E, jet_pt, jet_energy rebuilt from the particles.

Targets (about the stored jet axis; dR = hypot(deta, dphi)):
  log_minv = log invariant mass [GeV];  log_mA = 0.5 log(M0 M2), M0 = sum pT, M2 = sum pT dR^2 (axis-moment mass);
  log_r2 = log(M2 / M0) (pT-weighted mean squared radius, the per-particle radial moment);  tau21, tau32 (beta = 1, k-means axes
  as jetdistill.observables.compute);  log_D2 (beta = 1, 24 hardest);  N = number of particles;  log_pT = log jet pT.
Edits (every real particle; pT order kept except soft5):
  phi   : dphi + 0.06 (invariant mass, pair features unchanged; log mA up)
  dil   : deta, dphi x k_match, k_match = exp(mean 0.5 dlog M2 of phi) (matched real dilation: same mean dlog mA as phi)
  scr   : per-particle azimuth scramble about the stored axis (theta_i ~ U(0, 2 pi); pT_i, r_i, rank fixed -> M0, M2, N fixed;
          tau21, tau32, D2, m_inv change) (E24-style)
  soft5 : 5 on-manifold soft particles (H18-style), each a copy of a random member of the softer half of the jet (pT, mass,
          charge / PID / impact-parameter fields) at its donor's distance from the axis and a random azimuth (N + 5)
  pt    : pT and particle mass x 1.1 (log pT + 0.095; angles fixed)
Calibration edit per target (the matched real change): log_minv, log_mA, log_r2 -> dil; tau21, tau32, log_D2 -> scr;
  N -> soft5; log_pT -> pt.
Representations ("sites"): emb (embedding output) and b1..b8 (particle-block outputs) as the pT-weighted jet mean of the particle
  tokens; c1, c2 (class-attention block outputs) as the class token. (Class blocks do not update particle tokens.)
(a) Readability: ridge (p1v Ridge) and 2-layer MLP (128-64-1, GELU, Adam, early stop on val) probes fitted on the base jets of the
  train half; R^2 on the test half with NB jet-bootstrap resamples; controls perm / init / rot at each site.
  Indices per target: onset = first site whose R^2 lower bound exceeds the larger control's upper bound; t90 = first site with
  R^2 >= 0.9 x the target's maximum over sites (bootstrap distribution, paired over targets); rise = R^2(best) - R^2(emb).
(b) Calibrated edit response: per site, target and edit, the probe read-out change dp = p(edited) - p(base) on the test half vs
  the true change dt; gain g = sum(dp dt) / sum(dt^2) (slope through 0). g under the target's calibration edit is the fraction
  of a real change of the variable that the site's read-out follows ("moves at full size" at g = 1). Also mean dp / mean dt per
  edit, and for log_mA the ratio dp(phi) / dp(dil) (1 = reads phi as an m_A change; 0 = reads invariant mass).
(c) Causal 1-d interchange (DAS) per site and variable (edit): base jet x, counterfactual = the same jet with the edit applied to
  the per-particle inputs and the base 4-vectors (pair features) kept (soft5: the full edited jet), as P8. At particle sites the
  pT-weighted mean along q is set to the counterfactual's: h_p <- h_p + ((src - mean_w h) . q) q for every particle; at c1 / c2
  the class token: h <- h + ((src - h) . q) q. q (unit) fitted by Adam (STEPS x BS jets, lr 0.01, unit-norm random start) on the DAS-train jets (train
  half) so that the interchanged logits match the counterfactual logits (MSE). Evaluation on N_EVAL test-half jets:
  IIA (logit recovery) = 1 - sum ||L_int - L_cf||^2 / sum ||L_base - L_cf||^2; call IIA = share of call-changed jets that get the
  counterfactual call. Directions evaluated: DAS q, the ridge-probe direction of the variable's target, NRAND_EXACT random q (exact
  runs) and NRAND_LIN = 1,000 random q through the first-order (Jacobian) approximation of the interchange (checked against the
  exact random runs and the DAS q). Controls: random-direction null (p = (1 + #{lin random >= lin DAS}) / (1 + NRAND_LIN));
  DAS refitted on the perm network at the same site (paired bootstrap over jets of IIA_trained - IIA_perm). Holm over
  11 sites x 5 variables per model for each null. IIA step = the largest increase of DAS IIA between consecutive sites.
  Variable -> target: phi -> log_mA (at fixed m_inv); dil -> log_minv (and log_mA); scr -> tau21 (tau32, D2 move together);
  soft5 -> N; pt -> log_pT.
H0: R^2 and IIA of every target flat across depth (no stage order). H1 (C4 / JEDI-like): N, log_pT, log_r2 reach t90 in blocks
1-2, tau21 / tau32 / D2 later, log_mA last with an IIA step. H_axis-alt: log_mA readable from block 1 with no later increase.

  ./sync.sh && ./part_job.sh c-p18-smoke 'python -m interp.p18_stage_pipeline_part --smoke'
  ./sync.sh && ./part_job.sh c-p18-run 'python -m interp.p18_stage_pipeline_part --models full,kin'
  python -m interp.p18_stage_pipeline_part --plot   (figure from the JSON; CPU)
"""
import argparse, contextlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser()
ap.add_argument('--smoke', action='store_true'); ap.add_argument('--models', default='full,kin'); ap.add_argument('--plot', action='store_true')
ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED = 18
TEST = Path(os.environ.get('P18_TEST', '/j-jepa-vol/JetClass/Pythia/test_20M'))
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = OUT / ('p18_stage_pipeline_smoke.json' if SMOKE else 'p18_stage_pipeline.json')
FIG = OUT / ('p18_stage_pipeline_smoke.png' if SMOKE else 'p18_stage_pipeline.png')
PER = 200 if SMOKE else 5000
MAXROW = 30000 if SMOKE else None
NB = 200 if SMOKE else 1000
CHUNK = 2000 if SMOKE else 2500
N_DAS_TR = 600 if SMOKE else 4000
N_EVAL = 200 if SMOKE else 1000
STEPS = 15 if SMOKE else 200
BS = 64 if SMOKE else 128
NRAND_EXACT = 3 if SMOKE else 10
NRAND_LIN = 1000
BATCH = 256
C = 128
SITES = ['emb'] + [f'b{l}' for l in range(1, 9)] + ['c1', 'c2']
SKEY = {'emb': ('e',), 'c1': ('c', 0), 'c2': ('c', 1), **{f'b{l}': ('b', l - 1) for l in range(1, 9)}}
TARGETS = ['N', 'log_pT', 'log_r2', 'tau21', 'tau32', 'log_D2', 'log_minv', 'log_mA']
EDITS = ['phi', 'dil', 'scr', 'soft5', 'pt']
CAL = dict(log_minv='dil', log_mA='dil', log_r2='dil', tau21='scr', tau32='scr', log_D2='scr', N='soft5', log_pT='pt')
VARS = dict(phi='log_mA', dil='log_minv', scr='tau21', soft5='N', pt='log_pT')
GROUPS = dict(low=['N', 'log_pT', 'log_r2'], prong=['tau21', 'tau32', 'log_D2'], mass=['log_mA'])
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def strip(o):
    if isinstance(o, dict): return {str(k): strip(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [strip(v) for v in o]
    if isinstance(o, np.integer): return o.item()
    if isinstance(o, np.floating): o = float(o)
    if isinstance(o, np.ndarray): return strip(o.tolist())
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def p_two(bs, v0=0.0):
    bs = np.asarray(bs, float); bs = bs[np.isfinite(bs)]; nb = max(len(bs), 1)
    return float(max(1.0 / nb, min(1.0, 2 * min((bs <= v0).mean(), (bs >= v0).mean())))) if len(bs) else 1.0


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


# ------------------------------------------------------------------------------------------------------------ physics
R0 = 0.8


def kmeans(pt, eta, phi, k, iters=6):           # jetdistill.observables.compute._kmeans (copied verbatim via e24)
    J = len(pt); ax = np.zeros((J, k, 2)); pts = np.stack([eta, phi], -1)
    ax[:, 0] = pts[np.arange(J), pt.argmax(1)]
    for j in range(1, k):
        d = np.min(np.linalg.norm(pts[:, :, None] - ax[:, None, :j], axis=-1), -1) * pt
        ax[:, j] = pts[np.arange(J), d.argmax(1)]
    for _ in range(iters):
        a = np.linalg.norm(pts[:, :, None] - ax[:, None], axis=-1).argmin(-1)
        for j in range(k):
            w = pt * (a == j); s = w.sum(1, keepdims=True)
            ax[:, j] = np.where(s > 0, (w[..., None] * pts).sum(1) / np.maximum(s, 1e-12), ax[:, j])
    return ax, np.linalg.norm(pts[:, :, None] - ax[:, None], axis=-1)


def substructure(pt, eta, phi, chunk=1000):
    """tau21, tau32 (beta = 1, k-means axes) and D2 (beta = 1, 24 hardest), as interp/e24_azimuth_prong.substructure"""
    t21, t32, d2 = [], [], []
    for i in range(0, len(pt), chunk):
        p, e, f = (np.asarray(a[i:i + chunk], np.float64) for a in (pt, eta, phi)); z = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
        tau = {k: (z * kmeans(p, e, f, k)[1].min(-1)).sum(1) / R0 for k in (1, 2, 3)}
        t21.append(tau[2] / np.maximum(tau[1], 1e-12)); t32.append(tau[3] / np.maximum(tau[2], 1e-12))
        top = np.argsort(-p, 1, kind='stable')[:, :24]
        zt, et, pht = (np.take_along_axis(a, top, 1) for a in (z, e, f))
        R = np.sqrt((et[:, :, None] - et[:, None]) ** 2 + (pht[:, :, None] - pht[:, None]) ** 2)
        W = zt[:, :, None] * zt[:, None, :] * R
        e2 = W.sum((1, 2)) / 2; RzR = np.einsum('jik,jk,jkl->jil', R, zt, R); e3 = (W * RzR).sum((1, 2)) / 6
        d2.append(e3 / np.maximum(e2 ** 3, 1e-12))
    return np.concatenate(t21), np.concatenate(t32), np.concatenate(d2)


def targets(J):
    from interp.part_core import kin, jmass
    pt, de, dp, _ = kin(J); pt = pt.astype(np.float64); de = de.astype(np.float64); dp = dp.astype(np.float64)
    M0 = pt.sum(1); M2 = (pt * (de ** 2 + dp ** 2)).sum(1); m = np.maximum(jmass(J).astype(np.float64), 1.0)
    t21, t32, d2 = substructure(pt, de, dp)
    return dict(N=(J['mask'] > 0).sum(1).astype(np.float64), log_pT=np.log(np.maximum(J['jet_pt'].astype(np.float64), 1e-3)),
                log_r2=np.log(np.maximum(M2 / np.maximum(M0, 1e-12), 1e-12)), tau21=np.clip(t21, 0, 2), tau32=np.clip(t32, 0, 2),
                log_D2=np.log(np.clip(d2, 1e-3, 1e4)), log_minv=np.log(m), log_mA=0.5 * np.log(np.maximum(M0 * M2, 1e-12)))


EXTRA = ['charge', 'isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon', 'd0val', 'd0err', 'dzval', 'dzerr']


def make_edit(J, name, k_match, rng):
    from interp.part_core import kin, build
    pt, de, dp, ms = kin(J); r = (pt > 0).astype(np.float32)
    if name == 'base': return build(J, pt, de, dp, ms)
    if name == 'phi': return build(J, pt, de, dp + 0.06 * r, ms)
    if name == 'dil': return build(J, pt, de * k_match, dp * k_match, ms)
    if name == 'pt': return build(J, pt * 1.1, de, dp, ms * 1.1)
    if name == 'scr':
        rr = np.hypot(de, dp); th = rng.uniform(0, 2 * np.pi, de.shape).astype(np.float32)
        return build(J, pt, rr * np.cos(th) * r, rr * np.sin(th) * r, ms)
    if name == 'soft5':
        K = 5; n = (J['mask'] > 0).sum(1); h = np.maximum(n // 2, 1); rows = np.arange(len(n))
        ext = {k: np.concatenate([J[k], np.zeros((len(n), K), np.float32)], 1) for k in EXTRA}
        P2 = [np.concatenate([a, np.zeros((len(n), K), np.float32)], 1) for a in (pt, de, dp, ms)]
        for j in range(K):
            d = n - h + np.minimum((rng.random(len(n)) * h).astype(int), h - 1); d = np.clip(d, 0, 127)
            th = 2 * np.pi * rng.random(len(n)); rr = np.hypot(de[rows, d], dp[rows, d]); col = 128 + j
            P2[0][:, col] = pt[rows, d] * (n > 0); P2[1][:, col] = rr * np.cos(th); P2[2][:, col] = rr * np.sin(th); P2[3][:, col] = ms[rows, d]
            for k in EXTRA: ext[k][:, col] = J[k][rows, d]
        ext['mask'] = (P2[0] > 0).astype(np.float32)
        return build(J, *P2, extra=ext)
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------------------ data
def load_rows(path, n, rng, chunk=25000):
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries
    if MAXROW: N = min(N, MAXROW)
    rows = np.sort(rng.choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r): parts.append(sub(read_root(path, s, min(s + chunk, N)), r - s))
    return cat(parts)


# ------------------------------------------------------------------------------------------------------------ models
def make_models(names):
    import torch
    from interp.part_core import ParTNetwork
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import FEATURES, CFG
    M = {}
    for n in names: M[n] = ParTNetwork(n)
    pn = ParTNetwork('full'); g = torch.Generator().manual_seed(SEED); sd = pn.model.state_dict(); new = {}
    perms = {}
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
    pn.model.load_state_dict(new); pn.model.eval(); M['perm'] = pn
    pi = ParTNetwork('full'); torch.manual_seed(SEED); fresh = ParticleTransformer(input_dim=len(FEATURES['full']), **CFG)
    pi.model.load_state_dict(fresh.state_dict()); pi.model.eval(); M['init'] = pi
    log(f'models: {list(M)}; perm: {len(perms)} weight tensors row-permuted')
    return M


FEAT_OF = lambda mname: 'kin' if mname == 'kin' else 'full'


class Net:
    """forward hooks on embed, every particle block, both class blocks. fn[key] = f(out) applied in the forward; want = keys to
    record (pooled: particle sites -> pT-weighted mean over real particles; class sites -> token)."""

    def __init__(self, pn):
        import torch
        self.t, self.pn, self.m, self.dev = torch, pn, pn.model, pn.device
        for p in self.m.parameters(): p.requires_grad_(False)
        self.fn, self.want, self.rec, self.w = {}, set(), {}, None
        mods = {('e',): self.m.embed}
        for i, b in enumerate(self.m.blocks): mods[('b', i)] = b
        for i, b in enumerate(self.m.cls_blocks): mods[('c', i)] = b
        self.hooks = [mod.register_forward_hook(self._h(k)) for k, mod in mods.items()]

    def pool(self, key, out):
        return out[0] if key[0] == 'c' else (out * self.w).sum(0)

    def _h(self, key):
        def f(mod, inp, out):
            if key in self.fn: out = self.fn[key](out)
            if key in self.want: self.rec[key] = self.pool(key, out).detach()
            return out
        return f

    def run(self, x, v, m, grad=False):
        t = self.t; pt = t.hypot(v[:, 0], v[:, 1]) * m[:, 0]; w = pt / pt.sum(1, keepdim=True).clamp_min(1e-9)
        self.w = w.T[..., None]                                            # (P, n, 1)
        with (contextlib.nullcontext() if grad else t.no_grad()):
            L = self.m(x, v, m)
        return L


def T(net, a):
    return net.t.from_numpy(np.ascontiguousarray(a)).to(net.dev)


def feed(net, x, v, m):
    return T(net, x), T(net, v), T(net, m)


# ------------------------------------------------------------------------------------------------------------ (0) features
def extract(J0, models, nets, k_match, tr_das, ev_das):
    """chunked: edits, targets, inputs, pooled features at every site for every model and condition"""
    from interp.part_core import inputs, sub
    N = len(J0['label']); conds = ['base'] + EDITS
    F = {mn: {c: np.zeros((N, len(SITES), C), np.float16) for c in conds} for mn in models}
    Lg = {mn: {c: np.zeros((N, 10), np.float32) for c in conds} for mn in models}
    TG = {c: {t: np.zeros(N) for t in TARGETS} for c in conds}
    rng = np.random.default_rng(SEED + 7)
    for s0 in range(0, N, CHUNK):
        t1 = time.time(); Jc = sub(J0, np.arange(s0, min(s0 + CHUNK, N)))
        for c in conds:
            Je = make_edit(Jc, c, k_match, rng); tg = targets(Je)
            for t in TARGETS: TG[c][t][s0:s0 + len(Jc['label'])] = tg[t]
            cnt = (Je['mask'] > 0).sum(1); order = np.argsort(cnt, kind='stable')
            for i in range(0, len(order), BATCH):
                idx = order[i:i + BATCH]; Jb = sub(Je, idx); X = {}
                for mn in models:
                    fk = FEAT_OF(mn)
                    if fk not in X: X[fk] = inputs(Jb, fk)
                    net = nets[mn]; net.fn = {}; net.want = {SKEY[s] for s in SITES}
                    L = net.run(*feed(net, *X[fk]))
                    Lg[mn][c][s0 + idx] = L.float().cpu().numpy()
                    F[mn][c][s0 + idx] = np.stack([net.rec[SKEY[s]].float().cpu().numpy() for s in SITES], 1).astype(np.float16)
                    net.want = set(); net.rec = {}
            del Je
        log(f'  features: jets {s0 + len(Jc["label"])} / {N} ({time.time() - t1:.0f} s per chunk)')
    return F, Lg, TG


# ------------------------------------------------------------------------------------------------------------ (a, b) probes
class Ridge:            # as interp/p1v_verify_part.Ridge
    ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)

    def __init__(self, X, t, fit, val, tr):
        self.mu = X[tr].mean(0); self.sd = X[tr].std(0); self.sd[self.sd < 1e-9] = np.inf
        Z = (X - self.mu) / self.sd; self.tm = t[tr].mean(); self.ts = t[tr].std() + 1e-12; tt = (t - self.tm) / self.ts; best = None
        for a in self.ALPHAS:
            w = self._fit(Z[fit], tt[fit], a); e = ((Z[val] @ w - tt[val]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.alpha = best[1]; self.w = self._fit(Z[tr], tt[tr], self.alpha)

    @staticmethod
    def _fit(Z, t, a):
        Zc = Z - Z.mean(0); b = np.linalg.solve(Zc.T @ Zc + a * len(Z) * np.eye(Z.shape[1]), Zc.T @ (t - t.mean())); return b

    def __call__(self, X):
        Z = (X - self.mu) / self.sd; return (Z - 0) @ self.w * self.ts + self.tm + self._off

    def finish(self, X, t, tr):
        self._off = 0.0; self._off = float(t[tr].mean() - self(X[tr]).mean()); return self

    def direction(self):
        d = np.where(np.isfinite(self.sd), self.w / self.sd, 0.0); return d / (np.linalg.norm(d) + 1e-12)


def mlp_probe(t_mod, dev, X, y, fit, val, tr, preds, seed):
    """2-layer MLP (128-64-1, GELU), standardized in/out, Adam lr 3e-3, wd 1e-4, full batch, best val of 600 steps"""
    torch = t_mod; torch.manual_seed(seed)
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; ym, ys = y[tr].mean(), y[tr].std() + 1e-12
    Z = lambda A: torch.from_numpy(((A - mu) / sd).astype(np.float32)).to(dev)
    Xf, Xv = Z(X[fit]), Z(X[val]); yf = torch.from_numpy(((y[fit] - ym) / ys).astype(np.float32)).to(dev)
    yv = torch.from_numpy(((y[val] - ym) / ys).astype(np.float32)).to(dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4); best = (1e30, None, 0)
    for s in range(600):
        opt.zero_grad(); loss = ((net(Xf)[:, 0] - yf) ** 2).mean(); loss.backward(); opt.step()
        if s % 20 == 19:
            with torch.no_grad(): e = float(((net(Xv)[:, 0] - yv) ** 2).mean())
            if e < best[0]: best = (e, {k: v.clone() for k, v in net.state_dict().items()}, s + 1)
    net.load_state_dict(best[1])
    with torch.no_grad(): return [net(Z(P))[:, 0].cpu().numpy().astype(np.float64) * ys + ym for P in preds], best[2]


class Boot:
    """jet-bootstrap over the test half on the GPU: rows of W = resample counts"""

    def __init__(self, torch, dev, n, nb, seed):
        g = np.random.default_rng(seed); W = np.zeros((nb, n), np.float32)
        for b in range(nb): W[b] = np.bincount(g.integers(0, n, n), minlength=n)
        self.t, self.dev, self.W = torch, dev, torch.from_numpy(W).to(dev); self.S = self.W.sum(1)

    def dot(self, *vs):
        M = self.t.from_numpy(np.stack(vs, 1).astype(np.float32)).to(self.dev); return (self.W @ M).double().cpu().numpy().T

    def r2(self, p, t):
        p, t = np.asarray(p, np.float64), np.asarray(t, np.float64); tc = t - t.mean()
        a, b, c = self.dot((p - t) ** 2, tc, tc ** 2); S = self.S.double().cpu().numpy()
        return 1 - a / np.maximum(c - b ** 2 / S, 1e-30)

    def slope(self, dp, dt):
        a, b = self.dot(dp * dt, dt * dt); return a / np.maximum(b, 1e-30)

    def mean(self, x):
        return self.dot(x)[0] / self.S.double().cpu().numpy()


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-30))


def first_idx(curve, frac=0.9):
    """first index where curve >= frac x max (rows: bootstrap; last axis: sites)"""
    c = np.asarray(curve); mx = c.max(-1, keepdims=True); return np.argmax(c >= frac * mx - 1e-12, -1)


def probes(F, TG, models, tr, te, fit, val, boot, torch, dev):
    """R^2 curves, bootstrap, gains under each edit for ridge and MLP probes"""
    out = {}; dirs = {}; R2B = {}
    Rrot = np.linalg.qr(np.random.default_rng(SEED + 3).standard_normal((C, C)))[0]
    mlist = list(models) + (['rot'] if 'full' in models else [])
    for mn in mlist:
        out[mn] = {}; R2B[mn] = {}
        src = 'full' if mn == 'rot' else mn
        for si, s in enumerate(SITES):
            X = {c: F[src][c][:, si].astype(np.float64) for c in ['base'] + EDITS}
            if mn == 'rot': X = {c: v @ Rrot for c, v in X.items()}
            out[mn][s] = {}
            for t in TARGETS:
                y = TG['base'][t]; o = {}
                rd = Ridge(X['base'], y, fit, val, tr).finish(X['base'], y, tr)
                if mn in ('full', 'kin'): dirs[(mn, s, t)] = rd.direction()
                kinds = {'ridge': [rd(X[c][te]) for c in ['base'] + EDITS]}
                if True:
                    pm, st = mlp_probe(torch, dev, X['base'], y, fit, val, tr, [X[c][te] for c in ['base'] + EDITS], SEED + si * 17 + TARGETS.index(t))
                    kinds['mlp'] = pm
                for kind, P in kinds.items():
                    p0 = P[0]; bs = boot.r2(p0, y[te]); R2B[mn][(s, t, kind)] = bs
                    e = dict(r2=r2(p0, y[te]), r2_ci=ci(bs), edits={})
                    if kind == 'ridge': e['alpha'] = float(rd.alpha)
                    for ei, ed in enumerate(EDITS):
                        dp = P[ei + 1] - p0; dt = TG[ed][t][te] - y[te]
                        g = float((dp * dt).sum() / max((dt * dt).sum(), 1e-30)); gb = boot.slope(dp, dt)
                        e['edits'][ed] = dict(dp=float(dp.mean()), dp_ci=ci(boot.mean(dp)), dt=float(dt.mean()), gain=g, gain_ci=ci(gb),
                                              ratio_means=(float(dp.mean() / dt.mean()) if abs(dt.mean()) > 1e-6 else None))
                        if ed == CAL[t]: e['cal_gain'] = g; e['cal_gain_ci'] = ci(gb); e['_calb'] = gb
                    if t == 'log_mA':
                        dpp = P[1 + EDITS.index('phi')] - p0; dpk = P[1 + EDITS.index('dil')] - p0
                        a, b = boot.dot(dpp, dpk); e['phi_over_dil'] = float(dpp.mean() / dpk.mean()) if abs(dpk.mean()) > 1e-9 else None
                        e['phi_over_dil_ci'] = ci(a / b)
                    o[kind] = e
                out[mn][s][t] = o
        log(f'  probes {mn}: ' + ', '.join(f'{t} ' + '/'.join(f'{out[mn][s][t]["ridge"]["r2"]:.2f}' for s in ('emb', 'b2', 'b8', 'c2')) for t in TARGETS))
    return out, dirs, R2B


def depth_tests(P, R2B, models):
    """onset vs controls, t90 with bootstrap, rise, ordering tests"""
    res = {}
    ctrls = [c for c in ('perm', 'init') if c in P]
    for mn in [m for m in models if m in ('full', 'kin')]:
        res[mn] = {}
        for kind in ('ridge', 'mlp'):
            o = {}; T90 = {}
            for t in TARGETS:
                cur = np.array([P[mn][s][t][kind]['r2'] for s in SITES]); B = np.stack([R2B[mn][(s, t, kind)] for s in SITES], 1)
                lo = np.array([P[mn][s][t][kind]['r2_ci'][0] for s in SITES])
                chi = np.array([max(P[c][s][t][kind]['r2_ci'][1] for c in ctrls) for s in SITES]) if ctrls else np.full(len(SITES), -np.inf)
                onset = next((SITES[i] for i in range(len(SITES)) if lo[i] > chi[i]), None)
                t90b = first_idx(B); T90[t] = t90b; t90 = int(first_idx(cur))
                rise_b = B.max(1) - B[:, 0]; best = int(cur.argmax())
                dpk = B[:, best] - B[:, 0]
                gain = np.array([P[mn][s][t][kind].get('cal_gain', np.nan) for s in SITES])
                gB = np.stack([P[mn][s][t][kind]['_calb'] for s in SITES], 1)
                g_on = int(first_idx(gain[None])[0]) if np.isfinite(gain).all() else None
                o[t] = dict(r2=cur.tolist(), onset_vs_controls=onset, t90=SITES[t90], t90_boot_ci=[SITES[int(np.percentile(t90b, 2.5))], SITES[int(np.percentile(t90b, 97.5))]],
                            t90_boot_mean=float(t90b.mean()), best=SITES[best], rise_best_minus_emb=float(cur[best] - cur[0]), rise_ci=ci(dpk),
                            rise_p=p_two(dpk), cal_gain=gain.tolist(), cal_gain_t90=(SITES[g_on] if g_on is not None else None),
                            cal_gain_t90_boot_ci=[SITES[int(np.percentile(first_idx(gB), 2.5))], SITES[int(np.percentile(first_idx(gB), 97.5))]],
                            b8_minus_b1=float(cur[8] - cur[1]), b8_minus_b1_ci=ci(B[:, 8] - B[:, 1]), b8_minus_b1_p=p_two(B[:, 8] - B[:, 1]),
                            max_after_b1_minus_b1=float(cur[2:].max() - cur[1]), max_after_b1_minus_b1_ci=ci(B[:, 2:].max(1) - B[:, 1]))
            ps = holm([o[t]['rise_p'] for t in TARGETS])
            for t, p in zip(TARGETS, ps): o[t]['rise_p_holm'] = p
            gm = {g: np.mean([T90[t] for t in ts], 0) for g, ts in GROUPS.items()}
            order = {}
            for a, b in (('low', 'prong'), ('prong', 'mass'), ('low', 'mass')):
                d = gm[b] - gm[a]
                order[f't90_{b}_minus_{a}'] = dict(mean=float(d.mean()), ci=ci(d), p_one_sided_H1=float(max(1 / len(d), (d <= 0).mean())))
            order['all_t90_equal_boot_share'] = float(np.mean([len({int(T90[t][b]) for t in TARGETS}) == 1 for b in range(len(T90[TARGETS[0]]))]))
            res[mn][kind] = dict(targets=o, order=order)
    return res


# ------------------------------------------------------------------------------------------------------------ (c) DAS
def das_data(J0, idx, k_match, models_needed):
    """inputs for the DAS jets: base, and per edit the counterfactual (edited per-particle inputs; base 4-vectors and mask except
    soft5, which uses the full edited jet)"""
    from interp.part_core import inputs, sub
    rng = np.random.default_rng(SEED + 11); Jd = sub(J0, idx); D = {fk: {} for fk in models_needed}
    Jb = make_edit(Jd, 'base', k_match, rng)
    for fk in models_needed:
        xb, vb, mb = inputs(Jb, fk); D[fk]['base'] = (xb, vb, mb)
        for ed in EDITS:
            Je = make_edit(Jd, ed, k_match, np.random.default_rng(SEED + 13 + EDITS.index(ed)))
            xe, ve, me = inputs(Je, fk)
            if ed == 'soft5': D[fk][ed] = (xe, ve, me)
            else:
                assert np.array_equal(me, mb), ed
                D[fk][ed] = (xe, vb, mb)
    cnt = (Jb['mask'] > 0).sum(1)
    return D, cnt


def slc(trip, idx):
    x, v, m = trip; idx = np.asarray(idx); P = int(m[idx, 0].sum(1).max()); return x[idx][:, :, :P], v[idx][:, :, :P], m[idx][:, :, :P]


def src_pooled(net, Dk, ed, idx, sites):
    t = net.t; net.fn = {}; net.want = {SKEY[s] for s in sites}; L = net.run(*feed(net, *slc(Dk[ed], idx)))
    R = {s: net.rec[SKEY[s]] for s in sites}; net.want = set(); net.rec = {}
    return L, R


def das_fit(net, Dk, ed, site, tr_idx, rng):
    t = net.t; key = SKEY[site]
    g = t.Generator().manual_seed(int(rng.integers(1 << 30)))
    q0 = (t.randn(C, generator=g) / math.sqrt(C)).to(net.dev).requires_grad_(True); opt = t.optim.Adam([q0], lr=0.01); losses = []
    for s in range(STEPS):
        idx = np.sort(rng.choice(tr_idx, BS, replace=False))
        Lcf, R = src_pooled(net, Dk, ed, idx, [site])
        xb, vb, mb = feed(net, *slc(Dk['base'], idx))
        q = q0 / q0.norm()
        L = net_run_fn(net, xb, vb, mb, key, q, R[site]); loss = ((L - Lcf) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss)); net.fn = {}
    q = q0.detach() / q0.detach().norm()
    return q, float(np.mean(losses[:5])), float(np.mean(losses[-10:]))


def net_run_fn(net, x, v, m, key, q, src, grad=True):
    """run with the interchange at key; the pooling weights are those of the base run (set by Net.run before the forward)"""
    def fn(out):
        if key[0] == 'c': return out + (((src - out[0]) @ q)[:, None] * q[None, :])[None]
        return out + (((src - (out * net.w).sum(0)) @ q)[:, None] * q[None, :])[None]
    net.fn = {key: fn}; L = net.run(x, v, m, grad=grad); net.fn = {}; return L


def das_eval(net, Dk, eval_idx, cnt, Qs, sites):
    """exact runs: for each (ed, site) and each named direction -> per-jet squared error and calls; reference errors.
    Jacobians d logits / d (uniform shift at site) for the linearised random-direction null."""
    t = net.t; n = len(eval_idx); pos = np.full(cnt.shape[0], -1); pos[eval_idx] = np.arange(n)
    ref = {ed: dict(e0=np.zeros(n), cb=np.zeros(n, int), ccf=np.zeros(n, int), d=np.zeros((n, 10)), dh={s: np.zeros((n, C)) for s in sites}) for ed in EDITS}
    A = {}; JAC = {s: np.zeros((n, 10, C), np.float32) for s in sites}
    order = np.asarray(eval_idx)[np.argsort(cnt[eval_idx], kind='stable')]
    for i in range(0, n, 128):
        idx = order[i:i + 128]; rows = pos[idx]; xb, vb, mb = feed(net, *slc(Dk['base'], idx))
        net.fn = {}; net.want = {SKEY[s] for s in sites}; Lb = net.run(xb, vb, mb); Rb = {s: net.rec[SKEY[s]] for s in sites}; net.want = set()
        # Jacobian of logits wrt a uniform shift u_s at every site at once
        us = {s: t.zeros(len(idx), C, device=net.dev, requires_grad=True) for s in sites}
        net.fn = {SKEY[s]: (lambda u: (lambda out: out + u[None]))(us[s]) for s in sites}
        with t.enable_grad():
            Lg = net.run(xb, vb, mb, grad=True)
            for c in range(10):
                gr = t.autograd.grad(Lg[:, c].sum(), [us[s] for s in sites], retain_graph=c < 9)
                for s, gg in zip(sites, gr): JAC[s][rows, c] = gg.float().cpu().numpy()
        net.fn = {}
        for ed in EDITS:
            Lcf, R = src_pooled(net, Dk, ed, idx, sites)
            r = ref[ed]; r['e0'][rows] = ((Lb - Lcf) ** 2).sum(1).double().cpu().numpy(); r['cb'][rows] = Lb.argmax(1).cpu().numpy()
            r['ccf'][rows] = Lcf.argmax(1).cpu().numpy(); r['d'][rows] = (Lb - Lcf).double().cpu().numpy()
            for s in sites: r['dh'][s][rows] = (R[s] - Rb[s]).double().cpu().numpy()
            for (ed2, s, nm), q in Qs.items():
                if ed2 != ed: continue
                with t.no_grad(): L = net_run_fn(net, xb, vb, mb, SKEY[s], q, R[s], grad=False)
                a = A.setdefault((ed, s, nm), dict(e=np.zeros(n), c=np.zeros(n, int)))
                a['e'][rows] = ((L - Lcf) ** 2).sum(1).double().cpu().numpy(); a['c'][rows] = L.argmax(1).cpu().numpy()
    return ref, A, JAC


def lin_rec(ref, JAC, s, Q):
    """first-order interchange: L_int ~ L_b + (dh . q) J q  ->  recovery for each column of Q (C, K)"""
    d = ref['d']; a = ref['dh'][s] @ Q                                     # (n, K)
    U = np.einsum('jkc,cq->jkq', JAC[s].astype(np.float64), Q)              # (n, 10, K)
    e = ((d[:, :, None] + a[:, None, :] * U) ** 2).sum(1)                  # (n, K)
    return 1 - e.sum(0) / ref['e0'].sum()


def das_model(net, Dk, cnt, tr_idx, ev_idx, rng, dirs, mname, boot_ev, Qperm=None):
    t = net.t; Qs = {}; fits = {}
    for ed in EDITS:
        for s in SITES:
            t1 = time.time(); q, l0, l1 = das_fit(net, Dk, ed, s, tr_idx, rng); Qs[(ed, s, 'das')] = q; fits[f'{ed}|{s}'] = dict(loss_first5=l0, loss_last10=l1)
            if dirs is not None:
                Qs[(ed, s, 'probe')] = t.from_numpy(dirs[(mname, s, VARS[ed])].astype(np.float32)).to(net.dev)
                for r in range(NRAND_EXACT):
                    v_ = rng.standard_normal(C).astype(np.float32); Qs[(ed, s, f'rand{r}')] = t.from_numpy(v_ / np.linalg.norm(v_)).to(net.dev)
            log(f'  DAS {mname} {ed} {s}: loss {l0:.3f} -> {l1:.3f} ({time.time() - t1:.0f} s)')
    t1 = time.time(); ref, A, JAC = das_eval(net, Dk, ev_idx, cnt, Qs, SITES); log(f'  DAS {mname} evaluation ({time.time() - t1:.0f} s)')
    Rl = np.random.default_rng(SEED + 5).standard_normal((C, NRAND_LIN)); Rl /= np.linalg.norm(Rl, axis=0, keepdims=True)
    o = {}
    for ed in EDITS:
        r = ref[ed]; ch = r['ccf'] != r['cb']; o[ed] = dict(N_eval=len(ev_idx), n_call_changed=int(ch.sum()), sites={})
        for s in SITES:
            d = {}
            for nm in ['das', 'probe'] + [f'rand{i}' for i in range(NRAND_EXACT)]:
                if (ed, s, nm) not in A: continue
                a = A[(ed, s, nm)]
                ab = boot_ev.dot(a['e'], r['e0']); bs = 1 - ab[0] / ab[1]
                d[nm] = dict(iia=float(1 - a['e'].sum() / r['e0'].sum()), iia_ci=ci(bs),
                             call_iia=(float((a['c'][ch] == r['ccf'][ch]).mean()) if ch.any() else None), _bs=bs, _e=a['e'])
            q = Qs[(ed, s, 'das')].cpu().numpy().astype(np.float64)
            lin_d = float(lin_rec(r, JAC, s, q[:, None])[0]); lin_r = lin_rec(r, JAC, s, Rl)
            d['lin'] = dict(das=lin_d, rand_mean=float(lin_r.mean()), rand_q95=float(np.percentile(lin_r, 95)), rand_max=float(lin_r.max()),
                            p_rand=float((1 + (lin_r >= lin_d).sum()) / (1 + NRAND_LIN)))
            if f'rand0' in d:
                ex = [d[f'rand{i}']['iia'] for i in range(NRAND_EXACT)]
                lr = lin_rec(r, JAC, s, np.stack([Qs[(ed, s, f'rand{i}')].cpu().numpy().astype(np.float64) for i in range(NRAND_EXACT)], 1))
                d['lin']['check_rand_exact'] = ex; d['lin']['check_rand_lin'] = lr.tolist(); d['lin']['check_das_exact'] = d['das']['iia']
                d['rand_exact_mean'] = float(np.mean(ex)); d['rand_exact_max'] = float(np.max(ex))
            if 'probe' in d:
                pq = Qs[(ed, s, 'probe')].cpu().numpy(); d['cos_das_probe'] = float(abs(pq @ q))
            o[ed]['sites'][s] = d
    return o, {k: v for k, v in Qs.items() if k[2] == 'das'}, fits


def das_tests(O, Operm):
    """Holm over sites x variables: random-direction null (linearised) and perm-network DAS (paired bootstrap)"""
    keys, pr, pp = [], [], []
    for ed in EDITS:
        for s in SITES:
            d = O[ed]['sites'][s]; keys.append((ed, s)); pr.append(d['lin']['p_rand'])
            if Operm is not None:
                dd = d['das']['_bs'] - Operm[ed]['sites'][s]['das']['_bs']
                d['vs_perm'] = dict(diff=float(d['das']['iia'] - Operm[ed]['sites'][s]['das']['iia']), ci=ci(dd), p=p_two(dd))
                pp.append(d['vs_perm']['p'])
    for (ed, s), h in zip(keys, holm(pr)): O[ed]['sites'][s]['lin']['p_rand_holm'] = h
    if pp:
        for (ed, s), h in zip(keys, holm(pp)): O[ed]['sites'][s]['vs_perm']['p_holm'] = h
    steps = {}
    for ed in EDITS:
        B = np.stack([O[ed]['sites'][s]['das']['_bs'] for s in SITES], 1); cur = np.array([O[ed]['sites'][s]['das']['iia'] for s in SITES])
        dif = np.diff(cur); j = int(dif.argmax()); db = B[:, j + 1] - B[:, j]
        steps[ed] = dict(iia=cur.tolist(), step_at=SITES[j + 1], step=float(dif[j]), step_ci=ci(db), step_p=p_two(db),
                         t90=SITES[int(first_idx(cur))], t90_boot_ci=[SITES[int(np.percentile(first_idx(B), 2.5))], SITES[int(np.percentile(first_idx(B), 97.5))]],
                         b8_minus_emb=float(cur[8] - cur[0]), b8_minus_emb_ci=ci(B[:, 8] - B[:, 0]))
    return steps


# ------------------------------------------------------------------------------------------------------------ main
def main():
    import torch
    from interp.part_core import CL, PRE, sample, cat, sub
    OUT.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in ARGS.models.split(',') if m.strip()]
    R = dict(node='P18', script='interp/p18_stage_pipeline_part.py', smoke=SMOKE, seed=SEED, sites=SITES, targets=TARGETS, edits=EDITS,
             calibration_edit=CAL, das_variables=VARS, per_class=PER, NB=NB, STEPS=STEPS, BS=BS, N_DAS_train=N_DAS_TR, N_eval=N_EVAL,
             NRAND_EXACT=NRAND_EXACT, NRAND_LIN=NRAND_LIN)
    save = lambda: FN.write_text(json.dumps(strip(R), indent=1))
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Js = []
    for c in CL:
        t1 = time.time(); Js.append(load_rows(TEST / f'{PRE[c]}_100.root', PER, rng)); log(f'loaded {c}: {PER} random rows ({time.time() - t1:.0f} s)')
    J0 = cat(Js); del Js; N = len(J0['label']); R['N'] = N
    M = make_models(models); nets = {mn: Net(pn) for mn, pn in M.items()}; dev = M[models[0]].device; allm = list(M)
    # unedited accuracy on the P1 rows
    J1 = sample(per_class=1000, start=3000); J1b = make_edit(J1, 'base', 1.0, None); acc_p1, acc_raw = {}, {}
    for mn in allm:
        acc_p1[mn] = float((M[mn].run(J1b)['logits'].argmax(1) == J1['label']).mean())
        acc_raw[mn] = float((M[mn].run(J1)['logits'].argmax(1) == J1['label']).mean())
    R['accuracy_P1_rows'] = acc_p1; R['accuracy_P1_rows_raw_files'] = acc_raw
    log('accuracy on the P1 rows (built, as P8):', acc_p1, '; raw file jets:', acc_raw); del J1, J1b
    # matched dilation
    from interp.part_core import kin
    pt, de, dp, _ = kin(J0); pt = pt.astype(np.float64); r_ = (pt > 0)
    M2a = (pt * (de ** 2 + dp ** 2)).sum(1); M2b = (pt * (de ** 2 + (dp + 0.06 * r_) ** 2)).sum(1)
    k_match = float(np.exp(np.mean(0.5 * np.log(M2b / np.maximum(M2a, 1e-12))))); R['k_match'] = k_match; log(f'k_match = {k_match:.4f}')
    del pt, de, dp, r_
    perm = np.random.default_rng(SEED).permutation(N); tr, te = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
    fit, val = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    das_tr = np.sort(np.random.default_rng(SEED + 1).choice(tr, N_DAS_TR, replace=False))
    das_ev = np.sort(np.random.default_rng(SEED + 2).choice(te, N_EVAL, replace=False))
    # (0) features
    F, Lg, TG = extract(J0, allm, nets, k_match, das_tr, das_ev)
    y = J0['label'].astype(int); iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    R['accuracy_sample'] = {mn: float((Lg[mn]['base'].argmax(1) == y).mean()) for mn in allm}
    R['edit_calls'] = {mn: {ed: dict(changed=float((Lg[mn][ed].argmax(1) != Lg[mn]['base'].argmax(1)).mean()),
                                     W_to_Z_base=float((Lg[mn]['base'].argmax(1)[y == iW] == iZ).mean()),
                                     W_to_Z=float((Lg[mn][ed].argmax(1)[y == iW] == iZ).mean())) for ed in EDITS} for mn in allm}
    R['edit_true_changes'] = {ed: {t: dict(mean=float((TG[ed][t] - TG['base'][t]).mean()), sd=float((TG[ed][t] - TG['base'][t]).std()))
                                   for t in TARGETS} for ed in EDITS}
    R['target_stats'] = {t: dict(mean=float(TG['base'][t].mean()), sd=float(TG['base'][t].std())) for t in TARGETS}
    log('accuracy on the sample:', R['accuracy_sample']); log('edit calls:', json.dumps(strip(R['edit_calls'])))
    log('edit true changes (mean):', {ed: {t: round(v['mean'], 4) for t, v in d.items()} for ed, d in R['edit_true_changes'].items()})
    save()
    # (a, b) probes
    boot = Boot(torch, dev, len(te), NB, SEED + 21)
    P, dirs, R2B = probes(F, TG, allm, tr, te, fit, val, boot, torch, dev)
    R['probes'] = P; R['depth_tests'] = depth_tests(P, R2B, allm)
    for mn in R['depth_tests']:
        for t in TARGETS:
            d = R['depth_tests'][mn]['ridge']['targets'][t]
            log(f'  {mn} {t}: t90 {d["t90"]} {d["t90_boot_ci"]}, onset {d["onset_vs_controls"]}, rise {d["rise_best_minus_emb"]:.3f}, cal-gain t90 {d["cal_gain_t90"]}')
        log(f'  {mn} order (ridge): {json.dumps(R["depth_tests"][mn]["ridge"]["order"])}')
    save()
    del F
    # (c) DAS
    boot_ev = Boot(torch, dev, N_EVAL, NB, SEED + 22); R['das'] = {}; R['das_fits'] = {}
    dasidx = np.concatenate([das_tr, das_ev]); feats_needed = sorted({FEAT_OF(mn) for mn in allm})
    Dall, cnt = das_data(J0, dasidx, k_match, feats_needed)
    tr_i = np.arange(len(das_tr)); ev_i = np.arange(len(das_tr), len(dasidx))
    Operm = None
    if 'perm' in M:
        Operm, _, fits = das_model(nets['perm'], Dall['full'], cnt, tr_i, ev_i, np.random.default_rng(SEED + 31), None, 'perm', boot_ev)
        R['das']['perm'] = {'variables': Operm}; R['das_fits']['perm'] = fits; save()
    for mn in [m for m in models if m in ('full', 'kin')]:
        O, _, fits = das_model(nets[mn], Dall[FEAT_OF(mn)], cnt, tr_i, ev_i, np.random.default_rng(SEED + 32), dirs, mn, boot_ev)
        steps = das_tests(O, Operm); R['das'][mn] = dict(variables=O, steps=steps); R['das_fits'][mn] = fits
        for ed in EDITS:
            log(f'  DAS {mn} {ed}: IIA ' + ' '.join(f'{v:.2f}' for v in steps[ed]['iia']) + f'; step at {steps[ed]["step_at"]} {steps[ed]["step"]:.3f} {steps[ed]["step_ci"]}')
        save()
    if Operm is not None:
        R['das']['perm']['steps'] = das_tests(Operm, None)
    R['runtime_s'] = time.time() - T0; save(); log('wrote', FN)
    try: plot(json.loads(FN.read_text()))
    except Exception as e: log('plot failed:', repr(e))


# ------------------------------------------------------------------------------------------------------------ figure
def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = res or json.loads(FN.read_text()); S = res['sites']; x = np.arange(len(S)); lab = ['emb'] + [str(l) for l in range(1, 9)] + ['c1', 'c2']
    col = dict(N='#1b9e77', log_pT='#66a61e', log_r2='#a6761d', tau21='#7570b3', tau32='#e7298a', log_D2='#e6ab02', log_minv='#666666', log_mA='#d95f02')
    name = dict(N='multiplicity N', log_pT='log jet pT', log_r2=r'log $\langle\Delta R^2\rangle$', tau21=r'$\tau_{21}$', tau32=r'$\tau_{32}$',
                log_D2=r'log $D_2$', log_minv=r'log $m_{inv}$', log_mA=r'log $m_A$')
    ms = [m for m in ('full', 'kin') if m in res['probes']]
    fig, ax = plt.subplots(3, len(ms), figsize=(6.2 * len(ms), 12), squeeze=False)
    for j, mn in enumerate(ms):
        a = ax[0, j]
        for t in res['targets']:
            r = [res['probes'][mn][s][t]['ridge']['r2'] for s in S]; lo = [res['probes'][mn][s][t]['ridge']['r2_ci'][0] for s in S]
            hi = [res['probes'][mn][s][t]['ridge']['r2_ci'][1] for s in S]
            a.plot(x, r, '-o', ms=3, color=col[t], label=name[t]); a.fill_between(x, lo, hi, color=col[t], alpha=0.2)
            if 'perm' in res['probes']:
                c = [max(res['probes'][k][s][t]['ridge']['r2'] for k in ('perm', 'init') if k in res['probes']) for s in S]
                a.plot(x, c, ':', color=col[t], lw=1)
        a.set_title(f'ParT_{mn}: probe $R^2$ (dotted: control nets)'); a.set_ylim(-0.05, 1.02); a.set_ylabel('$R^2$ (test half)')
        a.set_xticks(x); a.set_xticklabels(lab); a.grid(alpha=0.3)
        if j == 0: a.legend(fontsize=8, ncol=2, loc='lower right')
        a = ax[1, j]
        for t in res['targets']:
            g = [res['probes'][mn][s][t]['ridge'].get('cal_gain') for s in S]; a.plot(x, g, '-o', ms=3, color=col[t], label=name[t])
            if 'perm' in res['probes']:
                a.plot(x, [res['probes']['perm'][s][t]['ridge'].get('cal_gain') for s in S], ':', color=col[t], lw=1)
        a.axhline(1, color='k', lw=0.6); a.set_title(f'ParT_{mn}: read-out / true change, matched edit')
        a.set_ylabel('calibration gain'); a.set_xticks(x); a.set_xticklabels(lab); a.grid(alpha=0.3)
        a = ax[2, j]
        if mn in res.get('das', {}):
            ecol = dict(phi='#d95f02', dil='#666666', scr='#7570b3', soft5='#1b9e77', pt='#66a61e')
            enm = dict(phi=r'$\phi$+0.06 ($m_A$)', dil=r'dilation ($m$)', scr=r'azimuth scramble ($\tau_{21}$)', soft5='5 soft particles (N)', pt=r'$p_T\times1.1$')
            for ed, d in res['das'][mn]['variables'].items():
                v = [d['sites'][s]['das']['iia'] for s in S]; lo = [d['sites'][s]['das']['iia_ci'][0] for s in S]; hi = [d['sites'][s]['das']['iia_ci'][1] for s in S]
                a.plot(x, v, '-o', ms=3, color=ecol[ed], label=enm[ed]); a.fill_between(x, lo, hi, color=ecol[ed], alpha=0.2)
                a.plot(x, [d['sites'][s]['lin']['rand_q95'] for s in S], '--', color=ecol[ed], lw=0.8)
                if 'perm' in res['das']:
                    a.plot(x, [res['das']['perm']['variables'][ed]['sites'][s]['das']['iia'] for s in S], ':', color=ecol[ed], lw=1)
            a.set_ylim(-0.1, 1.02); a.legend(fontsize=8, loc='upper left', title='dashed: random-q 95 %; dotted: permuted net', title_fontsize=7)
        a.set_title(f'ParT_{mn}: 1-d interchange (DAS) IIA'); a.set_ylabel('logit recovery (IIA)')
        a.set_xlabel('site (emb = embedding, 1-8 = particle blocks, c1-c2 = class blocks)'); a.set_xticks(x); a.set_xticklabels(lab); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG, dpi=110); print('wrote', FIG)


if __name__ == '__main__':
    plot() if ARGS.plot else main()
