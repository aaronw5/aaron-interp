"""P13 — learned-feature circuit in ParT_full (claim C8 at the feature level; Paper C review weaknesses 1, 4).

Units (dictionaries trained on ParT_full's own activations, activations streamed, not stored):
  P0..P7  skip-transcoders on the 8 particle-block MLPs: input u = pre_fc_norm output (128), target y = fc2 output (128);
          y_hat = TopK(u W_enc^T + b_enc) W_dec + b_dec + u W_skip^T, 1024 latents (8x), k = 32
  C0, C1  the same on the 2 class-block MLPs (class token only)
  R0, R1  TopK SAEs (2048 latents, 16x, k = 32) on the class-token residual after class block 1 / 2
  Training data: JetClass train_100M, NROUND rounds x 10 classes, each a random file of that class, NROW random rows
  (default 4 x 10 x 25,000 = 1,000,000 jets); particle sites trained online (2 passes per pool), class sites from a cache
  of the class-token activations (E_CLS epochs). AuxK loss (1/32) on latents dead for > DEAD steps.
Controls trained the same way: a weight-permuted ParT_full (every weight tensor's entries permuted, seed 5) at sites
  P0, P7, C0, C1, R0, R1; an SAE on a random orthogonal rotation Q of the R1 residual (rotR1).
Samples (never first-k; seed 13):
  ALL  : test_20M <sample>_105.root, NALL random rows per class -> accuracy check (expect ~86 %), FVU, dead fraction,
         spliced accuracy, atlas, 10-class read of circuits (U donors: uniform random other jet)
  DISC : test_20M WToQQ/ZToQQ_106.root, NDISC random rows each -> attribution (IG), m_A scan, steering calibration
  EVAL : test_20M WToQQ/ZToQQ_107.root, NEVAL random rows each -> circuits kept alone, steering / ablation vs phi + 0.06
Replacement model: at every site out = (a W_dec + b_dec + skip + err) * s_out, err = y/s_out - recon (exact with err).
Resample ablation (P9's corruption B): donor = an opposite-class (W<->Z) jet with matched jet pT (one of the 5 nearest);
  a feature that is not kept carries the donor's value (slot-wise by pT rank at particle sites); error terms likewise.
  Variants: main (skip terms and attention computed live; errors resampled), errlive (errors live), strict (skip terms and
  particle-block attention outputs also resampled). The R1 SAE is the bottleneck: nothing kept -> exactly the donor logits.
Attribution: D = sum_j s_j (l_W - l_Z); node score = mean over alpha in {.1,.3,.5,.7,.9} of dD/dm at m = alpha (IG over
  feature/error masks, all sites at once). Circuits = top-K features by |score|.
Atlas: per live feature, conditional-mean R^2 (20 quantile bins) against 20 fixed observables (particle-level list for P
  sites, jet-level list for C/R sites); monosemanticity = max over the 20. Controls: MLP hidden neurons, raw input coords,
  randomly rotated input coords, rotR1 SAE, weight-permuted-model dictionaries.
Steering: f* = class-level feature with the largest |corr| with log m_A (m_A = sqrt(sum pT * sum pT dR^2)), BH over the scan;
  beta = dA/dlog m_A from natural same-class pT-matched pairs; steer by beta*log(1.05) and by beta*dlog m_A(phi edit); ablate
  f* (mean) under phi + 0.06; 20 random same-site features as control.

  ./part_job.sh c-p13-smoke 'python -m interp.p13_transcoder_part --smoke'
  ./part_job.sh c-p13-run   'python -m interp.p13_transcoder_part'          (--reuse: load trained dictionaries)
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M'); TRAIN = Path('/j-jepa-vol/JetClass/Pythia/train_100M')
SMOKE = '--smoke' in sys.argv; REUSE = '--reuse' in sys.argv
E_ = lambda k, d: int(os.environ.get(k, d))
NROUND = E_('P13_NROUND', 1 if SMOKE else 4); NROW = E_('P13_NROW', 2000 if SMOKE else 25000)
NALL = E_('P13_NALL', 150 if SMOKE else 2000); NDISC = E_('P13_NDISC', 150 if SMOKE else 1500); NEVAL = E_('P13_NEVAL', 200 if SMOKE else 5000)
E_CLS = E_('P13_ECLS', 2 if SMOKE else 10); NRAND = E_('P13_NRAND', 3 if SMOKE else 100); NB = 200 if SMOKE else 1000
TB = 256; BATCH = 64; K_TOP = 32; DTC, DSAE = 1024, 2048
KGRID = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]; KRAND = [20, 50, 200]; K10 = [50, 200, 1000, 5000]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
SITES = [f'P{k}' for k in range(8)] + ['C0', 'R0', 'C1', 'R1']
PERM_SITES = ['P0', 'P7', 'C0', 'C1', 'R0', 'R1']
TAG = '_smoke' if SMOKE else ''
FN = OUT / f'p13_transcoder{TAG}.json'; DFN = OUT / f'p13_dicts{TAG}.pt'
T0 = time.time()
PARAMS_PER_FEATURE = 257          # encoder row (128) + encoder bias (1) + decoder column (128)
PARAMS_PER_P9_NODE = 8224         # P9: one head ~ 3*16*128 + 16*128 + biases; one 32-unit MLP group 32*(128+1+128)


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


from interp.p9_eap_wz_part import boot_W, ci, mcnemar, clean, load_sel, donors_B, donors_U, confusion  # noqa: E402


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]; run = min(run, p[i] * m / (r + 1)); q[i] = run
    return q


# =========================================================================================================== dictionaries
def make_dict_cls():
    import torch, torch.nn as nn

    class Dict(nn.Module):
        def __init__(self, d, D, k, skip):
            super().__init__(); self.k, self.D, self.d = k, D, d
            W = torch.randn(D, d); W = W / W.norm(dim=1, keepdim=True)
            self.W_dec = nn.Parameter(W.clone()); self.W_enc = nn.Parameter(0.1 * W.clone())
            self.b_enc = nn.Parameter(torch.zeros(D)); self.b_dec = nn.Parameter(torch.zeros(d))
            self.W_skip = nn.Parameter(torch.zeros(d, d)) if skip else None
            self.register_buffer('s_in', torch.ones(())); self.register_buffer('s_out', torch.ones(()))

        def pre(self, xn): return xn @ self.W_enc.T + self.b_enc

        def enc(self, xn):
            p = self.pre(xn); v, i = p.topk(self.k, -1); return torch.zeros_like(p).scatter(-1, i, v.relu())

        def skip(self, xn): return xn @ self.W_skip.T if self.W_skip is not None else torch.zeros_like(xn)

        def recon(self, x):
            xn = x / self.s_in; return (self.enc(xn) @ self.W_dec + self.b_dec + self.skip(xn)) * self.s_out
    return Dict


class Trainer:
    def __init__(self, Dm, lr, total, dead_steps):
        import torch
        self.t, self.Dm, self.lr, self.total, self.dead = torch, Dm, lr, max(total, 1), dead_steps
        self.opt = torch.optim.Adam(Dm.parameters(), lr=lr); self.i = 0
        self.since = torch.zeros(Dm.D, device=Dm.W_dec.device); self.hist = []

    def step(self, x, y):
        t, Dm = self.t, self.Dm
        if self.i == 0:
            with t.no_grad():
                Dm.s_in.fill_(float(x.pow(2).mean().sqrt())); Dm.s_out.fill_(float(y.pow(2).mean().sqrt()))
                Dm.b_dec.copy_((y / Dm.s_out).mean(0))
        xn, yn = x / Dm.s_in, y / Dm.s_out
        p = Dm.pre(xn); v, ix = p.topk(Dm.k, -1); a = t.zeros_like(p).scatter(-1, ix, v.relu())
        yh = a @ Dm.W_dec + Dm.b_dec + Dm.skip(xn); e = yn - yh
        var = (yn - yn.mean(0)).pow(2).sum(-1).mean().clamp_min(1e-8); fvu = e.pow(2).sum(-1).mean() / var; loss = fvu
        with t.no_grad():
            fired = (a > 0).any(0); self.since += 1; self.since[fired] = 0
        dead = self.since > self.dead
        if bool(dead.any()):
            ka = min(int(dead.sum()), 256); pd = p.masked_fill(~dead, -1e4); va, ia = pd.topk(ka, -1)
            eh = t.zeros_like(p).scatter(-1, ia, va.relu()) @ Dm.W_dec; ed = e.detach()
            loss = loss + ((ed - eh).pow(2).sum(-1).mean() / ed.pow(2).sum(-1).mean().clamp_min(1e-8)) / 32
        f = min(1.0, (self.i + 1) / 100) * min(1.0, (self.total - self.i) / (0.2 * self.total))
        for g in self.opt.param_groups: g['lr'] = self.lr * max(f, 0.02)
        self.opt.zero_grad(); loss.backward(); self.opt.step(); self.i += 1
        with t.no_grad(): Dm.W_dec.div_(Dm.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))
        if self.i % 50 == 0 or self.i == self.total: self.hist.append([self.i, float(fvu), int(dead.sum())])
        return float(fvu)


# =========================================================================================================== model runner
class Hook:
    def attn(self, s, a): return a
    def mlp(self, s, u, y, hid): return y
    def resid(self, s, r): return r


class Rec(Hook):
    def __init__(self, want): self.want, self.d = set(want), {}
    def mlp(self, s, u, y, hid):
        if s in self.want: self.d[s] = (u, y, hid)
        return y
    def resid(self, s, r):
        if s in self.want: self.d[s] = (r, r, None)
        return r


class Splice(Hook):
    def __init__(self, D, sites): self.D, self.sites = D, set(sites)
    def mlp(self, s, u, y, hid): return self.D[s].recon(u) if s in self.sites else y
    def resid(self, s, r): return self.D[s].recon(r) if s in self.sites else r


class MaskH(Hook):
    """mode 'donor': record a, err, skip, attention per site; mode 'clean': kept = m (a - a_d) + a_d"""
    def __init__(self, D):
        self.D, self.mode, self.st, self.spec = D, 'donor', {}, None

    def _site(self, s, x, y):
        Dm = self.D[s]; xn = x / Dm.s_in; a = Dm.enc(xn); sk = Dm.skip(xn)
        err = y / Dm.s_out - (a @ Dm.W_dec + Dm.b_dec + sk)
        if self.mode == 'donor':
            self.st[s] = (a.detach(), err.detach(), sk.detach()); return y
        ad, ed, skd = self.st[s]; mf, me = self.spec['m'][s]
        a2 = ad + mf * (a - ad); e2 = ed + me * (err - ed); sk2 = sk if self.spec.get('skip_live', True) else skd
        return (a2 @ Dm.W_dec + Dm.b_dec + sk2 + e2) * Dm.s_out

    def mlp(self, s, u, y, hid): return self._site(s, u, y)
    def resid(self, s, r): return self._site(s, r, r)

    def attn(self, s, a):
        if s[0] != 'P': return a
        if self.mode == 'donor': self.st['A' + s] = a.detach(); return a
        return self.st['A' + s] if self.spec.get('attn_res', False) else a


class Steer(Hook):
    """add delta (per jet, (N,)) to feature f at class-level site s, or set it to a constant (ablate)"""
    def __init__(self, D, s, f, delta=None, const=None): self.D, self.s, self.f, self.delta, self.const = D, s, f, delta, const

    def _do(self, s, x, y):
        if s != self.s: return y
        Dm = self.D[s]; a = Dm.enc(x / Dm.s_in)[..., self.f]
        new = (a + self.delta.view(1, -1)) if self.delta is not None else a * 0 + self.const
        return y + Dm.s_out * ((new - a)[..., None] * Dm.W_dec[self.f])

    def mlp(self, s, u, y, hid): return self._do(s, u, y)
    def resid(self, s, r): return self._do(s, r, r)


class Runner:
    def __init__(self, model):
        import torch
        self.t, self.m = torch, model

    def block(self, blk, X, cls, pad, am, site, H):
        t = self.t
        if cls is not None:
            pad2 = t.cat([t.zeros_like(pad[:, :1]), pad], 1); res = cls; u = blk.pre_attn_norm(t.cat([cls, X], 0))
            a = blk.attn(cls, u, u, key_padding_mask=pad2, need_weights=False)[0]
        else:
            res = X; u = blk.pre_attn_norm(X); a = blk.attn(u, u, u, key_padding_mask=pad, attn_mask=am, need_weights=False)[0]
        tl = a.size(0)
        a = t.einsum('tbhd,h->tbdh', a.reshape(tl, -1, blk.num_heads, blk.head_dim), blk.c_attn).reshape(tl, -1, blk.embed_dim)
        a = H.attn(site, blk.post_attn_norm(a)); x = a + res
        u = blk.pre_fc_norm(x); hid = blk.act(blk.fc1(u)); y = blk.fc2(blk.post_fc_norm(hid))
        y = H.mlp(site, u, y, hid)
        return y + blk.w_resid * x

    def __call__(self, x, v, mk, H=None):
        H = H or Hook(); m = self.m; mb = mk.bool(); pad = ~mb.squeeze(1); P = v.shape[-1]
        X = m.embed(x).masked_fill(~mb.permute(2, 0, 1), 0); am = m.pair_embed(v).view(-1, P, P)
        for k, blk in enumerate(m.blocks): X = self.block(blk, X, None, pad, am, f'P{k}', H)
        cls = m.cls_token.expand(1, X.size(1), -1)
        for b, blk in enumerate(m.cls_blocks):
            cls = self.block(blk, X, cls, pad, None, f'C{b}', H); cls = H.resid(f'R{b}', cls)
        z = m.norm(cls).squeeze(0); return m.fc(z)


# =========================================================================================================== data
def prepset(parts):
    from interp.part_core import kin, build, cat
    J0 = cat(parts) if isinstance(parts, list) else parts; J = build(J0, *kin(J0)); return J, J0['label'].astype(int)


def load_rows(path, n, rng):
    import uproot
    N = uproot.open(path)['tree'].num_entries; r = np.sort(rng.choice(N, min(n, N), replace=False))
    (J,), _ = load_sel(path, [r]); return J, dict(file=path.name, entries=N, n=len(r), first_rows=r[:5].tolist())


def to_dev(t, dev, arrs): return [t.from_numpy(np.ascontiguousarray(a)).to(dev) for a in arrs]


def jet_batches(mult, size_jets, P_ref=64):
    """indices sorted by multiplicity, batches of at most size_jets * P_ref particle slots"""
    o = np.argsort(mult, kind='stable'); out, cur, mx = [], [], 1
    for i in o:
        mi = max(int(mult[i]), 1)
        if cur and (len(cur) + 1) * max(mx, mi) > size_jets * P_ref: out.append(np.array(cur)); cur, mx = [], 1
        cur.append(i); mx = max(mx, mi)
    if cur: out.append(np.array(cur))
    return out


# =========================================================================================================== observables
def observables(J):
    from interp.part_core import kin, moments, jmass
    pt, de, dp, ms = kin(J); real = J['mask'] > 0; dR = np.hypot(de, dp); N = len(pt)
    M = moments(pt, de, dp); mjet = jmass(J); mA = np.sqrt(np.maximum(M['M0M2'], 1e-9)); mult = real.sum(1)
    lpt = np.log(np.maximum(pt, 1e-3)); E = J['energy']
    jo = {}
    jo['log_m'] = np.log(np.maximum(mjet, 1.0)); jo['log_mA'] = np.log(np.maximum(mA, 1e-3)); jo['log_jet_pt'] = np.log(J['jet_pt'])
    jo['mult'] = mult.astype(float); jo['M2_over_M0'] = M['M2'] / np.maximum(M['M0'], 1e-9); jo['M1_over_M0'] = M['M1'] / np.maximum(M['M0'], 1e-9)
    jo['pTD'] = np.sqrt((pt ** 2).sum(1)) / np.maximum(pt.sum(1), 1e-9); jo['lead_frac'] = pt[:, 0] / np.maximum(pt.sum(1), 1e-9)
    jo['dR12'] = np.hypot(de[:, 0] - de[:, 1], dp[:, 0] - dp[:, 1])
    e01 = J['energy'][:, 0] + J['energy'][:, 1]; p01 = [J[k][:, 0] + J[k][:, 1] for k in ('px', 'py', 'pz')]
    jo['log_m12'] = np.log(np.maximum(np.sqrt(np.maximum(e01 ** 2 - sum(q ** 2 for q in p01), 0)), 0.1))
    e2 = np.zeros(N); z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-9)
    for s in range(0, N, 500):
        d1 = de[s:s + 500, :, None] - de[s:s + 500, None, :]; d2 = dp[s:s + 500, :, None] - dp[s:s + 500, None, :]
        e2[s:s + 500] = 0.5 * (z[s:s + 500, :, None] * z[s:s + 500, None, :] * np.hypot(d1, d2)).sum((1, 2))
    jo['e2'] = e2
    cs = np.cumsum(pt, 1) / np.maximum(pt.sum(1, keepdims=True), 1e-9); jo['N95'] = (cs < 0.95).sum(1) + 1.0
    jo['n_charged'] = (J['isChargedHadron'] * real).sum(1); jo['n_neutral_had'] = (J['isNeutralHadron'] * real).sum(1)
    jo['n_photon'] = (J['isPhoton'] * real).sum(1); jo['n_lepton'] = ((J['isElectron'] + J['isMuon']) * real).sum(1)
    jo['max_abs_tanh_d0'] = (np.abs(np.tanh(J['d0val'])) * real).max(1); jo['jet_charge'] = (J['charge'] * pt).sum(1) / np.maximum(pt.sum(1), 1e-9)
    jo['abs_jet_eta'] = np.abs(J['jet_eta']); jo['lead_dR'] = dR[:, 0]
    po = {}
    po['log_pt'] = lpt; po['log_ptrel'] = lpt - np.log(J['jet_pt'])[:, None]; po['log_E'] = np.log(np.maximum(E, 1e-3)); po['dR'] = dR
    po['deta'] = de; po['dphi'] = dp; po['rank'] = np.broadcast_to(np.arange(pt.shape[1])[None], pt.shape).astype(float)
    for k in ('isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon', 'charge'): po[k] = J[k]
    po['tanh_d0'] = np.tanh(J['d0val']); po['tanh_dz'] = np.tanh(J['dzval'])
    for k in ('log_m', 'log_mA', 'log_jet_pt', 'mult'): po['jet_' + k] = np.broadcast_to(jo[k][:, None], pt.shape)
    po['dR_to_lead'] = np.hypot(de - de[:, :1], dp - dp[:, :1])
    assert len(jo) == 20 and len(po) == 20, (len(jo), len(po))
    return dict(jet=jo, part=po, real=real, mA=mA, log_mA=jo['log_mA'], mjet=mjet)


def binning(obs, real=None, nb=20):
    """obs: dict name -> array; returns ids (stacked, -1 where invalid), nbins list, offsets, edges"""
    ids, nbs, edges = [], [], {}
    for k, v in obs.items():
        v = np.asarray(v, float); vv = v[real] if real is not None else v; u = np.unique(vv)
        if len(u) <= nb: b = np.clip(np.searchsorted(u, v), 0, len(u) - 1); n = len(u); edges[k] = dict(categories=u.tolist())
        else:
            e = np.unique(np.quantile(vv, np.linspace(0, 1, nb + 1)[1:-1])); b = np.searchsorted(e, v, side='right'); n = len(e) + 1
            edges[k] = dict(edges=e.tolist())
        if real is not None: b = np.where(real, b, -1)
        ids.append(b.astype(np.int32)); nbs.append(n)
    off = np.concatenate([[0], np.cumsum(nbs)[:-1]]).astype(np.int64)
    return np.stack(ids, -1), nbs, off, edges


class Acc:
    """conditional-mean R^2 accumulators for several unit sets"""
    def __init__(self, t, dev, NBtot):
        self.t, self.dev, self.NB, self.u = t, dev, NBtot, {}

    def add(self, name, OH, LOH, A):
        t = self.t; A = A.float()
        if name not in self.u:
            D = A.shape[1]; z = lambda *s: t.zeros(*s, dtype=t.float64, device=self.dev)
            self.u[name] = dict(S=z(self.NB, D), n=z(self.NB), s1=z(D), s2=z(D), N=0, Sc=z(10, D), nc=z(10), fire=z(D))
        U = self.u[name]; U['S'] += (OH.T @ A).double(); U['n'] += OH.sum(0).double(); U['s1'] += A.sum(0).double()
        U['s2'] += A.pow(2).sum(0).double(); U['N'] += A.shape[0]; U['Sc'] += (LOH.T @ A).double(); U['nc'] += LOH.sum(0).double()
        U['fire'] += (A > 0).sum(0).double()

    def r2(self, name, nbs, off):
        U = self.u[name]; N = U['N']; s1, s2 = U['s1'].cpu().numpy(), U['s2'].cpu().numpy(); S = U['S'].cpu().numpy(); n = U['n'].cpu().numpy()
        tot = s2 - s1 ** 2 / N; R = np.zeros((len(nbs), len(s1)))
        for o, (nb, of) in enumerate(zip(nbs, off)):
            nn_ = n[of:of + nb]; ok = nn_ > 0; R[o] = ((S[of:of + nb][ok] ** 2) / nn_[ok][:, None]).sum(0) - s1 ** 2 / N
        with np.errstate(invalid='ignore', divide='ignore'): R = R / tot[None]
        live = (U['fire'].cpu().numpy() > 0) & (tot > 1e-12 * max(N, 1))
        return np.clip(R, 0, 1), live, tot / N

    def profile(self, name, of, nb, f):
        U = self.u[name]; S = U['S'][of:of + nb, f].cpu().numpy(); n = U['n'][of:of + nb].cpu().numpy()
        return (S / np.maximum(n, 1)).tolist()

    def cls_profile(self, name, f):
        U = self.u[name]; return (U['Sc'][:, f] / U['nc'].clamp_min(1)).cpu().numpy().tolist()


# =========================================================================================================== main
def main():
    import torch
    from interp.part_core import ParTNetwork, inputs, sub, cat, kin, build, moments, jmass, CL, PRE
    t = torch; OUT.mkdir(parents=True, exist_ok=True)
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    res = dict(smoke=SMOKE, NROUND=NROUND, NROW=NROW, NALL=NALL, NDISC=NDISC, NEVAL=NEVAL, E_CLS=E_CLS, NRAND=NRAND, NB=NB, k=K_TOP,
               D_tc=DTC, D_sae=DSAE, KGRID=KGRID, ALPHAS=ALPHAS, sites=SITES,
               naming='P{k}#i: latent i of the skip-transcoder on particle block k+1 MLP; C{b}#i: class block b+1 MLP transcoder; '
                      'R{b}#i: SAE latent on the class-token residual after class block b+1; ERR_s: error node of site s')

    def dump():
        FN.write_text(json.dumps(clean(res), indent=1))

    net = ParTNetwork('full'); model = net.model; dev = net.device
    for p in model.parameters(): p.requires_grad_(False)
    run = Runner(model)
    # weight-permuted control model
    netp = ParTNetwork('full'); g = t.Generator().manual_seed(5)
    with t.no_grad():
        for p in netp.model.parameters():
            if p.numel() > 1: p.copy_(p.flatten()[t.randperm(p.numel(), generator=g).to(p.device)].view_as(p))
            p.requires_grad_(False)
    runp = Runner(netp.model)
    Dict = make_dict_cls(); rngQ = np.random.default_rng(17)
    Q = t.from_numpy(np.linalg.qr(rngQ.standard_normal((128, 128)))[0].astype(np.float32)).to(dev)

    def new_dict(s):
        tc = s[0] in 'PC'; return Dict(128, DTC if tc else DSAE, K_TOP, tc).to(dev)

    # ------------------------------------------------------------------------------------------------ 1. dictionaries
    if REUSE and DFN.exists():
        blob = t.load(DFN, map_location=dev); D = {}
        for k, sd in blob['dicts'].items():
            D[k] = new_dict(k.replace('perm_', '').replace('rot', '')); D[k].load_state_dict(sd)
        res['training'] = blob['training']; log(f'loaded {len(D)} dictionaries from {DFN}')
    else:
        from concurrent.futures import ThreadPoolExecutor
        rngT = np.random.default_rng(13); avail = {c: sorted(p.name for p in TRAIN.glob(f'{PRE[c]}_*.root')) for c in CL}
        picks = {c: rngT.choice(len(avail[c]), NROUND, replace=False) for c in CL}

        def load_pool(r):
            parts, ys, info = [], [], []
            for c in CL:
                J, inf = load_rows(TRAIN / avail[c][picks[c][r]], NROW, np.random.default_rng(1000 * r + CL.index(c)))
                Jb, y = prepset(J); del J; parts.append(Jb); ys.append(y); info.append(inf)
            return cat(parts), np.concatenate(ys), info
        D = {s: new_dict(s) for s in SITES}; D.update({f'perm_{s}': new_dict(s) for s in PERM_SITES})
        n_tot = NROUND * 10 * NROW; steps_p = 2 * n_tot // TB
        TR = {k: Trainer(D[k], 1e-3, steps_p, 50) for k in D if k.split('_')[-1][0] == 'P'}
        cache = {k: [] for k in ('C0', 'C1', 'R0', 'R1')}; cachep = {k: [] for k in ('C0', 'C1', 'R0', 'R1')}
        ex = ThreadPoolExecutor(1); fut = ex.submit(load_pool, 0); files_used = []
        for r in range(NROUND):
            tl = time.time(); Jp, yp, info = fut.result(); files_used += info
            if r + 1 < NROUND: fut = ex.submit(load_pool, r + 1)
            log(f'pool {r}: {len(yp)} jets loaded (waited {time.time() - tl:.0f} s), classes {np.bincount(yp, minlength=10).tolist()}')
            for ep in range(2):
                perm = np.random.default_rng(100 + 10 * r + ep).permutation(len(yp))
                for bi in range(0, len(perm), TB):
                    idx = np.sort(perm[bi:bi + TB]); x, v, mk = to_dev(t, dev, inputs(sub(Jp, idx), 'full'))
                    real = mk[:, 0].bool().T                                           # (P, N)
                    for rr, R_, pre in ((run, D, ''), (runp, D, 'perm_')):
                        want = SITES if pre == '' else PERM_SITES; H = Rec(want)
                        with t.no_grad(): rr(x, v, mk, H)
                        for s in want:
                            u, yv, _ = H.d[s]
                            if s[0] == 'P':
                                TR[pre + s].step(u[real].float(), yv[real].float())
                            elif ep == 0:
                                (cache if pre == '' else cachep)[s].append((t.stack([u[0], yv[0]], 1) if s[0] == 'C' else u[0][:, None]).half())
                        del H
                log(f'  pool {r} epoch {ep}: P-site FVU ' + ' '.join(f'{k} {TR[k].hist[-1][1]:.3f}' for k in TR if TR[k].hist))
            del Jp, yp
        ex.shutdown()
        # class sites from the cache
        for pre, C_ in (('', cache), ('perm_', cachep)):
            for s in ('C0', 'C1', 'R0', 'R1'):
                A = t.cat(C_[s]); C_[s] = None; n = len(A); steps = E_CLS * (n // 1024)
                tr = Trainer(D[pre + s], 1e-3, steps, 500); TR[pre + s] = tr
                for ep in range(E_CLS):
                    pm = t.randperm(n)
                    for i in range(0, n - 1023, 1024):
                        b = A[pm[i:i + 1024].to(A.device)].float(); tr.step(b[:, 0], b[:, -1])
                if pre == '' and s == 'R1':
                    D['rotR1'] = new_dict('R1'); tr2 = Trainer(D['rotR1'], 1e-3, steps, 500); TR['rotR1'] = tr2
                    for ep in range(E_CLS):
                        pm = t.randperm(n)
                        for i in range(0, n - 1023, 1024):
                            b = A[pm[i:i + 1024].to(A.device)].float()[:, 0] @ Q.T; tr2.step(b, b)
                log(f'{pre}{s}: {n} class tokens, {steps} steps, FVU {tr.hist[-1][1] if tr.hist else float("nan"):.3f}')
                del A
        res['training'] = dict(n_jets=n_tot, files=files_used, steps={k: v.i for k, v in TR.items()},
                               final_train_fvu={k: (v.hist[-1][1] if v.hist else None) for k, v in TR.items()},
                               final_dead_steps={k: (v.hist[-1][2] if v.hist else None) for k, v in TR.items()},
                               curve={k: v.hist[::max(1, len(v.hist) // 40)] for k, v in TR.items()})
        t.save(dict(dicts={k: v.state_dict() for k, v in D.items()}, training=res['training']), DFN)
        del cache, cachep, TR
    for Dm in D.values():
        for p in Dm.parameters(): p.requires_grad_(False)
    dump()

    # ------------------------------------------------------------------------------------------------ 2. samples
    rng = np.random.default_rng(13); files = {}
    pA = []
    for c in CL:
        J, inf = load_rows(TEST / f'{PRE[c]}_105.root', NALL, rng); pA.append(J); files[f'all_{c}'] = inf
    JA, yA = prepset(pA); del pA
    pD, pE = [], []
    for c in ('Wqq', 'Zqq'):
        J, inf = load_rows(TEST / f'{PRE[c]}_106.root', NDISC, rng); pD.append(J); files[f'disc_{c}'] = inf
        J, inf = load_rows(TEST / f'{PRE[c]}_107.root', NEVAL, rng); pE.append(J); files[f'eval_{c}'] = inf
    JD, yD = prepset(pD); JE, yE = prepset(pE); del pD, pE
    res['samples'] = files; log(f'samples: all {len(yA)}, disc {len(yD)}, eval {len(yE)}')

    def logits(rr, J, H=None, idx=None, size=256):
        idx = np.arange(len(J['mask'])) if idx is None else np.asarray(idx); L = np.zeros((len(idx), 10), np.float32)
        for bb in jet_batches(J['mask'][idx].sum(1), size // 4):
            x, v, mk = to_dev(t, dev, inputs(sub(J, idx[bb]), 'full'))
            with t.no_grad(): L[bb] = rr(x, v, mk, H).float().cpu().numpy()
        return L

    # exactness + accuracy on the all-class sample
    LA = logits(run, JA); accA = float((LA.argmax(1) == yA).mean())
    x, v, mk = to_dev(t, dev, inputs(sub(JA, np.arange(64)), 'full'))
    with t.no_grad(): dmax = float((model(x, v, mk) - run(x, v, mk)).abs().max())
    WbA = boot_W(len(yA), NB, np.random.default_rng(1)); okA = (LA.argmax(1) == yA)
    res['unedited'] = dict(acc10_all=accA, acc10_all_ci=ci((WbA @ okA) / len(yA)), runner_vs_weaver_max_abs_logit=dmax)
    log(f'ParT_full 10-class accuracy on ALL (N {len(yA)}, seed 13): {accA:.4f} (expected ~0.86); runner vs weaver {dmax:.2e}')
    dump()

    # ------------------------------------------------------------------------------------------------ 3. reconstruction / splice
    ss = {s: dict(se=0.0, s1=t.zeros(128, device=dev, dtype=t.float64), s2=0.0, n=0) for s in list(D)}
    for bb in jet_batches(JA['mask'].sum(1), 64):
        x, v, mk = to_dev(t, dev, inputs(sub(JA, bb), 'full')); real = mk[:, 0].bool().T
        for rr, pre, want in ((run, '', SITES), (runp, 'perm_', PERM_SITES)):
            H = Rec(want)
            with t.no_grad():
                rr(x, v, mk, H)
                for s in want:
                    u, yv, _ = H.d[s]
                    if s[0] == 'P': u, yv = u[real], yv[real]
                    else: u, yv = u[0], yv[0]
                    for key, uu in ((pre + s, u),) + ((('rotR1', u @ Q.T),) if (pre == '' and s == 'R1') else ()):
                        yy = yv if key != 'rotR1' else uu
                        e = yy - D[key].recon(uu); S_ = ss[key]
                        S_['se'] += float(e.double().pow(2).sum()); S_['s1'] += yy.double().sum(0); S_['s2'] += float(yy.double().pow(2).sum()); S_['n'] += len(yy)
    fvu = {k: float(S_['se'] / (S_['s2'] - float((S_['s1'] ** 2).sum()) / S_['n'])) for k, S_ in ss.items() if S_['n']}
    res['fvu_eval'] = fvu; log('FVU on ALL: ' + ' '.join(f'{k} {v:.3f}' for k, v in fvu.items()))
    spl = {}
    for s in SITES:
        L_ = logits(run, JA, Splice(D, [s])); spl[s] = float((L_.argmax(1) == yA).mean())
    L_ = logits(run, JA, Splice(D, SITES)); okS = L_.argmax(1) == yA; spl['all'] = float(okS.mean())
    L_ = logits(run, JA, Splice(D, [s for s in SITES if s[0] == 'P'])); spl['all_particle'] = float((L_.argmax(1) == yA).mean())
    L_ = logits(run, JA, Splice(D, [s for s in SITES if s[0] != 'P'])); spl['all_class'] = float((L_.argmax(1) == yA).mean())
    res['splice_acc10'] = spl; res['splice_all_drop'] = dict(drop=accA - spl['all'], ci=ci((WbA @ okA - WbA @ okS) / len(yA)),
                                                             mcnemar=mcnemar(okA, okS))
    log('spliced 10-class accuracy: ' + ' '.join(f'{k} {v:.4f}' for k, v in spl.items()))
    dump()

    # ------------------------------------------------------------------------------------------------ 4. atlas
    tA = time.time(); OA = observables(JA)
    idsP, nbP, offP, edgP = binning(OA['part'], OA['real']); idsJ, nbJ, offJ, edgJ = binning(OA['jet'])
    namesP, namesJ = list(OA['part']), list(OA['jet'])
    accP = Acc(t, dev, int(sum(nbP))); accJ = Acc(t, dev, int(sum(nbJ))); jetmax = {s: np.zeros((len(yA), D[s].D), np.float16) for s in SITES}
    offPt, offJt = t.from_numpy(offP).to(dev), t.from_numpy(offJ).to(dev)

    def onehot(ids, offt, NBt):
        ids = t.from_numpy(ids.astype(np.int64)).to(dev); OH = t.zeros(ids.shape[0], NBt, device=dev)
        return OH.scatter_(1, ids + offt[None], 1.0)

    for bb in jet_batches(JA['mask'].sum(1), 64):
        x, v, mk = to_dev(t, dev, inputs(sub(JA, bb), 'full')); P = x.shape[-1]; real = mk[:, 0].bool().T; rNP = mk[:, 0].bool(); realN = rNP.cpu().numpy()
        tok = lambda X: X.transpose(0, 1)[rNP]                                   # (P, N, .) -> tokens in (jet, slot) order
        OHp = onehot(idsP[bb, :P][realN], offPt, accP.NB); OHj = onehot(idsJ[bb], offJt, accJ.NB)
        lab = t.from_numpy(yA[bb]).to(dev); LOHj = t.nn.functional.one_hot(lab, 10).float()
        LOHp = LOHj[:, None].expand(-1, P, -1)[rNP]
        for rr, pre, want in ((run, '', SITES), (runp, 'perm_', PERM_SITES)):
            H = Rec(want)
            with t.no_grad():
                rr(x, v, mk, H)
                for s in want:
                    u, yv, hid = H.d[s]; Dm = D[pre + s]
                    if s[0] == 'P':
                        a = Dm.enc(u / Dm.s_in)
                        accP.add(pre + s, OHp, LOHp, tok(a))
                        if pre == '':
                            accP.add('neur_' + s, OHp, LOHp, tok(hid)); accP.add('raw_' + s, OHp, LOHp, tok(u)); accP.add('rot_' + s, OHp, LOHp, tok(u) @ Q.T)
                            jetmax[s][bb] = (a * real[..., None]).max(0)[0].cpu().numpy().astype(np.float16)
                    else:
                        a = Dm.enc(u[0] / Dm.s_in); accJ.add(pre + s, OHj, LOHj, a)
                        if pre == '':
                            jetmax[s][bb] = a.cpu().numpy().astype(np.float16)
                            if hid is not None: accJ.add('neur_' + s, OHj, LOHj, hid[0])
                            accJ.add('raw_' + s, OHj, LOHj, u[0]); accJ.add('rot_' + s, OHj, LOHj, u[0] @ Q.T)
                            if s == 'R1': accJ.add('rotR1', OHj, LOHj, D['rotR1'].enc((u[0] @ Q.T) / D['rotR1'].s_in))
    atlas = dict(observables_particle=namesP, observables_jet=namesJ, sites={}, compare={})
    MONO = {}
    for nm in list(accP.u) + list(accJ.u):
        A_, nbs, off, names = (accP, nbP, offP, namesP) if nm in accP.u else (accJ, nbJ, offJ, namesJ)
        R, live, var = A_.r2(nm, nbs, off); mono = np.where(live, np.nanmax(np.nan_to_num(R), 0), np.nan); MONO[nm] = (R, live, mono, names)
        atlas['sites'][nm] = dict(n_units=int(len(live)), live=int(live.sum()), dead_frac=float(1 - live.mean()),
                                  mono_mean=float(np.nanmean(mono)) if live.any() else None, mono_median=float(np.nanmedian(mono)) if live.any() else None,
                                  mono_frac_gt_0p5=float(np.nanmean(mono[live] > 0.5)) if live.any() else None,
                                  best_obs_counts={names[o]: int(c) for o, c in zip(*np.unique(np.nanargmax(np.nan_to_num(R[:, live]), 0), return_counts=True))} if live.any() else {},
                                  fire_freq_median=float(np.median((A_.u[nm]['fire'].cpu().numpy() / A_.u[nm]['N'])[live])) if live.any() else None)
    from scipy.stats import mannwhitneyu
    rngb = np.random.default_rng(3)

    def cmp(a, b):
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if not len(a) or not len(b): return None
        bs = [rngb.choice(a, len(a)).mean() - rngb.choice(b, len(b)).mean() for _ in range(NB)]
        return dict(mean_a=float(a.mean()), mean_b=float(b.mean()), diff=float(a.mean() - b.mean()), diff_ci=ci(bs),
                    p_mannwhitney_greater=float(mannwhitneyu(a, b, alternative='greater').pvalue))
    for s in SITES:
        c = {}
        for ctl in ('neur_', 'raw_', 'rot_', 'perm_'):
            if ctl + s in MONO: c[ctl.rstrip('_')] = cmp(MONO[s][2], MONO[ctl + s][2])
        if s == 'R1': c['rotR1_trained'] = cmp(MONO['R1'][2], MONO['rotR1'][2])
        atlas['compare'][s] = c
    res['atlas'] = atlas; res['atlas_seconds'] = time.time() - tA
    log('atlas mono mean (dict / neur / raw / rot / perm): ' + ' | '.join(
        f'{s} ' + '/'.join(f'{atlas["sites"][k]["mono_mean"]:.2f}' if k in atlas['sites'] and atlas['sites'][k]['mono_mean'] is not None else '-'
                           for k in (s, 'neur_' + s, 'raw_' + s, 'rot_' + s, 'perm_' + s)) for s in SITES))
    dump()

    # ------------------------------------------------------------------------------------------------ 5. attribution graph
    tI = time.time(); rngd = np.random.default_rng(13)
    donD = donors_B(JD['jet_pt'], yD, iW, iZ, rngd); donE = donors_B(JE['jet_pt'], yE, iW, iZ, rngd); donA = donors_U(len(yA), rngd)
    res['donors'] = dict(B_eval_rel_dpt_median=float(np.median(np.abs(JE['jet_pt'][donE] - JE['jet_pt']) / JE['jet_pt'])),
                         U_all_same_class=float((yA[donA] == yA).mean()))

    def pair_batches(J, idx, don, size=BATCH):
        idx = np.asarray(idx); m = np.maximum(J['mask'][idx].sum(1), J['mask'][don[idx]].sum(1))
        for bb in jet_batches(m, size):
            bi = idx[bb]; x, v, mk = inputs(cat([sub(J, bi), sub(J, don[bi])]), 'full'); n = len(bi)
            xs = to_dev(t, dev, (x, v, mk)); yield bb, [a[:n] for a in xs], [a[n:] for a in xs]

    def mvec(val, grad=False):
        return {s: (t.full((D[s].D,), float(val), device=dev, requires_grad=grad), t.tensor(float(val), device=dev, requires_grad=grad)) for s in SITES}

    scores = {s: np.zeros(D[s].D) for s in SITES}; serr = {s: 0.0 for s in SITES}; Dtot = dict(clean=0.0, corr=0.0)
    sD = np.where(yD == iW, 1.0, -1.0); H = MaskH(D)
    for bb, cl, dn in pair_batches(JD, np.arange(len(yD)), donD, 48):
        s_ = t.from_numpy(sD[bb]).float().to(dev)
        H.mode = 'donor'
        with t.no_grad(): Ld = run(*dn, H); Lc = run(*cl)
        Dtot['clean'] += float((s_ * (Lc[:, iW] - Lc[:, iZ])).sum()); Dtot['corr'] += float((s_ * (Ld[:, iW] - Ld[:, iZ])).sum())
        H.mode = 'clean'
        for al in ALPHAS:
            mm = mvec(al, True); H.spec = dict(m=mm); L = run(*cl, H)
            Dm_ = (s_ * (L[:, iW] - L[:, iZ])).sum(); gr = t.autograd.grad(Dm_, [q for s in SITES for q in mm[s]])
            for i, s in enumerate(SITES):
                scores[s] += gr[2 * i].cpu().numpy() / len(ALPHAS); serr[s] += float(gr[2 * i + 1]) / len(ALPHAS)
    nD = len(yD); live_any = {s: MONO[s][1] for s in SITES}
    allf = [(s, i, scores[s][i]) for s in SITES for i in range(D[s].D)]
    tot_abs = sum(abs(v) for _, _, v in allf) + sum(abs(v) for v in serr.values())
    rank = sorted([(s, i, v) for s, i, v in allf if live_any[s][i]], key=lambda r: -abs(r[2]))
    sum_ig = sum(v for _, _, v in allf) + sum(serr.values())
    res['attribution'] = dict(n_jets=nD, D_clean_minus_corr_per_jet=(Dtot['clean'] - Dtot['corr']) / nD, sum_ig_per_jet=sum_ig / nD,
                              completeness_ratio=sum_ig / (Dtot['clean'] - Dtot['corr']), error_abs_share=sum(abs(v) for v in serr.values()) / tot_abs,
                              error_signed_share=sum(serr.values()) / (Dtot['clean'] - Dtot['corr']), error_scores_per_jet={s: v / nD for s, v in serr.items()},
                              site_abs_share={s: float(np.abs(scores[s]).sum() / tot_abs) for s in SITES},
                              top_abs_share={str(k): float(sum(abs(r[2]) for r in rank[:k]) / tot_abs) for k in (10, 50, 200, 1000)},
                              top50=[dict(feature=f'{s}#{i}', score_per_jet=v / nD) for s, i, v in rank[:50]], seconds=time.time() - tI)
    log(f'IG: D(1)-D(0) {res["attribution"]["D_clean_minus_corr_per_jet"]:.3f}/jet, completeness {res["attribution"]["completeness_ratio"]:.3f}, '
        f'error |share| {res["attribution"]["error_abs_share"]:.3f}, top-50 |share| {res["attribution"]["top_abs_share"]["50"]:.3f}; top: '
        + ', '.join(f'{r["feature"]} {r["score_per_jet"]:+.3f}' for r in res['attribution']['top50'][:8]))
    np.savez_compressed(OUT / f'p13_scores{TAG}.npz', **{s: scores[s] for s in SITES})
    dump()

    # ------------------------------------------------------------------------------------------------ 6. circuits kept alone
    def spec_of(feats, err=0.0, **kw):
        m = {s: (t.zeros(D[s].D, device=dev), t.tensor(float(err), device=dev)) for s in SITES}
        for s, i in feats: m[s][0][i] = 1.0
        return dict(m=m, **kw)

    def run_specs(J, idx, don, specs, check=False):
        idx = np.asarray(idx); out = {k: np.zeros((len(idx), 10), np.float32) for k in list(specs) + ['clean', 'corr']}; H = MaskH(D); chk = [0.0, 0.0]
        with t.no_grad():
            for bb, cl, dn in pair_batches(J, idx, don):
                H.mode = 'donor'; Ld = run(*dn, H); Lc = run(*cl); H.mode = 'clean'
                out['clean'][bb] = Lc.cpu().numpy(); out['corr'][bb] = Ld.cpu().numpy()
                if check:
                    H.spec = dict(m=mvec(1.0)); chk[0] = max(chk[0], float((run(*cl, H) - Lc).abs().max()))
                    H.spec = dict(m=mvec(0.0)); chk[1] = max(chk[1], float((run(*cl, H) - Ld).abs().max()))
                for k, sp in specs.items(): H.spec = sp; out[k][bb] = run(*cl, H).cpu().numpy()
        if check: out['_check'] = dict(all_kept_vs_clean=chk[0], none_kept_vs_donor=chk[1])
        return out

    sE = np.where(yE == iW, 1.0, -1.0); WbE = boot_W(len(yE), NB, np.random.default_rng(2))

    def summ(L, Lc, Lco, y, s, Wb):
        d = L[:, iW] - L[:, iZ]; dc = Lc[:, iW] - Lc[:, iZ]; dco = Lco[:, iW] - Lco[:, iZ]
        num = s * (d - dco); den = s * (dc - dco); ok = np.where(d > 0, iW, iZ) == y; ok0 = np.where(dc > 0, iW, iZ) == y
        return dict(F=float(num.sum() / den.sum()), F_ci=ci((Wb @ num) / (Wb @ den)), wz_acc=float(ok.mean()), wz_acc_ci=ci((Wb @ ok) / len(y)),
                    mcnemar_vs_full=mcnemar(ok0, ok), _ok=ok, _accb=(Wb @ ok) / len(y))

    tC = time.time(); specs = {}
    for K in KGRID:
        f = [(s, i) for s, i, _ in rank[:K]]
        specs[f'main_K{K}'] = spec_of(f); specs[f'errlive_K{K}'] = spec_of(f, 1.0); specs[f'strict_K{K}'] = spec_of(f, 0.0, skip_live=False, attn_res=True)
    allfe = [(s, i) for s in SITES for i in range(D[s].D)]
    specs['strict_all_features_and_errors'] = spec_of(allfe, 1.0, skip_live=False, attn_res=True)
    specs['main_K0'] = spec_of([]); specs['errlive_K0'] = spec_of([], 1.0)
    specs['main_Kall_live'] = spec_of([(s, i) for s, i, _ in rank])          # every live feature kept, errors resampled
    OE = run_specs(JE, np.arange(len(yE)), donE, specs, check=True)
    circ = dict(unedited_wz_acc=float((np.where(OE['clean'][:, iW] > OE['clean'][:, iZ], iW, iZ) == yE).mean()),
                donor_wz_acc=float((np.where(OE['corr'][:, iW] > OE['corr'][:, iZ], iW, iZ) == yE).mean()), exactness=OE['_check'], K={})
    SUM = {}
    for k in specs:
        SUM[k] = summ(OE[k], OE['clean'], OE['corr'], yE, sE, WbE)
        if k.startswith('main_K') and k[6:].isdigit() and k != 'main_K0':
            K = int(k[6:]); SUM[k]['params'] = K * PARAMS_PER_FEATURE; SUM[k]['site_counts'] = {s: sum(1 for r in rank[:K] if r[0] == s) for s in SITES}
        circ['K'][k] = {a: b for a, b in SUM[k].items() if not a.startswith('_')}
    b50 = SUM['main_K50']['_accb']
    circ['hypothesis'] = dict(H1='a <= 50-feature circuit (errors resampled) reaches >= 75 % W/Z', H0='feature circuits cap near the P9v 63 % ceiling',
                              wz_acc_K50=SUM['main_K50']['wz_acc'], p_boot_acc_K50_le_0p75=float(np.mean(b50 <= 0.75)),
                              p_boot_acc_K50_le_0p63=float(np.mean(b50 <= 0.63)),
                              smallest_K_ge_0p75={v: next((K for K in KGRID if SUM[f'{v}_K{K}']['wz_acc'] >= 0.75), None) for v in ('main', 'errlive', 'strict')})
    log(f'circuits (eval N {len(yE)}): unedited W/Z {circ["unedited_wz_acc"]:.4f}, donor {circ["donor_wz_acc"]:.4f}, exact {OE["_check"]}')
    log(f'  all {len(rank)} live features, errors resampled: F {SUM["main_Kall_live"]["F"]:.3f} W/Z {SUM["main_Kall_live"]["wz_acc"]:.3f}')
    for K in [0] + KGRID:
        log(f'  K {K:5d}: ' + ' | '.join(f'{v} F {SUM[f"{v}_K{K}"]["F"]:.3f} W/Z {SUM[f"{v}_K{K}"]["wz_acc"]:.3f}'
                                         for v in ('main', 'errlive', 'strict') if f'{v}_K{K}' in SUM))
    res['circuits'] = circ; dump()
    # random feature sets of equal size (matched per-site counts, among live features)
    sub_idx = np.arange(0, len(yE), 10); rngr = np.random.default_rng(21); rnd = {}
    livelist = {s: np.where(live_any[s])[0] for s in SITES}
    for K in KRAND:
        cnt = SUM[f'main_K{K}']['site_counts']; sp = {}
        for j in range(NRAND):
            f = [(s, int(i)) for s in SITES for i in rngr.choice(livelist[s], min(cnt[s], len(livelist[s])), replace=False)]; sp[f'r{j}'] = spec_of(f)
        sp['circ'] = specs[f'main_K{K}']
        O = run_specs(JE, sub_idx, donE, sp); ys, ss_ = yE[sub_idx], sE[sub_idx]; Wb = boot_W(len(ys), NB, np.random.default_rng(4))
        cs = summ(O['circ'], O['clean'], O['corr'], ys, ss_, Wb); rs = [summ(O[f'r{j}'], O['clean'], O['corr'], ys, ss_, Wb) for j in range(NRAND)]
        ra, rF = np.array([r['wz_acc'] for r in rs]), np.array([r['F'] for r in rs])
        rnd[str(K)] = dict(n_jets=len(ys), circuit_wz_acc=cs['wz_acc'], circuit_F=cs['F'], random_wz_acc_mean=float(ra.mean()), random_wz_acc_ci=ci(ra),
                           random_wz_acc_max=float(ra.max()), random_F_mean=float(rF.mean()), random_F_max=float(rF.max()),
                           p_perm_acc=float((1 + (ra >= cs['wz_acc']).sum()) / (NRAND + 1)), p_perm_F=float((1 + (rF >= cs['F']).sum()) / (NRAND + 1)))
        log(f'  random K {K}: circuit W/Z {cs["wz_acc"]:.3f} F {cs["F"]:.3f} | random W/Z mean {ra.mean():.3f} max {ra.max():.3f} | p {rnd[str(K)]["p_perm_acc"]:.4f}')
    res['circuits']['random_sets'] = rnd; dump()
    # 10-class read (U donors)
    sp = {f'main_K{K}': specs[f'main_K{K}'] for K in K10}; sp['main_K0'] = specs['main_K0']
    O = run_specs(JA, np.arange(len(yA)), donA, sp); ten = dict(n_jets=len(yA), unedited_acc10=float((O['clean'].argmax(1) == yA).mean()), circuits={})
    wzA = np.isin(yA, [iW, iZ])
    for k in sp:
        L_ = O[k]; ten['circuits'][k] = dict(acc10=float((L_.argmax(1) == yA).mean()), acc10_ci=ci((WbA @ (L_.argmax(1) == yA)) / len(yA)),
                                             wz_binary=float((np.where(L_[wzA, iW] > L_[wzA, iZ], iW, iZ) == yA[wzA]).mean()),
                                             follows_donor=float((L_.argmax(1) == yA[donA]).mean()), confusion=confusion(L_, yA))
        log(f'  10-class {k}: acc10 {ten["circuits"][k]["acc10"]:.3f}, W/Z binary {ten["circuits"][k]["wz_binary"]:.3f}')
    res['circuits']['ten_class'] = ten; res['circuits']['seconds'] = time.time() - tC; dump()

    # atlas entries of the top-50 circuit features
    cards = []; x, v, mk = to_dev(t, dev, inputs(sub(JA, np.arange(0, len(yA), max(1, len(yA) // 256))), 'full')); H = Rec(['R1'])
    with t.no_grad(): run(x, v, mk, H)
    lnsd_R1 = float(H.d['R1'][0][0].std(-1, unbiased=False).mean()); res['atlas']['lnsd_R1'] = lnsd_R1
    for s, i, v in rank[:50]:
        R, live, mono, names = MONO[s]; A_, nbs, off = (accP, nbP, offP) if s[0] == 'P' else (accJ, nbJ, offJ)
        o3 = np.argsort(-R[:, i])[:3]; top = np.argsort(-jetmax[s][:, i].astype(np.float32))[:100]
        card = dict(feature=f'{s}#{i}', score_per_jet=v / nD, mono=float(mono[i]), top_obs=[(names[o], float(R[o, i])) for o in o3],
                    fire_freq=float(A_.u[s]['fire'][i] / A_.u[s]['N']), class_profile=A_.cls_profile(s, i),
                    top100_class_counts=np.bincount(yA[top], minlength=10).tolist(), top10_jets=top[:10].tolist())
        for ob in (['dR', 'log_pt', 'jet_log_mA', 'isChargedHadron', 'isPhoton'] if s[0] == 'P' else ['log_mA', 'log_m', 'log_jet_pt', 'mult', 'M2_over_M0']):
            o = names.index(ob); card[f'profile_{ob}'] = A_.profile(s, off[o], nbs[o], i)
        cp = A_.cls_profile(s, i); card['negative_pole_class'] = CL[int(np.argmin(cp))]; card['positive_pole_class'] = CL[int(np.argmax(cp))]
        if s == 'R1':        # linearised direct effect on l_W - l_Z of +1 unit of the feature (final LayerNorm scale = mean over ALL jets)
            lnw = model.norm.weight.float(); Wd = (model.fc[0].weight[iW] - model.fc[0].weight[iZ]).float(); dv = D[s].W_dec[i] - D[s].W_dec[i].mean()
            card['direct_dlogit_WZ_per_unit'] = float((dv * D[s].s_out * lnw / lnsd_R1) @ Wd)
        ups = [(s2, i2) for s2, i2, _ in rank[:50] if SITES.index(s2) < SITES.index(s) and (s2[0] == s[0] or s[0] != 'P')]
        if ups:
            cs_ = [(f'{s2}#{i2}', float(t.nn.functional.cosine_similarity(D[s].W_enc[i], D[s2].W_dec[i2], 0))) for s2, i2 in ups]
            card['upstream_cos_top3'] = sorted(cs_, key=lambda r: -abs(r[1]))[:3]
        cards.append(card)
    res['atlas']['circuit_cards'] = cards
    res['atlas']['bin_edges'] = dict(particle={k: edgP[k] for k in ('dR', 'log_pt', 'jet_log_mA')}, jet={k: edgJ[k] for k in ('log_mA', 'log_m', 'log_jet_pt', 'mult', 'M2_over_M0')})
    dump()

    # ------------------------------------------------------------------------------------------------ 7. m_A feature: steering and ablation
    tS = time.time(); OD = observables(JD); lmD = OD['log_mA']
    acts = {s: [] for s in ('C0', 'R0', 'C1', 'R1')}; ordD = []
    for bb in jet_batches(JD['mask'].sum(1), 64):
        x, v, mk = to_dev(t, dev, inputs(sub(JD, bb), 'full')); H = Rec(list(acts))
        with t.no_grad():
            run(x, v, mk, H)
            for s in acts: acts[s].append(D[s].enc(H.d[s][0][0] / D[s].s_in).cpu().numpy())
        ordD.append(bb)
    ordD = np.concatenate(ordD); inv = np.argsort(ordD); acts = {s: np.concatenate(a)[inv] for s, a in acts.items()}
    from scipy.stats import pearsonr
    scan = []
    for s, A in acts.items():
        for i in np.where(A.std(0) > 0)[0]:
            r, p = pearsonr(A[:, i], lmD); scan.append((s, int(i), float(r), float(p)))
    q = bh([r[3] for r in scan]); top = int(np.argmax([abs(r[2]) for r in scan])); fs, fi, fr, fp = scan[top]
    # calibration on natural same-class pT-matched pairs
    rngp = np.random.default_rng(31); part = np.zeros(len(yD), int)
    for c in (iW, iZ):
        ix = np.where(yD == c)[0]; o = ix[np.argsort(JD['jet_pt'][ix])]
        for k, i in enumerate(o):
            cand = np.concatenate([o[max(0, k - 3):k], o[k + 1:k + 4]]); part[i] = rngp.choice(cand)
    da = acts[fs][part, fi] - acts[fs][:, fi]; dl = lmD[part] - lmD; beta = float((da * dl).sum() / (dl * dl).sum())
    bs = []
    for _ in range(NB):
        b = rngp.integers(0, len(da), len(da)); bs.append((da[b] * dl[b]).sum() / (dl[b] ** 2).sum())
    sdf = float(acts[fs][:, fi].std()); meanf = float(acts[fs][:, fi].mean())
    steer = dict(scan_n=len(scan), scan_bh_sig=int((q < 0.05).sum()), feature=f'{fs}#{fi}', r_log_mA=fr, p=fp, q_bh=float(q[top]),
                 top5=[dict(feature=f'{a}#{b}', r=c) for a, b, c, _ in sorted(scan, key=lambda r: -abs(r[2]))[:5]],
                 beta_per_dlog_mA=beta, beta_ci=ci(bs), feature_sd=sdf, feature_mean=meanf, n_pairs=len(da))
    log(f'm_A scan: {len(scan)} class-level features, {steer["scan_bh_sig"]} BH-significant; top {steer["feature"]} r {fr:.3f}; beta {beta:.3f} {steer["beta_ci"]}')
    # W jets of EVAL: base vs phi + 0.06
    wi = np.where(yE == iW)[0]; JW = sub(JE, wi); pt, de, dp, ms = kin(JW); real = pt > 0
    JWp = build(JW, pt, de, dp + 0.06 * real, ms); lmb = 0.5 * np.log(moments(*kin(JW)[:3])['M0M2']); lmp = 0.5 * np.log(moments(*kin(JWp)[:3])['M0M2'])
    dlm_phi = lmp - lmb; nW = len(wi); WbW = boot_W(nW, NB, np.random.default_rng(5))
    Lb = logits(run, JW); Lp = logits(run, JWp); cb = Lb.argmax(1); cpp = Lp.argmax(1)
    dvec = lambda a: t.from_numpy(np.asarray(a, np.float32)).to(dev)

    def flow(L): return (L.argmax(1) == iZ).astype(float)
    base_wz = flow(Lb); phi_wz = flow(Lp); fl_phi = phi_wz - base_wz

    def steer_run(J, s, f, delta=None, const=None, idx_order=True):
        out = np.zeros((len(J['mask']), 10), np.float32)
        for bb in jet_batches(J['mask'].sum(1), 64):
            x, v, mk = to_dev(t, dev, inputs(sub(J, bb), 'full'))
            H = Steer(D, s, f, None if delta is None else dvec(delta[bb]), const)
            with t.no_grad(): out[bb] = run(x, v, mk, H).cpu().numpy()
        return out
    L5 = steer_run(JW, fs, fi, np.full(nW, beta * math.log(1.05))); Lm = steer_run(JW, fs, fi, beta * dlm_phi)
    La_b = steer_run(JW, fs, fi, const=meanf); La_p = steer_run(JWp, fs, fi, const=meanf)
    fl5, flm = flow(L5) - base_wz, flow(Lm) - base_wz; fl_abl = flow(La_p) - flow(La_b)

    def rb(a, b): return ci((WbW @ a) / np.maximum(WbW @ b, 1e-9))
    steer.update(n_W=nW, mean_dlog_mA_phi=float(dlm_phi.mean()), W_to_Z_base=float(base_wz.mean()), W_to_Z_phi=float(phi_wz.mean()),
                 W_to_Z_steer5=float(flow(L5).mean()), W_to_Z_steer_matched=float(flow(Lm).mean()),
                 flow_phi=float(fl_phi.mean()), flow_phi_ci=ci(WbW @ fl_phi / nW), flow_steer5=float(fl5.mean()), flow_steer5_ci=ci(WbW @ fl5 / nW),
                 flow_steer_matched=float(flm.mean()), ratio_steer5_over_phi=float(fl5.mean() / fl_phi.mean()), ratio_steer5_ci=rb(fl5, fl_phi),
                 ratio_matched_over_phi=float(flm.mean() / fl_phi.mean()), ratio_matched_ci=rb(flm, fl_phi),
                 mcnemar_steer5_vs_base=mcnemar(cb == iW, L5.argmax(1) == iW),
                 ablate=dict(W_to_Z_base=float(flow(La_b).mean()), W_to_Z_phi=float(flow(La_p).mean()), flow_phi_ablated=float(fl_abl.mean()),
                             flow_phi_ablated_ci=ci(WbW @ fl_abl / nW), change_vs_intact=float(fl_abl.mean() - fl_phi.mean()),
                             change_ci=ci(WbW @ (fl_abl - fl_phi) / nW)))
    # random same-site feature control (steered by the same amount in s.d. units; ablated to their mean)
    rngc = np.random.default_rng(41); A = acts[fs]; cand = [i for i in np.where(A.std(0) > 0)[0] if i != fi]
    rf = rngc.choice(cand, min(20 if not SMOKE else 3, len(cand)), replace=False); rs5, rabl = [], []
    for i in rf:
        sg = np.sign(np.corrcoef(A[:, i], lmD)[0, 1]) or 1.0; amt = sg * abs(beta * math.log(1.05)) * A[:, i].std() / max(sdf, 1e-9)
        rs5.append(float((flow(steer_run(JW, fs, int(i), np.full(nW, amt))) - base_wz).mean()))
        mi = float(A[:, i].mean()); rabl.append(float((flow(steer_run(JWp, fs, int(i), const=mi)) - flow(steer_run(JW, fs, int(i), const=mi))).mean()))
    rs5, rabl = np.array(rs5), np.array(rabl)
    steer['random_control'] = dict(features=[f'{fs}#{int(i)}' for i in rf], flow_steer5=rs5.tolist(), flow_phi_ablated=rabl.tolist(),
                                   p_perm_steer5=float((1 + (rs5 >= fl5.mean()).sum()) / (len(rs5) + 1)),
                                   p_perm_ablate=float((1 + (rabl <= fl_abl.mean()).sum()) / (len(rabl) + 1)))
    steer['seconds'] = time.time() - tS; res['steering'] = steer
    log(f'steer {steer["feature"]}: W->Z base {steer["W_to_Z_base"]:.3f}, phi {steer["W_to_Z_phi"]:.3f} (dlog mA {steer["mean_dlog_mA_phi"]:.4f}), '
        f'steer5 {steer["W_to_Z_steer5"]:.3f}, matched {steer["W_to_Z_steer_matched"]:.3f}; ratio5 {steer["ratio_steer5_over_phi"]:.3f} {steer["ratio_steer5_ci"]}; '
        f'ablated phi flow {steer["ablate"]["flow_phi_ablated"]:.3f} vs {steer["flow_phi"]:.3f}; random steer5 mean {rs5.mean():.4f}')
    res['seconds'] = time.time() - T0; dump()
    plot(res)


def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = res or json.loads(FN.read_text()); C = res['circuits']['K']
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    col = dict(main='#2a78d6', strict='#eb6834', errlive='#1baf7a', p9='#52514e')
    for v, lab in (('main', 'feature circuit (errors resampled)'), ('strict', 'feature circuit, skip + particle attention also resampled')):
        Ks = [K for K in KGRID if f'{v}_K{K}' in C]
        ax.plot([K * PARAMS_PER_FEATURE for K in Ks], [100 * C[f'{v}_K{K}']['wz_acc'] for K in Ks], 'o-', color=col[v], lw=2, ms=6, label=lab)
    p9n = [8, 16, 32, 64, 128, 256, 433]; p9a = [27.2, 31.4, 36.6, 50.8, 65.3, 82.2, 83.25]
    ax.plot([n * PARAMS_PER_P9_NODE for n in p9n], p9a, 's--', color=col['p9'], lw=1.5, ms=6, label='P9 literal head/MLP-group circuits')
    for K, r in res['circuits'].get('random_sets', {}).items():
        ax.errorbar(int(K) * PARAMS_PER_FEATURE, 100 * r['random_wz_acc_mean'], yerr=[[100 * (r['random_wz_acc_mean'] - r['random_wz_acc_ci'][0])], [100 * (r['random_wz_acc_ci'][1] - r['random_wz_acc_mean'])]],
                    fmt='v', color='#888888', ms=6, label='random feature sets (same size)' if K == str(KRAND[0]) else None)
    ax.axhline(100 * res['circuits']['unedited_wz_acc'], color='k', lw=1, ls=':'); ax.text(4e6, 100 * res['circuits']['unedited_wz_acc'] + 1.5, 'full model', fontsize=8, ha='right')
    ax.axhline(100 * res['circuits']['donor_wz_acc'], color='k', lw=1, ls=':'); ax.text(4e6, 100 * res['circuits']['donor_wz_acc'] - 4.5, 'everything resampled (donor jet)', fontsize=8, ha='right')
    ax.set_xscale('log'); ax.set_xlabel('parameters in the kept units (feature: 257; P9 node: ~8,200)'); ax.set_ylabel('W vs Z accuracy, circuit kept alone [%]')
    ax.set_title(f'ParT_full: W/Z accuracy of circuits kept alone vs size (N = {len(res["samples"]) and 2 * res["NEVAL"]:,} W/Z jets)', fontsize=10); ax.grid(alpha=0.25)
    ax.set_xlim(8e2, 5e6); ax.set_ylim(0, 108); ax.legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=True, framealpha=0.95); fig.tight_layout(); fig.savefig(OUT / f'p13_feature_circuit{TAG}.png'); plt.close(fig)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if '--plot' in sys.argv: plot()
    else: main()
