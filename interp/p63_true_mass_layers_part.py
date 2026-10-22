"""P63 (claims C3 / C7 / C4; loop C round C14, user question): do later ParT_full layers compute the true invariant mass m,
or only the axis-moment mass m_A = sqrt(M0 * M2)?  Matched to loop A's E41 (JEDI-linear), so the two can be compared.

PLAN (fixed before the first run)
Definitions (per jet; sums over real particles; (deta_i, dphi_i) as stored, i.e. about the stored JetClass jet axis;
r_i = hypot(deta_i, dphi_i)):
  m    = invariant mass of the summed particle four-vectors [GeV] (part_core.jmass);
  M0   = sum pT_i;  M1 = sum pT_i (deta_i, dphi_i) (2-vector);  M2 = sum pT_i r_i^2;
  m_A  = sqrt(M0 M2) (axis-moment mass). Small-angle, massless: m^2 ~ M0 M2 - |M1|^2, so at fixed m_A the true mass differs
         from m_A only through the pT-weighted centroid offset M1 of the particles from the stored axis;
  Delta = log m - log m_A.
Models: ParT_full (released JetClass weights, weaver-core 0.4.17), ParT_kin (secondary; read-outs only); controls: untrained
  ParT_full, 2 inits (ParTNetwork('full', untrained=True, seed=0/1): every tensor redrawn from a Gaussian with the trained
  tensor's mean and s.d.), and a weight-permuted ParT_full (P33 recipe: output rows of every >= 2-d weight tensor permuted by one
  random permutation per tensor, matching biases permuted; torch seed 6363). Own forward pass (embedding, pair embedding, 8
  particle blocks, 2 class blocks, norm, fc) checked against the weaver forward on the first batch of every model.
Sample: JetClass test_20M WToQQ_112, ZToQQ_112, ZJetsToNuNu_112, TTBar_112 (the planned _125 does not exist: test_20M holds
  _100.._119 and every index has been used before; _112 is the least used one in this loop). 8,000 random rows per file
  (numpy default_rng(6363), never first-k), split at random into 4,000 fit and 4,000 evaluation jets per class (32,000 jets).
  Unedited 10-class accuracy reported on this 4-class sample and on the P1 sample (part_core.sample(1000, 3000), 10 classes;
  expected 85.9 % / 74.5 %). All edits use part_core.build (eta = jet_eta + sign(jet_eta) * deta; px, py, pz, E, jet_pt and
  jet_energy rebuilt; pT and particle masses kept).
Versions per jet (rows):
  N   natural jet (rebuilt through build).
  A   rigid axis shift: every real particle's (deta, dphi) + d (cos a, sin a), d drawn per jet from {0.02, 0.04, 0.06} rad, a
      uniform in [0, 2 pi). Pair distances and m are unchanged (to float32 rounding); M2 about the stored axis and so m_A rise.
  T1, T2  prong turns: the jet is clustered into 2 exclusive-kT subjets (kT distance min(pT_a^2, pT_b^2) dR_ab^2 in rapidity-
      azimuth, E-scheme merging, pairwise merges until two objects remain); every particle of the softer subjet (lower subjet
      pT) is rotated about the stored axis by theta in the (deta, dphi) plane, |theta| uniform in [10, 40] degrees, random sign,
      theta drawn independently for T1 and T2. Each particle keeps pT and r_i, so M0, M2 and m_A are fixed exactly; the angle
      between the prongs and so m change. (T1, T2) is a pair with the same m_A and different m (the interchange pairs).
  D   matched dilation (calibration only, not in the read-out pool): all (deta, dphi) x k, k solved per jet (4 iterations) so
      that m(D) = m(T1); m and m_A both change by the same dlog (a real mass change). Kept if |log m(D) - log m(T1)| < 1e-3.
  Pool = N, A, T1, T2. Logged: corr(log m, log m_A) on N and on the pool, and the within-jet correlation of the deviations
  from each jet's version mean (the part the read-outs must separate).
C2ST (classifier two-sample test): HistGradientBoosting, 5-fold out-of-fold AUC, natural jets (one random half of the fit
  jets) vs one edit family's versions of the other half (A, T1, D); feature sets: free (axis-free jet features: log m,
  log pT_J, N, pT^D, e2 at beta 0.5 / 1 / 2, leading pT fraction, dR of the two hardest particles), axis (|M1|^2/(M0 M2),
  |M1|/M0, girth, M2/M0), part (the 8 hardest particles' pT/pT_J and r_i) and all; null = natural vs natural. Families with
  AUC(all) > 0.7 flagged; R^2 is also reported per family (N, A, T rows separately).
Read-out sites (per model): inp (jet mean of the scaled per-particle inputs), mom (moment baseline: log M0, M1/M0 (2), log
  M2/M0, |M1|^2/(M0 M2); model-independent), emb, b1..b8 (jet mean over real particles of the residual stream after the
  embedding and after each particle block), cls1, cls2 (class token after class blocks 1 and 2), pairF (the pair-feature tensor
  ln kT, ln z, ln dR, ln m^2, pooled over real i != j pairs by plain mean and by pT_i pT_j-weighted mean: 8 dims;
  model-independent: the 'trivial pair-feature reconstruction'), pairH (the 64-channel pair-embedding hidden layer before the
  head projection, both poolings: 128 dims), pairU (the per-head pair-bias matrices U_h, 8 heads, both poolings: 16 dims),
  pairSum (oracle scalar 0.5 log sum_{i<j} m_ij^2 from the pair tensor's ln m^2; ~ log m).
Read-outs fitted on fit-split pool rows, scored on evaluation-split pool rows (held-out R^2 = 1 - SSE/SST, 95 % CI from 1,000
  Poisson bootstrap resamples over evaluation jets, all versions of a jet together):
  ridge (alpha by 3-fold CV grouped by jet) and MLP (torch, 2 hidden layers of 64 ReLU units, Adam, early stopping on 10 % of
  the fit jets) for log m, log m_A, Delta;
  partial R^2 of log m given log m_A: 1 - SSE(read-out on [X, log m_A, log m_A^2, log m_A^3]) / SSE(cubic in log m_A alone)
  on the evaluation rows (share of the log m variance left after m_A that the site explains); and the reverse (log m_A | log m).
  Calibration (ridge and MLP log m read-out): slope of the read-out change on the true dlog m under the matched dilation
  (D - N, c_dil) and under the turn (T2 - T1, m_A fixed, c_turn); R_turn = c_turn / c_dil (1 = the read-out follows a true-mass
  change at fixed m_A as it follows a real mass change; 0 = it ignores it). Transfer variant: ridge fitted on N + A + D rows
  only (Delta moves only through m_A there), R_turn_transfer on T2 - T1.
  Statistics: H0 = partial R^2(log m | log m_A) at a site <= the untrained value; paired bootstrap (same resamples) against each
  untrained init, p = the larger of the two one-sided p-values; Holm over the sites. Also vs the permuted network and, for the
  pair sites, vs pairF.
Interchange (ParT_full; site s* = residual site (emb..cls2) with the largest ridge partial R^2(log m | log m_A); also cls1):
  pairs (base T1, donor T2) and (base T2, donor T1) of the same jet: same m_A, different m.
  1-d: u = unit gradient of the ridge read-out of r = log m - cubic(log m_A) on the site's jet-mean features (fit split);
  swap h_p <- h_p + ((mean_donor - mean_base) . u) u at every particle (class sites: the token). 4-d DAS: orthonormal Q
  (128 x 4, QR of a free matrix), Adam lr 0.02, 200 steps of 128 fit-split pairs, loss = MSE between the swapped run's 10 logits
  and the donor run's logits.
  Metrics on evaluation pairs: (i) read-out change: the cls2 ridge read-out of r after the swap minus before, per unit true
  d r (donor - base); (ii) W/Z margin d = (logit_W - logit_Z)_swap - (logit_W - logit_Z)_base vs the change expected if the
  network used m: s_dil * (log m_donor - log m_base), with s_dil = sum d_D dlog m_D / sum dlog m_D^2 from the matched dilation
  on W/Z evaluation jets (D vs N); rho = projection ratio sum d e / sum e^2 (95 % bootstrap); the same ratio for the full input
  donor (the turned jet itself) as reference; (iii) calls: share of changed 10-class calls, IIA = share of pairs whose swapped
  call equals the donor run's call (all pairs; informative pairs = donor call differs from base call); McNemar on 10-class
  correctness base vs swapped. Nulls: 200 random unit directions (1-d) and 200 random 4-d subspaces at s* on 2,000 W/Z pairs
  (rank p for IIA and |rho|); the same probe-direction swap and DAS fit in untrained ParT_full (init 0).
Predictions: thesis (m never formed): partial R^2(log m | m_A) at every block and on the pair path at the untrained level
  (pair path not above pairF); interchange moves neither logit nor call (|rho| < 0.1). Alt 1 (m computed, unused): some site
  partial R^2 > 0.3 above untrained, swap moves read-outs, not calls. Alt 2 (m computed and used): rho > 0.3 with McNemar
  p < 0.01.
Figure: one real W jet before/after a prong turn and a rigid shift; per-site partial R^2 and R^2(log m_A) bars, trained vs
  untrained.
  ./sync.sh && ./part_job.sh c-p63-smoke 'python -m interp.p63_true_mass_layers_part --smoke'
  ./sync.sh && ./part_job.sh c-p63-run 'python -m interp.p63_true_mass_layers_part'
  -> $AI_RESULTS/part/p63_true_mass_layers[_smoke].json, p63_example[_smoke].npz, p63_true_mass_layers[_smoke].png
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SMOKE = '--smoke' in sys.argv
SEED = 6363
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
FILES = dict(Wqq='WToQQ_112.root', Zqq='ZToQQ_112.root', QCD='ZJetsToNuNu_112.root', Tbqq='TTBar_112.root')
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TAG = '_smoke' if SMOKE else ''
FN, NPZ, PNG = OUT / f'p63_true_mass_layers{TAG}.json', OUT / f'p63_example{TAG}.npz', OUT / f'p63_true_mass_layers{TAG}.png'
PER = 300 if SMOKE else 8000
NB = 100 if SMOKE else 1000
MLP_EP = 4 if SMOKE else 40
DAS_STEPS = 10 if SMOKE else 200
NRAND = 5 if SMOKE else 200
NNULL = 200 if SMOKE else 2000
BATCH = 128
VERS = ['N', 'A', 'T1', 'T2', 'D']
POOLV = ['N', 'A', 'T1', 'T2']
RES_SITES = ['emb'] + [f'b{l}' for l in range(1, 9)] + ['cls1', 'cls2']
PAIR_SITES = ['pairH', 'pairU']
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


if '--plot' not in sys.argv:   # the figure step needs only numpy / matplotlib (CPU job)
    from interp.part_core import ParTNetwork, inputs, read_root, sample, kin, build, jmass, sub, cat, CL  # noqa: E402
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')


# ================================================================================================ statistics
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def mcnemar(ok0, ok1):
    from scipy.stats import binomtest
    b = int((ok0 & ~ok1).sum()); c = int((~ok0 & ok1).sum())
    return dict(right_to_wrong=b, wrong_to_right=c, p=(float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0))


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


def jsum(jid, x, n):
    return np.bincount(jid, x, n)


def r2_boot(y, p, jid, Wb):
    n = Wb.shape[1]; cnt = np.bincount(jid, minlength=n).astype(float); sy = jsum(jid, y, n); syy = jsum(jid, y * y, n)
    sse = jsum(jid, (y - p) ** 2, n)

    def r(w):
        N = w @ cnt; SY = w @ sy; return 1 - (w @ sse) / np.maximum(w @ syy - SY ** 2 / np.maximum(N, 1e-12), 1e-30)
    b = r(Wb)
    return dict(r2=float(r(np.ones(n))), ci=qci(b)), b


def partial_boot(y, pf, pr, jid, Wb):
    n = Wb.shape[1]; sf = jsum(jid, (y - pf) ** 2, n); sr = jsum(jid, (y - pr) ** 2, n)
    b = 1 - (Wb @ sf) / (Wb @ sr)
    return dict(r2=float(1 - sf.sum() / sr.sum()), ci=qci(b)), b


def slope_boot(dp, dt, jid, Wb):
    n = Wb.shape[1]; a = jsum(jid, dp * dt, n); c = jsum(jid, dt * dt, n); b = (Wb @ a) / np.maximum(Wb @ c, 1e-30)
    return float(a.sum() / max(c.sum(), 1e-30)), b


def p_greater(b):
    """one-sided bootstrap p for H0: statistic <= 0"""
    b = np.asarray(b, float); b = b[np.isfinite(b)]; return float((1 + (b <= 0).sum()) / (len(b) + 1))


# ================================================================================================ data and edits
def load_rows(path, n, rng, chunk=25000):
    import uproot
    N = uproot.open(path)['tree'].num_entries
    if SMOKE: N = min(N, 20000)
    rows = np.sort(rng.choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r): parts.append(sub(read_root(path, s, min(s + chunk, N)), r - s))
    return cat(parts), rows


def obs(J):
    pt, de, dp, _ = kin(J); pt, de, dp = (np.asarray(a, np.float64) for a in (pt, de, dp))
    M0 = pt.sum(1); M1e, M1p = (pt * de).sum(1), (pt * dp).sum(1); M2 = (pt * (de ** 2 + dp ** 2)).sum(1)
    m = np.maximum(jmass(J).astype(np.float64), 1e-3); mA = np.sqrt(M0 * M2)
    return dict(m=m, mA=mA, logm=np.log(m), logmA=np.log(np.maximum(mA, 1e-6)), Delta=np.log(m) - np.log(np.maximum(mA, 1e-6)),
                M0=M0, M1e=M1e, M1p=M1p, M2=M2, girth=(pt * np.hypot(de, dp)).sum(1) / M0)


def kt2_soft(J, dev, chunk=2048):
    """exclusive-kT 2-subjet labels -> boolean (J, 128) mask of the softer subjet's particles (False for jets with < 2 particles)"""
    import torch
    N = len(J['mask']); out = np.zeros(J['mask'].shape, bool); cnt_all = (J['mask'] > 0).sum(1)
    for c0 in range(0, N, chunk):
        jj = np.arange(c0, min(c0 + chunk, N)); L = int(cnt_all[jj].max())
        V = torch.from_numpy(np.stack([J[k][jj, :L] for k in ('px', 'py', 'pz', 'energy')], -1).astype(np.float64)).to(dev)
        real = torch.from_numpy(J['mask'][jj, :L] > 0).to(dev); alive = real.clone(); B = len(jj)
        lab = torch.arange(L, device=dev)[None].repeat(B, 1); ar = torch.arange(B, device=dev)
        iu = torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), 1)[None]
        for _ in range(L):
            act = alive.sum(1) > 2
            if not bool(act.any()): break
            pt = torch.hypot(V[..., 0], V[..., 1]); ph = torch.atan2(V[..., 1], V[..., 0])
            E = torch.maximum(V[..., 3], V[..., 2].abs() + 1e-9); y = 0.5 * torch.log((E + V[..., 2]) / (E - V[..., 2]))
            dph = (ph[:, :, None] - ph[:, None, :] + math.pi) % (2 * math.pi) - math.pi
            d = torch.minimum(pt[:, :, None] ** 2, pt[:, None, :] ** 2) * ((y[:, :, None] - y[:, None, :]) ** 2 + dph ** 2)
            ok = iu & alive[:, :, None] & alive[:, None, :]; d = torch.where(ok, d, torch.full_like(d, float('inf')))
            k = d.view(B, -1).argmin(1); i, j = k // L, k % L
            Vi = V[ar, i] + V[ar, j]; V[ar[act], i[act]] = Vi[act]
            alive[ar[act], j[act]] = False
            lab = torch.where(act[:, None] & (lab == j[:, None]), i[:, None], lab)
        pt = torch.hypot(V[..., 0], V[..., 1]).masked_fill(~alive, -1.0)
        two = alive.sum(1) == 2
        soft_idx = torch.where(alive, pt, torch.full_like(pt, float('inf'))).argmin(1)
        sm = (lab == soft_idx[:, None]) & real & two[:, None]
        out[jj, :L] = sm.cpu().numpy()
    return out


def rot(de, dp, sel, th):
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    return np.where(sel, de * c - dp * s, de), np.where(sel, de * s + dp * c, dp)


def make_versions(J0, rng, dev):
    pt, de, dp, ms = (np.asarray(a, np.float64) for a in kin(J0)); real = pt > 0; n = len(pt)
    Vs = {'N': build(J0, pt, de, dp, ms)}
    d = rng.choice([0.02, 0.04, 0.06], n); a = rng.uniform(0, 2 * np.pi, n)
    Vs['A'] = build(J0, pt, de + (d * np.cos(a))[:, None] * real, dp + (d * np.sin(a))[:, None] * real, ms)
    soft = kt2_soft(Vs['N'], dev)
    th = {k: np.deg2rad(rng.uniform(10, 40, n)) * rng.choice([-1.0, 1.0], n) for k in ('T1', 'T2')}
    for k in ('T1', 'T2'):
        e2, p2 = rot(de, dp, soft, th[k]); Vs[k] = build(J0, pt, e2, p2, ms)
    mT = jmass(Vs['T1']).astype(np.float64); kf = mT / np.maximum(jmass(Vs['N']).astype(np.float64), 1e-6)
    for _ in range(4):
        Jd = build(J0, pt, de * kf[:, None], dp * kf[:, None], ms); kf = kf * mT / np.maximum(jmass(Jd).astype(np.float64), 1e-6)
    Vs['D'] = build(J0, pt, de * kf[:, None], dp * kf[:, None], ms)
    O = {k: obs(v) for k, v in Vs.items()}
    keep = (soft.any(1)) & (np.abs(O['D']['logm'] - O['T1']['logm']) < 1e-3)
    info = dict(shift_dose=d, shift_angle=a, theta1=th['T1'], theta2=th['T2'], k_dil=kf, soft=soft)
    return Vs, O, keep, info


# ================================================================================================ models
def load_model(name):
    import torch
    if name == 'ParT_full': pn = ParTNetwork('full')
    elif name == 'ParT_kin': pn = ParTNetwork('kin')
    elif name.startswith('ParT_untr'): pn = ParTNetwork('full', untrained=True, seed=int(name[-1]))
    elif name == 'ParT_perm':
        pn = ParTNetwork('full'); g = torch.Generator().manual_seed(SEED); sd = pn.model.state_dict(); new = {}; perms = {}
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
        pn.model.load_state_dict(new)
    else: raise ValueError(name)
    pn.model.eval()
    for p in pn.model.parameters(): p.requires_grad_(False)
    return pn


class Run:
    """own forward of weaver's ParticleTransformer with site capture and interchange"""

    def __init__(self, pn):
        import torch
        from weaver.nn.model.ParticleTransformer import pairwise_lv_fts
        self.t, self.pn, self.mod, self.dev, self.plf = torch, pn, pn.model, pn.device, pairwise_lv_fts
        L = list(self.mod.pair_embed.embed); assert len(L) == 13 and isinstance(L[9], torch.nn.GELU) and isinstance(L[12], torch.nn.GELU)
        self.H = self.mod.blocks[0].num_heads if hasattr(self.mod.blocks[0], 'num_heads') else 8

    def T(self, a):
        return self.t.from_numpy(np.ascontiguousarray(a)).to(self.dev)

    def prefix(self, x, v, m, cap=False):
        """-> state dict (pair bias, masks, residual states, class tokens) and, if cap, pooled site features"""
        t, mod = self.t, self.mod; B, _, P = x.shape; mask = m.bool(); pad = ~mask.squeeze(1)
        F = self.plf(v.unsqueeze(-1), v.unsqueeze(-2), num_outputs=4, eps=1e-8)          # (B, 4, P, P)
        z = F.reshape(B, 4, P * P); hid = None
        for k, layer in enumerate(mod.pair_embed.embed):
            z = layer(z)
            if k == 9: hid = z
        U = z.view(B, -1, P, P); am = U.reshape(-1, P, P)
        mp = mask.permute(2, 0, 1).float()                                                  # (P, B, 1)
        cnt = mp.sum(0)
        S = dict(am=am, pad=pad, mp=mp, cnt=cnt, hs=[])
        h = mod.embed(x).masked_fill(~mask.permute(2, 0, 1), 0); S['hs'].append(h)
        for b in mod.blocks:
            h = b(h, x_cls=None, padding_mask=pad, attn_mask=am); S['hs'].append(h)
        c = mod.cls_token.expand(1, h.size(1), -1)
        c1 = mod.cls_blocks[0](h, x_cls=c, padding_mask=pad); c2 = mod.cls_blocks[1](h, x_cls=c1, padding_mask=pad)
        S['c1'], S['c2'] = c1, c2
        L = mod.fc(mod.norm(c2).squeeze(0))
        feats = None
        if cap:
            feats = {}
            jm = lambda h: ((h * mp).sum(0) / cnt)
            for i, nm in enumerate(['emb'] + [f'b{l}' for l in range(1, 9)]): feats[nm] = jm(S['hs'][i])
            feats['cls1'], feats['cls2'] = c1[0], c2[0]
            mk = mask.squeeze(1).float(); pt = t.hypot(v[:, 0], v[:, 1]) * mk
            eye = t.eye(P, device=self.dev)[None]
            M = (mk[:, :, None] * mk[:, None, :]) * (1 - eye); W = pt[:, :, None] * pt[:, None, :] * M
            pool = lambda Z: t.cat([(Z * M[:, None]).sum((2, 3)) / M.sum((1, 2))[:, None].clamp(min=1e-9),
                                    (Z * W[:, None]).sum((2, 3)) / W.sum((1, 2))[:, None].clamp(min=1e-30)], 1)
            feats['pairF'] = pool(F); feats['pairH'] = pool(hid.view(B, -1, P, P)); feats['pairU'] = pool(U)
            feats['pairSum'] = 0.5 * t.log((t.exp(F[:, 3]) * M).sum((1, 2)).clamp(min=1e-12) * 0.5)[:, None]
            feats['inp'] = (x * mk[:, None]).sum(2) / mk.sum(1)[:, None]
        return S, L, feats

    def tail(self, S, k, fn):
        """rerun from site k (0 = emb, l = after block l, 9 = cls1, 10 = cls2) with the site state replaced by fn(state)"""
        mod = self.mod; pad = S['pad']
        if k <= 8:
            h = fn(S['hs'][k])
            for b in mod.blocks[k:]: h = b(h, x_cls=None, padding_mask=pad, attn_mask=S['am'])
            c = mod.cls_token.expand(1, h.size(1), -1)
            c1 = mod.cls_blocks[0](h, x_cls=c, padding_mask=pad); c2 = mod.cls_blocks[1](h, x_cls=c1, padding_mask=pad)
        elif k == 9:
            c2 = mod.cls_blocks[1](S['hs'][8], x_cls=fn(S['c1']), padding_mask=pad)
        else:
            c2 = fn(S['c2'])
        return mod.fc(mod.norm(c2).squeeze(0)), c2[0]


def site_state(S, k):
    return S['hs'][k] if k <= 8 else (S['c1'] if k == 9 else S['c2'])


def swap_fn(S_base, S_don, k, Q):
    """jet-mean (particle sites) or token (class sites) interchange of the Q-subspace component from donor to base"""
    if k <= 8:
        mp, cnt = S_base['mp'], S_base['cnt']
        md = (S_don['hs'][k] * S_don['mp']).sum(0) / S_don['cnt']
        def f(h):
            mb = (h * mp).sum(0) / cnt; return h + (((md - mb) @ Q) @ Q.T)[None]
        return f
    cd = site_state(S_don, k)
    return lambda c: c + ((cd - c) @ Q) @ Q.T


def mbatches(idx, cnt, size=BATCH):
    idx = np.asarray(idx); o = idx[np.argsort(cnt[idx], kind='stable')]
    return [o[i:i + size] for i in range(0, len(o), size)]


def capture(run, Vs, feat, cnt, log_every=60):
    """all versions, all jets -> features {site: (V, N, d) float32}, logits (V, N, 10)"""
    t = run.t; N = len(cnt); Fe = {}; Lg = np.zeros((len(VERS), N, 10), np.float32); checked = False
    for vi, vn in enumerate(VERS):
        J = Vs[vn]
        for bi, idx in enumerate(mbatches(np.arange(N), cnt)):
            x, v, m = inputs(sub(J, idx), feat)
            with t.no_grad():
                xt, vt, mt = run.T(x), run.T(v), run.T(m)
                S, L, f = run.prefix(xt, vt, mt, cap=True)
                if not checked:
                    Lw = run.mod(xt, vt, mt); dmax = float((Lw - L).abs().max()); log(f'    own forward vs weaver: max |dlogit| {dmax:.2e}')
                    assert dmax < 1e-3, dmax; checked = True
            Lg[vi, idx] = L.float().cpu().numpy()
            for k, a in f.items():
                if k not in Fe: Fe[k] = np.zeros((len(VERS), N, a.shape[1]), np.float32)
                Fe[k][vi, idx] = a.float().cpu().numpy()
            del S
        log(f'    captured version {vn}')
    return Fe, Lg


# ================================================================================================ read-outs
class Ridge:
    ALPHAS = 10.0 ** np.arange(-7, 2.1, 1)

    def __init__(self, X, y, jid):
        X = np.asarray(X, np.float64); self.mu = X.mean(0); sd = X.std(0); self.live = sd > 1e-9 * (1 + np.abs(self.mu))
        self.sd = np.where(self.live, sd, np.inf); Z = ((X - self.mu) / self.sd)[:, self.live]
        if not self.live.any():
            self.alpha = None; self.w = np.zeros(0); self.b = float(y.mean()); return
        fold = np.random.default_rng(0).permutation(int(jid.max()) + 1)[jid] % 3; err = np.zeros(len(self.ALPHAS))
        for f in range(3):
            tr, te = fold != f, fold == f
            mz, my = Z[tr].mean(0), y[tr].mean(); Zc = Z[tr] - mz; lam, Vv = np.linalg.eigh(Zc.T @ Zc); Vb = Vv.T @ (Zc.T @ (y[tr] - my))
            Zt = Z[te] - mz
            for i, a in enumerate(self.ALPHAS):
                w = Vv @ (Vb / (lam + a * tr.sum())); err[i] += ((Zt @ w + my - y[te]) ** 2).sum()
        self.alpha = float(self.ALPHAS[int(np.argmin(err))]); mz, my = Z.mean(0), y.mean(); Zc = Z - mz
        self.w = np.linalg.solve(Zc.T @ Zc + self.alpha * len(y) * np.eye(Z.shape[1]), Zc.T @ (y - my)); self.b = my - mz @ self.w

    def __call__(self, X): return ((np.asarray(X, np.float64) - self.mu) / self.sd)[:, self.live] @ self.w + self.b

    def grad_raw(self):
        g = np.zeros(len(self.mu)); g[self.live] = self.w / self.sd[self.live]; return g


def mlp_fit(X, y, jid, Xs, dev, seed=63):
    """2 hidden layers x 64 ReLU, Adam 1e-3, wd 1e-4, batch 1024, early stopping (patience 5) on 10 % of the fit jets;
    returns predictions for each array in Xs"""
    import torch
    X = np.asarray(X, np.float32); mu = X.mean(0); sd = X.std(0); sd = np.where(sd > 1e-9 * (1 + np.abs(mu)), sd, np.inf)
    ym, ys = float(y.mean()), float(y.std() + 1e-12)
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    va = np.random.default_rng(seed).permutation(int(jid.max()) + 1)[jid] % 10 == 0
    Z = torch.from_numpy((X - mu) / sd).to(dev); Y = torch.from_numpy(((y - ym) / ys).astype(np.float32)).to(dev)
    tr_i = torch.from_numpy(np.flatnonzero(~va)).to(dev); va_i = torch.from_numpy(np.flatnonzero(va)).to(dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); best, bad, state = 1e30, 0, None
    for ep in range(MLP_EP):
        net.train(); perm = tr_i[torch.randperm(len(tr_i), generator=g).to(dev)]
        for s in range(0, len(perm), 1024):
            b = perm[s:s + 1024]; loss = ((net(Z[b])[:, 0] - Y[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = float(((net(Z[va_i])[:, 0] - Y[va_i]) ** 2).mean())
        if vl < best - 1e-5: best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5: break
    if state is not None: net.load_state_dict(state)
    net.eval(); outs = []
    with torch.no_grad():
        for A in Xs:
            A = torch.from_numpy((np.asarray(A, np.float32) - mu) / sd).to(dev)
            outs.append(np.concatenate([net(A[i:i + 65536])[:, 0].cpu().numpy() for i in range(0, len(A), 65536)]).astype(np.float64) * ys + ym)
    return outs


def poly3(z, ref):
    mu, sd = ref.mean(), ref.std() + 1e-12; u = (z - mu) / sd
    return np.stack([u, u ** 2, u ** 3], 1)


def ols(X, y):
    A = np.c_[np.ones(len(X)), X]; w = np.linalg.lstsq(A, y, rcond=None)[0]; return lambda Z: np.c_[np.ones(len(Z)), Z] @ w


def rows(F, vers, jets):
    """stack versions -> (rows, d) and the jet index (position in `jets`) per row"""
    return np.concatenate([F[VERS.index(v)][jets] for v in vers]), np.tile(np.arange(len(jets)), len(vers))


def readout_site(X5, O5, fj, ej, Wb, dev, do_mlp, mlp_targets, keep_dir=False):
    """X5 (V, N, d); O5 dict target -> (V, N) -> per-site results; bootstrap arrays of the partial R^2 kept for tests"""
    res, boots = {}, {}
    Xf, jf = rows(X5, POOLV, fj); Xe, je = rows(X5, POOLV, ej)
    y = {k: (rows(O5[k], POOLV, fj)[0], rows(O5[k], POOLV, ej)[0]) for k in ('logm', 'logmA', 'Delta')}
    fam_e = np.repeat(np.array(['N', 'A', 'T', 'T']), len(ej))
    vi = {v: VERS.index(v) for v in VERS}
    Xn, Xd, X1, X2 = (X5[vi[v]][ej] for v in ('N', 'D', 'T1', 'T2'))
    ne = len(ej)
    for kind in (['ridge', 'mlp'] if do_mlp else ['ridge']):
        o = {}
        for tg in ('logm', 'logmA', 'Delta'):
            if kind == 'mlp' and tg not in mlp_targets: continue
            yf, ye = y[tg]
            if kind == 'ridge':
                pr = Ridge(Xf, yf, jf); pe = pr(Xe); pN, pD, p1, p2 = pr(Xn), pr(Xd), pr(X1), pr(X2)
            else:
                pe, pN, pD, p1, p2 = mlp_fit(Xf, yf, jf, [Xe, Xn, Xd, X1, X2], dev)
            e, _ = r2_boot(ye, pe, je, Wb); e = dict(pool=e)
            for fm in ('N', 'A', 'T'):
                k = fam_e == fm; e[f'on_{fm}'] = r2_boot(ye[k], pe[k], je[k], Wb)[0]
            if tg in ('logm', 'logmA'):
                jj = np.arange(ne); tt = O5[tg]
                dtD = tt[vi['D']][ej] - tt[vi['N']][ej]; dtT = tt[vi['T2']][ej] - tt[vi['T1']][ej]
                cD, bD = slope_boot(pD - pN, dtD, jj, Wb); e['c_dil'] = dict(v=cD, ci=qci(bD))
                if tg == 'logm':
                    cT, bT = slope_boot(p2 - p1, dtT, jj, Wb); e['c_turn'] = dict(v=cT, ci=qci(bT))
                    e['R_turn'] = dict(v=cT / cD if abs(cD) > 1e-9 else None, ci=qci(bT / bD))
                    e['turn_readout_mean_abs_change'] = float(np.abs(p2 - p1).mean()); e['turn_true_mean_abs_dlogm'] = float(np.abs(dtT).mean())
                else:
                    pA = pr(X5[vi['A']][ej]) if kind == 'ridge' else None
                    if pA is not None:
                        dtA = tt[vi['A']][ej] - tt[vi['N']][ej]; cA, bA = slope_boot(pA - pN, dtA, jj, Wb); e['c_shift'] = dict(v=cA, ci=qci(bA))
            o[tg] = e
        # partial R^2: log m given log m_A, and the reverse
        for tg, cond, nm in (('logm', 'logmA', 'partial_m_given_mA'), ('logmA', 'logm', 'partial_mA_given_m')):
            yf, ye = y[tg]; cf, ce = y[cond]; Pf, Pe = poly3(cf, cf), poly3(ce, cf)
            red = ols(Pf, yf); pr_e = red(Pe); rf = yf - red(Pf)      # debug 2: fit the residual after the cubic in the other mass
            if kind == 'ridge':
                pfm = Ridge(np.c_[Xf, Pf], rf, jf); pf_e = pr_e + pfm(np.c_[Xe, Pe])
            else:
                pf_e, = mlp_fit(np.c_[Xf, Pf], rf, jf, [np.c_[Xe, Pe]], dev); pf_e = pr_e + pf_e
            e, b = partial_boot(ye, pf_e, pr_e, je, Wb); o[nm] = e; boots[(kind, nm)] = b
            for fm in ('A', 'T'):
                k = fam_e == fm; o[nm][f'on_{fm}'] = partial_boot(ye[k], pf_e[k], pr_e[k], je[k], Wb)[0]
        res[kind] = o
    # transfer: ridge log m fitted on N + A + D rows only, R_turn on T2 - T1
    Xt, jt = rows(X5, ['N', 'A', 'D'], fj); yt = rows(O5['logm'], ['N', 'A', 'D'], fj)[0]
    prt = Ridge(Xt, yt, jt); jj = np.arange(ne); tt = O5['logm']
    cD, bD = slope_boot(prt(Xd) - prt(Xn), tt[vi['D']][ej] - tt[vi['N']][ej], jj, Wb)
    cT, bT = slope_boot(prt(X2) - prt(X1), tt[vi['T2']][ej] - tt[vi['T1']][ej], jj, Wb)
    res['ridge']['transfer_NAD'] = dict(c_dil=cD, c_turn=cT, R_turn=cT / cD if abs(cD) > 1e-9 else None, R_turn_ci=qci(bT / bD))
    if keep_dir:   # probe of r = log m - cubic(log m_A) on the site features (the 1-d 'true-m direction')
        yf, cf = y['logm'][0], y['logmA'][0]; red = ols(poly3(cf, cf), yf); r = yf - red(poly3(cf, cf))
        pr = Ridge(Xf, r, jf); res['_probe'] = (pr, red, cf)
    return res, boots


# ================================================================================================ C2ST
def c2st_feats(J):
    pt, de, dp, _ = (np.asarray(a, np.float64) for a in kin(J)); real = pt > 0; o = obs(J); ptJ = pt.sum(1); z = pt / ptJ[:, None]
    e2 = {b: np.zeros(len(pt)) for b in (0.5, 1.0, 2.0)}
    for i in range(0, len(pt), 400):
        sl = slice(i, i + 400); dR = np.hypot(de[sl, :, None] - de[sl, None, :], dp[sl, :, None] - dp[sl, None, :])
        zz = z[sl, :, None] * z[sl, None, :] * (real[sl, :, None] & real[sl, None, :])
        for b in e2: e2[b][sl] = 0.5 * (zz * dR ** b).sum((1, 2))
    order = np.argsort(-pt, 1); ar = np.arange(len(pt)); i0, i1 = order[:, 0], order[:, 1]
    dR01 = np.hypot(de[ar, i0] - de[ar, i1], dp[ar, i0] - dp[ar, i1])
    free = np.stack([o['logm'], np.log(ptJ), real.sum(1), np.sqrt((pt ** 2).sum(1)) / ptJ, e2[0.5], e2[1.0], e2[2.0], pt.max(1) / ptJ, dR01], 1)
    M1sq = o['M1e'] ** 2 + o['M1p'] ** 2
    axis = np.stack([M1sq / np.maximum(o['M0'] * o['M2'], 1e-30), np.sqrt(M1sq) / o['M0'], o['girth'], o['M2'] / o['M0']], 1)
    r = np.hypot(de, dp); top = order[:, :8]
    part = np.concatenate([np.take_along_axis(z, top, 1), np.take_along_axis(r * real, top, 1)], 1)
    return dict(free=np.nan_to_num(free), axis=np.nan_to_num(axis), part=np.nan_to_num(part), all=np.nan_to_num(np.c_[free, axis, part]))


def c2st_auc(F0, F1, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.metrics import roc_auc_score
    X = np.concatenate([F0, F1]); y = np.r_[np.zeros(len(F0)), np.ones(len(F1))]; rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % 5; s = np.zeros(len(y))
    for f in range(5):
        m = HGB(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False, random_state=seed).fit(X[fold != f], y[fold != f])
        s[fold == f] = m.predict_proba(X[fold == f])[:, 1]
    b = []
    for _ in range(200):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): b.append(roc_auc_score(y[i], s[i]))
    return dict(auc=float(roc_auc_score(y, s)), ci=qci(b), n0=len(F0), n1=len(F1))


# ================================================================================================ interchange
def interchange(run, Vs, O, feat, cnt, y, ej, site, Qs, probe_cls2, rng, s_dil, null=False):
    """Qs: dict name -> torch (128, k) orthonormal. Evaluates on pairs (T1<-T2) and (T2<-T1) of the evaluation jets.
    Returns per name: metrics dict."""
    t = run.t; k = RES_SITES.index(site); out = {}
    pairs = [('T1', 'T2'), ('T2', 'T1')]
    rec = {nm: dict(L=[], c2=[]) for nm in Qs}; base = dict(L=[], Ld=[], c2=[], c2d=[], jet=[], dlogm=[])
    for bv, dv in pairs:
        for idx in mbatches(ej, cnt):
            xb, vb, mb = (run.T(a) for a in inputs(sub(Vs[bv], idx), feat)); xd, vd, md = (run.T(a) for a in inputs(sub(Vs[dv], idx), feat))
            with t.no_grad():
                Sb, Lb, _ = run.prefix(xb, vb, mb); Sd, Ld, _ = run.prefix(xd, vd, md)
                base['L'].append(Lb.float().cpu().numpy()); base['Ld'].append(Ld.float().cpu().numpy())
                base['c2'].append(Sb['c2'][0].float().cpu().numpy()); base['c2d'].append(Sd['c2'][0].float().cpu().numpy())
                base['jet'].append(idx); base['dlogm'].append(O[dv]['logm'][idx] - O[bv]['logm'][idx])
                for nm, Q in Qs.items():
                    L, c2 = run.tail(Sb, k, swap_fn(Sb, Sd, k, Q)); rec[nm]['L'].append(L.float().cpu().numpy()); rec[nm]['c2'].append(c2.float().cpu().numpy())
            del Sb, Sd
    B = {kk: np.concatenate(v) for kk, v in base.items()}
    jet = B['jet']; pos = np.full(len(y), -1); pos[ej] = np.arange(len(ej)); jj = pos[jet]
    yy = y[jet]; wz = (yy == iW) | (yy == iZ)
    marg = lambda L: L[:, iW] - L[:, iZ]
    e = s_dil * B['dlogm']; Wb = None
    cb, cd = B['L'].argmax(1), B['Ld'].argmax(1); inf = cd != cb
    dIn = marg(B['Ld']) - marg(B['L'])
    rp_b, rp_d = probe_cls2(B['c2']), probe_cls2(B['c2d'])
    for nm in Qs:
        L = np.concatenate(rec[nm]['L']); c2 = np.concatenate(rec[nm]['c2']); ci_ = L.argmax(1); dmg = marg(L) - marg(B['L'])
        o = dict(n_pairs=int(len(L)), n_wz_pairs=int(wz.sum()), iia=float((ci_ == cd).mean()), iia_informative=float((ci_ == cd)[inf].mean()) if inf.any() else None,
                 n_informative=int(inf.sum()), calls_changed=float((ci_ != cb).mean()), calls_changed_by_input_turn=float(inf.mean()),
                 rho=float((dmg[wz] * e[wz]).sum() / max((e[wz] ** 2).sum(), 1e-30)), rho_input=float((dIn[wz] * e[wz]).sum() / max((e[wz] ** 2).sum(), 1e-30)),
                 mean_abs_dmargin=float(np.abs(dmg[wz]).mean()), mean_abs_dmargin_input=float(np.abs(dIn[wz]).mean()), mean_abs_expected=float(np.abs(e[wz]).mean()),
                 readout_cls2_shift=float(((probe_cls2(c2) - rp_b)[wz] * (rp_d - rp_b)[wz]).sum() / max(((rp_d - rp_b)[wz] ** 2).sum(), 1e-30)))
        if not null:
            ok0, ok1 = cb == yy, ci_ == yy; o['mcnemar_correct'] = mcnemar(ok0, ok1)
            o['mcnemar_call_change_vs_input'] = mcnemar(ci_ == cb, cd == cb)
            o['_per'] = dict(jj=jj, dmg=dmg, e=e, wz=wz, dIn=dIn, hit=(ci_ == cd).astype(float), inf=inf)
        out[nm] = o
    return out


def boot_pairs(per, Wb):
    jj, wz = per['jj'], per['wz']; n = Wb.shape[1]
    a = np.bincount(jj[wz], (per['dmg'] * per['e'])[wz], n); c = np.bincount(jj[wz], (per['e'] ** 2)[wz], n)
    ai = np.bincount(jj[wz], (per['dIn'] * per['e'])[wz], n)
    h = np.bincount(jj, per['hit'], n); cn = np.bincount(jj, minlength=n).astype(float)
    hi = np.bincount(jj[per['inf']], per['hit'][per['inf']], n); ci_ = np.bincount(jj[per['inf']], minlength=n).astype(float)
    return dict(rho_ci=qci((Wb @ a) / (Wb @ c)), rho_input_ci=qci((Wb @ ai) / (Wb @ c)), iia_ci=qci((Wb @ h) / (Wb @ cn)),
                iia_inf_ci=qci((Wb @ hi) / np.maximum(Wb @ ci_, 1e-9)))


def das_fit(run, Vs, feat, cnt, fj, site, k_dim, rng):
    t = run.t; k = RES_SITES.index(site); g = t.Generator().manual_seed(int(rng.integers(1 << 30)))
    Wp = t.randn(128, k_dim, generator=g).to(run.dev).requires_grad_(True); opt = t.optim.Adam([Wp], lr=0.02); losses = []
    for s in range(DAS_STEPS):
        idx = np.sort(rng.choice(fj, 128, replace=False)); bv, dv = (('T1', 'T2') if s % 2 == 0 else ('T2', 'T1'))
        xb, vb, mb = (run.T(a) for a in inputs(sub(Vs[bv], idx), feat)); xd, vd, md = (run.T(a) for a in inputs(sub(Vs[dv], idx), feat))
        with t.no_grad():
            Sb, _, _ = run.prefix(xb, vb, mb); Sd, Ld, _ = run.prefix(xd, vd, md)
        with t.enable_grad():
            Q = t.linalg.qr(Wp)[0]; L, _ = run.tail(Sb, k, swap_fn(Sb, Sd, k, Q)); loss = ((L - Ld) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    return t.linalg.qr(Wp.detach())[0], float(np.mean(losses[:10])), float(np.mean(losses[-20:]))


# ================================================================================================ main
def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # debug 1: TF32 made own vs weaver differ by 6e-3
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'P63 start; device {dev}; smoke {SMOKE}; PER {PER}')
    Js, split = [], []
    for c, f in FILES.items():
        J, r = load_rows(TEST / f, PER, rng); Js.append(J); log(f'  {f}: {len(r)} random rows (first {r[:3].tolist()}), labels {np.unique(J["label"]).tolist()}')
    J0 = cat(Js); N = len(J0['label']); y = J0['label'].astype(int)
    lab = np.repeat(np.arange(len(FILES)), PER); fit = np.zeros(N, bool)
    for c in range(len(FILES)):
        ii = np.flatnonzero(lab == c); fit[rng.permutation(ii)[:len(ii) // 2]] = True
    Vs, O, keep, info = make_versions(J0, rng, dev)
    fj, ej = np.flatnonzero(fit & keep), np.flatnonzero(~fit & keep)
    log(f'versions built: {N} jets, kept {int(keep.sum())} (soft subjet found and dilation matched); fit {len(fj)}, eval {len(ej)}')
    R = dict(meta=dict(seed=SEED, files=FILES, per_class=PER, n_jets=N, n_kept=int(keep.sum()), n_fit=len(fj), n_eval=len(ej), NB=NB, smoke=SMOKE))
    # ---- pool description
    k_ = keep; d = {}
    for v in VERS:
        d[v] = dict(dlogm_median=float(np.median(O[v]['logm'][k_] - O['N']['logm'][k_])), dlogmA_median=float(np.median(O[v]['logmA'][k_] - O['N']['logmA'][k_])),
                    dlogm_q=np.quantile(O[v]['logm'][k_] - O['N']['logm'][k_], [0.05, 0.25, 0.75, 0.95]).tolist(),
                    dlogmA_q=np.quantile(O[v]['logmA'][k_] - O['N']['logmA'][k_], [0.05, 0.25, 0.75, 0.95]).tolist(),
                    max_abs_dlogm=float(np.abs(O[v]['logm'][k_] - O['N']['logm'][k_]).max()), max_abs_dlogmA=float(np.abs(O[v]['logmA'][k_] - O['N']['logmA'][k_]).max()))
    lm = np.stack([O[v]['logm'][k_] for v in POOLV]); la = np.stack([O[v]['logmA'][k_] for v in POOLV])
    dm, da = lm - lm.mean(0), la - la.mean(0)
    d['corr_logm_logmA_natural'] = float(np.corrcoef(O['N']['logm'][k_], O['N']['logmA'][k_])[0, 1])
    d['corr_logm_logmA_pool'] = float(np.corrcoef(lm.ravel(), la.ravel())[0, 1])
    d['corr_within_jet_deviation'] = float(np.corrcoef(dm.ravel(), da.ravel())[0, 1])
    d['corr_Delta_logmA_pool'] = float(np.corrcoef((lm - la).ravel(), la.ravel())[0, 1])
    d['sd_Delta'] = {v: float((O[v]['logm'] - O[v]['logmA'])[k_].std()) for v in VERS}
    d['shift_dlogm_by_dose'] = {str(x): float(np.abs(O['A']['logm'] - O['N']['logm'])[k_ & (info['shift_dose'] == x)].max()) for x in (0.02, 0.04, 0.06)}
    d['soft_subjet_particles_median'] = float(np.median(info['soft'][k_].sum(1))); d['k_dil_q'] = np.quantile(info['k_dil'][k_], [0.05, 0.5, 0.95]).tolist()
    R['pool'] = d
    log('pool: ' + json.dumps({kk: d[kk] for kk in ('corr_logm_logmA_natural', 'corr_logm_logmA_pool', 'corr_within_jet_deviation', 'sd_Delta')}))
    for v in VERS: log(f'  {v}: dlogm median {d[v]["dlogm_median"]:+.4f} q {np.round(d[v]["dlogm_q"], 4).tolist()} | dlogmA median {d[v]["dlogmA_median"]:+.4f} q {np.round(d[v]["dlogmA_q"], 4).tolist()}')
    # ---- C2ST
    t1 = time.time(); half = rng.permutation(fj); h0, h1 = half[:len(half) // 2], half[len(half) // 2:]
    if not SMOKE: h0, h1 = h0[:4000], h1[:4000]
    F0 = c2st_feats(sub(Vs['N'], h0)); cc = {}
    for v in ('A', 'T1', 'D', 'N'):
        F1 = c2st_feats(sub(Vs[v], h1)); cc[v if v != 'N' else 'null_N_vs_N'] = {fs: c2st_auc(F0[fs], F1[fs], SEED) for fs in F0}
    R['c2st'] = cc; R['c2st_flagged'] = [v for v in cc if cc[v]['all']['auc'] > 0.7]
    log(f'C2ST ({time.time() - t1:.0f} s): ' + '; '.join(f'{v}: ' + ', '.join(f'{fs} {cc[v][fs]["auc"]:.3f}' for fs in cc[v]) for v in cc))
    # ---- example jet for the figure
    pt, de, dp, _ = kin(Vs['N']); cand = [j for j in ej if y[j] == iW and (Vs['N']['mask'][j] > 0).sum() >= 25
                                         and 0.25 < (pt[j] * info['soft'][j]).sum() / pt[j].sum() < 0.5 and O['T1']['logm'][j] - O['N']['logm'][j] < -0.06]
    jx = int(cand[0]) if cand else int(ej[0])
    ex = dict(pt=pt[jx], deN=de[jx], dpN=dp[jx], soft=info['soft'][jx])
    for v in ('T1', 'A'):
        p_, e_, q_, _ = kin(Vs[v]); ex[f'de{v}'], ex[f'dp{v}'] = e_[jx], q_[jx]
    for v in VERS: ex[f'm_{v}'], ex[f'mA_{v}'] = O[v]['m'][jx], O[v]['mA'][jx]
    ex['theta1'] = info['theta1'][jx]; ex['shift'] = np.array([info['shift_dose'][jx], info['shift_angle'][jx]]); ex['jet_pt'] = J0['jet_pt'][jx]
    np.savez(NPZ, **ex)
    # ---- accuracy on the P1 sample
    acc = {}
    JP = sample(per_class=100 if SMOKE else 1000, start=3000)
    for nm, ft in (('ParT_full', 'full'), ('ParT_kin', 'kin')):
        pn = ParTNetwork(ft); L = pn.run(JP)['logits']; acc[nm] = float((L.argmax(1) == JP['label']).mean()); del pn
    R['accuracy_P1_sample'] = acc; log(f'P1-sample 10-class accuracy: {acc} (expected 0.859 / 0.745)')
    # ---- capture and read-outs per model
    Wb = np.random.default_rng(SEED + 1).poisson(1.0, (NB, len(ej))).astype(np.float64)
    O5 = {k: np.stack([O[v][k] for v in VERS]) for k in ('logm', 'logmA', 'Delta')}
    cnt = (J0['mask'] > 0).sum(1)
    models = ['ParT_full', 'ParT_untr0', 'ParT_untr1', 'ParT_perm', 'ParT_kin']
    RO, BO, ACC, LOGITS, PROBES = {}, {}, {}, {}, {}
    for nm in models:
        t1 = time.time(); pn = load_model(nm); run = Run(pn); feat = 'kin' if nm == 'ParT_kin' else 'full'
        Fe, Lg = capture(run, Vs, feat, cnt)
        acc = float((Lg[0][keep].argmax(1) == y[keep]).mean()); ACC[nm] = acc
        ccall = {v: float((Lg[VERS.index(v)][keep].argmax(1) != Lg[0][keep].argmax(1)).mean()) for v in VERS[1:]}
        wW = keep & (y == iW); wz_flow = {v: float((Lg[VERS.index(v)][wW].argmax(1) == iZ).mean()) for v in VERS}
        for dose in (0.02, 0.04, 0.06):
            s_ = wW & (info['shift_dose'] == dose); wz_flow[f'A{dose}'] = float((Lg[1][s_].argmax(1) == iZ).mean()); wz_flow[f'N_at_A{dose}'] = float((Lg[0][s_].argmax(1) == iZ).mean())
        LOGITS[nm] = dict(accuracy_4class_sample=acc, calls_changed_vs_N=ccall, true_W_called_Z=wz_flow)
        log(f'  {nm}: 4-class-sample 10-class accuracy {acc:.4f}; calls changed {ccall}; true W called Z {wz_flow}  ({time.time() - t1:.0f} s capture)')
        # dilation slope of the W/Z margin
        wzE = np.zeros(N, bool); wzE[ej] = True; wzE &= (y == iW) | (y == iZ)
        mg = lambda L: L[:, iW] - L[:, iZ]
        dmD = mg(Lg[4]) - mg(Lg[0]); dlD = O['D']['logm'] - O['N']['logm']
        s_dil = float((dmD[wzE] * dlD[wzE]).sum() / (dlD[wzE] ** 2).sum())
        dmT = mg(Lg[3]) - mg(Lg[2]); dlT = O['T2']['logm'] - O['T1']['logm']
        LOGITS[nm]['s_dil_margin_per_dlogm'] = s_dil
        LOGITS[nm]['turn_margin_ratio_to_m_prediction'] = float((dmT[wzE] * s_dil * dlT[wzE]).sum() / ((s_dil * dlT[wzE]) ** 2).sum())
        jjw = np.flatnonzero(wzE[ej]); Wbw = Wb[:, jjw]; ejw = ej[wzE[ej]]
        a_ = dmT[ejw] * s_dil * dlT[ejw]; c_ = (s_dil * dlT[ejw]) ** 2; LOGITS[nm]['turn_margin_ratio_ci'] = qci((Wbw @ a_) / (Wbw @ c_))
        log(f'    W/Z margin per dlog m under the matched dilation s_dil {s_dil:.3f}; input turn margin / m-prediction {LOGITS[nm]["turn_margin_ratio_to_m_prediction"]:.3f} {np.round(LOGITS[nm]["turn_margin_ratio_ci"], 3).tolist()}')
        sites = (['inp', 'mom', 'pairF', 'pairSum'] if nm in ('ParT_full', 'ParT_kin') else []) + RES_SITES + PAIR_SITES
        if nm == 'ParT_kin': sites = ['inp'] + RES_SITES + PAIR_SITES
        if nm == 'ParT_full':
            o = O; M1sq = lambda v: o[v]['M1e'] ** 2 + o[v]['M1p'] ** 2
            Fe['mom'] = np.stack([np.stack([np.log(o[v]['M0']), o[v]['M1e'] / o[v]['M0'], o[v]['M1p'] / o[v]['M0'], np.log(o[v]['M2'] / o[v]['M0']),
                                            M1sq(v) / (o[v]['M0'] * o[v]['M2'])], 1) for v in VERS]).astype(np.float32)
        RO[nm], BO[nm] = {}, {}
        do_mlp_full = nm in ('ParT_full', 'ParT_untr0', 'ParT_untr1')
        for s in sites:
            t2 = time.time()
            r_, b_ = readout_site(Fe[s], O5, fj, ej, Wb, dev, do_mlp=True, mlp_targets=(['logm', 'logmA'] if do_mlp_full else []),
                                  keep_dir=(nm in ('ParT_full', 'ParT_untr0') and s in RES_SITES))
            if '_probe' in r_: PROBES[(nm, s)] = r_.pop('_probe')
            RO[nm][s], BO[nm][s] = r_, b_
            rr = r_['ridge']; rm = r_['mlp']
            log(f'    {nm} {s:6s} ({time.time() - t2:3.0f} s): ridge R2 logm {rr["logm"]["pool"]["r2"]:.3f} logmA {rr["logmA"]["pool"]["r2"]:.3f} Delta {rr["Delta"]["pool"]["r2"]:.3f} '
                f'| partial m|mA ridge {rr["partial_m_given_mA"]["r2"]:.3f} mlp {rm["partial_m_given_mA"]["r2"]:.3f} | mA|m ridge {rr["partial_mA_given_m"]["r2"]:.3f} mlp {rm["partial_mA_given_m"]["r2"]:.3f} '
                f'| R_turn ridge {rr["logm"]["R_turn"]["v"] if rr["logm"]["R_turn"]["v"] is None else round(rr["logm"]["R_turn"]["v"], 3)} transfer {rr["transfer_NAD"]["R_turn"]}')
        if nm == 'ParT_full':
            FULL_LG, FULL_RUN = Lg, run; del Fe
        elif nm == 'ParT_untr0':
            U_LG, U_RUN = Lg, run; del Fe
        else:
            del Fe, run, pn
        torch.cuda.empty_cache() if dev == 'cuda' else None
    R['accuracy_this_sample'] = ACC; R['logits'] = LOGITS
    # ---- tests vs untrained (paired bootstrap, Holm over sites)
    tests = {}
    for kind in ('ridge', 'mlp'):
        for target in ('partial_m_given_mA', 'partial_mA_given_m'):
            for nmT in ('ParT_full', 'ParT_kin'):
                ss = RES_SITES + PAIR_SITES; ps, rows_ = [], []
                for s in ss:
                    bT = BO[nmT][s][(kind, target)]; pv = []; diffs = {}
                    for cn in ('ParT_untr0', 'ParT_untr1'):
                        dd = bT - BO[cn][s][(kind, target)]; pv.append(p_greater(dd)); diffs[cn] = dict(v=RO[nmT][s][kind][target]['r2'] - RO[cn][s][kind][target]['r2'], ci=qci(dd))
                    dperm = bT - BO['ParT_perm'][s][(kind, target)]
                    row = dict(site=s, trained=RO[nmT][s][kind][target]['r2'], trained_ci=RO[nmT][s][kind][target]['ci'],
                               untr0=RO['ParT_untr0'][s][kind][target]['r2'], untr1=RO['ParT_untr1'][s][kind][target]['r2'], perm=RO['ParT_perm'][s][kind][target]['r2'],
                               diff_vs_untr=diffs, diff_vs_perm=dict(v=RO[nmT][s][kind][target]['r2'] - RO['ParT_perm'][s][kind][target]['r2'], ci=qci(dperm)),
                               p_vs_untrained=max(pv))
                    if s in PAIR_SITES and nmT == 'ParT_full':
                        dpf = bT - BO['ParT_full']['pairF'][(kind, target)]; row['diff_vs_pairF'] = dict(v=RO[nmT][s][kind][target]['r2'] - RO['ParT_full']['pairF'][kind][target]['r2'], ci=qci(dpf), p=p_greater(dpf))
                    rows_.append(row); ps.append(max(pv))
                for row, ph in zip(rows_, holm(ps)): row['p_holm'] = ph
                tests[f'{nmT}|{kind}|{target}'] = rows_
    R['tests_vs_untrained'] = tests
    for kind in ('ridge', 'mlp'):
        tr = tests[f'ParT_full|{kind}|partial_m_given_mA']
        log(f'partial R2(log m | log m_A) {kind}: ' + ', '.join(f'{r["site"]} {r["trained"]:.3f}/{r["untr0"]:.3f}/{r["untr1"]:.3f} p_holm {r["p_holm"]:.3g}' for r in tr))
    # ---- interchange
    resS = [s for s in RES_SITES]
    best = max(resS, key=lambda s: RO['ParT_full'][s]['ridge']['partial_m_given_mA']['r2'])
    isites = [best] + (['cls1'] if best != 'cls1' else [])
    log(f'interchange sites: {isites} (best ridge partial R2 site {best})')
    ICH = {}
    for nm, run, Lg in (('ParT_full', FULL_RUN, FULL_LG), ('ParT_untr0', U_RUN, U_LG)):
        s_dil = LOGITS[nm]['s_dil_margin_per_dlogm']
        pr2, _, _ = PROBES[(nm, 'cls2')]; pc2 = lambda C, pr2=pr2: pr2(C)
        ICH[nm] = {}
        for site in isites:
            t1 = time.time(); pr, red, cf = PROBES[(nm, site)]; g = pr.grad_raw(); u = g / max(np.linalg.norm(g), 1e-30)
            Qs = {'probe1d': torch.from_numpy(u[:, None].astype(np.float32)).to(run.dev)}
            Q4, l0, l1 = das_fit(run, Vs, 'full', cnt, fj, site, 4, rng); Qs['das4'] = Q4
            r_ = interchange(run, Vs, O, 'full', cnt, y, ej, site, Qs, pc2, rng, s_dil)
            for q in r_: r_[q].update(boot_pairs(r_[q].pop('_per'), Wb))
            r_['das4']['loss_first10'], r_['das4']['loss_last20'] = l0, l1
            ent = dict(main=r_)
            if nm == 'ParT_full' and site == best:
                # random nulls on NNULL W/Z evaluation pairs
                ejw = ej[(y[ej] == iW) | (y[ej] == iZ)]; sel = np.sort(rng.choice(ejw, min(NNULL, len(ejw)), replace=False))
                Qn = {}
                for i in range(NRAND):
                    v_ = rng.standard_normal(128); Qn[f'r1_{i}'] = torch.from_numpy((v_ / np.linalg.norm(v_))[:, None].astype(np.float32)).to(run.dev)
                    Qn[f'r4_{i}'] = torch.linalg.qr(torch.from_numpy(rng.standard_normal((128, 4)).astype(np.float32)))[0].to(run.dev)
                Qn['probe1d'], Qn['das4'] = Qs['probe1d'], Qs['das4']
                rn = interchange(run, Vs, O, 'full', cnt, y, sel, site, Qn, pc2, rng, s_dil, null=True)
                nul = {}
                for dim, ref in (('r1', 'probe1d'), ('r4', 'das4')):
                    xs = [rn[q] for q in rn if q.startswith(dim + '_')]
                    for met in ('iia', 'iia_informative', 'rho', 'calls_changed', 'readout_cls2_shift'):
                        vals = np.array([z[met] for z in xs if z[met] is not None], float); ov = rn[ref][met]
                        nul[f'{ref}_{met}'] = dict(obs=ov, null_q=np.quantile(vals, [0.025, 0.5, 0.975]).tolist() if len(vals) else None,
                                                   p_rank=float((1 + (np.abs(vals) >= abs(ov)).sum()) / (len(vals) + 1)) if (len(vals) and ov is not None) else None)
                ent['null'] = nul; ent['null_n_pairs'] = int(rn['probe1d']['n_pairs'])
            ICH[nm][site] = ent
            m_ = r_
            log(f'  {nm} @ {site} ({time.time() - t1:.0f} s): ' + ' | '.join(f'{q}: IIA {m_[q]["iia"]:.3f} (inf {m_[q]["iia_informative"]}) calls {m_[q]["calls_changed"]:.4f} rho {m_[q]["rho"]:.3f} {np.round(m_[q]["rho_ci"], 3).tolist()} '
                                                                       f'rho_input {m_[q]["rho_input"]:.3f} cls2-readout {m_[q]["readout_cls2_shift"]:.3f} McNemar p {m_[q]["mcnemar_correct"]["p"]:.3g}' for q in m_ if isinstance(m_[q], dict) and 'iia' in m_[q]))
            if 'null' in ent: log('    null: ' + json.dumps({k_: (round(v_['obs'], 4) if v_['obs'] is not None else None, v_['null_q'], v_['p_rank']) for k_, v_ in ent['null'].items()}))
    R['interchange'] = dict(sites=isites, best_site=best, results=ICH)
    R['readouts'] = RO
    FN.write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, np.generic) else str(o))))
    log(f'wrote {FN}')
    plot(R)
    log('done')


# ================================================================================================ figure
def plot(R=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = R or json.loads(FN.read_text()); ex = np.load(NPZ)
    fig = plt.figure(figsize=(17, 5.0)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 2.0, 1.05])
    pt = ex['pt']; real = pt > 0; s = 400 * pt[real] / pt.max(); soft = ex['soft'][real]
    ax = fig.add_subplot(gs[0])
    ax.scatter(ex['deN'][real][~soft], ex['dpN'][real][~soft], s=s[~soft], c='0.55', alpha=0.6, label='harder subjet (not moved)')
    ax.scatter(ex['deN'][real][soft], ex['dpN'][real][soft], s=s[soft], facecolors='none', edgecolors='C0', label='softer subjet, before')
    ax.scatter(ex['deT1'][real][soft], ex['dpT1'][real][soft], s=s[soft], c='C3', alpha=0.7, label=f'after turn by {np.rad2deg(float(ex["theta1"])):+.0f} deg')
    ax.plot(0, 0, 'k+', ms=14, mew=2)
    ax.set_title(f'(a) prong turn about the stored axis (+), W jet\nm {float(ex["m_N"]):.1f} -> {float(ex["m_T1"]):.1f} GeV, m_A {float(ex["mA_N"]):.1f} -> {float(ex["mA_T1"]):.1f} GeV', fontsize=9)
    ax.set_xlabel('Δη to stored axis (rad)'); ax.set_ylabel('Δφ to stored axis (rad)'); ax.legend(fontsize=7, loc='lower left'); ax.set_aspect('equal', 'datalim')
    ax = fig.add_subplot(gs[1])
    ax.scatter(ex['deN'][real], ex['dpN'][real], s=s, facecolors='none', edgecolors='0.4', label='before')
    ax.scatter(ex['deA'][real], ex['dpA'][real], s=s, c='C2', alpha=0.6, label=f'after rigid shift by {float(ex["shift"][0]):.2f} rad')
    ax.plot(0, 0, 'k+', ms=14, mew=2)
    ax.set_title(f'(b) rigid shift off the stored axis (+)\nm {float(ex["m_N"]):.1f} -> {float(ex["m_A"]):.1f} GeV, m_A {float(ex["mA_N"]):.1f} -> {float(ex["mA_A"]):.1f} GeV', fontsize=9)
    ax.set_xlabel('Δη to stored axis (rad)'); ax.legend(fontsize=7, loc='lower left'); ax.set_aspect('equal', 'datalim')
    ax = fig.add_subplot(gs[2]); RO = R['readouts']
    sites = ['inp', 'emb'] + [f'b{l}' for l in range(1, 9)] + ['cls1', 'cls2', 'pairF', 'pairH', 'pairU']
    x = np.arange(len(sites)); w = 0.22
    pm = lambda nm, s_, key='r2': (RO[nm][s_]['mlp']['partial_m_given_mA'][key] if key == 'r2' else RO[nm][s_]['mlp']['partial_m_given_mA'][key]['r2'])
    full = [pm('ParT_full', s_) for s_ in sites]; fullT = [pm('ParT_full', s_, 'on_T') for s_ in sites]
    unt = [np.mean([pm(u, s_) for u in ('ParT_untr0', 'ParT_untr1')]) if s_ not in ('inp', 'pairF') else pm('ParT_full', s_) for s_ in sites]
    untT = [np.mean([pm(u, s_, 'on_T') for u in ('ParT_untr0', 'ParT_untr1')]) if s_ not in ('inp', 'pairF') else pm('ParT_full', s_, 'on_T') for s_ in sites]
    ax.bar(x - 1.5 * w, full, w, color='C3', label='ParT_full, all evaluation rows')
    ax.bar(x - 0.5 * w, unt, w, color='0.6', label='untrained ParT (mean of 2 inits), all rows')
    ax.bar(x + 0.5 * w, fullT, w, color='C3', alpha=0.45, hatch='//', label='ParT_full, prong-turn rows only (m_A fixed)')
    ax.bar(x + 1.5 * w, untT, w, color='0.6', alpha=0.45, hatch='//', label='untrained, prong-turn rows only')
    ax.axhline(0, color='k', lw=0.6); ax.set_xticks(x); ax.set_xticklabels(sites, rotation=45, fontsize=8)
    ax.set_ylabel('held-out partial R² of log m given log m_A\n(MLP read-out; share of the true-mass\nvariation left after m_A)', fontsize=8)
    ax.legend(fontsize=7, loc='upper right'); ax.set_ylim(-0.4, 1.0)
    ax.set_title('(c) true mass beyond m_A: decodable in every trained block, not in untrained', fontsize=9)
    ax = fig.add_subplot(gs[3]); I = R['interchange']['results']['ParT_full']; b = R['interchange']['best_site']
    rows = [('input turn\n(whole jet)', I[b]['main']['das4']['rho_input'], I[b]['main']['das4']['rho_input_ci'], 'C3'),
            (f'DAS 4-d swap\nat cls1', I['cls1']['main']['das4']['rho'], I['cls1']['main']['das4']['rho_ci'], 'C1'),
            (f'DAS 4-d swap\nat {b}', I[b]['main']['das4']['rho'], I[b]['main']['das4']['rho_ci'], 'C1'),
            (f'true-m probe\n1-d swap at {b}', I[b]['main']['probe1d']['rho'], I[b]['main']['probe1d']['rho_ci'], 'C0')]
    yy = np.arange(len(rows))[::-1]
    for yv, (lb, v, c, col) in zip(yy, rows):
        ax.barh(yv, v, color=col, alpha=0.8); ax.errorbar(v, yv, xerr=[[v - c[0]], [c[1] - v]], color='k', capsize=3)
    nq = I[b]['null']['das4_rho']['null_q'][2]; ax.axvline(nq, color='0.4', ls=':', label=f'random 4-d swaps, 97.5 % ({nq:.3f})')
    ax.axvline(1.0, color='k', ls='--', lw=0.8, label='1 = as if the network used m')
    ax.set_yticks(yy); ax.set_yticklabels([r_[0] for r_ in rows], fontsize=8); ax.set_xlim(-0.05, 1.1)
    ax.set_xlabel('W−Z logit change / change predicted\nfrom a real mass change (matched dilation)', fontsize=8)
    ax.legend(fontsize=7, loc='lower right'); ax.set_title('(d) the decision follows a true-mass\nchange at fixed m_A weakly', fontsize=9)
    fig.suptitle('P63 (ParT_full, JetClass W/Z/QCD/top, 16,000 evaluation jets): later blocks encode the true mass m beyond the axis-moment mass m_A = √(ΣpT·ΣpT ΔR²), '
                 'but the W/Z decision moves 0.24 as much as for a real mass change', fontsize=9.5)
    fig.tight_layout(); fig.savefig(PNG, dpi=130); print(f'wrote {PNG}', flush=True)


if __name__ == '__main__':
    if '--plot' in sys.argv: plot()
    else: main()
