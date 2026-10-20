"""P11 — ParT seed replication: the pre-registered statements S1-S4 of P10 on the six retrained checkpoints (claims C3s, C3,
C2b, C7: is the axis-relative mass response, its input route and its head-level route a property of training ParT on JetClass
or of the one released checkpoint).

Models (per feature set full / kin): released ParT_<feat> (reference, 100M-jet training), seeds 0/1/2 retrained in P10 on the
2M-jet subset (/j-jepa-vol/aaron-interp/models/part_seeds/<feat>_s<k>/net_best_epoch_state.pt, best validation epoch of 50),
and two controls: 'untrained' = the same architecture freshly initialised by weaver's own initialisation with torch seed 0
(no training), and for kin 'shuffled' = the P14 arm-s model (ParT, kin inputs, 2 outputs, labels permuted over the training
jets, 2 epochs; /j-jepa-vol/aaron-interp/models/p14/s/final.pt; only the read-out and attention statements apply to it).

Samples (JetClass test_20M, files <sample>_100.root; never first-k):
  P1 sample: rows 3,000-4,000 of each class file (N = 10,000): 10-class accuracy and confusion; ridge read-out fit.
  WZ sample: 10,000 random rows each of WToQQ_100 and ZToQQ_100 (numpy default_rng(11), rows 3,000-3,999 excluded so the
             read-out is not fitted on the evaluation jets), N = 20,000; jets with fewer than 2 particles dropped (M2 = 0).
  S3 subset: 2,000 W + 2,000 Z random jets of the WZ sample (default_rng(12)).
Build: part_core.build (eta = jet_eta + sign(jet_eta) * deta); baseline = the unedited particles rebuilt.

Edits on the WZ sample (P1 / P10 definitions):
  phi    every real particle's dphi + 0.06 rad about the stored axis (rigid rotation: invariant mass unchanged; the axis
         moment sqrt(M0 M2), M0 = sum pT, M2 = sum pT dR^2, rises by 1/2 dlog(M0 M2) = pred_j per jet).
  match  matched dilation: every particle's (deta, dphi) x k_j with k_j = exp(pred_j): the same per-jet change of
         sqrt(M0 M2) as phi, but a real mass change (dlog m ~ pred_j).
  k1.1   angles x 1.1 (real mass change, dlog m +0.094).
  rot    whole jet incl. the stored axis rotated by 0.06 rad in phi (inputs equal up to float rounding): numerical null.
  Route variants (P1v / P2): per-particle features (as fed) from one jet and four-vectors (the only source of the pair
  features) from another: phi_pair = (x_base, v_phi), phi_pp = (x_phi, v_base), phi_dR = x_base with only deltaR from
  x_phi, phi_dphi = only dphi, phi_dphi_dR = both; k1.1_pair, k1.1_pp; rot_pair = (x_base, v_rot) (pair-route null).

S1 read-out: ridge regression of log m (invariant mass of the built jet) on the class token z (128 numbers after the final
   LayerNorm), P1's Ridge (alpha grid 1e-4..1e4) fitted on the P1 sample (fit 3/4, alpha on the last 1/4 of a seed-11
   permutation, refit on all), applied to the WZ sample. dlog m_hat(edit) = readout(z_edit) - readout(z_base), per jet.
   ratio_match = mean dlog m_hat(phi) / mean dlog m_hat(match): H_axis (the network reads sqrt(M0 M2) about the stored axis)
   predicts 1, H_inv (invariant mass) predicts 0; pre-registered window 0.7-1.4. ratio_cal = the P10 form: (dm_hat(phi) /
   pred(phi)) / (dm_hat(k1.1) / pred(k1.1)). R = OLS slope through 0, over true-W jets, of dd(phi) on dd(match),
   dd = per-jet change of logit_W - logit_Z; R_mean = mean dd(phi) / mean dd(match). W->Z = fraction of true W called Z
   (10-class argmax) before -> after, paired bootstrap CI, exact McNemar (Holm over all models x edits of the job).
S2 route: share_X = mean dm_hat(X) / mean dm_hat(phi) (and the same for dd on W); pre-registered: share_dR > 0.8 (full),
   share_dphi_dR > 0.8 (kin); pair share vs the rot_pair null; k1.1 pair / per-particle shares (P2: 1-6 % pair).
S3 block route (class-token unit): u* = argmax_u |W[u, W] - W[u, Z]| * sd(z_u) (last-layer weights, sd over the P1 sample;
   u* = 97 = 'n98' in the released ParT_full). Response = dz_u*(k1.1) / sd on the S3 subset. Removal share of a head set =
   1 - mean dz_u*(edited run with those heads patched to their unedited-run values) / mean dz_u*(edited run), over the 4,000
   W/Z jets (P3v 'direct patch removal'), bootstrap CI over jets; primary form (robust when W and Z respond with opposite
   signs): per-jet projection 1 - sum_j dP_j dE_j / sum_j dE_j^2; also on W and Z separately and on logit_W - logit_Z. Head sets: class-attention block 1 (cls_blocks[0], 8 heads;
   P3v: 0.987 in the released model; the class token's query in this block is the constant cls token, so every
   particle-to-class-token path through class block 1 runs through these 8 heads), class block 2, each of the 16 class heads,
   30 block-stratified random sets (4 heads of each class block), a resample ablation of class block 1 (each jet's 8 head
   contexts replaced by a random other jet's unedited contexts in both runs); particle block 1 (blocks[0], 8 heads; the literal
   'particle block 1' of the P10 text), each of the 64 particle heads, 20 block-stratified random sets (one head per particle
   block), all 64 particle heads. Per-head shares: bootstrap two-sided p of share = 0, Holm within the 16 class heads and
   within the 64 particle heads. Pre-registered: block-1 share > 0.9; expected under the thesis: the head with the largest
   single share differs between seeds.
S4 switch scan: per head, a per-jet statistic vs jet mass on the WZ sample: class heads (16): the class token's attention
   share (key 0); particle heads (64): attention-weighted pair distance sum_j a_ij dR_ij averaged over real query particles,
   divided by the jet's mean pair dR (relative). Binned means (5 GeV bins, 50-140 GeV) fitted by y = a + c / (1 + exp((m -
   m50) / w)) (weighted least squares; grid m50 40-160 GeV step 0.5, w 0.5-80 GeV, 48 log steps; a, c closed form). Switch
   (pre-registered form) = m50 in 60-110 GeV, w < 15 GeV, |c| >= 0.2 max(|a|, |a + c|), met in >= 95 % of 200 jet bootstraps,
   and BH q < 0.05 over the 80 heads of the permutation p (4,000 mass shuffles; H0: the statistic does not depend on mass;
   test statistic = span, max - min binned mean). Strong switch: in addition |c| >= 0.2 (P3's 20-point rule for shares). Released ParT_full reference: b2h8 (class block 2 head 8), m50 87 GeV, w 4.3 GeV (P3).

  ./part_job.sh c-p11-full 'python -m interp.p11_seed_tests --feat full --seeds all'
  ./part_job.sh c-p11-kin  'python -m interp.p11_seed_tests --feat kin --seeds all'
  ./part_job.sh c-p11-smoke 'python -m interp.p11_seed_tests --feat full --smoke'
  python -m interp.p11_seed_tests --plot    (figure from the jsons)
  -> $AI_RESULTS/part/p11_seed_tests_<feat>[_smoke].json, results/part/p11_seed_tests.png
"""
import argparse, glob, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

T0 = time.time()
H, HD, C = 8, 16, 128
SEEDS_DIR = '/j-jepa-vol/aaron-interp/models/part_seeds'
P14S = '/j-jepa-vol/aaron-interp/models/p14/s/final.pt'
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ statistics
def boot_W(n, nb, rng):
    W = np.zeros((nb, n), np.float32)
    for b in range(nb): W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def q95(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.quantile(a, .025)), float(np.quantile(a, .975))] if len(a) else [None, None]


def p_two(bs, v0=0.0):
    bs = np.asarray(bs, float); bs = bs[np.isfinite(bs)]; nb = len(bs)
    if nb == 0: return None
    return float(max(1.0 / nb, min(1.0, 2 * min((bs <= v0).mean(), (bs >= v0).mean()))))


def ratio_stat(Wb, num, den):
    """ratio of means num/den over the same jets, bootstrap CI and p (ratio = 0)"""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    r = float(num.mean() / den.mean()); rb = (Wb @ num) / (Wb @ den)
    return dict(v=r, ci=q95(rb), p0=p_two(rb), _b=rb)


def mean_stat(Wb, x):
    x = np.asarray(x, np.float64); mb = (Wb @ x) / Wb.sum(1)
    return dict(v=float(x.mean()), ci=q95(mb), p0=p_two(mb))


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]; run = min(run, ps[i] * m / (r + 1)); q[i] = run
    return q.tolist()


def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o


# ------------------------------------------------------------------------------------------------ models
def make_net(feat, name, path):
    from interp.part_core import ParTNetwork
    net = ParTNetwork(feat); torch = net.torch
    strip_mod = lambda sd: {k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}
    if path is None:
        pass
    elif path == 'init0':
        from weaver.nn.model.ParticleTransformer import ParticleTransformer
        from jetdistill.part.network import CFG, FEATURES
        torch.manual_seed(0); fresh = ParticleTransformer(input_dim=len(FEATURES[feat]), **CFG)
        net.model.load_state_dict(fresh.state_dict(), strict=True)
    elif name == 'shuffled':
        net.model.fc[0] = torch.nn.Linear(C, 2)
        net.model.load_state_dict(strip_mod(torch.load(path, map_location='cpu')), strict=True)
    else:
        net.model.load_state_dict(strip_mod(torch.load(path, map_location='cpu')), strict=True)
    net.model.to(net.device).eval()
    for p in net.model.parameters(): p.requires_grad_(False)
    W = net.model.fc[0].weight.detach().cpu().double().numpy(); b = net.model.fc[0].bias.detach().cpu().double().numpy()
    net.last = (W.T.copy(), b.copy(), None, None); net.K = W.shape[0]; net.path = path
    return net


def fwd(net, x, v, m, batch=512):
    torch = net.torch; N = len(x); order = np.argsort(m[:, 0].sum(1), kind='stable')
    Z = np.zeros((N, C), np.float32); L = np.zeros((N, net.K), np.float32)
    dev = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(net.device)
    with torch.no_grad():
        for i in range(0, N, batch):
            idx = order[i:i + batch]; P = int(m[idx, 0].sum(1).max())
            out = net.model(dev(x[idx, :, :P]), dev(v[idx, :, :P]), dev(m[idx, :, :P]))
            L[idx] = out.float().cpu().numpy(); Z[idx] = net._cls['h'].float().cpu().numpy()
    return Z.astype(np.float64), L.astype(np.float64)


class St:
    """class-attention stage re-implemented (copied from interp/p3_wz_circuit_part.Stage, with the network passed in) plus
    per-head particle-block patching ('record' / list of (block, head)) and a per-head attention statistic ('stat')"""

    def __init__(self, net, feat):
        import torch, torch.nn.functional as F
        self.t, self.F, self.name, self.net = torch, F, feat, net
        self.m = net.model; self.dev = net.device
        g = lambda p: p.detach().float()
        ln = lambda l: (g(l.weight), g(l.bias), l.eps)
        self.cls0 = g(self.m.cls_token).reshape(C); self.B = []
        for blk in self.m.cls_blocks:
            a = blk.attn; Wi, bi = g(a.in_proj_weight), g(a.in_proj_bias)
            self.B.append(dict(ln_pre=ln(blk.pre_attn_norm), Wq=Wi[:C], bq=bi[:C], Wk=Wi[C:2 * C], bk=bi[C:2 * C], Wv=Wi[2 * C:], bv=bi[2 * C:],
                               Wo=g(a.out_proj.weight), bo=g(a.out_proj.bias), c=g(blk.c_attn), ln_pa=ln(blk.post_attn_norm), ln_pf=ln(blk.pre_fc_norm),
                               W1=g(blk.fc1.weight), b1=g(blk.fc1.bias), ln_po=ln(blk.post_fc_norm), W2=g(blk.fc2.weight), b2=g(blk.fc2.bias), wr=g(blk.w_resid)))
        self.lnf = ln(self.m.norm); self.Wfc, self.bfc = g(self.m.fc[0].weight), g(self.m.fc[0].bias)
        self.cap = {}; self.hk = [self.m.cls_blocks[0].register_forward_pre_hook(lambda mod, args: self.cap.__setitem__('X', args[0]))]
        self.pm, self.rec, self.recon, self.stat = None, {}, 0.0, None
        self.orig = []
        for i, blk in enumerate(self.m.blocks): self._wrap(i, blk.attn)

    def remove(self):
        for h in self.hk: h.remove()
        for blk, o in zip(self.m.blocks, self.orig): blk.attn.forward = o

    def _wrap(self, i, attn):
        orig, S, t = attn.forward, self, self.t; self.orig.append(orig)

        def f(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
            if S.pm is None and S.stat is None:
                return orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask,
                            average_attn_weights=average_attn_weights)
            out, w = orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
            if S.stat is not None:                                  # w: (n, H, P, P)
                num = (w * S.stat['dR'][:, None]).sum(-1)            # (n, H, P)
                qm = S.stat['qm']; s = (num * qm[:, None]).sum(-1) / qm.sum(-1)[:, None] / S.stat['dRm'][:, None]
                S.stat['out'][i] = s
            if S.pm is not None:
                P, n, _ = value.shape
                v = (value @ attn.in_proj_weight[2 * C:].T + attn.in_proj_bias[2 * C:]).view(P, n, H, HD).permute(1, 2, 0, 3)
                oh = t.einsum('nhpd,chd->nhpc', w @ v, attn.out_proj.weight.view(C, H, HD))
                if S.pm == 'record':
                    S.rec[i] = oh
                    S.recon = max(S.recon, float((oh.sum(1) + attn.out_proj.bias - out.permute(1, 0, 2)).abs().max()))
                else:
                    hs = [h for (b, h) in S.pm if b == i]
                    if hs: out = out + (S.rec[i][:, hs] - oh[:, hs]).sum(1).permute(1, 0, 2)
            return out, None
        attn.forward = f

    def run_X(self, J):
        from interp.part_core import inputs
        t = self.t; x, v, m = inputs(J, self.name)
        with t.no_grad():
            L = self.m(*(t.from_numpy(a).to(self.dev) for a in (x, v, m)))
        X = self.cap['X'].permute(1, 0, 2).contiguous().float(); pad = t.from_numpy(m[:, 0] == 0).to(self.dev)
        return X, pad, L.float()

    def cls(self, X, pad, ctl=None, rec=False):
        t, F = self.t, self.F; ctl = ctl or {}; n, P, _ = X.shape
        r = self.cls0.expand(n, C)
        padf = t.cat([t.zeros(n, 1, dtype=t.bool, device=X.device), pad], 1)
        R = dict(A=[], ctx=[])
        for b, B in enumerate(self.B):
            u = t.cat([r[:, None], X], 1); u = F.layer_norm(u, (C,), B['ln_pre'][0], B['ln_pre'][1], B['ln_pre'][2])
            K = (u @ B['Wk'].T + B['bk']).view(n, P + 1, H, HD); V = (u @ B['Wv'].T + B['bv']).view(n, P + 1, H, HD)
            q = (r @ B['Wq'].T + B['bq']).view(n, H, HD) / math.sqrt(HD)
            A = t.softmax(t.einsum('nhd,nphd->nhp', q, K).masked_fill(padf[:, None, :], float('-inf')), -1)
            ctx = t.einsum('nhp,nphd->nhd', A, V)
            if ('ctx', b) in ctl:
                Co, sel = ctl[('ctx', b)]; ctx = t.where(sel[None, :, None], Co, ctx)
            o = ctx.reshape(n, C) @ B['Wo'].T + B['bo']
            o = (o.view(n, H, HD) * B['c'][None, :, None]).permute(0, 2, 1).reshape(n, C)
            x1 = F.layer_norm(o, (C,), *B['ln_pa'][:2], B['ln_pa'][2]) + r
            hh = F.gelu(F.layer_norm(x1, (C,), *B['ln_pf'][:2], B['ln_pf'][2]) @ B['W1'].T + B['b1'])
            hh = F.layer_norm(hh, (hh.shape[-1],), *B['ln_po'][:2], B['ln_po'][2])
            r = hh @ B['W2'].T + B['b2'] + B['wr'] * x1
            if rec: R['A'].append(A); R['ctx'].append(ctx)
        z = F.layer_norm(r, (C,), *self.lnf[:2], self.lnf[2])
        return z, z @ self.Wfc.T + self.bfc, R


def mbatches(J, idx, size):
    order = idx[np.argsort(J['mask'][idx].sum(1), kind='stable')]
    return [order[i:i + size] for i in range(0, len(order), size)]


# ------------------------------------------------------------------------------------------------ data
def class_files():
    from interp.part_core import fof, CL
    for _ in range(6):               # files.json can be rewritten in place by a concurrent ./sync.sh (P10)
        try:
            return {c: fof(c) for c in CL}
        except json.JSONDecodeError:
            time.sleep(10)
    raise RuntimeError('files.json unreadable')


def wz_sample(per, seed, F):
    import uproot
    from interp.part_core import read_root, cat
    rng = np.random.default_rng(seed); Js, rows = [], {}
    for c in ('Wqq', 'Zqq'):
        f = F[c]; n = uproot.open(f)['tree'].num_entries
        cand = np.setdiff1d(np.arange(n), np.arange(3000, 4000)); r = np.sort(rng.choice(cand, per, replace=False))
        J = read_root(f); Js.append({k: (v[r] if np.ndim(v) else v) for k, v in J.items()}); del J
        rows[c] = dict(file=str(f), n_entries=int(n), first=r[:5].tolist()); log(f'WZ sample {c}: {per} random rows of {n} ({f})')
    return cat(Js), rows


# ------------------------------------------------------------------------------------------------ S4 fit
M50G = np.arange(40.0, 160.01, 0.5); WG = np.geomspace(0.5, 80.0, 48)
CENT = 52.5 + 5.0 * np.arange(18)


def grid_fit(Y, wt):
    """Y (M, 18) binned means, wt (18,) bin weights -> m50, w, a, c, per row (best weighted LS over the grid)"""
    sw = np.sqrt(wt); cands = [(m5, w) for m5 in M50G for w in WG]
    S = np.array([1.0 / (1.0 + np.exp(np.clip((CENT - m5) / w, -60, 60))) for m5, w in cands])          # (Cn, 18)
    Qs = []
    for s in S:
        A = np.stack([np.ones(18), s], 1) * sw[:, None]; q, _ = np.linalg.qr(A); Qs.append(q)
    Q = np.concatenate(Qs, 1).astype(np.float32)                    # (18, 2 Cn)
    Yw = (Y * sw[None]).astype(np.float32); best = np.zeros(len(Y), int)
    for i in range(0, len(Y), 500):
        Pj = (Yw[i:i + 500] @ Q) ** 2; Pj = Pj.reshape(len(Pj), -1, 2).sum(-1); best[i:i + 500] = Pj.argmax(1)
    s = S[best]                                                      # (M, 18)
    w = wt[None]; S0, S1, S2 = w.sum(1), (w * s).sum(1), (w * s * s).sum(1); T0_, T1 = (w * Y).sum(1), (w * s * Y).sum(1)
    det = S0 * S2 - S1 ** 2; det = np.where(np.abs(det) < 1e-12, np.nan, det)
    a = (S2 * T0_ - S1 * T1) / det; c = (S0 * T1 - S1 * T0_) / det
    m50 = np.array([cands[k][0] for k in best]); wd = np.array([cands[k][1] for k in best])
    return m50, wd, a, c


def is_switch(m50, w, a, c, strong=False):
    """pre-registered: m50 in 60-110 GeV, w < 15 GeV (plus |c| >= 0.2 max(|a|, |a + c|) so that a flat curve does not count);
    strong: in addition |c| >= 0.2 in the statistic's own units (P3's 20-point span rule for the class-token share)"""
    rel = np.abs(c) / np.maximum(np.maximum(np.abs(a), np.abs(a + c)), 1e-12)
    ok = (m50 >= 60) & (m50 <= 110) & (w < 15) & (rel >= 0.2)
    return ok & (np.abs(c) >= 0.2) if strong else ok


def switch_scan(S, mass, rng, NB, NPERM, labels):
    inb = (mass >= 50) & (mass < 140); Ms, Ss = mass[inb], S[inb]; bi = ((Ms - 50) // 5).astype(int); n = len(Ms)
    Bm = np.zeros((n, 18), np.float32); Bm[np.arange(n), bi] = 1
    cnt = Bm.sum(0); ok = cnt >= 50

    def bmeans(Wt):                                                  # Wt (B, n) -> (B, 18, heads)
        return np.einsum('bn,nk,nh->bkh', Wt, Bm, Ss.astype(np.float32), optimize=True) / np.maximum(Wt @ Bm, 1)[:, :, None]
    Y0 = (Bm.T @ Ss) / np.maximum(cnt, 1)[:, None]                   # (18, heads)
    m50, w, a, c = grid_fit(Y0.T, cnt)
    Wb = boot_W(n, NB, rng); Yb = np.concatenate([bmeans(Wb[i:i + 20]) for i in range(0, NB, 20)])   # (NB, 18, h)
    nh = S.shape[1]; mb, wb, ab, cb = grid_fit(Yb.transpose(0, 2, 1).reshape(-1, 18), cnt)
    sw_b = is_switch(mb, wb, ab, cb).reshape(NB, nh); swS_b = is_switch(mb, wb, ab, cb, True).reshape(NB, nh)
    span0 = Y0[ok].max(0) - Y0[ok].min(0)
    BmT = Bm.T.astype(np.float32); S32 = Ss.astype(np.float32); ge = np.zeros(nh)
    for k in range(NPERM):
        pm = rng.permutation(n); Yp = (BmT @ S32[pm]) / np.maximum(cnt, 1)[:, None]; ge += (Yp[ok].max(0) - Yp[ok].min(0)) >= span0
    p_perm = (1 + ge) / (1 + NPERM)                                  # H0: statistic independent of mass (span of binned means)
    sw0 = is_switch(m50, w, a, c); swS = is_switch(m50, w, a, c, True); q = bh(p_perm)
    stab = sw_b.mean(0) >= 0.95; stabS = swS_b.mean(0) >= 0.95
    out = {}
    for j, lb in enumerate(labels):
        out[lb] = dict(m50=float(m50[j]), w=float(w[j]), lo_mass_value=float(a[j] + c[j]), hi_mass_value=float(a[j]),
                       m50_ci=q95(mb.reshape(NB, nh)[:, j]), w_ci=q95(wb.reshape(NB, nh)[:, j]), span=float(span0[j]),
                       amp=float(c[j]), switch=bool(sw0[j]), frac_boot_switch=float(sw_b[:, j].mean()), switch_strong=bool(swS[j]),
                       frac_boot_switch_strong=float(swS_b[:, j].mean()), p_perm=float(p_perm[j]), q_bh=float(q[j]),
                       sig=bool(sw0[j] and stab[j] and q[j] < 0.05), sig_strong=bool(swS[j] and stabS[j] and q[j] < 0.05))
    return out, dict(n_in_window=int(n), bins_ge50=int(ok.sum()))


# ------------------------------------------------------------------------------------------------ per model
def analyse(name, net, feat, D, args, rng):
    from interp.part_core import inputs, sub, CL
    iW, iZ = CL.index('Wqq'), CL.index('Zqq'); ten = net.K == 10; t1 = time.time()
    res = dict(path=net.path, n_outputs=net.K)
    # ---- P1 sample: accuracy, confusion, read-out fit
    z1, L1 = fwd(net, *D['X1'])
    y1 = D['y1']
    if ten:
        c1 = L1.argmax(1); conf = np.zeros((10, 10))
        for k in range(10):
            sel = y1 == k
            if sel.any(): conf[k] = np.bincount(c1[sel], minlength=10) / sel.sum()
        res['p1_acc'] = float((c1 == y1).mean()); res['p1_acc_ci'] = q95([(c1 == y1)[rng.integers(0, len(y1), len(y1))].mean() for _ in range(500)])
        res['p1_confusion'] = conf.round(4).tolist(); res['p1_own_class'] = np.diag(conf).round(4).tolist()
        log(f'  [{name}] P1 sample 10-class accuracy {100 * res["p1_acc"]:.2f} % {np.round(100 * np.array(res["p1_acc_ci"]), 2).tolist()}; '
            f'own-class {np.round(100 * np.diag(conf), 1).tolist()}')
    perm = np.random.default_rng(11).permutation(len(y1)); fit, val = np.sort(perm[:len(perm) * 3 // 4]), np.sort(perm[len(perm) * 3 // 4:])
    from interp.p1v_verify_part import Ridge, r2
    rdg = Ridge(z1, D['logm1'], fit, val, np.arange(len(y1)))
    rdg_fitonly = Ridge(z1, D['logm1'], fit, val, fit)
    res['readout'] = dict(alpha=float(rdg.alpha), r2_P1_val=r2(rdg_fitonly(z1[val]), D['logm1'][val]))
    # ---- WZ sample: edits and routes
    X = D['X']; xb, vb, mb = X['base']; y = D['y']; Wm = y == iW; N = len(y)
    def xmix(src, cols):
        x = xb.copy(); x[:, cols] = X[src][0][:, cols]; return x
    iDR, iDPHI = D['iDR'], D['iDPHI']
    variants = {'base': (xb, vb), 'phi': X['phi'][:2], 'match': X['match'][:2], 'k1.1': X['k1.1'][:2], 'rot': X['rot'][:2],
                'phi_pair': (xb, X['phi'][1]), 'phi_pp': (X['phi'][0], vb), 'phi_dR': (xmix('phi', [iDR]), vb),
                'phi_dphi': (xmix('phi', [iDPHI]), vb), 'phi_dphi_dR': (xmix('phi', [iDR, iDPHI]), vb),
                'k1.1_pair': (xb, X['k1.1'][1]), 'k1.1_pp': (X['k1.1'][0], vb), 'rot_pair': (xb, X['rot'][1])}
    Z, L = {}, {}
    for k, (x, v) in variants.items():
        Z[k], L[k] = fwd(net, x, v, mb)
    mhat = {k: rdg(Z[k]) for k in Z}; dm = {k: mhat[k] - mhat['base'] for k in Z if k != 'base'}
    res['readout']['r2_WZ'] = r2(mhat['base'], D['logm'])
    Wb, WbW = D['Wb'], D['WbW']
    S1 = dict(dlogm_hat={k: mean_stat(Wb, v) for k, v in dm.items()},
              ratio_match=ratio_stat(Wb, dm['phi'], dm['match']))
    pr = D['pred']
    num = (Wb @ dm['phi']) / (Wb @ pr['phi']); den = (Wb @ dm['k1.1']) / (Wb @ pr['k1.1'])
    S1['ratio_cal'] = dict(v=float((dm['phi'].mean() / pr['phi'].mean()) / (dm['k1.1'].mean() / pr['k1.1'].mean())), ci=q95(num / den))
    S1['ratio_match']['in_window_0.7_1.4'] = bool(0.7 <= S1['ratio_match']['v'] <= 1.4)
    S1['ratio_match']['ci_inside_window'] = bool(S1['ratio_match']['ci'][0] is not None and S1['ratio_match']['ci'][0] >= 0.7 and S1['ratio_match']['ci'][1] <= 1.4)
    S1['ratio_rot_over_match'] = ratio_stat(Wb, dm['rot'], dm['match'])
    res['S1'] = S1
    if ten:
        g = {k: L[k][:, iW] - L[k][:, iZ] for k in L}; dd = {k: (g[k] - g['base'])[Wm] for k in L if k != 'base'}
        xs, ys = dd['match'], dd['phi']
        Rb = (WbW @ (xs * ys)) / (WbW @ (xs * xs))
        res['R'] = dict(v=float((xs * ys).sum() / (xs * xs).sum()), ci=q95(Rb), p0=p_two(Rb))
        res['R_mean'] = ratio_stat(WbW, ys, xs)
        res['R_k11'] = ratio_stat(WbW, ys, dd['k1.1'])
        res['R_rot'] = ratio_stat(WbW, dd['rot'], xs)
        res['dd_on_W'] = {k: mean_stat(WbW, v) for k, v in dd.items()}
        c = {k: L[k].argmax(1) for k in L}
        a0 = c['base'][Wm] == iZ; res['WZ'] = dict(base=float(a0.mean())); res['calls_changed'] = {}
        res['acc_WZ_10class'] = float((c['base'] == y).mean())
        res['WZ_binary_acc'] = float((np.where(g['base'] > 0, iW, iZ) == y).mean())
        from interp.e19_common import mcnemar
        for k in ('phi', 'match', 'k1.1', 'rot', 'phi_pair', 'phi_pp', 'phi_dR', 'phi_dphi', 'phi_dphi_dR', 'rot_pair'):
            a1 = c[k][Wm] == iZ; d = a1.astype(float) - a0
            res['WZ'][k] = dict(after=float(a1.mean()), d=mean_stat(WbW, d), mcnemar=mcnemar(a0, a1))
            res['calls_changed'][k] = float((c[k] != c['base']).mean())
    # ---- S2: route shares
    S2 = {}
    for k in ('phi_pair', 'phi_pp', 'phi_dR', 'phi_dphi', 'phi_dphi_dR', 'rot_pair', 'rot'):
        S2[f'share_{k}'] = ratio_stat(Wb, dm[k], dm['phi'])
        if ten: S2[f'share_dd_{k}'] = ratio_stat(WbW, dd[k], dd['phi'])
    for k in ('k1.1_pair', 'k1.1_pp'):
        S2[f'share_{k}'] = ratio_stat(Wb, dm[k], dm['k1.1'])
        if ten: S2[f'share_dd_{k}'] = ratio_stat(WbW, dd[k], dd['k1.1'])
    key = 'share_phi_dR' if feat == 'full' else 'share_phi_dphi_dR'
    S2['prereg_key'] = key; S2['prereg_pass'] = bool(S2[key]['v'] > 0.8); S2['prereg_ci_above'] = bool(S2[key]['ci'][0] is not None and S2[key]['ci'][0] > 0.8)
    res['S2'] = S2
    log(f'  [{name}] read-out alpha {rdg.alpha:g}, R2 P1-val {res["readout"]["r2_P1_val"]:.3f}, WZ {res["readout"]["r2_WZ"]:.3f}; '
        f'dlogm_hat phi {S1["dlogm_hat"]["phi"]["v"]:+.4f} match {S1["dlogm_hat"]["match"]["v"]:+.4f} k1.1 {S1["dlogm_hat"]["k1.1"]["v"]:+.4f} rot {S1["dlogm_hat"]["rot"]["v"]:+.5f}; '
        f'S1 ratio_match {S1["ratio_match"]["v"]:+.3f} {np.round(S1["ratio_match"]["ci"], 3).tolist()} (cal {S1["ratio_cal"]["v"]:+.3f}); '
        f'S2 {key} {S2[key]["v"]:+.3f} {np.round(S2[key]["ci"], 3).tolist()}, pair {S2["share_phi_pair"]["v"]:+.3f}, rot_pair null {S2["share_rot_pair"]["v"]:+.4f}, '
        f'k1.1 pair {S2["share_k1.1_pair"]["v"]:+.3f}')
    if ten:
        log(f'  [{name}] W->Z phi {100 * res["WZ"]["base"]:.1f} -> {100 * res["WZ"]["phi"]["after"]:.1f} % (d {100 * res["WZ"]["phi"]["d"]["v"]:+.1f} '
            f'{np.round(100 * np.array(res["WZ"]["phi"]["d"]["ci"]), 1).tolist()}), match -> {100 * res["WZ"]["match"]["after"]:.1f}, k1.1 -> {100 * res["WZ"]["k1.1"]["after"]:.1f}, '
            f'rot -> {100 * res["WZ"]["rot"]["after"]:.1f}; R {res["R"]["v"]:+.3f} {np.round(res["R"]["ci"], 3).tolist()}, R_mean {res["R_mean"]["v"]:+.3f}; '
            f'WZ 10-class acc {100 * res["acc_WZ_10class"]:.2f} %, W/Z binary {100 * res["WZ_binary_acc"]:.2f} % ({time.time() - t1:.0f} s)')
    del Z, L, variants
    # ---- S3 and S4 (re-implemented class stage)
    st = St(net, feat)
    try:
        res['S4'] = s4(st, D, args, rng, name)
        if ten: res['S3'] = s3(st, D, args, rng, name, z1, iW, iZ)
    finally:
        st.remove()
    log(f'  [{name}] done ({time.time() - t1:.0f} s)')
    return res


def s4(st, D, args, rng, name):
    from interp.part_core import sub
    t = st.t; Jb = D['Jb']; N = len(Jb['mask']); Sx = np.zeros((N, 16 + 64), np.float32); md = 0.0
    for idx in mbatches(Jb, np.arange(N), 256):
        Jx = sub(Jb, idx); P = int(Jx['mask'].sum(1).max())
        de, dp = (t.from_numpy(Jx[k][:, :P].astype(np.float32)).to(st.dev) for k in ('deta', 'dphi'))
        qm = t.from_numpy((Jx['mask'][:, :P] > 0).astype(np.float32)).to(st.dev)
        dR = t.sqrt((de[:, :, None] - de[:, None, :]) ** 2 + (dp[:, :, None] - dp[:, None, :]) ** 2) * qm[:, :, None] * qm[:, None, :]
        npair = (qm.sum(1) * (qm.sum(1) - 1)).clamp_min(1)
        st.stat = dict(dR=dR, qm=qm, dRm=(dR.sum((1, 2)) / npair).clamp_min(1e-6), out={})
        X, pad, Lm = st.run_X(Jx); st.stat_out = st.stat['out']; st.stat = None
        with t.no_grad():
            z, L, R = st.cls(X, pad, rec=True)
        md = max(md, float((L - Lm).abs().max()))
        Sx[idx, :16] = t.cat([R['A'][b][:, :, 0] for b in range(2)], 1).cpu().numpy()
        Sx[idx, 16:] = t.cat([st.stat_out[i] for i in range(8)], 1).cpu().numpy()
    labels = [f'c{b + 1}h{h + 1}' for b in range(2) for h in range(H)] + [f'p{i + 1}h{h + 1}' for i in range(8) for h in range(H)]
    out, info = switch_scan(Sx.astype(np.float64), D['mass'], rng, args.nb4, args.nperm, labels)
    sig = [k for k, v in out.items() if v['sig']]; sigS = [k for k, v in out.items() if v['sig_strong']]
    best = sorted([k for k in out if out[k]['sig']], key=lambda k: -abs(out[k]['amp']))[:5]
    log(f'  [{name}] S4: reimpl max |dlogit| {md:.2e}; {len(sig)} switch heads (m50 60-110, w < 15, >= 95 % of bootstraps, BH q < 0.05): {sig}; '
        f'with |amplitude| >= 0.2: {sigS}; largest amplitude: '
        + '; '.join(f'{k} m50 {out[k]["m50"]:.1f} w {out[k]["w"]:.1f} {out[k]["lo_mass_value"]:.3f}->{out[k]["hi_mass_value"]:.3f} q {out[k]["q_bh"]:.4f}' for k in best))
    return dict(heads=out, info=info, reimpl_max_dlogit=md, switch_heads=sig, switch_heads_strong=sigS, switch_class_heads=[k for k in sig if k.startswith('c')],
                switch_particle_heads=[k for k in sig if k.startswith('p')], top5_by_amplitude=best)


def s3(st, D, args, rng, name, z1, iW, iZ):
    from interp.part_core import sub
    t = st.t; Wl = st.Wfc.cpu().double().numpy().T          # (128, 10)
    sd1 = z1.std(0); score = np.abs(Wl[:, iW] - Wl[:, iZ]) * sd1; u = int(np.argmax(score)); sdu = float(sd1[u])
    top3 = np.argsort(-score)[:3].tolist()
    Jb, Jk = D['Jb'], D['Jk']; idx3 = D['idx3']; y3 = D['y'][idx3]; n3 = len(idx3); pos = {j: i for i, j in enumerate(idx3)}
    lab_c = [(b, h) for b in range(2) for h in range(H)]; lab_p = [(i, h) for i in range(8) for h in range(H)]
    rr = np.random.default_rng(args.seed + 3)
    rand_c = [[(0, h) for h in rr.choice(H, 4, replace=False)] + [(1, h) for h in rr.choice(H, 4, replace=False)] for _ in range(args.nrand_c)]
    rand_p = [[(i, int(rr.integers(0, H))) for i in range(8)] for _ in range(args.nrand_p)]
    cfg = [('cls', 'cb1', [(0, h) for h in range(H)]), ('cls', 'cb2', [(1, h) for h in range(H)])] + \
          [('cls', f'c{b + 1}h{h + 1}', [(b, h)]) for b, h in lab_c] + [('cls', f'randc{k}', s) for k, s in enumerate(rand_c)] + \
          [('clsres', 'cb1_resample', [(0, h) for h in range(H)])] + \
          [('part', 'pb1', [(0, h) for h in range(H)]), ('part', 'pall', lab_p)] + [('part', f'p{i + 1}h{h + 1}', [(i, h)]) for i, h in lab_p] + \
          [('part', f'randp{k}', s) for k, s in enumerate(rand_p)]
    if args.smoke: cfg = [c for c in cfg if not (c[0] == 'part' and c[1][0] == 'p' and c[1] not in ('pb1', 'pall') and c[1] not in ('p1h1', 'p2h3'))]
    dE = np.zeros(n3); dEw = np.zeros(n3); dP = {c[1]: np.zeros(n3) for c in cfg}; dPw = {c[1]: np.zeros(n3) for c in cfg}
    sel = lambda hs, b: t.tensor([(b, h) in hs for h in range(H)], device=st.dev)
    gw = lambda L: (L[:, iW] - L[:, iZ])
    for bidx in mbatches(Jb, idx3, args.s3batch):
        ii = np.array([pos[j] for j in bidx]); Jbb, Jkb = sub(Jb, bidx), sub(Jk, bidx)
        with t.no_grad():
            st.pm, st.rec = 'record', {}; X0, pad, _ = st.run_X(Jbb); rec0 = st.rec; st.pm = None
            Xe, pad_e, _ = st.run_X(Jkb); assert bool((pad == pad_e).all())
            z0, L0, R0 = st.cls(X0, pad, rec=True); ze, Le, _ = st.cls(Xe, pad)
            ctx0 = t.stack(R0['ctx'], 1)
            base_u, base_g = z0[:, u], gw(L0)
            dE[ii] = ((ze[:, u] - base_u) / sdu).cpu().numpy(); dEw[ii] = (gw(Le) - base_g).cpu().numpy()
            don = t.from_numpy(np.roll(rr.permutation(len(bidx)), 1)).to(st.dev)
            for kind, nm, hs in cfg:
                if kind == 'cls':
                    ctl = {('ctx', b): (ctx0[:, b], sel(hs, b)) for b in range(2) if any(bb == b for bb, _ in hs)}
                    z, L, _ = st.cls(Xe, pad, ctl); zb_, gb_ = base_u, base_g
                elif kind == 'clsres':
                    ctl = {('ctx', 0): (ctx0[don, 0], sel(hs, 0))}
                    zr0, Lr0, _ = st.cls(X0, pad, ctl); z, L, _ = st.cls(Xe, pad, ctl); zb_, gb_ = zr0[:, u], gw(Lr0)
                else:
                    st.pm, st.rec = hs, rec0; Xp, _, _ = st.run_X(Jkb); st.pm = None
                    z, L, _ = st.cls(Xp, pad); zb_, gb_ = base_u, base_g
                dP[nm][ii] = ((z[:, u] - zb_) / sdu).cpu().numpy(); dPw[nm][ii] = (gw(L) - gb_).cpu().numpy()
        st.rec = {}
    Wb = boot_W(n3, args.nb, rng)
    out = dict(u=u, u_top3=top3, sd_u=sdu, score_u=float(score[u]), w_W_minus_Z_times_sd=float((Wl[u, iW] - Wl[u, iZ]) * sdu),
               n=n3, record_recon_max=float(st.recon), dE=mean_stat(Wb, dE), dE_W=float(dE[y3 == iW].mean()), dE_Z=float(dE[y3 == iZ].mean()),
               dEw=mean_stat(Wb, dEw), sets={})
    for kind, nm, hs in cfg:
        r1 = ratio_stat(Wb, dP[nm], dE); r2_ = ratio_stat(Wb, dPw[nm], dEw); rp = ratio_stat(Wb, dP[nm] * dE, dE * dE)
        rW = ratio_stat(Wb[:, y3 == iW], dP[nm][y3 == iW], dE[y3 == iW]); rZ = ratio_stat(Wb[:, y3 == iZ], dP[nm][y3 == iZ], dE[y3 == iZ])
        out['sets'][nm] = dict(kind=kind, heads=[f'{"c" if kind != "part" else "p"}{b + 1}h{h + 1}' for b, h in hs] if len(hs) <= 8 else f'{len(hs)} heads',
                               removed=1 - r1['v'], removed_ci=[1 - r1['ci'][1], 1 - r1['ci'][0]], p0=p_two(1 - rp['_b']),
                               removed_wz=1 - r2_['v'], removed_wz_ci=[1 - r2_['ci'][1], 1 - r2_['ci'][0]],
                               removed_proj=1 - rp['v'], removed_proj_ci=[1 - rp['ci'][1], 1 - rp['ci'][0]],
                               removed_W=1 - rW['v'], removed_Z=1 - rZ['v'])
    S = out['sets']
    for fam, pre in (('class', 'c'), ('particle', 'p')):
        names = [k for k in S if k[0] == pre and 'h' in k and k[1].isdigit()]
        if names:
            for k, pa in zip(names, holm_list([S[k]['p0'] for k in names])): S[k]['p0_holm'] = pa
            best = max(names, key=lambda k: S[k]['removed_proj']); out[f'top_{fam}_head'] = best
            out[f'top3_{fam}_heads'] = sorted(names, key=lambda k: -S[k]['removed_proj'])[:3]
            out[f'n_holm_sig_{fam}'] = int(sum(S[k]['p0_holm'] < 0.05 for k in names))
    rc = [S[k]['removed_proj'] for k in S if k.startswith('randc')]; rp = [S[k]['removed_proj'] for k in S if k.startswith('randp')]
    out['cb1_vs_random'] = dict(random_mean=float(np.mean(rc)), random_max=float(np.max(rc)), p=(1 + sum(v >= S['cb1']['removed_proj'] for v in rc)) / (1 + len(rc)))
    out['pb1_vs_random'] = dict(random_mean=float(np.mean(rp)), random_max=float(np.max(rp)), p=(1 + sum(v >= S['pb1']['removed_proj'] for v in rp)) / (1 + len(rp)))
    out['prereg_cb1_gt_0.9'] = bool(S['cb1']['removed_proj'] > 0.9); out['prereg_pb1_gt_0.9'] = bool(S['pb1']['removed_proj'] > 0.9)
    log(f'  [{name}] S3: u* = {u} (top3 {top3}; (W_W - W_Z) sd {out["w_W_minus_Z_times_sd"]:+.2f}), k1.1 dz_u/sd {out["dE"]["v"]:+.3f} (W {out["dE_W"]:+.3f} Z {out["dE_Z"]:+.3f}); '
        f'removed: class block 1 {S["cb1"]["removed"]:.3f} {np.round(S["cb1"]["removed_ci"], 3).tolist()} (random 4+4 sets mean {out["cb1_vs_random"]["random_mean"]:.3f} max {out["cb1_vs_random"]["random_max"]:.3f}), '
        f'resample {S["cb1_resample"]["removed"]:.3f}, class block 2 {S["cb2"]["removed"]:.3f}; particle block 1 {S["pb1"]["removed"]:.3f} {np.round(S["pb1"]["removed_ci"], 3).tolist()} '
        f'(random stratified mean {out["pb1_vs_random"]["random_mean"]:.3f} max {out["pb1_vs_random"]["random_max"]:.3f}), all 64 {S["pall"]["removed"]:.3f}; '
        f'per-jet projection: cb1 {S["cb1"]["removed_proj"]:.3f} {np.round(S["cb1"]["removed_proj_ci"], 3).tolist()}, pb1 {S["pb1"]["removed_proj"]:.3f}, W/Z logit: cb1 {S["cb1"]["removed_wz"]:.3f}, pb1 {S["pb1"]["removed_wz"]:.3f}; '
        f'top class head {out.get("top_class_head")} {S[out["top_class_head"]]["removed_proj"]:.3f} ({out.get("n_holm_sig_class")} Holm-sig), top particle head {out.get("top_particle_head")} '
        f'{S[out["top_particle_head"]]["removed_proj"]:.3f} ({out.get("n_holm_sig_particle")} Holm-sig); record recon {st.recon:.1e}')
    return out


def holm_list(ps):
    from interp.e19_common import holm
    return holm(list(ps))


# ------------------------------------------------------------------------------------------------ main
def prepare(feat, args, F):
    from interp.part_core import read_root, cat, sub, kin, build, jmass, moments, inputs, CL
    from jetdistill.part.network import FEATURES
    names = [f[0] for f in FEATURES[feat]]
    per1, start = (60, 3000) if args.smoke else (1000, 3000)
    J1 = cat([read_root(F[c], start=start, stop=start + per1) for c in CL])
    p1, d1, f1, s1 = kin(J1); Jb1 = build(J1, p1, d1, f1, s1); m1 = jmass(Jb1)
    log(f'P1 sample: {len(m1)} jets (rows {start}-{start + per1} of each class file)')
    J0, rows = wz_sample(args.nwz, 11, F)
    pt, de, dp, ms = kin(J0); M = moments(pt, de, dp); valid = ((pt > 0).sum(1) >= 2) & (M['M2'] > 0)
    J0 = sub(J0, valid); pt, de, dp, ms = kin(J0); real = pt > 0
    Jb = build(J0, pt, de, dp, ms); M0 = moments(*kin(Jb)[:3]); m0 = jmass(Jb)
    Jphi = build(J0, pt, de, dp + 0.06 * real, ms)
    pred_phi = 0.5 * np.log(moments(*kin(Jphi)[:3])['M0M2'] / M0['M0M2']); kj = np.exp(pred_phi)
    Jm = build(J0, pt, de * kj[:, None], dp * kj[:, None], ms); Jk = build(J0, pt, de * 1.1, dp * 1.1, ms)
    Jr = dict(Jb); Jr['jet_phi'] = (Jb['jet_phi'] + 0.06).astype(np.float32); c_, s_ = np.cos(0.06), np.sin(0.06)
    Jr['px'] = (c_ * Jb['px'] - s_ * Jb['py']).astype(np.float32); Jr['py'] = (s_ * Jb['px'] + c_ * Jb['py']).astype(np.float32)
    E = {'phi': Jphi, 'match': Jm, 'k1.1': Jk, 'rot': Jr}
    pred = {k: 0.5 * np.log(moments(*kin(Je)[:3])['M0M2'] / M0['M0M2']) for k, Je in E.items()}
    dlogm = {k: np.log(jmass(Je)) - np.log(m0) for k, Je in E.items()}
    y = J0['label'].astype(int)
    info = dict(n_dropped_lt2=int((~valid).sum()), N=int(len(y)), rows=rows, mass_median=float(np.median(m0)),
                edits={k: dict(pred_mean=float(pred[k].mean()), dlogm_mean=float(dlogm[k].mean()), dlogm_absmax=float(np.abs(dlogm[k]).max()))
                       for k in E}, match_minus_phi_pred_absmax=float(np.abs(pred['match'] - pred['phi']).max()),
                k_match_mean=float(kj.mean()), k_match_q=[float(np.quantile(kj, q)) for q in (0.05, 0.5, 0.95)])
    log(f'WZ sample: {len(y)} jets ({info["n_dropped_lt2"]} with < 2 particles dropped); mass median {np.median(m0):.1f} GeV; k_match mean '
        f'{kj.mean():.4f} (5/50/95 % {np.round(info["k_match_q"], 4).tolist()}); |pred(match) - pred(phi)| max {info["match_minus_phi_pred_absmax"]:.1e}; '
        + '; '.join(f'{k}: 1/2 dlogM0M2 {v["pred_mean"]:+.4f}, dlog m {v["dlogm_mean"]:+.5f} (max |.| {v["dlogm_absmax"]:.4f})' for k, v in info['edits'].items()))
    X = {'base': inputs(Jb, feat)}
    for k, Je in E.items():
        X[k] = inputs(Je, feat); assert np.array_equal(X[k][2], X['base'][2])
    rng = np.random.default_rng(args.seed)
    iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    n3 = args.n3; r3 = np.random.default_rng(12)
    idx3 = np.sort(np.concatenate([r3.choice(np.where(y == c)[0], n3, replace=False) for c in (iW, iZ)]))
    D = dict(X1=inputs(Jb1, feat), y1=J1['label'].astype(int), logm1=np.log(np.maximum(m1, 1.0)), X=X, y=y, Jb=Jb, Jk=Jk, mass=m0,
             logm=np.log(np.maximum(m0, 1.0)), pred=pred, idx3=idx3, iDR=names.index('deltaR'), iDPHI=names.index('dphi'),
             Wb=boot_W(len(y), args.nb, rng), WbW=boot_W(int((y == iW).sum()), args.nb, rng))
    return D, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', default='full'); ap.add_argument('--seeds', default='all'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seed', type=int, default=11); ap.add_argument('--plot', action='store_true'); ap.add_argument('--models', default=None)
    a = ap.parse_args()
    if a.plot: return plot()
    a.nwz = 300 if a.smoke else 10000; a.n3 = 60 if a.smoke else 2000; a.nb = 100 if a.smoke else 1000
    a.nb4 = 20 if a.smoke else 200; a.nperm = 20 if a.smoke else 4000; a.nrand_c = 4 if a.smoke else 30; a.nrand_p = 3 if a.smoke else 20
    a.s3batch = 128
    OUTD.mkdir(parents=True, exist_ok=True)
    import torch
    log(f'P11 feat {a.feat} seeds {a.seeds} smoke {a.smoke}; torch {torch.__version__}, device {os.environ.get("JETDISTILL_DEVICE")}, '
        f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda"}')
    F = class_files(); D, info = prepare(a.feat, a, F)
    seeds = sorted(int(d.rsplit('_s', 1)[1]) for d in glob.glob(f'{SEEDS_DIR}/{a.feat}_s*')) if a.seeds == 'all' else [int(s) for s in a.seeds.split(',')]
    models = [('released', None)] + [(f's{s}', f'{SEEDS_DIR}/{a.feat}_s{s}/net_best_epoch_state.pt') for s in seeds
                                     if os.path.exists(f'{SEEDS_DIR}/{a.feat}_s{s}/net_best_epoch_state.pt')] + [('untrained', 'init0')]
    if a.feat == 'kin' and os.path.exists(P14S): models.append(('shuffled', P14S))
    if a.models: models = [m for m in models if m[0] in a.models.split(',')]
    log(f'models: {[m[0] for m in models]}')
    res = dict(feat=a.feat, smoke=a.smoke, sample=info, models={}, settings=dict(nb=a.nb, nb4=a.nb4, nperm=a.nperm, n3_per_class=a.n3,
               nrand_c=a.nrand_c, nrand_p=a.nrand_p, seed=a.seed))
    fn = OUTD / f'p11_seed_tests_{a.feat}{"_smoke" if a.smoke else ""}.json'
    for name, path in models:
        rng = np.random.default_rng(a.seed + 100)
        net = make_net(a.feat, name, path); log(f'== {a.feat} {name}: {path}')
        res['models'][name] = analyse(name, net, a.feat, D, a, rng); del net
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        json.dump(strip(res), open(fn, 'w'), indent=1)
    # Holm over the W->Z McNemar tests of this job (models x edits)
    keys, ps = [], []
    for n, r in res['models'].items():
        for k, e in r.get('WZ', {}).items():
            if isinstance(e, dict): keys.append((n, k)); ps.append(e['mcnemar']['p'])
    for (n, k), p in zip(keys, holm_list(ps)): res['models'][n]['WZ'][k]['mcnemar']['p_holm'] = p
    # across-seed summary
    sd = [n for n in res['models'] if n[0] == 's' and n[1:].isdigit()]   # retrained seeds only (not 'shuffled')
    if sd:
        summ = {}
        for lab_, f in (('R', lambda r: r['R']['v']), ('dWZ_phi', lambda r: r['WZ']['phi']['d']['v']), ('S1_ratio_match', lambda r: r['S1']['ratio_match']['v']),
                        ('S2_share', lambda r: r['S2'][r['S2']['prereg_key']]['v']), ('S3_cb1', lambda r: r['S3']['sets']['cb1']['removed_proj']),
                        ('S3_pb1', lambda r: r['S3']['sets']['pb1']['removed_proj']), ('S3_cb1_wz', lambda r: r['S3']['sets']['cb1']['removed_wz']), ('p1_acc', lambda r: r['p1_acc'])):
            v = [f(res['models'][n]) for n in sd]; summ[lab_] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), values=v)
        summ['S3_top_class_head'] = {n: res['models'][n]['S3'].get('top_class_head') for n in sd}
        summ['S3_top_particle_head'] = {n: res['models'][n]['S3'].get('top_particle_head') for n in sd}
        summ['S4_switch_heads'] = {n: res['models'][n]['S4']['switch_heads'] for n in sd}
        summ['S4_switch_heads_strong'] = {n: res['models'][n]['S4']['switch_heads_strong'] for n in sd}
        res['seed_summary'] = summ
        log('seed summary: ' + '; '.join(f'{k} mean {v["mean"]:+.3f} range [{v["min"]:+.3f}, {v["max"]:+.3f}]' for k, v in summ.items() if isinstance(v, dict) and 'mean' in v))
        log(f'seed summary: top class head {summ["S3_top_class_head"]}, top particle head {summ["S3_top_particle_head"]}, S4 switch heads {summ["S4_switch_heads"]}')
    json.dump(strip(res), open(fn, 'w'), indent=1); log(f'wrote {fn}')
    try:
        plot()
    except Exception as ex:
        log(f'plot skipped: {ex!r}')
    log('P11 DONE')


def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = {f: json.load(open(OUTD / f'p11_seed_tests_{f}.json')) for f in ('full', 'kin') if (OUTD / f'p11_seed_tests_{f}.json').exists()}
    if len(R) < 2: raise RuntimeError('needs both p11_seed_tests_full.json and _kin.json')
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    panels = [('S1: log-mass read-out change,\nφ shift / matched dilation', lambda r: (r['S1']['ratio_match']['v'], r['S1']['ratio_match']['ci']), (0.7, 1.4), 'ratio (axis-moment model 1, invariant mass 0)'),
              ('R: W-vs-Z logit change,\nφ shift / matched dilation', lambda r: (r['R']['v'], r['R']['ci']) if 'R' in r and r['WZ_binary_acc'] > 0.6 else None, (1.0, None), 'slope over true-W jets (axis model 1, invariant mass 0)'),
              ('S3: W/Z unit response to angles×1.1 removed\nby patching the 8 heads of class-attention block 1', lambda r: (r['S3']['sets']['cb1']['removed_proj'], r['S3']['sets']['cb1']['removed_proj_ci']) if 'S3' in r and abs(r['S3']['dE']['v']) > 0.02 else None, (0.9, None), 'removed share (per-jet projection)')]
    cols = dict(released='#444444', s0='#3b6fb6', s1='#3b6fb6', s2='#3b6fb6', untrained='#bbbbbb', shuffled='#d08a2c')
    for ax, (title, f, win, yl) in zip(axs, panels):
        xt, xl, x = [], [], 0
        for feat in ('full', 'kin'):
            for n, r in R[feat]['models'].items():
                v = f(r)
                if v is not None:
                    ax.bar(x, v[0], 0.7, color=cols.get(n, '#3b6fb6'))
                    if v[1][0] is not None: ax.errorbar(x, v[0], yerr=[[v[0] - v[1][0]], [v[1][1] - v[0]]], color='k', lw=1, capsize=2)
                xt.append(x); xl.append(dict(released='rel.', untrained='untr.', shuffled='shuf.').get(n, n)); x += 1
            ax.text(x - 0.5 - len(R[feat]['models']) / 2, -0.16, f'ParT_{feat}', transform=ax.get_xaxis_transform(), ha='center', fontsize=8.5)
            x += 0.6
        if title.startswith('R:'): ax.axhline(1.0, color='#777', ls=':', lw=1)
        elif win[1] is None: ax.axhline(win[0], color='#b33', ls='--', lw=1)
        else: ax.axhspan(*win, color='#b33', alpha=0.08, zorder=0)
        ax.axhline(0, color='k', lw=0.6); ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7); ax.set_title(title, fontsize=9); ax.set_ylabel(yl, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Retrained ParT seeds (s0-s2, 2M training jets) keep the φ-shift response, with a smaller class-block-1 share than the released model (rel.)\n'
                 'grey: untrained network, orange: shuffled-label network; red: pre-registered threshold', fontsize=9.5)
    fig.text(0.5, -0.03, f'N = {R["full"]["sample"]["N"]} true W/Z test jets (S1, S2), 4,000 (S3); 95 % bootstrap CI over jets. φ shift: all particles +0.06 rad about the stored axis (invariant mass fixed); '
             'matched dilation: angles scaled to give the same axis-moment change.', ha='center', fontsize=7.5)
    fig.tight_layout(); p = OUTD / 'p11_seed_tests.png'; fig.savefig(p, dpi=130, bbox_inches='tight'); log(f'wrote {p}')


if __name__ == '__main__':
    main()
