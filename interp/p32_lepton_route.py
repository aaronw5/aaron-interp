"""P32 — lepton route in ParT and ParticleNet for H->lvqq (new claim C15 setup; loop C round C7, exploratory).

Question. Do ParT_full / ParticleNet_full identify H->WW*->lvqq' jets (class 'Hqql') mainly through one lepton-candidate token,
read by a small set of class-attention heads, with the call depending on the isElectron / isMuon flag and the lepton's
kinematics? ParT_kin (no PID inputs) must reach the class by another (kinematic) route.

Models: ParT_full, ParT_kinpid, ParT_kin (released, models/part), PN_full, PN_kinpid (released ParticleNet, models/particlenet).
kinpid input list = JetClass_kinpid.yaml (5 kinematic + charge + 5 PID flags + deta, dphi; no impact parameters).
Sample: JetClass test_20M HToWW2Q1L_115.root, NLV random rows, + NOTH random rows of each other class's <sample>_115.root
(numpy default_rng(3215); never first-k). Unedited 10-class accuracy and confusion per model on it; P1 sample (rows 3,000-4,000
of each class file _100, part_core.sample) as the reference check.
Build: every jet (unedited included) passes through part_core.build (eta = jet_eta + sign(jet_eta) deta).
Lepton candidate: highest-pT particle with isElectron or isMuon; jets with none are counted per class and left out of the edits.
Matched hadron (control token): a random charged hadron (isChargedHadron) whose pT is within a factor 1.5 of the lepton's, else
the charged hadron of nearest pT.
Edits on Hqql jets with a lepton (Delta log p(Hqql) per jet against the rebuilt unedited jet; calls changed; McNemar on Hqql calls):
  e1     lepton flag -> charged hadron (isElectron = isMuon = 0, isChargedHadron = 1); kinematics, charge, d0/dz kept.
  e2     control: matched charged hadron's flag -> the lepton's flag (same one-hot change, other sign, on a matched-pT token).
  e3     lepton removed; its four-vector added to the nearest charged hadron in (deta, dphi) (jet four-vector unchanged).
  e3c    control: matched hadron merged into its nearest other charged hadron.
  e4h/e4d  lepton pT x0.5 / x2 (angles and mass kept), every other particle's pT scaled so the scalar pT sum is unchanged
         (jets with lepton pT share >= 0.5 left out of x2);  e4hc/e4dc same on the matched hadron.
  e5mu / e5el  reverse on QCD, H->4q, t->bqq' jets of the sample: leading charged hadron's flag -> isMuon / isElectron;
         e5mu_r: a random non-leading charged hadron -> isMuon.  Metric: Delta log p(Hqql), calls moved to Hqql (and to t->blv).
Internals (ParT, on NKO random Hqql jets with lepton and matched hadron):
  attention: class-attention weight of the class token on the lepton key per head (b1h1..b2h8 = cls_blocks[0..1], heads 1-8)
    vs on the matched hadron; particle-block heads (pXhY = blocks[X-1], head Y-1): mean over real queries.
  knock-out: per head (64 particle + 16 class heads) the attention logit of the lepton key set to -1e9 (all queries), vs the same
    on the matched-hadron key (random-token control); H0: mean Delta log p(Hqql) under lepton KO = under matched KO; paired
    bootstrap p, Holm over 80 heads. Joint KOs: all 16 class heads, all 64 particle heads, all 80; the Holm-significant set S
    (lepton effect more negative) vs NRAND random head sets of the same size and stage composition and vs matched-token KO in S;
    greedy top-k (single-head rank) joint KO, k = 1..10.
  e1 head patching (class heads): each class head's output (before out_proj sum) taken from the e1 run -> share of e1's Delta.
  residual patching (e1 donor): the lepton token's residual at layer l (0 = embedding, l = output of particle block l) taken
    from the e1 run; complement: every other token at layer l from the e1 run. Share = Delta_patch / Delta_e1 (mean ratio).
  literal circuit (ParT_full, ParT_kinpid, on the whole 10-class sample): heads not in the circuit replaced by their mean output
    over the sample's real tokens (pool = all 10 classes); (a) class stage: only circuit class heads kept among the 16 class
    heads, particle blocks intact; (b) all heads: only circuit heads kept among all 80, MLPs intact. Circuit = S (if S has no
    class head: the 3 class heads with the largest single-head lepton KO effect, flagged). Hqql recall, 10-class accuracy,
    balanced accuracy, confusion; NRAND random head sets of the same composition; permutation p.
ParticleNet: e1-e5 as above; kNN neighbours of the lepton (k = 16) at EdgeConv 1-3 under e1 (fraction of neighbours changed),
  vs the matched hadron's under e2 and vs all tokens; e1 with the kNN graphs held at the unedited ones (share of e1 Delta
  carried by the graph change).
Statistics: 1,000 bootstrap resamples over jets (95 % percentile intervals), McNemar for paired call changes, Holm over heads.
  p0 = normal approximation with the bootstrap s.e. (resolves p < 1/NB for Holm over 80 heads); p0_boot = percentile p.
  Unedited base = rebuild(BL), so base and edited jets pass through part_core.build the same number of times (attempt 1 did not:
  ParT_kin showed a spurious -0.03 under e1).

  ./sync.sh && ./part_job.sh c-p32-run 'python -m interp.p32_lepton_route'
  -> $AI_RESULTS/part/p32_lepton_route[_smoke].json; --plot -> results/part/p32_lepton_route.png
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--plot', action='store_true')
ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED = 3215
NLV = 200 if SMOKE else 5000
NOTH = 30 if SMOKE else 556
NKO = 100 if SMOKE else 2000
NB = 100 if SMOKE else 1000
NRAND = 5 if SMOKE else 100
BS = 250
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TAG = '_smoke' if SMOKE else ''
FN = OUTD / f'p32_lepton_route{TAG}.json'
HQQL, TBL = 5, 9
C, H, HD = 128, 8, 16
FLAGS = ('isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon')
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ statistics
def boot_W(n, nb, rng):
    W = np.zeros((nb, n), np.float32)
    for b in range(nb): W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W / n


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def p_two(bs, v0=0.0):
    bs = np.asarray(bs, float); bs = bs[np.isfinite(bs)]; nb = len(bs)
    if nb == 0: return None
    return float(max(1.0 / nb, min(1.0, 2 * min((bs <= v0).mean(), (bs >= v0).mean()))))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def mstat(x, W):
    """mean of per-jet x with bootstrap CI and two-sided p against 0 (W: bootstrap weights rows, same jets)"""
    x = np.asarray(x, np.float64); bs = W.astype(np.float64) @ x; sd = float(bs.std())
    from scipy.stats import norm
    pn = float(2 * norm.sf(abs(x.mean()) / sd)) if sd > 0 else (1.0 if x.mean() == 0 else 0.0)
    # p0: normal approximation with the bootstrap standard error (resolves p < 1/NB, needed for Holm over 80 heads);
    # p0_boot: percentile-bootstrap p (floored at 1/NB)
    return dict(v=float(x.mean()), ci=ci(bs), se=sd, p0=pn, p0_boot=p_two(bs))


def mcnemar(a, b):
    """paired binary outcomes a (before), b (after): counts and exact two-sided p"""
    from scipy.stats import binomtest
    a = np.asarray(a, bool); b = np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
    p = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 > 0 else 1.0
    return dict(before=float(a.mean()), after=float(b.mean()), lost=n10, gained=n01, p=p)


def strip(o):
    if isinstance(o, dict): return {str(k): strip(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)): return o.item()
    if isinstance(o, np.ndarray): return strip(o.tolist())
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def lodds(L, k=HQQL):
    """log-odds of class k: logit_k - logsumexp(other logits) = log p_k - log(1 - p_k) (does not saturate near p = 1)"""
    L = np.asarray(L, np.float64); o = np.delete(L, k, 1); m = o.max(1)
    return L[:, k] - m - np.log(np.exp(o - m[:, None]).sum(1))


def logsm(L):
    L = np.asarray(L, np.float64); m = L.max(1, keepdims=True); return L - m - np.log(np.exp(L - m).sum(1, keepdims=True))


# ------------------------------------------------------------------------------------------------ data
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


def add_kinpid():
    from jetdistill.part import network as NW
    if 'kinpid' not in NW.FEATURES:
        NW.FEATURES['kinpid'] = NW.KIN + [(k, None, 1, None) for k in ('charge',) + FLAGS] + [('deta', None, 1, None), ('dphi', None, 1, None)]
    return NW


def rebuild(J):
    from interp.part_core import build, kin
    return build(J, *kin(J))


def lepton_idx(J):
    m = J['mask'] > 0; isl = ((J['isElectron'] > 0.5) | (J['isMuon'] > 0.5)) & m
    return np.where(isl.any(1), isl.argmax(1), -1)


def matched_hadron(J, tok, rng, exclude=None):
    from interp.part_core import kin
    pt = kin(J)[0]; ch = (J['isChargedHadron'] > 0.5) & (J['mask'] > 0); out = np.full(len(tok), -1); q = np.zeros(len(tok))
    for j in range(len(tok)):
        if tok[j] < 0: continue
        c = np.where(ch[j])[0]; c = c[c != tok[j]]
        if exclude is not None: c = c[c != exclude[j]]
        if not len(c): continue
        r = np.abs(np.log(pt[j, c] / pt[j, tok[j]])); ok = c[r < np.log(1.5)]
        out[j] = rng.choice(ok) if len(ok) else c[np.argmin(r)]; q[j] = np.abs(np.log(pt[j, out[j]] / pt[j, tok[j]]))
    return out, q


def nearest_charged(J, tok, exclude=None):
    ch = (J['isChargedHadron'] > 0.5) & (J['mask'] > 0); out = np.full(len(tok), -1)
    for j in range(len(tok)):
        if tok[j] < 0: continue
        c = np.where(ch[j])[0]; c = c[c != tok[j]]
        if exclude is not None: c = c[c != exclude[j]]
        if not len(c): continue
        d = np.hypot(J['deta'][j, c] - J['deta'][j, tok[j]], J['dphi'][j, c] - J['dphi'][j, tok[j]]); out[j] = c[np.argmin(d)]
    return out


def ed_flag(B, tok, flag):
    from interp.part_core import build, kin
    E = {k: (v.copy() if np.ndim(v) else v) for k, v in B.items()}; r = np.arange(len(tok))
    for k in FLAGS: E[k][r, tok] = 0
    E[flag][r, tok] = 1
    return build(E, *kin(E))


def ed_merge(B, src, dst):
    from interp.part_core import build, kin
    pt, de, dp, ms = (a.astype(np.float64) for a in kin(B)); r = np.arange(len(src))
    P4 = {k: B[k].astype(np.float64)[r, src] + B[k].astype(np.float64)[r, dst] for k in ('px', 'py', 'pz', 'energy')}
    npt = np.hypot(P4['px'], P4['py']); eta = np.arcsinh(P4['pz'] / npt); phi = np.arctan2(P4['py'], P4['px'])
    sgn = np.where(B['jet_eta'] < 0, -1.0, 1.0)
    nde = sgn * (eta - B['jet_eta']); ndp = (phi - B['jet_phi'] + np.pi) % (2 * np.pi) - np.pi
    nm = np.sqrt(np.maximum(P4['energy'] ** 2 - npt ** 2 - P4['pz'] ** 2, 0))
    pt[r, dst], de[r, dst], dp[r, dst], ms[r, dst] = npt, nde, ndp, nm; pt[r, src] = 0
    return build(B, pt, de, dp, ms)


def ed_scale(B, tok, k):
    from interp.part_core import build, kin
    pt, de, dp, ms = (a.astype(np.float64) for a in kin(B)); r = np.arange(len(tok)); S = pt.sum(1); pl = pt[r, tok]
    s = (S - k * pl) / (S - pl); ok = s > 0; s = np.where(ok, s, 1.0)
    pt2 = pt * s[:, None]; pt2[r, tok] = np.where(ok, k * pl, pl)
    return build(B, pt2, de, dp, ms), ok


# ------------------------------------------------------------------------------------------------ ParT with head controls
class PRun:
    """ParT with per-head key masks, per-head output replacement (mean or donor), attention capture and residual patching.
    Attention modules 0-7 = particle blocks, 8-9 = class blocks (b1, b2). ctl (set per batch):
      kmask {i: (n, H, S) additive}; mean {i: (sel (H,) bool, M (H, C))}; rep {i: (sel (H,) bool, D (n, H, Lq, C))};
      cap {'tok': (n,), 'tokc': (n,)}; recch set(i) (store head outputs); recmean True; recres True; patch {l: (sel (P, n), donor (P, n, C))}"""

    def __init__(self, name, feat):
        import torch, torch.nn.functional as F
        from interp.part_core import ParTNetwork
        self.t, self.F, self.name, self.feat = torch, F, name, feat
        self.net = ParTNetwork(feat); self.m = self.net.model; self.dev = self.net.device; self.m.eval()
        self.ctl = {}; self.out = {}; self.recon = 0.0
        self.msum = {i: torch.zeros(H, C, dtype=torch.float64, device=self.dev) for i in range(10)}; self.mcnt = {i: 0.0 for i in range(10)}
        att = [b.attn for b in self.m.blocks] + [b.attn for b in self.m.cls_blocks]
        for i, a in enumerate(att): self._wrap(i, a)
        self.m.blocks[0].register_forward_pre_hook(lambda mod, args: (self._res(0, args[0]),) + tuple(args[1:]))
        for i, b in enumerate(self.m.blocks): b.register_forward_hook(lambda mod, args, out, i=i: self._res(i + 1, out))

    def _res(self, l, x):
        c = self.ctl
        if c.get('recres'): self.out.setdefault('res', {})[l] = x.detach().clone()
        p = c.get('patch', {}).get(l)
        if p is not None: x = self.t.where(p[0][..., None], p[1], x)
        return x

    def _wrap(self, i, attn):
        orig, S, t = attn.forward, self, self.t

        def f(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
            c = S.ctl; km = c.get('kmask', {}).get(i); mean = c.get('mean', {}).get(i); rep = c.get('rep', {}).get(i)
            cap = c.get('cap'); recm = c.get('recmean'); recch = i in c.get('recch', ())
            if km is None and mean is None and rep is None and not cap and not recm and not recch:
                return orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=False, attn_mask=attn_mask)
            Lq, n, _ = query.shape; Sk = key.shape[0]
            if km is not None:
                am = km[:, :, None, :Sk].expand(n, H, Lq, Sk).reshape(n * H, Lq, Sk)
                attn_mask = am if attn_mask is None else attn_mask + am
            out, w = orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
            realq = (~key_padding_mask).float() if i < 8 else t.ones(n, 1, device=query.device)     # (n, Lq)
            if cap:
                ar = t.arange(n, device=query.device); off = 0 if i < 8 else 1
                wq = (w * realq[:, None, :, None]).sum(2) / realq.sum(1)[:, None, None]             # (n, H, Sk)
                nk = (~key_padding_mask).float().sum(1)[:, None]                          # real keys (class blocks: + class token)
                S.out.setdefault('A', {})[i] = (wq[ar, :, cap['tok'] + off].cpu().numpy(), wq[ar, :, cap['tokc'] + off].cpu().numpy(),
                                                (1.0 / nk).expand(n, H).cpu().numpy())
            if mean is not None or rep is not None or recm or recch:
                v = (value @ attn.in_proj_weight[2 * C:].T + attn.in_proj_bias[2 * C:]).view(Sk, n, H, HD).permute(1, 2, 0, 3)
                ch = t.einsum('nhqd,chd->nhqc', w @ v, attn.out_proj.weight.view(C, H, HD))                  # (n, H, Lq, C)
                if recm or recch:
                    S.recon = max(S.recon, float((ch.sum(1) + attn.out_proj.bias - out.permute(1, 0, 2)).abs().max()))
                if recm:
                    S.msum[i] += (ch * realq[:, None, :, None]).sum((0, 2)).double(); S.mcnt[i] += float(realq.sum())
                if recch: S.out.setdefault('ch', {})[i] = ch.detach().clone()
                d = 0
                if mean is not None:
                    sel, M = mean; d = d + ((M[None, :, None, :] - ch) * sel[None, :, None, None]).sum(1)
                if rep is not None:
                    sel, D = rep; d = d + ((D - ch) * sel[None, :, None, None]).sum(1)
                if mean is not None or rep is not None: out = out + d.permute(1, 0, 2)
            return out, None
        attn.forward = f

    def means(self):
        return {i: (self.msum[i] / max(self.mcnt[i], 1)).float() for i in range(10)}

    def fwd(self, Jb):
        from interp.part_core import inputs
        t = self.t; x, v, m = inputs(Jb, self.feat)
        with t.no_grad(): L = self.m(*(t.from_numpy(a).to(self.dev) for a in (x, v, m)))
        return L.float().cpu().numpy(), x.shape[-1]

    def run(self, J, mk=None, bs=BS, after=None):
        from interp.part_core import sub
        N = len(J['mask']); order = np.argsort(J['mask'].sum(1), kind='stable'); L = np.zeros((N, 10), np.float32)
        for s in range(0, N, bs):
            idx = order[s:s + bs]; Jb = sub(J, idx); P = int(Jb['mask'].sum(1).max())
            self.ctl = mk(idx, P) if mk else {}; self.out = {}
            L[idx], _ = self.fwd(Jb)
            if after: after(idx, self)
        self.ctl = {}; return L


def kmask_for(prun, n, P, heads, tok):
    """heads: list of (i, h) -> {i: (n, H, P+1) additive masks with -1e9 on the token's key}"""
    t = prun.t; d = {}; ar = t.arange(n, device=prun.dev); tk = t.from_numpy(np.asarray(tok)).to(prun.dev).long()
    for (i, h) in heads:
        if i not in d: d[i] = t.zeros(n, H, P + 1, device=prun.dev)
        d[i][ar, h, tk + (0 if i < 8 else 1)] = -1e9
    return d


HEADS = [(i, h) for i in range(10) for h in range(H)]


def hlab(ih):
    i, h = ih; return f'p{i + 1}h{h + 1}' if i < 8 else f'b{i - 7}h{h + 1}'


# ------------------------------------------------------------------------------------------------ ParticleNet
class PNRun:
    def __init__(self, feat):
        import torch
        from interp import pn_net
        NW = add_kinpid(); self.t, self.feat, self.pn = torch, feat, pn_net
        PND = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.m = pn_net.load_particlenet(PND / f'ParticleNet_{feat}.pt', len(NW.FEATURES[feat]), self.dev)
        self.names = [f[0] for f in NW.FEATURES[feat]]; self.kst = dict(mode=None, idx=[], c=0)
        if not hasattr(pn_net, '_orig_knn'): pn_net._orig_knn = pn_net.knn
        ok = pn_net._orig_knn; S = self

        def kfn(x, k):
            st = S.kst
            if st['mode'] == 'fix':
                r = st['idx'][st['c']]; st['c'] += 1; return r
            r = ok(x, k)
            if st['mode'] == 'rec': st['idx'].append(r)
            return r
        pn_net.knn = kfn

    def fwd(self, Jb):
        from interp.part_core import inputs
        t = self.t; x, v, m = inputs(Jb, self.feat); P = x.shape[-1]
        pad = lambda a: np.concatenate([a, np.zeros(a.shape[:2] + (128 - P,), np.float32)], -1) if P < 128 else a
        xp, mp = pad(x), pad(m); pts = xp[:, [self.names.index('deta'), self.names.index('dphi')]]
        T = lambda a: t.from_numpy(np.ascontiguousarray(a)).to(self.dev)
        with t.no_grad(): L = self.m(T(pts), T(xp), T(mp))
        return L.float().cpu().numpy()

    def run(self, J, bs=BS):
        from interp.part_core import sub
        N = len(J['mask']); order = np.argsort(J['mask'].sum(1), kind='stable'); L = np.zeros((N, 10), np.float32)
        for s in range(0, N, bs):
            idx = order[s:s + bs]; L[idx] = self.fwd(sub(J, idx))
        return L


# ------------------------------------------------------------------------------------------------ analysis pieces
def acc_block(L, y, W=None):
    c = L.argmax(1); rec = [float((c[y == k] == k).mean()) if (y == k).any() else None for k in range(10)]
    e = dict(acc10=float((c == y).mean()), balanced=float(np.mean([r for r in rec if r is not None])), recall=rec,
             confusion=[[int(((y == i) & (c == j)).sum()) for j in range(10)] for i in range(10)])
    if W is not None: e['acc10_ci'] = ci(W @ (c == y).astype(np.float32))
    return e


def edit_stats(L0, L1, W, y_ok=HQQL, name=''):
    lp0, lp1 = logsm(L0), logsm(L1); d = lp1[:, y_ok] - lp0[:, y_ok]; c0, c1 = L0.argmax(1), L1.argmax(1)
    ch = c0 != c1; to = np.bincount(c1[ch], minlength=10).tolist()
    return dict(dlogp=mstat(d, W), dlogodds=mstat(lodds(L1, y_ok) - lodds(L0, y_ok), W), calls_changed=mstat(ch.astype(float), W), changed_to=to,
                mcnemar_hqql=mcnemar(c0 == y_ok, c1 == y_ok), median_dlogp=float(np.median(d)), _d=d, _ch=ch)


def main():
    import torch
    NW = add_kinpid()
    from interp.part_core import sub, cat, kin, jmass, PRE, CL, sample
    OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    R = dict(node='P32', seed=SEED, smoke=SMOKE, files='test_20M <sample>_115.root', NLV=NLV, NOTH=NOTH, NKO=NKO, NB=NB, NRAND=NRAND)
    # ---------------- sample
    parts = []
    for k, c in enumerate(CL):
        J, rows = load_rows(TEST / f'{PRE[c]}_115.root', NLV if c == 'Hqql' else NOTH, rng); parts.append(J)
        log(f'loaded {c}: {len(rows)} rows of {PRE[c]}_115.root')
    J0 = cat(parts); B = rebuild(J0); y = B['label'].astype(int); N = len(y)
    tok = lepton_idx(B); R['sample'] = dict(N=N, per_class=np.bincount(y, minlength=10).tolist(),
                                            no_lepton_frac=[float((tok[y == k] < 0).mean()) for k in range(10)])
    lv = np.where((y == HQQL) & (tok >= 0))[0]
    mh, mq = matched_hadron(sub(B, lv), tok[lv], rng)
    lv = lv[mh >= 0]; mh = mh[mh >= 0]; tl = tok[lv]
    BL = sub(B, lv); pt = kin(BL)[0]; r_ = np.arange(len(lv)); zl = pt[r_, tl] / pt.sum(1)
    ise = BL['isElectron'][r_, tl] > 0.5
    R['lepton'] = dict(n_hqql=int((y == HQQL).sum()), n_with_lepton_and_match=int(len(lv)), electron_frac=float(ise.mean()),
                       lepton_pt_share=dict(median=float(np.median(zl)), q10=float(np.percentile(zl, 10)), q90=float(np.percentile(zl, 90))),
                       lepton_rank=dict(median=float(np.median(tl)), frac_rank0=float((tl == 0).mean())),
                       match_abs_log_pt_ratio_median=float(np.median(matched_hadron(BL, tl, np.random.default_rng(SEED))[1])))
    log('sample', R['sample'], 'lepton', R['lepton'])
    # ---------------- edits
    nb = nearest_charged(BL, tl); okm = nb >= 0
    nbc = nearest_charged(BL, mh, exclude=tl); okc = nbc >= 0
    E = {}
    E['base'] = rebuild(BL)          # same number of build passes as every edited jet (edits rebuild BL once more)
    E['e1'] = ed_flag(BL, tl, 'isChargedHadron')
    lflag = np.where(ise, 'isElectron', 'isMuon')
    e2 = {k: (v.copy() if np.ndim(v) else v) for k, v in BL.items()}
    for k in FLAGS: e2[k][r_, mh] = 0
    e2['isElectron'][r_[ise], mh[ise]] = 1; e2['isMuon'][r_[~ise], mh[~ise]] = 1; E['e2'] = rebuild(e2)
    sel3 = np.where(okm & okc)[0]
    E['e3'] = ed_merge(sub(BL, sel3), tl[sel3], nb[sel3]); E['e3c'] = ed_merge(sub(BL, sel3), mh[sel3], nbc[sel3])
    E['e4h'], _ = ed_scale(BL, tl, 0.5); E['e4hc'], _ = ed_scale(BL, mh, 0.5)
    e4d, okd = ed_scale(BL, tl, 2.0); e4dc, okdc = ed_scale(BL, mh, 2.0); sel4 = np.where(okd & okdc)[0]
    E['e4d'] = sub(e4d, sel4); E['e4dc'] = sub(e4dc, sel4)
    SUBSET = dict(e3=sel3, e3c=sel3, e4d=sel4, e4dc=sel4)
    R['edit_checks'] = dict(
        e3_n=int(len(sel3)), e4d_n=int(len(sel4)),
        e3_dmass_max=float(np.abs(jmass(E['e3']) - jmass(sub(BL, sel3))).max()),
        e3_dmass_median=float(np.median(np.abs(jmass(E['e3']) - jmass(sub(BL, sel3))))),
        e4h_dmass_rel_median=float(np.median(np.abs(jmass(E['e4h']) / jmass(BL) - 1))),
        e4h_djetpt_rel_median=float(np.median(np.abs(E['e4h']['jet_pt'] / BL['jet_pt'] - 1))),
        e1_lepton_left=int((lepton_idx(E['e1']) >= 0).sum()), e1_second_lepton_frac=float((lepton_idx(E['e1']) >= 0).mean()))
    log('edit checks', R['edit_checks'])
    # reverse edits on QCD / H4q / Tbqq jets
    rv = np.where(np.isin(y, [0, 4, 8]))[0]; BR = sub(B, rv); chm = (BR['isChargedHadron'] > 0.5) & (BR['mask'] > 0)
    lead = np.where(chm.any(1), chm.argmax(1), -1); rnd = np.full(len(rv), -1)
    for j in range(len(rv)):
        c_ = np.where(chm[j])[0]
        if len(c_) > 1: rnd[j] = rng.choice(c_[1:])
    okr = (lead >= 0) & (rnd >= 0); rvs = np.where(okr)[0]; BRs = sub(BR, rvs)
    ER = dict(base=rebuild(BRs), e5mu=ed_flag(BRs, lead[rvs], 'isMuon'), e5el=ed_flag(BRs, lead[rvs], 'isElectron'), e5mu_r=ed_flag(BRs, rnd[rvs], 'isMuon'))
    yR = BRs['label'].astype(int); R['reverse'] = dict(n=int(len(rvs)), had_lepton_frac=float((lepton_idx(BRs) >= 0).mean()))
    # P1 reference sample
    P1 = rebuild(sample(1000, 3000)); yP1 = P1['label'].astype(int)
    # ---------------- models
    MODELS = [('ParT_full', 'part', 'full'), ('ParT_kinpid', 'part', 'kinpid'), ('ParT_kin', 'part', 'kin'),
              ('PN_full', 'pn', 'full'), ('PN_kinpid', 'pn', 'kinpid')]
    Wall = boot_W(N, NB, rng); Wl = boot_W(len(lv), NB, rng); WR = boot_W(len(rvs), NB, rng)
    Wsub = {k: boot_W(len(v), NB, rng) for k, v in (('e3', sel3), ('e4d', sel4))}
    ko = rng.choice(len(lv), min(NKO, len(lv)), replace=False); ko.sort(); Wk = boot_W(len(ko), NB, rng)
    R['models'] = {}
    for name, kind, feat in MODELS:
        log(f'==== {name}'); res = {}
        M = PRun(name, feat) if kind == 'part' else PNRun(feat)
        res['acc_P1'] = acc_block(M.run(P1), yP1)['acc10']
        Lall = M.run(B); res['sample'] = acc_block(Lall, y, Wall)
        res['sample']['acc10_raw_unrebuilt'] = acc_block(M.run(J0), y)['acc10']
        nl = (tok < 0) & (y == HQQL); res['hqql_recall_with_lepton'] = float((Lall[(tok >= 0) & (y == HQQL)].argmax(1) == HQQL).mean())
        res['hqql_recall_no_lepton'] = float((Lall[nl].argmax(1) == HQQL).mean()) if nl.any() else None
        log(f'  acc P1 {res["acc_P1"]:.4f}  sample acc10 {res["sample"]["acc10"]:.4f}  Hqql recall {res["sample"]["recall"][HQQL]:.4f} '
            f'(with lepton {res["hqql_recall_with_lepton"]:.4f}, without {res["hqql_recall_no_lepton"]})')
        LE = {k: M.run(v) for k, v in E.items()}; ed = {}
        for k in E:
            if k == 'base': continue
            s = SUBSET.get(k); L0 = LE['base'] if s is None else LE['base'][s]
            Wx = Wl if s is None else Wsub['e3' if k.startswith('e3') else 'e4d']
            ed[k] = edit_stats(L0, LE[k], Wx, name=k)
        # paired comparisons edit vs its control
        for a, b, Wx in (('e1', 'e2', Wl), ('e3', 'e3c', Wsub['e3']), ('e4h', 'e4hc', Wl), ('e4d', 'e4dc', Wsub['e4d'])):
            ed[a]['vs_control'] = dict(control=b, ddlogp=mstat(ed[a]['_d'] - ed[b]['_d'], Wx),
                                       mcnemar_changed=mcnemar(ed[b]['_ch'], ed[a]['_ch']))
        ed['e1']['by_flavour'] = dict(electron=dict(n=int(ise.sum()), dlogp=float(ed['e1']['_d'][ise].mean()), changed=float(ed['e1']['_ch'][ise].mean())),
                                      muon=dict(n=int((~ise).sum()), dlogp=float(ed['e1']['_d'][~ise].mean()), changed=float(ed['e1']['_ch'][~ise].mean())))
        d1 = ed['e1']['_d']; ed['e1']['dlogp_by_share_tercile'] = [float(d1[(zl >= lo) & (zl < hi)].mean()) for lo, hi in
                                                                   zip(np.percentile(zl, [0, 33.3, 66.7]), np.percentile(zl, [33.3, 66.7, 100]) + [0, 0, 1e-6])]
        for k in ed:
            log(f'  {k:5s} dlogp {ed[k]["dlogp"]["v"]:+.3f} {np.round(ed[k]["dlogp"]["ci"], 3).tolist()}  changed {ed[k]["calls_changed"]["v"]:.3f}  '
                f'Hqql {ed[k]["mcnemar_hqql"]["before"]:.3f}->{ed[k]["mcnemar_hqql"]["after"]:.3f} p {ed[k]["mcnemar_hqql"]["p"]:.2g} to {ed[k]["changed_to"]}'
                + (f'  vs {ed[k]["vs_control"]["control"]}: dd {ed[k]["vs_control"]["ddlogp"]["v"]:+.3f} McN p {ed[k]["vs_control"]["mcnemar_changed"]["p"]:.2g}' if 'vs_control' in ed[k] else ''))
        res['edits'] = ed
        LR = {k: M.run(v) for k, v in ER.items()}; rvr = {}
        for k in ('e5mu', 'e5el', 'e5mu_r'):
            e = edit_stats(LR['base'], LR[k], WR); c1 = LR[k].argmax(1); c0 = LR['base'].argmax(1)
            e['to_hqql'] = mstat(((c1 == HQQL) & (c0 != HQQL)).astype(float), WR); e['to_tbl'] = mstat(((c1 == TBL) & (c0 != TBL)).astype(float), WR)
            e['per_class'] = {CL[k2]: dict(dlogp=float(e['_d'][yR == k2].mean()), to_hqql=float(((c1 == HQQL) & (c0 != HQQL))[yR == k2].mean()))
                              for k2 in (0, 4, 8)}
            rvr[k] = e
        rvr['e5mu_vs_rand'] = mstat(rvr['e5mu']['_d'] - rvr['e5mu_r']['_d'], WR)
        for k in ('e5mu', 'e5el', 'e5mu_r'):
            log(f'  {k:6s} dlogp(Hqql) {rvr[k]["dlogp"]["v"]:+.3f} {np.round(rvr[k]["dlogp"]["ci"], 3).tolist()} to Hqql {rvr[k]["to_hqql"]["v"]:.4f} '
                f'to Tbl {rvr[k]["to_tbl"]["v"]:.4f}  per class {rvr[k]["per_class"]}')
        res['reverse'] = rvr
        if kind == 'part': res['internals'] = part_internals(M, name, E, LE, lv, tl, mh, ko, Wk, rng, B, y, Wall)
        else: res['internals'] = pn_internals(M, E, LE, tl, mh, ko, Wk)
        R['models'][name] = res
        FN.write_text(json.dumps(strip(R), indent=1)); log(f'  wrote {FN}')
        del M; torch.cuda.empty_cache()
    R['runtime_s'] = time.time() - T0
    FN.write_text(json.dumps(strip(R), indent=1)); log(f'done; wrote {FN}')


def part_internals(M, name, E, LE, lv, tl, mh, ko, Wk, rng, B, y, Wall):
    from interp.part_core import sub
    t = M.t; out = {}; BK = sub(E['base'], ko); B1 = sub(E['e1'], ko); tk, tc = tl[ko], mh[ko]; n = len(ko)
    L0 = M.run(BK); lp0 = logsm(L0)[:, HQQL]; lo0 = lodds(L0); c0 = L0.argmax(1)
    out['metric'] = 'internals: Delta log-odds of Hqql (logit_Hqql - logsumexp of the other 9 logits) per jet; dlogp = Delta log p(Hqql)'
    # attention capture
    A = {i: [np.zeros((n, H)), np.zeros((n, H)), np.zeros((n, H))] for i in range(10)}

    def mk_cap(idx, P): return dict(cap=dict(tok=t.from_numpy(tk[idx]).to(M.dev).long(), tokc=t.from_numpy(tc[idx]).to(M.dev).long()))

    def aft(idx, S):
        for i, (a, b, u) in S.out['A'].items(): A[i][0][idx], A[i][1][idx], A[i][2][idx] = a, b, u
    M.run(BK, mk_cap, after=aft)
    att = {}
    for (i, h) in HEADS:
        a, b, u = A[i][0][:, h], A[i][1][:, h], A[i][2][:, h]
        att[hlab((i, h))] = dict(lepton=float(a.mean()), matched=float(b.mean()), uniform=float(u.mean()), diff=mstat(a - b, Wk))
    pr = holm([att[hlab(x)]['diff']['p0'] for x in HEADS])
    for x, p in zip(HEADS, pr): att[hlab(x)]['p_holm'] = p
    out['attention'] = att
    cl = [hlab(x) for x in HEADS if x[0] >= 8]
    log('  class-attn weight on lepton / matched: ' + ', '.join(f'{k} {att[k]["lepton"]:.3f}/{att[k]["matched"]:.3f}' for k in cl))

    def run_ko(heads, toks):
        if not heads: return L0
        return M.run(BK, lambda idx, P: dict(kmask=kmask_for(M, len(idx), P, heads, toks[idx])))

    def dl(L): return lodds(L) - lo0

    def dlp(L): return logsm(L)[:, HQQL] - lp0
    single = {}; dlep = {}; dctl = {}
    for x in HEADS:
        a, b = dl(run_ko([x], tk)), dl(run_ko([x], tc)); dlep[x], dctl[x] = a, b
        single[hlab(x)] = dict(lepton=mstat(a, Wk), matched=mstat(b, Wk), diff=mstat(a - b, Wk))
        if x[0] >= 8: single[hlab(x)]['lepton_dlogp'] = float(dlp(run_ko([x], tk)).mean())
    ph = holm([single[hlab(x)]['diff']['p0'] for x in HEADS])
    for x, p in zip(HEADS, ph): single[hlab(x)]['p_holm'] = p
    sig = [x for x in HEADS if single[hlab(x)]['p_holm'] < 0.05 and single[hlab(x)]['diff']['v'] < 0]
    sig.sort(key=lambda x: single[hlab(x)]['diff']['v'])
    out['ko_single'] = single; out['ko_significant'] = [hlab(x) for x in sig]
    rank = sorted(HEADS, key=lambda x: single[hlab(x)]['diff']['v'])
    log(f'  single-head lepton KO: {len(sig)} Holm-significant heads {[hlab(x) for x in sig][:20]}; top 8 by diff: '
        + ', '.join(f'{hlab(x)} {single[hlab(x)]["lepton"]["v"]:+.3f} (ctl {single[hlab(x)]["matched"]["v"]:+.3f})' for x in rank[:8]))

    def joint(heads, label):
        La, Lb = run_ko(heads, tk), run_ko(heads, tc); a, b = dl(La), dl(Lb)
        e = dict(heads=[hlab(x) for x in heads], lepton=mstat(a, Wk), lepton_dlogp=mstat(dlp(La), Wk), matched=mstat(b, Wk), diff=mstat(a - b, Wk),
                 calls_changed=float((La.argmax(1) != c0).mean()), calls_changed_matched=float((Lb.argmax(1) != c0).mean()),
                 mcnemar_hqql=mcnemar(c0 == HQQL, La.argmax(1) == HQQL), _a=a)
        log(f'  joint KO {label} ({len(heads)} heads): lepton {e["lepton"]["v"]:+.3f} {np.round(e["lepton"]["ci"], 3).tolist()} matched {e["matched"]["v"]:+.3f} '
            f'changed {e["calls_changed"]:.3f} / {e["calls_changed_matched"]:.3f}  Hqql {e["mcnemar_hqql"]["before"]:.3f}->{e["mcnemar_hqql"]["after"]:.3f}')
        return e
    J = {}
    J['all80'] = joint(HEADS, 'all80'); J['class16'] = joint([x for x in HEADS if x[0] >= 8], 'class16')
    J['particle64'] = joint([x for x in HEADS if x[0] < 8], 'particle64')
    for i in range(10): J[f'block_{hlab((i, 0))[:2]}'] = joint([(i, h) for h in range(H)], f'block {hlab((i, 0))[:2]}')
    tot = J['all80']['_a']
    for k in J: J[k]['share_of_all80'] = float(J[k]['lepton']['v'] / tot.mean()) if tot.mean() != 0 else None
    S_ = sig if sig else rank[:3]
    J['significant'] = joint(S_, 'significant' if sig else 'top3 (no Holm-significant head)')
    J['significant']['share_of_all80'] = float(J['significant']['lepton']['v'] / tot.mean())
    ncls = sum(1 for x in S_ if x[0] >= 8); npar = len(S_) - ncls; rv = []
    for _ in range(NRAND):
        hs = [HEADS[64 + j] for j in rng.choice(16, ncls, replace=False)] + [HEADS[j] for j in rng.choice(64, npar, replace=False)]
        rv.append(float(dl(run_ko(hs, tk)).mean()))
    rv = np.array(rv); J['significant']['random_sets'] = dict(n=len(rv), mean=float(rv.mean()), p5=float(np.percentile(rv, 5)), min=float(rv.min()),
                                                             p_perm=float((1 + (rv <= J['significant']['lepton']['v']).sum()) / (1 + len(rv))))
    greedy = []
    for k in range(1, 11):
        a = dl(run_ko(rank[:k], tk)); greedy.append(dict(k=k, heads=[hlab(x) for x in rank[:k]], lepton=float(a.mean()), share_of_all80=float(a.mean() / tot.mean())))
    J['greedy_topk'] = greedy
    log('  greedy top-k share of all-80 KO: ' + ', '.join(f'{g["k"]}:{g["share_of_all80"]:.2f}' for g in greedy))
    out['ko_joint'] = J
    # e1 effects: head patching (class heads) and residual patching
    if name != 'ParT_kin':
        L1 = M.run(B1); d1 = lodds(L1) - lo0; out['e1_on_ko_subset'] = mstat(d1, Wk); out['e1_on_ko_subset_dlogp'] = mstat(dlp(L1), Wk)
        hp = {x: np.zeros((n, 10), np.float32) for x in HEADS if x[0] >= 8}; hp['all16'] = np.zeros((n, 10), np.float32)
        rp = {(l, m): np.zeros((n, 10), np.float32) for l in range(9) for m in ('lepton', 'others')}
        from interp.part_core import sub as sb
        order = np.argsort(BK['mask'].sum(1), kind='stable')
        for s in range(0, n, BS):
            idx = order[s:s + BS]; Jb0, Jb1 = sb(BK, idx), sb(B1, idx); nb_ = len(idx)
            M.ctl = dict(recch={8, 9}, recres=True); M.out = {}; M.fwd(Jb1); don_ch = M.out['ch']; don_res = M.out['res']
            P = don_res[0].shape[0]
            for x in hp:
                if x == 'all16': sel = {8: t.ones(H, dtype=t.bool, device=M.dev), 9: t.ones(H, dtype=t.bool, device=M.dev)}
                else:
                    sel = {x[0]: t.zeros(H, dtype=t.bool, device=M.dev)}; sel[x[0]][x[1]] = True
                M.ctl = dict(rep={i: (sel[i], don_ch[i]) for i in sel}); hp[x][idx], _ = M.fwd(Jb0)
            ar = t.arange(nb_, device=M.dev); tkk = t.from_numpy(tk[idx]).to(M.dev).long()
            for l in range(9):
                lepsel = t.zeros(P, nb_, dtype=t.bool, device=M.dev); lepsel[tkk, ar] = True
                for m, sl in (('lepton', lepsel), ('others', ~lepsel)):
                    M.ctl = dict(patch={l: (sl, don_res[l])}); rp[(l, m)][idx], _ = M.fwd(Jb0)
            M.ctl = {}
        dm = d1.mean()
        out['e1_head_patch'] = {(hlab(x) if x != 'all16' else 'all16'): dict(d=mstat(dl(L), Wk), share=float(dl(L).mean() / dm)) for x, L in hp.items()}
        out['e1_residual_patch'] = {f'L{l}_{m}': dict(d=mstat(dl(rp[(l, m)]), Wk), share=float(dl(rp[(l, m)]).mean() / dm)) for (l, m) in rp}
        out['reconstruction_max_abs'] = M.recon
        log(f'  e1 on KO subset {d1.mean():+.3f}; class-head patch shares: ' + ', '.join(f'{k} {v["share"]:.2f}' for k, v in out['e1_head_patch'].items()))
        log('  residual patch shares (lepton / others): ' + ', '.join(f'L{l} {out["e1_residual_patch"][f"L{l}_lepton"]["share"]:.2f}/{out["e1_residual_patch"][f"L{l}_others"]["share"]:.2f}' for l in range(9)))
    # literal circuit on the whole 10-class sample
    if name != 'ParT_kin':
        M.ctl = {}
        pool = np.concatenate([np.where(y != HQQL)[0], rng.choice(np.where(y == HQQL)[0], int(np.bincount(y[y != HQQL]).max()), replace=False)])
        M.run(sub(B, pool), lambda idx, P: dict(recmean=True)); MU = M.means(); out['mean_pool'] = dict(n=int(len(pool)), per_class=np.bincount(y[pool], minlength=10).tolist())
        clsS = [x for x in S_ if x[0] >= 8]; flag = 'Holm-significant' if (sig and clsS) else 'top-3 class heads by single-head KO (fallback)'
        if not (sig and clsS): clsS = [x for x in rank if x[0] >= 8][:3]
        allS = list(dict.fromkeys(list(S_) + clsS))

        def circ(keep, scope):
            pool = [x for x in HEADS if (x[0] >= 8 if scope == 'class' else True)]   # 'all' and 'top3': all 80 heads
            abl = [x for x in pool if x not in keep]

            def mk(idx, P):
                d = {}
                for (i, h) in abl:
                    if i not in d: d[i] = (t.zeros(H, dtype=t.bool, device=M.dev), MU[i])
                    d[i][0][h] = True
                return dict(mean=d)
            return M.run(B, mk)

        def metr(L, full=False):
            e = acc_block(L, y, Wall if full else None); c = L.argmax(1)
            e['hqql_recall'] = e['recall'][HQQL]; e['other_called_hqql'] = float((c[y != HQQL] == HQQL).mean())
            e['hqql_vs_rest_balanced'] = 0.5 * (e['hqql_recall'] + 1 - e['other_called_hqql'])
            if full:
                e['hqql_recall_ci'] = ci(Wall[:, y == HQQL] @ (c[y == HQQL] == HQQL).astype(np.float32) / Wall[:, y == HQQL].sum(1))
            else: e.pop('confusion')
            return e
        CI = {}; top3 = rank[:3]
        for scope, keep in (('class', clsS), ('all', allS), ('top3', top3)):
            e = metr(circ(keep, scope), True); e['heads'] = [hlab(x) for x in keep]; e['selection'] = flag if scope != 'top3' else 'top-3 heads by single-head lepton KO (all 80 heads, others mean-ablated)'
            e['none_kept'] = metr(circ([], scope))
            ncls = sum(1 for x in keep if x[0] >= 8); npar = len(keep) - ncls; rv = []
            for _ in range(NRAND):
                hs = [HEADS[64 + j] for j in rng.choice(16, ncls, replace=False)]
                if scope != 'class': hs += [HEADS[j] for j in rng.choice(64, npar, replace=False)]
                rv.append(metr(circ(hs, scope)))
            for k in ('hqql_vs_rest_balanced', 'hqql_recall', 'acc10', 'balanced'):
                v = np.array([r[k] for r in rv]); e[f'random_{k}'] = dict(mean=float(v.mean()), p95=float(np.percentile(v, 95)), max=float(v.max()))
                e[f'p_perm_{k}'] = float((1 + (v >= e[k]).sum()) / (1 + len(v)))
            CI[scope] = e
            log(f'  circuit [{scope}] {e["heads"]} ({flag}): Hqql recall {e["hqql_recall"]:.4f} {np.round(e["hqql_recall_ci"], 4).tolist()} others->Hqql {e["other_called_hqql"]:.4f} Hqql-vs-rest {e["hqql_vs_rest_balanced"]:.4f} (random mean {e["random_hqql_vs_rest_balanced"]["mean"]:.4f} max {e["random_hqql_vs_rest_balanced"]["max"]:.4f} p {e["p_perm_hqql_vs_rest_balanced"]:.3f}) acc10 {e["acc10"]:.4f} '
                f'balanced {e["balanced"]:.4f} | none kept Hqql {e["none_kept"]["hqql_recall"]:.4f} others->Hqql {e["none_kept"]["other_called_hqql"]:.4f} acc10 {e["none_kept"]["acc10"]:.4f} | random Hqql mean '
                f'{e["random_hqql_recall"]["mean"]:.4f} max {e["random_hqql_recall"]["max"]:.4f} p {e["p_perm_hqql_recall"]:.3f}; acc10 mean {e["random_acc10"]["mean"]:.4f} p {e["p_perm_acc10"]:.3f}')
        out['circuit'] = CI
    return out


def pn_internals(M, E, LE, tl, mh, ko, Wk):
    from interp.part_core import sub
    t = M.t; BK, B1, B2 = sub(E['base'], ko), sub(E['e1'], ko), sub(E['e2'], ko); tk, tc = tl[ko], mh[ko]; n = len(ko)
    order = np.argsort(BK['mask'].sum(1), kind='stable')
    ov = dict(lep_e1=np.zeros((n, 3)), had_e2=np.zeros((n, 3)), all_e1=np.zeros((n, 3))); Lfix = np.zeros((n, 10), np.float32)
    L0 = np.zeros((n, 10), np.float32); L1 = np.zeros((n, 10), np.float32)
    for s in range(0, n, BS):
        idx = order[s:s + BS]; ar = np.arange(len(idx)); real = BK['mask'][idx] > 0
        G = {}
        for nm, J in (('0', BK), ('1', B1), ('2', B2)):
            M.kst = dict(mode='rec', idx=[], c=0); Lx = M.fwd(sub(J, idx)); G[nm] = [g.cpu().numpy() for g in M.kst['idx']]
            if nm == '0': L0[idx] = Lx
            if nm == '1': L1[idx] = Lx
        for l in range(3):
            g0, g1, g2 = G['0'][l], G['1'][l], G['2'][l]
            same = lambda a, b: (a[..., :, None] == b[..., None, :]).any(-1).mean(-1)          # (n, P) neighbour-set overlap
            s1, s2 = same(g0, g1), same(g0, g2); Pq = s1.shape[1]
            ov['lep_e1'][idx, l] = 1 - s1[ar, tk[idx]]; ov['had_e2'][idx, l] = 1 - s2[ar, tc[idx]]
            ov['all_e1'][idx, l] = 1 - (s1 * real[:, :Pq]).sum(1) / real[:, :Pq].sum(1)
        M.kst = dict(mode='rec', idx=[], c=0); M.fwd(sub(BK, idx)); fixed = M.kst['idx']
        M.kst = dict(mode='fix', idx=fixed, c=0); Lfix[idx] = M.fwd(sub(B1, idx)); M.kst = dict(mode=None, idx=[], c=0)
    lp0 = lodds(L0); d1 = lodds(L1) - lp0; dfix = lodds(Lfix) - lp0
    out = dict(knn_changed_frac={k: [mstat(v[:, l], Wk) for l in range(3)] for k, v in ov.items()},
               e1=mstat(d1, Wk), e1_graph_fixed=mstat(dfix, Wk), graph_share=mstat(d1 - dfix, Wk),
               graph_share_ratio=float((d1 - dfix).mean() / d1.mean()) if d1.mean() != 0 else None)
    log('  PN kNN neighbours changed (EdgeConv 1/2/3): lepton e1 ' + ' '.join(f'{x["v"]:.3f}' for x in out['knn_changed_frac']['lep_e1'])
        + ' | matched hadron e2 ' + ' '.join(f'{x["v"]:.3f}' for x in out['knn_changed_frac']['had_e2'])
        + ' | all tokens e1 ' + ' '.join(f'{x["v"]:.3f}' for x in out['knn_changed_frac']['all_e1']))
    log(f'  PN e1 {d1.mean():+.3f}, graph fixed {dfix.mean():+.3f}, graph-change share {out["graph_share_ratio"]}')
    return out


# ------------------------------------------------------------------------------------------------ figure
def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads(FN.read_text()); ms = list(R['models'])
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1.4]))
    ks = [('e1', 'lepton flag\n-> hadron'), ('e2', 'control:\nhadron flag\n-> lepton'), ('e3', 'lepton\nmerged into\nneighbour'),
          ('e3c', 'control:\nhadron\nmerged'), ('e4h', 'lepton\npT x0.5'), ('e4hc', 'control:\nhadron\npT x0.5')]
    w = 0.8 / len(ms); cols = ['#1f4e79', '#4f81bd', '#9dc3e6', '#c55a11', '#f4b183']
    for j, m in enumerate(ms):
        v = [100 * R['models'][m]['edits'][k]['calls_changed']['v'] for k, _ in ks]
        e = [[100 * (R['models'][m]['edits'][k]['calls_changed']['v'] - R['models'][m]['edits'][k]['calls_changed']['ci'][0]) for k, _ in ks],
             [100 * (R['models'][m]['edits'][k]['calls_changed']['ci'][1] - R['models'][m]['edits'][k]['calls_changed']['v']) for k, _ in ks]]
        ax[0].bar(np.arange(len(ks)) + (j - len(ms) / 2 + 0.5) * w, v, w, yerr=e, color=cols[j], label=m.replace('_', ' '))
    ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels([l for _, l in ks], fontsize=8); ax[0].set_ylabel('H->lvqq jets whose call changes [%]')
    ax[0].set_title('Removing the lepton flag or the lepton changes ~all H->lvqq calls\n(N = %d jets; ParT kin has no flag input: 0 %%)' % R['lepton']['n_with_lepton_and_match'], fontsize=10)
    ax[0].legend(fontsize=8)
    I = R['models']['ParT_full']['internals']['ko_single']; labs = list(I)
    v = np.array([I[k]['lepton']['v'] for k in labs]); c = np.array([I[k]['matched']['v'] for k in labs])
    x = np.arange(len(labs)); ax[1].bar(x, v, 0.8, color=['#1f4e79' if k.startswith('b') else '#7f7f7f' for k in labs])
    ax[1].plot(x, c, 'o', ms=2.5, color='#c55a11')
    ax[1].legend(handles=[Patch(color='#7f7f7f', label='lepton key masked, particle-block head'), Patch(color='#1f4e79', label='lepton key masked, class-attention head'),
                          Line2D([], [], marker='o', ls='', ms=4, color='#c55a11', label='control: matched-pT hadron key masked')], fontsize=8, loc='lower left')
    for k in R['models']['ParT_full']['internals']['ko_significant'][:5]:
        i = labs.index(k); ax[1].annotate(k, (i, v[i]), fontsize=7, ha='center', va='top', xytext=(0, -3), textcoords='offset points')
    ax[1].set_xticks([8 * i + 3.5 for i in range(10)]); ax[1].set_xticklabels([f'P{i + 1}' for i in range(8)] + ['cls 1', 'cls 2'], fontsize=8)
    ax[1].set_ylabel('mean change of H->lvqq log-odds'); ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set_title('ParT_full: masking the lepton in one class-attention head (b2h1)\nlowers the H->lvqq log-odds most (one head at a time, N = %d jets)' % R['NKO'], fontsize=10)
    fig.tight_layout(); p = OUTD / 'p32_lepton_route.png'; fig.savefig(p, dpi=130); print('wrote', p)


if __name__ == '__main__':
    plot() if ARGS.plot else main()
