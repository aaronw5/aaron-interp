"""P65 (loop C, round C14; new claim C29, exploratory) — H->bb vs H->cc vs Z->qq flavour circuit in ParT_full: which attention heads
and class-token MLP neurons read the track-displacement inputs (d0, dz and their errors), and does a literal circuit of them alone
separate b from c.

Models: ParT_full (released); references ParT_kin (no displacement inputs; its b-vs-c accuracy is the floor for "no displacement")
and an untrained ParT_full (every weight redrawn from a Gaussian with the trained tensor's mean and spread, ParTNetwork(untrained)).

Sample. JetClass test_20M <sample>_117.root (the planned _127 does not exist: test_20M holds _100.._119; _117 has not been used for a
flavour study), numpy default_rng(6565), random rows (never first-k):
  HToBB, HToCC, ZToQQ, ZJetsToNuNu (QCD): NR rows each = ranking set; a disjoint NE rows each = evaluation set;
  the six other classes: NOTH rows each (evaluation set, so the circuit has a 10-class confusion);
  all 10 classes: NDON further rows each = donor / reference pool (mean ablation means; donors of the literal circuit);
  QCD: NPOOL further rows = light-track pool (donor tracks of the displacement edit).
  Every jet passes through part_core.build (rebuild); edited jets are compared with the rebuilt unedited jets.
Scores: s_bc = logit(H->bb) - logit(H->cc) = log(p_bb / p_cc); s_bl = logit(H->bb) - logit(Z->qq). Signed per-jet margin
  u_bc = +s_bc on H->bb jets, -s_bc on H->cc jets (u_bl: +s_bl on H->bb, -s_bl on Z->qq). An intervention's effect is the mean
  change du over the jets of the pair (negative = the intervention removes separation). b-vs-c accuracy = share of H->bb and H->cc
  jets with sign(s_bc) right; AUC of s_bc for H->bb vs H->cc. Z->qq includes Z->bb / Z->cc (about 22 % / 17 % of hadronic Z decays).

Step 1 — input check (evaluation rows; matched to loop D's H44, which scaled displacements; here they are replaced):
  "tracks" = particles with charge != 0. Displacement significance S = |d0| / sigma(d0). Edit = (d0val, d0err, dzval, dzerr) of a
  track set to those of a track from the light-track pool (QCD jets) with the same charge and nearest pT (random among the 5 nearest);
  four-momenta, flags, charge unchanged (checked).
  a   all tracks;  b   the 3 tracks of largest S;  c (control)  the 3 tracks of smallest S (same count as b in every jet);
  d (control)  all tracks replaced by same-class tracks (pool = ranking-set jets of the same class; pT- and charge-matched):
       measures the response to resampling the IP values without removing flavour information.
  Applied to H->bb, H->cc, Z->qq, QCD; du_bc, du_bl, s_bc on H->bb and H->cc separately, accuracy / AUC before -> after, McNemar.
Step 2 — single-head scans (ranking rows; 80 heads: p1h1..p8h8 = particle blocks 1-8, b1h1..b2h8 = class blocks 1-2):
  mean ablation: the head's 128-dim output contribution (after out_proj) replaced by its mean over the donor pool's real tokens.
  resample ablation: the head's output replaced by the head's output in a pT-matched jet of the other flavour (H->bb <- H->cc and
    H->cc <- H->bb for b-vs-c; H->bb <- Z->qq and Z->qq <- H->bb for b-vs-light), token by token in pT order (tokens beyond the donor's
    multiplicity get the donor's mean over its real tokens).
  du_bc, du_bl with 1,000-resample bootstrap intervals, p (normal approximation with the bootstrap s.e.), Holm over 160 tests
  (80 heads x 2 ablations) per metric. Frozen ranking = resample-ablation du_bc on the ranking rows (most negative first).
  key masking: for one head, the key projection reads the residual stream of the edit-a jet (all tracks' IP values light-matched),
    queries and values read the unedited stream: du_key / du_input(a) = the head's displacement share through its keys.
    Joint key masking for the top-k heads, the 64 particle heads, the 16 class heads, all 80.
  class-token MLP neurons (2 blocks x 512, GELU output of fc1): mean ablation one neuron at a time, Holm over 1,024 per metric.
  held-out check: single-head resample effects of the top 8 heads re-measured on the evaluation rows.
  joint resample ablation of the frozen top-k heads (k = 1, 2, 4, 8, 16, 32) on the evaluation rows vs NRJ block-matched random sets.
Step 3 — literal circuit (evaluation rows; user definition: only its heads / neurons): kept = the top-k heads of the frozen ranking
  (k = 1, 2, 4, 8, 16, 32) + the top NN class-MLP neurons (Holm-significant, most negative du_bc); everything else resample-ablated
  from a pT-matched random jet of the 10-class donor pool: the other 80 - k heads, all 8 particle-block MLP outputs, the other
  class-MLP neurons. The input embedding, pair embedding (attention bias of kept heads), LayerNorms, class token and last layer stay.
  b-vs-c accuracy / AUC, b-vs-light accuracy / AUC, 10-class accuracy and confusion; NRS block-matched random head sets (same
  number of heads per block, same neurons) on a class-balanced subsample (NSUB per class of H->bb, H->cc, Z->qq), rank p =
  (1 + #random >= selected) / (1 + NRS); all-ablated floor (= the donor's own output, checked); variants (k = 0 neurons only,
  heads without neurons, all class neurons kept, particle MLPs kept).
References: ParT_kin, untrained: unedited accuracy, b-vs-c accuracy / AUC, input edits a-d, key-masking scan (untrained).
Figure data: one H->bb evaluation jet with >= 3 tracks of S > 5, the top head's attention (class head: from the class token;
  particle head: mean over real queries of the weight each key receives); share of that head's attention on the 3 largest-S tracks
  vs the 3 smallest-S tracks over all H->bb evaluation jets, unedited vs edit a.

  ./sync.sh && ./part_job.sh c-p65-run 'python -m interp.p65_flavour_circuit_part --smoke && python -m interp.p65_flavour_circuit_part && python -m interp.p65_plot'
  -> $AI_RESULTS/part/p65_flavour_circuit[_smoke].json
"""
import json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p32_lepton_route as P   # noqa: E402  (stats helpers, load_rows, rebuild; parses --smoke / --plot)

SMOKE = P.SMOKE
SEED = 6565
FILE = '117'
NR = 60 if SMOKE else 3000
NE = 60 if SMOKE else 3000
NOTH = 20 if SMOKE else 500
NDON = 20 if SMOKE else 300
NPOOL = 80 if SMOKE else 2000
NSUB = 30 if SMOKE else 400
NB = 100 if SMOKE else 1000
NRS = 4 if SMOKE else 100
NRJ = 3 if SMOKE else 20
NN = 32
NREFM = 40 if SMOKE else 1000        # per class (bb, cc, qq) for the ParT_kin / untrained references
KS = (1, 2, 4, 8, 16, 32)
BS = 100 if SMOKE else 200
QCD, HBB, HCC, ZQQ = 0, 1, 2, 6
H, C, HD, NU = 8, 128, 16, 512
IPK = ('d0val', 'd0err', 'dzval', 'dzerr')
HEADS = P.HEADS
hlab = P.hlab
FN = P.OUTD / f'p65_flavour_circuit{P.TAG}.json'
log = P.log


# ------------------------------------------------------------------------------------------------ inputs with a fixed length
def inputs_P(J, n, Pn):
    """as jetdistill.part.network.inputs, padded (weaver 'wrap' padding) to a given length Pn >= the batch's multiplicity"""
    from jetdistill.part.network import FEATURES, raw_features
    m = J['mask'][:, :Pn] > 0; cnt = m.sum(1)
    wrap = np.arange(Pn)[None, :] % np.maximum(cnt, 1)[:, None]
    fill = lambda a: np.where(m, a[:, :Pn], np.take_along_axis(a[:, :Pn], wrap, 1))
    feats = []
    for k, c, s, cl in FEATURES[n]:
        x = raw_features(J, k)
        if c is not None: x = np.clip((x - np.float32(c)) * np.float32(s), *cl)
        feats.append(fill(np.where(J['mask'] > 0, x, 0)))
    vec = np.stack([fill(J[k]) for k in ('px', 'py', 'pz', 'energy')], 1)
    return np.stack(feats, 1).astype(np.float32), vec.astype(np.float32), m[:, None, :].astype(np.float32)


# ------------------------------------------------------------------------------------------------ ParT with unit controls
class Run:
    """ParT with per-head output replacement (mean / donor), per-head key patching, particle-MLP donor replacement, class-MLP neuron
    mean / donor replacement, donor recording and attention capture. The sequence trimmer is off and every batch is padded to a
    common length, so a recipient batch and its donor batch have the same token positions. Modules 0-7 = particle blocks, 8-9 =
    class blocks. ctl keys (per forward):
      mean {i: (sel (H,) bool, M (H, C))}; rep {i: (sel (H,) bool, D (n, H, Lq, C))}; kpatch {i: (sel (H,) bool, K (Sk, n, C))};
      mlp_rep {i: D (Lq, n, C)} (i < 8); act_mean_rows {j: (rows, units)}; act_rep {j: (keep (512,) bool, D (1, n, 512))};
      rec {'ch': set, 'mlp': set, 'act': set, 'key': set}; recmean True; cap (i, h); check True (manual attention vs torch)."""

    def __init__(self, feat, untrained=False):
        import torch
        from interp.part_core import ParTNetwork
        self.t, self.feat = torch, feat
        self.net = ParTNetwork(feat, untrained=untrained, seed=SEED); self.m = self.net.model; self.dev = self.net.device; self.m.eval()
        self.m.trimmer.enabled = False
        self.ctl, self.st, self.cap, self.recon, self.pe, self.pe_on, self.x7, self.qmask = {}, {}, None, 0.0, None, False, None, None
        dz = lambda shp: torch.zeros(*shp, dtype=torch.float64, device=self.dev)
        self.hs = {i: dz((H, C)) for i in range(10)}; self.hc = {i: 0.0 for i in range(10)}
        self.ms = {i: dz((C,)) for i in range(10)}; self.mc = {i: 0.0 for i in range(10)}
        self.as_ = {j: dz((NU,)) for j in range(2)}; self.ac = {j: 0.0 for j in range(2)}
        self.HM = self.MM = self.AM = None
        blocks = list(self.m.blocks) + list(self.m.cls_blocks)
        for i, b in enumerate(blocks):
            self._wrap(i, b.attn); b.fc2.register_forward_hook(lambda mod, inp, out, i=i: self._mlp(i, out))
        for j, b in enumerate(self.m.cls_blocks):
            b.act.register_forward_hook(lambda mod, inp, out, j=j: self._act(j, out))
        self.m.blocks[-1].register_forward_hook(lambda mod, inp, out: self._x7(out))
        pe = self.m.pair_embed; ope = pe.forward; S = self

        def pef(x, uu=None):
            if S.pe_on and S.pe is not None: return S.pe
            o = ope(x, uu)
            if S.pe_on: S.pe = o
            return o
        pe.forward = pef

    def _x7(self, out):
        if self.ctl.get('capx'): self.x7 = out

    def _fill(self, a, realq):
        """a (n, H, Lq, C) or (Lq, n, C): padded query positions <- the jet's mean over its real positions (donor alignment)"""
        t = self.t
        if a.dim() == 4:
            q = realq[:, None, :, None]; mu = (a * q).sum(2, keepdim=True) / q.sum(2, keepdim=True).clamp(min=1)
            return t.where(q > 0, a, mu.expand_as(a))
        q = realq.T[..., None]; mu = (a * q).sum(0, keepdim=True) / q.sum(0, keepdim=True).clamp(min=1)
        return t.where(q > 0, a, mu.expand_as(a))

    def _mlp(self, i, out):
        c = self.ctl; t = self.t
        rec = c.get('rec')
        if rec and i in rec.get('mlp', ()):
            self.st.setdefault('mlp', {})[i] = self._fill(out, self.qmask) if i < 8 else out
        if c.get('recmean'):
            if i < 8:
                q = self.qmask.T                                                     # (Lq, n)
                self.ms[i] += (out * q[..., None]).sum((0, 1)).double(); self.mc[i] += float(q.sum())
            else:
                self.ms[i] += out.sum((0, 1)).double(); self.mc[i] += float(out.shape[0] * out.shape[1])
        D = c.get('mlp_rep', {}).get(i)
        if D is not None: return D.clone()
        return None

    def _act(self, j, out):
        c = self.ctl; t = self.t
        rec = c.get('rec')
        if rec and j in rec.get('act', ()): self.st.setdefault('act', {})[j] = out
        if c.get('recmean'): self.as_[j] += out.sum((0, 1)).double(); self.ac[j] += float(out.shape[0] * out.shape[1])
        r = c.get('act_mean_rows', {}).get(j)
        a = c.get('act_rep', {}).get(j)
        if r is None and a is None: return None
        o = out.clone()
        if r is not None:
            rows, units = r; o[0, rows, units] = self.AM[j][units]
        if a is not None:
            keep, D = a; o = t.where(keep[None, None, :], o, D)
        return o

    def _wrap(self, i, attn):
        orig, S, t = attn.forward, self, self.t
        Wq, Wk, Wv = attn.in_proj_weight[:C], attn.in_proj_weight[C:2 * C], attn.in_proj_weight[2 * C:]
        bq, bk, bv = attn.in_proj_bias[:C], attn.in_proj_bias[C:2 * C], attn.in_proj_bias[2 * C:]
        Wo = attn.out_proj.weight.view(C, H, HD); bo = attn.out_proj.bias

        def f(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
            c = S.ctl; rec = c.get('rec') or {}
            if i in rec.get('key', ()): S.st.setdefault('key', {})[i] = key
            mean = c.get('mean', {}).get(i); rep = c.get('rep', {}).get(i); kp = c.get('kpatch', {}).get(i)
            cap = c.get('cap'); capi = cap is not None and cap[0] == i
            if mean is None and rep is None and kp is None and i not in rec.get('ch', ()) and not c.get('recmean') and not capi and not c.get('check'):
                return orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=False, attn_mask=attn_mask)
            Lq, n, _ = query.shape; Sk = key.shape[0]
            q = (query @ Wq.T + bq).view(Lq, n, H, HD).permute(1, 2, 0, 3)
            k = (key @ Wk.T + bk).view(Sk, n, H, HD).permute(1, 2, 0, 3)
            if kp is not None:
                sel, K = kp; ke = (K @ Wk.T + bk).view(Sk, n, H, HD).permute(1, 2, 0, 3); k = t.where(sel[None, :, None, None], ke, k)
            v = (value @ Wv.T + bv).view(Sk, n, H, HD).permute(1, 2, 0, 3)
            lg = (q / math.sqrt(HD)) @ k.transpose(-1, -2)
            if attn_mask is not None: lg = lg + attn_mask.view(n, H, Lq, Sk)
            if key_padding_mask is not None: lg = lg.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
            w = t.softmax(lg, -1)
            ch = t.einsum('nhqd,chd->nhqc', w @ v, Wo)                                  # (n, H, Lq, C)
            realq = (~key_padding_mask).float() if i < 8 else t.ones(n, 1, device=query.device)
            if c.get('check'):
                o0 = orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=False, attn_mask=attn_mask)[0]
                d = (ch.sum(1) + bo - o0.permute(1, 0, 2)).abs() * realq[..., None]
                S.recon = max(S.recon, float(d.max()))
            if i in rec.get('ch', ()): S.st.setdefault('ch', {})[i] = S._fill(ch, realq) if i < 8 else ch
            if c.get('recmean'): S.hs[i] += (ch * realq[:, None, :, None]).sum((0, 2)).double(); S.hc[i] += float(realq.sum())
            if capi: S.cap = w[:, cap[1]].detach()                                       # (n, Lq, Sk)
            if mean is not None:
                sel, M = mean; ch = t.where(sel[None, :, None, None], M[None, :, None, :].expand_as(ch), ch)
            if rep is not None:
                sel, D = rep; ch = t.where(sel[None, :, None, None], D, ch)
            return (ch.sum(1) + bo).permute(1, 0, 2).contiguous(), None
        attn.forward = f

    def fix_means(self):
        self.HM = {i: (self.hs[i] / max(self.hc[i], 1)).float() for i in range(10)}
        self.MM = {i: (self.ms[i] / max(self.mc[i], 1)).float() for i in range(10)}
        self.AM = {j: (self.as_[j] / max(self.ac[j], 1)).float() for j in range(2)}

    def fwd(self, Jb, Pn):
        t = self.t; x, v, m = inputs_P(Jb, self.feat, Pn)
        self.qmask = t.from_numpy((m[:, 0] > 0).astype(np.float32)).to(self.dev)        # (n, Pn)
        with t.no_grad(): L = self.m(*(t.from_numpy(a).to(self.dev) for a in (x, v, m)))
        return L.float().cpu().numpy()

    def multi(self, J, mks, Jd=None, D=None, dctl=None, bs=BS, after=None):
        """recipient jets J under each control builder mk(idx, Pn, rec) -> ctl (None = clean); with donors (Jd, D: donor row per
        recipient) the donor batch is run first under dctl and its recordings `rec` are passed to the builders.
        Returns ([L (N, 10)] per builder, donor logits or None)."""
        from interp.part_core import sub
        N = len(J['mask']); mr = J['mask'].sum(1).astype(int)
        key = np.maximum(mr, Jd['mask'].sum(1).astype(int)[D]) if Jd is not None else mr
        order = np.argsort(key, kind='stable'); Ls = [np.zeros((N, 10), np.float32) for _ in mks]
        LD = np.zeros((N, 10), np.float32) if Jd is not None else None
        for s in range(0, N, bs):
            idx = order[s:s + bs]; Jb = sub(J, idx); Pn = int(max(key[idx].max(), 1)); rec = None
            if Jd is not None:
                self.pe_on = False; self.ctl = dctl or {}; self.st = {}
                LD[idx] = self.fwd(sub(Jd, D[idx]), Pn); rec = self.st; self.st = {}
            self.pe = None; self.pe_on = True
            for vv, mk in enumerate(mks):
                self.ctl = mk(idx, Pn, rec) if mk is not None else {}
                Ls[vv][idx] = self.fwd(Jb, Pn)
                if after is not None: after(vv, idx, self)
            self.pe_on = False; self.pe = None
        self.ctl = {}; return Ls, LD

    def run(self, J, mk=None, **kw):
        return self.multi(J, [mk], **kw)[0][0]

    def cls_part(self, x7, pm):
        t = self.t; n = x7.shape[1]
        with t.no_grad():
            cls = self.m.cls_token.expand(1, n, -1)
            for b in self.m.cls_blocks: cls = b(x7, x_cls=cls, padding_mask=pm)
            return self.m.fc(self.m.norm(cls).squeeze(0)).float()


# ------------------------------------------------------------------------------------------------ helpers
def sbc(L): return (L[:, HBB] - L[:, HCC]).astype(np.float64)
def sbl(L): return (L[:, HBB] - L[:, ZQQ]).astype(np.float64)


def auc(pos, neg):
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg): return None
    r = rankdata(np.concatenate([pos, neg])); return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Pair:
    """the jets of a class pair (a = H->bb, b = H->cc or Z->qq) inside a sample with labels y; sign +1 / -1; score function"""

    def __init__(self, y, kb, score, W):
        self.ia = np.where(y == HBB)[0]; self.ib = np.where(y == kb)[0]; self.idx = np.concatenate([self.ia, self.ib])
        self.sg = np.concatenate([np.ones(len(self.ia)), -np.ones(len(self.ib))]); self.score = score; self.W = W

    def u(self, L): return self.sg * self.score(L[self.idx])

    def acc(self, L): return float((self.u(L) > 0).mean())

    def auc(self, L): s = self.score(L); return auc(s[self.ia], s[self.ib])

    def du(self, L0, L1): return P.mstat(self.u(L1) - self.u(L0), self.W)

    def acc_ci(self, L): return P.ci(self.W.astype(np.float64) @ (self.u(L) > 0).astype(np.float64))


def eff(pair, L0, L1, ref=None):
    """du with CI, as a share of the unedited margin and (if ref = per-jet du of edit a) of the displacement-input effect"""
    d = pair.u(L1) - pair.u(L0); u0 = pair.u(L0); W = pair.W.astype(np.float64)
    e = P.mstat(d, W); bu = W @ u0; bd = W @ d
    e['share_margin'] = float(d.mean() / u0.mean()); e['share_margin_ci'] = P.ci(bd / bu)
    if ref is not None:
        br = W @ ref; e['share_input'] = float(d.mean() / ref.mean()); e['share_input_ci'] = P.ci(bd / br)
    return e


def strip_e(e): return {k: v for k, v in e.items() if k in ('v', 'ci', 'p0', 'share_margin', 'share_margin_ci', 'share_input', 'share_input_ci', 'p_holm')}


def charged(J): return (J['charge'] != 0) & (J['mask'] > 0)


def sig(J):
    return np.where(charged(J), np.abs(J['d0val']) / np.maximum(J['d0err'], 1e-6), -1.0)


def track_pool(J):
    from interp.part_core import kin
    pt = kin(J)[0]; ch = charged(J); pool = {}
    for q in (-1, 1):
        mk = ch & (np.sign(J['charge']) == q); lp = np.log(pt[mk]); o = np.argsort(lp)
        pool[q] = dict(lp=lp[o], **{k: J[k][mk][o] for k in IPK})
    return pool


def ed_ip(B, sel, pool, rng):
    """(d0val, d0err, dzval, dzerr) of the tracks sel (J, 128) bool set to a pool track of the same charge, nearest pT (random of 5)"""
    from interp.part_core import kin
    E = {k: (v.copy() if np.ndim(v) else v) for k, v in B.items()}; pt = kin(B)[0]
    for q in (-1, 1):
        mk = sel & (np.sign(B['charge']) == q)
        if not mk.any(): continue
        pl = pool[q]; n = len(pl['lp']); pos = np.searchsorted(pl['lp'], np.log(pt[mk])) + rng.integers(-2, 3, int(mk.sum()))
        pos = np.clip(pos, 0, n - 1)
        for k in IPK: E[k][mk] = pl[k][pos]
    return E                      # four-momenta untouched: no build pass (base and edit share the same rebuilt kinematics)


def topk_sel(J, k, largest=True):
    S = sig(J); ch = charged(J); out = np.zeros_like(ch)
    key = np.where(ch, S if largest else -S, -np.inf); o = np.argsort(-key, 1)[:, :k]
    r = np.arange(len(S))[:, None]; out[r, o] = True; return out & ch


def match_donors(pt_r, pt_d, rng, k=10):
    o = np.argsort(pt_d); s = pt_d[o]; pos = np.searchsorted(s, pt_r) + rng.integers(-k // 2, k // 2 + 1, len(pt_r))
    return o[np.clip(pos, 0, len(s) - 1)]


def block_matched(sel, rng):
    out = []
    for b in range(10):
        nb = sum(1 for x in sel if x[0] == b)
        if nb: out += [(b, h) for h in rng.choice(H, nb, replace=False)]
    return out


def emp_p(sel, rand):
    rand = np.asarray([r for r in rand if r is not None], float)
    return float((1 + (rand >= sel).sum()) / (1 + len(rand))) if len(rand) else None


def summ(a):
    a = np.asarray([x for x in a if x is not None], float)
    return dict(mean=float(a.mean()), q05=float(np.percentile(a, 5)), q95=float(np.percentile(a, 95)), min=float(a.min()), max=float(a.max()))


# ------------------------------------------------------------------------------------------------ control builders
def sel_mask(R, heads_in_block):
    t = R.t; s = t.zeros(H, dtype=t.bool, device=R.dev)
    for h in heads_in_block: s[h] = True
    return s


def by_block(heads):
    d = {}
    for (i, h) in heads: d.setdefault(i, []).append(h)
    return d


def mk_mean(R, heads):
    bb = by_block(heads)
    return lambda idx, Pn, rec: dict(mean={i: (sel_mask(R, hs), R.HM[i]) for i, hs in bb.items()})


def mk_rep(R, heads):
    bb = by_block(heads)
    return lambda idx, Pn, rec: dict(rep={i: (sel_mask(R, hs), rec['ch'][i]) for i, hs in bb.items()})


def mk_key(R, heads):
    bb = by_block(heads)
    return lambda idx, Pn, rec: dict(kpatch={i: (sel_mask(R, hs), rec['key'][i]) for i, hs in bb.items()})


def mk_circ(R, keep, neurons, pmlp_keep=False, all_neurons=False):
    """literal circuit: heads not in keep <- donor; particle MLPs <- donor; class-MLP neurons not in `neurons` <- donor"""
    keep = set(keep); abl = by_block([x for x in HEADS if x not in keep]); t = R.t
    km = {j: t.zeros(NU, dtype=t.bool, device=R.dev) for j in range(2)}
    for (j, u) in neurons: km[j][u] = True
    if all_neurons:
        for j in range(2): km[j][:] = True

    def f(idx, Pn, rec):
        c = dict(rep={i: (sel_mask(R, hs), rec['ch'][i]) for i, hs in abl.items()}, act_rep={j: (km[j], rec['act'][j]) for j in range(2)})
        if not pmlp_keep: c['mlp_rep'] = {i: rec['mlp'][i] for i in range(8)}
        return c
    return f


REC_ALL = dict(rec={'ch': set(range(10)), 'mlp': set(range(8)), 'act': {0, 1}})
REC_CH = dict(rec={'ch': set(range(10))})
REC_KEY = dict(rec={'key': set(range(10))})


# ------------------------------------------------------------------------------------------------ main
def main():
    import torch
    from interp.part_core import sub, cat, PRE, CL, sample, kin
    P.OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    R = dict(node='P65', seed=SEED, smoke=SMOKE, files=f'test_20M <sample>_{FILE}.root', NR=NR, NE=NE, NOTH=NOTH, NDON=NDON, NPOOL=NPOOL,
             NSUB=NSUB, NB=NB, NRS=NRS, NRJ=NRJ, NN=NN, KS=list(KS),
             note='planned files _127 do not exist in test_20M (_100.._119); _117 used')
    rank, ev, don = {}, [], []
    for k, c in enumerate(CL):
        main4 = k in (QCD, HBB, HCC, ZQQ)
        n = (NR + NE if main4 else NOTH) + NDON + (NPOOL if k == QCD else 0)
        J, rows = P.load_rows(P.TEST / f'{PRE[c]}_{FILE}.root', n, rng); perm = rng.permutation(len(rows)); o = 0
        if main4: rank[k] = P.rebuild(sub(J, np.sort(perm[:NR]))); o = NR
        ne = NE if main4 else NOTH; ev.append(sub(J, np.sort(perm[o:o + ne]))); o += ne
        don.append(sub(J, np.sort(perm[o:o + NDON]))); o += NDON
        if k == QCD: lpool = sub(J, np.sort(perm[o:o + NPOOL]))
        log(f'loaded {c}: {len(rows)} random rows of {PRE[c]}_{FILE}.root')
    EV = P.rebuild(cat(ev)); DON = P.rebuild(cat(don)); y = EV['label'].astype(int); yd = DON['label'].astype(int)
    LP = track_pool(lpool); SP = {k: track_pool(rank[k]) for k in rank}
    nch = charged(EV).sum(1)
    R['sample'] = dict(N_eval=int(len(y)), per_class=np.bincount(y, minlength=10).tolist(), N_donor=int(len(yd)), N_rank=int(sum(len(v['label']) for v in rank.values())),
                       light_pool_tracks=int(sum(len(v['lp']) for v in LP.values())),
                       tracks_per_jet={CL[k]: float(nch[y == k].mean()) for k in (QCD, HBB, HCC, ZQQ)},
                       frac_jets_ge6_tracks={CL[k]: float((nch[y == k] >= 6).mean()) for k in (QCD, HBB, HCC, ZQQ)},
                       n_S_gt3={CL[k]: float(((sig(EV) > 3).sum(1))[y == k].mean()) for k in (QCD, HBB, HCC, ZQQ)})
    log('sample', R['sample'])
    W_bc = P.boot_W(int(((y == HBB) | (y == HCC)).sum()), NB, rng); W_bl = P.boot_W(int(((y == HBB) | (y == ZQQ)).sum()), NB, rng)
    pe_bc, pe_bl = Pair(y, HCC, sbc, W_bc), Pair(y, ZQQ, sbl, W_bl)
    # edits on the evaluation set
    E = dict(a=ed_ip(EV, charged(EV), LP, rng), b=ed_ip(EV, topk_sel(EV, 3, True), LP, rng), c=ed_ip(EV, topk_sel(EV, 3, False), LP, rng))
    Ed = {k: (v.copy() if np.ndim(v) else v) for k, v in EV.items()}
    for k in rank:
        r = np.where(y == k)[0]; part = ed_ip(sub(EV, r), charged(sub(EV, r)), SP[k], rng)
        for kk in Ed:
            if np.ndim(Ed[kk]): Ed[kk][r] = part[kk]
    E['d'] = Ed
    other = np.isin(y, [QCD, HBB, HCC, ZQQ], invert=True)
    R['edit_checks'] = {k: dict(fourmom_identical=bool(all(np.array_equal(v[q], EV[q]) for q in ('px', 'py', 'pz', 'energy', 'charge', 'isChargedHadron', 'isPhoton'))),
                                ip_changed_frac=float((v['d0val'] != EV['d0val'])[charged(EV)].mean()),
                                neutral_ip_unchanged=bool(np.array_equal(v['d0val'][~charged(EV)], EV['d0val'][~charged(EV)])))
                        for k, v in E.items()}
    log('edit checks', R['edit_checks'])
    P1s = P.rebuild(sample(100 if SMOKE else 1000, 3000)); yP1 = P1s['label'].astype(int)
    R['models'] = {}
    for name, feat, untr in (('ParT_full', 'full', False), ('ParT_kin', 'kin', False), ('ParT_untrained', 'full', True)):
        t0 = time.time(); log(f'==== {name}'); M = Run(feat, untrained=untr); res = {}
        res['acc_P1'] = float((M.run(P1s).argmax(1) == yP1).mean())
        M.run(DON, lambda idx, Pn, rec: dict(recmean=True, check=True)); M.fix_means(); res['manual_attention_recon_max'] = M.recon
        full = name == 'ParT_full'
        if full:
            res.update(analyse_full(M, R, EV, y, DON, yd, rank, E, pe_bc, pe_bl, LP, rng))
        else:
            sel = np.concatenate([rng.choice(np.where(y == k)[0], min(NREFM, (y == k).sum()), replace=False) for k in (HBB, HCC, ZQQ)])
            res.update(analyse_ref(M, name, sub(EV, sel), {k: sub(v, sel) for k, v in E.items()}, rng))
        log(f'  acc P1 {res["acc_P1"]:.4f}; manual attention max |diff| {M.recon:.2e}')
        res['runtime_s'] = time.time() - t0; R['models'][name] = res
        FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'  wrote {FN} ({res["runtime_s"]:.0f} s)')
        del M; torch.cuda.empty_cache()
    R['runtime_s'] = time.time() - P.T0
    FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'done; wrote {FN}')


def acc_block(L, y):
    c = L.argmax(1)
    return dict(acc10=float((c == y).mean()), confusion=[[int(((y == i) & (c == j)).sum()) for j in range(10)] for i in range(10)])


def pair_metrics(L, y, W=True):
    e = {}
    for nm, kb, sc in (('bc', HCC, sbc), ('bl', ZQQ, sbl)):
        ia, ib = np.where(y == HBB)[0], np.where(y == kb)[0]; s = sc(L); u = np.concatenate([s[ia], -s[ib]])
        e[f'acc_{nm}'] = float((u > 0).mean()); e[f'auc_{nm}'] = auc(s[ia], s[ib]); e[f'margin_{nm}'] = float(u.mean())
    return e


def input_check(M, EV, E, y, pe_bc, pe_bl, L0=None):
    """edits a-d: L per edit; du_bc / du_bl (share of margin), per-class s_bc change, accuracies before -> after, McNemar"""
    L0 = M.run(EV) if L0 is None else L0; LE = {k: M.run(v) for k, v in E.items()}; out = dict(unedited=pair_metrics(L0, y))
    ref = pe_bc.u(LE['a']) - pe_bc.u(L0); refl = pe_bl.u(LE['a']) - pe_bl.u(L0)
    for k, L1 in LE.items():
        e = dict(du_bc=strip_e(eff(pe_bc, L0, L1, ref)), du_bl=strip_e(eff(pe_bl, L0, L1, refl)), after=pair_metrics(L1, y))
        for cls, nm in ((HBB, 'Hbb'), (HCC, 'Hcc'), (ZQQ, 'Zqq'), (QCD, 'QCD')):
            r = y == cls
            if r.any(): e[f'ds_bc_{nm}'] = float((sbc(L1[r]) - sbc(L0[r])).mean()); e[f'ds_bl_{nm}'] = float((sbl(L1[r]) - sbl(L0[r])).mean())
        e['mcnemar_bc_correct'] = P.mcnemar(pe_bc.u(L0) > 0, pe_bc.u(L1) > 0)
        e['calls_changed'] = float((L1.argmax(1) != L0.argmax(1)).mean()); e['max_abs_logit_diff'] = float(np.abs(L1 - L0).max())
        e['acc_bc_after_ci'] = pe_bc.acc_ci(L1)
        out[k] = e
        log(f'  input edit {k}: du_bc {e["du_bc"]["v"]:+.3f} {np.round(e["du_bc"]["ci"], 3).tolist()} (share of margin {e["du_bc"]["share_margin"]:.3f}) '
            f'du_bl {e["du_bl"]["v"]:+.3f}; acc_bc {out["unedited"]["acc_bc"]:.4f} -> {e["after"]["acc_bc"]:.4f}; AUC_bc {out["unedited"]["auc_bc"]:.4f} -> '
            f'{e["after"]["auc_bc"]:.4f}; acc_bl {out["unedited"]["acc_bl"]:.4f} -> {e["after"]["acc_bl"]:.4f}; ds_bc Hbb {e.get("ds_bc_Hbb", 0):+.3f} Hcc {e.get("ds_bc_Hcc", 0):+.3f}; '
            f'calls changed {e["calls_changed"]:.3f}')
    for a, b in (('b', 'c'), ('a', 'd')):
        da = pe_bc.u(LE[a]) - pe_bc.u(L0); db = pe_bc.u(LE[b]) - pe_bc.u(L0)
        out[f'{a}_minus_{b}'] = dict(du_bc=P.mstat(da - db, pe_bc.W), mcnemar=P.mcnemar(pe_bc.u(LE[b]) > 0, pe_bc.u(LE[a]) > 0))
        log(f'  edit {a} - {b}: du_bc {out[f"{a}_minus_{b}"]["du_bc"]["v"]:+.3f} {np.round(out[f"{a}_minus_{b}"]["du_bc"]["ci"], 3).tolist()}')
    return out, L0, LE


def analyse_full(M, R, EV, y, DON, yd, rank, E, pe_bc, pe_bl, LP, rng):
    from interp.part_core import sub, cat, kin
    out = {}; t = M.t
    L0 = M.run(EV, lambda idx, Pn, rec: dict(check=True)); out['eval'] = dict(acc_block(L0, y), **pair_metrics(L0, y))
    out['eval']['acc10_main4'] = float((L0.argmax(1) == y)[np.isin(y, [QCD, HBB, HCC, ZQQ])].mean())
    out['eval']['acc_bc_ci'] = pe_bc.acc_ci(L0)
    log(f'  eval: acc10 {out["eval"]["acc10"]:.4f} (4 main classes {out["eval"]["acc10_main4"]:.4f}); acc_bc {out["eval"]["acc_bc"]:.4f} AUC_bc {out["eval"]["auc_bc"]:.4f}; '
        f'acc_bl {out["eval"]["acc_bl"]:.4f} AUC_bl {out["eval"]["auc_bl"]:.4f}')
    ic, _, LE = input_check(M, EV, E, y, pe_bc, pe_bl, L0); out['input'] = ic
    # ---------------- ranking set scans
    RK = cat([rank[HBB], rank[HCC], rank[ZQQ]]); yr = RK['label'].astype(int); nr = len(yr)
    Wr_bc = P.boot_W(int(((yr == HBB) | (yr == HCC)).sum()), NB, rng); Wr_bl = P.boot_W(int(((yr == HBB) | (yr == ZQQ)).sum()), NB, rng)
    pr_bc, pr_bl = Pair(yr, HCC, sbc, Wr_bc), Pair(yr, ZQQ, sbl, Wr_bl)
    ptr = kin(RK)[0].sum(1)
    RKa = ed_ip(RK, charged(RK), LP, rng)
    LR0 = M.run(RK); LRa = M.run(RKa); out['rank'] = pair_metrics(LR0, yr)
    ref_bc = pr_bc.u(LRa) - pr_bc.u(LR0); ref_bl = pr_bl.u(LRa) - pr_bl.u(LR0)
    log(f'  ranking set N {nr}: acc_bc {out["rank"]["acc_bc"]:.4f}, acc_bl {out["rank"]["acc_bl"]:.4f}; input edit a du_bc {ref_bc.mean():+.3f} du_bl {ref_bl.mean():+.3f}')
    tt = time.time(); Lm, _ = M.multi(RK, [mk_mean(M, [h]) for h in HEADS]); log(f'  mean scan {time.time() - tt:.0f} s')
    tt = time.time(); Lk, _ = M.multi(RK, [mk_key(M, [h]) for h in HEADS], Jd=RKa, D=np.arange(nr), dctl=REC_KEY); log(f'  key scan {time.time() - tt:.0f} s')
    Lrep = {}; rank_check = {}
    for nm, kb in (('bc', HCC), ('bl', ZQQ)):
        isub = np.where(np.isin(yr, [HBB, kb]))[0]; Js = sub(RK, isub); ys_ = yr[isub]; pts_ = ptr[isub]; Ds = np.arange(len(isub))
        for ka, kd in ((HBB, kb), (kb, HBB)):
            ia, ib = np.where(ys_ == ka)[0], np.where(ys_ == kd)[0]; Ds[ia] = ib[match_donors(pts_[ia], pts_[ib], rng)]
        tt = time.time(); Ls_, LD_ = M.multi(Js, [mk_rep(M, [h]) for h in HEADS], Jd=Js, D=Ds, dctl=REC_CH)
        log(f'  resample scan ({nm} donors, N {len(isub)}) {time.time() - tt:.0f} s')
        Lfull = []
        for L_ in Ls_:
            Lf = LR0.copy(); Lf[isub] = L_; Lfull.append(Lf)        # jets outside the pair keep their clean logits (not used)
        Lrep[nm] = Lfull; rank_check[f'donor_logits_minus_clean_{nm}'] = float(np.abs(LD_ - LR0[isub][Ds]).max())
    sc = {}
    for nm, pr, ref in (('bc', pr_bc, ref_bc), ('bl', pr_bl, ref_bl)):
        em = [eff(pr, LR0, L, ref) for L in Lm]; er = [eff(pr, LR0, L, ref) for L in Lrep[nm]]; ek = [eff(pr, LR0, L, ref) for L in Lk]
        ph = P.holm([e['p0'] for e in em] + [e['p0'] for e in er])
        for j, e in enumerate(em + er): e['p_holm'] = ph[j]
        pk = P.holm([e['p0'] for e in ek])
        for j, e in enumerate(ek): e['p_holm'] = pk[j]
        sc[nm] = dict(mean={hlab(h): strip_e(e) for h, e in zip(HEADS, em)}, resample={hlab(h): strip_e(e) for h, e in zip(HEADS, er)},
                      key={hlab(h): strip_e(e) for h, e in zip(HEADS, ek)})
        sm = lambda es: dict(n_holm_sig=int(sum(e['p_holm'] < 0.05 for e in es)),
                             n_holm_sig_gt5pct=int(sum((e['p_holm'] < 0.05) and abs(e['share_margin']) > 0.05 for e in es)),
                             max_abs_share_margin=float(max(abs(e['share_margin']) for e in es)))
        sc[nm]['summary'] = dict(mean=sm(em), resample=sm(er), key=sm(ek), input_a_du=float(ref.mean()), margin=float(pr.u(LR0).mean()))
        log(f'  scan {nm}: margin {pr.u(LR0).mean():.3f}, input-a du {ref.mean():+.3f}; summary {sc[nm]["summary"]}')
    rk_bc = sorted(HEADS, key=lambda h: sc['bc']['resample'][hlab(h)]['v'])
    rk_key = sorted(HEADS, key=lambda h: sc['bc']['key'][hlab(h)]['v'])
    out['scan'] = sc; out['ranking'] = [hlab(h) for h in rk_bc]; out['ranking_key'] = [hlab(h) for h in rk_key]
    for h in rk_bc[:10]:
        a, b, k = sc['bc']['resample'][hlab(h)], sc['bc']['mean'][hlab(h)], sc['bc']['key'][hlab(h)]
        log(f'   {hlab(h)}: resample du_bc {a["v"]:+.3f} {np.round(a["ci"], 3).tolist()} (margin share {a["share_margin"]:.3f}, Holm p {a["p_holm"]:.2e}); '
            f'mean {b["v"]:+.3f}; key du_bc {k["v"]:+.3f} (input share {k.get("share_input", 0):.3f}); bl resample {sc["bl"]["resample"][hlab(h)]["v"]:+.3f}')
    out['rank_check'] = rank_check; log(f'  rank checks {rank_check}')
    # ---------------- class-MLP neurons (mean ablation, one at a time), ranking set
    tt = time.time(); nsc = neuron_scan(M, RK, LR0, pr_bc, pr_bl); out['neurons'] = nsc; log(f'  neuron scan {time.time() - tt:.0f} s')
    NEUR = [tuple(x) for x in nsc['top_bc']]
    # ---------------- held-out (evaluation rows): single-head resample of the top 8; joint ablations; key masking joint
    ptE = kin(EV)[0].sum(1); N = len(y); DE_bc = np.arange(N); DE_bl = np.arange(N)
    for ka, kb, D in ((HBB, HCC, DE_bc), (HCC, HBB, DE_bc), (HBB, ZQQ, DE_bl), (ZQQ, HBB, DE_bl)):
        ia, ib = np.where(y == ka)[0], np.where(y == kb)[0]; D[ia] = ib[match_donors(ptE[ia], ptE[ib], rng)]
    m3 = np.where(np.isin(y, [HBB, HCC, ZQQ]))[0]; E3 = sub(EV, m3); y3 = y[m3]; L03 = L0[m3]
    W3bc = P.boot_W(int(((y3 == HBB) | (y3 == HCC)).sum()), NB, rng); W3bl = P.boot_W(int(((y3 == HBB) | (y3 == ZQQ)).sum()), NB, rng)
    p3bc, p3bl = Pair(y3, HCC, sbc, W3bc), Pair(y3, ZQQ, sbl, W3bl)
    inv = -np.ones(N, int); inv[m3] = np.arange(len(m3)); D3bc, D3bl = inv[DE_bc[m3]], inv[DE_bl[m3]]
    La3 = LE['a'][m3]; r3bc = p3bc.u(La3) - p3bc.u(L03); r3bl = p3bl.u(La3) - p3bl.u(L03)
    top8 = rk_bc[:8]
    Lh, _ = M.multi(E3, [mk_rep(M, [h]) for h in top8], Jd=E3, D=D3bc, dctl=REC_CH)
    Lhl, _ = M.multi(E3, [mk_rep(M, [h]) for h in top8], Jd=E3, D=D3bl, dctl=REC_CH)
    out['heldout_single'] = {hlab(h): dict(bc=strip_e(eff(p3bc, L03, a, r3bc)), bl=strip_e(eff(p3bl, L03, b, r3bl))) for h, a, b in zip(top8, Lh, Lhl)}
    for h in top8:
        e = out['heldout_single'][hlab(h)]
        log(f'  held-out {hlab(h)}: du_bc {e["bc"]["v"]:+.3f} {np.round(e["bc"]["ci"], 3).tolist()} share margin {e["bc"]["share_margin"]:.3f}; du_bl {e["bl"]["v"]:+.3f}')
    # joint resample ablation of the top-k heads vs NRJ block-matched random sets (subsample for random)
    sets = {k: rk_bc[:k] for k in KS}
    Lj, _ = M.multi(E3, [mk_rep(M, sets[k]) for k in KS] + [mk_rep(M, HEADS)], Jd=E3, D=D3bc, dctl=REC_CH)
    Ljl, _ = M.multi(E3, [mk_rep(M, sets[k]) for k in KS] + [mk_rep(M, HEADS)], Jd=E3, D=D3bl, dctl=REC_CH)
    sub3 = np.sort(np.concatenate([rng.choice(np.where(y3 == k)[0], min(NSUB, (y3 == k).sum()), replace=False) for k in (HBB, HCC, ZQQ)]))
    Es = sub(E3, sub3); ys = y3[sub3]; inv2 = -np.ones(len(y3), int); inv2[sub3] = np.arange(len(sub3))
    # donors of the subsample must lie in the subsample: redraw within it
    pts = kin(Es)[0].sum(1); Dsbc = np.arange(len(ys))
    for ka, kb in ((HBB, HCC), (HCC, HBB), (ZQQ, HBB)):
        ia, ib = np.where(ys == ka)[0], np.where(ys == kb)[0]; Dsbc[ia] = ib[match_donors(pts[ia], pts[ib], rng)]
    Ws = P.boot_W(int(((ys == HBB) | (ys == HCC)).sum()), 10, rng); ps_bc = Pair(ys, HCC, sbc, Ws)
    Ls0 = L03[sub3]; rj = {}
    rand_sets = {k: [block_matched(sets[k], rng) for _ in range(NRJ)] for k in KS}
    Lsel_s, _ = M.multi(Es, [mk_rep(M, sets[k]) for k in KS], Jd=Es, D=Dsbc, dctl=REC_CH)
    Lrand_s, _ = M.multi(Es, [mk_rep(M, s) for k in KS for s in rand_sets[k]], Jd=Es, D=Dsbc, dctl=REC_CH)
    jt = {}
    for j, k in enumerate(KS + ('all80',)):
        e = dict(bc=strip_e(eff(p3bc, L03, Lj[j], r3bc)), bl=strip_e(eff(p3bl, L03, Ljl[j], r3bl)), acc_bc=p3bc.acc(Lj[j]), acc_bl=p3bl.acc(Ljl[j]))
        if k != 'all80':
            e['heads'] = [hlab(h) for h in sets[k]]
            dsel = float((ps_bc.u(Lsel_s[j]) - ps_bc.u(Ls0)).mean())
            dr = [float((ps_bc.u(Lrand_s[j * NRJ + r]) - ps_bc.u(Ls0)).mean()) for r in range(NRJ)]
            e['vs_random_sub'] = dict(selected_du_bc=dsel, random=summ(dr), p_more_negative=float((1 + sum(d <= dsel for d in dr)) / (1 + NRJ)))
        jt[str(k)] = e
        log(f'  joint resample ablation top-{k}: du_bc {e["bc"]["v"]:+.3f} (share of margin {e["bc"]["share_margin"]:.3f} {np.round(e["bc"]["share_margin_ci"], 3).tolist()}), '
            f'acc_bc {e["acc_bc"]:.4f}; du_bl {e["bl"]["v"]:+.3f} share {e["bl"]["share_margin"]:.3f}'
            + (f'; random (sub) mean {e["vs_random_sub"]["random"]["mean"]:+.3f} q05 {e["vs_random_sub"]["random"]["q05"]:+.3f} vs selected {e["vs_random_sub"]["selected_du_bc"]:+.3f}, p {e["vs_random_sub"]["p_more_negative"]:.3f}' if k != 'all80' else ''))
    out['joint_resample'] = jt
    # joint key masking on the evaluation rows (edit a as the key source)
    E3a = sub(E['a'], m3); grp = [('top' + str(k), sets[k]) for k in KS] + [('keytop8', rk_key[:8]), ('particle64', [h for h in HEADS if h[0] < 8]),
                                                                          ('class16', [h for h in HEADS if h[0] >= 8]), ('all80', HEADS)]
    Lkj, _ = M.multi(E3, [mk_key(M, g) for _, g in grp], Jd=E3a, D=np.arange(len(y3)), dctl=REC_KEY)
    out['joint_key'] = {}
    for (nm, g), L in zip(grp, Lkj):
        e = dict(bc=strip_e(eff(p3bc, L03, L, r3bc)), bl=strip_e(eff(p3bl, L03, L, r3bl)), acc_bc=p3bc.acc(L), heads=[hlab(h) for h in g] if len(g) <= 32 else len(g))
        out['joint_key'][nm] = e
        log(f'  joint key masking {nm}: du_bc {e["bc"]["v"]:+.3f} share of input-a effect {e["bc"]["share_input"]:.3f} {np.round(e["bc"]["share_input_ci"], 3).tolist()}; acc_bc {e["acc_bc"]:.4f}')
    out['heldout_ref'] = dict(input_a_du_bc=float(r3bc.mean()), input_a_du_bl=float(r3bl.mean()), margin_bc=float(p3bc.u(L03).mean()), margin_bl=float(p3bl.u(L03).mean()))
    # ---------------- literal circuit (evaluation rows, donors from the 10-class pool)
    out['circuit'] = circuit(M, EV, y, DON, yd, L0, sets, NEUR, Es, ys, sub3, m3, rng)
    # ---------------- figure data: top head's attention
    out['figure'] = fig_data(M, EV, E['a'], y, L0, rk_bc[0])
    return out


def neuron_scan(M, RK, LR0, pr_bc, pr_bl, G=32):
    from interp.part_core import sub
    t = M.t; N = len(RK['mask']); order = np.argsort(RK['mask'].sum(1), kind='stable')
    units = [(j, u) for j in range(2) for u in range(NU)]; dbc = np.zeros((len(units), N), np.float32); dbl = np.zeros((len(units), N), np.float32)
    S0 = np.zeros((N, 10), np.float32); chk = 0.0
    for s in range(0, N, BS):
        idx = order[s:s + BS]; Jb = sub(RK, idx); Pn = int(Jb['mask'].sum(1).max()); n = len(idx)
        M.ctl = dict(capx=True); S0[idx] = M.fwd(Jb, Pn); x7 = M.x7; pm = M.qmask < 0.5; M.ctl = {}
        Lc = M.cls_part(x7, pm).cpu().numpy(); chk = max(chk, float(np.abs(Lc - S0[idx]).max()))
        for g in range(0, len(units), G):
            ug = units[g:g + G]; ng = len(ug)
            xr = x7.repeat(1, ng, 1); pmr = pm.repeat(ng, 1)
            rows = t.arange(n * ng, device=M.dev); uidx = t.tensor([u for (_, u) in ug], device=M.dev).repeat_interleave(n)
            jj = t.tensor([j for (j, _) in ug], device=M.dev).repeat_interleave(n)
            M.ctl = dict(act_mean_rows={j: (rows[jj == j], uidx[jj == j]) for j in range(2) if bool((jj == j).any())})
            L = M.cls_part(xr, pmr).cpu().numpy().reshape(ng, n, 10); M.ctl = {}
            for q in range(ng):
                dbc[g + q, idx] = sbc(L[q]) - sbc(S0[idx]); dbl[g + q, idx] = sbl(L[q]) - sbl(S0[idx])
    out = dict(check_cls_part_max=chk, check_clean_vs_scan_max=float(np.abs(S0 - LR0).max())); log(f'  neuron-scan checks {out}')
    for nm, pr, d in (('bc', pr_bc, dbc), ('bl', pr_bl, dbl)):
        u0 = pr.u(LR0); es = []
        for q in range(len(units)):
            x = pr.sg * d[q, pr.idx]; e = P.mstat(x, pr.W); e['share_margin'] = float(x.mean() / u0.mean()); es.append(e)
        ph = P.holm([e['p0'] for e in es])
        for e, p in zip(es, ph): e['p_holm'] = p
        o = np.argsort([e['v'] for e in es])
        out[nm] = dict(n_holm_sig=int(sum(p < 0.05 for p in ph)), n_holm_sig_neg=int(sum((p < 0.05) and e['v'] < 0 for e, p in zip(es, ph))),
                       n_holm_sig_gt5pct=int(sum((p < 0.05) and abs(e['share_margin']) > 0.05 for e, p in zip(es, ph))),
                       top20=[dict(unit=f'b{units[q][0] + 1}n{units[q][1]}', **strip_e(es[q])) for q in o[:20]],
                       sum_abs_share=float(sum(abs(e['share_margin']) for e in es)))
        if nm == 'bc':
            out['top_bc'] = [list(units[q]) for q in o[:NN] if es[q]['p_holm'] < 0.05 and es[q]['v'] < 0]
        log(f'  neurons {nm}: Holm-sig {out[nm]["n_holm_sig"]} (negative {out[nm]["n_holm_sig_neg"]}, >5 % of margin {out[nm]["n_holm_sig_gt5pct"]}); '
            f'top {[(x["unit"], round(x["v"], 3), round(x["share_margin"], 3)) for x in out[nm]["top20"][:6]]}')
    return out


def circuit(M, EV, y, DON, yd, L0, sets, NEUR, Es, ys, sub3, m3, rng):
    from interp.part_core import kin
    out = dict(neurons=[f'b{j + 1}n{u}' for j, u in NEUR], n_neurons=len(NEUR))
    ptE, ptD = kin(EV)[0].sum(1), kin(DON)[0].sum(1); D = match_donors(ptE, ptD, rng)
    Wbc = P.boot_W(int(((y == HBB) | (y == HCC)).sum()), NB, rng); pbc = Pair(y, HCC, sbc, Wbc)
    Wbl = P.boot_W(int(((y == HBB) | (y == ZQQ)).sum()), NB, rng); pbl = Pair(y, ZQQ, sbl, Wbl)
    allc = [h for h in HEADS]
    var = [(f'k{k}', mk_circ(M, sets[k], NEUR)) for k in KS] + [
        ('k0_neurons_only', mk_circ(M, [], NEUR)), ('k8_no_neurons', mk_circ(M, sets[8], [])), ('k8_all_neurons', mk_circ(M, sets[8], [], all_neurons=True)),
        ('k8_particle_mlps_kept', mk_circ(M, sets[8], NEUR, pmlp_keep=True)), ('k32_particle_mlps_kept', mk_circ(M, sets[32], NEUR, pmlp_keep=True)),
        ('all80_neurons_only_particle_mlps_ablated', mk_circ(M, allc, NEUR)), ('all80_all_neurons_particle_mlps_ablated', mk_circ(M, allc, [], all_neurons=True)),
        ('nothing_kept', mk_circ(M, [], [])), ('everything_kept_via_hooks', mk_circ(M, allc, [], pmlp_keep=True, all_neurons=True))]
    Ls, LD = M.multi(EV, [f for _, f in var], Jd=DON, D=D, dctl=REC_ALL)
    sel = {}
    for (nm, _), L in zip(var, Ls):
        e = dict(acc_block(L, y), **pair_metrics(L, y)); e['acc_bc_ci'] = pbc.acc_ci(L); e['acc_bl_ci'] = pbl.acc_ci(L)
        e['acc_bc_share_of_full'] = (e['acc_bc'] - 0.5) / (pair_metrics(L0, y)['acc_bc'] - 0.5)
        e['n_distinct_calls'] = int(len(np.unique(L.argmax(1)))); e['pred_share'] = (np.bincount(L.argmax(1), minlength=10) / len(L)).tolist()
        sel[nm] = e
        log(f'  circuit {nm}: acc_bc {e["acc_bc"]:.4f} {np.round(e["acc_bc_ci"], 4).tolist()} AUC_bc {e["auc_bc"]:.4f}; acc_bl {e["acc_bl"]:.4f} AUC_bl {e["auc_bl"]:.4f}; '
            f'acc10 {e["acc10"]:.4f}; distinct calls {e["n_distinct_calls"]}')
    out['selected'] = sel; out['heads'] = {f'k{k}': [hlab(h) for h in sets[k]] for k in KS}
    out['checks'] = dict(nothing_kept_minus_donor_max=float(np.abs(Ls[[n for n, _ in var].index('nothing_kept')] - LD).max()),
                         everything_kept_minus_clean_max=float(np.abs(Ls[-1] - L0).max()),
                         donor_acc10=float((LD.argmax(1) == y).mean()), donor_class_share=(np.bincount(yd[D], minlength=10) / len(D)).tolist())
    log(f'  circuit checks {out["checks"]}')
    # random block-matched head sets (same neurons) on the subsample
    idx_s = m3[sub3]; Esub = {k: (v[idx_s] if np.ndim(v) else v) for k, v in EV.items()}; Dsub = D[idx_s]
    rs = {k: [block_matched(sets[k], rng) for _ in range(NRS)] for k in KS}
    mks = [mk_circ(M, sets[k], NEUR) for k in KS] + [mk_circ(M, s, NEUR) for k in KS for s in rs[k]] + [mk_circ(M, [], [])]
    Lr, _ = M.multi(Esub, mks, Jd=DON, D=Dsub, dctl=REC_ALL); nk = len(KS)
    rnd = {}
    for j, k in enumerate(KS):
        s0 = pair_metrics(Lr[j], ys); rr = [pair_metrics(Lr[nk + j * NRS + r], ys) for r in range(NRS)]
        rnd[f'k{k}'] = {m: dict(selected=s0[m], random=summ([x[m] for x in rr]), p=emp_p(s0[m], [x[m] for x in rr])) for m in ('acc_bc', 'auc_bc', 'acc_bl', 'auc_bl')}
        rnd[f'k{k}']['random_sets'] = [[hlab(h) for h in s] for s in rs[k][:5]]
        v = rnd[f'k{k}']
        log(f'  circuit k{k} vs {NRS} random (sub): acc_bc {v["acc_bc"]["selected"]:.4f} (random mean {v["acc_bc"]["random"]["mean"]:.4f}, q95 {v["acc_bc"]["random"]["q95"]:.4f}, p {v["acc_bc"]["p"]:.3f}); '
            f'AUC_bc {v["auc_bc"]["selected"]:.4f} (random mean {v["auc_bc"]["random"]["mean"]:.4f}, p {v["auc_bc"]["p"]:.3f})')
    rnd['floor_sub'] = pair_metrics(Lr[-1], ys); out['random'] = rnd
    return out


def fig_data(M, EV, EA, y, L0, top):
    from interp.part_core import sub, kin
    i, h = top; ib = np.where((y == HBB) & (L0.argmax(1) == HBB))[0]; S = sig(EV)
    shares = {}
    for nm, J in (('unedited', EV), ('edit_a', EA)):
        Jb_ = sub(J, ib); Sb = sig(sub(EV, ib)); nt = charged(sub(EV, ib)).sum(1); ok = nt >= 6
        hi = topk_sel(sub(EV, ib), 3, True); lo = topk_sel(sub(EV, ib), 3, False); A = np.zeros((len(ib), 128), np.float32)

        def after(vv, idx, S_, A=A):
            w = S_.cap; off = 0 if i < 8 else 1
            a = w[:, 0, off:] if i >= 8 else (w * S_.qmask[:, :, None]).sum(1) / S_.qmask.sum(1)[:, None]
            A[idx, :a.shape[1]] = a.cpu().numpy()
        M.multi(Jb_, [lambda idx, Pn, rec: dict(cap=(i, h))], after=after)
        A = A * (sub(EV, ib)['mask'] > 0)
        shares[nm] = dict(n_jets=int(ok.sum()), top3_S=float((A * hi).sum(1)[ok].mean()), bottom3_S=float((A * lo).sum(1)[ok].mean()),
                          all_tracks=float((A * charged(sub(EV, ib))).sum(1)[ok].mean()), count_share_3=float((3 / np.maximum(sub(EV, ib)['mask'].sum(1), 1))[ok].mean()))
        if nm == 'unedited': A0 = A
    log(f'  attention of {hlab(top)} on H->bb jets: {shares}')
    cand = [j for j in range(len(ib)) if (S[ib[j]] > 5).sum() >= 3 and charged(EV)[ib[j]].sum() >= 8]
    j = cand[0] if cand else 0; r = ib[j]; m = EV['mask'][r] > 0; pt, de, dp, _ = kin(sub(EV, [r]))
    jet = dict(row=int(r), head=hlab(top), pt=pt[0][m].tolist(), deta=de[0][m].tolist(), dphi=dp[0][m].tolist(), S=S[r][m].tolist(),
               charge=EV['charge'][r][m].tolist(), attention=A0[j][m].tolist(), jet_pt=float(EV['jet_pt'][r]), s_bc=float(sbc(L0[[r]])[0]),
               d0=EV['d0val'][r][m].tolist(), d0err=EV['d0err'][r][m].tolist())
    return dict(shares=shares, jet=jet)


def analyse_ref(M, name, EVs, Es, rng):
    y = EVs['label'].astype(int); out = {}
    W1 = P.boot_W(int(((y == HBB) | (y == HCC)).sum()), NB, rng); W2 = P.boot_W(int(((y == HBB) | (y == ZQQ)).sum()), NB, rng)
    pbc, pbl = Pair(y, HCC, sbc, W1), Pair(y, ZQQ, sbl, W2)
    L0 = M.run(EVs); out['eval'] = pair_metrics(L0, y); out['eval']['acc_bc_ci'] = pbc.acc_ci(L0)
    log(f'  {name} (N {len(y)}): acc_bc {out["eval"]["acc_bc"]:.4f} AUC_bc {out["eval"]["auc_bc"]:.4f}; acc_bl {out["eval"]["acc_bl"]:.4f}')
    ic, _, LE = input_check(M, EVs, Es, y, pbc, pbl, L0); out['input'] = ic
    if M.feat == 'full':
        ref = pbc.u(LE['a']) - pbc.u(L0)
        Lk, _ = M.multi(EVs, [mk_key(M, [h]) for h in HEADS], Jd=Es['a'], D=np.arange(len(y)), dctl=REC_KEY)
        ek = [eff(pbc, L0, L, ref) for L in Lk]
        out['key'] = {hlab(h): strip_e(e) for h, e in zip(HEADS, ek)}
        out['key_summary'] = dict(max_abs_du=float(max(abs(e['v']) for e in ek)), sum_du=float(sum(e['v'] for e in ek)), input_a_du=float(ref.mean()),
                                  top5=sorted([(hlab(h), e['v']) for h, e in zip(HEADS, ek)], key=lambda x: x[1])[:5])
        log(f'  {name} key scan: {out["key_summary"]}')
    else:
        out['key_note'] = 'no displacement inputs: the edited and unedited inputs are identical (max |logit diff| in input.a)'
    return out


if __name__ == '__main__':
    main()
