"""P37 — top-quark (t->bqq') head circuit in ParT (loop C round C8, exploratory; claim C17, new).

Question. Is ParT's t->bqq' call carried by a few attention heads (so that keeping only those heads keeps the top call far above
random head sets of the same size), or is it spread over many heads (as P9/P13 found for W vs Z)?

Models: ParT_full, ParT_kin (replication; released weights, models/part); untrained ParT_full (every weight redrawn from a
Gaussian with the trained tensor's mean and spread, seed 37) as the scan null.
Sample: JetClass test_20M <sample>_119.root, numpy default_rng(3737), random rows (never first-k):
  EVAL  1,000 jets per class (10 classes, N 10,000): accuracy, literal circuits, edits;
  REF     200 jets per class (N 2,000, disjoint rows): the mean head outputs and mean MLP outputs used for mean ablation;
  SCAN  1,000 further TTBar (t->bqq') jets (disjoint rows): the per-head scan, so that head selection and circuit evaluation use
        different jets.
  P1 reference: rows 3,000-4,000 of each class file _100 (part_core.sample), known accuracy ParT_full 85.93 %, ParT_kin 74.47 %.
Every jet passes once through part_core.build (eta = jet_eta + sign(jet_eta) deta).
Heads: 8 particle blocks (p1..p8) and 2 class-attention blocks (b1, b2), 8 heads each = 80 heads. A head's output is its
contribution to the attention module output, sum_h (A_h V_h) W_O[h] (verified: sum over heads + out_proj bias = module output).
Mean ablation of a head = its output at every token replaced by its mean over the REF set's real tokens (class blocks: over the
class token). Mean ablation of an MLP = the fc2 output of that block replaced by its mean over REF real tokens.
Step 1 (scan, SCAN jets, 1,000 true t->bqq'): per head, mean-ablate it alone; per jet
  d_QCD = Delta (logit_t - logit_QCD), d_W = Delta (logit_t - logit_W) (= Delta log p_t/p_QCD, Delta log p_t/p_W).
  Mean with 95 % bootstrap interval (1,000 resamples), sign-flip permutation p (20,000 flips; H0: ablation leaves the contrast
  unchanged on average), Holm over 80 heads x 2 contrasts = 160 tests.
Step 2 (literal circuits, EVAL): heads ranked by the mean of their two single-head effects (most negative first; Holm-significant
  negative heads first). Circuit k = the first k heads. Variant 'literal' (user definition): only the circuit heads kept, every
  other head AND all 10 MLPs mean-ablated (embeddings, pair embedding, LayerNorms, class token and final linear layer kept).
  Variant 'heads_only' (labelled): other heads mean-ablated, MLPs kept. k = 0, 1, 3, 5, 10, 20, 80 (80 = all heads kept).
  Metrics: t->bqq' recall, binary t-vs-QCD accuracy (logit_t > logit_QCD on true t and true QCD jets), class-balanced 10-class
  accuracy, 10-class accuracy, confusion. Null: NRAND random head sets of the same size, evaluated on a random 200-per-class
  subset SUB of EVAL (the circuit also evaluated on SUB); empirical p = (1 + #random >= circuit) / (1 + NRAND).
Step 3 (causal content, EVAL true t->bqq' jets): exclusive kT clustering (d_ij = min(pT_i^2, pT_j^2) DR_ij^2, E-scheme, rapidity-phi,
  no beam distance) of the jet's particles down to 3 subjets; W pair = the subjet pair with invariant mass nearest 80.4 GeV;
  b-prong = the third subjet's particles.
  edit a: the b-prong's particles replaced by one particle at their four-momentum sum (placed in the slot of the hardest of them;
          charge / PID / impact-parameter features of that hardest particle kept). Jet four-momentum unchanged.
  edit b (control): a cluster of W-prong particles with the same summed pT merged the same way (seed = random W-prong particle,
          then its nearest W-prong neighbours in DR until the scalar pT sum reaches the b-prong's).
  edit b2 (control): the same with the b-prong's particle count instead of its pT (full-network change only).
  Full change: Delta lo_t (lo_t = logit_t - logsumexp of the other 9 logits = log p_t/(1-p_t)) and Delta(logit_t - logit_QCD),
  per jet, edited - unedited. Patching: the unedited jet's run with the circuit heads' outputs replaced by their outputs in the
  edited run (tokens aligned by particle; tokens removed by the merge keep their unedited outputs); share = mean patched change /
  mean full change (bootstrap); null: NRAND random head sets of the same size; empirical p. tau32 = tau3/tau2 (beta 1, exclusive-kT
  axes) before and after.
  ./sync.sh && ./part_job.sh c-p37-run 'python -m interp.p37_top_circuit && python -m interp.p37_top_circuit --plot'
  -> $AI_RESULTS/part/p37_top_circuit[_smoke].json, results/part/p37_top_circuit.png
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--plot', action='store_true')
ap.add_argument('--nrand', type=int, default=None); ap.add_argument('--nsub', type=int, default=None)
ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED = 3737
NEVAL = 40 if SMOKE else 1000
NREF = 20 if SMOKE else 200
NSCAN = 60 if SMOKE else 1000
NSUB = ARGS.nsub or (20 if SMOKE else 200)
NB = 100 if SMOKE else 1000
NPERM = 2000 if SMOKE else 20000
NRAND = ARGS.nrand or (4 if SMOKE else 100)
KS = [0, 1, 3, 5, 10, 20, 80]
KPATCH = [5, 10]
BS = 250
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TAG = '_smoke' if SMOKE else ''
FN = OUTD / f'p37_top_circuit{TAG}.json'
QCD, WQQ, TOP = 0, 7, 8
C, H, HD = 128, 8, 16
MW = 80.4
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


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def signflip_p(x, S):
    """two-sided sign-flip permutation p of mean(x) = 0 (S: (nperm, n) random +-1)"""
    x = np.asarray(x, np.float64); obs = abs(x.mean()); null = np.abs(S @ x) / len(x)
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(S)))


def mstat(x, W, S=None):
    x = np.asarray(x, np.float64); bs = W.astype(np.float64) @ x
    e = dict(v=float(x.mean()), ci=ci(bs), se=float(bs.std()))
    if S is not None: e['p_perm'] = signflip_p(x, S)
    return e


def mcnemar(a, b):
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


def lodds(L, k=TOP):
    L = np.asarray(L, np.float64); o = np.delete(L, k, 1); m = o.max(1)
    return L[:, k] - m - np.log(np.exp(o - m[:, None]).sum(1))


def ctr(L, a=TOP, b=QCD):
    L = np.asarray(L, np.float64); return L[:, a] - L[:, b]


def acc_block(L, y, W=None, conf=True):
    c = L.argmax(1); rec = [float((c[y == k] == k).mean()) if (y == k).any() else None for k in range(10)]
    tq = np.isin(y, [QCD, TOP]); bin_ok = ((L[:, TOP] > L[:, QCD]) == (y == TOP))[tq]
    e = dict(acc10=float((c == y).mean()), balanced=float(np.mean([r for r in rec if r is not None])), recall=rec,
             top_recall=rec[TOP], bin_tq=float(bin_ok.mean()))
    if conf: e['confusion'] = [[int(((y == i) & (c == j)).sum()) for j in range(10)] for i in range(10)]
    if W is not None:
        e['acc10_ci'] = ci(W @ (c == y).astype(np.float32))
    return e


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


def rebuild(J):
    from interp.part_core import build, kin
    return build(J, *kin(J))


# ------------------------------------------------------------------------------------------------ kT clustering, N-subjettiness
def _ypp(P):
    pt = np.hypot(P[..., 0], P[..., 1]); E, pz = P[..., 3], P[..., 2]
    y = 0.5 * np.log(np.maximum(E + pz, 1e-12) / np.maximum(E - pz, 1e-12)); return pt, y, np.arctan2(P[..., 1], P[..., 0])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def kt_exclusive(p4, stops=(3, 2)):
    """exclusive kT (no beam distance) of particles p4 (n, 4) down to each n in stops -> {n: (members [list], P (n, 4))}"""
    P = np.asarray(p4, np.float64).copy(); n = len(P); alive = np.ones(n, bool); memb = [[i] for i in range(n)]; out = {}
    pt, y, ph = _ypp(P); pt2 = pt ** 2
    D = np.minimum(pt2[:, None], pt2[None]) * ((y[:, None] - y[None]) ** 2 + _wrap(ph[:, None] - ph[None]) ** 2)
    np.fill_diagonal(D, np.inf); na = n
    if na in stops: out[na] = ([list(m) for m in memb], P.copy())
    while na > min(stops):
        k = int(np.argmin(D)); i, j = divmod(k, n)
        if i > j: i, j = j, i
        P[i] += P[j]; memb[i] = memb[i] + memb[j]; alive[j] = False; D[j, :] = np.inf; D[:, j] = np.inf
        a, b_, c_ = _ypp(P[i]); pt2[i], y[i], ph[i] = a ** 2, b_, c_
        row = np.minimum(pt2[i], pt2) * ((y[i] - y) ** 2 + _wrap(ph[i] - ph) ** 2); row[~alive] = np.inf; row[i] = np.inf
        D[i, :] = row; D[:, i] = row; na -= 1
        if na in stops:
            al = np.where(alive)[0]; out[na] = ([list(memb[q]) for q in al], P[al].copy())
    return out


def tau(p4, axes):
    pt, y, ph = _ypp(np.asarray(p4, np.float64)); ay, ap_ = _ypp(np.asarray(axes, np.float64))[1:]
    dr = np.sqrt((y[:, None] - ay[None]) ** 2 + _wrap(ph[:, None] - ap_[None]) ** 2)
    return float((pt * dr.min(1)).sum() / pt.sum())


def p4_of(B, j):
    n = int(B['mask'][j].sum()); return np.stack([B[k][j, :n] for k in ('px', 'py', 'pz', 'energy')], 1).astype(np.float64)


def m_of(P):
    P = np.asarray(P, np.float64); s = P.sum(0); return float(np.sqrt(max(s[3] ** 2 - s[0] ** 2 - s[1] ** 2 - s[2] ** 2, 0)))


def tau32_of(p4):
    if len(p4) < 4: return np.nan
    o = kt_exclusive(p4, (3, 2)); t3 = tau(p4, o[3][1]); t2 = tau(p4, o[2][1]); return t3 / t2 if t2 > 0 else np.nan


# ------------------------------------------------------------------------------------------------ merge edit
PK = ['px', 'py', 'pz', 'energy', 'deta', 'dphi', 'd0val', 'd0err', 'dzval', 'dzerr', 'charge', 'isChargedHadron', 'isNeutralHadron',
      'isPhoton', 'isElectron', 'isMuon', 'mask']


def merge_edit(B, groups):
    """each jet's particles in groups[j] (indices into B's pT-sorted slots, >= 2) replaced by one particle at their four-momentum
    sum, placed in the slot of the hardest of them (other features of that particle kept); the remaining particles keep their
    order and are left-packed. Returns (E, map (N, 128): base slot -> edited slot, -1 if removed / padding, ok (N,))."""
    N = len(B['mask']); E = {k: (v.copy() if np.ndim(v) else v) for k, v in B.items()}
    mp = np.full((N, 128), -1, np.int32); ok = np.zeros(N, bool)
    for j in range(N):
        n = int(B['mask'][j].sum()); g = groups[j]
        if g is None or len(g) < 2:
            mp[j, :n] = np.arange(n); continue
        g = np.sort(np.asarray(g, int)); keep, rem = g[0], g[1:]
        s = {k: float(B[k][j, g].astype(np.float64).sum()) for k in ('px', 'py', 'pz', 'energy')}
        pt = math.hypot(s['px'], s['py']); eta = math.asinh(s['pz'] / pt); phi = math.atan2(s['py'], s['px'])
        sg = -1.0 if B['jet_eta'][j] < 0 else 1.0
        km = np.ones(n, bool); km[rem] = False; idx = np.where(km)[0]
        for k in PK:
            row = B[k][j].astype(np.float64).copy()
            if k in s: row[keep] = s[k]
            elif k == 'deta': row[keep] = sg * (eta - float(B['jet_eta'][j]))
            elif k == 'dphi': row[keep] = _wrap(phi - float(B['jet_phi'][j]))
            new = np.zeros(128); new[:len(idx)] = row[idx]; E[k][j] = new.astype(np.float32)
        mp[j, idx] = np.arange(len(idx)); ok[j] = True
    E['jet_pt'] = np.hypot(E['px'].sum(1), E['py'].sum(1)).astype(np.float32); E['jet_energy'] = E['energy'].sum(1).astype(np.float32)
    return E, mp, ok


def prongs(B, rng):
    """per jet: 3 exclusive-kT subjets; W pair (mass nearest 80.4 GeV); b-prong members; pT-matched and count-matched W-prong clusters"""
    N = len(B['mask']); info = dict(gb=[None] * N, gw=[None] * N, gw2=[None] * N, mjet=np.full(N, np.nan), mW=np.full(N, np.nan),
                                   nb=np.zeros(N, int), nw=np.zeros(N, int), nw2=np.zeros(N, int), ptb=np.full(N, np.nan),
                                   ptw=np.full(N, np.nan), tau32=np.full(N, np.nan), lab=[None] * N)
    for j in range(N):
        p4 = p4_of(B, j); n = len(p4)
        if n < 6: continue
        o = kt_exclusive(p4, (3, 2)); memb, S3 = o[3]
        pairs = [(0, 1), (0, 2), (1, 2)]; mm = [m_of(S3[list(pq)]) for pq in pairs]; w = int(np.argmin(np.abs(np.array(mm) - MW)))
        bi = 3 - sum(pairs[w]); gb = memb[bi]; wm = memb[pairs[w][0]] + memb[pairs[w][1]]
        info['mjet'][j] = m_of(p4); info['mW'][j] = mm[w]; info['tau32'][j] = tau(p4, S3) / max(tau(p4, o[2][1]), 1e-12)
        lab = np.zeros(n, int); lab[memb[pairs[w][0]]] = 1; lab[memb[pairs[w][1]]] = 2; info['lab'][j] = lab
        pt = np.hypot(p4[:, 0], p4[:, 1]); ptb = pt[gb].sum(); info['ptb'][j] = ptb; info['nb'][j] = len(gb)
        if len(gb) >= 2: info['gb'][j] = gb
        wm = np.asarray(wm); seed = rng.choice(wm); _, y, ph = _ypp(p4)
        dr = np.hypot(y[wm] - y[seed], _wrap(ph[wm] - ph[seed])); order = wm[np.argsort(dr, kind='stable')]
        cs = np.cumsum(pt[order])
        if cs[-1] >= ptb:
            q = int(np.searchsorted(cs, ptb)) + 1
            if q >= 2 and q < len(wm) + 1: info['gw'][j] = order[:q].tolist(); info['nw'][j] = q; info['ptw'][j] = cs[q - 1]
        if len(gb) >= 2 and len(order) >= len(gb): info['gw2'][j] = order[:len(gb)].tolist(); info['nw2'][j] = len(gb)
    return info


# ------------------------------------------------------------------------------------------------ ParT with head / MLP controls
class HRun:
    """ParT with per-head mean ablation, per-head donor replacement at aligned tokens, MLP (fc2 output) mean ablation, and the
    recording of means / head outputs. Attention modules 0-7 = particle blocks p1..p8, 8-9 = class blocks b1, b2. ctl per batch:
      mean {i: (sel (H,) float, M (H, C))}; mlp {i: Mf (C,)}; rep {i: (sel (H,), D (n, H, Lq, C), pos (n, Lq) float)};
      recmean True; recch set(i)"""

    def __init__(self, feat, untrained=False):
        import torch
        from interp.part_core import ParTNetwork
        self.t, self.feat = torch, feat
        self.net = ParTNetwork(feat, untrained=untrained, seed=37); self.m = self.net.model; self.dev = self.net.device; self.m.eval()
        self.ctl = {}; self.out = {}; self.recon = 0.0; self.curq = {}
        self.msum = {i: torch.zeros(H, C, dtype=torch.float64, device=self.dev) for i in range(10)}; self.mcnt = {i: 0.0 for i in range(10)}
        self.fsum = {i: torch.zeros(C, dtype=torch.float64, device=self.dev) for i in range(10)}; self.fcnt = {i: 0.0 for i in range(10)}
        blocks = list(self.m.blocks) + list(self.m.cls_blocks)
        for i, b in enumerate(blocks):
            self._wrap(i, b.attn); b.fc2.register_forward_hook(lambda mod, inp, out, i=i: self._mlp(i, out))

    def _mlp(self, i, out):
        c = self.ctl
        if c.get('recmean'):
            q = self.curq[i]; self.fsum[i] += (out.permute(1, 0, 2) * q[..., None]).sum((0, 1)).double(); self.fcnt[i] += float(q.sum())
        Mf = c.get('mlp', {}).get(i)
        if Mf is not None: return Mf[None, None, :].expand(out.shape).clone()
        return None

    def _wrap(self, i, attn):
        orig, S, t = attn.forward, self, self.t

        def f(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
            c = S.ctl; mean = c.get('mean', {}).get(i); rep = c.get('rep', {}).get(i); recm = c.get('recmean'); recch = i in c.get('recch', ())
            Lq, n, _ = query.shape; Sk = key.shape[0]
            realq = (~key_padding_mask).float() if i < 8 else t.ones(n, 1, device=query.device)
            S.curq[i] = realq
            if mean is None and rep is None and not recm and not recch:
                return orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=False, attn_mask=attn_mask)
            out, w = orig(query, key, value, key_padding_mask=key_padding_mask, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
            v = (value @ attn.in_proj_weight[2 * C:].T + attn.in_proj_bias[2 * C:]).view(Sk, n, H, HD).permute(1, 2, 0, 3)
            ch = t.einsum('nhqd,chd->nhqc', w @ v, attn.out_proj.weight.view(C, H, HD))                  # (n, H, Lq, C)
            if recm or recch:
                S.recon = max(S.recon, float(((ch.sum(1) + attn.out_proj.bias - out.permute(1, 0, 2)) * realq[..., None]).abs().max()))
            if recm:
                S.msum[i] += (ch * realq[:, None, :, None]).sum((0, 2)).double(); S.mcnt[i] += float(realq.sum())
            if recch: S.out.setdefault('ch', {})[i] = ch.detach().clone()
            d = 0
            if mean is not None:
                sel, M = mean; d = d + ((M[None, :, None, :] - ch) * sel[None, :, None, None]).sum(1)
            if rep is not None:
                sel, D, pos = rep; d = d + ((D - ch) * sel[None, :, None, None] * pos[:, None, :, None]).sum(1)
            if mean is not None or rep is not None: out = out + d.permute(1, 0, 2)
            return out, None
        attn.forward = f

    def means(self):
        Mh = {i: (self.msum[i] / max(self.mcnt[i], 1)).float() for i in range(10)}
        Mf = {i: (self.fsum[i] / max(self.fcnt[i], 1)).float() for i in range(10)}
        return Mh, Mf

    def fwd(self, Jb):
        from interp.part_core import inputs
        t = self.t; x, v, m = inputs(Jb, self.feat)
        with t.no_grad(): L = self.m(*(t.from_numpy(a).to(self.dev) for a in (x, v, m)))
        return L.float().cpu().numpy()

    def run(self, J, ctl=None, bs=BS):
        from interp.part_core import sub
        N = len(J['mask']); order = np.argsort(J['mask'].sum(1), kind='stable'); L = np.zeros((N, 10), np.float32)
        for s in range(0, N, bs):
            idx = order[s:s + bs]; self.ctl = ctl or {}; self.out = {}
            L[idx] = self.fwd(sub(J, idx))
        self.ctl = {}; return L


HEADS = [(i, h) for i in range(10) for h in range(H)]


def hlab(ih):
    i, h = ih; return f'p{i + 1}h{h + 1}' if i < 8 else f'b{i - 7}h{h + 1}'


def ctl_keep(M, keep, Mh, Mf, mlp_ablate):
    """keep: iterable of head indices 0..79 -> ctl with every other head mean-ablated (+ all MLPs if mlp_ablate)"""
    t = M.t; ks = set(int(x) for x in keep); mean = {}
    for i in range(10):
        sel = t.tensor([0.0 if 8 * i + h in ks else 1.0 for h in range(H)], device=M.dev)
        if float(sel.sum()) > 0: mean[i] = (sel, Mh[i])
    c = dict(mean=mean)
    if mlp_ablate: c['mlp'] = {i: Mf[i] for i in range(10)}
    return c


def ctl_ablate(M, heads, Mh):
    t = M.t; mean = {}
    for k in heads:
        i, h = divmod(int(k), H)
        if i not in mean: mean[i] = (t.zeros(H, device=M.dev), Mh[i])
        mean[i][0][h] = 1.0
    return dict(mean=mean)


def patch_runs(M, B, E, mp, sets, bs=200):
    """logits of B (unedited), E (edited), and of B with the heads of each set (list of head-index lists) replaced by their
    outputs in the edited run at aligned tokens -> (L0, L1, {name: L})"""
    from interp.part_core import sub
    t = M.t; N = len(B['mask']); order = np.argsort(B['mask'].sum(1), kind='stable')
    L0 = np.zeros((N, 10), np.float32); L1 = np.zeros((N, 10), np.float32); LP = {k: np.zeros((N, 10), np.float32) for k in sets}
    for s in range(0, N, bs):
        idx = order[s:s + bs]; Bb, Eb = sub(B, idx), sub(E, idx); n = len(idx)
        Pb = int(Bb['mask'].sum(1).max()); Pe = int(Eb['mask'].sum(1).max())
        M.ctl = {}; L0[idx] = M.fwd(Bb)
        M.ctl = dict(recch=set(range(10))); M.out = {}; L1[idx] = M.fwd(Eb); ch = M.out['ch']; M.ctl = {}
        m_ = mp[idx, :Pb]; pos = t.from_numpy(((m_ >= 0) & (Bb['mask'][:, :Pb] > 0)).astype(np.float32)).to(M.dev)
        mc = t.from_numpy(np.clip(m_, 0, Pe - 1).astype(np.int64)).to(M.dev); ar = t.arange(n, device=M.dev)[:, None]
        D = {}
        for i in range(10):
            if i < 8: D[i] = ch[i].permute(0, 2, 1, 3)[ar, mc].permute(0, 2, 1, 3)        # (n, H, Pb, C)
            else: D[i] = ch[i]
        one = t.ones(n, 1, device=M.dev)
        for name, hs in sets.items():
            rep = {}
            for k in hs:
                i, h = divmod(int(k), H)
                if i not in rep: rep[i] = [t.zeros(H, device=M.dev), D[i], pos if i < 8 else one]
                rep[i][0][h] = 1.0
            M.ctl = dict(rep={i: tuple(v) for i, v in rep.items()}); LP[name][idx] = M.fwd(Bb)
        M.ctl = {}; del ch, D; M.out = {}
    return L0, L1, LP


# ------------------------------------------------------------------------------------------------ main
def main():
    import torch
    from interp.part_core import sub, cat, kin, jmass, PRE, CL, sample, build
    OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    R = dict(node='P37', seed=SEED, smoke=SMOKE, files='test_20M <sample>_119.root', NEVAL=NEVAL, NREF=NREF, NSCAN=NSCAN, NSUB=NSUB,
             NB=NB, NPERM=NPERM, NRAND=NRAND, KS=KS, KPATCH=KPATCH, heads=[hlab(x) for x in HEADS])
    # ---------------- samples
    ev, rf, sc = [], [], []
    for c in CL:
        n_sc = NSCAN if c == 'Tbqq' else 0
        J, rows = load_rows(TEST / f'{PRE[c]}_119.root', NEVAL + NREF + n_sc, rng)
        perm = rng.permutation(len(rows))
        ev.append(sub(J, np.sort(perm[:NEVAL]))); rf.append(sub(J, np.sort(perm[NEVAL:NEVAL + NREF])))
        if n_sc: sc.append(sub(J, np.sort(perm[NEVAL + NREF:])))
        log(f'loaded {c}: {len(rows)} random rows of {PRE[c]}_119.root (eval {NEVAL}, ref {NREF}, scan {n_sc})')
    B = rebuild(cat(ev)); BR = rebuild(cat(rf)); BSC = rebuild(cat(sc)); y = B['label'].astype(int); N = len(y)
    assert (BSC['label'].astype(int) == TOP).all()
    sub_idx = np.sort(np.concatenate([rng.choice(np.where(y == k)[0], min(NSUB, (y == k).sum()), replace=False) for k in range(10)]))
    BSUB = sub(B, sub_idx); ysub = y[sub_idx]
    P1 = rebuild(sample(1000, 3000)); yP1 = P1['label'].astype(int)
    R['sample'] = dict(N_eval=N, N_ref=len(BR['label']), N_scan=len(BSC['label']), N_sub=len(sub_idx), N_P1=len(yP1))
    # ---------------- top-jet prongs and edits
    tj = np.where(y == TOP)[0]; BT = sub(B, tj)
    info = prongs(BT, rng)
    va = np.array([g is not None for g in info['gb']]); vb = np.array([g is not None for g in info['gw']])
    vb2 = np.array([g is not None for g in info['gw2']])
    V = np.where(va & vb)[0]; V2 = np.where(va & vb2)[0]
    BV = sub(BT, V)
    Ea, mpa, oka = merge_edit(BV, [info['gb'][j] for j in V]); Eb, mpb, okb = merge_edit(BV, [info['gw'][j] for j in V])
    BV2 = sub(BT, V2); Eb2, mpb2, _ = merge_edit(BV2, [info['gw2'][j] for j in V2]); Ea2, _, _ = merge_edit(BV2, [info['gb'][j] for j in V2])
    good = (info['mW'][V] > 65) & (info['mW'][V] < 95) & (info['mjet'][V] > 140) & (info['mjet'][V] < 205)
    t32a = np.array([tau32_of(p4_of(Ea, j)) for j in range(len(V))]); t32b = np.array([tau32_of(p4_of(Eb, j)) for j in range(len(V))])
    t320 = info['tau32'][V]
    R['edit_checks'] = dict(
        n_top=int(len(tj)), n_valid_a_b=int(len(V)), n_valid_b2=int(len(V2)), frac_lt6=float(np.mean([int(BT['mask'][j].sum()) < 6 for j in range(len(tj))])),
        n_good_window=int(good.sum()), good_def='W-pair mass 65-95 GeV and jet mass 140-205 GeV',
        nb=dict(median=float(np.median(info['nb'][V])), mean=float(info['nb'][V].mean())),
        nw=dict(median=float(np.median(info['nw'][V])), mean=float(info['nw'][V].mean())),
        ptb_frac=dict(median=float(np.median(info['ptb'][V] / BV['jet_pt']))), ptw_over_ptb=dict(median=float(np.median(info['ptw'][V] / info['ptb'][V]))),
        mjet_median=float(np.nanmedian(info['mjet'][V])), mW_median=float(np.nanmedian(info['mW'][V])),
        dmass_a_max=float(np.abs(jmass(Ea) - jmass(BV)).max()), dmass_b_max=float(np.abs(jmass(Eb) - jmass(BV)).max()),
        djetpt_a_max=float(np.abs(Ea['jet_pt'] - BV['jet_pt']).max()),
        nparticles_removed_a=float((BV['mask'].sum(1) - Ea['mask'].sum(1)).mean()), nparticles_removed_b=float((BV['mask'].sum(1) - Eb['mask'].sum(1)).mean()),
        tau32=dict(before=float(np.nanmedian(t320)), a=float(np.nanmedian(t32a)), b=float(np.nanmedian(t32b)),
                   d_a_mean=float(np.nanmean(t32a - t320)), d_b_mean=float(np.nanmean(t32b - t320)),
                   d_a_ci=None, d_b_ci=None))
    Wv = boot_W(len(V), NB, rng)
    for k_, d_ in (('d_a_ci', t32a - t320), ('d_b_ci', t32b - t320)):
        dd = np.nan_to_num(d_); R['edit_checks']['tau32'][k_] = ci(Wv @ dd)
    log('edit checks', R['edit_checks'])
    # example jet for the figure: the first valid jet in the good window with >= 4 b-prong particles
    ex = next((q for q in range(len(V)) if good[q] and info['nb'][V[q]] >= 4), 0); jv = V[ex]; n0 = int(BV['mask'][ex].sum())
    pt0 = np.hypot(BV['px'][ex, :n0], BV['py'][ex, :n0]); na_ = int(Ea['mask'][ex].sum())
    R['example'] = dict(deta=BV['deta'][ex, :n0], dphi=BV['dphi'][ex, :n0], pt=pt0, lab=info['lab'][jv], mjet=info['mjet'][jv], mW=info['mW'][jv],
                        a_deta=Ea['deta'][ex, :na_], a_dphi=Ea['dphi'][ex, :na_], a_pt=np.hypot(Ea['px'][ex, :na_], Ea['py'][ex, :na_]),
                        a_merged_slot=int(np.min(info['gb'][jv])), mjet_a=float(jmass(sub(Ea, [ex]))[0]),
                        tau32_before=float(t320[ex]), tau32_a=float(t32a[ex]))
    SF = np.where(np.random.default_rng(SEED + 1).random((NPERM, NSCAN)) < 0.5, -1.0, 1.0).astype(np.float64)
    Wsc = boot_W(NSCAN, NB, rng); Wall = boot_W(N, NB, rng)
    SFv = np.where(np.random.default_rng(SEED + 2).random((NPERM, len(V))) < 0.5, -1.0, 1.0)
    R['models'] = {}
    for name, feat, untr in (('ParT_full', 'full', False), ('ParT_kin', 'kin', False), ('ParT_full_untrained', 'full', True)):
        log(f'==== {name}'); res = {}; M = HRun(feat, untrained=untr); t1 = time.time()
        res['acc_P1'] = acc_block(M.run(P1), yP1, conf=False)['acc10']
        L = M.run(B); res['eval'] = acc_block(L, y, Wall)
        log(f'  acc P1 {res["acc_P1"]:.4f}  eval acc10 {res["eval"]["acc10"]:.4f} {np.round(res["eval"]["acc10_ci"], 4).tolist()} '
            f'top recall {res["eval"]["top_recall"]:.4f}  bin t/QCD {res["eval"]["bin_tq"]:.4f}  ({time.time() - t1:.0f}s)')
        M.run(BR, dict(recmean=True)); Mh, Mf = M.means(); res['recon_max'] = M.recon
        res['mean_ablate_all_check'] = acc_block(M.run(BSUB, ctl_keep(M, [], Mh, Mf, True)), ysub, conf=False)
        # ---------------- step 1: scan
        t1 = time.time(); L0 = M.run(BSC); a0, b0 = ctr(L0, TOP, QCD), ctr(L0, TOP, WQQ)
        res['scan_unedited'] = dict(top_recall=float((L0.argmax(1) == TOP).mean()), lo_t=float(lodds(L0).mean()))
        dq, dw, lost = np.zeros((80, NSCAN)), np.zeros((80, NSCAN)), np.zeros(80)
        for k in range(80):
            Lk = M.run(BSC, ctl_ablate(M, [k], Mh)); dq[k] = ctr(Lk, TOP, QCD) - a0; dw[k] = ctr(Lk, TOP, WQQ) - b0
            lost[k] = ((L0.argmax(1) == TOP) & (Lk.argmax(1) != TOP)).mean()
        sq = [mstat(dq[k], Wsc, SF) for k in range(80)]; sw = [mstat(dw[k], Wsc, SF) for k in range(80)]
        padj = holm([s['p_perm'] for s in sq] + [s['p_perm'] for s in sw])
        scan = {}
        for k in range(80):
            scan[hlab(HEADS[k])] = dict(d_QCD=sq[k], d_W=sw[k], p_holm_QCD=padj[k], p_holm_W=padj[80 + k], top_calls_lost=float(lost[k]))
        score = np.array([(sq[k]['v'] + sw[k]['v']) / 2 for k in range(80)])
        sig_neg = [k for k in range(80) if (padj[k] < 0.05 and sq[k]['v'] < 0) or (padj[80 + k] < 0.05 and sw[k]['v'] < 0)]
        sig_pos = [k for k in range(80) if (padj[k] < 0.05 and sq[k]['v'] > 0) or (padj[80 + k] < 0.05 and sw[k]['v'] > 0)]
        rank = sorted(sig_neg, key=lambda k: score[k]) + sorted([k for k in range(80) if k not in sig_neg], key=lambda k: score[k])
        absq = np.abs([s['v'] for s in sq]); top_share = np.sort(absq)[::-1]
        res['scan'] = dict(heads=scan, n_tests=160, sig_neg=[hlab(HEADS[k]) for k in sig_neg], sig_pos=[hlab(HEADS[k]) for k in sig_pos],
                           n_sig_neg=len(sig_neg), n_sig_pos=len(sig_pos), rank=[hlab(HEADS[k]) for k in rank], _rank=rank,
                           score=[float(score[k]) for k in range(80)],
                           abs_dQCD_share_top5=float(top_share[:5].sum() / max(top_share.sum(), 1e-12)),
                           abs_dQCD_share_top10=float(top_share[:10].sum() / max(top_share.sum(), 1e-12)),
                           n_heads_for_80pct_abs_dQCD=int(np.searchsorted(np.cumsum(top_share) / max(top_share.sum(), 1e-12), 0.8) + 1))
        log(f'  scan ({time.time() - t1:.0f}s): Holm-significant lowering heads {len(sig_neg)}, raising {len(sig_pos)}; rank top 10 '
            + ', '.join(f'{hlab(HEADS[k])} dQCD {sq[k]["v"]:+.3f} dW {sw[k]["v"]:+.3f}' for k in rank[:10]))
        if untr:
            R['models'][name] = res; FN.write_text(json.dumps(strip(R), indent=1)); del M; torch.cuda.empty_cache(); continue
        # all-heads joint knock-out of the significant set, for reference
        if sig_neg:
            Lj = M.run(BSC, ctl_ablate(M, sig_neg, Mh)); res['scan']['joint_sig_neg'] = dict(dQCD=mstat(ctr(Lj) - a0, Wsc),
                                                                                           top_recall=float((Lj.argmax(1) == TOP).mean()))
        # ---------------- step 2: literal circuits
        t1 = time.time(); circ = {}
        for variant, mlpab in (('literal', True), ('heads_only', False)):
            circ[variant] = {}
            for k in KS:
                keep = rank[:k]; ctl = ctl_keep(M, keep, Mh, Mf, mlpab)
                e = acc_block(M.run(B, ctl), y, Wall); es = acc_block(M.run(BSUB, ctl), ysub, conf=False)
                e['heads'] = [hlab(HEADS[q]) for q in keep]; e['n_holm_sig_in'] = int(sum(q in sig_neg for q in keep)); e['sub'] = es
                if 0 < k < 80:
                    rr = dict(top_recall=[], balanced=[], bin_tq=[], acc10=[])
                    for r_ in range(NRAND):
                        ks_ = rng.choice(80, k, replace=False); er = acc_block(M.run(BSUB, ctl_keep(M, ks_, Mh, Mf, mlpab)), ysub, conf=False)
                        for q in rr: rr[q].append(er[q])
                    e['random'] = {q: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)), p5=float(np.percentile(v, 5)),
                                           max=float(np.max(v)), p_emp=float((1 + sum(x >= es[q] for x in v)) / (1 + NRAND))) for q, v in rr.items()}
                circ[variant][str(k)] = e
                log(f'  {variant} k={k}: top recall {e["top_recall"]:.3f} bin t/QCD {e["bin_tq"]:.3f} balanced {e["balanced"]:.3f} acc10 {e["acc10"]:.3f}'
                    + (f' | random top recall {e["random"]["top_recall"]["mean"]:.3f} (p95 {e["random"]["top_recall"]["p95"]:.3f}) p {e["random"]["top_recall"]["p_emp"]:.3f}; '
                       f'balanced {e["random"]["balanced"]["mean"]:.3f} p {e["random"]["balanced"]["p_emp"]:.3f}' if 'random' in e else '') + f' ({time.time() - t1:.0f}s)')
        res['circuits'] = circ
        FN.write_text(json.dumps(strip(dict(R, models=dict(R['models'], **{name: res}))), indent=1))
        # ---------------- step 3: edits and patching
        t1 = time.time(); sets = {}
        for k in KPATCH:
            sets[f'circ{k}'] = rank[:k]
            for r_ in range(NRAND): sets[f'r{k}_{r_}'] = rng.choice(80, k, replace=False).tolist()
        ed = {}
        for en, E_, mp_ in (('a', Ea, mpa), ('b', Eb, mpb)):
            L0e, L1e, LP = patch_runs(M, BV, E_, mp_, sets)
            dfull = lodds(L1e) - lodds(L0e); dq_full = ctr(L1e) - ctr(L0e)
            e = dict(dlo_t=mstat(dfull, Wv, SFv), dQCD=mstat(dq_full, Wv, SFv), calls_changed=float((L0e.argmax(1) != L1e.argmax(1)).mean()),
                     mcnemar_top=mcnemar(L0e.argmax(1) == TOP, L1e.argmax(1) == TOP), changed_to=np.bincount(L1e.argmax(1)[L0e.argmax(1) != L1e.argmax(1)], minlength=10),
                     dlo_t_good=float(dfull[good].mean()) if good.any() else None, _d=dfull)
            for k in KPATCH:
                dp = lodds(LP[f'circ{k}']) - lodds(L0e); shb = (Wv @ dp) / (Wv @ dfull)
                sh = float(dp.mean() / dfull.mean()); rs = []
                for r_ in range(NRAND):
                    rs.append(float((lodds(LP[f'r{k}_{r_}']) - lodds(L0e)).mean() / dfull.mean()))
                e[f'patch{k}'] = dict(heads=[hlab(HEADS[q]) for q in sets[f'circ{k}']], dpatch=mstat(dp, Wv), share=sh, share_ci=ci(shb),
                                      random_share=dict(mean=float(np.mean(rs)), p95=float(np.percentile(rs, 95)), p5=float(np.percentile(rs, 5)),
                                                        max=float(np.max(rs)), p_emp=float((1 + sum(x >= sh for x in rs)) / (1 + NRAND))))
            ed[en] = e
            log(f'  edit {en}: d lo_t {e["dlo_t"]["v"]:+.3f} {np.round(e["dlo_t"]["ci"], 3).tolist()} dQCD {e["dQCD"]["v"]:+.3f} calls changed {e["calls_changed"]:.3f} '
                f'top {e["mcnemar_top"]["before"]:.3f}->{e["mcnemar_top"]["after"]:.3f} | '
                + '; '.join(f'circ{k} share {e[f"patch{k}"]["share"]:.3f} {np.round(e[f"patch{k}"]["share_ci"], 3).tolist()} vs random {e[f"patch{k}"]["random_share"]["mean"]:.3f} '
                            f'(p95 {e[f"patch{k}"]["random_share"]["p95"]:.3f}) p {e[f"patch{k}"]["random_share"]["p_emp"]:.3f}' for k in KPATCH) + f' ({time.time() - t1:.0f}s)')
        ed['a_vs_b'] = dict(ddlo_t=mstat(ed['a']['_d'] - ed['b']['_d'], Wv, SFv))
        # count-matched control b2 (full network only)
        L0c = M.run(BV2); Lb2 = M.run(Eb2); La2 = M.run(Ea2); d2 = lodds(Lb2) - lodds(L0c); da2 = lodds(La2) - lodds(L0c)
        W2 = boot_W(len(V2), NB, rng); SF2 = np.where(np.random.default_rng(SEED + 3).random((NPERM, len(V2))) < 0.5, -1.0, 1.0)
        ed['b2'] = dict(n=int(len(V2)), dlo_t=mstat(d2, W2, SF2), a_on_same=mstat(da2, W2), ddlo_t_a_minus_b2=mstat(da2 - d2, W2, SF2),
                        calls_changed=float((L0c.argmax(1) != Lb2.argmax(1)).mean()), mcnemar_top=mcnemar(L0c.argmax(1) == TOP, Lb2.argmax(1) == TOP))
        # permutation-invariance check of the edit layout (unsorted merged slot vs part_core.build re-sorted)
        Ls = M.run(build(Ea, *kin(Ea))); Lu = M.run(Ea); ed['layout_check_max_abs_dlogit'] = float(np.abs(Ls - Lu).max())
        ed['layout_check_median_abs_dlogit'] = float(np.median(np.abs(Ls - Lu)))
        log(f'  a - b: {ed["a_vs_b"]["ddlo_t"]["v"]:+.3f} {np.round(ed["a_vs_b"]["ddlo_t"]["ci"], 3).tolist()} p {ed["a_vs_b"]["ddlo_t"]["p_perm"]:.2g}; '
            f'b2 {ed["b2"]["dlo_t"]["v"]:+.3f} (a on same {ed["b2"]["a_on_same"]["v"]:+.3f}); layout check {ed["layout_check_max_abs_dlogit"]:.2e}')
        res['edits'] = ed; res['runtime_s'] = time.time() - T0
        R['models'][name] = res; FN.write_text(json.dumps(strip(R), indent=1)); log(f'  wrote {FN}')
        del M; torch.cuda.empty_cache()
    R['runtime_s'] = time.time() - T0
    FN.write_text(json.dumps(strip(R), indent=1)); log(f'done; wrote {FN}')


# ------------------------------------------------------------------------------------------------ figure
def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads(FN.read_text()); F = R['models']['ParT_full']
    fig = plt.figure(figsize=(16, 9.5)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    # A: example jet before/after
    ex = R['example']; axA = fig.add_subplot(gs[0, 0]); cols = np.array(['#c55a11', '#1f4e79', '#4f81bd'])
    lab = np.asarray(ex['lab']); pt = np.asarray(ex['pt'])
    axA.scatter(ex['deta'], ex['dphi'], s=8 + 400 * pt / pt.max(), c=cols[lab], alpha=0.6, edgecolors='none')
    apt = np.asarray(ex['a_pt']); i0 = ex['a_merged_slot']
    axA.scatter([ex['a_deta'][i0]], [ex['a_dphi'][i0]], s=8 + 400 * apt[i0] / pt.max(), facecolors='none', edgecolors='k', lw=1.5)
    axA.set_xlabel(r'$\Delta\eta$ to jet axis'); axA.set_ylabel(r'$\Delta\phi$ to jet axis'); axA.set_aspect('equal')
    axA.set_title(f'Edit a on a real t->bqq\' jet (m = {ex["mjet"]:.0f} GeV): the b-prong\n(orange, {int((lab == 0).sum())} particles) becomes one '
                  f'particle (black ring); m unchanged ({ex["mjet_a"]:.0f} GeV)', fontsize=9)
    from matplotlib.lines import Line2D
    axA.legend(handles=[Line2D([], [], marker='o', ls='', color='#c55a11', label='b-prong (3rd kT subjet)'),
                        Line2D([], [], marker='o', ls='', color='#1f4e79', label='W prongs (pair mass nearest 80 GeV)'),
                        Line2D([], [], marker='o', ls='', mfc='none', mec='k', label='merged b-prong after edit a')], fontsize=7, loc='lower left')
    # B: scan heatmap
    axB = fig.add_subplot(gs[0, 1]); labs = R['heads']; v = np.array([F['scan']['heads'][h]['d_QCD']['v'] for h in labs]).reshape(10, 8)
    lim = np.abs(v).max(); im = axB.imshow(v, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
    for h in F['scan']['sig_neg']:
        k = labs.index(h); axB.text(k % 8, k // 8, '*', ha='center', va='center', fontsize=9)
    axB.set_yticks(range(10)); axB.set_yticklabels([f'particle block {i + 1}' for i in range(8)] + ['class block 1', 'class block 2'], fontsize=7)
    axB.set_xticks(range(8)); axB.set_xticklabels([f'h{i + 1}' for i in range(8)], fontsize=7)
    plt.colorbar(im, ax=axB, fraction=0.046).set_label(r'mean $\Delta(\mathrm{logit}_t-\mathrm{logit}_{QCD})$', fontsize=8)
    axB.set_title(f'ParT_full: one head mean-ablated at a time, true t->bqq\' jets (N = {R["NSCAN"]})\n'
                  f'* Holm-significant lowering ({F["scan"]["n_sig_neg"]} of 80 heads)', fontsize=9)
    # C: circuit recall vs k
    axC = fig.add_subplot(gs[0, 2]); ks = [k for k in R['KS'] if 0 < k < 80]
    for var, col in (('literal', '#1f4e79'), ('heads_only', '#9dc3e6')):
        cc = F['circuits'][var]; axC.plot(ks, [cc[str(k)]['sub']['top_recall'] for k in ks], 'o-', color=col, label=f'circuit ({"heads only; MLPs ablated" if var == "literal" else "heads; MLPs kept"})')
        axC.fill_between(ks, [cc[str(k)]['random']['top_recall']['p5'] for k in ks], [cc[str(k)]['random']['top_recall']['p95'] for k in ks], color=col, alpha=0.2)
        axC.axhline(cc['80']['top_recall'], color=col, ls=':', lw=1)
    axC.axhline(F['eval']['top_recall'], color='k', lw=1, label='unablated network')
    axC.set_xscale('log'); axC.set_xticks(ks); axC.set_xticklabels(ks); axC.set_xlabel('heads kept (k)'); axC.set_ylabel("t->bqq' recall")
    axC.set_ylim(-0.02, 1.02); axC.legend(fontsize=7); axC.set_title('Heads kept alone vs 100 random sets of k heads\n(band: 5-95 % of random sets; dotted: all 80 heads kept)', fontsize=9)
    # D: edits
    axD = fig.add_subplot(gs[1, 0:2]); groups = []
    for m in ('ParT_full', 'ParT_kin'):
        e = R['models'][m]['edits']
        for en, nm in (('a', 'b-prong merged'), ('b', 'control: pT-matched W-prong cluster merged')):
            groups.append((f'{m}\n{nm}', e[en]))
    x = np.arange(len(groups)); w = 0.25
    full = [g[1]['dlo_t']['v'] for g in groups]; err = np.array([[g[1]['dlo_t']['v'] - g[1]['dlo_t']['ci'][0], g[1]['dlo_t']['ci'][1] - g[1]['dlo_t']['v']] for g in groups]).T
    axD.bar(x - w, full, w, yerr=err, color='#7f7f7f', label='full network change')
    for j, (k, col) in enumerate(zip(R['KPATCH'], ('#1f4e79', '#4f81bd'))):
        axD.bar(x + j * w, [g[1][f'patch{k}']['dpatch']['v'] for g in groups], w, color=col, label=f'only the top-{k} heads see the edit')
        axD.plot(x + j * w, [g[1][f'patch{k}']['random_share']['mean'] * g[1]['dlo_t']['v'] for g in groups], '_', ms=18, mew=2, color='#c55a11',
                 label='mean of 100 random head sets' if j == 0 else None)
    axD.set_xticks(x); axD.set_xticklabels([g[0] for g in groups], fontsize=8); axD.axhline(0, color='k', lw=0.5)
    axD.set_ylabel(r"mean $\Delta\,\log[p_t/(1-p_t)]$"); axD.legend(fontsize=7)
    axD.set_title(f"Merging the b-prong into one particle at fixed jet four-momentum (N = {R['edit_checks']['n_valid_a_b']} true t->bqq' jets)", fontsize=9)
    # E: literal balanced accuracy
    axE = fig.add_subplot(gs[1, 2]); cc = F['circuits']['literal']
    axE.plot(ks, [cc[str(k)]['sub']['balanced'] for k in ks], 'o-', color='#1f4e79', label='literal circuit, 10-class balanced accuracy')
    axE.fill_between(ks, [cc[str(k)]['random']['balanced']['p5'] for k in ks], [cc[str(k)]['random']['balanced']['p95'] for k in ks], color='#1f4e79', alpha=0.2)
    axE.plot(ks, [cc[str(k)]['sub']['bin_tq'] for k in ks], 's-', color='#c55a11', label='literal circuit, t vs QCD accuracy')
    axE.fill_between(ks, [cc[str(k)]['random']['bin_tq']['p5'] for k in ks], [cc[str(k)]['random']['bin_tq']['p95'] for k in ks], color='#c55a11', alpha=0.2)
    axE.axhline(0.1, color='k', ls=':', lw=1); axE.axhline(0.5, color='#c55a11', ls=':', lw=1)
    axE.set_xscale('log'); axE.set_xticks(ks); axE.set_xticklabels(ks); axE.set_xlabel('heads kept (k)'); axE.set_ylim(0, 1); axE.legend(fontsize=7)
    axE.set_title('ParT_full literal circuit (all MLPs mean-ablated)\nband: 5-95 % of random head sets', fontsize=9)
    fig.tight_layout(); p = OUTD / 'p37_top_circuit.png'; fig.savefig(p, dpi=120); print('wrote', p)


if __name__ == '__main__':
    plot() if ARGS.plot else main()
